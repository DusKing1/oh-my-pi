//! Stable identifiers for the terminal attached to standard input.

use std::{ffi::OsString, path::Path};

use omp_core::{SmolStr, format_smol};

#[cfg(unix)]
mod platform {
	use std::path::PathBuf;

	pub(super) fn tty_path() -> Option<PathBuf> {
		crate::tty::override_path().or_else(|| nix::unistd::ttyname(std::io::stdin()).ok())
	}
}

#[cfg(windows)]
mod platform {
	use windows_sys::Win32::{
		Foundation::INVALID_HANDLE_VALUE,
		System::Console::{GetConsoleMode, GetStdHandle, STD_INPUT_HANDLE},
	};

	pub(super) fn has_console() -> bool {
		let handle = unsafe { GetStdHandle(STD_INPUT_HANDLE) };
		if handle.is_null() || handle == INVALID_HANDLE_VALUE {
			return false;
		}
		let mut mode = 0;
		unsafe { GetConsoleMode(handle, &mut mode) != 0 }
	}
}

/// Stable identifier used when neither a TTY nor a terminal environment
/// variable is available.
pub const UNKNOWN_TERMINAL_ID: &str = "unknown";

/// Resolve a stable identifier for the terminal attached to standard input.
///
/// The TTY device name takes priority. When standard input is not a TTY, common
/// multiplexer and terminal-emulator environment variables are consulted from
/// the innermost to the outermost terminal.
#[must_use]
pub fn terminal_id() -> SmolStr {
	#[cfg(unix)]
	{
		let tty_path = platform::tty_path();
		terminal_id_with(tty_path.as_deref(), |name| std::env::var_os(name))
	}
	#[cfg(windows)]
	{
		let id = terminal_id_with(None, |name| std::env::var_os(name));
		if id != UNKNOWN_TERMINAL_ID || !platform::has_console() {
			return id;
		}
		"console".into()
	}
}

/// Resolve a terminal identifier from an injected TTY path and environment.
///
/// This is the deterministic core of [`terminal_id`]. The environment callback
/// should return the value for the requested variable, or `None` when unset.
#[must_use]
pub fn terminal_id_with(
	tty_path: Option<&Path>,
	mut env: impl FnMut(&str) -> Option<OsString>,
) -> SmolStr {
	if let Some(id) = tty_path.and_then(normalize_tty_path) {
		return id;
	}

	if let Some(pane) = nonempty_env(&mut env, "ZELLIJ_PANE_ID") {
		if let Some(session) = nonempty_env(&mut env, "ZELLIJ_SESSION_NAME") {
			let session = session.replace(['/', '\\'], "-");
			return format_smol!("zellij-{session}-{pane}");
		}
		return format_smol!("zellij-{pane}");
	}

	for (name, prefix) in [
		("TMUX_PANE", "tmux"),
		("CMUX_SURFACE_ID", "cmux"),
		("KITTY_WINDOW_ID", "kitty"),
		("WEZTERM_PANE", "wezterm"),
		("TERM_SESSION_ID", "apple"),
		("WT_SESSION", "wt"),
	] {
		if let Some(value) = nonempty_env(&mut env, name) {
			return format_smol!("{prefix}-{value}");
		}
	}

	UNKNOWN_TERMINAL_ID.into()
}

fn normalize_tty_path(path: &Path) -> Option<SmolStr> {
	let path = path.to_str()?;
	let relative = path.strip_prefix("/dev/")?;
	if relative.is_empty() {
		return None;
	}
	Some(SmolStr::from(relative.replace('/', "-")))
}

fn nonempty_env(env: &mut impl FnMut(&str) -> Option<OsString>, name: &str) -> Option<String> {
	env(name)?
		.into_string()
		.ok()
		.filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
	use std::{collections::HashMap, ffi::OsString, path::Path};

	use super::{UNKNOWN_TERMINAL_ID, terminal_id_with};

	#[test]
	fn normalizes_posix_tty_paths() {
		let no_env = |_: &str| None;
		assert_eq!(terminal_id_with(Some(Path::new("/dev/pts/3")), no_env), "pts-3");
		assert_eq!(terminal_id_with(Some(Path::new("/dev/ttys004")), no_env), "ttys004");
	}

	#[test]
	fn tty_path_precedes_environment() {
		assert_eq!(
			terminal_id_with(Some(Path::new("/dev/pts/8")), |_| Some("ignored".into())),
			"pts-8"
		);
	}

	#[test]
	fn environment_uses_exact_precedence() {
		let variables = HashMap::from([
			("ZELLIJ_PANE_ID", "1"),
			("ZELLIJ_SESSION_NAME", "work/tree\\leaf"),
			("TMUX_PANE", "%2"),
			("CMUX_SURFACE_ID", "3"),
			("KITTY_WINDOW_ID", "4"),
			("WEZTERM_PANE", "5"),
			("TERM_SESSION_ID", "6"),
			("WT_SESSION", "7"),
		]);
		let env = |name: &str| variables.get(name).map(OsString::from);
		assert_eq!(terminal_id_with(None, env), "zellij-work-tree-leaf-1");

		let ordered = [
			("TMUX_PANE", "tmux-%2"),
			("CMUX_SURFACE_ID", "cmux-3"),
			("KITTY_WINDOW_ID", "kitty-4"),
			("WEZTERM_PANE", "wezterm-5"),
			("TERM_SESSION_ID", "apple-6"),
			("WT_SESSION", "wt-7"),
		];
		for (index, &(winner, expected)) in ordered.iter().enumerate() {
			let env = |name: &str| {
				ordered[index..]
					.iter()
					.find(|&&(candidate, _)| candidate == name)
					.map(|_| OsString::from(variables[name]))
			};
			assert_eq!(terminal_id_with(None, env), expected, "winner: {winner}");
		}
	}

	#[test]
	fn empty_and_non_unicode_values_are_ignored() {
		let env = |name: &str| match name {
			"TMUX_PANE" => Some(OsString::new()),
			"KITTY_WINDOW_ID" => Some("9".into()),
			_ => None,
		};
		assert_eq!(terminal_id_with(None, env), "kitty-9");
		assert_eq!(terminal_id_with(None, |_| None), UNKNOWN_TERMINAL_ID);
	}
}
