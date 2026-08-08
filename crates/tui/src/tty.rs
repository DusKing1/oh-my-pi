//! Terminal device resolution honoring the `OMP_TTY` override.
//!
//! The UI talks to the controlling terminal directly (`/dev/tty` on Unix), so
//! a harness that only owns the process's pipes cannot observe or drive it.
//! Setting `OMP_TTY` to an alternate terminal device — typically a pty slave
//! whose master the harness holds — reroutes every terminal open (input,
//! output via [`TtyOut`], capability probes) and the terminal identity to
//! that device, delivering the complete byte stream a real terminal would
//! see on the master side.
//!
//! Limitations under an override: `SIGWINCH` is only delivered for the
//! controlling terminal, so live resizes will not propagate unless the
//! terminal supports in-band resize; set the window size up front with
//! `TIOCSWINSZ` on the master.

use std::{
	fs::{File, OpenOptions},
	io::{self, IoSlice, Write},
	path::PathBuf,
	sync::LazyLock,
};

/// Environment variable naming an alternate terminal device.
pub const TTY_OVERRIDE: &str = "OMP_TTY";

/// The overriding device path, when [`TTY_OVERRIDE`] is set and non-empty.
pub fn override_path() -> Option<PathBuf> {
	std::env::var_os(TTY_OVERRIDE)
		.filter(|value| !value.is_empty())
		.map(PathBuf::from)
}

/// Whether an override device is configured. Cached: the environment is
/// read once, matching the process-lifetime scope of the override.
pub fn overridden() -> bool {
	static OVERRIDDEN: LazyLock<bool> = LazyLock::new(|| override_path().is_some());
	*OVERRIDDEN
}

/// Opens the terminal device with the given options, honoring [`TTY_OVERRIDE`].
#[cfg(unix)]
pub fn open(options: &OpenOptions) -> io::Result<File> {
	match override_path() {
		Some(path) => options.open(path),
		None => options.open("/dev/tty"),
	}
}

/// Terminal output sink: stdout normally, the `OMP_TTY` device when set.
///
/// [`crate::App`] and full-screen frontends render through this so that an
/// `OMP_TTY` override captures rendered frames alongside the control
/// sequences, not just the lifecycle bytes.
pub struct TtyOut(Sink);

enum Sink {
	Stdout(io::Stdout),
	Device(File),
}

impl TtyOut {
	/// Opens the terminal output sink.
	///
	/// # Errors
	/// Fails when `OMP_TTY` names a path that cannot be opened for writing;
	/// a misconfigured override is reported rather than silently ignored.
	pub fn new() -> io::Result<Self> {
		match override_path() {
			Some(path) => Ok(Self(Sink::Device(OpenOptions::new().write(true).open(path)?))),
			None => Ok(Self(Sink::Stdout(io::stdout()))),
		}
	}
}

impl Write for TtyOut {
	fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
		match &mut self.0 {
			Sink::Stdout(out) => out.write(buf),
			Sink::Device(out) => out.write(buf),
		}
	}

	fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
		match &mut self.0 {
			Sink::Stdout(out) => out.write_vectored(bufs),
			Sink::Device(out) => out.write_vectored(bufs),
		}
	}

	fn flush(&mut self) -> io::Result<()> {
		match &mut self.0 {
			Sink::Stdout(out) => out.flush(),
			Sink::Device(out) => out.flush(),
		}
	}
}
