//! Terminal mode support for builtins that read character-by-character.

use crate::{error, openfiles, sys};

/// High-level terminal settings requested by input builtins.
#[derive(Default, bon::Builder)]
pub(crate) struct Settings {
	/// Whether input is echoed.
	pub(crate) echo_input:        Option<bool>,
	/// Whether input is line-buffered.
	pub(crate) line_input:        Option<bool>,
	/// Whether control characters generate interrupt signals.
	pub(crate) interrupt_signals: Option<bool>,
	/// Whether newlines are emitted as CRLF pairs.
	pub(crate) output_nl_as_nlcr: Option<bool>,
}

/// Restores a terminal's original mode when dropped.
pub(crate) struct AutoModeGuard {
	initial: sys::terminal::Config,
	file:    openfiles::OpenFile,
}

impl AutoModeGuard {
	/// Captures the current mode for `file`.
	pub(crate) fn new(file: openfiles::OpenFile) -> Result<Self, error::Error> {
		let initial = sys::terminal::Config::from_term(&file)?;
		Ok(Self { initial, file })
	}

	/// Applies settings until this guard is dropped.
	pub(crate) fn apply_settings(&self, settings: &Settings) -> Result<(), error::Error> {
		let mut config = sys::terminal::Config::from_term(&self.file)?;
		config.update(settings);
		config.apply_to_term(&self.file)?;
		Ok(())
	}
}

impl Drop for AutoModeGuard {
	fn drop(&mut self) {
		let _ = self.initial.apply_to_term(&self.file);
	}
}
