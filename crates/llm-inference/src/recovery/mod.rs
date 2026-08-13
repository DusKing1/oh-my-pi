//! Incremental, deterministic, bounded recovery stages.

use std::fmt;

use bytes::Bytes;
use omp_core::Str;

/// Incremental sans-I/O transform used by every recovery component.
///
/// Implementations retain only incomplete input between calls and must resolve
/// that input deterministically from [`Stage::finish`].
pub trait Stage<I, O> {
	/// Consumes one input fragment and synchronously emits zero or more outputs.
	fn push(&mut self, input: I, emit: &mut dyn FnMut(O)) -> Result<(), RecoveryError>;

	/// Resolves held suffixes and emits terminal recovery output.
	fn finish(&mut self, emit: &mut dyn FnMut(O)) -> Result<(), RecoveryError>;
}

/// Secret-safe bounded context retained for a recovery failure.
///
/// `Debug` deliberately reports only lengths. Callers must explicitly request
/// the preview bytes, and receipts never store them.
#[derive(Clone, Eq, PartialEq)]
pub struct DiagnosticContext {
	preview:     Bytes,
	input_bytes: usize,
	truncated:   bool,
}

impl DiagnosticContext {
	/// Captures at most `limit` bytes from the beginning and end of `input`.
	#[must_use]
	pub fn capture(input: &[u8], limit: usize) -> Self {
		if input.len() <= limit {
			return Self {
				preview:     Bytes::copy_from_slice(input),
				input_bytes: input.len(),
				truncated:   false,
			};
		}
		let head = limit.div_ceil(2);
		let tail = limit.saturating_sub(head);
		let mut preview = Vec::with_capacity(limit);
		preview.extend_from_slice(&input[..head]);
		preview.extend_from_slice(&input[input.len() - tail..]);
		Self { preview: Bytes::from(preview), input_bytes: input.len(), truncated: true }
	}

	/// Borrows the explicitly bounded byte preview.
	#[must_use]
	pub fn preview(&self) -> &[u8] {
		&self.preview
	}

	/// Returns the complete input length without retaining the complete input.
	#[must_use]
	pub const fn input_bytes(&self) -> usize {
		self.input_bytes
	}

	/// Returns whether bytes were omitted between the retained ends.
	#[must_use]
	pub const fn is_truncated(&self) -> bool {
		self.truncated
	}
}

impl fmt::Debug for DiagnosticContext {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("DiagnosticContext")
			.field("preview_bytes", &self.preview.len())
			.field("input_bytes", &self.input_bytes)
			.field("truncated", &self.truncated)
			.finish()
	}
}

/// Typed failure from an incremental recovery stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecoveryError {
	/// A deterministic resource bound was exceeded.
	LimitExceeded { stage: &'static str, limit: usize },
	/// Input was complete but invalid for the stage contract.
	InvalidInput { stage: &'static str, reason: Str },
	/// Invalid input with an explicitly bounded byte diagnostic.
	InvalidDocument {
		/// Recovery stage which rejected the document.
		stage:      &'static str,
		/// Secret-safe structural reason.
		reason:     Str,
		/// Bounded context whose `Debug` output hides its bytes.
		diagnostic: DiagnosticContext,
	},
	/// End of input arrived while a required construct remained incomplete.
	Incomplete { stage: &'static str },
	/// Recovery was available but forbidden by strict enforcement.
	RepairRejected { stage: &'static str, diagnostic: DiagnosticContext },
}

impl fmt::Display for RecoveryError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::LimitExceeded { stage, limit } => {
				write!(formatter, "{stage} recovery limit exceeded ({limit})")
			},
			Self::InvalidInput { stage, reason } => {
				write!(formatter, "invalid {stage} recovery input: {reason}")
			},
			Self::InvalidDocument { stage, reason, .. } => {
				write!(formatter, "invalid {stage} recovery document: {reason}")
			},
			Self::Incomplete { stage } => write!(formatter, "incomplete {stage} recovery input"),
			Self::RepairRejected { stage, .. } => {
				write!(formatter, "{stage} repair rejected by strict enforcement")
			},
		}
	}
}

impl std::error::Error for RecoveryError {}

pub mod dialect;
pub mod empty;
pub mod json;
pub mod scanner;
pub mod thinking;

pub mod projection;
pub mod reasoning;
pub mod repetition;
pub mod tools;
