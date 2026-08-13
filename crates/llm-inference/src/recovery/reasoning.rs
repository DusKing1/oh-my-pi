//! Incremental reasoning-progress and stall detection.

use omp_core::Str;

use super::{
	RecoveryError, Stage,
	repetition::{
		LoopDisposition, LoopEvidence, LoopKind, LoopSignal, OutputVisibility, stable_hash,
	},
};

/// Bounds for reasoning stall detection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReasoningLimits {
	/// Consecutive equivalent deltas that declare a direct repetition loop.
	pub repeated_delta_limit: u32,
	/// Non-empty deltas without external semantic progress that declare a stall.
	pub no_progress_limit:    u32,
	/// Maximum retained normalized bytes per delta.
	pub max_delta_bytes:      usize,
}

impl Default for ReasoningLimits {
	fn default() -> Self {
		Self { repeated_delta_limit: 4, no_progress_limit: 12, max_delta_bytes: 16 * 1024 }
	}
}

/// Incremental input to the reasoning guard.
#[derive(Clone, Debug)]
pub struct ReasoningObservation<'a> {
	/// Reasoning text received in this increment.
	pub delta:             &'a str,
	/// An external semantic transition, such as producing answer text or a valid
	/// tool call.
	pub semantic_progress: bool,
	/// Current output visibility at the recovery boundary.
	pub visibility:        OutputVisibility,
}

/// Bounded state machine detecting repeated or zero-progress reasoning.
#[derive(Debug)]
pub struct ReasoningStallGuard {
	limits:      ReasoningLimits,
	last:        Option<(u64, Str)>,
	repeated:    u32,
	no_progress: u32,
	input_bytes: u64,
}

impl ReasoningStallGuard {
	/// Creates a reasoning guard with fixed memory bounds.
	pub fn new(limits: ReasoningLimits) -> Self {
		Self { limits, last: None, repeated: 0, no_progress: 0, input_bytes: 0 }
	}

	/// Observes one delta and emits at most one stable loop decision.
	pub fn observe(&mut self, observation: ReasoningObservation<'_>) -> Option<LoopSignal> {
		self.input_bytes = self
			.input_bytes
			.saturating_add(observation.delta.len() as u64);
		if observation.semantic_progress {
			self.no_progress = 0;
			self.repeated = 0;
			self.last = None;
			return None;
		}
		let normalized = normalize_reasoning(observation.delta, self.limits.max_delta_bytes)?;
		let fingerprint = stable_hash(normalized.as_bytes());
		let exact_repeat = self.last.as_ref().is_some_and(|(previous_hash, previous)| {
			*previous_hash == fingerprint && previous.as_str() == normalized
		});
		self.repeated = if exact_repeat {
			self.repeated.saturating_add(1)
		} else {
			1
		};
		let unit = Str::from(normalized);
		self.no_progress = self.no_progress.saturating_add(1);
		self.last = Some((fingerprint, unit));
		let repetitions = if self.repeated >= self.limits.repeated_delta_limit {
			self.repeated
		} else if self.no_progress >= self.limits.no_progress_limit {
			self.no_progress
		} else {
			return None;
		};
		Some(LoopSignal {
			evidence:    LoopEvidence {
				kind: LoopKind::ReasoningStall,
				fingerprint,
				repetitions,
				input_bytes: self.input_bytes,
			},
			disposition: LoopDisposition::from(observation.visibility),
		})
	}

	/// Clears attempt-local state while retaining configuration.
	pub fn reset(&mut self) {
		self.last = None;
		self.repeated = 0;
		self.no_progress = 0;
		self.input_bytes = 0;
	}
}

fn normalize_reasoning(input: &str, limit: usize) -> Option<String> {
	let mut output = String::with_capacity(input.len().min(limit));
	for word in input.split_ascii_whitespace() {
		if !output.is_empty() {
			output.push(' ');
		}
		output.push_str(word);
		if output.len() > limit {
			return None;
		}
	}
	(!output.is_empty()).then_some(output)
}
impl<'a> Stage<ReasoningObservation<'a>, LoopSignal> for ReasoningStallGuard {
	fn push(
		&mut self,
		input: ReasoningObservation<'a>,
		emit: &mut dyn FnMut(LoopSignal),
	) -> Result<(), RecoveryError> {
		if let Some(signal) = self.observe(input) {
			emit(signal);
		}
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(LoopSignal)) -> Result<(), RecoveryError> {
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn reasoning_stall_obeys_commit_boundary() {
		let limits = ReasoningLimits { repeated_delta_limit: 2, ..ReasoningLimits::default() };
		let mut guard = ReasoningStallGuard::new(limits);
		assert!(
			guard
				.observe(ReasoningObservation {
					delta:             "I should inspect",
					semantic_progress: false,
					visibility:        OutputVisibility::Gated,
				})
				.is_none()
		);
		let gated = guard
			.observe(ReasoningObservation {
				delta:             "I should inspect",
				semantic_progress: false,
				visibility:        OutputVisibility::Gated,
			})
			.unwrap();
		assert_eq!(gated.disposition, LoopDisposition::RetryEligible);
		guard.reset();
		guard.observe(ReasoningObservation {
			delta:             "again",
			semantic_progress: false,
			visibility:        OutputVisibility::Committed,
		});
		let committed = guard
			.observe(ReasoningObservation {
				delta:             "again",
				semantic_progress: false,
				visibility:        OutputVisibility::Committed,
			})
			.unwrap();
		assert_eq!(committed.disposition, LoopDisposition::SurfaceCommitted);
	}

	#[test]
	fn explicit_semantic_progress_breaks_the_stall() {
		let limits = ReasoningLimits { repeated_delta_limit: 2, ..ReasoningLimits::default() };
		let mut guard = ReasoningStallGuard::new(limits);
		guard.observe(ReasoningObservation {
			delta:             "same",
			semantic_progress: false,
			visibility:        OutputVisibility::Gated,
		});
		assert!(
			guard
				.observe(ReasoningObservation {
					delta:             "same",
					semantic_progress: true,
					visibility:        OutputVisibility::Gated,
				})
				.is_none()
		);
		assert!(
			guard
				.observe(ReasoningObservation {
					delta:             "same",
					semantic_progress: false,
					visibility:        OutputVisibility::Gated,
				})
				.is_none()
		);
	}
}
