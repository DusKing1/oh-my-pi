//! Deterministic within-attempt and cross-turn repetition guards.

use std::collections::VecDeque;

use omp_core::Str;
use serde_json::Value;

use super::{RecoveryError, Stage};
use crate::{
	call::OpaqueJson,
	id::ToolCallId,
	receipt::{ReasonId, RecoveryKind, RecoveryRecord},
};

/// Whether provisional output is still hidden from the consumer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputVisibility {
	/// Output is owned by a semantic gate and may be discarded safely.
	Gated,
	/// Ordinary output has reached the consumer.
	Committed,
}

/// Retry consequence of a detected loop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopDisposition {
	/// The gate may discard the attempt and semantic policy may retry it.
	RetryEligible,
	/// The committed partial response must surface as an error without retry.
	SurfaceCommitted,
}

impl From<OutputVisibility> for LoopDisposition {
	fn from(value: OutputVisibility) -> Self {
		match value {
			OutputVisibility::Gated => Self::RetryEligible,
			OutputVisibility::Committed => Self::SurfaceCommitted,
		}
	}
}

/// Stable loop category consumed by semantic and session layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopKind {
	/// The same output unit repeated within one provider attempt.
	WithinAttempt,
	/// Equivalent tool call/result observations recurred across committed turns.
	CrossTurnTool,
	/// Reasoning continued without semantic progress.
	ReasoningStall,
}

/// Bounded evidence accompanying a loop decision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoopEvidence {
	/// Stable loop category.
	pub kind:        LoopKind,
	/// Stable non-secret fingerprint of the repeated unit.
	pub fingerprint: u64,
	/// Consecutive observations that established the loop.
	pub repetitions: u32,
	/// Total bytes examined for this decision.
	pub input_bytes: u64,
}

/// A loop detection and its required retry behavior.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoopSignal {
	/// Bounded deterministic evidence.
	pub evidence:    LoopEvidence,
	/// Whether retry remains legal.
	pub disposition: LoopDisposition,
}

/// Configuration for within-attempt repetition detection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepetitionLimits {
	/// Consecutive equivalent units required to declare a loop.
	pub consecutive_limit: u32,
	/// Maximum normalized bytes retained for exact collision checking.
	pub max_unit_bytes:    usize,
	/// Maximum history entries retained.
	pub history_limit:     usize,
}

impl Default for RepetitionLimits {
	fn default() -> Self {
		Self { consecutive_limit: 4, max_unit_bytes: 16 * 1024, history_limit: 32 }
	}
}

#[derive(Clone, Debug)]
struct Unit {
	normalized:  Str,
	fingerprint: u64,
}

/// Detects repeated text, reasoning, or tool signatures during one attempt.
#[derive(Debug)]
pub struct AttemptRepetitionGuard {
	limits:      RepetitionLimits,
	history:     VecDeque<Unit>,
	consecutive: u32,
	input_bytes: u64,
}

impl AttemptRepetitionGuard {
	/// Creates a bounded guard.
	pub fn new(limits: RepetitionLimits) -> Self {
		Self { limits, history: VecDeque::new(), consecutive: 0, input_bytes: 0 }
	}

	/// Observes one semantic output unit.
	pub fn observe(&mut self, unit: &str, visibility: OutputVisibility) -> Option<LoopSignal> {
		self.input_bytes = self.input_bytes.saturating_add(unit.len() as u64);
		let normalized = normalize_unit(unit, self.limits.max_unit_bytes)?;
		let fingerprint = stable_hash(normalized.as_bytes());
		let repeated = self.history.back().is_some_and(|previous| {
			previous.fingerprint == fingerprint && previous.normalized.as_str() == normalized
		});
		self.consecutive = if repeated {
			self.consecutive.saturating_add(1)
		} else {
			1
		};
		self
			.history
			.push_back(Unit { normalized: Str::from(normalized), fingerprint });
		while self.history.len() > self.limits.history_limit {
			self.history.pop_front();
		}
		let cycle = repeated_cycle(&self.history, self.limits.consecutive_limit);
		let (repetitions, evidence_fingerprint) = cycle.unwrap_or((self.consecutive, fingerprint));
		(repetitions >= self.limits.consecutive_limit).then(|| LoopSignal {
			evidence:    LoopEvidence {
				kind: LoopKind::WithinAttempt,
				fingerprint: evidence_fingerprint,
				repetitions,
				input_bytes: self.input_bytes,
			},
			disposition: visibility.into(),
		})
	}

	/// Clears state before a new provider attempt.
	pub fn reset(&mut self) {
		self.history.clear();
		self.consecutive = 0;
		self.input_bytes = 0;
	}
}

impl<'a> Stage<(&'a str, OutputVisibility), LoopSignal> for AttemptRepetitionGuard {
	fn push(
		&mut self,
		(unit, visibility): (&'a str, OutputVisibility),
		emit: &mut dyn FnMut(LoopSignal),
	) -> Result<(), RecoveryError> {
		if let Some(signal) = self.observe(unit, visibility) {
			emit(signal);
		}
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(LoopSignal)) -> Result<(), RecoveryError> {
		Ok(())
	}
}

/// One authorized tool call plus its real caller-supplied result.
#[derive(Clone, Debug)]
pub struct ToolExchangeObservation {
	/// Authorized call identity; excluded from semantic equivalence.
	pub call_id:   ToolCallId,
	/// Declared tool name.
	pub name:      Str,
	/// Schema-valid call arguments.
	pub arguments: OpaqueJson,
	/// Caller-supplied tool result.
	pub result:    OpaqueJson,
	/// Whether the executor reported an error.
	pub is_error:  bool,
}

/// Cross-turn input recorded by session and consumed by semantic recovery.
#[derive(Clone, Debug, Default)]
pub struct TurnRecoveryObservation {
	/// Ordered authorized call/result exchanges committed in the turn.
	pub tool_exchanges:        Vec<ToolExchangeObservation>,
	/// Whether visible assistant output made progress besides repeated calls.
	pub made_textual_progress: bool,
}

/// Bounded cross-turn loop configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CrossTurnLimits {
	/// Equivalent consecutive turns required to stop the loop.
	pub consecutive_limit:   u32,
	/// Number of committed turn fingerprints retained.
	pub history_limit:       usize,
	/// Maximum canonical structural bytes retained per turn.
	pub max_canonical_bytes: usize,
}

impl Default for CrossTurnLimits {
	fn default() -> Self {
		Self { consecutive_limit: 3, history_limit: 16, max_canonical_bytes: 256 * 1024 }
	}
}

/// Detects repeated call/result cycles across committed session turns.
#[derive(Debug)]
pub struct CrossTurnLoopGuard {
	limits:  CrossTurnLimits,
	history: VecDeque<u64>,
	last:    Option<(u64, Vec<u8>, u32)>,
}

impl CrossTurnLoopGuard {
	/// Creates a cross-turn guard.
	pub fn new(limits: CrossTurnLimits) -> Self {
		Self { limits, history: VecDeque::new(), last: None }
	}

	/// Consumes one committed turn observation.
	pub fn observe(&mut self, observation: &TurnRecoveryObservation) -> Option<LoopSignal> {
		if observation.made_textual_progress || observation.tool_exchanges.is_empty() {
			self.last = None;
			return None;
		}
		let (fingerprint, canonical) =
			fingerprint_turn(observation, self.limits.max_canonical_bytes)?;
		let repetitions = match self.last.as_ref() {
			Some((previous, previous_canonical, count))
				if *previous == fingerprint && previous_canonical == &canonical =>
			{
				count.saturating_add(1)
			},
			_ => 1,
		};
		let input_bytes = canonical.len() as u64;
		self.last = Some((fingerprint, canonical, repetitions));
		self.history.push_back(fingerprint);
		while self.history.len() > self.limits.history_limit {
			self.history.pop_front();
		}
		(repetitions >= self.limits.consecutive_limit).then(|| LoopSignal {
			evidence:    LoopEvidence {
				kind: LoopKind::CrossTurnTool,
				fingerprint,
				repetitions,
				input_bytes,
			},
			disposition: LoopDisposition::SurfaceCommitted,
		})
	}

	/// Returns retained fingerprints for append-only session persistence.
	pub fn fingerprints(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
		self.history.iter().copied()
	}
}

impl Stage<TurnRecoveryObservation, LoopSignal> for CrossTurnLoopGuard {
	fn push(
		&mut self,
		observation: TurnRecoveryObservation,
		emit: &mut dyn FnMut(LoopSignal),
	) -> Result<(), RecoveryError> {
		if let Some(signal) = self.observe(&observation) {
			emit(signal);
		}
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(LoopSignal)) -> Result<(), RecoveryError> {
		Ok(())
	}
}

/// Converts a loop signal to bounded receipt evidence.
pub fn recovery_record(attempt: u32, signal: &LoopSignal) -> RecoveryRecord {
	let (kind, rule) = match signal.evidence.kind {
		LoopKind::WithinAttempt => (RecoveryKind::WithinAttemptRepetition, "loop.within-attempt"),
		LoopKind::CrossTurnTool => (RecoveryKind::CrossTurnToolLoop, "loop.cross-turn-tool"),
		LoopKind::ReasoningStall => (RecoveryKind::ReasoningStall, "loop.reasoning-stall"),
	};
	RecoveryRecord {
		attempt,
		kind,
		rule: ReasonId(Str::from(rule)),
		input_bytes: signal.evidence.input_bytes,
		steps: signal.evidence.repetitions,
	}
}

fn fingerprint_turn(observation: &TurnRecoveryObservation, limit: usize) -> Option<(u64, Vec<u8>)> {
	let mut encoded = Vec::with_capacity(limit.min(4096));
	for exchange in &observation.tool_exchanges {
		push_field(exchange.name.as_bytes(), &mut encoded, limit)?;
		write_canonical_value(exchange.arguments.as_value(), &mut encoded, limit)?;
		write_canonical_value(exchange.result.as_value(), &mut encoded, limit)?;
		push_bounded(&[u8::from(exchange.is_error)], &mut encoded, limit)?;
	}
	Some((stable_hash(&encoded), encoded))
}

fn write_canonical_value(value: &Value, output: &mut Vec<u8>, limit: usize) -> Option<()> {
	match value {
		Value::Null => push_bounded(b"n", output, limit),
		Value::Bool(value) => push_bounded(if *value { b"t" } else { b"f" }, output, limit),
		Value::Number(value) => {
			push_bounded(b"d", output, limit)?;
			push_field(value.to_string().as_bytes(), output, limit)
		},
		Value::String(value) => {
			push_bounded(b"s", output, limit)?;
			push_field(value.as_bytes(), output, limit)
		},
		Value::Array(values) => {
			push_bounded(b"a", output, limit)?;
			push_bounded(&(values.len() as u64).to_le_bytes(), output, limit)?;
			if output.len().saturating_add(values.len()) > limit {
				return None;
			}
			for value in values {
				write_canonical_value(value, output, limit)?;
			}
			Some(())
		},
		Value::Object(values) => {
			push_bounded(b"o", output, limit)?;
			push_bounded(&(values.len() as u64).to_le_bytes(), output, limit)?;
			let minimum_key_bytes = values.len().checked_mul(8)?;
			if output.len().saturating_add(minimum_key_bytes) > limit {
				return None;
			}
			let mut keys: Vec<_> = values.keys().collect();
			keys.sort_unstable();
			for key in keys {
				push_field(key.as_bytes(), output, limit)?;
				write_canonical_value(&values[key], output, limit)?;
			}
			Some(())
		},
	}
}

fn push_field(bytes: &[u8], output: &mut Vec<u8>, limit: usize) -> Option<()> {
	push_bounded(&(bytes.len() as u64).to_le_bytes(), output, limit)?;
	push_bounded(bytes, output, limit)
}

fn push_bounded(bytes: &[u8], output: &mut Vec<u8>, limit: usize) -> Option<()> {
	(output.len().saturating_add(bytes.len()) <= limit).then(|| output.extend_from_slice(bytes))
}
fn repeated_cycle(history: &VecDeque<Unit>, threshold: u32) -> Option<(u32, u64)> {
	let length = history.len();
	for period in 1..=length / 2 {
		let mut repetitions = 1_u32;
		while (repetitions as usize + 1) * period <= length {
			let right_start = length - repetitions as usize * period;
			let left_start = right_start - period;
			let same = (0..period).all(|offset| {
				let left = history.get(left_start + offset).expect("index is bounded");
				let right = history.get(right_start + offset).expect("index is bounded");
				left.fingerprint == right.fingerprint && left.normalized == right.normalized
			});
			if !same {
				break;
			}
			repetitions += 1;
		}
		if repetitions >= threshold {
			let mut fingerprint = 0xcbf29ce484222325_u64;
			for offset in 0..period {
				let unit = history
					.get(length - period + offset)
					.expect("index is bounded");
				fingerprint ^= unit.fingerprint;
				fingerprint = fingerprint.wrapping_mul(0x100000001b3);
			}
			return Some((repetitions, fingerprint));
		}
	}
	None
}

fn normalize_unit(unit: &str, limit: usize) -> Option<String> {
	let mut normalized = String::with_capacity(unit.len().min(limit));
	for word in unit.split_ascii_whitespace() {
		if !normalized.is_empty() {
			normalized.push(' ');
		}
		normalized.push_str(word);
		if normalized.len() > limit {
			return None;
		}
	}
	(!normalized.is_empty()).then_some(normalized)
}

pub(crate) fn stable_hash(bytes: &[u8]) -> u64 {
	let mut hash = 0xcbf29ce484222325_u64;
	for byte in bytes {
		hash ^= u64::from(*byte);
		hash = hash.wrapping_mul(0x100000001b3);
	}
	hash
}

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::*;

	#[test]
	fn committed_loops_surface_and_gated_loops_may_retry() {
		let limits = RepetitionLimits { consecutive_limit: 2, ..RepetitionLimits::default() };
		let mut guard = AttemptRepetitionGuard::new(limits);
		assert!(guard.observe("same", OutputVisibility::Gated).is_none());
		assert_eq!(
			guard
				.observe(" same ", OutputVisibility::Gated)
				.unwrap()
				.disposition,
			LoopDisposition::RetryEligible
		);
		guard.reset();
		guard.observe("same", OutputVisibility::Committed);
		assert_eq!(
			guard
				.observe("same", OutputVisibility::Committed)
				.unwrap()
				.disposition,
			LoopDisposition::SurfaceCommitted
		);
	}

	#[test]
	fn repeated_multi_unit_cycle_is_detected() {
		let limits = RepetitionLimits { consecutive_limit: 3, ..RepetitionLimits::default() };
		let mut guard = AttemptRepetitionGuard::new(limits);
		for unit in ["alpha", "beta", "alpha", "beta", "alpha"] {
			assert!(guard.observe(unit, OutputVisibility::Gated).is_none());
		}
		assert!(guard.observe("beta", OutputVisibility::Gated).is_some());
	}

	#[test]
	fn semantic_json_order_does_not_hide_cross_turn_loop() {
		let limits = CrossTurnLimits { consecutive_limit: 2, ..CrossTurnLimits::default() };
		let make = |arguments| TurnRecoveryObservation {
			tool_exchanges:        vec![ToolExchangeObservation {
				call_id:   ToolCallId::new("ignored"),
				name:      Str::from("search"),
				arguments: OpaqueJson::new(arguments),
				result:    OpaqueJson::new(json!({"ok":true})),
				is_error:  false,
			}],
			made_textual_progress: false,
		};
		let mut guard = CrossTurnLoopGuard::new(limits);
		assert!(guard.observe(&make(json!({"a":1,"b":2}))).is_none());
		assert_eq!(
			guard
				.observe(&make(json!({"b":2,"a":1})))
				.unwrap()
				.disposition,
			LoopDisposition::SurfaceCommitted
		);
	}
}
