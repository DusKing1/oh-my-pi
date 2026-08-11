//! Opt-in reconstruction of delta-only chat output.
//!
//! The previous implementation attached a complete accumulated `partial`
//! message to every delta. For an $n$-byte answer that repeatedly cloned the
//! growing prefix, producing an $O(n^2)$ allocation treadmill. Wire events in
//! this crate remain delta-only; consumers that actually need a running view
//! pay for [`StreamAccumulator`] explicitly.

use std::collections::BTreeMap;

use bytes::{Bytes, BytesMut};
use omp_core::Str;

use crate::{Message, Part, Role, StreamPartKind, Thinking, ToolCall, TurnEvent, ids::CallId};

/// A malformed delta sequence that cannot be projected into canonical output.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum AccumulatorError {
	/// A part index was started more than once.
	#[error("part {0} started more than once")]
	DuplicateStart(u32),
	/// A delta arrived for an index that has not started.
	#[error("delta for unknown part {0}")]
	UnknownPart(u32),
	/// A delta or end marker arrived after the named part was already closed.
	#[error("part {0} is already closed")]
	PartAlreadyEnded(u32),
	/// Text or thinking bytes were not valid UTF-8 at snapshot time.
	#[error("part {0} is not valid UTF-8")]
	InvalidUtf8(u32),
	/// A tool-call start carried no valid canonical ULID.
	#[error("tool-call part {0} has no valid call id")]
	InvalidCallId(u32),
	/// A tool-call start omitted its dispatch name.
	#[error("tool-call part {0} has no tool name")]
	MissingToolName(u32),
}

/// Opt-in reconstruction of interleaved chat part deltas.
///
/// Parts are keyed by their explicit stream index rather than arrival order,
/// so providers may interleave text, thinking, and multiple tool calls. Tool
/// argument fragments are appended to [`BytesMut`] verbatim and frozen only
/// when read; they are never parsed and reserialized. [`TurnEvent::PartEnd`]
/// finalizes an existing open part and supplies the opaque signature retained
/// on thinking output.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
	parts: BTreeMap<u32, PendingPart>,
}

#[derive(Debug)]
struct PendingPart {
	body:  PendingBody,
	ended: bool,
}

#[derive(Debug)]
enum PendingBody {
	Text(BytesMut),
	Thinking { text: BytesMut, signature: Bytes },
	ToolCall { id: CallId, name: Str, args: BytesMut },
}

impl StreamAccumulator {
	/// Creates an empty assistant-output accumulator.
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	/// Folds one event into the running output.
	///
	/// Non-part events are deliberately ignored; the terminal outcome remains
	/// the authoritative committed record.
	#[inline]
	pub fn push(&mut self, event: &TurnEvent) -> Result<(), AccumulatorError> {
		match event {
			TurnEvent::PartStart { index, kind, tool_call_id, tool_name } => {
				self.start(*index, *kind, tool_call_id, tool_name.clone())
			},
			TurnEvent::PartDelta { index, chunk } => self.delta(*index, chunk),
			TurnEvent::PartEnd { index, signature } => self.end(*index, signature),
			_ => Ok(()),
		}
	}

	/// Builds the current assistant message in ascending part-index order.
	///
	/// This is the explicit snapshot allocation that delta events themselves
	/// avoid. Tool calls are returned separately by [`Self::tool_calls`] because
	/// canonical transcript tool calls are items, not message content parts.
	pub fn message(&self) -> Result<Message, AccumulatorError> {
		let mut parts = Vec::with_capacity(self.parts.len());
		for (&index, part) in &self.parts {
			match &part.body {
				PendingBody::Text(bytes) => {
					let text =
						std::str::from_utf8(bytes).map_err(|_| AccumulatorError::InvalidUtf8(index))?;
					parts.push(Part::Text(Str::from(text)));
				},
				PendingBody::Thinking { text: bytes, signature } => {
					let text =
						std::str::from_utf8(bytes).map_err(|_| AccumulatorError::InvalidUtf8(index))?;
					parts.push(Part::Thinking(Thinking {
						text:      Str::from(text),
						signature: signature.clone(),
						redacted:  false,
					}));
				},
				PendingBody::ToolCall { .. } => {},
			}
		}
		Ok(Message { role: Role::Assistant, parts })
	}

	/// Snapshots all running and finalized tool calls in part-index order.
	#[must_use]
	pub fn tool_calls(&self) -> Vec<ToolCall> {
		self
			.parts
			.values()
			.filter_map(|part| match &part.body {
				PendingBody::ToolCall { id, name, args } => Some(ToolCall {
					id:                *id,
					name:              name.clone(),
					args_json:         args.clone().freeze(),
					thought_signature: Bytes::new(),
					intent:            None,
					raw:               None,
					custom_wire_name:  None,
					provider_metadata: None,
				}),
				PendingBody::Text(_) | PendingBody::Thinking { .. } => None,
			})
			.collect()
	}

	/// Snapshots finalized tool calls in part-index order.
	///
	/// This excludes starts that never received a matching
	/// [`TurnEvent::PartEnd`], so a malformed invocation cannot be committed to
	/// an authoritative turn outcome.
	#[must_use]
	pub fn completed_tool_calls(&self) -> Vec<ToolCall> {
		self
			.parts
			.values()
			.filter(|part| part.ended)
			.filter_map(|part| match &part.body {
				PendingBody::ToolCall { id, name, args } => Some(ToolCall {
					id:                *id,
					name:              name.clone(),
					args_json:         args.clone().freeze(),
					thought_signature: Bytes::new(),
					intent:            None,
					raw:               None,
					custom_wire_name:  None,
					provider_metadata: None,
				}),
				PendingBody::Text(_) | PendingBody::Thinking { .. } => None,
			})
			.collect()
	}

	/// Reports whether the named part has received its end marker.
	#[must_use]
	pub fn is_finalized(&self, index: u32) -> bool {
		self.parts.get(&index).is_some_and(|part| part.ended)
	}

	fn start(
		&mut self,
		index: u32,
		kind: StreamPartKind,
		tool_call_id: &str,
		tool_name: Str,
	) -> Result<(), AccumulatorError> {
		if self.parts.contains_key(&index) {
			return Err(AccumulatorError::DuplicateStart(index));
		}
		let body = match kind {
			StreamPartKind::Text => PendingBody::Text(BytesMut::new()),
			StreamPartKind::Thinking => {
				PendingBody::Thinking { text: BytesMut::new(), signature: Bytes::new() }
			},
			StreamPartKind::ToolCall => {
				let id: CallId = tool_call_id
					.parse()
					.map_err(|_| AccumulatorError::InvalidCallId(index))?;
				if id.as_ulid().to_bytes() == [0; 16] {
					return Err(AccumulatorError::InvalidCallId(index));
				}
				PendingBody::ToolCall {
					id,
					name: if tool_name.is_empty() {
						return Err(AccumulatorError::MissingToolName(index));
					} else {
						tool_name
					},
					args: BytesMut::new(),
				}
			},
		};
		self.parts.insert(index, PendingPart { body, ended: false });
		Ok(())
	}

	#[inline]
	fn delta(&mut self, index: u32, chunk: &Bytes) -> Result<(), AccumulatorError> {
		let Some(part) = self.parts.get_mut(&index) else {
			return Err(AccumulatorError::UnknownPart(index));
		};
		match &mut part.body {
			PendingBody::Text(bytes) | PendingBody::Thinking { text: bytes, .. } => {
				if part.ended {
					return Err(AccumulatorError::PartAlreadyEnded(index));
				}
				bytes.extend_from_slice(chunk);
			},
			PendingBody::ToolCall { args, .. } => {
				if part.ended {
					return Err(AccumulatorError::PartAlreadyEnded(index));
				}
				args.extend_from_slice(chunk);
			},
		}
		Ok(())
	}

	fn end(&mut self, index: u32, signature: &Bytes) -> Result<(), AccumulatorError> {
		let Some(part) = self.parts.get_mut(&index) else {
			return Err(AccumulatorError::UnknownPart(index));
		};
		if part.ended {
			return Err(AccumulatorError::PartAlreadyEnded(index));
		}
		if let PendingBody::Thinking { signature: stored, .. } = &mut part.body {
			stored.clone_from(signature);
		}
		part.ended = true;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use bytes::Bytes;
	use omp_core::Str;

	use super::{AccumulatorError, StreamAccumulator};
	use crate::{Part, StreamPartKind, TurnEvent, ids::CallId};

	fn start(index: u32, kind: StreamPartKind) -> TurnEvent {
		TurnEvent::PartStart {
			index,
			kind,
			tool_call_id: Str::from(""),
			tool_name: Str::from(""),
		}
	}

	fn delta(index: u32, chunk: &'static [u8]) -> TurnEvent {
		TurnEvent::PartDelta { index, chunk: Bytes::from_static(chunk) }
	}

	#[test]
	fn text_thinking_and_tool_parts_interleave_by_index() {
		let call_id = CallId::new();
		let mut accumulator = StreamAccumulator::new();
		let events = [
			start(1, StreamPartKind::Thinking),
			start(0, StreamPartKind::Text),
			TurnEvent::PartStart {
				index:        2,
				kind:         StreamPartKind::ToolCall,
				tool_call_id: Str::from(call_id.to_string()),
				tool_name:    Str::from("lookup"),
			},
			delta(0, b"hello "),
			delta(1, b"rea"),
			delta(1, b"son"),
			delta(2, br#"{"q":"#),
			delta(0, b"world"),
			delta(2, br#"rust"}"#),
			TurnEvent::PartEnd { index: 1, signature: Bytes::from_static(b"signed-thinking") },
		];
		for event in &events {
			accumulator.push(event).unwrap();
		}

		let message = accumulator.message().unwrap();
		assert!(matches!(&message.parts[0], Part::Text(text) if text == "hello world"));
		assert!(matches!(
			&message.parts[1],
			Part::Thinking(value)
				if value.text == "reason"
					&& value.signature == Bytes::from_static(b"signed-thinking")
		));
		assert_eq!(accumulator.tool_calls()[0].id, call_id);
		assert!(accumulator.completed_tool_calls().is_empty());
	}

	#[test]
	fn tool_arguments_remain_byte_identical() {
		let mut accumulator = StreamAccumulator::new();
		let call_id = CallId::new();
		accumulator
			.push(&TurnEvent::PartStart {
				index:        0,
				kind:         StreamPartKind::ToolCall,
				tool_call_id: Str::from(call_id.to_string()),
				tool_name:    Str::from("raw"),
			})
			.unwrap();
		for chunk in [b"{ \"n\" : 1".as_slice(), b".00, \"x\":\"\\u0061\" }".as_slice()] {
			accumulator
				.push(&TurnEvent::PartDelta { index: 0, chunk: Bytes::copy_from_slice(chunk) })
				.unwrap();
		}
		accumulator
			.push(&TurnEvent::PartEnd { index: 0, signature: Bytes::new() })
			.unwrap();
		assert_eq!(
			accumulator.tool_calls()[0].args_json,
			Bytes::from_static(b"{ \"n\" : 1.00, \"x\":\"\\u0061\" }"),
		);
		assert_eq!(accumulator.completed_tool_calls().len(), 1);
	}

	#[test]
	fn malformed_part_ordering_fails() {
		let mut accumulator = StreamAccumulator::new();
		assert_eq!(
			accumulator.push(&TurnEvent::PartEnd { index: 4, signature: Bytes::new() }),
			Err(AccumulatorError::UnknownPart(4)),
		);
		accumulator.push(&start(4, StreamPartKind::Text)).unwrap();
		accumulator
			.push(&TurnEvent::PartEnd { index: 4, signature: Bytes::new() })
			.unwrap();
		assert_eq!(accumulator.push(&delta(4, b"late")), Err(AccumulatorError::PartAlreadyEnded(4)),);
	}

	#[test]
	fn zero_tool_call_id_is_not_canonical() {
		let mut accumulator = StreamAccumulator::new();
		let event = TurnEvent::PartStart {
			index:        0,
			kind:         StreamPartKind::ToolCall,
			tool_call_id: Str::new_static("00000000000000000000000000"),
			tool_name:    Str::new_static("lookup"),
		};
		assert_eq!(accumulator.push(&event), Err(AccumulatorError::InvalidCallId(0)));
	}
}
