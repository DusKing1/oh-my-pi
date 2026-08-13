//! Canonical generative chat events and execution-authorization semantics.

use bytes::Bytes;
use omp_core::Str;

use crate::{
	answer::{Artifact, ResponseMeta},
	call::OpaqueJson,
	id::ToolCallId,
	receipt::{ExecutionReceipt, Usage},
};

/// Canonical content-block category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockKind {
	/// User-visible text.
	Text,
	/// Model reasoning or a reasoning summary.
	Thinking,
	/// A model-requested tool invocation.
	ToolCall,
	/// Generated media or other artifact.
	Artifact,
}

/// Fully assembled and schema-validated tool invocation.
#[derive(Clone, Debug)]
pub struct ToolCall {
	/// Stable call identity.
	pub id:        ToolCallId,
	/// Declared tool name.
	pub name:      Str,
	/// Opaque validated JSON arguments.
	pub arguments: OpaqueJson,
}

/// Incremental usage observation within a response stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UsageUpdate {
	/// Cumulative usage observed through this event.
	pub usage:        Usage,
	/// Whether no later usage correction is expected for this attempt.
	pub final_update: bool,
}

/// Why a chat attempt completed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FinishReason {
	/// The model reached a natural stop.
	Stop,
	/// The configured output-token limit was reached.
	Length,
	/// The response completed after emitting tool calls.
	ToolCalls,
	/// A content or safety filter stopped output.
	ContentFilter,
	/// The caller cancelled generation.
	Cancelled,
	/// A provider-specific reason was normalized but remains named.
	Other(Str),
}

/// Final chat stream completion metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Completion {
	/// Normalized finish reason.
	pub reason:  FinishReason,
	/// Number of canonical content blocks emitted.
	pub blocks:  u32,
	/// Final attempt usage.
	pub usage:   Usage,
	/// Authoritative final accounting after every attempt, recovery, adjustment,
	/// and telemetry merge.
	pub receipt: ExecutionReceipt,
}

/// Canonical chat stream vocabulary.
///
/// There is deliberately no restart or rollback event. Once ordinary output is
/// visible, later failures surface as stream errors.
#[derive(Debug)]
pub enum ChatEvent {
	/// Response handshake metadata.
	Started(ResponseMeta),
	/// A canonical content block began.
	BlockStarted {
		/// Stable content-block index.
		index: u32,
		/// Canonical block category.
		kind:  BlockKind,
	},
	/// User-visible text delta.
	TextDelta {
		/// Stable content-block index.
		index: u32,
		/// Incremental visible text.
		text:  Str,
	},
	/// Reasoning or reasoning-summary delta.
	ThinkingDelta {
		/// Stable content-block index.
		index: u32,
		/// Incremental reasoning text.
		text:  Str,
	},
	/// A tool call began, before its arguments are complete.
	ToolCallStarted {
		/// Stable content-block index.
		index: u32,
		/// Stable tool-call identity.
		id:    ToolCallId,
		/// Tool name.
		name:  Str,
	},
	/// Incomplete tool argument bytes for display or telemetry only.
	ToolArgumentsDelta {
		/// Stable content-block index.
		index: u32,
		/// Incremental unvalidated argument bytes.
		bytes: Bytes,
	},
	/// Fully assembled, validated tool call; the sole execution authorization.
	ToolCallReady {
		/// Stable content-block index.
		index: u32,
		/// Validated executable tool call.
		call:  ToolCall,
	},
	/// Generated canonical artifact.
	Artifact {
		/// Stable content-block index.
		index:    u32,
		/// Generated artifact.
		artifact: Artifact,
	},
	/// Incremental usage observation.
	Usage(UsageUpdate),
	/// Successful terminal completion.
	Completed(Completion),
}

impl ChatEvent {
	/// Returns an executable tool call only for `ToolCallReady`.
	pub const fn authorized_tool_call(&self) -> Option<&ToolCall> {
		match self {
			Self::ToolCallReady { call, .. } => Some(call),
			_ => None,
		}
	}

	/// Returns whether this event is ordinary output that commits the stream.
	pub const fn commits_output(&self) -> bool {
		matches!(
			self,
			Self::BlockStarted { .. }
				| Self::TextDelta { .. }
				| Self::ThinkingDelta { .. }
				| Self::ToolCallStarted { .. }
				| Self::ToolArgumentsDelta { .. }
				| Self::ToolCallReady { .. }
				| Self::Artifact { .. }
				| Self::Completed(_)
		)
	}
}

#[cfg(test)]
mod tests {
	use bytes::Bytes;
	use omp_core::Str;
	use serde_json::json;

	use super::{ChatEvent, ToolCall};
	use crate::{call::OpaqueJson, id::ToolCallId};

	#[test]
	fn only_ready_tool_calls_authorize_execution() {
		let started = ChatEvent::ToolCallStarted {
			index: 0,
			id:    ToolCallId::from("call"),
			name:  Str::from("lookup"),
		};
		let partial =
			ChatEvent::ToolArgumentsDelta { index: 0, bytes: Bytes::from_static(b"{\"q\":") };
		assert!(started.authorized_tool_call().is_none());
		assert!(partial.authorized_tool_call().is_none());
		let ready = ChatEvent::ToolCallReady {
			index: 0,
			call:  ToolCall {
				id:        ToolCallId::from("call"),
				name:      Str::from("lookup"),
				arguments: OpaqueJson::new(json!({"q": "rust"})),
			},
		};
		assert_eq!(ready.authorized_tool_call().map(|call| call.name.as_str()), Some("lookup"));
	}
}
