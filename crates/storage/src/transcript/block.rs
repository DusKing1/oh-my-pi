//! Neutral assistant blocks and provider replay residue.

use std::collections::BTreeMap;

use omp_core::Str;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use super::raweq::map_raw_eq;
pub use super::types::{CallId, DialectId};
use crate::blob::BlobRef;

/// A neutral assistant block with optional provider-native replay residue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Block {
	/// Provider-neutral content and structure.
	pub kind: BlockKind,
	/// Sparse provider-native residue needed for exact replay.
	pub re:   Option<Replay>,
}

/// The provider-neutral projection of an assistant output item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum BlockKind {
	/// Assistant-visible text.
	Text {
		/// Text exactly as accumulated from the stream.
		text: Str,
	},
	/// Model reasoning text.
	Think {
		/// Reasoning text exactly as accumulated from the stream.
		text: Str,
	},
	/// A tool invocation.
	Tool {
		/// Bare call identifier paired with a tool result.
		id:   CallId,
		/// Harness-visible tool name.
		name: Str,
		/// Custom-tool wire name, when distinct from the harness name.
		wire: Option<Str>,
		/// Wire argument string, stored verbatim without parse/serialize drift.
		args: Str,
	},
	/// An image emitted by the assistant.
	Image {
		/// Content-addressed image payload.
		blob: BlobRef,
	},
	/// A native item with no neutral projection.
	Opaque,
}

/// A sparse field-level diff over a dialect's deterministic defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Replay {
	/// Provider replay dialect.
	pub p: DialectId,
	/// Verbatim replacement fields and reserved replay markers.
	pub f: BTreeMap<Str, Box<RawValue>>,
}
/// Equality is byte equality of stored JSON text, preserving verbatim round
/// trips.
impl PartialEq for Replay {
	fn eq(&self, other: &Self) -> bool {
		self.p == other.p && map_raw_eq(&self.f, &other.f)
	}
}

impl Eq for Replay {}
