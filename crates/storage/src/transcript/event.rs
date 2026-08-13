//! In-memory event model for transcript v4 journals.

use std::path::PathBuf;

use omp_core::Str;
use omp_proto::thread::v1::Item;
use serde_json::value::RawValue;

use super::{
	msg::{Content, Msg},
	patch::Patch,
	raweq::{opt_raw_eq, raw_eq},
	types::{
		AmendPatch, CallId, ModelChange, ModelId, ModelRef, Pin, ProviderId, RequestError, SessionId,
		ThinkingSel, Tier, TitleSource, Usage,
	},
};
use crate::blob::BlobRef;

/// One canonical thread item with journal-only turn metadata.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ItemRecord {
	/// Canonical gateway thread item.
	pub item:        Item,
	/// Turn that committed the item, absent for optimistic local input.
	pub turn_id:     Option<Str>,
	/// Deterministic system-prompt hash active when the item was recorded.
	pub prompt_hash: Option<[u8; 32]>,
}

impl Eq for ItemRecord {}

/// Durable proof that one gateway turn outcome was fully journaled.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TurnReceipt {
	/// Gateway turn identifier.
	pub turn_id:     Str,
	/// Physical event indexes of canonical items committed by the outcome.
	pub item_events: Vec<u64>,
}

/// A timestamped transcript event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
	/// Epoch-millisecond timestamp.
	pub ts:   u64,
	/// Event payload.
	pub kind: Kind,
}

/// An append-only transcript event kind.
#[derive(Debug, Clone)]
pub enum Kind {
	/// Initialize a session's prompt, tools, and spawn options.
	Init {
		/// Content-addressed system prompt.
		system_prompt: BlobRef,
		/// Tool names available to the session.
		tools:         Vec<Str>,
		/// Optional spawning agent identifier.
		agent:         Option<Str>,
		/// Optional response schema, preserved verbatim.
		output_schema: Option<Box<RawValue>>,
	},
	/// Add a conversation message.
	Msg(Msg),
	/// Add one canonical gateway thread item.
	Item(ItemRecord),
	/// Record an inference failure with no conversational content.
	Failed {
		/// Request failure details.
		error: RequestError,
		/// Model selected for the failed request.
		model: ModelRef,
		/// Usage reported despite the failure, when available.
		usage: Option<Usage>,
	},
	/// Change one or more inference selections.
	Infer {
		/// Thinking-mode update.
		thinking: Patch<ThinkingSel>,
		/// Model-selection update.
		model:    Patch<ModelChange>,
		/// Service-tier update.
		tier:     Patch<Tier>,
		/// Credential-pin update.
		cred_pin: Patch<Pin>,
	},
	/// Move the implicit chain point to an earlier event or the root.
	Rewind {
		/// Target event index, or `None` for the root.
		to: Option<u64>,
	},
	/// Replace an old context prefix with a neutral summary.
	Compact {
		/// Full summary used for model context.
		summary:       Str,
		/// Optional shorter display summary.
		short:         Option<Str>,
		/// First pre-compaction event retained after the summary.
		first_kept:    u64,
		/// Token count before compaction.
		tokens_before: u64,
		/// Optional compaction warning.
		warning:       Option<Str>,
	},
	/// Summarize a branch before returning to another chain point.
	Branch {
		/// Event index from which the summarized branch began.
		from:    u64,
		/// Branch summary.
		summary: Str,
	},
	/// Start a fresh chain boundary, as for `/clear`.
	Reset,
	/// Assign a session title.
	Title {
		/// New title.
		title:  Str,
		/// Source that assigned the title.
		source: TitleSource,
	},
	/// Add working directories available to the session.
	AddDirs {
		/// Directories added by this event.
		dirs: Vec<PathBuf>,
	},
	/// Record lineage from a source session.
	ForkedFrom {
		/// Source session identifier.
		session: SessionId,
		/// Source event index, or the source session head when absent.
		at:      Option<u64>,
	},
	/// Replace accumulated provider-native history with checkpoint items.
	NativeCheckpoint {
		/// Provider whose replay stream the checkpoint replaces.
		provider: ProviderId,
		/// Model whose replay stream the checkpoint replaces.
		model:    ModelId,
		/// Content-addressed checkpoint item payload.
		items:    BlobRef,
	},
	/// Record tool calls aborted by an interrupted turn.
	Aborted {
		/// Bare call identifiers aborted by this event.
		tool_call_ids: Vec<CallId>,
	},
	/// Correct an earlier event without editing it.
	Amend {
		/// Event index receiving the correction.
		target: u64,
		/// Append-only correction.
		patch:  AmendPatch,
	},
	/// Record completion of a gateway turn after all of its items were appended.
	TurnReceipt(TurnReceipt),
	/// Add, replace, or clear a label on an earlier event.
	Label {
		/// Event index receiving the label.
		target: u64,
		/// New label, or `None` to clear it.
		label:  Option<Str>,
	},
	/// Store an extension event.
	Custom {
		/// Extension-defined kind name.
		kind:    Str,
		/// Verbatim extension data.
		data:    Option<Box<RawValue>>,
		/// Optional content participating in model context.
		context: Option<Content>,
		/// Whether clients should display the event.
		display: bool,
	},
	/// Preserve an unrecognized or foreign journal object verbatim.
	Unknown(Box<RawValue>),
}
/// Equality is byte equality of stored JSON text, preserving verbatim round
/// trips.
impl PartialEq for Kind {
	fn eq(&self, other: &Self) -> bool {
		match (self, other) {
			(
				Self::Init {
					system_prompt: a_system_prompt,
					tools: a_tools,
					agent: a_agent,
					output_schema: a_output_schema,
				},
				Self::Init {
					system_prompt: b_system_prompt,
					tools: b_tools,
					agent: b_agent,
					output_schema: b_output_schema,
				},
			) => {
				a_system_prompt == b_system_prompt
					&& a_tools == b_tools
					&& a_agent == b_agent
					&& opt_raw_eq(a_output_schema.as_deref(), b_output_schema.as_deref())
			},
			(Self::Msg(a_message), Self::Msg(b_message)) => a_message == b_message,
			(Self::Item(a), Self::Item(b)) => a == b,
			(
				Self::Failed { error: a_error, model: a_model, usage: a_usage },
				Self::Failed { error: b_error, model: b_model, usage: b_usage },
			) => (a_error, a_model, a_usage) == (b_error, b_model, b_usage),
			(
				Self::Infer {
					thinking: a_thinking,
					model: a_model,
					tier: a_tier,
					cred_pin: a_cred_pin,
				},
				Self::Infer {
					thinking: b_thinking,
					model: b_model,
					tier: b_tier,
					cred_pin: b_cred_pin,
				},
			) => (a_thinking, a_model, a_tier, a_cred_pin) == (b_thinking, b_model, b_tier, b_cred_pin),
			(Self::Rewind { to: a }, Self::Rewind { to: b }) => a == b,
			(
				Self::Compact {
					summary: a_summary,
					short: a_short,
					first_kept: a_first_kept,
					tokens_before: a_tokens_before,
					warning: a_warning,
				},
				Self::Compact {
					summary: b_summary,
					short: b_short,
					first_kept: b_first_kept,
					tokens_before: b_tokens_before,
					warning: b_warning,
				},
			) => {
				(a_summary, a_short, a_first_kept, a_tokens_before, a_warning)
					== (b_summary, b_short, b_first_kept, b_tokens_before, b_warning)
			},
			(
				Self::Branch { from: a_from, summary: a_summary },
				Self::Branch { from: b_from, summary: b_summary },
			) => (a_from, a_summary) == (b_from, b_summary),
			(Self::Reset, Self::Reset) => true,
			(
				Self::Title { title: a_title, source: a_source },
				Self::Title { title: b_title, source: b_source },
			) => (a_title, a_source) == (b_title, b_source),
			(Self::AddDirs { dirs: a }, Self::AddDirs { dirs: b }) => a == b,
			(
				Self::ForkedFrom { session: a_session, at: a_at },
				Self::ForkedFrom { session: b_session, at: b_at },
			) => (a_session, a_at) == (b_session, b_at),
			(
				Self::NativeCheckpoint { provider: a_provider, model: a_model, items: a_items },
				Self::NativeCheckpoint { provider: b_provider, model: b_model, items: b_items },
			) => (a_provider, a_model, a_items) == (b_provider, b_model, b_items),
			(Self::Aborted { tool_call_ids: a }, Self::Aborted { tool_call_ids: b }) => a == b,
			(
				Self::Amend { target: a_target, patch: a_patch },
				Self::Amend { target: b_target, patch: b_patch },
			) => (a_target, a_patch) == (b_target, b_patch),
			(Self::TurnReceipt(a), Self::TurnReceipt(b)) => a == b,
			(
				Self::Label { target: a_target, label: a_label },
				Self::Label { target: b_target, label: b_label },
			) => (a_target, a_label) == (b_target, b_label),
			(
				Self::Custom { kind: a_kind, data: a_data, context: a_context, display: a_display },
				Self::Custom { kind: b_kind, data: b_data, context: b_context, display: b_display },
			) => {
				a_kind == b_kind
					&& opt_raw_eq(a_data.as_deref(), b_data.as_deref())
					&& a_context == b_context
					&& a_display == b_display
			},
			(Self::Unknown(a), Self::Unknown(b)) => raw_eq(a, b),
			_ => false,
		}
	}
}

impl Eq for Kind {}
