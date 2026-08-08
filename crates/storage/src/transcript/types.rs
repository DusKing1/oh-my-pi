//! Leaf value types used by transcript events.

use omp_core::SmolStr;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::value::RawValue;

use super::{block::Block, raweq::opt_raw_eq};

macro_rules! string_id {
	($(#[$meta:meta])* $name:ident, $doc:literal) => {
		$(#[$meta])*
		#[doc = $doc]
		#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
		#[serde(transparent)]
		pub struct $name(
			/// The identifier text.
			pub SmolStr,
		);
	};
}

string_id!(SessionId, "A stable transcript session identifier.");
string_id!(CallId, "A bare provider tool-call identifier.");
string_id!(DialectId, "A replay-capsule dialect identifier such as `oai` or `ant`.");
string_id!(FeatureId, "A feature identifier reported as unavailable for a turn.");
string_id!(ProviderId, "A model-provider identifier.");
string_id!(ModelId, "A provider model identifier.");

/// The fully qualified model selected for an inference request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
	/// Provider serving the request.
	pub provider: ProviderId,
	/// Provider API family used for the request.
	pub api:      SmolStr,
	/// Provider model name.
	pub model:    ModelId,
}

/// Token usage reported for an inference request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
	/// Non-cached input tokens.
	pub input:       u64,
	/// Generated output tokens.
	pub output:      u64,
	/// Input tokens served from a provider cache.
	pub cache_read:  u64,
	/// Input tokens written into a provider cache.
	pub cache_write: u64,
}

/// Why an assistant turn stopped, with reason-specific provider details.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum Stop {
	/// The model completed its turn normally.
	EndTurn,
	/// Generation reached its configured token limit.
	MaxTokens,
	/// The model stopped to request one or more tools.
	ToolUse,
	/// The model refused the request.
	Refusal {
		/// Verbatim provider details, when supplied.
		details: Option<Box<RawValue>>,
	},
	/// The turn was aborted after producing partial content.
	Aborted {
		/// Verbatim provider details, when supplied.
		details: Option<Box<RawValue>>,
	},
	/// Provider content filtering stopped the turn.
	ContentFilter {
		/// Verbatim provider details, when supplied.
		details: Option<Box<RawValue>>,
	},
}
/// Equality is byte equality of stored JSON text, preserving verbatim round
/// trips.
impl PartialEq for Stop {
	fn eq(&self, other: &Self) -> bool {
		match (self, other) {
			(Self::EndTurn, Self::EndTurn)
			| (Self::MaxTokens, Self::MaxTokens)
			| (Self::ToolUse, Self::ToolUse) => true,
			(Self::Refusal { details: a }, Self::Refusal { details: b })
			| (Self::Aborted { details: a }, Self::Aborted { details: b })
			| (Self::ContentFilter { details: a }, Self::ContentFilter { details: b }) => {
				opt_raw_eq(a.as_deref(), b.as_deref())
			},
			_ => false,
		}
	}
}

impl Eq for Stop {}

#[derive(Deserialize)]
struct StopProbe {
	reason: SmolStr,
}

#[derive(Deserialize)]
struct StopDetails {
	details: Option<Box<RawValue>>,
}

impl<'de> Deserialize<'de> for Stop {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		let raw = Box::<RawValue>::deserialize(deserializer)?;
		let probe: StopProbe = serde_json::from_str(raw.get()).map_err(D::Error::custom)?;
		match probe.reason.as_str() {
			"end_turn" => Ok(Self::EndTurn),
			"max_tokens" => Ok(Self::MaxTokens),
			"tool_use" => Ok(Self::ToolUse),
			"refusal" => {
				let payload: StopDetails = serde_json::from_str(raw.get()).map_err(D::Error::custom)?;
				Ok(Self::Refusal { details: payload.details })
			},
			"aborted" => {
				let payload: StopDetails = serde_json::from_str(raw.get()).map_err(D::Error::custom)?;
				Ok(Self::Aborted { details: payload.details })
			},
			"content_filter" => {
				let payload: StopDetails = serde_json::from_str(raw.get()).map_err(D::Error::custom)?;
				Ok(Self::ContentFilter { details: payload.details })
			},
			reason => Err(D::Error::custom(format_args!("unknown stop reason `{reason}`"))),
		}
	}
}

/// Wall-clock measurements for an assistant turn.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timing {
	/// Total request duration in milliseconds.
	pub duration_ms: u64,
	/// Time to the first generated token in milliseconds.
	pub ttft_ms:     u64,
}

/// Context-window state observed when a request was sent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CtxSnapshot {
	/// Tokens occupying the model context window.
	pub tokens: u64,
	/// Maximum tokens available in the context window.
	pub limit:  u64,
}

/// Origin information attached to user or developer content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attribution {
	/// Stable source kind, such as a user, hook, or imported session.
	pub source: SmolStr,
	/// Optional source-specific identifier.
	pub id:     Option<SmolStr>,
}

/// A failed inference request that produced no conversational content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestError {
	/// Human-readable error message.
	pub message: SmolStr,
	/// Provider or protocol error code.
	pub code:    Option<SmolStr>,
	/// HTTP or provider-equivalent status code.
	pub status:  Option<u16>,
	/// Verbatim structured error details.
	pub details: Option<Box<RawValue>>,
}
/// Equality is byte equality of stored JSON text, preserving verbatim round
/// trips.
impl PartialEq for RequestError {
	fn eq(&self, other: &Self) -> bool {
		self.message == other.message
			&& self.code == other.code
			&& self.status == other.status
			&& opt_raw_eq(self.details.as_deref(), other.details.as_deref())
	}
}

impl Eq for RequestError {}

/// The source that assigned a transcript title.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TitleSource {
	/// A person explicitly chose the title.
	User,
	/// An assistant generated the title.
	Assistant,
	/// The runtime assigned the title.
	System,
	/// Migration imported the title from an older journal.
	Imported,
}

/// An append-only correction to an earlier message event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum AmendPatch {
	/// Prune an earlier assistant message to a prefix of its blocks.
	Prune {
		/// Number of leading blocks that remain live.
		keep_blocks: u64,
	},
	/// Restore the original assistant turn after a failed retry attempt.
	RetryRecovery {
		/// Original assistant blocks replaced by the retry.
		content:     Vec<Block>,
		/// Original stop reason.
		stop:        Stop,
		/// Original token usage.
		usage:       Usage,
		/// Original provider response identifier, when present.
		response_id: Option<SmolStr>,
	},
}

#[derive(Deserialize)]
struct AmendProbe {
	op: SmolStr,
}

#[derive(Deserialize)]
struct PrunePayload {
	keep_blocks: u64,
}

#[derive(Deserialize)]
struct RetryRecoveryPayload {
	content:     Vec<Block>,
	stop:        Stop,
	usage:       Usage,
	response_id: Option<SmolStr>,
}

impl<'de> Deserialize<'de> for AmendPatch {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		let raw = Box::<RawValue>::deserialize(deserializer)?;
		let probe: AmendProbe = serde_json::from_str(raw.get()).map_err(D::Error::custom)?;
		match probe.op.as_str() {
			"prune" => {
				let payload: PrunePayload =
					serde_json::from_str(raw.get()).map_err(D::Error::custom)?;
				Ok(Self::Prune { keep_blocks: payload.keep_blocks })
			},
			"retry_recovery" => {
				let payload: RetryRecoveryPayload =
					serde_json::from_str(raw.get()).map_err(D::Error::custom)?;
				Ok(Self::RetryRecovery {
					content:     payload.content,
					stop:        payload.stop,
					usage:       payload.usage,
					response_id: payload.response_id,
				})
			},
			op => Err(D::Error::custom(format_args!("unknown amendment operation `{op}`"))),
		}
	}
}

/// Effective and user-configured thinking selections for an inference request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingSel {
	/// Selection actually sent to the provider.
	pub effective:  SmolStr,
	/// Selection configured by the user, including automatic modes.
	pub configured: SmolStr,
}

/// A role-specific model selection change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelChange {
	/// Model role affected by the change.
	pub role:     SmolStr,
	/// New model selection.
	pub model:    ModelRef,
	/// Whether this selection is a fallback rather than the primary choice.
	pub fallback: bool,
}

/// A provider service-tier selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Tier(
	/// The provider tier name.
	pub SmolStr,
);

/// A credential pin used to keep a session on a stable provider account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pin {
	/// Provider whose credential is pinned.
	pub provider:   ProviderId,
	/// Provider-local credential identifier.
	pub credential: SmolStr,
}
