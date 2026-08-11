//! Conversation messages stored by transcript events.

use omp_core::Str;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::value::RawValue;

use super::{
	block::Block,
	raweq::opt_raw_eq,
	types::{Attribution, CallId, CtxSnapshot, FeatureId, ModelRef, Stop, Timing, Usage},
};
use crate::blob::BlobRef;

/// User-shaped content that may participate in model context.
pub type Content = Vec<UserBlock>;

/// A conversation message.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Msg {
	/// User-authored or system-injected user content.
	User {
		/// Ordered user content blocks.
		content:     Vec<UserBlock>,
		/// Whether the runtime injected this message automatically.
		synthetic:   bool,
		/// Whether this message was interjected during an active turn.
		steering:    bool,
		/// Optional origin metadata.
		attribution: Option<Attribution>,
	},
	/// Developer instruction content.
	Developer {
		/// Ordered developer content blocks.
		content:     Vec<UserBlock>,
		/// Optional origin metadata.
		attribution: Option<Attribution>,
	},
	/// A completed or partially completed assistant turn.
	Assistant {
		/// Ordered provider-neutral assistant blocks.
		content:     Vec<Block>,
		/// Model that generated the turn.
		model:       ModelRef,
		/// Reason generation stopped.
		stop:        Stop,
		/// Token usage for the turn.
		usage:       Usage,
		/// Provider response identifier, when supplied.
		response_id: Option<Str>,
		/// Aggregator-reported upstream route, when supplied.
		upstream:    Option<Str>,
		/// Context-window snapshot at request time.
		ctx:         Option<CtxSnapshot>,
		/// Request timing measurements.
		timing:      Timing,
		/// Requested features silently dropped by the serving path.
		disabled:    Vec<FeatureId>,
	},
	/// Result returned for a preceding tool invocation.
	ToolResult {
		/// Bare call identifier paired with the assistant tool block.
		call:          CallId,
		/// Harness-visible tool name.
		tool:          Str,
		/// Ordered result content.
		content:       Vec<UserBlock>,
		/// Verbatim renderer-specific details.
		details:       Option<Box<RawValue>>,
		/// Whether tool execution failed.
		error:         bool,
		/// Whether this result should be omitted from future model context.
		useless:       bool,
		/// Verbatim provider metadata used for computer-use replay.
		provider_meta: Option<Box<RawValue>>,
	},
}
/// Equality is byte equality of stored JSON text, preserving verbatim round
/// trips.
impl PartialEq for Msg {
	fn eq(&self, other: &Self) -> bool {
		match (self, other) {
			(
				Self::User {
					content: a_content,
					synthetic: a_synthetic,
					steering: a_steering,
					attribution: a_attribution,
				},
				Self::User {
					content: b_content,
					synthetic: b_synthetic,
					steering: b_steering,
					attribution: b_attribution,
				},
			) => {
				a_content == b_content
					&& a_synthetic == b_synthetic
					&& a_steering == b_steering
					&& a_attribution == b_attribution
			},
			(
				Self::Developer { content: a_content, attribution: a_attribution },
				Self::Developer { content: b_content, attribution: b_attribution },
			) => a_content == b_content && a_attribution == b_attribution,
			(
				Self::Assistant {
					content: a_content,
					model: a_model,
					stop: a_stop,
					usage: a_usage,
					response_id: a_response_id,
					upstream: a_upstream,
					ctx: a_ctx,
					timing: a_timing,
					disabled: a_disabled,
				},
				Self::Assistant {
					content: b_content,
					model: b_model,
					stop: b_stop,
					usage: b_usage,
					response_id: b_response_id,
					upstream: b_upstream,
					ctx: b_ctx,
					timing: b_timing,
					disabled: b_disabled,
				},
			) => {
				a_content == b_content
					&& a_model == b_model
					&& a_stop == b_stop
					&& a_usage == b_usage
					&& a_response_id == b_response_id
					&& a_upstream == b_upstream
					&& a_ctx == b_ctx
					&& a_timing == b_timing
					&& a_disabled == b_disabled
			},
			(
				Self::ToolResult {
					call: a_call,
					tool: a_tool,
					content: a_content,
					details: a_details,
					error: a_error,
					useless: a_useless,
					provider_meta: a_provider_meta,
				},
				Self::ToolResult {
					call: b_call,
					tool: b_tool,
					content: b_content,
					details: b_details,
					error: b_error,
					useless: b_useless,
					provider_meta: b_provider_meta,
				},
			) => {
				a_call == b_call
					&& a_tool == b_tool
					&& a_content == b_content
					&& opt_raw_eq(a_details.as_deref(), b_details.as_deref())
					&& a_error == b_error
					&& a_useless == b_useless
					&& opt_raw_eq(a_provider_meta.as_deref(), b_provider_meta.as_deref())
			},
			_ => false,
		}
	}
}

impl Eq for Msg {}

#[derive(Deserialize)]
struct RoleProbe {
	role: Str,
}

#[derive(Deserialize)]
struct UserPayload {
	content:     Vec<UserBlock>,
	synthetic:   bool,
	steering:    bool,
	attribution: Option<Attribution>,
}

#[derive(Deserialize)]
struct DeveloperPayload {
	content:     Vec<UserBlock>,
	attribution: Option<Attribution>,
}

#[derive(Deserialize)]
struct AssistantPayload {
	content:     Vec<Block>,
	model:       ModelRef,
	stop:        Stop,
	usage:       Usage,
	response_id: Option<Str>,
	upstream:    Option<Str>,
	ctx:         Option<CtxSnapshot>,
	timing:      Timing,
	disabled:    Vec<FeatureId>,
}

#[derive(Deserialize)]
struct ToolResultPayload {
	call:          CallId,
	tool:          Str,
	content:       Vec<UserBlock>,
	details:       Option<Box<RawValue>>,
	error:         bool,
	useless:       bool,
	provider_meta: Option<Box<RawValue>>,
}

impl<'de> Deserialize<'de> for Msg {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		let raw = Box::<RawValue>::deserialize(deserializer)?;
		let probe: RoleProbe = serde_json::from_str(raw.get()).map_err(D::Error::custom)?;
		match probe.role.as_str() {
			"user" => {
				let payload: UserPayload = serde_json::from_str(raw.get()).map_err(D::Error::custom)?;
				Ok(Self::User {
					content:     payload.content,
					synthetic:   payload.synthetic,
					steering:    payload.steering,
					attribution: payload.attribution,
				})
			},
			"developer" => {
				let payload: DeveloperPayload =
					serde_json::from_str(raw.get()).map_err(D::Error::custom)?;
				Ok(Self::Developer { content: payload.content, attribution: payload.attribution })
			},
			"assistant" => {
				let payload: AssistantPayload =
					serde_json::from_str(raw.get()).map_err(D::Error::custom)?;
				Ok(Self::Assistant {
					content:     payload.content,
					model:       payload.model,
					stop:        payload.stop,
					usage:       payload.usage,
					response_id: payload.response_id,
					upstream:    payload.upstream,
					ctx:         payload.ctx,
					timing:      payload.timing,
					disabled:    payload.disabled,
				})
			},
			"tool_result" => {
				let payload: ToolResultPayload =
					serde_json::from_str(raw.get()).map_err(D::Error::custom)?;
				Ok(Self::ToolResult {
					call:          payload.call,
					tool:          payload.tool,
					content:       payload.content,
					details:       payload.details,
					error:         payload.error,
					useless:       payload.useless,
					provider_meta: payload.provider_meta,
				})
			},
			role => Err(D::Error::custom(format_args!("unknown message role `{role}`"))),
		}
	}
}

/// A user-shaped content block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum UserBlock {
	/// Text content.
	Text {
		/// Text exactly as supplied.
		text: Str,
	},
	/// Image content.
	Image {
		/// Content-addressed image payload.
		blob: BlobRef,
	},
}
