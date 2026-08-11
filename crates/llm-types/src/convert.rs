use std::collections::BTreeMap;

use omp_core::Str;
use omp_proto::{inference::v1 as pb, thread::v1 as pb_thread};
use serde_json::{Map, Number, Value};
use thiserror::Error;

use super::*;

/// Failure converting an incomplete or invalid protobuf binding into canonical
/// native types.
#[derive(Debug, Error)]
pub enum ConvertError {
	/// A required proto3 message or oneof was absent.
	#[error("required protobuf field `{0}` is absent")]
	MissingField(&'static str),
	/// An `*_UNSPECIFIED` enum cannot represent a concrete native value.
	#[error("protobuf enum `{0}` is UNSPECIFIED")]
	Unspecified(&'static str),
	/// An integer does not name a known value of the protobuf enum.
	#[error("protobuf enum `{name}` has unknown value {value}")]
	UnknownEnum {
		/// Enum type or field name.
		name:  &'static str,
		/// Unknown wire value.
		value: i32,
	},
	/// A content digest did not have the required BLAKE3-256 width.
	#[error("blob hash has {0} bytes; expected 32")]
	InvalidHash(usize),
	/// A protobuf double cannot be represented by `serde_json::Number`.
	#[error("protobuf Value contains a non-finite JSON number")]
	NonFiniteNumber,
	/// A protobuf tool-call identity was not a canonical ULID.
	#[error("invalid canonical tool-call id: {0}")]
	InvalidCallId(#[from] ulid::DecodeError),
	/// A gateway-only request form has no direct in-process chat representation.
	#[error("protobuf request form `{0}` is not a stateless native ChatRequest")]
	UnsupportedRequestForm(&'static str),
}

const fn missing(name: &'static str) -> ConvertError {
	ConvertError::MissingField(name)
}

fn props_to_proto(props: Props) -> Option<pb::ValueMap> {
	(!props.is_empty()).then(|| props.into())
}

fn props_from_proto(value: Option<pb::ValueMap>) -> Result<Props, ConvertError> {
	value.map_or_else(|| Ok(Props::default()), TryInto::try_into)
}

impl From<Props> for pb::ValueMap {
	fn from(value: Props) -> Self {
		Self {
			fields: value
				.0
				.into_iter()
				.map(|(key, value)| (key.into(), json_to_proto(value)))
				.collect(),
		}
	}
}

impl TryFrom<pb::ValueMap> for Props {
	type Error = ConvertError;

	fn try_from(value: pb::ValueMap) -> Result<Self, Self::Error> {
		value
			.fields
			.into_iter()
			.map(|(key, value)| Ok((Str::from(key), proto_to_json(value)?)))
			.collect::<Result<BTreeMap<_, _>, _>>()
			.map(Self)
	}
}

fn json_to_proto(value: Value) -> pb::Value {
	use pb::value::Kind;

	let kind = match value {
		Value::Null => Kind::Null(true),
		Value::Bool(value) => Kind::Bool(value),
		Value::Number(value) => {
			if let Some(value) = value.as_i64() {
				Kind::Int(value)
			} else if let Some(value) = value.as_u64() {
				Kind::Uint(value)
			} else {
				Kind::Double(value.as_f64().expect("JSON numbers are finite"))
			}
		},
		Value::String(value) => Kind::String(value),
		Value::Array(values) => {
			Kind::List(pb::ValueList { values: values.into_iter().map(json_to_proto).collect() })
		},
		Value::Object(fields) => Kind::Map(pb::ValueMap {
			fields: fields
				.into_iter()
				.map(|(key, value)| (key, json_to_proto(value)))
				.collect(),
		}),
	};
	pb::Value { kind: Some(kind) }
}

fn proto_to_json(value: pb::Value) -> Result<Value, ConvertError> {
	use pb::value::Kind;

	match value.kind.ok_or_else(|| missing("Value.kind"))? {
		Kind::Null(_) => Ok(Value::Null),
		Kind::Int(value) => Ok(Value::Number(Number::from(value))),
		Kind::Uint(value) => Ok(Value::Number(Number::from(value))),
		Kind::Double(value) => Number::from_f64(value)
			.map(Value::Number)
			.ok_or(ConvertError::NonFiniteNumber),
		Kind::String(value) => Ok(Value::String(value)),
		Kind::Bool(value) => Ok(Value::Bool(value)),
		Kind::Map(value) => value
			.fields
			.into_iter()
			.map(|(key, value)| Ok((key, proto_to_json(value)?)))
			.collect::<Result<Map<_, _>, _>>()
			.map(Value::Object),
		Kind::List(value) => value
			.values
			.into_iter()
			.map(proto_to_json)
			.collect::<Result<Vec<_>, _>>()
			.map(Value::Array),
	}
}

impl From<Revision> for pb_thread::Revision {
	fn from(value: Revision) -> Self {
		Self { head: value.head, token: value.token }
	}
}

impl From<pb_thread::Revision> for Revision {
	fn from(value: pb_thread::Revision) -> Self {
		Self { head: value.head, token: value.token }
	}
}

impl From<Thread> for pb_thread::Thread {
	fn from(value: Thread) -> Self {
		Self { items: value.items.into_iter().map(Into::into).collect() }
	}
}

impl TryFrom<pb_thread::Thread> for Thread {
	type Error = ConvertError;

	fn try_from(value: pb_thread::Thread) -> Result<Self, Self::Error> {
		Ok(Self {
			items: value
				.items
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
		})
	}
}

impl From<Item> for pb_thread::Item {
	fn from(value: Item) -> Self {
		use pb_thread::item::Kind;
		let kind = match value.kind {
			ItemKind::Message(value) => Kind::Message(value.into()),
			ItemKind::ToolCall(value) => Kind::ToolCall(value.into()),
			ItemKind::ToolResult(value) => Kind::ToolResult(value.into()),
		};
		Self {
			seq:           value.seq,
			created_at_ms: value.created_at_ms,
			props:         props_to_proto(value.props),
			kind:          Some(kind),
		}
	}
}

impl TryFrom<pb_thread::Item> for Item {
	type Error = ConvertError;

	fn try_from(value: pb_thread::Item) -> Result<Self, Self::Error> {
		use pb_thread::item::Kind;
		let kind = match value.kind.ok_or_else(|| missing("Item.kind"))? {
			Kind::Message(value) => ItemKind::Message(value.try_into()?),
			Kind::ToolCall(value) => ItemKind::ToolCall(value.try_into()?),
			Kind::ToolResult(value) => ItemKind::ToolResult(value.try_into()?),
		};
		Ok(Self {
			seq: value.seq,
			created_at_ms: value.created_at_ms,
			kind,
			props: props_from_proto(value.props)?,
		})
	}
}

impl From<Role> for i32 {
	fn from(value: Role) -> Self {
		match value {
			Role::System => pb_thread::Role::System as Self,
			Role::User => pb_thread::Role::User as Self,
			Role::Assistant => pb_thread::Role::Assistant as Self,
		}
	}
}

const fn role_from_proto(value: i32) -> Result<Role, ConvertError> {
	match value {
		x if x == pb_thread::Role::Unspecified as i32 => Err(ConvertError::Unspecified("Role")),
		x if x == pb_thread::Role::System as i32 => Ok(Role::System),
		x if x == pb_thread::Role::User as i32 => Ok(Role::User),
		x if x == pb_thread::Role::Assistant as i32 => Ok(Role::Assistant),
		value => Err(ConvertError::UnknownEnum { name: "Role", value }),
	}
}

impl From<Message> for pb_thread::Message {
	fn from(value: Message) -> Self {
		Self { role: value.role.into(), parts: value.parts.into_iter().map(Into::into).collect() }
	}
}

impl TryFrom<pb_thread::Message> for Message {
	type Error = ConvertError;

	fn try_from(value: pb_thread::Message) -> Result<Self, Self::Error> {
		Ok(Self {
			role:  role_from_proto(value.role)?,
			parts: value
				.parts
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
		})
	}
}

impl From<Part> for pb_thread::Part {
	fn from(value: Part) -> Self {
		use pb_thread::part::Kind;
		let kind = match value {
			Part::Text(value) => Kind::Text(value.into()),
			Part::Thinking(value) => Kind::Thinking(value.into()),
			Part::Blob(value) => Kind::Blob(value.into()),
			Part::Fallback(value) => Kind::Fallback(value.into()),
			Part::ServerTool(value) => Kind::ServerTool(value.into()),
		};
		Self { kind: Some(kind) }
	}
}

impl TryFrom<pb_thread::Part> for Part {
	type Error = ConvertError;

	fn try_from(value: pb_thread::Part) -> Result<Self, Self::Error> {
		use pb_thread::part::Kind;
		match value.kind.ok_or_else(|| missing("Part.kind"))? {
			Kind::Text(value) => Ok(Self::Text(value.into())),
			Kind::Thinking(value) => Ok(Self::Thinking(value.into())),
			Kind::Blob(value) => Ok(Self::Blob(value.try_into()?)),
			Kind::Fallback(value) => Ok(Self::Fallback(value.into())),
			Kind::ServerTool(value) => Ok(Self::ServerTool(value.try_into()?)),
		}
	}
}

impl From<Thinking> for pb_thread::Thinking {
	fn from(value: Thinking) -> Self {
		Self { text: value.text.into(), signature: value.signature, redacted: value.redacted }
	}
}

impl From<pb_thread::Thinking> for Thinking {
	fn from(value: pb_thread::Thinking) -> Self {
		Self { text: value.text.into(), signature: value.signature, redacted: value.redacted }
	}
}

impl From<ModelFallback> for pb_thread::ModelFallback {
	fn from(value: ModelFallback) -> Self {
		Self { from_model: value.from_model.into(), to_model: value.to_model.into() }
	}
}

impl From<pb_thread::ModelFallback> for ModelFallback {
	fn from(value: pb_thread::ModelFallback) -> Self {
		Self { from_model: value.from_model.into(), to_model: value.to_model.into() }
	}
}

impl From<ServerTool> for pb_thread::ServerTool {
	fn from(value: ServerTool) -> Self {
		let kind = match value.kind {
			ServerToolKind::Call => pb_thread::server_tool::Kind::Call,
			ServerToolKind::Result => pb_thread::server_tool::Kind::Result,
		};
		Self {
			provider:          value.provider.into(),
			kind:              kind as i32,
			id:                value.id.into(),
			name:              value.name.into(),
			payload_json:      value.payload_json,
			provider_metadata: value.provider_metadata.map(Into::into),
		}
	}
}

impl TryFrom<pb_thread::ServerTool> for ServerTool {
	type Error = ConvertError;

	fn try_from(value: pb_thread::ServerTool) -> Result<Self, Self::Error> {
		let kind = match value.kind {
			x if x == pb_thread::server_tool::Kind::Unspecified as i32 => {
				return Err(ConvertError::Unspecified("ServerTool.Kind"));
			},
			x if x == pb_thread::server_tool::Kind::Call as i32 => ServerToolKind::Call,
			x if x == pb_thread::server_tool::Kind::Result as i32 => ServerToolKind::Result,
			value => return Err(ConvertError::UnknownEnum { name: "ServerTool.Kind", value }),
		};
		Ok(Self {
			provider: value.provider.into(),
			kind,
			id: value.id.into(),
			name: value.name.into(),
			payload_json: value.payload_json,
			provider_metadata: value.provider_metadata.map(TryInto::try_into).transpose()?,
		})
	}
}

impl From<BlobPart> for pb_thread::Blob {
	fn from(value: BlobPart) -> Self {
		let detail = match value.detail {
			None => pb_thread::blob::Detail::Unspecified,
			Some(ImageDetail::Auto) => pb_thread::blob::Detail::Auto,
			Some(ImageDetail::Low) => pb_thread::blob::Detail::Low,
			Some(ImageDetail::High) => pb_thread::blob::Detail::High,
			Some(ImageDetail::Original) => pb_thread::blob::Detail::Original,
		};
		Self {
			hash:   bytes::Bytes::copy_from_slice(&value.hash),
			mime:   value.mime.into(),
			size:   value.size,
			inline: value.inline,
			detail: detail as i32,
		}
	}
}

impl TryFrom<pb_thread::Blob> for BlobPart {
	type Error = ConvertError;

	fn try_from(value: pb_thread::Blob) -> Result<Self, Self::Error> {
		let hash = value
			.hash
			.as_ref()
			.try_into()
			.map_err(|_| ConvertError::InvalidHash(value.hash.len()))?;
		let detail = match value.detail {
			x if x == pb_thread::blob::Detail::Unspecified as i32 => None,
			x if x == pb_thread::blob::Detail::Auto as i32 => Some(ImageDetail::Auto),
			x if x == pb_thread::blob::Detail::Low as i32 => Some(ImageDetail::Low),
			x if x == pb_thread::blob::Detail::High as i32 => Some(ImageDetail::High),
			x if x == pb_thread::blob::Detail::Original as i32 => Some(ImageDetail::Original),
			value => return Err(ConvertError::UnknownEnum { name: "Blob.Detail", value }),
		};
		Ok(Self { hash, mime: value.mime.into(), size: value.size, inline: value.inline, detail })
	}
}

impl From<ToolCall> for pb_thread::ToolCall {
	fn from(value: ToolCall) -> Self {
		Self {
			id:                value.id.to_string(),
			name:              value.name.into(),
			args_json:         value.args_json,
			thought_signature: value.thought_signature,
			intent:            value.intent.map(Into::into),
			raw:               value.raw,
			custom_wire_name:  value.custom_wire_name.map(Into::into),
			provider_metadata: value.provider_metadata.map(Into::into),
		}
	}
}

impl TryFrom<pb_thread::ToolCall> for ToolCall {
	type Error = ConvertError;

	fn try_from(value: pb_thread::ToolCall) -> Result<Self, Self::Error> {
		Ok(Self {
			id:                value.id.parse()?,
			name:              value.name.into(),
			args_json:         value.args_json,
			thought_signature: value.thought_signature,
			intent:            value.intent.map(Into::into),
			raw:               value.raw,
			custom_wire_name:  value.custom_wire_name.map(Into::into),
			provider_metadata: value.provider_metadata.map(TryInto::try_into).transpose()?,
		})
	}
}

impl From<ToolResult> for pb_thread::ToolResult {
	fn from(value: ToolResult) -> Self {
		let attribution = match value.attribution {
			None => pb_thread::tool_result::Attribution::Unspecified,
			Some(MessageAttribution::User) => pb_thread::tool_result::Attribution::User,
			Some(MessageAttribution::Agent) => pb_thread::tool_result::Attribution::Agent,
		};
		Self {
			call_id:           value.call_id.to_string(),
			name:              value.name.into(),
			parts:             value.parts.into_iter().map(Into::into).collect(),
			is_error:          value.is_error,
			details:           value.details.map(json_to_proto),
			attribution:       attribution as i32,
			pruned_at_ms:      value.pruned_at_ms,
			useless:           value.useless,
			provider_metadata: value.provider_metadata.map(Into::into),
		}
	}
}

impl TryFrom<pb_thread::ToolResult> for ToolResult {
	type Error = ConvertError;

	fn try_from(value: pb_thread::ToolResult) -> Result<Self, Self::Error> {
		let attribution = match value.attribution {
			x if x == pb_thread::tool_result::Attribution::Unspecified as i32 => None,
			x if x == pb_thread::tool_result::Attribution::User as i32 => {
				Some(MessageAttribution::User)
			},
			x if x == pb_thread::tool_result::Attribution::Agent as i32 => {
				Some(MessageAttribution::Agent)
			},
			value => return Err(ConvertError::UnknownEnum { name: "ToolResult.Attribution", value }),
		};
		Ok(Self {
			call_id: value.call_id.parse()?,
			name: value.name.into(),
			parts: value
				.parts
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
			is_error: value.is_error,
			details: value.details.map(proto_to_json).transpose()?,
			attribution,
			pruned_at_ms: value.pruned_at_ms,
			useless: value.useless,
			provider_metadata: value.provider_metadata.map(TryInto::try_into).transpose()?,
		})
	}
}

const fn fallback_to_proto(value: Fallback) -> i32 {
	match value {
		Fallback::Error => pb::Fallback::Error as i32,
		Fallback::Ignore => pb::Fallback::Ignore as i32,
		Fallback::Emulate => pb::Fallback::Emulate as i32,
	}
}

const fn fallback_from_proto(value: i32) -> Result<Fallback, ConvertError> {
	match value {
		x if x == pb::Fallback::Unspecified as i32 => Err(ConvertError::Unspecified("Fallback")),
		x if x == pb::Fallback::Error as i32 => Ok(Fallback::Error),
		x if x == pb::Fallback::Ignore as i32 => Ok(Fallback::Ignore),
		x if x == pb::Fallback::Emulate as i32 => Ok(Fallback::Emulate),
		value => Err(ConvertError::UnknownEnum { name: "Fallback", value }),
	}
}

const fn effort_to_proto(value: Option<Effort>) -> i32 {
	match value {
		None => pb::Effort::Unspecified as i32,
		Some(Effort::Off) => pb::Effort::Off as i32,
		Some(Effort::Minimal) => pb::Effort::Minimal as i32,
		Some(Effort::Low) => pb::Effort::Low as i32,
		Some(Effort::Medium) => pb::Effort::Medium as i32,
		Some(Effort::High) => pb::Effort::High as i32,
		Some(Effort::XHigh) => pb::Effort::Xhigh as i32,
		Some(Effort::Max) => pb::Effort::Max as i32,
	}
}

const fn effort_from_proto(value: i32) -> Result<Option<Effort>, ConvertError> {
	match value {
		x if x == pb::Effort::Unspecified as i32 => Ok(None),
		x if x == pb::Effort::Off as i32 => Ok(Some(Effort::Off)),
		x if x == pb::Effort::Minimal as i32 => Ok(Some(Effort::Minimal)),
		x if x == pb::Effort::Low as i32 => Ok(Some(Effort::Low)),
		x if x == pb::Effort::Medium as i32 => Ok(Some(Effort::Medium)),
		x if x == pb::Effort::High as i32 => Ok(Some(Effort::High)),
		x if x == pb::Effort::Xhigh as i32 => Ok(Some(Effort::XHigh)),
		x if x == pb::Effort::Max as i32 => Ok(Some(Effort::Max)),
		value => Err(ConvertError::UnknownEnum { name: "Effort", value }),
	}
}

impl From<ToolDef> for pb::ToolDef {
	fn from(value: ToolDef) -> Self {
		Self {
			name:        value.name.into(),
			description: value.description.into(),
			schema_json: value.schema_json,
			strict:      value.strict,
		}
	}
}

impl From<pb::ToolDef> for ToolDef {
	fn from(value: pb::ToolDef) -> Self {
		Self {
			name:        value.name.into(),
			description: value.description.into(),
			schema_json: value.schema_json,
			strict:      value.strict,
		}
	}
}

impl From<Feature<ToolChoice>> for pb::ToolChoice {
	fn from(value: Feature<ToolChoice>) -> Self {
		let (mode, name) = match value.value {
			ToolChoice::Auto => (pb::tool_choice::Mode::Auto, Str::default()),
			ToolChoice::None => (pb::tool_choice::Mode::None, Str::default()),
			ToolChoice::Required => (pb::tool_choice::Mode::Required, Str::default()),
			ToolChoice::Named(name) => (pb::tool_choice::Mode::Named, name),
		};
		Self {
			mode:           mode as i32,
			name:           name.into(),
			on_unsupported: fallback_to_proto(value.on_unsupported),
		}
	}
}

impl TryFrom<pb::ToolChoice> for Feature<ToolChoice> {
	type Error = ConvertError;

	fn try_from(value: pb::ToolChoice) -> Result<Self, Self::Error> {
		let choice = match value.mode {
			x if x == pb::tool_choice::Mode::Unspecified as i32 => {
				return Err(ConvertError::Unspecified("ToolChoice.Mode"));
			},
			x if x == pb::tool_choice::Mode::Auto as i32 => ToolChoice::Auto,
			x if x == pb::tool_choice::Mode::None as i32 => ToolChoice::None,
			x if x == pb::tool_choice::Mode::Required as i32 => ToolChoice::Required,
			x if x == pb::tool_choice::Mode::Named as i32 => ToolChoice::Named(value.name.into()),
			value => return Err(ConvertError::UnknownEnum { name: "ToolChoice.Mode", value }),
		};
		Ok(Self {
			value:          choice,
			on_unsupported: fallback_from_proto(value.on_unsupported)?,
		})
	}
}

impl From<Sampling> for pb::Sampling {
	fn from(value: Sampling) -> Self {
		let stop_present = value.stop.is_some();
		Self {
			temperature:        value.temperature,
			top_p:              value.top_p,
			top_k:              value.top_k,
			min_p:              value.min_p,
			frequency_penalty:  value.frequency_penalty,
			presence_penalty:   value.presence_penalty,
			repetition_penalty: value.repetition_penalty,
			stop:               value
				.stop
				.unwrap_or_default()
				.into_iter()
				.map(Into::into)
				.collect(),
			stop_present:       stop_present.then_some(true),
			max_output_tokens:  value.max_output_tokens,
		}
	}
}

impl From<pb::Sampling> for Sampling {
	fn from(value: pb::Sampling) -> Self {
		let stop_present = value.stop_present.is_some() || !value.stop.is_empty();
		Self {
			temperature:        value.temperature,
			top_p:              value.top_p,
			top_k:              value.top_k,
			min_p:              value.min_p,
			frequency_penalty:  value.frequency_penalty,
			presence_penalty:   value.presence_penalty,
			repetition_penalty: value.repetition_penalty,
			stop:               stop_present.then(|| value.stop.into_iter().map(Into::into).collect()),
			max_output_tokens:  value.max_output_tokens,
		}
	}
}

impl From<Feature<Reasoning>> for pb::Reasoning {
	fn from(value: Feature<Reasoning>) -> Self {
		Self {
			effort:         effort_to_proto(value.value.effort),
			budget_tokens:  value.value.budget_tokens,
			hide_summary:   value.value.hide_summary,
			on_unsupported: fallback_to_proto(value.on_unsupported),
		}
	}
}

impl TryFrom<pb::Reasoning> for Feature<Reasoning> {
	type Error = ConvertError;

	fn try_from(value: pb::Reasoning) -> Result<Self, Self::Error> {
		Ok(Self {
			value:          Reasoning {
				effort:        effort_from_proto(value.effort)?,
				budget_tokens: value.budget_tokens,
				hide_summary:  value.hide_summary,
			},
			on_unsupported: fallback_from_proto(value.on_unsupported)?,
		})
	}
}

impl From<CacheHint> for pb::CacheHint {
	fn from(value: CacheHint) -> Self {
		let retention = match value.retention {
			None => pb::cache_hint::Retention::Unspecified,
			Some(CacheRetention::None) => pb::cache_hint::Retention::None,
			Some(CacheRetention::Short) => pb::cache_hint::Retention::Short,
			Some(CacheRetention::Long) => pb::cache_hint::Retention::Long,
		};
		let mode = match value.mode {
			None => pb::cache_hint::Mode::Unspecified,
			Some(PromptCacheMode::Implicit) => pb::cache_hint::Mode::Implicit,
			Some(PromptCacheMode::Explicit) => pb::cache_hint::Mode::Explicit,
		};
		let ttl = match value.ttl {
			None => pb::cache_hint::Ttl::Unspecified,
			Some(PromptCacheTtl::ThirtyMinutes) => pb::cache_hint::Ttl::ThirtyMinutes,
		};
		let breakpoint = match value.breakpoint {
			None => pb::cache_hint::Breakpoint::Unspecified,
			Some(PromptCacheBreakpoint::LatestStableMessage) => {
				pb::cache_hint::Breakpoint::LatestStableMessage
			},
			Some(PromptCacheBreakpoint::TailTwo) => pb::cache_hint::Breakpoint::TailTwo,
			Some(PromptCacheBreakpoint::None) => pb::cache_hint::Breakpoint::None,
		};
		Self {
			session_key: value.session_key.into(),
			retention:   retention as i32,
			mode:        mode as i32,
			ttl:         ttl as i32,
			breakpoint:  breakpoint as i32,
		}
	}
}

impl TryFrom<pb::CacheHint> for CacheHint {
	type Error = ConvertError;

	fn try_from(value: pb::CacheHint) -> Result<Self, Self::Error> {
		let retention = match value.retention {
			x if x == pb::cache_hint::Retention::Unspecified as i32 => None,
			x if x == pb::cache_hint::Retention::None as i32 => Some(CacheRetention::None),
			x if x == pb::cache_hint::Retention::Short as i32 => Some(CacheRetention::Short),
			x if x == pb::cache_hint::Retention::Long as i32 => Some(CacheRetention::Long),
			value => return Err(ConvertError::UnknownEnum { name: "CacheHint.Retention", value }),
		};
		let mode = match value.mode {
			x if x == pb::cache_hint::Mode::Unspecified as i32 => None,
			x if x == pb::cache_hint::Mode::Implicit as i32 => Some(PromptCacheMode::Implicit),
			x if x == pb::cache_hint::Mode::Explicit as i32 => Some(PromptCacheMode::Explicit),
			value => return Err(ConvertError::UnknownEnum { name: "CacheHint.Mode", value }),
		};
		let ttl = match value.ttl {
			x if x == pb::cache_hint::Ttl::Unspecified as i32 => None,
			x if x == pb::cache_hint::Ttl::ThirtyMinutes as i32 => Some(PromptCacheTtl::ThirtyMinutes),
			value => return Err(ConvertError::UnknownEnum { name: "CacheHint.Ttl", value }),
		};
		let breakpoint = match value.breakpoint {
			x if x == pb::cache_hint::Breakpoint::Unspecified as i32 => None,
			x if x == pb::cache_hint::Breakpoint::LatestStableMessage as i32 => {
				Some(PromptCacheBreakpoint::LatestStableMessage)
			},
			x if x == pb::cache_hint::Breakpoint::TailTwo as i32 => {
				Some(PromptCacheBreakpoint::TailTwo)
			},
			x if x == pb::cache_hint::Breakpoint::None as i32 => Some(PromptCacheBreakpoint::None),
			value => return Err(ConvertError::UnknownEnum { name: "CacheHint.Breakpoint", value }),
		};
		Ok(Self { session_key: value.session_key.into(), retention, mode, ttl, breakpoint })
	}
}

impl From<Feature<ResponseFormat>> for pb::ResponseFormat {
	fn from(value: Feature<ResponseFormat>) -> Self {
		use pb::response_format::{self, Kind};
		let kind = match value.value.kind {
			ResponseFormatKind::JsonSchema(value) => Kind::JsonSchema(response_format::JsonSchema {
				name:        value.name.into(),
				schema_json: value.schema_json,
				strict:      value.strict,
			}),
			ResponseFormatKind::Grammar(value) => {
				let flavor = match value.flavor {
					GrammarFlavor::Lark => response_format::grammar::Flavor::Lark,
					GrammarFlavor::Regex => response_format::grammar::Flavor::Regex,
					GrammarFlavor::Gbnf => response_format::grammar::Flavor::Gbnf,
				};
				Kind::Grammar(response_format::Grammar {
					flavor:     flavor as i32,
					definition: value.definition.into(),
				})
			},
		};
		Self { on_unsupported: fallback_to_proto(value.on_unsupported), kind: Some(kind) }
	}
}

impl TryFrom<pb::ResponseFormat> for Feature<ResponseFormat> {
	type Error = ConvertError;

	fn try_from(value: pb::ResponseFormat) -> Result<Self, Self::Error> {
		use pb::response_format::Kind;
		let kind = match value.kind.ok_or_else(|| missing("ResponseFormat.kind"))? {
			Kind::JsonSchema(value) => ResponseFormatKind::JsonSchema(JsonSchema {
				name:        value.name.into(),
				schema_json: value.schema_json,
				strict:      value.strict,
			}),
			Kind::Grammar(value) => {
				let flavor = match value.flavor {
					x if x == pb::response_format::grammar::Flavor::Unspecified as i32 => {
						return Err(ConvertError::Unspecified("ResponseFormat.Grammar.Flavor"));
					},
					x if x == pb::response_format::grammar::Flavor::Lark as i32 => GrammarFlavor::Lark,
					x if x == pb::response_format::grammar::Flavor::Regex as i32 => GrammarFlavor::Regex,
					x if x == pb::response_format::grammar::Flavor::Gbnf as i32 => GrammarFlavor::Gbnf,
					value => {
						return Err(ConvertError::UnknownEnum {
							name: "ResponseFormat.Grammar.Flavor",
							value,
						});
					},
				};
				ResponseFormatKind::Grammar(Grammar { flavor, definition: value.definition.into() })
			},
		};
		Ok(Self {
			value:          ResponseFormat { kind },
			on_unsupported: fallback_from_proto(value.on_unsupported)?,
		})
	}
}

impl From<RequestMeta> for pb::RequestMeta {
	fn from(value: RequestMeta) -> Self {
		Self {
			initiator:  value.initiator.into(),
			session_id: value.session_id.into(),
			telemetry:  value
				.telemetry
				.into_iter()
				.map(|(key, value)| (key.into(), value.into()))
				.collect(),
		}
	}
}

impl From<pb::RequestMeta> for RequestMeta {
	fn from(value: pb::RequestMeta) -> Self {
		Self {
			initiator:  value.initiator.into(),
			session_id: value.session_id.into(),
			telemetry:  value
				.telemetry
				.into_iter()
				.map(|(key, value)| (key.into(), value.into()))
				.collect(),
		}
	}
}

const fn service_tier_to_proto(value: Option<ServiceTier>) -> i32 {
	match value {
		None => pb::ServiceTier::Unspecified as i32,
		Some(ServiceTier::Auto) => pb::ServiceTier::Auto as i32,
		Some(ServiceTier::Default) => pb::ServiceTier::Default as i32,
		Some(ServiceTier::Flex) => pb::ServiceTier::Flex as i32,
		Some(ServiceTier::Scale) => pb::ServiceTier::Scale as i32,
		Some(ServiceTier::Priority) => pb::ServiceTier::Priority as i32,
	}
}

const fn service_tier_from_proto(value: i32) -> Result<Option<ServiceTier>, ConvertError> {
	match value {
		x if x == pb::ServiceTier::Unspecified as i32 => Ok(None),
		x if x == pb::ServiceTier::Auto as i32 => Ok(Some(ServiceTier::Auto)),
		x if x == pb::ServiceTier::Default as i32 => Ok(Some(ServiceTier::Default)),
		x if x == pb::ServiceTier::Flex as i32 => Ok(Some(ServiceTier::Flex)),
		x if x == pb::ServiceTier::Scale as i32 => Ok(Some(ServiceTier::Scale)),
		x if x == pb::ServiceTier::Priority as i32 => Ok(Some(ServiceTier::Priority)),
		value => Err(ConvertError::UnknownEnum { name: "ServiceTier", value }),
	}
}

impl From<ServiceTierByFamily> for pb::ServiceTierByFamily {
	fn from(value: ServiceTierByFamily) -> Self {
		Self {
			openai:    service_tier_to_proto(value.openai),
			anthropic: service_tier_to_proto(value.anthropic),
			google:    service_tier_to_proto(value.google),
		}
	}
}

impl TryFrom<pb::ServiceTierByFamily> for ServiceTierByFamily {
	type Error = ConvertError;

	fn try_from(value: pb::ServiceTierByFamily) -> Result<Self, Self::Error> {
		Ok(Self {
			openai:    service_tier_from_proto(value.openai)?,
			anthropic: service_tier_from_proto(value.anthropic)?,
			google:    service_tier_from_proto(value.google)?,
		})
	}
}

impl From<TaskBudget> for pb::TaskBudget {
	fn from(value: TaskBudget) -> Self {
		Self { total_tokens: value.total_tokens, remaining_tokens: value.remaining_tokens }
	}
}

impl From<pb::TaskBudget> for TaskBudget {
	fn from(value: pb::TaskBudget) -> Self {
		Self { total_tokens: value.total_tokens, remaining_tokens: value.remaining_tokens }
	}
}

const fn response_include_to_proto(value: ResponseInclude) -> i32 {
	use pb::responses_include::Field;
	match value {
		ResponseInclude::FileSearchResults => Field::FileSearchResults as i32,
		ResponseInclude::WebSearchResults => Field::WebSearchResults as i32,
		ResponseInclude::WebSearchSources => Field::WebSearchSources as i32,
		ResponseInclude::InputImageUrl => Field::InputImageUrl as i32,
		ResponseInclude::ComputerOutputImageUrl => Field::ComputerOutputImageUrl as i32,
		ResponseInclude::CodeInterpreterOutputs => Field::CodeInterpreterOutputs as i32,
		ResponseInclude::ReasoningEncryptedContent => Field::ReasoningEncryptedContent as i32,
		ResponseInclude::OutputTextLogprobs => Field::OutputTextLogprobs as i32,
	}
}

fn response_include_from_proto(value: i32) -> Result<ResponseInclude, ConvertError> {
	use pb::responses_include::Field;
	match value {
		x if x == Field::Unspecified as i32 => {
			Err(ConvertError::Unspecified("ResponsesInclude.Field"))
		},
		x if x == Field::FileSearchResults as i32 => Ok(ResponseInclude::FileSearchResults),
		x if x == Field::WebSearchResults as i32 => Ok(ResponseInclude::WebSearchResults),
		x if x == Field::WebSearchSources as i32 => Ok(ResponseInclude::WebSearchSources),
		x if x == Field::InputImageUrl as i32 => Ok(ResponseInclude::InputImageUrl),
		x if x == Field::ComputerOutputImageUrl as i32 => Ok(ResponseInclude::ComputerOutputImageUrl),
		x if x == Field::CodeInterpreterOutputs as i32 => Ok(ResponseInclude::CodeInterpreterOutputs),
		x if x == Field::ReasoningEncryptedContent as i32 => {
			Ok(ResponseInclude::ReasoningEncryptedContent)
		},
		x if x == Field::OutputTextLogprobs as i32 => Ok(ResponseInclude::OutputTextLogprobs),
		value => Err(ConvertError::UnknownEnum { name: "ResponsesInclude.Field", value }),
	}
}

fn responses_include_to_proto(value: Vec<ResponseInclude>) -> pb::ResponsesInclude {
	pb::ResponsesInclude { fields: value.into_iter().map(response_include_to_proto).collect() }
}

fn responses_include_from_proto(
	value: pb::ResponsesInclude,
) -> Result<Vec<ResponseInclude>, ConvertError> {
	value
		.fields
		.into_iter()
		.map(response_include_from_proto)
		.collect()
}

impl From<ChatParams> for pb::ChatParams {
	fn from(value: ChatParams) -> Self {
		Self {
			// Server-resolved policy deliberately has no protobuf representation.
			model:                  value.model.into(),
			tools:                  value.tools.into_iter().map(Into::into).collect(),
			tool_choice:            value.tool_choice.map(Into::into),
			sampling:               value.sampling.map(Into::into),
			thinking:               value.thinking.map(Into::into),
			cache:                  value.cache.map(Into::into),
			response_format:        value.response_format.map(Into::into),
			meta:                   value.meta.map(Into::into),
			provider_options:       value.provider_options.map(Into::into),
			service_tier:           service_tier_to_proto(value.service_tier),
			service_tier_by_family: value.service_tier_by_family.map(Into::into),
			task_budget:            value.task_budget.map(Into::into),
			responses_include:      value.responses_include.map(responses_include_to_proto),
		}
	}
}

impl TryFrom<pb::ChatParams> for ChatParams {
	type Error = ConvertError;

	fn try_from(value: pb::ChatParams) -> Result<Self, Self::Error> {
		// Foreign requests always enter without trusted model policy.
		Ok(Self {
			model_policy:           None,
			model:                  value.model.into(),
			tools:                  value.tools.into_iter().map(Into::into).collect(),
			tool_choice:            value.tool_choice.map(TryInto::try_into).transpose()?,
			sampling:               value.sampling.map(Into::into),
			thinking:               value.thinking.map(TryInto::try_into).transpose()?,
			cache:                  value.cache.map(TryInto::try_into).transpose()?,
			response_format:        value.response_format.map(TryInto::try_into).transpose()?,
			meta:                   value.meta.map(Into::into),
			provider_options:       value.provider_options.map(TryInto::try_into).transpose()?,
			service_tier:           service_tier_from_proto(value.service_tier)?,
			service_tier_by_family: value
				.service_tier_by_family
				.map(TryInto::try_into)
				.transpose()?,
			task_budget:            value.task_budget.map(Into::into),
			responses_include:      value
				.responses_include
				.map(responses_include_from_proto)
				.transpose()?,
		})
	}
}

impl From<ChatRequest> for pb::TurnRequest {
	fn from(value: ChatRequest) -> Self {
		let (thread, params) = value.into_parts();
		Self {
			turn_id:  String::new(),
			params:   Some(params.into()),
			executor: None,
			props:    None,
			input:    Some(pb::turn_request::Input::Seed(pb::Seed {
				context_id: String::new(),
				thread:     Some(thread.into()),
			})),
		}
	}
}

impl TryFrom<pb::TurnRequest> for ChatRequest {
	type Error = ConvertError;

	fn try_from(value: pb::TurnRequest) -> Result<Self, Self::Error> {
		if !value.turn_id.is_empty() {
			return Err(ConvertError::UnsupportedRequestForm("turn_id"));
		}
		if value.executor.is_some() {
			return Err(ConvertError::UnsupportedRequestForm("executor"));
		}
		if value.props.is_some() {
			return Err(ConvertError::UnsupportedRequestForm("turn props"));
		}
		let params = value
			.params
			.ok_or_else(|| missing("TurnRequest.params"))?
			.try_into()?;
		let thread = match value.input.ok_or_else(|| missing("TurnRequest.input"))? {
			pb::turn_request::Input::Seed(seed) => {
				if !seed.context_id.is_empty() {
					return Err(ConvertError::UnsupportedRequestForm("stateful seed"));
				}
				seed
					.thread
					.ok_or_else(|| missing("Seed.thread"))?
					.try_into()?
			},
			pb::turn_request::Input::Incremental(_) => {
				return Err(ConvertError::UnsupportedRequestForm("incremental"));
			},
		};
		Ok(Self::from_parts(thread, params))
	}
}

impl From<ContextRef> for pb::ContextRef {
	fn from(value: ContextRef) -> Self {
		Self { context_id: value.context_id.into(), expected: Some(value.expected.into()) }
	}
}

impl TryFrom<pb::ContextRef> for ContextRef {
	type Error = ConvertError;

	fn try_from(value: pb::ContextRef) -> Result<Self, Self::Error> {
		Ok(Self {
			context_id: value.context_id.into(),
			expected:   value
				.expected
				.ok_or_else(|| missing("ContextRef.expected"))?
				.into(),
		})
	}
}

impl From<ThreadDelta> for pb::ThreadDelta {
	fn from(value: ThreadDelta) -> Self {
		Self {
			truncate_to: value.truncate_to,
			append:      value.append.into_iter().map(Into::into).collect(),
		}
	}
}

impl TryFrom<pb::ThreadDelta> for ThreadDelta {
	type Error = ConvertError;

	fn try_from(value: pb::ThreadDelta) -> Result<Self, Self::Error> {
		Ok(Self {
			truncate_to: value.truncate_to,
			append:      value
				.append
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
		})
	}
}

impl From<Usage> for pb::Usage {
	fn from(value: Usage) -> Self {
		let accuracy = match value.accuracy {
			Accuracy::Exact => pb::usage::Accuracy::Exact,
			Accuracy::Estimated => pb::usage::Accuracy::Estimated,
			Accuracy::Mixed => pb::usage::Accuracy::Mixed,
		};
		Self {
			input_tokens:       value.input_tokens,
			output_tokens:      value.output_tokens,
			cache_read_tokens:  value.cache_read_tokens,
			cache_write_tokens: value.cache_write_tokens,
			total_tokens:       value.total_tokens,
			context_tokens:     value.context_tokens,
			orchestration:      value.orchestration.map(Into::into),
			premium_requests:   value.premium_requests,
			reasoning_tokens:   value.reasoning_tokens,
			cache_ttl:          value.cache_ttl.map(Into::into),
			server_tools:       value.server_tools.map(Into::into),
			accuracy:           accuracy as i32,
			detail:             props_to_proto(value.detail),
		}
	}
}

impl TryFrom<pb::Usage> for Usage {
	type Error = ConvertError;

	fn try_from(value: pb::Usage) -> Result<Self, Self::Error> {
		let accuracy = match value.accuracy {
			x if x == pb::usage::Accuracy::Unspecified as i32 => {
				return Err(ConvertError::Unspecified("Usage.Accuracy"));
			},
			x if x == pb::usage::Accuracy::Exact as i32 => Accuracy::Exact,
			x if x == pb::usage::Accuracy::Estimated as i32 => Accuracy::Estimated,
			x if x == pb::usage::Accuracy::Mixed as i32 => Accuracy::Mixed,
			value => return Err(ConvertError::UnknownEnum { name: "Usage.Accuracy", value }),
		};
		Ok(Self {
			input_tokens: value.input_tokens,
			output_tokens: value.output_tokens,
			cache_read_tokens: value.cache_read_tokens,
			cache_write_tokens: value.cache_write_tokens,
			total_tokens: value.total_tokens,
			context_tokens: value.context_tokens,
			orchestration: value.orchestration.map(Into::into),
			premium_requests: value.premium_requests,
			reasoning_tokens: value.reasoning_tokens,
			cache_ttl: value.cache_ttl.map(Into::into),
			server_tools: value.server_tools.map(Into::into),
			accuracy,
			detail: props_from_proto(value.detail)?,
		})
	}
}

impl From<OrchestrationUsage> for pb::OrchestrationUsage {
	fn from(value: OrchestrationUsage) -> Self {
		Self {
			input_tokens:      value.input_tokens,
			cache_read_tokens: value.cache_read_tokens,
			output_tokens:     value.output_tokens,
		}
	}
}

impl From<pb::OrchestrationUsage> for OrchestrationUsage {
	fn from(value: pb::OrchestrationUsage) -> Self {
		Self {
			input_tokens:      value.input_tokens,
			cache_read_tokens: value.cache_read_tokens,
			output_tokens:     value.output_tokens,
		}
	}
}

impl From<CacheTtlUsage> for pb::CacheTtlUsage {
	fn from(value: CacheTtlUsage) -> Self {
		Self {
			ephemeral_5m_tokens: value.ephemeral_5m_tokens,
			ephemeral_1h_tokens: value.ephemeral_1h_tokens,
		}
	}
}

impl From<pb::CacheTtlUsage> for CacheTtlUsage {
	fn from(value: pb::CacheTtlUsage) -> Self {
		Self {
			ephemeral_5m_tokens: value.ephemeral_5m_tokens,
			ephemeral_1h_tokens: value.ephemeral_1h_tokens,
		}
	}
}

impl From<ServerToolUsage> for pb::ServerToolUsage {
	fn from(value: ServerToolUsage) -> Self {
		Self {
			web_search_requests: value.web_search_requests,
			web_fetch_requests:  value.web_fetch_requests,
		}
	}
}

impl From<pb::ServerToolUsage> for ServerToolUsage {
	fn from(value: pb::ServerToolUsage) -> Self {
		Self {
			web_search_requests: value.web_search_requests,
			web_fetch_requests:  value.web_fetch_requests,
		}
	}
}

impl From<Cost> for pb::Cost {
	fn from(value: Cost) -> Self {
		Self {
			nanos_usd:             value.nanos_usd,
			estimated:             value.estimated,
			input_nanos_usd:       value.input_nanos_usd,
			output_nanos_usd:      value.output_nanos_usd,
			cache_read_nanos_usd:  value.cache_read_nanos_usd,
			cache_write_nanos_usd: value.cache_write_nanos_usd,
		}
	}
}

impl From<pb::Cost> for Cost {
	fn from(value: pb::Cost) -> Self {
		Self {
			nanos_usd:             value.nanos_usd,
			estimated:             value.estimated,
			input_nanos_usd:       value.input_nanos_usd,
			output_nanos_usd:      value.output_nanos_usd,
			cache_read_nanos_usd:  value.cache_read_nanos_usd,
			cache_write_nanos_usd: value.cache_write_nanos_usd,
		}
	}
}

impl From<Unsupported> for pb::Unsupported {
	fn from(value: Unsupported) -> Self {
		let action = match value.action {
			UnsupportedAction::Dropped => pb::unsupported::Action::Dropped,
			UnsupportedAction::Emulated => pb::unsupported::Action::Emulated,
			UnsupportedAction::Clamped => pb::unsupported::Action::Clamped,
		};
		Self { what: value.what.into(), detail: value.detail.into(), action: action as i32 }
	}
}

impl TryFrom<pb::Unsupported> for Unsupported {
	type Error = ConvertError;

	fn try_from(value: pb::Unsupported) -> Result<Self, Self::Error> {
		let action = match value.action {
			x if x == pb::unsupported::Action::Unspecified as i32 => {
				return Err(ConvertError::Unspecified("Unsupported.Action"));
			},
			x if x == pb::unsupported::Action::Dropped as i32 => UnsupportedAction::Dropped,
			x if x == pb::unsupported::Action::Emulated as i32 => UnsupportedAction::Emulated,
			x if x == pb::unsupported::Action::Clamped as i32 => UnsupportedAction::Clamped,
			value => return Err(ConvertError::UnknownEnum { name: "Unsupported.Action", value }),
		};
		Ok(Self { what: value.what.into(), detail: value.detail.into(), action })
	}
}

const fn stop_to_proto(value: StopReason) -> i32 {
	match value {
		StopReason::EndTurn => pb::StopReason::StopEndTurn as i32,
		StopReason::ToolUse => pb::StopReason::StopToolUse as i32,
		StopReason::MaxTokens => pb::StopReason::StopMaxTokens as i32,
		StopReason::ContentFilter => pb::StopReason::StopContentFilter as i32,
	}
}

const fn stop_from_proto(value: i32) -> Result<StopReason, ConvertError> {
	match value {
		x if x == pb::StopReason::StopUnspecified as i32 => {
			Err(ConvertError::Unspecified("StopReason"))
		},
		x if x == pb::StopReason::StopEndTurn as i32 => Ok(StopReason::EndTurn),
		x if x == pb::StopReason::StopToolUse as i32 => Ok(StopReason::ToolUse),
		x if x == pb::StopReason::StopMaxTokens as i32 => Ok(StopReason::MaxTokens),
		x if x == pb::StopReason::StopContentFilter as i32 => Ok(StopReason::ContentFilter),
		value => Err(ConvertError::UnknownEnum { name: "StopReason", value }),
	}
}

impl From<Retryability> for i32 {
	fn from(value: Retryability) -> Self {
		match value {
			Retryability::Unspecified => pb::Retryability::Unspecified as Self,
			Retryability::Never => pb::Retryability::Never as Self,
			Retryability::SameRoute => pb::Retryability::SameRoute as Self,
			Retryability::AfterRepair => pb::Retryability::AfterRepair as Self,
			Retryability::AfterCredential => pb::Retryability::AfterCredential as Self,
			Retryability::AfterDelay => pb::Retryability::AfterDelay as Self,
		}
	}
}

const fn retryability_from_proto(value: i32) -> Result<Retryability, ConvertError> {
	match value {
		x if x == pb::Retryability::Unspecified as i32 => Ok(Retryability::Unspecified),
		x if x == pb::Retryability::Never as i32 => Ok(Retryability::Never),
		x if x == pb::Retryability::SameRoute as i32 => Ok(Retryability::SameRoute),
		x if x == pb::Retryability::AfterRepair as i32 => Ok(Retryability::AfterRepair),
		x if x == pb::Retryability::AfterCredential as i32 => Ok(Retryability::AfterCredential),
		x if x == pb::Retryability::AfterDelay as i32 => Ok(Retryability::AfterDelay),
		value => Err(ConvertError::UnknownEnum { name: "Retryability", value }),
	}
}

impl From<Diagnostic> for pb::Diagnostic {
	fn from(value: Diagnostic) -> Self {
		Self {
			provider:     value.provider.into(),
			model:        value.model.into(),
			attempt:      value.attempt,
			code:         value.code.into(),
			detail:       value.detail.into(),
			retryability: value.retryability.into(),
		}
	}
}

impl TryFrom<pb::Diagnostic> for Diagnostic {
	type Error = ConvertError;

	fn try_from(value: pb::Diagnostic) -> Result<Self, Self::Error> {
		Ok(Self {
			provider:     value.provider.into(),
			model:        value.model.into(),
			attempt:      value.attempt,
			code:         value.code.into(),
			detail:       value.detail.into(),
			retryability: retryability_from_proto(value.retryability)?,
		})
	}
}

impl From<ContextSnapshot> for pb::ContextSnapshot {
	fn from(value: ContextSnapshot) -> Self {
		Self {
			prompt_tokens:                  value.prompt_tokens,
			non_message_tokens:             value.non_message_tokens,
			history_rewrite_tokens_removed: value.history_rewrite_tokens_removed,
			last_message_timestamp_ms:      value.last_message_timestamp_ms,
		}
	}
}

impl From<pb::ContextSnapshot> for ContextSnapshot {
	fn from(value: pb::ContextSnapshot) -> Self {
		Self {
			prompt_tokens:                  value.prompt_tokens,
			non_message_tokens:             value.non_message_tokens,
			history_rewrite_tokens_removed: value.history_rewrite_tokens_removed,
			last_message_timestamp_ms:      value.last_message_timestamp_ms,
		}
	}
}

impl From<ChatOutcome> for pb::Outcome {
	fn from(value: ChatOutcome) -> Self {
		Self {
			output:            value.output.into_iter().map(Into::into).collect(),
			stop:              stop_to_proto(value.stop),
			usage:             value.usage.map(Into::into),
			cost:              value.cost.map(Into::into),
			unsupported:       value.unsupported.into_iter().map(Into::into).collect(),
			revision:          value.revision.map(Into::into),
			provider:          value.provider.into(),
			model:             value.model.into(),
			diagnostics:       value.diagnostics.into_iter().map(Into::into).collect(),
			upstream_provider: value.upstream_provider.map(Into::into),
			duration_ms:       value.duration_ms,
			ttft_ms:           value.ttft_ms,
			context_snapshot:  value.context_snapshot.map(Into::into),
			props:             props_to_proto(value.props),
		}
	}
}

impl TryFrom<pb::Outcome> for ChatOutcome {
	type Error = ConvertError;

	fn try_from(value: pb::Outcome) -> Result<Self, Self::Error> {
		Ok(Self {
			output:            value
				.output
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
			stop:              stop_from_proto(value.stop)?,
			usage:             value.usage.map(TryInto::try_into).transpose()?,
			cost:              value.cost.map(Into::into),
			unsupported:       value
				.unsupported
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
			revision:          value.revision.map(Into::into),
			provider:          value.provider.into(),
			model:             value.model.into(),
			diagnostics:       value
				.diagnostics
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
			upstream_provider: value.upstream_provider.map(Into::into),
			duration_ms:       value.duration_ms,
			ttft_ms:           value.ttft_ms,
			context_snapshot:  value.context_snapshot.map(Into::into),
			props:             props_from_proto(value.props)?,
		})
	}
}

impl From<TurnErrorKind> for i32 {
	fn from(value: TurnErrorKind) -> Self {
		match value {
			TurnErrorKind::Conflict => pb::turn_error::Kind::Conflict as Self,
			TurnErrorKind::NeedFull => pb::turn_error::Kind::NeedFull as Self,
			TurnErrorKind::Unsupported => pb::turn_error::Kind::Unsupported as Self,
			TurnErrorKind::Auth => pb::turn_error::Kind::Auth as Self,
			TurnErrorKind::RateLimited => pb::turn_error::Kind::RateLimited as Self,
			TurnErrorKind::Upstream => pb::turn_error::Kind::Upstream as Self,
			TurnErrorKind::Overloaded => pb::turn_error::Kind::Overloaded as Self,
			TurnErrorKind::InvokeTimeout => pb::turn_error::Kind::InvokeTimeout as Self,
		}
	}
}

const fn error_kind_from_proto(value: i32) -> Result<TurnErrorKind, ConvertError> {
	match value {
		x if x == pb::turn_error::Kind::Unspecified as i32 => {
			Err(ConvertError::Unspecified("TurnError.Kind"))
		},
		x if x == pb::turn_error::Kind::Conflict as i32 => Ok(TurnErrorKind::Conflict),
		x if x == pb::turn_error::Kind::NeedFull as i32 => Ok(TurnErrorKind::NeedFull),
		x if x == pb::turn_error::Kind::Unsupported as i32 => Ok(TurnErrorKind::Unsupported),
		x if x == pb::turn_error::Kind::Auth as i32 => Ok(TurnErrorKind::Auth),
		x if x == pb::turn_error::Kind::RateLimited as i32 => Ok(TurnErrorKind::RateLimited),
		x if x == pb::turn_error::Kind::Upstream as i32 => Ok(TurnErrorKind::Upstream),
		x if x == pb::turn_error::Kind::Overloaded as i32 => Ok(TurnErrorKind::Overloaded),
		x if x == pb::turn_error::Kind::InvokeTimeout as i32 => Ok(TurnErrorKind::InvokeTimeout),
		value => Err(ConvertError::UnknownEnum { name: "TurnError.Kind", value }),
	}
}

impl From<TurnError> for pb::TurnError {
	fn from(value: TurnError) -> Self {
		Self {
			kind:           value.kind.into(),
			detail:         value.detail.into(),
			actual:         value.actual.map(Into::into),
			unsupported:    value.unsupported.into_iter().map(Into::into).collect(),
			retry_after_ms: value.retry_after_ms,
			diagnostics:    value.diagnostics.into_iter().map(Into::into).collect(),
			error_id:       value.error_id,
		}
	}
}

impl TryFrom<pb::TurnError> for TurnError {
	type Error = ConvertError;

	fn try_from(value: pb::TurnError) -> Result<Self, Self::Error> {
		Ok(Self {
			kind:           error_kind_from_proto(value.kind)?,
			detail:         value.detail.into(),
			actual:         value.actual.map(Into::into),
			unsupported:    value
				.unsupported
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
			retry_after_ms: value.retry_after_ms,
			diagnostics:    value
				.diagnostics
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
			error_id:       value.error_id,
		})
	}
}

impl From<Invoke> for pb::Invoke {
	fn from(value: Invoke) -> Self {
		Self {
			invocation_id: value.invocation_id.into(),
			name:          value.name.into(),
			tool_call:     value.tool_call.map(Into::into),
			vendor:        value.vendor,
			timeout_ms:    value.timeout_ms,
			props:         props_to_proto(value.props),
		}
	}
}

impl TryFrom<pb::Invoke> for Invoke {
	type Error = ConvertError;

	fn try_from(value: pb::Invoke) -> Result<Self, Self::Error> {
		Ok(Self {
			invocation_id: value.invocation_id.into(),
			name:          value.name.into(),
			tool_call:     value.tool_call.map(TryInto::try_into).transpose()?,
			vendor:        value.vendor,
			timeout_ms:    value.timeout_ms,
			props:         props_from_proto(value.props)?,
		})
	}
}

impl From<InvokeInput> for pb::InvokeInput {
	fn from(value: InvokeInput) -> Self {
		use pb::invoke_input::{self, Payload};
		let payload = match value.payload {
			InvokePayload::Chunk(value) => {
				let channel = match value.channel {
					InvokeChannel::Stdout => invoke_input::chunk::Channel::Stdout,
					InvokeChannel::Stderr => invoke_input::chunk::Channel::Stderr,
					InvokeChannel::Progress => invoke_input::chunk::Channel::Progress,
				};
				Payload::Chunk(invoke_input::Chunk { channel: channel as i32, data: value.data })
			},
			InvokePayload::Vendor(value) => Payload::Vendor(value),
		};
		Self { invocation_id: value.invocation_id.into(), payload: Some(payload) }
	}
}

impl TryFrom<pb::InvokeInput> for InvokeInput {
	type Error = ConvertError;

	fn try_from(value: pb::InvokeInput) -> Result<Self, Self::Error> {
		use pb::invoke_input::Payload;
		let payload = match value
			.payload
			.ok_or_else(|| missing("InvokeInput.payload"))?
		{
			Payload::Vendor(value) => InvokePayload::Vendor(value),
			Payload::Chunk(value) => {
				let channel = match value.channel {
					x if x == pb::invoke_input::chunk::Channel::Unspecified as i32 => {
						return Err(ConvertError::Unspecified("InvokeInput.Chunk.Channel"));
					},
					x if x == pb::invoke_input::chunk::Channel::Stdout as i32 => InvokeChannel::Stdout,
					x if x == pb::invoke_input::chunk::Channel::Stderr as i32 => InvokeChannel::Stderr,
					x if x == pb::invoke_input::chunk::Channel::Progress as i32 => {
						InvokeChannel::Progress
					},
					value => {
						return Err(ConvertError::UnknownEnum {
							name: "InvokeInput.Chunk.Channel",
							value,
						});
					},
				};
				InvokePayload::Chunk(InvokeChunk { channel, data: value.data })
			},
		};
		Ok(Self { invocation_id: value.invocation_id.into(), payload })
	}
}

impl From<InvokeComplete> for pb::InvokeComplete {
	fn from(value: InvokeComplete) -> Self {
		Self {
			invocation_id: value.invocation_id.into(),
			tool_result:   value.tool_result.map(Into::into),
			status:        value.status.map(Into::into),
			vendor:        value.vendor,
			props:         props_to_proto(value.props),
		}
	}
}

impl TryFrom<pb::InvokeComplete> for InvokeComplete {
	type Error = ConvertError;

	fn try_from(value: pb::InvokeComplete) -> Result<Self, Self::Error> {
		Ok(Self {
			invocation_id: value.invocation_id.into(),
			tool_result:   value.tool_result.map(TryInto::try_into).transpose()?,
			status:        value.status.map(TryInto::try_into).transpose()?,
			vendor:        value.vendor,
			props:         props_from_proto(value.props)?,
		})
	}
}

impl From<ExecStatus> for pb::ExecStatus {
	fn from(value: ExecStatus) -> Self {
		let outcome = match value.outcome {
			ExecOutcome::Exited => pb::exec_status::Outcome::Exited,
			ExecOutcome::Failed => pb::exec_status::Outcome::Failed,
			ExecOutcome::Rejected => pb::exec_status::Outcome::Rejected,
			ExecOutcome::Denied => pb::exec_status::Outcome::Denied,
			ExecOutcome::Timeout => pb::exec_status::Outcome::Timeout,
			ExecOutcome::Cancelled => pb::exec_status::Outcome::Cancelled,
		};
		Self {
			outcome:                 outcome as i32,
			exit_code:               value.exit_code,
			signal:                  value.signal.into(),
			reason:                  value.reason.into(),
			cwd:                     value.cwd.into(),
			aborted:                 value.aborted,
			output_location:         value.output_location.into(),
			local_execution_time_ms: value.local_execution_time_ms,
			is_readonly:             value.is_readonly,
			command_timeout_ms:      value.command_timeout_ms,
		}
	}
}

impl TryFrom<pb::ExecStatus> for ExecStatus {
	type Error = ConvertError;

	fn try_from(value: pb::ExecStatus) -> Result<Self, Self::Error> {
		let outcome = match value.outcome {
			x if x == pb::exec_status::Outcome::Unspecified as i32 => {
				return Err(ConvertError::Unspecified("ExecStatus.Outcome"));
			},
			x if x == pb::exec_status::Outcome::Exited as i32 => ExecOutcome::Exited,
			x if x == pb::exec_status::Outcome::Failed as i32 => ExecOutcome::Failed,
			x if x == pb::exec_status::Outcome::Rejected as i32 => ExecOutcome::Rejected,
			x if x == pb::exec_status::Outcome::Denied as i32 => ExecOutcome::Denied,
			x if x == pb::exec_status::Outcome::Timeout as i32 => ExecOutcome::Timeout,
			x if x == pb::exec_status::Outcome::Cancelled as i32 => ExecOutcome::Cancelled,
			value => return Err(ConvertError::UnknownEnum { name: "ExecStatus.Outcome", value }),
		};
		Ok(Self {
			outcome,
			exit_code: value.exit_code,
			signal: value.signal.into(),
			reason: value.reason.into(),
			cwd: value.cwd.into(),
			aborted: value.aborted,
			output_location: value.output_location.into(),
			local_execution_time_ms: value.local_execution_time_ms,
			is_readonly: value.is_readonly,
			command_timeout_ms: value.command_timeout_ms,
		})
	}
}

const fn stream_kind_to_proto(value: StreamPartKind) -> i32 {
	match value {
		StreamPartKind::Text => pb::part_start::Kind::Text as i32,
		StreamPartKind::Thinking => pb::part_start::Kind::Thinking as i32,
		StreamPartKind::ToolCall => pb::part_start::Kind::ToolCall as i32,
	}
}

const fn stream_kind_from_proto(value: i32) -> Result<StreamPartKind, ConvertError> {
	match value {
		x if x == pb::part_start::Kind::Unspecified as i32 => {
			Err(ConvertError::Unspecified("PartStart.Kind"))
		},
		x if x == pb::part_start::Kind::Text as i32 => Ok(StreamPartKind::Text),
		x if x == pb::part_start::Kind::Thinking as i32 => Ok(StreamPartKind::Thinking),
		x if x == pb::part_start::Kind::ToolCall as i32 => Ok(StreamPartKind::ToolCall),
		value => Err(ConvertError::UnknownEnum { name: "PartStart.Kind", value }),
	}
}

impl From<TurnEvent> for pb::TurnEvent {
	fn from(value: TurnEvent) -> Self {
		use pb::turn_event::Event;
		let event = match value {
			TurnEvent::Accepted { replay } => Event::Accepted(pb::Accepted { replay }),
			TurnEvent::Attempt { number, reason } => {
				Event::Attempt(pb::Attempt { number, reason: reason.into() })
			},
			TurnEvent::PartStart { index, kind, tool_call_id, tool_name } => {
				Event::PartStart(pb::PartStart {
					index,
					kind: stream_kind_to_proto(kind),
					tool_call_id: tool_call_id.into(),
					tool_name: tool_name.into(),
				})
			},
			TurnEvent::PartDelta { index, chunk } => Event::PartDelta(pb::PartDelta { index, chunk }),
			TurnEvent::PartEnd { index, signature } => {
				Event::PartEnd(pb::PartEnd { index, signature })
			},
			TurnEvent::Invoke(value) => Event::Invoke(value.into()),
			TurnEvent::InvokeCancel { invocation_id } => {
				Event::InvokeCancel(pb::InvokeCancel { invocation_id: invocation_id.into() })
			},
			TurnEvent::Outcome(value) => Event::Outcome(value.into()),
			TurnEvent::Error(value) => Event::Error(value.into()),
		};
		Self { event: Some(event) }
	}
}

impl TryFrom<pb::TurnEvent> for TurnEvent {
	type Error = ConvertError;

	fn try_from(value: pb::TurnEvent) -> Result<Self, ConvertError> {
		use pb::turn_event::Event;
		match value.event.ok_or_else(|| missing("TurnEvent.event"))? {
			Event::Accepted(value) => Ok(Self::Accepted { replay: value.replay }),
			Event::Attempt(value) => {
				Ok(Self::Attempt { number: value.number, reason: value.reason.into() })
			},
			Event::PartStart(value) => Ok(Self::PartStart {
				index:        value.index,
				kind:         stream_kind_from_proto(value.kind)?,
				tool_call_id: value.tool_call_id.into(),
				tool_name:    value.tool_name.into(),
			}),
			Event::PartDelta(value) => Ok(Self::PartDelta { index: value.index, chunk: value.chunk }),
			Event::PartEnd(value) => {
				Ok(Self::PartEnd { index: value.index, signature: value.signature })
			},
			Event::Invoke(value) => Ok(Self::Invoke(value.try_into()?)),
			Event::InvokeCancel(value) => {
				Ok(Self::InvokeCancel { invocation_id: value.invocation_id.into() })
			},
			Event::Outcome(value) => Ok(Self::Outcome(value.try_into()?)),
			Event::Error(value) => Ok(Self::Error(value.try_into()?)),
		}
	}
}

impl From<CountRequest> for pb::CountTokensRequest {
	fn from(value: CountRequest) -> Self {
		let input = match value.input {
			CountInput::Context(value) => pb::count_tokens_request::Input::Context(value.into()),
			CountInput::Thread(value) => pb::count_tokens_request::Input::Thread(value.into()),
		};
		Self {
			model: value.model.into(),
			input: Some(input),
			tools: value.tools.into_iter().map(Into::into).collect(),
		}
	}
}

impl TryFrom<pb::CountTokensRequest> for CountRequest {
	type Error = ConvertError;

	fn try_from(value: pb::CountTokensRequest) -> Result<Self, Self::Error> {
		let input = match value
			.input
			.ok_or_else(|| missing("CountTokensRequest.input"))?
		{
			pb::count_tokens_request::Input::Context(value) => CountInput::Context(value.try_into()?),
			pb::count_tokens_request::Input::Thread(value) => CountInput::Thread(value.try_into()?),
		};
		Ok(Self {
			model: value.model.into(),
			input,
			tools: value.tools.into_iter().map(Into::into).collect(),
		})
	}
}

impl From<CountResponse> for pb::CountTokensResponse {
	fn from(value: CountResponse) -> Self {
		let accuracy = match value.accuracy {
			Accuracy::Exact => pb::usage::Accuracy::Exact,
			Accuracy::Estimated => pb::usage::Accuracy::Estimated,
			Accuracy::Mixed => pb::usage::Accuracy::Mixed,
		};
		Self { tokens: value.tokens, accuracy: accuracy as i32 }
	}
}

impl TryFrom<pb::CountTokensResponse> for CountResponse {
	type Error = ConvertError;

	fn try_from(value: pb::CountTokensResponse) -> Result<Self, Self::Error> {
		let accuracy = match value.accuracy {
			x if x == pb::usage::Accuracy::Unspecified as i32 => {
				return Err(ConvertError::Unspecified("Usage.Accuracy"));
			},
			x if x == pb::usage::Accuracy::Exact as i32 => Accuracy::Exact,
			x if x == pb::usage::Accuracy::Estimated as i32 => Accuracy::Estimated,
			x if x == pb::usage::Accuracy::Mixed as i32 => Accuracy::Mixed,
			value => return Err(ConvertError::UnknownEnum { name: "Usage.Accuracy", value }),
		};
		Ok(Self { tokens: value.tokens, accuracy })
	}
}

impl From<EmbedRequest> for pb::EmbedRequest {
	fn from(value: EmbedRequest) -> Self {
		Self {
			model:      value.model.into(),
			texts:      value.texts.into_iter().map(Into::into).collect(),
			dimensions: value.dimensions,
			props:      props_to_proto(value.props),
		}
	}
}

impl TryFrom<pb::EmbedRequest> for EmbedRequest {
	type Error = ConvertError;

	fn try_from(value: pb::EmbedRequest) -> Result<Self, Self::Error> {
		Ok(Self {
			model:      value.model.into(),
			texts:      value.texts.into_iter().map(Into::into).collect(),
			dimensions: value.dimensions,
			props:      props_from_proto(value.props)?,
		})
	}
}

impl From<EmbedResponse> for pb::EmbedResponse {
	fn from(value: EmbedResponse) -> Self {
		Self {
			vectors: value
				.vectors
				.into_iter()
				.map(|value| pb::embed_response::Vector { values: value.values })
				.collect(),
			usage:   value.usage.map(Into::into),
		}
	}
}

impl TryFrom<pb::EmbedResponse> for EmbedResponse {
	type Error = ConvertError;

	fn try_from(value: pb::EmbedResponse) -> Result<Self, Self::Error> {
		Ok(Self {
			vectors: value
				.vectors
				.into_iter()
				.map(|value| EmbeddingVector { values: value.values })
				.collect(),
			usage:   value.usage.map(TryInto::try_into).transpose()?,
		})
	}
}

const fn aspect_to_proto(value: Option<AspectRatio>) -> i32 {
	(match value {
		None => pb::AspectRatio::Unspecified,
		Some(AspectRatio::Square) => pb::AspectRatio::AspectRatio11,
		Some(AspectRatio::Wide16x9) => pb::AspectRatio::AspectRatio169,
		Some(AspectRatio::Tall9x16) => pb::AspectRatio::AspectRatio916,
		Some(AspectRatio::Landscape4x3) => pb::AspectRatio::AspectRatio43,
		Some(AspectRatio::Portrait3x4) => pb::AspectRatio::AspectRatio34,
		Some(AspectRatio::Landscape3x2) => pb::AspectRatio::AspectRatio32,
		Some(AspectRatio::Portrait2x3) => pb::AspectRatio::AspectRatio23,
		Some(AspectRatio::Ultrawide21x9) => pb::AspectRatio::AspectRatio219,
	}) as i32
}

const fn aspect_from_proto(value: i32) -> Result<Option<AspectRatio>, ConvertError> {
	match value {
		x if x == pb::AspectRatio::Unspecified as i32 => Ok(None),
		x if x == pb::AspectRatio::AspectRatio11 as i32 => Ok(Some(AspectRatio::Square)),
		x if x == pb::AspectRatio::AspectRatio169 as i32 => Ok(Some(AspectRatio::Wide16x9)),
		x if x == pb::AspectRatio::AspectRatio916 as i32 => Ok(Some(AspectRatio::Tall9x16)),
		x if x == pb::AspectRatio::AspectRatio43 as i32 => Ok(Some(AspectRatio::Landscape4x3)),
		x if x == pb::AspectRatio::AspectRatio34 as i32 => Ok(Some(AspectRatio::Portrait3x4)),
		x if x == pb::AspectRatio::AspectRatio32 as i32 => Ok(Some(AspectRatio::Landscape3x2)),
		x if x == pb::AspectRatio::AspectRatio23 as i32 => Ok(Some(AspectRatio::Portrait2x3)),
		x if x == pb::AspectRatio::AspectRatio219 as i32 => Ok(Some(AspectRatio::Ultrawide21x9)),
		value => Err(ConvertError::UnknownEnum { name: "AspectRatio", value }),
	}
}

impl From<GenerateImageRequest> for pb::GenerateImageRequest {
	fn from(value: GenerateImageRequest) -> Self {
		let quality = match value.quality {
			None => pb::generate_image_request::Quality::Unspecified,
			Some(ImageQuality::Low) => pb::generate_image_request::Quality::Low,
			Some(ImageQuality::Medium) => pb::generate_image_request::Quality::Medium,
			Some(ImageQuality::High) => pb::generate_image_request::Quality::High,
		};
		let format = match value.format {
			None => pb::generate_image_request::Format::Unspecified,
			Some(ImageFormat::Png) => pb::generate_image_request::Format::Png,
			Some(ImageFormat::Webp) => pb::generate_image_request::Format::Webp,
			Some(ImageFormat::Jpeg) => pb::generate_image_request::Format::Jpeg,
			Some(ImageFormat::Svg) => pb::generate_image_request::Format::Svg,
		};
		let background = match value.background {
			None => pb::generate_image_request::Background::Unspecified,
			Some(ImageBackground::Opaque) => pb::generate_image_request::Background::Opaque,
			Some(ImageBackground::Transparent) => pb::generate_image_request::Background::Transparent,
		};
		Self {
			model:        value.model.into(),
			prompt:       value.prompt.into(),
			n:            value.n,
			aspect_ratio: aspect_to_proto(value.aspect_ratio),
			size:         value
				.size
				.map(|value| pb::generate_image_request::ImageSize {
					width:  value.width,
					height: value.height,
				}),
			quality:      quality as i32,
			format:       format as i32,
			background:   background as i32,
			compression:  value.compression,
			seed:         value.seed,
			input_images: value.input_images.into_iter().map(Into::into).collect(),
			props:        props_to_proto(value.props),
		}
	}
}

impl TryFrom<pb::GenerateImageRequest> for GenerateImageRequest {
	type Error = ConvertError;

	fn try_from(value: pb::GenerateImageRequest) -> Result<Self, Self::Error> {
		let quality = match value.quality {
			x if x == pb::generate_image_request::Quality::Unspecified as i32 => None,
			x if x == pb::generate_image_request::Quality::Low as i32 => Some(ImageQuality::Low),
			x if x == pb::generate_image_request::Quality::Medium as i32 => Some(ImageQuality::Medium),
			x if x == pb::generate_image_request::Quality::High as i32 => Some(ImageQuality::High),
			value => {
				return Err(ConvertError::UnknownEnum { name: "GenerateImageRequest.Quality", value });
			},
		};
		let format = match value.format {
			x if x == pb::generate_image_request::Format::Unspecified as i32 => None,
			x if x == pb::generate_image_request::Format::Png as i32 => Some(ImageFormat::Png),
			x if x == pb::generate_image_request::Format::Webp as i32 => Some(ImageFormat::Webp),
			x if x == pb::generate_image_request::Format::Jpeg as i32 => Some(ImageFormat::Jpeg),
			x if x == pb::generate_image_request::Format::Svg as i32 => Some(ImageFormat::Svg),
			value => {
				return Err(ConvertError::UnknownEnum { name: "GenerateImageRequest.Format", value });
			},
		};
		let background = match value.background {
			x if x == pb::generate_image_request::Background::Unspecified as i32 => None,
			x if x == pb::generate_image_request::Background::Opaque as i32 => {
				Some(ImageBackground::Opaque)
			},
			x if x == pb::generate_image_request::Background::Transparent as i32 => {
				Some(ImageBackground::Transparent)
			},
			value => {
				return Err(ConvertError::UnknownEnum {
					name: "GenerateImageRequest.Background",
					value,
				});
			},
		};
		Ok(Self {
			model: value.model.into(),
			prompt: value.prompt.into(),
			n: value.n,
			aspect_ratio: aspect_from_proto(value.aspect_ratio)?,
			size: value
				.size
				.map(|value| ImageSize { width: value.width, height: value.height }),
			quality,
			format,
			background,
			compression: value.compression,
			seed: value.seed,
			input_images: value
				.input_images
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
			props: props_from_proto(value.props)?,
		})
	}
}

impl From<ImageEvent> for pb::ImageEvent {
	fn from(value: ImageEvent) -> Self {
		use pb::image_event::Event;
		let event = match value {
			ImageEvent::Partial(value) => Event::Partial(pb::image_event::Partial {
				index:   value.index,
				preview: Some(value.preview.into()),
			}),
			ImageEvent::Done(value) => Event::Done(pb::image_event::Done {
				images:         value.images.into_iter().map(Into::into).collect(),
				revised_prompt: value.revised_prompt.into(),
				text:           value.text.into(),
				usage:          value.usage.map(Into::into),
				cost:           value.cost.map(Into::into),
				unsupported:    value.unsupported.into_iter().map(Into::into).collect(),
				props:          props_to_proto(value.props),
			}),
		};
		Self { event: Some(event) }
	}
}

impl TryFrom<pb::ImageEvent> for ImageEvent {
	type Error = ConvertError;

	fn try_from(value: pb::ImageEvent) -> Result<Self, Self::Error> {
		match value.event.ok_or_else(|| missing("ImageEvent.event"))? {
			pb::image_event::Event::Partial(value) => Ok(Self::Partial(ImagePartial {
				index:   value.index,
				preview: value
					.preview
					.ok_or_else(|| missing("ImageEvent.Partial.preview"))?
					.try_into()?,
			})),
			pb::image_event::Event::Done(value) => Ok(Self::Done(ImageDone {
				images:         value
					.images
					.into_iter()
					.map(TryInto::try_into)
					.collect::<Result<_, _>>()?,
				revised_prompt: value.revised_prompt.into(),
				text:           value.text.into(),
				usage:          value.usage.map(TryInto::try_into).transpose()?,
				cost:           value.cost.map(Into::into),
				unsupported:    value
					.unsupported
					.into_iter()
					.map(TryInto::try_into)
					.collect::<Result<_, _>>()?,
				props:          props_from_proto(value.props)?,
			})),
		}
	}
}

const fn audio_encoding_to_proto(value: AudioEncoding) -> i32 {
	(match value {
		AudioEncoding::Mp3 => pb::AudioEncoding::Mp3,
		AudioEncoding::Pcm16 => pb::AudioEncoding::Pcm16,
		AudioEncoding::Wav => pb::AudioEncoding::Wav,
		AudioEncoding::Opus => pb::AudioEncoding::Opus,
		AudioEncoding::Aac => pb::AudioEncoding::Aac,
		AudioEncoding::Flac => pb::AudioEncoding::Flac,
	}) as i32
}

const fn audio_encoding_from_proto(value: i32) -> Result<AudioEncoding, ConvertError> {
	match value {
		x if x == pb::AudioEncoding::Unspecified as i32 => {
			Err(ConvertError::Unspecified("AudioEncoding"))
		},
		x if x == pb::AudioEncoding::Mp3 as i32 => Ok(AudioEncoding::Mp3),
		x if x == pb::AudioEncoding::Pcm16 as i32 => Ok(AudioEncoding::Pcm16),
		x if x == pb::AudioEncoding::Wav as i32 => Ok(AudioEncoding::Wav),
		x if x == pb::AudioEncoding::Opus as i32 => Ok(AudioEncoding::Opus),
		x if x == pb::AudioEncoding::Aac as i32 => Ok(AudioEncoding::Aac),
		x if x == pb::AudioEncoding::Flac as i32 => Ok(AudioEncoding::Flac),
		value => Err(ConvertError::UnknownEnum { name: "AudioEncoding", value }),
	}
}

impl From<SpeakRequest> for pb::SpeakRequest {
	fn from(value: SpeakRequest) -> Self {
		Self {
			model:          value.model.into(),
			text:           value.text.into(),
			voice:          value.voice.into(),
			encoding:       audio_encoding_to_proto(value.encoding),
			sample_rate_hz: value.sample_rate_hz,
			speed:          value.speed,
			instructions:   value.instructions.into(),
			clone:          value.clone.map(|value| pb::speak_request::Clone {
				reference:  Some(value.reference.into()),
				transcript: value.transcript.into(),
			}),
			props:          props_to_proto(value.props),
		}
	}
}

impl TryFrom<pb::SpeakRequest> for SpeakRequest {
	type Error = ConvertError;

	fn try_from(value: pb::SpeakRequest) -> Result<Self, Self::Error> {
		Ok(Self {
			model:          value.model.into(),
			text:           value.text.into(),
			voice:          value.voice.into(),
			encoding:       audio_encoding_from_proto(value.encoding)?,
			sample_rate_hz: value.sample_rate_hz,
			speed:          value.speed,
			instructions:   value.instructions.into(),
			clone:          value
				.clone
				.map(|value| -> Result<VoiceClone, ConvertError> {
					Ok(VoiceClone {
						reference:  value
							.reference
							.ok_or_else(|| missing("SpeakRequest.Clone.reference"))?
							.try_into()?,
						transcript: value.transcript.into(),
					})
				})
				.transpose()?,
			props:          props_from_proto(value.props)?,
		})
	}
}

impl From<SpeakEvent> for pb::SpeakEvent {
	fn from(value: SpeakEvent) -> Self {
		use pb::speak_event::Event;
		let event = match value {
			SpeakEvent::Chunk(value) => Event::Chunk(pb::speak_event::Chunk {
				audio:            value.audio,
				transcript_delta: value.transcript_delta.into(),
			}),
			SpeakEvent::Done(value) => Event::Done(pb::speak_event::Done {
				audio:       Some(value.audio.into()),
				duration_ms: value.duration_ms,
				usage:       value.usage.map(Into::into),
				cost:        value.cost.map(Into::into),
				unsupported: value.unsupported.into_iter().map(Into::into).collect(),
				props:       props_to_proto(value.props),
			}),
		};
		Self { event: Some(event) }
	}
}

impl TryFrom<pb::SpeakEvent> for SpeakEvent {
	type Error = ConvertError;

	fn try_from(value: pb::SpeakEvent) -> Result<Self, Self::Error> {
		match value.event.ok_or_else(|| missing("SpeakEvent.event"))? {
			pb::speak_event::Event::Chunk(value) => Ok(Self::Chunk(SpeakChunk {
				audio:            value.audio,
				transcript_delta: value.transcript_delta.into(),
			})),
			pb::speak_event::Event::Done(value) => Ok(Self::Done(SpeakDone {
				audio:       value
					.audio
					.ok_or_else(|| missing("SpeakEvent.Done.audio"))?
					.try_into()?,
				duration_ms: value.duration_ms,
				usage:       value.usage.map(TryInto::try_into).transpose()?,
				cost:        value.cost.map(Into::into),
				unsupported: value
					.unsupported
					.into_iter()
					.map(TryInto::try_into)
					.collect::<Result<_, _>>()?,
				props:       props_from_proto(value.props)?,
			})),
		}
	}
}

impl From<TranscribeRequest> for pb::TranscribeRequest {
	fn from(value: TranscribeRequest) -> Self {
		Self {
			model:         value.model.into(),
			audio:         Some(value.audio.into()),
			language:      value.language.into(),
			prompt:        value.prompt.into(),
			translate:     value.translate,
			granularities: value
				.granularities
				.into_iter()
				.map(|value| {
					(match value {
						TranscriptionGranularity::Segment => pb::transcribe_request::Granularity::Segment,
						TranscriptionGranularity::Word => pb::transcribe_request::Granularity::Word,
					}) as i32
				})
				.collect(),
			diarize:       value.diarize,
			temperature:   value.temperature,
			props:         props_to_proto(value.props),
		}
	}
}

impl TryFrom<pb::TranscribeRequest> for TranscribeRequest {
	type Error = ConvertError;

	fn try_from(value: pb::TranscribeRequest) -> Result<Self, Self::Error> {
		let granularities = value
			.granularities
			.into_iter()
			.map(|value| match value {
				x if x == pb::transcribe_request::Granularity::Unspecified as i32 => {
					Err(ConvertError::Unspecified("TranscribeRequest.Granularity"))
				},
				x if x == pb::transcribe_request::Granularity::Segment as i32 => {
					Ok(TranscriptionGranularity::Segment)
				},
				x if x == pb::transcribe_request::Granularity::Word as i32 => {
					Ok(TranscriptionGranularity::Word)
				},
				value => {
					Err(ConvertError::UnknownEnum { name: "TranscribeRequest.Granularity", value })
				},
			})
			.collect::<Result<_, _>>()?;
		Ok(Self {
			model: value.model.into(),
			audio: value
				.audio
				.ok_or_else(|| missing("TranscribeRequest.audio"))?
				.try_into()?,
			language: value.language.into(),
			prompt: value.prompt.into(),
			translate: value.translate,
			granularities,
			diarize: value.diarize,
			temperature: value.temperature,
			props: props_from_proto(value.props)?,
		})
	}
}

impl From<TranscribeResponse> for pb::TranscribeResponse {
	fn from(value: TranscribeResponse) -> Self {
		Self {
			text:        value.text.into(),
			language:    value.language.into(),
			duration_ms: value.duration_ms,
			segments:    value
				.segments
				.into_iter()
				.map(|value| pb::transcribe_response::Segment {
					start_ms:   value.start_ms,
					end_ms:     value.end_ms,
					text:       value.text.into(),
					speaker:    value.speaker,
					confidence: value.confidence,
				})
				.collect(),
			words:       value
				.words
				.into_iter()
				.map(|value| pb::transcribe_response::Word {
					start_ms: value.start_ms,
					end_ms:   value.end_ms,
					word:     value.word.into(),
					speaker:  value.speaker,
				})
				.collect(),
			usage:       value.usage.map(Into::into),
			cost:        value.cost.map(Into::into),
			unsupported: value.unsupported.into_iter().map(Into::into).collect(),
			props:       props_to_proto(value.props),
		}
	}
}

impl TryFrom<pb::TranscribeResponse> for TranscribeResponse {
	type Error = ConvertError;

	fn try_from(value: pb::TranscribeResponse) -> Result<Self, Self::Error> {
		Ok(Self {
			text:        value.text.into(),
			language:    value.language.into(),
			duration_ms: value.duration_ms,
			segments:    value
				.segments
				.into_iter()
				.map(|value| TranscriptSegment {
					start_ms:   value.start_ms,
					end_ms:     value.end_ms,
					text:       value.text.into(),
					speaker:    value.speaker,
					confidence: value.confidence,
				})
				.collect(),
			words:       value
				.words
				.into_iter()
				.map(|value| TranscriptWord {
					start_ms: value.start_ms,
					end_ms:   value.end_ms,
					word:     value.word.into(),
					speaker:  value.speaker,
				})
				.collect(),
			usage:       value.usage.map(TryInto::try_into).transpose()?,
			cost:        value.cost.map(Into::into),
			unsupported: value
				.unsupported
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
			props:       props_from_proto(value.props)?,
		})
	}
}

impl From<GenerateVideoRequest> for pb::GenerateVideoRequest {
	fn from(value: GenerateVideoRequest) -> Self {
		let resolution = match value.resolution {
			None => pb::generate_video_request::Resolution::Unspecified,
			Some(VideoResolution::P480) => pb::generate_video_request::Resolution::Resolution480p,
			Some(VideoResolution::P720) => pb::generate_video_request::Resolution::Resolution720p,
			Some(VideoResolution::P1080) => pb::generate_video_request::Resolution::Resolution1080p,
			Some(VideoResolution::K4) => pb::generate_video_request::Resolution::Resolution4k,
		};
		Self {
			model:            value.model.into(),
			prompt:           value.prompt.into(),
			duration_seconds: value.duration_seconds,
			aspect_ratio:     aspect_to_proto(value.aspect_ratio),
			resolution:       resolution as i32,
			seed:             value.seed,
			audio:            value.audio,
			start_frame:      value.start_frame.map(Into::into),
			end_frame:        value.end_frame.map(Into::into),
			references:       value.references.into_iter().map(Into::into).collect(),
			props:            props_to_proto(value.props),
		}
	}
}

impl TryFrom<pb::GenerateVideoRequest> for GenerateVideoRequest {
	type Error = ConvertError;

	fn try_from(value: pb::GenerateVideoRequest) -> Result<Self, Self::Error> {
		let resolution = match value.resolution {
			x if x == pb::generate_video_request::Resolution::Unspecified as i32 => None,
			x if x == pb::generate_video_request::Resolution::Resolution480p as i32 => {
				Some(VideoResolution::P480)
			},
			x if x == pb::generate_video_request::Resolution::Resolution720p as i32 => {
				Some(VideoResolution::P720)
			},
			x if x == pb::generate_video_request::Resolution::Resolution1080p as i32 => {
				Some(VideoResolution::P1080)
			},
			x if x == pb::generate_video_request::Resolution::Resolution4k as i32 => {
				Some(VideoResolution::K4)
			},
			value => {
				return Err(ConvertError::UnknownEnum {
					name: "GenerateVideoRequest.Resolution",
					value,
				});
			},
		};
		Ok(Self {
			model: value.model.into(),
			prompt: value.prompt.into(),
			duration_seconds: value.duration_seconds,
			aspect_ratio: aspect_from_proto(value.aspect_ratio)?,
			resolution,
			seed: value.seed,
			audio: value.audio,
			start_frame: value.start_frame.map(TryInto::try_into).transpose()?,
			end_frame: value.end_frame.map(TryInto::try_into).transpose()?,
			references: value
				.references
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
			props: props_from_proto(value.props)?,
		})
	}
}

impl From<GenerationStatus> for pb::GenerationStatus {
	fn from(value: GenerationStatus) -> Self {
		let state = match value.state {
			GenerationState::Queued => pb::generation_status::State::Queued,
			GenerationState::Running => pb::generation_status::State::Running,
			GenerationState::Completed => pb::generation_status::State::Completed,
			GenerationState::Failed => pb::generation_status::State::Failed,
			GenerationState::Cancelled => pb::generation_status::State::Cancelled,
		};
		Self {
			generation_id:    value.generation_id.into(),
			state:            state as i32,
			progress_percent: value.progress_percent,
			detail:           value.detail.into(),
			artifacts:        value
				.artifacts
				.into_iter()
				.map(|value| pb::generation_status::Artifact {
					blob:              value.blob.map(Into::into),
					variant:           value.variant.into(),
					url:               value.url.into(),
					url_expires_at_ms: value.url_expires_at_ms,
				})
				.collect(),
			usage:            value.usage.map(Into::into),
			cost:             value.cost.map(Into::into),
			unsupported:      value.unsupported.into_iter().map(Into::into).collect(),
			created_at_ms:    value.created_at_ms,
			updated_at_ms:    value.updated_at_ms,
			props:            props_to_proto(value.props),
		}
	}
}

impl TryFrom<pb::GenerationStatus> for GenerationStatus {
	type Error = ConvertError;

	fn try_from(value: pb::GenerationStatus) -> Result<Self, Self::Error> {
		let state = match value.state {
			x if x == pb::generation_status::State::Unspecified as i32 => {
				return Err(ConvertError::Unspecified("GenerationStatus.State"));
			},
			x if x == pb::generation_status::State::Queued as i32 => GenerationState::Queued,
			x if x == pb::generation_status::State::Running as i32 => GenerationState::Running,
			x if x == pb::generation_status::State::Completed as i32 => GenerationState::Completed,
			x if x == pb::generation_status::State::Failed as i32 => GenerationState::Failed,
			x if x == pb::generation_status::State::Cancelled as i32 => GenerationState::Cancelled,
			value => {
				return Err(ConvertError::UnknownEnum { name: "GenerationStatus.State", value });
			},
		};
		Ok(Self {
			generation_id: value.generation_id.into(),
			state,
			progress_percent: value.progress_percent,
			detail: value.detail.into(),
			artifacts: value
				.artifacts
				.into_iter()
				.map(|value| {
					Ok(GenerationArtifact {
						blob:              value.blob.map(TryInto::try_into).transpose()?,
						variant:           value.variant.into(),
						url:               value.url.into(),
						url_expires_at_ms: value.url_expires_at_ms,
					})
				})
				.collect::<Result<_, ConvertError>>()?,
			usage: value.usage.map(TryInto::try_into).transpose()?,
			cost: value.cost.map(Into::into),
			unsupported: value
				.unsupported
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
			created_at_ms: value.created_at_ms,
			updated_at_ms: value.updated_at_ms,
			props: props_from_proto(value.props)?,
		})
	}
}

impl From<SearchRequest> for pb::SearchRequest {
	fn from(value: SearchRequest) -> Self {
		let recency = match value.recency {
			None => pb::search_request::Recency::Unspecified,
			Some(SearchRecency::Day) => pb::search_request::Recency::Day,
			Some(SearchRecency::Week) => pb::search_request::Recency::Week,
			Some(SearchRecency::Month) => pb::search_request::Recency::Month,
			Some(SearchRecency::Year) => pb::search_request::Recency::Year,
		};
		let safesearch = match value.safesearch {
			None => pb::search_request::SafeSearch::Unspecified,
			Some(SafeSearch::Off) => pb::search_request::SafeSearch::Off,
			Some(SafeSearch::Moderate) => pb::search_request::SafeSearch::Moderate,
			Some(SafeSearch::Strict) => pb::search_request::SafeSearch::Strict,
		};
		Self {
			query:            value.query.into(),
			limit:            value.limit,
			recency:          recency as i32,
			after:            value.after.into(),
			before:           value.before.into(),
			allowed_domains:  value.allowed_domains.into_iter().map(Into::into).collect(),
			excluded_domains: value.excluded_domains.into_iter().map(Into::into).collect(),
			country:          value.country.into(),
			language:         value.language.into(),
			location:         value.location.map(|value| pb::search_request::Location {
				city:     value.city.into(),
				region:   value.region.into(),
				country:  value.country.into(),
				timezone: value.timezone.into(),
			}),
			safesearch:       safesearch as i32,
			engine:           value.engine.into(),
			timeout_ms:       value.timeout_ms,
			props:            props_to_proto(value.props),
		}
	}
}

impl TryFrom<pb::SearchRequest> for SearchRequest {
	type Error = ConvertError;

	fn try_from(value: pb::SearchRequest) -> Result<Self, Self::Error> {
		let recency = match value.recency {
			x if x == pb::search_request::Recency::Unspecified as i32 => None,
			x if x == pb::search_request::Recency::Day as i32 => Some(SearchRecency::Day),
			x if x == pb::search_request::Recency::Week as i32 => Some(SearchRecency::Week),
			x if x == pb::search_request::Recency::Month as i32 => Some(SearchRecency::Month),
			x if x == pb::search_request::Recency::Year as i32 => Some(SearchRecency::Year),
			value => return Err(ConvertError::UnknownEnum { name: "SearchRequest.Recency", value }),
		};
		let safesearch = match value.safesearch {
			x if x == pb::search_request::SafeSearch::Unspecified as i32 => None,
			x if x == pb::search_request::SafeSearch::Off as i32 => Some(SafeSearch::Off),
			x if x == pb::search_request::SafeSearch::Moderate as i32 => Some(SafeSearch::Moderate),
			x if x == pb::search_request::SafeSearch::Strict as i32 => Some(SafeSearch::Strict),
			value => {
				return Err(ConvertError::UnknownEnum { name: "SearchRequest.SafeSearch", value });
			},
		};
		Ok(Self {
			query: value.query.into(),
			limit: value.limit,
			recency,
			after: value.after.into(),
			before: value.before.into(),
			allowed_domains: value.allowed_domains.into_iter().map(Into::into).collect(),
			excluded_domains: value.excluded_domains.into_iter().map(Into::into).collect(),
			country: value.country.into(),
			language: value.language.into(),
			location: value.location.map(|value| SearchLocation {
				city:     value.city.into(),
				region:   value.region.into(),
				country:  value.country.into(),
				timezone: value.timezone.into(),
			}),
			safesearch,
			engine: value.engine.into(),
			timeout_ms: value.timeout_ms,
			props: props_from_proto(value.props)?,
		})
	}
}

impl From<SearchResponse> for pb::SearchResponse {
	fn from(value: SearchResponse) -> Self {
		Self {
			engine:         value.engine.into(),
			answer:         value.answer.into(),
			sources:        value
				.sources
				.into_iter()
				.map(|value| pb::search_response::Source {
					url:          value.url.into(),
					title:        value.title.into(),
					snippet:      value.snippet.into(),
					published_at: value.published_at.into(),
					author:       value.author.into(),
					score:        value.score,
				})
				.collect(),
			citations:      value
				.citations
				.into_iter()
				.map(|value| pb::search_response::Citation {
					url:        value.url.into(),
					title:      value.title.into(),
					cited_text: value.cited_text.into(),
					start:      value.start,
					end:        value.end,
				})
				.collect(),
			search_queries: value.search_queries.into_iter().map(Into::into).collect(),
			related:        value.related.into_iter().map(Into::into).collect(),
			warnings:       value.warnings.into_iter().map(Into::into).collect(),
			usage:          value.usage.map(Into::into),
			cost:           value.cost.map(Into::into),
			unsupported:    value.unsupported.into_iter().map(Into::into).collect(),
			props:          props_to_proto(value.props),
		}
	}
}

impl TryFrom<pb::SearchResponse> for SearchResponse {
	type Error = ConvertError;

	fn try_from(value: pb::SearchResponse) -> Result<Self, Self::Error> {
		Ok(Self {
			engine:         value.engine.into(),
			answer:         value.answer.into(),
			sources:        value
				.sources
				.into_iter()
				.map(|value| SearchSource {
					url:          value.url.into(),
					title:        value.title.into(),
					snippet:      value.snippet.into(),
					published_at: value.published_at.into(),
					author:       value.author.into(),
					score:        value.score,
				})
				.collect(),
			citations:      value
				.citations
				.into_iter()
				.map(|value| SearchCitation {
					url:        value.url.into(),
					title:      value.title.into(),
					cited_text: value.cited_text.into(),
					start:      value.start,
					end:        value.end,
				})
				.collect(),
			search_queries: value.search_queries.into_iter().map(Into::into).collect(),
			related:        value.related.into_iter().map(Into::into).collect(),
			warnings:       value.warnings.into_iter().map(Into::into).collect(),
			usage:          value.usage.map(TryInto::try_into).transpose()?,
			cost:           value.cost.map(Into::into),
			unsupported:    value
				.unsupported
				.into_iter()
				.map(TryInto::try_into)
				.collect::<Result<_, _>>()?,
			props:          props_from_proto(value.props)?,
		})
	}
}

#[cfg(test)]
mod tests {
	use bytes::Bytes;
	use pretty_assertions::assert_eq;
	use serde_json::json;

	use super::*;
	use crate::ids::CallId;

	fn call_id() -> CallId {
		"01ARZ3NDEKTSV4RRFFQ69G5FAV"
			.parse()
			.expect("valid fixed ULID")
	}

	fn props() -> Props {
		let mut props = Props::default();
		props.insert_ns("x", "test", json!({"nested": [true, 7, null]}));
		props
	}

	#[test]
	fn props_preserve_every_json_kind_and_large_integers() {
		let value = json!({
			"large": 9_007_199_254_740_993_i64,
			"unsigned": u64::MAX,
			"float": 7.25,
			"bool": true,
			"null": null,
			"list": [
				-3,
				2.5,
				false,
				null,
				{"nested": 9_007_199_254_740_995_i64}
			]
		});
		let mut props = Props::default();
		props.insert_ns("x", "all-kinds", value);

		let wire: pb::ValueMap = props.clone().into();
		let Some(pb::value::Kind::Map(fields)) = wire
			.fields
			.get("x/all-kinds")
			.and_then(|value| value.kind.as_ref())
		else {
			panic!("top-level prop must use the map arm");
		};
		assert!(matches!(
			fields
				.fields
				.get("large")
				.and_then(|value| value.kind.as_ref()),
			Some(pb::value::Kind::Int(9_007_199_254_740_993))
		));
		assert!(matches!(
			fields
				.fields
				.get("unsigned")
				.and_then(|value| value.kind.as_ref()),
			Some(pb::value::Kind::Uint(u64::MAX))
		));
		assert!(matches!(
			fields.fields.get("float").and_then(|value| value.kind.as_ref()),
			Some(pb::value::Kind::Double(value)) if *value == 7.25
		));
		let roundtrip = Props::try_from(wire).expect("props convert");

		assert_eq!(roundtrip, props);
	}

	fn blob() -> BlobPart {
		BlobPart {
			hash:   [7; 32],
			mime:   "image/png".into(),
			size:   3,
			inline: Bytes::from_static(b"png"),
			detail: Some(ImageDetail::Original),
		}
	}

	fn item() -> Item {
		Item {
			seq:           1,
			created_at_ms: 1_726_000_000_123,
			kind:          ItemKind::Message(Message {
				role:  Role::Assistant,
				parts: vec![Part::Text("answer".into())],
			}),
			props:         Props::default(),
		}
	}

	fn usage() -> Usage {
		Usage {
			input_tokens:       11,
			output_tokens:      12,
			cache_read_tokens:  3,
			cache_write_tokens: 4,
			total_tokens:       Some(30),
			context_tokens:     Some(29),
			orchestration:      Some(OrchestrationUsage {
				input_tokens:      Some(1),
				cache_read_tokens: Some(2),
				output_tokens:     Some(3),
			}),
			premium_requests:   Some(1),
			reasoning_tokens:   Some(5),
			cache_ttl:          Some(CacheTtlUsage {
				ephemeral_5m_tokens: Some(2),
				ephemeral_1h_tokens: Some(2),
			}),
			server_tools:       Some(ServerToolUsage {
				web_search_requests: Some(2),
				web_fetch_requests:  Some(1),
			}),
			accuracy:           Accuracy::Exact,
			detail:             props(),
		}
	}

	fn diagnostic(retryability: Retryability) -> Diagnostic {
		Diagnostic {
			provider: "provider".into(),
			model: "model".into(),
			attempt: 2,
			code: "rate_throttle".into(),
			detail: "retry later".into(),
			retryability,
		}
	}

	fn outcome() -> ChatOutcome {
		ChatOutcome {
			output:            vec![item()],
			stop:              StopReason::EndTurn,
			usage:             Some(usage()),
			cost:              Some(Cost {
				nanos_usd:             123,
				input_nanos_usd:       Some(20),
				output_nanos_usd:      Some(80),
				cache_read_nanos_usd:  Some(3),
				cache_write_nanos_usd: Some(20),
				estimated:             true,
			}),
			unsupported:       vec![Unsupported {
				what:   "sampling.top_k".into(),
				detail: "not native".into(),
				action: UnsupportedAction::Dropped,
			}],
			revision:          Some(Revision { head: 1, token: Bytes::from_static(b"revision") }),
			provider:          "provider".into(),
			model:             "model".into(),
			upstream_provider: Some("upstream".into()),
			duration_ms:       Some(987),
			ttft_ms:           Some(123),
			context_snapshot:  Some(ContextSnapshot {
				prompt_tokens:                  18,
				non_message_tokens:             2,
				history_rewrite_tokens_removed: Some(3),
				last_message_timestamp_ms:      Some(1_726_000_000_000),
			}),
			diagnostics:       vec![diagnostic(Retryability::AfterDelay), Diagnostic {
				provider:     Str::new_static(""),
				model:        Str::new_static(""),
				attempt:      0,
				code:         Str::new_static(""),
				detail:       Str::new_static(""),
				retryability: Retryability::Unspecified,
			}],
			props:             props(),
		}
	}

	fn turn_error() -> TurnError {
		TurnError {
			kind:           TurnErrorKind::Conflict,
			detail:         "stale".into(),
			actual:         Some(Revision { head: 2, token: Bytes::from_static(b"actual") }),
			unsupported:    Vec::new(),
			retry_after_ms: 0,
			diagnostics:    vec![diagnostic(Retryability::AfterCredential)],
			error_id:       Some(0x0102_0304),
		}
	}

	fn exec_status(outcome: ExecOutcome) -> ExecStatus {
		ExecStatus {
			outcome,
			exit_code: 9,
			signal: "SIGKILL".into(),
			reason: "reason".into(),
			cwd: "/tmp".into(),
			aborted: true,
			output_location: "blob://output".into(),
			local_execution_time_ms: 44,
			is_readonly: true,
			command_timeout_ms: 500,
		}
	}

	fn assert_event_roundtrip(event: TurnEvent) {
		let wire: pb::TurnEvent = event.clone().into();
		assert_eq!(TurnEvent::try_from(wire).expect("event converts"), event);
	}

	#[test]
	fn thread_roundtrips_every_part_without_copying_bytes() {
		let signature = Bytes::from_static(b"signature");
		let signature_ptr = signature.as_ptr();
		let thread = Thread {
			items: vec![
				Item {
					seq:           0,
					created_at_ms: 1_726_000_000_456,
					kind:          ItemKind::Message(Message {
						role:  Role::User,
						parts: vec![
							Part::Text("hello".into()),
							Part::Thinking(Thinking { text: "chain".into(), signature, redacted: false }),
							Part::Blob(blob()),
							Part::Fallback(ModelFallback {
								from_model: "claude-opus".into(),
								to_model:   "claude-sonnet".into(),
							}),
							Part::ServerTool(ServerTool {
								provider:          "anthropic".into(),
								kind:              ServerToolKind::Call,
								id:                "srv_1".into(),
								name:              "web_search".into(),
								payload_json:      Bytes::from_static(br#"{"query":"rust"}"#),
								provider_metadata: Some(props()),
							}),
						],
					}),
					props:         props(),
				},
				Item {
					seq:           0,
					created_at_ms: 0,
					kind:          ItemKind::ToolCall(ToolCall {
						id:                call_id(),
						name:              "read".into(),
						args_json:         Bytes::from_static(br#"{"path":"x"}"#),
						thought_signature: Bytes::from_static(b"thought"),
						intent:            Some("inspect".into()),
						raw:               Some(Bytes::from_static(b"<tool>read</tool>")),
						custom_wire_name:  Some("read_file".into()),
						provider_metadata: Some(props()),
					}),
					props:         Props::default(),
				},
				Item {
					seq:           0,
					created_at_ms: 0,
					kind:          ItemKind::ToolResult(ToolResult {
						call_id:           call_id(),
						name:              "read".into(),
						parts:             vec![Part::Text("ok".into()), Part::Blob(blob())],
						is_error:          false,
						details:           Some(json!([{"elapsed_ms": 12}, null, 7])),
						attribution:       Some(MessageAttribution::Agent),
						pruned_at_ms:      Some(1_726_000_000_999),
						useless:           Some(false),
						provider_metadata: Some(Props::default()),
					}),
					props:         Props::default(),
				},
			],
		};
		let wire: pb_thread::Thread = thread.clone().into();
		let wire_signature = match &wire.items[0].kind {
			Some(pb_thread::item::Kind::Message(message)) => match &message.parts[1].kind {
				Some(pb_thread::part::Kind::Thinking(thinking)) => &thinking.signature,
				_ => panic!("thinking part"),
			},
			_ => panic!("message item"),
		};
		assert_eq!(wire_signature.as_ptr(), signature_ptr);
		let Some(pb_thread::item::Kind::ToolResult(wire_result)) = &wire.items[2].kind else {
			panic!("tool result item");
		};
		assert_eq!(wire_result.name, "read");
		assert_eq!(Thread::try_from(wire).expect("thread converts"), thread);
	}

	#[test]
	fn terminal_diagnostics_preserve_empty_collections() {
		let mut success = outcome();
		success.diagnostics.clear();
		let wire: pb::Outcome = success.clone().into();
		assert!(wire.diagnostics.is_empty());
		assert_eq!(ChatOutcome::try_from(wire).expect("outcome converts"), success);

		let mut failure = turn_error();
		failure.diagnostics.clear();
		let wire: pb::TurnError = failure.clone().into();
		assert!(wire.diagnostics.is_empty());
		assert_eq!(TurnError::try_from(wire).expect("turn error converts"), failure);
	}

	#[test]
	fn chat_request_roundtrips_every_extended_feature() {
		let request = ChatRequest {
			model_policy:           Some(std::sync::Arc::new(ResolvedModelPolicy {
				request_model_id: Some("wire-model".into()),
				..ResolvedModelPolicy::default()
			})),
			model:                  "provider/model".into(),
			thread:                 Thread { items: vec![item()] },
			tools:                  vec![ToolDef {
				name:        "read".into(),
				description: "read a file".into(),
				schema_json: Bytes::from_static(br#"{"type":"object"}"#),
				strict:      Some(true),
			}],
			tool_choice:            Some(Feature {
				value:          ToolChoice::Named("read".into()),
				on_unsupported: Fallback::Emulate,
			}),
			sampling:               Some(Sampling {
				temperature:        Some(0.2),
				top_p:              Some(0.9),
				top_k:              Some(20),
				min_p:              Some(0.05),
				frequency_penalty:  Some(0.1),
				presence_penalty:   Some(-0.1),
				repetition_penalty: Some(1.1),
				stop:               Some(vec!["done".into()]),
				max_output_tokens:  Some(512),
			}),
			thinking:               Some(Feature {
				value:          Reasoning {
					effort:        Some(Effort::High),
					budget_tokens: Some(2048),
					hide_summary:  Some(true),
				},
				on_unsupported: Fallback::Error,
			}),
			cache:                  Some(CacheHint {
				session_key: "conversation".into(),
				retention:   Some(CacheRetention::Long),
				mode:        Some(PromptCacheMode::Explicit),
				ttl:         Some(PromptCacheTtl::ThirtyMinutes),
				breakpoint:  Some(PromptCacheBreakpoint::LatestStableMessage),
			}),
			response_format:        Some(Feature {
				value:          ResponseFormat {
					kind: ResponseFormatKind::JsonSchema(JsonSchema {
						name:        "answer".into(),
						schema_json: Bytes::from_static(br#"{"type":"string"}"#),
						strict:      Some(true),
					}),
				},
				on_unsupported: Fallback::Ignore,
			}),
			meta:                   Some(RequestMeta {
				initiator:  "agent".into(),
				session_id: "session".into(),
				telemetry:  BTreeMap::from([("trace".into(), "value".into())]),
			}),
			provider_options:       Some(props()),
			service_tier:           Some(ServiceTier::Priority),
			service_tier_by_family: Some(ServiceTierByFamily {
				openai:    Some(ServiceTier::Flex),
				anthropic: Some(ServiceTier::Priority),
				google:    Some(ServiceTier::Default),
			}),
			task_budget:            Some(TaskBudget {
				total_tokens:     10_000,
				remaining_tokens: Some(7_500),
			}),
			responses_include:      Some(vec![
				ResponseInclude::FileSearchResults,
				ResponseInclude::WebSearchResults,
				ResponseInclude::WebSearchSources,
				ResponseInclude::InputImageUrl,
				ResponseInclude::ComputerOutputImageUrl,
				ResponseInclude::CodeInterpreterOutputs,
				ResponseInclude::ReasoningEncryptedContent,
				ResponseInclude::OutputTextLogprobs,
			]),
		};
		let wire: pb::TurnRequest = request.clone().into();
		let decoded = ChatRequest::try_from(wire).expect("request converts");
		assert!(decoded.model_policy.is_none(), "foreign wire cannot inject trusted policy");
		let mut expected = request;
		expected.model_policy = None;
		assert_eq!(decoded, expected);
	}

	#[test]
	fn request_presence_and_every_tool_choice_roundtrip() {
		let choices = [
			ToolChoice::Auto,
			ToolChoice::None,
			ToolChoice::Required,
			ToolChoice::Named("read".into()),
		];
		for choice in choices {
			let params = ChatParams {
				model_policy:           None,
				model:                  "provider/model".into(),
				tools:                  vec![ToolDef {
					name:        "read".into(),
					description: "read".into(),
					schema_json: Bytes::from_static(b"{}"),
					strict:      Some(false),
				}],
				tool_choice:            Some(Feature {
					value:          choice,
					on_unsupported: Fallback::Error,
				}),
				sampling:               Some(Sampling {
					stop: Some(Vec::new()),
					repetition_penalty: Some(0.0),
					..Sampling::default()
				}),
				thinking:               Some(Feature {
					value:          Reasoning {
						effort:        None,
						budget_tokens: None,
						hide_summary:  Some(false),
					},
					on_unsupported: Fallback::Ignore,
				}),
				cache:                  None,
				response_format:        Some(Feature {
					value:          ResponseFormat {
						kind: ResponseFormatKind::JsonSchema(JsonSchema {
							name:        "result".into(),
							schema_json: Bytes::from_static(b"{}"),
							strict:      Some(false),
						}),
					},
					on_unsupported: Fallback::Emulate,
				}),
				meta:                   None,
				provider_options:       Some(Props::default()),
				service_tier:           Some(ServiceTier::Auto),
				service_tier_by_family: Some(ServiceTierByFamily::default()),
				task_budget:            Some(TaskBudget {
					total_tokens:     0,
					remaining_tokens: Some(0),
				}),
				responses_include:      Some(Vec::new()),
			};
			let wire: pb::ChatParams = params.clone().into();
			assert_eq!(wire.tools[0].strict, Some(false));
			assert_eq!(wire.thinking.as_ref().and_then(|value| value.hide_summary), Some(false));
			assert_eq!(wire.sampling.as_ref().and_then(|value| value.stop_present), Some(true));
			assert!(wire.responses_include.is_some());
			assert!(wire.service_tier_by_family.is_some());
			let schema = match wire
				.response_format
				.as_ref()
				.and_then(|value| value.kind.as_ref())
			{
				Some(pb::response_format::Kind::JsonSchema(schema)) => schema,
				_ => panic!("json schema response format"),
			};
			assert_eq!(schema.strict, Some(false));
			assert!(wire.provider_options.is_some());
			assert_eq!(ChatParams::try_from(wire).expect("params convert"), params);
		}

		let absent = ChatParams {
			model_policy:           None,
			model:                  "provider/model".into(),
			tools:                  vec![ToolDef {
				name:        "read".into(),
				description: "read".into(),
				schema_json: Bytes::new(),
				strict:      None,
			}],
			tool_choice:            None,
			sampling:               None,
			thinking:               None,
			cache:                  None,
			response_format:        None,
			meta:                   None,
			provider_options:       None,
			service_tier:           None,
			service_tier_by_family: None,
			task_budget:            None,
			responses_include:      None,
		};
		let wire: pb::ChatParams = absent.clone().into();
		assert_eq!(wire.tools[0].strict, None);
		assert!(wire.tool_choice.is_none());
		assert!(wire.thinking.is_none());
		assert!(wire.response_format.is_none());
		assert!(wire.provider_options.is_none());
		assert_eq!(ChatParams::try_from(wire).expect("params convert"), absent);
	}
	#[test]
	fn xhigh_and_max_efforts_roundtrip_distinctly() {
		for effort in [
			Effort::Off,
			Effort::Minimal,
			Effort::Low,
			Effort::Medium,
			Effort::High,
			Effort::XHigh,
			Effort::Max,
		] {
			assert_eq!(
				effort_from_proto(effort_to_proto(Some(effort))).expect("effort converts"),
				Some(effort),
			);
		}
		assert_ne!(effort_to_proto(Some(Effort::XHigh)), effort_to_proto(Some(Effort::Max)),);
	}

	#[test]
	fn every_extended_enum_and_unsupported_fallback_roundtrips() {
		for tier in [
			ServiceTier::Auto,
			ServiceTier::Default,
			ServiceTier::Flex,
			ServiceTier::Scale,
			ServiceTier::Priority,
		] {
			assert_eq!(
				service_tier_from_proto(service_tier_to_proto(Some(tier)))
					.expect("service tier converts"),
				Some(tier),
			);
		}
		assert_eq!(
			service_tier_from_proto(service_tier_to_proto(None)).expect("absence converts"),
			None,
		);

		for retention in [CacheRetention::None, CacheRetention::Short, CacheRetention::Long] {
			let cache = CacheHint {
				session_key: "session".into(),
				retention:   Some(retention),
				mode:        None,
				ttl:         None,
				breakpoint:  None,
			};
			let wire: pb::CacheHint = cache.clone().into();
			assert_eq!(CacheHint::try_from(wire).expect("cache retention converts"), cache);
		}
		for mode in [PromptCacheMode::Implicit, PromptCacheMode::Explicit] {
			let cache = CacheHint {
				session_key: "session".into(),
				retention:   None,
				mode:        Some(mode),
				ttl:         Some(PromptCacheTtl::ThirtyMinutes),
				breakpoint:  None,
			};
			let wire: pb::CacheHint = cache.clone().into();
			assert_eq!(CacheHint::try_from(wire).expect("cache mode converts"), cache);
		}
		for breakpoint in [PromptCacheBreakpoint::LatestStableMessage, PromptCacheBreakpoint::None] {
			let cache = CacheHint {
				session_key: "session".into(),
				retention:   None,
				mode:        None,
				ttl:         None,
				breakpoint:  Some(breakpoint),
			};
			let wire: pb::CacheHint = cache.clone().into();
			assert_eq!(CacheHint::try_from(wire).expect("cache breakpoint converts"), cache);
		}

		let mut no_detail = blob();
		no_detail.detail = None;
		let wire: pb_thread::Blob = no_detail.clone().into();
		assert_eq!(BlobPart::try_from(wire).expect("absent image detail converts"), no_detail);
		for detail in [ImageDetail::Auto, ImageDetail::Low, ImageDetail::High, ImageDetail::Original]
		{
			let mut image = blob();
			image.detail = Some(detail);
			let wire: pb_thread::Blob = image.clone().into();
			assert_eq!(BlobPart::try_from(wire).expect("image detail converts"), image);
		}
		for kind in [ServerToolKind::Call, ServerToolKind::Result] {
			let block = ServerTool {
				provider: "anthropic".into(),
				kind,
				id: "server_1".into(),
				name: "web_search".into(),
				payload_json: Bytes::from_static(b"{}"),
				provider_metadata: Some(Props::default()),
			};
			let wire: pb_thread::ServerTool = block.clone().into();
			assert_eq!(ServerTool::try_from(wire).expect("server tool converts"), block);
		}
		for attribution in [MessageAttribution::User, MessageAttribution::Agent] {
			let result = ToolResult {
				call_id:           call_id(),
				name:              "read".into(),
				parts:             Vec::new(),
				is_error:          false,
				details:           None,
				attribution:       Some(attribution),
				pruned_at_ms:      Some(0),
				useless:           Some(false),
				provider_metadata: None,
			};
			let wire: pb_thread::ToolResult = result.clone().into();
			assert_eq!(ToolResult::try_from(wire).expect("attribution converts"), result);
		}

		for fallback in [Fallback::Error, Fallback::Ignore, Fallback::Emulate] {
			let feature = Feature { value: ToolChoice::Auto, on_unsupported: fallback };
			let wire: pb::ToolChoice = feature.clone().into();
			assert_eq!(Feature::<ToolChoice>::try_from(wire).expect("fallback converts"), feature);
		}
		for action in
			[UnsupportedAction::Dropped, UnsupportedAction::Emulated, UnsupportedAction::Clamped]
		{
			let unsupported =
				Unsupported { what: "request.feature".into(), detail: "unsupported".into(), action };
			let wire: pb::Unsupported = unsupported.clone().into();
			assert_eq!(Unsupported::try_from(wire).expect("unsupported converts"), unsupported);
		}
		for accuracy in [Accuracy::Exact, Accuracy::Estimated, Accuracy::Mixed] {
			let mut value = usage();
			value.accuracy = accuracy;
			let wire: pb::Usage = value.clone().into();
			assert_eq!(Usage::try_from(wire).expect("accuracy converts"), value);
		}
		for retryability in [
			Retryability::Unspecified,
			Retryability::Never,
			Retryability::SameRoute,
			Retryability::AfterRepair,
			Retryability::AfterCredential,
			Retryability::AfterDelay,
		] {
			let value = diagnostic(retryability);
			let wire: pb::Diagnostic = value.clone().into();
			assert_eq!(Diagnostic::try_from(wire).expect("retryability converts"), value);
		}
	}

	#[test]
	fn extended_optional_outcome_and_error_fields_preserve_absence() {
		let mut success = outcome();
		success.upstream_provider = None;
		success.duration_ms = None;
		success.ttft_ms = None;
		success.context_snapshot = None;
		let wire: pb::Outcome = success.clone().into();
		assert_eq!(ChatOutcome::try_from(wire).expect("optional outcome converts"), success);

		let minimal_usage = Usage {
			input_tokens:       1,
			output_tokens:      2,
			cache_read_tokens:  0,
			cache_write_tokens: 0,
			total_tokens:       None,
			context_tokens:     None,
			orchestration:      None,
			premium_requests:   None,
			reasoning_tokens:   None,
			cache_ttl:          None,
			server_tools:       None,
			accuracy:           Accuracy::Estimated,
			detail:             Props::default(),
		};
		let wire: pb::Usage = minimal_usage.clone().into();
		assert_eq!(Usage::try_from(wire).expect("optional usage converts"), minimal_usage);

		let minimal_cost = Cost {
			nanos_usd:             7,
			input_nanos_usd:       None,
			output_nanos_usd:      None,
			cache_read_nanos_usd:  None,
			cache_write_nanos_usd: None,
			estimated:             false,
		};
		let wire: pb::Cost = minimal_cost.into();
		assert_eq!(Cost::from(wire), minimal_cost);

		let budget = TaskBudget { total_tokens: 1, remaining_tokens: None };
		let wire: pb::TaskBudget = budget.clone().into();
		assert_eq!(TaskBudget::from(wire), budget);

		let mut failure = turn_error();
		failure.error_id = None;
		let wire: pb::TurnError = failure.clone().into();
		assert_eq!(TurnError::try_from(wire).expect("optional error converts"), failure);
	}

	#[test]
	fn every_turn_event_variant_roundtrips() {
		let invoke = Invoke {
			invocation_id: "invoke-1".into(),
			name:          "exec.shell".into(),
			tool_call:     Some(ToolCall {
				id:                call_id(),
				name:              "exec.shell".into(),
				args_json:         Bytes::from_static(b"{}"),
				thought_signature: Bytes::new(),
				intent:            None,
				raw:               None,
				custom_wire_name:  None,
				provider_metadata: None,
			}),
			vendor:        Bytes::from_static(b"vendor"),
			timeout_ms:    1000,
			props:         props(),
		};
		for event in [
			TurnEvent::Accepted { replay: true },
			TurnEvent::Attempt { number: 2, reason: "rotated credential".into() },
			TurnEvent::PartStart {
				index:        0,
				kind:         StreamPartKind::ToolCall,
				tool_call_id: call_id().to_string().into(),
				tool_name:    "read".into(),
			},
			TurnEvent::PartDelta { index: 0, chunk: Bytes::from_static(b"{") },
			TurnEvent::PartEnd { index: 0, signature: Bytes::from_static(b"signed-thinking") },
			TurnEvent::Invoke(invoke),
			TurnEvent::InvokeCancel { invocation_id: "invoke-1".into() },
			TurnEvent::Outcome(outcome()),
			TurnEvent::Error(turn_error()),
		] {
			assert_event_roundtrip(event);
		}
	}

	#[test]
	fn every_exec_outcome_roundtrips() {
		for outcome in [
			ExecOutcome::Exited,
			ExecOutcome::Failed,
			ExecOutcome::Rejected,
			ExecOutcome::Denied,
			ExecOutcome::Timeout,
			ExecOutcome::Cancelled,
		] {
			let status = exec_status(outcome);
			let wire: pb::ExecStatus = status.clone().into();
			assert_eq!(ExecStatus::try_from(wire).expect("status converts"), status);
		}
	}

	#[test]
	fn unspecified_required_enums_are_rejected() {
		let message =
			pb_thread::Message { role: pb_thread::Role::Unspecified as i32, parts: Vec::new() };
		assert!(matches!(Message::try_from(message), Err(ConvertError::Unspecified("Role"))));
		assert!(matches!(
			ExecStatus::try_from(pb::ExecStatus::default()),
			Err(ConvertError::Unspecified("ExecStatus.Outcome"))
		));
	}
}
