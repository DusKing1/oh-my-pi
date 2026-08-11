//! Google Generative Language and Vertex AI `GenerateContent` wire codec.

pub mod adc;
pub mod cca;
pub mod discovery;
pub mod embeddings;
pub mod leak_filter;
mod request;
pub mod stream;
pub mod vertex;

use std::collections::BTreeMap;

use bytes::Bytes;
use omp_core::Str;
use omp_llm_catalog::{
	compat::{Compat, ReasoningWireFormat},
	provider::TransportId,
};
use omp_llm_transport::{DecodeState, Frame, Transport, with_tool_use_precedence};
use omp_llm_types::{
	Accuracy, CallId, CallIdMapper, ChatOutcome, ChatRequest, Effort, Error, Fallback, Item,
	ItemKind, Message, Part, Props, ResolvedThinkingMode, ResolvedThinkingPolicy, Role, StopReason,
	StreamPartKind, Thinking, ToolCall, ToolCallIdProfile, ToolChoice, ToolDef, TurnError,
	TurnErrorKind, TurnEvent, Unsupported, UnsupportedAction, Usage,
};
use serde::Deserialize;
use serde_json::{Map, Value, json, value::RawValue};
use smallvec::SmallVec;
pub use vertex::vertex_stream_url;

/// Google `GenerateContent` codec, parameterized by the endpoint variant.
#[derive(Clone, Copy, Debug)]
pub struct GoogleCodec {
	variant: GoogleVariant,
}

impl GoogleCodec {
	/// Creates a codec for the public Generative Language API.
	#[must_use]
	pub const fn gen_ai() -> Self {
		Self { variant: GoogleVariant::GEN_AI }
	}

	/// Creates a codec for Vertex AI's publisher-model endpoint.
	#[must_use]
	pub const fn vertex() -> Self {
		Self { variant: GoogleVariant::VERTEX }
	}
}

impl Default for GoogleCodec {
	fn default() -> Self {
		Self::gen_ai()
	}
}

impl Transport for GoogleCodec {
	fn id(&self) -> TransportId {
		self.variant.id
	}

	fn encode(
		&self,
		req: &ChatRequest,
		compat: &Compat,
	) -> Result<(Bytes, Vec<Unsupported>), Error> {
		let (value, unsupported) = encode_request(req, compat, self.variant)?;
		let body = serialize_preserving_args(&value, req)?;
		Ok((Bytes::from(body), unsupported))
	}

	fn decode(
		&self,
		frame: Frame<'_>,
		state: &mut DecodeState,
	) -> Result<SmallVec<TurnEvent, 2>, Error> {
		match frame {
			Frame::Data(data) | Frame::Event { data, .. } => decode_bytes(data, state),
			Frame::Done => Ok(finish_stream(state)),
			_ => Ok(SmallVec::new()),
		}
	}
}

/// Data-only differences between Google `GenerateContent` endpoint families.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GoogleVariant {
	id:                TransportId,
	function_part_ids: bool,
}

impl GoogleVariant {
	pub(crate) const CCA: Self =
		Self { id: TransportId::GoogleCca, function_part_ids: true };
	pub(crate) const GEN_AI: Self =
		Self { id: TransportId::GoogleGenAi, function_part_ids: true };
	pub(crate) const VERTEX: Self =
		Self { id: TransportId::GoogleVertex, function_part_ids: false };
}

pub(crate) fn encode_request(
	req: &ChatRequest,
	compat: &Compat,
	variant: GoogleVariant,
) -> Result<(Value, Vec<Unsupported>), Error> {
	encode_request_with_tools(req, &req.tools, compat, variant)
}

pub(crate) fn encode_request_with_tools(
	req: &ChatRequest,
	tools: &[ToolDef],
	compat: &Compat,
	variant: GoogleVariant,
) -> Result<(Value, Vec<Unsupported>), Error> {
	let mapper = CallIdMapper::new();
	let mut unsupported = Vec::new();
	let mut contents = Vec::<Value>::new();
	let mut system_parts = Vec::<Value>::new();
	let mut call_names = BTreeMap::<CallId, Str>::new();

	for item in &req.thread.items {
		match &item.kind {
			ItemKind::Message(message) => {
				let mut parts = Vec::new();
				for (index, part) in message.parts.iter().enumerate() {
					if message.role == Role::System {
						match part {
							Part::Text(text) if !text.is_empty() => {
								parts.push(json!({ "text": text }));
							},
							Part::Text(_) => {},
							_ => report(
								&mut unsupported,
								"thread.system.parts",
								"Gemini systemInstruction accepts text parts only",
								Fallback::Ignore,
							)?,
						}
					} else if let Some(part) =
						encode_message_part(part, index, &item.props, &mut unsupported)?
					{
						parts.push(part);
					}
				}
				if parts.is_empty() {
					continue;
				}
				match message.role {
					Role::System => system_parts.extend(parts),
					Role::User => append_content(&mut contents, "user", parts, false),
					Role::Assistant => append_content(&mut contents, "model", parts, false),
					_ => {},
				}
			},
			ItemKind::ToolCall(call) => {
				let args = serde_json::from_slice::<Value>(&call.args_json).map_err(provider_error)?;
				let wire_id = mapper.to_wire(&call.id, ToolCallIdProfile::Preserve);
				call_names.insert(call.id, call.name.clone());
				let mut function = Map::new();
				if variant.function_part_ids {
					function.insert("id".into(), Value::String(wire_id.to_string()));
				}
				function.insert("name".into(), Value::String(call.name.to_string()));
				function.insert("args".into(), args);
				let mut part = Map::new();
				part.insert("functionCall".into(), Value::Object(function));
				if !call.thought_signature.is_empty() {
					part.insert(
						"thoughtSignature".into(),
						Value::String(signature_string(&call.thought_signature)?),
					);
				}
				append_content(&mut contents, "model", [Value::Object(part)], false);
			},
			ItemKind::ToolResult(result) => {
				let wire_id = mapper.to_wire(&result.call_id, ToolCallIdProfile::Preserve);
				// Empty names only occur in legacy canonical threads written before
				// ToolResult carried the harness tool name.
				let name = if result.name.is_empty() {
					call_names.get(&result.call_id).cloned().unwrap_or_default()
				} else {
					result.name.clone()
				};
				let mut text = String::new();
				let mut response_parts = Vec::new();
				for (index, part) in result.parts.iter().enumerate() {
					match part {
						Part::Text(value) => {
							if !text.is_empty() {
								text.push('\n');
							}
							text.push_str(value);
						},
						Part::Blob(blob) if blob.inline.is_empty() => {
							if let Some(encoded) = request::project_file_data(blob, index, &item.props)? {
								response_parts.push(encoded);
							} else {
								report(
									&mut unsupported,
									"thread.tool_result.blob",
									"Google file result requires inline bytes or google/file_data",
									Fallback::Ignore,
								)?;
							}
						},
						Part::Blob(_) => {
							if let Some(encoded) = encode_part(part, &mut unsupported)? {
								response_parts.push(encoded);
							}
						},
						Part::Thinking(_) => report(
							&mut unsupported,
							"thread.tool_result.thinking",
							"thinking cannot appear in a Google function response",
							Fallback::Ignore,
						)?,
						_ => {},
					}
				}
				let mut response = Map::new();
				response.insert(
					if result.is_error { "error" } else { "output" }.into(),
					Value::String(text),
				);
				let mut function = Map::new();
				if variant.function_part_ids {
					function.insert("id".into(), Value::String(wire_id.to_string()));
				}
				function.insert("name".into(), Value::String(name.to_string()));
				function.insert("response".into(), Value::Object(response));
				if !response_parts.is_empty() {
					function.insert("parts".into(), Value::Array(response_parts));
				}
				append_content(&mut contents, "user", [json!({ "functionResponse": function })], true);
			},
			_ => {},
		}
	}

	let mut body = Map::new();
	body.insert("contents".into(), Value::Array(contents));
	if !system_parts.is_empty() {
		body.insert("systemInstruction".into(), json!({ "parts": system_parts }));
	}
	encode_generation_config(req, compat, &mut body, &mut unsupported)?;
	encode_tools(req, tools, compat, variant, &mut body, &mut unsupported)?;
	request::project_provider_options(req, &mut body, &mut unsupported)?;
	Ok((Value::Object(body), unsupported))
}

fn encode_message_part(
	part: &Part,
	index: usize,
	props: &Props,
	unsupported: &mut Vec<Unsupported>,
) -> Result<Option<Value>, Error> {
	if let Some(signature) = props
		.get_ns("google", "text_thought_signatures")
		.and_then(Value::as_array)
		.and_then(|signatures| signatures.get(index))
		.and_then(Value::as_str)
		&& let Part::Text(text) = part
	{
		return Ok(Some(json!({ "text": text, "thoughtSignature": signature })));
	}
	if let Part::Blob(blob) = part
		&& blob.inline.is_empty()
		&& let Some(file) = request::project_file_data(blob, index, props)?
	{
		return Ok(Some(file));
	}
	encode_part(part, unsupported)
}

fn encode_part(part: &Part, unsupported: &mut Vec<Unsupported>) -> Result<Option<Value>, Error> {
	match part {
		Part::Text(text) => Ok(Some(json!({ "text": text }))),
		Part::Thinking(thinking) => {
			let mut object = Map::new();
			object.insert("text".into(), Value::String(thinking.text.to_string()));
			object.insert("thought".into(), Value::Bool(true));
			if !thinking.signature.is_empty() {
				object.insert(
					"thoughtSignature".into(),
					Value::String(signature_string(&thinking.signature)?),
				);
			}
			Ok(Some(Value::Object(object)))
		},
		Part::Blob(blob) if !blob.inline.is_empty() => Ok(Some(request::encode_inline(blob))),
		Part::Blob(_) => {
			report(
				unsupported,
				"thread.parts.blob.inline",
				"Google inlineData requires payload bytes",
				Fallback::Ignore,
			)?;
			Ok(None)
		},
		_ => Ok(None),
	}
}

fn append_content(
	contents: &mut Vec<Value>,
	role: &str,
	parts: impl IntoIterator<Item = Value>,
	function_response: bool,
) {
	if let Some(last) = contents.last_mut()
		&& last.get("role").and_then(Value::as_str) == Some(role)
		&& (!function_response
			|| last
				.get("parts")
				.and_then(Value::as_array)
				.is_some_and(|parts| {
					parts
						.iter()
						.any(|part| part.get("functionResponse").is_some())
				}))
	{
		last
			.get_mut("parts")
			.and_then(Value::as_array_mut)
			.expect("content parts are arrays")
			.extend(parts);
		return;
	}
	let parts = parts.into_iter().collect::<Vec<_>>();
	contents.push(json!({ "role": role, "parts": parts }));
}

fn encode_generation_config(
	req: &ChatRequest,
	compat: &Compat,
	body: &mut Map<String, Value>,
	unsupported: &mut Vec<Unsupported>,
) -> Result<(), Error> {
	let mut generation = Map::new();
	let reasoning_enabled = reasoning_enabled(req);
	let sampling_params = model_compat_bool(req, "supports_sampling_params", reasoning_enabled)
		.unwrap_or(compat.sampling_params);
	if let Some(sampling) = &req.sampling {
		if sampling_params {
			insert_option(&mut generation, "temperature", sampling.temperature);
			insert_option(&mut generation, "topP", sampling.top_p);
			insert_option(&mut generation, "topK", sampling.top_k);
		} else {
			for (name, present) in [
				("sampling.temperature", sampling.temperature.is_some()),
				("sampling.top_p", sampling.top_p.is_some()),
				("sampling.top_k", sampling.top_k.is_some()),
			] {
				if present {
					report(
						unsupported,
						name,
						"model compatibility disables sampling controls",
						Fallback::Ignore,
					)?;
				}
			}
		}
		if req
			.model_policy
			.as_deref()
			.and_then(|policy| policy.omit_max_output_tokens)
			!= Some(true)
		{
			insert_option(&mut generation, "maxOutputTokens", sampling.max_output_tokens);
		}
		if let Some(stop) = &sampling.stop {
			if compat.stop_sequences {
				generation.insert("stopSequences".into(), json!(stop));
			} else {
				report(
					unsupported,
					"sampling.stop",
					"provider compatibility disables stop sequences",
					Fallback::Ignore,
				)?;
			}
		}
		for (name, present) in [
			("sampling.min_p", sampling.min_p.is_some()),
			("sampling.frequency_penalty", sampling.frequency_penalty.is_some()),
			("sampling.presence_penalty", sampling.presence_penalty.is_some()),
		] {
			if present {
				report(
					unsupported,
					name,
					"control has no portable Google GenerateContent projection",
					Fallback::Ignore,
				)?;
			}
		}
	}
	if let Some(reasoning) = &req.thinking {
		let thinking = if let Some(policy) = req.model_policy.as_deref() {
			if let Some(thinking_policy) = policy.thinking.as_ref() {
				encode_policy_thinking(
					&reasoning.value,
					thinking_policy,
					unsupported,
					reasoning.on_unsupported,
				)?
			} else {
				report(
					unsupported,
					"thinking",
					"resolved model policy does not advertise native thinking",
					reasoning.on_unsupported,
				)?;
				None
			}
		} else if compat.reasoning_wire_format == ReasoningWireFormat::Google {
			let mut thinking = Map::new();
			thinking.insert(
				"includeThoughts".into(),
				Value::Bool(!reasoning.value.hide_summary.unwrap_or(false)),
			);
			if let Some(effort) = reasoning.value.effort {
				thinking.insert("thinkingLevel".into(), Value::String(thinking_level(effort).into()));
			} else if let Some(budget) = reasoning.value.budget_tokens {
				thinking.insert("thinkingBudget".into(), Value::Number(budget.into()));
			}
			Some(thinking)
		} else {
			report(
				unsupported,
				"thinking",
				"provider compatibility disables Google thinkingConfig",
				reasoning.on_unsupported,
			)?;
			None
		};
		if let Some(thinking) = thinking {
			generation.insert("thinkingConfig".into(), Value::Object(thinking));
		}
	}
	request::project_response_format(req, compat, &mut generation, unsupported)?;
	if !generation.is_empty() {
		body.insert("generationConfig".into(), Value::Object(generation));
	}
	if req.cache.is_some() {
		report(
			unsupported,
			"cache",
			"a session key is not a Google cachedContent resource name",
			Fallback::Ignore,
		)?;
	}
	Ok(())
}

fn encode_tools(
	req: &ChatRequest,
	tools: &[ToolDef],
	compat: &Compat,
	variant: GoogleVariant,
	body: &mut Map<String, Value>,
	unsupported: &mut Vec<Unsupported>,
) -> Result<(), Error> {
	if !tools.is_empty() {
		let mut declarations = Vec::with_capacity(tools.len());
		for tool in tools {
			let schema = serde_json::from_slice::<Value>(&tool.schema_json).map_err(provider_error)?;
			let (schema, mut reports) = request::normalize_tool_schema(compat, &schema);
			unsupported.append(&mut reports);
			if tool.strict.is_some() {
				report(
					unsupported,
					Str::from(format!("tools.{}.strict", tool.name)),
					"Gemini function declarations do not expose a strict boolean",
					Fallback::Ignore,
				)?;
			}
			let mut declaration = Map::new();
			declaration.insert("name".into(), Value::String(tool.name.to_string()));
			if !tool.description.is_empty() {
				declaration.insert("description".into(), Value::String(tool.description.to_string()));
			}
			declaration.insert(
				if variant.id == TransportId::GoogleCca {
					"parameters"
				} else {
					"parametersJsonSchema"
				}
				.into(),
				schema,
			);
			declarations.push(Value::Object(declaration));
		}
		body.insert("tools".into(), json!([{ "functionDeclarations": declarations }]));
	}
	if let Some(choice) = &req.tool_choice {
		if tools.is_empty() {
			match &choice.value {
				ToolChoice::Auto | ToolChoice::None => return Ok(()),
				ToolChoice::Required | ToolChoice::Named(_) => {
					report(
						unsupported,
						"tool_choice",
						"Gemini cannot force function calling without function declarations",
						choice.on_unsupported,
					)?;
					return Ok(());
				},
				_ => return Ok(()),
			}
		}
		if let ToolChoice::Named(name) = &choice.value
			&& !tools.iter().any(|tool| tool.name == *name)
		{
			return Err(Error::Provider(Str::from(format!(
				"named Google tool choice `{name}` is not declared"
			))));
		}
		let (mode, allowed): (&str, Option<Value>) = match &choice.value {
			ToolChoice::Auto => ("AUTO", None),
			ToolChoice::None => ("NONE", None),
			ToolChoice::Required => ("ANY", None),
			ToolChoice::Named(name) if compat.named_tool_choice && compat.forced_tool_choice => {
				("ANY", Some(json!([name])))
			},
			ToolChoice::Named(_) => {
				report(
					unsupported,
					"tool_choice.named",
					"provider compatibility disables named tool choice",
					choice.on_unsupported,
				)?;
				("AUTO", None)
			},
			_ => ("AUTO", None),
		};
		let mut config = Map::new();
		config.insert("mode".into(), Value::String(mode.into()));
		if let Some(allowed) = allowed {
			config.insert("allowedFunctionNames".into(), allowed);
		}
		body.insert("toolConfig".into(), json!({ "functionCallingConfig": config }));
	}
	Ok(())
}

fn reasoning_enabled(req: &ChatRequest) -> bool {
	req.thinking
		.as_ref()
		.is_some_and(|reasoning| match reasoning.value.effort {
			Some(Effort::Off) => false,
			Some(_) => true,
			None => reasoning.value.budget_tokens != Some(0),
		})
}

fn model_compat_bool(req: &ChatRequest, name: &str, reasoning_enabled: bool) -> Option<bool> {
	let compat = &req.model_policy.as_deref()?.compat;
	reasoning_enabled
		.then(|| compat.get_ns("wire", "when_thinking"))
		.flatten()
		.and_then(Value::as_object)
		.and_then(|overlay| overlay.get(name))
		.and_then(Value::as_bool)
		.or_else(|| compat.get_ns("wire", name).and_then(Value::as_bool))
}

fn encode_policy_thinking(
	reasoning: &omp_llm_types::Reasoning,
	policy: &ResolvedThinkingPolicy,
	unsupported: &mut Vec<Unsupported>,
	fallback: Fallback,
) -> Result<Option<Map<String, Value>>, Error> {
	if reasoning.effort == Some(Effort::Off) {
		if policy.suppress_when_off != Some(true) {
			report(
				unsupported,
				"thinking.effort",
				"resolved model policy does not define an explicit Google thinking-off shape",
				fallback,
			)?;
			return Ok(None);
		}
		let mut thinking = Map::new();
		thinking.insert("includeThoughts".into(), Value::Bool(false));
		match policy.mode {
			ResolvedThinkingMode::Budget => {
				thinking.insert("thinkingBudget".into(), Value::from(0));
			},
			ResolvedThinkingMode::GoogleLevel => {
				let Some(floor) = policy
					.efforts
					.iter()
					.copied()
					.find(|effort| *effort != Effort::Off)
				else {
					report(
						unsupported,
						"thinking.effort",
						"Google level-mode off suppression requires an advertised non-off floor",
						fallback,
					)?;
					return Ok(None);
				};
				let level = policy
					.effort_map
					.get(&floor)
					.map_or_else(|| thinking_level(floor), Str::as_str);
				thinking.insert("thinkingLevel".into(), Value::String(level.into()));
			},
			_ => {
				report(
					unsupported,
					"thinking",
					"resolved thinking mode has no Google GenerateContent projection",
					fallback,
				)?;
				return Ok(None);
			},
		}
		return Ok(Some(thinking));
	}

	let mut thinking = Map::new();
	thinking.insert("includeThoughts".into(), Value::Bool(!reasoning.hide_summary.unwrap_or(false)));
	match policy.mode {
		ResolvedThinkingMode::GoogleLevel => {
			if reasoning.budget_tokens.is_some() {
				report(
					unsupported,
					"thinking.budget_tokens",
					"Google level-mode models do not accept a token thinking budget",
					fallback,
				)?;
			}
			if let Some(effort) = reasoning.effort {
				let level = policy
					.effort_map
					.get(&effort)
					.map_or_else(|| thinking_level(effort), Str::as_str);
				thinking.insert("thinkingLevel".into(), Value::String(level.into()));
			}
		},
		ResolvedThinkingMode::Budget => {
			if let Some(budget) = reasoning.budget_tokens {
				thinking.insert("thinkingBudget".into(), Value::from(budget));
			} else if let Some(effort) = reasoning.effort {
				let budget = if policy.effort_budgets.is_empty() {
					default_thinking_budget(effort)
				} else {
					policy.effort_budgets.get(&effort).copied()
				};
				if let Some(budget) = budget {
					thinking.insert("thinkingBudget".into(), Value::from(budget));
				} else {
					report(
						unsupported,
						"thinking.effort",
						"resolved budget-mode policy has no budget for the selected effort",
						fallback,
					)?;
				}
			}
		},
		_ => {
			report(
				unsupported,
				"thinking",
				"resolved thinking mode has no Google GenerateContent projection",
				fallback,
			)?;
			return Ok(None);
		},
	}
	Ok(Some(thinking))
}

const fn default_thinking_budget(effort: Effort) -> Option<u64> {
	match effort {
		Effort::Off => Some(0),
		Effort::Minimal => Some(1_024),
		Effort::Low => Some(2_048),
		Effort::Medium => Some(8_192),
		Effort::High => Some(16_384),
		Effort::XHigh | Effort::Max => None,
		_ => None,
	}
}

fn insert_option<T: serde::Serialize>(
	object: &mut Map<String, Value>,
	key: &str,
	value: Option<T>,
) {
	if let Some(value) = value {
		object.insert(
			key.into(),
			serde_json::to_value(value).expect("numeric sampling value serializes"),
		);
	}
}

const fn thinking_level(effort: Effort) -> &'static str {
	match effort {
		Effort::Off | Effort::Minimal => "MINIMAL",
		Effort::Low => "LOW",
		Effort::Medium => "MEDIUM",
		Effort::High | Effort::Max => "HIGH",
		Effort::XHigh => "THINKING_LEVEL_UNSPECIFIED",
		_ => "THINKING_LEVEL_UNSPECIFIED",
	}
}

fn report(
	unsupported: &mut Vec<Unsupported>,
	what: impl Into<Str>,
	detail: impl Into<Str>,
	fallback: Fallback,
) -> Result<(), Error> {
	let action = match fallback {
		Fallback::Ignore => UnsupportedAction::Dropped,
		Fallback::Emulate => UnsupportedAction::Emulated,
		Fallback::Error => {
			return Err(Error::Unsupported(vec![
				Unsupported::builder()
					.what(what.into())
					.detail(detail.into())
					.action(UnsupportedAction::Dropped)
					.build(),
			]));
		},
		_ => UnsupportedAction::Dropped,
	};
	unsupported.push(
		Unsupported::builder()
			.what(what.into())
			.detail(detail.into())
			.action(action)
			.build(),
	);
	Ok(())
}

#[derive(Default)]
struct GoogleDecodeState {
	next_index:         u32,
	open:               Option<(u32, StreamPartKind)>,
	usage:              Option<WireUsage>,
	response_id:        Option<Str>,
	parts:              BTreeMap<u32, DecodedPart>,
	part_props:         BTreeMap<u32, Props>,
	thought_signatures: BTreeMap<u32, Bytes>,
	props:              Props,
	completed:          bool,
}

enum DecodedPart {
	Text(String),
	Thinking(String),
	ToolCall { id: CallId, name: Str, args: Bytes, signature: Bytes },
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireUsage {
	#[serde(default, rename = "promptTokenCount")]
	prompt:         u64,
	#[serde(default, rename = "candidatesTokenCount")]
	candidates:     u64,
	#[serde(default, rename = "cachedContentTokenCount")]
	cached_content: u64,
	#[serde(default, rename = "thoughtsTokenCount")]
	thoughts:       u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireResponse {
	#[serde(default)]
	candidates:      Vec<WireCandidate>,
	response_id:     Option<Str>,
	usage_metadata:  Option<WireUsage>,
	prompt_feedback: Option<WirePromptFeedback>,
	error:           Option<WireError>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireCandidate {
	content:       Option<WireContent>,
	finish_reason: Option<Str>,
	#[serde(flatten)]
	metadata:      stream::CandidateMetadata,
}

#[derive(Deserialize)]
struct WireContent {
	#[serde(default)]
	parts: Vec<WirePart>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WirePart {
	text:                  Option<Str>,
	#[serde(default)]
	thought:               bool,
	thought_signature:     Option<Str>,
	function_call:         Option<WireFunctionCall>,
	executable_code:       Option<stream::ExecutableCode>,
	code_execution_result: Option<stream::CodeExecutionResult>,
}

#[derive(Deserialize)]
struct WireFunctionCall {
	id:   Option<Str>,
	name: Option<Str>,
	args: Option<Box<RawValue>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WirePromptFeedback {
	block_reason:         Option<Str>,
	block_reason_message: Option<Str>,
}

#[derive(Deserialize)]
struct WireError {
	code:    Option<u16>,
	message: Option<Str>,
	status:  Option<Str>,
}

fn decode_bytes(data: &[u8], state: &mut DecodeState) -> Result<SmallVec<TurnEvent, 2>, Error> {
	if data.is_empty() {
		return Ok(SmallVec::new());
	}
	if data == b"[DONE]" {
		return Ok(finish_stream(state));
	}
	let response: WireResponse = serde_json::from_slice(data).map_err(provider_error)?;
	decode_response(response, state)
}

fn decode_response(
	response: WireResponse,
	state: &mut DecodeState,
) -> Result<SmallVec<TurnEvent, 2>, Error> {
	let state = state.get_or_insert_with(GoogleDecodeState::default);
	if state.completed {
		return Ok(SmallVec::new());
	}
	if response.response_id.is_some() {
		state.response_id = response.response_id.clone();
	}
	let mut events = SmallVec::new();
	if let Some(error) = response.error {
		close_open(state, &mut events);
		state.completed = true;
		events.push(TurnEvent::Error(stream::stream_error(
			error.code,
			error.status.as_deref(),
			error.message.as_deref(),
		)));
		return Ok(events);
	}
	if let Some(feedback) = response.prompt_feedback
		&& feedback.block_reason.is_some()
	{
		close_open(state, &mut events);
		state.completed = true;
		let detail = feedback
			.block_reason_message
			.or(feedback.block_reason)
			.unwrap_or_else(|| "Google blocked the prompt".into());
		events.push(TurnEvent::Error(turn_error(detail)));
		return Ok(events);
	}
	if let Some(usage) = response.usage_metadata {
		state.usage = Some(usage);
	}
	let mut finish = None;
	for candidate in response.candidates {
		stream::retain_candidate_metadata(&mut state.props, candidate.metadata);
		if let Some(content) = candidate.content {
			for part in content.parts {
				if let Some(call) = part.function_call {
					close_open(state, &mut events);
					let Some(name) = call.name.filter(|name| !name.is_empty()) else {
						state.completed = true;
						events.push(TurnEvent::Error(turn_error(
							"Google functionCall is missing a non-empty name".into(),
						)));
						return Ok(events);
					};
					if part.thought_signature.as_ref().is_some_and(Str::is_empty) {
						state.completed = true;
						events.push(TurnEvent::Error(turn_error(
							"Google functionCall carried an empty thoughtSignature".into(),
						)));
						return Ok(events);
					}
					let index = state.next_index;
					state.next_index += 1;
					let id = CallId::new();
					let args = call.args.map_or_else(
						|| Bytes::from_static(b"{}"),
						|args| Bytes::copy_from_slice(args.get().as_bytes()),
					);
					let signature = part.thought_signature.map_or_else(Bytes::new, |signature| {
						Bytes::copy_from_slice(signature.as_bytes())
					});
					state.parts.insert(index, DecodedPart::ToolCall {
						id,
						name: name.clone(),
						args: args.clone(),
						signature: signature.clone(),
					});
					let _wire_id = call.id;
					events.push(TurnEvent::PartStart {
						index,
						kind: StreamPartKind::ToolCall,
						tool_call_id: Str::from(id.to_string()),
						tool_name: name,
					});
					events.push(TurnEvent::PartDelta { index, chunk: args });
					events.push(TurnEvent::PartEnd { index, signature: signature.clone() });
					if !signature.is_empty() {
						state.thought_signatures.insert(index, signature);
					}
				} else if let Some(text) = part.text {
					if text.is_empty() {
						if let Some(signature) = part.thought_signature
							&& let Some((index, _)) = state.open
						{
							state
								.thought_signatures
								.insert(index, Bytes::copy_from_slice(signature.as_bytes()));
						}
					} else {
						let kind = if part.thought {
							StreamPartKind::Thinking
						} else {
							StreamPartKind::Text
						};
						let index = ensure_open(state, kind, &mut events);
						events.push(TurnEvent::PartDelta {
							index,
							chunk: Bytes::copy_from_slice(text.as_bytes()),
						});
						match state.parts.get_mut(&index) {
							Some(DecodedPart::Text(output) | DecodedPart::Thinking(output)) => {
								output.push_str(&text);
							},
							Some(DecodedPart::ToolCall { .. }) | None => {
								return Err(Error::Provider("Google part state mismatch".into()));
							},
						}
						if let Some(signature) = part.thought_signature {
							state
								.thought_signatures
								.insert(index, Bytes::copy_from_slice(signature.as_bytes()));
						}
					}
				}
				if let Some(auxiliary) = part.executable_code.and_then(stream::executable_code) {
					push_auxiliary_text(state, auxiliary, &mut events);
				}
				if let Some(auxiliary) = part
					.code_execution_result
					.and_then(stream::code_execution_result)
				{
					push_auxiliary_text(state, auxiliary, &mut events);
				}
			}
		}
		finish = candidate.finish_reason.or(finish);
	}
	if let Some(finish) = finish {
		state.completed = true;
		close_open(state, &mut events);
		let mapped = match stream::finish_reason(&finish) {
			Ok(mapped) => mapped,
			Err(detail) => {
				events.push(TurnEvent::Error(turn_error(detail)));
				return Ok(events);
			},
		};
		let has_tool_calls = state
			.parts
			.values()
			.any(|part| matches!(part, DecodedPart::ToolCall { .. }));
		events.push(TurnEvent::Outcome(outcome(
			with_tool_use_precedence(mapped, has_tool_calls),
			state,
		)));
	}
	Ok(events)
}

pub(crate) fn decode_value(
	value: Value,
	state: &mut DecodeState,
) -> Result<SmallVec<TurnEvent, 2>, Error> {
	let encoded = serde_json::to_vec(&value).map_err(provider_error)?;
	let response: WireResponse = serde_json::from_slice(&encoded).map_err(provider_error)?;
	decode_response(response, state)
}

fn ensure_open(
	state: &mut GoogleDecodeState,
	kind: StreamPartKind,
	events: &mut SmallVec<TurnEvent, 2>,
) -> u32 {
	if let Some((index, open_kind)) = state.open {
		if open_kind == kind {
			return index;
		}
		close_open(state, events);
	}
	let index = state.next_index;
	state.next_index += 1;
	state.open = Some((index, kind));
	state.parts.insert(
		index,
		if kind == StreamPartKind::Thinking {
			DecodedPart::Thinking(String::new())
		} else {
			DecodedPart::Text(String::new())
		},
	);
	events.push(TurnEvent::PartStart {
		index,
		kind,
		tool_call_id: Str::default(),
		tool_name: Str::default(),
	});
	index
}

fn push_auxiliary_text(
	state: &mut GoogleDecodeState,
	auxiliary: stream::AuxiliaryText,
	events: &mut SmallVec<TurnEvent, 2>,
) {
	close_open(state, events);
	let index = ensure_open(state, StreamPartKind::Text, events);
	events.push(TurnEvent::PartDelta {
		index,
		chunk: Bytes::copy_from_slice(auxiliary.text.as_bytes()),
	});
	if let Some(DecodedPart::Text(output)) = state.parts.get_mut(&index) {
		output.push_str(&auxiliary.text);
	}
	let mut props = Props::default();
	props.insert_ns("google", "part_kind", json!(auxiliary.kind));
	props.insert_ns("google", "part_metadata", auxiliary.props);
	state.part_props.insert(index, props);
	close_open(state, events);
}

fn close_open(state: &mut GoogleDecodeState, events: &mut SmallVec<TurnEvent, 2>) {
	if let Some((index, _)) = state.open.take() {
		let signature = state
			.thought_signatures
			.get(&index)
			.cloned()
			.unwrap_or_default();
		events.push(TurnEvent::PartEnd { index, signature });
	}
}

pub(crate) fn finish_stream(state: &mut DecodeState) -> SmallVec<TurnEvent, 2> {
	let state = state.get_or_insert_with(GoogleDecodeState::default);
	if state.completed {
		return SmallVec::new();
	}
	state.completed = true;
	let mut events = SmallVec::new();
	close_open(state, &mut events);
	events.push(TurnEvent::Error(stream::incomplete_stream_error()));
	events
}
fn outcome(stop: StopReason, state: &GoogleDecodeState) -> ChatOutcome {
	let mut output = Vec::with_capacity(state.parts.len());
	for (index, part) in &state.parts {
		match part {
			DecodedPart::Text(text) if !text.is_empty() => {
				output.push(assistant_text_item(
					text,
					state.thought_signatures.get(index),
					state.part_props.get(index),
				));
			},
			DecodedPart::Thinking(text) if !text.is_empty() => {
				output.push(assistant_item(Part::Thinking(
					Thinking::builder()
						.text(Str::from(text.as_str()))
						.signature(
							state
								.thought_signatures
								.get(index)
								.cloned()
								.unwrap_or_default(),
						)
						.redacted(false)
						.build(),
				)));
			},
			DecodedPart::ToolCall { id, name, args, signature } => output.push(
				Item::builder()
					.seq(0)
					.kind(ItemKind::ToolCall(
						ToolCall::builder()
							.id(*id)
							.name(name.clone())
							.args_json(args.clone())
							.thought_signature(signature.clone())
							.build(),
					))
					.props(Props::default())
					.build(),
			),
			DecodedPart::Text(_) | DecodedPart::Thinking(_) => {},
		}
	}
	let mut props = Props::default();
	if let Some(response_id) = &state.response_id {
		props.insert_ns("google", "response_id", Value::String(response_id.to_string()));
	}
	for (key, value) in &state.props.0 {
		props.0.insert(key.clone(), value.clone());
	}
	ChatOutcome::builder()
		.output(output)
		.stop(stop)
		.maybe_usage(state.usage.map(|usage| {
			let mut detail = Props::default();
			detail.insert_ns("google", "thoughts_tokens", json!(usage.thoughts));
			Usage::builder()
				.input_tokens(usage.prompt)
				.output_tokens(usage.candidates.saturating_add(usage.thoughts))
				.cache_read_tokens(usage.cached_content)
				.cache_write_tokens(0)
				.accuracy(Accuracy::Exact)
				.detail(detail)
				.build()
		}))
		.maybe_cost(None)
		.unsupported(Vec::new())
		.maybe_revision(None)
		.provider(Str::from("google"))
		.model(Str::default())
		.props(props)
		.build()
}

fn assistant_item(part: Part) -> Item {
	Item::builder()
		.seq(0)
		.kind(ItemKind::Message(
			Message::builder()
				.role(Role::Assistant)
				.parts(vec![part])
				.build(),
		))
		.props(Props::default())
		.build()
}

fn assistant_text_item(text: &str, signature: Option<&Bytes>, extra: Option<&Props>) -> Item {
	let mut item = assistant_item(Part::Text(Str::from(text)));
	if let Some(signature) = signature {
		item.props.insert_ns(
			"google",
			"text_thought_signatures",
			json!([String::from_utf8_lossy(signature)]),
		);
	}
	if let Some(extra) = extra {
		for (key, value) in &extra.0 {
			item.props.0.insert(key.clone(), value.clone());
		}
	}
	item
}

fn turn_error(detail: Str) -> TurnError {
	TurnError::builder()
		.kind(TurnErrorKind::Upstream)
		.detail(detail)
		.unsupported(Vec::new())
		.retry_after_ms(0)
		.build()
}

fn serialize_preserving_args(value: &Value, req: &ChatRequest) -> Result<Vec<u8>, Error> {
	let mut body = serde_json::to_vec(value).map_err(provider_error)?;
	let mut replacements = SmallVec::<(std::ops::Range<usize>, &[u8]), 4>::new();
	let mut cursor = 0;
	for raw in req.thread.items.iter().filter_map(|item| match &item.kind {
		ItemKind::ToolCall(call) => Some(call.args_json.as_ref()),
		_ => None,
	}) {
		let relative = body[cursor..]
			.windows(b"\"args\":".len())
			.position(|window| window == b"\"args\":")
			.ok_or_else(|| {
				Error::Provider("serialized Google functionCall lost its args field".into())
			})?;
		let start = cursor + relative + b"\"args\":".len();
		let end = json_value_end(&body, start).ok_or_else(|| {
			Error::Provider("serialized Google functionCall has malformed args".into())
		})?;
		replacements.push((start..end, raw));
		cursor = end;
	}
	for (range, raw) in replacements.into_iter().rev() {
		body.splice(range, raw.iter().copied());
	}
	Ok(body)
}

fn json_value_end(input: &[u8], start: usize) -> Option<usize> {
	let first = *input.get(start)?;
	if first != b'{' && first != b'[' && first != b'"' {
		return input[start..]
			.iter()
			.position(|byte| matches!(byte, b',' | b'}' | b']'))
			.map(|length| start + length)
			.or(Some(input.len()));
	}
	let mut depth = 0_u32;
	let mut quoted = false;
	let mut escaped = false;
	for (offset, byte) in input[start..].iter().copied().enumerate() {
		if quoted {
			if escaped {
				escaped = false;
			} else if byte == b'\\' {
				escaped = true;
			} else if byte == b'"' {
				quoted = false;
				if first == b'"' && depth == 0 {
					return Some(start + offset + 1);
				}
			}
			continue;
		}
		match byte {
			b'"' => quoted = true,
			b'{' | b'[' => depth += 1,
			b'}' | b']' => {
				depth = depth.checked_sub(1)?;
				if depth == 0 {
					return Some(start + offset + 1);
				}
			},
			_ => {},
		}
	}
	None
}

fn signature_string(signature: &Bytes) -> Result<String, Error> {
	String::from_utf8(signature.to_vec()).map_err(provider_error)
}

#[cold]
fn provider_error(error: impl std::fmt::Display) -> Error {
	Error::Provider(Str::from(error.to_string()))
}

#[cfg(test)]
mod tests {
	use std::{collections::BTreeMap, sync::Arc};

	use bytes::Bytes;
	use omp_core::Str;
	use omp_llm_catalog::{
		compat::{Compat, ReasoningWireFormat, ToolSchemaFlavor},
		provider::{TransportId, load_builtin},
	};
	use omp_llm_transport::{DecodeState, Transport};
	use omp_llm_types::{
		ChatRequest, Effort, Fallback, Feature, Item, ItemKind, Message, Part, Props, Reasoning,
		ResolvedModelPolicy, ResolvedThinkingMode, ResolvedThinkingPolicy, Role, Sampling,
		StopReason, Thread, ToolCall, ToolResult, TurnEvent, ids::CallId,
	};
	use serde_json::{Value, json};
	use smallvec::smallvec;

	use super::{
		GoogleCodec, GoogleDecodeState, GoogleVariant, decode_bytes, encode_request, finish_stream,
		vertex_stream_url,
	};

	fn item(kind: ItemKind) -> Item {
		Item::builder()
			.seq(0)
			.kind(kind)
			.props(Props::default())
			.build()
	}

	fn request(items: Vec<Item>) -> ChatRequest {
		ChatRequest::builder()
			.model("gemini-3.5-flash".into())
			.thread(Thread::builder().items(items).build())
			.tools(Vec::new())
			.build()
	}

	fn compat() -> Compat {
		let mut compat = Compat::default();
		compat.reasoning_wire_format = ReasoningWireFormat::Google;
		compat.tool_schema_flavor = ToolSchemaFlavor::Google;
		compat
	}

	#[test]
	fn contiguous_parallel_function_responses_merge_into_one_user_content() {
		let first: CallId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
		let second: CallId = "01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap();
		let req = request(vec![
			item(ItemKind::ToolCall(
				ToolCall::builder()
					.id(first)
					.name("read".into())
					.args_json(Bytes::from_static(br#"{"path":"a"}"#))
					.thought_signature(Bytes::new())
					.build(),
			)),
			item(ItemKind::ToolCall(
				ToolCall::builder()
					.id(second)
					.name("read".into())
					.args_json(Bytes::from_static(br#"{"path":"b"}"#))
					.thought_signature(Bytes::new())
					.build(),
			)),
			item(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(first)
					.name("read".into())
					.parts(vec![Part::Text("A".into())])
					.is_error(false)
					.build(),
			)),
			item(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(second)
					.name("read".into())
					.parts(vec![Part::Text("B".into())])
					.is_error(false)
					.build(),
			)),
		]);
		let (body, _) = encode_request(&req, &compat(), GoogleVariant::GEN_AI).unwrap();
		let contents = body["contents"].as_array().unwrap();
		assert_eq!(contents.len(), 2);
		let responses = contents[1]["parts"].as_array().unwrap();
		assert_eq!(responses.len(), 2);
		assert!(
			responses
				.iter()
				.all(|part| part.get("functionResponse").is_some())
		);
	}

	#[test]
	fn vertex_omits_function_ids_instead_of_sending_empty_fields() {
		let id: CallId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
		let req = request(vec![
			item(ItemKind::ToolCall(
				ToolCall::builder()
					.id(id)
					.name("read".into())
					.args_json(Bytes::from_static(b"{}"))
					.thought_signature(Bytes::new())
					.build(),
			)),
			item(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(id)
					.name("read".into())
					.parts(vec![Part::Text("ok".into())])
					.is_error(false)
					.build(),
			)),
		]);
		let (body, _) = encode_request(&req, &compat(), GoogleVariant::VERTEX).unwrap();
		assert!(
			body["contents"][0]["parts"][0]["functionCall"]
				.get("id")
				.is_none()
		);

		assert!(
			body["contents"][1]["parts"][0]["functionResponse"]
				.get("id")
				.is_none()
		);
		assert_eq!(GoogleCodec::vertex().id(), TransportId::GoogleVertex);
	}

	#[test]
	fn vertex_url_expands_region_and_adc_resource_path() {
		let providers = load_builtin().unwrap();
		let provider = &providers["google-vertex"];
		let url =
			vertex_stream_url(provider, "my-project", "us-central1", "gemini-3.5-flash").unwrap();
		assert_eq!(
			url,
			"https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-3.5-flash:streamGenerateContent?alt=sse"
		);
	}

	#[test]
	fn request_tool_arguments_keep_their_exact_json_lexeme() {
		let id: CallId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
		let req = request(vec![item(ItemKind::ToolCall(
			ToolCall::builder()
				.id(id)
				.name("lookup".into())
				.args_json(Bytes::from_static(br#"{ "n" : 1.00, "x" : "\u0061" }"#))
				.thought_signature(Bytes::new())
				.build(),
		))]);
		let (body, _) = GoogleCodec::gen_ai().encode(&req, &compat()).unwrap();
		assert!(
			body
				.windows(br#"{ "n" : 1.00, "x" : "\u0061" }"#.len())
				.any(|window| window == br#"{ "n" : 1.00, "x" : "\u0061" }"#)
		);
	}

	#[test]
	fn thought_signature_replays_verbatim_and_is_retained_on_decode() {
		let id: CallId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
		let signature = Bytes::from_static(b"function-call-sig_REDACTED");
		let req = request(vec![item(ItemKind::ToolCall(
			ToolCall::builder()
				.id(id)
				.name("lookup".into())
				.args_json(Bytes::from_static(br#"{"q":"x"}"#))
				.thought_signature(signature.clone())
				.build(),
		))]);
		let (body, _) = encode_request(&req, &compat(), GoogleVariant::GEN_AI).unwrap();
		assert_eq!(body["contents"][0]["parts"][0]["thoughtSignature"], "function-call-sig_REDACTED");

		let mut state = DecodeState::default();
		decode_bytes(br#"{"candidates":[{"content":{"parts":[{"functionCall":{"id":"01ARZ3NDEKTSV4RRFFQ69G5FAV","name":"lookup","args":{"q":"x"}},"thoughtSignature":"function-call-sig_REDACTED"}]}}]}"#, &mut state).unwrap();
		let decoded = state.get_or_insert_with(GoogleDecodeState::default);
		assert_eq!(decoded.thought_signatures[&0], signature);
	}

	#[test]
	fn signed_text_replays_on_text_part_and_tool_call_takes_stop_precedence() {
		let mut state = DecodeState::default();
		decode_bytes(
			br#"{"candidates":[{"content":{"parts":[{"text":"Hello","thoughtSignature":"text-sig_REDACTED"}]}}]}"#,
			&mut state,
		)
		.unwrap();
		let events = decode_bytes(
			br#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup","args":{}}}]},"finishReason":"STOP"}]}"#,
			&mut state,
		)
		.unwrap();
		let TurnEvent::Outcome(outcome) = events.last().unwrap() else {
			panic!("missing outcome")
		};
		assert_eq!(outcome.stop, StopReason::ToolUse);
		let text = outcome
			.output
			.iter()
			.find(|item| matches!(item.kind, ItemKind::Message(_)))
			.unwrap();
		assert_eq!(
			text.props.get_ns("google", "text_thought_signatures"),
			Some(&json!(["text-sig_REDACTED"]))
		);

		let replay = request(outcome.output.clone());
		let (body, _) = encode_request(&replay, &compat(), GoogleVariant::CCA).unwrap();
		let text_part = &body["contents"][0]["parts"][0];
		assert_eq!(text_part["text"], "Hello");
		assert_eq!(text_part["thoughtSignature"], "text-sig_REDACTED");
		assert!(text_part.get("thought").is_none());
	}

	#[test]
	fn usage_metadata_maps_to_terminal_outcome() {
		let mut state = DecodeState::default();
		let events = decode_bytes(
			br#"{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"cachedContentTokenCount":6,"thoughtsTokenCount":2}}"#,
			&mut state,
		).unwrap();
		let TurnEvent::Outcome(outcome) = events.last().unwrap() else {
			panic!("missing outcome")
		};
		let usage = outcome.usage.as_ref().unwrap();
		assert_eq!((usage.input_tokens, usage.output_tokens, usage.cache_read_tokens), (10, 7, 6));
		assert_eq!(usage.detail.get_ns("google", "thoughts_tokens"), Some(&json!(2)));
	}

	#[test]
	fn recorded_thought_tool_usage_fixture_preserves_signatures_and_usage() {
		let fixture = include_str!("../tests/fixtures/google_genai/stream.thought_tool_usage.sse");
		let mut state = DecodeState::default();
		let mut all_events = Vec::new();
		for payload in fixture
			.lines()
			.filter_map(|line| line.strip_prefix("data: "))
		{
			all_events.extend(decode_bytes(payload.as_bytes(), &mut state).unwrap());
		}
		let mut outcomes = all_events.iter().filter_map(|event| match event {
			TurnEvent::Outcome(outcome) => Some(outcome),
			_ => None,
		});
		let outcome = outcomes.next().expect("fixture has a terminal outcome");
		assert!(outcomes.next().is_none(), "fixture has exactly one terminal outcome");
		let tool = outcome
			.output
			.iter()
			.find_map(|item| match &item.kind {
				ItemKind::ToolCall(call) => Some(call),
				_ => None,
			})
			.expect("fixture has a tool call");
		assert_eq!(tool.args_json, Bytes::from_static(br#"{"city":"SF"}"#));
		assert_eq!(tool.thought_signature, Bytes::from_static(b"toolcall-sig_REDACTED"));
		let usage = outcome.usage.as_ref().expect("fixture has usage");
		assert_eq!((usage.input_tokens, usage.output_tokens, usage.cache_read_tokens), (14, 4, 8));
	}

	#[test]
	fn fixture_parallel_response_shape_matches() {
		let fixture: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/google_genai/request.parallel_function_responses.json"
		))
		.unwrap();
		assert_eq!(
			fixture["wire_body"]["contents"][1]["parts"]
				.as_array()
				.unwrap()
				.len(),
			2
		);
	}

	#[test]
	fn system_instruction_is_hoisted_out_of_contents() {
		let req = request(vec![item(ItemKind::Message(
			Message::builder()
				.role(Role::System)
				.parts(vec![Part::Text(Str::from("be concise"))])
				.build(),
		))]);
		let (body, _) = encode_request(&req, &compat(), GoogleVariant::GEN_AI).unwrap();
		assert_eq!(body["systemInstruction"]["parts"][0]["text"], "be concise");
		assert_eq!(body["contents"], json!([]));
	}

	#[test]
	fn semantic_fixture_retains_tools_grounding_code_and_one_terminal() {
		let fixture = include_str!("../tests/fixtures/google_genai/stream.semantic_parity.sse");
		let mut state = DecodeState::default();
		let mut events = Vec::new();
		for payload in fixture
			.lines()
			.filter_map(|line| line.strip_prefix("data: "))
		{
			events.extend(decode_bytes(payload.as_bytes(), &mut state).unwrap());
		}
		assert_eq!(
			events
				.iter()
				.filter(|event| matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)))
				.count(),
			1
		);
		let outcome = events
			.iter()
			.find_map(|event| match event {
				TurnEvent::Outcome(outcome) => Some(outcome),
				_ => None,
			})
			.expect("semantic fixture has an outcome");
		assert_eq!(outcome.stop, StopReason::ToolUse);
		assert!(
			outcome
				.props
				.get_ns("google", "grounding_metadata")
				.is_some()
		);
		assert!(
			outcome
				.props
				.get_ns("google", "citation_metadata")
				.is_some()
		);
		assert!(outcome.props.get_ns("google", "safety_ratings").is_some());
		let tool = outcome.output.iter().find_map(|item| match &item.kind {
			ItemKind::ToolCall(call) => Some(call),
			_ => None,
		});
		assert_eq!(
			tool.expect("semantic fixture has tool").thought_signature,
			Bytes::from_static(b"dG9vbC1zaWc=")
		);
		let auxiliary = outcome
			.output
			.iter()
			.filter_map(|item| {
				item
					.props
					.get_ns("google", "part_kind")
					.and_then(Value::as_str)
			})
			.collect::<Vec<_>>();
		assert_eq!(auxiliary, vec!["executable_code", "code_execution_result"]);
		assert!(finish_stream(&mut state).is_empty(), "terminal is unique after stream end");
	}

	#[test]
	fn malformed_tool_chunk_terminates_once_before_emitting_a_call() {
		let mut state = DecodeState::default();
		let events = decode_bytes(
			br#"{"candidates":[{"content":{"parts":[{"functionCall":{"args":{}},"thoughtSignature":"sig"}]}}]}"#,
			&mut state,
		)
		.unwrap();
		assert!(matches!(events.as_slice(), [TurnEvent::Error(_)]));
		assert!(finish_stream(&mut state).is_empty());
		let malformed = include_str!("../tests/fixtures/google_genai/stream.malformed.sse")
			.trim()
			.strip_prefix("data: ")
			.unwrap();
		assert!(decode_bytes(malformed.as_bytes(), &mut DecodeState::default()).is_err());
	}

	#[test]
	fn dropped_stream_without_finish_is_one_terminal_error() {
		let mut state = DecodeState::default();
		let events = decode_bytes(
			br#"{"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}"#,
			&mut state,
		)
		.unwrap();
		assert!(
			!events
				.iter()
				.any(|event| matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)))
		);
		assert!(matches!(finish_stream(&mut state).as_slice(), [
			TurnEvent::PartEnd { .. },
			TurnEvent::Error(_)
		]));
		assert!(finish_stream(&mut state).is_empty());
	}
	#[test]
	fn resolved_policy_selects_gemini_level_or_budget_without_changing_legacy_shape() {
		let mut legacy = request(Vec::new());
		legacy.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::High).build())
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		let (legacy_body, _) = encode_request(&legacy, &compat(), GoogleVariant::GEN_AI).unwrap();
		assert_eq!(
			legacy_body["generationConfig"]["thinkingConfig"],
			json!({"includeThoughts": true, "thinkingLevel": "HIGH"})
		);

		let mut level = legacy.clone();
		level.model_policy = Some(Arc::new(ResolvedModelPolicy {
			thinking: Some(ResolvedThinkingPolicy {
				mode:              ResolvedThinkingMode::GoogleLevel,
				efforts:           smallvec![
					Effort::Minimal,
					Effort::Low,
					Effort::Medium,
					Effort::High
				],
				default_effort:    None,
				effort_map:        BTreeMap::new(),
				effort_routing:    BTreeMap::new(),
				effort_budgets:    BTreeMap::new(),
				supports_display:  None,
				suppress_when_off: None,
				requires_effort:   Some(true),
			}),
			..ResolvedModelPolicy::default()
		}));
		let mut budget = legacy;
		budget.model_policy = Some(Arc::new(ResolvedModelPolicy {
			thinking: Some(ResolvedThinkingPolicy {
				mode:              ResolvedThinkingMode::Budget,
				efforts:           smallvec![
					Effort::Minimal,
					Effort::Low,
					Effort::Medium,
					Effort::High
				],
				default_effort:    None,
				effort_map:        BTreeMap::new(),
				effort_routing:    BTreeMap::new(),
				effort_budgets:    BTreeMap::new(),
				supports_display:  None,
				suppress_when_off: None,
				requires_effort:   None,
			}),
			..ResolvedModelPolicy::default()
		}));

		let (level_body, _) = encode_request(&level, &compat(), GoogleVariant::GEN_AI).unwrap();
		let (budget_body, _) = encode_request(&budget, &compat(), GoogleVariant::GEN_AI).unwrap();
		assert_eq!(
			level_body["generationConfig"]["thinkingConfig"],
			json!({"includeThoughts": true, "thinkingLevel": "HIGH"})
		);
		assert_eq!(
			budget_body["generationConfig"]["thinkingConfig"],
			json!({"includeThoughts": true, "thinkingBudget": 16_384})
		);
	}

	#[test]
	fn budget_policy_uses_only_exact_sparse_entries_and_keeps_xhigh_distinct_from_max() {
		let policy = Arc::new(ResolvedModelPolicy {
			thinking: Some(ResolvedThinkingPolicy {
				mode:              ResolvedThinkingMode::Budget,
				efforts:           smallvec![
					Effort::Low,
					Effort::Medium,
					Effort::High,
					Effort::XHigh,
					Effort::Max
				],
				default_effort:    None,
				effort_map:        BTreeMap::new(),
				effort_routing:    BTreeMap::new(),
				effort_budgets:    BTreeMap::from([
					(Effort::Low, 1_001),
					(Effort::High, 10_001),
					(Effort::XHigh, 32_001),
					(Effort::Max, 64_001),
				]),
				supports_display:  None,
				suppress_when_off: None,
				requires_effort:   None,
			}),
			..ResolvedModelPolicy::default()
		});
		for (effort, expected) in [
			(Effort::Low, Some(1_001)),
			(Effort::Medium, None),
			(Effort::High, Some(10_001)),
			(Effort::XHigh, Some(32_001)),
			(Effort::Max, Some(64_001)),
		] {
			let mut req = request(Vec::new());
			req.model_policy = Some(Arc::clone(&policy));
			req.thinking = Some(
				Feature::builder()
					.value(Reasoning::builder().effort(effort).build())
					.on_unsupported(Fallback::Ignore)
					.build(),
			);
			let (body, unsupported) = encode_request(&req, &compat(), GoogleVariant::VERTEX).unwrap();
			assert_eq!(
				body["generationConfig"]["thinkingConfig"]
					.get("thinkingBudget")
					.and_then(Value::as_u64),
				expected
			);
			assert_eq!(
				unsupported
					.iter()
					.any(|entry| entry.what == "thinking.effort"),
				expected.is_none()
			);
		}
	}

	#[test]
	fn model_sampling_overlay_applies_only_while_reasoning_is_enabled() {
		let mut policy = ResolvedModelPolicy {
			thinking: Some(ResolvedThinkingPolicy {
				mode:              ResolvedThinkingMode::GoogleLevel,
				efforts:           smallvec![
					Effort::Minimal,
					Effort::Low,
					Effort::Medium,
					Effort::High
				],
				default_effort:    None,
				effort_map:        BTreeMap::new(),
				effort_routing:    BTreeMap::new(),
				effort_budgets:    BTreeMap::new(),
				supports_display:  None,
				suppress_when_off: Some(true),
				requires_effort:   None,
			}),
			omit_max_output_tokens: Some(true),
			..ResolvedModelPolicy::default()
		};
		policy
			.compat
			.insert_ns("wire", "supports_sampling_params", Value::Bool(true));
		policy
			.compat
			.insert_ns("wire", "when_thinking", json!({"supports_sampling_params": false}));
		let mut req = request(Vec::new());
		req.model_policy = Some(Arc::new(policy));
		req.sampling = Some(
			Sampling::builder()
				.temperature(0.25)
				.max_output_tokens(4_096)
				.build(),
		);
		req.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::High).build())
				.on_unsupported(Fallback::Error)
				.build(),
		);
		let (thinking_body, thinking_unsupported) =
			encode_request(&req, &compat(), GoogleVariant::GEN_AI).unwrap();
		assert!(
			thinking_body["generationConfig"]
				.get("temperature")
				.is_none()
		);
		assert!(
			thinking_body["generationConfig"]
				.get("maxOutputTokens")
				.is_none()
		);
		assert!(
			thinking_unsupported
				.iter()
				.any(|entry| entry.what == "sampling.temperature")
		);

		req.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::Off).build())
				.on_unsupported(Fallback::Error)
				.build(),
		);
		let (off_body, off_unsupported) =
			encode_request(&req, &compat(), GoogleVariant::GEN_AI).unwrap();
		assert_eq!(off_body["generationConfig"]["temperature"], 0.25);
		assert!(
			!off_unsupported
				.iter()
				.any(|entry| entry.what == "sampling.temperature")
		);
	}
}
