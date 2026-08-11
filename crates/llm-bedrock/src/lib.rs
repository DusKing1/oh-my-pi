//! Amazon Bedrock `ConverseStream` wire codec and model discovery.
//!
//! AWS `EventStream` framing and `SigV4` request mutation intentionally remain
//! in the shared Bedrock infrastructure; this crate owns Converse JSON and the
//! `ListFoundationModels` control-plane listing in [`discovery`], which
//! attaches non-secret signing context but never signs.

pub mod discovery;

use std::{
	borrow::Cow,
	collections::{BTreeMap, BTreeSet},
};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use bytes::Bytes;
use omp_core::Str;
use omp_llm_catalog::{
	compat::{CacheControlFormat, Compat},
	provider::TransportId,
};
use omp_llm_transport::{DecodeState, Frame, Transport};
use omp_llm_types::{
	Accuracy, CacheRetention, ChatOutcome, ChatRequest, Effort, Error, Fallback, Item, ItemKind,
	Message, Part, PromptCacheBreakpoint, PromptCacheMode, Props, ResolvedModelPolicy,
	ResolvedThinkingMode, Role, StopReason, StreamPartKind, Thinking, ToolCall, ToolChoice,
	TurnError, TurnErrorKind, TurnEvent, Unsupported, UnsupportedAction, Usage,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use smallvec::{SmallVec, smallvec};

/// Reserved tool used only when Bedrock requires `toolConfig` for historical
/// tool blocks but the current turn exposes no tools.
pub const NO_TOOLS_SENTINEL_NAME: &str = "__no_tools__";

/// Codec for Amazon Bedrock's model-independent Converse streaming API.
#[derive(Clone, Copy, Debug, Default)]
pub struct BedrockConverseCodec;

impl Transport for BedrockConverseCodec {
	fn id(&self) -> TransportId {
		TransportId::BedrockConverse
	}

	fn encode(
		&self,
		req: &ChatRequest,
		compat: &Compat,
	) -> Result<(Bytes, Vec<Unsupported>), Error> {
		let (body, unsupported) = encode_request(req, compat)?;
		serde_json::to_vec(&body)
			.map(Bytes::from)
			.map(|body| (body, unsupported))
			.map_err(provider_error)
	}

	fn decode(
		&self,
		frame: Frame<'_>,
		state: &mut DecodeState,
	) -> Result<SmallVec<TurnEvent, 2>, Error> {
		match frame {
			Frame::Data(data) | Frame::Event { data, .. } => decode_event(data, state),
			Frame::Done => finish_stream(state),
			_ => Ok(SmallVec::new()),
		}
	}
}

fn encode_request(req: &ChatRequest, compat: &Compat) -> Result<(Value, Vec<Unsupported>), Error> {
	let mut unsupported = Vec::new();
	let thinking_enabled = req
		.thinking
		.as_ref()
		.is_some_and(|feature| feature.value.effort != Some(Effort::Off))
		|| policy_bool(req.model_policy.as_deref(), "requires_thinking_enabled", false) == Some(true);
	let escape_tool_names =
		policy_bool(req.model_policy.as_deref(), "escape_builtin_tool_names", thinking_enabled)
			== Some(true);
	let explicit_cache = req.cache.as_ref().is_some_and(|cache| {
		cache.retention != Some(CacheRetention::None)
			&& (cache.mode == Some(PromptCacheMode::Explicit)
				|| (cache.mode.is_none()
					&& compat.cache_control_format == CacheControlFormat::Anthropic))
			&& cache.breakpoint != Some(PromptCacheBreakpoint::None)
	});
	let requested_long_cache = req
		.provider_options
		.as_ref()
		.and_then(|options| options.get_ns("amazon-bedrock", "cache_ttl"))
		.and_then(Value::as_str)
		== Some("1h")
		|| req.model_policy.is_some()
			&& req
				.cache
				.as_ref()
				.is_some_and(|cache| cache.retention == Some(CacheRetention::Long));
	let supports_long_cache =
		policy_bool(req.model_policy.as_deref(), "supports_long_cache_retention", thinking_enabled)
			!= Some(false);
	let cache_ttl = (requested_long_cache && supports_long_cache).then_some("1h");
	if requested_long_cache && !supports_long_cache {
		unsupported.push(emulated(
			"cache.retention",
			"this Bedrock model does not support one-hour cache retention; the default TTL was used",
		));
	}
	let mut messages = Vec::<Value>::new();
	let mut system = Vec::<Value>::new();
	let mut history_has_tools = false;
	let mut wire_ids = BTreeMap::<String, String>::new();
	let mut saw_conversation = false;
	let mid_system_policy = policy_bool(
		req.model_policy.as_deref(),
		"supports_mid_conversation_system",
		thinking_enabled,
	);
	if policy_bool(
		req.model_policy.as_deref(),
		"supports_eager_tool_input_streaming",
		thinking_enabled,
	) == Some(true)
		&& !req.tools.is_empty()
	{
		unsupported.push(dropped(
			"tools.eager_input_streaming",
			"Bedrock Converse toolSpec has no eager tool-input streaming field",
		));
	}

	for item in &req.thread.items {
		match &item.kind {
			ItemKind::Message(message) => match message.role {
				Role::System => {
					if saw_conversation && mid_system_policy.is_some() {
						unsupported.push(emulated(
							"thread.system.position",
							"Bedrock Converse hoists mid-conversation system content to its system field",
						));
					}
					for part in &message.parts {
						if let Part::Text(text) = part
							&& !text.trim().is_empty()
						{
							system.push(json!({ "text": text }));
						} else if !matches!(part, Part::Text(_)) {
							unsupported.push(dropped(
								"thread.system.part",
								"Bedrock system blocks only accept text",
							));
						}
					}
				},
				Role::User | Role::Assistant => {
					let role = if message.role == Role::Assistant {
						"assistant"
					} else {
						"user"
					};
					let content = encode_message_parts(
						message.role,
						&message.parts,
						&item.props,
						req.model.as_str(),
						req.model_policy.as_deref(),
						thinking_enabled,
						&mut unsupported,
					)?;
					append_message(&mut messages, role, content);
					saw_conversation = true;
				},
				_ => unsupported
					.push(dropped("thread.role", "Bedrock Converse does not support this role")),
			},
			ItemKind::ToolCall(call) => {
				history_has_tools = true;
				let canonical = call.id.to_string();
				let wire = call
					.provider_metadata
					.as_ref()
					.and_then(|props| props.get_ns("amazon-bedrock", "tool_use_id"))
					.and_then(Value::as_str)
					.filter(|id| !id.is_empty())
					.unwrap_or(canonical.as_str())
					.to_owned();
				wire_ids.insert(canonical, wire.clone());
				let input: Value = serde_json::from_slice(&call.args_json).map_err(|error| {
					Error::Provider(Str::from(format!("Bedrock tool arguments are not JSON: {error}")))
				})?;
				let name = escaped_tool_name(&call.name, escape_tool_names);
				append_message(&mut messages, "assistant", vec![json!({
					"toolUse": { "toolUseId": wire, "name": name, "input": input }
				})]);
				saw_conversation = true;
			},
			ItemKind::ToolResult(result) => {
				history_has_tools = true;
				let canonical = result.call_id.to_string();
				let wire = wire_ids
					.get(&canonical)
					.map_or(canonical.as_str(), String::as_str);
				let mut content = Vec::new();
				for part in &result.parts {
					if let Some(part) = encode_user_part(part, &mut unsupported)? {
						content.push(part);
					}
				}
				if content.is_empty() {
					content.push(json!({ "text": "" }));
				}
				append_message(&mut messages, "user", vec![json!({
					"toolResult": {
						"toolUseId": wire,
						"content": content,
						"status": if result.is_error { "error" } else { "success" }
					}
				})]);
				saw_conversation = true;
			},
			_ => unsupported.push(dropped("thread.item", "Bedrock Converse cannot project this item")),
		}
	}

	if explicit_cache {
		if let Some(message) = messages
			.iter_mut()
			.rev()
			.find(|message| message["role"] == "user")
			&& let Some(content) = message.get_mut("content").and_then(Value::as_array_mut)
		{
			content.push(cache_point(cache_ttl));
		}
		if !system.is_empty() {
			system.push(cache_point(cache_ttl));
		}
	}

	let mut body = Map::new();
	body.insert("messages".into(), Value::Array(messages));
	if !system.is_empty() {
		body.insert("system".into(), Value::Array(system));
	}
	encode_inference(req, &mut body, &mut unsupported);
	encode_tools(req, history_has_tools, escape_tool_names, &mut body, &mut unsupported)?;
	encode_reasoning(req, &mut body, &mut unsupported)?;
	report_unhandled(req, &mut unsupported)?;
	Ok((Value::Object(body), unsupported))
}

fn encode_message_parts(
	role: Role,
	parts: &[Part],
	item_props: &Props,
	model: &str,
	model_policy: Option<&ResolvedModelPolicy>,
	thinking_enabled: bool,
	unsupported: &mut Vec<Unsupported>,
) -> Result<Vec<Value>, Error> {
	let mut output = Vec::new();
	for part in parts {
		match (role, part) {
			(_, Part::Text(text)) if !text.trim().is_empty() => output.push(json!({ "text": text })),
			(Role::User, part) => {
				if let Some(part) = encode_user_part(part, unsupported)? {
					output.push(part);
				}
			},
			(Role::Assistant, Part::Thinking(thinking)) if !thinking.text.trim().is_empty() => {
				let source_model = item_props
					.get_ns("amazon-bedrock", "model")
					.or_else(|| item_props.get_ns("anthropic", "model"))
					.and_then(Value::as_str);
				let same_model = source_model.is_none_or(|source| source == model);
				let signing = policy_bool(model_policy, "signing_endpoint", thinking_enabled)
					.or_else(|| policy_bool(model_policy, "official_endpoint", thinking_enabled))
					.unwrap_or_else(|| is_claude(model));
				let replay_unsigned =
					policy_bool(model_policy, "replay_unsigned_thinking", thinking_enabled)
						.unwrap_or(!signing);
				if same_model && (!thinking.signature.is_empty() || replay_unsigned) {
					let mut reasoning = Map::new();
					reasoning.insert("text".into(), Value::String(thinking.text.to_string()));
					if !thinking.signature.is_empty() {
						reasoning.insert(
							"signature".into(),
							Value::String(String::from_utf8_lossy(&thinking.signature).into_owned()),
						);
					}
					output.push(json!({ "reasoningContent": { "reasoningText": reasoning } }));
				} else {
					output.push(json!({ "text": format!("<thinking>{}</thinking>", thinking.text) }));
					unsupported.push(dropped(
						"thread.assistant.thinking",
						"unsigned or foreign thinking was replayed as visible text",
					));
				}
			},
			(_, Part::Fallback(_)) => unsupported.push(dropped(
				"thread.message.fallback",
				"Bedrock Converse cannot replay model fallback markers",
			)),
			(_, Part::ServerTool(_)) => unsupported.push(dropped(
				"thread.message.server_tool",
				"Bedrock Converse cannot replay provider-hosted tool blocks",
			)),
			_ => {},
		}
	}
	Ok(output)
}

fn encode_user_part(
	part: &Part,
	unsupported: &mut Vec<Unsupported>,
) -> Result<Option<Value>, Error> {
	match part {
		Part::Text(text) => Ok((!text.trim().is_empty()).then(|| json!({ "text": text }))),
		Part::Blob(blob) => {
			let format = match blob.mime.as_str() {
				"image/jpeg" | "image/jpg" => "jpeg",
				"image/png" => "png",
				"image/gif" => "gif",
				"image/webp" => "webp",
				_ => {
					unsupported.push(dropped(
						"thread.message.image",
						"Bedrock Converse accepts JPEG, PNG, GIF, or WebP images",
					));
					return Ok(None);
				},
			};
			if blob.inline.is_empty() {
				return Err(Error::Provider("Bedrock image is not available inline".into()));
			}
			Ok(Some(json!({
				"image": { "format": format, "source": { "bytes": BASE64.encode(&blob.inline) } }
			})))
		},
		_ => {
			unsupported.push(dropped(
				"thread.message.part",
				"Bedrock user content cannot represent this part",
			));
			Ok(None)
		},
	}
}

fn append_message(messages: &mut Vec<Value>, role: &str, mut content: Vec<Value>) {
	if content.is_empty() {
		return;
	}
	if let Some(last) = messages.last_mut()
		&& last["role"] == role
		&& let Some(existing) = last.get_mut("content").and_then(Value::as_array_mut)
	{
		existing.append(&mut content);
		return;
	}
	messages.push(json!({ "role": role, "content": content }));
}

fn cache_point(ttl: Option<&str>) -> Value {
	match ttl {
		Some("1h") => json!({ "cachePoint": { "type": "default", "ttl": "1h" } }),
		_ => json!({ "cachePoint": { "type": "default" } }),
	}
}

fn encode_inference(
	req: &ChatRequest,
	body: &mut Map<String, Value>,
	unsupported: &mut Vec<Unsupported>,
) {
	let Some(sampling) = &req.sampling else {
		return;
	};
	let mut inference = Map::new();
	if let Some(max) = sampling.max_output_tokens {
		inference.insert("maxTokens".into(), json!(max));
	}
	if let Some(temperature) = sampling.temperature {
		inference.insert("temperature".into(), json!(temperature));
	}
	if let Some(top_p) = sampling.top_p {
		inference.insert("topP".into(), json!(top_p));
	}
	if let Some(stop) = &sampling.stop {
		inference.insert("stopSequences".into(), json!(stop));
	}
	for (present, what) in [
		(sampling.top_k.is_some(), "sampling.top_k"),
		(sampling.min_p.is_some(), "sampling.min_p"),
		(sampling.frequency_penalty.is_some(), "sampling.frequency_penalty"),
		(sampling.presence_penalty.is_some(), "sampling.presence_penalty"),
		(sampling.repetition_penalty.is_some(), "sampling.repetition_penalty"),
	] {
		if present {
			unsupported.push(dropped(
				what,
				"Bedrock Converse inferenceConfig does not expose this sampling control",
			));
		}
	}
	if !inference.is_empty() {
		body.insert("inferenceConfig".into(), Value::Object(inference));
	}
}

fn encode_tools(
	req: &ChatRequest,
	history_has_tools: bool,
	escape_tool_names: bool,
	body: &mut Map<String, Value>,
	unsupported: &mut Vec<Unsupported>,
) -> Result<(), Error> {
	let mut tools = Vec::with_capacity(req.tools.len().max(1));
	for tool in &req.tools {
		let schema: Value = serde_json::from_slice(&tool.schema_json).map_err(|error| {
			Error::Provider(Str::from(format!("Bedrock tool schema is not JSON: {error}")))
		})?;
		let name = escaped_tool_name(&tool.name, escape_tool_names);
		tools.push(json!({ "toolSpec": {
			"name": name,
			"description": tool.description,
			"inputSchema": { "json": schema }
		} }));
		if tool.strict.is_some() {
			unsupported.push(dropped("tools.strict", "Bedrock toolSpec has no strict-schema switch"));
		}
	}
	let mut choice = None;
	let requires_thinking =
		policy_bool(req.model_policy.as_deref(), "requires_thinking_enabled", false) == Some(true);
	if let Some(feature) = &req.tool_choice {
		choice = match &feature.value {
			ToolChoice::Auto => Some(json!({ "auto": {} })),
			ToolChoice::Required | ToolChoice::Named(_) if requires_thinking => {
				report(
					unsupported,
					"tool_choice",
					"this model requires thinking and cannot force a tool; auto was used",
					feature.on_unsupported,
				)?;
				Some(json!({ "auto": {} }))
			},
			ToolChoice::Required => Some(json!({ "any": {} })),
			ToolChoice::Named(name) => {
				let name = escaped_tool_name(name, escape_tool_names);
				Some(json!({ "tool": { "name": name } }))
			},
			ToolChoice::None => None,
			_ => None,
		};
		if matches!(feature.value, ToolChoice::None) && !req.tools.is_empty() {
			report(
				unsupported,
				"tool_choice",
				"Bedrock has no native none choice; tool definitions remain available for historical \
				 validation",
				feature.on_unsupported,
			)?;
		}
	}
	let sentinel = req.tools.is_empty() && history_has_tools;
	if sentinel {
		tools.push(json!({ "toolSpec": {
			"name": NO_TOOLS_SENTINEL_NAME,
			"description": "Placeholder required by Bedrock validation. Do not call; answer with text.",
			"inputSchema": { "json": { "type": "object", "properties": {} } }
		} }));
		choice = Some(json!({ "auto": {} }));
	}
	if !tools.is_empty() {
		let mut config = Map::new();
		config.insert("tools".into(), Value::Array(tools));
		if let Some(choice) = choice {
			config.insert("toolChoice".into(), choice);
		}
		body.insert("toolConfig".into(), Value::Object(config));
	}
	Ok(())
}

fn encode_reasoning(
	req: &ChatRequest,
	body: &mut Map<String, Value>,
	unsupported: &mut Vec<Unsupported>,
) -> Result<(), Error> {
	let model_policy = req.model_policy.as_deref();
	let requires_thinking =
		policy_bool(model_policy, "requires_thinking_enabled", false) == Some(true);
	let default_reasoning;
	let (reasoning, fallback) = if let Some(feature) = &req.thinking {
		(&feature.value, feature.on_unsupported)
	} else if requires_thinking {
		default_reasoning = omp_llm_types::Reasoning::builder().build();
		(&default_reasoning, Fallback::Ignore)
	} else {
		return Ok(());
	};
	let policy_thinking = model_policy.and_then(|policy| policy.thinking.as_ref());
	let options = req.provider_options.as_ref();
	let caller_mode = options
		.and_then(|props| props.get_ns("amazon-bedrock", "reasoning_mode"))
		.and_then(Value::as_str);
	let disable_adaptive =
		policy_bool(model_policy, "disable_adaptive_thinking", reasoning.effort != Some(Effort::Off))
			== Some(true);
	let adaptive = match caller_mode {
		Some("adaptive") => true,
		Some("budget") => false,
		_ => {
			policy_thinking
				.is_some_and(|thinking| thinking.mode == ResolvedThinkingMode::AnthropicAdaptive)
				|| policy_thinking.is_none() && is_adaptive_claude(req.model.as_str())
		},
	} && !disable_adaptive;
	let budget_effort = caller_mode.is_none()
		&& policy_thinking
			.is_some_and(|thinking| thinking.mode == ResolvedThinkingMode::AnthropicBudgetEffort);
	let forced_choice = req
		.tool_choice
		.as_ref()
		.is_some_and(|choice| matches!(choice.value, ToolChoice::Required | ToolChoice::Named(_)))
		&& !requires_thinking;
	let disable_on_tool_choice =
		policy_bool(model_policy, "disable_reasoning_on_tool_choice", true).unwrap_or(forced_choice);
	if forced_choice && disable_on_tool_choice {
		report(
			unsupported,
			"thinking",
			"Bedrock rejects reasoning with forced tool choice; reasoning was omitted",
			fallback,
		)?;
		return Ok(());
	}
	let requested_effort = if requires_thinking && reasoning.effort == Some(Effort::Off) {
		policy_thinking
			.and_then(|thinking| {
				thinking
					.default_effort
					.or_else(|| thinking.efforts.first().copied())
			})
			.unwrap_or(Effort::Low)
	} else {
		reasoning
			.effort
			.or_else(|| policy_thinking.and_then(|thinking| thinking.default_effort))
			.unwrap_or(Effort::Medium)
	};
	if requires_thinking && reasoning.effort == Some(Effort::Off) {
		report(
			unsupported,
			"thinking.effort",
			"this model requires thinking; its minimum effort was used",
			fallback,
		)?;
	}
	let effort = policy_thinking
		.and_then(|thinking| thinking.effort_map.get(&requested_effort))
		.map_or_else(
			|| {
				if policy_thinking.is_some() {
					bedrock_effort(requested_effort)
				} else {
					legacy_bedrock_effort(requested_effort)
				}
			},
			Str::as_str,
		);
	if reasoning.effort == Some(Effort::Off) && !requires_thinking {
		if adaptive && policy_thinking.is_some() {
			body.insert(
				"additionalModelRequestFields".into(),
				json!({ "output_config": { "effort": policy_thinking
					.and_then(|thinking| thinking.effort_map.get(&Effort::Low))
					.map_or("low", Str::as_str) } }),
			);
		}
		return Ok(());
	}
	let requested_display = options
		.and_then(|props| props.get_ns("amazon-bedrock", "thinking_display"))
		.and_then(Value::as_str)
		.filter(|value| matches!(*value, "summarized" | "omitted"))
		.unwrap_or_else(|| {
			if reasoning.hide_summary == Some(true) {
				"omitted"
			} else {
				"summarized"
			}
		});
	let supports_display = policy_thinking.map_or_else(
		|| {
			options
				.and_then(|props| props.get_ns("amazon-bedrock", "thinking_supports_display"))
				.and_then(Value::as_bool)
				.unwrap_or_else(|| supports_adaptive_display(req.model.as_str()))
		},
		|thinking| thinking.supports_display == Some(true),
	);
	if reasoning.hide_summary.is_some()
		&& policy_thinking.is_some_and(|thinking| thinking.supports_display != Some(true))
	{
		report(
			unsupported,
			"thinking.hide_summary",
			"this model does not support the adaptive thinking display control",
			fallback,
		)?;
	}
	if options
		.and_then(|props| props.get_ns("amazon-bedrock", "thinking_display"))
		.is_some()
		&& !supports_display
	{
		unsupported.push(dropped(
			"amazon-bedrock/thinking_display",
			"this model does not support the adaptive thinking display control",
		));
	}
	let mut fields = Map::new();
	if adaptive {
		let mut thinking = Map::new();
		thinking.insert("type".into(), Value::String("adaptive".into()));
		if supports_display {
			thinking.insert("display".into(), Value::String(requested_display.into()));
		}
		fields.insert("thinking".into(), Value::Object(thinking));
		fields.insert("output_config".into(), json!({ "effort": effort }));
	} else {
		let budget = reasoning.budget_tokens.unwrap_or_else(|| {
			policy_thinking
				.and_then(|thinking| thinking.effort_budgets.get(&requested_effort).copied())
				.unwrap_or_else(|| {
					if policy_thinking.is_some() {
						bedrock_budget(requested_effort)
					} else {
						legacy_bedrock_budget(requested_effort)
					}
				})
		});
		let mut thinking = Map::new();
		thinking.insert("type".into(), Value::String("enabled".into()));
		thinking.insert("budget_tokens".into(), json!(budget));
		if policy_thinking.is_none() || supports_display {
			thinking.insert("display".into(), Value::String(requested_display.into()));
		}
		fields.insert("thinking".into(), Value::Object(thinking));
		if budget_effort {
			fields.insert("output_config".into(), json!({ "effort": effort }));
		}
		if options
			.and_then(|props| props.get_ns("amazon-bedrock", "interleaved_thinking"))
			.and_then(Value::as_bool)
			== Some(true)
		{
			fields.insert("anthropic_beta".into(), json!(["interleaved-thinking-2025-05-14"]));
		}
	}
	body.insert("additionalModelRequestFields".into(), Value::Object(fields));
	Ok(())
}

fn report_unhandled(req: &ChatRequest, unsupported: &mut Vec<Unsupported>) -> Result<(), Error> {
	if let Some(feature) = &req.response_format {
		report(
			unsupported,
			"response_format",
			"Bedrock Converse has no portable structured-output field",
			feature.on_unsupported,
		)?;
	}
	if req.service_tier.is_some() || req.service_tier_by_family.is_some() {
		unsupported.push(dropped("service_tier", "Bedrock Converse does not expose service tiers"));
	}
	if req.task_budget.is_some() {
		unsupported.push(dropped("task_budget", "Task budgets are not sent to Bedrock"));
	}
	if req.responses_include.is_some() {
		unsupported.push(dropped(
			"responses_include",
			"OpenAI Responses include controls do not apply to Bedrock",
		));
	}
	if let Some(options) = &req.provider_options {
		for key in options.0.keys() {
			if key.starts_with("amazon-bedrock/")
				&& !matches!(
					key.as_str(),
					"amazon-bedrock/cache_ttl"
						| "amazon-bedrock/reasoning_mode"
						| "amazon-bedrock/thinking_display"
						| "amazon-bedrock/thinking_supports_display"
						| "amazon-bedrock/interleaved_thinking"
				) {
				unsupported
					.push(dropped(Str::from(key.as_str()), "Unknown Amazon Bedrock provider option"));
			}
		}
	}
	Ok(())
}

#[derive(Default)]
struct BedrockDecodeState {
	parts:         BTreeMap<u32, DecodedPart>,
	ignored:       BTreeSet<u32>,
	sentinel_seen: bool,
	usage:         Option<WireUsage>,
	metrics:       Option<Value>,
	stop:          Option<StopReason>,
	completed:     bool,
}

enum DecodedPart {
	Text {
		text:  String,
		ended: bool,
	},
	Thinking {
		text:      String,
		signature: Vec<u8>,
		ended:     bool,
	},
	Tool {
		id:      omp_llm_types::CallId,
		wire_id: Str,
		name:    Str,
		args:    Vec<u8>,
		ended:   bool,
	},
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireEvent {
	role:                Option<Str>,
	content_block_index: Option<u32>,
	start:               Option<WireStart>,
	delta:               Option<WireDelta>,
	stop_reason:         Option<Str>,
	usage:               Option<WireUsage>,
	metrics:             Option<Value>,
	#[serde(rename = "type")]
	kind:                Option<Str>,
	error:               Option<WireError>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireStart {
	tool_use: Option<WireToolStart>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireToolStart {
	#[serde(default)]
	tool_use_id: Str,
	#[serde(default)]
	name:        Str,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireDelta {
	text:              Option<Str>,
	tool_use:          Option<WireToolDelta>,
	reasoning_content: Option<WireReasoningDelta>,
}

#[derive(Deserialize)]
struct WireToolDelta {
	input: Option<Str>,
}

#[derive(Deserialize)]
struct WireReasoningDelta {
	text:      Option<Str>,
	signature: Option<Str>,
}

#[expect(
	clippy::struct_field_names,
	reason = "token suffixes mirror Bedrock's usage wire fields; aliases would obscure their units \
	          and require redundant serde renames"
)]
#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireUsage {
	#[serde(default)]
	input_tokens:             u64,
	#[serde(default)]
	output_tokens:            u64,
	#[serde(default)]
	total_tokens:             u64,
	#[serde(default)]
	cache_read_input_tokens:  u64,
	#[serde(default)]
	cache_write_input_tokens: u64,
}

#[derive(Deserialize)]
struct WireError {
	#[serde(default, rename = "type")]
	kind:    Str,
	#[serde(default)]
	message: Str,
}

fn decode_event(data: &[u8], state: &mut DecodeState) -> Result<SmallVec<TurnEvent, 2>, Error> {
	if data.is_empty() {
		return Ok(SmallVec::new());
	}
	let event: WireEvent = serde_json::from_slice(data).map_err(provider_error)?;
	let state = state.get_or_insert_with(BedrockDecodeState::default);
	if state.completed {
		return Ok(SmallVec::new());
	}
	if event.kind.as_deref() == Some("error") || event.error.is_some() {
		state.completed = true;
		let error = event.error.unwrap_or(WireError {
			kind:    "api_error".into(),
			message: "Bedrock stream error".into(),
		});
		return Ok(smallvec![TurnEvent::Error(turn_error(error.kind.as_str(), error.message))]);
	}
	if let Some(role) = event.role {
		if role != "assistant" {
			state.completed = true;
			return Ok(smallvec![TurnEvent::Error(turn_error(
				"api_error",
				"Bedrock messageStart role was not assistant".into()
			))]);
		}
		return Ok(SmallVec::new());
	}
	let mut events = SmallVec::new();
	if let (Some(index), Some(start)) = (event.content_block_index, event.start) {
		if let Some(tool) = start.tool_use {
			if tool.name == NO_TOOLS_SENTINEL_NAME {
				state.ignored.insert(index);
				state.sentinel_seen = true;
				return Ok(events);
			}
			if tool.name.is_empty() {
				state.completed = true;
				return Ok(smallvec![TurnEvent::Error(turn_error(
					"api_error",
					"Bedrock toolUse is missing a name".into()
				))]);
			}
			let id = omp_llm_types::CallId::new();
			state.parts.insert(index, DecodedPart::Tool {
				id,
				wire_id: tool.tool_use_id.clone(),
				name: tool.name.clone(),
				args: Vec::new(),
				ended: false,
			});
			events.push(TurnEvent::PartStart {
				index,
				kind: StreamPartKind::ToolCall,
				tool_call_id: Str::from(id.to_string()),
				tool_name: tool.name,
			});
		}
		return Ok(events);
	}
	if let (Some(index), Some(delta)) = (event.content_block_index, event.delta) {
		if state.ignored.contains(&index) {
			return Ok(events);
		}
		if let Some(text) = delta.text {
			ensure_part(state, index, StreamPartKind::Text, &mut events)?;
			if let Some(DecodedPart::Text { text: output, .. }) = state.parts.get_mut(&index) {
				output.push_str(&text);
			}
			events
				.push(TurnEvent::PartDelta { index, chunk: Bytes::copy_from_slice(text.as_bytes()) });
		} else if let Some(tool) = delta.tool_use {
			let chunk = tool.input.unwrap_or_default();
			let Some(DecodedPart::Tool { args, .. }) = state.parts.get_mut(&index) else {
				return Err(Error::Provider(
					"Bedrock toolUse delta arrived before contentBlockStart".into(),
				));
			};
			args.extend_from_slice(chunk.as_bytes());
			events
				.push(TurnEvent::PartDelta { index, chunk: Bytes::copy_from_slice(chunk.as_bytes()) });
		} else if let Some(reasoning) = delta.reasoning_content {
			ensure_part(state, index, StreamPartKind::Thinking, &mut events)?;
			let Some(DecodedPart::Thinking { text, signature, .. }) = state.parts.get_mut(&index)
			else {
				return Err(Error::Provider("Bedrock reasoning state mismatch".into()));
			};
			if let Some(chunk) = reasoning.text {
				text.push_str(&chunk);
				events.push(TurnEvent::PartDelta {
					index,
					chunk: Bytes::copy_from_slice(chunk.as_bytes()),
				});
			}
			if let Some(chunk) = reasoning.signature {
				signature.extend_from_slice(chunk.as_bytes());
			}
		}
		return Ok(events);
	}
	if let Some(index) = event.content_block_index {
		if state.ignored.remove(&index) {
			return Ok(events);
		}
		if let Some(part) = state.parts.get_mut(&index) {
			let signature = match part {
				DecodedPart::Thinking { signature, ended, .. } => {
					*ended = true;
					Bytes::copy_from_slice(signature)
				},
				DecodedPart::Text { ended, .. } | DecodedPart::Tool { ended, .. } => {
					*ended = true;
					Bytes::new()
				},
			};
			events.push(TurnEvent::PartEnd { index, signature });
		}
		return Ok(events);
	}
	if let Some(reason) = event.stop_reason {
		state.stop = Some(if state.sentinel_seen && reason == "tool_use" {
			StopReason::EndTurn
		} else {
			map_stop(reason.as_str()).ok_or_else(|| {
				Error::Provider(Str::from(format!(
					"Bedrock generation failed with stop reason: {reason}"
				)))
			})?
		});
		if state.usage.is_some() {
			return complete(state, events);
		}
		return Ok(events);
	}
	if let Some(usage) = event.usage {
		state.usage = Some(usage);
		state.metrics = event.metrics;
		if state.stop.is_some() {
			return complete(state, events);
		}
	}
	Ok(events)
}

fn ensure_part(
	state: &mut BedrockDecodeState,
	index: u32,
	kind: StreamPartKind,
	events: &mut SmallVec<TurnEvent, 2>,
) -> Result<(), Error> {
	if state.parts.contains_key(&index) {
		return Ok(());
	}
	let part = match kind {
		StreamPartKind::Text => DecodedPart::Text { text: String::new(), ended: false },
		StreamPartKind::Thinking => {
			DecodedPart::Thinking { text: String::new(), signature: Vec::new(), ended: false }
		},
		StreamPartKind::ToolCall => {
			return Err(Error::Provider("Bedrock tool part requires a start event".into()));
		},
		_ => return Err(Error::Provider("unsupported Bedrock stream part".into())),
	};
	state.parts.insert(index, part);
	events.push(TurnEvent::PartStart {
		index,
		kind,
		tool_call_id: Str::default(),
		tool_name: Str::default(),
	});
	Ok(())
}

fn finish_stream(state: &mut DecodeState) -> Result<SmallVec<TurnEvent, 2>, Error> {
	let state = state.get_or_insert_with(BedrockDecodeState::default);
	if state.completed {
		return Ok(SmallVec::new());
	}
	if state.stop.is_none() {
		state.completed = true;
		return Ok(smallvec![TurnEvent::Error(turn_error(
			"api_error",
			"Bedrock Converse stream ended before messageStop".into(),
		))]);
	}
	complete(state, SmallVec::new())
}

fn complete(
	state: &mut BedrockDecodeState,
	mut events: SmallVec<TurnEvent, 2>,
) -> Result<SmallVec<TurnEvent, 2>, Error> {
	if state.completed {
		return Ok(events);
	}
	for (index, part) in &mut state.parts {
		let (ended, signature) = match part {
			DecodedPart::Text { ended, .. } | DecodedPart::Tool { ended, .. } => (ended, Bytes::new()),
			DecodedPart::Thinking { signature, ended, .. } => {
				(ended, Bytes::copy_from_slice(signature))
			},
		};
		if !*ended {
			*ended = true;
			events.push(TurnEvent::PartEnd { index: *index, signature });
		}
	}
	state.completed = true;
	let stop = state.stop.unwrap_or(StopReason::EndTurn);
	events.push(TurnEvent::Outcome(outcome(stop, state)?));
	Ok(events)
}

fn outcome(stop: StopReason, state: &BedrockDecodeState) -> Result<ChatOutcome, Error> {
	let mut output = Vec::with_capacity(state.parts.len());
	for part in state.parts.values() {
		match part {
			DecodedPart::Text { text, .. } if !text.is_empty() => {
				output.push(assistant_item(Part::Text(text.as_str().into())));
			},
			DecodedPart::Thinking { text, signature, .. }
				if !text.is_empty() || !signature.is_empty() =>
			{
				output.push(assistant_item(Part::Thinking(
					Thinking::builder()
						.text(Str::from(text.as_str()))
						.signature(Bytes::copy_from_slice(signature))
						.redacted(text.is_empty())
						.build(),
				)));
			},
			DecodedPart::Tool { id, wire_id, name, args, .. } => {
				let args = if args.is_empty() {
					Bytes::from_static(b"{}")
				} else {
					serde_json::from_slice::<Value>(args).map_err(|error| {
						Error::Provider(Str::from(format!(
							"Bedrock tool arguments are incomplete JSON: {error}"
						)))
					})?;
					Bytes::copy_from_slice(args)
				};
				let mut metadata = Props::default();
				metadata.insert_ns("amazon-bedrock", "tool_use_id", Value::String(wire_id.to_string()));
				output.push(
					Item::builder()
						.seq(0)
						.kind(ItemKind::ToolCall(
							ToolCall::builder()
								.id(*id)
								.name(name.clone())
								.args_json(args)
								.thought_signature(Bytes::new())
								.provider_metadata(metadata)
								.build(),
						))
						.props(Props::default())
						.build(),
				);
			},
			_ => {},
		}
	}
	let mut props = Props::default();
	if let Some(metrics) = &state.metrics {
		props.insert_ns("amazon-bedrock", "metrics", metrics.clone());
	}
	Ok(ChatOutcome::builder()
		.output(output)
		.stop(stop)
		.maybe_usage(state.usage.map(|usage| {
			let total = if usage.total_tokens == 0 {
				usage.input_tokens.saturating_add(usage.output_tokens)
			} else {
				usage.total_tokens
			};
			Usage::builder()
				.input_tokens(usage.input_tokens)
				.output_tokens(usage.output_tokens)
				.cache_read_tokens(usage.cache_read_input_tokens)
				.cache_write_tokens(usage.cache_write_input_tokens)
				.total_tokens(total)
				.accuracy(Accuracy::Exact)
				.detail(Props::default())
				.build()
		}))
		.maybe_cost(None)
		.unsupported(Vec::new())
		.maybe_revision(None)
		.provider(Str::new_static("amazon-bedrock"))
		.model(Str::default())
		.props(props)
		.build())
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

fn map_stop(reason: &str) -> Option<StopReason> {
	match reason {
		"end_turn" | "stop_sequence" => Some(StopReason::EndTurn),
		"tool_use" => Some(StopReason::ToolUse),
		"max_tokens" | "model_context_window_exceeded" => Some(StopReason::MaxTokens),
		"content_filtered" | "guardrail_intervened" => Some(StopReason::ContentFilter),
		_ => None,
	}
}

fn turn_error(kind: &str, detail: Str) -> TurnError {
	let kind = match kind {
		"authentication_error" => TurnErrorKind::Auth,
		"rate_limit_error" => TurnErrorKind::RateLimited,
		"overloaded_error" => TurnErrorKind::Overloaded,
		_ => TurnErrorKind::Upstream,
	};
	TurnError::builder()
		.kind(kind)
		.detail(detail)
		.unsupported(Vec::new())
		.retry_after_ms(0)
		.build()
}

fn policy_bool(
	model_policy: Option<&ResolvedModelPolicy>,
	name: &str,
	thinking_enabled: bool,
) -> Option<bool> {
	let compat = &model_policy?.compat;
	if thinking_enabled
		&& let Some(value) = compat
			.get_ns("wire", "when_thinking")
			.and_then(Value::as_object)
			.and_then(|overlay| overlay.get(name))
			.and_then(Value::as_bool)
	{
		return Some(value);
	}
	compat.get_ns("wire", name).and_then(Value::as_bool)
}

fn escaped_tool_name(name: &str, escape: bool) -> Cow<'_, str> {
	if escape {
		Cow::Owned(format!("_{name}"))
	} else {
		Cow::Borrowed(name)
	}
}

const fn bedrock_effort(effort: Effort) -> &'static str {
	match effort {
		Effort::Off | Effort::Minimal | Effort::Low => "low",
		Effort::Medium => "medium",
		Effort::High => "high",
		Effort::XHigh => "xhigh",
		Effort::Max => "max",
		_ => "medium",
	}
}

const fn bedrock_budget(effort: Effort) -> u64 {
	match effort {
		Effort::Off | Effort::Minimal => 1_024,
		Effort::Low => 2_048,
		Effort::Medium => 8_192,
		Effort::High => 16_384,
		Effort::XHigh => 24_576,
		Effort::Max => 32_768,
		_ => 8_192,
	}
}

const fn legacy_bedrock_effort(effort: Effort) -> &'static str {
	match effort {
		Effort::Off | Effort::Minimal | Effort::Low => "low",
		Effort::Medium => "medium",
		Effort::High => "high",
		Effort::Max => "max",
		_ => "medium",
	}
}

const fn legacy_bedrock_budget(effort: Effort) -> u64 {
	match effort {
		Effort::Off | Effort::Minimal => 1_024,
		Effort::Low => 2_048,
		Effort::Medium => 8_192,
		Effort::High => 16_384,
		Effort::Max => 32_768,
		_ => 8_192,
	}
}

fn is_claude(model: &str) -> bool {
	let model = model.to_ascii_lowercase();
	model.contains("anthropic.claude") || model.contains("anthropic/claude")
}

fn is_adaptive_claude(model: &str) -> bool {
	if !is_claude(model) {
		return false;
	}
	let model = model.to_ascii_lowercase();
	["opus-4-6", "opus-4-7", "opus-4-8", "opus-5", "sonnet-4-6", "sonnet-5", "fable-5", "mythos-5"]
		.iter()
		.any(|generation| model.contains(generation))
}

fn supports_adaptive_display(model: &str) -> bool {
	let model = model.to_ascii_lowercase();
	["opus-4-7", "opus-4-8", "opus-5", "sonnet-5", "fable-5", "mythos-5"]
		.iter()
		.any(|generation| model.contains(generation))
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

fn dropped(what: impl Into<Str>, detail: impl Into<Str>) -> Unsupported {
	Unsupported::builder()
		.what(what.into())
		.detail(detail.into())
		.action(UnsupportedAction::Dropped)
		.build()
}

fn emulated(what: impl Into<Str>, detail: impl Into<Str>) -> Unsupported {
	Unsupported::builder()
		.what(what.into())
		.detail(detail.into())
		.action(UnsupportedAction::Emulated)
		.build()
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
	use omp_llm_catalog::compat::Compat;
	use omp_llm_transport::{DecodeState, Frame, Transport};
	use omp_llm_types::{
		CacheHint, CacheRetention, CallId, ChatRequest, Effort, Fallback, Feature, Item, ItemKind,
		Message, Part, PromptCacheBreakpoint, PromptCacheMode, Props, Reasoning, ResolvedModelPolicy,
		ResolvedThinkingMode, ResolvedThinkingPolicy, Role, Thinking, Thread, ToolCall, ToolDef,
		ToolResult, TurnEvent,
	};
	use serde_json::{Value, json};

	use super::BedrockConverseCodec;

	fn item(kind: ItemKind) -> Item {
		Item::builder()
			.seq(0)
			.kind(kind)
			.props(Props::default())
			.build()
	}

	fn message(role: Role, parts: Vec<Part>) -> Item {
		item(ItemKind::Message(Message::builder().role(role).parts(parts).build()))
	}

	fn request(model: &str, items: Vec<Item>) -> ChatRequest {
		ChatRequest::builder()
			.model(Str::new(model))
			.thread(Thread::builder().items(items).build())
			.tools(Vec::new())
			.build()
	}

	fn model_policy(
		mode: ResolvedThinkingMode,
		supports_display: Option<bool>,
	) -> ResolvedModelPolicy {
		ResolvedModelPolicy {
			thinking: Some(ResolvedThinkingPolicy {
				mode,
				efforts: [Effort::Low, Effort::High, Effort::XHigh, Effort::Max]
					.into_iter()
					.collect(),
				default_effort: Some(Effort::High),
				effort_map: BTreeMap::new(),
				effort_routing: BTreeMap::new(),
				effort_budgets: BTreeMap::new(),
				supports_display,
				suppress_when_off: None,
				requires_effort: None,
			}),
			..ResolvedModelPolicy::default()
		}
	}

	fn set_wire_bool(policy: &mut ResolvedModelPolicy, key: &str, value: bool) {
		policy.compat.insert_ns("wire", key, json!(value));
	}

	#[test]
	fn claude_history_projects_signature_tools_sentinel_and_cache_points() {
		let call_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().expect("call id");
		let mut req = request("eu.anthropic.claude-sonnet-4-6-v1:0", vec![
			message(Role::System, vec![Part::Text("system".into())]),
			message(Role::User, vec![Part::Text("question".into())]),
			message(Role::Assistant, vec![Part::Thinking(
				Thinking::builder()
					.text("reason".into())
					.signature(Bytes::from_static(b"signed"))
					.redacted(false)
					.build(),
			)]),
			item(ItemKind::ToolCall(
				ToolCall::builder()
					.id(call_id)
					.name("lookup".into())
					.args_json(Bytes::from_static(br#"{"q":"x"}"#))
					.thought_signature(Bytes::new())
					.build(),
			)),
			item(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(call_id)
					.name("lookup".into())
					.parts(vec![Part::Text("answer".into())])
					.is_error(false)
					.build(),
			)),
			message(Role::User, vec![Part::Text("continue".into())]),
		]);
		req.cache = Some(
			CacheHint::builder()
				.session_key("session".into())
				.retention(CacheRetention::Long)
				.mode(PromptCacheMode::Explicit)
				.breakpoint(PromptCacheBreakpoint::LatestStableMessage)
				.build(),
		);
		let (wire, unsupported) = BedrockConverseCodec
			.encode(&req, &Compat::default())
			.expect("encode");
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		let body: Value = serde_json::from_slice(&wire).expect("wire JSON");
		assert_eq!(
			body["messages"][1]["content"][0]["reasoningContent"]["reasoningText"]["signature"],
			"signed"
		);
		assert_eq!(body["messages"][1]["content"][1]["toolUse"]["toolUseId"], call_id.to_string());
		assert_eq!(body["messages"][2]["content"][0]["toolResult"]["toolUseId"], call_id.to_string());
		assert_eq!(body["toolConfig"]["tools"][0]["toolSpec"]["name"], "__no_tools__");
		assert!(
			body["system"]
				.as_array()
				.is_some_and(|blocks| blocks.iter().any(|block| block.get("cachePoint").is_some()))
		);
		assert!(
			body["messages"].as_array().is_some_and(|messages| messages
				.iter()
				.any(|message| message["content"].as_array().is_some_and(|blocks| blocks
					.iter()
					.any(|block| block.get("cachePoint").is_some()))))
		);
	}

	#[test]
	fn non_anthropic_model_replays_unsigned_reasoning() {
		let req =
			request("amazon.nova-pro-v1:0", vec![message(Role::Assistant, vec![Part::Thinking(
				Thinking::builder()
					.text("native reasoning".into())
					.signature(Bytes::new())
					.redacted(false)
					.build(),
			)])]);
		let (wire, _) = BedrockConverseCodec
			.encode(&req, &Compat::default())
			.expect("encode");
		let body: Value = serde_json::from_slice(&wire).expect("wire JSON");
		let reasoning = &body["messages"][0]["content"][0]["reasoningContent"]["reasoningText"];
		assert_eq!(reasoning["text"], "native reasoning");
		assert!(reasoning.get("signature").is_none());
	}

	#[test]
	fn adaptive_reasoning_uses_converse_additional_fields() {
		let mut req =
			request("us.anthropic.claude-opus-4-7-v1:0", vec![message(Role::User, vec![Part::Text(
				"solve".into(),
			)])]);
		req.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::High).build())
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		let mut options = Props::default();
		options.insert_ns("amazon-bedrock", "reasoning_mode", Value::String("adaptive".into()));
		options.insert_ns("amazon-bedrock", "thinking_display", Value::String("summarized".into()));
		req.provider_options = Some(options);
		let (wire, unsupported) = BedrockConverseCodec
			.encode(&req, &Compat::default())
			.expect("encode");
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		let body: Value = serde_json::from_slice(&wire).expect("wire JSON");
		assert_eq!(
			body["additionalModelRequestFields"]["thinking"],
			serde_json::json!({"type":"adaptive","display":"summarized"})
		);
		assert_eq!(body["additionalModelRequestFields"]["output_config"]["effort"], "high");
	}
	#[test]
	fn stream_preserves_reasoning_signature_usage_metrics_and_one_terminal() {
		let codec = BedrockConverseCodec;
		let mut state = DecodeState::default();
		let frames: &[&[u8]] = &[
			br#"{"role":"assistant"}"#,
			br#"{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"why ","signature":"sig"}}}"#,
			br#"{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"now","signature":"nature"}}}"#,
			br#"{"contentBlockIndex":0}"#,
			br#"{"stopReason":"end_turn"}"#,
			br#"{"usage":{"inputTokens":10,"outputTokens":4,"totalTokens":14,"cacheReadInputTokens":3,"cacheWriteInputTokens":2},"metrics":{"latencyMs":123}}"#,
		];
		let mut events = Vec::new();
		for frame in frames {
			events.extend(
				codec
					.decode(Frame::Data(frame), &mut state)
					.expect("decode"),
			);
		}
		let outcomes: Vec<_> = events
			.iter()
			.filter_map(|event| match event {
				TurnEvent::Outcome(outcome) => Some(outcome),
				_ => None,
			})
			.collect();
		assert_eq!(outcomes.len(), 1);
		let outcome = outcomes[0];
		let usage = outcome.usage.as_ref().expect("usage");
		assert_eq!(
			(
				usage.input_tokens,
				usage.output_tokens,
				usage.cache_read_tokens,
				usage.cache_write_tokens
			),
			(10, 4, 3, 2)
		);
		let ItemKind::Message(message) = &outcome.output[0].kind else {
			panic!("thinking item")
		};
		let Part::Thinking(thinking) = &message.parts[0] else {
			panic!("thinking part")
		};
		assert_eq!(thinking.text, "why now");
		assert_eq!(thinking.signature, Bytes::from_static(b"signature"));
		assert!(
			codec
				.decode(Frame::Done, &mut state)
				.expect("done")
				.is_empty()
		);
	}

	#[test]
	fn stream_error_is_terminal_and_classified() {
		let codec = BedrockConverseCodec;
		let mut state = DecodeState::default();
		let events = codec
			.decode(
				Frame::Data(
					br#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
				),
				&mut state,
			)
			.expect("decode error");
		assert!(
			matches!(events.as_slice(), [TurnEvent::Error(error)] if error.kind == omp_llm_types::TurnErrorKind::RateLimited)
		);
		assert!(
			codec
				.decode(Frame::Done, &mut state)
				.expect("done")
				.is_empty()
		);
	}
	#[test]
	fn model_policy_selects_equivalent_adaptive_and_budget_bodies() {
		let mut req =
			request("us.anthropic.shared-endpoint-v1:0", vec![message(Role::User, vec![Part::Text(
				"solve".into(),
			)])]);
		req.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::XHigh).build())
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		let mut adaptive = model_policy(ResolvedThinkingMode::AnthropicAdaptive, Some(true));
		adaptive
			.thinking
			.as_mut()
			.unwrap()
			.effort_map
			.insert(Effort::XHigh, "xhigh".into());
		req.model_policy = Some(Arc::new(adaptive));
		let (wire, _) = BedrockConverseCodec
			.encode(&req, &Compat::default())
			.unwrap();
		let body: Value = serde_json::from_slice(&wire).unwrap();
		assert_eq!(
			body["additionalModelRequestFields"]["thinking"],
			json!({"type":"adaptive","display":"summarized"})
		);
		assert_eq!(body["additionalModelRequestFields"]["output_config"]["effort"], "xhigh");

		let mut budget = model_policy(ResolvedThinkingMode::AnthropicBudgetEffort, Some(false));
		let thinking = budget.thinking.as_mut().unwrap();
		thinking.effort_map.insert(Effort::XHigh, "max".into());
		thinking.effort_budgets.insert(Effort::XHigh, 13_579);
		req.model_policy = Some(Arc::new(budget));
		let (wire, _) = BedrockConverseCodec
			.encode(&req, &Compat::default())
			.unwrap();
		let body: Value = serde_json::from_slice(&wire).unwrap();
		assert_eq!(
			body["additionalModelRequestFields"]["thinking"],
			json!({"type":"enabled","budget_tokens":13579})
		);
		assert_eq!(body["additionalModelRequestFields"]["output_config"]["effort"], "max");
	}

	#[test]
	fn model_policy_projects_cache_tool_and_replay_semantics_to_converse() {
		let call_id = CallId::new();
		let mut req = request("us.anthropic.shared-endpoint-v1:0", vec![
			item(ItemKind::ToolCall(
				ToolCall::builder()
					.id(call_id)
					.name("read".into())
					.args_json(Bytes::from_static(b"{}"))
					.thought_signature(Bytes::new())
					.build(),
			)),
			item(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(call_id)
					.name("read".into())
					.parts(vec![Part::Text("ok".into())])
					.is_error(false)
					.build(),
			)),
			message(Role::User, vec![Part::Text("continue".into())]),
		]);
		req.tools.push(
			ToolDef::builder()
				.name("read".into())
				.description("read".into())
				.schema_json(Bytes::from_static(br#"{"type":"object"}"#))
				.build(),
		);
		req.cache = Some(
			CacheHint::builder()
				.session_key("session".into())
				.retention(CacheRetention::Long)
				.mode(PromptCacheMode::Explicit)
				.breakpoint(PromptCacheBreakpoint::LatestStableMessage)
				.build(),
		);
		let mut policy = model_policy(ResolvedThinkingMode::Budget, None);
		set_wire_bool(&mut policy, "escape_builtin_tool_names", true);
		set_wire_bool(&mut policy, "requires_tool_result_id", true);
		set_wire_bool(&mut policy, "supports_long_cache_retention", true);
		req.model_policy = Some(Arc::new(policy));
		let (wire, _) = BedrockConverseCodec
			.encode(&req, &Compat::default())
			.unwrap();
		let body: Value = serde_json::from_slice(&wire).unwrap();
		assert_eq!(body["toolConfig"]["tools"][0]["toolSpec"]["name"], "_read");
		assert_eq!(body["messages"][0]["content"][0]["toolUse"]["name"], "_read");
		assert_eq!(
			body["messages"][1]["content"][0]["toolResult"]["toolUseId"],
			body["messages"][0]["content"][0]["toolUse"]["toolUseId"]
		);
		assert!(body["messages"].as_array().unwrap().iter().any(|message| {
			message["content"].as_array().is_some_and(|blocks| {
				blocks
					.iter()
					.any(|block| block["cachePoint"]["ttl"] == "1h")
			})
		}));

		let mut props = Props::default();
		props.insert_ns("amazon-bedrock", "model", json!("foreign-model"));
		let history = Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::Assistant)
					.parts(vec![Part::Thinking(
						Thinking::builder()
							.text("private".into())
							.signature(Bytes::new())
							.redacted(false)
							.build(),
					)])
					.build(),
			))
			.props(props)
			.build();
		let mut replay = request("target-model", vec![history]);
		let mut policy = model_policy(ResolvedThinkingMode::Budget, None);
		set_wire_bool(&mut policy, "replay_unsigned_thinking", true);
		set_wire_bool(&mut policy, "signing_endpoint", false);
		replay.model_policy = Some(Arc::new(policy));
		let (wire, _) = BedrockConverseCodec
			.encode(&replay, &Compat::default())
			.unwrap();
		let body: Value = serde_json::from_slice(&wire).unwrap();
		assert!(body["messages"][0]["content"][0].get("text").is_some());
	}
}
