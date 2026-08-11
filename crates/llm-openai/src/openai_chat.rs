//! `OpenAI` Chat Completions request and streaming response codec.

use std::{
	borrow::Cow,
	collections::{BTreeMap, BTreeSet},
};

use bytes::{Bytes, BytesMut};
use omp_core::Str;
use omp_llm_catalog::{
	TransportId,
	compat::{
		Compat, ImageEncodingFormat, MaxTokensField, ReasoningWireFormat, ThinkingToolChoiceConflict,
		ToolCallIdProfile as CompatToolCallIdProfile, ToolStrictMode,
	},
};
use omp_llm_error::{Evidence, Kind, WireApi, classify, envelope};
use omp_llm_transport::{
	DecodeState, Frame, Transport,
	normalize::{
		merge_stop_reason, normalize as normalize_schema, openai_strict, with_tool_use_precedence,
	},
};
use omp_llm_types::{
	Accuracy, CallId, CallIdMapper, ChatOutcome, ChatRequest, Cost, Effort, Error, Fallback,
	Feature, Item, ItemKind, Message, Part, Props, Reasoning, ResolvedThinkingMode,
	ResponseFormatKind, Role, StopReason, StreamPartKind, Thinking, ToolCall, ToolCallIdProfile,
	ToolChoice, TurnError, TurnErrorKind, TurnEvent, Unsupported, UnsupportedAction, Usage,
};
use serde_json::{Map, Value, json};
use smallvec::SmallVec;

use crate::model_policy::OpenAiModelPolicy;

/// Codec for the widely implemented `/v1/chat/completions` protocol.
#[derive(Debug, Default)]
pub struct OpenAiChatCodec;

impl Transport for OpenAiChatCodec {
	fn id(&self) -> TransportId {
		TransportId::OpenAiChat
	}

	fn encode(
		&self,
		req: &ChatRequest,
		compat: &Compat,
	) -> Result<(Bytes, Vec<Unsupported>), Error> {
		let mut unsupported = Vec::new();
		let policy = OpenAiModelPolicy::resolve(req, compat, &mut unsupported);
		let compat = &policy.compat;
		let mapper = CallIdMapper::new();
		let id_limits = match compat.tool_call_id_profile {
			CompatToolCallIdProfile::Unconstrained => ToolCallIdProfile::Preserve,
			CompatToolCallIdProfile::OpenAi40 => ToolCallIdProfile::OpenAi,
			CompatToolCallIdProfile::Mistral9Alnum => ToolCallIdProfile::Mistral9,
		};
		let default_system_role = if policy.supports_developer_role {
			"developer"
		} else {
			"system"
		};
		let system_role = match req
			.provider_options
			.as_ref()
			.and_then(|options| options.get_ns("openai", "system_role"))
			.and_then(Value::as_str)
		{
			Some("developer") => "developer",
			Some("system") => "system",
			None => default_system_role,
			Some(_) => {
				unsupported
					.push(dropped("openai/system_role", "system_role must be `system` or `developer`"));
				default_system_role
			},
		};
		let mut body = Map::new();
		body.insert("model".into(), Value::String(req.model.to_string()));
		body.insert(
			"messages".into(),
			Value::Array(encode_messages(
				req,
				&policy,
				&mapper,
				id_limits,
				system_role,
				&mut unsupported,
			)?),
		);
		body.insert("stream".into(), Value::Bool(true));
		if compat.usage_in_streaming {
			body.insert("stream_options".into(), json!({ "include_usage": true }));
		}
		if policy.store_override == Some(true) {
			body.insert("store".into(), Value::Bool(false));
		}

		if let Some(sampling) = &req.sampling {
			if compat.sampling_params {
				insert_if_some(&mut body, "temperature", sampling.temperature);
				insert_if_some(&mut body, "top_p", sampling.top_p);
				insert_if_some(&mut body, "top_k", sampling.top_k);
				insert_if_some(&mut body, "min_p", sampling.min_p);
			} else {
				for (name, present) in [
					("temperature", sampling.temperature.is_some()),
					("top_p", sampling.top_p.is_some()),
					("top_k", sampling.top_k.is_some()),
					("min_p", sampling.min_p.is_some()),
				] {
					if present {
						unsupported.push(dropped(name, "reasoning endpoint rejects sampling controls"));
					}
				}
			}
			if compat.penalties {
				insert_if_some(&mut body, "frequency_penalty", sampling.frequency_penalty);
				insert_if_some(&mut body, "presence_penalty", sampling.presence_penalty);
			} else {
				if sampling.frequency_penalty.is_some() {
					unsupported
						.push(dropped("frequency_penalty", "reasoning endpoint rejects penalties"));
				}
				if sampling.presence_penalty.is_some() {
					unsupported
						.push(dropped("presence_penalty", "reasoning endpoint rejects penalties"));
				}
			}
			if let Some(stop) = &sampling.stop {
				if compat.stop_sequences {
					body.insert(
						"stop".into(),
						Value::Array(stop.iter().map(|v| Value::String(v.to_string())).collect()),
					);
				} else {
					unsupported.push(dropped("stop", "endpoint rejects stop sequences"));
				}
			}
			if let Some(tokens) = sampling.max_output_tokens
				&& !req
					.model_policy
					.as_deref()
					.and_then(|policy| policy.omit_max_output_tokens)
					.unwrap_or(false)
			{
				let field = match compat.max_tokens_field {
					MaxTokensField::MaxTokens => "max_tokens",
					MaxTokensField::MaxCompletionTokens => "max_completion_tokens",
					MaxTokensField::MaxOutputTokens => "max_output_tokens",
				};
				body.insert(field.into(), Value::from(tokens));
			}
		}
		let mut tools = encode_tools(req, compat, &mut unsupported)?;
		let mut forced = false;
		let mut any_choice = false;
		if let Some(choice) = &req.tool_choice
			&& policy.supports_tool_choice
		{
			any_choice = true;
			let mut wire_choice = match &choice.value {
				ToolChoice::Auto => Value::String("auto".into()),
				ToolChoice::None => Value::String("none".into()),
				ToolChoice::Required => {
					forced = true;
					Value::String("required".into())
				},
				ToolChoice::Named(name) => {
					forced = true;
					if compat.named_tool_choice {
						json!({"type":"function", "function":{"name":name.as_str()}})
					} else if tools.iter().any(|tool| {
						tool.pointer("/function/name").and_then(Value::as_str) == Some(name.as_str())
					}) {
						tools.retain(|tool| {
							tool.pointer("/function/name").and_then(Value::as_str) == Some(name.as_str())
						});
						Value::String("required".into())
					} else {
						unsupported.push(feature_unsupported(
							"tool_choice",
							"named tool is not advertised",
							choice.on_unsupported,
						));
						Value::Null
					}
				},
				_ => {
					unsupported.push(feature_unsupported(
						"tool_choice",
						"tool choice mode is unsupported by Chat Completions",
						choice.on_unsupported,
					));
					Value::Null
				},
			};
			if forced && !compat.forced_tool_choice {
				unsupported.push(feature_unsupported(
					"tool_choice",
					"endpoint cannot force tool selection",
					choice.on_unsupported,
				));
				wire_choice = Value::String("auto".into());
				forced = false;
			}
			if !wire_choice.is_null() {
				body.insert("tool_choice".into(), wire_choice);
			}
		}
		if req.tool_choice.is_some() && !policy.supports_tool_choice {
			unsupported.push(dropped("tool_choice", "model policy marks tool choice unsupported"));
		}
		if !tools.is_empty() {
			body.insert("tools".into(), Value::Array(tools));
		}

		let drop_reasoning = match compat.thinking_tool_choice_conflict {
			ThinkingToolChoiceConflict::None => false,
			ThinkingToolChoiceConflict::DropThinkingWhenForced => forced,
			ThinkingToolChoiceConflict::DropThinkingWhenAny => any_choice,
			ThinkingToolChoiceConflict::DropThinkingWhenEffort => false,
		};
		if let Some(reasoning) = &req.thinking {
			if drop_reasoning {
				unsupported.push(feature_unsupported(
					"reasoning",
					"reasoning conflicts with tool choice",
					reasoning.on_unsupported,
				));
			} else {
				encode_reasoning(&mut body, reasoning, &policy, &mut unsupported);
			}
		}

		if let Some(format) = &req.response_format {
			match &format.value.kind {
				ResponseFormatKind::JsonSchema(schema) => {
					let mut schema_value: serde_json::Value =
						serde_json::from_slice(&schema.schema_json).map_err(|error| {
							Error::Provider(Str::from(format!("invalid response schema: {error}")))
						})?;
					let (normalized, reports) =
						normalize_schema(compat.tool_schema_flavor, &schema_value);
					schema_value = normalized;
					unsupported.extend(reports.into_iter().map(|report| {
						Unsupported::builder()
							.what(Str::from("response_format.schema"))
							.detail(report.detail)
							.action(report.action)
							.build()
					}));
					let mut strict = schema.strict;
					if strict == Some(true) {
						let (normalized, reports) = openai_strict(&schema_value);
						if reports.is_empty() {
							schema_value = normalized;
						} else {
							strict = Some(false);
							unsupported.extend(reports.into_iter().map(|report| {
								Unsupported::builder()
									.what(Str::from("response_format.schema.strict"))
									.detail(report.detail)
									.action(report.action)
									.build()
							}));
						}
					}
					let mut json_schema = json!({"name":schema.name.as_str(),"schema":schema_value});
					if let Some(strict) = strict {
						json_schema["strict"] = Value::Bool(strict);
					}
					body.insert(
						"response_format".into(),
						json!({"type":"json_schema","json_schema":json_schema}),
					);
				},
				ResponseFormatKind::Grammar(_) => unsupported.push(feature_unsupported(
					"format",
					"Chat Completions has no portable grammar field",
					format.on_unsupported,
				)),
				_ => {
					unsupported.push(feature_unsupported(
						"format",
						"response format is unsupported by Chat Completions",
						format.on_unsupported,
					));
				},
			}
		}
		if req.cache.is_some() {
			unsupported.push(dropped(
				"cache",
				"cache affinity is handled by routing rather than this wire protocol",
			));
		}
		for (key, value) in req.provider_options.iter().flat_map(|options| &options.0) {
			if key == "openai/system_role" {
				continue;
			}
			if let Some(name) = key.strip_prefix("openai/") {
				body.insert(name.to_string(), value.clone());
			} else {
				unsupported.push(dropped(key, "property is not in the openai namespace"));
			}
		}
		if let Some(extra) = &policy.extra_body {
			for (key, value) in extra {
				body.insert(key.clone(), value.clone());
			}
		}
		if compat.thinking_tool_choice_conflict == ThinkingToolChoiceConflict::DropThinkingWhenEffort
			&& req
				.thinking
				.as_ref()
				.is_some_and(|reasoning| reasoning.value.effort.is_some())
			&& body.remove("thinking").is_some()
		{
			unsupported
				.push(dropped("openai/thinking", "thinking object conflicts with reasoning effort"));
		}

		if unsupported
			.iter()
			.any(|entry| entry.action == UnsupportedAction::Dropped)
			&& requested_error_fallback(req, &unsupported)
		{
			return Err(Error::Unsupported(unsupported));
		}
		let bytes = serde_json::to_vec(&body)
			.map(Bytes::from)
			.map_err(|error| {
				Error::Provider(Str::from(format!("request serialization failed: {error}")))
			})?;
		Ok((bytes, unsupported))
	}

	fn decode(
		&self,
		frame: Frame<'_>,
		state: &mut DecodeState,
	) -> Result<SmallVec<TurnEvent, 2>, Error> {
		let state = state.get_or_insert_with(ChatDecodeState::default);
		if state.done {
			return Ok(SmallVec::new());
		}
		let data = match frame {
			Frame::Data(data) | Frame::Event { data, .. } => data,
			Frame::Done => return Ok(finish_stream(state)),
			_ => return Ok(SmallVec::new()),
		};
		if data == b"[DONE]" {
			return Ok(finish_stream(state));
		}
		let chunk: Value = serde_json::from_slice(data).map_err(|error| {
			Error::Provider(Str::from(format!("invalid Chat Completions chunk: {error}")))
		})?;
		if chunk.get("error").is_some() || chunk.get("type").and_then(Value::as_str) == Some("error")
		{
			let error = classify_error_body(std::str::from_utf8(data).expect("valid JSON is UTF-8"));
			return Ok(finish_error_stream(state, error));
		}
		Ok(decode_chunk(&chunk, state))
	}
}

fn encode_messages(
	req: &ChatRequest,
	policy: &OpenAiModelPolicy,
	mapper: &CallIdMapper,
	profile: ToolCallIdProfile,
	system_role: &str,
	unsupported: &mut Vec<Unsupported>,
) -> Result<Vec<Value>, Error> {
	let compat = &policy.compat;
	let mut messages = Vec::new();
	let mut coalesced_system = String::new();
	let first_system = req.thread.items.iter().position(
		|item| matches!(&item.kind, ItemKind::Message(message) if message.role == Role::System),
	);
	if !compat.multiple_system_messages {
		for item in &req.thread.items {
			if let ItemKind::Message(message) = &item.kind
				&& message.role == Role::System
			{
				for part in &message.parts {
					if !matches!(part, Part::Text(_)) {
						unsupported.push(dropped(
							"message.part",
							"coalesced system/developer messages support text only",
						));
					}
				}
				let text = message_text(message);
				if !coalesced_system.is_empty() && !text.is_empty() {
					coalesced_system.push_str("\n\n");
				}
				coalesced_system.push_str(&text);
			}
		}
	}

	for (position, item) in req.thread.items.iter().enumerate() {
		match &item.kind {
			ItemKind::Message(message) => {
				if message.role == Role::System && !compat.multiple_system_messages {
					if Some(position) == first_system {
						messages.push(json!({"role":system_role, "content":coalesced_system}));
					}
					continue;
				}
				messages.push(encode_message(message, policy, system_role, unsupported)?);
			},
			ItemKind::ToolCall(call) => {
				let wire_id = mapper.to_wire(&call.id, profile);
				let arguments = std::str::from_utf8(&call.args_json)
					.map_err(|_| Error::Provider(Str::from("tool arguments are not UTF-8")))?;
				let tool = json!({"id":wire_id.as_str(),"type":"function","function":{"name":call.name.as_str(),"arguments":arguments}});
				let reasoning_detail = if call.thought_signature.is_empty() {
					None
				} else if compat.reasoning_wire_format == ReasoningWireFormat::OpenRouter {
					Some(reasoning_detail_value(&call.thought_signature, wire_id.as_str()))
				} else {
					unsupported.push(dropped(
						"thread.tool_call.thought_signature",
						"selected Chat Completions dialect cannot replay encrypted reasoning",
					));
					None
				};
				if let Some(last) = messages
					.last_mut()
					.and_then(Value::as_object_mut)
					.filter(|value| value.get("role").and_then(Value::as_str) == Some("assistant"))
				{
					last
						.entry("tool_calls")
						.or_insert_with(|| Value::Array(Vec::new()))
						.as_array_mut()
						.expect("tool_calls created as array")
						.push(tool);
					ensure_tool_call_history_fields(last, policy);
					if let Some(detail) = reasoning_detail {
						last
							.entry("reasoning_details")
							.or_insert_with(|| Value::Array(Vec::new()))
							.as_array_mut()
							.expect("reasoning_details created as array")
							.push(detail);
					}
				} else {
					let mut assistant =
						json!({"role":"assistant","content":Value::Null,"tool_calls":[tool]});
					if let Some(detail) = reasoning_detail {
						assistant["reasoning_details"] = Value::Array(vec![detail]);
					}
					if let Some(assistant) = assistant.as_object_mut() {
						ensure_tool_call_history_fields(assistant, policy);
					}
					messages.push(assistant);
				}
			},
			ItemKind::ToolResult(result) => {
				for part in &result.parts {
					if !matches!(part, Part::Text(_)) {
						unsupported.push(dropped(
							"thread.tool_result.part",
							"Chat Completions tool results support text only",
						));
					}
				}
				messages.push(json!({"role":"tool","tool_call_id":mapper.to_wire(&result.call_id, profile).as_str(),"content":parts_text(&result.parts)}));
			},
			_ => {
				unsupported
					.push(dropped("thread.item", "item kind is unsupported by Chat Completions"));
			},
		}
	}
	Ok(messages)
}

fn encode_message(
	message: &Message,
	policy: &OpenAiModelPolicy,
	system_role: &str,
	unsupported: &mut Vec<Unsupported>,
) -> Result<Value, Error> {
	let compat = &policy.compat;
	let role = match message.role {
		Role::System => system_role,
		Role::User => "user",
		Role::Assistant => "assistant",
		_ => {
			return Err(Error::Provider(Str::new_static("unsupported message role")));
		},
	};
	let mut object = Map::new();
	object.insert("role".into(), Value::String(role.into()));
	let mut content = Vec::new();
	let mut reasoning = String::new();
	let mut reasoning_details = Vec::new();
	for part in &message.parts {
		match part {
			Part::Text(text) => content.push(json!({"type":"text","text":text.as_str()})),
			Part::Thinking(thinking) => {
				if policy.filter_reasoning_history {
					continue;
				}
				if !reasoning.is_empty() {
					reasoning.push_str("\n\n");
				}
				reasoning.push_str(&thinking.text);
				if !thinking.signature.is_empty() {
					if message.role == Role::Assistant
						&& compat.reasoning_wire_format == ReasoningWireFormat::OpenRouter
					{
						reasoning_details.push(reasoning_detail_value(&thinking.signature, ""));
					} else {
						unsupported.push(dropped(
							"message.reasoning.signature",
							"selected Chat Completions dialect cannot replay signed reasoning",
						));
					}
				}
			}
			Part::Blob(blob) if !blob.mime.as_str().starts_with("image/") => unsupported.push(dropped(
				"message.blob",
				"Chat Completions content accepts image blobs only",
			)),
			Part::Blob(blob) => match compat.image_encoding_format {
				ImageEncodingFormat::OpenAiUrl if !blob.inline.is_empty() => content.push(json!({"type":"image_url","image_url":{"url":format!("data:{};base64,{}", blob.mime, base64(&blob.inline))}})),
				ImageEncodingFormat::AnthropicSource if !blob.inline.is_empty() => content.push(json!({"type":"image","source":{"type":"base64","media_type":blob.mime.as_str(),"data":base64(&blob.inline)}})),
				ImageEncodingFormat::GoogleInlineData if !blob.inline.is_empty() => content.push(json!({"inline_data":{"mime_type":blob.mime.as_str(),"data":base64(&blob.inline)}})),
				_ => unsupported.push(dropped("message.image", "image is unavailable in the selected wire representation")),
			},
			_ => {
				unsupported.push(dropped(
					"message.part",
					"part kind is unsupported by Chat Completions",
				));
			},
		}
	}
	if content.len() == 1 && content[0].get("type").and_then(Value::as_str) == Some("text") {
		object.insert(
			"content".into(),
			content
				.pop()
				.and_then(|part| part.get("text").cloned())
				.unwrap_or(Value::String(String::new())),
		);
	} else if content.is_empty() {
		object.insert("content".into(), Value::Null);
	} else {
		object.insert("content".into(), Value::Array(content));
	}
	if !reasoning.is_empty() && message.role == Role::Assistant {
		match compat.reasoning_wire_format {
			ReasoningWireFormat::None | ReasoningWireFormat::OpenAiResponses => {
				unsupported.push(dropped(
					"message.reasoning",
					"reasoning history is not accepted by this endpoint",
				));
			},
			ReasoningWireFormat::OpenRouter => {
				object.insert("reasoning".into(), Value::String(reasoning));
			},
			_ => {
				let field = policy
					.reasoning_content_field
					.as_deref()
					.unwrap_or("reasoning_content");
				object.insert(field.into(), Value::String(reasoning));
			},
		}
	}
	if !reasoning_details.is_empty() {
		object.insert("reasoning_details".into(), Value::Array(reasoning_details));
	}
	Ok(Value::Object(object))
}

fn encode_tools(
	req: &ChatRequest,
	compat: &Compat,
	unsupported: &mut Vec<Unsupported>,
) -> Result<Vec<Value>, Error> {
	req.tools
		.iter()
		.map(|tool| {
			let mut schema: Value = serde_json::from_slice(&tool.schema_json).map_err(|error| {
				Error::Provider(Str::from(format!("invalid tool schema for {}: {error}", tool.name)))
			})?;
			let (normalized, reports) = normalize_schema(compat.tool_schema_flavor, &schema);
			schema = normalized;
			unsupported.extend(reports);
			let requested_strict = match compat.tool_strict_mode {
				ToolStrictMode::AllStrict => {
					if tool.strict != Some(true) {
						unsupported.push(
							Unsupported::builder()
								.what(Str::from("tools.strict"))
								.detail(Str::from("endpoint requires every tool to be strict"))
								.action(UnsupportedAction::Clamped)
								.build(),
						);
					}
					Some(true)
				},
				ToolStrictMode::Mixed => tool.strict,
				ToolStrictMode::None => {
					if tool.strict.is_some() {
						unsupported.push(dropped(
							"tools.strict",
							"endpoint does not accept strict tool definitions",
						));
					}
					None
				},
			};
			let mut emitted_strict = requested_strict;
			if requested_strict == Some(true) {
				let (normalized, reports) = openai_strict(&schema);
				if reports.is_empty() {
					schema = normalized;
				} else {
					emitted_strict = Some(false);
					unsupported.extend(reports);
				}
			}
			let mut function = Map::new();
			function.insert("name".into(), Value::String(tool.name.to_string()));
			function.insert("description".into(), Value::String(tool.description.to_string()));
			function.insert("parameters".into(), schema);
			if let Some(strict) = emitted_strict {
				function.insert("strict".into(), Value::Bool(strict));
			}
			Ok(json!({"type":"function","function":function}))
		})
		.collect()
}

fn encode_reasoning(
	body: &mut Map<String, Value>,
	feature: &Feature<Reasoning>,
	policy: &OpenAiModelPolicy,
	unsupported: &mut Vec<Unsupported>,
) {
	let format = policy.compat.reasoning_wire_format;
	let effort = feature
		.value
		.effort
		.map(|effort| policy.mapped_effort(effort));
	let budget = feature.value.budget_tokens.or_else(|| {
		matches!(policy.thinking_mode, Some(ResolvedThinkingMode::Budget))
			.then(|| {
				feature
					.value
					.effort
					.and_then(|effort| policy.effort_budgets.get(&effort).copied())
			})
			.flatten()
	});
	if feature.value.hide_summary.is_some() && format != ReasoningWireFormat::OpenRouter {
		unsupported.push(feature_unsupported(
			"reasoning.hide_summary",
			"selected reasoning field cannot control summary visibility",
			feature.on_unsupported,
		));
	}
	match format {
		ReasoningWireFormat::None | ReasoningWireFormat::OpenAiResponses => {
			unsupported.push(feature_unsupported(
				"reasoning",
				"selected Chat Completions endpoint has no compatible reasoning field",
				feature.on_unsupported,
			));
		},
		ReasoningWireFormat::OpenAi => {
			if let Some(effort) = effort
				&& !policy.omit_reasoning_effort
			{
				body.insert("reasoning_effort".into(), Value::String(effort.to_string()));
			}
			if budget.is_some() {
				unsupported.push(feature_unsupported(
					"reasoning.budget_tokens",
					"reasoning_effort has no exact token budget",
					feature.on_unsupported,
				));
			}
		},
		ReasoningWireFormat::OpenRouter => {
			let mut value = Map::new();
			if let Some(effort) = effort
				&& !policy.omit_reasoning_effort
			{
				value.insert("effort".into(), Value::String(effort.to_string()));
			}
			if let Some(tokens) = budget {
				value.insert("max_tokens".into(), Value::from(tokens));
			}
			if let Some(exclude) = feature.value.hide_summary {
				value.insert("exclude".into(), Value::Bool(exclude));
			}
			body.insert("reasoning".into(), Value::Object(value));
		},
		ReasoningWireFormat::Zai => {
			let mut thinking = Map::new();
			thinking.insert(
				"type".into(),
				Value::String(
					if feature.value.effort == Some(Effort::Off) {
						"disabled"
					} else {
						"enabled"
					}
					.into(),
				),
			);
			if feature.value.effort != Some(Effort::Off)
				&& let Some(effort) = effort
				&& !policy.omit_reasoning_effort
			{
				thinking.insert("effort".into(), Value::String(effort.to_string()));
			}
			body.insert("thinking".into(), Value::Object(thinking));
			if budget.is_some() {
				unsupported.push(feature_unsupported(
					"reasoning.budget_tokens",
					"thinking type has no exact token budget",
					feature.on_unsupported,
				));
			}
		},
		ReasoningWireFormat::QwenEnableThinking => {
			body.insert(
				"enable_thinking".into(),
				Value::Bool(feature.value.effort != Some(Effort::Off)),
			);
			if feature.value.budget_tokens.is_some() {
				unsupported.push(feature_unsupported(
					"reasoning.budget_tokens",
					"enable_thinking has no exact token budget",
					feature.on_unsupported,
				));
			}
		},
		ReasoningWireFormat::NvidiaChatTemplateKwargs => {
			body.insert(
				"chat_template_kwargs".into(),
				json!({"enable_thinking": feature.value.effort != Some(Effort::Off)}),
			);
			if feature.value.budget_tokens.is_some() {
				unsupported.push(feature_unsupported(
					"reasoning.budget_tokens",
					"chat template control has no exact token budget",
					feature.on_unsupported,
				));
			}
		},
		ReasoningWireFormat::Google | ReasoningWireFormat::Anthropic => {
			unsupported.push(feature_unsupported(
				"reasoning",
				"selected Chat Completions endpoint has no compatible reasoning field",
				feature.on_unsupported,
			));
		},
	}
}
fn ensure_tool_call_history_fields(assistant: &mut Map<String, Value>, policy: &OpenAiModelPolicy) {
	if policy.requires_reasoning_content_for_tool_calls {
		let field = policy
			.reasoning_content_field
			.as_deref()
			.unwrap_or("reasoning_content");
		if !assistant.contains_key(field) {
			assistant.insert(
				field.into(),
				Value::String(
					if policy.allows_synthetic_reasoning_content_for_tool_calls {
						"."
					} else {
						""
					}
					.into(),
				),
			);
		}
	}
	if policy.requires_assistant_content_for_tool_calls
		&& assistant.get("content").is_none_or(Value::is_null)
	{
		assistant.insert("content".into(), Value::String(".".into()));
	}
}

#[derive(Default)]
struct ChatDecodeState {
	choices:        BTreeMap<u32, ChoiceDecodeState>,
	call_ids:       CallIdMapper,
	next_index:     u32,
	usage:          Option<Usage>,
	usage_raw:      Map<String, Value>,
	response_props: Props,
	provider:       Str,
	model:          Str,
	done:           bool,
}

#[derive(Default)]
struct ChoiceDecodeState {
	text_index:         Option<u32>,
	thinking_index:     Option<u32>,
	tools:              BTreeMap<u32, PendingTool>,
	tool_signatures:    BTreeMap<Str, Bytes>,
	text:               BytesMut,
	thinking:           BytesMut,
	thinking_signature: Bytes,
	content_mode:       ContentMode,
	content_pending:    BytesMut,
	refusal:            BytesMut,
	props:              Props,
	stop:               Option<StopReason>,
	error:              Option<TurnError>,
}

#[derive(Clone, Copy, Default)]
enum ContentMode {
	#[default]
	First,
	Text,
	ToolMarkup,
}

struct PendingTool {
	stream_index: u32,
	id:           CallId,
	wire_id:      Str,
	name:         Str,
	args:         BytesMut,
	object_args:  Option<Value>,
	started:      bool,
	signature:    Bytes,
}

fn decode_chunk(chunk: &Value, state: &mut ChatDecodeState) -> SmallVec<TurnEvent, 2> {
	let mut events = SmallVec::new();
	if let Some(usage) = chunk.get("usage").and_then(Value::as_object) {
		merge_json_object(&mut state.usage_raw, usage);
		state.usage = Some(decode_usage(&Value::Object(state.usage_raw.clone())));
	}
	capture_response_metadata(chunk, state);
	let Some(choices) = chunk.get("choices").and_then(Value::as_array) else {
		return events;
	};
	for (position, choice) in choices.iter().enumerate() {
		let choice_index = choice
			.get("index")
			.and_then(Value::as_u64)
			.unwrap_or(position as u64) as u32;
		let mut choice_state = state.choices.remove(&choice_index).unwrap_or_default();
		if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
			match normalize_finish_reason(reason) {
				Ok(stop) => {
					choice_state.stop = Some(match choice_state.stop {
						Some(current) => merge_stop_reason(current, stop),
						None => stop,
					});
				},
				Err(detail) => choice_state.error = Some(upstream_error(detail)),
			}
		}
		capture_choice_metadata(choice, &mut choice_state);
		let payload = choice
			.get("delta")
			.or_else(|| choice.get("message"))
			.unwrap_or(choice);
		decode_choice_payload(payload, state, &mut choice_state, &mut events);
		state.choices.insert(choice_index, choice_state);
	}
	events
}

fn decode_choice_payload(
	payload: &Value,
	state: &mut ChatDecodeState,
	choice: &mut ChoiceDecodeState,
	events: &mut SmallVec<TurnEvent, 2>,
) {
	if let Some(error) = payload.get("error") {
		let body = json!({"error": error}).to_string();
		choice.error = Some(classify_error_body(&body));
		return;
	}
	for field in ["reasoning_content", "reasoning", "reasoning_text"] {
		if let Some(reasoning) = payload
			.get(field)
			.and_then(Value::as_str)
			.filter(|text| !text.is_empty())
		{
			emit_part(state, choice, events, true, reasoning.as_bytes());
			break;
		}
	}
	if let Some(details) = payload.get("reasoning_details").and_then(Value::as_array) {
		append_prop_value(&mut choice.props, "reasoning_details", Value::Array(details.clone()));
		for detail in details {
			let Some(object) = detail.as_object() else {
				continue;
			};
			if let Some(data) = object.get("data").and_then(Value::as_str) {
				let signature = Bytes::copy_from_slice(data.as_bytes());
				if let Some(id) = object.get("id").and_then(Value::as_str) {
					let signature = signature.clone();
					choice
						.tool_signatures
						.insert(Str::from(id), signature.clone());
					for tool in choice.tools.values_mut().filter(|tool| tool.wire_id == id) {
						tool.signature = signature.clone();
					}
				} else if choice.thinking_signature.is_empty() {
					choice.thinking_signature = signature;
				}
			}
		}
	}
	if let Some(content) = normalized_content(payload.get("content")) {
		append_content(state, choice, events, content.as_bytes());
	} else if let Some(text) = payload.get("text").and_then(Value::as_str) {
		append_content(state, choice, events, text.as_bytes());
	}
	if let Some(refusal) = payload
		.get("refusal")
		.and_then(Value::as_str)
		.filter(|text| !text.is_empty())
	{
		choice.refusal.extend_from_slice(refusal.as_bytes());
		append_content(state, choice, events, refusal.as_bytes());
	}
	if let Some(annotations) = payload.get("annotations").and_then(Value::as_array) {
		append_prop_value(&mut choice.props, "annotations", Value::Array(annotations.clone()));
	}
	let Some(calls) = payload.get("tool_calls").and_then(Value::as_array) else {
		return;
	};
	for (position, call) in calls.iter().enumerate() {
		let wire_index = call
			.get("index")
			.and_then(Value::as_u64)
			.unwrap_or(position as u64) as u32;
		let name = call.pointer("/function/name").and_then(Value::as_str);
		if !choice.tools.contains_key(&wire_index) {
			let stream_index = state.next_index;
			state.next_index += 1;
			let wire_id = call.get("id").and_then(Value::as_str).unwrap_or_default();
			let id = if wire_id.is_empty() {
				CallId::new()
			} else {
				state.call_ids.observe(wire_id)
			};
			let signature = choice
				.tool_signatures
				.get(wire_id)
				.cloned()
				.unwrap_or_default();
			choice.tools.insert(wire_index, PendingTool {
				stream_index,
				id,
				wire_id: Str::from(wire_id),
				name: Str::from(name.unwrap_or_default()),
				args: BytesMut::new(),
				object_args: None,
				started: false,
				signature,
			});
		}
		let tool = choice
			.tools
			.get_mut(&wire_index)
			.expect("tool inserted above");
		if tool.wire_id.is_empty()
			&& let Some(wire_id) = call.get("id").and_then(Value::as_str)
		{
			tool.id = state.call_ids.observe(wire_id);
			tool.wire_id = Str::from(wire_id);
			if let Some(signature) = choice.tool_signatures.get(wire_id) {
				tool.signature = signature.clone();
			}
		}
		if tool.name.is_empty()
			&& let Some(name) = name
		{
			tool.name = Str::from(name);
		}
		start_tool(tool, events);
		if let Some(arguments) = call.pointer("/function/arguments") {
			match arguments {
				Value::String(arguments) => {
					tool.args.extend_from_slice(arguments.as_bytes());
					if tool.started && !arguments.is_empty() {
						events.push(TurnEvent::PartDelta {
							index: tool.stream_index,
							chunk: Bytes::copy_from_slice(arguments.as_bytes()),
						});
					}
				},
				Value::Object(arguments) => {
					let target = tool
						.object_args
						.get_or_insert_with(|| Value::Object(Map::new()))
						.as_object_mut()
						.expect("object argument accumulator");
					for (key, value) in arguments {
						target.insert(key.clone(), value.clone());
					}
				},
				_ => {},
			}
		}
	}
}

fn start_tool(tool: &mut PendingTool, events: &mut SmallVec<TurnEvent, 2>) {
	if tool.started || tool.name.is_empty() {
		return;
	}
	tool.started = true;
	events.push(TurnEvent::PartStart {
		index:        tool.stream_index,
		kind:         StreamPartKind::ToolCall,
		tool_call_id: Str::from(tool.id.to_string()),
		tool_name:    tool.name.clone(),
	});
	if !tool.args.is_empty() {
		events.push(TurnEvent::PartDelta {
			index: tool.stream_index,
			chunk: Bytes::copy_from_slice(&tool.args),
		});
	}
}

fn append_content(
	state: &mut ChatDecodeState,
	choice: &mut ChoiceDecodeState,
	events: &mut SmallVec<TurnEvent, 2>,
	bytes: &[u8],
) {
	if bytes.is_empty() {
		return;
	}
	if matches!(choice.content_mode, ContentMode::First) {
		choice.content_mode = ContentMode::Text;
	}
	match choice.content_mode {
		ContentMode::Text => process_visible_content(state, choice, events, bytes),
		ContentMode::ToolMarkup => process_tool_markup(state, choice, events, bytes),
		ContentMode::First => unreachable!("first content mode is promoted above"),
	}
}

fn process_visible_content(
	state: &mut ChatDecodeState,
	choice: &mut ChoiceDecodeState,
	events: &mut SmallVec<TurnEvent, 2>,
	bytes: &[u8],
) {
	const OPEN: &[u8] = b"<tool_call>";
	choice.content_pending.extend_from_slice(bytes);
	if let Some(at) = find_bytes(&choice.content_pending, OPEN) {
		let rest = choice.content_pending.split_off(at + OPEN.len());
		let mut visible = choice.content_pending.split_to(at).freeze();
		choice.content_pending.clear();
		for fence in [b"```xml\n".as_slice(), b"```json\n".as_slice(), b"```\n".as_slice()] {
			if visible.ends_with(fence) {
				visible.truncate(visible.len() - fence.len());
				break;
			}
		}
		emit_part(state, choice, events, false, &visible);
		choice.content_pending.extend_from_slice(&rest);
		choice.content_mode = ContentMode::ToolMarkup;
		process_tool_markup(state, choice, events, &[]);
		return;
	}
	let keep = trailing_prefix_len(&choice.content_pending, OPEN);
	let flush = choice.content_pending.len().saturating_sub(keep);
	if flush != 0 {
		let visible = choice.content_pending.split_to(flush).freeze();
		emit_part(state, choice, events, false, &visible);
	}
}

fn process_tool_markup(
	state: &mut ChatDecodeState,
	choice: &mut ChoiceDecodeState,
	events: &mut SmallVec<TurnEvent, 2>,
	bytes: &[u8],
) {
	const CLOSE: &[u8] = b"</tool_call>";
	choice.content_pending.extend_from_slice(bytes);
	let Some(at) = find_bytes(&choice.content_pending, CLOSE) else {
		return;
	};
	let mut rest = choice.content_pending.split_off(at + CLOSE.len());
	let body = choice.content_pending.split_to(at).freeze();
	choice.content_pending.clear();
	if let Ok(body) = std::str::from_utf8(&body)
		&& let (Some(name), Some(arguments)) =
			(between(body, "<name>", "</name>"), between(body, "<arguments>", "</arguments>"))
	{
		let wire_index = choice
			.tools
			.keys()
			.next_back()
			.copied()
			.unwrap_or(0)
			.saturating_add(1);
		let stream_index = state.next_index;
		state.next_index += 1;
		let mut tool = PendingTool {
			stream_index,
			id: CallId::new(),
			wire_id: Str::default(),
			name: Str::from(name),
			args: BytesMut::from(arguments.as_bytes()),
			object_args: None,
			started: false,
			signature: Bytes::new(),
		};
		start_tool(&mut tool, events);
		choice.tools.insert(wire_index, tool);
	}
	for fence in [b"\n```".as_slice(), b"```".as_slice()] {
		if rest.starts_with(fence) {
			rest = rest.split_off(fence.len());
			break;
		}
	}
	choice.content_mode = ContentMode::Text;
	process_visible_content(state, choice, events, &rest);
}

fn between<'a>(value: &'a str, open: &str, close: &str) -> Option<&'a str> {
	let start = value.find(open)? + open.len();
	let end = value[start..].find(close)? + start;
	Some(&value[start..end])
}

fn emit_part(
	state: &mut ChatDecodeState,
	choice: &mut ChoiceDecodeState,
	events: &mut SmallVec<TurnEvent, 2>,
	thinking: bool,
	bytes: &[u8],
) {
	if bytes.is_empty() {
		return;
	}
	let existing = if thinking {
		choice.thinking_index
	} else {
		choice.text_index
	};
	let index = existing.unwrap_or_else(|| {
		let index = state.next_index;
		state.next_index += 1;
		if thinking {
			choice.thinking_index = Some(index);
		} else {
			choice.text_index = Some(index);
		}
		events.push(TurnEvent::PartStart {
			index,
			kind: if thinking {
				StreamPartKind::Thinking
			} else {
				StreamPartKind::Text
			},
			tool_call_id: Str::default(),
			tool_name: Str::default(),
		});
		index
	});
	if thinking {
		choice.thinking.extend_from_slice(bytes);
	} else {
		choice.text.extend_from_slice(bytes);
	}
	events.push(TurnEvent::PartDelta { index, chunk: Bytes::copy_from_slice(bytes) });
}

fn finish_stream(state: &mut ChatDecodeState) -> SmallVec<TurnEvent, 2> {
	state.done = true;
	let mut events = SmallVec::new();
	let mut choices = std::mem::take(&mut state.choices);
	for choice in choices.values_mut() {
		flush_pending_content(state, choice, &mut events);
		for tool in choice.tools.values_mut() {
			start_tool(tool, &mut events);
			if tool.args.is_empty()
				&& let Some(arguments) = tool.object_args.take()
			{
				let bytes = serde_json::to_vec(&arguments).unwrap_or_else(|_| b"{}".to_vec());
				tool.args.extend_from_slice(&bytes);
				if tool.started {
					events.push(TurnEvent::PartDelta {
						index: tool.stream_index,
						chunk: Bytes::from(bytes),
					});
				}
			}
		}
	}
	let mut ended = BTreeSet::new();
	for choice in choices.values() {
		if let Some(index) = choice.text_index {
			ended.insert((index, Bytes::new()));
		}
		if let Some(index) = choice.thinking_index {
			ended.insert((index, choice.thinking_signature.clone()));
		}
		for tool in choice.tools.values().filter(|tool| tool.started) {
			ended.insert((tool.stream_index, tool.signature.clone()));
		}
	}
	for (index, signature) in ended {
		events.push(TurnEvent::PartEnd { index, signature });
	}
	if let Some(error) = choices.values_mut().find_map(|choice| choice.error.take()) {
		events.push(TurnEvent::Error(error));
		return events;
	}
	let mut output = Vec::new();
	let mut stop = StopReason::EndTurn;
	for (choice_index, choice) in &mut choices {
		let has_tools = choice.tools.values().any(|tool| tool.started);
		let choice_stop =
			with_tool_use_precedence(choice.stop.unwrap_or(StopReason::EndTurn), has_tools);
		stop = merge_stop_reason(stop, choice_stop);
		let has_message_metadata = !choice.props.is_empty() || !choice.refusal.is_empty();
		let mut item_props = choice.props.clone();
		item_props.insert_ns("openai", "choice_index", Value::from(*choice_index));
		if !choice.refusal.is_empty() {
			item_props.insert_ns(
				"openai",
				"refusal",
				Value::String(String::from_utf8_lossy(&choice.refusal).into_owned()),
			);
		}
		let mut parts = Vec::new();
		if !choice.thinking.is_empty() {
			let text = std::mem::take(&mut choice.thinking).freeze();
			parts.push(Part::Thinking(
				Thinking::builder()
					.text(Str::from_utf8_owned(text).expect("JSON thinking deltas are UTF-8"))
					.signature(choice.thinking_signature.clone())
					.redacted(false)
					.build(),
			));
		}
		if !choice.text.is_empty() {
			let text = std::mem::take(&mut choice.text).freeze();
			parts.push(Part::Text(Str::from_utf8_owned(text).expect("JSON content deltas are UTF-8")));
		}
		if !parts.is_empty() || has_message_metadata {
			output.push(
				Item::builder()
					.seq(u64::from(*choice_index))
					.kind(ItemKind::Message(
						Message::builder()
							.role(Role::Assistant)
							.parts(parts)
							.build(),
					))
					.props(item_props.clone())
					.build(),
			);
		}
		for tool in choice.tools.values_mut().filter(|tool| tool.started) {
			output.push(
				Item::builder()
					.seq(u64::from(*choice_index))
					.kind(ItemKind::ToolCall(
						ToolCall::builder()
							.id(tool.id)
							.name(std::mem::take(&mut tool.name))
							.args_json(std::mem::take(&mut tool.args).freeze())
							.thought_signature(tool.signature.clone())
							.build(),
					))
					.props(item_props.clone())
					.build(),
			);
		}
	}
	events.push(TurnEvent::Outcome(
		ChatOutcome::builder()
			.output(output)
			.stop(stop)
			.maybe_usage(state.usage.take())
			.maybe_cost(reported_cost(state.usage_raw.get("cost")))
			.provider(std::mem::take(&mut state.provider))
			.model(std::mem::take(&mut state.model))
			.unsupported(Vec::new())
			.props(std::mem::take(&mut state.response_props))
			.build(),
	));
	events
}

fn finish_error_stream(state: &mut ChatDecodeState, error: TurnError) -> SmallVec<TurnEvent, 2> {
	state.done = true;
	let mut events = SmallVec::new();
	let mut ended = BTreeSet::new();
	for choice in state.choices.values() {
		if let Some(index) = choice.text_index {
			ended.insert((index, Bytes::new()));
		}
		if let Some(index) = choice.thinking_index {
			ended.insert((index, choice.thinking_signature.clone()));
		}
		for tool in choice.tools.values().filter(|tool| tool.started) {
			ended.insert((tool.stream_index, tool.signature.clone()));
		}
	}
	for (index, signature) in ended {
		events.push(TurnEvent::PartEnd { index, signature });
	}
	state.choices.clear();
	events.push(TurnEvent::Error(error));
	events
}

fn flush_pending_content(
	state: &mut ChatDecodeState,
	choice: &mut ChoiceDecodeState,
	events: &mut SmallVec<TurnEvent, 2>,
) {
	let pending = std::mem::take(&mut choice.content_pending).freeze();
	match choice.content_mode {
		ContentMode::Text => {
			if !b"<tool_call>".starts_with(&pending) {
				emit_part(state, choice, events, false, &pending);
			}
		},
		ContentMode::ToolMarkup => {},
		ContentMode::First => emit_part(state, choice, events, false, &pending),
	}
}

fn merge_json_object(target: &mut Map<String, Value>, incoming: &Map<String, Value>) {
	for (key, value) in incoming {
		if let Value::Object(incoming_object) = value
			&& let Some(Value::Object(target_object)) = target.get_mut(key)
		{
			merge_json_object(target_object, incoming_object);
			continue;
		}
		target.insert(key.clone(), value.clone());
	}
}

fn decode_usage(value: &Value) -> Usage {
	let prompt = value
		.get("prompt_tokens")
		.and_then(Value::as_u64)
		.unwrap_or(0);
	let output = value
		.get("completion_tokens")
		.and_then(Value::as_u64)
		.unwrap_or(0);
	let cache_read = [
		value.get("cached_tokens"),
		value.get("prompt_cache_hit_tokens"),
		value.pointer("/prompt_tokens_details/cached_tokens"),
	]
	.into_iter()
	.flatten()
	.filter_map(Value::as_u64)
	.find(|tokens| *tokens != 0)
	.unwrap_or(0);
	let explicit_cache_write = value
		.pointer("/prompt_tokens_details/cache_write_tokens")
		.and_then(Value::as_u64);
	let deepseek_cache_write = value
		.get("prompt_cache_miss_tokens")
		.and_then(Value::as_u64)
		.filter(|_| {
			value
				.get("prompt_cache_hit_tokens")
				.and_then(Value::as_u64)
				.is_some()
		});
	let cache_write = explicit_cache_write.or(deepseek_cache_write).unwrap_or(0);
	let mut detail = Props::default();
	detail.insert_ns("openai", "usage", value.clone());
	Usage::builder()
		.input_tokens(prompt)
		.output_tokens(output)
		.cache_read_tokens(cache_read)
		.cache_write_tokens(cache_write)
		.accuracy(Accuracy::Exact)
		.detail(detail)
		.build()
}

fn reported_cost(value: Option<&Value>) -> Option<Cost> {
	const NANOS_PER_USD: f64 = 1_000_000_000.0;
	const U64_UPPER_EXCLUSIVE: f64 = 18_446_744_073_709_551_616.0;
	let dollars = value?.as_f64()?;
	if !dollars.is_finite() || dollars < 0.0 {
		return None;
	}
	let nanos = (dollars * NANOS_PER_USD).round();
	if !nanos.is_finite() || nanos < 0.0 || nanos >= U64_UPPER_EXCLUSIVE {
		return None;
	}
	Some(
		Cost::builder()
			.nanos_usd(nanos as u64)
			.estimated(false)
			.build(),
	)
}

fn normalized_content(value: Option<&Value>) -> Option<Cow<'_, str>> {
	match value? {
		Value::String(text) => Some(Cow::Borrowed(text)),
		Value::Array(parts) => Some(Cow::Owned(
			parts
				.iter()
				.filter_map(|part| {
					part.as_str().or_else(|| {
						let kind = part.get("type").and_then(Value::as_str);
						(kind.is_none() || kind == Some("text"))
							.then(|| part.get("text").and_then(Value::as_str))
							.flatten()
					})
				})
				.collect(),
		)),
		Value::Object(part) => {
			let kind = part.get("type").and_then(Value::as_str);
			if kind.is_none() || kind == Some("text") {
				part.get("text").and_then(Value::as_str).map(Cow::Borrowed)
			} else {
				None
			}
		},
		_ => None,
	}
}

fn normalize_finish_reason(value: &str) -> Result<StopReason, Str> {
	match value {
		"stop" | "end" | "end_turn" => Ok(StopReason::EndTurn),
		"tool_calls" | "function_call" | "tool_use" => Ok(StopReason::ToolUse),
		"length" | "max_tokens" | "max_output_tokens" => Ok(StopReason::MaxTokens),
		"content_filter" | "safety" => Ok(StopReason::ContentFilter),
		"error" | "network_error" => {
			Err(Str::from(format!("provider returned `{value}` finish_reason")))
		},
		other => Err(Str::from(format!("unknown provider finish_reason `{other}`"))),
	}
}

fn capture_response_metadata(chunk: &Value, state: &mut ChatDecodeState) {
	let Some(object) = chunk.as_object() else {
		return;
	};
	let mut metadata = Map::new();
	for (key, value) in object {
		if !matches!(key.as_str(), "choices" | "usage") {
			metadata.insert(key.clone(), value.clone());
		}
	}
	if !metadata.is_empty() {
		let key = Str::new("openai/response");
		match state.response_props.0.entry(key) {
			std::collections::btree_map::Entry::Vacant(entry) => {
				entry.insert(Value::Object(metadata));
			},
			std::collections::btree_map::Entry::Occupied(mut entry) => {
				let target = entry
					.get_mut()
					.as_object_mut()
					.expect("OpenAI response metadata is always an object");
				target.extend(metadata);
			},
		}
	}
	if let Some(provider) = object
		.get("provider")
		.and_then(Value::as_str)
		.filter(|v| !v.is_empty())
	{
		state.provider = Str::from(provider);
	}
	if let Some(model) = object
		.get("model")
		.and_then(Value::as_str)
		.filter(|v| !v.is_empty())
	{
		state.model = Str::from(model);
	}
}

fn capture_choice_metadata(choice: &Value, state: &mut ChoiceDecodeState) {
	for key in ["logprobs", "annotations"] {
		if let Some(value) = choice.get(key).filter(|value| !value.is_null()) {
			append_prop_value(&mut state.props, key, value.clone());
		}
	}
}

fn append_prop_value(props: &mut Props, name: &str, value: Value) {
	let key = Str::from(format!("openai/{name}"));
	match props.0.get_mut(key.as_str()) {
		Some(Value::Array(existing)) => match value {
			Value::Array(mut values) => existing.append(&mut values),
			value => existing.push(value),
		},
		Some(existing) => {
			let previous = std::mem::replace(existing, Value::Null);
			*existing = Value::Array(vec![previous, value]);
		},
		None => {
			props.0.insert(key, value);
		},
	}
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
	haystack
		.windows(needle.len())
		.position(|window| window == needle)
}

fn trailing_prefix_len(value: &[u8], delimiter: &[u8]) -> usize {
	(1..delimiter.len())
		.rev()
		.find(|length| value.ends_with(&delimiter[..*length]))
		.unwrap_or(0)
}

fn upstream_error(detail: impl Into<Str>) -> TurnError {
	TurnError::builder()
		.kind(TurnErrorKind::Upstream)
		.detail(detail.into())
		.maybe_actual(None)
		.unsupported(Vec::new())
		.retry_after_ms(0)
		.build()
}

fn classify_error_body(body: &str) -> TurnError {
	let evidence =
		Evidence { body: Some(body), api: Some(WireApi::OpenAiCompletions), ..Evidence::default() };
	let classification = classify(&evidence);
	let parsed = serde_json::from_str::<Value>(body).ok();
	let envelope = envelope::parse(body);
	let mut detail = envelope
		.as_ref()
		.and_then(|value| value.message.as_deref())
		.unwrap_or(body)
		.to_owned();
	if let Some(parameter) = envelope.as_ref().and_then(|value| value.param.as_deref())
		&& !detail.contains(parameter)
	{
		detail.push_str(" (parameter: ");
		detail.push_str(parameter);
		detail.push(')');
	}
	if let Some(raw) = parsed
		.as_ref()
		.and_then(|value| value.pointer("/error/metadata/raw"))
		.and_then(Value::as_str)
		.filter(|raw| !detail.contains(raw))
	{
		detail.push('\n');
		detail.push_str(raw);
	}
	let kind = if classification.is(Kind::AuthFailed) || classification.is(Kind::OAuthExpired) {
		TurnErrorKind::Auth
	} else if classification.is(Kind::RateThrottle)
		|| classification.is(Kind::ConcurrencyCap)
		|| classification.is(Kind::UsageLimit)
		|| classification.is(Kind::Billing)
	{
		TurnErrorKind::RateLimited
	} else if classification.is(Kind::ModelCapacity) {
		TurnErrorKind::Overloaded
	} else {
		TurnErrorKind::Upstream
	};
	TurnError::builder()
		.kind(kind)
		.detail(Str::from(detail))
		.maybe_actual(None)
		.unsupported(Vec::new())
		.retry_after_ms(classification.retry.map_or(0, |hint| hint.delay_ms))
		.build()
}

fn insert_if_some<T: Into<Value>>(body: &mut Map<String, Value>, key: &str, value: Option<T>) {
	if let Some(value) = value {
		body.insert(key.into(), value.into());
	}
}

fn message_text(message: &Message) -> String {
	parts_text(&message.parts)
}
fn parts_text(parts: &[Part]) -> String {
	parts
		.iter()
		.filter_map(|part| match part {
			Part::Text(text) => Some(text.as_str()),
			_ => None,
		})
		.collect::<Vec<_>>()
		.join("\n")
}

fn feature_unsupported(what: &str, detail: &str, fallback: Fallback) -> Unsupported {
	Unsupported::builder()
		.what(Str::from(what))
		.detail(Str::from(detail))
		.action(match fallback {
			Fallback::Emulate => UnsupportedAction::Emulated,
			_ => UnsupportedAction::Dropped,
		})
		.build()
}
fn dropped(what: impl AsRef<str>, detail: &str) -> Unsupported {
	Unsupported::builder()
		.what(Str::from(what.as_ref()))
		.detail(Str::from(detail))
		.action(UnsupportedAction::Dropped)
		.build()
}
fn requested_error_fallback(req: &ChatRequest, entries: &[Unsupported]) -> bool {
	entries.iter().any(|entry| {
		if entry.what.starts_with("reasoning") {
			req.thinking
				.as_ref()
				.is_some_and(|feature| feature.on_unsupported == Fallback::Error)
		} else if entry.what == "tool_choice" {
			req.tool_choice
				.as_ref()
				.is_some_and(|feature| feature.on_unsupported == Fallback::Error)
		} else if entry.what.starts_with("format") || entry.what.starts_with("response_format") {
			req.response_format
				.as_ref()
				.is_some_and(|feature| feature.on_unsupported == Fallback::Error)
		} else {
			false
		}
	})
}

fn reasoning_detail_value(signature: &[u8], wire_id: &str) -> Value {
	if let Ok(value) = serde_json::from_slice(signature) {
		return value;
	}
	let data = std::str::from_utf8(signature).map_or_else(|_| base64(signature), str::to_owned);
	if wire_id.is_empty() {
		json!({"type":"reasoning.encrypted","data":data})
	} else {
		json!({"type":"reasoning.encrypted","id":wire_id,"data":data})
	}
}

fn base64(bytes: &[u8]) -> String {
	const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
	for chunk in bytes.chunks(3) {
		let bits = (u32::from(chunk[0]) << 16)
			| (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
			| u32::from(*chunk.get(2).unwrap_or(&0));
		out.push(TABLE[((bits >> 18) & 63) as usize] as char);
		out.push(TABLE[((bits >> 12) & 63) as usize] as char);
		out.push(if chunk.len() > 1 {
			TABLE[((bits >> 6) & 63) as usize] as char
		} else {
			'='
		});
		out.push(if chunk.len() > 2 {
			TABLE[(bits & 63) as usize] as char
		} else {
			'='
		});
	}
	out
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use bytes::Bytes;
	use omp_core::Str;
	use omp_llm_transport::sse::SseDecoder;
	use omp_llm_types::{
		BlobPart, ChatRequest, Feature, JsonSchema, Reasoning, ResolvedModelPolicy,
		ResolvedThinkingMode, ResolvedThinkingPolicy, ResponseFormat, Sampling, Thread, ToolChoice,
		ToolDef,
	};
	use serde_json::Value;

	use super::*;

	fn request() -> ChatRequest {
		ChatRequest::builder()
			.model(Str::from("gpt-4.1"))
			.thread(
				Thread::builder()
					.items(vec![
						Item::builder()
							.seq(0)
							.kind(ItemKind::Message(
								Message::builder()
									.role(Role::User)
									.parts(vec![Part::Text(Str::from("Say hello."))])
									.build(),
							))
							.props(Props::default())
							.build(),
					])
					.build(),
			)
			.tools(Vec::new())
			.provider_options(Props::default())
			.build()
	}

	fn encoded(req: &ChatRequest, compat: &Compat) -> Value {
		serde_json::from_slice(&OpenAiChatCodec.encode(req, compat).unwrap().0).unwrap()
	}

	#[test]
	fn plain_text_fixture_encodes_exactly() {
		let actual = encoded(&request(), &Compat::default());
		let fixture: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/openai_chat/request.plain_text.json"
		))
		.unwrap();
		assert_eq!(actual, fixture["wire_body"]);
	}

	#[test]
	fn non_image_blob_is_reported_instead_of_encoded_as_an_image() {
		let mut req = request();
		if let ItemKind::Message(message) = &mut req.thread.items[0].kind {
			message.parts.push(Part::Blob(
				BlobPart::builder()
					.hash([7; 32])
					.mime(Str::from("audio/wav"))
					.size(4)
					.inline(Bytes::from_static(b"RIFF"))
					.build(),
			));
		}
		let (body, unsupported) = OpenAiChatCodec.encode(&req, &Compat::default()).unwrap();
		let wire: Value = serde_json::from_slice(&body).unwrap();
		let content = wire["messages"][0]["content"].to_string();
		assert!(!content.contains("image_url"), "audio must not ride the image wire: {content}");
		assert!(!content.contains("RIFF"));
		assert_eq!(unsupported.len(), 1);
		assert_eq!(unsupported[0].what, "message.blob");
	}

	fn model_policy(compat: Props, effort_map: BTreeMap<Effort, Str>) -> Arc<ResolvedModelPolicy> {
		Arc::new(ResolvedModelPolicy {
			thinking: Some(ResolvedThinkingPolicy {
				mode: ResolvedThinkingMode::Effort,
				efforts: SmallVec::new(),
				default_effort: None,
				effort_map,
				effort_routing: BTreeMap::new(),
				effort_budgets: BTreeMap::new(),
				supports_display: None,
				suppress_when_off: None,
				requires_effort: None,
			}),
			compat,
			..ResolvedModelPolicy::default()
		})
	}

	#[test]
	fn model_policy_effort_map_and_omit_max_are_per_request() {
		let mut req = request();
		req.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::XHigh).build())
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		req.sampling = Some(
			Sampling::builder()
				.temperature(0.7)
				.max_output_tokens(512)
				.build(),
		);
		let mut fireworks = BTreeMap::new();
		fireworks.insert(Effort::XHigh, Str::new_static("high"));
		let mut fireworks_compat = Props::default();
		fireworks_compat.insert_ns("wire", "max_tokens_field", json!("max_tokens"));
		fireworks_compat.insert_ns("wire", "supports_sampling_params", json!(false));
		fireworks_compat.insert_ns("wire", "supports_usage_in_streaming", json!(false));
		req.model_policy = Some(model_policy(fireworks_compat, fireworks));
		let fireworks = encoded(&req, &Compat::default());
		assert_eq!(fireworks["reasoning_effort"], "high");
		assert_eq!(fireworks["max_tokens"], 512);
		assert!(fireworks.get("temperature").is_none());
		assert!(fireworks.get("stream_options").is_none());

		let mut ollama_policy = (*model_policy(Props::default(), BTreeMap::new())).clone();
		ollama_policy.omit_max_output_tokens = Some(true);
		req.model_policy = Some(Arc::new(ollama_policy));
		let ollama = encoded(&req, &Compat::default());
		assert_eq!(ollama["reasoning_effort"], "xhigh");
		assert!(ollama.get("max_completion_tokens").is_none());
	}

	#[test]
	fn budget_thinking_policy_projects_effort_budget_where_supported() {
		let mut req = request();
		req.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::High).build())
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		let mut compat = Props::default();
		compat.insert_ns("wire", "thinking_format", json!("openrouter"));
		let mut policy = (*model_policy(compat, BTreeMap::new())).clone();
		let thinking = policy.thinking.as_mut().unwrap();
		thinking.mode = ResolvedThinkingMode::Budget;
		thinking.effort_budgets.insert(Effort::High, 4096);
		req.model_policy = Some(Arc::new(policy));
		assert_eq!(
			encoded(&req, &Compat::default())["reasoning"],
			json!({"effort":"high","max_tokens":4096})
		);
	}

	#[test]
	fn conditional_policy_changes_role_extra_body_and_reasoning_placeholders() {
		let mut compat = Props::default();
		compat.insert_ns("wire", "extra_body", json!({"policy_variant":"base"}));
		compat.insert_ns(
			"wire",
			"when_thinking",
			json!({
				"supports_developer_role": true,
				"extra_body": {"policy_variant":"thinking"},
				"requires_reasoning_content_for_tool_calls": true,
				"allows_synthetic_reasoning_content_for_tool_calls": true,
				"requires_assistant_content_for_tool_calls": true
			}),
		);
		let mut req = request();
		req.thread.items.insert(
			0,
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::System)
						.parts(vec![Part::Text(Str::new_static("policy"))])
						.build(),
				))
				.props(Props::default())
				.build(),
		);
		req.thread.items.push(
			Item::builder()
				.seq(1)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::Assistant)
						.parts(Vec::new())
						.build(),
				))
				.props(Props::default())
				.build(),
		);
		req.thread.items.push(
			Item::builder()
				.seq(2)
				.kind(ItemKind::ToolCall(
					ToolCall::builder()
						.id("01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap())
						.name(Str::new_static("inspect"))
						.args_json(Bytes::from_static(b"{}"))
						.thought_signature(Bytes::new())
						.build(),
				))
				.props(Props::default())
				.build(),
		);
		req.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::High).build())
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		req.model_policy = Some(model_policy(compat, BTreeMap::new()));
		let thinking = encoded(&req, &Compat::default());
		assert_eq!(thinking["messages"][0]["role"], "developer");
		assert_eq!(thinking["policy_variant"], "thinking");
		let assistant = thinking["messages"].as_array().unwrap().last().unwrap();
		assert_eq!(assistant["reasoning_content"], ".");
		assert_eq!(assistant["content"], ".");

		req.thinking = None;
		let base = encoded(&req, &Compat::default());
		assert_eq!(base["messages"][0]["role"], "system");
		assert_eq!(base["policy_variant"], "base");
		assert!(base["messages"].as_array().unwrap().last().unwrap()["reasoning_content"].is_null());
	}

	#[test]
	fn model_policy_tool_choice_support_overrides_provider_compat() {
		let mut req = request();
		req.tool_choice = Some(
			Feature::builder()
				.value(ToolChoice::Required)
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		let mut disabled = Props::default();
		disabled.insert_ns("wire", "supports_tool_choice", json!(false));
		req.model_policy = Some(model_policy(disabled, BTreeMap::new()));
		assert!(
			encoded(&req, &Compat::default())
				.get("tool_choice")
				.is_none()
		);

		let mut unforceable = Props::default();
		unforceable.insert_ns("wire", "supports_tool_choice", json!(true));
		unforceable.insert_ns("wire", "supports_forced_tool_choice", json!(false));
		req.model_policy = Some(model_policy(unforceable, BTreeMap::new()));
		assert_eq!(encoded(&req, &Compat::default())["tool_choice"], "auto");
	}

	#[test]
	fn malformed_trusted_compat_is_reported_without_panicking() {
		let mut compat = Props::default();
		compat.insert_ns("wire", "supports_sampling_params", json!("not-a-boolean"));
		let mut req = request();
		req.model_policy = Some(model_policy(compat, BTreeMap::new()));
		let (_, unsupported) = OpenAiChatCodec.encode(&req, &Compat::default()).unwrap();
		assert!(
			unsupported
				.iter()
				.any(|entry| { entry.what == "model_policy.compat:wire/supports_sampling_params" })
		);
	}

	fn changed_keys(left: &Value, right: &Value) -> BTreeSet<String> {
		let left = left.as_object().unwrap();
		let right = right.as_object().unwrap();
		left
			.keys()
			.chain(right.keys())
			.filter(|key| left.get(*key) != right.get(*key))
			.cloned()
			.collect()
	}

	#[test]
	fn every_request_compat_axis_changes_only_its_wire_field() {
		let mut req = request();
		req.thread.items.insert(
			0,
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::System)
						.parts(vec![Part::Text(Str::from("one"))])
						.build(),
				))
				.props(Props::default())
				.build(),
		);
		req.thread.items.insert(
			1,
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::System)
						.parts(vec![Part::Text(Str::from("two"))])
						.build(),
				))
				.props(Props::default())
				.build(),
		);
		if let ItemKind::Message(message) = &mut req.thread.items[2].kind {
			message.parts.push(Part::Blob(
				BlobPart::builder()
					.hash([0; 32])
					.mime(Str::from("image/png"))
					.size(1)
					.inline(Bytes::from_static(b"x"))
					.build(),
			));
		}
		req.tools = ["one", "two"]
			.into_iter()
			.map(|name| {
				ToolDef::builder()
					.name(Str::from(name))
					.description(Str::default())
					.schema_json(Bytes::from_static(b"{\"type\":\"object\",\"properties\":{}}"))
					.strict(name == "one")
					.build()
			})
			.collect();
		req.tool_choice = Some(
			Feature::builder()
				.value(ToolChoice::Named(Str::from("two")))
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		req.thinking = Some(
			Feature::builder()
				.value(
					Reasoning::builder()
						.effort(Effort::High)
						.hide_summary(false)
						.build(),
				)
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		req.sampling = Some(
			Sampling::builder()
				.temperature(0.2)
				.top_p(0.9)
				.frequency_penalty(0.1)
				.stop(vec![Str::from("END")])
				.max_output_tokens(42)
				.build(),
		);
		req.provider_options
			.as_mut()
			.expect("request has provider options")
			.insert_ns("openai", "thinking", serde_json::json!({"type":"enabled"}));
		let baseline = encoded(&req, &Compat::default());
		let check = |compat: Compat, expected: &[&str]| {
			let actual = changed_keys(&baseline, &encoded(&req, &compat));
			let expected = expected.iter().map(|key| (*key).to_owned()).collect();
			assert_eq!(actual, expected);
		};

		let mut compat = Compat::default();
		compat.usage_in_streaming = false;
		check(compat, &["stream_options"]);
		let mut compat = Compat::default();
		compat.multiple_system_messages = false;
		check(compat, &["messages"]);
		let mut compat = Compat::default();
		compat.max_tokens_field = MaxTokensField::MaxTokens;
		check(compat, &["max_completion_tokens", "max_tokens"]);
		let mut compat = Compat::default();
		compat.sampling_params = false;
		check(compat, &["temperature", "top_p"]);
		let mut compat = Compat::default();
		compat.penalties = false;
		check(compat, &["frequency_penalty"]);
		let mut compat = Compat::default();
		compat.tool_strict_mode = ToolStrictMode::AllStrict;
		check(compat, &["tools"]);
		let mut compat = Compat::default();
		compat.tool_strict_mode = ToolStrictMode::None;
		check(compat, &["tools"]);
		let mut compat = Compat::default();
		compat.named_tool_choice = false;
		check(compat, &["tool_choice", "tools"]);
		let mut compat = Compat::default();
		compat.forced_tool_choice = false;
		check(compat, &["tool_choice"]);
		let mut compat = Compat::default();
		compat.reasoning_wire_format = ReasoningWireFormat::OpenRouter;
		check(compat, &["reasoning", "reasoning_effort"]);
		let mut compat = Compat::default();
		compat.thinking_tool_choice_conflict = ThinkingToolChoiceConflict::DropThinkingWhenForced;
		check(compat, &["reasoning_effort"]);
		let mut compat = Compat::default();
		compat.thinking_tool_choice_conflict = ThinkingToolChoiceConflict::DropThinkingWhenEffort;
		check(compat, &["thinking"]);
		let mut compat = Compat::default();
		compat.image_encoding_format = ImageEncodingFormat::None;
		check(compat, &["messages"]);
		let mut compat = Compat::default();
		compat.stop_sequences = false;
		check(compat, &["stop"]);
	}

	#[test]
	fn tool_call_id_profile_changes_only_tool_call_ids() {
		let id: CallId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
		let mut req = request();
		req.thread.items.push(
			Item::builder()
				.seq(1)
				.kind(ItemKind::ToolCall(
					ToolCall::builder()
						.id(id)
						.name(Str::from("lookup"))
						.args_json(Bytes::from_static(b"{}"))
						.thought_signature(Bytes::new())
						.build(),
				))
				.props(Props::default())
				.build(),
		);
		let baseline = encoded(&req, &Compat::default());
		let mut compat = Compat::default();
		compat.tool_call_id_profile = CompatToolCallIdProfile::Mistral9Alnum;
		let mistral = encoded(&req, &compat);
		assert_eq!(changed_keys(&baseline, &mistral), BTreeSet::from(["messages".to_owned()]));
		let wire_id = mistral["messages"][1]["tool_calls"][0]["id"]
			.as_str()
			.unwrap();
		assert_eq!(wire_id.len(), 9);
		assert!(wire_id.bytes().all(|byte| byte.is_ascii_alphanumeric()));
	}

	#[test]
	fn chat_reasoning_formats_change_only_their_named_field() {
		let mut req = request();
		req.thinking = Some(
			Feature::builder()
				.value(
					Reasoning::builder()
						.effort(Effort::High)
						.hide_summary(false)
						.build(),
				)
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		let baseline = encoded(&req, &Compat::default());

		let mut qwen = Compat::default();
		qwen.reasoning_wire_format = ReasoningWireFormat::QwenEnableThinking;
		let qwen = encoded(&req, &qwen);
		assert_eq!(
			changed_keys(&baseline, &qwen),
			BTreeSet::from(["enable_thinking".to_owned(), "reasoning_effort".to_owned()])
		);
		assert_eq!(qwen["enable_thinking"], true);

		let mut nim = Compat::default();
		nim.reasoning_wire_format = ReasoningWireFormat::NvidiaChatTemplateKwargs;
		let nim = encoded(&req, &nim);
		assert_eq!(
			changed_keys(&baseline, &nim),
			BTreeSet::from(["chat_template_kwargs".to_owned(), "reasoning_effort".to_owned()])
		);
		assert_eq!(nim["chat_template_kwargs"]["enable_thinking"], true);
	}

	#[test]
	fn compat_matrix_changes_only_selected_fields() {
		let mut req = request();
		req.sampling = Some(
			Sampling::builder()
				.temperature(0.2)
				.top_p(0.9)
				.frequency_penalty(0.1)
				.max_output_tokens(42)
				.build(),
		);
		let baseline = encoded(&req, &Compat::default());
		let mut compat = Compat::default();
		compat.usage_in_streaming = false;
		let changed = encoded(&req, &compat);
		assert_eq!(changed.get("stream_options"), None);
		assert_eq!(changed["messages"], baseline["messages"]);
		let mut compat = Compat::default();
		compat.max_tokens_field = MaxTokensField::MaxTokens;
		let changed = encoded(&req, &compat);
		assert_eq!(changed["max_tokens"], 42);
		assert!(changed.get("max_completion_tokens").is_none());
		let mut compat = Compat::default();
		compat.sampling_params = false;
		let changed = encoded(&req, &compat);
		assert!(changed.get("temperature").is_none());
		assert_eq!(changed["frequency_penalty"], baseline["frequency_penalty"]);
		let mut compat = Compat::default();
		compat.penalties = false;
		let changed = encoded(&req, &compat);
		assert!(changed.get("frequency_penalty").is_none());
		assert_eq!(changed["temperature"], baseline["temperature"]);
	}

	#[test]
	fn coalesces_system_messages() {
		let mut req = request();
		req.thread.items.insert(
			0,
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::System)
						.parts(vec![Part::Text(Str::from("one"))])
						.build(),
				))
				.props(Props::default())
				.build(),
		);
		req.thread.items.insert(
			1,
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::System)
						.parts(vec![Part::Text(Str::from("two"))])
						.build(),
				))
				.props(Props::default())
				.build(),
		);
		let mut compat = Compat::default();
		compat.multiple_system_messages = false;
		let body = encoded(&req, &compat);
		assert_eq!(body["messages"][0]["content"], "one\n\ntwo");
		assert_eq!(body["messages"].as_array().unwrap().len(), 2);
	}

	#[test]
	fn named_choice_fallback_filters_tools() {
		let mut req = request();
		req.tools = ["one", "two"]
			.into_iter()
			.map(|name| {
				ToolDef::builder()
					.name(Str::from(name))
					.description(Str::default())
					.schema_json(Bytes::from_static(b"{}"))
					.strict(false)
					.build()
			})
			.collect();
		req.tool_choice = Some(
			Feature::builder()
				.value(ToolChoice::Named(Str::from("two")))
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		let mut compat = Compat::default();
		compat.named_tool_choice = false;
		let body = encoded(&req, &compat);
		assert_eq!(body["tool_choice"], "required");
		assert_eq!(body["tools"].as_array().unwrap().len(), 1);
		assert_eq!(body["tools"][0]["function"]["name"], "two");
	}

	#[test]
	fn fragmented_tool_arguments_are_byte_identical() {
		let codec = OpenAiChatCodec;
		let mut state = DecodeState::default();
		let mut seen = Vec::new();
		for data in [br#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"wire","function":{"name":"raw","arguments":"{ \"n\" : 1"}}]}}]}"#.as_slice(), br#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":".00 }"}}]},"finish_reason":"tool_calls"}]}"#.as_slice()] {
			for event in codec.decode(Frame::Data(data), &mut state).unwrap() { if let TurnEvent::PartDelta { chunk, .. } = event { seen.extend_from_slice(&chunk); } }
		}
		assert_eq!(seen, br#"{ "n" : 1.00 }"#);
	}

	#[test]
	fn streaming_fixture_decodes_reasoning_tools_usage_and_done() {
		let codec = OpenAiChatCodec;
		let mut state = DecodeState::default();
		let mut text = Vec::new();
		let mut thinking = Vec::new();
		let mut tool_args = Vec::new();
		let mut kinds = BTreeMap::new();
		let mut outcome = None;
		for record in
			include_str!("../tests/fixtures/openai_chat/stream.tool_reasoning_usage.sse").split("\n\n")
		{
			let Some(data) = record.strip_prefix("data: ") else {
				continue;
			};
			for event in codec
				.decode(Frame::Data(data.trim_end().as_bytes()), &mut state)
				.unwrap()
			{
				match event {
					TurnEvent::PartStart { index, kind, .. } => {
						kinds.insert(index, kind);
					},
					TurnEvent::PartDelta { index, chunk } => match kinds.get(&index) {
						Some(StreamPartKind::Text) => text.extend_from_slice(&chunk),
						Some(StreamPartKind::Thinking) => thinking.extend_from_slice(&chunk),
						Some(StreamPartKind::ToolCall) => tool_args.extend_from_slice(&chunk),
						None => panic!("delta preceded its part start"),
						_ => {},
					},
					TurnEvent::Outcome(value) => outcome = Some(value),
					_ => {},
				}
			}
		}
		assert_eq!(text, b"Checking.");
		assert_eq!(thinking, b"Need weather. ");
		assert_eq!(tool_args, r#"{"city":"Zürich"}"#.as_bytes());
		let outcome = outcome.expect("DONE emits an outcome");
		assert_eq!(outcome.stop, StopReason::ToolUse);
		let usage = outcome.usage.unwrap();
		assert_eq!((usage.input_tokens, usage.output_tokens, usage.cache_read_tokens), (20, 9, 12));
	}

	#[test]
	fn projects_presence_aware_choices_strict_schemas_developer_role_and_images() {
		let mut req = request();
		req.thread.items.insert(
			0,
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::System)
						.parts(vec![Part::Text(Str::from("policy"))])
						.build(),
				))
				.props(Props::default())
				.build(),
		);
		if let ItemKind::Message(message) = &mut req.thread.items[1].kind {
			message.parts.push(Part::Blob(
				BlobPart::builder()
					.hash([0; 32])
					.mime(Str::from("image/png"))
					.size(1)
					.inline(Bytes::from_static(b"x"))
					.build(),
			));
		}
		req.provider_options.as_mut().unwrap().insert_ns(
			"openai",
			"system_role",
			Value::from("developer"),
		);
		req.tools = vec![
			ToolDef::builder()
				.name(Str::from("lookup"))
				.description(Str::from("Lookup"))
				.schema_json(Bytes::from_static(
					br#"{"type":"object","properties":{"q":{"type":"string"}}}"#,
				))
				.strict(true)
				.build(),
		];
		req.response_format = Some(
			Feature::builder()
				.value(
					ResponseFormat::builder()
						.kind(ResponseFormatKind::JsonSchema(
							JsonSchema::builder()
								.name(Str::from("answer"))
								.schema_json(Bytes::from_static(
									br#"{"type":"object","properties":{"ok":{"type":"boolean"}}}"#,
								))
								.strict(false)
								.build(),
						))
						.build(),
				)
				.on_unsupported(Fallback::Error)
				.build(),
		);
		for (choice, expected) in [
			(ToolChoice::Auto, Value::from("auto")),
			(ToolChoice::None, Value::from("none")),
			(ToolChoice::Required, Value::from("required")),
			(
				ToolChoice::Named(Str::from("lookup")),
				json!({"type":"function","function":{"name":"lookup"}}),
			),
		] {
			req.tool_choice = Some(
				Feature::builder()
					.value(choice)
					.on_unsupported(Fallback::Error)
					.build(),
			);
			let body = encoded(&req, &Compat::default());
			assert_eq!(body["tool_choice"], expected);
			assert_eq!(body["messages"][0]["role"], "developer");
			assert!(body["messages"][1]["content"].is_array());
			assert!(
				body["messages"][1]["content"][1]["image_url"]["url"]
					.as_str()
					.unwrap()
					.starts_with("data:image/png;base64,")
			);
			assert_eq!(body["tools"][0]["function"]["strict"], true);
			assert_eq!(body["tools"][0]["function"]["parameters"]["required"], json!(["q"]));
			assert_eq!(body["tools"][0]["function"]["parameters"]["additionalProperties"], false);
			assert_eq!(body["response_format"]["json_schema"]["strict"], false);
		}
	}

	fn decode_fixture(fixture: &str) -> Vec<TurnEvent> {
		let codec = OpenAiChatCodec;
		let mut state = DecodeState::default();
		let mut events = Vec::new();
		for record in fixture.split("\n\n") {
			let Some(data) = record.strip_prefix("data: ") else {
				continue;
			};
			events.extend(
				codec
					.decode(Frame::Data(data.trim_end().as_bytes()), &mut state)
					.unwrap(),
			);
		}
		events
	}

	#[test]
	fn parity_fixture_preserves_parallel_multi_choice_metadata_and_trailing_usage() {
		let events = decode_fixture(include_str!("../tests/fixtures/openai_chat/stream.parity.sse"));
		assert_eq!(
			events
				.iter()
				.filter(|event| matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)))
				.count(),
			1
		);
		let outcome = events
			.into_iter()
			.find_map(|event| match event {
				TurnEvent::Outcome(outcome) => Some(outcome),
				_ => None,
			})
			.expect("fixture has one outcome");
		assert_eq!(outcome.stop, StopReason::ContentFilter);
		assert_eq!(outcome.provider, "azure-east");
		assert_eq!(outcome.model, "gpt-parity");
		let usage = outcome.usage.expect("usage-only chunk is retained");
		assert_eq!(
			(
				usage.input_tokens,
				usage.output_tokens,
				usage.cache_read_tokens,
				usage.cache_write_tokens,
			),
			(10, 4, 6, 2)
		);
		assert_eq!(
			usage.detail.get_ns("openai", "usage").unwrap()["completion_tokens_details"]
				["reasoning_tokens"],
			2
		);
		let calls = outcome
			.output
			.iter()
			.filter_map(|item| match &item.kind {
				ItemKind::ToolCall(call) => Some((call.name.as_str(), call.args_json.as_ref())),
				_ => None,
			})
			.collect::<Vec<_>>();
		assert_eq!(calls, vec![
			("alpha", br#"{"x":1}"#.as_slice()),
			("beta", br#"{"y":2}"#.as_slice())
		]);
		let choice_one = outcome
			.output
			.iter()
			.find(|item| item.props.get_ns("openai", "choice_index") == Some(&Value::from(1)))
			.expect("second choice retained");
		assert!(choice_one.props.get_ns("openai", "logprobs").is_some());
		assert!(choice_one.props.get_ns("openai", "annotations").is_some());
		assert_eq!(choice_one.props.get_ns("openai", "refusal"), Some(&Value::from(" blocked")));
	}

	#[test]
	fn tool_calls_precede_length_but_not_content_filter() {
		let events =
			decode_fixture(include_str!("../tests/fixtures/openai_chat/stream.tool_length.sse"));
		let outcome = events
			.into_iter()
			.find_map(|event| match event {
				TurnEvent::Outcome(outcome) => Some(outcome),
				_ => None,
			})
			.expect("fixture has one outcome");
		assert_eq!(outcome.stop, StopReason::ToolUse);
		let call = outcome
			.output
			.iter()
			.find_map(|item| match &item.kind {
				ItemKind::ToolCall(call) => Some(call),
				_ => None,
			})
			.expect("parallel tool call retained");
		assert_eq!(call.args_json, br#"{"q":"hello"}"#.as_slice());
	}

	#[test]
	fn leaked_thinking_markup_remains_visible_until_the_configured_healer() {
		let events =
			decode_fixture(include_str!("../tests/fixtures/openai_chat/stream.leaked_tags.sse"));
		let outcome = events
			.into_iter()
			.find_map(|event| match event {
				TurnEvent::Outcome(outcome) => Some(outcome),
				_ => None,
			})
			.expect("leaked-markup fixture completes");
		assert_eq!(outcome.stop, StopReason::ToolUse);
		let mut text = String::new();
		let mut thinking = String::new();
		let mut call = None;
		for item in &outcome.output {
			match &item.kind {
				ItemKind::Message(message) => {
					for part in &message.parts {
						match part {
							Part::Text(value) => text.push_str(value),
							Part::Thinking(value) => thinking.push_str(&value.text),
							_ => {},
						}
					}
				},
				ItemKind::ToolCall(value) => call = Some(value),
				_ => {},
			}
		}
		assert_eq!(thinking, "");
		assert_eq!(text, "<think>Need to inspect</think>Answer.\n");
		let call = call.expect("tool markup became a tool call");
		assert_eq!(call.name, "read");
		assert_eq!(call.args_json, br#"{"path":"README.md"}"#.as_slice());
	}
	#[test]
	fn azure_and_openrouter_error_envelopes_are_terminal() {
		for (fixture, expected) in [
			(include_str!("../tests/fixtures/openai_chat/error.azure.json"), "Azure policy"),
			(
				include_str!("../tests/fixtures/openai_chat/error.openrouter.json"),
				"MALFORMED_FUNCTION_CALL",
			),
		] {
			let codec = OpenAiChatCodec;
			let mut state = DecodeState::default();
			let events = codec
				.decode(Frame::Data(fixture.trim().as_bytes()), &mut state)
				.unwrap();
			assert!(matches!(
				events.as_slice(),
				[TurnEvent::Error(error)] if error.detail.contains(expected)
			));
			assert!(codec.decode(Frame::Done, &mut state).unwrap().is_empty());
		}
	}

	#[test]
	fn hostile_sse_boundaries_and_message_fallback_preserve_raw_thinking_tags() {
		let fixture: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/openai_chat/stream.chunk_boundaries.json"
		))
		.unwrap();
		let bytes = fixture["concatenated_utf8"].as_str().unwrap().as_bytes();
		let boundaries = fixture["boundaries"]
			.as_array()
			.unwrap()
			.iter()
			.map(|value| value.as_u64().unwrap() as usize)
			.collect::<Vec<_>>();
		let mut sse = SseDecoder::new();
		let codec = OpenAiChatCodec;
		let mut state = DecodeState::default();
		let mut events = Vec::new();
		let mut start = 0;
		for end in boundaries.into_iter().chain(std::iter::once(bytes.len())) {
			for event in sse.push(Bytes::copy_from_slice(&bytes[start..end])) {
				events.extend(codec.decode(Frame::Data(&event.data), &mut state).unwrap());
			}
			start = end;
		}
		events.extend(codec.decode(Frame::Done, &mut state).unwrap());
		let outcome = events
			.into_iter()
			.find_map(|event| match event {
				TurnEvent::Outcome(outcome) => Some(outcome),
				_ => None,
			})
			.expect("boundary fixture completes");
		assert!(matches!(
			&outcome.output[0].kind,
			ItemKind::Message(message)
				if message.parts == vec![Part::Text(Str::from("Café"))]
		));

		let mut state = DecodeState::default();
		let first = br#"{"choices":[{"index":0,"message":{"content":"<thi"},"finish_reason":null}]}"#;
		let second = br#"{"choices":[{"index":0,"message":{"content":"nk>private</think>public"},"finish_reason":"stop"}]}"#;
		let mut projected = Vec::new();
		projected.extend(codec.decode(Frame::Data(first), &mut state).unwrap());
		projected.extend(codec.decode(Frame::Data(second), &mut state).unwrap());
		projected.extend(codec.decode(Frame::Done, &mut state).unwrap());
		let outcome = projected
			.into_iter()
			.find_map(|event| match event {
				TurnEvent::Outcome(outcome) => Some(outcome),
				_ => None,
			})
			.expect("message fallback completes");
		let ItemKind::Message(message) = &outcome.output[0].kind else {
			panic!("expected assistant message")
		};
		assert_eq!(message.parts, vec![Part::Text(Str::from("<think>private</think>public"))]);
	}

	#[test]
	fn strict_response_schema_normalizes_nested_objects_before_advertising_strictness() {
		let mut req = request();
		req.response_format = Some(
			Feature::builder()
				.value(
					ResponseFormat::builder()
						.kind(ResponseFormatKind::JsonSchema(
							JsonSchema::builder()
								.name(Str::from("nested"))
								.schema_json(Bytes::from_static(
									br#"{"type":"object","properties":{"outer":{"type":"object","properties":{"leaf":{"type":"string"}}}}}"#,
								))
								.strict(true)
								.build(),
						))
						.build(),
				)
				.on_unsupported(Fallback::Error)
				.build(),
		);
		let (wire, unsupported) = OpenAiChatCodec.encode(&req, &Compat::default()).unwrap();
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		let wire: Value = serde_json::from_slice(&wire).unwrap();
		let schema = &wire["response_format"]["json_schema"];
		assert_eq!(schema["strict"], true);
		assert_eq!(schema["schema"]["required"], json!(["outer"]));
		assert_eq!(schema["schema"]["additionalProperties"], false);
		assert_eq!(schema["schema"]["properties"]["outer"]["anyOf"][0]["required"], json!(["leaf"]));
		assert_eq!(
			schema["schema"]["properties"]["outer"]["anyOf"][0]["additionalProperties"],
			false
		);
	}

	#[test]
	fn provider_reported_cost_is_authoritative_and_invalid_values_are_ignored() {
		let codec = OpenAiChatCodec;
		let mut state = DecodeState::default();
		let chunk = br#"{"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"cost":0.012345678}}"#;
		let mut events = codec.decode(Frame::Data(chunk), &mut state).unwrap();
		events.extend(codec.decode(Frame::Done, &mut state).unwrap());
		let outcome = events
			.into_iter()
			.find_map(|event| match event {
				TurnEvent::Outcome(outcome) => Some(outcome),
				_ => None,
			})
			.expect("stream completes");
		let cost = outcome
			.cost
			.expect("numeric provider cost is canonicalized");
		assert_eq!(cost.nanos_usd, 12_345_678);
		assert!(!cost.estimated);
		assert_eq!(
			outcome
				.usage
				.as_ref()
				.and_then(|usage| usage.detail.get_ns("openai", "usage"))
				.and_then(|usage| usage.get("cost")),
			Some(&json!(0.012345678))
		);

		for invalid in [json!(-1), json!("0.1"), json!(1e100)] {
			let mut state = DecodeState::default();
			let chunk = serde_json::to_vec(&json!({
				"choices": [{"index": 0, "finish_reason": "stop"}],
				"usage": {"cost": invalid},
			}))
			.unwrap();
			let mut events = codec.decode(Frame::Data(&chunk), &mut state).unwrap();
			events.extend(codec.decode(Frame::Done, &mut state).unwrap());
			let outcome = events
				.into_iter()
				.find_map(|event| match event {
					TurnEvent::Outcome(outcome) => Some(outcome),
					_ => None,
				})
				.expect("invalid-cost stream still completes");
			assert!(outcome.cost.is_none());
		}
	}
}
