//! Native Ollama `/api/chat` request and NDJSON stream codec.

pub mod discovery;

use std::collections::BTreeMap;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use bytes::Bytes;
use omp_core::Str;
use omp_llm_catalog::{TransportId, compat::Compat};
use omp_llm_transport::{
	DecodeState, Frame, Transport,
	normalize::{normalize_tools, with_tool_use_precedence},
};
use omp_llm_types::{
	Accuracy, CallId, ChatOutcome, ChatRequest, Effort, Error, Fallback, Item, ItemKind, Message,
	Part, Props, Reasoning, ResolvedModelPolicy, ResolvedThinkingMode, Role, StopReason,
	StreamPartKind, Thinking, ToolCall, ToolChoice, TurnError, TurnErrorKind, TurnEvent,
	Unsupported, UnsupportedAction, Usage,
};
use serde_json::{Map, Value, json};
use smallvec::SmallVec;

const CLOUD_OUTPUT_CAP: u64 = 65_536;
const OPEN_SCHEMA_TYPES: [&str; 6] = ["string", "number", "boolean", "object", "array", "null"];

/// Codec for Ollama Cloud's native `POST /api/chat` protocol.
#[derive(Debug, Default)]
pub struct OllamaChatCodec;

impl Transport for OllamaChatCodec {
	fn id(&self) -> TransportId {
		TransportId::OllamaChat
	}

	fn encode(
		&self,
		req: &ChatRequest,
		compat: &Compat,
	) -> Result<(Bytes, Vec<Unsupported>), Error> {
		let mut unsupported = Vec::new();
		let mut body = Map::new();
		body.insert("model".into(), Value::String(req.model.to_string()));
		body.insert("messages".into(), Value::Array(encode_messages(req, &mut unsupported)?));
		body.insert("stream".into(), Value::Bool(true));

		let (tools, mut reports) = normalize_tools(compat.tool_schema_flavor, &req.tools);
		unsupported.append(&mut reports);
		let selected_name = req
			.tool_choice
			.as_ref()
			.and_then(|choice| match &choice.value {
				ToolChoice::Named(name) => Some(name.as_str()),
				_ => None,
			});
		let mut encoded_tools = Vec::new();
		for tool in &tools {
			if selected_name.is_some_and(|selected| selected != tool.name) {
				continue;
			}
			let schema: Value = serde_json::from_slice(&tool.schema_json).map_err(|error| {
				Error::Provider(Str::from(format!(
					"invalid Ollama tool schema for {}: {error}",
					tool.name
				)))
			})?;
			encoded_tools.push(json!({
				"type": "function",
				"function": {
					"name": tool.name.as_str(),
					"description": tool.description.as_str(),
					"parameters": sanitize_schema(schema),
				}
			}));
		}
		if !encoded_tools.is_empty() {
			body.insert("tools".into(), Value::Array(encoded_tools));
		}
		if let Some(choice) = &req.tool_choice {
			let wire = match &choice.value {
				ToolChoice::Auto => None,
				ToolChoice::None => Some("none"),
				ToolChoice::Required | ToolChoice::Named(_) => Some("required"),
				_ => {
					report_feature(
						&mut unsupported,
						"tool_choice",
						"Ollama supports auto, none, required, and named function selection",
						choice.on_unsupported,
					)?;
					None
				},
			};
			if matches!(&choice.value, ToolChoice::Named(name) if !req.tools.iter().any(|tool| tool.name == name.as_str()))
			{
				report_feature(
					&mut unsupported,
					"tool_choice",
					"the named Ollama tool is not advertised",
					choice.on_unsupported,
				)?;
			}
			if let Some(wire) = wire {
				body.insert("tool_choice".into(), Value::String(wire.into()));
			}
		}

		if let Some(reasoning) = &req.thinking {
			encode_reasoning(
				&mut body,
				&reasoning.value,
				req.model_policy.as_deref(),
				&mut unsupported,
				reasoning.on_unsupported,
			)?;
		}
		if let Some(sampling) = &req.sampling {
			let mut options = Map::new();
			if req
				.model_policy
				.as_deref()
				.and_then(|policy| policy.omit_max_output_tokens)
				!= Some(true)
				&& let Some(requested) = sampling.max_output_tokens
			{
				let bounded = requested.min(CLOUD_OUTPUT_CAP);
				options.insert("num_predict".into(), Value::from(bounded));
				if bounded != requested {
					unsupported.push(
						Unsupported::builder()
							.what(Str::new_static("sampling.max_output_tokens"))
							.detail(Str::new_static("Ollama Cloud caps generated output at 65,536 tokens"))
							.action(UnsupportedAction::Clamped)
							.build(),
					);
				}
			}
			let sampling_params =
				model_compat_bool(req, "supports_sampling_params", reasoning_enabled(req))
					.unwrap_or(compat.sampling_params);
			if sampling_params {
				insert_option(&mut options, "temperature", sampling.temperature);
				insert_option(&mut options, "top_p", sampling.top_p);
				insert_option(&mut options, "top_k", sampling.top_k);
				insert_option(&mut options, "min_p", sampling.min_p);
				insert_option(&mut options, "repeat_penalty", sampling.repetition_penalty);
				if let Some(stop) = &sampling.stop {
					options.insert("stop".into(), json!(stop));
				}
			} else {
				for (name, present) in [
					("sampling.temperature", sampling.temperature.is_some()),
					("sampling.top_p", sampling.top_p.is_some()),
					("sampling.top_k", sampling.top_k.is_some()),
					("sampling.min_p", sampling.min_p.is_some()),
					("sampling.repetition_penalty", sampling.repetition_penalty.is_some()),
					("sampling.stop", sampling.stop.is_some()),
				] {
					if present {
						unsupported.push(dropped(name, "model compatibility disables sampling controls"));
					}
				}
			}
			for (name, present) in [
				("sampling.frequency_penalty", sampling.frequency_penalty.is_some()),
				("sampling.presence_penalty", sampling.presence_penalty.is_some()),
			] {
				if present {
					unsupported.push(dropped(name, "Ollama has no matching native sampling control"));
				}
			}
			if !options.is_empty() {
				body.insert("options".into(), Value::Object(options));
			}
		}
		if req.cache.is_some() {
			unsupported.push(dropped("cache", "Ollama Cloud does not expose portable cache controls"));
		}
		if req.response_format.is_some() {
			unsupported.push(dropped(
				"response_format",
				"Pi's native Ollama Cloud path does not project structured output controls",
			));
		}
		if req.service_tier.is_some() || req.service_tier_by_family.is_some() {
			unsupported.push(dropped("service_tier", "Ollama Cloud has no portable service tier"));
		}
		if req.task_budget.is_some() {
			unsupported.push(dropped("task_budget", "task budgets are not Ollama wire controls"));
		}
		if req.responses_include.is_some() {
			unsupported.push(dropped(
				"responses_include",
				"Responses include records do not exist in the Ollama chat protocol",
			));
		}
		for key in req
			.provider_options
			.iter()
			.flat_map(|options| options.0.keys())
		{
			unsupported.push(dropped(
				key.as_str(),
				"property is not supported by the native Ollama Cloud path",
			));
		}

		serde_json::to_vec(&body)
			.map(Bytes::from)
			.map(|body| (body, unsupported))
			.map_err(|error| {
				Error::Provider(Str::from(format!("Ollama request serialization failed: {error}")))
			})
	}

	fn decode(
		&self,
		frame: Frame<'_>,
		state: &mut DecodeState,
	) -> Result<SmallVec<TurnEvent, 2>, Error> {
		match frame {
			Frame::Data(data) | Frame::Event { data, .. } => decode_line(data, state),
			Frame::Done => Ok(finish_incomplete(state)),
			_ => Ok(SmallVec::new()),
		}
	}
}

fn encode_reasoning(
	body: &mut Map<String, Value>,
	reasoning: &Reasoning,
	model_policy: Option<&ResolvedModelPolicy>,
	unsupported: &mut Vec<Unsupported>,
	fallback: Fallback,
) -> Result<(), Error> {
	if reasoning.budget_tokens.is_some() {
		report_feature(
			unsupported,
			"thinking.budget_tokens",
			"Ollama accepts qualitative effort rather than a token budget",
			fallback,
		)?;
	}
	if reasoning.hide_summary.is_some() {
		report_feature(
			unsupported,
			"thinking.hide_summary",
			"Ollama streams reasoning text when thinking is enabled",
			fallback,
		)?;
	}
	let thinking_policy = if let Some(model_policy) = model_policy {
		let Some(thinking) = model_policy.thinking.as_ref() else {
			report_feature(
				unsupported,
				"thinking",
				"resolved model policy does not advertise native thinking",
				fallback,
			)?;
			return Ok(());
		};
		if thinking.mode != ResolvedThinkingMode::Effort {
			report_feature(
				unsupported,
				"thinking",
				"resolved thinking mode has no Ollama Cloud projection",
				fallback,
			)?;
			return Ok(());
		}
		Some(thinking)
	} else {
		None
	};
	let think = match reasoning.effort {
		Some(Effort::Off) => Some(Value::Bool(false)),
		Some(effort) => {
			let mapped = thinking_policy
				.and_then(|policy| policy.effort_map.get(&effort))
				.map_or_else(|| ollama_effort(effort), Str::as_str);
			Some(Value::String(mapped.into()))
		},
		None => None,
	};
	if let Some(think) = think {
		body.insert("think".into(), think);
	}
	Ok(())
}

const fn ollama_effort(effort: Effort) -> &'static str {
	match effort {
		Effort::Off => "off",
		Effort::Minimal | Effort::Low => "low",
		Effort::Medium => "medium",
		Effort::High => "high",
		Effort::XHigh => "xhigh",
		Effort::Max => "max",
		_ => "medium",
	}
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

fn insert_option<T: Into<Value>>(object: &mut Map<String, Value>, key: &str, value: Option<T>) {
	if let Some(value) = value {
		object.insert(key.into(), value.into());
	}
}

fn encode_messages(
	req: &ChatRequest,
	unsupported: &mut Vec<Unsupported>,
) -> Result<Vec<Value>, Error> {
	let mut messages = Vec::new();
	for item in &req.thread.items {
		match &item.kind {
			ItemKind::Message(message) => messages.push(encode_message(message, unsupported)?),
			ItemKind::ToolCall(call) => {
				let arguments: Value = serde_json::from_slice(&call.args_json).map_err(|error| {
					Error::Provider(Str::from(format!("invalid Ollama tool arguments: {error}")))
				})?;
				let tool_call = json!({
					"type": "function",
					"function": { "name": call.name.as_str(), "arguments": arguments }
				});
				if let Some(previous) = messages
					.last_mut()
					.and_then(Value::as_object_mut)
					.filter(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
				{
					previous
						.entry("tool_calls")
						.or_insert_with(|| Value::Array(Vec::new()))
						.as_array_mut()
						.expect("tool_calls was inserted as an array")
						.push(tool_call);
				} else {
					messages.push(json!({"role":"assistant", "content":"", "tool_calls":[tool_call]}));
				}
				if !call.thought_signature.is_empty() {
					unsupported.push(dropped(
						"thread.tool_call.thought_signature",
						"Ollama cannot replay provider thought signatures",
					));
				}
			},
			ItemKind::ToolResult(result) => {
				let (content, images) = encode_parts(&result.parts, Role::User, unsupported);
				let mut message = Map::new();
				message.insert("role".into(), Value::String("tool".into()));
				message.insert("content".into(), Value::String(content));
				message.insert("tool_name".into(), Value::String(result.name.to_string()));
				if !images.is_empty() {
					message.insert(
						"images".into(),
						Value::Array(images.into_iter().map(Value::String).collect()),
					);
				}
				messages.push(Value::Object(message));
				unsupported.push(dropped(
					"thread.tool_result.call_id",
					"Ollama pairs tool results by tool name and order rather than call id",
				));
			},
			_ => unsupported.push(dropped("thread.item", "item kind is unsupported by Ollama chat")),
		}
	}
	Ok(messages)
}

fn encode_message(message: &Message, unsupported: &mut Vec<Unsupported>) -> Result<Value, Error> {
	let role = match message.role {
		Role::System => "system",
		Role::User => "user",
		Role::Assistant => "assistant",
		_ => return Err(Error::Provider(Str::new_static("unsupported Ollama message role"))),
	};
	let (content, images) = encode_parts(&message.parts, message.role, unsupported);
	let mut object = Map::new();
	object.insert("role".into(), Value::String(role.into()));
	object.insert("content".into(), Value::String(content));
	if !images.is_empty() {
		object.insert("images".into(), Value::Array(images.into_iter().map(Value::String).collect()));
	}
	Ok(Value::Object(object))
}

fn encode_parts(
	parts: &[Part],
	role: Role,
	unsupported: &mut Vec<Unsupported>,
) -> (String, Vec<String>) {
	let mut text = String::new();
	let mut images = Vec::new();
	for part in parts {
		match part {
			Part::Text(value) => {
				if !text.is_empty() {
					text.push('\n');
				}
				text.push_str(value);
			},
			Part::Thinking(_) if role == Role::Assistant => unsupported.push(dropped(
				"thread.assistant.thinking",
				"Ollama Cloud rejects assistant thinking fields in request history",
			)),
			Part::Thinking(_) => unsupported.push(dropped(
				"thread.message.thinking",
				"reasoning history is only meaningful on assistant messages",
			)),
			Part::Blob(blob) if blob.mime.starts_with("image/") && !blob.inline.is_empty() => {
				images.push(BASE64.encode(&blob.inline));
			},
			Part::Blob(_) => unsupported
				.push(dropped("thread.message.image", "Ollama images require inline image bytes")),
			_ => unsupported
				.push(dropped("thread.message.part", "part kind is unsupported by Ollama chat")),
		}
	}
	(text, images)
}

fn sanitize_schema(value: Value) -> Value {
	match value {
		Value::Bool(true) => json!({"anyOf": OPEN_SCHEMA_TYPES.map(|kind| json!({"type": kind}))}),
		Value::Bool(false) => {
			json!({"not":{"anyOf": OPEN_SCHEMA_TYPES.map(|kind| json!({"type": kind}))}})
		},
		Value::Object(mut object) => {
			for key in ["additionalProperties", "unevaluatedProperties"] {
				if object.get(key).is_some_and(Value::is_boolean) {
					object.remove(key);
				}
			}
			for key in ["properties", "patternProperties", "$defs", "definitions", "dependentSchemas"]
			{
				if let Some(Value::Object(entries)) = object.get_mut(key) {
					for value in entries.values_mut() {
						sanitize_slot(value);
					}
				}
			}
			for key in ["allOf", "anyOf", "oneOf", "prefixItems"] {
				if let Some(Value::Array(entries)) = object.get_mut(key) {
					for value in entries {
						sanitize_slot(value);
					}
				}
			}
			for key in [
				"items",
				"additionalItems",
				"contains",
				"contentSchema",
				"propertyNames",
				"if",
				"then",
				"else",
				"not",
				"unevaluatedItems",
				"additionalProperties",
				"unevaluatedProperties",
			] {
				if let Some(value) = object.get_mut(key) {
					sanitize_slot(value);
				}
			}
			if object.get("type").is_some_and(Value::is_array) {
				let Value::Array(types) = object.remove("type").expect("type was checked as an array")
				else {
					unreachable!("type was checked as an array");
				};
				let mut unique = Vec::new();
				for kind in types
					.into_iter()
					.filter_map(|value| value.as_str().map(str::to_owned))
				{
					if !unique.contains(&kind) {
						unique.push(kind);
					}
				}
				let non_null: Vec<_> = unique
					.iter()
					.filter(|kind| kind.as_str() != "null")
					.cloned()
					.collect();
				if non_null.len() <= 1 {
					if let Some(kind) = non_null.first().or_else(|| unique.first()) {
						object.insert("type".into(), Value::String(kind.clone()));
					}
				} else {
					let union = json!({
						"anyOf": unique
							.into_iter()
							.map(|kind| json!({"type":kind}))
							.collect::<Vec<_>>()
					});
					let mut all_of = vec![union];
					if let Some(Value::Array(existing)) = object.remove("allOf") {
						all_of.extend(existing);
					}
					object.insert("allOf".into(), Value::Array(all_of));
				}
			}
			Value::Object(object)
		},
		other => other,
	}
}

fn sanitize_slot(value: &mut Value) {
	let owned = std::mem::take(value);
	*value = sanitize_schema(owned);
}

#[derive(Default)]
struct OllamaDecodeState {
	next_index: u32,
	open:       Option<(u32, StreamPartKind)>,
	parts:      BTreeMap<u32, DecodedPart>,
	usage:      Option<(u64, u64)>,
	completed:  bool,
}

enum DecodedPart {
	Text(String),
	Thinking(String),
	ToolCall { id: CallId, name: Str, args: Bytes },
}

fn decode_line(data: &[u8], state: &mut DecodeState) -> Result<SmallVec<TurnEvent, 2>, Error> {
	if data.is_empty() {
		return Ok(SmallVec::new());
	}
	let chunk: Value = serde_json::from_slice(data).map_err(|error| {
		Error::Provider(Str::from(format!("invalid Ollama NDJSON chunk: {error}")))
	})?;
	let state = state.get_or_insert_with(OllamaDecodeState::default);
	if state.completed {
		return Ok(SmallVec::new());
	}
	let mut events = SmallVec::new();
	if let Some(detail) = chunk.get("error").and_then(Value::as_str) {
		close_open(state, &mut events);
		state.completed = true;
		events.push(TurnEvent::Error(turn_error(TurnErrorKind::Upstream, detail)));
		return Ok(events);
	}
	if let Some(message) = chunk.get("message").and_then(Value::as_object) {
		if let Some(thinking) = message
			.get("thinking")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
		{
			append_part(state, StreamPartKind::Thinking, thinking, &mut events)?;
		}
		if let Some(content) = message
			.get("content")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
		{
			append_part(state, StreamPartKind::Text, content, &mut events)?;
		}
		if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
			for call in calls {
				close_open(state, &mut events);
				let function = call
					.get("function")
					.and_then(Value::as_object)
					.ok_or_else(|| {
						Error::Provider(Str::new_static("Ollama tool call is missing function"))
					})?;
				let name = function
					.get("name")
					.and_then(Value::as_str)
					.filter(|name| !name.is_empty())
					.ok_or_else(|| {
						Error::Provider(Str::new_static("Ollama tool call is missing name"))
					})?;
				let args = match function.get("arguments") {
					Some(Value::String(arguments)) => Bytes::copy_from_slice(arguments.as_bytes()),
					Some(arguments) => {
						serde_json::to_vec(arguments)
							.map(Bytes::from)
							.map_err(|error| {
								Error::Provider(Str::from(format!(
									"invalid Ollama tool arguments: {error}"
								)))
							})?
					},
					None => Bytes::from_static(b"{}"),
				};
				let index = state.next_index;
				state.next_index += 1;
				let id = CallId::new();
				state.parts.insert(index, DecodedPart::ToolCall {
					id,
					name: Str::new(name),
					args: args.clone(),
				});
				events.push(TurnEvent::PartStart {
					index,
					kind: StreamPartKind::ToolCall,
					tool_call_id: Str::from(id.to_string()),
					tool_name: Str::new(name),
				});
				events.push(TurnEvent::PartDelta { index, chunk: args });
				events.push(TurnEvent::PartEnd { index, signature: Bytes::new() });
			}
		}
	}
	if chunk.get("done").and_then(Value::as_bool) == Some(true) {
		close_open(state, &mut events);
		state.completed = true;
		let reason = chunk.get("done_reason").and_then(Value::as_str);
		if reason == Some("load") {
			events.push(TurnEvent::Error(turn_error(
				TurnErrorKind::Upstream,
				"Ollama loaded the model but generated nothing because the request had no user message",
			)));
			return Ok(events);
		}
		let input = chunk
			.get("prompt_eval_count")
			.and_then(Value::as_u64)
			.unwrap_or(0);
		let output = chunk.get("eval_count").and_then(Value::as_u64).unwrap_or(0);
		state.usage = Some((input, output));
		let mapped = if reason == Some("length") {
			StopReason::MaxTokens
		} else if reason == Some("tool_calls") {
			StopReason::ToolUse
		} else {
			StopReason::EndTurn
		};
		let has_tools = state
			.parts
			.values()
			.any(|part| matches!(part, DecodedPart::ToolCall { .. }));
		events.push(TurnEvent::Outcome(outcome(with_tool_use_precedence(mapped, has_tools), state)));
	}
	Ok(events)
}

fn append_part(
	state: &mut OllamaDecodeState,
	kind: StreamPartKind,
	chunk: &str,
	events: &mut SmallVec<TurnEvent, 2>,
) -> Result<(), Error> {
	let index = if let Some((index, open_kind)) = state.open {
		if open_kind == kind {
			index
		} else {
			close_open(state, events);
			open_part(state, kind, events)
		}
	} else {
		open_part(state, kind, events)
	};
	match state.parts.get_mut(&index) {
		Some(DecodedPart::Text(text) | DecodedPart::Thinking(text)) => text.push_str(chunk),
		_ => return Err(Error::Provider(Str::new_static("Ollama stream part state mismatch"))),
	}
	events.push(TurnEvent::PartDelta { index, chunk: Bytes::copy_from_slice(chunk.as_bytes()) });
	Ok(())
}

fn open_part(
	state: &mut OllamaDecodeState,
	kind: StreamPartKind,
	events: &mut SmallVec<TurnEvent, 2>,
) -> u32 {
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

fn close_open(state: &mut OllamaDecodeState, events: &mut SmallVec<TurnEvent, 2>) {
	if let Some((index, _)) = state.open.take() {
		events.push(TurnEvent::PartEnd { index, signature: Bytes::new() });
	}
}

fn finish_incomplete(state: &mut DecodeState) -> SmallVec<TurnEvent, 2> {
	let state = state.get_or_insert_with(OllamaDecodeState::default);
	if state.completed {
		return SmallVec::new();
	}
	state.completed = true;
	let mut events = SmallVec::new();
	close_open(state, &mut events);
	events.push(TurnEvent::Error(turn_error(
		TurnErrorKind::Upstream,
		"Ollama NDJSON stream ended before a done terminal chunk",
	)));
	events
}

fn outcome(stop: StopReason, state: &OllamaDecodeState) -> ChatOutcome {
	let mut output = Vec::with_capacity(state.parts.len());
	for part in state.parts.values() {
		match part {
			DecodedPart::Text(text) if !text.is_empty() => {
				output.push(assistant_item(Part::Text(Str::from(text.as_str()))));
			},
			DecodedPart::Thinking(text) if !text.is_empty() => {
				output.push(assistant_item(Part::Thinking(
					Thinking::builder()
						.text(Str::from(text.as_str()))
						.signature(Bytes::new())
						.redacted(false)
						.build(),
				)));
			},
			DecodedPart::ToolCall { id, name, args } => output.push(
				Item::builder()
					.seq(0)
					.kind(ItemKind::ToolCall(
						ToolCall::builder()
							.id(*id)
							.name(name.clone())
							.args_json(args.clone())
							.thought_signature(Bytes::new())
							.build(),
					))
					.props(Props::default())
					.build(),
			),
			DecodedPart::Text(_) | DecodedPart::Thinking(_) => {},
		}
	}
	ChatOutcome::builder()
		.output(output)
		.stop(stop)
		.maybe_usage(state.usage.map(|(input, output)| {
			Usage::builder()
				.input_tokens(input)
				.output_tokens(output)
				.cache_read_tokens(0)
				.cache_write_tokens(0)
				.total_tokens(input.saturating_add(output))
				.accuracy(Accuracy::Exact)
				.detail(Props::default())
				.build()
		}))
		.maybe_cost(None)
		.unsupported(Vec::new())
		.maybe_revision(None)
		.provider(Str::new_static("ollama-cloud"))
		.model(Str::default())
		.props(Props::default())
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

fn turn_error(kind: TurnErrorKind, detail: &str) -> TurnError {
	TurnError::builder()
		.kind(kind)
		.detail(Str::new(detail))
		.unsupported(Vec::new())
		.retry_after_ms(0)
		.build()
}

fn dropped(what: impl Into<Str>, detail: impl Into<Str>) -> Unsupported {
	Unsupported::builder()
		.what(what.into())
		.detail(detail.into())
		.action(UnsupportedAction::Dropped)
		.build()
}

fn report_feature(
	unsupported: &mut Vec<Unsupported>,
	what: &'static str,
	detail: &'static str,
	fallback: Fallback,
) -> Result<(), Error> {
	let entry = dropped(what, detail);
	if fallback == Fallback::Error {
		return Err(Error::Unsupported(vec![entry]));
	}
	unsupported.push(entry);
	Ok(())
}
