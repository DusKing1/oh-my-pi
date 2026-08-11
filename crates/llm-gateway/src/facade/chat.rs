//! `OpenAI` Chat Completions facade over the canonical
//! [`Chat`](omp_llm_types::facet::Chat) facet.

use std::{collections::BTreeMap, fmt::Display, sync::Arc};

use async_stream::stream;
use bytes::Bytes;
use futures::StreamExt;
use http::{Request, StatusCode};
use hyper::body::Body;
use omp_core::Str;
use omp_llm_types::{
	CacheHint, ChatOutcome, ChatRequest, Effort, Fallback, Feature, Item, ItemKind, JsonSchema,
	Message, Part, Props, Reasoning, ResponseFormat, ResponseFormatKind, Role, Sampling, StopReason,
	StreamPartKind, Thread, ToolCall, ToolChoice, ToolDef, ToolResult, TurnEvent, Usage,
};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
	FacadeError, FacadeResponse, FacadeState, Vendor, canonical_call_id, error_response,
	json_response, provider_options, read_json, request_meta, sse_data, sse_response,
};

#[derive(Default, Deserialize)]
struct WireRequest {
	model:                 Str,
	#[serde(default)]
	messages:              Vec<Value>,
	#[serde(default)]
	tools:                 Vec<Value>,
	tool_choice:           Option<Value>,
	temperature:           Option<f64>,
	top_p:                 Option<f64>,
	max_tokens:            Option<u64>,
	max_completion_tokens: Option<u64>,
	frequency_penalty:     Option<f64>,
	presence_penalty:      Option<f64>,
	stop:                  Option<Value>,
	reasoning_effort:      Option<Value>,
	user:                  Option<Str>,
	prompt_cache_key:      Option<Str>,
	response_format:       Option<Value>,
	#[serde(default)]
	stream:                bool,
	#[serde(default)]
	stream_options:        StreamOptions,
	#[serde(flatten)]
	extra:                 BTreeMap<Str, Value>,
}

#[derive(Default, Deserialize)]
struct StreamOptions {
	#[serde(default)]
	include_usage: bool,
	#[serde(flatten)]
	extra:         BTreeMap<Str, Value>,
}

/// Handles `POST /v1/chat/completions` against the shared chat facet.
pub(crate) async fn handle<B>(request: Request<B>, state: Arc<FacadeState>) -> FacadeResponse
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
{
	let wire: WireRequest = match read_json(request, Vendor::OpenAi).await {
		Ok(value) => value,
		Err(response) => return *response,
	};
	let request = match canonical_request(&wire) {
		Ok(value) => value,
		Err(error) => return error_response(Vendor::OpenAi, error),
	};
	let Some(chat) = state.facets.chat.clone() else {
		return error_response(
			Vendor::OpenAi,
			FacadeError::Invalid(Str::from("chat facet is unavailable")),
		);
	};
	let events = match chat.turn(request, None).await {
		Ok(events) => events,
		Err(error) => return error_response(Vendor::OpenAi, FacadeError::Facet(error)),
	};
	if wire.stream {
		streaming(events, wire.model, wire.stream_options.include_usage)
	} else {
		non_streaming(events, wire.model).await
	}
}

fn canonical_request(wire: &WireRequest) -> Result<ChatRequest, FacadeError> {
	let mut items = Vec::new();
	for message in &wire.messages {
		append_message(message, &mut items)?;
	}
	let mut tools = Vec::with_capacity(wire.tools.len());
	for tool in &wire.tools {
		let function = tool
			.get("function")
			.ok_or_else(|| invalid("tool.function is required"))?;
		let name = string_at(function, "name")?;
		let description = function
			.get("description")
			.and_then(Value::as_str)
			.unwrap_or_default();
		let schema = function
			.get("parameters")
			.cloned()
			.unwrap_or_else(|| json!({}));
		tools.push(
			ToolDef::builder()
				.name(Str::from(name))
				.description(Str::from(description))
				.schema_json(Bytes::from(serde_json::to_vec(&schema).expect("JSON values serialize")))
				.maybe_strict(function.get("strict").and_then(Value::as_bool))
				.build(),
		);
	}
	let mut options = provider_options("openai", &wire.extra);
	for (name, value) in &wire.stream_options.extra {
		options.insert_ns("openai", &format!("stream_options.{name}"), value.clone());
	}
	let format = wire
		.response_format
		.as_ref()
		.map(response_format)
		.transpose()?
		.flatten();
	if let Some(wire_format) = &wire.response_format {
		if format.is_none() {
			options.insert_ns("openai", "response_format", wire_format.clone());
		} else if let Some(residual) = response_format_residual(wire_format) {
			options.insert_ns("openai", "response_format", residual);
		}
	}
	Ok(ChatRequest::builder()
		.model(wire.model.clone())
		.thread(Thread::builder().items(items).build())
		.tools(tools)
		.maybe_tool_choice(tool_choice(wire.tool_choice.as_ref())?)
		.maybe_sampling(sampling(wire)?)
		.maybe_thinking(thinking(wire.reasoning_effort.as_ref())?)
		.maybe_response_format(format)
		.maybe_meta(request_meta(wire.user.as_ref()))
		.maybe_cache(
			wire
				.prompt_cache_key
				.as_ref()
				.map(|key| CacheHint::builder().session_key(key.clone()).build()),
		)
		.maybe_provider_options((!options.is_empty()).then_some(options))
		.build())
}

pub(crate) fn tool_choice(
	value: Option<&Value>,
) -> Result<Option<Feature<ToolChoice>>, FacadeError> {
	let Some(value) = value else {
		return Ok(None);
	};
	let choice = match value {
		Value::String(value) => match value.as_str() {
			"auto" => ToolChoice::Auto,
			"none" => ToolChoice::None,
			"required" | "any" => ToolChoice::Required,
			_ => return Err(invalid("unsupported tool_choice")),
		},
		Value::Object(value) => {
			let name = value
				.get("function")
				.and_then(|function| function.get("name"))
				.or_else(|| value.get("name"))
				.and_then(Value::as_str)
				.ok_or_else(|| invalid("tool_choice name is required"))?;
			ToolChoice::Named(Str::from(name))
		},
		_ => return Err(invalid("tool_choice must be a string or object")),
	};
	Ok(Some(
		Feature::builder()
			.value(choice)
			.on_unsupported(Fallback::Error)
			.build(),
	))
}

fn sampling(wire: &WireRequest) -> Result<Option<Sampling>, FacadeError> {
	let stop = match wire.stop.as_ref() {
		None | Some(Value::Null) => None,
		Some(Value::String(value)) => Some(vec![Str::from(value.as_str())]),
		Some(Value::Array(values)) => {
			if values.len() > 4 {
				return Err(invalid("stop accepts at most four strings"));
			}
			Some(
				values
					.iter()
					.map(|value| {
						value
							.as_str()
							.map(Str::from)
							.ok_or_else(|| invalid("stop entries must be strings"))
					})
					.collect::<Result<Vec<_>, _>>()?,
			)
		},
		Some(_) => return Err(invalid("stop must be a string or string array")),
	};
	let max_output_tokens = wire.max_completion_tokens.or(wire.max_tokens);
	if wire.temperature.is_none()
		&& wire.top_p.is_none()
		&& wire.frequency_penalty.is_none()
		&& wire.presence_penalty.is_none()
		&& stop.is_none()
		&& max_output_tokens.is_none()
	{
		return Ok(None);
	}
	Ok(Some(
		Sampling::builder()
			.maybe_temperature(wire.temperature)
			.maybe_top_p(wire.top_p)
			.maybe_frequency_penalty(wire.frequency_penalty)
			.maybe_presence_penalty(wire.presence_penalty)
			.maybe_stop(stop)
			.maybe_max_output_tokens(max_output_tokens)
			.build(),
	))
}

pub(crate) fn thinking(value: Option<&Value>) -> Result<Option<Feature<Reasoning>>, FacadeError> {
	let Some(value) = value else {
		return Ok(None);
	};
	let effort = match value.as_str() {
		Some("minimal") => Effort::Minimal,
		Some("low") => Effort::Low,
		Some("medium") => Effort::Medium,
		Some("high") => Effort::High,
		Some("xhigh") => Effort::XHigh,
		Some("max") => Effort::Max,
		_ => return Err(invalid("unsupported reasoning effort")),
	};
	Ok(Some(
		Feature::builder()
			.value(Reasoning::builder().effort(effort).build())
			.on_unsupported(Fallback::Error)
			.build(),
	))
}
pub(crate) fn response_format(
	value: &Value,
) -> Result<Option<Feature<ResponseFormat>>, FacadeError> {
	if value.get("type").and_then(Value::as_str) != Some("json_schema") {
		return Ok(None);
	}
	let definition = value
		.get("json_schema")
		.ok_or_else(|| invalid("response_format.json_schema is required"))?;
	let name = string_at(definition, "name")?;
	let schema = definition
		.get("schema")
		.cloned()
		.unwrap_or_else(|| json!({}));
	Ok(Some(
		Feature::builder()
			.value(
				ResponseFormat::builder()
					.kind(ResponseFormatKind::JsonSchema(
						JsonSchema::builder()
							.name(Str::from(name))
							.schema_json(Bytes::from(
								serde_json::to_vec(&schema).expect("JSON values serialize"),
							))
							.maybe_strict(definition.get("strict").and_then(Value::as_bool))
							.build(),
					))
					.build(),
			)
			.on_unsupported(Fallback::Error)
			.build(),
	))
}

fn response_format_residual(value: &Value) -> Option<Value> {
	let mut residual = value.as_object()?.clone();
	let definition = residual.remove("json_schema");
	residual.remove("type");
	if let Some(Value::Object(mut definition)) = definition {
		definition.remove("name");
		definition.remove("schema");
		definition.remove("strict");
		if !definition.is_empty() {
			residual.insert("json_schema".to_owned(), Value::Object(definition));
		}
	}
	(!residual.is_empty()).then_some(Value::Object(residual))
}

pub(crate) fn append_message(value: &Value, items: &mut Vec<Item>) -> Result<(), FacadeError> {
	let role = string_at(value, "role")?;
	if role == "tool" {
		let call = value
			.get("tool_call_id")
			.and_then(Value::as_str)
			.ok_or_else(|| invalid("tool_call_id is required"))?;
		let call_id = canonical_call_id(call);
		let name = items
			.iter()
			.rev()
			.find_map(|item| match &item.kind {
				ItemKind::ToolCall(tool_call) if tool_call.id == call_id => {
					Some(tool_call.name.clone())
				},
				_ => None,
			})
			.unwrap_or_default();
		let parts = content_parts(value.get("content"));
		items.push(item(ItemKind::ToolResult(
			ToolResult::builder()
				.call_id(call_id)
				.name(name)
				.parts(parts)
				.is_error(false)
				.build(),
		)));
		return Ok(());
	}
	let canonical_role = match role {
		"system" | "developer" => Role::System,
		"user" => Role::User,
		"assistant" => Role::Assistant,
		_ => return Err(invalid("unsupported message role")),
	};
	let parts = content_parts(value.get("content"));
	if !parts.is_empty() {
		items.push(item(ItemKind::Message(
			Message::builder().role(canonical_role).parts(parts).build(),
		)));
	}
	if let Some(calls) = value.get("tool_calls").and_then(Value::as_array) {
		for call in calls {
			let wire_id = string_at(call, "id")?;
			let function = call
				.get("function")
				.ok_or_else(|| invalid("tool_call.function is required"))?;
			let arguments = string_at(function, "arguments")?;
			items.push(item(ItemKind::ToolCall(
				ToolCall::builder()
					.id(canonical_call_id(wire_id))
					.name(Str::from(string_at(function, "name")?))
					.args_json(Bytes::copy_from_slice(arguments.as_bytes()))
					.thought_signature(Bytes::new())
					.build(),
			)));
		}
	}
	Ok(())
}

pub(crate) fn item(kind: ItemKind) -> Item {
	Item::builder()
		.seq(0)
		.kind(kind)
		.props(Props::default())
		.build()
}

pub(crate) fn content_parts(value: Option<&Value>) -> Vec<Part> {
	match value {
		Some(Value::String(text)) => vec![Part::Text(Str::from(text.as_str()))],
		Some(Value::Array(parts)) => parts
			.iter()
			.filter_map(|part| {
				part
					.as_str()
					.or_else(|| part.get("text").and_then(Value::as_str))
					.map(|text| Part::Text(Str::from(text)))
			})
			.collect(),
		_ => Vec::new(),
	}
}

pub(crate) fn string_at<'a>(value: &'a Value, key: &str) -> Result<&'a str, FacadeError> {
	value
		.get(key)
		.and_then(Value::as_str)
		.ok_or_else(|| invalid(&format!("{key} is required")))
}

fn invalid(detail: &str) -> FacadeError {
	FacadeError::Invalid(Str::from(detail))
}

async fn non_streaming(
	mut events: futures::stream::BoxStream<'static, TurnEvent>,
	requested_model: Str,
) -> FacadeResponse {
	let mut outcome = None;
	while let Some(event) = events.next().await {
		match event {
			TurnEvent::Outcome(value) => outcome = Some(value),
			TurnEvent::Error(error) => {
				return error_response(Vendor::OpenAi, FacadeError::Turn(error));
			},
			_ => {},
		}
	}
	let Some(outcome) = outcome else {
		return error_response(Vendor::OpenAi, invalid("turn ended without an outcome"));
	};
	let message = openai_message(&outcome);
	let body = json!({
		"id": response_id(&outcome), "object":"chat.completion", "created":0,
		"model": requested_model,
		"choices":[{"index":0,"message":message,"finish_reason":finish_reason(outcome.stop)}],
		"usage": usage_json(outcome.usage.as_ref()),
	});
	json_response(StatusCode::OK, &body)
}

fn streaming(
	mut events: futures::stream::BoxStream<'static, TurnEvent>,
	model: Str,
	include_usage: bool,
) -> FacadeResponse {
	let output = stream! {
		let id = format!("chatcmpl-{}", ulid::Ulid::generate());
		let mut kinds = BTreeMap::new();
		let mut outcome_usage = None;
		let mut failed = false;
		while let Some(event) = events.next().await {
			let payload = match event {
				TurnEvent::PartStart { index, kind, tool_call_id, tool_name } => {
					kinds.insert(index, kind);
					match kind {
						StreamPartKind::Text => Some(json!({"id":id,"object":"chat.completion.chunk","created":0,"model":model,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":Value::Null}]})),
						StreamPartKind::ToolCall => Some(json!({"id":id,"object":"chat.completion.chunk","created":0,"model":model,"choices":[{"index":0,"delta":{"tool_calls":[{"index":index,"id":tool_call_id,"type":"function","function":{"name":tool_name,"arguments":""}}]},"finish_reason":Value::Null}]})),
						_ => None,
					}
				}
				TurnEvent::PartDelta { index, chunk } => match kinds.get(&index) {
					Some(StreamPartKind::Text) => Some(json!({"id":id,"object":"chat.completion.chunk","created":0,"model":model,"choices":[{"index":0,"delta":{"content":String::from_utf8_lossy(&chunk)},"finish_reason":Value::Null}]})),
					Some(StreamPartKind::ToolCall) => Some(json!({"id":id,"object":"chat.completion.chunk","created":0,"model":model,"choices":[{"index":0,"delta":{"tool_calls":[{"index":index,"function":{"arguments":String::from_utf8_lossy(&chunk)}}]},"finish_reason":Value::Null}]})),
					_ => None,
				},
				TurnEvent::Outcome(outcome) => { outcome_usage = outcome.usage; Some(json!({"id":id,"object":"chat.completion.chunk","created":0,"model":model,"choices":[{"index":0,"delta":{},"finish_reason":finish_reason(outcome.stop)}]})) },
				TurnEvent::Error(error) => { failed = true; yield sse_data(&error_value(Vendor::OpenAi, &error.detail)); break; }
				_ => None,
			};
			if let Some(payload) = payload { yield sse_data(&payload); }
		}
		if !failed {
			if include_usage { yield sse_data(&json!({"id":id,"object":"chat.completion.chunk","created":0,"model":model,"choices":[],"usage":usage_json(outcome_usage.as_ref())})); }
			yield Bytes::from_static(b"data: [DONE]\n\n");
		}
	};
	sse_response(output)
}

pub(crate) const fn finish_reason(reason: StopReason) -> &'static str {
	match reason {
		StopReason::ToolUse => "tool_calls",
		StopReason::MaxTokens => "length",
		StopReason::ContentFilter => "content_filter",
		_ => "stop",
	}
}

pub(crate) fn usage_json(usage: Option<&Usage>) -> Value {
	usage.map_or(Value::Null, |usage| {
		let prompt_details = json!({
			"cached_tokens": usage.cache_read_tokens,
			"audio_tokens": openai_usage_detail(usage, "prompt_tokens_details", "input_tokens_details", "audio_tokens"),
			"cache_write_tokens": usage.cache_write_tokens,
		});
		let reasoning_tokens = usage.reasoning_tokens.or_else(|| {
			openai_usage_detail(
				usage,
				"completion_tokens_details",
				"output_tokens_details",
				"reasoning_tokens",
			)
		});
		let completion_details = json!({
			"accepted_prediction_tokens": openai_usage_detail(usage, "completion_tokens_details", "output_tokens_details", "accepted_prediction_tokens"),
			"audio_tokens": openai_usage_detail(usage, "completion_tokens_details", "output_tokens_details", "audio_tokens"),
			"reasoning_tokens": reasoning_tokens,
			"rejected_prediction_tokens": openai_usage_detail(usage, "completion_tokens_details", "output_tokens_details", "rejected_prediction_tokens"),
		});
		json!({
			"prompt_tokens": usage.input_tokens,
			"completion_tokens": usage.output_tokens,
			"total_tokens": openai_total_tokens(usage),
			"prompt_tokens_details": prompt_details,
			"completion_tokens_details": completion_details,
		})
	})
}

pub(crate) fn responses_usage_json(usage: Option<&Usage>) -> Value {
	usage.map_or(Value::Null, |usage| {
		let reasoning_tokens = usage.reasoning_tokens.or_else(|| {
			openai_usage_detail(
				usage,
				"completion_tokens_details",
				"output_tokens_details",
				"reasoning_tokens",
			)
		});
		json!({
			"input_tokens": usage.input_tokens,
			"output_tokens": usage.output_tokens,
			"total_tokens": openai_total_tokens(usage),
			"input_tokens_details": {
				"cached_tokens": usage.cache_read_tokens,
				"cache_write_tokens": usage.cache_write_tokens,
			},
			"output_tokens_details": {
				"reasoning_tokens": reasoning_tokens.unwrap_or(0),
			},
		})
	})
}
fn openai_total_tokens(usage: &Usage) -> u64 {
	usage
		.total_tokens
		.or_else(|| {
			usage
				.detail
				.get_ns("openai", "usage")
				.and_then(|value| value.get("total_tokens"))
				.and_then(Value::as_u64)
		})
		.or_else(|| {
			usage
				.detail
				.get_ns("openai", "total_tokens")
				.and_then(Value::as_u64)
		})
		.unwrap_or_else(|| usage.input_tokens.saturating_add(usage.output_tokens))
}

fn openai_usage_detail(
	usage: &Usage,
	chat_bucket: &str,
	responses_bucket: &str,
	key: &str,
) -> Option<u64> {
	usage
		.detail
		.get_ns("openai", "usage")
		.and_then(|value| value.get(chat_bucket))
		.or_else(|| usage.detail.get_ns("openai", responses_bucket))
		.and_then(|value| value.get(key))
		.and_then(Value::as_u64)
}

fn openai_message(outcome: &ChatOutcome) -> Value {
	let mut text = String::new();
	let mut calls = Vec::new();
	for item in &outcome.output {
		match &item.kind {
			ItemKind::Message(message) if message.role == Role::Assistant => for part in &message.parts { if let Part::Text(value) = part { text.push_str(value); } },
			ItemKind::ToolCall(call) => calls.push(json!({"id":call.id.to_string(),"type":"function","function":{"name":call.name,"arguments":String::from_utf8_lossy(&call.args_json)}})),
			_ => {}
		}
	}
	let mut message = json!({"role":"assistant","content":text});
	if !calls.is_empty() {
		message["tool_calls"] = Value::Array(calls);
	}
	message
}

fn response_id(outcome: &ChatOutcome) -> String {
	outcome
		.props
		.get_ns("openai", "response_id")
		.and_then(Value::as_str)
		.map_or_else(|| format!("chatcmpl-{}", ulid::Ulid::generate()), ToOwned::to_owned)
}

pub(crate) fn error_value(vendor: Vendor, detail: &str) -> Value {
	match vendor {
		Vendor::Anthropic => json!({"type":"error","error":{"type":"api_error","message":detail}}),
		_ => {
			json!({"error":{"message":detail,"type":"api_error","param":Value::Null,"code":Value::Null}})
		},
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn tool_argument_bytes_survive_request_projection() {
		let arguments = r#"{ "z": [3, 2], "a": "x&y" }"#;
		let wire = WireRequest {
			model: Str::from("test"),
			messages: vec![json!({
				"role":"assistant",
				"tool_calls":[{"id":"call_vendor","type":"function","function":{"name":"lookup","arguments":arguments}}]
			})],
			tools: Vec::new(),
			stream: false,
			stream_options: StreamOptions::default(),
			..WireRequest::default()
		};
		let request = canonical_request(&wire).expect("valid chat request");
		let call = request
			.thread
			.items
			.iter()
			.find_map(|item| match &item.kind {
				ItemKind::ToolCall(call) => Some(call),
				_ => None,
			})
			.expect("tool call projected");
		assert_eq!(call.args_json.as_ref(), arguments.as_bytes());
	}
}
