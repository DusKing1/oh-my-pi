//! `OpenAI` Responses facade over the canonical chat facet.
//!
//! `previous_response_id` is treated as gateway context affinity (a cache
//! session key), not forwarded to an upstream provider. The gateway remains the
//! sole owner of provider-side chaining and may route a later turn elsewhere.

use std::{collections::BTreeMap, fmt::Display, sync::Arc};

use async_stream::stream;
use bytes::Bytes;
use futures::StreamExt;
use http::{Request, StatusCode};
use hyper::body::Body;
use omp_core::Str;
use omp_llm_types::{
	CacheHint, ChatOutcome, ChatRequest, Fallback, Feature, Item, ItemKind, Message, Part,
	Reasoning, Role, Sampling, StopReason, StreamPartKind, Thread, ToolCall, ToolDef, ToolResult,
	TurnEvent,
};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
	FacadeError, FacadeResponse, FacadeState, Vendor, canonical_call_id,
	chat::{
		content_parts, finish_reason, item, response_format, responses_usage_json, string_at,
		thinking, tool_choice,
	},
	error_response, json_response, provider_options, read_json, request_meta, sse_named,
	sse_response,
};

#[derive(Default, Deserialize)]
struct WireRequest {
	model:                Str,
	#[serde(default)]
	input:                Value,
	instructions:         Option<Str>,
	#[serde(default)]
	tools:                Vec<Value>,
	tool_choice:          Option<Value>,
	max_output_tokens:    Option<u64>,
	temperature:          Option<f64>,
	top_p:                Option<f64>,
	presence_penalty:     Option<f64>,
	frequency_penalty:    Option<f64>,
	stop:                 Option<Value>,
	user:                 Option<Str>,
	reasoning:            Option<Value>,
	text:                 Option<Value>,
	#[serde(default)]
	stream:               bool,
	prompt_cache_key:     Option<Str>,
	previous_response_id: Option<Str>,
	#[serde(flatten)]
	extra:                BTreeMap<Str, Value>,
}

/// Handles `POST /v1/responses` through the shared chat service.
pub(crate) async fn handle<B>(request: Request<B>, state: Arc<FacadeState>) -> FacadeResponse
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
{
	let wire: WireRequest = match read_json(request, Vendor::Responses).await {
		Ok(value) => value,
		Err(response) => return *response,
	};
	let canonical = match canonical_request(&wire) {
		Ok(value) => value,
		Err(error) => return error_response(Vendor::Responses, error),
	};
	let Some(chat) = state.facets.chat.clone() else {
		return error_response(Vendor::Responses, invalid("chat facet is unavailable"));
	};
	let events = match chat.turn(canonical, None).await {
		Ok(value) => value,
		Err(error) => return error_response(Vendor::Responses, FacadeError::Facet(error)),
	};
	if wire.stream {
		streaming(events, wire.model)
	} else {
		non_streaming(events, wire.model).await
	}
}

fn canonical_request(wire: &WireRequest) -> Result<ChatRequest, FacadeError> {
	let mut items = Vec::new();
	if let Some(instructions) = &wire.instructions {
		items.push(item(ItemKind::Message(
			Message::builder()
				.role(Role::System)
				.parts(vec![Part::Text(instructions.clone())])
				.build(),
		)));
	}
	match &wire.input {
		Value::String(text) => items.push(item(ItemKind::Message(
			Message::builder()
				.role(Role::User)
				.parts(vec![Part::Text(Str::from(text.as_str()))])
				.build(),
		))),
		Value::Array(values) => {
			for value in values {
				append_item(value, &mut items)?;
			}
		},
		Value::Null => {},
		_ => return Err(invalid("input must be a string or item array")),
	}
	let mut tools = Vec::new();
	for tool in &wire.tools {
		let name = string_at(tool, "name")?;
		let schema = tool.get("parameters").cloned().unwrap_or_else(|| json!({}));
		tools.push(
			ToolDef::builder()
				.name(Str::from(name))
				.description(Str::from(
					tool
						.get("description")
						.and_then(Value::as_str)
						.unwrap_or_default(),
				))
				.schema_json(Bytes::from(serde_json::to_vec(&schema).expect("JSON values serialize")))
				.maybe_strict(tool.get("strict").and_then(Value::as_bool))
				.build(),
		);
	}
	let thread = Thread::builder().items(items).build();
	let mut options = provider_options("openai", &wire.extra);
	let format = wire
		.text
		.as_ref()
		.and_then(|text| text.get("format"))
		.map(responses_format)
		.transpose()?
		.flatten();
	if let Some(text) = &wire.text
		&& !text.is_object()
	{
		options.insert_ns("openai", "text", text.clone());
	}
	if let Some(text) = residual_object(wire.text.as_ref(), &["format"]) {
		options.insert_ns("openai", "text", text);
	}
	if let Some(wire_format) = wire.text.as_ref().and_then(|text| text.get("format")) {
		if format.is_none() {
			options.insert_ns("openai", "text.format", wire_format.clone());
		} else if let Some(residual) =
			residual_object(Some(wire_format), &["type", "name", "schema", "strict"])
		{
			options.insert_ns("openai", "text.format", residual);
		}
	}
	if let Some(reasoning) = residual_object(wire.reasoning.as_ref(), &["effort", "summary"]) {
		options.insert_ns("openai", "reasoning", reasoning);
	}
	if wire.previous_response_id.is_some()
		&& let Some(prompt_cache_key) = &wire.prompt_cache_key
	{
		options.insert_ns("openai", "prompt_cache_key", Value::String(prompt_cache_key.to_string()));
	}
	let builder = ChatRequest::builder()
		.model(wire.model.clone())
		.thread(thread)
		.tools(tools)
		.maybe_tool_choice(tool_choice(wire.tool_choice.as_ref())?)
		.maybe_sampling(sampling(wire)?)
		.maybe_thinking(responses_thinking(wire.reasoning.as_ref())?)
		.maybe_response_format(format)
		.maybe_meta(request_meta(wire.user.as_ref()))
		.maybe_provider_options((!options.is_empty()).then_some(options));
	let cache_key = wire
		.previous_response_id
		.as_ref()
		.or(wire.prompt_cache_key.as_ref());
	Ok(if let Some(cache_key) = cache_key {
		builder
			.cache(CacheHint::builder().session_key(cache_key.clone()).build())
			.build()
	} else {
		builder.build()
	})
}

fn sampling(wire: &WireRequest) -> Result<Option<Sampling>, FacadeError> {
	let stop = match wire.stop.as_ref() {
		None | Some(Value::Null) => None,
		Some(Value::String(value)) => Some(vec![Str::from(value.as_str())]),
		Some(Value::Array(values)) => Some(
			values
				.iter()
				.map(|value| {
					value
						.as_str()
						.map(Str::from)
						.ok_or_else(|| invalid("stop entries must be strings"))
				})
				.collect::<Result<Vec<_>, _>>()?,
		),
		Some(_) => return Err(invalid("stop must be a string or string array")),
	};
	if wire.temperature.is_none()
		&& wire.top_p.is_none()
		&& wire.frequency_penalty.is_none()
		&& wire.presence_penalty.is_none()
		&& stop.is_none()
		&& wire.max_output_tokens.is_none()
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
			.maybe_max_output_tokens(wire.max_output_tokens)
			.build(),
	))
}

fn responses_thinking(value: Option<&Value>) -> Result<Option<Feature<Reasoning>>, FacadeError> {
	let Some(value) = value else {
		return Ok(None);
	};
	if !value.is_object() {
		return Err(invalid("reasoning must be an object"));
	}
	let hide_summary = match value.get("summary").and_then(Value::as_str) {
		Some("none") => Some(true),
		Some("auto" | "concise" | "detailed") => Some(false),
		None => None,
		Some(_) => return Err(invalid("unsupported reasoning summary")),
	};
	let mut feature = thinking(value.get("effort"))?;
	if let Some(feature) = &mut feature {
		feature.value.hide_summary = hide_summary;
	} else {
		feature = Some(
			Feature::builder()
				.value(
					Reasoning::builder()
						.maybe_hide_summary(hide_summary)
						.build(),
				)
				.on_unsupported(Fallback::Error)
				.build(),
		);
	}
	Ok(feature)
}

fn responses_format(
	value: &Value,
) -> Result<Option<Feature<omp_llm_types::ResponseFormat>>, FacadeError> {
	if value.get("type").and_then(Value::as_str) != Some("json_schema") {
		return Ok(None);
	}
	response_format(&json!({"type":"json_schema","json_schema":value}))
}

fn residual_object(value: Option<&Value>, known: &[&str]) -> Option<Value> {
	let mut object = value?.as_object()?.clone();
	for field in known {
		object.remove(*field);
	}
	(!object.is_empty()).then_some(Value::Object(object))
}

fn append_item(value: &Value, items: &mut Vec<Item>) -> Result<(), FacadeError> {
	match value
		.get("type")
		.and_then(Value::as_str)
		.unwrap_or("message")
	{
		"message" => {
			let role = match value.get("role").and_then(Value::as_str).unwrap_or("user") {
				"assistant" => Role::Assistant,
				"system" | "developer" => Role::System,
				_ => Role::User,
			};
			items.push(item(ItemKind::Message(
				Message::builder()
					.role(role)
					.parts(content_parts(value.get("content")))
					.build(),
			)));
		},
		"function_call" => {
			let id = value
				.get("call_id")
				.or_else(|| value.get("id"))
				.and_then(Value::as_str)
				.ok_or_else(|| invalid("function_call.call_id is required"))?;
			let arguments = string_at(value, "arguments")?;
			items.push(item(ItemKind::ToolCall(
				ToolCall::builder()
					.id(canonical_call_id(id))
					.name(Str::from(string_at(value, "name")?))
					.args_json(Bytes::copy_from_slice(arguments.as_bytes()))
					.thought_signature(Bytes::new())
					.build(),
			)));
		},
		"function_call_output" => {
			let id = string_at(value, "call_id")?;
			let call_id = canonical_call_id(id);
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
			items.push(item(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(call_id)
					.name(name)
					.parts(content_parts(value.get("output")))
					.is_error(false)
					.build(),
			)));
		},
		_ => return Err(invalid("unsupported Responses input item")),
	}
	Ok(())
}

async fn non_streaming(
	mut events: futures::stream::BoxStream<'static, TurnEvent>,
	model: Str,
) -> FacadeResponse {
	let mut outcome = None;
	while let Some(event) = events.next().await {
		match event {
			TurnEvent::Outcome(value) => outcome = Some(value),
			TurnEvent::Error(error) => {
				return error_response(Vendor::Responses, FacadeError::Turn(error));
			},
			_ => {},
		}
	}
	let Some(outcome) = outcome else {
		return error_response(Vendor::Responses, invalid("turn ended without an outcome"));
	};
	let id = format!("resp_{}", ulid::Ulid::generate());
	json_response(StatusCode::OK, &response_value(&outcome, model.as_str(), &id))
}

fn streaming(
	mut events: futures::stream::BoxStream<'static, TurnEvent>,
	model: Str,
) -> FacadeResponse {
	let output = stream! {
		let id = format!("resp_{}", ulid::Ulid::generate());
		let mut kinds = BTreeMap::new();
		let mut output_indices = BTreeMap::new();
		let mut content_indices = BTreeMap::new();
		let mut text_parts = BTreeMap::<_, String>::new();
		let mut tool_args = BTreeMap::<_, String>::new();
		let mut next_output_index = 0_usize;
		let mut next_content_index = 0_usize;
		let mut text_item = None::<(usize, String)>;
		yield sse_named("response.created", &json!({"type":"response.created","response":{"id":id,"object":"response","status":"in_progress","model":model,"output":[]}}));
		while let Some(event) = events.next().await {
			match event {
				TurnEvent::PartStart { index, kind, tool_call_id, tool_name } => {
					kinds.insert(index, kind);
					match kind {
						StreamPartKind::Text => {
							let (output_index, item_id) = if let Some((output_index, item_id)) = &text_item { (*output_index, item_id.clone()) } else {
											 let output_index = next_output_index;
											 next_output_index += 1;
											 let item_id = format!("msg_{}", ulid::Ulid::generate());
											 yield sse_named("response.output_item.added", &json!({"type":"response.output_item.added","output_index":output_index,"item":{"type":"message","id":item_id,"status":"in_progress","role":"assistant","content":[]}}));
											 text_item = Some((output_index, item_id.clone()));
											 (output_index, item_id)
										 };
							let content_index = next_content_index;
							next_content_index += 1;
							output_indices.insert(index, output_index);
							content_indices.insert(index, content_index);
							text_parts.insert(content_index, String::new());
							yield sse_named("response.content_part.added", &json!({"type":"response.content_part.added","item_id":item_id,"output_index":output_index,"content_index":content_index,"part":{"type":"output_text","text":"","annotations":[]}}));
						},
						StreamPartKind::ToolCall => {
							let output_index = next_output_index;
							next_output_index += 1;
							output_indices.insert(index, output_index);
							tool_args.insert(index, String::new());
							yield sse_named("response.output_item.added", &json!({"type":"response.output_item.added","output_index":output_index,"item":{"type":"function_call","id":tool_call_id,"call_id":tool_call_id,"name":tool_name,"arguments":"","status":"in_progress"}}));
						},
						_ => {},
					}
				}
				TurnEvent::PartDelta { index, chunk } => match kinds.get(&index) {
					Some(StreamPartKind::Text) => {
						let output_index = output_indices[&index];
						let content_index = content_indices[&index];
						let delta = String::from_utf8_lossy(&chunk);
						text_parts.entry(content_index).or_default().push_str(&delta);
						let item_id = &text_item.as_ref().expect("text item exists").1;
						yield sse_named("response.output_text.delta", &json!({"type":"response.output_text.delta","item_id":item_id,"output_index":output_index,"content_index":content_index,"delta":delta}));
					},
					Some(StreamPartKind::ToolCall) => {
						let output_index = output_indices[&index];
						let delta = String::from_utf8_lossy(&chunk);
						tool_args.entry(index).or_default().push_str(&delta);
						yield sse_named("response.function_call_arguments.delta", &json!({"type":"response.function_call_arguments.delta","output_index":output_index,"delta":delta}));
					},
					_ => {}
				},
				TurnEvent::PartEnd { index, .. } => match kinds.get(&index) {
					Some(StreamPartKind::Text) => {
						let output_index = output_indices[&index];
						let content_index = content_indices[&index];
						let item_id = &text_item.as_ref().expect("text item exists").1;
						let text = &text_parts[&content_index];
						yield sse_named("response.output_text.done", &json!({"type":"response.output_text.done","item_id":item_id,"output_index":output_index,"content_index":content_index,"text":text}));
						yield sse_named("response.content_part.done", &json!({"type":"response.content_part.done","item_id":item_id,"output_index":output_index,"content_index":content_index,"part":{"type":"output_text","text":text,"annotations":[]}}));
					},
					Some(StreamPartKind::ToolCall) => {
						let output_index = output_indices[&index];
						let arguments = &tool_args[&index];
						yield sse_named("response.function_call_arguments.done", &json!({"type":"response.function_call_arguments.done","output_index":output_index,"arguments":arguments}));
					},
					_ => {}
				},
				TurnEvent::Outcome(outcome) => {
					if let Some((output_index, item_id)) = &text_item {
						let content = text_parts
							.values()
							.map(|text| json!({"type":"output_text","text":text,"annotations":[]}))
							.collect::<Vec<_>>();
						yield sse_named("response.output_item.done", &json!({"type":"response.output_item.done","output_index":output_index,"item":{"type":"message","id":item_id,"status":"completed","role":"assistant","content":content}}));
					}
					yield sse_named("response.completed", &json!({"type":"response.completed","response":response_value(&outcome, model.as_str(), &id)}));
					break;
				},
				TurnEvent::Error(error) => {
					yield sse_named("error", &json!({"type":"error","error":{"type":"api_error","message":error.detail}}));
					break;
				},
				_ => {}
			}
		}
	};
	sse_response(output)
}

fn response_value(outcome: &ChatOutcome, requested_model: &str, response_id: &str) -> Value {
	let mut output = Vec::new();
	for item in &outcome.output {
		match &item.kind {
			ItemKind::Message(message) if message.role == Role::Assistant => {
				let text: String = message.parts.iter().filter_map(|part| if let Part::Text(text) = part { Some(text.as_str()) } else { None }).collect();
				output.push(json!({"type":"message","id":format!("msg_{}", item.seq),"status":"completed","role":"assistant","content":[{"type":"output_text","text":text,"annotations":[]}]}));
			}
			ItemKind::ToolCall(call) => output.push(json!({"type":"function_call","id":call.id.to_string(),"call_id":call.id.to_string(),"name":call.name,"arguments":String::from_utf8_lossy(&call.args_json),"status":"completed"})),
			_ => {}
		}
	}
	json!({"id":response_id,"object":"response","created_at":0,"status":"completed","model":if outcome.model.is_empty(){requested_model}else{outcome.model.as_str()},"output":output,"usage":responses_usage_json(outcome.usage.as_ref()),"error":Value::Null,"incomplete_details":if outcome.stop==StopReason::MaxTokens{json!({"reason":finish_reason(outcome.stop)})}else{Value::Null}})
}

fn invalid(detail: &str) -> FacadeError {
	FacadeError::Invalid(Str::from(detail))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn previous_response_is_gateway_affinity() {
		let wire = WireRequest {
			model: Str::from("test"),
			input: Value::String("continue".into()),
			tools: Vec::new(),
			stream: false,
			previous_response_id: Some(Str::from("resp_previous")),
			..WireRequest::default()
		};
		let request = canonical_request(&wire).expect("valid Responses request");
		assert_eq!(
			request
				.cache
				.as_ref()
				.map(|cache| cache.session_key.as_str()),
			Some("resp_previous")
		);
	}
}
