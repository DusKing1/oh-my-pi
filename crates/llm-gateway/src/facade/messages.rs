//! Anthropic Messages and token-counting facades over canonical facets.

use std::{collections::BTreeMap, fmt::Display, sync::Arc};

use async_stream::stream;
use bytes::Bytes;
use futures::StreamExt;
use http::{Request, StatusCode};
use hyper::body::Body;
use omp_core::{Str, encoding::base64};
use omp_llm_types::{
	BlobPart, ChatOutcome, ChatRequest, CountInput, CountRequest, Effort, Fallback, Feature,
	ItemKind, Message, ModelFallback, Part, Props, Reasoning, ResponseFormat, Role, Sampling,
	ServerTool, ServerToolKind, StopReason, StreamPartKind, Thinking, Thread, ToolCall, ToolChoice,
	ToolDef, ToolResult, TurnEvent, Usage,
};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
	FacadeError, FacadeResponse, FacadeState, Vendor, canonical_call_id,
	chat::{item, response_format, string_at},
	error_response, json_response, provider_options, read_json, request_meta, sse_named,
	sse_response,
};

#[derive(Default, Deserialize)]
struct WireRequest {
	model:          Str,
	#[serde(default)]
	system:         Value,
	#[serde(default)]
	messages:       Vec<Value>,
	#[serde(default)]
	tools:          Vec<Value>,
	tool_choice:    Option<Value>,
	max_tokens:     Option<u64>,
	temperature:    Option<f64>,
	top_p:          Option<f64>,
	top_k:          Option<u32>,
	stop_sequences: Option<Vec<Str>>,
	thinking:       Option<Value>,
	output_config:  Option<Value>,
	metadata:       Option<Value>,
	#[serde(default)]
	stream:         bool,
	#[serde(flatten)]
	extra:          BTreeMap<Str, Value>,
}

/// Handles Anthropic message generation and exact token-count routes.
pub(crate) async fn handle<B>(request: Request<B>, state: Arc<FacadeState>) -> FacadeResponse
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
{
	let count = request.uri().path().ends_with("/count_tokens");
	let wire: WireRequest = match read_json(request, Vendor::Anthropic).await {
		Ok(value) => value,
		Err(response) => return *response,
	};
	let canonical = match canonical_input(&wire) {
		Ok(value) => value,
		Err(error) => return error_response(Vendor::Anthropic, error),
	};
	if count {
		if wire.stream
			|| canonical.tool_choice.is_some()
			|| canonical.sampling.is_some()
			|| canonical.thinking.is_some()
			|| canonical.response_format.is_some()
			|| canonical.provider_options.is_some()
			|| canonical.meta.is_some()
		{
			return error_response(
				Vendor::Anthropic,
				invalid("generation controls are unsupported by count_tokens"),
			);
		}
		let Some(counter) = state.facets.count_tokens.clone() else {
			return error_response(Vendor::Anthropic, invalid("count_tokens facet is unavailable"));
		};
		let request = CountRequest::builder()
			.model(wire.model)
			.input(CountInput::Thread(canonical.thread))
			.tools(canonical.tools)
			.build();
		return match counter.count(request).await {
			Ok(result) => json_response(StatusCode::OK, &json!({"input_tokens":result.tokens})),
			Err(error) => error_response(Vendor::Anthropic, FacadeError::Facet(error)),
		};
	}
	if wire.max_tokens.is_none() {
		return error_response(
			Vendor::Anthropic,
			invalid("max_tokens is required for message generation"),
		);
	}
	let request = ChatRequest::builder()
		.model(wire.model.clone())
		.thread(canonical.thread)
		.tools(canonical.tools)
		.maybe_tool_choice(canonical.tool_choice)
		.maybe_sampling(canonical.sampling)
		.maybe_meta(canonical.meta)
		.maybe_thinking(canonical.thinking)
		.maybe_response_format(canonical.response_format)
		.maybe_provider_options(canonical.provider_options)
		.build();
	let Some(chat) = state.facets.chat.clone() else {
		return error_response(Vendor::Anthropic, invalid("chat facet is unavailable"));
	};
	let events = match chat.turn(request, None).await {
		Ok(value) => value,
		Err(error) => return error_response(Vendor::Anthropic, FacadeError::Facet(error)),
	};
	if wire.stream {
		streaming(events, wire.model)
	} else {
		non_streaming(events, wire.model).await
	}
}

struct CanonicalInput {
	thread:           Thread,
	tools:            Vec<ToolDef>,
	tool_choice:      Option<Feature<ToolChoice>>,
	sampling:         Option<Sampling>,
	thinking:         Option<Feature<Reasoning>>,
	response_format:  Option<Feature<ResponseFormat>>,
	provider_options: Option<Props>,
	meta:             Option<omp_llm_types::RequestMeta>,
}

fn canonical_input(wire: &WireRequest) -> Result<CanonicalInput, FacadeError> {
	let mut items = Vec::new();
	let (system, system_blocks) = anthropic_parts(&wire.system)?;
	if !system.is_empty() {
		items.push(message_item(Role::System, system, system_blocks));
	}
	for message in &wire.messages {
		let role = match string_at(message, "role")? {
			"assistant" => Role::Assistant,
			"user" => Role::User,
			_ => return Err(invalid("unsupported message role")),
		};
		let content = message.get("content").unwrap_or(&Value::Null);
		let blocks = match content {
			Value::String(value) => vec![json!({"type":"text","text":value})],
			Value::Array(blocks) => blocks.clone(),
			_ => return Err(invalid("Anthropic message content must be a string or array")),
		};
		let mut parts = Vec::new();
		let mut part_blocks = Vec::new();
		for block in blocks {
			match block.get("type").and_then(Value::as_str).unwrap_or("text") {
				"tool_use" => {
					flush_message(&mut items, role, &mut parts, &mut part_blocks);
					let id = string_at(&block, "id")?;
					let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
					let mut metadata = Props::default();
					metadata.insert_ns("anthropic", "content_block", block.clone());
					items.push(item(ItemKind::ToolCall(
						ToolCall::builder()
							.id(canonical_call_id(id))
							.name(Str::from(string_at(&block, "name")?))
							.args_json(Bytes::from(
								serde_json::to_vec(&input).expect("JSON values serialize"),
							))
							.thought_signature(Bytes::new())
							.provider_metadata(metadata)
							.build(),
					)));
				},
				"tool_result" => {
					flush_message(&mut items, role, &mut parts, &mut part_blocks);
					let id = string_at(&block, "tool_use_id")?;
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
					let (result_parts, result_blocks) = match block.get("content") {
						None => (Vec::new(), Vec::new()),
						Some(Value::Null) => {
							return Err(invalid("Anthropic tool_result content cannot be null"));
						},
						Some(content) => anthropic_parts(content)?,
					};
					let mut metadata = Props::default();
					metadata.insert_ns("anthropic", "content_block", block.clone());
					metadata.insert_ns("anthropic", "content_blocks", Value::Array(result_blocks));
					items.push(item(ItemKind::ToolResult(
						ToolResult::builder()
							.call_id(call_id)
							.name(name)
							.parts(result_parts)
							.is_error(
								block
									.get("is_error")
									.and_then(Value::as_bool)
									.unwrap_or(false),
							)
							.provider_metadata(metadata)
							.build(),
					)));
				},
				_ => {
					let (mut block_parts, mut metadata) = anthropic_parts(&Value::Array(vec![block]))?;
					parts.append(&mut block_parts);
					part_blocks.append(&mut metadata);
				},
			}
		}
		flush_message(&mut items, role, &mut parts, &mut part_blocks);
	}
	let mut tools = Vec::new();
	for tool in &wire.tools {
		let schema = tool
			.get("input_schema")
			.cloned()
			.unwrap_or_else(|| json!({}));
		tools.push(
			ToolDef::builder()
				.name(Str::from(string_at(tool, "name")?))
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
	let sampling = if wire.max_tokens.is_some()
		|| wire.temperature.is_some()
		|| wire.top_p.is_some()
		|| wire.top_k.is_some()
		|| wire.stop_sequences.is_some()
	{
		Some(
			Sampling::builder()
				.maybe_temperature(wire.temperature)
				.maybe_top_p(wire.top_p)
				.maybe_top_k(wire.top_k)
				.maybe_stop(wire.stop_sequences.clone())
				.maybe_max_output_tokens(wire.max_tokens)
				.build(),
		)
	} else {
		None
	};
	let response_format = anthropic_format(wire.output_config.as_ref())?;
	let initiator = wire
		.metadata
		.as_ref()
		.and_then(|metadata| metadata.get("user_id"))
		.and_then(Value::as_str)
		.map(Str::from);
	let mut options = provider_options("anthropic", &wire.extra);
	if let Some(metadata) = residual_object(wire.metadata.as_ref(), &["user_id"]) {
		options.insert_ns("anthropic", "metadata", metadata);
	} else if wire
		.metadata
		.as_ref()
		.is_some_and(|metadata| !metadata.is_object())
	{
		options.insert_ns("anthropic", "metadata", wire.metadata.clone().expect("present"));
	}
	if let Some(tool_choice) = residual_object(wire.tool_choice.as_ref(), &["type", "name"]) {
		options.insert_ns("anthropic", "tool_choice", tool_choice);
	}
	if let Some(thinking) =
		residual_object(wire.thinking.as_ref(), &["type", "budget_tokens", "display"])
	{
		options.insert_ns("anthropic", "thinking", thinking);
	}
	if let Some(output_config) = residual_object(wire.output_config.as_ref(), &["effort", "format"])
	{
		options.insert_ns("anthropic", "output_config", output_config);
	}
	if let Some(output_config) = &wire.output_config
		&& !output_config.is_object()
	{
		options.insert_ns("anthropic", "output_config", output_config.clone());
	}
	if let Some(format) = wire
		.output_config
		.as_ref()
		.and_then(|value| value.get("format"))
	{
		if response_format.is_none() {
			options.insert_ns("anthropic", "output_config.format", format.clone());
		} else if let Some(residual) =
			residual_object(Some(format), &["type", "name", "schema", "strict"])
		{
			options.insert_ns("anthropic", "output_config.format", residual);
		}
	}
	Ok(CanonicalInput {
		thread: Thread::builder().items(items).build(),
		tools,
		tool_choice: anthropic_tool_choice(wire.tool_choice.as_ref())?,
		sampling,
		thinking: anthropic_thinking(wire)?,
		meta: request_meta(initiator.as_ref()),
		response_format,
		provider_options: (!options.is_empty()).then_some(options),
	})
}
fn anthropic_tool_choice(
	value: Option<&Value>,
) -> Result<Option<Feature<ToolChoice>>, FacadeError> {
	let Some(value) = value else {
		return Ok(None);
	};
	let choice = match value.get("type").and_then(Value::as_str) {
		Some("auto") => ToolChoice::Auto,
		Some("none") => ToolChoice::None,
		Some("any") => ToolChoice::Required,
		Some("tool") => ToolChoice::Named(Str::from(string_at(value, "name")?)),
		_ => return Err(invalid("unsupported Anthropic tool_choice")),
	};
	Ok(Some(
		Feature::builder()
			.value(choice)
			.on_unsupported(Fallback::Error)
			.build(),
	))
}

fn anthropic_thinking(wire: &WireRequest) -> Result<Option<Feature<Reasoning>>, FacadeError> {
	let mut effort = wire
		.output_config
		.as_ref()
		.and_then(|value| value.get("effort"))
		.and_then(Value::as_str)
		.map(|value| match value {
			"low" => Ok(Effort::Low),
			"medium" => Ok(Effort::Medium),
			"high" => Ok(Effort::High),
			"xhigh" => Ok(Effort::XHigh),
			"max" => Ok(Effort::Max),
			_ => Err(invalid("unsupported Anthropic output effort")),
		})
		.transpose()?;
	let mut budget_tokens = None;
	let mut hide_summary = None;
	if let Some(thinking) = &wire.thinking {
		match thinking.get("type").and_then(Value::as_str) {
			Some("enabled") => {
				budget_tokens = Some(
					thinking
						.get("budget_tokens")
						.and_then(Value::as_u64)
						.ok_or_else(|| invalid("thinking.budget_tokens is required"))?,
				);
			},
			Some("disabled") => effort = Some(Effort::Off),
			Some("adaptive") => {
				budget_tokens = thinking.get("budget_tokens").and_then(Value::as_u64);
			},
			_ => return Err(invalid("unsupported Anthropic thinking type")),
		}
		hide_summary = match thinking.get("display").and_then(Value::as_str) {
			Some("omitted") => Some(true),
			Some("summarized") => Some(false),
			None => None,
			Some(_) => return Err(invalid("unsupported Anthropic thinking display")),
		};
	}
	if wire.thinking.is_none()
		&& effort.is_none()
		&& budget_tokens.is_none()
		&& hide_summary.is_none()
	{
		return Ok(None);
	}
	Ok(Some(
		Feature::builder()
			.value(
				Reasoning::builder()
					.maybe_effort(effort)
					.maybe_budget_tokens(budget_tokens)
					.maybe_hide_summary(hide_summary)
					.build(),
			)
			.on_unsupported(Fallback::Error)
			.build(),
	))
}

fn anthropic_format(
	output_config: Option<&Value>,
) -> Result<Option<Feature<ResponseFormat>>, FacadeError> {
	let Some(format) = output_config.and_then(|value| value.get("format")) else {
		return Ok(None);
	};
	if format.get("type").and_then(Value::as_str) != Some("json_schema") {
		return Ok(None);
	}
	response_format(&json!({"type":"json_schema","json_schema":format}))
}

fn residual_object(value: Option<&Value>, known: &[&str]) -> Option<Value> {
	let mut object = value?.as_object()?.clone();
	for field in known {
		object.remove(*field);
	}
	(!object.is_empty()).then_some(Value::Object(object))
}

fn message_item(role: Role, parts: Vec<Part>, blocks: Vec<Value>) -> omp_llm_types::Item {
	let mut value = item(ItemKind::Message(Message::builder().role(role).parts(parts).build()));
	let mut image_sources = Vec::new();
	let mut document_sources = Vec::new();
	for block in &blocks {
		if let Some(source) = block.get("source") {
			match block.get("type").and_then(Value::as_str) {
				Some("image") => image_sources.push(source.clone()),
				Some("document") => document_sources.push(source.clone()),
				_ => {},
			}
		}
	}
	value
		.props
		.insert_ns("anthropic", "content_blocks", Value::Array(blocks));
	if !image_sources.is_empty() {
		value
			.props
			.insert_ns("anthropic", "image_sources", Value::Array(image_sources));
	}
	if !document_sources.is_empty() {
		value
			.props
			.insert_ns("anthropic", "document_sources", Value::Array(document_sources));
	}
	value
}

fn flush_message(
	items: &mut Vec<omp_llm_types::Item>,
	role: Role,
	parts: &mut Vec<Part>,
	blocks: &mut Vec<Value>,
) {
	if !parts.is_empty() {
		items.push(message_item(role, std::mem::take(parts), std::mem::take(blocks)));
	}
}

fn anthropic_parts(value: &Value) -> Result<(Vec<Part>, Vec<Value>), FacadeError> {
	let blocks = match value {
		Value::Null => return Ok((Vec::new(), Vec::new())),
		Value::String(text) => vec![json!({"type":"text","text":text})],
		Value::Array(blocks) => blocks.clone(),
		_ => return Err(invalid("Anthropic content must be a string or array")),
	};
	let mut parts = Vec::with_capacity(blocks.len());
	for block in &blocks {
		let kind = block.get("type").and_then(Value::as_str).unwrap_or("text");
		let part = match kind {
			"text" => Part::Text(Str::from(string_at(block, "text")?)),
			"image" | "document" => Part::Blob(anthropic_blob(block, kind)?),
			"thinking" => Part::Thinking(
				Thinking::builder()
					.text(Str::from(string_at(block, "thinking")?))
					.signature(Bytes::copy_from_slice(string_at(block, "signature")?.as_bytes()))
					.redacted(false)
					.build(),
			),
			"redacted_thinking" => Part::Thinking(
				Thinking::builder()
					.text(Str::new_static(""))
					.signature(Bytes::copy_from_slice(string_at(block, "data")?.as_bytes()))
					.redacted(true)
					.build(),
			),
			"server_tool_use" => Part::ServerTool(server_tool(block, ServerToolKind::Call)?),
			"web_search_tool_result"
			| "web_fetch_tool_result"
			| "code_execution_tool_result"
			| "bash_code_execution_tool_result"
			| "text_editor_code_execution_tool_result" => {
				Part::ServerTool(server_tool(block, ServerToolKind::Result)?)
			},
			"fallback" => Part::Fallback(
				ModelFallback::builder()
					.from_model(Str::from(fallback_model(block, "from")?))
					.to_model(Str::from(fallback_model(block, "to")?))
					.build(),
			),
			_ => return Err(invalid("unsupported Anthropic content block")),
		};
		parts.push(part);
	}
	Ok((parts, blocks))
}
fn fallback_model<'a>(block: &'a Value, field: &str) -> Result<&'a str, FacadeError> {
	block
		.get(field)
		.and_then(|value| value.get("model"))
		.and_then(Value::as_str)
		.ok_or_else(|| invalid("Anthropic fallback endpoint requires model"))
}

fn anthropic_blob(block: &Value, kind: &str) -> Result<BlobPart, FacadeError> {
	let source = block
		.get("source")
		.and_then(Value::as_object)
		.ok_or_else(|| invalid("Anthropic media block requires source"))?;
	let source_type = source
		.get("type")
		.and_then(Value::as_str)
		.unwrap_or("base64");
	let mime = source
		.get("media_type")
		.and_then(Value::as_str)
		.unwrap_or(if kind == "image" {
			"image/png"
		} else if source_type == "text" {
			"text/plain"
		} else {
			"application/pdf"
		});
	let inline = match source_type {
		"base64" => {
			let data = source
				.get("data")
				.and_then(Value::as_str)
				.ok_or_else(|| invalid("Anthropic base64 source requires data"))?;
			base64::decode(data.as_bytes())
				.into_vec()
				.map_err(|_| invalid("Anthropic media source contains invalid base64"))?
		},
		"text" if kind == "document" => source
			.get("data")
			.and_then(Value::as_str)
			.ok_or_else(|| invalid("Anthropic text document source requires data"))?
			.as_bytes()
			.to_vec(),
		"text" => return Err(invalid("Anthropic text sources are only valid for documents")),
		"url" => {
			source
				.get("url")
				.and_then(Value::as_str)
				.filter(|value| !value.is_empty())
				.ok_or_else(|| invalid("Anthropic URL source requires url"))?;
			Vec::new()
		},
		"file" => {
			source
				.get("file_id")
				.and_then(Value::as_str)
				.filter(|value| !value.is_empty())
				.ok_or_else(|| invalid("Anthropic file source requires file_id"))?;
			Vec::new()
		},
		"content" if kind == "document" => {
			source
				.get("content")
				.ok_or_else(|| invalid("Anthropic content document source requires content"))?;
			Vec::new()
		},
		"content" => return Err(invalid("Anthropic content sources are only valid for documents")),
		_ => return Err(invalid("unsupported Anthropic media source")),
	};
	let identity = if inline.is_empty() {
		serde_json::to_vec(source).expect("JSON values serialize")
	} else {
		inline.clone()
	};
	Ok(BlobPart::builder()
		.hash(*blake3::hash(&identity).as_bytes())
		.mime(Str::from(mime))
		.size(inline.len() as u64)
		.inline(Bytes::from(inline))
		.build())
}

fn server_tool(block: &Value, kind: ServerToolKind) -> Result<ServerTool, FacadeError> {
	let (id_field, payload_field) = match kind {
		ServerToolKind::Call => ("id", "input"),
		ServerToolKind::Result => ("tool_use_id", "content"),
		_ => return Err(invalid("unsupported server tool direction")),
	};
	let name = block
		.get("name")
		.and_then(Value::as_str)
		.or_else(|| block.get("type").and_then(Value::as_str))
		.ok_or_else(|| invalid("Anthropic server tool block requires a name"))?;
	let payload = match kind {
		ServerToolKind::Call => block.get(payload_field).cloned().unwrap_or(Value::Null),
		ServerToolKind::Result => block
			.get(payload_field)
			.cloned()
			.ok_or_else(|| invalid("Anthropic server tool result requires content"))?,
		_ => return Err(invalid("unsupported server tool direction")),
	};
	let mut metadata = Props::default();
	metadata.insert_ns("anthropic", "content_block", block.clone());
	Ok(ServerTool::builder()
		.provider(Str::from("anthropic"))
		.kind(kind)
		.id(Str::from(string_at(block, id_field)?))
		.name(Str::from(name))
		.payload_json(Bytes::from(serde_json::to_vec(&payload).expect("JSON values serialize")))
		.provider_metadata(metadata)
		.build())
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
				return error_response(Vendor::Anthropic, FacadeError::Turn(error));
			},
			_ => {},
		}
	}
	let Some(outcome) = outcome else {
		return error_response(Vendor::Anthropic, invalid("turn ended without an outcome"));
	};
	json_response(StatusCode::OK, &match message_value(&outcome, model.as_str()) {
		Ok(value) => value,
		Err(error) => return error_response(Vendor::Anthropic, error),
	})
}

fn streaming(
	mut events: futures::stream::BoxStream<'static, TurnEvent>,
	model: Str,
) -> FacadeResponse {
	let output = stream! {
		let id = format!("msg_{}", ulid::Ulid::generate());
		let mut kinds = BTreeMap::new();
		yield sse_named("message_start", &json!({"type":"message_start","message":{"id":id,"type":"message","role":"assistant","model":model,"content":[],"stop_reason":Value::Null,"stop_sequence":Value::Null,"usage":{"input_tokens":0,"output_tokens":0}}}));
		while let Some(event) = events.next().await {
			match event {
				TurnEvent::PartStart { index, kind, tool_call_id, tool_name } => {
					kinds.insert(index, kind);
					let block = match kind { StreamPartKind::Text => json!({"type":"text","text":""}), StreamPartKind::Thinking => json!({"type":"thinking","thinking":"","signature":""}), StreamPartKind::ToolCall => json!({"type":"tool_use","id":tool_call_id,"name":tool_name,"input":{}}), _ => continue };
					yield sse_named("content_block_start", &json!({"type":"content_block_start","index":index,"content_block":block}));
				}
				TurnEvent::PartDelta { index, chunk } => {
					let delta = match kinds.get(&index) { Some(StreamPartKind::Text) => json!({"type":"text_delta","text":String::from_utf8_lossy(&chunk)}), Some(StreamPartKind::Thinking) => json!({"type":"thinking_delta","thinking":String::from_utf8_lossy(&chunk)}), Some(StreamPartKind::ToolCall) => json!({"type":"input_json_delta","partial_json":String::from_utf8_lossy(&chunk)}), _ => continue };
					yield sse_named("content_block_delta", &json!({"type":"content_block_delta","index":index,"delta":delta}));
				}
				TurnEvent::PartEnd { index, .. } => yield sse_named("content_block_stop", &json!({"type":"content_block_stop","index":index})),
				TurnEvent::Outcome(outcome) => {
					yield sse_named("message_delta", &json!({"type":"message_delta","delta":{"stop_reason":anthropic_stop(outcome.stop),"stop_sequence":Value::Null},"usage":usage_value(outcome.usage.as_ref())}));
					yield sse_named("message_stop", &json!({"type":"message_stop"}));
					break;
				}
				TurnEvent::Error(error) => { yield sse_named("error", &json!({"type":"error","error":{"type":"api_error","message":error.detail}})); break; }
				_ => {}
			}
		}
	};
	sse_response(output)
}

fn message_value(outcome: &ChatOutcome, requested_model: &str) -> Result<Value, FacadeError> {
	let mut content = Vec::new();
	for item in &outcome.output {
		match &item.kind {
			ItemKind::Message(message) if message.role == Role::Assistant => {
				if message.parts.is_empty() {
					if let Some(block) = item.props.get_ns("anthropic", "server_tool_block") {
						content.push(block.clone());
					}
					continue;
				}
				let templates = item
					.props
					.get_ns("anthropic", "content_blocks")
					.and_then(Value::as_array);
				if let Some(values) = templates
					&& values.len() != message.parts.len()
				{
					return Err(invalid("Anthropic content block metadata does not match parts"));
				}
				let citations = item
					.props
					.get_ns("anthropic", "citations")
					.and_then(Value::as_array);
				for (index, part) in message.parts.iter().enumerate() {
					let mut block =
						anthropic_part_value(part, templates.and_then(|values| values.get(index)))?;
					if let Some(citations) = citations {
						let values = citations
							.iter()
							.filter(|entry| {
								entry.get("part_index").and_then(Value::as_u64) == Some(index as u64)
							})
							.filter_map(|entry| entry.get("citation").cloned())
							.collect::<Vec<_>>();
						if !values.is_empty() {
							block["citations"] = Value::Array(values);
						}
					}
					content.push(block);
				}
			},
			ItemKind::ToolCall(call) => {
				let template = call
					.provider_metadata
					.as_ref()
					.and_then(|props| props.get_ns("anthropic", "content_block"));
				let mut block = template.cloned().unwrap_or_else(|| json!({}));
				let object = block
					.as_object_mut()
					.ok_or_else(|| invalid("Anthropic tool metadata must be an object"))?;
				object.insert("type".into(), json!("tool_use"));
				if template.is_none() {
					object.insert("id".into(), json!(call.id.to_string()));
				}
				object.insert("name".into(), json!(call.name));
				object.insert(
					"input".into(),
					serde_json::from_slice::<Value>(&call.args_json)
						.map_err(|_| invalid("tool arguments are not valid JSON"))?,
				);
				content.push(block);
			},
			_ => {},
		}
	}
	Ok(
		json!({"id":format!("msg_{}",ulid::Ulid::generate()),"type":"message","role":"assistant","model":if outcome.model.is_empty(){requested_model}else{outcome.model.as_str()},"content":content,"stop_reason":anthropic_stop(outcome.stop),"stop_sequence":Value::Null,"usage":usage_value(outcome.usage.as_ref())}),
	)
}

fn anthropic_part_value(part: &Part, template: Option<&Value>) -> Result<Value, FacadeError> {
	let mut value = template.cloned().unwrap_or_else(|| json!({}));
	let object = value
		.as_object_mut()
		.ok_or_else(|| invalid("Anthropic content block metadata must be an object"))?;
	match part {
		Part::Text(text) => {
			object.insert("type".into(), json!("text"));
			object.insert("text".into(), json!(text));
		},
		Part::Blob(blob) => {
			if template.is_none() && blob.inline.is_empty() {
				return Err(invalid("Anthropic media requires inline bytes or source metadata"));
			}
			let kind = template
				.and_then(|value| value.get("type"))
				.and_then(Value::as_str)
				.unwrap_or(if blob.mime.starts_with("image/") {
					"image"
				} else {
					"document"
				});
			object.insert("type".into(), json!(kind));
			let source_type = object
				.get("source")
				.and_then(|source| source.get("type"))
				.and_then(Value::as_str);
			if template.is_none() || source_type == Some("base64") {
				object.insert(
					"source".into(),
					json!({
						"type":"base64",
						"media_type":blob.mime,
						"data":base64::encode(&blob.inline).into_string()
					}),
				);
			} else if source_type == Some("text") {
				let text = std::str::from_utf8(&blob.inline)
					.map_err(|_| invalid("Anthropic text document bytes are not UTF-8"))?;
				let source = object
					.get_mut("source")
					.and_then(Value::as_object_mut)
					.expect("validated source object");
				source.insert("media_type".into(), json!(blob.mime));
				source.insert("data".into(), json!(text));
			}
		},
		Part::Thinking(thinking) if thinking.redacted => {
			object.insert("type".into(), json!("redacted_thinking"));
			object.insert(
				"data".into(),
				json!(
					std::str::from_utf8(&thinking.signature)
						.map_err(|_| invalid("Anthropic redacted thinking data is not UTF-8"))?
				),
			);
			object.remove("thinking");
			object.remove("signature");
		},
		Part::Thinking(thinking) => {
			object.insert("type".into(), json!("thinking"));
			object.insert("thinking".into(), json!(thinking.text));
			object.insert(
				"signature".into(),
				json!(
					std::str::from_utf8(&thinking.signature)
						.map_err(|_| invalid("Anthropic thinking signature is not UTF-8"))?
				),
			);
			object.remove("data");
		},
		Part::Fallback(fallback) => {
			object.insert("type".into(), json!("fallback"));
			for (field, model) in
				[("from", fallback.from_model.as_str()), ("to", fallback.to_model.as_str())]
			{
				let mut endpoint = object.get(field).cloned().unwrap_or_else(|| json!({}));
				let endpoint_object = endpoint
					.as_object_mut()
					.ok_or_else(|| invalid("Anthropic fallback endpoint must be an object"))?;
				endpoint_object.insert("model".into(), json!(model));
				object.insert(field.into(), endpoint);
			}
		},
		Part::ServerTool(tool) => {
			if tool.provider != "anthropic" {
				return Err(invalid("cannot project a foreign server tool as Anthropic"));
			}
			let payload = serde_json::from_slice::<Value>(&tool.payload_json)
				.map_err(|_| invalid("server tool payload is not valid JSON"))?;
			match tool.kind {
				ServerToolKind::Call => {
					object.insert("type".into(), json!("server_tool_use"));
					object.insert("id".into(), json!(tool.id));
					object.insert("name".into(), json!(tool.name));
					if template.is_none() || template.and_then(|value| value.get("input")).is_some() {
						object.insert("input".into(), payload);
					}
				},
				ServerToolKind::Result => {
					let wire_type = template
						.and_then(|value| value.get("type"))
						.and_then(Value::as_str)
						.unwrap_or(tool.name.as_str());
					object.insert("type".into(), json!(wire_type));
					object.insert("tool_use_id".into(), json!(tool.id));
					object.insert("content".into(), payload);
				},
				_ => return Err(invalid("unsupported server tool direction")),
			}
		},
		_ => return Err(invalid("unsupported canonical Anthropic part")),
	}
	Ok(value)
}

fn usage_value(usage: Option<&Usage>) -> Value {
	let mut value = json!({
		"input_tokens": usage.map_or(0, |value| value.input_tokens),
		"output_tokens": usage.map_or(0, |value| value.output_tokens),
		"cache_read_input_tokens": usage.map_or(0, |value| value.cache_read_tokens),
		"cache_creation_input_tokens": usage.map_or(0, |value| value.cache_write_tokens),
	});
	let Some(usage) = usage else {
		return value;
	};
	if let Some(cache) = usage.cache_ttl {
		value["cache_creation"] = json!({
			"ephemeral_5m_input_tokens": cache.ephemeral_5m_tokens.unwrap_or(0),
			"ephemeral_1h_input_tokens": cache.ephemeral_1h_tokens.unwrap_or(0),
		});
	}
	if let Some(tools) = usage.server_tools {
		value["server_tool_use"] = json!({
			"web_search_requests": tools.web_search_requests.unwrap_or(0),
			"web_fetch_requests": tools.web_fetch_requests.unwrap_or(0),
		});
	}
	value
}

const fn anthropic_stop(reason: StopReason) -> &'static str {
	match reason {
		StopReason::ToolUse => "tool_use",
		StopReason::MaxTokens => "max_tokens",
		_ => "end_turn",
	}
}

fn invalid(detail: &str) -> FacadeError {
	FacadeError::Invalid(Str::from(detail))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn projects_anthropic_tool_blocks() {
		let wire = WireRequest {
			model: Str::from("test"),
			system: Value::String("be terse".into()),
			messages: vec![json!({"role":"assistant","content":[
				{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"omp"}}
			]})],
			tools: vec![json!({"name":"lookup","input_schema":{"type":"object"}})],
			stream: false,
			..WireRequest::default()
		};
		let canonical = canonical_input(&wire).expect("valid Anthropic request");
		assert_eq!(canonical.tools.len(), 1);
		assert!(
			canonical
				.thread
				.items
				.iter()
				.any(|item| matches!(&item.kind, ItemKind::ToolCall(_)))
		);
	}

	fn rendered_message_blocks(input: &CanonicalInput) -> Vec<Value> {
		let mut output = Vec::new();
		for item in &input.thread.items {
			if let ItemKind::Message(message) = &item.kind {
				let templates = item
					.props
					.get_ns("anthropic", "content_blocks")
					.and_then(Value::as_array)
					.expect("block metadata");
				for (part, template) in message.parts.iter().zip(templates) {
					output.push(anthropic_part_value(part, Some(template)).expect("rendered part"));
				}
			}
		}
		output
	}

	#[test]
	fn round_trips_supported_anthropic_content_blocks() {
		let blocks = vec![
			json!({"type":"text","text":"answer","citations":[{"type":"char_location","start_char_index":0,"end_char_index":6}],"cache_control":{"type":"ephemeral","ttl":"1h"}}),
			json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="},"cache_control":{"type":"ephemeral"}}),
			json!({"type":"document","source":{"type":"text","media_type":"text/plain","data":"manual"},"title":"Guide","context":"trusted","citations":{"enabled":true},"cache_control":{"type":"ephemeral"}}),
			json!({"type":"thinking","thinking":"reason","signature":"signed-value"}),
			json!({"type":"redacted_thinking","data":"opaque-value"}),
			json!({"type":"server_tool_use","id":"srv_1","name":"web_search","input":{"query":"rust"},"caller":{"type":"direct"}}),
			json!({"type":"web_search_tool_result","tool_use_id":"srv_1","content":[{"type":"web_search_result","url":"https://example.test","title":"Result","encrypted_content":"cipher"}],"cache_control":{"type":"ephemeral"}}),
			json!({"type":"fallback","from":{"model":"claude-opus-4-1"},"to":{"model":"claude-sonnet-4-5"}}),
		];
		let wire = WireRequest {
			model: Str::from("test"),
			messages: vec![json!({"role":"assistant","content":blocks})],
			max_tokens: Some(128),
			..WireRequest::default()
		};
		let canonical = canonical_input(&wire).expect("supported blocks");
		assert_eq!(rendered_message_blocks(&canonical), blocks);

		let message = canonical
			.thread
			.items
			.iter()
			.find_map(|item| match &item.kind {
				ItemKind::Message(message) => Some(message),
				_ => None,
			})
			.expect("assistant message");
		assert!(matches!(&message.parts[1], Part::Blob(_)));
		assert!(matches!(&message.parts[2], Part::Blob(_)));
		assert!(matches!(&message.parts[3], Part::Thinking(_)));
		assert!(matches!(&message.parts[4], Part::Thinking(_)));
		assert!(matches!(&message.parts[5], Part::ServerTool(_)));
		assert!(matches!(&message.parts[6], Part::ServerTool(_)));
		assert!(matches!(&message.parts[7], Part::Fallback(_)));
	}

	#[test]
	fn round_trips_nested_tool_result_media_without_filtering() {
		let nested = vec![
			json!({"type":"text","text":"found","cache_control":{"type":"ephemeral"}}),
			json!({"type":"image","source":{"type":"url","url":"https://example.test/image.png"},"cache_control":{"type":"ephemeral"}}),
			json!({"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERg=="},"title":"Report","citations":{"enabled":true}}),
		];
		let outer = json!({
			"type":"tool_result",
			"tool_use_id":"toolu_1",
			"content":nested,
			"is_error":false,
			"cache_control":{"type":"ephemeral","ttl":"1h"}
		});
		let wire = WireRequest {
			model: Str::from("test"),
			messages: vec![
				json!({"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"omp"},"cache_control":{"type":"ephemeral"}}]}),
				json!({"role":"user","content":[outer.clone()]}),
			],
			max_tokens: Some(128),
			..WireRequest::default()
		};
		let canonical = canonical_input(&wire).expect("nested multimodal tool result");
		let result = canonical
			.thread
			.items
			.iter()
			.find_map(|item| match &item.kind {
				ItemKind::ToolResult(result) => Some(result),
				_ => None,
			})
			.expect("tool result");
		assert_eq!(result.parts.len(), 3);
		assert!(matches!(&result.parts[1], Part::Blob(_)));
		assert!(matches!(&result.parts[2], Part::Blob(_)));
		let templates = result
			.provider_metadata
			.as_ref()
			.and_then(|props| props.get_ns("anthropic", "content_blocks"))
			.and_then(Value::as_array)
			.expect("nested metadata");
		let rendered = result
			.parts
			.iter()
			.zip(templates)
			.map(|(part, template)| anthropic_part_value(part, Some(template)).expect("part"))
			.collect::<Vec<_>>();
		assert_eq!(rendered, nested);
		assert_eq!(
			result
				.provider_metadata
				.as_ref()
				.and_then(|props| props.get_ns("anthropic", "content_block")),
			Some(&outer)
		);
	}

	#[test]
	fn rejects_unknown_top_level_and_nested_blocks() {
		let unknown = WireRequest {
			model: Str::from("test"),
			messages: vec![json!({"role":"user","content":[{"type":"future_block","value":1}]})],
			max_tokens: Some(1),
			..WireRequest::default()
		};
		assert!(canonical_input(&unknown).is_err());

		let nested = WireRequest {
			model: Str::from("test"),
			messages: vec![json!({"role":"user","content":[{
				"type":"tool_result",
				"tool_use_id":"toolu_missing",
				"content":[{"type":"future_nested_block","value":1}]
			}]})],
			max_tokens: Some(1),
			..WireRequest::default()
		};
		assert!(canonical_input(&nested).is_err());
	}
}
