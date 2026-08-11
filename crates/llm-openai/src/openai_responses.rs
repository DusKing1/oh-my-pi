//! `OpenAI` Responses (`/v1/responses`) item and typed-event codec.

use std::collections::{BTreeMap, BTreeSet};

use bytes::{Bytes, BytesMut};
use omp_core::SmolStr;
use omp_llm_catalog::{
	TransportId,
	compat::{Compat, ReasoningWireFormat, ToolStrictMode},
};
use omp_llm_error::{Evidence, Kind, WireApi, classify, envelope};
use omp_llm_transport::{
	DecodeState, Frame, Transport,
	normalize::{normalize as normalize_schema, openai_strict, with_tool_use_precedence},
};
use omp_llm_types::{
	Accuracy, BlobPart, CacheRetention, CallId, CallIdMapper, ChatOutcome, ChatRequest, Cost, Error,
	Fallback, Item, ItemKind, Message, Part, Props, ResponseFormatKind, Role, StopReason,
	StreamPartKind, Thinking, ToolCall, ToolCallIdProfile, ToolChoice, TurnError, TurnErrorKind,
	TurnEvent, Unsupported, UnsupportedAction, Usage,
};
use serde_json::{Map, Value, json};
use smallvec::SmallVec;

use crate::{
	model_policy::OpenAiModelPolicy,
	responses_tool_repair::{SentCall, ToolKind, call_kind_of, repair_responses_tool_pairs},
};

/// `OpenAI` Responses codec configured with gateway-held state for one turn.
///
/// The codec never discovers or invents `previous_response_id`. The gateway may
/// supply the identifier obtained from an earlier authoritative terminal event
/// either in `openai/previous_response_id` request options or at construction.
/// When gateway options also carry `openai/previous_response_item_count`, the
/// full canonical thread stays owned by the request while only items after that
/// committed boundary are encoded into `input`.
#[derive(Debug, Default)]
pub struct OpenAiResponsesCodec {
	previous_response_id: Option<SmolStr>,
	call_ids:             CallIdMapper,
}

impl OpenAiResponsesCodec {
	/// Creates a codec without constructor-held continuation state.
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	/// Creates a codec whose next turn chains from a gateway-held response.
	#[must_use]
	pub fn with_previous_response_id(response_id: impl Into<SmolStr>) -> Self {
		Self {
			previous_response_id: Some(response_id.into()),
			call_ids:             CallIdMapper::new(),
		}
	}

	/// Returns the gateway-supplied chaining identifier, if any.
	#[must_use]
	pub fn previous_response_id(&self) -> Option<&str> {
		self.previous_response_id.as_deref()
	}
}

impl Transport for OpenAiResponsesCodec {
	fn id(&self) -> TransportId {
		TransportId::OpenAiResponses
	}

	fn encode(
		&self,
		req: &ChatRequest,
		compat: &Compat,
	) -> Result<(Bytes, Vec<Unsupported>), Error> {
		let mut unsupported = Vec::new();
		let policy = OpenAiModelPolicy::resolve(req, compat, &mut unsupported);
		let compat = &policy.compat;
		let mut fail_closed = false;
		let mut body = Map::new();
		body.insert("model".into(), Value::String(req.model.to_string()));
		let mut include = Vec::new();
		if policy.include_encrypted_reasoning {
			include.push(Value::String("reasoning.encrypted_content".into()));
		}
		body.insert("stream".into(), Value::Bool(true));

		let instructions = system_instructions(req, &mut unsupported);
		if !instructions.is_empty() && !policy.supports_developer_role {
			body.insert("instructions".into(), Value::String(instructions.clone()));
		}

		if let Some(cache) = &req.cache {
			body.insert("prompt_cache_key".into(), Value::String(cache.session_key.to_string()));
			if matches!(cache.retention, Some(CacheRetention::Long)) {
				body.insert("prompt_cache_retention".into(), Value::String("24h".into()));
			}
		}
		if policy.supports_store {
			body.insert("store".into(), Value::Bool(false));
		}

		let option_previous_response_id = req
			.provider_options
			.as_ref()
			.and_then(|props| props.get_ns("openai", "previous_response_id"))
			.and_then(Value::as_str);
		let previous_response_id =
			option_previous_response_id.or_else(|| self.previous_response_id.as_deref());
		let boundary_value = req
			.provider_options
			.as_ref()
			.and_then(|props| props.get_ns("openai", "previous_response_item_count"));
		let boundary = boundary_value
			.and_then(Value::as_u64)
			.and_then(|value| usize::try_from(value).ok());
		let boundary_valid = boundary.is_none_or(|value| value <= req.thread.items.len());
		let boundary_has_anchor = boundary.is_none() || previous_response_id.is_some();
		let effective_previous_response_id = (boundary_valid && boundary_has_anchor)
			.then_some(previous_response_id)
			.flatten();
		let input_start =
			if compat.stateful_response_chaining && effective_previous_response_id.is_some() {
				boundary.unwrap_or(0)
			} else {
				0
			};
		let mut input = encode_input(req, input_start, &self.call_ids, &policy, &mut unsupported);
		if policy.supports_developer_role && !instructions.is_empty() {
			input.insert(0, json!({"role":"developer", "content":instructions}));
		}
		body.insert("input".into(), Value::Array(input));
		if boundary_value.is_some() && (!boundary_valid || !boundary_has_anchor) {
			unsupported.push(dropped(
				"provider_options:openai/previous_response_item_count",
				"continuation boundary requires a previous response and must fit the canonical thread",
			));
		}
		if compat.stateful_response_chaining && policy.supports_store {
			body.insert("store".into(), Value::Bool(true));
			if let Some(previous) = effective_previous_response_id {
				body.insert("previous_response_id".into(), Value::String(previous.into()));
			}
		} else if previous_response_id.is_some() {
			unsupported.push(dropped(
				"provider_options:openai/previous_response_id",
				"resolved provider path does not support Responses continuation",
			));
		}

		if let Some(sampling) = &req.sampling {
			if let Some(maximum) = sampling.max_output_tokens
				&& !req
					.model_policy
					.as_deref()
					.and_then(|policy| policy.omit_max_output_tokens)
					.unwrap_or(false)
			{
				body.insert("max_output_tokens".into(), Value::from(maximum));
			}
			if compat.sampling_params {
				insert_optional_number(&mut body, "temperature", sampling.temperature);
				insert_optional_number(&mut body, "top_p", sampling.top_p);
				insert_optional_number(&mut body, "frequency_penalty", sampling.frequency_penalty);
				insert_optional_number(&mut body, "presence_penalty", sampling.presence_penalty);
			} else {
				for (name, present) in [
					("sampling.temperature", sampling.temperature.is_some()),
					("sampling.top_p", sampling.top_p.is_some()),
					("sampling.frequency_penalty", sampling.frequency_penalty.is_some()),
					("sampling.presence_penalty", sampling.presence_penalty.is_some()),
				] {
					if present {
						report(
							&mut unsupported,
							name,
							"endpoint rejects sampling parameters",
							Fallback::Ignore,
							&mut fail_closed,
						);
					}
				}
			}
			if sampling.top_k.is_some() {
				report(
					&mut unsupported,
					"sampling.top_k",
					"Responses has no top-k parameter",
					Fallback::Ignore,
					&mut fail_closed,
				);
			}
			if sampling.min_p.is_some() {
				report(
					&mut unsupported,
					"sampling.min_p",
					"Responses has no min-p parameter",
					Fallback::Ignore,
					&mut fail_closed,
				);
			}
			if sampling.stop.is_some() {
				report(
					&mut unsupported,
					"sampling.stop",
					"Responses has no stop-sequence parameter",
					Fallback::Ignore,
					&mut fail_closed,
				);
			}
		}

		if let Some(reasoning) = &req.thinking {
			if compat.reasoning_wire_format == ReasoningWireFormat::OpenAiResponses {
				let mut value = Map::new();
				if let Some(effort) = reasoning.value.effort
					&& !policy.omit_reasoning_effort
				{
					value
						.insert("effort".into(), Value::String(policy.mapped_effort(effort).to_string()));
				}
				match reasoning.value.hide_summary {
					Some(false) => {
						value.insert("summary".into(), Value::String("auto".into()));
					},
					Some(true) => {
						value.insert("summary".into(), Value::Null);
					},
					None => {},
				}
				if matches!(
					req.model_policy
						.as_deref()
						.and_then(|policy| policy.reasoning_mode),
					Some(omp_llm_types::ResolvedReasoningMode::Pro)
				) {
					value.insert("mode".into(), Value::String("pro".into()));
				}
				body.insert("reasoning".into(), Value::Object(value));
				if reasoning.value.budget_tokens.is_some() {
					report(
						&mut unsupported,
						"thinking.budget_tokens",
						"Responses accepts qualitative effort, not a token budget",
						reasoning.on_unsupported,
						&mut fail_closed,
					);
				}
			} else {
				report(
					&mut unsupported,
					"thinking",
					"provider path does not accept Responses reasoning controls",
					reasoning.on_unsupported,
					&mut fail_closed,
				);
			}
		} else if matches!(
			req.model_policy
				.as_deref()
				.and_then(|policy| policy.reasoning_mode),
			Some(omp_llm_types::ResolvedReasoningMode::Pro)
		) {
			body.insert("reasoning".into(), json!({"mode":"pro"}));
		}

		encode_tools(req, &policy, &mut body, &mut unsupported, &mut fail_closed);
		encode_format(req, compat, &mut body, &mut unsupported, &mut fail_closed);
		encode_provider_options(req, &mut body, &mut include, &mut unsupported);
		if !include.is_empty() {
			body.insert("include".into(), Value::Array(include));
		}
		if let Some(extra) = &policy.extra_body {
			for (key, value) in extra {
				body.insert(key.clone(), value.clone());
			}
		}
		if fail_closed {
			return Err(Error::Unsupported(unsupported));
		}
		serde_json::to_vec(&Value::Object(body))
			.map(|bytes| (Bytes::from(bytes), unsupported))
			.map_err(|error| Error::Provider(SmolStr::from(error.to_string())))
	}

	fn decode(
		&self,
		frame: Frame<'_>,
		state: &mut DecodeState,
	) -> Result<SmallVec<TurnEvent, 2>, Error> {
		let state = state.get_or_insert_with(ResponsesDecodeState::default);
		if state.terminal {
			return Ok(SmallVec::new());
		}
		let data = match frame {
			Frame::Data(data) | Frame::Event { data, .. } => data,
			Frame::Done => {
				state.terminal = true;
				let mut out = SmallVec::new();
				out.push(TurnEvent::Error(upstream_error(
					"Responses stream ended before an authoritative terminal event",
				)));
				return Ok(out);
			},
			_ => return Ok(SmallVec::new()),
		};
		let event: Value = match serde_json::from_slice(data) {
			Ok(event) => event,
			Err(error) => {
				state.terminal = true;
				let mut out = SmallVec::new();
				out.push(TurnEvent::Error(upstream_error(&format!(
					"invalid Responses event: {error}",
				))));
				return Ok(out);
			},
		};
		let kind = event
			.get("type")
			.and_then(Value::as_str)
			.unwrap_or_default();
		if kind == "error" || event.get("error").is_some() {
			state.terminal = true;
			let body = std::str::from_utf8(data).expect("valid JSON is UTF-8");
			let mut out = SmallVec::new();
			out.push(TurnEvent::Error(classify_error_body(body)));
			return Ok(out);
		}
		Ok(decode_event(kind, &event, state))
	}
}

#[derive(Debug)]
enum OutputSlot {
	Text {
		wire_id: SmolStr,
		text:    BytesMut,
	},
	Thinking {
		wire_id:   SmolStr,
		text:      BytesMut,
		encrypted: Bytes,
	},
	Tool {
		id:        CallId,
		item_id:   SmolStr,
		wire_id:   SmolStr,
		name:      SmolStr,
		arguments: BytesMut,
		custom:    bool,
	},
	Image {
		data: BytesMut,
	},
	Server {
		item:   Value,
		events: Vec<Value>,
	},
}

#[derive(Debug, Default)]
struct ResponsesDecodeState {
	response_id:  SmolStr,
	model:        SmolStr,
	outputs:      BTreeMap<u32, OutputSlot>,
	item_options: BTreeMap<u32, Props>,
	ended:        BTreeSet<u32>,
	terminal:     bool,
}

fn decode_event(
	kind: &str,
	event: &Value,
	state: &mut ResponsesDecodeState,
) -> SmallVec<TurnEvent, 2> {
	let mut out = SmallVec::new();
	match kind {
		"response.created" | "response.queued" | "response.in_progress" => {
			capture_response(event.get("response"), state);
		},
		"response.output_item.added" => {
			let index = added_output_index(event, state);
			let item = event.get("item").unwrap_or(&Value::Null);
			if let Some(slot) = slot_from_item(item) {
				if let Some(start) = part_start(index, &slot) {
					out.push(start);
				}
				state.outputs.insert(index, slot);
			}
			capture_output_item_options(index, item, state);
		},
		"response.output_text.delta" | "response.refusal.delta" => {
			append_delta(event, state, &mut out, SlotKind::Text);
		},
		"response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
			append_delta(event, state, &mut out, SlotKind::Thinking);
		},
		"response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
			append_delta(event, state, &mut out, SlotKind::Tool);
		},
		"response.output_text.done" | "response.refusal.done" => {
			replace_done_text(event, state, SlotKind::Text);
		},
		"response.reasoning_summary_text.done" | "response.reasoning_text.done" => {
			replace_done_text(event, state, SlotKind::Thinking);
		},
		"response.function_call_arguments.done" | "response.custom_tool_call_input.done" => {
			replace_done_text(event, state, SlotKind::Tool);
		},
		"response.image_generation_call.partial_image" => {
			let Some(index) = event_output_index(event, state) else {
				return out;
			};
			let delta = event
				.get("partial_image_b64")
				.or_else(|| event.get("delta"))
				.and_then(Value::as_str)
				.unwrap_or_default();
			if let Some(OutputSlot::Image { data }) = state.outputs.get_mut(&index) {
				data.extend_from_slice(delta.as_bytes());
			}
		},
		"response.output_item.done" => {
			let index =
				event_output_index(event, state).unwrap_or_else(|| added_output_index(event, state));
			if let Some(item) = event.get("item") {
				if !state.outputs.contains_key(&index)
					&& let Some(slot) = slot_from_item(item)
				{
					if let Some(start) = part_start(index, &slot) {
						out.push(start);
					}
					state.outputs.insert(index, slot);
				}
				complete_slot(item, state.outputs.get_mut(&index));
				capture_output_item_options(index, item, state);
			}
			if state.ended.insert(index)
				&& let Some(slot) = state.outputs.get(&index)
				&& is_stream_part(slot)
			{
				let signature = match slot {
					OutputSlot::Thinking { encrypted, .. } => encrypted.clone(),
					_ => Bytes::new(),
				};
				out.push(TurnEvent::PartEnd { index, signature });
			}
		},
		"response.completed" | "response.incomplete" | "response.done" => {
			state.terminal = true;
			let response = event.get("response");
			capture_response(response, state);
			capture_terminal_outputs(response, state);
			let status = response
				.and_then(|value| value.get("status"))
				.and_then(Value::as_str)
				.unwrap_or_default();
			if matches!(status, "failed" | "cancelled") {
				out.push(TurnEvent::Error(response_error(response, status)));
			} else {
				close_open_parts(state, &mut out);
				out.push(TurnEvent::Outcome(build_outcome(
					response,
					kind == "response.incomplete" || status == "incomplete",
					state,
				)));
			}
		},
		"response.failed" | "response.cancelled" => {
			state.terminal = true;
			let response = event.get("response");
			capture_response(response, state);
			out.push(TurnEvent::Error(response_error(response, kind)));
		},
		_ => update_server_slot(kind, event, state),
	}
	out
}

fn slot_from_item(item: &Value) -> Option<OutputSlot> {
	match item.get("type").and_then(Value::as_str).unwrap_or_default() {
		"reasoning" => Some(OutputSlot::Thinking {
			wire_id:   str_field(item, "id"),
			text:      BytesMut::new(),
			encrypted: item
				.get("encrypted_content")
				.and_then(Value::as_str)
				.map_or_else(Bytes::new, |value| Bytes::copy_from_slice(value.as_bytes())),
		}),
		"function_call" | "custom_tool_call" => {
			let custom = item.get("type").and_then(Value::as_str) == Some("custom_tool_call");
			Some(OutputSlot::Tool {
				id: CallId::new(),
				item_id: str_field(item, "id"),
				wire_id: str_field(item, "call_id"),
				name: str_field(item, "name"),
				arguments: BytesMut::from(
					item
						.get(if custom { "input" } else { "arguments" })
						.and_then(Value::as_str)
						.unwrap_or_default()
						.as_bytes(),
				),
				custom,
			})
		},
		"message" => {
			Some(OutputSlot::Text { wire_id: str_field(item, "id"), text: BytesMut::new() })
		},
		"image_generation_call" => Some(OutputSlot::Image { data: BytesMut::new() }),
		"web_search_call"
		| "file_search_call"
		| "code_interpreter_call"
		| "computer_call"
		| "mcp_call"
		| "local_shell_call" => Some(OutputSlot::Server { item: item.clone(), events: Vec::new() }),
		_ if item.is_object() => {
			Some(OutputSlot::Server { item: item.clone(), events: Vec::new() })
		},
		_ => None,
	}
}

fn added_output_index(event: &Value, state: &ResponsesDecodeState) -> u32 {
	event
		.get("output_index")
		.and_then(Value::as_u64)
		.and_then(|value| u32::try_from(value).ok())
		.unwrap_or_else(|| {
			state
				.outputs
				.last_key_value()
				.map_or(0, |(index, _)| index.saturating_add(1))
		})
}

fn event_output_index(event: &Value, state: &ResponsesDecodeState) -> Option<u32> {
	if let Some(index) = event
		.get("output_index")
		.and_then(Value::as_u64)
		.and_then(|value| u32::try_from(value).ok())
	{
		return Some(index);
	}
	let wire_id = event
		.get("item_id")
		.and_then(Value::as_str)
		.or_else(|| event.pointer("/item/id").and_then(Value::as_str))
		.or_else(|| event.pointer("/item/call_id").and_then(Value::as_str))?;
	state
		.outputs
		.iter()
		.find_map(|(index, slot)| slot_matches_wire_id(slot, wire_id).then_some(*index))
}

fn slot_matches_wire_id(slot: &OutputSlot, wire_id: &str) -> bool {
	match slot {
		OutputSlot::Text { wire_id: id, .. } | OutputSlot::Thinking { wire_id: id, .. } => {
			id.as_str() == wire_id
		},
		OutputSlot::Tool { item_id, wire_id: call_id, .. } => {
			item_id.as_str() == wire_id
				|| call_id.as_str() == wire_id
				|| wire_id
					.strip_prefix("fc_")
					.is_some_and(|unprefixed| call_id.as_str() == unprefixed)
		},
		OutputSlot::Server { item, .. } => {
			item
				.get("id")
				.or_else(|| item.get("call_id"))
				.and_then(Value::as_str)
				== Some(wire_id)
		},
		OutputSlot::Image { .. } => false,
	}
}

fn part_start(index: u32, slot: &OutputSlot) -> Option<TurnEvent> {
	let (kind, tool_call_id, tool_name) = match slot {
		OutputSlot::Text { .. } => (StreamPartKind::Text, SmolStr::default(), SmolStr::default()),
		OutputSlot::Thinking { .. } => {
			(StreamPartKind::Thinking, SmolStr::default(), SmolStr::default())
		},
		OutputSlot::Tool { id, name, .. } => {
			(StreamPartKind::ToolCall, SmolStr::from(id.to_string()), name.clone())
		},
		OutputSlot::Image { .. } | OutputSlot::Server { .. } => return None,
	};
	Some(TurnEvent::PartStart { index, kind, tool_call_id, tool_name })
}

fn is_stream_part(slot: &OutputSlot) -> bool {
	matches!(slot, OutputSlot::Text { .. } | OutputSlot::Thinking { .. } | OutputSlot::Tool { .. })
}

fn close_open_parts(state: &mut ResponsesDecodeState, out: &mut SmallVec<TurnEvent, 2>) {
	let pending: Vec<(u32, Bytes)> = state
		.outputs
		.iter()
		.filter_map(|(index, slot)| {
			(!state.ended.contains(index) && is_stream_part(slot)).then(|| {
				let signature = match slot {
					OutputSlot::Thinking { encrypted, .. } => encrypted.clone(),
					_ => Bytes::new(),
				};
				(*index, signature)
			})
		})
		.collect();
	for (index, signature) in pending {
		state.ended.insert(index);
		out.push(TurnEvent::PartEnd { index, signature });
	}
}

fn replace_done_text(event: &Value, state: &mut ResponsesDecodeState, expected: SlotKind) {
	let Some(index) = event_output_index(event, state) else {
		return;
	};
	let value = event
		.get(match expected {
			SlotKind::Tool if event.get("input").is_some() => "input",
			SlotKind::Tool => "arguments",
			_ => "text",
		})
		.and_then(Value::as_str)
		.unwrap_or_default();
	match (state.outputs.get_mut(&index), expected) {
		(Some(OutputSlot::Text { text, .. }), SlotKind::Text)
		| (Some(OutputSlot::Thinking { text, .. }), SlotKind::Thinking)
		| (Some(OutputSlot::Tool { arguments: text, .. }), SlotKind::Tool) => {
			text.clear();
			text.extend_from_slice(value.as_bytes());
		},
		_ => {},
	}
}

fn capture_terminal_outputs(response: Option<&Value>, state: &mut ResponsesDecodeState) {
	let Some(items) = response
		.and_then(|value| value.get("output"))
		.and_then(Value::as_array)
	else {
		return;
	};
	for (position, item) in items.iter().enumerate() {
		let Ok(index) = u32::try_from(position) else {
			break;
		};
		if !state.outputs.contains_key(&index)
			&& let Some(slot) = slot_from_item(item)
		{
			state.outputs.insert(index, slot);
			// A terminal-only item has no matching PartStart; retain it in the
			// outcome without manufacturing an orphan PartEnd.
			state.ended.insert(index);
		}
		complete_slot(item, state.outputs.get_mut(&index));
		capture_output_item_options(index, item, state);
	}
}

fn capture_output_item_options(index: u32, item: &Value, state: &mut ResponsesDecodeState) {
	let options = state.item_options.entry(index).or_default();
	for key in ["cache_control", "metadata", "prompt_cache_breakpoint"] {
		if let Some(value) = item.get(key) {
			options.insert_ns("openai", key, value.clone());
		}
	}
}

fn update_server_slot(kind: &str, event: &Value, state: &mut ResponsesDecodeState) {
	if !(kind.contains("web_search_call")
		|| kind.contains("file_search_call")
		|| kind.contains("code_interpreter_call")
		|| kind.contains("computer_call")
		|| kind.contains("mcp_call"))
	{
		return;
	}
	let Some(index) = event_output_index(event, state) else {
		return;
	};
	if let Some(OutputSlot::Server { events, .. }) = state.outputs.get_mut(&index) {
		events.push(event.clone());
	}
}

fn response_error(response: Option<&Value>, fallback: &str) -> TurnError {
	let detail = response
		.and_then(|value| {
			value
				.get("error")
				.or_else(|| value.pointer("/status_details/error"))
		})
		.and_then(|error| error.get("message"))
		.and_then(Value::as_str)
		.or_else(|| {
			response
				.and_then(|value| value.pointer("/incomplete_details/reason"))
				.and_then(Value::as_str)
		})
		.unwrap_or(fallback);
	upstream_error(detail)
}

fn upstream_error(detail: &str) -> TurnError {
	TurnError::builder()
		.kind(TurnErrorKind::Upstream)
		.detail(SmolStr::from(detail))
		.maybe_actual(None)
		.unsupported(Vec::new())
		.retry_after_ms(0)
		.build()
}

fn classify_error_body(body: &str) -> TurnError {
	let evidence =
		Evidence { body: Some(body), api: Some(WireApi::OpenAiResponses), ..Evidence::default() };
	let classification = classify(&evidence);
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
		.detail(SmolStr::from(detail))
		.maybe_actual(None)
		.unsupported(Vec::new())
		.retry_after_ms(classification.retry.map_or(0, |hint| hint.delay_ms))
		.build()
}

#[derive(Clone, Copy)]
enum SlotKind {
	Text,
	Thinking,
	Tool,
}

fn append_delta(
	event: &Value,
	state: &mut ResponsesDecodeState,
	out: &mut SmallVec<TurnEvent, 2>,
	expected: SlotKind,
) {
	let Some(index) = event_output_index(event, state) else {
		return;
	};
	if state.ended.contains(&index) {
		return;
	}
	let delta = event
		.get("delta")
		.and_then(Value::as_str)
		.unwrap_or_default();
	match (state.outputs.get_mut(&index), expected) {
		(Some(OutputSlot::Text { text, .. }), SlotKind::Text)
		| (Some(OutputSlot::Thinking { text, .. }), SlotKind::Thinking)
		| (Some(OutputSlot::Tool { arguments: text, .. }), SlotKind::Tool) => {
			text.extend_from_slice(delta.as_bytes());
		},
		_ => return,
	}
	if !delta.is_empty() {
		out.push(TurnEvent::PartDelta { index, chunk: Bytes::copy_from_slice(delta.as_bytes()) });
	}
}

fn complete_slot(item: &Value, slot: Option<&mut OutputSlot>) {
	match slot {
		Some(OutputSlot::Thinking { wire_id, text, encrypted }) => {
			if let Some(id) = item.get("id").and_then(Value::as_str) {
				*wire_id = SmolStr::from(id);
			}
			if let Some(value) = item.get("encrypted_content").and_then(Value::as_str) {
				*encrypted = Bytes::copy_from_slice(value.as_bytes());
			}
			let summaries = item.get("summary").and_then(Value::as_array);
			if let Some(summaries) = summaries {
				text.clear();
				for summary in summaries {
					if let Some(value) = summary.get("text").and_then(Value::as_str) {
						if !text.is_empty() {
							text.extend_from_slice(b"\n\n");
						}
						text.extend_from_slice(value.as_bytes());
					}
				}
			} else if let Some(value) = item
				.get("content")
				.and_then(Value::as_array)
				.and_then(|content| content.first())
				.and_then(|content| content.get("text"))
				.and_then(Value::as_str)
			{
				text.clear();
				text.extend_from_slice(value.as_bytes());
			}
		},
		Some(OutputSlot::Text { wire_id, text }) => {
			if let Some(id) = item.get("id").and_then(Value::as_str) {
				*wire_id = SmolStr::from(id);
			}
			if let Some(content) = item.get("content").and_then(Value::as_array) {
				text.clear();
				for part in content {
					if let Some(value) = part
						.get("text")
						.or_else(|| part.get("refusal"))
						.and_then(Value::as_str)
					{
						text.extend_from_slice(value.as_bytes());
					}
				}
			}
		},
		Some(OutputSlot::Tool { item_id, wire_id, arguments, .. }) => {
			if let Some(id) = item.get("id").and_then(Value::as_str) {
				*item_id = SmolStr::from(id);
			}
			if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
				*wire_id = SmolStr::from(call_id);
			}
			let key = if item.get("type").and_then(Value::as_str) == Some("custom_tool_call") {
				"input"
			} else {
				"arguments"
			};
			if let Some(value) = item.get(key).and_then(Value::as_str) {
				arguments.clear();
				arguments.extend_from_slice(value.as_bytes());
			}
		},
		Some(OutputSlot::Image { data }) => {
			if let Some(value) = item.get("result").and_then(Value::as_str) {
				data.clear();
				data.extend_from_slice(value.as_bytes());
			}
		},
		Some(OutputSlot::Server { item: stored, .. }) => *stored = item.clone(),
		None => {},
	}
}

fn capture_response(response: Option<&Value>, state: &mut ResponsesDecodeState) {
	let Some(response) = response else {
		return;
	};
	if let Some(id) = response.get("id").and_then(Value::as_str) {
		state.response_id = SmolStr::from(id);
	}
	if let Some(model) = response.get("model").and_then(Value::as_str) {
		state.model = SmolStr::from(model);
	}
}

fn build_outcome(
	response: Option<&Value>,
	incomplete: bool,
	state: &mut ResponsesDecodeState,
) -> ChatOutcome {
	let mut output = Vec::with_capacity(state.outputs.len());
	let (outputs, item_options) = (&mut state.outputs, &mut state.item_options);
	for (index, slot) in outputs {
		let kind = match &mut *slot {
			OutputSlot::Text { text, .. } => {
				let text = std::mem::take(text).freeze();
				ItemKind::Message(
					Message::builder()
						.role(Role::Assistant)
						.parts(vec![Part::Text(
							SmolStr::from_utf8_owned(text).expect("JSON text deltas are UTF-8"),
						)])
						.build(),
				)
			},
			OutputSlot::Thinking { text, encrypted, .. } => {
				let text = std::mem::take(text).freeze();
				let redacted = !encrypted.is_empty();
				ItemKind::Message(
					Message::builder()
						.role(Role::Assistant)
						.parts(vec![Part::Thinking(
							Thinking::builder()
								.text(
									SmolStr::from_utf8_owned(text).expect("JSON reasoning deltas are UTF-8"),
								)
								.signature(std::mem::take(encrypted))
								.redacted(redacted)
								.build(),
						)])
						.build(),
				)
			},
			OutputSlot::Tool { id, name, arguments, .. } => ItemKind::ToolCall(
				ToolCall::builder()
					.id(*id)
					.name(std::mem::take(name))
					.args_json(std::mem::take(arguments).freeze())
					.thought_signature(Bytes::new())
					.build(),
			),
			OutputSlot::Image { data } => {
				let encoded = std::mem::take(data).freeze();
				let decoded = decode_base64(&encoded).unwrap_or_else(|| encoded.to_vec());
				let inline = Bytes::from(decoded);
				let size = inline.len() as u64;
				ItemKind::Message(
					Message::builder()
						.role(Role::Assistant)
						.parts(vec![Part::Blob(
							BlobPart::builder()
								.hash([0; 32])
								.mime(SmolStr::from(image_mime(&inline)))
								.size(size)
								.inline(inline)
								.build(),
						)])
						.build(),
				)
			},
			OutputSlot::Server { .. } => ItemKind::Message(
				Message::builder()
					.role(Role::Assistant)
					.parts(Vec::new())
					.build(),
			),
		};
		let mut item_props = item_options.remove(index).unwrap_or_default();
		if let OutputSlot::Text { wire_id, .. } | OutputSlot::Thinking { wire_id, .. } = slot
			&& !wire_id.is_empty()
		{
			item_props.insert_ns("openai", "item_id", Value::String(wire_id.to_string()));
		}
		if let OutputSlot::Tool { item_id, wire_id, custom, .. } = slot {
			if !item_id.is_empty() {
				item_props.insert_ns("openai", "item_id", Value::String(item_id.to_string()));
			}
			if !wire_id.is_empty() {
				item_props.insert_ns("openai", "call_id", Value::String(wire_id.to_string()));
			}
			if *custom {
				item_props.insert_ns("openai", "custom_tool", Value::Bool(true));
			}
		}
		if let OutputSlot::Image { .. } = slot {
			item_props.insert_ns("openai", "image_generation", Value::Bool(true));
		}
		if let OutputSlot::Server { item, events } = slot {
			item_props.insert_ns("openai", "server_tool_item", item.clone());
			if !events.is_empty() {
				item_props.insert_ns("openai", "server_tool_events", Value::Array(events.clone()));
			}
		}
		output.push(Item::builder().seq(0).kind(kind).props(item_props).build());
	}
	let mut usage = response
		.and_then(|value| value.get("usage"))
		.map(decode_usage);
	let has_tool = state
		.outputs
		.values()
		.any(|slot| matches!(slot, OutputSlot::Tool { .. }));
	let mut props = Props::default();
	if !state.response_id.is_empty() {
		props.insert_ns("openai", "response_id", Value::String(state.response_id.to_string()));
	}
	if let Some(service_tier) = response
		.and_then(|value| value.get("service_tier"))
		.and_then(Value::as_str)
	{
		props.insert_ns("openai", "service_tier", Value::String(service_tier.into()));
		if let Some(usage) = &mut usage {
			usage
				.detail
				.insert_ns("openai", "service_tier", Value::String(service_tier.into()));
		}
	}
	if let Some(status) = response
		.and_then(|value| value.get("status"))
		.and_then(Value::as_str)
	{
		props.insert_ns("openai", "response_status", Value::String(status.into()));
	}
	let mapped = if incomplete {
		if response
			.and_then(|value| value.get("incomplete_details"))
			.and_then(|details| details.get("reason"))
			.and_then(Value::as_str)
			== Some("content_filter")
		{
			StopReason::ContentFilter
		} else {
			StopReason::MaxTokens
		}
	} else {
		StopReason::EndTurn
	};
	let stop = with_tool_use_precedence(mapped, has_tool);
	ChatOutcome::builder()
		.output(output)
		.stop(stop)
		.maybe_usage(usage)
		.maybe_cost(reported_cost(
			response
				.and_then(|value| value.get("usage"))
				.and_then(|usage| usage.get("cost")),
		))
		.provider(SmolStr::from("openai"))
		.model(std::mem::take(&mut state.model))
		.unsupported(Vec::new())
		.props(props)
		.build()
}

fn decode_usage(value: &Value) -> Usage {
	let input = value
		.get("input_tokens")
		.and_then(Value::as_u64)
		.unwrap_or(0);
	let output = value
		.get("output_tokens")
		.and_then(Value::as_u64)
		.unwrap_or(0);
	let cached = value
		.pointer("/input_tokens_details/cached_tokens")
		.and_then(Value::as_u64)
		.unwrap_or(0);
	let mut detail = Props::default();
	detail.insert_ns("openai", "usage", value.clone());
	for key in ["input_tokens_details", "output_tokens_details", "total_tokens"] {
		if let Some(entry) = value.get(key) {
			detail.insert_ns("openai", key, entry.clone());
		}
	}
	Usage::builder()
		.input_tokens(input)
		.output_tokens(output)
		.cache_read_tokens(cached)
		.cache_write_tokens(0)
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

fn encode_input(
	req: &ChatRequest,
	start: usize,
	mapper: &CallIdMapper,
	policy: &OpenAiModelPolicy,
	unsupported: &mut Vec<Unsupported>,
) -> Vec<Value> {
	let mut input = Vec::new();
	let mut deferred_media = Vec::new();
	let custom_names = custom_tool_names(req);
	let mut replay_calls: BTreeMap<CallId, (SmolStr, bool)> = BTreeMap::new();
	let sent_calls = seed_sent_calls(req, start, mapper, &custom_names, &mut replay_calls);
	for item in req.thread.items.iter().skip(start) {
		if matches!(&item.kind, ItemKind::Message(_)) {
			input.append(&mut deferred_media);
		}
		match &item.kind {
			ItemKind::Message(message) if message.role != Role::System => {
				encode_message(message, &item.props, policy, &mut input, unsupported);
			},
			ItemKind::Message(_) => {},
			ItemKind::ToolCall(call) => {
				let call_id = item
					.props
					.get_ns("openai", "call_id")
					.and_then(Value::as_str)
					.map_or_else(|| mapper.to_wire(&call.id, ToolCallIdProfile::OpenAi), SmolStr::from);
				let custom = is_custom(&item.props) || custom_names.contains(&call.name);
				let mut value = Map::new();
				value.insert(
					"type".into(),
					Value::String(
						if custom {
							"custom_tool_call"
						} else {
							"function_call"
						}
						.into(),
					),
				);
				value.insert("call_id".into(), Value::String(call_id.to_string()));
				value.insert("name".into(), Value::String(call.name.to_string()));
				if let Some(id) = item
					.props
					.get_ns("openai", "item_id")
					.and_then(Value::as_str)
				{
					value.insert("id".into(), Value::String(id.into()));
				}
				let args = if let Ok(args) = std::str::from_utf8(&call.args_json) {
					args.to_owned()
				} else {
					unsupported.push(dropped(
						"thread.tool_call.args_json",
						"Responses requires UTF-8 tool input",
					));
					String::new()
				};
				value.insert(
					if custom {
						"input".into()
					} else {
						"arguments".into()
					},
					Value::String(args),
				);
				apply_item_options(&mut value, &item.props);
				replay_calls.insert(call.id, (call_id, custom));
				input.push(Value::Object(value));
			},
			ItemKind::ToolResult(result) => {
				let replay = replay_calls.get(&result.call_id);
				let custom = replay.map_or_else(|| is_custom(&item.props), |(_, custom)| *custom);
				let call_id = replay
					.map(|(wire, _)| wire.clone())
					.or_else(|| {
						item
							.props
							.get_ns("openai", "call_id")
							.and_then(Value::as_str)
							.map(SmolStr::from)
					})
					.unwrap_or_else(|| mapper.to_wire(&result.call_id, ToolCallIdProfile::OpenAi));
				let output = Value::String(tool_result_text(&result.parts, unsupported));
				let mut value = Map::new();
				value.insert(
					"type".into(),
					Value::String(
						if custom {
							"custom_tool_call_output"
						} else {
							"function_call_output"
						}
						.into(),
					),
				);
				value.insert("call_id".into(), Value::String(call_id.to_string()));
				value.insert("output".into(), output);
				apply_item_options(&mut value, &item.props);
				input.push(Value::Object(value));
				if let Some(media) =
					encode_tool_result_media(&result.parts, &item.props, policy, unsupported)
				{
					deferred_media.push(media);
				}
				if result.is_error {
					unsupported.push(dropped(
						"thread.tool_result.is_error",
						"Responses tool outputs have no portable error flag",
					));
				}
			},
			_ => {
				unsupported.push(dropped("thread.item", "item kind is not representable by Responses"));
			},
		}
		report_unknown_item_props(&item.props, unsupported);
	}
	input.append(&mut deferred_media);
	unsupported.extend(repair_responses_tool_pairs(&mut input, &sent_calls));
	input
}

/// Records the tool calls a stateful continuation already delivered.
///
/// Their outputs stay valid on the wire even though the matching call is not
/// re-sent, so orphan repair must treat them as paired. `replay_calls` is
/// seeded from the same pass so a post-boundary result keeps the wire call id
/// and the custom/function shape its pre-boundary call was sent with.
fn seed_sent_calls(
	req: &ChatRequest,
	start: usize,
	mapper: &CallIdMapper,
	custom_names: &BTreeSet<SmolStr>,
	replay_calls: &mut BTreeMap<CallId, (SmolStr, bool)>,
) -> BTreeSet<SentCall> {
	let mut sent = BTreeSet::new();
	for item in req.thread.items.iter().take(start) {
		let wire_id = item
			.props
			.get_ns("openai", "call_id")
			.and_then(Value::as_str)
			.map(SmolStr::from);
		match &item.kind {
			ItemKind::ToolCall(call) => {
				let call_id =
					wire_id.unwrap_or_else(|| mapper.to_wire(&call.id, ToolCallIdProfile::OpenAi));
				let custom = is_custom(&item.props) || custom_names.contains(&call.name);
				let kind = if custom {
					ToolKind::Custom
				} else {
					ToolKind::Function
				};
				sent.insert((kind, call_id.clone()));
				replay_calls.insert(call.id, (call_id, custom));
			},
			ItemKind::Message(_) => {
				let Some(native) = item.props.get_ns("openai", "server_tool_item") else {
					continue;
				};
				let Some(kind) = native
					.get("type")
					.and_then(Value::as_str)
					.and_then(call_kind_of)
				else {
					continue;
				};
				if let Some(call_id) = native.get("call_id").and_then(Value::as_str) {
					sent.insert((kind, SmolStr::from(call_id)));
				}
			},
			_ => {},
		}
	}
	sent
}

fn encode_message(
	message: &Message,
	props: &Props,
	policy: &OpenAiModelPolicy,
	input: &mut Vec<Value>,
	unsupported: &mut Vec<Unsupported>,
) {
	if let Some(server_item) = props.get_ns("openai", "server_tool_item") {
		if let Some(mut object) = server_item.as_object().cloned() {
			if policy.filter_reasoning_history
				&& object.get("type").and_then(Value::as_str) == Some("reasoning")
			{
				return;
			}
			demote_computer_item(&mut object, policy, unsupported);
			apply_item_options(&mut object, props);
			input.push(Value::Object(object));
		} else {
			unsupported.push(dropped(
				"thread.message.props:openai/server_tool_item",
				"server tool history must be a JSON object",
			));
		}
		return;
	}
	if !policy.filter_reasoning_history {
		for part in &message.parts {
			if let Part::Thinking(thinking) = part {
				let mut reasoning = Map::new();
				reasoning.insert("type".into(), Value::String("reasoning".into()));
				reasoning.insert(
					"summary".into(),
					if thinking.text.is_empty() {
						Value::Array(Vec::new())
					} else {
						json!([{ "type": "summary_text", "text": thinking.text }])
					},
				);
				if !thinking.signature.is_empty() {
					match std::str::from_utf8(&thinking.signature) {
						Ok(encrypted) => {
							reasoning
								.insert("encrypted_content".into(), Value::String(encrypted.to_owned()));
						},
						Err(_) => unsupported.push(dropped(
							"thread.message.thinking.signature",
							"Responses encrypted content must be UTF-8",
						)),
					}
				}
				if let Some(id) = props.get_ns("openai", "item_id").and_then(Value::as_str) {
					reasoning.insert("id".into(), Value::String(id.into()));
				}
				apply_item_options(&mut reasoning, props);
				input.push(Value::Object(reasoning));
			}
		}
	}
	let mut content = Vec::new();
	for part in &message.parts {
		match part {
			Part::Text(text) => content.push(json!({
				"type": if message.role == Role::Assistant { "output_text" } else { "input_text" },
				"text": text,
			})),
			Part::Blob(blob) => {
				if let Some(value) =
					encode_blob(blob, props, policy, unsupported, "thread.message.blob")
				{
					content.push(value);
				}
			},
			Part::Thinking(_) => {},
			_ => unsupported
				.push(dropped("thread.message.part", "content part is not representable by Responses")),
		}
	}
	if !content.is_empty() {
		let role = match message.role {
			Role::User => "user",
			Role::Assistant => "assistant",
			Role::System => "system",
			_ => "user",
		};
		let mut value = Map::new();
		value.insert("role".into(), Value::String(role.into()));
		value.insert("content".into(), Value::Array(content));
		if message.role == Role::Assistant {
			value.insert("type".into(), Value::String("message".into()));
			value.insert("status".into(), Value::String("completed".into()));
		}
		if let Some(id) = props.get_ns("openai", "item_id").and_then(Value::as_str) {
			value.insert("id".into(), Value::String(id.into()));
		}
		apply_item_options(&mut value, props);
		input.push(Value::Object(value));
	}
}

fn tool_result_text(parts: &[Part], unsupported: &mut Vec<Unsupported>) -> String {
	let mut text = String::new();
	let mut has_media = false;
	let mut only_images = true;
	for part in parts {
		match part {
			Part::Text(value) => {
				if !text.is_empty() {
					text.push('\n');
				}
				text.push_str(value);
			},
			Part::Blob(blob) => {
				has_media = true;
				only_images &= blob.mime.starts_with("image/");
			},
			_ => unsupported.push(dropped(
				"thread.tool_result.part",
				"tool output part is not representable by Responses",
			)),
		}
	}
	if text.is_empty() && has_media {
		text.push_str(if only_images {
			"(see attached image)"
		} else {
			"(see attached media)"
		});
	}
	text
}

fn encode_tool_result_media(
	parts: &[Part],
	props: &Props,
	policy: &OpenAiModelPolicy,
	unsupported: &mut Vec<Unsupported>,
) -> Option<Value> {
	let only_images = parts
		.iter()
		.filter_map(|part| match part {
			Part::Blob(blob) => Some(blob.mime.starts_with("image/")),
			_ => None,
		})
		.all(|is_image| is_image);
	let mut content = vec![json!({
		"type": "input_text",
		"text": if only_images {
			"Attached image(s) from tool result:"
		} else {
			"Attached media from tool result:"
		},
	})];
	for part in parts {
		if let Part::Blob(blob) = part
			&& let Some(value) =
				encode_blob(blob, props, policy, unsupported, "thread.tool_result.blob")
		{
			content.push(value);
		}
	}
	if content.len() == 1 {
		return None;
	}
	let mut message = Map::new();
	message.insert("role".into(), Value::String("user".into()));
	message.insert("content".into(), Value::Array(content));
	apply_item_options(&mut message, props);
	Some(Value::Object(message))
}

fn encode_blob(
	blob: &BlobPart,
	props: &Props,
	policy: &OpenAiModelPolicy,
	unsupported: &mut Vec<Unsupported>,
	path: &str,
) -> Option<Value> {
	let is_image = blob.mime.starts_with("image/");
	if is_image {
		let mut image = Map::new();
		image.insert("type".into(), Value::String("input_image".into()));
		if let Some(file_id) = props.get_ns("openai", "file_id").and_then(Value::as_str) {
			image.insert("file_id".into(), Value::String(file_id.into()));
		} else if let Some(url) = props.get_ns("openai", "image_url").and_then(Value::as_str) {
			image.insert("image_url".into(), Value::String(url.into()));
		} else if !blob.inline.is_empty() {
			image.insert(
				"image_url".into(),
				Value::String(format!("data:{};base64,{}", blob.mime, encode_base64(&blob.inline))),
			);
		} else {
			unsupported.push(dropped(path, "image bytes or an OpenAI image reference are required"));
			return None;
		}
		image.insert("detail".into(), Value::String("auto".into()));
		if let Some(detail) = props.get_ns("openai", "image_detail") {
			if detail.as_str().is_some_and(|value| {
				matches!(value, "auto" | "low" | "high")
					|| (value == "original" && policy.supports_image_detail_original)
			}) {
				image.insert("detail".into(), detail.clone());
			} else if detail.as_str() == Some("original") {
				image.insert("detail".into(), Value::String("auto".into()));
				unsupported.push(clamped(
					&format!("{path}.detail"),
					"model policy does not support original image detail; using auto",
				));
			} else {
				unsupported.push(dropped(
					&format!("{path}.detail"),
					"image detail must be auto, low, high, or supported original",
				));
			}
		}
		return Some(Value::Object(image));
	}

	if blob.mime.starts_with("audio/") || blob.mime.starts_with("video/") {
		unsupported.push(dropped(path, "Responses input accepts image and document blobs only"));
		return None;
	}

	let mut file = Map::new();
	file.insert("type".into(), Value::String("input_file".into()));
	if let Some(file_id) = props.get_ns("openai", "file_id").and_then(Value::as_str) {
		file.insert("file_id".into(), Value::String(file_id.into()));
	} else if let Some(url) = props.get_ns("openai", "file_url").and_then(Value::as_str) {
		file.insert("file_url".into(), Value::String(url.into()));
	} else if !blob.inline.is_empty() {
		file.insert(
			"file_data".into(),
			Value::String(format!("data:{};base64,{}", blob.mime, encode_base64(&blob.inline))),
		);
	} else {
		unsupported.push(dropped(path, "file bytes or an OpenAI image reference are required"));
		return None;
	}
	if let Some(filename) = props.get_ns("openai", "filename").and_then(Value::as_str) {
		file.insert("filename".into(), Value::String(filename.into()));
	}
	Some(Value::Object(file))
}

fn demote_computer_item(
	object: &mut Map<String, Value>,
	policy: &OpenAiModelPolicy,
	unsupported: &mut Vec<Unsupported>,
) {
	if policy.supports_computer_use != Some(false) {
		return;
	}
	match object.get("type").and_then(Value::as_str) {
		Some("computer_call") => {
			let arguments = object
				.remove("actions")
				.or_else(|| object.remove("action"))
				.map_or_else(|| "{}".to_owned(), |value| json!({"actions":value}).to_string());
			object.remove("pending_safety_checks");
			object.remove("status");
			object.insert("type".into(), Value::String("function_call".into()));
			object.insert("name".into(), Value::String("computer".into()));
			object.insert("arguments".into(), Value::String(arguments));
			unsupported.push(emulated(
				"thread.message.props:openai/server_tool_item",
				"native computer history was projected as a function call",
			));
		},
		Some("computer_call_output") => {
			let output = object.remove("output").unwrap_or(Value::Null).to_string();
			object.remove("acknowledged_safety_checks");
			object.remove("status");
			object.insert("type".into(), Value::String("function_call_output".into()));
			object.insert("output".into(), Value::String(output));
			unsupported.push(emulated(
				"thread.message.props:openai/server_tool_item",
				"native computer output was projected as a function result",
			));
		},
		_ => {},
	}
}

fn apply_item_options(object: &mut Map<String, Value>, props: &Props) {
	for key in ["cache_control", "metadata", "prompt_cache_breakpoint"] {
		if let Some(value) = props.get_ns("openai", key)
			&& value.is_object()
		{
			object.insert(key.into(), value.clone());
		}
	}
}

fn report_unknown_item_props(props: &Props, unsupported: &mut Vec<Unsupported>) {
	for (key, value) in &props.0 {
		let known = matches!(
			key.as_str(),
			"openai/call_id"
				| "openai/item_id"
				| "openai/custom_tool"
				| "openai/type"
				| "openai/server_tool_item"
				| "openai/server_tool_events"
				| "openai/image_generation"
				| "openai/cache_control"
				| "openai/metadata"
				| "openai/prompt_cache_breakpoint"
				| "openai/file_id"
				| "openai/file_url"
				| "openai/image_url"
				| "openai/image_detail"
				| "openai/filename"
		);
		let valid = match key.as_str() {
			"openai/cache_control" | "openai/metadata" | "openai/prompt_cache_breakpoint" => {
				value.is_object()
			},
			"openai/call_id"
			| "openai/item_id"
			| "openai/type"
			| "openai/file_id"
			| "openai/file_url"
			| "openai/image_url"
			| "openai/image_detail"
			| "openai/filename" => value.is_string(),
			"openai/custom_tool" | "openai/image_generation" => value.is_boolean(),
			"openai/server_tool_item" => value.is_object(),
			"openai/server_tool_events" => value.is_array(),
			_ => false,
		};
		if known && valid {
			continue;
		}
		unsupported.push(dropped(
			&format!("thread.item.props:{key}"),
			if known {
				"item property has the wrong JSON type"
			} else {
				"item property is not implemented by the Responses codec"
			},
		));
	}
}

fn system_instructions(req: &ChatRequest, unsupported: &mut Vec<Unsupported>) -> String {
	let mut chunks = Vec::new();
	for item in &req.thread.items {
		if let ItemKind::Message(message) = &item.kind
			&& message.role == Role::System
		{
			let text = parts_as_text(&message.parts, "thread.system", unsupported);
			if !text.is_empty() {
				chunks.push(text);
			}
		}
	}
	chunks.join("\n\n")
}

fn parts_as_text(parts: &[Part], path: &str, unsupported: &mut Vec<Unsupported>) -> String {
	let mut text = String::new();
	for part in parts {
		match part {
			Part::Text(value) => text.push_str(value),
			_ => unsupported
				.push(dropped(path, "non-text content cannot be flattened without changing semantics")),
		}
	}
	text
}

fn encode_tools(
	req: &ChatRequest,
	policy: &OpenAiModelPolicy,
	body: &mut Map<String, Value>,
	unsupported: &mut Vec<Unsupported>,
	fail_closed: &mut bool,
) {
	let compat = &policy.compat;
	let custom_names = custom_tool_names(req);
	let mut tools = Vec::new();
	for tool in &req.tools {
		if custom_names.contains(&tool.name) {
			tools.push(json!({
				"type": "custom",
				"name": tool.name,
				"description": tool.description,
			}));
			continue;
		}
		let mut parameters = match serde_json::from_slice(&tool.schema_json) {
			Ok(value) => value,
			Err(_) => {
				unsupported.push(dropped("tools.schema_json", "tool schema is not valid JSON"));
				Value::Object(Map::new())
			},
		};
		let (normalized, reports) = normalize_schema(compat.tool_schema_flavor, &parameters);
		parameters = normalized;
		unsupported.extend(reports);
		let requested_strict = match compat.tool_strict_mode {
			ToolStrictMode::AllStrict => {
				if tool.strict != Some(true) {
					unsupported.push(
						Unsupported::builder()
							.what(SmolStr::from("tools.strict"))
							.detail(SmolStr::from("endpoint requires every tool to be strict"))
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
			let (normalized, reports) = openai_strict(&parameters);
			if reports.is_empty() {
				parameters = normalized;
			} else {
				emitted_strict = Some(false);
				unsupported.extend(reports);
			}
		}
		let mut value = Map::new();
		value.insert("type".into(), Value::String("function".into()));
		value.insert("name".into(), Value::String(tool.name.to_string()));
		value.insert("description".into(), Value::String(tool.description.to_string()));
		value.insert("parameters".into(), parameters);
		if let Some(strict) = emitted_strict {
			value.insert("strict".into(), Value::Bool(strict));
		}
		tools.push(Value::Object(value));
	}
	if let Some(hosted) = req
		.provider_options
		.as_ref()
		.and_then(|props| props.get_ns("openai", "hosted_tools"))
	{
		if let Some(values) = hosted.as_array() {
			for value in values {
				let supported = value
					.get("type")
					.and_then(Value::as_str)
					.is_some_and(|kind| {
						matches!(
							kind,
							"web_search"
								| "web_search_preview"
								| "file_search" | "code_interpreter"
								| "computer" | "computer_use_preview"
								| "image_generation"
								| "mcp"
						)
					});
				if supported {
					let kind = value.get("type").and_then(Value::as_str);
					if matches!(kind, Some("computer" | "computer_use_preview"))
						&& policy.supports_computer_use == Some(false)
					{
						tools.push(json!({
							"type":"function",
							"name":"computer",
							"description":"Control the host computer",
							"parameters":{"type":"object","properties":{}}
						}));
						unsupported.push(emulated(
							"provider_options:openai/hosted_tools",
							"native computer use is unavailable; advertised the function fallback",
						));
					} else {
						tools.push(value.clone());
					}
				} else {
					unsupported.push(dropped(
						"provider_options:openai/hosted_tools",
						"hosted tool must be a supported Responses tool object",
					));
				}
			}
		} else {
			unsupported
				.push(dropped("provider_options:openai/hosted_tools", "hosted tools must be an array"));
		}
	}
	let has_computer_tool = tools
		.iter()
		.any(|tool| tool.get("type").and_then(Value::as_str) == Some("computer"));
	if !tools.is_empty() {
		body.insert("tools".into(), Value::Array(tools));
	}
	if let Some(choice) = &req.tool_choice {
		if !policy.supports_tool_choice {
			report(
				unsupported,
				"tool_choice",
				"model policy marks tool choice unsupported",
				choice.on_unsupported,
				fail_closed,
			);
			return;
		}
		let value = match &choice.value {
			ToolChoice::Auto => json!("auto"),
			ToolChoice::None => json!("none"),
			ToolChoice::Required if compat.forced_tool_choice => json!("required"),
			ToolChoice::Named(name) if compat.forced_tool_choice && compat.named_tool_choice => {
				if name.as_str() == "computer"
					&& policy.supports_computer_use == Some(true)
					&& has_computer_tool
				{
					json!({"type":"computer"})
				} else {
					json!({
						"type": if custom_names.contains(name) { "custom" } else { "function" },
						"name": name,
					})
				}
			},
			_ => {
				report(
					unsupported,
					"tool_choice",
					"provider path cannot force the requested tool choice",
					choice.on_unsupported,
					fail_closed,
				);
				return;
			},
		};
		body.insert("tool_choice".into(), value);
	}
}

fn encode_format(
	req: &ChatRequest,
	compat: &Compat,
	body: &mut Map<String, Value>,
	unsupported: &mut Vec<Unsupported>,
	fail_closed: &mut bool,
) {
	let Some(format) = &req.response_format else {
		return;
	};
	match &format.value.kind {
		ResponseFormatKind::JsonSchema(schema) => {
			let mut parsed = match serde_json::from_slice(&schema.schema_json) {
				Ok(value) => value,
				Err(_) => {
					report(
						unsupported,
						"response_format.json_schema",
						"JSON schema bytes are not valid JSON",
						format.on_unsupported,
						fail_closed,
					);
					return;
				},
			};
			let (normalized, reports) = normalize_schema(compat.tool_schema_flavor, &parsed);
			parsed = normalized;
			unsupported.extend(reports.into_iter().map(|report| {
				Unsupported::builder()
					.what(SmolStr::from("response_format.schema"))
					.detail(report.detail)
					.action(report.action)
					.build()
			}));
			let mut strict = schema.strict;
			if strict == Some(true) {
				let (normalized, reports) = openai_strict(&parsed);
				if reports.is_empty() {
					parsed = normalized;
				} else {
					strict = Some(false);
					unsupported.extend(reports.into_iter().map(|report| {
						Unsupported::builder()
							.what(SmolStr::from("response_format.schema.strict"))
							.detail(report.detail)
							.action(report.action)
							.build()
					}));
				}
			}
			let mut wire_format = Map::new();
			wire_format.insert("type".into(), Value::String("json_schema".into()));
			wire_format.insert("name".into(), Value::String(schema.name.to_string()));
			wire_format.insert("schema".into(), parsed);
			if let Some(strict) = strict {
				wire_format.insert("strict".into(), Value::Bool(strict));
			}
			body.insert("text".into(), json!({ "format": wire_format }));
		},
		_ => report(
			unsupported,
			"response_format.grammar",
			"Responses does not accept the requested grammar",
			format.on_unsupported,
			fail_closed,
		),
	}
}

fn encode_provider_options(
	req: &ChatRequest,
	body: &mut Map<String, Value>,
	include: &mut Vec<Value>,
	unsupported: &mut Vec<Unsupported>,
) {
	let Some(props) = &req.provider_options else {
		return;
	};
	for (key, value) in &props.0 {
		match key.as_str() {
			"openai/verbosity" if value.is_string() => {
				body
					.entry("text")
					.or_insert_with(|| Value::Object(Map::new()))["verbosity"] = value.clone();
			},
			"openai/service_tier" if value.is_string() => {
				body.insert("service_tier".into(), value.clone());
			},
			"openai/metadata" if value.is_object() => {
				body.insert("metadata".into(), value.clone());
			},
			"openai/cache_control" | "openai/prompt_cache_options" if value.is_object() => {
				let field = key.strip_prefix("openai/").expect("matched OpenAI key");
				body.insert(field.into(), value.clone());
			},
			"openai/include" if value.is_array() => {
				for entry in value.as_array().into_iter().flatten() {
					if let Some(entry) = entry.as_str() {
						if is_supported_include(entry)
							&& !include
								.iter()
								.any(|existing| existing.as_str() == Some(entry))
						{
							include.push(Value::String(entry.into()));
						} else if !is_supported_include(entry) {
							unsupported.push(dropped(
								"provider_options:openai/include",
								"include entry is not supported by Responses",
							));
						}
					} else {
						unsupported.push(dropped(
							"provider_options:openai/include",
							"include entries must be strings",
						));
					}
				}
			},
			"openai/parallel_tool_calls" | "openai/store" if value.is_boolean() => {
				let field = key.strip_prefix("openai/").expect("matched OpenAI key");
				body.insert(field.into(), value.clone());
			},
			"openai/prompt_cache_retention" | "openai/truncation" if value.is_string() => {
				let field = key.strip_prefix("openai/").expect("matched OpenAI key");
				body.insert(field.into(), value.clone());
			},
			"openai/reasoning_summary" if value.is_string() || value.is_null() => {
				body
					.entry("reasoning")
					.or_insert_with(|| Value::Object(Map::new()))["summary"] = value.clone();
			},
			"openai/custom_tools" | "openai/hosted_tools" => {},
			"openai/previous_response_id" if value.is_string() => {},
			"openai/previous_response_item_count" if value.as_u64().is_some() => {},
			_ => unsupported.push(dropped(
				&format!("provider_options:{key}"),
				"property is invalid or not implemented by the Responses codec",
			)),
		}
	}
}

fn is_supported_include(value: &str) -> bool {
	matches!(
		value,
		"file_search_call.results"
			| "web_search_call.results"
			| "web_search_call.action.sources"
			| "message.input_image.image_url"
			| "computer_call_output.output.image_url"
			| "code_interpreter_call.outputs"
			| "reasoning.encrypted_content"
			| "message.output_text.logprobs"
	)
}

fn is_custom(props: &Props) -> bool {
	props
		.get_ns("openai", "custom_tool")
		.and_then(Value::as_bool)
		.unwrap_or(false)
		|| props.get_ns("openai", "type").and_then(Value::as_str) == Some("custom_tool_call")
}

fn custom_tool_names(req: &ChatRequest) -> BTreeSet<SmolStr> {
	let mut names: BTreeSet<SmolStr> = req
		.provider_options
		.as_ref()
		.and_then(|props| props.get_ns("openai", "custom_tools"))
		.and_then(Value::as_array)
		.into_iter()
		.flatten()
		.filter_map(Value::as_str)
		.map(SmolStr::from)
		.collect();
	if matches!(
		req.model_policy
			.as_deref()
			.and_then(|policy| policy.apply_patch_shape),
		Some(omp_llm_types::ApplyPatchShape::Freeform)
	) {
		names.insert(SmolStr::new_static("apply_patch"));
	} else if matches!(
		req.model_policy
			.as_deref()
			.and_then(|policy| policy.apply_patch_shape),
		Some(omp_llm_types::ApplyPatchShape::Function)
	) {
		names.remove("apply_patch");
	}
	names
}

fn insert_optional_number(body: &mut Map<String, Value>, key: &str, value: Option<f64>) {
	if let Some(value) = value.and_then(serde_json::Number::from_f64) {
		body.insert(key.into(), Value::Number(value));
	}
}

fn report(
	unsupported: &mut Vec<Unsupported>,
	what: &str,
	detail: &str,
	fallback: Fallback,
	fail_closed: &mut bool,
) {
	if fallback == Fallback::Error {
		*fail_closed = true;
	}
	unsupported.push(
		Unsupported::builder()
			.what(SmolStr::from(what))
			.detail(SmolStr::from(detail))
			.action(match fallback {
				Fallback::Emulate => UnsupportedAction::Emulated,
				_ => UnsupportedAction::Dropped,
			})
			.build(),
	);
}

fn dropped(what: &str, detail: &str) -> Unsupported {
	Unsupported::builder()
		.what(SmolStr::from(what))
		.detail(SmolStr::from(detail))
		.action(UnsupportedAction::Dropped)
		.build()
}

fn clamped(what: &str, detail: &str) -> Unsupported {
	Unsupported::builder()
		.what(SmolStr::from(what))
		.detail(SmolStr::from(detail))
		.action(UnsupportedAction::Clamped)
		.build()
}

fn emulated(what: &str, detail: &str) -> Unsupported {
	Unsupported::builder()
		.what(SmolStr::from(what))
		.detail(SmolStr::from(detail))
		.action(UnsupportedAction::Emulated)
		.build()
}

fn str_field(value: &Value, key: &str) -> SmolStr {
	SmolStr::from(value.get(key).and_then(Value::as_str).unwrap_or_default())
}

fn encode_base64(bytes: &[u8]) -> String {
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

fn decode_base64(encoded: &[u8]) -> Option<Vec<u8>> {
	if !encoded.len().is_multiple_of(4) {
		return None;
	}
	let mut out = Vec::with_capacity(encoded.len() / 4 * 3);
	for (chunk_index, chunk) in encoded.chunks_exact(4).enumerate() {
		let mut values = [0_u8; 4];
		let mut padding = 0;
		for (index, byte) in chunk.iter().copied().enumerate() {
			values[index] = match byte {
				b'A'..=b'Z' => byte - b'A',
				b'a'..=b'z' => byte - b'a' + 26,
				b'0'..=b'9' => byte - b'0' + 52,
				b'+' => 62,
				b'/' => 63,
				b'=' if index >= 2 => {
					padding += 1;
					0
				},
				_ => return None,
			};
		}
		if padding > 0 && chunk_index + 1 != encoded.len() / 4 {
			return None;
		}
		let bits = (u32::from(values[0]) << 18)
			| (u32::from(values[1]) << 12)
			| (u32::from(values[2]) << 6)
			| u32::from(values[3]);
		out.push((bits >> 16) as u8);
		if padding < 2 {
			out.push((bits >> 8) as u8);
		}
		if padding == 0 {
			out.push(bits as u8);
		}
	}
	Some(out)
}

fn image_mime(bytes: &[u8]) -> &'static str {
	if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
		"image/png"
	} else if bytes.starts_with(b"\xff\xd8\xff") {
		"image/jpeg"
	} else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
		"image/gif"
	} else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
		"image/webp"
	} else {
		"application/octet-stream"
	}
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use bytes::Bytes;
	use omp_core::SmolStr;
	use omp_llm_catalog::compat::{Compat, ReasoningWireFormat};
	use omp_llm_transport::{DecodeState, Frame, Transport};
	use omp_llm_types::{
		ApplyPatchShape, BlobPart, CacheHint, CacheRetention, ChatOutcome, ChatRequest, Effort,
		Fallback, Feature, Item, ItemKind, JsonSchema, Message, Part, Props, Reasoning,
		ResolvedModelCapabilities, ResolvedModelPolicy, ResolvedReasoningMode, ResponseFormat,
		ResponseFormatKind, Role, StopReason, Thinking, Thread, ToolCall, ToolDef, ToolResult,
		TurnErrorKind, TurnEvent, UnsupportedAction, ids::CallId,
	};
	use serde_json::{Value, json};

	use super::OpenAiResponsesCodec;

	fn request(items: Vec<Item>) -> ChatRequest {
		ChatRequest::builder()
			.model(SmolStr::from("gpt-5"))
			.thread(Thread::builder().items(items).build())
			.tools(Vec::new())
			.build()
	}

	#[test]
	fn standard_encode_repairs_orphan_tool_output_as_pi_note() {
		let call_id = CallId::new();
		let mut props = Props::default();
		props.insert_ns("openai", "call_id", json!("call_orphan"));
		let result = Item::builder()
			.seq(0)
			.kind(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(call_id)
					.name(SmolStr::new_static("lookup"))
					.parts(vec![Part::Text(SmolStr::new_static("not found"))])
					.is_error(false)
					.build(),
			))
			.props(props)
			.build();
		let (wire, unsupported) = OpenAiResponsesCodec::new()
			.encode(&request(vec![result]), &Compat::default())
			.unwrap();
		let wire: Value = serde_json::from_slice(&wire).unwrap();
		assert_eq!(
			wire["input"],
			json!([{
				"type": "message",
				"role": "assistant",
				"content": "[Orphan tool result; call_id=call_orphan]: not found",
			}]),
		);
		assert_eq!(unsupported.len(), 1);
		assert_eq!(unsupported[0].what, "thread.tool_result");
		assert_eq!(unsupported[0].action, UnsupportedAction::Emulated);
	}

	#[test]
	fn standard_encode_appends_pi_placeholder_to_orphan_tool_call() {
		let call = Item::builder()
			.seq(0)
			.kind(ItemKind::ToolCall(
				ToolCall::builder()
					.id(CallId::new())
					.name(SmolStr::new_static("lookup"))
					.args_json(Bytes::from_static(b"{}"))
					.thought_signature(Bytes::new())
					.build(),
			))
			.props({
				let mut props = Props::default();
				props.insert_ns("openai", "call_id", json!("call_orphan"));
				props
			})
			.build();
		let (wire, unsupported) = OpenAiResponsesCodec::new()
			.encode(&request(vec![call]), &Compat::default())
			.unwrap();
		let wire: Value = serde_json::from_slice(&wire).unwrap();
		assert!(unsupported.is_empty());
		assert_eq!(wire["input"][0]["type"], "function_call");
		assert_eq!(
			wire["input"][1],
			json!({
				"type": "function_call_output",
				"call_id": "call_orphan",
				"output": "[No tool output recorded: the tool call was interrupted before it produced a result.]",
			}),
		);
	}
	fn encoded(req: &ChatRequest) -> Value {
		let mut compat = Compat::default();
		compat.reasoning_wire_format = ReasoningWireFormat::OpenAiResponses;
		serde_json::from_slice(&OpenAiResponsesCodec::new().encode(req, &compat).unwrap().0).unwrap()
	}

	fn policy(compat: Props) -> ResolvedModelPolicy {
		ResolvedModelPolicy { compat, ..ResolvedModelPolicy::default() }
	}

	#[test]
	fn audio_blob_is_reported_instead_of_encoded_as_an_input_file() {
		let audio = Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Blob(
						BlobPart::builder()
							.hash([9; 32])
							.mime(SmolStr::from("audio/wav"))
							.size(4)
							.inline(Bytes::from_static(b"RIFF"))
							.build(),
					)])
					.build(),
			))
			.props(Props::default())
			.build();
		let mut compat = Compat::default();
		compat.reasoning_wire_format = ReasoningWireFormat::OpenAiResponses;
		let (body, unsupported) = OpenAiResponsesCodec::new()
			.encode(&request(vec![audio]), &compat)
			.unwrap();
		let wire = String::from_utf8(body.to_vec()).unwrap();
		assert!(!wire.contains("input_file"), "audio must not ride the file wire: {wire}");
		assert!(!wire.contains("RIFF"));
		assert_eq!(unsupported.len(), 1);
		assert_eq!(unsupported[0].what, "thread.message.blob");
	}

	#[test]
	fn model_policy_projects_pro_reasoning_and_freeform_apply_patch() {
		let user = Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Text(SmolStr::new_static("patch it"))])
					.build(),
			))
			.props(Props::default())
			.build();
		let mut req = request(vec![user]);
		req.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::XHigh).build())
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		req.tools = vec![
			ToolDef::builder()
				.name(SmolStr::new_static("apply_patch"))
				.description(SmolStr::new_static("Apply a patch"))
				.schema_json(Bytes::from_static(b"{\"type\":\"object\",\"properties\":{}}"))
				.build(),
		];
		let mut freeform = policy(Props::default());
		freeform.reasoning_mode = Some(ResolvedReasoningMode::Pro);
		freeform.apply_patch_shape = Some(ApplyPatchShape::Freeform);
		req.model_policy = Some(Arc::new(freeform));
		let freeform = encoded(&req);
		assert_eq!(freeform["reasoning"], json!({"effort":"xhigh","mode":"pro"}));
		assert_eq!(freeform["tools"][0]["type"], "custom");

		let mut function = policy(Props::default());
		function.apply_patch_shape = Some(ApplyPatchShape::Function);
		req.model_policy = Some(Arc::new(function));
		let function = encoded(&req);
		assert_eq!(function["tools"][0]["type"], "function");
		assert_eq!(function["reasoning"], json!({"effort":"xhigh"}));
	}

	#[test]
	fn model_policy_selects_native_or_fallback_computer_tool() {
		let mut req = request(Vec::new());
		let mut options = Props::default();
		options.insert_ns("openai", "hosted_tools", json!([{"type":"computer"}]));
		req.provider_options = Some(options);

		let mut native = policy(Props::default());
		let mut native_capabilities = ResolvedModelCapabilities::default();
		native_capabilities.computer_use = Some(true);
		native.capabilities = native_capabilities;
		req.model_policy = Some(Arc::new(native));
		assert_eq!(encoded(&req)["tools"][0], json!({"type":"computer"}));

		let mut fallback = policy(Props::default());
		let mut fallback_capabilities = ResolvedModelCapabilities::default();
		fallback_capabilities.computer_use = Some(false);
		fallback.capabilities = fallback_capabilities;
		req.model_policy = Some(Arc::new(fallback));
		let fallback = encoded(&req);
		assert_eq!(fallback["tools"][0]["type"], "function");
		assert_eq!(fallback["tools"][0]["name"], "computer");

		let mut history_props = Props::default();
		history_props.insert_ns(
			"openai",
			"server_tool_item",
			json!({
				"type":"computer_call",
				"id":"cu_1",
				"call_id":"call_1",
				"actions":[{"type":"screenshot"}],
				"pending_safety_checks":[],
				"status":"completed"
			}),
		);
		req.thread.items.push(
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::Assistant)
						.parts(Vec::new())
						.build(),
				))
				.props(history_props)
				.build(),
		);
		let mut output_props = Props::default();
		output_props.insert_ns(
			"openai",
			"server_tool_item",
			json!({
				"type":"computer_call_output",
				"call_id":"call_1",
				"output":{"type":"computer_screenshot","image_url":"data:image/png;base64,aW1n"},
				"acknowledged_safety_checks":[],
				"status":"completed"
			}),
		);
		req.thread.items.push(
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::User)
						.parts(Vec::new())
						.build(),
				))
				.props(output_props)
				.build(),
		);
		let mut native_history = policy(Props::default());
		let mut native_capabilities = ResolvedModelCapabilities::default();
		native_capabilities.computer_use = Some(true);
		native_history.capabilities = native_capabilities;
		req.model_policy = Some(Arc::new(native_history));
		assert_eq!(encoded(&req)["input"][0]["type"], "computer_call");

		let mut fallback_history = policy(Props::default());
		let mut fallback_capabilities = ResolvedModelCapabilities::default();
		fallback_capabilities.computer_use = Some(false);
		fallback_history.capabilities = fallback_capabilities;
		req.model_policy = Some(Arc::new(fallback_history));
		let fallback_history = encoded(&req);
		assert_eq!(fallback_history["input"][0]["type"], "function_call");
		assert_eq!(fallback_history["input"][0]["name"], "computer");
	}

	#[test]
	fn model_policy_filters_reasoning_and_controls_encryption_image_detail_and_store() {
		let thinking = Thinking::builder()
			.text(SmolStr::new_static("private"))
			.signature(Bytes::from_static(b"encrypted"))
			.redacted(true)
			.build();
		let mut image_props = Props::default();
		image_props.insert_ns("openai", "image_detail", json!("original"));
		let items = vec![
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::Assistant)
						.parts(vec![Part::Thinking(thinking)])
						.build(),
				))
				.props(Props::default())
				.build(),
			Item::builder()
				.seq(1)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::User)
						.parts(vec![Part::Blob(
							BlobPart::builder()
								.hash([0; 32])
								.mime(SmolStr::new_static("image/png"))
								.size(1)
								.inline(Bytes::from_static(b"x"))
								.build(),
						)])
						.build(),
				))
				.props(image_props)
				.build(),
		];
		let mut req = request(items);
		let mut direct_compat = Props::default();
		direct_compat.insert_ns("wire", "filter_reasoning_history", json!(false));
		direct_compat.insert_ns("wire", "include_encrypted_reasoning", json!(true));
		direct_compat.insert_ns("wire", "supports_image_detail_original", json!(true));
		direct_compat.insert_ns("wire", "supports_store", json!(true));
		req.model_policy = Some(Arc::new(policy(direct_compat)));
		let direct = encoded(&req);
		assert_eq!(direct["include"], json!(["reasoning.encrypted_content"]));
		assert_eq!(direct["input"][0]["type"], "reasoning");
		assert_eq!(direct["input"][1]["content"][0]["detail"], "original");
		assert_eq!(direct["store"], false);

		let mut proxy_compat = Props::default();
		proxy_compat.insert_ns("wire", "filter_reasoning_history", json!(true));
		proxy_compat.insert_ns("wire", "include_encrypted_reasoning", json!(false));
		proxy_compat.insert_ns("wire", "supports_image_detail_original", json!(false));
		proxy_compat.insert_ns("wire", "supports_store", json!(false));
		req.model_policy = Some(Arc::new(policy(proxy_compat)));
		let proxy = encoded(&req);
		assert!(proxy.get("include").is_none());
		assert_eq!(proxy["input"][0]["content"][0]["detail"], "auto");
		assert!(proxy.get("store").is_none());
		assert!(
			proxy["input"]
				.as_array()
				.unwrap()
				.iter()
				.all(|item| item["type"] != "reasoning")
		);
	}

	#[test]
	fn chaining_and_encrypted_reasoning_are_gateway_supplied_and_verbatim() {
		let thinking = Thinking::builder()
			.text(SmolStr::default())
			.signature(Bytes::from_static(b"enc_REDACTED"))
			.redacted(true)
			.build();
		let items = vec![
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::Assistant)
						.parts(vec![Part::Thinking(thinking)])
						.build(),
				))
				.props(Props::default())
				.build(),
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::User)
						.parts(vec![Part::Text(SmolStr::from("Continue."))])
						.build(),
				))
				.props(Props::default())
				.build(),
		];
		let mut compat = Compat::default();
		compat.reasoning_wire_format = ReasoningWireFormat::OpenAiResponses;
		compat.stateful_response_chaining = true;
		let (body, unsupported) =
			OpenAiResponsesCodec::with_previous_response_id("resp_previous_REDACTED")
				.encode(&request(items), &compat)
				.unwrap();
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		let body: Value = serde_json::from_slice(&body).unwrap();
		assert_eq!(body["previous_response_id"], "resp_previous_REDACTED");
		assert_eq!(body["input"][0]["encrypted_content"], "enc_REDACTED");
		assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
	}

	#[test]
	fn chaining_request_matches_recorded_fixture() {
		let fixture: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/openai_responses/request.previous_encrypted.json"
		))
		.unwrap();
		let item = Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Text(SmolStr::from("Continue."))])
					.build(),
			))
			.props(Props::default())
			.build();
		let mut compat = Compat::default();
		compat.reasoning_wire_format = ReasoningWireFormat::OpenAiResponses;
		compat.stateful_response_chaining = true;
		let (body, unsupported) =
			OpenAiResponsesCodec::with_previous_response_id("resp_previous_REDACTED")
				.encode(&request(vec![item]), &compat)
				.unwrap();

		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		let actual: Value = serde_json::from_slice(&body).unwrap();
		assert_eq!(actual, fixture["wire_body"]);
	}

	#[test]
	fn stateful_chaining_stores_the_response_and_accepts_gateway_request_state() {
		let item = Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Text(SmolStr::from("Continue."))])
					.build(),
			))
			.props(Props::default())
			.build();
		let codec = OpenAiResponsesCodec::with_previous_response_id("resp_previous_REDACTED");
		let (without, _) = codec
			.encode(&request(vec![item.clone()]), &Compat::default())
			.unwrap();
		let mut enabled = Compat::default();
		enabled.stateful_response_chaining = true;
		let (with, _) = codec
			.encode(&request(vec![item.clone()]), &enabled)
			.unwrap();
		let without: Value = serde_json::from_slice(&without).unwrap();
		let mut with: Value = serde_json::from_slice(&with).unwrap();
		assert_eq!(with["previous_response_id"], "resp_previous_REDACTED");
		assert_eq!(with["store"], true);
		with.as_object_mut().unwrap().remove("previous_response_id");
		with["store"] = Value::Bool(false);
		assert_eq!(with, without);

		let mut gateway_options = Props::default();
		gateway_options.insert_ns("openai", "previous_response_id", json!("resp_gateway_REDACTED"));
		let mut gateway_request = request(vec![item]);
		gateway_request.provider_options = Some(gateway_options);
		let (gateway_wire, unsupported) = OpenAiResponsesCodec::new()
			.encode(&gateway_request, &enabled)
			.unwrap();
		assert!(unsupported.is_empty());
		let gateway_wire: Value = serde_json::from_slice(&gateway_wire).unwrap();
		assert_eq!(gateway_wire["previous_response_id"], "resp_gateway_REDACTED");
	}

	#[test]
	fn stateful_chaining_encodes_only_items_after_the_committed_boundary() {
		let message = |role, text| {
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(role)
						.parts(vec![Part::Text(SmolStr::from(text))])
						.build(),
				))
				.props(Props::default())
				.build()
		};
		let mut options = Props::default();
		options.insert_ns("openai", "previous_response_id", json!("resp_first"));
		options.insert_ns("openai", "previous_response_item_count", json!(2));
		let mut chained = request(vec![
			message(Role::User, "First question"),
			message(Role::Assistant, "First answer"),
			message(Role::User, "Follow up"),
		]);
		chained.provider_options = Some(options);
		let mut compat = Compat::default();
		compat.stateful_response_chaining = true;

		let (wire, unsupported) = OpenAiResponsesCodec::new()
			.encode(&chained, &compat)
			.unwrap();
		assert!(unsupported.is_empty());
		let wire: Value = serde_json::from_slice(&wire).unwrap();
		assert_eq!(wire["previous_response_id"], "resp_first");
		assert_eq!(
			wire["input"],
			json!([{"role":"user","content":[{"type":"input_text","text":"Follow up"}]}]),
		);
		assert_eq!(chained.thread.items.len(), 3, "encoding retains the full canonical thread");
	}

	#[test]
	fn result_of_a_pre_boundary_custom_call_keeps_its_wire_shape_and_is_not_repaired() {
		let call_id = CallId::new();
		let mut call_props = Props::default();
		call_props.insert_ns("openai", "custom_tool", Value::Bool(true));
		call_props.insert_ns("openai", "call_id", json!("call_shell"));
		let call = Item::builder()
			.seq(0)
			.kind(ItemKind::ToolCall(
				ToolCall::builder()
					.id(call_id)
					.name(SmolStr::from("shell"))
					.args_json(Bytes::from_static(b"ls"))
					.thought_signature(Bytes::new())
					.build(),
			))
			.props(call_props)
			.build();
		let result = Item::builder()
			.seq(0)
			.kind(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(call_id)
					.name(SmolStr::from("shell"))
					.parts(vec![Part::Text(SmolStr::new_static("README.md"))])
					.is_error(false)
					.build(),
			))
			.props(Props::default())
			.build();
		let mut options = Props::default();
		options.insert_ns("openai", "previous_response_id", json!("resp_first"));
		options.insert_ns("openai", "previous_response_item_count", json!(1));
		let mut chained = request(vec![call, result]);
		chained.provider_options = Some(options);
		let mut compat = Compat::default();
		compat.stateful_response_chaining = true;

		let (wire, unsupported) = OpenAiResponsesCodec::new()
			.encode(&chained, &compat)
			.unwrap();
		assert!(unsupported.is_empty(), "{unsupported:?}");
		let wire: Value = serde_json::from_slice(&wire).unwrap();
		assert_eq!(
			wire["input"],
			json!([{
				"type":"custom_tool_call_output",
				"call_id":"call_shell",
				"output":"README.md",
			}]),
		);
	}

	#[test]
	fn custom_tool_input_is_the_argument_string_without_a_wrapper() {
		let mut props = Props::default();
		props.insert_ns("openai", "custom_tool", Value::Bool(true));
		let item = Item::builder()
			.seq(0)
			.kind(ItemKind::ToolCall(
				ToolCall::builder()
					.id(CallId::new())
					.name(SmolStr::from("shell"))
					.args_json(Bytes::from_static(b""))
					.thought_signature(Bytes::new())
					.build(),
			))
			.props(props)
			.build();
		let (body, unsupported) = OpenAiResponsesCodec::new()
			.encode(&request(vec![item]), &Compat::default())
			.unwrap();
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		let body: Value = serde_json::from_slice(&body).unwrap();
		assert_eq!(body["input"][0]["type"], "custom_tool_call");
		assert_eq!(body["input"][0]["input"], "");
		assert!(body["input"][0]["input"].is_string());
	}

	#[test]
	fn encrypted_tool_usage_stream_matches_recorded_fixture() {
		let codec = OpenAiResponsesCodec::new();
		let mut state = DecodeState::default();
		let stream =
			include_str!("../tests/fixtures/openai_responses/stream.encrypted_tool_usage.sse");
		let mut outcome = None;
		for line in stream.lines() {
			let Some(data) = line.strip_prefix("data: ") else {
				continue;
			};
			for event in codec
				.decode(Frame::Data(data.as_bytes()), &mut state)
				.unwrap()
			{
				if let TurnEvent::Outcome(value) = event {
					outcome = Some(value);
				}
			}
		}
		let outcome = outcome.expect("fixture has an authoritative completion");
		assert_eq!(outcome.stop, StopReason::ToolUse);
		let usage = outcome.usage.expect("fixture reports usage");
		assert_eq!(usage.input_tokens, 30);
		assert_eq!(usage.output_tokens, 8);
		assert_eq!(usage.cache_read_tokens, 20);
		let thinking = outcome
			.output
			.iter()
			.find_map(|item| {
				let ItemKind::Message(message) = &item.kind else {
					return None;
				};
				message.parts.iter().find_map(|part| {
					let Part::Thinking(thinking) = part else {
						return None;
					};
					Some(thinking)
				})
			})
			.expect("fixture contains encrypted reasoning");
		assert_eq!(thinking.text, "Inspect first.");
		assert_eq!(thinking.signature, Bytes::from_static(b"enc_REDACTED"));
		assert!(thinking.redacted);
		let reasoning_item = outcome
			.output
			.iter()
			.find(|item| {
				matches!(&item.kind, ItemKind::Message(message) if message.parts.iter().any(|part| matches!(part, Part::Thinking(_))))
			})
			.expect("fixture contains reasoning item");
		assert_eq!(
			reasoning_item
				.props
				.get_ns("openai", "item_id")
				.and_then(Value::as_str),
			Some("rs_REDACTED")
		);
		let (replayed, _) = codec
			.encode(&request(vec![reasoning_item.clone()]), &Compat::default())
			.unwrap();
		let replayed: Value = serde_json::from_slice(&replayed).unwrap();
		assert_eq!(replayed["input"][0]["encrypted_content"], "enc_REDACTED");
		let tool = outcome
			.output
			.iter()
			.find_map(|item| {
				let ItemKind::ToolCall(call) = &item.kind else {
					return None;
				};
				Some(call)
			})
			.expect("fixture contains a tool call");
		assert_eq!(tool.name, "read");
		assert_eq!(tool.args_json, Bytes::from_static(br#"{"path":"README.md"}"#));
	}

	#[test]
	fn classifies_recorded_invalid_parameter_body() {
		let fixture: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/openai_responses/response.error_400.json"
		))
		.unwrap();
		let body = serde_json::to_vec(&fixture["body"]).unwrap();
		let events = OpenAiResponsesCodec::new()
			.decode(Frame::Data(&body), &mut DecodeState::default())
			.unwrap();
		assert!(matches!(
			events.as_slice(),
			[TurnEvent::Error(error)]
				if error.kind == TurnErrorKind::Upstream
					&& error.detail.contains("temperature")
		));
	}

	#[test]
	fn multimodal_metadata_and_provider_controls_match_recorded_fixture() {
		let fixture: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/openai_responses/request.multimodal_options.json"
		))
		.unwrap();
		let mut image_props = Props::default();
		image_props.insert_ns("openai", "image_detail", json!("high"));
		image_props.insert_ns("openai", "cache_control", json!({"type":"ephemeral"}));
		image_props.insert_ns("openai", "metadata", json!({"source":"camera"}));
		let image = Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Blob(
						BlobPart::builder()
							.hash([1; 32])
							.mime(SmolStr::from("image/png"))
							.size(3)
							.inline(Bytes::from_static(b"img"))
							.build(),
					)])
					.build(),
			))
			.props(image_props)
			.build();
		let mut file_props = Props::default();
		file_props.insert_ns("openai", "filename", json!("notes.pdf"));
		let file = Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Blob(
						BlobPart::builder()
							.hash([2; 32])
							.mime(SmolStr::from("application/pdf"))
							.size(3)
							.inline(Bytes::from_static(b"pdf"))
							.build(),
					)])
					.build(),
			))
			.props(file_props)
			.build();
		let mut options = Props::default();
		options.insert_ns("openai", "service_tier", json!("priority"));
		options.insert_ns("openai", "verbosity", json!("low"));
		options.insert_ns(
			"openai",
			"include",
			json!(["web_search_call.results", "code_interpreter_call.outputs"]),
		);
		options.insert_ns("openai", "metadata", json!({"trace":"trace-redacted"}));
		options.insert_ns("openai", "parallel_tool_calls", Value::Bool(true));
		options.insert_ns(
			"openai",
			"hosted_tools",
			json!([
				{"type":"web_search","search_context_size":"low"},
				{"type":"code_interpreter","container":{"type":"auto"}}
			]),
		);
		let mut req = request(vec![image, file]);
		req.cache = Some(
			CacheHint::builder()
				.session_key(SmolStr::from("session-redacted"))
				.retention(CacheRetention::Long)
				.build(),
		);
		req.provider_options = Some(options);
		let (body, unsupported) = OpenAiResponsesCodec::new()
			.encode(&req, &Compat::default())
			.unwrap();
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), fixture["wire_body"],);
	}

	#[test]
	fn server_tool_fixture_preserves_order_usage_and_terminal_identity() {
		let codec = OpenAiResponsesCodec::new();
		let mut state = DecodeState::default();
		let mut terminal_count = 0;
		let mut outcome: Option<ChatOutcome> = None;
		for line in
			include_str!("../tests/fixtures/openai_responses/stream.server_tools_ordering.sse").lines()
		{
			let Some(data) = line.strip_prefix("data: ") else {
				continue;
			};
			for event in codec
				.decode(Frame::Data(data.as_bytes()), &mut state)
				.unwrap()
			{
				if matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)) {
					terminal_count += 1;
				}
				if let TurnEvent::Outcome(value) = event {
					outcome = Some(value);
				}
			}
		}
		let outcome = outcome.expect("fixture has a terminal outcome");
		assert_eq!(terminal_count, 1);
		assert_eq!(outcome.output.len(), 2);
		assert_eq!(
			outcome.output[0]
				.props
				.get_ns("openai", "server_tool_item")
				.and_then(|item| item.get("type"))
				.and_then(Value::as_str),
			Some("web_search_call"),
		);
		assert_eq!(
			outcome.output[0]
				.props
				.get_ns("openai", "server_tool_events")
				.and_then(Value::as_array)
				.map(Vec::len),
			Some(2),
		);
		assert!(matches!(
			&outcome.output[1].kind,
			ItemKind::Message(message)
				if message.parts == vec![Part::Text(SmolStr::from("Found it."))]
		));
		assert_eq!(
			outcome
				.props
				.get_ns("openai", "response_id")
				.and_then(Value::as_str),
			Some("resp_server_REDACTED"),
		);
		let usage = outcome.usage.as_ref().expect("usage is present");
		assert_eq!(usage.cache_read_tokens, 32);
		assert_eq!(
			usage
				.detail
				.get_ns("openai", "output_tokens_details")
				.and_then(|details| details.get("reasoning_tokens"))
				.and_then(Value::as_u64),
			Some(7),
		);
		assert_eq!(
			usage
				.detail
				.get_ns("openai", "service_tier")
				.and_then(Value::as_str),
			Some("priority"),
		);
		let (replayed, unsupported) = codec
			.encode(&request(vec![outcome.output[0].clone()]), &Compat::default())
			.unwrap();
		assert!(unsupported.is_empty());
		let replayed: Value = serde_json::from_slice(&replayed).unwrap();
		assert_eq!(replayed["input"][0]["type"], "web_search_call");
		assert!(
			codec
				.decode(
					Frame::Data(
						br#"{"type":"response.output_text.delta","output_index":1,"delta":"late"}"#
					),
					&mut state,
				)
				.unwrap()
				.is_empty()
		);
		assert!(codec.decode(Frame::Done, &mut state).unwrap().is_empty());
	}

	#[test]
	fn encrypted_reasoning_signature_closes_the_stream_part() {
		let codec = OpenAiResponsesCodec::new();
		let mut state = DecodeState::default();
		codec
			.decode(
				Frame::Data(br#"{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}"#),
				&mut state,
			)
			.unwrap();
		let events = codec
			.decode(
				Frame::Data(br#"{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"summary"}],"encrypted_content":"enc_1"}}"#),
				&mut state,
			)
			.unwrap();
		assert!(matches!(
			events.as_slice(),
			[TurnEvent::PartEnd { index: 0, signature }]
				if signature == &Bytes::from_static(b"enc_1")
		));
	}

	#[test]
	fn malformed_cancelled_and_premature_end_each_emit_one_error() {
		let codec = OpenAiResponsesCodec::new();
		let mut malformed = DecodeState::default();
		let events = codec.decode(Frame::Data(b"{"), &mut malformed).unwrap();
		assert!(
			matches!(events.as_slice(), [TurnEvent::Error(error)] if error.detail.contains("invalid Responses event"))
		);
		assert!(
			codec
				.decode(Frame::Done, &mut malformed)
				.unwrap()
				.is_empty()
		);

		let mut cancelled = DecodeState::default();
		let events = codec
			.decode(
				Frame::Data(br#"{"type":"response.cancelled","response":{"id":"resp_1","status":"cancelled","error":{"message":"caller cancelled"}}}"#),
				&mut cancelled,
			)
			.unwrap();
		assert!(
			matches!(events.as_slice(), [TurnEvent::Error(error)] if error.detail == "caller cancelled")
		);
		assert!(
			codec
				.decode(Frame::Done, &mut cancelled)
				.unwrap()
				.is_empty()
		);

		let mut premature = DecodeState::default();
		let events = codec.decode(Frame::Done, &mut premature).unwrap();
		assert!(matches!(events.as_slice(), [TurnEvent::Error(_)]));
		assert!(
			codec
				.decode(Frame::Done, &mut premature)
				.unwrap()
				.is_empty()
		);
	}

	#[test]
	fn only_authoritative_response_terminals_end_the_turn() {
		let codec = OpenAiResponsesCodec::new();
		let mut state = DecodeState::default();
		for data in [
			br#"{"type":"response.created","response":{"id":"resp_1"}}"#.as_slice(),
			br#"{"type":"response.output_item.done","output_index":0,"item":{"type":"message"}}"#
				.as_slice(),
		] {
			let events = codec.decode(Frame::Data(data), &mut state).unwrap();
			assert!(
				!events
					.iter()
					.any(|event| matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)))
			);
		}
		assert!(
			!codec
				.decode(Frame::Done, &mut DecodeState::default())
				.unwrap()
				.is_empty()
		);
		let events = codec.decode(
			Frame::Data(br#"{"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":30,"output_tokens":8,"input_tokens_details":{"cached_tokens":20}}}}"#),
			&mut state,
		).unwrap();
		assert!(matches!(events.as_slice(), [TurnEvent::Outcome(_)]));
		assert!(codec.decode(Frame::Done, &mut state).unwrap().is_empty());
	}

	#[test]
	fn strict_tools_and_response_formats_normalize_nested_objects_before_advertising_strictness() {
		let item = Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Text(SmolStr::from("nested"))])
					.build(),
			))
			.props(Props::default())
			.build();
		let mut req = request(vec![item]);
		let nested = Bytes::from_static(
			br#"{"type":"object","properties":{"outer":{"type":"object","properties":{"leaf":{"type":"string"}}}}}"#,
		);
		req.tools = vec![
			ToolDef::builder()
				.name(SmolStr::from("lookup"))
				.description(SmolStr::default())
				.schema_json(nested.clone())
				.strict(true)
				.build(),
		];
		req.response_format = Some(
			Feature::builder()
				.value(
					ResponseFormat::builder()
						.kind(ResponseFormatKind::JsonSchema(
							JsonSchema::builder()
								.name(SmolStr::from("nested"))
								.schema_json(nested)
								.strict(true)
								.build(),
						))
						.build(),
				)
				.on_unsupported(Fallback::Error)
				.build(),
		);
		let (wire, unsupported) = OpenAiResponsesCodec::new()
			.encode(&req, &Compat::default())
			.unwrap();
		assert!(unsupported.is_empty());
		let wire: Value = serde_json::from_slice(&wire).unwrap();
		assert_eq!(wire["tools"][0]["strict"], true);
		assert_eq!(wire["text"]["format"]["strict"], true);
		for schema in [&wire["tools"][0]["parameters"], &wire["text"]["format"]["schema"]] {
			assert_eq!(schema["required"], json!(["outer"]));
			assert_eq!(schema["additionalProperties"], false);
			assert_eq!(schema["properties"]["outer"]["anyOf"][0]["required"], json!(["leaf"]));
			assert_eq!(schema["properties"]["outer"]["anyOf"][0]["additionalProperties"], false);
		}
	}

	#[test]
	fn provider_reported_cost_is_authoritative_and_invalid_values_are_ignored() {
		let codec = OpenAiResponsesCodec::new();
		let mut state = DecodeState::default();
		let events = codec
			.decode(
				Frame::Data(
					br#"{"type":"response.completed","response":{"id":"resp_cost","status":"completed","usage":{"input_tokens":1,"output_tokens":1,"cost":0.012345678}}}"#,
				),
				&mut state,
			)
			.unwrap();
		let outcome = events
			.into_iter()
			.find_map(|event| match event {
				TurnEvent::Outcome(outcome) => Some(outcome),
				_ => None,
			})
			.expect("completion produces an outcome");
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
				.and_then(|usage| usage.get("cost"))
				.and_then(Value::as_f64),
			Some(0.012345678)
		);

		for invalid in [json!(-1), json!("0.1"), json!(1e100)] {
			let event = serde_json::to_vec(&json!({
				"type": "response.completed",
				"response": {
					"id": "resp_invalid_cost",
					"status": "completed",
					"usage": {"cost": invalid},
				},
			}))
			.unwrap();
			let events = codec
				.decode(Frame::Data(&event), &mut DecodeState::default())
				.unwrap();
			let outcome = events
				.into_iter()
				.find_map(|event| match event {
					TurnEvent::Outcome(outcome) => Some(outcome),
					_ => None,
				})
				.expect("invalid-cost completion still produces an outcome");
			assert!(outcome.cost.is_none());
		}
	}
}
