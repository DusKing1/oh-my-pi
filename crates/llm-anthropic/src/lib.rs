//! Anthropic Messages request projection and typed-SSE response decoding.

pub mod bedrock;
pub mod compat;
pub mod vertex;
use std::{borrow::Cow, collections::BTreeMap, sync::LazyLock};
mod controls;
mod documents;
mod thinking;
mod tools;

use bytes::{Bytes, BytesMut};
use controls::OutputConfig;
use documents::{MediaKind, MediaSource};
use omp_core::{Str, StrMut};
use omp_llm_catalog::{
	compat::{Compat, ReasoningWireFormat, ThinkingToolChoiceConflict, ToolStrictMode},
	provider::TransportId,
};
use omp_llm_transport::{DecodeState, Frame, Transport};
use omp_llm_types::{
	Accuracy, BlobPart, CacheHint, CacheRetention, CallId, CallIdMapper, ChatOutcome, ChatRequest,
	CountInput, CountRequest, CountResponse, Effort, Error, Fallback, Feature, Item, ItemKind,
	Message, Part, PromptCacheBreakpoint, Props, Reasoning, RequestMeta, ResolvedModelPolicy,
	ResponseFormat, Role, Sampling, StopReason, StreamPartKind, Thinking, Thread, ToolCall,
	ToolCallIdProfile, ToolChoice, ToolDef, TurnError, TurnErrorKind, TurnEvent, Unsupported,
	UnsupportedAction, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, value::RawValue};
use smallvec::SmallVec;
pub use thinking::{AnthropicHeader, CLAUDE_CODE_VERSION, request_headers};
use thinking::{HistoryProjection, WireThinking};
use tools::{ClientTool, ToolNamePolicy, WireTool, WireToolChoice};

/// Codec for Anthropic's typed-SSE `/v1/messages` API.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnthropicCodec {
	tool_names: ToolNamePolicy,
}

impl AnthropicCodec {
	/// Constructs the standard API-key Messages codec.
	#[must_use]
	pub const fn new() -> Self {
		Self { tool_names: ToolNamePolicy::Unchanged }
	}

	/// Constructs a Claude OAuth Messages codec.
	///
	/// The selection is explicit non-secret authentication policy; token bytes
	/// are never inspected to choose the wire mapping.
	#[must_use]
	pub const fn claude_oauth() -> Self {
		Self { tool_names: ToolNamePolicy::ClaudeOauth }
	}

	/// Encodes the request body accepted by `/v1/messages/count_tokens`.
	pub fn encode_count(
		&self,
		request: &CountRequest,
		compat: &Compat,
	) -> Result<(Bytes, Vec<Unsupported>), Error> {
		let CountInput::Thread(thread) = &request.input else {
			return Err(Error::Provider(Str::from(
				"Anthropic token counting requires an inline thread",
			)));
		};
		let none_tool_choice = None;
		let none_sampling = None;
		let none_thinking = None;
		let none_cache = None;
		let none_response_format = None;
		let none_meta = None;
		let no_provider_options = None;
		let view = RequestView {
			model: &request.model,
			thread,
			tools: &request.tools,
			tool_choice: &none_tool_choice,
			sampling: &none_sampling,
			thinking: &none_thinking,
			cache: &none_cache,
			response_format: &none_response_format,
			meta: &none_meta,
			provider_options: &no_provider_options,
			model_policy: None,
		};
		let (body, unsupported) = build_body(&view, compat, false, self.tool_names)?;
		serialize(&CountBody {
			model:    &request.model,
			messages: body.messages,
			system:   body.system,
			tools:    body.tools,
		})
		.map(|bytes| (bytes, unsupported))
	}

	/// Decodes the JSON response returned by `/v1/messages/count_tokens`.
	pub fn decode_count(&self, body: &[u8]) -> Result<CountResponse, Error> {
		#[derive(Deserialize)]
		struct Response {
			input_tokens: u64,
		}
		let response: Response = serde_json::from_slice(body).map_err(json_error)?;
		Ok(CountResponse::builder()
			.tokens(response.input_tokens)
			.accuracy(Accuracy::Exact)
			.build())
	}
}

impl Transport for AnthropicCodec {
	fn id(&self) -> TransportId {
		TransportId::AnthropicMessages
	}

	fn encode(
		&self,
		req: &ChatRequest,
		compat: &Compat,
	) -> Result<(Bytes, Vec<Unsupported>), Error> {
		let view = RequestView::chat(req);
		let (body, unsupported) = build_body(&view, compat, true, self.tool_names)?;
		let body = serialize(&body)?;
		Ok((thinking::patch_billing_attestation(body), unsupported))
	}

	fn decode(
		&self,
		frame: Frame<'_>,
		state: &mut DecodeState,
	) -> Result<SmallVec<TurnEvent, 2>, Error> {
		let state = state.get_or_insert_with(|| AnthropicState::new(self.tool_names));
		if state.completed {
			return Ok(SmallVec::new());
		}
		let data = match frame {
			Frame::Event { data, .. } | Frame::Data(data) => data,
			Frame::Done if state.completed => return Ok(SmallVec::new()),
			Frame::Done => {
				state.completed = true;
				let mut events = SmallVec::new();
				events.push(TurnEvent::Error(
					TurnError::builder()
						.kind(TurnErrorKind::Upstream)
						.detail(Str::from("Anthropic stream ended before message_stop"))
						.unsupported(Vec::new())
						.retry_after_ms(0)
						.build(),
				));
				return Ok(events);
			},
			_ => return Ok(SmallVec::new()),
		};
		if data.is_empty() || data == b"[DONE]" {
			return Ok(SmallVec::new());
		}
		let event = parse_event(data)?;
		decode_event(event, state)
	}
}

#[derive(Clone, Copy, Serialize)]
pub(crate) struct CacheControl {
	pub(crate) r#type: &'static str,
	pub(crate) ttl:    &'static str,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) scope:  Option<&'static str>,
}
#[derive(Serialize)]
pub(crate) struct Body<'a> {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) model:              Option<&'a str>,
	pub(crate) messages:           Vec<WireMessage<'a>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) system:             Option<Vec<Block<'a>>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) tools:              Option<Vec<WireTool<'a>>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) metadata:           Option<Value>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) max_tokens:         Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) thinking:           Option<WireThinking>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) context_management: Option<&'a Value>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) output_config:      Option<OutputConfig<'a>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) tool_choice:        Option<WireToolChoice<'a>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) temperature:        Option<f64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) top_p:              Option<f64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) top_k:              Option<u32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) stop_sequences:     Option<Vec<&'a str>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) service_tier:       Option<&'a str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) speed:              Option<&'static str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) container:          Option<&'a Value>,
	#[serde(skip_serializing_if = "is_false")]
	pub(crate) stream:             bool,
}

#[derive(Serialize)]
struct CountBody<'a> {
	model:    &'a str,
	messages: Vec<WireMessage<'a>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	system:   Option<Vec<Block<'a>>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tools:    Option<Vec<WireTool<'a>>>,
}

#[derive(Serialize)]
struct WireMessage<'a> {
	role:    &'static str,
	content: Vec<Block<'a>>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Block<'a> {
	Text {
		text:          &'a str,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	#[serde(rename = "text")]
	OwnedText {
		text:          String,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	#[serde(rename = "text")]
	FingerprintText {
		text: String,
	},
	Thinking {
		thinking:  &'a str,
		signature: &'a str,
	},
	RedactedThinking {
		data: &'a str,
	},
	Image {
		source:        MediaSource<'a>,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	Document {
		source:        MediaSource<'a>,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	ToolUse {
		id:            Str,
		name:          Cow<'a, str>,
		input:         &'a RawValue,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	ToolResult {
		tool_use_id:   Str,
		#[serde(skip_serializing_if = "Option::is_none")]
		id:            Option<Str>,
		is_error:      bool,
		content:       Vec<Self>,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	ServerToolUse {
		id:    &'a str,
		name:  &'a str,
		#[serde(skip_serializing_if = "Option::is_none")]
		input: Option<&'a Value>,
	},
	WebSearchToolResult {
		tool_use_id: &'a str,
		content:     &'a Value,
	},
	WebFetchToolResult {
		tool_use_id: &'a str,
		content:     &'a Value,
	},
	CodeExecutionToolResult {
		tool_use_id: &'a str,
		content:     &'a Value,
	},
	BashCodeExecutionToolResult {
		tool_use_id: &'a str,
		content:     &'a Value,
	},
	TextEditorCodeExecutionToolResult {
		tool_use_id: &'a str,
		content:     &'a Value,
	},
	Fallback {
		from: &'a Value,
		to:   &'a Value,
	},
}

struct RequestView<'a> {
	model:            &'a Str,
	model_policy:     Option<&'a ResolvedModelPolicy>,
	thread:           &'a Thread,
	tools:            &'a Vec<ToolDef>,
	tool_choice:      &'a Option<Feature<ToolChoice>>,
	sampling:         &'a Option<Sampling>,
	thinking:         &'a Option<Feature<Reasoning>>,
	cache:            &'a Option<CacheHint>,
	response_format:  &'a Option<Feature<ResponseFormat>>,
	meta:             &'a Option<RequestMeta>,
	provider_options: &'a Option<Props>,
}

impl<'a> RequestView<'a> {
	fn chat(request: &'a ChatRequest) -> Self {
		Self {
			model:            &request.model,
			model_policy:     request.model_policy.as_deref(),
			thread:           &request.thread,
			tools:            &request.tools,
			tool_choice:      &request.tool_choice,
			sampling:         &request.sampling,
			thinking:         &request.thinking,
			cache:            &request.cache,
			response_format:  &request.response_format,
			meta:             &request.meta,
			provider_options: &request.provider_options,
		}
	}
}

fn build_body<'a>(
	req: &RequestView<'a>,
	compat: &Compat,
	stream: bool,
	tool_names: ToolNamePolicy,
) -> Result<(Body<'a>, Vec<Unsupported>), Error> {
	let mut unsupported = Vec::new();
	let props = match req.provider_options {
		Some(props) => props,
		None => empty_props(),
	};
	thinking::validate_options(props)?;
	let mut controls = controls::project(req.response_format, req.meta, props, &mut unsupported)?;
	let thinking_enabled = req
		.thinking
		.as_ref()
		.is_some_and(|feature| feature.value.effort != Some(Effort::Off))
		|| thinking::policy_bool(req.model_policy, "requires_thinking_enabled", false) == Some(true);
	let escape_tool_names =
		thinking::policy_bool(req.model_policy, "escape_builtin_tool_names", thinking_enabled)
			== Some(true);
	let require_tool_result_id =
		thinking::policy_bool(req.model_policy, "requires_tool_result_id", thinking_enabled)
			== Some(true);
	let supports_mid_system =
		thinking::policy_bool(req.model_policy, "supports_mid_conversation_system", thinking_enabled)
			== Some(true);
	match (
		controls.eager_input_streaming,
		thinking::policy_bool(
			req.model_policy,
			"supports_eager_tool_input_streaming",
			thinking_enabled,
		),
	) {
		(Some(true), Some(false)) => {
			report(
				&mut unsupported,
				"anthropic/eager_input_streaming",
				"this model does not support eager tool-input streaming",
				UnsupportedAction::Dropped,
			);
			controls.eager_input_streaming = None;
		},
		(None, Some(true)) => controls.eager_input_streaming = Some(true),
		_ => {},
	}
	let mapper = CallIdMapper::new();
	let mut messages: Vec<WireMessage<'a>> = Vec::new();
	let mut system = Vec::new();
	if let Some(prelude) = thinking::claude_code_system_prelude_for(req.thread, req.provider_options)
	{
		system.extend(
			prelude
				.into_iter()
				.map(|text| Block::FingerprintText { text }),
		);
	}
	let mut index = 0;
	let mut saw_conversation = false;
	while index < req.thread.items.len() {
		let item = &req.thread.items[index];
		match &item.kind {
			ItemKind::Message(message) if message.role == Role::System => {
				let mid_conversation = saw_conversation;
				let eligible_mid = mid_conversation
					&& supports_mid_system
					&& messages
						.last()
						.is_some_and(|message| message.role == "user")
					&& req.thread.items.get(index + 1).is_none_or(|item| {
						matches!(
							&item.kind,
							ItemKind::Message(message) if message.role == Role::Assistant
						)
					});
				let mut blocks = Vec::new();
				for part in &message.parts {
					match part {
						Part::Text(text) => {
							blocks.push(Block::Text { text, cache_control: None });
						},
						_ => report(
							&mut unsupported,
							"thread.system.parts",
							"Anthropic system blocks only support text",
							UnsupportedAction::Dropped,
						),
					}
				}
				if eligible_mid {
					append_message(&mut messages, "system", blocks);
				} else {
					if mid_conversation {
						report(
							&mut unsupported,
							"thread.system.position",
							"mid-conversation system content was hoisted to the top-level system prompt",
							UnsupportedAction::Emulated,
						);
					}
					system.extend(blocks);
				}
			},
			ItemKind::Message(message) => {
				let role = if message.role == Role::Assistant {
					"assistant"
				} else {
					"user"
				};
				let blocks = message_blocks(
					message,
					&item.props,
					req.model,
					req.provider_options,
					compat,
					&mut unsupported,
					req.model_policy,
					thinking_enabled,
				)?;
				append_message(&mut messages, role, blocks);
				saw_conversation = true;
			},
			ItemKind::ToolCall(call) => {
				let input = raw(&call.args_json)?;
				let block = Block::ToolUse {
					id: mapper.to_wire(&call.id, ToolCallIdProfile::Anthropic),
					name: tool_names.encode(&call.name, escape_tool_names),
					input,
					cache_control: None,
				};
				append_message(&mut messages, "assistant", vec![block]);
				saw_conversation = true;
			},
			ItemKind::ToolResult(_) => {
				let hoist_error_images = requires_error_image_hoist(req.model);
				let mut results = Vec::new();
				let mut images = Vec::new();
				while index < req.thread.items.len() {
					let result_item = &req.thread.items[index];
					let ItemKind::ToolResult(result) = &result_item.kind else {
						break;
					};
					let mut native_sources = NativeMediaSources::from_props(&result_item.props)?;
					let mut content = Vec::new();
					for part in &result.parts {
						match part {
							Part::Text(text) => content.push(Block::Text { text, cache_control: None }),
							Part::Blob(blob) => match documents::media_kind(blob) {
								Ok((kind, _)) => {
									let native_source = native_sources.next(kind);
									let block = media_block(blob, kind, native_source)?;
									if result.is_error && hoist_error_images && kind == MediaKind::Image {
										images.push(block);
									} else {
										content.push(block);
									}
								},
								Err(detail) => report(
									&mut unsupported,
									"thread.tool_result.blob",
									detail,
									UnsupportedAction::Dropped,
								),
							},
							Part::Thinking(_) => report(
								&mut unsupported,
								"thread.tool_result.thinking",
								"Anthropic tool results cannot contain thinking blocks",
								UnsupportedAction::Dropped,
							),
							_ => {},
						}
					}
					native_sources.finish()?;
					if result.is_error && content.is_empty() {
						content.push(Block::Text {
							text:          "Tool failed with no text output.",
							cache_control: None,
						});
					}
					let tool_use_id = mapper.to_wire(&result.call_id, ToolCallIdProfile::Anthropic);
					results.push(Block::ToolResult {
						id: require_tool_result_id.then(|| tool_use_id.clone()),
						tool_use_id,
						is_error: result.is_error,
						content,
						cache_control: None,
					});
					index += 1;
				}
				if !images.is_empty() {
					results.push(Block::Text {
						text:          "Attached image(s) from the tool result(s) above:",
						cache_control: None,
					});
					results.extend(images);
				}
				append_message(&mut messages, "user", results);
				saw_conversation = true;
				continue;
			},
			_ => {},
		}
		index += 1;
	}

	for message in &mut messages {
		if message.role == "assistant" && reorder_tool_uses(&mut message.content) {
			report(
				&mut unsupported,
				"thread.assistant.tool_use_order",
				"Anthropic requires tool_use blocks at the end of an assistant message",
				UnsupportedAction::Emulated,
			);
		}
	}

	let server_tools = tools::server_tools(props)?;
	let mut tools = if req.tools.is_empty() && server_tools.is_empty() {
		None
	} else {
		let mut projected = Vec::with_capacity(req.tools.len() + server_tools.len());
		for tool in req.tools {
			let strict = match compat.tool_strict_mode {
				ToolStrictMode::AllStrict => Some(true),
				ToolStrictMode::Mixed => tool.strict,
				ToolStrictMode::None => {
					if tool.strict.is_some_and(|strict| strict) {
						report(
							&mut unsupported,
							"tools.strict",
							"this Anthropic host rejects strict tool definitions",
							UnsupportedAction::Dropped,
						);
					}
					None
				},
			};
			projected.push(WireTool::Client(ClientTool {
				name: tool_names.encode(&tool.name, escape_tool_names),
				description: &tool.description,
				input_schema: tools::normalize_schema(&tool.schema_json)?,
				strict,
				eager_input_streaming: controls.eager_input_streaming,
				cache_control: None,
			}));
		}
		projected.extend(server_tools.into_iter().cloned().map(WireTool::Server));
		Some(projected)
	};
	let requires_thinking =
		thinking::policy_bool(req.model_policy, "requires_thinking_enabled", false) == Some(true);

	let thinking_supported = props
		.get_ns("anthropic", "thinking_supported")
		.and_then(Value::as_bool)
		!= Some(false);
	let projection =
		if compat.reasoning_wire_format == ReasoningWireFormat::Anthropic && thinking_supported {
			thinking::reasoning_projection_for(
				req.model,
				req.thinking,
				req.provider_options,
				req.model_policy,
			)
		} else {
			if let Some(feature) = req.thinking {
				feature_report(
					&mut unsupported,
					feature.on_unsupported,
					"thinking",
					"this Anthropic host does not advertise native thinking controls",
				)?;
			} else if requires_thinking {
				report(
					&mut unsupported,
					"thinking",
					"this model requires thinking but the provider disabled native thinking controls",
					UnsupportedAction::Dropped,
				);
			}
			Default::default()
		};
	if let Some(feature) = req.thinking {
		if requires_thinking && feature.value.effort == Some(Effort::Off) {
			feature_report(
				&mut unsupported,
				feature.on_unsupported,
				"thinking.effort",
				"this model requires thinking; its minimum effort was used",
			)?;
		}
		if feature.value.hide_summary.is_some()
			&& req
				.model_policy
				.and_then(|policy| policy.thinking.as_ref())
				.is_some_and(|thinking| thinking.supports_display != Some(true))
		{
			feature_report(
				&mut unsupported,
				feature.on_unsupported,
				"thinking.hide_summary",
				"this model does not support the adaptive thinking display control",
			)?;
		}
	}
	let mut thinking = projection.thinking;
	let mut budget = projection.budget;
	if req.thinking.as_ref().is_some_and(|feature| {
		feature.value.budget_tokens.is_none() && feature.value.effort.is_some()
	}) && projection.budget.is_some()
	{
		report(
			&mut unsupported,
			"thinking.effort",
			"qualitative effort was mapped to an Anthropic token budget",
			UnsupportedAction::Emulated,
		);
	}
	if let Some(effort) = projection.effort {
		controls
			.output_config
			.get_or_insert(OutputConfig { effort: None, task_budget: None, format: None })
			.effort = Some(effort);
	}

	let parallel = controls.disable_parallel_tool_use;
	let mut tool_choice = None;
	let mut forced = false;
	if let Some(feature) = &req.tool_choice {
		match &feature.value {
			ToolChoice::Auto => {
				tool_choice = Some(WireToolChoice::Auto { disable_parallel_tool_use: parallel });
			},
			ToolChoice::None => tool_choice = Some(WireToolChoice::None),
			ToolChoice::Required | ToolChoice::Named(_)
				if compat.forced_tool_choice && !requires_thinking =>
			{
				forced = true;
				tool_choice = Some(match &feature.value {
					ToolChoice::Required => WireToolChoice::Any { disable_parallel_tool_use: parallel },
					ToolChoice::Named(name) => WireToolChoice::Tool {
						name: tool_names.encode(name, escape_tool_names),
						disable_parallel_tool_use: parallel,
					},
					_ => unreachable!(),
				});
			},
			ToolChoice::Required | ToolChoice::Named(_) => {
				feature_report(
					&mut unsupported,
					feature.on_unsupported,
					"tool_choice.forced",
					"this Anthropic host rejects forced tool choice; auto was used",
				)?;
				tool_choice = Some(WireToolChoice::Auto { disable_parallel_tool_use: parallel });
			},
			_ => {},
		}
	} else if parallel.is_some() {
		tool_choice = Some(WireToolChoice::Auto { disable_parallel_tool_use: parallel });
	}

	let conflict = match thinking::policy_bool(
		req.model_policy,
		"disable_reasoning_on_tool_choice",
		thinking.is_some() || projection.effort.is_some(),
	) {
		Some(value) => value && req.tool_choice.is_some(),
		None => match compat.thinking_tool_choice_conflict {
			ThinkingToolChoiceConflict::DropThinkingWhenForced => forced,
			ThinkingToolChoiceConflict::DropThinkingWhenAny => req.tool_choice.is_some(),
			ThinkingToolChoiceConflict::DropThinkingWhenEffort => req
				.thinking
				.as_ref()
				.is_some_and(|feature| feature.value.effort.is_some()),
			ThinkingToolChoiceConflict::None => false,
		},
	};
	if conflict && (thinking.is_some() || projection.effort.is_some()) {
		let fallback = req
			.thinking
			.as_ref()
			.map_or(Fallback::Ignore, |feature| feature.on_unsupported);
		feature_report(
			&mut unsupported,
			fallback,
			"thinking.tool_choice_conflict",
			"thinking was disabled because this host cannot combine it with tool choice",
		)?;
		thinking = None;
		budget = None;
		if projection.effort.is_some()
			&& let Some(output_config) = controls.output_config.as_mut()
		{
			output_config.effort = None;
		}
		if controls
			.output_config
			.as_ref()
			.is_some_and(|output_config| {
				output_config.effort.is_none()
					&& output_config.task_budget.is_none()
					&& output_config.format.is_none()
			}) {
			controls.output_config = None;
		}
	}
	if thinking.is_some()
		&& controls.context_management.is_none()
		&& !thinking::signing_endpoint(
			req.model,
			req.provider_options.as_ref(),
			req.model_policy,
			true,
		) {
		controls.context_management = Some(keep_all_thinking());
	}

	let mut max_tokens = req
		.sampling
		.as_ref()
		.and_then(|sampling| sampling.max_output_tokens);
	if let (Some(tokens), Some(current)) = (budget, max_tokens) {
		let minimum = tokens.saturating_add(1024);
		if current < minimum {
			max_tokens = Some(minimum);
			report(
				&mut unsupported,
				"sampling.max_output_tokens",
				"max_tokens was raised to leave 1024 tokens beyond the thinking budget",
				UnsupportedAction::Clamped,
			);
		}
	}

	let (mut temperature, mut top_p, mut top_k, stop_sequences) =
		if let Some(sampling) = &req.sampling {
			if sampling.min_p.is_some() {
				report(
					&mut unsupported,
					"sampling.min_p",
					"Anthropic does not support min_p",
					UnsupportedAction::Dropped,
				);
			}
			if sampling.frequency_penalty.is_some() {
				report(
					&mut unsupported,
					"sampling.frequency_penalty",
					"Anthropic does not support frequency penalties",
					UnsupportedAction::Dropped,
				);
			}
			if sampling.presence_penalty.is_some() {
				report(
					&mut unsupported,
					"sampling.presence_penalty",
					"Anthropic does not support presence penalties",
					UnsupportedAction::Dropped,
				);
			}
			let stop = sampling.stop.as_ref().map(|sequences| {
				if sequences.len() > 4 {
					report(
						&mut unsupported,
						"sampling.stop",
						"Anthropic accepts at most four stop sequences; extras were removed",
						UnsupportedAction::Clamped,
					);
				}
				sequences.iter().take(4).map(Str::as_str).collect()
			});
			(sampling.temperature, sampling.top_p, sampling.top_k, stop)
		} else {
			(None, None, None, None)
		};
	if matches!(thinking, Some(WireThinking::Adaptive { .. } | WireThinking::Enabled { .. }))
		|| projection.effort.is_some()
	{
		for (requested, what) in [
			(temperature.take().is_some(), "sampling.temperature"),
			(top_p.take().is_some(), "sampling.top_p"),
			(top_k.take().is_some(), "sampling.top_k"),
		] {
			if requested {
				report(
					&mut unsupported,
					what,
					"sampling controls are incompatible with active Anthropic thinking",
					UnsupportedAction::Dropped,
				);
			}
		}
	}

	for key in props.0.keys() {
		if !controls::is_known_option(key) && !thinking::is_known_option(key) {
			report(
				&mut unsupported,
				key,
				"unknown request property was not sent to Anthropic",
				UnsupportedAction::Dropped,
			);
		}
	}

	let supports_long_cache =
		thinking::policy_bool(req.model_policy, "supports_long_cache_retention", thinking.is_some())
			!= Some(false);
	let mut option_cache_control = controls.cache_control;
	let requested_long_cache = req
		.cache
		.as_ref()
		.is_some_and(|cache| cache.retention == Some(CacheRetention::Long))
		|| option_cache_control.is_some_and(|cache| cache.ttl == "1h");
	if requested_long_cache && !supports_long_cache {
		report(
			&mut unsupported,
			"cache.retention",
			"this model does not support one-hour cache retention; five minutes was used",
			UnsupportedAction::Emulated,
		);
		if let Some(cache) = option_cache_control.as_mut() {
			cache.ttl = "5m";
		}
	}
	let cache_control = if let Some(cache) = &req.cache {
		Some(CacheControl {
			r#type: "ephemeral",
			ttl:    match cache.retention {
				Some(CacheRetention::Long) if supports_long_cache => "1h",
				Some(CacheRetention::Long | CacheRetention::Short) | None => "5m",
				_ => "5m",
			},
			scope:  option_cache_control.and_then(|value| value.scope),
		})
	} else {
		option_cache_control
	};
	let breakpoint = req.cache.as_ref().and_then(|cache| cache.breakpoint);
	if let Some(cache_control) = cache_control
		&& breakpoint != Some(PromptCacheBreakpoint::None)
	{
		if breakpoint == Some(PromptCacheBreakpoint::TailTwo) {
			// Mark the tail of each of the final two messages and nothing else.
			// The deeper marker already caches system and tools transitively,
			// so dedicated markers there only spend slots. Both markers share
			// `cache_control`'s TTL: a longer-lived marker underneath a shorter
			// one cannot be founded on it, so the rolling anchor would re-buy
			// every short-lived region it overtakes.
			let mut placed = 0usize;
			for message in messages.iter_mut().rev() {
				let Some(block) = message
					.content
					.iter_mut()
					.rev()
					.find(|block| block.cacheable())
				else {
					continue;
				};
				block.set_cache(cache_control);
				placed += 1;
				if placed == 2 {
					break;
				}
			}
		} else {
			if let Some(block) = system.iter_mut().rev().find(|block| block.cacheable()) {
				block.set_cache(cache_control);
			}
			if let Some(tools) = tools.as_mut() {
				let _ = tools
					.iter_mut()
					.rev()
					.any(|tool| tool.set_cache(cache_control));
			}
			if let Some(block) = messages
				.iter_mut()
				.rev()
				.flat_map(|message| message.content.iter_mut().rev())
				.find(|block| block.cacheable())
			{
				block.set_cache(cache_control);
			}
		}
	}

	let metadata = controls.metadata;
	Ok((
		Body {
			model: (!req.model.is_empty()).then_some(req.model.as_str()),
			messages,
			system: (!system.is_empty()).then_some(system),
			tools,
			metadata,
			max_tokens,
			thinking,
			context_management: controls.context_management,
			output_config: controls.output_config,
			tool_choice,
			temperature,
			top_p,
			top_k,
			stop_sequences,
			service_tier: controls.service_tier,
			speed: controls.speed,
			container: controls.container,
			stream: stream && !req.model.is_empty(),
		},
		unsupported,
	))
}

fn requires_error_image_hoist(model: &str) -> bool {
	let Some((_, version)) = model.rsplit_once("-4-6") else {
		return false;
	};
	version.is_empty()
		|| version.strip_prefix('-').is_some_and(|suffix| {
			!suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
		})
}

impl Block<'_> {
	const fn cacheable(&self) -> bool {
		matches!(
			self,
			Self::Text { .. }
				| Self::OwnedText { .. }
				| Self::Image { .. }
				| Self::Document { .. }
				| Self::ToolUse { .. }
				| Self::ToolResult { .. }
		)
	}

	const fn set_cache(&mut self, value: CacheControl) {
		match self {
			Self::Text { cache_control, .. }
			| Self::OwnedText { cache_control, .. }
			| Self::Image { cache_control, .. }
			| Self::Document { cache_control, .. }
			| Self::ToolUse { cache_control, .. }
			| Self::ToolResult { cache_control, .. } => *cache_control = Some(value),
			_ => {},
		}
	}
}
struct NativeMediaSources<'a> {
	images:         Option<&'a [Value]>,
	documents:      Option<&'a [Value]>,
	image_index:    usize,
	document_index: usize,
}

impl<'a> NativeMediaSources<'a> {
	fn from_props(props: &'a Props) -> Result<Self, Error> {
		let array = |name: &'static str| {
			props
				.get_ns("anthropic", name)
				.map(|value| {
					value.as_array().map(Vec::as_slice).ok_or_else(|| {
						Error::Provider(format!("anthropic/{name} must be an array").into())
					})
				})
				.transpose()
		};
		Ok(Self {
			images:         array("image_sources")?,
			documents:      array("document_sources")?,
			image_index:    0,
			document_index: 0,
		})
	}

	fn next(&mut self, kind: MediaKind) -> Option<&'a Value> {
		let (sources, index) = match kind {
			MediaKind::Image => (self.images, &mut self.image_index),
			MediaKind::Document => (self.documents, &mut self.document_index),
		};
		let source = sources.and_then(|values| values.get(*index));
		*index += 1;
		source
	}

	fn finish(self) -> Result<(), Error> {
		for (name, sources, used) in [
			("image_sources", self.images, self.image_index),
			("document_sources", self.documents, self.document_index),
		] {
			if sources.is_some_and(|values| values.len() > used) {
				return Err(Error::Provider(
					format!("anthropic/{name} has more entries than matching canonical Blob parts")
						.into(),
				));
			}
		}
		Ok(())
	}
}

fn message_blocks<'a>(
	message: &'a Message,
	item_props: &'a Props,
	target_model: &str,
	provider_options: &Option<Props>,
	compat: &Compat,
	unsupported: &mut Vec<Unsupported>,
	model_policy: Option<&ResolvedModelPolicy>,
	thinking_enabled: bool,
) -> Result<Vec<Block<'a>>, Error> {
	if let Some(block) = server_history_block(item_props)? {
		return Ok(vec![block]);
	}
	let mut blocks = Vec::with_capacity(message.parts.len());
	let mut native_sources = NativeMediaSources::from_props(item_props)?;
	for part in &message.parts {
		match part {
			Part::Text(text) => blocks.push(Block::Text { text, cache_control: None }),
			Part::Blob(blob) => match documents::media_kind(blob) {
				Ok((kind, _)) => {
					let native_source = native_sources.next(kind);
					blocks.push(media_block(blob, kind, native_source)?);
				},
				Err(detail) => {
					report(unsupported, "thread.message.blob", detail, UnsupportedAction::Dropped);
				},
			},
			Part::Thinking(value) if message.role == Role::Assistant => {
				match thinking::project_history(
					value,
					item_props,
					target_model,
					provider_options,
					model_policy,
					thinking_enabled,
					compat,
				) {
					HistoryProjection::Native { text, signature } => {
						blocks.push(Block::Thinking { thinking: text, signature });
					},
					HistoryProjection::Redacted { data } => {
						blocks.push(Block::RedactedThinking { data });
					},
					HistoryProjection::Demoted(text) => {
						blocks.push(Block::OwnedText {
							text:          text.into_owned(),
							cache_control: None,
						});
						report(
							unsupported,
							"thread.assistant.thinking",
							"unsigned or foreign thinking was replayed as visible text",
							UnsupportedAction::Emulated,
						);
					},
					HistoryProjection::Drop => report(
						unsupported,
						"thread.assistant.thinking",
						"incompatible opaque thinking could not be replayed",
						UnsupportedAction::Dropped,
					),
				}
			},
			Part::Thinking(_) => report(
				unsupported,
				"thread.user.thinking",
				"thinking history is only valid in assistant messages",
				UnsupportedAction::Dropped,
			),
			_ => {},
		}
	}
	native_sources.finish()?;
	Ok(blocks)
}

fn server_history_block(props: &Props) -> Result<Option<Block<'_>>, Error> {
	let Some(value) = props.get_ns("anthropic", "server_tool_block") else {
		return Ok(None);
	};
	let object = value
		.as_object()
		.ok_or_else(|| Error::Provider("anthropic/server_tool_block must be an object".into()))?;
	let kind = object.get("type").and_then(Value::as_str).ok_or_else(|| {
		Error::Provider("anthropic/server_tool_block requires a string type".into())
	})?;
	let string = |name: &'static str| {
		object.get(name).and_then(Value::as_str).ok_or_else(|| {
			Error::Provider(format!("anthropic/server_tool_block requires string field {name}").into())
		})
	};
	let required = |name: &'static str| {
		object.get(name).ok_or_else(|| {
			Error::Provider(format!("anthropic/server_tool_block requires field {name}").into())
		})
	};
	let block = match kind {
		"server_tool_use" => Block::ServerToolUse {
			id:    string("id")?,
			name:  string("name")?,
			input: object.get("input"),
		},
		"web_search_tool_result" => Block::WebSearchToolResult {
			tool_use_id: string("tool_use_id")?,
			content:     required("content")?,
		},
		"web_fetch_tool_result" => Block::WebFetchToolResult {
			tool_use_id: string("tool_use_id")?,
			content:     required("content")?,
		},
		"code_execution_tool_result" => Block::CodeExecutionToolResult {
			tool_use_id: string("tool_use_id")?,
			content:     required("content")?,
		},
		"bash_code_execution_tool_result" => Block::BashCodeExecutionToolResult {
			tool_use_id: string("tool_use_id")?,
			content:     required("content")?,
		},
		"text_editor_code_execution_tool_result" => Block::TextEditorCodeExecutionToolResult {
			tool_use_id: string("tool_use_id")?,
			content:     required("content")?,
		},
		"fallback" => Block::Fallback { from: required("from")?, to: required("to")? },
		_ => {
			return Err(Error::Provider(
				format!("unsupported anthropic/server_tool_block type `{kind}`").into(),
			));
		},
	};
	Ok(Some(block))
}

fn media_block<'a>(
	blob: &'a BlobPart,
	kind: MediaKind,
	native_source: Option<&'a Value>,
) -> Result<Block<'a>, Error> {
	let source = media_source(blob, native_source)?;
	Ok(match kind {
		MediaKind::Image => Block::Image { source, cache_control: None },
		MediaKind::Document => Block::Document { source, cache_control: None },
	})
}

fn media_source<'a>(
	blob: &'a BlobPart,
	native: Option<&'a Value>,
) -> Result<MediaSource<'a>, Error> {
	let Some(native) = native else {
		let media = documents::inline_media(blob).map_err(|detail| Error::Provider(detail.into()))?;
		return Ok(MediaSource::inline(media));
	};
	let object = native
		.as_object()
		.ok_or_else(|| Error::Provider("Anthropic native media source must be an object".into()))?;
	let source_type = object.get("type").and_then(Value::as_str).ok_or_else(|| {
		Error::Provider("Anthropic native media source requires a string type".into())
	})?;
	let exact_fields = |fields: &[&str]| {
		if object.len() != fields.len() || fields.iter().any(|field| !object.contains_key(*field)) {
			return Err(Error::Provider(
				format!("Anthropic {source_type} source must contain exactly {}", fields.join(" and "))
					.into(),
			));
		}
		Ok(())
	};
	match source_type {
		"base64" => {
			exact_fields(&["type"])?;
			let media =
				documents::inline_media(blob).map_err(|detail| Error::Provider(detail.into()))?;
			Ok(MediaSource::inline(media))
		},
		"url" => {
			exact_fields(&["type", "url"])?;
			let value = object
				.get("url")
				.and_then(Value::as_str)
				.ok_or_else(|| Error::Provider("Anthropic URL source requires a string url".into()))?;
			let url = documents::url_source(value).map_err(|detail| Error::Provider(detail.into()))?;
			Ok(MediaSource::Url { url })
		},
		"file" => {
			exact_fields(&["type", "file_id"])?;
			let value = object
				.get("file_id")
				.and_then(Value::as_str)
				.ok_or_else(|| {
					Error::Provider("Anthropic file source requires a string file_id".into())
				})?;
			let file_id =
				documents::file_source(value).map_err(|detail| Error::Provider(detail.into()))?;
			Ok(MediaSource::File { file_id })
		},
		_ => Err(Error::Provider(
			"Anthropic native media source type must be base64, url, or file".into(),
		)),
	}
}

fn append_message<'a>(
	messages: &mut Vec<WireMessage<'a>>,
	role: &'static str,
	blocks: Vec<Block<'a>>,
) {
	if blocks.is_empty() {
		return;
	}
	if let Some(last) = messages.last_mut().filter(|message| message.role == role) {
		last.content.extend(blocks);
	} else {
		messages.push(WireMessage { role, content: blocks });
	}
}

fn reorder_tool_uses(blocks: &mut Vec<Block<'_>>) -> bool {
	let mut saw_tool = false;
	let mut needs_reorder = false;
	for block in blocks.iter() {
		if matches!(block, Block::ToolUse { .. }) {
			saw_tool = true;
		} else if saw_tool {
			needs_reorder = true;
			break;
		}
	}
	if needs_reorder {
		let mut tools = Vec::new();
		let mut content = Vec::with_capacity(blocks.len());
		for block in blocks.drain(..) {
			if matches!(block, Block::ToolUse { .. }) {
				tools.push(block);
			} else {
				content.push(block);
			}
		}
		content.extend(tools);
		*blocks = content;
	}
	needs_reorder
}

fn raw(bytes: &[u8]) -> Result<&RawValue, Error> {
	serde_json::from_slice(bytes).map_err(json_error)
}

fn feature_report(
	unsupported: &mut Vec<Unsupported>,
	fallback: Fallback,
	what: &str,
	detail: &str,
) -> Result<(), Error> {
	let action = match fallback {
		Fallback::Error => {
			return Err(Error::Unsupported(vec![
				Unsupported::builder()
					.what(Str::from(what))
					.detail(Str::from(detail))
					.action(UnsupportedAction::Dropped)
					.build(),
			]));
		},
		Fallback::Ignore => UnsupportedAction::Dropped,
		Fallback::Emulate => UnsupportedAction::Emulated,
		_ => UnsupportedAction::Dropped,
	};
	report(unsupported, what, detail, action);
	Ok(())
}
fn report(unsupported: &mut Vec<Unsupported>, what: &str, detail: &str, action: UnsupportedAction) {
	unsupported.push(
		Unsupported::builder()
			.what(Str::from(what))
			.detail(Str::from(detail))
			.action(action)
			.build(),
	);
}
#[allow(
	clippy::trivially_copy_pass_by_ref,
	reason = "serde skip_serializing_if callbacks receive a reference"
)]
const fn is_false(value: &bool) -> bool {
	!*value
}

fn empty_props() -> &'static Props {
	static EMPTY: LazyLock<Props> = LazyLock::new(Props::default);
	&EMPTY
}
fn keep_all_thinking() -> &'static Value {
	static KEEP_ALL: LazyLock<Value> = LazyLock::new(
		|| serde_json::json!({"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}),
	);
	&KEEP_ALL
}

fn serialize(value: &impl Serialize) -> Result<Bytes, Error> {
	serde_json::to_vec(value)
		.map(Bytes::from)
		.map_err(json_error)
}

#[cold]
fn json_error(error: serde_json::Error) -> Error {
	Error::Provider(Str::from(error.to_string()))
}

enum Incoming<'a> {
	MessageStart { message: IncomingMessage<'a> },
	ContentBlockStart { index: u32, content_block: IncomingBlock<'a> },
	ContentBlockDelta { index: u32, delta: IncomingDelta<'a> },
	ContentBlockStop { index: u32 },
	MessageDelta { delta: IncomingMessageDelta<'a>, usage: Option<IncomingUsage> },
	MessageStop,
	Error { error: IncomingError<'a> },
	Ping,
}

#[derive(Deserialize)]
struct IncomingMessage<'a> {
	#[serde(default, borrow)]
	model:              &'a str,
	#[serde(default)]
	usage:              Option<IncomingUsage>,
	#[serde(default, borrow)]
	context_management: Option<&'a RawValue>,
	#[serde(default, borrow)]
	container:          Option<&'a RawValue>,
	#[serde(default, borrow)]
	service_tier:       Option<&'a str>,
}

enum IncomingBlock<'a> {
	Text { text: &'a str, citations: Vec<Value> },
	Thinking { thinking: &'a str, signature: &'a str },
	RedactedThinking { data: &'a str },
	ToolUse { id: &'a str, name: &'a str, input: &'a RawValue },
	Server { raw: &'a RawValue },
}

enum IncomingDelta<'a> {
	Text { text: Cow<'a, str> },
	Thinking { thinking: Cow<'a, str> },
	Signature { signature: Cow<'a, str> },
	InputJson { partial_json: Cow<'a, str> },
	Citation { citation: &'a RawValue },
}

#[derive(Deserialize)]
struct IncomingMessageDelta<'a> {
	#[serde(default, borrow)]
	stop_reason:        Option<&'a str>,
	#[serde(default, borrow)]
	stop_sequence:      Option<&'a str>,
	#[serde(default, borrow)]
	stop_details:       Option<&'a RawValue>,
	#[serde(default, borrow)]
	context_management: Option<&'a RawValue>,
	#[serde(default, borrow)]
	container:          Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct IncomingUsage {
	#[serde(rename = "input_tokens")]
	input:           Option<u64>,
	#[serde(rename = "output_tokens")]
	output:          Option<u64>,
	#[serde(rename = "cache_read_input_tokens")]
	cache_read:      Option<u64>,
	#[serde(rename = "cache_creation_input_tokens")]
	cache_write:     Option<u64>,
	#[serde(default)]
	cache_creation:  Option<Value>,
	#[serde(default)]
	server_tool_use: Option<Value>,
	#[serde(default)]
	iterations:      Option<Value>,
	#[serde(flatten)]
	extra:           serde_json::Map<String, Value>,
}

#[derive(Deserialize)]
struct IncomingError<'a> {
	#[serde(rename = "type", borrow)]
	kind:    &'a str,
	#[serde(borrow)]
	message: &'a str,
}

#[derive(Deserialize)]
struct IncomingTag<'a> {
	#[serde(rename = "type", borrow)]
	kind: &'a str,
}

fn parse_event(data: &[u8]) -> Result<Incoming<'_>, Error> {
	let tag: IncomingTag<'_> = serde_json::from_slice(data).map_err(json_error)?;
	match tag.kind {
		"message_start" => {
			#[derive(Deserialize)]
			struct Event<'a> {
				#[serde(borrow)]
				message: IncomingMessage<'a>,
			}
			let event: Event<'_> = serde_json::from_slice(data).map_err(json_error)?;
			Ok(Incoming::MessageStart { message: event.message })
		},
		"content_block_start" => {
			#[derive(Deserialize)]
			struct Event<'a> {
				index:         u32,
				#[serde(borrow)]
				content_block: &'a RawValue,
			}
			let event: Event<'_> = serde_json::from_slice(data).map_err(json_error)?;
			Ok(Incoming::ContentBlockStart {
				index:         event.index,
				content_block: parse_block(event.content_block)?,
			})
		},
		"content_block_delta" => {
			#[derive(Deserialize)]
			struct Event<'a> {
				index: u32,
				#[serde(borrow)]
				delta: &'a RawValue,
			}
			let event: Event<'_> = serde_json::from_slice(data).map_err(json_error)?;
			Ok(Incoming::ContentBlockDelta { index: event.index, delta: parse_delta(event.delta)? })
		},
		"content_block_stop" => {
			#[derive(Deserialize)]
			struct Event {
				index: u32,
			}
			let event: Event = serde_json::from_slice(data).map_err(json_error)?;
			Ok(Incoming::ContentBlockStop { index: event.index })
		},
		"message_delta" => {
			#[derive(Deserialize)]
			struct Event<'a> {
				#[serde(borrow)]
				delta: IncomingMessageDelta<'a>,
				usage: Option<IncomingUsage>,
			}
			let event: Event<'_> = serde_json::from_slice(data).map_err(json_error)?;
			Ok(Incoming::MessageDelta { delta: event.delta, usage: event.usage })
		},
		"message_stop" => Ok(Incoming::MessageStop),
		"error" => {
			#[derive(Deserialize)]
			struct Event<'a> {
				#[serde(borrow)]
				error: IncomingError<'a>,
			}
			let event: Event<'_> = serde_json::from_slice(data).map_err(json_error)?;
			Ok(Incoming::Error { error: event.error })
		},
		"ping" => Ok(Incoming::Ping),
		kind => Err(Error::Provider(Str::from(format!("unknown Anthropic event type `{kind}`")))),
	}
}

fn parse_block(raw: &RawValue) -> Result<IncomingBlock<'_>, Error> {
	let tag: IncomingTag<'_> = serde_json::from_str(raw.get()).map_err(json_error)?;
	match tag.kind {
		"text" => {
			#[derive(Deserialize)]
			struct Block<'a> {
				#[serde(default, borrow)]
				text:      &'a str,
				#[serde(default)]
				citations: Vec<Value>,
			}
			let block: Block<'_> = serde_json::from_str(raw.get()).map_err(json_error)?;
			Ok(IncomingBlock::Text { text: block.text, citations: block.citations })
		},
		"thinking" => {
			#[derive(Deserialize)]
			struct Block<'a> {
				#[serde(default, borrow)]
				thinking:  &'a str,
				#[serde(default, borrow)]
				signature: &'a str,
			}
			let block: Block<'_> = serde_json::from_str(raw.get()).map_err(json_error)?;
			Ok(IncomingBlock::Thinking { thinking: block.thinking, signature: block.signature })
		},
		"redacted_thinking" => {
			#[derive(Deserialize)]
			struct Block<'a> {
				#[serde(borrow)]
				data: &'a str,
			}
			let block: Block<'_> = serde_json::from_str(raw.get()).map_err(json_error)?;
			Ok(IncomingBlock::RedactedThinking { data: block.data })
		},
		"tool_use" => {
			#[derive(Deserialize)]
			struct Block<'a> {
				#[serde(borrow)]
				id:    &'a str,
				#[serde(borrow)]
				name:  &'a str,
				#[serde(borrow)]
				input: &'a RawValue,
			}
			let block: Block<'_> = serde_json::from_str(raw.get()).map_err(json_error)?;
			Ok(IncomingBlock::ToolUse { id: block.id, name: block.name, input: block.input })
		},
		"server_tool_use"
		| "web_search_tool_result"
		| "web_fetch_tool_result"
		| "code_execution_tool_result"
		| "bash_code_execution_tool_result"
		| "text_editor_code_execution_tool_result" => Ok(IncomingBlock::Server { raw }),
		"fallback" => {
			#[derive(Deserialize)]
			struct ModelRef<'a> {
				#[serde(borrow)]
				model: &'a str,
			}
			#[derive(Deserialize)]
			struct Block<'a> {
				#[serde(borrow)]
				from: ModelRef<'a>,
				#[serde(borrow)]
				to:   ModelRef<'a>,
			}
			let block: Block<'_> = serde_json::from_str(raw.get()).map_err(|error| {
				Error::Provider(Str::from(format!(
					"malformed Anthropic fallback content block: {error}"
				)))
			})?;
			if block.from.model.trim().is_empty() || block.to.model.trim().is_empty() {
				return Err(Error::Provider(Str::from(
					"malformed Anthropic fallback content block: model references must be non-empty",
				)));
			}
			Ok(IncomingBlock::Server { raw })
		},
		kind => {
			Err(Error::Provider(Str::from(format!("unknown Anthropic content block type `{kind}`"))))
		},
	}
}

fn parse_delta(raw: &RawValue) -> Result<IncomingDelta<'_>, Error> {
	let tag: IncomingTag<'_> = serde_json::from_str(raw.get()).map_err(json_error)?;
	match tag.kind {
		"text_delta" => {
			#[derive(Deserialize)]
			struct Delta<'a> {
				#[serde(borrow)]
				text: Cow<'a, str>,
			}
			let delta: Delta<'_> = serde_json::from_str(raw.get()).map_err(json_error)?;
			Ok(IncomingDelta::Text { text: delta.text })
		},
		"thinking_delta" => {
			#[derive(Deserialize)]
			struct Delta<'a> {
				#[serde(borrow)]
				thinking: Cow<'a, str>,
			}
			let delta: Delta<'_> = serde_json::from_str(raw.get()).map_err(json_error)?;
			Ok(IncomingDelta::Thinking { thinking: delta.thinking })
		},
		"signature_delta" => {
			#[derive(Deserialize)]
			struct Delta<'a> {
				#[serde(borrow)]
				signature: Cow<'a, str>,
			}
			let delta: Delta<'_> = serde_json::from_str(raw.get()).map_err(json_error)?;
			Ok(IncomingDelta::Signature { signature: delta.signature })
		},
		"input_json_delta" => {
			#[derive(Deserialize)]
			struct Delta<'a> {
				#[serde(borrow)]
				partial_json: Cow<'a, str>,
			}
			let delta: Delta<'_> = serde_json::from_str(raw.get()).map_err(json_error)?;
			Ok(IncomingDelta::InputJson { partial_json: delta.partial_json })
		},
		"citations_delta" | "citation_delta" => {
			#[derive(Deserialize)]
			struct Delta<'a> {
				#[serde(borrow)]
				citation: &'a RawValue,
			}
			let delta: Delta<'_> = serde_json::from_str(raw.get()).map_err(json_error)?;
			Ok(IncomingDelta::Citation { citation: delta.citation })
		},
		kind => {
			Err(Error::Provider(Str::from(format!("unknown Anthropic content delta type `{kind}`"))))
		},
	}
}
#[derive(Default)]
struct AnthropicState {
	model:          Str,
	blocks:         BTreeMap<u32, DecodedBlock>,
	usage:          Option<Usage>,
	stop:           Option<StopReason>,
	response_props: Props,
	completed:      bool,
	id_mapper:      CallIdMapper,
	tool_names:     ToolNamePolicy,
}

impl AnthropicState {
	fn new(tool_names: ToolNamePolicy) -> Self {
		Self { tool_names, ..Self::default() }
	}
}

enum DecodedBlock {
	Text { text: StrMut, citations: Vec<Value> },
	Thinking { text: StrMut, signature: BytesMut, redacted: bool },
	Tool { id: CallId, name: Str, args: BytesMut, initial: Bytes, streamed: bool },
	Server(Value),
}

fn decode_event(
	event: Incoming<'_>,
	state: &mut AnthropicState,
) -> Result<SmallVec<TurnEvent, 2>, Error> {
	let mut events = SmallVec::new();
	match event {
		Incoming::MessageStart { message } => {
			state.model = Str::from(message.model);
			if let Some(usage) = message.usage {
				state.usage = Some(canonical_usage(usage, state.usage.as_ref()));
			}
			capture_raw(&mut state.response_props, "context_management", message.context_management)?;
			capture_raw(&mut state.response_props, "container", message.container)?;
			if let Some(tier) = message.service_tier {
				state.response_props.insert_ns(
					"anthropic",
					"service_tier",
					Value::String(tier.to_owned()),
				);
			}
		},
		Incoming::ContentBlockStart { index, content_block } => {
			let (block, kind, id, name, initial) = match content_block {
				IncomingBlock::Text { text, citations } => (
					DecodedBlock::Text { text: StrMut::new(text), citations },
					StreamPartKind::Text,
					Str::default(),
					Str::default(),
					Some(text.as_bytes()),
				),
				IncomingBlock::Thinking { thinking, signature } => (
					DecodedBlock::Thinking {
						text:      StrMut::new(thinking),
						signature: BytesMut::from(signature.as_bytes()),
						redacted:  false,
					},
					StreamPartKind::Thinking,
					Str::default(),
					Str::default(),
					Some(thinking.as_bytes()),
				),
				IncomingBlock::RedactedThinking { data } => (
					DecodedBlock::Thinking {
						text:      StrMut::default(),
						signature: BytesMut::from(data.as_bytes()),
						redacted:  true,
					},
					StreamPartKind::Thinking,
					Str::default(),
					Str::default(),
					None,
				),
				IncomingBlock::ToolUse { id, name, input } => {
					let canonical = id
						.strip_prefix("toolu_")
						.unwrap_or(id)
						.parse()
						.unwrap_or_else(|_| state.id_mapper.observe(id));
					let name = state.tool_names.decode(name);
					(
						DecodedBlock::Tool {
							id:       canonical,
							name:     Str::from(name),
							args:     BytesMut::new(),
							initial:  Bytes::copy_from_slice(input.get().as_bytes()),
							streamed: false,
						},
						StreamPartKind::ToolCall,
						Str::from(canonical.to_string()),
						Str::from(name),
						None,
					)
				},
				IncomingBlock::Server { raw } => {
					state.blocks.insert(
						index,
						DecodedBlock::Server(serde_json::from_str(raw.get()).map_err(json_error)?),
					);
					return Ok(events);
				},
			};
			state.blocks.insert(index, block);
			events.push(TurnEvent::PartStart { index, kind, tool_call_id: id, tool_name: name });
			if let Some(bytes) = initial.filter(|bytes| !bytes.is_empty()) {
				events.push(TurnEvent::PartDelta { index, chunk: Bytes::copy_from_slice(bytes) });
			}
		},
		Incoming::ContentBlockDelta { index, delta } => match delta {
			IncomingDelta::Text { text } => {
				append_delta(state, &mut events, index, text.as_ref(), DeltaKind::Text)?;
			},
			IncomingDelta::Thinking { thinking } => {
				append_delta(state, &mut events, index, thinking.as_ref(), DeltaKind::Thinking)?;
			},
			IncomingDelta::Signature { signature } => {
				let Some(DecodedBlock::Thinking { signature: target, .. }) =
					state.blocks.get_mut(&index)
				else {
					return Err(Error::Provider(Str::from("signature delta for a non-thinking block")));
				};
				target.extend_from_slice(signature.as_bytes());
			},
			IncomingDelta::InputJson { partial_json } => {
				append_delta(state, &mut events, index, partial_json.as_ref(), DeltaKind::Tool)?;
			},
			IncomingDelta::Citation { citation } => {
				let Some(DecodedBlock::Text { citations, .. }) = state.blocks.get_mut(&index) else {
					return Err(Error::Provider(Str::from("citation delta for a non-text block")));
				};
				citations.push(serde_json::from_str(citation.get()).map_err(json_error)?);
			},
		},
		Incoming::ContentBlockStop { index } => {
			let signature = match state.blocks.get(&index) {
				Some(DecodedBlock::Thinking { signature, .. }) => signature.clone().freeze(),
				Some(DecodedBlock::Server(_)) => return Ok(events),
				Some(DecodedBlock::Text { .. } | DecodedBlock::Tool { .. }) | None => Bytes::new(),
			};
			events.push(TurnEvent::PartEnd { index, signature });
		},
		Incoming::MessageDelta { delta, usage } => {
			if let Some(reason) = delta.stop_reason {
				state.stop = Some(stop_reason(reason));
			}
			if let Some(sequence) = delta.stop_sequence {
				state.response_props.insert_ns(
					"anthropic",
					"stop_sequence",
					Value::String(sequence.to_owned()),
				);
			}
			capture_raw(&mut state.response_props, "stop_details", delta.stop_details)?;
			capture_raw(&mut state.response_props, "context_management", delta.context_management)?;
			capture_raw(&mut state.response_props, "container", delta.container)?;
			if let Some(usage) = usage {
				state.usage = Some(canonical_usage(usage, state.usage.as_ref()));
			}
		},
		Incoming::MessageStop => {
			state.completed = true;
			events.push(TurnEvent::Outcome(build_outcome(state)));
		},
		Incoming::Error { error } => {
			state.completed = true;
			events.push(TurnEvent::Error(
				TurnError::builder()
					.kind(error_kind(error.kind))
					.detail(Str::from(error.message))
					.unsupported(Vec::new())
					.retry_after_ms(0)
					.build(),
			));
		},
		Incoming::Ping => {},
	}
	Ok(events)
}

enum DeltaKind {
	Text,
	Thinking,
	Tool,
}

fn append_delta(
	state: &mut AnthropicState,
	events: &mut SmallVec<TurnEvent, 2>,
	index: u32,
	value: &str,
	kind: DeltaKind,
) -> Result<(), Error> {
	match (state.blocks.get_mut(&index), kind) {
		(Some(DecodedBlock::Text { text: target, .. }), DeltaKind::Text) => target.push_str(value),
		(Some(DecodedBlock::Thinking { text, .. }), DeltaKind::Thinking) => text.push_str(value),
		(Some(DecodedBlock::Tool { args, streamed, .. }), DeltaKind::Tool) => {
			*streamed = true;
			args.extend_from_slice(value.as_bytes());
		},
		_ => {
			return Err(Error::Provider(Str::from("content delta type did not match its block")));
		},
	}
	events.push(TurnEvent::PartDelta { index, chunk: Bytes::copy_from_slice(value.as_bytes()) });
	Ok(())
}

fn build_outcome(state: &mut AnthropicState) -> ChatOutcome {
	let mut output = Vec::new();
	let mut parts = Vec::new();
	let mut citations = Vec::new();
	for block in std::mem::take(&mut state.blocks).into_values() {
		match block {
			DecodedBlock::Text { text, citations: block_citations } => {
				let part_index = parts.len();
				parts.push(Part::Text(text.freeze()));
				citations.extend(block_citations.into_iter().map(
					|citation| serde_json::json!({"part_index": part_index, "citation": citation}),
				));
			},
			DecodedBlock::Thinking { text, signature, redacted } => {
				parts.push(Part::Thinking(
					Thinking::builder()
						.text(text.freeze())
						.signature(signature.freeze())
						.redacted(redacted)
						.build(),
				));
			},
			DecodedBlock::Tool { id, name, args, initial, streamed } => {
				push_message(&mut output, &mut parts, &mut citations, &state.model);
				output.push(
					Item::builder()
						.seq(0)
						.kind(ItemKind::ToolCall(
							ToolCall::builder()
								.id(id)
								.name(name)
								.args_json(if streamed { args.freeze() } else { initial })
								.thought_signature(Bytes::new())
								.build(),
						))
						.props(Props::default())
						.build(),
				);
			},
			DecodedBlock::Server(block) => {
				push_message(&mut output, &mut parts, &mut citations, &state.model);
				let mut props = Props::default();
				props.insert_ns("anthropic", "server_tool_block", block);
				output.push(
					Item::builder()
						.seq(0)
						.kind(ItemKind::Message(
							Message::builder()
								.role(Role::Assistant)
								.parts(Vec::new())
								.build(),
						))
						.props(props)
						.build(),
				);
			},
		}
	}
	push_message(&mut output, &mut parts, &mut citations, &state.model);
	ChatOutcome::builder()
		.output(output)
		.stop(state.stop.unwrap_or(StopReason::EndTurn))
		.maybe_usage(state.usage.take())
		.provider(Str::from("anthropic"))
		.model(std::mem::take(&mut state.model))
		.unsupported(Vec::new())
		.props(std::mem::take(&mut state.response_props))
		.build()
}

fn push_message(
	output: &mut Vec<Item>,
	parts: &mut Vec<Part>,
	citations: &mut Vec<Value>,
	model: &str,
) {
	if parts.is_empty() {
		return;
	}
	let mut props = Props::default();
	if !model.is_empty() {
		props.insert_ns("anthropic", "model", Value::String(model.to_owned()));
	}
	if !citations.is_empty() {
		props.insert_ns("anthropic", "citations", Value::Array(std::mem::take(citations)));
	}
	output.push(
		Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::Assistant)
					.parts(std::mem::take(parts))
					.build(),
			))
			.props(props)
			.build(),
	);
}

fn canonical_usage(usage: IncomingUsage, prior: Option<&Usage>) -> Usage {
	let mut detail = prior.map_or_else(Props::default, |value| value.detail.clone());
	if let Some(value) = usage.cache_creation {
		detail.insert_ns("anthropic", "cache_creation", value);
	}
	if let Some(value) = usage.server_tool_use {
		detail.insert_ns("anthropic", "server_tool_use", value);
	}
	if let Some(value) = usage.iterations {
		detail.insert_ns("anthropic", "iterations", value);
	}
	if !usage.extra.is_empty() {
		detail.insert_ns("anthropic", "usage_extra", Value::Object(usage.extra));
	}
	Usage::builder()
		.input_tokens(
			usage
				.input
				.or_else(|| prior.map(|value| value.input_tokens))
				.unwrap_or(0),
		)
		.output_tokens(
			usage
				.output
				.or_else(|| prior.map(|value| value.output_tokens))
				.unwrap_or(0),
		)
		.cache_read_tokens(
			usage
				.cache_read
				.or_else(|| prior.map(|value| value.cache_read_tokens))
				.unwrap_or(0),
		)
		.cache_write_tokens(
			usage
				.cache_write
				.or_else(|| prior.map(|value| value.cache_write_tokens))
				.unwrap_or(0),
		)
		.accuracy(Accuracy::Exact)
		.detail(detail)
		.build()
}

fn capture_raw(
	props: &mut Props,
	name: &'static str,
	value: Option<&RawValue>,
) -> Result<(), Error> {
	if let Some(value) = value {
		props.insert_ns("anthropic", name, serde_json::from_str(value.get()).map_err(json_error)?);
	}
	Ok(())
}

fn stop_reason(reason: &str) -> StopReason {
	match reason {
		"tool_use" => StopReason::ToolUse,
		"max_tokens" | "model_context_window_exceeded" => StopReason::MaxTokens,
		"refusal" => StopReason::ContentFilter,
		"end_turn" | "stop_sequence" | "pause_turn" => StopReason::EndTurn,
		_ => StopReason::EndTurn,
	}
}

fn error_kind(kind: &str) -> TurnErrorKind {
	match kind {
		"authentication_error" | "permission_error" => TurnErrorKind::Auth,
		"rate_limit_error" => TurnErrorKind::RateLimited,
		"overloaded_error" => TurnErrorKind::Overloaded,
		_ => TurnErrorKind::Upstream,
	}
}

#[cfg(test)]
mod tests {
	use std::{collections::BTreeMap, sync::Arc};

	use bytes::{Bytes, BytesMut};
	use omp_llm_catalog::compat::{
		Compat, ReasoningWireFormat, ThinkingToolChoiceConflict, ToolStrictMode,
	};
	use omp_llm_transport::{DecodeState, Frame, Transport, sse::SseDecoder};
	use omp_llm_types::{
		BlobPart, CacheHint, CacheRetention, CallId, ChatRequest, Effort, Fallback, Feature, Item,
		ItemKind, JsonSchema, Message, Part, PromptCacheBreakpoint, Props, Reasoning,
		ResolvedModelPolicy, ResolvedThinkingMode, ResolvedThinkingPolicy, ResponseFormat,
		ResponseFormatKind, Role, Sampling, Thinking, Thread, ToolCall, ToolChoice, ToolDef,
		ToolResult, TurnErrorKind, TurnEvent, Unsupported, UnsupportedAction,
	};
	use serde_json::{Value, json};

	use super::{AnthropicCodec, request_headers};

	fn item(kind: ItemKind) -> Item {
		Item::builder()
			.seq(0)
			.kind(kind)
			.props(Props::default())
			.build()
	}

	fn request(items: Vec<Item>) -> ChatRequest {
		ChatRequest::builder()
			.model("claude-sonnet-4-5".into())
			.thread(Thread::builder().items(items).build())
			.tools(Vec::new())
			.build()
	}

	fn encoded(request: &ChatRequest, compat: &Compat) -> (Value, Vec<Unsupported>) {
		let (body, unsupported) = AnthropicCodec::new().encode(request, compat).unwrap();
		(serde_json::from_slice(&body).unwrap(), unsupported)
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
	fn fixture_tool_uses_are_stably_moved_to_the_tail() {
		let fixture: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/anthropic/request.tool_use_ordering.json"
		))
		.unwrap();
		assert!(fixture["invariant"].as_str().unwrap().contains("stable"));
		let call = |id: CallId, name: &str| {
			ItemKind::ToolCall(
				ToolCall::builder()
					.id(id)
					.name(name.into())
					.args_json(Bytes::from_static(b"{}"))
					.thought_signature(Bytes::new())
					.build(),
			)
		};
		let message = |text: &str| {
			ItemKind::Message(
				Message::builder()
					.role(Role::Assistant)
					.parts(vec![Part::Text(text.into())])
					.build(),
			)
		};
		let request = request(vec![
			item(message("before")),
			item(call(CallId::new(), "read")),
			item(message("after")),
			item(call(CallId::new(), "grep")),
		]);
		let (wire, unsupported) = encoded(&request, &Compat::default());
		let kinds = wire["messages"][0]["content"]
			.as_array()
			.unwrap()
			.iter()
			.map(|block| block["type"].as_str().unwrap())
			.collect::<Vec<_>>();
		assert_eq!(kinds, ["text", "text", "tool_use", "tool_use"]);
		assert!(unsupported.iter().any(|entry| {
			entry.what == "thread.assistant.tool_use_order"
				&& entry.action == UnsupportedAction::Emulated
		}));
	}

	#[test]
	fn error_image_placement_tracks_the_model_wire_contract() {
		let fixture: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/anthropic/request.tool_result_image.json"
		))
		.unwrap();
		assert_eq!(fixture["canonical_intent"]["messages"][1]["content"][1]["data"], "aGVsbG8=");
		let id = CallId::new();
		let mut request = request(vec![
			item(ItemKind::ToolCall(
				ToolCall::builder()
					.id(id)
					.name("inspect".into())
					.args_json(Bytes::from_static(br#"{"path":"shot.png"}"#))
					.thought_signature(Bytes::new())
					.build(),
			)),
			item(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(id)
					.name("inspect".into())
					.parts(vec![
						Part::Text("invalid image".into()),
						Part::Blob(
							BlobPart::builder()
								.hash([0; 32])
								.mime("image/png".into())
								.size(5)
								.inline(Bytes::from_static(b"hello"))
								.build(),
						),
					])
					.is_error(true)
					.build(),
			)),
		]);

		let (wire, _) = encoded(&request, &Compat::default());
		let content = wire["messages"][1]["content"].as_array().unwrap();
		assert_eq!(content.len(), 1);
		assert_eq!(content[0]["type"], "tool_result");
		assert_eq!(content[0]["content"][1]["type"], "image");

		request.model = "claude-sonnet-4-6".into();
		let (wire, _) = encoded(&request, &Compat::default());
		let content = wire["messages"][1]["content"].as_array().unwrap();
		assert_eq!(content[0]["type"], "tool_result");
		assert_eq!(content[0]["content"].as_array().unwrap().len(), 1);
		assert_eq!(content[1]["type"], "text");
		assert_eq!(content[2]["type"], "image");
		assert_eq!(content[2]["source"]["data"], "aGVsbG8=");
	}

	#[test]
	fn thinking_clamp_cache_and_host_quirks_are_explicit() {
		let mut request = request(vec![
			item(ItemKind::Message(
				Message::builder()
					.role(Role::System)
					.parts(vec![Part::Text("system".into())])
					.build(),
			)),
			item(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Text("hello".into())])
					.build(),
			)),
		]);
		request.thinking = Some(
			Feature::builder()
				.value(
					Reasoning::builder()
						.budget_tokens(8_000)
						.hide_summary(false)
						.build(),
				)
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		request.sampling = Some(Sampling::builder().max_output_tokens(4_000).build());
		request.cache = Some(
			CacheHint::builder()
				.session_key("session".into())
				.retention(CacheRetention::Long)
				.build(),
		);
		request.tools.push(
			ToolDef::builder()
				.name("run".into())
				.description("run".into())
				.schema_json(Bytes::from_static(br#"{"type":"object"}"#))
				.strict(true)
				.build(),
		);
		request.tool_choice = Some(
			Feature::builder()
				.value(ToolChoice::Named("run".into()))
				.on_unsupported(Fallback::Emulate)
				.build(),
		);
		let mut compat = Compat::default();
		compat.tool_strict_mode = ToolStrictMode::None;
		compat.reasoning_wire_format = ReasoningWireFormat::Anthropic;
		compat.thinking_tool_choice_conflict = ThinkingToolChoiceConflict::None;
		let (wire, unsupported) = encoded(&request, &compat);
		assert_eq!(wire["max_tokens"], 9024);
		assert_eq!(wire["system"][0]["cache_control"]["ttl"], "1h");
		assert_eq!(wire["tool_choice"], json!({"type":"tool","name":"run"}));
		assert_eq!(wire["thinking"], json!({"type":"enabled","budget_tokens":8000}));
		assert!(wire["tools"][0].get("strict").is_none());
		{
			let path = "tools.strict";
			assert!(unsupported.iter().any(|entry| entry.what == path));
		}
		assert!(unsupported.iter().any(|entry| {
			entry.what == "sampling.max_output_tokens" && entry.action == UnsupportedAction::Clamped
		}));

		let conflict_compat = {
			let mut c = compat;
			c.thinking_tool_choice_conflict = ThinkingToolChoiceConflict::DropThinkingWhenAny;
			c
		};
		let (wire, unsupported) = encoded(&request, &conflict_compat);
		assert!(wire.get("thinking").is_none());
		assert!(
			unsupported
				.iter()
				.any(|entry| entry.what == "thinking.tool_choice_conflict")
		);
		request.tool_choice = None;
		let (wire, _) = encoded(&request, &compat);
		assert_eq!(wire["thinking"], json!({"type":"enabled","budget_tokens":8000}));
		request.cache.as_mut().unwrap().retention = Some(CacheRetention::Short);
		let (wire, _) = encoded(&request, &compat);
		assert_eq!(wire["system"][0]["cache_control"]["ttl"], "5m");

		request.sampling = None;
		let (wire, unsupported) = encoded(&request, &compat);
		assert!(wire.get("max_tokens").is_none());
		assert!(!unsupported.iter().any(|entry| {
			entry.what == "sampling.max_output_tokens" && entry.action == UnsupportedAction::Clamped
		}));
	}
	#[test]
	fn signed_replay_media_and_adaptive_controls_match_anthropic_wire() {
		let fixture: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/anthropic/request.media_projection.json"
		))
		.unwrap();
		let mut compat = Compat::default();
		compat.reasoning_wire_format = ReasoningWireFormat::Anthropic;

		let thinking_item = |source_model: &str, signature: Bytes| {
			let mut props = Props::default();
			props.insert_ns("anthropic", "model", json!(source_model));
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::Assistant)
						.parts(vec![Part::Thinking(
							Thinking::builder()
								.text("private plan".into())
								.signature(signature)
								.redacted(false)
								.build(),
						)])
						.build(),
				))
				.props(props)
				.build()
		};
		let mut same =
			request(vec![thinking_item("claude-sonnet-4-6", Bytes::from_static(b"sig_same"))]);
		same.model = "claude-sonnet-4-6".into();
		let (wire, _) = encoded(&same, &compat);
		assert_eq!(wire["messages"][0]["content"][0]["type"], "thinking");
		assert_eq!(wire["messages"][0]["content"][0]["signature"], "sig_same");

		let mut foreign =
			request(vec![thinking_item("claude-sonnet-4-6", Bytes::from_static(b"sig_foreign"))]);
		foreign.model = "claude-opus-4-6".into();
		let (wire, unsupported) = encoded(&foreign, &compat);
		assert_eq!(wire["messages"][0]["content"][0], json!({"type":"text","text":"private plan"}));
		assert!(
			unsupported
				.iter()
				.any(|entry| entry.what == "thread.assistant.thinking")
		);

		let mut unsigned = request(vec![thinking_item("claude-sonnet-4-6", Bytes::new())]);
		unsigned.model = "claude-sonnet-4-6".into();
		let (wire, _) = encoded(&unsigned, &compat);
		assert_eq!(wire["messages"][0]["content"][0]["type"], "text");

		let image = BlobPart::builder()
			.hash([0; 32])
			.mime("image/png".into())
			.size(5)
			.inline(Bytes::from_static(b"hello"))
			.build();
		let pdf = BlobPart::builder()
			.hash([1; 32])
			.mime("application/pdf".into())
			.size(4)
			.inline(Bytes::from_static(b"%PDF"))
			.build();
		let mut media = request(vec![item(ItemKind::Message(
			Message::builder()
				.role(Role::User)
				.parts(vec![Part::Blob(image.clone()), Part::Blob(pdf)])
				.build(),
		))]);
		media.cache = Some(
			CacheHint::builder()
				.session_key("media-session".into())
				.retention(CacheRetention::Long)
				.build(),
		);
		let (wire, unsupported) = encoded(&media, &compat);
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(wire["messages"][0]["content"][0]["source"], fixture["image"]["source"]);
		assert_eq!(
			wire["messages"][0]["content"][1]["source"],
			fixture["documents"]["base64"]["source"]
		);
		assert_eq!(wire["messages"][0]["content"][1]["type"], "document");
		let beta = request_headers(&media, &compat)
			.into_iter()
			.find(|header| header.name == "anthropic-beta")
			.expect("negotiated beta header");
		for expected in fixture["required_betas"].as_array().unwrap() {
			assert!(
				beta
					.value
					.split(",")
					.any(|value| value == expected.as_str().unwrap())
			);
		}
		let mut native_url = item(ItemKind::Message(
			Message::builder()
				.role(Role::User)
				.parts(vec![Part::Blob(image.clone())])
				.build(),
		));
		native_url.props.insert_ns(
			"anthropic",
			"image_sources",
			json!([{"type":"url","url":"https://example.test/image.png"}]),
		);
		let (wire, unsupported) = encoded(&request(vec![native_url]), &compat);
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(
			wire["messages"][0]["content"][0]["source"],
			json!({"type":"url","url":"https://example.test/image.png"})
		);

		let mut native_file = item(ItemKind::Message(
			Message::builder()
				.role(Role::User)
				.parts(vec![Part::Blob(image)])
				.build(),
		));
		native_file.props.insert_ns(
			"anthropic",
			"image_sources",
			json!([{"type":"file","file_id":"file_012345"}]),
		);
		let native_file_request = request(vec![native_file]);
		let (wire, unsupported) = encoded(&native_file_request, &compat);
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(
			wire["messages"][0]["content"][0]["source"],
			json!({"type":"file","file_id":"file_012345"})
		);
		assert!(
			request_headers(&native_file_request, &compat)
				.iter()
				.any(|header| {
					header.name == "anthropic-beta"
						&& header
							.value
							.split(",")
							.any(|beta| beta == "files-api-2025-04-14")
				})
		);

		let mut adaptive = request(Vec::new());
		adaptive.model = "claude-sonnet-4-6".into();
		adaptive.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::High).build())
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		let (wire, _) = encoded(&adaptive, &compat);
		assert_eq!(wire["thinking"], json!({"type":"adaptive"}));
		assert_eq!(wire["output_config"]["effort"], "high");
	}

	#[test]
	fn fixture_stream_preserves_signature_usage_and_split_tool_json() {
		let expected: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/anthropic/expect.thinking_tool_usage.json"
		))
		.unwrap();
		assert_eq!(expected["outcome"]["tool_calls"][0]["arguments"]["city"], "Montréal");
		let mut decoder = SseDecoder::new();
		let frames = decoder
			.push(Bytes::from_static(include_bytes!(
				"../tests/fixtures/anthropic/stream.thinking_tool_usage.sse"
			)))
			.collect::<Vec<_>>();
		let mut state = DecodeState::default();
		let mut args = BytesMut::new();
		let mut ended_signature = None;
		let mut outcome = None;
		for frame in &frames {
			for event in AnthropicCodec::new()
				.decode(Frame::Event { name: frame.name.as_deref(), data: &frame.data }, &mut state)
				.unwrap()
			{
				match event {
					TurnEvent::PartDelta { index: 2, chunk } => args.extend_from_slice(&chunk),
					TurnEvent::PartEnd { index: 0, signature } => ended_signature = Some(signature),
					TurnEvent::Outcome(value) => outcome = Some(value),
					_ => {},
				}
			}
		}
		assert_eq!(args.as_ref(), r#"{"city":"Montréal"}"#.as_bytes());
		assert_eq!(ended_signature, Some(Bytes::from_static(b"sig_REDACTED")));
		let outcome = outcome.unwrap();
		assert_eq!(outcome.usage.as_ref().unwrap().cache_read_tokens, 7);
		assert_eq!(outcome.usage.as_ref().unwrap().cache_write_tokens, 3);
		let ItemKind::Message(message) = &outcome.output[0].kind else {
			panic!()
		};
		let Part::Thinking(thinking) = &message.parts[0] else {
			panic!()
		};
		assert_eq!(thinking.signature, Bytes::from_static(b"sig_REDACTED"));
		assert_eq!(
			outcome.output[0].props.get_ns("anthropic", "model"),
			Some(&json!("claude-sonnet-4-5"))
		);
		let ItemKind::ToolCall(call) = &outcome.output[1].kind else {
			panic!()
		};
		assert_eq!(call.args_json, Bytes::from_static(r#"{"city":"Montréal"}"#.as_bytes()));
	}

	#[test]
	fn atomic_fallback_fixture_survives_sse_fragmentation_and_precedes_terminal() {
		fn decode(chunks: impl IntoIterator<Item = Bytes>) -> Vec<TurnEvent> {
			let mut decoder = SseDecoder::new();
			let mut state = DecodeState::default();
			let mut events = Vec::new();
			for chunk in chunks {
				let frames = decoder.push(chunk).collect::<Vec<_>>();
				for frame in &frames {
					events.extend(
						AnthropicCodec::new()
							.decode(
								Frame::Event { name: frame.name.as_deref(), data: &frame.data },
								&mut state,
							)
							.unwrap(),
					);
				}
			}
			events.extend(
				AnthropicCodec::new()
					.decode(Frame::Done, &mut state)
					.unwrap(),
			);
			events
		}

		let fixture = include_bytes!("../tests/fixtures/anthropic/stream.fallback.sse");
		let whole = decode([Bytes::from_static(fixture)]);
		let fragmented = decode(fixture.chunks(7).map(Bytes::copy_from_slice));
		assert_eq!(fragmented, whole);
		assert!(matches!(whole.as_slice(), [
			TurnEvent::PartStart { index: 1, .. },
			TurnEvent::PartDelta { index: 1, .. },
			TurnEvent::PartEnd { index: 1, .. },
			TurnEvent::Outcome(_),
		]));

		let TurnEvent::Outcome(outcome) = whole.last().unwrap() else {
			panic!()
		};
		assert_eq!(
			outcome.output[0]
				.props
				.get_ns("anthropic", "server_tool_block")
				.unwrap(),
			&json!({
				"type": "fallback",
				"from": {"model": "claude-sonnet-4-5"},
				"to": {"model": "claude-opus-4-8"}
			})
		);
		let ItemKind::Message(message) = &outcome.output[1].kind else {
			panic!()
		};
		assert_eq!(message.parts, vec![Part::Text("continued".into())]);
	}

	#[test]
	fn malformed_atomic_fallback_is_an_explicit_protocol_error() {
		let mut state = DecodeState::default();
		let error = AnthropicCodec::new()
			.decode(
				Frame::Data(
					br#"{"type":"content_block_start","index":0,"content_block":{"type":"fallback","from":{"model":"claude-sonnet-4-5"},"to":{}}}"#,
				),
				&mut state,
			)
			.unwrap_err();
		assert!(
			error
				.to_string()
				.contains("malformed Anthropic fallback content block")
		);
	}

	#[test]
	fn fixture_error_and_truncated_stream_map_to_terminal_errors() {
		let fixture: Value =
			serde_json::from_str(include_str!("../tests/fixtures/anthropic/response.error_429.json"))
				.unwrap();
		let body = serde_json::to_vec(&fixture["body"]).unwrap();
		let mut state = DecodeState::default();
		let events = AnthropicCodec::new()
			.decode(Frame::Data(&body), &mut state)
			.unwrap();
		assert!(matches!(
			events.as_slice(),
			[TurnEvent::Error(error)] if error.kind == TurnErrorKind::RateLimited
		));

		let mut decoder = SseDecoder::new();
		let frames = decoder
			.push(Bytes::from_static(include_bytes!(
				"../tests/fixtures/anthropic/stream.truncated_tool.sse"
			)))
			.collect::<Vec<_>>();
		let mut state = DecodeState::default();
		for frame in &frames {
			AnthropicCodec::new()
				.decode(Frame::Event { name: frame.name.as_deref(), data: &frame.data }, &mut state)
				.unwrap();
		}
		let events = AnthropicCodec::new()
			.decode(Frame::Done, &mut state)
			.unwrap();
		assert!(matches!(
			events.as_slice(),
			[TurnEvent::Error(error)] if error.kind == TurnErrorKind::Upstream
		));
	}

	#[test]
	fn native_tool_choices_server_tools_and_structured_controls_match_fixture() {
		let fixture: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/anthropic/request.native_controls.json"
		))
		.unwrap();
		let mut request = request(vec![item(ItemKind::Message(
			Message::builder()
				.role(Role::User)
				.parts(vec![Part::Text("hello".into())])
				.build(),
		))]);
		request.tools.push(
			ToolDef::builder()
				.name("lookup".into())
				.description("Look up a value".into())
				.schema_json(Bytes::from_static(
					br#"{"type":"object","properties":{"q":{"type":"string","pattern":"^x"}}}"#,
				))
				.strict(false)
				.build(),
		);
		let mut options = Props::default();
		options.insert_ns("anthropic", "server_tools", fixture["server_tools"].clone());
		options.insert_ns("anthropic", "disable_parallel_tool_use", json!(true));
		options.insert_ns("anthropic", "context_management", fixture["context_management"].clone());
		options.insert_ns("anthropic", "service_tier", fixture["service_tier"].clone());
		options.insert_ns("anthropic", "container", fixture["container"].clone());
		options.insert_ns("anthropic", "cache_control", json!({"ttl":"1h","scope":"global"}));
		request.provider_options = Some(options);
		request.response_format = Some(
			Feature::builder()
				.value(
					ResponseFormat::builder()
						.kind(ResponseFormatKind::JsonSchema(
							JsonSchema::builder()
								.name("answer".into())
								.schema_json(Bytes::from_static(br#"{"type":"object"}"#))
								.strict(true)
								.build(),
						))
						.build(),
				)
				.on_unsupported(Fallback::Error)
				.build(),
		);
		request.sampling = Some(
			Sampling::builder()
				.stop(vec!["one".into(), "two".into(), "three".into(), "four".into(), "five".into()])
				.build(),
		);
		let mut compat = Compat::default();
		compat.forced_tool_choice = true;
		compat.tool_strict_mode = ToolStrictMode::Mixed;
		for (choice, expected) in [
			(ToolChoice::Auto, &fixture["tool_choices"]["auto"]),
			(ToolChoice::None, &fixture["tool_choices"]["none"]),
			(ToolChoice::Required, &fixture["tool_choices"]["required"]),
			(ToolChoice::Named("lookup".into()), &fixture["tool_choices"]["named"]),
		] {
			request.tool_choice = Some(
				Feature::builder()
					.value(choice)
					.on_unsupported(Fallback::Error)
					.build(),
			);
			let (wire, unsupported) = encoded(&request, &compat);
			assert_eq!(&wire["tool_choice"], expected);
			assert_eq!(wire["tools"].as_array().unwrap().len(), 6);
			assert_eq!(wire["tools"][0]["strict"], false);
			assert_eq!(wire["tools"][0]["input_schema"]["additionalProperties"], false);
			assert!(
				wire["tools"][0]["input_schema"]["properties"]["q"]
					.get("pattern")
					.is_none()
			);
			assert_eq!(
				wire["tools"][0]["input_schema"]["properties"]["q"]["description"],
				r#"{pattern: "^x"}"#
			);
			assert_eq!(wire["output_config"], fixture["output_config"]);
			assert_eq!(wire["context_management"], fixture["context_management"]);
			assert_eq!(wire["service_tier"], fixture["service_tier"]);
			assert_eq!(wire["container"], fixture["container"]);
			assert_eq!(wire["tools"][5]["cache_control"], fixture["cache_control"]);
			assert_eq!(wire["stop_sequences"], fixture["stop_sequences"]);
			assert!(unsupported.iter().any(|entry| {
				entry.what == "sampling.stop" && entry.action == UnsupportedAction::Clamped
			}));
		}
		request.tools[0].strict = Some(true);
		let (wire, _) = encoded(&request, &compat);
		assert_eq!(wire["tools"][0]["strict"], true);
	}

	#[test]
	fn tool_result_documents_use_native_document_content() {
		let fixture: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/anthropic/request.tool_result_document.json"
		))
		.unwrap();
		let result = ToolResult::builder()
			.call_id(CallId::new())
			.name("read_pdf".into())
			.parts(vec![Part::Blob(
				BlobPart::builder()
					.hash([7; 32])
					.mime("application/pdf".into())
					.size(4)
					.inline(Bytes::from_static(b"%PDF"))
					.build(),
			)])
			.is_error(false)
			.build();
		let (wire, unsupported) =
			encoded(&request(vec![item(ItemKind::ToolResult(result))]), &Compat::default());
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(wire["messages"][0]["content"][0]["content"], fixture["content"]);
	}

	#[test]
	fn server_tool_citations_context_and_terminal_are_preserved_in_order() {
		let mut decoder = SseDecoder::new();
		let frames = decoder
			.push(Bytes::from_static(include_bytes!(
				"../tests/fixtures/anthropic/stream.server_tools_citations.sse"
			)))
			.collect::<Vec<_>>();
		let mut state = DecodeState::default();
		let mut terminal_count = 0;
		let mut outcome = None;
		for frame in &frames {
			for event in AnthropicCodec::new()
				.decode(Frame::Event { name: frame.name.as_deref(), data: &frame.data }, &mut state)
				.unwrap()
			{
				if matches!(&event, TurnEvent::Outcome(_) | TurnEvent::Error(_)) {
					terminal_count += 1;
				}
				if let TurnEvent::Outcome(value) = event {
					outcome = Some(value);
				}
			}
		}
		for event in AnthropicCodec::new()
			.decode(Frame::Done, &mut state)
			.unwrap()
		{
			if matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)) {
				terminal_count += 1;
			}
		}
		assert_eq!(terminal_count, 1);
		let outcome = outcome.unwrap();
		assert_eq!(outcome.output.len(), 5);
		assert!(matches!(&outcome.output[0].kind, ItemKind::Message(_)));
		assert_eq!(
			outcome.output[1]
				.props
				.get_ns("anthropic", "server_tool_block")
				.unwrap()["type"],
			"server_tool_use"
		);
		assert_eq!(
			outcome.output[2]
				.props
				.get_ns("anthropic", "server_tool_block")
				.unwrap()["type"],
			"web_search_tool_result"
		);
		assert!(matches!(&outcome.output[3].kind, ItemKind::ToolCall(_)));
		assert!(matches!(&outcome.output[4].kind, ItemKind::Message(_)));
		assert_eq!(
			outcome.output[0]
				.props
				.get_ns("anthropic", "citations")
				.unwrap()[0]["citation"]["url"],
			"https://example.com"
		);
		assert_eq!(outcome.props.get_ns("anthropic", "container").unwrap()["id"], "container_01");
		assert_eq!(outcome.props.get_ns("anthropic", "stop_sequence"), Some(&json!("END")));
		let usage = outcome.usage.unwrap();
		assert_eq!(usage.cache_read_tokens, 4);
		assert_eq!(usage.cache_write_tokens, 3);
		assert_eq!(
			usage.detail.get_ns("anthropic", "server_tool_use").unwrap()["web_search_requests"],
			1
		);
	}
	#[test]
	fn model_policy_selects_adaptive_or_budget_effort_per_model() {
		let mut request = request(vec![item(ItemKind::Message(
			Message::builder()
				.role(Role::User)
				.parts(vec![Part::Text("reason".into())])
				.build(),
		))]);
		request.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::XHigh).build())
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		let mut compat = Compat::default();
		compat.reasoning_wire_format = ReasoningWireFormat::Anthropic;

		let mut adaptive = model_policy(ResolvedThinkingMode::AnthropicAdaptive, Some(true));
		let thinking = adaptive.thinking.as_mut().unwrap();
		thinking.effort_map.insert(Effort::XHigh, "xhigh".into());
		thinking.effort_map.insert(Effort::Max, "max".into());
		request.model_policy = Some(Arc::new(adaptive));
		let (wire, _) = encoded(&request, &compat);
		assert_eq!(wire["thinking"], json!({"type":"adaptive","display":"summarized"}));
		assert_eq!(wire["output_config"]["effort"], "xhigh");

		request.thinking.as_mut().unwrap().value.effort = Some(Effort::Max);
		let (wire, _) = encoded(&request, &compat);
		assert_eq!(wire["output_config"]["effort"], "max");

		request.thinking.as_mut().unwrap().value.effort = Some(Effort::XHigh);
		let mut budget = model_policy(ResolvedThinkingMode::AnthropicBudgetEffort, Some(false));
		let thinking = budget.thinking.as_mut().unwrap();
		thinking.effort_map.insert(Effort::XHigh, "max".into());
		thinking.effort_budgets.insert(Effort::XHigh, 12_345);
		request.model_policy = Some(Arc::new(budget));
		let (wire, _) = encoded(&request, &compat);
		assert_eq!(wire["thinking"], json!({"type":"enabled","budget_tokens":12345}));
		assert_eq!(wire["output_config"]["effort"], "max");
	}

	#[test]
	fn model_policy_controls_mid_system_and_long_cache_ttl() {
		let mut request = request(vec![
			item(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Text("first".into())])
					.build(),
			)),
			item(ItemKind::Message(
				Message::builder()
					.role(Role::System)
					.parts(vec![Part::Text("mid".into())])
					.build(),
			)),
			item(ItemKind::Message(
				Message::builder()
					.role(Role::Assistant)
					.parts(vec![Part::Text("answer".into())])
					.build(),
			)),
		]);
		request.cache = Some(
			CacheHint::builder()
				.session_key("session".into())
				.retention(CacheRetention::Long)
				.build(),
		);
		let mut policy = model_policy(ResolvedThinkingMode::Budget, None);
		set_wire_bool(&mut policy, "supports_mid_conversation_system", true);
		set_wire_bool(&mut policy, "supports_long_cache_retention", false);
		request.model_policy = Some(Arc::new(policy));
		let (wire, unsupported) = encoded(&request, &Compat::default());
		assert!(wire.get("system").is_none());
		assert_eq!(wire["messages"][1]["role"], "system");
		assert_eq!(wire["messages"][2]["content"][0]["cache_control"]["ttl"], "5m");
		assert!(
			unsupported
				.iter()
				.any(|entry| entry.what == "cache.retention")
		);
		let beta = request_headers(&request, &Compat::default())
			.into_iter()
			.find(|header| header.name == "anthropic-beta")
			.unwrap();
		assert!(beta.value.contains("mid-conversation-system-2026-04-07"));
		assert!(!beta.value.contains("extended-cache-ttl-2025-04-11"));

		let mut policy = (*request.model_policy.take().unwrap()).clone();
		set_wire_bool(&mut policy, "supports_long_cache_retention", true);
		request.model_policy = Some(Arc::new(policy));
		let (wire, _) = encoded(&request, &Compat::default());
		assert_eq!(wire["messages"][2]["content"][0]["cache_control"]["ttl"], "1h");
	}

	/// `TailTwo` marks the tail of each of the final two messages and nothing
	/// else, where the default policy spends breakpoints on system and tools
	/// and marks only one message.
	#[test]
	fn tail_two_breakpoints_replace_the_system_and_tool_markers() {
		let mut request = request(vec![
			item(ItemKind::Message(
				Message::builder()
					.role(Role::System)
					.parts(vec![Part::Text("rules".into())])
					.build(),
			)),
			item(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Text("first".into())])
					.build(),
			)),
			item(ItemKind::Message(
				Message::builder()
					.role(Role::Assistant)
					.parts(vec![Part::Text("answer".into())])
					.build(),
			)),
			item(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Text("again".into())])
					.build(),
			)),
		]);
		request.tools = vec![
			ToolDef::builder()
				.name("read".into())
				.description("Read a file".into())
				.schema_json(Bytes::from_static(br#"{"type":"object"}"#))
				.strict(false)
				.build(),
		];
		let hint = |breakpoint| {
			Some(
				CacheHint::builder()
					.session_key("session".into())
					.retention(CacheRetention::Short)
					.breakpoint(breakpoint)
					.build(),
			)
		};

		request.cache = hint(PromptCacheBreakpoint::TailTwo);
		let (wire, _) = encoded(&request, &Compat::default());
		assert!(wire["system"][0].get("cache_control").is_none(), "{wire}");
		assert!(wire["tools"][0].get("cache_control").is_none(), "{wire}");
		assert_eq!(wire["messages"][1]["content"][0]["cache_control"]["ttl"], "5m");
		assert_eq!(wire["messages"][2]["content"][0]["cache_control"]["ttl"], "5m");
		assert!(
			wire["messages"][0]["content"][0]
				.get("cache_control")
				.is_none(),
			"{wire}"
		);

		// Default placement still marks system, tools, and only the deepest message.
		request.cache = hint(PromptCacheBreakpoint::LatestStableMessage);
		let (wire, _) = encoded(&request, &Compat::default());
		assert_eq!(wire["system"][0]["cache_control"]["ttl"], "5m");
		assert_eq!(wire["tools"][0]["cache_control"]["ttl"], "5m");
		assert!(
			wire["messages"][1]["content"][0]
				.get("cache_control")
				.is_none(),
			"{wire}"
		);
		assert_eq!(wire["messages"][2]["content"][0]["cache_control"]["ttl"], "5m");

		// And `None` suppresses every marker.
		request.cache = hint(PromptCacheBreakpoint::None);
		let (wire, _) = encoded(&request, &Compat::default());
		assert!(wire["system"][0].get("cache_control").is_none(), "{wire}");
		assert!(
			wire["messages"][2]["content"][0]
				.get("cache_control")
				.is_none(),
			"{wire}"
		);
	}

	#[test]
	fn model_policy_escapes_tools_streams_eagerly_and_requires_result_ids() {
		let id = CallId::new();
		let mut request = request(vec![
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
		request.tools.push(
			ToolDef::builder()
				.name("read".into())
				.description("read".into())
				.schema_json(Bytes::from_static(br#"{"type":"object"}"#))
				.build(),
		);
		request.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::High).build())
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		let mut policy = model_policy(ResolvedThinkingMode::Budget, None);
		set_wire_bool(&mut policy, "supports_eager_tool_input_streaming", true);
		set_wire_bool(&mut policy, "requires_tool_result_id", true);
		policy
			.compat
			.insert_ns("wire", "when_thinking", json!({"escape_builtin_tool_names": true}));
		request.model_policy = Some(Arc::new(policy));
		let mut compat = Compat::default();
		compat.reasoning_wire_format = ReasoningWireFormat::Anthropic;
		let (wire, _) = encoded(&request, &compat);
		assert_eq!(wire["tools"][0]["name"], "_read");
		assert_eq!(wire["tools"][0]["eager_input_streaming"], true);
		assert_eq!(wire["messages"][0]["content"][0]["name"], "_read");
		assert_eq!(
			wire["messages"][1]["content"][0]["id"],
			wire["messages"][1]["content"][0]["tool_use_id"]
		);

		request.thinking = None;
		let (wire, _) = encoded(&request, &compat);
		assert_eq!(wire["tools"][0]["name"], "read");
	}

	#[test]
	fn replay_unsigned_policy_never_replays_across_models() {
		let mut props = Props::default();
		props.insert_ns("anthropic", "model", json!("source-model"));
		let thinking = Thinking::builder()
			.text("private".into())
			.signature(Bytes::new())
			.redacted(false)
			.build();
		let mut request = request(vec![
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::Assistant)
						.parts(vec![Part::Thinking(thinking)])
						.build(),
				))
				.props(props)
				.build(),
		]);
		request.model = "target-model".into();
		let mut policy = model_policy(ResolvedThinkingMode::Budget, None);
		set_wire_bool(&mut policy, "replay_unsigned_thinking", true);
		set_wire_bool(&mut policy, "signing_endpoint", false);
		request.model_policy = Some(Arc::new(policy));
		let mut compat = Compat::default();
		compat.reasoning_wire_format = ReasoningWireFormat::Anthropic;
		let (wire, _) = encoded(&request, &compat);
		assert_eq!(wire["messages"][0]["content"][0]["type"], "text");
	}
	#[test]
	fn document_sources_match_anthropic_base64_url_and_file_shapes() {
		let fixture: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/anthropic/request.media_projection.json"
		))
		.unwrap();
		let pdf = |mime: &str, size: u64, inline: Bytes| {
			BlobPart::builder()
				.hash([7; 32])
				.mime(mime.into())
				.size(size)
				.inline(inline)
				.build()
		};

		let inline = item(ItemKind::Message(
			Message::builder()
				.role(Role::User)
				.parts(vec![Part::Blob(pdf(" Application/PDF ", 4, Bytes::from_static(b"%PDF")))])
				.build(),
		));
		let (wire, unsupported) = encoded(&request(vec![inline]), &Compat::default());
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(wire["messages"][0]["content"][0], fixture["documents"]["base64"]);

		let mut mixed = item(ItemKind::Message(
			Message::builder()
				.role(Role::User)
				.parts(vec![
					Part::Blob(pdf("application/pdf", 8192, Bytes::new())),
					Part::Blob(pdf("image/png", 4096, Bytes::new())),
					Part::Blob(pdf("application/pdf", 16384, Bytes::new())),
				])
				.build(),
		));
		mixed.props.insert_ns(
			"anthropic",
			"document_sources",
			json!([
				{"type":"url","url":"https://example.test/manual.pdf"},
				{"type":"file","file_id":"file_012345"}
			]),
		);
		mixed.props.insert_ns(
			"anthropic",
			"image_sources",
			json!([{"type":"file","file_id":"file_image012345"}]),
		);
		let mut remote_request = request(vec![mixed]);
		let (wire, unsupported) = encoded(&remote_request, &Compat::default());
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(wire["messages"][0]["content"][0], fixture["documents"]["url"]);
		assert_eq!(
			wire["messages"][0]["content"][1]["source"],
			json!({"type":"file","file_id":"file_image012345"})
		);
		assert_eq!(wire["messages"][0]["content"][2], fixture["documents"]["file"]);
		remote_request.cache = Some(
			CacheHint::builder()
				.session_key("document-sources".into())
				.retention(CacheRetention::Long)
				.build(),
		);
		let (wire, unsupported) = encoded(&remote_request, &Compat::default());
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(
			wire["messages"][0]["content"][2]["cache_control"],
			json!({"type":"ephemeral","ttl":"1h"})
		);
		assert!(
			request_headers(&remote_request, &Compat::default())
				.iter()
				.any(|header| {
					header.name == "anthropic-beta"
						&& header
							.value
							.split(",")
							.any(|beta| beta == "files-api-2025-04-14")
				})
		);
	}

	#[test]
	fn malformed_or_ambiguous_document_sources_fail_explicitly() {
		let request_with_sources = |sources: Value| {
			let pdf = BlobPart::builder()
				.hash([8; 32])
				.mime("application/pdf".into())
				.size(1024)
				.inline(Bytes::new())
				.build();
			let mut item = item(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Blob(pdf)])
					.build(),
			));
			item
				.props
				.insert_ns("anthropic", "document_sources", sources);
			request(vec![item])
		};
		let assert_provider_error = |request: ChatRequest, needle: &str| {
			let error = AnthropicCodec::new()
				.encode(&request, &Compat::default())
				.expect_err("malformed document source must fail");
			let omp_llm_types::Error::Provider(detail) = error else {
				panic!("expected provider error, got {error:?}");
			};
			assert!(detail.contains(needle), "unexpected provider error: {detail}");
		};

		assert_provider_error(
			request_with_sources(json!([{
				"type":"url",
				"url":"https://example.test/manual.pdf",
				"file_id":"file_conflict"
			}])),
			"contain exactly",
		);
		assert_provider_error(
			request_with_sources(json!([{"type":"url","url":"/relative/manual.pdf"}])),
			"absolute HTTP(S) URL",
		);
		assert_provider_error(
			request_with_sources(json!([{"type":"file","file_id":"not-an-anthropic-file"}])),
			"`file_` identifier",
		);
		assert_provider_error(
			request_with_sources(json!([
				{"type":"file","file_id":"file_first"},
				{"type":"file","file_id":"file_extra"}
			])),
			"more entries",
		);
		assert_provider_error(
			request_with_sources(json!([{"type":"base64","url":"https://example.test/manual.pdf"}])),
			"contain exactly",
		);

		let mut conflicting = request_with_sources(json!([{
			"type":"url",
			"url":"https://example.test/manual.pdf"
		}]));
		conflicting.thread.items[0].props.insert_ns(
			"anthropic",
			"image_sources",
			json!([{"type":"file","file_id":"file_wrong_kind"}]),
		);
		assert_provider_error(conflicting, "more entries");

		let unresolved_inline = request_with_sources(json!([{"type":"base64"}]));
		assert_provider_error(unresolved_inline, "non-empty resolved inline bytes");
	}
	#[test]
	fn claude_oauth_tool_names_round_trip_without_touching_builtins() {
		let call = |name: &str| {
			item(ItemKind::ToolCall(
				ToolCall::builder()
					.id(CallId::new())
					.name(name.into())
					.args_json(Bytes::from_static(b"{}"))
					.thought_signature(Bytes::new())
					.build(),
			))
		};
		let mut request = request(vec![call("foo"), call("_foo"), call("web_search")]);
		for name in ["foo", "_foo", "web_search", "code_execution", "text_editor", "computer"] {
			request.tools.push(
				ToolDef::builder()
					.name(name.into())
					.description("test".into())
					.schema_json(Bytes::from_static(br#"{"type":"object"}"#))
					.build(),
			);
		}
		request.tool_choice = Some(
			Feature::builder()
				.value(ToolChoice::Named("foo".into()))
				.on_unsupported(Fallback::Error)
				.build(),
		);
		let mut compat = Compat::default();
		compat.forced_tool_choice = true;

		let (api_body, _) = AnthropicCodec::new().encode(&request, &compat).unwrap();
		let api: Value = serde_json::from_slice(&api_body).unwrap();
		assert_eq!(api["tools"][0]["name"], "foo");
		assert_eq!(api["tools"][1]["name"], "_foo");
		assert_eq!(api["tool_choice"]["name"], "foo");
		assert_eq!(api["messages"][0]["content"][0]["name"], "foo");
		assert_eq!(api["messages"][0]["content"][1]["name"], "_foo");

		let oauth = AnthropicCodec::claude_oauth();
		let (oauth_body, _) = oauth.encode(&request, &compat).unwrap();
		let wire: Value = serde_json::from_slice(&oauth_body).unwrap();
		let definitions = wire["tools"].as_array().unwrap();
		let names = definitions
			.iter()
			.map(|tool| tool["name"].as_str().unwrap())
			.collect::<Vec<_>>();
		assert_eq!(names, [
			"_foo",
			"__foo",
			"web_search",
			"code_execution",
			"text_editor",
			"computer"
		]);
		assert_eq!(wire["tool_choice"]["name"], "_foo");
		let replay = wire["messages"][0]["content"].as_array().unwrap();
		assert_eq!(replay[0]["name"], "_foo");
		assert_eq!(replay[1]["name"], "__foo");
		assert_eq!(replay[2]["name"], "web_search");

		let decoded_name = |wire_name: &str| {
			let body = format!(
				r#"{{"type":"content_block_start","index":0,"content_block":{{"type":"tool_use","id":"toolu_01","name":"{wire_name}","input":{{}}}}}}"#
			);
			let mut state = DecodeState::default();
			let events = oauth
				.decode(Frame::Data(body.as_bytes()), &mut state)
				.unwrap();
			match &events[0] {
				TurnEvent::PartStart { tool_name, .. } => tool_name.clone(),
				event => panic!("expected tool-call start, got {event:?}"),
			}
		};
		assert_eq!(decoded_name("_foo"), "foo");
		assert_eq!(decoded_name("__foo"), "_foo");
		assert_eq!(decoded_name("web_search"), "web_search");
		assert_eq!(decoded_name("code_execution"), "code_execution");
		assert_eq!(decoded_name("text_editor"), "text_editor");
		assert_eq!(decoded_name("computer"), "computer");
		for (canonical, wire_name) in ["foo", "_foo"].into_iter().zip(&names[..2]) {
			assert_eq!(decoded_name(wire_name), canonical);
		}
	}
}
