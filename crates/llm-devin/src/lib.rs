//! Devin Cascade Connect server-streaming transport codec.
//!
//! Devin is an ordinary request/server-stream turn: tool calls finish the turn
//! and are answered in a later request. It has no in-turn invocation channel.

use std::{collections::BTreeMap, error::Error as StdError, future::Future, sync::Arc};

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use omp_core::{Str, base64};
use omp_llm_types::{
	Accuracy, CallId, CallIdMapper, Chat, ChatOutcome, ChatRequest, Error, Executor, Fallback, Item,
	ItemKind, Message, Part, Props, Role, StopReason, StreamPartKind, Thinking, ToolCall,
	ToolCallIdProfile, ToolChoice, TurnError, TurnErrorKind, TurnEvent, Unsupported,
	UnsupportedAction, Usage,
};
use prost::Message as ProstMessage;
use smallvec::SmallVec;
use tonic::transport::Channel;

pub mod wire;

use wire::{
	CacheControlType, ChatMessagePrompt, ChatMessageRequestType, ChatMessageSource, ChatToolCall,
	ChatToolChoice, ChatToolDefinition, CompletionConfiguration, ConversationalPlannerMode,
	GetChatMessageRequest, GetChatMessageResponse, GetCliModelConfigsRequest,
	GetCliModelConfigsResponse, ImageData, Metadata, PromptCacheOptions,
	StopReason as WireStopReason, chat_tool_choice,
};

const DEFAULT_STOP_PATTERNS: [&str; 5] =
	["<|user|>", "<|bot|>", "<|context_request|>", "<|endoftext|>", "<|end_of_turn|>"];
/// Non-secret model metadata returned by Devin model discovery.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveredModel {
	/// Provider-local model uid.
	pub id:                Str,
	/// Devin display label.
	pub name:              Str,
	/// Whether the model accepts image input.
	pub supports_images:   bool,
	/// Whether the model exposes thinking behavior.
	pub reasoning:         bool,
	/// Advertised input context window.
	pub context_window:    u64,
	/// Advertised maximum output tokens.
	pub max_output_tokens: u64,
}

/// Encodes Devin's pinned `GetCliModelConfigs` protobuf request.
#[must_use]
pub fn model_discovery_request(api_key: &[u8]) -> Bytes {
	let token = String::from_utf8_lossy(api_key);
	let api_key = if token.starts_with("devin-session-token$") {
		token.into_owned()
	} else {
		format!("devin-session-token${token}")
	};
	Bytes::from(
		GetCliModelConfigsRequest {
			metadata: Some(Metadata {
				ide_name: "windsurf".into(),
				ide_version: "3.2.23".into(),
				extension_name: "windsurf".into(),
				extension_version: "1.48.2".into(),
				api_key,
				..Metadata::default()
			}),
		}
		.encode_to_vec(),
	)
}

/// Decodes Devin's `GetCliModelConfigs` protobuf response.
///
/// # Errors
///
/// Returns a transport error when the response is malformed.
pub fn decode_model_discovery(payload: &[u8]) -> Result<Vec<DiscoveredModel>, Error> {
	let response = GetCliModelConfigsResponse::decode(payload)
		.map_err(|error| Error::Transport(Str::from(error.to_string())))?;
	let mut models = BTreeMap::new();
	for config in response.client_model_configs {
		if config.disabled || config.model_uid.trim().is_empty() {
			continue;
		}
		let features = config
			.model_info
			.as_ref()
			.and_then(|info| info.model_features.as_ref());
		let lower_label = config.label.to_ascii_lowercase();
		let reasoning = features.is_some_and(|features| features.supports_thinking)
			|| (!lower_label.contains("no thinking")
				&& ["think", "minimal", "high", "medium", "low", "xhigh", "max", "reasoning"]
					.iter()
					.any(|term| lower_label.contains(term)));
		let context_window = u64::try_from(config.max_tokens)
			.ok()
			.filter(|value| *value > 0)
			.unwrap_or(200_000);
		let max_output_tokens = config
			.model_info
			.as_ref()
			.and_then(|info| u64::try_from(info.max_output_tokens).ok())
			.filter(|value| *value > 0)
			.unwrap_or_else(|| context_window.min(64_000));
		let id = Str::from(config.model_uid.trim());
		let name = if config.label.trim().is_empty() {
			id.clone()
		} else {
			Str::from(config.label.trim())
		};
		models.insert(id.clone(), DiscoveredModel {
			id,
			name,
			supports_images: config.supports_images
				|| features.is_some_and(|features| features.supports_images),
			reasoning,
			context_window,
			max_output_tokens,
		});
	}
	Ok(models.into_values().collect())
}

/// Applies sealed Devin authentication directly to one protobuf request.
///
/// Implementations may perform Devin's `GetUserJwt` exchange and replace the
/// channel when the account selects a custom API server. The mutation-only
/// contract never returns credential bytes to application or gateway code.
pub trait DevinAuth: Send + Sync + 'static {
	/// Authentication failure reported without exposing credential material.
	type Error: StdError + Send + Sync + 'static;

	/// Authenticates one request and resolves its account-specific channel.
	fn apply<'a>(
		&'a self,
		channel: &'a mut Channel,
		request: &'a mut GetChatMessageRequest,
	) -> impl Future<Output = Result<(), Self::Error>> + Send + 'a;
}

/// A Devin Cascade client over its Connect-compatible service endpoint.
///
/// Authentication is embedded in each protobuf request, matching Cascade's
/// wire contract. The client owns request sending and response
/// server-streaming; there is deliberately no executor or invocation channel.
#[derive(Clone)]
pub struct DevinChat<A> {
	channel: Channel,
	auth:    A,
}

impl<A> DevinChat<A> {
	/// Creates a Devin chat facet from a provider channel and sealed auth
	/// applier.
	#[must_use]
	pub const fn new(channel: Channel, auth: A) -> Self {
		Self { channel, auth }
	}
}

#[async_trait]
impl<A> Chat for DevinChat<A>
where
	A: DevinAuth,
{
	async fn turn(
		&self,
		request: ChatRequest,
		_executor: Option<Arc<dyn Executor>>,
	) -> Result<BoxStream<'static, TurnEvent>, Error> {
		let model = request.model.clone();
		let (mut wire, unsupported) = build_request(&request)?;
		let mut channel = self.channel.clone();
		self
			.auth
			.apply(&mut channel, &mut wire)
			.await
			.map_err(|_| Error::Transport(Str::new_static("Devin authentication failed")))?;

		let mut grpc = tonic::client::Grpc::new(channel);
		grpc
			.ready()
			.await
			.map_err(|status| Error::Provider(Str::from(status.to_string())))?;

		let path =
			http::uri::PathAndQuery::from_static("/exa.api_server_pb.ApiServerService/GetChatMessage");
		let codec =
			tonic_prost::ProstCodec::<GetChatMessageRequest, GetChatMessageResponse>::default();

		let mut stream = grpc
			.server_streaming(tonic::Request::new(wire), path, codec)
			.await
			.map_err(|status| Error::Provider(Str::from(status.to_string())))?
			.into_inner();

		Ok(Box::pin(async_stream::stream! {
			let mut state = State {
				model,
				unsupported,
				..State::default()
			};
			loop {
				match stream.message().await {
					Ok(Some(response)) => {
						for event in decode_response(response, &mut state) {
							yield event;
						}
					}
					Ok(None) => {
						for event in finish(&mut state) {
							yield event;
						}
						break;
					}
					Err(status) => {
						yield TurnEvent::Error(
							TurnError::builder()
								.kind(TurnErrorKind::Upstream)
								.detail(Str::from(status.to_string()))
								.unsupported(state.unsupported.clone())
								.retry_after_ms(0)
								.build(),
						);
						break;
					}
				}
			}
		}))
	}
}

/// Mutable decoder state exposed as the pin-test entry point.
///
/// This is deliberately transport-independent so recorded frames can be
/// replayed without a connection.
#[derive(Default)]
pub struct State {
	next_index:     u32,
	open_text:      Option<u32>,
	open_thinking:  Option<u32>,
	parts:          BTreeMap<u32, OutputPart>,
	tools:          BTreeMap<Str, ToolState>,
	active_tool_id: Option<Str>,
	latest_stop:    i32,
	usage:          Option<Usage>,
	model:          Str,
	unsupported:    Vec<Unsupported>,
	finished:       bool,
}

enum OutputPart {
	Text(String),
	Thinking { text: String, signature: Bytes },
	Tool(Str),
}

struct ToolState {
	canonical_id: CallId,
	index:        u32,
	name:         Str,
	arguments:    String,
}

fn build_request(req: &ChatRequest) -> Result<(GetChatMessageRequest, Vec<Unsupported>), Error> {
	let mut unsupported = Vec::new();
	let mapper = CallIdMapper::new();
	let cascade_id = req
		.meta
		.as_ref()
		.map(|meta| meta.session_id.as_str())
		.filter(|value| !value.is_empty())
		.unwrap_or("omp");
	let mut system = String::new();
	let mut prompts = Vec::new();

	for (index, item) in req.thread.items.iter().enumerate() {
		match &item.kind {
			ItemKind::Message(message) if message.role == Role::System => {
				append_message_text(&mut system, message, &mut unsupported, index);
			},
			ItemKind::Message(message) => {
				prompts.push(message_prompt(message, cascade_id, index, &mut unsupported));
			},
			ItemKind::ToolCall(call) => {
				let wire_call = ChatToolCall {
					id:             mapper
						.to_wire(&call.id, ToolCallIdProfile::Preserve)
						.to_string(),
					name:           call.name.to_string(),
					arguments_json: std::str::from_utf8(&call.args_json)
						.map_err(|_| {
							Error::Provider(Str::from("Devin tool arguments are not UTF-8"))
						})?
						.to_owned(),
				};
				if let Some(prompt) = prompts
					.last_mut()
					.filter(|prompt| prompt.source == ChatMessageSource::System as i32)
				{
					prompt.tool_calls.push(wire_call);
				} else {
					prompts.push(ChatMessagePrompt {
						message_id: format!("bot-{cascade_id}-{index}"),
						source: ChatMessageSource::System as i32,
						tool_calls: vec![wire_call],
						..Default::default()
					});
				}
			},
			ItemKind::ToolResult(result) => {
				let mut prompt = String::new();
				let mut images = Vec::new();
				append_parts(&mut prompt, &mut images, &result.parts, &mut unsupported, index);
				prompts.push(ChatMessagePrompt {
					message_id: format!("{cascade_id}-{index}-tool"),
					source: ChatMessageSource::Tool as i32,
					prompt,
					tool_call_id: mapper
						.to_wire(&result.call_id, ToolCallIdProfile::Preserve)
						.to_string(),
					tool_result_is_error: result.is_error,
					images,
					..Default::default()
				});
			},
			_ => {},
		}
	}

	let sampling = req.sampling.as_ref();
	let mut stop_patterns = DEFAULT_STOP_PATTERNS
		.iter()
		.map(ToString::to_string)
		.collect::<Vec<_>>();
	if let Some(extra) = sampling.and_then(|sampling| sampling.stop.as_ref()) {
		stop_patterns.extend(extra.iter().map(ToString::to_string));
	}
	let temperature = sampling
		.and_then(|sampling| sampling.temperature)
		.unwrap_or(0.4);
	let mut configuration = CompletionConfiguration {
		num_completions: 1,
		max_tokens: sampling
			.and_then(|sampling| sampling.max_output_tokens)
			.unwrap_or(64_000),
		max_newlines: 200,
		temperature,
		first_temperature: temperature,
		top_k: u64::from(sampling.and_then(|sampling| sampling.top_k).unwrap_or(50)),
		top_p: sampling.and_then(|sampling| sampling.top_p).unwrap_or(1.0),
		stop_patterns,
		fim_eot_prob_threshold: 1.0,
	};
	if let Some(sampling) = sampling {
		if sampling.min_p.is_some()
			|| sampling.frequency_penalty.is_some()
			|| sampling.presence_penalty.is_some()
		{
			unsupported.push(dropped("sampling", "Devin does not expose min-p or token penalties"));
		}
		if configuration.top_k == 0 {
			configuration.top_k = 50;
		}
	}
	if req.thinking.is_some() {
		unsupported
			.push(dropped("thinking", "Devin selects reasoning behavior through its model uid"));
	}
	if req.response_format.is_some() {
		unsupported.push(dropped(
			"response_format",
			"Devin Cascade does not expose structured output constraints",
		));
	}
	if req.cache.is_some() {
		unsupported.push(dropped("cache", "Devin controls prompt caching internally"));
	}
	if let Some(provider_options) = &req.provider_options {
		for key in provider_options.0.keys() {
			unsupported.push(dropped(key.clone(), "unknown Devin extension property"));
		}
	}

	let choice = req.tool_choice.as_ref().map_or_else(
		|| chat_tool_choice::Choice::OptionName("auto".to_owned()),
		|feature| match &feature.value {
			ToolChoice::Auto => chat_tool_choice::Choice::OptionName("auto".to_owned()),
			ToolChoice::Named(name) => chat_tool_choice::Choice::ToolName(name.to_string()),
			ToolChoice::None | ToolChoice::Required | _ => {
				unsupported.push(dropped("tool_choice", "Devin supports auto or a named tool choice"));
				chat_tool_choice::Choice::OptionName("auto".to_owned())
			},
		},
	);
	if req
		.thinking
		.as_ref()
		.is_some_and(|feature| feature.on_unsupported == Fallback::Error)
		|| req
			.response_format
			.as_ref()
			.is_some_and(|feature| feature.on_unsupported == Fallback::Error)
		|| req.tool_choice.as_ref().is_some_and(|feature| {
			matches!(&feature.value, ToolChoice::None | ToolChoice::Required)
				&& feature.on_unsupported == Fallback::Error
		}) {
		return Err(Error::Unsupported(unsupported));
	}
	let tools = req
		.tools
		.iter()
		.map(|tool| {
			Ok(ChatToolDefinition {
				name:               tool.name.to_string(),
				description:        tool.description.to_string(),
				json_schema_string: std::str::from_utf8(&tool.schema_json)
					.map_err(|_| Error::Provider(Str::from("Devin tool schema is not UTF-8")))?
					.to_owned(),
				strict:             tool.strict.unwrap_or(false),
			})
		})
		.collect::<Result<Vec<_>, Error>>()?;

	Ok((
		GetChatMessageRequest {
			prompt: system,
			chat_message_prompts: prompts,
			chat_model_uid: req.model.to_string(),
			request_type: ChatMessageRequestType::Cascade as i32,
			configuration: Some(configuration),
			tools,
			disable_parallel_tool_calls: true,
			tool_choice: Some(ChatToolChoice { choice: Some(choice) }),
			system_prompt_cache_options: Some(PromptCacheOptions {
				r#type: CacheControlType::Ephemeral as i32,
			}),
			cascade_id: cascade_id.to_owned(),
			execution_id: format!("{cascade_id}-turn"),
			planner_mode: ConversationalPlannerMode::Default as i32,
			..Default::default()
		},
		unsupported,
	))
}

fn message_prompt(
	message: &Message,
	cascade_id: &str,
	index: usize,
	unsupported: &mut Vec<Unsupported>,
) -> ChatMessagePrompt {
	let mut prompt = String::new();
	let mut thinking = String::new();
	let mut signature = String::new();
	let mut images = Vec::new();
	for part in &message.parts {
		match part {
			Part::Text(text) => prompt.push_str(text),
			Part::Thinking(value) => {
				thinking.push_str(&value.text);
				if signature.is_empty() {
					if let Ok(value) = std::str::from_utf8(&value.signature) {
						signature.push_str(value);
					} else {
						unsupported.push(dropped(
							format!("thread.items[{index}].thinking.signature"),
							"Devin thinking signatures must be UTF-8",
						));
					}
				}
			},
			Part::Blob(blob) if !blob.inline.is_empty() => images.push(ImageData {
				base64_data: base64::encode(&blob.inline).into_string(),
				mime_type:   blob.mime.to_string(),
			}),
			Part::Blob(_) | _ => unsupported.push(dropped(
				format!("thread.items[{index}].blob"),
				"blob is not available inline at the codec edge",
			)),
		}
	}
	ChatMessagePrompt {
		message_id: format!("{cascade_id}-{index}"),
		source: match message.role {
			Role::User => ChatMessageSource::User as i32,
			Role::Assistant => ChatMessageSource::System as i32,
			Role::System | _ => ChatMessageSource::SystemPrompt as i32,
		},
		prompt,
		images,
		thinking,
		signature,
		..Default::default()
	}
}

fn append_message_text(
	output: &mut String,
	message: &Message,
	unsupported: &mut Vec<Unsupported>,
	index: usize,
) {
	if !output.is_empty() {
		output.push_str("\n\n");
	}
	let mut ignored_images = Vec::new();
	append_parts(output, &mut ignored_images, &message.parts, unsupported, index);
	if !ignored_images.is_empty() {
		unsupported.push(dropped(
			format!("thread.items[{index}].blob"),
			"system prompt images are unsupported",
		));
	}
}

fn append_parts(
	text: &mut String,
	images: &mut Vec<ImageData>,
	parts: &[Part],
	unsupported: &mut Vec<Unsupported>,
	index: usize,
) {
	for part in parts {
		match part {
			Part::Text(value) => text.push_str(value),
			Part::Thinking(value) => text.push_str(&value.text),
			Part::Blob(blob) if !blob.inline.is_empty() => images.push(ImageData {
				base64_data: base64::encode(&blob.inline).into_string(),
				mime_type:   blob.mime.to_string(),
			}),
			Part::Blob(_) | _ => unsupported.push(dropped(
				format!("thread.items[{index}].blob"),
				"blob is not available inline at the codec edge",
			)),
		}
	}
}

fn dropped(what: impl Into<Str>, detail: impl Into<Str>) -> Unsupported {
	Unsupported::builder()
		.what(what.into())
		.detail(detail.into())
		.action(UnsupportedAction::Dropped)
		.build()
}

/// Decodes one message delivered by the generated server stream.
///
/// This is the deliberately transport-independent pin-test entry point, so
/// recorded frames can be replayed without a live connection.
pub fn decode_response(
	mut response: GetChatMessageResponse,
	state: &mut State,
) -> SmallVec<TurnEvent, 2> {
	let mut events = SmallVec::new();
	if !response.delta_thinking.is_empty() {
		close_text(state, &mut events);
		let index = ensure_text_part(state, &mut events, true);
		if let Some(OutputPart::Thinking { text, signature }) = state.parts.get_mut(&index) {
			text.push_str(&response.delta_thinking);
			if !response.delta_signature.is_empty() {
				*signature = Bytes::from(std::mem::take(&mut response.delta_signature));
			}
		}
		events.push(TurnEvent::PartDelta { index, chunk: Bytes::from(response.delta_thinking) });
	}
	if !response.delta_text.is_empty() {
		close_thinking(state, &mut events);
		let index = ensure_text_part(state, &mut events, false);
		if let Some(OutputPart::Text(text)) = state.parts.get_mut(&index) {
			text.push_str(&response.delta_text);
		}
		events.push(TurnEvent::PartDelta { index, chunk: Bytes::from(response.delta_text) });
	}
	if !response.delta_tool_calls.is_empty() {
		close_text(state, &mut events);
		close_thinking(state, &mut events);
		for call in response.delta_tool_calls {
			decode_tool_delta(call, state, &mut events);
		}
	}
	if response.stop_reason != WireStopReason::Unspecified as i32 {
		state.latest_stop = response.stop_reason;
	}
	if let Some(usage) = response.usage {
		state.usage = Some(
			Usage::builder()
				.input_tokens(usage.input_tokens)
				.output_tokens(usage.output_tokens)
				.cache_read_tokens(usage.cache_read_tokens)
				.cache_write_tokens(usage.cache_write_tokens)
				.accuracy(Accuracy::Exact)
				.detail(Props::default())
				.build(),
		);
	}
	events
}

fn ensure_text_part(state: &mut State, events: &mut SmallVec<TurnEvent, 2>, thinking: bool) -> u32 {
	let current = if thinking {
		state.open_thinking
	} else {
		state.open_text
	};
	if let Some(index) = current {
		return index;
	}
	let index = state.next_index;
	state.next_index += 1;
	if thinking {
		state.open_thinking = Some(index);
		state
			.parts
			.insert(index, OutputPart::Thinking { text: String::new(), signature: Bytes::new() });
	} else {
		state.open_text = Some(index);
		state.parts.insert(index, OutputPart::Text(String::new()));
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
}

fn decode_tool_delta(
	mut call: ChatToolCall,
	state: &mut State,
	events: &mut SmallVec<TurnEvent, 2>,
) {
	let wire_id = if call.id.is_empty() {
		state.active_tool_id.clone()
	} else {
		Some(Str::from(std::mem::take(&mut call.id)))
	};
	let Some(wire_id) = wire_id else {
		return;
	};
	if !state.tools.contains_key(&wire_id) {
		let canonical_id = CallId::new();
		let index = state.next_index;
		state.next_index += 1;
		let name = Str::from(std::mem::take(&mut call.name));
		state.parts.insert(index, OutputPart::Tool(wire_id.clone()));
		state.tools.insert(wire_id.clone(), ToolState {
			canonical_id,
			index,
			name: name.clone(),
			arguments: String::new(),
		});
		events.push(TurnEvent::PartStart {
			index,
			kind: StreamPartKind::ToolCall,
			tool_call_id: Str::from(canonical_id.to_string()),
			tool_name: name,
		});
	}
	state.active_tool_id = Some(wire_id.clone());
	let tool = state.tools.get_mut(&wire_id).expect("inserted above");
	if !call.name.is_empty() {
		tool.name = Str::from(std::mem::take(&mut call.name));
	}
	if call.arguments_json.is_empty() {
		return;
	}
	let previous_len = tool.arguments.len();
	let delta = if call.arguments_json.starts_with(&tool.arguments) {
		tool.arguments = call.arguments_json;
		Bytes::copy_from_slice(&tool.arguments.as_bytes()[previous_len..])
	} else {
		tool.arguments.push_str(&call.arguments_json);
		Bytes::from(call.arguments_json)
	};
	events.push(TurnEvent::PartDelta { index: tool.index, chunk: delta });
}

fn close_text(state: &mut State, events: &mut SmallVec<TurnEvent, 2>) {
	if let Some(index) = state.open_text.take() {
		events.push(TurnEvent::PartEnd { index, signature: Default::default() });
	}
}

fn close_thinking(state: &mut State, events: &mut SmallVec<TurnEvent, 2>) {
	if let Some(index) = state.open_thinking.take() {
		events.push(TurnEvent::PartEnd { index, signature: Default::default() });
	}
}

/// Finishes the deliberately transport-independent pin-test decoder after all
/// recorded frames have been replayed without a live connection.
pub fn finish(state: &mut State) -> SmallVec<TurnEvent, 2> {
	if state.finished {
		return SmallVec::new();
	}
	state.finished = true;
	let mut events = SmallVec::new();
	close_text(state, &mut events);
	close_thinking(state, &mut events);
	for tool in state.tools.values() {
		events.push(TurnEvent::PartEnd { index: tool.index, signature: Default::default() });
	}
	let mut output = Vec::new();
	let mut message_parts = Vec::new();
	for part in state.parts.values_mut() {
		match part {
			OutputPart::Text(text) => {
				message_parts.push(Part::Text(Str::from(std::mem::take(text))));
			},
			OutputPart::Thinking { text, signature } => {
				message_parts.push(Part::Thinking(
					Thinking::builder()
						.text(Str::from(std::mem::take(text)))
						.signature(std::mem::take(signature))
						.redacted(false)
						.build(),
				));
			},
			OutputPart::Tool(wire_id) => {
				if !message_parts.is_empty() {
					output.push(assistant_item(std::mem::take(&mut message_parts)));
				}
				let tool = state
					.tools
					.get_mut(wire_id)
					.expect("output tool references state");
				output.push(
					Item::builder()
						.seq(0)
						.kind(ItemKind::ToolCall(
							ToolCall::builder()
								.id(tool.canonical_id)
								.name(std::mem::take(&mut tool.name))
								.args_json(Bytes::from(std::mem::take(&mut tool.arguments)))
								.thought_signature(Bytes::new())
								.build(),
						))
						.props(Props::default())
						.build(),
				);
			},
		}
	}
	if !message_parts.is_empty() {
		output.push(assistant_item(message_parts));
	}
	let stop = if !state.tools.is_empty() {
		StopReason::ToolUse
	} else if state.latest_stop == WireStopReason::MaxTokens as i32 {
		StopReason::MaxTokens
	} else {
		StopReason::EndTurn
	};
	events.push(TurnEvent::Outcome(
		ChatOutcome::builder()
			.output(output)
			.stop(stop)
			.unsupported(std::mem::take(&mut state.unsupported))
			.provider(Str::from("devin"))
			.model(std::mem::take(&mut state.model))
			.props(Props::default())
			.maybe_usage(state.usage.take())
			.build(),
	));
	events
}

fn assistant_item(parts: Vec<Part>) -> Item {
	Item::builder()
		.seq(0)
		.kind(ItemKind::Message(
			Message::builder()
				.role(Role::Assistant)
				.parts(parts)
				.build(),
		))
		.props(Props::default())
		.build()
}

#[cfg(test)]
mod tests {
	use bytes::Bytes;
	use omp_core::Str;
	use omp_llm_types::{
		ChatRequest, Item, ItemKind, Message, Part, Props, Role, StopReason, Thread, TurnEvent,
	};
	use prost::Message as ProstMessage;

	use super::{
		GetChatMessageResponse, State, WireStopReason, build_request, decode_model_discovery,
		decode_response, finish,
	};
	use crate::wire::{
		ChatToolCall, ClientModelConfig, GetCliModelConfigsResponse, ModelFeatures, ModelInfo,
		ModelUsageStats,
	};

	fn accumulated(chunks: &[&str]) -> Bytes {
		let mut state = State::default();
		let mut arguments = Bytes::new();
		for chunk in chunks {
			decode_response(
				GetChatMessageResponse {
					delta_tool_calls: vec![ChatToolCall {
						id:             "wire-call".to_owned(),
						name:           "task".to_owned(),
						arguments_json: (*chunk).to_owned(),
					}],
					..Default::default()
				},
				&mut state,
			);
		}
		for event in finish(&mut state) {
			if let TurnEvent::Outcome(outcome) = event {
				for item in outcome.output {
					if let ItemKind::ToolCall(call) = item.kind {
						arguments = call.args_json;
					}
				}
			}
		}
		arguments
	}

	#[test]
	fn plain_text_fixture_maps_to_one_user_prompt() {
		let request = ChatRequest::builder()
			.model(Str::from("devin-test"))
			.thread(
				Thread::builder()
					.items(vec![
						Item::builder()
							.seq(0)
							.kind(ItemKind::Message(
								Message::builder()
									.role(Role::User)
									.parts(vec![Part::Text(Str::from("Hello"))])
									.build(),
							))
							.props(Props::default())
							.build(),
					])
					.build(),
			)
			.tools(Vec::new())
			.provider_options(Props::default())
			.build();
		let (wire, unsupported) = build_request(&request).unwrap();
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(wire.chat_model_uid, "devin-test");
		assert_eq!(wire.chat_message_prompts.len(), 1);
		assert_eq!(wire.chat_message_prompts[0].prompt, "Hello");
	}

	#[test]
	fn cumulative_and_incremental_arguments_are_byte_identical() {
		let incremental = accumulated(&[r#"{"a":"#, r#"1,"b":"#, r"2}"]);
		let cumulative = accumulated(&[r#"{"a":"#, r#"{"a":1,"b":"#, r#"{"a":1,"b":2}"#]);
		assert_eq!(incremental, cumulative);
		assert_eq!(incremental, Bytes::from_static(br#"{"a":1,"b":2}"#));
	}

	#[test]
	fn tool_calls_take_precedence_over_max_tokens() {
		let mut state = State::default();
		decode_response(
			GetChatMessageResponse {
				delta_tool_calls: vec![ChatToolCall {
					id:             "wire-call".to_owned(),
					name:           "task".to_owned(),
					arguments_json: "{}".to_owned(),
				}],
				stop_reason: WireStopReason::MaxTokens as i32,
				..Default::default()
			},
			&mut state,
		);
		let events = finish(&mut state);
		assert!(events.iter().any(
			|event| matches!(event, TurnEvent::Outcome(outcome) if outcome.stop == StopReason::ToolUse)
		));
	}

	#[test]
	fn usage_maps_all_cache_fields() {
		let mut state = State::default();
		decode_response(
			GetChatMessageResponse {
				usage: Some(ModelUsageStats {
					input_tokens:       11,
					output_tokens:      7,
					cache_read_tokens:  5,
					cache_write_tokens: 3,
				}),
				..Default::default()
			},
			&mut state,
		);
		let events = finish(&mut state);
		assert!(events.iter().any(|event| matches!(event, TurnEvent::Outcome(outcome) if outcome.usage.as_ref().is_some_and(|usage| usage.input_tokens == 11 && usage.output_tokens == 7 && usage.cache_read_tokens == 5 && usage.cache_write_tokens == 3))));
	}
	#[test]
	fn model_discovery_decodes_pinned_proto_fields() {
		let payload = GetCliModelConfigsResponse {
			client_model_configs: vec![
				ClientModelConfig {
					label: "Claude Thinking".into(),
					model_uid: "claude-sonnet".into(),
					supports_images: true,
					max_tokens: 200_000,
					model_info: Some(ModelInfo {
						model_features:    Some(ModelFeatures {
							supports_thinking: true,
							..Default::default()
						}),
						max_output_tokens: 64_000,
					}),
					..Default::default()
				},
				ClientModelConfig {
					model_uid: "disabled".into(),
					disabled: true,
					..Default::default()
				},
			],
		}
		.encode_to_vec();
		let models = decode_model_discovery(&payload).expect("protocol fixture");
		assert_eq!(models.len(), 1);
		assert_eq!(models[0].id, "claude-sonnet");
		assert!(models[0].reasoning);
		assert!(models[0].supports_images);
	}
}
