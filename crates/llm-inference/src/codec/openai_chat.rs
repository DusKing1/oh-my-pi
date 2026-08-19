//! Typed `OpenAI` Chat Completions request and incremental response codec.

use std::{collections::BTreeMap, time::Duration};

use bytes::{Bytes, BytesMut};
use omp_core::{IntoStr, Str, encoding::base64};
use omp_llm_catalog::{
	OperationKind, ReasoningEffort, ServiceTier, ThinkingEffort,
	policy::{
		MaxTokensField as CatalogMaxTokensField, ReasoningWireFormat as CatalogReasoningWireFormat,
		ToolCallIdProfile as CatalogToolCallIdProfile, ToolStrictMode,
	},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
	call::{
		ChatRequest, ContentPart, HostedTool, MediaInput, Message, OperationCall, Role, Setting,
		StructuredOutput, ToolChoice, ToolDefinition, ToolResultContent,
	},
	codec::{
		Codec, DecodeContext, Decoder, DecoderState, EncodeContext, EncodedRequest, RawCompletion,
		RawEvent, RequestHeader, RequestMethod, SizeBounds, UnvalidatedToolCall,
	},
	error::{Error, ErrorKind, ErrorPhase, RetryAction},
	event::{BlockKind, ChatEvent, FinishReason, UsageUpdate},
	id::ToolCallId,
	receipt::{ExecutionReceipt, Usage, UsageSource},
	transport::{Frame, FramingProtocol},
};

/// Name of the output-token field accepted by a Chat Completions endpoint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MaxTokensField {
	/// Legacy `max_tokens`.
	MaxTokens,
	/// Current `OpenAI` `max_completion_tokens`.
	#[default]
	MaxCompletionTokens,
	/// Compatibility endpoint `max_output_tokens`.
	MaxOutputTokens,
	/// Do not send an output-token field.
	Omit,
}

/// Reasoning request shape accepted by an OpenAI-compatible endpoint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReasoningWireFormat {
	/// `OpenAI` `reasoning_effort` string.
	#[default]
	OpenAiEffort,
	/// `OpenRouter` `reasoning` object.
	OpenRouter,
	/// Z.ai `thinking` object.
	Zai,
	/// Qwen `enable_thinking` boolean.
	Qwen,
	/// NVIDIA `chat_template_kwargs.enable_thinking` boolean.
	Nvidia,
	/// Endpoint has no reasoning request shape.
	Unsupported,
}
/// Historical reasoning text field accepted by a Chat Completions endpoint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReasoningHistoryField {
	/// `reasoning_content`.
	#[default]
	ReasoningContent,
	/// `reasoning_text`.
	ReasoningText,
	/// Historical reasoning cannot be replayed.
	Unsupported,
}
/// Tool strictness emitted by this route.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ToolStrictWire {
	/// Honor each canonical tool declaration.
	#[default]
	Mixed,
	/// Force strict mode and strict-schema normalization.
	All,
	/// Endpoint rejects the strict field.
	Unsupported,
}

/// Tool-call identifier constraint imposed by the route.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ToolIdWireProfile {
	/// Preserve canonical identifiers.
	#[default]
	Preserve,
	/// At most forty OpenAI-compatible characters.
	OpenAi40,
	/// Exactly nine ASCII alphanumeric characters.
	Mistral9,
}

/// Hosted-tool vocabulary accepted by this concrete endpoint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum HostedToolWireFormat {
	/// Hosted tools are unavailable.
	#[default]
	Unsupported,
	/// Current OpenAI-compatible hosted-tool type tags.
	OpenAi,
}

/// Data-driven wire axes for one Chat Completions route.
#[derive(Clone, Debug)]
pub struct OpenAiChatProfile {
	/// Relative request path.
	pub path: Str,
	/// Role used for canonical system instructions.
	pub system_role: WireRole,
	/// Whether multiple system/developer messages are accepted.
	pub multiple_system_messages: bool,
	/// Whether sampling controls are accepted.
	pub sampling: bool,
	/// Whether presence and frequency penalties are accepted.
	pub penalties: bool,
	/// Whether stop sequences are accepted.
	pub stop_sequences: bool,
	/// Output-token field selection.
	pub max_tokens_field: MaxTokensField,
	/// Whether streaming usage is requested.
	pub streaming_usage: bool,
	/// Whether `store:false` is required.
	pub disable_store: bool,
	/// Whether tool choice is accepted.
	pub tool_choice: bool,
	/// Whether named tool choice is accepted.
	pub named_tool_choice: bool,
	/// Whether required/named forcing is accepted.
	pub forced_tool_choice: bool,
	/// Tool strictness projection.
	pub tool_strict: ToolStrictWire,
	/// Whether object-root tool-parameter unions are flattened or withheld.
	pub flatten_root_unions: bool,
	/// Tool-call identifier projection.
	pub tool_id: ToolIdWireProfile,
	/// Reasoning request shape.
	pub reasoning: ReasoningWireFormat,
	/// Whether Qwen-style template dialects route the selected effort onto the
	/// chat template's `reasoning_effort` kwarg.
	pub template_reasoning_effort: bool,
	/// Historical reasoning text field.
	pub reasoning_history: ReasoningHistoryField,
	/// Whether provider-scoped continuation proofs may be replayed.
	pub reasoning_proofs: bool,
	/// Hosted-tool shape.
	pub hosted_tools: HostedToolWireFormat,
	/// Request body size bound.
	pub max_request_bytes: u64,
	/// Individual response-frame bound.
	pub max_frame_bytes: u64,
	/// Aggregate response bound.
	pub max_response_bytes: u64,
}

impl Default for OpenAiChatProfile {
	fn default() -> Self {
		Self {
			path: Str::from("/v1/chat/completions"),
			system_role: WireRole::System,
			multiple_system_messages: true,
			sampling: true,
			penalties: true,
			stop_sequences: true,
			max_tokens_field: MaxTokensField::MaxCompletionTokens,
			streaming_usage: true,
			disable_store: false,
			tool_choice: true,
			named_tool_choice: true,
			forced_tool_choice: true,
			tool_strict: ToolStrictWire::Mixed,
			flatten_root_unions: false,
			tool_id: ToolIdWireProfile::Preserve,
			reasoning: ReasoningWireFormat::OpenAiEffort,
			template_reasoning_effort: false,
			reasoning_history: ReasoningHistoryField::ReasoningContent,
			reasoning_proofs: false,
			hosted_tools: HostedToolWireFormat::Unsupported,
			max_request_bytes: 16 * 1024 * 1024,
			max_frame_bytes: 16 * 1024 * 1024,
			max_response_bytes: 256 * 1024 * 1024,
		}
	}
}
impl OpenAiChatProfile {
	const fn apply_policy(&mut self, policy: &omp_llm_catalog::policy::WirePolicy) {
		if let Some(value) = policy.role.supports_developer_role {
			self.system_role = if value {
				WireRole::Developer
			} else {
				WireRole::System
			};
		}
		if let Some(value) = policy.role.multiple_system_messages {
			self.multiple_system_messages = value;
		}
		if let Some(value) = policy.structured.sampling_params {
			self.sampling = value;
		}
		if let Some(value) = policy.structured.penalties {
			self.penalties = value;
		}
		if let Some(value) = policy.structured.stop_sequences {
			self.stop_sequences = value;
		}
		if let Some(value) = policy.usage.in_streaming {
			self.streaming_usage = value;
		}
		if let Some(value) = policy.context.supports_store {
			self.disable_store = value;
		}
		if let Some(value) = policy.tool.supports_tool_choice {
			self.tool_choice = value;
		}
		if let Some(value) = policy.tool.named_choice {
			self.named_tool_choice = value;
		}
		if let Some(value) = policy.tool.forced_choice {
			self.forced_tool_choice = value;
		}
		if let Some(value) = policy.context.max_tokens_field {
			self.max_tokens_field = match value {
				CatalogMaxTokensField::MaxTokens => MaxTokensField::MaxTokens,
				CatalogMaxTokensField::MaxCompletionTokens => MaxTokensField::MaxCompletionTokens,
				CatalogMaxTokensField::MaxOutputTokens => MaxTokensField::MaxOutputTokens,
			};
		}
		if let Some(value) = policy.tool.strict_mode {
			self.tool_strict = match value {
				ToolStrictMode::AllStrict => ToolStrictWire::All,
				ToolStrictMode::Mixed => ToolStrictWire::Mixed,
				ToolStrictMode::None => ToolStrictWire::Unsupported,
			};
		}
		if let Some(value) = policy.tool.flatten_root_unions {
			self.flatten_root_unions = value;
		}
		if let Some(value) = policy.tool.id_profile {
			self.tool_id = match value {
				CatalogToolCallIdProfile::Unconstrained => ToolIdWireProfile::Preserve,
				CatalogToolCallIdProfile::OpenAi40 => ToolIdWireProfile::OpenAi40,
				CatalogToolCallIdProfile::Mistral9Alnum => ToolIdWireProfile::Mistral9,
			};
		}
		if let Some(value) = policy.reasoning.wire_format {
			self.reasoning = match value {
				CatalogReasoningWireFormat::OpenAi => ReasoningWireFormat::OpenAiEffort,
				CatalogReasoningWireFormat::OpenRouter => ReasoningWireFormat::OpenRouter,
				CatalogReasoningWireFormat::Zai => ReasoningWireFormat::Zai,
				CatalogReasoningWireFormat::QwenEnableThinking => ReasoningWireFormat::Qwen,
				CatalogReasoningWireFormat::NvidiaChatTemplateKwargs => ReasoningWireFormat::Nvidia,
				_ => ReasoningWireFormat::Unsupported,
			};
		}
		if let Some(value) = policy.reasoning.supports_effort {
			// Qwen 3.8+ chat templates expose a `reasoning_effort` kwarg; the
			// dialect arms consult this flag so older templates never receive
			// an unknown template variable (pi bf490ae024).
			self.template_reasoning_effort = value;
		}
		if let Some(value) = policy.reasoning.include_encrypted {
			self.reasoning_proofs = value;
		}
	}
}

/// Explicit `OpenAI` request extensions.
#[derive(Clone, Debug, Default)]
pub struct OpenAiOptions {
	/// Stable prompt-cache identity.
	pub prompt_cache_key:       Option<Str>,
	/// Prompt-cache retention sent by compatible endpoints.
	pub prompt_cache_retention: Option<Str>,
	/// Request streaming tool-call fragments from compatible endpoints.
	pub tool_stream:            Option<bool>,
}

/// Explicit `OpenRouter` routing extensions.
#[derive(Clone, Debug, Default)]
pub struct OpenRouterOptions {
	/// Ordered upstream provider slugs.
	pub provider_order:  Box<[Str]>,
	/// Permit fallbacks between the explicitly ordered upstreams.
	pub allow_fallbacks: Option<bool>,
}

/// Explicit Vercel AI Gateway extensions.
#[derive(Clone, Debug, Default)]
pub struct VercelGatewayOptions {
	/// Enable gateway caching.
	pub cache: Option<bool>,
}

/// Typed adapter extension selected at registry construction.
#[derive(Clone, Debug)]
pub enum OpenAiChatAdapterOptions {
	/// Direct OpenAI-compatible request extensions.
	OpenAi(OpenAiOptions),
	/// `OpenRouter` routing extensions.
	OpenRouter(OpenRouterOptions),
	/// Vercel AI Gateway extensions.
	Vercel(VercelGatewayOptions),
}

/// Typed codec for `/v1/chat/completions` and structurally compatible routes.
#[derive(Clone, Debug, Default)]
pub struct OpenAiChatCodec {
	profile: OpenAiChatProfile,
	adapter: Option<OpenAiChatAdapterOptions>,
}

impl OpenAiChatCodec {
	/// Constructs a route-specific codec without provider-name or model-name
	/// inspection.
	pub const fn new(profile: OpenAiChatProfile, adapter: Option<OpenAiChatAdapterOptions>) -> Self {
		Self { profile, adapter }
	}

	/// Encodes a chat request to exact JSON bytes for fixture and cassette
	/// assertions.
	pub fn encode_chat(&self, model: &str, request: &ChatRequest) -> Result<Bytes, Error> {
		let wire = self.lower_request(model, request)?;
		serde_json::to_vec(&wire)
			.map(Bytes::from)
			.map_err(|_| encoding_error(ErrorKind::InternalInvariant))
	}

	/// Creates a fresh sans-I/O decoder for one Chat Completions response.
	pub fn chat_decoder(&self) -> OpenAiChatDecoder {
		OpenAiChatDecoder::default()
	}

	/// Returns the exact route-policy-adjusted response frame bound.
	pub(crate) fn maximum_frame_bytes(&self, policy: &omp_llm_catalog::policy::WirePolicy) -> u64 {
		let mut profile = self.profile.clone();
		profile.apply_policy(policy);
		profile.max_frame_bytes
	}

	fn lower_request(&self, model: &str, request: &ChatRequest) -> Result<WireRequest, Error> {
		let messages = lower_messages(&self.profile, &request.messages)?;
		let (mut tools, withheld) = lower_tools(&self.profile, &request.tools)?;
		tools.extend(lower_hosted_tools(&self.profile, &request.hosted_tools)?);
		let tool_choice =
			lower_tool_choice(&self.profile, &mut tools, &request.tool_choice, &withheld)?;
		let response_format = lower_output(&request.output)?;
		let reasoning = lower_reasoning(&self.profile, &request.reasoning)?;
		let sampling = &request.sampling;
		if !request.safety.is_empty() || !matches!(&request.verbosity, Setting::Unset) {
			return Err(capability_error());
		}
		if !self.profile.sampling
			&& (sampling.temperature.is_some() || sampling.top_p.is_some() || sampling.top_k.is_some())
		{
			return Err(capability_error());
		}
		if !self.profile.penalties
			&& (sampling.presence_penalty.is_some() || sampling.frequency_penalty.is_some())
		{
			return Err(capability_error());
		}
		if !self.profile.stop_sequences && !sampling.stop.is_empty() {
			return Err(capability_error());
		}
		let (max_tokens, max_completion_tokens, max_output_tokens) =
			match self.profile.max_tokens_field {
				MaxTokensField::MaxTokens => (request.max_output_tokens, None, None),
				MaxTokensField::MaxCompletionTokens => (None, request.max_output_tokens, None),
				MaxTokensField::MaxOutputTokens => (None, None, request.max_output_tokens),
				MaxTokensField::Omit if request.max_output_tokens.is_some() => {
					return Err(capability_error());
				},
				MaxTokensField::Omit => (None, None, None),
			};
		let (prompt_cache_key, prompt_cache_options, provider, provider_options, tool_stream) =
			lower_adapter(self.adapter.as_ref());
		let service_tier = lower_service_tier(&request.service_tier);
		let cache_requested = !matches!(&request.cache_retention, Setting::Unset);
		if cache_requested && prompt_cache_key.is_none() && prompt_cache_options.is_none() {
			return Err(capability_error());
		}
		Ok(WireRequest {
			model: Str::from(model),
			messages,
			stream: true,
			stream_options: self
				.profile
				.streaming_usage
				.then_some(StreamOptions { include_usage: true }),
			store: self.profile.disable_store.then_some(false),
			temperature: self
				.profile
				.sampling
				.then_some(sampling.temperature)
				.flatten(),
			top_p: self.profile.sampling.then_some(sampling.top_p).flatten(),
			top_k: self.profile.sampling.then_some(sampling.top_k).flatten(),
			presence_penalty: self
				.profile
				.penalties
				.then_some(sampling.presence_penalty)
				.flatten(),
			frequency_penalty: self
				.profile
				.penalties
				.then_some(sampling.frequency_penalty)
				.flatten(),
			stop: (!sampling.stop.is_empty()).then(|| sampling.stop.to_vec()),
			seed: sampling.seed,
			max_tokens,
			max_completion_tokens,
			logprobs: request.top_logprobs.map(|_| true),
			top_logprobs: request.top_logprobs,
			max_output_tokens,
			tools: (!tools.is_empty()).then_some(tools),
			tool_choice,
			response_format,
			reasoning_effort: reasoning.effort,
			reasoning: reasoning.openrouter,
			thinking: reasoning.zai,
			enable_thinking: reasoning.qwen,
			chat_template_kwargs: reasoning.nvidia,
			service_tier,
			prompt_cache_key,
			prompt_cache_options,
			provider,
			provider_options,
			tool_stream,
		})
	}
}

impl Codec for OpenAiChatCodec {
	fn encode(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<EncodedRequest, Error> {
		let OperationCall::Chat(request) = operation else {
			return Err(capability_error());
		};
		let target = context
			.target
			.filter(|_| context.policy_model.is_some())
			.ok_or_else(|| encoding_error(ErrorKind::InvalidRequest))?;
		validate_thinking_selection(request, context.thinking_selection)?;
		let mut selected = self.clone();
		selected.profile.apply_policy(context.policy);
		let wire_model = context
			.thinking_selection
			.map_or(&target.wire_model, |selection| &selection.wire_model);
		let body = selected.encode_chat(wire_model.as_str(), request)?;
		if body.len() as u64 > selected.profile.max_request_bytes {
			return Err(encoding_error(ErrorKind::InvalidRequest));
		}
		let uri = join_uri(target.endpoint.base_url.as_str(), selected.profile.path.as_str());
		Ok(EncodedRequest {
			operation:   OperationKind::Chat,
			method:      RequestMethod::Post,
			uri:         Str::from(uri),
			headers:     vec![RequestHeader {
				name:  Str::from("content-type"),
				value: Str::from("application/json"),
			}]
			.into_boxed_slice(),
			body:        crate::body::BodySource::Bytes(body),
			framing:     FramingProtocol::Sse,
			bounds:      SizeBounds {
				request_body: selected.profile.max_request_bytes,
				frame:        selected.profile.max_frame_bytes,
				response:     selected.profile.max_response_bytes,
			},
			sealed_body: None,
		})
	}

	fn decoder(&self, context: &DecodeContext<'_>) -> Result<DecoderState, Error> {
		if context.operation != OperationKind::Chat
			|| context.operation_call.kind() != OperationKind::Chat
			|| context.target.is_none()
			|| context.policy_model.is_none()
		{
			return Err(encoding_error(ErrorKind::InvalidRequest));
		}
		Ok(Box::new(self.chat_decoder()))
	}
}

fn join_uri(base: &str, path: &str) -> String {
	let mut uri = String::with_capacity(base.len() + path.len() + 1);
	uri.push_str(base.trim_end_matches('/'));
	if !path.starts_with('/') {
		uri.push('/');
	}
	uri.push_str(path);
	uri
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
/// Role vocabulary serialized in Chat Completions messages.
pub enum WireRole {
	/// System instruction message.
	#[default]
	System,
	/// Developer instruction message.
	Developer,
	/// End-user input message.
	User,
	/// Assistant response message.
	Assistant,
	/// Tool result message.
	Tool,
}

#[derive(Serialize)]
struct WireRequest {
	model:                 Str,
	messages:              Vec<WireMessage>,
	stream:                bool,
	#[serde(skip_serializing_if = "Option::is_none")]
	stream_options:        Option<StreamOptions>,
	#[serde(skip_serializing_if = "Option::is_none")]
	store:                 Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	temperature:           Option<f32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	top_p:                 Option<f32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	top_k:                 Option<u32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	presence_penalty:      Option<f32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	frequency_penalty:     Option<f32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	stop:                  Option<Vec<Str>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	seed:                  Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	max_tokens:            Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	max_completion_tokens: Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	max_output_tokens:     Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tools:                 Option<Vec<WireTool>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_choice:           Option<WireToolChoice>,
	#[serde(skip_serializing_if = "Option::is_none")]
	response_format:       Option<ResponseFormat>,
	#[serde(skip_serializing_if = "Option::is_none")]
	reasoning_effort:      Option<WireEffort>,
	#[serde(skip_serializing_if = "Option::is_none")]
	reasoning:             Option<OpenRouterReasoning>,
	#[serde(skip_serializing_if = "Option::is_none")]
	logprobs:              Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	top_logprobs:          Option<u8>,
	#[serde(skip_serializing_if = "Option::is_none")]
	thinking:              Option<ZaiThinking>,
	#[serde(skip_serializing_if = "Option::is_none")]
	enable_thinking:       Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	chat_template_kwargs:  Option<ChatTemplateKwargs>,
	#[serde(skip_serializing_if = "Option::is_none")]
	service_tier:          Option<Str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	prompt_cache_key:      Option<Str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	prompt_cache_options:  Option<PromptCacheOptions>,
	#[serde(skip_serializing_if = "Option::is_none")]
	provider:              Option<ProviderRouting>,
	#[serde(rename = "providerOptions", skip_serializing_if = "Option::is_none")]
	provider_options:      Option<GatewayProviderOptions>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_stream:           Option<bool>,
}

#[derive(Serialize)]
struct StreamOptions {
	include_usage: bool,
}

#[derive(Serialize)]
struct WireMessage {
	role:              WireRole,
	#[serde(skip_serializing_if = "Option::is_none")]
	content:           Option<NullableContent>,
	#[serde(skip_serializing_if = "Option::is_none")]
	name:              Option<Str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_call_id:      Option<Str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_calls:        Option<Vec<WireAssistantToolCall>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	reasoning_content: Option<Str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	reasoning_text:    Option<Str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	reasoning_details: Option<Vec<WireReasoningReplay>>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum NullableContent {
	Text(Str),
	Parts(Vec<WireContentPart>),
	Null(()),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireContentPart {
	Text { text: Str },
	ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize)]
struct ImageUrl {
	url: String,
}
#[derive(Serialize)]
struct WireAssistantToolCall {
	id:       Str,
	#[serde(rename = "type")]
	kind:     FunctionTag,
	function: WireAssistantFunction,
}

#[derive(Serialize)]
struct WireAssistantFunction {
	name:      Str,
	arguments: String,
}
#[derive(Serialize)]
#[serde(untagged)]
enum WireReasoningReplay {
	Opaque(Box<serde_json::value::RawValue>),
	Encrypted {
		#[serde(rename = "type")]
		kind: ReasoningEncryptedTag,
		#[serde(skip_serializing_if = "Option::is_none")]
		id:   Option<Str>,
		data: String,
	},
}

#[derive(Serialize)]
enum ReasoningEncryptedTag {
	#[serde(rename = "reasoning.encrypted")]
	Encrypted,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireTool {
	Function {
		function: WireFunction,
	},
	WebSearch {
		#[serde(skip_serializing_if = "Option::is_none")]
		web_search: Option<WebSearchOptions>,
	},
	CodeInterpreter,
	FileSearch {
		file_search: FileSearchOptions,
	},
}

#[derive(Serialize)]
struct WireFunction {
	name:        Str,
	#[serde(skip_serializing_if = "Option::is_none")]
	description: Option<Str>,
	parameters:  Value,
	#[serde(skip_serializing_if = "Option::is_none")]
	strict:      Option<bool>,
}

#[derive(Serialize)]
struct WebSearchOptions {
	#[serde(skip_serializing_if = "Vec::is_empty")]
	allowed_domains: Vec<Str>,
	#[serde(skip_serializing_if = "Vec::is_empty")]
	blocked_domains: Vec<Str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	recency_days:    Option<u32>,
}

#[derive(Serialize)]
struct FileSearchOptions {
	vector_store_ids: Vec<Str>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum WireToolChoice {
	Mode(ToolChoiceMode),
	Named { r#type: FunctionTag, function: NamedFunction },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum ToolChoiceMode {
	Auto,
	None,
	Required,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum FunctionTag {
	Function,
}

#[derive(Serialize)]
struct NamedFunction {
	name: Str,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseFormat {
	JsonObject,
	JsonSchema { json_schema: JsonSchemaFormat },
}

#[derive(Serialize)]
struct JsonSchemaFormat {
	name:   Str,
	schema: Value,
	strict: bool,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
enum WireEffort {
	None,
	Minimal,
	Low,
	Medium,
	High,
	Xhigh,
	Max,
}

#[derive(Serialize)]
struct OpenRouterReasoning {
	#[serde(skip_serializing_if = "Option::is_none")]
	effort:     Option<WireEffort>,
	exclude:    bool,
	#[serde(skip_serializing_if = "Option::is_none")]
	max_tokens: Option<u64>,
}

#[derive(Serialize)]
struct ZaiThinking {
	r#type: ThinkingType,
	#[serde(skip_serializing_if = "Option::is_none")]
	effort: Option<WireEffort>,
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum ThinkingType {
	Enabled,
}

#[derive(Serialize)]
struct ChatTemplateKwargs {
	#[serde(skip_serializing_if = "Option::is_none")]
	enable_thinking:  Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	reasoning_effort: Option<WireEffort>,
}

#[derive(Serialize)]
struct PromptCacheOptions {
	retention: Str,
}

#[derive(Serialize)]
struct ProviderRouting {
	order:           Vec<Str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	allow_fallbacks: Option<bool>,
}

#[derive(Serialize)]
struct GatewayProviderOptions {
	gateway: GatewayOptions,
}

#[derive(Serialize)]
struct GatewayOptions {
	cache: bool,
}

struct ReasoningFields {
	effort:     Option<WireEffort>,
	openrouter: Option<OpenRouterReasoning>,
	zai:        Option<ZaiThinking>,
	qwen:       Option<bool>,
	nvidia:     Option<ChatTemplateKwargs>,
}

fn lower_messages(
	profile: &OpenAiChatProfile,
	messages: &[Message],
) -> Result<Vec<WireMessage>, Error> {
	let mut lowered = Vec::new();
	for message in messages {
		let role = match message.role {
			Role::System => profile.system_role,
			Role::Developer if profile.system_role == WireRole::Developer => WireRole::Developer,
			Role::Developer => WireRole::System,
			Role::User => WireRole::User,
			Role::Assistant => WireRole::Assistant,
			Role::Tool => WireRole::Tool,
		};
		if message.role == Role::Tool {
			for part in message.content.iter() {
				let ContentPart::ToolResult { call, name, content, .. } = part else {
					return Err(encoding_error(ErrorKind::InvalidRequest));
				};
				lowered.push(WireMessage {
					role,
					content: Some(NullableContent::Text(lower_tool_result_content(content)?)),
					name: name.clone().or_else(|| message.name.clone()),
					tool_call_id: Some(project_call_id(profile.tool_id, call.as_str())),
					tool_calls: None,
					reasoning_content: None,
					reasoning_text: None,
					reasoning_details: None,
				});
			}
			continue;
		}
		let mut ordinary = Vec::new();
		let mut reasoning = String::new();
		let mut calls = Vec::new();
		let mut details = Vec::new();
		for part in message.content.iter() {
			match part {
				ContentPart::Text { text, proof } => {
					if let Some(proof) = proof {
						if !profile.reasoning_proofs {
							return Err(capability_error());
						}
						details.push(proof_detail(&proof.value, None));
					}
					ordinary.push(ContentPart::Text { text: text.clone(), proof: None });
				},
				ContentPart::Reasoning { text, proof } if message.role == Role::Assistant => {
					if let Some(proof) = proof {
						if !profile.reasoning_proofs {
							return Err(capability_error());
						}
						details.push(proof_detail(&proof.value, None));
					}
					reasoning.push_str(text.as_str());
				},
				ContentPart::ToolCall { call, name, arguments, proof }
					if message.role == Role::Assistant =>
				{
					let wire_id = project_call_id(profile.tool_id, call.as_str());
					if let Some(proof) = proof {
						if !profile.reasoning_proofs {
							return Err(capability_error());
						}
						details.push(proof_detail(&proof.value, Some(wire_id.clone())));
					}
					calls.push(WireAssistantToolCall {
						id:       wire_id,
						kind:     FunctionTag::Function,
						function: WireAssistantFunction {
							name:      name.clone(),
							arguments: serde_json::to_string(arguments.as_value())
								.map_err(|_| encoding_error(ErrorKind::InvalidRequest))?,
						},
					});
				},
				ContentPart::Reasoning { .. } | ContentPart::ToolCall { .. } => {
					return Err(encoding_error(ErrorKind::InvalidRequest));
				},
				other => ordinary.push(other.clone()),
			}
		}
		let content = lower_content(&ordinary)?;
		let (reasoning_content, reasoning_text) = if reasoning.is_empty() {
			(None, None)
		} else {
			match profile.reasoning_history {
				ReasoningHistoryField::ReasoningContent => (Some(Str::from(reasoning)), None),
				ReasoningHistoryField::ReasoningText => (None, Some(Str::from(reasoning))),
				ReasoningHistoryField::Unsupported => return Err(capability_error()),
			}
		};
		lowered.push(WireMessage {
			role,
			content: Some(content),
			name: message.name.clone(),
			tool_call_id: None,
			tool_calls: (!calls.is_empty()).then_some(calls),
			reasoning_content,
			reasoning_text,
			reasoning_details: (!details.is_empty()).then_some(details),
		});
	}
	if !profile.multiple_system_messages {
		coalesce_system_messages(&mut lowered, profile.system_role)?;
	}
	Ok(lowered)
}

fn lower_content(parts: &[ContentPart]) -> Result<NullableContent, Error> {
	if let [ContentPart::Text { text, .. }] = parts {
		return Ok(NullableContent::Text(text.clone()));
	}
	if parts.is_empty() {
		return Ok(NullableContent::Null(()));
	}
	let mut wire = Vec::with_capacity(parts.len());
	for part in parts {
		match part {
			ContentPart::Text { text, .. } => wire.push(WireContentPart::Text { text: text.clone() }),
			ContentPart::Image(MediaInput::Bytes { media_type, data }) => {
				let encoded = base64::encode(data).into_string();
				let mut url = String::with_capacity(media_type.len() + encoded.len() + 13);
				url.push_str("data:");
				url.push_str(media_type.as_str());
				url.push_str(";base64,");
				url.push_str(&encoded);
				wire.push(WireContentPart::ImageUrl { image_url: ImageUrl { url } });
			},
			ContentPart::Image(MediaInput::Remote { uri, .. }) => {
				wire.push(WireContentPart::ImageUrl { image_url: ImageUrl { url: uri.to_string() } });
			},
			ContentPart::Image(MediaInput::Stored(_) | MediaInput::Body { .. })
			| ContentPart::Reasoning { .. }
			| ContentPart::Audio(_)
			| ContentPart::Document(_)
			| ContentPart::ToolCall { .. }
			| ContentPart::ToolResult { .. }
			| ContentPart::CachePoint(_) => return Err(capability_error()),
		}
	}
	Ok(NullableContent::Parts(wire))
}

fn lower_tool_result_content(content: &[ToolResultContent]) -> Result<Str, Error> {
	let mut output = String::new();
	for (index, part) in content.iter().enumerate() {
		if index != 0 {
			output.push('\n');
		}
		match part {
			ToolResultContent::Text(text) => output.push_str(text.as_str()),
			ToolResultContent::Json(value) => output.push_str(
				&serde_json::to_string(value.as_value())
					.map_err(|_| encoding_error(ErrorKind::InvalidRequest))?,
			),
			ToolResultContent::Image(_) | ToolResultContent::Document(_) => {
				return Err(capability_error());
			},
		}
	}
	Ok(Str::from(output))
}

fn validate_thinking_selection(
	request: &ChatRequest,
	selection: Option<&omp_llm_catalog::ThinkingSelection>,
) -> Result<(), Error> {
	let reasoning = match &request.reasoning {
		Setting::Unset => return Ok(()),
		Setting::Require(reasoning) | Setting::Prefer(reasoning) => reasoning,
	};
	let selection = selection.ok_or_else(capability_error)?;
	if reasoning.max_tokens != selection.budget {
		return Err(capability_error());
	}
	if let Some(effort) = reasoning.effort
		&& canonical_thinking_effort(effort) != selection.effort
	{
		return Err(capability_error());
	}
	Ok(())
}

const fn canonical_thinking_effort(effort: ReasoningEffort) -> ThinkingEffort {
	match effort {
		ReasoningEffort::Off => ThinkingEffort::Off,
		ReasoningEffort::Minimal => ThinkingEffort::Minimal,
		ReasoningEffort::Low => ThinkingEffort::Low,
		ReasoningEffort::Medium => ThinkingEffort::Medium,
		ReasoningEffort::High => ThinkingEffort::High,
		ReasoningEffort::Xhigh => ThinkingEffort::XHigh,
		ReasoningEffort::Max => ThinkingEffort::Max,
	}
}

fn proof_detail(value: &[u8], id: Option<Str>) -> WireReasoningReplay {
	if let Ok(raw) = serde_json::from_slice::<Box<serde_json::value::RawValue>>(value) {
		return WireReasoningReplay::Opaque(raw);
	}
	let data = std::str::from_utf8(value)
		.map_or_else(|_| base64::encode(value).into_string(), str::to_owned);
	WireReasoningReplay::Encrypted { kind: ReasoningEncryptedTag::Encrypted, id, data }
}

fn project_call_id(profile: ToolIdWireProfile, value: &str) -> Str {
	match profile {
		ToolIdWireProfile::Preserve => Str::from(value),
		ToolIdWireProfile::OpenAi40 => {
			let projected: String = value
				.chars()
				.filter(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
				.take(40)
				.collect();
			Str::from(projected)
		},
		ToolIdWireProfile::Mistral9 => {
			let mut hash = 0xcbf2_9ce4_8422_2325_u64;
			for byte in value.bytes() {
				hash ^= u64::from(byte);
				hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
			}
			let mut output = [b'0'; 9];
			for slot in output.iter_mut().rev() {
				let digit = (hash % 36) as u8;
				*slot = if digit < 10 {
					b'0' + digit
				} else {
					b'a' + digit - 10
				};
				hash /= 36;
			}
			Str::from(std::str::from_utf8(&output).expect("ASCII identifier"))
		},
	}
}

fn coalesce_system_messages(messages: &mut Vec<WireMessage>, role: WireRole) -> Result<(), Error> {
	let mut first = None;
	let mut text = String::new();
	let mut remove = Vec::new();
	for (index, message) in messages.iter().enumerate() {
		if message.role != role {
			continue;
		}
		let Some(NullableContent::Text(content)) = &message.content else {
			return Err(capability_error());
		};
		if first.is_none() {
			first = Some(index);
		} else {
			text.push_str("\n\n");
		}
		text.push_str(content.as_str());
		if first != Some(index) {
			remove.push(index);
		}
	}
	if let Some(index) = first {
		messages[index].content = Some(NullableContent::Text(Str::from(text)));
		for index in remove.into_iter().rev() {
			messages.remove(index);
		}
	}
	Ok(())
}

fn lower_tools(
	profile: &OpenAiChatProfile,
	tools: &[ToolDefinition],
) -> Result<(Vec<WireTool>, Vec<Str>), Error> {
	let mut lowered = Vec::with_capacity(tools.len());
	let mut withheld = Vec::new();
	for tool in tools {
		let Some((parameters, declared_strict)) = tool.input.json_schema() else {
			return Err(tool_capability_error("openai.chat.tools.grammar_unsupported"));
		};
		let (strict, normalize) = match profile.tool_strict {
			ToolStrictWire::Mixed => (Some(declared_strict), declared_strict),
			ToolStrictWire::All => (Some(true), true),
			ToolStrictWire::Unsupported if declared_strict => {
				return Err(tool_capability_error("openai.chat.tools.strict_unsupported"));
			},
			ToolStrictWire::Unsupported => (None, false),
		};
		let flattened = if profile.flatten_root_unions {
			flatten_exclusive_required_root_union(parameters.as_value())
		} else {
			None
		};
		let parameters = match (normalize, flattened) {
			(true, Some(flattened)) => strict_schema(&flattened)?,
			(true, None) => strict_schema(parameters.as_value())?,
			(false, Some(flattened)) => flattened,
			(false, None) => parameters.as_value().clone(),
		};
		if profile.flatten_root_unions && leftover_root_object_union(&parameters) {
			// xAI rejects the entire request over one leftover object-root
			// union ("tool parameter root must be an object type"); withhold
			// just the offending tool so the other tools stay usable.
			withheld.push(tool.name.clone());
			continue;
		}
		lowered.push(WireTool::Function {
			function: WireFunction {
				name: tool.name.clone(),
				description: tool.description.clone(),
				parameters,
				strict,
			},
		});
	}
	Ok((lowered, withheld))
}

/// Returns a copy of an object-root schema with its `anyOf`/`oneOf` union
/// removed when every branch is a typeless exclusive-`required` fragment.
/// Returns `None` — leave the schema untouched — for every other shape: away
/// from providers that reject object-root unions, the union is a real
/// model-facing constraint.
pub(crate) fn flatten_exclusive_required_root_union(schema: &Value) -> Option<Value> {
	let object = schema.as_object()?;
	let union_key = ["anyOf", "oneOf"]
		.into_iter()
		.find(|key| object.get(*key).is_some_and(Value::is_array))?;
	let branches = object.get(union_key).and_then(Value::as_array)?;
	if branches.is_empty()
		|| !declares_object_root(object)
		|| !branches.iter().all(exclusive_required_branch)
	{
		return None;
	}
	let mut flattened = object.clone();
	flattened.remove(union_key);
	Some(Value::Object(flattened))
}

/// True when an object-root schema still carries an `anyOf`/`oneOf` with a
/// typeless or non-object branch; xAI rejects the whole request for one such
/// tool ("tool parameter root must be an object type"). Nested unions and
/// pure (untyped) root unions are not this error.
pub(crate) fn leftover_root_object_union(schema: &Value) -> bool {
	let Some(object) = schema.as_object() else {
		return false;
	};
	if !declares_object_root(object) {
		return false;
	}
	["anyOf", "oneOf"].into_iter().any(|key| {
		object
			.get(key)
			.and_then(Value::as_array)
			.is_some_and(|branches| {
				!branches.is_empty()
					&& branches
						.iter()
						.any(|branch| !branch.as_object().is_some_and(declares_object_type))
			})
	})
}

/// Whether the node's declared `type` names or includes `object`.
fn declares_object_type(object: &serde_json::Map<String, Value>) -> bool {
	match object.get("type") {
		Some(Value::String(kind)) => kind == "object",
		Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some("object")),
		_ => false,
	}
}

/// Whether the schema root is object-shaped: object-typed or property-bearing.
fn declares_object_root(object: &serde_json::Map<String, Value>) -> bool {
	declares_object_type(object) || object.get("properties").is_some_and(Value::is_object)
}

/// One branch of an exclusive-required union: a typeless fragment carrying
/// only a non-empty `required` list plus optional `description`/`title`.
fn exclusive_required_branch(branch: &Value) -> bool {
	let Some(object) = branch.as_object() else {
		return false;
	};
	if object.contains_key("type") {
		return false;
	}
	let Some(required) = object.get("required").and_then(Value::as_array) else {
		return false;
	};
	if required.is_empty()
		|| !required
			.iter()
			.all(|name| name.as_str().is_some_and(|name| !name.is_empty()))
	{
		return false;
	}
	object
		.keys()
		.all(|key| matches!(key.as_str(), "required" | "description" | "title"))
}

fn strict_schema(schema: &Value) -> Result<Value, Error> {
	match schema {
		Value::Object(object) => {
			let mut output = object.clone();
			if let Some(Value::Object(properties)) = object.get("properties") {
				let mut normalized = serde_json::Map::with_capacity(properties.len());
				let mut required = Vec::with_capacity(properties.len());
				for (name, property) in properties {
					normalized.insert(name.clone(), strict_schema(property)?);
					required.push(Value::String(name.clone()));
				}
				output.insert("properties".into(), Value::Object(normalized));
				output.insert("required".into(), Value::Array(required));
				output.insert("additionalProperties".into(), Value::Bool(false));
			}
			for keyword in ["items", "additionalProperties", "not", "if", "then", "else"] {
				if let Some(value) = object.get(keyword).filter(|value| value.is_object()) {
					output.insert(keyword.into(), strict_schema(value)?);
				}
			}
			for keyword in ["allOf", "anyOf", "oneOf", "prefixItems"] {
				if let Some(Value::Array(values)) = object.get(keyword) {
					output.insert(
						keyword.into(),
						Value::Array(values.iter().map(strict_schema).collect::<Result<_, _>>()?),
					);
				}
			}
			Ok(Value::Object(output))
		},
		_ => Ok(schema.clone()),
	}
}

fn lower_hosted_tools(
	profile: &OpenAiChatProfile,
	tools: &[HostedTool],
) -> Result<Vec<WireTool>, Error> {
	if tools.is_empty() {
		return Ok(Vec::new());
	}
	if profile.hosted_tools == HostedToolWireFormat::Unsupported {
		return Err(capability_error());
	}
	Ok(tools
		.iter()
		.map(|tool| match tool {
			HostedTool::WebSearch { allowed_domains, blocked_domains, recency_days } => {
				WireTool::WebSearch {
					web_search: Some(WebSearchOptions {
						allowed_domains: allowed_domains.to_vec(),
						blocked_domains: blocked_domains.to_vec(),
						recency_days:    *recency_days,
					}),
				}
			},
			HostedTool::CodeExecution => WireTool::CodeInterpreter,
			HostedTool::Retrieval { stores } => WireTool::FileSearch {
				file_search: FileSearchOptions { vector_store_ids: stores.to_vec() },
			},
		})
		.collect())
}

fn lower_tool_choice(
	profile: &OpenAiChatProfile,
	tools: &mut Vec<WireTool>,
	choice: &Setting<ToolChoice>,
	withheld: &[Str],
) -> Result<Option<WireToolChoice>, Error> {
	let choice = match choice {
		Setting::Unset => return Ok(None),
		Setting::Require(value) | Setting::Prefer(value) => value,
	};
	if !profile.tool_choice {
		return Err(capability_error());
	}
	// A force naming a withheld tool — or a bare `required` once withholding
	// emptied the list — would 400 exactly like the schema it was withheld
	// for; drop the force and let the model choose.
	match choice {
		ToolChoice::Named(name)
			if withheld
				.iter()
				.any(|withheld| withheld.as_str() == name.as_str()) =>
		{
			return Ok(None);
		},
		ToolChoice::Required if tools.is_empty() && !withheld.is_empty() => {
			return Ok(None);
		},
		_ => {},
	}
	Ok(Some(match choice {
		ToolChoice::Disabled => WireToolChoice::Mode(ToolChoiceMode::None),
		ToolChoice::Auto => WireToolChoice::Mode(ToolChoiceMode::Auto),
		ToolChoice::Required if profile.forced_tool_choice => {
			WireToolChoice::Mode(ToolChoiceMode::Required)
		},
		ToolChoice::Required => return Err(capability_error()),
		ToolChoice::Named(name) if profile.named_tool_choice && profile.forced_tool_choice => {
			WireToolChoice::Named {
				r#type:   FunctionTag::Function,
				function: NamedFunction { name: name.clone() },
			}
		},
		ToolChoice::Named(name) if profile.forced_tool_choice => {
			let before = tools.len();
			tools.retain(|tool| matches!(tool, WireTool::Function { function } if function.name.as_str() == name.as_str()));
			if tools.len() != 1 || before == 0 {
				return Err(encoding_error(ErrorKind::InvalidRequest));
			}
			WireToolChoice::Mode(ToolChoiceMode::Required)
		},
		ToolChoice::Named(_) => return Err(capability_error()),
	}))
}

fn lower_output(output: &Setting<StructuredOutput>) -> Result<Option<ResponseFormat>, Error> {
	let output = match output {
		Setting::Unset => return Ok(None),
		Setting::Require(value) | Setting::Prefer(value) => value,
	};
	Ok(Some(match output {
		StructuredOutput::JsonObject => ResponseFormat::JsonObject,
		StructuredOutput::JsonSchema { name, schema, strict } => ResponseFormat::JsonSchema {
			json_schema: JsonSchemaFormat {
				name:   name.clone(),
				schema: if *strict {
					strict_schema(schema.as_value())?
				} else {
					schema.as_value().clone()
				},
				strict: *strict,
			},
		},
		StructuredOutput::Regex(_) | StructuredOutput::Lark(_) | StructuredOutput::Ebnf(_) => {
			return Err(capability_error());
		},
	}))
}

fn lower_reasoning(
	profile: &OpenAiChatProfile,
	reasoning: &Setting<crate::call::ReasoningRequest>,
) -> Result<ReasoningFields, Error> {
	let reasoning = match reasoning {
		Setting::Unset => {
			return Ok(ReasoningFields {
				effort:     None,
				openrouter: None,
				zai:        None,
				qwen:       None,
				nvidia:     None,
			});
		},
		Setting::Require(value) | Setting::Prefer(value) => value,
	};
	let effort = reasoning.effort.map(lower_effort);
	let fields = match profile.reasoning {
		ReasoningWireFormat::OpenAiEffort if reasoning.max_tokens.is_some() => {
			return Err(capability_error());
		},
		ReasoningWireFormat::OpenAiEffort => {
			ReasoningFields { effort, openrouter: None, zai: None, qwen: None, nvidia: None }
		},
		ReasoningWireFormat::OpenRouter => ReasoningFields {
			effort:     None,
			openrouter: Some(OpenRouterReasoning {
				effort,
				exclude: reasoning.visibility == crate::call::ReasoningVisibility::Hidden,
				max_tokens: reasoning.max_tokens,
			}),
			zai:        None,
			qwen:       None,
			nvidia:     None,
		},
		ReasoningWireFormat::Zai => ReasoningFields {
			effort:     None,
			openrouter: None,
			zai:        Some(ZaiThinking { r#type: ThinkingType::Enabled, effort }),
			qwen:       None,
			nvidia:     None,
		},
		ReasoningWireFormat::Qwen => {
			// Twin emission for Qwen 3.8+ templates (pi bf490ae024): newer
			// llama.cpp builds map the top-level `reasoning_effort` into the
			// template, older builds read only the kwargs copy. Older Qwen
			// templates have no `reasoning_effort` kwarg, so the policy flag
			// gates the emission entirely.
			let template_effort = profile
				.template_reasoning_effort
				.then_some(effort)
				.flatten();
			ReasoningFields {
				effort:     template_effort,
				openrouter: None,
				zai:        None,
				qwen:       Some(true),
				nvidia:     template_effort.map(|effort| ChatTemplateKwargs {
					enable_thinking:  None,
					reasoning_effort: Some(effort),
				}),
			}
		},
		ReasoningWireFormat::Nvidia => ReasoningFields {
			effort:     None,
			openrouter: None,
			zai:        None,
			qwen:       None,
			// NIM-style request schemas reject unknown top-level fields, so
			// the effort rides `chat_template_kwargs` alone on this dialect.
			nvidia:     Some(ChatTemplateKwargs {
				enable_thinking:  Some(true),
				reasoning_effort: profile
					.template_reasoning_effort
					.then_some(effort)
					.flatten(),
			}),
		},
		ReasoningWireFormat::Unsupported => return Err(capability_error()),
	};
	Ok(fields)
}

const fn lower_effort(effort: ReasoningEffort) -> WireEffort {
	match effort {
		ReasoningEffort::Off => WireEffort::None,
		ReasoningEffort::Minimal => WireEffort::Minimal,
		ReasoningEffort::Low => WireEffort::Low,
		ReasoningEffort::Medium => WireEffort::Medium,
		ReasoningEffort::High => WireEffort::High,
		ReasoningEffort::Xhigh => WireEffort::Xhigh,
		ReasoningEffort::Max => WireEffort::Max,
	}
}

fn lower_service_tier(tier: &Setting<ServiceTier>) -> Option<Str> {
	match tier {
		Setting::Unset => None,
		Setting::Require(tier) | Setting::Prefer(tier) => Some(tier.name.clone()),
	}
}

#[allow(
	clippy::type_complexity,
	reason = "typed adapter fields map one-to-one to separate wire objects"
)]
fn lower_adapter(
	adapter: Option<&OpenAiChatAdapterOptions>,
) -> (
	Option<Str>,
	Option<PromptCacheOptions>,
	Option<ProviderRouting>,
	Option<GatewayProviderOptions>,
	Option<bool>,
) {
	match adapter {
		Some(OpenAiChatAdapterOptions::OpenAi(options)) => (
			options.prompt_cache_key.clone(),
			options
				.prompt_cache_retention
				.clone()
				.map(|retention| PromptCacheOptions { retention }),
			None,
			None,
			options.tool_stream,
		),
		Some(OpenAiChatAdapterOptions::OpenRouter(options)) => (
			None,
			None,
			Some(ProviderRouting {
				order:           options.provider_order.to_vec(),
				allow_fallbacks: options.allow_fallbacks,
			}),
			None,
			None,
		),
		Some(OpenAiChatAdapterOptions::Vercel(options)) => (
			None,
			None,
			None,
			options
				.cache
				.map(|cache| GatewayProviderOptions { gateway: GatewayOptions { cache } }),
			None,
		),
		None => (None, None, None, None, None),
	}
}

/// Incremental typed Chat Completions decoder.
#[derive(Default)]
pub struct OpenAiChatDecoder {
	choices:    BTreeMap<u32, ChoiceState>,
	next_block: u32,
	usage:      Usage,
	done:       bool,
	committed:  bool,
}

#[derive(Default)]
struct ChoiceState {
	text_block:     Option<u32>,
	thinking_block: Option<u32>,
	tools:          BTreeMap<u32, PendingTool>,
	finish:         Option<FinishReason>,
}

struct PendingTool {
	block:     u32,
	id:        ToolCallId,
	name:      Str,
	arguments: BytesMut,
	started:   bool,
	completed: bool,
}

impl Decoder for OpenAiChatDecoder {
	fn push(&mut self, frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.done {
			return Ok(());
		}
		let Frame::Sse(event) = frame else {
			return Err(self.decode_error(None));
		};
		if event.data.as_ref() == b"[DONE]" {
			return self.complete(emit);
		}
		let chunk: WireChunk =
			serde_json::from_slice(&event.data).map_err(|_| self.decode_error(None))?;
		if let Some(error) = chunk.error {
			self.done = true;
			emit(RawEvent::Failure(classify_error(error, self.committed)));
			return Ok(());
		}
		let final_usage = chunk.choices.is_empty();
		for choice in chunk.choices {
			self.decode_choice(choice, emit);
		}
		if let Some(usage) = chunk.usage {
			merge_usage(&mut self.usage, usage.canonical());
			emit(RawEvent::Chat(ChatEvent::Usage(UsageUpdate {
				usage:        self.usage,
				final_update: final_usage,
			})));
		}
		Ok(())
	}

	fn finish(&mut self, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.done {
			return Ok(());
		}
		self.complete(emit)
	}
}

impl OpenAiChatDecoder {
	fn decode_choice(&mut self, choice: WireChoice, emit: &mut dyn FnMut(RawEvent)) {
		let index = choice.index;
		let mut state = self.choices.remove(&index).unwrap_or_default();
		if let Some(reason) = choice.finish_reason {
			if matches!(reason, WireFinishReason::InsufficientSystemResource) {
				self.done = true;
				emit(RawEvent::Failure(resource_finish_error(self.committed)));
				return;
			}
			state.finish = Some(reason.normalize());
		}
		let payload = choice.delta.or(choice.message).unwrap_or_default();
		if let Some(error) = payload.error {
			self.done = true;
			emit(RawEvent::Failure(classify_error(error, self.committed)));
			return;
		}
		let reasoning = payload
			.reasoning_content
			.or(payload.reasoning_text)
			.or(payload.reasoning);
		if let Some(reasoning) = reasoning.filter(|text| !text.is_empty()) {
			let block = *state
				.thinking_block
				.get_or_insert_with(|| self.start_block(BlockKind::Thinking, emit));
			emit(RawEvent::Chat(ChatEvent::ThinkingDelta { index: block, text: reasoning }));
			self.committed = true;
		}
		if let Some(content) = payload
			.content
			.map(WireDeltaContent::into_text)
			.or(payload.text)
			&& !content.is_empty()
		{
			let block = *state
				.text_block
				.get_or_insert_with(|| self.start_block(BlockKind::Text, emit));
			emit(RawEvent::Chat(ChatEvent::TextDelta { index: block, text: content }));
			self.committed = true;
		}
		if let Some(refusal) = payload.refusal.filter(|text| !text.is_empty()) {
			let block = *state
				.text_block
				.get_or_insert_with(|| self.start_block(BlockKind::Text, emit));
			emit(RawEvent::Chat(ChatEvent::TextDelta { index: block, text: refusal }));
			self.committed = true;
		}
		for detail in payload.reasoning_details {
			if let Some(signature) = detail.data {
				let block = state.thinking_block.unwrap_or(0);
				emit(RawEvent::ProviderState(crate::codec::ProviderStateEvent::ReasoningSignature {
					index:     block,
					signature: Bytes::from(signature.into_bytes()),
				}));
			}
		}
		for (position, call) in payload.tool_calls.into_iter().enumerate() {
			let wire_index = call.index.unwrap_or(position as u32);
			if let std::collections::btree_map::Entry::Vacant(e) = state.tools.entry(wire_index) {
				let block = self.next_block;
				self.next_block = self.next_block.saturating_add(1);
				let id = call
					.id
					.clone()
					.unwrap_or_else(|| Str::from(format!("tool-{index}-{wire_index}")));
				e.insert(PendingTool {
					block,
					id: ToolCallId::from(id.as_str()),
					name: Str::default(),
					arguments: BytesMut::new(),
					started: false,
					completed: false,
				});
			}
			let tool = state
				.tools
				.get_mut(&wire_index)
				.expect("tool inserted above");
			if let Some(id) = call.id {
				tool.id = ToolCallId::from(id.as_str());
			}
			if let Some(name) = call.function.name {
				tool.name = name;
			}
			if !tool.started && !tool.name.is_empty() {
				tool.started = true;
				emit(RawEvent::Chat(ChatEvent::BlockStarted {
					index: tool.block,
					kind:  BlockKind::ToolCall,
				}));
				emit(RawEvent::Chat(ChatEvent::ToolCallStarted {
					index: tool.block,
					id:    tool.id.clone(),
					name:  tool.name.clone(),
				}));
				if !tool.arguments.is_empty() {
					emit(RawEvent::Chat(ChatEvent::ToolArgumentsDelta {
						index: tool.block,
						bytes: Bytes::copy_from_slice(&tool.arguments),
					}));
				}
				self.committed = true;
			}
			if let Some(arguments) = call.function.arguments.filter(|bytes| !bytes.is_empty()) {
				tool.arguments.extend_from_slice(arguments.as_bytes());
				if tool.started {
					emit(RawEvent::Chat(ChatEvent::ToolArgumentsDelta {
						index: tool.block,
						bytes: Bytes::copy_from_slice(arguments.as_bytes()),
					}));
					self.committed = true;
				}
			}
		}
		self.choices.insert(index, state);
	}

	fn start_block(&mut self, kind: BlockKind, emit: &mut dyn FnMut(RawEvent)) -> u32 {
		let index = self.next_block;
		self.next_block = self.next_block.saturating_add(1);
		emit(RawEvent::Chat(ChatEvent::BlockStarted { index, kind }));
		index
	}

	fn complete(&mut self, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.done {
			return Ok(());
		}
		if self.committed && !self.choices.values().any(|state| state.finish.is_some()) {
			self.done = true;
			return Err(incomplete_stream_error());
		}
		let mut finish = FinishReason::Stop;
		let committed = self.committed;
		for state in self.choices.values_mut() {
			let has_tools = !state.tools.is_empty();
			let choice_finish = if has_tools {
				FinishReason::ToolCalls
			} else {
				state.finish.clone().unwrap_or(FinishReason::Stop)
			};
			finish = merge_finish(finish, choice_finish);
			for tool in state.tools.values_mut() {
				if tool.completed {
					continue;
				}
				if !tool.started || tool.name.is_empty() {
					return Err(protocol_error(committed, None));
				}
				serde_json::from_slice::<Box<serde_json::value::RawValue>>(&tool.arguments)
					.map_err(|_| protocol_error(committed, None))?;
				tool.completed = true;
				emit(RawEvent::ToolCallComplete {
					index: tool.block,
					call:  UnvalidatedToolCall {
						id:         tool.id.clone(),
						name:       tool.name.clone(),
						input_kind: crate::codec::ToolInputKind::Json,
						arguments:  tool.arguments.clone().freeze(),
					},
				});
			}
		}
		self.done = true;
		emit(RawEvent::Completion(RawCompletion {
			reason: finish,
			blocks: self.next_block,
			usage:  self.usage,
		}));
		Ok(())
	}

	fn decode_error(&self, code: Option<Str>) -> Error {
		Error::new(
			ErrorKind::Protocol,
			if self.committed {
				ErrorPhase::Streaming
			} else {
				ErrorPhase::Handshake
			},
			RetryAction::Never,
			ExecutionReceipt::default(),
		)
		.optional_code(code)
		.committed(self.committed)
	}
}

#[derive(Deserialize)]
struct WireChunk {
	#[serde(default)]
	choices: Vec<WireChoice>,
	#[serde(default)]
	usage:   Option<WireUsage>,
	#[serde(default)]
	error:   Option<WireError>,
}

#[derive(Deserialize)]
struct WireChoice {
	index:         u32,
	#[serde(default)]
	delta:         Option<WirePayload>,
	#[serde(default)]
	message:       Option<WirePayload>,
	#[serde(default)]
	finish_reason: Option<WireFinishReason>,
}

#[derive(Default, Deserialize)]
struct WirePayload {
	#[serde(default)]
	content:           Option<WireDeltaContent>,
	#[serde(default, rename = "text")]
	text:              Option<Str>,
	#[serde(default)]
	reasoning_content: Option<Str>,
	#[serde(default)]
	reasoning_text:    Option<Str>,
	#[serde(default)]
	reasoning:         Option<Str>,
	#[serde(default)]
	refusal:           Option<Str>,
	#[serde(default)]
	tool_calls:        Vec<WireToolCallDelta>,
	#[serde(default)]
	reasoning_details: Vec<WireReasoningDetail>,
	#[serde(default)]
	error:             Option<WireError>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum WireDeltaContent {
	Text(Str),
	Parts(Vec<WireTextPart>),
}

impl WireDeltaContent {
	fn into_text(self) -> Str {
		match self {
			Self::Text(text) => text,
			Self::Parts(parts) => {
				let mut output = String::new();
				for part in parts {
					output.push_str(part.text.as_str());
				}
				Str::from(output)
			},
		}
	}
}

#[derive(Deserialize)]
struct WireTextPart {
	#[serde(rename = "type", default)]
	_kind: Option<TextPartKind>,
	text:  Str,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum TextPartKind {
	Text,
	OutputText,
}
#[derive(Deserialize)]
struct WireToolCallDelta {
	#[serde(default)]
	index:    Option<u32>,
	#[serde(default)]
	id:       Option<Str>,
	function: WireFunctionDelta,
}

#[derive(Deserialize)]
struct WireFunctionDelta {
	#[serde(default)]
	name:      Option<Str>,
	#[serde(default)]
	arguments: Option<Str>,
}

#[derive(Deserialize)]
struct WireReasoningDetail {
	#[serde(default, rename = "id")]
	_id:  Option<Str>,
	#[serde(default)]
	data: Option<String>,
}

#[derive(Deserialize)]
struct WireUsage {
	#[serde(default)]
	prompt_tokens:             u64,
	#[serde(default)]
	completion_tokens:         u64,
	#[serde(default)]
	cached_tokens:             u64,
	#[serde(default)]
	prompt_cache_hit_tokens:   u64,
	#[serde(default)]
	prompt_cache_miss_tokens:  u64,
	#[serde(default)]
	prompt_tokens_details:     WirePromptDetails,
	#[serde(default)]
	completion_tokens_details: WireCompletionDetails,
}

impl WireUsage {
	fn canonical(self) -> Usage {
		let cache_read = self
			.prompt_tokens_details
			.cached_tokens
			.max(self.cached_tokens)
			.max(self.prompt_cache_hit_tokens);
		let cache_write =
			self
				.prompt_tokens_details
				.cache_write_tokens
				.max(if self.prompt_cache_hit_tokens > 0 {
					self.prompt_cache_miss_tokens
				} else {
					0
				});
		Usage {
			input_tokens: self.prompt_tokens,
			output_tokens: self.completion_tokens,
			reasoning_tokens: self.completion_tokens_details.reasoning_tokens,
			cache_read_tokens: cache_read,
			cache_write_tokens: cache_write,
			source: UsageSource::Provider,
			..Usage::default()
		}
	}
}

#[derive(Default, Deserialize)]
struct WirePromptDetails {
	#[serde(default)]
	cached_tokens:      u64,
	#[serde(default)]
	cache_write_tokens: u64,
}

#[derive(Default, Deserialize)]
struct WireCompletionDetails {
	#[serde(default)]
	reasoning_tokens: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireFinishReason {
	Stop,
	End,
	EndTurn,
	ToolCalls,
	FunctionCall,
	ToolUse,
	Length,
	MaxTokens,
	MaxOutputTokens,
	ContentFilter,
	Safety,
	InsufficientSystemResource,
}

impl WireFinishReason {
	fn normalize(self) -> FinishReason {
		match self {
			Self::Stop | Self::End | Self::EndTurn => FinishReason::Stop,
			Self::ToolCalls | Self::FunctionCall | Self::ToolUse => FinishReason::ToolCalls,
			Self::Length | Self::MaxTokens | Self::MaxOutputTokens => FinishReason::Length,
			Self::ContentFilter | Self::Safety => FinishReason::ContentFilter,
			Self::InsufficientSystemResource => {
				unreachable!("resource finish is handled before normalization")
			},
		}
	}
}

#[derive(Deserialize)]
struct WireError {
	#[serde(default)]
	code:     Option<ErrorCode>,
	#[serde(default)]
	message:  Option<Str>,
	#[serde(default, rename = "param")]
	_param:   Option<Str>,
	#[serde(default)]
	metadata: Option<ErrorMetadata>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ErrorCode {
	Text(Str),
	Number(i64),
}

impl ErrorCode {
	fn text(&self) -> Str {
		match self {
			Self::Text(value) => value.clone(),
			Self::Number(value) => value.to_str(),
		}
	}
}

#[derive(Deserialize)]
struct ErrorMetadata {
	#[serde(default)]
	raw: Option<Str>,
}

fn classify_error(error: WireError, committed: bool) -> Error {
	let status = match error.code.as_ref() {
		Some(ErrorCode::Number(value)) => u16::try_from(*value).ok(),
		_ => None,
	};
	let code = error.code.as_ref().map(ErrorCode::text);
	let code_text = code.as_ref().map(Str::as_str).unwrap_or_default();
	let message = error.message.as_ref().map(Str::as_str).unwrap_or_default();
	// DashScope / Bailian reports its per-minute TPM/TPS throttle (429
	// `Throttling.AllocationQuota`, type `insufficient_quota`) with
	// OpenAI-compatible billing wording, but links the error-code doc's
	// `token-limit` anchor. That doc anchor also covers permanent errors such
	// as "Free allocated quota exceeded", so BOTH the anchor and the exact
	// throttle wording are required. The identical wording WITHOUT the anchor
	// is OpenAI's real account-quota error and stays quota-exhausted.
	let dashscope_token_limit = code_text == "insufficient_quota"
		&& dashscope_token_limit_anchor(message.as_bytes())
		&& contains_ascii_case_insensitive(
			message.as_bytes(),
			b"you exceeded your current quota, please check your plan and billing details",
		);
	// llama.cpp reports deterministic tool-call JSON parse failures as HTTP
	// 500; replaying the same prompt produces the same malformed output, so
	// they never enter the transient lane.
	let deterministic_parse_failure =
		contains_ascii_case_insensitive(
			message.as_bytes(),
			b"failed to parse tool call arguments as json",
		) || contains_ascii_case_insensitive(message.as_bytes(), b"[json.exception.parse_error.101]");
	let transient_server_error = !deterministic_parse_failure
		&& matches!(
			code_text,
			"500"
				| "502" | "503"
				| "529" | "server_error"
				| "internal_server_error"
				| "overloaded_error"
		);
	let (kind, action) = if matches!(code_text, "invalid_api_key" | "authentication_error" | "401") {
		(ErrorKind::Authentication, RetryAction::Never)
	} else if matches!(code_text, "permission_denied" | "403") {
		(ErrorKind::Authorization, RetryAction::Never)
	} else if matches!(code_text, "rate_limit_exceeded" | "429") {
		// Transient rate throttle: short backoff on the same credential.
		(ErrorKind::RateLimited, RetryAction::SameRoute { after: Duration::from_secs(30) })
	} else if code_text == "insufficient_quota" && dashscope_token_limit {
		(ErrorKind::RateLimited, RetryAction::SameRoute { after: Duration::from_secs(30) })
	} else if code_text == "insufficient_quota" {
		(ErrorKind::QuotaExhausted, RetryAction::Never)
	} else if matches!(code_text, "content_filter" | "safety") {
		(ErrorKind::ContentFilter, RetryAction::Never)
	} else if code_text == "context_length_exceeded" || message.contains("context length") {
		(ErrorKind::ContextOverflow, RetryAction::Never)
	} else if transient_server_error {
		// A oneshot blip (HTTP 500/502/503/529, provider overload) replays
		// safely pre-commit; the retry layer bounds attempts and refuses once
		// output committed.
		(ErrorKind::ResourceExhausted, RetryAction::SameRoute { after: Duration::from_millis(500) })
	} else if matches!(code_text, "402" | "payment_required") {
		(ErrorKind::PaymentRequired, RetryAction::Never)
	} else if matches!(code_text, "400" | "invalid_request_error") {
		(ErrorKind::InvalidRequest, RetryAction::Never)
	} else {
		(ErrorKind::ProviderContractMismatch, RetryAction::Never)
	};
	Error::new(
		kind,
		if committed {
			ErrorPhase::Streaming
		} else {
			ErrorPhase::Handshake
		},
		action,
		ExecutionReceipt::default(),
	)
	.status(status)
	.optional_code(error.metadata.and_then(|metadata| metadata.raw).or(code))
	.committed(committed)
}

/// True when `error-code` and `#token-limit` occur inside one URL-like token,
/// mirroring DashScope's documented `error-code…#token-limit` doc anchor.
fn dashscope_token_limit_anchor(message: &[u8]) -> bool {
	const PREFIX: &[u8] = b"error-code";
	const ANCHOR: &[u8] = b"#token-limit";
	let mut offset = 0;
	while offset + PREFIX.len() <= message.len() {
		let window = &message[offset..];
		if !window[..PREFIX.len()].eq_ignore_ascii_case(PREFIX) {
			offset += 1;
			continue;
		}
		let token = &window[PREFIX.len()..];
		let token_end = token
			.iter()
			.position(|&byte| byte.is_ascii_whitespace() || matches!(byte, b'(' | b')'))
			.unwrap_or(token.len());
		if contains_ascii_case_insensitive(&token[..token_end], ANCHOR) {
			return true;
		}
		offset += 1;
	}
	false
}
fn contains_ascii_case_insensitive(haystack: &[u8], needle: &[u8]) -> bool {
	haystack
		.windows(needle.len())
		.any(|window| window.eq_ignore_ascii_case(needle))
}

fn incomplete_stream_error() -> Error {
	Error::new(
		ErrorKind::StreamCorruption,
		ErrorPhase::Streaming,
		RetryAction::SemanticRetry,
		ExecutionReceipt::default(),
	)
	.code(Str::new_static("openai_chat.incomplete_stream"))
	.committed(true)
}

fn resource_finish_error(committed: bool) -> Error {
	Error::new(
		ErrorKind::ResourceExhausted,
		if committed {
			ErrorPhase::Streaming
		} else {
			ErrorPhase::Handshake
		},
		RetryAction::SemanticRetry,
		ExecutionReceipt::default(),
	)
	.code(Str::new_static("insufficient_system_resource"))
	.committed(committed)
}

fn protocol_error(committed: bool, code: Option<Str>) -> Error {
	Error::new(
		ErrorKind::Protocol,
		if committed {
			ErrorPhase::Streaming
		} else {
			ErrorPhase::Handshake
		},
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
	.optional_code(code)
	.committed(committed)
}

const fn merge_usage(current: &mut Usage, update: Usage) {
	if update.input_tokens != 0 {
		current.input_tokens = update.input_tokens;
	}
	if update.output_tokens != 0 {
		current.output_tokens = update.output_tokens;
	}
	if update.reasoning_tokens != 0 {
		current.reasoning_tokens = update.reasoning_tokens;
	}
	if update.cache_read_tokens != 0 {
		current.cache_read_tokens = update.cache_read_tokens;
	}
	if update.cache_write_tokens != 0 {
		current.cache_write_tokens = update.cache_write_tokens;
	}
	current.source = UsageSource::Provider;
}

fn merge_finish(current: FinishReason, incoming: FinishReason) -> FinishReason {
	const fn rank(reason: &FinishReason) -> u8 {
		match reason {
			FinishReason::ContentFilter => 4,
			FinishReason::ToolCalls => 3,
			FinishReason::Length => 2,
			FinishReason::Stop => 1,
			_ => 0,
		}
	}
	if rank(&incoming) > rank(&current) {
		incoming
	} else {
		current
	}
}

fn capability_error() -> Error {
	Error::new(
		ErrorKind::CapabilityMismatch,
		ErrorPhase::Encoding,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}

fn tool_capability_error(reason: &'static str) -> Error {
	capability_error().code(Str::new_static(reason))
}

fn encoding_error(kind: ErrorKind) -> Error {
	Error::new(kind, ErrorPhase::Encoding, RetryAction::Never, ExecutionReceipt::default())
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use bytes::Bytes;
	use serde::Deserialize;

	use super::{OpenAiChatCodec, OpenAiChatDecoder, OpenAiChatProfile, ToolIdWireProfile};
	use crate::{
		call::{
			ChatRequest, ContentPart, Message, NegotiationPolicy, OpaqueJson, Role, Sampling, Setting,
			ToolDefinition, ToolInputConstraint, ToolResultContent,
		},
		codec::{Decoder, RawEvent},
		error::{ErrorKind, RetryAction},
		event::{ChatEvent, FinishReason},
		transport::{Frame, SseDecoder},
	};

	fn request(messages: Arc<[Message]>) -> ChatRequest {
		ChatRequest {
			messages,
			tools: Arc::from([]),
			hosted_tools: Arc::from([]),
			tool_choice: Setting::Unset,
			output: Setting::Unset,
			reasoning: Setting::Unset,
			verbosity: Setting::Unset,
			cache_retention: Setting::Unset,
			service_tier: Setting::Unset,
			sampling: Sampling::default(),
			max_output_tokens: None,
			top_logprobs: None,
			safety: Arc::from([]),
			negotiation: NegotiationPolicy::default(),
		}
	}

	fn text_message(text: &str) -> Message {
		Message {
			role:    Role::User,
			content: Arc::from([ContentPart::Text { text: text.into(), proof: None }]),
			name:    None,
		}
	}

	fn decode_fixture(source: &str) -> Result<Vec<RawEvent>, crate::error::Error> {
		let mut framer = SseDecoder::new();
		let mut decoder = OpenAiChatDecoder::default();
		let mut events = Vec::new();
		for chunk in source.as_bytes().chunks(7) {
			for event in framer
				.push(Bytes::copy_from_slice(chunk))
				.expect("valid SSE fixture")
			{
				decoder.push(Frame::Sse(event), &mut |event| events.push(event))?;
			}
		}
		for event in framer.finish().expect("complete SSE fixture") {
			decoder.push(Frame::Sse(event), &mut |event| events.push(event))?;
		}
		decoder.finish(&mut |event| events.push(event))?;
		Ok(events)
	}

	#[test]
	fn plain_request_matches_exact_wire_bytes() {
		let codec = OpenAiChatCodec::default();
		let request = request(Arc::from([text_message("Say hello.")]));
		let bytes = codec
			.encode_chat("gpt-4.1", &request)
			.expect("request encodes");
		assert_eq!(
			bytes.as_ref(),
			br#"{"model":"gpt-4.1","messages":[{"role":"user","content":"Say hello."}],"stream":true,"stream_options":{"include_usage":true}}"#,
		);
	}

	#[derive(Deserialize)]
	struct StrictEnvelope {
		tools: Vec<StrictTool>,
	}

	#[derive(Deserialize)]
	struct StrictTool {
		function: StrictFunction,
	}

	#[derive(Deserialize)]
	struct StrictFunction {
		strict:     bool,
		parameters: StrictObject,
	}

	#[derive(Deserialize)]
	#[serde(rename_all = "camelCase")]
	struct StrictObject {
		required:              Vec<String>,
		additional_properties: bool,
		properties:            serde_json::Map<String, serde_json::Value>,
	}

	#[test]
	fn strict_tools_close_objects_and_require_every_property() {
		let mut request = request(Arc::from([text_message("lookup")]));
		request.tools = Arc::from([ToolDefinition {
			name:        "lookup".into(),
			description: None,
			input:       ToolInputConstraint::JsonSchema {
				parameters: OpaqueJson::new(
					serde_json::from_str(r#"{"type":"object","properties":{"q":{"type":"string"}}}"#)
						.expect("schema fixture"),
				),
				strict:     true,
			},
		}]);
		let bytes = OpenAiChatCodec::default()
			.encode_chat("gpt", &request)
			.expect("request encodes");
		let decoded: StrictEnvelope = serde_json::from_slice(&bytes).expect("typed wire request");
		let function = &decoded.tools[0].function;
		assert!(function.strict);
		assert_eq!(function.parameters.required, ["q"]);
		assert!(!function.parameters.additional_properties);
		assert!(function.parameters.properties.contains_key("q"));
	}

	#[derive(Deserialize)]
	struct UnionEnvelope {
		#[serde(default)]
		tools:       Vec<UnionTool>,
		#[serde(default)]
		tool_choice: Option<serde_json::Value>,
	}

	#[derive(Deserialize)]
	struct UnionTool {
		function: UnionFunction,
	}

	#[derive(Deserialize)]
	struct UnionFunction {
		name:       String,
		parameters: serde_json::Value,
	}

	fn json_tool(name: &str, parameters: serde_json::Value) -> ToolDefinition {
		ToolDefinition {
			name:        name.into(),
			description: None,
			input:       ToolInputConstraint::JsonSchema {
				parameters: OpaqueJson::new(parameters),
				strict:     false,
			},
		}
	}

	/// Exclusive-required MCP shape from pi issue #2652 / the xAI 400 class.
	fn coverage_tool() -> ToolDefinition {
		json_tool(
			"mcp__codebase_memory_check_index_coverage",
			serde_json::json!({
				"type": "object",
				"properties": {
					"project": {"type": "string"},
					"paths": {"type": "array", "items": {"type": "string"}},
					"scopes": {"type": "array", "items": {"type": "string"}}
				},
				"required": ["project"],
				"anyOf": [{"required": ["paths"]}, {"required": ["scopes"]}]
			}),
		)
	}

	/// A root union whose branches carry real constraints and must survive.
	fn leftover_tool() -> ToolDefinition {
		json_tool(
			"mcp__leftover_union",
			serde_json::json!({
				"type": "object",
				"properties": {"kind": {"type": "string"}},
				"anyOf": [
					{"required": ["kind"], "minProperties": 1},
					{"required": ["kind"], "minProperties": 2}
				]
			}),
		)
	}

	fn good_tool() -> ToolDefinition {
		json_tool(
			"read_file",
			serde_json::json!({
				"type": "object",
				"properties": {"path": {"type": "string"}},
				"required": ["path"]
			}),
		)
	}

	fn flatten_codec() -> OpenAiChatCodec {
		OpenAiChatCodec::new(
			super::OpenAiChatProfile {
				flatten_root_unions: true,
				..super::OpenAiChatProfile::default()
			},
			None,
		)
	}

	fn encode_union_request(
		codec: &OpenAiChatCodec,
		tools: Arc<[ToolDefinition]>,
		tool_choice: Setting<crate::call::ToolChoice>,
	) -> UnionEnvelope {
		let mut request = request(Arc::from([text_message("check coverage")]));
		request.tools = tools;
		request.tool_choice = tool_choice;
		let bytes = codec
			.encode_chat("grok-4", &request)
			.expect("request encodes");
		serde_json::from_slice(&bytes).expect("typed wire request")
	}

	fn tool_names(envelope: &UnionEnvelope) -> Vec<&str> {
		envelope
			.tools
			.iter()
			.map(|tool| tool.function.name.as_str())
			.collect()
	}

	#[test]
	fn flatten_profile_flattens_exclusive_required_root_union() {
		let envelope = encode_union_request(
			&flatten_codec(),
			Arc::from([coverage_tool(), good_tool()]),
			Setting::Unset,
		);
		assert_eq!(tool_names(&envelope), ["mcp__codebase_memory_check_index_coverage", "read_file"]);
		assert!(envelope.tools[0].function.parameters.get("anyOf").is_none());
		assert!(
			envelope.tools[0]
				.function
				.parameters
				.get("required")
				.is_some()
		);
	}

	#[test]
	fn default_profile_preserves_exclusive_required_root_union() {
		let envelope = encode_union_request(
			&OpenAiChatCodec::default(),
			Arc::from([coverage_tool(), good_tool()]),
			Setting::Unset,
		);
		assert_eq!(tool_names(&envelope), ["mcp__codebase_memory_check_index_coverage", "read_file"]);
		assert_eq!(
			envelope.tools[0]
				.function
				.parameters
				.get("anyOf")
				.and_then(serde_json::Value::as_array)
				.map(Vec::len),
			Some(2)
		);
	}

	#[test]
	fn default_profile_keeps_leftover_object_root_union() {
		let envelope = encode_union_request(
			&OpenAiChatCodec::default(),
			Arc::from([leftover_tool(), good_tool()]),
			Setting::Unset,
		);
		assert_eq!(tool_names(&envelope), ["mcp__leftover_union", "read_file"]);
		assert_eq!(
			envelope.tools[0]
				.function
				.parameters
				.get("anyOf")
				.and_then(serde_json::Value::as_array)
				.map(Vec::len),
			Some(2)
		);
	}

	#[test]
	fn flatten_profile_withholds_leftover_object_root_union() {
		let envelope = encode_union_request(
			&flatten_codec(),
			Arc::from([leftover_tool(), good_tool()]),
			Setting::Unset,
		);
		assert_eq!(tool_names(&envelope), ["read_file"]);
	}

	#[test]
	fn forced_choice_naming_a_withheld_tool_is_dropped() {
		let envelope = encode_union_request(
			&flatten_codec(),
			Arc::from([leftover_tool(), good_tool()]),
			Setting::Require(crate::call::ToolChoice::Named("mcp__leftover_union".into())),
		);
		assert_eq!(tool_names(&envelope), ["read_file"]);
		assert!(envelope.tool_choice.is_none());
	}

	#[test]
	fn required_choice_is_dropped_once_withholding_empties_the_tools() {
		let envelope = encode_union_request(
			&flatten_codec(),
			Arc::from([leftover_tool()]),
			Setting::Require(crate::call::ToolChoice::Required),
		);
		assert!(envelope.tools.is_empty());
		assert!(envelope.tool_choice.is_none());
	}

	#[test]
	fn root_union_flattening_is_scoped_to_the_tool_root() {
		// Nested exclusive-required unions are real constraints; only the tool
		// root 400s xAI.
		let nested = serde_json::json!({
			"type": "object",
			"properties": {
				"outputSchema": {
					"type": "object",
					"properties": {
						"paths": {"type": "array", "items": {"type": "string"}},
						"scopes": {"type": "array", "items": {"type": "string"}}
					},
					"anyOf": [{"required": ["paths"]}, {"required": ["scopes"]}]
				}
			},
			"required": ["outputSchema"]
		});
		assert_eq!(super::flatten_exclusive_required_root_union(&nested), None);
		assert!(!super::leftover_root_object_union(&nested));

		let flattened = super::flatten_exclusive_required_root_union(
			coverage_tool()
				.input
				.json_schema()
				.expect("json schema")
				.0
				.as_value(),
		)
		.expect("exclusive-required root union flattens");
		assert!(flattened.get("anyOf").is_none());
		assert_eq!(flattened.get("required"), Some(&serde_json::json!(["project"])));
	}

	#[test]
	fn root_union_constraining_properties_is_not_flattened_but_is_withheld() {
		let constraining = serde_json::json!({
			"type": "object",
			"properties": {"kind": {"type": "string"}},
			"anyOf": [
				{"properties": {"kind": {"const": "a"}}},
				{"properties": {"kind": {"const": "b"}}}
			]
		});
		assert_eq!(super::flatten_exclusive_required_root_union(&constraining), None);
		assert!(super::leftover_root_object_union(&constraining));
	}

	#[test]
	fn pure_and_object_branch_root_unions_are_not_leftover_violations() {
		// A pure (untyped, property-less) root union is not this error class.
		let pure = serde_json::json!({
			"anyOf": [{"type": "string"}, {"type": "number"}]
		});
		assert!(!super::leftover_root_object_union(&pure));
		// All-object branches are valid object-root unions everywhere.
		let object_branches = serde_json::json!({
			"type": "object",
			"properties": {"kind": {"type": "string"}},
			"oneOf": [
				{"type": "object", "required": ["kind"]},
				{"type": "object", "properties": {"extra": {"type": "string"}}}
			]
		});
		assert!(!super::leftover_root_object_union(&object_branches));
	}

	#[test]
	fn fragmented_tool_arguments_remain_byte_exact_and_unvalidated() {
		let events = decode_fixture(include_str!(
			"../../../../fixtures/llm-oracle/openai/chat/stream.tool_reasoning_usage.sse"
		))
		.expect("fixture decodes");
		let mut arguments = Vec::new();
		let mut complete = None;
		let mut finish = None;
		for event in events {
			match event {
				RawEvent::Chat(ChatEvent::ToolArgumentsDelta { bytes, .. }) => {
					arguments.extend_from_slice(&bytes);
				},
				RawEvent::ToolCallComplete { call, .. } => complete = Some(call),
				RawEvent::Completion(completion) => finish = Some(completion.reason),
				_ => {},
			}
		}
		assert_eq!(arguments, r#"{"city":"Zürich"}"#.as_bytes());
		let complete = complete.expect("complete tool input");
		assert_eq!(complete.name.as_str(), "lookup_weather");
		assert_eq!(complete.arguments.as_ref(), arguments);
		assert_eq!(finish, Some(FinishReason::ToolCalls));
	}

	#[test]
	fn parity_fixture_preserves_usage_and_finish_precedence() {
		let events = decode_fixture(include_str!(
			"../../../../fixtures/llm-oracle/openai/chat/stream.parity.sse"
		))
		.expect("fixture decodes");
		let usage = events
			.iter()
			.filter_map(|event| match event {
				RawEvent::Chat(ChatEvent::Usage(update)) => Some(update.usage),
				_ => None,
			})
			.last()
			.expect("usage event");
		assert_eq!(usage.input_tokens, 10);
		assert_eq!(usage.output_tokens, 4);
		assert_eq!(usage.reasoning_tokens, 2);
		assert_eq!(usage.cache_read_tokens, 6);
		assert_eq!(usage.cache_write_tokens, 2);
		let finish = events
			.iter()
			.find_map(|event| match event {
				RawEvent::Completion(completion) => Some(&completion.reason),
				_ => None,
			})
			.expect("completion");
		assert_eq!(finish, &FinishReason::ContentFilter);
	}

	#[test]
	fn typed_error_envelopes_preserve_classification_evidence() {
		// The OpenRouter fixture is a gateway 502 for an upstream model failure
		// (metadata.raw carries the upstream detail). pi auto-retries bare
		// gateway upstream failures as transient (PR #8370's 429/500/502/503/529
		// set), so the 502 classifies as a retryable capacity failure rather
		// than a terminal provider-contract mismatch.
		for (fixture, kind, code) in [
			(
				include_bytes!("../../../../fixtures/llm-oracle/openai/chat/error.azure.json")
					.as_slice(),
				ErrorKind::ContentFilter,
				"content_filter",
			),
			(
				include_bytes!("../../../../fixtures/llm-oracle/openai/chat/error.openrouter.json")
					.as_slice(),
				ErrorKind::ResourceExhausted,
				"MALFORMED_FUNCTION_CALL",
			),
			(
				br#"{"error":{"code":429,"message":"rate limited"}}"#.as_slice(),
				ErrorKind::RateLimited,
				"429",
			),
		] {
			let mut decoder = OpenAiChatDecoder::default();
			let mut events = Vec::new();
			decoder
				.push(
					Frame::Sse(crate::transport::SseEvent {
						name: None,
						data: Bytes::copy_from_slice(fixture),
					}),
					&mut |event| events.push(event),
				)
				.expect("typed provider error decodes");
			let error = events
				.into_iter()
				.find_map(|event| match event {
					RawEvent::Failure(error) => Some(error),
					_ => None,
				})
				.expect("terminal error");
			assert_eq!(error.kind, kind);
			assert_eq!(error.code.as_ref().map(|value| value.as_str()), Some(code));
			assert!(!error.committed);
		}
	}

	#[test]
	fn dashscope_token_limit_requires_anchor_and_exact_billing_wording() {
		let classify = |message: &str| {
			super::classify_error(
				super::WireError {
					code:     Some(super::ErrorCode::Text("insufficient_quota".into())),
					message:  Some(message.into()),
					_param:   None,
					metadata: None,
				},
				false,
			)
		};
		// Documented Bailian TPM/TPS throttle: billing wording plus doc anchor.
		let throttle = classify(
			"You exceeded your current quota, please check your plan and billing details. See \
			 https://help.aliyun.com/zh/model-studio/error-code#token-limit \
			 (type=insufficient_quota param=insufficient_quota)",
		);
		assert_eq!(throttle.kind, ErrorKind::RateLimited);
		assert!(matches!(throttle.action, RetryAction::SameRoute { .. }));

		// OpenAI's real account-quota error: identical wording, no doc anchor.
		let openai_quota = classify(
			"You exceeded your current quota, please check your plan and billing details. See \
			 https://platform.openai.com/account/usage",
		);
		assert_eq!(openai_quota.kind, ErrorKind::QuotaExhausted);
		assert_eq!(openai_quota.action, RetryAction::Never);

		// Same doc anchor with permanent wording ("Free allocated quota
		// exceeded") must stay quota-exhausted: the anchor alone is not a
		// throttle signature.
		let free_quota = classify(
			"Free allocated quota exceeded. See \
			 https://help.aliyun.com/zh/model-studio/error-code#token-limit",
		);
		assert_eq!(free_quota.kind, ErrorKind::QuotaExhausted);
		assert_eq!(free_quota.action, RetryAction::Never);

		// Anchor split across separate tokens is not the documented signature.
		let split_anchor = classify(
			"You exceeded your current quota, please check your plan and billing details. See \
			 error-code docs (#token-limit)",
		);
		assert_eq!(split_anchor.kind, ErrorKind::QuotaExhausted);
		assert_eq!(split_anchor.action, RetryAction::Never);
	}

	#[test]
	fn transient_provider_blips_retry_same_route_and_deterministic_denials_do_not() {
		let classify = |code: &str, message: &str| {
			super::classify_error(
				super::WireError {
					code:     Some(super::ErrorCode::Text(code.into())),
					message:  Some(message.into()),
					_param:   None,
					metadata: None,
				},
				false,
			)
		};
		// The oneshot transient set: HTTP 429/500/502/503/529 and overload.
		for code in ["429", "rate_limit_exceeded"] {
			let error = classify(code, "slow down");
			assert_eq!(error.kind, ErrorKind::RateLimited, "{code}");
			assert!(matches!(error.action, RetryAction::SameRoute { .. }), "{code}");
		}
		for code in ["500", "502", "503", "529", "server_error", "overloaded_error"] {
			let error = classify(code, "upstream blip");
			assert_eq!(error.kind, ErrorKind::ResourceExhausted, "{code}");
			assert!(matches!(error.action, RetryAction::SameRoute { .. }), "{code}");
		}
		// Deterministic denials never enter the transient lane.
		for (code, kind) in [
			("invalid_api_key", ErrorKind::Authentication),
			("permission_denied", ErrorKind::Authorization),
			("content_filter", ErrorKind::ContentFilter),
			("invalid_request_error", ErrorKind::InvalidRequest),
		] {
			let error = classify(code, "denied");
			assert_eq!(error.kind, kind, "{code}");
			assert_eq!(error.action, RetryAction::Never, "{code}");
		}
		// llama.cpp deterministic tool-call parse failures arrive as HTTP 500
		// but replay identically: never transient.
		let parse_failure = classify(
			"500",
			"Failed to parse tool call arguments as JSON: [json.exception.parse_error.101]",
		);
		assert_eq!(parse_failure.action, RetryAction::Never);
	}

	#[test]
	fn context_overflow_classifies_as_never_retried() {
		// A deterministic overflow replays identically: the summarizer (or any
		// caller) must see a fail-fast `ContextOverflow` rather than a transient
		// classification that burns the attempt budget on identical calls.
		let classify = |code: &str, message: &str| {
			super::classify_error(
				super::WireError {
					code:     Some(super::ErrorCode::Text(code.into())),
					message:  Some(message.into()),
					_param:   None,
					metadata: None,
				},
				false,
			)
		};
		let coded = classify("context_length_exceeded", "input exceeds the model window");
		assert_eq!(coded.kind, ErrorKind::ContextOverflow);
		assert_eq!(coded.action, RetryAction::Never);

		let textual = classify(
			"400",
			"This model's maximum context length is 200000 tokens, however you requested 3031925 \
			 tokens",
		);
		assert_eq!(textual.kind, ErrorKind::ContextOverflow);
		assert_eq!(textual.action, RetryAction::Never);
	}

	#[test]
	fn truncated_content_stream_and_resource_finish_are_retryable_failures() {
		let truncated = match decode_fixture(
			"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n",
		) {
			Ok(_) => panic!("content without a finish reason must be truncated"),
			Err(error) => error,
		};
		assert_eq!(truncated.kind, ErrorKind::StreamCorruption);
		assert_eq!(truncated.action, RetryAction::SemanticRetry);
		let empty = decode_fixture("").expect("empty close remains an empty completion");
		assert!(empty.iter().any(|event| {
			matches!(event, RawEvent::Completion(completion) if completion.reason == FinishReason::Stop)
		}));

		let events = decode_fixture(
			"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\ndata: \
			 {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"\
			 insufficient_system_resource\"}]}\n\ndata: [DONE]\n\n",
		)
		.expect("resource finish decodes to a typed failure");
		let failure = events
			.into_iter()
			.find_map(|event| match event {
				RawEvent::Failure(error) => Some(error),
				_ => None,
			})
			.expect("resource finish emits a failure");
		assert_eq!(failure.kind, ErrorKind::ResourceExhausted);
		assert_eq!(failure.action, RetryAction::SemanticRetry);
	}

	#[test]
	fn wrong_known_field_type_is_rejected_without_value_fallback() {
		let mut decoder = OpenAiChatDecoder::default();
		let error = decoder
			.push(
				Frame::Sse(crate::transport::SseEvent {
					name: None,
					data: Bytes::from_static(
						br#"{"choices":[{"index":0,"delta":{"content":7},"finish_reason":null}]}"#,
					),
				}),
				&mut |_| {},
			)
			.expect_err("numeric content is not a Chat Completions content shape");
		assert_eq!(error.kind, ErrorKind::Protocol);
		assert!(!error.committed);
	}

	#[test]
	fn terminal_decoder_is_idempotent() {
		let mut decoder = OpenAiChatDecoder::default();
		let mut events = Vec::new();
		decoder
			.finish(&mut |event| events.push(event))
			.expect("first finish");
		let count = events.len();
		decoder
			.finish(&mut |event| events.push(event))
			.expect("second finish");
		assert_eq!(events.len(), count);
	}

	#[test]
	fn opaque_tool_call_ids_round_trip_verbatim_on_unconstrained_routes() {
		// Provider-issued correlation tokens (vLLM/SGLang gateways emit pipe-
		// and plus-laden ids) must replay byte-for-byte on the route that
		// issued them; only a route whose policy declares the OpenAI
		// 40-character projection may rewrite them.
		let opaque = "call_abc||gateway_state||opaque+/=";
		let chunk = serde_json::json!({
			"choices": [{"index": 0, "delta": {"tool_calls": [{
				"index": 0,
				"id": opaque,
				"type": "function",
				"function": {"name": "read", "arguments": "{\"path\":\"README.md\"}"}
			}]}}]
		});
		let fixture = format!(
			"data: {chunk}\n\ndata: \
			 {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\\
			 ndata: [DONE]\n\n"
		);
		let events = decode_fixture(&fixture).expect("tool stream decodes");
		let call = events
			.iter()
			.find_map(|event| match event {
				RawEvent::ToolCallComplete { call, .. } => Some(call),
				_ => None,
			})
			.expect("complete tool call");
		assert_eq!(call.id.as_str(), opaque);

		let replay = request(Arc::from([
			Message {
				role:    Role::User,
				content: Arc::from([ContentPart::Text { text: "Read README".into(), proof: None }]),
				name:    None,
			},
			Message {
				role:    Role::Assistant,
				content: Arc::from([ContentPart::ToolCall {
					call:      call.id.clone(),
					name:      call.name.clone(),
					arguments: OpaqueJson::new(serde_json::json!({"path": "README.md"})),
					proof:     None,
				}]),
				name:    None,
			},
			Message {
				role:    Role::Tool,
				content: Arc::from([ContentPart::ToolResult {
					call:     call.id.clone(),
					name:     Some("read".into()),
					content:  Arc::from([ToolResultContent::Text("done".into())]),
					is_error: false,
				}]),
				name:    None,
			},
		]));
		let body = OpenAiChatCodec::default()
			.encode_chat("gateway-model", &replay)
			.expect("replay encodes");
		let wire: serde_json::Value = serde_json::from_slice(&body).expect("wire body is JSON");
		assert_eq!(wire["messages"][1]["tool_calls"][0]["id"], opaque);
		assert_eq!(wire["messages"][2]["tool_call_id"], opaque);

		// The 40-character projection stays a route-scoped policy, not a
		// blanket rewrite of provider-issued ids.
		let profile =
			OpenAiChatProfile { tool_id: ToolIdWireProfile::OpenAi40, ..OpenAiChatProfile::default() };
		let projected = OpenAiChatCodec::new(profile, None)
			.encode_chat("gpt-4o-mini", &replay)
			.expect("projected replay encodes");
		let wire: serde_json::Value =
			serde_json::from_slice(&projected).expect("projected body is JSON");
		let id = wire["messages"][1]["tool_calls"][0]["id"]
			.as_str()
			.expect("projected id");
		assert!(id.len() <= 40);
		assert!(
			id.chars()
				.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
		);
		assert_eq!(wire["messages"][2]["tool_call_id"].as_str(), Some(id));
	}

	fn thinking_request(effort: crate::catalog::ReasoningEffort) -> ChatRequest {
		let mut request = request(Arc::from([text_message("think")]));
		request.reasoning = Setting::Require(crate::call::ReasoningRequest {
			visibility:          crate::call::ReasoningVisibility::Visible,
			effort:              Some(effort),
			max_tokens:          None,
			preserve_signatures: false,
		});
		request
	}

	#[test]
	fn qwen_template_effort_rides_top_level_and_kwargs() {
		// pi bf490ae024: without routing the selected effort onto the Qwen 3.8+
		// template's `reasoning_effort` kwarg, `enable_thinking` alone leaves
		// the model reasoning at its xhigh default. Twin emission: top-level
		// for newer llama.cpp builds, kwargs for older builds.
		let profile = OpenAiChatProfile {
			reasoning: super::ReasoningWireFormat::Qwen,
			template_reasoning_effort: true,
			..OpenAiChatProfile::default()
		};
		let body = OpenAiChatCodec::new(profile, None)
			.encode_chat("qwen3.8-27b", &thinking_request(crate::catalog::ReasoningEffort::Medium))
			.expect("request encodes");
		let wire: serde_json::Value = serde_json::from_slice(&body).expect("wire body is JSON");
		assert_eq!(wire["enable_thinking"], true);
		assert_eq!(wire["reasoning_effort"], "medium");
		assert_eq!(wire["chat_template_kwargs"]["reasoning_effort"], "medium");
		assert!(
			wire["chat_template_kwargs"]
				.as_object()
				.expect("kwargs object")
				.get("enable_thinking")
				.is_none(),
			"the qwen dialect keeps enable_thinking top-level only"
		);
	}

	#[test]
	fn qwen_pre_38_templates_keep_the_bare_enable_thinking_toggle() {
		// Older Qwen templates have no `reasoning_effort` kwarg; leaking one
		// would inject an undefined template variable.
		let profile = OpenAiChatProfile {
			reasoning: super::ReasoningWireFormat::Qwen,
			..OpenAiChatProfile::default()
		};
		let body = OpenAiChatCodec::new(profile, None)
			.encode_chat("qwen3.6-27b", &thinking_request(crate::catalog::ReasoningEffort::High))
			.expect("request encodes");
		let wire: serde_json::Value = serde_json::from_slice(&body).expect("wire body is JSON");
		assert_eq!(wire["enable_thinking"], true);
		assert!(wire.get("reasoning_effort").is_none());
		assert!(wire.get("chat_template_kwargs").is_none());
	}

	#[test]
	fn qwen_chat_template_dialect_rides_kwargs_only() {
		// vLLM/NIM-style schemas reject unknown top-level fields; the effort
		// rides `chat_template_kwargs` alone on the kwargs dialect.
		let profile = OpenAiChatProfile {
			reasoning: super::ReasoningWireFormat::Nvidia,
			template_reasoning_effort: true,
			..OpenAiChatProfile::default()
		};
		let body = OpenAiChatCodec::new(profile, None)
			.encode_chat("qwen3.8-27b", &thinking_request(crate::catalog::ReasoningEffort::Xhigh))
			.expect("request encodes");
		let wire: serde_json::Value = serde_json::from_slice(&body).expect("wire body is JSON");
		assert!(wire.get("reasoning_effort").is_none());
		assert!(wire.get("enable_thinking").is_none());
		assert_eq!(wire["chat_template_kwargs"]["enable_thinking"], true);
		assert_eq!(wire["chat_template_kwargs"]["reasoning_effort"], "xhigh");
	}
}
