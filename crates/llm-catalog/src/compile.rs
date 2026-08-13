//! Deterministic offline compilation of checked-in catalog oracle records.

use std::collections::{BTreeMap, BTreeSet};

use omp_core::{Str, hex};
use serde::{Deserialize, Serialize};
use serde_json::{Number, value::RawValue};
use sha2::{Digest, Sha256};
use smallvec::SmallVec;

use crate::{
	capability::{
		AudioFormatBits, Availability, ChatCapabilities, DimensionRange, EmbeddingCapabilities,
		EmbeddingFormatBits, EmbeddingInputBits, ImageCapabilities, ImageFeatureBits, ModalityBits,
		ModelCapabilities, OperationBits, OperationKind, RealtimeCapabilities, RealtimeFeatureBits,
		SearchCapabilities, SearchFeatureBits, SpeechCapabilities, SpeechFeatureBits,
		TokenizationCapabilities, TokenizationFeatureBits, ToolCapabilities, ToolFeatureBits,
		TranscriptionCapabilities, TranscriptionFeatureBits, VideoCapabilities, VideoFeatureBits,
	},
	classify::{ClassificationInput, ClassificationPhase, ModelClassification, classify},
	discover::DiscoveryDefaults,
	id::{
		AuthSpecId, CatalogRevision, CodecId, DiscoverySpecId, ModelKey, OAuthSpecId, ProviderId,
		RouteId, WireModelId, WirePolicyId,
	},
	model::{
		ContextStrategy, EvidenceConfidence, ModelAvailability, ModelLimits, ModelProvenance,
		ModelRemoteCompaction, ModelSpec, ProvenanceKind, ProvenanceSource,
	},
	policy::{
		ApplyPatchWireKind, ComputerUseConfigSupport, ComputerUseWireSupport, ExtendedContextMode,
		MaxOutputTokensEmission, ReasoningBodyOverride, StreamWatchdog, ToolCallIdProfile,
		WhenThinkingPolicy, WirePolicy,
	},
	pricing::{PremiumMultiplier, Price, PriceTier, PriceUnit, Pricing},
	provider::{
		AccountScope, ApplicationDefaultSource, AuthSpec, AuthSpecKind, CodecProfile,
		CodexTransportPreference, CredentialSourceSpec, DiscoveryKind, DiscoveryPagination,
		DiscoverySpec, EndpointSpec, HeaderProfile, ManagementCapabilities, OAuthCompletion,
		OAuthExchangeKind, OAuthFlowSpec, OAuthParameter, OAuthPollingSpec, OAuthRefreshBehavior,
		OAuthSpec, OAuthTokenPlacement, PrincipalResolution, ProviderDef, RedirectTrust,
		RegionSource, RegistryMapping, RouteDef, RouteRestrictions, SealedBodyPlacement, SigV4Spec,
		StaticHeader, TransportKind, TrustDomain,
	},
	thinking::{ReasoningMode, ThinkingEffort, ThinkingMode, ThinkingPolicy, ThinkingRouting},
};
/// Schema version of reviewable normalized compiler output.
pub const COMPILED_SCHEMA_VERSION: u32 = 1;
/// Verified raw row count of the checked-in oracle.
pub const ORACLE_RAW_MODELS: usize = 4_302;
/// Verified normalized logical model count of the checked-in oracle.
pub const ORACLE_LOGICAL_MODELS: usize = 4_227;
/// Verified curated provider count.
pub const ORACLE_PROVIDERS: usize = 94;
/// Verified number of provider keys present in raw model records.
pub const ORACLE_RAW_PROVIDER_KEYS: usize = 80;
/// Verified number of distinct route URLs.
pub const ORACLE_URLS: usize = 108;
/// Full transport vocabulary size in the oracle.
pub const ORACLE_TRANSPORTS: usize = 16;
/// Transport variants active in the checked-in oracle.
pub const ORACLE_ACTIVE_TRANSPORTS: usize = 13;

/// An explicit opaque source-model property boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RawModelProperties(Box<RawValue>);

impl RawModelProperties {
	/// Borrows the original JSON token sequence.
	pub fn json(&self) -> &str {
		self.0.get()
	}
}

/// Typed source modality.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceModality {
	/// Text.
	Text,
	/// Images.
	Image,
	/// Audio.
	Audio,
	/// Video.
	Video,
	/// PDF or document data.
	Pdf,
}

/// Closed source wire vocabulary retained from model and provider oracles.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum SourceTransport {
	/// Anthropic Messages.
	#[serde(rename = "anthropic-messages")]
	AnthropicMessages,
	/// Anthropic on Bedrock.
	#[serde(rename = "anthropic-bedrock")]
	AnthropicBedrock,
	/// Bedrock Converse.
	#[serde(rename = "bedrock-converse", alias = "bedrock-converse-stream")]
	BedrockConverse,
	/// Anthropic on Vertex.
	#[serde(rename = "anthropic-vertex")]
	AnthropicVertex,
	/// OpenAI Chat Completions.
	#[serde(rename = "open-ai-chat", alias = "openai-completions", alias = "openrouter")]
	OpenAiChat,
	/// OpenAI Responses.
	#[serde(
		rename = "open-ai-responses",
		alias = "openai-responses",
		alias = "azure-openai-responses"
	)]
	OpenAiResponses,
	/// OpenAI Codex.
	#[serde(rename = "open-ai-codex", alias = "openai-codex-responses")]
	OpenAiCodex,
	/// Google Generative AI.
	#[serde(rename = "google-gen-ai", alias = "google-generative-ai")]
	GoogleGenAi,
	/// Google Vertex.
	#[serde(rename = "google-vertex")]
	GoogleVertex,
	/// Google Cloud Code Assist.
	#[serde(rename = "google-cca", alias = "google-gemini-cli")]
	GoogleCca,
	/// Ollama native chat.
	#[serde(rename = "ollama-chat")]
	OllamaChat,
	/// Cursor Connect.
	#[serde(rename = "cursor", alias = "cursor-agent")]
	Cursor,
	/// Devin Connect.
	#[serde(rename = "devin", alias = "devin-agent")]
	Devin,
	/// GitLab Duo workflow.
	#[serde(rename = "gitlab-duo-workflow", alias = "gitlab-duo-agent")]
	GitlabDuoWorkflow,
	/// OMP federation.
	#[serde(rename = "omp")]
	Omp,
	/// In-process inference.
	#[serde(rename = "embedded", alias = "apple-intelligence-api")]
	Embedded,
}

/// Typed source price components in decimal US dollars.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceCost {
	/// Input price per million tokens.
	#[serde(default = "zero_number")]
	pub input:        Number,
	/// Output price per million tokens.
	#[serde(default = "zero_number")]
	pub output:       Number,
	/// Cache-read price per million tokens.
	#[serde(default = "zero_number")]
	pub cache_read:   Number,
	/// Cache-write price per million tokens.
	#[serde(default = "zero_number")]
	pub cache_write:  Number,
	/// Long-context replacement schedule.
	#[serde(default)]
	pub long_context: Option<SourceLongContextCost>,
}

/// Typed long-context source price schedule.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceLongContextCost {
	/// Exclusive prompt-token threshold.
	pub input_threshold: u64,
	/// Input price.
	#[serde(default = "zero_number")]
	pub input:           Number,
	/// Output price.
	#[serde(default = "zero_number")]
	pub output:          Number,
	/// Cache-read price.
	#[serde(default = "zero_number")]
	pub cache_read:      Number,
	/// Cache-write price.
	#[serde(default = "zero_number")]
	pub cache_write:     Number,
}

/// Closed typed record parsed from one oracle model row.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceModelRecord {
	/// Optional denormalized identity.
	#[serde(default)]
	pub id: Option<Str>,
	/// Display name.
	#[serde(default)]
	pub name: Option<Str>,
	/// Optional denormalized provider.
	#[serde(default)]
	pub provider: Option<Str>,
	/// Optional per-model transport override.
	#[serde(default)]
	pub api: Option<SourceTransport>,
	/// Optional per-model endpoint override.
	#[serde(default)]
	pub base_url: Option<Str>,
	/// Declared reasoning support.
	#[serde(default)]
	pub reasoning: bool,
	/// Declared input modalities.
	#[serde(default)]
	pub input: Vec<SourceModality>,
	/// Declared output modalities.
	#[serde(default)]
	pub output: Vec<SourceModality>,
	/// Typed pricing.
	#[serde(default)]
	pub cost: SourceCost,
	/// Context window.
	#[serde(default)]
	pub context_window: Option<u64>,
	/// Maximum output tokens.
	#[serde(default)]
	pub max_tokens: Option<u64>,
	/// Typed native reasoning properties.
	#[serde(default)]
	pub thinking: Option<SourceThinking>,
	/// Fixed embedding dimension.
	#[serde(default)]
	pub embedding_dimensions: Option<u32>,
	/// Deprecation declaration.
	#[serde(default)]
	pub deprecated: bool,
	/// Explicit tool support evidence.
	#[serde(default)]
	pub supports_tools: Option<bool>,
	/// Explicit computer-use evidence.
	#[serde(default)]
	pub supports_computer_use: Option<bool>,
	/// Authored computer-use evidence.
	#[serde(default)]
	pub supports_computer_use_config: Option<bool>,
	/// Cursor max-mode evidence.
	#[serde(default)]
	pub cursor_max_mode: Option<bool>,
	/// Output-token field omission.
	#[serde(default)]
	pub omit_max_output_tokens: Option<bool>,
	/// Apply-patch wire spelling.
	#[serde(default)]
	pub apply_patch_tool_type: Option<Str>,
	/// Context promotion target.
	#[serde(default)]
	pub context_promotion_target: Option<Str>,
	/// Wire model override.
	#[serde(default)]
	pub request_model_id: Option<Str>,
	/// Typed remote-compaction source properties.
	#[serde(default)]
	pub remote_compaction: Option<SourceRemoteCompaction>,
	/// Exact premium multiplier source number.
	#[serde(default)]
	pub premium_multiplier: Option<Number>,
	/// Reasoning serving mode.
	#[serde(default)]
	pub reasoning_mode: Option<Str>,
	/// Responses-lite choice.
	#[serde(default)]
	pub use_responses_lite: Option<bool>,
	/// WebSocket preference.
	#[serde(default)]
	pub prefer_websockets: Option<bool>,
	/// Route priority.
	#[serde(default)]
	pub priority: Option<u32>,
	/// Static source headers.
	#[serde(default)]
	pub headers: BTreeMap<Str, Str>,
	/// Typed compatibility properties.
	#[serde(default)]
	pub compat: Option<SourceWirePolicy>,
	/// Compiler-derived canonical metadata reference.
	#[serde(skip)]
	pub inherited_from: Option<ModelKey>,
	/// Compiler evidence that dynamic-pricing sentinels were intentionally
	/// omitted.
	#[serde(skip)]
	pub omitted_dynamic_pricing: bool,
}

/// Typed source provider-side compaction routing.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceRemoteCompaction {
	/// Explicit enablement.
	#[serde(default)]
	pub enabled:              Option<bool>,
	/// Compaction transport override.
	#[serde(default)]
	pub api:                  Option<SourceTransport>,
	/// Primary endpoint.
	#[serde(default)]
	pub endpoint:             Option<Str>,
	/// V2 streaming enablement.
	#[serde(default)]
	pub v2_streaming_enabled: Option<bool>,
	/// V2 endpoint.
	#[serde(default)]
	pub v2_endpoint:          Option<Str>,
	/// Streaming endpoint.
	#[serde(default)]
	pub streaming_endpoint:   Option<Str>,
	/// Wire model override.
	#[serde(default)]
	pub model:                Option<Str>,
}
/// Typed model reasoning source, with route spellings excluded from profile
/// identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceThinking {
	/// Native control mode.
	pub mode:              ThinkingMode,
	/// Ordered advertised efforts.
	pub efforts:           SmallVec<ThinkingEffort, 6>,
	/// Default effort.
	#[serde(default)]
	pub default_level:     Option<ThinkingEffort>,
	/// Native effort spellings.
	#[serde(default)]
	pub effort_map:        BTreeMap<ThinkingEffort, Str>,
	/// Effort-specific wire routes.
	#[serde(default)]
	pub effort_routing:    BTreeMap<ThinkingEffort, Str>,
	/// Additional provider serving path.
	#[serde(default)]
	pub reasoning_mode:    Option<ReasoningMode>,
	/// Effort-specific token budgets.
	#[serde(default)]
	pub effort_budgets:    BTreeMap<ThinkingEffort, u64>,
	/// Adaptive display support.
	#[serde(default)]
	pub supports_display:  Option<bool>,
	/// Off-wire suppression.
	#[serde(default)]
	pub suppress_when_off: Option<bool>,
	/// Required effort evidence.
	#[serde(default)]
	pub requires_effort:   Option<bool>,
}

/// Authentication record in the provider oracle.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum SourceAuth {
	/// No credentials.
	#[default]
	None,
	/// Required bearer token.
	Bearer {
		/// Environment lookup order, retained only as provenance.
		#[serde(default)]
		env: Vec<Str>,
	},
	/// Optional bearer token.
	OptionalBearer {
		/// Environment lookup order.
		#[serde(default)]
		env: Vec<Str>,
	},
	/// Devin session token bound into sealed protobuf metadata.
	DevinSession {
		/// Environment lookup order, retained only as provenance.
		#[serde(default)]
		env: Vec<Str>,
	},
	/// Custom credential header.
	Header {
		/// Header name.
		name: Str,
		/// Environment lookup order.
		#[serde(default)]
		env:  Vec<Str>,
	},
	/// Credential query parameter.
	Query {
		/// Query parameter name.
		param: Str,
		/// Environment lookup order.
		#[serde(default)]
		env:   Vec<Str>,
	},
	/// AWS Signature Version 4.
	AwsSigV4,
	/// Google application-default credentials.
	GoogleAdc {
		/// API-key environment order.
		#[serde(default)]
		api_key_env:  Vec<Str>,
		/// Project environment order.
		#[serde(default)]
		project_env:  Vec<Str>,
		/// Location environment order.
		#[serde(default)]
		location_env: Vec<Str>,
	},
	/// OAuth flow.
	Oauth {
		/// Stable flow identifier.
		flow: Str,
	},
}

/// Provider registry mapping source.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceMapping {
	/// Concrete provider.
	#[default]
	Concrete,
	/// Provider alias.
	Alias {
		/// Canonical provider.
		target: Str,
		/// Reviewed rationale.
		reason: Str,
	},
	/// Provider implementation replacement.
	Replacement {
		/// Component name.
		component: Str,
		/// Reviewed rationale.
		reason:    Str,
	},
}

/// Provider discovery source.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceDiscovery {
	/// Discovery schema kind.
	pub kind:          Str,
	/// Human-readable label.
	pub label:         Str,
	/// Whether absence proves unavailability.
	#[serde(default)]
	pub authoritative: bool,
}

/// Sparse typed provider/model wire-policy source.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SourceWirePolicy {
	/// Streaming usage support.
	#[serde(alias = "supportsUsageInStreaming")]
	pub usage_in_streaming: Option<bool>,
	/// Multiple system-message support.
	pub multiple_system_messages: Option<bool>,
	/// Output-token field spelling.
	#[serde(alias = "maxTokensField")]
	pub max_tokens_field: Option<Str>,
	/// Sampling support.
	pub sampling_params: Option<bool>,
	/// Model-level sampling support.
	#[serde(alias = "supportsSamplingParams")]
	pub supports_sampling_params: Option<bool>,
	/// Penalty support.
	pub penalties: Option<bool>,
	/// Tool strictness.
	pub tool_strict_mode: Option<Str>,
	/// Named tool choice.
	pub named_tool_choice: Option<bool>,
	/// Forced tool choice.
	#[serde(alias = "supportsForcedToolChoice")]
	pub forced_tool_choice: Option<bool>,
	/// General tool-choice support.
	#[serde(alias = "supportsToolChoice")]
	pub supports_tool_choice: Option<bool>,
	/// Tool-call identifier profile.
	pub tool_call_id_profile: Option<Str>,
	/// Reasoning wire format.
	pub reasoning_wire_format: Option<Str>,
	/// Whether thinking may be interleaved with tool-use blocks.
	pub interleaved_thinking: Option<bool>,
	/// Stateful chaining.
	pub stateful_response_chaining: Option<bool>,
	/// Thinking/tool conflict policy.
	pub thinking_tool_choice_conflict: Option<Str>,
	/// Cache-control format.
	pub cache_control_format: Option<Str>,
	/// Image encoding.
	pub image_encoding_format: Option<Str>,
	/// Stop-sequence support.
	pub stop_sequences: Option<bool>,
	/// Tool-schema flavor.
	pub tool_schema_flavor: Option<Str>,
	/// Leaked-thinking healer.
	pub leaked_thinking_healer: Option<Str>,
	/// Thinking loop guard.
	pub thinking_loop_guard: Option<bool>,
	/// Stream watchdog.
	pub stream_watchdog: Option<SourceStreamWatchdog>,
	/// Model stream idle timeout.
	#[serde(alias = "streamIdleTimeoutMs")]
	pub stream_idle_timeout_ms: Option<u64>,
	/// Stream protocol.
	pub stream_protocol: Option<Str>,
	/// Audio API version.
	pub audio_api_version: Option<Str>,
	/// Developer role support.
	#[serde(alias = "supportsDeveloperRole")]
	pub supports_developer_role: Option<bool>,
	/// Mid-conversation system role support.
	#[serde(alias = "supportsMidConversationSystem")]
	pub supports_mid_conversation_system: Option<bool>,
	/// Built-in tool-name escaping.
	#[serde(alias = "escapeBuiltinToolNames")]
	pub escape_builtin_tool_names: Option<bool>,
	/// Required tool result identifier.
	#[serde(alias = "requiresToolResultId")]
	pub requires_tool_result_id: Option<bool>,
	/// Eager tool-input streaming.
	#[serde(alias = "supportsEagerToolInputStreaming")]
	pub supports_eager_tool_input_streaming: Option<bool>,
	/// Required assistant content on tool calls.
	#[serde(alias = "requiresAssistantContentForToolCalls")]
	pub requires_assistant_content_for_tool_calls: Option<bool>,
	/// Disable reasoning on tool choice.
	#[serde(alias = "disableReasoningOnToolChoice")]
	pub disable_reasoning_on_tool_choice: Option<bool>,
	/// Reasoning effort support.
	#[serde(alias = "supportsReasoningEffort")]
	pub supports_reasoning_effort: Option<bool>,
	/// Omit native reasoning effort.
	#[serde(alias = "omitReasoningEffort")]
	pub omit_reasoning_effort: Option<bool>,
	/// Reasoning effort spellings.
	#[serde(alias = "reasoningEffortMap")]
	pub reasoning_effort_map: BTreeMap<ThinkingEffort, Str>,
	/// Reasoning disable operation.
	#[serde(alias = "reasoningDisableMode")]
	pub reasoning_disable_mode: Option<Str>,
	/// Reasoning content field.
	#[serde(alias = "reasoningContentField")]
	pub reasoning_content_field: Option<Str>,
	/// Required reasoning on tool-call turns.
	#[serde(alias = "requiresReasoningContentForToolCalls")]
	pub requires_reasoning_content_for_tool_calls: Option<bool>,
	/// Required reasoning on all assistant turns.
	#[serde(alias = "requiresReasoningContentForAllAssistantTurns")]
	pub requires_reasoning_content_for_all_assistant_turns: Option<bool>,
	/// Synthetic reasoning permission.
	#[serde(alias = "allowsSyntheticReasoningContentForToolCalls")]
	pub allows_synthetic_reasoning_content_for_tool_calls: Option<bool>,
	/// Reasoning history filtering.
	#[serde(alias = "filterReasoningHistory")]
	pub filter_reasoning_history: Option<bool>,
	/// Encrypted reasoning inclusion.
	#[serde(alias = "includeEncryptedReasoning")]
	pub include_encrypted_reasoning: Option<bool>,
	/// Unsigned thinking replay.
	#[serde(alias = "replayUnsignedThinking")]
	pub replay_unsigned_thinking: Option<bool>,
	/// Required thinking enablement.
	#[serde(alias = "requiresThinkingEnabled")]
	pub requires_thinking_enabled: Option<bool>,
	/// Adaptive thinking disablement.
	#[serde(alias = "disableAdaptiveThinking")]
	pub disable_adaptive_thinking: Option<bool>,
	/// Official endpoint evidence.
	#[serde(alias = "officialEndpoint")]
	pub official_endpoint: Option<bool>,
	/// Signing endpoint evidence.
	#[serde(alias = "signingEndpoint")]
	pub signing_endpoint: Option<bool>,
	/// Additional thinking text format.
	#[serde(alias = "thinkingFormat")]
	pub thinking_format: Option<Str>,
	/// Typed fixed body override kept opaque at the model property boundary.
	#[serde(alias = "extraBody")]
	pub extra_body: Option<RawModelProperties>,
	/// Typed conditional body override kept opaque at the model property
	/// boundary.
	#[serde(alias = "whenThinking")]
	pub when_thinking: Option<RawModelProperties>,
	/// Long cache retention.
	#[serde(alias = "supportsLongCacheRetention")]
	pub supports_long_cache_retention: Option<bool>,
	/// Store support.
	#[serde(alias = "supportsStore")]
	pub supports_store: Option<bool>,
	/// Original image detail support.
	#[serde(alias = "supportsImageDetailOriginal")]
	pub supports_image_detail_original: Option<bool>,
}

/// Typed source watchdog bounds.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SourceStreamWatchdog {
	/// First-event timeout.
	pub first_event_ms: Option<u64>,
	/// Inter-event timeout.
	pub idle_ms:        Option<u64>,
}

/// Authored provider operation evidence.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFacet {
	/// Conversational generation.
	Chat,
	/// Vector embeddings.
	Embeddings,
	/// Image generation or editing.
	ImageGeneration,
	/// Video generation or editing.
	VideoGeneration,
	/// Speech synthesis.
	AudioSpeech,
	/// Audio transcription.
	AudioTranscription,
	/// Bidirectional realtime sessions.
	Realtime,
	/// Standalone web search.
	WebSearch,
	/// Token counting and conversion.
	Tokenization,
}

/// One curated provider source record.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceProviderRecord {
	/// Source transport.
	pub transport:            SourceTransport,
	/// Typed codec-construction discriminator.
	#[serde(default)]
	pub codec_profile:        CodecProfile,
	/// Explicit operation codec identifier when it differs from the source
	/// transport vocabulary.
	#[serde(default)]
	pub codec:                Option<CodecId>,
	/// Explicit runtime transport when it differs from the source transport
	/// vocabulary.
	#[serde(default)]
	pub route_transport:      Option<TransportKind>,
	/// Primary base URL.
	pub base_url:             Str,
	/// Optional API version.
	#[serde(default)]
	pub api_version:          Option<Str>,
	/// Codex transport preference.
	#[serde(default)]
	pub codex_transport:      Option<Str>,
	/// Codex Responses-lite choice.
	#[serde(default)]
	pub codex_responses_lite: bool,
	/// Additional route URLs.
	#[serde(default)]
	pub fallback_base_urls:   Vec<Str>,
	/// Authentication source.
	#[serde(default)]
	pub auth:                 SourceAuth,
	/// Declared provider facets.
	#[serde(default)]
	pub facets:               Vec<SourceFacet>,
	/// Static non-secret headers.
	#[serde(default)]
	pub headers:              BTreeMap<Str, Str>,
	/// Typed wire policy overrides.
	#[serde(default)]
	pub compat:               SourceWirePolicy,
	/// Registry mapping.
	#[serde(default)]
	pub mapping:              SourceMapping,
	/// Optional login flow.
	#[serde(default)]
	pub oauth_flow:           Option<Str>,
	/// Optional OAuth credential placement.
	#[serde(default)]
	pub oauth_auth:           Option<SourceAuth>,
	/// Optional discovery source.
	#[serde(default)]
	pub discovery:            Option<SourceDiscovery>,
	/// Facets withheld until a transport exists.
	#[serde(default)]
	pub pending_facets:       Vec<SourceFacet>,
	/// Withheld transport source name.
	#[serde(default)]
	pub pending_transport:    Option<Str>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderDocument {
	providers: BTreeMap<Str, SourceProviderRecord>,
}

/// Typed in-memory compiler source.
#[derive(Debug)]
pub struct CatalogSource {
	/// Curated provider records.
	pub providers: BTreeMap<Str, SourceProviderRecord>,
	/// Raw provider-keyed model records.
	pub models:    BTreeMap<Str, BTreeMap<Str, SourceModelRecord>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceOAuthDocument {
	provider: Vec<SourceOAuthSpec>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceOAuthSpec {
	provider:            Str,
	credential_provider: Str,
	kind:                SourceOAuthKind,
	client_id:           Str,
	authorize_url:       Str,
	token_url:           Str,
	#[serde(default)]
	scopes:              Vec<Str>,
	callback_port:       Option<u16>,
	#[serde(default)]
	extra_auth_params:   BTreeMap<Str, Str>,
	exchange:            Option<OAuthExchangeKind>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum SourceOAuthKind {
	Pkce,
	DeviceCode,
	CustomExchange,
}

#[derive(Debug, Deserialize)]
struct ReviewedCompatDocument {
	profiles: Vec<ReviewedCompatProfile>,
}

#[derive(Debug, Deserialize)]
struct ReviewedCompatProfile {
	models: Vec<Str>,
	shape:  ReviewedCompatShape,
}

#[derive(Debug, Deserialize)]
struct ReviewedCompatShape {
	#[serde(default, rename = "wire/requires_reasoning_content_for_all_assistant_turns")]
	requires_reasoning_content_for_all_assistant_turns: bool,
}

fn reviewed_reasoning_content_models() -> Result<BTreeSet<Str>, CompileError> {
	let document: ReviewedCompatDocument = serde_json::from_str(include_str!(
		"../../../fixtures/llm-oracle/catalog-policy/compat-profiles.json"
	))?;
	Ok(document
		.profiles
		.into_iter()
		.filter(|profile| {
			profile
				.shape
				.requires_reasoning_content_for_all_assistant_turns
		})
		.flat_map(|profile| profile.models)
		.collect())
}

/// Stable alias emitted by normalization.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CatalogAlias {
	/// Alias selector.
	pub alias:      Str,
	/// Canonical model key.
	pub target:     ModelKey,
	/// Review rationale.
	pub rationale:  Str,
	/// Evidence provenance.
	pub provenance: Str,
}

/// Reviewable compiler census encoded with normalized output.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompilerCensus {
	/// Raw model rows.
	pub raw_models:        usize,
	/// Logical model rows.
	pub logical_models:    usize,
	/// Curated providers.
	pub providers:         usize,
	/// Provider keys in raw model data.
	pub raw_provider_keys: usize,
	/// Distinct route URLs.
	pub urls:              usize,
	/// Full source transport vocabulary.
	pub transports:        usize,
	/// Active source transports.
	pub active_transports: usize,
}

/// Deterministically compiled catalog and all structurally interned profiles.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompiledCatalog {
	/// Normalized schema version.
	pub schema_version:    u32,
	/// Content-derived catalog revision.
	pub revision:          CatalogRevision,
	/// Verified compilation census.
	pub census:            CompilerCensus,
	/// Providers sorted by identifier.
	pub providers:         Box<[ProviderDef]>,
	/// Structurally interned authentication specifications.
	pub auth_specs:        Box<[AuthSpec]>,
	/// Structurally interned public OAuth flow specifications.
	pub oauth_specs:       Box<[OAuthSpec]>,
	/// Structurally interned safe header profiles.
	pub header_profiles:   Box<[HeaderProfile]>,
	/// Structurally interned discovery specifications.
	pub discovery_specs:   Box<[DiscoverySpec]>,
	/// Routes sorted by identifier.
	pub routes:            Box<[RouteDef]>,
	/// Logical models sorted by key.
	pub models:            Box<[ModelSpec]>,
	/// Structurally interned wire policies.
	pub wire_policies:     Box<[WirePolicy]>,
	/// Structurally interned thinking policies.
	pub thinking_policies: Box<[ThinkingPolicy]>,
	/// Aliases sorted by alias and target.
	pub aliases:           Box<[CatalogAlias]>,
}

impl CompiledCatalog {
	/// Serializes the review schema as deterministic pretty JSON with one
	/// trailing newline.
	pub fn normalized_json(&self) -> Result<Vec<u8>, CompileError> {
		let mut bytes = serde_json::to_vec_pretty(self)?;
		bytes.push(b'\n');
		Ok(bytes)
	}
}

/// Offline compiler failure.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
	/// Provider TOML did not match the closed schema.
	#[error("provider oracle is invalid: {0}")]
	Provider(#[from] toml::de::Error),
	/// Model JSON did not match the closed schema.
	#[error("model oracle is invalid: {0}")]
	Json(#[from] serde_json::Error),
	/// Compressed model source could not be decoded.
	#[error("model oracle compression is invalid: {0}")]
	Compression(#[from] std::io::Error),
	/// Source data violated a catalog invariant.
	#[error("catalog invariant failed: {0}")]
	Invariant(Str),
}

/// Parses the two checked-in oracle source formats into typed records.
pub fn parse_oracle(
	providers_toml: &str,
	models_json_zstd: &[u8],
) -> Result<CatalogSource, CompileError> {
	let mut providers: ProviderDocument = toml::from_str(providers_toml)?;
	for (provider, profile) in [
		("google-gemini-cli", CodecProfile::GoogleCcaGeminiCli),
		("google-antigravity", CodecProfile::GoogleCcaAntigravity),
		("apple-intelligence", CodecProfile::AppleFm),
	] {
		if let Some(record) = providers.providers.get_mut(provider) {
			record.codec_profile = profile;
		}
	}
	let json = zstd::stream::decode_all(models_json_zstd)?;
	let models = serde_json::from_slice(&json)?;
	Ok(CatalogSource { providers: providers.providers, models })
}

/// Compiles the checked-in provider, model, and OAuth oracles without network
/// access.
pub fn compile_oracle(
	providers_toml: &str,
	models_json_zstd: &[u8],
	oauth_toml: &str,
) -> Result<CompiledCatalog, CompileError> {
	compile_with_oauth(parse_oracle(providers_toml, models_json_zstd)?, oauth_toml)
}

/// Compiles typed source records with the bundled public OAuth table.
pub fn compile(source: CatalogSource) -> Result<CompiledCatalog, CompileError> {
	compile_with_oauth(source, include_str!("../../../fixtures/llm-oracle/catalog/oauth.toml"))
}

fn compile_with_oauth(
	source: CatalogSource,
	oauth_toml: &str,
) -> Result<CompiledCatalog, CompileError> {
	let CatalogSource { providers: provider_sources, models: mut model_sources } = source;
	let provider_facets = provider_sources
		.iter()
		.map(|(provider, source)| (provider.clone(), source.facets.clone()))
		.collect::<BTreeMap<_, _>>();
	inherit_source_references(&mut model_sources)?;
	let raw_models = model_sources.values().map(BTreeMap::len).sum();
	let active_transports = provider_sources
		.values()
		.map(|provider| provider.transport)
		.collect::<BTreeSet<_>>()
		.len();
	let raw_provider_keys = model_sources.len();
	let urls = source_url_census(&provider_sources, &model_sources);
	let (oauth_specs, oauth_ids) = compile_oauth_specs(oauth_toml)?;
	let (
		mut providers,
		auth_specs,
		mut header_profiles,
		discovery_specs,
		mut routes,
		provider_routes,
		mut wire_policy_table,
		provider_policies,
	) = compile_providers(provider_sources, &oauth_ids, &oauth_specs)?;
	let model_routes = compile_model_routes(
		&model_sources,
		&mut providers,
		&mut routes,
		&mut header_profiles,
		&provider_routes,
	)?;
	let reviewed_reasoning_content = reviewed_reasoning_content_models()?;
	let mut thinking_policy_table = BTreeMap::new();
	let (models, aliases) = compile_models(
		model_sources,
		&model_routes,
		&provider_policies,
		&mut wire_policy_table,
		&mut thinking_policy_table,
		&provider_facets,
		&reviewed_reasoning_content,
	)?;
	let census = CompilerCensus {
		raw_models,
		logical_models: models.len(),
		providers: providers.len(),
		raw_provider_keys,
		urls,
		transports: ORACLE_TRANSPORTS,
		active_transports,
	};
	if raw_models == ORACLE_RAW_MODELS {
		validate_oracle_census(census)?;
	}
	let wire_policies: Vec<WirePolicy> = wire_policy_table.into_values().collect();
	let thinking_policies: Vec<ThinkingPolicy> = thinking_policy_table.into_values().collect();
	let revision = revision_for(&providers, &routes, &models)?;
	Ok(CompiledCatalog {
		schema_version: COMPILED_SCHEMA_VERSION,
		revision,
		census,
		oauth_specs: oauth_specs.into_boxed_slice(),
		providers: providers.into_boxed_slice(),
		auth_specs: auth_specs.into_boxed_slice(),
		header_profiles: header_profiles.into_boxed_slice(),
		discovery_specs: discovery_specs.into_boxed_slice(),
		routes: routes.into_boxed_slice(),
		models: models.into_boxed_slice(),
		wire_policies: wire_policies.into_boxed_slice(),
		thinking_policies: thinking_policies.into_boxed_slice(),
		aliases: aliases.into_boxed_slice(),
	})
}

#[derive(Clone, Copy)]
struct ExactSourceReference {
	provider:           &'static str,
	model:              &'static str,
	reference_provider: &'static str,
	reference_model:    &'static str,
	rationale:          &'static str,
	provenance:         &'static str,
	expires_at_ms:      Option<u64>,
}

const fn reviewed_source_reference(
	provider: &'static str,
	model: &'static str,
	reference_provider: &'static str,
	reference_model: &'static str,
	rationale: &'static str,
) -> ExactSourceReference {
	ExactSourceReference {
		provider,
		model,
		reference_provider,
		reference_model,
		rationale,
		provenance: "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	}
}

const EXACT_SOURCE_REFERENCES: &[ExactSourceReference] = &[
	ExactSourceReference {
		provider:           "kilo",
		model:              "deepseek/deepseek-v4-flash:free",
		reference_provider: "kilo",
		reference_model:    "deepseek/deepseek-v4-flash:discounted",
		rationale:          "The free selector inherits the reviewed discounted wire sibling's \
		                     effective pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "kilo",
		model:              "minimax/minimax-m2",
		reference_provider: "minimax",
		reference_model:    "MiniMax-M2.5",
		rationale:          "The reseller card inherits canonical MiniMax cache-write pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "kilo",
		model:              "minimax/minimax-m2.1",
		reference_provider: "minimax",
		reference_model:    "MiniMax-M2.1",
		rationale:          "The reseller card inherits canonical MiniMax cache-write pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "kilo",
		model:              "minimax/minimax-m2.5",
		reference_provider: "minimax",
		reference_model:    "MiniMax-M2.5",
		rationale:          "The reseller card inherits canonical MiniMax cache-write pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "kilo",
		model:              "minimax/minimax-m2.5:free",
		reference_provider: "openrouter",
		reference_model:    "minimax/minimax-m2.5:free",
		rationale:          "The free selector inherits reviewed free-tier prices before canonical \
		                     cache-write pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "kilo",
		model:              "minimax/minimax-m2.7",
		reference_provider: "minimax",
		reference_model:    "MiniMax-M2.7",
		rationale:          "The reseller card inherits canonical MiniMax cache-write pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "openrouter",
		model:              "minimax/minimax-m2.5:free",
		reference_provider: "openrouter",
		reference_model:    "minimax/minimax-m2.5",
		rationale:          "The free-tier row inherits reviewed free pricing before canonical \
		                     cache-write pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "openrouter",
		model:              "minimax/minimax-m2.5",
		reference_provider: "minimax",
		reference_model:    "MiniMax-M2.5",
		rationale:          "The gateway price wins component-wise while canonical cache-write \
		                     pricing fills its zero.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "aiand",
		model:              "qwen/qwen3.6-27b",
		reference_provider: "coreweave",
		reference_model:    "Qwen/Qwen3.6-27B",
		rationale:          "The reseller's explicit prices win while the reviewed deployment fills \
		                     cache-read pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "Qwen/Qwen3-235B-A22B-Instruct-2507",
		reference_provider: "coreweave",
		reference_model:    "Qwen/Qwen3-235B-A22B-Instruct-2507",
		rationale:          "Every exact deployment spelling preserves its explicit prices while \
		                     the reviewed canonical deployment fills cache-read pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "Qwen/Qwen3-235B-A22B-Thinking-2507",
		reference_provider: "coreweave",
		reference_model:    "Qwen/Qwen3-235B-A22B-Thinking-2507",
		rationale:          "Every exact deployment spelling preserves its explicit prices while \
		                     the reviewed canonical deployment fills cache-read pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "Qwen/Qwen3-Coder-480B-A35B-Instruct",
		reference_provider: "coreweave",
		reference_model:    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
		rationale:          "Every exact deployment spelling preserves its explicit prices while \
		                     the reviewed canonical deployment fills cache-read pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "Qwen/Qwen3-Coder-Next",
		reference_provider: "openrouter",
		reference_model:    "qwen/qwen3-coder-next",
		rationale:          "Every exact deployment spelling preserves its explicit prices while \
		                     the reviewed canonical route fills cache-read pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "Qwen/Qwen3.5-27B",
		reference_provider: "coreweave",
		reference_model:    "Qwen/Qwen3.5-27B",
		rationale:          "Every exact deployment spelling preserves its explicit prices while \
		                     the reviewed canonical deployment fills cache-read pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "Qwen/Qwen3.5-35B-A3B",
		reference_provider: "coreweave",
		reference_model:    "Qwen/Qwen3.5-35B-A3B",
		rationale:          "Every exact deployment spelling preserves its explicit prices while \
		                     the reviewed canonical deployment fills cache-read pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "Qwen/Qwen3.6-27B",
		reference_provider: "coreweave",
		reference_model:    "Qwen/Qwen3.6-27B",
		rationale:          "Every exact deployment spelling preserves its explicit prices while \
		                     the reviewed canonical deployment fills cache-read pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "Qwen/Qwen3.6-35B-A3B",
		reference_provider: "coreweave",
		reference_model:    "Qwen/Qwen3.6-35B-A3B",
		rationale:          "Every exact deployment spelling preserves its explicit prices while \
		                     the reviewed canonical deployment fills cache-read pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "Qwen/Qwen3.5-397B-A17B",
		reference_provider: "together",
		reference_model:    "Qwen/Qwen3.5-397B-A17B",
		rationale:          "Every exact deployment spelling preserves its explicit prices while \
		                     the reviewed canonical deployment fills cache-read pricing.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "qwen/qwen-max",
		reference_provider: "openrouter",
		reference_model:    "qwen/qwen-max",
		rationale:          "The exact OpenRouter card supplies reviewed Qwen Max limits and cache \
		                     pricing without replacing explicit reseller prices.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "qwen/qwen-turbo",
		reference_provider: "openrouter",
		reference_model:    "qwen/qwen-turbo",
		rationale:          "The exact OpenRouter card supplies reviewed Qwen Turbo limits and \
		                     cache pricing without replacing explicit reseller prices.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "qwen/qwen-vl-max",
		reference_provider: "openrouter",
		reference_model:    "qwen/qwen-vl-max",
		rationale:          "The exact OpenRouter card supplies reviewed Qwen VL Max limits without \
		                     replacing explicit reseller prices.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "qwen/qwq-32b",
		reference_provider: "openrouter",
		reference_model:    "qwen/qwq-32b",
		rationale:          "The exact OpenRouter card supplies reviewed QwQ limits without \
		                     replacing explicit reseller prices.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "qwen/qwen3-30b-a3b-instruct-2507",
		reference_provider: "coreweave",
		reference_model:    "Qwen/Qwen3-30B-A3B-Instruct-2507",
		rationale:          "The exact canonical deployment fills cache-read pricing without \
		                     replacing explicit reseller prices or limits.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "qwen/qwen3-30b-a3b-thinking-2507",
		reference_provider: "openrouter",
		reference_model:    "qwen/qwen3-30b-a3b-thinking-2507",
		rationale:          "The exact reviewed route fills cache-read pricing without replacing \
		                     explicit reseller prices or limits.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "qwen/qwen3-8b",
		reference_provider: "openrouter",
		reference_model:    "qwen/qwen3-8b",
		rationale:          "The exact reviewed route fills cache-read pricing without replacing \
		                     explicit reseller prices or limits.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "qwen/qwen3-coder-30b-a3b-instruct",
		reference_provider: "nanogpt",
		reference_model:    "qwen3-coder-30b-a3b-instruct",
		rationale:          "The exact reviewed deployment fills cache-read pricing without \
		                     replacing explicit reseller prices or limits.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "qwen/qwen3-vl-235b-a22b-instruct",
		reference_provider: "openrouter",
		reference_model:    "qwen/qwen3-vl-235b-a22b-instruct",
		rationale:          "The exact reviewed route fills cache-read pricing without replacing \
		                     explicit reseller prices or limits.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "qwen/qwen3-vl-235b-a22b-thinking",
		reference_provider: "openrouter",
		reference_model:    "qwen/qwen3-vl-235b-a22b-instruct",
		rationale:          "The reviewed sibling supplies the shared cache-read price without \
		                     replacing explicit thinking-route prices or limits.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "*",
		model:              "qwen/qwen3-coder",
		reference_provider: "openrouter",
		reference_model:    "qwen/qwen3-coder",
		rationale:          "The exact reviewed route fills cache-read pricing without replacing \
		                     explicit reseller prices or limits.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "kilo",
		model:              "qwen/qwen3.6-plus:free",
		reference_provider: "opencode-go",
		reference_model:    "qwen3.6-plus",
		rationale:          "The free selector retains its explicit prices while the reviewed \
		                     canonical card fills both cache components.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "kilo",
		model:              "qwen/qwen3.7-plus:free",
		reference_provider: "kilo",
		reference_model:    "qwen/qwen3.7-plus",
		rationale:          "The free selector retains its explicit prices while the reviewed \
		                     sibling fills both cache components.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "kilo",
		model:              "stepfun/step-3.5-flash:free",
		reference_provider: "openrouter",
		reference_model:    "stepfun/step-3.5-flash",
		rationale:          "The free selector inherits reviewed public prices while retaining its \
		                     explicit limits.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "kilo",
		model:              "stepfun/step-3.7-flash:free",
		reference_provider: "kilo",
		reference_model:    "stepfun/step-3.7-flash",
		rationale:          "The free selector inherits reviewed sibling prices while retaining its \
		                     explicit limits.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "kilo",
		model:              "tencent/hy3-preview:free",
		reference_provider: "kilo",
		reference_model:    "tencent/hy3-preview",
		rationale:          "The free selector inherits the reviewed sibling's complete price while \
		                     retaining its explicit limits.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "kilo",
		model:              "tencent/hy3:free",
		reference_provider: "nanogpt",
		reference_model:    "tencent/hy3",
		rationale:          "The free selector inherits the reviewed canonical deployment's \
		                     complete price while retaining its explicit limits.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "kilo",
		model:              "x-ai/grok-code-fast-1:optimized:free",
		reference_provider: "xai",
		reference_model:    "grok-code-fast-1",
		rationale:          "The stacked optimized/free selector inherits the reviewed native \
		                     card's complete price while retaining its explicit limits.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	ExactSourceReference {
		provider:           "nanogpt",
		model:              "Qwen/Qwen3-Next-80B-A3B-Instruct",
		reference_provider: "huggingface",
		reference_model:    "Qwen/Qwen3-Next-80B-A3B-Instruct",
		rationale:          "The reviewed Hugging Face card fills the NanoGPT selector's absent \
		                     prices while preserving its explicit limits.",
		provenance:         "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms:      None,
	},
	reviewed_source_reference(
		"nanogpt",
		"anthropic/claude-opus-4.6:thinking",
		"openrouter",
		"anthropic/claude-opus-4.6",
		"The thinking selector inherits the reviewed base card's missing cache-write component.",
	),
	reviewed_source_reference(
		"nanogpt",
		"anthropic/claude-opus-4.6:thinking:low",
		"openrouter",
		"anthropic/claude-opus-4.6",
		"The low thinking selector inherits the reviewed base card's missing cache-write component.",
	),
	reviewed_source_reference(
		"nanogpt",
		"anthropic/claude-opus-4.6:thinking:medium",
		"openrouter",
		"anthropic/claude-opus-4.6",
		"The medium thinking selector inherits the reviewed base card's missing cache-write \
		 component.",
	),
	reviewed_source_reference(
		"nanogpt",
		"anthropic/claude-opus-4.7:thinking",
		"openrouter",
		"anthropic/claude-opus-4.7",
		"The thinking selector inherits the reviewed base card's missing cache-write component.",
	),
	reviewed_source_reference(
		"nanogpt",
		"anthropic/claude-opus-4.8:thinking",
		"openrouter",
		"anthropic/claude-opus-4.8",
		"The thinking selector inherits the reviewed base card's missing cache-write component.",
	),
	reviewed_source_reference(
		"nanogpt",
		"anthropic/claude-sonnet-4.6:thinking",
		"openrouter",
		"anthropic/claude-sonnet-4.6",
		"The thinking selector inherits the reviewed base card's missing cache-write component.",
	),
	reviewed_source_reference(
		"*",
		"minimax/minimax-m2.1",
		"minimax",
		"MiniMax-M2.1",
		"Every exact deployment preserves explicit prices while the reviewed native card fills \
		 missing cache-write pricing.",
	),
	reviewed_source_reference(
		"*",
		"minimax/minimax-m2.5",
		"minimax",
		"MiniMax-M2.5",
		"Every exact deployment preserves explicit prices while the reviewed native card fills \
		 missing cache-write pricing.",
	),
	reviewed_source_reference(
		"*",
		"minimax/minimax-m2.7",
		"minimax",
		"MiniMax-M2.7",
		"Every exact deployment preserves explicit prices while the reviewed native card fills \
		 missing cache-write pricing.",
	),
	reviewed_source_reference(
		"*",
		"minimaxai/minimax-m2.1",
		"minimax",
		"MiniMax-M2.1",
		"Every exact deployment preserves explicit prices while the reviewed native card fills \
		 missing cache pricing.",
	),
	reviewed_source_reference(
		"*",
		"minimaxai/minimax-m2.5",
		"minimax",
		"MiniMax-M2.5",
		"Every exact deployment preserves explicit prices while the reviewed native card fills \
		 missing cache pricing.",
	),
	reviewed_source_reference(
		"*",
		"minimaxai/minimax-m2.7",
		"minimax",
		"MiniMax-M2.7",
		"Every exact deployment preserves explicit prices while the reviewed native card fills \
		 missing cache pricing.",
	),
	reviewed_source_reference(
		"nanogpt",
		"claude-opus-4-5-20251101:thinking",
		"anthropic",
		"claude-opus-4-5-20251101",
		"The thinking selector inherits the reviewed native card's missing cache-write component.",
	),
	reviewed_source_reference(
		"nanogpt",
		"moonshotai/kimi-k2-thinking-original",
		"nanogpt",
		"moonshotai/kimi-k2-thinking",
		"The original selector inherits the reviewed canonical deployment's complete price.",
	),
	reviewed_source_reference(
		"nanogpt",
		"moonshotai/kimi-k2-thinking-turbo-original",
		"moonshot",
		"kimi-k2-thinking-turbo",
		"The original turbo selector inherits the reviewed native deployment's complete price.",
	),
	reviewed_source_reference(
		"nanogpt",
		"openai/o1-pro",
		"openai",
		"o1-pro",
		"The namespaced selector inherits the reviewed native deployment's complete price.",
	),
	reviewed_source_reference(
		"*",
		"qwen/qwen3-coder-plus",
		"openrouter",
		"qwen/qwen3-coder-plus",
		"Every exact deployment preserves explicit prices while the reviewed route fills missing \
		 cache-write pricing.",
	),
	reviewed_source_reference(
		"*",
		"qwen/qwen3-max",
		"openrouter",
		"qwen/qwen3-max",
		"Every exact deployment preserves explicit prices while the reviewed route fills missing \
		 cache pricing.",
	),
	reviewed_source_reference(
		"*",
		"qwen/qwen3-next-80b-a3b-instruct",
		"huggingface",
		"Qwen/Qwen3-Next-80B-A3B-Instruct",
		"Every exact deployment preserves explicit limits while the reviewed deployment fills \
		 absent prices.",
	),
	reviewed_source_reference(
		"*",
		"qwen/qwen3-next-80b-a3b-thinking",
		"huggingface",
		"Qwen/Qwen3-Next-80B-A3B-Thinking",
		"Every exact deployment preserves explicit limits while the reviewed deployment fills \
		 absent prices.",
	),
	reviewed_source_reference(
		"nanogpt",
		"qwen3-vl-235b-a22b-instruct-original",
		"openrouter",
		"qwen/qwen3-vl-235b-a22b-instruct",
		"The original selector inherits the reviewed route's complete prices and limits.",
	),
	reviewed_source_reference(
		"nanogpt",
		"x-ai/grok-4.20-multi-agent",
		"xai",
		"grok-4.20-multi-agent-beta-latest",
		"The reviewed native card fills the deployment's missing cache-read price without replacing \
		 explicit input or output prices.",
	),
	reviewed_source_reference(
		"nanogpt",
		"x-ai/grok-4.20-multi-agent-beta",
		"vercel-ai-gateway",
		"xai/grok-4.20-multi-agent-beta",
		"The beta selector inherits the reviewed gateway deployment's complete price.",
	),
	reviewed_source_reference(
		"nanogpt",
		"xiaomi/mimo-v2-flash-original",
		"nanogpt",
		"xiaomi/mimo-v2-flash",
		"The original selector inherits the reviewed canonical deployment's complete price.",
	),
	reviewed_source_reference(
		"nanogpt",
		"zai-org/glm-4.5",
		"zai",
		"glm-4.5",
		"The reviewed native card fills the deployment's missing cache-read price.",
	),
	reviewed_source_reference(
		"nanogpt",
		"zai-org/glm-4.6-original",
		"novita",
		"zai-org/glm-4.6",
		"The original selector inherits the reviewed version-exact deployment's complete price.",
	),
	reviewed_source_reference(
		"nanogpt",
		"zai-org/glm-4.6v-original",
		"novita",
		"zai-org/glm-4.6v",
		"The original vision selector inherits the reviewed deployment's complete price.",
	),
	reviewed_source_reference(
		"nanogpt",
		"zai-org/glm-4.6v-flash-original",
		"zenmux",
		"z-ai/glm-4.6v-flash",
		"The original vision-flash selector inherits the reviewed deployment's complete price.",
	),
	reviewed_source_reference(
		"novita",
		"deepseek/deepseek-v3/community",
		"vercel-ai-gateway",
		"deepseek/deepseek-v3",
		"The community deployment preserves explicit input and output prices while the reviewed \
		 route fills cache-read pricing.",
	),
	reviewed_source_reference(
		"nvidia",
		"meta/llama-4-scout-17b-16e-instruct",
		"zenmux",
		"meta/llama-4-scout-17b-16e-instruct",
		"The deployment inherits reviewed input and output prices before canonical cache components \
		 are filled.",
	),
	reviewed_source_reference(
		"nvidia",
		"qwen/qwen3.5-122b-a10b",
		"kilo",
		"qwen/qwen3.5-122b-a10b",
		"The deployment inherits reviewed input and output prices before the exact cache-read \
		 component is filled.",
	),
	reviewed_source_reference(
		"nvidia",
		"thinkingmachines/inkling",
		"huggingface",
		"thinkingmachines/Inkling",
		"The deployment inherits reviewed input and output prices before the deployment-specific \
		 cache-read component is filled.",
	),
	reviewed_source_reference(
		"openrouter",
		"arcee-ai/trinity-large-thinking:free",
		"openrouter",
		"arcee-ai/trinity-large-thinking",
		"The reviewed free selector inherits its canonical paid sibling's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"baidu/cobuddy:free",
		"novita",
		"baidu/cobuddy",
		"The reviewed free selector inherits its canonical paid deployment's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"deepseek/deepseek-v4-flash:free",
		"deepseek",
		"deepseek-v4-flash",
		"The reviewed free selector inherits the native deployment's complete price.",
	),
	reviewed_source_reference(
		"zenmux",
		"deepseek/deepseek-v4-flash-free",
		"deepseek",
		"deepseek-v4-flash",
		"The reviewed free selector inherits the native deployment's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"inclusionai/ling-2.6-1t:free",
		"nanogpt",
		"inclusionai/ling-2.6-1t",
		"The reviewed free selector inherits the canonical deployment's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"inclusionai/ling-2.6-flash:free",
		"nanogpt",
		"inclusionai/ling-2.6-flash",
		"The reviewed free selector inherits the canonical deployment's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"inclusionai/ring-2.6-1t:free",
		"nanogpt",
		"inclusionai/ring-2.6-1t",
		"The reviewed free selector inherits the canonical deployment's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"moonshotai/kimi-k2",
		"opencode-zen",
		"kimi-k2",
		"The deployment preserves explicit input and output prices while the reviewed route fills \
		 cache-read pricing.",
	),
	reviewed_source_reference(
		"openrouter",
		"moonshotai/kimi-k2-0905:exacto",
		"openrouter",
		"moonshotai/kimi-k2-0905",
		"The reviewed exact selector preserves explicit prices while inheriting canonical \
		 cache-read pricing.",
	),
	reviewed_source_reference(
		"openrouter",
		"moonshotai/kimi-k2.6:free",
		"coreweave",
		"moonshotai/Kimi-K2.6",
		"The reviewed free selector inherits the canonical deployment's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"nex-agi/nex-n2-pro:free",
		"openrouter",
		"nex-agi/nex-n2-pro",
		"The reviewed free selector inherits its canonical paid sibling's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"nvidia/nemotron-3-super-120b-a12b:free",
		"openrouter",
		"nvidia/nemotron-3-super-120b-a12b",
		"The reviewed free selector inherits its canonical paid sibling's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"nvidia/nemotron-3-ultra-550b-a55b:free",
		"nanogpt",
		"nvidia/nemotron-3-ultra-550b-a55b",
		"The reviewed free selector inherits the canonical deployment's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"openai/gpt-5.4-pro",
		"opencode",
		"gpt-5.4-pro",
		"The deployment preserves explicit input and output prices while the reviewed route fills \
		 cache-read pricing.",
	),
	reviewed_source_reference(
		"openrouter",
		"openai/gpt-5.5-pro",
		"opencode-zen",
		"gpt-5.5-pro",
		"The deployment preserves explicit input and output prices while the reviewed route fills \
		 cache-read pricing.",
	),
	reviewed_source_reference(
		"vercel-ai-gateway",
		"openai/gpt-5.4-pro",
		"opencode",
		"gpt-5.4-pro",
		"The deployment preserves explicit input and output prices while the reviewed route fills \
		 cache-read pricing.",
	),
	reviewed_source_reference(
		"vercel-ai-gateway",
		"openai/gpt-5.5-pro",
		"opencode-zen",
		"gpt-5.5-pro",
		"The deployment preserves explicit input and output prices while the reviewed route fills \
		 cache-read pricing.",
	),
	reviewed_source_reference(
		"zenmux",
		"openai/gpt-5.4-pro",
		"opencode",
		"gpt-5.4-pro",
		"The deployment preserves explicit input and output prices while the reviewed route fills \
		 cache-read pricing.",
	),
	reviewed_source_reference(
		"zenmux",
		"openai/gpt-5.5-pro",
		"opencode-zen",
		"gpt-5.5-pro",
		"The deployment preserves explicit input and output prices while the reviewed route fills \
		 cache-read pricing.",
	),
	reviewed_source_reference(
		"openrouter",
		"openai/gpt-oss-120b:exacto",
		"coreweave",
		"openai/gpt-oss-120b",
		"The exact selector preserves explicit input and output prices while the reviewed \
		 deployment fills cache-read pricing.",
	),
	reviewed_source_reference(
		"openrouter",
		"poolside/laguna-m.1:free",
		"openrouter",
		"poolside/laguna-m.1",
		"The reviewed free selector inherits its canonical paid sibling's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"poolside/laguna-xs-2.1:free",
		"kilo",
		"poolside/laguna-xs-2.1",
		"The reviewed free selector inherits the canonical deployment's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"poolside/laguna-xs.2:free",
		"openrouter",
		"poolside/laguna-xs.2",
		"The reviewed free selector inherits its canonical paid sibling's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"qwen/qwen3.6-plus:free",
		"opencode-zen",
		"qwen3.6-plus",
		"The reviewed free selector inherits the native deployment's complete price.",
	),
	reviewed_source_reference(
		"openrouter",
		"stepfun/step-3.5-flash:free",
		"openrouter",
		"stepfun/step-3.5-flash",
		"The reviewed free selector inherits its canonical paid sibling's complete price.",
	),
	reviewed_source_reference(
		"zenmux",
		"xiaomi/mimo-v2-flash-free",
		"xiaomi",
		"mimo-v2-flash",
		"The reviewed free selector inherits the native deployment's complete price.",
	),
	reviewed_source_reference(
		"nanogpt",
		"baseten/Kimi-K2-Instruct-FP4",
		"nanogpt",
		"moonshotai/kimi-k2-instruct",
		"The reviewed native Kimi deployment supplies the FP4 selector's absent prices.",
	),
];

#[derive(Clone, Copy)]
struct SourceInheritanceOverride {
	provider:      Option<&'static str>,
	model:         &'static str,
	max_hops:      usize,
	prefer_suffix: bool,
	rationale:     &'static str,
	provenance:    &'static str,
	expires_at_ms: Option<u64>,
}

const fn reviewed_no_inheritance(
	provider: &'static str,
	model: &'static str,
	rationale: &'static str,
) -> SourceInheritanceOverride {
	SourceInheritanceOverride {
		provider: Some(provider),
		model,
		max_hops: 0,
		prefer_suffix: false,
		rationale,
		provenance: "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	}
}

const SOURCE_INHERITANCE_OVERRIDES: &[SourceInheritanceOverride] = &[
	SourceInheritanceOverride {
		provider:      Some("azure"),
		model:         "gpt-chat-latest",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The declared Azure price is complete; the fuzzy Nanogpt suffix match is not \
		                upstream evidence.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      None,
		model:         "MiniMaxAI/MiniMax-M2.1",
		max_hops:      1,
		prefer_suffix: true,
		rationale:     "The namespace is a reseller decoration; the bare MiniMax card is \
		                authoritative.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      None,
		model:         "MiniMaxAI/MiniMax-M2.5",
		max_hops:      1,
		prefer_suffix: true,
		rationale:     "The namespace is a reseller decoration; the bare MiniMax card is \
		                authoritative.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      None,
		model:         "MiniMaxAI/MiniMax-M2.7",
		max_hops:      1,
		prefer_suffix: true,
		rationale:     "The namespace is a reseller decoration; the bare MiniMax card is \
		                authoritative.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      None,
		model:         "MiniMaxAI/MiniMax-M3",
		max_hops:      1,
		prefer_suffix: true,
		rationale:     "The namespace is a reseller decoration; the bare MiniMax card is \
		                authoritative.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      None,
		model:         "Qwen/Qwen3.5-122B-A10B",
		max_hops:      1,
		prefer_suffix: true,
		rationale:     "The Qwen namespace is a reseller decoration; the reviewed bare card owns \
		                cache pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("huggingface"),
		model:         "deepseek-ai/DeepSeek-V3",
		max_hops:      1,
		prefer_suffix: true,
		rationale:     "The provider namespace is decorative; the reviewed suffix index carries \
		                cache pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("aimlapi"),
		model:         "nemotron-3-nano-omni-30b-a3b-reasoning:free",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The free variant explicitly retains unknown limits and zero pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("kilo"),
		model:         "google/gemini-3-pro-preview",
		max_hops:      1,
		prefer_suffix: true,
		rationale:     "The Google namespace is decorative; the reviewed native card owns cache \
		                pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("kilo"),
		model:         "minimax/minimax-m2.5:free",
		max_hops:      3,
		prefer_suffix: false,
		rationale:     "Three reviewed references separate free-tier prices from canonical \
		                cache-write pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("kilo"),
		model:         "minimax/minimax-m3:discounted",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed discounted selector explicitly remains zero-priced.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("kilo"),
		model:         "moonshotai/kimi-k2",
		max_hops:      1,
		prefer_suffix: true,
		rationale:     "The Moonshot namespace is decorative; the reviewed bare Kimi card owns \
		                cache pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("mistral"),
		model:         "devstral-medium-latest",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed Mistral card explicitly declares zero cache pricing; a fuzzy \
		                dated sibling must not overwrite it.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("mistral"),
		model:         "codestral-latest",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed Mistral rolling card explicitly declares zero cache pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("mistral"),
		model:         "devstral-latest",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed Mistral rolling card explicitly declares zero cache pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("mistral"),
		model:         "magistral-medium-latest",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed Mistral rolling card explicitly declares zero cache pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("mistral"),
		model:         "ministral-3b-latest",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed Mistral rolling card explicitly declares zero cache pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("mistral"),
		model:         "ministral-8b-latest",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed Mistral rolling card explicitly declares zero cache pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("mistral"),
		model:         "mistral-large-latest",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed Mistral rolling card explicitly declares zero cache pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("mistral"),
		model:         "mistral-medium-latest",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed Mistral rolling card explicitly declares zero cache pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("mistral"),
		model:         "mistral-small-latest",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed Mistral rolling card explicitly declares zero cache pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("mistral"),
		model:         "pixtral-large-latest",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed Mistral rolling card explicitly declares zero cache pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("mistral"),
		model:         "voxtral-small-latest",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed Mistral rolling card explicitly declares zero cache pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("nanogpt"),
		model:         "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed NanoGPT row explicitly leaves limits unknown and prices zero; \
		                a namespaced source match is not evidence.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("nanogpt"),
		model:         "meituan-longcat/LongCat-Flash-Chat-FP8",
		max_hops:      0,
		prefer_suffix: false,
		rationale:     "The reviewed NanoGPT row explicitly leaves limits unknown and prices zero; \
		                a namespaced source match is not evidence.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("nvidia"),
		model:         "meta/llama-4-scout-17b-16e-instruct",
		max_hops:      2,
		prefer_suffix: true,
		rationale:     "Two reviewed exact references preserve the lower input/output price while \
		                adding cache-read and cache-write pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("nvidia"),
		model:         "qwen/qwen3.5-122b-a10b",
		max_hops:      2,
		prefer_suffix: true,
		rationale:     "Two reviewed exact references preserve lower input/output pricing while \
		                adding the cache-read component.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("nvidia"),
		model:         "thinkingmachines/inkling",
		max_hops:      2,
		prefer_suffix: true,
		rationale:     "Two reviewed exact references preserve input/output pricing while adding \
		                the deployment-specific cache-read component.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	SourceInheritanceOverride {
		provider:      Some("openrouter"),
		model:         "minimax/minimax-m2.5:free",
		max_hops:      2,
		prefer_suffix: true,
		rationale:     "Two reviewed references preserve free-tier prices while adding canonical \
		                cache-write pricing.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	reviewed_no_inheritance(
		"opencode-zen",
		"hy3-preview-free",
		"The reviewed free deployment explicitly preserves zero pricing.",
	),
	reviewed_no_inheritance(
		"opencode-zen",
		"laguna-s-2.1-free",
		"The reviewed free deployment explicitly preserves zero pricing.",
	),
	reviewed_no_inheritance(
		"opencode-zen",
		"ling-2.6-flash-free",
		"The reviewed free deployment explicitly preserves zero pricing.",
	),
	reviewed_no_inheritance(
		"opencode-zen",
		"ling-3.0-flash-free",
		"The reviewed free deployment explicitly preserves zero pricing.",
	),
	reviewed_no_inheritance(
		"opencode-zen",
		"ling-3.0-tiny-free",
		"The reviewed free deployment explicitly preserves zero pricing.",
	),
	reviewed_no_inheritance(
		"opencode-zen",
		"ring-2.6-1t-free",
		"The reviewed free deployment explicitly preserves zero pricing.",
	),
];

#[derive(Clone, Copy)]
struct ReferenceCandidatePolicy {
	provider:            &'static str,
	exclude_zero_prices: bool,
	rationale:           &'static str,
	provenance:          &'static str,
	expires_at_ms:       Option<u64>,
}

const REFERENCE_CANDIDATE_POLICIES: &[ReferenceCandidatePolicy] = &[ReferenceCandidatePolicy {
	provider:            "xai-oauth",
	exclude_zero_prices: true,
	rationale:           "Account-scoped OAuth rows carry unresolved zero prices and cannot serve \
	                      as canonical price references.",
	provenance:          "fixtures/llm-oracle/catalog/models.normalized.json",
	expires_at_ms:       None,
}];

#[derive(Clone, Copy)]
struct ExactInheritancePolicy {
	provider:                  &'static str,
	model:                     &'static str,
	inherit_limits:            bool,
	preserve_zero_cache_read:  bool,
	preserve_zero_cache_write: bool,
	rationale:                 &'static str,
	provenance:                &'static str,
	expires_at_ms:             Option<u64>,
}

const fn reviewed_inheritance_policy(
	provider: &'static str,
	model: &'static str,
	inherit_limits: bool,
	preserve_zero_cache_write: bool,
	rationale: &'static str,
) -> ExactInheritancePolicy {
	ExactInheritancePolicy {
		provider,
		model,
		inherit_limits,
		preserve_zero_cache_read: false,
		preserve_zero_cache_write,
		rationale,
		provenance: "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	}
}

const fn reviewed_zero_cache_read_policy(
	provider: &'static str,
	model: &'static str,
	rationale: &'static str,
) -> ExactInheritancePolicy {
	ExactInheritancePolicy {
		provider,
		model,
		inherit_limits: true,
		preserve_zero_cache_read: true,
		preserve_zero_cache_write: false,
		rationale,
		provenance: "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	}
}

const EXACT_INHERITANCE_POLICIES: &[ExactInheritancePolicy] = &[
	reviewed_inheritance_policy(
		"aimlapi",
		"nemotron-3-nano-omni-30b-a3b-reasoning:free",
		false,
		false,
		"The reviewed free selector preserves unknown limits while inheriting its canonical price.",
	),
	reviewed_inheritance_policy(
		"nanogpt",
		"google/gemini-3.5-flash",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"nanogpt",
		"google/gemini-3.6-flash",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"vercel-ai-gateway",
		"google/gemini-2.5-flash",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"vercel-ai-gateway",
		"google/gemini-2.5-flash-lite-preview-09-2025",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"vercel-ai-gateway",
		"google/gemini-2.5-flash-preview-09-2025",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"vercel-ai-gateway",
		"google/gemini-2.5-pro",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"vercel-ai-gateway",
		"google/gemini-3-pro-preview",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"vercel-ai-gateway",
		"google/gemini-3.1-flash-lite",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"vercel-ai-gateway",
		"google/gemini-3.5-flash",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"vercel-ai-gateway",
		"google/gemini-3.5-flash-lite",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"vercel-ai-gateway",
		"google/gemini-3.6-flash",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_zero_cache_read_policy(
		"vercel-ai-gateway",
		"mistral/devstral-small",
		"The reviewed gateway deployment explicitly preserves zero cache-read pricing.",
	),
	reviewed_inheritance_policy(
		"zenmux",
		"bytedance/doubao-seed-code",
		false,
		false,
		"The reviewed deployment preserves unknown output limits while inheriting canonical prices.",
	),
	reviewed_inheritance_policy(
		"zenmux",
		"kuaishou/kat-coder-air-v2.5",
		false,
		false,
		"The reviewed deployment preserves unknown output limits while inheriting canonical prices.",
	),
	reviewed_inheritance_policy(
		"zenmux",
		"kuaishou/kat-coder-pro-v2.5",
		false,
		false,
		"The reviewed deployment preserves unknown output limits while inheriting canonical prices.",
	),
	reviewed_inheritance_policy(
		"zenmux",
		"google/gemini-3.1-flash-lite",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"zenmux",
		"google/gemini-3.5-flash",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"zenmux",
		"google/gemini-3.5-flash-free",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"zenmux",
		"google/gemini-3.5-flash-lite",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
	reviewed_inheritance_policy(
		"zenmux",
		"google/gemini-3.6-flash",
		true,
		true,
		"The reviewed route explicitly preserves zero cache-write pricing.",
	),
];
fn inherit_source_references(
	models: &mut BTreeMap<Str, BTreeMap<Str, SourceModelRecord>>,
) -> Result<(), CompileError> {
	let snapshot = models.clone();
	let mut exact: BTreeMap<Str, (Str, Str)> = BTreeMap::new();
	for (provider, rows) in &snapshot {
		for (model, row) in rows {
			let candidate_policy = REFERENCE_CANDIDATE_POLICIES
				.iter()
				.find(|policy| policy.provider == provider.as_str());
			if candidate_policy.is_some_and(|policy| {
				let _review_evidence = (policy.rationale, policy.provenance, policy.expires_at_ms);
				policy.exclude_zero_prices && source_cost_is_zero(&row.cost)
			}) {
				continue;
			}
			let key = Str::from(model.trim().to_ascii_lowercase());
			let identity = (provider.clone(), model.clone());
			let replace = exact.get(&key).is_some_and(|existing| {
				reference_rank(&identity, row)
					> reference_rank(existing, &snapshot[&existing.0][&existing.1])
			});
			if replace || !exact.contains_key(&key) {
				exact.insert(key, identity);
			}
		}
	}
	let mut suffix: BTreeMap<Str, (Str, Str)> = BTreeMap::new();
	for identity in exact.values() {
		let Some(candidate) = identity
			.1
			.rsplit('/')
			.next()
			.filter(|candidate| *candidate != identity.1.as_str())
		else {
			continue;
		};
		let candidate = Str::from(candidate.trim().to_ascii_lowercase());
		let row = &snapshot[&identity.0][&identity.1];
		let replace = suffix.get(&candidate).is_some_and(|existing| {
			reference_rank(identity, row)
				> reference_rank(existing, &snapshot[&existing.0][&existing.1])
		});
		if replace || !suffix.contains_key(&candidate) {
			suffix.insert(candidate, identity.clone());
		}
	}
	for (provider, rows) in models {
		for (model, row) in rows {
			row.omitted_dynamic_pricing =
				[&row.cost.input, &row.cost.output, &row.cost.cache_read, &row.cost.cache_write]
					.into_iter()
					.any(|number| number.to_string() == "-1000000");
			if model.starts_with('@') {
				continue;
			}
			let max_hops = SOURCE_INHERITANCE_OVERRIDES
				.iter()
				.find(|override_| {
					override_
						.provider
						.is_none_or(|candidate| candidate == provider)
						&& override_.model == model
				})
				.map_or(1, |override_| {
					let _review_evidence =
						(override_.rationale, override_.provenance, override_.expires_at_ms);
					override_.max_hops
				});
			if max_hops == 0 {
				continue;
			}
			let exact_reference = EXACT_SOURCE_REFERENCES.iter().find(|reference| {
				(reference.provider == "*" || reference.provider == provider.as_str())
					&& reference.model == model.as_str()
			});
			let mut current = (provider.clone(), model.clone());
			let mut visited = BTreeSet::new();
			while let Some(reference) = exact_reference
				.filter(|_| visited.is_empty())
				.map(|reference| {
					let _review_evidence =
						(reference.rationale, reference.provenance, reference.expires_at_ms);
					(Str::from(reference.reference_provider), Str::from(reference.reference_model))
				})
				.or_else(|| select_reference(&current.1, &current.0, &current.1, &exact, &suffix))
			{
				if !visited.insert(reference.clone()) {
					break;
				}
				let reference_row = &snapshot[&reference.0][&reference.1];
				let inheritance_policy = EXACT_INHERITANCE_POLICIES.iter().find(|policy| {
					policy.provider == provider.as_str() && policy.model == model.as_str()
				});
				let (inherit_limits, preserve_zero_cache_read, preserve_zero_cache_write) =
					inheritance_policy.map_or((true, false, false), |policy| {
						let _review_evidence =
							(policy.rationale, policy.provenance, policy.expires_at_ms);
						(
							policy.inherit_limits,
							policy.preserve_zero_cache_read,
							policy.preserve_zero_cache_write,
						)
					});
				inherit_source_row(
					row,
					reference_row,
					inherit_limits,
					preserve_zero_cache_read,
					preserve_zero_cache_write,
				);
				row.inherited_from
					.get_or_insert_with(|| ModelKey::new(format!("{}/{}", reference.0, reference.1)));
				current = reference;
				if visited.len() >= max_hops {
					break;
				}
			}
		}
	}
	Ok(())
}

fn source_inheritance_override(
	provider: &str,
	model: &str,
) -> Option<&'static SourceInheritanceOverride> {
	SOURCE_INHERITANCE_OVERRIDES.iter().find(|override_| {
		override_
			.provider
			.is_none_or(|candidate| candidate == provider)
			&& override_.model == model
	})
}

fn select_reference(
	model: &str,
	provider: &str,
	original_model: &str,
	exact: &BTreeMap<Str, (Str, Str)>,
	suffix: &BTreeMap<Str, (Str, Str)>,
) -> Option<(Str, Str)> {
	if let Some(override_) = EXACT_SOURCE_REFERENCES.iter().find(|override_| {
		(override_.provider == "*" || override_.provider == provider)
			&& override_.model.eq_ignore_ascii_case(model)
	}) {
		let _review_evidence = (override_.rationale, override_.provenance, override_.expires_at_ms);
		return Some((Str::from(override_.reference_provider), Str::from(override_.reference_model)));
	}
	let mut candidates = reference_keys(model);
	let prefer_suffix = source_inheritance_override(provider, model)
		.is_some_and(|override_| override_.prefer_suffix)
		|| model.split_once('/').is_some_and(|(namespace, bare)| {
			let bare = classify(ClassificationInput {
				phase: ClassificationPhase::CatalogCompiler,
				provider,
				model: bare,
				observed_at_ms: None,
			});
			bare.family.as_str() == "qwen" && namespace.eq_ignore_ascii_case("qwen")
		});
	if prefer_suffix && candidates.len() > 1 {
		candidates.swap(0, 1);
	}
	let classified = classify(ClassificationInput {
		phase: ClassificationPhase::CatalogCompiler,
		provider,
		model,
		observed_at_ms: None,
	});
	if classified.logical_model.as_str() != model {
		candidates.push(classified.logical_model);
	}
	for candidate in candidates {
		let key = Str::from(candidate.trim().to_ascii_lowercase());
		let Some(reference) = exact.get(&key).or_else(|| suffix.get(&key)) else {
			continue;
		};
		if reference.0.as_str() != provider || reference.1.as_str() != original_model {
			return Some(reference.clone());
		}
	}
	None
}

fn reference_keys(model: &str) -> Vec<Str> {
	const MARKERS: &[&str] = &["cloud", "free", "discounted", "latest", "exacto", "search", "fp8"];
	let mut keys = Vec::new();
	let mut queue = vec![model.trim().to_ascii_lowercase()];
	let mut next = 0;
	while let Some(candidate) = queue.get(next).cloned() {
		next += 1;
		let candidate = candidate.trim().to_owned();
		if candidate.is_empty() || keys.iter().any(|seen: &Str| seen.as_str() == candidate) {
			continue;
		}
		keys.push(Str::from(candidate.clone()));
		if let Some((_, suffix)) = candidate.rsplit_once('/') {
			queue.push(suffix.to_owned());
		}
		if candidate.contains(':') {
			queue.push(candidate.replace(':', "-"));
		}
		for marker in MARKERS {
			if let Some(prefix) = candidate.strip_suffix(marker)
				&& let Some(stripped) = prefix.strip_suffix(['-', ':'])
			{
				queue.push(stripped.to_owned());
			}
		}
	}
	keys
}

fn reference_rank(identity: &(Str, Str), row: &SourceModelRecord) -> (u64, u64, bool, bool) {
	(
		row.context_window.unwrap_or(0),
		row.max_tokens.unwrap_or(0),
		source_price_present(&row.cost.cache_read) || source_price_present(&row.cost.cache_write),
		identity.0.as_str() == "openai",
	)
}

fn source_cost_is_zero(cost: &SourceCost) -> bool {
	[&cost.input, &cost.output, &cost.cache_read, &cost.cache_write]
		.into_iter()
		.all(|number| number.as_u64() == Some(0))
}

fn inherit_source_row(
	target: &mut SourceModelRecord,
	reference: &SourceModelRecord,
	inherit_limits: bool,
	preserve_zero_cache_read: bool,
	preserve_zero_cache_write: bool,
) {
	for (index, (target, source)) in [
		(&mut target.cost.input, &reference.cost.input),
		(&mut target.cost.output, &reference.cost.output),
		(&mut target.cost.cache_read, &reference.cost.cache_read),
		(&mut target.cost.cache_write, &reference.cost.cache_write),
	]
	.into_iter()
	.enumerate()
	{
		if preserve_zero_cache_read && index == 2 {
			continue;
		}
		if preserve_zero_cache_write && index == 3 {
			continue;
		}
		if target.as_u64() == Some(0) && source_price_present(source) {
			*target = source.clone();
		}
	}
	if inherit_limits {
		if target.context_window.is_none() {
			target.context_window = reference.context_window;
		}
		if target.max_tokens.is_none() {
			target.max_tokens = reference.max_tokens;
		}
	}
}

fn source_url_census(
	providers: &BTreeMap<Str, SourceProviderRecord>,
	models: &BTreeMap<Str, BTreeMap<Str, SourceModelRecord>>,
) -> usize {
	let mut urls = BTreeSet::new();
	for provider in providers.values() {
		urls.insert(provider.base_url.as_str());
		urls.extend(provider.fallback_base_urls.iter().map(Str::as_str));
		urls.extend(
			provider
				.headers
				.values()
				.map(Str::as_str)
				.filter(|value| value.starts_with("http://") || value.starts_with("https://")),
		);
	}
	urls.extend(
		models
			.values()
			.flat_map(BTreeMap::values)
			.filter_map(|model| model.base_url.as_ref())
			.map(Str::as_str)
			.filter(|value| !value.trim().is_empty()),
	);
	urls.len()
}

fn validate_oracle_census(census: CompilerCensus) -> Result<(), CompileError> {
	for (actual, expected, label) in [
		(census.logical_models, ORACLE_LOGICAL_MODELS, "logical models"),
		(census.providers, ORACLE_PROVIDERS, "providers"),
		(census.raw_provider_keys, ORACLE_RAW_PROVIDER_KEYS, "raw provider keys"),
		(census.urls, ORACLE_URLS, "URLs"),
		(census.transports, ORACLE_TRANSPORTS, "transports"),
		(census.active_transports, ORACLE_ACTIVE_TRANSPORTS, "active transports"),
	] {
		if actual != expected {
			return Err(CompileError::Invariant(Str::from(format!(
				"expected {expected} {label}, found {actual}"
			))));
		}
	}
	Ok(())
}
fn compile_oauth_specs(
	input: &str,
) -> Result<(Vec<OAuthSpec>, BTreeMap<Str, OAuthSpecId>), CompileError> {
	let document: SourceOAuthDocument = toml::from_str(input)?;
	let mut specs = Vec::with_capacity(document.provider.len());
	let mut ids = BTreeMap::new();
	for source in document.provider {
		validate_url(&source.authorize_url)?;
		if !source.token_url.is_empty() {
			validate_url(&source.token_url)?;
		}
		let mut parameters = source
			.extra_auth_params
			.iter()
			.filter(|(name, _)| {
				!matches!(
					name.as_str(),
					"callback_host" | "callback_path" | "client_secret" | "refresh_url"
				)
			})
			.map(|(name, value)| OAuthParameter { name: name.clone(), value: value.clone() })
			.collect::<Vec<_>>();
		parameters.sort_by(|left, right| {
			left
				.name
				.cmp(&right.name)
				.then_with(|| left.value.cmp(&right.value))
		});
		let flow = match source.kind {
			SourceOAuthKind::Pkce => {
				let host = source
					.extra_auth_params
					.get("callback_host")
					.map_or("127.0.0.1", Str::as_str);
				let path = source
					.extra_auth_params
					.get("callback_path")
					.map_or("/callback", Str::as_str);
				let port = source.callback_port.ok_or_else(|| {
					CompileError::Invariant(Str::from(format!(
						"OAuth PKCE flow `{}` has no callback port",
						source.provider
					)))
				})?;
				OAuthFlowSpec::Pkce {
					authorize_url:        source.authorize_url.clone(),
					redirect_uri:         Str::from(format!("http://{host}:{port}{path}")),
					completion:           OAuthCompletion::CallbackUrl,
					authorize_parameters: parameters.clone().into_boxed_slice(),
				}
			},
			SourceOAuthKind::DeviceCode => OAuthFlowSpec::DeviceCode {
				device_authorization_url: source.authorize_url.clone(),
				polling:                  OAuthPollingSpec {
					maximum_polls:       120,
					default_interval_ms: 5_000,
					maximum_interval_ms: 30_000,
				},
			},
			SourceOAuthKind::CustomExchange => OAuthFlowSpec::Custom {
				authorize_url: source.authorize_url.clone(),
				exchange:      source.exchange.ok_or_else(|| {
					CompileError::Invariant(Str::from(format!(
						"custom OAuth flow `{}` has no exchange engine",
						source.provider
					)))
				})?,
				parameters:    parameters.clone().into_boxed_slice(),
				polling:       None,
			},
		};
		let refresh = match source.extra_auth_params.get("refresh_url") {
			Some(url) => {
				OAuthRefreshBehavior::Endpoint { url: url.clone(), parameters: Box::new([]) }
			},
			None if source.token_url.is_empty() => OAuthRefreshBehavior::Unsupported,
			None => OAuthRefreshBehavior::TokenEndpoint,
		};
		let principal_resolution = match source.exchange {
			Some(OAuthExchangeKind::OpenAiCodexClaims) => Some(PrincipalResolution::IdTokenClaim {
				claim: Str::from("https://api.openai.com/auth/chatgpt_account_id"),
			}),
			_ => None,
		};
		let canonical = serde_json::to_vec(&(
			&source.client_id,
			&source.token_url,
			&source.scopes,
			&flow,
			&refresh,
			&principal_resolution,
		))?;
		let id = OAuthSpecId::new(content_id("oauth", &canonical));
		if ids.insert(source.provider.clone(), id.clone()).is_some() {
			return Err(CompileError::Invariant(Str::from(format!(
				"duplicate OAuth flow `{}`",
				source.provider
			))));
		}
		specs.push(OAuthSpec {
			id,
			client_id: source.client_id,
			token_url: source.token_url,
			scopes: source.scopes.into_boxed_slice(),
			audience: None,
			placement: OAuthTokenPlacement::Header {
				name:   Str::from("authorization"),
				prefix: Str::from("Bearer "),
			},
			token_parameters: parameters.into_boxed_slice(),
			flow,
			refresh,
			principal_resolution,
		});
	}
	specs.sort_by(|left, right| left.id.cmp(&right.id));
	Ok((specs, ids))
}

fn facet_operations(facets: &[SourceFacet]) -> OperationBits {
	let mut operations = OperationBits::empty();
	for facet in facets {
		match facet {
			SourceFacet::Chat => operations.insert_kind(OperationKind::Chat),
			SourceFacet::Embeddings => operations.insert_kind(OperationKind::Embed),
			SourceFacet::ImageGeneration => operations.insert_kind(OperationKind::GenerateImage),
			SourceFacet::VideoGeneration => operations.insert_kind(OperationKind::GenerateVideo),
			SourceFacet::AudioSpeech => operations.insert_kind(OperationKind::Speak),
			SourceFacet::AudioTranscription => operations.insert_kind(OperationKind::Transcribe),
			SourceFacet::Realtime => operations.insert_kind(OperationKind::Realtime),
			SourceFacet::WebSearch => operations.insert_kind(OperationKind::Search),
			SourceFacet::Tokenization => {
				operations.insert_kind(OperationKind::CountTokens);
				operations.insert_kind(OperationKind::Tokenize);
				operations.insert_kind(OperationKind::Detokenize);
			},
		}
	}
	operations
}

#[allow(
	clippy::type_complexity,
	reason = "compiler phase returns each independently interned table"
)]
fn compile_providers(
	providers: BTreeMap<Str, SourceProviderRecord>,
	oauth_ids: &BTreeMap<Str, OAuthSpecId>,
	oauth_specs: &[OAuthSpec],
) -> Result<
	(
		Vec<ProviderDef>,
		Vec<AuthSpec>,
		Vec<HeaderProfile>,
		Vec<DiscoverySpec>,
		Vec<RouteDef>,
		BTreeMap<Str, Vec<RouteId>>,
		BTreeMap<WirePolicyId, WirePolicy>,
		BTreeMap<Str, WirePolicyId>,
	),
	CompileError,
> {
	let mut output = Vec::new();
	let mut auth_by_id = BTreeMap::new();
	let mut headers_by_id = BTreeMap::new();
	let mut discovery_by_id = BTreeMap::new();
	let mut routes = Vec::new();
	let mut provider_routes = BTreeMap::new();
	let mut policies = BTreeMap::new();
	let mut provider_policies = BTreeMap::new();
	for (provider_key, source) in providers {
		let provider_id = ProviderId::new(provider_key.clone());
		let auth = compile_auth(&source.auth, oauth_ids)?;
		let auth_id = auth.id.clone();
		auth_by_id.entry(auth_id.clone()).or_insert(auth);
		let mut provider_auth_ids = vec![auth_id.clone()];
		if let Some(flow) = source.oauth_flow.as_ref()
			&& !matches!(&source.auth, SourceAuth::Oauth { flow: request_flow } if request_flow == flow)
		{
			let login_auth = compile_auth(&SourceAuth::Oauth { flow: flow.clone() }, oauth_ids)?;
			provider_auth_ids.push(login_auth.id.clone());
			auth_by_id
				.entry(login_auth.id.clone())
				.or_insert(login_auth);
		}
		let header = compile_headers(&source.headers)?;
		let header_id = header.id.clone();
		headers_by_id.entry(header_id.clone()).or_insert(header);
		let discovery = source
			.discovery
			.as_ref()
			.map(compile_discovery)
			.transpose()?;
		let discovery_id = discovery.as_ref().map(|entry| entry.id.clone());
		if let Some(discovery) = discovery {
			discovery_by_id
				.entry(discovery.id.clone())
				.or_insert(discovery);
		}
		let policy = compile_wire_policy(WirePolicy::overrides(), &source.compat)?;
		let policy_id = policy.content_id();
		policies.entry(policy_id.clone()).or_insert(policy);
		provider_policies.insert(provider_key.clone(), policy_id.clone());
		let mut urls = Vec::with_capacity(1 + source.fallback_base_urls.len());
		urls.push(source.base_url.clone());
		urls.extend(source.fallback_base_urls.iter().cloned());
		let mut owned_routes = Vec::with_capacity(urls.len());
		let mut route_operations = facet_operations(&source.facets);
		if discovery_id.is_some() {
			route_operations.insert_kind(OperationKind::DiscoverModels);
		}
		if !matches!(&source.auth, SourceAuth::None) || source.oauth_flow.is_some() {
			route_operations.insert_kind(OperationKind::Auth);
		}
		for (index, url) in urls.into_iter().enumerate() {
			validate_url(&url)?;
			let suffix = if index == 0 {
				"primary".to_owned()
			} else {
				format!("fallback-{index}")
			};
			let route_id = RouteId::new(format!("{provider_key}/{suffix}"));
			let (default_codec, default_transport) = translate_transport(source.transport);
			let codec = source.codec.clone().unwrap_or(default_codec);
			let transport = source.route_transport.unwrap_or(default_transport);
			let origin = url
				.as_str()
				.split('/')
				.take(3)
				.collect::<Vec<_>>()
				.join("/");
			routes.push(RouteDef {
				id: route_id.clone(),
				provider: provider_id.clone(),
				codec_profile: source.codec_profile,
				codec,
				transport,
				endpoint: EndpointSpec { base_url: url, region: None },
				auth: auth_id.clone(),
				headers: header_id.clone(),
				discovery: discovery_id.clone(),
				capability_limits: RouteRestrictions {
					operations: (route_operations != OperationBits::empty()).then_some(route_operations),
					..RouteRestrictions::default()
				},
				trust_domain: TrustDomain {
					origin:          Str::from(origin),
					redirects:       RedirectTrust::SameOrigin,
					allow_plaintext: false,
				},
				codex_transport: if source.codex_transport.as_deref() == Some("websocket-preferred") {
					CodexTransportPreference::WebsocketPreferred
				} else {
					CodexTransportPreference::HttpOnly
				},
				use_responses_lite: Some(source.codex_responses_lite),
				priority: None,
			});
			owned_routes.push(route_id);
		}
		let mapping = match source.mapping {
			SourceMapping::Concrete => RegistryMapping::Concrete,
			SourceMapping::Alias { target, reason } => {
				RegistryMapping::Alias { target: ProviderId::new(target), reason }
			},
			SourceMapping::Replacement { component, reason } => {
				RegistryMapping::Replacement { component, reason }
			},
		};
		let mut management_operations = OperationBits::empty();
		if discovery_id.is_some() {
			management_operations.insert_kind(OperationKind::DiscoverModels);
		}
		if !matches!(&source.auth, SourceAuth::None) || source.oauth_flow.is_some() {
			management_operations.insert_kind(OperationKind::Auth);
		}
		let refresh_flow = source.oauth_flow.as_ref().or_else(|| match &source.auth {
			SourceAuth::Oauth { flow } => Some(flow),
			_ => None,
		});
		let refresh = refresh_flow
			.and_then(|flow| oauth_ids.get(flow))
			.and_then(|id| oauth_specs.iter().find(|spec| &spec.id == id))
			.is_some_and(|spec| !matches!(spec.refresh, OAuthRefreshBehavior::Unsupported));
		let discovery_defaults = discovery_id.is_some().then(|| DiscoveryDefaults {
			wire_policy:          policy_id.clone(),
			extended_wire_policy: None,
			context:              ContextStrategy::Replay,
			thinking:             None,
			pricing:              Pricing::default(),
		});
		output.push(ProviderDef {
			id: provider_id,
			name: humanize(&provider_key),
			auth: provider_auth_ids.into_boxed_slice(),
			management: ManagementCapabilities {
				operations: management_operations,
				multiple_accounts: source.oauth_flow.is_some(),
				refresh,
				principal_quota: true,
			},
			routes: owned_routes.clone().into_boxed_slice(),
			wire_policy: policy_id.clone(),
			discovery_defaults,
			mapping,
		});
		provider_routes.insert(provider_key, owned_routes);
	}
	output.sort_by(|left, right| left.id.cmp(&right.id));
	routes.sort_by(|left, right| left.id.cmp(&right.id));
	Ok((
		output,
		auth_by_id.into_values().collect(),
		headers_by_id.into_values().collect(),
		discovery_by_id.into_values().collect(),
		routes,
		provider_routes,
		policies,
		provider_policies,
	))
}

fn compile_model_routes(
	models: &BTreeMap<Str, BTreeMap<Str, SourceModelRecord>>,
	providers: &mut [ProviderDef],
	routes: &mut Vec<RouteDef>,
	header_profiles: &mut Vec<HeaderProfile>,
	provider_routes: &BTreeMap<Str, Vec<RouteId>>,
) -> Result<BTreeMap<(Str, Str), Vec<RouteId>>, CompileError> {
	let mut output = BTreeMap::new();
	let mut route_by_shape: BTreeMap<Vec<u8>, RouteId> = BTreeMap::new();
	for (provider, rows) in models {
		let inherited = provider_routes.get(provider).ok_or_else(|| {
			CompileError::Invariant(Str::from(format!(
				"model provider `{provider}` has no curated route"
			)))
		})?;
		let primary_id = inherited.first().ok_or_else(|| {
			CompileError::Invariant(Str::from(format!("provider `{provider}` has no primary route")))
		})?;
		let primary = routes
			.iter()
			.find(|route| &route.id == primary_id)
			.cloned()
			.ok_or_else(|| CompileError::Invariant(Str::from("provider primary route is missing")))?;
		for (model, row) in rows {
			let embedding_override = exact_capability_override(provider, model)
				.is_some_and(|override_| override_.correction == CapabilityCorrection::Embedding);
			let has_override = row
				.base_url
				.as_ref()
				.is_some_and(|url| !url.trim().is_empty())
				|| row.api.is_some()
				|| !row.headers.is_empty()
				|| row.use_responses_lite.is_some()
				|| row.prefer_websockets.is_some()
				|| row.priority.is_some()
				|| embedding_override;
			if !has_override {
				output.insert((provider.clone(), model.clone()), inherited.clone());
				continue;
			}
			let mut route = primary.clone();
			if let Some(url) = row.base_url.as_ref().filter(|url| !url.trim().is_empty()) {
				validate_url(url)?;
				route.endpoint.base_url = url.clone();
				route.trust_domain.origin = Str::from(
					url.as_str()
						.split('/')
						.take(3)
						.collect::<Vec<_>>()
						.join("/"),
				);
			}
			if let Some(transport) = row.api {
				(route.codec, route.transport) = translate_transport(transport);
			}
			if embedding_override {
				let operations = route
					.capability_limits
					.operations
					.get_or_insert_with(OperationBits::empty);
				operations.insert_kind(OperationKind::Embed);
			}
			if !row.headers.is_empty() {
				let profile = compile_headers(&row.headers).map_err(|error| {
					CompileError::Invariant(Str::from(format!(
						"source model `{provider}/{model}` has invalid headers: {error}"
					)))
				})?;
				route.headers = profile.id.clone();
				if !header_profiles
					.iter()
					.any(|existing| existing.id == profile.id)
				{
					header_profiles.push(profile);
				}
			}
			if let Some(lite) = row.use_responses_lite {
				route.use_responses_lite = Some(lite);
			}
			if row.prefer_websockets == Some(true) {
				route.codex_transport = CodexTransportPreference::WebsocketPreferred;
			}
			route.priority = row.priority;
			let shape = serde_json::to_vec(&(
				&route.provider,
				&route.codec,
				route.transport,
				&route.endpoint,
				&route.auth,
				&route.headers,
				&route.capability_limits,
				&route.codex_transport,
				route.use_responses_lite,
				route.priority,
			))?;
			let route_id = if let Some(existing) = route_by_shape.get(&shape) {
				existing.clone()
			} else {
				let id = RouteId::new(content_id("route", &shape));
				route.id = id.clone();
				routes.push(route);
				route_by_shape.insert(shape, id.clone());
				if let Some(owner) = providers
					.iter_mut()
					.find(|entry| entry.id.as_str() == provider.as_str())
				{
					let mut owned = owner.routes.to_vec();
					owned.push(id.clone());
					owned.sort();
					owned.dedup();
					owner.routes = owned.into_boxed_slice();
				}
				id
			};
			output.insert((provider.clone(), model.clone()), vec![route_id]);
		}
	}
	routes.sort_by(|left, right| left.id.cmp(&right.id));
	header_profiles.sort_by(|left, right| left.id.cmp(&right.id));
	Ok(output)
}

fn compile_models(
	providers: BTreeMap<Str, BTreeMap<Str, SourceModelRecord>>,
	model_routes: &BTreeMap<(Str, Str), Vec<RouteId>>,
	provider_policies: &BTreeMap<Str, WirePolicyId>,
	policies: &mut BTreeMap<WirePolicyId, WirePolicy>,
	thinking_policies: &mut BTreeMap<crate::id::ThinkingPolicyId, ThinkingPolicy>,
	provider_facets: &BTreeMap<Str, Vec<SourceFacet>>,
	reviewed_reasoning_content: &BTreeSet<Str>,
) -> Result<(Vec<ModelSpec>, Vec<CatalogAlias>), CompileError> {
	let mut output = Vec::new();
	let mut aliases = Vec::new();
	for (provider, rows) in providers {
		let provider_policy_id = provider_policies.get(&provider).ok_or_else(|| {
			CompileError::Invariant(Str::from(format!("provider `{provider}` has no wire policy")))
		})?;
		if !policies.contains_key(provider_policy_id) {
			return Err(CompileError::Invariant(Str::from("provider wire policy was not interned")));
		}
		let facets = provider_facets
			.get(&provider)
			.map(Vec::as_slice)
			.unwrap_or_default();
		let identities: BTreeMap<Str, ModelClassification> = rows
			.keys()
			.map(|model| {
				let classified = classify(ClassificationInput {
					phase: ClassificationPhase::CatalogCompiler,
					provider: &provider,
					model,
					observed_at_ms: None,
				});
				(model.clone(), classified)
			})
			.collect();
		let collapsible = collapsible_groups(&identities);
		let mut logical: BTreeMap<Str, Vec<(Str, SourceModelRecord, ModelClassification)>> =
			BTreeMap::new();
		for (wire, row) in rows {
			let classified = identities
				.get(&wire)
				.expect("classification index is complete");
			let key = if collapsible.contains(classified.logical_model.as_str()) {
				classified.logical_model.clone()
			} else {
				wire.clone()
			};
			logical
				.entry(key)
				.or_default()
				.push((wire, row, classified.clone()));
		}
		for (logical_id, members) in logical {
			let first = &members[0];
			let mut merged_row = first.1.clone();
			for (_, row, _) in members.iter().skip(1) {
				for input in &row.input {
					if !merged_row.input.contains(input) {
						merged_row.input.push(*input);
					}
				}
				for output in &row.output {
					if !merged_row.output.contains(output) {
						merged_row.output.push(*output);
					}
				}
			}
			merged_row.reasoning = merged_row.reasoning
				|| (members.len() > 1
					&& members.iter().any(|(_, _, classified)| {
						classified.effort.is_some() || classified.thinking_variant
					}));
			let family = first.2.family.clone();
			let display_name = first
				.1
				.name
				.clone()
				.unwrap_or_else(|| humanize(&logical_id));
			let context_window = members
				.iter()
				.filter_map(|(_, row, _)| row.context_window)
				.max();
			let maximum_output_tokens = members
				.iter()
				.filter_map(|(_, row, _)| row.max_tokens)
				.max();
			let mut routes = Vec::new();
			let mut wire_ids = Vec::new();
			for (wire, row, _) in &members {
				let member_routes = model_routes
					.get(&(provider.clone(), wire.clone()))
					.ok_or_else(|| {
						CompileError::Invariant(Str::from(format!(
							"source model `{provider}/{wire}` has no route"
						)))
					})?;
				for route in member_routes {
					routes.push(route.clone());
					wire_ids.push((
						route.clone(),
						WireModelId::new(row.request_model_id.clone().unwrap_or_else(|| wire.clone())),
					));
				}
			}
			routes.sort();
			routes.dedup();
			if members.len() > 1 {
				for route in &routes {
					wire_ids.push((route.clone(), WireModelId::new(logical_id.clone())));
				}
			}
			let pricing = compile_pricing(provider.as_str(), logical_id.as_str(), &first.1.cost)?;
			let capability_override = exact_capability_override(&provider, &logical_id);
			let capabilities = conservative_capabilities(
				&merged_row,
				facets,
				capability_override.map(|override_| override_.correction),
			);
			let reviewed_model_key = Str::from(format!("{provider}/{logical_id}"));
			let reasoning_content_overlay = reviewed_reasoning_content.contains(&reviewed_model_key);
			let mut wire_policy = if let Some(overrides) = first.1.compat.as_ref() {
				compile_wire_policy(WirePolicy::overrides(), overrides)?
			} else if reasoning_content_overlay {
				WirePolicy::overrides()
			} else {
				WirePolicy::baseline()
			};
			if reasoning_content_overlay
				&& wire_policy
					.reasoning
					.requires_content_for_all_assistant_turns
					.is_none()
			{
				wire_policy
					.reasoning
					.requires_content_for_all_assistant_turns = Some(true);
			}
			if first.1.compat.is_some() {
				policies
					.entry(wire_policy.content_id())
					.or_insert_with(|| wire_policy.clone());
			}
			if let Some(enabled) = first.1.cursor_max_mode {
				wire_policy.context.extended_mode = Some(ExtendedContextMode::from_enabled(enabled));
			}
			if let Some(omit) = first.1.omit_max_output_tokens {
				wire_policy.context.max_output_tokens = Some(if omit {
					MaxOutputTokensEmission::Omit
				} else {
					MaxOutputTokensEmission::Emit
				});
			}
			if let Some(kind) = first.1.apply_patch_tool_type.as_deref() {
				wire_policy.tool.apply_patch =
					Some(kind.parse::<ApplyPatchWireKind>().map_err(|_| {
						CompileError::Invariant(Str::from(format!(
							"unknown apply-patch wire kind `{kind}` for `{provider}/{logical_id}`"
						)))
					})?);
			}
			if let Some(supported) = first.1.supports_computer_use {
				wire_policy.tool.computer_use = Some(if supported {
					ComputerUseWireSupport::Native
				} else {
					ComputerUseWireSupport::Unsupported
				});
			}
			if let Some(supported) = first.1.supports_computer_use_config {
				wire_policy.tool.computer_use_config = Some(if supported {
					ComputerUseConfigSupport::Supported
				} else {
					ComputerUseConfigSupport::Unsupported
				});
			}
			let wire_policy_id = wire_policy.content_id();
			policies
				.entry(wire_policy_id.clone())
				.or_insert(wire_policy);
			let (thinking, mut thinking_routing) = compile_thinking(&members)?;
			if let Some(mode) = first.1.reasoning_mode.as_deref() {
				thinking_routing.reasoning_mode =
					Some(mode.parse::<ReasoningMode>().map_err(|_| {
						CompileError::Invariant(Str::from(format!("unknown reasoning mode `{mode}`")))
					})?);
			}
			let key = ModelKey::new(format!("{provider}/{logical_id}"));
			let thinking_id = thinking.as_ref().map(|profile| {
				let id = profile.content_id();
				thinking_policies
					.entry(id.clone())
					.or_insert_with(|| profile.clone());
				id
			});
			for (wire, _, classified) in &members {
				if wire.as_str() != logical_id.as_str() {
					aliases.push(CatalogAlias {
						alias:      Str::from(format!("{provider}/{wire}")),
						target:     key.clone(),
						rationale:  classified.evidence.rationale.clone(),
						provenance: classified.evidence.provenance.clone(),
					});
				}
			}
			let mut provenance_sources = vec![ProvenanceSource {
				kind:           ProvenanceKind::Bundled,
				origin:         Str::from("catalog-oracle/models.json.zst"),
				revision:       None,
				confidence:     EvidenceConfidence::Declared,
				observed_at_ms: None,
			}];
			if let Some(reference) = members
				.iter()
				.find_map(|(_, row, _)| row.inherited_from.as_ref())
			{
				provenance_sources.push(ProvenanceSource {
					kind:           ProvenanceKind::Bundled,
					origin:         Str::from(format!("catalog-oracle:inherit:{reference}")),
					revision:       None,
					confidence:     EvidenceConfidence::Inferred,
					observed_at_ms: None,
				});
			}
			if members
				.iter()
				.any(|(_, row, _)| row.omitted_dynamic_pricing)
			{
				provenance_sources.push(ProvenanceSource {
					kind:           ProvenanceKind::Bundled,
					origin:         Str::from("catalog-oracle:omit:dynamic-pricing-sentinel"),
					revision:       None,
					confidence:     EvidenceConfidence::Inferred,
					observed_at_ms: None,
				});
			}
			if let Some(override_) = capability_override {
				provenance_sources.push(ProvenanceSource {
					kind:           ProvenanceKind::Bundled,
					origin:         Str::from(format!("{}#{}", override_.provenance, override_.id)),
					revision:       None,
					confidence:     EvidenceConfidence::Verified,
					observed_at_ms: None,
				});
			}
			output.push(ModelSpec {
				key,
				family,
				display_name,
				wire_ids: wire_ids.into_boxed_slice(),
				routes: routes.into_boxed_slice(),
				capabilities,
				limits: ModelLimits {
					context_window,
					maximum_input_tokens: None,
					maximum_output_tokens,
					maximum_batch: None,
				},
				thinking: thinking_id,
				thinking_routing,
				wire_policy: wire_policy_id,
				context: ContextStrategy::Replay,
				pricing,
				availability: ModelAvailability::Unspecified,
				provenance: ModelProvenance {
					sources:          provenance_sources.into_boxed_slice(),
					updated_at_ms:    None,
					blocked_until_ms: None,
					deprecated:       members.iter().all(|(_, row, _)| row.deprecated),
				},
				context_promotion_target: first.1.context_promotion_target.as_ref().map(|target| {
					ModelKey::new(if target.contains('/') {
						target.clone()
					} else {
						Str::from(format!("{provider}/{target}"))
					})
				}),
				remote_compaction: first.1.remote_compaction.as_ref().map(|source| {
					ModelRemoteCompaction {
						enabled:              source.enabled,
						transport:            source
							.api
							.map(|transport| translate_transport(transport).0),
						endpoint:             source.endpoint.clone(),
						v2_streaming_enabled: source.v2_streaming_enabled,
						v2_endpoint:          source.v2_endpoint.clone(),
						streaming_endpoint:   source.streaming_endpoint.clone(),
						model:                source.model.clone().map(WireModelId::new),
						trigger_tokens:       None,
						target_tokens:        None,
					}
				}),
				premium_multiplier_millionths: first
					.1
					.premium_multiplier
					.as_ref()
					.map(decimal_millionths)
					.transpose()?
					.map(PremiumMultiplier::from_millionths),
			});
		}
	}
	aliases.sort_by(|left, right| {
		left
			.alias
			.cmp(&right.alias)
			.then_with(|| left.target.cmp(&right.target))
	});
	aliases.dedup_by(|left, right| left.alias == right.alias && left.target == right.target);
	let attached = output
		.iter()
		.filter_map(|model| model.thinking.as_ref())
		.cloned()
		.collect::<BTreeSet<_>>();
	thinking_policies.retain(|id, _| attached.contains(id));
	Ok((output, aliases))
}

fn collapsible_groups(classified: &BTreeMap<Str, ModelClassification>) -> BTreeSet<Str> {
	let raw: BTreeSet<&str> = classified.keys().map(Str::as_str).collect();
	let mut tiers: BTreeMap<&str, usize> = BTreeMap::new();
	let mut result = BTreeSet::new();
	for value in classified.values() {
		if value.thinking_variant && raw.contains(value.logical_model.as_str()) {
			result.insert(value.logical_model.clone());
		}
		if value.effort.is_some() {
			*tiers.entry(value.logical_model.as_str()).or_default() += 1;
		}
	}
	for (logical, count) in tiers {
		if count >= 2 {
			result.insert(Str::from(logical));
		}
	}
	result
}

fn compile_wire_policy(
	mut policy: WirePolicy,
	source: &SourceWirePolicy,
) -> Result<WirePolicy, CompileError> {
	policy.usage.in_streaming = source.usage_in_streaming.or(policy.usage.in_streaming);
	policy.role.multiple_system_messages = source
		.multiple_system_messages
		.or(policy.role.multiple_system_messages);
	policy.context.max_tokens_field =
		parse_policy(source.max_tokens_field.as_deref(), policy.context.max_tokens_field)?;
	policy.structured.sampling_params = source.sampling_params.or(policy.structured.sampling_params);
	policy.structured.penalties = source.penalties.or(policy.structured.penalties);
	policy.tool.strict_mode =
		parse_policy(source.tool_strict_mode.as_deref(), policy.tool.strict_mode)?;
	policy.tool.named_choice = source.named_tool_choice.or(policy.tool.named_choice);
	policy.tool.forced_choice = source.forced_tool_choice.or(policy.tool.forced_choice);
	policy.tool.id_profile = match source.tool_call_id_profile.as_deref() {
		Some("mistral9_alnum") => Some(ToolCallIdProfile::Mistral9Alnum),
		Some("open_ai40") => Some(ToolCallIdProfile::OpenAi40),
		value => parse_policy(value, policy.tool.id_profile)?,
	};
	policy.reasoning.wire_format =
		parse_policy(source.reasoning_wire_format.as_deref(), policy.reasoning.wire_format)?;
	policy.reasoning.interleaved_thinking = source
		.interleaved_thinking
		.or(policy.reasoning.interleaved_thinking);
	policy.context.stateful_response_chaining = source
		.stateful_response_chaining
		.or(policy.context.stateful_response_chaining);
	policy.tool.thinking_conflict =
		parse_policy(source.thinking_tool_choice_conflict.as_deref(), policy.tool.thinking_conflict)?;
	policy.cache.control_format =
		parse_policy(source.cache_control_format.as_deref(), policy.cache.control_format)?;
	policy.image.encoding =
		parse_policy(source.image_encoding_format.as_deref(), policy.image.encoding)?;
	policy.structured.stop_sequences = source.stop_sequences.or(policy.structured.stop_sequences);
	policy.tool.schema_flavor =
		parse_policy(source.tool_schema_flavor.as_deref(), policy.tool.schema_flavor)?;
	policy.reasoning.leaked_healer =
		parse_policy(source.leaked_thinking_healer.as_deref(), policy.reasoning.leaked_healer)?;
	policy.reasoning.loop_guard = source.thinking_loop_guard.or(policy.reasoning.loop_guard);
	if let Some(watchdog) = source.stream_watchdog {
		policy.streaming.watchdog = Some(StreamWatchdog {
			first_event_ms: watchdog.first_event_ms,
			idle_ms:        watchdog.idle_ms,
		});
	}
	policy.streaming.protocol =
		parse_policy(source.stream_protocol.as_deref(), policy.streaming.protocol)?;
	policy.audio.api_version =
		parse_policy(source.audio_api_version.as_deref(), policy.audio.api_version)?;
	policy.role.supports_developer_role = source
		.supports_developer_role
		.or(policy.role.supports_developer_role);
	policy.role.supports_mid_conversation_system = source
		.supports_mid_conversation_system
		.or(policy.role.supports_mid_conversation_system);
	policy.tool.supports_tool_choice = source
		.supports_tool_choice
		.or(policy.tool.supports_tool_choice);
	policy.tool.escape_builtin_names = source
		.escape_builtin_tool_names
		.or(policy.tool.escape_builtin_names);
	policy.tool.requires_result_id = source
		.requires_tool_result_id
		.or(policy.tool.requires_result_id);
	policy.tool.eager_input_streaming = source
		.supports_eager_tool_input_streaming
		.or(policy.tool.eager_input_streaming);
	policy.tool.requires_assistant_content = source
		.requires_assistant_content_for_tool_calls
		.or(policy.tool.requires_assistant_content);
	policy.tool.disable_reasoning_on_choice = source
		.disable_reasoning_on_tool_choice
		.or(policy.tool.disable_reasoning_on_choice);
	policy.structured.sampling_params = source
		.supports_sampling_params
		.or(policy.structured.sampling_params);
	policy.reasoning.supports_effort = source
		.supports_reasoning_effort
		.or(policy.reasoning.supports_effort);
	policy.reasoning.omit_effort = source
		.omit_reasoning_effort
		.or(policy.reasoning.omit_effort);
	if !source.reasoning_effort_map.is_empty() {
		policy
			.reasoning
			.effort_map
			.clone_from(&source.reasoning_effort_map);
	}
	policy.reasoning.disable_mode =
		parse_policy(source.reasoning_disable_mode.as_deref(), policy.reasoning.disable_mode)?;
	policy.reasoning.content_field = source
		.reasoning_content_field
		.clone()
		.or(policy.reasoning.content_field);
	policy.reasoning.requires_content_for_tool_calls = source
		.requires_reasoning_content_for_tool_calls
		.or(policy.reasoning.requires_content_for_tool_calls);
	policy.reasoning.requires_content_for_all_assistant_turns = source
		.requires_reasoning_content_for_all_assistant_turns
		.or(policy.reasoning.requires_content_for_all_assistant_turns);
	policy.reasoning.allows_synthetic_content_for_tool_calls = source
		.allows_synthetic_reasoning_content_for_tool_calls
		.or(policy.reasoning.allows_synthetic_content_for_tool_calls);
	policy.reasoning.filter_history = source
		.filter_reasoning_history
		.or(policy.reasoning.filter_history);
	policy.reasoning.include_encrypted = source
		.include_encrypted_reasoning
		.or(policy.reasoning.include_encrypted);
	policy.reasoning.replay_unsigned = source
		.replay_unsigned_thinking
		.or(policy.reasoning.replay_unsigned);
	policy.reasoning.requires_enabled = source
		.requires_thinking_enabled
		.or(policy.reasoning.requires_enabled);
	policy.reasoning.disable_adaptive = source
		.disable_adaptive_thinking
		.or(policy.reasoning.disable_adaptive);
	policy.reasoning.official_endpoint = source
		.official_endpoint
		.or(policy.reasoning.official_endpoint);
	policy.reasoning.signing_endpoint = source
		.signing_endpoint
		.or(policy.reasoning.signing_endpoint);
	policy.reasoning.thinking_format =
		parse_policy(source.thinking_format.as_deref(), policy.reasoning.thinking_format)?;
	if let Some(raw) = &source.extra_body {
		policy.reasoning.extra_body =
			Some(serde_json::from_str::<ReasoningBodyOverride>(raw.json())?);
	}
	if let Some(raw) = &source.when_thinking {
		policy.reasoning.when_thinking =
			Some(serde_json::from_str::<WhenThinkingPolicy>(raw.json())?);
	}
	policy.cache.supports_long_retention = source
		.supports_long_cache_retention
		.or(policy.cache.supports_long_retention);
	policy.context.supports_store = source.supports_store.or(policy.context.supports_store);
	policy.image.supports_detail_original = source
		.supports_image_detail_original
		.or(policy.image.supports_detail_original);
	if let Some(idle_ms) = source.stream_idle_timeout_ms {
		let mut watchdog = policy.streaming.watchdog.unwrap_or_default();
		watchdog.idle_ms = Some(idle_ms);
		policy.streaming.watchdog = Some(watchdog);
	}
	Ok(policy)
}

fn parse_policy<T>(source: Option<&str>, inherited: Option<T>) -> Result<Option<T>, CompileError>
where
	T: std::str::FromStr,
{
	source
		.map(|value| {
			value.parse().map_err(|_| {
				CompileError::Invariant(Str::from(format!("unknown policy value `{value}`")))
			})
		})
		.transpose()
		.map(|parsed| parsed.or(inherited))
}

fn compile_thinking(
	members: &[(Str, SourceModelRecord, ModelClassification)],
) -> Result<(Option<ThinkingPolicy>, ThinkingRouting), CompileError> {
	let source = members.iter().find_map(|(_, row, _)| row.thinking.as_ref());
	let mut classified_efforts: SmallVec<ThinkingEffort, 6> = members
		.iter()
		.filter_map(|(_, _, classified)| classified.effort.map(translate_effort))
		.collect();
	classified_efforts.sort();
	classified_efforts.dedup();
	let tier_collapsed = classified_efforts.len() >= 2;
	let profile = if let Some(source) = source {
		let mut efforts: SmallVec<ThinkingEffort, 6> = if tier_collapsed {
			classified_efforts.clone()
		} else {
			source
				.efforts
				.iter()
				.copied()
				.filter(|effort| *effort != ThinkingEffort::Off)
				.collect()
		};
		efforts.sort();
		efforts.dedup();
		let mut profile = ThinkingPolicy {
			mode: source.mode,
			efforts,
			default_level: source
				.default_level
				.filter(|effort| *effort != ThinkingEffort::Off),
			effort_budgets: source.effort_budgets.clone(),
			supports_display: source.supports_display,
			suppress_when_off: source.suppress_when_off,
			requires_effort: source.requires_effort,
		};
		profile.effort_budgets.remove(&ThinkingEffort::Off);
		profile.validate().map_err(|error| {
			CompileError::Invariant(Str::from(format!("invalid thinking profile: {error}")))
		})?;
		Some(profile)
	} else {
		None
	};
	let mut routing = ThinkingRouting::default();
	if !tier_collapsed && let Some(thinking) = source {
		routing.effort_map = thinking.effort_map.clone();
		routing.effort_routing = thinking
			.effort_routing
			.iter()
			.filter(|(effort, _)| {
				profile
					.as_ref()
					.is_some_and(|policy| policy.supports(**effort))
			})
			.map(|(effort, wire)| (*effort, WireModelId::new(wire.clone())))
			.collect();
		routing.reasoning_mode = thinking.reasoning_mode;
	}
	for (wire, row, classified) in members {
		if let Some(effort) = classified.effort.map(translate_effort)
			&& profile
				.as_ref()
				.is_some_and(|policy| policy.supports(effort))
		{
			let selected =
				WireModelId::new(row.request_model_id.clone().unwrap_or_else(|| wire.clone()));
			routing.effort_routing.entry(effort).or_insert(selected);
		}
	}
	if classified_efforts.is_empty()
		&& let Some(profile) = &profile
		&& let Some((thinking_wire, thinking_row, _)) = members
			.iter()
			.find(|(_, _, classified)| classified.thinking_variant)
		&& let Some((base_wire, base_row, _)) = members
			.iter()
			.find(|(_, _, classified)| !classified.thinking_variant && classified.effort.is_none())
	{
		let thinking_wire = WireModelId::new(
			thinking_row
				.request_model_id
				.clone()
				.unwrap_or_else(|| thinking_wire.clone()),
		);
		for effort in &profile.efforts {
			routing
				.effort_routing
				.insert(*effort, thinking_wire.clone());
		}
		routing.effort_routing.insert(
			ThinkingEffort::Off,
			WireModelId::new(
				base_row
					.request_model_id
					.clone()
					.unwrap_or_else(|| base_wire.clone()),
			),
		);
	}
	if let Some(profile) = &profile {
		routing.validate(profile).map_err(|error| {
			CompileError::Invariant(Str::from(format!("invalid thinking routing: {error}")))
		})?;
	} else if !routing.effort_map.is_empty() || !routing.effort_routing.is_empty() {
		return Err(CompileError::Invariant(Str::from(
			"thinking routing exists without a thinking profile",
		)));
	}
	Ok((profile, routing))
}

fn translate_effort(effort: crate::classify::EffortTier) -> ThinkingEffort {
	match effort {
		crate::classify::EffortTier::Off => ThinkingEffort::Off,
		crate::classify::EffortTier::Minimal => ThinkingEffort::Minimal,
		crate::classify::EffortTier::Low => ThinkingEffort::Low,
		crate::classify::EffortTier::Medium => ThinkingEffort::Medium,
		crate::classify::EffortTier::High => ThinkingEffort::High,
		crate::classify::EffortTier::XHigh => ThinkingEffort::XHigh,
		crate::classify::EffortTier::Max => ThinkingEffort::Max,
	}
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CapabilityCorrection {
	Embedding,
	Operationless,
}

#[derive(Clone, Copy, Debug)]
struct ExactCapabilityOverride {
	id:            &'static str,
	provider:      &'static str,
	model:         &'static str,
	correction:    CapabilityCorrection,
	rationale:     &'static str,
	provenance:    &'static str,
	expires_at_ms: Option<u64>,
}

const fn exact_capability(
	id: &'static str,
	provider: &'static str,
	model: &'static str,
	correction: CapabilityCorrection,
	rationale: &'static str,
) -> ExactCapabilityOverride {
	ExactCapabilityOverride {
		id,
		provider,
		model,
		correction,
		rationale,
		provenance: "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	}
}

const EXACT_CAPABILITY_OVERRIDES: &[ExactCapabilityOverride] = &[
	exact_capability(
		"aimlapi-voyage-2-embedding",
		"aimlapi",
		"voyage-2",
		CapabilityCorrection::Embedding,
		"The reviewed Voyage deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"aimlapi-voyage-code-2-embedding",
		"aimlapi",
		"voyage-code-2",
		CapabilityCorrection::Embedding,
		"The reviewed Voyage deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"aimlapi-voyage-finance-2-embedding",
		"aimlapi",
		"voyage-finance-2",
		CapabilityCorrection::Embedding,
		"The reviewed Voyage deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"aimlapi-voyage-large-2-embedding",
		"aimlapi",
		"voyage-large-2",
		CapabilityCorrection::Embedding,
		"The reviewed Voyage deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"aimlapi-voyage-large-2-instruct-embedding",
		"aimlapi",
		"voyage-large-2-instruct",
		CapabilityCorrection::Embedding,
		"The reviewed Voyage deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"aimlapi-voyage-law-2-embedding",
		"aimlapi",
		"voyage-law-2",
		CapabilityCorrection::Embedding,
		"The reviewed Voyage deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"aimlapi-voyage-multilingual-2-embedding",
		"aimlapi",
		"voyage-multilingual-2",
		CapabilityCorrection::Embedding,
		"The reviewed Voyage deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"fireworks-qwen3-embedding-8b",
		"fireworks",
		"qwen3-embedding-8b",
		CapabilityCorrection::Embedding,
		"The reviewed Fireworks deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"nvidia-baai-bge-m3-embedding",
		"nvidia",
		"baai/bge-m3",
		CapabilityCorrection::Embedding,
		"The reviewed NVIDIA deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"nvidia-embed-qa-4-embedding",
		"nvidia",
		"nvidia/embed-qa-4",
		CapabilityCorrection::Embedding,
		"The reviewed NVIDIA deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"nvidia-nemoretriever-vlm-embedding",
		"nvidia",
		"nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
		CapabilityCorrection::Embedding,
		"The reviewed NVIDIA deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"nvidia-nv-embedqa-1b-embedding",
		"nvidia",
		"nvidia/llama-3.2-nv-embedqa-1b-v1",
		CapabilityCorrection::Embedding,
		"The reviewed NVIDIA deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"nvidia-nemotron-embed-1b-v2",
		"nvidia",
		"nvidia/llama-nemotron-embed-1b-v2",
		CapabilityCorrection::Embedding,
		"The reviewed NVIDIA deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"nvidia-nemotron-embed-vl-1b-v2",
		"nvidia",
		"nvidia/llama-nemotron-embed-vl-1b-v2",
		CapabilityCorrection::Embedding,
		"The reviewed NVIDIA deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"nvidia-nv-embed-v1",
		"nvidia",
		"nvidia/nv-embed-v1",
		CapabilityCorrection::Embedding,
		"The reviewed NVIDIA deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"nvidia-nv-embedcode-7b-v1",
		"nvidia",
		"nvidia/nv-embedcode-7b-v1",
		CapabilityCorrection::Embedding,
		"The reviewed NVIDIA deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"nvidia-nv-embedqa-e5-v5",
		"nvidia",
		"nvidia/nv-embedqa-e5-v5",
		CapabilityCorrection::Embedding,
		"The reviewed NVIDIA deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"nvidia-nv-embedqa-mistral-7b-v2",
		"nvidia",
		"nvidia/nv-embedqa-mistral-7b-v2",
		CapabilityCorrection::Embedding,
		"The reviewed NVIDIA deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"nvidia-snowflake-arctic-embed-l",
		"nvidia",
		"snowflake/arctic-embed-l",
		CapabilityCorrection::Embedding,
		"The reviewed NVIDIA deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"zenmux-gemini-embedding-2",
		"zenmux",
		"google/gemini-embedding-2",
		CapabilityCorrection::Embedding,
		"The reviewed ZenMux deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"zenmux-text-embedding-3-large",
		"zenmux",
		"openai/text-embedding-3-large",
		CapabilityCorrection::Embedding,
		"The reviewed ZenMux deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"zenmux-text-embedding-3-small",
		"zenmux",
		"openai/text-embedding-3-small",
		CapabilityCorrection::Embedding,
		"The reviewed ZenMux deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"zenmux-qwen3-vl-embedding",
		"zenmux",
		"qwen/qwen3-vl-embedding",
		CapabilityCorrection::Embedding,
		"The reviewed ZenMux deployment is an embedding model despite absent dimensional metadata",
	),
	exact_capability(
		"fireworks-qwen3-reranker-8b-operationless",
		"fireworks",
		"qwen3-reranker-8b",
		CapabilityCorrection::Operationless,
		"The reviewed reranker deployment has no supported operation in the canonical catalog \
		 vocabulary",
	),
];

fn exact_capability_override(
	provider: &str,
	model: &str,
) -> Option<&'static ExactCapabilityOverride> {
	EXACT_CAPABILITY_OVERRIDES.iter().find(|override_| {
		override_.provider == provider
			&& override_.model == model
			&& override_.expires_at_ms.is_none()
			&& !override_.rationale.is_empty()
			&& !override_.provenance.is_empty()
	})
}

fn conservative_capabilities(
	row: &SourceModelRecord,
	facets: &[SourceFacet],
	correction: Option<CapabilityCorrection>,
) -> ModelCapabilities {
	let embedding =
		row.embedding_dimensions.is_some() || correction == Some(CapabilityCorrection::Embedding);
	let operationless = correction == Some(CapabilityCorrection::Operationless);
	let mut operations = if operationless {
		OperationBits::empty()
	} else if embedding {
		OperationBits::for_kind(OperationKind::Embed)
	} else if facets.contains(&SourceFacet::Chat) {
		OperationBits::for_kind(OperationKind::Chat)
	} else {
		facet_operations(facets)
	};
	if operations == OperationBits::empty() && !operationless {
		operations.insert_kind(OperationKind::Chat);
	}
	let chat = operations
		.contains_kind(OperationKind::Chat)
		.then(|| ChatCapabilities {
			roles:             Availability::Unknown,
			mid_session_roles: Availability::Unknown,
			structured_output: Availability::Unknown,
			grammar:           Availability::Unknown,
			text_verbosity:    Availability::Unknown,
			reasoning:         if row.reasoning {
				Availability::Unknown
			} else {
				Availability::Unsupported
			},
			input_modalities:  if row.input.is_empty() {
				Availability::Unknown
			} else {
				Availability::Native(modalities(&row.input))
			},
			tools:             match row.supports_tools {
				Some(true) => Availability::Native(ToolCapabilities {
					features:      ToolFeatureBits::empty(),
					maximum_tools: None,
				}),
				Some(false) => Availability::Unsupported,
				None => Availability::Unknown,
			},
			hosted_tools:      Availability::Unknown,
			prompt_caching:    Availability::Unknown,
			service_tiers:     Availability::Unknown,
			sampling:          Availability::Unknown,
			safety:            Availability::Unknown,
			determinism:       Availability::Unknown,
			server_state:      Availability::Unknown,
			logprobs:          Availability::Unknown,
		});
	let embeddings = operations
		.contains_kind(OperationKind::Embed)
		.then(|| EmbeddingCapabilities {
			input_modalities: if row.input.is_empty() {
				ModalityBits::TEXT
			} else {
				modalities(&row.input)
			},
			input_kinds:      if row.input.is_empty() || row.input.contains(&SourceModality::Text) {
				EmbeddingInputBits::TEXT
			} else {
				EmbeddingInputBits::empty()
			},
			formats:          EmbeddingFormatBits::FLOAT,
			maximum_batch:    None,
			dimensions:       row
				.embedding_dimensions
				.map_or(Availability::Unknown, |dimensions| {
					Availability::Native(DimensionRange { minimum: dimensions, maximum: dimensions })
				}),
		});
	let image = operations
		.contains_kind(OperationKind::GenerateImage)
		.then_some(ImageCapabilities {
			features:         ImageFeatureBits::GENERATE,
			input_modalities: modalities(&row.input),
			maximum_outputs:  None,
			maximum_pixels:   None,
		});
	let video = operations
		.contains_kind(OperationKind::GenerateVideo)
		.then_some(VideoCapabilities {
			features:             VideoFeatureBits::GENERATE,
			maximum_duration_ms:  None,
			maximum_frame_pixels: None,
		});
	let speech = operations
		.contains_kind(OperationKind::Speak)
		.then_some(SpeechCapabilities {
			features:                 SpeechFeatureBits::empty(),
			maximum_input_characters: None,
			output_formats:           AudioFormatBits::empty(),
		});
	let transcription = operations
		.contains_kind(OperationKind::Transcribe)
		.then_some(TranscriptionCapabilities {
			features:            TranscriptionFeatureBits::empty(),
			input_formats:       AudioFormatBits::empty(),
			maximum_duration_ms: None,
		});
	let realtime =
		operations
			.contains_kind(OperationKind::Realtime)
			.then_some(RealtimeCapabilities {
				features:           RealtimeFeatureBits::empty(),
				maximum_session_ms: None,
				audio_formats:      AudioFormatBits::empty(),
			});
	let search = operations
		.contains_kind(OperationKind::Search)
		.then_some(SearchCapabilities {
			features:        SearchFeatureBits::empty(),
			maximum_results: None,
		});
	let tokenization = (operations.contains_kind(OperationKind::CountTokens)
		|| operations.contains_kind(OperationKind::Tokenize)
		|| operations.contains_kind(OperationKind::Detokenize))
	.then_some(TokenizationCapabilities {
		features:            TokenizationFeatureBits::COUNT
			| TokenizationFeatureBits::TOKENIZE
			| TokenizationFeatureBits::DETOKENIZE,
		maximum_input_bytes: None,
	});
	ModelCapabilities {
		operations,
		chat,
		embeddings,
		image,
		video,
		speech,
		transcription,
		realtime,
		search,
		tokenization,
	}
}

fn modalities(values: &[SourceModality]) -> ModalityBits {
	values.iter().fold(ModalityBits::empty(), |bits, value| {
		bits
			| match value {
				SourceModality::Text => ModalityBits::TEXT,
				SourceModality::Image => ModalityBits::IMAGE,
				SourceModality::Audio => ModalityBits::AUDIO,
				SourceModality::Video => ModalityBits::VIDEO,
				SourceModality::Pdf => ModalityBits::DOCUMENT,
			}
	})
}

#[derive(Clone, Copy)]
struct CompleteZeroPricingPolicy {
	provider:      &'static str,
	model:         &'static str,
	rationale:     &'static str,
	provenance:    &'static str,
	expires_at_ms: Option<u64>,
}

const COMPLETE_ZERO_PRICING_POLICIES: &[CompleteZeroPricingPolicy] = &[
	CompleteZeroPricingPolicy {
		provider:      "openrouter",
		model:         "openrouter/auto",
		rationale:     "The reviewed automatic selector explicitly advertises a complete zero-price \
		                schedule.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	CompleteZeroPricingPolicy {
		provider:      "openrouter",
		model:         "openrouter/auto-beta",
		rationale:     "The reviewed beta automatic selector explicitly advertises a complete \
		                zero-price schedule.",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
];

fn compile_pricing(
	provider: &str,
	model: &str,
	cost: &SourceCost,
) -> Result<Pricing, CompileError> {
	let mut components =
		price_components(&cost.input, &cost.output, &cost.cache_read, &cost.cache_write)?;
	if let Some(policy) = COMPLETE_ZERO_PRICING_POLICIES
		.iter()
		.find(|policy| policy.provider == provider && policy.model == model)
	{
		let _review_evidence = (policy.rationale, policy.provenance, policy.expires_at_ms);
		for unit in [
			PriceUnit::MtokInput,
			PriceUnit::MtokOutput,
			PriceUnit::MtokCacheRead,
			PriceUnit::MtokCacheWrite,
		] {
			if components.iter().all(|component| component.unit != unit) {
				components.push(Price { unit, nanos_usd: 0 });
			}
		}
	}
	let mut tiers = Vec::new();
	if let Some(tier) = &cost.long_context {
		tiers.push(PriceTier {
			prompt_tokens_above: tier.input_threshold,
			components:          price_components(
				&tier.input,
				&tier.output,
				&tier.cache_read,
				&tier.cache_write,
			)?
			.into_boxed_slice(),
		});
	}
	Pricing::new(components, tiers).map_err(|error| {
		CompileError::Invariant(Str::from(format!("invalid pricing schedule: {error}")))
	})
}

fn price_components(
	input: &Number,
	output: &Number,
	cache_read: &Number,
	cache_write: &Number,
) -> Result<Vec<Price>, CompileError> {
	[
		(PriceUnit::MtokInput, input),
		(PriceUnit::MtokOutput, output),
		(PriceUnit::MtokCacheRead, cache_read),
		(PriceUnit::MtokCacheWrite, cache_write),
	]
	.into_iter()
	.filter(|(_, number)| number.to_string() != "-1000000")
	.map(|(unit, number)| Ok(Price { unit, nanos_usd: decimal_nanos(number)? }))
	.collect()
}

fn source_price_present(number: &Number) -> bool {
	number.as_u64() != Some(0) && number.to_string() != "-1000000"
}

fn decimal_nanos(number: &Number) -> Result<u64, CompileError> {
	decimal_scaled(number, 9)
}
fn decimal_millionths(number: &Number) -> Result<u64, CompileError> {
	decimal_scaled(number, 6)
}
fn decimal_scaled(number: &Number, scale: usize) -> Result<u64, CompileError> {
	let text = number.to_string();
	if text.starts_with('-') {
		return Err(CompileError::Invariant(Str::from(format!("negative decimal `{text}`"))));
	}
	let (mantissa, exponent) = text
		.split_once(['e', 'E'])
		.map_or((text.as_str(), 0_i32), |(mantissa, exponent)| {
			(mantissa, exponent.parse::<i32>().unwrap_or(i32::MIN))
		});
	if exponent == i32::MIN {
		return Err(CompileError::Invariant(Str::from(format!("invalid decimal `{text}`"))));
	}
	let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
	let digits = format!("{whole}{fraction}");
	let coefficient: u128 = digits
		.parse()
		.map_err(|_| CompileError::Invariant(Str::from("decimal is out of range")))?;
	let shift = exponent + i32::try_from(scale).expect("small fixed decimal scale")
		- i32::try_from(fraction.len())
			.map_err(|_| CompileError::Invariant(Str::from("decimal is out of range")))?;
	let scaled = if shift >= 0 {
		coefficient
			.checked_mul(
				10_u128
					.checked_pow(shift as u32)
					.ok_or_else(|| CompileError::Invariant(Str::from("decimal is out of range")))?,
			)
			.ok_or_else(|| CompileError::Invariant(Str::from("decimal is out of range")))?
	} else {
		let divisor = 10_u128
			.checked_pow((-shift) as u32)
			.ok_or_else(|| CompileError::Invariant(Str::from("decimal is out of range")))?;
		let quotient = coefficient / divisor;
		let remainder = coefficient % divisor;
		quotient
			.checked_add(u128::from(remainder >= (divisor + 1) / 2))
			.ok_or_else(|| CompileError::Invariant(Str::from("decimal is out of range")))?
	};
	u64::try_from(scaled).map_err(|_| CompileError::Invariant(Str::from("decimal is out of range")))
}
fn zero_number() -> Number {
	Number::from(0)
}

impl Default for SourceCost {
	fn default() -> Self {
		Self {
			input:        zero_number(),
			output:       zero_number(),
			cache_read:   zero_number(),
			cache_write:  zero_number(),
			long_context: None,
		}
	}
}

fn compile_auth(
	source: &SourceAuth,
	oauth_ids: &BTreeMap<Str, OAuthSpecId>,
) -> Result<AuthSpec, CompileError> {
	let canonical = serde_json::to_vec(source)?;
	let id = AuthSpecId::new(content_id("auth", &canonical));
	let mut credential_sources = Vec::new();
	let (kind, header_name, query_parameter, prefix, sealed_body, account_scope, oauth, signing) =
		match source {
			SourceAuth::None => {
				(AuthSpecKind::None, None, None, None, None, AccountScope::Provider, None, None)
			},
			SourceAuth::Bearer { env } | SourceAuth::OptionalBearer { env } => {
				if !env.is_empty() {
					credential_sources.push(CredentialSourceSpec::Environment {
						ordered_names: canonical_env_names(env)?,
					});
				}
				credential_sources.push(CredentialSourceSpec::Stored);
				(
					AuthSpecKind::Bearer,
					Some(Str::from("authorization")),
					None,
					Some(Str::from("Bearer ")),
					None,
					AccountScope::Provider,
					None,
					None,
				)
			},
			SourceAuth::DevinSession { env } => {
				if !env.is_empty() {
					credential_sources.push(CredentialSourceSpec::Environment {
						ordered_names: canonical_env_names(env)?,
					});
				}
				credential_sources.push(CredentialSourceSpec::Session);
				(
					AuthSpecKind::OmpSession,
					None,
					None,
					None,
					Some(SealedBodyPlacement::DevinMetadata),
					AccountScope::Provider,
					None,
					None,
				)
			},
			SourceAuth::Header { name, env } => {
				validate_header_name(name)?;
				if !env.is_empty() {
					credential_sources.push(CredentialSourceSpec::Environment {
						ordered_names: canonical_env_names(env)?,
					});
				}
				credential_sources.push(CredentialSourceSpec::Stored);
				(
					AuthSpecKind::ApiKey,
					Some(name.clone()),
					None,
					None,
					None,
					AccountScope::Provider,
					None,
					None,
				)
			},
			SourceAuth::Query { param, env } => {
				if !env.is_empty() {
					credential_sources.push(CredentialSourceSpec::Environment {
						ordered_names: canonical_env_names(env)?,
					});
				}
				credential_sources.push(CredentialSourceSpec::Stored);
				(
					AuthSpecKind::ApiKey,
					None,
					Some(param.clone()),
					None,
					None,
					AccountScope::Provider,
					None,
					None,
				)
			},
			SourceAuth::AwsSigV4 => {
				credential_sources.push(CredentialSourceSpec::AwsChain);
				(
					AuthSpecKind::AwsSigv4,
					None,
					None,
					None,
					None,
					AccountScope::Region,
					None,
					Some(SigV4Spec {
						service: Str::from("bedrock"),
						region:  RegionSource::RouteEndpoint,
					}),
				)
			},
			SourceAuth::GoogleAdc { api_key_env, project_env, location_env } => {
				let api_key_env = canonical_env_names(api_key_env)?;
				let project_env = canonical_env_names(project_env)?;
				let location_env = canonical_env_names(location_env)?;
				let mut sources = api_key_env
					.iter()
					.cloned()
					.map(|variable| ApplicationDefaultSource::EnvironmentAccessToken { variable })
					.collect::<Vec<_>>();
				sources.push(ApplicationDefaultSource::CredentialFile {
					path_environment: Some(Str::from("OMP_GOOGLE_APPLICATION_CREDENTIALS")),
					default_path:     None,
				});
				sources.push(ApplicationDefaultSource::Metadata { url: Str::from("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"), headers: Box::new([StaticHeader { name: Str::from("metadata-flavor"), value: Str::from("Google") }]) });
				credential_sources.push(CredentialSourceSpec::ApplicationDefault {
					api_key_env,
					project_env,
					location_env,
					sources: sources.into_boxed_slice(),
				});
				(
					AuthSpecKind::GcpAdc,
					Some(Str::from("authorization")),
					None,
					Some(Str::from("Bearer ")),
					None,
					AccountScope::Provider,
					None,
					None,
				)
			},
			SourceAuth::Oauth { flow } => {
				let oauth = oauth_ids.get(flow).cloned().ok_or_else(|| {
					CompileError::Invariant(Str::from(format!("unknown OAuth flow `{flow}`")))
				})?;
				credential_sources.push(CredentialSourceSpec::Oauth { flow: oauth.clone() });
				credential_sources.push(CredentialSourceSpec::Stored);
				(
					AuthSpecKind::Oauth,
					Some(Str::from("authorization")),
					None,
					Some(Str::from("Bearer ")),
					None,
					AccountScope::Provider,
					Some(oauth),
					None,
				)
			},
		};
	Ok(AuthSpec {
		id,
		kind,
		header_name,
		query_parameter,
		prefix,
		sealed_body,
		scopes: Box::new([]),
		audience: None,
		account_scope,
		credential_sources: credential_sources.into_boxed_slice(),
		oauth,
		signing,
	})
}

fn canonical_env_names(names: &[Str]) -> Result<Box<[Str]>, CompileError> {
	for name in names {
		if !name.starts_with("OMP_") {
			return Err(CompileError::Invariant(Str::from(format!(
				"credential environment variable `{name}` must use the OMP_ prefix"
			))));
		}
	}
	Ok(names.to_vec().into_boxed_slice())
}

fn compile_headers(headers: &BTreeMap<Str, Str>) -> Result<HeaderProfile, CompileError> {
	let entries = headers
		.iter()
		.map(|(name, value)| StaticHeader { name: name.clone(), value: value.clone() });
	HeaderProfile::try_new(entries).map_err(|error| {
		CompileError::Invariant(Str::from(format!("invalid static header profile: {error}")))
	})
}

fn compile_discovery(source: &SourceDiscovery) -> Result<DiscoverySpec, CompileError> {
	let kind = match source.kind.as_str() {
		"open-ai-models" => DiscoveryKind::OpenAiModels,
		"google-models" => DiscoveryKind::GoogleModels,
		"ollama-tags" => DiscoveryKind::OllamaTags,
		"account-models" => DiscoveryKind::AccountModels,
		"specialized" => DiscoveryKind::Specialized,
		other => {
			return Err(CompileError::Invariant(Str::from(format!(
				"unknown discovery kind `{other}`"
			))));
		},
	};
	let canonical = serde_json::to_vec(source)?;
	Ok(DiscoverySpec {
		id: DiscoverySpecId::new(content_id("discovery", &canonical)),
		kind,
		label: source.label.clone(),
		path: Str::from("/models"),
		pagination: DiscoveryPagination::SinglePage,
		authoritative: source.authoritative,
	})
}

fn translate_transport(source: SourceTransport) -> (CodecId, TransportKind) {
	let (codec, transport) = match source {
		SourceTransport::AnthropicMessages => ("anthropic", TransportKind::Http),
		SourceTransport::AnthropicBedrock => ("anthropic-bedrock", TransportKind::AwsEventStream),
		SourceTransport::BedrockConverse => ("bedrock-converse", TransportKind::AwsEventStream),
		SourceTransport::AnthropicVertex => ("anthropic-vertex", TransportKind::Http),
		SourceTransport::OpenAiChat => ("openai-chat", TransportKind::Http),
		SourceTransport::OpenAiResponses => ("openai-responses", TransportKind::Http),
		SourceTransport::OpenAiCodex => ("openai-codex", TransportKind::Http),
		SourceTransport::GoogleGenAi => ("google-genai", TransportKind::Http),
		SourceTransport::GoogleVertex => ("google-vertex", TransportKind::Http),
		SourceTransport::GoogleCca => ("google-cca", TransportKind::Http),
		SourceTransport::OllamaChat => ("ollama", TransportKind::Http),
		SourceTransport::Cursor => ("cursor", TransportKind::Connect),
		SourceTransport::Devin => ("devin", TransportKind::Connect),
		SourceTransport::GitlabDuoWorkflow => ("gitlab-duo", TransportKind::Websocket),
		SourceTransport::Omp => ("omp", TransportKind::Http),
		SourceTransport::Embedded => ("local", TransportKind::Local),
	};
	(CodecId::new(codec), transport)
}

fn validate_header_name(name: &str) -> Result<(), CompileError> {
	if name.is_empty()
		|| !name.bytes().all(|byte| {
			byte.is_ascii_alphanumeric()
				|| matches!(
					byte,
					b'!'
						| b'#' | b'$'
						| b'%' | b'&'
						| b'\'' | b'*'
						| b'+' | b'-'
						| b'.' | b'^'
						| b'_' | b'`'
						| b'|' | b'~'
				)
		}) {
		return Err(CompileError::Invariant(Str::from(format!("invalid header name `{name}`"))));
	}
	Ok(())
}
fn validate_url(url: &str) -> Result<(), CompileError> {
	if url.starts_with("https://")
		|| url.starts_with("http://127.0.0.1")
		|| url.starts_with("http://localhost")
		|| url.starts_with("local://")
	{
		Ok(())
	} else {
		Err(CompileError::Invariant(Str::from(format!("untrusted route URL `{url}`"))))
	}
}
fn humanize(value: &str) -> Str {
	Str::from(
		value
			.split(['-', '_'])
			.filter(|part| !part.is_empty())
			.map(|part| {
				let mut chars = part.chars();
				chars
					.next()
					.map_or_else(String::new, |first| first.to_uppercase().chain(chars).collect())
			})
			.collect::<Vec<_>>()
			.join(" "),
	)
}
fn content_id(prefix: &str, bytes: &[u8]) -> String {
	let digest: [u8; 32] = Sha256::digest(bytes).into();
	format!("{prefix}-{}", hex::encode_n(&digest))
}
fn revision_for(
	providers: &[ProviderDef],
	routes: &[RouteDef],
	models: &[ModelSpec],
) -> Result<CatalogRevision, CompileError> {
	let bytes = serde_json::to_vec(&(providers, routes, models))?;
	Ok(CatalogRevision::new(content_id("catalog", &bytes)))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn rejects_header_injection_and_credentials() {
		assert!(
			compile_headers(&BTreeMap::from([(Str::from("x-ok"), Str::from("a\r\nb"))])).is_err()
		);
		assert!(
			compile_headers(&BTreeMap::from([(Str::from("authorization"), Str::from("secret"))]))
				.is_err()
		);
	}

	#[test]
	fn canonical_decimal_conversion_is_exact() {
		assert_eq!(
			decimal_nanos(&Number::from_f64(1.25).expect("finite")).expect("exact decimal"),
			1_250_000_000
		);
		assert_eq!(
			decimal_nanos(&Number::from_f64(0.000_000_000_1).expect("finite"))
				.expect("deterministically rounded"),
			0
		);
	}

	#[test]
	fn effort_collapse_requires_siblings() {
		let single = BTreeMap::from([(
			Str::from("model-low"),
			classify(ClassificationInput {
				phase:          ClassificationPhase::CatalogCompiler,
				provider:       "p",
				model:          "model-low",
				observed_at_ms: None,
			}),
		)]);
		assert!(collapsible_groups(&single).is_empty());
		let siblings = BTreeMap::from([
			(
				Str::from("model-low"),
				classify(ClassificationInput {
					phase:          ClassificationPhase::CatalogCompiler,
					provider:       "p",
					model:          "model-low",
					observed_at_ms: None,
				}),
			),
			(
				Str::from("model-high"),
				classify(ClassificationInput {
					phase:          ClassificationPhase::CatalogCompiler,
					provider:       "p",
					model:          "model-high",
					observed_at_ms: None,
				}),
			),
		]);
		assert!(collapsible_groups(&siblings).contains("model"));
	}

	#[test]
	fn authored_operation_facets_compile_end_to_end() {
		let providers = r#"
[providers.synthetic]
transport = "open-ai-chat"
base_url = "https://example.test/v1"
facets = ["audio_speech", "audio_transcription", "realtime", "web_search"]
discovery = { kind = "open-ai-models", label = "Synthetic", authoritative = true }
pending_facets = ["image_generation"]
"#;
		let models = br#"{"synthetic":{"model":{"input":["text"],"output":["text"]}}}"#;
		let compressed = zstd::stream::encode_all(&models[..], 1).expect("fixture compression");
		let compiled = compile(parse_oracle(providers, &compressed).expect("typed source"))
			.expect("catalog compilation");
		let model = compiled
			.models
			.iter()
			.find(|model| model.key.as_str() == "synthetic/model")
			.expect("compiled model");
		for operation in [
			OperationKind::Speak,
			OperationKind::Transcribe,
			OperationKind::Realtime,
			OperationKind::Search,
		] {
			assert!(model.capabilities.operations.contains_kind(operation), "{operation}");
		}
		assert!(
			!model
				.capabilities
				.operations
				.contains_kind(OperationKind::GenerateImage)
		);
		assert!(model.capabilities.speech.is_some());
		assert!(model.capabilities.transcription.is_some());
		assert!(model.capabilities.realtime.is_some());
		assert!(model.capabilities.search.is_some());
		let route = compiled
			.routes
			.iter()
			.find(|route| route.provider.as_str() == "synthetic")
			.expect("compiled route");
		let route_operations = route
			.capability_limits
			.operations
			.expect("authored route operations");
		assert!(route_operations.contains(model.capabilities.operations));
		assert!(route_operations.contains_kind(OperationKind::DiscoverModels));
	}
	#[test]
	fn exact_capability_overrides_are_auditable_and_unique() {
		let mut ids = BTreeSet::new();
		let mut identities = BTreeSet::new();
		for override_ in EXACT_CAPABILITY_OVERRIDES {
			assert!(ids.insert(override_.id), "duplicate override ID {}", override_.id);
			assert!(
				identities.insert((override_.provider, override_.model)),
				"duplicate exact capability override {}/{}",
				override_.provider,
				override_.model
			);
			assert!(!override_.rationale.is_empty());
			assert!(!override_.provenance.is_empty());
		}
		assert_eq!(
			exact_capability_override("aimlapi", "voyage-2").map(|override_| override_.correction),
			Some(CapabilityCorrection::Embedding)
		);
		assert_eq!(
			exact_capability_override("fireworks", "qwen3-reranker-8b")
				.map(|override_| override_.correction),
			Some(CapabilityCorrection::Operationless)
		);
		assert!(exact_capability_override("aimlapi", "voyage-2-preview").is_none());
	}
}
