//! Typed OpenAI Responses wire shapes and sans-I/O event projection.

use std::collections::{BTreeMap, BTreeSet};

use bytes::{Bytes, BytesMut};
use omp_core::Str;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
	answer::{Artifact, ArtifactBody},
	call::OpaqueJson,
	event::{BlockKind, ChatEvent, FinishReason, ToolCall},
	id::ToolCallId,
	receipt::{Usage, UsageSource},
};

/// A typed metadata value accepted by the Responses API.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesMetadataValue {
	/// JSON null.
	Null,
	/// Boolean metadata.
	Bool(bool),
	/// Signed integer metadata.
	Integer(i64),
	/// Floating-point metadata.
	Number(f64),
	/// String metadata.
	String(Str),
	/// Array metadata.
	Array(Vec<Self>),
	/// Object metadata.
	Object(BTreeMap<Str, Self>),
}

/// Metadata object carried without interpreting application-owned keys.
pub type ResponsesMetadata = BTreeMap<Str, ResponsesMetadataValue>;

/// Responses message role.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesRole {
	/// System instruction.
	System,
	/// Developer instruction.
	Developer,
	/// User input.
	User,
	/// Assistant output.
	Assistant,
}

/// Input-item discriminator.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesInputItemKind {
	/// Message item.
	Message,
	/// Function call.
	FunctionCall,
	/// Function result.
	FunctionCallOutput,
	/// Freeform custom-tool call.
	CustomToolCall,
	/// Freeform custom-tool result.
	CustomToolCallOutput,
	/// Computer-use call.
	ComputerCall,
	/// Computer-use result.
	ComputerCallOutput,
	/// Provider reasoning item.
	Reasoning,
	/// Provider item reference.
	ItemReference,
	/// Responses Lite additional-tool declaration.
	AdditionalTools,
}

/// Input-content discriminator.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesContentKind {
	/// Text input.
	InputText,
	/// Image input.
	InputImage,
	/// File input.
	InputFile,
	/// Text output replay.
	OutputText,
	/// Refusal replay.
	Refusal,
}

/// Image quality selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesImageDetail {
	/// Let the provider choose.
	Auto,
	/// Low-resolution input.
	Low,
	/// High-resolution input.
	High,
	/// Preserve original image resolution where the route supports it.
	Original,
}

/// One typed message content entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesContent {
	/// Content discriminator.
	#[serde(rename = "type")]
	pub kind:      ResponsesContentKind,
	/// Text or refusal content.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub text:      Option<Str>,
	/// Data or remote image URL.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub image_url: Option<Str>,
	/// Image detail, omitted to preserve the server default.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub detail:    Option<ResponsesImageDetail>,
	/// Data URL carrying an inline file.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub file_data: Option<Str>,
	/// Remote file URL.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub file_url:  Option<Str>,
	/// Original file name.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub filename:  Option<Str>,
	/// Provider file identity.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub file_id:   Option<Str>,
}

impl ResponsesContent {
	/// Constructs a text input part.
	pub fn input_text(text: impl Into<Str>) -> Self {
		Self {
			kind:      ResponsesContentKind::InputText,
			text:      Some(text.into()),
			image_url: None,
			detail:    None,
			file_data: None,
			file_url:  None,
			filename:  None,
			file_id:   None,
		}
	}
}

/// Message input content, preserving the API's string and typed-part shapes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInputContent {
	/// Compact plain-text content.
	Text(Str),
	/// Typed multimodal content parts.
	Parts(Vec<ResponsesContent>),
}

impl Default for ResponsesInputContent {
	fn default() -> Self {
		Self::Parts(Vec::new())
	}
}

impl ResponsesInputContent {
	/// Returns whether no visible content is present.
	pub fn is_empty(&self) -> bool {
		match self {
			Self::Text(text) => text.is_empty(),
			Self::Parts(parts) => parts.is_empty(),
		}
	}

	/// Mutably visits typed parts; compact text has no part list.
	pub fn parts_mut(&mut self) -> Option<&mut [ResponsesContent]> {
		match self {
			Self::Text(_) => None,
			Self::Parts(parts) => Some(parts),
		}
	}
}

/// Prompt-cache marker attached to an individual input item.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesCacheControl {
	/// Cache-control kind, normally `ephemeral`.
	#[serde(rename = "type")]
	pub kind: Str,
}

/// Reasoning summary entry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesSummaryPart {
	/// Summary entry kind.
	#[serde(rename = "type")]
	pub kind: Str,
	/// Summary text.
	pub text: Str,
}

/// A computer-use action.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesComputerAction {
	/// Action discriminator such as `click`, `type`, or `screenshot`.
	#[serde(rename = "type")]
	pub kind:     Str,
	/// Horizontal coordinate when applicable.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub x:        Option<i64>,
	/// Vertical coordinate when applicable.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub y:        Option<i64>,
	/// Text entered by a typing action.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub text:     Option<Str>,
	/// Keyboard key names.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub keys:     Vec<Str>,
	/// Mouse button name.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub button:   Option<Str>,
	/// Scroll distance on the x axis.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub scroll_x: Option<i64>,
	/// Scroll distance on the y axis.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub scroll_y: Option<i64>,
}

/// One pending or acknowledged computer safety check.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesSafetyCheck {
	/// Stable check identity.
	pub id:      Str,
	/// Provider safety code.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub code:    Option<Str>,
	/// Human-readable check message.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub message: Option<Str>,
}

/// Canonical computer-call arguments assembled for validation and execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesComputerArguments {
	/// Ordered computer actions.
	pub actions:               Vec<ResponsesComputerAction>,
	/// Safety checks that must be acknowledged before execution.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub pending_safety_checks: Vec<ResponsesSafetyCheck>,
}

/// Computer screenshot result.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesComputerScreenshot {
	/// Output discriminator.
	#[serde(rename = "type")]
	pub kind:      Str,
	/// Data or remote image URL.
	pub image_url: Str,
}

/// Function/custom text output or typed computer screenshot output.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesToolOutput {
	/// Textual tool output.
	Text(Str),
	/// Computer screenshot output.
	Computer(ResponsesComputerScreenshot),
}

/// One typed Responses input item.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesInputItem {
	/// Optional item discriminator; ordinary input messages deliberately omit
	/// it.
	#[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
	pub kind: Option<ResponsesInputItemKind>,
	/// Provider item identity used only for native replay.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub id: Option<Str>,
	/// Message role.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub role: Option<ResponsesRole>,
	/// Message content.
	#[serde(default, skip_serializing_if = "ResponsesInputContent::is_empty")]
	pub content: ResponsesInputContent,
	/// Function or custom-tool name.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub name: Option<Str>,
	/// Stable call identity.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub call_id: Option<Str>,
	/// Function-call JSON arguments.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub arguments: Option<Str>,
	/// Freeform custom-tool input.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub input: Option<Str>,
	/// Tool output.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub output: Option<ResponsesToolOutput>,
	/// Reasoning summaries.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub summary: Vec<ResponsesSummaryPart>,
	/// Encrypted reasoning continuation payload.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub encrypted_content: Option<Str>,
	/// Computer actions.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub actions: Vec<ResponsesComputerAction>,
	/// Pending safety checks for a computer call.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub pending_safety_checks: Vec<ResponsesSafetyCheck>,
	/// Acknowledged safety checks for a computer result.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub acknowledged_safety_checks: Vec<ResponsesSafetyCheck>,
	/// Provider item status.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub status: Option<Str>,
	/// Responses Lite additional tools.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub tools: Vec<ResponsesTool>,
	/// Per-item cache control.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cache_control: Option<ResponsesCacheControl>,
	/// Per-item metadata.
	#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
	pub metadata: ResponsesMetadata,
}

impl ResponsesInputItem {
	/// Constructs an ordinary message input with the wire `type` intentionally
	/// omitted.
	pub fn message(role: ResponsesRole, content: Vec<ResponsesContent>) -> Self {
		Self {
			kind: None,
			id: None,
			role: Some(role),
			content: ResponsesInputContent::Parts(content),
			name: None,
			call_id: None,
			arguments: None,
			input: None,
			output: None,
			summary: Vec::new(),
			encrypted_content: None,
			actions: Vec::new(),
			pending_safety_checks: Vec::new(),
			acknowledged_safety_checks: Vec::new(),
			status: None,
			tools: Vec::new(),
			cache_control: None,
			metadata: BTreeMap::new(),
		}
	}
}

/// Responses tool discriminator.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesToolKind {
	/// JSON-schema function tool.
	Function,
	/// Freeform custom tool.
	Custom,
	/// Computer-use tool.
	Computer,
	/// Hosted web search.
	WebSearch,
	/// Hosted file search.
	FileSearch,
	/// Hosted code interpreter.
	CodeInterpreter,
	/// Hosted image generation.
	ImageGeneration,
	/// Hosted MCP server.
	Mcp,
}

/// Custom tool input grammar.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesCustomToolFormat {
	/// Format kind, for example `text` or `grammar`.
	#[serde(rename = "type")]
	pub kind:       Str,
	/// Grammar syntax when the kind is `grammar`.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub syntax:     Option<Str>,
	/// Grammar definition.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub definition: Option<Str>,
}

/// Hosted code-interpreter container selection.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesCodeContainer {
	/// Container selection kind.
	#[serde(rename = "type")]
	pub kind: Str,
}

/// A typed Responses tool declaration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesTool {
	/// Tool discriminator.
	#[serde(rename = "type")]
	pub kind:                ResponsesToolKind,
	/// Tool name for function and custom tools.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub name:                Option<Str>,
	/// Tool description.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub description:         Option<Str>,
	/// Opaque function JSON Schema.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub parameters:          Option<Value>,
	/// Strict JSON-schema enforcement.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub strict:              Option<bool>,
	/// Freeform input format.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub format:              Option<ResponsesCustomToolFormat>,
	/// Computer viewport width.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub display_width:       Option<u32>,
	/// Computer viewport height.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub display_height:      Option<u32>,
	/// Computer environment label.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub environment:         Option<Str>,
	/// Hosted search context size.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub search_context_size: Option<Str>,
	/// Allowed search domains.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub allowed_domains:     Vec<Str>,
	/// Blocked search domains.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub blocked_domains:     Vec<Str>,
	/// File-search vector store identities.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub vector_store_ids:    Vec<Str>,
	/// Code-interpreter container policy.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub container:           Option<ResponsesCodeContainer>,
}

/// Tool-choice object type.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesNamedToolKind {
	/// Function tool.
	Function,
	/// Custom tool.
	Custom,
	/// Computer tool.
	Computer,
	/// Web-search tool.
	WebSearch,
	/// File-search tool.
	FileSearch,
	/// Code-interpreter tool.
	CodeInterpreter,
}

/// A named Responses tool choice.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesNamedToolChoice {
	/// Selected tool kind.
	#[serde(rename = "type")]
	pub kind: ResponsesNamedToolKind,
	/// Selected caller tool name.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub name: Option<Str>,
}

/// Responses tool-choice value.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
	/// `none`, `auto`, or `required`.
	Mode(Str),
	/// Named or hosted tool selection.
	Named(ResponsesNamedToolChoice),
}

/// Responses reasoning effort.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesReasoningEffort {
	/// Disable reasoning.
	None,
	/// Minimal reasoning.
	Minimal,
	/// Low reasoning.
	Low,
	/// Medium reasoning.
	Medium,
	/// High reasoning.
	High,
	/// Extra-high reasoning.
	Xhigh,
}

/// Responses reasoning controls.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesReasoning {
	/// Effort selection.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub effort:  Option<ResponsesReasoningEffort>,
	/// Summary selection (`auto`, `concise`, or `detailed`); explicit null
	/// suppresses summaries.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub summary: Option<Option<Str>>,
	/// Provider reasoning mode.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub mode:    Option<Str>,
	/// Codex Responses Lite reasoning context.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub context: Option<Str>,
}

/// Responses text format kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesTextFormatKind {
	/// Plain text.
	Text,
	/// JSON object.
	JsonObject,
	/// JSON Schema.
	JsonSchema,
}

/// Structured text output configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesTextFormat {
	/// Format discriminator.
	#[serde(rename = "type")]
	pub kind:   ResponsesTextFormatKind,
	/// Schema name.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub name:   Option<Str>,
	/// Opaque output JSON Schema.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub schema: Option<Value>,
	/// Strict conformance.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub strict: Option<bool>,
}

/// Responses text controls.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesTextOptions {
	/// Output verbosity.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub verbosity: Option<Str>,
	/// Structured output format.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub format:    Option<ResponsesTextFormat>,
}

/// Codex stream controls.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesStreamOptions {
	/// Reasoning-summary delivery strategy.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub reasoning_summary_delivery: Option<Str>,
}

/// Complete typed request body for `/v1/responses` and Codex Responses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesRequest {
	/// Opaque codec-facing wire model identifier.
	pub model:                  Str,
	/// Ordered input items.
	pub input:                  Vec<ResponsesInputItem>,
	/// Request streaming delivery.
	#[serde(default)]
	pub stream:                 bool,
	/// Store provider-side state.
	#[serde(default)]
	pub store:                  bool,
	/// Coalesced system instructions.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub instructions:           Option<Str>,
	/// Authoritative prior response identity.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub previous_response_id:   Option<Str>,
	/// Prompt-cache identity.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub prompt_cache_key:       Option<Str>,
	/// Prompt-cache retention string.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub prompt_cache_retention: Option<Str>,
	/// Requested native output inclusions.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub include:                Vec<Str>,
	/// Tool declarations.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub tools:                  Vec<ResponsesTool>,
	/// Responses Lite tool declarations moved into input.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub additional_tools:       Vec<ResponsesTool>,
	/// Tool selection.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub tool_choice:            Option<ResponsesToolChoice>,
	/// Whether tools may be called concurrently.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub parallel_tool_calls:    Option<bool>,
	/// Reasoning controls.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub reasoning:              Option<ResponsesReasoning>,
	/// Text controls.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub text:                   Option<ResponsesTextOptions>,
	/// Temperature.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub temperature:            Option<f32>,
	/// Nucleus probability.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub top_p:                  Option<f32>,
	/// Presence penalty.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub presence_penalty:       Option<f32>,
	/// Frequency penalty.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub frequency_penalty:      Option<f32>,
	/// Maximum generated tokens.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub max_output_tokens:      Option<u64>,
	/// Service tier.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub service_tier:           Option<Str>,
	/// Request metadata.
	#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
	pub metadata:               ResponsesMetadata,
	/// Codex client fingerprint metadata.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub client_metadata:        Option<ResponsesMetadata>,
	/// Codex stream options.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub stream_options:         Option<ResponsesStreamOptions>,
}

/// Lossless continuation boundary selected by session planning.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResponsesContinuation {
	/// Authoritative response identity.
	pub response_id:     Str,
	/// Number of canonical input items committed into that response.
	pub committed_items: usize,
}

/// OpenAI Responses-specific typed options.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpenAiResponsesOptions {
	/// Enable provider-side response storage and continuation.
	pub stateful:               bool,
	/// Authoritative continuation boundary.
	pub continuation:           Option<ResponsesContinuation>,
	/// Explicit native include entries.
	pub include:                Vec<Str>,
	/// Prompt-cache key.
	pub prompt_cache_key:       Option<Str>,
	/// Prompt-cache retention.
	pub prompt_cache_retention: Option<Str>,
	/// Parallel tool-call preference.
	pub parallel_tool_calls:    Option<bool>,
	/// Provider reasoning mode.
	pub reasoning_mode:         Option<Str>,
	/// Explicit reasoning summary selection; `Some(None)` sends JSON null.
	pub reasoning_summary:      Option<Option<Str>>,
	/// Extra typed custom tools not expressible as canonical function tools.
	pub custom_tools:           Vec<ResponsesTool>,
	/// Native computer-use tool declaration.
	pub computer_tool:          Option<ResponsesTool>,
	/// Native continuation/replay items that carry provider proof.
	pub native_input:           Vec<ResponsesInputItem>,
	/// Request metadata.
	pub metadata:               ResponsesMetadata,
}

/// Explicit codec adjustment; every unsupported requested axis produces one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResponsesAdjustment {
	/// A requested field has no Responses wire representation.
	Dropped { field: Str, reason: Str },
	/// A native representation was safely emulated.
	Emulated { field: Str, method: Str },
}

/// An encoded Responses body and its exact adjustment evidence.
#[derive(Clone, Debug)]
pub struct EncodedResponses {
	/// Typed request body.
	pub request:     ResponsesRequest,
	/// Explicit omissions or emulations.
	pub adjustments: Vec<ResponsesAdjustment>,
}

/// Codec-local encoding failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResponsesEncodeError {
	/// Continuation proof belongs to a different codec.
	MismatchedProviderProof,
	/// A valid provider proof cannot be represented at this canonical position.
	UnreplayableProviderProof,
	/// Chat encoding requires a model-bearing wire target.
	MissingWireTarget,
	/// A replay item lacks required provider call identity.
	MissingCallIdentity,
	/// Stored media was not resolved before wire encoding.
	UnresolvedStoredMedia,
	/// A required output format is unsupported by Responses.
	UnsupportedOutputFormat,
	/// Route policy explicitly rejects native computer use.
	UnsupportedComputerUse,
	/// Route policy explicitly rejects the supplied computer-use configuration.
	UnsupportedComputerUseConfig,
	/// Compatible session binding contained malformed or contradictory provider
	/// state.
	MalformedServerState,
	/// Explicit codec continuation conflicts with authoritative session state.
	MismatchedServerState,
}

/// Responses output-item discriminator.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesOutputItemKind {
	/// Assistant message.
	Message,
	/// Reasoning item.
	Reasoning,
	/// Function call.
	FunctionCall,
	/// Custom tool call.
	CustomToolCall,
	/// Computer call.
	ComputerCall,
	/// Hosted web-search call.
	WebSearchCall,
	/// Hosted file-search call.
	FileSearchCall,
	/// Hosted code-interpreter call.
	CodeInterpreterCall,
	/// Hosted image-generation call.
	ImageGenerationCall,
	/// Hosted MCP call.
	McpCall,
	/// Local shell call.
	LocalShellCall,
}

/// Typed output item carried by stream events and terminal responses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesOutputItem {
	/// Item discriminator.
	#[serde(rename = "type")]
	pub kind:                  ResponsesOutputItemKind,
	/// Provider item identity.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub id:                    Option<Str>,
	/// Stable call identity.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub call_id:               Option<Str>,
	/// Tool name.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub name:                  Option<Str>,
	/// Function arguments.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub arguments:             Option<Str>,
	/// Custom-tool input.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub input:                 Option<Str>,
	/// Message content.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub content:               Vec<ResponsesContent>,
	/// Reasoning summaries.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub summary:               Vec<ResponsesSummaryPart>,
	/// Encrypted reasoning proof.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub encrypted_content:     Option<Str>,
	/// Computer actions.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub actions:               Vec<ResponsesComputerAction>,
	/// Pending computer safety checks.
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub pending_safety_checks: Vec<ResponsesSafetyCheck>,
	/// Provider status.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub status:                Option<Str>,
	/// Image-generation base64 result.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub result:                Option<Str>,
}

/// Detailed token accounting from a Responses terminal event.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesUsage {
	/// Input tokens.
	#[serde(default)]
	pub input_tokens:          u64,
	/// Output tokens.
	#[serde(default)]
	pub output_tokens:         u64,
	/// Total tokens, retained for wire parity.
	#[serde(default)]
	pub total_tokens:          u64,
	/// Input-token details.
	#[serde(default)]
	pub input_tokens_details:  ResponsesInputTokenDetails,
	/// Output-token details.
	#[serde(default)]
	pub output_tokens_details: ResponsesOutputTokenDetails,
}

/// Responses input-token details.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesInputTokenDetails {
	/// Cached input tokens.
	#[serde(default)]
	pub cached_tokens: u64,
}

/// Responses output-token details.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesOutputTokenDetails {
	/// Reasoning tokens.
	#[serde(default)]
	pub reasoning_tokens: u64,
}

impl From<&ResponsesUsage> for Usage {
	fn from(value: &ResponsesUsage) -> Self {
		Self {
			input_tokens: value.input_tokens,
			output_tokens: value.output_tokens,
			reasoning_tokens: value.output_tokens_details.reasoning_tokens,
			cache_read_tokens: value.input_tokens_details.cached_tokens,
			source: UsageSource::Provider,
			..Self::default()
		}
	}
}

/// Structured provider error object.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesErrorObject {
	/// Stable provider error code.
	#[serde(default)]
	pub code:    Option<Str>,
	/// Provider error category.
	#[serde(rename = "type", default)]
	pub kind:    Option<Str>,
	/// Sanitized provider message.
	#[serde(default)]
	pub message: Option<Str>,
	/// Invalid request parameter.
	#[serde(default)]
	pub param:   Option<Str>,
}

/// Incomplete-response details.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesIncompleteDetails {
	/// Incomplete reason.
	pub reason: Str,
}

/// Provider status details attached to failed or incomplete responses.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesStatusDetails {
	/// Structured nested error.
	#[serde(default)]
	pub error: Option<ResponsesErrorObject>,
}

/// Typed response envelope carried by lifecycle events.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponsesResponse {
	/// Authoritative response identity.
	#[serde(default)]
	pub id:                 Option<Str>,
	/// Wire model identity.
	#[serde(default)]
	pub model:              Option<Str>,
	/// Response status.
	#[serde(default)]
	pub status:             Option<Str>,
	/// Terminal error.
	#[serde(default)]
	pub error:              Option<ResponsesErrorObject>,
	/// Nested status details.
	#[serde(default)]
	pub status_details:     Option<ResponsesStatusDetails>,
	/// Incomplete details.
	#[serde(default)]
	pub incomplete_details: Option<ResponsesIncompleteDetails>,
	/// Terminal output items.
	#[serde(default)]
	pub output:             Vec<ResponsesOutputItem>,
	/// Terminal usage.
	#[serde(default)]
	pub usage:              Option<ResponsesUsage>,
	/// Applied service tier.
	#[serde(default)]
	pub service_tier:       Option<Str>,
}

/// Responses stream event discriminator.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ResponsesStreamEventKind {
	/// Response created.
	#[serde(rename = "response.created")]
	Created,
	/// Response queued.
	#[serde(rename = "response.queued")]
	Queued,
	/// Response in progress.
	#[serde(rename = "response.in_progress")]
	InProgress,
	/// Output item began.
	#[serde(rename = "response.output_item.added")]
	OutputItemAdded,
	/// Output item completed.
	#[serde(rename = "response.output_item.done")]
	OutputItemDone,
	/// Output text delta.
	#[serde(rename = "response.output_text.delta")]
	OutputTextDelta,
	/// Output refusal delta.
	#[serde(rename = "response.refusal.delta")]
	RefusalDelta,
	/// Output text completed.
	#[serde(rename = "response.output_text.done")]
	OutputTextDone,
	/// Refusal completed.
	#[serde(rename = "response.refusal.done")]
	RefusalDone,
	/// Reasoning summary delta.
	#[serde(rename = "response.reasoning_summary_text.delta")]
	ReasoningSummaryDelta,
	/// Raw reasoning delta.
	#[serde(rename = "response.reasoning_text.delta")]
	ReasoningDelta,
	/// Reasoning summary completed.
	#[serde(rename = "response.reasoning_summary_text.done")]
	ReasoningSummaryDone,
	/// Raw reasoning completed.
	#[serde(rename = "response.reasoning_text.done")]
	ReasoningDone,
	/// Function arguments delta.
	#[serde(rename = "response.function_call_arguments.delta")]
	FunctionArgumentsDelta,
	/// Function arguments completed.
	#[serde(rename = "response.function_call_arguments.done")]
	FunctionArgumentsDone,
	/// Custom-tool input delta.
	#[serde(rename = "response.custom_tool_call_input.delta")]
	CustomInputDelta,
	/// Custom-tool input completed.
	#[serde(rename = "response.custom_tool_call_input.done")]
	CustomInputDone,
	/// Partial generated image.
	#[serde(rename = "response.image_generation_call.partial_image")]
	PartialImage,
	/// Successful terminal response.
	#[serde(rename = "response.completed")]
	Completed,
	/// Incomplete terminal response.
	#[serde(rename = "response.incomplete")]
	Incomplete,
	/// Alternate terminal response used by Codex.
	#[serde(rename = "response.done")]
	Done,
	/// Failed terminal response.
	#[serde(rename = "response.failed")]
	Failed,
	/// Cancelled terminal response.
	#[serde(rename = "response.cancelled")]
	Cancelled,
	/// Hosted web-search lifecycle event.
	#[serde(rename = "response.web_search_call.in_progress")]
	WebSearchInProgress,
	/// Hosted web-search searching event.
	#[serde(rename = "response.web_search_call.searching")]
	WebSearchSearching,
	/// Hosted web-search completed event.
	#[serde(rename = "response.web_search_call.completed")]
	WebSearchCompleted,
	/// Top-level streamed error envelope.
	#[serde(rename = "error")]
	Error,
	/// Unknown forward-compatible provider event.
	#[serde(other)]
	Other,
}

/// One fully typed Responses stream event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesStreamEvent {
	/// Event discriminator.
	#[serde(rename = "type")]
	pub kind:              ResponsesStreamEventKind,
	/// Provider sequence number.
	#[serde(default)]
	pub sequence_number:   Option<u64>,
	/// Output index.
	#[serde(default)]
	pub output_index:      Option<u32>,
	/// Item identity for item-correlated deltas.
	#[serde(default)]
	pub item_id:           Option<Str>,
	/// Output item.
	#[serde(default)]
	pub item:              Option<ResponsesOutputItem>,
	/// Text or argument delta.
	#[serde(default)]
	pub delta:             Option<Str>,
	/// Authoritative completed text.
	#[serde(default)]
	pub text:              Option<Str>,
	/// Authoritative completed function arguments.
	#[serde(default)]
	pub arguments:         Option<Str>,
	/// Authoritative completed custom input.
	#[serde(default)]
	pub input:             Option<Str>,
	/// Partial image base64 payload.
	#[serde(default)]
	pub partial_image_b64: Option<Str>,
	/// Lifecycle response envelope.
	#[serde(default)]
	pub response:          Option<ResponsesResponse>,
	/// Top-level provider error code.
	#[serde(default)]
	pub code:              Option<Str>,
	/// Top-level provider error message.
	#[serde(default)]
	pub message:           Option<Str>,
	/// Top-level invalid parameter.
	#[serde(default)]
	pub param:             Option<Str>,
	/// Nested provider error envelope.
	#[serde(default)]
	pub error:             Option<ResponsesErrorObject>,
}

/// Structured continuation failure classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponsesContinuationFailure {
	/// The previous response identity is stale or unavailable.
	StalePreviousResponse,
	/// A referenced server item is stale or unavailable.
	StaleServerItem,
	/// The error is unrelated to continuation state.
	NotStale,
	/// The body was not a typed provider error envelope.
	Malformed,
}

/// Evidence surfaced by the decoder without leaking arbitrary response bodies.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResponsesErrorEvidence {
	/// Stable provider code.
	pub code:         Option<Str>,
	/// Sanitized provider message.
	pub message:      Str,
	/// Continuation classification.
	pub continuation: ResponsesContinuationFailure,
}

/// Non-canonical state evidence preserved beside canonical chat events.
#[derive(Debug)]
pub enum ResponsesProjection {
	/// Canonical chat event.
	Canonical(ChatEvent),
	/// Terminal chat facts awaiting authoritative receipt accounting.
	Completion(super::RawCompletion),
	/// A provider tool call is complete but not yet schema-validated.
	ToolCallComplete {
		/// Output index.
		index:     u32,
		/// Stable call identity.
		id:        ToolCallId,
		/// Tool name.
		name:      Str,
		/// Complete arguments or freeform input bytes.
		arguments: Bytes,
		/// Whether this is a custom/freeform tool call.
		custom:    bool,
	},
	/// Stable output-item identity for continuation replay.
	OutputItem {
		/// Output index.
		index: u32,
		/// Provider item identity.
		id:    Str,
	},
	/// Encrypted reasoning proof for session continuation.
	ReasoningSignature {
		/// Output index.
		index:     u32,
		/// Provider item identity.
		item_id:   Option<Str>,
		/// Opaque encrypted content.
		signature: Bytes,
	},
	/// Provider-hosted tool lifecycle output.
	HostedTool {
		/// Output index.
		index:     u32,
		/// Hosted tool kind.
		kind:      ResponsesOutputItemKind,
		/// Whether the provider reported completion.
		completed: bool,
	},
	/// Authoritative continuation identity.
	Continuation {
		/// Response identity.
		response_id:  Str,
		/// Wire model identity.
		model:        Option<Str>,
		/// Applied service tier.
		service_tier: Option<Str>,
	},
	/// Terminal protocol/provider failure.
	Error(ResponsesErrorEvidence),
}

#[derive(Debug)]
enum OutputSlot {
	Text {
		item_id: Option<Str>,
		text:    BytesMut,
		emitted: bool,
	},
	Thinking {
		item_id:   Option<Str>,
		text:      BytesMut,
		encrypted: Bytes,
		emitted:   bool,
	},
	Tool {
		item_id:   Option<Str>,
		call_id:   ToolCallId,
		name:      Str,
		arguments: BytesMut,
		custom:    bool,
	},
	Computer {
		item_id:   Option<Str>,
		call_id:   ToolCallId,
		arguments: Bytes,
	},
	Hosted {
		kind:      ResponsesOutputItemKind,
		completed: bool,
	},
	Image {
		encoded: BytesMut,
	},
}

/// Incremental sans-I/O Responses event decoder.
#[derive(Debug, Default)]
pub struct OpenAiResponsesDecoder {
	response_id:               Option<Str>,
	model:                     Option<Str>,
	outputs:                   BTreeMap<u32, OutputSlot>,
	ended:                     BTreeSet<u32>,
	terminal:                  bool,
	next_index:                u32,
	saw_completed_hosted_tool: bool,
	saw_visible_output:        bool,
}

impl OpenAiResponsesDecoder {
	/// Decodes one complete SSE or WebSocket JSON payload.
	pub fn push_json(&mut self, payload: &[u8]) -> Vec<ResponsesProjection> {
		if self.terminal {
			return Vec::new();
		}
		let event = match serde_json::from_slice::<ResponsesStreamEvent>(payload) {
			Ok(event) => event,
			Err(_) => {
				self.terminal = true;
				return vec![ResponsesProjection::Error(ResponsesErrorEvidence {
					code:         Some(Str::from("invalid_responses_event")),
					message:      Str::from("invalid Responses event"),
					continuation: ResponsesContinuationFailure::Malformed,
				})];
			},
		};
		self.push_event(event)
	}

	/// Projects one already-decoded typed event.
	pub fn push_event(&mut self, event: ResponsesStreamEvent) -> Vec<ResponsesProjection> {
		if self.terminal {
			return Vec::new();
		}
		let mut out = Vec::new();
		match event.kind {
			ResponsesStreamEventKind::Created
			| ResponsesStreamEventKind::Queued
			| ResponsesStreamEventKind::InProgress => {
				self.capture_response(event.response.as_ref());
			},
			ResponsesStreamEventKind::OutputItemAdded => {
				let index = self.event_index(&event);
				if let Some(item) = event.item.as_ref() {
					self.add_item(index, item, &mut out);
				}
			},
			ResponsesStreamEventKind::OutputTextDelta | ResponsesStreamEventKind::RefusalDelta => {
				self.append_delta(&event, SlotClass::Text, &mut out);
			},
			ResponsesStreamEventKind::ReasoningSummaryDelta
			| ResponsesStreamEventKind::ReasoningDelta => {
				self.append_delta(&event, SlotClass::Thinking, &mut out);
			},
			ResponsesStreamEventKind::FunctionArgumentsDelta
			| ResponsesStreamEventKind::CustomInputDelta => {
				self.append_delta(&event, SlotClass::Tool, &mut out);
			},
			ResponsesStreamEventKind::OutputTextDone | ResponsesStreamEventKind::RefusalDone => {
				self.replace_done(&event, SlotClass::Text)
			},
			ResponsesStreamEventKind::ReasoningSummaryDone
			| ResponsesStreamEventKind::ReasoningDone => self.replace_done(&event, SlotClass::Thinking),
			ResponsesStreamEventKind::FunctionArgumentsDone
			| ResponsesStreamEventKind::CustomInputDone => self.replace_done(&event, SlotClass::Tool),
			ResponsesStreamEventKind::PartialImage => {
				if let Some(index) = self.lookup_index(&event)
					&& let Some(OutputSlot::Image { encoded }) = self.outputs.get_mut(&index)
					&& let Some(delta) = event.partial_image_b64.or(event.delta)
				{
					encoded.extend_from_slice(delta.as_bytes());
				}
			},
			ResponsesStreamEventKind::OutputItemDone => {
				let index = self.event_index(&event);
				if let Some(item) = event.item.as_ref() {
					if !self.outputs.contains_key(&index) {
						self.add_item(index, item, &mut out);
					}
					self.complete_item(index, item);
				}
				self.end_slot(index, &mut out);
			},
			ResponsesStreamEventKind::Error => {
				self.terminal = true;
				let nested = event.error.as_ref();
				out.push(ResponsesProjection::Error(ResponsesErrorEvidence {
					code:         nested.and_then(|error| error.code.clone()).or(event.code),
					message:      nested
						.and_then(|error| error.message.clone())
						.or(event.message)
						.unwrap_or_else(|| Str::from("Responses request failed")),
					continuation: ResponsesContinuationFailure::NotStale,
				}));
			},
			ResponsesStreamEventKind::WebSearchCompleted => {
				if let Some(index) = self.lookup_index(&event) {
					self.saw_completed_hosted_tool = true;
					out.push(ResponsesProjection::HostedTool {
						index,
						kind: ResponsesOutputItemKind::WebSearchCall,
						completed: true,
					});
				}
			},
			ResponsesStreamEventKind::Completed
			| ResponsesStreamEventKind::Incomplete
			| ResponsesStreamEventKind::Done => {
				let incomplete = event.kind == ResponsesStreamEventKind::Incomplete;
				self.finish_response(event.response.as_ref(), incomplete, &mut out);
			},
			ResponsesStreamEventKind::Failed | ResponsesStreamEventKind::Cancelled => {
				self.terminal = true;
				self.capture_response(event.response.as_ref());
				out.push(ResponsesProjection::Error(error_from_response(
					event.response.as_ref(),
					event.kind,
				)));
			},
			ResponsesStreamEventKind::WebSearchInProgress
			| ResponsesStreamEventKind::WebSearchSearching
			| ResponsesStreamEventKind::Other => {},
		}
		out
	}

	/// Finishes framing; a nonterminal stream is a protocol error.
	pub fn finish(&mut self) -> Vec<ResponsesProjection> {
		if self.terminal {
			return Vec::new();
		}
		self.terminal = true;
		vec![ResponsesProjection::Error(ResponsesErrorEvidence {
			code:         Some(Str::from("premature_end")),
			message:      Str::from("Responses stream ended before an authoritative terminal event"),
			continuation: ResponsesContinuationFailure::NotStale,
		})]
	}

	fn committed_output(&self) -> bool {
		self.outputs.values().any(|slot| {
			matches!(
				slot,
				OutputSlot::Text { .. }
					| OutputSlot::Thinking { .. }
					| OutputSlot::Tool { .. }
					| OutputSlot::Computer { .. }
			)
		})
	}

	/// Returns whether an authoritative terminal event was received.
	pub const fn is_terminal(&self) -> bool {
		self.terminal
	}

	fn capture_response(&mut self, response: Option<&ResponsesResponse>) {
		if let Some(response) = response {
			if let Some(id) = &response.id {
				self.response_id = Some(id.clone());
			}
			if let Some(model) = &response.model {
				self.model = Some(model.clone());
			}
		}
	}

	fn event_index(&mut self, event: &ResponsesStreamEvent) -> u32 {
		if let Some(index) = event.output_index {
			self.next_index = self.next_index.max(index.saturating_add(1));
			index
		} else if let Some(index) = self.lookup_index(event) {
			index
		} else {
			let index = self.next_index;
			self.next_index = self.next_index.saturating_add(1);
			index
		}
	}

	fn lookup_index(&self, event: &ResponsesStreamEvent) -> Option<u32> {
		event.output_index.or_else(|| {
			event.item_id.as_ref().and_then(|id| {
				self.outputs.iter().find_map(|(index, slot)| {
					slot_item_id(slot)
						.is_some_and(|candidate| candidate == id.as_str())
						.then_some(*index)
				})
			})
		})
	}

	fn add_item(
		&mut self,
		index: u32,
		item: &ResponsesOutputItem,
		out: &mut Vec<ResponsesProjection>,
	) {
		if matches!(
			item.kind,
			ResponsesOutputItemKind::FunctionCall
				| ResponsesOutputItemKind::CustomToolCall
				| ResponsesOutputItemKind::ComputerCall
		) && (item.call_id.is_none()
			|| (item.kind != ResponsesOutputItemKind::ComputerCall && item.name.is_none()))
		{
			self.terminal = true;
			out.push(ResponsesProjection::Error(ResponsesErrorEvidence {
				code:         Some(Str::from("missing_tool_call_identity")),
				message:      Str::from("Responses tool call omitted required identity"),
				continuation: ResponsesContinuationFailure::NotStale,
			}));
			return;
		}
		if let Some(id) = item.id.as_ref() {
			out.push(ResponsesProjection::OutputItem { index, id: id.clone() });
		}
		let slot = match item.kind {
			ResponsesOutputItemKind::Message => {
				OutputSlot::Text { item_id: item.id.clone(), text: BytesMut::new(), emitted: false }
			},
			ResponsesOutputItemKind::Reasoning => OutputSlot::Thinking {
				item_id:   item.id.clone(),
				text:      BytesMut::new(),
				encrypted: item
					.encrypted_content
					.as_ref()
					.map_or_else(Bytes::new, |value| Bytes::copy_from_slice(value.as_bytes())),
				emitted:   false,
			},
			ResponsesOutputItemKind::FunctionCall | ResponsesOutputItemKind::CustomToolCall => {
				let call_id = item.call_id.clone().unwrap_or_default();
				OutputSlot::Tool {
					item_id:   item.id.clone(),
					call_id:   ToolCallId::from(call_id),
					name:      item.name.clone().unwrap_or_default(),
					arguments: BytesMut::from(
						item
							.arguments
							.as_deref()
							.or(item.input.as_deref())
							.unwrap_or_default()
							.as_bytes(),
					),
					custom:    item.kind == ResponsesOutputItemKind::CustomToolCall,
				}
			},
			ResponsesOutputItemKind::ComputerCall => {
				let call_id = ToolCallId::from(item.call_id.clone().unwrap_or_default());
				let arguments = serde_json::to_vec(&ResponsesComputerArguments {
					actions:               item.actions.clone(),
					pending_safety_checks: item.pending_safety_checks.clone(),
				})
				.map_or_else(|_| Bytes::new(), Bytes::from);
				OutputSlot::Computer { item_id: item.id.clone(), call_id, arguments }
			},
			ResponsesOutputItemKind::ImageGenerationCall => {
				OutputSlot::Image { encoded: BytesMut::new() }
			},
			kind => {
				let completed = item
					.status
					.as_deref()
					.is_none_or(|status| status == "completed");
				if completed {
					self.saw_completed_hosted_tool = true;
				}
				out.push(ResponsesProjection::HostedTool { index, kind, completed });
				OutputSlot::Hosted { kind, completed }
			},
		};
		match &slot {
			OutputSlot::Text { .. } => {
				out.push(ResponsesProjection::Canonical(ChatEvent::BlockStarted {
					index,
					kind: BlockKind::Text,
				}))
			},
			OutputSlot::Thinking { .. } => {
				out.push(ResponsesProjection::Canonical(ChatEvent::BlockStarted {
					index,
					kind: BlockKind::Thinking,
				}))
			},
			OutputSlot::Tool { call_id, .. } | OutputSlot::Computer { call_id, .. } => {
				let name = match &slot {
					OutputSlot::Tool { name, .. } => name.clone(),
					OutputSlot::Computer { .. } => Str::from("computer"),
					_ => unreachable!("tool arm only"),
				};
				out.push(ResponsesProjection::Canonical(ChatEvent::BlockStarted {
					index,
					kind: BlockKind::ToolCall,
				}));
				out.push(ResponsesProjection::Canonical(ChatEvent::ToolCallStarted {
					index,
					id: call_id.clone(),
					name,
				}));
			},
			OutputSlot::Hosted { .. } | OutputSlot::Image { .. } => {},
		}
		self.outputs.insert(index, slot);
	}

	fn append_delta(
		&mut self,
		event: &ResponsesStreamEvent,
		class: SlotClass,
		out: &mut Vec<ResponsesProjection>,
	) {
		let Some(index) = self.lookup_index(event) else {
			return;
		};
		if self.ended.contains(&index) {
			return;
		}
		let Some(delta) = event.delta.as_ref() else {
			return;
		};
		if delta.is_empty() {
			return;
		}
		match (self.outputs.get_mut(&index), class) {
			(Some(OutputSlot::Text { text, emitted, .. }), SlotClass::Text) => {
				text.extend_from_slice(delta.as_bytes());
				*emitted = true;
				self.saw_visible_output = true;
				out.push(ResponsesProjection::Canonical(ChatEvent::TextDelta {
					index,
					text: delta.clone(),
				}));
			},
			(Some(OutputSlot::Thinking { text, emitted, .. }), SlotClass::Thinking) => {
				text.extend_from_slice(delta.as_bytes());
				*emitted = true;
				out.push(ResponsesProjection::Canonical(ChatEvent::ThinkingDelta {
					index,
					text: delta.clone(),
				}));
			},
			(Some(OutputSlot::Tool { arguments, .. }), SlotClass::Tool) => {
				arguments.extend_from_slice(delta.as_bytes());
				self.saw_visible_output = true;
				out.push(ResponsesProjection::Canonical(ChatEvent::ToolArgumentsDelta {
					index,
					bytes: Bytes::copy_from_slice(delta.as_bytes()),
				}));
			},
			_ => {},
		}
	}

	fn replace_done(&mut self, event: &ResponsesStreamEvent, class: SlotClass) {
		let Some(index) = self.lookup_index(event) else {
			return;
		};
		let complete = match class {
			SlotClass::Text | SlotClass::Thinking => event.text.as_ref(),
			SlotClass::Tool => event.arguments.as_ref().or(event.input.as_ref()),
		};
		let Some(complete) = complete else {
			return;
		};
		match (self.outputs.get_mut(&index), class) {
			(Some(OutputSlot::Text { text, .. }), SlotClass::Text)
			| (Some(OutputSlot::Thinking { text, .. }), SlotClass::Thinking)
			| (Some(OutputSlot::Tool { arguments: text, .. }), SlotClass::Tool) => {
				text.clear();
				text.extend_from_slice(complete.as_bytes());
			},
			_ => {},
		}
	}

	fn complete_item(&mut self, index: u32, item: &ResponsesOutputItem) {
		match self.outputs.get_mut(&index) {
			Some(OutputSlot::Text { item_id, text, .. }) => {
				if let Some(id) = &item.id {
					*item_id = Some(id.clone());
				}
				if !item.content.is_empty() {
					text.clear();
					for part in &item.content {
						if let Some(value) = &part.text {
							text.extend_from_slice(value.as_bytes());
						}
					}
				}
			},
			Some(OutputSlot::Thinking { item_id, text, encrypted, .. }) => {
				if let Some(id) = &item.id {
					*item_id = Some(id.clone());
				}
				if let Some(value) = &item.encrypted_content {
					*encrypted = Bytes::copy_from_slice(value.as_bytes());
				}
				if !item.summary.is_empty() {
					text.clear();
					for (position, summary) in item.summary.iter().enumerate() {
						if position != 0 {
							text.extend_from_slice(b"\n\n");
						}
						text.extend_from_slice(summary.text.as_bytes());
					}
				}
			},
			Some(OutputSlot::Tool { item_id, call_id, name, arguments, custom }) => {
				if let Some(id) = &item.id {
					*item_id = Some(id.clone());
				}
				if let Some(id) = &item.call_id {
					*call_id = ToolCallId::from(id.clone());
				}
				if let Some(value) = &item.name {
					*name = value.clone();
				}
				let complete = if *custom {
					item.input.as_ref()
				} else {
					item.arguments.as_ref()
				};
				if let Some(value) = complete {
					arguments.clear();
					arguments.extend_from_slice(value.as_bytes());
				}
			},
			Some(OutputSlot::Computer { item_id, call_id, arguments }) => {
				if let Some(id) = &item.id {
					*item_id = Some(id.clone());
				}
				if let Some(id) = &item.call_id {
					*call_id = ToolCallId::from(id.clone());
				}
				*arguments = serde_json::to_vec(&ResponsesComputerArguments {
					actions:               item.actions.clone(),
					pending_safety_checks: item.pending_safety_checks.clone(),
				})
				.map_or_else(|_| Bytes::new(), Bytes::from);
			},
			Some(OutputSlot::Hosted { completed, .. }) => {
				*completed = item
					.status
					.as_deref()
					.is_none_or(|status| status == "completed")
			},
			Some(OutputSlot::Image { encoded }) => {
				if let Some(value) = &item.result {
					encoded.clear();
					encoded.extend_from_slice(value.as_bytes());
				}
			},
			None => {},
		}
	}

	fn end_slot(&mut self, index: u32, out: &mut Vec<ResponsesProjection>) {
		if !self.ended.insert(index) {
			return;
		}
		match self.outputs.get(&index) {
			Some(OutputSlot::Text { text, emitted: false, .. }) if !text.is_empty() => {
				out.push(ResponsesProjection::Canonical(ChatEvent::TextDelta {
					index,
					text: Str::from_utf8_lossy(text),
				}));
			},
			Some(OutputSlot::Thinking { item_id, text, encrypted, emitted }) => {
				if !*emitted && !text.is_empty() {
					out.push(ResponsesProjection::Canonical(ChatEvent::ThinkingDelta {
						index,
						text: Str::from_utf8_lossy(text),
					}));
				}
				out.push(ResponsesProjection::ReasoningSignature {
					index,
					item_id: item_id.clone(),
					signature: encrypted.clone(),
				});
			},
			Some(OutputSlot::Tool { call_id, name, arguments, custom, .. }) => {
				out.push(ResponsesProjection::ToolCallComplete {
					index,
					id: call_id.clone(),
					name: name.clone(),
					arguments: arguments.clone().freeze(),
					custom: *custom,
				})
			},
			Some(OutputSlot::Computer { call_id, arguments, .. }) => {
				out.push(ResponsesProjection::ToolCallComplete {
					index,
					id: call_id.clone(),
					name: Str::from("computer"),
					arguments: arguments.clone(),
					custom: false,
				})
			},
			Some(OutputSlot::Hosted { kind, completed }) => {
				out.push(ResponsesProjection::HostedTool { index, kind: *kind, completed: *completed })
			},
			Some(OutputSlot::Image { encoded }) => {
				if let Ok(bytes) = omp_core::encoding::base64::decode(encoded).into_vec() {
					let bytes = Bytes::from(bytes);
					out.push(ResponsesProjection::Canonical(ChatEvent::Artifact {
						index,
						artifact: Artifact {
							media_type: Str::from("image/png"),
							size:       Some(bytes.len() as u64),
							digest:     None,
							body:       ArtifactBody::Bytes(bytes),
						},
					}));
				}
			},
			_ => {},
		}
	}

	fn finish_response(
		&mut self,
		response: Option<&ResponsesResponse>,
		incomplete: bool,
		out: &mut Vec<ResponsesProjection>,
	) {
		self.terminal = true;
		self.capture_response(response);
		if let Some(response) = response {
			for (index, item) in response
				.output
				.iter()
				.enumerate()
				.filter_map(|(i, item)| u32::try_from(i).ok().map(|i| (i, item)))
			{
				if !self.outputs.contains_key(&index) {
					self.add_item(index, item, out);
				}
				self.complete_item(index, item);
			}
		}
		let open = self
			.outputs
			.keys()
			.copied()
			.filter(|index| !self.ended.contains(index))
			.collect::<Vec<_>>();
		for index in open {
			self.end_slot(index, out);
		}
		if let Some(response) = response {
			if response
				.status
				.as_deref()
				.is_some_and(|status| matches!(status, "failed" | "cancelled"))
			{
				out.push(ResponsesProjection::Error(error_from_response(
					Some(response),
					ResponsesStreamEventKind::Failed,
				)));
				return;
			}
			if let Some(usage) = &response.usage {
				let mut usage = Usage::from(usage);
				usage.search_calls = u32::from(self.saw_completed_hosted_tool);
				out.push(ResponsesProjection::Canonical(ChatEvent::Usage(crate::event::UsageUpdate {
					usage,
					final_update: true,
				})));
			}
		}
		if let Some(response_id) = self.response_id.clone() {
			out.push(ResponsesProjection::Continuation {
				response_id,
				model: self.model.clone(),
				service_tier: response.and_then(|value| value.service_tier.clone()),
			});
		}
		let reason = if incomplete {
			if response
				.and_then(|value| value.incomplete_details.as_ref())
				.is_some_and(|details| details.reason == "content_filter")
			{
				FinishReason::ContentFilter
			} else {
				FinishReason::Length
			}
		} else if self
			.outputs
			.values()
			.any(|slot| matches!(slot, OutputSlot::Tool { .. } | OutputSlot::Computer { .. }))
			|| (self.saw_completed_hosted_tool && !self.saw_visible_output)
		{
			FinishReason::ToolCalls
		} else {
			FinishReason::Stop
		};
		let mut usage = response
			.and_then(|value| value.usage.as_ref())
			.map_or_else(Usage::default, Usage::from);
		usage.search_calls = u32::from(self.saw_completed_hosted_tool);
		out.push(ResponsesProjection::Completion(super::RawCompletion {
			reason,
			blocks: self.outputs.len().try_into().unwrap_or(u32::MAX),
			usage,
		}));
	}
}

#[derive(Clone, Copy)]
enum SlotClass {
	Text,
	Thinking,
	Tool,
}

fn slot_item_id(slot: &OutputSlot) -> Option<&str> {
	match slot {
		OutputSlot::Text { item_id, .. }
		| OutputSlot::Thinking { item_id, .. }
		| OutputSlot::Computer { item_id, .. } => item_id.as_deref(),
		OutputSlot::Tool { item_id, call_id, .. } => item_id.as_deref().or(Some(call_id.as_str())),
		OutputSlot::Hosted { .. } | OutputSlot::Image { .. } => None,
	}
}

fn error_from_response(
	response: Option<&ResponsesResponse>,
	event: ResponsesStreamEventKind,
) -> ResponsesErrorEvidence {
	let error = response.and_then(|value| {
		value.error.as_ref().or_else(|| {
			value
				.status_details
				.as_ref()
				.and_then(|details| details.error.as_ref())
		})
	});
	let message = error
		.and_then(|value| value.message.clone())
		.or_else(|| {
			response.and_then(|value| {
				value
					.incomplete_details
					.as_ref()
					.map(|details| details.reason.clone())
			})
		})
		.unwrap_or_else(|| {
			Str::from(match event {
				ResponsesStreamEventKind::Cancelled => "caller cancelled",
				_ => "Responses request failed",
			})
		});
	ResponsesErrorEvidence {
		code: error.and_then(|value| value.code.clone()),
		message,
		continuation: ResponsesContinuationFailure::NotStale,
	}
}

/// Classifies an HTTP error using a typed error envelope and exact stale-state
/// evidence.
pub fn classify_continuation_error(status: u16, body: &[u8]) -> ResponsesErrorEvidence {
	#[derive(Deserialize)]
	struct Envelope {
		error: ResponsesErrorObject,
	}
	let Ok(envelope) = serde_json::from_slice::<Envelope>(body) else {
		return ResponsesErrorEvidence {
			code:         None,
			message:      Str::from_utf8_lossy(body),
			continuation: ResponsesContinuationFailure::Malformed,
		};
	};
	let code = envelope.error.code.clone();
	let message = envelope
		.error
		.message
		.clone()
		.unwrap_or_else(|| Str::from("Responses request failed"));
	let kind =
		if matches!(status, 400 | 404) && code.as_deref() == Some("previous_response_not_found") {
			ResponsesContinuationFailure::StalePreviousResponse
		} else if status == 404
			&& envelope.error.kind.as_deref() == Some("invalid_request_error")
			&& message.starts_with("previous_response_id '")
			&& message.ends_with("' was not found")
		{
			ResponsesContinuationFailure::StalePreviousResponse
		} else if status == 404
			&& envelope.error.kind.as_deref() == Some("invalid_request_error")
			&& message.starts_with("Item with id '")
			&& message.ends_with("' not found.")
		{
			ResponsesContinuationFailure::StaleServerItem
		} else {
			ResponsesContinuationFailure::NotStale
		};
	ResponsesErrorEvidence { code, message, continuation: kind }
}

/// Validates a completed function call syntactically without authorizing
/// execution.
pub fn syntactically_valid_function_call(arguments: &[u8]) -> Option<OpaqueJson> {
	serde_json::from_slice::<Value>(arguments)
		.ok()
		.map(OpaqueJson::new)
}

/// Converts a schema-validated complete call into the sole executable canonical
/// event.
pub fn authorize_validated_tool_call(
	index: u32,
	id: ToolCallId,
	name: Str,
	arguments: OpaqueJson,
) -> ChatEvent {
	ChatEvent::ToolCallReady { index, call: ToolCall { id, name, arguments } }
}

/// Opaque provider proof payload owned by this codec.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponsesProviderProof {
	/// Proof format revision.
	pub version:             u8,
	/// Authoritative response identity, when this proof establishes a
	/// continuation.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub response_id:         Option<Str>,
	/// Stable output item identity.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub item_id:             Option<Str>,
	/// Stable wire call identity.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub call_id:             Option<Str>,
	/// Whether the call used the custom/freeform wire shape.
	#[serde(default)]
	pub custom_tool:         bool,
	/// Opaque encrypted reasoning continuation.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub encrypted_reasoning: Option<Str>,
	/// Native computer-use call, when applicable.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub computer:            Option<ResponsesInputItem>,
}

/// Serializes typed codec proof bytes for storage in a canonical
/// `ProviderProof`.
pub fn encode_provider_proof(proof: &ResponsesProviderProof) -> Result<Bytes, serde_json::Error> {
	serde_json::to_vec(proof).map(Bytes::from)
}

/// Decodes typed codec proof bytes previously emitted by this codec.
pub fn decode_provider_proof(bytes: &[u8]) -> Result<ResponsesProviderProof, serde_json::Error> {
	serde_json::from_slice(bytes)
}

/// Pure OpenAI Responses codec.
#[derive(Clone, Debug, Default)]
pub struct OpenAiResponsesCodec {
	options: OpenAiResponsesOptions,
}

impl OpenAiResponsesCodec {
	/// Constructs a codec with typed route/session options.
	pub fn new(options: OpenAiResponsesOptions) -> Self {
		Self { options }
	}

	/// Borrows the configured typed options.
	pub const fn options(&self) -> &OpenAiResponsesOptions {
		&self.options
	}

	/// Encodes a canonical chat request into a typed Responses body.
	pub fn encode_chat(
		&self,
		context: &super::EncodeContext<'_>,
		request: &crate::call::ChatRequest,
	) -> Result<EncodedResponses, ResponsesEncodeError> {
		use crate::call::{
			CacheRetention, ContentPart, HostedTool, ReasoningVisibility, Role, Setting,
			StructuredOutput, TextVerbosity, ToolChoice, ToolResultContent,
		};

		let target = context
			.target
			.ok_or(ResponsesEncodeError::MissingWireTarget)?;
		let mut adjustments = Vec::new();
		let mut input = Vec::new();
		let mut instructions = Vec::new();
		let mut continuation_id = self
			.options
			.continuation
			.as_ref()
			.map(|value| value.response_id.clone());
		if let Some(binding) = context.server_state {
			let state = binding
				.provider_state()
				.map_err(|_| ResponsesEncodeError::MalformedServerState)?;
			let mut state_continuation = None;
			let mut output_items = Vec::new();
			for event in state {
				match event {
					crate::session::StoredProviderStateEvent::Continuation { handle } => {
						if state_continuation.replace(handle).is_some() {
							return Err(ResponsesEncodeError::MalformedServerState);
						}
					},
					crate::session::StoredProviderStateEvent::OutputItem { id, .. } => {
						output_items.push(id)
					},
					crate::session::StoredProviderStateEvent::ReasoningSignature { .. }
					| crate::session::StoredProviderStateEvent::ToolCallProof { .. }
					| crate::session::StoredProviderStateEvent::HistoryBlock { .. }
					| crate::session::StoredProviderStateEvent::Checkpoint { .. } => {},
				}
			}
			if state_continuation.is_none() {
				for id in output_items {
					input.push(ResponsesInputItem {
						kind: Some(ResponsesInputItemKind::ItemReference),
						id: Some(id),
						role: None,
						content: ResponsesInputContent::default(),
						name: None,
						call_id: None,
						arguments: None,
						input: None,
						output: None,
						summary: Vec::new(),
						encrypted_content: None,
						actions: Vec::new(),
						pending_safety_checks: Vec::new(),
						acknowledged_safety_checks: Vec::new(),
						status: None,
						tools: Vec::new(),
						cache_control: None,
						metadata: BTreeMap::new(),
					});
				}
			}
			if let Some(handle) = state_continuation {
				if continuation_id
					.as_ref()
					.is_some_and(|configured| configured != &handle)
				{
					return Err(ResponsesEncodeError::MismatchedServerState);
				}
				continuation_id = Some(handle);
			}
		}
		let start = self.options.continuation.as_ref().map_or(0, |value| {
			if value.committed_items <= request.messages.len() {
				value.committed_items
			} else {
				adjustments.push(ResponsesAdjustment::Dropped {
					field:  Str::from("previous_response_item_count"),
					reason: Str::from("continuation boundary exceeds canonical history"),
				});
				0
			}
		});
		for message in request.messages.iter().skip(start) {
			if matches!(message.role, Role::System) {
				for part in message.content.iter() {
					if let ContentPart::Text { text, proof } = part {
						if proof.is_some() {
							return Err(ResponsesEncodeError::UnreplayableProviderProof);
						}
						instructions.push(text.as_str());
					}
				}
				continue;
			}
			let role = match message.role {
				Role::System => ResponsesRole::System,
				Role::Developer => ResponsesRole::Developer,
				Role::User | Role::Tool => ResponsesRole::User,
				Role::Assistant => ResponsesRole::Assistant,
			};
			let mut content = Vec::new();
			for part in message.content.iter() {
				match part {
					ContentPart::Text { text, proof } => {
						if let Some(proof) = proof {
							if proof.provider != context.route.provider || proof.codec != target.codec {
								return Err(ResponsesEncodeError::MismatchedProviderProof);
							}
							let decoded = decode_provider_proof(&proof.value)
								.map_err(|_| ResponsesEncodeError::MismatchedProviderProof)?;
							if let Some(response) = decoded.response_id {
								continuation_id.get_or_insert(response);
							}
							if !content.is_empty() {
								input.push(ResponsesInputItem::message(role, std::mem::take(&mut content)));
							}
							let mut item =
								ResponsesInputItem::message(role, vec![ResponsesContent::input_text(
									text.clone(),
								)]);
							item.id = decoded.item_id;
							input.push(item);
						} else {
							content.push(ResponsesContent::input_text(text.clone()));
						}
					},
					ContentPart::Reasoning { text, proof } => {
						if !content.is_empty() {
							input.push(ResponsesInputItem::message(role, std::mem::take(&mut content)));
						}
						let Some(proof) = proof else {
							if !text.is_empty() {
								adjustments.push(ResponsesAdjustment::Dropped {
									field:  Str::from("reasoning_history"),
									reason: Str::from("Responses reasoning replay requires provider proof"),
								});
							}
							continue;
						};
						if proof.provider != context.route.provider || proof.codec != target.codec {
							return Err(ResponsesEncodeError::MismatchedProviderProof);
						}
						let decoded = decode_provider_proof(&proof.value)
							.map_err(|_| ResponsesEncodeError::MismatchedProviderProof)?;
						if let Some(response) = decoded.response_id {
							continuation_id.get_or_insert(response);
						}
						input.push(ResponsesInputItem {
							kind: Some(ResponsesInputItemKind::Reasoning),
							id: decoded.item_id,
							role: None,
							content: ResponsesInputContent::default(),
							name: None,
							call_id: None,
							arguments: None,
							input: None,
							output: None,
							summary: if text.is_empty() {
								Vec::new()
							} else {
								vec![ResponsesSummaryPart {
									kind: Str::from("summary_text"),
									text: text.clone(),
								}]
							},
							encrypted_content: decoded.encrypted_reasoning,
							actions: Vec::new(),
							pending_safety_checks: Vec::new(),
							acknowledged_safety_checks: Vec::new(),
							status: None,
							tools: Vec::new(),
							cache_control: None,
							metadata: BTreeMap::new(),
						});
					},
					ContentPart::Image(media) => content.push(encode_media_content(media, true)?),
					ContentPart::Document(media) => content.push(encode_media_content(media, false)?),
					ContentPart::Audio(_) => {
						return Err(ResponsesEncodeError::UnsupportedOutputFormat);
					},
					ContentPart::ToolCall { call, name, arguments, proof } => {
						if !content.is_empty() {
							input.push(ResponsesInputItem::message(role, std::mem::take(&mut content)));
						}
						let decoded = if let Some(proof) = proof {
							if proof.provider != context.route.provider || proof.codec != target.codec {
								return Err(ResponsesEncodeError::MismatchedProviderProof);
							}
							Some(
								decode_provider_proof(&proof.value)
									.map_err(|_| ResponsesEncodeError::MismatchedProviderProof)?,
							)
						} else {
							None
						};
						if let Some(response) =
							decoded.as_ref().and_then(|value| value.response_id.clone())
						{
							continuation_id.get_or_insert(response);
						}
						if let Some(computer) = decoded.as_ref().and_then(|value| value.computer.clone())
						{
							if computer.call_id.is_none() {
								return Err(ResponsesEncodeError::MissingCallIdentity);
							}
							input.push(computer);
						} else {
							let custom = decoded.as_ref().is_some_and(|value| value.custom_tool);
							let call_id = decoded
								.as_ref()
								.and_then(|value| value.call_id.clone())
								.unwrap_or_else(|| Str::from(call.as_str()));
							let serialized = serde_json::to_string(arguments.as_value())
								.map_err(|_| ResponsesEncodeError::MissingCallIdentity)?;
							input.push(ResponsesInputItem {
								kind: Some(if custom {
									ResponsesInputItemKind::CustomToolCall
								} else {
									ResponsesInputItemKind::FunctionCall
								}),
								id: decoded.and_then(|value| value.item_id),
								role: None,
								content: ResponsesInputContent::default(),
								name: Some(name.clone()),
								call_id: Some(call_id),
								arguments: (!custom).then(|| serialized.clone().into()),
								input: custom.then(|| serialized.into()),
								output: None,
								summary: Vec::new(),
								encrypted_content: None,
								actions: Vec::new(),
								pending_safety_checks: Vec::new(),
								acknowledged_safety_checks: Vec::new(),
								status: None,
								tools: Vec::new(),
								cache_control: None,
								metadata: BTreeMap::new(),
							});
						}
					},
					ContentPart::ToolResult { call, content: result, .. } => {
						if !content.is_empty() {
							input.push(ResponsesInputItem::message(role, std::mem::take(&mut content)));
						}
						let mut output = String::new();
						for (position, part) in result.iter().enumerate() {
							if position != 0 {
								output.push('\n');
							}
							match part {
								ToolResultContent::Text(text) => output.push_str(text),
								ToolResultContent::Json(value) => output.push_str(
									&serde_json::to_string(value.as_value())
										.map_err(|_| ResponsesEncodeError::MissingCallIdentity)?,
								),
								ToolResultContent::Image(_) | ToolResultContent::Document(_) => {
									return Err(ResponsesEncodeError::UnsupportedOutputFormat);
								},
							}
						}
						input.push(ResponsesInputItem {
							kind: Some(ResponsesInputItemKind::FunctionCallOutput),
							id: None,
							role: None,
							content: ResponsesInputContent::default(),
							name: None,
							call_id: Some(Str::from(call.as_str())),
							arguments: None,
							input: None,
							output: Some(ResponsesToolOutput::Text(output.into())),
							summary: Vec::new(),
							encrypted_content: None,
							actions: Vec::new(),
							pending_safety_checks: Vec::new(),
							acknowledged_safety_checks: Vec::new(),
							status: None,
							tools: Vec::new(),
							cache_control: None,
							metadata: BTreeMap::new(),
						});
					},
					ContentPart::CachePoint(retention) => {
						let kind = match retention {
							CacheRetention::Request | CacheRetention::Session | CacheRetention::Short => {
								"ephemeral"
							},
							CacheRetention::Long => "persistent",
						};
						if !content.is_empty() {
							let mut item = ResponsesInputItem::message(role, std::mem::take(&mut content));
							item.cache_control = Some(ResponsesCacheControl { kind: Str::from(kind) });
							input.push(item);
						} else if let Some(item) = input.last_mut() {
							item.cache_control = Some(ResponsesCacheControl { kind: Str::from(kind) });
						}
					},
				}
			}
			if !content.is_empty() {
				input.push(ResponsesInputItem::message(role, content));
			}
		}
		input.extend(self.options.native_input.iter().cloned());
		let instructions = if instructions.is_empty() {
			None
		} else {
			let joined = Str::from(instructions.join("\n\n"));
			if context.policy.role.supports_developer_role == Some(true) {
				let mut item = ResponsesInputItem::message(ResponsesRole::Developer, Vec::new());
				item.content = ResponsesInputContent::Text(joined);
				input.insert(0, item);
				None
			} else {
				Some(joined)
			}
		};

		let apply_patch = context.policy.tool.apply_patch;
		let mut tools = request
			.tools
			.iter()
			.map(|tool| {
				let freeform_patch =
					matches!(&tool.input, crate::call::ToolInputConstraint::JsonSchema { .. })
						&& tool.name == "apply_patch"
						&& apply_patch == Some(crate::catalog::policy::ApplyPatchWireKind::Freeform);
				let (kind, parameters, strict, format) = match &tool.input {
					crate::call::ToolInputConstraint::JsonSchema { parameters, strict }
						if !freeform_patch =>
					{
						(
							ResponsesToolKind::Function,
							Some(parameters.as_value().clone()),
							Some(*strict),
							None,
						)
					},
					crate::call::ToolInputConstraint::JsonSchema { .. } => {
						(ResponsesToolKind::Custom, None, None, None)
					},
					crate::call::ToolInputConstraint::Grammar(grammar) => (
						ResponsesToolKind::Custom,
						None,
						None,
						Some(ResponsesCustomToolFormat {
							kind:       Str::new_static("grammar"),
							syntax:     Some(Str::new_static(match grammar.syntax {
								crate::call::ToolGrammarSyntax::Lark => "lark",
								crate::call::ToolGrammarSyntax::Regex => "regex",
								crate::call::ToolGrammarSyntax::Ebnf => "ebnf",
							})),
							definition: Some(grammar.definition.clone()),
						}),
					),
				};
				ResponsesTool {
					kind,
					name: Some(tool.name.clone()),
					description: tool.description.clone(),
					parameters,
					strict,
					format,
					display_width: None,
					display_height: None,
					environment: None,
					search_context_size: None,
					allowed_domains: Vec::new(),
					blocked_domains: Vec::new(),
					vector_store_ids: Vec::new(),
					container: None,
				}
			})
			.collect::<Vec<_>>();
		for custom in &self.options.custom_tools {
			let function_patch = custom.name.as_deref() == Some("apply_patch")
				&& apply_patch == Some(crate::catalog::policy::ApplyPatchWireKind::Function);
			if function_patch {
				continue;
			}
			if let Some(name) = &custom.name {
				tools.retain(|tool| tool.name.as_ref() != Some(name));
			}
			tools.push(custom.clone());
		}
		if let Some(computer) = &self.options.computer_tool {
			if context.policy.tool.computer_use
				== Some(crate::catalog::policy::ComputerUseWireSupport::Unsupported)
			{
				return Err(ResponsesEncodeError::UnsupportedComputerUse);
			}
			let configured = computer.display_width.is_some()
				|| computer.display_height.is_some()
				|| computer.environment.is_some();
			if configured
				&& context.policy.tool.computer_use_config
					== Some(crate::catalog::policy::ComputerUseConfigSupport::Unsupported)
			{
				return Err(ResponsesEncodeError::UnsupportedComputerUseConfig);
			}
			tools.push(computer.clone());
		}
		for hosted in request.hosted_tools.iter() {
			tools.push(match hosted {
				HostedTool::WebSearch { allowed_domains, blocked_domains, recency_days } => {
					if recency_days.is_some() {
						adjustments.push(ResponsesAdjustment::Dropped {
							field:  Str::from("hosted_tools.web_search.recency_days"),
							reason: Str::from("Responses web search has no exact recency-days field"),
						});
					}
					ResponsesTool {
						kind:                ResponsesToolKind::WebSearch,
						name:                None,
						description:         None,
						parameters:          None,
						strict:              None,
						format:              None,
						display_width:       None,
						display_height:      None,
						environment:         None,
						search_context_size: None,
						allowed_domains:     allowed_domains.to_vec(),
						blocked_domains:     blocked_domains.to_vec(),
						vector_store_ids:    Vec::new(),
						container:           None,
					}
				},
				HostedTool::CodeExecution => ResponsesTool {
					kind:                ResponsesToolKind::CodeInterpreter,
					name:                None,
					description:         None,
					parameters:          None,
					strict:              None,
					format:              None,
					display_width:       None,
					display_height:      None,
					environment:         None,
					search_context_size: None,
					allowed_domains:     Vec::new(),
					blocked_domains:     Vec::new(),
					vector_store_ids:    Vec::new(),
					container:           Some(ResponsesCodeContainer { kind: Str::from("auto") }),
				},
				HostedTool::Retrieval { stores } => ResponsesTool {
					kind:                ResponsesToolKind::FileSearch,
					name:                None,
					description:         None,
					parameters:          None,
					strict:              None,
					format:              None,
					display_width:       None,
					display_height:      None,
					environment:         None,
					search_context_size: None,
					allowed_domains:     Vec::new(),
					blocked_domains:     Vec::new(),
					vector_store_ids:    stores.to_vec(),
					container:           None,
				},
			});
		}
		let tool_choice = match &request.tool_choice {
			Setting::Unset => None,
			Setting::Require(value) | Setting::Prefer(value) => Some(match value {
				ToolChoice::Disabled => ResponsesToolChoice::Mode(Str::from("none")),
				ToolChoice::Auto => ResponsesToolChoice::Mode(Str::from("auto")),
				ToolChoice::Required => ResponsesToolChoice::Mode(Str::from("required")),
				ToolChoice::Named(name) => {
					let kind = tools
						.iter()
						.find(|tool| tool.name.as_ref() == Some(name))
						.map_or(ResponsesNamedToolKind::Function, |tool| match tool.kind {
							ResponsesToolKind::Custom => ResponsesNamedToolKind::Custom,
							ResponsesToolKind::Computer => ResponsesNamedToolKind::Computer,
							_ => ResponsesNamedToolKind::Function,
						});
					ResponsesToolChoice::Named(ResponsesNamedToolChoice {
						kind,
						name: (kind != ResponsesNamedToolKind::Computer).then(|| name.clone()),
					})
				},
			}),
		};
		let reasoning =
			match &request.reasoning {
				Setting::Unset => None,
				Setting::Require(value) | Setting::Prefer(value) => {
					if value.max_tokens.is_some() {
						adjustments.push(ResponsesAdjustment::Dropped {
							field:  Str::from("reasoning.max_tokens"),
							reason: Str::from("Responses accepts qualitative effort only"),
						});
					}
					let effort = value.effort.map(|effort| match effort {
						crate::catalog::ReasoningEffort::Off => ResponsesReasoningEffort::None,
						crate::catalog::ReasoningEffort::Minimal => ResponsesReasoningEffort::Minimal,
						crate::catalog::ReasoningEffort::Low => ResponsesReasoningEffort::Low,
						crate::catalog::ReasoningEffort::Medium => ResponsesReasoningEffort::Medium,
						crate::catalog::ReasoningEffort::High => ResponsesReasoningEffort::High,
						crate::catalog::ReasoningEffort::Xhigh | crate::catalog::ReasoningEffort::Max => {
							ResponsesReasoningEffort::Xhigh
						},
					});
					Some(ResponsesReasoning {
						effort,
						summary: self.options.reasoning_summary.clone().or_else(|| {
							match value.visibility {
								ReasoningVisibility::Hidden => Some(None),
								ReasoningVisibility::Summary | ReasoningVisibility::Visible => {
									Some(Some(Str::from("auto")))
								},
							}
						}),
						mode: self.options.reasoning_mode.clone(),
						context: None,
					})
				},
			};
		let text = {
			let verbosity = match &request.verbosity {
				Setting::Unset => None,
				Setting::Require(value) | Setting::Prefer(value) => Some(Str::from(match value {
					TextVerbosity::Low => "low",
					TextVerbosity::Medium => "medium",
					TextVerbosity::High => "high",
				})),
			};
			let format = match &request.output {
				Setting::Unset => None,
				Setting::Require(value) | Setting::Prefer(value) => Some(match value {
					StructuredOutput::JsonObject => ResponsesTextFormat {
						kind:   ResponsesTextFormatKind::JsonObject,
						name:   None,
						schema: None,
						strict: None,
					},
					StructuredOutput::JsonSchema { name, schema, strict } => ResponsesTextFormat {
						kind:   ResponsesTextFormatKind::JsonSchema,
						name:   Some(name.clone()),
						schema: Some(schema.as_value().clone()),
						strict: Some(*strict),
					},
					StructuredOutput::Regex(_)
					| StructuredOutput::Lark(_)
					| StructuredOutput::Ebnf(_) => return Err(ResponsesEncodeError::UnsupportedOutputFormat),
				}),
			};
			(verbosity.is_some() || format.is_some())
				.then_some(ResponsesTextOptions { verbosity, format })
		};
		let prompt_cache_retention =
			self
				.options
				.prompt_cache_retention
				.clone()
				.or_else(|| match &request.cache_retention {
					Setting::Unset => None,
					Setting::Require(CacheRetention::Long) | Setting::Prefer(CacheRetention::Long) => {
						Some(Str::from("24h"))
					},
					Setting::Require(_) | Setting::Prefer(_) => Some(Str::from("in_memory")),
				});
		let service_tier = match &request.service_tier {
			Setting::Unset => None,
			Setting::Require(value) | Setting::Prefer(value) => Some(value.name.clone()),
		};
		if request.sampling.top_k.is_some() {
			adjustments.push(ResponsesAdjustment::Dropped {
				field:  Str::from("sampling.top_k"),
				reason: Str::from("Responses has no top-k field"),
			});
		}
		if request.sampling.seed.is_some() {
			adjustments.push(ResponsesAdjustment::Dropped {
				field:  Str::from("sampling.seed"),
				reason: Str::from("Responses has no deterministic seed field"),
			});
		}
		if !request.sampling.stop.is_empty() {
			adjustments.push(ResponsesAdjustment::Dropped {
				field:  Str::from("sampling.stop"),
				reason: Str::from("Responses has no stop-sequence field"),
			});
		}
		if request.top_logprobs.is_some() {
			adjustments.push(ResponsesAdjustment::Dropped {
				field:  Str::from("top_logprobs"),
				reason: Str::from("Responses streaming projection does not expose logprobs"),
			});
		}
		if !request.safety.is_empty() {
			adjustments.push(ResponsesAdjustment::Dropped {
				field:  Str::from("safety"),
				reason: Str::from("Responses has no per-request safety thresholds"),
			});
		}
		let include = {
			let mut values = self.options.include.clone();
			if matches!(&request.reasoning, Setting::Require(value) | Setting::Prefer(value) if value.preserve_signatures)
				&& !values
					.iter()
					.any(|value| value == "reasoning.encrypted_content")
			{
				values.push(Str::from("reasoning.encrypted_content"));
			}
			values
		};
		let previous_response_id = if self.options.stateful {
			continuation_id
		} else {
			if continuation_id.is_some() {
				adjustments.push(ResponsesAdjustment::Dropped {
					field:  Str::from("previous_response_id"),
					reason: Str::from("stateful Responses is disabled"),
				});
			}
			None
		};
		Ok(EncodedResponses {
			request: ResponsesRequest {
				model: Str::from(target.wire_model.as_str()),
				input,
				stream: true,
				store: self.options.stateful,
				instructions,
				previous_response_id,
				prompt_cache_key: self.options.prompt_cache_key.clone(),
				prompt_cache_retention,
				include,
				tools,
				additional_tools: Vec::new(),
				tool_choice,
				parallel_tool_calls: self.options.parallel_tool_calls,
				reasoning,
				text,
				temperature: request.sampling.temperature,
				top_p: request.sampling.top_p,
				presence_penalty: request.sampling.presence_penalty,
				frequency_penalty: request.sampling.frequency_penalty,
				max_output_tokens: request.max_output_tokens,
				service_tier,
				metadata: self.options.metadata.clone(),
				client_metadata: None,
				stream_options: None,
			},
			adjustments,
		})
	}
}

fn encode_media_content(
	media: &crate::call::MediaInput,
	image: bool,
) -> Result<ResponsesContent, ResponsesEncodeError> {
	use crate::call::MediaInput;
	let kind = if image {
		ResponsesContentKind::InputImage
	} else {
		ResponsesContentKind::InputFile
	};
	match media {
		MediaInput::Bytes { media_type, data } => {
			let encoded = omp_core::encoding::base64::encode(data).into_string();
			let url = Str::from(format!("data:{media_type};base64,{encoded}"));
			Ok(ResponsesContent {
				kind,
				text: None,
				image_url: image.then(|| url.clone()),
				detail: None,
				file_data: (!image).then_some(url),
				file_url: None,
				filename: None,
				file_id: None,
			})
		},
		MediaInput::Remote { uri, name, .. } => Ok(ResponsesContent {
			kind,
			text: None,
			image_url: image.then(|| uri.clone()),
			detail: None,
			file_data: None,
			file_url: (!image).then(|| uri.clone()),
			filename: (!image).then(|| name.clone()).flatten(),
			file_id: None,
		}),
		MediaInput::Stored(_) | MediaInput::Body { .. } => {
			Err(ResponsesEncodeError::UnresolvedStoredMedia)
		},
	}
}

#[derive(Serialize)]
struct HostedCheckpoint {
	index:     u32,
	kind:      ResponsesOutputItemKind,
	completed: bool,
}

#[derive(Serialize)]
struct ContinuationCheckpoint<'a> {
	response_id:  &'a Str,
	model:        Option<&'a Str>,
	service_tier: Option<&'a Str>,
}

struct ResponsesDecoderAdapter {
	inner:      OpenAiResponsesDecoder,
	request_id: crate::id::RequestId,
	provider:   crate::catalog::ProviderId,
	route:      crate::catalog::RouteId,
	wire_model: Option<Str>,
}

impl ResponsesDecoderAdapter {
	fn emit_projection(
		&self,
		projection: ResponsesProjection,
		emit: &mut dyn FnMut(super::RawEvent),
	) {
		match projection {
			ResponsesProjection::Canonical(event) => emit(super::RawEvent::Chat(event)),
			ResponsesProjection::Completion(completion) => {
				emit(super::RawEvent::Completion(completion))
			},
			ResponsesProjection::ToolCallComplete { index, id, name, arguments, custom } => {
				emit(super::RawEvent::ToolCallComplete {
					index,
					call: super::UnvalidatedToolCall {
						id,
						name,
						input_kind: if custom {
							super::ToolInputKind::Freeform
						} else {
							super::ToolInputKind::Json
						},
						arguments,
					},
				});
			},
			ResponsesProjection::OutputItem { index, id } => {
				emit(super::RawEvent::ProviderState(super::ProviderStateEvent::OutputItem {
					index,
					id,
				}))
			},
			ResponsesProjection::ReasoningSignature { index, item_id, signature } => {
				if let Some(id) = item_id {
					emit(super::RawEvent::ProviderState(super::ProviderStateEvent::OutputItem {
						index,
						id,
					}));
				}
				emit(super::RawEvent::ProviderState(super::ProviderStateEvent::ReasoningSignature {
					index,
					signature,
				}));
			},
			ResponsesProjection::HostedTool { index, kind, completed } => {
				let data = serde_json::to_vec(&HostedCheckpoint { index, kind, completed })
					.map_or_else(|_| Bytes::new(), Bytes::from);
				emit(super::RawEvent::ProviderState(super::ProviderStateEvent::Checkpoint {
					id: None,
					data,
				}));
			},
			ResponsesProjection::Continuation { response_id, model, service_tier } => {
				emit(super::RawEvent::ProviderState(super::ProviderStateEvent::Continuation {
					handle: response_id.clone(),
				}));
				let data = serde_json::to_vec(&ContinuationCheckpoint {
					response_id:  &response_id,
					model:        self.wire_model.as_ref().or(model.as_ref()),
					service_tier: service_tier.as_ref(),
				})
				.map_or_else(|_| Bytes::new(), Bytes::from);
				emit(super::RawEvent::ProviderState(super::ProviderStateEvent::Checkpoint {
					id: Some(response_id),
					data,
				}));
			},
			ResponsesProjection::Error(evidence) => {
				emit(super::RawEvent::Failure(self.error_from_evidence(evidence)));
			},
		}
	}

	fn error_from_evidence(&self, evidence: ResponsesErrorEvidence) -> crate::error::Error {
		use crate::{
			error::{Error, ErrorKind, ErrorPhase, RetryAction},
			receipt::ExecutionReceipt,
		};
		let (kind, action) = match evidence.continuation {
			ResponsesContinuationFailure::StalePreviousResponse
			| ResponsesContinuationFailure::StaleServerItem => {
				(ErrorKind::SessionExpired, RetryAction::ReseedSession)
			},
			ResponsesContinuationFailure::Malformed => {
				(ErrorKind::StreamCorruption, RetryAction::Never)
			},
			ResponsesContinuationFailure::NotStale => (ErrorKind::Protocol, RetryAction::Never),
		};
		let mut error = Error::new(kind, ErrorPhase::Streaming, action, ExecutionReceipt::default());
		error.provider = Some(self.provider.clone());
		error.route = Some(self.route.clone());
		error.request_id = Some(self.request_id.clone());
		error.code = evidence.code;
		error.committed = self.inner.committed_output();
		error
	}
}

impl super::Decoder for ResponsesDecoderAdapter {
	fn push(
		&mut self,
		frame: crate::transport::Frame,
		emit: &mut dyn FnMut(super::RawEvent),
	) -> Result<(), crate::error::Error> {
		use crate::transport::{Frame, WebSocketMessage};
		let payload = match frame {
			Frame::Raw(payload) | Frame::Ndjson(payload) => payload,
			Frame::Sse(event) => event.data,
			Frame::WebSocket(WebSocketMessage::Text(payload) | WebSocketMessage::Binary(payload)) => {
				payload
			},
			Frame::WebSocket(
				WebSocketMessage::Close { .. } | WebSocketMessage::Ping(_) | WebSocketMessage::Pong(_),
			) => return Ok(()),
			Frame::Connect(_) | Frame::EventStream(_) => {
				return Err(self.error_from_evidence(ResponsesErrorEvidence {
					code:         Some(Str::from("wrong_framing_protocol")),
					message:      Str::from("Responses decoder received incompatible framing"),
					continuation: ResponsesContinuationFailure::Malformed,
				}));
			},
		};
		for projection in self.inner.push_json(&payload) {
			self.emit_projection(projection, emit);
		}
		Ok(())
	}

	fn finish(&mut self, emit: &mut dyn FnMut(super::RawEvent)) -> Result<(), crate::error::Error> {
		for projection in self.inner.finish() {
			self.emit_projection(projection, emit);
		}
		Ok(())
	}
}

fn encoding_error(code: &'static str) -> crate::error::Error {
	use crate::{
		error::{Error, ErrorKind, ErrorPhase, RetryAction},
		receipt::ExecutionReceipt,
	};
	let mut error = Error::new(
		ErrorKind::InvalidRequest,
		ErrorPhase::Encoding,
		RetryAction::Never,
		ExecutionReceipt::default(),
	);
	error.code = Some(Str::from(code));
	error
}

fn responses_uri(base_url: &str) -> Str {
	let base = base_url.trim_end_matches('/');
	if base.ends_with("/responses") {
		Str::from(base)
	} else if base.ends_with("/v1") {
		Str::from(format!("{base}/responses"))
	} else {
		Str::from(format!("{base}/v1/responses"))
	}
}

impl super::Codec for OpenAiResponsesCodec {
	fn encode(
		&self,
		context: &super::EncodeContext<'_>,
		operation: &crate::call::OperationCall,
	) -> Result<super::EncodedRequest, crate::error::Error> {
		let crate::call::OperationCall::Chat(request) = operation else {
			return Err(encoding_error("responses_chat_only"));
		};
		let encoded = self
			.encode_chat(context, request)
			.map_err(|error| match error {
				ResponsesEncodeError::MismatchedProviderProof => {
					encoding_error("mismatched_responses_provider_proof")
				},
				ResponsesEncodeError::MissingWireTarget => {
					encoding_error("missing_responses_wire_target")
				},
				ResponsesEncodeError::MissingCallIdentity => {
					encoding_error("missing_responses_call_identity")
				},
				ResponsesEncodeError::UnresolvedStoredMedia => {
					encoding_error("unresolved_responses_media")
				},
				ResponsesEncodeError::UnreplayableProviderProof => {
					encoding_error("unreplayable_responses_provider_proof")
				},
				ResponsesEncodeError::UnsupportedOutputFormat => {
					encoding_error("unsupported_responses_output_format")
				},
				ResponsesEncodeError::UnsupportedComputerUse => {
					encoding_error("unsupported_responses_computer_use")
				},
				ResponsesEncodeError::UnsupportedComputerUseConfig => {
					encoding_error("unsupported_responses_computer_use_config")
				},
				ResponsesEncodeError::MalformedServerState => {
					encoding_error("malformed_responses_server_state")
				},
				ResponsesEncodeError::MismatchedServerState => {
					encoding_error("mismatched_responses_server_state")
				},
			})?;
		if !encoded.adjustments.is_empty() {
			return Err(encoding_error("responses_adjustment_requires_planning"));
		}
		let body = serde_json::to_vec(&encoded.request)
			.map(Bytes::from)
			.map_err(|_| encoding_error("responses_request_serialization"))?;
		Ok(super::EncodedRequest {
			operation:   crate::catalog::OperationKind::Chat,
			method:      super::RequestMethod::Post,
			uri:         responses_uri(
				context
					.target
					.expect("chat encoding checked the wire target")
					.endpoint
					.base_url
					.as_str(),
			),
			headers:     vec![
				super::RequestHeader {
					name:  Str::from("content-type"),
					value: Str::from("application/json"),
				},
				super::RequestHeader {
					name:  Str::from("accept"),
					value: Str::from("text/event-stream"),
				},
			]
			.into_boxed_slice(),
			body:        crate::body::BodySource::Bytes(body),
			framing:     crate::transport::FramingProtocol::Sse,
			bounds:      super::SizeBounds {
				request_body: 64 * 1024 * 1024,
				frame:        16 * 1024 * 1024,
				response:     256 * 1024 * 1024,
			},
			sealed_body: None,
		})
	}

	fn decoder(
		&self,
		context: &super::DecodeContext<'_>,
	) -> Result<super::DecoderState, crate::error::Error> {
		if context.operation != context.operation_call.kind()
			|| !matches!(context.operation_call, crate::call::OperationCall::Chat(_))
		{
			return Err(encoding_error("responses_decode_operation_mismatch"));
		}
		Ok(Box::new(ResponsesDecoderAdapter {
			inner:      OpenAiResponsesDecoder::default(),
			request_id: context.request_id.clone(),
			provider:   context.provider.clone(),
			route:      context.route.clone(),
			wire_model: context
				.target
				.map(|target| Str::from(target.wire_model.as_str())),
		}))
	}
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use omp_core::Str;
	use omp_llm_catalog::{Catalog, WireTarget};

	use super::{
		OpenAiResponsesCodec, OpenAiResponsesDecoder, ResponsesContinuationFailure,
		ResponsesProjection, classify_continuation_error,
	};
	use crate::{
		call::{
			ChatRequest, NegotiationPolicy, OpaqueJson, Sampling, Setting, ToolDefinition,
			ToolGrammar, ToolGrammarSyntax, ToolInputConstraint,
		},
		codec::{EncodeAttempt, EncodeContext},
		event::{ChatEvent, FinishReason},
		id::RequestId,
	};

	fn replay_sse(fixture: &str) -> Vec<ResponsesProjection> {
		let mut decoder = OpenAiResponsesDecoder::default();
		let mut events = Vec::new();
		for block in fixture.split("\n\n") {
			for line in block.lines() {
				if let Some(data) = line.strip_prefix("data: ") {
					events.extend(decoder.push_json(data.as_bytes()));
				}
			}
		}
		events.extend(decoder.finish());
		events
	}

	fn request_with_tool(input: ToolInputConstraint) -> ChatRequest {
		ChatRequest {
			messages:          Arc::from([]),
			tools:             Arc::from([ToolDefinition {
				name: Str::new_static("match_input"),
				description: None,
				input,
			}]),
			hosted_tools:      Arc::from([]),
			tool_choice:       Setting::Unset,
			output:            Setting::Unset,
			reasoning:         Setting::Unset,
			verbosity:         Setting::Unset,
			cache_retention:   Setting::Unset,
			service_tier:      Setting::Unset,
			sampling:          Sampling::default(),
			max_output_tokens: None,
			top_logprobs:      None,
			safety:            Arc::from([]),
			negotiation:       NegotiationPolicy::default(),
		}
	}

	fn encode_tool(input: ToolInputConstraint) -> Vec<u8> {
		let catalog = Catalog::embedded();
		let model = catalog
			.models()
			.iter()
			.find(|model| {
				model.routes.iter().any(|route| {
					catalog
						.route(route)
						.is_some_and(|route| route.codec.as_str() == "openai-responses")
				})
			})
			.expect("embedded Responses model");
		let route = model
			.routes
			.iter()
			.filter_map(|route| catalog.route(route))
			.find(|route| route.codec.as_str() == "openai-responses")
			.expect("embedded Responses route");
		let wire_model = model
			.wire_ids
			.iter()
			.find(|(candidate, _)| candidate == &route.id)
			.expect("embedded Responses wire model")
			.1
			.clone();
		let target = WireTarget {
			route: route.id.clone(),
			codec: route.codec.clone(),
			endpoint: route.endpoint.clone(),
			wire_model,
		};
		let policy = catalog
			.wire_policy(&model.wire_policy)
			.expect("embedded Responses wire policy");
		let request_id = RequestId::new("responses-tool-encoding");
		let context = EncodeContext {
			request_id: &request_id,
			route,
			target: Some(&target),
			policy_model: None,
			policy,
			thinking_policy: None,
			thinking_selection: None,
			session: None,
			server_state: None,
			account: None,
			attempt: EncodeAttempt { index: 0, provisional: false },
		};
		let encoded = OpenAiResponsesCodec::default()
			.encode_chat(&context, &request_with_tool(input))
			.expect("tool request encodes");
		serde_json::to_vec(&encoded.request.tools).expect("tools serialize")
	}

	#[test]
	fn custom_tool_grammars_preserve_exact_syntax_and_definition_on_wire() {
		let cases: [(ToolGrammarSyntax, &'static str, &'static [u8]); 3] = [
			(
				ToolGrammarSyntax::Regex,
				"[a-z]+",
				br#"[{"type":"custom","name":"match_input","format":{"type":"grammar","syntax":"regex","definition":"[a-z]+"}}]"#,
			),
			(
				ToolGrammarSyntax::Lark,
				"start: WORD\n%import common.WORD",
				br#"[{"type":"custom","name":"match_input","format":{"type":"grammar","syntax":"lark","definition":"start: WORD\n%import common.WORD"}}]"#,
			),
			(
				ToolGrammarSyntax::Ebnf,
				r#"root = "yes" | "no";"#,
				br#"[{"type":"custom","name":"match_input","format":{"type":"grammar","syntax":"ebnf","definition":"root = \"yes\" | \"no\";"}}]"#,
			),
		];
		for (syntax, definition, expected) in cases {
			assert_eq!(
				encode_tool(ToolInputConstraint::Grammar(ToolGrammar {
					syntax,
					definition: Str::from(definition),
				})),
				expected,
			);
		}
	}

	#[test]
	fn json_schema_tool_encoding_remains_a_strict_function_tool() {
		assert_eq!(
			encode_tool(ToolInputConstraint::JsonSchema {
				parameters: OpaqueJson::new(serde_json::json!({"type": "object"})),
				strict: true,
			}),
			br#"[{"type":"function","name":"match_input","parameters":{"type":"object"},"strict":true}]"#,
		);
	}

	#[test]
	fn replays_encrypted_reasoning_tool_and_usage_fixture() {
		let events = replay_sse(include_str!(
			"../../../../fixtures/llm-oracle/openai/responses/stream.encrypted_tool_usage.sse"
		));
		assert!(events.iter().any(|event| matches!(
			event,
			ResponsesProjection::Canonical(ChatEvent::ThinkingDelta { text, .. })
				if text == "Inspect first."
		)));
		assert!(events.iter().any(|event| matches!(
			event,
			ResponsesProjection::ReasoningSignature { signature, .. }
				if signature == b"enc_REDACTED".as_slice()
		)));
		assert!(events.iter().any(|event| matches!(
			event,
			ResponsesProjection::ToolCallComplete { name, arguments, custom: false, .. }
				if name == "read" && arguments == br#"{"path":"README.md"}"#.as_slice()
		)));
		assert!(!events.iter().any(|event| matches!(
			event,
			ResponsesProjection::Canonical(ChatEvent::ToolCallReady { .. })
		)));
		assert!(events.iter().any(|event| matches!(
			event,
			ResponsesProjection::Canonical(ChatEvent::Usage(update))
				if update.usage.input_tokens == 30
					&& update.usage.output_tokens == 8
					&& update.usage.cache_read_tokens == 20
		)));
		assert!(events.iter().any(|event| matches!(
			event,
			ResponsesProjection::Completion(completion)
				if completion.reason == FinishReason::ToolCalls
		)));
	}

	#[test]
	fn replays_hosted_tool_ordering_and_reasoning_usage_fixture() {
		let events = replay_sse(include_str!(
			"../../../../fixtures/llm-oracle/openai/responses/stream.server_tools_ordering.sse"
		));
		let text = events
			.iter()
			.position(|event| {
				matches!(
					event,
					ResponsesProjection::Canonical(ChatEvent::TextDelta { text, .. }) if text == "Found it."
				)
			})
			.expect("text delta");
		let usage = events
			.iter()
			.position(|event| {
				matches!(
					event,
					ResponsesProjection::Canonical(ChatEvent::Usage(update))
						if update.usage.reasoning_tokens == 7 && update.usage.cache_read_tokens == 32
				)
			})
			.expect("usage");
		let complete = events
			.iter()
			.position(|event| {
				matches!(
					event,
					ResponsesProjection::Completion(completion)
						if completion.reason == FinishReason::Stop
				)
			})
			.expect("completion");
		assert!(text < usage && usage < complete);
	}

	#[test]
	fn custom_tool_input_remains_freeform_and_unvalidated() {
		let mut decoder = OpenAiResponsesDecoder::default();
		let mut events = decoder.push_json(br#"{"type":"response.output_item.added","output_index":0,"item":{"type":"custom_tool_call","id":"ct_1","call_id":"call_1","name":"shell","input":""}}"#);
		events.extend(decoder.push_json(br#"{"type":"response.custom_tool_call_input.delta","output_index":0,"item_id":"ct_1","delta":"cat README.md"}"#));
		events.extend(decoder.push_json(br#"{"type":"response.output_item.done","output_index":0,"item":{"type":"custom_tool_call","id":"ct_1","call_id":"call_1","name":"shell","input":"cat README.md"}}"#));
		assert!(events.iter().any(|event| matches!(
			event,
			ResponsesProjection::ToolCallComplete { name, arguments, custom: true, .. }
				if name == "shell" && arguments == b"cat README.md".as_slice()
		)));
		assert!(!events.iter().any(|event| matches!(
			event,
			ResponsesProjection::Canonical(ChatEvent::ToolCallReady { .. })
		)));
	}

	#[test]
	fn preserves_leaked_tags_as_visible_text_for_recovery() {
		let mut decoder = OpenAiResponsesDecoder::default();
		let mut events = decoder.push_json(br#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}"#);
		events.extend(decoder.push_json(br#"{"type":"response.output_text.delta","output_index":0,"item_id":"msg_1","delta":"<think>leaked</think>answer"}"#));
		assert!(events.iter().any(|event| matches!(
			event,
			ResponsesProjection::Canonical(ChatEvent::TextDelta { text, .. })
				if text == "<think>leaked</think>answer"
		)));
	}

	#[test]
	fn continuation_recovery_requires_exact_typed_evidence() {
		let stale = classify_continuation_error(
			400,
			br#"{"error":{"code":"previous_response_not_found","message":"Previous response expired."}}"#,
		);
		assert_eq!(stale.continuation, ResponsesContinuationFailure::StalePreviousResponse);
		let orphan = classify_continuation_error(
			404,
			br#"{"error":{"type":"invalid_request_error","message":"Item with id 'fc_server_stale' not found."}}"#,
		);
		assert_eq!(orphan.continuation, ResponsesContinuationFailure::StaleServerItem);
		let unrelated = classify_continuation_error(
			400,
			br#"{"error":{"code":"invalid_request_error","message":"The request schema is invalid."}}"#,
		);
		assert_eq!(unrelated.continuation, ResponsesContinuationFailure::NotStale);
		let malformed = classify_continuation_error(400, b"{not-json previous response words only");
		assert_eq!(malformed.continuation, ResponsesContinuationFailure::Malformed);
	}

	#[test]
	fn malformed_and_post_terminal_frames_are_bounded() {
		let mut decoder = OpenAiResponsesDecoder::default();
		let malformed = decoder.push_json(b"{");
		assert!(matches!(malformed.as_slice(), [ResponsesProjection::Error(_)]));
		assert!(
			decoder
				.push_json(br#"{"type":"response.created","response":{"id":"late"}}"#)
				.is_empty()
		);
		assert!(decoder.finish().is_empty());
	}
}
