//! Data-selected provider compatibility axes.

use bon::Builder;
use serde::{Deserialize, Serialize};

/// Provider behavior which differs while retaining the same high-level
/// transport.
///
/// Every field is a single independent wire variance axis. Defaults describe a
/// conventional `OpenAI` Chat Completions endpoint.
#[non_exhaustive]
#[derive(Builder, Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub struct Compat {
	/// Whether streaming usage may be requested; Cerebras rejects
	/// `include_usage`.
	pub usage_in_streaming:            bool,
	/// Whether separate system messages are accepted; Ollama requires
	/// coalescing.
	pub multiple_system_messages:      bool,
	/// Output-token field spelling; Mistral requires `max_tokens`.
	pub max_tokens_field:              MaxTokensField,
	/// Whether temperature and top-p are accepted; `OpenAI` o1 models reject
	/// them.
	pub sampling_params:               bool,
	/// Whether frequency/presence penalties are accepted; `OpenAI` o1 rejects
	/// them.
	pub penalties:                     bool,
	/// Strict-schema policy; Cerebras requires every tool to be strict.
	pub tool_strict_mode:              ToolStrictMode,
	/// Whether object-form named tool choice works; llama.cpp accepts strings
	/// only.
	pub named_tool_choice:             bool,
	/// Whether a tool may be forced; `DeepSeek` reasoning models reject forcing.
	pub forced_tool_choice:            bool,
	/// Tool-call identifier projection; Mistral requires exactly nine
	/// alphanumeric characters.
	pub tool_call_id_profile:          ToolCallIdProfile,
	/// Reasoning request/history representation; Z.AI uses a `thinking` object.
	pub reasoning_wire_format:         ReasoningWireFormat,
	/// Whether Responses accepts `previous_response_id`; official `OpenAI` does,
	/// third-party proxies often reject it.
	pub stateful_response_chaining:    bool,
	/// Resolution when thinking and tool choice conflict; Fireworks rejects both
	/// knobs.
	pub thinking_tool_choice_conflict: ThinkingToolChoiceConflict,
	/// Prompt-cache marker representation; `OpenRouter` Anthropic models use
	/// cache-control parts.
	pub cache_control_format:          CacheControlFormat,
	/// Image payload representation; Anthropic expects source blocks rather than
	/// image URLs.
	pub image_encoding_format:         ImageEncodingFormat,
	/// Whether stop sequences are accepted; `OpenAI` reasoning models can reject
	/// `stop`.
	pub stop_sequences:                bool,
	/// Tool JSON-schema normalization; local llama.cpp commonly needs
	/// grammar-safe schemas.
	pub tool_schema_flavor:            ToolSchemaFlavor,
	/// Leaked-thinking repair applicability; DeepSeek-compatible streams can
	/// leak `<think>`.
	pub leaked_thinking_healer:        LeakedThinkingHealer,
	/// Repeated-thinking / zero-progress detection; Gemini models stall in
	/// reasoning loops.
	pub thinking_loop_guard:           bool,
	/// Provider-specific first-event and inter-event timeout policy.
	pub stream_watchdog:               StreamWatchdog,
	/// Framing and terminal-event rules; Ollama's native API is NDJSON rather
	/// than SSE.
	pub stream_protocol:               StreamProtocol,
	/// API-version query required by otherwise OpenAI-compatible audio routes.
	pub audio_api_version:             AudioApiVersion,
}

impl Default for Compat {
	fn default() -> Self {
		Self {
			usage_in_streaming:            true,
			multiple_system_messages:      true,
			max_tokens_field:              MaxTokensField::MaxCompletionTokens,
			sampling_params:               true,
			penalties:                     true,
			tool_strict_mode:              ToolStrictMode::Mixed,
			named_tool_choice:             true,
			forced_tool_choice:            true,
			tool_call_id_profile:          ToolCallIdProfile::Unconstrained,
			reasoning_wire_format:         ReasoningWireFormat::OpenAi,
			stateful_response_chaining:    false,
			thinking_tool_choice_conflict: ThinkingToolChoiceConflict::None,
			cache_control_format:          CacheControlFormat::None,
			image_encoding_format:         ImageEncodingFormat::OpenAiUrl,
			stop_sequences:                true,
			tool_schema_flavor:            ToolSchemaFlavor::JsonSchema,
			leaked_thinking_healer:        LeakedThinkingHealer::None,
			thinking_loop_guard:           false,
			stream_watchdog:               StreamWatchdog::default(),
			stream_protocol:               StreamProtocol::SseData,
			audio_api_version:             AudioApiVersion::None,
		}
	}
}

/// API-version suffix for OpenAI-compatible audio endpoints.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioApiVersion {
	/// The endpoint does not require an API-version query.
	#[default]
	None,
	/// Azure OpenAI's April 2025 preview audio contract.
	#[serde(rename = "2025-04-01-preview")]
	V2025_04_01Preview,
}

/// Name of the Chat Completions output-token field.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MaxTokensField {
	/// `max_tokens`, required by Mistral and direct `DeepSeek`.
	MaxTokens,
	/// `max_completion_tokens`, used by current `OpenAI` Chat Completions.
	#[default]
	MaxCompletionTokens,
	/// `max_output_tokens`, used by `OpenAI` Responses.
	MaxOutputTokens,
}

/// How strictness is emitted on tool definitions.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolStrictMode {
	/// Force strict mode on every tool, as Cerebras requires.
	AllStrict,
	/// Honor each tool's requested strictness, as `OpenAI` does.
	#[default]
	Mixed,
	/// Never emit strictness, as Groq's older endpoints require.
	None,
}

/// Provider constraints on tool-call identifiers.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallIdProfile {
	/// Preserve the canonical identifier, for endpoints without a documented
	/// limit.
	#[default]
	Unconstrained,
	/// Emit an OpenAI-compatible identifier of at most 40 characters.
	OpenAi40,
	/// Emit exactly nine ASCII alphanumeric characters, as Mistral requires.
	Mistral9Alnum,
}

/// Provider-native reasoning controls and history fields.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningWireFormat {
	/// No provider-native reasoning fields, suitable for basic local servers.
	None,
	/// `OpenAI` `reasoning_effort` / `reasoning_content` fields.
	#[default]
	OpenAi,
	/// `OpenAI` Responses `reasoning` object and reasoning items.
	OpenAiResponses,
	/// Anthropic thinking budget/adaptive blocks.
	Anthropic,
	/// Google `thinkingConfig` controls and thought parts.
	Google,
	/// `OpenRouter`'s nested `reasoning` object.
	OpenRouter,
	/// Z.AI's `thinking: { type = ... }` object.
	Zai,
	/// Qwen Chat Completions `enable_thinking` boolean.
	QwenEnableThinking,
	/// NVIDIA NIM `chat_template_kwargs: { enable_thinking }` object.
	NvidiaChatTemplateKwargs,
}

/// Policy when reasoning controls cannot coexist with tool choice.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingToolChoiceConflict {
	/// Both controls may be sent, as on standard `OpenAI` endpoints.
	#[default]
	None,
	/// Remove thinking only for forced choice, as Kimi requires.
	DropThinkingWhenForced,
	/// Remove thinking for any choice, as direct `DeepSeek` reasoning requires.
	DropThinkingWhenAny,
	/// Remove the thinking object when effort is present, as Fireworks requires.
	DropThinkingWhenEffort,
}

/// Prompt-cache breakpoint encoding.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheControlFormat {
	/// No explicit cache-control markers, used by most OpenAI-compatible hosts.
	#[default]
	None,
	/// Anthropic `cache_control` content-part markers.
	Anthropic,
	/// `OpenAI` prompt-cache retention controls.
	OpenAi,
	/// Google explicit cached-content resource names.
	Google,
}

/// Encoding of image inputs in message parts.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageEncodingFormat {
	/// `OpenAI` `image_url` data URLs.
	#[default]
	OpenAiUrl,
	/// Anthropic base64 `source` blocks.
	AnthropicSource,
	/// Google inline-data parts.
	GoogleInlineData,
	/// Images are unsupported, as on some text-only local servers.
	None,
}

/// Normalization applied to tool parameter schemas.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSchemaFlavor {
	/// Ordinary JSON Schema, accepted by `OpenAI`.
	#[default]
	JsonSchema,
	/// Anthropic's supported JSON-schema subset.
	Anthropic,
	/// Google's function declaration schema subset.
	Google,
	/// Moonshot/Kimi MFJS normalization.
	MoonshotMfjs,
	/// Grammar-safe schema conversion for llama.cpp servers.
	Grammar,
	/// Cloud Code Assist schema stripping.
	Cca,
}

/// Streaming markup repair policy.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LeakedThinkingHealer {
	/// No repair, appropriate for official `OpenAI` streams.
	#[default]
	None,
	/// Generic `<think>` markup repair used by local reasoning models.
	Thinking,
	/// Kimi/Moonshot markup repair.
	Kimi,
	/// `DeepSeek` DSML token repair.
	Dsml,
}

/// Provider-specific stream timeout bounds.
///
/// The previous implementation used roughly 300–600 second idle bounds. Those
/// bounds belong in each provider row because the right timeout is
/// provider-specific.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub struct StreamWatchdog {
	/// Maximum delay before the first stream event, in milliseconds; `None`
	/// disables it.
	pub first_event_ms: Option<u64>,
	/// Maximum delay between stream events, in milliseconds; `None` disables it.
	pub idle_ms:        Option<u64>,
}

/// Provider stream framing and completion convention.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamProtocol {
	/// SSE `data:` records with a sentinel, used by `OpenAI` Chat Completions.
	#[default]
	SseData,
	/// Typed SSE events, used by Anthropic and `OpenAI` Responses.
	SseEvents,
	/// Newline-delimited JSON, used by native Ollama.
	Ndjson,
	/// Connect/gRPC framing, used by Cursor and Devin.
	Connect,
}

#[cfg(test)]
mod tests {
	use super::{AudioApiVersion, ReasoningWireFormat, ToolCallIdProfile};
	use crate::provider::{BUILTIN_PROVIDERS_TOML, load_providers};

	#[test]
	fn shipped_provider_rows_select_compatibility_axes() {
		let providers =
			load_providers(BUILTIN_PROVIDERS_TOML).expect("shipped providers.toml must parse");

		assert_eq!(
			providers["mistral"].compat.tool_call_id_profile,
			ToolCallIdProfile::Mistral9Alnum
		);
		assert_eq!(
			providers["qwen-portal"].compat.reasoning_wire_format,
			ReasoningWireFormat::QwenEnableThinking
		);
		assert_eq!(
			providers["nvidia"].compat.reasoning_wire_format,
			ReasoningWireFormat::NvidiaChatTemplateKwargs
		);
		assert!(providers["openai"].compat.stateful_response_chaining);
		assert_eq!(providers["azure"].compat.audio_api_version, AudioApiVersion::V2025_04_01Preview);
		assert!(!providers["openai-codex"].compat.stateful_response_chaining);
	}
}
