use std::{collections::BTreeMap, sync::Arc};

use bon::Builder;
use bytes::Bytes;
use omp_core::Str;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use super::{Props, Thread};
/// Server-resolved, transport-neutral behavior for one catalog model.
///
/// This policy is attached only after trusted catalog resolution. It is never
/// encoded in protobuf or accepted from clients. Requests share it through an
/// [`Arc`], so cloning a request keeps policy cloning O(1).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResolvedModelPolicy {
	/// Model id sent to the provider when it differs from the logical catalog
	/// id.
	pub request_model_id:       Option<Str>,
	/// Native reasoning controls and effort routing.
	pub thinking:               Option<ResolvedThinkingPolicy>,
	/// Model tool and computer-use capabilities.
	pub capabilities:           ResolvedModelCapabilities,
	/// Cursor premium max-mode flag.
	pub cursor_max_mode:        Option<bool>,
	/// Suppress the provider's maximum-output-token field.
	pub omit_max_output_tokens: Option<bool>,
	/// OpenAI apply-patch tool encoding.
	pub apply_patch_shape:      Option<ApplyPatchShape>,
	/// Exact Copilot premium multiplier scaled by 1,000,000.
	pub premium_millionths:     Option<u64>,
	/// Provider reasoning serving mode.
	pub reasoning_mode:         Option<ResolvedReasoningMode>,
	/// Use the Codex Responses Lite request shape.
	pub use_responses_lite:     Option<bool>,
	/// Prefer websocket transport when supported.
	pub prefer_websockets:      Option<bool>,
	/// Credential-free static headers approved for model-level use.
	pub headers:                ResolvedModelHeaders,
	/// Canonical sparse compatibility properties in the `wire/*` namespace.
	pub compat:                 Props,
}

/// Provider-native reasoning policy resolved from trusted catalog metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedThinkingPolicy {
	/// Provider-native control used to express reasoning.
	pub mode:              ResolvedThinkingMode,
	/// Advertised efforts, ordered from least to most intensive.
	pub efforts:           SmallVec<Effort, 7>,
	/// Default effort selected when the caller does not provide one.
	pub default_effort:    Option<Effort>,
	/// Per-effort native string overrides.
	pub effort_map:        BTreeMap<Effort, Str>,
	/// Per-effort wire-model overrides.
	pub effort_routing:    BTreeMap<Effort, Str>,
	/// Per-effort thinking token budgets.
	pub effort_budgets:    BTreeMap<Effort, u64>,
	/// Whether native adaptive-thinking display control is supported.
	pub supports_display:  Option<bool>,
	/// Whether disabling reasoning must be explicit on the wire.
	pub suppress_when_off: Option<bool>,
	/// Whether the provider requires a non-off effort.
	pub requires_effort:   Option<bool>,
}

/// Provider-native mechanism used to express reasoning.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResolvedThinkingMode {
	/// Send a named effort.
	Effort,
	/// Send a token budget.
	Budget,
	/// Send a Google thinking level.
	GoogleLevel,
	/// Use Anthropic adaptive thinking.
	AnthropicAdaptive,
	/// Use Anthropic budget thinking plus an effort.
	AnthropicBudgetEffort,
}

/// Resolved model capabilities which affect provider request projection.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResolvedModelCapabilities {
	/// Explicit native tool-call support.
	pub tools:               Option<bool>,
	/// Effective computer-use support.
	pub computer_use:        Option<bool>,
	/// Explicit authored computer-use value before inference.
	pub computer_use_config: Option<bool>,
}

/// OpenAI apply-patch tool encoding.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ApplyPatchShape {
	/// Send an unwrapped custom-tool string.
	Freeform,
	/// Send a JSON function argument.
	Function,
}

/// Provider reasoning serving mode.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResolvedReasoningMode {
	/// Use the provider's pro reasoning path.
	Pro,
}

/// Credential-free model headers retained behind the server trust boundary.
///
/// Construction is intentionally owned by the catalog crate. The wrapper has
/// no serialization implementation, preventing accidental foreign-wire use.
#[repr(transparent)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResolvedModelHeaders(pub BTreeMap<Str, Str>);

impl ResolvedModelHeaders {
	/// Returns a static header by case-insensitive name.
	#[must_use]
	pub fn get(&self, name: &str) -> Option<&str> {
		self
			.0
			.iter()
			.find_map(|(key, value)| key.eq_ignore_ascii_case(name).then_some(value.as_str()))
	}

	/// Iterates over approved header names and values in deterministic order.
	pub fn iter(&self) -> impl Iterator<Item = (&Str, &Str)> {
		self.0.iter()
	}
}

/// A portable feature paired with the caller's policy when the resolved
/// provider cannot honor it.
///
/// In particular, [`Fallback::Emulate`] makes soft tool-choice forcing a shared
/// resolver behavior: prompt injection, verification, and a bounded retry loop
/// live in one place instead of every caller reimplementing them.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct Feature<T> {
	/// Typed value requested by the caller.
	pub value:          T,
	/// Required behavior when the provider path lacks native support.
	pub on_unsupported: Fallback,
}

/// Policy applied when a portable feature is unavailable on the selected
/// provider path.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Fallback {
	/// Fail the turn rather than changing the request's meaning.
	Error,
	/// Omit the feature and report the omission in the outcome.
	Ignore,
	/// Apply the bounded portable emulation strategy and report that fact.
	Emulate,
}

/// A requested feature or extension property the provider path could not honor.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct Unsupported {
	/// Stable path naming the typed feature or namespaced property.
	pub what:   Str,
	/// Human-readable explanation suitable for diagnostics.
	pub detail: Str,
	/// Honest account of how request semantics changed.
	pub action: UnsupportedAction,
}

/// The action taken for an unsupported request element.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UnsupportedAction {
	/// The request element was omitted.
	Dropped,
	/// A portable substitute was applied.
	Emulated,
	/// A value was bounded to a supported range.
	Clamped,
}

/// Provider-independent parameters for a chat turn.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct ChatParams {
	/// Trusted server-resolved model policy; absent on every foreign request.
	pub model_policy:           Option<Arc<ResolvedModelPolicy>>,
	/// Catalog id, alias, or role resolved by the gateway at admission.
	pub model:                  Str,
	/// Tool contracts exposed to the model.
	pub tools:                  Vec<ToolDef>,
	/// Optional tool selection policy and its unsupported-feature behavior.
	pub tool_choice:            Option<Feature<ToolChoice>>,
	/// Sampling controls; each absent field preserves the provider default.
	pub sampling:               Option<Sampling>,
	/// Optional thinking controls and their unsupported-feature behavior.
	pub thinking:               Option<Feature<Reasoning>>,
	/// Cache affinity and retention hints.
	pub cache:                  Option<CacheHint>,
	/// Optional structured-output constraint and its unsupported-feature
	/// behavior.
	pub response_format:        Option<Feature<ResponseFormat>>,
	/// Request attribution and telemetry correlation.
	pub meta:                   Option<RequestMeta>,
	/// Namespaced provider-specific controls. `None` means the caller did not
	/// supply provider options; `Some(Props::default())` is an explicit empty
	/// bag.
	pub provider_options:       Option<Props>,
	/// Optional portable service tier for the resolved route.
	pub service_tier:           Option<ServiceTier>,
	/// Optional per-provider-family service-tier overrides.
	pub service_tier_by_family: Option<ServiceTierByFamily>,
	/// Advisory token budget for the complete agent task.
	pub task_budget:            Option<TaskBudget>,
	/// OpenAI Responses fields requested verbatim. `Some([])` explicitly
	/// requests no includes and remains distinct from absence.
	pub responses_include:      Option<Vec<ResponseInclude>>,
}

/// A complete in-process chat turn request.
///
/// The fields are deliberately flat for codec and middleware access;
/// [`ChatParams`] is the transport-facing parameter subset without the
/// conversation.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct ChatRequest {
	/// Trusted server-resolved model policy; absent on every foreign request.
	pub model_policy:           Option<Arc<ResolvedModelPolicy>>,
	/// Catalog id, alias, or role resolved by the gateway at admission.
	pub model:                  Str,
	/// Complete conversation projected into the selected transport.
	pub thread:                 Thread,
	/// Tool contracts exposed to the model.
	pub tools:                  Vec<ToolDef>,
	/// Optional tool selection policy and its unsupported-feature behavior.
	pub tool_choice:            Option<Feature<ToolChoice>>,
	/// Sampling controls; each absent field preserves the provider default.
	pub sampling:               Option<Sampling>,
	/// Optional thinking controls and their unsupported-feature behavior.
	pub thinking:               Option<Feature<Reasoning>>,
	/// Cache affinity and retention hints.
	pub cache:                  Option<CacheHint>,
	/// Optional structured-output constraint and its unsupported-feature
	/// behavior.
	pub response_format:        Option<Feature<ResponseFormat>>,
	/// Request attribution and telemetry correlation.
	pub meta:                   Option<RequestMeta>,
	/// Namespaced provider-specific controls. `None` means the caller did not
	/// supply provider options; `Some(Props::default())` is an explicit empty
	/// bag.
	pub provider_options:       Option<Props>,
	/// Optional portable service tier for the resolved route.
	pub service_tier:           Option<ServiceTier>,
	/// Optional per-provider-family service-tier overrides.
	pub service_tier_by_family: Option<ServiceTierByFamily>,
	/// Advisory token budget for the complete agent task.
	pub task_budget:            Option<TaskBudget>,
	/// OpenAI Responses fields requested verbatim. `Some([])` explicitly
	/// requests no includes and remains distinct from absence.
	pub responses_include:      Option<Vec<ResponseInclude>>,
}

impl ChatRequest {
	pub(crate) fn into_parts(self) -> (Thread, ChatParams) {
		(self.thread, ChatParams {
			model:                  self.model,
			model_policy:           self.model_policy,
			tools:                  self.tools,
			tool_choice:            self.tool_choice,
			sampling:               self.sampling,
			thinking:               self.thinking,
			cache:                  self.cache,
			response_format:        self.response_format,
			meta:                   self.meta,
			provider_options:       self.provider_options,
			service_tier:           self.service_tier,
			service_tier_by_family: self.service_tier_by_family,
			task_budget:            self.task_budget,
			responses_include:      self.responses_include,
		})
	}

	pub(crate) fn from_parts(thread: Thread, params: ChatParams) -> Self {
		Self {
			model: params.model,
			model_policy: params.model_policy,
			thread,
			tools: params.tools,
			tool_choice: params.tool_choice,
			sampling: params.sampling,
			thinking: params.thinking,
			cache: params.cache,
			response_format: params.response_format,
			meta: params.meta,
			provider_options: params.provider_options,
			service_tier: params.service_tier,
			service_tier_by_family: params.service_tier_by_family,
			task_budget: params.task_budget,
			responses_include: params.responses_include,
		}
	}
}

/// One tool contract advertised to a model.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct ToolDef {
	/// Portable dispatch name.
	pub name:        Str,
	/// Model-facing purpose and usage guidance.
	pub description: Str,
	/// JSON Schema retained as bytes so transport-specific normalization happens
	/// only at the edge.
	pub schema_json: Bytes,
	/// Whether the provider should enforce the schema rather than treat it as
	/// guidance. Absence preserves the provider default.
	pub strict:      Option<bool>,
}

/// Portable tool-selection request whose fallback policy is carried by
/// [`Feature`].
#[non_exhaustive]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ToolChoice {
	/// Let the model decide whether and which tool to invoke.
	Auto,
	/// Prevent tool invocation.
	None,
	/// Require at least one tool invocation.
	Required,
	/// Require the named tool.
	Named(Str),
}

/// Portable request processing tier.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ServiceTier {
	/// Let the provider select its default tier.
	Auto,
	/// Explicitly request normal/default processing.
	Default,
	/// Trade latency for lower cost.
	Flex,
	/// Request reserved scale capacity.
	Scale,
	/// Request the provider's priority path.
	Priority,
}

/// Independent service-tier choices for provider families.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Default, Eq, PartialEq)]
pub struct ServiceTierByFamily {
	/// OpenAI-family tier, when explicitly selected.
	pub openai:    Option<ServiceTier>,
	/// Anthropic-family tier, when explicitly selected.
	pub anthropic: Option<ServiceTier>,
	/// Google-family tier, when explicitly selected.
	pub google:    Option<ServiceTier>,
}

/// Advisory token budget for an entire agent task.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct TaskBudget {
	/// Total task allocation.
	pub total_tokens:     u64,
	/// Remaining allocation when the caller already consumed part of the task.
	pub remaining_tokens: Option<u64>,
}

/// OpenAI Responses fields that callers may request verbatim.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResponseInclude {
	/// File-search result records.
	FileSearchResults,
	/// Web-search result records.
	WebSearchResults,
	/// Source URLs from web-search actions.
	WebSearchSources,
	/// Input-image URLs.
	InputImageUrl,
	/// Computer-call output image URLs.
	ComputerOutputImageUrl,
	/// Code-interpreter output records.
	CodeInterpreterOutputs,
	/// Encrypted reasoning replay content.
	ReasoningEncryptedContent,
	/// Output-text token log probabilities.
	OutputTextLogprobs,
}

/// Optional sampling controls; absence of every field preserves provider
/// defaults.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Default, PartialEq)]
pub struct Sampling {
	/// Randomness scale.
	pub temperature:        Option<f64>,
	/// Cumulative-probability nucleus threshold.
	pub top_p:              Option<f64>,
	/// Maximum token candidate count.
	pub top_k:              Option<u32>,
	/// Minimum probability relative to the leading token.
	pub min_p:              Option<f64>,
	/// Penalty based on prior token frequency.
	pub frequency_penalty:  Option<f64>,
	/// Penalty based on prior token presence.
	pub presence_penalty:   Option<f64>,
	/// Penalty applied to tokens already present in generated text.
	pub repetition_penalty: Option<f64>,
	/// Stop strings; absence preserves the provider default.
	pub stop:               Option<Vec<Str>>,
	/// Maximum generated tokens.
	pub max_output_tokens:  Option<u64>,
}

/// Portable model reasoning controls.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct Reasoning {
	/// Optional qualitative effort; absence preserves the provider default.
	pub effort:        Option<Effort>,
	/// Explicit token budget, taking precedence over qualitative effort where
	/// supported.
	pub budget_tokens: Option<u64>,
	/// Optionally suppress the visible reasoning summary while preserving
	/// required replay state. Absence preserves the provider default.
	pub hide_summary:  Option<bool>,
}

/// Qualitative reasoning effort after model-variant collapse.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
	/// Disable explicit reasoning.
	Off,
	/// Smallest nonzero reasoning allocation.
	Minimal,
	/// Low reasoning allocation.
	Low,
	/// Balanced reasoning allocation.
	Medium,
	/// High reasoning allocation.
	High,
	/// OpenAI's extra-high reasoning tier.
	XHigh,
	/// Provider-defined maximum reasoning allocation.
	Max,
}

/// Conversation-stable cache affinity and retention hints.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct CacheHint {
	/// Stable conversation key used for provider prompt-cache affinity and
	/// credential pinning.
	pub session_key: Str,
	/// Optional retention class; absence preserves the provider default.
	pub retention:   Option<CacheRetention>,
	/// Optional automatic versus explicit breakpoint selection.
	pub mode:        Option<PromptCacheMode>,
	/// Optional minimum lifetime for explicit cache entries.
	pub ttl:         Option<PromptCacheTtl>,
	/// Optional breakpoint placement policy.
	pub breakpoint:  Option<PromptCacheBreakpoint>,
}

/// Portable prompt-cache retention classes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CacheRetention {
	/// Explicitly disable prompt caching.
	None,
	/// Short-lived reuse suited to active interactive turns.
	Short,
	/// Longer reuse requested for durable working context.
	Long,
}

/// Prompt-cache breakpoint selection mode.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PromptCacheMode {
	/// Let the provider place its automatic breakpoint.
	Implicit,
	/// Disable automatic placement and honor the explicit breakpoint policy.
	Explicit,
}

/// Minimum prompt-cache entry lifetime.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PromptCacheTtl {
	/// Thirty-minute minimum lifetime.
	ThirtyMinutes,
}

/// Explicit prompt-cache breakpoint placement.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PromptCacheBreakpoint {
	/// Mark the latest stable historical message.
	LatestStableMessage,
	/// Mark the last cacheable block of each of the final two messages, and
	/// place no dedicated system/tools breakpoint.
	///
	/// The deeper of the two markers already caches the system prompt and tool
	/// definitions transitively, so spending separate breakpoints on them buys
	/// nothing while consuming half the provider's slot budget.
	TailTwo,
	/// Suppress an explicit breakpoint marker.
	None,
}

/// A structured-output constraint whose fallback policy is carried by
/// [`Feature`].
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct ResponseFormat {
	/// Typed output constraint projected by the transport.
	pub kind: ResponseFormatKind,
}

/// Portable structured-output constraint forms.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResponseFormatKind {
	/// Validate output against a named JSON schema.
	JsonSchema(JsonSchema),
	/// Constrain output with a formal grammar.
	Grammar(Grammar),
}

/// Named JSON Schema output constraint.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct JsonSchema {
	/// Provider-visible schema name.
	pub name:        Str,
	/// JSON Schema bytes normalized by each transport.
	pub schema_json: Bytes,
	/// Whether the provider must enforce strict conformance. Absence preserves
	/// the provider default.
	pub strict:      Option<bool>,
}

/// Formal grammar output constraint.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct Grammar {
	/// Grammar syntax understood by the definition.
	pub flavor:     GrammarFlavor,
	/// Grammar source projected or translated at the provider edge.
	pub definition: Str,
}

/// Supported formal grammar syntaxes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GrammarFlavor {
	/// Lark grammar syntax.
	Lark,
	/// Regular-expression syntax.
	Regex,
	/// GBNF grammar syntax.
	Gbnf,
}

/// Attribution and correlation metadata separate from model-visible content.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Default, Eq, PartialEq)]
pub struct RequestMeta {
	/// User, agent, or subsystem tag used for vendor initiator headers.
	pub initiator:  Str,
	/// Metering and telemetry correlation id that is never sent upstream.
	pub session_id: Str,
	/// Deterministic telemetry dimensions retained by the gateway.
	pub telemetry:  BTreeMap<Str, Str>,
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use super::{ChatRequest, ResolvedModelPolicy};
	use crate::Thread;

	#[test]
	fn resolved_policy_survives_native_split_and_retry_clones_by_arc() {
		let policy = Arc::new(ResolvedModelPolicy {
			request_model_id: Some("wire-model".into()),
			..ResolvedModelPolicy::default()
		});
		let request = ChatRequest::builder()
			.model("logical-model".into())
			.thread(Thread::default())
			.tools(Vec::new())
			.model_policy(Arc::clone(&policy))
			.build();
		let retry = request.clone();
		assert!(Arc::ptr_eq(retry.model_policy.as_ref().expect("retry policy"), &policy));

		let (thread, params) = request.into_parts();
		assert!(Arc::ptr_eq(params.model_policy.as_ref().expect("params policy"), &policy));
		let rebuilt = ChatRequest::from_parts(thread, params);
		assert!(Arc::ptr_eq(rebuilt.model_policy.as_ref().expect("rebuilt policy"), &policy));
	}
}
