//! Anthropic reasoning controls, signed-history replay, and request
//! fingerprints.

use std::borrow::Cow;

use bytes::{Bytes, BytesMut};
use omp_core::SmolStr;
use omp_llm_catalog::compat::{Compat, ReasoningWireFormat};
use omp_llm_types::{
	ChatRequest, Effort, Feature, Item, ItemKind, Part, Props, Reasoning, ResolvedModelPolicy,
	ResolvedThinkingMode, ResolvedThinkingPolicy, Role, Thinking,
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use xxhash_rust::xxh64::xxh64;

/// Claude runtime version represented by the current Cowork request
/// fingerprint.
pub const CLAUDE_CODE_VERSION: &str = "2.1.220";

const CCH_SEED: u64 = 0x4d65_9218_e32a_3268;
const CCH_PLACEHOLDER: &[u8] = b"cch=00000";
const BILLING_MARKER: &[u8] =
	b"\"system\":[{\"type\":\"text\",\"text\":\"x-anthropic-billing-header:";

/// One non-secret HTTP header required by the Anthropic wire profile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnthropicHeader {
	/// Header field name with the canonical casing used by Anthropic's clients.
	pub name:  SmolStr,
	/// Public protocol or client-fingerprint value; credentials are never
	/// returned.
	pub value: SmolStr,
}

/// Computes non-secret request headers and negotiated beta features.
///
/// Authentication remains the egress lease's responsibility. In particular,
/// this function never returns `authorization`, `x-api-key`, cookies, or an
/// upstream account identifier.
#[must_use]
pub fn request_headers(request: &ChatRequest, compat: &Compat) -> Vec<AnthropicHeader> {
	let options = request.provider_options.as_ref();
	let thinking_enabled = request
		.thinking
		.as_ref()
		.is_some_and(|feature| feature.value.effort != Some(Effort::Off))
		|| policy_bool(request.model_policy.as_deref(), "requires_thinking_enabled", false)
			== Some(true);
	let claude_code = option_bool(options, "claude_code").unwrap_or(false);
	let mut betas = Vec::<String>::new();
	if claude_code {
		for beta in [
			"claude-code-20250219",
			"interleaved-thinking-2025-05-14",
			"thinking-token-count-2026-05-13",
			"context-management-2025-06-27",
			"prompt-caching-scope-2026-01-05",
			"mid-conversation-system-2026-04-07",
			"advanced-tool-use-2025-11-20",
			"effort-2025-11-24",
			"fallback-credit-2026-06-01",
		] {
			push_beta(&mut betas, beta);
		}
	} else {
		if request.thinking.is_some() && !request.tools.is_empty() {
			push_beta(&mut betas, "interleaved-thinking-2025-05-14");
		}
		if compat.reasoning_wire_format == ReasoningWireFormat::Anthropic
			&& reasoning_projection(request).effort.is_some()
		{
			push_beta(&mut betas, "effort-2025-11-24");
		}
	}
	if request
		.cache
		.as_ref()
		.is_some_and(|cache| matches!(cache.retention, Some(omp_llm_types::CacheRetention::Long)))
		&& policy_bool(
			request.model_policy.as_deref(),
			"supports_long_cache_retention",
			thinking_enabled,
		) != Some(false)
	{
		push_beta(&mut betas, "extended-cache-ttl-2025-04-11");
	}
	if let Some(cache) = options
		.and_then(|props| props.get_ns("anthropic", "cache_control"))
		.and_then(Value::as_object)
	{
		if cache.get("ttl").and_then(Value::as_str) == Some("1h")
			&& policy_bool(
				request.model_policy.as_deref(),
				"supports_long_cache_retention",
				thinking_enabled,
			) != Some(false)
		{
			push_beta(&mut betas, "extended-cache-ttl-2025-04-11");
		}
		if cache.get("scope").and_then(Value::as_str) == Some("global") {
			push_beta(&mut betas, "prompt-caching-scope-2026-01-05");
		}
	}
	if request.thread.items.iter().any(item_contains_pdf) {
		push_beta(&mut betas, "pdfs-2024-09-25");
	}
	if request.thread.items.iter().any(item_contains_file_source) {
		push_beta(&mut betas, "files-api-2025-04-14");
	}
	if request.response_format.is_some() {
		push_beta(&mut betas, "structured-outputs-2025-12-15");
	}
	let eager_support = policy_bool(
		request.model_policy.as_deref(),
		"supports_eager_tool_input_streaming",
		thinking_enabled,
	);
	if (option_bool(options, "eager_input_streaming") == Some(true) && eager_support != Some(false))
		|| option_bool(options, "eager_input_streaming").is_none()
			&& eager_support == Some(true)
			&& !request.tools.is_empty()
	{
		push_beta(&mut betas, "fine-grained-tool-streaming-2025-05-14");
	}
	if policy_bool(
		request.model_policy.as_deref(),
		"supports_mid_conversation_system",
		thinking_enabled,
	) == Some(true)
		&& has_mid_conversation_system(request)
	{
		push_beta(&mut betas, "mid-conversation-system-2026-04-07");
	}
	if options.is_some_and(|props| props.get_ns("anthropic", "context_management").is_some()) {
		push_beta(&mut betas, "context-management-2025-06-27");
	}
	if option_str(options, "service_tier") == Some("priority") {
		push_beta(&mut betas, "fast-mode-2026-02-01");
	}
	if let Some(tools) = options
		.and_then(|props| props.get_ns("anthropic", "server_tools"))
		.and_then(Value::as_array)
	{
		for kind in tools
			.iter()
			.filter_map(Value::as_object)
			.filter_map(|tool| tool.get("type"))
			.filter_map(Value::as_str)
		{
			if kind.starts_with("web_search_") {
				push_beta(&mut betas, "web-search-2025-03-05");
			} else {
				push_beta(&mut betas, "advanced-tool-use-2025-11-20");
			}
		}
	}
	if let Some(extra) = options.and_then(|props| props.get_ns("anthropic", "betas")) {
		match extra {
			Value::String(value) => {
				for beta in value.split(',') {
					push_beta(&mut betas, beta);
				}
			},
			Value::Array(values) => {
				for beta in values.iter().filter_map(Value::as_str) {
					push_beta(&mut betas, beta);
				}
			},
			_ => {},
		}
	}
	let beta = (!betas.is_empty()).then(|| betas.join(","));
	let version = option_str(options, "version").unwrap_or("2023-06-01");
	if !claude_code {
		let mut headers = vec![header("anthropic-version", version)];
		if let Some(beta) = beta {
			headers.push(header("anthropic-beta", &beta));
		}
		return headers;
	}

	let mut headers = vec![
		header("Accept", "application/json"),
		header("Content-Type", "application/json"),
		header("User-Agent", &format!("claude-cli/{CLAUDE_CODE_VERSION} (external, claude-desktop)")),
	];
	if let Some(session) = option_str(options, "claude_code_session_id") {
		headers.push(header("X-Claude-Code-Session-Id", session));
	}
	headers.extend([
		header("X-Stainless-Arch", stainless_arch()),
		header("X-Stainless-Lang", "js"),
		header("X-Stainless-OS", "Linux"),
		header("X-Stainless-Package-Version", "0.94.0"),
		header("X-Stainless-Retry-Count", "0"),
		header("X-Stainless-Runtime", "node"),
		header("X-Stainless-Runtime-Version", "v26.3.0"),
		header("X-Stainless-Timeout", "600"),
	]);
	if let Some(beta) = beta {
		headers.push(header("anthropic-beta", &beta));
	}
	headers.extend([
		header("anthropic-dangerous-direct-browser-access", "true"),
		header("anthropic-version", version),
		header("x-app", "cli"),
		header("Connection", "keep-alive"),
		header("Accept-Encoding", "gzip, deflate, br, zstd"),
	]);
	headers
}

fn header(name: &str, value: &str) -> AnthropicHeader {
	AnthropicHeader { name: name.into(), value: value.into() }
}
fn push_beta(values: &mut Vec<String>, value: &str) {
	let value = value.trim();
	if !value.is_empty() && !values.iter().any(|existing| existing == value) {
		values.push(value.to_owned());
	}
}

fn stainless_arch() -> &'static str {
	match std::env::consts::ARCH {
		"x86_64" => "x64",
		"aarch64" => "arm64",
		"x86" => "x86",
		_ => "other::unknown",
	}
}

fn item_contains_pdf(item: &Item) -> bool {
	match &item.kind {
		ItemKind::Message(message) => message.parts.iter().any(part_is_pdf),
		ItemKind::ToolResult(result) => result.parts.iter().any(part_is_pdf),
		_ => false,
	}
}

fn part_is_pdf(part: &Part) -> bool {
	matches!(part, Part::Blob(blob) if blob.mime.trim().eq_ignore_ascii_case("application/pdf"))
}
fn item_contains_file_source(item: &Item) -> bool {
	["image_sources", "document_sources"]
		.into_iter()
		.any(|name| {
			item
				.props
				.get_ns("anthropic", name)
				.and_then(Value::as_array)
				.is_some_and(|sources| {
					sources
						.iter()
						.any(|source| source.get("type").and_then(Value::as_str) == Some("file"))
				})
		})
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireThinking {
	Adaptive {
		#[serde(skip_serializing_if = "Option::is_none")]
		display: Option<&'static str>,
	},
	Enabled {
		budget_tokens: u64,
		#[serde(skip_serializing_if = "Option::is_none")]
		display:       Option<&'static str>,
	},
	Disabled,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ReasoningProjection<'a> {
	pub(crate) thinking: Option<WireThinking>,
	pub(crate) effort:   Option<&'a str>,
	pub(crate) budget:   Option<u64>,
}

pub(crate) fn reasoning_projection(request: &ChatRequest) -> ReasoningProjection<'_> {
	reasoning_projection_for(
		&request.model,
		&request.thinking,
		&request.provider_options,
		request.model_policy.as_deref(),
	)
}

pub(crate) fn reasoning_projection_for<'a>(
	model: &'a str,
	thinking: &'a Option<Feature<Reasoning>>,
	provider_options: &'a Option<Props>,
	model_policy: Option<&'a ResolvedModelPolicy>,
) -> ReasoningProjection<'a> {
	let requires_thinking =
		policy_bool(model_policy, "requires_thinking_enabled", false) == Some(true);
	let default_reasoning;
	let reasoning = if let Some(feature) = thinking {
		&feature.value
	} else if requires_thinking {
		default_reasoning = Reasoning::builder().build();
		&default_reasoning
	} else {
		return ReasoningProjection::default();
	};

	if option_bool(provider_options.as_ref(), "thinking_supported") == Some(false) {
		return ReasoningProjection::default();
	}
	let reasoning = reasoning;
	let requested_effort = if requires_thinking && reasoning.effort == Some(Effort::Off) {
		model_policy
			.and_then(|policy| policy.thinking.as_ref())
			.and_then(|thinking| {
				thinking
					.default_effort
					.or_else(|| thinking.efforts.first().copied())
			})
			.or(Some(Effort::Low))
	} else {
		reasoning
			.effort
			.or_else(|| model_policy.and_then(|policy| policy.thinking.as_ref()?.default_effort))
	};
	let policy_mode = model_policy.and_then(|policy| policy.thinking.as_ref());
	let caller_mode = option_str(provider_options.as_ref(), "thinking_mode");
	let disable_adaptive = option_bool(provider_options.as_ref(), "disable_adaptive_thinking")
		.or_else(|| {
			policy_bool(
				model_policy,
				"disable_adaptive_thinking",
				reasoning.effort != Some(Effort::Off),
			)
		}) == Some(true);
	let adaptive = match caller_mode {
		Some("adaptive") => true,
		Some("budget") => false,
		_ => {
			policy_mode
				.is_some_and(|thinking| thinking.mode == ResolvedThinkingMode::AnthropicAdaptive)
				|| policy_mode.is_none() && adaptive_model(model, provider_options.as_ref())
		},
	} && !disable_adaptive;
	let budget_effort = caller_mode.is_none()
		&& policy_mode
			.is_some_and(|thinking| thinking.mode == ResolvedThinkingMode::AnthropicBudgetEffort);
	let display = if let Some(policy) = policy_mode {
		(policy.supports_display == Some(true)).then_some(if reasoning.hide_summary == Some(true) {
			"omitted"
		} else {
			"summarized"
		})
	} else {
		reasoning
			.hide_summary
			.and_then(|hidden| hidden.then_some("omitted"))
	};
	if reasoning.effort == Some(Effort::Off) && !requires_thinking {
		return if adaptive {
			ReasoningProjection {
				thinking: None,
				effort:   Some(mapped_effort(policy_mode, Effort::Low)),
				budget:   None,
			}
		} else {
			ReasoningProjection {
				thinking: Some(WireThinking::Disabled),
				effort:   None,
				budget:   None,
			}
		};
	}
	if policy_mode.is_none()
		&& let Some(budget) = reasoning.budget_tokens
	{
		return ReasoningProjection {
			thinking: Some(WireThinking::Enabled { budget_tokens: budget, display }),
			effort:   None,
			budget:   Some(budget),
		};
	}
	if adaptive {
		return ReasoningProjection {
			thinking: Some(WireThinking::Adaptive { display }),
			effort:   requested_effort.map(|effort| mapped_effort(policy_mode, effort)),
			budget:   None,
		};
	}
	let budget = reasoning.budget_tokens.unwrap_or_else(|| {
		requested_effort
			.and_then(|effort| policy_mode?.effort_budgets.get(&effort).copied())
			.unwrap_or_else(|| {
				requested_effort
					.map(|effort| {
						if policy_mode.is_some() {
							effort_budget(effort)
						} else {
							legacy_effort_budget(effort)
						}
					})
					.unwrap_or(1024)
			})
	});
	ReasoningProjection {
		thinking: Some(WireThinking::Enabled { budget_tokens: budget, display }),
		effort:   budget_effort
			.then(|| requested_effort.map(|effort| mapped_effort(policy_mode, effort)))
			.flatten(),
		budget:   Some(budget),
	}
}

fn mapped_effort(policy: Option<&ResolvedThinkingPolicy>, effort: Effort) -> &str {
	policy
		.and_then(|thinking| thinking.effort_map.get(&effort))
		.map_or_else(
			|| {
				if policy.is_some() {
					effort_name(effort)
				} else {
					legacy_effort_name(effort)
				}
			},
			SmolStr::as_str,
		)
}

fn adaptive_model(model: &str, provider_options: Option<&Props>) -> bool {
	if let Some(mode) = option_str(provider_options, "thinking_mode") {
		return mode == "adaptive";
	}
	let model = model.to_ascii_lowercase();
	(model.contains("opus-4-6")
		|| model.contains("sonnet-4-6")
		|| model.contains("fable")
		|| model.contains("mythos"))
		&& option_bool(provider_options, "disable_adaptive_thinking") != Some(true)
}

const fn effort_name(effort: Effort) -> &'static str {
	match effort {
		Effort::Off | Effort::Minimal | Effort::Low => "low",
		Effort::Medium => "medium",
		Effort::High => "high",
		Effort::XHigh => "xhigh",
		Effort::Max => "max",
		_ => "medium",
	}
}

const fn legacy_effort_name(effort: Effort) -> &'static str {
	match effort {
		Effort::Off | Effort::Minimal | Effort::Low => "low",
		Effort::Medium => "medium",
		Effort::High => "high",
		Effort::Max => "max",
		_ => "medium",
	}
}

const fn legacy_effort_budget(effort: Effort) -> u64 {
	match effort {
		Effort::Off => 0,
		Effort::Minimal => 1_024,
		Effort::Low => 2_048,
		Effort::Medium => 8_192,
		Effort::High => 16_384,
		Effort::Max => 32_768,
		_ => 8_192,
	}
}

pub(crate) const fn effort_budget(effort: Effort) -> u64 {
	match effort {
		Effort::Off => 0,
		Effort::Minimal => 1_024,
		Effort::Low => 2_048,
		Effort::Medium => 8_192,
		Effort::High => 16_384,
		Effort::XHigh => 24_576,
		Effort::Max => 32_768,
		_ => 8_192,
	}
}

pub(crate) enum HistoryProjection<'a> {
	Native { text: &'a str, signature: &'a str },
	Redacted { data: &'a str },
	Demoted(Cow<'a, str>),
	Drop,
}

pub(crate) fn project_history<'a>(
	thinking: &'a Thinking,
	item_props: &Props,
	target_model: &str,
	_provider_options: &Option<Props>,
	_model_policy: Option<&ResolvedModelPolicy>,
	_thinking_enabled: bool,
	compat: &Compat,
) -> HistoryProjection<'a> {
	let signature = std::str::from_utf8(&thinking.signature).ok();
	let source_model = item_props
		.get_ns("anthropic", "model")
		.and_then(Value::as_str);
	let same_model = source_model.is_none_or(|source| source == target_model);
	let native_supported = compat.reasoning_wire_format == ReasoningWireFormat::Anthropic;
	if thinking.redacted {
		return match (same_model, signature) {
			(true, Some(data)) if !data.is_empty() => HistoryProjection::Redacted { data },
			_ => HistoryProjection::Drop,
		};
	}
	if !native_supported || !same_model {
		return demoted(target_model, &thinking.text);
	}
	match signature {
		Some(value) if !value.trim().is_empty() => {
			HistoryProjection::Native { text: &thinking.text, signature: value }
		},
		_ => demoted(target_model, &thinking.text),
	}
}
pub(crate) fn signing_endpoint(
	model: &str,
	provider_options: Option<&Props>,
	model_policy: Option<&ResolvedModelPolicy>,
	thinking_enabled: bool,
) -> bool {
	option_bool(provider_options, "signing_endpoint")
		.or_else(|| policy_bool(model_policy, "signing_endpoint", thinking_enabled))
		.or_else(|| policy_bool(model_policy, "official_endpoint", thinking_enabled))
		.unwrap_or_else(|| model.to_ascii_lowercase().contains("claude"))
}

fn demoted<'a>(target_model: &str, text: &'a str) -> HistoryProjection<'a> {
	if text.trim().is_empty() {
		return HistoryProjection::Drop;
	}
	if target_model.to_ascii_lowercase().contains("claude") {
		HistoryProjection::Demoted(Cow::Borrowed(text))
	} else {
		HistoryProjection::Demoted(Cow::Owned(format!("```thinking\n{text}\n```")))
	}
}

pub(crate) fn policy_bool(
	model_policy: Option<&ResolvedModelPolicy>,
	name: &str,
	thinking_enabled: bool,
) -> Option<bool> {
	let compat = &model_policy?.compat;
	if thinking_enabled
		&& let Some(value) = compat
			.get_ns("wire", "when_thinking")
			.and_then(Value::as_object)
			.and_then(|overlay| overlay.get(name))
			.and_then(Value::as_bool)
	{
		return Some(value);
	}
	compat.get_ns("wire", name).and_then(Value::as_bool)
}

fn has_mid_conversation_system(request: &ChatRequest) -> bool {
	let mut saw_conversation = false;
	for item in &request.thread.items {
		if let ItemKind::Message(message) = &item.kind {
			if message.role == Role::System && saw_conversation {
				return true;
			}
			if message.role != Role::System {
				saw_conversation = true;
			}
		} else {
			saw_conversation = true;
		}
	}
	false
}

pub(crate) fn claude_code_system_prelude_for(
	thread: &omp_llm_types::Thread,
	provider_options: &Option<Props>,
) -> Option<[String; 2]> {
	if option_bool(provider_options.as_ref(), "claude_code") != Some(true) {
		return None;
	}
	let first = first_user_text(thread);
	let key = [4_usize, 7, 20]
		.into_iter()
		.map(|index| first.chars().nth(index).unwrap_or('0'))
		.collect::<String>();
	let digest = Sha256::digest(format!("59cf53e54c78{key}{CLAUDE_CODE_VERSION}").as_bytes());
	let suffix = format!("{:02x}{:01x}", digest[0], digest[1] >> 4);
	Some([
		format!(
			"x-anthropic-billing-header: cc_version={CLAUDE_CODE_VERSION}.{suffix}; \
			 cc_entrypoint=claude-desktop; cch=00000;"
		),
		"You are a Claude agent, built on Anthropic's Claude Agent SDK.".to_owned(),
	])
}
fn first_user_text(thread: &omp_llm_types::Thread) -> &str {
	thread
		.items
		.iter()
		.find_map(|item| match &item.kind {
			ItemKind::Message(message) if message.role == Role::User => {
				message.parts.iter().find_map(|part| {
					if let Part::Text(text) = part {
						Some(text.as_str())
					} else {
						None
					}
				})
			},
			_ => None,
		})
		.unwrap_or("")
}

pub(crate) fn patch_billing_attestation(body: Bytes) -> Bytes {
	let Some(marker) = find_bytes(&body, BILLING_MARKER) else {
		return body;
	};
	let search_start = marker + BILLING_MARKER.len();
	let search_end = body.len().min(search_start + 150);
	let Some(relative) = find_bytes(&body[search_start..search_end], CCH_PLACEHOLDER) else {
		return body;
	};
	let hash = xxh64(&body, CCH_SEED) & 0x000f_ffff;
	let digits = format!("{hash:05x}");
	let offset = search_start + relative + 4;
	let mut patched = BytesMut::from(body);
	patched[offset..offset + 5].copy_from_slice(digits.as_bytes());
	patched.freeze()
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
	if needle.is_empty() {
		return Some(0);
	}
	haystack
		.windows(needle.len())
		.position(|window| window == needle)
}
pub(crate) fn is_known_option(key: &str) -> bool {
	matches!(
		key,
		"anthropic/claude_code"
			| "anthropic/claude_code_session_id"
			| "anthropic/thinking_supported"
			| "anthropic/thinking_mode"
			| "anthropic/disable_adaptive_thinking"
			| "anthropic/signing_endpoint"
	)
}

fn option_bool(options: Option<&Props>, name: &str) -> Option<bool> {
	options?.get_ns("anthropic", name)?.as_bool()
}

fn option_str<'a>(options: Option<&'a Props>, name: &str) -> Option<&'a str> {
	options?.get_ns("anthropic", name)?.as_str()
}
pub(crate) fn validate_options(options: &Props) -> Result<(), omp_llm_types::Error> {
	for name in
		["claude_code", "thinking_supported", "disable_adaptive_thinking", "signing_endpoint"]
	{
		if options
			.get_ns("anthropic", name)
			.is_some_and(|value| !value.is_boolean())
		{
			return Err(provider_error(
				"Anthropic thinking/fingerprint boolean option had a non-boolean value",
			));
		}
	}
	for name in ["claude_code_session_id", "version"] {
		if options
			.get_ns("anthropic", name)
			.is_some_and(|value| value.as_str().is_none_or(str::is_empty))
		{
			return Err(provider_error("Anthropic header option must be a non-empty string"));
		}
	}
	if let Some(mode) = options.get_ns("anthropic", "thinking_mode") {
		if !matches!(mode.as_str(), Some("adaptive" | "budget")) {
			return Err(provider_error("anthropic/thinking_mode must be adaptive or budget"));
		}
	}
	if let Some(betas) = options.get_ns("anthropic", "betas") {
		let valid = betas.is_string()
			|| betas
				.as_array()
				.is_some_and(|values| values.iter().all(Value::is_string));
		if !valid {
			return Err(provider_error("anthropic/betas must be a string or an array of strings"));
		}
	}
	Ok(())
}

fn provider_error(detail: &'static str) -> omp_llm_types::Error {
	omp_llm_types::Error::Provider(detail.into())
}

#[cfg(test)]
mod tests {
	use bytes::Bytes;
	use omp_llm_catalog::compat::{Compat, ReasoningWireFormat};
	use omp_llm_types::{
		ChatRequest, Effort, Fallback, Feature, Item, ItemKind, JsonSchema, Message, Part, Props,
		Reasoning, ResponseFormat, ResponseFormatKind, Role, Thinking, Thread,
	};
	use serde_json::json;

	use super::{
		HistoryProjection, patch_billing_attestation, project_history, reasoning_projection,
		request_headers,
	};

	fn request(model: &str, thinking: Thinking, source_model: Option<&str>) -> ChatRequest {
		let mut props = Props::default();
		if let Some(source) = source_model {
			props.insert_ns("anthropic", "model", json!(source));
		}
		ChatRequest::builder()
			.model(model.into())
			.thread(
				Thread::builder()
					.items(vec![
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
					])
					.build(),
			)
			.tools(Vec::new())
			.build()
	}

	fn signed() -> Thinking {
		Thinking::builder()
			.text("private plan".into())
			.signature(Bytes::from_static(b"sig"))
			.redacted(false)
			.build()
	}

	fn anthropic_compat() -> Compat {
		let mut compat = Compat::default();
		compat.reasoning_wire_format = ReasoningWireFormat::Anthropic;
		compat
	}

	#[test]
	fn signed_history_replays_only_for_the_same_model() {
		let same_thinking = signed();
		let same = request("claude-sonnet-4-6", same_thinking.clone(), Some("claude-sonnet-4-6"));
		assert!(matches!(
			project_history(
				&same_thinking,
				&same.thread.items[0].props,
				&same.model,
				&same.provider_options,
				same.model_policy.as_deref(),
				true,
				&anthropic_compat()
			),
			HistoryProjection::Native { signature: "sig", .. }
		));
		let foreign_thinking = signed();
		let foreign = request("claude-opus-4-6", foreign_thinking.clone(), Some("claude-sonnet-4-6"));
		assert!(matches!(
			project_history(
				&foreign_thinking,
				&foreign.thread.items[0].props,
				&foreign.model,
				&foreign.provider_options,
				foreign.model_policy.as_deref(),
				true,
				&anthropic_compat()
			),
			HistoryProjection::Demoted(_)
		));
	}

	#[test]
	fn unsigned_replay_is_losslessly_demoted_and_never_serializes_an_empty_signature() {
		let unsigned = Thinking::builder()
			.text("plan".into())
			.signature(Bytes::new())
			.redacted(false)
			.build();
		let mut req = request("claude-sonnet-4-6", unsigned, Some("claude-sonnet-4-6"));
		let mut options = Props::default();
		options.insert_ns("anthropic", "replay_unsigned_thinking", json!(true));
		req.provider_options = Some(options);
		let ItemKind::Message(message) = &req.thread.items[0].kind else {
			panic!("expected assistant message");
		};
		let mut unsupported = Vec::new();
		let blocks = crate::message_blocks(
			message,
			&req.thread.items[0].props,
			&req.model,
			&req.provider_options,
			&anthropic_compat(),
			&mut unsupported,
			req.model_policy.as_deref(),
			true,
		)
		.expect("thinking history should project");
		let wire = serde_json::to_value(&blocks).expect("message blocks should serialize");

		assert_eq!(wire, json!([{"type": "text", "text": "plan"}]));
		assert!(!wire.to_string().contains(r#""signature":"""#));
		assert!(!super::is_known_option("anthropic/replay_unsigned_thinking"));
	}

	#[test]
	fn signed_replay_serializes_the_original_signature_bytes() {
		let signature = "sig_REDACTED+/=";
		let thinking = Thinking::builder()
			.text("private plan".into())
			.signature(Bytes::copy_from_slice(signature.as_bytes()))
			.redacted(false)
			.build();
		let req = request("claude-sonnet-4-6", thinking, Some("claude-sonnet-4-6"));
		let ItemKind::Message(message) = &req.thread.items[0].kind else {
			panic!("expected assistant message");
		};
		let mut unsupported = Vec::new();
		let blocks = crate::message_blocks(
			message,
			&req.thread.items[0].props,
			&req.model,
			&req.provider_options,
			&anthropic_compat(),
			&mut unsupported,
			req.model_policy.as_deref(),
			true,
		)
		.expect("thinking history should project");

		assert_eq!(
			serde_json::to_value(&blocks).expect("message blocks should serialize"),
			json!([{
				"type": "thinking",
				"thinking": "private plan",
				"signature": signature,
			}])
		);
	}

	#[test]
	fn adaptive_effort_and_explicit_budget_map_independently() {
		let mut req = request("claude-sonnet-4-6", signed(), None);
		req.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::High).build())
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		let projected = reasoning_projection(&req);
		assert_eq!(projected.effort, Some("high"));
		assert!(matches!(projected.thinking, Some(super::WireThinking::Adaptive { .. })));
		req.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().budget_tokens(4096).build())
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		let projected = reasoning_projection(&req);
		assert_eq!(projected.budget, Some(4096));
		assert!(matches!(
			projected.thinking,
			Some(super::WireThinking::Enabled { budget_tokens: 4096, .. })
		));
	}

	#[test]
	fn claude_code_headers_never_contain_auth_material_and_billing_is_attested() {
		let mut req = request("claude-sonnet-4-6", signed(), None);
		let mut options = Props::default();
		options.insert_ns("anthropic", "claude_code", json!(true));
		options.insert_ns("anthropic", "eager_input_streaming", json!(true));
		options.insert_ns("anthropic", "context_management", json!({"edits":[]}));
		options.insert_ns("anthropic", "service_tier", json!("priority"));
		options.insert_ns(
			"anthropic",
			"cache_control",
			json!({"type":"ephemeral","ttl":"1h","scope":"global"}),
		);
		options.insert_ns(
			"anthropic",
			"server_tools",
			json!([{"type":"web_search_20250305","name":"web_search"}]),
		);
		options.insert_ns(
			"anthropic",
			"betas",
			json!(["caller-beta-2026-01-01", "effort-2025-11-24"]),
		);
		options.insert_ns("anthropic", "version", json!("2024-01-01"));
		req.provider_options = Some(options);
		req.response_format = Some(
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
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
		let headers = request_headers(&req, &anthropic_compat());
		assert!(headers.iter().any(|header| header.name == "User-Agent"));
		let beta = headers
			.iter()
			.find(|header| header.name == "anthropic-beta")
			.expect("beta header")
			.value
			.as_str();
		for expected in [
			"fine-grained-tool-streaming-2025-05-14",
			"context-management-2025-06-27",
			"fast-mode-2026-02-01",
			"web-search-2025-03-05",
			"structured-outputs-2025-12-15",
			"caller-beta-2026-01-01",
			"extended-cache-ttl-2025-04-11",
			"prompt-caching-scope-2026-01-05",
		] {
			assert!(beta.split(',').any(|value| value == expected));
		}
		assert_eq!(
			beta
				.split(',')
				.filter(|value| *value == "effort-2025-11-24")
				.count(),
			1
		);
		assert!(
			headers
				.iter()
				.any(|header| { header.name == "anthropic-version" && header.value == "2024-01-01" })
		);
		assert!(!headers.iter().any(|header| matches!(
			header.name.as_str().to_ascii_lowercase().as_str(),
			"authorization" | "x-api-key" | "cookie"
		)));
		let body = Bytes::from_static(b"{\"system\":[{\"type\":\"text\",\"text\":\"x-anthropic-billing-header: cc_version=2.1.220.abc; cc_entrypoint=claude-desktop; cch=00000;\"}]}");
		let patched = patch_billing_attestation(body);
		assert!(
			!patched
				.windows(b"cch=00000".len())
				.any(|window| window == b"cch=00000")
		);
	}
}
