use std::collections::BTreeMap;

use omp_core::Str;
use omp_llm_catalog::compat::{
	Compat, MaxTokensField, ReasoningWireFormat, ThinkingToolChoiceConflict,
};
use omp_llm_types::{ChatRequest, Effort, ResolvedThinkingMode, Unsupported, UnsupportedAction};
use serde_json::{Map, Value};

#[derive(Debug)]
pub struct OpenAiModelPolicy {
	pub compat: Compat,
	pub supports_tool_choice: bool,
	pub supports_developer_role: bool,
	pub reasoning_content_field: Option<Str>,
	pub omit_reasoning_effort: bool,
	pub allows_synthetic_reasoning_content_for_tool_calls: bool,
	pub requires_reasoning_content_for_tool_calls: bool,
	pub requires_reasoning_content_for_all_assistant_turns: bool,
	pub reasoning_enabled: bool,
	pub requires_assistant_content_for_tool_calls: bool,
	pub filter_reasoning_history: bool,
	pub include_encrypted_reasoning: bool,
	pub supports_image_detail_original: bool,
	pub supports_store: bool,
	pub store_override: Option<bool>,
	pub supports_computer_use: Option<bool>,
	pub extra_body: Option<Map<String, Value>>,
	pub effort_map: BTreeMap<Effort, Str>,
	pub thinking_mode: Option<ResolvedThinkingMode>,
	pub effort_budgets: BTreeMap<Effort, u64>,
}

impl OpenAiModelPolicy {
	pub(crate) fn resolve(
		req: &ChatRequest,
		provider: &Compat,
		unsupported: &mut Vec<Unsupported>,
	) -> Self {
		let mut compat = *provider;
		let Some(policy) = req.model_policy.as_deref() else {
			return Self {
				compat,
				supports_tool_choice: true,
				supports_developer_role: false,
				reasoning_content_field: None,
				omit_reasoning_effort: false,
				allows_synthetic_reasoning_content_for_tool_calls: false,
				requires_reasoning_content_for_tool_calls: false,
				requires_reasoning_content_for_all_assistant_turns: false,
				reasoning_enabled: thinking_enabled(req),
				requires_assistant_content_for_tool_calls: false,
				filter_reasoning_history: false,
				include_encrypted_reasoning: compat.reasoning_wire_format
					== ReasoningWireFormat::OpenAiResponses,
				supports_image_detail_original: true,
				supports_store: true,
				store_override: None,
				supports_computer_use: None,
				extra_body: None,
				effort_map: BTreeMap::new(),
				thinking_mode: None,
				effort_budgets: BTreeMap::new(),
			};
		};

		let thinking_enabled = thinking_enabled(req);
		let overlay = policy
			.compat
			.get_ns("wire", "when_thinking")
			.and_then(|value| {
				if !thinking_enabled {
					return None;
				}
				if let Some(value) = value.as_object() {
					Some(value)
				} else {
					unsupported.push(malformed("when_thinking", "must be a JSON object"));
					None
				}
			});
		let value = |name: &str| {
			overlay
				.and_then(|values| values.get(name))
				.or_else(|| policy.compat.get_ns("wire", name))
		};
		let boolean = |name: &str, default: bool, unsupported: &mut Vec<Unsupported>| {
			value(name).map_or(default, |value| {
				if let Some(value) = value.as_bool() {
					value
				} else {
					unsupported.push(malformed(name, "must be a boolean"));
					default
				}
			})
		};

		compat.usage_in_streaming =
			boolean("supports_usage_in_streaming", compat.usage_in_streaming, unsupported);
		compat.sampling_params =
			boolean("supports_sampling_params", compat.sampling_params, unsupported);
		compat.forced_tool_choice =
			boolean("supports_forced_tool_choice", compat.forced_tool_choice, unsupported);
		if let Some(disable) = value("disable_reasoning_on_tool_choice") {
			match disable.as_bool() {
				Some(true) => {
					compat.thinking_tool_choice_conflict =
						ThinkingToolChoiceConflict::DropThinkingWhenAny;
				},
				Some(false) => {
					compat.thinking_tool_choice_conflict = ThinkingToolChoiceConflict::None;
				},
				None => {
					unsupported.push(malformed("disable_reasoning_on_tool_choice", "must be a boolean"));
				},
			}
		}
		if let Some(field) = value("max_tokens_field") {
			compat.max_tokens_field = match field.as_str() {
				Some("max_tokens") => MaxTokensField::MaxTokens,
				Some("max_completion_tokens") => MaxTokensField::MaxCompletionTokens,
				Some("max_output_tokens") => MaxTokensField::MaxOutputTokens,
				_ => {
					unsupported.push(malformed(
						"max_tokens_field",
						"must be max_tokens, max_completion_tokens, or max_output_tokens",
					));
					compat.max_tokens_field
				},
			};
		}
		if let Some(format) = value("thinking_format") {
			compat.reasoning_wire_format = match format.as_str() {
				Some("none") => ReasoningWireFormat::None,
				Some("openai") => {
					if provider.reasoning_wire_format == ReasoningWireFormat::OpenAiResponses {
						ReasoningWireFormat::OpenAiResponses
					} else {
						ReasoningWireFormat::OpenAi
					}
				},
				Some("openai_responses" | "openai-responses") => ReasoningWireFormat::OpenAiResponses,
				Some("openrouter") => {
					if provider.reasoning_wire_format == ReasoningWireFormat::OpenAiResponses {
						ReasoningWireFormat::OpenAiResponses
					} else {
						ReasoningWireFormat::OpenRouter
					}
				},
				Some("zai") => ReasoningWireFormat::Zai,
				Some("qwen") => ReasoningWireFormat::QwenEnableThinking,
				Some("qwen_chat_template" | "qwen-chat-template") => {
					ReasoningWireFormat::NvidiaChatTemplateKwargs
				},
				_ => {
					unsupported
						.push(malformed("thinking_format", "has an unsupported OpenAI wire format"));
					compat.reasoning_wire_format
				},
			};
		}
		let reasoning_content_field = value("reasoning_content_field").and_then(|value| {
			if let Some("reasoning" | "reasoning_content" | "reasoning_text") = value.as_str() {
				Some(Str::from(value.as_str().expect("matched string")))
			} else {
				unsupported.push(malformed(
					"reasoning_content_field",
					"must be reasoning, reasoning_content, or reasoning_text",
				));
				None
			}
		});
		let mut effort_map = policy
			.thinking
			.as_ref()
			.map_or_else(BTreeMap::new, |thinking| thinking.effort_map.clone());
		if let Some(map) = value("reasoning_effort_map") {
			if let Some(map) = map.as_object() {
				for (name, value) in map {
					let effort = match name.as_str() {
						"off" => Some(Effort::Off),
						"minimal" => Some(Effort::Minimal),
						"low" => Some(Effort::Low),
						"medium" => Some(Effort::Medium),
						"high" => Some(Effort::High),
						"xhigh" => Some(Effort::XHigh),
						"max" => Some(Effort::Max),
						_ => None,
					};
					match (effort, value.as_str()) {
						(Some(effort), Some(value)) => {
							effort_map.insert(effort, Str::from(value));
						},
						_ => unsupported.push(malformed(
							"reasoning_effort_map",
							"keys must be portable efforts and values must be strings",
						)),
					}
				}
			} else {
				unsupported.push(malformed("reasoning_effort_map", "must be a JSON object"));
			}
		}
		let extra_body = value("extra_body").and_then(|value| {
			if let Some(value) = value.as_object() {
				Some(value.clone())
			} else {
				unsupported.push(malformed("extra_body", "must be a JSON object"));
				None
			}
		});
		let omit_reasoning_effort = boolean("omit_reasoning_effort", false, unsupported);
		let supports_reasoning_effort = boolean("supports_reasoning_effort", true, unsupported);

		Self {
			compat,
			supports_tool_choice: boolean("supports_tool_choice", true, unsupported),
			supports_developer_role: boolean("supports_developer_role", false, unsupported),
			reasoning_content_field,
			omit_reasoning_effort: omit_reasoning_effort || !supports_reasoning_effort,
			allows_synthetic_reasoning_content_for_tool_calls: boolean(
				"allows_synthetic_reasoning_content_for_tool_calls",
				false,
				unsupported,
			),
			requires_reasoning_content_for_tool_calls: boolean(
				"requires_reasoning_content_for_tool_calls",
				false,
				unsupported,
			),
			requires_reasoning_content_for_all_assistant_turns: boolean(
				"requires_reasoning_content_for_all_assistant_turns",
				false,
				unsupported,
			),
			reasoning_enabled: thinking_enabled,
			requires_assistant_content_for_tool_calls: boolean(
				"requires_assistant_content_for_tool_calls",
				false,
				unsupported,
			),
			filter_reasoning_history: boolean("filter_reasoning_history", false, unsupported),
			include_encrypted_reasoning: boolean(
				"include_encrypted_reasoning",
				compat.reasoning_wire_format == ReasoningWireFormat::OpenAiResponses,
				unsupported,
			),
			supports_image_detail_original: boolean(
				"supports_image_detail_original",
				true,
				unsupported,
			),
			supports_store: boolean("supports_store", true, unsupported),
			store_override: value("supports_store").and_then(Value::as_bool),
			supports_computer_use: policy.capabilities.computer_use,
			extra_body,
			effort_map,
			thinking_mode: policy.thinking.as_ref().map(|thinking| thinking.mode),
			effort_budgets: policy
				.thinking
				.as_ref()
				.map_or_else(BTreeMap::new, |thinking| thinking.effort_budgets.clone()),
		}
	}

	pub(crate) fn mapped_effort(&self, effort: Effort) -> Str {
		self.effort_map.get(&effort).cloned().unwrap_or_else(|| {
			if effort == Effort::Off
				&& self.compat.reasoning_wire_format == ReasoningWireFormat::OpenAiResponses
			{
				Str::new_static("off")
			} else {
				Str::new(effort_name(effort))
			}
		})
	}
}

fn thinking_enabled(req: &ChatRequest) -> bool {
	req.thinking.as_ref().is_some_and(|feature| {
		feature.value.effort != Some(Effort::Off) && feature.value.budget_tokens != Some(0)
	})
}

pub const fn effort_name(value: Effort) -> &'static str {
	match value {
		Effort::Off => "none",
		Effort::Minimal => "minimal",
		Effort::Low => "low",
		Effort::Medium => "medium",
		Effort::High => "high",
		Effort::XHigh => "xhigh",
		Effort::Max => "max",
		_ => "medium",
	}
}

fn malformed(name: &str, detail: &str) -> Unsupported {
	Unsupported::builder()
		.what(Str::from(format!("model_policy.compat:wire/{name}")))
		.detail(Str::from(detail))
		.action(UnsupportedAction::Dropped)
		.build()
}
