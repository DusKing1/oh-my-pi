//! `ChatGPT` Codex Responses Lite request shaping.
//!
//! This module only transforms JSON wire bodies. Authentication and HTTP
//! dispatch remain owned by the gateway egress stack.

use std::{collections::BTreeSet, env};

use omp_core::Str;
use omp_llm_types::{Error, Unsupported};
use serde_json::{Map, Value, json};

use crate::responses_tool_repair::repair_responses_tool_pairs;

/// Provider-option namespace for Codex-only controls.
pub const CODEX_PROVIDER_NAMESPACE: &str = "openai-codex";

/// Provider-option name selecting the Responses Lite wire shape.
pub const RESPONSES_LITE_OPTION: &str = "responses_lite";

/// Applies the Codex Responses contract and, when requested, the Responses Lite
/// rewrite to an encoded `OpenAI` Responses request.
///
/// The rewrite is intentionally lossless for conversation items while removing
/// fields rejected by the subscription endpoint. Lite requests move top-level
/// instructions and tool declarations into leading developer input items, force
/// serial tool use, and leave image detail selection to the server.
///
/// The returned records describe orphan outputs whose wire item shape had to be
/// emulated as an assistant note.
pub fn transform_codex_request(
	body: &mut Value,
	responses_lite: bool,
) -> Result<Vec<Unsupported>, Error> {
	transform_codex_request_with_concurrent_summaries(
		body,
		responses_lite,
		concurrent_summaries_enabled(),
	)
}

fn transform_codex_request_with_concurrent_summaries(
	body: &mut Value,
	responses_lite: bool,
	concurrent_summaries: bool,
) -> Result<Vec<Unsupported>, Error> {
	let object = body
		.as_object_mut()
		.ok_or_else(|| Error::Provider(Str::new("Codex request body must be a JSON object")))?;
	object.insert("store".into(), Value::Bool(false));
	object.insert("stream".into(), Value::Bool(true));
	include_encrypted_reasoning(object);
	apply_reasoning_summary_controls(object);
	remove_rejected_controls(object);
	filter_replayed_items(object);
	let sent_calls = BTreeSet::new();
	let unsupported = object
		.get_mut("input")
		.and_then(Value::as_array_mut)
		.map_or_else(Vec::new, |input| repair_responses_tool_pairs(input, &sent_calls));
	ensure_visible_input(object);
	if responses_lite {
		apply_responses_lite(object);
	}
	apply_reasoning_summary_delivery(object, concurrent_summaries);
	Ok(unsupported)
}

fn apply_reasoning_summary_controls(object: &mut Map<String, Value>) {
	let supports_summary = object
		.get("model")
		.and_then(Value::as_str)
		.is_some_and(supports_reasoning_summary);
	let Some(reasoning) = object.get_mut("reasoning").and_then(Value::as_object_mut) else {
		return;
	};
	if !supports_summary {
		reasoning.remove("summary");
		return;
	}
	if reasoning.get("summary").is_some_and(Value::is_null) {
		reasoning.remove("summary");
	} else if reasoning.contains_key("effort") && !reasoning.contains_key("summary") {
		reasoning.insert("summary".into(), Value::String("auto".into()));
	}
}

fn apply_reasoning_summary_delivery(object: &mut Map<String, Value>, concurrent_summaries: bool) {
	let summary_requested = object
		.get("reasoning")
		.and_then(Value::as_object)
		.is_some_and(|reasoning| reasoning.contains_key("summary"));
	if summary_requested && concurrent_summaries {
		object.insert(
			"stream_options".into(),
			json!({"reasoning_summary_delivery": "sequential_cutoff"}),
		);
	} else {
		object.remove("stream_options");
	}
}

fn concurrent_summaries_enabled() -> bool {
	env::var("OMP_CODEX_CONCURRENT_SUMMARIES").is_ok_and(|value| {
		let value = value.trim();
		value == "1" || value.eq_ignore_ascii_case("true")
	})
}

fn supports_reasoning_summary(model: &str) -> bool {
	let model = model.rsplit('/').next().unwrap_or(model);
	let Some(version) = model
		.get(..4)
		.filter(|prefix| prefix.eq_ignore_ascii_case("gpt-"))
		.and_then(|_| model.get(4..))
	else {
		return false;
	};
	let major_end = version
		.bytes()
		.position(|byte| !byte.is_ascii_digit())
		.unwrap_or(version.len());
	let Ok(major) = version[..major_end].parse::<u32>() else {
		return false;
	};
	if major != 5 {
		return major > 5;
	}
	let Some(minor) = version[major_end..].strip_prefix('.') else {
		return false;
	};
	let minor_end = minor
		.bytes()
		.position(|byte| !byte.is_ascii_digit())
		.unwrap_or(minor.len());
	minor[..minor_end]
		.parse::<u32>()
		.is_ok_and(|minor| minor >= 4)
}

fn include_encrypted_reasoning(object: &mut Map<String, Value>) {
	let include = object
		.entry("include")
		.or_insert_with(|| Value::Array(Vec::new()));
	let values = include.as_array_mut();
	if let Some(values) = values {
		if !values
			.iter()
			.any(|value| value.as_str() == Some("reasoning.encrypted_content"))
		{
			values.push(Value::String("reasoning.encrypted_content".into()));
		}
	} else {
		*include = json!(["reasoning.encrypted_content"]);
	}
}

fn remove_rejected_controls(object: &mut Map<String, Value>) {
	for key in [
		"temperature",
		"top_p",
		"top_k",
		"min_p",
		"presence_penalty",
		"frequency_penalty",
		"repetition_penalty",
		"stop",
		"max_output_tokens",
		"max_completion_tokens",
	] {
		object.remove(key);
	}
}

fn filter_replayed_items(object: &mut Map<String, Value>) {
	let Some(input) = object.get_mut("input").and_then(Value::as_array_mut) else {
		return;
	};
	input.retain(|item| item.get("type").and_then(Value::as_str) != Some("item_reference"));
	for item in input {
		if item.get("type").and_then(Value::as_str) != Some("computer_call")
			&& let Some(item) = item.as_object_mut()
		{
			item.remove("id");
		}
	}
}

fn ensure_visible_input(object: &mut Map<String, Value>) {
	let instruction = object
		.get("instructions")
		.and_then(Value::as_str)
		.filter(|value| !value.trim().is_empty())
		.map(str::to_owned);
	let Some(instruction) = instruction else {
		return;
	};
	let input = object
		.entry("input")
		.or_insert_with(|| Value::Array(Vec::new()));
	let Some(input) = input.as_array_mut() else {
		return;
	};
	if input
		.iter()
		.any(|item| item.get("role").and_then(Value::as_str) != Some("developer"))
	{
		return;
	}
	input.push(json!({
		"type": "message",
		"role": "user",
		"content": [{"type": "input_text", "text": instruction}],
	}));
}

fn apply_responses_lite(object: &mut Map<String, Value>) {
	if let Some(input) = object.get_mut("input") {
		strip_image_detail(input);
	}
	object.insert("parallel_tool_calls".into(), Value::Bool(false));

	let declared_tools = object
		.remove("tools")
		.and_then(|value| value.as_array().cloned())
		.unwrap_or_default();
	let mut additional_tools = declared_tools.clone();
	let choice = object.get("tool_choice").cloned();
	if let Some(choice) = choice.as_ref().and_then(Value::as_object) {
		let selected = select_forced_tool(choice, &declared_tools);
		if let Some(selected) = selected {
			additional_tools = vec![selected.clone()];
			object.insert("tool_choice".into(), Value::String("required".into()));
		} else {
			object.insert("tool_choice".into(), Value::String("auto".into()));
		}
	} else if !matches!(choice.as_ref().and_then(Value::as_str), Some("none" | "required")) {
		object.insert("tool_choice".into(), Value::String("auto".into()));
	}

	let mut prefix = vec![json!({
		"type": "additional_tools",
		"role": "developer",
		"tools": additional_tools,
	})];
	if let Some(instructions) = object
		.remove("instructions")
		.and_then(|value| value.as_str().map(str::to_owned))
		&& !instructions.is_empty()
	{
		prefix.push(json!({
			"type": "message",
			"role": "developer",
			"content": [{"type": "input_text", "text": instructions}],
		}));
	}
	let input = object
		.remove("input")
		.and_then(|value| value.as_array().cloned())
		.unwrap_or_default();
	prefix.extend(input);
	object.insert("input".into(), Value::Array(prefix));

	let reasoning = object
		.entry("reasoning")
		.or_insert_with(|| Value::Object(Map::new()));
	if !reasoning.is_object() {
		*reasoning = Value::Object(Map::new());
	}
	reasoning
		.as_object_mut()
		.expect("reasoning was normalized to an object")
		.insert("context".into(), Value::String("all_turns".into()));
}

fn select_forced_tool<'a>(choice: &Map<String, Value>, tools: &'a [Value]) -> Option<&'a Value> {
	let choice_type = choice.get("type").and_then(Value::as_str)?;
	let choice_name = choice.get("name").and_then(Value::as_str);
	tools.iter().find(|tool| {
		let Some(tool) = tool.as_object() else {
			return false;
		};
		match choice_type {
			"computer" => tool.get("type").and_then(Value::as_str) == Some("computer"),
			"function" => {
				tool.get("type").and_then(Value::as_str) == Some("function")
					&& tool.get("name").and_then(Value::as_str) == choice_name
			},
			_ => false,
		}
	})
}

fn strip_image_detail(value: &mut Value) {
	match value {
		Value::Array(values) => {
			for value in values {
				strip_image_detail(value);
			}
		},
		Value::Object(object) => {
			if object.get("type").and_then(Value::as_str) == Some("input_image") {
				object.remove("detail");
			}
			for value in object.values_mut() {
				strip_image_detail(value);
			}
		},
		_ => {},
	}
}

#[cfg(test)]
mod tests {
	use omp_llm_types::UnsupportedAction;
	use serde_json::json;

	use super::{transform_codex_request, transform_codex_request_with_concurrent_summaries};

	#[test]
	fn codex_transform_uses_shared_pair_repair() {
		let mut body = json!({
			"input": [
				{
					"type": "function_call_output",
					"call_id": "orphan_output",
					"output": "failed",
				},
				{
					"type": "function_call",
					"call_id": "orphan_call",
					"name": "lookup",
					"arguments": "{}",
				},
			],
		});
		let unsupported = transform_codex_request(&mut body, true).unwrap();
		assert_eq!(
			&body["input"].as_array().unwrap()[1..],
			json!([
				{
					"type": "message",
					"role": "assistant",
					"content": "[Orphan tool result; call_id=orphan_output]: failed",
				},
				{
					"type": "function_call",
					"call_id": "orphan_call",
					"name": "lookup",
					"arguments": "{}",
				},
				{
					"type": "function_call_output",
					"call_id": "orphan_call",
					"output": "[No tool output recorded: the tool call was interrupted before it produced a result.]",
				},
			])
			.as_array()
			.unwrap(),
		);
		assert_eq!(unsupported.len(), 1);
		assert_eq!(unsupported[0].what, "thread.tool_result");
		assert_eq!(unsupported[0].action, UnsupportedAction::Emulated);
	}

	#[test]
	fn codex_reasoning_summaries_default_on_and_delivery_is_opt_in() {
		let mut defaulted = json!({
			"model": "gpt-5.5",
			"reasoning": {"effort": "medium"},
		});
		transform_codex_request_with_concurrent_summaries(&mut defaulted, false, false).unwrap();
		assert_eq!(defaulted["reasoning"], json!({"effort": "medium", "summary": "auto"}));
		assert!(defaulted.get("stream_options").is_none());

		let mut opted_in = json!({
			"model": "gpt-5.6-terra",
			"reasoning": {"effort": "medium", "summary": "detailed"},
		});
		transform_codex_request_with_concurrent_summaries(&mut opted_in, false, true).unwrap();
		assert_eq!(
			opted_in["stream_options"],
			json!({"reasoning_summary_delivery": "sequential_cutoff"}),
		);
		assert_eq!(opted_in["reasoning"]["summary"], "detailed");

		let mut suppressed = json!({
			"model": "gpt-5.6-terra",
			"reasoning": {"effort": "medium", "summary": null},
		});
		transform_codex_request_with_concurrent_summaries(&mut suppressed, false, true).unwrap();
		assert_eq!(suppressed["reasoning"], json!({"effort": "medium"}));
		assert!(suppressed.get("stream_options").is_none());

		let mut unsupported = json!({
			"model": "gpt-5.3-codex",
			"reasoning": {"effort": "medium", "summary": "detailed"},
		});
		transform_codex_request_with_concurrent_summaries(&mut unsupported, false, true).unwrap();
		assert_eq!(unsupported["reasoning"], json!({"effort": "medium"}));
		assert!(unsupported.get("stream_options").is_none());
	}
}
