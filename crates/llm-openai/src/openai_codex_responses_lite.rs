//! ChatGPT Codex Responses Lite request shaping.
//!
//! This module only transforms JSON wire bodies. Authentication and HTTP
//! dispatch remain owned by the gateway egress stack.

use std::collections::BTreeMap;

use omp_core::SmolStr;
use omp_llm_types::Error;
use serde_json::{Map, Value, json};

/// Provider-option namespace for Codex-only controls.
pub const CODEX_PROVIDER_NAMESPACE: &str = "openai-codex";

/// Provider-option name selecting the Responses Lite wire shape.
pub const RESPONSES_LITE_OPTION: &str = "responses_lite";

/// Applies the Codex Responses contract and, when requested, the Responses Lite
/// rewrite to an encoded OpenAI Responses request.
///
/// The rewrite is intentionally lossless for conversation items while removing
/// fields rejected by the subscription endpoint. Lite requests move top-level
/// instructions and tool declarations into leading developer input items, force
/// serial tool use, and leave image detail selection to the server.
pub fn transform_codex_request(body: &mut Value, responses_lite: bool) -> Result<(), Error> {
	let object = body
		.as_object_mut()
		.ok_or_else(|| Error::Provider(SmolStr::new("Codex request body must be a JSON object")))?;
	object.insert("store".into(), Value::Bool(false));
	object.insert("stream".into(), Value::Bool(true));
	include_encrypted_reasoning(object);
	remove_rejected_controls(object);
	filter_replayed_items(object);
	repair_tool_pairs(object);
	ensure_visible_input(object);
	if responses_lite {
		apply_responses_lite(object);
	}
	Ok(())
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

fn repair_tool_pairs(object: &mut Map<String, Value>) {
	let Some(input) = object.get_mut("input").and_then(Value::as_array_mut) else {
		return;
	};
	let mut calls = BTreeMap::new();
	let mut outputs = BTreeMap::new();
	for item in input.iter() {
		let Some(call_id) = item.get("call_id").and_then(Value::as_str) else {
			continue;
		};
		if let Some(kind) = tool_call_kind(item.get("type").and_then(Value::as_str)) {
			calls.insert(call_id.to_owned(), kind);
		}
		if let Some(kind) = tool_output_kind(item.get("type").and_then(Value::as_str)) {
			outputs.insert(call_id.to_owned(), kind);
		}
	}
	let mut repaired = Vec::with_capacity(input.len());
	for item in std::mem::take(input) {
		let call_id = item
			.get("call_id")
			.and_then(Value::as_str)
			.map(str::to_owned);
		let call_kind = tool_call_kind(item.get("type").and_then(Value::as_str));
		let output_kind = tool_output_kind(item.get("type").and_then(Value::as_str));
		if let (Some(call_id), Some(output_kind)) = (call_id.as_deref(), output_kind)
			&& calls.get(call_id).copied() != Some(output_kind)
		{
			let output = item.get("output").cloned().unwrap_or(Value::Null);
			repaired.push(json!({
				"type": "message",
				"role": "assistant",
				"content": format!("[Previous tool result; call_id={call_id}]: {output}"),
			}));
			continue;
		}
		if let (Some(call_id), Some(call_kind)) = (call_id.as_deref(), call_kind)
			&& outputs.get(call_id).copied() != Some(call_kind)
		{
			if call_kind == "computer" {
				repaired.push(json!({
					"type": "message",
					"role": "assistant",
					"content": format!("[Computer call interrupted before a screenshot was recorded; call_id={call_id}]"),
				}));
			} else {
				repaired.push(item);
				repaired.push(json!({
					"type": if call_kind == "custom" { "custom_tool_call_output" } else { "function_call_output" },
					"call_id": call_id,
					"output": "[No tool output recorded: the tool call was interrupted before it produced a result.]",
				}));
			}
			continue;
		}
		repaired.push(item);
	}
	*input = repaired;
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

fn tool_call_kind(kind: Option<&str>) -> Option<&'static str> {
	match kind {
		Some("function_call") => Some("function"),
		Some("custom_tool_call") => Some("custom"),
		Some("computer_call") => Some("computer"),
		_ => None,
	}
}

fn tool_output_kind(kind: Option<&str>) -> Option<&'static str> {
	match kind {
		Some("function_call_output") => Some("function"),
		Some("custom_tool_call_output") => Some("custom"),
		Some("computer_call_output") => Some("computer"),
		_ => None,
	}
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
