use std::collections::BTreeSet;

use omp_core::Str;
use omp_llm_types::{Unsupported, UnsupportedAction};
use serde_json::{Value, json};
use xutf::{Utf8, Utf16};

const ORPHAN_OUTPUT_LIMIT: usize = 16_000;
const ORPHAN_TOOL_CALL_PLACEHOLDER: &str =
	"[No tool output recorded: the tool call was interrupted before it produced a result.]";

/// Responses tool-pair family. A call and its output pair only within one kind.
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub enum ToolKind {
	Function,
	Custom,
	Computer,
}

/// Wire identity of a tool call already delivered to the server.
pub type SentCall = (ToolKind, Str);

/// Applies pi's directional orphan repair to a Responses `input` array.
///
/// `sent_calls` names the tool calls that a stateful continuation already
/// delivered under `previous_response_id`; their outputs in `input` are
/// correctly paired and must not be folded into assistant notes.
pub fn repair_responses_tool_pairs(
	input: &mut Vec<Value>,
	sent_calls: &BTreeSet<SentCall>,
) -> Vec<Unsupported> {
	let mut unsupported = Vec::new();
	let mut preceding_calls = sent_calls.clone();
	for item in input.iter_mut() {
		let call_id = item.get("call_id").and_then(Value::as_str).map(Str::from);
		if let (Some(kind), Some(call_id)) = (tool_call_kind(item), call_id.as_ref()) {
			preceding_calls.insert((kind, call_id.clone()));
		}
		let Some(kind) = tool_output_kind(item) else {
			continue;
		};
		let Some(call_id) = call_id else {
			continue;
		};
		if preceding_calls.contains(&(kind, call_id.clone())) {
			continue;
		}

		let text = orphan_output_text(item.get("output"));
		let tool_name = if kind == ToolKind::Computer {
			"computer"
		} else {
			"tool"
		};
		*item = json!({
			"type": "message",
			"role": "assistant",
			"content": format!("[Orphan {tool_name} result; call_id={call_id}]: {text}"),
		});
		unsupported.push(repair_report(
			"thread.tool_result",
			"orphan Responses tool output was converted to an assistant note",
		));
	}

	let mut later_outputs = BTreeSet::new();
	let mut orphan_indexes = BTreeSet::new();
	for (index, item) in input.iter().enumerate().rev() {
		let Some(call_id) = item.get("call_id").and_then(Value::as_str) else {
			continue;
		};
		let call_id = Str::from(call_id);
		if let Some(kind) = tool_output_kind(item) {
			later_outputs.insert((kind, call_id.clone()));
		}
		if let Some(kind) = tool_call_kind(item)
			&& !later_outputs.contains(&(kind, call_id))
		{
			orphan_indexes.insert(index);
		}
	}
	if orphan_indexes.is_empty() {
		return unsupported;
	}

	let mut repaired = Vec::with_capacity(input.len() + orphan_indexes.len());
	for (index, item) in std::mem::take(input).into_iter().enumerate() {
		if !orphan_indexes.contains(&index) {
			repaired.push(item);
			continue;
		}
		let Some(call_id) = item.get("call_id").and_then(Value::as_str).map(Str::from) else {
			repaired.push(item);
			continue;
		};
		match tool_call_kind(&item) {
			Some(ToolKind::Computer) => {
				repaired.push(json!({
					"type": "message",
					"role": "assistant",
					"content": format!("[Computer call interrupted before a screenshot was recorded; call_id={call_id}]"),
				}));
				unsupported.push(repair_report(
					"thread.tool_call",
					"orphan Responses computer call was converted to an assistant note",
				));
			},
			Some(kind @ (ToolKind::Function | ToolKind::Custom)) => {
				repaired.push(item);
				repaired.push(json!({
					"type": if kind == ToolKind::Custom {
						"custom_tool_call_output"
					} else {
						"function_call_output"
					},
					"call_id": call_id,
					"output": ORPHAN_TOOL_CALL_PLACEHOLDER,
				}));
			},
			None => repaired.push(item),
		}
	}
	*input = repaired;
	unsupported
}

fn orphan_output_text(output: Option<&Value>) -> String {
	let mut text = match output {
		Some(Value::String(text)) => text.clone(),
		None | Some(Value::Null) => String::new(),
		Some(output) => serde_json::to_string(output).unwrap_or_default(),
	};
	if xutf::transcoded_len::<Utf8, Utf16>(text.as_bytes()) <= ORPHAN_OUTPUT_LIMIT {
		return text;
	}
	let mut utf16 = xutf::transcode::<Utf8, Utf16>(text.as_bytes());
	utf16.truncate(ORPHAN_OUTPUT_LIMIT);
	let prefix = xutf::transcode::<Utf16, Utf8>(&utf16);
	text = String::from_utf8(prefix).unwrap_or_default();
	text.push_str("\n...[truncated]");
	text
}

fn repair_report(what: &str, detail: &str) -> Unsupported {
	Unsupported::builder()
		.what(Str::from(what))
		.detail(Str::from(detail))
		.action(UnsupportedAction::Emulated)
		.build()
}

fn tool_call_kind(item: &Value) -> Option<ToolKind> {
	match item.get("type").and_then(Value::as_str) {
		Some("function_call") => Some(ToolKind::Function),
		Some("custom_tool_call") => Some(ToolKind::Custom),
		Some("computer_call") => Some(ToolKind::Computer),
		_ => None,
	}
}

fn tool_output_kind(item: &Value) -> Option<ToolKind> {
	match item.get("type").and_then(Value::as_str) {
		Some("function_call_output") => Some(ToolKind::Function),
		Some("custom_tool_call_output") => Some(ToolKind::Custom),
		Some("computer_call_output") => Some(ToolKind::Computer),
		_ => None,
	}
}

/// Maps a Responses item type to the tool-call family it belongs to.
pub fn call_kind_of(type_name: &str) -> Option<ToolKind> {
	tool_call_kind(&json!({ "type": type_name }))
}

#[cfg(test)]
mod tests {
	use std::collections::BTreeSet;

	use omp_core::Str;
	use serde_json::{Value, json};

	use super::{ToolKind, repair_responses_tool_pairs};

	#[test]
	fn paired_items_are_byte_identical() {
		let mut input = vec![
			json!({"type":"function_call","call_id":"call_a","name":"lookup","arguments":"{}","id":"fc_1"}),
			json!({"type":"function_call_output","call_id":"call_a","output":"ok","id":"out_1"}),
		];
		let before = serde_json::to_vec(&input).unwrap();
		assert_eq!(repair_responses_tool_pairs(&mut input, &BTreeSet::new()), [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(serde_json::to_vec(&input).unwrap(), before);
	}

	#[test]
	fn interleaved_pairs_keep_order_and_directional_pairing() {
		let mut input = vec![
			json!({"type":"function_call","call_id":"a","name":"a","arguments":"{}"}),
			json!({"type":"function_call","call_id":"b","name":"b","arguments":"{}"}),
			json!({"type":"function_call_output","call_id":"a","output":"A"}),
			json!({"type":"message","role":"user","content":"between"}),
			json!({"type":"function_call_output","call_id":"b","output":"B"}),
		];
		assert_eq!(repair_responses_tool_pairs(&mut input, &BTreeSet::new()), [] as [omp_llm_types::Unsupported; 0]);
		let call_ids = input
			.iter()
			.map(|item| item.get("call_id").cloned().unwrap_or(Value::Null))
			.collect::<Vec<_>>();
		assert_eq!(call_ids, vec![json!("a"), json!("b"), json!("a"), Value::Null, json!("b")]);
	}

	#[test]
	fn output_paired_by_an_already_sent_call_is_left_intact() {
		let mut input =
			vec![json!({"type":"function_call_output","call_id":"call_sent","output":"Sunny"})];
		let sent = BTreeSet::from([(ToolKind::Function, Str::new_static("call_sent"))]);
		assert_eq!(repair_responses_tool_pairs(&mut input, &sent), [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(input[0]["type"], "function_call_output");

		let mut unsent = input.clone();
		let reports = repair_responses_tool_pairs(&mut unsent, &BTreeSet::new());
		assert_eq!(reports.len(), 1);
		assert_eq!(unsent[0]["type"], "message");
	}
}
