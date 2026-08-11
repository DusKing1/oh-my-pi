//! Borrowed rendering primitives and complete owned-dialect transcripts.

use std::{fmt, io};

use omp_llm_types::{Item, ItemKind, Message, Part, Role, Thread, ToolCall, ToolResult};
use serde_json::Value;

use crate::{
	Dialect,
	types::{DialectError, DialectRenderOptions, DialectResult, DialectToolResult},
};

const DS_CALLS_BEGIN: &str = "<｜tool▁calls▁begin｜>";
const DS_CALLS_END: &str = "<｜tool▁calls▁end｜>";
const DS_CALL_BEGIN: &str = "<｜tool▁call▁begin｜>";
const DS_CALL_END: &str = "<｜tool▁call▁end｜>";
const DS_SEPARATOR: &str = "<｜tool▁sep｜>";
const DS_OUTPUT_BEGIN: &str = "<｜tool▁output▁begin｜>";
const DS_OUTPUT_END: &str = "<｜tool▁output▁end｜>";
const H_START: &str = "<\u{7c}start\u{7c}>";
const H_END: &str = "<\u{7c}end\u{7c}>";
const H_CHANNEL: &str = "<\u{7c}channel\u{7c}>";
const H_MESSAGE: &str = "<\u{7c}message\u{7c}>";
const H_CALL: &str = "<\u{7c}call\u{7c}>";
const GEMMA_STRING: &str = "<|\"|>";

/// Writes a JSON value directly into a formatting destination.
///
/// This adapter avoids constructing a serialized `String` for large schemas,
/// arguments, and nested values.
pub fn write_json_value<W: fmt::Write + ?Sized>(out: &mut W, value: &Value) -> fmt::Result {
	serde_json::to_writer(FmtIo(out), value).map_err(|_| fmt::Error)
}

/// Writes a Python keyword call from a JSON object.
///
/// `leading` is emitted before schema arguments and is used by prompt examples
/// to teach an injected intent field without cloning the argument object.
pub fn write_py_call_value<W: fmt::Write + ?Sized>(
	out: &mut W,
	name: &str,
	arguments: &Value,
	leading: Option<(&str, &str)>,
) -> fmt::Result {
	out.write_str(name)?;
	out.write_str("(")?;
	let mut separated = if let Some((key, value)) = leading {
		write!(out, "{key}=")?;
		write_py_string(out, value)?;
		true
	} else {
		false
	};
	if let Some(arguments) = arguments.as_object() {
		for (key, value) in arguments {
			if separated {
				out.write_str(", ")?;
			}
			write!(out, "{key}=")?;
			write_py_arg_value(out, value)?;
			separated = true;
		}
	}
	out.write_str(")")
}

/// Escapes reserved Harmony tokens in arbitrary message text.
///
/// The persisted source remains untouched; only the rendered transport copy is
/// escaped, preventing data from opening or closing synthetic turns.
pub fn write_escaped_harmony_text<W: fmt::Write + ?Sized>(out: &mut W, text: &str) -> fmt::Result {
	write_harmony_escaped(out, text, false)
}

/// Escapes reserved Harmony tokens inside a JSON document while retaining
/// valid JSON string escaping.
pub fn write_escaped_harmony_json<W: fmt::Write + ?Sized>(out: &mut W, json: &str) -> fmt::Result {
	write_harmony_escaped(out, json, true)
}

/// Writes one dialect-native tool invocation without a parallel-call envelope.
pub fn render_tool_call<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	call: &ToolCall,
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	match dialect {
		Dialect::Anthropic | Dialect::Xml | Dialect::MiniMax => write_xml_invoke(out, call, options)?,
		Dialect::DeepSeek => {
			out.write_str(DS_CALL_BEGIN)?;
			out.write_str(call.name.as_str())?;
			out.write_str(DS_SEPARATOR)?;
			write_args_raw(out, call, false)?;
			out.write_str(DS_CALL_END)?;
		},
		Dialect::Gemini => write_gemini_call(out, call)?,
		Dialect::Gemma => write_gemma_call(out, call)?,
		Dialect::Glm => write_glm_call(out, call, options)?,
		Dialect::Harmony => {
			out.write_str(H_START)?;
			out.write_str("assistant")?;
			out.write_str(H_CHANNEL)?;
			out.write_str("commentary to=")?;
			write_harmony_recipient(out, call.name.as_str())?;
			out.write_str(H_MESSAGE)?;
			write_args_raw(out, call, true)?;
			out.write_str(H_CALL)?;
		},
		Dialect::Hermes | Dialect::Qwen3 => {
			out.write_str("<tool_call>\n{\"name\":")?;
			write_json_string(out, call.name.as_str())?;
			out.write_str(",\"arguments\":")?;
			write_args_raw(out, call, false)?;
			out.write_str("}\n</tool_call>")?;
		},
		Dialect::Kimi => write_kimi_call(out, call, 0)?,
	}
	Ok(())
}

/// Writes a complete one-or-many assistant tool-call block for a dialect.
pub fn render_assistant_tool_calls<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	calls: &[ToolCall],
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	write_call_run(out, dialect, calls, options)
}

/// Writes one consecutive run of tool results with the target dialect's exact
/// grouping and envelopes.
pub fn render_tool_results<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	results: &[DialectToolResult<'_>],
	_options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	write_result_run(out, dialect, results)
}

/// Writes model reasoning in the target dialect's native text transport form.
pub fn render_thinking<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	text: &str,
) -> fmt::Result {
	if text.is_empty() {
		return Ok(());
	}
	match dialect {
		Dialect::Anthropic | Dialect::Hermes | Dialect::MiniMax | Dialect::Xml => {
			write_delimited_thinking(out, "<thinking>", "</thinking>", text)
		},
		Dialect::DeepSeek | Dialect::Glm | Dialect::Kimi | Dialect::Qwen3 => {
			write!(out, "<think>\n{text}\n</think>")
		},
		Dialect::Gemini => write!(out, "```thinking\n{text}\n```"),
		Dialect::Gemma => write!(out, "<|channel>thought{text}<channel|>"),
		Dialect::Harmony => {
			out.write_str(H_START)?;
			out.write_str("assistant")?;
			out.write_str(H_CHANNEL)?;
			out.write_str("analysis")?;
			out.write_str(H_MESSAGE)?;
			write_escaped_harmony_text(out, text)?;
			out.write_str(H_END)
		},
	}
}

/// Serializes a canonical thread into a complete model-native transcript.
///
/// Consecutive calls and results are consumed as runs directly from the source
/// slice. No intermediate message vector or accumulated per-delta string is
/// constructed.
pub fn render_transcript<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	thread: &Thread,
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	match dialect {
		Dialect::Anthropic | Dialect::MiniMax | Dialect::Xml => {
			write_legacy_transcript(out, dialect, thread, options)?;
		},
		Dialect::Hermes | Dialect::Qwen3 => write_chatml_transcript(out, dialect, thread, options)?,
		Dialect::DeepSeek => write_deepseek_transcript(out, thread, options)?,
		Dialect::Gemini => write_gemini_transcript(out, thread, options)?,
		Dialect::Gemma => write_gemma_transcript(out, thread, options)?,
		Dialect::Glm => write_glm_transcript(out, thread, options)?,
		Dialect::Harmony => write_harmony_transcript(out, thread, options)?,
		Dialect::Kimi => write_kimi_transcript(out, thread, options)?,
	}
	Ok(())
}

fn write_call_run<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	calls: &[ToolCall],
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	if calls.is_empty() {
		return Ok(());
	}
	match dialect {
		Dialect::Anthropic => out.write_str("<function_calls>\n")?,
		Dialect::DeepSeek => out.write_str(DS_CALLS_BEGIN)?,
		Dialect::Gemini => out.write_str("```tool_code\n")?,
		Dialect::Kimi => out.write_str("<|tool_calls_section_begin|>")?,
		Dialect::MiniMax => out.write_str("<minimax:tool_call>\n")?,
		_ => {},
	}
	if dialect == Dialect::Gemini && calls.len() > 1 {
		out.write_str("[")?;
	}
	for (index, call) in calls.iter().enumerate() {
		if index != 0
			&& matches!(
				dialect,
				Dialect::Anthropic
					| Dialect::Glm
					| Dialect::Hermes
					| Dialect::MiniMax
					| Dialect::Qwen3
					| Dialect::Xml
			) {
			out.write_str("\n")?;
		}
		if dialect == Dialect::Gemini && index != 0 {
			out.write_str(", ")?;
		}
		if dialect == Dialect::Kimi {
			write_kimi_call(out, call, index)?;
		} else {
			render_tool_call(out, dialect, call, options)?;
		}
	}
	if dialect == Dialect::Gemini && calls.len() > 1 {
		out.write_str("]")?;
	}
	match dialect {
		Dialect::Anthropic => out.write_str("\n</function_calls>")?,
		Dialect::DeepSeek => out.write_str(DS_CALLS_END)?,
		Dialect::Gemini => out.write_str("\n```")?,
		Dialect::Kimi => out.write_str("<|tool_calls_section_end|>")?,
		Dialect::MiniMax => out.write_str("\n</minimax:tool_call>")?,
		_ => {},
	}
	Ok(())
}

fn write_result_run<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	results: &[DialectToolResult<'_>],
) -> DialectResult<()> {
	match dialect {
		Dialect::Anthropic | Dialect::MiniMax => out.write_str("<function_results>\n")?,
		Dialect::Glm => out.write_str("<observation>\n")?,
		_ => {},
	}
	for (index, result) in results.iter().enumerate() {
		if index != 0 && !matches!(dialect, Dialect::Gemma | Dialect::Harmony | Dialect::Kimi) {
			out.write_str("\n")?;
		}
		write_result(out, dialect, result)?;
	}
	match dialect {
		Dialect::Anthropic | Dialect::MiniMax => out.write_str("\n</function_results>")?,
		Dialect::Glm => out.write_str("\n</observation>")?,
		_ => {},
	}
	Ok(())
}

fn write_result<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	result: &DialectToolResult<'_>,
) -> fmt::Result {
	match dialect {
		Dialect::Anthropic | Dialect::MiniMax => {
			let (tag, stream) = if result.is_error {
				("error", "stderr")
			} else {
				("result", "stdout")
			};
			write!(out, "<{tag}>\n<tool_name>")?;
			write_xml_text(out, result.name)?;
			write!(out, "</tool_name>\n<{stream}>")?;
			out.write_str(result.text)?;
			write!(out, "</{stream}>\n</{tag}>")
		},
		Dialect::DeepSeek => write!(out, "{DS_OUTPUT_BEGIN}{}{DS_OUTPUT_END}", result.text),
		Dialect::Gemini => write!(out, "```tool_outputs\n{}\n```", result.text),
		Dialect::Gemma => {
			out.write_str("<|tool_response>response:")?;
			out.write_str(result.name)?;
			out.write_str("{output:")?;
			if let Ok(value) = serde_json::from_str::<Value>(result.text) {
				write_gemma_value(out, &value)?;
			} else {
				out.write_str(GEMMA_STRING)?;
				out.write_str(result.text)?;
				out.write_str(GEMMA_STRING)?;
			}
			out.write_str("}<tool_response|>")
		},
		Dialect::Glm | Dialect::Hermes | Dialect::Qwen3 | Dialect::Xml => {
			write!(out, "<tool_response>\n{}\n</tool_response>", result.text)
		},
		Dialect::Harmony => {
			out.write_str(H_START)?;
			write_harmony_recipient(out, result.name)?;
			out.write_str(" to=assistant")?;
			out.write_str(H_CHANNEL)?;
			out.write_str("commentary")?;
			out.write_str(H_MESSAGE)?;
			write_escaped_harmony_text(out, result.text)?;
			out.write_str(H_END)
		},
		Dialect::Kimi => {
			out.write_str("<|im_system|>")?;
			out.write_str(result.name)?;
			out.write_str("<|im_middle|>## Return of ")?;
			write_kimi_result_id(out, result.name, result.id, result.index)?;
			write!(out, "\n{}<|im_end|>", result.text)
		},
	}
}

fn write_legacy_transcript<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	thread: &Thread,
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	let items = &thread.items;
	let mut index = 0;
	let mut emitted = false;
	while index < items.len() {
		match &items[index].kind {
			ItemKind::Message(message) if message.role == Role::System => {
				if emitted {
					out.write_str("\n\n")?;
				}
				write_message_text(out, message, dialect == Dialect::Harmony)?;
				emitted = true;
				index += 1;
			},
			ItemKind::Message(message) if message.role == Role::Assistant => {
				out.write_str("\n\nAssistant: ")?;
				write_assistant_parts(out, dialect, message)?;
				let end = call_run_end(items, index + 1);
				write_call_run_from_items(out, dialect, &items[index + 1..end], options)?;
				index = end;
				emitted = true;
			},
			ItemKind::ToolCall(_) => {
				let end = call_run_end(items, index);
				out.write_str("\n\nAssistant: ")?;
				write_call_run_from_items(out, dialect, &items[index..end], options)?;
				index = end;
				emitted = true;
			},
			ItemKind::ToolResult(_) => {
				let end = result_run_end(items, index);
				out.write_str("\n\nHuman: ")?;
				write_canonical_result_run(out, dialect, &items[index..end])?;
				index = end;
				emitted = true;
			},
			ItemKind::Message(message) => {
				out.write_str("\n\nHuman: ")?;
				write_message_text(out, message, dialect == Dialect::Harmony)?;
				index += 1;
				emitted = true;
			},
			_ => {
				index += 1;
			},
		}
	}
	Ok(())
}

fn write_chatml_transcript<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	thread: &Thread,
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	let items = &thread.items;
	let mut index = 0;
	while index < items.len() {
		match &items[index].kind {
			ItemKind::Message(message) if message.role == Role::Assistant => {
				write_chatml_open(out, "assistant")?;
				write_assistant_parts(out, dialect, message)?;
				let end = call_run_end(items, index + 1);
				write_call_run_from_items(out, dialect, &items[index + 1..end], options)?;
				out.write_str("<|im_end|>\n")?;
				index = end;
			},
			ItemKind::ToolCall(_) => {
				let end = call_run_end(items, index);
				write_chatml_open(out, "assistant")?;
				write_call_run_from_items(out, dialect, &items[index..end], options)?;
				out.write_str("<|im_end|>\n")?;
				index = end;
			},
			ItemKind::ToolResult(_) => {
				let end = result_run_end(items, index);
				write_chatml_open(
					out,
					if dialect == Dialect::Hermes {
						"tool"
					} else {
						"user"
					},
				)?;
				write_canonical_result_run(out, dialect, &items[index..end])?;
				out.write_str("<|im_end|>\n")?;
				index = end;
			},
			ItemKind::Message(message) => {
				write_chatml_open(
					out,
					if message.role == Role::System {
						"system"
					} else {
						"user"
					},
				)?;
				write_message_text(out, message, false)?;
				out.write_str("<|im_end|>\n")?;
				index += 1;
			},
			_ => {
				index += 1;
			},
		}
	}
	Ok(())
}

fn write_deepseek_transcript<W: fmt::Write + ?Sized>(
	out: &mut W,
	thread: &Thread,
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	if thread.items.is_empty() {
		return Ok(());
	}
	out.write_str("<｜begin▁of▁sentence｜>")?;
	write_simple_token_transcript(
		out,
		Dialect::DeepSeek,
		thread,
		options,
		"<｜User｜>",
		"<｜Assistant｜>",
		"<｜end▁of▁sentence｜>",
	)
}

fn write_glm_transcript<W: fmt::Write + ?Sized>(
	out: &mut W,
	thread: &Thread,
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	if thread.items.is_empty() {
		return Ok(());
	}
	out.write_str("[gMASK]<sop>")?;
	let items = &thread.items;
	let mut index = 0;
	while index < items.len() {
		match &items[index].kind {
			ItemKind::Message(message) if message.role == Role::Assistant => {
				out.write_str("<|assistant|>\n")?;
				write_assistant_parts(out, Dialect::Glm, message)?;
				let end = call_run_end(items, index + 1);
				write_call_run_from_items(out, Dialect::Glm, &items[index + 1..end], options)?;
				index = end;
			},
			ItemKind::ToolCall(_) => {
				let end = call_run_end(items, index);
				out.write_str("<|assistant|>\n")?;
				write_call_run_from_items(out, Dialect::Glm, &items[index..end], options)?;
				index = end;
			},
			ItemKind::ToolResult(_) => {
				let end = result_run_end(items, index);
				out.write_str("<|observation|>\n")?;
				write_canonical_tool_responses(out, &items[index..end])?;
				index = end;
			},
			ItemKind::Message(message) => {
				out.write_str(if message.role == Role::System {
					"<|system|>\n"
				} else {
					"<|user|>\n"
				})?;
				write_message_text(out, message, false)?;
				index += 1;
			},
			_ => {
				index += 1;
			},
		}
	}
	Ok(())
}

fn write_harmony_transcript<W: fmt::Write + ?Sized>(
	out: &mut W,
	thread: &Thread,
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	let items = &thread.items;
	let mut index = 0;
	while index < items.len() {
		match &items[index].kind {
			ItemKind::Message(message) if message.role == Role::Assistant => {
				let mut had_part = false;
				for part in &message.parts {
					match part {
						Part::Thinking(thinking) if !thinking.text.is_empty() => {
							render_thinking(out, Dialect::Harmony, thinking.text.as_str())?;
							had_part = true;
						},
						Part::Text(text) if !text.is_empty() => {
							out.write_str(H_START)?;
							out.write_str("assistant")?;
							out.write_str(H_CHANNEL)?;
							out.write_str("final")?;
							out.write_str(H_MESSAGE)?;
							write_escaped_harmony_text(out, text.as_str())?;
							out.write_str(H_END)?;
							had_part = true;
						},
						Part::Blob(blob) => {
							out.write_str(H_START)?;
							out.write_str("assistant")?;
							out.write_str(H_CHANNEL)?;
							out.write_str("final")?;
							out.write_str(H_MESSAGE)?;
							write_blob_marker(out, blob.mime.as_str())?;
							out.write_str(H_END)?;
							had_part = true;
						},
						_ => {},
					}
				}
				let end = call_run_end(items, index + 1);
				if end > index + 1 {
					write_call_run_from_items(out, Dialect::Harmony, &items[index + 1..end], options)?;
					had_part = true;
				}
				if !had_part {
					out.write_str(H_START)?;
					out.write_str("assistant")?;
					out.write_str(H_CHANNEL)?;
					out.write_str("final")?;
					out.write_str(H_MESSAGE)?;
					out.write_str(H_END)?;
				}
				index = end;
			},
			ItemKind::ToolCall(_) => {
				let end = call_run_end(items, index);
				write_call_run_from_items(out, Dialect::Harmony, &items[index..end], options)?;
				index = end;
			},
			ItemKind::ToolResult(_) => {
				let end = result_run_end(items, index);
				write_canonical_result_run(out, Dialect::Harmony, &items[index..end])?;
				index = end;
			},
			ItemKind::Message(message) => {
				out.write_str(H_START)?;
				out.write_str(if message.role == Role::System {
					"system"
				} else {
					"user"
				})?;
				out.write_str(H_MESSAGE)?;
				write_message_text(out, message, true)?;
				out.write_str(H_END)?;
				index += 1;
			},
			_ => {
				index += 1;
			},
		}
	}
	Ok(())
}

fn write_kimi_transcript<W: fmt::Write + ?Sized>(
	out: &mut W,
	thread: &Thread,
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	let items = &thread.items;
	let mut index = 0;
	while index < items.len() {
		match &items[index].kind {
			ItemKind::Message(message) if message.role == Role::Assistant => {
				out.write_str("<|im_assistant|>assistant<|im_middle|>")?;
				write_assistant_parts(out, Dialect::Kimi, message)?;
				let end = call_run_end(items, index + 1);
				write_call_run_from_items(out, Dialect::Kimi, &items[index + 1..end], options)?;
				out.write_str("<|im_end|>")?;
				index = end;
			},
			ItemKind::ToolCall(_) => {
				let end = call_run_end(items, index);
				out.write_str("<|im_assistant|>assistant<|im_middle|>")?;
				write_call_run_from_items(out, Dialect::Kimi, &items[index..end], options)?;
				out.write_str("<|im_end|>")?;
				index = end;
			},
			ItemKind::ToolResult(_) => {
				let end = result_run_end(items, index);
				write_canonical_result_run(out, Dialect::Kimi, &items[index..end])?;
				index = end;
			},
			ItemKind::Message(message) => {
				let (role, name) = if message.role == Role::System {
					("system", "system")
				} else {
					("user", "user")
				};
				write!(out, "<|im_{role}|>{name}<|im_middle|>")?;
				write_message_text(out, message, false)?;
				out.write_str("<|im_end|>")?;
				index += 1;
			},
			_ => {
				index += 1;
			},
		}
	}
	Ok(())
}

fn write_gemini_transcript<W: fmt::Write + ?Sized>(
	out: &mut W,
	thread: &Thread,
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	if thread.items.is_empty() {
		return Ok(());
	}
	out.write_str("<bos>")?;
	let items = &thread.items;
	let mut index = 0;
	while index < items.len() {
		match &items[index].kind {
			ItemKind::Message(message) if message.role == Role::Assistant => {
				out.write_str("<start_of_turn>model\n")?;
				write_assistant_parts(out, Dialect::Gemini, message)?;
				let end = call_run_end(items, index + 1);
				write_call_run_from_items(out, Dialect::Gemini, &items[index + 1..end], options)?;
				out.write_str("<end_of_turn>\n")?;
				index = end;
			},
			ItemKind::ToolCall(_) => {
				let end = call_run_end(items, index);
				out.write_str("<start_of_turn>model\n")?;
				write_call_run_from_items(out, Dialect::Gemini, &items[index..end], options)?;
				out.write_str("<end_of_turn>\n")?;
				index = end;
			},
			ItemKind::ToolResult(_) => {
				let end = result_run_end(items, index);
				out.write_str("<start_of_turn>user\n")?;
				write_canonical_result_run(out, Dialect::Gemini, &items[index..end])?;
				out.write_str("<end_of_turn>\n")?;
				index = end;
			},
			ItemKind::Message(message) => {
				out.write_str("<start_of_turn>user\n")?;
				write_message_text(out, message, false)?;
				out.write_str("<end_of_turn>\n")?;
				index += 1;
			},
			_ => {
				index += 1;
			},
		}
	}
	Ok(())
}

fn write_gemma_transcript<W: fmt::Write + ?Sized>(
	out: &mut W,
	thread: &Thread,
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	if thread.items.is_empty() {
		return Ok(());
	}
	out.write_str("<bos>")?;
	let items = &thread.items;
	let mut index = 0;
	while index < items.len() {
		match &items[index].kind {
			ItemKind::Message(message) if message.role == Role::Assistant => {
				out.write_str("<|turn>model\n")?;
				write_assistant_parts(out, Dialect::Gemma, message)?;
				let calls_end = call_run_end(items, index + 1);
				write_call_run_from_items(out, Dialect::Gemma, &items[index + 1..calls_end], options)?;
				let results_end = result_run_end(items, calls_end);
				write_canonical_result_run(out, Dialect::Gemma, &items[calls_end..results_end])?;
				out.write_str("<turn|>")?;
				index = results_end;
			},
			ItemKind::ToolCall(_) => {
				let calls_end = call_run_end(items, index);
				let results_end = result_run_end(items, calls_end);
				out.write_str("<|turn>model\n")?;
				write_call_run_from_items(out, Dialect::Gemma, &items[index..calls_end], options)?;
				write_canonical_result_run(out, Dialect::Gemma, &items[calls_end..results_end])?;
				out.write_str("<turn|>")?;
				index = results_end;
			},
			ItemKind::ToolResult(_) => {
				let end = result_run_end(items, index);
				out.write_str("<|turn>model\n")?;
				write_canonical_result_run(out, Dialect::Gemma, &items[index..end])?;
				out.write_str("<turn|>")?;
				index = end;
			},
			ItemKind::Message(message) => {
				out.write_str(if message.role == Role::System {
					"<|turn>system\n"
				} else {
					"<|turn>user\n"
				})?;
				write_message_text(out, message, false)?;
				out.write_str("<turn|>")?;
				index += 1;
			},
			_ => {
				index += 1;
			},
		}
	}
	Ok(())
}

fn write_simple_token_transcript<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	thread: &Thread,
	options: DialectRenderOptions<'_>,
	user: &str,
	assistant: &str,
	assistant_end: &str,
) -> DialectResult<()> {
	let items = &thread.items;
	let mut index = 0;
	while index < items.len() {
		match &items[index].kind {
			ItemKind::Message(message) if message.role == Role::Assistant => {
				out.write_str(assistant)?;
				write_assistant_parts(out, dialect, message)?;
				let end = call_run_end(items, index + 1);
				write_call_run_from_items(out, dialect, &items[index + 1..end], options)?;
				out.write_str(assistant_end)?;
				index = end;
			},
			ItemKind::ToolCall(_) => {
				let end = call_run_end(items, index);
				out.write_str(assistant)?;
				write_call_run_from_items(out, dialect, &items[index..end], options)?;
				out.write_str(assistant_end)?;
				index = end;
			},
			ItemKind::ToolResult(_) => {
				let end = result_run_end(items, index);
				write_canonical_result_run(out, dialect, &items[index..end])?;
				index = end;
			},
			ItemKind::Message(message) => {
				if message.role != Role::System {
					out.write_str(user)?;
				}
				write_message_text(out, message, false)?;
				index += 1;
			},
			_ => {
				index += 1;
			},
		}
	}
	Ok(())
}

pub(crate) fn write_call_run_from_items<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	items: &[Item],
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	if items.is_empty() {
		return Ok(());
	}
	match dialect {
		Dialect::Anthropic => out.write_str("<function_calls>\n")?,
		Dialect::DeepSeek => out.write_str(DS_CALLS_BEGIN)?,
		Dialect::Gemini => out.write_str("```tool_code\n")?,
		Dialect::Kimi => out.write_str("<|tool_calls_section_begin|>")?,
		Dialect::MiniMax => out.write_str("<minimax:tool_call>\n")?,
		_ => {},
	}
	if dialect == Dialect::Gemini && items.len() > 1 {
		out.write_str("[")?;
	}
	for (index, item) in items.iter().enumerate() {
		let ItemKind::ToolCall(call) = &item.kind else {
			continue;
		};
		if index != 0
			&& matches!(
				dialect,
				Dialect::Anthropic
					| Dialect::Glm
					| Dialect::Hermes
					| Dialect::MiniMax
					| Dialect::Qwen3
					| Dialect::Xml
			) {
			out.write_str("\n")?;
		}
		if dialect == Dialect::Gemini && index != 0 {
			out.write_str(", ")?;
		}
		if dialect == Dialect::Kimi {
			write_kimi_call(out, call, index)?;
		} else {
			render_tool_call(out, dialect, call, options)?;
		}
	}
	if dialect == Dialect::Gemini && items.len() > 1 {
		out.write_str("]")?;
	}
	match dialect {
		Dialect::Anthropic => out.write_str("\n</function_calls>")?,
		Dialect::DeepSeek => out.write_str(DS_CALLS_END)?,
		Dialect::Gemini => out.write_str("\n```")?,
		Dialect::Kimi => out.write_str("<|tool_calls_section_end|>")?,
		Dialect::MiniMax => out.write_str("\n</minimax:tool_call>")?,
		_ => {},
	}
	Ok(())
}

pub(crate) fn write_canonical_result_run<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	items: &[Item],
) -> DialectResult<()> {
	write_canonical_result_run_mode(out, dialect, items, true)
}

/// Writes result text for history projection while leaving blob parts for the
/// projected multimodal message.
pub(crate) fn write_canonical_result_run_text_only<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	items: &[Item],
) -> DialectResult<()> {
	write_canonical_result_run_mode(out, dialect, items, false)
}

fn write_canonical_result_run_mode<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	items: &[Item],
	include_images: bool,
) -> DialectResult<()> {
	if items.is_empty() {
		return Ok(());
	}
	if matches!(dialect, Dialect::Anthropic | Dialect::MiniMax) {
		out.write_str("<function_results>\n")?;
	}
	if dialect == Dialect::Glm {
		out.write_str("<observation>\n")?;
	}
	for (index, item) in items.iter().enumerate() {
		let ItemKind::ToolResult(result) = &item.kind else {
			continue;
		};
		if index != 0 && !matches!(dialect, Dialect::Gemma | Dialect::Harmony | Dialect::Kimi) {
			out.write_str("\n")?;
		}
		write_canonical_result(out, dialect, result, index, include_images)?;
	}
	if matches!(dialect, Dialect::Anthropic | Dialect::MiniMax) {
		out.write_str("\n</function_results>")?;
	}
	if dialect == Dialect::Glm {
		out.write_str("\n</observation>")?;
	}
	Ok(())
}

fn write_canonical_result<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	result: &ToolResult,
	index: usize,
	include_images: bool,
) -> fmt::Result {
	match dialect {
		Dialect::Anthropic | Dialect::MiniMax => {
			let (tag, stream) = if result.is_error {
				("error", "stderr")
			} else {
				("result", "stdout")
			};
			write!(out, "<{tag}>\n<tool_name>")?;
			write_xml_text(out, result.name.as_str())?;
			write!(out, "</tool_name>\n<{stream}>")?;
			write_result_parts(out, &result.parts, false, include_images)?;
			write!(out, "</{stream}>\n</{tag}>")
		},
		Dialect::DeepSeek => {
			out.write_str(DS_OUTPUT_BEGIN)?;
			write_result_parts(out, &result.parts, false, include_images)?;
			out.write_str(DS_OUTPUT_END)
		},
		Dialect::Gemini => {
			out.write_str("```tool_outputs\n")?;
			write_result_parts(out, &result.parts, false, include_images)?;
			out.write_str("\n```")
		},
		Dialect::Gemma => {
			out.write_str("<|tool_response>response:")?;
			out.write_str(result.name.as_str())?;
			out.write_str("{output:")?;
			out.write_str(GEMMA_STRING)?;
			write_result_parts(out, &result.parts, false, include_images)?;
			out.write_str(GEMMA_STRING)?;
			out.write_str("}<tool_response|>")
		},
		Dialect::Glm | Dialect::Hermes | Dialect::Qwen3 | Dialect::Xml => {
			out.write_str("<tool_response>\n")?;
			write_result_parts(out, &result.parts, false, include_images)?;
			out.write_str("\n</tool_response>")
		},
		Dialect::Harmony => {
			out.write_str(H_START)?;
			write_harmony_recipient(out, result.name.as_str())?;
			out.write_str(" to=assistant")?;
			out.write_str(H_CHANNEL)?;
			out.write_str("commentary")?;
			out.write_str(H_MESSAGE)?;
			write_result_parts(out, &result.parts, true, include_images)?;
			out.write_str(H_END)
		},
		Dialect::Kimi => {
			out.write_str("<|im_system|>")?;
			out.write_str(result.name.as_str())?;
			out.write_str("<|im_middle|>## Return of ")?;
			writeln!(out, "functions.{}:{index}", result.name)?;
			write_result_parts(out, &result.parts, false, include_images)?;
			out.write_str("<|im_end|>")
		},
	}
}

fn write_canonical_tool_responses<W: fmt::Write + ?Sized>(
	out: &mut W,
	items: &[Item],
) -> fmt::Result {
	for (index, item) in items.iter().enumerate() {
		let ItemKind::ToolResult(result) = &item.kind else {
			continue;
		};
		if index != 0 {
			out.write_str("\n")?;
		}
		out.write_str("<tool_response>\n")?;
		write_parts_text(out, &result.parts, false)?;
		out.write_str("\n</tool_response>")?;
	}
	Ok(())
}

pub(crate) fn write_assistant_parts<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	message: &Message,
) -> fmt::Result {
	let mut first_thinking = true;
	for part in &message.parts {
		if let Part::Thinking(thinking) = part
			&& !thinking.text.is_empty()
		{
			if !first_thinking {
				out.write_str("\n")?;
			}
			render_thinking(out, dialect, thinking.text.as_str())?;
			first_thinking = false;
		}
	}
	for part in &message.parts {
		match part {
			Part::Text(text) => {
				if dialect == Dialect::Harmony {
					write_escaped_harmony_text(out, text.as_str())?;
				} else {
					out.write_str(text.as_str())?;
				}
			},
			Part::Blob(blob) => write_blob_marker(out, blob.mime.as_str())?,
			Part::Thinking(_) => {},
			_ => {},
		}
	}
	Ok(())
}

fn write_message_text<W: fmt::Write + ?Sized>(
	out: &mut W,
	message: &Message,
	harmony_escape: bool,
) -> fmt::Result {
	write_parts_text(out, &message.parts, harmony_escape)
}

fn write_parts_text<W: fmt::Write + ?Sized>(
	out: &mut W,
	parts: &[Part],
	harmony_escape: bool,
) -> fmt::Result {
	for part in parts {
		match part {
			Part::Text(text) => {
				if harmony_escape {
					write_escaped_harmony_text(out, text.as_str())?;
				} else {
					out.write_str(text.as_str())?;
				}
			},
			Part::Thinking(thinking) => {
				if harmony_escape {
					write_escaped_harmony_text(out, thinking.text.as_str())?;
				} else {
					out.write_str(thinking.text.as_str())?;
				}
			},
			Part::Blob(blob) => write_blob_marker(out, blob.mime.as_str())?,
			_ => {},
		}
	}
	Ok(())
}

fn write_result_parts<W: fmt::Write + ?Sized>(
	out: &mut W,
	parts: &[Part],
	harmony_escape: bool,
	include_images: bool,
) -> fmt::Result {
	for part in parts {
		match part {
			Part::Text(text) => {
				if harmony_escape {
					write_escaped_harmony_text(out, text.as_str())?;
				} else {
					out.write_str(text.as_str())?;
				}
			},
			Part::Thinking(thinking) => {
				if harmony_escape {
					write_escaped_harmony_text(out, thinking.text.as_str())?;
				} else {
					out.write_str(thinking.text.as_str())?;
				}
			},
			Part::Blob(blob) if include_images => write_blob_marker(out, blob.mime.as_str())?,
			Part::Blob(_) => {},
			_ => {},
		}
	}
	Ok(())
}

fn write_blob_marker<W: fmt::Write + ?Sized>(out: &mut W, mime: &str) -> fmt::Result {
	if mime.is_empty() {
		out.write_str("[Image]")
	} else {
		write!(out, "[Image: {mime}]")
	}
}

const fn call_run_end(items: &[Item], mut index: usize) -> usize {
	while index < items.len() && matches!(&items[index].kind, ItemKind::ToolCall(_)) {
		index += 1;
	}
	index
}

const fn result_run_end(items: &[Item], mut index: usize) -> usize {
	while index < items.len() && matches!(&items[index].kind, ItemKind::ToolResult(_)) {
		index += 1;
	}
	index
}

fn write_chatml_open<W: fmt::Write + ?Sized>(out: &mut W, role: &str) -> fmt::Result {
	writeln!(out, "<|im_start|>{role}")
}

fn write_xml_invoke<W: fmt::Write + ?Sized>(
	out: &mut W,
	call: &ToolCall,
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	let args = parse_args(call)?;
	out.write_str("<invoke name=\"")?;
	write_xml_attr(out, call.name.as_str())?;
	out.write_str("\">")?;
	if let Some(args) = args.as_object() {
		for (key, value) in args {
			out.write_str("<parameter name=\"")?;
			write_xml_attr(out, key)?;
			out.write_str("\">")?;
			if value.is_string() && schema_arg_is_string(options, call.name.as_str(), key) {
				out.write_str(value.as_str().unwrap_or_default())?;
			} else {
				write_json_value(out, value)?;
			}
			out.write_str("</parameter>")?;
		}
	}
	out.write_str("</invoke>")?;
	Ok(())
}

fn write_glm_call<W: fmt::Write + ?Sized>(
	out: &mut W,
	call: &ToolCall,
	options: DialectRenderOptions<'_>,
) -> DialectResult<()> {
	let args = parse_args(call)?;
	out.write_str("<tool_call>")?;
	out.write_str(call.name.as_str())?;
	if let Some(args) = args.as_object() {
		for (key, value) in args {
			out.write_str("\n<arg_key>")?;
			out.write_str(key)?;
			out.write_str("</arg_key>\n<arg_value>")?;
			if value.is_string() && schema_arg_is_string(options, call.name.as_str(), key) {
				out.write_str(value.as_str().unwrap_or_default())?;
			} else {
				write_json_value(out, value)?;
			}
			out.write_str("</arg_value>")?;
		}
	}
	out.write_str("\n</tool_call>")?;
	Ok(())
}

fn write_gemini_call<W: fmt::Write + ?Sized>(out: &mut W, call: &ToolCall) -> DialectResult<()> {
	let args = parse_args(call)?;
	out.write_str("default_api.")?;
	out.write_str(call.name.as_str())?;
	out.write_str("(")?;
	if let Some(args) = args.as_object() {
		for (index, (key, value)) in args.iter().enumerate() {
			if index != 0 {
				out.write_str(", ")?;
			}
			write!(out, "{key}=")?;
			write_py_value(out, value)?;
		}
	}
	out.write_str(")")?;
	Ok(())
}

fn write_gemma_call<W: fmt::Write + ?Sized>(out: &mut W, call: &ToolCall) -> DialectResult<()> {
	let args = parse_args(call)?;
	out.write_str("<|tool_call>call:")?;
	out.write_str(call.name.as_str())?;
	out.write_str("{")?;
	if let Some(args) = args.as_object() {
		for (index, (key, value)) in args.iter().enumerate() {
			if index != 0 {
				out.write_str(",")?;
			}
			out.write_str(key)?;
			out.write_str(":")?;
			write_gemma_value(out, value)?;
		}
	}
	out.write_str("}<tool_call|>")?;
	Ok(())
}

fn write_kimi_call<W: fmt::Write + ?Sized>(
	out: &mut W,
	call: &ToolCall,
	index: usize,
) -> DialectResult<()> {
	out.write_str("<|tool_call_begin|>functions.")?;
	out.write_str(call.name.as_str())?;
	write!(out, ":{index}")?;
	out.write_str("<|tool_call_argument_begin|>")?;
	write_args_raw(out, call, false)?;
	out.write_str("<|tool_call_end|>")?;
	Ok(())
}

fn write_kimi_result_id<W: fmt::Write + ?Sized>(
	out: &mut W,
	name: &str,
	id: &str,
	index: usize,
) -> fmt::Result {
	let id = id.trim();
	if id.starts_with("functions.") {
		out.write_str(id)
	} else {
		write!(out, "functions.{name}:{index}")
	}
}

fn parse_args(call: &ToolCall) -> DialectResult<Value> {
	serde_json::from_slice(&call.args_json)
		.map_err(|source| DialectError::InvalidToolArguments { tool: call.name.clone(), source })
}

fn write_args_raw<W: fmt::Write + ?Sized>(
	out: &mut W,
	call: &ToolCall,
	harmony_escape: bool,
) -> DialectResult<()> {
	let text =
		std::str::from_utf8(&call.args_json).map_err(|error| DialectError::InvalidToolArguments {
			tool:   call.name.clone(),
			source: serde_json::Error::io(io::Error::new(io::ErrorKind::InvalidData, error)),
		})?;
	serde_json::from_str::<&serde_json::value::RawValue>(text)
		.map_err(|source| DialectError::InvalidToolArguments { tool: call.name.clone(), source })?;
	if harmony_escape {
		write_escaped_harmony_json(out, text)?;
	} else {
		out.write_str(text)?;
	}
	Ok(())
}

fn schema_arg_is_string(options: DialectRenderOptions<'_>, tool_name: &str, key: &str) -> bool {
	options
		.tools
		.iter()
		.find(|tool| tool.name == tool_name)
		.and_then(|tool| tool.parameters.get("properties"))
		.and_then(|properties| properties.get(key))
		.and_then(|property| property.get("type"))
		.and_then(Value::as_str)
		== Some("string")
}

fn write_py_arg_value<W: fmt::Write + ?Sized>(out: &mut W, value: &Value) -> fmt::Result {
	if let Some(text) = value.as_str()
		&& text.contains('\n')
		&& !text.contains("\"\"\"")
		&& !text.starts_with('"')
		&& !text.ends_with('"')
		&& !text.ends_with('\\')
	{
		return write!(out, "\"\"\"{text}\"\"\"");
	}
	write_py_value(out, value)
}

fn write_py_value<W: fmt::Write + ?Sized>(out: &mut W, value: &Value) -> fmt::Result {
	match value {
		Value::Null => out.write_str("None"),
		Value::Bool(value) => out.write_str(if *value { "True" } else { "False" }),
		Value::Number(value) => write!(out, "{value}"),
		Value::String(value) => write_py_string(out, value),
		Value::Array(values) => {
			out.write_str("[")?;
			for (index, value) in values.iter().enumerate() {
				if index != 0 {
					out.write_str(", ")?;
				}
				write_py_value(out, value)?;
			}
			out.write_str("]")
		},
		Value::Object(values) => {
			out.write_str("{")?;
			for (index, (key, value)) in values.iter().enumerate() {
				if index != 0 {
					out.write_str(", ")?;
				}
				write_py_string(out, key)?;
				out.write_str(": ")?;
				write_py_value(out, value)?;
			}
			out.write_str("}")
		},
	}
}

fn write_py_string<W: fmt::Write + ?Sized>(out: &mut W, value: &str) -> fmt::Result {
	out.write_str("\"")?;
	for ch in value.chars() {
		match ch {
			'\\' => out.write_str("\\\\")?,
			'"' => out.write_str("\\\"")?,
			'\n' => out.write_str("\\n")?,
			'\r' => out.write_str("\\r")?,
			'\t' => out.write_str("\\t")?,
			ch => out.write_char(ch)?,
		}
	}
	out.write_str("\"")
}

fn write_gemma_value<W: fmt::Write + ?Sized>(out: &mut W, value: &Value) -> fmt::Result {
	match value {
		Value::Null => out.write_str("null"),
		Value::Bool(value) => out.write_str(if *value { "true" } else { "false" }),
		Value::Number(value) => write!(out, "{value}"),
		Value::String(value) => {
			out.write_str(GEMMA_STRING)?;
			out.write_str(value)?;
			out.write_str(GEMMA_STRING)
		},
		Value::Array(values) => {
			out.write_str("[")?;
			for (index, value) in values.iter().enumerate() {
				if index != 0 {
					out.write_str(",")?;
				}
				write_gemma_value(out, value)?;
			}
			out.write_str("]")
		},
		Value::Object(values) => {
			out.write_str("{")?;
			for (index, (key, value)) in values.iter().enumerate() {
				if index != 0 {
					out.write_str(",")?;
				}
				out.write_str(key)?;
				out.write_str(":")?;
				write_gemma_value(out, value)?;
			}
			out.write_str("}")
		},
	}
}

fn write_json_string<W: fmt::Write + ?Sized>(out: &mut W, value: &str) -> fmt::Result {
	serde_json::to_writer(FmtIo(out), value).map_err(|_| fmt::Error)
}

fn write_xml_attr<W: fmt::Write + ?Sized>(out: &mut W, text: &str) -> fmt::Result {
	for ch in text.chars() {
		match ch {
			'&' => out.write_str("&amp;")?,
			'"' => out.write_str("&quot;")?,
			'<' => out.write_str("&lt;")?,
			'>' => out.write_str("&gt;")?,
			ch => out.write_char(ch)?,
		}
	}
	Ok(())
}
fn write_xml_text<W: fmt::Write + ?Sized>(out: &mut W, text: &str) -> fmt::Result {
	for ch in text.chars() {
		match ch {
			'&' => out.write_str("&amp;")?,
			'<' => out.write_str("&lt;")?,
			'>' => out.write_str("&gt;")?,
			ch => out.write_char(ch)?,
		}
	}
	Ok(())
}

fn write_harmony_recipient<W: fmt::Write + ?Sized>(out: &mut W, name: &str) -> fmt::Result {
	if !name.starts_with("functions.") {
		out.write_str("functions.")?;
	}
	out.write_str(name)
}

fn write_harmony_escaped<W: fmt::Write + ?Sized>(
	out: &mut W,
	text: &str,
	json: bool,
) -> fmt::Result {
	const TOKENS: [&str; 7] = ["start", "end", "message", "channel", "constrain", "return", "call"];
	let mut cursor = 0;
	let mut search_from = 0;
	while let Some(relative) = text[search_from..].find("<|") {
		let start = search_from + relative;
		let Some(close_relative) = text[start + 2..].find("|>") else {
			break;
		};
		let close = start + 2 + close_relative;
		let name = &text[start + 2..close];
		search_from = close + 2;
		if TOKENS.contains(&name) {
			out.write_str(&text[cursor..start])?;
			if json {
				write!(out, "<\\\\|{name}\\\\|>")?;
			} else {
				write!(out, "<\\|{name}\\|>")?;
			}
			cursor = search_from;
		}
	}
	out.write_str(&text[cursor..])
}

fn write_delimited_thinking<W: fmt::Write + ?Sized>(
	out: &mut W,
	open: &str,
	close: &str,
	text: &str,
) -> fmt::Result {
	out.write_str(open)?;
	out.write_str("\n")?;
	write_unwrapped_thinking(out, open, close, text.trim())?;
	out.write_str("\n")?;
	out.write_str(close)
}

fn write_unwrapped_thinking<W: fmt::Write + ?Sized>(
	out: &mut W,
	open: &str,
	close: &str,
	text: &str,
) -> fmt::Result {
	let mut cursor = 0;
	let mut wrote = false;
	while text[cursor..].starts_with(open) {
		let inner_start = cursor + open.len();
		let Some(relative_close) = find_balanced_close(open, close, text, inner_start) else {
			return out.write_str(text);
		};
		if wrote {
			out.write_str("\n")?;
		}
		write_unwrapped_thinking(out, open, close, text[inner_start..relative_close].trim())?;
		wrote = true;
		cursor = relative_close + close.len();
		while cursor < text.len() && text.as_bytes()[cursor].is_ascii_whitespace() {
			cursor += 1;
		}
	}
	if wrote && cursor == text.len() {
		Ok(())
	} else {
		out.write_str(text)
	}
}

fn find_balanced_close(open: &str, close: &str, text: &str, mut cursor: usize) -> Option<usize> {
	let mut depth = 1;
	while cursor < text.len() {
		let next_close = text[cursor..].find(close).map(|at| cursor + at)?;
		let next_open = text[cursor..].find(open).map(|at| cursor + at);
		if next_open.is_some_and(|at| at < next_close) {
			depth += 1;
			cursor = next_open.unwrap() + open.len();
		} else {
			depth -= 1;
			if depth == 0 {
				return Some(next_close);
			}
			cursor = next_close + close.len();
		}
	}
	None
}

struct FmtIo<'a, W: fmt::Write + ?Sized>(&'a mut W);
impl<W: fmt::Write + ?Sized> io::Write for FmtIo<'_, W> {
	fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
		let text = std::str::from_utf8(bytes).map_err(io::Error::other)?;
		self
			.0
			.write_str(text)
			.map_err(|_| io::Error::other("format destination failed"))?;
		Ok(bytes.len())
	}

	fn flush(&mut self) -> io::Result<()> {
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use bytes::Bytes;
	use omp_llm_types::{BlobPart, CallId, Thinking};

	use super::*;

	fn item(kind: ItemKind) -> Item {
		Item::builder()
			.seq(0)
			.kind(kind)
			.props(Default::default())
			.build()
	}

	fn call(id: CallId, name: &str, args: &'static [u8]) -> ToolCall {
		ToolCall::builder()
			.id(id)
			.name(name.into())
			.args_json(Bytes::from_static(args))
			.thought_signature(Bytes::new())
			.build()
	}

	#[test]
	fn one_and_batched_calls_keep_each_dialects_envelope() {
		let first = call(CallId::new(), "read", br#"{"path":"a"}"#);
		let second = call(CallId::new(), "write", br#"{"path":"b","text":"x"}"#);
		for dialect in Dialect::ALL {
			let mut one = String::new();
			render_tool_call(&mut one, dialect, &first, DialectRenderOptions::default()).unwrap();
			assert!(!one.is_empty(), "{dialect}");
			let mut batch = String::new();
			render_assistant_tool_calls(
				&mut batch,
				dialect,
				&[first.clone(), second.clone()],
				DialectRenderOptions::default(),
			)
			.unwrap();
			assert!(batch.contains("read"), "{dialect}: {batch}");
			assert!(batch.contains("write"), "{dialect}: {batch}");
		}
		let mut gemini = String::new();
		render_assistant_tool_calls(
			&mut gemini,
			Dialect::Gemini,
			&[first, second],
			DialectRenderOptions::default(),
		)
		.unwrap();
		assert_eq!(
			gemini,
			"```tool_code\n[default_api.read(path=\"a\"), default_api.write(path=\"b\", \
			 text=\"x\")]\n```"
		);
	}

	#[test]
	fn transcript_groups_results_preserves_thinking_and_serializes_images() {
		let id = CallId::new();
		let thread = Thread::builder()
			.items(vec![
				item(ItemKind::Message(
					Message::builder()
						.role(Role::Assistant)
						.parts(vec![
							Part::Thinking(
								Thinking::builder()
									.text("inspect".into())
									.signature(Bytes::new())
									.redacted(false)
									.build(),
							),
							Part::Text("working".into()),
						])
						.build(),
				)),
				item(ItemKind::ToolCall(call(id, "read", br#"{"path":"a"}"#))),
				item(ItemKind::ToolResult(
					ToolResult::builder()
						.call_id(id)
						.name("read".into())
						.parts(vec![
							Part::Text("ok".into()),
							Part::Blob(
								BlobPart::builder()
									.hash([0; 32])
									.mime("image/png".into())
									.size(1)
									.inline(Bytes::from_static(b"x"))
									.build(),
							),
						])
						.is_error(false)
						.build(),
				)),
			])
			.build();
		for dialect in Dialect::ALL {
			let mut transcript = String::new();
			render_transcript(&mut transcript, dialect, &thread, DialectRenderOptions::default())
				.unwrap();
			assert!(transcript.contains("inspect"), "{dialect}: {transcript}");
			assert!(transcript.contains("read"), "{dialect}: {transcript}");
			assert!(transcript.contains("[Image: image/png]"), "{dialect}: {transcript}");
		}
	}

	#[test]
	fn harmony_escapes_data_but_not_renderer_control_tokens() {
		let call = call(CallId::new(), "echo", b"{\"text\":\"<\x7ccall\x7c>\"}");
		let mut rendered = String::new();
		render_tool_call(&mut rendered, Dialect::Harmony, &call, DialectRenderOptions::default())
			.unwrap();
		assert!(rendered.starts_with("<\u{7c}start\u{7c}>assistant<\u{7c}channel\u{7c}>commentary"));
		assert!(rendered.contains("<\\\\|call\\\\|>"));
		assert!(rendered.ends_with("<\u{7c}call\u{7c}>"));
	}

	#[test]
	fn nested_thinking_is_not_double_wrapped() {
		let mut rendered = String::new();
		render_thinking(
			&mut rendered,
			Dialect::Anthropic,
			"<thinking><thinking>plan</thinking></thinking>",
		)
		.unwrap();
		assert_eq!(rendered, "<thinking>\nplan\n</thinking>");
	}
}
