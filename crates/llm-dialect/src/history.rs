//! Projection of canonical history into owned in-band tool dialects.

use std::fmt::Write as _;

use omp_core::SmolStrMut;
use omp_llm_types::{Item, ItemKind, Message, Part, Role, Thread};

use crate::{
	Dialect,
	rendering::{
		render_thinking, write_call_run_from_items, write_canonical_result_run_text_only,
		write_escaped_harmony_text,
	},
	types::{DialectRenderOptions, DialectResult},
};

/// Projects native canonical calls, results, and reasoning into model-facing
/// text history for an owned dialect.
///
/// Consecutive calls and results become one model turn. Tool-result blob parts
/// are retained after the rendered result text in their original run order;
/// cloning those parts is cheap because blob bytes are reference counted.
pub fn project_inband_history(
	thread: &Thread,
	dialect: Dialect,
	options: DialectRenderOptions<'_>,
) -> DialectResult<Thread> {
	let source = &thread.items;
	let mut projected = Vec::with_capacity(source.len());
	let mut index = 0;
	while index < source.len() {
		match &source[index].kind {
			ItemKind::Message(message) if message.role == Role::Assistant => {
				let calls_end = call_run_end(source, index + 1);
				let has_thinking = message
					.parts
					.iter()
					.any(|part| matches!(part, Part::Thinking(_)));
				if calls_end == index + 1 && !has_thinking {
					projected.push(source[index].clone());
					index += 1;
					continue;
				}
				let mut text = SmolStrMut::default();
				write_history_assistant_text(&mut text, message, dialect)?;
				if !text.is_empty() && calls_end > index + 1 {
					text.write_char('\n')?;
				}
				write_call_run_from_items(&mut text, dialect, &source[index + 1..calls_end], options)?;
				let mut parts = Vec::with_capacity(
					1 + message
						.parts
						.iter()
						.filter(|part| matches!(part, Part::Blob(_)))
						.count(),
				);
				if !text.is_empty() {
					parts.push(Part::Text(text.freeze()));
				}
				parts.extend(message.parts.iter().filter_map(|part| match part {
					Part::Blob(blob) => Some(Part::Blob(blob.clone())),
					_ => None,
				}));
				projected.push(projected_message(&source[index], Role::Assistant, parts));
				index = calls_end;
			},
			ItemKind::ToolCall(_) => {
				let end = call_run_end(source, index);
				let mut text = SmolStrMut::default();
				write_call_run_from_items(&mut text, dialect, &source[index..end], options)?;
				projected.push(projected_message(&source[index], Role::Assistant, vec![Part::Text(
					text.freeze(),
				)]));
				index = end;
			},
			ItemKind::ToolResult(_) => {
				let end = result_run_end(source, index);
				let mut text = SmolStrMut::default();
				write_canonical_result_run_text_only(&mut text, dialect, &source[index..end])?;
				let image_count = source[index..end]
					.iter()
					.map(|item| match &item.kind {
						ItemKind::ToolResult(result) => result
							.parts
							.iter()
							.filter(|part| matches!(part, Part::Blob(_)))
							.count(),
						_ => 0,
					})
					.sum::<usize>();
				let mut parts = Vec::with_capacity(1 + image_count);
				parts.push(Part::Text(text.freeze()));
				for item in &source[index..end] {
					if let ItemKind::ToolResult(result) = &item.kind {
						parts.extend(result.parts.iter().filter_map(|part| match part {
							Part::Blob(blob) => Some(Part::Blob(blob.clone())),
							_ => None,
						}));
					}
				}
				projected.push(projected_message(&source[index], Role::User, parts));
				index = end;
			},
			_ => {
				projected.push(source[index].clone());
				index += 1;
			},
		}
	}
	Ok(Thread::builder().items(projected).build())
}

fn write_history_assistant_text<W: std::fmt::Write + ?Sized>(
	out: &mut W,
	message: &Message,
	dialect: Dialect,
) -> std::fmt::Result {
	let mut wrote_thinking = false;
	for part in &message.parts {
		if let Part::Thinking(thinking) = part
			&& !thinking.text.is_empty()
		{
			if wrote_thinking {
				out.write_char('\n')?;
			}
			render_thinking(out, dialect, thinking.text.as_str())?;
			wrote_thinking = true;
		}
	}
	for part in &message.parts {
		if let Part::Text(text) = part {
			if dialect == Dialect::Harmony {
				write_escaped_harmony_text(out, text.as_str())?;
			} else {
				out.write_str(text.as_str())?;
			}
		}
	}
	Ok(())
}

fn projected_message(source: &Item, role: Role, parts: Vec<Part>) -> Item {
	Item::builder()
		.seq(source.seq)
		.kind(ItemKind::Message(Message::builder().role(role).parts(parts).build()))
		.props(source.props.clone())
		.build()
}

fn call_run_end(items: &[Item], mut index: usize) -> usize {
	while index < items.len() && matches!(&items[index].kind, ItemKind::ToolCall(_)) {
		index += 1;
	}
	index
}

fn result_run_end(items: &[Item], mut index: usize) -> usize {
	while index < items.len() && matches!(&items[index].kind, ItemKind::ToolResult(_)) {
		index += 1;
	}
	index
}

#[cfg(test)]
mod tests {
	use bytes::Bytes;
	use omp_llm_types::{BlobPart, CallId, Props, ToolResult};

	use super::*;

	fn result_item(call_id: CallId, name: &str, text: &str, byte: u8) -> Item {
		Item::builder()
			.seq(0)
			.kind(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(call_id)
					.name(name.into())
					.parts(vec![
						Part::Text(text.into()),
						Part::Blob(
							BlobPart::builder()
								.hash([byte; 32])
								.mime("image/png".into())
								.size(1)
								.inline(Bytes::from(vec![byte]))
								.build(),
						),
					])
					.is_error(false)
					.build(),
			))
			.props(Props::default())
			.build()
	}

	#[test]
	fn consecutive_results_become_one_user_message_without_losing_images() {
		let thread = Thread::builder()
			.items(vec![
				result_item(CallId::new(), "first", "a", 1),
				result_item(CallId::new(), "second", "b", 2),
			])
			.build();
		let projected =
			project_inband_history(&thread, Dialect::Gemini, DialectRenderOptions::default()).unwrap();
		assert_eq!(projected.items.len(), 1);
		let ItemKind::Message(message) = &projected.items[0].kind else {
			panic!("projected result run was not a message");
		};
		assert_eq!(message.role, Role::User);
		assert_eq!(message.parts.len(), 3);
		assert!(matches!(&message.parts[1], Part::Blob(_)));
		assert!(matches!(&message.parts[2], Part::Blob(_)));
		let Part::Text(text) = &message.parts[0] else {
			panic!("first projected part was not text");
		};
		assert_eq!(text.as_str(), "```tool_outputs\na\n```\n```tool_outputs\nb\n```");
	}
}
