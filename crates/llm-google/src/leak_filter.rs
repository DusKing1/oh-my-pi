//! Chunk-safe suppression of leaked Gemini Flash planning objects.

use std::str::Utf8Error;

use bytes::Bytes;
use omp_core::Str;
use serde_json::Value;
use smallvec::SmallVec;

/// Stateful filter for JSON-like planning text leaked into visible CCA output.
///
/// The filter holds a possible leading planning object until it can classify
/// the complete object. Input may split both marker text and UTF-8 scalar
/// values at arbitrary byte boundaries. Only complete UTF-8 is ever returned.
#[derive(Clone, Debug, Default)]
pub struct PlanningLeakFilter {
	utf8_tail:  Vec<u8>,
	probe:      String,
	tool_names: SmallVec<Str, 8>,
}

impl PlanningLeakFilter {
	/// Creates a filter whose leak detector recognizes the supplied active
	/// tools.
	#[must_use]
	pub fn new(tool_names: impl IntoIterator<Item = Str>) -> Self {
		Self {
			utf8_tail:  Vec::new(),
			probe:      String::new(),
			tool_names: tool_names.into_iter().collect(),
		}
	}

	/// Feeds an arbitrarily split byte chunk and returns visible UTF-8
	/// fragments.
	///
	/// An error means the input contains an invalid UTF-8 sequence, rather than
	/// an incomplete scalar split across chunks. The invalid bytes remain
	/// buffered so they are never emitted as corrupted text.
	pub fn feed(&mut self, chunk: &[u8]) -> Result<SmallVec<Bytes, 2>, Utf8Error> {
		self.utf8_tail.extend_from_slice(chunk);
		let mut visible = SmallVec::new();
		loop {
			match std::str::from_utf8(&self.utf8_tail) {
				Ok(text) => {
					let owned = text.to_owned();
					self.utf8_tail.clear();
					self.feed_text(&owned, false, &mut visible);
					return Ok(visible);
				},
				Err(error) if error.error_len().is_none() => {
					let valid = error.valid_up_to();
					if valid == 0 {
						return Ok(visible);
					}
					let tail = self.utf8_tail.split_off(valid);
					let prefix = String::from_utf8(std::mem::replace(&mut self.utf8_tail, tail))
						.expect("valid_up_to is valid UTF-8");
					self.feed_text(&prefix, false, &mut visible);
				},
				Err(error) => return Err(error),
			}
		}
	}

	/// Finishes the stream, classifying or suppressing any buffered prefix.
	pub fn finish(&mut self) -> Result<SmallVec<Bytes, 2>, Utf8Error> {
		let text = std::str::from_utf8(&self.utf8_tail)?.to_owned();
		self.utf8_tail.clear();
		let mut visible = SmallVec::new();
		self.feed_text(&text, true, &mut visible);
		if !self.probe.is_empty() {
			match consume_planning_buffer(&self.probe, &self.tool_names, true) {
				PlanningResult::Plain(text) | PlanningResult::Leak(text) => {
					push_visible(text, &mut visible)
				},
				PlanningResult::Incomplete => {},
			}
			self.probe.clear();
		}
		Ok(visible)
	}

	/// Discards a buffered planning prefix before a structured tool call.
	///
	/// CCA tool calls are authoritative; raw JSON planning text immediately
	/// before one must not be released alongside the structured invocation.
	pub fn discard_probe(&mut self) {
		self.probe.clear();
	}

	fn feed_text(&mut self, text: &str, final_: bool, visible: &mut SmallVec<Bytes, 2>) {
		if text.is_empty() && !final_ {
			return;
		}
		self.probe.push_str(text);
		match consume_planning_buffer(&self.probe, &self.tool_names, final_) {
			PlanningResult::Incomplete => {},
			PlanningResult::Plain(text) | PlanningResult::Leak(text) => {
				push_visible(text, visible);
				self.probe.clear();
			},
		}
	}
}

#[derive(Debug, Eq, PartialEq)]
enum PlanningResult<'a> {
	Incomplete,
	Plain(&'a str),
	Leak(&'a str),
}

fn push_visible(text: &str, output: &mut SmallVec<Bytes, 2>) {
	if !text.is_empty() {
		output.push(Bytes::copy_from_slice(text.as_bytes()));
	}
}

fn consume_planning_buffer<'a>(
	text: &'a str,
	tool_names: &[Str],
	final_: bool,
) -> PlanningResult<'a> {
	if !is_planning_leak_prefix(text) {
		return PlanningResult::Plain(text);
	}
	let Some((object, rest)) =
		split_leading_object(text).or_else(|| split_leading_object_ignoring_quotes(text))
	else {
		if final_ {
			return if contains_leak_signature(text, tool_names) {
				PlanningResult::Leak("")
			} else {
				PlanningResult::Plain(text)
			};
		}
		return PlanningResult::Incomplete;
	};

	let leak = serde_json::from_str::<Value>(object).map_or_else(
		|_| contains_leak_signature(object, tool_names),
		|value| is_planning_object(&value, tool_names),
	);
	if leak {
		PlanningResult::Leak(rest)
	} else {
		PlanningResult::Plain(text)
	}
}

fn is_planning_leak_prefix(text: &str) -> bool {
	let trimmed = text.trim_start();
	if trimmed.is_empty() {
		return true;
	}
	if !trimmed.starts_with('{') {
		return false;
	}
	let after_brace = trimmed[1..].trim_start();
	if after_brace.is_empty() {
		return trimmed.len() <= 100;
	}
	let Some(after_quote) = after_brace.strip_prefix('"') else {
		return false;
	};
	let Some(next_quote) = after_quote.find('"') else {
		return "thought".starts_with(after_quote) && trimmed.len() <= 100;
	};
	if &after_quote[..next_quote] != "thought" {
		return false;
	}
	let after_key = after_quote[next_quote + 1..].trim_start();
	after_key.is_empty() && trimmed.len() <= 100 || after_key.starts_with(':')
}

fn split_leading_object(text: &str) -> Option<(&str, &str)> {
	let offset = text.len() - text.trim_start().len();
	let trimmed = &text[offset..];
	if !trimmed.starts_with('{') {
		return None;
	}
	let mut depth = 0_u32;
	let mut in_string = false;
	let mut escaped = false;
	for (index, byte) in trimmed.bytes().enumerate() {
		if in_string {
			if escaped {
				escaped = false;
			} else if byte == b'\\' {
				escaped = true;
			} else if byte == b'"' {
				in_string = false;
			}
			continue;
		}
		match byte {
			b'"' => in_string = true,
			b'{' => depth += 1,
			b'}' => {
				depth = depth.checked_sub(1)?;
				if depth == 0 {
					let end = offset + index + 1;
					return Some((&text[offset..end], &text[end..]));
				}
			},
			_ => {},
		}
	}
	None
}

fn split_leading_object_ignoring_quotes(text: &str) -> Option<(&str, &str)> {
	let offset = text.len() - text.trim_start().len();
	let trimmed = &text[offset..];
	if !trimmed.starts_with('{') {
		return None;
	}
	let mut depth = 0_u32;
	for (index, byte) in trimmed.bytes().enumerate() {
		match byte {
			b'{' => depth += 1,
			b'}' => {
				depth = depth.checked_sub(1)?;
				if depth == 0 {
					let end = offset + index + 1;
					return Some((&text[offset..end], &text[end..]));
				}
			},
			_ => {},
		}
	}
	None
}

fn is_planning_object(value: &Value, tool_names: &[Str]) -> bool {
	let Some(object) = value.as_object() else {
		return false;
	};
	object.get("thought").is_some_and(Value::is_string)
		|| object
			.get("call")
			.and_then(Value::as_str)
			.is_some_and(|call| tool_names.iter().any(|name| name == call))
		|| object.contains_key("_i")
		|| object.contains_key("paths")
		|| object.contains_key("command")
		|| object.contains_key("path") && object.contains_key("content")
}

fn contains_leak_signature(text: &str, tool_names: &[Str]) -> bool {
	text.contains("\"thought\"")
		|| text.contains("\"_i\"")
		|| text.contains("\"paths\"")
		|| text.contains("\"command\"")
		|| text.contains("\"path\"") && text.contains("\"content\"")
		|| tool_names
			.iter()
			.any(|name| text.contains(&format!("\"{name}\"")))
}
/// A cleaned fragment recovered from visible CCA text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HealedFragment {
	/// Text that remains in the visible answer channel.
	Text(Bytes),
	/// Reasoning recovered from leaked thinking delimiters.
	Thinking(Bytes),
}

/// Chunk-safe healer for reasoning markup leaked into visible text.
#[derive(Clone, Debug, Default)]
pub struct ThinkingLeakFilter {
	buffer: String,
	close:  Option<&'static str>,
}

impl ThinkingLeakFilter {
	/// Feeds one valid UTF-8 text delta and removes split control delimiters.
	#[must_use]
	pub fn feed(&mut self, text: &str) -> SmallVec<HealedFragment, 2> {
		self.buffer.push_str(text);
		self.consume(false)
	}

	/// Flushes held partial markers without exposing complete control text.
	#[must_use]
	pub fn finish(&mut self) -> SmallVec<HealedFragment, 2> {
		self.consume(true)
	}

	fn consume(&mut self, final_: bool) -> SmallVec<HealedFragment, 2> {
		const TAGS: [(&str, &str); 4] = [
			("<think>", "</think>"),
			("<thinking>", "</thinking>"),
			("<scratchpad>", "</scratchpad>"),
			("```thinking\n", "```"),
		];
		let mut output = SmallVec::new();
		loop {
			if let Some(close) = self.close {
				if let Some(index) = self.buffer.find(close) {
					push_healed(&mut output, true, &self.buffer[..index]);
					self.buffer.drain(..index + close.len());
					self.close = None;
					continue;
				}
				let hold = if final_ {
					0
				} else {
					suffix_overlap(&self.buffer, &[close])
				};
				let emit = self.buffer.len() - hold;
				push_healed(&mut output, true, &self.buffer[..emit]);
				self.buffer.drain(..emit);
				if final_ {
					self.close = None;
				}
				break;
			}

			let hit = TAGS
				.iter()
				.filter_map(|(open, close)| self.buffer.find(open).map(|index| (index, *open, *close)))
				.min_by_key(|(index, ..)| *index);
			if let Some((index, open, close)) = hit {
				push_healed(&mut output, false, &self.buffer[..index]);
				self.buffer.drain(..index + open.len());
				self.close = Some(close);
				continue;
			}
			let opens = TAGS.map(|(open, _)| open);
			let hold = if final_ {
				0
			} else {
				suffix_overlap(&self.buffer, &opens)
			};
			let emit = self.buffer.len() - hold;
			push_healed(&mut output, false, &self.buffer[..emit]);
			self.buffer.drain(..emit);
			break;
		}
		output
	}
}

fn push_healed(output: &mut SmallVec<HealedFragment, 2>, thinking: bool, text: &str) {
	if text.is_empty() {
		return;
	}
	let bytes = Bytes::copy_from_slice(text.as_bytes());
	output.push(if thinking {
		HealedFragment::Thinking(bytes)
	} else {
		HealedFragment::Text(bytes)
	});
}

fn suffix_overlap(text: &str, markers: &[&str]) -> usize {
	let max = markers.iter().map(|marker| marker.len()).max().unwrap_or(0);
	for length in (1..=text.len().min(max)).rev() {
		if text.is_char_boundary(text.len() - length)
			&& markers
				.iter()
				.any(|marker| marker.len() > length && marker.starts_with(&text[text.len() - length..]))
		{
			return length;
		}
	}
	0
}

#[cfg(test)]
mod tests {
	use super::*;

	fn collect(chunks: &[&[u8]], final_: bool) -> String {
		let mut filter = PlanningLeakFilter::new([Str::from("read")]);
		let mut output = Vec::new();
		for chunk in chunks {
			for visible in filter.feed(chunk).unwrap() {
				output.extend_from_slice(&visible);
			}
		}
		if final_ {
			for visible in filter.finish().unwrap() {
				output.extend_from_slice(&visible);
			}
		}
		String::from_utf8(output).unwrap()
	}

	#[test]
	fn suppresses_every_split_of_marker_and_utf8_suffix() {
		let input = "{\"thought\":\"plan\",\"call\":\"read\",\"paths\":[\"x\"]}好的，正文";
		for first in 0..=input.len() {
			for second in first..=input.len() {
				assert_eq!(
					collect(
						&[
							&input.as_bytes()[..first],
							&input.as_bytes()[first..second],
							&input.as_bytes()[second..]
						],
						true
					),
					"好的，正文",
					"split at {first}, {second}"
				);
			}
		}
	}

	#[test]
	fn retains_legitimate_json_and_incomplete_plain_text() {
		assert_eq!(collect(&[b"{\"some\":", b"1}"], true), "{\"some\":1}");
		assert_eq!(collect(&[b"{just visible"], true), "{just visible");
	}

	#[test]
	fn suppresses_incomplete_leak_at_end() {
		assert_eq!(collect(&[b"{\"thought\":\"unfinished"], true), "");
	}

	#[test]
	fn heals_thinking_markers_at_every_boundary_without_touching_utf8() {
		let input = "visible <thinking>秘密推理</thinking> 正文";
		for split in 0..=input.len() {
			if !input.is_char_boundary(split) {
				continue;
			}
			let mut filter = ThinkingLeakFilter::default();
			let mut fragments = filter.feed(&input[..split]);
			fragments.extend(filter.feed(&input[split..]));
			fragments.extend(filter.finish());
			let mut visible = Vec::new();
			let mut thinking = Vec::new();
			for fragment in fragments {
				match fragment {
					HealedFragment::Text(bytes) => visible.extend_from_slice(&bytes),
					HealedFragment::Thinking(bytes) => thinking.extend_from_slice(&bytes),
				}
			}
			assert_eq!(String::from_utf8(visible).unwrap(), "visible  正文");
			assert_eq!(String::from_utf8(thinking).unwrap(), "秘密推理");
		}
	}
}
