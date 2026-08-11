//! Incremental recovery of leaked in-band reasoning channels.

use bytes::{Buf, Bytes, BytesMut};
use smallvec::SmallVec;

use crate::types::{ScanBatch, ScanEvent};

const TAGS: &[(&[u8], &[u8], bool)] = &[
	(b"<think>", b"</think>", false),
	(b"<thinking>", b"</thinking>", false),
	(b"<scratchpad>", b"</scratchpad>", false),
	(b"```thinking\n", b"```", true),
	(b"<|channel>thought\n", b"<channel|>", false),
	(b"<\x7cstart\x7c>assistant<\x7cchannel\x7c>analysis<\x7cmessage\x7c>", b"<\x7cend\x7c>", false),
	(b"<\x7cchannel\x7c>analysis<\x7cmessage\x7c>", b"<\x7cend\x7c>", false),
];

/// Incremental scanner which moves leaked reasoning markup out of visible text.
///
/// Delimiters are held only while they remain a possible prefix. Markdown code
/// spans and fences protect literal examples containing reasoning tags.
#[derive(Debug, Default)]
pub struct ThinkingScanner {
	buffer:      BytesMut,
	state:       ThinkingState,
	code_ticks:  usize,
	code_fenced: bool,
	line_indent: i16,
}

#[derive(Debug, Default)]
enum ThinkingState {
	#[default]
	Visible,
	Delimited(&'static [u8]),
	Fenced(FencedThinkingScanner),
}

impl ThinkingScanner {
	/// Creates a scanner at the beginning of visible text.
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	/// Feeds an arbitrary byte chunk. Incomplete UTF-8 code points are retained
	/// until a later feed makes them complete.
	pub fn feed(&mut self, chunk: Bytes) -> ScanBatch {
		self.buffer.extend_from_slice(&chunk);
		self.consume(false)
	}

	/// Resolves every held suffix and deterministically closes open reasoning.
	pub fn flush(&mut self) -> ScanBatch {
		self.consume(true)
	}

	fn consume(&mut self, final_chunk: bool) -> ScanBatch {
		let mut out = ScanBatch::new();
		loop {
			match &mut self.state {
				ThinkingState::Delimited(close) => {
					let close = *close;
					let valid = valid_prefix(&self.buffer, final_chunk);
					if let Some(at) = find(&self.buffer[..valid], close) {
						self.emit_thinking(at, &mut out);
						self.buffer.advance(close.len());
						out.push(ScanEvent::ThinkingEnd { signature: Bytes::new() });
						self.state = ThinkingState::Visible;
						continue;
					}
					let hold = if final_chunk {
						0
					} else {
						partial_suffix_overlap(&self.buffer[..valid], close)
					};
					self.emit_thinking(valid.saturating_sub(hold), &mut out);
					if final_chunk {
						if !self.buffer.is_empty() {
							let bytes = self.buffer.split().freeze();
							out.push(ScanEvent::ThinkingDelta(bytes));
						}
						out.push(ScanEvent::ThinkingEnd { signature: Bytes::new() });
						self.state = ThinkingState::Visible;
					}
					break;
				},
				ThinkingState::Fenced(scanner) => {
					let input = self.buffer.split().freeze();
					let result = scanner.feed(input, final_chunk);
					for delta in result.thinking {
						if !delta.is_empty() {
							out.push(ScanEvent::ThinkingDelta(delta));
						}
					}
					if result.closed || final_chunk {
						out.push(ScanEvent::ThinkingEnd { signature: Bytes::new() });
						self.buffer.extend_from_slice(&result.rest);
						self.state = ThinkingState::Visible;
						continue;
					}
					break;
				},
				ThinkingState::Visible => {},
			}

			if self.code_ticks != 0 {
				if !self.consume_code(final_chunk, &mut out) {
					break;
				}
				continue;
			}

			let valid = valid_prefix(&self.buffer, final_chunk);
			if valid == 0 {
				if final_chunk && !self.buffer.is_empty() {
					self.emit_text(self.buffer.len(), &mut out);
				}
				break;
			}
			match visible_hit(&self.buffer[..valid], final_chunk) {
				VisibleHit::None => {
					self.emit_text(valid, &mut out);
					break;
				},
				VisibleHit::Hold(at) => {
					self.emit_text(at, &mut out);
					break;
				},
				VisibleHit::Code { at, ticks } => {
					let fenced = ticks >= 3 && (0..=3).contains(&self.line_indent);
					self.emit_text(at + ticks, &mut out);
					self.code_ticks = ticks;
					self.code_fenced = fenced;
				},
				VisibleHit::Tag { at, open, close, fenced } => {
					self.emit_text(at, &mut out);
					self.buffer.advance(open.len());
					out.push(ScanEvent::ThinkingStart);
					self.state = if fenced {
						ThinkingState::Fenced(FencedThinkingScanner::new())
					} else {
						ThinkingState::Delimited(close)
					};
				},
			}
		}
		out
	}

	fn consume_code(&mut self, final_chunk: bool, out: &mut ScanBatch) -> bool {
		let valid = valid_prefix(&self.buffer, final_chunk);
		if self.code_fenced {
			if let Some(end) = fence_close_end(&self.buffer[..valid], self.code_ticks, final_chunk) {
				self.emit_text(end, out);
				self.code_ticks = 0;
				self.code_fenced = false;
				return true;
			}
			if final_chunk {
				self.emit_text(valid, out);
				self.code_ticks = 0;
				self.code_fenced = false;
				return false;
			}
			if let Some(last_nl) = self.buffer[..valid].iter().rposition(|byte| *byte == b'\n') {
				self.emit_text(last_nl + 1, out);
			}
			return false;
		}

		if let Some(at) = exact_backtick_run(&self.buffer[..valid], self.code_ticks)
			&& (final_chunk || at + self.code_ticks < valid) {
				self.emit_text(at + self.code_ticks, out);
				self.code_ticks = 0;
				return true;
			}
		let hold = if final_chunk {
			0
		} else {
			trailing_backticks(&self.buffer[..valid])
		};
		self.emit_text(valid.saturating_sub(hold), out);
		if final_chunk {
			self.code_ticks = 0;
		}
		false
	}

	fn emit_text(&mut self, length: usize, out: &mut ScanBatch) {
		if length == 0 {
			return;
		}
		let bytes = self.buffer.split_to(length).freeze();
		self.line_indent = trailing_indent(&bytes, self.line_indent);
		out.push(ScanEvent::Text(bytes));
	}

	fn emit_thinking(&mut self, length: usize, out: &mut ScanBatch) {
		if length != 0 {
			out.push(ScanEvent::ThinkingDelta(self.buffer.split_to(length).freeze()));
		}
	}
}

/// The outcome of feeding a nested-fence-aware thinking close matcher.
#[derive(Debug)]
pub struct FencedThinkingResult {
	/// Zero-copy deltas known to belong to the reasoning channel.
	pub thinking: SmallVec<Bytes, 4>,
	/// Whether the outer thinking fence was consumed.
	pub closed:   bool,
	/// Bytes following the outer closer.
	pub rest:     Bytes,
}

/// Line-aware close matcher for a ` ```thinking ` block.
///
/// Language-tagged Markdown fences inside reasoning are nested; a bare outer
/// fence closes reasoning. Tilde fences are nested with the same rules.
#[derive(Debug, Default)]
pub struct FencedThinkingScanner {
	buffer:  BytesMut,
	inner:   Option<(u8, usize)>,
	emitted: usize,
}

impl FencedThinkingScanner {
	/// Creates a matcher at the first byte after ` ```thinking\n `.
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	/// Feeds one chunk, optionally resolving the final unterminated tail.
	pub fn feed(&mut self, chunk: Bytes, final_chunk: bool) -> FencedThinkingResult {
		self.buffer.extend_from_slice(&chunk);
		let mut thinking = SmallVec::new();
		loop {
			let valid = valid_prefix(&self.buffer, final_chunk);
			let Some(nl) = self.buffer[..valid].iter().position(|byte| *byte == b'\n') else {
				break;
			};
			let line = &self.buffer[..nl];
			if self.inner.is_none()
				&& let Some(rest_at) = top_level_close(line, false) {
					let mut rest = self.buffer.split_off(rest_at);
					if rest.starts_with(b"```") {
						let ticks = rest.iter().take_while(|byte| **byte == b'`').count();
						rest.advance(ticks);
					}
					self.reset();
					return FencedThinkingResult { thinking, closed: true, rest: rest.freeze() };
				}
			if nl + 1 > self.emitted {
				thinking.push(Bytes::copy_from_slice(&self.buffer[self.emitted..=nl]));
			}
			Self::update_inner(&mut self.inner, line);
			self.buffer.advance(nl + 1);
			self.emitted = 0;
		}

		let valid = valid_prefix(&self.buffer, final_chunk);
		if self.inner.is_some() {
			if valid > self.emitted {
				thinking.push(Bytes::copy_from_slice(&self.buffer[self.emitted..valid]));
				self.emitted = valid;
			}
			return FencedThinkingResult { thinking, closed: false, rest: Bytes::new() };
		}

		let tail = &self.buffer[..valid];
		let close = if final_chunk {
			top_level_close(tail, true)
		} else {
			streaming_inline_close(tail)
		};
		if let Some(rest_at) = close {
			let line = self.buffer.split().freeze();
			let ticks = line[rest_at..]
				.iter()
				.take_while(|byte| **byte == b'`')
				.count();
			let rest = line.slice(rest_at + ticks..);
			self.reset();
			return FencedThinkingResult { thinking, closed: true, rest };
		}
		if !final_chunk && must_hold_fence(tail) {
			return FencedThinkingResult { thinking, closed: false, rest: Bytes::new() };
		}
		if valid > self.emitted {
			thinking.push(Bytes::copy_from_slice(&self.buffer[self.emitted..valid]));
		}
		if final_chunk {
			self.reset();
		} else {
			self.emitted = valid;
		}
		FencedThinkingResult { thinking, closed: false, rest: Bytes::new() }
	}

	fn update_inner(inner: &mut Option<(u8, usize)>, line: &[u8]) {
		let trimmed = trim_start_spaces(line, 3);
		let Some(&marker) = trimmed
			.first()
			.filter(|byte| **byte == b'`' || **byte == b'~')
		else {
			return;
		};
		let run = trimmed.iter().take_while(|byte| **byte == marker).count();
		if run < 3 {
			return;
		}
		let info = trim_ascii(&trimmed[run..]);
		match *inner {
			None => *inner = Some((marker, run)),
			Some((open, width)) if open == marker && run >= width && info.is_empty() => *inner = None,
			Some(_) => {},
		}
	}

	fn reset(&mut self) {
		self.buffer.clear();
		self.inner = None;
		self.emitted = 0;
	}
}

#[derive(Clone, Copy)]
enum VisibleHit {
	None,
	Hold(usize),
	Code { at: usize, ticks: usize },
	Tag { at: usize, open: &'static [u8], close: &'static [u8], fenced: bool },
}

fn visible_hit(buffer: &[u8], final_chunk: bool) -> VisibleHit {
	let mut index = 0;
	while index < buffer.len() {
		for &(open, close, fenced) in TAGS {
			if buffer[index..].starts_with(open) {
				return VisibleHit::Tag { at: index, open, close, fenced };
			}
			if !final_chunk && open.len() > buffer.len() - index && open.starts_with(&buffer[index..])
			{
				return VisibleHit::Hold(index);
			}
		}
		if buffer[index] == b'`' {
			let ticks = buffer[index..]
				.iter()
				.take_while(|byte| **byte == b'`')
				.count();
			if !final_chunk && index + ticks == buffer.len() {
				return VisibleHit::Hold(index);
			}
			return VisibleHit::Code { at: index, ticks };
		}
		index += 1;
	}
	VisibleHit::None
}

fn valid_prefix(buffer: &[u8], final_chunk: bool) -> usize {
	match std::str::from_utf8(buffer) {
		Ok(_) => buffer.len(),
		Err(error) if error.error_len().is_none() && !final_chunk => error.valid_up_to(),
		Err(error) => {
			let bad = error.valid_up_to();
			if bad == 0 { buffer.len().min(1) } else { bad }
		},
	}
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
	(!needle.is_empty())
		.then(|| {
			haystack
				.windows(needle.len())
				.position(|window| window == needle)
		})
		.flatten()
}

pub(crate) fn partial_suffix_overlap(text: &[u8], tag: &[u8]) -> usize {
	let max = text.len().min(tag.len().saturating_sub(1));
	(1..=max)
		.rev()
		.find(|&length| text.ends_with(&tag[..length]))
		.unwrap_or(0)
}

fn exact_backtick_run(buffer: &[u8], ticks: usize) -> Option<usize> {
	let mut index = 0;
	while index < buffer.len() {
		if buffer[index] != b'`' {
			index += 1;
			continue;
		}
		let run = buffer[index..]
			.iter()
			.take_while(|byte| **byte == b'`')
			.count();
		if run == ticks {
			return Some(index);
		}
		index += run;
	}
	None
}

fn trailing_backticks(buffer: &[u8]) -> usize {
	buffer
		.iter()
		.rev()
		.take_while(|byte| **byte == b'`')
		.count()
}

fn trailing_indent(text: &[u8], prior: i16) -> i16 {
	let start = text
		.iter()
		.rposition(|byte| *byte == b'\n')
		.map_or(0, |at| at + 1);
	let mut indent = if start == 0 { prior } else { 0 };
	for byte in &text[start..] {
		if indent < 0 {
			break;
		}
		if *byte == b' ' {
			indent += 1;
		} else {
			indent = -1;
		}
	}
	indent
}

fn fence_close_end(buffer: &[u8], ticks: usize, final_chunk: bool) -> Option<usize> {
	let mut start = 0;
	while start <= buffer.len() {
		let nl = buffer[start..]
			.iter()
			.position(|byte| *byte == b'\n')
			.map(|at| start + at);
		let end = nl.unwrap_or(buffer.len());
		let line = trim_ascii(&buffer[start..end]);
		if line.len() >= ticks
			&& line.iter().all(|byte| *byte == b'`')
			&& (nl.is_some() || final_chunk)
		{
			return Some(nl.map_or(end, |at| at + 1));
		}
		let Some(nl) = nl else { break };
		start = nl + 1;
	}
	None
}

fn top_level_close(line: &[u8], final_tail: bool) -> Option<usize> {
	let trimmed = trim_start_spaces(line, 3);
	let offset = line.len() - trimmed.len();
	let ticks = trimmed.iter().take_while(|byte| **byte == b'`').count();
	if ticks < 3 {
		return None;
	}
	let rest = &trimmed[ticks..];
	if final_tail || rest.is_empty() || trim_ascii(rest).is_empty() || !is_language_token(rest) {
		Some(offset)
	} else {
		None
	}
}

fn streaming_inline_close(line: &[u8]) -> Option<usize> {
	let trimmed = trim_start_spaces(line, 3);
	let offset = line.len() - trimmed.len();
	let ticks = trimmed.iter().take_while(|byte| **byte == b'`').count();
	if ticks < 3 {
		return None;
	}
	let rest = &trimmed[ticks..];
	(!rest.is_empty() && !trim_ascii(rest).is_empty() && !is_language_token(rest)).then_some(offset)
}

fn must_hold_fence(line: &[u8]) -> bool {
	let trimmed = trim_start_spaces(line, 3);
	if trimmed.is_empty() {
		return line.len() <= 3;
	}
	let ticks = trimmed.iter().take_while(|byte| **byte == b'`').count();
	let rest = &trimmed[ticks..];
	rest.is_empty() || trim_ascii(rest).is_empty() || (ticks >= 3 && is_language_token(rest))
}

fn trim_start_spaces(mut value: &[u8], max: usize) -> &[u8] {
	let count = value
		.iter()
		.take_while(|byte| **byte == b' ')
		.count()
		.min(max);
	value = &value[count..];
	value
}

fn trim_ascii(value: &[u8]) -> &[u8] {
	let start = value
		.iter()
		.position(|byte| !byte.is_ascii_whitespace())
		.unwrap_or(value.len());
	let end = value
		.iter()
		.rposition(|byte| !byte.is_ascii_whitespace())
		.map_or(start, |at| at + 1);
	&value[start..end]
}

fn is_language_token(value: &[u8]) -> bool {
	!value.is_empty()
		&& value
			.iter()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(*byte, b'_' | b'+' | b'#' | b'-'))
}
