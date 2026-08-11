//! Projection from dialect scanner events into canonical turn-stream deltas.

use std::collections::BTreeMap;

use bytes::{Bytes, BytesMut};
use omp_core::Str;
use omp_llm_types::{StreamPartKind, TurnEvent, ids::CallId};
use smallvec::SmallVec;

use crate::{
	Dialect,
	coercion::partial_suffix_overlap_any,
	factory::create_scanner,
	scanner::Scanner,
	types::{ScanBatch, ScanEvent, ScannerOptions},
};

/// One projector output, including the non-provider failure used when a model
/// starts fabricating tool results after an in-band call.
#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
	event: Option<TurnEvent>,
}

impl Projection {
	/// Creates a projection containing one canonical turn event.
	#[must_use]
	pub fn event(event: TurnEvent) -> Self {
		Self { event: Some(event) }
	}

	/// Creates the stop signal emitted for a fabricated in-band tool result.
	#[must_use]
	pub const fn abort_fabricated_tool_result() -> Self {
		Self { event: None }
	}

	/// Returns the canonical event, or `None` for a fabricated-result stop
	/// signal.
	#[must_use]
	pub fn into_event(self) -> Option<TurnEvent> {
		self.event
	}
}

/// Inline batch emitted by projector operations.
pub type ProjectionBatch = SmallVec<Projection, 8>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolChannel {
	Native,
	Inband,
}

#[derive(Clone, Copy, Debug)]
struct OpenPart {
	index: u32,
	kind:  StreamPartKind,
}

/// Stateful owned-mode stream projector.
///
/// The first named tool channel wins for the turn. This presence-based rule
/// de-conflicts provider-native calls from model-authored in-band calls without
/// guessing from argument emptiness. Nameless native ghost parts do not claim
/// the channel.
#[derive(Debug)]
pub struct StreamProjector {
	scanner:         Scanner,
	scanner_active:  bool,
	response_tokens: &'static [&'static [u8]],
	pending:         BytesMut,
	stopped:         bool,
	channel:         Option<ToolChannel>,
	next_index:      u32,
	open:            Option<OpenPart>,
	inband_tools:    BTreeMap<Str, u32>,
	native_tools:    BTreeMap<u32, u32>,
}

impl StreamProjector {
	/// Creates a projector and its concrete dialect scanner.
	#[must_use]
	pub fn new(dialect: Dialect, options: ScannerOptions<'_>) -> Self {
		Self {
			scanner:         create_scanner(dialect, options),
			scanner_active:  false,
			response_tokens: response_tokens(dialect),
			pending:         BytesMut::new(),
			stopped:         false,
			channel:         None,
			next_index:      0,
			open:            None,
			inband_tools:    BTreeMap::new(),
			native_tools:    BTreeMap::new(),
		}
	}

	/// Feeds model text and returns whether fabricated-result protection fired.
	///
	/// A partial response opener is retained across feeds. Once a full opener is
	/// observed, bytes before it are projected, open parts close, and generation
	/// becomes permanently stopped.
	pub fn feed_text(&mut self, chunk: Bytes) -> ProjectionBatch {
		let mut out = ProjectionBatch::new();
		if self.stopped {
			return out;
		}
		if self.pending.is_empty() {
			if let Some(index) = first_token_index(&chunk, self.response_tokens) {
				self.scanner_active = true;
				let scanned = self.scanner.feed(chunk.slice(..index));
				self.apply(scanned, &mut out);
				let flushed = self.scanner.flush();
				self.apply(flushed, &mut out);
				self.close_open(&mut out);
				self.stopped = true;
				out.push(Projection::abort_fabricated_tool_result());
				return out;
			}
			let valid = valid_utf8_prefix(&chunk);
			if !self.scanner_active
				&& !chunk.is_empty()
				&& valid == chunk.len()
				&& !chunk.iter().any(|byte| matches!(byte, b'<' | b'`'))
			{
				let index =
					self.open_part(StreamPartKind::Text, Str::default(), Str::default(), &mut out);
				out.push(Projection::event(TurnEvent::PartDelta { index, chunk }));
				return out;
			}
			if valid == chunk.len() && partial_suffix_overlap_any(&chunk, self.response_tokens) == 0 {
				self.scanner_active = true;
				let scanned = self.scanner.feed(chunk);
				self.apply(scanned, &mut out);
				return out;
			}
		}
		self.pending.extend_from_slice(&chunk);
		if let Some(index) = first_token_index(&self.pending, self.response_tokens) {
			let prefix = self.pending.split_to(index).freeze();
			self.scanner_active = true;
			let scanned = self.scanner.feed(prefix);
			self.apply(scanned, &mut out);
			self.pending.clear();
			let flushed = self.scanner.flush();
			self.apply(flushed, &mut out);
			self.close_open(&mut out);
			self.stopped = true;
			out.push(Projection::abort_fabricated_tool_result());
			return out;
		}
		let hold = partial_suffix_overlap_any(&self.pending, self.response_tokens);
		let valid = valid_utf8_prefix(&self.pending);
		let emit = valid.saturating_sub(hold.min(valid));
		if emit != 0 {
			let bytes = self.pending.split_to(emit).freeze();
			self.scanner_active = true;
			let scanned = self.scanner.feed(bytes);
			self.apply(scanned, &mut out);
		}
		out
	}

	/// Begins a provider-native tool call. Empty names are ignored and do not
	/// reserve the native channel. Non-canonical or zero provider IDs are
	/// replaced once with a nonzero canonical ID for the emitted lifecycle.
	pub fn native_tool_start(&mut self, source_index: u32, id: Str, name: Str) -> ProjectionBatch {
		let mut out = ProjectionBatch::new();
		if self.stopped || name.trim().is_empty() || self.channel == Some(ToolChannel::Inband) {
			return out;
		}
		self.channel = Some(ToolChannel::Native);
		self.close_open(&mut out);
		let index = self.allocate_index();
		self.native_tools.insert(source_index, index);
		out.push(Projection::event(TurnEvent::PartStart {
			index,
			kind: StreamPartKind::ToolCall,
			tool_call_id: canonical_call_id(id),
			tool_name: name,
		}));
		out
	}

	/// Forwards one provider-native JSON argument fragment when native owns the
	/// turn's tool channel.
	pub fn native_tool_delta(&mut self, source_index: u32, chunk: Bytes) -> ProjectionBatch {
		let mut out = ProjectionBatch::new();
		if self.stopped {
			return out;
		}
		if let Some(&index) = self.native_tools.get(&source_index) {
			out.push(Projection::event(TurnEvent::PartDelta { index, chunk }));
		}
		out
	}

	/// Ends a provider-native tool call. A never-started named call can be
	/// salvaged by invoking `native_tool_start` first with its final metadata.
	pub fn native_tool_end(&mut self, source_index: u32) -> ProjectionBatch {
		let mut out = ProjectionBatch::new();
		if let Some(index) = self.native_tools.remove(&source_index) {
			out.push(Projection::event(TurnEvent::PartEnd { index, signature: Default::default() }));
		}
		out
	}

	/// Flushes every retained UTF-8/delimiter suffix and closes the active part.
	pub fn finish(&mut self) -> ProjectionBatch {
		let mut out = ProjectionBatch::new();
		if self.stopped {
			return out;
		}
		if !self.pending.is_empty() {
			let bytes = self.pending.split().freeze();
			self.scanner_active = true;
			let scanned = self.scanner.feed(bytes);
			self.apply(scanned, &mut out);
		}
		let flushed = self.scanner.flush();
		self.apply(flushed, &mut out);
		self.close_open(&mut out);
		for index in std::mem::take(&mut self.native_tools).into_values() {
			out.push(Projection::event(TurnEvent::PartEnd { index, signature: Default::default() }));
		}
		out
	}

	/// Returns whether fabricated output permanently stopped projection.
	#[must_use]
	pub const fn is_stopped(&self) -> bool {
		self.stopped
	}

	fn apply(&mut self, events: ScanBatch, out: &mut ProjectionBatch) {
		for event in events {
			match event {
				ScanEvent::Text(chunk) => {
					let index =
						self.open_part(StreamPartKind::Text, Str::default(), Str::default(), out);
					out.push(Projection::event(TurnEvent::PartDelta { index, chunk }));
				},
				ScanEvent::ThinkingStart => {
					self.open_part(StreamPartKind::Thinking, Str::default(), Str::default(), out);
				},
				ScanEvent::ThinkingDelta(chunk) => {
					let index =
						self.open_part(StreamPartKind::Thinking, Str::default(), Str::default(), out);
					out.push(Projection::event(TurnEvent::PartDelta { index, chunk }));
				},
				ScanEvent::ThinkingEnd { signature } => {
					self.close_open_kind(StreamPartKind::Thinking, signature, out);
				},
				ScanEvent::ToolStart { id, name } => {
					if name.trim().is_empty() {
						continue;
					}
					if self.channel == Some(ToolChannel::Native) {
						continue;
					}
					self.channel = Some(ToolChannel::Inband);
					self.close_open(out);
					let index = self.allocate_index();
					let canonical_id = canonical_call_id(id.clone());
					self.inband_tools.insert(id, index);
					out.push(Projection::event(TurnEvent::PartStart {
						index,
						kind: StreamPartKind::ToolCall,
						tool_call_id: canonical_id,
						tool_name: name,
					}));
				},
				ScanEvent::ToolArgumentDelta { id, delta } => {
					if self.channel != Some(ToolChannel::Inband) {
						continue;
					}
					if let Some(&index) = self.inband_tools.get(&id) {
						out.push(Projection::event(TurnEvent::PartDelta { index, chunk: delta }));
					}
				},
				ScanEvent::ToolEnd { id, .. } => {
					if self.channel != Some(ToolChannel::Inband) {
						continue;
					}
					if let Some(index) = self.inband_tools.remove(&id) {
						out.push(Projection::event(TurnEvent::PartEnd {
							index,
							signature: Default::default(),
						}));
					}
				},
			}
		}
	}

	fn open_part(
		&mut self,
		kind: StreamPartKind,
		id: Str,
		name: Str,
		out: &mut ProjectionBatch,
	) -> u32 {
		if let Some(open) = self.open.filter(|open| open.kind == kind) {
			return open.index;
		}
		self.close_open(out);
		let index = self.allocate_index();
		self.open = Some(OpenPart { index, kind });
		out.push(Projection::event(TurnEvent::PartStart {
			index,
			kind,
			tool_call_id: id,
			tool_name: name,
		}));
		index
	}

	fn close_open_kind(
		&mut self,
		kind: StreamPartKind,
		signature: Bytes,
		out: &mut ProjectionBatch,
	) {
		if let Some(open) = self.open.filter(|open| open.kind == kind) {
			self.open = None;
			out.push(Projection::event(TurnEvent::PartEnd { index: open.index, signature }));
		}
	}

	fn close_open(&mut self, out: &mut ProjectionBatch) {
		if let Some(open) = self.open.take() {
			out.push(Projection::event(TurnEvent::PartEnd {
				index:     open.index,
				signature: Default::default(),
			}));
		}
	}

	const fn allocate_index(&mut self) -> u32 {
		let index = self.next_index;
		self.next_index = self.next_index.saturating_add(1);
		index
	}
}

fn canonical_call_id(id: Str) -> Str {
	let valid = id
		.parse::<CallId>()
		.is_ok_and(|id| id.as_ulid().to_bytes() != [0; 16]);
	if valid {
		id
	} else {
		Str::from(CallId::new().to_string())
	}
}

fn first_token_index(text: &[u8], tokens: &[&[u8]]) -> Option<usize> {
	tokens
		.iter()
		.filter_map(|token| {
			text
				.windows(token.len())
				.position(|window| window == *token)
		})
		.min()
}

const fn valid_utf8_prefix(bytes: &[u8]) -> usize {
	match std::str::from_utf8(bytes) {
		Ok(_) => bytes.len(),
		Err(error) => error.valid_up_to(),
	}
}

static DEEPSEEK_RESPONSE_TOKENS: &[&[u8]] =
	&["<｜tool▁outputs▁begin｜>".as_bytes(), "<｜tool▁output▁begin｜>".as_bytes()];

fn response_tokens(dialect: Dialect) -> &'static [&'static [u8]] {
	match dialect {
		Dialect::Glm | Dialect::Hermes | Dialect::Xml | Dialect::Qwen3 => &[b"<tool_response>"],
		Dialect::Kimi => &[b"<|im_system|>"],
		Dialect::Anthropic | Dialect::MiniMax => &[b"<function_results>", b"<tool_response>"],
		Dialect::DeepSeek => DEEPSEEK_RESPONSE_TOKENS,
		Dialect::Harmony => &[b"<\x7cstart\x7c>functions."],
		Dialect::Gemini => &[b"```tool_outputs"],
		Dialect::Gemma => &[b"<|tool_response>"],
	}
}
