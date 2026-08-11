//! Concrete, allocation-disciplined scanners for every owned prompt dialect.

use bytes::{Buf, Bytes, BytesMut};
use omp_core::Str;
use serde_json::{Map, Value};

use crate::{
	coercion::{
		ArgShapes, as_object, build_arg_shapes, coerce_named_value, decode_value, mint_tool_call_id,
		normalize_kimi_function_name, partial_suffix_overlap, partial_suffix_overlap_any,
	},
	thinking::FencedThinkingScanner,
	tool::{parse_gemma_call, parse_python_calls},
	types::{ScanBatch, ScanEvent, ScannerOptions, XmlTagset},
};

/// Concrete scanner dispatch. No dialect scanner or scanner future is boxed.
#[derive(Debug)]
#[non_exhaustive]
pub enum Scanner {
	/// GLM keyed XML calls.
	Glm(GlmScanner),
	/// Hermes JSON-in-tag calls.
	Hermes(JsonTagScanner),
	/// Kimi token-section calls.
	Kimi(KimiScanner),
	/// XML fallback calls.
	Xml(XmlScanner),
	/// Anthropic invoke/parameter calls.
	Anthropic(XmlScanner),
	/// DeepSeek token and DSML calls.
	DeepSeek(DeepSeekScanner),
	/// Harmony control-token messages.
	Harmony(HarmonyScanner),
	/// Qwen JSON-in-tag calls.
	Qwen3(JsonTagScanner),
	/// Gemini Python fenced calls.
	Gemini(GeminiScanner),
	/// Gemma token-delimited calls.
	Gemma(GemmaScanner),
	/// MiniMax wrapped XML calls.
	MiniMax(XmlScanner),
}

impl Scanner {
	/// Feeds arbitrary bytes, retaining incomplete UTF-8 and delimiter suffixes.
	pub fn feed(&mut self, chunk: Bytes) -> ScanBatch {
		match self {
			Self::Glm(scanner) => scanner.feed(chunk),
			Self::Hermes(scanner) | Self::Qwen3(scanner) => scanner.feed(chunk),
			Self::Kimi(scanner) => scanner.feed(chunk),
			Self::Xml(scanner) | Self::Anthropic(scanner) | Self::MiniMax(scanner) => {
				scanner.feed(chunk)
			},
			Self::DeepSeek(scanner) => scanner.feed(chunk),
			Self::Harmony(scanner) => scanner.feed(chunk),
			Self::Gemini(scanner) => scanner.feed(chunk),
			Self::Gemma(scanner) => scanner.feed(chunk),
		}
	}

	/// Flushes held suffixes and closes unterminated thinking channels.
	pub fn flush(&mut self) -> ScanBatch {
		match self {
			Self::Glm(scanner) => scanner.flush(),
			Self::Hermes(scanner) | Self::Qwen3(scanner) => scanner.flush(),
			Self::Kimi(scanner) => scanner.flush(),
			Self::Xml(scanner) | Self::Anthropic(scanner) | Self::MiniMax(scanner) => scanner.flush(),
			Self::DeepSeek(scanner) => scanner.flush(),
			Self::Harmony(scanner) => scanner.flush(),
			Self::Gemini(scanner) => scanner.flush(),
			Self::Gemma(scanner) => scanner.flush(),
		}
	}
}

#[derive(Debug, Default)]
struct Buffer(BytesMut);

impl Buffer {
	fn push(&mut self, chunk: &Bytes) {
		self.0.extend_from_slice(chunk);
	}

	fn bytes(&self) -> &[u8] {
		&self.0
	}

	fn valid_len(&self, final_chunk: bool) -> usize {
		match std::str::from_utf8(&self.0) {
			Ok(_) => self.0.len(),
			Err(error) if error.error_len().is_none() && !final_chunk => error.valid_up_to(),
			Err(error) => {
				if error.valid_up_to() == 0 {
					self.0.len().min(1)
				} else {
					error.valid_up_to()
				}
			},
		}
	}

	fn take(&mut self, len: usize) -> Bytes {
		self.0.split_to(len).freeze()
	}

	fn discard(&mut self, len: usize) {
		self.0.advance(len);
	}

	fn clear(&mut self) {
		self.0.clear();
	}

	fn is_empty(&self) -> bool {
		self.0.is_empty()
	}
}

#[derive(Debug, Default)]
struct RawBlock(Option<BytesMut>);

impl RawBlock {
	fn new(enabled: bool) -> Self {
		Self(enabled.then(BytesMut::new))
	}

	fn push(&mut self, bytes: &[u8]) {
		if let Some(raw) = &mut self.0 {
			raw.extend_from_slice(bytes);
		}
	}

	fn finish(self) -> Option<Bytes> {
		self.0.map(BytesMut::freeze)
	}
}

fn raw_envelope(enabled: bool, parts: &[&[u8]]) -> Option<Bytes> {
	enabled.then(|| {
		let length = parts.iter().map(|part| part.len()).sum();
		let mut raw = BytesMut::with_capacity(length);
		for part in parts {
			raw.extend_from_slice(part);
		}
		raw.freeze()
	})
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

fn earliest(haystack: &[u8], needles: &[&'static [u8]]) -> Option<(usize, &'static [u8])> {
	needles
		.iter()
		.filter_map(|needle| find(haystack, needle).map(|at| (at, *needle)))
		.min_by_key(|(at, _)| *at)
}

fn emit_visible_tail(buffer: &mut Buffer, tags: &[&[u8]], final_chunk: bool, out: &mut ScanBatch) {
	let valid = buffer.valid_len(final_chunk);
	let hold = if final_chunk {
		0
	} else {
		partial_suffix_overlap_any(&buffer.bytes()[..valid], tags)
	};
	let emit = valid.saturating_sub(hold);
	if emit != 0 {
		out.push(ScanEvent::Text(buffer.take(emit)));
	}
}

fn consume_reasoning(
	buffer: &mut Buffer,
	close: &[u8],
	final_chunk: bool,
	out: &mut ScanBatch,
) -> bool {
	let valid = buffer.valid_len(final_chunk);
	if let Some(at) = find(&buffer.bytes()[..valid], close) {
		if at != 0 {
			out.push(ScanEvent::ThinkingDelta(buffer.take(at)));
		}
		buffer.discard(close.len());
		out.push(ScanEvent::ThinkingEnd { signature: Bytes::new() });
		return true;
	}
	let hold = if final_chunk {
		0
	} else {
		partial_suffix_overlap(&buffer.bytes()[..valid], close)
	};
	let emit = valid.saturating_sub(hold);
	if emit != 0 {
		out.push(ScanEvent::ThinkingDelta(buffer.take(emit)));
	}
	if final_chunk {
		if !buffer.is_empty() {
			out.push(ScanEvent::ThinkingDelta(buffer.take(buffer.bytes().len())));
		}
		out.push(ScanEvent::ThinkingEnd { signature: Bytes::new() });
		return true;
	}
	false
}

fn json_bytes(value: Value) -> Bytes {
	Bytes::from(serde_json::to_vec(&value).unwrap_or_else(|_| b"{}".to_vec()))
}

fn parse_json_object(raw: &[u8]) -> Map<String, Value> {
	let Ok(mut value) = serde_json::from_slice::<Value>(trim_ascii(raw)) else {
		return Map::new();
	};
	if let Value::String(encoded) = &value {
		if let Ok(decoded) = serde_json::from_str(encoded) {
			value = decoded;
		}
	}
	as_object(value)
}

fn trim_ascii(mut value: &[u8]) -> &[u8] {
	while value.first().is_some_and(u8::is_ascii_whitespace) {
		value = &value[1..];
	}
	while value.last().is_some_and(u8::is_ascii_whitespace) {
		value = &value[..value.len() - 1];
	}
	value
}

fn decode_text(raw: &[u8]) -> Str {
	Str::from(String::from_utf8_lossy(raw).trim())
}

// ---- Anthropic / XML / MiniMax
// ------------------------------------------------

const MAX_PARAMETER_BYTES: usize = 1_000_000;

/// Anthropic-compatible invoke/parameter scanner shared by XML and MiniMax.
#[derive(Debug)]
pub struct XmlScanner {
	buffer:           Buffer,
	state:            XmlState,
	shapes:           ArgShapes,
	parse_thinking:   bool,
	include_raw_tool: bool,
	dsml:             Option<DeepSeekScanner>,
	minimax:          bool,
}

#[derive(Clone, Copy, Debug, Default)]
enum XmlReturn {
	#[default]
	Outside,
	Section,
}

#[derive(Debug, Default)]
enum XmlState {
	#[default]
	Outside,
	Section,
	Invoke {
		call:      XmlCall,
		return_to: XmlReturn,
	},
	Parameter {
		call:            XmlCall,
		return_to:       XmlReturn,
		name:            Str,
		value:           BytesMut,
		explicit_string: Option<bool>,
		truncated:       bool,
	},
	Thinking {
		local:     Str,
		return_to: XmlReturn,
	},
}

#[derive(Debug)]
struct XmlCall {
	id:   Str,
	name: Str,
	args: Map<String, Value>,
	raw:  RawBlock,
}

impl XmlScanner {
	/// Creates the XML primitive requested by `xml_tagset`.
	#[must_use]
	pub fn new(options: ScannerOptions<'_>, minimax: bool) -> Self {
		let dsml = (options.xml_tagset == XmlTagset::Dsml).then(|| DeepSeekScanner::new(options));
		Self {
			buffer: Buffer::default(),
			state: XmlState::Outside,
			shapes: build_arg_shapes(options.tools),
			parse_thinking: options.parse_thinking,
			include_raw_tool: options.include_raw_tool,
			dsml,
			minimax,
		}
	}

	/// Feeds one XML-family byte chunk.
	pub fn feed(&mut self, chunk: Bytes) -> ScanBatch {
		if let Some(scanner) = &mut self.dsml {
			return scanner.feed(chunk);
		}
		self.buffer.push(&chunk);
		self.consume(false)
	}

	/// Flushes XML-family state; incomplete calls are discarded.
	pub fn flush(&mut self) -> ScanBatch {
		if let Some(scanner) = &mut self.dsml {
			return scanner.flush();
		}
		self.consume(true)
	}

	fn consume(&mut self, final_chunk: bool) -> ScanBatch {
		let mut out = ScanBatch::new();
		loop {
			let state = std::mem::take(&mut self.state);
			match state {
				XmlState::Outside => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some(lt) = self.buffer.bytes()[..valid]
						.iter()
						.position(|byte| *byte == b'<')
					else {
						if valid != 0 {
							out.push(ScanEvent::Text(self.buffer.take(valid)));
						}
						self.state = XmlState::Outside;
						break;
					};
					if lt != 0 {
						out.push(ScanEvent::Text(self.buffer.take(lt)));
						self.state = XmlState::Outside;
						continue;
					}
					let Some(tag) = self.take_xml_tag(final_chunk) else {
						self.state = XmlState::Outside;
						break;
					};
					let Some(parsed) = parse_xml_tag(&tag) else {
						out.push(ScanEvent::Text(tag));
						self.state = XmlState::Outside;
						continue;
					};
					if !self.is_control_tag(&parsed) {
						out.push(ScanEvent::Text(tag));
						self.state = XmlState::Outside;
					} else if !parsed.closing && self.is_wrapper(&parsed) {
						self.state = XmlState::Section;
					} else if parsed.closing && self.is_wrapper(&parsed) {
						self.state = XmlState::Outside;
					} else if !parsed.closing && parsed.local == "invoke" {
						self.start_xml_call(parsed, &tag, XmlReturn::Outside, &mut out);
					} else if self.is_thinking_open(&parsed) {
						self.start_xml_thinking(parsed, XmlReturn::Outside, &mut out);
					} else {
						out.push(ScanEvent::Text(tag));
						self.state = XmlState::Outside;
					}
				},
				XmlState::Section => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some(lt) = self.buffer.bytes()[..valid]
						.iter()
						.position(|byte| *byte == b'<')
					else {
						if valid != 0 {
							self.buffer.discard(valid);
						}
						self.state = XmlState::Section;
						break;
					};
					if lt != 0 {
						self.buffer.discard(lt);
						self.state = XmlState::Section;
						continue;
					}
					let Some(tag) = self.take_xml_tag(final_chunk) else {
						self.state = XmlState::Section;
						break;
					};
					let Some(parsed) = parse_xml_tag(&tag) else {
						self.state = XmlState::Section;
						continue;
					};
					if parsed.closing && self.is_wrapper(&parsed) {
						self.state = XmlState::Outside;
					} else if !parsed.closing && parsed.local == "invoke" {
						self.start_xml_call(parsed, &tag, XmlReturn::Section, &mut out);
					} else if self.is_thinking_open(&parsed) {
						self.start_xml_thinking(parsed, XmlReturn::Section, &mut out);
					} else {
						self.state = XmlState::Section;
					}
				},
				XmlState::Thinking { local, return_to } => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some(lt) = self.buffer.bytes()[..valid]
						.iter()
						.position(|byte| *byte == b'<')
					else {
						if valid != 0 {
							out.push(ScanEvent::ThinkingDelta(self.buffer.take(valid)));
						}
						if final_chunk {
							out.push(ScanEvent::ThinkingEnd { signature: Bytes::new() });
							self.state = return_to.into_state();
						} else {
							self.state = XmlState::Thinking { local, return_to };
						}
						break;
					};
					if lt != 0 {
						out.push(ScanEvent::ThinkingDelta(self.buffer.take(lt)));
						self.state = XmlState::Thinking { local, return_to };
						continue;
					}
					let Some(tag) = self.take_xml_tag(final_chunk) else {
						if final_chunk {
							let remaining = self.buffer.bytes().len();
							if remaining != 0 {
								out.push(ScanEvent::ThinkingDelta(self.buffer.take(remaining)));
							}
							out.push(ScanEvent::ThinkingEnd { signature: Bytes::new() });
							self.state = return_to.into_state();
						} else {
							self.state = XmlState::Thinking { local, return_to };
						}
						break;
					};
					if parse_xml_tag(&tag).is_some_and(|parsed| parsed.closing && parsed.local == local)
					{
						out.push(ScanEvent::ThinkingEnd { signature: Bytes::new() });
						self.state = return_to.into_state();
					} else {
						out.push(ScanEvent::ThinkingDelta(tag));
						self.state = XmlState::Thinking { local, return_to };
					}
				},
				XmlState::Invoke { mut call, return_to } => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some(lt) = self.buffer.bytes()[..valid]
						.iter()
						.position(|byte| *byte == b'<')
					else {
						if final_chunk {
							self.buffer.clear();
							self.state = return_to.into_state();
						} else {
							let bytes = self.buffer.take(valid);
							call.raw.push(&bytes);
							self.state = XmlState::Invoke { call, return_to };
						}
						break;
					};
					if lt != 0 {
						let bytes = self.buffer.take(lt);
						call.raw.push(&bytes);
						self.state = XmlState::Invoke { call, return_to };
						continue;
					}
					let Some(tag) = self.take_xml_tag(final_chunk) else {
						if final_chunk {
							self.buffer.clear();
							self.state = return_to.into_state();
						} else {
							self.state = XmlState::Invoke { call, return_to };
						}
						break;
					};
					let parsed = parse_xml_tag(&tag);
					call.raw.push(&tag);
					if parsed
						.as_ref()
						.is_some_and(|parsed| parsed.closing && parsed.local == "invoke")
					{
						if !call.name.is_empty() {
							out.push(ScanEvent::ToolEnd {
								id:        call.id,
								name:      call.name,
								args_json: json_bytes(Value::Object(call.args)),
								raw_block: call.raw.finish(),
							});
						}
						self.state = return_to.into_state();
					} else if let Some(parsed) =
						parsed.filter(|parsed| !parsed.closing && parsed.local == "parameter")
					{
						let name = parsed.attr("name").unwrap_or_default();
						let explicit_string = parsed.attr("string").map(|value| parse_xml_bool(&value));
						if parsed.self_closing {
							if !name.is_empty() {
								call.args.insert(
									name.to_string(),
									coerce_named_value(self.shapes.get(&call.name), &name, ""),
								);
							}
							self.state = XmlState::Invoke { call, return_to };
						} else {
							self.state = XmlState::Parameter {
								call,
								return_to,
								name,
								value: BytesMut::new(),
								explicit_string,
								truncated: false,
							};
						}
					} else {
						self.state = XmlState::Invoke { call, return_to };
					}
				},
				XmlState::Parameter {
					mut call,
					return_to,
					name,
					mut value,
					explicit_string,
					mut truncated,
				} => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some(lt) = self.buffer.bytes()[..valid]
						.iter()
						.position(|byte| *byte == b'<')
					else {
						if final_chunk {
							self.buffer.clear();
							self.state = return_to.into_state();
						} else {
							let delta = self.buffer.take(valid);
							append_capped(&mut value, &delta, &mut truncated);
							call.raw.push(&delta);
							if !delta.is_empty() {
								out.push(ScanEvent::ToolArgumentDelta { id: call.id.clone(), delta });
							}
							self.state = XmlState::Parameter {
								call,
								return_to,
								name,
								value,
								explicit_string,
								truncated,
							};
						}
						break;
					};
					if lt != 0 {
						let delta = self.buffer.take(lt);
						append_capped(&mut value, &delta, &mut truncated);
						call.raw.push(&delta);
						if !delta.is_empty() {
							out.push(ScanEvent::ToolArgumentDelta { id: call.id.clone(), delta });
						}
						self.state = XmlState::Parameter {
							call,
							return_to,
							name,
							value,
							explicit_string,
							truncated,
						};
						continue;
					}
					let Some(tag) = self.take_xml_tag(final_chunk) else {
						if final_chunk {
							self.buffer.clear();
							self.state = return_to.into_state();
						} else {
							self.state = XmlState::Parameter {
								call,
								return_to,
								name,
								value,
								explicit_string,
								truncated,
							};
						}
						break;
					};
					if parse_xml_tag(&tag)
						.is_some_and(|parsed| parsed.closing && parsed.local == "parameter")
					{
						call.raw.push(&tag);
						let mut raw = String::from_utf8_lossy(&value).into_owned();
						if truncated {
							raw.push_str("\n…[parameter truncated: exceeded 1000000 bytes]");
						}
						if !name.is_empty() {
							let value = if explicit_string == Some(true) {
								Value::String(raw)
							} else {
								coerce_named_value(self.shapes.get(&call.name), &name, &raw)
							};
							call.args.insert(name.to_string(), value);
						}
						self.state = XmlState::Invoke { call, return_to };
					} else {
						append_capped(&mut value, &tag, &mut truncated);
						call.raw.push(&tag);
						out.push(ScanEvent::ToolArgumentDelta { id: call.id.clone(), delta: tag });
						self.state = XmlState::Parameter {
							call,
							return_to,
							name,
							value,
							explicit_string,
							truncated,
						};
					}
				},
			}
		}
		out
	}

	fn take_xml_tag(&mut self, final_chunk: bool) -> Option<Bytes> {
		debug_assert_eq!(self.buffer.bytes().first(), Some(&b'<'));
		if let Some(end) = self.buffer.bytes().iter().position(|byte| *byte == b'>') {
			return Some(self.buffer.take(end + 1));
		}
		if final_chunk {
			return Some(self.buffer.take(1));
		}
		None
	}

	fn start_xml_call(
		&mut self,
		parsed: ParsedXmlTag,
		tag: &[u8],
		return_to: XmlReturn,
		out: &mut ScanBatch,
	) {
		let name = parsed.attr("name").unwrap_or_default();
		let id = mint_tool_call_id();
		if !name.is_empty() {
			out.push(ScanEvent::ToolStart { id: id.clone(), name: name.clone() });
		}
		let mut raw = RawBlock::new(self.include_raw_tool);
		raw.push(tag);
		if parsed.self_closing {
			if !name.is_empty() {
				out.push(ScanEvent::ToolEnd {
					id,
					name,
					args_json: json_bytes(Value::Object(Map::new())),
					raw_block: raw.finish(),
				});
			}
			self.state = return_to.into_state();
		} else {
			self.state =
				XmlState::Invoke { call: XmlCall { id, name, args: Map::new(), raw }, return_to };
		}
	}

	fn start_xml_thinking(
		&mut self,
		parsed: ParsedXmlTag,
		return_to: XmlReturn,
		out: &mut ScanBatch,
	) {
		out.push(ScanEvent::ThinkingStart);
		if parsed.self_closing {
			out.push(ScanEvent::ThinkingEnd { signature: Bytes::new() });
			self.state = return_to.into_state();
		} else {
			self.state = XmlState::Thinking { local: parsed.local, return_to };
		}
	}

	fn is_wrapper(&self, tag: &ParsedXmlTag) -> bool {
		matches!(tag.local.as_str(), "function_calls" | "tool_calls")
			|| (self.minimax && tag.local == "tool_call")
	}

	fn is_thinking_open(&self, tag: &ParsedXmlTag) -> bool {
		self.parse_thinking
			&& !tag.closing
			&& matches!(tag.local.as_str(), "thinking" | "think" | "scratchpad")
	}

	fn is_control_tag(&self, tag: &ParsedXmlTag) -> bool {
		tag.local == "invoke"
			|| self.is_wrapper(tag)
			|| (self.parse_thinking
				&& matches!(tag.local.as_str(), "thinking" | "think" | "scratchpad"))
	}
}

impl XmlReturn {
	fn into_state(self) -> XmlState {
		match self {
			Self::Outside => XmlState::Outside,
			Self::Section => XmlState::Section,
		}
	}
}

fn append_capped(target: &mut BytesMut, delta: &[u8], truncated: &mut bool) {
	let remaining = MAX_PARAMETER_BYTES.saturating_sub(target.len());
	target.extend_from_slice(&delta[..delta.len().min(remaining)]);
	*truncated |= delta.len() > remaining;
}

#[derive(Debug)]
struct ParsedXmlTag {
	local:        Str,
	closing:      bool,
	self_closing: bool,
	attrs:        Vec<(Str, Str)>,
}

impl ParsedXmlTag {
	fn attr(&self, key: &str) -> Option<Str> {
		self
			.attrs
			.iter()
			.find(|(name, _)| name == key)
			.map(|(_, value)| value.clone())
	}
}

fn parse_xml_tag(raw: &[u8]) -> Option<ParsedXmlTag> {
	let text = std::str::from_utf8(raw).ok()?;
	let body = text.trim().strip_prefix('<')?.strip_suffix('>')?.trim();
	let closing = body.starts_with('/');
	let self_closing = !closing && body.ends_with('/');
	let body = body.strip_prefix('/').unwrap_or(body);
	let body = body.strip_suffix('/').unwrap_or(body).trim();
	let head_end = body.find(char::is_whitespace).unwrap_or(body.len());
	let qualified = &body[..head_end];
	let local = qualified.rsplit(':').next()?;
	if !is_xml_name(local) {
		return None;
	}
	let mut attrs = Vec::new();
	let mut rest = &body[head_end..];
	while !rest.trim_start().is_empty() {
		rest = rest.trim_start();
		let name_end = rest
			.find(|ch: char| ch.is_whitespace() || ch == '=')
			.unwrap_or(rest.len());
		let raw_name = &rest[..name_end];
		if raw_name.is_empty() {
			return None;
		}
		rest = rest[name_end..].trim_start();
		let Some(after_equal) = rest.strip_prefix('=') else {
			return None;
		};
		rest = after_equal.trim_start();
		let (value, tail) = if let Some(quote) = rest
			.as_bytes()
			.first()
			.copied()
			.filter(|byte| matches!(*byte, b'\'' | b'"'))
		{
			let end = rest[1..].find(quote as char)? + 1;
			(&rest[1..end], &rest[end + 1..])
		} else {
			let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
			(&rest[..end], &rest[end..])
		};
		let name = raw_name
			.rsplit(':')
			.next()
			.unwrap_or(raw_name)
			.to_ascii_lowercase();
		attrs.push((Str::from(name), Str::from(value)));
		rest = tail;
	}
	Some(ParsedXmlTag {
		local: Str::from(local.to_ascii_lowercase()),
		closing,
		self_closing,
		attrs,
	})
}

fn is_xml_name(name: &str) -> bool {
	name
		.as_bytes()
		.first()
		.is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
		&& name
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

fn parse_xml_bool(value: &str) -> bool {
	!matches!(value.trim().to_ascii_lowercase().as_str(), "false" | "0" | "no")
}

// ---- Hermes / Qwen
// -----------------------------------------------------------

/// JSON-in-`<tool_call>` scanner used by Hermes and Qwen without conflating
/// their thinking defaults.
#[derive(Debug)]
pub struct JsonTagScanner {
	buffer:           Buffer,
	state:            JsonTagState,
	parse_thinking:   bool,
	include_raw_tool: bool,
}

#[derive(Debug, Default)]
enum JsonTagState {
	#[default]
	Outside,
	Thinking(&'static [u8]),
	Tool,
}

impl JsonTagScanner {
	/// Creates a JSON-tag scanner.
	#[must_use]
	pub fn new(options: ScannerOptions<'_>) -> Self {
		Self {
			buffer:           Buffer::default(),
			state:            JsonTagState::Outside,
			parse_thinking:   options.parse_thinking,
			include_raw_tool: options.include_raw_tool,
		}
	}

	/// Feeds one chunk.
	pub fn feed(&mut self, chunk: Bytes) -> ScanBatch {
		self.buffer.push(&chunk);
		self.consume(false)
	}

	/// Flushes the scanner, dropping an incomplete tool envelope.
	pub fn flush(&mut self) -> ScanBatch {
		self.consume(true)
	}

	fn consume(&mut self, final_chunk: bool) -> ScanBatch {
		const TOOL_OPEN: &[u8] = b"<tool_call>";
		const TOOL_CLOSE: &[u8] = b"</tool_call>";
		const THINK_OPEN: &[u8] = b"<think>";
		const THINK_CLOSE: &[u8] = b"</think>";
		const THINKING_OPEN: &[u8] = b"<thinking>";
		const THINKING_CLOSE: &[u8] = b"</thinking>";
		let mut out = ScanBatch::new();
		loop {
			match self.state {
				JsonTagState::Outside => {
					let valid = self.buffer.valid_len(final_chunk);
					let tags: &[&[u8]] = if self.parse_thinking {
						&[TOOL_OPEN, THINK_OPEN, THINKING_OPEN]
					} else {
						&[TOOL_OPEN]
					};
					let Some((at, tag)) = earliest(&self.buffer.bytes()[..valid], tags) else {
						emit_visible_tail(&mut self.buffer, tags, final_chunk, &mut out);
						break;
					};
					if at != 0 {
						out.push(ScanEvent::Text(self.buffer.take(at)));
					}
					self.buffer.discard(tag.len());
					if tag == THINK_OPEN || tag == THINKING_OPEN {
						out.push(ScanEvent::ThinkingStart);
						self.state = JsonTagState::Thinking(if tag == THINK_OPEN {
							THINK_CLOSE
						} else {
							THINKING_CLOSE
						});
					} else {
						self.state = JsonTagState::Tool;
					}
				},
				JsonTagState::Thinking(close) => {
					if consume_reasoning(&mut self.buffer, close, final_chunk, &mut out) {
						self.state = JsonTagState::Outside;
					} else {
						break;
					}
				},
				JsonTagState::Tool => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some(close) = find(&self.buffer.bytes()[..valid], TOOL_CLOSE) else {
						if final_chunk {
							self.buffer.clear();
							self.state = JsonTagState::Outside;
						}
						break;
					};
					let body = self.buffer.take(close);
					self.buffer.discard(TOOL_CLOSE.len());
					if let Some((name, args)) = parse_named_json_call(&body) {
						let id = mint_tool_call_id();
						out.push(ScanEvent::ToolStart { id: id.clone(), name: name.clone() });
						let args_json = json_bytes(Value::Object(args));
						out.push(ScanEvent::ToolArgumentDelta {
							id:    id.clone(),
							delta: args_json.clone(),
						});
						out.push(ScanEvent::ToolEnd {
							id,
							name,
							args_json,
							raw_block: raw_envelope(self.include_raw_tool, &[
								TOOL_OPEN, &body, TOOL_CLOSE,
							]),
						});
					}
					self.state = JsonTagState::Outside;
				},
			}
		}
		out
	}
}

fn parse_named_json_call(body: &[u8]) -> Option<(Str, Map<String, Value>)> {
	let value: Value = serde_json::from_slice(trim_ascii(body)).ok()?;
	let object = value.as_object()?;
	let name = Str::from(object.get("name")?.as_str()?.trim());
	if name.is_empty() {
		return None;
	}
	let arguments = match object.get("arguments") {
		None => Map::new(),
		Some(Value::Object(arguments)) => arguments.clone(),
		Some(Value::String(encoded)) => serde_json::from_str::<Value>(encoded)
			.ok()?
			.as_object()?
			.clone(),
		Some(_) => return None,
	};
	Some((name, arguments))
}

// ---- GLM ---------------------------------------------------------------------

/// GLM keyed argument scanner with schema-driven raw-string retention.
#[derive(Debug)]
pub struct GlmScanner {
	buffer:           Buffer,
	state:            GlmState,
	shapes:           ArgShapes,
	parse_thinking:   bool,
	include_raw_tool: bool,
}
#[derive(Debug, Default)]
enum GlmState {
	#[default]
	Outside,
	Thinking,
	Name,
	Body(GlmCall),
	Key(GlmCall),
	AfterKey(GlmCall),
	Value(GlmCall),
}
#[derive(Debug)]
struct GlmCall {
	id:    Str,
	name:  Str,
	args:  Map<String, Value>,
	key:   Str,
	value: BytesMut,
	raw:   RawBlock,
}
impl GlmScanner {
	/// Creates a GLM scanner and derives its coercion table once.
	#[must_use]
	pub fn new(options: ScannerOptions<'_>) -> Self {
		Self {
			buffer:           Buffer::default(),
			state:            GlmState::Outside,
			shapes:           build_arg_shapes(options.tools),
			parse_thinking:   options.parse_thinking,
			include_raw_tool: options.include_raw_tool,
		}
	}

	/// Feeds one GLM chunk. pub fn feed
	pub fn feed(&mut self, chunk: Bytes) -> ScanBatch {
		self.buffer.push(&chunk);
		self.consume(false)
	}

	/// Flushes reasoning and drops malformed incomplete calls.
	pub fn flush(&mut self) -> ScanBatch {
		self.consume(true)
	}

	fn consume(&mut self, final_chunk: bool) -> ScanBatch {
		const OPEN: &[u8] = b"<tool_call>";
		const CLOSE: &[u8] = b"</tool_call>";
		const KEY_OPEN: &[u8] = b"<arg_key>";
		const KEY_CLOSE: &[u8] = b"</arg_key>";
		const VALUE_OPEN: &[u8] = b"<arg_value>";
		const VALUE_CLOSE: &[u8] = b"</arg_value>";
		const THINK_OPEN: &[u8] = b"<think>";
		const THINK_CLOSE: &[u8] = b"</think>";
		const RESPONSE: &[u8] = b"<tool_response>";
		let mut out = ScanBatch::new();
		loop {
			let state = std::mem::take(&mut self.state);
			match state {
				GlmState::Outside => {
					let valid = self.buffer.valid_len(final_chunk);
					let tags: &[&[u8]] = if self.parse_thinking {
						&[OPEN, THINK_OPEN, RESPONSE]
					} else {
						&[OPEN, RESPONSE]
					};
					let static_tags = if self.parse_thinking {
						&[OPEN, THINK_OPEN, RESPONSE][..]
					} else {
						&[OPEN, RESPONSE][..]
					};
					let Some((at, tag)) = earliest(&self.buffer.bytes()[..valid], static_tags) else {
						emit_visible_tail(&mut self.buffer, tags, final_chunk, &mut out);
						self.state = GlmState::Outside;
						break;
					};
					if at != 0 {
						out.push(ScanEvent::Text(self.buffer.take(at)));
					}
					self.buffer.discard(tag.len());
					if tag == OPEN {
						self.state = GlmState::Name;
					} else if tag == THINK_OPEN {
						out.push(ScanEvent::ThinkingStart);
						self.state = GlmState::Thinking;
					} else {
						self.buffer.clear();
						self.state = GlmState::Outside;
						break;
					}
				},
				GlmState::Thinking => {
					if consume_reasoning(&mut self.buffer, THINK_CLOSE, final_chunk, &mut out) {
						self.state = GlmState::Outside
					} else {
						self.state = GlmState::Thinking;
						break;
					}
				},
				GlmState::Name => {
					let valid = self.buffer.valid_len(final_chunk);
					let newline = self.buffer.bytes()[..valid]
						.iter()
						.position(|b| *b == b'\n');
					let key = find(&self.buffer.bytes()[..valid], KEY_OPEN);
					let close = find(&self.buffer.bytes()[..valid], CLOSE);
					let Some(end) = [newline, key, close].into_iter().flatten().min() else {
						if final_chunk {
							self.buffer.clear();
							self.state = GlmState::Outside
						} else {
							self.state = GlmState::Name
						}
						break;
					};
					let raw_name = self.buffer.take(end);
					let name = decode_text(&raw_name);
					if name.is_empty() {
						self.state = GlmState::Outside;
						continue;
					}
					let id = mint_tool_call_id();
					out.push(ScanEvent::ToolStart { id: id.clone(), name: name.clone() });
					let mut raw = BytesMut::new();
					raw.extend_from_slice(OPEN);
					raw.extend_from_slice(&raw_name);
					let call = GlmCall {
						id,
						name,
						args: Map::new(),
						key: Str::default(),
						value: BytesMut::new(),
						raw: RawBlock(self.include_raw_tool.then_some(raw)),
					};
					if newline == Some(end) {
						let b = self.buffer.take(1);
						let mut call = call;
						call.raw.push(&b);
						self.state = GlmState::Body(call)
					} else if key == Some(end) {
						self.buffer.discard(KEY_OPEN.len());
						let mut call = call;
						call.raw.push(KEY_OPEN);
						self.state = GlmState::Key(call)
					} else {
						self.buffer.discard(CLOSE.len());
						let mut call = call;
						call.raw.push(CLOSE);
						end_glm(call, &mut out);
						self.state = GlmState::Outside
					}
				},
				GlmState::Body(mut call) => {
					while self
						.buffer
						.bytes()
						.first()
						.is_some_and(u8::is_ascii_whitespace)
					{
						let b = self.buffer.take(1);
						call.raw.push(&b);
					}
					if self.buffer.bytes().starts_with(KEY_OPEN) {
						self.buffer.discard(KEY_OPEN.len());
						call.raw.push(KEY_OPEN);
						self.state = GlmState::Key(call)
					} else if self.buffer.bytes().starts_with(CLOSE) {
						self.buffer.discard(CLOSE.len());
						call.raw.push(CLOSE);
						end_glm(call, &mut out);
						self.state = GlmState::Outside
					} else if !final_chunk
						&& partial_suffix_overlap_any(self.buffer.bytes(), &[KEY_OPEN, CLOSE])
							== self.buffer.bytes().len()
					{
						self.state = GlmState::Body(call);
						break;
					} else if self.buffer.is_empty() {
						self.state = GlmState::Body(call);
						break;
					} else {
						let b = self.buffer.take(1);
						call.raw.push(&b);
						self.state = GlmState::Body(call)
					}
				},
				GlmState::Key(mut call) => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some(end) = find(&self.buffer.bytes()[..valid], KEY_CLOSE) else {
						if final_chunk {
							self.buffer.clear();
							self.state = GlmState::Outside
						} else {
							self.state = GlmState::Key(call)
						}
						break;
					};
					let raw = self.buffer.take(end);
					call.key = decode_text(&raw);
					call.raw.push(&raw);
					self.buffer.discard(KEY_CLOSE.len());
					call.raw.push(KEY_CLOSE);
					self.state = GlmState::AfterKey(call)
				},
				GlmState::AfterKey(mut call) => {
					while self
						.buffer
						.bytes()
						.first()
						.is_some_and(u8::is_ascii_whitespace)
					{
						let b = self.buffer.take(1);
						call.raw.push(&b);
					}
					if self.buffer.bytes().starts_with(VALUE_OPEN) {
						self.buffer.discard(VALUE_OPEN.len());
						call.raw.push(VALUE_OPEN);
						call.value.clear();
						self.state = GlmState::Value(call)
					} else if !final_chunk && VALUE_OPEN.starts_with(self.buffer.bytes()) {
						self.state = GlmState::AfterKey(call);
						break;
					} else if self.buffer.is_empty() {
						self.state = GlmState::AfterKey(call);
						break;
					} else {
						let b = self.buffer.take(1);
						call.raw.push(&b);
						self.state = GlmState::AfterKey(call)
					}
				},
				GlmState::Value(mut call) => {
					let valid = self.buffer.valid_len(final_chunk);
					let close = find(&self.buffer.bytes()[..valid], VALUE_CLOSE);
					let heal = earliest(&self.buffer.bytes()[..valid], &[KEY_OPEN, CLOSE]);
					let boundary = close
						.map(|at| (at, VALUE_CLOSE, false))
						.or_else(|| heal.map(|(at, tag)| (at, tag, true)));
					if let Some((at, tag, healed)) = boundary {
						let delta = self.buffer.take(at);
						call.value.extend_from_slice(&delta);
						call.raw.push(&delta);
						if !delta.is_empty() {
							out.push(ScanEvent::ToolArgumentDelta { id: call.id.clone(), delta });
						}
						if healed {
							let value = String::from_utf8_lossy(&call.value).trim_end().to_owned();
							finish_glm_value(&self.shapes, &mut call, &value);
							self.state = GlmState::Body(call)
						} else {
							self.buffer.discard(tag.len());
							call.raw.push(tag);
							let value = String::from_utf8_lossy(&call.value).into_owned();
							finish_glm_value(&self.shapes, &mut call, &value);
							self.state = GlmState::Body(call)
						}
					} else if final_chunk {
						self.buffer.clear();
						self.state = GlmState::Outside;
						break;
					} else {
						let hold = partial_suffix_overlap(self.buffer.bytes(), VALUE_CLOSE);
						let emit = valid.saturating_sub(hold);
						let delta = self.buffer.take(emit);
						call.value.extend_from_slice(&delta);
						call.raw.push(&delta);
						if !delta.is_empty() {
							out.push(ScanEvent::ToolArgumentDelta { id: call.id.clone(), delta });
						}
						self.state = GlmState::Value(call);
						break;
					}
				},
			}
		}
		out
	}
}
fn finish_glm_value(shapes: &ArgShapes, call: &mut GlmCall, raw: &str) {
	if !call.key.is_empty() {
		call
			.args
			.insert(call.key.to_string(), coerce_named_value(shapes.get(&call.name), &call.key, raw));
	}
	call.key = Str::default();
	call.value.clear();
}
fn end_glm(call: GlmCall, out: &mut ScanBatch) {
	out.push(ScanEvent::ToolEnd {
		id:        call.id,
		name:      call.name,
		args_json: json_bytes(Value::Object(call.args)),
		raw_block: call.raw.finish(),
	});
}

// ---- DeepSeek ---------------------------------------------------------------

const DS_CALL_OPEN: &[u8] = "<｜tool▁call▁begin｜>".as_bytes();
const DS_CALL_CLOSE: &[u8] = "<｜tool▁call▁end｜>".as_bytes();
const DS_SEP: &[u8] = "<｜tool▁sep｜>".as_bytes();
const DS_SECTION_OPEN: &[u8] = "<｜tool▁calls▁begin｜>".as_bytes();
const DS_SECTION_CLOSE: &[u8] = "<｜tool▁calls▁end｜>".as_bytes();
const DSML_OPEN_FULL: &[u8] = "<｜DSML｜tool_calls>".as_bytes();
const DSML_CLOSE_FULL: &[u8] = "</｜DSML｜tool_calls>".as_bytes();
const DSML_OPEN_ASCII: &[u8] = b"<|DSML|tool_calls>";
const DSML_CLOSE_ASCII: &[u8] = b"</|DSML|tool_calls>";
const DS_THINK_OPEN: &[u8] = b"<think>";
const DS_THINK_CLOSE: &[u8] = b"</think>";
const DS_OUTSIDE: &[&[u8]] = &[
	DS_CALL_OPEN,
	DS_SECTION_OPEN,
	DS_SECTION_CLOSE,
	DSML_OPEN_FULL,
	DSML_OPEN_ASCII,
	DSML_CLOSE_FULL,
	DSML_CLOSE_ASCII,
	DS_THINK_OPEN,
	"<｜begin▁of▁sentence｜>".as_bytes(),
	"<｜end▁of▁sentence｜>".as_bytes(),
	"<｜Assistant｜>".as_bytes(),
	"<｜User｜>".as_bytes(),
];
const DS_OUTSIDE_NO_THINK: &[&[u8]] = &[
	DS_CALL_OPEN,
	DS_SECTION_OPEN,
	DS_SECTION_CLOSE,
	DSML_OPEN_FULL,
	DSML_OPEN_ASCII,
	DSML_CLOSE_FULL,
	DSML_CLOSE_ASCII,
	"<｜begin▁of▁sentence｜>".as_bytes(),
	"<｜end▁of▁sentence｜>".as_bytes(),
	"<｜Assistant｜>".as_bytes(),
	"<｜User｜>".as_bytes(),
];

/// DeepSeek scanner supporting native special tokens, legacy fenced calls, and
/// both DSML alphabets.
#[derive(Debug)]
pub struct DeepSeekScanner {
	buffer:           Buffer,
	state:            DeepSeekState,
	parse_thinking:   bool,
	include_raw_tool: bool,
}

#[derive(Debug, Default)]
enum DeepSeekState {
	#[default]
	Outside,
	Thinking,
	Call,
	Dsml {
		wide: bool,
	},
}

impl DeepSeekScanner {
	/// Creates a DeepSeek/DSML scanner.
	#[must_use]
	pub fn new(options: ScannerOptions<'_>) -> Self {
		Self {
			buffer:           Buffer::default(),
			state:            DeepSeekState::Outside,
			parse_thinking:   options.parse_thinking,
			include_raw_tool: options.include_raw_tool,
		}
	}

	/// Feeds one chunk.
	pub fn feed(&mut self, chunk: Bytes) -> ScanBatch {
		self.buffer.push(&chunk);
		self.consume(false)
	}

	/// Flushes reasoning and discards incomplete calls.
	pub fn flush(&mut self) -> ScanBatch {
		self.consume(true)
	}

	fn consume(&mut self, final_chunk: bool) -> ScanBatch {
		let mut out = ScanBatch::new();
		loop {
			match self.state {
				DeepSeekState::Outside => {
					let valid = self.buffer.valid_len(final_chunk);
					let tags = if self.parse_thinking {
						DS_OUTSIDE
					} else {
						DS_OUTSIDE_NO_THINK
					};
					let Some((at, tag)) = earliest(&self.buffer.bytes()[..valid], tags) else {
						emit_visible_tail(&mut self.buffer, tags, final_chunk, &mut out);
						break;
					};
					if at != 0 {
						out.push(ScanEvent::Text(self.buffer.take(at)));
					}
					self.buffer.discard(tag.len());
					if tag == DS_CALL_OPEN {
						self.state = DeepSeekState::Call;
					} else if tag == DSML_OPEN_FULL {
						self.state = DeepSeekState::Dsml { wide: true };
					} else if tag == DSML_OPEN_ASCII {
						self.state = DeepSeekState::Dsml { wide: false };
					} else if tag == DS_THINK_OPEN {
						out.push(ScanEvent::ThinkingStart);
						self.state = DeepSeekState::Thinking;
					} else {
						self.state = DeepSeekState::Outside;
					}
				},
				DeepSeekState::Thinking => {
					if consume_reasoning(&mut self.buffer, DS_THINK_CLOSE, final_chunk, &mut out) {
						self.state = DeepSeekState::Outside;
					} else {
						break;
					}
				},
				DeepSeekState::Call => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some(end) = find(&self.buffer.bytes()[..valid], DS_CALL_CLOSE) else {
						if final_chunk {
							self.buffer.clear();
							self.state = DeepSeekState::Outside;
						}
						break;
					};
					let body = self.buffer.take(end);
					self.buffer.discard(DS_CALL_CLOSE.len());
					if let Some((name, args)) = parse_deepseek_call(&body) {
						let id = mint_tool_call_id();
						out.push(ScanEvent::ToolStart { id: id.clone(), name: name.clone() });
						let args_json = json_bytes(Value::Object(args));
						out.push(ScanEvent::ToolArgumentDelta {
							id:    id.clone(),
							delta: args_json.clone(),
						});
						out.push(ScanEvent::ToolEnd {
							id,
							name,
							args_json,
							raw_block: raw_envelope(self.include_raw_tool, &[
								DS_CALL_OPEN,
								&body,
								DS_CALL_CLOSE,
							]),
						});
					}
					self.state = DeepSeekState::Outside;
				},
				DeepSeekState::Dsml { wide } => {
					let valid = self.buffer.valid_len(final_chunk);
					let close = if wide {
						DSML_CLOSE_FULL
					} else {
						DSML_CLOSE_ASCII
					};
					let Some(end) = find(&self.buffer.bytes()[..valid], close) else {
						if final_chunk {
							self.buffer.clear();
							self.state = DeepSeekState::Outside;
						}
						break;
					};
					let section = self.buffer.take(end);
					self.buffer.discard(close.len());
					parse_dsml_invokes(section, wide, self.include_raw_tool, &mut out);
					self.state = DeepSeekState::Outside;
				},
			}
		}
		out
	}
}

fn parse_deepseek_call(body: &[u8]) -> Option<(Str, Map<String, Value>)> {
	let separator = find(body, DS_SEP)?;
	let head = trim_ascii(&body[..separator]);
	let tail = trim_ascii(&body[separator + DS_SEP.len()..]);
	let (name, arguments) = if head == b"function" {
		let fence = find(tail, b"```json")?;
		let name = decode_text(&tail[..fence]);
		let arguments = tail[fence + 7..]
			.strip_prefix(b"\n")
			.unwrap_or(&tail[fence + 7..]);
		let arguments = arguments.strip_suffix(b"```").unwrap_or(arguments);
		(name, strict_json_object(arguments)?)
	} else {
		(decode_text(head), strict_json_object(tail)?)
	};
	(!name.is_empty()).then_some((name, arguments))
}

fn strict_json_object(raw: &[u8]) -> Option<Map<String, Value>> {
	let value: Value = serde_json::from_slice(trim_ascii(raw)).ok()?;
	value.as_object().cloned()
}

fn parse_dsml_invokes(section: Bytes, wide: bool, include_raw_tool: bool, out: &mut ScanBatch) {
	let invoke_open = if wide {
		"<｜DSML｜invoke".as_bytes()
	} else {
		b"<|DSML|invoke"
	};
	let invoke_close = if wide {
		"</｜DSML｜invoke>".as_bytes()
	} else {
		b"</|DSML|invoke>"
	};
	let mut cursor = 0;
	while let Some(relative) = find(&section[cursor..], invoke_open) {
		let start = cursor + relative;
		let Some(open_end) = section[start..]
			.iter()
			.position(|byte| *byte == b'>')
			.map(|offset| start + offset)
		else {
			break;
		};
		let Some(name) = pseudo_xml_attr(&section[start..=open_end], "name") else {
			cursor = open_end + 1;
			continue;
		};
		let Some(end) =
			find(&section[open_end + 1..], invoke_close).map(|offset| open_end + 1 + offset)
		else {
			break;
		};
		if !name.is_empty() {
			let id = mint_tool_call_id();
			out.push(ScanEvent::ToolStart { id: id.clone(), name: name.clone() });
			let mut args = Map::new();
			parse_dsml_parameters(&section, open_end + 1, end, wide, &id, &mut args, out);
			out.push(ScanEvent::ToolEnd {
				id,
				name,
				args_json: json_bytes(Value::Object(args)),
				raw_block: include_raw_tool.then(|| section.slice(start..end + invoke_close.len())),
			});
		}
		cursor = end + invoke_close.len();
	}
}

fn parse_dsml_parameters(
	section: &Bytes,
	start: usize,
	end: usize,
	wide: bool,
	id: &Str,
	args: &mut Map<String, Value>,
	out: &mut ScanBatch,
) {
	let parameter_open = if wide {
		"<｜DSML｜parameter".as_bytes()
	} else {
		b"<|DSML|parameter"
	};
	let parameter_close = if wide {
		"</｜DSML｜parameter>".as_bytes()
	} else {
		b"</|DSML|parameter>"
	};
	let mut cursor = start;
	while cursor < end {
		let Some(relative) = find(&section[cursor..end], parameter_open) else {
			break;
		};
		let open = cursor + relative;
		let Some(open_end) = section[open..end]
			.iter()
			.position(|byte| *byte == b'>')
			.map(|offset| open + offset)
		else {
			break;
		};
		let name = pseudo_xml_attr(&section[open..=open_end], "name").unwrap_or_default();
		let Some(close) =
			find(&section[open_end + 1..end], parameter_close).map(|offset| open_end + 1 + offset)
		else {
			break;
		};
		let value = section.slice(open_end + 1..close);
		if !value.is_empty() {
			out.push(ScanEvent::ToolArgumentDelta { id: id.clone(), delta: value.clone() });
		}
		if !name.is_empty() {
			let string = pseudo_xml_attr(&section[open..=open_end], "string")
				.as_deref()
				.is_none_or(parse_xml_bool);
			let parsed = if string {
				Value::String(String::from_utf8_lossy(&value).into_owned())
			} else {
				decode_value(&String::from_utf8_lossy(&value))
			};
			args.insert(name.to_string(), parsed);
		}
		cursor = close + parameter_close.len();
	}
}

fn pseudo_xml_attr(raw: &[u8], wanted: &str) -> Option<Str> {
	let text = std::str::from_utf8(raw).ok()?;
	let body = text.trim().strip_prefix('<')?.strip_suffix('>')?.trim();
	let head_end = body.find(char::is_whitespace)?;
	let mut rest = &body[head_end..];
	while !rest.trim_start().is_empty() {
		rest = rest.trim_start();
		let name_end = rest
			.find(|ch: char| ch.is_whitespace() || ch == '=')
			.unwrap_or(rest.len());
		let name = &rest[..name_end];
		if name.is_empty() {
			return None;
		}
		rest = rest[name_end..].trim_start();
		rest = rest.strip_prefix('=')?.trim_start();
		let quote = rest
			.as_bytes()
			.first()
			.copied()
			.filter(|byte| matches!(*byte, b'\'' | b'"'))?;
		let end = rest[1..].find(quote as char)? + 1;
		if name == wanted {
			return Some(Str::from(&rest[1..end]));
		}
		rest = &rest[end + 1..];
	}
	None
}

// ---- Kimi -------------------------------------------------------------------

const K_SECTION_OPEN: &[u8] = b"<|tool_calls_section_begin|>";
const K_SECTION_CLOSE: &[u8] = b"<|tool_calls_section_end|>";
const K_CALL_OPEN: &[u8] = b"<|tool_call_begin|>";
const K_CALL_CLOSE: &[u8] = b"<|tool_call_end|>";
const K_ARG_OPEN: &[u8] = b"<|tool_call_argument_begin|>";
const K_THINK_OPEN: &[u8] = b"<think>";
const K_THINK_CLOSE: &[u8] = b"</think>";

/// Kimi section scanner retaining model call identifiers.
#[derive(Debug)]
pub struct KimiScanner {
	buffer:           Buffer,
	state:            KimiState,
	parse_thinking:   bool,
	include_raw_tool: bool,
}

#[derive(Debug, Default)]
enum KimiState {
	#[default]
	Outside,
	Thinking,
	Section,
	Header,
	Arguments {
		header: Bytes,
	},
}

impl KimiScanner {
	/// Creates a Kimi scanner.
	#[must_use]
	pub fn new(options: ScannerOptions<'_>) -> Self {
		Self {
			buffer:           Buffer::default(),
			state:            KimiState::Outside,
			parse_thinking:   options.parse_thinking,
			include_raw_tool: options.include_raw_tool,
		}
	}

	/// Feeds one chunk.
	pub fn feed(&mut self, chunk: Bytes) -> ScanBatch {
		self.buffer.push(&chunk);
		self.consume(false)
	}

	/// Flushes incomplete state.
	pub fn flush(&mut self) -> ScanBatch {
		self.consume(true)
	}

	fn consume(&mut self, final_chunk: bool) -> ScanBatch {
		let mut out = ScanBatch::new();
		loop {
			let state = std::mem::take(&mut self.state);
			match state {
				KimiState::Outside => {
					let valid = self.buffer.valid_len(final_chunk);
					let tags: &[&[u8]] = if self.parse_thinking {
						&[K_SECTION_OPEN, K_THINK_OPEN]
					} else {
						&[K_SECTION_OPEN]
					};
					let Some((at, tag)) = earliest(&self.buffer.bytes()[..valid], tags) else {
						emit_visible_tail(&mut self.buffer, tags, final_chunk, &mut out);
						self.state = KimiState::Outside;
						break;
					};
					if at != 0 {
						out.push(ScanEvent::Text(self.buffer.take(at)));
					}
					self.buffer.discard(tag.len());
					if tag == K_THINK_OPEN {
						out.push(ScanEvent::ThinkingStart);
						self.state = KimiState::Thinking;
					} else {
						self.state = KimiState::Section;
					}
				},
				KimiState::Thinking => {
					if consume_reasoning(&mut self.buffer, K_THINK_CLOSE, final_chunk, &mut out) {
						self.state = KimiState::Outside;
					} else {
						self.state = KimiState::Thinking;
						break;
					}
				},
				KimiState::Section => {
					while self
						.buffer
						.bytes()
						.first()
						.is_some_and(u8::is_ascii_whitespace)
					{
						self.buffer.discard(1);
					}
					if self.buffer.bytes().starts_with(K_SECTION_CLOSE) {
						self.buffer.discard(K_SECTION_CLOSE.len());
						self.state = KimiState::Outside;
					} else if self.buffer.bytes().starts_with(K_CALL_OPEN) {
						self.buffer.discard(K_CALL_OPEN.len());
						self.state = KimiState::Header;
					} else if !final_chunk
						&& (K_SECTION_CLOSE.starts_with(self.buffer.bytes())
							|| K_CALL_OPEN.starts_with(self.buffer.bytes()))
					{
						self.state = KimiState::Section;
						break;
					} else if self.buffer.is_empty() {
						self.state = KimiState::Section;
						break;
					} else {
						self.buffer.discard(1);
						self.state = KimiState::Section;
					}
				},
				KimiState::Header => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some(argument) = find(&self.buffer.bytes()[..valid], K_ARG_OPEN) else {
						if final_chunk {
							self.buffer.clear();
							self.state = KimiState::Outside;
						} else {
							self.state = KimiState::Header;
						}
						break;
					};
					let header = self.buffer.take(argument);
					self.buffer.discard(K_ARG_OPEN.len());
					self.state = KimiState::Arguments { header };
				},
				KimiState::Arguments { header } => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some(end) = find(&self.buffer.bytes()[..valid], K_CALL_CLOSE) else {
						if final_chunk {
							self.buffer.clear();
							self.state = KimiState::Outside;
						} else {
							self.state = KimiState::Arguments { header };
						}
						break;
					};
					let body = self.buffer.take(end);
					self.buffer.discard(K_CALL_CLOSE.len());
					let id = decode_text(trim_ascii(&header));
					let name = Str::from(normalize_kimi_function_name(&id));
					if !id.is_empty() && !name.is_empty() {
						if let Some(arguments) = strict_json_object(&body) {
							out.push(ScanEvent::ToolStart { id: id.clone(), name: name.clone() });
							let args_json = json_bytes(Value::Object(arguments));
							out.push(ScanEvent::ToolArgumentDelta {
								id:    id.clone(),
								delta: args_json.clone(),
							});
							out.push(ScanEvent::ToolEnd {
								id,
								name,
								args_json,
								raw_block: raw_envelope(self.include_raw_tool, &[
									K_CALL_OPEN,
									&header,
									K_ARG_OPEN,
									&body,
									K_CALL_CLOSE,
								]),
							});
						}
					}
					self.state = KimiState::Section;
				},
			}
		}
		out
	}
}

// ---- Harmony ----------------------------------------------------------------

const H_START: &[u8] = b"<\x7cstart\x7c>";
const H_END: &[u8] = b"<\x7cend\x7c>";
const H_MESSAGE: &[u8] = b"<\x7cmessage\x7c>";
const H_CHANNEL: &[u8] = b"<\x7cchannel\x7c>";
const H_CONSTRAIN: &[u8] = b"<\x7cconstrain\x7c>";
const H_CALL: &[u8] = b"<\x7ccall\x7c>";
const H_RETURN: &[u8] = b"<\x7creturn\x7c>";
const H_ALL: &[&[u8]] = &[H_START, H_END, H_MESSAGE, H_CHANNEL, H_CONSTRAIN, H_CALL, H_RETURN];
const H_BODY: &[&[u8]] = &[H_END, H_CALL, H_RETURN, H_START, H_CHANNEL, H_MESSAGE, H_CONSTRAIN];

/// Harmony message/channel scanner with signal-gated recipient parsing.
#[derive(Debug)]
pub struct HarmonyScanner {
	buffer:           Buffer,
	state:            HarmonyState,
	include_raw_tool: bool,
}

#[derive(Debug, Default)]
enum HarmonyState {
	#[default]
	Outside,
	Header {
		raw: RawBlock,
	},
	Body {
		mode: HarmonyMode,
		id:   Str,
		name: Str,
		body: BytesMut,
		raw:  RawBlock,
	},
}

#[derive(Clone, Copy, Debug)]
enum HarmonyMode {
	Text,
	Thinking,
	Tool,
	Skip,
}

impl HarmonyScanner {
	/// Creates a Harmony scanner.
	#[must_use]
	pub fn new(options: ScannerOptions<'_>) -> Self {
		Self {
			buffer:           Buffer::default(),
			state:            HarmonyState::Outside,
			include_raw_tool: options.include_raw_tool,
		}
	}

	/// Feeds one chunk.
	pub fn feed(&mut self, chunk: Bytes) -> ScanBatch {
		self.buffer.push(&chunk);
		self.consume(false)
	}

	/// Flushes the current body deterministically.
	pub fn flush(&mut self) -> ScanBatch {
		self.consume(true)
	}

	fn consume(&mut self, final_chunk: bool) -> ScanBatch {
		let mut out = ScanBatch::new();
		loop {
			let state = std::mem::take(&mut self.state);
			match state {
				HarmonyState::Outside => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some((at, token)) = earliest(&self.buffer.bytes()[..valid], H_ALL) else {
						emit_visible_tail(&mut self.buffer, H_ALL, final_chunk, &mut out);
						self.state = HarmonyState::Outside;
						break;
					};
					if at != 0 {
						out.push(ScanEvent::Text(self.buffer.take(at)));
					}
					if token == H_START {
						self.buffer.discard(H_START.len());
						let mut raw = RawBlock::new(self.include_raw_tool);
						raw.push(H_START);
						self.state = HarmonyState::Header { raw };
					} else if token == H_CHANNEL {
						self.state = HarmonyState::Header { raw: RawBlock::new(self.include_raw_tool) };
					} else {
						self.buffer.discard(token.len());
						self.state = HarmonyState::Outside;
					}
				},
				HarmonyState::Header { mut raw } => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some(message) = find(&self.buffer.bytes()[..valid], H_MESSAGE) else {
						if final_chunk {
							self.buffer.clear();
							self.state = HarmonyState::Outside;
						} else {
							self.state = HarmonyState::Header { raw };
						}
						break;
					};
					let header = self.buffer.take(message);
					raw.push(&header);
					raw.push(H_MESSAGE);
					self.buffer.discard(H_MESSAGE.len());
					let (role, channel, recipient) = parse_harmony_header(&header);
					let assistant = role.is_empty() || role == "assistant";
					let (mode, id, name) = if !assistant {
						(HarmonyMode::Skip, Str::default(), Str::default())
					} else if !recipient.is_empty() && recipient != "assistant" {
						let name =
							Str::from(recipient.strip_prefix("functions.").unwrap_or(&recipient));
						let id = mint_tool_call_id();
						out.push(ScanEvent::ToolStart { id: id.clone(), name: name.clone() });
						(HarmonyMode::Tool, id, name)
					} else if channel == "analysis" {
						out.push(ScanEvent::ThinkingStart);
						(HarmonyMode::Thinking, Str::default(), Str::default())
					} else {
						(HarmonyMode::Text, Str::default(), Str::default())
					};
					self.state = HarmonyState::Body { mode, id, name, body: BytesMut::new(), raw };
				},
				HarmonyState::Body { mode, id, name, mut body, mut raw } => {
					let valid = self.buffer.valid_len(final_chunk);
					if let Some((at, token)) = earliest(&self.buffer.bytes()[..valid], H_BODY) {
						let delta = self.buffer.take(at);
						emit_harmony(mode, &delta, &mut body, &mut raw, &mut out);
						if token == H_END || token == H_CALL || token == H_RETURN {
							self.buffer.discard(token.len());
							if matches!(mode, HarmonyMode::Tool) {
								raw.push(token);
							}
							finish_harmony(mode, id, name, body, raw, &mut out);
							self.state = HarmonyState::Outside;
						} else if token == H_START {
							finish_harmony(mode, id, name, body, raw, &mut out);
							self.buffer.discard(H_START.len());
							let mut raw = RawBlock::new(self.include_raw_tool);
							raw.push(H_START);
							self.state = HarmonyState::Header { raw };
						} else if token == H_CHANNEL {
							finish_harmony(mode, id, name, body, raw, &mut out);
							self.state =
								HarmonyState::Header { raw: RawBlock::new(self.include_raw_tool) };
						} else {
							self.buffer.discard(token.len());
							if matches!(mode, HarmonyMode::Tool) {
								raw.push(token);
							}
							self.state = HarmonyState::Body { mode, id, name, body, raw };
						}
					} else {
						let hold = if final_chunk {
							0
						} else {
							partial_suffix_overlap_any(&self.buffer.bytes()[..valid], H_BODY)
						};
						let delta = self.buffer.take(valid.saturating_sub(hold));
						emit_harmony(mode, &delta, &mut body, &mut raw, &mut out);
						if final_chunk {
							finish_harmony(mode, id, name, body, raw, &mut out);
							self.state = HarmonyState::Outside;
						} else {
							self.state = HarmonyState::Body { mode, id, name, body, raw };
						}
						break;
					}
				},
			}
		}
		out
	}
}

fn parse_harmony_header(header: &[u8]) -> (String, String, String) {
	let header = String::from_utf8_lossy(header);
	let channel_at = header.find("<\x7cchannel\x7c>");
	let role = header[..channel_at.unwrap_or(header.len())]
		.trim_start()
		.split(|character: char| character == '<' || character.is_whitespace())
		.next()
		.unwrap_or("")
		.to_owned();
	let channel = channel_at
		.map(|at| {
			header[at + H_CHANNEL.len()..]
				.trim_start()
				.split(|character: char| character == '<' || character.is_whitespace())
				.next()
				.unwrap_or("")
				.to_owned()
		})
		.unwrap_or_default();
	let recipient = header
		.split_whitespace()
		.find_map(|part| part.strip_prefix("to="))
		.unwrap_or("")
		.trim_end_matches('<')
		.to_owned();
	(role, channel, recipient)
}

fn emit_harmony(
	mode: HarmonyMode,
	delta: &Bytes,
	body: &mut BytesMut,
	raw: &mut RawBlock,
	out: &mut ScanBatch,
) {
	if delta.is_empty() {
		return;
	}
	match mode {
		HarmonyMode::Text => out.push(ScanEvent::Text(delta.clone())),
		HarmonyMode::Thinking => out.push(ScanEvent::ThinkingDelta(delta.clone())),
		HarmonyMode::Tool => {
			body.extend_from_slice(delta);
			raw.push(delta);
		},
		HarmonyMode::Skip => {},
	}
}

fn finish_harmony(
	mode: HarmonyMode,
	id: Str,
	name: Str,
	body: BytesMut,
	raw: RawBlock,
	out: &mut ScanBatch,
) {
	match mode {
		HarmonyMode::Thinking => {
			out.push(ScanEvent::ThinkingEnd { signature: Bytes::new() });
		},
		HarmonyMode::Tool if !name.is_empty() => {
			let args_json = json_bytes(Value::Object(parse_json_object(&body)));
			out.push(ScanEvent::ToolArgumentDelta { id: id.clone(), delta: args_json.clone() });
			out.push(ScanEvent::ToolEnd { id, name, args_json, raw_block: raw.finish() });
		},
		_ => {},
	}
}

// ---- Gemini -----------------------------------------------------------------

const G_CODE_OPEN: &[u8] = b"```tool_code";
const G_THINK_OPEN: &[u8] = b"```thinking\n";
const G_FENCE: &[u8] = b"```";

/// Gemini Python `tool_code` scanner with nested-fence reasoning protection.
#[derive(Debug)]
pub struct GeminiScanner {
	buffer:           Buffer,
	state:            GeminiState,
	parse_thinking:   bool,
	include_raw_tool: bool,
}

#[derive(Debug, Default)]
enum GeminiState {
	#[default]
	Outside,
	Tool,
	Fenced(FencedThinkingScanner),
}

impl GeminiScanner {
	/// Creates a Gemini scanner.
	#[must_use]
	pub fn new(options: ScannerOptions<'_>) -> Self {
		Self {
			buffer:           Buffer::default(),
			state:            GeminiState::Outside,
			parse_thinking:   options.parse_thinking,
			include_raw_tool: options.include_raw_tool,
		}
	}

	/// Feeds one chunk.
	pub fn feed(&mut self, chunk: Bytes) -> ScanBatch {
		self.buffer.push(&chunk);
		self.consume(false)
	}

	/// Flushes reasoning and drops incomplete code blocks.
	pub fn flush(&mut self) -> ScanBatch {
		self.consume(true)
	}

	fn consume(&mut self, final_chunk: bool) -> ScanBatch {
		let mut out = ScanBatch::new();
		loop {
			let state = std::mem::take(&mut self.state);
			match state {
				GeminiState::Outside => {
					let valid = self.buffer.valid_len(final_chunk);
					let tags: &[&[u8]] = if self.parse_thinking {
						&[G_CODE_OPEN, G_THINK_OPEN]
					} else {
						&[G_CODE_OPEN]
					};
					let Some((at, tag)) = earliest(&self.buffer.bytes()[..valid], tags) else {
						emit_visible_tail(&mut self.buffer, tags, final_chunk, &mut out);
						self.state = GeminiState::Outside;
						break;
					};
					if at != 0 {
						out.push(ScanEvent::Text(self.buffer.take(at)));
					}
					self.buffer.discard(tag.len());
					if tag == G_THINK_OPEN {
						out.push(ScanEvent::ThinkingStart);
						self.state = GeminiState::Fenced(FencedThinkingScanner::new());
					} else {
						self.state = GeminiState::Tool;
					}
				},
				GeminiState::Fenced(mut fenced) => {
					let result = fenced.feed(self.buffer.0.split().freeze(), final_chunk);
					for delta in result.thinking {
						if !delta.is_empty() {
							out.push(ScanEvent::ThinkingDelta(delta));
						}
					}
					if result.closed || final_chunk {
						out.push(ScanEvent::ThinkingEnd { signature: Bytes::new() });
						self.buffer.push(&result.rest);
						self.state = GeminiState::Outside;
					} else {
						self.state = GeminiState::Fenced(fenced);
						break;
					}
				},
				GeminiState::Tool => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some(end) = find(&self.buffer.bytes()[..valid], G_FENCE) else {
						if final_chunk {
							self.buffer.clear();
							self.state = GeminiState::Outside;
						} else {
							self.state = GeminiState::Tool;
						}
						break;
					};
					let body = self.buffer.take(end);
					self.buffer.discard(G_FENCE.len());
					let raw = raw_envelope(self.include_raw_tool, &[G_CODE_OPEN, &body, G_FENCE]);
					for (name, arguments) in parse_python_calls(&body) {
						let id = mint_tool_call_id();
						out.push(ScanEvent::ToolStart { id: id.clone(), name: name.clone() });
						let args_json = json_bytes(Value::Object(arguments));
						out.push(ScanEvent::ToolArgumentDelta {
							id:    id.clone(),
							delta: args_json.clone(),
						});
						out.push(ScanEvent::ToolEnd { id, name, args_json, raw_block: raw.clone() });
					}
					self.state = GeminiState::Outside;
				},
			}
		}
		out
	}
}

// ---- Gemma ------------------------------------------------------------------

const GM_CALL_OPEN: &[u8] = b"<|tool_call>";
const GM_CALL_CLOSE: &[u8] = b"<tool_call|>";
const GM_STRING: &[u8] = b"<|\"|>";
const GM_THINK_OPEN: &[u8] = b"<|channel>thought";
const GM_THINK_CLOSE: &[u8] = b"<channel|>";

/// Gemma token scanner with `<|\"|>` string-span protection.
#[derive(Debug)]
pub struct GemmaScanner {
	buffer:           Buffer,
	state:            GemmaState,
	parse_thinking:   bool,
	include_raw_tool: bool,
}

#[derive(Debug, Default)]
enum GemmaState {
	#[default]
	Outside,
	Thinking {
		initial: bool,
	},
	Tool,
}

impl GemmaScanner {
	/// Creates a Gemma scanner.
	#[must_use]
	pub fn new(options: ScannerOptions<'_>) -> Self {
		Self {
			buffer:           Buffer::default(),
			state:            GemmaState::Outside,
			parse_thinking:   options.parse_thinking,
			include_raw_tool: options.include_raw_tool,
		}
	}

	/// Feeds one chunk.
	pub fn feed(&mut self, chunk: Bytes) -> ScanBatch {
		self.buffer.push(&chunk);
		self.consume(false)
	}

	/// Flushes reasoning and discards incomplete calls.
	pub fn flush(&mut self) -> ScanBatch {
		self.consume(true)
	}

	fn consume(&mut self, final_chunk: bool) -> ScanBatch {
		let mut out = ScanBatch::new();
		loop {
			match self.state {
				GemmaState::Outside => {
					let valid = self.buffer.valid_len(final_chunk);
					let tags: &[&[u8]] = if self.parse_thinking {
						&[GM_CALL_OPEN, GM_THINK_OPEN]
					} else {
						&[GM_CALL_OPEN]
					};
					let Some((at, tag)) = earliest(&self.buffer.bytes()[..valid], tags) else {
						emit_visible_tail(&mut self.buffer, tags, final_chunk, &mut out);
						break;
					};
					if at != 0 {
						out.push(ScanEvent::Text(self.buffer.take(at)));
					}
					self.buffer.discard(tag.len());
					if tag == GM_THINK_OPEN {
						out.push(ScanEvent::ThinkingStart);
						self.state = GemmaState::Thinking { initial: true };
					} else {
						self.state = GemmaState::Tool;
					}
				},
				GemmaState::Thinking { initial } => {
					if initial {
						if self.buffer.bytes().first() == Some(&b'\n') {
							self.buffer.discard(1);
						} else if self.buffer.bytes().is_empty() && !final_chunk {
							break;
						}
						self.state = GemmaState::Thinking { initial: false };
					}
					if consume_reasoning(&mut self.buffer, GM_THINK_CLOSE, final_chunk, &mut out) {
						self.state = GemmaState::Outside;
					} else {
						break;
					}
				},
				GemmaState::Tool => {
					let valid = self.buffer.valid_len(final_chunk);
					let Some(end) = find_gemma_close(&self.buffer.bytes()[..valid]) else {
						if final_chunk {
							self.buffer.clear();
							self.state = GemmaState::Outside;
						}
						break;
					};
					let body = self.buffer.take(end);
					self.buffer.discard(GM_CALL_CLOSE.len());
					if let Some((name, arguments)) = parse_gemma_call(&body) {
						let id = mint_tool_call_id();
						out.push(ScanEvent::ToolStart { id: id.clone(), name: name.clone() });
						let args_json = json_bytes(Value::Object(arguments));
						out.push(ScanEvent::ToolArgumentDelta {
							id:    id.clone(),
							delta: args_json.clone(),
						});
						out.push(ScanEvent::ToolEnd {
							id,
							name,
							args_json,
							raw_block: raw_envelope(self.include_raw_tool, &[
								GM_CALL_OPEN,
								&body,
								GM_CALL_CLOSE,
							]),
						});
					}
					self.state = GemmaState::Outside;
				},
			}
		}
		out
	}
}

fn find_gemma_close(text: &[u8]) -> Option<usize> {
	let mut index = 0;
	while index < text.len() {
		if text[index..].starts_with(GM_STRING) {
			let close = find(&text[index + GM_STRING.len()..], GM_STRING)?;
			index += GM_STRING.len() + close + GM_STRING.len();
			continue;
		}
		if text[index..].starts_with(GM_CALL_CLOSE) {
			return Some(index);
		}
		index += 1;
	}
	None
}

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::*;
	use crate::{Dialect, factory::create_scanner, types::InbandTool};

	fn tool_end(events: impl IntoIterator<Item = ScanEvent>) -> Option<(Str, Value, Bytes)> {
		events.into_iter().find_map(|event| match event {
			ScanEvent::ToolEnd { name, args_json, raw_block, .. } => {
				Some((name, serde_json::from_slice(&args_json).unwrap(), raw_block.unwrap()))
			},
			_ => None,
		})
	}

	fn scan_with_raw(
		dialect: Dialect,
		wire: &[u8],
		chunks: &[usize],
		include_raw_tool: bool,
	) -> Vec<ScanEvent> {
		let schema = json!({"type":"object","properties":{"msg":{"type":"string"}}});
		let tool = InbandTool {
			name:        "echo",
			description: None,
			parameters:  &schema,
			examples:    &[],
		};
		let tools = [tool];
		let options = ScannerOptions {
			tools: &tools,
			parse_thinking: true,
			include_raw_tool,
			xml_tagset: XmlTagset::Anthropic,
		};
		let mut scanner = create_scanner(dialect, options);
		let mut out = Vec::new();
		let mut start = 0;
		for &end in chunks {
			out.extend(scanner.feed(Bytes::copy_from_slice(&wire[start..end])));
			start = end;
		}
		out.extend(scanner.feed(Bytes::copy_from_slice(&wire[start..])));
		out.extend(scanner.flush());
		out
	}

	fn scan(dialect: Dialect, wire: &[u8], chunks: &[usize]) -> Vec<ScanEvent> {
		scan_with_raw(dialect, wire, chunks, true)
	}

	#[derive(Debug, PartialEq)]
	struct Semantics {
		text:       Vec<u8>,
		thinking:   Vec<u8>,
		boundaries: Vec<Vec<u8>>,
		tools:      Vec<(String, Value, Option<Vec<u8>>)>,
	}

	fn semantics(events: impl IntoIterator<Item = ScanEvent>) -> Semantics {
		let mut value = Semantics {
			text:       Vec::new(),
			thinking:   Vec::new(),
			boundaries: Vec::new(),
			tools:      Vec::new(),
		};
		for event in events {
			match event {
				ScanEvent::Text(bytes) => value.text.extend_from_slice(&bytes),
				ScanEvent::ThinkingDelta(bytes) => value.thinking.extend_from_slice(&bytes),
				ScanEvent::ThinkingEnd { signature } => value.boundaries.push(signature.to_vec()),
				ScanEvent::ToolEnd { name, args_json, raw_block, .. } => value.tools.push((
					name.to_string(),
					serde_json::from_slice(&args_json).unwrap(),
					raw_block.map(|raw| raw.to_vec()),
				)),
				_ => {},
			}
		}
		value
	}

	fn fixtures() -> [(Dialect, &'static str); 11] {
		[
			(
				Dialect::Glm,
				"<tool_call>echo\n<arg_key>msg</arg_key><arg_value>x</arg_value></tool_call>",
			),
			(
				Dialect::Hermes,
				"<tool_call>{\"name\":\"echo\",\"arguments\":{\"msg\":\"x\"}}</tool_call>",
			),
			(
				Dialect::Kimi,
				"<|tool_calls_section_begin|><|tool_call_begin|>functions.echo:\
				 0<|tool_call_argument_begin|>{\"msg\":\"x\"\
				 }<|tool_call_end|><|tool_calls_section_end|>",
			),
			(Dialect::Xml, "<invoke name=\"echo\"><parameter name=\"msg\">x</parameter></invoke>"),
			(
				Dialect::Anthropic,
				"<function_calls><invoke name=\"echo\"><parameter \
				 name=\"msg\">x</parameter></invoke></function_calls>",
			),
			(
				Dialect::DeepSeek,
				"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>echo<｜tool▁sep｜>{\"msg\":\"x\"\
				 }<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
			),
			(
				Dialect::Harmony,
				"<\x7cstart\x7c>assistant<\x7cchannel\x7c>analysis \
				 to=functions.echo<\x7cmessage\x7c>{\"msg\":\"x\"}<\x7ccall\x7c>",
			),
			(
				Dialect::Qwen3,
				"<tool_call>{\"name\":\"echo\",\"arguments\":{\"msg\":\"x\"}}</tool_call>",
			),
			(Dialect::Gemini, "```tool_code\ndefault_api.echo(msg=\"x\")\n```"),
			(Dialect::Gemma, "<|tool_call>call:echo{msg:<|\"|>x<|\"|>}<tool_call|>"),
			(
				Dialect::MiniMax,
				"<minimax:tool_call><invoke name=\"echo\"><parameter \
				 name=\"msg\">x</parameter></invoke></minimax:tool_call>",
			),
		]
	}

	#[test]
	fn every_dialect_parses_whole_stream_and_retains_raw_block() {
		for (dialect, wire) in fixtures() {
			let (name, args, raw) =
				tool_end(scan(dialect, wire.as_bytes(), &[])).unwrap_or_else(|| panic!("{dialect:?}"));
			assert_eq!(name, "echo");
			assert_eq!(args, json!({"msg":"x"}));
			assert!(!raw.is_empty());
		}
	}

	#[test]
	fn every_delimiter_split_and_byte_at_a_time_is_stable() {
		for (dialect, wire) in fixtures() {
			let expected = semantics(scan(dialect, wire.as_bytes(), &[]));
			for split in 0..=wire.len() {
				assert_eq!(
					semantics(scan(dialect, wire.as_bytes(), &[split])),
					expected,
					"{dialect:?} split {split}"
				);
			}
			let boundaries: Vec<usize> = (1..wire.len()).collect();
			let actual = semantics(scan(dialect, wire.as_bytes(), &boundaries));
			assert_eq!(actual, expected, "{dialect:?} byte chunks");
			assert!(
				actual.text.is_empty(),
				"{dialect:?}: leaked {}",
				String::from_utf8_lossy(&actual.text)
			);
		}
	}

	#[test]
	fn utf8_code_points_may_span_input_chunks() {
		let wire = "前<think>理🙂由</think>后".as_bytes();
		let boundaries: Vec<usize> = (1..wire.len()).collect();
		let events = scan(Dialect::Qwen3, wire, &boundaries);
		let mut text = Vec::new();
		let mut thinking = Vec::new();
		for event in events {
			match event {
				ScanEvent::Text(bytes) => text.extend_from_slice(&bytes),
				ScanEvent::ThinkingDelta(bytes) => thinking.extend_from_slice(&bytes),
				_ => {},
			}
		}
		assert_eq!(String::from_utf8(text).unwrap(), "前后");
		assert_eq!(String::from_utf8(thinking).unwrap(), "理🙂由");
	}

	#[test]
	fn schema_string_retains_json_looking_raw_value() {
		let events = scan(
			Dialect::Glm,
			b"<tool_call>echo\n<arg_key>msg</arg_key><arg_value>{\"raw\":true}</arg_value></tool_call>",
			&[],
		);
		let (_, args, _) = tool_end(events).unwrap();
		assert_eq!(args, json!({"msg":"{\"raw\":true}"}));
	}

	#[test]
	fn malformed_calls_drop_and_unterminated_thinking_closes() {
		assert!(tool_end(scan(Dialect::Qwen3, b"<tool_call>{\"name\":\"echo\"", &[])).is_none());
		let events = scan(Dialect::Qwen3, b"<think>unfinished", &[]);
		assert!(events.iter().any(
			|event| matches!(event, ScanEvent::ThinkingEnd { signature } if signature.is_empty())
		));
	}

	#[test]
	fn dsml_fullwidth_and_ascii_envelopes_stay_distinct_but_equivalent() {
		for wire in [
			"<｜DSML｜tool_calls><｜DSML｜invoke name=\"echo\"><｜DSML｜parameter \
			 name=\"msg\">x</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
			"<|DSML|tool_calls><|DSML|invoke name=\"echo\"><|DSML|parameter \
			 name=\"msg\">x</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>",
		] {
			let (_, args, _) = tool_end(scan(Dialect::DeepSeek, wire.as_bytes(), &[])).unwrap();
			assert_eq!(args, json!({"msg":"x"}));
		}
	}

	#[test]
	fn raw_capture_can_be_disabled_without_changing_arguments() {
		for (dialect, wire) in fixtures() {
			let events = scan_with_raw(dialect, wire.as_bytes(), &[], false);
			let (name, args, raw) = events
				.into_iter()
				.find_map(|event| match event {
					ScanEvent::ToolEnd { name, args_json, raw_block, .. } => {
						Some((name, serde_json::from_slice::<Value>(&args_json).unwrap(), raw_block))
					},
					_ => None,
				})
				.unwrap_or_else(|| panic!("{dialect:?}"));
			assert_eq!(name, "echo", "{dialect:?}");
			assert_eq!(args, json!({"msg":"x"}), "{dialect:?}");
			assert!(raw.is_none(), "{dialect:?}");
		}
	}

	#[test]
	fn non_tool_xml_and_an_incomplete_final_tag_pass_through_verbatim() {
		let wire = b"before<b data-x='1'>ok</b><ordinary";
		let boundaries: Vec<usize> = (1..wire.len()).collect();
		assert_eq!(semantics(scan(Dialect::Xml, wire, &boundaries)).text, wire);
	}

	#[test]
	fn qualified_xml_thinking_and_finish_flush_emit_signed_boundaries() {
		let cases = [
			(Dialect::Anthropic, "<antml:thinking>partial"),
			(Dialect::Xml, "<scratchpad>partial"),
			(Dialect::Glm, "<think>partial"),
			(Dialect::Hermes, "<think>partial"),
			(Dialect::Kimi, "<think>partial"),
			(Dialect::DeepSeek, "<think>partial"),
			(Dialect::Harmony, "<|start|>assistant<|channel|>analysis<|message|>partial"),
			(Dialect::Qwen3, "<think>partial"),
			(Dialect::Gemini, "```thinking\npartial"),
			(Dialect::Gemma, "<|channel>thought\npartial"),
			(Dialect::MiniMax, "<thinking>partial"),
		];
		for (dialect, wire) in cases {
			let output = semantics(scan(dialect, wire.as_bytes(), &[]));
			assert_eq!(output.thinking, b"partial", "{dialect:?}");
			assert_eq!(output.boundaries, vec![Vec::<u8>::new()], "{dialect:?}");
			assert!(output.text.is_empty(), "{dialect:?}");
		}
	}

	#[test]
	fn python_literals_comments_unicode_and_parse_rejection() {
		let wire = concat!(
			"```tool_code\n[default_api.echo(",
			"msg={\"emoji\":\"\\U0001F600\",\"items\":[1,2],\"ok\":True,\"none\":None}),",
			" # default_api.echo(msg=\"ignored\")\n",
			"default_api.echo(msg=r\"raw\\path\")]\n```",
		);
		let tools = semantics(scan(Dialect::Gemini, wire.as_bytes(), &[])).tools;
		assert_eq!(tools.iter().map(|(_, args, _)| args).collect::<Vec<_>>(), vec![
			&json!({"msg":{"emoji":"😀","items":[1,2],"ok":true,"none":null}}),
			&json!({"msg":"raw\\path"}),
		],);
		assert!(
			tool_end(scan(Dialect::Gemini, b"```tool_code\ndefault_api.echo(msg=unknown)\n```", &[],))
				.is_none()
		);
	}

	#[test]
	fn gemma_string_spans_protect_close_markers() {
		let wire = b"<|tool_call>call:echo{msg:<|\"|>literal <tool_call|> marker<|\"|>}<tool_call|>";
		let boundaries: Vec<usize> = (1..wire.len()).collect();
		let (_, args, _) = tool_end(scan(Dialect::Gemma, wire, &boundaries)).unwrap();
		assert_eq!(args, json!({"msg":"literal <tool_call|> marker"}));
	}

	#[test]
	fn malformed_complete_and_incomplete_calls_are_rejected() {
		for (dialect, wire) in [
			(Dialect::Hermes, "<tool_call>{\"name\":\"echo\",\"arguments\":42}</tool_call>"),
			(
				Dialect::Kimi,
				"<|tool_calls_section_begin|><|tool_call_begin|>functions.echo:\
				 0<|tool_call_argument_begin|>{bad}<|tool_call_end|>",
			),
			(Dialect::Xml, "<invoke name=\"echo\"><parameter name=\"msg\">x"),
			(Dialect::DeepSeek, "<｜tool▁call▁begin｜>echo<｜tool▁sep｜>{bad}<｜tool▁call▁end｜>"),
			(Dialect::Gemma, "<|tool_call>call:echo{msg:[1,}<tool_call|>"),
		] {
			assert!(tool_end(scan(dialect, wire.as_bytes(), &[])).is_none(), "{dialect:?}");
		}
	}
}
