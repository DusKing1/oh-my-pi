//! Stateful line tokenization.

use omp_core::Str;

use crate::{
	format::{
		ABORT_MARKER, BEGIN_PATCH_MARKER, END_PATCH_MARKER, HL_CUT_KEYWORD, HL_FILE_HASH_LENGTH,
		HL_MOVE_KEYWORD, HL_PUT_KEYWORD, HL_REM_KEYWORD,
	},
	types::{Anchor, Cursor, Diagnostic, DiagnosticCode, ParseError, ParsedRange},
};

const REGISTER_NAME_MAX: usize = 64;

/// A parsed operation target before body rows are consumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockTarget {
	/// Replace an inclusive range, optionally from a register.
	Replace {
		/// The source range.
		range:    ParsedRange,
		/// A named register when this is a bodyless paste.
		register: Option<Str>,
	},
	/// Replace a syntactic block, optionally from a register.
	Block {
		/// The block opener.
		anchor:   Anchor,
		/// A named register when this is a bodyless paste.
		register: Option<Str>,
	},
	/// Insert or paste before an anchor.
	InsertBefore {
		/// The source anchor.
		anchor:   Anchor,
		/// An optional named register.
		register: Option<Str>,
	},
	/// Insert or paste after an anchor.
	InsertAfter {
		/// The source anchor.
		anchor:   Anchor,
		/// An optional named register.
		register: Option<Str>,
	},
	/// Insert or paste after a resolved syntactic block.
	InsertAfterBlock {
		/// The block opener.
		anchor:   Anchor,
		/// An optional named register.
		register: Option<Str>,
	},
	/// Capture and remove an inclusive range.
	Cut {
		/// The captured range.
		range:    ParsedRange,
		/// An optional named register.
		register: Option<Str>,
	},
	/// Capture and remove a resolved syntactic block.
	CutBlock {
		/// The block opener.
		anchor:   Anchor,
		/// An optional named register.
		register: Option<Str>,
	},
	/// Insert or paste at the beginning of the file.
	Bof {
		/// An optional named register.
		register: Option<Str>,
	},
	/// Insert or paste at the end of the file.
	Eof {
		/// An optional named register.
		register: Option<Str>,
	},
	/// Remove the whole file.
	Rem,
	/// Move the whole file.
	Move {
		/// The destination path.
		dest: Str,
	},
}

impl BlockTarget {
	/// Returns the target's register when it supports one.
	pub const fn register(&self) -> Option<&Str> {
		match self {
			Self::Replace { register, .. }
			| Self::Block { register, .. }
			| Self::InsertBefore { register, .. }
			| Self::InsertAfter { register, .. }
			| Self::InsertAfterBlock { register, .. }
			| Self::Cut { register, .. }
			| Self::CutBlock { register, .. }
			| Self::Bof { register }
			| Self::Eof { register } => register.as_ref(),
			Self::Rem | Self::Move { .. } => None,
		}
	}

	/// Returns whether this target is a clipboard cut.
	pub const fn is_cut(&self) -> bool {
		matches!(self, Self::Cut { .. } | Self::CutBlock { .. })
	}
}

/// A line-oriented token retaining its authored patch line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
	/// An empty physical line.
	Blank {
		/// The authored patch line.
		line_num: usize,
	},
	/// The optional opening envelope marker.
	EnvelopeBegin {
		/// The authored patch line.
		line_num: usize,
	},
	/// The closing envelope marker.
	EnvelopeEnd {
		/// The authored patch line.
		line_num: usize,
	},
	/// The truncation abort marker.
	Abort {
		/// The authored patch line.
		line_num: usize,
	},
	/// A strict section header.
	Header {
		/// Authored patch line.
		line_num:  usize,
		/// Header path.
		path:      Str,
		/// Optional uppercase four-hex snapshot tag.
		file_hash: Option<Str>,
	},
	/// A parsed hunk or file-operation header.
	Operation {
		/// Authored patch line.
		line_num:  usize,
		/// Parsed operation target.
		target:    BlockTarget,
		/// Whether the header ended in `:`.
		had_colon: bool,
	},
	/// A body row with exactly one leading `+` removed.
	PayloadLiteral {
		/// Authored patch line.
		line_num: usize,
		/// Literal row text.
		text:     Str,
	},
	/// Any otherwise unclassified row.
	Raw {
		/// Authored patch line.
		line_num: usize,
		/// Unclassified row text.
		text:     Str,
	},
}

impl Token {
	/// Returns the token's one-indexed authored patch line.
	pub const fn line_num(&self) -> usize {
		match self {
			Self::Blank { line_num }
			| Self::EnvelopeBegin { line_num }
			| Self::EnvelopeEnd { line_num }
			| Self::Abort { line_num }
			| Self::Header { line_num, .. }
			| Self::Operation { line_num, .. }
			| Self::PayloadLiteral { line_num, .. }
			| Self::Raw { line_num, .. } => *line_num,
		}
	}
}

/// Lazily splits LF or CRLF text into physical lines without retaining
/// terminators.
pub fn split_hashline_lines(text: &str) -> impl Clone + std::iter::FusedIterator<Item = &str> {
	text
		.split_terminator('\n')
		.map(|line| line.strip_suffix('\r').unwrap_or(line))
		.chain(text.is_empty().then_some(""))
}

/// Parses a bare positive line anchor.
pub fn parse_lid(raw: &str, line_num: usize) -> Result<Anchor, ParseError> {
	let trimmed = raw.trim();
	match parse_positive_usize(trimmed).filter(|_| trimmed.bytes().all(|byte| byte.is_ascii_digit()))
	{
		Some(line) => Ok(Anchor { line }),
		None => Err(ParseError::new(Diagnostic::error(
			DiagnosticCode::InvalidLocator,
			Some(line_num),
			None,
			format!(
				"line {line_num}: expected a positive line number such as \"119\", \"42\", or \"7\"; \
				 got {raw:?}."
			),
		))),
	}
}

/// Returns whether text parses as a complete hunk or file-operation header.
pub fn is_hunk_header_text(text: &str) -> bool {
	try_parse_hunk_header(text).is_some()
}

/// Incremental line tokenizer.
#[derive(Debug)]
pub struct Tokenizer {
	buffer:        String,
	next_line_num: usize,
	closed:        bool,
}

impl Default for Tokenizer {
	fn default() -> Self {
		Self::new()
	}
}

impl Tokenizer {
	/// Constructs a tokenizer beginning at patch line one.
	pub const fn new() -> Self {
		Self { buffer: String::new(), next_line_num: 1, closed: false }
	}

	/// Feeds a text chunk and returns tokens for complete physical lines.
	pub fn feed(&mut self, chunk: &str) -> Result<Vec<Token>, ParseError> {
		if self.closed {
			return Err(ParseError::new(Diagnostic::error(
				DiagnosticCode::OrphanPayload,
				None,
				None,
				"Tokenizer is closed; call reset() before reusing it.",
			)));
		}
		if chunk.is_empty() {
			return Ok(Vec::new());
		}
		self.buffer.push_str(chunk);
		Ok(self.drain_complete_lines())
	}

	/// Closes the stream and emits its unterminated final line, if any.
	pub fn end(&mut self) -> Vec<Token> {
		if self.closed {
			return Vec::new();
		}
		self.closed = true;
		if self.buffer.is_empty() {
			return Vec::new();
		}
		let line = self.buffer.strip_suffix('\r').unwrap_or(&self.buffer);
		let token = classify_line(line, self.next_line_num);
		self.next_line_num += 1;
		self.buffer.clear();
		vec![token]
	}

	/// Resets buffered text and authored line numbering.
	pub fn reset(&mut self) {
		self.buffer.clear();
		self.next_line_num = 1;
		self.closed = false;
	}

	/// Tokenizes a complete input in one call.
	pub fn tokenize_all(&mut self, text: &str) -> Result<Vec<Token>, ParseError> {
		self.reset();
		let mut tokens = self.feed(text)?;
		tokens.extend(self.end());
		Ok(tokens)
	}

	/// Classifies one physical line using an explicit authored line number.
	pub fn tokenize(&self, line: &str, line_num: usize) -> Token {
		classify_line(line, line_num)
	}

	/// Returns whether a line is a complete operation header.
	pub fn is_op(&self, line: &str) -> bool {
		try_parse_hunk_header(line).is_some()
	}

	/// Returns whether a line is a strict section header.
	pub fn is_header(&self, line: &str) -> bool {
		try_parse_header(line).is_some()
	}

	/// Returns whether a line is an envelope control marker.
	pub fn is_envelope_marker(&self, line: &str) -> bool {
		marker_equals(line, BEGIN_PATCH_MARKER)
			|| marker_equals(line, END_PATCH_MARKER)
			|| marker_equals(line, ABORT_MARKER)
	}

	fn drain_complete_lines(&mut self) -> Vec<Token> {
		let mut tokens = Vec::new();
		let mut start = 0;
		for (index, byte) in self.buffer.bytes().enumerate() {
			if byte != b'\n' {
				continue;
			}
			let mut stop = index;
			if stop > start && self.buffer.as_bytes()[stop - 1] == b'\r' {
				stop -= 1;
			}
			tokens.push(classify_line(&self.buffer[start..stop], self.next_line_num));
			self.next_line_num += 1;
			start = index + 1;
		}
		if start > 0 {
			self.buffer.drain(..start);
		}
		tokens
	}
}

fn marker_equals(line: &str, marker: &str) -> bool {
	line.trim_end() == marker
}

fn classify_line(line: &str, line_num: usize) -> Token {
	if line.is_empty() {
		return Token::Blank { line_num };
	}
	if marker_equals(line, BEGIN_PATCH_MARKER) {
		return Token::EnvelopeBegin { line_num };
	}
	if marker_equals(line, END_PATCH_MARKER) {
		return Token::EnvelopeEnd { line_num };
	}
	if marker_equals(line, ABORT_MARKER) {
		return Token::Abort { line_num };
	}
	if let Some((path, file_hash)) = try_parse_header(line) {
		return Token::Header { line_num, path, file_hash };
	}
	if let Some((target, had_colon)) = try_parse_hunk_header(line) {
		return Token::Operation { line_num, target, had_colon };
	}
	if let Some(text) = line.strip_prefix('+') {
		return Token::PayloadLiteral { line_num, text: text.into() };
	}
	Token::Raw { line_num, text: line.into() }
}

fn try_parse_header(line: &str) -> Option<(Str, Option<Str>)> {
	let trimmed = line.trim_end();
	let body = trimmed.strip_prefix('[')?.strip_suffix(']')?;
	if body.is_empty() {
		return None;
	}
	if let Some((path, tag)) = body.rsplit_once('#') {
		if path.is_empty()
			|| path.contains('#')
			|| tag.len() != HL_FILE_HASH_LENGTH
			|| !tag.bytes().all(|byte| byte.is_ascii_hexdigit())
		{
			return None;
		}
		let mut uppercase = omp_core::StrMut::with_capacity(HL_FILE_HASH_LENGTH);
		for byte in tag.bytes() {
			uppercase.push(char::from(byte.to_ascii_uppercase()));
		}
		return Some((path.into(), Some(uppercase.freeze())));
	}
	Some((body.into(), None))
}

fn try_parse_hunk_header(line: &str) -> Option<(BlockTarget, bool)> {
	let trimmed = line.trim();
	if trimmed == HL_REM_KEYWORD {
		return Some((BlockTarget::Rem, false));
	}
	if let Some(rest) = keyword_rest(trimmed, HL_MOVE_KEYWORD) {
		let dest = parse_move_dest(rest)?;
		return Some((BlockTarget::Move { dest: dest.into() }, false));
	}
	if let Some(rest) = keyword_rest(trimmed, HL_PUT_KEYWORD) {
		return parse_put_target(rest);
	}
	if let Some(rest) = keyword_rest(trimmed, HL_CUT_KEYWORD) {
		return parse_cut_target(rest);
	}
	None
}

fn keyword_rest<'a>(line: &'a str, keyword: &str) -> Option<&'a str> {
	let rest = line.strip_prefix(keyword)?;
	let first = rest.chars().next()?;
	if !first.is_whitespace() && first != ':' {
		return None;
	}
	Some(rest.trim_start())
}

fn split_colon_register(rest: &str) -> Option<(&str, Option<Str>, bool)> {
	let trimmed = rest.trim_end();
	let (without_colon, had_colon) = match trimmed.strip_suffix(':') {
		Some(prefix) => (prefix.trim_end(), true),
		None => (trimmed, false),
	};
	let (locator, register) = if let Some(split) = without_colon.rfind('@') {
		let name = &without_colon[split + 1..];
		if !valid_register(name) {
			return None;
		}
		(without_colon[..split].trim_end(), Some(name.into()))
	} else {
		(without_colon, None)
	};
	Some((locator.trim(), register, had_colon))
}

fn valid_register(name: &str) -> bool {
	!name.is_empty()
		&& name.len() <= REGISTER_NAME_MAX
		&& name
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

fn parse_put_target(rest: &str) -> Option<(BlockTarget, bool)> {
	let (locator, register, had_colon) = split_colon_register(rest)?;
	let first = locator.chars().next();
	if matches!(first, Some('<' | '>')) {
		let after = first == Some('>');
		let gap = locator[1..].trim_start();
		if after && gap == "$" {
			return Some((BlockTarget::Eof { register }, had_colon));
		}
		let (number, block) = parse_number_star(gap)?;
		let anchor = Anchor { line: number };
		let target = if after && block {
			BlockTarget::InsertAfterBlock { anchor, register }
		} else if after {
			BlockTarget::InsertAfter { anchor, register }
		} else if number == 1 {
			BlockTarget::Bof { register }
		} else {
			BlockTarget::InsertBefore { anchor, register }
		};
		return Some((target, had_colon));
	}
	let (range, had_separator, block) = parse_locator_range(locator)?;
	if block {
		if had_separator {
			return None;
		}
		return Some((BlockTarget::Block { anchor: range.start, register }, had_colon));
	}
	Some((BlockTarget::Replace { range, register }, had_colon))
}

fn parse_cut_target(rest: &str) -> Option<(BlockTarget, bool)> {
	let (locator, register, had_colon) = split_colon_register(rest)?;
	let (range, had_separator, block) = parse_locator_range(locator)?;
	if block {
		if had_separator {
			return None;
		}
		return Some((BlockTarget::CutBlock { anchor: range.start, register }, had_colon));
	}
	Some((BlockTarget::Cut { range, register }, had_colon))
}

fn parse_number_star(text: &str) -> Option<(usize, bool)> {
	let trimmed = text.trim();
	let (digits, block) = match trimmed.strip_suffix('*') {
		Some(digits) => (digits, true),
		None => (trimmed, false),
	};
	Some((parse_positive_usize(digits)?, block))
}

fn parse_locator_range(text: &str) -> Option<(ParsedRange, bool, bool)> {
	let trimmed = text.trim();
	if let Some(digits) = trimmed.strip_suffix('*') {
		let line = parse_positive_usize(digits)?;
		let anchor = Anchor { line };
		return Some((ParsedRange { start: anchor, end: anchor }, false, true));
	}
	let first_end = trimmed
		.bytes()
		.position(|byte| !byte.is_ascii_digit())
		.unwrap_or(trimmed.len());
	let start = parse_positive_usize(&trimmed[..first_end])?;
	if first_end == trimmed.len() {
		let anchor = Anchor { line: start };
		return Some((ParsedRange { start: anchor, end: anchor }, false, false));
	}
	let rest = &trimmed[first_end..];
	let second_start = rest
		.char_indices()
		.find_map(|(index, ch)| ch.is_ascii_digit().then_some(index))?;
	let separator = &rest[..second_start];
	if separator.is_empty()
		|| !separator
			.chars()
			.all(|ch| ch.is_whitespace() || matches!(ch, '-' | '.' | '=' | '…'))
	{
		return None;
	}
	let end_text = &rest[second_start..];
	if !end_text.bytes().all(|byte| byte.is_ascii_digit()) {
		return None;
	}
	let end = parse_positive_usize(end_text)?;
	Some((ParsedRange { start: Anchor { line: start }, end: Anchor { line: end } }, true, false))
}

fn parse_positive_usize(text: &str) -> Option<usize> {
	const JS_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
	if text.is_empty() || text.starts_with('0') || !text.bytes().all(|byte| byte.is_ascii_digit()) {
		return None;
	}
	let line: u64 = text.parse().ok()?;
	(line <= JS_MAX_SAFE_INTEGER)
		.then(|| usize::try_from(line).ok())
		.flatten()
}

fn parse_move_dest(rest: &str) -> Option<&str> {
	let trimmed = rest.trim();
	if trimmed.is_empty() {
		return None;
	}
	let quote = trimmed.as_bytes()[0];
	if quote != b'\'' && quote != b'"' {
		return Some(trimmed);
	}
	if trimmed.len() < 2 || trimmed.as_bytes()[trimmed.len() - 1] != quote {
		return None;
	}
	let inner = &trimmed[1..trimmed.len() - 1];
	let mut escaped = false;
	for byte in inner.bytes() {
		if escaped {
			escaped = false;
		} else if byte == b'\\' {
			escaped = true;
		} else if byte == quote {
			return None;
		}
	}
	(!escaped).then_some(inner)
}

/// Clones a cursor while preserving its anchor value.
pub const fn clone_cursor(cursor: Cursor) -> Cursor {
	cursor
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn tokenizes_every_locator_family() {
		let tokenizer = Tokenizer::new();
		for header in [
			"PUT 2.=4:",
			"PUT 2*:",
			"PUT <2:",
			"PUT >2:",
			"PUT >2*:",
			"PUT <1:",
			"PUT >$:",
			"PUT >2 @r",
			"PUT 2.=4 @r",
			"PUT 2* @r",
			"CUT 2.=4",
			"CUT 2*",
			"REM",
			"MV 'new path.rs'",
		] {
			assert!(tokenizer.is_op(header), "{header}");
		}
	}

	#[test]
	fn accepts_lenient_range_separators() {
		let tokenizer = Tokenizer::new();
		for separator in ["-", "=", ".", "..", "…", " ", ".=-"] {
			assert!(tokenizer.is_op(&format!("PUT 2{separator}4:")));
			assert!(tokenizer.is_op(&format!("CUT 2{separator}4")));
		}
	}

	#[test]
	fn strict_headers_preserve_spaces_and_normalize_tags() {
		assert_eq!(classify_line("[dir with spaces/a.rs#1a2b]", 7), Token::Header {
			line_num:  7,
			path:      "dir with spaces/a.rs".into(),
			file_hash: Some("1A2B".into()),
		});
		assert!(matches!(classify_line("[a.rs#1A2G]", 1), Token::Raw { .. }));
	}

	#[test]
	fn streaming_preserves_physical_line_numbers() {
		let mut tokenizer = Tokenizer::new();
		assert_eq!(tokenizer.feed("PUT 2:\r\n+B").unwrap().len(), 1);
		assert_eq!(tokenizer.end()[0].line_num(), 2);
	}
}
