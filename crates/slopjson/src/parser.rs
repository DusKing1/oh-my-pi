//! Shared tolerant lexer: token-level primitives that both the strict
//! [`Deserializer`](crate::Deserializer) and the streaming partial builder
//! ([`parse_streaming`](crate::parse_streaming)) are built on.
//!
//! The grammar is a forgiving superset of JSON covering the malformations
//! LLM tool-call bodies leak in practice:
//!
//! - single-quoted strings and unquoted object keys (JSON5);
//! - trailing / stray commas, and `//` + block comments;
//! - Python literals `True` / `False` / `None`, plus `0x` / `0b` numeric
//!   literals;
//! - raw control characters and invalid `\x` escapes inside strings (kept
//!   literally);
//! - unescaped quotes inside strings — a quote only closes a string when
//!   followed by a value terminator, recovering apostrophes such as `'it's'`;
//! - unquoted string values in value position (strict mode only) — an
//!   unrecognized bareword such as `{"paths": packages/foo/*}` is recovered as
//!   a string up to the next `,` / `}` / `]` / newline.

use omp_core::{CowStr, StrMut};

use crate::{
	error::ParseError,
	hex4, is_whitespace,
	value::{Number, Value},
};

/// Maximum container nesting before the parser refuses (strict) or rolls
/// back to the last valid prefix (partial).
pub const MAX_DEPTH: u32 = 128;

/// A keyword literal: standard JSON plus Python `True`/`False`/`None`.
#[derive(Clone, Copy)]
pub enum Atom {
	Bool(bool),
	Null,
}

impl From<Atom> for Value {
	fn from(atom: Atom) -> Self {
		match atom {
			Atom::Bool(b) => Self::Bool(b),
			Atom::Null => Self::Null,
		}
	}
}

const KEYWORDS: [(&str, Atom); 6] = [
	("true", Atom::Bool(true)),
	("false", Atom::Bool(false)),
	("null", Atom::Null),
	("True", Atom::Bool(true)),
	("False", Atom::Bool(false)),
	("None", Atom::Null),
];

const fn is_ident_char(b: u8) -> bool {
	b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}

/// Cursor over the input with the tolerant token readers.
///
/// `lenient` selects streaming semantics: unterminated strings return their
/// content instead of failing, double-quoted strings recover unescaped inner
/// quotes, and malformed numbers report as absent (`Ok(None)`) so the caller
/// can roll back instead of erroring.
pub struct Parser<'a> {
	src:     &'a str,
	s:       &'a [u8],
	i:       usize,
	lenient: bool,
}

impl<'a> Parser<'a> {
	pub(crate) const fn new(src: &'a str, lenient: bool) -> Self {
		Self { src, s: src.as_bytes(), i: 0, lenient }
	}

	pub(crate) const fn pos(&self) -> usize {
		self.i
	}

	/// Source text from `start` to the current position.
	pub(crate) fn src_from(&self, start: usize) -> &'a str {
		&self.src[start..self.i]
	}

	pub(crate) const fn at_end(&self) -> bool {
		self.i >= self.s.len()
	}

	pub(crate) fn peek(&self) -> Option<u8> {
		self.s.get(self.i).copied()
	}

	pub(crate) const fn bump(&mut self) {
		self.i += 1;
	}

	/// Consume the rest of the input (partial-mode rollback of an
	/// unrecognized trailing token).
	pub(crate) const fn skip_to_end(&mut self) {
		self.i = self.s.len();
	}

	/// Skip whitespace plus `//` line and `/* */` block comments.
	pub(crate) fn ws(&mut self) {
		let s = self.s;
		let n = s.len();
		loop {
			while self.i < n && is_whitespace(s[self.i]) {
				self.i += 1;
			}
			if self.i + 1 < n && s[self.i] == b'/' {
				match s[self.i + 1] {
					b'/' => {
						self.i += 2;
						while self.i < n && s[self.i] != b'\n' {
							self.i += 1;
						}
						continue;
					},
					b'*' => {
						self.i += 2;
						while self.i + 1 < n && !(s[self.i] == b'*' && s[self.i + 1] == b'/') {
							self.i += 1;
						}
						self.i = (self.i + 2).min(n);
						continue;
					},
					_ => {},
				}
			}
			break;
		}
	}

	/// Read a string starting at the opening `quote`. Borrowed (zero-copy)
	/// when the literal needs no unescaping.
	pub(crate) fn string(&mut self, quote: u8) -> Result<CowStr<'a>, ParseError> {
		let s = self.s;
		let n = s.len();
		let mut i = self.i + 1; // skip opening quote
		let mut out: Option<StrMut> = None;
		let mut run_start = i;
		while i < n {
			let b = s[i];
			if b != b'\\' && b != quote {
				i += 1;
				continue;
			}
			if b == quote {
				// Apostrophe / inner-quote recovery (a quote that isn't followed by a
				// value terminator is literal) is safe for single quotes and in
				// lenient (streaming) mode. For double quotes in strict mode, close on
				// the first unescaped quote like standard JSON so malformed structure
				// fails loudly instead of silently swallowing commas/colons.
				if (quote != b'\'' && !self.lenient) || self.closes_string(i + 1) {
					self.i = i + 1;
					return Ok(finish(out, &self.src[run_start..i]));
				}
				// Unescaped inner quote (e.g. apostrophe in `'it's'`) — keep it literal.
				i += 1;
				continue;
			}
			// Backslash escape.
			let out = out.get_or_insert_default();
			out.push_str(&self.src[run_start..i]);
			i += 1;
			if i >= n {
				out.push('\\');
				run_start = i;
				break;
			}
			match s[i] {
				b'"' => out.push('"'),
				b'\'' => out.push('\''),
				b'\\' => out.push('\\'),
				b'/' => out.push('/'),
				b'b' => out.push('\u{0008}'),
				b'f' => out.push('\u{000C}'),
				b'n' => out.push('\n'),
				b'r' => out.push('\r'),
				b't' => out.push('\t'),
				b'u' => match hex4(s, i + 1) {
					Some(unit) => {
						i += 4;
						if let Some(ch) = char::from_u32(unit) {
							out.push(ch);
						} else if (0xd800..0xdc00).contains(&unit)
							&& s.get(i + 1) == Some(&b'\\')
							&& s.get(i + 2) == Some(&b'u')
							&& let Some(low) = hex4(s, i + 3)
							&& (0xdc00..0xe000).contains(&low)
						{
							// Surrogate pair split across two \u escapes.
							let combined = 0x10000 + ((unit - 0xd800) << 10) + (low - 0xdc00);
							out.push(
								char::from_u32(combined)
									.expect("surrogate pair combines to a valid scalar"),
							);
							i += 6;
						} else {
							// Lone surrogate: representable in a JS string, not in Rust.
							out.push('\u{FFFD}');
						}
					},
					None => out.push_str("\\u"), // invalid \u — keep literal
				},
				_ => {
					// Invalid escape — keep the backslash and the escaped char literal.
					let ch = self.src[i..]
						.chars()
						.next()
						.expect("escape byte starts a char");
					out.push('\\');
					out.push(ch);
					i += ch.len_utf8() - 1;
				},
			}
			i += 1;
			run_start = i;
		}
		// Unterminated string: keep the content in lenient mode, fail strict.
		if self.lenient {
			self.i = i;
			return Ok(finish(out, &self.src[run_start..n]));
		}
		Err(ParseError::UnterminatedString)
	}

	/// A quote closes a string only when the next non-space char ends a value.
	fn closes_string(&self, from: usize) -> bool {
		let mut k = from;
		while k < self.s.len() && is_whitespace(self.s[k]) {
			k += 1;
		}
		matches!(self.s.get(k), None | Some(b',' | b'}' | b']' | b':'))
	}

	/// Read a numeric token. `Ok(None)` (lenient mode only) marks a malformed
	/// or truncated number the caller must roll back; strict mode errors.
	pub(crate) fn number(&mut self) -> Result<Option<Number>, ParseError> {
		let start = self.i;
		while self.i < self.s.len()
			&& matches!(
				self.s[self.i],
				b'0'..=b'9' | b'-' | b'+' | b'.' | b'x' | b'X' | b'a'..=b'f' | b'A'..=b'F'
			) {
			self.i += 1;
		}
		match parse_number_token(&self.src[start..self.i]) {
			Some(number) => Ok(Some(number)),
			None if self.lenient => Ok(None),
			None => Err(ParseError::InvalidNumber(start)),
		}
	}

	/// Match a keyword literal at the cursor; consumes only on success.
	/// Requires a non-identifier boundary so `Truex` / `nullish` are not
	/// misread as the keyword followed by junk.
	pub(crate) fn match_keyword(&mut self) -> Option<Atom> {
		for (word, atom) in KEYWORDS {
			if self.s[self.i..].starts_with(word.as_bytes())
				&& !self
					.s
					.get(self.i + word.len())
					.copied()
					.is_some_and(is_ident_char)
			{
				self.i += word.len();
				return Some(atom);
			}
		}
		None
	}

	/// Consume a `null` / `None` literal if present (for `Option` fields).
	/// Never consumes on a non-null keyword such as `true`.
	pub(crate) fn eat_null(&mut self) -> bool {
		for word in ["null", "None"] {
			if self.s[self.i..].starts_with(word.as_bytes())
				&& !self
					.s
					.get(self.i + word.len())
					.copied()
					.is_some_and(is_ident_char)
			{
				self.i += word.len();
				return true;
			}
		}
		false
	}

	/// Read an unquoted object key: everything up to a structural delimiter
	/// or whitespace. May be empty.
	pub(crate) fn unquoted_key(&mut self) -> &'a str {
		let start = self.i;
		while self.i < self.s.len() {
			let b = self.s[self.i];
			if matches!(b, b':' | b',' | b'}') || is_whitespace(b) {
				break;
			}
			self.i += 1;
		}
		&self.src[start..self.i]
	}

	/// Strict-mode recovery of an unquoted string value, e.g.
	/// `{"paths": packages/foo/*}`: consume until `,` / `}` / `]` / newline
	/// and trim trailing whitespace. Recovery still fails — so a final parse
	/// never accepts a half-formed or non-finite argument — when the token:
	/// - hits end-of-input before a delimiter (truncated value);
	/// - contains a `"`, `{`, `[`, or a key-like `:` — this parser accepts
	///   unquoted keys, so a missed comma (`{"a": foo "b": 1}`) would otherwise
	///   silently swallow the following field. A colon followed by `/` or `\`
	///   stays literal so URL and Windows-path values recover;
	/// - is a non-finite JS atom (`NaN` / `Infinity` / `undefined`).
	pub(crate) fn bareword(&mut self) -> Result<&'a str, ParseError> {
		let s = self.s;
		let start = self.i;
		let mut i = start;
		while i < s.len() {
			let b = s[i];
			if matches!(b, b',' | b'}' | b']' | b'\n' | b'\r') {
				break;
			}
			if b == b'"'
				|| b == b'{'
				|| b == b'['
				|| (b == b':' && !matches!(s.get(i + 1).copied(), Some(b'/' | b'\\')))
			{
				return Err(ParseError::UnexpectedToken(start));
			}
			i += 1;
		}
		if i >= s.len() {
			return Err(ParseError::UnexpectedToken(start));
		}
		let mut end = i;
		while end > start && is_whitespace(s[end - 1]) {
			end -= 1;
		}
		let word = &self.src[start..end];
		if matches!(word, "NaN" | "Infinity" | "-Infinity" | "+Infinity" | "undefined") {
			return Err(ParseError::UnexpectedToken(start));
		}
		self.i = i;
		Ok(word)
	}
}

/// Assemble the final string: borrowed when nothing needed unescaping.
fn finish(owned: Option<StrMut>, tail: &str) -> CowStr<'_> {
	match owned {
		None => CowStr::Borrowed(tail),
		Some(mut out) => {
			out.push_str(tail);
			CowStr::Owned(out)
		},
	}
}

/// Parse a relaxed numeric token with JS `Number()` semantics: decimal
/// (optional sign, leading/trailing dot, exponent) plus unsigned `0x` hex and
/// `0b` binary. Integers that fit stay exact integers; everything else is
/// `f64`. `None` for malformed or non-finite tokens — unlike JS, an
/// overflow-to-infinity (`1e999`) is rejected rather than surfaced.
fn parse_number_token(token: &str) -> Option<Number> {
	let bytes = token.as_bytes();
	if bytes.len() > 2 && bytes[0] == b'0' && bytes[1] | 0x20 == b'x' {
		return parse_radix(&bytes[2..], 16);
	}
	if bytes.len() > 2 && bytes[0] == b'0' && bytes[1] | 0x20 == b'b' {
		return parse_radix(&bytes[2..], 2);
	}
	let signed = token.strip_prefix('+').unwrap_or(token);
	let digits = signed.strip_prefix('-').unwrap_or(signed);
	if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
		if signed.starts_with('-') {
			if let Ok(value) = signed.parse::<i64>() {
				return Some(Number::from(value));
			}
		} else if let Ok(value) = digits.parse::<u64>() {
			return Some(Number::from(value));
		}
		// Out-of-range integer: fall through to f64 like JS.
	}
	Number::from_f64(signed.parse::<f64>().ok()?)
}

/// Fold hex/binary digits into an integer, spilling to `f64` on overflow the
/// way JS `Number("0x…")` loses precision instead of failing.
fn parse_radix(digits: &[u8], radix: u32) -> Option<Number> {
	if digits.is_empty() {
		return None;
	}
	let mut int = 0u64;
	let mut float = 0f64;
	let mut overflowed = false;
	for &b in digits {
		let d = u64::from((b as char).to_digit(radix)?);
		if !overflowed {
			if let Some(v) = int
				.checked_mul(u64::from(radix))
				.and_then(|v| v.checked_add(d))
			{
				int = v;
				continue;
			}
			overflowed = true;
			float = int as f64;
		}
		float = float.mul_add(f64::from(radix), d as f64);
	}
	if overflowed {
		Number::from_f64(float)
	} else {
		Some(Number::from(int))
	}
}
