//! In-memory BOM and line-ending normalization helpers.

use std::borrow::Cow;

/// A source file's line-ending convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LineEnding {
	/// Line feed (`\n`).
	Lf,
	/// Carriage-return plus line feed (`\r\n`).
	CrLf,
}

/// A borrowed text body with its optional UTF-8 BOM separated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BomResult<'a> {
	/// Whether the input began with the Unicode UTF-8 BOM scalar.
	pub had_bom: bool,
	/// Input text after removing that leading BOM.
	pub text:    &'a str,
}

/// Detects the first line-ending style, defaulting to LF when none occurs.
pub fn detect_line_ending(content: &str) -> LineEnding {
	match content.find('\n') {
		Some(index) if index > 0 && content.as_bytes()[index - 1] == b'\r' => LineEnding::CrLf,
		_ => LineEnding::Lf,
	}
}

/// Normalizes CRLF and lone CR endings to LF without allocating when possible.
pub fn normalize_to_lf(text: &str) -> Cow<'_, str> {
	if !text.as_bytes().contains(&b'\r') {
		return Cow::Borrowed(text);
	}
	let mut normalized = String::with_capacity(text.len());
	let mut chars = text.chars().peekable();
	while let Some(ch) = chars.next() {
		if ch == '\r' {
			if chars.peek() == Some(&'\n') {
				chars.next();
			}
			normalized.push('\n');
		} else {
			normalized.push(ch);
		}
	}
	Cow::Owned(normalized)
}

/// Restores LF text to the selected line-ending style.
pub fn restore_line_endings(text: &str, ending: LineEnding) -> Cow<'_, str> {
	match ending {
		LineEnding::Lf => Cow::Borrowed(text),
		LineEnding::CrLf => Cow::Owned(text.replace('\n', "\r\n")),
	}
}

/// Removes a leading UTF-8 BOM while retaining whether it was present.
pub fn strip_bom(content: &str) -> BomResult<'_> {
	match content.strip_prefix('\u{feff}') {
		Some(text) => BomResult { had_bom: true, text },
		None => BomResult { had_bom: false, text: content },
	}
}

/// Restores a previously stripped UTF-8 BOM without allocating when absent.
pub fn restore_bom(text: &str, had_bom: bool) -> Cow<'_, str> {
	if !had_bom {
		return Cow::Borrowed(text);
	}
	let mut restored = String::with_capacity(text.len() + 3);
	restored.push('\u{feff}');
	restored.push_str(text);
	Cow::Owned(restored)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn detects_first_ending_and_round_trips_shape() {
		assert_eq!(detect_line_ending("a\r\nb\nc"), LineEnding::CrLf);
		assert_eq!(detect_line_ending("a\nb\r\nc"), LineEnding::Lf);
		assert_eq!(normalize_to_lf("a\r\nb\rc"), "a\nb\nc");
		assert_eq!(restore_line_endings("a\nb", LineEnding::CrLf), "a\r\nb");
	}

	#[test]
	fn separates_and_restores_bom() {
		let stripped = strip_bom("\u{feff}text");
		assert!(stripped.had_bom);
		assert_eq!(stripped.text, "text");
		assert_eq!(restore_bom(stripped.text, stripped.had_bom), "\u{feff}text");
	}
}
