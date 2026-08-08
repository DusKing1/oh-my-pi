//! Hashline sigils, display helpers, and snapshot-tag computation.

use std::fmt::Write;

use omp_core::{SmolStr, format_smol};
use xxhash_rust::xxh32::Xxh32;

use crate::types::Cursor;

/// File-section opening delimiter.
pub const HL_FILE_PREFIX: &str = "[";
/// File-section closing delimiter.
pub const HL_FILE_SUFFIX: &str = "]";
/// Literal body-row sigil.
pub const HL_PAYLOAD_REPLACE: char = '+';
/// Content or clipboard write keyword.
pub const HL_PUT_KEYWORD: &str = "PUT";
/// Clipboard capture and deletion keyword.
pub const HL_CUT_KEYWORD: &str = "CUT";
/// Whole-file removal keyword.
pub const HL_REM_KEYWORD: &str = "REM";
/// Whole-file move keyword.
pub const HL_MOVE_KEYWORD: &str = "MV";
/// Separator between a path and snapshot tag.
pub const HL_FILE_HASH_SEP: char = '#';
/// Canonical inclusive-range separator.
pub const HL_RANGE_SEP: &str = ".=";
/// Number of hexadecimal characters in a snapshot tag.
pub const HL_FILE_HASH_LENGTH: usize = 4;
/// Optional patch-envelope opening marker.
pub const BEGIN_PATCH_MARKER: &str = "*** Begin Patch";
/// Optional patch-envelope closing marker.
pub const END_PATCH_MARKER: &str = "*** End Patch";
/// Truncated-call marker that terminates parsing without a warning.
pub const ABORT_MARKER: &str = "*** Abort";

/// Normalizes file text exactly as the TypeScript tag implementation does.
///
/// Spaces, tabs, and carriage returns immediately before each LF or end of
/// input are removed. Other whitespace and every LF are retained.
pub fn normalize_file_hash_text(text: &str) -> String {
	let mut normalized = String::with_capacity(text.len());
	for segment in text.split_inclusive('\n') {
		let (body, newline) = segment
			.strip_suffix('\n')
			.map_or((segment, ""), |body| (body, "\n"));
		normalized.push_str(body.trim_end_matches([' ', '\t', '\r']));
		normalized.push_str(newline);
	}
	normalized
}

/// Hashes the tag-normalized byte stream without materializing a copy.
pub(crate) fn normalized_file_xxh32(exact: &[u8]) -> u32 {
	let mut hasher = Xxh32::new(0);
	let mut segment_start = 0;
	for (index, byte) in exact.iter().enumerate() {
		if *byte != b'\n' {
			continue;
		}
		let mut end = index;
		while end > segment_start && matches!(exact[end - 1], b' ' | b'\t' | b'\r') {
			end -= 1;
		}
		hasher.update(&exact[segment_start..end]);
		hasher.update(b"\n");
		segment_start = index + 1;
	}
	let mut end = exact.len();
	while end > segment_start && matches!(exact[end - 1], b' ' | b'\t' | b'\r') {
		end -= 1;
	}
	hasher.update(&exact[segment_start..end]);
	hasher.digest()
}

/// Computes the uppercase four-hex xxHash32 snapshot tag used by `/work/pi`.
pub fn compute_file_hash(text: &str) -> SmolStr {
	format_smol!("{:04X}", normalized_file_xxh32(text.as_bytes()) & 0xffff)
}

/// Formats a concrete replacement header such as `PUT 5.=9:`.
pub fn format_replace_header(start: usize, end: usize) -> SmolStr {
	format_smol!("PUT {start}.={end}:")
}

/// Formats a concrete cut header such as `CUT 5.=9`.
pub fn format_cut_header(start: usize, end: usize) -> SmolStr {
	format_smol!("CUT {start}.={end}")
}

/// Formats a gap locator such as `<5`, `>5`, `<1`, or `>$`.
pub fn format_gap_locator(cursor: Cursor) -> SmolStr {
	match cursor {
		Cursor::Bof => SmolStr::from("<1"),
		Cursor::Eof => SmolStr::from(">$"),
		Cursor::BeforeAnchor { anchor } => format_smol!("<{}", anchor.line),
		Cursor::AfterAnchor { anchor } => format_smol!(">{}", anchor.line),
	}
}

/// Formats an insertion header for a cursor.
pub fn format_insert_header(cursor: Cursor) -> SmolStr {
	format_smol!("PUT {}:", format_gap_locator(cursor))
}

/// Formats a named register reference.
pub fn format_register(name: &str) -> SmolStr {
	format_smol!("@{name}")
}

/// Formats a section header from a path and snapshot tag.
pub fn format_hashline_header(path: &str, tag: &str) -> SmolStr {
	format_smol!("[{path}#{tag}]")
}

/// Formats one displayed source row as `LINE:TEXT`.
pub fn format_numbered_line(line: usize, text: &str) -> SmolStr {
	format_smol!("{line}:{text}")
}

/// Formats source text with one-indexed line prefixes.
pub fn format_numbered_lines(text: &str, start_line: usize) -> String {
	let mut output = String::with_capacity(text.len());
	for (index, line) in text.split('\n').enumerate() {
		if index > 0 {
			output.push('\n');
		}
		let _ = write!(output, "{}:{line}", start_line + index);
	}
	output
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn hash_normalization_matches_trailing_whitespace_contract() {
		assert_eq!(normalize_file_hash_text("a  \r\nb\t\n c "), "a\nb\n c");
		assert_eq!(compute_file_hash("a  \r\nb\t\n c "), compute_file_hash("a\nb\n c"));
		assert_eq!(compute_file_hash("").len(), 4);
		assert!(
			compute_file_hash("hello")
				.chars()
				.all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_lowercase())
		);
	}
}
