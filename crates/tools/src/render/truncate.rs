use std::{borrow::Cow, fmt::Write as _};

use bytes::Bytes;
use omp_core::Str;
use omp_tool::BlobRef;
use xutf::{Encoding as _, Utf8, Utf16};

use crate::read::{Fault as ReadFault, ReadBlobs};

/// Default maximum number of rendered output lines.
pub(crate) const DEFAULT_MAX_LINES: usize = 3000;
/// Default maximum rendered output size in UTF-8 bytes.
pub(crate) const DEFAULT_MAX_BYTES: usize = 50 * 1024;
/// Default maximum number of UTF-16 code units in one rendered line.
pub(crate) const DEFAULT_MAX_COLUMN: u32 = 512;

/// Limit that caused a head truncation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TruncatedBy {
	/// The line limit was reached first.
	Lines,
	/// The byte limit was reached first.
	Bytes,
}

/// Limits applied by [`truncate_head`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TruncationOptions {
	/// Maximum number of complete lines to retain.
	pub max_lines: usize,
	/// Maximum number of UTF-8 bytes to retain.
	pub max_bytes: usize,
}

impl Default for TruncationOptions {
	fn default() -> Self {
		Self { max_lines: DEFAULT_MAX_LINES, max_bytes: DEFAULT_MAX_BYTES }
	}
}

/// A borrowed head-truncation result with counts for notices and blob spills.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TruncationResult<'a> {
	/// Complete retained lines from the start of the input.
	pub content:                  &'a str,
	/// Whether any input was omitted.
	pub truncated:                bool,
	/// The limit that caused truncation, when truncated.
	pub truncated_by:             Option<TruncatedBy>,
	/// Number of lines in the original input.
	pub total_lines:              usize,
	/// UTF-8 byte length of the original input.
	pub total_bytes:              usize,
	/// Number of complete lines retained when truncated.
	pub output_lines:             Option<usize>,
	/// UTF-8 byte length retained when truncated.
	pub output_bytes:             Option<usize>,
	/// Whether the retained last line is partial.
	pub last_line_partial:        bool,
	/// Whether no content fit because the first line exceeded the byte limit.
	pub first_line_exceeds_limit: bool,
}

impl TruncationResult<'_> {
	/// Number of lines represented by `content`.
	pub fn shown_lines(&self) -> usize {
		self.output_lines.unwrap_or(self.total_lines)
	}
}

/// Complete pre-projection text after applying the shared output bounds.
///
/// When content was omitted, `blob` names the durable copy of the original
/// text and the line counts retain the exact footer truth.
pub(crate) struct SpilledText {
	pub content:     Str,
	pub blob:        Option<BlobRef>,
	pub shown_lines: u64,
	pub total_lines: u64,
}

/// Applies the standard text bounds and durably stores the complete text before
/// returning a bounded projection.
pub(crate) async fn spill_truncated_text<B: ReadBlobs>(
	full_text: String,
	blobs: &B,
) -> Result<SpilledText, ReadFault> {
	let truncation = truncate_head(&full_text, TruncationOptions::default());
	let shown_lines = u64::try_from(truncation.shown_lines()).unwrap_or(u64::MAX);
	let total_lines = u64::try_from(truncation.total_lines).unwrap_or(u64::MAX);
	if !truncation.truncated {
		return Ok(SpilledText {
			content: Str::from(full_text),
			blob: None,
			shown_lines,
			total_lines,
		});
	}

	let content = truncation.content.to_owned();
	let bytes = Bytes::from(full_text);
	let blob = blobs
		.store(bytes, Str::new_static("text/plain; charset=utf-8"))
		.await?;
	let mut content = content;
	append_blob_truncation_notice_counts(&mut content, shown_lines, total_lines, &blob.hash);
	Ok(SpilledText { content: Str::from(content), blob: Some(blob), shown_lines, total_lines })
}

/// A borrowed result from [`truncate_head_bytes`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ByteTruncationResult<'a> {
	/// Longest valid UTF-8 prefix within the byte limit.
	pub text:  &'a str,
	/// UTF-8 byte length of `text`.
	pub bytes: usize,
}

/// A possibly-owned result from [`truncate_line`].
#[expect(dead_code, reason = "pi parity primitive retained for streaming adapters")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LineTruncationResult<'a> {
	/// Original line, or its retained prefix followed by an ellipsis.
	pub text:          Cow<'a, str>,
	/// Whether the line exceeded the column limit.
	pub was_truncated: bool,
}

/// Retains the longest valid UTF-8 prefix no larger than `max_bytes`.
///
/// The returned text borrows the input and never ends inside a UTF-8 scalar.
pub(crate) fn truncate_head_bytes(text: &str, max_bytes: usize) -> ByteTruncationResult<'_> {
	if text.len() <= max_bytes {
		return ByteTruncationResult { text, bytes: text.len() };
	}

	let mut rest = text.as_bytes();
	let mut end = 0usize;
	while !rest.is_empty() {
		let mut tail = rest;
		Utf8::decode(&mut tail);
		let decoded_bytes = rest.len() - tail.len();
		if end + decoded_bytes > max_bytes {
			break;
		}
		end += decoded_bytes;
		rest = tail;
	}
	ByteTruncationResult { text: &text[..end], bytes: end }
}

/// Truncates one line at pi's JavaScript UTF-16 column boundary.
///
/// A truncated line ends with `…`; an unmodified line remains borrowed.
#[expect(dead_code, reason = "pi parity primitive retained for streaming adapters")]
pub(crate) fn truncate_line(line: &str, max_chars: usize) -> LineTruncationResult<'_> {
	// Every UTF-16 code unit occupies at least one UTF-8 byte, so this is the
	// overwhelmingly common no-truncation path without scanning the string.
	if line.len() <= max_chars {
		return LineTruncationResult { text: Cow::Borrowed(line), was_truncated: false };
	}

	let mut code_units = 0usize;
	let mut rest = line.as_bytes();
	while !rest.is_empty() {
		let index = line.len() - rest.len();
		let codepoint = Utf8::decode(&mut rest);
		let codepoint_units = Utf16::<false>::encoded_length(codepoint);
		let next_units = code_units + codepoint_units;
		if next_units > max_chars {
			let split_surrogate = code_units < max_chars;
			let mut text =
				String::with_capacity(index + usize::from(split_surrogate) * 3 + '…'.len_utf8());
			text.push_str(&line[..index]);
			// JavaScript String.slice can retain one half of a surrogate pair.
			// Its UTF-8 projection is the replacement character.
			if split_surrogate {
				text.push('\u{FFFD}');
			}
			text.push('…');
			return LineTruncationResult { text: Cow::Owned(text), was_truncated: true };
		}

		code_units = next_units;
		if code_units == max_chars {
			let end = line.len() - rest.len();
			if rest.is_empty() {
				return LineTruncationResult {
					text:          Cow::Borrowed(line),
					was_truncated: false,
				};
			}
			let mut text = String::with_capacity(end + '…'.len_utf8());
			text.push_str(&line[..end]);
			text.push('…');
			return LineTruncationResult { text: Cow::Owned(text), was_truncated: true };
		}
	}

	LineTruncationResult { text: Cow::Borrowed(line), was_truncated: false }
}

/// Retains complete lines from the head within both line and UTF-8 byte limits.
///
/// No partial line is returned. If the first line exceeds the byte budget,
/// `content` is empty and `first_line_exceeds_limit` is set.
pub(crate) fn truncate_head(content: &str, options: TruncationOptions) -> TruncationResult<'_> {
	let total_bytes = content.len();
	let total_lines = content.bytes().filter(|byte| *byte == b'\n').count() + 1;

	if total_lines <= options.max_lines && total_bytes <= options.max_bytes {
		return TruncationResult {
			content,
			truncated: false,
			truncated_by: None,
			total_lines,
			total_bytes,
			output_lines: None,
			output_bytes: None,
			last_line_partial: false,
			first_line_exceeds_limit: false,
		};
	}

	let mut included_lines = 0usize;
	let mut bytes_used = 0usize;
	let mut cut_index = 0usize;
	let mut cursor = 0usize;
	let mut truncated_by = TruncatedBy::Lines;

	while included_lines < options.max_lines {
		let newline = content[cursor..].find('\n').map(|offset| cursor + offset);
		let line_end = newline.unwrap_or(content.len());
		let separator_bytes = usize::from(included_lines > 0);
		let Some(remaining) = options
			.max_bytes
			.checked_sub(bytes_used)
			.and_then(|remaining| remaining.checked_sub(separator_bytes))
		else {
			truncated_by = TruncatedBy::Bytes;
			break;
		};

		let line_bytes = line_end - cursor;
		if line_bytes > remaining {
			if included_lines == 0 {
				return TruncationResult {
					content: "",
					truncated: true,
					truncated_by: Some(TruncatedBy::Bytes),
					total_lines,
					total_bytes,
					output_lines: Some(0),
					output_bytes: Some(0),
					last_line_partial: false,
					first_line_exceeds_limit: true,
				};
			}
			truncated_by = TruncatedBy::Bytes;
			break;
		}

		bytes_used += separator_bytes + line_bytes;
		included_lines += 1;
		cut_index = newline.unwrap_or(content.len());
		let Some(newline) = newline else {
			break;
		};
		cursor = newline + 1;
	}

	if included_lines >= options.max_lines && bytes_used <= options.max_bytes {
		truncated_by = TruncatedBy::Lines;
	}

	TruncationResult {
		content: &content[..cut_index],
		truncated: true,
		truncated_by: Some(truncated_by),
		total_lines,
		total_bytes,
		output_lines: Some(included_lines),
		output_bytes: Some(bytes_used),
		last_line_partial: false,
		first_line_exceeds_limit: false,
	}
}

/// Appends pi's read continuation notice when `truncation` omitted content.
pub(crate) fn append_head_truncation_notice(
	output: &mut String,
	truncation: &TruncationResult<'_>,
	start_line: usize,
	total_file_lines: Option<usize>,
) {
	if !truncation.truncated {
		return;
	}
	let total_file_lines = total_file_lines.unwrap_or(truncation.total_lines);
	let end_line = start_line
		.saturating_add(truncation.shown_lines())
		.saturating_sub(1);
	let next_offset = end_line + 1;
	let _ = write!(
		output,
		"\n\n[Showing lines {start_line}-{end_line} of {total_file_lines}. Use :{next_offset} to \
		 continue]"
	);
}

/// Appends the exact footer used after spilling the complete output to a blob.
pub(crate) fn append_blob_truncation_notice(
	output: &mut String,
	truncation: &TruncationResult<'_>,
	blob_id: &str,
) {
	if !truncation.truncated {
		return;
	}
	append_blob_truncation_notice_counts(
		output,
		u64::try_from(truncation.shown_lines()).unwrap_or(u64::MAX),
		u64::try_from(truncation.total_lines).unwrap_or(u64::MAX),
		blob_id,
	);
}

fn append_blob_truncation_notice_counts(
	output: &mut String,
	shown_lines: u64,
	total_lines: u64,
	blob_id: &str,
) {
	let _ = write!(
		output,
		"\n\n[truncated: {shown_lines} of {total_lines} lines shown; full output in blob {blob_id}]"
	);
}
