//! Compact current-coordinate previews for numbered exact-edit diffs.

use std::{fmt::Write as _, str::Utf8Error};

use omp_core::{Str, StrMut, fmts};
use similar::{Algorithm, DiffOp, capture_diff_slices};

use crate::{format::split_addressable_file_lines, normalize::{normalize_to_lf, strip_bom}};

const DEFAULT_ADDED_RUN_CONTEXT_LINES: usize = 2;
const ELISION: &str = "…";

/// Canonical full numbered diff over exact base and current UTF-8 bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NumberedDiff {
	/// Stable ` N|context`, `-N|old`, and `+N|new` rows.
	pub text:          Str,
	/// Number of current rows added.
	pub added_lines:   usize,
	/// Number of base rows removed.
	pub removed_lines: usize,
}

/// Produces a stable full line diff with one-based original/current coordinates.
///
/// Context and deletion row numbers address `base`; insertion row numbers
/// address `current`. UTF-8 BOM and line-ending conventions are normalized
/// before comparison, while row text remains otherwise exact.
pub fn numbered_diff(base: &[u8], current: &[u8]) -> Result<NumberedDiff, Utf8Error> {
	let base = normalize_to_lf(strip_bom(std::str::from_utf8(base)?).text);
	let current = normalize_to_lf(strip_bom(std::str::from_utf8(current)?).text);
	let base_lines: Vec<&str> = split_addressable_file_lines(&base).collect();
	let current_lines: Vec<&str> = split_addressable_file_lines(&current).collect();
	let mut text = StrMut::with_capacity(base.len().saturating_add(current.len()));
	let mut added_lines = 0;
	let mut removed_lines = 0;
	let mut first = true;

	for operation in capture_diff_slices(Algorithm::Myers, &base_lines, &current_lines) {
		match operation {
			DiffOp::Equal { old_index, new_index: _, len } => {
				for offset in 0..len {
					push_numbered_row(&mut text, &mut first, ' ', old_index + offset + 1, base_lines[old_index + offset]);
				}
			},
			DiffOp::Delete { old_index, old_len, new_index: _ } => {
				for offset in 0..old_len {
					push_numbered_row(&mut text, &mut first, '-', old_index + offset + 1, base_lines[old_index + offset]);
					removed_lines += 1;
				}
			},
			DiffOp::Insert { old_index: _, new_index, new_len } => {
				for offset in 0..new_len {
					push_numbered_row(&mut text, &mut first, '+', new_index + offset + 1, current_lines[new_index + offset]);
					added_lines += 1;
				}
			},
			DiffOp::Replace { old_index, old_len, new_index, new_len } => {
				for offset in 0..old_len {
					push_numbered_row(&mut text, &mut first, '-', old_index + offset + 1, base_lines[old_index + offset]);
					removed_lines += 1;
				}
				for offset in 0..new_len {
					push_numbered_row(&mut text, &mut first, '+', new_index + offset + 1, current_lines[new_index + offset]);
					added_lines += 1;
				}
			},
		}
	}
	Ok(NumberedDiff { text: text.freeze(), added_lines, removed_lines })
}

fn push_numbered_row(output: &mut StrMut, first: &mut bool, kind: char, line: usize, value: &str) {
	if !*first {
		output.push('\n');
	}
	*first = false;
	write!(output, "{kind}{line}|{value}").expect("writing to StrMut cannot fail");
}

/// Rendering controls for a compact numbered diff preview.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactDiffOptions {
	/// Number of lines retained at each edge of a long contiguous added run.
	pub max_added_run_context: usize,
}

impl Default for CompactDiffOptions {
	fn default() -> Self {
		Self { max_added_run_context: DEFAULT_ADDED_RUN_CONTEXT_LINES }
	}
}

/// A compact post-edit-coordinate preview and exact line statistics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactDiffPreview {
	/// Numbered visible current lines with removed rows omitted.
	pub preview:       Str,
	/// Number of added rows in the source diff.
	pub added_lines:   usize,
	/// Number of removed rows in the source diff.
	pub removed_lines: usize,
}

#[derive(Clone, Copy)]
struct ParsedLine<'a> {
	kind:    u8,
	number:  i64,
	content: &'a str,
}

/// Builds a compact preview from `+N|text`, `-N|text`, and ` N|text` rows.
///
/// Added rows already use post-edit numbers. Context rows are renumbered by the
/// running add/remove offset, removed text is omitted, and long added runs
/// elide their middle without losing statistics.
#[must_use]
pub fn build_compact_diff_preview(diff: &str, options: CompactDiffOptions) -> CompactDiffPreview {
	let edge_lines = options.max_added_run_context.max(1);
	let mut formatted = Vec::<Str>::new();
	let mut added_run = Vec::<Str>::new();
	let mut added_lines = 0usize;
	let mut removed_lines = 0usize;

	let flush = |formatted: &mut Vec<Str>, added_run: &mut Vec<Str>| {
		append_added_run(formatted, added_run, edge_lines);
		added_run.clear();
	};

	if !diff.is_empty() {
		for line in diff.split('\n') {
			let Some(parsed) = parse_line(line) else {
				flush(&mut formatted, &mut added_run);
				append_line(&mut formatted, line);
				continue;
			};
			match parsed.kind {
				b'+' => {
					added_lines += 1;
					added_run.push(fmts!("{}:{}", parsed.number, parsed.content));
				},
				b'-' => {
					flush(&mut formatted, &mut added_run);
					removed_lines += 1;
				},
				_ => {
					flush(&mut formatted, &mut added_run);
					let offset = i64::try_from(added_lines).unwrap_or(i64::MAX)
						- i64::try_from(removed_lines).unwrap_or(i64::MAX);
					append_formatted_line(
						&mut formatted,
						fmts!("{}:{}", parsed.number.saturating_add(offset), parsed.content),
					);
				},
			}
		}
	}
	flush(&mut formatted, &mut added_run);
	while formatted.last().is_some_and(|line| is_separator(line)) {
		formatted.pop();
	}
	let mut preview =
		StrMut::with_capacity(formatted.iter().map(Str::len).sum::<usize>() + formatted.len());
	for (index, line) in formatted.iter().enumerate() {
		if index > 0 {
			preview.push('\n');
		}
		preview.push_str(line);
	}
	CompactDiffPreview { preview: preview.freeze(), added_lines, removed_lines }
}

fn parse_line(line: &str) -> Option<ParsedLine<'_>> {
	let kind = *line.as_bytes().first()?;
	if !matches!(kind, b'+' | b'-' | b' ') {
		return None;
	}
	let body = &line[1..];
	let separator = body.find('|')?;
	let number_text = &body[..separator];
	let digit_end = number_text
		.char_indices()
		.take_while(|(index, ch)| ch.is_ascii_digit() || *index == 0 && matches!(ch, '+' | '-'))
		.map(|(index, ch)| index + ch.len_utf8())
		.last()?;
	let number = number_text[..digit_end].parse().ok()?;
	Some(ParsedLine { kind, number, content: &body[separator + 1..] })
}

fn append_added_run(output: &mut Vec<Str>, run: &[Str], edge_lines: usize) {
	if run.is_empty() {
		return;
	}
	let threshold = edge_lines.saturating_mul(2).saturating_add(1);
	if run.len() <= threshold {
		for line in run {
			append_line(output, line);
		}
		return;
	}
	for line in &run[..edge_lines] {
		append_line(output, line);
	}
	append_line(output, ELISION);
	for line in &run[run.len() - edge_lines..] {
		append_line(output, line);
	}

}

fn append_line(output: &mut Vec<Str>, line: &str) {
	let normalized = if matches!(line, "..." | "…" | "+…") {
		ELISION
	} else {
		line
	};
	if is_separator(normalized)
		&& (output.is_empty() || output.last().is_some_and(|prior| is_separator(prior)))
	{
		return;
	}
	output.push(normalized.into());
}

fn append_formatted_line(output: &mut Vec<Str>, line: Str) {
	if is_separator(&line)
		&& (output.is_empty() || output.last().is_some_and(|prior| is_separator(prior)))
	{
		return;
	}
	output.push(line);
}

fn is_separator(line: &str) -> bool {
	line.is_empty() || line == ELISION
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn renumbers_current_rows_and_omits_removed_content() {
		let preview = build_compact_diff_preview(
			" 1|alpha\n-2|beta\n+2|DELTA\n+3|EPSILON\n 3|gamma",
			CompactDiffOptions::default(),
		);
		assert_eq!(preview.preview, "1:alpha\n2:DELTA\n3:EPSILON\n4:gamma");
		assert_eq!(preview.added_lines, 2);
		assert_eq!(preview.removed_lines, 1);
	}

	#[test]
	fn collapses_long_added_runs_to_edges() {
		let diff = (0..7)
			.map(|index| format!("+{}|line {}", 10 + index, index + 1))
			.collect::<Vec<_>>()
			.join("\n");
		let preview = build_compact_diff_preview(&diff, CompactDiffOptions::default());
		assert_eq!(preview.preview, "10:line 1\n11:line 2\n…\n15:line 6\n16:line 7");
		assert_eq!(preview.added_lines, 7);
	}

	#[test]
	fn normalizes_elision_and_gap_separators() {
		let preview = build_compact_diff_preview(
			"\n 1|alpha\n\n-5|beta\n\n...\n…\n 9|gamma\n\n-12|omitted",
			CompactDiffOptions::default(),
		);
		assert_eq!(preview.preview, "1:alpha\n\n8:gamma");
		assert_eq!(preview.removed_lines, 2);
	}

	#[test]
	fn retains_short_added_runs_in_full() {
		let preview = build_compact_diff_preview("+3|x\n+4|y\n+5|z", CompactDiffOptions {
			max_added_run_context: 1,
		});
		assert_eq!(preview.preview, "3:x\n4:y\n5:z");
	}
	#[test]
	fn numbered_replacement_has_true_counts_and_context() {
		let diff = numbered_diff(b"alpha\nold\nomega\n", b"alpha\nnew\nomega\n").unwrap();
		assert_eq!(diff.text, " 1|alpha\n-2|old\n+2|new\n 3|omega");
		assert_eq!(diff.added_lines, 1);
		assert_eq!(diff.removed_lines, 1);
	}

	#[test]
	fn numbered_diff_retains_distant_hunks_and_unicode() {
		let diff = numbered_diff(
			"α\nleft\nmiddle\nright\n終\n".as_bytes(),
			"α\nLEFT\nmiddle\nRIGHT\n終\n".as_bytes(),
		)
		.unwrap();
		assert_eq!(
			diff.text,
			" 1|α\n-2|left\n+2|LEFT\n 3|middle\n-4|right\n+4|RIGHT\n 5|終"
		);
		assert_eq!((diff.added_lines, diff.removed_lines), (2, 2));
	}

	#[test]
	fn numbered_diff_uses_final_formatter_altered_bytes() {
		let diff = numbered_diff(b"fn x(){ }\n", b"fn x() {\n\tbody();\n}\n").unwrap();
		assert_eq!(diff.text, "-1|fn x(){ }\n+1|fn x() {\n+2|\tbody();\n+3|}");
		assert_eq!((diff.added_lines, diff.removed_lines), (3, 1));
	}
}
