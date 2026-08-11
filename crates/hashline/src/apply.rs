//! Pure application of parsed hashline edits to exact UTF-8 bytes.

use std::{
	collections::{BTreeMap, BTreeSet},
	error::Error,
	fmt,
};

use bytes::Bytes;
use omp_core::{Str, fmts};
use similar::{Algorithm, DiffOp, capture_diff_slices};
use smallvec::SmallVec;

use crate::{
	block::{BlockError, BlockLowering, UnresolvedBlockMode, resolve_block_edits},
	clipboard::{Clipboard, ClipboardError, EmptyPasteMode, resolve_clipboard_edits},
	normalize::{detect_line_ending, normalize_to_lf, restore_bom, restore_line_endings, strip_bom},
	types::{Cursor, Edit, InsertMode, ParsedPatch},
};

/// One canonical replacement in coordinates of the exact input bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteEdit {
	/// Inclusive byte start in the input.
	pub start:       usize,
	/// Exclusive byte end in the input.
	pub end:         usize,
	/// Exact replacement bytes.
	pub replacement: Bytes,
}

/// Strict final application or streaming-tolerant preview behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyMode {
	/// Reject every unresolved or structurally invalid edit.
	Strict,
	/// Drop incomplete block and empty-register edits while retaining valid
	/// work.
	Partial,
}

/// Application options.
#[derive(Debug, Clone, Copy)]
pub struct ApplyOptions<'a> {
	/// Strict or streaming-tolerant behavior.
	pub mode: ApplyMode,
	/// Optional path used to infer the syntax language.
	pub path: Option<&'a str>,
}
impl Default for ApplyOptions<'_> {
	fn default() -> Self {
		Self { mode: ApplyMode::Strict, path: None }
	}
}

/// Fully materialized application output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyResult {
	/// Final bytes with the original BOM and line-ending convention restored.
	pub bytes:              Bytes,
	/// Canonical non-overlapping edits in exact input-byte coordinates.
	pub edits:              Vec<ByteEdit>,
	/// First changed model-facing line, when any.
	pub first_changed_line: Option<usize>,
	/// Non-fatal application diagnostics.
	pub warnings:           Vec<Str>,
	/// Syntax-aware block resolutions.
	pub block_resolutions:  Vec<crate::types::BlockResolution>,
}

/// Structural application failure.
#[derive(Debug)]
pub enum ApplyError {
	/// Input was not exact UTF-8.
	InvalidUtf8(std::str::Utf8Error),
	/// A syntax-aware block could not be lowered.
	Block(BlockError),
	/// A clipboard operation was invalid.
	Clipboard(ClipboardError),
	/// An original-coordinate edit was invalid.
	InvalidEdit(Str),
}
impl fmt::Display for ApplyError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::InvalidUtf8(e) => write!(f, "source is not UTF-8: {e}"),
			Self::Block(e) => write!(f, "{e}"),
			Self::Clipboard(e) => write!(f, "{e}"),
			Self::InvalidEdit(e) => write!(f, "{e}"),
		}
	}
}
impl Error for ApplyError {}
impl From<BlockError> for ApplyError {
	fn from(value: BlockError) -> Self {
		Self::Block(value)
	}
}
impl From<ClipboardError> for ApplyError {
	fn from(value: ClipboardError) -> Self {
		Self::Clipboard(value)
	}
}

const fn anchor_line(edit: &Edit) -> Option<usize> {
	match edit {
		Edit::Delete { anchor, .. } => Some(anchor.line),
		Edit::Insert {
			cursor: Cursor::BeforeAnchor { anchor } | Cursor::AfterAnchor { anchor },
			..
		} => Some(anchor.line),
		_ => None,
	}
}

fn closer(text: &str) -> bool {
	let t = text.trim();
	let t = t
		.strip_suffix(';')
		.or_else(|| t.strip_suffix(','))
		.unwrap_or(t);
	!t.is_empty() && t.bytes().all(|b| matches!(b, b')' | b']' | b'}'))
}
fn indent(text: &str) -> &str {
	&text[..text.len() - text.trim_start_matches([' ', '\t']).len()]
}

fn repair_replacement_indentation(
	edits: &mut [Edit],
	lines: &[Str],
	warnings: &mut Vec<Str>,
) {
	let mut at = 0;
	let mut repaired = false;
	while at < edits.len() {
		let Edit::Insert {
			cursor: Cursor::BeforeAnchor { anchor },
			line_num,
			mode: InsertMode::Replacement,
			..
		} = &edits[at]
		else {
			at += 1;
			continue;
		};
		let anchor_line = anchor.line;
		let op_line = *line_num;
		let insert_start = at;
		while at < edits.len()
			&& matches!(&edits[at], Edit::Insert { cursor: Cursor::BeforeAnchor { anchor }, line_num, mode: InsertMode::Replacement, .. } if anchor.line == anchor_line && *line_num == op_line)
		{
			at += 1;
		}
		let insert_end = at;
		let delete_start = at;
		let mut expected = anchor_line;
		while at < edits.len()
			&& matches!(&edits[at], Edit::Delete { anchor, line_num, .. } if anchor.line == expected && *line_num == op_line)
		{
			expected += 1;
			at += 1;
		}
		let count = at - delete_start;
		if count == 0 || count != insert_end - insert_start || anchor_line < 2 {
			continue;
		}
		let preceding = &lines[anchor_line - 2];
		let source_first = &lines[anchor_line - 1];
		let Edit::Insert { text: payload_first, .. } = &edits[insert_start] else {
			continue;
		};
		if !preceding.trim_end().ends_with('{')
			|| !indent(source_first).starts_with(indent(preceding))
			|| indent(source_first) == indent(preceding)
			|| (indent(payload_first).starts_with(indent(preceding))
				&& indent(payload_first) != indent(preceding))
		{
			continue;
		}
		let mut shift: Option<String> = None;
		let mut matches = 0;
		let mut consistent = true;
		for offset in 0..count {
			let source = &lines[anchor_line - 1 + offset];
			let Edit::Insert { text: payload, .. } = &edits[insert_start + offset] else {
				continue;
			};
			if source.trim().is_empty() || source.trim_start() != payload.trim_start() {
				continue;
			}
			let si = indent(source);
			let pi = indent(payload);
			if !si.ends_with(pi) {
				consistent = false;
				break;
			}
			let candidate = &si[..si.len() - pi.len()];
			match &shift {
				None => shift = Some(candidate.to_owned()),
				Some(old) if old == candidate => {},
				_ => {
					consistent = false;
					break;
				},
			}
			matches += 1;
		}
		let Some(shift) =
			shift.filter(|s| !s.is_empty() && consistent && matches >= 2 && matches * 2 > count)
		else {
			continue;
		};
		for edit in &mut edits[insert_start..insert_end] {
			if let Edit::Insert { text, .. } = edit
				&& !text.trim().is_empty()
			{
				*text = Str::new(format!("{shift}{text}"));
			}
		}
		repaired = true;
	}
	if repaired {
		warnings.push(Str::new(
			"Auto-indented a replacement body to preserve its surrounding block depth",
		));
	}
}

fn repair_landings(edits: &mut [Edit], lines: &[Str], warnings: &mut Vec<Str>) {
	let mut groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
	let targeted: BTreeSet<usize> = edits.iter().filter_map(anchor_line).collect();
	for (i, edit) in edits.iter().enumerate() {
		if let Edit::Insert {
			cursor: Cursor::AfterAnchor { anchor },
			line_num,
			mode: InsertMode::Literal,
			..
		} = edit
		{
			groups.entry((anchor.line, *line_num)).or_default().push(i);
		}
	}
	for ((anchor, _), members) in groups {
		let mut target: Option<&str> = None;
		let mut comparable = true;
		for &i in &members {
			let Edit::Insert { text, .. } = &edits[i] else {
				continue;
			};
			if text.trim().is_empty() || closer(text) {
				continue;
			}
			let current = indent(text);
			target = match target {
				None => Some(current),
				Some(old) if current.starts_with(old) => Some(old),
				Some(old) if old.starts_with(current) => Some(current),
				_ => {
					comparable = false;
					None
				},
			};
			if !comparable {
				break;
			}
		}
		let Some(target) = target.filter(|_| comparable) else {
			continue;
		};
		let Some(anchor_text) = lines.get(anchor - 1) else {
			continue;
		};
		if !indent(anchor_text).starts_with(target) || indent(anchor_text) == target {
			continue;
		}
		let mut landing = anchor;
		let mut crossed = 0;
		for line in anchor + 1..=lines.len() {
			let text = &lines[line - 1];
			if text.trim().is_empty() {
				continue;
			}
			if !closer(text) || !indent(text).starts_with(target) || targeted.contains(&line) {
				break;
			}
			landing = line;
			crossed += 1;
			if indent(text) == target {
				break;
			}
		}
		if landing != anchor {
			for i in members {
				if let Edit::Insert { cursor, .. } = &mut edits[i] {
					*cursor = Cursor::AfterAnchor { anchor: crate::types::Anchor { line: landing } };
				}
			}
			warnings.push(Str::new(format!(
				"after-line insertion shifted from line {anchor} to {landing} across {crossed} closer \
				 line(s)"
			)));
		}
	}
}

fn validate(edits: &[Edit], lines: &[Str]) -> Result<(), ApplyError> {
	let mut deleted = BTreeSet::new();
	let phantom = lines.len() > 1 && lines.last().is_some_and(|line| line.is_empty());
	for edit in edits {
		match edit {
			Edit::Block { .. } | Edit::Cut { .. } | Edit::Paste { .. } => {
				return Err(ApplyError::InvalidEdit(Str::new(
					"unresolved high-level edit reached materialization",
				)));
			},
			Edit::Delete { anchor, .. } => {
				if anchor.line < 1 || anchor.line > lines.len() {
					return Err(ApplyError::InvalidEdit(fmts!(
						"line {} does not exist (file has {} lines)",
						anchor.line,
						lines.len()
					)));
				}
				if phantom && anchor.line == lines.len() {
					continue;
				}
				if !deleted.insert(anchor.line) {
					return Err(ApplyError::InvalidEdit(fmts!(
						"overlapping delete at line {}",
						anchor.line
					)));
				}
			},
			Edit::Insert {
				cursor: Cursor::BeforeAnchor { anchor } | Cursor::AfterAnchor { anchor },
				..
			} if anchor.line < 1 || anchor.line > lines.len() => {
				return Err(ApplyError::InvalidEdit(fmts!(
					"line {} does not exist (file has {} lines)",
					anchor.line,
					lines.len()
				)));
			},
			Edit::Insert { .. } => {},
		}
	}
	Ok(())
}

fn repair_boundaries(
	edits: &mut Vec<Edit>,
	lines: &[Str],
	path: Option<&str>,
	warnings: &mut Vec<Str>,
) {
	let mut original =
		String::with_capacity(lines.iter().map(Str::len).sum::<usize>() + lines.len());
	for (index, line) in lines.iter().enumerate() {
		if index > 0 {
			original.push('\n');
		}
		original.push_str(line);
	}
	if !crate::syntax::parses_cleanly(path, &original) {
		return;
	}
	let (authored, _) = materialize(lines, edits);
	if crate::syntax::parses_cleanly(path, &authored) {
		return;
	}
	let mut proposed = edits.clone();
	let mut changed = false;
	let op_lines: BTreeSet<usize> = proposed
		.iter()
		.filter_map(|edit| match edit {
			Edit::Insert { mode: InsertMode::Replacement, line_num, .. } => Some(*line_num),
			_ => None,
		})
		.collect();
	for op_line in op_lines {
		let deleted: SmallVec<(usize, usize), 8> = proposed
			.iter()
			.enumerate()
			.filter_map(|(i, edit)| match edit {
				Edit::Delete { anchor, line_num, .. } if *line_num == op_line => Some((i, anchor.line)),
				_ => None,
			})
			.collect();
		let inserted: SmallVec<usize, 8> = proposed
			.iter()
			.enumerate()
			.filter_map(|(i, edit)| match edit {
				Edit::Insert { mode: InsertMode::Replacement, line_num, .. }
					if *line_num == op_line =>
				{
					Some(i)
				},
				_ => None,
			})
			.collect();
		let (Some(&(_, start)), Some(&(last_delete_index, end))) = (deleted.first(), deleted.last())
		else {
			continue;
		};
		if let Some(&last_insert_index) = inserted.last() {
			let Edit::Insert { text, .. } = &proposed[last_insert_index] else {
				continue;
			};
			if lines.get(end).is_some_and(|next| next == text) && closer(text) {
				proposed.remove(last_insert_index);
				changed = true;
				continue;
			}
		}
		if lines.get(end - 1).is_some_and(|line| closer(line)) && inserted.iter().any(|&i| matches!(&proposed[i], Edit::Insert { text, .. } if !text.trim().is_empty() && indent(text).len() > indent(&lines[end - 1]).len())) {
            proposed.remove(last_delete_index);
            changed = true;
        } else if lines.get(start - 1).is_some_and(|line| closer(line)) && inserted.iter().any(|&i| matches!(&proposed[i], Edit::Insert { text, .. } if !text.trim().is_empty() && indent(text).len() <= indent(&lines[start - 1]).len()))
            && let Some((first_delete_index, _)) = deleted.first() { proposed.remove(*first_delete_index); changed = true; }
	}
	if changed {
		let (candidate, _) = materialize(lines, &proposed);
		if crate::syntax::parses_cleanly(path, &candidate) {
			*edits = proposed;
			warnings.push(Str::new(
				"Repaired replacement boundaries after a syntax-verified delimiter-balance check",
			));
		}
	}
}

fn materialize(lines: &[Str], edits: &[Edit]) -> (String, Option<usize>) {
	let phantom = lines.len() > 1 && lines.last().is_some_and(|line| line.is_empty());
	let mut buckets: BTreeMap<usize, Vec<(usize, &Edit)>> = BTreeMap::new();
	let mut bof = Vec::new();
	let mut eof = Vec::new();
	for (order, edit) in edits.iter().enumerate() {
		match edit {
			Edit::Insert { cursor: Cursor::Bof, text, .. } => bof.push(text.as_str()),
			Edit::Insert { cursor: Cursor::Eof, text, .. } => eof.push(text.as_str()),
			_ => {
				if let Some(line) = anchor_line(edit) {
					buckets.entry(line).or_default().push((order, edit));
				}
			},
		}
	}
	let mut out = lines.to_vec();
	let mut first = None;
	for (line, bucket) in buckets.into_iter().rev() {
		let mut before = Vec::new();
		let mut replacement = Vec::new();
		let mut after = Vec::new();
		let mut delete = false;
		for (_, edit) in bucket {
			match edit {
				Edit::Delete { .. } => {
					if !(phantom && line == lines.len()) {
						delete = true;
					}
				},
				Edit::Insert { text, mode: InsertMode::Replacement, .. } => {
					replacement.push(text.clone());
				},
				Edit::Insert { cursor: Cursor::AfterAnchor { .. }, text, .. } => {
					after.push(text.clone());
				},
				Edit::Insert { text, .. } => before.push(text.clone()),
				_ => {},
			}
		}
		if before.is_empty() && replacement.is_empty() && after.is_empty() && !delete {
			continue;
		}
		let idx = line - 1;
		let mut rows = before;
		rows.extend(replacement);
		if !delete {
			rows.push(out[idx].clone());
		}
		rows.extend(after);
		out.splice(idx..=idx, rows);
		first = Some(first.map_or(line, |old: usize| old.min(line)));
	}
	if !bof.is_empty() {
		let rows = bof.into_iter().map(Str::new).collect::<Vec<_>>();
		if out.len() == 1 && out[0].is_empty() {
			out = rows;
		} else {
			out.splice(0..0, rows);
		}
		first = Some(1);
	}
	if !eof.is_empty() {
		let at = if out.last().is_some_and(|line| line.is_empty()) {
			out.len() - 1
		} else {
			out.len()
		};
		out.splice(at..at, eof.into_iter().map(Str::new));
		first = Some(first.map_or(at + 1, |old| old.min(at + 1)));
	}
	(
		out.iter()
			.map(AsRef::as_ref)
			.collect::<Vec<&str>>()
			.join("\n"),
		first,
	)
}

fn canonical_edit(base: &[u8], final_bytes: &[u8]) -> Vec<ByteEdit> {
	capture_diff_slices(Algorithm::Myers, base, final_bytes)
		.into_iter()
		.filter_map(|operation| {
			let (start, old_len, new_start, new_len) = match operation {
				DiffOp::Equal { .. } => return None,
				DiffOp::Delete { old_index, old_len, new_index } => (old_index, old_len, new_index, 0),
				DiffOp::Insert { old_index, new_index, new_len } => (old_index, 0, new_index, new_len),
				DiffOp::Replace { old_index, old_len, new_index, new_len } => {
					(old_index, old_len, new_index, new_len)
				},
			};
			Some(ByteEdit {
				start,
				end: start + old_len,
				replacement: Bytes::copy_from_slice(&final_bytes[new_start..new_start + new_len]),
			})
		})
		.collect()
}

/// Applies parsed edits to exact UTF-8 source bytes without filesystem access.
pub fn apply_parsed_patch(
	source: Bytes,
	patch: &ParsedPatch,
	clipboard: &mut Clipboard,
	options: ApplyOptions<'_>,
) -> Result<ApplyResult, ApplyError> {
	let exact = std::str::from_utf8(&source).map_err(ApplyError::InvalidUtf8)?;
	let ending = detect_line_ending(exact);
	let bom = strip_bom(exact);
	let normalized = normalize_to_lf(bom.text);
	let lines: Vec<Str> = normalized.split('\n').map(Str::new).collect();
	let BlockLowering { edits: blocked, resolutions, mut warnings } = resolve_block_edits(
		&patch.edits,
		&normalized,
		options.path,
		if options.mode == ApplyMode::Strict {
			UnresolvedBlockMode::Strict
		} else {
			UnresolvedBlockMode::Drop
		},
	)?;
	let clip = resolve_clipboard_edits(
		&blocked,
		&lines,
		clipboard,
		if options.mode == ApplyMode::Strict {
			EmptyPasteMode::Strict
		} else {
			EmptyPasteMode::Drop
		},
	)?;
	warnings.extend(clip.warnings);
	let mut concrete = clip.edits;
	validate(&concrete, &lines)?;
	repair_replacement_indentation(&mut concrete, &lines, &mut warnings);
	repair_boundaries(&mut concrete, &lines, options.path, &mut warnings);
	repair_landings(&mut concrete, &lines, &mut warnings);
	let (model, first_changed_line) = materialize(&lines, &concrete);
	let restored_endings = restore_line_endings(&model, ending);
	let restored = restore_bom(&restored_endings, bom.had_bom);
	let bytes = Bytes::copy_from_slice(restored.as_bytes());
	let edits = canonical_edit(&source, &bytes);
	Ok(ApplyResult { bytes, edits, first_changed_line, warnings, block_resolutions: resolutions })
}

/// Applies already-parsed edits using a private clipboard.
pub fn apply_edits(
	source: Bytes,
	edits: &[Edit],
	options: ApplyOptions<'_>,
) -> Result<ApplyResult, ApplyError> {
	let patch =
		ParsedPatch { edits: edits.to_vec(), file_op: None, diagnostics: Vec::new() };
	apply_parsed_patch(source, &patch, &mut Clipboard::default(), options)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn canonical_diff_keeps_distant_changes_separate() {
		let base = b"first\nunchanged middle\nlast\n";
		let proposed = b"FIRST\nunchanged middle\nLAST\n";
		let edits = canonical_edit(base, proposed);
		assert_eq!(edits.len(), 2);
		assert!(edits[0].end <= b"first\n".len());
		assert!(edits[1].start >= b"first\nunchanged middle\n".len());

		let mut replayed = Vec::new();
		let mut cursor = 0;
		for edit in &edits {
			replayed.extend_from_slice(&base[cursor..edit.start]);
			replayed.extend_from_slice(&edit.replacement);
			cursor = edit.end;
		}
		replayed.extend_from_slice(&base[cursor..]);
		assert_eq!(replayed, proposed);
		assert_eq!(
			&replayed[b"FIRST\n".len()..b"FIRST\nunchanged middle\n".len()],
			b"unchanged middle\n"
		);
	}
}
