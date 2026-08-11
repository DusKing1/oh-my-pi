//! Pure in-memory splitting and section-header recovery for authored patches.

use std::{
	collections::HashMap,
	path::{Component, Path, PathBuf},
};

use omp_core::{Str, StrMut};

use crate::{
	format::{ABORT_MARKER, BEGIN_PATCH_MARKER, END_PATCH_MARKER, HL_FILE_HASH_LENGTH},
	normalize::strip_bom,
	parser::parse_patch,
	tokenizer::{Token, Tokenizer},
	types::{
		BlockMode, Diagnostic, DiagnosticCode, Edit, FileOp, ParseError, ParsedPatch, SplitOptions,
	},
};

#[derive(Debug, Clone)]
struct RawSection {
	path:        Str,
	file_hash:   Option<Str>,
	diff:        Str,
	interleaved: bool,
}

/// One target-file section of an authored patch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchSection {
	/// The normalized authored target path.
	pub path:              Str,
	/// The optional uppercase four-hex snapshot tag.
	pub file_hash:         Option<Str>,
	/// The section body without its header.
	pub diff:              Str,
	/// Whether same-path section merging crossed another target file.
	pub interleaved_merge: bool,
}

impl PatchSection {
	/// Parses this section's operation body.
	pub fn parse(&self) -> Result<ParsedPatch, ParseError> {
		let mut parsed = parse_patch(&self.diff)?;
		if let Some(FileOp::Move { dest }) = &mut parsed.file_op {
			*dest = normalize_hashline_path(dest, None).into();
		}
		if self.interleaved_merge
			&& parsed.edits.iter().any(|edit| {
				matches!(edit, Edit::Cut { .. } | Edit::Paste { .. })
					|| matches!(
						edit,
						Edit::Block { mode: BlockMode::Cut, .. }
							| Edit::Block { mode: BlockMode::PasteAfter, .. }
					) || matches!(edit, Edit::Block { register: Some(_), .. })
			}) {
			return Err(ParseError::new(Diagnostic::error(
				DiagnosticCode::InterleavedClipboard,
				None,
				None,
				"`CUT`/register `PUT` operations cannot be used when same-path sections are \
				 interleaved with another file; keep each file's operations under one header.",
			)));
		}
		Ok(parsed)
	}

	/// Returns whether any parsed edit anchors to concrete source content.
	pub fn has_anchor_scoped_edit(&self) -> Result<bool, ParseError> {
		Ok(self.parse()?.edits.iter().any(|edit| match edit {
			Edit::Delete { .. } | Edit::Cut { .. } | Edit::Block { .. } => true,
			Edit::Paste { at, .. } => match at {
				crate::types::PasteTarget::Span { .. } => true,
				crate::types::PasteTarget::Gap { cursor } => matches!(
					cursor,
					crate::types::Cursor::BeforeAnchor { .. } | crate::types::Cursor::AfterAnchor { .. }
				),
			},
			Edit::Insert { cursor, .. } => matches!(
				cursor,
				crate::types::Cursor::BeforeAnchor { .. } | crate::types::Cursor::AfterAnchor { .. }
			),
		}))
	}

	/// Collects all concrete anchor lines in ascending deduplicated order.
	pub fn collect_anchor_lines(&self) -> Result<Vec<usize>, ParseError> {
		let mut lines = std::collections::BTreeSet::new();
		for edit in self.parse()?.edits {
			match edit {
				Edit::Delete { anchor, .. } | Edit::Block { anchor, .. } => {
					lines.insert(anchor.line);
				},
				Edit::Cut { range, .. }
				| Edit::Paste { at: crate::types::PasteTarget::Span { range }, .. } => {
					lines.extend(range.start.line..=range.end.line);
				},
				Edit::Paste { at: crate::types::PasteTarget::Gap { cursor }, .. }
				| Edit::Insert { cursor, .. } => match cursor {
					crate::types::Cursor::BeforeAnchor { anchor }
					| crate::types::Cursor::AfterAnchor { anchor } => {
						lines.insert(anchor.line);
					},
					crate::types::Cursor::Bof | crate::types::Cursor::Eof => {},
				},
			}
		}
		Ok(lines.into_iter().collect())
	}

	/// Returns the section's parsed whole-file operation.
	pub fn file_op(&self) -> Result<Option<FileOp>, ParseError> {
		Ok(self.parse()?.file_op)
	}
}

/// A fully split hashline patch containing zero or more target sections.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Patch {
	/// Sections ordered by their first authored occurrence.
	pub sections: Vec<PatchSection>,
}

impl Patch {
	/// Splits authored input into normalized, same-path-coalesced sections.
	pub fn parse(input: &str, options: &SplitOptions) -> Result<Self, ParseError> {
		let sections = merge_same_path_sections(split_raw_sections(input, options)?)?
			.into_iter()
			.map(|raw| PatchSection {
				path:              raw.path,
				file_hash:         raw.file_hash,
				diff:              raw.diff,
				interleaved_merge: raw.interleaved,
			})
			.collect();
		Ok(Self { sections })
	}

	/// Splits input using no cwd or fallback-path hints.
	pub fn parse_default(input: &str) -> Result<Self, ParseError> {
		Self::parse(input, &SplitOptions::default())
	}

	/// Parses exactly the first non-empty section or reports an empty patch.
	pub fn parse_single(input: &str, options: &SplitOptions) -> Result<PatchSection, ParseError> {
		Self::parse(input, options)?
			.sections
			.into_iter()
			.next()
			.ok_or_else(|| {
				ParseError::new(Diagnostic::error(
					DiagnosticCode::InvalidSectionHeader,
					None,
					None,
					"Patch input did not produce any sections.",
				))
			})
	}
}

/// Returns whether any physical line is a recognizable hashline operation.
pub fn contains_recognizable_hashline_operations(input: &str) -> bool {
	let tokenizer = Tokenizer::new();
	input.lines().any(|line| tokenizer.is_op(line))
}

/// Splits an authored patch with optional lexical path hints.
pub fn split_patch_input(input: &str, options: &SplitOptions) -> Result<Patch, ParseError> {
	Patch::parse(input, options)
}

fn split_raw_sections(input: &str, options: &SplitOptions) -> Result<Vec<RawSection>, ParseError> {
	let fallback = normalize_fallback_input(input, options)?;
	let stripped = strip_leading_layout(&fallback);
	let lines: Vec<_> = stripped
		.split('\n')
		.map(|line| line.strip_suffix('\r').unwrap_or(line))
		.collect();
	let first = lines.first().copied().unwrap_or("");
	if parse_header_line(first, 1, options)?.is_none() {
		let trimmed = first.trim_end();
		let message: Str = if trimmed.starts_with("@@") {
			"unified-diff hunk header is not valid in hashline. File sections start with \
			 `[path#HASH]`; use `PUT`, `CUT`, `REM`, or `MV`."
				.into()
		} else {
			format!(
				"input must begin with \"[PATH#HASH]\" on the first non-blank line for anchored \
				 edits; got {:?}. Example: \"[src/foo.ts#1A2B]\" then edit ops.",
				first.chars().take(120).collect::<String>()
			)
			.into()
		};
		return Err(ParseError::new(Diagnostic::error(
			DiagnosticCode::InvalidSectionHeader,
			Some(1),
			None,
			message,
		)));
	}

	let tokenizer = Tokenizer::new();
	let mut sections = Vec::new();
	let mut current: Option<RawSection> = None;
	let mut body = Vec::<&str>::new();

	let flush =
		|current: &mut Option<RawSection>, body: &mut Vec<&str>, sections: &mut Vec<RawSection>| {
			let Some(mut section) = current.take() else {
				return;
			};
			if body.iter().any(|line| !line.trim().is_empty()) {
				section.diff = body.join("\n").into();
				sections.push(section);
			}
			body.clear();
		};

	for (index, line) in lines.into_iter().enumerate() {
		let line_num = index + 1;
		let trimmed = line.trim_end();
		if trimmed == END_PATCH_MARKER || trimmed == ABORT_MARKER {
			break;
		}
		if trimmed == BEGIN_PATCH_MARKER {
			continue;
		}
		if trimmed.starts_with('[') {
			if let Some(header) = parse_header_line(line, line_num, options)? {
				flush(&mut current, &mut body, &mut sections);
				current = Some(header);
				continue;
			}
		} else if matches!(tokenizer.tokenize(line, line_num), Token::Header { .. }) {
			unreachable!("strict headers always begin with `[`")
		}
		body.push(line);
	}
	flush(&mut current, &mut body, &mut sections);
	Ok(sections)
}

fn normalize_fallback_input(input: &str, options: &SplitOptions) -> Result<String, ParseError> {
	let stripped = strip_bom(input).text;
	for (index, line) in stripped.lines().enumerate() {
		if parse_header_line(line, index + 1, options)?.is_some() {
			return Ok(input.to_owned());
		}
	}
	let Some(path) = options.path.as_deref() else {
		return Ok(input.to_owned());
	};
	if !contains_recognizable_hashline_operations(input) {
		return Ok(input.to_owned());
	}
	let path = normalize_hashline_path(path, options.cwd.as_deref());
	if path.is_empty() {
		return Ok(input.to_owned());
	}
	Ok(format!("[{path}]\n{input}"))
}

fn strip_leading_layout(input: &str) -> String {
	let stripped = strip_bom(input).text;
	let mut lines: Vec<_> = stripped.split('\n').collect();
	while lines.first().is_some_and(|line| {
		let line = line.strip_suffix('\r').unwrap_or(line);
		line.trim().is_empty() || line.trim_end() == BEGIN_PATCH_MARKER
	}) {
		lines.remove(0);
	}
	lines.join("\n")
}

fn parse_header_line(
	line: &str,
	line_num: usize,
	options: &SplitOptions,
) -> Result<Option<RawSection>, ParseError> {
	let trimmed = line.trim_end();
	if !trimmed.starts_with('[') {
		return Ok(None);
	}
	let tokenizer = Tokenizer::new();
	if let Token::Header { path, file_hash, .. } = tokenizer.tokenize(trimmed, line_num) {
		let path = normalize_hashline_path(&path, options.cwd.as_deref());
		if path.is_empty() {
			return Err(invalid_header(line_num, "Input header `[]` is empty; provide a file path."));
		}
		return Ok(Some(RawSection {
			path: path.into(),
			file_hash,
			diff: Str::default(),
			interleaved: false,
		}));
	}
	if let Some(recovered) = recover_header(trimmed, options.cwd.as_deref()) {
		return Ok(Some(recovered));
	}
	Err(invalid_header(
		line_num,
		format!(
			"Input header must be `[PATH]` or `[PATH#TAG]` with a {HL_FILE_HASH_LENGTH}-hex \
			 content-hash tag; got {trimmed:?}."
		),
	))
}

fn recover_header(line: &str, cwd: Option<&str>) -> Option<RawSection> {
	let body = line.strip_prefix('[')?.strip_suffix(']')?.trim();
	let body = strip_apply_patch_path_noise(body);
	if body.is_empty() {
		return None;
	}
	let (path, file_hash) = if let Some((path, tag)) = body.rsplit_once('#') {
		if path.contains('#')
			|| tag.len() != HL_FILE_HASH_LENGTH
			|| !tag.bytes().all(|byte| byte.is_ascii_hexdigit())
		{
			return None;
		}
		(path.trim_end(), Some(Str::from(tag.to_ascii_uppercase())))
	} else {
		if body.contains('#') {
			return None;
		}
		(body.trim_end(), None)
	};
	let path = normalize_hashline_path(path, cwd);
	if path.is_empty() {
		return None;
	}
	Some(RawSection { path: path.into(), file_hash, diff: Str::default(), interleaved: false })
}

fn strip_apply_patch_path_noise(text: &str) -> &str {
	let mut rest = text.trim_start();
	let star_count = rest.bytes().take_while(|byte| *byte == b'*').count().min(3);
	rest = rest[star_count..].trim_start();
	if let Some(colon) = rest.find(':') {
		let keyword: String = rest[..colon]
			.chars()
			.filter(|ch| ch.is_ascii_alphabetic())
			.flat_map(char::to_lowercase)
			.collect();
		if matches!(
			keyword.as_str(),
			"update" | "updatefile" | "add" | "addfile" | "delete" | "deletefile" | "move" | "moveto"
		) {
			rest = rest[colon + 1..].trim_start();
		}
	}
	let star_count = rest.bytes().take_while(|byte| *byte == b'*').count().min(3);
	rest[star_count..].trim_start()
}

fn normalize_hashline_path(raw: &str, cwd: Option<&str>) -> String {
	let trimmed = raw.trim();
	let unquoted = if trimmed.len() >= 2 {
		let bytes = trimmed.as_bytes();
		if matches!(bytes[0], b'\'' | b'"') && bytes[trimmed.len() - 1] == bytes[0] {
			&trimmed[1..trimmed.len() - 1]
		} else {
			trimmed
		}
	} else {
		trimmed
	};
	let cleaned = strip_apply_patch_path_noise(unquoted);
	let Some(cwd) = cwd else {
		return cleaned.to_owned();
	};
	if !Path::new(cleaned).is_absolute() {
		return cleaned.to_owned();
	}
	let path = lexical_normalize(Path::new(cleaned));
	let cwd = lexical_normalize(Path::new(cwd));
	if let Ok(relative) = path.strip_prefix(&cwd) {
		let text = path_to_slashes(relative);
		return if text.is_empty() {
			".".to_owned()
		} else {
			text
		};
	}
	cleaned.to_owned()
}

fn lexical_normalize(path: &Path) -> PathBuf {
	let mut normalized = PathBuf::new();
	for component in path.components() {
		match component {
			Component::CurDir => {},
			Component::ParentDir if normalized.file_name().is_some() => {
				normalized.pop();
			},
			Component::ParentDir if normalized.has_root() => {},
			component => normalized.push(component.as_os_str()),
		}
	}
	normalized
}

fn path_to_slashes(path: &Path) -> String {
	path
		.to_string_lossy()
		.replace(std::path::MAIN_SEPARATOR, "/")
}

fn invalid_header(line_num: usize, message: impl Into<Str>) -> ParseError {
	ParseError::new(Diagnostic::error(
		DiagnosticCode::InvalidSectionHeader,
		Some(line_num),
		None,
		message,
	))
}

fn merge_same_path_sections(sections: Vec<RawSection>) -> Result<Vec<RawSection>, ParseError> {
	#[derive(Debug)]
	struct Entry {
		index:       usize,
		file_hash:   Option<Str>,
		diffs:       Vec<Str>,
		interleaved: bool,
	}
	let mut entries = HashMap::<Str, Entry>::new();
	let mut order = Vec::<Str>::new();
	let mut previous_path: Option<Str> = None;
	for section in sections {
		if let Some(existing) = entries.get_mut(&section.path) {
			if let (Some(left), Some(right)) = (&existing.file_hash, &section.file_hash)
				&& left != right
			{
				return Err(ParseError::new(Diagnostic::error(
					DiagnosticCode::ConflictingSnapshotTags,
					None,
					None,
					format!(
						"Conflicting hashline snapshot tags for {}: #{left} and #{right}. Re-read the \
						 file and retry with one current header.",
						section.path
					),
				)));
			}
			if existing.file_hash.is_none() {
				existing.file_hash = section.file_hash;
			}
			if previous_path.as_ref() != Some(&section.path) {
				existing.interleaved = true;
			}
			existing.diffs.push(section.diff);
			previous_path = Some(section.path);
			continue;
		}
		let index = order.len();
		order.push(section.path.clone());
		entries.insert(section.path.clone(), Entry {
			index,
			file_hash: section.file_hash,
			diffs: vec![section.diff],
			interleaved: false,
		});
		previous_path = Some(section.path);
	}
	let mut merged = Vec::with_capacity(order.len());
	for path in order {
		let entry = entries.remove(&path).unwrap();
		debug_assert_eq!(entry.index, merged.len());
		let mut diff =
			StrMut::with_capacity(entry.diffs.iter().map(Str::len).sum::<usize>() + entry.diffs.len());
		for (index, section) in entry.diffs.iter().enumerate() {
			if index > 0 {
				diff.push('\n');
			}
			diff.push_str(section);
		}
		merged.push(RawSection {
			path,
			file_hash: entry.file_hash,
			diff: diff.freeze(),
			interleaved: entry.interleaved,
		});
	}
	Ok(merged)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn splits_envelopes_sections_and_recovers_noisy_headers() {
		let patch = Patch::parse_default(
			"\u{feff}\n*** Begin Patch\n[*** Update File: dir with spaces/a.rs#1a2b]\nPUT \
			 1:\n+A\n[b.rs]\nCUT 2\n*** End Patch\n[c.rs]\nREM",
		)
		.unwrap();
		assert_eq!(patch.sections.len(), 2);
		assert_eq!(patch.sections[0].path, "dir with spaces/a.rs");
		assert_eq!(patch.sections[0].file_hash.as_deref(), Some("1A2B"));
		assert_eq!(patch.sections[1].diff, "CUT 2");
	}

	#[test]
	fn supplies_headerless_fallback_only_for_recognizable_ops() {
		let options = SplitOptions { cwd: None, path: Some("a.rs".into()) };
		assert_eq!(Patch::parse_single("PUT <1:\n+x", &options).unwrap().path, "a.rs");
		assert_eq!(
			Patch::parse("plain text", &options)
				.unwrap_err()
				.diagnostic
				.code,
			DiagnosticCode::InvalidSectionHeader
		);
	}

	#[test]
	fn merges_same_paths_and_rejects_conflicting_tags() {
		let patch =
			Patch::parse_default("[a.rs#1A2B]\nCUT 1\n[b.rs]\nREM\n[a.rs#1A2B]\nCUT 2").unwrap();
		assert_eq!(patch.sections.len(), 2);
		assert!(patch.sections[0].interleaved_merge);
		assert_eq!(patch.sections[0].diff, "CUT 1\nCUT 2");
		assert_eq!(
			Patch::parse_default("[a.rs#1A2B]\nCUT 1\n[a.rs#3C4D]\nCUT 2")
				.unwrap_err()
				.diagnostic
				.code,
			DiagnosticCode::ConflictingSnapshotTags
		);
	}

	#[test]
	fn rejects_malformed_snapshot_headers() {
		for header in ["[a.rs#1A2]", "[a.rs#1A2G]", "[a.rs#1A2B5]", "[a.rs#1A2B:9]"] {
			assert_eq!(
				Patch::parse_default(&format!("{header}\nCUT 1"))
					.unwrap_err()
					.diagnostic
					.code,
				DiagnosticCode::InvalidSectionHeader
			);
		}
	}
}
