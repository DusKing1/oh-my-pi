//! In-memory clipboard lowering for `CUT` and register-backed `PUT` edits.

use std::{collections::BTreeMap, error::Error, fmt};

use omp_core::{Str, fmts};

use crate::types::{Anchor, Cursor, Edit, InsertMode, ParsedRange, PasteTarget};

/// Clipboard state shared by sections in one transaction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Clipboard {
	anonymous:              Option<Vec<Str>>,
	named:                  BTreeMap<Str, Vec<Str>>,
	pending_anonymous_cuts: Vec<Str>,
}

impl Clipboard {
	/// Starts a new batch, retaining named registers and clearing anonymous
	/// state.
	pub fn start_batch(&self) -> Self {
		Self { named: self.named.clone(), ..Self::default() }
	}

	/// Returns a named register.
	pub fn named(&self, name: &str) -> Option<&[Str]> {
		self.named.get(name).map(Vec::as_slice)
	}

	/// Publishes named registers from a transactional fork.
	pub fn commit_named_from(&mut self, fork: &Self) {
		self.named.extend(fork.named.clone());
	}
}

/// Empty-register handling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmptyPasteMode {
	/// Reject an empty or ambiguous anonymous paste.
	Strict,
	/// Drop an incomplete paste while producing a streaming preview.
	Drop,
}

/// Clipboard lowering failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipboardError {
	/// Patch-language source line associated with the failure.
	pub line_num: usize,
	/// Human-readable failure description.
	pub message:  Str,
}

impl fmt::Display for ClipboardError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "line {}: {}", self.line_num, self.message)
	}
}
impl Error for ClipboardError {}

/// Result of clipboard lowering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipboardResolution {
	/// Concrete edits with all cuts and pastes removed.
	pub edits:    Vec<Edit>,
	/// Non-fatal diagnostics.
	pub warnings: Vec<Str>,
}

/// Returns whether any edit reads or writes a register.
pub fn has_clipboard_edit(edits: &[Edit]) -> bool {
	edits.iter().any(|edit| {
		matches!(
			edit,
			Edit::Cut { .. }
				| Edit::Paste { .. }
				| Edit::Block {
					mode: crate::types::BlockMode::Cut | crate::types::BlockMode::PasteAfter,
					..
				} | Edit::Block { register: Some(_), .. }
		)
	})
}

const fn range_ok(range: ParsedRange, line_count: usize) -> bool {
	range.start.line >= 1 && range.end.line >= range.start.line && range.end.line <= line_count
}

fn cut_description(range: ParsedRange, register: Option<&str>) -> Str {
	let span = if range.start.line == range.end.line {
		range.start.line.to_string()
	} else {
		format!("{}.={}", range.start.line, range.end.line)
	};
	fmts!("CUT {span}{}", register.map_or(String::new(), |r| format!(" @{r}")))
}

fn read_register(
	clipboard: &mut Clipboard,
	register: Option<&Str>,
	span: bool,
	line_num: usize,
	mode: EmptyPasteMode,
	warnings: &mut Vec<Str>,
) -> Result<Option<Vec<Str>>, ClipboardError> {
	if let Some(name) = register {
		if let Some(lines) = clipboard.named.get(name) {
			return Ok(Some(lines.clone()));
		}
		if mode == EmptyPasteMode::Drop {
			return Ok(None);
		}
		if span {
			return Err(ClipboardError {
				line_num,
				message: fmts!("register @{name} is empty; refusing to delete a span"),
			});
		}
		warnings.push(fmts!("line {line_num}: register @{name} is empty; pasted nothing"));
		return Ok(Some(Vec::new()));
	}
	if clipboard.pending_anonymous_cuts.len() > 1 {
		if mode == EmptyPasteMode::Drop {
			return Ok(None);
		}
		return Err(ClipboardError {
			line_num,
			message: fmts!(
				"anonymous paste is ambiguous after cuts: {}",
				clipboard
					.pending_anonymous_cuts
					.iter()
					.map(AsRef::as_ref)
					.collect::<Vec<_>>()
					.join(", ")
			),
		});
	}
	let Some(lines) = clipboard.anonymous.clone() else {
		if mode == EmptyPasteMode::Drop {
			return Ok(None);
		}
		return Err(ClipboardError { line_num, message: Str::new("anonymous register is empty") });
	};
	clipboard.pending_anonymous_cuts.clear();
	Ok(Some(lines))
}

/// Captures cuts from original lines and lowers pastes in authored order.
pub fn resolve_clipboard_edits(
	edits: &[Edit],
	original_lines: &[Str],
	clipboard: &mut Clipboard,
	mode: EmptyPasteMode,
) -> Result<ClipboardResolution, ClipboardError> {
	if !has_clipboard_edit(edits) {
		return Ok(ClipboardResolution { edits: edits.to_vec(), warnings: Vec::new() });
	}
	let mut out = Vec::new();
	let mut warnings = Vec::new();
	let mut index = 0;
	for edit in edits {
		match edit {
			Edit::Cut { range, register, line_num, .. } => {
				if !range_ok(*range, original_lines.len()) {
					return Err(ClipboardError {
						line_num: *line_num,
						message:  fmts!(
							"cut {}..={} is out of range (file has {} lines)",
							range.start.line,
							range.end.line,
							original_lines.len()
						),
					});
				}
				let captured = original_lines[range.start.line - 1..range.end.line].to_vec();
				if let Some(name) = register {
					clipboard.named.insert(name.clone(), captured);
				} else {
					clipboard.anonymous = Some(captured);
					clipboard
						.pending_anonymous_cuts
						.push(cut_description(*range, None));
				}
			},
			Edit::Paste { at, register, line_num, block_start, .. } => {
				let is_span = matches!(at, PasteTarget::Span { .. });
				let Some(lines) = read_register(
					clipboard,
					register.as_ref(),
					is_span,
					*line_num,
					mode,
					&mut warnings,
				)?
				else {
					continue;
				};
				match at {
					PasteTarget::Gap { cursor } => {
						for text in lines {
							out.push(Edit::Insert {
								cursor: *cursor,
								text,
								line_num: *line_num,
								index,
								mode: InsertMode::Literal,
								block_start: *block_start,
							});
							index += 1;
						}
					},
					PasteTarget::Span { range } => {
						if !range_ok(*range, original_lines.len()) {
							return Err(ClipboardError {
								line_num: *line_num,
								message:  fmts!(
									"paste span {}..={} is out of range (file has {} lines)",
									range.start.line,
									range.end.line,
									original_lines.len()
								),
							});
						}
						for text in lines {
							out.push(Edit::Insert {
								cursor: Cursor::BeforeAnchor { anchor: range.start },
								text,
								line_num: *line_num,
								index,
								mode: InsertMode::Replacement,
								block_start: None,
							});
							index += 1;
						}
						for line in range.start.line..=range.end.line {
							out.push(Edit::Delete { anchor: Anchor { line }, line_num: *line_num, index });
							index += 1;
						}
					},
				}
			},
			other => out.push(other.clone()),
		}
	}
	Ok(ClipboardResolution { edits: out, warnings })
}

/// Validates anonymous register sequencing without reading file content.
pub fn validate_clipboard_sequence(
	edits: &[Edit],
	clipboard: &Clipboard,
) -> Result<(), ClipboardError> {
	let mut fork = clipboard.clone();
	let mut warnings = Vec::new();
	for edit in edits {
		match edit {
			Edit::Cut { range, register, .. } => {
				if let Some(name) = register {
					fork.named.insert(name.clone(), Vec::new());
				} else {
					fork.anonymous = Some(Vec::new());
					fork
						.pending_anonymous_cuts
						.push(cut_description(*range, None));
				}
			},
			Edit::Paste { at, register, line_num, .. } => {
				let _ = read_register(
					&mut fork,
					register.as_ref(),
					matches!(at, PasteTarget::Span { .. }),
					*line_num,
					EmptyPasteMode::Strict,
					&mut warnings,
				)?;
			},
			_ => {},
		}
	}
	Ok(())
}
