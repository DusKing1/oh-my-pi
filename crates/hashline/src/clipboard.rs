//! In-memory clipboard lowering for `CUT` and register-backed `PUT` edits.

use std::collections::BTreeMap;

use omp_core::Str;

use crate::types::{Anchor, ApplyWarning, Cursor, Edit, InsertMode, ParsedRange, PasteTarget};

/// Clipboard state shared by sections in one transaction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Clipboard {
	anonymous:              Option<Vec<Str>>,
	named:                  BTreeMap<Str, Vec<Str>>,
	pending_anonymous_cuts: Vec<ParsedRange>,
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

/// The clipboard operation whose source range is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::Display)]
#[strum(serialize_all = "lowercase")]
pub enum ClipboardRangeOperation {
	/// Capturing source rows into a register.
	Cut,
	/// Replacing a span with register contents.
	Paste,
}

/// Clipboard lowering failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ClipboardError {
	/// A named register is empty and a span paste would delete source rows.
	#[error(
		"line {patch_line}: register @{register} is empty; refusing to delete a span. Populate the \
		 register with a named CUT before replacing a span."
	)]
	EmptyNamedSpan {
		/// The authored patch line containing the paste.
		patch_line: usize,
		/// The empty register name without the `@` prefix.
		register:   Str,
	},
	/// Multiple anonymous cuts make the anonymous paste source ambiguous.
	#[error(
		"line {patch_line}: anonymous paste is ambiguous because more than one anonymous CUT can \
		 supply it. Name each CUT register and paste the intended register explicitly."
	)]
	AmbiguousAnonymousPaste {
		/// The authored patch line containing the paste.
		patch_line: usize,
		/// The candidate anonymous cut ranges in authored order.
		cuts:       Vec<ParsedRange>,
	},
	/// No anonymous cut has populated the anonymous register.
	#[error(
		"line {patch_line}: anonymous register is empty; issue an anonymous CUT before this paste"
	)]
	EmptyAnonymousRegister {
		/// The authored patch line containing the paste.
		patch_line: usize,
	},
	/// A cut or span paste addresses source rows outside the file.
	#[error(
		"line {patch_line}: {operation} range {start}..={end} is out of range (file has {total} \
		 addressable lines); re-read the file and use an existing range"
	)]
	RangeOutOfBounds {
		/// The authored patch line containing the operation.
		patch_line: usize,
		/// Whether the invalid range belongs to a cut or paste.
		operation:  ClipboardRangeOperation,
		/// The first requested source line.
		start:      usize,
		/// The last requested source line.
		end:        usize,
		/// The number of addressable source lines.
		total:      usize,
	},
}

impl ClipboardError {
	/// Returns the stable machine-readable diagnostic code.
	#[must_use]
	pub fn code(&self) -> &'static str {
		self.into()
	}
}

/// Result of clipboard lowering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipboardResolution {
	/// Concrete edits with all cuts and pastes removed.
	pub edits:    Vec<Edit>,
	/// Non-fatal diagnostics.
	pub warnings: Vec<ApplyWarning>,
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

fn read_register(
	clipboard: &mut Clipboard,
	register: Option<&Str>,
	span: bool,
	line_num: usize,
	mode: EmptyPasteMode,
	warnings: &mut Vec<ApplyWarning>,
) -> Result<Option<Vec<Str>>, ClipboardError> {
	if let Some(name) = register {
		if let Some(lines) = clipboard.named.get(name) {
			return Ok(Some(lines.clone()));
		}
		if mode == EmptyPasteMode::Drop {
			return Ok(None);
		}
		if span {
			return Err(ClipboardError::EmptyNamedSpan {
				patch_line: line_num,
				register:   name.clone(),
			});
		}
		warnings
			.push(ApplyWarning::EmptyRegisterPaste { patch_line: line_num, register: name.clone() });
		return Ok(Some(Vec::new()));
	}
	if clipboard.pending_anonymous_cuts.len() > 1 {
		if mode == EmptyPasteMode::Drop {
			return Ok(None);
		}
		return Err(ClipboardError::AmbiguousAnonymousPaste {
			patch_line: line_num,
			cuts:       clipboard.pending_anonymous_cuts.clone(),
		});
	}
	let Some(lines) = clipboard.anonymous.clone() else {
		if mode == EmptyPasteMode::Drop {
			return Ok(None);
		}
		return Err(ClipboardError::EmptyAnonymousRegister { patch_line: line_num });
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
					return Err(ClipboardError::RangeOutOfBounds {
						patch_line: *line_num,
						operation:  ClipboardRangeOperation::Cut,
						start:      range.start.line,
						end:        range.end.line,
						total:      original_lines.len(),
					});
				}
				let captured = original_lines[range.start.line - 1..range.end.line].to_vec();
				if let Some(name) = register {
					clipboard.named.insert(name.clone(), captured);
				} else {
					clipboard.anonymous = Some(captured);
					clipboard.pending_anonymous_cuts.push(*range);
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
							return Err(ClipboardError::RangeOutOfBounds {
								patch_line: *line_num,
								operation:  ClipboardRangeOperation::Paste,
								start:      range.start.line,
								end:        range.end.line,
								total:      original_lines.len(),
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
					fork.pending_anonymous_cuts.push(*range);
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
