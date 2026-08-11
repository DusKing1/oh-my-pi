//! Syntax-aware lowering of deferred block edits.

use std::{error::Error, fmt};

use omp_ast::block::{BlockRangeOptions, block_range_at};
use omp_core::Str;

use crate::types::{
	Anchor, BlockMode, BlockResolution, Cursor, Edit, InsertMode, ParsedRange, PasteTarget,
};

/// Handling for an unresolved block anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnresolvedBlockMode {
	/// Return a structural error.
	Strict,
	/// Drop replace/cut edits for a streaming preview.
	Drop,
}

/// Block lowering failure.
#[derive(Debug)]
pub struct BlockError {
	/// Patch-language source line associated with the failure.
	pub line_num: usize,
	/// Human-readable failure description.
	pub message:  Str,
}
impl fmt::Display for BlockError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "line {}: {}", self.line_num, self.message)
	}
}
impl Error for BlockError {}

/// Result of syntax-aware block lowering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockLowering {
	/// Concrete edits containing no block variants.
	pub edits:       Vec<Edit>,
	/// Successfully resolved original-coordinate spans.
	pub resolutions: Vec<BlockResolution>,
	/// Non-fatal fallback diagnostics.
	pub warnings:    Vec<Str>,
}

/// Returns whether any edit still requires block resolution.
pub fn has_block_edit(edits: &[Edit]) -> bool {
	edits.iter().any(|edit| matches!(edit, Edit::Block { .. }))
}

fn structural_closer(text: &str) -> bool {
	let t = text.trim();
	let core = t
		.strip_suffix(';')
		.or_else(|| t.strip_suffix(','))
		.unwrap_or(t);
	!core.is_empty() && core.bytes().all(|b| matches!(b, b')' | b']' | b'}'))
}

/// Resolves every block edit against the original normalized-LF source.
pub fn resolve_block_edits(
	edits: &[Edit],
	text: &str,
	path: Option<&str>,
	mode: UnresolvedBlockMode,
) -> Result<BlockLowering, BlockError> {
	if !has_block_edit(edits) {
		return Ok(BlockLowering {
			edits:       edits.to_vec(),
			resolutions: Vec::new(),
			warnings:    Vec::new(),
		});
	}
	let lines: Vec<&str> = text.split('\n').collect();
	let mut out = Vec::new();
	let mut resolutions = Vec::new();
	let mut warnings = Vec::new();
	let mut index = 0;
	for edit in edits {
		let Edit::Block { anchor, payloads, mode: block_mode, register, line_num, .. } = edit else {
			out.push(edit.clone());
			continue;
		};
		let resolved = block_range_at(BlockRangeOptions {
			code: text.to_owned(),
			lang: None,
			path: path.map(str::to_owned),
			line: u32::try_from(anchor.line).unwrap_or(u32::MAX),
		})
		.ok()
		.flatten();
		let Some(span) = resolved else {
			if matches!(block_mode, BlockMode::InsertAfter | BlockMode::PasteAfter) {
				let closer = lines
					.get(anchor.line.saturating_sub(1))
					.is_some_and(|line| structural_closer(line));
				warnings.push(Str::new(format!(
					"line {line_num}: block at line {} could not be resolved{}; lowered to after-line \
					 insertion",
					anchor.line,
					if closer {
						" because the anchor is a closer"
					} else {
						""
					}
				)));
				let cursor = Cursor::AfterAnchor { anchor: *anchor };
				if *block_mode == BlockMode::PasteAfter {
					out.push(Edit::Paste {
						at: PasteTarget::Gap { cursor },
						register: register.clone(),
						line_num: *line_num,
						index,
						block_start: None,
					});
					index += 1;
				} else {
					for payload in payloads {
						out.push(Edit::Insert {
							cursor,
							text: payload.clone(),
							line_num: *line_num,
							index,
							mode: InsertMode::Literal,
							block_start: None,
						});
						index += 1;
					}
				}
				continue;
			}
			if mode == UnresolvedBlockMode::Drop {
				continue;
			}
			return Err(BlockError {
				line_num: *line_num,
				message:  Str::new(format!(
					"no multi-line syntactic block begins on line {}",
					anchor.line
				)),
			});
		};
		let start = span.start_line as usize;
		let end = span.end_line as usize;
		if start == end {
			if mode == UnresolvedBlockMode::Drop {
				continue;
			}
			return Err(BlockError {
				line_num: *line_num,
				message:  Str::new(format!(
					"line {} is a single-line statement, not a multi-line block",
					anchor.line
				)),
			});
		}
		resolutions.push(BlockResolution { anchor_line: anchor.line, start, end, mode: *block_mode });
		match block_mode {
			BlockMode::Cut => {
				out.push(Edit::Cut {
					range: ParsedRange { start: Anchor { line: start }, end: Anchor { line: end } },
					register: register.clone(),
					line_num: *line_num,
					index,
				});
				index += 1;
				for line in start..=end {
					out.push(Edit::Delete { anchor: Anchor { line }, line_num: *line_num, index });
					index += 1;
				}
			},
			BlockMode::PasteAfter => {
				out.push(Edit::Paste {
					at: PasteTarget::Gap {
						cursor: Cursor::AfterAnchor { anchor: Anchor { line: end } },
					},
					register: register.clone(),
					line_num: *line_num,
					index,
					block_start: Some(start),
				});
				index += 1;
			},
			BlockMode::InsertAfter => {
				for payload in payloads {
					out.push(Edit::Insert {
						cursor: Cursor::AfterAnchor { anchor: Anchor { line: end } },
						text: payload.clone(),
						line_num: *line_num,
						index,
						mode: InsertMode::Literal,
						block_start: Some(start),
					});
					index += 1;
				}
			},
			BlockMode::Replace if register.is_some() => {
				out.push(Edit::Paste {
					at: PasteTarget::Span {
						range: ParsedRange { start: Anchor { line: start }, end: Anchor { line: end } },
					},
					register: register.clone(),
					line_num: *line_num,
					index,
					block_start: None,
				});
				index += 1;
			},
			BlockMode::Replace => {
				for payload in payloads {
					out.push(Edit::Insert {
						cursor: Cursor::BeforeAnchor { anchor: Anchor { line: start } },
						text: payload.clone(),
						line_num: *line_num,
						index,
						mode: InsertMode::Replacement,
						block_start: None,
					});
					index += 1;
				}
				for line in start..=end {
					out.push(Edit::Delete { anchor: Anchor { line }, line_num: *line_num, index });
					index += 1;
				}
			},
		}
	}
	Ok(BlockLowering { edits: out, resolutions, warnings })
}
