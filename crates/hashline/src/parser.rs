//! Lenient token-driven state machine for hashline section bodies.

use std::collections::{BTreeSet, HashMap, HashSet};

use omp_core::Str;

use crate::{
	format::HL_RANGE_SEP,
	tokenizer::{BlockTarget, Token, Tokenizer, is_hunk_header_text},
	types::{
		Anchor, BlockMode, Cursor, Diagnostic, DiagnosticCode, Edit, FileOp, InsertMode, ParseError,
		ParsedPatch, ParsedRange, PasteTarget,
	},
};

/// Maximum range size expanded before the target file's line count is known.
pub const MAX_EXPANDED_RANGE_LINES: usize = 100_000;

#[derive(Debug, Clone)]
struct PayloadRow {
	text:     Str,
	line_num: usize,
	bare:     bool,
	minus:    bool,
}

#[derive(Debug, Clone)]
struct Pending {
	target:          BlockTarget,
	line_num:        usize,
	authored_index:  usize,
	payloads:        Vec<PayloadRow>,
	had_colon:       bool,
	deferred_blanks: Vec<PayloadRow>,
}

/// Stateful token consumer used by complete and streaming parsers.
#[derive(Debug, Default)]
pub struct Executor {
	edits:                    Vec<Edit>,
	diagnostics:              Vec<Diagnostic>,
	edit_index:               usize,
	hunk_index:               usize,
	pending:                  Option<Pending>,
	file_op:                  Option<FileOp>,
	file_op_location:         Option<(usize, usize)>,
	terminated:               bool,
	recovered_snapshot_lines: HashSet<usize>,
	skippable_comments:       Vec<(Str, usize)>,
}

impl Executor {
	/// Constructs an empty parser executor.
	pub fn new() -> Self {
		Self::default()
	}

	/// Consumes one tokenizer token.
	pub fn feed(&mut self, token: Token) -> Result<(), ParseError> {
		if self.terminated {
			return Ok(());
		}
		match token {
			Token::EnvelopeBegin { .. } => self.consume_skippable_comments(),
			Token::EnvelopeEnd { .. } => {
				self.consume_skippable_comments()?;
				self.terminated = true;
				Ok(())
			},
			Token::Abort { .. } => {
				self.terminated = true;
				Ok(())
			},
			Token::Header { .. } => {
				self.consume_skippable_comments()?;
				self.flush_pending()
			},
			Token::Blank { line_num } => {
				self.consume_skippable_comments()?;
				self.handle_blank("", line_num);
				Ok(())
			},
			Token::PayloadLiteral { line_num, text } => {
				self.consume_skippable_comments()?;
				self.handle_literal_payload(text, line_num)
			},
			Token::Raw { line_num, text } => {
				if self.pending.is_none() && text.trim_start().starts_with('#') {
					self.skippable_comments.push((text, line_num));
					return Ok(());
				}
				self.consume_skippable_comments()?;
				self.handle_raw(text, line_num)
			},
			Token::Operation { line_num, target, had_colon } => {
				self.skippable_comments.clear();
				self.handle_operation(target, line_num, had_colon)
			},
		}
	}

	/// Finishes a complete parse, flushing even an empty replacement hunk.
	pub fn finish(mut self) -> Result<ParsedPatch, ParseError> {
		self.consume_skippable_comments()?;
		self.flush_pending()?;
		self.validate_file_op()?;
		self.normalize_overlapping_ranges()?;
		Ok(ParsedPatch {
			edits:       self.edits,
			file_op:     self.file_op,
			diagnostics: self.diagnostics,
		})
	}

	/// Finishes a streaming parse while dropping an incomplete trailing hunk.
	pub fn finish_streaming(mut self) -> Result<ParsedPatch, ParseError> {
		self.consume_skippable_comments()?;
		let flush = self.pending.as_ref().is_some_and(|pending| {
			!pending.payloads.is_empty()
				|| complete_bodyless_target(&pending.target, pending.had_colon)
		});
		if flush {
			self.flush_pending()?;
		} else {
			self.pending = None;
		}
		self.validate_file_op()?;
		self.normalize_overlapping_ranges()?;
		Ok(ParsedPatch {
			edits:       self.edits,
			file_op:     self.file_op,
			diagnostics: self.diagnostics,
		})
	}

	/// Clears all emitted state so this executor can parse another section.
	pub fn reset(&mut self) {
		*self = Self::default();
	}

	fn consume_skippable_comments(&mut self) -> Result<(), ParseError> {
		for (text, line_num) in std::mem::take(&mut self.skippable_comments) {
			self.handle_raw(text, line_num)?;
		}
		Ok(())
	}

	fn handle_operation(
		&mut self,
		target: BlockTarget,
		line_num: usize,
		had_colon: bool,
	) -> Result<(), ParseError> {
		match &target {
			BlockTarget::Replace { range, register } => {
				validate_range(*range, line_num, "replace", register.as_deref(), self.hunk_index)?;
			},
			BlockTarget::Cut { range, register } => {
				validate_range(*range, line_num, "cut", register.as_deref(), self.hunk_index)?;
			},
			_ => {},
		}
		if had_colon && target.is_cut() {
			self.warn_once(
				DiagnosticCode::CutColonIgnored,
				line_num,
				self.hunk_index,
				"Ignored a trailing `:` on bodyless `CUT`. Prefer `CUT N.=M` / `CUT N*` without a \
				 colon.",
			);
		}
		if had_colon
			&& target.register().is_some()
			&& !matches!(&target, BlockTarget::Rem | BlockTarget::Move { .. })
		{
			return Self::fail(
				DiagnosticCode::RegisterColon,
				line_num,
				self.hunk_index,
				"`PUT … @name` pastes the register and never takes `:`; drop the colon or drop \
				 `@name` and write `+TEXT` body rows.",
			);
		}
		self.flush_pending()?;
		let authored_index = self.hunk_index;
		self.hunk_index += 1;
		match target {
			BlockTarget::Rem => self.set_file_op(FileOp::Rem, line_num, authored_index),
			BlockTarget::Move { dest } => {
				self.set_file_op(FileOp::Move { dest }, line_num, authored_index)
			},
			target => {
				self.pending = Some(Pending {
					target,
					line_num,
					authored_index,
					payloads: Vec::new(),
					had_colon,
					deferred_blanks: Vec::new(),
				});
				Ok(())
			},
		}
	}

	fn set_file_op(
		&mut self,
		file_op: FileOp,
		line_num: usize,
		authored_index: usize,
	) -> Result<(), ParseError> {
		if self.file_op.is_some() {
			return Self::fail(
				DiagnosticCode::FileOperationConflict,
				line_num,
				authored_index,
				format!(
					"line {line_num}: only one file-level op (`REM` or `MV`) is allowed per section."
				),
			);
		}
		if matches!(&file_op, FileOp::Rem) && !self.edits.is_empty() {
			return Self::fail(
				DiagnosticCode::FileOperationConflict,
				line_num,
				authored_index,
				"`REM` deletes the whole file and takes no body rows or line ops. Issue it alone \
				 under the header.",
			);
		}
		self.file_op = Some(file_op);
		self.file_op_location = Some((line_num, authored_index));
		Ok(())
	}

	fn validate_file_op(&self) -> Result<(), ParseError> {
		if matches!(&self.file_op, Some(FileOp::Rem)) && !self.edits.is_empty() {
			let (line_num, authored_index) = self.file_op_location.unwrap_or((0, 0));
			return Err(ParseError::new(Diagnostic::error(
				DiagnosticCode::FileOperationConflict,
				(line_num != 0).then_some(line_num),
				(line_num != 0).then_some(authored_index),
				"`REM` deletes the whole file and cannot be combined with line ops.",
			)));
		}
		Ok(())
	}

	fn handle_literal_payload(&mut self, text: Str, line_num: usize) -> Result<(), ParseError> {
		if self.pending.is_none() {
			if self.file_op.is_some() {
				return Self::fail(
					DiagnosticCode::BodyNotAllowed,
					line_num,
					self.hunk_index,
					"`MV DEST` and `REM` do not take body rows.",
				);
			}
			return Self::fail(
				DiagnosticCode::OrphanPayload,
				line_num,
				self.hunk_index,
				format!("line {line_num}: payload line has no preceding hunk header. Got `+{text}`."),
			);
		}
		let pending = self.pending.as_ref().unwrap();
		if let Some(message) = bodyless_target_message(&pending.target, pending.had_colon) {
			return Self::fail(
				DiagnosticCode::BodyNotAllowed,
				line_num,
				pending.authored_index,
				message,
			);
		}
		let authored_index = pending.authored_index;
		self.commit_deferred_blanks();
		if is_hunk_header_text(&text) {
			self.diagnostics.push(Diagnostic::warning(
				DiagnosticCode::LiteralOperationRow,
				Some(line_num),
				Some(authored_index),
				format!(
					"line {line_num}: body row `+{text}` is itself a valid hunk header, so it is \
					 inserted as literal text rather than executed."
				),
			));
		}
		self.pending.as_mut().unwrap().payloads.push(PayloadRow {
			text,
			line_num,
			bare: false,
			minus: false,
		});
		Ok(())
	}

	fn handle_raw(&mut self, text: Str, line_num: usize) -> Result<(), ParseError> {
		if self.pending.is_none() && is_read_metadata_line(&text) {
			self.warn_once(
				DiagnosticCode::ReadMetadataIgnored,
				line_num,
				self.hunk_index,
				"Ignored copied read-output elision row(s). Re-read elided ranges before editing them.",
			);
			return Ok(());
		}
		if let Some(message) = foreign_patch_message(&text) {
			return Self::fail(
				DiagnosticCode::ForeignPatchSyntax,
				line_num,
				self
					.pending
					.as_ref()
					.map_or(self.hunk_index, |pending| pending.authored_index),
				format!("line {line_num}: {message}"),
			);
		}
		if self.file_op.is_some() {
			return Self::fail(
				DiagnosticCode::BodyNotAllowed,
				line_num,
				self.hunk_index,
				"`MV DEST` and `REM` do not take body rows.",
			);
		}
		if let Some(pending) = self.pending.as_ref() {
			if text.trim().is_empty() {
				self.handle_blank(&text, line_num);
				return Ok(());
			}
			if let Some(message) = bodyless_target_message(&pending.target, pending.had_colon) {
				return Self::fail(
					DiagnosticCode::BodyNotAllowed,
					line_num,
					pending.authored_index,
					message,
				);
			}
			let authored_index = pending.authored_index;
			let minus = text.trim_start().starts_with('-');
			if !minus {
				self.warn_once(
					DiagnosticCode::BareBodyRecovered,
					line_num,
					authored_index,
					"Auto-prefixed bare body row(s) with `+`. Body rows must be `+TEXT` literal lines.",
				);
			}
			self.commit_deferred_blanks();
			self.pending.as_mut().unwrap().payloads.push(PayloadRow {
				text,
				line_num,
				bare: true,
				minus,
			});
			return Ok(());
		}
		if text.trim().is_empty() {
			return Ok(());
		}
		if let Some(range) = parse_top_level_bare_range_header(&text) {
			let authored_index = self.hunk_index;
			validate_range(range, line_num, "replace", None, authored_index)?;
			self.hunk_index += 1;
			self.pending = Some(Pending {
				target: BlockTarget::Replace { range, register: None },
				line_num,
				authored_index,
				payloads: Vec::new(),
				had_colon: true,
				deferred_blanks: Vec::new(),
			});
			self.warn_once(
				DiagnosticCode::BareRangeRecovered,
				line_num,
				authored_index,
				"Recovered a bare `N.=M:` header as `PUT N.=M:`. Prefix replacement ranges with `PUT`.",
			);
			return Ok(());
		}
		if let Some((line, body)) = parse_top_level_snapshot_row(&text) {
			let authored_index = self.hunk_index;
			self.hunk_index += 1;
			if !self.recovered_snapshot_lines.insert(line) {
				return Self::fail(
					DiagnosticCode::DuplicateSnapshotRow,
					line_num,
					authored_index,
					format!(
						"line {line_num}: two or more pasted `{line}:TEXT` rows name line {line}; \
						 repeating a snapshot-row number would keep only the last row and drop the \
						 rest. Write one explicit `PUT {line}.=M:` hunk."
					),
				);
			}
			let range = ParsedRange { start: Anchor { line }, end: Anchor { line } };
			self.push_insert(
				Cursor::BeforeAnchor { anchor: range.start },
				body,
				line_num,
				InsertMode::Replacement,
			);
			self.push_delete_range(range, line_num);
			self.warn_once(
				DiagnosticCode::SnapshotRowRecovered,
				line_num,
				authored_index,
				"Recovered top-level `N:TEXT` snapshot row(s) as single-line `PUT N.=N:` \
				 replacements. Use explicit `PUT` headers for reliable edits.",
			);
			return Ok(());
		}
		Self::fail(
			DiagnosticCode::OrphanPayload,
			line_num,
			self.hunk_index,
			format!(
				"line {line_num}: payload line has no preceding hunk header. Use `PUT N.=M:`, `CUT \
				 N.=M`, or `PUT <N:`/`PUT >N:` above the body. Got {text:?}."
			),
		)
	}

	fn handle_blank(&mut self, text: &str, line_num: usize) {
		let Some(pending) = self.pending.as_mut() else {
			return;
		};
		if bodyless_target_message(&pending.target, pending.had_colon).is_some()
			|| pending.payloads.is_empty()
		{
			return;
		}
		pending.deferred_blanks.push(PayloadRow {
			text: text.into(),
			line_num,
			bare: true,
			minus: false,
		});
	}

	fn commit_deferred_blanks(&mut self) {
		let Some(pending) = self.pending.as_mut() else {
			return;
		};
		if pending.deferred_blanks.is_empty() {
			return;
		}
		let line_num = pending.deferred_blanks[0].line_num;
		let authored_index = pending.authored_index;
		pending.payloads.append(&mut pending.deferred_blanks);
		self.warn_once(
			DiagnosticCode::BareBodyRecovered,
			line_num,
			authored_index,
			"Auto-prefixed bare body row(s) with `+`. Body rows must be `+TEXT` literal lines.",
		);
	}

	fn flush_pending(&mut self) -> Result<(), ParseError> {
		let Some(mut pending) = self.pending.take() else {
			return Ok(());
		};
		self.resolve_minus_rows(&mut pending.payloads, pending.authored_index)?;
		strip_bare_prefixes_if_uniform(&mut pending.payloads);
		let Pending { target, line_num, authored_index, payloads, had_colon, .. } = pending;
		match target {
			BlockTarget::Cut { range, register } => {
				self.push_cut(range, line_num, register);
			},
			BlockTarget::CutBlock { anchor, register } => {
				self.push_block(anchor, Vec::new(), BlockMode::Cut, register, line_num);
			},
			BlockTarget::Replace { range, register } => {
				if let Some(register) = register {
					self.push_paste(PasteTarget::Span { range }, Some(register), line_num);
				} else if payloads.is_empty() {
					if !had_colon {
						return Self::fail(
							DiagnosticCode::AnonymousSpanPaste,
							line_num,
							authored_index,
							"Colonless `PUT` is clipboard-backed, and span targets need a named register \
							 (`PUT 5.=9 @name`). Add `:` and `+TEXT` rows for literal content.",
						);
					}
					self.push_delete_range(range, line_num);
					self.warn_once(
						DiagnosticCode::EmptyPutRecovered,
						line_num,
						authored_index,
						"Interpreted an empty `PUT` body as deletion. Use `CUT N.=M` or `CUT N*` for \
						 bodyless deletes.",
					);
				} else {
					for row in payloads {
						self.push_insert(
							Cursor::BeforeAnchor { anchor: range.start },
							row.text,
							line_num,
							InsertMode::Replacement,
						);
					}
					self.push_delete_range(range, line_num);
				}
			},
			BlockTarget::Block { anchor, register } => {
				if register.is_some() {
					self.push_block(anchor, Vec::new(), BlockMode::Replace, register, line_num);
				} else if payloads.is_empty() {
					if !had_colon {
						return Self::fail(
							DiagnosticCode::AnonymousSpanPaste,
							line_num,
							authored_index,
							"Colonless block `PUT` needs a named register, or `:` with literal body rows.",
						);
					}
					self.push_block(anchor, Vec::new(), BlockMode::Replace, None, line_num);
					self.warn_once(
						DiagnosticCode::EmptyPutRecovered,
						line_num,
						authored_index,
						"Interpreted an empty `PUT` body as deletion. Use `CUT N.=M` or `CUT N*` for \
						 bodyless deletes.",
					);
				} else {
					self.push_block(
						anchor,
						payloads.into_iter().map(|row| row.text).collect(),
						BlockMode::Replace,
						None,
						line_num,
					);
				}
			},
			BlockTarget::InsertAfterBlock { anchor, register } => {
				if register.is_some() || (!had_colon && payloads.is_empty()) {
					self.push_block(anchor, Vec::new(), BlockMode::PasteAfter, register, line_num);
				} else if payloads.is_empty() {
					return Self::fail(
						DiagnosticCode::MissingBody,
						line_num,
						authored_index,
						"`PUT >N*:` promises body rows and got none. Write `+TEXT` rows or drop `:` to \
						 paste a register.",
					);
				} else {
					self.push_block(
						anchor,
						payloads.into_iter().map(|row| row.text).collect(),
						BlockMode::InsertAfter,
						None,
						line_num,
					);
				}
			},
			target @ (BlockTarget::InsertBefore { .. }
			| BlockTarget::InsertAfter { .. }
			| BlockTarget::Bof { .. }
			| BlockTarget::Eof { .. }) => {
				let (cursor, register) = gap_target_parts(target);
				if register.is_some() || (!had_colon && payloads.is_empty()) {
					self.push_paste(PasteTarget::Gap { cursor }, register, line_num);
				} else if payloads.is_empty() {
					return Self::fail(
						DiagnosticCode::MissingBody,
						line_num,
						authored_index,
						"`PUT <N:` / `PUT >N:` promises body rows and got none. Write `+TEXT` rows, or \
						 drop `:` to paste a register.",
					);
				} else {
					for row in payloads {
						self.push_insert(cursor, row.text, line_num, InsertMode::Literal);
					}
				}
			},
			BlockTarget::Rem | BlockTarget::Move { .. } => {},
		}
		Ok(())
	}

	fn resolve_minus_rows(
		&mut self,
		payloads: &mut Vec<PayloadRow>,
		authored_index: usize,
	) -> Result<(), ParseError> {
		let first_minus = payloads.iter().find(|row| row.minus).cloned();
		let Some(first_minus) = first_minus else {
			return Ok(());
		};
		let all_bullet_shaped = payloads
			.iter()
			.filter(|row| row.minus)
			.all(|row| markdown_bullet(&row.text));
		let has_explicit = payloads.iter().any(|row| !row.minus && !row.bare);
		let has_explicit_bullet = payloads
			.iter()
			.any(|row| !row.minus && !row.bare && markdown_bullet(&row.text));
		if all_bullet_shaped && (!has_explicit || has_explicit_bullet) {
			self.warn_once(
				DiagnosticCode::MinusBulletRecovered,
				first_minus.line_num,
				authored_index,
				"Auto-prefixed bare `- ` bullet row(s) as literal content. Always prefix literal rows \
				 with `+`: `+- item`.",
			);
			return Ok(());
		}
		if has_explicit && !all_bullet_shaped {
			payloads.retain(|row| !row.minus);
			self.warn_once(
				DiagnosticCode::DiffOldRowsIgnored,
				first_minus.line_num,
				authored_index,
				"Ignored unified-diff `-old` row(s); the range already removes old content, so only \
				 `+new` rows were kept.",
			);
			return Ok(());
		}
		Self::fail(
			DiagnosticCode::MinusRowRejected,
			first_minus.line_num,
			authored_index,
			"`-` rows are not valid; the range already names changed lines. For Markdown bullets or \
			 other literal `-` lines, prefix with `+`: `+- item`.",
		)
	}

	fn push_insert(&mut self, cursor: Cursor, text: Str, line_num: usize, mode: InsertMode) {
		let index = self.next_edit_index();
		self
			.edits
			.push(Edit::Insert { cursor, text, line_num, index, mode, block_start: None });
	}

	fn push_delete(&mut self, anchor: Anchor, line_num: usize) {
		let index = self.next_edit_index();
		self.edits.push(Edit::Delete { anchor, line_num, index });
	}

	fn push_delete_range(&mut self, range: ParsedRange, line_num: usize) {
		for line in range.start.line..=range.end.line {
			self.push_delete(Anchor { line }, line_num);
		}
	}

	fn push_cut(&mut self, range: ParsedRange, line_num: usize, register: Option<Str>) {
		let index = self.next_edit_index();
		self
			.edits
			.push(Edit::Cut { range, register, line_num, index });
		self.push_delete_range(range, line_num);
	}

	fn push_paste(&mut self, at: PasteTarget, register: Option<Str>, line_num: usize) {
		let index = self.next_edit_index();
		self
			.edits
			.push(Edit::Paste { at, register, line_num, index, block_start: None });
	}

	fn push_block(
		&mut self,
		anchor: Anchor,
		payloads: Vec<Str>,
		mode: BlockMode,
		register: Option<Str>,
		line_num: usize,
	) {
		let index = self.next_edit_index();
		self
			.edits
			.push(Edit::Block { anchor, payloads, mode, register, line_num, index });
	}

	const fn next_edit_index(&mut self) -> usize {
		let index = self.edit_index;
		self.edit_index += 1;
		index
	}

	fn warn_once(
		&mut self,
		code: DiagnosticCode,
		line_num: usize,
		authored_index: usize,
		message: impl Into<Str>,
	) {
		if self
			.diagnostics
			.iter()
			.any(|diagnostic| diagnostic.code == code)
		{
			return;
		}
		self.diagnostics.push(Diagnostic::warning(
			code,
			Some(line_num),
			Some(authored_index),
			message,
		));
	}

	fn fail<T>(
		code: DiagnosticCode,
		line_num: usize,
		authored_index: usize,
		message: impl Into<Str>,
	) -> Result<T, ParseError> {
		Err(ParseError::new(Diagnostic::error(code, Some(line_num), Some(authored_index), message)))
	}

	fn normalize_overlapping_ranges(&mut self) -> Result<(), ParseError> {
		#[derive(Debug)]
		struct Hunk {
			line_num:            usize,
			source_lines:        BTreeSet<usize>,
			clipboard_dependent: bool,
		}
		let mut hunks: Vec<Hunk> = Vec::new();
		let mut index_by_line = HashMap::<usize, usize>::new();
		for edit in &self.edits {
			let line_num = edit.line_num();
			let hunk_index = *index_by_line.entry(line_num).or_insert_with(|| {
				hunks.push(Hunk {
					line_num,
					source_lines: BTreeSet::new(),
					clipboard_dependent: false,
				});
				hunks.len() - 1
			});
			let hunk = &mut hunks[hunk_index];
			match edit {
				Edit::Cut { .. } => hunk.clipboard_dependent = true,
				Edit::Paste { at: PasteTarget::Span { range }, .. } => {
					hunk.clipboard_dependent = true;
					hunk.source_lines.extend(range.start.line..=range.end.line);
				},
				Edit::Delete { anchor, .. } => {
					hunk.source_lines.insert(anchor.line);
				},
				_ => {},
			}
		}
		let mut owner_by_line = HashMap::<usize, usize>::new();
		let mut dropped_lines = HashSet::<usize>::new();
		for current_index in 0..hunks.len() {
			if hunks[current_index].source_lines.is_empty() {
				continue;
			}
			let overlaps: BTreeSet<_> = hunks[current_index]
				.source_lines
				.iter()
				.filter_map(|line| owner_by_line.get(line).copied())
				.collect();
			if overlaps.is_empty() {
				for line in &hunks[current_index].source_lines {
					owner_by_line.insert(*line, current_index);
				}
				continue;
			}
			let previous_index = *overlaps.first().unwrap();
			let exact = overlaps.len() == 1
				&& hunks[previous_index].source_lines == hunks[current_index].source_lines;
			if exact && !hunks[previous_index].clipboard_dependent {
				dropped_lines.insert(hunks[previous_index].line_num);
				owner_by_line.retain(|_, owner| *owner != previous_index);
				for line in &hunks[current_index].source_lines {
					owner_by_line.insert(*line, current_index);
				}
				self.warn_once(
					DiagnosticCode::DuplicateRangeCoalesced,
					hunks[current_index].line_num,
					current_index,
					"Multiple hunks targeted the same exact range; kept only the last. Issue one `PUT` \
					 or `CUT` hunk per range.",
				);
				continue;
			}
			let first_overlap = hunks[current_index]
				.source_lines
				.iter()
				.find(|line| owner_by_line.contains_key(line))
				.copied()
				.unwrap();
			return Self::fail(
				DiagnosticCode::OverlappingRange,
				hunks[current_index].line_num,
				current_index,
				format!(
					"line {}: anchor line {first_overlap} is already targeted by another hunk on line \
					 {}. Issue one hunk per range; payload is only final desired content.",
					hunks[current_index].line_num, hunks[previous_index].line_num
				),
			);
		}
		if !dropped_lines.is_empty() {
			self
				.edits
				.retain(|edit| !dropped_lines.contains(&edit.line_num()));
		}
		Ok(())
	}
}

/// Parses a complete hashline section body.
pub fn parse_patch(diff: &str) -> Result<ParsedPatch, ParseError> {
	let mut tokenizer = Tokenizer::new();
	let mut executor = Executor::new();
	for token in tokenizer.feed(diff)? {
		executor.feed(token)?;
	}
	for token in tokenizer.end() {
		executor.feed(token)?;
	}
	executor.finish()
}

/// Parses a partial section body without materializing an unfinished last hunk.
pub fn parse_patch_streaming(diff: &str) -> Result<ParsedPatch, ParseError> {
	let mut tokenizer = Tokenizer::new();
	let mut executor = Executor::new();
	for token in tokenizer.feed(diff)? {
		executor.feed(token)?;
	}
	for token in tokenizer.end() {
		executor.feed(token)?;
	}
	executor.finish_streaming()
}

fn validate_range(
	range: ParsedRange,
	line_num: usize,
	operation: &str,
	register: Option<&str>,
	authored_index: usize,
) -> Result<(), ParseError> {
	if range.end.line < range.start.line {
		let register = register.map_or(String::new(), |name| format!(" @{name}"));
		return Err(ParseError::new(Diagnostic::error(
			DiagnosticCode::InvalidRange,
			Some(line_num),
			Some(authored_index),
			format!(
				"line {line_num}: invalid absolute {operation} range: start {}, end {}. The value \
				 after `{HL_RANGE_SEP}` is an absolute source line, not a count. For one line use \
				 `{}{}{register}`.",
				range.start.line,
				range.end.line,
				if operation == "replace" {
					"PUT "
				} else {
					"CUT "
				},
				range.start.line
			),
		)));
	}
	let span = range.end.line - range.start.line + 1;
	if span > MAX_EXPANDED_RANGE_LINES {
		return Err(ParseError::new(Diagnostic::error(
			DiagnosticCode::RangeTooLarge,
			Some(line_num),
			Some(authored_index),
			format!(
				"line {line_num}: {operation} range spans {span} lines; the maximum is \
				 {MAX_EXPANDED_RANGE_LINES}. Split it into smaller hunks."
			),
		)));
	}
	Ok(())
}

const fn bodyless_target_message(target: &BlockTarget, had_colon: bool) -> Option<&'static str> {
	if target.is_cut() {
		return Some(
			"`CUT` deletes and captures the named lines and takes no body rows. Use `PUT N.=M:` with \
			 `+TEXT` rows to write content.",
		);
	}
	if matches!(target, BlockTarget::Rem | BlockTarget::Move { .. }) {
		return Some("`REM` and `MV DEST` take no body rows.");
	}
	if target.register().is_some() {
		return Some(
			"A register `PUT` pastes captured lines and takes no `+` body rows. Drop `@name` to \
			 write literal text.",
		);
	}
	if !had_colon {
		return Some(
			"`PUT` without `:` is clipboard-backed and takes no body rows. Add `:` after the locator \
			 to write literal content.",
		);
	}
	None
}

const fn complete_bodyless_target(target: &BlockTarget, had_colon: bool) -> bool {
	target.is_cut()
		|| target.register().is_some()
		|| (!had_colon
			&& matches!(
				target,
				BlockTarget::InsertBefore { .. }
					| BlockTarget::InsertAfter { .. }
					| BlockTarget::InsertAfterBlock { .. }
					| BlockTarget::Bof { .. }
					| BlockTarget::Eof { .. }
			))
}

fn gap_target_parts(target: BlockTarget) -> (Cursor, Option<Str>) {
	match target {
		BlockTarget::InsertBefore { anchor, register } => (Cursor::BeforeAnchor { anchor }, register),
		BlockTarget::InsertAfter { anchor, register } => (Cursor::AfterAnchor { anchor }, register),
		BlockTarget::Bof { register } => (Cursor::Bof, register),
		BlockTarget::Eof { register } => (Cursor::Eof, register),
		_ => unreachable!("only gap targets reach gap_target_parts"),
	}
}

fn parse_top_level_bare_range_header(text: &str) -> Option<ParsedRange> {
	let body = text.trim().strip_suffix(':')?.trim_end();
	let (range, had_separator, block) = parse_bare_range(body)?;
	(had_separator && !block).then_some(range)
}

fn parse_bare_range(text: &str) -> Option<(ParsedRange, bool, bool)> {
	let trimmed = text.trim();
	let first_end = trimmed
		.bytes()
		.position(|byte| !byte.is_ascii_digit())
		.unwrap_or(trimmed.len());
	let start = parse_positive(trimmed.get(..first_end)?)?;
	if first_end == trimmed.len() {
		let anchor = Anchor { line: start };
		return Some((ParsedRange { start: anchor, end: anchor }, false, false));
	}
	let rest = &trimmed[first_end..];
	let second_start = rest
		.char_indices()
		.find_map(|(index, ch)| ch.is_ascii_digit().then_some(index))?;
	let separator = &rest[..second_start];
	if separator.is_empty()
		|| !separator
			.chars()
			.all(|ch| ch.is_whitespace() || matches!(ch, '-' | '.' | '=' | '…'))
	{
		return None;
	}
	let end = parse_positive(&rest[second_start..])?;
	Some((ParsedRange { start: Anchor { line: start }, end: Anchor { line: end } }, true, false))
}

fn parse_positive(text: &str) -> Option<usize> {
	const JS_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
	if text.is_empty() || text.starts_with('0') || !text.bytes().all(|byte| byte.is_ascii_digit()) {
		return None;
	}
	let line: u64 = text.parse().ok()?;
	(line <= JS_MAX_SAFE_INTEGER)
		.then(|| usize::try_from(line).ok())
		.flatten()
}

fn parse_top_level_snapshot_row(text: &str) -> Option<(usize, Str)> {
	let trimmed = text.trim_start();
	let separator = trimmed.find([':', '|'])?;
	let line = parse_positive(&trimmed[..separator])?;
	Some((line, trimmed[separator + 1..].into()))
}

fn markdown_bullet(text: &str) -> bool {
	let trimmed = text.trim_start();
	trimmed
		.strip_prefix("- ")
		.and_then(|body| body.chars().next())
		.is_some_and(|ch| !ch.is_whitespace())
}

fn strip_bare_prefixes_if_uniform(payloads: &mut [PayloadRow]) {
	let mut saw_bare = false;
	let mut all_literal_values = true;
	for row in payloads.iter() {
		if !row.bare || row.text.trim().is_empty() {
			continue;
		}
		saw_bare = true;
		let Some(stripped) = strip_one_read_prefix(&row.text) else {
			return;
		};
		all_literal_values &= bare_literal_value(stripped);
	}
	if !saw_bare || all_literal_values {
		return;
	}
	for row in payloads {
		if row.bare
			&& !row.text.trim().is_empty()
			&& let Some(stripped) = strip_one_read_prefix(&row.text)
		{
			row.text = stripped.into();
		}
	}
}

fn strip_one_read_prefix(line: &str) -> Option<&str> {
	let mut rest = line.trim_start();
	if let Some(next) = rest.strip_prefix(">>>").or_else(|| rest.strip_prefix(">>")) {
		rest = next.trim_start();
	}
	if matches!(rest.chars().next(), Some('+' | '*' | '-')) {
		rest = rest[1..].trim_start();
	}
	let digits = rest
		.bytes()
		.take_while(|byte| byte.is_ascii_digit())
		.count();
	if digits == 0 {
		return None;
	}
	let suffix = &rest[digits..];
	if suffix.starts_with(':') || suffix.starts_with('|') {
		Some(&suffix[1..])
	} else {
		None
	}
}

fn bare_literal_value(text: &str) -> bool {
	let trimmed = text.trim();
	let trimmed = trimmed.strip_suffix(',').unwrap_or(trimmed).trim();
	if trimmed.len() >= 2 {
		let bytes = trimmed.as_bytes();
		if matches!(bytes[0], b'\'' | b'"')
			&& bytes[trimmed.len() - 1] == bytes[0]
			&& !trimmed[1..trimmed.len() - 1]
				.bytes()
				.any(|byte| byte == bytes[0])
		{
			return true;
		}
	}
	let numeric = trimmed
		.strip_prefix('-')
		.or_else(|| trimmed.strip_prefix('+'))
		.unwrap_or(trimmed);
	match numeric.split_once('.') {
		Some((whole, fraction)) => {
			!whole.is_empty()
				&& !fraction.is_empty()
				&& whole.bytes().all(|byte| byte.is_ascii_digit())
				&& fraction.bytes().all(|byte| byte.is_ascii_digit())
		},
		None => !numeric.is_empty() && numeric.bytes().all(|byte| byte.is_ascii_digit()),
	}
}

fn is_read_metadata_line(line: &str) -> bool {
	let trimmed = line.trim();
	if matches!(trimmed, "…" | "...") {
		return true;
	}
	if trimmed.starts_with('[')
		&& trimmed.ends_with(']')
		&& (trimmed.contains("Showing lines")
			|| trimmed.contains("more line")
			|| trimmed.contains("ln elided;"))
	{
		return true;
	}
	let Some((prefix, body)) = trimmed.split_once(':') else {
		return false;
	};
	prefix.contains('-')
		&& prefix
			.chars()
			.all(|ch| ch.is_ascii_digit() || ch == '-' || ch.is_whitespace())
		&& (body.contains('…') || body.contains("..."))
}

fn foreign_patch_message(text: &str) -> Option<Str> {
	let trimmed = text.trim_start();
	if ["*** Update File:", "*** Add File:", "*** Delete File:", "*** Move to:"]
		.iter()
		.any(|prefix| trimmed.starts_with(prefix))
	{
		return Some(
			"apply_patch sentinel is not valid in hashline. File sections start with `[path#HASH]`; \
			 use `PUT`, `CUT`, `REM`, or `MV`."
				.into(),
		);
	}
	if trimmed.starts_with("@@") {
		return Some(
			"unified-diff hunk header is not valid in hashline. Drop the `@@ ... @@` wrapper and \
			 write `PUT N.=M:` or `CUT N.=M`."
				.into(),
		);
	}
	if parse_positive(trimmed).is_some() {
		return Some(
			format!(
				"hunk headers need a verb and both endpoints. Use `PUT {trimmed}.={trimmed}:` to \
				 replace or `CUT {trimmed}.={trimmed}` to delete."
			)
			.into(),
		);
	}
	let parts: Vec<_> = trimmed.trim_end_matches(':').split_whitespace().collect();
	if parts.len() == 2 && parts.iter().all(|part| parse_positive(part).is_some()) {
		return Some(
			"bare range hunk header is not valid. Hunk headers need a verb: use `PUT N.=M:` or `CUT \
			 N.=M`."
				.into(),
		);
	}
	None
}

#[cfg(test)]
mod tests {
	use super::*;

	fn codes(parsed: &ParsedPatch) -> Vec<DiagnosticCode> {
		parsed
			.diagnostics
			.iter()
			.map(|diagnostic| diagnostic.code)
			.collect()
	}

	#[test]
	fn parses_ranges_gaps_blocks_registers_and_file_ops() {
		let parsed = parse_patch("PUT 2.=3:\n+A\n+B\nPUT >4 @r\nCUT 6* @block").unwrap();
		assert!(matches!(parsed.edits[0], Edit::Insert { mode: InsertMode::Replacement, .. }));
		assert!(
			parsed
				.edits
				.iter()
				.any(|edit| matches!(edit, Edit::Paste { register: Some(name), .. } if name == "r"))
		);
		assert!(parsed.edits.iter().any(|edit| matches!(edit, Edit::Block { mode: BlockMode::Cut, register: Some(name), .. } if name == "block")));
		assert_eq!(parse_patch("REM").unwrap().file_op, Some(FileOp::Rem));
		assert_eq!(
			parse_patch("MV 'new path.rs'").unwrap().file_op,
			Some(FileOp::Move { dest: "new path.rs".into() })
		);
	}

	#[test]
	fn recovers_bare_bodies_snapshot_rows_and_ranges() {
		let body = parse_patch("PUT 2.=3:\n2:foo\n\n3:bar").unwrap();
		assert!(codes(&body).contains(&DiagnosticCode::BareBodyRecovered));
		assert!(
			body
				.edits
				.iter()
				.any(|edit| matches!(edit, Edit::Insert { text, .. } if text == "foo"))
		);
		let rows = parse_patch("2:B\n4|D").unwrap();
		assert_eq!(rows.edits.len(), 4);
		assert!(codes(&rows).contains(&DiagnosticCode::SnapshotRowRecovered));
		assert!(
			codes(&parse_patch("2..3:\n+X").unwrap()).contains(&DiagnosticCode::BareRangeRecovered)
		);
	}

	#[test]
	fn handles_minus_rows_without_diff_corruption() {
		let diff = parse_patch("PUT 2:\n-old\n+new").unwrap();
		assert!(codes(&diff).contains(&DiagnosticCode::DiffOldRowsIgnored));
		assert!(
			!diff
				.edits
				.iter()
				.any(|edit| matches!(edit, Edit::Insert { text, .. } if text == "-old"))
		);
		let bullets = parse_patch("PUT 2:\n- item\n  - nested").unwrap();
		assert!(codes(&bullets).contains(&DiagnosticCode::MinusBulletRecovered));
		assert_eq!(
			parse_patch("PUT 2:\n-old").unwrap_err().diagnostic.code,
			DiagnosticCode::MinusRowRejected
		);
	}

	#[test]
	fn rejects_contamination_duplicates_and_overlap_with_locations() {
		let contamination = parse_patch("@@ -1,2 +1,2 @@").unwrap_err();
		assert_eq!(contamination.diagnostic.code, DiagnosticCode::ForeignPatchSyntax);
		assert_eq!(contamination.diagnostic.patch_line, Some(1));
		assert_eq!(
			parse_patch("2:B\n2:C").unwrap_err().diagnostic.code,
			DiagnosticCode::DuplicateSnapshotRow
		);
		assert_eq!(
			parse_patch("PUT 2.=4:\n+X\nPUT 3.=5:\n+Y")
				.unwrap_err()
				.diagnostic
				.code,
			DiagnosticCode::OverlappingRange
		);
	}

	#[test]
	fn abort_and_streaming_drop_unfinished_tail() {
		let parsed = parse_patch("PUT >1:\n+yes\n*** Abort\nPUT >9:\n+no").unwrap();
		assert_eq!(parsed.edits.len(), 1);
		assert!(matches!(&parsed.edits[0], Edit::Insert { text, .. } if text == "yes"));
		assert_eq!(parse_patch_streaming("PUT 5.=5:\n").unwrap().edits.len(), 0);
	}
}
