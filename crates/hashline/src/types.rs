//! Pure domain types shared by hashline parsing and later application stages.

use omp_core::Str;

/// A one-indexed source line anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Anchor {
	/// The one-indexed source line.
	pub line: usize,
}

/// A stable insertion gap relative to the pre-edit source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cursor {
	/// The beginning of the file.
	Bof,
	/// The end of the file.
	Eof,
	/// The gap immediately before an anchor.
	BeforeAnchor {
		/// The source row immediately after the gap.
		anchor: Anchor,
	},
	/// The gap immediately after an anchor.
	AfterAnchor {
		/// The source row immediately before the gap.
		anchor: Anchor,
	},
}

/// An inclusive source-line range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParsedRange {
	/// The first source line.
	pub start: Anchor,
	/// The last source line.
	pub end:   Anchor,
}

/// A clipboard paste destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PasteTarget {
	/// Insert into a source gap.
	Gap {
		/// The insertion cursor naming the gap.
		cursor: Cursor,
	},
	/// Replace an inclusive source span.
	Span {
		/// The inclusive source range replaced by the paste.
		range: ParsedRange,
	},
}

/// The semantic role of an inserted row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InsertMode {
	/// A literal insertion that consumes no source row.
	Literal,
	/// New content paired with a source replacement.
	Replacement,
}

/// The operation deferred until a syntactic block can be resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockMode {
	/// Replace the resolved block.
	Replace,
	/// Insert after the resolved block.
	InsertAfter,
	/// Capture and remove the resolved block.
	Cut,
	/// Paste a register after the resolved block.
	PasteAfter,
}

/// A resolved one-indexed inclusive syntactic-block span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockSpan {
	/// The first line of the block.
	pub start: usize,
	/// The last line of the block.
	pub end:   usize,
}

/// One block operation mapped to its concrete source span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockResolution {
	/// The line on which the block locator was authored.
	pub anchor_line: usize,
	/// The resolved first source line.
	pub start:       usize,
	/// The resolved last source line.
	pub end:         usize,
	/// The deferred operation that produced the resolution.
	pub mode:        BlockMode,
}

/// A non-fatal repair or fallback reported while applying a patch.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ApplyWarning {
	/// Replacement rows were re-indented to match their source block.
	#[error(
		"line {line}: auto-indented {bodies} replacement body line(s) to preserve the surrounding \
		 block depth. Re-issue the replacement body with indentation matching the selected source \
		 block."
	)]
	ReplacementIndentationRepaired {
		/// The authored patch line containing the replacement.
		line:   usize,
		/// The number of non-empty body rows re-indented.
		bodies: usize,
	},
	/// An after-line insertion was moved past structural closers.
	#[error(
		"shifted an after-line insertion from line {from} to line {to} across {crossed} structural \
		 closer line(s) so it lands at the surrounding block depth. Use an explicit adjacent gap if \
		 the insertion belongs inside those closers."
	)]
	AfterLineLandingShifted {
		/// The originally selected source line.
		from:    usize,
		/// The repaired source line.
		to:      usize,
		/// The number of structural closer rows crossed.
		crossed: usize,
	},
	/// Exact body rows duplicated outside a replacement were removed.
	#[error(
		"auto-repaired a replacement boundary echo at line {line}: dropped {leading} leading and \
		 {trailing} trailing body line(s) already present outside the range. Issue the body as \
		 final content for the selected range only."
	)]
	BoundaryEchoDropped {
		/// The first selected source line.
		line:     usize,
		/// The number of duplicated leading body rows removed.
		leading:  usize,
		/// The number of duplicated trailing body rows removed.
		trailing: usize,
	},
	/// Syntax-essential selected boundary rows were retained.
	#[error(
		"auto-repaired replacement boundaries at line {line}: retained {kept} syntax-essential \
		 source boundary row(s) selected by the range. The syntax probe verified the result; \
		 re-issue with the range covering exactly the changed lines and the body as their complete \
		 final content."
	)]
	BoundaryRowsRetained {
		/// The first selected source line.
		line: usize,
		/// The number of selected boundary rows retained.
		kept: usize,
	},
	/// Body rows duplicated outside a replacement were removed after syntax
	/// probing.
	#[error(
		"auto-repaired replacement boundaries at line {line}: dropped {dropped} body row(s) \
		 duplicated just outside the range. The syntax probe verified the result; re-issue with the \
		 range covering exactly the changed lines and the body as their complete final content."
	)]
	BoundaryRowsDropped {
		/// The first selected source line.
		line:    usize,
		/// The number of duplicated body rows removed.
		dropped: usize,
	},
	/// Selected boundary rows were retained and duplicated body rows removed.
	#[error(
		"auto-repaired replacement boundaries at line {line}: retained {kept} syntax-essential \
		 source boundary row(s) selected by the range and dropped {dropped} body row(s) duplicated \
		 just outside the range. The syntax probe verified the result; re-issue with the range \
		 covering exactly the changed lines and the body as their complete final content."
	)]
	BoundaryRowsRetainedAndDropped {
		/// The first selected source line.
		line:    usize,
		/// The number of selected boundary rows retained.
		kept:    usize,
		/// The number of duplicated body rows removed.
		dropped: usize,
	},
	/// A patch caused a previously valid file to fail its syntax probe.
	#[error(
		"this edit introduced a syntax error near line {line}: the file parsed before the patch and \
		 no longer does. It was applied exactly as written, so a line number or range endpoint is \
		 likely wrong; re-read the touched region and re-issue a correcting edit."
	)]
	SyntaxBreak {
		/// The path whose syntax probe failed, when language inference used one.
		path: Option<Str>,
		/// The first source line changed by the patch.
		line: usize,
	},
	/// A block operation fell back to a plain after-line insertion.
	#[error(
		"line {patch_line}: block at line {block_line} could not be resolved; lowered to after-line \
		 insertion (anchor is a structural closer: {structural_closer}). Re-read the region and use \
		 a plain after-line insertion if this fallback is intended."
	)]
	BlockLoweringFallback {
		/// The authored patch line containing the block operation.
		patch_line:        usize,
		/// The source line used as the block anchor.
		block_line:        usize,
		/// Whether the anchor consists only of structural closers.
		structural_closer: bool,
	},
	/// A paste from an empty named register inserted no rows.
	#[error(
		"line {patch_line}: register @{register} is empty; pasted nothing. Populate the register \
		 with a named CUT before pasting it."
	)]
	EmptyRegisterPaste {
		/// The authored patch line containing the paste.
		patch_line: usize,
		/// The empty register name without the `@` prefix.
		register:   Str,
	},
}

impl ApplyWarning {
	/// Returns the stable machine-readable diagnostic code.
	#[must_use]
	pub fn code(&self) -> &'static str {
		self.into()
	}

	/// Returns the authored patch line this warning was reported against, when
	/// the warning names one.
	///
	/// Distinct from [`ApplyWarning::source_line`]: hashline addresses both
	/// patch-language rows and source rows, and a warning names at most one of
	/// the two coordinate spaces.
	#[must_use]
	pub const fn patch_line(&self) -> Option<usize> {
		match self {
			Self::ReplacementIndentationRepaired { line, .. } => Some(*line),
			Self::BlockLoweringFallback { patch_line, .. }
			| Self::EmptyRegisterPaste { patch_line, .. } => Some(*patch_line),
			Self::AfterLineLandingShifted { .. }
			| Self::BoundaryEchoDropped { .. }
			| Self::BoundaryRowsRetained { .. }
			| Self::BoundaryRowsDropped { .. }
			| Self::BoundaryRowsRetainedAndDropped { .. }
			| Self::SyntaxBreak { .. } => None,
		}
	}

	/// Returns the source line this warning refers to, when the warning names
	/// one.
	///
	/// For [`ApplyWarning::AfterLineLandingShifted`] this is the repaired
	/// landing, not the originally authored anchor.
	#[must_use]
	pub const fn source_line(&self) -> Option<usize> {
		match self {
			Self::AfterLineLandingShifted { to, .. } => Some(*to),
			Self::BoundaryEchoDropped { line, .. }
			| Self::BoundaryRowsRetained { line, .. }
			| Self::BoundaryRowsDropped { line, .. }
			| Self::BoundaryRowsRetainedAndDropped { line, .. }
			| Self::SyntaxBreak { line, .. } => Some(*line),
			Self::BlockLoweringFallback { block_line, .. } => Some(*block_line),
			Self::ReplacementIndentationRepaired { .. } | Self::EmptyRegisterPaste { .. } => None,
		}
	}
}

/// Optional lexical path hints used while splitting patch input.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SplitOptions {
	/// A working directory used to shorten absolute paths within it.
	pub cwd:  Option<Str>,
	/// A fallback path for headerless input containing recognizable operations.
	pub path: Option<Str>,
}

/// A low-level edit emitted in authored order by the parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Edit {
	/// Insert one row at a stable cursor.
	Insert {
		/// The pre-edit insertion cursor.
		cursor:      Cursor,
		/// The row text without a line terminator.
		text:        Str,
		/// The authored patch-language line.
		line_num:    usize,
		/// The authored edit sequence index.
		index:       usize,
		/// Whether this is literal insertion or replacement content.
		mode:        InsertMode,
		/// The resolved block start, populated by later block lowering.
		block_start: Option<usize>,
	},
	/// Delete one anchored source row.
	Delete {
		/// The row to delete.
		anchor:   Anchor,
		/// The authored patch-language line.
		line_num: usize,
		/// The authored edit sequence index.
		index:    usize,
	},
	/// Capture an inclusive span into a clipboard register.
	Cut {
		/// The span captured before deletion.
		range:    ParsedRange,
		/// A named register, or the anonymous register when absent.
		register: Option<Str>,
		/// The authored patch-language line.
		line_num: usize,
		/// The authored edit sequence index.
		index:    usize,
	},
	/// Paste a clipboard register at a gap or over a span.
	Paste {
		/// The paste destination.
		at:          PasteTarget,
		/// A named register, or the anonymous register when absent.
		register:    Option<Str>,
		/// The authored patch-language line.
		line_num:    usize,
		/// The authored edit sequence index.
		index:       usize,
		/// The resolved block start, populated by later block lowering.
		block_start: Option<usize>,
	},
	/// An edit whose concrete span requires syntax-aware block resolution.
	Block {
		/// The authored block opener line.
		anchor:   Anchor,
		/// Literal replacement or insertion rows.
		payloads: Vec<Str>,
		/// The block operation.
		mode:     BlockMode,
		/// A named register, or the anonymous register when absent.
		register: Option<Str>,
		/// The authored patch-language line.
		line_num: usize,
		/// The authored edit sequence index.
		index:    usize,
	},
}

impl Edit {
	/// Returns the patch-language line that authored this edit.
	pub const fn line_num(&self) -> usize {
		match self {
			Self::Insert { line_num, .. }
			| Self::Delete { line_num, .. }
			| Self::Cut { line_num, .. }
			| Self::Paste { line_num, .. }
			| Self::Block { line_num, .. } => *line_num,
		}
	}

	/// Returns this edit's authored sequence index.
	pub const fn index(&self) -> usize {
		match self {
			Self::Insert { index, .. }
			| Self::Delete { index, .. }
			| Self::Cut { index, .. }
			| Self::Paste { index, .. }
			| Self::Block { index, .. } => *index,
		}
	}
}

/// A whole-file operation parsed from a section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileOp {
	/// Remove the section's file.
	Rem,
	/// Move the section's file to a destination path.
	Move {
		/// The authored destination path.
		dest: Str,
	},
}

/// The severity of a parser diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticSeverity {
	/// A rejected patch construct.
	Error,
	/// A lenient recovery that should be surfaced to the author.
	Warning,
}

/// A stable machine-readable parser diagnostic category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticCode {
	/// A line anchor or range endpoint is malformed.
	InvalidLocator,
	/// An absolute range ends before it starts.
	InvalidRange,
	/// A range exceeds the parser's expansion bound.
	RangeTooLarge,
	/// A body row has no hunk header.
	OrphanPayload,
	/// Input from another patch grammar was detected.
	ForeignPatchSyntax,
	/// A body-bearing operation received no rows.
	MissingBody,
	/// A bodyless operation received rows.
	BodyNotAllowed,
	/// A register paste used a body colon.
	RegisterColon,
	/// A colonless span paste omitted a named register.
	AnonymousSpanPaste,
	/// A bare minus row was ambiguous or invalid.
	MinusRowRejected,
	/// A bare body row was recovered as literal content.
	BareBodyRecovered,
	/// Bare Markdown bullet rows were recovered as literals.
	MinusBulletRecovered,
	/// Unified-diff old rows were discarded.
	DiffOldRowsIgnored,
	/// A bare range was recovered as a `PUT` header.
	BareRangeRecovered,
	/// A top-level snapshot row was recovered as a replacement.
	SnapshotRowRecovered,
	/// A copied read-output metadata row was ignored.
	ReadMetadataIgnored,
	/// A snapshot row repeated a source line.
	DuplicateSnapshotRow,
	/// A literal body row itself resembles an operation header.
	LiteralOperationRow,
	/// An empty replacement was recovered as deletion.
	EmptyPutRecovered,
	/// A trailing colon on `CUT` was ignored.
	CutColonIgnored,
	/// Two concrete hunks target overlapping source lines.
	OverlappingRange,
	/// An exact duplicate target was normalized to its final hunk.
	DuplicateRangeCoalesced,
	/// File-level operations conflict with one another or line edits.
	FileOperationConflict,
	/// Clipboard order became ambiguous after interleaved section merging.
	InterleavedClipboard,
	/// A section header is malformed or missing.
	InvalidSectionHeader,
	/// Same-path sections carry conflicting snapshot tags.
	ConflictingSnapshotTags,
}

/// A structured parser diagnostic retaining authored location metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
	/// The stable diagnostic category.
	pub code:           DiagnosticCode,
	/// Whether the diagnostic rejected or recovered input.
	pub severity:       DiagnosticSeverity,
	/// The one-indexed patch-language line, when available.
	pub patch_line:     Option<usize>,
	/// The authored hunk/edit index, when available.
	pub authored_index: Option<usize>,
	/// The human-readable explanation and repair guidance.
	pub message:        Str,
}

impl Diagnostic {
	/// Constructs an error diagnostic at an optional authored location.
	pub fn error(
		code: DiagnosticCode,
		patch_line: Option<usize>,
		authored_index: Option<usize>,
		message: impl Into<Str>,
	) -> Self {
		Self {
			code,
			severity: DiagnosticSeverity::Error,
			patch_line,
			authored_index,
			message: message.into(),
		}
	}

	/// Constructs a warning diagnostic at an optional authored location.
	pub fn warning(
		code: DiagnosticCode,
		patch_line: Option<usize>,
		authored_index: Option<usize>,
		message: impl Into<Str>,
	) -> Self {
		Self {
			code,
			severity: DiagnosticSeverity::Warning,
			patch_line,
			authored_index,
			message: message.into(),
		}
	}
}

/// A fatal parse failure with a structured diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{}", diagnostic.message)]
pub struct ParseError {
	/// The diagnostic describing the rejected construct.
	pub diagnostic: Diagnostic,
}

impl ParseError {
	/// Constructs a parse error from an error diagnostic.
	pub const fn new(diagnostic: Diagnostic) -> Self {
		Self { diagnostic }
	}
}

/// The parser output for one section body.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParsedPatch {
	/// Low-level edits in authored order.
	pub edits:       Vec<Edit>,
	/// An optional whole-file operation.
	pub file_op:     Option<FileOp>,
	/// Leniency and contamination-recovery warnings.
	pub diagnostics: Vec<Diagnostic>,
}
