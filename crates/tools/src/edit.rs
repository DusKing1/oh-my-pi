//! Streaming hashline edits over one revision-pinned multi-document
//! transaction.

pub mod projection;

use std::{fmt::Write as _, future::Future};

use async_stream::stream;
use bytes::Bytes;
use futures::{FutureExt, Stream, pin_mut, select_biased};
use omp_core::Str;
use omp_hashline::{
	ApplyMode, ApplyOptions, BlockMode, Clipboard, FileOp, MismatchDetails, MismatchError, Patch,
	apply_parsed_patch, compute_snapshot_tag,
	diff_preview::{CompactDiffOptions, build_compact_diff_preview},
	format_hashline_header, numbered_diff,
	recovery::{ByteRange, RecoveryEdit, recover_exact},
};
use omp_tool::{
	Abort, ArgIssue, ArgIssueKind, ArgPath, CommitError, Constraint, Ev, IncomingParams,
	InterruptWaitError, Outcome, ParamError, Part, PromptCaps, Rev, Tool, ToolSpec,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::render::TextProjection;

const DESCRIPTION: &str = include_str!("edit_prompt.txt");

/// Streaming arguments for `edit@hl.1`.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[schemars(description = "")]
#[serde(deny_unknown_fields)]
pub struct Params {
	/// Complete hashline input, including every `[PATH#TAG]` section header.
	#[schemars(description = "", with = "String")]
	pub input: Str,
}

/// A dry-run projection emitted whenever another complete section becomes
/// applicable.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EditUpdate {
	/// Number of parsed low-level operations currently applied.
	pub applied_ops:   usize,
	/// Compact, numbered preview of the current candidate.
	pub preview:       Str,
	/// Added rows represented by the preview source diff.
	pub added_lines:   usize,
	/// Removed rows represented by the preview source diff.
	pub removed_lines: usize,
}

/// One durable applied hashline operation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AppliedOp {
	/// Stable operation family.
	pub kind:       Str,
	/// One-indexed line in the submitted section body.
	pub patch_line: usize,
	/// Authored operation sequence index.
	pub index:      usize,
}

/// The durable operation performed for one section.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SectionOp {
	/// Existing file content changed in place.
	Update,
	/// The section parsed and applied but changed no bytes.
	Noop,
	/// The file was removed.
	Delete,
	/// The file was moved, optionally after its content changed.
	Move,
}

/// One syntax-aware block resolution retained for projection.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResolvedBlock {
	/// Authored source anchor.
	pub anchor_line: usize,
	/// Resolved first source line.
	pub start:       usize,
	/// Resolved last source line.
	pub end:         usize,
	/// Stable operation label used by the renderer.
	pub operation:   Str,
}

/// Durable successful truth for one file section.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SectionPayload {
	/// Authored source path.
	pub path:               Str,
	/// Canonical source path used for duplicate and snapshot checks.
	pub canonical_path:     Str,
	/// Operation performed by this section.
	pub op:                 SectionOp,
	/// Move destination when `op` is [`SectionOp::Move`].
	pub move_dest:          Option<Str>,
	/// Pinned document base revision.
	pub old_revision:       Str,
	/// Committed target revision, absent after deletion.
	pub new_revision:       Option<Str>,
	/// Sequence of applied operations.
	pub applied_ops:        Vec<AppliedOp>,
	/// Whether the document host rebased the committed transition.
	pub rebased:            bool,
	/// Exact pre-edit bytes.
	pub before:             Bytes,
	/// Exact post-edit bytes, empty after deletion.
	pub after:              Bytes,
	/// Hashline header for the resulting file, absent after deletion.
	pub header:             Option<Str>,
	/// Complete numbered diff.
	pub diff:               Str,
	/// Compact current-file preview.
	pub preview:            Str,
	/// First changed source line when known.
	pub first_changed_line: Option<usize>,
	/// Syntax-aware block resolutions.
	pub block_resolutions:  Vec<ResolvedBlock>,
	/// Parser and application warnings in authored order.
	pub warnings:           Vec<Str>,
}

/// Durable successful multi-file transaction truth.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Payload {
	/// Results in authored section order.
	pub sections: Vec<SectionPayload>,
}

/// Formatting requested of the document transaction coordinator.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FormatPolicy {
	/// Apply the configured formatter when one is available.
	Configured,
}

/// Stale-base behavior requested of the transaction coordinator.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StalePolicy {
	/// Rebase only edits whose base spans do not overlap intervening changes.
	RebaseNonOverlapping,
}

/// Facts needed to prepare one authored section against shared session state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrepareRequest {
	/// Authored path from the section header.
	pub path:         Str,
	/// Four-hex snapshot tag from the section header.
	pub file_hash:    Option<Str>,
	/// Concrete one-indexed anchors used for stale mismatch context.
	pub anchor_lines: Vec<usize>,
}

/// Borrowed view exposed by an opaque, revision-pinned prepared lease.
pub trait EditPrepared: Send + Sync {
	/// Canonical path pinned by this lease.
	fn path(&self) -> &Str;
	/// Model-facing path after any tag-based path recovery.
	fn display_path(&self) -> &Str {
		self.path()
	}
	/// Opaque pinned live revision.
	fn base_revision(&self) -> &Str;
	/// Exact bytes at the pinned live revision.
	fn base_bytes(&self) -> &Bytes;
	/// Non-fatal preparation diagnostics such as tag-based path recovery.
	fn warnings(&self) -> &[Str] {
		&[]
	}
	/// Exact retained bytes named by the authored tag, or live bytes when
	/// untagged.
	fn authored_bytes(&self) -> &Bytes;
}

/// The final filesystem transition for one prepared section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EditAction {
	/// Replace the source file with these exact bytes.
	Write {
		/// Final file contents.
		content: Bytes,
	},
	/// Remove the source file.
	Delete,
	/// Move the source identity and persist the supplied final bytes.
	Move {
		/// New path for the source identity.
		destination: Str,
		/// Final contents persisted at the destination.
		content:     Bytes,
	},
}

/// One fully preflighted proposal in authored section order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditProposal {
	/// Final filesystem transition.
	pub action:        EditAction,
	/// Pinned base revision string.
	pub base_revision: Str,
	/// Configured stale-base handling policy.
	pub stale_policy:  StalePolicy,
	/// Configured code formatting policy.
	pub format_policy: FormatPolicy,
}

/// Resource-owned commit result for one authored section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommittedSection {
	/// Committed target revision, absent after deletion.
	pub new_revision: Option<Str>,
	/// Whether the resource rebased this section.
	pub rebased:      bool,
	/// Exact committed view bytes after formatting, absent after deletion.
	pub content:      Option<Bytes>,
}

/// Structured successful response from the atomic transaction owner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommitResult {
	/// Results in authored section order.
	pub sections: Vec<CommittedSection>,
}

/// One conflicting base/current range retained from transaction rejection.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Conflict {
	/// One-based starting line number of the conflicting range.
	pub start_line: usize,
	/// One-based ending line number of the conflicting range.
	pub end_line:   usize,
	/// Explanation of the line-range conflict.
	pub message:    Str,
}

/// Typed transaction rejection reason.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RejectionReason {
	/// Edit collided with intervening workspace edits.
	Conflict,
	/// Base revision is stale and cannot be automatically recovered.
	StaleUnrecoverable {
		/// Exact stale-snapshot diagnostic.
		message: Str,
	},
	/// Formatter execution failed on the edited document.
	Format {
		/// Exact formatter diagnostic.
		message: Str,
	},
	/// Submitted patch syntax or structure was invalid.
	InvalidPatch {
		/// Exact patch diagnostic.
		message: Str,
	},
}

/// Durable typed edit failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Fault {
	/// Transaction rejection classification.
	pub reason:    RejectionReason,
	/// List of conflicting line ranges, if applicable.
	pub conflicts: Vec<Conflict>,
}

impl Fault {
	fn invalid(message: impl Into<Str>) -> Self {
		Self {
			reason:    RejectionReason::InvalidPatch { message: message.into() },
			conflicts: Vec::new(),
		}
	}

	fn stale(message: impl Into<Str>) -> Self {
		Self {
			reason:    RejectionReason::StaleUnrecoverable { message: message.into() },
			conflicts: Vec::new(),
		}
	}
}

/// Session loop-guard result for one byte-identical edit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NoopResult {
	/// Exact pi-compatible soft or hard diagnostic.
	pub diagnostic: Str,
	/// Whether this attempt reached the mandatory hard-failure threshold.
	pub escalate:   bool,
}

/// Truthful resource-owned commit failure classification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EditCommitError {
	/// Atomic transaction rejection; no section or clipboard state landed.
	Rejected(Fault),
	/// The resource cannot prove whether effects landed.
	EffectsUnknown {
		/// Why the document owner cannot determine the final state.
		reason: Str,
	},
}

/// Resource boundary implemented by the environment's document host.
pub trait EditDocuments: Send + Sync + 'static {
	/// Concrete owner of one pinned resource; values are moved together to
	/// commit.
	type Prepared: EditPrepared;

	/// Opens a section and resolves its authored snapshot through session state.
	fn prepare(
		&self,
		request: PrepareRequest,
	) -> impl Future<Output = Result<Self::Prepared, Fault>> + Send + '_;

	/// Starts a call-local clipboard retaining named session registers only.
	fn start_clipboard_batch(&self) -> Clipboard;

	/// Records one byte-identical result under canonical identity.
	fn record_noop(&self, canonical_path: &str, display_path: &str, input: Bytes) -> NoopResult;

	/// Clears one path's no-op streak after a real commit.
	fn reset_noop(&self, canonical_path: &str);

	/// Commits every proposal as one resource transaction and publishes named
	/// registers only if that transaction commits completely.
	fn commit<'a>(
		&'a self,
		prepared: Vec<&'a mut Self::Prepared>,
		proposals: Vec<EditProposal>,
		clipboard: Clipboard,
	) -> impl Future<Output = Result<CommitResult, EditCommitError>> + Send + 'a;
}

/// `edit@hl.1` executor.
pub struct EditTool<D> {
	documents:     D,
	format_policy: FormatPolicy,
	spec:          ToolSpec,
}

/// Constructs the built-in hashline edit tool.
pub fn tool<D: EditDocuments>(documents: D, format_policy: FormatPolicy) -> EditTool<D> {
	EditTool {
		documents,
		format_policy,
		spec: ToolSpec {
			name:        "edit".into(),
			rev:         Rev { family: "hl".into(), n: 1 },
			description: DESCRIPTION.into(),
			schema:      omp_tool::schema::<Params>(),
			constraint:  Constraint::Schema { priority: 100 },
		},
	}
}

impl<D: EditDocuments> Tool for EditTool<D> {
	type Fault = Fault;
	type Params = Params;
	type Payload = Payload;
	type Update = EditUpdate;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'c>(
		&'c self,
		mut params: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<EditUpdate, Payload, Fault>> + Send + 'c {
		stream! {
			let Params { input } = match params.whole::<Params>().await {
				Ok(params) => params,
				Err(error) => { yield param_event(error); return; },
			};

			if input.trim().is_empty() {
				yield done_fault(Fault::invalid("No hashline sections found in input."));
				return;
			}

			let patch = match Patch::parse_default(&input) {
				Ok(patch) if !patch.sections.is_empty() => patch,
				Ok(_) => {
					yield done_fault(Fault::invalid("No hashline sections found in input."));
					return;
				},
				Err(error) => {
					yield Ev::Args(ArgIssue {
						path: vec![ArgPath::Key("input".into())],
						expected: "complete hashline input beginning with [PATH#TAG]".into(),
						kind: ArgIssueKind::Malformed,
						example: Some("[src/a.rs#1A2B]\nPUT 1.=1:\n+replacement".into()),
						found: Some(error.to_string().into()),
					});
					return;
				},
			};

			let mut parsed_sections = Vec::with_capacity(patch.sections.len());
			for section in patch.sections {
				let parsed = match section.parse() {
					Ok(parsed) => parsed,
					Err(error) => { yield done_fault(Fault::invalid(error.to_string())); return; },
				};
				let anchors = match section.collect_anchor_lines() {
					Ok(anchors) => anchors,
					Err(error) => { yield done_fault(Fault::invalid(error.to_string())); return; },
				};
				let request = PrepareRequest {
					path: section.path.clone(), file_hash: section.file_hash.clone(), anchor_lines: anchors,
				};
				let prepared = match self.documents.prepare(request).await {
					Ok(prepared) => prepared,
					Err(fault) => { yield done_fault(fault); return; },
				};
				if let Some(previous) = parsed_sections.iter().find(|entry: &&PreparedWork<D::Prepared>| entry.prepared.path() == prepared.path()) {
					yield done_fault(Fault::invalid(format!(
						"Multiple hashline sections resolve to the same file ({} and {}). Merge their ops under one header before applying.",
						previous.section_path, section.path
					)));
					return;
				}
				parsed_sections.push(PreparedWork {
					section_path: prepared.display_path().clone(),
					file_hash: section.file_hash,
					parsed,
					prepared,
				});
			}

			let mut clipboard = self.documents.start_clipboard_batch();
			let mut proposals = Vec::with_capacity(parsed_sections.len());
			let mut projections = Vec::with_capacity(parsed_sections.len());
			for work in &parsed_sections {
				let applied = match apply_parsed_patch(
					work.prepared.authored_bytes().clone(), &work.parsed, &mut clipboard,
					ApplyOptions { mode: ApplyMode::Strict, path: Some(work.prepared.path()) },
				) {
					Ok(applied) => applied,
					Err(error) => { yield done_fault(Fault::invalid(error.to_string())); return; },
				};

				let after = if work.prepared.authored_bytes() == work.prepared.base_bytes() {
					applied.bytes.clone()
				} else if applied.edits.is_empty() {
					let message = stale_message(work, true);
					yield done_fault(Fault::stale(message));
					return;
				} else {
					let recovery_edits = match recovery_edits(&applied.edits) {
						Ok(edits) => edits,
						Err(fault) => { yield done_fault(fault); return; },
					};
					if let Ok(recovered) = recover_exact(work.prepared.authored_bytes(), work.prepared.base_bytes(), &recovery_edits) { recovered.content().clone() } else {
								  yield done_fault(Fault::stale(stale_message(work, true)));
								  return;
							  }
				};

				let action = match &work.parsed.file_op {
					Some(FileOp::Rem) => EditAction::Delete,
					Some(FileOp::Move { dest }) => EditAction::Move {
						destination: dest.clone(), content: after.clone(),
					},
					None => EditAction::Write { content: after.clone() },
				};
				proposals.push(EditProposal {
					action, base_revision: work.prepared.base_revision().clone(),
					stale_policy: StalePolicy::RebaseNonOverlapping, format_policy: self.format_policy,
				});
				projections.push(ProjectionWork {
					after,
					applied_ops: op_details(&work.parsed.edits),
					first_changed_line: applied.first_changed_line,
					block_resolutions: applied.block_resolutions.into_iter().map(|resolution| ResolvedBlock {
						anchor_line: resolution.anchor_line, start: resolution.start, end: resolution.end,
						operation: match resolution.mode {
							BlockMode::Replace => "replace", BlockMode::InsertAfter => "insert_after",
							BlockMode::Cut => "cut", BlockMode::PasteAfter => "paste_after",
						}.into(),
					}).collect(),
					warnings: work.prepared.warnings().iter().cloned()
						.chain(work.parsed.diagnostics.iter().map(|warning| warning.message.clone()))
						.chain(applied.warnings.iter().map(|warning| warning.to_string().into())).collect(),
				});
			}

			let mut preview = String::new();
			let mut added_lines = 0;
			let mut removed_lines = 0;
			for (work, projection) in parsed_sections.iter().zip(&projections) {
				let Ok(diff) = numbered_diff(
					work.prepared.base_bytes(),
					&projection.after,
					Some(std::path::Path::new(work.section_path.as_str())),
				) else {
					continue;
				};
				let compact =
					build_compact_diff_preview(diff.text.as_str(), CompactDiffOptions::default());
				if !preview.is_empty() && !compact.preview.is_empty() {
					preview.push('\n');
				}
				preview.push_str(compact.preview.as_str());
				added_lines += compact.added_lines;
				removed_lines += compact.removed_lines;
			}
			yield Ev::Update(EditUpdate {
				applied_ops: projections.iter().map(|projection| projection.applied_ops.len()).sum(),
				preview: preview.into(),
				added_lines,
				removed_lines,
			});

			match params.committed().await {
				Ok(_) => {},
				Err(error) => { yield commit_event(error); return; },
			}

			let noop_index = parsed_sections.iter().zip(&projections).position(|(work, projection)| {
				work.parsed.file_op.is_none() && work.prepared.base_bytes() == &projection.after
			});
			if parsed_sections.len() == 1 && noop_index.is_some() {
				let work = &parsed_sections[0];
				let noop = self.documents.record_noop(
					work.prepared.path(), &work.section_path,
					Bytes::copy_from_slice(input.as_bytes()),
				);
				if noop.escalate {
					yield done_fault(Fault::invalid(noop.diagnostic));
				} else {
					let payload = build_payload(&parsed_sections, &projections, None);
					yield Ev::Done(Outcome::Done { result: Ok(payload), useless: true });
				}
				return;
			}
			if let Some(index) = noop_index {
				let work = &parsed_sections[index];
				let noop = self.documents.record_noop(
					work.prepared.path(), &work.section_path,
					Bytes::copy_from_slice(input.as_bytes()),
				);
				yield done_fault(Fault::invalid(noop.diagnostic));
				return;
			}

			let result = {
				let prepared =
					parsed_sections.iter_mut().map(|work| &mut work.prepared).collect();
				let commit = self.documents.commit(prepared, proposals, clipboard).fuse();
				let interrupt = params.next_interrupt().fuse();
				pin_mut!(commit, interrupt);
				let result = select_biased! {
					result = commit => Some(result),
					interrupted = interrupt => {
						yield Ev::Aborted(match interrupted {
							Ok(value) => Abort::EffectsUnknown { reason: value.reason },
							Err(InterruptWaitError::Closed) => Abort::EffectsUnknown { reason: "invocation owner disappeared during transaction".into() },
							Err(InterruptWaitError::Protocol(reason)) => Abort::EffectsUnknown { reason },
						});
						None
					},
				};
				result
			};
			let Some(result) = result else { return; };
			match result {
				Ok(result) if result.sections.len() == parsed_sections.len() => {
					for work in &parsed_sections {
						self.documents.reset_noop(work.prepared.path());
					}
					let payload = build_payload(&parsed_sections, &projections, Some(&result.sections));
					yield Ev::Done(Outcome::Done { result: Ok(payload), useless: false });
				},
				Ok(_) => yield Ev::Aborted(Abort::EffectsUnknown { reason: "document transaction returned the wrong section count".into() }),
				Err(EditCommitError::Rejected(fault)) => yield done_fault(fault),
				Err(EditCommitError::EffectsUnknown { reason }) => yield Ev::Aborted(Abort::EffectsUnknown { reason }),
			}
		}
	}

	fn prompt(&self, view: Result<&Payload, &Fault>, caps: &PromptCaps) -> Vec<Part> {
		let Some(mut out) = TextProjection::new(*caps) else {
			return Vec::new();
		};
		match view {
			Ok(payload) => {
				let rendered = payload
					.sections
					.iter()
					.map(|section| {
						let noop_diagnostic = format!(
							"Edits to {} parsed and applied cleanly, but produced no change: your body \
							 row(s) are byte-identical to the file at the targeted lines. The bug is \
							 somewhere else — re-read the file before issuing another edit. Do NOT widen \
							 the payload or add lines; verify the anchor first.",
							section.path
						);
						projection::render_section(projection::SectionView {
							op:                match section.op {
								SectionOp::Delete => projection::SectionOp::Delete,
								SectionOp::Noop => projection::SectionOp::Noop,
								SectionOp::Update | SectionOp::Move => projection::SectionOp::Update,
							},
							path:              &section.path,
							header:            section.header.as_deref().unwrap_or_default(),
							noop_diagnostic:   &noop_diagnostic,
							move_dest:         section.move_dest.as_deref(),
							preview:           &section.preview,
							block_resolutions: &section.block_resolutions,
							warnings:          &section.warnings,
						})
					})
					.collect::<Vec<_>>();
				let _ = out.push(&projection::render_sections(&rendered));
			},
			Err(fault) => {
				let _ = out.push(&rejection_text(fault));
			},
		}
		out.finish()
	}
}

struct PreparedWork<P> {
	section_path: Str,
	file_hash:    Option<Str>,
	parsed:       omp_hashline::ParsedPatch,
	prepared:     P,
}

struct ProjectionWork {
	after:              Bytes,
	applied_ops:        Vec<AppliedOp>,
	first_changed_line: Option<usize>,
	block_resolutions:  Vec<ResolvedBlock>,
	warnings:           Vec<Str>,
}

fn recovery_edits(edits: &[omp_hashline::ByteEdit]) -> Result<Vec<RecoveryEdit>, Fault> {
	edits
		.iter()
		.map(|edit| {
			let start =
				u64::try_from(edit.start).map_err(|_| Fault::invalid("edit byte offset overflow"))?;
			let end =
				u64::try_from(edit.end).map_err(|_| Fault::invalid("edit byte offset overflow"))?;
			let range =
				ByteRange::new(start, end).map_err(|error| Fault::invalid(error.to_string()))?;
			Ok(RecoveryEdit::new(range, edit.replacement.clone()))
		})
		.collect()
}

fn stale_message<P: EditPrepared>(work: &PreparedWork<P>, recognized: bool) -> Str {
	let lines = String::from_utf8_lossy(work.prepared.base_bytes())
		.lines()
		.map(Str::from)
		.collect();
	let anchors = work
		.parsed
		.edits
		.iter()
		.filter_map(|edit| match edit {
			omp_hashline::Edit::Delete { anchor, .. } | omp_hashline::Edit::Block { anchor, .. } => {
				Some(anchor.line)
			},
			omp_hashline::Edit::Cut { range, .. } => Some(range.start.line),
			omp_hashline::Edit::Paste { .. } | omp_hashline::Edit::Insert { .. } => None,
		})
		.collect();
	MismatchError::new(MismatchDetails {
		path:               Some(work.section_path.clone()),
		expected_file_hash: work.file_hash.clone().unwrap_or_default(),
		actual_file_hash:   compute_snapshot_tag(work.prepared.base_bytes()),
		file_lines:         lines,
		anchor_lines:       anchors,
		hash_recognized:    recognized,
	})
	.to_string()
	.into()
}

fn build_payload<P: EditPrepared>(
	works: &[PreparedWork<P>],
	projections: &[ProjectionWork],
	committed: Option<&[CommittedSection]>,
) -> Payload {
	let sections = works
		.iter()
		.zip(projections)
		.enumerate()
		.map(|(index, (work, projection))| {
			let move_dest = match &work.parsed.file_op {
				Some(FileOp::Move { dest }) => Some(dest.clone()),
				_ => None,
			};
			let op = match work.parsed.file_op {
				Some(FileOp::Rem) => SectionOp::Delete,
				Some(FileOp::Move { .. }) => SectionOp::Move,
				None if work.prepared.base_bytes() == &projection.after => SectionOp::Noop,
				None => SectionOp::Update,
			};
			let output_path = move_dest.as_ref().unwrap_or(&work.section_path);
			let committed_section = committed.and_then(|sections| sections.get(index));
			let after = if op == SectionOp::Delete {
				Bytes::new()
			} else {
				committed_section
					.and_then(|section| section.content.clone())
					.unwrap_or_else(|| projection.after.clone())
			};
			let header = (op != SectionOp::Delete)
				.then(|| format_hashline_header(output_path, &compute_snapshot_tag(&after)));
			let numbered = numbered_diff(
				work.prepared.base_bytes(),
				&after,
				Some(std::path::Path::new(output_path.as_str())),
			)
			.ok();
			let diff = numbered
				.as_ref()
				.map_or_else(Str::default, |diff| diff.text.clone());
			let preview = build_compact_diff_preview(&diff, CompactDiffOptions::default()).preview;
			SectionPayload {
				path: work.section_path.clone(),
				canonical_path: work.prepared.path().clone(),
				op,
				move_dest,
				old_revision: work.prepared.base_revision().clone(),
				new_revision: committed_section.and_then(|section| section.new_revision.clone()),
				applied_ops: projection.applied_ops.clone(),
				rebased: committed_section.is_some_and(|section| section.rebased),
				before: work.prepared.base_bytes().clone(),
				after,
				header,
				diff,
				preview,
				first_changed_line: projection.first_changed_line,
				block_resolutions: projection.block_resolutions.clone(),
				warnings: projection.warnings.clone(),
			}
		})
		.collect();
	Payload { sections }
}

fn op_details(edits: &[omp_hashline::Edit]) -> Vec<AppliedOp> {
	edits
		.iter()
		.map(|edit| AppliedOp {
			kind:       match edit {
				omp_hashline::Edit::Insert { .. } => "insert",
				omp_hashline::Edit::Delete { .. } => "delete",
				omp_hashline::Edit::Cut { .. } => "cut",
				omp_hashline::Edit::Paste { .. } => "paste",
				omp_hashline::Edit::Block { .. } => "block",
			}
			.into(),
			patch_line: edit.line_num(),
			index:      edit.index(),
		})
		.collect()
}

const fn done_fault(fault: Fault) -> Ev<EditUpdate, Payload, Fault> {
	Ev::Done(Outcome::Done { result: Err(fault), useless: false })
}

fn param_event(error: ParamError) -> Ev<EditUpdate, Payload, Fault> {
	match error {
		ParamError::Args(issue) => Ev::Args(*issue),
		ParamError::Interrupted(value) => Ev::Aborted(Abort::Interrupted { reason: value.reason }),
		ParamError::Protocol(reason) => Ev::Aborted(Abort::Skipped { reason }),
	}
}

fn commit_event(error: CommitError) -> Ev<EditUpdate, Payload, Fault> {
	match error {
		CommitError::Aborted => Ev::Aborted(Abort::InputDropped),
		CommitError::Interrupted(value) => Ev::Aborted(Abort::Interrupted { reason: value.reason }),
		CommitError::Protocol(reason) => Ev::Aborted(Abort::Skipped { reason }),
	}
}

fn rejection_text(fault: &Fault) -> Str {
	match &fault.reason {
		RejectionReason::Conflict => {
			let mut text =
				format!("Edit rejected: conflict ({} overlapping range(s))", fault.conflicts.len());
			for conflict in &fault.conflicts {
				write!(text, "\n{}-{}: {}", conflict.start_line, conflict.end_line, conflict.message)
					.expect("writing to String cannot fail");
			}
			text.into()
		},
		RejectionReason::StaleUnrecoverable { message }
		| RejectionReason::Format { message }
		| RejectionReason::InvalidPatch { message } => message.clone(),
	}
}
