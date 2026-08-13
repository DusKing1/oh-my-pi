//! Streaming hashline edits over revision-pinned document transactions.

use std::{fmt::Write as _, future::Future};

use async_stream::stream;
use bytes::Bytes;
use futures::{FutureExt, Stream, pin_mut, select_biased};
use omp_core::Str;
use omp_hashline::{
	ApplyMode, ApplyOptions, Clipboard, Edit, apply_parsed_patch, compute_snapshot_tag,
	diff_preview::{CompactDiffOptions, build_compact_diff_preview},
	format_hashline_header, numbered_diff, parse_patch_streaming,
};
use omp_tool::{
	Abort, ArgIssue, ArgIssueKind, ArgPath, CommitError, Constraint, Ev, IncomingParams,
	InterruptWaitError, Outcome, ParamError, Part, PromptCaps, Rev, Tool, ToolSpec,
};
use serde::{Deserialize, Serialize};

use crate::render::TextProjection;

const SCHEMA: &[u8] = br#"{"type":"object","properties":{"path":{"type":"string","minLength":1},"patch":{"type":"string","minLength":1}},"required":["path","patch"],"additionalProperties":false}"#;

/// Streaming arguments for `edit@hl.1`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Params {
	/// Workspace-relative document path.
	pub path:  Str,
	/// Hashline patch body (without a file-section header).
	pub patch: Str,
}

/// A dry-run projection emitted whenever another complete operation becomes
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
	/// One-indexed line in the submitted patch.
	pub patch_line: usize,
	/// Authored operation sequence index.
	pub index:      usize,
}

/// Durable successful transaction truth.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Payload {
	/// Workspace-relative document path.
	pub path:         Str,
	/// Pinned document base revision.
	pub old_revision: Str,
	/// Committed target document revision.
	pub new_revision: Str,
	/// Sequence of applied operations.
	pub applied_ops:  Vec<AppliedOp>,
	/// Whether the edit was rebased over intervening changes.
	pub rebased:      bool,
	/// Complete transaction diff, never the capped preview.
	pub diff:         Str,
}

/// Formatting requested of the document transaction coordinator.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FormatPolicy {
	/// Apply the configured formatter when one is available and require it to
	/// succeed.
	Configured,
}

/// Stale-base behavior requested of the transaction coordinator.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StalePolicy {
	/// Rebase only edits whose base spans do not overlap intervening changes.
	RebaseNonOverlapping,
}

/// Borrowed view exposed by an opaque, revision-pinned prepared lease.
pub trait EditPrepared: Send {
	/// Canonical path pinned by this lease.
	fn path(&self) -> &Str;
	/// Opaque pinned base revision.
	fn base_revision(&self) -> &Str;
	/// Exact bytes at the pinned base revision.
	fn base_bytes(&self) -> &Bytes;
}

/// The sole proposal accepted by this executor's resource boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditProposal {
	/// Edit dialect identifier.
	pub format:        Str,
	/// Strictly parsed operation details for the complete submitted patch.
	pub applied_ops:   Vec<AppliedOp>,
	/// Submitted edit patch payload.
	pub payload:       Str,
	/// Pinned base revision string.
	pub base_revision: Str,
	/// Configured stale-base handling policy.
	pub stale_policy:  StalePolicy,
	/// Configured code formatting policy.
	pub format_policy: FormatPolicy,
}

/// Structured successful response from the atomic transaction owner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommitResult {
	/// Committed target document revision.
	pub new_revision: Str,
	/// Sequence of applied operations.
	pub applied_ops:  Vec<AppliedOp>,
	/// Whether the edit was rebased over intervening changes.
	pub rebased:      bool,
	/// Complete transaction diff.
	pub diff:         Str,
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
	/// Base revision is stale and cannot be automatically rebased.
	StaleUnrecoverable,
	/// Formatter execution failed on the edited document.
	Format {
		/// Formatter output or error details.
		message: Str,
	},
	/// Submitted patch syntax or structure was invalid.
	InvalidPatch {
		/// Patch parsing error details.
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

/// Truthful resource-owned commit failure classification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EditCommitError {
	/// Atomic transaction rejection; the head is known untouched.
	Rejected(Fault),
	/// The resource cannot prove whether effects landed.
	EffectsUnknown {
		/// Context describing the uncertainty surrounding potential side effects.
		reason: Str,
	},
}

/// Resource boundary implemented by the environment's document host.
pub trait EditDocuments: Send + Sync + 'static {
	/// Concrete owner of the pinned resource; the same value is moved to commit.
	type Prepared: EditPrepared;

	/// Opens and pins the exact current revision without changing it.
	fn prepare(&self, path: Str) -> impl Future<Output = Result<Self::Prepared, Fault>> + Send + '_;

	/// Atomically commits one `omp.hashline` proposal against the prepared
	/// lease.
	fn commit(
		&self,
		prepared: Self::Prepared,
		proposal: EditProposal,
	) -> impl Future<Output = Result<CommitResult, EditCommitError>> + Send + '_;
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
			description: "Apply a hashline patch to one revision-pinned document.".into(),
			schema:      Bytes::from_static(SCHEMA),
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
			let pulled = {
				let (updates_tx, updates_rx) = flume::unbounded();
				let pull = params.pull(|mut doc| async move {
					let mut object = doc.json().object();
					let path = object.key("path").string().finish().await?;
					let prepared = match self.documents.prepare(path.clone()).await {
						Ok(prepared) => prepared,
						Err(fault) => return Ok(Err(fault)),
					};
					let mut patch_cursor = object.key("patch").string();
					let mut patch = String::new();
					let mut last_update = None;
					while let Some(chunk) = patch_cursor.next_chunk().await? {
						patch.push_str(&chunk);
						if let Some(update) = preview(&prepared, &patch)
							&& last_update.as_ref() != Some(&update)
						{
							last_update = Some(update.clone());
							let _ = updates_tx.send(update);
						}
					}
					Ok(Ok((path, Str::from(patch), prepared)))
				}).fuse();
				pin_mut!(pull);
				loop {
					let update = updates_rx.recv_async().fuse();
					pin_mut!(update);
					select_biased! {
						result = pull => break result,
						result = update => if let Ok(update) = result { yield Ev::Update(update); },
					}
				}
			};
			let (_path, patch, prepared) = match pulled {
				Ok(Ok(value)) => value,
				Ok(Err(fault)) => {
					yield Ev::Done(Outcome::Done { result: Err(fault), useless: false });
					return;
				},
				Err(error) => { yield param_event(error); return; },
			};
			let parsed = match omp_hashline::parse_patch(&patch) {
				Ok(parsed) => parsed,
				Err(_) => {
					yield Ev::Args(ArgIssue {
						path: vec![ArgPath::Key("patch".into())],
						expected: "complete omp.hashline patch".into(),
						kind: ArgIssueKind::Malformed,
						example: Some("PUT 1.=1:\n+replacement".into()),
						found: Some("malformed hashline patch".into()),
					});
					return;
				},
			};
			let applied_ops = op_details(&parsed.edits);
			match params.committed().await {
				Ok(_) => {},
				Err(error) => { yield commit_event(error); return; },
			}
			let path = prepared.path().clone();
			let old_revision = prepared.base_revision().clone();
			let tag = compute_snapshot_tag(prepared.base_bytes());
			let header = format_hashline_header(&path, &tag);
			let mut enveloped = String::with_capacity(header.len() + patch.len() + 1);
			write!(enveloped, "{header}\n{patch}").expect("writing to String cannot fail");
			let proposal = EditProposal {
				format: "omp.hashline".into(), payload: enveloped.into(), base_revision: old_revision.clone(),
				stale_policy: StalePolicy::RebaseNonOverlapping, format_policy: self.format_policy,
				applied_ops,
			};
			let commit = self.documents.commit(prepared, proposal).fuse();
			let interrupt = params.next_interrupt().fuse();
			pin_mut!(commit, interrupt);
			select_biased! {
				result = commit => match result {
					Ok(result) => yield Ev::Done(Outcome::Done { result: Ok(Payload {
						path, old_revision, new_revision: result.new_revision,
						applied_ops: result.applied_ops, rebased: result.rebased, diff: result.diff,
					}), useless: false }),
					Err(EditCommitError::Rejected(fault)) => yield Ev::Done(Outcome::Done { result: Err(fault), useless: false }),
					Err(EditCommitError::EffectsUnknown { reason }) => yield Ev::Aborted(Abort::EffectsUnknown { reason }),
				},
				interrupted = interrupt => yield Ev::Aborted(match interrupted {
					Ok(value) => Abort::EffectsUnknown { reason: value.reason },
					Err(InterruptWaitError::Closed) => Abort::EffectsUnknown { reason: "invocation owner disappeared during transaction".into() },
					Err(InterruptWaitError::Protocol(reason)) => Abort::EffectsUnknown { reason },
				}),
			}
		}
	}

	fn prompt(&self, view: Result<&Payload, &Fault>, caps: &PromptCaps) -> Vec<Part> {
		let Some(mut out) = TextProjection::new(caps) else {
			return Vec::new();
		};
		match view {
			Ok(payload) => {
				let _ = out.push(&format!(
					"Applied {} operation(s) to {} ({} -> {}){}",
					payload.applied_ops.len(),
					payload.path,
					payload.old_revision,
					payload.new_revision,
					if payload.rebased { " [rebased]" } else { "" }
				));
			},
			Err(fault) => {
				let _ = out.push(&format!("Edit rejected: {}", rejection_text(fault)));
			},
		}
		out.finish()
	}
}

fn preview(prepared: &impl EditPrepared, patch: &str) -> Option<EditUpdate> {
	let parsed = parse_patch_streaming(patch).ok()?;
	if parsed.edits.is_empty() {
		return None;
	}
	let applied = apply_parsed_patch(
		prepared.base_bytes().clone(),
		&parsed,
		&mut Clipboard::default(),
		ApplyOptions { mode: ApplyMode::Partial, path: Some(prepared.path()) },
	)
	.ok()?;
	let diff = numbered_diff(prepared.base_bytes(), &applied.bytes).ok()?;
	let compact = build_compact_diff_preview(&diff.text, CompactDiffOptions::default());
	Some(EditUpdate {
		applied_ops: op_details(&parsed.edits).len(),
		preview: compact.preview,
		added_lines: diff.added_lines,
		removed_lines: diff.removed_lines,
	})
}

fn op_details(edits: &[Edit]) -> Vec<AppliedOp> {
	edits
		.iter()
		.map(|edit| AppliedOp {
			kind:       match edit {
				Edit::Insert { .. } => "insert",
				Edit::Delete { .. } => "delete",
				Edit::Cut { .. } => "cut",
				Edit::Paste { .. } => "paste",
				Edit::Block { .. } => "block",
			}
			.into(),
			patch_line: edit.line_num(),
			index:      edit.index(),
		})
		.collect()
}

fn param_event(error: ParamError) -> Ev<EditUpdate, Payload, Fault> {
	match error {
		ParamError::Args(issue) => Ev::Args(issue),
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

fn rejection_text(fault: &Fault) -> String {
	match &fault.reason {
		RejectionReason::Conflict => {
			let mut text = format!("conflict ({} overlapping range(s))", fault.conflicts.len());
			for conflict in &fault.conflicts {
				write!(
					text,
					"\n{}-{}: {}",
					conflict.start_line, conflict.end_line, conflict.message
				)
				.expect("writing to String cannot fail");
			}
			text
		},
		RejectionReason::StaleUnrecoverable => "stale base could not be rebased".into(),
		RejectionReason::Format { message } => format!("format: {message}"),
		RejectionReason::InvalidPatch { message } => format!("invalid patch: {message}"),
	}
}
