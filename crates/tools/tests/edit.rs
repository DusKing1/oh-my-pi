//! Exact pi-facing contracts for hashline edit execution and projection.

use std::{collections::HashMap, sync::Arc};

use bytes::Bytes;
use futures::StreamExt;
use omp_core::Str;
use omp_hashline::{
	Clipboard, MismatchDetails, MismatchError, compute_snapshot_tag, loop_guard::NoopLoopGuard,
};
use omp_tool::{Ev, IncomingParams, Outcome, Part, PromptCaps, Tool};
use omp_tools::edit::{
	CommitResult, CommittedSection, Conflict, EditAction, EditCommitError, EditDocuments,
	EditPrepared, EditProposal, Fault, FormatPolicy, NoopResult, PrepareRequest, RejectionReason,
	tool,
};
use parking_lot::Mutex;

#[derive(Default)]
struct State {
	prepared:   Vec<PrepareRequest>,
	noop_guard: NoopLoopGuard,
	commits:    Vec<EditProposal>,
}

#[derive(Clone)]
struct Fake {
	files: Arc<HashMap<Str, Bytes>>,
	state: Arc<Mutex<State>>,
	fault: Option<Fault>,
}

impl Fake {
	fn with_files(files: &[(&str, &'static [u8])]) -> Self {
		Self {
			files: Arc::new(
				files
					.iter()
					.map(|(path, bytes)| (Str::from(*path), Bytes::from_static(bytes)))
					.collect(),
			),
			state: Arc::default(),
			fault: None,
		}
	}
}

struct Lease {
	path:     Str,
	revision: Str,
	base:     Bytes,
	authored: Bytes,
}

impl EditPrepared for Lease {
	fn path(&self) -> &Str {
		&self.path
	}

	fn base_revision(&self) -> &Str {
		&self.revision
	}

	fn base_bytes(&self) -> &Bytes {
		&self.base
	}

	fn authored_bytes(&self) -> &Bytes {
		&self.authored
	}
}

impl EditDocuments for Fake {
	type Prepared = Lease;

	async fn prepare(&self, request: PrepareRequest) -> Result<Self::Prepared, Fault> {
		self.state.lock().prepared.push(request.clone());
		if let Some(fault) = &self.fault {
			return Err(fault.clone());
		}
		let Some(content) = self.files.get(&request.path).cloned() else {
			return Err(Fault {
				reason:    RejectionReason::InvalidPatch { message: "file not found".into() },
				conflicts: Vec::new(),
			});
		};
		Ok(Lease {
			path:     request.path,
			revision: "r1".into(),
			base:     content.clone(),
			authored: content,
		})
	}

	fn record_noop(&self, canonical_path: &str, display_path: &str, input: Bytes) -> NoopResult {
		let record =
			self
				.state
				.lock()
				.noop_guard
				.record_noop_for(canonical_path, display_path, input);
		NoopResult { diagnostic: record.diagnostic().into(), escalate: record.should_escalate() }
	}

	fn reset_noop(&self, canonical_path: &str) {
		self.state.lock().noop_guard.reset(canonical_path);
	}

	fn start_clipboard_batch(&self) -> Clipboard {
		Clipboard::default()
	}

	async fn commit(
		&self,
		_prepared: Vec<&mut Self::Prepared>,
		proposals: Vec<EditProposal>,
		_clipboard: Clipboard,
	) -> Result<CommitResult, EditCommitError> {
		let sections = proposals
			.iter()
			.map(|proposal| CommittedSection {
				new_revision: (!matches!(proposal.action, EditAction::Delete)).then(|| "r2".into()),
				rebased:      false,
				content:      match &proposal.action {
					EditAction::Write { content } | EditAction::Move { content, .. } => {
						Some(content.clone())
					},
					EditAction::Delete => None,
				},
			})
			.collect();
		self.state.lock().commits.extend(proposals);
		Ok(CommitResult { sections })
	}
}

const fn caps() -> PromptCaps {
	PromptCaps { maximum_parts: 1, maximum_text_bytes: 16 * 1024, media: false }
}

async fn invoke(fake: Fake, input: &str) -> (omp_tools::edit::Payload, Vec<Part>) {
	let edit = tool(fake, FormatPolicy::Configured);
	let raw = serde_json::json!({ "input": input }).to_string();
	let (feed, incoming) = IncomingParams::channel();
	feed.arg_text(raw.clone().into()).unwrap();
	feed.args_committed(raw.into()).unwrap();
	let events = edit.call(incoming).collect::<Vec<_>>().await;
	let payload = events
		.into_iter()
		.find_map(|event| match event {
			Ev::Done(Outcome::Done { result: Ok(payload), .. }) => Some(payload),
			_ => None,
		})
		.expect("successful edit payload");
	let parts = edit.prompt(Ok(&payload), &caps());
	(payload, parts)
}

fn text(parts: &[Part]) -> &str {
	match parts {
		[Part::Text { text }] => text,
		_ => panic!("expected one text part"),
	}
}

#[test]
fn generated_schema_is_semantically_the_pi_edit_schema() {
	let edit = tool(Fake::with_files(&[]), FormatPolicy::Configured);
	let actual: serde_json::Value =
		serde_json::from_slice(&edit.spec().schema).expect("edit schema JSON");
	assert_eq!(
		actual,
		serde_json::json!({
			"type": "object",
			"additionalProperties": false,
			"required": ["input"],
			"properties": {
				"input": {"type": "string"}
			}
		})
	);
	assert!(
		serde_json::from_value::<omp_tools::edit::Params>(
			serde_json::json!({"input": "[a.txt#A1B2]", "extra": true})
		)
		.is_err(),
		"edit params must reject unknown fields"
	);
}

#[tokio::test]
async fn put_and_cut_render_exact_post_edit_headers_and_previews() {
	let fake = Fake::with_files(&[("a.txt", b"one\ntwo\nthree\n"), ("b.txt", b"alpha\nbeta\n")]);
	let a_tag = compute_snapshot_tag(b"one\ntwo\nthree\n");
	let b_tag = compute_snapshot_tag(b"alpha\nbeta\n");
	let input = format!("[a.txt#{a_tag}]\nPUT 2.=2:\n+TWO\n[b.txt#{b_tag}]\nCUT 1.=1");
	let (payload, parts) = invoke(fake.clone(), &input).await;
	let a_after = b"one\nTWO\nthree\n";
	let b_after = b"beta\n";
	assert_eq!(
		text(&parts),
		format!(
			"[a.txt#{}]\n1:one\n2:TWO\n3:three\n\n[b.txt#{}]\n1:beta",
			compute_snapshot_tag(a_after),
			compute_snapshot_tag(b_after)
		)
	);
	assert_eq!(payload.sections.len(), 2);
	let state = fake.state.lock();
	let commits = &state.commits;
	assert!(
		matches!(&commits[0].action, EditAction::Write { content } if content.as_ref() == a_after)
	);
	assert!(
		matches!(&commits[1].action, EditAction::Write { content } if content.as_ref() == b_after)
	);
}

#[tokio::test]
async fn rem_and_mv_render_exact_file_operation_text() {
	let fake = Fake::with_files(&[("old.txt", b"one\ntwo\n"), ("gone.txt", b"bye\n")]);
	let old_tag = compute_snapshot_tag(b"one\ntwo\n");
	let gone_tag = compute_snapshot_tag(b"bye\n");
	let input = format!("[old.txt#{old_tag}]\nMV new.txt\n[gone.txt#{gone_tag}]\nREM");
	let (_, parts) = invoke(fake.clone(), &input).await;
	assert_eq!(
		text(&parts),
		format!(
			"[new.txt#{}]\nMoved to new.txt\n\nDeleted gone.txt",
			compute_snapshot_tag(b"one\ntwo\n")
		)
	);
	let state = fake.state.lock();
	let commits = &state.commits;
	assert!(
		matches!(&commits[0].action, EditAction::Move { destination, content } if destination == "new.txt" && content.as_ref() == b"one\ntwo\n")
	);
	assert!(matches!(&commits[1].action, EditAction::Delete));
}

#[tokio::test]
async fn edits_followed_by_mv_form_one_move_with_final_content() {
	let fake = Fake::with_files(&[("old.txt", b"one\ntwo\n")]);
	let old_tag = compute_snapshot_tag(b"one\ntwo\n");
	let input = format!("[old.txt#{old_tag}]\nPUT 2.=2:\n+TWO\nMV new.txt");
	let _ = invoke(fake.clone(), &input).await;
	let edited = b"one\nTWO\n";
	let state = fake.state.lock();
	assert_eq!(state.commits.len(), 1);
	assert!(matches!(
		&state.commits[0].action,
		EditAction::Move { destination, content }
			if destination == "new.txt" && content.as_ref() == edited
	));
}

#[tokio::test]
async fn byte_identical_put_escalates_from_exact_soft_diagnostic_to_loop_guard_failure() {
	let fake = Fake::with_files(&[("a.txt", b"same\n")]);
	let tag = compute_snapshot_tag(b"same\n");
	let input = format!("[a.txt#{tag}]\nPUT 1.=1:\n+same");
	for _ in 0..2 {
		let (_, parts) = invoke(fake.clone(), &input).await;
		assert_eq!(
			text(&parts),
			"Edits to a.txt parsed and applied cleanly, but produced no change: your body row(s) are \
			 byte-identical to the file at the targeted lines. The bug is somewhere else — re-read \
			 the file before issuing another edit. Do NOT widen the payload or add lines; verify the \
			 anchor first."
		);
	}

	let edit = tool(fake.clone(), FormatPolicy::Configured);
	let raw = serde_json::json!({ "input": input }).to_string();
	let (feed, incoming) = IncomingParams::channel();
	feed.arg_text(raw.clone().into()).unwrap();
	feed.args_committed(raw.into()).unwrap();
	let events = edit.call(incoming).collect::<Vec<_>>().await;
	let fault = events
		.into_iter()
		.find_map(|event| match event {
			Ev::Done(Outcome::Done { result: Err(fault), .. }) => Some(fault),
			_ => None,
		})
		.expect("third identical no-op must fail");
	assert_eq!(
		text(&edit.prompt(Err(&fault), &caps())),
		"STOP. Edits to a.txt have been a byte-identical no-op 3 times in a row — the patch body \
		 matches the file at the targeted lines and the soft hint did not break the cycle. Cease \
		 re-issuing this payload. Either the intended change is already on disk (move on), or your \
		 anchor is wrong (re-read the file with `read` to observe the current line numbers and tag, \
		 then author a different edit). This exact payload will keep being rejected until it \
		 changes."
	);
	assert_eq!(fake.state.lock().commits, [] as [omp_tools::edit::EditProposal; 0]);
}

#[tokio::test]
async fn stale_tag_and_transaction_conflict_messages_are_projected_verbatim() {
	let mismatch = MismatchError::new(MismatchDetails {
		path:               Some("a.txt".into()),
		expected_file_hash: "1A2B".into(),
		actual_file_hash:   "C3D4".into(),
		file_lines:         vec!["one".into(), "two".into(), "three".into()],
		anchor_lines:       vec![2],
		hash_recognized:    true,
	});
	let stale = Fault {
		reason:    RejectionReason::StaleUnrecoverable { message: mismatch.to_string().into() },
		conflicts: Vec::new(),
	};
	let edit = tool(Fake::with_files(&[]), FormatPolicy::Configured);
	assert_eq!(text(&edit.prompt(Err(&stale), &caps())), mismatch.display_message());

	let conflict = Fault {
		reason:    RejectionReason::Conflict,
		conflicts: vec![Conflict {
			start_line: 4,
			end_line:   6,
			message:    "overlapping concurrent edit".into(),
		}],
	};
	assert_eq!(
		text(&edit.prompt(Err(&conflict), &caps())),
		"Edit rejected: conflict (1 overlapping range(s))\n4-6: overlapping concurrent edit"
	);
}

#[tokio::test]
async fn malformed_and_headerless_input_never_commit_and_preserve_parser_diagnostics() {
	let fake = Fake::with_files(&[("a.txt", b"one\n")]);
	let edit = tool(fake.clone(), FormatPolicy::Configured);
	for (input, expected) in [
		("", "No hashline sections found in input."),
		(
			"@@ -1,1 +1,1 @@\n-old\n+new",
			"unified-diff hunk header is not valid in hashline. File sections start with \
			 `[path#HASH]`; use `PUT`, `CUT`, `REM`, or `MV`.",
		),
		(
			"[a.txt#1A2B]\nPUT 1.=:\n+x",
			"line 1: payload line has no preceding hunk header. Use `PUT N.=M:`, `CUT N.=M`, or `PUT \
			 <N:`/`PUT >N:` above the body. Got \"PUT 1.=:\".",
		),
		(
			"[a.txt#1A2B]\nPUT 1.=2:\n+X\nPUT 2.=3:\n+Y",
			"line 3: anchor line 2 is already targeted by another hunk on line 1. Issue ONE hunk per \
			 range; payload is only the final desired content, never a before/after pair.",
		),
	] {
		let raw = serde_json::json!({ "input": input }).to_string();
		let (feed, incoming) = IncomingParams::channel();
		feed.arg_text(raw.clone().into()).unwrap();
		feed.args_committed(raw.into()).unwrap();
		let events = edit.call(incoming).collect::<Vec<_>>().await;
		let rendered = events
			.iter()
			.find_map(|event| match event {
				Ev::Done(Outcome::Done { result: Err(fault), .. }) => {
					Some(text(&edit.prompt(Err(fault), &caps())).to_owned())
				},
				Ev::Args(issue) => issue.found.as_deref().map(str::to_owned),
				_ => None,
			})
			.unwrap_or_else(|| panic!("diagnostic event for {input:?}: {events:?}"));
		assert_eq!(rendered, expected);
	}
	assert_eq!(fake.state.lock().commits, [] as [omp_tools::edit::EditProposal; 0]);
}

#[tokio::test]
async fn copied_read_elision_is_ignored_but_reported_as_a_warning() {
	let fake = Fake::with_files(&[("a.txt", b"one\ntwo\n")]);
	let tag = compute_snapshot_tag(b"one\ntwo\n");
	let input = format!(
		"[a.txt#{tag}]\n[…8ln elided; re-read needed ranges with |, e.g. a.txt:10-17]\nPUT \
		 1.=1:\n+ONE"
	);
	let (_, parts) = invoke(fake, &input).await;
	let output = text(&parts);
	assert!(
		output.ends_with(
			"\n\nWarnings:\nIgnored copied read-output elision row(s). Re-read elided ranges before \
			 editing them."
		),
		"{output:?}"
	);
}
