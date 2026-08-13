use std::{
	future::{Future, pending},
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
};

use bytes::Bytes;
use futures::{StreamExt, pin_mut};
use omp_core::Str;
use omp_tool::{Ev, IncomingParams, Interrupt, Outcome, Part, PromptCaps, Tool};
use omp_tools::edit::{
	AppliedOp, CommitResult, EditCommitError, EditDocuments, EditPrepared, EditProposal, Fault,
	FormatPolicy, RejectionReason, tool,
};
use parking_lot::Mutex;

#[derive(Clone)]
struct Fake {
	state:         Arc<Mutex<State>>,
	result:        Result<CommitResult, EditCommitError>,
	prepare_fault: Option<Fault>,
}

#[derive(Default)]
struct State {
	prepared:      Vec<Str>,
	commits:       Vec<EditProposal>,
	committed_ids: Vec<u64>,
}

impl Fake {
	fn success() -> Self {
		Self {
			state:         Arc::default(),
			result:        Ok(CommitResult {
				new_revision: "r2".into(),
				applied_ops:  vec![AppliedOp {
					kind:       "insert".into(),
					patch_line: 1,
					index:      0,
				}],
				rebased:      false,
				diff:         "-1|old\n+1|new".into(),
			}),
			prepare_fault: None,
		}
	}
}

struct Lease {
	id:            u64,
	path:          Str,
	base_revision: Str,
	base_bytes:    Bytes,
}

impl EditPrepared for Lease {
	fn path(&self) -> &Str {
		&self.path
	}

	fn base_revision(&self) -> &Str {
		&self.base_revision
	}

	fn base_bytes(&self) -> &Bytes {
		&self.base_bytes
	}
}

impl EditDocuments for Fake {
	type Prepared = Lease;

	fn prepare(&self, path: Str) -> impl Future<Output = Result<Lease, Fault>> + Send + '_ {
		async move {
			self.state.lock().prepared.push(path.clone());
			if let Some(fault) = &self.prepare_fault {
				return Err(fault.clone());
			}
			let path = if path == "relative.txt" {
				"/workspace/relative.txt".into()
			} else {
				path
			};
			Ok(Lease {
				id: 7,
				path,
				base_revision: "r1".into(),
				base_bytes: Bytes::from_static(b"old\n"),
			})
		}
	}

	fn commit(
		&self,
		prepared: Lease,
		proposal: EditProposal,
	) -> impl Future<Output = Result<CommitResult, EditCommitError>> + Send + '_ {
		async move {
			assert_eq!(prepared.base_revision, "r1");
			let mut state = self.state.lock();
			state.committed_ids.push(prepared.id);
			state.commits.push(proposal);
			drop(state);
			self.result.clone()
		}
	}
}

#[derive(Clone)]
struct CancelFake {
	head_changed: Arc<AtomicBool>,
}

impl EditDocuments for CancelFake {
	type Prepared = Lease;

	fn prepare(&self, path: Str) -> impl Future<Output = Result<Lease, Fault>> + Send + '_ {
		async move {
			Ok(Lease {
				id: 9,
				path,
				base_revision: "r1".into(),
				base_bytes: Bytes::from_static(b"old\n"),
			})
		}
	}

	fn commit(
		&self,
		_prepared: Lease,
		_proposal: EditProposal,
	) -> impl Future<Output = Result<CommitResult, EditCommitError>> + Send + '_ {
		async move {
			pending::<()>().await;
			self.head_changed.store(true, Ordering::SeqCst);
			unreachable!()
		}
	}
}

#[tokio::test]
async fn pins_early_previews_progressively_and_waits_for_commit_gate() {
	let fake = Fake::success();
	let state = Arc::clone(&fake.state);
	let edit = tool(fake, FormatPolicy::Configured);
	let (feed, incoming) = IncomingParams::channel();
	let stream = edit.call(incoming);
	pin_mut!(stream);

	let prefix = r#"{"path":"src/a.rs","patch":"PUT 1.=1:\n+new"#;
	feed.arg_text(prefix.into()).unwrap();
	let update = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
		.await
		.unwrap()
		.unwrap();
	assert!(matches!(
		&update,
		Ev::Update(value)
			if value.preview.contains("new")
				&& value.added_lines == 1
				&& value.removed_lines == 1
	));
	assert_eq!(state.lock().prepared.as_slice(), ["src/a.rs"]);
	assert!(state.lock().commits.is_empty(), "preview must not cross the effect gate");

	let suffix = "\"}";
	feed.arg_text(suffix.into()).unwrap();
	let raw = format!("{prefix}{suffix}");
	feed.args_committed(raw.into()).unwrap();
	let done = stream.next().await.unwrap();
	let Ev::Done(Outcome::Done { result: Ok(payload), .. }) = done else {
		panic!("expected success")
	};
	assert_eq!(payload.old_revision, "r1");
	assert_eq!(payload.new_revision, "r2");
	assert_eq!(payload.diff, "-1|old\n+1|new");
	assert_eq!(state.lock().commits[0].format, "omp.hashline");
	let tag = omp_hashline::compute_snapshot_tag(b"old\n");
	assert_eq!(
		state.lock().commits[0].payload,
		format!("[src/a.rs#{tag}]\nPUT 1.=1:\n+new")
	);
	assert!(!state.lock().commits[0].applied_ops.is_empty());
	assert_eq!(state.lock().committed_ids, [7]);
}

#[tokio::test]
async fn relative_input_commits_and_reports_the_canonical_prepared_path() {
	let fake = Fake::success();
	let state = Arc::clone(&fake.state);
	let edit = tool(fake, FormatPolicy::Configured);
	let (feed, incoming) = IncomingParams::channel();
	let raw = r#"{"path":"relative.txt","patch":"PUT 1.=1:\n+new"}"#;
	feed.arg_text(raw.into()).unwrap();
	feed.args_committed(raw.into()).unwrap();
	let stream = edit.call(incoming);
	pin_mut!(stream);
	let mut payload = None;
	while let Some(event) = stream.next().await {
		if let Ev::Done(Outcome::Done { result: Ok(value), .. }) = event {
			payload = Some(value);
			break;
		}
	}
	assert_eq!(payload.unwrap().path, "/workspace/relative.txt");
	let tag = omp_hashline::compute_snapshot_tag(b"old\n");
	assert_eq!(
		state.lock().commits[0].payload,
		format!("[/workspace/relative.txt#{tag}]\nPUT 1.=1:\n+new")
	);
}

#[tokio::test]
async fn malformed_pulled_path_lowers_to_args_without_commit() {
	let fake = Fake::success();
	let state = Arc::clone(&fake.state);
	let edit = tool(fake, FormatPolicy::Configured);
	let (feed, incoming) = IncomingParams::channel();
	let raw = r#"{"path":42,"patch":"PUT 1.=1:\n+x"}"#;
	feed.arg_text(raw.into()).unwrap();
	feed.args_committed(raw.into()).unwrap();
	let stream = edit.call(incoming);
	pin_mut!(stream);
	assert!(matches!(stream.next().await, Some(Ev::Args(_))));
	assert!(state.lock().commits.is_empty());
}

#[tokio::test]
async fn typed_rejection_is_preserved() {
	let fault = Fault {
		reason:    RejectionReason::Format { message: "formatter failed".into() },
		conflicts: Vec::new(),
	};
	let fake = Fake {
		state:         Arc::default(),
		result:        Err(EditCommitError::Rejected(fault.clone())),
		prepare_fault: None,
	};
	let edit = tool(fake, FormatPolicy::Configured);
	let (feed, incoming) = IncomingParams::channel();
	let raw = r#"{"path":"a","patch":"PUT 1.=1:\n+x"}"#;
	feed.arg_text(raw.into()).unwrap();
	feed.args_committed(raw.into()).unwrap();
	let stream = edit.call(incoming);
	pin_mut!(stream);
	let mut terminal = None;
	while let Some(event) = stream.next().await {
		if matches!(event, Ev::Done(_)) {
			terminal = Some(event);
			break;
		}
	}
	assert!(
		matches!(terminal, Some(Ev::Done(Outcome::Done { result: Err(value), .. })) if value == fault)
	);
}

#[tokio::test]
async fn disjoint_stale_rebase_truth_is_preserved() {
	let fake = Fake {
		state: Arc::default(),
		result: Ok(CommitResult {
			new_revision: "r3".into(),
			applied_ops: vec![AppliedOp { kind: "insert".into(), patch_line: 1, index: 0 }],
			rebased: true,
			diff: " 1|other\n-2|old\n+2|new".into(),
		}),
		prepare_fault: None,
	};
	let edit = tool(fake, FormatPolicy::Configured);
	let (feed, incoming) = IncomingParams::channel();
	let raw = r#"{"path":"a","patch":"PUT 1.=1:\n+x"}"#;
	feed.arg_text(raw.into()).unwrap();
	feed.args_committed(raw.into()).unwrap();
	let stream = edit.call(incoming);
	pin_mut!(stream);
	let mut payload = None;
	while let Some(event) = stream.next().await {
		if let Ev::Done(Outcome::Done { result: Ok(value), .. }) = event {
			payload = Some(value);
			break;
		}
	}
	assert!(payload.is_some_and(|value| value.rebased && value.new_revision == "r3"));
}

#[tokio::test]
async fn overlapping_rejection_preserves_typed_conflicts() {
	let fault = Fault {
		reason: RejectionReason::Conflict,
		conflicts: vec![omp_tools::edit::Conflict {
			start_line: 1,
			end_line: 1,
			message: "overlapping concurrent edit".into(),
		}],
	};
	let fake = Fake {
		state: Arc::default(),
		result: Err(EditCommitError::Rejected(fault.clone())),
		prepare_fault: None,
	};
	let edit = tool(fake, FormatPolicy::Configured);
	let (feed, incoming) = IncomingParams::channel();
	let raw = r#"{"path":"a","patch":"PUT 1.=1:\n+x"}"#;
	feed.arg_text(raw.into()).unwrap();
	feed.args_committed(raw.into()).unwrap();
	let stream = edit.call(incoming);
	pin_mut!(stream);
	let mut terminal = None;
	while let Some(event) = stream.next().await {
		if matches!(event, Ev::Done(_)) {
			terminal = Some(event);
			break;
		}
	}
	let parts = edit.prompt(
		Err(&fault),
		&PromptCaps { maximum_parts: 1, maximum_text_bytes: 1024, media: false },
	);
	assert!(matches!(
		parts.as_slice(),
		[Part::Text { text }] if text.contains("1-1: overlapping concurrent edit")
	));
	assert!(matches!(
		terminal,
		Some(Ev::Done(Outcome::Done { result: Err(value), .. })) if value == fault
	));
}

#[tokio::test]
async fn uncertain_resource_commit_is_not_misreported_as_rejection() {
	let fake = Fake {
		state: Arc::default(),
		result: Err(EditCommitError::EffectsUnknown {
			reason: "partially committed".into(),
		}),
		prepare_fault: None,
	};
	let edit = tool(fake, FormatPolicy::Configured);
	let (feed, incoming) = IncomingParams::channel();
	let raw = r#"{"path":"a","patch":"PUT 1.=1:\n+x"}"#;
	feed.arg_text(raw.into()).unwrap();
	feed.args_committed(raw.into()).unwrap();
	let stream = edit.call(incoming);
	pin_mut!(stream);
	let mut terminal = None;
	while let Some(event) = stream.next().await {
		if matches!(event, Ev::Aborted(_)) {
			terminal = Some(event);
			break;
		}
	}
	assert!(matches!(
		terminal,
		Some(Ev::Aborted(omp_tool::Abort::EffectsUnknown { .. }))
	));
}

#[tokio::test]
async fn prepare_failure_is_a_typed_fault() {
	let fault = Fault { reason: RejectionReason::StaleUnrecoverable, conflicts: Vec::new() };
	let fake = Fake {
		state:         Arc::default(),
		result:        Ok(CommitResult {
			new_revision: "unused".into(),
			applied_ops:  Vec::new(),
			rebased:      false,
			diff:         Str::default(),
		}),
		prepare_fault: Some(fault.clone()),
	};
	let state = Arc::clone(&fake.state);
	let edit = tool(fake, FormatPolicy::Configured);
	let (feed, incoming) = IncomingParams::channel();
	let raw = r#"{"path":"a","patch":"PUT 1.=1:\n+x"}"#;
	feed.arg_text(raw.into()).unwrap();
	feed.args_committed(raw.into()).unwrap();
	let stream = edit.call(incoming);
	pin_mut!(stream);
	assert!(matches!(
		stream.next().await,
		Some(Ev::Done(Outcome::Done { result: Err(value), .. })) if value == fault
	));
	assert!(state.lock().committed_ids.is_empty());
}

#[tokio::test]
async fn malformed_complete_hashline_patch_never_crosses_commit_gate() {
	let fake = Fake::success();
	let state = Arc::clone(&fake.state);
	let edit = tool(fake, FormatPolicy::Configured);
	let (feed, incoming) = IncomingParams::channel();
	let raw = r#"{"path":"a","patch":"PUT 1.=:"}"#;
	feed.arg_text(raw.into()).unwrap();
	feed.args_committed(raw.into()).unwrap();
	let stream = edit.call(incoming);
	pin_mut!(stream);
	assert!(matches!(stream.next().await, Some(Ev::Args(_))));
	assert!(state.lock().committed_ids.is_empty());
}

#[tokio::test]
async fn precommit_cancellation_never_reaches_adapter_head() {
	let fake = Fake::success();
	let state = Arc::clone(&fake.state);
	let edit = tool(fake, FormatPolicy::Configured);
	let (feed, incoming) = IncomingParams::channel();
	let prefix = r#"{"path":"a","patch":"PUT 1.=1:\n+x"#;
	feed.arg_text(prefix.into()).unwrap();
	feed
		.interrupt(Interrupt { class: "immediate".into(), reason: "stop".into() })
		.unwrap();
	drop(feed);
	let stream = edit.call(incoming);
	pin_mut!(stream);
	while let Some(event) = stream.next().await {
		if matches!(event, Ev::Aborted(_)) {
			break;
		}
	}
	assert!(state.lock().commits.is_empty());
}

#[tokio::test]
async fn postcommit_interrupt_drops_transaction_without_touching_adapter_head() {
	let head_changed = Arc::new(AtomicBool::new(false));
	let edit = tool(CancelFake { head_changed: Arc::clone(&head_changed) }, FormatPolicy::Configured);
	let (feed, incoming) = IncomingParams::channel();
	let raw = r#"{"path":"a","patch":"PUT 1.=1:\n+x"}"#;
	feed.arg_text(raw.into()).unwrap();
	feed.args_committed(raw.into()).unwrap();
	feed
		.interrupt(Interrupt { class: "immediate".into(), reason: "stop".into() })
		.unwrap();
	let stream = edit.call(incoming);
	pin_mut!(stream);
	let mut terminal = None;
	while let Some(event) = stream.next().await {
		if matches!(event, Ev::Aborted(_)) {
			terminal = Some(event);
			break;
		}
	}
	assert!(matches!(
		terminal,
		Some(Ev::Aborted(omp_tool::Abort::EffectsUnknown { .. }))
	));
	assert!(!head_changed.load(Ordering::SeqCst));
}
