//! Registry-level contracts for deterministic, cancellable workspace path matching.

use std::{
	future::{Future, pending},
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
};

use futures::{StreamExt, executor::block_on};
use omp_core::Str;
use omp_tool::{
	ArgIssueKind, ErasedEv, ErasedOutcome, IncomingParams, Interrupt, Part, PromptCaps, Registry,
	Tool, Verdict,
};
use omp_tools::{glob, grep};
use parking_lot::Mutex;

#[derive(Clone)]
struct FakeWorkspace {
	result:  Result<glob::WalkResult, glob::Fault>,
	seen:    Arc<Mutex<Vec<glob::WalkRequest>>>,
	pending: bool,
	dropped: Arc<AtomicBool>,
}

struct ActiveGuard(Arc<AtomicBool>);

impl Drop for ActiveGuard {
	fn drop(&mut self) {
		self.0.store(true, Ordering::SeqCst);
	}
}

impl grep::WorkspaceSearch for FakeWorkspace {
	fn search(
		&self,
		_request: grep::SearchRequest,
	) -> impl Future<Output = Result<grep::SearchResult, grep::Fault>> + Send + '_ {
		async { Err(grep::Fault::Workspace { message: Str::from("unused fake search boundary") }) }
	}

	fn glob(
		&self,
		request: glob::WalkRequest,
	) -> impl Future<Output = Result<glob::WalkResult, glob::Fault>> + Send + '_ {
		let result = self.result.clone();
		let seen = Arc::clone(&self.seen);
		let dropped = Arc::clone(&self.dropped);
		let remains_pending = self.pending;
		async move {
			seen.lock().push(request);
			let _guard = remains_pending.then(|| ActiveGuard(dropped));
			if remains_pending {
				pending::<()>().await;
			}
			result
		}
	}
}

fn fake(result: Result<glob::WalkResult, glob::Fault>) -> FakeWorkspace {
	FakeWorkspace {
		result,
		seen: Arc::new(Mutex::new(Vec::new())),
		pending: false,
		dropped: Arc::new(AtomicBool::new(false)),
	}
}

fn invoke(workspace: FakeWorkspace, raw: &str) -> Verdict<glob::Payload, glob::Fault> {
	let mut registry = Registry::new();
	registry
		.register(glob::tool(workspace))
		.expect("glob schema and revision register");
	let (feed, params) = IncomingParams::channel();
	feed
		.args_committed(Str::from(raw))
		.expect("invocation consumer remains live");
	let events = block_on(
		registry
			.invoke("glob", params)
			.expect("registered glob is invokable")
			.collect::<Vec<_>>(),
	);
	let [Ok(ErasedEv::Done(ErasedOutcome::Done { verdict, useless: false }))] = events.as_slice()
	else {
		panic!("expected one terminal glob event: {events:?}");
	};
	serde_json::from_slice(verdict).expect("typed glob verdict survives registry erasure")
}

#[test]
fn paths_are_sorted_capped_and_controls_are_forwarded_exactly() {
	let workspace = fake(Ok(glob::WalkResult {
		paths:     vec![Str::from("z.rs"), Str::from("a.rs"), Str::from("m.rs")],
		truncated: false,
	}));
	let seen = Arc::clone(&workspace.seen);
	let verdict = invoke(
		workspace,
		r#"{"path":"src","patterns":["**/*.rs","build.rs"],"exclude":["generated/**"],"gitignore":false,"hidden":true,"limit":2}"#,
	);
	assert_eq!(
		verdict,
		Verdict::Ok(glob::Payload {
			paths:     vec![Str::from("a.rs"), Str::from("m.rs")],
			truncated: true,
		})
	);
	let requests = seen.lock();
	assert_eq!(requests.len(), 1);
	assert_eq!(requests[0].path, "src");
	assert_eq!(requests[0].patterns, [Str::from("**/*.rs"), Str::from("build.rs")]);
	assert_eq!(requests[0].exclude, [Str::from("generated/**")]);
	assert!(!requests[0].gitignore);
	assert!(requests[0].hidden);
	assert_eq!(requests[0].limit, 3);
}

#[test]
fn zero_limit_preserves_the_truncated_boundary() {
	let workspace =
		fake(Ok(glob::WalkResult { paths: vec![Str::from("present.rs")], truncated: false }));
	let seen = Arc::clone(&workspace.seen);
	assert_eq!(
		invoke(workspace, r#"{"patterns":["**"],"limit":0}"#),
		Verdict::Ok(glob::Payload { paths: Vec::new(), truncated: true })
	);
	assert_eq!(seen.lock()[0].limit, 1);

	let empty = fake(Ok(glob::WalkResult { paths: Vec::new(), truncated: false }));
	assert_eq!(
		invoke(empty, r#"{"patterns":["**"],"limit":0}"#),
		Verdict::Ok(glob::Payload { paths: Vec::new(), truncated: false })
	);
}

#[test]
fn invalid_workspace_glob_is_a_typed_fault() {
	let workspace = fake(Err(glob::Fault::InvalidPattern {
		pattern: Str::from("["),
		message: Str::from("unclosed character class"),
	}));
	assert_eq!(
		invoke(workspace, r#"{"patterns":["["],"limit":10}"#),
		Verdict::Fault(glob::Fault::InvalidPattern {
			pattern: Str::from("["),
			message: Str::from("unclosed character class"),
		})
	);
}

#[test]
fn malformed_pulled_params_become_args_verdicts() {
	let workspace = fake(Ok(glob::WalkResult { paths: Vec::new(), truncated: false }));
	let seen = Arc::clone(&workspace.seen);
	let verdict = invoke(workspace, r#"{"patterns":{"not":"an array"},"limit":4}"#);
	let Verdict::Args(issue) = verdict else {
		panic!("mistyped pulled patterns must be an args verdict");
	};
	assert!(matches!(issue.kind, ArgIssueKind::TypeMismatch | ArgIssueKind::Malformed));
	assert!(seen.lock().is_empty());
}

#[test]
fn owner_drop_before_commit_is_an_input_drop_abort() {
	let workspace = fake(Ok(glob::WalkResult { paths: Vec::new(), truncated: false }));
	let mut registry = Registry::new();
	registry.register(glob::tool(workspace)).expect("glob registers");
	let (feed, params) = IncomingParams::channel();
	drop(feed);
	let events = block_on(registry.invoke("glob", params).unwrap().collect::<Vec<_>>());
	let [Ok(ErasedEv::Done(ErasedOutcome::Done { verdict, .. }))] = events.as_slice() else {
		panic!("input drop must settle once: {events:?}");
	};
	let verdict: Verdict<glob::Payload, glob::Fault> =
		serde_json::from_slice(verdict).expect("abort verdict decodes");
	assert_eq!(verdict, Verdict::Aborted(omp_tool::Abort::InputDropped));
}

#[test]
fn interrupt_drops_the_active_walk_future() {
	let mut workspace = fake(Ok(glob::WalkResult { paths: Vec::new(), truncated: false }));
	workspace.pending = true;
	let dropped = Arc::clone(&workspace.dropped);
	let mut registry = Registry::new();
	registry
		.register(glob::tool(workspace))
		.expect("glob registers");
	let (feed, params) = IncomingParams::channel();
	let raw = r#"{"patterns":["**/*.rs"],"limit":10}"#;
	feed
		.args_committed(Str::from(raw))
		.expect("commit reaches executor");
	feed
		.interrupt(Interrupt { class: Str::from("immediate"), reason: Str::from("stop walking") })
		.expect("interrupt reaches executor");
	let events = block_on(registry.invoke("glob", params).unwrap().collect::<Vec<_>>());
	let [Ok(ErasedEv::Done(ErasedOutcome::Done { verdict, .. }))] = events.as_slice() else {
		panic!("interrupt must settle once: {events:?}");
	};
	let verdict: Verdict<glob::Payload, glob::Fault> =
		serde_json::from_slice(verdict).expect("abort verdict decodes");
	assert!(matches!(
		verdict,
		Verdict::Aborted(omp_tool::Abort::Interrupted { reason }) if reason == "stop walking"
	));
	assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn prompt_caps_keep_only_whole_utf8_path_records() {
	let tool = glob::tool(fake(Ok(glob::WalkResult { paths: Vec::new(), truncated: false })));
	let first_record = "src/λ.rs\n";
	let payload = glob::Payload {
		paths:     vec![Str::from("src/λ.rs"), Str::from("src/z.rs")],
		truncated: false,
	};
	let parts = tool.prompt(Ok(&payload), &PromptCaps {
		maximum_parts:      1,
		maximum_text_bytes: u32::try_from(first_record.len()).unwrap(),
		media:              false,
	});
	assert_eq!(parts, vec![Part::Text { text: Str::from(first_record) }]);
	assert!(
		tool
			.prompt(Ok(&payload), &PromptCaps {
				maximum_parts:      1,
				maximum_text_bytes: 0,
				media:              false,
			})
			.is_empty()
	);
}
