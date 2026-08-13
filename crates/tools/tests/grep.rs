//! Registry-level contracts for deterministic, cancellable exact workspace search.

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
	result:  grep::SearchResult,
	seen:    Arc<Mutex<Vec<grep::SearchRequest>>>,
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
		request: grep::SearchRequest,
	) -> impl Future<Output = Result<grep::SearchResult, grep::Fault>> + Send + '_ {
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
			Ok(result)
		}
	}

	fn glob(
		&self,
		_request: glob::WalkRequest,
	) -> impl Future<Output = Result<glob::WalkResult, glob::Fault>> + Send + '_ {
		async { Err(glob::Fault::Workspace { message: Str::from("unused fake glob boundary") }) }
	}
}

fn matched(path: &str, line: u64, start: u64, end: u64, text: &str) -> grep::SearchMatch {
	grep::SearchMatch {
		path: Str::from(path),
		line,
		spans: vec![grep::ByteSpan { start, end }],
		line_text: Str::from(text),
	}
}

fn fake(result: grep::SearchResult) -> FakeWorkspace {
	FakeWorkspace {
		result,
		seen: Arc::new(Mutex::new(Vec::new())),
		pending: false,
		dropped: Arc::new(AtomicBool::new(false)),
	}
}

fn invoke(workspace: FakeWorkspace, raw: &str) -> Verdict<grep::Payload, grep::Fault> {
	let mut registry = Registry::new();
	registry
		.register(grep::tool(workspace))
		.expect("grep schema and revision register");
	let (feed, params) = IncomingParams::channel();
	feed
		.args_committed(Str::from(raw))
		.expect("invocation consumer remains live");
	let events = block_on(
		registry
			.invoke("grep", params)
			.expect("registered grep is invokable")
			.collect::<Vec<_>>(),
	);
	let [Ok(ErasedEv::Done(ErasedOutcome::Done { verdict, useless: false }))] = events.as_slice()
	else {
		panic!("expected one terminal grep event: {events:?}");
	};
	serde_json::from_slice(verdict).expect("typed grep verdict survives registry erasure")
}

#[test]
fn exact_matches_are_sorted_capped_and_retain_multibyte_line_truth() {
	let workspace = fake(grep::SearchResult {
		matches:        vec![
			matched("z.rs", 9, 0, 6, "needle"),
			matched("a.rs", 3, 2, 8, "λneedle tail"),
			matched("a.rs", 1, 4, 10, "two needle"),
		],
		binary_skipped: vec![Str::from("z.bin"), Str::from("a.bin")],
		truncated:      false,
	});
	let seen = Arc::clone(&workspace.seen);
	let verdict = invoke(
		workspace,
		r#"{"path":"src","patterns":["needle"],"include":["**/*.rs"],"exclude":["vendor/**"],"gitignore":false,"hidden":true,"case_sensitive":true,"mode":"fixed","limit":2}"#,
	);
	let Verdict::Ok(payload) = verdict else {
		panic!("expected successful grep verdict");
	};
	assert_eq!(payload.matches.len(), 2);
	assert_eq!(payload.matches[0], matched("a.rs", 1, 4, 10, "two needle"));
	assert_eq!(payload.matches[1], matched("a.rs", 3, 2, 8, "λneedle tail"));
	assert_eq!(payload.matches[1].spans[0], grep::ByteSpan { start: 2, end: 8 });
	assert_eq!(payload.matches[1].line_text, "λneedle tail");
	assert_eq!(payload.binary_skipped, [Str::from("a.bin"), Str::from("z.bin")]);
	assert!(payload.truncated);

	let requests = seen.lock();
	assert_eq!(requests.len(), 1);
	assert_eq!(requests[0].path, "src");
	assert_eq!(requests[0].patterns, [Str::from("needle")]);
	assert_eq!(requests[0].include, [Str::from("**/*.rs")]);
	assert_eq!(requests[0].exclude, [Str::from("vendor/**")]);
	assert!(!requests[0].gitignore);
	assert!(requests[0].hidden);
	assert_eq!(requests[0].limit, 3);
}

#[test]
fn zero_limit_returns_no_records_and_reports_resource_truth() {
	let workspace = fake(grep::SearchResult {
		matches:        vec![matched("one.rs", 1, 0, 1, "x")],
		binary_skipped: vec![Str::from("raw.bin")],
		truncated:      false,
	});
	let seen = Arc::clone(&workspace.seen);
	let verdict = invoke(workspace, r#"{"patterns":["x"],"limit":0}"#);
	let Verdict::Ok(payload) = verdict else {
		panic!("zero remains a successful hard limit");
	};
	assert!(payload.matches.is_empty());
	assert!(payload.truncated);
	assert_eq!(payload.binary_skipped, [Str::from("raw.bin")]);
	assert_eq!(seen.lock()[0].limit, 1);
}

#[test]
fn unsupported_search_semantics_are_typed_faults_not_approximations() {
	let regex = fake(grep::SearchResult {
		matches:        Vec::new(),
		binary_skipped: Vec::new(),
		truncated:      false,
	});
	let regex_seen = Arc::clone(&regex.seen);
	assert!(matches!(
		invoke(regex, r#"{"patterns":["n.*e"],"mode":"regex","limit":10}"#),
		Verdict::Fault(grep::Fault::UnsupportedPatternMode { mode: grep::PatternMode::Regex })
	));
	assert!(regex_seen.lock().is_empty());

	let insensitive = fake(grep::SearchResult {
		matches:        Vec::new(),
		binary_skipped: Vec::new(),
		truncated:      false,
	});
	let insensitive_seen = Arc::clone(&insensitive.seen);
	assert!(matches!(
		invoke(insensitive, r#"{"patterns":["Needle"],"case_sensitive":false,"limit":10}"#),
		Verdict::Fault(grep::Fault::CaseInsensitiveUnsupported)
	));
	assert!(insensitive_seen.lock().is_empty());
}

#[test]
fn malformed_pulled_params_become_args_verdicts() {
	let workspace = fake(grep::SearchResult {
		matches:        Vec::new(),
		binary_skipped: Vec::new(),
		truncated:      false,
	});
	let seen = Arc::clone(&workspace.seen);
	let verdict = invoke(workspace, r#"{"patterns":"needle","limit":4}"#);
	let Verdict::Args(issue) = verdict else {
		panic!("mistyped pulled patterns must be an args verdict");
	};
	assert!(matches!(issue.kind, ArgIssueKind::TypeMismatch | ArgIssueKind::Malformed));
	assert!(seen.lock().is_empty());
}

#[test]
fn owner_drop_before_commit_is_an_input_drop_abort() {
	let workspace = fake(grep::SearchResult {
		matches: Vec::new(),
		binary_skipped: Vec::new(),
		truncated: false,
	});
	let mut registry = Registry::new();
	registry.register(grep::tool(workspace)).expect("grep registers");
	let (feed, params) = IncomingParams::channel();
	drop(feed);
	let events = block_on(registry.invoke("grep", params).unwrap().collect::<Vec<_>>());
	let [Ok(ErasedEv::Done(ErasedOutcome::Done { verdict, .. }))] = events.as_slice() else {
		panic!("input drop must settle once: {events:?}");
	};
	let verdict: Verdict<grep::Payload, grep::Fault> =
		serde_json::from_slice(verdict).expect("abort verdict decodes");
	assert_eq!(verdict, Verdict::Aborted(omp_tool::Abort::InputDropped));
}

#[test]
fn interrupt_drops_the_active_search_future() {
	let mut workspace = fake(grep::SearchResult {
		matches:        Vec::new(),
		binary_skipped: Vec::new(),
		truncated:      false,
	});
	workspace.pending = true;
	let dropped = Arc::clone(&workspace.dropped);
	let mut registry = Registry::new();
	registry
		.register(grep::tool(workspace))
		.expect("grep registers");
	let (feed, params) = IncomingParams::channel();
	let raw = r#"{"patterns":["needle"],"limit":10}"#;
	feed
		.args_committed(Str::from(raw))
		.expect("commit reaches executor");
	feed
		.interrupt(Interrupt { class: Str::from("immediate"), reason: Str::from("stop walking") })
		.expect("interrupt reaches executor");
	let events = block_on(registry.invoke("grep", params).unwrap().collect::<Vec<_>>());
	let [Ok(ErasedEv::Done(ErasedOutcome::Done { verdict, .. }))] = events.as_slice() else {
		panic!("interrupt must settle once: {events:?}");
	};
	let verdict: Verdict<grep::Payload, grep::Fault> =
		serde_json::from_slice(verdict).expect("abort verdict decodes");
	assert!(matches!(
		verdict,
		Verdict::Aborted(omp_tool::Abort::Interrupted { reason }) if reason == "stop walking"
	));
	assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn prompt_caps_keep_only_whole_utf8_match_records() {
	let tool = grep::tool(fake(grep::SearchResult {
		matches:        Vec::new(),
		binary_skipped: Vec::new(),
		truncated:      false,
	}));
	let first = matched("λ.rs", 1, 0, 2, "λambda");
	let second = matched("z.rs", 2, 0, 1, "z");
	let first_record = "λ.rs:1:0-2: λambda\n";
	let payload = grep::Payload {
		matches:        vec![first, second],
		binary_skipped: Vec::new(),
		truncated:      false,
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
				maximum_parts:      0,
				maximum_text_bytes: u32::MAX,
				media:              false,
			})
			.is_empty()
	);
}
