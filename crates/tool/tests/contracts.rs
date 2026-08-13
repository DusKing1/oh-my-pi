//! Observable contracts for typed tools, lowering, invocation input, and
//! history.

use std::{
	convert::Infallible,
	future::{Ready, ready},
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
};

use async_stream::stream;
use bytes::Bytes;
use futures::{FutureExt, Stream, StreamExt, executor::block_on};
use omp_core::Str;
use omp_llm_catalog::GrammarBits;
use omp_llm_inference::{Adjustment, ToolGrammarSyntax};
use omp_tool::{
	Abort, ArgIssue, ArgIssueKind, ArgPath, ArtifactLifetime, BlobRef, CommitError, Constraint,
	ConstraintDisposition, ErasedEv, ErasedOutcome, Ev, ExpectedArtifact, GrammarSyntax,
	IncomingParams, JobRef, LiftedCall, LoweringCaps, Outcome, ParamError, Part, ProjectedCall,
	PromptCaps, RecordedCall, RecordedCallOwned, Registry, RegistryError, Rev, Tool, ToolIdentity,
	ToolSpec, Verdict, VerdictDetails, VerdictSpill, verdict_details,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct FakeParams {
	value: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct FakePayload {
	implementation: Str,
	raw:            Str,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct FakeFault {
	message: Str,
}

struct FakeTool {
	spec:      ToolSpec,
	marker:    Str,
	calls:     Arc<AtomicUsize>,
	lift_from: Option<u16>,
}

impl FakeTool {
	fn new(
		n: u16,
		marker: &str,
		schema: &'static [u8],
		constraint: Constraint,
		calls: Arc<AtomicUsize>,
	) -> Self {
		Self {
			spec: ToolSpec {
				name: Str::from("typed_fake"),
				rev: Rev { family: Str::from("fake"), n },
				description: Str::from(format!("fake revision {n}")),
				schema: Bytes::from_static(schema),
				constraint,
			},
			marker: Str::from(marker),
			calls,
			lift_from: None,
		}
	}

	fn lifting_from(mut self, n: u16) -> Self {
		self.lift_from = Some(n);
		self
	}
}

impl Tool for FakeTool {
	type Fault = FakeFault;
	type Params = FakeParams;
	type Payload = FakePayload;
	type Update = Str;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'c>(
		&'c self,
		mut params: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<Self::Update, Self::Payload, Self::Fault>> + Send + 'c {
		stream! {
			let raw = params.committed().await.expect("test invocation commits its arguments");
			self.calls.fetch_add(1, Ordering::SeqCst);
			yield Ev::Update(self.marker.clone());
			yield Ev::Done(Outcome::Done {
				result: Ok(FakePayload { implementation: self.marker.clone(), raw }),
				useless: false,
			});
		}
	}

	fn prompt(&self, view: Result<&Self::Payload, &Self::Fault>, caps: &PromptCaps) -> Vec<Part> {
		let branch = match view {
			Ok(payload) => format!("ok:{}:{}", payload.implementation, payload.raw),
			Err(fault) => format!("fault:{}", fault.message),
		};
		vec![
			Part::Text {
				text: Str::from(format!(
					"{}|{branch}|{}/{}/{}",
					self.marker, caps.maximum_parts, caps.maximum_text_bytes, caps.media
				)),
			},
			Part::Json { json: Bytes::from(serde_json::to_vec(&branch).expect("string serializes")) },
		]
	}

	fn lift(&self, from: &Rev, call: RecordedCall<'_>) -> Option<LiftedCall> {
		if from.family != self.spec.rev.family || self.lift_from != Some(from.n) {
			return None;
		}
		let suffix = format!(">{}", self.spec.rev.n);
		let mut raw_args = call.raw_args.to_vec();
		raw_args.extend_from_slice(suffix.as_bytes());
		let mut verdict = call.verdict.to_vec();
		verdict.extend_from_slice(suffix.as_bytes());
		Some(LiftedCall { raw_args: Bytes::from(raw_args), verdict: Bytes::from(verdict) })
	}
}

struct PullingTool {
	spec: ToolSpec,
}

impl PullingTool {
	fn new() -> Self {
		Self {
			spec: ToolSpec {
				name:        Str::from("pulling_fake"),
				rev:         Rev { family: Str::from("fake"), n: 1 },
				description: Str::from("pulls one typed argument"),
				schema:      Bytes::from_static(
					br#"{"type":"object","properties":{"wanted":{"type":"number"}}}"#,
				),
				constraint:  Constraint::None,
			},
		}
	}
}

impl Tool for PullingTool {
	type Fault = FakeFault;
	type Params = FakeParams;
	type Payload = FakePayload;
	type Update = Str;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'c>(
		&'c self,
		mut params: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<Self::Update, Self::Payload, Self::Fault>> + Send + 'c {
		stream! {
			let error = params
				.pull(|mut doc| async move {
					let mut root = doc.json();
					let mut object = root.object();
					let mut value = object.key("wanted");
					value.number().await
				})
				.await
				.expect_err("test supplies a mistyped pulled value");
			let ParamError::Args(issue) = error else {
				panic!("typed pull must report an argument issue")
			};
			yield Ev::Args(issue);
			yield Ev::Update(Str::from("post-terminal update"));
			yield Ev::Done(Outcome::Done {
				result: Ok(FakePayload {
					implementation: Str::from("post-terminal"),
					raw: Str::from("must not escape"),
				}),
				useless: false,
			});
		}
	}

	fn prompt(&self, _view: Result<&Self::Payload, &Self::Fault>, _caps: &PromptCaps) -> Vec<Part> {
		Vec::new()
	}
}

struct AbortingTool {
	spec: ToolSpec,
}

impl AbortingTool {
	fn new() -> Self {
		Self {
			spec: ToolSpec {
				name:        Str::from("aborting_fake"),
				rev:         Rev { family: Str::from("fake"), n: 1 },
				description: Str::from("aborts before completion"),
				schema:      Bytes::from_static(br#"{"type":"object"}"#),
				constraint:  Constraint::None,
			},
		}
	}
}

impl Tool for AbortingTool {
	type Fault = FakeFault;
	type Params = FakeParams;
	type Payload = FakePayload;
	type Update = Str;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'c>(
		&'c self,
		params: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<Self::Update, Self::Payload, Self::Fault>> + Send + 'c {
		drop(params);
		stream! {
			yield Ev::Aborted(Abort::Skipped { reason: Str::from("policy denied") });
			yield Ev::Update(Str::from("post-terminal update"));
			yield Ev::Done(Outcome::Done {
				result: Err(FakeFault { message: Str::from("must not escape") }),
				useless: false,
			});
		}
	}

	fn prompt(&self, _view: Result<&Self::Payload, &Self::Fault>, _caps: &PromptCaps) -> Vec<Part> {
		Vec::new()
	}
}

fn fake_tool(n: u16, marker: &str, calls: Arc<AtomicUsize>) -> FakeTool {
	FakeTool::new(
		n,
		marker,
		br#"{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]}"#,
		Constraint::None,
		calls,
	)
}

fn identity(n: u16) -> ToolIdentity {
	ToolIdentity { name: Str::from("typed_fake"), rev: Rev { family: Str::from("fake"), n } }
}

#[test]
fn duplicate_registration_never_replaces_the_erased_implementation() {
	let original_calls = Arc::new(AtomicUsize::new(0));
	let rejected_calls = Arc::new(AtomicUsize::new(0));
	let mut registry = Registry::new();
	registry
		.register(fake_tool(1, "original", Arc::clone(&original_calls)))
		.expect("first typed registration succeeds");
	let error = registry
		.register(fake_tool(1, "replacement", Arc::clone(&rejected_calls)))
		.expect_err("the same durable revision is erased only once");
	assert!(
		matches!(error, RegistryError::Duplicate(name, rev) if name == "typed_fake" && rev == identity(1).rev)
	);

	let (feed, params) = IncomingParams::channel();
	feed
		.args_committed(Str::from("{value:1}"))
		.expect("consumer remains live");
	let events = block_on(
		registry
			.invoke("typed_fake", params)
			.expect("live tool is invokable")
			.collect::<Vec<_>>(),
	);
	assert_eq!(original_calls.load(Ordering::SeqCst), 1);
	assert_eq!(rejected_calls.load(Ordering::SeqCst), 0);
	let [
		Ok(ErasedEv::Update(update)),
		Ok(ErasedEv::Done(ErasedOutcome::Done { verdict, useless: false })),
	] = events.as_slice()
	else {
		panic!("expected an erased update and terminal outcome: {events:?}")
	};
	assert_eq!(
		serde_json::from_slice::<Str>(update)
			.expect("typed update remains recoverable after erasure"),
		"original"
	);
	let verdict: Verdict<FakePayload, FakeFault> =
		serde_json::from_slice(verdict).expect("typed verdict remains recoverable after erasure");
	assert_eq!(
		verdict,
		Verdict::Ok(FakePayload {
			implementation: Str::from("original"),
			raw:            Str::from("{value:1}"),
		})
	);
}

#[test]
fn erased_tool_does_not_run_before_explicit_argument_commitment() {
	let calls = Arc::new(AtomicUsize::new(0));
	let mut registry = Registry::new();
	registry
		.register(fake_tool(1, "gated", Arc::clone(&calls)))
		.unwrap();
	let (feed, params) = IncomingParams::channel();
	let mut events = registry.invoke("typed_fake", params).unwrap();

	assert!(events.next().now_or_never().is_none());
	assert_eq!(calls.load(Ordering::SeqCst), 0);

	feed.args_committed(Str::from("{value:1}")).unwrap();
	assert!(matches!(block_on(events.next()), Some(Ok(ErasedEv::Update(_)))));
	assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn pulled_mismatch_erases_to_args_verdict_and_fuses_every_later_event() {
	let mut registry = Registry::new();
	registry.register(PullingTool::new()).unwrap();
	let raw = r#"{"wanted":"seven","ignored":true}"#;
	let (feed, params) = IncomingParams::channel();
	feed.args_committed(Str::from(raw)).unwrap();

	let events = block_on(
		registry
			.invoke("pulling_fake", params)
			.unwrap()
			.collect::<Vec<_>>(),
	);
	let [Ok(ErasedEv::Done(ErasedOutcome::Done { verdict, useless: false }))] = events.as_slice()
	else {
		panic!("Args must be the sole erased terminal event: {events:?}")
	};
	let verdict: Verdict<FakePayload, FakeFault> = serde_json::from_slice(verdict).unwrap();
	assert_eq!(
		verdict,
		Verdict::Args(ArgIssue {
			path:     vec![ArgPath::Key(Str::from("wanted"))],
			expected: Str::from("number"),
			kind:     ArgIssueKind::TypeMismatch,
			example:  None,
			found:    Some(Str::from("string")),
		})
	);
}

#[test]
fn aborted_verdict_is_terminal_and_fuses_every_later_event() {
	let mut registry = Registry::new();
	registry.register(AbortingTool::new()).unwrap();
	let (_feed, params) = IncomingParams::channel();

	let events = block_on(
		registry
			.invoke("aborting_fake", params)
			.unwrap()
			.collect::<Vec<_>>(),
	);
	let [Ok(ErasedEv::Done(ErasedOutcome::Done { verdict, useless: false }))] = events.as_slice()
	else {
		panic!("Aborted must be the sole erased terminal event: {events:?}")
	};
	let verdict: Verdict<FakePayload, FakeFault> = serde_json::from_slice(verdict).unwrap();
	assert_eq!(verdict, Verdict::Aborted(Abort::Skipped { reason: Str::from("policy denied") }));
}

#[test]
fn advertisement_contains_only_the_live_schema_and_preserves_supported_grammar() {
	let calls = Arc::new(AtomicUsize::new(0));
	let mut registry = Registry::new();
	registry
		.register(FakeTool::new(
			1,
			"old",
			br#"{"type":"object","properties":{"old":{"type":"boolean"}}}"#,
			Constraint::None,
			Arc::clone(&calls),
		))
		.unwrap();
	registry
		.register(FakeTool::new(
			2,
			"live",
			br#"{"type":"object","properties":{"live":{"const":true}},"required":["live"]}"#,
			Constraint::Grammar {
				syntax:     GrammarSyntax::Regex,
				definition: Str::from(r"live=(true|false)"),
				priority:   7,
			},
			calls,
		))
		.unwrap();

	let advertised =
		registry.advertise(LoweringCaps { strict_schema: false, grammar: GrammarBits::REGEX });
	let [tool] = advertised.as_slice() else {
		panic!("historical revisions must not be advertised")
	};
	assert_eq!(tool.identity, identity(2));
	assert_eq!(tool.definition.name, "typed_fake");
	assert_eq!(tool.definition.description.as_deref(), Some("fake revision 2"));
	let grammar = tool
		.definition
		.input
		.grammar()
		.expect("supported grammar remains native");
	assert_eq!(grammar.syntax, ToolGrammarSyntax::Regex);
	assert_eq!(grammar.definition, r"live=(true|false)");
	assert_eq!(tool.disposition, Some(ConstraintDisposition::Required));
	assert_eq!(tool.priority, Some(7));
	assert!(tool.adjustments.is_empty());
}

#[test]
fn live_identity_and_advertisement_are_the_same_exact_revision() {
	let calls = Arc::new(AtomicUsize::new(0));
	let mut registry = Registry::new();
	registry
		.register(FakeTool::new(
			1,
			"historical",
			br#"{"type":"object","properties":{"hl1_only":{"const":true}}}"#,
			Constraint::None,
			Arc::clone(&calls),
		))
		.unwrap();
	registry
		.register(FakeTool::new(
			2,
			"live",
			br#"{"type":"object","properties":{"hl2_only":{"const":true}}}"#,
			Constraint::None,
			calls,
		))
		.unwrap();

	let (name, revision) = registry
		.live_identity("typed_fake")
		.expect("registered live identity");
	let [advertised] = registry
		.advertise(LoweringCaps { strict_schema: false, grammar: GrammarBits::empty() })
		.try_into()
		.expect("only one live definition");
	assert_eq!(name, &advertised.identity.name);
	assert_eq!(revision, &advertised.identity.rev);
	assert_eq!(revision.to_string(), "fake.2");
	let (schema, _) = advertised
		.definition
		.input
		.json_schema()
		.expect("unconstrained tool lowers to JSON Schema");
	let schema_bytes = serde_json::to_vec(schema.as_value()).expect("schema serializes");
	assert!(
		schema_bytes
			.windows(b"hl2_only".len())
			.any(|window| window == b"hl2_only")
	);
	assert!(
		!schema_bytes
			.windows(b"hl1_only".len())
			.any(|window| window == b"hl1_only")
	);
}

#[test]
fn unsupported_grammar_degrades_to_live_lenient_schema_with_a_receipt() {
	let live_schema = json!({
		"type": "object",
		"properties": {"live": {"const": true}},
		"required": ["live"]
	});
	let mut registry = Registry::new();
	registry
		.register(FakeTool::new(
			1,
			"old",
			br#"{"type":"object","properties":{"obsolete":{"type":"string"}}}"#,
			Constraint::None,
			Arc::new(AtomicUsize::new(0)),
		))
		.unwrap();
	registry
		.register(FakeTool::new(
			2,
			"live",
			br#"{"type":"object","properties":{"live":{"const":true}},"required":["live"]}"#,
			Constraint::Grammar {
				syntax:     GrammarSyntax::Ebnf,
				definition: Str::from("root = 'live';"),
				priority:   11,
			},
			Arc::new(AtomicUsize::new(0)),
		))
		.unwrap();

	let [tool] = registry
		.advertise(LoweringCaps { strict_schema: true, grammar: GrammarBits::empty() })
		.try_into()
		.expect("one live tool");
	assert_eq!(tool.identity, identity(2));
	let (schema, strict) = tool
		.definition
		.input
		.json_schema()
		.expect("unsupported grammar falls back to JSON Schema");
	assert_eq!(schema.as_value(), &live_schema);
	assert!(!strict, "grammar fallback must remain non-strict even when strict schema is available");
	assert_eq!(tool.disposition, Some(ConstraintDisposition::Prefer));
	assert_eq!(tool.priority, Some(11));
	assert_eq!(tool.adjustments.len(), 1);
	assert!(matches!(
		&tool.adjustments[0],
		Adjustment::Dropped { feature, reason }
			if feature.0 == "tool.typed_fake.ebnf" && reason.0 == "catalog.grammar-unsupported"
	));
}

#[test]
fn pull_validates_only_the_requested_value_and_ignores_unknown_malformed_json() {
	let raw = r#"{"wanted":7,"unknown":[}"#;
	let (feed, mut params) = IncomingParams::channel();
	feed.args_committed(Str::from(raw)).unwrap();

	let wanted = block_on(params.pull(|mut doc| async move {
		let mut root = doc.json();
		let mut object = root.object();
		let mut value = object.key("wanted");
		value.number().await
	}))
	.expect("an unknown unpulled sibling cannot fail the requested pull");
	assert_eq!(wanted.as_f64(), 7.0);
	assert_eq!(block_on(params.committed()).unwrap(), raw);
}

#[test]
fn pulled_type_failure_is_a_structured_argument_issue() {
	let raw = r#"{"wanted":"seven","unknown":[}"#;
	let (feed, mut params) = IncomingParams::channel();
	feed.args_committed(Str::from(raw)).unwrap();

	let error = block_on(params.pull(|mut doc| async move {
		let mut root = doc.json();
		let mut object = root.object();
		let mut value = object.key("wanted");
		value.number().await
	}))
	.expect_err("the requested number has the wrong shape");
	let ParamError::Args(issue) = error else {
		panic!("pull failures must retain their structured argument issue")
	};
	assert_eq!(issue.path, vec![ArgPath::Key(Str::from("wanted"))]);
	assert_eq!(issue.kind, ArgIssueKind::TypeMismatch);
	assert_eq!(issue.expected, "number");
	assert_eq!(issue.found.as_deref(), Some("string"));
}

#[test]
fn commitment_is_explicit_and_feed_guard_drop_aborts() {
	let (feed, mut committed) = IncomingParams::channel();
	feed.arg_text(Str::from("{value:1}")).unwrap();
	feed.args_committed(Str::from("{value:1}")).unwrap();
	assert_eq!(block_on(committed.committed()).unwrap(), "{value:1}");

	let (guard, mut abandoned) = IncomingParams::channel();
	guard.arg_text(Str::from("{value:")).unwrap();
	drop(guard);
	assert!(matches!(block_on(abandoned.committed()), Err(CommitError::Aborted)));
}

#[test]
fn prompt_projection_is_exact_and_deterministic_for_the_same_input() {
	let mut registry = Registry::new();
	registry
		.register(fake_tool(1, "renderer", Arc::new(AtomicUsize::new(0))))
		.unwrap();
	let verdict = serde_json::to_vec(&Verdict::<FakePayload, FakeFault>::Ok(FakePayload {
		implementation: Str::from("engine"),
		raw:            Str::from("{value:9}"),
	}))
	.unwrap();
	let caps =
		PromptCaps { maximum_parts: 3, maximum_text_bytes: 256, media: true };

	let first = registry
		.prompt(&identity(1), &verdict, &caps)
		.unwrap()
		.unwrap();
	let second = registry
		.prompt(&identity(1), &verdict, &caps)
		.unwrap()
		.unwrap();
	assert_eq!(first, second);
	assert_eq!(first, vec![
		Part::Text { text: Str::from("renderer|ok:engine:{value:9}|3/256/true") },
		Part::Json { json: Bytes::from_static(br#""ok:engine:{value:9}""#) },
	]);
}

#[test]
fn all_adjacent_lifts_compose_to_the_live_revision_byte_identically() {
	let mut registry = Registry::new();
	registry
		.register(fake_tool(1, "one", Arc::new(AtomicUsize::new(0))))
		.unwrap();
	registry
		.register(fake_tool(2, "two", Arc::new(AtomicUsize::new(0))).lifting_from(1))
		.unwrap();
	registry
		.register(fake_tool(3, "three", Arc::new(AtomicUsize::new(0))).lifting_from(2))
		.unwrap();
	let original = RecordedCallOwned {
		identity: identity(1),
		raw_args: Bytes::from_static(b"raw"),
		verdict:  Bytes::from_static(b"verdict"),
	};

	let first = registry.project(original.clone());
	let second = registry.project(original);
	assert_eq!(first, second, "same projection inputs must produce identical bytes");
	assert_eq!(
		first,
		ProjectedCall::Live(RecordedCallOwned {
			identity: identity(3),
			raw_args: Bytes::from_static(b"raw>2>3"),
			verdict:  Bytes::from_static(b"verdict>2>3"),
		})
	);
}

#[test]
fn incomplete_lift_chain_preserves_the_exact_original_as_data() {
	let mut registry = Registry::new();
	registry
		.register(fake_tool(1, "one", Arc::new(AtomicUsize::new(0))))
		.unwrap();
	registry
		.register(fake_tool(3, "three", Arc::new(AtomicUsize::new(0))).lifting_from(2))
		.unwrap();
	let original = RecordedCallOwned {
		identity: identity(1),
		raw_args: Bytes::from_static(b"{ not rewritten "),
		verdict:  Bytes::from_static(b"opaque verdict bytes\0\xff"),
	};

	assert_eq!(registry.project(original.clone()), ProjectedCall::Data(original));
}

struct RecordingSpill {
	tx: flume::Sender<Bytes>,
	rx: flume::Receiver<Bytes>,
}

impl RecordingSpill {
	fn new() -> Self {
		let (tx, rx) = flume::unbounded();
		Self { tx, rx }
	}
}

impl VerdictSpill for RecordingSpill {
	type Error = Infallible;

	fn spill(&self, json: Bytes) -> Ready<Result<BlobRef, Self::Error>> {
		self
			.tx
			.send(json.clone())
			.expect("test receiver remains live");
		ready(Ok(BlobRef {
			hash:       Str::from("sha256:fake"),
			media_type: Str::from("application/json"),
			byte_len:   json.len() as u64,
		}))
	}
}

#[test]
fn verdict_spill_hook_runs_only_beyond_the_inline_boundary_with_exact_bytes() {
	let verdict = Verdict::<FakePayload, FakeFault>::Ok(FakePayload {
		implementation: Str::from("engine"),
		raw:            Str::from("{value:5}"),
	});
	let expected = Bytes::from(serde_json::to_vec(&verdict).unwrap());
	let spill = RecordingSpill::new();

	let inline = block_on(verdict_details(&verdict, expected.len(), &spill)).unwrap();
	assert_eq!(inline, VerdictDetails::Inline { json: expected.clone() });
	assert!(spill.rx.try_recv().is_err());

	let spilled = block_on(verdict_details(&verdict, expected.len() - 1, &spill)).unwrap();
	assert_eq!(spilled, VerdictDetails::Spilled {
		blob:     BlobRef {
			hash:       Str::from("sha256:fake"),
			media_type: Str::from("application/json"),
			byte_len:   expected.len() as u64,
		},
		byte_len: expected.len() as u64,
	});
	assert_eq!(spill.rx.try_recv().unwrap(), expected);
	assert!(spill.rx.try_recv().is_err());
}

#[test]
fn detached_artifact_lifetime_is_explicit_and_session_is_the_conservative_default() {
	assert_eq!(ArtifactLifetime::default(), ArtifactLifetime::Session);

	for (lifetime, encoded) in [
		(ArtifactLifetime::Ephemeral, "ephemeral"),
		(ArtifactLifetime::Session, "session"),
		(ArtifactLifetime::Durable, "durable"),
	] {
		let job = JobRef {
			id:       Str::from("job-7"),
			artifact: ExpectedArtifact {
				description: Str::from("rendered video"),
				media_type: Some(Str::from("video/mp4")),
				lifetime,
			},
		};
		let value = serde_json::to_value(&job).expect("job reference serializes");
		assert_eq!(value["artifact"]["lifetime"], encoded);
		assert_eq!(
			serde_json::from_value::<JobRef>(value).expect("explicit lifetime deserializes"),
			job
		);
	}

	assert!(
		serde_json::from_value::<JobRef>(json!({
			"id": "job-7",
			"artifact": {
				"description": "rendered video",
				"media_type": "video/mp4"
			}
		}))
		.is_err(),
		"wire descriptors must carry an explicit lifetime"
	);
}
