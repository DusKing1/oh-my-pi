//! End-to-end turn protocol tests.
use std::{
	collections::BTreeMap,
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicBool, AtomicUsize, Ordering},
	},
	task::{Context, Poll},
};

use bytes::Bytes;
use futures::{
	Stream, StreamExt,
	stream::{self, BoxStream},
};
use omp_core::SmolStr;
use omp_llm_catalog::{
	models::{Availability, Modality, ModelCard, ModelCatalog, Source},
	provider::Facet as CatalogFacet,
	registry::{CredentialView, Registry},
};
use omp_llm_gateway::{
	context::ContextStore,
	turn::{ChatResolver, ChatRoute, TurnEngine, TurnStream},
};
use omp_llm_types::{
	ChatOutcome, ChatRequest, ContextRef, ExecOutcome, ExecStatus, Invoke, InvokeChannel,
	InvokeChunk, InvokeComplete, InvokeInput, InvokePayload, Item, ItemKind, Message, Part, Props,
	Revision, Role, StopReason, Thread, ThreadDelta, ToolCall, ToolResult, TurnError, TurnErrorKind,
	TurnEvent,
	facet::{Chat, Error as FacetError, Executor},
	ids::CallId,
};
use omp_proto::inference::v1 as pb;
use parking_lot::{Mutex, RwLock};
use smallvec::smallvec;

#[derive(Clone, Copy)]
struct Available;

impl CredentialView for Available {
	fn availability(&self, _provider: &str) -> Availability {
		Availability::Available
	}
}

#[derive(Clone)]
enum Behavior {
	Events(Vec<TurnEvent>),
	Blocked { release: Arc<Mutex<Option<flume::Receiver<()>>>>, dropped: Arc<AtomicBool> },
	Interactive(Arc<InteractiveState>),
}

#[derive(Default)]
struct InteractiveState {
	input:    Mutex<Option<InvokeInput>>,
	complete: Mutex<Option<InvokeComplete>>,
}

struct DropTracked {
	inner:   BoxStream<'static, TurnEvent>,
	dropped: Arc<AtomicBool>,
}

impl Stream for DropTracked {
	type Item = TurnEvent;

	fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		self.inner.as_mut().poll_next(context)
	}
}

impl Drop for DropTracked {
	fn drop(&mut self) {
		self.dropped.store(true, Ordering::SeqCst);
	}
}

struct MockChat {
	behavior: Behavior,
	calls:    Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl Chat for MockChat {
	async fn turn(
		&self,
		_request: ChatRequest,
		executor: Option<Arc<dyn Executor>>,
	) -> Result<futures::stream::BoxStream<'static, TurnEvent>, FacetError> {
		self.calls.fetch_add(1, Ordering::SeqCst);
		match &self.behavior {
			Behavior::Events(events) => Ok(Box::pin(stream::iter(events.clone()))),
			Behavior::Blocked { release, dropped } => {
				let release = release.lock().take().expect("one upstream turn");
				let inner = Box::pin(async_stream::stream! {
					release.recv_async().await.expect("test releases upstream");
					yield successful_outcome();
				});
				Ok(Box::pin(DropTracked { inner, dropped: Arc::clone(dropped) }))
			},
			Behavior::Interactive(state) => {
				let state = Arc::clone(state);
				let executor = executor.expect("interactive mock needs executor");
				Ok(Box::pin(async_stream::stream! {
					let call_id = call_id();
					let tool_call = ToolCall::builder()
						.id(call_id)
						.name(SmolStr::new_static("shell"))
						.args_json(Bytes::from_static(br#"{"command":"pwd"}"#))
						.thought_signature(Bytes::new())
						.build();
					let invocation = Invoke::builder()
						.invocation_id(SmolStr::new_static("invoke-1"))
						.name(SmolStr::new_static("cursor/shell"))
						.tool_call(tool_call)
						.vendor(Bytes::new())
						.timeout_ms(5_000)
						.props(Props::default())
						.build();
					yield TurnEvent::Invoke(invocation.clone());
					let (inputs, received) = flume::bounded(4);
					let completion = executor.invoke(invocation, inputs);
					tokio::pin!(completion);
					let input = tokio::select! {
						input = received.recv_async() => input.expect("forwarded input"),
						_ = &mut completion => panic!("completion arrived before input"),
					};
					*state.input.lock() = Some(input);
					yield TurnEvent::PartDelta { index: 0, chunk: Bytes::from_static(b"input-ok") };
					let complete = completion.await;
					*state.complete.lock() = Some(complete);
					yield successful_outcome();
				}))
			},
		}
	}
}

fn model_card() -> ModelCard {
	ModelCard::builder()
		.id(SmolStr::new_static("test/model"))
		.provider(SmolStr::new_static("test"))
		.model(SmolStr::new_static("model"))
		.name(SmolStr::new_static("Model"))
		.family(SmolStr::new_static("test"))
		.facets(smallvec![CatalogFacet::Chat])
		.inputs(smallvec![Modality::Text])
		.outputs(smallvec![Modality::Text])
		.reasoning(false)
		.efforts(smallvec![])
		.context_window(4_096)
		.max_output_tokens(1_024)
		.pricing(smallvec![])
		.availability(Availability::Available)
		.source(Source::Configured)
		.blocked_until_ms(0)
		.deprecated(false)
		.updated_at_ms(0)
		.props(Props::default())
		.effort_routing(BTreeMap::new())
		.build()
}

fn engine(
	contexts: Arc<ContextStore>,
	behavior: Behavior,
	requires_executor: bool,
) -> (TurnEngine, Arc<AtomicUsize>) {
	let catalog = ModelCatalog::new(vec![model_card()]);
	let registry = Arc::new(RwLock::new(Registry::new(&catalog, Arc::new(Available))));
	let resolver = Arc::new(ChatResolver::new(registry));
	let calls = Arc::new(AtomicUsize::new(0));
	resolver.register(ChatRoute {
		provider: SmolStr::new_static("test"),
		credential_id: SmolStr::new_static("cred-a"),
		requires_executor,
		chat: Arc::new(MockChat { behavior, calls: Arc::clone(&calls) }),
	});
	(TurnEngine::new(contexts, resolver), calls)
}

fn call_id() -> CallId {
	"01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().expect("call id")
}

fn item(role: Role, text: &'static str) -> Item {
	Item::builder()
		.seq(0)
		.kind(ItemKind::Message(
			Message::builder()
				.role(role)
				.parts(vec![Part::Text(SmolStr::new_static(text))])
				.build(),
		))
		.props(Props::default())
		.build()
}

fn successful_outcome() -> TurnEvent {
	TurnEvent::Outcome(
		ChatOutcome::builder()
			.output(vec![item(Role::Assistant, "answer")])
			.stop(StopReason::EndTurn)
			.unsupported(Vec::new())
			.provider(SmolStr::new_static("test"))
			.model(SmolStr::new_static("model"))
			.props(Props::default())
			.build(),
	)
}

fn failure() -> TurnEvent {
	TurnEvent::Error(
		TurnError::builder()
			.kind(TurnErrorKind::Upstream)
			.detail(SmolStr::new_static("mock failure"))
			.unsupported(Vec::new())
			.retry_after_ms(0)
			.build(),
	)
}

fn context_ref(context_id: &'static str, expected: Revision) -> ContextRef {
	ContextRef::builder()
		.context_id(SmolStr::new_static(context_id))
		.expected(expected)
		.build()
}

fn incremental_open(
	turn_id: &'static str,
	context_id: &'static str,
	expected: Revision,
	truncate_to: Option<u64>,
	append: Vec<Item>,
	executor: bool,
) -> pb::TurnFrame {
	pb::TurnFrame {
		frame: Some(pb::turn_frame::Frame::Open(pb::TurnRequest {
			turn_id:  turn_id.into(),
			input:    Some(pb::turn_request::Input::Incremental(pb::Incremental {
				context: Some(context_ref(context_id, expected).into()),
				delta:   Some(
					ThreadDelta::builder()
						.maybe_truncate_to(truncate_to)
						.append(append)
						.build()
						.into(),
				),
			})),
			params:   Some(params()),
			executor: executor.then(|| pb::Executor { tools: vec!["cursor/*".into()] }),
			props:    None,
		})),
	}
}

fn seed_open(turn_id: &'static str, context_id: &'static str, thread: Thread) -> pb::TurnFrame {
	pb::TurnFrame {
		frame: Some(pb::turn_frame::Frame::Open(pb::TurnRequest {
			turn_id:  turn_id.into(),
			input:    Some(pb::turn_request::Input::Seed(pb::Seed {
				context_id: context_id.into(),
				thread:     Some(thread.into()),
			})),
			params:   Some(params()),
			executor: None,
			props:    None,
		})),
	}
}

fn params() -> pb::ChatParams {
	pb::ChatParams { model: "test/model".into(), ..pb::ChatParams::default() }
}

async fn open_stream(engine: &TurnEngine, open: pb::TurnFrame) -> TurnStream {
	engine
		.turn_frames(stream::iter(vec![Ok(open)]))
		.await
		.expect("turn stream")
}

async fn next_event(events: &mut TurnStream) -> TurnEvent {
	events
		.next()
		.await
		.expect("event")
		.expect("transport status")
		.try_into()
		.expect("canonical event")
}

async fn commit(
	engine: &TurnEngine,
	turn_id: &'static str,
	context_id: &'static str,
	expected: Revision,
	append: Vec<Item>,
) -> ChatOutcome {
	let mut events =
		open_stream(engine, incremental_open(turn_id, context_id, expected, None, append, false))
			.await;
	assert!(matches!(next_event(&mut events).await, TurnEvent::Accepted { replay: false }));
	let TurnEvent::Outcome(outcome) = next_event(&mut events).await else {
		panic!("terminal outcome");
	};
	outcome
}

#[tokio::test]
async fn precondition_never_mutates_even_with_truncate_intent() {
	let contexts = Arc::new(ContextStore::default());
	let stale = contexts
		.seed("ctx-conflict", thread(vec![item(Role::User, "original")]))
		.expect("seed");
	let (engine, calls) =
		engine(Arc::clone(&contexts), Behavior::Events(vec![successful_outcome()]), false);
	let committed = commit(&engine, "turn-current", "ctx-conflict", stale.clone(), vec![item(
		Role::User,
		"current",
	)])
	.await;
	let current = committed.revision.expect("revision");
	let before = contexts
		.snapshot(&context_ref("ctx-conflict", current.clone()))
		.expect("before");

	let mut rejected = open_stream(
		&engine,
		incremental_open(
			"turn-stale",
			"ctx-conflict",
			stale,
			Some(0),
			vec![item(Role::User, "must not append")],
			false,
		),
	)
	.await;
	let TurnEvent::Error(error) = next_event(&mut rejected).await else {
		panic!("conflict");
	};
	assert_eq!(error.kind, TurnErrorKind::Conflict);
	assert_eq!(error.actual.as_ref(), Some(&current));
	assert_eq!(contexts.revision("ctx-conflict").expect("revision"), current);
	assert_eq!(
		contexts
			.snapshot(&context_ref("ctx-conflict", current))
			.expect("after"),
		before
	);
	assert_eq!(calls.load(Ordering::SeqCst), 1, "stale turn never reached the facet");
}

#[tokio::test]
async fn aba_same_head_different_chain_token_conflicts() {
	let contexts = Arc::new(ContextStore::default());
	let actual = contexts
		.seed("ctx-aba", thread(vec![item(Role::User, "A")]))
		.expect("seed");
	let mut forged = actual.clone();
	forged.token = Bytes::from_static(b"equal-head-divergent-chain");
	let (engine, calls) =
		engine(Arc::clone(&contexts), Behavior::Events(vec![successful_outcome()]), false);
	let mut events = open_stream(
		&engine,
		incremental_open("turn-aba", "ctx-aba", forged, None, Vec::new(), false),
	)
	.await;
	let TurnEvent::Error(error) = next_event(&mut events).await else {
		panic!("conflict");
	};
	assert_eq!(error.kind, TurnErrorKind::Conflict);
	assert_eq!(error.actual, Some(actual.clone()));
	assert_eq!(contexts.revision("ctx-aba").expect("revision"), actual);
	assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn turn_id_replays_committed_and_attaches_in_flight() {
	let contexts = Arc::new(ContextStore::default());
	let initial = contexts
		.seed("ctx-replay", Thread::default())
		.expect("seed");
	let (replay_engine, calls) =
		engine(Arc::clone(&contexts), Behavior::Events(vec![successful_outcome()]), false);
	let open = incremental_open(
		"turn-replay",
		"ctx-replay",
		initial,
		None,
		vec![item(Role::User, "once")],
		false,
	);
	let mut first = open_stream(&replay_engine, open.clone()).await;
	assert!(matches!(next_event(&mut first).await, TurnEvent::Accepted { replay: false }));
	let TurnEvent::Outcome(first_outcome) = next_event(&mut first).await else {
		panic!("outcome")
	};
	let revision = first_outcome.revision.clone().expect("revision");
	let mut replay = open_stream(&replay_engine, open).await;
	assert!(matches!(next_event(&mut replay).await, TurnEvent::Accepted { replay: true }));
	let TurnEvent::Outcome(replayed) = next_event(&mut replay).await else {
		panic!("replayed outcome")
	};
	assert_eq!(replayed, first_outcome);
	assert_eq!(contexts.revision("ctx-replay").expect("revision"), revision);
	assert_eq!(calls.load(Ordering::SeqCst), 1);

	let attached_contexts = Arc::new(ContextStore::default());
	let attached_initial = attached_contexts
		.seed("ctx-attach", Thread::default())
		.expect("seed");
	let (release_tx, release_rx) = flume::bounded(1);
	let dropped = Arc::new(AtomicBool::new(false));
	let (attached_engine, attached_calls) = engine(
		Arc::clone(&attached_contexts),
		Behavior::Blocked { release: Arc::new(Mutex::new(Some(release_rx))), dropped },
		false,
	);
	let attached_open = incremental_open(
		"turn-attach",
		"ctx-attach",
		attached_initial,
		None,
		vec![item(Role::User, "once")],
		false,
	);
	let mut owner = open_stream(&attached_engine, attached_open.clone()).await;
	assert!(matches!(next_event(&mut owner).await, TurnEvent::Accepted { replay: false }));
	let mut attachment = open_stream(&attached_engine, attached_open).await;
	assert!(matches!(next_event(&mut attachment).await, TurnEvent::Accepted { replay: false }));
	assert_eq!(attached_calls.load(Ordering::SeqCst), 1);
	release_tx.send_async(()).await.expect("release");
	let TurnEvent::Outcome(owner_outcome) = next_event(&mut owner).await else {
		panic!("owner outcome")
	};
	let TurnEvent::Outcome(attached_outcome) = next_event(&mut attachment).await else {
		panic!("attached outcome")
	};
	assert_eq!(attached_outcome, owner_outcome);
	assert_eq!(attached_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn atomic_rollback_on_midstream_error_and_client_cancellation() {
	let error_contexts = Arc::new(ContextStore::default());
	let error_revision = error_contexts
		.seed("ctx-error", Thread::default())
		.expect("seed");
	let (error_engine, _) =
		engine(Arc::clone(&error_contexts), Behavior::Events(vec![failure()]), false);
	let mut errored = open_stream(
		&error_engine,
		incremental_open(
			"turn-error",
			"ctx-error",
			error_revision.clone(),
			None,
			vec![item(Role::User, "must roll back")],
			false,
		),
	)
	.await;
	assert!(matches!(next_event(&mut errored).await, TurnEvent::Accepted { .. }));
	assert!(matches!(next_event(&mut errored).await, TurnEvent::Error(_)));
	assert_eq!(error_contexts.revision("ctx-error").expect("revision"), error_revision);
	assert_eq!(
		error_contexts
			.snapshot(&context_ref("ctx-error", error_revision))
			.expect("snapshot")
			.items,
		[] as [omp_llm_types::Item; 0]
	);

	let cancel_contexts = Arc::new(ContextStore::default());
	let cancel_revision = cancel_contexts
		.seed("ctx-cancel", Thread::default())
		.expect("seed");
	let (_release_tx, release_rx) = flume::bounded(1);
	let dropped = Arc::new(AtomicBool::new(false));
	let (cancel_engine, _) = engine(
		Arc::clone(&cancel_contexts),
		Behavior::Blocked {
			release: Arc::new(Mutex::new(Some(release_rx))),
			dropped: Arc::clone(&dropped),
		},
		false,
	);
	let mut cancelled = open_stream(
		&cancel_engine,
		incremental_open(
			"turn-cancel",
			"ctx-cancel",
			cancel_revision.clone(),
			None,
			vec![item(Role::User, "must roll back")],
			false,
		),
	)
	.await;
	assert!(matches!(next_event(&mut cancelled).await, TurnEvent::Accepted { .. }));
	drop(cancelled);
	assert!(dropped.load(Ordering::SeqCst), "dropping the client stream dropped upstream");
	assert_eq!(cancel_contexts.revision("ctx-cancel").expect("revision"), cancel_revision);
	assert_eq!(
		cancel_contexts
			.snapshot(&context_ref("ctx-cancel", cancel_revision))
			.expect("snapshot")
			.items,
		[] as [omp_llm_types::Item; 0]
	);
}

#[tokio::test]
async fn fork_isolation_survives_commits_on_both_branches() {
	let contexts = Arc::new(ContextStore::default());
	let parent_initial = contexts
		.seed("parent", thread(vec![item(Role::User, "shared"), item(Role::Assistant, "old")]))
		.expect("seed");
	let fork_initial = contexts
		.fork(&context_ref("parent", parent_initial.clone()), Some(1), "fork")
		.expect("fork");
	let (engine, _) =
		engine(Arc::clone(&contexts), Behavior::Events(vec![successful_outcome()]), false);
	let parent_outcome = commit(&engine, "turn-parent", "parent", parent_initial, vec![item(
		Role::User,
		"parent-only",
	)])
	.await;
	let fork_outcome =
		commit(&engine, "turn-fork", "fork", fork_initial, vec![item(Role::User, "fork-only")]).await;
	let parent = contexts
		.snapshot(&context_ref("parent", parent_outcome.revision.expect("parent revision")))
		.expect("parent");
	let fork = contexts
		.snapshot(&context_ref("fork", fork_outcome.revision.expect("fork revision")))
		.expect("fork");
	assert_eq!(texts(&parent), vec!["shared", "old", "parent-only", "answer"]);
	assert_eq!(texts(&fork), vec!["shared", "fork-only", "answer"]);
}

#[tokio::test]
async fn need_full_reseed_then_incremental_turn_resumes() {
	let contexts = Arc::new(ContextStore::default());
	let evicted = contexts
		.seed("ctx-reseed", thread(vec![item(Role::User, "remembered")]))
		.expect("seed");
	assert!(contexts.evict("ctx-reseed"));
	let (engine, calls) =
		engine(Arc::clone(&contexts), Behavior::Events(vec![successful_outcome()]), false);
	let mut missing = open_stream(
		&engine,
		incremental_open("turn-missing", "ctx-reseed", evicted, None, Vec::new(), false),
	)
	.await;
	let TurnEvent::Error(error) = next_event(&mut missing).await else {
		panic!("need full")
	};
	assert_eq!(error.kind, TurnErrorKind::NeedFull);

	let mut reseed = open_stream(
		&engine,
		seed_open("turn-reseed", "ctx-reseed", thread(vec![item(Role::User, "remembered")])),
	)
	.await;
	assert!(matches!(next_event(&mut reseed).await, TurnEvent::Accepted { replay: false }));
	let TurnEvent::Outcome(reseeded) = next_event(&mut reseed).await else {
		panic!("reseed outcome")
	};
	let resumed =
		commit(&engine, "turn-resumed", "ctx-reseed", reseeded.revision.expect("revision"), vec![
			item(Role::User, "resumed"),
		])
		.await;
	let snapshot = contexts
		.snapshot(&context_ref("ctx-reseed", resumed.revision.expect("revision")))
		.expect("snapshot");
	assert_eq!(texts(&snapshot), vec!["remembered", "answer", "resumed", "answer"]);
	assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn per_context_serialization_rejects_second_turn_without_queueing() {
	let contexts = Arc::new(ContextStore::default());
	let revision = contexts.seed("ctx-busy", Thread::default()).expect("seed");
	let (release_tx, release_rx) = flume::bounded(1);
	let dropped = Arc::new(AtomicBool::new(false));
	let (engine, calls) = engine(
		Arc::clone(&contexts),
		Behavior::Blocked { release: Arc::new(Mutex::new(Some(release_rx))), dropped },
		false,
	);
	let mut first = open_stream(
		&engine,
		incremental_open("turn-one", "ctx-busy", revision.clone(), None, Vec::new(), false),
	)
	.await;
	assert!(matches!(next_event(&mut first).await, TurnEvent::Accepted { .. }));
	let mut second = open_stream(
		&engine,
		incremental_open("turn-two", "ctx-busy", revision, None, Vec::new(), false),
	)
	.await;
	let TurnEvent::Error(error) = next_event(&mut second).await else {
		panic!("busy")
	};
	assert_eq!(error.kind, TurnErrorKind::Overloaded);
	assert_eq!(calls.load(Ordering::SeqCst), 1, "second turn was rejected before routing");
	release_tx.send_async(()).await.expect("release first");
	assert!(matches!(next_event(&mut first).await, TurnEvent::Outcome(_)));
}

#[tokio::test]
async fn interactive_turn_commits_tool_pair_and_assistant_atomically() {
	let contexts = Arc::new(ContextStore::default());
	let revision = contexts
		.seed("ctx-interactive", Thread::default())
		.expect("seed");
	let state = Arc::new(InteractiveState::default());
	let (engine, _) = engine(Arc::clone(&contexts), Behavior::Interactive(Arc::clone(&state)), true);
	let (frames, incoming) = flume::bounded(8);
	frames
		.send_async(Ok(incremental_open(
			"turn-interactive",
			"ctx-interactive",
			revision,
			None,
			vec![item(Role::User, "question")],
			true,
		)))
		.await
		.expect("open");
	let mut events = engine
		.turn_frames(incoming.into_stream())
		.await
		.expect("turn");
	assert!(matches!(next_event(&mut events).await, TurnEvent::Accepted { replay: false }));
	let TurnEvent::Invoke(invoke) = next_event(&mut events).await else {
		panic!("invoke")
	};
	assert!(invoke.tool_call.is_some());
	let input = InvokeInput::builder()
		.invocation_id(SmolStr::new_static("invoke-1"))
		.payload(InvokePayload::Chunk(
			InvokeChunk::builder()
				.channel(InvokeChannel::Stdout)
				.data(Bytes::from_static(b"/work/omp\n"))
				.build(),
		))
		.build();
	frames
		.send_async(Ok(pb::TurnFrame {
			frame: Some(pb::turn_frame::Frame::Input(input.clone().into())),
		}))
		.await
		.expect("input");
	assert!(matches!(next_event(&mut events).await, TurnEvent::PartDelta { .. }));
	let result = ToolResult::builder()
		.call_id(call_id())
		.name(SmolStr::new_static("shell"))
		.parts(vec![Part::Text(SmolStr::new_static("/work/omp"))])
		.is_error(false)
		.build();
	let complete = InvokeComplete::builder()
		.invocation_id(SmolStr::new_static("invoke-1"))
		.tool_result(result.clone())
		.status(
			ExecStatus::builder()
				.outcome(ExecOutcome::Exited)
				.exit_code(0)
				.signal(SmolStr::new_static(""))
				.reason(SmolStr::new_static(""))
				.cwd(SmolStr::new_static("/work/omp"))
				.aborted(false)
				.output_location(SmolStr::new_static(""))
				.local_execution_time_ms(1)
				.is_readonly(false)
				.command_timeout_ms(0)
				.build(),
		)
		.vendor(Bytes::new())
		.props(Props::default())
		.build();
	frames
		.send_async(Ok(pb::TurnFrame {
			frame: Some(pb::turn_frame::Frame::Complete(complete.clone().into())),
		}))
		.await
		.expect("complete");
	let TurnEvent::Outcome(outcome) = next_event(&mut events).await else {
		panic!("outcome")
	};
	assert_eq!(state.input.lock().as_ref(), Some(&input));
	assert_eq!(state.complete.lock().as_ref(), Some(&complete));
	let revision = outcome.revision.expect("revision");
	let snapshot = contexts
		.snapshot(&context_ref("ctx-interactive", revision.clone()))
		.expect("snapshot");
	assert_eq!(snapshot.items.len(), 4);
	assert!(
		matches!(&snapshot.items[0].kind, ItemKind::Message(message) if message.role == Role::User)
	);
	assert!(matches!(&snapshot.items[1].kind, ItemKind::ToolCall(call) if call.id == call_id()));

	assert!(matches!(&snapshot.items[2].kind, ItemKind::ToolResult(actual) if actual == &result));
	assert!(
		matches!(&snapshot.items[3].kind, ItemKind::Message(message) if message.role == Role::Assistant)
	);
	assert_eq!(revision.head, 4, "one atomic commit contains all four items");
}
fn thread(items: Vec<Item>) -> Thread {
	Thread::builder().items(items).build()
}

fn texts(thread: &Thread) -> Vec<&str> {
	thread
		.items
		.iter()
		.filter_map(|item| match &item.kind {
			ItemKind::Message(message) => message.parts.iter().find_map(|part| match part {
				Part::Text(text) => Some(text.as_str()),
				_ => None,
			}),
			_ => None,
		})
		.collect()
}
