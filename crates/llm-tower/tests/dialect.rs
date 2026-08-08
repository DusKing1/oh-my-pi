//! Production-stack proof for owned model-prompt dialect request and stream
//! projection.

use std::{
	collections::VecDeque,
	convert::Infallible,
	future::{Future, Ready},
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	task::{Context, Poll},
};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use omp_core::SmolStr;
use omp_llm_catalog::{
	compat::{Compat, ThinkingToolChoiceConflict},
	identity::{Dialect, DialectSelection},
};
use omp_llm_error::{Classification, Feature, policy::BlockTable};
use omp_llm_tower::{
	dialect::{OwnedDialectChat, OwnedDialectConfig, OwnedDialectLayer},
	envelope::ProviderRequest,
	learn::RequestRepair,
	preflight::{Admission, UsageOracle},
	refresh::{CredentialRefresher, RefreshFailure},
	select::{CredentialCandidates, CredentialLease, CredentialPool, LeaseSource, Routed},
	stack::builder::{RouteDependencies, RouteStackBuilder, RouteStackConfig},
	tap::FrameSink,
};
use omp_llm_types::{
	Chat, ChatRequest, Error as ChatError, Executor, Item, ItemKind, Message, Part, Props,
	ResolvedModelCapabilities, ResolvedModelPolicy, Role, ToolResult,
};
use omp_proto::inference::v1::{
	ChatParams, Effort, Fallback, Outcome, PartDelta, PartEnd, PartStart, Reasoning, Seed,
	StopReason, ToolChoice, ToolDef, TurnEvent, TurnRequest, part_start, tool_choice, turn_event,
	turn_request,
};
use parking_lot::Mutex;
use tower::{Layer, Service, ServiceExt};

#[derive(Clone)]
struct RoutedScript {
	streams: Arc<Mutex<VecDeque<Vec<TurnEvent>>>>,
	calls:   Arc<Mutex<Vec<TurnRequest>>>,
}

impl RoutedScript {
	fn new(streams: impl IntoIterator<Item = Vec<TurnEvent>>) -> Self {
		Self {
			streams: Arc::new(Mutex::new(streams.into_iter().collect())),
			calls:   Arc::new(Mutex::new(Vec::new())),
		}
	}
}

impl Service<Routed> for RoutedScript {
	type Error = Infallible;
	type Future = Ready<Result<Self::Response, Self::Error>>;
	type Response = futures::stream::Iter<std::vec::IntoIter<TurnEvent>>;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, routed: Routed) -> Self::Future {
		self.calls.lock().push(routed.request);
		let events = self.streams.lock().pop_front().unwrap_or_default();
		std::future::ready(Ok(futures::stream::iter(events)))
	}
}

struct Allow;
impl UsageOracle for Allow {
	fn admit(&self, _model: &str) -> Admission {
		Admission::Allow
	}
}

struct Pool;
impl CredentialPool for Pool {
	fn candidates(&self, _model: &str) -> CredentialCandidates {
		std::iter::once(7).collect()
	}
}

struct Leases;
impl LeaseSource for Leases {
	fn lease(&self, id: u64) -> Option<CredentialLease> {
		Some(CredentialLease::new("local", id, 1))
	}
}

struct Fresh;
impl CredentialRefresher for Fresh {
	fn expires_at_ms(&self) -> Option<u64> {
		None
	}

	fn refresh(
		&self,
		_force: bool,
	) -> Pin<Box<dyn Future<Output = Result<(), RefreshFailure>> + Send + '_>> {
		Box::pin(std::future::ready(Ok(())))
	}
}

struct NoRepair;
impl RequestRepair for NoRepair {
	fn strip(
		&self,
		_request: &TurnRequest,
		_feature: Feature,
		_classification: &Classification,
	) -> Option<TurnRequest> {
		None
	}
}

struct NoopSink;
impl FrameSink for NoopSink {
	fn on_request(&self, _request: &TurnRequest) {}

	fn on_frame(&self, _frame: &TurnEvent) {}

	fn on_end(&self) {}
}

fn dependencies() -> RouteDependencies {
	RouteDependencies {
		usage:          Arc::new(Allow),
		credentials:    Arc::new(Pool),
		leases:         Arc::new(Leases),
		refresher:      Arc::new(Fresh),
		repair:         Arc::new(NoRepair),
		observer:       Arc::new(NoopSink),
		usage_observer: Arc::new(omp_llm_tower::stack::meter::NoopUsageObserver),
		blocks:         Arc::new(Mutex::new(BlockTable::new())),
	}
}

fn request(thread: omp_llm_types::Thread, choice: tool_choice::Mode) -> TurnRequest {
	TurnRequest {
		params: Some(ChatParams {
			model: "qwen/qwen3-8b".to_owned(),
			tools: vec![ToolDef {
				name: "echo".to_owned(),
				description: "Echo a message".to_owned(),
				schema_json: Bytes::from_static(
					br#"{"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"]}"#,
				),
				..ToolDef::default()
			}],
			tool_choice: Some(ToolChoice {
				mode: choice as i32,
				on_unsupported: Fallback::Emulate as i32,
				..ToolChoice::default()
			}),
			thinking: Some(Reasoning {
				effort: Effort::High as i32,
				on_unsupported: Fallback::Emulate as i32,
				budget_tokens: Some(2_048),
				..Reasoning::default()
			}),
			..ChatParams::default()
		}),
		input: Some(turn_request::Input::Seed(Seed {
			thread: Some(thread.into()),
			..Seed::default()
		})),
		..TurnRequest::default()
	}
}

fn user(text: &str) -> Item {
	Item::builder()
		.seq(0)
		.kind(ItemKind::Message(
			Message::builder()
				.role(Role::User)
				.parts(vec![Part::Text(SmolStr::new(text))])
				.build(),
		))
		.props(Props::default())
		.build()
}

fn text_stream(chunks: &[&'static [u8]]) -> Vec<TurnEvent> {
	let mut events = vec![TurnEvent {
		event: Some(turn_event::Event::PartStart(PartStart {
			index: 4,
			kind: part_start::Kind::Text as i32,
			..PartStart::default()
		})),
	}];
	events.extend(chunks.iter().map(|chunk| TurnEvent {
		event: Some(turn_event::Event::PartDelta(PartDelta {
			index: 4,
			chunk: Bytes::from_static(chunk),
		})),
	}));
	events.push(TurnEvent {
		event: Some(turn_event::Event::PartEnd(PartEnd { index: 4, ..PartEnd::default() })),
	});
	events.push(TurnEvent {
		event: Some(turn_event::Event::Outcome(Outcome {
			stop: StopReason::StopEndTurn as i32,
			..Outcome::default()
		})),
	});
	events
}

fn native_events(events: Vec<TurnEvent>) -> Vec<omp_llm_types::TurnEvent> {
	events
		.into_iter()
		.map(|event| omp_llm_types::TurnEvent::try_from(event).unwrap())
		.collect()
}

#[derive(Clone)]
struct LocalChatScript {
	streams: Arc<Mutex<VecDeque<Vec<omp_llm_types::TurnEvent>>>>,
	calls:   Arc<Mutex<Vec<ChatRequest>>>,
}

impl LocalChatScript {
	fn new(streams: impl IntoIterator<Item = Vec<omp_llm_types::TurnEvent>>) -> Self {
		Self {
			streams: Arc::new(Mutex::new(streams.into_iter().collect())),
			calls:   Arc::new(Mutex::new(Vec::new())),
		}
	}
}

#[async_trait::async_trait]
impl Chat for LocalChatScript {
	async fn turn(
		&self,
		request: ChatRequest,
		_executor: Option<Arc<dyn Executor>>,
	) -> Result<futures::stream::BoxStream<'static, omp_llm_types::TurnEvent>, ChatError> {
		self.calls.lock().push(request);
		Ok(futures::stream::iter(self.streams.lock().pop_front().unwrap_or_default()).boxed())
	}
}

#[tokio::test]
async fn production_stack_projects_owned_tools_and_renders_result_history() {
	let script = RoutedScript::new([
		text_stream(&[
			b"<tool_call>\n{\"name\":\"echo\",\"arguments\":{\"msg\":\"hel",
			b"lo\"}}\n</tool_call>",
		]),
		text_stream(&[b"clean follow-up"]),
	]);
	let mut stack = RouteStackBuilder::new(dependencies(), RouteStackConfig {
		dialect: Some(DialectSelection::Explicit(Dialect::Qwen3)),
		..RouteStackConfig::default()
	})
	.build(script.clone());

	let first = stack
		.ready()
		.await
		.unwrap()
		.call(ProviderRequest::new(
			request(
				omp_llm_types::Thread::builder()
					.items(vec![user("say hello")])
					.build(),
				tool_choice::Mode::Auto,
			),
			None,
		))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	let first = native_events(first);
	let args = first
		.iter()
		.filter_map(|event| match event {
			omp_llm_types::TurnEvent::PartDelta { chunk, .. } => Some(chunk.clone()),
			_ => None,
		})
		.fold(Vec::new(), |mut all, chunk| {
			all.extend_from_slice(&chunk);
			all
		});
	assert_eq!(args, br#"{"msg":"hello"}"#);
	let outcome = first
		.into_iter()
		.find_map(|event| match event {
			omp_llm_types::TurnEvent::Outcome(outcome) => Some(outcome),
			_ => None,
		})
		.expect("owned projection must remain successful");
	assert_eq!(outcome.stop, omp_llm_types::StopReason::ToolUse);
	let call = outcome
		.output
		.iter()
		.find_map(|item| match &item.kind {
			ItemKind::ToolCall(call) => Some(call.clone()),
			_ => None,
		})
		.expect("split in-band arguments must produce one canonical tool call");

	let mut history = outcome.output;
	history.push(
		Item::builder()
			.seq(0)
			.kind(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(call.id)
					.name(call.name)
					.parts(vec![Part::Text(SmolStr::new_static("hello"))])
					.is_error(false)
					.build(),
			))
			.props(Props::default())
			.build(),
	);
	history.push(user("continue"));
	let follow_up = stack
		.ready()
		.await
		.unwrap()
		.call(ProviderRequest::new(
			request(omp_llm_types::Thread::builder().items(history).build(), tool_choice::Mode::Auto),
			None,
		))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert!(native_events(follow_up).iter().any(|event| {
		matches!(event, omp_llm_types::TurnEvent::PartDelta { chunk, .. } if chunk.as_ref() == b"clean follow-up")
	}));

	let calls = script.calls.lock();
	assert_eq!(calls.len(), 2);
	for call in calls.iter() {
		let params = call.params.as_ref().unwrap();
		assert!(params.tools.is_empty(), "owned requests must remove native schemas");
		assert!(params.tool_choice.is_none(), "owned requests must remove native tool choice");
		assert_eq!(params.thinking.as_ref().unwrap().effort(), Effort::Off);
		assert!(params.thinking.as_ref().unwrap().budget_tokens.is_none());
	}
	let second_thread = match calls[1].input.as_ref().unwrap() {
		turn_request::Input::Seed(seed) => seed.thread.clone().unwrap(),
		turn_request::Input::Incremental(_) => panic!("owned request was not a full seed"),
	};
	let second: omp_llm_types::Thread = second_thread.try_into().unwrap();
	let wire_text = second
		.items
		.iter()
		.filter_map(|item| match &item.kind {
			ItemKind::Message(message) => Some(message),
			_ => None,
		})
		.flat_map(|message| &message.parts)
		.filter_map(|part| match part {
			Part::Text(text) => Some(text.as_str()),
			_ => None,
		})
		.collect::<Vec<_>>()
		.join("\n");
	assert!(wire_text.contains("# Tools"));
	assert!(wire_text.contains("<tool_call>"));
	assert!(wire_text.contains("<tool_response>\nhello\n</tool_response>"));
}

#[tokio::test]
async fn local_chat_auto_keeps_native_tools_and_completes_owned_tool_result_turn() {
	let owned_script = LocalChatScript::new([
		native_events(text_stream(&[
			b"<tool_call>\n{\"name\":\"echo\",\"arguments\":{\"msg\":\"local\"}}\n</tool_call>",
		])),
		native_events(text_stream(&[b"local follow-up"])),
	]);
	let owned = OwnedDialectChat::latest_user(
		Arc::new(owned_script.clone()),
		OwnedDialectConfig::new(DialectSelection::Auto, Compat::default()),
	);
	let owned_policy = Arc::new(ResolvedModelPolicy {
		capabilities: ResolvedModelCapabilities {
			tools: Some(false),
			..ResolvedModelCapabilities::default()
		},
		..ResolvedModelPolicy::default()
	});
	let mut first = ChatRequest::try_from(request(
		omp_llm_types::Thread::builder()
			.items(vec![user("use the local tool")])
			.build(),
		tool_choice::Mode::Auto,
	))
	.unwrap();
	first.model_policy = Some(Arc::clone(&owned_policy));
	let first_events = owned
		.turn(first, None)
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	let outcome = first_events
		.into_iter()
		.find_map(|event| match event {
			omp_llm_types::TurnEvent::Outcome(outcome) => Some(outcome),
			_ => None,
		})
		.expect("owned local outcome");
	assert_eq!(outcome.stop, omp_llm_types::StopReason::ToolUse);
	let call = outcome
		.output
		.iter()
		.find_map(|item| match &item.kind {
			ItemKind::ToolCall(call) => Some(call.clone()),
			_ => None,
		})
		.expect("owned local tool call");
	let mut history = vec![user("use the local tool")];
	history.extend(outcome.output);
	history.push(
		Item::builder()
			.seq(0)
			.kind(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(call.id)
					.name(call.name)
					.parts(vec![Part::Text(SmolStr::new_static("local"))])
					.is_error(false)
					.build(),
			))
			.props(Props::default())
			.build(),
	);
	history.push(user("continue"));
	let mut second = ChatRequest::try_from(request(
		omp_llm_types::Thread::builder().items(history).build(),
		tool_choice::Mode::Auto,
	))
	.unwrap();
	second.model_policy = Some(Arc::clone(&owned_policy));
	let second_events = owned
		.turn(second, None)
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert!(second_events.iter().any(
		|event| matches!(event, omp_llm_types::TurnEvent::PartDelta { chunk, .. } if chunk.as_ref() == b"local follow-up")
	));
	let calls = owned_script.calls.lock();
	assert_eq!(calls.len(), 2);
	assert!(calls.iter().all(|request| request.tools.is_empty()));
	assert!(calls.iter().all(|request| {
		Arc::ptr_eq(request.model_policy.as_ref().expect("trusted policy"), &owned_policy)
	}));
	let rendered = calls[1]
		.thread
		.items
		.iter()
		.filter_map(|item| match &item.kind {
			ItemKind::Message(message) => Some(message),
			_ => None,
		})
		.flat_map(|message| &message.parts)
		.filter_map(|part| match part {
			Part::Text(text) => Some(text.as_str()),
			_ => None,
		})
		.collect::<Vec<_>>()
		.join("\n");
	assert!(rendered.contains("# Tools"));
	assert!(rendered.contains("<tool_response>\nlocal\n</tool_response>"));
	drop(calls);

	let native_script = LocalChatScript::new([native_events(text_stream(&[b"native"]))]);
	let native = OwnedDialectChat::new(
		Arc::new(native_script.clone()),
		OwnedDialectConfig::new(DialectSelection::Auto, Compat::default()),
	);
	let mut native_request = ChatRequest::try_from(request(
		omp_llm_types::Thread::builder()
			.items(vec![user("native")])
			.build(),
		tool_choice::Mode::Auto,
	))
	.unwrap();
	native_request.model_policy = Some(Arc::new(ResolvedModelPolicy {
		capabilities: ResolvedModelCapabilities {
			tools: Some(true),
			..ResolvedModelCapabilities::default()
		},
		..ResolvedModelPolicy::default()
	}));
	native
		.turn(native_request, None)
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(native_script.calls.lock()[0].tools.len(), 1);

	let forced_script = LocalChatScript::new([native_events(text_stream(&[b"forced"]))]);
	let forced = OwnedDialectChat::new(
		Arc::new(forced_script.clone()),
		OwnedDialectConfig::new(DialectSelection::Auto, Compat::default())
			.with_override(Some(SmolStr::new_static("qwen3"))),
	);
	let mut forced_request = ChatRequest::try_from(request(
		omp_llm_types::Thread::builder()
			.items(vec![user("forced")])
			.build(),
		tool_choice::Mode::Auto,
	))
	.unwrap();
	forced_request.model_policy = Some(Arc::new(ResolvedModelPolicy {
		capabilities: ResolvedModelCapabilities {
			tools: Some(true),
			..ResolvedModelCapabilities::default()
		},
		..ResolvedModelPolicy::default()
	}));
	forced
		.turn(forced_request, None)
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert!(forced_script.calls.lock()[0].tools.is_empty());
}

#[tokio::test]
async fn native_channel_wins_and_reasoning_conflicts_are_removed() {
	let mut compat = Compat::default();
	compat.thinking_tool_choice_conflict = ThinkingToolChoiceConflict::DropThinkingWhenAny;
	let script = RoutedScript::new([vec![
		TurnEvent {
			event: Some(turn_event::Event::PartStart(PartStart {
				index:        1,
				kind:         part_start::Kind::ToolCall as i32,
				tool_call_id: omp_llm_types::ids::CallId::new().to_string(),
				tool_name:    "echo".to_owned(),
			})),
		},
		TurnEvent {
			event: Some(turn_event::Event::PartDelta(PartDelta {
				index: 1,
				chunk: Bytes::from_static(br#"{"msg":"native"}"#),
			})),
		},
		TurnEvent {
			event: Some(turn_event::Event::PartEnd(PartEnd { index: 1, ..PartEnd::default() })),
		},
		TurnEvent {
			event: Some(turn_event::Event::PartStart(PartStart {
				index: 2,
				kind: part_start::Kind::Text as i32,
				..PartStart::default()
			})),
		},
		TurnEvent {
			event: Some(turn_event::Event::PartDelta(PartDelta {
				index: 2,
				chunk: Bytes::from_static(
					b"<tool_call>\n{\"name\":\"echo\",\"arguments\":{\"msg\":\"owned\"}}\n</tool_call>",
				),
			})),
		},
		TurnEvent {
			event: Some(turn_event::Event::PartEnd(PartEnd { index: 2, ..PartEnd::default() })),
		},
		TurnEvent { event: Some(turn_event::Event::Outcome(Outcome::default())) },
	]]);
	let mut stack = RouteStackBuilder::new(dependencies(), RouteStackConfig {
		compat,
		dialect: Some(DialectSelection::Explicit(Dialect::Qwen3)),
		..RouteStackConfig::default()
	})
	.build(script.clone());
	let events = stack
		.ready()
		.await
		.unwrap()
		.call(ProviderRequest::new(
			request(omp_llm_types::Thread::default(), tool_choice::Mode::Auto),
			None,
		))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	let names: Vec<_> = native_events(events)
		.into_iter()
		.filter_map(|event| match event {
			omp_llm_types::TurnEvent::PartStart {
				kind: omp_llm_types::StreamPartKind::ToolCall,
				tool_name,
				..
			} => Some(tool_name),
			_ => None,
		})
		.collect();
	assert_eq!(names.as_slice(), ["echo"]);
	assert!(
		script.calls.lock()[0]
			.params
			.as_ref()
			.unwrap()
			.thinking
			.is_none()
	);
}

#[derive(Clone)]
struct CancelAttempt {
	dropped:     Arc<AtomicBool>,
	tail_polled: Arc<AtomicBool>,
}

struct CancelStream {
	events:      VecDeque<TurnEvent>,
	dropped:     Arc<AtomicBool>,
	tail_polled: Arc<AtomicBool>,
}

impl Drop for CancelStream {
	fn drop(&mut self) {
		self.dropped.store(true, Ordering::Release);
	}
}

impl Stream for CancelStream {
	type Item = TurnEvent;

	fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		let next = self.events.pop_front();
		if self.events.is_empty() && next.is_some() {
			self.tail_polled.store(true, Ordering::Release);
		}
		Poll::Ready(next)
	}
}

impl Service<TurnRequest> for CancelAttempt {
	type Error = Infallible;
	type Future = Ready<Result<Self::Response, Self::Error>>;
	type Response = CancelStream;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, _request: TurnRequest) -> Self::Future {
		let mut events = text_stream(&[
			b"<tool_call>\n{\"name\":\"echo\",\"arguments\":{\"msg\":\"x\"}}\n</tool_call>",
			b"<tool_response>fabricated</tool_response>",
		]);
		events.push(TurnEvent {
			event: Some(turn_event::Event::PartDelta(PartDelta {
				index: 4,
				chunk: Bytes::from_static(b"must not be polled"),
			})),
		});
		std::future::ready(Ok(CancelStream {
			events:      events.into(),
			dropped:     Arc::clone(&self.dropped),
			tail_polled: Arc::clone(&self.tail_polled),
		}))
	}
}

#[tokio::test]
async fn fabricated_result_drops_upstream_and_native_selection_is_passthrough() {
	let dropped = Arc::new(AtomicBool::new(false));
	let tail_polled = Arc::new(AtomicBool::new(false));
	let attempt =
		CancelAttempt { dropped: Arc::clone(&dropped), tail_polled: Arc::clone(&tail_polled) };
	let mut owned = OwnedDialectLayer::new(OwnedDialectConfig::new(
		DialectSelection::Explicit(Dialect::Qwen3),
		Compat::default(),
	))
	.layer(attempt);
	let events = owned
		.ready()
		.await
		.unwrap()
		.call(request(omp_llm_types::Thread::default(), tool_choice::Mode::Auto))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert!(dropped.load(Ordering::Acquire));
	assert!(!tail_polled.load(Ordering::Acquire));
	assert!(native_events(events).iter().any(|event| {
		matches!(event, omp_llm_types::TurnEvent::Outcome(outcome) if outcome.stop == omp_llm_types::StopReason::ToolUse)
	}));

	let native_script = omp_llm_tower::testing::Script::new([text_stream(&[b"native"])]);
	let mut native = OwnedDialectLayer::new(
		OwnedDialectConfig::new(DialectSelection::Native, Compat::default())
			.with_override(Some(SmolStr::new_static("native"))),
	)
	.layer(native_script.clone());
	let original = request(omp_llm_types::Thread::default(), tool_choice::Mode::Auto);
	let _ = native
		.ready()
		.await
		.unwrap()
		.call(original.clone())
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(native_script.calls.lock()[0], original);

	let unknown_script = omp_llm_tower::testing::Script::new([text_stream(&[b"plain"])]);
	let mut unknown = OwnedDialectLayer::new(OwnedDialectConfig::new(
		DialectSelection::Explicit(Dialect::Qwen3),
		Compat::default(),
	))
	.layer(unknown_script.clone());
	let mut unknown_request = request(omp_llm_types::Thread::default(), tool_choice::Mode::Auto);
	unknown_request.params.as_mut().unwrap().model = "unmapped/local-model".to_owned();
	let _ = unknown
		.ready()
		.await
		.unwrap()
		.call(unknown_request)
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	let sent = &unknown_script.calls.lock()[0];
	assert!(sent.params.as_ref().unwrap().tools.is_empty());
	let thread = match sent.input.as_ref().unwrap() {
		turn_request::Input::Seed(seed) => seed.thread.clone().unwrap(),
		turn_request::Input::Incremental(_) => panic!("unexpected incremental input"),
	};
	let thread: omp_llm_types::Thread = thread.try_into().unwrap();
	assert!(thread.items.iter().any(|item| {
		matches!(&item.kind, ItemKind::Message(message) if message.parts.iter().any(|part| matches!(part, Part::Text(text) if text.contains("# Tools"))))
	}));
}
