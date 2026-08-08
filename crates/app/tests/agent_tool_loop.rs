//! Production-boundary tests for the application-owned native tool loop.

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
use futures::{Stream, stream::BoxStream};
use omp_app::agent::{OwnedToolHandler, OwnedToolLoopError, OwnedToolOutput, run_owned_tool_loop};
use omp_core::SmolStr;
use omp_llm_catalog::{
	compat::Compat,
	identity::{Dialect, DialectSelection},
};
use omp_llm_tower::{
	dialect::{OwnedDialectConfig, OwnedDialectLayer},
	envelope::ProviderRequest,
	provider::ServiceChat,
};
use omp_llm_types::{
	BlobPart, Chat, ChatRequest, Executor, ImageDetail, Item, ItemKind, Message, MessageAttribution,
	Part, Props, Role, StopReason, Thinking, ToolCall, ToolDef, ToolResult, TurnErrorKind,
	TurnEvent, ids::CallId,
};
use omp_proto::inference::v1::{
	Outcome, PartDelta, PartEnd, PartStart, Seed, StopReason as PbStopReason,
	TurnError as PbTurnError, TurnEvent as PbTurnEvent, part_start, turn_error, turn_event,
	turn_request,
};
use parking_lot::Mutex;
use serde_json::json;
use tokio::sync::oneshot;
use tower::{Layer, Service};

/// Scripted provider text sits below the real owned-dialect layer; canonical
/// calls and outcomes are produced only by production middleware.
type ProviderStream = Pin<Box<dyn Stream<Item = PbTurnEvent> + Send>>;

#[derive(Clone)]
struct DialectFixtureAttempt {
	streams:  Arc<Mutex<VecDeque<Vec<PbTurnEvent>>>>,
	requests: Arc<Mutex<Vec<omp_proto::inference::v1::TurnRequest>>>,
}

impl DialectFixtureAttempt {
	fn new(streams: impl IntoIterator<Item = Vec<PbTurnEvent>>) -> Self {
		Self {
			streams:  Arc::new(Mutex::new(streams.into_iter().collect())),
			requests: Arc::new(Mutex::new(Vec::new())),
		}
	}
}

impl Service<ProviderRequest> for DialectFixtureAttempt {
	type Error = Infallible;
	type Future = Ready<Result<Self::Response, Self::Error>>;
	type Response = ProviderStream;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, request: ProviderRequest) -> Self::Future {
		self.requests.lock().push(request.request);
		let events = self
			.streams
			.lock()
			.pop_front()
			.expect("unexpected provider turn");
		let stream: ProviderStream = Box::pin(futures::stream::iter(events));
		std::future::ready(Ok(stream))
	}
}

struct RecordingProductionChat {
	inner:    Arc<dyn Chat>,
	requests: Arc<Mutex<Vec<ChatRequest>>>,
}

#[async_trait::async_trait]
impl Chat for RecordingProductionChat {
	async fn turn(
		&self,
		request: ChatRequest,
		executor: Option<Arc<dyn Executor>>,
	) -> Result<BoxStream<'static, TurnEvent>, omp_llm_types::Error> {
		self.requests.lock().push(request.clone());
		self.inner.turn(request, executor).await
	}
}

fn production_chat(
	attempt: DialectFixtureAttempt,
) -> (RecordingProductionChat, Arc<Mutex<Vec<ChatRequest>>>) {
	let layer = OwnedDialectLayer::new(OwnedDialectConfig::new(
		DialectSelection::Explicit(Dialect::Qwen3),
		Compat::default(),
	));
	let chat: Arc<dyn Chat> = Arc::new(ServiceChat::new(layer.layer(attempt)));
	let requests = Arc::new(Mutex::new(Vec::new()));
	(RecordingProductionChat { inner: chat, requests: requests.clone() }, requests)
}

fn provider_text(chunks: &[&'static [u8]]) -> Vec<PbTurnEvent> {
	let mut events = vec![PbTurnEvent {
		event: Some(turn_event::Event::PartStart(PartStart {
			index: 0,
			kind: part_start::Kind::Text as i32,
			..PartStart::default()
		})),
	}];
	events.extend(chunks.iter().map(|chunk| PbTurnEvent {
		event: Some(turn_event::Event::PartDelta(PartDelta {
			index: 0,
			chunk: Bytes::from_static(chunk),
		})),
	}));
	events.push(PbTurnEvent {
		event: Some(turn_event::Event::PartEnd(PartEnd { index: 0, ..PartEnd::default() })),
	});
	events.push(PbTurnEvent {
		event: Some(turn_event::Event::Outcome(Outcome {
			stop: PbStopReason::StopEndTurn as i32,
			..Outcome::default()
		})),
	});
	events
}

fn provider_error(kind: turn_error::Kind, detail: &str) -> Vec<PbTurnEvent> {
	vec![PbTurnEvent {
		event: Some(turn_event::Event::Error(PbTurnError {
			kind: kind as i32,
			detail: detail.to_owned(),
			..PbTurnError::default()
		})),
	}]
}

fn tool_definition() -> ToolDef {
	ToolDef::builder()
		.name("echo".into())
		.description("Echo one message".into())
		.schema_json(Bytes::from_static(
			br#"{"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"]}"#,
		))
		.build()
}

struct EchoHandler {
	definition: ToolDef,
	calls:      Arc<Mutex<Vec<Bytes>>>,
	output:     OwnedToolOutput,
}

impl EchoHandler {
	fn new(output: OwnedToolOutput) -> Self {
		Self { definition: tool_definition(), calls: Arc::new(Mutex::new(Vec::new())), output }
	}
}

impl OwnedToolHandler for EchoHandler {
	type Execute<'a> = Ready<OwnedToolOutput>;

	fn definition(&self) -> &ToolDef {
		&self.definition
	}

	fn execute(&self, args_json: Bytes) -> Self::Execute<'_> {
		self.calls.lock().push(args_json);
		std::future::ready(self.output.clone())
	}
}

fn props(namespace: &str, name: &str, value: serde_json::Value) -> Props {
	let mut props = Props::default();
	props.insert_ns(namespace, name, value);
	props
}

fn item(seq: u64, created_at_ms: u64, kind: ItemKind, props: Props) -> Item {
	Item::builder()
		.seq(seq)
		.created_at_ms(created_at_ms)
		.kind(kind)
		.props(props)
		.build()
}

fn image(byte: u8, mime: &str) -> BlobPart {
	BlobPart::builder()
		.hash([byte; 32])
		.mime(SmolStr::new(mime))
		.size(3)
		.inline(Bytes::from(vec![byte; 3]))
		.detail(ImageDetail::High)
		.build()
}

fn canonical_history() -> omp_llm_types::Thread {
	let historical_id: CallId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
	let image = image(7, "image/png");
	let call_metadata = props("fixture", "call", json!({ "trace": 17 }));
	let result_metadata = props("fixture", "result", json!({ "source": "archive" }));
	omp_llm_types::Thread::builder()
		.items(vec![
			item(
				1,
				101,
				ItemKind::Message(
					Message::builder()
						.role(Role::User)
						.parts(vec![Part::Text("inspect the image".into()), Part::Blob(image.clone())])
						.build(),
				),
				props("fixture", "message", json!({ "request": 9 })),
			),
			item(
				2,
				102,
				ItemKind::Message(
					Message::builder()
						.role(Role::Assistant)
						.parts(vec![
							Part::Thinking(
								Thinking::builder()
									.text("signed reasoning".into())
									.signature(Bytes::from_static(b"reasoning-signature"))
									.redacted(false)
									.build(),
							),
							Part::Text("I will inspect it.".into()),
						])
						.build(),
				),
				props("fixture", "assistant", json!(true)),
			),
			item(
				3,
				103,
				ItemKind::ToolCall(
					ToolCall::builder()
						.id(historical_id)
						.name("lookup".into())
						.args_json(Bytes::from_static(br#"{"id":9}"#))
						.thought_signature(Bytes::from_static(b"tool-thought-signature"))
						.maybe_intent(Some("inspect".into()))
						.maybe_raw(Some(Bytes::from_static(b"<historical-call/>")))
						.maybe_custom_wire_name(Some("lookup_wire".into()))
						.maybe_provider_metadata(Some(call_metadata))
						.build(),
				),
				props("fixture", "call-item", json!(3)),
			),
			item(
				4,
				104,
				ItemKind::ToolResult(
					ToolResult::builder()
						.call_id(historical_id)
						.name("lookup".into())
						.parts(vec![Part::Text("archived result".into()), Part::Blob(image)])
						.is_error(false)
						.maybe_details(Some(json!({ "rows": 1 })))
						.maybe_attribution(Some(MessageAttribution::User))
						.maybe_pruned_at_ms(None)
						.maybe_useless(Some(false))
						.maybe_provider_metadata(Some(result_metadata))
						.build(),
				),
				props("fixture", "result-item", json!(4)),
			),
		])
		.build()
}

fn request(thread: omp_llm_types::Thread) -> ChatRequest {
	ChatRequest::builder()
		.model("qwen/qwen3-8b".into())
		.thread(thread)
		.tools(vec![
			ToolDef::builder()
				.name("stale".into())
				.description("must be replaced".into())
				.schema_json(Bytes::from_static(b"{}"))
				.build(),
		])
		.build()
}

fn wire_thread(request: &omp_proto::inference::v1::TurnRequest) -> omp_llm_types::Thread {
	let Some(turn_request::Input::Seed(Seed { thread: Some(thread), .. })) = &request.input else {
		panic!("owned dialect request must be a full canonical seed");
	};
	thread.clone().try_into().unwrap()
}

fn text_of(items: &[Item]) -> String {
	items
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
		.join("\n")
}

#[tokio::test]
async fn production_dialect_runs_one_native_tool_and_serializes_the_follow_up() {
	let attempt = DialectFixtureAttempt::new([
		provider_text(&[
			b"<tool_call>\n{\"name\":\"echo\",\"arguments\":{\"msg\":\"hel",
			b"lo\"}}\n</tool_call>",
		]),
		provider_text(&[b"clean final outcome"]),
	]);
	let wire_requests = attempt.requests.clone();
	let (chat, canonical_requests) = production_chat(attempt);
	let history = canonical_history();
	let tool_image = image(8, "image/webp");
	let handler = EchoHandler::new(OwnedToolOutput {
		parts:    vec![Part::Text("echoed:hello".into()), Part::Blob(tool_image.clone())],
		is_error: false,
		details:  Some(json!({ "executions": 1 })),
	});

	let outcome = run_owned_tool_loop(&chat, request(history.clone()), &handler)
		.await
		.expect("production owned tool loop");

	assert_eq!(handler.calls.lock().as_slice(), &[Bytes::from_static(br#"{"msg":"hello"}"#)]);
	assert_eq!(outcome.tool_turn.stop, StopReason::ToolUse);
	assert_eq!(outcome.tool_result.name, "echo");
	assert_eq!(outcome.tool_result.parts, handler.output.parts);
	assert_eq!(outcome.tool_result.details, handler.output.details);
	assert_eq!(outcome.tool_result.attribution, Some(MessageAttribution::Agent));
	assert_eq!(outcome.follow_up_turn.stop, StopReason::EndTurn);
	assert_eq!(text_of(&outcome.follow_up_turn.output), "clean final outcome");

	let canonical = canonical_requests.lock();
	assert_eq!(canonical.len(), 2);
	assert_eq!(canonical[0].tools, vec![tool_definition()]);
	assert_eq!(canonical[1].tools, vec![tool_definition()]);
	assert_eq!(canonical[0].thread, history);
	assert_eq!(
		&canonical[1].thread.items[..history.items.len()],
		history.items.as_slice(),
		"images, item/provider metadata, reasoning and tool signatures, and historical attribution \
		 must survive serialization",
	);
	let appended = &canonical[1].thread.items[history.items.len()..];
	assert_eq!(appended.len(), outcome.tool_turn.output.len() + 1);
	assert!(matches!(
		appended.last().map(|item| &item.kind),
		Some(ItemKind::ToolResult(result)) if result == &outcome.tool_result
	));

	let wire = wire_requests.lock();
	assert_eq!(wire.len(), 2);
	for request in wire.iter() {
		let params = request.params.as_ref().unwrap();
		assert!(params.tools.is_empty(), "owned dialect must not leak native tool schemas");
		assert!(params.tool_choice.is_none());
	}
	let first_wire = wire_thread(&wire[0]);
	assert!(text_of(&first_wire.items).contains("# Tools"));
	let second_wire = wire_thread(&wire[1]);
	let second_text = text_of(&second_wire.items);
	assert!(second_text.contains("<tool_call>"));
	assert!(second_text.contains("{\"msg\":\"hello\"}"));
	assert!(second_text.contains("<tool_response>\nechoed:hello\n</tool_response>"));
	let wire_images = second_wire
		.items
		.iter()
		.filter_map(|item| match &item.kind {
			ItemKind::Message(message) => Some(message),
			_ => None,
		})
		.flat_map(|message| &message.parts)
		.filter_map(|part| match part {
			Part::Blob(blob) => Some(blob),
			_ => None,
		})
		.collect::<Vec<_>>();
	assert!(wire_images.iter().any(|blob| blob.hash == [7; 32]));
	assert!(wire_images.iter().any(|blob| *blob == &tool_image));
}

#[tokio::test]
async fn tool_failure_is_an_authoritative_result_not_a_loop_failure() {
	let attempt = DialectFixtureAttempt::new([
		provider_text(&[
			b"<tool_call>\n{\"name\":\"echo\",\"arguments\":{\"msg\":\"bad\"}}\n</tool_call>",
		]),
		provider_text(&[b"failure acknowledged"]),
	]);
	let wire_requests = attempt.requests.clone();
	let (chat, _) = production_chat(attempt);
	let handler = EchoHandler::new(OwnedToolOutput {
		parts:    vec![Part::Text("echo failed".into())],
		is_error: true,
		details:  Some(json!({ "code": "ECHO_FAILED" })),
	});

	let outcome = run_owned_tool_loop(&chat, request(omp_llm_types::Thread::default()), &handler)
		.await
		.expect("a failed tool result still permits model recovery");

	assert!(outcome.tool_result.is_error);
	assert_eq!(outcome.tool_result.details, handler.output.details);
	assert_eq!(text_of(&outcome.follow_up_turn.output), "failure acknowledged");
	let wire = wire_requests.lock();
	assert_eq!(wire.len(), 2);
	assert!(text_of(&wire_thread(&wire[1]).items).contains("echo failed"));
}

#[tokio::test]
async fn terminal_turn_error_stops_before_tool_execution() {
	let attempt = DialectFixtureAttempt::new([provider_error(
		turn_error::Kind::Upstream,
		"fixture upstream failure",
	)]);
	let wire_requests = attempt.requests.clone();
	let (chat, canonical_requests) = production_chat(attempt);
	let handler = EchoHandler::new(OwnedToolOutput::text("must not execute"));

	let error = run_owned_tool_loop(&chat, request(omp_llm_types::Thread::default()), &handler)
		.await
		.expect_err("terminal error must escape the loop");

	assert!(matches!(
		error,
		OwnedToolLoopError::Turn(turn)
			if turn.kind == TurnErrorKind::Upstream && turn.detail == "fixture upstream failure"
	));
	assert!(handler.calls.lock().is_empty());
	assert_eq!(canonical_requests.lock().len(), 1);
	assert_eq!(wire_requests.lock().len(), 1);
}

struct PendingHandler {
	definition: ToolDef,
	started:    Mutex<Option<oneshot::Sender<()>>>,
	dropped:    Arc<AtomicBool>,
}

struct PendingExecution {
	dropped: Arc<AtomicBool>,
}

impl Future for PendingExecution {
	type Output = OwnedToolOutput;

	fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
		Poll::Pending
	}
}

impl Drop for PendingExecution {
	fn drop(&mut self) {
		self.dropped.store(true, Ordering::SeqCst);
	}
}

impl OwnedToolHandler for PendingHandler {
	type Execute<'a> = PendingExecution;

	fn definition(&self) -> &ToolDef {
		&self.definition
	}

	fn execute(&self, _args_json: Bytes) -> Self::Execute<'_> {
		if let Some(started) = self.started.lock().take() {
			let _ = started.send(());
		}
		PendingExecution { dropped: self.dropped.clone() }
	}
}

#[tokio::test]
async fn cancelling_the_loop_drops_the_live_handler_and_never_starts_follow_up() {
	let attempt = DialectFixtureAttempt::new([provider_text(&[
		b"<tool_call>\n{\"name\":\"echo\",\"arguments\":{\"msg\":\"wait\"}}\n</tool_call>",
	])]);
	let wire_requests = attempt.requests.clone();
	let (chat, canonical_requests) = production_chat(attempt);
	let (started_tx, started_rx) = oneshot::channel();
	let dropped = Arc::new(AtomicBool::new(false));
	let handler = PendingHandler {
		definition: tool_definition(),
		started:    Mutex::new(Some(started_tx)),
		dropped:    dropped.clone(),
	};
	let task = tokio::spawn(async move {
		run_owned_tool_loop(&chat, request(omp_llm_types::Thread::default()), &handler).await
	});

	started_rx.await.expect("handler execution started");
	task.abort();
	assert!(task.await.expect_err("cancelled task").is_cancelled());
	assert!(dropped.load(Ordering::SeqCst));
	assert_eq!(canonical_requests.lock().len(), 1);
	assert_eq!(wire_requests.lock().len(), 1);
}
