#![feature(impl_trait_in_assoc_type)]

use std::{
	future::{Future, Ready, ready},
	path::Path,
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicBool, AtomicUsize, Ordering},
	},
	task::{Context, Poll},
	time::SystemTime,
};

use bytes::Bytes;
use futures::stream;
use omp_app::agent::{OwnedToolHandler, OwnedToolLoopError, OwnedToolOutput, run_owned_tool_loop};
use omp_core::Str;
use omp_llm_catalog::{ModelKey, OperationKind, ProviderId, RouteId};
use omp_llm_inference::{
	Answer, AnswerBody, Call, CallMeta, ChatRequest, Client, ContentPart, Error, ErrorKind,
	ErrorPhase, ExecutionBudget, ExecutionReceipt, Message, NegotiationPolicy, OpaqueJson,
	OperationCall, RequestId, ResponseMeta, RetryAction, Role, Sampling, Setting, Target,
	ToolCallId, ToolDefinition,
	auth::{CredentialStore, HeadlessKeySource, KeyId},
	event::{ChatEvent, Completion, FinishReason, ToolCall},
	receipt::Usage,
	router::Router,
};
use parking_lot::Mutex;
use serde_json::json;
use tokio::sync::oneshot;
use tower::Service;

fn credential_store(path: &Path) -> Arc<CredentialStore> {
	omp_app::daemon::open_credential_store_with_key_source(
		path,
		Arc::new(HeadlessKeySource::new(KeyId::new("agent-tool-loop"), [0x33; 32])),
	)
	.expect("credential store")
}

#[derive(Clone, Copy)]
enum FirstTurn {
	Tool,
	Error,
}

#[derive(Clone)]
struct TwoTurnService {
	calls: Arc<Mutex<Vec<Call>>>,
	first: FirstTurn,
}

impl TwoTurnService {
	fn tool() -> Self {
		Self { calls: Arc::new(Mutex::new(Vec::new())), first: FirstTurn::Tool }
	}

	fn error() -> Self {
		Self { calls: Arc::new(Mutex::new(Vec::new())), first: FirstTurn::Error }
	}
}

impl Service<Call> for TwoTurnService {
	type Error = Error;
	type Future = Ready<Result<Answer, Error>>;
	type Response = Answer;

	fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, call: Call) -> Self::Future {
		let attempt = {
			let mut calls = self.calls.lock();
			calls.push(call);
			calls.len()
		};
		if attempt == 1 && matches!(self.first, FirstTurn::Error) {
			return ready(Err(Error::new(
				ErrorKind::Protocol,
				ErrorPhase::Streaming,
				RetryAction::Never,
				ExecutionReceipt::default(),
			)));
		}
		let complete = Completion {
			reason:  FinishReason::Stop,
			blocks:  1,
			usage:   Usage::default(),
			receipt: ExecutionReceipt::default(),
		};
		let events = if attempt == 1 {
			vec![
				ChatEvent::ToolCallReady {
					index: 0,
					call:  ToolCall {
						id:        ToolCallId::from("call-1"),
						name:      Str::from("lookup"),
						arguments: OpaqueJson::new(json!({"q":"rust"})),
					},
				},
				ChatEvent::Completed(complete),
			]
		} else {
			vec![
				ChatEvent::TextDelta { index: 0, text: Str::from("done") },
				ChatEvent::Completed(complete),
			]
		};
		ready(Ok(Answer {
			meta:    ResponseMeta {
				request_id:          RequestId::from("request"),
				provider:            ProviderId::from("provider"),
				route:               RouteId::from("route"),
				model:               Some(ModelKey::from("model")),
				provider_request_id: None,
				created_at:          SystemTime::UNIX_EPOCH,
			},
			receipt: ExecutionReceipt::default(),
			body:    AnswerBody::Chat(Box::pin(stream::iter(events.into_iter().map(Ok)))),
		}))
	}
}

struct Lookup {
	definition: ToolDefinition,
	calls:      Arc<AtomicUsize>,
	output:     OwnedToolOutput,
}

impl Lookup {
	fn new(output: OwnedToolOutput) -> Self {
		Self { definition: tool_definition(), calls: Arc::new(AtomicUsize::new(0)), output }
	}
}

impl OwnedToolHandler for Lookup {
	type Execute<'a> = Ready<OwnedToolOutput>;

	fn definition(&self) -> &ToolDefinition {
		&self.definition
	}

	fn execute(&self, args: Bytes) -> Self::Execute<'_> {
		assert_eq!(args, Bytes::from_static(br#"{"q":"rust"}"#));
		self.calls.fetch_add(1, Ordering::SeqCst);
		ready(self.output.clone())
	}
}

fn tool_definition() -> ToolDefinition {
	ToolDefinition {
		name:        Str::from("lookup"),
		description: None,
		parameters:  OpaqueJson::new(json!({"type":"object"})),
		strict:      true,
	}
}

fn request() -> ChatRequest {
	ChatRequest {
		messages:          Arc::from([Message {
			role:    Role::User,
			content: Arc::from([ContentPart::Text { text: Str::from("help"), proof: None }]),
			name:    None,
		}]),
		tools:             Arc::from([]),
		hosted_tools:      Arc::from([]),
		tool_choice:       Setting::Unset,
		output:            Setting::Unset,
		reasoning:         Setting::Unset,
		verbosity:         Setting::Unset,
		cache_retention:   Setting::Unset,
		service_tier:      Setting::Unset,
		sampling:          Sampling::default(),
		max_output_tokens: None,
		top_logprobs:      None,
		safety:            Arc::from([]),
		negotiation:       NegotiationPolicy::default(),
	}
}

async fn client(service: TwoTurnService) -> (tempfile::TempDir, Client<TwoTurnService, Router>) {
	let state = tempfile::tempdir().expect("temporary state");
	let store = credential_store(&state.path().join("credentials.db"));
	let registry = omp_app::daemon::production_registry(state.path(), store)
		.await
		.expect("registry");
	let model = registry
		.catalog()
		.models()
		.iter()
		.find(|model| {
			model
				.capabilities
				.operations
				.contains_kind(OperationKind::Chat)
				&& model
					.routes
					.iter()
					.any(|route| registry.contains_service(route))
		})
		.expect("constructed chat model")
		.key
		.clone();
	let meta = CallMeta {
		id:       RequestId::from("request"),
		target:   Target::Model(model),
		deadline: None,
		budget:   ExecutionBudget::default(),
		session:  None,
	};
	let planner = Router::new(registry, std::time::Duration::from_secs(30));
	(state, Client::new(service, planner, meta))
}

fn second_request_contains_error_result(calls: &[Call], expected: bool) -> bool {
	let OperationCall::Chat(request) = &calls[1].operation else {
		panic!("owned tool loop must issue chat operations")
	};
	request.messages.iter().any(|message| {
		message.content.iter().any(
			|part| matches!(part, ContentPart::ToolResult { is_error, .. } if *is_error == expected),
		)
	})
}

#[tokio::test]
async fn ready_event_alone_authorizes_one_tool_and_follow_up() {
	let service = TwoTurnService::tool();
	let calls = service.calls.clone();
	let (_state, mut client) = client(service).await;
	let handler = Lookup::new(OwnedToolOutput {
		result:   OpaqueJson::new(json!({"ok":true})),
		is_error: false,
	});
	let output = run_owned_tool_loop(&mut client, request(), &handler)
		.await
		.expect("typed tool loop");
	assert!(!output.tool_result.is_error);
	assert_eq!(handler.calls.load(Ordering::SeqCst), 1);
	assert_eq!(calls.lock().len(), 2);
	assert!(second_request_contains_error_result(&calls.lock(), false));
}

#[tokio::test]
async fn tool_failure_is_an_authoritative_result_not_a_loop_failure() {
	let service = TwoTurnService::tool();
	let calls = service.calls.clone();
	let (_state, mut client) = client(service).await;
	let handler = Lookup::new(OwnedToolOutput {
		result:   OpaqueJson::new(json!({"error":"lookup failed"})),
		is_error: true,
	});
	let output = run_owned_tool_loop(&mut client, request(), &handler)
		.await
		.expect("failed tool result still permits model recovery");
	assert!(output.tool_result.is_error);
	assert_eq!(handler.calls.load(Ordering::SeqCst), 1);
	assert!(second_request_contains_error_result(&calls.lock(), true));
}

#[tokio::test]
async fn terminal_turn_error_stops_before_tool_execution() {
	let service = TwoTurnService::error();
	let calls = service.calls.clone();
	let (_state, mut client) = client(service).await;
	let handler = Lookup::new(OwnedToolOutput::text("must not execute"));
	let error = run_owned_tool_loop(&mut client, request(), &handler)
		.await
		.expect_err("terminal error must escape the loop");
	assert!(matches!(
		error,
		OwnedToolLoopError::Inference(error) if error.kind == ErrorKind::Protocol
	));
	assert_eq!(handler.calls.load(Ordering::SeqCst), 0);
	assert_eq!(calls.lock().len(), 1);
}

struct PendingHandler {
	definition: ToolDefinition,
	started:    Mutex<Option<oneshot::Sender<()>>>,
	dropped:    Arc<AtomicBool>,
}

struct PendingExecution {
	dropped: Arc<AtomicBool>,
}

impl Future for PendingExecution {
	type Output = OwnedToolOutput;

	fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
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

	fn definition(&self) -> &ToolDefinition {
		&self.definition
	}

	fn execute(&self, _: Bytes) -> Self::Execute<'_> {
		if let Some(started) = self.started.lock().take() {
			let _ = started.send(());
		}
		PendingExecution { dropped: self.dropped.clone() }
	}
}

#[tokio::test]
async fn cancelling_the_loop_drops_the_live_handler_and_never_starts_follow_up() {
	let service = TwoTurnService::tool();
	let calls = service.calls.clone();
	let (_state, mut client) = client(service).await;
	let (started_tx, started_rx) = oneshot::channel();
	let dropped = Arc::new(AtomicBool::new(false));
	let handler = PendingHandler {
		definition: tool_definition(),
		started:    Mutex::new(Some(started_tx)),
		dropped:    dropped.clone(),
	};
	let task =
		tokio::spawn(async move { run_owned_tool_loop(&mut client, request(), &handler).await });
	started_rx.await.expect("handler execution started");
	task.abort();
	assert!(task.await.expect_err("cancelled task").is_cancelled());
	assert!(dropped.load(Ordering::SeqCst));
	assert_eq!(calls.lock().len(), 1);
}
