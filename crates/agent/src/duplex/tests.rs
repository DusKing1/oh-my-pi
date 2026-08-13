use std::{collections::HashMap, sync::Arc, thread, time::Duration};

use bytes::Bytes;
use futures::{Stream, stream};
use omp_core::Str;
use omp_env::{EnvClient, frame};
use omp_proto::{
	inference::v1::{Invoke, InvokeComplete, InvokeInput, exec_status, invoke_input},
	thread::v1::ToolCall,
};
use omp_tool::{
	Constraint, Ev, IncomingParams, Part, PromptCaps, Registry, Rev, Tool, ToolSpec, Verdict,
};
use serde::{Deserialize, Serialize};

use super::{DuplexError, DuplexManager};
use crate::{EventBus, InvokeFrame};

const WAIT: Duration = Duration::from_secs(2);

fn expect_complete(
	result: Result<InvokeFrame, DuplexError>,
	context: &str,
) -> InvokeComplete {
	match result.expect(context) {
		InvokeFrame::Complete(complete) => complete,
		InvokeFrame::Input(_) => panic!("expected terminal invocation completion"),
	}
}

#[derive(Debug, Deserialize, Serialize)]
struct Params {
	value: u64,
}

#[derive(Debug, Deserialize, Serialize)]
struct Payload {
	answer: Str,
}

#[derive(Debug, Deserialize, Serialize)]
struct Fault {
	message: Str,
}

struct ScriptTool {
	spec:           ToolSpec,
	project_inputs: bool,
}

impl ScriptTool {
	fn new(project_inputs: bool) -> Self {
		Self {
			spec: ToolSpec {
				name: Str::from("script"),
				rev: Rev { family: Str::from("script"), n: 1 },
				description: Str::from("scripted duplex tool"),
				schema: Bytes::from_static(
					br#"{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]}"#,
				),
				constraint: Constraint::None,
			},
			project_inputs,
		}
	}
}

impl Tool for ScriptTool {
	type Fault = Fault;
	type Params = Params;
	type Payload = Payload;
	type Update = Str;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'c>(
		&'c self,
		_params: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<Self::Update, Self::Payload, Self::Fault>> + Send + 'c {
		stream::empty()
	}

	fn prompt(&self, view: Result<&Self::Payload, &Self::Fault>, _caps: &PromptCaps) -> Vec<Part> {
		let text = match view {
			Ok(payload) => payload.answer.clone(),
			Err(fault) => fault.message.clone(),
		};
		vec![Part::Text { text }]
	}

	fn invoke_input(&self, update: &Self::Update, invocation_id: &str) -> Option<InvokeInput> {
		self.project_inputs.then(|| InvokeInput {
			invocation_id: invocation_id.to_owned(),
			payload: Some(invoke_input::Payload::Chunk(invoke_input::Chunk {
				channel: invoke_input::chunk::Channel::Progress as i32,
				data: Bytes::copy_from_slice(update.as_bytes()),
			})),
		})
	}
}

fn manager_with_projection(
	project_inputs: bool,
) -> (DuplexManager, flume::Receiver<frame::ClientFrame>, flume::Sender<frame::ServerFrame>) {
	let (env, transport) = EnvClient::in_process(0);
	let (requests, responses) = transport.into_parts();
	let mut registry = Registry::new();
	registry
		.register(ScriptTool::new(project_inputs))
		.expect("register scripted tool");
	(
		DuplexManager::new(
			env,
			Arc::new(registry),
			EventBus::new(),
			PromptCaps { maximum_parts: 8, maximum_text_bytes: 4096, media: false },
			Duration::from_millis(20),
		),
		requests,
		responses,
	)
}

fn manager()
-> (DuplexManager, flume::Receiver<frame::ClientFrame>, flume::Sender<frame::ServerFrame>) {
	manager_with_projection(false)
}

fn invoke(invocation_id: &str, call_id: &str, value: u64) -> Invoke {
	Invoke {
		invocation_id: invocation_id.to_owned(),
		name: "script".to_owned(),
		tool_call: Some(ToolCall {
			id: call_id.to_owned(),
			name: "script".to_owned(),
			args_json: Bytes::from(format!("{{\"value\":{value}}}")),
			..Default::default()
		}),
		timeout_ms: 2_000,
		..Default::default()
	}
}

fn recv(requests: &flume::Receiver<frame::ClientFrame>) -> frame::ClientFrame {
	requests.recv_timeout(WAIT).expect("client frame")
}

fn respond(
	responses: &flume::Sender<frame::ServerFrame>,
	request_id: u64,
	body: frame::server_frame::Body,
) {
	responses
		.send(frame::ServerFrame { request_id, body: Some(body), ..Default::default() })
		.expect("live response channel");
}

fn serve_one(
	requests: flume::Receiver<frame::ClientFrame>,
	responses: flume::Sender<frame::ServerFrame>,
	call_id: &'static str,
	expected_args: Bytes,
	verdict: Bytes,
	is_error: bool,
) -> thread::JoinHandle<()> {
	thread::spawn(move || {
		let open = recv(&requests);
		let request_id = open.request_id;
		match open.body {
			Some(frame::client_frame::Body::InvokeTool(open)) => {
				assert_eq!(open.invocation_id, call_id);
				assert_eq!(open.name, "script");
				assert_eq!(open.rev, "script.1");
			},
			body => panic!("expected InvokeTool, got {body:?}"),
		}
		match recv(&requests).body {
			Some(frame::client_frame::Body::ArgText(args)) => {
				assert_eq!(args.invocation_id, call_id);
				assert_eq!(args.fragment.as_bytes(), expected_args.as_ref());
			},
			body => panic!("expected ArgText, got {body:?}"),
		}
		match recv(&requests).body {
			Some(frame::client_frame::Body::ArgsCommitted(args)) => {
				assert_eq!(args.invocation_id, call_id);
				assert_eq!(args.raw, expected_args);
			},
			body => panic!("expected ArgsCommitted, got {body:?}"),
		}
		respond(
			&responses,
			request_id,
			frame::server_frame::Body::Verdict(frame::Verdict {
				invocation_id: call_id.to_owned(),
				json: verdict,
				is_error,
				..Default::default()
			}),
		);
	})
}

#[tokio::test]
async fn successful_invocation_preserves_canonical_result() {
	let (mut manager, requests, responses) = manager();
	let args = Bytes::from_static(br#"{"value":7}"#);
	let verdict = Bytes::from(
		serde_json::to_vec(&Verdict::<Payload, Fault>::Ok(Payload { answer: Str::from("seven") }))
			.expect("serialize verdict"),
	);
	let server = serve_one(requests, responses, "call-1", args, verdict, false);
	manager.start(invoke("invoke-1", "call-1", 7));
	let (id, complete) = manager.next().await.expect("completion");
	let complete = expect_complete(complete, "successful duplex execution");
	assert_eq!(id, "invoke-1");
	assert_eq!(complete.invocation_id, "invoke-1");
	assert_eq!(complete.status.expect("typed status").outcome, exec_status::Outcome::Exited as i32);
	let result = complete.tool_result.expect("canonical tool result");
	assert_eq!(result.call_id, "call-1");
	assert_eq!(result.name, "script");
	assert!(!result.is_error);
	assert_eq!(result.useless, Some(false));
	assert!(manager.is_empty());
	server.join().expect("scripted server");
}

#[tokio::test]
async fn typed_fault_completes_as_failed_without_losing_result_fields() {
	let (mut manager, requests, responses) = manager();
	let args = Bytes::from_static(br#"{"value":9}"#);
	let verdict = Bytes::from(
		serde_json::to_vec(&Verdict::<Payload, Fault>::Fault(Fault { message: Str::from("boom") }))
			.expect("serialize verdict"),
	);
	let server = serve_one(requests, responses, "call-f", args, verdict, true);
	manager.start(invoke("invoke-f", "call-f", 9));
	let (_, complete) = manager.next().await.expect("completion");
	let complete = expect_complete(complete, "fault is a canonical completion");
	assert_eq!(complete.status.expect("typed status").outcome, exec_status::Outcome::Failed as i32);
	let result = complete.tool_result.expect("fault result");
	assert!(result.is_error);
	assert_eq!(result.call_id, "call-f");
	assert_eq!(result.name, "script");
	assert_eq!(result.useless, Some(false));
	server.join().expect("scripted server");
}

#[tokio::test]
async fn two_invocations_remain_concurrent_and_complete_independently() {
	let (mut manager, requests, responses) = manager();
	let server = thread::spawn(move || {
		let mut ids = HashMap::new();
		let mut commits = 0;
		while commits < 2 {
			let frame = recv(&requests);
			match frame.body {
				Some(frame::client_frame::Body::InvokeTool(open)) => {
					ids.insert(open.invocation_id, frame.request_id);
				},
				Some(frame::client_frame::Body::ArgsCommitted(_)) => commits += 1,
				Some(frame::client_frame::Body::ArgText(_)) => {},
				body => panic!("unexpected concurrent frame: {body:?}"),
			}
		}
		assert_eq!(ids.len(), 2, "both invocations opened before either completed");
		for call_id in ["call-b", "call-a"] {
			let verdict = Verdict::<Payload, Fault>::Ok(Payload { answer: Str::from(call_id) });
			respond(
				&responses,
				ids[call_id],
				frame::server_frame::Body::Verdict(frame::Verdict {
					invocation_id: call_id.to_owned(),
					json: Bytes::from(serde_json::to_vec(&verdict).expect("serialize verdict")),
					..Default::default()
				}),
			);
		}
	});
	manager.start(invoke("invoke-a", "call-a", 1));
	manager.start(invoke("invoke-b", "call-b", 2));
	let first = manager.next().await.expect("first completion").0;
	let second = manager.next().await.expect("second completion").0;
	assert_eq!([first.as_str(), second.as_str()], ["invoke-b", "invoke-a"]);
	assert!(manager.is_empty());
	server.join().expect("scripted server");
}
#[tokio::test]
async fn cancellation_interrupts_then_structurally_cancels_and_suppresses_completion() {
	let (mut manager, requests, responses) = manager_with_projection(true);
	let (committed_tx, committed_rx) = flume::bounded(1);
	let server = thread::spawn(move || {
		let open = recv(&requests);
		let request_id = open.request_id;
		assert!(matches!(open.body, Some(frame::client_frame::Body::InvokeTool(_))));
		assert!(matches!(recv(&requests).body, Some(frame::client_frame::Body::ArgText(_))));
		assert!(matches!(recv(&requests).body, Some(frame::client_frame::Body::ArgsCommitted(_))));
		committed_tx.send(()).expect("notify commit");
		respond(
			&responses,
			request_id,
			frame::server_frame::Body::Update(frame::Update {
				invocation_id: "call-c".to_owned(),
				json: Bytes::from_static(br#""late""#),
				..Default::default()
			}),
		);
		let interrupt = recv(&requests);
		assert_eq!(interrupt.request_id, request_id);
		assert!(matches!(interrupt.body, Some(frame::client_frame::Body::Interrupt(_))));
		let cancel = recv(&requests);
		assert_eq!(cancel.request_id, 0);
		assert!(matches!(cancel.body, Some(frame::client_frame::Body::Cancel(_))));
		drop(responses);
	});
	manager.start(invoke("invoke-c", "call-c", 3));
	committed_rx
		.recv_async()
		.await
		.expect("invocation committed");
	manager.cancel("invoke-c");
	assert!(manager.is_empty());
	assert!(manager.next().await.is_none(), "cancelled completion must be suppressed");
	server
		.join()
		.expect("interrupt and structural cancellation");
}

#[tokio::test]
async fn unknown_cancellation_is_a_no_op() {
	let (mut manager, _requests, _responses) = manager();
	manager.cancel("not-live");
	assert!(manager.is_empty());
	assert!(manager.next().await.is_none());
}

#[tokio::test]
async fn invalid_control_invocation_fails_once_without_environment_dispatch() {
	let (mut manager, requests, _responses) = manager();
	manager.start(Invoke {
		invocation_id: "control".to_owned(),
		name: "script".to_owned(),
		..Default::default()
	});
	let (_, complete) = manager.next().await.expect("typed invalid completion");
	let complete = expect_complete(complete, "invalid invocation is represented, not raised");
	let status = complete.status.expect("failed status");
	assert_eq!(status.outcome, exec_status::Outcome::Failed as i32);
	assert!(!status.reason.is_empty());
	assert!(complete.tool_result.is_none());
	assert!(manager.next().await.is_none(), "one invocation yields at most one completion");
	assert!(matches!(
		requests.recv_timeout(Duration::from_millis(50)),
		Err(flume::RecvTimeoutError::Timeout)
	));
}

#[tokio::test]
async fn zero_deadline_fails_without_environment_dispatch() {
	let (mut manager, requests, _responses) = manager();
	let mut request = invoke("invoke-zero", "call-zero", 0);
	request.timeout_ms = 0;
	manager.start(request);
	let (_, complete) = manager.next().await.expect("typed deadline completion");
	let complete = expect_complete(complete, "zero deadline is represented, not raised");
	let status = complete.status.expect("failed status");
	assert_eq!(status.outcome, exec_status::Outcome::Failed as i32);
	assert!(status.reason.contains("deadline"));
	assert!(complete.tool_result.is_none());
	assert!(matches!(
		requests.recv_timeout(Duration::from_millis(50)),
		Err(flume::RecvTimeoutError::Timeout)
	));
}

#[tokio::test]
async fn dropping_manager_interrupts_active_tasks_before_sender_destruction() {
	let (mut manager, requests, responses) = manager();
	let server = thread::spawn(move || {
		let mut opened = false;
		let mut relayed = false;
		loop {
			let frame = recv(&requests);
			match frame.body {
				Some(frame::client_frame::Body::InvokeTool(_)) => opened = true,
				Some(frame::client_frame::Body::ArgText(_)) => relayed = true,
				Some(frame::client_frame::Body::Cancel(_)) => break,
				Some(frame::client_frame::Body::ArgsCommitted(_)) => {
					panic!("shutdown observed before execution must not commit effects")
				},
				body => panic!("unexpected shutdown frame: {body:?}"),
			}
		}
		assert!(opened);
		assert!(relayed);
		drop(responses);
	});
	manager.start(invoke("invoke-drop", "call-drop", 4));
	drop(manager);
	tokio::task::spawn_blocking(move || server.join())
		.await
		.expect("join task")
		.expect("manager drop structurally cancelled invocation");
}

#[tokio::test]
async fn typed_updates_are_ordered_before_completion() {
	let (mut manager, requests, responses) = manager_with_projection(true);
	let server = thread::spawn(move || {
		let open = recv(&requests);
		let request_id = open.request_id;
		assert!(matches!(open.body, Some(frame::client_frame::Body::InvokeTool(_))));
		assert!(matches!(recv(&requests).body, Some(frame::client_frame::Body::ArgText(_))));
		assert!(matches!(recv(&requests).body, Some(frame::client_frame::Body::ArgsCommitted(_))));
		for raw in [
			Bytes::from_static(br#""first""#),
			Bytes::from_static(br#""second""#),
		] {
			respond(
				&responses,
				request_id,
				frame::server_frame::Body::Update(frame::Update {
					invocation_id: "call-stream".to_owned(),
					json: raw,
					..Default::default()
				}),
			);
		}
		let verdict = Verdict::<Payload, Fault>::Ok(Payload { answer: Str::from("done") });
		respond(
			&responses,
			request_id,
			frame::server_frame::Body::Verdict(frame::Verdict {
				invocation_id: "call-stream".to_owned(),
				json: Bytes::from(serde_json::to_vec(&verdict).expect("serialize verdict")),
				..Default::default()
			}),
		);
	});
	manager.start(invoke("invoke-stream", "call-stream", 5));
	for expected in [b"first".as_slice(), b"second".as_slice()] {
		let (id, frame) = manager.next().await.expect("streamed input");
		assert_eq!(id, "invoke-stream");
		let InvokeFrame::Input(input) = frame.expect("typed input projection") else {
			panic!("completion arrived before all projected updates");
		};
		let Some(invoke_input::Payload::Chunk(chunk)) = input.payload else {
			panic!("expected canonical input chunk");
		};
		assert_eq!(chunk.channel, invoke_input::chunk::Channel::Progress as i32);
		assert_eq!(chunk.data.as_ref(), expected);
	}
	let (_, terminal) = manager.next().await.expect("terminal completion");
	let terminal = expect_complete(terminal, "streaming invocation completion");
	assert_eq!(terminal.invocation_id, "invoke-stream");
	assert!(manager.is_empty());
	server.join().expect("scripted streaming server");
}

#[tokio::test]
async fn default_update_projection_yields_only_completion() {
	let (mut manager, requests, responses) = manager();
	let server = thread::spawn(move || {
		let open = recv(&requests);
		let request_id = open.request_id;
		assert!(matches!(open.body, Some(frame::client_frame::Body::InvokeTool(_))));
		assert!(matches!(recv(&requests).body, Some(frame::client_frame::Body::ArgText(_))));
		assert!(matches!(recv(&requests).body, Some(frame::client_frame::Body::ArgsCommitted(_))));
		respond(
			&responses,
			request_id,
			frame::server_frame::Body::Update(frame::Update {
				invocation_id: "call-default".to_owned(),
				json: Bytes::from_static(br#""event-only""#),
				..Default::default()
			}),
		);
		let verdict = Verdict::<Payload, Fault>::Ok(Payload { answer: Str::from("done") });
		respond(
			&responses,
			request_id,
			frame::server_frame::Body::Verdict(frame::Verdict {
				invocation_id: "call-default".to_owned(),
				json: Bytes::from(serde_json::to_vec(&verdict).expect("serialize verdict")),
				..Default::default()
			}),
		);
	});
	manager.start(invoke("invoke-default", "call-default", 6));
	let (_, only) = manager.next().await.expect("only completion");
	let _ = expect_complete(only, "default projection completion");
	assert!(manager.next().await.is_none());
	server.join().expect("default projection server");
}

#[tokio::test]
async fn malformed_typed_update_returns_projection_error_without_panic() {
	let (mut manager, requests, responses) = manager_with_projection(true);
	let server = thread::spawn(move || {
		let open = recv(&requests);
		let request_id = open.request_id;
		assert!(matches!(open.body, Some(frame::client_frame::Body::InvokeTool(_))));
		assert!(matches!(recv(&requests).body, Some(frame::client_frame::Body::ArgText(_))));
		assert!(matches!(recv(&requests).body, Some(frame::client_frame::Body::ArgsCommitted(_))));
		respond(
			&responses,
			request_id,
			frame::server_frame::Body::Update(frame::Update {
				invocation_id: "call-malformed".to_owned(),
				json: Bytes::from_static(br#"{"not":"a string"}"#),
				..Default::default()
			}),
		);
		assert!(matches!(recv(&requests).body, Some(frame::client_frame::Body::Cancel(_))));
	});
	manager.start(invoke("invoke-malformed", "call-malformed", 7));
	let (_, result) = manager.next().await.expect("projection failure");
	assert!(matches!(
		result,
		Err(DuplexError::Registry(omp_tool::RegistryError::UpdateShape {
			name,
			rev,
			..
		})) if name.as_str() == "script" && rev.to_string() == "script.1"
	));
	assert!(manager.is_empty());
	server.join().expect("malformed update server");
}
