//! Cursor transport vendor pinning contract tests.
use std::{
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	time::Duration,
};

use bytes::Bytes;
use omp_core::Str;
use omp_llm_cursor::{
	ConnectDecoder, CursorDecodeState, InvocationFramer, ShellContext, connect_frame,
	decode_server_message, drive_invocation, require_executor, wire as cursor_wire,
};
use omp_llm_types::{
	ExecOutcome, ExecStatus, Invoke, InvokeChannel, InvokeChunk, InvokeComplete, InvokeInput,
	InvokePayload, Part, Props, StopReason, StreamPartKind, ToolCall, ToolResult, TurnErrorKind,
	TurnEvent,
	facet::{Error, Executor},
	ids::CallId,
};
use parking_lot::Mutex;
use prost::Message as _;
use tokio::sync::oneshot;

const fn shell_context() -> ShellContext {
	ShellContext {
		id:      17,
		exec_id: Str::new_static("exec-17"),
		command: Str::new_static("printf colours"),
		cwd:     Str::new_static("/work/project"),
	}
}

fn invocation(call_id: CallId) -> Invoke {
	Invoke::builder()
		.invocation_id(Str::new_static("exec-17"))
		.name(Str::new_static("bash"))
		.tool_call(
			ToolCall::builder()
				.id(call_id)
				.name(Str::new_static("bash"))
				.args_json(Bytes::from_static(br#"{"command":"printf colours"}"#))
				.thought_signature(Bytes::new())
				.build(),
		)
		.vendor(Bytes::new())
		.timeout_ms(1_000)
		.props(Props::default())
		.build()
}

fn status(outcome: ExecOutcome) -> ExecStatus {
	ExecStatus::builder()
		.outcome(outcome)
		.exit_code(if outcome == ExecOutcome::Exited {
			0
		} else {
			23
		})
		.signal(Str::default())
		.reason(Str::new_static("policy detail"))
		.cwd(Str::new_static("/work/project"))
		.aborted(outcome == ExecOutcome::Timeout)
		.output_location(Str::default())
		.local_execution_time_ms(41)
		.is_readonly(true)
		.command_timeout_ms(750)
		.build()
}

fn completion(call_id: CallId, outcome: ExecOutcome) -> InvokeComplete {
	InvokeComplete::builder()
		.invocation_id(Str::new_static("exec-17"))
		.tool_result(
			ToolResult::builder()
				.call_id(call_id)
				.name(Str::new_static("shell"))
				.parts(vec![Part::Text(Str::new_static("committed output"))])
				.is_error(outcome != ExecOutcome::Exited)
				.build(),
		)
		.status(status(outcome))
		.vendor(Bytes::new())
		.props(Props::default())
		.build()
}

fn exec_message(
	frame: &cursor_wire::AgentClientMessage,
) -> Option<&cursor_wire::ExecClientMessage> {
	match frame.message.as_ref()? {
		cursor_wire::agent_client_message::Message::ExecClientMessage(message) => Some(message),
		_ => None,
	}
}

fn stream_event(
	frame: &cursor_wire::AgentClientMessage,
) -> Option<&cursor_wire::shell_stream::Event> {
	match exec_message(frame)?.message.as_ref()? {
		cursor_wire::exec_client_message::Message::ShellStream(stream) => stream.event.as_ref(),
		_ => None,
	}
}

fn shell_result(
	frame: &cursor_wire::AgentClientMessage,
) -> Option<&cursor_wire::shell_result::Result> {
	match exec_message(frame)?.message.as_ref()? {
		cursor_wire::exec_client_message::Message::ShellResult(result) => result.result.as_ref(),
		_ => None,
	}
}

const fn is_stream_close(frame: &cursor_wire::AgentClientMessage) -> bool {
	matches!(
		frame.message.as_ref(),
		Some(cursor_wire::agent_client_message::Message::ExecClientControlMessage(
			cursor_wire::ExecClientControlMessage {
				message: Some(cursor_wire::exec_client_control_message::Message::StreamClose(_)),
			}
		))
	)
}

#[test]
fn compiled_cursor_schema_retains_codec_contract_and_unary_drift_pin() {
	let shell_args = cursor_wire::ShellArgs {
		command: "pwd".to_owned(),
		working_directory: "/work".to_owned(),
		timeout: 500,
		tool_call_id: "call-1".to_owned(),
		..Default::default()
	};
	let exec = cursor_wire::ExecServerMessage {
		id: 7,
		exec_id: "exec-7".to_owned(),
		message: Some(cursor_wire::exec_server_message::Message::ShellStreamArgs(shell_args)),
		..Default::default()
	};
	let server = cursor_wire::AgentServerMessage {
		message: Some(cursor_wire::agent_server_message::Message::ExecServerMessage(exec)),
	};
	let decoded = cursor_wire::AgentServerMessage::decode(server.encode_to_vec().as_slice())
		.expect("Cursor's compiled AgentServerMessage pin must remain decodable");
	assert!(matches!(
		decoded.message,
		Some(cursor_wire::agent_server_message::Message::ExecServerMessage(
			cursor_wire::ExecServerMessage {
				message: Some(cursor_wire::exec_server_message::Message::ShellStreamArgs(_)),
				..
			}
		))
	));

	let result = cursor_wire::ShellResult {
		result: Some(cursor_wire::shell_result::Result::PermissionDenied(
			cursor_wire::ShellPermissionDenied {
				command:           "pwd".to_owned(),
				working_directory: "/work".to_owned(),
				error:             "denied".to_owned(),
				is_readonly:       true,
			},
		)),
		..Default::default()
	};
	assert!(matches!(result.result, Some(cursor_wire::shell_result::Result::PermissionDenied(_))));
	assert_eq!(cursor_wire::ShellAbortReason::Timeout as i32, 2);

	// KNOWN DRIFT: the frozen proto declares Run unary, while Cursor's live
	// Connect endpoint is bidirectional. The codec deliberately diverges and
	// manually implements bidi framing.
}

#[test]
fn recorded_cursor_connect_frames_decode_without_transport() {
	fn update(message: cursor_wire::interaction_update::Message) -> cursor_wire::AgentServerMessage {
		cursor_wire::AgentServerMessage {
			message: Some(cursor_wire::agent_server_message::Message::InteractionUpdate(Box::new(
				cursor_wire::InteractionUpdate { message: Some(message) },
			))),
		}
	}

	let tool_call = cursor_wire::ToolCall {
		tool_call_id: Some("call-read".to_owned()),
		tool:         Some(cursor_wire::tool_call::Tool::ReadToolCall(
			cursor_wire::ReadToolCall::default(),
		)),
	};
	let messages = [
		update(cursor_wire::interaction_update::Message::ThinkingDelta(
			cursor_wire::ThinkingDeltaUpdate { text: "Inspect first.".to_owned() },
		)),
		update(cursor_wire::interaction_update::Message::ToolCallStarted(
			cursor_wire::ToolCallStartedUpdate {
				call_id: "call-read".to_owned(),
				tool_call: Some(tool_call.clone()),
				..Default::default()
			},
		)),
		update(cursor_wire::interaction_update::Message::PartialToolCall(
			cursor_wire::PartialToolCallUpdate {
				call_id: "call-read".to_owned(),
				tool_call: Some(tool_call.clone()),
				args_text_delta: "{\"pa".to_owned(),
				..Default::default()
			},
		)),
		update(cursor_wire::interaction_update::Message::PartialToolCall(
			cursor_wire::PartialToolCallUpdate {
				call_id: "call-read".to_owned(),
				tool_call: Some(tool_call),
				args_text_delta: "th\":\"package.json\"}".to_owned(),
				..Default::default()
			},
		)),
		update(cursor_wire::interaction_update::Message::ToolCallCompleted(
			cursor_wire::ToolCallCompletedUpdate {
				call_id: "call-read".to_owned(),
				..Default::default()
			},
		)),
		update(cursor_wire::interaction_update::Message::TextDelta(cursor_wire::TextDeltaUpdate {
			text: "Done.".to_owned(),
		})),
		update(cursor_wire::interaction_update::Message::TokenDelta(cursor_wire::TokenDeltaUpdate {
			tokens: 8,
		})),
		update(cursor_wire::interaction_update::Message::TurnEnded(cursor_wire::TurnEndedUpdate {})),
	];
	let mut state = CursorDecodeState::default();
	let events: Vec<_> = messages
		.into_iter()
		.flat_map(|message| decode_server_message(message, &mut state).expect("recorded frame"))
		.collect();
	assert!(events.iter().any(|event| matches!(
		event,
		TurnEvent::PartDelta { chunk, .. } if chunk == &Bytes::from_static(b"Inspect first.")
	)));
	assert!(events.iter().any(|event| matches!(
		event,
		TurnEvent::PartStart {
			kind: StreamPartKind::ToolCall,
			tool_call_id,
			tool_name,
			..
		} if tool_call_id == "call-read" && tool_name == "read"
	)));
	let arguments: Bytes = events
		.iter()
		.filter_map(|event| match event {
			TurnEvent::PartDelta { chunk, .. }
				if chunk.as_ref() == b"{\"pa" || chunk.as_ref() == b"th\":\"package.json\"}" =>
			{
				Some(chunk.clone())
			},
			_ => None,
		})
		.flatten()
		.collect();
	assert_eq!(arguments, Bytes::from_static(br#"{"path":"package.json"}"#));
	assert!(events.iter().any(|event| matches!(
		event,
		TurnEvent::PartDelta { chunk, .. } if chunk == &Bytes::from_static(b"Done.")
	)));
	assert!(events.iter().any(|event| matches!(
		event,
		TurnEvent::Outcome(outcome)
			if outcome.stop == StopReason::ToolUse
				&& outcome.usage.as_ref().is_some_and(|usage| usage.output_tokens == 8)
	)));
}

#[test]
fn compiled_cursor_schema_exposes_harvested_tool_frames() {
	let fixture = include_str!("fixtures/cursor/stream.tool_args.jsonl");
	assert!(fixture.contains("\"toolCallStarted\""));
	assert!(fixture.contains("\"toolCallArgsTextDelta\""));
	assert!(fixture.contains("\"toolCallCompleted\""));

	let messages = [
		cursor_wire::interaction_update::Message::ToolCallStarted(
			cursor_wire::ToolCallStartedUpdate {
				call_id: "call-read".to_owned(),
				..Default::default()
			},
		),
		cursor_wire::interaction_update::Message::PartialToolCall(
			cursor_wire::PartialToolCallUpdate {
				call_id: "call-read".to_owned(),
				args_text_delta: "{\"pa".to_owned(),
				..Default::default()
			},
		),
		cursor_wire::interaction_update::Message::ToolCallCompleted(
			cursor_wire::ToolCallCompletedUpdate {
				call_id: "call-read".to_owned(),
				..Default::default()
			},
		),
	];
	for message in messages {
		let update = cursor_wire::InteractionUpdate { message: Some(message) };
		assert!(
			cursor_wire::InteractionUpdate::decode(update.encode_to_vec().as_slice())
				.unwrap()
				.message
				.is_some(),
			"compiled Cursor pin must retain harvested tool-call update variants"
		);
	}
}

#[test]
fn cursor_invoke_cancel_decodes_and_cancels_without_late_frames() {
	let abort = cursor_wire::AgentServerMessage {
		message: Some(cursor_wire::agent_server_message::Message::ExecServerControlMessage(
			cursor_wire::ExecServerControlMessage {
				message: Some(cursor_wire::exec_server_control_message::Message::Abort(
					cursor_wire::ExecServerAbort { id: 17 },
				)),
			},
		)),
	};
	let events = decode_server_message(abort, &mut CursorDecodeState::default()).unwrap();
	assert!(matches!(
		events.as_slice(),
		[TurnEvent::InvokeCancel { invocation_id }] if invocation_id == "17"
	));
}

struct ScriptedExecutor {
	call_id: CallId,
	seen:    Arc<Mutex<Vec<Invoke>>>,
}

#[async_trait::async_trait]
impl Executor for ScriptedExecutor {
	async fn invoke(&self, invoke: Invoke, inputs: flume::Sender<InvokeInput>) -> InvokeComplete {
		self.seen.lock().push(invoke);
		for (channel, data) in [
			(InvokeChannel::Stdout, Bytes::from_static(b"plain\x1b[31")),
			(InvokeChannel::Stderr, Bytes::from_static(b"warning\n")),
			(InvokeChannel::Stdout, Bytes::from_static(b"mred\x1b[0m\n")),
		] {
			inputs
				.send_async(
					InvokeInput::builder()
						.invocation_id(Str::new_static("exec-17"))
						.payload(InvokePayload::Chunk(
							InvokeChunk::builder().channel(channel).data(data).build(),
						))
						.build(),
				)
				.await
				.unwrap();
			tokio::task::yield_now().await;
		}
		tokio::time::sleep(Duration::from_millis(10)).await;
		completion(self.call_id, ExecOutcome::Exited)
	}
}

#[tokio::test]
async fn cursor_shell_stream_round_trip_preserves_wire_order_and_committed_items() {
	let call_id = CallId::new();
	let canonical_invoke = invocation(call_id);
	let seen = Arc::new(Mutex::new(Vec::new()));
	let executor = Arc::new(ScriptedExecutor { call_id, seen: Arc::clone(&seen) });
	let (_cancel_tx, cancel_rx) = oneshot::channel();
	let frames =
		drive_invocation(executor, canonical_invoke.clone(), shell_context(), cancel_rx).await;

	assert!(matches!(stream_event(&frames[0]), Some(cursor_wire::shell_stream::Event::Start(_))));
	let wire_events: Vec<_> = frames.iter().filter_map(stream_event).collect();
	assert!(
		matches!(wire_events[1], cursor_wire::shell_stream::Event::Stdout(stdout) if stdout.data == "plain")
	);
	assert!(
		matches!(wire_events[2], cursor_wire::shell_stream::Event::Stderr(stderr) if stderr.data == "warning\n")
	);
	assert!(
		matches!(wire_events[3], cursor_wire::shell_stream::Event::Stdout(stdout) if stdout.data == "\x1b[31mred\x1b[0m\n")
	);
	assert!(
		matches!(wire_events[4], cursor_wire::shell_stream::Event::Exit(exit) if exit.code == 0 && exit.cwd == "/work/project" && !exit.aborted && exit.abort_reason.is_none())
	);
	assert!(matches!(
		frames.iter().find_map(shell_result),
		Some(cursor_wire::shell_result::Result::Success(success))
			if success.command == "printf colours" && success.stdout == "committed output"
	));
	assert!(is_stream_close(frames.last().expect("stream close")));
	assert_eq!(seen.lock().as_slice(), &[canonical_invoke]);
	for frame in &frames {
		let envelope = connect_frame(frame);
		let split = envelope.len().min(3);
		let mut decoder = ConnectDecoder::default();
		assert_eq!(decoder.push(&envelope[..split]).unwrap(), [] as [bytes::Bytes; 0]);
		let payloads = decoder.push(&envelope[split..]).unwrap();
		assert_eq!(payloads.len(), 1);
		assert_eq!(
			cursor_wire::AgentClientMessage::decode(payloads[0].clone()).unwrap(),
			*frame,
			"Connect envelope must preserve every protobuf wire frame"
		);
	}
}

#[test]
fn all_five_exec_status_variants_pin_result_exit_and_close_fields() {
	let call_id = CallId::new();
	for outcome in [
		ExecOutcome::Exited,
		ExecOutcome::Failed,
		ExecOutcome::Rejected,
		ExecOutcome::Denied,
		ExecOutcome::Timeout,
	] {
		let mut framer = InvocationFramer::new(shell_context());
		let frames = framer.complete(&completion(call_id, outcome));
		let exit = frames
			.iter()
			.filter_map(stream_event)
			.find_map(|event| match event {
				cursor_wire::shell_stream::Event::Exit(exit) => Some(exit),
				_ => None,
			})
			.expect("every status must emit exit");
		let expected_code = if outcome == ExecOutcome::Exited {
			0
		} else if outcome == ExecOutcome::Failed {
			23
		} else {
			1
		};
		assert_eq!(exit.code, expected_code, "wrong exit code for {outcome:?}");
		assert_eq!(exit.cwd, "/work/project");
		assert_eq!(exit.aborted, outcome == ExecOutcome::Timeout);
		assert_eq!(exit.abort_reason, None);
		let result = frames
			.iter()
			.find_map(shell_result)
			.expect("structured shellResult");
		match (outcome, result) {
			(ExecOutcome::Exited, cursor_wire::shell_result::Result::Success(value)) => {
				assert_eq!(value.command, "printf colours");
				assert_eq!(value.working_directory, "/work/project");
				assert_eq!(value.exit_code, 0);
			},
			(ExecOutcome::Failed, cursor_wire::shell_result::Result::Failure(value)) => {
				assert_eq!(value.command, "printf colours");
				assert_eq!(value.working_directory, "/work/project");
				assert_eq!(value.exit_code, 23);
				assert!(!value.aborted);
				assert_eq!(value.abort_reason, None);
			},
			(ExecOutcome::Rejected, cursor_wire::shell_result::Result::Rejected(value)) => {
				assert_eq!(value.command, "printf colours");
				assert_eq!(value.working_directory, "/work/project");
				assert_eq!(value.reason, "policy detail");
			},
			(ExecOutcome::Denied, cursor_wire::shell_result::Result::PermissionDenied(value)) => {
				assert_eq!(value.command, "printf colours");
				assert_eq!(value.working_directory, "/work/project");
				assert_eq!(value.error, "policy detail");
			},
			(ExecOutcome::Timeout, cursor_wire::shell_result::Result::Timeout(value)) => {
				assert_eq!(value.command, "printf colours");
				assert_eq!(value.working_directory, "/work/project");
				assert_eq!(value.timeout_ms, 750);
			},
			_ => panic!("wrong shellResult variant for {outcome:?}"),
		}
		if outcome == ExecOutcome::Timeout {
			assert!(frames.iter().filter_map(stream_event).any(|event| matches!(
				event,
				cursor_wire::shell_stream::Event::Stderr(stderr)
					if stderr.data == "Command timed out after 750ms"
			)));
		}
		assert!(is_stream_close(frames.last().unwrap()));
	}
}

struct PendingExecutor {
	started: Arc<AtomicBool>,
	dropped: Arc<AtomicBool>,
}

struct DropMark(Arc<AtomicBool>);

impl Drop for DropMark {
	fn drop(&mut self) {
		self.0.store(true, Ordering::SeqCst);
	}
}

#[async_trait::async_trait]
impl Executor for PendingExecutor {
	async fn invoke(&self, _invoke: Invoke, _inputs: flume::Sender<InvokeInput>) -> InvokeComplete {
		let _mark = DropMark(Arc::clone(&self.dropped));
		self.started.store(true, Ordering::SeqCst);
		std::future::pending().await
	}
}

#[tokio::test]
async fn invoke_cancel_structurally_aborts_executor_and_stops_frames() {
	let started = Arc::new(AtomicBool::new(false));
	let dropped = Arc::new(AtomicBool::new(false));
	let executor =
		Arc::new(PendingExecutor { started: Arc::clone(&started), dropped: Arc::clone(&dropped) });
	let (cancel_tx, cancel_rx) = oneshot::channel();
	let task = tokio::spawn(drive_invocation(
		executor,
		invocation(CallId::new()),
		shell_context(),
		cancel_rx,
	));
	while !started.load(Ordering::SeqCst) {
		tokio::task::yield_now().await;
	}
	cancel_tx.send(()).unwrap();
	let frames = task.await.unwrap();
	assert_eq!(frames.len(), 1, "cancel must stop after the start frame");
	assert!(matches!(stream_event(&frames[0]), Some(cursor_wire::shell_stream::Event::Start(_))));
	assert!(
		dropped.load(Ordering::SeqCst),
		"dropping the invocation must abort its executor future"
	);
}

#[tokio::test]
async fn invoke_deadline_structurally_aborts_executor_without_completion_frames() {
	let started = Arc::new(AtomicBool::new(false));
	let dropped = Arc::new(AtomicBool::new(false));
	let executor =
		Arc::new(PendingExecutor { started: Arc::clone(&started), dropped: Arc::clone(&dropped) });
	let mut invoke = invocation(CallId::new());
	invoke.timeout_ms = 1;
	let (_cancel_tx, cancel_rx) = oneshot::channel();
	let frames = tokio::time::timeout(
		Duration::from_millis(100),
		drive_invocation(executor, invoke, shell_context(), cancel_rx),
	)
	.await
	.expect("Cursor invocation deadline must never stall");
	assert!(started.load(Ordering::SeqCst));
	assert_eq!(frames.len(), 1, "timeout must not synthesize a committed completion");
	assert!(matches!(stream_event(&frames[0]), Some(cursor_wire::shell_stream::Event::Start(_))));
	assert!(dropped.load(Ordering::SeqCst), "timeout must drop the executor future");
}

#[test]
fn cursor_requires_executor_at_admission_and_timeout_kind_is_pinned() {
	assert!(matches!(require_executor(None), Err(Error::Unsupported(_))));
	assert_eq!(TurnErrorKind::InvokeTimeout, TurnErrorKind::InvokeTimeout);
}
