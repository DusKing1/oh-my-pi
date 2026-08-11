//! GitLab Duo WebSocket authentication, tool, and reconnect fixtures.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt as _, StreamExt as _};
use http::{HeaderMap, HeaderValue, header};
use omp_core::Str;
use omp_llm_gitlab::{GitLabDuoChat, WorkflowAuth, WorkflowConfig};
use omp_llm_types::{
	Chat, ChatRequest, Error, Executor, Invoke, InvokeComplete, InvokeInput, Item, ItemKind,
	Message, Part, Props, Role, Thread, ToolDef, ToolResult, TurnEvent,
};
use tokio::{net::TcpListener, sync::oneshot};
use tokio_tungstenite::{accept_async, accept_hdr_async, tungstenite::Message as WsMessage};

struct FixtureAuth;

#[async_trait]
impl WorkflowAuth for FixtureAuth {
	async fn apply(&self, headers: &mut HeaderMap) -> Result<(), Error> {
		headers.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer fixture-lease"));
		Ok(())
	}
}

struct FixtureExecutor;

#[async_trait]
impl Executor for FixtureExecutor {
	async fn invoke(
		&self,
		invocation: Invoke,
		_inputs: flume::Sender<InvokeInput>,
	) -> InvokeComplete {
		let call = invocation
			.tool_call
			.expect("workflow action carries canonical tool call");
		InvokeComplete::builder()
			.invocation_id(invocation.invocation_id)
			.tool_result(
				ToolResult::builder()
					.call_id(call.id)
					.name(call.name)
					.parts(vec![Part::Text(Str::new_static("tool-ok"))])
					.is_error(false)
					.build(),
			)
			.vendor(Bytes::new())
			.props(Props::default())
			.build()
	}
}

fn request() -> ChatRequest {
	let prior = Item::builder()
		.seq(0)
		.kind(ItemKind::Message(
			Message::builder()
				.role(Role::Assistant)
				.parts(vec![Part::Text(Str::new_static("ready"))])
				.build(),
		))
		.props(Props::default())
		.build();
	let current = Item::builder()
		.seq(0)
		.kind(ItemKind::Message(
			Message::builder()
				.role(Role::User)
				.parts(vec![Part::Text(Str::new_static("fix the file"))])
				.build(),
		))
		.props(Props::default())
		.build();
	ChatRequest::builder()
		.model(Str::new_static("duo-agent"))
		.thread(Thread::builder().items(vec![prior, current]).build())
		.tools(vec![
			ToolDef::builder()
				.name(Str::new_static("edit"))
				.description(Str::new_static("edit a file"))
				.schema_json(Bytes::from_static(br#"{"type":"object"}"#))
				.build(),
		])
		.provider_options(Props::default())
		.build()
}

#[tokio::test]
async fn authenticated_turn_maps_tool_result_and_resumes_after_disconnect() {
	let listener = TcpListener::bind("127.0.0.1:0")
		.await
		.expect("bind WebSocket fixture");
	let address = listener.local_addr().expect("fixture address");
	let server = tokio::spawn(async move {
		let (first, _) = listener.accept().await.expect("first workflow connection");
		let mut first = accept_hdr_async(
			first,
			#[allow(
				clippy::result_large_err,
				reason = "tungstenite's handshake callback API requires this response error type"
			)]
			|request: &tokio_tungstenite::tungstenite::handshake::server::Request,
			 response: tokio_tungstenite::tungstenite::handshake::server::Response| {
				assert_eq!(request.headers()[header::AUTHORIZATION], "Bearer fixture-lease");
				assert_eq!(request.headers()["x-gitlab-client-type"], "node-websocket");
				assert_eq!(
					request.headers()[header::ORIGIN]
						.to_str()
						.expect("origin header"),
					format!("http://{address}")
				);
				Ok(response)
			},
		)
		.await
		.expect("first handshake");
		let start = first
			.next()
			.await
			.expect("start frame")
			.expect("valid start")
			.into_text()
			.expect("text");
		let start: serde_json::Value = serde_json::from_str(&start).expect("start JSON");
		assert_eq!(
			start
				.pointer("/startRequest/workflowID")
				.and_then(serde_json::Value::as_str),
			Some("workflow-7")
		);
		assert!(
			start
				.pointer("/startRequest/goal")
				.and_then(serde_json::Value::as_str)
				.expect("goal")
				.contains("<|im_start|>user")
		);
		first
			.send(WsMessage::Text(
				serde_json::json!({"eventID":"1","text":"working "})
					.to_string()
					.into(),
			))
			.await
			.expect("text checkpoint");
		first.send(WsMessage::Text(serde_json::json!({
            "eventID":"2",
            "action": {"name":"runMCPTool","requestID":"req-9","args":{"name":"edit","arguments":{"path":"a.rs"}}}
        }).to_string().into())).await.expect("tool action");
		let response = first
			.next()
			.await
			.expect("action response")
			.expect("valid response")
			.into_text()
			.expect("response text");
		let response: serde_json::Value = serde_json::from_str(&response).expect("response JSON");
		assert_eq!(
			response
				.pointer("/actionResponse/requestID")
				.and_then(serde_json::Value::as_str),
			Some("req-9")
		);
		assert_eq!(
			response
				.pointer("/actionResponse/plainTextResponse/response")
				.and_then(serde_json::Value::as_str),
			Some("tool-ok")
		);
		drop(first);

		let (second, _) = listener.accept().await.expect("resume connection");
		let mut second = accept_async(second).await.expect("resume handshake");
		let resume = second
			.next()
			.await
			.expect("resume frame")
			.expect("valid resume")
			.into_text()
			.expect("resume text");
		let resume: serde_json::Value = serde_json::from_str(&resume).expect("resume JSON");
		assert_eq!(
			resume
				.pointer("/resumeRequest/workflowID")
				.and_then(serde_json::Value::as_str),
			Some("workflow-7")
		);
		assert_eq!(
			resume
				.pointer("/resumeRequest/lastEventID")
				.and_then(serde_json::Value::as_str),
			Some("2")
		);
		second
			.send(WsMessage::Text(
				serde_json::json!({
					 "eventID":"3",
					 "text":"working done",
					 "agent_context_usage":{"Chat Agent":{"total_tokens":321,"max_tokens":4096}},
					 "status":"FINISHED"
				})
				.to_string()
				.into(),
			))
			.await
			.expect("terminal frame");
	});

	let config = WorkflowConfig::new(format!("ws://{address}"), "workflow-7", "session-a");
	let chat = GitLabDuoChat::new(config, Arc::new(FixtureAuth));
	let events: Vec<_> = chat
		.turn(request(), Some(Arc::new(FixtureExecutor)))
		.await
		.expect("admit turn")
		.collect()
		.await;
	server.await.expect("fixture task");

	assert_eq!(
		events
			.iter()
			.filter(|event| matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)))
			.count(),
		1,
		"one terminal event"
	);
	assert!(
		events
			.iter()
			.any(|event| matches!(event, TurnEvent::Attempt { .. })),
		"disconnect resumes explicitly"
	);
	assert!(
		events.iter().any(
			|event| matches!(event, TurnEvent::Invoke(invoke) if invoke.invocation_id == "req-9")
		)
	);
	let outcome = events
		.iter()
		.find_map(|event| {
			if let TurnEvent::Outcome(outcome) = event {
				Some(outcome)
			} else {
				None
			}
		})
		.expect("successful outcome");
	assert_eq!(outcome.usage.as_ref().map(|usage| usage.input_tokens), Some(321));
	assert!(outcome.output.iter().any(|item| matches!(&item.kind, ItemKind::ToolResult(result) if result.parts == vec![Part::Text(Str::new_static("tool-ok"))])));
}

#[tokio::test]
async fn malformed_frame_is_classified_once_and_stream_drop_closes_socket() {
	let listener = TcpListener::bind("127.0.0.1:0")
		.await
		.expect("bind malformed fixture");
	let address = listener.local_addr().expect("fixture address");
	let server = tokio::spawn(async move {
		let (stream, _) = listener.accept().await.expect("malformed connection");
		let mut socket = accept_async(stream).await.expect("malformed handshake");
		let _ = socket
			.next()
			.await
			.expect("start frame")
			.expect("valid start");
		socket
			.send(WsMessage::Text("{not-json".into()))
			.await
			.expect("malformed frame");
		let closed = socket.next().await;
		assert!(
			matches!(closed, Some(Ok(WsMessage::Close(_)) | Err(_)) | None),
			"client closes after terminal classification"
		);
	});
	let config = WorkflowConfig::new(format!("ws://{address}"), "workflow-bad", "session-bad");
	let chat = GitLabDuoChat::new(config, Arc::new(FixtureAuth));
	let events: Vec<_> = chat
		.turn(request(), Some(Arc::new(FixtureExecutor)))
		.await
		.expect("admit turn")
		.collect()
		.await;
	server.await.expect("fixture task");
	assert_eq!(
		events
			.iter()
			.filter(|event| matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)))
			.count(),
		1
	);
	assert!(events.iter().any(|event| matches!(event, TurnEvent::Error(error) if error.detail.contains("malformed GitLab Duo Workflow frame"))));
	assert!(events.iter().any(|event| matches!(
		 event,
		 TurnEvent::Error(error)
			  if error.diagnostics.iter().any(|diagnostic| diagnostic.code == "malformed_frame")
	)));
}

#[tokio::test]
async fn dropping_turn_stream_closes_the_socket_without_fallback() {
	let listener = TcpListener::bind("127.0.0.1:0")
		.await
		.expect("bind cancellation fixture");
	let address = listener.local_addr().expect("fixture address");
	let (observed_tx, observed_rx) = oneshot::channel();
	let server = tokio::spawn(async move {
		let (stream, _) = listener.accept().await.expect("cancel connection");
		let mut socket = accept_async(stream).await.expect("cancel handshake");
		let _ = socket
			.next()
			.await
			.expect("start frame")
			.expect("valid start");
		socket
			.send(WsMessage::Text(
				serde_json::json!({"eventID":"cancel-1","text":"started"})
					.to_string()
					.into(),
			))
			.await
			.expect("started frame");
		let closed = socket.next().await;
		let _ = observed_tx.send(matches!(closed, Some(Ok(WsMessage::Close(_)) | Err(_)) | None));
	});
	let config = WorkflowConfig::new(format!("ws://{address}"), "workflow-cancel", "session-cancel");
	let chat = GitLabDuoChat::new(config, Arc::new(FixtureAuth));
	let mut stream = chat
		.turn(request(), Some(Arc::new(FixtureExecutor)))
		.await
		.expect("admit turn");
	assert!(matches!(stream.next().await, Some(TurnEvent::Accepted { .. })));
	assert!(matches!(stream.next().await, Some(TurnEvent::PartStart { .. })));
	drop(stream);
	assert!(
		tokio::time::timeout(std::time::Duration::from_secs(2), observed_rx)
			.await
			.expect("socket cancellation deadline")
			.expect("cancellation observation")
	);
	server.await.expect("fixture task");
}
