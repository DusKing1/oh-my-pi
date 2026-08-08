//! Live Codex WebSocket continuation, cancellation, and HTTP fallback fixtures.

use std::{
	convert::Infallible,
	future::Future,
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
};

use bytes::Bytes;
use futures::{SinkExt as _, StreamExt as _};
use http::{Request, Response, header};
use http_body_util::{BodyExt as _, Full};
use omp_llm_egress::{
	auth_inject::{CredentialLease, CredentialSource},
	client::Body,
};
use omp_llm_openai::CodexRequestIdentity;
use omp_llm_tower::codex_websocket::{CodexWebSocketEgress, CodexWebSocketRequest};
use serde_json::{Value, json};
use tokio::{net::TcpListener, sync::oneshot};
use tokio_tungstenite::{
	accept_hdr_async,
	tungstenite::{
		Message,
		handshake::server::{Request as HandshakeRequest, Response as HandshakeResponse},
	},
};
use tower::{Service as _, ServiceExt as _, service_fn};

#[derive(Clone)]
struct Credentials;

impl CredentialSource for Credentials {
	type Error = Infallible;

	fn lease(&self, provider: &str) -> Result<Option<CredentialLease>, Self::Error> {
		Ok(Some(CredentialLease::new(provider, 7, 1)))
	}

	fn apply(
		&self,
		_lease: &CredentialLease,
		request: &mut Request<Body>,
	) -> Result<(), Self::Error> {
		let mut value = header::HeaderValue::from_static("Bearer sealed-fixture-token");
		value.set_sensitive(true);
		request.headers_mut().insert(header::AUTHORIZATION, value);
		Ok(())
	}

	fn refresh(
		&self,
		lease: CredentialLease,
	) -> impl Future<Output = Result<CredentialLease, Self::Error>> + Send + 'static {
		async move { Ok(lease) }
	}
}

fn identity(turn: &str) -> CodexRequestIdentity {
	CodexRequestIdentity {
		installation_id: "10000000-0000-4000-8000-000000000001".into(),
		session_id:      "20000000-0000-4000-8000-000000000002".into(),
		thread_id:       "30000000-0000-4000-8000-000000000003".into(),
		window_id:       "40000000-0000-4000-8000-000000000004".into(),
		turn_id:         turn.into(),
		turn_metadata:   format!(r#"{{"turn_id":"{turn}"}}"#).into(),
	}
}

fn request(url: &str, session: &str, turn: &str, input: Value) -> Request<Body> {
	let body = json!({
		"model": "gpt-5.2-codex",
		"input": input,
		"stream": true,
		"client_metadata": {},
	});
	let mut request = Request::post(url)
		.header(header::CONTENT_TYPE, "application/json")
		.header("x-oai-attestation", "fixture-attestation")
		.body(Full::new(Bytes::from(serde_json::to_vec(&body).unwrap())))
		.unwrap();
	request
		.extensions_mut()
		.insert(CredentialLease::new("openai-codex", 7, 1));
	request.extensions_mut().insert(CodexWebSocketRequest {
		session_key:    session.into(),
		identity:       identity(turn),
		account_id:     Some("account-fixture".into()),
		responses_lite: true,
	});
	request
}

async fn send(
	socket: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
	value: Value,
) {
	socket
		.send(Message::Text(value.to_string().into()))
		.await
		.unwrap();
}

async fn completed_turn(
	socket: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
	id: &str,
	text: &str,
) {
	let output = json!({
		"type": "message",
		"id": format!("msg_{id}"),
		"role": "assistant",
		"status": "completed",
		"content": [{"type":"output_text", "text":text}],
	});
	send(socket, json!({"type":"response.created", "response":{"id":id}})).await;
	send(
		socket,
		json!({"type":"response.metadata", "headers":{
			"x-codex-turn-state":format!("state-{id}"),
			"x-models-etag":"models-v1"
		}}),
	)
	.await;
	send(socket, json!({"type":"response.output_item.added", "output_index":0, "item":{
		"type":"message", "id":format!("msg_{id}"), "role":"assistant", "status":"in_progress", "content":[]
	}})).await;
	send(socket, json!({"type":"response.output_text.delta", "output_index":0, "delta":text})).await;
	send(
		socket,
		json!({"type":"response.output_item.done", "output_index":0, "item":output.clone()}),
	)
	.await;
	send(
		socket,
		json!({"type":"response.completed", "response":{
			"id":id, "status":"completed", "output":[output]
		}}),
	)
	.await;
}

#[tokio::test]
async fn codex_websocket_executes_continues_cancels_and_falls_back_before_output() {
	let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
	let address = listener.local_addr().unwrap();
	let (cancelled_tx, cancelled_rx) = oneshot::channel();
	let mut cancelled_tx = Some(cancelled_tx);
	let server = tokio::spawn(async move {
		for attempt in 0..4 {
			let (stream, _) = listener.accept().await.unwrap();
			let mut socket = accept_hdr_async(
				stream,
				move |request: &HandshakeRequest, mut response: HandshakeResponse| {
					assert_eq!(request.headers()[header::AUTHORIZATION], "Bearer sealed-fixture-token");
					assert_eq!(request.headers()["openai-beta"], "responses_websockets=2026-02-06");
					assert_eq!(request.headers()["chatgpt-account-id"], "account-fixture");
					assert_eq!(request.headers()["x-openai-internal-codex-responses-lite"], "true");
					assert_eq!(request.headers()["x-oai-attestation"], "fixture-attestation");
					assert!(request.headers().contains_key("x-codex-window-id"));
					if attempt == 1 {
						assert!(!request.headers().contains_key("x-codex-turn-state"));
						assert_eq!(request.headers()["x-models-etag"], "models-v1");
					}
					if attempt == 2 {
						assert_eq!(request.headers()["x-codex-turn-state"], "state-resp_2");
						assert_eq!(request.headers()["x-models-etag"], "models-v1");
					}
					response
						.headers_mut()
						.insert("x-models-etag", "models-v1".parse().unwrap());
					Ok(response)
				},
			)
			.await
			.unwrap();
			let Message::Text(payload) = socket.next().await.unwrap().unwrap() else {
				panic!("text response.create")
			};
			let frame: Value = serde_json::from_str(&payload).unwrap();
			assert_eq!(frame["type"], "response.create");
			assert_eq!(
				frame["client_metadata"]["ws_request_header_x_openai_internal_codex_responses_lite"],
				"true"
			);
			match attempt {
				0 => {
					assert!(frame.get("previous_response_id").is_none());
					completed_turn(&mut socket, "resp_1", "first").await;
				},
				1 => {
					assert_eq!(frame["previous_response_id"], "resp_1");
					assert_eq!(frame["input"].as_array().unwrap().len(), 1);
					completed_turn(&mut socket, "resp_2", "second").await;
				},
				2 => {
					assert_eq!(frame["previous_response_id"], "resp_2");
					assert_eq!(frame["input"].as_array().unwrap().len(), 1);
					send(
						&mut socket,
						json!({"type":"response.created", "response":{"id":"resp_cancel"}}),
					)
					.await;
					send(
						&mut socket,
						json!({"type":"response.output_item.added", "output_index":0, "item":{
							"type":"message", "id":"msg_cancel", "role":"assistant", "status":"in_progress", "content":[]
						}}),
					)
					.await;
					let closed = socket.next().await;
					assert!(matches!(closed, None | Some(Ok(Message::Close(_))) | Some(Err(_))));
					let _ = cancelled_tx.take().expect("single cancellation").send(());
				},
				3 => {
					socket.close(None).await.unwrap();
				},
				_ => unreachable!(),
			}
		}
	});

	let fallback_count = Arc::new(AtomicUsize::new(0));
	let count = Arc::clone(&fallback_count);
	let http = service_fn(move |request: Request<Body>| {
		let count = Arc::clone(&count);
		async move {
			let body = request.body().clone().into_inner().unwrap_or_default();
			let body: Value = serde_json::from_slice(&body).unwrap();
			assert!(body.get("previous_response_id").is_none());
			count.fetch_add(1, Ordering::SeqCst);
			Ok::<_, Infallible>(Response::builder()
				.header(header::CONTENT_TYPE, "text/event-stream")
				.body(Full::new(Bytes::from_static(b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_http\",\"status\":\"completed\",\"output\":[]}}\n\n")))
				.unwrap())
		}
	});
	let mut egress = CodexWebSocketEgress::new(http, Credentials).with_retry_budget(0);
	let url = format!("http://{address}/backend-api/codex/responses");
	let user =
		json!({"type":"message", "role":"user", "content":[{"type":"input_text", "text":"first"}]});
	let first = egress
		.ready()
		.await
		.unwrap()
		.call(request(&url, "thread", "turn-1", json!([user.clone()])))
		.await
		.unwrap();
	let first_body = first.into_body().collect().await.unwrap().to_bytes();
	assert!(String::from_utf8_lossy(&first_body).contains("first"));
	let prior_output = json!({
		"type":"message", "id":"msg_resp_1", "role":"assistant", "status":"completed",
		"content":[{"type":"output_text", "text":"first"}]
	});
	let next_user =
		json!({"type":"message", "role":"user", "content":[{"type":"input_text", "text":"second"}]});
	let second = egress
		.ready()
		.await
		.unwrap()
		.call(request(
			&url,
			"thread",
			"turn-2",
			json!([user.clone(), prior_output.clone(), next_user.clone()]),
		))
		.await
		.unwrap();
	let second_body = second.into_body().collect().await.unwrap().to_bytes();
	assert!(String::from_utf8_lossy(&second_body).contains("second"));

	let second_output = json!({
		"type":"message", "id":"msg_resp_2", "role":"assistant", "status":"completed",
		"content":[{"type":"output_text", "text":"second"}]
	});
	let tool_output = json!({
		"type":"function_call_output", "call_id":"call_fixture", "output":"cancel now"
	});
	let cancelled = egress
		.ready()
		.await
		.unwrap()
		.call(request(
			&url,
			"thread",
			"turn-cancel",
			json!([user, prior_output, next_user, second_output, tool_output]),
		))
		.await
		.unwrap();
	drop(cancelled);
	cancelled_rx.await.unwrap();

	let fallback = egress
		.ready()
		.await
		.unwrap()
		.call(request(
			&url,
			"fallback-thread",
			"turn-fallback",
			json!([{"type":"message", "role":"user", "content":[{"type":"input_text", "text":"fallback"}]}]),
		))
		.await
		.unwrap();
	let fallback_body = fallback.into_body().collect().await.unwrap().to_bytes();
	assert!(String::from_utf8_lossy(&fallback_body).contains("resp_http"));
	assert_eq!(fallback_count.load(Ordering::SeqCst), 1);
	server.await.unwrap();
}
