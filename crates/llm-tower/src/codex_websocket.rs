//! Replay-safe `ChatGPT` Codex WebSocket execution in front of HTTP egress.
//!
//! The wrapper consumes the typed marker attached by the Codex provider
//! adapter. Every other request passes through unchanged. Credentials remain
//! sealed: the broker source mutates the outbound handshake request in place,
//! exactly as it does for HTTP, and no API returns bearer bytes.

use std::{
	collections::HashMap,
	fmt,
	future::Future,
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
};

use bytes::Bytes;
use futures::{SinkExt as _, StreamExt as _};
use http::{Request, Response, StatusCode, header};
use http_body_util::Full;
use hyper::body::{Body as HttpBody, Frame as HttpFrame};
use omp_core::{Str, fmts};
use omp_llm_egress::{
	auth_inject::{CredentialLease, CredentialSource},
	client::Body,
};
use omp_llm_openai::{
	CodexContinuationState, CodexFallbackAction, CodexFrameDisposition, CodexFrameRouter,
	CodexHeaderContext, CodexReplaySafety, CodexRequestIdentity, CodexWebSocketFailure,
	CodexWireTransport, apply_codex_client_metadata, build_codex_header_plan,
	classify_codex_fallback, classify_codex_websocket_failure, codex_websocket_url,
};
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_tungstenite::{
	MaybeTlsStream, WebSocketStream, connect_async,
	tungstenite::{Message, client::IntoClientRequest as _},
};
use tower::Service;

const MAX_CODEX_SESSIONS: usize = 1_024;

/// Per-request information proving that a catalog row selected Codex WebSocket
/// execution. The marker contains only request identity and broker-released
/// non-secret account metadata.
#[derive(Clone, Debug)]
pub struct CodexWebSocketRequest {
	/// Stable key for continuation state.
	pub session_key:    Str,
	/// Dynamic Codex session/window/turn identity.
	pub identity:       CodexRequestIdentity,
	/// Broker-released `ChatGPT` account id.
	pub account_id:     Option<Str>,
	/// Whether the transformed body is Responses Lite compatible.
	pub responses_lite: bool,
}

/// Egress wrapper that attempts Codex WebSocket execution for typed requests
/// and otherwise delegates to the ordinary HTTP service.
#[derive(Clone)]
pub struct CodexWebSocketEgress<S, C> {
	inner:        S,
	credentials:  C,
	sessions:     Arc<Mutex<HashMap<Str, Arc<Mutex<CodexContinuationState>>>>>,
	retry_budget: u32,
}

impl<S, C> CodexWebSocketEgress<S, C> {
	/// Creates a WebSocket-first egress wrapper with one bounded reconnect.
	#[must_use]
	pub fn new(inner: S, credentials: C) -> Self {
		Self { inner, credentials, sessions: Arc::new(Mutex::new(HashMap::new())), retry_budget: 1 }
	}

	/// Overrides the reconnect budget. HTTP replay remains the terminal safe
	/// fallback and is only possible before observable output.
	#[must_use]
	pub const fn with_retry_budget(mut self, retry_budget: u32) -> Self {
		self.retry_budget = retry_budget;
		self
	}
}

/// Error returned before an HTTP or WebSocket response body is established.
#[derive(Debug)]
pub enum CodexWebSocketEgressError<E, C> {
	/// Ordinary HTTP egress failed.
	Http(E),
	/// Broker rejected credential redemption for the handshake.
	Credential(C),
	/// WebSocket request construction or execution failed without a replay-safe
	/// fallback.
	WebSocket(Str),
}

impl<E: fmt::Display, C: fmt::Display> fmt::Display for CodexWebSocketEgressError<E, C> {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Http(error) => write!(formatter, "HTTP egress failed: {error}"),
			Self::Credential(error) => write!(formatter, "credential redemption failed: {error}"),
			Self::WebSocket(error) => write!(formatter, "Codex WebSocket failed: {error}"),
		}
	}
}

impl<E, C> std::error::Error for CodexWebSocketEgressError<E, C>
where
	E: std::error::Error + 'static,
	C: std::error::Error + 'static,
{
}

impl<B, C, S> Service<Request<Body>> for CodexWebSocketEgress<S, C>
where
	S: Service<Request<Body>, Response = Response<B>> + Clone + Send + 'static,
	S::Future: Send + 'static,
	S::Error: Send + 'static,
	B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
	B::Error: Send + 'static,
	C: CredentialSource + Clone,
{
	type Error = CodexWebSocketEgressError<S::Error, C::Error>;
	type Response = Response<CodexEgressBody<B>>;

	type Future = impl Future<Output = Result<Self::Response, Self::Error>> + Send + 'static;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self
			.inner
			.poll_ready(cx)
			.map_err(CodexWebSocketEgressError::Http)
	}

	fn call(&mut self, request: Request<Body>) -> Self::Future {
		let replacement = self.inner.clone();
		let mut inner = std::mem::replace(&mut self.inner, replacement);
		let credentials = self.credentials.clone();
		let sessions = Arc::clone(&self.sessions);
		let retry_budget = self.retry_budget;
		async move {
			let Some(marker) = request.extensions().get::<CodexWebSocketRequest>().cloned() else {
				return inner
					.call(request)
					.await
					.map(|response| response.map(CodexEgressBody::Http))
					.map_err(CodexWebSocketEgressError::Http);
			};
			let state_key = request.extensions().get::<CredentialLease>().map_or_else(
				|| marker.session_key.clone(),
				|lease| {
					fmts!(
						"{}:{}:{}:{}",
						lease.provider(),
						lease.credential_id(),
						lease.generation(),
						marker.session_key
					)
				},
			);
			let session = {
				let mut sessions = sessions.lock();
				if sessions.len() >= MAX_CODEX_SESSIONS
					&& !sessions.contains_key(&state_key)
					&& let Some(evicted) = sessions.keys().next().cloned()
				{
					sessions.remove(&evicted);
				}
				Arc::clone(
					sessions
						.entry(state_key)
						.or_insert_with(|| Arc::new(Mutex::new(CodexContinuationState::default()))),
				)
			};
			codex_call(inner, credentials, request, marker, session, retry_budget).await
		}
	}
}

async fn codex_call<S, B, C>(
	mut inner: S,
	credentials: C,
	request: Request<Body>,
	marker: CodexWebSocketRequest,
	session: Arc<Mutex<CodexContinuationState>>,
	retry_budget: u32,
) -> Result<Response<CodexEgressBody<B>>, CodexWebSocketEgressError<S::Error, C::Error>>
where
	S: Service<Request<Body>, Response = Response<B>> + Send,
	S::Future: Send,
	B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
	B::Error: Send + 'static,
	C: CredentialSource,
{
	let body = request.body().clone().into_inner().unwrap_or_default();
	let full_request: Value = serde_json::from_slice(body.as_ref())
		.map_err(|error| CodexWebSocketEgressError::WebSocket(Str::from(error.to_string())))?;
	let (wire_request, turn_state, models_etag) = {
		let mut state = session.lock();
		if !is_within_turn_continuation(&full_request) {
			state.start_new_turn();
		}
		let turn_state = state.turn_state().map(Str::new);
		let models_etag = state.models_etag().map(Str::new);
		let mut wire = state
			.response_create(&full_request)
			.map_err(|error| CodexWebSocketEgressError::WebSocket(Str::from(error.to_string())))?;
		apply_codex_client_metadata(
			&mut wire,
			&marker.identity,
			CodexWireTransport::WebSocket,
			marker.responses_lite,
			turn_state.as_deref(),
		)
		.map_err(|error| CodexWebSocketEgressError::WebSocket(Str::from(error.to_string())))?;
		(wire, turn_state, models_etag)
	};
	let mut retries = 0;
	loop {
		let opened = open_until_observable(
			&credentials,
			&request,
			&marker,
			&wire_request,
			turn_state.as_deref(),
			models_etag.as_deref(),
		)
		.await;
		match opened {
			Ok(opened) => {
				return Ok(websocket_response(opened, session, full_request));
			},
			Err(error) => {
				let action = classify_codex_fallback(
					error.failure,
					CodexReplaySafety::default(),
					retries,
					retry_budget,
				);
				match action {
					CodexFallbackAction::ReconnectWebSocket => {
						retries = retries.saturating_add(1);
					},
					CodexFallbackAction::ReplayOverHttp => {
						session.lock().reset();
						return inner
							.call(request)
							.await
							.map(|response| response.map(CodexEgressBody::Http))
							.map_err(CodexWebSocketEgressError::Http);
					},
					CodexFallbackAction::Surface | CodexFallbackAction::Cancelled => {
						return Err(CodexWebSocketEgressError::WebSocket(error.message));
					},
				}
			},
		}
	}
}

type Socket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

struct OpenedSocket {
	socket:            Socket,
	router:            CodexFrameRouter,
	buffered:          Vec<Value>,
	turn_state:        Option<Str>,
	models_etag:       Option<Str>,
	terminal:          Option<TerminalResponse>,
	protocol_terminal: bool,
}

#[derive(Clone)]
struct TerminalResponse {
	id:    Str,
	items: Vec<Value>,
}

struct AttemptFailure {
	failure: CodexWebSocketFailure,
	message: Str,
}

async fn open_until_observable<C: CredentialSource>(
	credentials: &C,
	request: &Request<Body>,
	marker: &CodexWebSocketRequest,
	wire_request: &Value,
	turn_state: Option<&str>,
	models_etag: Option<&str>,
) -> Result<OpenedSocket, AttemptFailure> {
	let handshake =
		handshake_request(request, marker, turn_state, models_etag).map_err(protocol_failure)?;
	let lease = request
		.extensions()
		.get::<CredentialLease>()
		.ok_or_else(|| AttemptFailure {
			failure: CodexWebSocketFailure::Provider,
			message: Str::new_static("Codex WebSocket request has no selected credential lease"),
		})?;
	let url = codex_websocket_url(&request.uri().to_string()).map_err(protocol_failure)?;
	let mut upgrade = url
		.as_str()
		.into_client_request()
		.map_err(|error| AttemptFailure {
			failure: CodexWebSocketFailure::ConnectionFatal,
			message: Str::from(format!("websocket error: {error}")),
		})?;
	for (name, value) in handshake.headers() {
		if !is_upgrade_owned_header(name.as_str()) {
			upgrade.headers_mut().insert(name.clone(), value.clone());
		}
	}
	credentials
		.apply_headers(lease, upgrade.headers_mut())
		.map_err(|error| AttemptFailure {
			failure: CodexWebSocketFailure::Provider,
			message: Str::from(error.to_string()),
		})?;
	let (mut socket, response) = connect_async(upgrade)
		.await
		.map_err(|error| AttemptFailure {
			failure: CodexWebSocketFailure::ConnectionFatal,
			message: Str::from(format!("websocket error: {error}")),
		})?;
	let mut learned_turn_state = response
		.headers()
		.get("x-codex-turn-state")
		.and_then(|value| value.to_str().ok())
		.map(Str::new);
	let mut learned_models_etag = response
		.headers()
		.get("x-models-etag")
		.and_then(|value| value.to_str().ok())
		.map(Str::new);
	let payload = serde_json::to_string(wire_request).map_err(protocol_failure)?;
	socket
		.send(Message::Text(payload.into()))
		.await
		.map_err(|error| AttemptFailure {
			failure: CodexWebSocketFailure::RetryableTransport,
			message: Str::from(format!("websocket send failed: {error}")),
		})?;
	let mut router = CodexFrameRouter::default();
	let mut buffered = Vec::new();
	loop {
		let value = next_json(&mut socket).await?;
		if let Some(error) = in_band_failure(&value) {
			return Err(error);
		}
		let disposition = router.route(&value).map_err(|error| AttemptFailure {
			failure: CodexWebSocketFailure::RetryableTransport,
			message: Str::from(error.to_string()),
		})?;
		if disposition == CodexFrameDisposition::Drop {
			continue;
		}
		update_metadata(&value, &mut learned_turn_state, &mut learned_models_etag);
		let terminal = terminal_response(&value);
		let observable = is_observable(&value) || disposition == CodexFrameDisposition::Terminal;
		buffered.push(value);
		if buffered.len() > 64 {
			return Err(AttemptFailure {
				failure: CodexWebSocketFailure::RetryableTransport,
				message: Str::new_static("websocket message queue exceeded 64 items"),
			});
		}
		if observable {
			return Ok(OpenedSocket {
				socket,
				router,
				buffered,
				turn_state: learned_turn_state,
				models_etag: learned_models_etag,
				terminal,
				protocol_terminal: disposition == CodexFrameDisposition::Terminal,
			});
		}
	}
}

fn handshake_request(
	request: &Request<Body>,
	marker: &CodexWebSocketRequest,
	turn_state: Option<&str>,
	models_etag: Option<&str>,
) -> Result<Request<Body>, Str> {
	let credential =
		omp_llm_openai::CodexCredentialMetadata { account_id: marker.account_id.clone() };
	let plan = build_codex_header_plan(&CodexHeaderContext {
		transport: CodexWireTransport::WebSocket,
		identity: Some(&marker.identity),
		credential: &credential,
		attestation: None,
		turn_state,
		models_etag,
		responses_lite: marker.responses_lite,
	});
	let mut handshake = Request::get(request.uri().clone())
		.body(Full::new(Bytes::new()))
		.map_err(|error| Str::from(error.to_string()))?;
	*handshake.headers_mut() = request.headers().clone();
	for name in [header::ACCEPT, header::CONTENT_TYPE, header::CONTENT_LENGTH] {
		handshake.headers_mut().remove(name);
	}
	for (name, planned) in plan.iter() {
		let name: header::HeaderName = name
			.parse()
			.map_err(|_| Str::new_static("invalid Codex header name"))?;
		let mut value = header::HeaderValue::from_bytes(planned.as_bytes())
			.map_err(|_| Str::new_static("invalid Codex header value"))?;
		value.set_sensitive(planned.is_sensitive());
		handshake.headers_mut().insert(name, value);
	}
	Ok(handshake)
}

fn protocol_failure(error: impl fmt::Display) -> AttemptFailure {
	AttemptFailure {
		failure: CodexWebSocketFailure::Provider,
		message: Str::from(error.to_string()),
	}
}

fn is_upgrade_owned_header(name: &str) -> bool {
	matches!(
		name,
		"host"
			| "connection"
			| "upgrade"
			| "sec-websocket-key"
			| "sec-websocket-version"
			| "sec-websocket-protocol"
			| "sec-websocket-extensions"
	)
}

async fn next_json(socket: &mut Socket) -> Result<Value, AttemptFailure> {
	loop {
		match socket.next().await {
			Some(Ok(Message::Text(text))) => {
				return serde_json::from_str(&text).map_err(|error| AttemptFailure {
					failure: CodexWebSocketFailure::RetryableTransport,
					message: Str::from(format!("json decode failed: {error}")),
				});
			},
			Some(Ok(Message::Binary(bytes))) => {
				return serde_json::from_slice(&bytes).map_err(|error| AttemptFailure {
					failure: CodexWebSocketFailure::RetryableTransport,
					message: Str::from(format!("json decode failed: {error}")),
				});
			},
			Some(Ok(Message::Ping(payload))) => {
				socket
					.send(Message::Pong(payload))
					.await
					.map_err(|error| AttemptFailure {
						failure: CodexWebSocketFailure::RetryableTransport,
						message: Str::from(format!("websocket ping failed: {error}")),
					})?;
			},
			Some(Ok(Message::Pong(_) | Message::Frame(_))) => {},
			Some(Ok(Message::Close(frame))) => {
				return Err(AttemptFailure {
					failure: CodexWebSocketFailure::RetryableTransport,
					message: Str::from(format!(
						"websocket closed before response completion: {frame:?}"
					)),
				});
			},
			Some(Err(error)) => {
				let message = format!("websocket error: {error}");
				return Err(AttemptFailure {
					failure: classify_codex_websocket_failure(None, &message, false),
					message: Str::from(message),
				});
			},
			None => {
				return Err(AttemptFailure {
					failure: CodexWebSocketFailure::RetryableTransport,
					message: Str::new_static("websocket closed before response completion"),
				});
			},
		}
	}
}

fn in_band_failure(value: &Value) -> Option<AttemptFailure> {
	if value.get("type").and_then(Value::as_str) != Some("error") {
		return None;
	}
	let nested = value.get("error");
	let code = value
		.get("code")
		.or_else(|| nested.and_then(|error| error.get("code")))
		.and_then(Value::as_str);
	let message = value
		.get("message")
		.or_else(|| nested.and_then(|error| error.get("message")))
		.and_then(Value::as_str)
		.unwrap_or("Codex WebSocket provider error");
	let stale_continuation =
		matches!(code, Some("previous_response_not_found" | "codex_previous_response_stale"));
	Some(AttemptFailure {
		failure: if stale_continuation {
			CodexWebSocketFailure::ConnectionFatal
		} else {
			classify_codex_websocket_failure(code, message, false)
		},
		message: Str::new(message),
	})
}

fn is_observable(value: &Value) -> bool {
	matches!(
		value
			.get("type")
			.and_then(Value::as_str)
			.unwrap_or_default(),
		"response.output_item.added"
			| "response.output_text.delta"
			| "response.refusal.delta"
			| "response.reasoning_summary_text.delta"
			| "response.reasoning_text.delta"
			| "response.function_call_arguments.delta"
			| "response.custom_tool_call_input.delta"
			| "response.image_generation_call.partial_image"
	)
}

fn update_metadata(value: &Value, turn_state: &mut Option<Str>, models_etag: &mut Option<Str>) {
	if value.get("type").and_then(Value::as_str) != Some("response.metadata") {
		return;
	}
	let Some(headers) = value.get("headers").and_then(Value::as_object) else {
		return;
	};
	if let Some(value) = headers.get("x-codex-turn-state").and_then(Value::as_str) {
		*turn_state = Some(Str::new(value));
	}
	if let Some(value) = headers.get("x-models-etag").and_then(Value::as_str) {
		*models_etag = Some(Str::new(value));
	}
}

fn is_within_turn_continuation(request: &Value) -> bool {
	let Some(input) = request.get("input").and_then(Value::as_array) else {
		return false;
	};
	for item in input.iter().rev() {
		if matches!(
			item.get("type").and_then(Value::as_str),
			Some("function_call_output" | "custom_tool_call_output" | "computer_call_output")
		) {
			continue;
		}
		return item.get("role").and_then(Value::as_str) == Some("assistant");
	}
	false
}

fn terminal_response(value: &Value) -> Option<TerminalResponse> {
	if !matches!(
		value.get("type").and_then(Value::as_str),
		Some("response.completed" | "response.done")
	) {
		return None;
	}
	let response = value.get("response")?.as_object()?;
	let id = Str::new(response.get("id")?.as_str()?);
	let items = response
		.get("output")
		.and_then(Value::as_array)
		.cloned()
		.unwrap_or_default();
	Some(TerminalResponse { id, items })
}

fn websocket_response<B: HttpBody<Data = Bytes> + Unpin + 'static>(
	opened: OpenedSocket,
	session: Arc<Mutex<CodexContinuationState>>,
	full_request: Value,
) -> Response<CodexEgressBody<B>> {
	let (tx, rx) = mpsc::channel(64);
	for value in &opened.buffered {
		let _ = tx.try_send(Ok(sse_frame(value)));
	}
	if let Some(terminal) = opened.terminal {
		commit_session(
			&session,
			full_request,
			terminal,
			opened.turn_state.as_deref(),
			opened.models_etag.as_deref(),
		);
		drop(tx);
	} else if opened.protocol_terminal {
		drop(tx);
	} else {
		tokio::spawn(pump_socket(
			opened.socket,
			opened.router,
			tx,
			session,
			full_request,
			opened.turn_state,
			opened.models_etag,
		));
	}
	Response::builder()
		.status(StatusCode::OK)
		.header(header::CONTENT_TYPE, "text/event-stream")
		.body(CodexEgressBody::WebSocket(CodexChannelBody { receiver: rx }))
		.expect("static Codex WebSocket response is valid")
}

async fn pump_socket(
	mut socket: Socket,
	mut router: CodexFrameRouter,
	tx: mpsc::Sender<Result<Bytes, CodexBodyError>>,
	session: Arc<Mutex<CodexContinuationState>>,
	full_request: Value,
	mut turn_state: Option<Str>,
	mut models_etag: Option<Str>,
) {
	loop {
		tokio::select! {
			biased;
			() = tx.closed() => {
				router.cancel();
				session.lock().reset();
				let _ = socket.close(None).await;
				return;
			},
			result = next_json(&mut socket) => {
				let value = match result {
					Ok(value) => value,
					Err(error) => {
						let _ = tx.send(Err(CodexBodyError(error.message))).await;
						return;
					},
				};
				if let Some(error) = in_band_failure(&value) {
					let _ = tx.send(Err(CodexBodyError(error.message))).await;
					return;
				}
				let disposition = match router.route(&value) {
					Ok(disposition) => disposition,
					Err(error) => {
						let _ = tx.send(Err(CodexBodyError(Str::from(error.to_string())))).await;
						return;
					},
				};
				if disposition == CodexFrameDisposition::Drop {
					continue;
				}
				update_metadata(&value, &mut turn_state, &mut models_etag);
				let terminal = terminal_response(&value);
				if tx.send(Ok(sse_frame(&value))).await.is_err() {
					router.cancel();
					session.lock().reset();
					let _ = socket.close(None).await;
					return;
				}
				if let Some(terminal) = terminal {
					commit_session(
						&session,
						full_request,
						terminal,
						turn_state.as_deref(),
						models_etag.as_deref(),
					);
					return;
				}
			},
		}
	}
}

fn commit_session(
	session: &Arc<Mutex<CodexContinuationState>>,
	full_request: Value,
	terminal: TerminalResponse,
	turn_state: Option<&str>,
	models_etag: Option<&str>,
) {
	let mut state = session.lock();
	state.commit(full_request, terminal.id, terminal.items);
	state.update_metadata(turn_state, models_etag);
}

fn sse_frame(value: &Value) -> Bytes {
	let mut bytes =
		Vec::with_capacity(serde_json::to_string(value).map_or(16, |value| value.len() + 8));
	bytes.extend_from_slice(b"data: ");
	serde_json::to_writer(&mut bytes, value).expect("JSON value serializes");
	bytes.extend_from_slice(b"\n\n");
	Bytes::from(bytes)
}

/// Response body returned by [`CodexWebSocketEgress`].
pub enum CodexEgressBody<B> {
	/// Ordinary HTTP body.
	Http(B),
	/// WebSocket events projected as SSE frames for the existing Codex decoder.
	WebSocket(CodexChannelBody),
}

impl<B> HttpBody for CodexEgressBody<B>
where
	B: HttpBody<Data = Bytes> + Unpin,
{
	type Data = Bytes;
	type Error = CodexEgressBodyError<B::Error>;

	fn poll_frame(
		mut self: Pin<&mut Self>,
		cx: &mut Context<'_>,
	) -> Poll<Option<Result<HttpFrame<Self::Data>, Self::Error>>> {
		match self.as_mut().get_mut() {
			Self::Http(body) => Pin::new(body)
				.poll_frame(cx)
				.map(|frame| frame.map(|result| result.map_err(CodexEgressBodyError::Http))),
			Self::WebSocket(body) => Pin::new(body)
				.poll_frame(cx)
				.map(|frame| frame.map(|result| result.map_err(CodexEgressBodyError::WebSocket))),
		}
	}

	fn is_end_stream(&self) -> bool {
		match self {
			Self::Http(body) => body.is_end_stream(),
			Self::WebSocket(body) => body.receiver.is_closed() && body.receiver.is_empty(),
		}
	}
}

/// Error produced while reading a Codex egress response body.
#[derive(Debug)]
pub enum CodexEgressBodyError<E> {
	/// HTTP response body error.
	Http(E),
	/// WebSocket response body error.
	WebSocket(CodexBodyError),
}

impl<E: fmt::Display> fmt::Display for CodexEgressBodyError<E> {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Http(error) => write!(formatter, "{error}"),
			Self::WebSocket(error) => write!(formatter, "{error}"),
		}
	}
}

impl<E> std::error::Error for CodexEgressBodyError<E> where E: std::error::Error + 'static {}

/// Bounded WebSocket frame receiver.
pub struct CodexChannelBody {
	receiver: mpsc::Receiver<Result<Bytes, CodexBodyError>>,
}

impl HttpBody for CodexChannelBody {
	type Data = Bytes;
	type Error = CodexBodyError;

	fn poll_frame(
		mut self: Pin<&mut Self>,
		cx: &mut Context<'_>,
	) -> Poll<Option<Result<HttpFrame<Self::Data>, Self::Error>>> {
		self
			.receiver
			.poll_recv(cx)
			.map(|item| item.map(|result| result.map(HttpFrame::data)))
	}
}

/// WebSocket body failure, intentionally free of request and credential data.
#[derive(Clone, Debug)]
pub struct CodexBodyError(Str);

impl fmt::Display for CodexBodyError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(&self.0)
	}
}

impl std::error::Error for CodexBodyError {}
