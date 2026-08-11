//! Cursor's pinned Connect/protobuf codec and in-turn execution bridge.
//!
//! Cursor is unusual among the supported transports: the response stream
//! remains writable while the provider asks the client to execute tools.  The
//! helpers in this module keep protobuf ownership at this edge; an [`Executor`]
//! sees canonical invocation values only.

use std::{
	collections::BTreeMap, convert::Infallible, error::Error as StdError, sync::Arc, time::Duration,
};

use bytes::{Buf, BufMut, Bytes, BytesMut};
use futures::{StreamExt as _, stream::BoxStream};
use http::{
	Request, Uri,
	header::{self, HeaderMap},
};
use http_body_util::{BodyExt, StreamBody};
use hyper::body::Frame as BodyFrame;
use hyper_rustls::HttpsConnectorBuilder;
use hyper_util::rt::TokioExecutor;
use omp_core::Str;
use omp_llm_types::{
	Accuracy, CallId, Chat, ChatOutcome, ChatRequest, Error, ExecOutcome, ExecStatus, Executor,
	Invoke, InvokeChannel, InvokeComplete, InvokeInput, InvokePayload, ItemKind, Part, Props, Role,
	StopReason, StreamPartKind, ToolCall, ToolResult, TurnError, TurnErrorKind, TurnEvent,
	Unsupported, UnsupportedAction, UnsupportedSink, Usage,
};
use prost::Message as _;
use smallvec::{SmallVec, smallvec};
use tokio::{sync::oneshot, task::JoinHandle};
use tower::Service as _;

pub mod discovery;
#[allow(
	clippy::large_enum_variant,
	reason = "generated prost oneof layout follows Cursor's pinned external schema"
)]
pub mod wire;

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const DEFAULT_INVOKE_TIMEOUT_MS: u64 = 120_000;
const CONNECT_END_STREAM: u8 = 0x02;
const CURSOR_CLIENT_VERSION: &str = "cli-2026.07.23-e383d2b";

fn cursor_request_headers(
	request: &ChatRequest,
	auth_headers: HeaderMap,
) -> Result<HeaderMap, Error> {
	let mut headers = HeaderMap::new();
	headers.insert(header::CONTENT_TYPE, http::HeaderValue::from_static("application/connect+proto"));
	headers.insert(
		http::HeaderName::from_static("connect-protocol-version"),
		http::HeaderValue::from_static("1"),
	);
	headers.insert(
		http::HeaderName::from_static("x-ghost-mode"),
		http::HeaderValue::from_static("true"),
	);
	headers.insert(
		http::HeaderName::from_static("x-cursor-client-version"),
		http::HeaderValue::from_static(CURSOR_CLIENT_VERSION),
	);
	headers.insert(
		http::HeaderName::from_static("x-cursor-client-type"),
		http::HeaderValue::from_static("cli"),
	);

	if let Some(policy) = request.model_policy.as_deref() {
		for (raw_name, raw_value) in policy.headers.iter() {
			if raw_name.starts_with(':') {
				continue;
			}
			let name: http::HeaderName = raw_name.parse().map_err(|_| {
				Error::Transport(omp_core::fmts!("invalid Cursor caller header name: {raw_name}"))
			})?;
			let field = name.as_str();
			if matches!(
				field,
				"connection"
					| "keep-alive"
					| "proxy-connection"
					| "transfer-encoding"
					| "upgrade"
					| "http2-settings"
					| "host"
					| "content-length"
					| "content-type"
					| "connect-protocol-version"
					| "x-ghost-mode"
			) || field.starts_with("x-cursor-")
				|| auth_headers.contains_key(&name)
			{
				continue;
			}
			let value: http::HeaderValue = raw_value.parse().map_err(|_| {
				Error::Transport(omp_core::fmts!(
					"invalid value for Cursor caller header {raw_name}"
				))
			})?;
			if field == "te" && !value.as_bytes().eq_ignore_ascii_case(b"trailers") {
				continue;
			}
			headers.insert(name, value);
		}
	}
	headers.extend(auth_headers);
	Ok(headers)
}

/// Mutable decoder state exposed as the pin-test entry point.
///
/// This is deliberately transport-independent so recorded frames can be
/// replayed without a connection.
#[derive(Default)]
pub struct CursorDecodeState {
	calls:             BTreeMap<Str, CallId>,
	open_part:         Option<(u32, StreamPartKind)>,
	open_tool_call_id: Str,
	next_part:         u32,
	model:             Str,
	unsupported:       Vec<Unsupported>,
	output_tokens:     u64,
	saw_usage:         bool,
	saw_tool_call:     bool,
}

/// Applies a sealed Cursor credential directly to outbound request headers.
///
/// Implementations own credential leasing and redemption. The interface is
/// intentionally mutation-only: callers can authenticate a request but cannot
/// recover the credential bytes.
pub trait CursorAuth: Send + Sync + 'static {
	/// Authentication failure reported without exposing credential material.
	type Error: StdError + Send + Sync + 'static;

	/// Authenticates one Cursor request in place.
	fn apply(&self, headers: &mut HeaderMap) -> Result<(), Self::Error>;
}

/// Cursor's bespoke Connect transport.
///
/// The frozen schema declares `AgentService.Run` as unary even though the live
/// endpoint is a bidirectional Connect stream.  This transport intentionally
/// implements that observed framing without changing the drift pin: invocation
/// responses are written to the request body while server frames are decoded.
#[derive(Clone)]
pub struct CursorChat<A> {
	base_url: Str,
	auth:     A,
}

impl<A> CursorChat<A> {
	/// Constructs a Cursor chat transport with a sealed authentication applier.
	#[must_use]
	pub fn new(base_url: impl Into<Str>, auth: A) -> Self {
		Self { base_url: base_url.into(), auth }
	}
}

#[async_trait::async_trait]
impl<A> Chat for CursorChat<A>
where
	A: CursorAuth,
{
	async fn turn(
		&self,
		request: ChatRequest,
		executor: Option<Arc<dyn Executor>>,
	) -> Result<BoxStream<'static, TurnEvent>, Error> {
		let executor = require_executor(executor)?;
		let (run, _unsupported) = assemble_request(&request)?;
		let uri: Uri = format!("{}/agent.v1.AgentService/Run", self.base_url.trim_end_matches('/'))
			.parse()
			.map_err(|error| {
				Error::Transport(Str::from(format!("invalid Cursor endpoint: {error}")))
			})?;
		let mut auth_headers = HeaderMap::new();
		self
			.auth
			.apply(&mut auth_headers)
			.map_err(|_| Error::Transport(Str::new_static("Cursor authentication failed")))?;

		let stream = async_stream::try_stream! {
			yield TurnEvent::Accepted { replay: false };

			let (client_tx, client_rx) = flume::bounded::<Bytes>(32);
			client_tx.send_async(connect_frame(&run)).await
				.map_err(|_| Error::Transport(Str::new_static("Cursor request stream closed")))?;
			let request_stream = futures::stream::unfold(client_rx, |receiver| async move {
				receiver.recv_async().await.ok().map(|bytes| {
					(Ok::<BodyFrame<Bytes>, Infallible>(BodyFrame::data(bytes)), receiver)
				})
			});
			let body = BodyExt::boxed(StreamBody::new(request_stream));
			let mut outbound = Request::post(uri.clone())
				.body(body)
				.map_err(|error| Error::Transport(Str::from(error.to_string())))?;
			*outbound.headers_mut() = cursor_request_headers(&request, auth_headers)?;

			omp_llm_egress::client::ensure_crypto_provider();
			let mut connector = HttpsConnectorBuilder::new()
				.with_webpki_roots()
				.https_or_http()
				.enable_http2()
				.build();
			let io = connector.call(uri).await
				.map_err(|error| Error::Transport(Str::from(error.to_string())))?;
			let (mut sender, connection) = hyper::client::conn::http2::handshake(
				TokioExecutor::new(),
				io,
			)
			.await
			.map_err(|error| Error::Transport(Str::from(error.to_string())))?;
			let _connection = AbortOnDrop(tokio::spawn(async move {
				let _ = connection.await;
			}));
			let mut response = sender.send_request(outbound).await
				.map_err(|error| Error::Transport(Str::from(error.to_string())))?;
			if !response.status().is_success() {
				Err(Error::Transport(Str::from(format!(
					"Cursor Connect returned HTTP {}",
					response.status()
				))))?;
			}

			let mut decoder = ConnectDecoder::default();
			let mut state = CursorDecodeState {
				model: request.model.clone(),
				unsupported: _unsupported,
				..CursorDecodeState::default()
			};
			let mut cancels = BTreeMap::<u32, (Str, oneshot::Sender<()>)>::new();
			let mut invocations = Vec::<AbortOnDrop>::new();
			let (terminal_tx, terminal_rx) = flume::bounded::<TurnError>(1);
			loop {
				let next_frame = tokio::select! {
					timeout = terminal_rx.recv_async() => Err(timeout.expect("timeout sender retained")),
					frame = response.body_mut().frame() => Ok(frame),
				};
				let frame = match next_frame {
					Err(error) => {
						yield TurnEvent::Error(error);
						break;
					}
					Ok(Some(frame)) => frame
						.map_err(|error| Error::Transport(Str::from(error.to_string())))?,
					Ok(None) => break,
				};
				let Ok(data) = frame.into_data() else { continue };
				for payload in decoder.push_bytes(data)? {
					let server = wire::AgentServerMessage::decode(payload).map_err(|error| {
						Error::Transport(Str::from(error.to_string()))
					})?;
					match server.message {
						Some(wire::agent_server_message::Message::ExecServerMessage(exec)) => {
							let numeric_id = exec.id;
							let (invocation, context) = invoke_and_context_from_exec(exec, &mut state)?;
							let invocation_id = invocation.invocation_id.clone();
							yield TurnEvent::Invoke(invocation.clone());
							let (cancel_tx, cancel_rx) = oneshot::channel();
							cancels.insert(numeric_id, (invocation_id, cancel_tx));
							let executor = executor.clone();
							let client_tx = client_tx.clone();
							let terminal_tx = terminal_tx.clone();
							invocations.push(AbortOnDrop(tokio::spawn(async move {
								drive_invocation_to(
									executor,
									invocation,
									context,
									cancel_rx,
									client_tx,
									terminal_tx,
								)
								.await;
							})));
						}
						Some(wire::agent_server_message::Message::ExecServerControlMessage(control)) => {
							if let Some(wire::exec_server_control_message::Message::Abort(abort)) = control.message {
								let invocation_id = if let Some((invocation_id, cancel)) = cancels.remove(&abort.id) {
									let _ = cancel.send(());
									invocation_id
								} else {
									Str::from(abort.id.to_string())
								};
								yield TurnEvent::InvokeCancel { invocation_id };
							}
						}
						other => {
							let events = decode_server_message(
								wire::AgentServerMessage { message: other },
								&mut state,
							)?;
							for event in events {
								yield event;
							}
						}
					}
				}
			}
			drop(invocations);
		};
		Ok(Box::pin(
			stream.map(|event: Result<TurnEvent, Error>| event.unwrap_or_else(cursor_turn_error)),
		))
	}
}
fn cursor_turn_error(error: Error) -> TurnEvent {
	let (kind, detail, unsupported) = match error {
		Error::Unsupported(unsupported) => (
			TurnErrorKind::Unsupported,
			Str::new_static("Cursor request became unsupported"),
			unsupported,
		),
		Error::Provider(detail) | Error::Transport(detail) => {
			(TurnErrorKind::Upstream, detail, Vec::new())
		},
		_ => (TurnErrorKind::Upstream, Str::new_static("Cursor request failed"), Vec::new()),
	};
	TurnEvent::Error(
		TurnError::builder()
			.kind(kind)
			.detail(detail)
			.unsupported(unsupported)
			.retry_after_ms(0)
			.build(),
	)
}

struct AbortOnDrop(JoinHandle<()>);

impl Drop for AbortOnDrop {
	fn drop(&mut self) {
		self.0.abort();
	}
}
/// Non-secret model metadata returned by Cursor's `GetUsableModels` RPC.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveredModel {
	/// Provider-local model id.
	pub id:        Str,
	/// Cursor display label.
	pub name:      Str,
	/// Whether Cursor exposes thinking controls.
	pub reasoning: bool,
	/// Whether this entry advertises Cursor max mode.
	pub max_mode:  bool,
}

/// Encodes the real Cursor `GetUsableModels` protobuf request.
#[must_use]
pub fn model_discovery_request() -> Bytes {
	Bytes::from(wire::GetUsableModelsRequest { custom_model_ids: Vec::new() }.encode_to_vec())
}

/// Decodes a raw or Connect-framed Cursor `GetUsableModels` response.
///
/// # Errors
///
/// Returns a transport error for malformed protobuf or frame boundaries.
pub fn decode_model_discovery(payload: &[u8]) -> Result<Vec<DiscoveredModel>, Error> {
	let message = first_connect_message(payload)?;
	let response = wire::GetUsableModelsResponse::decode(message)
		.map_err(|error| Error::Transport(Str::from(error.to_string())))?;
	let mut models = BTreeMap::new();
	for model in response.models {
		let reasoning = model.thinking_details.is_some();
		let max_mode = model.max_mode.unwrap_or(false);
		let id = model.model_id.trim();
		if id.is_empty() {
			continue;
		}
		let name = [
			model.display_name.as_str(),
			model.display_name_short.as_str(),
			model.display_model_id.as_str(),
			id,
		]
		.into_iter()
		.find(|value| !value.trim().is_empty())
		.unwrap_or(id);
		models.insert(Str::from(id), DiscoveredModel {
			id: Str::from(id),
			name: Str::from(name),
			reasoning,
			max_mode,
		});
	}
	Ok(models.into_values().collect())
}

fn first_connect_message(payload: &[u8]) -> Result<&[u8], Error> {
	if payload.len() < 5 {
		return Ok(payload);
	}
	let length = u32::from_be_bytes([payload[1], payload[2], payload[3], payload[4]]) as usize;
	let Some(end) = 5usize.checked_add(length) else {
		return Err(Error::Transport(Str::new_static("Cursor discovery frame length overflow")));
	};
	if end > payload.len() {
		return Ok(payload);
	}
	if payload[0] & CONNECT_END_STREAM != 0 {
		return Err(Error::Transport(Str::new_static(
			"Cursor discovery returned only an end-stream frame",
		)));
	}
	Ok(&payload[5..end])
}

/// Rejects a Cursor turn before transport admission when no executor is
/// available.
pub fn require_executor(executor: Option<Arc<dyn Executor>>) -> Result<Arc<dyn Executor>, Error> {
	executor.ok_or_else(|| {
		Error::Unsupported(vec![
			Unsupported::builder()
				.what(Str::new_static("cursor/executor"))
				.detail(Str::new_static("Cursor requires an in-turn executor"))
				.action(UnsupportedAction::Dropped)
				.build(),
		])
	})
}

/// Drives one admitted invocation and returns the ordered client frames.
///
/// `cancel` represents an `ExecServerAbort`. Winning that branch drops the
/// executor future, which is the structural cancellation contract, and emits
/// no late execution frames. Heartbeats are emitted only while the invocation
/// remains outstanding.
pub async fn drive_invocation(
	executor: Arc<dyn Executor>,
	invocation: Invoke,
	context: ShellContext,
	mut cancel: oneshot::Receiver<()>,
) -> Vec<wire::AgentClientMessage> {
	let (inputs_tx, inputs_rx) = flume::bounded(16);
	let timeout_ms = invocation.timeout_ms;
	let mut execution = Box::pin(executor.invoke(invocation, inputs_tx));
	let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
	heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
	// `interval` ticks immediately; a heartbeat before the start frame is not
	// useful.
	heartbeat.tick().await;

	let mut frames = Vec::new();
	let mut framer = InvocationFramer::new(context);
	frames.push(framer.start());
	let deadline = tokio::time::sleep(Duration::from_millis(timeout_ms));
	tokio::pin!(deadline);
	loop {
		tokio::select! {
			biased;
			_ = &mut cancel => break,
			completion = &mut execution => {
				frames.extend(framer.complete(&completion));
				break;
			}
			() = &mut deadline => break,
			Ok(input) = inputs_rx.recv_async() => frames.extend(framer.input(&input)),
			_ = heartbeat.tick() => frames.push(framer.heartbeat()),
		}
	}
	frames
}

async fn drive_invocation_to(
	executor: Arc<dyn Executor>,
	invocation: Invoke,
	context: ShellContext,
	mut cancel: oneshot::Receiver<()>,
	client_tx: flume::Sender<Bytes>,
	terminal_tx: flume::Sender<TurnError>,
) {
	let shell = invocation.tool_call.is_some();
	let timeout_error = invocation_timeout_error(&invocation);
	let timeout_ms = invocation.timeout_ms;
	let (inputs_tx, inputs_rx) = flume::bounded(16);
	let mut execution = Box::pin(executor.invoke(invocation, inputs_tx));
	let deadline = tokio::time::sleep(Duration::from_millis(timeout_ms));
	tokio::pin!(deadline);
	let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
	heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
	heartbeat.tick().await;
	let mut framer = InvocationFramer::new(context);
	if shell
		&& client_tx
			.send_async(connect_frame(&framer.start()))
			.await
			.is_err()
	{
		return;
	}
	loop {
		let frames = tokio::select! {
			biased;
			_ = &mut cancel => return,
			completion = &mut execution => {
				let frames = if shell {
					framer.complete(&completion)
				} else if completion.vendor.is_empty() {
					vec![
						framer.throw("control invocation completed without vendor response", "outcome_failed"),
						framer.stream_close(),
					]
				} else {
					wire::AgentClientMessage::decode(completion.vendor.clone())
						.map_or_else(
							|_| vec![
								framer.throw("invalid control invocation response", "outcome_failed"),
								framer.stream_close(),
							],
							|message| vec![message],
						)
				};
				for frame in frames {
					if client_tx.send_async(connect_frame(&frame)).await.is_err() {
						return;
					}
				}
				return;
			}
			() = &mut deadline => {
				let _ = terminal_tx.send_async(timeout_error).await;
				return;
			}
			Ok(input) = inputs_rx.recv_async() => {
				if shell || matches!(&input.payload, InvokePayload::Vendor(_)) {
					framer.input(&input)
				} else {
					Vec::new()
				}
			},
			_ = heartbeat.tick() => {
				if client_tx.send_async(connect_frame(&framer.heartbeat())).await.is_err() {
					return;
				}
				continue;
			},
		};
		for frame in frames {
			if client_tx.send_async(connect_frame(&frame)).await.is_err() {
				return;
			}
		}
	}
}

fn invocation_timeout_error(invocation: &Invoke) -> TurnError {
	TurnError::builder()
		.kind(TurnErrorKind::InvokeTimeout)
		.detail(Str::from(format!(
			"Cursor invocation {} exceeded its {}ms deadline",
			invocation.invocation_id, invocation.timeout_ms
		)))
		.unsupported(Vec::new())
		.retry_after_ms(0)
		.build()
}

/// Metadata retained while translating a shell invocation in both directions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShellContext {
	/// Numeric correlation id on Cursor's exec channel.
	pub id:      u32,
	/// Optional attachable execution id.
	pub exec_id: Str,
	/// Command echoed by every structured result variant.
	pub command: Str,
	/// Normalized working directory echoed by exit and result frames.
	pub cwd:     Str,
}

/// Stateful ANSI-safe conversion of canonical invocation inputs to Cursor
/// frames.
pub struct InvocationFramer {
	context: ShellContext,
	stdout:  AnsiBuffer,
	stderr:  AnsiBuffer,
}

impl InvocationFramer {
	/// Starts a framer for one live shell exec.
	#[must_use]
	pub fn new(context: ShellContext) -> Self {
		Self { context, stdout: AnsiBuffer::default(), stderr: AnsiBuffer::default() }
	}

	/// Produces Cursor's required shell-stream start frame.
	#[must_use]
	pub fn start(&self) -> wire::AgentClientMessage {
		self.shell_stream(wire::shell_stream::Event::Start(wire::ShellStreamStart {
			sandbox_policy: None,
		}))
	}

	/// Converts one canonical input, retaining incomplete ANSI sequences.
	#[must_use]
	pub fn input(&mut self, input: &InvokeInput) -> Vec<wire::AgentClientMessage> {
		if input.invocation_id != invocation_id(&self.context) {
			return Vec::new();
		}
		match &input.payload {
			InvokePayload::Chunk(chunk) => match chunk.channel {
				InvokeChannel::Stdout => self.stdout.push(&chunk.data).map_or_else(Vec::new, |data| {
					vec![self.shell_stream(wire::shell_stream::Event::Stdout(wire::ShellStreamStdout {
						data,
					}))]
				}),
				InvokeChannel::Stderr => self.stderr.push(&chunk.data).map_or_else(Vec::new, |data| {
					vec![self.shell_stream(wire::shell_stream::Event::Stderr(wire::ShellStreamStderr {
						data,
					}))]
				}),
				InvokeChannel::Progress => Vec::new(),
				_ => Vec::new(),
			},
			InvokePayload::Vendor(payload) => wire::AgentClientMessage::decode(payload.clone())
				.map_or_else(|_| Vec::new(), |message| vec![message]),
			_ => Vec::new(),
		}
	}

	/// Synthesizes the exit, structured result, and stream-close sequence.
	#[must_use]
	pub fn complete(&mut self, completion: &InvokeComplete) -> Vec<wire::AgentClientMessage> {
		if completion.invocation_id != invocation_id(&self.context) {
			return Vec::new();
		}
		let mut frames = Vec::new();
		if let Some(data) = self.stdout.finish() {
			frames.push(
				self.shell_stream(wire::shell_stream::Event::Stdout(wire::ShellStreamStdout { data })),
			);
		}
		if let Some(data) = self.stderr.finish() {
			frames.push(
				self.shell_stream(wire::shell_stream::Event::Stderr(wire::ShellStreamStderr { data })),
			);
		}

		let Some(status) = &completion.status else {
			frames.push(self.throw("executor completed without ExecStatus", "outcome_failed"));
			frames.push(self.stream_close());
			return frames;
		};
		let output = completion
			.tool_result
			.as_ref()
			.map(tool_result_text)
			.unwrap_or_default();
		frames.extend(self.completion_frames(status, &output));
		frames.push(self.stream_close());
		frames
	}

	/// Produces a heartbeat for this outstanding invocation.
	#[must_use]
	pub const fn heartbeat(&self) -> wire::AgentClientMessage {
		client_control(wire::exec_client_control_message::Message::Heartbeat(
			wire::ExecClientHeartbeat { id: self.context.id },
		))
	}

	fn completion_frames(&self, status: &ExecStatus, output: &str) -> Vec<wire::AgentClientMessage> {
		let cwd = if status.cwd.is_empty() {
			self.context.cwd.as_str()
		} else {
			status.cwd.as_str()
		};
		let command = self.context.command.to_string();
		let cwd_owned = cwd.to_owned();
		let exit_code = status.exit_code;
		let local_ms = u64_to_i32(status.local_execution_time_ms);
		let mut frames = Vec::new();
		let result = match status.outcome {
			ExecOutcome::Exited => wire::shell_result::Result::Success(wire::ShellSuccess {
				command,
				working_directory: cwd_owned.clone(),
				exit_code,
				signal: status.signal.to_string(),
				stdout: output.to_owned(),
				stderr: String::new(),
				execution_time: local_ms,
				local_execution_time_ms: Some(local_ms),
				..Default::default()
			}),
			ExecOutcome::Failed | ExecOutcome::Cancelled => {
				wire::shell_result::Result::Failure(wire::ShellFailure {
					command,
					working_directory: cwd_owned.clone(),
					exit_code,
					signal: status.signal.to_string(),
					stdout: String::new(),
					stderr: output.to_owned(),
					execution_time: local_ms,
					aborted: status.aborted || matches!(status.outcome, ExecOutcome::Cancelled),
					local_execution_time_ms: Some(local_ms),
					..Default::default()
				})
			},
			ExecOutcome::Rejected => {
				let rejected = wire::ShellRejected {
					command,
					working_directory: cwd_owned.clone(),
					reason: status.reason.to_string(),
					is_readonly: status.is_readonly,
				};
				frames.push(self.shell_stream(wire::shell_stream::Event::Rejected(rejected.clone())));
				wire::shell_result::Result::Rejected(rejected)
			},
			ExecOutcome::Denied => {
				let denied = wire::ShellPermissionDenied {
					command,
					working_directory: cwd_owned.clone(),
					error: status.reason.to_string(),
					is_readonly: status.is_readonly,
				};
				frames.push(
					self.shell_stream(wire::shell_stream::Event::PermissionDenied(denied.clone())),
				);
				wire::shell_result::Result::PermissionDenied(denied)
			},
			ExecOutcome::Timeout => {
				frames.push(self.shell_stream(wire::shell_stream::Event::Stderr(
					wire::ShellStreamStderr {
						data: format!("Command timed out after {}ms", status.command_timeout_ms),
					},
				)));
				wire::shell_result::Result::Timeout(wire::ShellTimeout {
					command,
					working_directory: cwd_owned.clone(),
					timeout_ms: u64_to_i32(status.command_timeout_ms),
				})
			},
			_ => wire::shell_result::Result::Failure(wire::ShellFailure {
				command,
				working_directory: cwd_owned.clone(),
				exit_code,
				signal: status.signal.to_string(),
				stdout: String::new(),
				stderr: output.to_owned(),
				execution_time: local_ms,
				aborted: true,
				local_execution_time_ms: Some(local_ms),
				..Default::default()
			}),
		};
		let (code, aborted, abort_reason) = match status.outcome {
			ExecOutcome::Exited => (status.exit_code.max(0) as u32, false, None),
			ExecOutcome::Failed | ExecOutcome::Cancelled => (
				status.exit_code.max(0) as u32,
				status.aborted || matches!(status.outcome, ExecOutcome::Cancelled),
				abort_reason(status),
			),
			ExecOutcome::Rejected | ExecOutcome::Denied => (1, false, None),
			ExecOutcome::Timeout => (1, true, None),
			_ => (1, true, None),
		};
		frames.push(self.shell_stream(wire::shell_stream::Event::Exit(wire::ShellStreamExit {
			code,
			cwd: cwd_owned,
			output_location: None,
			aborted,
			abort_reason,
			local_execution_time_ms: Some(local_ms),
		})));
		frames.push(exec_message(
			&self.context,
			wire::exec_client_message::Message::ShellResult(Box::new(wire::ShellResult {
				sandbox_policy:   None,
				is_background:    None,
				terminals_folder: None,
				pid:              None,
				result:           Some(result),
			})),
		));
		frames
	}

	fn shell_stream(&self, event: wire::shell_stream::Event) -> wire::AgentClientMessage {
		exec_message(
			&self.context,
			wire::exec_client_message::Message::ShellStream(wire::ShellStream { event: Some(event) }),
		)
	}

	fn throw(&self, error: &str, error_code: &str) -> wire::AgentClientMessage {
		client_control(wire::exec_client_control_message::Message::Throw(wire::ExecClientThrow {
			id:          self.context.id,
			error:       error.to_owned(),
			stack_trace: None,
			error_code:  Some(error_code.to_owned()),
		}))
	}

	const fn stream_close(&self) -> wire::AgentClientMessage {
		client_control(wire::exec_client_control_message::Message::StreamClose(
			wire::ExecClientStreamClose { id: self.context.id },
		))
	}
}

#[derive(Default)]
struct AnsiBuffer {
	bytes: BytesMut,
}

impl AnsiBuffer {
	fn push(&mut self, chunk: &[u8]) -> Option<String> {
		self.bytes.put_slice(chunk);
		self.flush(false)
	}

	fn finish(&mut self) -> Option<String> {
		self.flush(true)
	}

	fn flush(&mut self, final_frame: bool) -> Option<String> {
		let utf8_end = valid_utf8_prefix(&self.bytes, final_frame);
		let ansi_end = if final_frame {
			utf8_end
		} else {
			ansi_safe_end(&self.bytes[..utf8_end])
		};
		if ansi_end == 0 {
			return None;
		}
		let data = String::from_utf8_lossy(&self.bytes[..ansi_end]).into_owned();
		let remaining = self.bytes.len() - ansi_end;
		self.bytes.copy_within(ansi_end.., 0);
		self.bytes.truncate(remaining);
		Some(data)
	}
}

const fn valid_utf8_prefix(bytes: &[u8], final_frame: bool) -> usize {
	match std::str::from_utf8(bytes) {
		Ok(_) => bytes.len(),
		Err(error) if error.error_len().is_none() && !final_frame => error.valid_up_to(),
		Err(_) => bytes.len(),
	}
}

fn ansi_safe_end(bytes: &[u8]) -> usize {
	let Some(index) = bytes.iter().rposition(|byte| *byte == 0x1b) else {
		return bytes.len();
	};
	let suffix = &bytes[index + 1..];
	let incomplete = suffix.is_empty()
		|| suffix == b"["
		|| suffix.first() == Some(&b'[')
			&& (suffix[1..].iter().all(u8::is_ascii_digit)
				|| suffix[1..] == *b"?"
				|| suffix[1..].first() == Some(&b'?') && suffix[2..].iter().all(u8::is_ascii_digit))
		|| suffix.first() == Some(&b']')
			&& suffix[1..]
				.iter()
				.all(|byte| byte.is_ascii_digit() || *byte == b';');
	if incomplete { index } else { bytes.len() }
}

/// Decodes one recorded server message through the transport-independent
/// pin-test entry point, without requiring a live connection.
pub fn decode_server_message(
	message: wire::AgentServerMessage,
	state: &mut CursorDecodeState,
) -> Result<SmallVec<TurnEvent, 2>, Error> {
	match message.message {
		Some(wire::agent_server_message::Message::ExecServerMessage(exec)) => {
			let invoke = invoke_from_exec(exec, state)?;
			Ok(smallvec![TurnEvent::Invoke(invoke)])
		},
		Some(wire::agent_server_message::Message::ExecServerControlMessage(control)) => {
			match control.message {
				Some(wire::exec_server_control_message::Message::Abort(abort)) => {
					Ok(smallvec![TurnEvent::InvokeCancel {
						invocation_id: Str::from(abort.id.to_string()),
					},])
				},
				None => Ok(SmallVec::new()),
			}
		},
		Some(wire::agent_server_message::Message::InteractionUpdate(update)) => {
			Ok(decode_interaction(*update, state))
		},
		_ => Ok(SmallVec::new()),
	}
}

fn decode_interaction(
	update: wire::InteractionUpdate,
	state: &mut CursorDecodeState,
) -> SmallVec<TurnEvent, 2> {
	let mut events = SmallVec::new();
	match update.message {
		Some(wire::interaction_update::Message::TextDelta(delta)) => {
			push_part_delta(state, StreamPartKind::Text, delta.text, &mut events);
		},
		Some(wire::interaction_update::Message::ThinkingDelta(delta)) => {
			push_part_delta(state, StreamPartKind::Thinking, delta.text, &mut events);
		},
		Some(wire::interaction_update::Message::ToolCallStarted(started)) => {
			start_tool_part(
				state,
				tool_call_id(started.call_id, started.tool_call.as_ref()),
				started.tool_call.as_ref(),
				&mut events,
			);
		},
		Some(wire::interaction_update::Message::PartialToolCall(partial)) => {
			let call_id = tool_call_id(partial.call_id, partial.tool_call.as_ref());
			if !matches!(state.open_part, Some((_, StreamPartKind::ToolCall)))
				|| state.open_tool_call_id != call_id
			{
				start_tool_part(state, call_id, partial.tool_call.as_ref(), &mut events);
			}
			if let Some((index, StreamPartKind::ToolCall)) = state.open_part
				&& !partial.args_text_delta.is_empty()
			{
				events
					.push(TurnEvent::PartDelta { index, chunk: Bytes::from(partial.args_text_delta) });
			}
		},
		Some(wire::interaction_update::Message::ToolCallCompleted(completed)) => {
			let call_id = tool_call_id(completed.call_id, completed.tool_call.as_ref());
			if let Some((index, StreamPartKind::ToolCall)) = state.open_part
				&& (call_id.is_empty() || state.open_tool_call_id == call_id)
			{
				state.open_part = None;
				state.open_tool_call_id = Str::default();
				events.push(TurnEvent::PartEnd { index, signature: Default::default() });
			}
		},
		Some(wire::interaction_update::Message::TokenDelta(delta)) => {
			state.output_tokens = state
				.output_tokens
				.saturating_add(delta.tokens.max(0) as u64);
			state.saw_usage = true;
		},
		Some(wire::interaction_update::Message::TurnEnded(_)) => {
			if let Some((index, _)) = state.open_part.take() {
				events.push(TurnEvent::PartEnd { index, signature: Default::default() });
			}
			state.open_tool_call_id = Str::default();
			let stop = if state.saw_tool_call {
				StopReason::ToolUse
			} else {
				StopReason::EndTurn
			};
			state.saw_tool_call = false;
			let usage = state.saw_usage.then(|| {
				Usage::builder()
					.input_tokens(0)
					.output_tokens(state.output_tokens)
					.cache_read_tokens(0)
					.cache_write_tokens(0)
					.accuracy(Accuracy::Exact)
					.detail(Props::default())
					.build()
			});
			state.saw_usage = false;
			state.output_tokens = 0;
			events.push(TurnEvent::Outcome(
				ChatOutcome::builder()
					.output(Vec::new())
					.stop(stop)
					.maybe_usage(usage)
					.unsupported(std::mem::take(&mut state.unsupported))
					.provider(Str::new_static("cursor"))
					.model(state.model.clone())
					.props(Props::default())
					.build(),
			));
		},
		_ => {},
	}
	events
}
fn start_tool_part(
	state: &mut CursorDecodeState,
	call_id: Str,
	tool_call: Option<&wire::ToolCall>,
	events: &mut SmallVec<TurnEvent, 2>,
) {
	if let Some((index, _)) = state.open_part.take() {
		events.push(TurnEvent::PartEnd { index, signature: Default::default() });
	}
	let call_id = if call_id.is_empty() {
		tool_call
			.and_then(|tool_call| tool_call.tool_call_id.as_deref())
			.map_or_else(Str::default, Str::from)
	} else {
		call_id
	};
	let index = state.next_part;
	state.next_part += 1;
	state.open_part = Some((index, StreamPartKind::ToolCall));
	state.open_tool_call_id = call_id.clone();
	state.saw_tool_call = true;
	events.push(TurnEvent::PartStart {
		index,
		kind: StreamPartKind::ToolCall,
		tool_call_id: call_id,
		tool_name: cursor_tool_name(tool_call),
	});
}

fn tool_call_id(call_id: String, tool_call: Option<&wire::ToolCall>) -> Str {
	if call_id.is_empty() {
		tool_call
			.and_then(|tool_call| tool_call.tool_call_id.as_deref())
			.map_or_else(Str::default, Str::from)
	} else {
		Str::from(call_id)
	}
}

fn cursor_tool_name(tool_call: Option<&wire::ToolCall>) -> Str {
	use wire::tool_call::Tool;

	let Some(tool) = tool_call.and_then(|tool_call| tool_call.tool.as_ref()) else {
		return Str::default();
	};
	let name = match tool {
		Tool::ShellToolCall(_) => "bash",
		Tool::DeleteToolCall(_) => "delete",
		Tool::GlobToolCall(_) => "glob",
		Tool::GrepToolCall(_) => "grep",
		Tool::ReadToolCall(_) => "read",
		Tool::UpdateTodosToolCall(_) => "update_todos",
		Tool::ReadTodosToolCall(_) => "read_todos",
		Tool::EditToolCall(_) => "edit",
		Tool::LsToolCall(_) => "ls",
		Tool::ReadLintsToolCall(_) => "read_lints",
		Tool::McpToolCall(call) => {
			return call.args.as_ref().map_or_else(Str::default, |args| {
				Str::from(if args.tool_name.is_empty() {
					args.name.as_str()
				} else {
					args.tool_name.as_str()
				})
			});
		},
		Tool::SemSearchToolCall(_) => "sem_search",
		Tool::CreatePlanToolCall(_) => "create_plan",
		Tool::WebSearchToolCall(_) => "web_search",
		Tool::TaskToolCall(_) => "task",
		Tool::ListMcpResourcesToolCall(_) => "list_mcp_resources",
		Tool::ReadMcpResourceToolCall(_) => "read_mcp_resource",
		Tool::ApplyAgentDiffToolCall(_) => "apply_agent_diff",
		Tool::AskQuestionToolCall(_) => "ask_question",
		Tool::FetchToolCall(_) => "fetch",
		Tool::SwitchModeToolCall(_) => "switch_mode",
		Tool::ExaSearchToolCall(_) => "exa_search",
		Tool::ExaFetchToolCall(_) => "exa_fetch",
		Tool::GenerateImageToolCall(_) => "generate_image",
		Tool::RecordScreenToolCall(_) => "record_screen",
		Tool::ComputerUseToolCall(_) => "computer_use",
		Tool::WriteShellStdinToolCall(_) => "write_shell_stdin",
		Tool::ReflectToolCall(_) => "reflect",
		Tool::SetupVmEnvironmentToolCall(_) => "setup_vm_environment",
		Tool::TruncatedToolCall(_) => "truncated",
		Tool::StartGrindExecutionToolCall(_) => "start_grind_execution",
		Tool::StartGrindPlanningToolCall(_) => "start_grind_planning",
		Tool::PiReadToolCall(_) => "read",
		Tool::PiBashToolCall(_) => "bash",
		Tool::PiEditToolCall(_) => "edit",
		Tool::PiWriteToolCall(_) => "write",
		Tool::PiGrepToolCall(_) => "grep",
		Tool::PiFindToolCall(_) => "find",
		Tool::PiLsToolCall(_) => "ls",
		Tool::ConnectScmToolCall(_) => "connect_scm",
		Tool::SearchConversationsToolCall(_) => "search_conversations",
	};
	Str::new_static(name)
}

fn push_part_delta(
	state: &mut CursorDecodeState,
	kind: StreamPartKind,
	text: String,
	events: &mut SmallVec<TurnEvent, 2>,
) {
	let index = match state.open_part {
		Some((index, open_kind)) if open_kind == kind => index,
		Some((index, _)) => {
			events.push(TurnEvent::PartEnd { index, signature: Default::default() });
			state.open_tool_call_id = Str::default();
			let next = state.next_part;
			state.next_part += 1;
			state.open_part = Some((next, kind));
			events.push(TurnEvent::PartStart {
				index: next,
				kind,
				tool_call_id: Str::default(),
				tool_name: Str::default(),
			});
			next
		},
		None => {
			let next = state.next_part;
			state.next_part += 1;
			state.open_part = Some((next, kind));
			state.open_tool_call_id = Str::default();
			events.push(TurnEvent::PartStart {
				index: next,
				kind,
				tool_call_id: Str::default(),
				tool_name: Str::default(),
			});
			next
		},
	};
	if !text.is_empty() {
		events.push(TurnEvent::PartDelta { index, chunk: Bytes::from(text) });
	}
}

fn invoke_and_context_from_exec(
	exec: wire::ExecServerMessage,
	state: &mut CursorDecodeState,
) -> Result<(Invoke, ShellContext), Error> {
	let (command, cwd) = match &exec.message {
		Some(
			wire::exec_server_message::Message::ShellStreamArgs(args)
			| wire::exec_server_message::Message::ShellArgs(args),
		) => (
			Str::from(args.command.as_str()),
			if args.working_directory.is_empty() {
				Str::from(
					std::env::current_dir()
						.ok()
						.and_then(|path| path.to_str().map(str::to_owned))
						.unwrap_or_default(),
				)
			} else {
				Str::from(args.working_directory.as_str())
			},
		),
		_ => (Str::default(), Str::default()),
	};
	let context =
		ShellContext { id: exec.id, exec_id: Str::from(exec.exec_id.as_str()), command, cwd };
	invoke_from_exec(exec, state).map(|invoke| (invoke, context))
}

fn invoke_from_exec(
	exec: wire::ExecServerMessage,
	state: &mut CursorDecodeState,
) -> Result<Invoke, Error> {
	let invocation_id = if exec.exec_id.is_empty() {
		Str::from(exec.id.to_string())
	} else {
		Str::from(exec.exec_id.as_str())
	};
	let mut props = Props::default();
	props.insert_ns("cursor", "exec_id", serde_json::Value::String(exec.exec_id.clone()));
	props.insert_ns("cursor", "numeric_id", serde_json::Value::from(exec.id));
	let encoded = exec.encode_to_vec();
	match exec.message {
		Some(
			wire::exec_server_message::Message::ShellStreamArgs(args)
			| wire::exec_server_message::Message::ShellArgs(args),
		) => {
			let call_id = args.tool_call_id.parse().unwrap_or_else(|_| CallId::new());
			state
				.calls
				.insert(Str::from(args.tool_call_id.as_str()), call_id);
			let mut args_value = serde_json::Map::new();
			args_value.insert("command".to_owned(), serde_json::Value::String(args.command));
			if !args.working_directory.is_empty() {
				args_value.insert(
					"working_directory".to_owned(),
					serde_json::Value::String(args.working_directory),
				);
			}
			if args.timeout > 0 {
				args_value.insert("timeout".to_owned(), serde_json::Value::from(args.timeout));
			}
			args_value.insert(
				"tool_call_id".to_owned(),
				serde_json::Value::String(args.tool_call_id),
			);
			let args_json = serde_json::to_vec(&args_value)
				.map_err(|error| Error::Provider(Str::from(error.to_string())))?;
			Ok(Invoke::builder()
				.invocation_id(invocation_id)
				.name(Str::new_static("bash"))
				.tool_call(
					ToolCall::builder()
						.id(call_id)
						.name(Str::new_static("bash"))
						.args_json(args_json.into())
						.thought_signature(Bytes::new())
						.build(),
				)
				.vendor(Bytes::new())
				.timeout_ms(if args.timeout > 0 {
					args.timeout as u64
				} else {
					DEFAULT_INVOKE_TIMEOUT_MS
				})
				.props(props)
				.build())
		},
		_ => Ok(Invoke::builder()
			.invocation_id(invocation_id)
			.name(Str::new_static("cursor/control"))
			.vendor(encoded.into())
			.timeout_ms(DEFAULT_INVOKE_TIMEOUT_MS)
			.props(props)
			.build()),
	}
}

fn assemble_request(
	req: &ChatRequest,
) -> Result<(wire::AgentClientMessage, Vec<Unsupported>), Error> {
	let mut unsupported = UnsupportedSink::new();
	if req.sampling.is_some() {
		unsupported.drop_feature("sampling", "Cursor Composer selects sampling server-side");
	}
	if req.thinking.is_some() {
		unsupported.drop_feature("thinking", "Cursor Composer selects reasoning server-side");
	}
	if req.response_format.is_some() {
		unsupported
			.drop_feature("response_format", "Cursor Agent has no structured-output request field");
	}
	if req.tool_choice.is_some() {
		unsupported.drop_feature("tool_choice", "Cursor requests tools through its exec channel");
	}
	if req.cache.is_some() {
		unsupported.drop_feature("cache", "Cursor owns conversation-state caching");
	}
	if !req.tools.is_empty() {
		unsupported.emulate(
			"tools",
			"tool contracts are supplied by Cursor's request-context exec handshake",
		);
	}
	if req
		.provider_options
		.as_ref()
		.is_some_and(|props| !props.is_empty())
		|| req.thread.items.iter().any(|item| !item.props.is_empty())
	{
		unsupported.drop_feature(
			"props",
			"Cursor's pinned AgentRunRequest has no projection for canonical extension properties",
		);
	}
	if req.thread.items.iter().any(|item| {
		matches!(
			&item.kind,
			ItemKind::Message(message) if message.role == Role::Assistant
		) || matches!(&item.kind, ItemKind::ToolCall(_) | ItemKind::ToolResult(_))
	}) || req
		.thread
		.items
		.iter()
		.filter(|item| matches!(&item.kind, ItemKind::Message(message) if message.role == Role::User))
		.count()
		> 1
	{
		unsupported.drop_feature(
			"thread.history",
			"the pinned stateless request projection cannot replay prior user, assistant, or tool \
			 history",
		);
	}
	if req.thread.items.iter().any(|item| {
		matches!(
			&item.kind,
			ItemKind::Message(message)
				if message.parts.iter().any(|part| !matches!(part, Part::Text(_)))
		) || matches!(
			&item.kind,
			ItemKind::ToolResult(result)
				if result.parts.iter().any(|part| !matches!(part, Part::Text(_)))
		)
	}) {
		unsupported.drop_feature(
			"thread.media",
			"Cursor request assembly cannot inline canonical blob or thinking parts",
		);
	}

	let mut system = String::new();
	let mut active_user = None;
	for item in &req.thread.items {
		if let ItemKind::Message(message) = &item.kind {
			let text = message
				.parts
				.iter()
				.filter_map(|part| match part {
					Part::Text(text) => Some(text.as_str()),
					_ => None,
				})
				.collect::<String>();
			match message.role {
				Role::System => {
					if !system.is_empty() {
						system.push('\n');
					}
					system.push_str(&text);
				},
				Role::User => active_user = Some(text),
				Role::Assistant => {},
				_ => {},
			}
		}
	}
	let action = active_user.map_or_else(
		|| {
			wire::conversation_action::Action::ResumeAction(wire::ResumeAction {
				request_context: None,
			})
		},
		|text| {
			wire::conversation_action::Action::UserMessageAction(wire::UserMessageAction {
				user_message:                 Some(wire::UserMessage {
					text,
					message_id: ulid::Ulid::generate().to_string(),
					..Default::default()
				}),
				request_context:              None,
				send_to_interaction_listener: None,
			})
		},
	);
	let root_prompt_messages_json = if system.is_empty() {
		Vec::new()
	} else {
		vec![
			serde_json::to_vec(&serde_json::json!({ "role": "system", "content": system }))
				.map_err(|error| Error::Provider(Str::from(error.to_string())))?,
		]
	};
	let max_mode = req
		.model_policy
		.as_deref()
		.and_then(|policy| policy.cursor_max_mode)
		.unwrap_or(false);
	let run = wire::AgentRunRequest {
		conversation_state: Some(wire::ConversationStateStructure {
			root_prompt_messages_json,
			..Default::default()
		}),
		action: Some(wire::ConversationAction { action: Some(action) }),
		model_details: Some(wire::ModelDetails {
			model_id: req.model.to_string(),
			max_mode: Some(max_mode),
			display_model_id: req.model.to_string(),
			display_name: req.model.to_string(),
			..Default::default()
		}),
		requested_model: Some(wire::RequestedModel {
			model_id: req.model.to_string(),
			max_mode,
			..Default::default()
		}),
		..Default::default()
	};
	Ok((
		wire::AgentClientMessage {
			message: Some(wire::agent_client_message::Message::RunRequest(Box::new(run))),
		},
		unsupported.into_vec(),
	))
}

/// Adds a Connect protocol envelope around one protobuf message.
#[must_use]
pub fn connect_frame(message: &wire::AgentClientMessage) -> Bytes {
	let payload_len = message.encoded_len();
	let mut framed = BytesMut::with_capacity(payload_len + 5);
	framed.put_u8(0);
	framed.put_u32(payload_len as u32);
	message.encode(&mut framed).expect("BytesMut is growable");
	framed.freeze()
}

/// Incrementally decodes Connect envelopes without copying complete payloads.
#[derive(Default)]
pub struct ConnectDecoder {
	buffer: BytesMut,
}

impl ConnectDecoder {
	/// Appends borrowed transport bytes and returns every complete protobuf
	/// payload.
	pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<Bytes>, Error> {
		self.buffer.put_slice(bytes);
		let mut messages = SmallVec::<Bytes, 2>::new();
		self.drain_buffer(&mut messages)?;
		Ok(messages.into_vec())
	}

	/// Decodes owned transport bytes without copying complete frames.
	fn push_bytes(&mut self, mut bytes: Bytes) -> Result<SmallVec<Bytes, 2>, Error> {
		let mut messages = SmallVec::new();
		if !self.buffer.is_empty() {
			if self.buffer.len() < 5 {
				let needed = 5 - self.buffer.len();
				let copied = needed.min(bytes.len());
				self.buffer.put_slice(&bytes[..copied]);
				bytes.advance(copied);
				if self.buffer.len() < 5 {
					return Ok(messages);
				}
			}

			let len =
				u32::from_be_bytes([self.buffer[1], self.buffer[2], self.buffer[3], self.buffer[4]])
					as usize;
			let Some(frame_len) = len.checked_add(5) else {
				return Err(Error::Transport(Str::new_static("Cursor frame length overflow")));
			};
			if self.buffer.len() < frame_len {
				let needed = frame_len - self.buffer.len();
				let copied = needed.min(bytes.len());
				self.buffer.put_slice(&bytes[..copied]);
				bytes.advance(copied);
				if self.buffer.len() < frame_len {
					return Ok(messages);
				}
			}
			self.drain_buffer(&mut messages)?;
		}

		while bytes.len() >= 5 {
			let flags = bytes[0];
			let len = u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
			let Some(frame_len) = len.checked_add(5) else {
				return Err(Error::Transport(Str::new_static("Cursor frame length overflow")));
			};
			if bytes.len() < frame_len {
				break;
			}
			bytes.advance(5);
			let payload = bytes.split_to(len);
			if flags & CONNECT_END_STREAM != 0 {
				if payload.windows(7).any(|window| window == b"\"error\"") {
					return Err(Error::Transport(Str::new_static("Cursor Connect end-stream error")));
				}
			} else {
				messages.push(payload);
			}
		}
		if !bytes.is_empty() {
			self.buffer.put_slice(&bytes);
		}
		Ok(messages)
	}

	fn drain_buffer(&mut self, messages: &mut SmallVec<Bytes, 2>) -> Result<(), Error> {
		while self.buffer.len() >= 5 {
			let flags = self.buffer[0];
			let len =
				u32::from_be_bytes([self.buffer[1], self.buffer[2], self.buffer[3], self.buffer[4]])
					as usize;
			let Some(frame_len) = len.checked_add(5) else {
				return Err(Error::Transport(Str::new_static("Cursor frame length overflow")));
			};
			if self.buffer.len() < frame_len {
				break;
			}
			self.buffer.advance(5);
			let payload = self.buffer.split_to(len).freeze();
			if flags & CONNECT_END_STREAM != 0 {
				if payload.windows(7).any(|window| window == b"\"error\"") {
					return Err(Error::Transport(Str::new_static("Cursor Connect end-stream error")));
				}
			} else {
				messages.push(payload);
			}
		}
		Ok(())
	}
}

fn exec_message(
	context: &ShellContext,
	message: wire::exec_client_message::Message,
) -> wire::AgentClientMessage {
	wire::AgentClientMessage {
		message: Some(wire::agent_client_message::Message::ExecClientMessage(
			wire::ExecClientMessage {
				id: context.id,
				exec_id: context.exec_id.to_string(),
				message: Some(message),
				..Default::default()
			},
		)),
	}
}

const fn client_control(
	message: wire::exec_client_control_message::Message,
) -> wire::AgentClientMessage {
	wire::AgentClientMessage {
		message: Some(wire::agent_client_message::Message::ExecClientControlMessage(
			wire::ExecClientControlMessage { message: Some(message) },
		)),
	}
}

fn invocation_id(context: &ShellContext) -> Str {
	if context.exec_id.is_empty() {
		Str::from(context.id.to_string())
	} else {
		context.exec_id.clone()
	}
}

fn tool_result_text(result: &ToolResult) -> String {
	let mut text = String::new();
	for part in &result.parts {
		if let Part::Text(value) = part {
			text.push_str(value);
		}
	}
	text
}

const fn abort_reason(status: &ExecStatus) -> Option<i32> {
	if !status.aborted {
		return None;
	}
	Some(if matches!(status.outcome, ExecOutcome::Timeout) {
		wire::ShellAbortReason::Timeout as i32
	} else {
		wire::ShellAbortReason::UserAbort as i32
	})
}

fn u64_to_i32(value: u64) -> i32 {
	i32::try_from(value).unwrap_or(i32::MAX)
}

#[cfg(test)]
mod tests {
	use std::{collections::BTreeMap, sync::Arc, time::Duration};

	use bytes::Bytes;
	use http::header;
	use omp_core::Str;
	use omp_llm_types::{
		ChatRequest, ExecOutcome, ExecStatus, Invoke, InvokeChannel, InvokeChunk, InvokeComplete,
		InvokeInput, InvokePayload, Props, ResolvedModelHeaders, ResolvedModelPolicy, StopReason,
		StreamPartKind, Thread, TurnErrorKind, TurnEvent, facet::Executor,
	};
	use tokio::sync::oneshot;

	use super::{
		CURSOR_CLIENT_VERSION, CursorDecodeState, InvocationFramer, ShellContext, assemble_request,
		cursor_request_headers, decode_server_message, drive_invocation_to, invocation_id,
		invoke_from_exec, require_executor,
	};

	fn context() -> ShellContext {
		ShellContext {
			id:      7,
			exec_id: Str::new_static("exec-7"),
			command: Str::new_static("printf test"),
			cwd:     Str::new_static("/work"),
		}
	}

	fn status(outcome: ExecOutcome) -> ExecStatus {
		ExecStatus::builder()
			.outcome(outcome)
			.exit_code(if matches!(outcome, ExecOutcome::Exited) {
				0
			} else {
				2
			})
			.signal(Str::default())
			.reason(Str::new_static("policy"))
			.cwd(Str::new_static("/work"))
			.aborted(matches!(outcome, ExecOutcome::Timeout))
			.output_location(Str::default())
			.local_execution_time_ms(12)
			.is_readonly(true)
			.command_timeout_ms(500)
			.build()
	}

	fn complete(outcome: ExecOutcome) -> InvokeComplete {
		InvokeComplete::builder()
			.invocation_id(Str::new_static("exec-7"))
			.status(status(outcome))
			.vendor(Bytes::new())
			.props(Props::default())
			.build()
	}

	#[test]
	fn caller_headers_reach_cursor_request_with_lowercase_names() {
		let request = ChatRequest::builder()
			.model(Str::new_static("cursor/model"))
			.thread(Thread::builder().items(Vec::new()).build())
			.tools(Vec::new())
			.model_policy(Arc::new(ResolvedModelPolicy {
				headers: ResolvedModelHeaders(BTreeMap::from([
					("X-Trace".into(), "abc".into()),
					("TE".into(), "trailers".into()),
				])),
				..ResolvedModelPolicy::default()
			}))
			.build();
		let headers = cursor_request_headers(&request, http::HeaderMap::new()).expect("headers");
		assert_eq!(headers["x-trace"], "abc");
		assert_eq!(headers["te"], "trailers");
	}

	#[test]
	fn cursor_transport_owned_and_http1_headers_cannot_be_overridden() {
		let request = ChatRequest::builder()
			.model(Str::new_static("cursor/model"))
			.thread(Thread::builder().items(Vec::new()).build())
			.tools(Vec::new())
			.model_policy(Arc::new(ResolvedModelPolicy {
				headers: ResolvedModelHeaders(BTreeMap::from([
					(":path".into(), "/evil".into()),
					("Authorization".into(), "Bearer stolen".into()),
					("Connection".into(), "keep-alive".into()),
					("HTTP2-Settings".into(), "forged".into()),
					("Keep-Alive".into(), "timeout=5".into()),
					("Proxy-Connection".into(), "keep-alive".into()),
					("Content-Length".into(), "999".into()),
					("Content-Type".into(), "text/plain".into()),
					("Host".into(), "evil.example.com".into()),
					("TE".into(), "gzip".into()),
					("Transfer-Encoding".into(), "chunked".into()),
					("Upgrade".into(), "h2c".into()),
					("X-Cursor-Client-Version".into(), "forged".into()),
					("X-Ghost-Mode".into(), "false".into()),
					("X-Trace".into(), "kept".into()),
				])),
				..ResolvedModelPolicy::default()
			}))
			.build();
		let mut auth = http::HeaderMap::new();
		auth.insert(header::AUTHORIZATION, http::HeaderValue::from_static("Bearer sealed"));
		let headers = cursor_request_headers(&request, auth).expect("headers");
		assert_eq!(headers[header::AUTHORIZATION], "Bearer sealed");
		assert_eq!(headers[header::CONTENT_TYPE], "application/connect+proto");
		assert_eq!(headers["x-cursor-client-version"], CURSOR_CLIENT_VERSION);
		assert_eq!(headers["x-ghost-mode"], "true");
		assert_eq!(headers["x-trace"], "kept");
		for name in [
			"connection",
			"content-length",
			"host",
			"http2-settings",
			"keep-alive",
			"proxy-connection",
			"te",
			"transfer-encoding",
			"upgrade",
		] {
			assert!(!headers.contains_key(name), "{name}");
		}
	}

	#[test]
	fn model_policy_sets_both_cursor_max_mode_fields() {
		for enabled in [false, true] {
			let policy = Arc::new(ResolvedModelPolicy {
				cursor_max_mode: Some(enabled),
				..ResolvedModelPolicy::default()
			});
			let request = ChatRequest::builder()
				.model(Str::new_static("cursor/model"))
				.thread(Thread::builder().items(Vec::new()).build())
				.tools(Vec::new())
				.provider_options(Props::default())
				.model_policy(policy)
				.build();
			let (message, _) = assemble_request(&request).expect("Cursor request");
			let super::wire::agent_client_message::Message::RunRequest(run) =
				message.message.expect("run request")
			else {
				panic!("expected run request");
			};
			assert_eq!(run.model_details.expect("model details").max_mode, Some(enabled));
			assert_eq!(run.requested_model.expect("requested model").max_mode, enabled);
		}
	}

	#[test]
	fn shell_exec_args_omit_unset_optionals_and_keep_set_values() {
		fn invoke(working_directory: &str, timeout: i32) -> Invoke {
			invoke_from_exec(
				super::wire::ExecServerMessage {
					id:      7,
					exec_id: "exec-7".to_owned(),
					message: Some(super::wire::exec_server_message::Message::ShellStreamArgs(
						super::wire::ShellArgs {
							command: "echo hi".to_owned(),
							working_directory: working_directory.to_owned(),
							timeout,
							tool_call_id: "call-shell".to_owned(),
							..Default::default()
						},
					)),
					..Default::default()
				},
				&mut CursorDecodeState::default(),
			)
			.expect("shell invocation")
		}

		let omitted = invoke("", 0);
		let omitted_args: serde_json::Value = serde_json::from_slice(
			&omitted.tool_call.expect("canonical shell tool call").args_json,
		)
		.expect("shell args JSON");
		assert_eq!(
			omitted_args,
			serde_json::json!({ "command": "echo hi", "tool_call_id": "call-shell" })
		);
		assert!(omitted_args.get("working_directory").is_none());
		assert!(omitted_args.get("timeout").is_none());

		let kept = invoke("/tmp", 12);
		let kept_args: serde_json::Value =
			serde_json::from_slice(&kept.tool_call.expect("canonical shell tool call").args_json)
				.expect("shell args JSON");
		assert_eq!(
			kept_args,
			serde_json::json!({
				"command": "echo hi",
				"working_directory": "/tmp",
				"timeout": 12,
				"tool_call_id": "call-shell",
			})
		);
	}

	#[test]
	fn all_five_statuses_emit_exit_result_and_close() {
		for outcome in [
			ExecOutcome::Exited,
			ExecOutcome::Failed,
			ExecOutcome::Rejected,
			ExecOutcome::Denied,
			ExecOutcome::Timeout,
		] {
			let mut framer = InvocationFramer::new(context());
			let frames = framer.complete(&complete(outcome));
			let result = frames.iter().find_map(|frame| match &frame.message {
				Some(super::wire::agent_client_message::Message::ExecClientMessage(message)) => {
					match &message.message {
						Some(super::wire::exec_client_message::Message::ShellResult(result)) => {
							result.result.as_ref()
						},
						_ => None,
					}
				},
				_ => None,
			});
			assert!(matches!(
				(outcome, result),
				(ExecOutcome::Exited, Some(super::wire::shell_result::Result::Success(_)))
					| (ExecOutcome::Failed, Some(super::wire::shell_result::Result::Failure(_)))
					| (ExecOutcome::Rejected, Some(super::wire::shell_result::Result::Rejected(_)))
					| (
						ExecOutcome::Denied,
						Some(super::wire::shell_result::Result::PermissionDenied(_))
					) | (ExecOutcome::Timeout, Some(super::wire::shell_result::Result::Timeout(_)))
			));
			assert!(matches!(
				frames.last().and_then(|frame| frame.message.as_ref()),
				Some(super::wire::agent_client_message::Message::ExecClientControlMessage(_))
			));
		}
	}

	#[test]
	fn ansi_escape_is_not_split_across_frames() {
		let mut framer = InvocationFramer::new(context());
		let first = InvokeInput::builder()
			.invocation_id(invocation_id(&context()))
			.payload(InvokePayload::Chunk(
				InvokeChunk::builder()
					.channel(InvokeChannel::Stdout)
					.data(Bytes::from_static(b"plain\x1b[31"))
					.build(),
			))
			.build();
		let second = InvokeInput::builder()
			.invocation_id(invocation_id(&context()))
			.payload(InvokePayload::Chunk(
				InvokeChunk::builder()
					.channel(InvokeChannel::Stdout)
					.data(Bytes::from_static(b"mred\x1b[0m"))
					.build(),
			))
			.build();
		let first_frames = framer.input(&first);
		let first_data = match &first_frames[0].message {
			Some(super::wire::agent_client_message::Message::ExecClientMessage(message)) => {
				match &message.message {
					Some(super::wire::exec_client_message::Message::ShellStream(stream)) => {
						match &stream.event {
							Some(super::wire::shell_stream::Event::Stdout(stdout)) => stdout.data.as_str(),
							_ => panic!("expected stdout"),
						}
					},
					_ => panic!("expected shell stream"),
				}
			},
			_ => panic!("expected exec client message"),
		};
		assert_eq!(first_data, "plain");
		assert_eq!(framer.input(&second).len(), 1);
	}

	#[test]
	fn shell_stream_round_trip_preserves_channels_and_completion_order() {
		let mut framer = InvocationFramer::new(context());
		let stdout = InvokeInput::builder()
			.invocation_id(Str::new_static("exec-7"))
			.payload(InvokePayload::Chunk(
				InvokeChunk::builder()
					.channel(InvokeChannel::Stdout)
					.data(Bytes::from_static(b"out\\n"))
					.build(),
			))
			.build();
		let stderr = InvokeInput::builder()
			.invocation_id(Str::new_static("exec-7"))
			.payload(InvokePayload::Chunk(
				InvokeChunk::builder()
					.channel(InvokeChannel::Stderr)
					.data(Bytes::from_static(b"err\\n"))
					.build(),
			))
			.build();
		assert_eq!(framer.input(&stdout).len(), 1);
		assert_eq!(framer.input(&stderr).len(), 1);
		let frames = framer.complete(&complete(ExecOutcome::Exited));
		assert_eq!(frames.len(), 3);
		assert!(matches!(
			frames[0].message,
			Some(super::wire::agent_client_message::Message::ExecClientMessage(_))
		));
		assert!(matches!(
			frames[2].message,
			Some(super::wire::agent_client_message::Message::ExecClientControlMessage(_))
		));
	}

	#[test]
	fn server_abort_maps_to_invoke_cancel() {
		let server = super::wire::AgentServerMessage {
			message: Some(super::wire::agent_server_message::Message::ExecServerControlMessage(
				super::wire::ExecServerControlMessage {
					message: Some(super::wire::exec_server_control_message::Message::Abort(
						super::wire::ExecServerAbort { id: 7 },
					)),
				},
			)),
		};
		let events = decode_server_message(server, &mut CursorDecodeState::default()).unwrap();
		assert!(matches!(
			events.as_slice(),
			[TurnEvent::InvokeCancel { invocation_id }] if invocation_id == "7"
		));
	}

	#[test]
	fn heartbeat_is_correlated_to_outstanding_exec() {
		let heartbeat = InvocationFramer::new(context()).heartbeat();
		assert!(matches!(
			heartbeat.message,
			Some(super::wire::agent_client_message::Message::ExecClientControlMessage(
				super::wire::ExecClientControlMessage {
					message: Some(super::wire::exec_client_control_message::Message::Heartbeat(
						super::wire::ExecClientHeartbeat { id: 7 }
					)),
				}
			))
		));
	}

	#[test]
	fn missing_executor_fails_admission() {
		assert!(matches!(require_executor(None), Err(omp_llm_types::Error::Unsupported(_))));
	}

	#[test]
	fn missing_status_uses_throw_then_close() {
		let mut framer = InvocationFramer::new(context());
		let completion = InvokeComplete::builder()
			.invocation_id(Str::new_static("exec-7"))
			.vendor(Bytes::new())
			.props(Props::default())
			.build();
		assert_eq!(framer.complete(&completion).len(), 2);
	}

	#[test]
	fn tool_call_frames_preserve_argument_lexemes_usage_and_stop_reason() {
		use super::wire;

		let messages = [
			wire::interaction_update::Message::ToolCallStarted(wire::ToolCallStartedUpdate {
				call_id: "call-read".to_owned(),
				tool_call: Some(wire::ToolCall {
					tool: Some(wire::tool_call::Tool::ReadToolCall(wire::ReadToolCall::default())),
					..Default::default()
				}),
				..Default::default()
			}),
			wire::interaction_update::Message::PartialToolCall(wire::PartialToolCallUpdate {
				call_id: "call-read".to_owned(),
				args_text_delta: "{\"pa".to_owned(),
				..Default::default()
			}),
			wire::interaction_update::Message::PartialToolCall(wire::PartialToolCallUpdate {
				call_id: "call-read".to_owned(),
				args_text_delta: "th\":\"package.json\"}".to_owned(),
				..Default::default()
			}),
			wire::interaction_update::Message::ToolCallCompleted(wire::ToolCallCompletedUpdate {
				call_id: "call-read".to_owned(),
				..Default::default()
			}),
			wire::interaction_update::Message::TokenDelta(wire::TokenDeltaUpdate { tokens: 8 }),
			wire::interaction_update::Message::TurnEnded(wire::TurnEndedUpdate {}),
		];
		let mut state = CursorDecodeState::default();
		let events = messages
			.into_iter()
			.flat_map(|message| {
				decode_server_message(
					wire::AgentServerMessage {
						message: Some(wire::agent_server_message::Message::InteractionUpdate(Box::new(
							wire::InteractionUpdate { message: Some(message) },
						))),
					},
					&mut state,
				)
				.unwrap()
			})
			.collect::<Vec<_>>();

		assert!(matches!(
			&events[0],
			TurnEvent::PartStart {
				kind: StreamPartKind::ToolCall,
				tool_call_id,
				tool_name,
				..
			} if tool_call_id == "call-read" && tool_name == "read"
		));
		let arguments = events
			.iter()
			.filter_map(|event| match event {
				TurnEvent::PartDelta { chunk, .. } => Some(chunk.as_ref()),
				_ => None,
			})
			.flatten()
			.copied()
			.collect::<Vec<_>>();
		assert_eq!(arguments, br#"{"path":"package.json"}"#);
		assert!(matches!(events[3], TurnEvent::PartEnd { .. }));
		assert!(matches!(
			events.last(),
			Some(TurnEvent::Outcome(outcome))
				if outcome.stop == StopReason::ToolUse
					&& outcome.usage.as_ref().is_some_and(|usage| usage.output_tokens == 8)
		));
	}

	struct PendingExecutor;

	#[async_trait::async_trait]
	impl Executor for PendingExecutor {
		async fn invoke(
			&self,
			_invoke: Invoke,
			_inputs: flume::Sender<InvokeInput>,
		) -> InvokeComplete {
			std::future::pending().await
		}
	}

	#[tokio::test]
	async fn invocation_deadline_emits_canonical_timeout() {
		let invocation = Invoke::builder()
			.invocation_id(Str::new_static("exec-7"))
			.name(Str::new_static("cursor/control"))
			.vendor(Bytes::new())
			.timeout_ms(1)
			.props(Props::default())
			.build();
		let (_cancel_tx, cancel_rx) = oneshot::channel();
		let (client_tx, _client_rx) = flume::bounded(1);
		let (terminal_tx, terminal_rx) = flume::bounded(1);
		tokio::time::timeout(
			Duration::from_millis(100),
			drive_invocation_to(
				Arc::new(PendingExecutor),
				invocation,
				context(),
				cancel_rx,
				client_tx,
				terminal_tx,
			),
		)
		.await
		.expect("invocation driver must stop at its deadline");
		assert_eq!(terminal_rx.recv().unwrap().kind, TurnErrorKind::InvokeTimeout);
	}
}
