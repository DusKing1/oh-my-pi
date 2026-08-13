use std::{
	collections::HashMap,
	sync::{
		Arc, Weak,
		atomic::{AtomicU64, Ordering},
	},
};

use bytes::Bytes;
use flume::{Receiver, Sender};
use omp_core::Str;
use omp_proto::{
	blob::v1::{
		Chunk, DeleteRequest, DeleteResponse, GetRequest, PutResponse, StatRequest, StatResponse,
	},
	env::v1::{
		ArgText, ArgsCommitted, AttachOutput, BlobGetComplete, CancelRequest, ClientFrame,
		ClientHello, CloseSessionRequest, CloseSessionResponse, CommitBlobPut, EventStreamError,
		ExecRequest, ExecStarted, ExitEvent, Interrupt, InvokeAccepted, InvokeTool, ListProcesses,
		OpenSessionRequest, OpenSessionResponse, OutputAttached, OutputFrame, ProcessCommandAccepted,
		ProcessList, ProcessOutput, ProcessStarted, ProcessStateEvent, ProtocolError, SendInput,
		ServerFrame, ServerHello, SignalProcess, SignalRequest, StartProcess, StdinFrame,
		StopProcess, Update, Verdict, cancel_request, client_frame, server_frame,
	},
};
use parking_lot::Mutex;
use thiserror::Error;

use crate::guard::RunGuard;

/// A client-side environment protocol failure.
#[derive(Debug, Error)]
pub enum ClientError {
	/// The frame transport closed before the operation completed.
	#[error("environment frame transport closed")]
	TransportClosed,
	/// All nonzero request identifiers have been consumed.
	#[error("environment request identifier space exhausted")]
	RequestIdExhausted,
	/// The server rejected the request.
	#[error("environment protocol error: {0:?}")]
	Protocol(ProtocolError),
	/// A response did not have the body required by the typed operation.
	#[error("unexpected environment response while waiting for {expected}")]
	UnexpectedResponse {
		/// The response body expected by the operation.
		expected: &'static str,
	},
}

/// The client half of a transport-neutral bidirectional `env/v1` frame channel.
///
/// A UDS, mTLS, or other remote transport can decode frames into `incoming`
/// and encode frames received from `outgoing`. [`Self::in_process`] creates the
/// same boundary from flume channels for a colocated environment host.
#[derive(Clone, Debug)]
pub struct EnvClient {
	inner: Arc<ClientInner>,
}

#[derive(Debug)]
struct ClientInner {
	outgoing: Sender<ClientFrame>,
	pending:  Mutex<HashMap<u64, Sender<ServerFrame>>>,
	hello:    Mutex<Option<Sender<ServerFrame>>>,
	events:   Receiver<ServerFrame>,
	next_id:  AtomicU64,
	cancel:   Sender<u64>,
}

/// The server half of an in-process `env/v1` frame transport.
///
/// This type contains transport endpoints only. It does not implement or own
/// any environment resources.
#[derive(Debug)]
pub struct InProcessEnvTransport {
	requests:  Receiver<ClientFrame>,
	responses: Sender<ServerFrame>,
}

/// A correlated stream of raw server frames for one request.
#[derive(Debug)]
pub struct RequestStream {
	request_id: u64,
	receiver:   Receiver<ServerFrame>,
	client:     Weak<ClientInner>,
	finished:   bool,
}

/// One open tool invocation and its correlated event stream.
#[derive(Debug)]
pub struct Invocation {
	client:        EnvClient,
	invocation_id: Str,
	stream:        RequestStream,
	guard:         Option<RunGuard>,
}

/// A typed event on a tool invocation stream.
#[derive(Debug)]
pub enum InvocationEvent {
	/// The environment accepted the invocation channel.
	Accepted(InvokeAccepted),
	/// Serialized typed progress from the executor.
	Update(Update),
	/// The terminal structured tool outcome and canonical model-facing parts.
	Verdict(Verdict),
	/// Continuity of the invocation event stream was lost.
	StreamError(EventStreamError),
}

/// One command running inside a server-owned exec session.
#[derive(Debug)]
pub struct ExecRun {
	client: EnvClient,
	stream: RequestStream,
	guard:  Option<RunGuard>,
}

/// A typed event on an exec request stream.
#[derive(Debug)]
pub enum ExecEvent {
	/// The command was created and has an exec identifier.
	Started(ExecStarted),
	/// Ordered stdout, stderr, or PTY bytes.
	Output(OutputFrame),
	/// The terminal command status.
	Exit(ExitEvent),
	/// Continuity of the exec event stream was lost.
	StreamError(EventStreamError),
}

/// A correlated named-process output attachment.
#[derive(Debug)]
pub struct ProcessAttachment {
	stream: RequestStream,
}

/// One event from a named-process output attachment.
#[derive(Debug)]
pub enum ProcessAttachmentEvent {
	/// The server established the attachment and identified its generation.
	Attached(OutputAttached),
	/// Ordered output from the attached process generation.
	Output(ProcessOutput),
	/// A lifecycle transition for the named process.
	State(ProcessStateEvent),
	/// Continuity of the attached output stream was lost.
	StreamError(EventStreamError),
}

/// A streaming blob download.
#[derive(Debug)]
pub struct BlobDownload {
	stream: RequestStream,
}

/// One event from a blob download.
#[derive(Debug)]
pub enum BlobDownloadEvent {
	/// The next ordered bytes in the download.
	Chunk(Chunk),
	/// The successful terminal download marker.
	Complete(BlobGetComplete),
}

/// A streaming, correlated blob upload.
#[derive(Debug)]
pub struct BlobUpload {
	client:     EnvClient,
	request_id: u64,
	stream:     RequestStream,
}

impl EnvClient {
	/// Builds a client over decoded bidirectional frame channels.
	///
	/// `outgoing` carries client frames to the transport and `incoming` carries
	/// decoded server frames back. A small dispatcher thread performs request
	/// correlation; no async runtime or world-resource implementation is owned
	/// by this crate.
	#[must_use]
	pub fn from_channels(outgoing: Sender<ClientFrame>, incoming: Receiver<ServerFrame>) -> Self {
		let (events_tx, events) = flume::unbounded();
		let (cancel, cancellations) = flume::unbounded();
		let inner = Arc::new(ClientInner {
			outgoing: outgoing.clone(),
			pending: Mutex::new(HashMap::new()),
			hello: Mutex::new(None),
			events,
			next_id: AtomicU64::new(1),
			cancel,
		});

		let router = Arc::downgrade(&inner);
		let _ = std::thread::spawn(move || route_responses(router, incoming, events_tx));
		let _ = std::thread::spawn(move || route_cancellations(cancellations, outgoing));
		Self { inner }
	}

	/// Creates an in-process client/server frame channel.
	///
	/// Capacity zero selects unbounded channels. A nonzero capacity applies
	/// backpressure to ordinary asynchronous frame sends; guard cancellation is
	/// first queued on a separate unbounded control channel so drop never
	/// blocks.
	#[must_use]
	pub fn in_process(capacity: usize) -> (Self, InProcessEnvTransport) {
		let (requests_tx, requests) = channel(capacity);
		let (responses, responses_rx) = channel(capacity);
		(Self::from_channels(requests_tx, responses_rx), InProcessEnvTransport {
			requests,
			responses,
		})
	}

	/// Performs the request-id-zero protocol handshake.
	pub async fn hello(&self, hello: ClientHello) -> Result<ServerHello, ClientError> {
		let (sender, receiver) = flume::bounded(1);
		let mut slot = self.inner.hello.lock();
		if slot.is_some() {
			return Err(ClientError::UnexpectedResponse { expected: "a single in-flight hello" });
		}
		*slot = Some(sender);
		drop(slot);
		let send = self
			.inner
			.outgoing
			.send_async(ClientFrame {
				request_id: 0,
				body: Some(client_frame::Body::Hello(hello)),
				..ClientFrame::default()
			})
			.await;
		if send.is_err() {
			self.inner.hello.lock().take();
			return Err(ClientError::TransportClosed);
		}
		let frame = receiver
			.recv_async()
			.await
			.map_err(|_| ClientError::TransportClosed)?;
		match frame.body {
			Some(server_frame::Body::Hello(response)) => Ok(response),
			Some(server_frame::Body::Error(error)) => Err(ClientError::Protocol(error)),
			_ => Err(ClientError::UnexpectedResponse { expected: "ServerHello" }),
		}
	}

	/// Returns the receiver for unsolicited request-id-zero server events.
	///
	/// Clones share one queue; callers should normally keep a single receiver
	/// and distribute events according to application policy.
	#[must_use]
	pub fn server_events(&self) -> Receiver<ServerFrame> {
		self.inner.events.clone()
	}

	/// Opens a tool invocation before its arguments have committed.
	pub async fn invoke(&self, request: InvokeTool) -> Result<Invocation, ClientError> {
		let invocation_id = Str::from(request.invocation_id.as_str());
		let (stream, guard) = self
			.open_guarded(client_frame::Body::InvokeTool(request))
			.await?;
		Ok(Invocation { client: self.clone(), invocation_id, stream, guard: Some(guard) })
	}

	/// Opens a persistent, server-owned exec session.
	pub async fn open_session(
		&self,
		request: OpenSessionRequest,
	) -> Result<OpenSessionResponse, ClientError> {
		match self
			.one_shot(client_frame::Body::OpenSession(request))
			.await?
		{
			server_frame::Body::SessionOpened(response) => Ok(response),
			_ => Err(ClientError::UnexpectedResponse { expected: "OpenSessionResponse" }),
		}
	}

	/// Explicitly closes a persistent exec session.
	pub async fn close_session(
		&self,
		request: CloseSessionRequest,
	) -> Result<CloseSessionResponse, ClientError> {
		match self
			.one_shot(client_frame::Body::CloseSession(request))
			.await?
		{
			server_frame::Body::SessionClosed(response) => Ok(response),
			_ => Err(ClientError::UnexpectedResponse { expected: "CloseSessionResponse" }),
		}
	}

	/// Starts one guarded command inside a persistent session.
	pub async fn exec(&self, request: ExecRequest) -> Result<ExecRun, ClientError> {
		let (stream, guard) = self.open_guarded(client_frame::Body::Exec(request)).await?;
		Ok(ExecRun { client: self.clone(), stream, guard: Some(guard) })
	}

	/// Starts or replaces a server-owned named process.
	pub async fn start_process(&self, request: StartProcess) -> Result<ProcessStarted, ClientError> {
		match self
			.one_shot(client_frame::Body::StartProcess(request))
			.await?
		{
			server_frame::Body::ProcessStarted(response) => Ok(response),
			_ => Err(ClientError::UnexpectedResponse { expected: "ProcessStarted" }),
		}
	}

	/// Lists the server-owned named processes visible to this environment.
	pub async fn list_processes(&self, request: ListProcesses) -> Result<ProcessList, ClientError> {
		match self
			.one_shot(client_frame::Body::ListProcesses(request))
			.await?
		{
			server_frame::Body::ProcessList(response) => Ok(response),
			_ => Err(ClientError::UnexpectedResponse { expected: "ProcessList" }),
		}
	}

	/// Attaches to ordered output and state events for a named process.
	pub async fn attach_output(
		&self,
		request: AttachOutput,
	) -> Result<ProcessAttachment, ClientError> {
		let stream = self.open(client_frame::Body::AttachOutput(request)).await?;
		Ok(ProcessAttachment { stream })
	}

	/// Sends bytes or EOF to a server-owned named process.
	pub async fn send_process_input(
		&self,
		request: SendInput,
	) -> Result<ProcessCommandAccepted, ClientError> {
		self
			.process_command(client_frame::Body::SendInput(request))
			.await
	}

	/// Sends a signal to a server-owned named process.
	pub async fn signal_process(
		&self,
		request: SignalProcess,
	) -> Result<ProcessCommandAccepted, ClientError> {
		self
			.process_command(client_frame::Body::SignalProcess(request))
			.await
	}

	/// Stops a server-owned named process.
	pub async fn stop_process(
		&self,
		request: StopProcess,
	) -> Result<ProcessCommandAccepted, ClientError> {
		self
			.process_command(client_frame::Body::StopProcess(request))
			.await
	}

	/// Checks whether a content-addressed blob is present.
	pub async fn blob_stat(&self, request: StatRequest) -> Result<StatResponse, ClientError> {
		match self.one_shot(client_frame::Body::BlobStat(request)).await? {
			server_frame::Body::BlobStat(response) => Ok(response),
			_ => Err(ClientError::UnexpectedResponse { expected: "StatResponse" }),
		}
	}

	/// Starts a streaming blob download.
	pub async fn blob_get(&self, request: GetRequest) -> Result<BlobDownload, ClientError> {
		let stream = self.open(client_frame::Body::BlobGet(request)).await?;
		Ok(BlobDownload { stream })
	}

	/// Starts a streaming blob upload.
	///
	/// Call [`BlobUpload::send_chunk`] in order and finish with
	/// [`BlobUpload::commit`]. Dropping an uncommitted upload only abandons its
	/// client-side response route; blob visibility remains gated by the commit
	/// frame.
	pub async fn blob_put(&self) -> Result<BlobUpload, ClientError> {
		let request_id = self.allocate_request_id()?;
		let stream = self.register(request_id);
		Ok(BlobUpload { client: self.clone(), request_id, stream })
	}

	/// Deletes one content-addressed blob.
	pub async fn blob_delete(&self, request: DeleteRequest) -> Result<DeleteResponse, ClientError> {
		match self
			.one_shot(client_frame::Body::BlobDelete(request))
			.await?
		{
			server_frame::Body::BlobDeleted(response) => Ok(response),
			_ => Err(ClientError::UnexpectedResponse { expected: "DeleteResponse" }),
		}
	}

	async fn process_command(
		&self,
		body: client_frame::Body,
	) -> Result<ProcessCommandAccepted, ClientError> {
		match self.one_shot(body).await? {
			server_frame::Body::ProcessCommandAccepted(response) => Ok(response),
			_ => Err(ClientError::UnexpectedResponse { expected: "ProcessCommandAccepted" }),
		}
	}

	async fn one_shot(&self, body: client_frame::Body) -> Result<server_frame::Body, ClientError> {
		let mut stream = self.open(body).await?;
		let frame = stream.next().await?.ok_or(ClientError::TransportClosed)?;
		stream.finish();
		response_body(frame)
	}

	async fn open(&self, body: client_frame::Body) -> Result<RequestStream, ClientError> {
		let request_id = self.allocate_request_id()?;
		let stream = self.register(request_id);
		if self.send(request_id, body).await.is_err() {
			stream.unregister();
			return Err(ClientError::TransportClosed);
		}
		Ok(stream)
	}

	async fn open_guarded(
		&self,
		body: client_frame::Body,
	) -> Result<(RequestStream, RunGuard), ClientError> {
		let request_id = self.allocate_request_id()?;
		let stream = self.register(request_id);
		let guard = RunGuard::new(request_id, self.inner.cancel.clone());
		if self.send(request_id, body).await.is_err() {
			stream.unregister();
			guard.relinquish();
			return Err(ClientError::TransportClosed);
		}
		Ok((stream, guard))
	}

	fn register(&self, request_id: u64) -> RequestStream {
		let (sender, receiver) = flume::unbounded();
		self.inner.pending.lock().insert(request_id, sender);
		RequestStream { request_id, receiver, client: Arc::downgrade(&self.inner), finished: false }
	}

	async fn send(&self, request_id: u64, body: client_frame::Body) -> Result<(), ClientError> {
		self
			.inner
			.outgoing
			.send_async(ClientFrame { request_id, body: Some(body), ..ClientFrame::default() })
			.await
			.map_err(|_| ClientError::TransportClosed)
	}

	fn allocate_request_id(&self) -> Result<u64, ClientError> {
		self
			.inner
			.next_id
			.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |request_id| request_id.checked_add(1))
			.map_err(|_| ClientError::RequestIdExhausted)
	}
}

impl InProcessEnvTransport {
	/// Receives the next client frame asynchronously.
	pub async fn recv(&self) -> Result<ClientFrame, flume::RecvError> {
		self.requests.recv_async().await
	}

	/// Sends one server frame asynchronously.
	pub async fn send(&self, frame: ServerFrame) -> Result<(), flume::SendError<ServerFrame>> {
		self.responses.send_async(frame).await
	}

	/// Splits this transport into the server's receive and send endpoints.
	#[must_use]
	pub fn into_parts(self) -> (Receiver<ClientFrame>, Sender<ServerFrame>) {
		(self.requests, self.responses)
	}
}

impl RequestStream {
	/// Returns the correlation identifier carried by every frame in this stream.
	#[must_use]
	pub const fn request_id(&self) -> u64 {
		self.request_id
	}

	/// Waits for the next correlated server frame.
	pub async fn next(&mut self) -> Result<Option<ServerFrame>, ClientError> {
		match self.receiver.recv_async().await {
			Ok(frame) => Ok(Some(frame)),
			Err(_) if self.finished => Ok(None),
			Err(_) => Err(ClientError::TransportClosed),
		}
	}

	/// Explicitly cancels this request and closes its local response route.
	///
	/// Unlike [`RunGuard`], a raw request stream is not cancelled on drop. This
	/// keeps the stream returned by detached work safe to discard; callers that
	/// own an ordinary long-lived request can cancel it explicitly here.
	pub fn cancel(mut self) {
		if let Some(client) = self.client.upgrade() {
			let _ = client.cancel.try_send(self.request_id);
		}
		self.finish();
	}

	fn finish(&mut self) {
		self.unregister();
		self.finished = true;
	}

	fn unregister(&self) {
		if let Some(client) = self.client.upgrade() {
			client.pending.lock().remove(&self.request_id);
		}
	}
}

impl Drop for RequestStream {
	fn drop(&mut self) {
		self.unregister();
	}
}

impl Invocation {
	/// Returns the invocation's logical identifier.
	#[must_use]
	pub fn invocation_id(&self) -> &str {
		&self.invocation_id
	}

	/// Returns the request-scoped cancellation guard.
	#[must_use]
	pub fn guard(&self) -> &RunGuard {
		self
			.guard
			.as_ref()
			.expect("invocation guard exists until relinquished")
	}

	/// Relays one raw provider argument fragment without validation.
	pub async fn arg_text(&self, fragment: Str) -> Result<(), ClientError> {
		self
			.client
			.send(
				self.stream.request_id,
				client_frame::Body::ArgText(ArgText {
					invocation_id: self.invocation_id.to_string(),
					fragment: fragment.to_string(),
					..ArgText::default()
				}),
			)
			.await
	}

	/// Sends the exact committed argument bytes, authorizing effects env-side.
	pub async fn commit_args(&self, raw: Bytes) -> Result<(), ClientError> {
		self
			.client
			.send(
				self.stream.request_id,
				client_frame::Body::ArgsCommitted(ArgsCommitted {
					invocation_id: self.invocation_id.to_string(),
					raw,
					..ArgsCommitted::default()
				}),
			)
			.await
	}

	/// Sends cooperative interrupt steering to this invocation only.
	pub async fn interrupt(&self, reason: Str) -> Result<(), ClientError> {
		self
			.client
			.send(
				self.stream.request_id,
				client_frame::Body::Interrupt(Interrupt {
					invocation_id: self.invocation_id.to_string(),
					reason: reason.to_string(),
					..Interrupt::default()
				}),
			)
			.await
	}

	/// Waits for the next typed invocation event.
	pub async fn next_event(&mut self) -> Result<Option<InvocationEvent>, ClientError> {
		let Some(frame) = self.stream.next().await? else {
			return Ok(None);
		};
		let body = match response_body(frame) {
			Ok(body) => body,
			Err(error) => {
				self.complete();
				return Err(error);
			},
		};
		match body {
			server_frame::Body::InvocationAccepted(event) => {
				Ok(Some(InvocationEvent::Accepted(event)))
			},
			server_frame::Body::Update(event) => Ok(Some(InvocationEvent::Update(event))),
			server_frame::Body::Verdict(event) => {
				self.complete();
				Ok(Some(InvocationEvent::Verdict(event)))
			},
			server_frame::Body::EventStreamError(event) => {
				self.stream.finish();
				Ok(Some(InvocationEvent::StreamError(event)))
			},
			_ => Err(ClientError::UnexpectedResponse { expected: "invocation event" }),
		}
	}

	/// Explicitly leaves detached work owned by the environment service.
	///
	/// The returned stream can continue observing its terminal event, but its
	/// drop no longer requests cancellation.
	#[must_use]
	pub fn relinquish(mut self) -> RequestStream {
		if let Some(guard) = self.guard.take() {
			guard.relinquish();
		}
		self.stream
	}

	fn complete(&mut self) {
		if let Some(guard) = self.guard.take() {
			guard.relinquish();
		}
		self.stream.finish();
	}
}

impl ExecRun {
	/// Returns the request-scoped command cancellation guard.
	#[must_use]
	pub fn guard(&self) -> &RunGuard {
		self
			.guard
			.as_ref()
			.expect("exec guard exists until relinquished")
	}

	/// Writes stdin bytes or EOF to this command.
	pub async fn stdin(&self, frame: StdinFrame) -> Result<(), ClientError> {
		self
			.client
			.send(self.stream.request_id, client_frame::Body::Stdin(frame))
			.await
	}

	/// Sends a signal to this command.
	pub async fn signal(&self, request: SignalRequest) -> Result<(), ClientError> {
		self
			.client
			.send(self.stream.request_id, client_frame::Body::Signal(request))
			.await
	}

	/// Resizes this command's PTY.
	pub async fn resize(
		&self,
		request: omp_proto::env::v1::ResizeRequest,
	) -> Result<(), ClientError> {
		self
			.client
			.send(self.stream.request_id, client_frame::Body::Resize(request))
			.await
	}

	/// Waits for the next typed command event.
	pub async fn next_event(&mut self) -> Result<Option<ExecEvent>, ClientError> {
		let Some(frame) = self.stream.next().await? else {
			return Ok(None);
		};
		let body = match response_body(frame) {
			Ok(body) => body,
			Err(error) => {
				self.complete();
				return Err(error);
			},
		};
		match body {
			server_frame::Body::ExecStarted(event) => Ok(Some(ExecEvent::Started(event))),
			server_frame::Body::Output(event) => Ok(Some(ExecEvent::Output(event))),
			server_frame::Body::Exit(event) => {
				self.complete();
				Ok(Some(ExecEvent::Exit(event)))
			},
			server_frame::Body::EventStreamError(event) => {
				self.stream.finish();
				Ok(Some(ExecEvent::StreamError(event)))
			},
			_ => Err(ClientError::UnexpectedResponse { expected: "exec event" }),
		}
	}

	/// Explicitly leaves a detached command owned by the environment service.
	#[must_use]
	pub fn relinquish(mut self) -> RequestStream {
		if let Some(guard) = self.guard.take() {
			guard.relinquish();
		}
		self.stream
	}

	fn complete(&mut self) {
		if let Some(guard) = self.guard.take() {
			guard.relinquish();
		}
		self.stream.finish();
	}
}

impl ProcessAttachment {
	/// Waits for the next ordered attachment event.
	pub async fn next_event(&mut self) -> Result<Option<ProcessAttachmentEvent>, ClientError> {
		let Some(frame) = self.stream.next().await? else {
			return Ok(None);
		};
		let body = match response_body(frame) {
			Ok(body) => body,
			Err(error) => {
				self.stream.finish();
				return Err(error);
			},
		};
		match body {
			server_frame::Body::OutputAttached(event) => {
				Ok(Some(ProcessAttachmentEvent::Attached(event)))
			},
			server_frame::Body::ProcessOutput(event) => {
				Ok(Some(ProcessAttachmentEvent::Output(event)))
			},
			server_frame::Body::ProcessState(event) => Ok(Some(ProcessAttachmentEvent::State(event))),
			server_frame::Body::EventStreamError(event) => {
				self.stream.finish();
				Ok(Some(ProcessAttachmentEvent::StreamError(event)))
			},
			_ => Err(ClientError::UnexpectedResponse { expected: "process attachment event" }),
		}
	}

	/// Stops the server-side output attachment.
	pub fn cancel(self) {
		self.stream.cancel();
	}
}

impl BlobDownload {
	/// Waits for the next ordered chunk or terminal completion marker.
	pub async fn next_event(&mut self) -> Result<Option<BlobDownloadEvent>, ClientError> {
		let Some(frame) = self.stream.next().await? else {
			return Ok(None);
		};
		let body = match response_body(frame) {
			Ok(body) => body,
			Err(error) => {
				self.stream.finish();
				return Err(error);
			},
		};
		match body {
			server_frame::Body::BlobChunk(chunk) => Ok(Some(BlobDownloadEvent::Chunk(chunk))),
			server_frame::Body::BlobGetComplete(complete) => {
				self.stream.finish();
				Ok(Some(BlobDownloadEvent::Complete(complete)))
			},
			_ => Err(ClientError::UnexpectedResponse { expected: "blob chunk or completion" }),
		}
	}

	/// Stops this download before its completion marker.
	pub fn cancel(self) {
		self.stream.cancel();
	}
}

impl BlobUpload {
	/// Returns the correlation identifier shared by every upload frame.
	#[must_use]
	pub const fn request_id(&self) -> u64 {
		self.request_id
	}

	/// Sends the next ordered blob chunk.
	pub async fn send_chunk(&self, chunk: Chunk) -> Result<(), ClientError> {
		self
			.client
			.send(self.request_id, client_frame::Body::BlobPutChunk(chunk))
			.await
	}

	/// Cancels this upload without making its staged bytes visible.
	pub fn abort(self) {
		self.stream.cancel();
	}

	/// Commits the upload and waits for its content identity.
	pub async fn commit(mut self) -> Result<PutResponse, ClientError> {
		self
			.client
			.send(self.request_id, client_frame::Body::BlobPutCommit(CommitBlobPut::default()))
			.await?;
		let frame = self
			.stream
			.next()
			.await?
			.ok_or(ClientError::TransportClosed)?;
		self.stream.finish();
		match response_body(frame)? {
			server_frame::Body::BlobPut(response) => Ok(response),
			_ => Err(ClientError::UnexpectedResponse { expected: "PutResponse" }),
		}
	}
}

fn response_body(frame: ServerFrame) -> Result<server_frame::Body, ClientError> {
	match frame.body {
		Some(server_frame::Body::Error(error)) => Err(ClientError::Protocol(error)),
		Some(body) => Ok(body),
		None => Err(ClientError::UnexpectedResponse { expected: "nonempty server frame" }),
	}
}

fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
	if capacity == 0 {
		flume::unbounded()
	} else {
		flume::bounded(capacity)
	}
}

fn route_responses(
	client: Weak<ClientInner>,
	incoming: Receiver<ServerFrame>,
	events: Sender<ServerFrame>,
) {
	while let Ok(frame) = incoming.recv() {
		let Some(client) = client.upgrade() else {
			break;
		};
		if frame.request_id == 0 {
			let is_hello_response = matches!(
				frame.body.as_ref(),
				Some(server_frame::Body::Hello(_) | server_frame::Body::Error(_))
			);
			if is_hello_response && let Some(waiter) = client.hello.lock().take() {
				let _ = waiter.send(frame);
			} else {
				let _ = events.send(frame);
			}
			continue;
		}
		let target = client.pending.lock().get(&frame.request_id).cloned();
		if let Some(target) = target {
			let _ = target.send(frame);
		}
	}
	if let Some(client) = client.upgrade() {
		client.pending.lock().clear();
		client.hello.lock().take();
	}
}

fn route_cancellations(cancellations: Receiver<u64>, outgoing: Sender<ClientFrame>) {
	while let Ok(request_id) = cancellations.recv() {
		let frame = ClientFrame {
			request_id: 0,
			body: Some(client_frame::Body::Cancel(CancelRequest {
				target: Some(cancel_request::Target::TargetRequestId(request_id)),
				..CancelRequest::default()
			})),
			..ClientFrame::default()
		};
		if outgoing.send(frame).is_err() {
			break;
		}
	}
}
