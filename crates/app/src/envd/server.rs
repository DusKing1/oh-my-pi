//! Transport-neutral `env/v1` dispatch and owner-local UDS serving.

use std::{
	collections::HashMap,
	io,
	path::{Path, PathBuf},
	sync::Arc,
	time::Duration,
};

use bytes::{Bytes, BytesMut};
use futures::StreamExt as _;
use omp_core::Str;
use omp_env::InProcessEnvTransport;
use omp_proto::{
	blob::v1 as blob_pb,
	env::v1::{self as pb, client_frame, server_frame},
	prost::Message as _,
};
use omp_tool::{ErasedEv, ErasedOutcome, IncomingParams, Interrupt, Registry};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt as _, AsyncWrite, AsyncWriteExt as _};
use tokio_util::sync::CancellationToken;

use super::{
	blobs::{BlobError, BlobHost},
	docs::{DocumentError, DocumentHost},
	exec::{ExecError, ExecEvent, ExecHost, ProcessEvent},
	worker::{CommittedToolCall, ToolWorkerConfig, ToolWorkerSupervisor, WorkerError, WorkerEvent},
	workspace::{WorkspaceError, WorkspaceHost},
};
use crate::cli::EnvdArgs;

const MIN_SCHEMA_REV: u32 = 4;
const FRAME_LIMIT: usize = 64 * 1024 * 1024;
const BLOB_CHUNK_BYTES: usize = 64 * 1024;
const DEFAULT_TOOL_DEADLINE: Duration = Duration::from_secs(300);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const NATIVE_CANCEL_GRACE: Duration = Duration::from_millis(250);
const INVOCATION_RESPONSE_SEND_GRACE: Duration = Duration::from_millis(250);

/// Environment-daemon assembly or serving failure.
#[derive(Debug, Error)]
pub enum EnvdError {
	/// A local filesystem, socket, or child-process operation failed.
	#[error(transparent)]
	Io(#[from] io::Error),
	/// The document authority could not be connected or verified.
	#[error("document authority failed: {0}")]
	Document(Str),
	/// The canonical workspace could not be opened.
	#[error("workspace host failed: {0}")]
	Workspace(Str),
	/// The content-addressed blob store could not be opened.
	#[error("blob host failed: {0}")]
	Blob(Str),
	/// The Python tool worker could not be started or supervised.
	#[error(transparent)]
	Worker(#[from] WorkerError),
	/// A spawned environment connection task failed.
	#[error("environment connection task failed: {0}")]
	Task(#[from] tokio::task::JoinError),
	/// The autostarted document server exited before accepting a verified hello.
	#[error("autostarted document server exited before its hello handshake")]
	DocserverExited,
}

impl From<DocumentError> for EnvdError {
	fn from(error: DocumentError) -> Self {
		Self::Document(Str::from(error.to_string()))
	}
}

impl From<WorkspaceError> for EnvdError {
	fn from(error: WorkspaceError) -> Self {
		Self::Workspace(Str::from(error.to_string()))
	}
}

impl From<BlobError> for EnvdError {
	fn from(error: BlobError) -> Self {
		Self::Blob(Str::from(error.to_string()))
	}
}

/// Identity advertised by every transport served from one environment.
#[derive(Clone, Debug)]
pub struct ServerIdentity {
	/// Canonical document workspace identity.
	pub workspace_id:   Bytes,
	/// Canonical workspace root URI.
	pub root_uri:       Str,
	/// Epoch of the connected document authority.
	pub server_epoch:   Bytes,
	/// Human-readable server build version.
	pub server_version: Str,
}

/// Concrete environment host shared by in-process and UDS connections.
///
/// Executors remain env-side beside these resources. The server never passes a
/// capability/facet trait bundle through a tool signature.
pub struct EnvServer {
	identity:   ServerIdentity,
	_documents: DocumentHost,
	exec:       ExecHost,
	_workspace: WorkspaceHost,
	blobs:      BlobHost,
	registry:   Arc<Registry>,
	workers:    Arc<ToolWorkerSupervisor>,
}

impl EnvServer {
	/// Assembles one server from concrete environment-owned resources.
	#[must_use]
	pub(crate) fn new(
		identity: ServerIdentity,
		documents: DocumentHost,
		exec: ExecHost,
		workspace: WorkspaceHost,
		blobs: BlobHost,
		registry: Arc<Registry>,
		workers: ToolWorkerSupervisor,
	) -> Self {
		Self {
			identity,
			_documents: documents,
			exec,
			_workspace: workspace,
			blobs,
			registry,
			workers: Arc::new(workers),
		}
	}

	/// Opens a complete local environment host rooted at `root`.
	///
	/// The document authority, workspace, blob store, executor, and Python
	/// worker are real environment-owned resources. `state_dir` is kept
	/// separate from the workspace so callers can use an isolated scratch
	/// directory without adding daemon state to the project tree.
	pub async fn open_local(
		root: &Path,
		state_dir: &Path,
		registry: Arc<Registry>,
		worker_config: ToolWorkerConfig,
	) -> Result<Self, EnvdError> {
		let workspace = WorkspaceHost::open(root)?;
		let doc_config = omp_docserver::ServerConfig::new(root)
			.map_err(|error| EnvdError::Document(Str::from(error.to_string())))?;
		let environment = omp_docserver::Environment::new(doc_config)
			.map_err(|error| EnvdError::Document(Str::from(error.to_string())))?;
		let (document_client, document_server) = tokio::io::duplex(64 * 1024);
		tokio::spawn(async move {
			let _ = omp_docserver::connection::serve_connection(
				environment,
				document_server,
				omp_docserver::connection::ConnectionConfig::default(),
			)
			.await;
		});
		let documents = DocumentHost::connect(document_client).await?;
		let hello = documents.hello().clone();
		let identity = ServerIdentity {
			workspace_id:   hello.workspace_id,
			root_uri:       hello.root_uri,
			server_epoch:   hello.server_epoch,
			server_version: Str::from(env!("CARGO_PKG_VERSION")),
		};
		Ok(Self::new(
			identity,
			documents,
			ExecHost::new(),
			workspace,
			BlobHost::open(state_dir.join("blobs"))?,
			registry,
			ToolWorkerSupervisor::spawn(worker_config).await?,
		))
	}

	/// Serves the server half returned by [`omp_env::EnvClient::in_process`].
	pub async fn serve_in_process(&self, transport: InProcessEnvTransport) {
		let (requests, responses) = transport.into_parts();
		self.serve_frames(requests, responses).await;
	}

	/// Serves one already-accepted byte stream with varint protobuf framing.
	pub async fn serve_io<S>(&self, stream: S) -> Result<(), EnvdError>
	where
		S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
	{
		let (mut reader, mut writer) = tokio::io::split(stream);
		let (request_tx, requests) = flume::bounded(64);
		let (responses, response_rx) = flume::bounded(64);
		let dispatch = self.serve_frames(requests, responses);
		let io_shutdown = CancellationToken::new();
		let read_shutdown = io_shutdown.clone();
		let read = async move {
			let mut scratch = BytesMut::new();
			loop {
				let frame = tokio::select! {
					() = read_shutdown.cancelled() => return Ok::<(), io::Error>(()),
					result = read_client_frame(&mut reader, &mut scratch) => result?,
				};
				let Some(frame) = frame else { return Ok(()) };
				if request_tx.send_async(frame).await.is_err() {
					return Ok(());
				}
			}
		};
		let write = async move {
			let result = async {
				let mut scratch = BytesMut::new();
				while let Ok(frame) = response_rx.recv_async().await {
					write_server_frame(&mut writer, &frame, &mut scratch).await?;
				}
				Ok::<(), io::Error>(())
			}
			.await;
			io_shutdown.cancel();
			result
		};
		let (read_result, (), write_result) = tokio::join!(read, dispatch, write);
		read_result?;
		write_result?;
		Ok(())
	}

	/// Binds and serves an owner-only project Unix socket until cancellation.
	#[cfg(unix)]
	pub async fn serve_uds(
		self: Arc<Self>,
		path: &Path,
		shutdown: CancellationToken,
	) -> Result<(), EnvdError> {
		use std::os::unix::fs::{FileTypeExt as _, MetadataExt as _, PermissionsExt as _};

		let parent = path.parent().ok_or_else(|| {
			io::Error::new(io::ErrorKind::InvalidInput, "environment socket has no parent")
		})?;
		secure_owner_directory(parent)?;
		match tokio::fs::symlink_metadata(path).await {
			Ok(metadata) if metadata.file_type().is_socket() => {
				if tokio::net::UnixStream::connect(path).await.is_ok() {
					return Err(
						io::Error::new(
							io::ErrorKind::AddrInUse,
							"environment socket is already accepting connections",
						)
						.into(),
					);
				}
				tokio::fs::remove_file(path).await?;
			},
			Ok(_) => {
				return Err(
					io::Error::new(
						io::ErrorKind::AlreadyExists,
						"refusing to replace a non-socket environment path",
					)
					.into(),
				);
			},
			Err(error) if error.kind() == io::ErrorKind::NotFound => {},
			Err(error) => return Err(error.into()),
		}
		let listener = tokio::net::UnixListener::bind(path)?;
		tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).await?;
		let socket_metadata = std::fs::symlink_metadata(path)?;
		let mut connections = tokio::task::JoinSet::new();
		loop {
			tokio::select! {
				() = shutdown.cancelled() => break,
				accepted = listener.accept() => {
					let (stream, _) = accepted?;
					let server = Arc::clone(&self);
					connections.spawn(async move { server.serve_io(stream).await });
				},
				completed = connections.join_next(), if !connections.is_empty() => {
					if let Some(Err(error)) = completed {
						return Err(error.into());
					}
				},
			}
		}
		drop(listener);
		if let Ok(metadata) = std::fs::symlink_metadata(path)
			&& metadata.dev() == socket_metadata.dev()
			&& metadata.ino() == socket_metadata.ino()
		{
			let _ = tokio::fs::remove_file(path).await;
		}
		connections.abort_all();
		while let Some(result) = connections.join_next().await {
			if let Err(error) = result
				&& !error.is_cancelled()
			{
				return Err(error.into());
			}
		}
		Ok(())
	}

	async fn serve_frames(
		&self,
		requests: flume::Receiver<pb::ClientFrame>,
		responses: flume::Sender<pb::ServerFrame>,
	) {
		let first = match tokio::time::timeout(HANDSHAKE_TIMEOUT, requests.recv_async()).await {
			Ok(Ok(first)) => first,
			Ok(Err(_)) => return,
			Err(_) => {
				send_error(
					&responses,
					0,
					pb::ProtocolErrorCode::DeadlineExceeded,
					"environment hello handshake timed out",
				)
				.await;
				return;
			},
		};
		if !self.accept_hello(first, &responses).await {
			return;
		}
		let (finished_tx, finished) = flume::unbounded();
		let mut connection = ConnectionState::new(self.exec.clone());
		loop {
			let next = tokio::select! {
				result = requests.recv_async() => match result {
					Ok(frame) => Some(LoopEvent::Frame(frame)),
					Err(_) => None,
				},
				result = finished.recv_async() => match result {
					Ok(done) => Some(LoopEvent::Finished(done)),
					Err(_) => None,
				},
			};
			let Some(next) = next else { break };
			match next {
				LoopEvent::Finished(done) => connection.finish(done),
				LoopEvent::Frame(frame) => {
					while let Ok(done) = finished.try_recv() {
						connection.finish(done);
					}
					self
						.dispatch(frame, &responses, &finished_tx, &mut connection)
						.await;
				},
			}
		}
		connection.cancel_all(&self.exec);
	}

	async fn accept_hello(
		&self,
		frame: pb::ClientFrame,
		responses: &flume::Sender<pb::ServerFrame>,
	) -> bool {
		let Some(client_frame::Body::Hello(hello)) = frame.body else {
			send_error(
				responses,
				frame.request_id,
				pb::ProtocolErrorCode::InvalidArgument,
				"the first client frame must be ClientHello",
			)
			.await;
			return false;
		};
		if frame.request_id != 0 {
			send_error(
				responses,
				frame.request_id,
				pb::ProtocolErrorCode::InvalidArgument,
				"ClientHello must use request_id 0",
			)
			.await;
			return false;
		}
		if hello.schema_rev < MIN_SCHEMA_REV || hello.schema_rev > omp_proto::SCHEMA_REV {
			send_error(
				responses,
				0,
				pb::ProtocolErrorCode::Unsupported,
				&format!(
					"unsupported env schema revision {}; server supports {MIN_SCHEMA_REV}..={}",
					hello.schema_rev,
					omp_proto::SCHEMA_REV
				),
			)
			.await;
			return false;
		}
		responses
			.send_async(server_frame(
				0,
				server_frame::Body::Hello(pb::ServerHello {
					schema_rev:     omp_proto::SCHEMA_REV,
					min_schema_rev: MIN_SCHEMA_REV,
					capabilities:   vec![
						"invocation".to_owned(),
						"exec".to_owned(),
						"named-process".to_owned(),
						"blob".to_owned(),
					],
					server_version: self.identity.server_version.to_string(),
					workspace_id:   self.identity.workspace_id.clone(),
					root_uri:       self.identity.root_uri.to_string(),
					server_epoch:   self.identity.server_epoch.clone(),
					props:          Default::default(),
				}),
			))
			.await
			.is_ok()
	}

	async fn dispatch(
		&self,
		frame: pb::ClientFrame,
		responses: &flume::Sender<pb::ServerFrame>,
		finished: &flume::Sender<Finished>,
		connection: &mut ConnectionState,
	) {
		let Some(body) = frame.body else {
			send_error(
				responses,
				frame.request_id,
				pb::ProtocolErrorCode::InvalidArgument,
				"client frame body is missing",
			)
			.await;
			return;
		};
		if let client_frame::Body::Cancel(cancel) = body {
			if frame.request_id != 0 {
				send_error(
					responses,
					frame.request_id,
					pb::ProtocolErrorCode::InvalidArgument,
					"cancel control frames must use request_id 0",
				)
				.await;
				return;
			}
			connection
				.cancel(cancel, &self.exec, responses, finished)
				.await;
			return;
		}
		if frame.request_id == 0 {
			send_error(
				responses,
				0,
				pb::ProtocolErrorCode::InvalidArgument,
				"ordinary frames must use a nonzero request_id",
			)
			.await;
			return;
		}
		let continuation = matches!(
			&body,
			client_frame::Body::ArgText(_)
				| client_frame::Body::ArgsCommitted(_)
				| client_frame::Body::Interrupt(_)
				| client_frame::Body::Stdin(_)
				| client_frame::Body::Signal(_)
				| client_frame::Body::Resize(_)
				| client_frame::Body::BlobPutChunk(_)
				| client_frame::Body::BlobPutCommit(_)
		);
		if !continuation && connection.requests.contains_key(&frame.request_id) {
			send_error(
				responses,
				frame.request_id,
				pb::ProtocolErrorCode::AlreadyExists,
				"request_id is already open",
			)
			.await;
			return;
		}

		match body {
			client_frame::Body::Hello(_) => {
				send_error(
					responses,
					frame.request_id,
					pb::ProtocolErrorCode::AlreadyExists,
					"the connection hello is already complete",
				)
				.await;
			},
			client_frame::Body::InvokeTool(request) => {
				self
					.open_invocation(frame.request_id, request, responses, finished, connection)
					.await;
			},
			client_frame::Body::ArgText(request) => {
				let result = connection.invocation_mut(frame.request_id, &request.invocation_id);
				match result {
					Ok(InvocationState::Native { feed, committed, .. }) if !*committed => {
						if feed.arg_text(Str::from(request.fragment)).is_err() {
							send_error(
								responses,
								frame.request_id,
								pb::ProtocolErrorCode::Cancelled,
								"invocation input is closed",
							)
							.await;
						}
					},
					Ok(InvocationState::Worker { committed, .. }) if !*committed => {},
					Ok(_) => {
						send_error(
							responses,
							frame.request_id,
							pb::ProtocolErrorCode::PreconditionFailed,
							"ArgText cannot follow ArgsCommitted",
						)
						.await
					},
					Err((code, message)) => send_error(responses, frame.request_id, code, message).await,
				}
			},
			client_frame::Body::ArgsCommitted(request) => {
				self
					.commit_invocation(frame.request_id, request, responses, finished, connection)
					.await;
			},
			client_frame::Body::Interrupt(request) => {
				connection
					.interrupt(frame.request_id, request, responses, finished)
					.await;
			},
			client_frame::Body::OpenSession(request) => {
				if reject_duplicate_open(connection, frame.request_id, responses).await {
					return;
				}
				match self.exec.open_session(request).await {
					Ok(response) => {
						send_body(
							responses,
							frame.request_id,
							server_frame::Body::SessionOpened(response),
						)
						.await
					},
					Err(error) => send_exec_error(responses, frame.request_id, &error).await,
				}
			},
			client_frame::Body::CloseSession(request) => {
				match self.exec.close_session(&request.session) {
					Ok(response) => {
						send_body(
							responses,
							frame.request_id,
							server_frame::Body::SessionClosed(response),
						)
						.await
					},
					Err(error) => send_exec_error(responses, frame.request_id, &error).await,
				}
			},
			client_frame::Body::Exec(request) => {
				if reject_duplicate_open(connection, frame.request_id, responses).await {
					return;
				}
				match self.exec.exec(request, None).await {
					Ok((started, run)) => {
						let exec = Bytes::copy_from_slice(run.id());
						let cancel = CancellationToken::new();
						connection
							.requests
							.insert(frame.request_id, RequestState::Exec {
								exec:   exec.clone(),
								cancel: cancel.clone(),
							});
						send_body(responses, frame.request_id, server_frame::Body::ExecStarted(started))
							.await;
						spawn_exec(frame.request_id, run, cancel, responses.clone(), finished.clone());
					},
					Err(error) => send_exec_error(responses, frame.request_id, &error).await,
				}
			},
			client_frame::Body::Stdin(request) => {
				if let Some(exec) = connection
					.exec_id(frame.request_id, &request.exec, responses)
					.await
				{
					let data = match request.input {
						Some(pb::stdin_frame::Input::Data(data)) => Some(data),
						Some(pb::stdin_frame::Input::Eof(true)) => None,
						_ => {
							send_error(
								responses,
								frame.request_id,
								pb::ProtocolErrorCode::InvalidArgument,
								"stdin frame has no data or eof marker",
							)
							.await;
							return;
						},
					};
					if let Err(error) = self.exec.stdin(&exec, data.as_deref()) {
						send_exec_error(responses, frame.request_id, &error).await;
					}
				}
			},
			client_frame::Body::Signal(request) => {
				if let Some(exec) = connection
					.exec_id(frame.request_id, &request.exec, responses)
					.await && let Err(error) = self.exec.signal(&exec, &request.signal)
				{
					send_exec_error(responses, frame.request_id, &error).await;
				}
			},
			client_frame::Body::Resize(request) => {
				if let Some(exec) = connection
					.exec_id(frame.request_id, &request.exec, responses)
					.await && let Err(error) = self.exec.resize(&exec, request.rows, request.columns)
				{
					send_exec_error(responses, frame.request_id, &error).await;
				}
			},
			client_frame::Body::StartProcess(request) => {
				match self.exec.start_process(request).await {
					Ok(response) => {
						send_body(
							responses,
							frame.request_id,
							server_frame::Body::ProcessStarted(response),
						)
						.await
					},
					Err(error) => send_exec_error(responses, frame.request_id, &error).await,
				}
			},
			client_frame::Body::ListProcesses(_) => {
				send_body(
					responses,
					frame.request_id,
					server_frame::Body::ProcessList(self.exec.list_processes()),
				)
				.await;
			},
			client_frame::Body::AttachOutput(request) => {
				if reject_duplicate_open(connection, frame.request_id, responses).await {
					return;
				}
				match self.exec.attach_output(&request) {
					Ok(attachment) => {
						let cancel = CancellationToken::new();
						let process_name = Str::from(request.name);
						connection
							.requests
							.insert(frame.request_id, RequestState::ProcessAttach {
								cancel: cancel.clone(),
							});
						send_body(
							responses,
							frame.request_id,
							server_frame::Body::OutputAttached(attachment.attached),
						)
						.await;
						for output in attachment.backlog {
							send_body(
								responses,
								frame.request_id,
								server_frame::Body::ProcessOutput(output),
							)
							.await;
						}
						spawn_process_attachment(
							frame.request_id,
							process_name,
							attachment.events,
							cancel,
							responses.clone(),
							finished.clone(),
						);
					},
					Err(error) => send_exec_error(responses, frame.request_id, &error).await,
				}
			},
			client_frame::Body::SendInput(request) => {
				let data = match request.input {
					Some(pb::send_input::Input::Data(data)) => Some(data),
					Some(pb::send_input::Input::Eof(true)) => None,
					_ => {
						send_error(
							responses,
							frame.request_id,
							pb::ProtocolErrorCode::InvalidArgument,
							"process input has no data or eof marker",
						)
						.await;
						return;
					},
				};
				match self.exec.send_process_input(&request.name, data.as_deref()) {
					Ok(response) => {
						send_body(
							responses,
							frame.request_id,
							server_frame::Body::ProcessCommandAccepted(response),
						)
						.await
					},
					Err(error) => send_exec_error(responses, frame.request_id, &error).await,
				}
			},
			client_frame::Body::SignalProcess(request) => {
				match self.exec.signal_process(&request.name, &request.signal) {
					Ok(response) => {
						send_body(
							responses,
							frame.request_id,
							server_frame::Body::ProcessCommandAccepted(response),
						)
						.await
					},
					Err(error) => send_exec_error(responses, frame.request_id, &error).await,
				}
			},
			client_frame::Body::StopProcess(request) => {
				match self
					.exec
					.stop_process(&request.name, Duration::from_millis(request.grace_ms))
				{
					Ok(response) => {
						send_body(
							responses,
							frame.request_id,
							server_frame::Body::ProcessCommandAccepted(response),
						)
						.await
					},
					Err(error) => send_exec_error(responses, frame.request_id, &error).await,
				}
			},
			client_frame::Body::BlobStat(request) => match self.blobs.stat(&request.hash) {
				Ok(response) => {
					send_body(responses, frame.request_id, server_frame::Body::BlobStat(response)).await
				},
				Err(error) => send_blob_error(responses, frame.request_id, &error).await,
			},
			client_frame::Body::BlobGet(request) => {
				if reject_duplicate_open(connection, frame.request_id, responses).await {
					return;
				}
				match self.blobs.get_request(&request) {
					Ok(read) => {
						let cancel = CancellationToken::new();
						connection
							.requests
							.insert(frame.request_id, RequestState::BlobGet { cancel: cancel.clone() });
						spawn_blob_get(
							frame.request_id,
							read,
							cancel,
							responses.clone(),
							finished.clone(),
						);
					},
					Err(error) => send_blob_error(responses, frame.request_id, &error).await,
				}
			},
			client_frame::Body::BlobPutChunk(chunk) => {
				self
					.put_chunk(frame.request_id, chunk, responses, connection)
					.await;
			},
			client_frame::Body::BlobPutCommit(_) => {
				self
					.commit_blob(frame.request_id, responses, connection)
					.await;
			},
			client_frame::Body::BlobDelete(request) => match self.blobs.delete(&request.hash) {
				Ok(response) => {
					send_body(responses, frame.request_id, server_frame::Body::BlobDeleted(response))
						.await
				},
				Err(error) => send_blob_error(responses, frame.request_id, &error).await,
			},
			client_frame::Body::Cancel(_) => unreachable!("cancel handled before ordinary dispatch"),
		}
	}

	async fn open_invocation(
		&self,
		request_id: u64,
		request: pb::InvokeTool,
		responses: &flume::Sender<pb::ServerFrame>,
		_finished: &flume::Sender<Finished>,
		connection: &mut ConnectionState,
	) {
		if reject_duplicate_open(connection, request_id, responses).await {
			return;
		}
		let invocation_id = Str::from(request.invocation_id.as_str());
		if invocation_id.is_empty() {
			send_error(
				responses,
				request_id,
				pb::ProtocolErrorCode::InvalidArgument,
				"invocation_id must not be empty",
			)
			.await;
			return;
		}
		if connection.invocation_ids.contains_key(&invocation_id) {
			send_error(
				responses,
				request_id,
				pb::ProtocolErrorCode::AlreadyExists,
				"invocation_id is already open on this connection",
			)
			.await;
			return;
		}
		let cancel = CancellationToken::new();
		if let Some((_, revision)) = self.registry.live_identity(&request.name) {
			if revision.to_string() != request.rev {
				send_error(
					responses,
					request_id,
					pb::ProtocolErrorCode::PreconditionFailed,
					"requested tool revision is not live",
				)
				.await;
				return;
			}
			let (feed, params) = IncomingParams::channel();
			connection.requests.insert(
				request_id,
				RequestState::Invocation(InvocationState::Native {
					id: invocation_id.clone(),
					name: Str::from(request.name),
					deadline: if request.deadline_ms == 0 {
						DEFAULT_TOOL_DEADLINE
					} else {
						Duration::from_millis(request.deadline_ms)
					},
					feed,
					params: Some(params),
					committed: false,
					cancel,
				}),
			);
			connection
				.invocation_ids
				.insert(invocation_id.clone(), request_id);
			send_body(
				responses,
				request_id,
				server_frame::Body::InvocationAccepted(pb::InvokeAccepted {
					invocation_id: invocation_id.to_string(),
					props:         Default::default(),
				}),
			)
			.await;
		} else if self.worker_decl(&request.name, &request.rev) {
			let (interrupt, interrupts) = flume::unbounded();
			connection.requests.insert(
				request_id,
				RequestState::Invocation(InvocationState::Worker {
					id: invocation_id.clone(),
					name: Str::from(request.name),
					rev: Str::from(request.rev),
					deadline: if request.deadline_ms == 0 {
						DEFAULT_TOOL_DEADLINE
					} else {
						Duration::from_millis(request.deadline_ms)
					},
					committed: false,
					interrupt,
					interrupts: Some(interrupts),
					cancel,
				}),
			);
			connection
				.invocation_ids
				.insert(invocation_id.clone(), request_id);
			send_body(
				responses,
				request_id,
				server_frame::Body::InvocationAccepted(pb::InvokeAccepted {
					invocation_id: invocation_id.to_string(),
					props:         Default::default(),
				}),
			)
			.await;
		} else {
			send_error(
				responses,
				request_id,
				pb::ProtocolErrorCode::NotFound,
				"tool name and revision are not registered",
			)
			.await;
			return;
		}
	}

	async fn commit_invocation(
		&self,
		request_id: u64,
		request: pb::ArgsCommitted,
		responses: &flume::Sender<pb::ServerFrame>,
		finished: &flume::Sender<Finished>,
		connection: &mut ConnectionState,
	) {
		let result = connection.invocation_mut(request_id, &request.invocation_id);
		match result {
			Ok(InvocationState::Native { id, name, feed, params, deadline, committed, cancel }) => {
				if *committed {
					send_error(
						responses,
						request_id,
						pb::ProtocolErrorCode::AlreadyExists,
						"ArgsCommitted was already received",
					)
					.await;
					return;
				}
				let Ok(raw) = std::str::from_utf8(&request.raw) else {
					send_error(
						responses,
						request_id,
						pb::ProtocolErrorCode::InvalidArgument,
						"committed arguments are not UTF-8",
					)
					.await;
					return;
				};
				if feed.args_committed(Str::from(raw)).is_err() {
					send_error(
						responses,
						request_id,
						pb::ProtocolErrorCode::Cancelled,
						"invocation input is closed",
					)
					.await;
					return;
				}
				let Some(params) = params.take() else {
					send_error(
						responses,
						request_id,
						pb::ProtocolErrorCode::PreconditionFailed,
						"native invocation was already dispatched",
					)
					.await;
					return;
				};
				*committed = true;
				spawn_native_invocation(
					request_id,
					id.clone(),
					name.clone(),
					feed.clone(),
					*deadline,
					params,
					Arc::clone(&self.registry),
					cancel.clone(),
					responses.clone(),
					finished.clone(),
				);
			},
			Ok(InvocationState::Worker {
				id,
				name,
				rev,
				deadline,
				committed,
				cancel,
				interrupts,
				..
			}) => {
				if *committed {
					send_error(
						responses,
						request_id,
						pb::ProtocolErrorCode::AlreadyExists,
						"ArgsCommitted was already received",
					)
					.await;
					return;
				}
				*committed = true;
				let call = CommittedToolCall {
					call_id:   id.clone(),
					name:      name.clone(),
					rev:       rev.clone(),
					args_json: request.raw,
					deadline:  *deadline,
				};
				let Some(interrupts) = interrupts.take() else {
					send_error(
						responses,
						request_id,
						pb::ProtocolErrorCode::PreconditionFailed,
						"worker invocation was already dispatched",
					)
					.await;
					return;
				};
				match self.workers.invoke_committed(call) {
					Ok(invocation) => spawn_worker_invocation(
						request_id,
						id.clone(),
						invocation,
						cancel.clone(),
						interrupts,
						responses.clone(),
						finished.clone(),
					),
					Err(error) => {
						let invocation_id = id.clone();
						send_error(
							responses,
							request_id,
							pb::ProtocolErrorCode::Internal,
							&error.to_string(),
						)
						.await;
						connection.finish(Finished { request_id, invocation_id: Some(invocation_id) });
					},
				}
			},
			Err((code, message)) => send_error(responses, request_id, code, message).await,
		}
	}

	fn worker_decl(&self, name: &str, rev: &str) -> bool {
		self.workers.registrations().iter().any(|decl| {
			decl.rev == rev
				&& decl
					.definition
					.as_ref()
					.is_some_and(|definition| definition.name == name)
		})
	}

	async fn put_chunk(
		&self,
		request_id: u64,
		chunk: blob_pb::Chunk,
		responses: &flume::Sender<pb::ServerFrame>,
		connection: &mut ConnectionState,
	) {
		if !connection.requests.contains_key(&request_id) {
			connection
				.requests
				.insert(request_id, RequestState::BlobPut(BlobUpload::default()));
		}
		let Some(RequestState::BlobPut(upload)) = connection.requests.get_mut(&request_id) else {
			send_error(
				responses,
				request_id,
				pb::ProtocolErrorCode::AlreadyExists,
				"request_id is already open for another operation",
			)
			.await;
			return;
		};
		if upload.chunks != 0 && (!chunk.hash.is_empty() || chunk.size.is_some()) {
			send_error(
				responses,
				request_id,
				pb::ProtocolErrorCode::InvalidArgument,
				"blob hash and size metadata are legal only on the first chunk",
			)
			.await;
			return;
		}
		if upload.chunks == 0 {
			upload.expected_hash = (!chunk.hash.is_empty()).then_some(chunk.hash);
			upload.expected_size = chunk.size;
		}
		upload.data.extend_from_slice(&chunk.data);
		upload.chunks += 1;
	}

	async fn commit_blob(
		&self,
		request_id: u64,
		responses: &flume::Sender<pb::ServerFrame>,
		connection: &mut ConnectionState,
	) {
		let upload = match connection.requests.remove(&request_id) {
			Some(RequestState::BlobPut(upload)) => upload,
			Some(other) => {
				connection.requests.insert(request_id, other);
				send_error(
					responses,
					request_id,
					pb::ProtocolErrorCode::AlreadyExists,
					"request_id is already open for another operation",
				)
				.await;
				return;
			},
			None => BlobUpload::default(),
		};
		match self.blobs.put_checked(
			&upload.data,
			upload.expected_hash.as_deref(),
			upload.expected_size,
		) {
			Ok(id) => {
				send_body(
					responses,
					request_id,
					server_frame::Body::BlobPut(blob_pb::PutResponse {
						hash: Bytes::copy_from_slice(&id.hash),
						size: id.size,
					}),
				)
				.await
			},
			Err(error) => send_blob_error(responses, request_id, &error).await,
		}
	}
}

struct ConnectionState {
	requests:       HashMap<u64, RequestState>,
	invocation_ids: HashMap<Str, u64>,
	exec_host:      ExecHost,
}

enum RequestState {
	Invocation(InvocationState),
	InvocationFinishing,
	Exec { exec: Bytes, cancel: CancellationToken },
	ProcessAttach { cancel: CancellationToken },
	BlobPut(BlobUpload),
	BlobGet { cancel: CancellationToken },
}

enum InvocationState {
	Native {
		id:        Str,
		name:      Str,
		feed:      omp_tool::InvocationFeed,
		deadline:  Duration,
		params:    Option<IncomingParams<'static>>,
		committed: bool,
		cancel:    CancellationToken,
	},
	Worker {
		id:         Str,
		name:       Str,
		rev:        Str,
		deadline:   Duration,
		committed:  bool,
		interrupt:  flume::Sender<Str>,
		interrupts: Option<flume::Receiver<Str>>,
		cancel:     CancellationToken,
	},
}

#[derive(Default)]
struct BlobUpload {
	data:          BytesMut,
	expected_hash: Option<Bytes>,
	expected_size: Option<u64>,
	chunks:        usize,
}

struct Finished {
	request_id:    u64,
	invocation_id: Option<Str>,
}

enum LoopEvent {
	Frame(pb::ClientFrame),
	Finished(Finished),
}

impl ConnectionState {
	fn new(exec_host: ExecHost) -> Self {
		Self { requests: HashMap::new(), invocation_ids: HashMap::new(), exec_host }
	}

	fn invocation_mut(
		&mut self,
		request_id: u64,
		invocation_id: &str,
	) -> Result<&mut InvocationState, (pb::ProtocolErrorCode, &'static str)> {
		match self.requests.get_mut(&request_id) {
			Some(RequestState::Invocation(state)) if state.id() == invocation_id => Ok(state),
			Some(RequestState::Invocation(_)) => Err((
				pb::ProtocolErrorCode::InvalidArgument,
				"invocation_id does not match the open request",
			)),
			Some(_) => Err((
				pb::ProtocolErrorCode::PreconditionFailed,
				"request_id is not an invocation stream",
			)),
			None => Err((pb::ProtocolErrorCode::NotFound, "invocation is not open")),
		}
	}

	async fn exec_id(
		&self,
		request_id: u64,
		expected: &[u8],
		responses: &flume::Sender<pb::ServerFrame>,
	) -> Option<Bytes> {
		match self.requests.get(&request_id) {
			Some(RequestState::Exec { exec, .. }) if exec.as_ref() == expected => Some(exec.clone()),
			Some(RequestState::Exec { .. }) => {
				send_error(
					responses,
					request_id,
					pb::ProtocolErrorCode::InvalidArgument,
					"exec id does not match the open request",
				)
				.await;
				None
			},
			Some(_) => {
				send_error(
					responses,
					request_id,
					pb::ProtocolErrorCode::PreconditionFailed,
					"request_id is not an exec stream",
				)
				.await;
				None
			},
			None => {
				send_error(
					responses,
					request_id,
					pb::ProtocolErrorCode::NotFound,
					"execution is not open",
				)
				.await;
				None
			},
		}
	}

	fn finish(&mut self, done: Finished) {
		self.requests.remove(&done.request_id);
		if let Some(invocation_id) = done.invocation_id {
			self.invocation_ids.remove(&invocation_id);
		}
	}

	async fn cancel(
		&mut self,
		request: pb::CancelRequest,
		exec_host: &ExecHost,
		responses: &flume::Sender<pb::ServerFrame>,
		finished: &flume::Sender<Finished>,
	) {
		use pb::cancel_request::Target;
		match request.target {
			Some(Target::TargetRequestId(request_id)) => {
				if let Some(RequestState::Exec { exec, .. }) = self.requests.get(&request_id) {
					let _ = exec_host.cancel(exec);
				} else {
					self
						.cancel_request(request_id, exec_host, responses, finished)
						.await;
				}
			},
			Some(Target::InvocationId(invocation_id)) => {
				if let Some(request_id) = self.invocation_ids.get(invocation_id.as_str()).copied() {
					self
						.cancel_request(request_id, exec_host, responses, finished)
						.await;
				}
			},
			Some(Target::Exec(exec_id)) => {
				let _ = exec_host.cancel(&exec_id);
			},
			None => {},
		}
	}

	async fn cancel_request(
		&mut self,
		request_id: u64,
		exec_host: &ExecHost,
		responses: &flume::Sender<pb::ServerFrame>,
		finished: &flume::Sender<Finished>,
	) {
		if let Some(RequestState::Invocation(state)) = self.requests.get_mut(&request_id) {
			let terminal = match state {
				InvocationState::Native { id, feed, committed, cancel, .. } => {
					if *committed {
						let _ = feed.interrupt(Interrupt {
							class:  Str::new_static("cancel"),
							reason: Str::new_static("invocation cancelled by client"),
						});
						cancel.cancel();
						None
					} else {
						cancel.cancel();
						Some((id.clone(), omp_tool::Abort::Skipped {
							reason: Str::new_static("invocation cancelled before argument commitment"),
						}))
					}
				},
				InvocationState::Worker { id, committed, cancel, .. } => {
					cancel.cancel();
					(!*committed).then(|| {
						(id.clone(), omp_tool::Abort::Skipped {
							reason: Str::new_static("invocation cancelled before argument commitment"),
						})
					})
				},
			};
			if terminal.is_some() {
				self
					.requests
					.insert(request_id, RequestState::InvocationFinishing);
			}
			if let Some((invocation_id, abort)) = terminal {
				send_abort_verdict(responses, request_id, &invocation_id, abort).await;
				let _ = finished
					.send_async(Finished { request_id, invocation_id: Some(invocation_id) })
					.await;
			}
			return;
		}
		if matches!(self.requests.get(&request_id), Some(RequestState::InvocationFinishing)) {
			return;
		}

		let Some(state) = self.requests.remove(&request_id) else {
			return;
		};
		match state {
			RequestState::Invocation(_) => unreachable!("invocations were handled without removal"),
			RequestState::InvocationFinishing => {},
			RequestState::Exec { exec, cancel } => {
				let _ = exec_host.cancel(&exec);
				cancel.cancel();
			},
			RequestState::ProcessAttach { cancel } | RequestState::BlobGet { cancel } => {
				cancel.cancel();
			},
			RequestState::BlobPut(_) => {},
		}
	}

	async fn interrupt(
		&mut self,
		request_id: u64,
		request: pb::Interrupt,
		responses: &flume::Sender<pb::ServerFrame>,
		finished: &flume::Sender<Finished>,
	) {
		let result = self.invocation_mut(request_id, &request.invocation_id);
		let terminal = match result {
			Ok(InvocationState::Native { id, feed, committed, cancel, .. }) => {
				let reason = Str::from(request.reason);
				let _ = feed.interrupt(Interrupt {
					class:  Str::new_static("immediate"),
					reason: reason.clone(),
				});
				if *committed {
					None
				} else {
					cancel.cancel();
					Some((id.clone(), omp_tool::Abort::Interrupted { reason }))
				}
			},
			Ok(InvocationState::Worker { id, committed, cancel, interrupt, .. }) => {
				let reason = Str::from(request.reason);
				if *committed {
					let _ = interrupt.send(reason);
					None
				} else {
					cancel.cancel();
					Some((id.clone(), omp_tool::Abort::Interrupted { reason }))
				}
			},
			Err((code, message)) => {
				send_error(responses, request_id, code, message).await;
				return;
			},
		};
		if terminal.is_some() {
			self
				.requests
				.insert(request_id, RequestState::InvocationFinishing);
		}
		if let Some((invocation_id, abort)) = terminal {
			send_abort_verdict(responses, request_id, &invocation_id, abort).await;
			let _ = finished
				.send_async(Finished { request_id, invocation_id: Some(invocation_id) })
				.await;
		}
	}

	fn cancel_all(&mut self, exec_host: &ExecHost) {
		for (_, state) in std::mem::take(&mut self.requests) {
			match state {
				RequestState::Invocation(InvocationState::Native {
					feed, committed, cancel, ..
				}) => {
					if committed {
						let _ = feed.interrupt(Interrupt {
							class:  Str::new_static("disconnect"),
							reason: Str::new_static("environment connection closed"),
						});
					}
					cancel.cancel();
				},
				RequestState::Invocation(InvocationState::Worker { cancel, .. }) => cancel.cancel(),
				RequestState::InvocationFinishing => {},
				RequestState::Exec { exec, cancel } => {
					let _ = exec_host.cancel(&exec);
					cancel.cancel();
				},
				RequestState::ProcessAttach { cancel } | RequestState::BlobGet { cancel } => {
					cancel.cancel()
				},
				RequestState::BlobPut(_) => {},
			}
		}
		self.invocation_ids.clear();
	}
}

impl Drop for ConnectionState {
	fn drop(&mut self) {
		let exec_host = self.exec_host.clone();
		self.cancel_all(&exec_host);
	}
}

impl InvocationState {
	fn id(&self) -> &str {
		match self {
			Self::Native { id, .. } | Self::Worker { id, .. } => id,
		}
	}
}

async fn reject_duplicate_open(
	connection: &ConnectionState,
	request_id: u64,
	responses: &flume::Sender<pb::ServerFrame>,
) -> bool {
	if connection.requests.contains_key(&request_id) {
		send_error(
			responses,
			request_id,
			pb::ProtocolErrorCode::AlreadyExists,
			"request_id is already open",
		)
		.await;
		true
	} else {
		false
	}
}

enum NativeForward {
	Continue,
	Terminal,
	Backpressure,
}

fn spawn_native_invocation(
	request_id: u64,
	invocation_id: Str,
	name: Str,
	feed: omp_tool::InvocationFeed,
	deadline: Duration,
	params: IncomingParams<'static>,
	registry: Arc<Registry>,
	cancel: CancellationToken,
	responses: flume::Sender<pb::ServerFrame>,
	finished: flume::Sender<Finished>,
) {
	tokio::spawn(async move {
		let result = registry.invoke(&name, params);
		match result {
			Ok(mut stream) => {
				let mut deadline = Box::pin(tokio::time::sleep(deadline));
				let mut cancel_grace: Option<std::pin::Pin<Box<tokio::time::Sleep>>> = None;
				let mut timed_out = false;
				let mut grace_expired = false;
				loop {
					if let Some(grace) = cancel_grace.as_mut() {
						tokio::select! {
							biased;
							() = grace.as_mut() => {
								grace_expired = true;
								break;
							},
							event = stream.next() => {
								let reason = if timed_out {
									"native invocation ended without reporting timeout truth"
								} else {
									"native invocation ended without reporting cancellation truth"
								};
								if matches!(
									forward_native_event(
										event,
										true,
										reason,
										request_id,
										&invocation_id,
										&responses,
									)
									.await,
									NativeForward::Terminal
								) {
									break;
								}
							},
						}
					} else {
						tokio::select! {
							biased;
							() = deadline.as_mut() => {
								timed_out = true;
								let _ = feed.interrupt(Interrupt {
									class: Str::new_static("deadline"),
									reason: Str::new_static("native invocation deadline exceeded"),
								});
								cancel_grace = Some(Box::pin(tokio::time::sleep(
									NATIVE_CANCEL_GRACE,
								)));
							},
							() = cancel.cancelled() => {
								cancel_grace = Some(Box::pin(tokio::time::sleep(
									NATIVE_CANCEL_GRACE,
								)));
							},
							event = stream.next() => {
								match forward_native_event(
									event,
									false,
									"",
									request_id,
									&invocation_id,
									&responses,
								)
								.await
								{
									NativeForward::Continue => {},
									NativeForward::Terminal => break,
									NativeForward::Backpressure => {
										let _ = feed.interrupt(Interrupt {
											class: Str::new_static("backpressure"),
											reason: Str::new_static(
												"invocation response consumer stopped reading",
											),
										});
										cancel_grace = Some(Box::pin(tokio::time::sleep(
											NATIVE_CANCEL_GRACE,
										)));
									},
								}
							},
						}
					}
				}
				if grace_expired {
					drop(stream);
					let reason = if timed_out {
						Str::new_static(
							"native invocation exceeded its deadline and did not stop within grace",
						)
					} else {
						Str::new_static("native invocation did not stop within cancellation grace")
					};
					send_abort_verdict(
						&responses,
						request_id,
						&invocation_id,
						omp_tool::Abort::EffectsUnknown { reason },
					)
					.await;
				}
			},
			Err(error) => {
				let _ = send_invocation_error(
					&responses,
					request_id,
					pb::ProtocolErrorCode::NotFound,
					&error.to_string(),
				)
				.await;
			},
		}
		let _ = finished
			.send_async(Finished { request_id, invocation_id: Some(invocation_id) })
			.await;
	});
}

async fn forward_native_event(
	event: Option<Result<ErasedEv, omp_tool::RegistryError>>,
	cancelling: bool,
	fallback_reason: &str,
	request_id: u64,
	invocation_id: &Str,
	responses: &flume::Sender<pb::ServerFrame>,
) -> NativeForward {
	match event {
		Some(Ok(ErasedEv::Update(_))) if cancelling => NativeForward::Continue,
		Some(Ok(ErasedEv::Update(json))) => {
			if send_invocation_body(
				responses,
				request_id,
				server_frame::Body::Update(pb::Update {
					invocation_id: invocation_id.to_string(),
					json,
					props: Default::default(),
				}),
			)
			.await
			{
				NativeForward::Continue
			} else {
				NativeForward::Backpressure
			}
		},
		Some(Ok(ErasedEv::Done(outcome))) => {
			let (json, is_error, useless) = erased_outcome_wire(outcome);
			send_invocation_terminal_body(
				responses,
				request_id,
				server_frame::Body::Verdict(pb::Verdict {
					invocation_id: invocation_id.to_string(),
					json,
					parts: Vec::new(),
					is_error,
					useless,
					props: Default::default(),
				}),
			)
			.await;
			NativeForward::Terminal
		},
		Some(Err(error)) if !cancelling => {
			let _ = send_invocation_error(
				responses,
				request_id,
				pb::ProtocolErrorCode::Internal,
				&error.to_string(),
			)
			.await;
			NativeForward::Terminal
		},
		None if !cancelling => {
			let _ = send_invocation_stream_error(
				responses,
				request_id,
				invocation_id,
				"tool event stream closed without a terminal verdict",
			)
			.await;
			NativeForward::Terminal
		},
		Some(Err(_)) | None => {
			send_abort_verdict(
				responses,
				request_id,
				invocation_id,
				omp_tool::Abort::EffectsUnknown { reason: Str::from(fallback_reason) },
			)
			.await;
			NativeForward::Terminal
		},
	}
}

fn spawn_worker_invocation(
	request_id: u64,
	invocation_id: Str,
	mut invocation: super::worker::WorkerInvocation,
	cancel: CancellationToken,
	interrupts: flume::Receiver<Str>,
	responses: flume::Sender<pb::ServerFrame>,
	finished: flume::Sender<Finished>,
) {
	tokio::spawn(async move {
		let mut cancel_requested = false;
		loop {
			let event = if cancel_requested {
				invocation.next().await.ok()
			} else {
				tokio::select! {
					biased;
					() = cancel.cancelled() => {
						invocation.cancel("environment invocation cancelled");
						cancel_requested = true;
						continue;
					},
					reason = interrupts.recv_async() => {
						if let Ok(reason) = reason {
							invocation.interrupt(reason);
						}
						continue;
					},
					event = invocation.next() => event.ok(),
				}
			};
			match event {
				Some(WorkerEvent::Update(_)) if cancel_requested => {},
				Some(WorkerEvent::Update(update)) => {
					if !send_invocation_body(
						&responses,
						request_id,
						server_frame::Body::Update(pb::Update {
							invocation_id: invocation_id.to_string(),
							json:          update.json,
							props:         Default::default(),
						}),
					)
					.await
					{
						invocation.cancel("invocation response consumer stopped reading");
						cancel_requested = true;
					}
				},
				Some(WorkerEvent::Complete(complete)) => {
					let is_error = complete.is_error;
					let json = match worker_verdict_json(complete.details_json, is_error) {
						Ok(json) => json,
						Err(_) => {
							send_abort_verdict(
								&responses,
								request_id,
								&invocation_id,
								omp_tool::Abort::EffectsUnknown {
									reason: Str::new_static(
										"worker returned invalid structured result JSON",
									),
								},
							)
							.await;
							break;
						},
					};
					send_invocation_terminal_body(
						&responses,
						request_id,
						server_frame::Body::Verdict(pb::Verdict {
							invocation_id: invocation_id.to_string(),
							json,
							parts: complete.parts,
							is_error,
							useless: false,
							props: Default::default(),
						}),
					)
					.await;
					break;
				},
				Some(WorkerEvent::Aborted(abort)) => {
					let reason = if abort.effects_unknown {
						omp_tool::Abort::EffectsUnknown { reason: abort.reason }
					} else {
						omp_tool::Abort::Skipped { reason: abort.reason }
					};
					send_abort_verdict(&responses, request_id, &invocation_id, reason).await;
					break;
				},
				None => {
					let _ = send_invocation_stream_error(
						&responses,
						request_id,
						&invocation_id,
						"tool worker event stream closed without a terminal verdict",
					)
					.await;
					break;
				},
			}
		}
		let _ = finished
			.send_async(Finished { request_id, invocation_id: Some(invocation_id) })
			.await;
	});
}

async fn send_abort_verdict(
	responses: &flume::Sender<pb::ServerFrame>,
	request_id: u64,
	invocation_id: &Str,
	abort: omp_tool::Abort,
) {
	let verdict = omp_tool::Verdict::<serde_json::Value, serde_json::Value>::Aborted(abort);
	let Ok(json) = serde_json::to_vec(&verdict) else {
		let _ = send_invocation_stream_error(
			responses,
			request_id,
			invocation_id,
			"failed to serialize invocation abort verdict",
		)
		.await;
		return;
	};
	send_invocation_terminal_body(
		responses,
		request_id,
		server_frame::Body::Verdict(pb::Verdict {
			invocation_id: invocation_id.to_string(),
			json:          Bytes::from(json),
			parts:         Vec::new(),
			is_error:      true,
			useless:       false,
			props:         Default::default(),
		}),
	)
	.await;
}

async fn send_invocation_error(
	responses: &flume::Sender<pb::ServerFrame>,
	request_id: u64,
	code: pb::ProtocolErrorCode,
	message: &str,
) -> bool {
	let body = server_frame::Body::Error(pb::ProtocolError {
		code:    code as i32,
		message: message.to_owned(),
		props:   Default::default(),
	});
	send_invocation_terminal_body(responses, request_id, body).await;
	true
}

async fn send_invocation_stream_error(
	responses: &flume::Sender<pb::ServerFrame>,
	request_id: u64,
	invocation_id: &str,
	message: &str,
) -> bool {
	let body = server_frame::Body::EventStreamError(pb::EventStreamError {
		stream:         pb::EventStreamKind::Invocation as i32,
		failure:        pb::EventStreamFailure::Closed as i32,
		invocation_id:  invocation_id.to_owned(),
		exec:           Bytes::new(),
		process_name:   String::new(),
		skipped_events: 0,
		message:        message.to_owned(),
		props:          Default::default(),
	});
	send_invocation_terminal_body(responses, request_id, body).await;
	true
}

fn spawn_exec(
	request_id: u64,
	run: super::exec::ExecRun,
	cancel: CancellationToken,
	responses: flume::Sender<pb::ServerFrame>,
	finished: flume::Sender<Finished>,
) {
	tokio::spawn(async move {
		let exec = Bytes::copy_from_slice(run.id());
		let mut terminal = false;
		loop {
			let event = tokio::select! {
				() = cancel.cancelled() => break,
				event = run.next_event() => event,
			};
			match event {
				Some(ExecEvent::Output(output)) => {
					send_body(&responses, request_id, server_frame::Body::Output(output)).await
				},
				Some(ExecEvent::Exit(exit)) => {
					terminal = true;
					send_body(&responses, request_id, server_frame::Body::Exit(exit)).await;
					break;
				},
				None => break,
			}
		}
		if !terminal && !cancel.is_cancelled() {
			send_stream_error(
				&responses,
				request_id,
				pb::EventStreamKind::Exec,
				"",
				&exec,
				"",
				"exec event stream closed without ExitEvent",
			)
			.await;
		}
		let _ = finished
			.send_async(Finished { request_id, invocation_id: None })
			.await;
	});
}

fn spawn_process_attachment(
	request_id: u64,
	process_name: Str,
	events: flume::Receiver<ProcessEvent>,
	cancel: CancellationToken,
	responses: flume::Sender<pb::ServerFrame>,
	finished: flume::Sender<Finished>,
) {
	tokio::spawn(async move {
		loop {
			let event = tokio::select! {
				() = cancel.cancelled() => break,
				event = events.recv_async() => event.ok(),
			};
			match event {
				Some(ProcessEvent::Output(output)) => {
					send_body(&responses, request_id, server_frame::Body::ProcessOutput(output)).await;
				},
				Some(ProcessEvent::State(process)) => {
					send_body(
						&responses,
						request_id,
						server_frame::Body::ProcessState(pb::ProcessStateEvent {
							process: Some(process),
							props:   Default::default(),
						}),
					)
					.await;
				},
				None => {
					send_stream_error(
						&responses,
						request_id,
						pb::EventStreamKind::ProcessOutput,
						"",
						&[],
						&process_name,
						"named-process output stream closed",
					)
					.await;
					break;
				},
			}
		}
		let _ = finished
			.send_async(Finished { request_id, invocation_id: None })
			.await;
	});
}

fn spawn_blob_get(
	request_id: u64,
	read: super::blobs::BlobRead,
	cancel: CancellationToken,
	responses: flume::Sender<pb::ServerFrame>,
	finished: flume::Sender<Finished>,
) {
	tokio::spawn(async move {
		if read.data.is_empty() {
			let send = send_body(
				&responses,
				request_id,
				server_frame::Body::BlobChunk(blob_pb::Chunk {
					data: Bytes::new(),
					hash: Bytes::copy_from_slice(&read.id.hash),
					size: Some(read.id.size),
				}),
			);
			tokio::select! {
				() = cancel.cancelled() => {},
				() = send => {},
			}
		}
		let mut offset = 0;
		while offset < read.data.len() {
			let first = offset == 0;
			let end = (offset + BLOB_CHUNK_BYTES).min(read.data.len());
			let send = send_body(
				&responses,
				request_id,
				server_frame::Body::BlobChunk(blob_pb::Chunk {
					data: read.data.slice(offset..end),
					hash: if first {
						Bytes::copy_from_slice(&read.id.hash)
					} else {
						Bytes::new()
					},
					size: first.then_some(read.id.size),
				}),
			);
			tokio::select! {
				() = cancel.cancelled() => break,
				() = send => offset = end,
			}
		}
		if !cancel.is_cancelled() {
			send_body(
				&responses,
				request_id,
				server_frame::Body::BlobGetComplete(pb::BlobGetComplete {
					hash:       Bytes::copy_from_slice(&read.id.hash),
					bytes_sent: read.data.len() as u64,
					props:      Default::default(),
				}),
			)
			.await;
		}
		let _ = finished
			.send_async(Finished { request_id, invocation_id: None })
			.await;
	});
}

fn worker_verdict_json(details: Bytes, is_error: bool) -> Result<Bytes, serde_json::Error> {
	let _: &serde_json::value::RawValue = serde_json::from_slice(&details)?;
	let prefix: &[u8] = if is_error {
		br#"{"kind":"fault","value":"#
	} else {
		br#"{"kind":"ok","value":"#
	};
	let mut verdict = BytesMut::with_capacity(prefix.len() + details.len() + 1);
	verdict.extend_from_slice(prefix);
	verdict.extend_from_slice(&details);
	verdict.extend_from_slice(b"}");
	Ok(verdict.freeze())
}

fn erased_outcome_wire(outcome: ErasedOutcome) -> (Bytes, bool, bool) {
	match outcome {
		ErasedOutcome::Done { verdict, useless } => {
			let is_error = serde_json::from_slice::<
				omp_tool::Verdict<serde_json::Value, serde_json::Value>,
			>(&verdict)
			.map_or(true, |verdict| !matches!(verdict, omp_tool::Verdict::Ok(_)));
			(verdict, is_error, useless)
		},
		ErasedOutcome::Detached(job) => {
			let json = serde_json::to_vec(
				&omp_tool::Outcome::<serde_json::Value, serde_json::Value>::Detached(job),
			)
			.map(Bytes::from)
			.unwrap_or_default();
			(json, false, false)
		},
	}
}

async fn send_exec_error(
	responses: &flume::Sender<pb::ServerFrame>,
	request_id: u64,
	error: &ExecError,
) {
	let code = match error {
		ExecError::SessionNotFound | ExecError::RunNotFound | ExecError::ProcessNotFound(_) => {
			pb::ProtocolErrorCode::NotFound
		},
		ExecError::ProcessExists(_) => pb::ProtocolErrorCode::AlreadyExists,
		ExecError::UnsupportedSignal(_) => pb::ProtocolErrorCode::Unsupported,
		ExecError::InvalidCwd(_) => pb::ProtocolErrorCode::InvalidArgument,
		ExecError::SessionClosed => pb::ProtocolErrorCode::PreconditionFailed,
		ExecError::Shell(_) | ExecError::Io(_) => pb::ProtocolErrorCode::Internal,
	};
	send_error(responses, request_id, code, &error.to_string()).await;
}

async fn send_blob_error(
	responses: &flume::Sender<pb::ServerFrame>,
	request_id: u64,
	error: &BlobError,
) {
	let code = match error {
		BlobError::InvalidHash
		| BlobError::HashMismatch
		| BlobError::SizeMismatch { .. }
		| BlobError::InvalidRange
		| BlobError::LengthOverflow => pb::ProtocolErrorCode::InvalidArgument,
		BlobError::Store(omp_storage::blob::Error::NotFound) => pb::ProtocolErrorCode::NotFound,
		BlobError::Store(_) | BlobError::Remove(_) => pb::ProtocolErrorCode::Internal,
	};
	send_error(responses, request_id, code, &error.to_string()).await;
}

async fn send_stream_error(
	responses: &flume::Sender<pb::ServerFrame>,
	request_id: u64,
	kind: pb::EventStreamKind,
	invocation_id: &str,
	exec: &[u8],
	process_name: &str,
	message: &str,
) {
	send_body(
		responses,
		request_id,
		server_frame::Body::EventStreamError(pb::EventStreamError {
			stream:         kind as i32,
			failure:        pb::EventStreamFailure::Closed as i32,
			invocation_id:  invocation_id.to_owned(),
			exec:           Bytes::copy_from_slice(exec),
			process_name:   process_name.to_owned(),
			skipped_events: 0,
			message:        message.to_owned(),
			props:          Default::default(),
		}),
	)
	.await;
}

async fn send_error(
	responses: &flume::Sender<pb::ServerFrame>,
	request_id: u64,
	code: pb::ProtocolErrorCode,
	message: &str,
) {
	send_body(
		responses,
		request_id,
		server_frame::Body::Error(pb::ProtocolError {
			code:    code as i32,
			message: message.to_owned(),
			props:   Default::default(),
		}),
	)
	.await;
}

async fn send_body(
	responses: &flume::Sender<pb::ServerFrame>,
	request_id: u64,
	body: server_frame::Body,
) {
	let _ = responses
		.send_async(checked_server_frame(request_id, body))
		.await;
}

async fn send_invocation_body(
	responses: &flume::Sender<pb::ServerFrame>,
	request_id: u64,
	body: server_frame::Body,
) -> bool {
	matches!(
		tokio::time::timeout(
			INVOCATION_RESPONSE_SEND_GRACE,
			responses.send_async(checked_server_frame(request_id, body)),
		)
		.await,
		Ok(Ok(()))
	)
}

async fn send_invocation_terminal_body(
	responses: &flume::Sender<pb::ServerFrame>,
	request_id: u64,
	body: server_frame::Body,
) {
	let frame = checked_server_frame(request_id, body);
	let retry = frame.clone();
	match tokio::time::timeout(INVOCATION_RESPONSE_SEND_GRACE, responses.send_async(frame)).await {
		Ok(_) => {},
		Err(_) => {
			let responses = responses.clone();
			tokio::spawn(async move {
				let _ = responses.send_async(retry).await;
			});
		},
	}
}

fn checked_server_frame(request_id: u64, body: server_frame::Body) -> pb::ServerFrame {
	let mut frame = server_frame(request_id, body);
	if frame.encoded_len() > FRAME_LIMIT {
		frame = server_frame(
			request_id,
			server_frame::Body::Error(pb::ProtocolError {
				code:    pb::ProtocolErrorCode::Internal as i32,
				message: "environment response exceeds the configured frame limit".to_owned(),
				props:   Default::default(),
			}),
		);
	}
	frame
}

fn server_frame(request_id: u64, body: server_frame::Body) -> pb::ServerFrame {
	pb::ServerFrame { request_id, body: Some(body), props: Default::default() }
}

async fn read_client_frame<R>(
	reader: &mut R,
	scratch: &mut BytesMut,
) -> io::Result<Option<pb::ClientFrame>>
where
	R: AsyncRead + Unpin,
{
	let Some(length) = read_length(reader).await? else {
		return Ok(None);
	};
	if length > FRAME_LIMIT {
		return Err(io::Error::new(
			io::ErrorKind::InvalidData,
			"environment frame exceeds the configured limit",
		));
	}
	scratch.clear();
	scratch.resize(length, 0);
	reader.read_exact(scratch).await?;
	pb::ClientFrame::decode(&scratch[..])
		.map(Some)
		.map_err(io::Error::other)
}

async fn write_server_frame<W>(
	writer: &mut W,
	frame: &pb::ServerFrame,
	scratch: &mut BytesMut,
) -> io::Result<()>
where
	W: AsyncWrite + Unpin,
{
	if frame.encoded_len() > FRAME_LIMIT {
		return Err(io::Error::new(
			io::ErrorKind::InvalidData,
			"environment frame exceeds the configured limit",
		));
	}
	scratch.clear();
	frame
		.encode_length_delimited(&mut *scratch)
		.map_err(io::Error::other)?;
	writer.write_all(scratch).await?;
	writer.flush().await
}

async fn read_length<R>(reader: &mut R) -> io::Result<Option<usize>>
where
	R: AsyncRead + Unpin,
{
	let mut value = 0_u64;
	for shift in (0..70).step_by(7) {
		let mut byte = [0_u8; 1];
		match reader.read_exact(&mut byte).await {
			Ok(_) => {},
			Err(error) if error.kind() == io::ErrorKind::UnexpectedEof && shift == 0 => {
				return Ok(None);
			},
			Err(error) => return Err(error),
		}
		let part = u64::from(byte[0] & 0x7f);
		if shift == 63 && part > 1 {
			return Err(io::Error::new(
				io::ErrorKind::InvalidData,
				"invalid environment frame length",
			));
		}
		value |= part << shift;
		if byte[0] & 0x80 == 0 {
			return usize::try_from(value).map(Some).map_err(io::Error::other);
		}
	}
	Err(io::Error::new(io::ErrorKind::InvalidData, "invalid environment frame length"))
}

/// Assembles and runs the standalone environment daemon with one empty shared
/// registry.
#[cfg(unix)]
pub async fn run(args: EnvdArgs) -> Result<(), EnvdError> {
	run_with_registry(args, Arc::new(Registry::new())).await
}

/// Assembles environment dispatch around a caller-shared revision registry.
#[cfg(unix)]
pub async fn run_with_registry(args: EnvdArgs, registry: Arc<Registry>) -> Result<(), EnvdError> {
	let workspace = WorkspaceHost::open(&args.root)?;
	let root = workspace.root().to_path_buf();
	let state_dir = args.state_dir.unwrap_or_else(|| root.join(".omp"));
	secure_owner_directory(&state_dir)?;
	let socket = args.socket.unwrap_or_else(|| state_dir.join("env.sock"));
	let docserver_socket = args
		.docserver_socket
		.unwrap_or_else(|| state_dir.join("docserver.sock"));
	let (documents, mut docserver) = connect_or_start_docserver(&root, &docserver_socket).await?;
	let hello = documents.hello().clone();
	let blobs = BlobHost::open(state_dir.join("blobs"))?;
	let exec = ExecHost::new();
	let workers = ToolWorkerSupervisor::spawn(ToolWorkerConfig::current()?).await?;
	let identity = ServerIdentity {
		workspace_id:   hello.workspace_id,
		root_uri:       hello.root_uri,
		server_epoch:   hello.server_epoch,
		server_version: Str::from(env!("CARGO_PKG_VERSION")),
	};
	let server =
		Arc::new(EnvServer::new(identity, documents, exec, workspace, blobs, registry, workers));
	let shutdown = CancellationToken::new();
	let signal = shutdown.clone();
	let signal_task = tokio::spawn(async move {
		let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate());
		match terminate.as_mut() {
			Ok(terminate) => {
				tokio::select! {
					_ = tokio::signal::ctrl_c() => {},
					_ = terminate.recv() => {},
				}
			},
			Err(_) => {
				let _ = tokio::signal::ctrl_c().await;
			},
		}
		signal.cancel();
	});
	server.serve_uds(&socket, shutdown).await?;
	signal_task.abort();
	if let Some(child) = docserver.as_mut() {
		let _ = child.kill().await;
	}
	Ok(())
}
/// Reports the Phase 1 transport limitation on platforms without Unix sockets.
#[cfg(not(unix))]
pub async fn run(_args: EnvdArgs) -> Result<(), EnvdError> {
	Err(
		io::Error::new(io::ErrorKind::Unsupported, "envd requires a Unix-domain socket in Phase 1")
			.into(),
	)
}

#[cfg(unix)]
async fn connect_or_start_docserver(
	root: &Path,
	socket: &Path,
) -> Result<(DocumentHost, Option<tokio::process::Child>), EnvdError> {
	if let Ok(stream) = tokio::net::UnixStream::connect(socket).await {
		return Ok((DocumentHost::connect(stream).await?, None));
	}
	if let Some(parent) = socket.parent() {
		secure_owner_directory(parent)?;
	}
	let executable = docserver_executable()?;
	let mut child = tokio::process::Command::new(executable)
		.arg("--root")
		.arg(root)
		.arg("--socket")
		.arg(socket)
		.kill_on_drop(true)
		.spawn()?;
	let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
	loop {
		if child.try_wait()?.is_some() {
			return Err(EnvdError::DocserverExited);
		}
		if let Ok(stream) = tokio::net::UnixStream::connect(socket).await {
			match DocumentHost::connect(stream).await {
				Ok(host) => return Ok((host, Some(child))),
				Err(error) if tokio::time::Instant::now() >= deadline => return Err(error.into()),
				Err(_) => {},
			}
		}
		if tokio::time::Instant::now() >= deadline {
			return Err(
				io::Error::new(io::ErrorKind::TimedOut, "document-server hello timed out").into(),
			);
		}
		tokio::time::sleep(Duration::from_millis(25)).await;
	}
}

#[cfg(unix)]
fn secure_owner_directory(path: &Path) -> io::Result<()> {
	use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

	match std::fs::create_dir(path) {
		Ok(()) => std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?,
		Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {},
		Err(error) if error.kind() == io::ErrorKind::NotFound => {
			std::fs::create_dir_all(path)?;
			std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
		},
		Err(error) => return Err(error),
	}
	let metadata = std::fs::symlink_metadata(path)?;
	let owner = nix::unistd::geteuid().as_raw();
	if !metadata.is_dir() || metadata.uid() != owner || metadata.mode() & 0o077 != 0 {
		return Err(io::Error::new(
			io::ErrorKind::PermissionDenied,
			"environment socket directory must be owner-only and owned by the current user",
		));
	}
	Ok(())
}

#[cfg(unix)]
fn docserver_executable() -> io::Result<PathBuf> {
	if let Some(path) = std::env::var_os("OMP_DOCSERVER_BIN") {
		return Ok(PathBuf::from(path));
	}
	let current = std::env::current_exe()?;
	let sibling = current.with_file_name("omp-docserverd");
	Ok(if sibling.is_file() {
		sibling
	} else {
		PathBuf::from("omp-docserverd")
	})
}
