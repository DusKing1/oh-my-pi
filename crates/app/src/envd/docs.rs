//! Document-server connection and revision-pinned document operations.

use std::{
	collections::HashMap,
	path::Path,
	sync::{
		Arc,
		atomic::{AtomicU64, Ordering},
	},
};

use bytes::{Bytes, BytesMut};
use omp_core::Str;
use omp_docserver::{
	connection::{PROTOCOL_MAJOR, PROTOCOL_MINOR},
	wire::{self, FrameConfig},
};
use omp_proto::document::v1 as pb;
use parking_lot::Mutex;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio_util::sync::CancellationToken;

/// Metadata established by the document protocol hello exchange.
#[derive(Clone, Debug)]
pub struct DocumentHello {
	/// Negotiated protocol major version.
	pub protocol_major: u32,
	/// Negotiated protocol minor version.
	pub protocol_minor: u32,
	/// Stable identity of the connected document workspace.
	pub workspace_id:   Bytes,
	/// Canonical file URI of the connected workspace root.
	pub root_uri:       Str,
	/// Epoch scoping transaction idempotency keys.
	pub server_epoch:   Bytes,
}

/// A document-server lease pinned to the revision returned by `OpenDocument`.
///
/// Dropping the lease sends a best-effort close request, keeping lease release
/// resource-owned even when an executor future is cancelled.
#[derive(Debug)]
pub struct DocumentLease {
	lease_id: Bytes,
	head:     pb::DocumentHead,
	host:     Arc<Inner>,
	released: bool,
}

impl DocumentLease {
	/// Returns the opaque connection-owned lease identity.
	pub fn id(&self) -> &Bytes {
		&self.lease_id
	}

	/// Returns the immutable head to which reads and edits are pinned.
	pub fn head(&self) -> &pb::DocumentHead {
		&self.head
	}

	fn revision(&self) -> Result<pb::Revision, DocumentError> {
		self
			.head
			.revision
			.clone()
			.ok_or(DocumentError::MalformedResponse(Str::new_static(
				"document head omitted its revision",
			)))
	}
}
impl Drop for DocumentLease {
	fn drop(&mut self) {
		if self.released || self.host.shutdown.is_cancelled() {
			return;
		}
		let request_id = self.host.next_request.fetch_add(1, Ordering::Relaxed);
		if request_id == 0 {
			return;
		}
		let _ = self.host.writer.try_send(pb::ClientFrame {
			request_id,
			body: Some(pb::client_frame::Body::CloseDocument(pb::CloseDocumentRequest {
				lease_id: self.lease_id.clone(),
			})),
		});
	}
}

/// A document host connection, protocol, or server operation failed.
#[derive(Debug, Error)]
pub enum DocumentError {
	#[error(transparent)]
	Wire(#[from] wire::WireError),
	#[error("document-server connection closed")]
	Disconnected,
	#[error("document operation was cancelled")]
	Cancelled,
	#[error("document server rejected the operation ({code}): {message}")]
	Protocol { code: i32, message: Str },
	#[error("malformed document-server response: {0}")]
	MalformedResponse(Str),
}

#[derive(Debug)]
struct Inner {
	hello:        DocumentHello,
	writer:       flume::Sender<pb::ClientFrame>,
	pending:      Arc<Mutex<HashMap<u64, flume::Sender<pb::ServerFrame>>>>,
	next_request: AtomicU64,
	shutdown:     CancellationToken,
}

/// Concrete env-side owner of one multiplexed `document/v1` client connection.
#[derive(Clone, Debug)]
pub struct DocumentHost {
	inner: Arc<Inner>,
}

impl DocumentHost {
	/// Connects to an already-running document server and completes its hello.
	pub async fn connect<S>(stream: S) -> Result<Self, DocumentError>
	where
		S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
	{
		let config = FrameConfig::default();
		let (mut reader, mut writer) = tokio::io::split(stream);
		let mut write_scratch = BytesMut::new();
		wire::write_client_frame(
			&mut writer,
			&pb::ClientFrame {
				request_id: 0,
				body:       Some(pb::client_frame::Body::Hello(pb::ClientHello {
					protocol_major: PROTOCOL_MAJOR,
					protocol_minor: PROTOCOL_MINOR,
					client_id:      Bytes::new(),
				})),
			},
			config,
			&mut write_scratch,
		)
		.await?;

		let mut read_scratch = BytesMut::new();
		let hello_frame = wire::read_server_frame(&mut reader, config, &mut read_scratch)
			.await?
			.ok_or(DocumentError::Disconnected)?;
		let hello = match hello_frame.body {
			Some(pb::server_frame::Body::Hello(hello)) if hello_frame.request_id == 0 => hello,
			Some(pb::server_frame::Body::Error(error)) => {
				return Err(DocumentError::Protocol {
					code:    error.code,
					message: Str::from(error.message),
				});
			},
			_ => {
				return Err(DocumentError::MalformedResponse(Str::new_static(
					"expected ServerHello as the first server frame",
				)));
			},
		};
		if hello.protocol_major != PROTOCOL_MAJOR || hello.protocol_minor > PROTOCOL_MINOR {
			return Err(DocumentError::MalformedResponse(Str::new_static(
				"document server negotiated an unsupported protocol version",
			)));
		}
		let hello = DocumentHello {
			protocol_major: hello.protocol_major,
			protocol_minor: hello.protocol_minor,
			workspace_id:   hello.workspace_id,
			root_uri:       Str::from(hello.root_uri),
			server_epoch:   hello.server_epoch,
		};

		let (write_tx, write_rx) = flume::unbounded();
		let inner = Arc::new(Inner {
			hello,
			writer: write_tx,
			pending: Arc::new(Mutex::new(HashMap::new())),
			next_request: AtomicU64::new(1),
			shutdown: CancellationToken::new(),
		});

		let writer_shutdown = inner.shutdown.clone();
		tokio::spawn(async move {
			let mut scratch = write_scratch;
			while let Ok(frame) = write_rx.recv_async().await {
				if wire::write_client_frame(&mut writer, &frame, config, &mut scratch)
					.await
					.is_err()
				{
					break;
				}
			}
			writer_shutdown.cancel();
		});

		let reader_pending = Arc::clone(&inner.pending);
		let reader_shutdown = inner.shutdown.clone();
		tokio::spawn(async move {
			loop {
				let frame = tokio::select! {
					_ = reader_shutdown.cancelled() => break,
					result = wire::read_server_frame(&mut reader, config, &mut read_scratch) => {
						match result {
							Ok(Some(frame)) => frame,
							Ok(None) | Err(_) => break,
						}
					},
				};
				if frame.request_id == 0 {
					continue;
				}
				if let Some(waiter) = reader_pending.lock().remove(&frame.request_id) {
					let _ = waiter.send(frame);
				}
			}
			reader_shutdown.cancel();
			let waiters = std::mem::take(&mut *reader_pending.lock());
			for (request_id, waiter) in waiters {
				let _ = waiter.send(disconnected_frame(request_id));
			}
		});

		Ok(Self { inner })
	}

	/// Connects to an already-running document server over a Unix-domain socket.
	#[cfg(unix)]
	pub async fn connect_uds(path: impl AsRef<Path>) -> Result<Self, DocumentError> {
		Self::connect(
			tokio::net::UnixStream::connect(path)
				.await
				.map_err(wire::WireError::from)?,
		)
		.await
	}

	/// Returns the negotiated server and workspace identity.
	pub fn hello(&self) -> &DocumentHello {
		&self.inner.hello
	}

	/// Acquires a document lease and pins it to the returned immutable revision.
	pub async fn open(
		&self,
		uri: Str,
		language_id: Option<Str>,
		cancel: &CancellationToken,
	) -> Result<DocumentLease, DocumentError> {
		let body = self
			.request(
				pb::client_frame::Body::OpenDocument(pb::OpenDocumentRequest {
					uri:         uri.into(),
					language_id: language_id.unwrap_or_default().into(),
				}),
				cancel,
			)
			.await?;
		let pb::server_frame::Body::DocumentOpened(opened) = body else {
			return Err(unexpected("OpenDocumentResponse"));
		};
		let head = opened
			.head
			.ok_or_else(|| unexpected("OpenDocumentResponse.head"))?;
		if opened.lease_id.len() != 16 || head.revision.is_none() {
			return Err(unexpected("valid lease id and pinned revision"));
		}
		Ok(DocumentLease {
			lease_id: opened.lease_id,
			head,
			host: Arc::clone(&self.inner),
			released: false,
		})
	}

	/// Reads ranges from the exact revision pinned by `lease`.
	pub async fn read(
		&self,
		lease: &DocumentLease,
		selection: pb::ReadSelection,
		cancel: &CancellationToken,
	) -> Result<pb::ReadDocumentResponse, DocumentError> {
		self.ensure_owned(lease)?;
		let body = self
			.request(
				pb::client_frame::Body::ReadDocument(pb::ReadDocumentRequest {
					document:  Some(lease_target(lease)),
					revision:  Some(lease.revision()?),
					selection: Some(selection),
				}),
				cancel,
			)
			.await?;
		let pb::server_frame::Body::DocumentRead(response) = body else {
			return Err(unexpected("ReadDocumentResponse"));
		};
		ensure_pinned_head(response.head.as_ref(), lease)?;
		Ok(response)
	}

	/// Produces a structural summary from the exact revision pinned by `lease`.
	pub async fn summarize(
		&self,
		lease: &DocumentLease,
		options: pb::CodeSummaryOptions,
		cancel: &CancellationToken,
	) -> Result<pb::SummarizeDocumentResponse, DocumentError> {
		self.ensure_owned(lease)?;
		let body = self
			.request(
				pb::client_frame::Body::SummarizeDocument(pb::SummarizeDocumentRequest {
					document: Some(lease_target(lease)),
					revision: Some(lease.revision()?),
					options:  Some(options),
				}),
				cancel,
			)
			.await?;
		let pb::server_frame::Body::DocumentSummarized(response) = body else {
			return Err(unexpected("SummarizeDocumentResponse"));
		};
		ensure_pinned_head(response.head.as_ref(), lease)?;
		Ok(response)
	}

	/// Commits one text mutation against the lease's pinned base revision.
	///
	/// The lease advances only after a committed operation; rejected and partial
	/// outcomes retain the old pin so callers cannot accidentally write from an
	/// unobserved head.
	pub async fn commit(
		&self,
		lease: &mut DocumentLease,
		transaction_id: Bytes,
		mut mutation: pb::TextMutation,
		cancel: &CancellationToken,
	) -> Result<pb::CommitTransactionResponse, DocumentError> {
		self.ensure_owned(lease)?;
		mutation.base_revision = Some(lease.revision()?);
		let body = self
			.request(
				pb::client_frame::Body::CommitTransaction(pb::CommitTransactionRequest {
					transaction_id,
					operations: vec![pb::DocumentMutation {
						document:  Some(lease_target(lease)),
						operation: Some(pb::document_mutation::Operation::Text(mutation)),
					}],
				}),
				cancel,
			)
			.await?;
		let pb::server_frame::Body::TransactionResult(response) = body else {
			return Err(unexpected("CommitTransactionResponse"));
		};
		if let Some(pb::commit_transaction_response::Outcome::Committed(committed)) =
			&response.outcome
		{
			let Some(head) = (committed.operations.len() == 1)
				.then(|| committed.operations[0].head.clone())
				.flatten()
			else {
				return Err(unexpected("one committed operation head"));
			};
			if head.revision.is_none() {
				return Err(unexpected("committed operation revision"));
			}
			lease.head = head;
		}
		Ok(response)
	}

	/// Releases a connection-owned document lease.
	pub async fn close(
		&self,
		mut lease: DocumentLease,
		cancel: &CancellationToken,
	) -> Result<(), DocumentError> {
		self.ensure_owned(&lease)?;
		let body = self
			.request(
				pb::client_frame::Body::CloseDocument(pb::CloseDocumentRequest {
					lease_id: lease.lease_id.clone(),
				}),
				cancel,
			)
			.await?;
		match body {
			pb::server_frame::Body::DocumentClosed(_) => {
				lease.released = true;
				Ok(())
			},
			_ => Err(unexpected("CloseDocumentResponse")),
		}
	}

	fn ensure_owned(&self, lease: &DocumentLease) -> Result<(), DocumentError> {
		if Arc::ptr_eq(&self.inner, &lease.host) {
			Ok(())
		} else {
			Err(DocumentError::MalformedResponse(Str::new_static(
				"document lease belongs to another document connection",
			)))
		}
	}

	async fn request(
		&self,
		body: pb::client_frame::Body,
		cancel: &CancellationToken,
	) -> Result<pb::server_frame::Body, DocumentError> {
		if self.inner.shutdown.is_cancelled() {
			return Err(DocumentError::Disconnected);
		}
		let request_id = self.inner.next_request.fetch_add(1, Ordering::Relaxed);
		if request_id == 0 {
			return Err(DocumentError::Disconnected);
		}
		let (response_tx, response_rx) = flume::bounded(1);
		self.inner.pending.lock().insert(request_id, response_tx);
		let mut pending = PendingRequest { inner: Arc::clone(&self.inner), request_id, armed: true };
		self
			.inner
			.writer
			.send_async(pb::ClientFrame { request_id, body: Some(body) })
			.await
			.map_err(|_| DocumentError::Disconnected)?;
		let frame = tokio::select! {
			_ = cancel.cancelled() => return Err(DocumentError::Cancelled),
			_ = self.inner.shutdown.cancelled() => return Err(DocumentError::Disconnected),
			result = response_rx.recv_async() => result.map_err(|_| DocumentError::Disconnected)?,
		};
		pending.armed = false;
		match frame.body {
			Some(pb::server_frame::Body::Error(error)) => {
				Err(DocumentError::Protocol { code: error.code, message: Str::from(error.message) })
			},
			Some(body) => Ok(body),
			None => Err(unexpected("non-empty server frame")),
		}
	}
}

impl Drop for Inner {
	fn drop(&mut self) {
		self.shutdown.cancel();
	}
}

struct PendingRequest {
	inner:      Arc<Inner>,
	request_id: u64,
	armed:      bool,
}

impl Drop for PendingRequest {
	fn drop(&mut self) {
		if !self.armed || self.inner.pending.lock().remove(&self.request_id).is_none() {
			return;
		}
		let _ = self.inner.writer.try_send(pb::ClientFrame {
			request_id: 0,
			body:       Some(pb::client_frame::Body::Cancel(pb::CancelRequest {
				target_request_id: self.request_id,
			})),
		});
	}
}
fn ensure_pinned_head(
	head: Option<&pb::DocumentHead>,
	lease: &DocumentLease,
) -> Result<(), DocumentError> {
	let Some(head) = head else {
		return Err(unexpected("response head"));
	};
	if head.revision != lease.head.revision {
		return Err(DocumentError::MalformedResponse(Str::new_static(
			"document server returned a revision other than the requested pin",
		)));
	}
	Ok(())
}

fn lease_target(lease: &DocumentLease) -> pb::DocumentTarget {
	pb::DocumentTarget { target: Some(pb::document_target::Target::LeaseId(lease.lease_id.clone())) }
}

fn unexpected(expected: &'static str) -> DocumentError {
	DocumentError::MalformedResponse(Str::new(expected))
}

fn disconnected_frame(request_id: u64) -> pb::ServerFrame {
	pb::ServerFrame {
		request_id,
		body: Some(pb::server_frame::Body::Error(pb::ProtocolError {
			code:    pb::ProtocolErrorCode::Internal.into(),
			message: "document-server connection closed".to_owned(),
		})),
	}
}
