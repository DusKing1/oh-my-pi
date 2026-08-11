//! Protobuf request conversion and dispatch for one document-server session.

use std::{
	future::Future,
	path::PathBuf,
	sync::Arc,
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use omp_core::Str;
use omp_proto::{document::v1 as proto, prost::Message};
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::{
	ByteEdit, ByteRange, DocumentHead, DocumentId, DocumentKind, DocumentLocator, DocumentPresence,
	DocumentSnapshot, EnvironmentSession, Error, FileKind, FollowSymlinks, LanguageId, LeaseId,
	LineRange, PathMetadata, PortablePermissions, ReadBody, ReadSelection, Revision, SymlinkTarget,
	SymlinkTargetForm, SymlinkTargetKind, TransactionId,
	lsp::{LspError, LspResponseOutcome, LspTransportError, TextDocumentSyncKind},
	lsp_registry::{
		DocumentEventStreamError, LspBindingId, LspRegistryError, LspRegistryEvent,
		StaleResponsePolicy,
	},
	path_ops::PathMutationResult,
	summary::{
		DocumentSummary, SummaryFallback, SummaryOptions, SummaryOutcome, SummaryRenderMode,
		SummarySegment, SummaryUnavailableReason,
	},
	transaction::{
		CreateMutation, DeleteMutation, DocumentMutation, DocumentTarget, ExistingDocumentPolicy,
		FormatPolicy, MoveDestinationPrecondition, MoveMutation, MutationOperation, OperationResult,
		StalePolicy, TextMutation, TextProposal, TransactionBuildError, TransactionOutcome,
		TransactionRejectReason,
	},
};

/// Dispatches one post-handshake request body. Framing, hello, and cancellation
/// routing remain connection-owned.
pub async fn dispatch_request(
	session: EnvironmentSession,
	request_id: u64,
	body: proto::client_frame::Body,
	protocol_minor: u32,
	events: flume::Sender<proto::ServerFrame>,
	event_frame_limit: usize,
	cancellation: CancellationToken,
) -> proto::ServerFrame {
	let result =
		dispatch(&session, body, protocol_minor, events, event_frame_limit, cancellation).await;
	proto::ServerFrame {
		request_id,
		body: Some(match result {
			Ok(body) => body,
			Err(error) => proto::server_frame::Body::Error(error.into_proto()),
		}),
	}
}

const CLOSE_SESSION_LEASE_DEADLINE: Duration = Duration::from_secs(1);

/// Cancels every session event forwarder and releases every registry-owned
/// lease, balancing LSP and document-store ownership.
pub async fn close_session(session: &EnvironmentSession) {
	let leases = session.take_leases();
	for lease_id in leases {
		let cancellation = CancellationToken::new();
		let close = session
			.environment()
			.lsp()
			.close_document(lease_id, cancellation.child_token());
		let _ = await_cooperative_cleanup(&cancellation, CLOSE_SESSION_LEASE_DEADLINE, close).await;
	}
}

async fn await_cooperative_cleanup<T>(
	cancellation: &CancellationToken,
	deadline: Duration,
	cleanup: impl Future<Output = T>,
) -> T {
	tokio::pin!(cleanup);
	tokio::select! {
		biased;
		output = &mut cleanup => output,
		() = tokio::time::sleep(deadline) => {
			cancellation.cancel();
			cleanup.await
		},
	}
}
/// Converts one registry-wide LSP event into a session-visible unsolicited
/// frame, filtering document-scoped events without a connection-owned lease.
pub async fn registry_event_frame(
	session: &EnvironmentSession,
	event: LspRegistryEvent,
) -> Option<proto::ServerFrame> {
	let body = match event {
		LspRegistryEvent::Inbound(event) => {
			if !inbound_event_is_resolved(
				event.method(),
				event.params_json(),
				event.document_identity().is_some(),
				event.revision().is_some(),
			) {
				return None;
			}
			if let Some(document_id) = event.document_id()
				&& session.lease_for_document(document_id).is_none()
			{
				return None;
			}
			let document = event
				.document_identity()
				.map(|(document_id, uri)| proto::DocumentRef {
					id:  Bytes::copy_from_slice(document_id.as_bytes()),
					uri: uri.to_string(),
				});
			proto::server_frame::Body::LspEvent(proto::LspEvent {
				server_id: binding_id_bytes(event.binding_id()),
				method: event.method().to_owned(),
				params_json: event.params_json().clone(),
				document,
				revision: event.revision().map(revision_to_proto),
			})
		},
		LspRegistryEvent::Binding(event) => {
			let document_id = event.document_id();
			let lease_id = if let Some(document_id) = document_id {
				match session.lease_for_document(document_id) {
					Some(lease_id) => Some(lease_id),
					None => return None,
				}
			} else {
				None
			};
			let binding = if let Some(lease_id) = lease_id {
				session
					.environment()
					.lsp()
					.lease_bindings(lease_id)
					.await
					.ok()
					.and_then(|bindings| {
						bindings
							.into_iter()
							.find(|binding| binding.info().id() == event.binding_id())
					})
					.as_ref()
					.map(binding_to_proto)
			} else {
				session
					.environment()
					.lsp()
					.bindings()
					.into_iter()
					.find(|binding| binding.id() == event.binding_id())
					.map(|binding| proto::LspServerBinding {
						server_id:         binding_id_bytes(binding.id()),
						name:              binding.spec().name().to_owned(),
						sync_policy:       None,
						capabilities_json: Bytes::new(),
					})
			}
			.or_else(|| {
				Some(proto::LspServerBinding {
					server_id:         binding_id_bytes(event.binding_id()),
					name:              String::new(),
					sync_policy:       None,
					capabilities_json: Bytes::new(),
				})
			});
			let document = match document_id {
				Some(document_id) => document_ref_to_proto(session, document_id).await,
				None => None,
			};
			proto::server_frame::Body::LspBindingEvent(proto::LspBindingEvent {
				kind: match event.kind() {
					crate::lsp_registry::LspBindingEventKind::Ready => proto::LspBindingEventKind::Ready,
					crate::lsp_registry::LspBindingEventKind::PolicyChanged => {
						proto::LspBindingEventKind::PolicyChanged
					},
					crate::lsp_registry::LspBindingEventKind::Restarted => {
						proto::LspBindingEventKind::Restarted
					},
					crate::lsp_registry::LspBindingEventKind::Stopped => {
						proto::LspBindingEventKind::Stopped
					},
				} as i32,
				binding,
				document,
			})
		},
	};
	Some(proto::ServerFrame { request_id: 0, body: Some(body) })
}

const EVENT_STREAM_ERROR_PROTOCOL_MINOR: u32 = 1;

fn document_event_stream_error_frame(
	protocol_minor: u32,
	lease_id: LeaseId,
	error: DocumentEventStreamError,
) -> proto::ServerFrame {
	let (failure, skipped_events, message, legacy_code) = match error {
		DocumentEventStreamError::Lagged { skipped } => (
			proto::EventStreamFailure::Lagged,
			skipped,
			format!("document event stream lagged by {skipped} events; reopen the document"),
			proto::ProtocolErrorCode::ContentModified,
		),
		DocumentEventStreamError::Synchronization { message } => (
			proto::EventStreamFailure::Synchronization,
			0,
			format!("document event synchronization failed: {message}; reopen the document"),
			proto::ProtocolErrorCode::Internal,
		),
		DocumentEventStreamError::Closed => (
			proto::EventStreamFailure::Closed,
			0,
			"document event stream closed unexpectedly; reopen the document".to_owned(),
			proto::ProtocolErrorCode::Internal,
		),
	};
	event_stream_error_frame(
		protocol_minor,
		proto::EventStreamKind::Document,
		failure,
		Bytes::copy_from_slice(lease_id.as_bytes()),
		skipped_events,
		message,
		legacy_code,
	)
}

/// Builds the terminal connection-wide LSP event continuity failure.
pub fn lsp_event_stream_error_frame(
	protocol_minor: u32,
	failure: proto::EventStreamFailure,
	skipped_events: u64,
) -> proto::ServerFrame {
	let (message, legacy_code) = match failure {
		proto::EventStreamFailure::Lagged => (
			format!(
				"LSP registry event stream lagged by {skipped_events} events; reconnect and reopen \
				 documents"
			),
			proto::ProtocolErrorCode::ContentModified,
		),
		_ => (
			"LSP registry event stream closed unexpectedly; reconnect and reopen documents".to_owned(),
			proto::ProtocolErrorCode::Internal,
		),
	};
	event_stream_error_frame(
		protocol_minor,
		proto::EventStreamKind::LspRegistry,
		failure,
		Bytes::new(),
		skipped_events,
		message,
		legacy_code,
	)
}

fn event_stream_error_frame(
	protocol_minor: u32,
	stream: proto::EventStreamKind,
	failure: proto::EventStreamFailure,
	lease_id: Bytes,
	skipped_events: u64,
	message: String,
	legacy_code: proto::ProtocolErrorCode,
) -> proto::ServerFrame {
	let body = if protocol_minor >= EVENT_STREAM_ERROR_PROTOCOL_MINOR {
		proto::server_frame::Body::EventStreamError(proto::EventStreamError {
			stream: stream as i32,
			failure: failure as i32,
			lease_id,
			skipped_events,
			message,
		})
	} else {
		proto::server_frame::Body::Error(proto::ProtocolError { code: legacy_code as i32, message })
	};
	proto::ServerFrame { request_id: 0, body: Some(body) }
}

async fn dispatch(
	session: &EnvironmentSession,
	body: proto::client_frame::Body,
	protocol_minor: u32,
	events: flume::Sender<proto::ServerFrame>,
	event_frame_limit: usize,
	cancellation: CancellationToken,
) -> DispatchResult<proto::server_frame::Body> {
	use proto::{client_frame::Body as Request, server_frame::Body as Response};
	match body {
		Request::Hello(_) => Err(Failure::invalid("ClientHello is connection-owned")),
		Request::Cancel(_) => Err(Failure::invalid("CancelRequest is connection-owned")),
		Request::OpenDocument(request) => {
			open_document(session, request, protocol_minor, events, event_frame_limit, cancellation)
				.await
				.map(Response::DocumentOpened)
		},
		Request::CloseDocument(request) => close_document(session, request, cancellation)
			.await
			.map(Response::DocumentClosed),
		Request::ReadDocument(request) => read_document(session, request, cancellation)
			.await
			.map(Response::DocumentRead),
		Request::SummarizeDocument(request) => summarize_document(session, request, cancellation)
			.await
			.map(Response::DocumentSummarized),
		Request::CommitTransaction(request) => commit_transaction(session, request, cancellation)
			.await
			.map(Response::TransactionResult),
		Request::GetLspBindings(request) => get_lsp_bindings(session, request, cancellation)
			.await
			.map(Response::LspBindings),
		Request::LspRequest(request) => lsp_request(session, request, cancellation)
			.await
			.map(Response::LspResponse),
		Request::LspNotification(request) => lsp_notification(session, request, cancellation)
			.await
			.map(Response::LspNotificationAccepted),
		Request::CanonicalizePath(request) => {
			canonicalize_path(session, request).map(Response::PathCanonicalized)
		},
		Request::StatPath(request) => stat_path(session, request).map(Response::PathStat),
		Request::ListDirectory(request) => {
			list_directory(session, request).map(Response::DirectoryListed)
		},
		Request::CreateDirectory(request) => create_directory(session, request, cancellation)
			.await
			.map(Response::DirectoryCreated),
		Request::RemovePath(request) => remove_path(session, request, cancellation)
			.await
			.map(Response::PathRemoved),
		Request::RenamePath(request) => rename_path(session, request, cancellation)
			.await
			.map(Response::PathRenamed),
		Request::CopyPath(request) => copy_path(session, request, cancellation)
			.await
			.map(Response::PathCopied),
		Request::ReadLink(request) => read_link(session, request).map(Response::LinkRead),
		Request::CreateSymlink(request) => create_symlink(session, request, cancellation)
			.await
			.map(Response::SymlinkCreated),
		Request::CreateHardLink(request) => create_hard_link(session, request, cancellation)
			.await
			.map(Response::HardLinkCreated),
		Request::SetPermissions(request) => set_permissions(session, request, cancellation)
			.await
			.map(Response::PermissionsSet),
	}
}

async fn open_document(
	session: &EnvironmentSession,
	request: proto::OpenDocumentRequest,
	protocol_minor: u32,
	events_sender: flume::Sender<proto::ServerFrame>,
	event_frame_limit: usize,
	cancellation: CancellationToken,
) -> DispatchResult<proto::OpenDocumentResponse> {
	let uri = parse_file_uri(&request.uri)?;
	let path = session
		.environment()
		.store()
		.resolve_entry_path(&uri)
		.map_err(Failure::from_core)?;
	let language = if request.language_id.is_empty() {
		None
	} else {
		Some(LanguageId::new(&request.language_id).map_err(Failure::from_core)?)
	};
	let lease = session
		.environment()
		.lsp()
		.open_document(path, language, cancellation.child_token())
		.await
		.map_err(Failure::from_registry)?;
	let (lease_id, head, _, receiver) = lease.into_parts();
	let forwarder_cancel = CancellationToken::new();
	let events_ready = CancellationToken::new();
	session.own_lease(lease_id, head.document_id(), forwarder_cancel.clone(), events_ready.clone());
	let response_head = head_to_proto(session, &head, &cancellation).await;
	let response_head = match response_head {
		Ok(head) => head,
		Err(error) => {
			close_owned_lease(session, lease_id).await;
			return Err(error);
		},
	};
	let event_session = session.clone();
	tokio::spawn(async move {
		tokio::select! {
			() = forwarder_cancel.cancelled() => return,
			() = events_ready.cancelled() => {},
		}
		loop {
			let received = tokio::select! {
				() = forwarder_cancel.cancelled() => break,
				event = receiver.recv_async() => event,
			};
			let event = match received {
				Ok(Ok(event)) => event,
				Ok(Err(error)) => {
					let frame = document_event_stream_error_frame(protocol_minor, lease_id, error);
					close_owned_lease(&event_session, lease_id).await;
					let _ = events_sender.send_async(frame).await;
					break;
				},
				Err(_) => {
					let frame = document_event_stream_error_frame(
						protocol_minor,
						lease_id,
						DocumentEventStreamError::Closed,
					);
					close_owned_lease(&event_session, lease_id).await;
					let _ = events_sender.send_async(frame).await;
					break;
				},
			};
			let body = match document_event_to_proto(&event_session, &event) {
				Ok(event) => proto::server_frame::Body::DocumentEvent(event),
				Err(error) => {
					let frame = document_event_stream_error_frame(
						protocol_minor,
						lease_id,
						DocumentEventStreamError::Synchronization { message: Str::new(error.message) },
					);
					close_owned_lease(&event_session, lease_id).await;
					let _ = events_sender.send_async(frame).await;
					break;
				},
			};
			let frame = proto::ServerFrame { request_id: 0, body: Some(body) };
			if frame.encoded_len() > event_frame_limit {
				let terminal = document_event_stream_error_frame(
					protocol_minor,
					lease_id,
					DocumentEventStreamError::Closed,
				);
				close_owned_lease(&event_session, lease_id).await;
				let _ = events_sender.send_async(terminal).await;
				break;
			}
			if events_sender.send_async(frame).await.is_err() {
				break;
			}
		}
	});
	Ok(proto::OpenDocumentResponse {
		lease_id: Bytes::copy_from_slice(lease_id.as_bytes()),
		head:     Some(response_head),
	})
}

async fn close_owned_lease(session: &EnvironmentSession, lease_id: LeaseId) {
	session.release_lease(lease_id);
	let cancellation = CancellationToken::new();
	let close = session
		.environment()
		.lsp()
		.close_document(lease_id, cancellation.child_token());
	let _ = await_cooperative_cleanup(&cancellation, CLOSE_SESSION_LEASE_DEADLINE, close).await;
}

async fn close_document(
	session: &EnvironmentSession,
	request: proto::CloseDocumentRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::CloseDocumentResponse> {
	let lease_id = parse_lease_id(&request.lease_id)?;
	if !session.owns_lease(lease_id) {
		return Err(Failure::not_found("document lease is not owned by this connection"));
	}
	let result = session
		.environment()
		.lsp()
		.close_document(lease_id, cancellation)
		.await
		.map_err(Failure::from_registry);
	let released = session.release_lease(lease_id);
	debug_assert!(released, "closed lease remained connection-owned");
	result?;
	Ok(proto::CloseDocumentResponse {})
}

async fn read_document(
	session: &EnvironmentSession,
	request: proto::ReadDocumentRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::ReadDocumentResponse> {
	let target = parse_target(required(request.document, "read document target")?)?;
	let revision = request.revision.map(parse_revision).transpose()?;
	let selection = parse_read_selection(required(request.selection, "read selection")?)?;
	let locator = locator_for_target(session, &target)?;
	let selected = tokio::select! {
		biased;
		() = cancellation.cancelled() => return Err(Failure::cancelled("read request cancelled")),
		result = session.environment().store().read(locator.clone(), revision, selection.clone()) => {
			result.map_err(Failure::from_core)?
		},
	};
	let retained = tokio::select! {
		biased;
		() = cancellation.cancelled() => return Err(Failure::cancelled("read request cancelled")),
		result = session.environment().store().read(
			locator.clone(),
			Some(selected.head().revision()),
			ReadSelection::Whole,
		) => result.map_err(Failure::from_core)?,
	};
	let ReadBody::Whole(content) = retained.body() else {
		return Err(Failure::internal("whole snapshot read returned slices"));
	};
	let snapshot = Arc::new(
		DocumentSnapshot::new(retained.head().clone(), content.clone())
			.map_err(Failure::from_core)?,
	);
	let path = canonical_path_for_locator(session, locator, &cancellation).await?;
	if cancellation.is_cancelled() {
		return Err(Failure::cancelled("read request cancelled"));
	}
	session
		.edit_adapters()
		.record_snapshot(&path, snapshot, &selection)
		.map_err(Failure::from_core)?;
	let body = match selected.body() {
		ReadBody::Whole(content) => proto::read_document_response::Body::Content(content.clone()),
		ReadBody::Slices(slices) => {
			proto::read_document_response::Body::Slices(proto::ContentSlices {
				slices: slices
					.iter()
					.map(|slice| proto::ContentSlice {
						start:   slice.start(),
						end:     slice.end(),
						content: slice.content().clone(),
					})
					.collect(),
			})
		},
	};
	Ok(proto::ReadDocumentResponse {
		head: Some(head_to_proto(session, selected.head(), &cancellation).await?),
		body: Some(body),
	})
}

async fn summarize_document(
	session: &EnvironmentSession,
	request: proto::SummarizeDocumentRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::SummarizeDocumentResponse> {
	let target = parse_target(required(request.document, "summary document target")?)?;
	let revision = request.revision.map(parse_revision).transpose()?;
	let options = parse_summary_options(required(request.options, "summary options")?)?;
	let locator = locator_for_target(session, &target)?;
	let read = tokio::select! {
		biased;
		() = cancellation.cancelled() => {
			return Err(Failure::cancelled("summary request cancelled"));
		},
		result = session.environment().store().read(
			locator.clone(),
			revision,
			ReadSelection::Whole,
		) => result.map_err(Failure::from_core)?,
	};
	let ReadBody::Whole(content) = read.body() else {
		return Err(Failure::internal("whole snapshot read returned slices"));
	};
	let snapshot = Arc::new(
		DocumentSnapshot::new(read.head().clone(), content.clone()).map_err(Failure::from_core)?,
	);
	let path = canonical_path_for_locator(session, locator, &cancellation).await?;
	let outcome = session
		.environment()
		.summaries()
		.summarize(snapshot, &path, options, &cancellation)
		.await;
	let outcome = match outcome {
		SummaryOutcome::Available(summary) => {
			proto::summarize_document_response::Outcome::Summary(summary_to_proto(&summary))
		},
		SummaryOutcome::Fallback(fallback) => {
			proto::summarize_document_response::Outcome::Unavailable(fallback_to_proto(&fallback))
		},
		SummaryOutcome::Cancelled => return Err(Failure::cancelled("summary request cancelled")),
	};
	Ok(proto::SummarizeDocumentResponse {
		head:    Some(head_to_proto(session, read.head(), &cancellation).await?),
		outcome: Some(outcome),
	})
}

async fn commit_transaction(
	session: &EnvironmentSession,
	request: proto::CommitTransactionRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::CommitTransactionResponse> {
	let transaction_id = parse_transaction_id(&request.transaction_id)?;
	let build_session = session.clone();
	let operations = request.operations;
	let build_cancellation = cancellation.child_token();
	let outcome = session
		.environment()
		.transactions()
		.commit_lazy(transaction_id, cancellation, move || async move {
			build_operations(build_session, operations, build_cancellation).await
		})
		.await;
	Ok(transaction_outcome_to_proto(outcome.as_ref()))
}

async fn build_operations(
	session: EnvironmentSession,
	operations: Vec<proto::DocumentMutation>,
	cancellation: CancellationToken,
) -> Result<Vec<DocumentMutation>, TransactionBuildError> {
	let mut built = Vec::with_capacity(operations.len());
	for operation in operations {
		if cancellation.is_cancelled() {
			return Err(build_cancelled("transaction cancelled during operation building"));
		}
		let target = operation
			.document
			.ok_or_else(|| build_invalid("document mutation target is required"))
			.and_then(|target| parse_target(target).map_err(build_from_failure))?;
		locator_for_target(&session, &target).map_err(|error| build_precondition(error.message))?;
		let native = match operation
			.operation
			.ok_or_else(|| build_invalid("document mutation operation is required"))?
		{
			proto::document_mutation::Operation::Text(text) => MutationOperation::Text(
				build_text_mutation(&session, &target, text, &cancellation).await?,
			),
			proto::document_mutation::Operation::Create(create) => {
				MutationOperation::Create(build_create_mutation(create)?)
			},
			proto::document_mutation::Operation::Delete(delete) => {
				let revision = delete
					.base_revision
					.ok_or_else(|| build_invalid("delete base revision is required"))
					.and_then(|revision| parse_revision(revision).map_err(build_from_failure))?;
				MutationOperation::Delete(DeleteMutation::new(revision))
			},
			proto::document_mutation::Operation::Move(moved) => {
				MutationOperation::Move(build_move_mutation(moved)?)
			},
		};
		built.push(DocumentMutation::new(target, native));
	}
	Ok(built)
}

async fn build_text_mutation(
	session: &EnvironmentSession,
	target: &DocumentTarget,
	text: proto::TextMutation,
	cancellation: &CancellationToken,
) -> Result<TextMutation, TransactionBuildError> {
	let base_revision = text
		.base_revision
		.ok_or_else(|| build_invalid("text base revision is required"))
		.and_then(|revision| parse_revision(revision).map_err(build_from_failure))?;
	let stale_policy = parse_stale_policy(text.stale_policy).map_err(build_from_failure)?;
	let format_policy = parse_format_policy(text.format_policy).map_err(build_from_failure)?;
	let change = text
		.change
		.ok_or_else(|| build_invalid("text mutation change is required"))?;
	let proposal = match change {
		proto::text_mutation::Change::ProposedContent(content) => TextProposal::Content(content),
		proto::text_mutation::Change::Edits(edits) => {
			let edits = edits
				.edits
				.into_iter()
				.map(|edit| {
					ByteRange::new(edit.start, edit.end)
						.map(|range| ByteEdit::new(range, edit.replacement))
				})
				.collect::<crate::Result<Vec<_>>>()
				.map_err(|error| build_invalid(error.to_string()))?;
			TextProposal::Edits(edits)
		},
		proto::text_mutation::Change::Proposal(proposal) => {
			if proposal.format.is_empty() {
				return Err(build_invalid("edit proposal format must not be empty"));
			}
			let locator = locator_for_target(session, target)
				.map_err(|error| build_precondition(error.message))?;
			let read = tokio::select! {
				biased;
				() = cancellation.cancelled() => {
					return Err(build_cancelled("transaction cancelled during proposal lowering"));
				},
				result = session.environment().store().read(
					locator.clone(),
					Some(base_revision),
					ReadSelection::Whole,
				) => result.map_err(build_snapshot_error)?,
			};
			let ReadBody::Whole(content) = read.body() else {
				return Err(build_precondition("whole base snapshot read returned slices"));
			};
			let snapshot = Arc::new(
				DocumentSnapshot::new(read.head().clone(), content.clone())
					.map_err(|error| build_invalid(error.to_string()))?,
			);
			let path = canonical_path_for_locator(session, locator, cancellation)
				.await
				.map_err(|error| build_precondition(error.message))?;
			if cancellation.is_cancelled() {
				return Err(build_cancelled("transaction cancelled during proposal lowering"));
			}
			let edits = session
				.edit_adapters()
				.lower(&proposal.format, &path, snapshot, proposal.payload, proposal.options_json)
				.map_err(|error| build_invalid(error.to_string()))?;
			TextProposal::Edits(edits)
		},
	};
	Ok(TextMutation::new(base_revision, proposal, stale_policy, format_policy))
}

fn build_create_mutation(
	create: proto::CreateMutation,
) -> Result<CreateMutation, TransactionBuildError> {
	let existing = match proto::ExistingDocumentPolicy::try_from(create.existing_document)
		.map_err(|_| build_invalid("unknown existing document policy"))?
	{
		proto::ExistingDocumentPolicy::FailIfExists => ExistingDocumentPolicy::FailIfExists,
		proto::ExistingDocumentPolicy::ReplaceExisting => ExistingDocumentPolicy::ReplaceExisting,
	};
	let format = parse_format_policy(create.format_policy).map_err(build_from_failure)?;
	Ok(CreateMutation::new(create.content, existing, format))
}

fn build_move_mutation(moved: proto::MoveMutation) -> Result<MoveMutation, TransactionBuildError> {
	let base = moved
		.base_revision
		.ok_or_else(|| build_invalid("move base revision is required"))
		.and_then(|revision| parse_revision(revision).map_err(build_from_failure))?;
	let destination = parse_file_uri(&moved.destination_uri).map_err(build_from_failure)?;
	let precondition = match moved
		.destination_precondition
		.ok_or_else(|| build_invalid("move destination precondition is required"))?
	{
		proto::move_mutation::DestinationPrecondition::DestinationRevision(revision) => {
			MoveDestinationPrecondition::Revision(
				parse_revision(revision).map_err(build_from_failure)?,
			)
		},
		proto::move_mutation::DestinationPrecondition::DestinationMustNotExist(true) => {
			MoveDestinationPrecondition::MustNotExist
		},
		proto::move_mutation::DestinationPrecondition::DestinationMustNotExist(false) => {
			return Err(build_invalid("destination_must_not_exist must be true"));
		},
	};
	Ok(MoveMutation::new(base, destination, precondition))
}

async fn get_lsp_bindings(
	session: &EnvironmentSession,
	request: proto::GetLspBindingsRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::GetLspBindingsResponse> {
	let target = parse_target(required(request.document, "LSP binding document target")?)?;
	let lease_id = connection_lease_for_target(session, &target, &cancellation).await?;
	let bindings = tokio::select! {
		biased;
		() = cancellation.cancelled() => {
			return Err(Failure::cancelled("LSP binding request cancelled"));
		},
		result = session.environment().lsp().lease_bindings(lease_id) => {
			result.map_err(Failure::from_registry)?
		},
	};
	Ok(proto::GetLspBindingsResponse { bindings: bindings.iter().map(binding_to_proto).collect() })
}

async fn lsp_request(
	session: &EnvironmentSession,
	request: proto::LspRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::LspResponse> {
	let binding_id = parse_binding_id(&request.server_id)?;
	if request.method.is_empty() {
		return Err(Failure::invalid("LSP request method must not be empty"));
	}
	let stale = match proto::LspStalePolicy::try_from(request.stale_policy)
		.map_err(|_| Failure::invalid("unknown LSP stale policy"))?
	{
		proto::LspStalePolicy::Fail => StaleResponsePolicy::ContentModified,
		proto::LspStalePolicy::RetryCurrentHead => StaleResponsePolicy::RetryOnce,
	};
	let is_document_method = request.method.starts_with("textDocument/");
	let result = match (request.document, request.revision) {
		(None, None) if !is_document_method => {
			session
				.environment()
				.lsp()
				.workspace_request(binding_id, &request.method, request.params_json, cancellation)
				.await
		},
		(Some(target), Some(revision)) => {
			let target = parse_target(target)?;
			let lease_id = connection_lease_for_target(session, &target, &cancellation).await?;
			let revision = parse_revision(revision)?;
			if is_document_method {
				validate_text_document_uri(session, lease_id, &request.params_json, &cancellation)
					.await?;
			}
			session
				.environment()
				.lsp()
				.semantic_request(
					binding_id,
					&request.method,
					request.params_json,
					lease_id,
					revision,
					stale,
					cancellation,
				)
				.await
		},
		(None, None) => {
			return Err(Failure::invalid(
				"textDocument requests require document and revision context",
			));
		},
		_ => return Err(Failure::invalid("LSP document and revision must be supplied together")),
	};
	match result {
		Ok(response) => {
			let outcome = match response.outcome {
				LspResponseOutcome::Result(result) => proto::lsp_response::Outcome::ResultJson(result),
				LspResponseOutcome::Error { code, message, data } => {
					proto::lsp_response::Outcome::Error(proto::LspError {
						code,
						message: message.to_string(),
						data_json: data.unwrap_or_default(),
					})
				},
			};
			Ok(proto::LspResponse {
				server_id: binding_id_bytes(binding_id),
				revision:  response.revision.map(revision_to_proto),
				outcome:   Some(outcome),
			})
		},
		Err(error) => Err(Failure::from_registry(error)),
	}
}

async fn lsp_notification(
	session: &EnvironmentSession,
	request: proto::LspNotificationRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::LspNotificationResponse> {
	let binding_id = parse_binding_id(&request.server_id)?;
	if request.method.is_empty() {
		return Err(Failure::invalid("LSP notification method must not be empty"));
	}
	session
		.environment()
		.lsp()
		.notification(binding_id, &request.method, request.params_json, cancellation)
		.await
		.map_err(Failure::from_registry)?;
	Ok(proto::LspNotificationResponse {})
}

fn canonicalize_path(
	session: &EnvironmentSession,
	request: proto::CanonicalizePathRequest,
) -> DispatchResult<proto::CanonicalizePathResponse> {
	let uri = parse_file_uri(&request.uri)?;
	let canonical = session
		.environment()
		.paths()
		.canonicalize(&uri)
		.map_err(Failure::from_core)?;
	Ok(proto::CanonicalizePathResponse { canonical_uri: canonical.to_string() })
}

fn stat_path(
	session: &EnvironmentSession,
	request: proto::StatPathRequest,
) -> DispatchResult<proto::StatPathResponse> {
	let uri = parse_file_uri(&request.uri)?;
	let follow = parse_follow(request.follow_symlinks)?;
	let metadata = session
		.environment()
		.paths()
		.stat(&uri, follow)
		.map_err(Failure::from_core)?;
	Ok(proto::StatPathResponse { metadata: Some(metadata_to_proto(session, &metadata)?) })
}

fn list_directory(
	session: &EnvironmentSession,
	request: proto::ListDirectoryRequest,
) -> DispatchResult<proto::ListDirectoryResponse> {
	let uri = parse_file_uri(&request.uri)?;
	let follow = parse_follow(request.follow_symlinks)?;
	let entries = session
		.environment()
		.paths()
		.list_directory(&uri, follow)
		.map_err(Failure::from_core)?;
	Ok(proto::ListDirectoryResponse {
		entries: entries
			.iter()
			.map(|entry| {
				Ok(proto::DirectoryEntry {
					name:     entry.name.to_string(),
					metadata: Some(metadata_to_proto(session, &entry.metadata)?),
				})
			})
			.collect::<DispatchResult<_>>()?,
	})
}

async fn create_directory(
	session: &EnvironmentSession,
	request: proto::CreateDirectoryRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::CreateDirectoryResponse> {
	let uri = parse_file_uri(&request.uri)?;
	let existing = match proto::ExistingDirectoryPolicy::try_from(request.existing_leaf)
		.map_err(|_| Failure::invalid("unknown existing directory policy"))?
	{
		proto::ExistingDirectoryPolicy::FailIfExists => crate::ExistingDirectoryPolicy::FailIfExists,
		proto::ExistingDirectoryPolicy::AllowExistingDirectory => {
			crate::ExistingDirectoryPolicy::AllowExistingDirectory
		},
	};
	let metadata = completed_path_result(
		session
			.environment()
			.paths()
			.create_directory(&uri, request.recursive, existing, &cancellation)
			.await
			.map_err(Failure::from_core)?,
	)?;
	Ok(proto::CreateDirectoryResponse { metadata: Some(metadata_to_proto(session, &metadata)?) })
}

async fn remove_path(
	session: &EnvironmentSession,
	request: proto::RemovePathRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::RemovePathResponse> {
	let uri = parse_file_uri(&request.uri)?;
	let revision = request.revision.map(parse_revision).transpose()?;
	completed_path_result(
		session
			.environment()
			.paths()
			.remove(&uri, request.recursive, revision, &cancellation)
			.await
			.map_err(Failure::from_core)?,
	)?;
	Ok(proto::RemovePathResponse {})
}

async fn rename_path(
	session: &EnvironmentSession,
	request: proto::RenamePathRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::RenamePathResponse> {
	let source = parse_file_uri(&request.source_uri)?;
	let destination = parse_file_uri(&request.destination_uri)?;
	let overwrite = parse_overwrite(request.overwrite, true)?;
	let source_revision = request.source_revision.map(parse_revision).transpose()?;
	let destination_revision = request
		.destination_revision
		.map(parse_revision)
		.transpose()?;
	let metadata = completed_path_result(
		session
			.environment()
			.paths()
			.rename(
				&source,
				&destination,
				overwrite,
				source_revision,
				destination_revision,
				&cancellation,
			)
			.await
			.map_err(Failure::from_core)?,
	)?;
	Ok(proto::RenamePathResponse { metadata: Some(metadata_to_proto(session, &metadata)?) })
}

async fn copy_path(
	session: &EnvironmentSession,
	request: proto::CopyPathRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::CopyPathResponse> {
	let source = parse_file_uri(&request.source_uri)?;
	let destination = parse_file_uri(&request.destination_uri)?;
	let follow = parse_follow(request.follow_source_symlinks)?;
	let overwrite = parse_overwrite(request.overwrite, false)?;
	let revision = request
		.destination_revision
		.map(parse_revision)
		.transpose()?;
	let copied = completed_path_result(
		session
			.environment()
			.paths()
			.copy(&source, &destination, follow, overwrite, revision, &cancellation)
			.await
			.map_err(Failure::from_core)?,
	)?;
	Ok(proto::CopyPathResponse {
		metadata:     Some(metadata_to_proto(session, &copied.metadata)?),
		bytes_copied: copied.bytes_copied,
	})
}

fn read_link(
	session: &EnvironmentSession,
	request: proto::ReadLinkRequest,
) -> DispatchResult<proto::ReadLinkResponse> {
	let uri = parse_file_uri(&request.uri)?;
	let target = session
		.environment()
		.paths()
		.read_link(&uri)
		.map_err(Failure::from_core)?;
	Ok(proto::ReadLinkResponse { target: Some(symlink_target_to_proto(session, &target)?) })
}

async fn create_symlink(
	session: &EnvironmentSession,
	request: proto::CreateSymlinkRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::CreateSymlinkResponse> {
	let target = parse_symlink_target(session, required(request.target, "symlink target")?)?;
	let link = parse_file_uri(&request.link_uri)?;
	let kind = match proto::SymlinkTargetKind::try_from(request.target_kind)
		.map_err(|_| Failure::invalid("unknown symlink target kind"))?
	{
		proto::SymlinkTargetKind::Unspecified => {
			return Err(Failure::invalid("symlink target kind is required"));
		},
		proto::SymlinkTargetKind::File => SymlinkTargetKind::File,
		proto::SymlinkTargetKind::Directory => SymlinkTargetKind::Directory,
	};
	let overwrite = parse_overwrite(request.overwrite, false)?;
	let metadata = completed_path_result(
		session
			.environment()
			.paths()
			.create_symlink(&target, &link, kind, overwrite, &cancellation)
			.await
			.map_err(Failure::from_core)?,
	)?;
	Ok(proto::CreateSymlinkResponse { metadata: Some(metadata_to_proto(session, &metadata)?) })
}

async fn create_hard_link(
	session: &EnvironmentSession,
	request: proto::CreateHardLinkRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::CreateHardLinkResponse> {
	let source = parse_file_uri(&request.source_uri)?;
	let link = parse_file_uri(&request.link_uri)?;
	let follow = parse_follow(request.follow_source_symlinks)?;
	let overwrite = parse_overwrite(request.overwrite, false)?;
	let metadata = completed_path_result(
		session
			.environment()
			.paths()
			.create_hard_link(&source, &link, follow, overwrite, &cancellation)
			.await
			.map_err(Failure::from_core)?,
	)?;
	Ok(proto::CreateHardLinkResponse { metadata: Some(metadata_to_proto(session, &metadata)?) })
}

async fn set_permissions(
	session: &EnvironmentSession,
	request: proto::SetPermissionsRequest,
	cancellation: CancellationToken,
) -> DispatchResult<proto::SetPermissionsResponse> {
	let uri = parse_file_uri(&request.uri)?;
	let permissions = required(request.permissions, "portable permissions")?;
	if permissions.read_only.is_none() && permissions.executable.is_none() {
		return Err(Failure::invalid("at least one portable permission is required"));
	}
	let follow = parse_follow(request.follow_symlinks)?;
	let revision = request.revision.map(parse_revision).transpose()?;
	let metadata = completed_path_result(
		session
			.environment()
			.paths()
			.set_permissions(
				&uri,
				PortablePermissions {
					read_only:  permissions.read_only,
					executable: permissions.executable,
				},
				follow,
				revision,
				&cancellation,
			)
			.await
			.map_err(Failure::from_core)?,
	)?;
	Ok(proto::SetPermissionsResponse { metadata: Some(metadata_to_proto(session, &metadata)?) })
}

fn completed_path_result<T>(result: PathMutationResult<T>) -> DispatchResult<T> {
	match result {
		PathMutationResult::Completed(value) => Ok(value),
		PathMutationResult::TransactionRejected(outcome) => Err(path_rejection(outcome.as_ref())),
	}
}

fn path_rejection(outcome: &TransactionOutcome) -> Failure {
	match outcome {
		TransactionOutcome::Rejected { reason, message, .. } => Failure::new(
			transaction_reject_code(*reason),
			format!("path transaction rejected ({reason:?}): {message}"),
		),
		TransactionOutcome::PartiallyCommitted {
			failed_operation_index, reason, message, ..
		} => Failure::internal(format!(
			"path transaction partially committed before operation {failed_operation_index} \
			 ({reason:?}): {message}"
		)),
		TransactionOutcome::Committed { .. } => {
			Failure::internal("path transaction did not return its expected operation result")
		},
	}
}

const fn transaction_reject_code(reason: TransactionRejectReason) -> proto::ProtocolErrorCode {
	match reason {
		TransactionRejectReason::StaleBase
		| TransactionRejectReason::OverlappingChange
		| TransactionRejectReason::ExternalModification => proto::ProtocolErrorCode::ContentModified,
		TransactionRejectReason::RevisionExpired => proto::ProtocolErrorCode::RevisionExpired,
		TransactionRejectReason::InvalidContent => proto::ProtocolErrorCode::InvalidArgument,
		TransactionRejectReason::FormatFailed => proto::ProtocolErrorCode::Internal,
		TransactionRejectReason::PersistFailed => proto::ProtocolErrorCode::Io,
		TransactionRejectReason::PreconditionFailed => proto::ProtocolErrorCode::PreconditionFailed,
		TransactionRejectReason::Cancelled => proto::ProtocolErrorCode::Cancelled,
	}
}

fn parse_target(target: proto::DocumentTarget) -> DispatchResult<DocumentTarget> {
	match required(target.target, "document target")? {
		proto::document_target::Target::DocumentId(bytes) => {
			Ok(DocumentTarget::Document(parse_document_id(&bytes)?))
		},
		proto::document_target::Target::LeaseId(bytes) => {
			Ok(DocumentTarget::Lease(parse_lease_id(&bytes)?))
		},
		proto::document_target::Target::Uri(uri) => Ok(DocumentTarget::Uri(parse_file_uri(&uri)?)),
	}
}

fn locator_for_target(
	session: &EnvironmentSession,
	target: &DocumentTarget,
) -> DispatchResult<DocumentLocator> {
	match target {
		DocumentTarget::Document(id) => Ok(DocumentLocator::Document(*id)),
		DocumentTarget::Lease(id) if session.owns_lease(*id) => Ok(DocumentLocator::Lease(*id)),
		DocumentTarget::Lease(_) => {
			Err(Failure::not_found("document lease is not owned by this connection"))
		},
		DocumentTarget::Uri(uri) => session
			.environment()
			.store()
			.resolve_entry_path(uri)
			.map(DocumentLocator::Path)
			.map_err(Failure::from_core),
	}
}

async fn canonical_path_for_locator(
	session: &EnvironmentSession,
	locator: DocumentLocator,
	cancellation: &CancellationToken,
) -> DispatchResult<PathBuf> {
	let handle = session
		.environment()
		.store()
		.actor_handle(locator)
		.map_err(Failure::from_core)?;
	tokio::select! {
		biased;
		() = cancellation.cancelled() => Err(Failure::cancelled("request cancelled")),
		state = handle.state() => Ok(state.map_err(Failure::from_core)?.path),
	}
}

async fn validate_text_document_uri(
	session: &EnvironmentSession,
	lease_id: LeaseId,
	params_json: &Bytes,
	cancellation: &CancellationToken,
) -> DispatchResult<()> {
	let value: serde_json::Value = serde_json::from_slice(params_json)
		.map_err(|error| Failure::invalid(format!("invalid LSP params JSON: {error}")))?;
	let supplied = value
		.pointer("/textDocument/uri")
		.and_then(serde_json::Value::as_str)
		.ok_or_else(|| Failure::invalid("textDocument.uri is required for textDocument requests"))?;
	let path =
		canonical_path_for_locator(session, DocumentLocator::Lease(lease_id), cancellation).await?;
	let canonical = session
		.environment()
		.store()
		.file_uri(&path)
		.map_err(Failure::from_core)?;
	if supplied != canonical.as_str() {
		return Err(Failure::precondition(
			"textDocument.uri does not match the synchronized document lease URI",
		));
	}
	Ok(())
}

fn inbound_event_is_document_scoped(method: &str, params_json: &Bytes) -> bool {
	if method.starts_with("textDocument/") {
		return true;
	}
	serde_json::from_slice::<serde_json::Value>(params_json).is_ok_and(|value| {
		value
			.pointer("/textDocument/uri")
			.and_then(serde_json::Value::as_str)
			.is_some()
			|| value
				.get("uri")
				.and_then(serde_json::Value::as_str)
				.is_some()
	})
}
fn inbound_event_is_resolved(
	method: &str,
	params_json: &Bytes,
	has_document: bool,
	has_revision: bool,
) -> bool {
	!inbound_event_is_document_scoped(method, params_json) || (has_document && has_revision)
}

async fn connection_lease_for_target(
	session: &EnvironmentSession,
	target: &DocumentTarget,
	cancellation: &CancellationToken,
) -> DispatchResult<LeaseId> {
	let locator = locator_for_target(session, target)?;
	let handle = session
		.environment()
		.store()
		.actor_handle(locator)
		.map_err(Failure::from_core)?;
	let state = tokio::select! {
		biased;
		() = cancellation.cancelled() => {
			return Err(Failure::cancelled("document lease lookup cancelled"));
		},
		state = handle.state() => state.map_err(Failure::from_core)?,
	};
	session
		.lease_for_document(state.document_id)
		.ok_or_else(|| Failure::precondition("document has no lease owned by this connection"))
}

fn parse_read_selection(selection: proto::ReadSelection) -> DispatchResult<ReadSelection> {
	match required(selection.selection, "read selection kind")? {
		proto::read_selection::Selection::Whole(_) => Ok(ReadSelection::Whole),
		proto::read_selection::Selection::Bytes(bytes) => Ok(ReadSelection::Bytes(
			bytes
				.ranges
				.into_iter()
				.map(|range| ByteRange::new(range.start, range.end).map_err(Failure::from_core))
				.collect::<DispatchResult<_>>()?,
		)),
		proto::read_selection::Selection::Lines(lines) => Ok(ReadSelection::Lines(
			lines
				.ranges
				.into_iter()
				.map(|range| LineRange::new(range.start, range.end).map_err(Failure::from_core))
				.collect::<DispatchResult<_>>()?,
		)),
	}
}

fn parse_summary_options(options: proto::CodeSummaryOptions) -> DispatchResult<SummaryOptions> {
	let render_mode = match proto::SummaryRenderMode::try_from(options.render_mode)
		.map_err(|_| Failure::invalid("unknown summary render mode"))?
	{
		proto::SummaryRenderMode::Unspecified => {
			return Err(Failure::invalid("summary render mode is required"));
		},
		proto::SummaryRenderMode::Hashline => SummaryRenderMode::Hashline,
		proto::SummaryRenderMode::Numbered => SummaryRenderMode::Numbered,
		proto::SummaryRenderMode::Plain => SummaryRenderMode::Plain,
	};
	Ok(SummaryOptions {
		min_total_lines: options.min_total_lines,
		min_body_lines: options.min_body_lines,
		min_comment_lines: options.min_comment_lines,
		unfold_until_lines: options.unfold_until_lines,
		unfold_limit_lines: options.unfold_limit_lines,
		enable_prose: options.enable_prose,
		language: (!options.language.is_empty()).then(|| Str::new(options.language)),
		render_mode,
	})
}

fn summary_to_proto(summary: &DocumentSummary) -> proto::DocumentSummaryResult {
	proto::DocumentSummaryResult {
		language:    summary.language.to_string(),
		parsed:      true,
		elided:      true,
		total_lines: summary.total_lines,
		segments:    summary
			.segments
			.iter()
			.map(|segment| match segment {
				SummarySegment::Kept { start_line, end_line, text } => proto::DocumentSummarySegment {
					kind:       proto::document_summary_segment::Kind::Kept as i32,
					start_line: *start_line,
					end_line:   *end_line,
					text:       Some(text.clone()),
				},
				SummarySegment::Elided { start_line, end_line } => proto::DocumentSummarySegment {
					kind:       proto::document_summary_segment::Kind::Elided as i32,
					start_line: *start_line,
					end_line:   *end_line,
					text:       None,
				},
			})
			.collect(),
		rendered:    Some(proto::RenderedDocumentSummary {
			text:          summary.rendered.text.clone(),
			display_text:  summary.rendered.display_text.clone(),
			elided_ranges: summary
				.rendered
				.elided_ranges
				.iter()
				.map(|range| proto::SummaryLineRange {
					start_line: range.start_line,
					end_line:   range.end_line,
				})
				.collect(),
			elided_lines:  summary.rendered.elided_lines,
		}),
	}
}

fn fallback_to_proto(fallback: &SummaryFallback) -> proto::DocumentSummaryUnavailable {
	proto::DocumentSummaryUnavailable {
		reason:      match fallback.reason {
			SummaryUnavailableReason::Binary => proto::SummaryUnavailableReason::Binary,
			SummaryUnavailableReason::MissingDocument => {
				proto::SummaryUnavailableReason::MissingDocument
			},
			SummaryUnavailableReason::TooLarge => proto::SummaryUnavailableReason::TooLarge,
			SummaryUnavailableReason::TooManyLines => proto::SummaryUnavailableReason::TooManyLines,
			SummaryUnavailableReason::BelowMinimumLines => {
				proto::SummaryUnavailableReason::BelowMinimumLines
			},
			SummaryUnavailableReason::ProseDisabled => proto::SummaryUnavailableReason::ProseDisabled,
			SummaryUnavailableReason::UnsupportedLanguage => {
				proto::SummaryUnavailableReason::UnsupportedLanguage
			},
			SummaryUnavailableReason::Empty => proto::SummaryUnavailableReason::Empty,
			SummaryUnavailableReason::SyntaxError => proto::SummaryUnavailableReason::SyntaxError,
			SummaryUnavailableReason::NoElisions => proto::SummaryUnavailableReason::NoElisions,
			SummaryUnavailableReason::ParserFailure => proto::SummaryUnavailableReason::ParserFailure,
		} as i32,
		total_lines: fallback.total_lines,
		language:    fallback
			.language
			.as_ref()
			.map_or_else(String::new, ToString::to_string),
		parsed:      fallback.parsed,
	}
}

fn transaction_outcome_to_proto(outcome: &TransactionOutcome) -> proto::CommitTransactionResponse {
	let outcome = match outcome {
		TransactionOutcome::Committed { transaction_id, operations } => {
			proto::commit_transaction_response::Outcome::Committed(proto::TransactionCommitted {
				transaction_id: Bytes::copy_from_slice(transaction_id.as_bytes()),
				operations:     operation_results_to_proto(operations),
			})
		},
		TransactionOutcome::Rejected { transaction_id, reason, message, conflicts } => {
			let converted = conflicts
				.iter()
				.map(|conflict| proto::DocumentConflict {
					operation_index:    conflict.operation_index(),
					expected:           Some(revision_to_proto(conflict.expected())),
					current:            Some(head_at_uri_to_proto(conflict.current(), conflict.uri())),
					conflicting_ranges: conflict
						.conflicting_ranges()
						.iter()
						.copied()
						.map(range_to_proto)
						.collect(),
				})
				.collect();
			proto::commit_transaction_response::Outcome::Rejected(proto::TransactionRejected {
				transaction_id: Bytes::copy_from_slice(transaction_id.as_bytes()),
				reason:         reject_reason_to_proto(*reason) as i32,
				message:        message.to_string(),
				conflicts:      converted,
			})
		},
		TransactionOutcome::PartiallyCommitted {
			transaction_id,
			committed_operations,
			failed_operation_index,
			reason,
			message,
		} => proto::commit_transaction_response::Outcome::PartiallyCommitted(
			proto::TransactionPartiallyCommitted {
				transaction_id:         Bytes::copy_from_slice(transaction_id.as_bytes()),
				committed_operations:   operation_results_to_proto(committed_operations),
				failed_operation_index: *failed_operation_index,
				reason:                 reject_reason_to_proto(*reason) as i32,
				message:                message.to_string(),
			},
		),
	};
	proto::CommitTransactionResponse { outcome: Some(outcome) }
}

fn operation_results_to_proto(operations: &[OperationResult]) -> Vec<proto::OperationResult> {
	operations
		.iter()
		.map(|operation| proto::OperationResult {
			operation_index: operation.operation_index(),
			head:            Some(head_at_uri_to_proto(operation.head(), operation.uri())),
			rebased:         operation.rebased(),
			formatted:       operation.formatted(),
			changed_ranges:  operation
				.changed_ranges()
				.iter()
				.copied()
				.map(range_to_proto)
				.collect(),
			previous_uri:    operation
				.previous_uri()
				.map_or_else(String::new, Url::to_string),
		})
		.collect()
}

const fn reject_reason_to_proto(reason: TransactionRejectReason) -> proto::TransactionRejectReason {
	match reason {
		TransactionRejectReason::StaleBase => proto::TransactionRejectReason::StaleBase,
		TransactionRejectReason::OverlappingChange => {
			proto::TransactionRejectReason::OverlappingChange
		},
		TransactionRejectReason::ExternalModification => {
			proto::TransactionRejectReason::ExternalModification
		},
		TransactionRejectReason::RevisionExpired => proto::TransactionRejectReason::RevisionExpired,
		TransactionRejectReason::InvalidContent => proto::TransactionRejectReason::InvalidContent,
		TransactionRejectReason::FormatFailed => proto::TransactionRejectReason::FormatFailed,
		TransactionRejectReason::PersistFailed => proto::TransactionRejectReason::PersistFailed,
		TransactionRejectReason::PreconditionFailed => {
			proto::TransactionRejectReason::PreconditionFailed
		},
		TransactionRejectReason::Cancelled => proto::TransactionRejectReason::Cancelled,
	}
}

async fn document_ref_to_proto(
	session: &EnvironmentSession,
	document_id: DocumentId,
) -> Option<proto::DocumentRef> {
	let state = session
		.environment()
		.store()
		.actor_handle(document_id)
		.ok()?
		.state()
		.await
		.ok()?;
	let uri = session.environment().store().file_uri(&state.path).ok()?;
	Some(proto::DocumentRef {
		id:  Bytes::copy_from_slice(document_id.as_bytes()),
		uri: uri.to_string(),
	})
}

async fn head_to_proto(
	session: &EnvironmentSession,
	head: &DocumentHead,
	cancellation: &CancellationToken,
) -> DispatchResult<proto::DocumentHead> {
	let handle = session
		.environment()
		.store()
		.actor_handle(head.document_id())
		.map_err(Failure::from_core)?;
	let state = tokio::select! {
		biased;
		() = cancellation.cancelled() => return Err(Failure::cancelled("request cancelled")),
		state = handle.state() => state.map_err(Failure::from_core)?,
	};
	let uri = session
		.environment()
		.store()
		.file_uri(&state.path)
		.map_err(Failure::from_core)?;
	Ok(head_at_uri_to_proto(head, &uri))
}

fn head_at_uri_to_proto(head: &DocumentHead, uri: &Url) -> proto::DocumentHead {
	let (kind, language_id) = match head.kind() {
		DocumentKind::Text(language) => (
			proto::DocumentKind::Text,
			language
				.as_ref()
				.map_or_else(String::new, |language| language.as_str().to_owned()),
		),
		DocumentKind::Binary => (proto::DocumentKind::Binary, String::new()),
	};
	proto::DocumentHead {
		document: Some(proto::DocumentRef {
			id:  Bytes::copy_from_slice(head.document_id().as_bytes()),
			uri: uri.to_string(),
		}),
		revision: Some(revision_to_proto(head.revision())),
		presence: match head.presence() {
			DocumentPresence::Present => proto::DocumentPresence::Present,
			DocumentPresence::Missing => proto::DocumentPresence::Missing,
		} as i32,
		kind: kind as i32,
		byte_length: head.byte_length(),
		language_id,
	}
}

fn document_event_to_proto(
	session: &EnvironmentSession,
	event: &crate::DocumentEvent,
) -> DispatchResult<proto::DocumentEvent> {
	let previous_uri = match event.previous_path() {
		Some(path) => session
			.environment()
			.store()
			.file_uri(path)
			.map_err(Failure::from_core)?
			.to_string(),
		None => String::new(),
	};
	let uri = session
		.environment()
		.store()
		.file_uri(event.path())
		.map_err(Failure::from_core)?;
	Ok(proto::DocumentEvent {
		event_sequence: event.event_sequence(),
		kind: match event.kind() {
			crate::DocumentEventKind::Committed => proto::DocumentEventKind::Committed,
			crate::DocumentEventKind::ExternalCreated => proto::DocumentEventKind::ExternalCreated,
			crate::DocumentEventKind::ExternalModified => proto::DocumentEventKind::ExternalModified,
			crate::DocumentEventKind::ExternalDeleted => proto::DocumentEventKind::ExternalDeleted,
			crate::DocumentEventKind::ExternalRenamed => proto::DocumentEventKind::ExternalRenamed,
			crate::DocumentEventKind::WatchRescanned => proto::DocumentEventKind::WatchRescanned,
		} as i32,
		head: Some(head_at_uri_to_proto(event.head(), &uri)),
		previous_revision: Some(revision_to_proto(event.previous_revision())),
		transaction_id: event
			.transaction_id()
			.map_or_else(Bytes::new, |id| Bytes::copy_from_slice(id.as_bytes())),
		invalidated_transaction_ids: event
			.invalidated_transaction_ids()
			.iter()
			.map(|id| Bytes::copy_from_slice(id.as_bytes()))
			.collect(),
		previous_uri,
	})
}

fn binding_to_proto(binding: &crate::lsp_registry::LspLeaseBinding) -> proto::LspServerBinding {
	let policy = binding.sync_policy();
	proto::LspServerBinding {
		server_id:         binding_id_bytes(binding.info().id()),
		name:              binding.info().spec().name().to_owned(),
		sync_policy:       Some(proto::SyncPolicy {
			change:               match policy.change {
				TextDocumentSyncKind::None => proto::TextDocumentSyncKind::TextDocumentSyncNone,
				TextDocumentSyncKind::Full => proto::TextDocumentSyncKind::TextDocumentSyncFull,
				TextDocumentSyncKind::Incremental => {
					proto::TextDocumentSyncKind::TextDocumentSyncIncremental
				},
			} as i32,
			open_close:           policy.open_close,
			will_save:            policy.will_save,
			will_save_wait_until: policy.will_save_wait_until,
			save:                 policy.save,
			save_include_text:    policy.save_include_text,
			position_encoding:    policy.position_encoding.as_lsp_name().to_owned(),
		}),
		capabilities_json: binding.capabilities_json().clone(),
	}
}

fn metadata_to_proto(
	session: &EnvironmentSession,
	metadata: &PathMetadata,
) -> DispatchResult<proto::PathMetadata> {
	let uri = session
		.environment()
		.store()
		.file_uri(&metadata.path)
		.map_err(Failure::from_core)?;
	Ok(proto::PathMetadata {
		uri: uri.to_string(),
		kind: match metadata.kind {
			FileKind::RegularFile => proto::FileKind::RegularFile,
			FileKind::Directory => proto::FileKind::Directory,
			FileKind::SymbolicLink => proto::FileKind::SymbolicLink,
			FileKind::Other => proto::FileKind::Other,
		} as i32,
		byte_length: metadata.byte_length,
		permissions: Some(proto::PortablePermissions {
			read_only:  metadata.permissions.read_only,
			executable: metadata.permissions.executable,
		}),
		modified_time_unix_nanos: metadata.modified.map(system_time_to_nanos).transpose()?,
		accessed_time_unix_nanos: metadata.accessed.map(system_time_to_nanos).transpose()?,
		created_time_unix_nanos: metadata.created.map(system_time_to_nanos).transpose()?,
	})
}

fn parse_symlink_target(
	session: &EnvironmentSession,
	target: proto::SymlinkTarget,
) -> DispatchResult<SymlinkTarget> {
	let uri = parse_file_uri(&target.uri)?;
	let path = session
		.environment()
		.store()
		.resolve_entry_path(&uri)
		.map_err(Failure::from_core)?;
	let form = match proto::SymlinkTargetForm::try_from(target.form)
		.map_err(|_| Failure::invalid("unknown symlink target form"))?
	{
		proto::SymlinkTargetForm::Absolute => SymlinkTargetForm::Absolute,
		proto::SymlinkTargetForm::Relative => SymlinkTargetForm::Relative,
	};
	Ok(SymlinkTarget { path, form })
}

fn symlink_target_to_proto(
	session: &EnvironmentSession,
	target: &SymlinkTarget,
) -> DispatchResult<proto::SymlinkTarget> {
	let uri = session
		.environment()
		.store()
		.file_uri(&target.path)
		.map_err(Failure::from_core)?;
	Ok(proto::SymlinkTarget {
		uri:  uri.to_string(),
		form: match target.form {
			SymlinkTargetForm::Absolute => proto::SymlinkTargetForm::Absolute,
			SymlinkTargetForm::Relative => proto::SymlinkTargetForm::Relative,
		} as i32,
	})
}

fn parse_follow(value: i32) -> DispatchResult<FollowSymlinks> {
	match proto::FollowSymlinks::try_from(value)
		.map_err(|_| Failure::invalid("unknown follow-symlinks policy"))?
	{
		proto::FollowSymlinks::No => Ok(FollowSymlinks::No),
		proto::FollowSymlinks::Yes => Ok(FollowSymlinks::Yes),
	}
}

fn parse_overwrite(
	value: i32,
	allow_empty_directory: bool,
) -> DispatchResult<crate::DestinationOverwritePolicy> {
	match proto::DestinationOverwritePolicy::try_from(value)
		.map_err(|_| Failure::invalid("unknown destination overwrite policy"))?
	{
		proto::DestinationOverwritePolicy::FailIfExists => {
			Ok(crate::DestinationOverwritePolicy::FailIfExists)
		},
		proto::DestinationOverwritePolicy::ReplaceNonDirectory => {
			Ok(crate::DestinationOverwritePolicy::ReplaceNonDirectory)
		},
		proto::DestinationOverwritePolicy::ReplaceEmptyDirectory if allow_empty_directory => {
			Ok(crate::DestinationOverwritePolicy::ReplaceEmptyDirectory)
		},
		proto::DestinationOverwritePolicy::ReplaceEmptyDirectory => {
			Err(Failure::invalid("replace-empty-directory is valid only for rename"))
		},
	}
}

fn parse_stale_policy(value: i32) -> DispatchResult<StalePolicy> {
	match proto::StalePolicy::try_from(value)
		.map_err(|_| Failure::invalid("unknown stale policy"))?
	{
		proto::StalePolicy::Fail => Ok(StalePolicy::Fail),
		proto::StalePolicy::RebaseNonOverlapping => Ok(StalePolicy::RebaseNonOverlapping),
		proto::StalePolicy::ForceReplace => Ok(StalePolicy::ForceReplace),
	}
}

fn parse_format_policy(value: i32) -> DispatchResult<FormatPolicy> {
	match proto::FormatPolicy::try_from(value)
		.map_err(|_| Failure::invalid("unknown format policy"))?
	{
		proto::FormatPolicy::Disabled => Ok(FormatPolicy::Disabled),
		proto::FormatPolicy::BestEffort => Ok(FormatPolicy::BestEffort),
		proto::FormatPolicy::Required => Ok(FormatPolicy::Required),
	}
}

fn parse_revision(revision: proto::Revision) -> DispatchResult<Revision> {
	let hash = exact_array::<32>(&revision.content_hash, "revision content hash")?;
	Ok(Revision::from_hash(revision.sequence, hash))
}

fn revision_to_proto(revision: Revision) -> proto::Revision {
	proto::Revision {
		sequence:     revision.sequence(),
		content_hash: Bytes::copy_from_slice(revision.content_hash()),
	}
}

const fn range_to_proto(range: ByteRange) -> proto::ByteRange {
	proto::ByteRange { start: range.start(), end: range.end() }
}

fn parse_document_id(bytes: &[u8]) -> DispatchResult<DocumentId> {
	Ok(DocumentId::from_bytes(exact_array(bytes, "document id")?))
}

fn parse_lease_id(bytes: &[u8]) -> DispatchResult<LeaseId> {
	Ok(LeaseId::from_bytes(exact_array(bytes, "lease id")?))
}

fn parse_transaction_id(bytes: &[u8]) -> DispatchResult<TransactionId> {
	Ok(TransactionId::from_bytes(exact_array(bytes, "transaction id")?))
}

fn parse_binding_id(bytes: &[u8]) -> DispatchResult<LspBindingId> {
	Ok(LspBindingId::from_u64(u64::from_be_bytes(exact_array(bytes, "LSP server id")?)))
}

fn binding_id_bytes(binding_id: LspBindingId) -> Bytes {
	Bytes::copy_from_slice(&binding_id.get().to_be_bytes())
}

fn exact_array<const N: usize>(bytes: &[u8], name: &str) -> DispatchResult<[u8; N]> {
	bytes
		.try_into()
		.map_err(|_| Failure::invalid(format!("{name} must be exactly {N} bytes")))
}

fn parse_file_uri(value: &str) -> DispatchResult<Url> {
	if value.is_empty() {
		return Err(Failure::invalid("file URI must not be empty"));
	}
	let uri =
		Url::parse(value).map_err(|error| Failure::invalid(format!("invalid URI: {error}")))?;
	if uri.scheme() != "file" {
		return Err(Failure::invalid("URI scheme must be file"));
	}
	if uri.cannot_be_a_base() {
		return Err(Failure::invalid("file URI must be hierarchical"));
	}
	Ok(uri)
}

fn system_time_to_nanos(time: SystemTime) -> DispatchResult<i64> {
	let nanos = match time.duration_since(UNIX_EPOCH) {
		Ok(duration) => {
			i128::from(duration.as_secs()) * 1_000_000_000 + i128::from(duration.subsec_nanos())
		},
		Err(error) => {
			let duration = error.duration();
			-(i128::from(duration.as_secs()) * 1_000_000_000 + i128::from(duration.subsec_nanos()))
		},
	};
	i64::try_from(nanos)
		.map_err(|_| Failure::internal("filesystem timestamp exceeds protocol range"))
}

fn required<T>(value: Option<T>, name: &str) -> DispatchResult<T> {
	value.ok_or_else(|| Failure::invalid(format!("{name} is required")))
}

fn build_invalid(message: impl AsRef<str>) -> TransactionBuildError {
	TransactionBuildError::new(TransactionRejectReason::InvalidContent, message)
}

fn build_precondition(message: impl AsRef<str>) -> TransactionBuildError {
	TransactionBuildError::new(TransactionRejectReason::PreconditionFailed, message)
}

fn build_from_failure(error: Failure) -> TransactionBuildError {
	build_invalid(error.message)
}

fn build_cancelled(message: impl AsRef<str>) -> TransactionBuildError {
	TransactionBuildError::new(TransactionRejectReason::Cancelled, message)
}
fn build_snapshot_error(error: Error) -> TransactionBuildError {
	let reason = match &error {
		Error::RevisionExpired { .. } | Error::RevisionMissing { .. } => {
			TransactionRejectReason::RevisionExpired
		},
		Error::InvalidContent { .. } | Error::InvalidRange { .. } => {
			TransactionRejectReason::InvalidContent
		},
		Error::ContentModified { .. }
		| Error::StaleTransaction { .. }
		| Error::ConflictingTransaction { .. }
		| Error::ExternalInvalidation { .. }
		| Error::StaleDiskState { .. } => TransactionRejectReason::ExternalModification,
		_ => TransactionRejectReason::PreconditionFailed,
	};
	TransactionBuildError::new(reason, error.to_string())
}

type DispatchResult<T> = Result<T, Failure>;

#[derive(Debug)]
struct Failure {
	code:    proto::ProtocolErrorCode,
	message: String,
}

impl Failure {
	fn new(code: proto::ProtocolErrorCode, message: impl Into<String>) -> Self {
		Self { code, message: message.into() }
	}

	fn invalid(message: impl Into<String>) -> Self {
		Self::new(proto::ProtocolErrorCode::InvalidArgument, message)
	}

	fn not_found(message: impl Into<String>) -> Self {
		Self::new(proto::ProtocolErrorCode::NotFound, message)
	}

	fn precondition(message: impl Into<String>) -> Self {
		Self::new(proto::ProtocolErrorCode::PreconditionFailed, message)
	}

	fn cancelled(message: impl Into<String>) -> Self {
		Self::new(proto::ProtocolErrorCode::Cancelled, message)
	}

	fn internal(message: impl Into<String>) -> Self {
		Self::new(proto::ProtocolErrorCode::Internal, message)
	}

	fn into_proto(self) -> proto::ProtocolError {
		proto::ProtocolError { code: self.code as i32, message: self.message }
	}

	fn from_core(error: Error) -> Self {
		let code = match &error {
			Error::PreconditionFailed { .. } => proto::ProtocolErrorCode::PreconditionFailed,
			Error::ContentModified { .. } => proto::ProtocolErrorCode::ContentModified,
			Error::InvalidTarget { .. }
			| Error::InvalidRange { .. }
			| Error::InvalidContent { .. } => proto::ProtocolErrorCode::InvalidArgument,
			Error::DocumentNotFound { .. } | Error::LeaseExpired { .. } => {
				proto::ProtocolErrorCode::NotFound
			},
			Error::RevisionMissing { .. } | Error::RevisionExpired { .. } => {
				proto::ProtocolErrorCode::RevisionExpired
			},
			Error::StaleTransaction { .. }
			| Error::ConflictingTransaction { .. }
			| Error::ExternalInvalidation { .. }
			| Error::StaleDiskState { .. } => proto::ProtocolErrorCode::ContentModified,
			Error::Watch { .. } => proto::ProtocolErrorCode::Io,
			Error::Persistence { source, .. } | Error::Io { source, .. }
				if source.kind() == std::io::ErrorKind::Interrupted =>
			{
				proto::ProtocolErrorCode::Cancelled
			},
			Error::Persistence { source, .. } | Error::Io { source, .. } => io_code(source.kind()),
			Error::Protocol { .. } => proto::ProtocolErrorCode::Internal,
		};
		Self::new(code, error.to_string())
	}

	fn from_registry(error: LspRegistryError) -> Self {
		match error {
			LspRegistryError::Store(error) => Self::from_core(error),
			LspRegistryError::Lsp(error) => Self::from_lsp(&error),
			error => {
				let code = match &error {
					LspRegistryError::InvalidBindingName
					| LspRegistryError::DuplicateBinding { .. }
					| LspRegistryError::InvalidSelector { .. }
					| LspRegistryError::InvalidInboundJson { .. } => proto::ProtocolErrorCode::InvalidArgument,
					LspRegistryError::UnknownBinding { .. } | LspRegistryError::UnknownLease { .. } => {
						proto::ProtocolErrorCode::NotFound
					},
					LspRegistryError::BindingBusy { .. }
					| LspRegistryError::BindingNotSelected { .. }
					| LspRegistryError::DocumentNotActivated { .. }
					| LspRegistryError::FormattingUnavailable => proto::ProtocolErrorCode::PreconditionFailed,
					LspRegistryError::ContentModified { .. }
					| LspRegistryError::BindingRestarted { .. } => proto::ProtocolErrorCode::ContentModified,
					LspRegistryError::PathCannotBeUri { .. }
					| LspRegistryError::BindingIdOverflow
					| LspRegistryError::BindingGenerationOverflow { .. }
					| LspRegistryError::Store(_)
					| LspRegistryError::Lsp(_) => proto::ProtocolErrorCode::Internal,
				};
				Self::new(code, error.to_string())
			},
		}
	}

	fn from_lsp(error: &LspError) -> Self {
		let code = match error {
			LspError::Transport(LspTransportError::Cancelled) => proto::ProtocolErrorCode::Cancelled,
			LspError::Transport(LspTransportError::Closed { .. }) => proto::ProtocolErrorCode::Io,
			LspError::Transport(LspTransportError::JsonRpc { .. }) => {
				proto::ProtocolErrorCode::Internal
			},
			LspError::Transport(LspTransportError::InvalidResponse { .. })
			| LspError::InvalidCapabilities { .. }
			| LspError::InvalidRegistration { .. } => proto::ProtocolErrorCode::Internal,
			LspError::InvalidJson { .. } | LspError::Position(_) | LspError::InvalidUtf8 => {
				proto::ProtocolErrorCode::InvalidArgument
			},
			LspError::LifecyclePassthrough { .. } => proto::ProtocolErrorCode::InvalidArgument,
			LspError::CapabilityNotAdvertised { .. }
			| LspError::SynchronizationUnavailable { .. }
			| LspError::DocumentNotTracked { .. }
			| LspError::NonTextDocument { .. } => proto::ProtocolErrorCode::Unsupported,
			LspError::StateChanged { .. } | LspError::LanguageChanged { .. } => {
				proto::ProtocolErrorCode::ContentModified
			},
			LspError::LeaseOverflow { .. }
			| LspError::StateGenerationOverflow { .. }
			| LspError::VersionOverflow { .. } => proto::ProtocolErrorCode::Internal,
		};
		Self::new(code, error.to_string())
	}
}

const fn io_code(kind: std::io::ErrorKind) -> proto::ProtocolErrorCode {
	match kind {
		std::io::ErrorKind::NotFound => proto::ProtocolErrorCode::NotFound,
		std::io::ErrorKind::PermissionDenied => proto::ProtocolErrorCode::PermissionDenied,
		std::io::ErrorKind::AlreadyExists => proto::ProtocolErrorCode::AlreadyExists,
		std::io::ErrorKind::NotADirectory => proto::ProtocolErrorCode::NotADirectory,
		std::io::ErrorKind::IsADirectory => proto::ProtocolErrorCode::IsADirectory,
		std::io::ErrorKind::DirectoryNotEmpty => proto::ProtocolErrorCode::DirectoryNotEmpty,
		std::io::ErrorKind::CrossesDevices => proto::ProtocolErrorCode::CrossDevice,
		std::io::ErrorKind::Unsupported => proto::ProtocolErrorCode::Unsupported,
		std::io::ErrorKind::InvalidInput | std::io::ErrorKind::InvalidData => {
			proto::ProtocolErrorCode::InvalidArgument
		},
		_ => proto::ProtocolErrorCode::Io,
	}
}

#[cfg(test)]
mod tests {
	use std::time::Duration;

	use bytes::Bytes;
	use tempfile::TempDir;

	use super::*;
	use crate::{Environment, ServerConfig, fs::DiskExpectation};

	fn environment(root: &TempDir) -> Environment {
		Environment::new(ServerConfig::new(root.path()).expect("server config")).expect("environment")
	}

	fn create_file(environment: &Environment, name: &str, content: &'static [u8]) -> PathBuf {
		let path = environment.store().local_fs().root_path().join(name);
		let prepared = environment
			.store()
			.local_fs()
			.prepare_write(&path, Bytes::from_static(content), DiskExpectation::Missing)
			.expect("prepare file");
		environment
			.store()
			.local_fs()
			.commit_prepared(prepared)
			.expect("commit file");
		path
	}

	#[test]
	fn fixed_size_ids_reject_short_and_long_inputs() {
		assert_eq!(
			parse_document_id(&[0; 15]).unwrap_err().code,
			proto::ProtocolErrorCode::InvalidArgument
		);
		assert_eq!(
			parse_lease_id(&[0; 17]).unwrap_err().code,
			proto::ProtocolErrorCode::InvalidArgument
		);
		assert_eq!(
			parse_binding_id(&[0; 7]).unwrap_err().code,
			proto::ProtocolErrorCode::InvalidArgument
		);
	}

	#[test]
	fn revisions_require_an_exact_blake3_hash() {
		let malformed =
			proto::Revision { sequence: 9, content_hash: Bytes::from_static(&[1; 31]) };
		assert_eq!(
			parse_revision(malformed).unwrap_err().code,
			proto::ProtocolErrorCode::InvalidArgument
		);
	}

	#[test]
	fn binding_ids_are_exact_big_endian_values() {
		let encoded = Bytes::from_static(&[0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef]);
		let id = parse_binding_id(&encoded).unwrap();
		assert_eq!(id.get(), 0x0123_4567_89ab_cdef);
		assert_eq!(binding_id_bytes(id), encoded);
	}

	#[test]
	fn event_stream_failures_are_explicit_after_minor_one() {
		let lease_id = LeaseId::from_bytes([7; 16]);
		let frame =
			document_event_stream_error_frame(1, lease_id, DocumentEventStreamError::Lagged {
				skipped: 3,
			});
		let Some(proto::server_frame::Body::EventStreamError(error)) = frame.body else {
			panic!("minor one must use the dedicated event stream error");
		};
		assert_eq!(error.stream(), proto::EventStreamKind::Document);
		assert_eq!(error.failure(), proto::EventStreamFailure::Lagged);
		assert_eq!(error.lease_id.as_ref(), lease_id.as_bytes());
		assert_eq!(error.skipped_events, 3);
	}

	#[test]
	fn event_stream_failures_remain_decodable_by_minor_zero() {
		let frame = document_event_stream_error_frame(
			0,
			LeaseId::from_bytes([7; 16]),
			DocumentEventStreamError::Lagged { skipped: 3 },
		);
		assert!(matches!(
			frame.body,
			Some(proto::server_frame::Body::Error(proto::ProtocolError {
				code,
				..
			})) if code == proto::ProtocolErrorCode::ContentModified as i32
		));
	}

	#[test]
	fn timestamps_preserve_the_pre_epoch_nanosecond_boundary() {
		let instant = UNIX_EPOCH - Duration::from_nanos(1);
		assert_eq!(system_time_to_nanos(instant).unwrap(), -1);
	}

	#[tokio::test]
	async fn lease_targets_are_connection_owned_before_lookup_or_building() {
		let root = tempfile::tempdir().expect("temporary root");
		let environment = environment(&root);
		let owner = environment.session();
		let other = environment.session();
		let lease_id = LeaseId::from_bytes([9; 16]);
		let document_id = DocumentId::from_bytes([3; 16]);
		owner.own_lease(lease_id, document_id, CancellationToken::new(), CancellationToken::new());
		let target = DocumentTarget::Lease(lease_id);

		assert_eq!(
			locator_for_target(&other, &target).unwrap_err().code,
			proto::ProtocolErrorCode::NotFound
		);
		let operation = proto::DocumentMutation {
			document:  Some(proto::DocumentTarget {
				target: Some(proto::document_target::Target::LeaseId(Bytes::copy_from_slice(
					lease_id.as_bytes(),
				))),
			}),
			operation: Some(proto::document_mutation::Operation::Create(
				proto::CreateMutation::default(),
			)),
		};
		let error = build_operations(other, vec![operation], CancellationToken::new())
			.await
			.expect_err("foreign lease must reject during operation building");
		assert_eq!(error.reason(), TransactionRejectReason::PreconditionFailed);
	}

	#[tokio::test]
	async fn text_document_requests_require_context_and_exact_lease_uri() {
		let root = tempfile::tempdir().expect("temporary root");
		let environment = environment(&root);
		let session = environment.session();
		let path = create_file(&environment, "document.txt", b"text\n");
		let opened = environment
			.store()
			.open(path.clone())
			.await
			.expect("open document");
		let (lease_id, head, _) = opened.into_parts();
		session.own_lease(
			lease_id,
			head.document_id(),
			CancellationToken::new(),
			CancellationToken::new(),
		);

		let omitted = lsp_request(
			&session,
			proto::LspRequest {
				server_id: Bytes::copy_from_slice(&1_u64.to_be_bytes()),
				method: "textDocument/hover".to_owned(),
				params_json: Bytes::from_static(br#"{"textDocument":{"uri":"file:///ignored"}}"#),
				..proto::LspRequest::default()
			},
			CancellationToken::new(),
		)
		.await
		.expect_err("text document context is required");
		assert_eq!(omitted.code, proto::ProtocolErrorCode::InvalidArgument);

		let mismatch = validate_text_document_uri(
			&session,
			lease_id,
			&Bytes::from_static(br#"{"textDocument":{"uri":"file:///different.txt"}}"#),
			&CancellationToken::new(),
		)
		.await
		.expect_err("mismatched URI must reject");
		assert_eq!(mismatch.code, proto::ProtocolErrorCode::PreconditionFailed);

		let canonical = environment.store().file_uri(&path).expect("canonical URI");
		let params = Bytes::from(
			serde_json::to_vec(&serde_json::json!({
				"textDocument": { "uri": canonical.as_str() }
			}))
			.expect("params JSON"),
		);
		validate_text_document_uri(&session, lease_id, &params, &CancellationToken::new())
			.await
			.expect("canonical URI must match");
	}

	#[test]
	fn unresolved_uri_scoped_inbound_events_are_filtered() {
		let diagnostics = Bytes::from_static(br#"{"uri":"file:///document.txt","diagnostics":[]}"#);
		assert!(!inbound_event_is_resolved(
			"textDocument/publishDiagnostics",
			&diagnostics,
			false,
			false,
		));
		assert!(!inbound_event_is_resolved(
			"textDocument/publishDiagnostics",
			&diagnostics,
			true,
			false,
		));
		assert!(inbound_event_is_resolved(
			"textDocument/publishDiagnostics",
			&diagnostics,
			true,
			true,
		));
		assert!(inbound_event_is_resolved(
			"workspace/configuration",
			&Bytes::from_static(br"{}"),
			false,
			false,
		));
	}

	#[tokio::test]
	async fn failed_close_still_releases_session_ownership() {
		let root = tempfile::tempdir().expect("temporary root");
		let session = environment(&root).session();
		let lease_id = LeaseId::from_bytes([7; 16]);
		session.own_lease(
			lease_id,
			DocumentId::from_bytes([8; 16]),
			CancellationToken::new(),
			CancellationToken::new(),
		);

		close_document(
			&session,
			proto::CloseDocumentRequest { lease_id: Bytes::copy_from_slice(lease_id.as_bytes()) },
			CancellationToken::new(),
		)
		.await
		.expect_err("unknown registry lease must fail");

		assert!(!session.owns_lease(lease_id));
	}

	#[tokio::test]
	async fn cleanup_deadline_cancels_and_awaits_cooperative_completion() {
		let cancellation = CancellationToken::new();
		let observed = cancellation.clone();
		let output = await_cooperative_cleanup(&cancellation, Duration::ZERO, async move {
			observed.cancelled().await;
			"cleaned"
		})
		.await;

		assert!(cancellation.is_cancelled());
		assert_eq!(output, "cleaned");
	}
}
