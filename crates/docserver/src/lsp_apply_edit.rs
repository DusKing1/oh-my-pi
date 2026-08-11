//! Transactional lowering for server-initiated `workspace/applyEdit` requests.

use std::collections::BTreeMap;

use bytes::Bytes;
use serde::Deserialize;
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::{
	ByteEdit, ByteRange, DocumentHead, DocumentKind, DocumentLocator, DocumentPresence, Environment,
	ReadBody, ReadSelection, TransactionId,
	lsp_process::InboundDispatch,
	position::TextEdit,
	transaction::{
		CreateMutation, DeleteMutation, DocumentMutation, DocumentTarget, ExistingDocumentPolicy,
		FormatPolicy, MoveDestinationPrecondition, MoveMutation, MutationOperation, StalePolicy,
		TextMutation, TextProposal, TransactionOutcome, TransactionRequest,
	},
};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApplyEditParams {
	edit: WorkspaceEdit,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkspaceEdit {
	changes:            Option<BTreeMap<String, Vec<TextEdit>>>,
	document_changes:   Option<Vec<Value>>,
	#[serde(default)]
	change_annotations: Map<String, Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TextDocumentEdit {
	text_document: OptionalVersionedDocument,
	edits:         Vec<TextEdit>,
}

#[derive(Deserialize)]
struct OptionalVersionedDocument {
	uri:     String,
	#[serde(default)]
	version: Option<i32>,
}

struct LoadedDocument {
	head:    DocumentHead,
	content: Bytes,
}

/// Applies one server-requested workspace edit through the document transaction
/// authority.
pub async fn apply_workspace_edit(
	environment: Environment,
	handle: crate::lsp_registry::LspBindingHandle,
	params: Bytes,
	cancellation: CancellationToken,
) -> InboundDispatch {
	let request = match lower_workspace_edit(&environment, handle, &params, &cancellation).await {
		Ok(request) => request,
		Err(reason) => return apply_failure(reason, None),
	};
	let transaction_id = request.transaction_id();
	let barrier = environment
		.lsp()
		.defer_transaction_publication(transaction_id);
	let outcome = environment
		.transactions()
		.commit_deferred_publication(request, cancellation)
		.await;
	let response = outcome_response(outcome.as_ref());
	InboundDispatch::success_then(
		response,
		Box::pin(async move {
			barrier.release();
		}),
	)
}

async fn lower_workspace_edit(
	environment: &Environment,
	handle: crate::lsp_registry::LspBindingHandle,
	params: &[u8],
	cancellation: &CancellationToken,
) -> Result<TransactionRequest, String> {
	let params: ApplyEditParams =
		serde_json::from_slice(params).map_err(|error| format!("invalid workspace edit: {error}"))?;
	if params.edit.changes.is_some() && params.edit.document_changes.is_some() {
		return Err(
			"workspace edits cannot mix changes and documentChanges without a declared order"
				.to_owned(),
		);
	}
	if params
		.edit
		.change_annotations
		.values()
		.any(|annotation| annotation.get("needsConfirmation").and_then(Value::as_bool) == Some(true))
	{
		return Err("workspace edit requires interactive confirmation".to_owned());
	}

	let mut operations = Vec::new();
	if let Some(changes) = params.edit.changes {
		for (uri, edits) in changes {
			operations
				.push(lower_text_edit(environment, handle, uri, None, edits, cancellation).await?);
		}
	}
	if let Some(document_changes) = params.edit.document_changes {
		for change in document_changes {
			if change.get("textDocument").is_some() {
				let edit: TextDocumentEdit = serde_json::from_value(change)
					.map_err(|error| format!("invalid text document edit: {error}"))?;
				operations.push(
					lower_text_edit(
						environment,
						handle,
						edit.text_document.uri,
						edit.text_document.version,
						edit.edits,
						cancellation,
					)
					.await?,
				);
				continue;
			}
			let kind = change
				.get("kind")
				.and_then(Value::as_str)
				.ok_or_else(|| "documentChanges entry requires textDocument or kind".to_owned())?;
			match kind {
				"create" => lower_create(environment, &change, cancellation, &mut operations).await?,
				"rename" => lower_rename(environment, &change, cancellation, &mut operations).await?,
				"delete" => lower_delete(environment, &change, cancellation, &mut operations).await?,
				_ => return Err(format!("unsupported workspace resource operation {kind}")),
			}
		}
	}
	Ok(TransactionRequest::new(TransactionId::from_bytes(rand::random()), operations))
}

async fn lower_text_edit(
	environment: &Environment,
	handle: crate::lsp_registry::LspBindingHandle,
	uri: String,
	version: Option<i32>,
	edits: Vec<TextEdit>,
	cancellation: &CancellationToken,
) -> Result<DocumentMutation, String> {
	let uri = parse_uri(&uri)?;
	let revision = match version {
		Some(version) => environment
			.lsp()
			.revision_for_version(handle, &uri, version)
			.map_err(|error| error.to_string())?
			.ok_or_else(|| {
				format!("LSP version {version} has no admitted daemon revision for {uri}")
			})?,
		None => load_document(environment, &uri, None, cancellation)
			.await?
			.head
			.revision(),
	};
	let loaded = load_document(environment, &uri, Some(revision), cancellation).await?;
	if loaded.head.presence() != DocumentPresence::Present {
		return Err(format!("text document {uri} is missing"));
	}
	let language_id = match loaded.head.kind() {
		DocumentKind::Text(language_id) => language_id.as_ref(),
		DocumentKind::Binary => {
			return Err(format!("text edits cannot target binary document {uri}"));
		},
	};
	let policy = environment
		.lsp()
		.sync_policy_for_handle(handle, &uri, language_id)
		.map_err(|error| error.to_string())?;
	let text = std::str::from_utf8(&loaded.content)
		.map_err(|_| format!("text document {uri} does not contain UTF-8"))?;
	let mut byte_edits = Vec::with_capacity(edits.len());
	for edit in edits {
		let (start, end) = policy
			.position_encoding
			.range_to_offsets(text, edit.range)
			.map_err(|error| error.to_string())?;
		let range = ByteRange::new(
			u64::try_from(start).map_err(|_| "edit start exceeds u64".to_owned())?,
			u64::try_from(end).map_err(|_| "edit end exceeds u64".to_owned())?,
		)
		.map_err(|error| error.to_string())?;
		byte_edits.push(ByteEdit::new(range, Bytes::from(edit.new_text)));
	}
	byte_edits.sort_by_key(|edit| edit.range().start());
	crate::validate_edits(
		u64::try_from(loaded.content.len()).map_err(|_| "document length exceeds u64".to_owned())?,
		&byte_edits,
	)
	.map_err(|error| error.to_string())?;
	Ok(DocumentMutation::new(
		DocumentTarget::Uri(uri),
		MutationOperation::Text(TextMutation::new(
			revision,
			TextProposal::Edits(byte_edits),
			StalePolicy::Fail,
			FormatPolicy::Disabled,
		)),
	))
}

async fn lower_create(
	environment: &Environment,
	change: &Value,
	cancellation: &CancellationToken,
	operations: &mut Vec<DocumentMutation>,
) -> Result<(), String> {
	let uri = parse_required_uri(change, "uri")?;
	let overwrite = option(change, "overwrite");
	let ignore = option(change, "ignoreIfExists");
	if ignore && !overwrite {
		let loaded = load_document(environment, &uri, None, cancellation).await?;
		if loaded.head.presence() == DocumentPresence::Present {
			return Ok(());
		}
	}
	let existing = if overwrite {
		ExistingDocumentPolicy::ReplaceExisting
	} else {
		ExistingDocumentPolicy::FailIfExists
	};
	operations.push(DocumentMutation::new(
		DocumentTarget::Uri(uri),
		MutationOperation::Create(CreateMutation::new(
			Bytes::new(),
			existing,
			FormatPolicy::Disabled,
		)),
	));
	Ok(())
}

async fn lower_delete(
	environment: &Environment,
	change: &Value,
	cancellation: &CancellationToken,
	operations: &mut Vec<DocumentMutation>,
) -> Result<(), String> {
	if option(change, "recursive") {
		return Err("recursive workspace deletes are not supported transactionally".to_owned());
	}
	let uri = parse_required_uri(change, "uri")?;
	let loaded = load_document(environment, &uri, None, cancellation).await?;
	if loaded.head.presence() == DocumentPresence::Missing && option(change, "ignoreIfNotExists") {
		return Ok(());
	}
	if loaded.head.presence() != DocumentPresence::Present {
		return Err(format!("delete target {uri} is missing"));
	}
	operations.push(DocumentMutation::new(
		DocumentTarget::Uri(uri),
		MutationOperation::Delete(DeleteMutation::new(loaded.head.revision())),
	));
	Ok(())
}

async fn lower_rename(
	environment: &Environment,
	change: &Value,
	cancellation: &CancellationToken,
	operations: &mut Vec<DocumentMutation>,
) -> Result<(), String> {
	let old_uri = parse_required_uri(change, "oldUri")?;
	let new_uri = parse_required_uri(change, "newUri")?;
	let source = load_document(environment, &old_uri, None, cancellation).await?;
	if source.head.presence() != DocumentPresence::Present {
		return Err(format!("rename source {old_uri} is missing"));
	}
	let destination = load_document(environment, &new_uri, None, cancellation).await?;
	let overwrite = option(change, "overwrite");
	if destination.head.presence() == DocumentPresence::Present
		&& option(change, "ignoreIfExists")
		&& !overwrite
	{
		return Ok(());
	}
	let destination_precondition = if destination.head.presence() == DocumentPresence::Present {
		if !overwrite {
			return Err(format!("rename destination {new_uri} already exists"));
		}
		MoveDestinationPrecondition::Revision(destination.head.revision())
	} else {
		MoveDestinationPrecondition::MustNotExist
	};
	operations.push(DocumentMutation::new(
		DocumentTarget::Uri(old_uri),
		MutationOperation::Move(MoveMutation::new(
			source.head.revision(),
			new_uri,
			destination_precondition,
		)),
	));
	Ok(())
}

async fn load_document(
	environment: &Environment,
	uri: &Url,
	revision: Option<crate::Revision>,
	cancellation: &CancellationToken,
) -> Result<LoadedDocument, String> {
	if cancellation.is_cancelled() {
		return Err("workspace edit was cancelled".to_owned());
	}
	let path = environment
		.store()
		.resolve_entry_path(uri)
		.map_err(|error| error.to_string())?;
	let opened = environment
		.store()
		.open(DocumentLocator::Path(path))
		.await
		.map_err(|error| error.to_string())?;
	let lease_id = opened.lease_id();
	let read = environment
		.store()
		.read(lease_id, revision, ReadSelection::Whole)
		.await;
	let close = environment.store().close(lease_id).await;
	let read = read.map_err(|error| error.to_string())?;
	close.map_err(|error| error.to_string())?;
	let content = match read.body() {
		ReadBody::Whole(content) => content.clone(),
		ReadBody::Slices(_) => unreachable!("whole read returns whole bytes"),
	};
	Ok(LoadedDocument { head: read.head().clone(), content })
}

fn parse_required_uri(value: &Value, field: &str) -> Result<Url, String> {
	let uri = value
		.get(field)
		.and_then(Value::as_str)
		.ok_or_else(|| format!("workspace resource operation requires {field}"))?;
	parse_uri(uri)
}

fn parse_uri(uri: &str) -> Result<Url, String> {
	Url::parse(uri).map_err(|error| format!("invalid workspace edit URI {uri:?}: {error}"))
}

fn option(value: &Value, name: &str) -> bool {
	value
		.get("options")
		.and_then(|options| options.get(name))
		.and_then(Value::as_bool)
		.unwrap_or(false)
}

fn outcome_response(outcome: &TransactionOutcome) -> Bytes {
	match outcome {
		TransactionOutcome::Committed { .. } => apply_response(true, None, None),
		TransactionOutcome::Rejected { message, .. } => {
			apply_response(false, Some(message.as_str()), None)
		},
		TransactionOutcome::PartiallyCommitted { failed_operation_index, message, .. } => {
			apply_response(false, Some(message.as_str()), Some(*failed_operation_index))
		},
	}
}

fn apply_failure(reason: String, failed_change: Option<u32>) -> InboundDispatch {
	InboundDispatch::success(apply_response(false, Some(&reason), failed_change))
}

fn apply_response(applied: bool, reason: Option<&str>, failed_change: Option<u32>) -> Bytes {
	let mut response = Map::new();
	response.insert("applied".to_owned(), Value::Bool(applied));
	if let Some(reason) = reason {
		response.insert("failureReason".to_owned(), Value::String(reason.to_owned()));
	}
	if let Some(failed_change) = failed_change {
		response.insert("failedChange".to_owned(), Value::from(failed_change));
	}
	Bytes::from(serde_json::to_vec(&response).expect("workspace edit response is serializable"))
}
