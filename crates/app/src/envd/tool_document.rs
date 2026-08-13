//! `omp-tools` adapters over the app-owned document and blob hosts.

use std::{
	future::{Future, ready},
	path::{Component, Path, PathBuf},
	sync::atomic::{AtomicU64, Ordering},
	time::{SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use omp_core::{Str, encoding::hex, fmts};
use omp_hashline::numbered_diff;
use omp_proto::document::v1 as pb;
use omp_tool::BlobRef;
use omp_tools::{
	edit::{
		CommitResult, Conflict, EditCommitError, EditDocuments, EditPrepared, EditProposal,
		Fault as EditFault, FormatPolicy, RejectionReason, StalePolicy,
	},
	read::{
		DocumentKind, Fault as ReadFault, LeaseContent, LineRange, ReadBlobs, ReadDocuments,
		ReadLease, SummarySegment, TextSlice,
	},
};
use tokio_util::sync::CancellationToken;
use url::Url;

use super::{
	blobs::BlobHost,
	docs::{DocumentHost, DocumentLease},
};

const BINARY_MEDIA_TYPE: &str = "application/octet-stream";
static NEXT_TRANSACTION: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct ResolvedDocument {
	uri:  Str,
	path: Str,
}

/// App-owned read lease pairing the protocol lease with stable tool metadata.
#[derive(Debug)]
pub struct DocumentReadLease {
	host:     DocumentHost,
	lease:    DocumentLease,
	revision: Str,
	kind:     DocumentKind,
}

/// Prepared hashline edit retaining its exact protocol lease and base snapshot.
#[derive(Debug)]
pub struct PreparedDocument {
	host:          DocumentHost,
	lease:         DocumentLease,
	path:          Str,
	base_revision: Str,
	base_bytes:    Bytes,
}

impl ReadDocuments for DocumentHost {
	type Lease = DocumentReadLease;

	fn open(&self, path: Str) -> impl Future<Output = Result<Self::Lease, ReadFault>> + Send + '_ {
		async move {
			let resolved = resolve_document(self, &path).map_err(read_open_fault)?;
			let lease = DocumentHost::open(self, resolved.uri, None, &CancellationToken::new())
				.await
				.map_err(|error| read_open_fault(error.to_string()))?;
			let revision = revision_identity(lease.head()).map_err(read_open_fault)?;
			let kind = classify_head(lease.head(), &resolved.path).map_err(read_open_fault)?;
			Ok(DocumentReadLease { host: self.clone(), lease, revision, kind })
		}
	}
}

impl ReadLease for DocumentReadLease {
	fn revision(&self) -> &Str {
		&self.revision
	}

	fn kind(&self) -> &DocumentKind {
		&self.kind
	}

	fn read<'a>(
		&'a self,
		ranges: &'a [LineRange],
		structural: bool,
	) -> impl Future<Output = Result<LeaseContent, ReadFault>> + Send + 'a {
		async move {
			if matches!(self.kind, DocumentKind::Binary { .. }) {
				return read_binary(&self.host, &self.lease).await;
			}
			if structural
				&& let Some(summary) = read_summary(&self.host, &self.lease).await?
			{
				return Ok(summary);
			}
			read_text(&self.host, &self.lease, ranges).await
		}
	}
}

impl ReadBlobs for BlobHost {
	fn store(
		&self,
		bytes: Bytes,
		media_type: Str,
	) -> impl Future<Output = Result<BlobRef, ReadFault>> + Send + '_ {
		let result = self
			.put(&bytes)
			.map_err(|error| ReadFault::Blob { message: Str::from(error.to_string()) })
			.map(|id| BlobRef {
				hash: Str::from(hex::encode_n(&id.hash).as_str()),
				media_type,
				byte_len: id.size,
			});
		ready(result)
	}
}

impl EditPrepared for PreparedDocument {
	fn path(&self) -> &Str {
		&self.path
	}

	fn base_revision(&self) -> &Str {
		&self.base_revision
	}

	fn base_bytes(&self) -> &Bytes {
		&self.base_bytes
	}
}

impl EditDocuments for DocumentHost {
	type Prepared = PreparedDocument;

	fn prepare(&self, path: Str) -> impl Future<Output = Result<Self::Prepared, EditFault>> + Send + '_ {
		async move {
			let resolved = resolve_document(self, &path).map_err(edit_invalid)?;
			let lease = DocumentHost::open(self, resolved.uri, None, &CancellationToken::new())
				.await
				.map_err(|error| edit_invalid(error.to_string()))?;
			if pb::DocumentKind::try_from(lease.head().kind) != Ok(pb::DocumentKind::Text) {
				return Err(edit_invalid("hashline edits require a text document"));
			}
			let base_revision = revision_identity(lease.head()).map_err(edit_invalid)?;
			let canonical_path = document_path(lease.head()).map_err(edit_invalid)?;
			let base_bytes = read_whole(self, &lease)
				.await
				.map_err(|error| edit_invalid(error.to_string()))?;
			Ok(PreparedDocument {
				host: self.clone(),
				lease,
				path: canonical_path,
				base_revision,
				base_bytes,
			})
		}
	}

	fn commit(
		&self,
		mut prepared: Self::Prepared,
		proposal: EditProposal,
	) -> impl Future<Output = Result<CommitResult, EditCommitError>> + Send + '_ {
		async move {
			let EditProposal {
				format,
				payload,
				base_revision,
				stale_policy,
				format_policy,
				applied_ops,
			} = proposal;
			if format != "omp.hashline" {
				return Err(EditCommitError::Rejected(edit_invalid(
					"document host accepts only omp.hashline proposals",
				)));
			}
			if base_revision != prepared.base_revision {
				return Err(EditCommitError::Rejected(EditFault {
					reason: RejectionReason::StaleUnrecoverable,
					conflicts: Vec::new(),
				}));
			}
			let mutation = text_mutation(format, payload, stale_policy, format_policy);
			let transaction_id = transaction_id(prepared.host.hello().server_epoch.as_ref());
			let response = prepared
				.host
				.commit(
					&mut prepared.lease,
					transaction_id.clone(),
					mutation,
					&CancellationToken::new(),
				)
				.await
				.map_err(|error| edit_unknown(error.to_string()))?;
			let committed = match response.outcome {
				Some(pb::commit_transaction_response::Outcome::Committed(committed))
					if committed.transaction_id == transaction_id =>
				{
					committed
				},
				Some(pb::commit_transaction_response::Outcome::Rejected(rejected))
					if rejected.transaction_id == transaction_id =>
				{
					return Err(EditCommitError::Rejected(map_rejection(
						&rejected,
						&prepared.base_bytes,
					)));
				},
				Some(pb::commit_transaction_response::Outcome::PartiallyCommitted(partial))
					if partial.transaction_id == transaction_id =>
				{
					return Err(edit_unknown(fmts!(
						"document transaction partially committed before operation {}: {}",
						partial.failed_operation_index,
						partial.message
					)));
				},
				Some(_) => return Err(edit_unknown("document transaction identity did not match")),
				None => return Err(edit_unknown("document transaction omitted its outcome")),
			};
			let operation = committed
				.operations
				.first()
				.filter(|operation| committed.operations.len() == 1 && operation.operation_index == 0)
				.ok_or_else(|| {
					edit_unknown("document transaction did not return exactly operation 0")
				})?;
			let new_revision = operation
				.head
				.as_ref()
				.ok_or_else(|| edit_unknown("committed operation omitted its document head"))
				.and_then(|head| revision_identity(head).map_err(edit_unknown))?;
			let rebased = operation.rebased;
			let committed_bytes = read_whole(&prepared.host, &prepared.lease)
				.await
				.map_err(|error| edit_unknown(error.to_string()))?;
			let diff = numbered_diff(&prepared.base_bytes, &committed_bytes)
				.map_err(|error| edit_unknown(error.to_string()))?
				.text;
			Ok(CommitResult { new_revision, applied_ops, rebased, diff })
		}
	}
}

async fn read_binary(
	host: &DocumentHost,
	lease: &DocumentLease,
) -> Result<LeaseContent, ReadFault> {
	read_whole(host, lease)
		.await
		.map(LeaseContent::Binary)
		.map_err(|error| read_fault(error.to_string()))
}

async fn read_text(
	host: &DocumentHost,
	lease: &DocumentLease,
	ranges: &[LineRange],
) -> Result<LeaseContent, ReadFault> {
	let selection = line_selection(ranges).map_err(read_fault)?;
	let response = host
		.read(lease, selection, &CancellationToken::new())
		.await
		.map_err(|error| read_fault(error.to_string()))?;
	let slices = match response.body {
		Some(pb::read_document_response::Body::Content(content)) => vec![TextSlice {
			start_line: 1,
			text: Str::from_utf8_owned(content)
				.map_err(|error| read_fault(error.to_string()))?,
		}],
		Some(pb::read_document_response::Body::Slices(content)) => content
			.slices
			.into_iter()
			.map(|slice| {
				Ok(TextSlice {
					start_line: slice
						.start
						.checked_add(1)
						.ok_or_else(|| read_fault("document line coordinate overflowed"))?,
					text: Str::from_utf8_owned(slice.content)
						.map_err(|error| read_fault(error.to_string()))?,
				})
			})
			.collect::<Result<Vec<_>, ReadFault>>()?,
		None => return Err(read_fault("document read omitted its body")),
	};
	Ok(LeaseContent::Text { slices, elided: Vec::new() })
}

async fn read_summary(
	host: &DocumentHost,
	lease: &DocumentLease,
) -> Result<Option<LeaseContent>, ReadFault> {
	let language = lease.head().language_id.clone();
	let response = host
		.summarize(
			lease,
			pb::CodeSummaryOptions {
				min_body_lines: 4,
				min_comment_lines: 6,
				unfold_until_lines: 50,
				unfold_limit_lines: 100,
				enable_prose: false,
				min_total_lines: 100,
				render_mode: pb::SummaryRenderMode::Hashline as i32,
				language,
			},
			&CancellationToken::new(),
		)
		.await
		.map_err(|error| read_fault(error.to_string()))?;
	let Some(outcome) = response.outcome else {
		return Err(read_fault("document summary omitted its outcome"));
	};
	let pb::summarize_document_response::Outcome::Summary(summary) = outcome else {
		return Ok(None);
	};
	let mut segments = Vec::with_capacity(summary.segments.len());
	let mut elided = Vec::new();
	for segment in summary.segments {
		let start_line = u64::from(segment.start_line);
		let end_line = u64::from(segment.end_line);
		match pb::document_summary_segment::Kind::try_from(segment.kind) {
			Ok(pb::document_summary_segment::Kind::Kept) => segments.push(SummarySegment {
				start_line,
				end_line,
				text: segment.text.map(Str::from).unwrap_or_default(),
				elided: false,
			}),
			Ok(pb::document_summary_segment::Kind::Elided) => {
				segments.push(SummarySegment {
					start_line,
					end_line,
					text: Str::default(),
					elided: true,
				});
				elided.push(LineRange { start: start_line, end: end_line });
			},
			_ => return Err(read_fault("document summary contained an unknown segment kind")),
		}
	}
	Ok(Some(LeaseContent::Summary { segments, elided }))
}

async fn read_whole(
	host: &DocumentHost,
	lease: &DocumentLease,
) -> Result<Bytes, super::docs::DocumentError> {
	let response = host
		.read(
			lease,
			pb::ReadSelection {
				selection: Some(pb::read_selection::Selection::Whole(pb::WholeDocument {})),
			},
			&CancellationToken::new(),
		)
		.await?;
	match response.body {
		Some(pb::read_document_response::Body::Content(content)) => Ok(content),
		_ => Err(super::docs::DocumentError::MalformedResponse(Str::new_static(
			"whole document read did not return content",
		))),
	}
}

fn line_selection(ranges: &[LineRange]) -> Result<pb::ReadSelection, &'static str> {
	if ranges.is_empty() {
		return Ok(pb::ReadSelection {
			selection: Some(pb::read_selection::Selection::Whole(pb::WholeDocument {})),
		});
	}
	let ranges = ranges
		.iter()
		.map(|range| {
			if range.start == 0 || range.end < range.start {
				return Err("line ranges must be one-based, inclusive, and ordered");
			}
			Ok(pb::LineRange { start: range.start - 1, end: range.end })
		})
		.collect::<Result<Vec<_>, _>>()?;
	Ok(pb::ReadSelection {
		selection: Some(pb::read_selection::Selection::Lines(pb::LineRangeSelection { ranges })),
	})
}

fn resolve_document(host: &DocumentHost, input: &str) -> Result<ResolvedDocument, String> {
	let root_url = Url::parse(host.hello().root_uri.as_str())
		.map_err(|error| format!("document workspace root is not a valid URI: {error}"))?;
	if root_url.scheme() != "file" {
		return Err("document workspace root is not a file URI".into());
	}
	if root_url.query().is_some() || root_url.fragment().is_some() {
		return Err("document workspace root file URI cannot contain a query or fragment".into());
	}
	let root_path = root_url
		.to_file_path()
		.map_err(|()| "document workspace root is not a local file URI".to_owned())?;
	let root_path = normalize_absolute(&root_path)?;
	let parsed = Url::parse(input).ok();
	let (candidate, preserve_uri) = match parsed {
		Some(uri) => {
			if uri.scheme() != "file" || uri.query().is_some() || uri.fragment().is_some() {
				return Err("document URI must be a query-free file URI inside the workspace".into());
			}
			let path = uri
				.to_file_path()
				.map_err(|()| "document URI is not a local file URI".to_owned())?;
			(normalize_absolute(&path)?, Some(uri))
		},
		None => {
			let relative = normalize_relative(Path::new(input))?;
			(root_path.join(relative), None)
		},
	};
	if candidate == root_path || !candidate.starts_with(&root_path) {
		return Err("document path escapes or names the workspace root".into());
	}
	ensure_canonical_containment(&root_path, &candidate)?;
	let relative = candidate
		.strip_prefix(&root_path)
		.map_err(|_| "document path escapes the workspace root".to_owned())?;
	let relative = relative
		.to_str()
		.ok_or_else(|| "document path is not valid UTF-8".to_owned())?
		.replace('\\', "/");
	let uri = match preserve_uri {
		Some(uri) => uri,
		None => Url::from_file_path(&candidate)
			.map_err(|()| "document path cannot be represented as a file URI".to_owned())?,
	};
	Ok(ResolvedDocument { uri: Str::from(uri.as_str()), path: Str::from(relative) })
}
fn ensure_canonical_containment(root: &Path, candidate: &Path) -> Result<(), String> {
	let canonical_root = std::fs::canonicalize(root)
		.map_err(|error| format!("cannot canonicalize document workspace root: {error}"))?;
	let canonical_candidate = std::fs::canonicalize(candidate)
		.map_err(|error| format!("cannot canonicalize document path: {error}"))?;
	if canonical_candidate == canonical_root || !canonical_candidate.starts_with(&canonical_root) {
		return Err("document path escapes the canonical workspace root".into());
	}
	Ok(())
}

fn normalize_relative(path: &Path) -> Result<PathBuf, String> {
	let mut normalized = PathBuf::new();
	for component in path.components() {
		match component {
			Component::CurDir => {},
			Component::Normal(component) => normalized.push(component),
			Component::ParentDir => {
				if !normalized.pop() {
					return Err("document path lexically escapes the workspace root".into());
				}
			},
			Component::RootDir | Component::Prefix(_) => {
				return Err("document path must be workspace-relative".into());
			},
		}
	}
	if normalized.as_os_str().is_empty() {
		return Err("document path must name a file below the workspace root".into());
	}
	Ok(normalized)
}

fn normalize_absolute(path: &Path) -> Result<PathBuf, String> {
	if !path.is_absolute() {
		return Err("file URI did not resolve to an absolute path".into());
	}
	let mut normalized = PathBuf::new();
	for component in path.components() {
		match component {
			Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
			Component::RootDir => normalized.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
			Component::CurDir => {},
			Component::Normal(component) => normalized.push(component),
			Component::ParentDir => {
				if !normalized.pop() {
					return Err("file URI lexically escapes its filesystem root".into());
				}
			},
		}
	}
	Ok(normalized)
}

fn classify_head(head: &pb::DocumentHead, path: &str) -> Result<DocumentKind, String> {
	match pb::DocumentKind::try_from(head.kind) {
		Ok(pb::DocumentKind::Text) => Ok(DocumentKind::Text),
		Ok(pb::DocumentKind::Binary) => Ok(DocumentKind::Binary {
			media_type: Str::new_static(BINARY_MEDIA_TYPE),
			fallback: fmts!("{path} is binary content ({0} bytes)", head.byte_length),
		}),
		_ => Err("document head omitted a recognized text/binary kind".into()),
	}
}

fn revision_identity(head: &pb::DocumentHead) -> Result<Str, String> {
	let revision = head
		.revision
		.as_ref()
		.ok_or_else(|| "document head omitted its revision".to_owned())?;
	let hash: &[u8; 32] = revision
		.content_hash
		.as_ref()
		.try_into()
		.map_err(|_| "document revision hash is not 32 bytes".to_owned())?;
	Ok(fmts!("{}:{}", revision.sequence, hex::encode_n(hash).as_str()))
}

fn document_path(head: &pb::DocumentHead) -> Result<Str, String> {
	let uri = head
		.document
		.as_ref()
		.ok_or_else(|| "document head omitted its canonical document reference".to_owned())?
		.uri
		.as_str();
	let uri = Url::parse(uri)
		.map_err(|error| format!("document head returned an invalid canonical URI: {error}"))?;
	if uri.scheme() != "file" {
		return Err("document head canonical URI is not a file URI".into());
	}
	let path = uri
		.to_file_path()
		.map_err(|()| "document head canonical URI is not a local file URI".to_owned())?;
	path
		.to_str()
		.map(Str::new)
		.ok_or_else(|| "document canonical path is not valid UTF-8".to_owned())
}

fn text_mutation(
	format: Str,
	payload: Str,
	stale_policy: StalePolicy,
	format_policy: FormatPolicy,
) -> pb::TextMutation {
	pb::TextMutation {
		base_revision: None,
		change: Some(pb::text_mutation::Change::Proposal(pb::EditFormatProposal {
			format: format.into(),
			payload: Bytes::from(payload),
			options_json: Bytes::new(),
		})),
		stale_policy: match stale_policy {
			StalePolicy::RebaseNonOverlapping => pb::StalePolicy::RebaseNonOverlapping as i32,
		},
		format_policy: match format_policy {
			FormatPolicy::Configured => pb::FormatPolicy::BestEffort as i32,
		},
	}
}

fn transaction_id(server_epoch: &[u8]) -> Bytes {
	let sequence = NEXT_TRANSACTION.fetch_add(1, Ordering::Relaxed);
	let now = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_nanos();
	let mut hasher = blake3::Hasher::new();
	hasher.update(server_epoch);
	hasher.update(&std::process::id().to_le_bytes());
	hasher.update(&sequence.to_le_bytes());
	hasher.update(&now.to_le_bytes());
	Bytes::copy_from_slice(&hasher.finalize().as_bytes()[..16])
}


fn map_rejection(rejected: &pb::TransactionRejected, base: &[u8]) -> EditFault {
	let reason = match pb::TransactionRejectReason::try_from(rejected.reason) {
		Ok(pb::TransactionRejectReason::OverlappingChange) => RejectionReason::Conflict,
		Ok(pb::TransactionRejectReason::StaleBase)
		| Ok(pb::TransactionRejectReason::ExternalModification)
		| Ok(pb::TransactionRejectReason::RevisionExpired) => RejectionReason::StaleUnrecoverable,
		Ok(pb::TransactionRejectReason::FormatFailed) => {
			RejectionReason::Format { message: Str::from(rejected.message.as_str()) }
		},
		Ok(pb::TransactionRejectReason::InvalidContent) => {
			RejectionReason::InvalidPatch { message: Str::from(rejected.message.as_str()) }
		},
		Ok(pb::TransactionRejectReason::PersistFailed) => RejectionReason::InvalidPatch {
			message: fmts!("document persistence failed: {}", rejected.message),
		},
		Ok(pb::TransactionRejectReason::PreconditionFailed) => RejectionReason::InvalidPatch {
			message: fmts!("document precondition failed: {}", rejected.message),
		},
		Ok(pb::TransactionRejectReason::Cancelled) => RejectionReason::InvalidPatch {
			message: fmts!("document transaction was cancelled: {}", rejected.message),
		},
		Ok(pb::TransactionRejectReason::Unspecified) | Err(_) => RejectionReason::InvalidPatch {
			message: fmts!("document transaction returned an unknown rejection: {}", rejected.message),
		},
	};
	let conflicts = rejected
		.conflicts
		.iter()
		.flat_map(|conflict| conflict.conflicting_ranges.iter())
		.map(|range| Conflict {
			start_line: line_at_offset(base, range.start),
			end_line: line_at_offset(base, range.end.saturating_sub(1).max(range.start)),
			message: Str::from(rejected.message.as_str()),
		})
		.collect();
	EditFault { reason, conflicts }
}


fn line_at_offset(bytes: &[u8], offset: u64) -> usize {
	let offset = usize::try_from(offset).unwrap_or(usize::MAX).min(bytes.len());
	bytes[..offset]
		.iter()
		.filter(|&&byte| byte == b'\n')
		.count()
		.saturating_add(1)
}


fn read_open_fault(message: impl Into<String>) -> ReadFault {
	ReadFault::Open { message: Str::from(message.into()) }
}

fn read_fault(message: impl Into<String>) -> ReadFault {
	ReadFault::Read { message: Str::from(message.into()) }
}

fn edit_invalid(message: impl Into<String>) -> EditFault {
	EditFault {
		reason: RejectionReason::InvalidPatch { message: Str::from(message.into()) },
		conflicts: Vec::new(),
	}
}

fn edit_unknown(reason: impl Into<Str>) -> EditCommitError {
	EditCommitError::EffectsUnknown { reason: reason.into() }
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn converts_one_based_inclusive_ranges_to_protocol_coordinates() {
		let selection = line_selection(&[LineRange { start: 1, end: 1 }, LineRange {
			start: 4,
			end: 9,
		}])
		.expect("valid ranges");
		let Some(pb::read_selection::Selection::Lines(lines)) = selection.selection else {
			panic!("expected line selection");
		};
		assert_eq!(lines.ranges, vec![
			pb::LineRange { start: 0, end: 1 },
			pb::LineRange { start: 3, end: 9 },
		]);
	}

	#[test]
	fn rejects_lexical_parent_escape() {
		assert!(normalize_relative(Path::new("../outside.rs")).is_err());
		assert_eq!(
			normalize_relative(Path::new("src/../inside.rs")).expect("contained parent"),
			PathBuf::from("inside.rs")
		);
	}
	#[cfg(unix)]
	#[test]
	fn rejects_symlink_escape_after_canonicalization() {
		use std::os::unix::fs::symlink;

		let sandbox = tempfile::tempdir().expect("sandbox");
		let root = sandbox.path().join("root");
		let outside = sandbox.path().join("outside");
		std::fs::create_dir_all(&root).expect("root");
		std::fs::create_dir_all(&outside).expect("outside");
		std::fs::write(outside.join("secret"), b"secret").expect("outside file");
		symlink(&outside, root.join("link")).expect("symlink");
		assert!(ensure_canonical_containment(&root, &root.join("link/secret")).is_err());
	}


	#[test]
	fn revision_identity_includes_sequence_and_exact_hash() {
		let head = pb::DocumentHead {
			revision: Some(pb::Revision {
				sequence: 7,
				content_hash: Bytes::from_static(&[0xab; 32]),
			}),
			..Default::default()
		};
		assert_eq!(
			revision_identity(&head).expect("valid revision"),
			"7:abababababababababababababababababababababababababababababababab"
		);
	}

	#[test]
	fn forwards_complete_hashline_payload_unchanged() {
		let payload = Str::new_static("[/repo/f.rs#ABCD]\nPUT 1.=1:\n+new");
		let mutation = text_mutation(
			"omp.hashline".into(),
			payload.clone(),
			StalePolicy::RebaseNonOverlapping,
			FormatPolicy::Configured,
		);
		let Some(pb::text_mutation::Change::Proposal(proposal)) = mutation.change else {
			panic!("expected edit format proposal");
		};
		assert_eq!(proposal.payload, Bytes::from(payload));
		assert_eq!(mutation.format_policy, pb::FormatPolicy::BestEffort as i32);
	}

	#[test]
	fn maps_format_and_conflict_rejections_without_message_parsing() {
		let format = map_rejection(
			&pb::TransactionRejected {
				reason: pb::TransactionRejectReason::FormatFailed as i32,
				message: "opaque formatter diagnostic".into(),
				..Default::default()
			},
			b"one\ntwo\nthree\n",
		);
		assert_eq!(
			format.reason,
			RejectionReason::Format { message: "opaque formatter diagnostic".into() }
		);

		let conflict = map_rejection(
			&pb::TransactionRejected {
				reason: pb::TransactionRejectReason::OverlappingChange as i32,
				message: "opaque overlap".into(),
				conflicts: vec![pb::DocumentConflict {
					conflicting_ranges: vec![pb::ByteRange { start: 4, end: 7 }],
					..Default::default()
				}],
				..Default::default()
			},
			b"one\ntwo\nthree\n",
		);
		assert_eq!(conflict.reason, RejectionReason::Conflict);
		assert_eq!(conflict.conflicts[0].start_line, 2);
		assert_eq!(conflict.conflicts[0].end_line, 2);
	}
}
