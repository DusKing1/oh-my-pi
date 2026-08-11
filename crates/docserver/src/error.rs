use std::{fmt, io, path::PathBuf};

use omp_core::Str;
use thiserror::Error as ThisError;

use crate::{DocumentId, LeaseId, Revision, TransactionId};

/// The range coordinate system associated with an invalid range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RangeKind {
	/// A range measured in bytes.
	Byte,
	/// A range measured in zero-based lines.
	Line,
}

impl fmt::Display for RangeKind {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(match self {
			Self::Byte => "byte",
			Self::Line => "line",
		})
	}
}

/// A failure produced while resolving or operating on documents.
#[derive(Debug, ThisError)]
pub enum Error {
	/// A path, URI, identifier, or other operation target is not valid.
	#[error("invalid target {target}: {reason}")]
	InvalidTarget {
		/// The rejected target.
		target: Str,
		/// Why the target was rejected.
		reason: Str,
	},

	/// A resource exists but its active ownership or state forbids the
	/// operation.
	#[error("precondition failed for {target}: {reason}")]
	PreconditionFailed {
		/// The resource whose state rejected the operation.
		target: Str,
		/// The failed ownership or state requirement.
		reason: Str,
	},

	/// A half-open range is reversed or extends past the available content.
	#[error("invalid {kind} range {start}..{end} for upper bound {upper_bound:?}")]
	InvalidRange {
		/// The range coordinate system.
		kind:        RangeKind,
		/// The inclusive start coordinate.
		start:       u64,
		/// The exclusive end coordinate.
		end:         u64,
		/// The known exclusive upper bound, if validation had one.
		upper_bound: Option<u64>,
	},

	/// Supplied document bytes violate the declared document state.
	#[error("invalid content: {reason}")]
	InvalidContent {
		/// Why the content was rejected.
		reason: Str,
	},

	/// The requested document is not known to the Environment.
	#[error("document {document_id} was not found")]
	DocumentNotFound {
		/// The unresolved document identifier.
		document_id: DocumentId,
	},

	/// A requested revision was never committed for the document.
	#[error("revision {revision} is not present for document {document_id}")]
	RevisionMissing {
		/// The document whose history was queried.
		document_id: DocumentId,
		/// The requested revision.
		revision:    Revision,
	},

	/// A formerly committed revision has fallen out of the retained cache.
	#[error("revision {revision} has expired for document {document_id}")]
	RevisionExpired {
		/// The document whose history was queried.
		document_id: DocumentId,
		/// The requested revision.
		revision:    Revision,
	},

	/// The lease no longer keeps an active document alive.
	#[error("document lease {lease_id} is missing or expired")]
	LeaseExpired {
		/// The expired or unknown lease.
		lease_id: LeaseId,
	},

	/// A current-revision precondition no longer matches the document head.
	#[error("document content changed: expected {expected}, current {current}")]
	ContentModified {
		/// Revision supplied by the caller.
		expected: Revision,
		/// Current committed revision.
		current:  Revision,
	},

	/// A transaction was based on a revision other than the current head.
	#[error("transaction {transaction_id} is stale: expected {expected}, current {current}")]
	StaleTransaction {
		/// The rejected transaction.
		transaction_id: TransactionId,
		/// The transaction's base revision.
		expected:       Revision,
		/// The current committed revision.
		current:        Revision,
	},

	/// A transaction overlaps another change and cannot be safely rebased.
	#[error("transaction {transaction_id} conflicts on document {document_id}")]
	ConflictingTransaction {
		/// The rejected transaction.
		transaction_id: TransactionId,
		/// The document containing the conflict.
		document_id:    DocumentId,
	},

	/// External filesystem activity invalidated provisional document state.
	#[error("external filesystem change invalidated {path:?}")]
	ExternalInvalidation {
		/// The canonical path whose disk state changed.
		path: PathBuf,
	},

	/// The persisted entry no longer matches the state required for an atomic
	/// replacement.
	#[error("persisted state changed before replacement of {path:?}")]
	StaleDiskState {
		/// The destination whose immediately observed state did not match.
		path: PathBuf,
	},

	/// Installing or receiving a native filesystem watch failed.
	#[error("watch failed for {path:?}: {source}")]
	Watch {
		/// The canonical watched path.
		path:   PathBuf,
		/// The watcher failure.
		#[source]
		source: notify::Error,
	},

	/// Preparing or atomically committing persisted content failed.
	#[error("persistence failed for {path:?}: {source}")]
	Persistence {
		/// The canonical destination path.
		path:   PathBuf,
		/// The underlying filesystem failure.
		#[source]
		source: io::Error,
	},

	/// A protocol message or protocol state transition was invalid.
	#[error("protocol failure: {reason}")]
	Protocol {
		/// The protocol violation.
		reason: Str,
	},

	/// A local filesystem operation failed.
	#[error("I/O operation {operation} failed for {path:?}: {source}")]
	Io {
		/// The operation being attempted.
		operation: Str,
		/// The canonical path, or best available unresolved path on resolution
		/// failure.
		path:      PathBuf,
		/// The underlying I/O failure.
		#[source]
		source:    io::Error,
	},
}

/// A document-server result.
pub type Result<T> = std::result::Result<T, Error>;
