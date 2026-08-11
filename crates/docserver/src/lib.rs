//! Local document authority for an OMP Environment.
//!
//! This crate owns the domain model used by filesystem, revision, transaction,
//! watch, and language-server components. Runtime components are exposed only
//! once they have complete implementations.

mod actor;
/// Concurrent framed protocol connections over a shared Environment.
pub mod connection;
/// Long-lived document authority over standard I/O or a Unix-domain socket.
pub mod daemon;
/// Session-scoped lowering of opaque edit-format proposals.
pub mod edit_adapter;
/// Project-scoped authority and connection-local sessions.
pub mod environment;
mod error;
/// Portable Environment filesystem value types.
pub mod fs;
/// Ordered LSP lifecycle, synchronization, and passthrough primitives.
pub mod lsp;
/// Transactional lowering for server-initiated workspace edit requests.
pub mod lsp_apply_edit;
/// Bounded child-process JSON-RPC transport and production LSP binding startup.
pub mod lsp_process;
pub mod lsp_registry;
/// Actor-aware Environment path operations.
pub mod path_ops;
/// Checked LSP position encoding and text-edit conversion.
pub mod position;
mod protocol;
mod rebase;
pub mod summary;
pub mod transaction;
mod types;
mod watch;
/// Bounded length-delimited protobuf transport framing.
pub mod wire;
pub use actor::{
	ContentSlice, DocumentEvent, DocumentEventKind, DocumentLocator, DocumentStore, OpenedDocument,
	ReadBody, ReadResult, ReadSelection,
};
pub use edit_adapter::{
	EditAdapterRegistry, HASHLINE_EDIT_FORMAT, REPLACE_EDIT_FORMAT, TextEditAdapter,
};
pub use environment::{Environment, EnvironmentSession};
pub use error::{Error, RangeKind, Result};
pub use fs::{
	CopyOutcome, DestinationOverwritePolicy, DirectoryEntry, ExistingDirectoryPolicy, FileKind,
	FollowSymlinks, PathMetadata, PortablePermissions, SymlinkTarget, SymlinkTargetForm,
	SymlinkTargetKind,
};
pub use lsp_apply_edit::ApplyWorkspaceEditError;
pub use lsp_process::{
	InboundDispatch, LspPostResponse, LspProcess, LspProcessConfig, LspProcessError,
	LspProcessSelectorConfig, LspTransportSettings, load_lsp_process_configs,
};
pub use path_ops::{PathMutationResult, PathService};
pub use rebase::{
	AppliedEdits, ByteEdit, RebaseConflict, apply_edits, canonical_edits, rebase_content,
	rebase_edits, validate_edits,
};
pub use types::{
	AuthorityLock, ByteRange, DocumentHead, DocumentId, DocumentKind, DocumentPresence,
	DocumentSnapshot, FileFingerprint, FileMetadata, LanguageId, LeaseId, LineRange, Revision,
	ServerConfig, TransactionId,
};
pub use watch::{ActiveFileWatch, FileWatchEvent, FileWatchKind, classify_event};
