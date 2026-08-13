//! Resource-owning built-in tools for the OMP environment.
//!
//! Executors consume the same streaming invocation contract as extensions:
//! speculative preparation may begin while arguments arrive, while filesystem
//! and process effects remain behind the explicit commitment gate. Durable
//! payloads are revisioned truth and prompt parts are deterministic
//! projections.

mod render;

/// Hashline document transactions with speculative previews.
pub mod edit;
/// Deterministic workspace path matching.
pub mod glob;
/// Workspace byte and pattern search.
pub mod grep;
/// Revision-pinned document reads and structural summaries.
pub mod read;
/// Persistent-session shell execution.
pub mod shell;
