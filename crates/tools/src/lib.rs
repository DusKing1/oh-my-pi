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
/// Persistent Python evaluation.
pub mod eval;
/// Deterministic workspace path matching.
pub mod glob;
/// Workspace byte and pattern search.
pub mod grep;
/// Pi-compatible reads across local and special sources.
pub mod read;
/// Persistent-session shell execution.
pub mod shell;
/// Pi-compatible whole-file writes.
pub mod write;
