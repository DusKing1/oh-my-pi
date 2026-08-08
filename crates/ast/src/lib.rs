//! Tree-sitter-backed source understanding and structural editing.

/// AST-aware block resolution.
pub mod block;
/// Supported language definitions and inference.
pub mod language;
/// Structural search and rewrite operations.
pub mod ops;
/// Structural source summarization.
pub mod summary;

pub use language::SupportLang;
