//! Tree-sitter-backed source understanding and structural editing.

/// AST-aware block resolution.
pub mod block;
/// AST error and result types.
pub mod error;
/// Supported language definitions and inference.
pub mod language;
/// Structural search and rewrite operations.
pub mod ops;
/// Structural source summarization.
pub mod summary;

pub use error::{AstError, Result};
pub use language::SupportLang;
