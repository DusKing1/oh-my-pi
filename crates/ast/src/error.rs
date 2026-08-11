//! Error types for tree-sitter operations, AST pattern compilation, and
//! structural rewrites.

use std::path::PathBuf;

use omp_core::Str;
use thiserror::Error;

/// An error that can occur during AST parsing, pattern compilation, or
/// structural editing.
#[derive(Debug, Error)]
pub enum AstError {
	/// Failed to load tree-sitter language.
	#[error("failed to load tree-sitter language: {source}")]
	LoadLanguage {
		/// Underlying tree-sitter language error.
		#[source]
		source: tree_sitter::LanguageError,
	},

	/// Language is not supported.
	#[error("unsupported language '{value}'. Supported: {supported}")]
	UnsupportedLanguage {
		/// Given language alias.
		value:     Str,
		/// List of supported language aliases.
		supported: Str,
	},

	/// Unable to infer language from file path.
	#[error("unable to infer language from file extension: {path:?}. Specify `lang` explicitly.")]
	InferLanguageFailed {
		/// Path of the file.
		path: PathBuf,
	},

	/// Structural pattern is invalid.
	#[error("invalid pattern: {source}")]
	InvalidPattern {
		/// Underlying pattern error.
		#[source]
		source: ast_grep_core::matcher::PatternError,
	},

	/// Overlapping replacements detected.
	#[error("overlapping replacements detected; refine pattern to avoid ambiguous edits")]
	OverlappingReplacements,

	/// Computed edit range is out of bounds.
	#[error("computed edit range is out of bounds")]
	EditRangeOutOfBounds,

	/// Replacement text is not valid UTF-8.
	#[error("replacement text is not valid UTF-8: {source}")]
	NonUtf8Replacement {
		/// Underlying UTF-8 error.
		#[source]
		source: std::str::Utf8Error,
	},
}

/// An AST result.
pub type Result<T, E = AstError> = std::result::Result<T, E>;
