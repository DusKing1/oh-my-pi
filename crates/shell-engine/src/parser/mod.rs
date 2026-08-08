//! Implements a tokenizer and parsers for POSIX / bash shell syntax.

#![allow(
	clippy::unwrap_used,
	reason = "parser diagnostics retain source locations during recovery"
)]

pub mod arithmetic;
pub mod ast;
pub mod pattern;
pub mod prompt;
pub mod test_command;
pub mod word;

mod error;
mod program;
mod source;
mod tokenizer;

pub use error::{ParseError, ParseErrorLocation, TestCommandParseError, WordParseError};
pub use program::{Parser, ParserBuilder, ParserImpl, ParserOptions, SourceInfo, parse_tokens};
pub use source::{SourcePosition, SourcePositionOffset, SourceSpan};
pub use tokenizer::{
	Token, TokenLocation, TokenizerError, TokenizerOptions, tokenize_str, tokenize_str_with_options,
	uncached_tokenize_str, unquote_str,
};

#[cfg(test)]
/// Result type for parser tests that propagate heterogeneous errors.
pub(crate) type TestResult<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;
