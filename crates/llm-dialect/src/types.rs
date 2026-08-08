//! Borrowed rendering inputs and allocation-disciplined scanner outputs.

use std::fmt;

use bytes::Bytes;
use omp_core::SmolStr;
use serde_json::Value;
use smallvec::SmallVec;
use thiserror::Error;

/// One borrowed example attached to a model-facing tool definition.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum ToolExample<'a> {
	/// A valid example invocation.
	Call {
		/// Optional model-facing explanation of the example.
		caption:   Option<&'a str>,
		/// Example JSON arguments.
		arguments: &'a Value,
	},
	/// A bad invocation paired with its corrected form.
	Contrast {
		/// Optional model-facing explanation of the correction.
		caption: Option<&'a str>,
		/// Arguments the model must avoid.
		bad:     &'a Value,
		/// Corrected arguments the model should emit.
		good:    &'a Value,
	},
	/// Free-form guidance that does not synthesize an invocation.
	Note {
		/// Optional model-facing heading for the note.
		caption: Option<&'a str>,
		/// Model-facing guidance.
		note:    &'a str,
	},
}

/// Borrowed tool metadata shared by prompt rendering and scanner coercion.
///
/// The caller owns the parsed schema and examples for the lifetime of one
/// dialect pipeline. Constructing this view does not clone tool metadata.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct InbandTool<'a> {
	/// Portable dispatch name.
	pub name:        &'a str,
	/// Optional model-facing usage guidance.
	pub description: Option<&'a str>,
	/// Parsed JSON Schema for the tool arguments.
	pub parameters:  &'a Value,
	/// Schema-aware examples and corrections.
	pub examples:    &'a [ToolExample<'a>],
}

impl<'a> InbandTool<'a> {
	/// Creates a borrowed tool definition without cloning its schema or
	/// examples.
	#[must_use]
	pub const fn new(
		name: &'a str,
		description: Option<&'a str>,
		parameters: &'a Value,
		examples: &'a [ToolExample<'a>],
	) -> Self {
		Self { name, description, parameters, examples }
	}
}

/// XML envelope vocabulary used by scanners that share the XML primitive.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum XmlTagset {
	/// Anthropic-compatible `tool_use` and `tool_result` tags.
	#[default]
	Anthropic,
	/// DeepSeek markup-language tool tags.
	Dsml,
}

/// Borrowed configuration used to construct one concrete dialect scanner.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ScannerOptions<'a> {
	/// Tools available for name validation and schema-driven argument coercion.
	pub tools:            &'a [InbandTool<'a>],
	/// Whether leaked model reasoning should become thinking events.
	pub parse_thinking:   bool,
	/// Whether completed tool events retain the original model-authored
	/// envelope.
	pub include_raw_tool: bool,
	/// XML vocabulary used by shared XML scanner primitives.
	pub xml_tagset:       XmlTagset,
}

impl<'a> ScannerOptions<'a> {
	/// Creates scanner options for a borrowed tool inventory.
	#[must_use]
	pub const fn new(tools: &'a [InbandTool<'a>]) -> Self {
		Self {
			tools,
			parse_thinking: true,
			include_raw_tool: false,
			xml_tagset: XmlTagset::Anthropic,
		}
	}
}

impl Default for ScannerOptions<'_> {
	fn default() -> Self {
		Self {
			tools:            &[],
			parse_thinking:   true,
			include_raw_tool: false,
			xml_tagset:       XmlTagset::default(),
		}
	}
}

/// One zero-copy event emitted by an incremental in-band scanner.
///
/// Textual payloads retain `Bytes` ownership so the stream projector can pass
/// deltas onward without rebuilding an accumulated response string.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ScanEvent {
	/// Visible assistant text.
	Text(Bytes),
	/// Start of a model-authored reasoning region.
	ThinkingStart,
	/// Incremental reasoning text.
	ThinkingDelta(Bytes),
	/// End of the current reasoning region.
	ThinkingEnd {
		/// Provider-verifiable reasoning signature, or empty bytes when absent.
		signature: Bytes,
	},
	/// Start of one in-band tool invocation.
	ToolStart {
		/// Model-provided or scanner-minted correlation token.
		id:   SmolStr,
		/// Portable tool name.
		name: SmolStr,
	},
	/// Incremental JSON argument bytes for an active invocation.
	ToolArgumentDelta {
		/// Correlation token from the matching start event.
		id:    SmolStr,
		/// Newly accepted argument bytes only.
		delta: Bytes,
	},
	/// Completed in-band tool invocation.
	ToolEnd {
		/// Correlation token from the matching start event.
		id:        SmolStr,
		/// Portable tool name.
		name:      SmolStr,
		/// Complete, coerced UTF-8 JSON arguments.
		args_json: Bytes,
		/// Original model-authored tool envelope when the dialect exposes one.
		raw_block: Option<Bytes>,
	},
}

/// Inline event batch used by scanner feed and flush operations.
pub type ScanBatch = SmallVec<ScanEvent, 8>;

/// Borrowed result in a consecutive tool-result rendering run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct DialectToolResult<'a> {
	/// Correlation token of the matching tool invocation.
	pub id:       &'a str,
	/// Portable tool name.
	pub name:     &'a str,
	/// Position in the consecutive result run.
	pub index:    usize,
	/// Textual tool output.
	pub text:     &'a str,
	/// Whether execution failed.
	pub is_error: bool,
}

impl<'a> DialectToolResult<'a> {
	/// Creates one borrowed member of a consecutive tool-result run.
	#[must_use]
	pub const fn new(
		id: &'a str,
		name: &'a str,
		index: usize,
		text: &'a str,
		is_error: bool,
	) -> Self {
		Self { id, name, index, text, is_error }
	}
}

/// Borrowed inputs shared by dialect rendering entrypoints.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct DialectRenderOptions<'a> {
	/// Tool inventory available to prompt and transcript rendering.
	pub tools: &'a [InbandTool<'a>],
}

impl<'a> DialectRenderOptions<'a> {
	/// Creates renderer options for a borrowed tool inventory.
	#[must_use]
	pub const fn new(tools: &'a [InbandTool<'a>]) -> Self {
		Self { tools }
	}
}

/// Failure while validating or rendering owned dialect data.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DialectError {
	/// A borrowed tool's JSON Schema is not a valid schema object.
	#[error("invalid schema for dialect tool `{tool}`")]
	InvalidToolSchema {
		/// Portable tool name.
		tool:   SmolStr,
		/// JSON decoding failure.
		#[source]
		source: serde_json::Error,
	},
	/// A tool invocation's argument bytes are not valid JSON.
	#[error("invalid arguments for dialect tool `{tool}`")]
	InvalidToolArguments {
		/// Portable tool name.
		tool:   SmolStr,
		/// JSON decoding failure.
		#[source]
		source: serde_json::Error,
	},
	/// A formatter rejected rendered output.
	#[error("dialect output formatting failed")]
	Format(#[from] fmt::Error),
}

/// Result returned by fallible dialect operations.
pub type DialectResult<T> = Result<T, DialectError>;
