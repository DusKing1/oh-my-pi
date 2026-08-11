//! Stable value vocabularies and span-name formatting for pi telemetry.
//!
//! Values in this module are wire contracts shared with the TypeScript pi
//! implementation. They intentionally preserve pi's exact spelling rather
//! than tracking whichever semantic-convention crate version is installed.

use std::str::FromStr;

/// Default OpenTelemetry tracer name used by the agent loop.
pub const TRACER_NAME: &str = "@omp/agent-core";
/// OpenTelemetry meter name used by the coding-agent exporter.
pub const METER_NAME: &str = "@omp/coding-agent";

/// Error returned when a string is not part of a bounded telemetry vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("invalid {0} telemetry vocabulary value")]
pub struct ParseSemconvError(&'static str);

/// Values of `gen_ai.operation.name` emitted by agent-loop spans.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Operation {
	/// A complete agent-loop invocation.
	InvokeAgent,
	/// One model chat request.
	Chat,
	/// One tool execution.
	ExecuteTool,
	/// A transition between named agents.
	Handoff,
}

impl Operation {
	/// Returns the byte-exact `gen_ai.operation.name` value.
	#[must_use]
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::InvokeAgent => "invoke_agent",
			Self::Chat => "chat",
			Self::ExecuteTool => "execute_tool",
			Self::Handoff => "handoff",
		}
	}

	/// Formats the span name exactly as pi does.
	///
	/// `primary` is the agent name for `invoke_agent`, model identifier for
	/// `chat`, tool name for `execute_tool`, and source agent name for
	/// `handoff`. `secondary` is ignored except for `handoff`, where it is the
	/// destination agent name. A handoff with only a destination is named
	/// `handoff to {destination}`; with both names it uses a literal U+2192
	/// right arrow: `handoff {source} → {destination}`.
	#[must_use]
	pub fn span_name(self, primary: Option<&str>, secondary: Option<&str>) -> String {
		match self {
			Self::InvokeAgent => {
				format_subject("invoke_agent", primary.filter(|name| !name.is_empty()))
			},
			Self::Chat => format_subject("chat", primary),
			Self::ExecuteTool => format_subject("execute_tool", primary),
			Self::Handoff => {
				let from = primary.filter(|name| !name.is_empty());
				let to = secondary.filter(|name| !name.is_empty());
				match (from, to) {
					(Some(from), Some(to)) => format!("handoff {from} → {to}"),
					(_, Some(to)) => format!("handoff to {to}"),
					_ => "handoff".to_owned(),
				}
			},
		}
	}
}

impl FromStr for Operation {
	type Err = ParseSemconvError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		match value {
			"invoke_agent" => Ok(Self::InvokeAgent),
			"chat" => Ok(Self::Chat),
			"execute_tool" => Ok(Self::ExecuteTool),
			"handoff" => Ok(Self::Handoff),
			_ => Err(ParseSemconvError("operation")),
		}
	}
}

/// Terminal status vocabulary for tool execution.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ToolStatus {
	/// The tool completed successfully.
	Ok,
	/// The tool failed.
	Error,
	/// The tool was intentionally skipped.
	Skipped,
	/// Policy or permissions blocked the tool.
	Blocked,
	/// The tool exceeded its deadline.
	Timeout,
	/// The tool was aborted.
	Aborted,
}

impl ToolStatus {
	/// Returns the byte-exact terminal-status value.
	#[must_use]
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Ok => "ok",
			Self::Error => "error",
			Self::Skipped => "skipped",
			Self::Blocked => "blocked",
			Self::Timeout => "timeout",
			Self::Aborted => "aborted",
		}
	}
}

impl FromStr for ToolStatus {
	type Err = ParseSemconvError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		match value {
			"ok" => Ok(Self::Ok),
			"error" => Ok(Self::Error),
			"skipped" => Ok(Self::Skipped),
			"blocked" => Ok(Self::Blocked),
			"timeout" => Ok(Self::Timeout),
			"aborted" => Ok(Self::Aborted),
			_ => Err(ParseSemconvError("tool status")),
		}
	}
}

/// Values of the `gen_ai.token.type` metric dimension.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TokenType {
	/// All input tokens, including cache buckets.
	Input,
	/// Output tokens.
	Output,
	/// Input plus output tokens.
	Total,
	/// Cache-read input tokens.
	CacheReadInput,
	/// Cache-write input tokens.
	CacheWriteInput,
	/// Reasoning output tokens.
	ReasoningOutput,
}

impl TokenType {
	/// Returns the byte-exact `gen_ai.token.type` value.
	#[must_use]
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Input => "input",
			Self::Output => "output",
			Self::Total => "total",
			Self::CacheReadInput => "cache_read_input",
			Self::CacheWriteInput => "cache_write_input",
			Self::ReasoningOutput => "reasoning_output",
		}
	}
}

impl FromStr for TokenType {
	type Err = ParseSemconvError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		match value {
			"input" => Ok(Self::Input),
			"output" => Ok(Self::Output),
			"total" => Ok(Self::Total),
			"cache_read_input" => Ok(Self::CacheReadInput),
			"cache_write_input" => Ok(Self::CacheWriteInput),
			"reasoning_output" => Ok(Self::ReasoningOutput),
			_ => Err(ParseSemconvError("token type")),
		}
	}
}

/// Message-content capture mode.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CaptureMode {
	/// Do not attach message content.
	#[default]
	None,
	/// Attach bounded dashboard-friendly summaries.
	Summary,
	/// Attach summaries and full OpenTelemetry message payloads.
	Full,
}

impl CaptureMode {
	/// Returns the byte-exact configuration value.
	#[must_use]
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::None => "none",
			Self::Summary => "summary",
			Self::Full => "full",
		}
	}

	/// Parses `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` exactly as
	/// pi.
	///
	/// Missing, empty, and unrecognized values disable capture. Matching is
	/// ASCII-case-insensitive after trimming. `true`, `1`, `yes`, and `full`
	/// select full capture; `summary` selects summary capture.
	#[must_use]
	pub fn from_env_value(value: Option<&str>) -> Self {
		let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
			return Self::None;
		};
		if value.eq_ignore_ascii_case("summary") {
			Self::Summary
		} else if value.eq_ignore_ascii_case("true")
			|| value == "1"
			|| value.eq_ignore_ascii_case("yes")
			|| value.eq_ignore_ascii_case("full")
		{
			Self::Full
		} else {
			Self::None
		}
	}
}

impl FromStr for CaptureMode {
	type Err = ParseSemconvError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		match value {
			"none" => Ok(Self::None),
			"summary" => Ok(Self::Summary),
			"full" => Ok(Self::Full),
			_ => Err(ParseSemconvError("content-capture mode")),
		}
	}
}

/// Stop-reason vocabulary received from pi providers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StopReason {
	/// The model stopped normally.
	Stop,
	/// The model reached a length limit.
	Length,
	/// The model requested one or more tool calls.
	ToolUse,
	/// The provider reported an error.
	Error,
	/// The request was aborted.
	Aborted,
}

impl StopReason {
	/// Returns the byte-exact pi stop-reason value.
	#[must_use]
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Stop => "stop",
			Self::Length => "length",
			Self::ToolUse => "toolUse",
			Self::Error => "error",
			Self::Aborted => "aborted",
		}
	}

	/// Maps this pi stop reason to the finish reason emitted on chat spans.
	#[must_use]
	pub const fn finish_reason(self) -> FinishReason {
		match self {
			Self::Stop => FinishReason::Stop,
			Self::Length => FinishReason::Length,
			Self::ToolUse => FinishReason::ToolCalls,
			Self::Error | Self::Aborted => FinishReason::Error,
		}
	}
}

impl FromStr for StopReason {
	type Err = ParseSemconvError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		match value {
			"stop" => Ok(Self::Stop),
			"length" => Ok(Self::Length),
			"toolUse" => Ok(Self::ToolUse),
			"error" => Ok(Self::Error),
			"aborted" => Ok(Self::Aborted),
			_ => Err(ParseSemconvError("stop reason")),
		}
	}
}

/// Normalized values emitted in `gen_ai.response.finish_reasons`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FinishReason {
	/// Normal completion (`stop`).
	Stop,
	/// Length-limited completion (`length`).
	Length,
	/// Completion requesting tools (`tool_calls`).
	ToolCalls,
	/// Errored or aborted completion (`error`).
	Error,
}

impl FinishReason {
	/// Returns the byte-exact finish-reason value.
	#[must_use]
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Stop => "stop",
			Self::Length => "length",
			Self::ToolCalls => "tool_calls",
			Self::Error => "error",
		}
	}
}

impl FromStr for FinishReason {
	type Err = ParseSemconvError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		match value {
			"stop" => Ok(Self::Stop),
			"length" => Ok(Self::Length),
			"tool_calls" => Ok(Self::ToolCalls),
			"error" => Ok(Self::Error),
			_ => Err(ParseSemconvError("finish reason")),
		}
	}
}

/// Maps a raw pi stop reason to the span's normalized finish-reason value.
///
/// Unknown values return `None`, matching pi's switch default.
#[must_use]
pub fn map_stop_reason(reason: &str) -> Option<&'static str> {
	reason
		.parse::<StopReason>()
		.ok()
		.map(StopReason::finish_reason)
		.map(FinishReason::as_str)
}

/// Fixed `error.type` classifications emitted by pi.
///
/// JavaScript error names and caller-supplied error types remain free-form and
/// therefore are not represented by this bounded vocabulary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ErrorType {
	/// A chat ended with stop reason `error`.
	TerminalError,
	/// A chat ended with stop reason `aborted`.
	TerminalAborted,
	/// A tool failed without a more specific thrown-error class.
	ToolError,
	/// A tool was skipped.
	ToolSkipped,
	/// A tool was blocked.
	ToolBlocked,
	/// A tool timed out.
	ToolTimeout,
	/// A tool was aborted.
	ToolAborted,
	/// Fallback classification when no JavaScript error class is available.
	FallbackError,
}

impl ErrorType {
	/// Returns the byte-exact `error.type` value.
	#[must_use]
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::TerminalError => "error",
			Self::TerminalAborted => "aborted",
			Self::ToolError => "tool_error",
			Self::ToolSkipped => "tool_skipped",
			Self::ToolBlocked => "tool_blocked",
			Self::ToolTimeout => "tool_timeout",
			Self::ToolAborted => "tool_aborted",
			Self::FallbackError => "Error",
		}
	}

	/// Returns pi's fixed error classification for a non-success tool status.
	///
	/// Successful tools have no `error.type`. A thrown error may replace the
	/// `ToolError` value with its free-form JavaScript error class name.
	#[must_use]
	pub const fn for_tool_status(status: ToolStatus) -> Option<Self> {
		match status {
			ToolStatus::Ok => None,
			ToolStatus::Error => Some(Self::ToolError),
			ToolStatus::Skipped => Some(Self::ToolSkipped),
			ToolStatus::Blocked => Some(Self::ToolBlocked),
			ToolStatus::Timeout => Some(Self::ToolTimeout),
			ToolStatus::Aborted => Some(Self::ToolAborted),
		}
	}
}

impl FromStr for ErrorType {
	type Err = ParseSemconvError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		match value {
			"error" => Ok(Self::TerminalError),
			"aborted" => Ok(Self::TerminalAborted),
			"tool_error" => Ok(Self::ToolError),
			"tool_skipped" => Ok(Self::ToolSkipped),
			"tool_blocked" => Ok(Self::ToolBlocked),
			"tool_timeout" => Ok(Self::ToolTimeout),
			"tool_aborted" => Ok(Self::ToolAborted),
			"Error" => Ok(Self::FallbackError),
			_ => Err(ParseSemconvError("error type")),
		}
	}
}

/// Normalizes a pi provider identifier for `gen_ai.provider.name`.
///
/// The match table is transcribed exactly from pi. Unknown and empty provider
/// identifiers pass through unchanged.
#[must_use]
pub fn normalize_provider(raw: &str) -> &str {
	match raw {
		"amazon-bedrock" => "aws.bedrock",
		"google" | "google-antigravity" | "google-gemini-cli" => "gcp.gemini",
		"google-vertex" => "gcp.vertex_ai",
		"mistral" => "mistral_ai",
		"openai-codex" => "openai",
		"xai" => "x_ai",
		_ => raw,
	}
}

fn format_subject(operation: &str, subject: Option<&str>) -> String {
	let Some(subject) = subject else {
		return operation.to_owned();
	};
	let mut name = String::with_capacity(operation.len() + 1 + subject.len());
	name.push_str(operation);
	name.push(' ');
	name.push_str(subject);
	name
}

#[cfg(test)]
mod tests {
	use super::*;

	fn assert_round_trip<T>(values: &[T])
	where
		T: Copy + std::fmt::Debug + Eq + FromStr<Err = ParseSemconvError>,
		T: AsWireStr,
	{
		for &value in values {
			assert_eq!(value.as_wire_str().parse::<T>(), Ok(value));
		}
	}

	trait AsWireStr {
		fn as_wire_str(&self) -> &'static str;
	}

	impl AsWireStr for Operation {
		fn as_wire_str(&self) -> &'static str {
			self.as_str()
		}
	}

	impl AsWireStr for ToolStatus {
		fn as_wire_str(&self) -> &'static str {
			self.as_str()
		}
	}

	impl AsWireStr for TokenType {
		fn as_wire_str(&self) -> &'static str {
			self.as_str()
		}
	}

	impl AsWireStr for CaptureMode {
		fn as_wire_str(&self) -> &'static str {
			self.as_str()
		}
	}

	impl AsWireStr for StopReason {
		fn as_wire_str(&self) -> &'static str {
			self.as_str()
		}
	}

	impl AsWireStr for FinishReason {
		fn as_wire_str(&self) -> &'static str {
			self.as_str()
		}
	}

	impl AsWireStr for ErrorType {
		fn as_wire_str(&self) -> &'static str {
			self.as_str()
		}
	}

	#[test]
	fn enum_values_round_trip() {
		assert_round_trip(&[
			Operation::InvokeAgent,
			Operation::Chat,
			Operation::ExecuteTool,
			Operation::Handoff,
		]);
		assert_round_trip(&[
			ToolStatus::Ok,
			ToolStatus::Error,
			ToolStatus::Skipped,
			ToolStatus::Blocked,
			ToolStatus::Timeout,
			ToolStatus::Aborted,
		]);
		assert_round_trip(&[
			TokenType::Input,
			TokenType::Output,
			TokenType::Total,
			TokenType::CacheReadInput,
			TokenType::CacheWriteInput,
			TokenType::ReasoningOutput,
		]);
		assert_round_trip(&[CaptureMode::None, CaptureMode::Summary, CaptureMode::Full]);
		assert_round_trip(&[
			StopReason::Stop,
			StopReason::Length,
			StopReason::ToolUse,
			StopReason::Error,
			StopReason::Aborted,
		]);
		assert_round_trip(&[
			FinishReason::Stop,
			FinishReason::Length,
			FinishReason::ToolCalls,
			FinishReason::Error,
		]);
		assert_round_trip(&[
			ErrorType::TerminalError,
			ErrorType::TerminalAborted,
			ErrorType::ToolError,
			ErrorType::ToolSkipped,
			ErrorType::ToolBlocked,
			ErrorType::ToolTimeout,
			ErrorType::ToolAborted,
			ErrorType::FallbackError,
		]);
	}

	#[test]
	fn formats_exact_span_names() {
		assert_eq!(Operation::InvokeAgent.span_name(Some("planner"), None), "invoke_agent planner");
		assert_eq!(Operation::InvokeAgent.span_name(None, None), "invoke_agent");
		assert_eq!(Operation::InvokeAgent.span_name(Some(""), None), "invoke_agent");
		assert_eq!(Operation::Chat.span_name(Some("gpt-5"), None), "chat gpt-5");
		assert_eq!(Operation::ExecuteTool.span_name(Some("read"), None), "execute_tool read");
		assert_eq!(Operation::Handoff.span_name(Some("plan"), Some("build")), "handoff plan → build");
		assert_eq!(Operation::Handoff.span_name(None, Some("build")), "handoff to build");
		assert_eq!(Operation::Handoff.span_name(Some("plan"), None), "handoff");
		assert_eq!(Operation::Handoff.span_name(Some(""), Some("build")), "handoff to build");
		assert_eq!(Operation::Handoff.span_name(Some("plan"), Some("")), "handoff");
	}

	#[test]
	fn parses_content_capture_environment_values() {
		assert_eq!(CaptureMode::from_env_value(None), CaptureMode::None);
		assert_eq!(CaptureMode::from_env_value(Some("")), CaptureMode::None);
		assert_eq!(CaptureMode::from_env_value(Some(" summary ")), CaptureMode::Summary);
		for raw in ["true", "TRUE", "1", "yes", "Yes", "full", "FULL"] {
			assert_eq!(CaptureMode::from_env_value(Some(raw)), CaptureMode::Full);
		}
		assert_eq!(CaptureMode::from_env_value(Some("on")), CaptureMode::None);
	}

	#[test]
	fn maps_every_stop_reason() {
		assert_eq!(map_stop_reason("stop"), Some("stop"));
		assert_eq!(map_stop_reason("length"), Some("length"));
		assert_eq!(map_stop_reason("toolUse"), Some("tool_calls"));
		assert_eq!(map_stop_reason("error"), Some("error"));
		assert_eq!(map_stop_reason("aborted"), Some("error"));
		assert_eq!(map_stop_reason("unknown"), None);
	}

	#[test]
	fn normalizes_every_provider_mapping() {
		for (raw, normalized) in [
			("amazon-bedrock", "aws.bedrock"),
			("google", "gcp.gemini"),
			("google-antigravity", "gcp.gemini"),
			("google-gemini-cli", "gcp.gemini"),
			("google-vertex", "gcp.vertex_ai"),
			("mistral", "mistral_ai"),
			("openai-codex", "openai"),
			("xai", "x_ai"),
		] {
			assert_eq!(normalize_provider(raw), normalized);
		}
		assert_eq!(normalize_provider("anthropic"), "anthropic");
		assert_eq!(normalize_provider(""), "");
	}
}
