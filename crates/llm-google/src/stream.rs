//! Gemini stream terminal, metadata, and semantic retry rules.

use omp_core::Str;
use omp_llm_types::{Props, StopReason, TurnError, TurnErrorKind};
use serde::Deserialize;
use serde_json::{Value, json};

/// Maximum re-dispatches after a Gemini stream ends successfully but empty.
pub const MAX_EMPTY_STREAM_RETRIES: u8 = 2;
/// Maximum re-dispatches after a non-streaming Gemini response is successfully
/// empty.
pub const MAX_EMPTY_RESPONSE_RETRIES: u8 = 2;
/// Maximum identical-request retries for an in-band Gemini overload.
pub const MAX_OVERLOAD_RETRIES: u8 = 2;

const EMPTY_BASE_DELAY_MS: u64 = 500;
const OVERLOAD_BASE_DELAY_MS: u64 = 1_000;
const MAX_OVERLOAD_DELAY_MS: u64 = 8_000;

/// Decision produced by [`SemanticRetryBudget`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryDecision {
	/// Repeat the same request after the bounded delay.
	RetryAfter(u64),
	/// The provider-specific retry budget is exhausted.
	Terminal,
}

/// Per-turn Gemini retry counters. The owner must drop the preceding response
/// body before retrying.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SemanticRetryBudget {
	empty_stream:   u8,
	empty_response: u8,
	overload:       u8,
	cancelled:      bool,
}

impl SemanticRetryBudget {
	/// Records an empty streaming success and returns the next bounded action.
	pub fn empty_stream(&mut self) -> RetryDecision {
		if self.cancelled {
			return RetryDecision::Terminal;
		}
		bounded(&mut self.empty_stream, MAX_EMPTY_STREAM_RETRIES, EMPTY_BASE_DELAY_MS, u64::MAX)
	}

	/// Records an empty non-streaming success and returns the next bounded
	/// action.
	pub fn empty_response(&mut self) -> RetryDecision {
		if self.cancelled {
			return RetryDecision::Terminal;
		}
		bounded(&mut self.empty_response, MAX_EMPTY_RESPONSE_RETRIES, EMPTY_BASE_DELAY_MS, u64::MAX)
	}

	/// Records a transient overload and returns the next bounded action.
	pub fn overload(&mut self) -> RetryDecision {
		if self.cancelled {
			return RetryDecision::Terminal;
		}
		bounded(
			&mut self.overload,
			MAX_OVERLOAD_RETRIES,
			OVERLOAD_BASE_DELAY_MS,
			MAX_OVERLOAD_DELAY_MS,
		)
	}

	/// Permanently disables retries after cancellation; the owner must then drop
	/// the upstream body.
	pub const fn cancel(&mut self) {
		self.cancelled = true;
	}
}

fn bounded(counter: &mut u8, maximum: u8, base: u64, ceiling: u64) -> RetryDecision {
	if *counter >= maximum {
		return RetryDecision::Terminal;
	}
	let shift = u32::from(*counter).min(63);
	let delay = base.saturating_mul(1_u64 << shift).min(ceiling);
	*counter += 1;
	RetryDecision::RetryAfter(delay)
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CandidateMetadata {
	pub(crate) grounding_metadata: Option<Value>,
	pub(crate) citation_metadata:  Option<Value>,
	pub(crate) safety_ratings:     Option<Value>,
	pub(crate) finish_message:     Option<Str>,
}

pub(crate) fn retain_candidate_metadata(props: &mut Props, metadata: CandidateMetadata) {
	if let Some(value) = metadata.grounding_metadata {
		props.insert_ns("google", "grounding_metadata", value);
	}
	if let Some(value) = metadata.citation_metadata {
		props.insert_ns("google", "citation_metadata", value);
	}
	if let Some(value) = metadata.safety_ratings {
		props.insert_ns("google", "safety_ratings", value);
	}
	if let Some(value) = metadata.finish_message {
		props.insert_ns("google", "finish_message", json!(value));
	}
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ExecutableCode {
	pub(crate) language: Option<Str>,
	pub(crate) code:     Option<Str>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct CodeExecutionResult {
	pub(crate) outcome: Option<Str>,
	pub(crate) output:  Option<Str>,
}

pub(crate) struct AuxiliaryText {
	pub(crate) text:  Str,
	pub(crate) kind:  &'static str,
	pub(crate) props: Value,
}

pub(crate) fn executable_code(value: ExecutableCode) -> Option<AuxiliaryText> {
	let code = value.code.filter(|code| !code.is_empty())?;
	let language = value
		.language
		.unwrap_or_else(|| "LANGUAGE_UNSPECIFIED".into());
	Some(AuxiliaryText {
		text:  code,
		kind:  "executable_code",
		props: json!({ "language": language }),
	})
}

pub(crate) fn code_execution_result(value: CodeExecutionResult) -> Option<AuxiliaryText> {
	let output = value.output.unwrap_or_default();
	let outcome = value
		.outcome
		.unwrap_or_else(|| "OUTCOME_UNSPECIFIED".into());
	if output.is_empty() && outcome == "OUTCOME_UNSPECIFIED" {
		return None;
	}
	Some(AuxiliaryText {
		text:  output,
		kind:  "code_execution_result",
		props: json!({ "outcome": outcome }),
	})
}

pub(crate) fn finish_reason(reason: &str) -> Result<StopReason, Str> {
	match reason {
		"STOP" => Ok(StopReason::EndTurn),
		"MAX_TOKENS" => Ok(StopReason::MaxTokens),
		"SAFETY"
		| "RECITATION"
		| "BLOCKLIST"
		| "PROHIBITED_CONTENT"
		| "SPII"
		| "IMAGE_SAFETY"
		| "IMAGE_PROHIBITED_CONTENT"
		| "IMAGE_RECITATION"
		| "IMAGE_OTHER"
		| "NO_IMAGE" => Ok(StopReason::ContentFilter),
		"FINISH_REASON_UNSPECIFIED"
		| "OTHER"
		| "LANGUAGE"
		| "MALFORMED_FUNCTION_CALL"
		| "UNEXPECTED_TOOL_CALL" => {
			Err(Str::from(format!("Google generation failed with finish reason: {reason}")))
		},
		other => Err(Str::from(format!("unknown Google finish reason: {other}"))),
	}
}

pub(crate) fn stream_error(
	code: Option<u16>,
	status: Option<&str>,
	message: Option<&str>,
) -> TurnError {
	let detail = message.or(status).unwrap_or("Google stream error");
	let status = status.unwrap_or_default();
	let kind = if code == Some(429) || status == "RESOURCE_EXHAUSTED" {
		TurnErrorKind::RateLimited
	} else if matches!(code, Some(500 | 502 | 503 | 504))
		|| matches!(status, "UNAVAILABLE" | "INTERNAL" | "ABORTED")
	{
		TurnErrorKind::Overloaded
	} else {
		TurnErrorKind::Upstream
	};
	let retry_after_ms = if matches!(kind, TurnErrorKind::RateLimited | TurnErrorKind::Overloaded) {
		OVERLOAD_BASE_DELAY_MS
	} else {
		0
	};
	TurnError::builder()
		.kind(kind)
		.detail(Str::from(detail))
		.unsupported(Vec::new())
		.retry_after_ms(retry_after_ms)
		.build()
}

pub(crate) fn incomplete_stream_error() -> TurnError {
	TurnError::builder()
		.kind(TurnErrorKind::Upstream)
		.detail("Google stream ended without a finish reason".into())
		.unsupported(Vec::new())
		.retry_after_ms(0)
		.build()
}

#[cfg(test)]
mod tests {
	use omp_llm_types::{StopReason, TurnErrorKind};

	use super::{RetryDecision, SemanticRetryBudget, finish_reason, stream_error};

	#[test]
	fn two_empty_retries_then_terminal() {
		let mut budget = SemanticRetryBudget::default();
		assert_eq!(budget.empty_stream(), RetryDecision::RetryAfter(500));
		assert_eq!(budget.empty_stream(), RetryDecision::RetryAfter(1_000));
		assert_eq!(budget.empty_stream(), RetryDecision::Terminal);
	}

	#[test]
	fn two_empty_response_retries_then_terminal() {
		let mut budget = SemanticRetryBudget::default();
		assert_eq!(budget.empty_response(), RetryDecision::RetryAfter(500));
		assert_eq!(budget.empty_response(), RetryDecision::RetryAfter(1_000));
		assert_eq!(budget.empty_response(), RetryDecision::Terminal);
	}

	#[test]
	fn overload_recovery_is_bounded() {
		let mut budget = SemanticRetryBudget::default();
		assert_eq!(budget.overload(), RetryDecision::RetryAfter(1_000));
		assert_eq!(budget.overload(), RetryDecision::RetryAfter(2_000));
		assert_eq!(budget.overload(), RetryDecision::Terminal);
	}

	#[test]
	fn cancellation_disables_every_retry_lane() {
		let mut budget = SemanticRetryBudget::default();
		budget.cancel();
		assert_eq!(budget.empty_stream(), RetryDecision::Terminal);
		assert_eq!(budget.empty_response(), RetryDecision::Terminal);
		assert_eq!(budget.overload(), RetryDecision::Terminal);
	}

	#[test]
	fn overload_and_context_overflow_are_classifiable() {
		let overloaded = stream_error(Some(503), Some("UNAVAILABLE"), Some("high demand"));
		assert_eq!(overloaded.kind, TurnErrorKind::Overloaded);
		assert_eq!(overloaded.retry_after_ms, 1_000);

		let overflow = stream_error(
			Some(400),
			Some("INVALID_ARGUMENT"),
			Some("input token count 1048577 exceeds the maximum 1048576"),
		);
		assert_eq!(overflow.kind, TurnErrorKind::Upstream);
		assert!(overflow.detail.contains("input token count"));
	}

	#[test]
	fn every_documented_finish_reason_is_explicit() {
		assert_eq!(finish_reason("STOP"), Ok(StopReason::EndTurn));
		assert_eq!(finish_reason("MAX_TOKENS"), Ok(StopReason::MaxTokens));
		assert_eq!(finish_reason("IMAGE_RECITATION"), Ok(StopReason::ContentFilter));
		assert!(finish_reason("MALFORMED_FUNCTION_CALL").is_err());
		assert!(finish_reason("NEW_REASON").is_err());
	}

	#[test]
	fn recorded_retry_fixture_drives_policy_and_error_classification() {
		let fixture: serde_json::Value = serde_json::from_str(include_str!(
			"../tests/fixtures/google_genai/stream.retry_cases.json"
		))
		.unwrap();
		let expected = fixture["empty_stream"]["retry_delays_ms"]
			.as_array()
			.unwrap()
			.iter()
			.map(|value| value.as_u64().unwrap())
			.collect::<Vec<_>>();
		let mut budget = SemanticRetryBudget::default();
		let actual = [budget.empty_stream(), budget.empty_stream()]
			.into_iter()
			.map(|decision| match decision {
				RetryDecision::RetryAfter(delay) => delay,
				RetryDecision::Terminal => panic!("fixture expects a retry"),
			})
			.collect::<Vec<_>>();
		assert_eq!(actual, expected);
		assert_eq!(budget.empty_stream(), RetryDecision::Terminal);

		let overload = &fixture["overload_recovery"]["first"]["error"];
		let classified = stream_error(
			overload["code"].as_u64().map(|code| code as u16),
			overload["status"].as_str(),
			overload["message"].as_str(),
		);
		assert_eq!(classified.kind, TurnErrorKind::Overloaded);

		let overflow = &fixture["context_overflow"]["error"];
		let classified = stream_error(
			overflow["code"].as_u64().map(|code| code as u16),
			overflow["status"].as_str(),
			overflow["message"].as_str(),
		);
		assert!(classified.detail.contains("input token count"));
	}
}
