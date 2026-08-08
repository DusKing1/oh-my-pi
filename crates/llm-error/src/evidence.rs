//! Classification input.
//!
//! Everything is optional; classification degrades gracefully from full
//! structured evidence (status + headers + body) down to a bare persisted
//! error string, lowering [`Fidelity`](crate::Fidelity) as it does. Borrows
//! only — callers keep ownership of the response they captured.

/// Where in the request lifecycle the failure was observed.
///
/// The phase gates two things: how strict prose matching must be (a live
/// transport error may match broad truncation text, a persisted string
/// requires the stricter diagnostic table because it lost its type
/// information), and whether a same-request retry can duplicate output.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Phase {
	/// Failure before the request was admitted (credential resolve, preflight).
	Preflight,
	/// Request sent, no output token received yet. Replay-safe.
	#[default]
	BeforeFirstToken,
	/// Output already streamed; replaying may duplicate visible content or
	/// tool side effects.
	MidStream,
	/// Classifying a persisted error string long after the fact. The live
	/// error object (transport context, headers, cause chain) is gone, so
	/// low-signal prose patterns are held to a stricter table.
	Persisted,
}

/// Wire API family the request was issued against.
///
/// Sharpens API-specific classification (e.g. stale `previous_response_id`
/// recovery only exists on the `OpenAI` Responses family); never gates the
/// generic tables.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WireApi {
	/// Anthropic Messages (native or compatible endpoint).
	AnthropicMessages,
	/// `OpenAI` Chat Completions and compatible hosts.
	OpenAiCompletions,
	/// `OpenAI` Responses.
	OpenAiResponses,
	/// `OpenAI` Codex Responses (`ChatGPT` backend).
	CodexResponses,
	/// Google Generative AI / Vertex / Cloud Code Assist.
	GoogleGenerativeAi,
	/// Anything else.
	Other,
}

impl WireApi {
	/// Whether stale-`previous_response_id` recovery semantics apply.
	pub const fn is_responses(self) -> bool {
		matches!(self, Self::OpenAiResponses | Self::CodexResponses)
	}
}

/// Borrowed evidence about one failure.
///
/// Build with a constructor and chain the `with_*` setters:
///
/// ```
/// use omp_llm_error::{Evidence, Phase};
///
/// let headers = [("retry-after", "30")];
/// let ev =
/// 	Evidence::http(429, r#"{"error":{"type":"rate_limit_error","message":"Rate limited"}}"#)
/// 		.with_headers(&headers)
/// 		.with_provider("anthropic")
/// 		.with_phase(Phase::BeforeFirstToken);
/// let cls = omp_llm_error::classify(&ev);
/// assert!(cls.kinds.has(omp_llm_error::Kind::RateThrottle));
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Evidence<'a> {
	/// HTTP status, when the failure surfaced as a non-2xx response.
	pub status:   Option<u16>,
	/// Response headers as `(name, value)` pairs; names matched
	/// case-insensitively. Source of retry timing.
	pub headers:  &'a [(&'a str, &'a str)],
	/// Raw response body (or terminal SSE error frame payload).
	pub body:     Option<&'a str>,
	/// Formatted/persisted error message, when the structured response is
	/// unavailable or has already been flattened to text.
	pub message:  Option<&'a str>,
	/// Machine error code, when the caller already extracted one
	/// (e.g. from an SDK error object).
	pub code:     Option<&'a str>,
	/// Provider id hint (`"anthropic"`, `"github-copilot"`, ...). Enables
	/// provider-gated rules like Copilot fleet-skew `model_not_supported`.
	pub provider: Option<&'a str>,
	/// Wire API family hint.
	pub api:      Option<WireApi>,
	/// Lifecycle phase; see [`Phase`].
	pub phase:    Phase,
}

impl<'a> Evidence<'a> {
	/// Evidence from a non-2xx HTTP response.
	pub fn http(status: u16, body: &'a str) -> Self {
		Self { status: Some(status), body: Some(body), ..Self::default() }
	}

	/// Evidence from a bare (possibly persisted) error message.
	pub fn prose(message: &'a str) -> Self {
		Self { message: Some(message), phase: Phase::Persisted, ..Self::default() }
	}

	/// Evidence from a live error message (transport context still known).
	pub fn live(message: &'a str) -> Self {
		Self { message: Some(message), ..Self::default() }
	}

	/// Sets response headers.
	#[must_use]
	pub const fn with_headers(mut self, headers: &'a [(&'a str, &'a str)]) -> Self {
		self.headers = headers;
		self
	}

	/// Sets the formatted error message alongside structured evidence.
	#[must_use]
	pub const fn with_message(mut self, message: &'a str) -> Self {
		self.message = Some(message);
		self
	}

	/// Sets a pre-extracted machine error code.
	#[must_use]
	pub const fn with_code(mut self, code: &'a str) -> Self {
		self.code = Some(code);
		self
	}

	/// Sets the provider id hint.
	#[must_use]
	pub const fn with_provider(mut self, provider: &'a str) -> Self {
		self.provider = Some(provider);
		self
	}

	/// Sets the wire API hint.
	#[must_use]
	pub const fn with_api(mut self, api: WireApi) -> Self {
		self.api = Some(api);
		self
	}

	/// Sets the lifecycle phase.
	#[must_use]
	pub const fn with_phase(mut self, phase: Phase) -> Self {
		self.phase = phase;
		self
	}

	/// Case-insensitive header lookup.
	pub fn header(&self, name: &str) -> Option<&'a str> {
		self
			.headers
			.iter()
			.find(|(k, _)| k.eq_ignore_ascii_case(name))
			.map(|&(_, v)| v)
	}
}
