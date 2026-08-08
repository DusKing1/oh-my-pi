//! Prose pattern corpus.
//!
//! Every regex here earned its place in production: the corpus is ported
//! from a mature multi-provider agent plus a full-history mine of real
//! provider error strings. Patterns are grouped by the decision they feed;
//! compound guards (multi-regex conjunctions, status gates) live in
//! [`crate::classify`], keeping this file a flat, auditable table.

use std::sync::LazyLock;

use regex::Regex;

use crate::kind::{Kind, Kinds};

macro_rules! pattern {
	($(#[$meta:meta])* $name:ident, $re:expr) => {
		$(#[$meta])*
		#[doc = "Detection pattern."]
		#[doc = ""]
		#[doc = concat!("`", $re, "`")]
		pub static $name: LazyLock<Regex> = LazyLock::new(|| Regex::new($re).unwrap());
	};
}

// ── Context overflow ─────────────────────────────────────────────────────
// Per-provider phrasings for "prompt exceeds the context window".

/// Provider-specific and generic context-overflow phrasings.
pub static OVERFLOW: LazyLock<Vec<Regex>> = LazyLock::new(|| {
	[
		// Anthropic / Bedrock / OpenAI / Google / xAI / Groq / OpenRouter / Copilot.
		r"(?i)prompt is too long",
		r"(?i)input is too long for requested model",
		r"(?i)exceeds the context window",
		r"(?i)input token count.*exceeds the maximum",
		r"(?i)maximum prompt length is \d+",
		r"(?i)reduce the length of the messages",
		r"(?i)maximum context length is \d+ tokens",
		r"(?i)exceeds the limit of \d+",
		// llama.cpp / LM Studio / MiniMax / Kimi / local runtimes.
		r"(?i)exceeds the available context size",
		r"(?i)requested tokens?.*exceed.*context (window|length|size)",
		r"(?i)context (window|length|size).*(exceeded|overflow|too small)",
		r"(?i)(prompt|input).*(too long|too large).*(context|n_ctx)",
		r"(?i)requested tokens?.*(exceeds?|greater than).*(n_ctx|context)",
		r"(?i)greater than the context length",
		r"(?i)context window exceeds limit",
		r"(?i)exceeded model token limit",
		// Generic / body-size / gateway.
		r"(?i)context[_ ]length[_ ]exceeded",
		r"(?i)too many tokens",
		r"(?i)token limit exceeded",
		r"(?i)request_too_large",
		r"(?i)request exceeds the maximum size",
		r"(?i)payload too large",
		r"(?i)entity too large",
		r"(?i)\b413\b.*\b(request|payload|entity)\b.*\btoo large\b",
		r"(?i)model_context_window_exceeded",
		r"(?i)prompt filled the context window",
	]
	.iter()
	.map(|p| Regex::new(p).unwrap())
	.collect()
});

// Bodyless 400/413 read as overflow: a broad but empirically reliable
// compatibility heuristic for providers that drop the body on oversized
// requests.
pattern!(OVERFLOW_NO_BODY, r"(?i)\b4(00|13)\s*(status code)?\s*\(no body\)");

// ── Transient / transport / timeout ─────────────────────────────────────

pattern!(TIMEOUT, r"(?i)\b(?:operation\s+)?timed?\s*out\b|\btimeout\b|\bstream stall\b");

pattern!(
	/// Capacity / throttle / 5xx / generic-retry phrasings. Broad by design:
	/// this feeds only the same-route retry lane, never credential decisions.
	TRANSIENT_TRANSPORT,
	r"(?i)\b(?:no[_ -]?capacity|(?:high|peak)[ _-]?demand|(?:at|over|insufficient)[ _-]?capacity|capacity[ _-]?(?:exceeded|exhausted)|peak[ _-]?load)\b|overloaded|provider.?returned.?error|rate.?limit|too many requests|429|500|502|503|504|service.?unavailable|server.?error|internal.?error|retry your request|retry delay|no error details in response|malformed.?function.?call"
);

pattern!(
	/// Socket / DNS / TLS / HTTP2 transport failures.
	TRANSPORT,
	r"(?i)network.?error|connection.?error|connection.?refused|connection.?lost|unable.?to.?connect|other side closed|fetch failed|getaddrinfo|ENOTFOUND|EAI_AGAIN|ECONN(?:REFUSED|RESET)|ETIMEDOUT|EPIPE|upstream.?connect|upstream.?request.?failed|reset before headers|socket hang up|socket connection was closed|websocket.?(?:closed|error)|\bterminated\b|HTTP2(?:StreamReset|RefusedStream|EnhanceYourCalm)|INTERNAL_ERROR.*received from peer|received from peer|bad record mac|http2 request did not get a response"
);

pattern!(
	/// Model-capacity phrasings that pin the failure on the fleet, not the account.
	MODEL_CAPACITY,
	r"(?i)\bno[_ -]?capacity\b|\b(?:high|peak)[ _-]?demand\b|overloaded|capacity[ _-]?(?:exceeded|exhausted)|\b529\b"
);

// ── Auth ─────────────────────────────────────────────────────────────────

pattern!(
	AUTH_FAILURE,
	r"(?i)\b(?:401|403|unauthorized|forbidden|authentication|auth[_ ]?unavailable|no auth available|(?:invalid|no)[_ ]?api[_ ]?key)\b"
);

pattern!(
	/// Definitive OAuth death: the stored grant/client is unusable and refresh
	/// cannot revive it.
	OAUTH_DEFINITIVE,
	r"(?i)invalid_grant|invalid_token|unauthorized_client|\brevoked\b|refresh[\s_]?token.*expired|\binvalidated oauth token\b"
);

pattern!(
	/// Failures that merely look like auth trouble but are transient transport
	/// or upstream throttling — a 401 inside one of these must NOT kill a grant.
	OAUTH_TRANSIENT,
	r"(?i)timeout|network|fetch failed|ECONN(?:REFUSED|RESET)|ETIMEDOUT|EAI_AGAIN|socket hang up|\b(?:408|425|429|5\d{2})\b|rate.?limit|too many requests|temporar|unavailable|forbidden|permission_denied|cloudflare|captcha"
);

pattern!(OAUTH_HTTP_401, r"\b401\b");

// ── Content / policy / model output ──────────────────────────────────────

pattern!(
	CONTENT_FILTER,
	r"(?i)\b(?:incomplete:\s*)?content_filter\b|blocked by content filtering policy"
);
pattern!(ACCOUNT_POLICY, r"(?i)\bcyber_policy\b|trusted access for cyber");
pattern!(MALFORMED_FUNCTION_CALL, r"(?i)\bmalformed.?function.?call\b");
pattern!(
	PROVIDER_FINISH_ERROR,
	r"(?i)\bProvider (?:returned error finish_reason|finish_reason:\s*error)\b"
);
pattern!(THINKING_LOOP, r"(?i)thinking loop detected");

pattern!(
	/// llama.cpp / Ollama deterministic tool-argument JSON parse failure.
	///
	/// Surfaces as HTTP 500 but replays identically — the Deterministic guard
	/// must strip retryability or the agent loops forever.
	LLAMA_TOOL_PARSE,
	r"(?i)failed to parse tool call arguments as json|\[json\.exception\.parse_error\.101\]"
);

// ── Stale server-side session state (OpenAI Responses family) ───────────

pattern!(
	/// Machine codes for stale server-side session state. Unlike the loose
	/// English phrases below (API-gated in classify), these tokens are
	/// unambiguous and match regardless of wire API.
	STALE_SESSION_CODES,
	r"(?i)previous_response_not_found|codex_previous_response_stale"
);
pattern!(STALE_ITEM_ID, r#"(?i)\bItem with id ['"][^'"]+['"] not found\.?"#);
pattern!(STALE_PREVIOUS_RESPONSE, r"(?i)previous[ _]?response");
pattern!(
	STALE_DETAIL,
	r"(?i)not[ _]?found|invalid|expired|stale|unsupported|zero[ _-]?data[ _-]?retention"
);

// ── Request-feature rejections ───────────────────────────────────────────

pattern!(
	/// Proxy re-signing rejection of replayed thinking blocks.
	THINKING_SIGNATURE, r"(?i)invalid\s+`?signature`?\s+in\s+`?thinking`?(?:\s+block)?");

// Strict-tool / structured-output rejection components (HTTP 400/422 gated
// in classify).
pattern!(GRAMMAR_TOO_LARGE, r"(?i)compiled grammar");
pattern!(TOO_LARGE_DETAIL, r"(?i)too large");
pattern!(SCHEMA_WORD, r"(?i)schema");
pattern!(TOO_COMPLEX_DETAIL, r"(?i)too complex");
pattern!(INVALID_REQUEST_TYPE, r"(?i)invalid_request_error");
pattern!(STRUCTURED_OUTPUTS, r"(?i)structured[_ -]?outputs?");
pattern!(
	FEATURE_NOT_SUPPORTED,
	r"(?i)not (?:supported|available|enabled)|unsupported|does(?: not|n'?t) support"
);
pattern!(STRICT_FIELD, r"(?i)\btools\.\d+\.custom\.strict\b");
pattern!(EXTRA_INPUTS, r"(?i)extra inputs? (?:are|is) not permitted");
pattern!(
	STRICT_TOOLS_COMPAT,
	r"(?i)wrong_api_format|mixed values for 'strict'|tools?\b.*strict|\bstrict\b.*tool|tool parameters? schema|invalid schema for function"
);
// Optional sampling controls. The field must appear in provider-error syntax,
// not merely in prose, and classify adds the HTTP 400/422 gate.
pattern!(
	SAMPLING_PARAMETER_REJECTION,
	r#"(?ix)
		(?:
			(?:unsupported|unknown|unrecognized|invalid)\s+
			(?:request\s+)?(?:parameter|field|argument|option)s?\s*[:=]?\s*
			[`"']?(?:temperature|top_p|top_k|min_p|frequency_penalty|presence_penalty|repetition_penalty|stop(?:_sequences)?)[`"']?
		)
		|
		(?:
			[`"'](?:temperature|top_p|top_k|min_p|frequency_penalty|presence_penalty|repetition_penalty|stop(?:_sequences)?)[`"']
			[^\n]{0,160}
			(?:unsupported|not\s+(?:supported|allowed|permitted)|unknown|unrecognized|extra\s+inputs?\s+(?:are|is)\s+not\s+permitted)
		)
		|
		(?:
			(?:temperature|top_p|top_k|min_p|frequency_penalty|presence_penalty|repetition_penalty|stop(?:_sequences)?)
			\s*:\s*(?:extra\s+inputs?\s+(?:are|is)\s+not\s+permitted|unsupported|not\s+supported)
		)
		|
		(?:
			(?:(?:this|the)\s+(?:model|provider|endpoint|api)|model|provider|endpoint|api|request|it)
			\s+(?:does\s+not|doesn['’]t|cannot)\s+support\s+(?:the\s+)?
			[`"']?(?:temperature|top_p|top_k|min_p|frequency_penalty|presence_penalty|repetition_penalty|stop(?:_sequences)?)[`"']?
			(?:\s+(?:parameter|field|argument|option|control))?
		)
		|
		(?:
			(?:parameter|field|argument|option)\s+
			[`"']?(?:temperature|top_p|top_k|min_p|frequency_penalty|presence_penalty|repetition_penalty|stop(?:_sequences)?)[`"']?
			[^\n]{0,80}(?:unsupported|not\s+(?:supported|allowed|permitted)|unknown|unrecognized)
		)
	"#
);

// Anthropic fast-mode (`speed` param) rejection components.
pattern!(FAST_MODE_SPEED, r"(?i)\bspeed\b");
pattern!(FAST_MODE_NOT_SUPPORTED, r"(?i)not support");
pattern!(FAST_MODE_RATE_LIMIT, r"(?i)rate_limit_error");
pattern!(FAST_MODE_ENTITLEMENT, r"(?i)fast mode");

// tool_choice unsupported (providers that only accept "auto").
pattern!(TOOL_CHOICE_WORD, r"(?i)\btool_choice\b");
pattern!(TOOL_CHOICE_AUTO, r"(?i)\bauto\b");
pattern!(TOOL_CHOICE_SUPPORTED, r"(?i)\bsupported\b");

// Unsupported reasoning-effort components.
pattern!(REASONING_FIELD, r"(?i)reasoning[_. ]effort|reasoning value");
pattern!(
	REASONING_REJECTED,
	r"(?i)invalid[^\n]*(?:reasoning[_. ]effort|reasoning value)|(?:reasoning[_. ]effort|reasoning value)[^\n]*(?:invalid|unsupported|not supported|must be|expected)|(?:unsupported|not supported)[^\n]*(?:reasoning[_. ]effort|reasoning value)"
);
pattern!(REASONING_ALLOWED_CUE, r"(?i)must be|one of|allowed values?|supported values?|expected");
pattern!(REASONING_ALLOWED_VALUE, r#"(?i)["'`](none|minimal|low|medium|high|xhigh|max)["'`]"#);

// ── Invalid model ─────────────────────────────────────────────────────────

pattern!(
	/// Model-scoped rejection. Requires a model-adjacent token — a bare `404`
	/// or "not found" in unrelated prose must not condemn a model id.
	INVALID_MODEL,
	r"(?i)model[_ ]?not[_ ]?found|model_not_available|invalid[_ -]model|model[_ -]is[_ -]not[_ -]valid|models?/[\w.-]+ is not found|\bmodel\b[^\n]{0,60}\b(?:does not exist|no longer supported|deprecated|decommissioned|not supported when)|(?:does not exist|no longer supported|deprecated|decommissioned)[^\n]{0,60}\bmodel\b"
);

pattern!(
	/// Copilot fleet skew: a 400 `model_not_supported` from a stale replica for
	/// a model `/models` advertises.
	///
	/// Transient, retried on a short flat delay —
	/// but ONLY for Copilot; elsewhere the same code is a stable entitlement
	/// denial. `model_not_available_for_integrator` is deliberately excluded:
	/// GitHub also uses it for durable per-integrator denials.
	COPILOT_TRANSIENT_MODEL, r"(?i)model_not_supported");

// ── Stream corruption ─────────────────────────────────────────────────────

pattern!(
	/// Live truncation matcher: safe only while the error still carries its
	/// transport context.
	STREAM_PARSE_LIVE,
	r"(?i)unterminated string|unexpected end of json input|unexpected end of data|unexpected eof|end of file|eof while parsing|truncated"
);

pattern!(
	/// Persisted-string matcher: stricter, because bare "truncated" / "end of
	/// file" is too low-signal once detached from the transport error.
	STREAM_PARSE_PERSISTED,
	r"(?i)json parse error:\s*(?:unterminated string|unexpected end of json input|unexpected end of data|unexpected eof|end of file|eof while parsing|truncated|invalid escape)|json\.parse:\s*(?:unterminated string|unexpected end of data)|unexpected end of json input|unexpected eof|eof while parsing"
);

pattern!(
	/// Stream event-order violations — retry-safe, unlike arbitrary corruption.
	STREAM_EVENT_ORDER, r"(?i)stream event order|before message_start");

pattern!(
	/// Stream ended without its terminal protocol event.
	STREAM_INCOMPLETE,
	r"(?i)ended without|stream ended before|stream closed before response\.completed|stream closed with reason|incomplete[_ -]stream|stream[_ -]?read[_ -]?error"
);

// ── Abort sentinels ──────────────────────────────────────────────────────

pattern!(ABORTED, r"(?i)\baborted\b|\babort signal\b|\binterrupted by user\b");

/// Simple `pattern → kinds` rules applied to every evidence text.
/// Compound guards (status gates, conjunctions, api/provider gates) live in
/// `classify` — this table is only the unconditional prose layer.
pub static TEXT_RULES: LazyLock<Vec<(&'static Regex, Kinds)>> = LazyLock::new(|| {
	vec![
		(&*TIMEOUT, Kind::Timeout | Kind::Transient),
		(&*TRANSIENT_TRANSPORT, Kinds::only(Kind::Transient)),
		(&*TRANSPORT, Kind::Transport | Kind::Transient),
		(&*MODEL_CAPACITY, Kind::ModelCapacity | Kind::Transient),
		(&*OVERFLOW_NO_BODY, Kinds::only(Kind::ContextOverflow)),
		(&*AUTH_FAILURE, Kinds::only(Kind::AuthFailed)),
		(&*CONTENT_FILTER, Kinds::only(Kind::ContentBlocked)),
		(&*ACCOUNT_POLICY, Kind::AccountPolicy | Kind::ContentBlocked),
		(&*MALFORMED_FUNCTION_CALL, Kind::MalformedToolCall | Kind::Transient),
		(&*PROVIDER_FINISH_ERROR, Kind::ProviderFinish | Kind::Transient),
		(&*THINKING_LOOP, Kinds::only(Kind::ThinkingLoop)),
		(&*LLAMA_TOOL_PARSE, Kind::MalformedToolCall | Kind::Deterministic),
		(&*THINKING_SIGNATURE, Kind::InvalidRequest | Kind::FeatureUnsupported),
		(&*STREAM_EVENT_ORDER, Kind::StreamOrder | Kind::StreamCorruption | Kind::Transient),
		(&*STREAM_INCOMPLETE, Kind::StreamCorruption | Kind::Transient),
		(&*ABORTED, Kinds::only(Kind::Aborted)),
	]
});
