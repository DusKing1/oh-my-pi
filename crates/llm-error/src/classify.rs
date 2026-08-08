//! Classification orchestrator.
//!
//! Precedence: structural evidence (machine codes, gRPC status words, HTTP
//! status) is applied first, then the prose corpus refines and fills gaps.
//! Several sources merge into one [`Kinds`] set — one error can carry
//! several truths, and downstream predicates pick the axis they care about.

use omp_core::SmolStr;
use smallvec::SmallVec;

use crate::{
	envelope::{self, Envelope},
	evidence::{Evidence, Phase},
	extract::{self, RetryHint},
	kind::{Kind, Kinds},
	oauth::{self, OAuthFailure},
	patterns,
	rate_limit::{self, RateLimit, RateLimitReason},
};

/// How trustworthy the evidence behind a classification is.
///
/// Destructive consumer actions (deleting a stored credential, condemning a
/// model id) should require [`Fidelity::Structured`]; prose-only verdicts
/// are for retry/backoff decisions.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Fidelity {
	/// Classified from structured evidence: HTTP status, parsed envelope,
	/// or a caller-supplied machine code.
	Structured,
	/// Classified from a flattened error string only.
	#[default]
	Prose,
}

/// Request feature a provider rejected; the recovery is to strip or mutate
/// exactly this feature and retry, not to treat the request as poisoned.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Feature {
	/// `strict: true` tool schemas (compiled grammar too large / schema too
	/// complex / strict flag rejected).
	StrictTools,
	/// Structured outputs unsupported by the hosted model.
	StructuredOutputs,
	/// `reasoning.effort` value rejected; see
	/// [`Classification::allowed_efforts`] for the provider's stated menu.
	ReasoningEffort,
	/// `tool_choice` other than `"auto"` unsupported.
	ToolChoice,
	/// Optional sampling controls rejected by the provider (`temperature`,
	/// nucleus/top-k/min-p sampling, penalties, or stop sequences). Repair
	/// preserves the independent maximum-output-token limit.
	SamplingParameters,
	/// Anthropic fast mode / `speed` parameter (unsupported or unentitled).
	FastMode,
	/// Replayed thinking-block signature rejected by a signing proxy.
	ThinkingSignature,
}

/// Structured verdict for one provider failure.
#[derive(Clone, Debug, Default)]
pub struct Classification {
	/// Failure kinds (non-exclusive set).
	pub kinds:           Kinds,
	/// Resolved HTTP status (given, embedded in body, or recovered from prose).
	pub status:          Option<u16>,
	/// Verbatim machine code (`usage_limit_reached`, `cyber_policy`, ...).
	pub code:            Option<SmolStr>,
	/// Verbatim error type token when distinct from `code`.
	pub error_type:      Option<SmolStr>,
	/// Provider request id when the envelope carried one.
	pub request_id:      Option<SmolStr>,
	/// Rejected request feature, when the failure is feature-scoped.
	pub feature:         Option<Feature>,
	/// Allowed `reasoning.effort` values scraped from the rejection, in
	/// provider order.
	pub allowed_efforts: SmallVec<SmolStr, 6>,
	/// Rate-limit lane verdict (429/402/quota family only).
	pub rate_limit:      Option<RateLimit>,
	/// Provider-stated retry timing (headers preferred over prose).
	pub retry:           Option<RetryHint>,
	/// OAuth refresh triage, when the text carries grant-death evidence.
	pub oauth:           Option<OAuthFailure>,
	/// Evidence quality behind this verdict.
	pub fidelity:        Fidelity,
}

impl Classification {
	/// Whether `kind` is present.
	#[inline]
	pub const fn is(&self, kind: Kind) -> bool {
		self.kinds.has(kind)
	}

	/// Whether re-issuing the EXACT same request has a chance of succeeding.
	///
	/// `replay_safe` is the caller's statement that re-issuing cannot
	/// duplicate visible output or tool side effects. Every lane whose
	/// recovery requires mutation (stale session state, unsupported feature,
	/// context overflow) or credential work (auth, OAuth, account policy,
	/// usage/billing) refuses: an identical replay keeps the exact condition
	/// that failed - a 500 wrapping `auth_unavailable` must reach auth
	/// resolution, not a replay loop.
	pub const fn retryable_exact_request(&self, replay_safe: bool) -> bool {
		replay_safe
			&& !self.kinds.intersects(Kinds::EXACT_REPLAY_BARS)
			// A rotatable rate limit belongs to the credential lane; spinning
			// on the same credential would burn the whole wait window.
			&& !matches!(self.rate_limit, Some(RateLimit { rotate: true, .. }))
			&& self.kinds.intersects(Kinds::RETRYABLE)
	}

	/// Whether the failure is recoverable at the credential layer (rotate to
	/// a sibling account or refresh/replace the credential), as opposed to
	/// the same-route retry lane.
	pub fn credential_recoverable(&self) -> bool {
		self.rate_limit.is_some_and(|r| r.rotate)
			|| self.kinds.has(Kind::AccountPolicy)
			|| self.kinds.has(Kind::AuthFailed)
			|| self.kinds.has(Kind::OAuthExpired)
	}

	/// Backoff the provider asked for or the lane suggests: explicit retry
	/// hint first, otherwise the rate-limit lane window `(min, max)`.
	pub fn suggested_backoff_ms(&self) -> Option<(u64, u64)> {
		if let Some(hint) = self.retry {
			return Some((hint.delay_ms, hint.delay_ms));
		}
		self.rate_limit.map(|r| r.reason.backoff_ms())
	}

	/// Whether the verdict is terminal for this request: nothing at the
	/// route, credential, or request-shape layer can recover it.
	pub fn terminal(&self) -> bool {
		!self.retryable_exact_request(true)
			&& !self.credential_recoverable()
			&& self.feature.is_none()
			&& !self.kinds.has(Kind::ContextOverflow)
			&& !self.kinds.has(Kind::RequestTooLarge)
			&& !self.kinds.has(Kind::StaleSessionItem)
	}
}

/// Classifies with the system clock (epoch-valued retry headers need "now").
pub fn classify(ev: &Evidence<'_>) -> Classification {
	let now_ms = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_or(0, |d| d.as_millis() as u64);
	classify_at(ev, now_ms)
}

/// Classifies with an explicit clock. `now_ms` is epoch milliseconds; it
/// only affects retry-timing extraction, never the verdict.
pub fn classify_at(ev: &Evidence<'_>, now_ms: u64) -> Classification {
	let env = ev.body.and_then(envelope::parse);
	let mut cls = Classification::default();

	// ── Assemble the text corpus ─────────────────────────────────────────
	let mut texts: SmallVec<&str, 4> = SmallVec::new();
	if let Some(m) = ev.message {
		texts.push(m);
	}
	if let Some(env) = &env {
		if let Some(m) = env.message.as_deref()
			&& ev.message != Some(m)
		{
			texts.push(m);
		}
		if let Some(w) = env.status_word.as_deref() {
			texts.push(w);
		}
	}
	// A body that failed envelope parsing entirely still deserves prose
	// classification (envelope::parse folds it into `message`, so only the
	// no-body-at-all case is left out here).

	// ── Structural: machine codes ────────────────────────────────────────
	let code = ev
		.code
		.map(SmolStr::new)
		.or_else(|| env.as_ref().and_then(|e| e.code.clone()));
	let error_type = env.as_ref().and_then(|e| e.error_type.clone());
	for token in [code.as_deref(), error_type.as_deref()]
		.into_iter()
		.flatten()
	{
		cls.kinds |= kinds_for_code(token, ev.provider);
	}
	if let Some(env) = &env {
		if let Some(word) = env.status_word.as_deref() {
			cls.kinds |= kinds_for_status_word(word);
		}
		cls.request_id.clone_from(&env.request_id);
	}

	// ── Status resolution ────────────────────────────────────────────────
	let status = ev
		.status
		.or_else(|| env.as_ref().and_then(|e| e.status_code))
		.or_else(|| texts.iter().find_map(|t| extract::status_from_text(t)));
	cls.kinds |= kinds_for_status(status);

	// ── Prose corpus ─────────────────────────────────────────────────────
	// Machine tokens (code / error type) participate in prose matching too:
	// compound guards like "invalid_request_error + compiled grammar" span
	// the type field and the message.
	let code_txt = code.as_deref();
	let type_txt = error_type.as_deref();
	let any = |re: &regex::Regex| {
		texts.iter().any(|t| re.is_match(t))
			|| code_txt.is_some_and(|t| re.is_match(t))
			|| type_txt.is_some_and(|t| re.is_match(t))
	};
	for (re, kinds) in patterns::TEXT_RULES.iter() {
		if any(re) {
			cls.kinds |= *kinds;
		}
	}
	if patterns::OVERFLOW.iter().any(&any) {
		cls.kinds |= Kind::ContextOverflow;
	}
	let stream_parse: &regex::Regex = if ev.phase == Phase::Persisted {
		&patterns::STREAM_PARSE_PERSISTED
	} else {
		&patterns::STREAM_PARSE_LIVE
	};
	if any(stream_parse) {
		cls.kinds |= Kind::StreamCorruption | Kind::Transient;
	}
	if any(&patterns::INVALID_MODEL) {
		cls.kinds |= Kind::InvalidModel;
	}
	if any(&patterns::OAUTH_DEFINITIVE) {
		cls.kinds |= Kind::OAuthExpired | Kind::AuthFailed;
		cls.oauth = Some(OAuthFailure::Definitive);
	} else if cls.kinds.has(Kind::AuthFailed)
		&& texts
			.iter()
			.any(|t| t.contains("oauth") || t.contains("OAuth"))
	{
		cls.oauth = Some(texts.iter().map(|t| oauth::classify_refresh(t)).fold(
			OAuthFailure::Transient,
			|acc, v| {
				if v == OAuthFailure::Definitive {
					v
				} else {
					acc
				}
			},
		));
	}

	// ── Compound feature guards ──────────────────────────────────────────
	apply_feature_guards(&mut cls, status, &texts, &any);

	// Machine-code stale tokens are unambiguous on any API.
	if any(&patterns::STALE_SESSION_CODES) {
		cls.kinds |= Kind::StaleSessionItem | Kind::Transient;
	}
	// Stale Responses-item recovery is API-gated: the same words on a
	// non-Responses API are a different bug.
	if ev.api.is_some_and(crate::WireApi::is_responses)
		&& (any(&patterns::STALE_ITEM_ID)
			|| (any(&patterns::STALE_PREVIOUS_RESPONSE) && any(&patterns::STALE_DETAIL)))
	{
		cls.kinds |= Kind::StaleSessionItem | Kind::Transient;
	}

	// Copilot fleet skew: 400 `model_not_supported` from a stale replica is
	// transient, not a durable entitlement verdict.
	if ev.provider.is_some_and(|p| p.contains("copilot"))
		&& status == Some(400)
		&& any(&patterns::COPILOT_TRANSIENT_MODEL)
	{
		cls.kinds.remove(Kind::InvalidModel);
		cls.kinds |= Kind::Transient;
	}

	// ── Rate-limit lane ──────────────────────────────────────────────────
	// Keep the corpus as borrowed pieces. Field boundaries remain hard
	// boundaries while the rate-limit compound guards aggregate signals,
	// avoiding a transient joined String allocation on every failure.
	let mut rate_texts: SmallVec<&str, 6> = SmallVec::new();
	rate_texts.extend(texts.iter().copied());
	rate_texts.extend([code_txt, type_txt].into_iter().flatten());
	let limit_family = matches!(status, Some(429 | 402))
		|| cls
			.kinds
			.intersects(Kind::RateThrottle | Kind::ModelCapacity | Kind::UsageLimit)
		|| rate_limit::matches_usage_limit_parts(&rate_texts)
		|| rate_limit::is_concurrency_cap_parts(status, &rate_texts)
		|| rate_limit::is_usage_limit_outcome_parts(status, &rate_texts);
	if limit_family {
		let mut reason = rate_limit::parse_reason_parts(&rate_texts);
		let rotate = rate_limit::is_usage_limit_outcome_parts(status, &rate_texts);
		if reason == RateLimitReason::Unknown && rotate {
			// Opaque 429/402: conservative quota treatment.
			reason = RateLimitReason::QuotaExhausted;
		}
		match reason {
			RateLimitReason::QuotaExhausted => cls.kinds |= Kind::UsageLimit,
			RateLimitReason::RateLimitExceeded => cls.kinds |= Kind::RateThrottle | Kind::Transient,
			RateLimitReason::ConcurrentLimit => {
				cls.kinds |= Kind::ConcurrencyCap | Kind::Transient;
				if status != Some(402) {
					// A concurrency cap must never consume the credential.
					cls.kinds.remove(Kind::UsageLimit);
				}
			},
			RateLimitReason::ModelCapacityExhausted => {
				cls.kinds |= Kind::ModelCapacity | Kind::Transient;
			},
			RateLimitReason::ServerError => cls.kinds |= Kind::ServerError | Kind::Transient,
			RateLimitReason::Unknown => {},
		}
		if rotate {
			cls.kinds |= Kind::UsageLimit;
		}
		if status == Some(402) {
			cls.kinds |= Kind::Billing | Kind::UsageLimit;
		}
		cls.rate_limit = Some(RateLimit { reason, rotate });
	}
	// `rate_texts` borrows `code`/`error_type`; end it before they move
	// into the returned classification.
	drop(rate_texts);

	// ── Guards and post-processing ───────────────────────────────────────
	// Deterministic failures strip same-route retryability regardless of how
	// retryable the surface looked (the 500-status llama.cpp case).
	if cls.kinds.has(Kind::Deterministic) {
		cls.kinds.remove(Kind::Transient);
	}

	// ── Retry timing ─────────────────────────────────────────────────────
	cls.retry = extract::retry_hint_from_headers(ev.headers, now_ms).or_else(|| {
		texts
			.iter()
			.find_map(|t| extract::retry_hint_from_text(t, now_ms))
	});

	cls.status = status;
	cls.code = code;
	cls.error_type = error_type;
	cls.fidelity =
		if ev.status.is_some() || ev.code.is_some() || env.as_ref().is_some_and(|e| !e.opaque) {
			Fidelity::Structured
		} else {
			Fidelity::Prose
		};
	cls
}

/// Kinds implied by a machine error code, independent of message text.
fn kinds_for_code(code: &str, provider: Option<&str>) -> Kinds {
	let mut lower_buf = [0u8; 64];
	let lower = to_lower(code, &mut lower_buf);
	match lower {
		"usage_limit_reached" | "usage_not_included" | "insufficient_quota" | "quota_exceeded" => {
			Kinds::only(Kind::UsageLimit)
		},
		"insufficient_balance" | "no_credit" => Kind::UsageLimit | Kind::Billing,
		"rate_limit_error" | "rate_limit_exceeded" => Kind::RateThrottle | Kind::Transient,
		"model_cooldown" => Kind::RateThrottle | Kind::Transient,
		"overloaded_error" => Kind::ModelCapacity | Kind::Transient,
		"context_length_exceeded" => Kinds::only(Kind::ContextOverflow),
		"content_filter" | "invalid_prompt" => Kinds::only(Kind::ContentBlocked),
		"cyber_policy" => Kind::AccountPolicy | Kind::ContentBlocked,
		"previous_response_not_found" | "codex_previous_response_stale" => {
			Kind::StaleSessionItem | Kind::Transient
		},
		"model_not_supported" => {
			if provider.is_some_and(|p| p.contains("copilot")) {
				Kinds::only(Kind::Transient)
			} else {
				Kinds::only(Kind::InvalidModel)
			}
		},
		"model_not_available_for_integrator" | "model_not_found" => Kinds::only(Kind::InvalidModel),
		"model_error" | "server_error" | "internal_error" | "internal_server_error" | "api_error" => {
			Kind::ServerError | Kind::Transient
		},
		"websocket_connection_limit_reached" => Kind::Transport | Kind::Transient,
		"invalid_request_error" => Kinds::only(Kind::InvalidRequest),
		"timeout" => Kind::Timeout | Kind::Transient,
		"authentication_error" | "invalid_api_key" | "permission_error" => {
			Kinds::only(Kind::AuthFailed)
		},
		_ => Kinds::EMPTY,
	}
}

/// Kinds implied by a gRPC status word.
fn kinds_for_status_word(word: &str) -> Kinds {
	match word {
		"INVALID_ARGUMENT" | "FAILED_PRECONDITION" | "OUT_OF_RANGE" => {
			Kinds::only(Kind::InvalidRequest)
		},
		"PERMISSION_DENIED" | "UNAUTHENTICATED" => Kinds::only(Kind::AuthFailed),
		"UNAVAILABLE" | "INTERNAL" | "ABORTED" => Kind::ServerError | Kind::Transient,
		"DEADLINE_EXCEEDED" => Kind::Timeout | Kind::Transient,
		// RESOURCE_EXHAUSTED deliberately maps to nothing here: the
		// rate-limit lane disambiguates it from the message detail (bare =
		// model capacity; explicit quota detail = usage limit).
		_ => Kinds::EMPTY,
	}
}

/// Kinds implied by the HTTP status alone (weakest evidence; codes and
/// text override).
fn kinds_for_status(status: Option<u16>) -> Kinds {
	match status {
		Some(401 | 403) => Kinds::only(Kind::AuthFailed),
		Some(402) => Kind::UsageLimit | Kind::Billing,
		Some(408) => Kind::Timeout | Kind::Transient,
		Some(413) => Kind::RequestTooLarge | Kind::ContextOverflow,
		Some(425) => Kinds::only(Kind::Transient),
		Some(429) => Kinds::only(Kind::Transient),
		Some(499) => Kinds::only(Kind::Aborted),
		Some(529) => Kind::ModelCapacity | Kind::Transient,
		Some(s) if s >= 500 => Kind::ServerError | Kind::Transient,
		_ => Kinds::EMPTY,
	}
}

fn apply_feature_guards(
	cls: &mut Classification,
	status: Option<u16>,
	texts: &[&str],
	any: &impl Fn(&regex::Regex) -> bool,
) {
	let bad_request = matches!(status, Some(400 | 422));

	// Thinking-signature rejection is not status-gated: signing proxies have
	// been observed emitting it with and without a clean 400.
	if any(&patterns::THINKING_SIGNATURE) {
		cls.feature = Some(Feature::ThinkingSignature);
		cls.kinds |= Kind::FeatureUnsupported | Kind::InvalidRequest;
		return;
	}
	if bad_request {
		let strict = (any(&patterns::STRICT_FIELD) && any(&patterns::EXTRA_INPUTS))
			|| any(&patterns::STRICT_TOOLS_COMPAT)
			|| (any(&patterns::INVALID_REQUEST_TYPE)
				&& ((any(&patterns::GRAMMAR_TOO_LARGE) && any(&patterns::TOO_LARGE_DETAIL))
					|| (any(&patterns::SCHEMA_WORD) && any(&patterns::TOO_COMPLEX_DETAIL))));
		let structured = any(&patterns::STRUCTURED_OUTPUTS) && any(&patterns::FEATURE_NOT_SUPPORTED);
		if strict || structured {
			cls.feature = Some(if structured && !strict {
				Feature::StructuredOutputs
			} else {
				Feature::StrictTools
			});
			cls.kinds |= Kind::FeatureUnsupported;
			return;
		}
		if any(&patterns::SAMPLING_PARAMETER_REJECTION) {
			cls.feature = Some(Feature::SamplingParameters);
			cls.kinds |= Kind::FeatureUnsupported;
			return;
		}
		if any(&patterns::REASONING_FIELD) && any(&patterns::REASONING_REJECTED) {
			cls.feature = Some(Feature::ReasoningEffort);
			cls.kinds |= Kind::FeatureUnsupported;
			if any(&patterns::REASONING_ALLOWED_CUE) {
				for text in texts {
					for cap in patterns::REASONING_ALLOWED_VALUE.captures_iter(text) {
						let value = &cap[1];
						if !cls.allowed_efforts.iter().any(|v| v == value) {
							cls.allowed_efforts.push(SmolStr::new(value));
						}
					}
				}
			}
			return;
		}
		if status == Some(400)
			&& any(&patterns::TOOL_CHOICE_WORD)
			&& any(&patterns::TOOL_CHOICE_AUTO)
			&& any(&patterns::TOOL_CHOICE_SUPPORTED)
		{
			cls.feature = Some(Feature::ToolChoice);
			cls.kinds |= Kind::FeatureUnsupported;
			return;
		}
		if any(&patterns::INVALID_REQUEST_TYPE)
			&& any(&patterns::FAST_MODE_SPEED)
			&& any(&patterns::FAST_MODE_NOT_SUPPORTED)
		{
			cls.feature = Some(Feature::FastMode);
			cls.kinds |= Kind::FeatureUnsupported;
			return;
		}
	}
	// Fast mode can also surface as a 429 rate_limit_error when the account
	// lacks the extra-usage entitlement.
	if status == Some(429)
		&& any(&patterns::FAST_MODE_RATE_LIMIT)
		&& any(&patterns::FAST_MODE_ENTITLEMENT)
	{
		cls.feature = Some(Feature::FastMode);
		cls.kinds |= Kind::FeatureUnsupported;
	}
}

/// ASCII-lowercases `s` into `buf` when it fits, otherwise returns `s`
/// unchanged (codes longer than the buffer are not in the table anyway).
fn to_lower<'a>(s: &'a str, buf: &'a mut [u8; 64]) -> &'a str {
	if s.len() > buf.len() || !s.is_ascii() {
		return s;
	}
	let out = &mut buf[..s.len()];
	out.copy_from_slice(s.as_bytes());
	out.make_ascii_lowercase();
	// SAFETY: input was ASCII; ASCII lowercasing preserves UTF-8.
	unsafe { core::str::from_utf8_unchecked(out) }
}

impl Envelope {
	/// Whether the envelope carried any machine-readable field.
	pub const fn is_structured(&self) -> bool {
		!self.opaque
	}
}
