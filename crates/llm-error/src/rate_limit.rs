//! 429/402 disambiguation — the hardest single problem in provider errors.
//!
//! A 429 is not one class. The recovery lane depends on the body:
//!
//! | verdict                    | lane                                | window |
//! |----------------------------|-------------------------------------|--------|
//! | account/plan quota         | rotate credential, block current    | 30 min |
//! | per-minute/burst throttle  | same credential, back off           | 30 s   |
//! | concurrency cap            | same credential, shed               | 5 s    |
//! | model capacity / overload  | same credential, jittered back off  | 45–75 s|
//! | server error               | same credential, back off           | 20 s   |
//! | opaque / unknown           | conservative rotate                 | 30 min |
//!
//! Getting a lane wrong is expensive in a specific direction each time:
//! rotating on a concurrency cap burns every healthy sibling credential on
//! a cap that clears in seconds; backing off on real quota exhaustion
//! wedges a session for half an hour when a sibling could have taken over
//! immediately. HTTP 402 is categorical: always a billing cap, even when
//! the body is worded as a concurrency complaint.

use std::sync::LazyLock;

use regex::Regex;

/// Why the provider limited the request.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RateLimitReason {
	/// Account/plan quota exhausted; credential-scoped and persistent.
	QuotaExhausted,
	/// Short-window throttle (per-minute, burst).
	RateLimitExceeded,
	/// Too many concurrent requests.
	ConcurrentLimit,
	/// Model/fleet capacity exhausted.
	ModelCapacityExhausted,
	/// Provider-side server error surfaced through the limit path.
	ServerError,
	/// Unrecognized; treated conservatively.
	Unknown,
}

impl RateLimitReason {
	/// Suggested backoff window in milliseconds as `(min, max)`; equal when
	/// the lane is not jittered. Values are production-tuned constants.
	pub const fn backoff_ms(self) -> (u64, u64) {
		match self {
			Self::QuotaExhausted | Self::Unknown => (1_800_000, 1_800_000),
			Self::RateLimitExceeded => (30_000, 30_000),
			Self::ConcurrentLimit => (5_000, 5_000),
			Self::ModelCapacityExhausted => (45_000, 75_000),
			Self::ServerError => (20_000, 20_000),
		}
	}
}

/// Rate-limit verdict attached to a classification.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RateLimit {
	/// Disambiguated reason.
	pub reason: RateLimitReason,
	/// Whether the current credential should be taken out of rotation
	/// (blocked/reranked) rather than merely backed off.
	pub rotate: bool,
}

// ── Persistent account-cap phrasings ─────────────────────────────────────

static ACCOUNT_RATE_LIMIT: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(
		r"(?i)\baccount(?:'s)?\b[^\n]{0,80}\brate.?limit\b|\brate.?limit\b[^\n]{0,80}\baccount\b",
	)
	.unwrap()
});
static INSUFFICIENT_BALANCE: LazyLock<Regex> =
	LazyLock::new(|| Regex::new(r"(?i)insufficient.?balance").unwrap());
static SPEND_LIMIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)spend.?limit").unwrap());
/// Subscription/plan cap — excluded when the text names a per-second/minute
/// window.
static SUBSCRIPTION_CAP: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(
		r"(?i)\b(?:subscription|plan|membership)\b[^\n]{0,80}\b(?:rate.?limits?|quota|cap)\b|\b(?:rate.?limits?|quota|cap)\b[^\n]{0,80}\b(?:subscription|plan|membership)\b",
	)
	.unwrap()
});
static PER_WINDOW: LazyLock<Regex> =
	LazyLock::new(|| Regex::new(r"(?i)\bper\s+(?:second|minute)\b").unwrap());
static OPENROUTER_FREE_DAILY: LazyLock<Regex> =
	LazyLock::new(|| Regex::new(r"(?i)\bfree[-_ ]models[-_ ]per[-_ ]day\b").unwrap());

// ── Concurrency ──────────────────────────────────────────────────────────

/// Concurrency cap. Requires a nearby cap signal so "concurrent invocation
/// is not supported" does not false-positive.
static CONCURRENCY: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(
		r"(?i)\btoo many\s+concurren\w*\s+(?:requests?|invocations?)\b|\bconcurren\w*\b[^\n]{0,60}\b(?:limit|quota|exceed\w*|reach\w*)\b|\b(?:limit|quota|exceed\w*|reach\w*)\b[^\n]{0,60}\bconcurren\w*\b|\bconcurren[a-z]*[-_](?:[a-z]+[_-])*(?:limit|quota|exceed\w*|reach\w*)",
	)
	.unwrap()
});

/// Account-scoped cap wording that upgrades a 403/statusless failure to a
/// usage limit. The `your … will reset` arm needs the possessive so a bare
/// "Rate limit will reset in 30 seconds" stays a throttle.
static ACCOUNT_SCOPED_CAP: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(
		r"(?i)\b(?:overall|account|organization|team|workspace)\b[^\n]{0,40}\b(?:message |request )?rate.?limit\b|\byour\b[^\n]{0,30}\b(?:limit )?will reset\b",
	)
	.unwrap()
});

// ── Simplified Chinese ───────────────────────────────────────────────────
// Chinese-market providers phrase quota exhaustion in zh-Hans. The
// transient exclusion keeps per-minute/concurrency caps from burning
// sibling credentials.

static CN_PERSISTENT: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(r"使用.{0,30}?上限|(?:额度|配额)已?(?:用|耗)(?:完|尽)|限额.{0,30}重置|余额不足")
		.unwrap()
});
static CN_TRANSIENT_CAP: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(r"速率.{0,30}上限|频率.{0,30}上限|每分钟.{0,30}上限|并发.{0,30}上限|使用.{0,30}(?:速率|频率|每分钟|并发).{0,30}上限")
		.unwrap()
});
static CN_THROTTLE: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(r"速率(?:限制|过快)|频率(?:过高|过快)|过于频繁|稍后[重再]试").unwrap()
});

// ── Resource exhausted ───────────────────────────────────────────────────

static RESOURCE_EXHAUSTED: LazyLock<Regex> =
	LazyLock::new(|| Regex::new(r"(?i)resource.?exhausted").unwrap());

/// Unified usage/quota matcher.
static USAGE_LIMIT_TEXT: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(
		r"(?i)usage.?limit|usage_limit_reached|usage_not_included|limit_reached|quota.?(?:exceeded|reached|insufficient)|额度不足|额度耗尽|exhausted your capacity|quota will reset|insufficient.?(?:balance|quota)|balance.?exhausted|run out of credits|out of credits|spending[- _]?limit|personal-team-blocked|GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|insufficient_quota|out of budget",
	)
	.unwrap()
});

#[derive(Clone, Copy)]
struct TextParts<'a>(&'a [&'a str]);

impl TextParts<'_> {
	#[inline]
	fn matches(self, regex: &Regex) -> bool {
		self.0.iter().any(|text| regex.is_match(text))
	}

	#[inline]
	fn contains_ascii_case(self, needle: &str) -> bool {
		self.0.iter().any(|text| {
			text
				.as_bytes()
				.windows(needle.len())
				.any(|window| window.eq_ignore_ascii_case(needle.as_bytes()))
		})
	}
}

/// Whether the text carries any account-usage/quota phrasing (before lane
/// disambiguation — a concurrency cap can still say "quota exceeded").
pub fn matches_usage_limit_text(text: &str) -> bool {
	matches_usage_limit_parts(&[text])
}

pub(crate) fn matches_usage_limit_parts(texts: &[&str]) -> bool {
	let texts = TextParts(texts);
	if texts.matches(&CN_PERSISTENT) && !texts.matches(&CN_TRANSIENT_CAP) {
		return true;
	}
	texts.matches(&USAGE_LIMIT_TEXT)
		|| texts.matches(&ACCOUNT_RATE_LIMIT)
		|| texts.matches(&INSUFFICIENT_BALANCE)
		|| texts.matches(&SPEND_LIMIT)
		|| (texts.matches(&SUBSCRIPTION_CAP) && !texts.matches(&PER_WINDOW))
		|| texts.matches(&OPENROUTER_FREE_DAILY)
}

/// Whether the text names an account-scoped cap (upgrades 403/statusless).
pub fn is_account_scoped_cap_text(text: &str) -> bool {
	is_account_scoped_cap_parts(&[text])
}

fn is_account_scoped_cap_parts(texts: &[&str]) -> bool {
	TextParts(texts).matches(&ACCOUNT_SCOPED_CAP)
}

/// Whether the failure is a concurrency cap that must be shed, not rotated.
/// HTTP 402 is excluded categorically: billing caps rotate even when the
/// provider words them as concurrency.
pub fn is_concurrency_cap(status: Option<u16>, text: &str) -> bool {
	is_concurrency_cap_parts(status, &[text])
}

pub(crate) fn is_concurrency_cap_parts(status: Option<u16>, texts: &[&str]) -> bool {
	status != Some(402) && TextParts(texts).matches(&CONCURRENCY)
}

/// Disambiguates the limit reason from body text.
///
/// Priority order is significant and production-derived: explicit detail beats
/// generic words, concurrency beats quota (a cap can say "quota"),
/// capacity/throttle literals beat the generic exhaustion bucket.
pub fn parse_reason(text: &str) -> RateLimitReason {
	parse_reason_parts(&[text])
}

pub(crate) fn parse_reason_parts(texts: &[&str]) -> RateLimitReason {
	let texts = TextParts(texts);
	// resource_exhausted with an explicit quota/rate/server detail is
	// authoritative; bare resource_exhausted means model capacity (handled
	// at the bottom).
	let bare_resource_exhausted = texts.matches(&RESOURCE_EXHAUSTED)
		&& !texts.matches(&USAGE_LIMIT_TEXT)
		&& !texts.contains_ascii_case("quota")
		&& !texts.contains_ascii_case("rate limit")
		&& !texts.contains_ascii_case("server error");
	if texts.contains_ascii_case("quota will reset")
		|| texts.contains_ascii_case("exhausted your capacity")
	{
		return RateLimitReason::QuotaExhausted;
	}
	if texts.matches(&CN_PERSISTENT) && !texts.matches(&CN_TRANSIENT_CAP) {
		return RateLimitReason::QuotaExhausted;
	}
	if is_concurrency_cap_parts(None, texts.0) {
		return RateLimitReason::ConcurrentLimit;
	}
	if texts.contains_ascii_case("capacity")
		|| texts.contains_ascii_case("overloaded")
		|| texts.contains_ascii_case("529")
	{
		return RateLimitReason::ModelCapacityExhausted;
	}
	if texts.matches(&ACCOUNT_RATE_LIMIT)
		|| texts.matches(&SPEND_LIMIT)
		|| texts.matches(&INSUFFICIENT_BALANCE)
		|| (texts.matches(&SUBSCRIPTION_CAP) && !texts.matches(&PER_WINDOW))
		|| texts.matches(&OPENROUTER_FREE_DAILY)
	{
		return RateLimitReason::QuotaExhausted;
	}
	if texts.contains_ascii_case("per minute")
		|| texts.contains_ascii_case("rate limit")
		|| texts.contains_ascii_case("rate_limit")
		|| texts.contains_ascii_case("too many requests")
		|| texts.matches(&CN_THROTTLE)
	{
		return RateLimitReason::RateLimitExceeded;
	}
	if !bare_resource_exhausted
		&& (texts.matches(&USAGE_LIMIT_TEXT)
			|| texts.contains_ascii_case("exhausted")
			|| texts.contains_ascii_case("quota")
			|| texts.contains_ascii_case("usage limit"))
	{
		return RateLimitReason::QuotaExhausted;
	}
	if texts.contains_ascii_case("500")
		|| texts.contains_ascii_case("internal error")
		|| texts.contains_ascii_case("internal server error")
	{
		return RateLimitReason::ServerError;
	}
	if bare_resource_exhausted {
		return RateLimitReason::ModelCapacityExhausted;
	}
	RateLimitReason::Unknown
}

/// Whether the HTTP status alone puts the failure in the usage-limit family
/// (429, or the categorical billing status 402).
pub const fn is_usage_limit_status(status: u16) -> bool {
	matches!(status, 429 | 402)
}

/// Body content check for the conservative-rotation rule: an empty or
/// content-free 429/402 body forces rotation.
///
/// There is nothing to prove the limit is transient. Strips the status numerals
/// and HTTP boilerplate first, then asks whether anything informative survives.
/// Unrecognized Han-only text deliberately stays opaque.
pub fn is_opaque_status_body(text: &str) -> bool {
	is_opaque_status_parts(&[text])
}

fn is_opaque_status_parts(texts: &[&str]) -> bool {
	static STATUS_NUM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(?:429|402)\b").unwrap());
	static BOILERPLATE: LazyLock<Regex> = LazyLock::new(|| {
		Regex::new(r"(?i)\b(?:http|https|status|error|code|response|message)\b|\(no body\)").unwrap()
	});
	static INFORMATIVE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)[a-z\d]{3,}").unwrap());

	let informative = texts.iter().any(|text| {
		INFORMATIVE.find_iter(text).any(|token| {
			!STATUS_NUM
				.find_iter(text)
				.chain(BOILERPLATE.find_iter(text))
				.any(|ignored| ignored.start() <= token.start() && ignored.end() >= token.end())
		})
	});
	let texts = TextParts(texts);
	!(informative
		|| texts.matches(&CN_PERSISTENT)
		|| texts.matches(&CN_TRANSIENT_CAP)
		|| texts.matches(&CN_THROTTLE))
}

/// The credential-rotation decision: does this limit consume the credential
/// (rotate to a sibling, block the current one) rather than back off?
///
/// The algorithm, in order:
/// 1. A non-402 concurrency cap never rotates (shed instead).
/// 2. Explicit usage-limit text always rotates.
/// 3. A 403 (or statusless failure) with account-scoped cap wording rotates; a
///    bare 403 stays an auth failure.
/// 4. Any status other than 429/402 does not rotate.
/// 5. An opaque 429/402 body rotates conservatively — nothing proves it is
///    transient, and waiting 30 minutes on a dead credential is worse than one
///    unnecessary rotation.
/// 6. An informative body rotates only for [`RateLimitReason::QuotaExhausted`],
///    plus 402 concurrency (billing categorical).
pub fn is_usage_limit_outcome(status: Option<u16>, text: &str) -> bool {
	is_usage_limit_outcome_parts(status, &[text])
}

pub(crate) fn is_usage_limit_outcome_parts(status: Option<u16>, texts: &[&str]) -> bool {
	if is_concurrency_cap_parts(status, texts) {
		return false;
	}
	if matches_usage_limit_parts(texts) {
		return true;
	}
	if (status == Some(403) || status.is_none()) && is_account_scoped_cap_parts(texts) {
		return true;
	}
	let Some(status) = status else { return false };
	if !is_usage_limit_status(status) {
		return false;
	}
	if is_opaque_status_parts(texts) {
		return true;
	}
	matches!(parse_reason_parts(texts), RateLimitReason::QuotaExhausted)
		|| (status == 402 && TextParts(texts).matches(&CONCURRENCY))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn concurrency_needs_cap_signal() {
		assert!(is_concurrency_cap(Some(429), "Too many concurrent requests"));
		assert!(is_concurrency_cap(
			Some(429),
			"Online prediction concurrent requests quota exceeded"
		));
		assert!(!is_concurrency_cap(Some(400), "concurrent invocation is not supported"));
	}

	#[test]
	fn cn_quota_vs_throttle() {
		assert_eq!(parse_reason("您的额度已用完"), RateLimitReason::QuotaExhausted);
		assert_eq!(parse_reason("请求速率达到上限，请稍后重试"), RateLimitReason::RateLimitExceeded);
		assert!(matches_usage_limit_text("余额不足"));
		assert!(!matches_usage_limit_text("并发数达到上限"));
	}

	#[test]
	fn possessive_reset_guard() {
		assert!(is_account_scoped_cap_text("Your limit will reset at midnight"));
		assert!(!is_account_scoped_cap_text("Rate limit will reset in 30 seconds"));
	}

	#[test]
	fn opaque_429_rotates() {
		assert!(is_usage_limit_outcome(Some(429), "429 status code (no body)"));
		assert!(is_opaque_status_body("HTTP 429"));
		assert!(!is_opaque_status_body("You have exceeded your rate limit for this API."));
	}

	#[test]
	fn throttle_does_not_rotate() {
		assert!(!is_usage_limit_outcome(
			Some(429),
			"You have exceeded your rate limit for this API. Please try again later."
		));
	}

	#[test]
	fn quota_rotates() {
		assert!(is_usage_limit_outcome(Some(429), "The usage limit has been reached"));
		assert!(is_usage_limit_outcome(
			Some(403),
			"Your overall message rate limit has been reached"
		));
		assert!(!is_usage_limit_outcome(Some(403), "Forbidden"));
	}

	#[test]
	fn categorical_402() {
		assert!(is_usage_limit_outcome(Some(402), "too many concurrent requests, upgrade your plan"));
	}

	#[test]
	fn bare_resource_exhausted_is_capacity() {
		assert_eq!(parse_reason("RESOURCE_EXHAUSTED"), RateLimitReason::ModelCapacityExhausted);
		assert_eq!(
			parse_reason("resource_exhausted: quota exceeded for this project"),
			RateLimitReason::QuotaExhausted
		);
	}
}
