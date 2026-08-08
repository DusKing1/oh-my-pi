//! Retry-timing and embedded-status extraction.
//!
//! Headers are consequence metadata, not classification input: they say WHEN
//! to come back, never WHAT went wrong. Prose scraping exists because
//! persisted error strings are frequently the only surviving evidence.

use std::sync::LazyLock;

use regex::Regex;

/// Parsed retry timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryHint {
	/// Relative delay before the next attempt.
	pub delay_ms:    u64,
	/// Absolute reset instant (epoch ms) when the source carried one.
	pub reset_at_ms: Option<u64>,
}

/// Extracts a retry hint from response headers.
///
/// Candidates: `retry-after-ms` (ms delta), `retry-after` (seconds or HTTP
/// date), `x-ratelimit-reset-ms`, `x-ratelimit-reset` (epoch-ms / epoch-s /
/// delta by magnitude), and Codex `x-codex-{primary,secondary}-reset-at`
/// (epoch). The MAXIMUM valid candidate wins: multiple limiters may be
/// tripped at once and coming back before the slowest one re-trips it.
///
/// `now_ms` is the caller's current epoch time; epoch-valued headers in the
/// past are ignored.
pub fn retry_hint_from_headers(headers: &[(&str, &str)], now_ms: u64) -> Option<RetryHint> {
	let mut best: Option<RetryHint> = None;
	let mut consider = |delay_ms: u64, reset_at_ms: Option<u64>| {
		if delay_ms > 0 && best.is_none_or(|b| delay_ms > b.delay_ms) {
			best = Some(RetryHint { delay_ms, reset_at_ms });
		}
	};
	for &(name, value) in headers {
		let value = value.trim();
		if name.eq_ignore_ascii_case("retry-after-ms") {
			if let Ok(ms) = value.parse::<f64>()
				&& ms.is_finite()
				&& ms > 0.0
			{
				consider(ms.ceil() as u64, None);
			}
		} else if name.eq_ignore_ascii_case("retry-after") {
			if let Ok(secs) = value.parse::<f64>() {
				if secs.is_finite() && secs > 0.0 {
					consider((secs * 1000.0).ceil() as u64, None);
				}
			} else if let Some(at) = parse_http_date_ms(value)
				&& at > now_ms
			{
				consider(at - now_ms, Some(at));
			}
		} else if name.eq_ignore_ascii_case("x-ratelimit-reset-ms") {
			if let Some(hint) = reset_value(value, 1, now_ms) {
				consider(hint.delay_ms, hint.reset_at_ms);
			}
		} else if name.eq_ignore_ascii_case("x-ratelimit-reset") {
			if let Some(hint) = reset_value(value, 1000, now_ms) {
				consider(hint.delay_ms, hint.reset_at_ms);
			}
		} else if (name.eq_ignore_ascii_case("x-codex-primary-reset-at")
			|| name.eq_ignore_ascii_case("x-codex-secondary-reset-at"))
			&& let Some(at) = parse_epoch_ms(value).or_else(|| parse_http_date_ms(value))
			&& at > now_ms
		{
			consider(at - now_ms, Some(at));
		}
	}
	best
}

/// Interprets an `x-ratelimit-reset[-ms]` value.
///
/// Magnitude heuristics: `> 1e12` is epoch milliseconds, `> 1e9` is epoch
/// seconds, anything smaller is a delta in the header's nominal unit
/// (`unit_ms` = 1 for the `-ms` variant, 1000 otherwise). Past epochs and
/// non-positive values are ignored.
fn reset_value(value: &str, unit_ms: u64, now_ms: u64) -> Option<RetryHint> {
	let n = value.parse::<f64>().ok()?;
	if !n.is_finite() || n <= 0.0 {
		return None;
	}
	if n > 1e12 {
		let at = n as u64;
		(at > now_ms).then(|| RetryHint { delay_ms: at - now_ms, reset_at_ms: Some(at) })
	} else if n > 1e9 {
		let at = (n * 1000.0) as u64;
		(at > now_ms).then(|| RetryHint { delay_ms: at - now_ms, reset_at_ms: Some(at) })
	} else {
		Some(RetryHint { delay_ms: (n * unit_ms as f64).ceil() as u64, reset_at_ms: None })
	}
}

fn parse_epoch_ms(value: &str) -> Option<u64> {
	let n = value.parse::<f64>().ok()?;
	if !n.is_finite() || n <= 0.0 {
		return None;
	}
	if n > 1e12 {
		Some(n as u64)
	} else if n > 1e9 {
		Some((n * 1000.0) as u64)
	} else {
		None
	}
}

fn parse_http_date_ms(value: &str) -> Option<u64> {
	let ts = jiff::fmt::rfc2822::DateTimeParser::new()
		.parse_timestamp(value)
		.ok()?;
	u64::try_from(ts.as_millisecond()).ok()
}

static TEXT_RETRY_PATTERNS: LazyLock<[Regex; 4]> = LazyLock::new(|| {
	[
		Regex::new(r"(?i)retry-after-ms\s*[:=]\s*(\d+)").unwrap(),
		Regex::new(r"(?i)retry-after\s*[:=]\s*(\d+)").unwrap(),
		Regex::new(r"(?i)x-ratelimit-reset-ms\s*[:=]\s*(\d+)").unwrap(),
		Regex::new(r"(?i)x-ratelimit-reset\s*[:=]\s*(\d+)").unwrap(),
	]
});

/// Scrapes retry timing from a flattened error message.
///
/// Recognizes the `header=value` forms that get appended when a structured
/// response is formatted to text (`retry-after-ms=1234`, `retry-after: 30`,
/// `x-ratelimit-reset[-ms]`). Units follow the header semantics; reset
/// values use the same magnitude heuristics as the header path.
pub fn retry_hint_from_text(message: &str, now_ms: u64) -> Option<RetryHint> {
	let pats = &*TEXT_RETRY_PATTERNS;
	let mut best: Option<RetryHint> = None;
	let mut consider = |hint: RetryHint| {
		if hint.delay_ms > 0 && best.is_none_or(|b| hint.delay_ms > b.delay_ms) {
			best = Some(hint);
		}
	};
	if let Some(c) = pats[0].captures(message)
		&& let Ok(ms) = c[1].parse::<u64>()
	{
		consider(RetryHint { delay_ms: ms, reset_at_ms: None });
	} else if let Some(c) = pats[1].captures(message)
		&& let Ok(secs) = c[1].parse::<u64>()
	{
		consider(RetryHint { delay_ms: secs.saturating_mul(1000), reset_at_ms: None });
	}
	if let Some(c) = pats[2].captures(message)
		&& let Some(hint) = reset_value(&c[1], 1, now_ms)
	{
		consider(hint);
	} else if let Some(c) = pats[3].captures(message)
		&& let Some(hint) = reset_value(&c[1], 1000, now_ms)
	{
		consider(hint);
	}
	best
}

/// Contextual status regexes. Each requires status-adjacent wording so
/// `"took 200ms"` or a byte count never reads as an HTTP status.
static STATUS_PATTERNS: LazyLock<[Regex; 6]> = LazyLock::new(|| {
	[
		Regex::new(r"(?i)\bstatus(?:_code)?[:=]\s*(\d{3})\b").unwrap(),
		Regex::new(r"(?i)\bstatus\s+(\d{3})\b").unwrap(),
		Regex::new(r"(?i)\bHTTP\s+(\d{3})\b").unwrap(),
		Regex::new(r"(?i)\b(?:error|failed)\s*[:=]?\s*\(?(\d{3})\)?").unwrap(),
		// `429 status code (no body)` — bodyless provider error surface.
		Regex::new(r"(?i)\b(\d{3})\s+status(?:\s+code)?\b").unwrap(),
		// `429 Too Many Requests` — a code followed by a reason phrase.
		Regex::new(r"(?:^|\s)(\d{3})\s+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)").unwrap(),
	]
});

/// Recovers an HTTP status from a flattened error message.
///
/// Only 100–599 pass; every pattern requires context (`status=`, `HTTP`,
/// `error:`, or a trailing reason phrase) to avoid matching durations,
/// counts, or model names.
pub fn status_from_text(message: &str) -> Option<u16> {
	for pat in &*STATUS_PATTERNS {
		if let Some(c) = pat.captures(message)
			&& let Ok(status) = c[1].parse::<u16>()
			&& (100..=599).contains(&status)
		{
			return Some(status);
		}
	}
	None
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn max_candidate_wins() {
		let headers = [("retry-after", "5"), ("x-ratelimit-reset", "30")];
		assert_eq!(retry_hint_from_headers(&headers, 0).unwrap().delay_ms, 30_000);
	}

	#[test]
	fn epoch_seconds_reset() {
		let now = 1_754_000_000_000u64; // epoch ms
		let headers = [("x-ratelimit-reset", "1754000060")]; // epoch s, +60s
		let hint = retry_hint_from_headers(&headers, now).unwrap();
		assert_eq!(hint.delay_ms, 60_000);
		assert_eq!(hint.reset_at_ms, Some(1_754_000_060_000));
	}

	#[test]
	fn past_epoch_ignored() {
		let now = 1_754_000_000_000u64;
		let headers = [("x-ratelimit-reset", "1753000000")];
		assert_eq!(retry_hint_from_headers(&headers, now), None);
	}

	#[test]
	fn http_date_retry_after() {
		let now = 0u64;
		let headers = [("retry-after", "Wed, 21 Oct 2015 07:28:00 GMT")];
		let hint = retry_hint_from_headers(&headers, now).unwrap();
		assert!(hint.delay_ms > 0);
		assert!(hint.reset_at_ms.is_some());
	}

	#[test]
	fn text_scrape_ms_form() {
		let hint = retry_hint_from_text("429 rate limited retry-after-ms=1234", 0).unwrap();
		assert_eq!(hint.delay_ms, 1234);
	}

	#[test]
	fn text_scrape_seconds_form() {
		let hint = retry_hint_from_text("throttled; retry-after: 30", 0).unwrap();
		assert_eq!(hint.delay_ms, 30_000);
	}

	#[test]
	fn status_needs_context() {
		assert_eq!(status_from_text("the request took 200ms total"), None);
		assert_eq!(status_from_text("HTTP 503"), Some(503));
		assert_eq!(status_from_text("status=429; body=..."), Some(429));
		assert_eq!(status_from_text("429 Too Many Requests"), Some(429));
	}
}
