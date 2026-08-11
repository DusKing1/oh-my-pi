//! Recovery policy: action advice, retry budgets, credential blocks.
//!
//! The classifier says WHAT failed; this module says WHAT TO DO. The two
//! are deliberately separate types — every attempt to fuse detection and
//! policy into one value has collapsed at least two of the six distinct
//! "retryable" questions (same-route retry, backoff, rotation,
//! invalidation, feature strip, context reduction) into one boolean.

use std::collections::HashMap;

use omp_core::Str;
use smallvec::SmallVec;

use crate::{
	classify::{Classification, Feature},
	kind::Kind,
	oauth::OAuthFailure,
	rate_limit::RateLimitReason,
};

/// One recovery action, in the order a consumer should attempt them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
	/// Re-issue the same request on the same route after `delay_ms`.
	RetrySameRoute {
		/// Delay before the attempt.
		delay_ms: u64,
	},
	/// Keep the credential but hold traffic for `delay_ms` (throttle,
	/// concurrency shed, capacity wait).
	Backoff {
		/// Hold duration.
		delay_ms: u64,
	},
	/// Take the current credential out of rotation for `block_ms` and move
	/// to a sibling.
	RotateCredential {
		/// How long the current credential stays blocked.
		block_ms: u64,
	},
	/// Refresh the OAuth token / re-resolve the credential in place.
	RefreshCredential,
	/// The grant is definitively dead — interactive re-login required.
	Relogin,
	/// Delete/invalidate the stored credential. Only advised on
	/// [`Fidelity::Structured`] evidence: a prose-only 401 has repeatedly
	/// turned out to be stale replayed session state, and deleting a valid
	/// credential on it is the worst outcome this module can cause.
	InvalidateCredential,
	/// Shrink the prompt (compact history or promote to a larger-context
	/// model) and retry.
	ReduceContext,
	/// Remove or mutate the named request feature and retry.
	StripFeature(Feature),
	/// Reset server-held session state (drop the stale `previous_response_id`
	/// chain, replay full context).
	ResetSession,
	/// Route to a different model.
	SwitchModel,
	/// Discard the output and sample again (thinking loop, malformed tool
	/// call, empty response).
	Resample,
	/// Nothing recovers this automatically; surface to the operator.
	SurfaceTerminal,
}

/// Ordered recovery plan for one classified failure.
pub type Advice = SmallVec<Action, 4>;

/// Context the advisor needs beyond the classification itself.
#[derive(Clone, Copy, Debug)]
pub struct AdviseContext {
	/// Re-issuing the request cannot duplicate visible output or tool side
	/// effects (no tokens streamed yet, or the protocol replays
	/// idempotently).
	pub replay_safe:            bool,
	/// A sibling credential exists to rotate to.
	pub has_sibling_credential: bool,
	/// A fallback model exists to switch to.
	pub has_fallback_model:     bool,
}

impl Default for AdviseContext {
	fn default() -> Self {
		Self {
			replay_safe:            true,
			has_sibling_credential: false,
			has_fallback_model:     false,
		}
	}
}

/// Derives the ordered recovery plan for a classification.
///
/// The order encodes hard-won precedence: feature strips and context
/// reduction beat blind retries (they fix the cause), credential moves beat
/// model switches (cheaper), and everything beats surfacing terminally.
pub fn advise(cls: &Classification, ctx: &AdviseContext) -> Advice {
	let mut plan = Advice::new();

	// Deliberate cancellation: never fight the caller.
	if cls.is(Kind::Aborted) {
		plan.push(Action::SurfaceTerminal);
		return plan;
	}

	// Cause-level fixes first.
	if let Some(feature) = cls.feature {
		plan.push(Action::StripFeature(feature));
	}
	if cls.is(Kind::ContextOverflow) || cls.is(Kind::RequestTooLarge) {
		plan.push(Action::ReduceContext);
	}
	if cls.is(Kind::StaleSessionItem) {
		plan.push(Action::ResetSession);
	}

	// Content policy: model/account scoped, not retryable in place.
	if cls.is(Kind::ContentBlocked) {
		if cls.is(Kind::AccountPolicy) && ctx.has_sibling_credential {
			plan.push(Action::RotateCredential { block_ms: 0 });
		}
		if ctx.has_fallback_model {
			plan.push(Action::SwitchModel);
		}
		plan.push(Action::SurfaceTerminal);
		return plan;
	}

	// Credential layer.
	if cls.is(Kind::OAuthExpired) || cls.oauth == Some(OAuthFailure::Definitive) {
		// Positive revocation evidence (invalid_grant, revoked, refresh token
		// expired, invalidated token): the stored grant is dead. This is the
		// ONLY lane that advises deleting a credential — a parsed 401/403 is
		// not revocation proof.
		plan.push(Action::Relogin);
		plan.push(Action::InvalidateCredential);
	} else if let Some(limit) = cls.rate_limit
		&& limit.rotate
	{
		let (block_ms, _) = limit.reason.backoff_ms();
		if ctx.has_sibling_credential {
			plan.push(Action::RotateCredential { block_ms });
		} else {
			plan.push(Action::Backoff { delay_ms: cls.retry.map_or(block_ms, |h| h.delay_ms) });
		}
	} else if cls.is(Kind::AccountPolicy) && ctx.has_sibling_credential {
		plan.push(Action::RotateCredential { block_ms: 0 });
	} else if cls.is(Kind::AuthFailed)
		&& !cls.is(Kind::UsageLimit)
		&& !cls.is(Kind::ConcurrencyCap)
		// A positively identified stale-session item owns the failure even
		// when it rode in on a 401/403: the credential is fine, the replayed
		// server-side state is not. Touching the credential here is the
		// documented way to delete a VALID token.
		&& !cls.is(Kind::StaleSessionItem)
	{
		// Ordinary auth failure: refresh in place, then move to a sibling.
		// Never delete on status evidence alone.
		plan.push(Action::RefreshCredential);
		if ctx.has_sibling_credential {
			plan.push(Action::RotateCredential { block_ms: 0 });
		}
	}

	// Same-credential wait lanes.
	if let Some(limit) = cls.rate_limit
		&& !limit.rotate
	{
		let (min, _) = limit.reason.backoff_ms();
		plan.push(Action::Backoff { delay_ms: cls.retry.map_or(min, |h| h.delay_ms) });
	}

	// Output-quality resampling.
	if cls.is(Kind::ThinkingLoop) || cls.is(Kind::MalformedToolCall) || cls.is(Kind::EmptyResponse) {
		if ctx.replay_safe && !cls.is(Kind::Deterministic) {
			plan.push(Action::Resample);
		} else if cls.is(Kind::Deterministic) && ctx.has_fallback_model {
			plan.push(Action::SwitchModel);
		}
	}

	// Same-route retry for whatever transient surface remains.
	if plan.is_empty() && cls.retryable_exact_request(ctx.replay_safe) {
		plan.push(Action::RetrySameRoute { delay_ms: cls.retry.map_or(0, |h| h.delay_ms) });
	}

	if plan.is_empty() {
		plan.push(Action::SurfaceTerminal);
	}
	plan
}

/// Why a retry budget refused another attempt.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GiveUp {
	/// The classification does not admit a same-route retry.
	NotRetryable,
	/// Attempt budget spent.
	BudgetExhausted,
	/// Provider asked for a wait longer than the caller tolerates; waiting
	/// silently would look like a hang.
	HintTooLong {
		/// The provider-requested delay.
		delay_ms: u64,
	},
}

/// Verdict from [`RetryBudget::decide`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RetryDecision {
	/// Sleep `delay_ms`, then attempt number `attempt` (1-based).
	Retry {
		/// Delay before the attempt.
		delay_ms: u64,
		/// 1-based attempt number.
		attempt:  u32,
	},
	/// Stop retrying.
	GiveUp(GiveUp),
}

/// Bounded jittered-exponential retry budget for one logical request.
///
/// Delay = `min(base · 2^(n-1), cap)` with up to 25% downward jitter, the
/// production-tuned curve; an explicit provider hint overrides the curve
/// and fails fast when it exceeds `max_hint_ms`.
#[derive(Clone, Debug)]
pub struct RetryBudget {
	/// Maximum attempts before giving up.
	pub max_attempts:  u32,
	/// Exponential base delay.
	pub base_delay_ms: u64,
	/// Exponential cap.
	pub delay_cap_ms:  u64,
	/// Longest provider-requested wait honored before failing fast.
	pub max_hint_ms:   u64,
	attempts:          u32,
}

impl Default for RetryBudget {
	/// Production defaults: 10 attempts, 500 ms base, 8 s cap, 5 min hint
	/// ceiling.
	fn default() -> Self {
		Self {
			max_attempts:  10,
			base_delay_ms: 500,
			delay_cap_ms:  8_000,
			max_hint_ms:   300_000,
			attempts:      0,
		}
	}
}

impl RetryBudget {
	/// Budget with explicit limits; see field docs for semantics.
	pub const fn new(
		max_attempts: u32,
		base_delay_ms: u64,
		delay_cap_ms: u64,
		max_hint_ms: u64,
	) -> Self {
		Self { max_attempts, base_delay_ms, delay_cap_ms, max_hint_ms, attempts: 0 }
	}

	/// Attempts consumed so far.
	pub const fn attempts(&self) -> u32 {
		self.attempts
	}

	/// Resets the budget (e.g. after a successful model switch, which earns
	/// a fresh budget).
	pub const fn reset(&mut self) {
		self.attempts = 0;
	}

	/// Decides whether to retry after a failure classified as `cls`.
	///
	/// `replay_safe` gates duplication risk; `jitter01` is caller-supplied
	/// entropy in `[0, 1)` (pass `0.0` for deterministic tests).
	pub fn decide(
		&mut self,
		cls: &Classification,
		replay_safe: bool,
		jitter01: f64,
	) -> RetryDecision {
		if !cls.retryable_exact_request(replay_safe) {
			return RetryDecision::GiveUp(GiveUp::NotRetryable);
		}
		self.decide_allowed(cls, jitter01)
	}

	/// Decides one retry after the caller successfully repairs stale
	/// Responses continuation state.
	///
	/// This lane is deliberately narrower than exact replay: `repaired` must
	/// mean the stale server anchor and item ids were removed before dispatch,
	/// and the classification must prove `StaleSessionItem`. It shares the
	/// ordinary attempt, delay, and provider-hint caps.
	pub fn decide_repaired(
		&mut self,
		cls: &Classification,
		repaired: bool,
		jitter01: f64,
	) -> RetryDecision {
		if !repaired || !cls.kinds.has(Kind::StaleSessionItem) {
			return RetryDecision::GiveUp(GiveUp::NotRetryable);
		}
		self.decide_allowed(cls, jitter01)
	}

	fn decide_allowed(&mut self, cls: &Classification, jitter01: f64) -> RetryDecision {
		if self.attempts >= self.max_attempts {
			return RetryDecision::GiveUp(GiveUp::BudgetExhausted);
		}
		self.attempts += 1;
		let exp = self
			.base_delay_ms
			.saturating_mul(1u64 << (self.attempts - 1).min(20))
			.min(self.delay_cap_ms);
		let delay_ms = match cls.retry {
			Some(hint) if hint.delay_ms > self.max_hint_ms => {
				return RetryDecision::GiveUp(GiveUp::HintTooLong { delay_ms: hint.delay_ms });
			},
			Some(hint) => hint.delay_ms.max(exp),
			None => {
				let lane = cls.rate_limit.map(|r| r.reason.backoff_ms());
				match lane {
					Some((min, max)) if cls.rate_limit.is_some_and(|r| !r.rotate) => {
						min + ((max - min) as f64 * jitter01) as u64
					},
					_ => {
						let jitterless = exp as f64;
						(jitterless * jitter01.clamp(0.0, 1.0).mul_add(-0.25, 1.0)) as u64
					},
				}
			},
		};
		RetryDecision::Retry { delay_ms, attempt: self.attempts }
	}
}

/// Key addressing one credential (optionally one metered scope within it —
/// e.g. a Codex usage window, so a Spark-meter block does not gate ordinary
/// chat requests on the same account).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct BlockKey {
	/// Stable credential identity (row id, account id, bearer hash — the
	/// caller's choice, opaque here).
	pub credential: Str,
	/// Optional meter/window scope. `None` blocks the whole credential.
	pub scope:      Option<Str>,
}

impl BlockKey {
	/// Whole-credential key.
	pub fn credential(id: &str) -> Self {
		Self { credential: Str::new(id), scope: None }
	}

	/// Meter-scoped key.
	pub fn scoped(id: &str, scope: &str) -> Self {
		Self { credential: Str::new(id), scope: Some(Str::new(scope)) }
	}
}

/// In-memory credential block table.
///
/// Tracks until-instants per credential/scope so usage-limit blocks apply
/// across the process. Persistence is the caller's concern — the table is
/// deliberately clock-agnostic (`now_ms` is always passed in) so it can be
/// driven from tests, a monotonic clock, or replayed state.
#[derive(Debug, Default)]
pub struct BlockTable {
	entries: HashMap<BlockKey, u64>,
}

impl BlockTable {
	/// Empty table.
	pub fn new() -> Self {
		Self::default()
	}

	/// Blocks `key` until `now_ms + duration_ms`, keeping the later deadline
	/// if one is already recorded. Returns the effective unblock instant.
	pub fn block(&mut self, key: BlockKey, now_ms: u64, duration_ms: u64) -> u64 {
		let until = now_ms.saturating_add(duration_ms);
		let entry = self.entries.entry(key).or_insert(0);
		*entry = (*entry).max(until);
		*entry
	}

	/// Blocks `key` for the duration the classification implies: the
	/// provider's stated reset first, otherwise the rate-limit lane window,
	/// otherwise the conservative quota window.
	pub fn block_for(&mut self, key: BlockKey, now_ms: u64, cls: &Classification) -> u64 {
		let duration = cls
			.retry
			.map(|h| h.delay_ms)
			.or_else(|| cls.rate_limit.map(|r| r.reason.backoff_ms().0))
			.unwrap_or_else(|| RateLimitReason::Unknown.backoff_ms().0);
		self.block(key, now_ms, duration)
	}

	/// Remaining block on `key`, if any. A scoped query also honors a
	/// whole-credential block.
	pub fn blocked_for_ms(&self, key: &BlockKey, now_ms: u64) -> Option<u64> {
		let direct = self.entries.get(key).copied();
		let whole = key
			.scope
			.is_some()
			.then(|| {
				self
					.entries
					.get(&BlockKey { credential: key.credential.clone(), scope: None })
					.copied()
			})
			.flatten();
		let until = direct.into_iter().chain(whole).max()?;
		(until > now_ms).then_some(until - now_ms)
	}

	/// Earliest unblock instant across all still-active blocks; `None` when
	/// nothing is blocked. This is the "when is ANY sibling usable again"
	/// question the wait-or-fail decision needs.
	pub fn earliest_unblock_ms(&self, now_ms: u64) -> Option<u64> {
		self
			.entries
			.values()
			.copied()
			.filter(|&until| until > now_ms)
			.min()
	}

	/// Clears a block (e.g. a provider-granted reset was redeemed).
	pub fn clear(&mut self, key: &BlockKey) {
		self.entries.remove(key);
	}

	/// Drops expired entries to bound memory in long-lived processes.
	pub fn sweep(&mut self, now_ms: u64) {
		self.entries.retain(|_, &mut until| until > now_ms);
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Evidence, classify_at};

	fn cls(status: u16, body: &str) -> Classification {
		classify_at(&Evidence::http(status, body), 1_000)
	}

	#[test]
	fn budget_curve_and_exhaustion() {
		let mut budget = RetryBudget { max_attempts: 3, ..RetryBudget::default() };
		let c = cls(503, "service unavailable");
		assert_eq!(budget.decide(&c, true, 0.0), RetryDecision::Retry { delay_ms: 500, attempt: 1 });
		assert_eq!(budget.decide(&c, true, 0.0), RetryDecision::Retry {
			delay_ms: 1000,
			attempt:  2,
		});
		assert_eq!(budget.decide(&c, true, 0.0), RetryDecision::Retry {
			delay_ms: 2000,
			attempt:  3,
		});
		assert_eq!(budget.decide(&c, true, 0.0), RetryDecision::GiveUp(GiveUp::BudgetExhausted));
	}

	#[test]
	fn repaired_stale_session_uses_bounded_non_exact_lane() {
		let stale = classify_at(
			&Evidence::http(404, "HTTP 404 Item with id 'rs_message' not found.")
				.with_api(crate::WireApi::OpenAiResponses),
			1_000,
		);
		assert!(stale.kinds.has(Kind::StaleSessionItem));
		let mut budget = RetryBudget::new(1, 0, 0, 1_000);
		assert_eq!(budget.decide(&stale, true, 0.0), RetryDecision::GiveUp(GiveUp::NotRetryable),);
		assert_eq!(budget.decide_repaired(&stale, true, 0.0), RetryDecision::Retry {
			delay_ms: 0,
			attempt:  1,
		},);
		assert_eq!(
			budget.decide_repaired(&stale, true, 0.0),
			RetryDecision::GiveUp(GiveUp::BudgetExhausted),
		);
	}

	#[test]
	fn hint_too_long_fails_fast() {
		let mut budget = RetryBudget::default();
		let headers = [("retry-after", "3600")];
		let c = classify_at(&Evidence::http(429, "slow down").with_headers(&headers), 0);
		assert_eq!(
			budget.decide(&c, true, 0.0),
			RetryDecision::GiveUp(GiveUp::HintTooLong { delay_ms: 3_600_000 })
		);
	}

	#[test]
	fn replay_unsafe_blocks_retry() {
		let mut budget = RetryBudget::default();
		let c = cls(503, "service unavailable");
		assert_eq!(budget.decide(&c, false, 0.0), RetryDecision::GiveUp(GiveUp::NotRetryable));
	}

	#[test]
	fn block_table_scoping() {
		let mut table = BlockTable::new();
		table.block(BlockKey::scoped("cred-1", "spark"), 0, 10_000);
		// Sibling meter unaffected.
		assert_eq!(table.blocked_for_ms(&BlockKey::scoped("cred-1", "chat"), 5_000), None);
		// Whole-credential block gates every scope.
		table.block(BlockKey::credential("cred-1"), 0, 30_000);
		assert_eq!(table.blocked_for_ms(&BlockKey::scoped("cred-1", "chat"), 5_000), Some(25_000));
		assert_eq!(table.earliest_unblock_ms(5_000), Some(10_000));
		table.sweep(40_000);
		assert_eq!(table.earliest_unblock_ms(0), None);
	}

	#[test]
	fn advise_orders_cause_fixes_first() {
		let c = cls(
			400,
			r#"{"error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens"}}"#,
		);
		let plan = advise(&c, &AdviseContext::default());
		assert_eq!(plan[0], Action::ReduceContext);
	}

	#[test]
	fn advise_quota_rotates_with_sibling() {
		let c = cls(
			429,
			r#"{"error":{"code":"usage_limit_reached","message":"The usage limit has been reached"}}"#,
		);
		let ctx = AdviseContext { has_sibling_credential: true, ..AdviseContext::default() };
		let plan = advise(&c, &ctx);
		assert!(matches!(plan[0], Action::RotateCredential { block_ms: 1_800_000 }));
	}

	#[test]
	fn advise_concurrency_sheds_without_rotating() {
		let c = cls(429, "Too many concurrent requests");
		let ctx = AdviseContext { has_sibling_credential: true, ..AdviseContext::default() };
		let plan = advise(&c, &ctx);
		assert!(
			plan
				.iter()
				.all(|a| !matches!(a, Action::RotateCredential { .. }))
		);
		assert!(matches!(plan[0], Action::Backoff { delay_ms: 5_000 }));
	}
}
