//! Classification, metadata extraction, and recovery policy for LLM
//! provider errors.
//!
//! Providers disagree about everything: envelope shape, error codes,
//! which HTTP status carries which meaning, whether a body exists at all,
//! and what language the quota message is in. This crate turns that mess
//! into one structured verdict plus an actionable recovery plan, with a
//! pattern corpus distilled from years of multi-provider production
//! traffic and a full-history mine of real observed error strings.
//!
//! # Layers
//!
//! 1. **Parse** — [`envelope`] walks any known error-body shape (nested
//!    `OpenAI` `.error` chains, Anthropic error frames, Google double-wrapped
//!    JSON, gRPC status words, HTML from proxies).
//! 2. **Classify** — [`classify`] merges structural evidence (codes, status
//!    words, HTTP status) with the prose corpus in [`patterns`] into a
//!    [`Kinds`] set plus extracted metadata ([`Classification`]): retry timing,
//!    rejected feature, rate-limit lane, OAuth triage, evidence [`Fidelity`].
//! 3. **Policy** — [`policy`] turns a verdict into an ordered [`Advice`] plan
//!    and provides the stateful primitives: [`RetryBudget`] (jittered
//!    exponential with provider-hint override) and [`BlockTable`] (scoped
//!    credential blocks).
//! 4. **Middleware** — `omp_llm_tower::recovery::Recovery` applies the whole
//!    stack as a tower [`Service`] over the `omp.inference.v1` turn protocol:
//!    terminal `TurnError` frames are classified, retried with honest `Attempt`
//!    frames, and normalized.
//!
//! [`Service`]: https://docs.rs/tower/latest/tower/trait.Service.html
//!
//! # Example
//!
//! ```
//! use omp_llm_error::{Evidence, Kind, classify_at, policy};
//!
//! let ev = Evidence::http(
//! 	429,
//! 	r#"{"error":{"code":"usage_limit_reached","message":"The usage limit has been reached"}}"#,
//! );
//! let cls = classify_at(&ev, 0);
//! assert!(cls.kinds.has(Kind::UsageLimit));
//! assert!(cls.rate_limit.unwrap().rotate); // consume credential, don't spin
//!
//! let ctx = policy::AdviseContext { has_sibling_credential: true, ..Default::default() };
//! let plan = policy::advise(&cls, &ctx);
//! assert!(matches!(plan[0], policy::Action::RotateCredential { .. }));
//! ```

pub mod envelope;
pub mod extract;
pub mod oauth;
pub mod patterns;
pub mod policy;
pub mod rate_limit;

mod classify;
mod evidence;
mod kind;

pub use classify::{Classification, Feature, Fidelity, classify, classify_at};
pub use evidence::{Evidence, Phase, WireApi};
pub use extract::RetryHint;
pub use kind::{Kind, Kinds};
pub use oauth::OAuthFailure;
pub use policy::{
	Action, Advice, AdviseContext, BlockKey, BlockTable, GiveUp, RetryBudget, RetryDecision, advise,
};
pub use rate_limit::{RateLimit, RateLimitReason};
