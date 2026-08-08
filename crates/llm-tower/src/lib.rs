#![feature(impl_trait_in_assoc_type)]
#![feature(type_alias_impl_trait)]
#![allow(
	incomplete_features,
	reason = "checked type aliases express TAIT bounds for allocation-free service futures"
)]
#![feature(checked_type_aliases)]

//! Tower middleware stack for `omp.inference.v1` provider attempts.
//!
//! Every layer here wraps the same boundary as
//! [`recovery::Recovery`]: a `Service<TurnRequest>`
//! returning a stream of `TurnEvent`s — the turn coordinator's internal,
//! half-closed "one provider attempt in, event stream out" contract, NOT
//! the bidirectional `Turn` RPC (the coordinator owns the client half).
//!
//! ALL of these layers sit BELOW the turn coordinator's commit and
//! idempotency machinery: frames at this altitude are uncommitted attempt
//! results (a terminal `Outcome` here has not advanced any revision).
//! That is what makes suppression and re-dispatch legal at all — above
//! commit, a replayed `turn_id` replays the committed outcome instead.
//!
//! Layer altitude, outermost first (see
//! `.research/llm-errors/tower/00-layer-stack.md` for the derivation):
//!
//! 1. [`preflight`] — usage/quota admission before any bytes leave.
//! 2. [`recovery::Recovery`] — terminal-error retry/normalize.
//! 3. [`select`] — credential selection, sticky pins, scoped blocks. Below it
//!    the request travels as [`select::Routed`] (lease beside the payload,
//!    never inside it).
//! 4. [`refresh`] — OAuth refresh: proactive skew, reactive, single-flight
//!    owned by the broker behind the trait.
//! 5. [`learn`] — sticky endpoint+model capability fallback (strip the rejected
//!    feature, remember, retry).
//! 6. [`resample`] — empty-success and loop re-sampling. Types on its own
//!    pre-commit [`resample::AttemptEvent`] boundary, so it cannot be mounted
//!    over a committed `TurnEvent` stream.
//! 7. [`admission`] — bounded in-flight attempts per pool; permits are held for
//!    the whole stream.
//! 8. [`timeout`] — phase deadlines (connect / first event / idle) with
//!    cancellation provenance. Innermost, directly over the adapter.
//!
//! [`tap`] observes at any altitude without participating in it.
//!
//! Shared conventions:
//! - Verdicts come from [`omp_llm_error::classify`]; layers NEVER match
//!   provider prose themselves.
//! - Pre-wire rejections are typed `TurnError` frames, never prose sentinels.
//! - Only [`omp_llm_error::Classification::retryable_exact_request`] justifies
//!   re-sending identical bytes.

pub mod admission;
pub mod audio;
pub mod cache;
pub mod codex_websocket;
pub mod dialect;
pub mod envelope;
pub mod learn;
pub mod preflight;
pub mod provider;
pub mod recovery;
pub mod refresh;
pub mod resample;
pub mod select;
/// Facet middleware, request routing, rotation, metering, capability checks,
/// and combinators.
pub mod stack;
pub mod tap;
pub mod timeout;

#[doc(hidden)]
pub mod testing;

pub use recovery::TurnStream;

/// One synthetic terminal frame as a concrete, unboxed stream.
///
/// Rejection and deadline branches return this instead of an erased
/// [`TurnStream`], so short-circuit paths stay allocation-free.
pub type SingleTurn =
	futures::stream::Once<std::future::Ready<omp_proto::inference::v1::TurnEvent>>;

/// Wraps one synthetic frame in a [`SingleTurn`] stream.
#[must_use]
pub fn single_turn(event: omp_proto::inference::v1::TurnEvent) -> SingleTurn {
	futures::stream::once(std::future::ready(event))
}
