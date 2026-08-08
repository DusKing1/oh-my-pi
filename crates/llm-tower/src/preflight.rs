//! Usage and quota admission before a provider attempt leaves the process.
//!
//! The gate emits typed [`TurnError`] frames for rejected requests. This is
//! deliberately unlike pi's `"Usage preflight blocked:"` prose sentinel
//! (report 13 #2): downstream code can inspect the protocol kind instead of
//! matching a magic detail prefix. Rejection details therefore remain purely
//! human-facing.

use std::{
	future::Future,
	sync::Arc,
	task::{Context, Poll},
};

use futures::{
	Stream,
	future::{self, Either},
};
use omp_proto::inference::v1::{TurnError, TurnEvent, turn_error, turn_event};
use tower::{Layer, Service, ServiceExt};

use crate::{SingleTurn, envelope::TurnRequestEnvelope, single_turn};

/// Synchronous usage/quota authority consulted before provider dispatch.
pub trait UsageOracle: Send + Sync + 'static {
	/// Returns the current admission verdict for `model`.
	fn admit(&self, model: &str) -> Admission;
}

/// Result of consulting a [`UsageOracle`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Admission {
	/// Dispatch may proceed.
	Allow,
	/// The current quota does not admit the request.
	DenyQuota {
		/// Human-facing explanation of the quota rejection.
		detail:         String,
		/// Delay advised before another admission attempt.
		retry_after_ms: u64,
	},
	/// Authentication or account state does not admit the request.
	DenyAuth {
		/// Human-facing explanation of the authentication rejection.
		detail: String,
	},
	/// The oracle could not determine whether the request is admissible.
	Unknown {
		/// Human-facing explanation of why admission could not be determined.
		detail: String,
	},
}

/// Policy for the usage preflight gate.
#[derive(Clone, Debug)]
pub struct PreflightConfig {
	/// Reject requests when the oracle cannot provide a verdict.
	pub fail_closed: bool,
}

impl Default for PreflightConfig {
	fn default() -> Self {
		Self { fail_closed: true }
	}
}

/// [`Layer`] producing usage-gated [`Preflight`] services.
#[derive(Clone)]
pub struct PreflightLayer {
	oracle: Arc<dyn UsageOracle>,
	config: Arc<PreflightConfig>,
}

impl PreflightLayer {
	/// Creates a layer backed by `oracle` and `config`.
	pub fn new(oracle: Arc<dyn UsageOracle>, config: PreflightConfig) -> Self {
		Self { oracle, config: Arc::new(config) }
	}
}

impl<S> Layer<S> for PreflightLayer {
	type Service = Preflight<S>;

	fn layer(&self, inner: S) -> Self::Service {
		Preflight { inner, oracle: Arc::clone(&self.oracle), config: Arc::clone(&self.config) }
	}
}

/// Usage-gated wrapper around an inference turn service.
#[derive(Clone)]
pub struct Preflight<S> {
	inner:  S,
	oracle: Arc<dyn UsageOracle>,
	config: Arc<PreflightConfig>,
}

impl<S> Preflight<S> {
	/// Wraps `inner` with an admission oracle and policy.
	pub fn new(inner: S, oracle: Arc<dyn UsageOracle>, config: PreflightConfig) -> Self {
		Self { inner, oracle, config: Arc::new(config) }
	}
}

impl<S, St, R> Service<R> for Preflight<S>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Clone + Send + 'static,
	S::Future: Send,
	S::Error: Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	type Error = S::Error;
	type Response = Either<SingleTurn, St>;

	type Future = impl Future<Output = Result<Self::Response, S::Error>> + Send;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), S::Error>> {
		// Request-dependent short-circuit layer: reserving inner readiness
		// here would leak the reservation on every denial (readiness-
		// sensitive inner services like concurrency limits reserve a slot in
		// poll_ready). Inner readiness is driven in the dispatch branch.
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, req: R) -> Self::Future {
		let model = req
			.request()
			.params
			.as_ref()
			.map_or("", |params| params.model.as_str());
		let admission = self.oracle.admit(model);

		let rejection = match admission {
			Admission::Allow => None,
			Admission::DenyQuota { detail, retry_after_ms } => {
				Some((turn_error::Kind::RateLimited, detail, retry_after_ms))
			},
			Admission::DenyAuth { detail } => Some((turn_error::Kind::Auth, detail, 0)),
			Admission::Unknown { detail } if self.config.fail_closed => {
				Some((turn_error::Kind::Auth, format!("preflight unavailable: {detail}"), 0))
			},
			Admission::Unknown { .. } => None,
		};

		if let Some((kind, detail, retry_after_ms)) = rejection {
			return Either::Left(future::ready(Ok(Either::Left(single_error(
				kind,
				detail,
				retry_after_ms,
			)))));
		}

		let clone = self.inner.clone();
		let mut inner = std::mem::replace(&mut self.inner, clone);
		Either::Right(async move {
			let stream = inner.ready().await?.call(req).await?;
			Ok(Either::Right(stream))
		})
	}
}

fn single_error(kind: turn_error::Kind, detail: String, retry_after_ms: u64) -> SingleTurn {
	let event = TurnEvent {
		event: Some(turn_event::Event::Error(TurnError {
			kind: kind as i32,
			detail,
			retry_after_ms,
			..TurnError::default()
		})),
	};
	single_turn(event)
}
