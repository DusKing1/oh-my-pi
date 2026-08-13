//! Pre-commit, replay-evidence-driven transport retry on the same route and
//! account.

use std::{
	mem,
	task::{Context, Poll},
};

use futures::future::poll_fn;
use tower::{Layer, Service};

use crate::{
	body::RetryDecision,
	error::{Error, ErrorPhase, RetryAction},
	layer::LayerCall,
};

/// Maximum same-route retries; the overall attempt budget remains
/// authoritative.
#[derive(Clone, Copy, Debug)]
pub struct TransportRetryLayer {
	max_retries: u32,
}
impl TransportRetryLayer {
	/// Creates a same-route retry layer.
	pub const fn new(max_retries: u32) -> Self {
		Self { max_retries }
	}
}

/// Service implementing retry inside account/auth and outside
/// rate/encode/transport.
#[derive(Clone, Debug)]
pub struct TransportRetryService<S> {
	inner:       S,
	max_retries: u32,
}
impl<S> Layer<S> for TransportRetryLayer {
	type Service = TransportRetryService<S>;

	fn layer(&self, inner: S) -> Self::Service {
		TransportRetryService { inner, max_retries: self.max_retries }
	}
}

impl<S, R> Service<LayerCall<R>> for TransportRetryService<S>
where
	S: Service<LayerCall<R>, Error = Error> + Clone,
	R: Clone,
{
	type Error = Error;
	type Response = S::Response;

	type Future = impl Future<Output = Result<S::Response, Error>>;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, request: LayerCall<R>) -> Self::Future {
		// Move the exact instance whose readiness was observed into the future; leave a
		// fresh clone for later callers.
		let replacement = self.inner.clone();
		let mut service = mem::replace(&mut self.inner, replacement);
		let max_retries = self.max_retries;
		async move {
			let mut retry_index = 0;
			loop {
				request.context.checkpoint(ErrorPhase::Readiness)?;
				request.context.reserve_attempt()?;
				request.context.clear_body_evidence();
				let result = service.call(request.clone()).await;
				let mut error = match result {
					Ok(response) => return Ok(response),
					Err(error) => error,
				};
				request.context.merge_receipt(&error.receipt);
				if let Some(attempt) = error.receipt.attempts.last() {
					request.context.set_body_evidence(attempt.body);
				}
				let delay = match &error.action {
					RetryAction::SameRoute { after }
						if !error.committed && retry_index < max_retries =>
					{
						*after
					},
					_ => {
						request.context.finalize_error(&mut error);
						return Err(error);
					},
				};
				let replay_safe = request
					.context
					.body_evidence()
					.is_some_and(|evidence| evidence.retry_decision == RetryDecision::Allow);
				if !replay_safe {
					error.action = RetryAction::Never;
					request.context.finalize_error(&mut error);
					return Err(error);
				}
				request.context.checkpoint(ErrorPhase::Readiness)?;
				if !delay.is_zero() {
					wait_retry_delay(request.context.clone(), delay).await?;
				}
				retry_index += 1;
				poll_fn(|cx| service.poll_ready(cx)).await?;
			}
		}
	}
}

async fn wait_retry_delay(
	context: crate::layer::ExecutionContext,
	delay: std::time::Duration,
) -> Result<(), Error> {
	let remaining = context
		.budget()
		.max_elapsed
		.map(|limit| limit.saturating_sub(context.elapsed()));
	match remaining {
		Some(remaining) => tokio::select! {
			_ = tokio::time::sleep(delay) => context.checkpoint(ErrorPhase::Readiness),
			_ = tokio::time::sleep(remaining) => context.checkpoint(ErrorPhase::Readiness),
			_ = wait_cancelled(context.clone()) => context.checkpoint(ErrorPhase::Readiness),
		},
		None => tokio::select! {
			_ = tokio::time::sleep(delay) => context.checkpoint(ErrorPhase::Readiness),
			_ = wait_cancelled(context.clone()) => context.checkpoint(ErrorPhase::Readiness),
		},
	}
}

async fn wait_cancelled(context: crate::layer::ExecutionContext) {
	while !context.is_cancelled() {
		tokio::task::yield_now().await;
	}
}

#[cfg(test)]
mod tests {
	use std::{
		sync::{
			Arc,
			atomic::{AtomicUsize, Ordering},
		},
		task::{Context, Poll},
		time::Duration,
	};

	use futures::future::{Ready, ready};
	use tower::Service;

	use super::TransportRetryService;
	use crate::{
		body::{AttemptBodyEvidence, Replayability, RetryDecision, RetryDecisionReason},
		error::{Error, ErrorKind, ErrorPhase, RetryAction},
		layer::{ExecutionContext, LayerCall},
		receipt::{
			AttemptOutcome, AttemptReceipt, Cost, ExecutionBudget, ExecutionReceipt, ProviderEvidence,
			Usage,
		},
	};

	#[derive(Clone)]
	struct Failing {
		calls: Arc<AtomicUsize>,
		body:  Option<AttemptBodyEvidence>,
	}
	impl Service<LayerCall<()>> for Failing {
		type Error = Error;
		type Future = Ready<Result<(), Error>>;
		type Response = ();

		fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Error>> {
			Poll::Ready(Ok(()))
		}

		fn call(&mut self, _: LayerCall<()>) -> Self::Future {
			let index = self.calls.fetch_add(1, Ordering::SeqCst) as u32;
			let mut receipt = ExecutionReceipt::default();
			if let Some(body) = self.body {
				receipt.record_attempt(AttemptReceipt {
					index,
					hidden: false,
					provider: None,
					route: None,
					account: None,
					principal: None,
					body,
					outcome: AttemptOutcome::FailedPreCommit,
					usage: Usage { input_tokens: 1, ..Usage::default() },
					cost: Cost::from_micro_usd(1),
					provider_evidence: ProviderEvidence::default(),
					elapsed: Duration::ZERO,
				});
			}
			ready(Err(Error::new(
				ErrorKind::Connectivity,
				ErrorPhase::Connecting,
				RetryAction::SameRoute { after: Duration::ZERO },
				receipt,
			)))
		}
	}

	fn context() -> ExecutionContext {
		let mut budget = ExecutionBudget::default();
		budget.max_attempts = 3;
		ExecutionContext::new(budget)
	}

	#[tokio::test]
	async fn missing_body_evidence_suppresses_retry() {
		let calls = Arc::new(AtomicUsize::new(0));
		let mut service = TransportRetryService {
			inner:       Failing { calls: calls.clone(), body: None },
			max_retries: 2,
		};
		futures::future::poll_fn(|cx| service.poll_ready(cx))
			.await
			.unwrap();
		let error = service
			.call(LayerCall { payload: (), context: context() })
			.await
			.unwrap_err();
		assert_eq!(error.action, RetryAction::Never);
		assert_eq!(calls.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn consumed_one_shot_evidence_suppresses_retry() {
		let calls = Arc::new(AtomicUsize::new(0));
		let body = AttemptBodyEvidence {
			opened:         true,
			consumed:       true,
			replayability:  Replayability::OneShot,
			retry_decision: RetryDecision::Suppress,
			reason:         RetryDecisionReason::ConsumedOneShot,
		};
		let mut service = TransportRetryService {
			inner:       Failing { calls: calls.clone(), body: Some(body) },
			max_retries: 2,
		};
		futures::future::poll_fn(|cx| service.poll_ready(cx))
			.await
			.unwrap();
		let error = service
			.call(LayerCall { payload: (), context: context() })
			.await
			.unwrap_err();
		assert_eq!(error.action, RetryAction::Never);
		assert_eq!(calls.load(Ordering::SeqCst), 1);
	}

	#[derive(Clone)]
	struct FailThenSuccess {
		calls: Arc<AtomicUsize>,
		body:  AttemptBodyEvidence,
	}
	impl Service<LayerCall<()>> for FailThenSuccess {
		type Error = Error;
		type Future = Ready<Result<(), Error>>;
		type Response = ();

		fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Error>> {
			Poll::Ready(Ok(()))
		}

		fn call(&mut self, _: LayerCall<()>) -> Self::Future {
			if self.calls.fetch_add(1, Ordering::SeqCst) > 0 {
				return ready(Ok(()));
			}
			let mut receipt = ExecutionReceipt::default();
			receipt.record_attempt(AttemptReceipt {
				index:             0,
				hidden:            false,
				provider:          None,
				route:             None,
				account:           None,
				principal:         None,
				body:              self.body,
				outcome:           AttemptOutcome::FailedPreCommit,
				usage:             Usage { input_tokens: 1, ..Usage::default() },
				cost:              Cost::from_micro_usd(1),
				provider_evidence: ProviderEvidence::default(),
				elapsed:           Duration::ZERO,
			});
			ready(Err(Error::new(
				ErrorKind::Connectivity,
				ErrorPhase::Connecting,
				RetryAction::SameRoute { after: Duration::ZERO },
				receipt,
			)))
		}
	}

	fn replayable() -> AttemptBodyEvidence {
		AttemptBodyEvidence {
			opened:         true,
			consumed:       true,
			replayability:  Replayability::Replayable,
			retry_decision: RetryDecision::Allow,
			reason:         RetryDecisionReason::ReplayableSource,
		}
	}

	#[tokio::test]
	async fn fail_then_success_retains_prior_attempt_once() {
		let calls = Arc::new(AtomicUsize::new(0));
		let context = context();
		let mut service = TransportRetryService {
			inner:       FailThenSuccess { calls, body: replayable() },
			max_retries: 1,
		};
		futures::future::poll_fn(|cx| service.poll_ready(cx))
			.await
			.unwrap();
		service
			.call(LayerCall { payload: (), context: context.clone() })
			.await
			.unwrap();
		assert_eq!(context.receipt().attempts.len(), 1);
		assert_eq!(context.receipt().usage.input_tokens, 1);
		assert_eq!(context.receipt().cost.micro_usd, 1);
	}

	#[tokio::test]
	async fn fail_then_fail_returns_ordered_deduplicated_receipt() {
		let calls = Arc::new(AtomicUsize::new(0));
		let mut service = TransportRetryService {
			inner:       Failing { calls, body: Some(replayable()) },
			max_retries: 1,
		};
		futures::future::poll_fn(|cx| service.poll_ready(cx))
			.await
			.unwrap();
		let error = service
			.call(LayerCall { payload: (), context: context() })
			.await
			.unwrap_err();
		assert_eq!(
			error
				.receipt
				.attempts
				.iter()
				.map(|attempt| attempt.index)
				.collect::<Vec<_>>(),
			vec![0, 1]
		);
		assert_eq!(error.receipt.usage.input_tokens, 2);
		assert_eq!(error.receipt.cost.micro_usd, 2);
	}
}
