//! Explicit action boundary around admission, account, auth, and same-route
//! retry.

use std::{
	mem,
	task::{Context, Poll},
};

use futures::future::poll_fn;
use tower::{Layer, Service};

use crate::{
	body::RetryDecision,
	error::{Error, RetryAction},
	layer::{AttemptAction, LayerCall},
};

/// Marks the complete attempt sub-stack without rebuilding it per call.
#[derive(Clone, Copy, Debug, Default)]
pub struct AttemptLayer;
/// Service routing only refresh and account-rotation actions back through the
/// inner attempt stack.
#[derive(Clone, Debug)]
pub struct AttemptService<S> {
	inner: S,
}
impl<S> Layer<S> for AttemptLayer {
	type Service = AttemptService<S>;

	fn layer(&self, inner: S) -> Self::Service {
		AttemptService { inner }
	}
}
impl<S, R> Service<LayerCall<R>> for AttemptService<S>
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
		let replacement = self.inner.clone();
		let mut service = mem::replace(&mut self.inner, replacement);
		async move {
			let mut reentries = 0_u32;
			request.context.set_attempt_action(AttemptAction::Initial);
			loop {
				request.context.clear_body_evidence();
				let result = service.call(request.clone()).await;
				let mut error = match result {
					Ok(response) => return Ok(response),
					Err(error) => error,
				};
				if let Some(attempt) = error.receipt.attempts.last() {
					request.context.set_body_evidence(attempt.body);
				}
				let replay_safe = request
					.context
					.body_evidence()
					.is_some_and(|evidence| evidence.retry_decision == RetryDecision::Allow);
				if error.committed || !replay_safe {
					error.action = RetryAction::Never;
					request.context.finalize_error(&mut error);
					return Err(error);
				}
				let previous_account = error
					.receipt
					.attempts
					.last()
					.and_then(|attempt| attempt.account.clone());
				let action = match &error.action {
					RetryAction::RefreshCredential => {
						AttemptAction::RefreshCredential { previous_account }
					},
					RetryAction::RotateAccount => AttemptAction::RotateAccount { previous_account },
					RetryAction::SameRoute { .. }
					| RetryAction::ReselectRoute
					| RetryAction::ReseedSession
					| RetryAction::SemanticRetry
					| RetryAction::Never => {
						request.context.finalize_error(&mut error);
						return Err(error);
					},
				};
				let mut hidden_receipt = error.receipt.clone();
				for attempt in &mut hidden_receipt.attempts {
					attempt.hidden = true;
				}
				request.context.merge_receipt(&hidden_receipt);
				if reentries >= request.context.budget().max_attempts.saturating_sub(1) {
					error.action = RetryAction::Never;
					request.context.finalize_error(&mut error);
					return Err(error);
				}
				reentries += 1;
				request.context.set_attempt_action(action);
				poll_fn(|cx| service.poll_ready(cx)).await?;
			}
		}
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
	use parking_lot::Mutex;
	use tower::Service;

	use super::AttemptService;
	use crate::{
		body::{AttemptBodyEvidence, Replayability, RetryDecision, RetryDecisionReason},
		error::{Error, ErrorKind, ErrorPhase, RetryAction},
		id::AccountId,
		layer::{AttemptAction, ExecutionContext, LayerCall},
		receipt::{
			AttemptOutcome, AttemptReceipt, Cost, ExecutionBudget, ExecutionReceipt, ProviderEvidence,
			Usage,
		},
	};

	#[derive(Clone)]
	struct RefreshOnce {
		calls:   Arc<AtomicUsize>,
		actions: Arc<Mutex<Vec<AttemptAction>>>,
	}
	impl Service<LayerCall<()>> for RefreshOnce {
		type Error = Error;
		type Future = Ready<Result<(), Error>>;
		type Response = ();

		fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Error>> {
			Poll::Ready(Ok(()))
		}

		fn call(&mut self, request: LayerCall<()>) -> Self::Future {
			self.actions.lock().push(request.context.attempt_action());
			if self.calls.fetch_add(1, Ordering::SeqCst) > 0 {
				return ready(Ok(()));
			}
			let mut receipt = ExecutionReceipt::default();
			receipt.record_attempt(AttemptReceipt {
				index:             0,
				hidden:            false,
				provider:          None,
				route:             None,
				account:           Some(AccountId::from("account")),
				principal:         None,
				body:              AttemptBodyEvidence {
					opened:         true,
					consumed:       true,
					replayability:  Replayability::Replayable,
					retry_decision: RetryDecision::Allow,
					reason:         RetryDecisionReason::ReplayableSource,
				},
				outcome:           AttemptOutcome::FailedPreCommit,
				usage:             Usage::default(),
				cost:              Cost::default(),
				provider_evidence: ProviderEvidence::default(),
				elapsed:           Duration::ZERO,
			});
			ready(Err(Error::new(
				ErrorKind::Authentication,
				ErrorPhase::Authentication,
				RetryAction::RefreshCredential,
				receipt,
			)))
		}
	}

	#[tokio::test]
	async fn refresh_reenters_with_same_previous_account_action() {
		let calls = Arc::new(AtomicUsize::new(0));
		let actions = Arc::new(Mutex::new(Vec::new()));
		let context =
			ExecutionContext::new(ExecutionBudget { max_attempts: 2, ..ExecutionBudget::default() });
		let mut service =
			AttemptService { inner: RefreshOnce { calls: calls.clone(), actions: actions.clone() } };
		futures::future::poll_fn(|cx| service.poll_ready(cx))
			.await
			.unwrap();
		service
			.call(LayerCall { payload: (), context: context.clone() })
			.await
			.unwrap();
		assert_eq!(calls.load(Ordering::SeqCst), 2);
		assert_eq!(actions.lock()[0], AttemptAction::Initial);
		assert_eq!(actions.lock()[1], AttemptAction::RefreshCredential {
			previous_account: Some(AccountId::from("account")),
		});
		assert_eq!(context.receipt().attempts.len(), 1);
		assert!(context.receipt().attempts[0].hidden);
	}

	#[derive(Clone)]
	struct RefreshTwice {
		calls: Arc<AtomicUsize>,
	}
	impl Service<LayerCall<()>> for RefreshTwice {
		type Error = Error;
		type Future = Ready<Result<(), Error>>;
		type Response = ();

		fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Error>> {
			Poll::Ready(Ok(()))
		}

		fn call(&mut self, _: LayerCall<()>) -> Self::Future {
			let index = self.calls.fetch_add(1, Ordering::SeqCst) as u32;
			let mut receipt = ExecutionReceipt::default();
			receipt.record_attempt(AttemptReceipt {
				index,
				hidden: false,
				provider: None,
				route: None,
				account: Some(AccountId::from("account")),
				principal: None,
				body: AttemptBodyEvidence {
					opened:         true,
					consumed:       true,
					replayability:  Replayability::Replayable,
					retry_decision: RetryDecision::Allow,
					reason:         RetryDecisionReason::ReplayableSource,
				},
				outcome: AttemptOutcome::FailedPreCommit,
				usage: Usage { input_tokens: 1, ..Usage::default() },
				cost: Cost::from_micro_usd(1),
				provider_evidence: ProviderEvidence::default(),
				elapsed: Duration::ZERO,
			});
			ready(Err(Error::new(
				ErrorKind::Authentication,
				ErrorPhase::Authentication,
				RetryAction::RefreshCredential,
				receipt,
			)))
		}
	}

	#[tokio::test]
	async fn fail_then_fail_merges_attempts_and_charges_once() {
		let calls = Arc::new(AtomicUsize::new(0));
		let context =
			ExecutionContext::new(ExecutionBudget { max_attempts: 2, ..ExecutionBudget::default() });
		let mut service = AttemptService { inner: RefreshTwice { calls: calls.clone() } };
		futures::future::poll_fn(|cx| service.poll_ready(cx))
			.await
			.unwrap();
		let error = service
			.call(LayerCall { payload: (), context })
			.await
			.unwrap_err();
		assert_eq!(calls.load(Ordering::SeqCst), 2);
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
