//! Transactional semantic retries whose provisional output is owned only by
//! `OutputGate`.

use std::{
	mem,
	task::{Context, Poll},
};

use futures::{StreamExt, future::poll_fn, stream};
use tower::{Layer, Service};

use crate::{
	body::RetryDecision,
	codec::{HandshakenResponse, RawEvent},
	error::{Error, ErrorKind, ErrorPhase, RetryAction},
	gate::{GateCondition, GateFinish, GateProgress, OutputGate, SecureGateSpool},
	layer::LayerCall,
	receipt::AttemptOutcome,
};

/// Selects whether an operation requires transactional semantic validation.
pub trait SemanticPolicy<R>: Clone + Send + 'static {
	/// Returns the gate condition, or `None` for ordinary first-event commit
	/// semantics.
	fn condition(&self, request: &R) -> Option<GateCondition>;
	/// Maximum hidden semantic retries; the overall attempt budget remains
	/// authoritative.
	fn max_retries(&self, request: &R) -> u32;
	/// Creates a fresh caller-explicit secure spool for one hidden attempt.
	///
	/// Returning `None` keeps the gate memory-only; the gate never creates or
	/// discovers a spool.
	fn secure_spool(&self, _request: &R) -> Result<Option<Box<dyn SecureGateSpool>>, Error> {
		Ok(None)
	}
}

/// Adds transactional semantic validation.
#[derive(Clone, Debug)]
pub struct SemanticLayer<P> {
	policy: P,
}
impl<P> SemanticLayer<P> {
	/// Creates a semantic layer.
	pub const fn new(policy: P) -> Self {
		Self { policy }
	}
}
/// Semantic-attempt service.
#[derive(Clone, Debug)]
pub struct SemanticService<S, P> {
	inner:  S,
	policy: P,
}
impl<S, P: Clone> Layer<S> for SemanticLayer<P> {
	type Service = SemanticService<S, P>;

	fn layer(&self, inner: S) -> Self::Service {
		SemanticService { inner, policy: self.policy.clone() }
	}
}

impl<S, P, R> Service<LayerCall<R>> for SemanticService<S, P>
where
	S: Service<LayerCall<R>, Response = HandshakenResponse, Error = Error> + Clone,
	P: SemanticPolicy<R>,
	R: Clone,
{
	type Error = Error;
	type Response = HandshakenResponse;

	type Future = impl Future<Output = Result<HandshakenResponse, Error>>;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, request: LayerCall<R>) -> Self::Future {
		let replacement = self.inner.clone();
		let mut service = mem::replace(&mut self.inner, replacement);
		let condition = self.policy.condition(&request.payload);
		let max_retries = self.policy.max_retries(&request.payload);
		let policy = self.policy.clone();
		async move {
			let Some(condition) = condition else {
				return service.call(request).await;
			};
			let mut semantic_retry = 0;
			loop {
				request.context.clear_body_evidence();
				request.context.begin_response_attempt();
				let mut response = service.call(request.clone()).await?;
				request.context.set_body_evidence(response.body.evidence());
				let Some(mut events) = response.events.take() else {
					if response.realtime.is_some() {
						return Ok(response);
					}
					return Err(Error::new(
						ErrorKind::ProviderContractMismatch,
						ErrorPhase::Recovery,
						RetryAction::Never,
						request.context.receipt(),
					));
				};
				let receipt = request.context.receipt();
				let mut gate = match policy.secure_spool(&request.payload)? {
					Some(spool) => OutputGate::with_secure_spool(
						condition.clone(),
						request.context.budget().max_provisional_bytes,
						spool,
						receipt,
					),
					None => OutputGate::with_receipt(
						condition.clone(),
						request.context.budget().max_provisional_bytes,
						receipt,
					),
				};
				let mut visible = Vec::new();
				let mut committed = false;
				let mut failure = None;
				let mut terminal = None;
				while let Some(item) = events.next().await {
					match item {
						Ok(RawEvent::Chat(event)) => {
							match gate.push(event, &mut |event| visible.push(Ok(RawEvent::Chat(event)))) {
								Ok(GateProgress::Committed { .. } | GateProgress::PassThrough) => {
									committed = true;
									break;
								},
								Ok(GateProgress::Rejected) => break,
								Ok(GateProgress::Provisional) => {},
								Err(error) => {
									failure = Some(error);
									break;
								},
							}
						},
						Ok(completion @ RawEvent::Completion(_)) => {
							terminal = Some(completion);
							if request.context.structured_output_valid()
								&& matches!(gate.condition(), GateCondition::ValidStructuredOutput)
							{
								match gate.mark_structured_output_valid(&mut |event| {
									visible.push(Ok(RawEvent::Chat(event)))
								}) {
									Ok(GateProgress::Committed { .. }) => committed = true,
									Ok(_) => {
										failure = Some(Error::new(
											ErrorKind::InternalInvariant,
											ErrorPhase::Recovery,
											RetryAction::Never,
											request.context.receipt(),
										))
									},
									Err(error) => failure = Some(error),
								}
							}
							break;
						},
						Ok(_) => {
							failure = Some(Error::new(
								ErrorKind::ProviderContractMismatch,
								ErrorPhase::Recovery,
								RetryAction::Never,
								request.context.receipt(),
							));
							break;
						},
						Err(error) => {
							request.context.set_body_evidence(response.body.evidence());
							if let Some(attempt) = error.receipt.attempts.last() {
								request.context.set_body_evidence(attempt.body);
							}
							failure = Some(gate.fail(error));
							break;
						},
					}
				}
				request.context.merge_receipt(gate.receipt());
				if committed {
					if let Some(terminal) = terminal.take() {
						visible.push(Ok(terminal));
					}
				}
				if committed {
					response.events = Some(Box::pin(stream::iter(visible).chain(events)));
					return Ok(response);
				}
				if failure.is_none() {
					match gate.finish(&mut |event| visible.push(Ok(RawEvent::Chat(event))))? {
						GateFinish::Committed { .. } | GateFinish::AlreadyCommitted => {
							if let Some(terminal) = terminal.take() {
								visible.push(Ok(terminal));
							}
							response.events = Some(Box::pin(stream::iter(visible)));
							return Ok(response);
						},
						GateFinish::Unsatisfied(_) => {},
					}
				}
				request.context.with_receipt(|receipt| {
					if let Some(attempt) = receipt.attempts.last_mut() {
						attempt.hidden = true;
						attempt.outcome = AttemptOutcome::RejectedSemantic;
					}
				});
				request.context.set_body_evidence(response.body.evidence());
				let mut error = failure.unwrap_or_else(|| {
					Error::new(
						ErrorKind::ToolNonCompliance,
						ErrorPhase::Recovery,
						RetryAction::SemanticRetry,
						gate.receipt().clone(),
					)
				});
				error.committed = false;
				let replay_safe = request
					.context
					.body_evidence()
					.is_some_and(|evidence| evidence.retry_decision == RetryDecision::Allow);
				let retryable_action =
					matches!(&error.action, RetryAction::SemanticRetry | RetryAction::SameRoute { .. });
				if semantic_retry >= max_retries || !replay_safe || !retryable_action {
					error.action = RetryAction::Never;
					request.context.finalize_error(&mut error);
					return Err(error);
				}
				semantic_retry += 1;
				poll_fn(|cx| service.poll_ready(cx)).await?;
			}
		}
	}
}
