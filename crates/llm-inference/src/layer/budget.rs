//! Logical-execution budgets shared by all retries and semantic attempts.

use std::{
	task::{Context, Poll},
	time::Instant,
};

use tower::{Layer, Service};

use crate::{
	call::Call,
	error::{Error, ErrorPhase},
	layer::{ExecutionContext, LayerCall},
};

/// Constructs the outer budget boundary.
#[derive(Clone, Copy, Debug, Default)]
pub struct OverallBudgetLayer;

/// Adds one execution context and enforces its deadline across the inner stack.
#[derive(Clone, Debug)]
pub struct OverallBudgetService<S> {
	inner: S,
}

impl<S> Layer<S> for OverallBudgetLayer {
	type Service = OverallBudgetService<S>;

	fn layer(&self, inner: S) -> Self::Service {
		OverallBudgetService { inner }
	}
}

impl<S> Service<Call> for OverallBudgetService<S>
where
	S: Service<LayerCall<Call>, Error = Error>,
{
	type Error = Error;
	type Response = S::Response;

	type Future = impl Future<Output = Result<Self::Response, Self::Error>>;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, mut call: Call) -> Self::Future {
		if let Some(deadline) = call.deadline {
			let remaining = deadline.saturating_duration_since(Instant::now());
			call.budget.max_elapsed = Some(
				call
					.budget
					.max_elapsed
					.map_or(remaining, |configured| configured.min(remaining)),
			);
		}
		let context = ExecutionContext::new(call.budget.clone());
		let result = context.checkpoint(ErrorPhase::Readiness);
		let future = if result.is_ok() {
			Some(
				self
					.inner
					.call(LayerCall { payload: call, context: context.clone() }),
			)
		} else {
			None
		};
		async move {
			result?;
			let response = future
				.expect("future exists after successful budget checkpoint")
				.await?;
			context.checkpoint(ErrorPhase::Streaming)?;
			Ok(response)
		}
	}
}
