//! Fixed-order request-plane middleware for inference execution.

pub mod account;
pub mod admission;
pub mod answer;
pub mod attempt;
pub mod auth;
pub mod budget;
pub mod encode;
pub mod intent;
pub mod observe;
pub mod operation;
pub mod rate;
pub mod recover;
pub mod retry;
pub mod semantic;
pub mod session;
pub mod stack;

use std::{
	sync::{
		Arc,
		atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
	},
	time::Instant,
};

use parking_lot::Mutex;

use crate::{
	body::AttemptBodyEvidence,
	call::AccountRoutingContext,
	codec::{Cancellation, ProviderMetadataEvent, ProviderStateEvent, ProviderTelemetryEvent},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	receipt::{Cost, ExecutionBudget, ExecutionReceipt},
};

/// Clone-cheap execution state shared by every layer and hidden attempt.
#[derive(Clone)]
pub struct ExecutionContext(Arc<ExecutionState>);

/// Typed outer-attempt action consumed by account and auth policy.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum AttemptAction {
	/// Initial account and credential selection.
	#[default]
	Initial,
	/// Re-enter with the same account and an explicit credential refresh.
	RefreshCredential { previous_account: Option<crate::id::AccountId> },
	/// Re-enter while excluding/cooling the previous account.
	RotateAccount { previous_account: Option<crate::id::AccountId> },
}

/// Session binding identity that account selection must preserve or explicitly
/// reseed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionAffinity {
	/// Principal owning provider-side state.
	pub principal:             crate::id::PrincipalId,
	/// Credential generation captured with the binding.
	pub credential_generation: u64,
	/// Whether ordinary refresh may preserve the binding.
	pub credential_policy:     crate::session::CredentialGenerationPolicy,
}

struct ExecutionState {
	started:                 Instant,
	budget:                  ExecutionBudget,
	cancelled:               AtomicBool,
	committed:               AtomicBool,
	attempts:                AtomicU32,
	input_tokens:            AtomicU64,
	output_tokens:           AtomicU64,
	provisional_bytes:       AtomicU64,
	cost_micro_usd:          Mutex<i128>,
	receipt:                 Mutex<ExecutionReceipt>,
	body:                    Mutex<Option<AttemptBodyEvidence>>,
	account:                 Mutex<Option<AccountRoutingContext>>,
	transport_cancel:        Mutex<Option<Cancellation>>,
	action:                  Mutex<AttemptAction>,
	session_affinity:        Mutex<Option<SessionAffinity>>,
	session_state:           Mutex<Option<crate::session::ServerStateBinding>>,
	provider_state:          Mutex<Vec<ProviderStateEvent>>,
	provider_response_id:    Mutex<Option<omp_core::Str>>,
	session_completion:      Mutex<Option<Arc<dyn session::SessionCompletion>>>,
	attempt_started:         Mutex<Vec<Instant>>,
	structured_output_valid: AtomicBool,
}

impl ExecutionContext {
	/// Creates state for one logical execution; it is constructed once outside
	/// the attempt loop.
	pub fn new(budget: ExecutionBudget) -> Self {
		Self(Arc::new(ExecutionState {
			started: Instant::now(),
			budget,
			cancelled: AtomicBool::new(false),
			committed: AtomicBool::new(false),
			attempts: AtomicU32::new(0),
			input_tokens: AtomicU64::new(0),
			output_tokens: AtomicU64::new(0),
			provisional_bytes: AtomicU64::new(0),
			cost_micro_usd: Mutex::new(0),
			receipt: Mutex::new(ExecutionReceipt::default()),
			body: Mutex::new(None),
			account: Mutex::new(None),
			transport_cancel: Mutex::new(None),
			action: Mutex::new(AttemptAction::Initial),
			session_affinity: Mutex::new(None),
			session_state: Mutex::new(None),
			provider_state: Mutex::new(Vec::new()),
			provider_response_id: Mutex::new(None),
			session_completion: Mutex::new(None),
			attempt_started: Mutex::new(Vec::new()),
			structured_output_valid: AtomicBool::new(false),
		}))
	}

	/// Returns the immutable configured budget.
	pub fn budget(&self) -> &ExecutionBudget {
		&self.0.budget
	}

	/// Returns elapsed wall time.
	pub fn elapsed(&self) -> std::time::Duration {
		self.0.started.elapsed()
	}

	/// Requests cooperative cancellation and wakes the active wire transport.
	pub fn cancel(&self) {
		self.0.cancelled.store(true, Ordering::Release);
		if let Some(cancel) = self.0.transport_cancel.lock().as_ref() {
			cancel.cancel();
		}
	}

	/// Returns whether cancellation was requested.
	pub fn is_cancelled(&self) -> bool {
		self.0.cancelled.load(Ordering::Acquire)
	}

	/// Marks the first ordinary consumer-visible event as committed.
	pub fn commit(&self) {
		self.0.committed.store(true, Ordering::Release);
	}

	/// Replaces exact request-body evidence after a transport acquisition or
	/// handshake.
	pub fn set_body_evidence(&self, evidence: AttemptBodyEvidence) {
		*self.0.body.lock() = Some(evidence);
	}

	/// Returns the latest exact body evidence consumed by retry, rotation,
	/// fallback, and reseed policy.
	pub fn body_evidence(&self) -> Option<AttemptBodyEvidence> {
		*self.0.body.lock()
	}

	/// Clears stale evidence before opening a distinct attempt.
	pub fn clear_body_evidence(&self) {
		*self.0.body.lock() = None;
	}

	/// Stores the canonical non-secret account routing metadata selected for
	/// this execution.
	pub fn set_account_routing(&self, routing: AccountRoutingContext) {
		*self.0.account.lock() = Some(routing);
	}

	/// Returns selected non-secret account routing metadata.
	pub fn account_routing(&self) -> Option<AccountRoutingContext> {
		self.0.account.lock().clone()
	}

	/// Stores the compatible provider-side state selected by session planning.
	pub fn set_session_state(&self, state: Option<crate::session::ServerStateBinding>) {
		*self.0.session_state.lock() = state;
	}

	/// Returns the compatible provider-side state selected for encoding.
	pub fn session_state(&self) -> Option<crate::session::ServerStateBinding> {
		self.0.session_state.lock().clone()
	}

	/// Registers the active transport cancellation handle for propagation.
	pub fn register_transport_cancel(&self, cancel: Cancellation) {
		if self.is_cancelled() {
			cancel.cancel();
		}
		*self.0.transport_cancel.lock() = Some(cancel);
	}

	/// Stores the typed action consumed by account/auth on the next outer
	/// attempt.
	pub fn set_attempt_action(&self, action: AttemptAction) {
		*self.0.action.lock() = action;
	}

	/// Returns the typed action for the current outer attempt.
	pub fn attempt_action(&self) -> AttemptAction {
		self.0.action.lock().clone()
	}

	/// Stores provider-state binding identity before account selection.
	pub fn set_session_affinity(&self, affinity: Option<SessionAffinity>) {
		*self.0.session_affinity.lock() = affinity;
	}

	/// Returns provider-state binding identity for deterministic account
	/// selection.
	pub fn session_affinity(&self) -> Option<SessionAffinity> {
		self.0.session_affinity.lock().clone()
	}

	/// Returns whether ordinary output has committed.
	pub fn is_committed(&self) -> bool {
		self.0.committed.load(Ordering::Acquire)
	}

	/// Returns the number of charged provider/local attempts.
	pub fn attempts(&self) -> u32 {
		self.0.attempts.load(Ordering::Acquire)
	}

	/// Returns charged provisional bytes.
	pub fn provisional_bytes(&self) -> u64 {
		self.0.provisional_bytes.load(Ordering::Acquire)
	}

	/// Returns a secret-free accounting snapshot.
	pub fn receipt(&self) -> ExecutionReceipt {
		self.0.receipt.lock().clone()
	}

	/// Mutates the secret-free receipt under a short non-async critical section.
	pub fn with_receipt<T>(&self, f: impl FnOnce(&mut ExecutionReceipt) -> T) -> T {
		f(&mut self.0.receipt.lock())
	}

	/// Publishes validator success only to the semantic gate for the current
	/// hidden attempt.
	pub(crate) fn mark_structured_output_valid(&self) {
		self
			.0
			.structured_output_valid
			.store(true, Ordering::Release);
	}

	/// Returns whether recovery validated the requested structured output.
	pub(crate) fn structured_output_valid(&self) -> bool {
		self.0.structured_output_valid.load(Ordering::Acquire)
	}

	/// Stages opaque provider state privately until the response reaches true
	/// terminal success.
	pub(crate) fn stage_provider_state(&self, state: ProviderStateEvent) {
		self.0.provider_state.lock().push(state);
	}

	/// Atomically consumes state captured by the successful response.
	pub(crate) fn take_provider_state(&self) -> Vec<ProviderStateEvent> {
		std::mem::take(&mut *self.0.provider_state.lock())
	}

	/// Aborts all uncommitted provider state.
	pub(crate) fn abort_provider_state(&self) {
		self.0.provider_state.lock().clear();
	}

	/// Clears state and metadata retained by a discarded hidden response
	/// attempt.
	pub(crate) fn begin_response_attempt(&self) {
		self.abort_provider_state();
		*self.0.provider_response_id.lock() = None;
		self
			.0
			.structured_output_valid
			.store(false, Ordering::Release);
	}

	/// Registers the private transaction associated with the prepared session
	/// turn.
	pub(crate) fn set_session_completion(
		&self,
		completion: Option<Arc<dyn session::SessionCompletion>>,
	) {
		*self.0.session_completion.lock() = completion;
	}

	/// Streams one recovered canonical event into the private session message
	/// builder.
	pub(crate) fn record_session_event(&self, event: &crate::event::ChatEvent) -> Result<(), Error> {
		if let Some(completion) = self.0.session_completion.lock().as_ref() {
			completion.record_chat_event(event, self)
		} else {
			Ok(())
		}
	}

	/// Atomically commits the private session transaction at true response
	/// success.
	pub(crate) fn commit_session(&self) -> Result<(), Error> {
		let state = self.take_provider_state();
		if let Some(completion) = self.0.session_completion.lock().take() {
			completion.commit(state, &self.receipt(), self)
		} else {
			Ok(())
		}
	}

	/// Aborts the private transaction and drops uncommitted provider-state
	/// evidence.
	pub(crate) fn abort_session(&self) {
		self.abort_session_inner(false);
	}

	/// Aborts the current draft while retaining original preparation for one
	/// deterministic reseed.
	pub(crate) fn abort_session_for_reseed(&self) {
		self.abort_session_inner(true);
	}

	fn abort_session_inner(&self, retain_preparation: bool) {
		self.abort_provider_state();
		if let Some(completion) = self.0.session_completion.lock().take() {
			completion.abort(retain_preparation);
		}
	}

	/// Applies sanitized provider metadata without forwarding it to consumers.
	pub(crate) fn observe_provider_metadata(&self, metadata: ProviderMetadataEvent) {
		if let ProviderMetadataEvent::ResponseId(id) = metadata {
			*self.0.provider_response_id.lock() = Some(id);
		}
	}

	/// Returns the codec-observed response identity, when more exact than the
	/// handshake.
	pub(crate) fn provider_response_id(&self) -> Option<omp_core::Str> {
		self.0.provider_response_id.lock().clone()
	}

	/// Applies typed, secret-free telemetry to the current receipt.
	pub(crate) fn observe_provider_telemetry(&self, telemetry: ProviderTelemetryEvent) {
		self.with_receipt(|receipt| match telemetry {
			ProviderTelemetryEvent::ModelLatency(elapsed) => {
				receipt.timings.streaming = receipt.timings.streaming.max(elapsed)
			},
			ProviderTelemetryEvent::SafetyAssessment { guardrail_latency: Some(elapsed), .. } => {
				receipt.timings.streaming = receipt.timings.streaming.saturating_add(elapsed)
			},
			ProviderTelemetryEvent::SafetyAssessment { guardrail_latency: None, .. } => {},
		});
	}

	/// Merges attempt accounting exactly once, preserving ascending attempt
	/// order.
	pub fn merge_receipt(&self, source: &ExecutionReceipt) {
		let mut destination = self.0.receipt.lock();
		for attempt in &source.attempts {
			if !destination
				.attempts
				.iter()
				.any(|stored| stored.index == attempt.index)
			{
				destination.record_attempt(attempt.clone());
			}
		}
		destination.attempts.sort_by_key(|attempt| attempt.index);
		for adjustment in &source.adjustments {
			if !destination.adjustments.contains(adjustment) {
				destination.adjustments.push(adjustment.clone());
			}
		}
		for recovery in &source.recoveries {
			if !destination.recoveries.contains(recovery) {
				destination.recoveries.push(recovery.clone());
			}
		}
		for staging in &source.staging {
			if !destination.staging.contains(staging) {
				destination.staging.push(staging.clone());
			}
		}
	}

	/// Replaces an error's partial receipt with the deduplicated
	/// logical-execution receipt.
	pub fn finalize_error(&self, error: &mut Error) {
		self.merge_receipt(&error.receipt);
		error.receipt = self.receipt();
	}

	fn error(
		&self,
		kind: ErrorKind,
		phase: ErrorPhase,
		dimension: &'static str,
		limit: u128,
		observed: u128,
	) -> Error {
		let mut error = Error::new(kind, phase, RetryAction::Never, self.receipt());
		error.committed = self.is_committed();
		error.detail = Some(ErrorDetail::Budget { dimension: dimension.into(), limit, observed });
		error
	}

	/// Checks cancellation and elapsed time at a cooperative boundary.
	pub fn checkpoint(&self, phase: ErrorPhase) -> Result<(), Error> {
		if self.is_cancelled() {
			let mut error =
				Error::new(ErrorKind::Cancelled, phase, RetryAction::Never, self.receipt());
			error.committed = self.is_committed();
			return Err(error);
		}
		if let Some(limit) = self.0.budget.max_elapsed {
			let observed = self.elapsed();
			if observed >= limit {
				return Err(self.error(
					ErrorKind::DeadlineExceeded,
					phase,
					"elapsed_nanoseconds",
					limit.as_nanos(),
					observed.as_nanos(),
				));
			}
		}
		Ok(())
	}

	/// Atomically reserves one hidden or visible attempt.
	pub fn reserve_attempt(&self) -> Result<u32, Error> {
		let index = self.0.attempts.fetch_add(1, Ordering::AcqRel);
		if index >= self.0.budget.max_attempts {
			self.0.attempts.fetch_sub(1, Ordering::AcqRel);
			return Err(self.error(
				ErrorKind::BudgetExhausted,
				ErrorPhase::Readiness,
				"attempts",
				self.0.budget.max_attempts as u128,
				index as u128 + 1,
			));
		}
		self.0.attempt_started.lock().push(Instant::now());
		Ok(index)
	}

	/// Returns elapsed wall time for a reserved attempt.
	pub(crate) fn attempt_elapsed(&self, index: u32) -> std::time::Duration {
		self
			.0
			.attempt_started
			.lock()
			.get(index as usize)
			.map_or_else(|| self.elapsed(), Instant::elapsed)
	}

	/// Charges token usage across every visible and hidden attempt.
	pub fn charge_tokens(&self, input: u64, output: u64) -> Result<(), Error> {
		let observed_input = self
			.0
			.input_tokens
			.fetch_add(input, Ordering::AcqRel)
			.saturating_add(input);
		let observed_output = self
			.0
			.output_tokens
			.fetch_add(output, Ordering::AcqRel)
			.saturating_add(output);
		if let Some(limit) = self.0.budget.max_input_tokens {
			if observed_input > limit {
				return Err(self.error(
					ErrorKind::BudgetExhausted,
					ErrorPhase::Streaming,
					"input_tokens",
					limit as u128,
					observed_input as u128,
				));
			}
		}
		if let Some(limit) = self.0.budget.max_output_tokens {
			if observed_output > limit {
				return Err(self.error(
					ErrorKind::BudgetExhausted,
					ErrorPhase::Streaming,
					"output_tokens",
					limit as u128,
					observed_output as u128,
				));
			}
		}
		Ok(())
	}

	/// Charges estimated or final integer monetary cost across all attempts.
	pub fn charge_cost(&self, cost: Cost) -> Result<(), Error> {
		let observed = {
			let mut total = self.0.cost_micro_usd.lock();
			*total = total.saturating_add(cost.micro_usd);
			*total
		};
		if let Some(limit) = self.0.budget.max_cost {
			if observed > limit.micro_usd {
				return Err(self.error(
					ErrorKind::BudgetExhausted,
					ErrorPhase::Streaming,
					"cost_micro_usd",
					limit.micro_usd.max(0) as u128,
					observed.max(0) as u128,
				));
			}
		}
		Ok(())
	}

	/// Reserves bytes hidden behind the semantic output gate.
	pub fn reserve_provisional(&self, bytes: u64) -> Result<(), Error> {
		let observed = self
			.0
			.provisional_bytes
			.fetch_add(bytes, Ordering::AcqRel)
			.saturating_add(bytes);
		if observed > self.0.budget.max_provisional_bytes {
			self.0.provisional_bytes.fetch_sub(bytes, Ordering::AcqRel);
			return Err(self.error(
				ErrorKind::BudgetExhausted,
				ErrorPhase::Recovery,
				"provisional_bytes",
				self.0.budget.max_provisional_bytes as u128,
				observed as u128,
			));
		}
		Ok(())
	}

	/// Releases provisional bytes after flush or discard.
	pub fn release_provisional(&self, bytes: u64) {
		self
			.0
			.provisional_bytes
			.fetch_sub(bytes.min(self.provisional_bytes()), Ordering::AcqRel);
	}
}

/// Request envelope carrying the one execution context through every concrete
/// layer.
#[derive(Clone)]
pub struct LayerCall<T> {
	/// Current typed payload.
	pub payload: T,
	/// Cross-layer state, budgets, cancellation, and receipt accounting.
	pub context: ExecutionContext,
}

impl<T> LayerCall<T> {
	/// Maps the typed payload without reconstructing the execution stack or
	/// context.
	pub fn map<U>(self, f: impl FnOnce(T) -> U) -> LayerCall<U> {
		LayerCall { payload: f(self.payload), context: self.context }
	}
}

#[cfg(test)]
mod tests {
	use std::time::Duration;

	use super::ExecutionContext;
	use crate::{
		error::ErrorKind,
		receipt::{Cost, ExecutionBudget},
	};

	#[test]
	fn all_budget_dimensions_are_shared_and_bounded() {
		let budget = ExecutionBudget {
			max_elapsed:           Some(Duration::from_secs(1)),
			max_attempts:          2,
			max_input_tokens:      Some(3),
			max_output_tokens:     Some(5),
			max_cost:              Some(Cost::from_micro_usd(7)),
			max_provisional_bytes: 11,
			max_staging_bytes:     0,
		};
		let context = ExecutionContext::new(budget);
		assert_eq!(context.reserve_attempt().unwrap(), 0);
		assert_eq!(context.reserve_attempt().unwrap(), 1);
		assert_eq!(context.reserve_attempt().unwrap_err().kind, ErrorKind::BudgetExhausted);
		context.charge_tokens(3, 5).unwrap();
		assert_eq!(context.charge_tokens(1, 0).unwrap_err().kind, ErrorKind::BudgetExhausted);
		context.charge_cost(Cost::from_micro_usd(7)).unwrap();
		assert_eq!(
			context
				.charge_cost(Cost::from_micro_usd(1))
				.unwrap_err()
				.kind,
			ErrorKind::BudgetExhausted
		);
		context.reserve_provisional(11).unwrap();
		assert_eq!(context.reserve_provisional(1).unwrap_err().kind, ErrorKind::BudgetExhausted);
	}

	#[test]
	fn cancellation_propagates_through_checkpoints() {
		let context = ExecutionContext::new(ExecutionBudget::default());
		context.cancel();
		assert_eq!(
			context
				.checkpoint(crate::error::ErrorPhase::Readiness)
				.unwrap_err()
				.kind,
			ErrorKind::Cancelled
		);
	}
}
