//! Shared operation validation and lifecycle state machines.

pub mod artifact;
pub mod discovery;
pub mod embedding;
pub mod image;
pub mod job;
pub mod native;
pub mod realtime;
pub mod search;
pub mod speech;
pub mod tokens;
pub mod transcription;
pub mod usage;
pub mod video;

use std::{sync::Arc, time::Instant};

use crate::{
	answer::{Answer, AnswerBody, ResponseMeta},
	call::{Call, OperationCall, SessionRequest, Target},
	catalog::{ModelKey, OperationKind, ProviderId, RouteId},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	id::RequestId,
	plan::ExecutionPlan,
	receipt::{ExecutionBudget, ExecutionReceipt, ReasonId},
};

/// Fixed selected-route identity used by route-local operation backends.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteIdentity {
	/// Provider domain.
	pub provider: ProviderId,
	/// Concrete route.
	pub route:    RouteId,
	/// Normalized selected model.
	pub model:    ModelKey,
}

impl RouteIdentity {
	/// Creates response metadata for one logical request.
	pub fn response_meta(&self, request_id: RequestId) -> ResponseMeta {
		ResponseMeta {
			request_id,
			provider: self.provider.clone(),
			route: self.route.clone(),
			model: Some(self.model.clone()),
			provider_request_id: None,
			created_at: std::time::SystemTime::now(),
		}
	}
}

/// Clone-cheap typed input passed from an operation service to its route
/// backend.
#[derive(Clone, Debug)]
pub struct OperationRequest<T> {
	/// Logical request identity.
	pub id:        RequestId,
	/// Resolved model and route constraint.
	pub target:    Target,
	/// Absolute execution deadline.
	pub deadline:  Option<Instant>,
	/// Cross-attempt budget.
	pub budget:    ExecutionBudget,
	/// Optional conversation state.
	pub session:   Option<SessionRequest>,
	/// Immutable selected execution plan carried into codec/auth composition.
	pub execution: Option<Arc<ExecutionPlan>>,
	/// Operation-specific immutable payload.
	pub payload:   Arc<T>,
}

impl<T> OperationRequest<T> {
	/// Creates a typed backend request while preserving all shared call
	/// metadata.
	pub(crate) fn from_call(call: &Call, payload: Arc<T>) -> Self {
		Self {
			id: call.id.clone(),
			target: call.target.clone(),
			deadline: call.deadline,
			budget: call.budget.clone(),
			session: call.session.clone(),
			execution: call.execution.clone(),
			payload,
		}
	}

	/// Reconstructs the closed call for one operation-specific RouteCodecSet
	/// entry.
	pub fn into_call(self, wrap: impl FnOnce(Arc<T>) -> OperationCall) -> Call {
		Call {
			id:        self.id,
			target:    self.target,
			deadline:  self.deadline,
			budget:    self.budget,
			session:   self.session,
			execution: self.execution,
			operation: wrap(self.payload),
		}
	}
}

/// Typed successful output returned by a route-local operation backend.
///
/// The backend supplies selected-route metadata and accounting because those
/// facts are known only after the inner auth/codec/transport stack runs.
#[derive(Clone, Debug)]
pub struct OperationResponse<T> {
	/// Selected-route response metadata.
	pub meta:    ResponseMeta,
	/// Accounting accumulated by the inner stack.
	pub receipt: ExecutionReceipt,
	/// Typed operation output.
	pub output:  T,
}

impl<T> OperationResponse<T> {
	/// Transforms the typed output without changing route evidence.
	pub fn map<U>(self, transform: impl FnOnce(T) -> U) -> OperationResponse<U> {
		OperationResponse {
			meta:    self.meta,
			receipt: self.receipt,
			output:  transform(self.output),
		}
	}

	/// Converts typed route output into the closed erased answer.
	pub fn into_answer(self, body: impl FnOnce(T) -> AnswerBody) -> Answer {
		Answer { meta: self.meta, receipt: self.receipt, body: body(self.output) }
	}
}

/// Merges accounting from a later unary subrequest into the first response.
pub(crate) fn merge_receipts(target: &mut ExecutionReceipt, mut later: ExecutionReceipt) {
	target.adjustments.append(&mut later.adjustments);
	target.attempts.append(&mut later.attempts);
	target.recoveries.append(&mut later.recoveries);
	target.staging.append(&mut later.staging);
	target.usage += later.usage;
	target.cost += later.cost;
	target.timings.queued += later.timings.queued;
	target.timings.planning += later.timings.planning;
	target.timings.authentication += later.timings.authentication;
	target.timings.encoding += later.timings.encoding;
	target.timings.streaming += later.timings.streaming;
	target.timings.total += later.timings.total;
	target.timings.first_frame = match (target.timings.first_frame, later.timings.first_frame) {
		(Some(left), Some(right)) => Some(left + right),
		(left, right) => left.or(right),
	};
	target.timings.completed_at = target.timings.completed_at.max(later.timings.completed_at);
}

pub(crate) fn wrong_operation(call: &Call, expected: OperationKind) -> Error {
	let mut error = Error::new(
		ErrorKind::InternalInvariant,
		ErrorPhase::Internal,
		RetryAction::Never,
		ExecutionReceipt::default(),
	);
	error.request_id = Some(call.id.clone());
	error.detail = Some(ErrorDetail::Capability {
		feature: omp_core::Str::from(expected.to_string()),
		reason:  ReasonId(omp_core::Str::from("operation_service_mismatch")),
	});
	error
}

pub(crate) fn media_validation_error(
	operation: OperationKind,
	reason: impl Into<omp_core::Str>,
) -> Error {
	let mut error = Error::new(
		ErrorKind::InvalidRequest,
		ErrorPhase::Planning,
		RetryAction::Never,
		ExecutionReceipt::default(),
	);
	error.detail = Some(ErrorDetail::Capability {
		feature: omp_core::Str::from(operation.to_string()),
		reason:  ReasonId(reason.into()),
	});
	error
}

pub(crate) fn media_protocol_error(
	operation: OperationKind,
	reason: impl Into<omp_core::Str>,
) -> Error {
	let mut error = Error::new(
		ErrorKind::ProviderContractMismatch,
		ErrorPhase::Streaming,
		RetryAction::Never,
		ExecutionReceipt::default(),
	);
	error.detail = Some(ErrorDetail::Capability {
		feature: omp_core::Str::from(operation.to_string()),
		reason:  ReasonId(reason.into()),
	});
	error.committed = true;
	error
}
