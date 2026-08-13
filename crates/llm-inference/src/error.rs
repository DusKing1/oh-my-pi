//! Cloneable structured failures with explicit retry scope.

use std::{fmt, time::Duration};

use omp_core::Str;

use crate::{
	answer::AnswerKind,
	catalog::{OperationKind, ProviderId, RouteId},
	id::RequestId,
	receipt::{ExecutionReceipt, ReasonId},
};

/// Stable, policy-consumable failure category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
	/// Caller cancellation.
	Cancelled,
	/// Absolute deadline elapsed.
	DeadlineExceeded,
	/// An execution budget dimension was exhausted.
	BudgetExhausted,
	/// Transactional output exceeded its configured in-memory or secure-spool
	/// bound.
	PolicyBufferExceeded,
	/// Domain-name resolution failed.
	Dns,
	/// TLS negotiation or verification failed.
	Tls,
	/// Connection establishment or transport connectivity failed.
	Connectivity,
	/// Wire protocol contract was violated.
	Protocol,
	/// A committed or provisional stream was corrupt.
	StreamCorruption,
	/// Credential was absent, expired, or rejected.
	Authentication,
	/// Principal lacks permission for the operation.
	Authorization,
	/// Account was disabled or rejected.
	AccountDisabled,
	/// A rate window rejected the attempt.
	RateLimited,
	/// Account quota was exhausted.
	QuotaExhausted,
	/// Provider requires payment or credit.
	PaymentRequired,
	/// Canonical request was invalid.
	InvalidRequest,
	/// Requested target or selector resolved to no catalog model.
	TargetNotFound,
	/// Capability support is unknown and caller policy forbids assuming it.
	CapabilityUnknown,
	/// Typed native options do not match the selected codec.
	CodecMismatch,
	/// No constructed service is available for an otherwise eligible route.
	RouteUnavailable,
	/// Catalog or route state changed after a plan was produced.
	StalePlan,
	/// Requested recovery requires replayable input.
	ReplayRequired,
	/// One-shot input requires explicitly enabled secure staging.
	StagingRequired,
	/// Selected model or route cannot satisfy required capability intent.
	CapabilityMismatch,
	/// Provider contradicted its catalog-advertised contract.
	ProviderContractMismatch,
	/// Canonical context exceeds an applicable model or wire limit.
	ContextOverflow,
	/// Provider content filtering stopped output.
	ContentFilter,
	/// Provider emitted a safety refusal.
	SafetyRefusal,
	/// Model output could not be decoded within protocol bounds.
	MalformedModelOutput,
	/// Structured output could not satisfy its declared contract.
	StructuredOutputFailure,
	/// Required tool-call intent was not satisfied.
	ToolNonCompliance,
	/// Reasoning loop bounds were exceeded.
	RepeatedReasoning,
	/// Repeated tool-call loop bounds were exceeded.
	RepeatedToolCall,
	/// Model produced no usable completion.
	EmptyCompletion,
	/// Provider-side session state expired.
	SessionExpired,
	/// Conversation or provider-state revision conflicted.
	SessionConflict,
	/// Required local model or runtime is unavailable.
	LocalModelUnavailable,
	/// Local memory, compute, storage, or concurrency was exhausted.
	ResourceExhausted,
	/// Native method/path or payload exceeded its allowlist contract.
	NativeRequestRejected,
	/// An internal invariant was violated.
	InternalInvariant,
}

/// Execution phase in which a failure was classified.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorPhase {
	/// Side-effect-free planning and capability negotiation.
	Planning,
	/// Tower readiness and queue admission.
	Readiness,
	/// Route/account concurrency admission.
	Admission,
	/// Credential acquisition, refresh, or signing.
	Authentication,
	/// Canonical-to-wire encoding.
	Encoding,
	/// DNS, TLS, or connection establishment.
	Connecting,
	/// Response handshake and first decodable frame.
	Handshake,
	/// Committed or provisional response streaming.
	Streaming,
	/// Sans-I/O recovery or validation.
	Recovery,
	/// Conversation or provider-state handling.
	Session,
	/// Local backend loading or inference.
	LocalRuntime,
	/// Artifact staging, upload, download, or verification.
	Artifact,
	/// Usage or model discovery.
	Discovery,
	/// No narrower phase applies to an invariant failure.
	Internal,
}

/// Explicit action that policy may take for a structured failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetryAction {
	/// No automatic action is safe.
	Never,
	/// Retry the same route and account after a structured delay.
	SameRoute {
		/// Minimum delay before retrying.
		after: Duration,
	},
	/// Refresh credentials for the same account and principal.
	RefreshCredential,
	/// Select another eligible account.
	RotateAccount,
	/// Select another allowed route for the same normalized model.
	ReselectRoute,
	/// Replay canonical history to reseed provider-side state.
	ReseedSession,
	/// Run another transactionally gated semantic attempt.
	SemanticRetry,
}

/// Typed supplemental evidence that contains no secret-bearing source text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ErrorDetail {
	/// Erased answer variant did not match the typed operation contract.
	BodyVariantMismatch {
		/// Expected operation.
		expected: OperationKind,
		/// Actual erased body variant.
		actual:   AnswerKind,
	},
	/// A named budget dimension was exhausted at an integer observed value.
	Budget {
		/// Exhausted budget dimension.
		dimension: Str,
		/// Configured integer limit.
		limit:     u128,
		/// Observed integer value.
		observed:  u128,
	},
	/// Context size evidence.
	Context {
		/// Configured context limit.
		limit:    u64,
		/// Observed context size.
		observed: u64,
	},
	/// A capability requirement could not be satisfied.
	Capability {
		/// Required feature.
		feature: Str,
		/// Typed failure reason.
		reason:  ReasonId,
	},
	/// A selector or target could not resolve.
	Target {
		/// Sanitized selector.
		selector: Str,
	},
	/// A previously produced execution plan is no longer valid.
	StalePlan {
		/// Revision used during planning.
		planned_revision: Str,
		/// Current registry revision.
		current_revision: Str,
	},
	/// Replay or staging requirement evidence.
	Replay {
		/// Typed replay reason.
		reason: ReasonId,
	},
	/// Sanitized bounded protocol evidence.
	Protocol {
		/// Typed protocol reason.
		reason: ReasonId,
	},
	/// Bounded provider message after codec-owned sanitization.
	Provider {
		/// Sanitized bounded provider message.
		sanitized_message: Str,
	},
	/// Local availability evidence.
	LocalUnavailable {
		/// Typed local-availability reason.
		reason: ReasonId,
	},
}

/// Concrete, cloneable, secret-free inference error.
#[derive(Clone)]
pub struct Error {
	/// Stable failure category.
	pub kind:       ErrorKind,
	/// Execution phase where the failure was classified.
	pub phase:      ErrorPhase,
	/// Explicit policy action.
	pub action:     RetryAction,
	/// Provider involved, if selection had completed.
	pub provider:   Option<ProviderId>,
	/// Route involved, if selection had completed.
	pub route:      Option<RouteId>,
	/// Logical request identity.
	pub request_id: Option<RequestId>,
	/// HTTP-like status when structurally available.
	pub status:     Option<u16>,
	/// Structured provider or runtime error code.
	pub code:       Option<Str>,
	/// Whether ordinary output had become visible.
	pub committed:  bool,
	/// Partial accounting through the failure point.
	pub receipt:    ExecutionReceipt,
	/// Typed supplemental evidence.
	pub detail:     Option<ErrorDetail>,
}

impl fmt::Debug for Error {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		let detail_kind = self.detail.as_ref().map(|detail| match detail {
			ErrorDetail::BodyVariantMismatch { .. } => "BodyVariantMismatch",
			ErrorDetail::Budget { .. } => "Budget",
			ErrorDetail::Context { .. } => "Context",
			ErrorDetail::Target { .. } => "Target",
			ErrorDetail::Capability { .. } => "Capability",
			ErrorDetail::Replay { .. } => "Replay",
			ErrorDetail::Protocol { .. } => "Protocol",
			ErrorDetail::Provider { .. } => "Provider",
			ErrorDetail::LocalUnavailable { .. } => "LocalUnavailable",
			ErrorDetail::StalePlan { .. } => "StalePlan",
		});
		formatter
			.debug_struct("Error")
			.field("kind", &self.kind)
			.field("phase", &self.phase)
			.field("action", &self.action)
			.field("provider", &self.provider)
			.field("route", &self.route)
			.field("request_id", &self.request_id)
			.field("status", &self.status)
			.field("code", &self.code)
			.field("committed", &self.committed)
			.field("receipt", &"<accounting redacted>")
			.field("detail_kind", &detail_kind)
			.finish()
	}
}

impl Error {
	/// Constructs a structured error with no provider-specific evidence.
	pub fn new(
		kind: ErrorKind,
		phase: ErrorPhase,
		action: RetryAction,
		receipt: ExecutionReceipt,
	) -> Self {
		Self {
			kind,
			phase,
			action,
			provider: None,
			route: None,
			request_id: None,
			status: None,
			code: None,
			committed: false,
			receipt,
			detail: None,
		}
	}

	/// Constructs a terminal planning error with typed evidence.
	pub fn planning(kind: ErrorKind, detail: ErrorDetail, receipt: ExecutionReceipt) -> Self {
		let mut error = Self::new(kind, ErrorPhase::Planning, RetryAction::Never, receipt);
		error.detail = Some(detail);
		error
	}

	/// Constructs the internal protocol error returned for a typed answer
	/// mismatch.
	pub fn body_variant_mismatch(
		expected: OperationKind,
		actual: AnswerKind,
		receipt: ExecutionReceipt,
	) -> Self {
		let mut error = Self::new(
			ErrorKind::ProviderContractMismatch,
			ErrorPhase::Internal,
			RetryAction::Never,
			receipt,
		);
		error.detail = Some(ErrorDetail::BodyVariantMismatch { expected, actual });
		error
	}
}

impl fmt::Display for Error {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(formatter, "inference {:?} error during {:?}", self.kind, self.phase)?;
		if let Some(code) = &self.code {
			write!(formatter, " ({code})")?;
		}
		Ok(())
	}
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
	use omp_core::Str;

	use super::{Error, ErrorKind, ErrorPhase, RetryAction};
	use crate::receipt::ExecutionReceipt;

	#[test]
	fn structured_error_debug_contains_no_external_source_text() {
		let mut error = Error::new(
			ErrorKind::Authentication,
			ErrorPhase::Authentication,
			RetryAction::Never,
			ExecutionReceipt::default(),
		);
		error.code = Some(Str::from("invalid_credential"));
		let debug = format!("{error:?}");
		assert!(debug.contains("invalid_credential"));
		assert!(!debug.contains("Authorization:"));
		assert!(!debug.contains("source"));
	}
}
