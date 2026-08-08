//! Request envelopes carried through replay-capable provider middleware.
//!
//! The canonical protobuf request contains only inference data. Routing
//! metadata, including credential lease identity and its validated non-secret
//! projection, travels beside it in a typed envelope and is cloned unchanged
//! whenever a pre-commit layer re-dispatches an attempt.

use std::sync::Arc;

use omp_llm_types::ResolvedModelPolicy;
use omp_proto::inference::v1::TurnRequest;

use crate::select::Routed;
/// Canonical protobuf request paired with trusted in-process model policy.
///
/// The policy intentionally travels beside protobuf, exactly like credential
/// routing metadata, and therefore cannot be supplied by a foreign client.
#[derive(Clone, Debug)]
pub struct ProviderRequest {
	/// Canonical foreign-wire request.
	pub request:      TurnRequest,
	/// Trusted policy resolved by the gateway catalog.
	pub model_policy: Option<Arc<ResolvedModelPolicy>>,
}

impl ProviderRequest {
	/// Creates a provider request without serializing trusted policy.
	#[must_use]
	pub const fn new(request: TurnRequest, model_policy: Option<Arc<ResolvedModelPolicy>>) -> Self {
		Self { request, model_policy }
	}
}

/// A replayable envelope containing one canonical turn request.
///
/// Middleware may inspect or repair the canonical request through this narrow
/// interface, but must preserve every other field in the envelope.
pub trait TurnRequestEnvelope: Clone + Send + 'static {
	/// Borrows the canonical request.
	fn request(&self) -> &TurnRequest;

	/// Mutably borrows the canonical request while retaining routing metadata.
	fn request_mut(&mut self) -> &mut TurnRequest;

	/// Borrows trusted server-resolved model policy, when attached.
	fn model_policy(&self) -> Option<&Arc<ResolvedModelPolicy>> {
		None
	}

	/// Returns the non-secret provider/account identity carried beside the
	/// request, when credential selection has already occurred.
	///
	/// Capability learning uses this identity with the model id so one
	/// account's rejection never disables a feature for sibling accounts.
	fn learning_identity(&self) -> Option<(&str, u64)> {
		None
	}
}

impl TurnRequestEnvelope for TurnRequest {
	fn request(&self) -> &TurnRequest {
		self
	}

	fn request_mut(&mut self) -> &mut TurnRequest {
		self
	}
}

impl TurnRequestEnvelope for ProviderRequest {
	fn request(&self) -> &TurnRequest {
		&self.request
	}

	fn request_mut(&mut self) -> &mut TurnRequest {
		&mut self.request
	}

	fn model_policy(&self) -> Option<&Arc<ResolvedModelPolicy>> {
		self.model_policy.as_ref()
	}
}

impl TurnRequestEnvelope for Routed {
	fn request(&self) -> &TurnRequest {
		&self.request
	}

	fn request_mut(&mut self) -> &mut TurnRequest {
		&mut self.request
	}

	fn model_policy(&self) -> Option<&Arc<ResolvedModelPolicy>> {
		self.model_policy.as_ref()
	}

	fn learning_identity(&self) -> Option<(&str, u64)> {
		self
			.lease
			.as_ref()
			.map(|lease| (lease.provider(), lease.credential_id()))
	}
}
