//! Replay-safe credential rotation before the egress commit point.

use std::marker::PhantomData;

use omp_core::SmolStr;
use omp_llm_egress::auth_inject::CredentialLease;
use omp_llm_types::TurnEvent;
use smallvec::SmallVec;

/// One leased credential eligible for provider routing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CredentialCandidate {
	/// Canonical non-secret identity and generation selected for redemption.
	pub lease:            CredentialLease,
	/// Earliest wall-clock millisecond at which this credential may be retried.
	pub blocked_until_ms: u64,
}

/// Injected credential inventory, deliberately independent of `omp-llm-broker`.
pub trait CredentialSource {
	/// Returns the identities configured for a provider.
	fn credentials(&self, provider: &str) -> SmallVec<CredentialCandidate, 4>;
}

/// Upstream failures which permit trying another identity before commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RotationCause {
	/// HTTP 401 authentication failure.
	Unauthorized,
	/// HTTP 403 authorization failure.
	Forbidden,
	/// Provider usage allocation or quota was exhausted.
	UsageLimit,
}

/// Proof that no response content has crossed the egress commit point.
#[derive(Clone, Copy, Debug, Default)]
pub struct PreCommit;

/// State after response content has crossed the egress commit point.
///
/// `CredentialRotate<Committed, _>` intentionally has no rotation method:
/// replay-unsafe content may already have reached the caller.
#[derive(Clone, Copy, Debug)]
pub struct Committed;

/// A selected replacement identity and its caller-visible attempt event.
#[derive(Clone, Debug, PartialEq)]
pub struct RotationAttempt {
	/// Replacement credential identity.
	pub credential: CredentialCandidate,
	/// Attempt event to emit before sending with the replacement identity.
	pub event:      TurnEvent,
	/// Failure which caused the rotation.
	pub cause:      RotationCause,
}

/// Deterministic, identity-keyed credential rotation state.
///
/// The state parameter makes post-commit rotation unrepresentable. Calling
/// [`CredentialRotate::commit`] consumes the pre-commit value and returns a
/// state for which `rotate` does not exist.
#[derive(Clone, Debug)]
pub struct CredentialRotate<S, L> {
	source:         L,
	current:        CredentialLease,
	tried:          SmallVec<u64, 4>,
	now_ms:         u64,
	attempt_number: u32,
	_state:         PhantomData<S>,
}

impl<L: CredentialSource> CredentialRotate<PreCommit, L> {
	/// Starts rotation with the lease used by outbound attempt one.
	#[must_use]
	pub fn new(source: L, current: CredentialLease, now_ms: u64) -> Self {
		let credential_id = current.credential_id();
		Self {
			source,
			tried: smallvec::smallvec![credential_id],
			current,
			now_ms,
			attempt_number: 1,
			_state: PhantomData,
		}
	}

	/// Selects the next unblocked credential in stable identity-key order.
	///
	/// Returns `None` when no distinct usable identity remains. Selection wraps
	/// after the current key so the order is stable regardless of inventory
	/// order.
	pub fn rotate(&mut self, cause: RotationCause) -> Option<RotationAttempt> {
		let mut credentials = self.source.credentials(self.current.provider());
		credentials.retain(|candidate| {
			candidate.lease.provider() == self.current.provider()
				&& !self.tried.contains(&candidate.lease.credential_id())
				&& candidate.blocked_until_ms <= self.now_ms
		});
		credentials.sort_unstable_by_key(|candidate| candidate.lease.credential_id());
		let next_index = credentials
			.iter()
			.position(|candidate| candidate.lease.credential_id() > self.current.credential_id())
			.unwrap_or(0);
		let credential = credentials.get(next_index)?.clone();
		self.current = credential.lease.clone();
		self.tried.push(credential.lease.credential_id());
		self.attempt_number = self.attempt_number.saturating_add(1);
		Some(RotationAttempt {
			credential,
			event: TurnEvent::Attempt {
				number: self.attempt_number,
				reason: SmolStr::new(match cause {
					RotationCause::Unauthorized => "credential returned 401; rotating identity",
					RotationCause::Forbidden => "credential returned 403; rotating identity",
					RotationCause::UsageLimit => "credential usage limit reached; rotating identity",
				}),
			},
			cause,
		})
	}

	/// Crosses the commit point, permanently removing the rotation operation.
	#[must_use]
	pub fn commit(self) -> CredentialRotate<Committed, L> {
		CredentialRotate {
			source:         self.source,
			current:        self.current,
			tried:          self.tried,
			now_ms:         self.now_ms,
			attempt_number: self.attempt_number,
			_state:         PhantomData,
		}
	}
}

impl<S, L> CredentialRotate<S, L> {
	/// Returns the numeric identity selected for the current outbound attempt.
	#[must_use]
	pub const fn current_identity(&self) -> u64 {
		self.current.credential_id()
	}

	/// Returns the current one-based outbound attempt number.
	#[must_use]
	pub const fn attempt_number(&self) -> u32 {
		self.attempt_number
	}
}

#[cfg(test)]
mod tests {
	use smallvec::smallvec;

	use super::*;

	#[derive(Clone, Debug)]
	struct Inventory;

	impl CredentialSource for Inventory {
		fn credentials(&self, _provider: &str) -> SmallVec<CredentialCandidate, 4> {
			smallvec![
				CredentialCandidate {
					lease:            CredentialLease::new("provider", 30, 2),
					blocked_until_ms: 0,
				},
				CredentialCandidate {
					lease:            CredentialLease::new("provider", 20, 2),
					blocked_until_ms: 200,
				},
				CredentialCandidate {
					lease:            CredentialLease::new("provider", 10, 1),
					blocked_until_ms: 0,
				},
			]
		}
	}

	#[test]
	fn skips_blocked_credentials_in_identity_order() {
		let current = CredentialLease::new("provider", 10, 1);
		let mut rotation = CredentialRotate::new(Inventory, current, 100);
		let attempt = rotation
			.rotate(RotationCause::Unauthorized)
			.expect("credential 30 remains usable");
		assert_eq!(attempt.credential.lease, CredentialLease::new("provider", 30, 2));
		assert!(matches!(attempt.event, TurnEvent::Attempt { number: 2, .. }));
	}

	#[test]
	fn committed_state_has_no_rotation_operation() {
		let current = CredentialLease::new("provider", 10, 1);
		let rotation = CredentialRotate::new(Inventory, current, 100).commit();
		assert_eq!(rotation.current_identity(), 10);
		// This contract is enforced by the type system:
		// `CredentialRotate<Committed, _>` has no `rotate` method, so replay
		// after content commit cannot compile.
	}
}
