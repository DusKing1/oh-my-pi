//! Account affinity, eligibility, cooldown, quota, and explicit rotation
//! handoff.

use std::task::{Context, Poll};

use tower::{Layer, Service};

use crate::{error::Error, layer::LayerCall};

/// Request paired with a selected non-secret account handle.
#[derive(Clone)]
pub struct Accounted<R, A> {
	/// Original planned request.
	pub request: R,
	/// Selected account handle or metadata; it must contain no credential
	/// material.
	pub account: A,
}

/// Selects an eligible account using affinity, cooldown, quota, and route
/// scope.
pub trait AccountSelector<R>: Clone + Send + 'static {
	/// Non-secret account selection passed to the auth boundary.
	type Account: Clone + Send + 'static;
	/// Selects and revalidates an account immediately before authentication.
	fn select(
		&self,
		request: &R,
		context: &crate::layer::ExecutionContext,
	) -> Result<Self::Account, Error>;
	/// Returns canonical non-secret routing metadata propagated to encoding.
	fn routing(&self, _: &Self::Account) -> Option<crate::call::AccountRoutingContext> {
		None
	}
}

/// Production adapter from a planned request into the shared deterministic
/// account pool.
#[derive(Clone)]
pub struct SharedAccountPool<M> {
	pool: crate::account::AccountPool,
	map:  M,
}
impl<M> SharedAccountPool<M> {
	/// Creates an adapter whose mapper consumes the typed outer-attempt action.
	pub fn new(pool: crate::account::AccountPool, map: M) -> Self {
		Self { pool, map }
	}
}
impl<R, M> AccountSelector<R> for SharedAccountPool<M>
where
	M: Fn(
			&R,
			crate::layer::AttemptAction,
			Option<&crate::layer::SessionAffinity>,
		) -> crate::account::AccountSelectionRequest
		+ Clone
		+ Send
		+ 'static,
{
	type Account = crate::account::AccountSelection;

	fn select(
		&self,
		request: &R,
		context: &crate::layer::ExecutionContext,
	) -> Result<Self::Account, Error> {
		let affinity = context.session_affinity();
		let selection = (self.map)(request, context.attempt_action(), affinity.as_ref());
		self.pool.select(&selection).map_err(|_| {
			Error::new(
				crate::error::ErrorKind::QuotaExhausted,
				crate::error::ErrorPhase::Admission,
				crate::error::RetryAction::Never,
				context.receipt(),
			)
		})
	}

	fn routing(&self, selection: &Self::Account) -> Option<crate::call::AccountRoutingContext> {
		Some(selection.routing.clone())
	}
}

/// Adds account selection.
#[derive(Clone, Debug)]
pub struct AccountPoolLayer<P> {
	pool: P,
}
impl<P> AccountPoolLayer<P> {
	/// Creates an account layer.
	pub const fn new(pool: P) -> Self {
		Self { pool }
	}
}
/// Account-selecting service.
#[derive(Clone, Debug)]
pub struct AccountPoolService<S, P> {
	inner: S,
	pool:  P,
}
impl<S, P: Clone> Layer<S> for AccountPoolLayer<P> {
	type Service = AccountPoolService<S, P>;

	fn layer(&self, inner: S) -> Self::Service {
		AccountPoolService { inner, pool: self.pool.clone() }
	}
}
impl<S, P, R> Service<LayerCall<R>> for AccountPoolService<S, P>
where
	P: AccountSelector<R>,
	S: Service<LayerCall<Accounted<R, P::Account>>, Error = Error>,
{
	type Error = Error;
	type Response = S::Response;

	type Future = impl Future<Output = Result<S::Response, Error>>;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, request: LayerCall<R>) -> Self::Future {
		let selected = self.pool.select(&request.payload, &request.context);
		let next = selected.and_then(|account| {
			if let Some(routing) = self.pool.routing(&account) {
				validate_session_affinity(&request.context, &routing)?;
				request.context.set_account_routing(routing);
			}
			Ok(LayerCall {
				payload: Accounted { request: request.payload, account },
				context: request.context,
			})
		});
		let future = next.map(|request| self.inner.call(request));
		async move { future?.await }
	}
}

fn validate_session_affinity(
	context: &crate::layer::ExecutionContext,
	routing: &crate::call::AccountRoutingContext,
) -> Result<(), Error> {
	let Some(binding) = context.session_affinity() else {
		return Ok(());
	};
	let principal_matches = routing.principal.as_ref() == Some(&binding.principal);
	let generation_matches = match binding.credential_policy {
		crate::session::CredentialGenerationPolicy::PrincipalBound => true,
		crate::session::CredentialGenerationPolicy::CredentialGenerationBound => {
			routing.credential_generation == Some(binding.credential_generation)
		},
	};
	if principal_matches && generation_matches {
		return Ok(());
	}
	Err(Error::new(
		crate::error::ErrorKind::SessionExpired,
		crate::error::ErrorPhase::Session,
		crate::error::RetryAction::ReseedSession,
		context.receipt(),
	))
}

#[cfg(test)]
mod tests {
	use super::validate_session_affinity;
	use crate::{
		call::AccountRoutingContext,
		error::{ErrorKind, RetryAction},
		id::PrincipalId,
		layer::{ExecutionContext, SessionAffinity},
		receipt::ExecutionBudget,
		session::CredentialGenerationPolicy,
	};

	#[test]
	fn principal_mismatch_requires_prebody_reseed() {
		let context = ExecutionContext::new(ExecutionBudget::default());
		context.set_session_affinity(Some(SessionAffinity {
			principal:             PrincipalId::from("bound"),
			credential_generation: 7,
			credential_policy:     CredentialGenerationPolicy::PrincipalBound,
		}));
		let routing = AccountRoutingContext {
			principal: Some(PrincipalId::from("other")),
			credential_generation: Some(7),
			..Default::default()
		};
		let error = validate_session_affinity(&context, &routing).unwrap_err();
		assert_eq!(error.kind, ErrorKind::SessionExpired);
		assert_eq!(error.action, RetryAction::ReseedSession);
		assert!(!error.committed);
	}

	#[test]
	fn generation_policy_controls_refresh_compatibility() {
		let context = ExecutionContext::new(ExecutionBudget::default());
		let routing = AccountRoutingContext {
			principal: Some(PrincipalId::from("bound")),
			credential_generation: Some(8),
			..Default::default()
		};
		context.set_session_affinity(Some(SessionAffinity {
			principal:             PrincipalId::from("bound"),
			credential_generation: 7,
			credential_policy:     CredentialGenerationPolicy::PrincipalBound,
		}));
		validate_session_affinity(&context, &routing).unwrap();
		context.set_session_affinity(Some(SessionAffinity {
			principal:             PrincipalId::from("bound"),
			credential_generation: 7,
			credential_policy:     CredentialGenerationPolicy::CredentialGenerationBound,
		}));
		assert_eq!(
			validate_session_affinity(&context, &routing)
				.unwrap_err()
				.action,
			RetryAction::ReseedSession
		);
	}
}
