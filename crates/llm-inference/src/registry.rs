//! Immutable provider route registry and construction-time builder.

use std::{
	collections::HashMap,
	sync::Arc,
	task::{Context, Poll},
	time::SystemTime,
};

use futures::future::BoxFuture;
use omp_core::Str;
use tower::{Service, ServiceExt};

use crate::{
	answer::{Answer, AnswerBody, ResponseMeta},
	auth::AuthManager,
	body::RetryDecision,
	call::{Call, OperationCall},
	catalog::{CatalogRevision, OperationKind, RouteDef, RouteId, snapshot::Catalog},
	error::{Error, ErrorDetail, ErrorKind, RetryAction},
	layer::{
		ExecutionContext, LayerCall,
		observe::{NoopObserver, ObserveLayer, Observer},
		stack::{
			BuiltinConfig, BuiltinRouteStackFactory, RouteProviderService, RouteStackFactory,
			build_execution_stack,
		},
	},
	provider::ProviderService,
	receipt::{AttemptOutcome, ExecutionReceipt, ReasonId},
};

/// Typed evidence explaining why a catalog route has no constructed service.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteUnavailable {
	/// Catalog route that could not be constructed.
	pub route:     RouteId,
	/// Stable secret-free reason.
	pub reason:    ReasonId,
	/// Operation affected when the failure is narrower than the route.
	pub operation: Option<OperationKind>,
}

#[derive(Clone)]
enum RouteBinding {
	Available(RouteProviderService),
	Unavailable(RouteUnavailable),
}

/// Immutable registry of catalog definitions and preconstructed route services.
#[derive(Clone)]
pub struct Registry {
	inner: Arc<RegistryInner>,
}

struct RegistryInner {
	catalog:      Arc<Catalog>,
	bindings:     HashMap<RouteId, RouteBinding>,
	auth_manager: Option<AuthManager>,
	generation:   u64,
}

impl Registry {
	/// Starts a construction-time builder for one immutable catalog snapshot.
	pub fn builder(catalog: Arc<Catalog>) -> RegistryBuilder {
		RegistryBuilder { catalog, bindings: HashMap::new(), auth_manager: None, generation: 1 }
	}

	/// Returns the immutable catalog revision used by this registry.
	pub fn catalog_revision(&self) -> &CatalogRevision {
		self.inner.catalog.revision()
	}

	/// Returns the registry state generation captured by execution plans.
	pub fn generation(&self) -> u64 {
		self.inner.generation
	}

	/// Borrows the immutable catalog snapshot.
	pub fn catalog(&self) -> &Catalog {
		&self.inner.catalog
	}

	/// Reports whether direct route-independent authentication operations are
	/// constructed.
	pub fn contains_auth_manager(&self) -> bool {
		self.inner.auth_manager.is_some()
	}

	/// Returns typed construction evidence for an unavailable route.
	pub fn unavailability(&self, route: &RouteId) -> Option<&RouteUnavailable> {
		match self.inner.bindings.get(route) {
			Some(RouteBinding::Unavailable(evidence)) => Some(evidence),
			Some(RouteBinding::Available(_)) | None => None,
		}
	}

	/// Returns whether a concrete route has a constructed comprehensive service.
	pub fn contains_service(&self, route: &RouteId) -> bool {
		matches!(self.inner.bindings.get(route), Some(RouteBinding::Available(_)))
	}

	/// Creates a clone-cheap comprehensive dispatch service with no observation
	/// sink.
	pub fn service(&self) -> ProviderService {
		self.service_with_observer(NoopObserver)
	}

	/// Creates one outer logical-execution boundary around all exact route
	/// fallbacks.
	pub fn service_with_observer<O: Observer>(&self, observer: O) -> ProviderService {
		ProviderService::new(build_execution_stack(
			RegistryDispatch { registry: self.clone() },
			ObserveLayer::new(observer),
		))
	}

	/// Validates a planned call against current catalog and registry state
	/// before route dispatch.
	pub fn validate_plan(&self, call: &Call, now: std::time::Instant) -> Result<(), Error> {
		let plan = call.execution.as_ref().ok_or_else(|| {
			Error::planning(
				ErrorKind::InvalidRequest,
				ErrorDetail::Target { selector: Str::from("call-has-no-execution-plan") },
				ExecutionReceipt::default(),
			)
		})?;
		plan.validate(now, self.catalog_revision(), self.generation())?;
		if call.operation.kind() != plan.operation {
			return Err(Error::planning(
				ErrorKind::ProviderContractMismatch,
				ErrorDetail::Capability {
					feature: Str::from(call.operation.kind().to_string()),
					reason:  ReasonId(Str::from("planned-operation-mismatch")),
				},
				ExecutionReceipt::default(),
			));
		}
		Ok(())
	}

	pub(crate) fn route_service(
		&self,
		route: &RouteId,
		operation: OperationKind,
	) -> Result<RouteProviderService, Error> {
		match self.inner.bindings.get(route) {
			Some(RouteBinding::Available(service)) => Ok(service.clone()),
			Some(RouteBinding::Unavailable(evidence)) => {
				Err(route_unavailable_error(evidence, operation))
			},
			None => Err(target_error(route.as_str())),
		}
	}
}

/// Construction-time builder; mutation ends permanently at
/// [`RegistryBuilder::build`].
pub struct RegistryBuilder {
	catalog:      Arc<Catalog>,
	bindings:     HashMap<RouteId, RouteBinding>,
	auth_manager: Option<AuthManager>,
	generation:   u64,
}

impl RegistryBuilder {
	/// Registers one preconstructed route-local service for a catalog route.
	pub fn register_route(
		mut self,
		route: RouteId,
		service: RouteProviderService,
	) -> Result<Self, Error> {
		self.require_catalog_route(&route)?;
		if self
			.bindings
			.insert(route.clone(), RouteBinding::Available(service))
			.is_some()
		{
			return Err(duplicate_route_error(&route));
		}
		self.generation = self.generation.saturating_add(1);
		Ok(self)
	}

	/// Registers typed unavailability for a catalog route that cannot be
	/// constructed.
	pub fn register_unavailable(mut self, evidence: RouteUnavailable) -> Result<Self, Error> {
		self.require_catalog_route(&evidence.route)?;
		if self
			.bindings
			.insert(evidence.route.clone(), RouteBinding::Unavailable(evidence.clone()))
			.is_some()
		{
			return Err(duplicate_route_error(&evidence.route));
		}
		self.generation = self.generation.saturating_add(1);
		Ok(self)
	}

	/// Registers the one route-independent authentication/account-management
	/// service.
	pub fn with_auth_manager(mut self, manager: AuthManager) -> Self {
		self.auth_manager = Some(manager);
		self.generation = self.generation.saturating_add(1);
		self
	}

	/// Constructs every catalog route exactly once through a production
	/// route-stack factory.
	pub fn with_factory(mut self, factory: Arc<dyn RouteStackFactory>) -> Result<Self, Error> {
		for route in self.catalog.routes() {
			if self.bindings.contains_key(&route.id) {
				continue;
			}
			let binding = match factory.build(&self.catalog, route) {
				Ok(service) => RouteBinding::Available(service),
				Err(evidence) => RouteBinding::Unavailable(evidence),
			};
			self.bindings.insert(route.id.clone(), binding);
			self.generation = self.generation.saturating_add(1);
		}
		Ok(self)
	}

	/// Constructs every built-in route once from complete production composition
	/// dependencies.
	pub fn with_builtins(self, config: BuiltinConfig) -> Result<Self, Error> {
		let manager = config.auth_manager().cloned();
		let builder = match manager {
			Some(manager) => self.with_auth_manager(manager),
			None => self,
		};
		builder.with_factory(Arc::new(BuiltinRouteStackFactory::new(config)))
	}

	/// Freezes all definitions and services into a clone-cheap immutable
	/// registry.
	pub fn build(self) -> Result<Registry, Error> {
		for route in self.catalog.routes() {
			if !self.bindings.contains_key(&route.id) {
				return Err(Error::planning(
					ErrorKind::RouteUnavailable,
					ErrorDetail::Capability {
						feature: Str::from(route.id.as_str()),
						reason:  ReasonId(Str::from("route-has-no-service-or-unavailability-evidence")),
					},
					ExecutionReceipt::default(),
				));
			}
		}
		if self
			.catalog
			.providers()
			.iter()
			.any(|provider| provider.management.supports(OperationKind::Auth))
			&& self.auth_manager.is_none()
		{
			return Err(Error::planning(
				ErrorKind::RouteUnavailable,
				ErrorDetail::Capability {
					feature: Str::from(OperationKind::Auth.to_string()),
					reason:  ReasonId(Str::from("auth-manager-not-constructed")),
				},
				ExecutionReceipt::default(),
			));
		}
		Ok(Registry {
			inner: Arc::new(RegistryInner {
				catalog:      self.catalog,
				bindings:     self.bindings,
				auth_manager: self.auth_manager,
				generation:   self.generation,
			}),
		})
	}

	fn require_catalog_route(&self, route: &RouteId) -> Result<&RouteDef, Error> {
		self
			.catalog
			.route(route)
			.ok_or_else(|| target_error(route.as_str()))
	}
}

#[derive(Clone)]
struct RegistryDispatch {
	registry: Registry,
}

impl Service<LayerCall<Call>> for RegistryDispatch {
	type Error = Error;
	type Future = BoxFuture<'static, Result<Answer, Error>>;
	type Response = Answer;

	/// Dispatch readiness is enforced inside each exact selected route service.
	fn poll_ready(&mut self, _context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, call: LayerCall<Call>) -> Self::Future {
		let registry = self.registry.clone();
		Box::pin(async move { dispatch_preplanned(registry, call).await })
	}
}

async fn dispatch_preplanned(
	registry: Registry,
	mut layered: LayerCall<Call>,
) -> Result<Answer, Error> {
	registry.validate_plan(&layered.payload, std::time::Instant::now())?;
	let mut plan = layered
		.payload
		.execution
		.as_ref()
		.expect("validated planned call")
		.as_ref()
		.clone();
	if let OperationCall::Auth(request) = &layered.payload.operation {
		let manager = registry.inner.auth_manager.as_ref().ok_or_else(|| {
			Error::planning(
				ErrorKind::RouteUnavailable,
				ErrorDetail::Capability {
					feature: Str::from(OperationKind::Auth.to_string()),
					reason:  ReasonId(Str::from("auth-manager-not-constructed")),
				},
				layered.context.receipt(),
			)
		})?;
		let body = match manager.execute(request.as_ref().clone()).await {
			Ok(body) => body,
			Err(mut error) => {
				layered.context.finalize_error(&mut error);
				return Err(error);
			},
		};
		return Ok(Answer {
			meta:    ResponseMeta {
				request_id:          layered.payload.id.clone(),
				provider:            plan.provider.clone(),
				route:               plan.route.clone(),
				model:               None,
				provider_request_id: None,
				created_at:          SystemTime::now(),
			},
			receipt: layered.context.receipt(),
			body:    AnswerBody::Auth(body),
		});
	}
	let candidates = plan.fallbacks.iter().cloned().collect::<Vec<_>>();
	for (index, fallback) in std::iter::once(None)
		.chain(candidates.iter().map(Some))
		.enumerate()
	{
		if let Some(fallback) = fallback {
			plan.model = fallback.model.clone();
			plan.provider = fallback.provider.clone();
			plan.route = fallback.route.clone();
			plan.codec = fallback.codec.clone();
			plan.policy_model = fallback.policy_model.clone();
			plan.wire_policy = fallback.wire_policy.clone();
			plan.thinking_policy = fallback.thinking_policy.clone();
			plan.thinking_selection = fallback.thinking_selection.clone();
			plan.decisions = fallback.decisions.clone();
			plan.runtime_evidence = fallback.runtime_evidence.clone();
			plan.wire_target = fallback.wire_target.clone();
			plan.fallbacks = candidates[index..].into();
			layered.payload.execution = Some(Arc::new(plan.clone()));
		}
		let service = match registry.route_service(&plan.route, layered.payload.operation.kind()) {
			Ok(service) => service,
			Err(mut error) => {
				layered.context.finalize_error(&mut error);
				return Err(error);
			},
		};
		let attempt_start = layered.context.receipt().attempts.len();
		match service.oneshot(layered.clone()).await {
			Ok(mut answer) => {
				layered.context.merge_receipt(&answer.receipt);
				answer.receipt = layered.context.receipt();
				return Ok(answer);
			},
			Err(mut error) => {
				let has_next = index < candidates.len();
				if fallback_is_safe(&error, has_next) {
					layered.context.merge_receipt(&error.receipt);
					hide_attempts_since(&layered.context, attempt_start);
					continue;
				}
				layered.context.finalize_error(&mut error);
				return Err(error);
			},
		}
	}
	unreachable!("primary route and finite preplanned fallbacks always return")
}

fn fallback_is_safe(error: &Error, has_next: bool) -> bool {
	has_next
		&& !error.committed
		&& error.action == RetryAction::ReselectRoute
		&& error.receipt.attempts.last().is_some_and(|attempt| {
			attempt.outcome != AttemptOutcome::FailedCommitted
				&& attempt.body.retry_decision == RetryDecision::Allow
		})
}

fn hide_attempts_since(context: &ExecutionContext, start: usize) {
	context.with_receipt(|receipt| {
		for attempt in receipt.attempts.iter_mut().skip(start) {
			attempt.hidden = true;
		}
	});
}

fn route_unavailable_error(evidence: &RouteUnavailable, operation: OperationKind) -> Error {
	let reason = if evidence.operation.is_none() || evidence.operation == Some(operation) {
		evidence.reason.clone()
	} else {
		ReasonId(Str::from("route-operation-not-constructed"))
	};
	let mut error = Error::planning(
		ErrorKind::RouteUnavailable,
		ErrorDetail::Capability { feature: Str::from(operation.to_string()), reason },
		ExecutionReceipt::default(),
	);
	error.route = Some(evidence.route.clone());
	error
}

fn target_error(selector: &str) -> Error {
	Error::planning(
		ErrorKind::TargetNotFound,
		ErrorDetail::Target { selector: Str::from(selector) },
		ExecutionReceipt::default(),
	)
}

fn duplicate_route_error(route: &RouteId) -> Error {
	Error::planning(
		ErrorKind::ProviderContractMismatch,
		ErrorDetail::Capability {
			feature: Str::from(route.as_str()),
			reason:  ReasonId(Str::from("duplicate-route-registration")),
		},
		ExecutionReceipt::default(),
	)
}

#[cfg(test)]
mod tests {
	use std::{
		sync::{
			Arc,
			atomic::{AtomicUsize, Ordering},
		},
		time::{Duration, Instant, SystemTime, UNIX_EPOCH},
	};

	use futures::FutureExt as _;
	use tower::service_fn;

	use super::*;
	use crate::{
		account::AccountPool,
		answer::{AccountSummary, AuthAnswer, AuthSession},
		auth::{
			AuthLoginEngine, AuthRefreshEngine, CredentialBroker, CredentialBrokerEngines,
			CredentialStore, HeadlessKeySource, KeyId,
		},
		body::{AttemptBodyEvidence, Replayability, RetryDecisionReason},
		call::{AuthMethod, AuthRequest, LoginRequest, Target},
		error::ErrorPhase,
		id::RequestId,
		layer::stack::RouteProviderService,
		plan::{
			CapabilityAvailability, ExecutionPlan, FallbackScope, ReplayPlan, RouteHealth,
			RuntimeRouteEvidence,
		},
		receipt::{AttemptReceipt, Cost, ExecutionBudget, ProviderEvidence, Usage},
	};

	#[derive(Clone, Copy)]
	struct UnusedLogin(AuthMethod);

	impl AuthLoginEngine for UnusedLogin {
		fn method(&self) -> AuthMethod {
			self.0
		}

		fn begin(
			&self,
			_request: LoginRequest,
			_spec: crate::catalog::AuthSpecId,
		) -> futures::future::BoxFuture<'_, Result<AuthSession, Error>> {
			async { Err(test_auth_error()) }.boxed()
		}
	}

	struct UnusedRefresh;

	impl AuthRefreshEngine for UnusedRefresh {
		fn refresh(
			&self,
			_account: crate::id::AccountId,
		) -> futures::future::BoxFuture<'_, Result<AccountSummary, Error>> {
			async { Err(test_auth_error()) }.boxed()
		}
	}

	fn test_auth_error() -> Error {
		Error::new(
			ErrorKind::InternalInvariant,
			ErrorPhase::Authentication,
			RetryAction::Never,
			ExecutionReceipt::default(),
		)
	}

	fn auth_manager(catalog: Arc<Catalog>) -> (AuthManager, std::path::PathBuf) {
		let suffix = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.unwrap()
			.as_nanos();
		let path = std::env::temp_dir()
			.join(format!("omp-auth-manager-{}-{suffix}.sqlite", std::process::id()));
		let store = Arc::new(
			CredentialStore::open(
				&path,
				Arc::new(HeadlessKeySource::new(KeyId::new("registry-test"), [7; 32])),
			)
			.unwrap(),
		);
		let broker = CredentialBroker::system(&catalog, CredentialBrokerEngines::default()).unwrap();
		let methods = [
			AuthMethod::ApiKey,
			AuthMethod::OAuthPkce,
			AuthMethod::OAuthDevice,
			AuthMethod::ApplicationDefault,
			AuthMethod::AwsCredentialChain,
			AuthMethod::SessionToken,
		];
		let engines = methods
			.into_iter()
			.map(|method| Arc::new(UnusedLogin(method)) as Arc<dyn AuthLoginEngine>)
			.collect();
		let manager = AuthManager::new(
			catalog,
			store,
			broker,
			AccountPool::new(),
			engines,
			Arc::new(UnusedRefresh),
		)
		.unwrap();
		(manager, path)
	}

	fn attempt(decision: RetryDecision) -> AttemptReceipt {
		AttemptReceipt {
			index:             0,
			hidden:            false,
			provider:          None,
			route:             None,
			account:           None,
			principal:         None,
			body:              AttemptBodyEvidence {
				opened:         true,
				consumed:       decision != RetryDecision::Allow,
				replayability:  Replayability::Replayable,
				retry_decision: decision,
				reason:         if decision == RetryDecision::Allow {
					RetryDecisionReason::ReplayableSource
				} else {
					RetryDecisionReason::ConsumedOneShot
				},
			},
			outcome:           AttemptOutcome::FailedPreCommit,
			usage:             Usage::default(),
			cost:              Cost::default(),
			provider_evidence: ProviderEvidence::default(),
			elapsed:           Duration::ZERO,
		}
	}

	fn route_error(decision: Option<RetryDecision>) -> Error {
		let mut receipt = ExecutionReceipt::default();
		if let Some(decision) = decision {
			receipt.record_attempt(attempt(decision));
		}
		Error::new(
			ErrorKind::Connectivity,
			ErrorPhase::Connecting,
			RetryAction::ReselectRoute,
			receipt,
		)
	}

	#[test]
	fn fallback_requires_precommit_reselect_and_explicit_body_permission() {
		assert!(fallback_is_safe(&route_error(Some(RetryDecision::Allow)), true));
		assert!(!fallback_is_safe(&route_error(Some(RetryDecision::Suppress)), true));
		assert!(!fallback_is_safe(&route_error(None), true));
		assert!(!fallback_is_safe(&route_error(Some(RetryDecision::Allow)), false));
		let mut committed = route_error(Some(RetryDecision::Allow));
		committed.committed = true;
		assert!(!fallback_is_safe(&committed, true));
		let mut committed_attempt = route_error(Some(RetryDecision::Allow));
		committed_attempt.receipt.attempts[0].outcome = AttemptOutcome::FailedCommitted;
		assert!(!fallback_is_safe(&committed_attempt, true));
	}

	#[test]
	fn failed_fallback_receipts_are_hidden_once_in_shared_context() {
		let context = ExecutionContext::new(crate::receipt::ExecutionBudget::default());
		let mut failed = ExecutionReceipt::default();
		failed.record_attempt(attempt(RetryDecision::Allow));
		context.merge_receipt(&failed);
		hide_attempts_since(&context, 0);
		let mut success = ExecutionReceipt::default();
		let mut visible = attempt(RetryDecision::Suppress);
		visible.index = 1;
		visible.outcome = AttemptOutcome::Succeeded;
		success.record_attempt(visible);
		context.merge_receipt(&success);
		let receipt = context.receipt();
		assert_eq!(receipt.attempts.len(), 2);
		assert!(receipt.attempts[0].hidden);
		assert!(!receipt.attempts[1].hidden);
		assert_eq!((receipt.attempts[0].index, receipt.attempts[1].index), (0, 1));
	}
	#[tokio::test]
	async fn auth_operations_bypass_route_codec_service() {
		let catalog = Arc::new(Catalog::embedded().clone());
		let route = catalog.routes().first().unwrap().clone();
		let provider = catalog.provider(&route.provider).unwrap().clone();
		let wire_policy = Arc::new(catalog.wire_policy(&provider.wire_policy).unwrap().clone());
		let (manager, store_path) = auth_manager(catalog.clone());
		let wire_calls = Arc::new(AtomicUsize::new(0));
		let calls = wire_calls.clone();
		let service = RouteProviderService::new(service_fn(move |_call: LayerCall<Call>| {
			calls.fetch_add(1, Ordering::Relaxed);
			async { Err::<Answer, Error>(test_auth_error()) }
		}));
		let registry = Registry {
			inner: Arc::new(RegistryInner {
				catalog:      catalog.clone(),
				bindings:     HashMap::from([(route.id.clone(), RouteBinding::Available(service))]),
				auth_manager: Some(manager),
				generation:   1,
			}),
		};
		let budget = ExecutionBudget::default();
		let now = Instant::now();
		let plan = ExecutionPlan {
			planned_at: SystemTime::now(),
			catalog_revision: catalog.revision().clone(),
			registry_generation: 1,
			expires_at: now + Duration::from_secs(30),
			operation: OperationKind::Auth,
			model: None,
			provider: provider.id.clone(),
			route: route.id.clone(),
			codec: route.codec.clone(),
			policy_model: None,
			wire_policy,
			thinking_policy: None,
			thinking_selection: None,
			decisions: Arc::from([]),
			fallback_scope: FallbackScope { primary: None, explicit: Arc::from([]) },
			fallbacks: Arc::from([]),
			replay: ReplayPlan::Replayable,
			budget: budget.clone(),
			runtime_evidence: RuntimeRouteEvidence {
				route:            route.id.clone(),
				generation:       1,
				health:           RouteHealth::Unknown,
				quota_millionths: 0,
				latency:          Duration::MAX,
				affinity:         false,
				operation:        CapabilityAvailability::Native,
				capabilities:     Arc::from([]),
			},
			wire_target: None,
		};
		let call = Call {
			id:        RequestId::from("auth-bypass"),
			target:    Target::ProviderService(provider.id),
			deadline:  None,
			budget:    budget.clone(),
			session:   None,
			execution: Some(Arc::new(plan)),
			operation: OperationCall::Auth(Arc::new(AuthRequest::ListAccounts { provider: None })),
		};
		let answer = dispatch_preplanned(registry, LayerCall {
			payload: call,
			context: ExecutionContext::new(budget),
		})
		.await
		.unwrap();
		assert!(
			matches!(answer.body, AnswerBody::Auth(AuthAnswer::Accounts(accounts)) if accounts.is_empty())
		);
		assert_eq!(wire_calls.load(Ordering::Relaxed), 0);
		let _ = std::fs::remove_file(store_path);
	}
}
