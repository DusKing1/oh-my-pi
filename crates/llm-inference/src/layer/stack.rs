//! Construction-time composition of the one fixed inference service stack.

use omp_llm_catalog::{provider::RouteDef, snapshot::Catalog};
use tower::Layer;

use super::{
	account::{AccountPoolLayer, AccountPoolService},
	admission::{AdmissionLayer, AdmissionService},
	answer::{AnswerLayer, AnswerService},
	attempt::{AttemptLayer, AttemptService},
	auth::{AuthLeaseLayer, AuthLeaseService},
	budget::{OverallBudgetLayer, OverallBudgetService},
	encode::{EncodeLayer, EncodeService},
	intent::{IntentLayer, IntentService},
	observe::{ObserveLayer, ObserveService},
	operation::{OperationPolicyLayer, OperationPolicyService},
	rate::{RateLayer, RateService},
	recover::{RecoveryLayer, RecoveryService},
	retry::{TransportRetryLayer, TransportRetryService},
	semantic::{SemanticLayer, SemanticService},
	session::{SessionLayer, SessionService},
};
use crate::{Answer, Call, Error, layer::LayerCall, registry::RouteUnavailable};

/// Construction-time erased route service that reuses the outer logical
/// execution context.
pub type RouteProviderService = tower::util::BoxCloneSyncService<LayerCall<Call>, Answer, Error>;

/// Construction-time route-stack factory consumed by the immutable registry
/// builder.
pub trait RouteStackFactory: Send + Sync + 'static {
	/// Builds the comprehensive service for one catalog route or returns typed
	/// construction evidence.
	fn build(
		&self,
		catalog: &Catalog,
		route: &RouteDef,
	) -> Result<RouteProviderService, RouteUnavailable>;
}

/// Complete production route composer supplied with account, auth, session,
/// codec, rate, transport, projection, and observation dependencies by the
/// application composition root.
pub trait RouteComposer: Send + Sync + 'static {
	/// Composes the fixed stack once for a concrete route.
	fn compose(
		&self,
		catalog: &Catalog,
		route: &RouteDef,
	) -> Result<RouteProviderService, RouteUnavailable>;
}

/// Production configuration for built-in route construction.
#[derive(Clone)]
pub struct BuiltinConfig {
	composer:     std::sync::Arc<dyn RouteComposer>,
	auth_manager: Option<crate::auth::manager::AuthManager>,
}
impl BuiltinConfig {
	/// Creates configuration from a production composer owning all route-scoped
	/// dependencies.
	pub fn new(composer: std::sync::Arc<dyn RouteComposer>) -> Self {
		Self { composer, auth_manager: None }
	}

	/// Creates the canonical production composer from explicit shared
	/// dependencies.
	pub fn production(dependencies: crate::provider::builtin::ProductionDependencies) -> Self {
		let auth_manager = dependencies.auth_manager();
		Self {
			composer:     std::sync::Arc::new(crate::provider::builtin::ProductionRouteComposer::new(
				dependencies,
			)),
			auth_manager: Some(auth_manager),
		}
	}

	/// Attaches the comprehensive auth-management service used by provider-level
	/// auth operations.
	pub fn with_auth_manager(mut self, auth_manager: crate::auth::manager::AuthManager) -> Self {
		self.auth_manager = Some(auth_manager);
		self
	}

	/// Borrows the auth manager for registry management-service injection.
	pub(crate) fn auth_manager(&self) -> Option<&crate::auth::manager::AuthManager> {
		self.auth_manager.as_ref()
	}
}

/// Registry-facing factory that delegates only construction; request stacks are
/// never rebuilt.
#[derive(Clone)]
pub struct BuiltinRouteStackFactory {
	config: BuiltinConfig,
}
impl BuiltinRouteStackFactory {
	/// Creates a factory from complete production dependencies.
	pub const fn new(config: BuiltinConfig) -> Self {
		Self { config }
	}
}
impl RouteStackFactory for BuiltinRouteStackFactory {
	fn build(
		&self,
		catalog: &Catalog,
		route: &RouteDef,
	) -> Result<RouteProviderService, RouteUnavailable> {
		self.config.composer.compose(catalog, route)
	}
}

/// Construction inputs for a route-local stack; outer execution state is
/// supplied by the registry.
#[derive(Clone)]
pub struct RouteStackLayers<I, SS, SM, AP, AC, RL, EN, CA> {
	/// Canonical operation-specific planning and response validation.
	pub operation:        OperationPolicyLayer,
	/// Intent negotiation.
	pub intent:           IntentLayer<I>,
	/// Session strategy and reseed policy.
	pub session:          SessionLayer<SS>,
	/// Transactional semantic attempts.
	pub semantic:         SemanticLayer<SM>,
	/// Route-scoped recovery and conservative discovery projection.
	pub recovery:         RecoveryLayer,
	/// Route/account admission.
	pub admission:        AdmissionLayer,
	/// Account selection.
	pub account:          AccountPoolLayer<AP>,
	/// Opaque credential lease acquisition.
	pub auth:             AuthLeaseLayer<AC>,
	/// Replay-safe same-route retry.
	pub retry:            TransportRetryLayer,
	/// Per-transport-attempt rate reservation.
	pub rate:             RateLayer<RL>,
	/// Pure codec lowering.
	pub encode:           EncodeLayer<EN>,
	/// Credential application immediately before transport.
	pub credential_apply: CA,
}

/// Stack segment from credential application through canonical recovery.
pub type RecoveryStack<W, CA, EN, RL, AC, AP>
where
	CA: Layer<W>,
= RecoveryService<
	AttemptService<
		AdmissionService<
			AccountPoolService<
				AuthLeaseService<
					TransportRetryService<RateService<EncodeService<<CA as Layer<W>>::Service, EN>, RL>>,
					AC,
				>,
				AP,
			>,
		>,
	>,
>;
/// Stack segment through semantic validation and typed answer projection.
pub type AnswerStack<W, CA, EN, RL, AC, AP, SM>
where
	CA: Layer<W>,
= AnswerService<SemanticService<RecoveryStack<W, CA, EN, RL, AC, AP>, SM>>;
/// Stack segment from intent through session and response processing.
pub type IntentStack<W, CA, EN, RL, AC, AP, SM, SS, I>
where
	CA: Layer<W>,
= OperationPolicyService<
	IntentService<SessionService<AnswerStack<W, CA, EN, RL, AC, AP, SM>, SS>, I>,
>;
/// Outer execution service type wrapping the full registry fallback loop
/// exactly once.
pub type OuterExecutionService<S, O> = ObserveService<OverallBudgetService<S>, O>;

/// Applies route-local layers exactly once; the returned service accepts an
/// existing `LayerCall`.
///
/// Outer to inner: Intent → Session → Answer → Semantic → Recovery →
/// Attempt(Admission → AccountPool → AuthLease → TransportRetry → Rate → Encode
/// → CredentialApply → WireTransport).
pub fn build_route_stack<W, I, SS, SM, AP, AC, RL, EN, CA>(
	wire: W,
	layers: RouteStackLayers<I, SS, SM, AP, AC, RL, EN, CA>,
) -> IntentStack<W, CA, EN, RL, AC, AP, SM, SS, I>
where
	CA: Layer<W>,
	EN: Clone,
	RL: Clone,
	AC: Clone,
	AP: Clone,
	SM: Clone,
	SS: Clone,
	I: Clone,
{
	let service = layers.credential_apply.layer(wire);
	let service = layers.encode.layer(service);
	let service = layers.rate.layer(service);
	let service = layers.retry.layer(service);
	let service = layers.auth.layer(service);
	let service = layers.account.layer(service);
	let service = layers.admission.layer(service);
	let service = AttemptLayer.layer(service);
	let service = layers.recovery.layer(service);
	let service = layers.semantic.layer(service);
	let service = AnswerLayer.layer(service);
	let service = layers.session.layer(service);
	layers.operation.layer(layers.intent.layer(service))
}

/// Wraps the complete preplanned registry fallback service in one budget and
/// observation boundary.
pub fn build_execution_stack<S, O>(
	dispatch: S,
	observer: ObserveLayer<O>,
) -> OuterExecutionService<S, O>
where
	O: Clone,
{
	observer.layer(OverallBudgetLayer.layer(dispatch))
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use parking_lot::Mutex;
	use tower::Layer;

	use super::{RouteStackLayers, build_route_stack};
	use crate::layer::{
		account::AccountPoolLayer,
		admission::{AdmissionController, AdmissionLayer},
		auth::AuthLeaseLayer,
		encode::EncodeLayer,
		intent::IntentLayer,
		operation::{OperationPolicyConfig, OperationPolicyLayer},
		rate::RateLayer,
		recover::RecoveryLayer,
		retry::TransportRetryLayer,
		semantic::SemanticLayer,
		session::SessionLayer,
	};

	#[derive(Clone)]
	struct CountLayer {
		name:  &'static str,
		trace: Arc<Mutex<Vec<&'static str>>>,
	}
	impl<S> Layer<S> for CountLayer {
		type Service = S;

		fn layer(&self, inner: S) -> S {
			self.trace.lock().push(self.name);
			inner
		}
	}

	#[test]
	fn stack_is_composed_once_at_construction() {
		let trace = Arc::new(Mutex::new(Vec::new()));
		let layers = RouteStackLayers {
			operation:        OperationPolicyLayer::new(OperationPolicyConfig {
				embedding:              None,
				native:                 None,
				usage:                  crate::operation::usage::UsageServiceConfig::new(
					std::time::Duration::ZERO,
				),
				discovery_maximum_page: None,
				exact_token_count:      false,
			}),
			intent:           IntentLayer::new(()),
			session:          SessionLayer::new(()),
			semantic:         SemanticLayer::new(()),
			recovery:         RecoveryLayer::without_discovery(),
			admission:        AdmissionLayer::new(AdmissionController::new(1, 0)),
			account:          AccountPoolLayer::new(()),
			auth:             AuthLeaseLayer::new(()),
			retry:            TransportRetryLayer::new(0),
			rate:             RateLayer::new(()),
			encode:           EncodeLayer::new((), false),
			credential_apply: CountLayer { name: "credential", trace: trace.clone() },
		};
		let _stack = build_route_stack((), layers);
		assert_eq!(&*trace.lock(), &["credential"]);
	}
}
