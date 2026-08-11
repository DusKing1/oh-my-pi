//! Once-built typed provider route stack.
//!
//! Construction happens once when a provider route is registered. Calls only
//! drive the already-composed concrete service; no layer is assembled and no
//! future is erased on the request path.

use std::{
	sync::Arc,
	task::{Context, Poll},
	time::Duration,
};

use futures::{Stream, StreamExt, TryFutureExt, future::MapOk, stream::Map};
use omp_core::Str;
use omp_llm_catalog::{compat::Compat, identity::DialectSelection};
use omp_llm_error::{BlockTable, RetryBudget};
use omp_proto::inference::v1::TurnEvent;
use parking_lot::Mutex;
use tower::{Layer, Service};

use crate::{
	admission::{Admission, AdmissionLayer},
	cache::{Cache, CacheLayer, CachePolicy},
	dialect::{OwnedDialect, OwnedDialectConfig, OwnedDialectLayer},
	envelope::TurnRequestEnvelope,
	learn::{Learn, LearnLayer, RequestRepair, ScopeFn},
	preflight::{Preflight, PreflightConfig, PreflightLayer, UsageOracle},
	recovery::{Recovery, RecoveryConfig, RecoveryLayer},
	refresh::{CredentialRefresher, Refresh, RefreshConfig, RefreshLayer},
	resample::{AttemptEvent, Resample, ResampleConfig, ResampleLayer},
	select::{CredentialPool, LeaseSource, Select, SelectLayer},
	stack::{
		combinators::{ForcedTool, ForcedToolLayer, ProductionPolicy, ProductionPolicyLayer},
		meter::{ObserveUsage, UsageObserver, UsageObserverLayer},
	},
	tap::{FrameSink, Tap, TapLayer},
	timeout::{PhaseTimeout, PhaseTimeoutConfig, PhaseTimeoutLayer},
};

/// Trait dependencies owned by one provider route.
///
/// Credential material is deliberately absent. `leases` resolves only opaque
/// lease identity; the innermost attempt service hands that identity to egress,
/// which owns secret lookup and header mutation.
pub struct RouteDependencies {
	/// Usage/quota authority consulted before dispatch.
	pub usage:          Arc<dyn UsageOracle>,
	/// Ranked credential ids for the selected provider/model.
	pub credentials:    Arc<dyn CredentialPool>,
	/// Current opaque lease generations for ranked ids.
	pub leases:         Arc<dyn LeaseSource>,
	/// Provider credential refresh operation.
	pub refresher:      Arc<dyn CredentialRefresher>,
	/// Capability-specific canonical request repair.
	pub repair:         Arc<dyn RequestRepair>,
	/// Request/frame observer without credential routing metadata.
	pub observer:       Arc<dyn FrameSink>,
	/// Authoritative terminal usage observer with the exact selected lease.
	pub usage_observer: Arc<dyn UsageObserver>,
	/// Shared credential block table for selection and rotation.
	pub blocks:         Arc<Mutex<BlockTable>>,
}

/// Replay-capable route policy assembled by [`RouteStackBuilder`].
#[derive(Clone)]
pub struct RouteStackConfig {
	/// Usage preflight behavior.
	pub preflight:            PreflightConfig,
	/// Exact-request recovery budget and normalization behavior.
	pub recovery:             RecoveryConfig,
	/// OAuth refresh timing.
	pub refresh:              RefreshConfig,
	/// Semantic empty/loop resampling policy.
	pub resample:             ResampleConfig,
	/// Catalog-selected stream and forced-tool compatibility axes.
	pub compat:               Compat,
	/// Maximum attempts in forced-tool emulation, including the first.
	pub forced_tool_attempts: u32,
	/// Owned model-prompt dialect selection. `None` keeps non-production test
	/// stacks provider-native; production registration promotes it to `Auto`.
	pub dialect:              Option<DialectSelection>,
	/// Captured `OMP_DIALECT` override, resolved against each request's model.
	pub omp_dialect:          Option<Str>,
	/// Maximum streams simultaneously admitted to this route.
	pub max_inflight:         usize,
	/// Provider call/first-event/idle deadlines.
	pub timeout:              PhaseTimeoutConfig,
	/// Prompt-cache breakpoint policy and keep-alive budget.
	pub cache:                CachePolicy,
	/// Optional endpoint/region namespace added to provider/model/account
	/// learning scope.
	pub learn_scope:          Option<ScopeFn>,
	/// Explicit lifetime of learned provider capability failures.
	pub learn_expiry:         Duration,
}

impl Default for RouteStackConfig {
	fn default() -> Self {
		Self {
			preflight:            PreflightConfig::default(),
			recovery:             RecoveryConfig {
				budget: RetryBudget::default(),
				..RecoveryConfig::default()
			},
			refresh:              RefreshConfig::default(),
			resample:             ResampleConfig::default(),
			compat:               Compat::default(),
			forced_tool_attempts: 3,
			dialect:              None,
			omp_dialect:          None,
			max_inflight:         8,
			timeout:              PhaseTimeoutConfig::default(),
			cache:                CachePolicy::default(),
			learn_scope:          None,
			learn_expiry:         Duration::from_hours(6),
		}
	}
}

/// Builder holding the route's once-allocated layers and shared state.
///
/// The altitude is fixed, outermost first: tap → preflight → recovery → select
/// → terminal usage → refresh → learn → resample → forced-tool → production
/// recovery → admission → timeout → owned dialect → provider attempt.
/// The two structural adapters around `resample` only witness the coordinator's
/// pre-commit domain; they do not copy or alter an event.
pub struct RouteStackBuilder {
	tap:            TapLayer,
	preflight:      PreflightLayer,
	recovery:       RecoveryLayer,
	select:         SelectLayer,
	usage_observer: UsageObserverLayer,
	cache:          CacheLayer,
	refresh:        RefreshLayer,
	learn:          LearnLayer,
	resample:       ResampleLayer,
	forced:         ForcedToolLayer,
	production:     ProductionPolicyLayer,
	admission:      AdmissionLayer,
	timeout:        PhaseTimeoutLayer,
	dialect:        OwnedDialectLayer,
}

impl RouteStackBuilder {
	/// Allocates all route-shared policy state exactly once.
	#[must_use]
	pub fn new(dependencies: RouteDependencies, config: RouteStackConfig) -> Self {
		let learn = config
			.learn_scope
			.map_or_else(
				|| LearnLayer::new(Arc::clone(&dependencies.repair)),
				|scope| LearnLayer::new(Arc::clone(&dependencies.repair)).with_scope(scope),
			)
			.with_expiry(config.learn_expiry);
		Self {
			tap: TapLayer::new(dependencies.observer),
			preflight: PreflightLayer::new(dependencies.usage, config.preflight),
			recovery: RecoveryLayer::new(config.recovery),
			select: SelectLayer::new(
				dependencies.credentials,
				dependencies.leases,
				dependencies.blocks,
			),
			usage_observer: UsageObserverLayer::new(dependencies.usage_observer),
			cache: CacheLayer::new(config.cache),
			refresh: RefreshLayer::new(dependencies.refresher, config.refresh),
			learn,
			resample: ResampleLayer::new(config.resample),
			forced: ForcedToolLayer::new(config.compat, config.forced_tool_attempts),
			production: ProductionPolicyLayer::new(config.compat),
			dialect: OwnedDialectLayer::new(
				OwnedDialectConfig::new(
					config.dialect.unwrap_or(DialectSelection::Native),
					config.compat,
				)
				.with_override(config.omp_dialect),
			),
			admission: AdmissionLayer::new(config.max_inflight),
			timeout: PhaseTimeoutLayer::new(config.timeout),
		}
	}

	/// Consumes the builder and composes one reusable concrete route service.
	///
	/// `attempt` receives [`crate::select::Routed`] directly. Every lower
	/// replay-capable layer clones that envelope, so the selected lease remains
	/// out-of-band and is retained on every legal re-dispatch.
	pub fn build<S>(self, attempt: S) -> RouteStack<S> {
		let dialect = self.dialect.layer(attempt);
		let timeout = self.timeout.layer(dialect);
		let admission = self.admission.layer(timeout);
		let production = self.production.layer(admission);
		let forced = self.forced.layer(production);
		let attempt_events = TurnsIntoAttemptEvents::new(forced);
		let resample = self.resample.layer(attempt_events);
		let turn_events = AttemptEventsIntoTurns::new(resample);
		let learn = self.learn.layer(turn_events);
		let refresh = self.refresh.layer(learn);
		let observed = self.usage_observer.layer(refresh);
		// Below `select` so a refresh inherits the same lease, above metering so
		// its reads are accounted like any other dispatch.
		let cache = self.cache.layer(observed);
		let select = self.select.layer(cache);
		let recovery = self.recovery.layer(select);
		let preflight = self.preflight.layer(recovery);
		self.tap.layer(preflight)
	}
}

/// Concrete service returned by [`RouteStackBuilder::build`].
pub type RouteStack<S> = Tap<
	Preflight<
		Recovery<
			Select<
				Cache<
					ObserveUsage<
						Refresh<
							Learn<
								AttemptEventsIntoTurns<
									Resample<
										TurnsIntoAttemptEvents<
											ForcedTool<
												ProductionPolicy<Admission<PhaseTimeout<OwnedDialect<S>>>>,
											>,
										>,
									>,
								>,
							>,
						>,
					>,
				>,
			>,
		>,
	>,
>;

/// Structural witness that canonical frames are still pre-commit.
#[derive(Clone, Debug)]
pub struct TurnsIntoAttemptEvents<S> {
	inner: S,
}

impl<S> TurnsIntoAttemptEvents<S> {
	const fn new(inner: S) -> Self {
		Self { inner }
	}
}

/// Stream mapping used by [`TurnsIntoAttemptEvents`].
pub type AttemptEvents<St> = Map<St, fn(TurnEvent) -> AttemptEvent>;

impl<S, St, R> Service<R> for TurnsIntoAttemptEvents<S>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St>,
	St: Stream<Item = TurnEvent>,
{
	type Error = S::Error;
	type Future = MapOk<S::Future, fn(St) -> AttemptEvents<St>>;
	type Response = AttemptEvents<St>;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, req: R) -> Self::Future {
		self
			.inner
			.call(req)
			.map_ok(to_attempt_events::<St> as fn(St) -> AttemptEvents<St>)
	}
}

fn to_attempt_events<St: Stream<Item = TurnEvent>>(stream: St) -> AttemptEvents<St> {
	stream.map(AttemptEvent::new as fn(TurnEvent) -> AttemptEvent)
}

/// Structural conversion back to the canonical frame plane above resampling.
#[derive(Clone, Debug)]
pub struct AttemptEventsIntoTurns<S> {
	inner: S,
}

impl<S> AttemptEventsIntoTurns<S> {
	const fn new(inner: S) -> Self {
		Self { inner }
	}
}

/// Stream mapping used by [`AttemptEventsIntoTurns`].
pub type TurnEvents<St> = Map<St, fn(AttemptEvent) -> TurnEvent>;

impl<S, St, R> Service<R> for AttemptEventsIntoTurns<S>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St>,
	St: Stream<Item = AttemptEvent>,
{
	type Error = S::Error;
	type Future = MapOk<S::Future, fn(St) -> TurnEvents<St>>;
	type Response = TurnEvents<St>;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, req: R) -> Self::Future {
		self
			.inner
			.call(req)
			.map_ok(to_turn_events::<St> as fn(St) -> TurnEvents<St>)
	}
}

fn to_turn_events<St: Stream<Item = AttemptEvent>>(stream: St) -> TurnEvents<St> {
	stream.map(AttemptEvent::into_inner as fn(AttemptEvent) -> TurnEvent)
}
