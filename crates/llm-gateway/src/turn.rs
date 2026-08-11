//! Stateful orchestration for the bidirectional chat-turn RPC.
//!
//! The response stream owns both the context guard and the upstream stream.
//! Dropping it therefore drops the upstream future and rolls the guard back;
//! cancellation has no cooperative flag that either side could ignore.

use std::{collections::BTreeMap, fmt, sync::Arc};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{Stream, StreamExt, stream::BoxStream};
use omp_core::{Str, fmts};
use omp_llm_catalog::{
	models::{Availability, ModelWire},
	provider::{Facet as CatalogFacet, TransportId},
	registry::{ListFilter, Registry},
};
use omp_llm_types::{
	Accuracy, ChatOutcome, ChatParams, ChatRequest, ContextRef, ConvertError, Cost, Effort,
	ExecOutcome, ExecStatus, Fallback, Feature, Invoke, InvokeComplete, InvokeInput, Item, ItemKind,
	Props, Reasoning, ResolvedModelPolicy, StopReason, Thread, ThreadDelta, TurnError,
	TurnErrorKind, TurnEvent, Unsupported, UnsupportedAction,
	facet::{Chat, Error as FacetError, Executor},
};
use omp_proto::inference::v1 as pb;
use omp_telemetry::{
	collector::Usage as MetricUsage,
	config::{
		ChatUsageEvent, ChatUsageSnapshot, CostDelta, CostEstimate, CostEstimatorContext,
		TelemetryAttributeContext, TelemetryConfig, TelemetryHookContext, TelemetrySpanKind,
		UsageAccuracy,
	},
	content::{RequestContent, ResponseContent},
	metrics::{ChatUsageMetric, MetricAgent, MetricRecorder},
	span,
};
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashMap;
use serde_json::Value;
use smallvec::SmallVec;
use tokio::sync::oneshot;
use tonic::{Request, Response, Status, Streaming};

use crate::context::{Begin, BeginInput, ContextStore, SessionAffinity, TurnGuard};

const EARLY_FRAME_LIMIT: usize = 64;

/// Server stream returned by [`TurnEngine::turn`].
pub type TurnStream = BoxStream<'static, Result<pb::TurnEvent, Status>>;

/// One credential-bound chat stack registered for a provider.
///
/// `chat` is the already-assembled capability → rotation → meter → codec stack;
/// this resolver only selects the catalog row and credential-bound stack.
#[derive(Clone)]
pub struct ChatRoute {
	/// Provider catalog id.
	pub provider:          Str,
	/// Credential identity pinned when this route is selected.
	pub credential_id:     Str,
	/// Whether the transport cannot progress without an in-turn executor.
	pub requires_executor: bool,
	/// Fully assembled chat facet stack.
	pub chat:              Arc<dyn Chat>,
}

impl fmt::Debug for ChatRoute {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("ChatRoute")
			.field("provider", &self.provider)
			.field("credential_id", &self.credential_id)
			.field("requires_executor", &self.requires_executor)
			.finish_non_exhaustive()
	}
}

/// Fully resolved transport tuple used to select a once-built provider stack.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct ChatRouteKey {
	pub(crate) transport: TransportId,
	pub(crate) base_url:  Str,
}

impl ChatRouteKey {
	fn from_model(
		wire: Option<&ModelWire>,
		default: &Self,
		base_url_overridden: bool,
		transport_overridden: bool,
	) -> Self {
		wire.map_or_else(
			|| default.clone(),
			|wire| Self {
				transport: if transport_overridden {
					default.transport
				} else {
					wire.transport
				},
				base_url:  if base_url_overridden {
					default.base_url.clone()
				} else {
					wire
						.base_url
						.clone()
						.unwrap_or_else(|| default.base_url.clone())
				},
			},
		)
	}
}

#[derive(Clone)]
struct RegisteredChatRoute {
	wire:              Option<ChatRouteKey>,
	credential_id:     Option<Str>,
	requires_executor: bool,
	chat:              Arc<dyn Chat>,
}

#[derive(Clone)]
struct CachedModelPolicy {
	updated_at_ms: u64,
	behavior:      omp_llm_catalog::models::ModelBehavior,
	policy:        Arc<ResolvedModelPolicy>,
}

/// Catalog resolver for once-built chat stacks.
///
/// Legacy test and specialized routes may bind a credential explicitly.
/// Production HTTP stacks own credential selection and session pinning in
/// their selection middleware.
pub struct ChatResolver {
	registry: Arc<RwLock<Registry>>,
	routes:   RwLock<BTreeMap<Str, SmallVec<RegisteredChatRoute, 2>>>,
	defaults: RwLock<BTreeMap<Str, (ChatRouteKey, bool, bool)>>,
	policies: RwLock<BTreeMap<Str, CachedModelPolicy>>,
}
impl ChatResolver {
	/// Creates an empty resolver over the live catalog registry.
	#[must_use]
	pub fn new(registry: Arc<RwLock<Registry>>) -> Self {
		Self {
			registry,
			routes: RwLock::new(BTreeMap::new()),
			defaults: RwLock::new(BTreeMap::new()),
			policies: RwLock::new(BTreeMap::new()),
		}
	}

	/// Registers one fully assembled, credential-bound provider stack.
	pub fn register(&self, route: ChatRoute) {
		self.register_route(
			route.provider,
			None,
			Some(route.credential_id),
			route.requires_executor,
			route.chat,
		);
	}

	/// Registers one provider stack that owns credential selection internally.
	///
	/// This is the production registration path for a once-built route stack;
	/// unlike [`Self::register`], it does not invent a gateway credential pin.
	pub fn register_stack(&self, provider: Str, requires_executor: bool, chat: Arc<dyn Chat>) {
		self.register_route(provider, None, None, requires_executor, chat);
	}

	/// Registers a production stack for one distinct provider/transport/base
	/// tuple.
	pub(crate) fn register_wire_stack(
		&self,
		provider: Str,
		wire: ChatRouteKey,
		default: ChatRouteKey,
		base_url_overridden: bool,
		transport_overridden: bool,
		requires_executor: bool,
		chat: Arc<dyn Chat>,
	) {
		self
			.defaults
			.write()
			.insert(provider.clone(), (default, base_url_overridden, transport_overridden));
		self.register_route(provider, Some(wire), None, requires_executor, chat);
	}

	/// Returns the distinct effective tuples required by this provider's cards.
	pub(crate) fn wire_routes(
		&self,
		provider: &Str,
		default: &ChatRouteKey,
		base_url_overridden: bool,
		transport_overridden: bool,
	) -> Vec<ChatRouteKey> {
		let mut filter = ListFilter::default();
		filter.provider = Some(provider.clone());
		filter.facet = Some(CatalogFacet::Chat);
		let registry = self.registry.read();
		let (cards, _) = registry.list(&filter);
		let mut routes = std::collections::BTreeSet::new();
		// Keep the provider tuple ready for later discovery cards, which do not
		// carry static Pi wire metadata.
		routes.insert(default.clone());
		routes.extend(cards.iter().map(|card| {
			ChatRouteKey::from_model(
				card.wire.as_ref(),
				default,
				base_url_overridden,
				transport_overridden,
			)
		}));
		routes.into_iter().collect()
	}

	fn register_route(
		&self,
		provider: Str,
		wire: Option<ChatRouteKey>,
		credential_id: Option<Str>,
		requires_executor: bool,
		chat: Arc<dyn Chat>,
	) {
		let mut routes = self.routes.write();
		let candidates = routes.entry(provider).or_default();
		let replacement = RegisteredChatRoute { wire, credential_id, requires_executor, chat };
		if let Some(existing) = candidates.iter_mut().find(|route| {
			route.wire == replacement.wire && route.credential_id == replacement.credential_id
		}) {
			*existing = replacement;
		} else {
			candidates.push(replacement);
		}
	}

	// Registry cards have already completed reference inheritance and variant
	// collapse. Cache only at this post-overlay boundary. Behavior equality
	// protects same-timestamp dynamic overlays; stable hits only compare and
	// bump the Arc. Discovery churn is bounded and evicts the lexicographically
	// first id deterministically on a new miss.
	fn policy_for(&self, card: &omp_llm_catalog::models::ModelCard) -> Arc<ResolvedModelPolicy> {
		const MAX_CACHED_MODEL_POLICIES: usize = 8_192;

		if let Some(cached) = self.policies.read().get(&card.id)
			&& cached.updated_at_ms == card.updated_at_ms
			&& cached.behavior == card.behavior
		{
			return Arc::clone(&cached.policy);
		}
		let mut policies = self.policies.write();
		if let Some(cached) = policies.get(&card.id)
			&& cached.updated_at_ms == card.updated_at_ms
			&& cached.behavior == card.behavior
		{
			return Arc::clone(&cached.policy);
		}
		if !policies.contains_key(&card.id)
			&& policies.len() >= MAX_CACHED_MODEL_POLICIES
			&& let Some(evicted) = policies.keys().next().cloned()
		{
			policies.remove(&evicted);
		}
		let policy = Arc::new(card.behavior.resolved_policy());
		policies.insert(card.id.clone(), CachedModelPolicy {
			updated_at_ms: card.updated_at_ms,
			behavior:      card.behavior.clone(),
			policy:        Arc::clone(&policy),
		});
		policy
	}

	fn resolve(
		&self,
		requested: &str,
		effort: Option<Effort>,
		affinity: Option<&SessionAffinity>,
	) -> Result<ResolvedChat, TurnError> {
		let card = {
			let registry = self.registry.read();
			if let Some(card) = registry.resolve_role(requested) {
				card.clone()
			} else {
				let mut filter = ListFilter::default();
				filter.facet = Some(CatalogFacet::Chat);
				filter.available_only = true;
				let (cards, _) = registry.list(&filter);
				cards
					.into_iter()
					.find(|card| card.id == requested || card.model == requested)
					.ok_or_else(|| {
						error(TurnErrorKind::Unsupported, "requested chat model is unavailable")
					})?
			}
		};
		if card.availability != Availability::Available {
			return Err(error(TurnErrorKind::Auth, "requested chat model has no usable credential"));
		}
		let policy = self.policy_for(&card);
		let (effective_effort, preflight_unsupported) = if policy.thinking.is_some() {
			effective_effort(effort, policy.thinking.as_ref())?
		} else {
			(
				effort.filter(|effort| {
					card.efforts.contains(effort) || card.effort_routing.contains_key(effort)
				}),
				None,
			)
		};
		let fallback_model = policy.request_model_id.as_ref().unwrap_or(&card.model);
		let model = effective_effort
			.and_then(|effort| {
				policy
					.thinking
					.as_ref()
					.and_then(|thinking| thinking.effort_routing.get(&effort))
					.or_else(|| card.effort_routing.get(&effort))
			})
			.unwrap_or(fallback_model)
			.clone();
		let route_id = route_identity(card.provider.as_str(), card.model.as_str());
		let pinned_credential = affinity
			.filter(|affinity| affinity.route_id.as_ref() == Some(&route_id))
			.and_then(|affinity| affinity.credential_id.as_deref());
		let routes = self.routes.read();
		let candidates = routes
			.get(&card.provider)
			.ok_or_else(|| error(TurnErrorKind::Unsupported, "provider has no chat stack"))?;
		let default_wire = self.defaults.read().get(&card.provider).cloned();
		let desired_wire = default_wire.as_ref().map(|(default, base_url, transport)| {
			ChatRouteKey::from_model(card.wire.as_ref(), default, *base_url, *transport)
		});
		let candidates: SmallVec<&RegisteredChatRoute, 2> = candidates
			.iter()
			.filter(|route| route.wire.is_none() || route.wire == desired_wire)
			.collect();
		if candidates.is_empty() {
			return Err(error(
				TurnErrorKind::Unsupported,
				"model wire route has no registered chat stack",
			));
		}
		let route = match pinned_credential {
			Some(credential) => candidates
				.iter()
				.copied()
				.find(|route| route.credential_id.as_deref() == Some(credential))
				.or_else(|| {
					candidates
						.iter()
						.copied()
						.find(|route| route.credential_id.is_none())
				})
				.ok_or_else(|| error(TurnErrorKind::Auth, "session credential is no longer usable"))?,
			None => candidates
				.first()
				.copied()
				.ok_or_else(|| error(TurnErrorKind::Auth, "provider has no usable credential"))?,
		};
		Ok(ResolvedChat {
			route_id,
			provider: card.provider,
			model,
			logical_model: card.model,
			policy,
			effective_effort,
			preflight_unsupported,
			credential_id: route.credential_id.clone(),
			requires_executor: route.requires_executor,
			chat: Arc::clone(&route.chat),
		})
	}
}

/// Native chat facade that routes every request through a shared catalog
/// resolver.
///
/// This adapter lets foreign-wire facades install one [`Chat`] implementation
/// in [`omp_llm_types::facet::Facets`] without bypassing model resolution,
/// session affinity, or any once-built provider stack.
#[derive(Clone)]
pub struct RoutedChat {
	resolver: Arc<ChatResolver>,
}

impl RoutedChat {
	/// Creates a native facade over `resolver`.
	#[must_use]
	pub const fn new(resolver: Arc<ChatResolver>) -> Self {
		Self { resolver }
	}
}

#[async_trait]
impl Chat for RoutedChat {
	async fn turn(
		&self,
		mut request: ChatRequest,
		executor: Option<Arc<dyn Executor>>,
	) -> Result<BoxStream<'static, TurnEvent>, FacetError> {
		let explicit_effort = request
			.thinking
			.as_ref()
			.and_then(|thinking| thinking.value.effort);
		let resolved = self
			.resolver
			.resolve(request.model.as_str(), explicit_effort, None)
			.map_err(resolver_facet_error)?;
		if resolved.requires_executor && executor.is_none() {
			return Err(resolver_facet_error(unsupported_executor()));
		}
		request.model = resolved.model.clone();
		request.model_policy = Some(Arc::clone(&resolved.policy));
		stamp_effective_effort(&mut request.thinking, resolved.effective_effort);
		let stream = resolved.chat.turn(request, executor).await?;
		Ok(report_routed_resolution(stream, resolved.preflight_unsupported, resolved.logical_model))
	}
}

fn resolver_facet_error(error: TurnError) -> FacetError {
	if error.kind == TurnErrorKind::Unsupported {
		FacetError::Unsupported(error.unsupported)
	} else {
		FacetError::Provider(error.detail)
	}
}

#[derive(Clone)]
struct ResolvedChat {
	route_id:              Str,
	provider:              Str,
	logical_model:         Str,
	policy:                Arc<ResolvedModelPolicy>,
	effective_effort:      Option<Effort>,
	preflight_unsupported: Option<Unsupported>,
	model:                 Str,
	credential_id:         Option<Str>,
	requires_executor:     bool,
	chat:                  Arc<dyn Chat>,
}

fn effective_effort(
	explicit: Option<Effort>,
	thinking: Option<&omp_llm_types::ResolvedThinkingPolicy>,
) -> Result<(Option<Effort>, Option<Unsupported>), TurnError> {
	let Some(thinking) = thinking else {
		return Ok((None, None));
	};
	if explicit == Some(Effort::Off) {
		if thinking.suppress_when_off == Some(true) || thinking.requires_effort != Some(true) {
			return Ok((Some(Effort::Off), None));
		}
		return clamp_to_lowest_non_off(thinking, "explicit Off");
	}
	let requested = explicit.or_else(|| {
		thinking
			.default_effort
			.filter(|effort| *effort != Effort::Off)
	});
	let Some(requested) = requested else {
		if thinking.requires_effort == Some(true) && thinking.suppress_when_off != Some(true) {
			return clamp_to_lowest_non_off(thinking, "omitted effort");
		}
		return Ok((None, None));
	};

	let supported = if thinking.efforts.contains(&requested) {
		Some(requested)
	} else {
		thinking
			.efforts
			.iter()
			.copied()
			.filter(|effort| *effort <= requested)
			.max()
			.or_else(|| thinking.efforts.first().copied())
	};
	let Some(supported) = supported else {
		return Ok((None, None));
	};
	if supported == requested {
		return Ok((Some(requested), None));
	}
	Ok(clamped_effort(supported, fmts!("requested {requested:?} was not advertised")))
}

fn clamp_to_lowest_non_off(
	thinking: &omp_llm_types::ResolvedThinkingPolicy,
	reason: &'static str,
) -> Result<(Option<Effort>, Option<Unsupported>), TurnError> {
	let Some(effort) = thinking
		.efforts
		.iter()
		.copied()
		.find(|effort| *effort != Effort::Off)
	else {
		return Err(error(
			TurnErrorKind::Unsupported,
			"model requires reasoning but advertises no non-off effort",
		));
	};
	Ok(clamped_effort(effort, Str::new(reason)))
}

fn clamped_effort(effort: Effort, reason: Str) -> (Option<Effort>, Option<Unsupported>) {
	let unsupported = Unsupported::builder()
		.what(Str::new_static("thinking.effort"))
		.detail(fmts!("{reason} and was clamped to the advertised {effort:?} effort"))
		.action(UnsupportedAction::Clamped)
		.build();
	(Some(effort), Some(unsupported))
}

fn stamp_effective_effort(
	thinking: &mut Option<Feature<Reasoning>>,
	effective_effort: Option<Effort>,
) {
	let Some(effort) = effective_effort else {
		return;
	};
	if let Some(thinking) = thinking {
		thinking.value.effort = Some(effort);
	} else {
		*thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(effort).build())
				.on_unsupported(Fallback::Ignore)
				.build(),
		);
	}
}

fn report_preflight(
	stream: BoxStream<'static, TurnEvent>,
	unsupported: Option<Unsupported>,
) -> BoxStream<'static, TurnEvent> {
	let Some(unsupported) = unsupported else {
		return stream;
	};
	stream
		.map(move |mut event| {
			if let TurnEvent::Outcome(outcome) = &mut event {
				outcome.unsupported.push(unsupported.clone());
			}
			event
		})
		.boxed()
}

fn report_routed_resolution(
	stream: BoxStream<'static, TurnEvent>,
	unsupported: Option<Unsupported>,
	logical_model: Str,
) -> BoxStream<'static, TurnEvent> {
	stream
		.map(move |mut event| {
			if let TurnEvent::Outcome(outcome) = &mut event {
				outcome.model = logical_model.clone();
				if let Some(unsupported) = &unsupported {
					outcome.unsupported.push(unsupported.clone());
				}
			}
			event
		})
		.boxed()
}

#[derive(Clone, Debug)]
struct AffinityKey {
	context_id:  Str,
	route_id:    Str,
	session_key: Str,
}

/// Stateful server-side implementation of the chat-turn protocol.
///
/// A context rejects a distinct concurrent turn rather than queueing it. A
/// queue would make the queued request's `expected` revision ambiguous and
/// silently turn an optimistic-concurrency conflict into an ordered mutation.
/// A retry with the same `turn_id` instead attaches to the live broadcast.
///
/// Affinity is staged inside [`TurnGuard`] and commits with the context.
/// `previous_response_id` and its canonical item boundary are learned only
/// from an authoritative provider outcome; client properties with either name
/// are removed before routing.
#[derive(Clone)]
pub struct TurnEngine {
	contexts:  Arc<ContextStore>,
	resolver:  Arc<ChatResolver>,
	metrics:   Arc<MetricRecorder>,
	telemetry: Arc<TelemetryConfig>,
}

impl TurnEngine {
	/// Creates a turn engine over a context store and live catalog resolver.
	#[must_use]
	pub fn new(contexts: Arc<ContextStore>, resolver: Arc<ChatResolver>) -> Self {
		Self {
			contexts,
			resolver,
			metrics: Arc::new(MetricRecorder::new()),
			telemetry: Arc::new(TelemetryConfig::default()),
		}
	}

	/// Creates an engine with an explicitly supplied metric recorder.
	#[must_use]
	pub fn with_metrics(
		contexts: Arc<ContextStore>,
		resolver: Arc<ChatResolver>,
		metrics: Arc<MetricRecorder>,
	) -> Self {
		Self { contexts, resolver, metrics, telemetry: Arc::new(TelemetryConfig::default()) }
	}

	/// Creates an engine with explicit metrics and host telemetry hooks.
	#[must_use]
	pub const fn with_telemetry(
		contexts: Arc<ContextStore>,
		resolver: Arc<ChatResolver>,
		metrics: Arc<MetricRecorder>,
		telemetry: Arc<TelemetryConfig>,
	) -> Self {
		Self { contexts, resolver, metrics, telemetry }
	}

	/// Implements `rpc Turn(stream TurnFrame) returns (stream TurnEvent)`.
	///
	/// The first frame is consumed before the response is admitted, making a
	/// missing or non-`Open` first frame a transport-level protocol error.
	pub async fn turn(
		&self,
		request: Request<Streaming<pb::TurnFrame>>,
	) -> Result<Response<TurnStream>, Status> {
		let mut incoming = request.into_inner();
		let first = incoming
			.message()
			.await?
			.ok_or_else(|| Status::invalid_argument("Turn requires an Open frame"))?;
		let open = first_open(first)?;
		Ok(Response::new(self.open(open, Box::pin(incoming))))
	}

	/// Handles an arbitrary decoded frame stream with the tonic RPC protocol.
	///
	/// This is also the deterministic test seam; it does not weaken the wire
	/// contract because the same first-frame validation and driver are used.
	pub async fn turn_frames<S>(&self, frames: S) -> Result<TurnStream, Status>
	where
		S: Stream<Item = Result<pb::TurnFrame, Status>> + Send + 'static,
	{
		let mut frames = Box::pin(frames);
		let first = frames
			.next()
			.await
			.transpose()?
			.ok_or_else(|| Status::invalid_argument("Turn requires an Open frame"))?;
		let open = first_open(first)?;
		Ok(self.open(open, frames))
	}

	fn open(
		&self,
		open: pb::TurnRequest,
		incoming: BoxStream<'static, Result<pb::TurnFrame, Status>>,
	) -> TurnStream {
		let engine = self.clone();
		Box::pin(async_stream::try_stream! {
			let prepared = match PreparedTurn::try_from(open) {
				Ok(prepared) => prepared,
				Err(failure) => {
					yield pb::TurnEvent::from(TurnEvent::Error(failure));
					return;
				},
			};
			let begin = match engine
				.contexts
				.begin(prepared.turn_id.clone(), prepared.input.clone())
			{
				Ok(begin) => begin,
				Err(failure) => {
					yield pb::TurnEvent::from(TurnEvent::Error(failure.into()));
					return;
				},
			};
			match begin {
				Begin::Replay { outcome, .. } => {
					yield pb::TurnEvent::from(TurnEvent::Accepted { replay: true });
					yield pb::TurnEvent::from(TurnEvent::Outcome(outcome));
				},
				Begin::Attached(mut attachment) => {
					while let Some(event) = attachment.recv().await {
						yield pb::TurnEvent::from(event);
					}
				},
				Begin::Started(guard) => {
					let mut events = engine.drive(guard, prepared, incoming);
					while let Some(event) = events.next().await {
						yield pb::TurnEvent::from(event?);
					}
				},
			}
		})
	}

	fn drive(
		&self,
		mut guard: TurnGuard,
		prepared: PreparedTurn,
		mut incoming: BoxStream<'static, Result<pb::TurnFrame, Status>>,
	) -> BoxStream<'static, Result<TurnEvent, Status>> {
		let engine = self.clone();
		Box::pin(async_stream::try_stream! {
			let mut affinity = prepared.session_key.as_ref().map_or_else(
				SessionAffinity::default,
				|session_key| guard.affinity(session_key).clone(),
			);
			let effort = prepared
				.params
				.thinking
				.as_ref()
				.and_then(|thinking| thinking.value.effort);
			let resolved = match engine
				.resolver
				.resolve(prepared.params.model.as_str(), effort, Some(&affinity))
			{
				Ok(resolved) => resolved,
				Err(failure) => {
					guard.publish(TurnEvent::Error(failure.clone()));
					drop(guard);
					yield TurnEvent::Error(failure);
					return;
				},
			};
			if affinity.route_id.as_ref() != Some(&resolved.route_id) {
				affinity = SessionAffinity::default();
			}
			if affinity
				.previous_response_item_count
				.is_some_and(|boundary| boundary > guard.input_start())
			{
				// The retained thread branched before the stored response
				// boundary. That response no longer names this canonical prefix.
				affinity.previous_response_id = None;
				affinity.previous_response_item_count = None;
			}
			affinity.route_id = Some(resolved.route_id.clone());
			let affinity_key = prepared.session_key.as_ref().map(|session_key| AffinityKey {
				context_id: prepared.context_id.clone().unwrap_or_default(),
				session_key: session_key.clone(),
				route_id: resolved.route_id.clone(),
			});
			if affinity.prompt_cache_key.is_none() {
				affinity.prompt_cache_key = affinity_key.as_ref().map(prompt_cache_key);
			}
			let executor = prepared.executor.map(ClientExecutor::new).map(Arc::new);
			let executor_trait = executor
				.clone()
				.map(|value| value as Arc<dyn Executor>);
			if resolved.requires_executor
				&& executor.as_ref().is_none_or(|value| value.tools.is_empty())
			{
				let failure = unsupported_executor();
				guard.publish(TurnEvent::Error(failure.clone()));
				drop(guard);
				yield TurnEvent::Error(failure);
				return;
			}
			affinity.credential_id = resolved.credential_id.clone();
			if let Some(session_key) = &prepared.session_key {
				*guard.affinity(session_key) = affinity.clone();
			}
			let upstream_session = prepared
				.context_id
				.as_deref()
				.or(prepared.session_key.as_deref())
				.map(Str::new);
			let request = build_request(
				guard.thread().clone(),
				prepared.params,
				&resolved,
				&affinity,
				upstream_session.as_deref(),
			);
			let mut telemetry = ChatTelemetry::start(
				&request,
				&resolved,
				Arc::clone(&engine.metrics),
				Arc::clone(&engine.telemetry),
			);
			let mut upstream = match resolved.chat.turn(request.clone(), executor_trait).await {
				Ok(stream) => report_preflight(stream, resolved.preflight_unsupported.clone()),
				Err(failure) => {
					let failure = facet_error(failure);
					telemetry.finish_error(&failure);
					guard.publish(TurnEvent::Error(failure.clone()));
					drop(guard);
					yield TurnEvent::Error(failure);
					return;
				},
			};

			let mut projected_output = Vec::new();
			let accepted = TurnEvent::Accepted { replay: false };
			guard.publish(accepted.clone());
			yield accepted;
			let mut incoming_open = true;
			loop {
				let event = if incoming_open {
					let progress = tokio::select! {
						frame = incoming.next() => match frame {
							Some(frame) => DriveProgress::Frame(Box::new(frame)),
							None => DriveProgress::InputClosed,
						},
						event = upstream.next() => DriveProgress::Event(event),
					};
					match progress {
						DriveProgress::Frame(frame) => match *frame {
							Ok(frame) => {
								match route_frame(frame, executor.as_deref()).await {
									Ok(Some(item)) if projection_has_call(&projected_output, &item) => {
										projected_output.push(item);
									},
									Ok(_) => {},
									Err(status) => {
										let failure = status_error(status);
										telemetry.finish_error(&failure);
										guard.publish(TurnEvent::Error(failure.clone()));
										drop(guard);
										yield TurnEvent::Error(failure);
										return;
									},
								}
								continue;
							},
							Err(status) => {
								let failure = status_error(status);
								telemetry.finish_error(&failure);
								guard.publish(TurnEvent::Error(failure.clone()));
								drop(guard);
								yield TurnEvent::Error(failure);
								return;
							},
						},
						DriveProgress::InputClosed => {
							incoming_open = false;
							continue;
						},
						DriveProgress::Event(event) => event,
					}
				} else {
					upstream.next().await
				};
				let Some(event) = event else {
					let failure = error(
						TurnErrorKind::Upstream,
						"upstream ended without a terminal event",
					);
					telemetry.finish_error(&failure);
					guard.publish(TurnEvent::Error(failure.clone()));
					drop(guard);
					yield TurnEvent::Error(failure);
					return;
				};
				match event {
					TurnEvent::Accepted { .. } => {},
					TurnEvent::Outcome(mut outcome) => {
						outcome.provider = resolved.provider.clone();
						outcome.model = resolved.logical_model.clone();
						merge_projected_output(&mut outcome.output, projected_output);
						let committed_item_count = guard
							.thread()
							.items
							.len()
							.saturating_add(outcome.output.len());
						update_affinity_from_outcome(
							&mut affinity,
							&outcome,
							committed_item_count,
						);
						if let Some(session_key) = &prepared.session_key {
							*guard.affinity(session_key) = affinity;
						}
						match guard.commit(outcome) {
							Ok(committed) => {
								telemetry.finish_success(&committed);
								yield TurnEvent::Outcome(committed);
							},
							Err(failure) => {
								let failure: TurnError = failure.into();
								telemetry.finish_error(&failure);
								yield TurnEvent::Error(failure);
							},
						}
						return;
					},
					TurnEvent::Error(failure) => {
						let failure = sanitize_upstream_error(failure);
						telemetry.finish_error(&failure);
						guard.publish(TurnEvent::Error(failure.clone()));
						drop(guard);
						yield TurnEvent::Error(failure);
						return;
					},
					TurnEvent::Invoke(invocation) => {
						if let Some(tool_call) = &invocation.tool_call {
							projected_output.push(
								Item::builder()
									.seq(0)
									.kind(ItemKind::ToolCall(tool_call.clone()))
									.props(Props::default())
									.build(),
							);
						}
						let event = TurnEvent::Invoke(invocation);
						guard.publish(event.clone());
						yield event;
					},
					other => {
						guard.publish(other.clone());
						yield other;
					},
				}
			}
		})
	}
}

enum DriveProgress {
	Frame(Box<Result<pb::TurnFrame, Status>>),
	InputClosed,
	Event(Option<TurnEvent>),
}

struct PreparedTurn {
	turn_id:     Str,
	input:       BeginInput,
	context_id:  Option<Str>,
	params:      ChatParams,
	executor:    Option<SmallVec<Str, 4>>,
	session_key: Option<Str>,
}

impl TryFrom<pb::TurnRequest> for PreparedTurn {
	type Error = TurnError;

	fn try_from(mut open: pb::TurnRequest) -> Result<Self, Self::Error> {
		if open.turn_id.is_empty() {
			return Err(error(TurnErrorKind::Unsupported, "TurnRequest.turn_id is required"));
		}
		let mut params: ChatParams = open
			.params
			.take()
			.ok_or_else(|| error(TurnErrorKind::Unsupported, "TurnRequest.params is required"))?
			.try_into()
			.map_err(convert_error)?;
		if let Some(props) = open.props.take() {
			let props: Props = props.try_into().map_err(convert_error)?;
			params
				.provider_options
				.get_or_insert_with(Props::default)
				.0
				.extend(props.0);
		}
		// Never let a client splice gateway-held upstream or credential state.
		if let Some(options) = &mut params.provider_options {
			for key in [
				"previous_response_id",
				"previous_response_item_count",
				"openai/previous_response_id",
				"openai/previous_response_item_count",
				"openai-codex/turn_state",
				"omp.gateway/previous_response_id",
				"omp.gateway/previous_response_item_count",
				"omp.gateway/credential_id",
				"omp.gateway/credential_generation",
				"omp.gateway/codex_state",
			] {
				options.0.remove(key);
			}
		}
		let session_key = params.cache.as_ref().map(|cache| cache.session_key.clone());
		let executor = open.executor.map(|value| {
			let tools: SmallVec<Str, 4> = value.tools.into_iter().map(Str::from).collect();
			tools
		});
		let (input, context_id) = match open
			.input
			.ok_or_else(|| error(TurnErrorKind::Unsupported, "TurnRequest.input is required"))?
		{
			pb::turn_request::Input::Incremental(incremental) => {
				let context: ContextRef = incremental
					.context
					.ok_or_else(|| error(TurnErrorKind::Unsupported, "Incremental.context is required"))?
					.try_into()
					.map_err(convert_error)?;
				let context_id = context.context_id.clone();
				let delta: ThreadDelta = incremental
					.delta
					.unwrap_or_default()
					.try_into()
					.map_err(convert_error)?;
				(BeginInput::Incremental { context, delta }, Some(context_id))
			},
			pb::turn_request::Input::Seed(seed) => {
				let thread = seed
					.thread
					.ok_or_else(|| error(TurnErrorKind::Unsupported, "Seed.thread is required"))?
					.try_into()
					.map_err(convert_error)?;
				let context_id = (!seed.context_id.is_empty()).then(|| Str::from(seed.context_id));
				(BeginInput::Seed { context_id: context_id.clone(), thread }, context_id)
			},
		};
		Ok(Self {
			turn_id: Str::from(open.turn_id),
			input,
			context_id,
			params,
			executor,
			session_key,
		})
	}
}

fn convert_error(_failure: ConvertError) -> TurnError {
	error(TurnErrorKind::Unsupported, "invalid Turn request payload")
}

fn first_open(frame: pb::TurnFrame) -> Result<pb::TurnRequest, Status> {
	match frame.frame {
		Some(pb::turn_frame::Frame::Open(open)) => Ok(open),
		_ => Err(Status::invalid_argument("the first Turn frame must be Open")),
	}
}

fn build_request(
	thread: Thread,
	mut params: ChatParams,
	resolved: &ResolvedChat,
	affinity: &SessionAffinity,
	upstream_session: Option<&str>,
) -> ChatRequest {
	params.model = resolved.model.clone();
	params.model_policy = Some(Arc::clone(&resolved.policy));
	stamp_effective_effort(&mut params.thinking, resolved.effective_effort);
	if let (Some(cache), Some(prompt_cache_key)) = (&mut params.cache, &affinity.prompt_cache_key) {
		cache.session_key = prompt_cache_key.clone();
	}
	if let Some(session) = upstream_session {
		let meta = params.meta.get_or_insert_with(Default::default);
		if meta.session_id.is_empty() {
			meta.session_id = Str::new(session);
		}
	}
	if let (Some(previous), Some(boundary)) =
		(&affinity.previous_response_id, affinity.previous_response_item_count)
	{
		let options = params.provider_options.get_or_insert_with(Props::default);
		options.insert_ns("openai", "previous_response_id", Value::String(previous.to_string()));
		options.insert_ns(
			"openai",
			"previous_response_item_count",
			Value::from(u64::try_from(boundary).expect("usize always fits in u64")),
		);
	}
	if !affinity.codex_state.is_empty() {
		params
			.provider_options
			.get_or_insert_with(Props::default)
			.insert_ns(
				"openai-codex",
				"turn_state",
				Value::String(String::from_utf8_lossy(&affinity.codex_state).into_owned()),
			);
	}
	ChatRequest::builder()
		.model(params.model)
		.maybe_model_policy(params.model_policy)
		.thread(thread)
		.tools(params.tools)
		.maybe_service_tier(params.service_tier)
		.maybe_service_tier_by_family(params.service_tier_by_family)
		.maybe_task_budget(params.task_budget)
		.maybe_responses_include(params.responses_include)
		.maybe_tool_choice(params.tool_choice)
		.maybe_sampling(params.sampling)
		.maybe_thinking(params.thinking)
		.maybe_cache(params.cache)
		.maybe_response_format(params.response_format)
		.maybe_meta(params.meta)
		.maybe_provider_options(params.provider_options)
		.build()
}

fn route_identity(provider: &str, model: &str) -> Str {
	let mut hasher = blake3::Hasher::new();
	hasher.update(provider.as_bytes());
	hasher.update(&[0]);
	hasher.update(model.as_bytes());
	fmts!("omp_route_{}", hasher.finalize().to_hex())
}

fn prompt_cache_key(key: &AffinityKey) -> Str {
	let mut hasher = blake3::Hasher::new();
	hasher.update(key.route_id.as_bytes());
	hasher.update(&[0]);
	hasher.update(key.context_id.as_bytes());
	hasher.update(&[0]);
	hasher.update(key.session_key.as_bytes());
	fmts!("omp_cache_{}", hasher.finalize().to_hex())
}

fn update_affinity_from_outcome(
	affinity: &mut SessionAffinity,
	outcome: &ChatOutcome,
	committed_item_count: usize,
) {
	let response_id = outcome
		.props
		.get_ns("openai", "response_id")
		.and_then(Value::as_str);
	affinity.previous_response_id = response_id.map(Str::new);
	affinity.previous_response_item_count = response_id.map(|_| committed_item_count);
	if let Some(state) = outcome
		.props
		.get_ns("openai-codex", "turn_state")
		.and_then(Value::as_str)
	{
		affinity.codex_state = Bytes::copy_from_slice(state.as_bytes());
	}
}

async fn route_frame(
	frame: pb::TurnFrame,
	executor: Option<&ClientExecutor>,
) -> Result<Option<Item>, Status> {
	let executor = executor.ok_or_else(|| {
		Status::invalid_argument("invocation frame sent without a declared executor")
	})?;
	match frame.frame {
		Some(pb::turn_frame::Frame::Input(input)) => {
			let input: InvokeInput = input
				.try_into()
				.map_err(|failure: ConvertError| Status::invalid_argument(failure.to_string()))?;
			executor.input(input).await?;
			Ok(None)
		},
		Some(pb::turn_frame::Frame::Complete(complete)) => {
			let complete: InvokeComplete = complete
				.try_into()
				.map_err(|failure: ConvertError| Status::invalid_argument(failure.to_string()))?;
			let projected = complete.tool_result.clone().map(|tool_result| {
				Item::builder()
					.seq(0)
					.kind(ItemKind::ToolResult(tool_result))
					.props(Props::default())
					.build()
			});
			executor.complete(complete)?;
			Ok(projected)
		},
		Some(pb::turn_frame::Frame::Open(_)) => {
			Err(Status::invalid_argument("Open may appear only as the first Turn frame"))
		},
		None => Err(Status::invalid_argument("TurnFrame.frame is required")),
	}
}

fn merge_projected_output(output: &mut Vec<Item>, projected: Vec<Item>) {
	let mut merged = Vec::with_capacity(output.len().saturating_add(projected.len()));
	for projection in projected {
		if let Some(index) = output
			.iter()
			.position(|existing| same_projection(existing, &projection))
		{
			merged.push(output.remove(index));
		} else {
			merged.push(projection);
		}
	}
	merged.append(output);
	*output = merged;
}

fn same_projection(left: &Item, right: &Item) -> bool {
	match (&left.kind, &right.kind) {
		(ItemKind::ToolCall(left), ItemKind::ToolCall(right)) => left.id == right.id,
		(ItemKind::ToolResult(left), ItemKind::ToolResult(right)) => left.call_id == right.call_id,
		_ => false,
	}
}

fn projection_has_call(projected: &[Item], candidate: &Item) -> bool {
	let ItemKind::ToolResult(result) = &candidate.kind else {
		return true;
	};
	projected
		.iter()
		.any(|item| matches!(&item.kind, ItemKind::ToolCall(call) if call.id == result.call_id))
}

struct PendingInvocation {
	inputs:   flume::Sender<InvokeInput>,
	complete: oneshot::Sender<InvokeComplete>,
}

#[derive(Default)]
struct ExecutorState {
	pending:           FxHashMap<Str, PendingInvocation>,
	early_inputs:      FxHashMap<Str, SmallVec<InvokeInput, 4>>,
	early_completions: FxHashMap<Str, InvokeComplete>,
}

#[derive(Clone)]
struct ClientExecutor {
	tools: SmallVec<Str, 4>,
	state: Arc<Mutex<ExecutorState>>,
}

impl ClientExecutor {
	fn new(tools: SmallVec<Str, 4>) -> Self {
		Self { tools, state: Arc::new(Mutex::new(ExecutorState::default())) }
	}

	fn supports(&self, name: &str) -> bool {
		self.tools.iter().any(|pattern| {
			pattern == name
				|| pattern
					.as_str()
					.strip_suffix('*')
					.is_some_and(|prefix| name.starts_with(prefix))
		})
	}

	async fn input(&self, input: InvokeInput) -> Result<(), Status> {
		let sender = {
			let mut state = self.state.lock();
			if let Some(pending) = state.pending.get(&input.invocation_id) {
				Some(pending.inputs.clone())
			} else {
				let inputs = state
					.early_inputs
					.entry(input.invocation_id.clone())
					.or_default();
				if inputs.len() >= EARLY_FRAME_LIMIT {
					return Err(Status::resource_exhausted("too many early invocation input frames"));
				}
				inputs.push(input.clone());
				None
			}
		};
		if let Some(sender) = sender {
			sender
				.send_async(input)
				.await
				.map_err(|_| Status::failed_precondition("invocation no longer accepts input"))?;
		}
		Ok(())
	}

	fn complete(&self, complete: InvokeComplete) -> Result<(), Status> {
		let pending = {
			let mut state = self.state.lock();
			if let Some(pending) = state.pending.remove(&complete.invocation_id) {
				Some(pending)
			} else {
				if state.early_completions.len() >= EARLY_FRAME_LIMIT {
					return Err(Status::resource_exhausted("too many early invocation completions"));
				}
				state
					.early_completions
					.insert(complete.invocation_id.clone(), complete.clone());
				None
			}
		};
		if let Some(pending) = pending {
			pending
				.complete
				.send(complete)
				.map_err(|_| Status::failed_precondition("invocation already ended"))?;
		}
		Ok(())
	}
}

#[async_trait]
impl Executor for ClientExecutor {
	async fn invoke(
		&self,
		invocation: Invoke,
		inputs: flume::Sender<InvokeInput>,
	) -> InvokeComplete {
		if !self.supports(&invocation.name) {
			return failed_completion(
				&invocation.invocation_id,
				"client executor did not declare this tool",
			);
		}
		let (complete, receiver) = oneshot::channel();
		let (early_inputs, early_complete) = {
			let mut state = self.state.lock();
			let early_inputs = state
				.early_inputs
				.remove(&invocation.invocation_id)
				.unwrap_or_default();
			let early_complete = state.early_completions.remove(&invocation.invocation_id);
			state
				.pending
				.insert(invocation.invocation_id.clone(), PendingInvocation {
					inputs: inputs.clone(),
					complete,
				});
			(early_inputs, early_complete)
		};
		for input in early_inputs {
			if inputs.send_async(input).await.is_err() {
				return failed_completion(
					&invocation.invocation_id,
					"transport stopped accepting invocation input",
				);
			}
		}
		if let Some(complete) = early_complete
			&& let Some(pending) = self.state.lock().pending.remove(&invocation.invocation_id)
		{
			let _ignored = pending.complete.send(complete);
		}
		receiver.await.unwrap_or_else(|_| {
			failed_completion(&invocation.invocation_id, "client executor stream closed")
		})
	}
}

fn failed_completion(invocation_id: &str, reason: &str) -> InvokeComplete {
	InvokeComplete::builder()
		.invocation_id(Str::new(invocation_id))
		.status(
			ExecStatus::builder()
				.outcome(ExecOutcome::Failed)
				.exit_code(0)
				.signal(Str::new_static(""))
				.reason(Str::new(reason))
				.cwd(Str::new_static(""))
				.aborted(true)
				.output_location(Str::new_static(""))
				.local_execution_time_ms(0)
				.is_readonly(false)
				.command_timeout_ms(0)
				.build(),
		)
		.vendor(Bytes::new())
		.props(Props::default())
		.build()
}

fn unsupported_executor() -> TurnError {
	TurnError::builder()
		.kind(TurnErrorKind::Unsupported)
		.detail(Str::new_static("selected transport requires an in-turn executor"))
		.unsupported(vec![
			Unsupported::builder()
				.what(Str::new_static("executor"))
				.detail(Str::new_static("client did not declare an in-turn executor"))
				.action(UnsupportedAction::Dropped)
				.build(),
		])
		.retry_after_ms(0)
		.build()
}

fn facet_error(failure: FacetError) -> TurnError {
	match failure {
		FacetError::Unsupported(unsupported) => TurnError::builder()
			.kind(TurnErrorKind::Unsupported)
			.detail(Str::new_static("selected provider cannot serve this request"))
			.unsupported(unsupported)
			.retry_after_ms(0)
			.build(),
		FacetError::Provider(_) | FacetError::Transport(_) => {
			error(TurnErrorKind::Upstream, "provider inference failed")
		},
		_ => error(TurnErrorKind::Upstream, "unknown inference failure"),
	}
}

fn sanitize_upstream_error(mut failure: TurnError) -> TurnError {
	failure.detail = Str::new_static("provider inference failed");
	for diagnostic in &mut failure.diagnostics {
		diagnostic.detail = Str::new_static("provider attempt failed");
	}
	failure
}

fn error(kind: TurnErrorKind, detail: &'static str) -> TurnError {
	TurnError::builder()
		.kind(kind)
		.detail(Str::new_static(detail))
		.unsupported(Vec::new())
		.retry_after_ms(0)
		.build()
}

fn status_error(_status: Status) -> TurnError {
	TurnError::builder()
		.kind(TurnErrorKind::Unsupported)
		.detail(Str::new_static("invalid Turn frame stream"))
		.unsupported(Vec::new())
		.retry_after_ms(0)
		.build()
}

struct ChatTelemetry {
	span:            Option<span::Span>,
	metrics:         Arc<MetricRecorder>,
	config:          Arc<TelemetryConfig>,
	model:           Str,
	provider:        Str,
	conversation_id: Option<Str>,
}

impl ChatTelemetry {
	fn start(
		request: &ChatRequest,
		resolved: &ResolvedChat,
		metrics: Arc<MetricRecorder>,
		config: Arc<TelemetryConfig>,
	) -> Self {
		let stops: Vec<&str> = request
			.sampling
			.as_ref()
			.and_then(|sampling| sampling.stop.as_ref())
			.map_or_else(Vec::new, |values| values.iter().map(Str::as_str).collect());
		let tools: Vec<&str> = request
			.tools
			.iter()
			.map(|tool| tool.name.as_str())
			.collect();
		let sampling = request.sampling.as_ref();
		let provider = config
			.normalized_provider(Some(resolved.provider.as_str()))
			.unwrap_or_else(|| resolved.provider.clone());
		let conversation_id = config
			.conversation_id
			.clone()
			.or_else(|| request.meta.as_ref().map(|meta| meta.session_id.clone()));
		let agent = config.agent.as_ref().map(|agent| span::AgentIdentity {
			id:          agent.id.as_deref(),
			name:        agent.name.as_deref(),
			description: agent.description.as_deref(),
		});
		let mut chat_span = span::start_chat(
			resolved.logical_model.as_str(),
			provider.as_str(),
			1,
			span::ChatRequest {
				max_tokens: sampling.and_then(|value| value.max_output_tokens),
				temperature: sampling.and_then(|value| value.temperature),
				top_p: sampling.and_then(|value| value.top_p),
				top_k: sampling.and_then(|value| value.top_k).map(u64::from),
				frequency_penalty: sampling.and_then(|value| value.frequency_penalty),
				presence_penalty: sampling.and_then(|value| value.presence_penalty),
				stop_sequences: &stops,
				available_tools: &tools,
				capture_mode: config.capture_message_content,
				content: RequestContent::default(),
				..span::ChatRequest::default()
			},
			span::SpanContext {
				conversation_id: conversation_id.as_deref(),
				agent,
				..span::SpanContext::default()
			},
		);
		let attributes = TelemetryAttributeContext {
			kind:            TelemetrySpanKind::Chat,
			model:           Some(resolved.logical_model.as_str()),
			agent:           config.agent.as_ref(),
			conversation_id: conversation_id.as_deref(),
			step_number:     Some(0),
			tool_call_id:    None,
			tool_name:       None,
		};
		config.apply_span_attributes(&mut chat_span, &attributes);
		config.span_started(&mut TelemetryHookContext { attributes, span: &mut chat_span });
		Self {
			span: Some(chat_span),
			metrics,
			config,
			model: resolved.logical_model.clone(),
			provider,
			conversation_id,
		}
	}

	fn finish_success(&mut self, outcome: &ChatOutcome) {
		let stop = stop_name(outcome.stop);
		let usage = outcome.usage.as_ref().map(|value| span::ChatUsage {
			input: value.input_tokens,
			output: value.output_tokens,
			cache_read: Some(value.cache_read_tokens),
			cache_write: Some(value.cache_write_tokens),
			total: Some(value.input_tokens.saturating_add(value.output_tokens)),
			..span::ChatUsage::default()
		});
		let usage_snapshot = outcome.usage.as_ref().map(chat_usage_snapshot);
		let estimate = usage_snapshot.and_then(|usage| {
			let context = CostEstimatorContext {
				provider: self.provider.as_str(),
				model: outcome.model.as_str(),
				service_tier: None,
				usage,
			};
			self.config.estimate_cost(&context, |_| {
				outcome.cost.map(|cost| CostEstimate::Available {
					usd:        cost_usd(cost),
					input_usd:  None,
					output_usd: None,
				})
			})
		});
		if let (Some(chat_span), Some(usage_snapshot)) = (self.span.as_mut(), usage_snapshot) {
			self.config.chat_usage(&mut ChatUsageEvent {
				span:            chat_span,
				agent:           self.config.agent.as_ref(),
				conversation_id: self.conversation_id.as_deref(),
				step_number:     Some(0),
				model:           outcome.model.as_str(),
				provider:        Some(self.provider.as_str()),
				service_tier:    None,
				usage:           usage_snapshot,
				cost:            estimate.as_ref(),
				attributes:      None,
				headers:         None,
			});
		}
		if let Some(usage_snapshot) = usage_snapshot {
			let (cost_usd, input_usd, output_usd, unavailable) = match estimate.as_ref() {
				Some(CostEstimate::Available { usd, input_usd, output_usd }) => {
					(Some(*usd), *input_usd, *output_usd, None)
				},
				Some(CostEstimate::Unavailable { reason }) => (None, None, None, Some(reason.as_str())),
				None => (None, None, None, None),
			};
			self.config.cost_delta(&CostDelta {
				conversation_id: self.conversation_id.as_deref(),
				agent: self.config.agent.as_ref(),
				step_number: Some(0),
				provider: self.provider.as_str(),
				model: outcome.model.as_str(),
				service_tier: None,
				usage: usage_snapshot,
				cost_usd,
				input_usd,
				output_usd,
				cost_unavailable_reason: unavailable,
			});
		}
		let span_cost = match estimate.as_ref() {
			Some(CostEstimate::Available { usd, input_usd, output_usd }) => span::ChatCost {
				estimated_usd:      Some(*usd),
				input_usd:          *input_usd,
				output_usd:         *output_usd,
				unavailable_reason: None,
			},
			Some(CostEstimate::Unavailable { reason }) => span::ChatCost {
				unavailable_reason: Some(reason.as_str()),
				..span::ChatCost::default()
			},
			None => span::ChatCost::default(),
		};
		self.span_ended();
		if let Some(mut chat_span) = self.span.take() {
			span::finish_chat(&mut chat_span, span::ChatOutcome {
				model: outcome.model.as_str(),
				upstream_provider: Some(self.provider.as_str()),
				stop_reason: Some(stop),
				usage,
				cost: span_cost,
				capture_mode: self.config.capture_message_content,
				content: ResponseContent { stop_reason: Some(stop), ..ResponseContent::default() },
				..span::ChatOutcome::default()
			});
		}
		if let Some(value) = &outcome.usage {
			self.metrics.record_chat_usage(&ChatUsageMetric {
				provider:       Some(self.provider.clone()),
				model:          outcome.model.clone(),
				service_tier:   None,
				agent:          self.config.agent.as_ref().map(|agent| MetricAgent {
					id:   agent.id.clone(),
					name: self.config.normalized_agent_name(agent.name.as_deref()),
				}),
				usage:          MetricUsage {
					input:            value.input_tokens,
					output:           value.output_tokens,
					cached_input:     value.cache_read_tokens,
					cache_write:      value.cache_write_tokens,
					reasoning_output: 0,
					total:            value.input_tokens.saturating_add(value.output_tokens),
				},
				usage_accuracy: Str::new_static(match value.accuracy {
					Accuracy::Exact => "provider",
					Accuracy::Estimated => "estimated",
					_ => "mixed",
				}),
				cost_usd:       match estimate {
					Some(CostEstimate::Available { usd, .. }) => Some(usd),
					_ => None,
				},
			});
		}
	}

	fn finish_error(&mut self, failure: &TurnError) {
		self.span_ended();
		if let Some(mut chat_span) = self.span.take() {
			span::finish_chat(&mut chat_span, span::ChatOutcome {
				model: self.model.as_str(),
				upstream_provider: Some(self.provider.as_str()),
				stop_reason: Some("error"),
				error_message: Some(failure.detail.as_str()),
				capture_mode: self.config.capture_message_content,
				content: ResponseContent { stop_reason: Some("error"), ..ResponseContent::default() },
				..span::ChatOutcome::default()
			});
		}
	}

	fn span_ended(&mut self) {
		let Some(chat_span) = self.span.as_mut() else {
			return;
		};
		let attributes = TelemetryAttributeContext {
			kind:            TelemetrySpanKind::Chat,
			model:           Some(self.model.as_str()),
			agent:           self.config.agent.as_ref(),
			conversation_id: self.conversation_id.as_deref(),
			step_number:     Some(0),
			tool_call_id:    None,
			tool_name:       None,
		};
		self
			.config
			.span_ended(&mut TelemetryHookContext { attributes, span: chat_span });
	}
}

const fn chat_usage_snapshot(value: &omp_llm_types::Usage) -> ChatUsageSnapshot {
	ChatUsageSnapshot {
		input_tokens:            value.input_tokens,
		output_tokens:           value.output_tokens,
		total_tokens:            value.input_tokens.saturating_add(value.output_tokens),
		cached_input_tokens:     Some(value.cache_read_tokens),
		cache_write_tokens:      Some(value.cache_write_tokens),
		reasoning_output_tokens: Some(0),
		accuracy:                match value.accuracy {
			Accuracy::Exact => UsageAccuracy::Actual,
			Accuracy::Estimated => UsageAccuracy::Estimated,
			_ => UsageAccuracy::Mixed,
		},
	}
}

impl Drop for ChatTelemetry {
	fn drop(&mut self) {
		self.span_ended();
		if let Some(mut chat_span) = self.span.take() {
			span::finish_chat(&mut chat_span, span::ChatOutcome {
				model: self.model.as_str(),
				upstream_provider: Some(self.provider.as_str()),
				stop_reason: Some("aborted"),
				error_message: Some("turn response stream was dropped"),
				capture_mode: self.config.capture_message_content,
				content: ResponseContent { stop_reason: Some("aborted"), ..ResponseContent::default() },
				..span::ChatOutcome::default()
			});
		}
	}
}

fn cost_usd(cost: Cost) -> f64 {
	cost.nanos_usd as f64 / 1_000_000_000.0
}

const fn stop_name(stop: StopReason) -> &'static str {
	match stop {
		StopReason::EndTurn => "end_turn",
		StopReason::ToolUse => "tool_use",
		StopReason::MaxTokens => "max_tokens",
		StopReason::ContentFilter => "content_filter",
		_ => "unknown",
	}
}

#[cfg(test)]
mod tests {
	use std::{collections::BTreeMap, sync::Arc};

	use bytes::Bytes;
	use futures::{StreamExt, stream};
	use omp_core::Str;
	use omp_llm_catalog::{
		models::{
			Availability, Modality, ModelCard, ModelCatalog, ModelWire, Source, load_catalog_json,
		},
		provider::{Facet as CatalogFacet, TransportId},
		registry::{CredentialView, Registry},
	};
	use omp_llm_types::{
		ChatOutcome, ChatRequest, ContextRef, Effort, ExecOutcome, ExecStatus, Fallback, Feature,
		Invoke, InvokeChannel, InvokeChunk, InvokeComplete, InvokeInput, InvokePayload, Item,
		ItemKind, Message, Part, Props, Reasoning, ResolvedThinkingMode, ResolvedThinkingPolicy,
		Revision, Role, StopReason, Thread, ThreadDelta, ToolCall, ToolResult, TurnError,
		TurnErrorKind, TurnEvent, UnsupportedAction,
		facet::{Chat, Error as FacetError, Executor},
		ids::CallId,
	};
	use parking_lot::{Mutex, RwLock};
	use smallvec::smallvec;

	use super::{
		ChatResolver, ChatRoute, ChatRouteKey, RoutedChat, TurnEngine, effective_effort, facet_error,
		pb, route_identity, sanitize_upstream_error, status_error,
	};
	#[test]
	fn explicit_provider_route_fields_override_model_wire_fieldwise() {
		let default = ChatRouteKey {
			transport: TransportId::OpenAiChat,
			base_url:  "http://fixture.invalid/v1".into(),
		};
		let wire = ModelWire {
			transport: TransportId::OpenAiResponses,
			base_url:  Some("https://catalog.invalid/v1".into()),
		};
		let base_only = ChatRouteKey::from_model(Some(&wire), &default, true, false);
		assert_eq!(base_only.transport, TransportId::OpenAiResponses);
		assert_eq!(base_only.base_url, default.base_url);
		let transport_only = ChatRouteKey::from_model(Some(&wire), &default, false, true);
		assert_eq!(transport_only.transport, TransportId::OpenAiChat);
		assert_eq!(transport_only.base_url.as_str(), "https://catalog.invalid/v1");
		let builtin = ChatRouteKey::from_model(Some(&wire), &default, false, false);
		assert_eq!(builtin.transport, TransportId::OpenAiResponses);
		assert_eq!(builtin.base_url.as_str(), "https://catalog.invalid/v1");
	}
	use crate::context::ContextStore;

	#[derive(Clone, Copy)]
	struct Available;

	impl CredentialView for Available {
		fn availability(&self, _provider: &str) -> Availability {
			Availability::Available
		}
	}

	#[derive(Clone)]
	enum Behavior {
		Events(Vec<TurnEvent>),
		Pending,
		Interactive(Arc<InteractiveState>),
	}

	#[derive(Default)]
	struct InteractiveState {
		call_id:  Mutex<Option<CallId>>,
		input:    Mutex<Option<InvokeInput>>,
		complete: Mutex<Option<InvokeComplete>>,
	}

	struct MockChat {
		behavior: Behavior,
		seen:     Arc<Mutex<Vec<ChatRequest>>>,
	}

	#[async_trait::async_trait]
	impl Chat for MockChat {
		async fn turn(
			&self,
			request: ChatRequest,
			executor: Option<Arc<dyn Executor>>,
		) -> Result<futures::stream::BoxStream<'static, TurnEvent>, FacetError> {
			self.seen.lock().push(request);
			match &self.behavior {
				Behavior::Events(events) => Ok(Box::pin(stream::iter(events.clone()))),
				Behavior::Pending => Ok(Box::pin(stream::pending())),
				Behavior::Interactive(state) => {
					let state = Arc::clone(state);
					let executor = executor.expect("interactive mock needs executor");
					Ok(Box::pin(async_stream::stream! {
						let call_id = CallId::new();
						*state.call_id.lock() = Some(call_id);
						let invocation = Invoke::builder()
							.invocation_id(Str::new_static("invoke-1"))
							.name(Str::new_static("cursor/shell"))
							.tool_call(
								ToolCall::builder()
									.id(call_id)
									.name(Str::new_static("cursor/shell"))
									.args_json(Bytes::from_static(br#"{"command":"echo hello"}"#))
									.thought_signature(Bytes::new())
									.build(),
							)
							.vendor(Bytes::new())
							.timeout_ms(5_000)
							.props(Props::default())
							.build();
						yield TurnEvent::Invoke(invocation.clone());
						let (inputs, received) = flume::bounded(4);
						let completion = executor.invoke(invocation, inputs);
						tokio::pin!(completion);
						let input = tokio::select! {
							input = received.recv_async() => input.expect("forwarded input"),
							_ = &mut completion => panic!("completion arrived before input"),
						};
						*state.input.lock() = Some(input);
						yield TurnEvent::PartDelta { index: 0, chunk: Bytes::from_static(b"input-ok") };
						let complete = completion.await;
						*state.complete.lock() = Some(complete);
						yield successful_outcome();
					}))
				},
			}
		}
	}

	fn model_card() -> ModelCard {
		ModelCard::builder()
			.id(Str::new_static("test/model"))
			.provider(Str::new_static("test"))
			.model(Str::new_static("model"))
			.name(Str::new_static("Model"))
			.family(Str::new_static("test"))
			.facets(smallvec![CatalogFacet::Chat])
			.inputs(smallvec![Modality::Text])
			.outputs(smallvec![Modality::Text])
			.reasoning(false)
			.efforts(smallvec![])
			.context_window(4_096)
			.max_output_tokens(1_024)
			.pricing(smallvec![])
			.availability(Availability::Available)
			.source(Source::Configured)
			.blocked_until_ms(0)
			.deprecated(false)
			.updated_at_ms(0)
			.props(Props::default())
			.effort_routing(BTreeMap::new())
			.build()
	}

	fn engine(
		contexts: Arc<ContextStore>,
		behavior: Behavior,
		requires_executor: bool,
	) -> (TurnEngine, Arc<Mutex<Vec<ChatRequest>>>, Arc<ChatResolver>) {
		let catalog = ModelCatalog::new(vec![model_card()]);
		let registry = Arc::new(RwLock::new(Registry::new(&catalog, Arc::new(Available))));
		let resolver = Arc::new(ChatResolver::new(registry));
		let seen = Arc::new(Mutex::new(Vec::new()));
		resolver.register(ChatRoute {
			provider: Str::new_static("test"),
			credential_id: Str::new_static("cred-a"),
			requires_executor,
			chat: Arc::new(MockChat { behavior, seen: Arc::clone(&seen) }),
		});
		(TurnEngine::new(contexts, Arc::clone(&resolver)), seen, resolver)
	}

	#[test]
	fn resolver_uses_catalog_wire_model_routes() {
		let mut card = model_card();
		card.id = Str::new_static("test/model-1m");
		card.model = Str::new_static("model-1m");
		card
			.effort_routing
			.insert(Effort::Off, Str::new_static("model"));
		card
			.effort_routing
			.insert(Effort::High, Str::new_static("model-high"));
		let catalog = ModelCatalog::new(vec![card]);
		let registry = Arc::new(RwLock::new(Registry::new(&catalog, Arc::new(Available))));
		let resolver = ChatResolver::new(registry);
		resolver.register(ChatRoute {
			provider:          Str::new_static("test"),
			credential_id:     Str::new_static("cred-a"),
			requires_executor: false,
			chat:              Arc::new(MockChat {
				behavior: Behavior::Events(Vec::new()),
				seen:     Arc::new(Mutex::new(Vec::new())),
			}),
		});

		assert_eq!(
			resolver
				.resolve("test/model-1m", None, None)
				.expect("default route")
				.model,
			"model-1m"
		);
		assert_eq!(
			resolver
				.resolve("test/model-1m", Some(Effort::High), None)
				.expect("high-effort route")
				.model,
			"model-high"
		);
	}

	#[test]
	fn policy_cache_reuses_stable_arcs_and_refreshes_same_timestamp_behavior() {
		let catalog = ModelCatalog::new(vec![model_card()]);
		let registry = Arc::new(RwLock::new(Registry::new(&catalog, Arc::new(Available))));
		let resolver = ChatResolver::new(registry);
		let mut card = model_card();
		let first = resolver.policy_for(&card);
		let stable = resolver.policy_for(&card);
		assert!(Arc::ptr_eq(&first, &stable));

		card.behavior.request_model_id = Some(Str::new_static("changed-wire"));
		let refreshed = resolver.policy_for(&card);
		assert!(!Arc::ptr_eq(&first, &refreshed));
		assert_eq!(refreshed.request_model_id.as_deref(), Some("changed-wire"));
	}

	#[test]
	fn copilot_request_model_id_preserves_canonical_affinity_identity() {
		let catalog = load_catalog_json(
			br#"{"github-copilot":{"claude-opus-4.6-1m":{"requestModelId":"claude-opus-4.6"}}}"#,
		)
		.expect("catalog");
		let registry = Arc::new(RwLock::new(Registry::new(&catalog, Arc::new(Available))));
		let resolver = ChatResolver::new(registry);
		resolver.register(ChatRoute {
			provider:          Str::new_static("github-copilot"),
			credential_id:     Str::new_static("cred-a"),
			requires_executor: false,
			chat:              Arc::new(MockChat {
				behavior: Behavior::Events(Vec::new()),
				seen:     Arc::new(Mutex::new(Vec::new())),
			}),
		});

		let resolved = resolver
			.resolve("github-copilot/claude-opus-4.6-1m", None, None)
			.expect("resolves");
		assert_eq!(resolved.model, "claude-opus-4.6");
		assert_eq!(resolved.logical_model, "claude-opus-4.6-1m");
		assert_eq!(resolved.route_id, route_identity("github-copilot", "claude-opus-4.6-1m"),);
	}

	#[test]
	fn sparse_effort_ladders_clamp_without_folding_xhigh_or_max() {
		let mut thinking = ResolvedThinkingPolicy {
			mode:              ResolvedThinkingMode::Effort,
			efforts:           smallvec![Effort::Low, Effort::XHigh],
			default_effort:    None,
			effort_map:        BTreeMap::new(),
			effort_routing:    BTreeMap::new(),
			effort_budgets:    BTreeMap::new(),
			supports_display:  None,
			suppress_when_off: None,
			requires_effort:   Some(true),
		};
		let (effort, report) =
			effective_effort(Some(Effort::Max), Some(&thinking)).expect("effort resolves");
		assert_eq!(effort, Some(Effort::XHigh));
		assert_eq!(report.map(|value| value.action), Some(UnsupportedAction::Clamped));
		let (effort, report) =
			effective_effort(Some(Effort::XHigh), Some(&thinking)).expect("effort resolves");
		assert_eq!(effort, Some(Effort::XHigh));
		assert!(report.is_none());

		thinking.requires_effort = None;
		let (effort, report) =
			effective_effort(Some(Effort::Off), Some(&thinking)).expect("effort resolves");
		assert_eq!(effort, Some(Effort::Off));
		assert!(report.is_none(), "ordinary explicit Off must not re-enable reasoning");

		thinking.requires_effort = Some(true);
		let (effort, report) = effective_effort(None, Some(&thinking)).expect("effort resolves");
		assert_eq!(effort, Some(Effort::Low));
		assert_eq!(report.map(|value| value.action), Some(UnsupportedAction::Clamped));
		thinking.suppress_when_off = Some(true);
		let (effort, report) =
			effective_effort(Some(Effort::Off), Some(&thinking)).expect("effort resolves");
		assert_eq!(effort, Some(Effort::Off));
		assert!(report.is_none());
		thinking.suppress_when_off = None;
		thinking.efforts = smallvec![Effort::Off];
		let failure =
			effective_effort(None, Some(&thinking)).expect_err("malformed mandatory policy");
		assert_eq!(failure.kind, TurnErrorKind::Unsupported);
		assert_eq!(failure.detail, "model requires reasoning but advertises no non-off effort",);
	}

	#[tokio::test]
	async fn resolved_policy_defaults_routes_and_clamps_required_off() {
		let catalog = load_catalog_json(
			br#"{"kimi":{"k2":{"reasoning":true,"requestModelId":"kimi-wire","thinking":{"mode":"effort","efforts":["off","low","high"],"defaultLevel":"high","effortRouting":{"off":"kimi-off","low":"kimi-low","high":"kimi-high"},"requiresEffort":true}}}}"#,
		)
		.expect("catalog");
		let registry = Arc::new(RwLock::new(Registry::new(&catalog, Arc::new(Available))));
		let resolver = Arc::new(ChatResolver::new(registry));
		let seen = Arc::new(Mutex::new(Vec::new()));
		resolver.register(ChatRoute {
			provider:          Str::new_static("kimi"),
			credential_id:     Str::new_static("cred-a"),
			requires_executor: false,
			chat:              Arc::new(MockChat {
				behavior: Behavior::Events(vec![successful_outcome()]),
				seen:     Arc::clone(&seen),
			}),
		});

		let default = resolver
			.resolve("kimi/k2", None, None)
			.expect("default resolves");
		assert_eq!(default.effective_effort, Some(Effort::High));
		assert_eq!(default.model, "kimi-high");
		let default_again = resolver
			.resolve("kimi/k2", None, None)
			.expect("default resolves");
		assert!(
			Arc::ptr_eq(&default.policy, &default_again.policy),
			"stable catalog resolutions must reuse the cached policy Arc",
		);

		let mut request = ChatRequest::builder()
			.model(Str::new_static("kimi/k2"))
			.thread(Thread::default())
			.tools(Vec::new())
			.thinking(
				Feature::builder()
					.value(Reasoning::builder().effort(Effort::Off).build())
					.on_unsupported(Fallback::Ignore)
					.build(),
			)
			.build();
		request.model_policy = None;
		let events: Vec<_> = RoutedChat::new(resolver)
			.turn(request, None)
			.await
			.expect("turn")
			.collect()
			.await;
		let sent = seen.lock().pop().expect("request captured");
		assert_eq!(sent.model, "kimi-low");
		assert_eq!(
			sent
				.thinking
				.as_ref()
				.and_then(|thinking| thinking.value.effort),
			Some(Effort::Low),
		);
		assert_eq!(
			sent
				.model_policy
				.as_ref()
				.and_then(|policy| policy.request_model_id.as_deref()),
			Some("kimi-wire"),
		);
		let TurnEvent::Outcome(outcome) = &events[0] else {
			panic!("outcome")
		};
		assert_eq!(outcome.model, "k2");
		assert!(outcome.unsupported.iter().any(|unsupported| {
			unsupported.what == "thinking.effort" && unsupported.action == UnsupportedAction::Clamped
		}));
	}

	fn item(role: Role, text: &'static str) -> Item {
		Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(role)
					.parts(vec![Part::Text(Str::new_static(text))])
					.build(),
			))
			.props(Props::default())
			.build()
	}

	fn successful_outcome() -> TurnEvent {
		TurnEvent::Outcome(
			ChatOutcome::builder()
				.output(vec![item(Role::Assistant, "answer")])
				.stop(StopReason::EndTurn)
				.unsupported(Vec::new())
				.provider(Str::new_static("test"))
				.model(Str::new_static("model"))
				.props(Props::default())
				.build(),
		)
	}

	fn successful_responses_outcome() -> TurnEvent {
		let mut props = Props::default();
		props.insert_ns("openai", "response_id", serde_json::json!("resp_first"));
		TurnEvent::Outcome(
			ChatOutcome::builder()
				.output(vec![item(Role::Assistant, "answer")])
				.stop(StopReason::EndTurn)
				.unsupported(Vec::new())
				.provider(Str::new_static("openai"))
				.model(Str::new_static("gpt-5"))
				.props(props)
				.build(),
		)
	}

	fn failure() -> TurnEvent {
		TurnEvent::Error(
			TurnError::builder()
				.kind(TurnErrorKind::Upstream)
				.detail(Str::new_static("mock failure"))
				.unsupported(Vec::new())
				.retry_after_ms(0)
				.build(),
		)
	}
	fn open(
		turn_id: &'static str,
		context_id: &'static str,
		expected: Revision,
		append: Vec<Item>,
		session_key: Option<&'static str>,
		executor: bool,
	) -> pb::TurnFrame {
		pb::TurnFrame {
			frame: Some(pb::turn_frame::Frame::Open(pb::TurnRequest {
				turn_id:  turn_id.into(),
				input:    Some(pb::turn_request::Input::Incremental(pb::Incremental {
					context: Some(
						ContextRef::builder()
							.context_id(Str::new_static(context_id))
							.expected(expected)
							.build()
							.into(),
					),
					delta:   Some(ThreadDelta::builder().append(append).build().into()),
				})),
				params:   Some(pb::ChatParams {
					model: "test/model".into(),
					cache: session_key.map(|key| pb::CacheHint {
						session_key: key.into(),
						retention: pb::cache_hint::Retention::Short as i32,
						..pb::CacheHint::default()
					}),
					..pb::ChatParams::default()
				}),
				executor: executor.then(|| pb::Executor { tools: vec!["cursor/*".into()] }),
				props:    None,
			})),
		}
	}

	async fn next_event(stream: &mut super::TurnStream) -> TurnEvent {
		stream
			.next()
			.await
			.expect("event")
			.expect("status")
			.try_into()
			.expect("canonical event")
	}

	fn seed(contexts: &ContextStore, context_id: &'static str) -> Revision {
		contexts.seed(context_id, Thread::default()).expect("seed")
	}

	fn context_ref(context_id: &'static str, expected: Revision) -> ContextRef {
		ContextRef::builder()
			.context_id(Str::new_static(context_id))
			.expected(expected)
			.build()
	}

	#[tokio::test]
	async fn open_delta_invocation_and_outcome_commit_atomically() {
		let contexts = Arc::new(ContextStore::default());
		let revision = seed(&contexts, "ctx-success");
		let interactive = Arc::new(InteractiveState::default());
		let (engine, ..) =
			engine(Arc::clone(&contexts), Behavior::Interactive(Arc::clone(&interactive)), true);
		let (frames, incoming) = flume::bounded(8);
		frames
			.send_async(Ok(open(
				"turn-success",
				"ctx-success",
				revision,
				vec![item(Role::User, "question")],
				None,
				true,
			)))
			.await
			.expect("open");
		let mut events = engine
			.turn_frames(incoming.into_stream())
			.await
			.expect("turn");
		assert!(matches!(next_event(&mut events).await, TurnEvent::Accepted { replay: false }));
		assert!(matches!(next_event(&mut events).await, TurnEvent::Invoke(_)));
		let input = InvokeInput::builder()
			.invocation_id(Str::new_static("invoke-1"))
			.payload(InvokePayload::Chunk(
				InvokeChunk::builder()
					.channel(InvokeChannel::Stdout)
					.data(Bytes::from_static(b"hello"))
					.build(),
			))
			.build();
		frames
			.send_async(Ok(pb::TurnFrame {
				frame: Some(pb::turn_frame::Frame::Input(input.clone().into())),
			}))
			.await
			.expect("input");
		assert!(matches!(next_event(&mut events).await, TurnEvent::PartDelta { .. }));
		let complete = InvokeComplete::builder()
			.invocation_id(Str::new_static("invoke-1"))
			.tool_result(
				ToolResult::builder()
					.call_id(
						interactive
							.call_id
							.lock()
							.as_ref()
							.copied()
							.expect("call id"),
					)
					.name(Str::new_static("shell"))
					.parts(vec![Part::Text(Str::new_static("done"))])
					.is_error(false)
					.build(),
			)
			.status(
				ExecStatus::builder()
					.outcome(ExecOutcome::Exited)
					.exit_code(0)
					.signal(Str::new_static(""))
					.reason(Str::new_static(""))
					.cwd(Str::new_static("/tmp"))
					.aborted(false)
					.output_location(Str::new_static(""))
					.local_execution_time_ms(1)
					.is_readonly(false)
					.command_timeout_ms(0)
					.build(),
			)
			.vendor(Bytes::new())
			.props(Props::default())
			.build();
		frames
			.send_async(Ok(pb::TurnFrame {
				frame: Some(pb::turn_frame::Frame::Complete(complete.clone().into())),
			}))
			.await
			.expect("complete");
		let TurnEvent::Outcome(outcome) = next_event(&mut events).await else {
			panic!("terminal outcome");
		};
		assert_eq!(outcome.revision.as_ref().expect("revision").head, 4);
		assert_eq!(interactive.input.lock().as_ref(), Some(&input));
		assert_eq!(interactive.complete.lock().as_ref(), Some(&complete));
		let snapshot = contexts
			.snapshot(&context_ref("ctx-success", outcome.revision.expect("revision")))
			.expect("committed snapshot");
		assert_eq!(snapshot.items.len(), 4);
	}

	#[tokio::test]
	async fn midstream_error_rolls_back() {
		let contexts = Arc::new(ContextStore::default());
		let revision = seed(&contexts, "ctx-error");
		let (upstream_engine, ..) =
			engine(Arc::clone(&contexts), Behavior::Events(vec![failure()]), false);
		let mut events = upstream_engine
			.turn_frames(stream::iter(vec![Ok(open(
				"turn-error",
				"ctx-error",
				revision.clone(),
				vec![item(Role::User, "not committed")],
				None,
				false,
			))]))
			.await
			.expect("turn");
		assert!(matches!(next_event(&mut events).await, TurnEvent::Accepted { .. }));
		assert!(matches!(next_event(&mut events).await, TurnEvent::Error(_)));
		assert_eq!(
			contexts
				.snapshot(&context_ref("ctx-error", revision))
				.expect("unchanged")
				.items,
			[] as [omp_llm_types::Item; 0]
		);

		let frame_revision = seed(&contexts, "ctx-frame-error");
		let (frame_engine, ..) = engine(Arc::clone(&contexts), Behavior::Pending, false);
		let (frames, incoming) = flume::bounded(4);
		frames
			.send_async(Ok(open(
				"turn-frame-error",
				"ctx-frame-error",
				frame_revision.clone(),
				vec![item(Role::User, "also not committed")],
				None,
				false,
			)))
			.await
			.expect("open");
		let mut frame_events = frame_engine
			.turn_frames(incoming.into_stream())
			.await
			.expect("turn");
		assert!(matches!(next_event(&mut frame_events).await, TurnEvent::Accepted { .. }));
		frames
			.send_async(Err(tonic::Status::invalid_argument("malformed invocation frame")))
			.await
			.expect("terminal input error");
		let TurnEvent::Error(failure) = next_event(&mut frame_events).await else {
			panic!("in-band frame error");
		};
		assert_eq!(failure.kind, TurnErrorKind::Unsupported);
		assert_eq!(
			contexts
				.snapshot(&context_ref("ctx-frame-error", frame_revision))
				.expect("unchanged after frame error")
				.items,
			[] as [omp_llm_types::Item; 0]
		);
	}

	#[tokio::test]
	async fn cancellation_drops_guard_and_rolls_back() {
		let contexts = Arc::new(ContextStore::default());
		let revision = seed(&contexts, "ctx-cancel");
		let (engine, ..) = engine(Arc::clone(&contexts), Behavior::Pending, false);
		let mut events = engine
			.turn_frames(stream::iter(vec![Ok(open(
				"turn-cancel",
				"ctx-cancel",
				revision.clone(),
				vec![item(Role::User, "not committed")],
				None,
				false,
			))]))
			.await
			.expect("turn");
		assert!(matches!(next_event(&mut events).await, TurnEvent::Accepted { .. }));
		drop(events);
		assert_eq!(
			contexts
				.snapshot(&context_ref("ctx-cancel", revision))
				.expect("unchanged")
				.items,
			[] as [omp_llm_types::Item; 0]
		);
	}

	#[tokio::test]
	async fn first_frame_must_be_open() {
		let contexts = Arc::new(ContextStore::default());
		let (engine, ..) = engine(contexts, Behavior::Pending, false);
		let result = engine
			.turn_frames(stream::iter(vec![Ok(pb::TurnFrame {
				frame: Some(pb::turn_frame::Frame::Input(pb::InvokeInput::default())),
			})]))
			.await;
		let Err(status) = result else {
			panic!("protocol error");
		};
		assert_eq!(status.code(), tonic::Code::InvalidArgument);
	}

	#[tokio::test]
	async fn required_executor_fails_before_chat_admission() {
		let contexts = Arc::new(ContextStore::default());
		let revision = seed(&contexts, "ctx-executor");
		let (engine, seen, _) = engine(Arc::clone(&contexts), Behavior::Pending, true);
		let mut events = engine
			.turn_frames(stream::iter(vec![Ok(open(
				"turn-executor",
				"ctx-executor",
				revision,
				Vec::new(),
				None,
				false,
			))]))
			.await
			.expect("turn");
		let TurnEvent::Error(failure) = next_event(&mut events).await else {
			panic!("unsupported error");
		};
		assert_eq!(failure.kind, TurnErrorKind::Unsupported);
		assert!(seen.lock().is_empty());
	}

	#[tokio::test]
	async fn concurrent_distinct_turn_is_rejected_not_queued() {
		let contexts = Arc::new(ContextStore::default());
		let revision = seed(&contexts, "ctx-busy");
		let (engine, ..) = engine(Arc::clone(&contexts), Behavior::Pending, false);
		let mut first = engine
			.turn_frames(stream::iter(vec![Ok(open(
				"turn-one",
				"ctx-busy",
				revision.clone(),
				Vec::new(),
				None,
				false,
			))]))
			.await
			.expect("first");
		assert!(matches!(next_event(&mut first).await, TurnEvent::Accepted { .. }));
		let mut second = engine
			.turn_frames(stream::iter(vec![Ok(open(
				"turn-two",
				"ctx-busy",
				revision,
				Vec::new(),
				None,
				false,
			))]))
			.await
			.expect("second stream");
		let TurnEvent::Error(failure) = next_event(&mut second).await else {
			panic!("busy error");
		};
		assert_eq!(failure.kind, TurnErrorKind::Overloaded);
		drop(first);
	}

	#[tokio::test]
	async fn one_session_reuses_credential_and_prompt_cache_key() {
		let contexts = Arc::new(ContextStore::default());
		let revision = seed(&contexts, "ctx-affinity");
		let (engine, seen, resolver) =
			engine(Arc::clone(&contexts), Behavior::Events(vec![successful_outcome()]), false);
		let alternate_seen = Arc::new(Mutex::new(Vec::new()));
		resolver.register(ChatRoute {
			provider:          Str::new_static("test"),
			credential_id:     Str::new_static("cred-b"),
			requires_executor: false,
			chat:              Arc::new(MockChat {
				behavior: Behavior::Events(vec![successful_outcome()]),
				seen:     Arc::clone(&alternate_seen),
			}),
		});
		let mut first = engine
			.turn_frames(stream::iter(vec![Ok(open(
				"turn-affinity-one",
				"ctx-affinity",
				revision,
				Vec::new(),
				Some("session"),
				false,
			))]))
			.await
			.expect("first");
		assert!(matches!(next_event(&mut first).await, TurnEvent::Accepted { .. }));
		let TurnEvent::Outcome(first_outcome) = next_event(&mut first).await else {
			panic!("first outcome");
		};
		let mut second = engine
			.turn_frames(stream::iter(vec![Ok(open(
				"turn-affinity-two",
				"ctx-affinity",
				first_outcome.revision.expect("revision"),
				Vec::new(),
				Some("session"),
				false,
			))]))
			.await
			.expect("second");
		assert!(matches!(next_event(&mut second).await, TurnEvent::Accepted { .. }));
		assert!(matches!(next_event(&mut second).await, TurnEvent::Outcome(_)));
		let seen = seen.lock();
		assert_eq!(seen.len(), 2);
		assert!(alternate_seen.lock().is_empty());
		assert_eq!(
			seen[0].cache.as_ref().map(|cache| &cache.session_key),
			seen[1].cache.as_ref().map(|cache| &cache.session_key),
		);
		assert_ne!(seen[0].cache.as_ref().expect("cache").session_key, "session");
	}

	#[tokio::test]
	async fn responses_affinity_commits_id_and_item_boundary_together() {
		let contexts = Arc::new(ContextStore::default());
		let revision = seed(&contexts, "ctx-responses-chain");
		let (engine, seen, _) = engine(
			Arc::clone(&contexts),
			Behavior::Events(vec![successful_responses_outcome()]),
			false,
		);
		let mut first = engine
			.turn_frames(stream::iter(vec![Ok(open(
				"turn-responses-one",
				"ctx-responses-chain",
				revision,
				vec![item(Role::User, "first")],
				Some("responses-session"),
				false,
			))]))
			.await
			.expect("first");
		assert!(matches!(next_event(&mut first).await, TurnEvent::Accepted { .. }));
		let TurnEvent::Outcome(first_outcome) = next_event(&mut first).await else {
			panic!("first outcome");
		};
		let mut second = engine
			.turn_frames(stream::iter(vec![Ok(open(
				"turn-responses-two",
				"ctx-responses-chain",
				first_outcome.revision.expect("revision"),
				vec![item(Role::User, "follow up")],
				Some("responses-session"),
				false,
			))]))
			.await
			.expect("second");
		assert!(matches!(next_event(&mut second).await, TurnEvent::Accepted { .. }));
		assert!(matches!(next_event(&mut second).await, TurnEvent::Outcome(_)));

		let seen = seen.lock();
		assert_eq!(seen.len(), 2);
		assert_eq!(seen[1].thread.items.len(), 3, "gateway retains the full canonical thread");
		let options = seen[1]
			.provider_options
			.as_ref()
			.expect("continuation options");
		assert_eq!(
			options.get_ns("openai", "previous_response_id"),
			Some(&serde_json::json!("resp_first")),
		);
		assert_eq!(
			options.get_ns("openai", "previous_response_item_count"),
			Some(&serde_json::json!(2)),
		);
	}
	#[test]
	fn provider_and_frame_failures_are_redacted_before_rpc_and_telemetry() {
		const CANARY: &str = "canary-api-key-cookie-and-refresh-token";
		let provider = facet_error(FacetError::Provider(CANARY.into()));
		let transport = facet_error(FacetError::Transport(CANARY.into()));
		let streamed = sanitize_upstream_error(
			TurnError::builder()
				.kind(TurnErrorKind::Upstream)
				.detail(Str::new_static(CANARY))
				.unsupported(Vec::new())
				.retry_after_ms(19)
				.build(),
		);
		let frame = status_error(tonic::Status::invalid_argument(CANARY));

		for failure in [&provider, &transport, &streamed, &frame] {
			assert!(!failure.detail.contains(CANARY));
			assert!(!format!("{failure:?}").contains(CANARY));
		}
		assert_eq!(streamed.kind, TurnErrorKind::Upstream);
		assert_eq!(streamed.retry_after_ms, 19);
		assert_eq!(frame.kind, TurnErrorKind::Unsupported);
	}
}
