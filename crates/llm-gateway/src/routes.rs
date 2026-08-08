//! Catalog-driven registration of production chat routes.

use std::{collections::BTreeMap, fmt, sync::Arc};

use bytes::Bytes;
use http::{Request, Response};
use hyper::body::Body as HttpBody;
use omp_llm_catalog::{
	identity::{DIALECT_ENV, DialectSelection},
	provider::{Facet, ProviderEntry, TransportId},
};
use omp_llm_egress::client::Body;
use omp_llm_tower::{
	dialect::{OwnedDialectChat, OwnedDialectConfig},
	provider::{ProviderAttempt, ProviderBuildError, ProviderRoute, ServiceChat},
	stack::builder::{RouteDependencies, RouteStackBuilder, RouteStackConfig},
};
use omp_llm_types::facet::Chat;
use tower::Service;

use crate::turn::{ChatResolver, ChatRouteKey};
/// Injected implementations for transports that are not ordinary HTTP APIs.
///
/// A populated field is used for every chat-capable catalog row with the
/// corresponding transport. Missing implementations are errors only when the
/// catalog actually requires that transport.
#[derive(Clone, Default)]
pub struct SpecializedChats {
	/// Provider-keyed specialized implementations for rows that cannot share
	/// transport-global endpoint or credential state.
	pub by_provider:         BTreeMap<omp_core::SmolStr, Arc<dyn Chat>>,
	/// Cursor's Connect/gRPC agent implementation.
	pub cursor:              Option<Arc<dyn Chat>>,
	/// Devin's Connect server-streaming implementation.
	pub devin:               Option<Arc<dyn Chat>>,
	/// GitLab Duo Workflow WebSocket agent implementation.
	pub gitlab_duo_workflow: Option<Arc<dyn Chat>>,
	/// In-process embedded inference implementation.
	pub embedded:            Option<Arc<dyn Chat>>,
	/// OMP federation implementation.
	pub omp:                 Option<Arc<dyn Chat>>,
}

/// Observable result of one catalog registration pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouteRegistration {
	/// Number of distinct provider/transport/base stacks registered.
	pub registered: usize,
}

/// Failure to assemble the catalog's production chat routes.
#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum RouteRegistrationError {
	/// A chat-capable specialized catalog row had no injected implementation.
	#[error("chat provider {provider} requires an injected {transport:?} transport")]
	MissingSpecialized {
		/// Provider catalog id that could not be registered.
		provider:  omp_core::SmolStr,
		/// Specialized transport required by the catalog row.
		transport: TransportId,
	},
	/// An HTTP catalog row could not construct its concrete provider adapter.
	#[error("chat provider {provider} could not be assembled: {source}")]
	Provider {
		/// Provider catalog id that could not be registered.
		provider: omp_core::SmolStr,
		/// Concrete adapter construction failure.
		source:   ProviderBuildError,
	},
	/// The supplied catalog contained no chat route that could be registered.
	#[error("provider catalog contains no usable chat routes")]
	NoUsableRoutes,
}

/// Registers every chat-capable catalog row in a production resolver.
///
/// HTTP rows are assembled here as `ProviderAttempt -> RouteStackBuilder ->
/// ServiceChat`; middleware state is created once per distinct route tuple.
/// Cursor, Devin, GitLab Duo Workflow, embedded, and OMP tuples use injected
/// native [`Chat`] implementations and never pass through the HTTP adapter.
/// Embedded chat is wrapped once with the model-aware owned dialect; Apple also
/// receives projected history and the tool prompt as one final user prompt.
/// HTTP rows default to model-family `Auto` and capture `OMP_DIALECT` once.
///
/// The dependency, policy, and endpoint factories are explicit because their
/// state is deployment-owned and may differ by provider. Every specialized
/// requirement and HTTP adapter is validated and assembled before the resolver
/// is mutated, so a failed pass cannot leave a partial or stale route set.
pub fn register_production_routes<'a, I, S, B, D, C, R>(
	resolver: &ChatResolver,
	providers: I,
	egress: S,
	mut dependencies: D,
	mut config: C,
	mut route: R,
	specialized: SpecializedChats,
) -> Result<RouteRegistration, RouteRegistrationError>
where
	I: IntoIterator<Item = &'a ProviderEntry>,
	S: Service<Request<Body>, Response = Response<B>> + Clone + Send + 'static,
	S::Future: Send + 'static,
	S::Error: fmt::Display + Send + 'static,
	B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
	B::Error: fmt::Display + Send + 'static,
	D: FnMut(&ProviderEntry) -> RouteDependencies,
	C: FnMut(&ProviderEntry) -> RouteStackConfig,
	R: FnMut(&ProviderEntry) -> ProviderRoute,
{
	let providers: Vec<_> = providers
		.into_iter()
		.filter(|provider| provider.facets.contains(&Facet::Chat))
		.cloned()
		.collect();
	if providers.is_empty() {
		return Err(RouteRegistrationError::NoUsableRoutes);
	}

	let omp_dialect = std::env::var(DIALECT_ENV).ok().map(omp_core::SmolStr::from);
	let mut assembled = Vec::with_capacity(providers.len());
	for provider in providers {
		let default =
			ChatRouteKey { transport: provider.transport, base_url: provider.base_url.clone() };
		for wire in resolver.wire_routes(
			&provider.id,
			&default,
			provider.base_url_overridden,
			provider.transport_overridden,
		) {
			let mut effective = provider.clone();
			effective.transport = wire.transport;
			effective.base_url.clone_from(&wire.base_url);
			let (chat, requires_executor): (Arc<dyn Chat>, bool) =
				if is_specialized(effective.transport) {
					let mut chat = specialized_chat(&specialized, &effective).ok_or_else(|| {
						RouteRegistrationError::MissingSpecialized {
							provider:  effective.id.clone(),
							transport: effective.transport,
						}
					})?;
					if effective.transport == TransportId::Embedded {
						let dialect =
							OwnedDialectConfig::new(DialectSelection::Auto, effective.compat.clone())
								.with_override(omp_dialect.clone());
						let wrapped = if effective.id == "apple-intelligence" {
							OwnedDialectChat::latest_user(chat, dialect)
						} else {
							OwnedDialectChat::new(chat, dialect)
						};
						chat = Arc::new(wrapped);
					}
					(
						chat,
						matches!(
							effective.transport,
							TransportId::Cursor | TransportId::GitLabDuoWorkflow
						),
					)
				} else {
					let attempt =
						ProviderAttempt::new(effective.clone(), route(&effective), egress.clone())
							.map_err(|source| RouteRegistrationError::Provider {
								provider: effective.id.clone(),
								source,
							})?;
					let mut stack_config = config(&effective);
					stack_config.compat = effective.compat;
					if matches!(
						effective.transport,
						TransportId::GoogleGenAi | TransportId::GoogleVertex | TransportId::GoogleCca
					) {
						// Google semantic retries are capped at two re-dispatches by
						// the transport contract; a route may choose a stricter bound.
						stack_config.resample.max_attempts = stack_config.resample.max_attempts.min(2);
					}
					stack_config.dialect.get_or_insert(DialectSelection::Auto);
					if stack_config.omp_dialect.is_none() {
						stack_config.omp_dialect.clone_from(&omp_dialect);
					}
					let stack =
						RouteStackBuilder::new(dependencies(&effective), stack_config).build(attempt);
					(Arc::new(ServiceChat::new(stack)), false)
				};
			assembled.push((
				provider.id.clone(),
				wire,
				default.clone(),
				provider.base_url_overridden,
				provider.transport_overridden,
				requires_executor,
				chat,
			));
		}
	}

	let registered = assembled.len();
	for (provider, wire, default, base_url, transport, requires_executor, chat) in assembled {
		resolver.register_wire_stack(
			provider,
			wire,
			default,
			base_url,
			transport,
			requires_executor,
			chat,
		);
	}
	Ok(RouteRegistration { registered })
}

const fn is_specialized(transport: TransportId) -> bool {
	matches!(
		transport,
		TransportId::Cursor
			| TransportId::Devin
			| TransportId::GitLabDuoWorkflow
			| TransportId::Embedded
			| TransportId::Omp
	)
}

fn specialized_chat(chats: &SpecializedChats, provider: &ProviderEntry) -> Option<Arc<dyn Chat>> {
	chats.by_provider.get(&provider.id).cloned().or_else(|| {
		match provider.transport {
			TransportId::Cursor => chats.cursor.as_ref(),
			TransportId::Devin => chats.devin.as_ref(),
			TransportId::GitLabDuoWorkflow => chats.gitlab_duo_workflow.as_ref(),
			TransportId::Embedded => chats.embedded.as_ref(),
			TransportId::Omp => chats.omp.as_ref(),
			_ => None,
		}
		.cloned()
	})
}
