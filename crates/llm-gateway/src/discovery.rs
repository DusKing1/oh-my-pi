//! Model and provider discovery RPCs backed by the joined inference registry.
//!
//! This module is intentionally only a wire boundary. The registry owns joins,
//! epoch rotation, retained deltas, and refresh policy; these RPCs translate
//! its snapshots and events without inventing a second cursor policy. In
//! particular, provider routing data stays in the gateway. The previous
//! implementation denormalized that data into 4,282 model rows; wire cards
//! deliberately contain only display, capability, pricing, and availability
//! data.

use std::{
	collections::{BTreeMap, BTreeSet},
	sync::Arc,
	time::{SystemTime, UNIX_EPOCH},
};

use futures::{StreamExt, future::join_all, stream::BoxStream};
use omp_core::Str;
use omp_llm_catalog::{
	discovery::{self, Account, Discovery},
	models::{Availability, Modality, ModelCard, PriceUnit, Source},
	provider::{AuthSpec, Facet, ProviderCatalog, ProviderEntry},
	registry::{Cursor, ListFilter, ListSnapshot, ModelEvent, Registry},
};
use omp_llm_types::Effort;
use omp_proto::inference::v1 as pb;
use parking_lot::{Mutex, RwLock};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};
use tonic::{Request, Response, Status};

/// Stream returned by [`DiscoveryService::watch_models`].
pub type WatchModelsStream = BoxStream<'static, Result<pb::ModelEvent, Status>>;

/// The discovery portion of `omp.inference.v1.Inference`.
///
/// Clones share one registry so a refresh and every list/watch observe the same
/// atomic cursor history. Credential state remains injected through the
/// registry's `CredentialView` seam.
#[derive(Clone)]
pub struct DiscoveryService {
	registry:      Arc<RwLock<Registry>>,
	providers:     Arc<ProviderCatalog>,
	discovery:     Option<Discovery>,
	refresh_locks: Arc<Mutex<BTreeMap<Str, Arc<AsyncMutex<()>>>>>,
	_maintenance:  Arc<Maintenance>,
}

struct Maintenance {
	task: Option<tokio::task::JoinHandle<()>>,
}

impl Maintenance {
	fn start(registry: &Arc<RwLock<Registry>>, interval_ms: u64) -> Self {
		let Ok(runtime) = tokio::runtime::Handle::try_current() else {
			return Self { task: None };
		};
		let registry = Arc::downgrade(registry);
		let task = runtime.spawn(async move {
			let mut interval = tokio::time::interval(std::time::Duration::from_millis(interval_ms));
			interval.tick().await;
			loop {
				interval.tick().await;
				let Some(registry) = registry.upgrade() else {
					break;
				};
				registry.write().expire_stale(unix_now_ms());
			}
		});
		Self { task: Some(task) }
	}
}

impl Drop for Maintenance {
	fn drop(&mut self) {
		if let Some(task) = self.task.take() {
			task.abort();
		}
	}
}

impl DiscoveryService {
	/// Creates a discovery service over a configured joined registry.
	#[must_use]
	pub fn new(registry: Registry, providers: ProviderCatalog) -> Self {
		let discovery = registry.discovery();
		let interval_ms = (registry.source_ttl_ms() / 2).clamp(1, 30_000);
		let registry = Arc::new(RwLock::new(registry));
		let maintenance = Arc::new(Maintenance::start(&registry, interval_ms));
		Self {
			registry,
			providers: Arc::new(providers),
			discovery,
			refresh_locks: Arc::new(Mutex::new(BTreeMap::new())),
			_maintenance: maintenance,
		}
	}

	/// Creates discovery over the exact registry shared by routing, facades,
	/// role resolution, list, and watch.
	#[must_use]
	pub fn from_shared(
		registry: Arc<RwLock<Registry>>,
		providers: Arc<ProviderCatalog>,
		discovery: Discovery,
	) -> Self {
		registry
			.write()
			.configure_discovery(providers.as_ref().clone(), discovery.clone());
		let interval_ms = (registry.read().source_ttl_ms() / 2).clamp(1, 30_000);
		let maintenance = Arc::new(Maintenance::start(&registry, interval_ms));
		Self {
			registry,
			providers,
			discovery: Some(discovery),
			refresh_locks: Arc::new(Mutex::new(BTreeMap::new())),
			_maintenance: maintenance,
		}
	}

	/// Returns the registry shared by all discovery and routing operations.
	#[must_use]
	pub fn registry(&self) -> Arc<RwLock<Registry>> {
		Arc::clone(&self.registry)
	}

	/// Implements `Inference.ListProviders`.
	pub async fn list_providers(
		&self,
		request: Request<pb::ListProvidersRequest>,
	) -> Result<Response<pb::ListProvidersResponse>, Status> {
		let requested_facet = facet_filter(request.into_inner().facet)?;
		let registry = self.registry.read();
		let (models, cursor) = registry.list(&ListFilter::default());
		let providers = self
			.providers
			.values()
			.filter(|provider| requested_facet.is_none_or(|facet| provider.facets.contains(&facet)))
			.map(|provider| {
				let (model_count, credentialed) = models
					.iter()
					.filter(|card| card.provider == provider.id)
					.fold((0_usize, false), |(count, credentialed), card| {
						(
							count.saturating_add(1),
							credentialed
								|| matches!(
									card.availability,
									Availability::Available | Availability::Blocked
								),
						)
					});
				pb::ProviderCard {
					id: provider.id.to_string(),
					name: provider.id.to_string(),
					facets: provider
						.facets
						.iter()
						.copied()
						.map(facet_to_proto)
						.collect(),
					auth: vec![auth_to_proto(&provider.auth)],
					credentialed,
					model_count: u32::try_from(model_count).unwrap_or(u32::MAX),
					props: None,
				}
			})
			.collect();
		Ok(Response::new(pb::ListProvidersResponse {
			providers,
			cursor: Some(cursor_to_proto(cursor)),
		}))
	}

	/// Implements `Inference.ListModels`.
	pub async fn list_models(
		&self,
		request: Request<pb::ListModelsRequest>,
	) -> Result<Response<pb::ListModelsResponse>, Status> {
		let request = request.into_inner();
		let filter = ListFilter::builder()
			.maybe_provider((!request.provider.is_empty()).then(|| request.provider.into()))
			.maybe_facet(facet_filter(request.facet)?)
			.available_only(request.available_only)
			.build();
		let snapshot = self.registry.read().list_snapshot(&filter);
		Ok(Response::new(pb::ListModelsResponse {
			models: snapshot.models.into_iter().map(model_to_proto).collect(),
			cursor: Some(cursor_to_proto(snapshot.cursor)),
			roles:  snapshot
				.roles
				.into_iter()
				.map(|(role, model)| (role.to_string(), model.to_string()))
				.collect(),
		}))
	}

	/// Implements resumable `Inference.WatchModels` without altering registry
	/// cursor semantics.
	pub async fn watch_models(
		&self,
		request: Request<pb::WatchModelsRequest>,
	) -> Result<Response<WatchModelsStream>, Status> {
		let wire_since = request.into_inner().since;
		let registry = self.registry.read();
		let since = wire_since.map(|cursor| {
			let (_, mut native) = registry.list(&ListFilter::default());
			native.epoch = cursor.epoch;
			native.generation = cursor.generation;
			native
		});
		let stream = registry
			.watch(since)
			.map(|event| Ok(event_to_proto(event)))
			.boxed();
		Ok(Response::new(stream))
	}

	/// Implements `Inference.RefreshModels` and returns the refreshed unfiltered
	/// list.
	///
	/// All provider/account I/O completes before the shared registry is write
	/// locked. Dropping this future therefore cancels upstream work and leaves
	/// the previous snapshot intact. Refreshes serialize only against the same
	/// provider; unrelated providers continue independently.
	pub async fn refresh_models(
		&self,
		request: Request<pb::RefreshModelsRequest>,
	) -> Result<Response<pb::ListModelsResponse>, Status> {
		let requested = request.into_inner().provider;
		let targets = self.refresh_targets(requested.as_str())?;
		let discovery = self
			.discovery
			.clone()
			.ok_or_else(|| Status::failed_precondition("model discovery is not configured"))?;
		let locks = targets
			.iter()
			.map(|entry| self.provider_refresh_lock(entry.id.as_str()))
			.collect::<Vec<_>>();
		let mut guards: Vec<OwnedMutexGuard<()>> = Vec::with_capacity(locks.len());
		for lock in locks {
			guards.push(lock.lock_owned().await);
		}

		let gathered = join_all(
			targets
				.into_iter()
				.map(|entry| gather_provider(discovery.clone(), entry)),
		)
		.await;
		let explicitly_requested = !requested.is_empty();
		let mut first_error = None;
		let mut successful_sources = 0usize;
		for provider in &gathered {
			successful_sources = successful_sources.saturating_add(provider.snapshots.len());
			if first_error.is_none() {
				first_error = provider.first_error.clone();
			}
		}
		if explicitly_requested
			&& successful_sources == 0
			&& let Some(error) = first_error
		{
			return Err(discovery_status(error));
		}

		let now_ms = unix_now_ms();
		let mut registry = self.registry.write();
		let ttl_ms = registry.source_ttl_ms();
		registry.expire_stale(now_ms);
		for provider in gathered {
			if let Some(accounts) = provider.accounts {
				registry.retain_discovered_accounts(provider.entry.id.as_str(), &accounts);
			}
			let authoritative = provider
				.entry
				.discovery
				.as_ref()
				.is_some_and(|discovery| discovery.authoritative);
			for snapshot in provider.snapshots {
				registry.apply_discovered_account(
					provider.entry.id.as_str(),
					snapshot.account.key.as_str(),
					snapshot.cards,
					authoritative,
					now_ms,
					ttl_ms,
				);
			}
		}
		drop(guards);
		let snapshot = registry.list_snapshot(&ListFilter::default());
		Ok(Response::new(snapshot_to_proto(snapshot)))
	}

	fn refresh_targets(&self, requested: &str) -> Result<Vec<ProviderEntry>, Status> {
		if !requested.is_empty() {
			let provider = self
				.providers
				.get(requested)
				.filter(|provider| discovery::supports(provider))
				.cloned()
				.ok_or_else(|| {
					Status::not_found(format!("provider {requested} is not discoverable"))
				})?;
			return Ok(vec![provider]);
		}
		Ok(self
			.providers
			.values()
			.filter(|provider| discovery::supports(provider))
			.cloned()
			.collect())
	}

	fn provider_refresh_lock(&self, provider: &str) -> Arc<AsyncMutex<()>> {
		let mut locks = self.refresh_locks.lock();
		Arc::clone(
			locks
				.entry(Str::from(provider))
				.or_insert_with(|| Arc::new(AsyncMutex::new(()))),
		)
	}
}

struct GatheredSnapshot {
	account: Account,
	cards:   Vec<ModelCard>,
}

struct GatheredProvider {
	entry:       ProviderEntry,
	accounts:    Option<BTreeSet<Str>>,
	snapshots:   Vec<GatheredSnapshot>,
	first_error: Option<discovery::Error>,
}

async fn gather_provider(discovery: Discovery, entry: ProviderEntry) -> GatheredProvider {
	let mut accounts = match discovery.accounts(&entry).await {
		Ok(accounts) => accounts,
		Err(error) => {
			return GatheredProvider {
				entry,
				accounts: None,
				snapshots: Vec::new(),
				first_error: Some(error),
			};
		},
	};
	accounts.sort();
	let account_keys = accounts.iter().map(|account| account.key.clone()).collect();
	let results = join_all(accounts.into_iter().map(|account| {
		let discovery = discovery.clone();
		let entry = entry.clone();
		async move {
			let result = discovery.discover(&entry, &account).await;
			(account, result)
		}
	}))
	.await;
	let mut snapshots = Vec::new();
	let mut first_error = None;
	for (account, result) in results {
		match result {
			Ok(cards) => snapshots.push(GatheredSnapshot { account, cards }),
			Err(error) => {
				first_error.get_or_insert(error);
			},
		}
	}
	GatheredProvider { entry, accounts: Some(account_keys), snapshots, first_error }
}
fn snapshot_to_proto(snapshot: ListSnapshot) -> pb::ListModelsResponse {
	pb::ListModelsResponse {
		models: snapshot.models.into_iter().map(model_to_proto).collect(),
		cursor: Some(cursor_to_proto(snapshot.cursor)),
		roles:  snapshot
			.roles
			.into_iter()
			.map(|(role, model)| (role.to_string(), model.to_string()))
			.collect(),
	}
}

fn facet_filter(value: i32) -> Result<Option<Facet>, Status> {
	match pb::Facet::try_from(value) {
		Ok(pb::Facet::Unspecified) => Ok(None),
		Ok(value) => Ok(Some(facet_from_proto(value))),
		Err(_) => Err(Status::invalid_argument(format!("unknown facet value {value}"))),
	}
}

fn facet_from_proto(value: pb::Facet) -> Facet {
	match value {
		pb::Facet::Unspecified => unreachable!("unspecified facets are filtered before conversion"),
		pb::Facet::Chat => Facet::Chat,
		pb::Facet::Embed => Facet::Embeddings,
		pb::Facet::ImageGen => Facet::ImageGeneration,
		pb::Facet::VideoGen => Facet::VideoGeneration,
		pb::Facet::Speak => Facet::AudioSpeech,
		pb::Facet::Transcribe => Facet::AudioTranscription,
		pb::Facet::Realtime => Facet::Chat,
		pb::Facet::Search => Facet::Chat,
	}
}

const fn facet_to_proto(value: Facet) -> i32 {
	match value {
		Facet::Chat => pb::Facet::Chat as i32,
		Facet::Embeddings => pb::Facet::Embed as i32,
		Facet::Rerank => pb::Facet::Chat as i32,
		Facet::AudioSpeech => pb::Facet::Speak as i32,
		Facet::AudioTranscription => pb::Facet::Transcribe as i32,
		Facet::ImageGeneration => pb::Facet::ImageGen as i32,
		Facet::VideoGeneration => pb::Facet::VideoGen as i32,
	}
}

const fn auth_to_proto(value: &AuthSpec) -> i32 {
	match value {
		AuthSpec::None => pb::provider_card::AuthKind::None as i32,
		AuthSpec::OAuth { .. } => pb::provider_card::AuthKind::Oauth as i32,
		AuthSpec::Bearer { .. }
		| AuthSpec::OptionalBearer { .. }
		| AuthSpec::Header { .. }
		| AuthSpec::Query { .. }
		| AuthSpec::AwsSigV4
		| AuthSpec::GoogleAdc { .. } => pb::provider_card::AuthKind::ApiKey as i32,
	}
}

fn cursor_to_proto(cursor: Cursor) -> pb::Cursor {
	pb::Cursor { epoch: cursor.epoch, generation: cursor.generation }
}

fn event_to_proto(event: ModelEvent) -> pb::ModelEvent {
	let cursor = cursor_to_proto(event.cursor().clone());
	let event = match event {
		ModelEvent::Upserted { card, .. } => pb::model_event::Event::Upserted(model_to_proto(*card)),
		ModelEvent::Removed { id, .. } => pb::model_event::Event::RemovedId(id.to_string()),
		ModelEvent::Reset { .. } => pb::model_event::Event::Reset(pb::model_event::Reset {}),
		_ => pb::model_event::Event::Reset(pb::model_event::Reset {}),
	};
	pb::ModelEvent { cursor: Some(cursor), event: Some(event) }
}

fn model_to_proto(card: ModelCard) -> pb::ModelCard {
	// Enumerating client fields here is the security boundary: effort routing and
	// provider base URLs, headers, transports, and compat flags cannot cross it.
	pb::ModelCard {
		id:                card.id.to_string(),
		provider:          card.provider.to_string(),
		model:             card.model.to_string(),
		name:              card.name.to_string(),
		family:            card.family.to_string(),
		facets:            card.facets.into_iter().map(facet_to_proto).collect(),
		inputs:            card.inputs.into_iter().map(modality_to_proto).collect(),
		outputs:           card.outputs.into_iter().map(modality_to_proto).collect(),
		reasoning:         card.reasoning,
		efforts:           card.efforts.into_iter().map(effort_to_proto).collect(),
		context_window:    card.context_window,
		max_output_tokens: card.max_output_tokens,
		pricing:           card
			.pricing
			.into_iter()
			.map(|price| pb::Price {
				unit:      price_unit_to_proto(price.unit),
				nanos_usd: price.nanos_usd,
			})
			.collect(),
		availability:      availability_to_proto(card.availability),
		source:            source_to_proto(card.source),
		blocked_until_ms:  card.blocked_until_ms,
		deprecated:        card.deprecated,
		updated_at_ms:     card.updated_at_ms,
		props:             (!card.props.is_empty()).then(|| card.props.into()),
	}
}

const fn modality_to_proto(value: Modality) -> i32 {
	match value {
		Modality::Unspecified => pb::Modality::Unspecified as i32,
		Modality::Text => pb::Modality::Text as i32,
		Modality::Image => pb::Modality::Image as i32,
		Modality::Audio => pb::Modality::Audio as i32,
		Modality::Video => pb::Modality::Video as i32,
		Modality::Pdf => pb::Modality::Pdf as i32,
		_ => pb::Modality::Unspecified as i32,
	}
}

const fn effort_to_proto(value: Effort) -> i32 {
	match value {
		Effort::Off => pb::Effort::Off as i32,
		Effort::Minimal => pb::Effort::Minimal as i32,
		Effort::Low => pb::Effort::Low as i32,
		Effort::Medium => pb::Effort::Medium as i32,
		Effort::High => pb::Effort::High as i32,
		Effort::XHigh => pb::Effort::Xhigh as i32,
		Effort::Max => pb::Effort::Max as i32,
		_ => pb::Effort::Unspecified as i32,
	}
}

const fn price_unit_to_proto(value: PriceUnit) -> i32 {
	match value {
		PriceUnit::MtokInput => pb::price::Unit::MtokInput as i32,
		PriceUnit::MtokOutput => pb::price::Unit::MtokOutput as i32,
		PriceUnit::MtokCacheRead => pb::price::Unit::MtokCacheRead as i32,
		PriceUnit::MtokCacheWrite => pb::price::Unit::MtokCacheWrite as i32,
		PriceUnit::Image => pb::price::Unit::Image as i32,
		PriceUnit::VideoSecond => pb::price::Unit::VideoSecond as i32,
		PriceUnit::AudioSecond => pb::price::Unit::AudioSecond as i32,
		PriceUnit::McharInput => pb::price::Unit::McharInput as i32,
		PriceUnit::Request => pb::price::Unit::Request as i32,
		_ => pb::price::Unit::Unspecified as i32,
	}
}

const fn availability_to_proto(value: Availability) -> i32 {
	match value {
		Availability::Unspecified => pb::Availability::Unspecified as i32,
		Availability::Available => pb::Availability::Available as i32,
		Availability::LoginRequired => pb::Availability::LoginRequired as i32,
		Availability::Blocked => pb::Availability::Blocked as i32,
		Availability::Disabled => pb::Availability::Disabled as i32,
		_ => pb::Availability::Unspecified as i32,
	}
}

const fn source_to_proto(value: Source) -> i32 {
	match value {
		Source::Unspecified => pb::model_card::Source::Unspecified as i32,
		Source::Bundled => pb::model_card::Source::Bundled as i32,
		Source::Discovered => pb::model_card::Source::Discovered as i32,
		Source::Configured => pb::model_card::Source::Configured as i32,
		_ => pb::model_card::Source::Unspecified as i32,
	}
}

fn discovery_status(error: discovery::Error) -> Status {
	match error {
		discovery::Error::UnsupportedProvider(_) | discovery::Error::UnregisteredProtocol { .. } => {
			Status::invalid_argument(error.to_string())
		},
		discovery::Error::Transport(_) | discovery::Error::HttpStatus { .. } => {
			Status::unavailable(error.to_string())
		},
		discovery::Error::InvalidUrl { .. } | discovery::Error::InvalidPayload { .. } => {
			Status::internal(error.to_string())
		},
		_ => Status::internal(error.to_string()),
	}
}
fn unix_now_ms() -> u64 {
	let millis = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis();
	u64::try_from(millis).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
	use std::{collections::BTreeMap, future::pending, sync::Arc};

	use async_trait::async_trait;
	use bytes::Bytes;
	use futures::StreamExt;
	use omp_core::Str;
	use omp_llm_catalog::{
		discovery::{Account, Discovery, DiscoveryHttp, Error, HttpResponse},
		models::{Availability, ModelCard},
		provider::{ProviderCatalog, ProviderEntry, load_providers},
		registry::{CredentialView, Registry},
	};
	use omp_proto::inference::v1 as pb;
	use serde_json::json;
	use tokio::sync::Notify;
	use tonic::Request;

	use super::DiscoveryService;

	struct Credentials(BTreeMap<Str, Availability>);

	impl CredentialView for Credentials {
		fn availability(&self, provider: &str) -> Availability {
			self
				.0
				.get(provider)
				.copied()
				.unwrap_or(Availability::LoginRequired)
		}
	}

	fn providers() -> ProviderCatalog {
		load_providers(
			r#"
[providers.alpha]
transport = "open-ai-chat"
base_url = "https://alpha.invalid/v1"
auth = { type = "none" }
facets = ["chat", "embeddings"]

[providers.beta]
transport = "open-ai-chat"
base_url = "https://beta.invalid/v1"
auth = { type = "bearer", env = ["BETA_KEY"] }
facets = ["chat"]
"#,
		)
		.expect("provider fixture must parse")
	}

	fn card(provider: &str, model: &str, facets: &[&str]) -> ModelCard {
		serde_json::from_value(json!({
			"id": format!("{provider}/{model}"),
			"provider": provider,
			"model": model,
			"name": model,
			"family": model,
			"facets": facets,
			"inputs": ["text"],
			"outputs": ["text"],
			"reasoning": false,
			"efforts": [],
			"context_window": 8192,
			"max_output_tokens": 1024,
			"pricing": [],
			"availability": "unspecified",
			"source": "bundled",
			"blocked_until_ms": 0,
			"deprecated": false,
			"updated_at_ms": 1,
			"props": {}
		}))
		.expect("model fixture must parse")
	}

	fn service() -> DiscoveryService {
		let cards = [
			card("alpha", "mini", &["chat"]),
			card("alpha", "embedder", &["embeddings"]),
			card("alpha", "gpt-5.4", &["chat"]),
			card("beta", "locked", &["chat"]),
		];
		let credentials = Arc::new(Credentials(BTreeMap::from([
			(Str::new_static("alpha"), Availability::Available),
			(Str::new_static("beta"), Availability::LoginRequired),
		])));
		DiscoveryService::new(Registry::from_cards(&cards, credentials), providers())
	}

	#[tokio::test]
	async fn list_filters_by_facet_and_availability() {
		let service = service();
		let response = service
			.list_models(Request::new(pb::ListModelsRequest {
				facet: pb::Facet::Chat as i32,
				available_only: true,
				..Default::default()
			}))
			.await
			.expect("list must succeed")
			.into_inner();
		assert_eq!(
			response
				.models
				.iter()
				.map(|card| card.id.as_str())
				.collect::<Vec<_>>(),
			["alpha/gpt-5.4", "alpha/mini"]
		);

		let providers = service
			.list_providers(Request::new(pb::ListProvidersRequest { facet: pb::Facet::Embed as i32 }))
			.await
			.expect("provider list must succeed")
			.into_inner();
		assert_eq!(providers.providers.len(), 1);
		assert_eq!(providers.providers[0].id, "alpha");
		assert!(providers.providers[0].credentialed);
		assert_eq!(providers.providers[0].model_count, 3);
	}

	#[tokio::test]
	async fn roles_appear_and_resolve() {
		let response = service()
			.list_models(Request::new(pb::ListModelsRequest::default()))
			.await
			.expect("list must succeed")
			.into_inner();
		assert_eq!(response.roles.get("tiny").map(String::as_str), Some("alpha/mini"));
		assert_eq!(response.roles.get("smol").map(String::as_str), Some("alpha/mini"));
		assert_eq!(response.roles.get("slow").map(String::as_str), Some("alpha/gpt-5.4"));
	}

	#[tokio::test]
	async fn watch_resumes_from_a_live_cursor() {
		let service = service();
		let listed = service
			.list_models(Request::new(pb::ListModelsRequest::default()))
			.await
			.expect("list must succeed")
			.into_inner();
		service
			.registry()
			.write()
			.apply_discovered("alpha", vec![card("alpha", "new", &["chat"])]);
		let mut watch = service
			.watch_models(Request::new(pb::WatchModelsRequest { since: listed.cursor }))
			.await
			.expect("watch must open")
			.into_inner();
		let event = watch
			.next()
			.await
			.expect("replayed event")
			.expect("valid event");
		assert!(matches!(
			event.event,
			Some(pb::model_event::Event::Upserted(card)) if card.id == "alpha/new"
		));
	}

	#[tokio::test]
	async fn stale_epoch_cursor_opens_with_reset() {
		let service = service();
		let mut watch = service
			.watch_models(Request::new(pb::WatchModelsRequest {
				since: Some(pb::Cursor { epoch: Bytes::from_static(b"dead"), generation: 99 }),
			}))
			.await
			.expect("watch must open")
			.into_inner();
		let event = watch
			.next()
			.await
			.expect("reset event")
			.expect("valid event");
		assert!(matches!(event.event, Some(pb::model_event::Event::Reset(_))));
	}

	#[tokio::test]
	async fn relist_after_reset_loses_and_duplicates_nothing() {
		let service = service();
		let old = service
			.list_models(Request::new(pb::ListModelsRequest::default()))
			.await
			.expect("initial list")
			.into_inner()
			.cursor;
		service.registry().write().rebuild();
		let mut watch = service
			.watch_models(Request::new(pb::WatchModelsRequest { since: old }))
			.await
			.expect("watch must open")
			.into_inner();
		let reset = watch
			.next()
			.await
			.expect("reset event")
			.expect("valid event");
		assert!(matches!(reset.event, Some(pb::model_event::Event::Reset(_))));
		let relisted = service
			.list_models(Request::new(pb::ListModelsRequest::default()))
			.await
			.expect("re-list")
			.into_inner();
		assert_eq!(reset.cursor, relisted.cursor);
		let ids = relisted
			.models
			.iter()
			.map(|card| card.id.as_str())
			.collect::<std::collections::BTreeSet<_>>();
		assert_eq!(ids.len(), relisted.models.len());
		assert_eq!(relisted.models.len(), 4);
	}

	struct OllamaHttp;

	#[async_trait]
	impl DiscoveryHttp for OllamaHttp {
		async fn execute(
			&self,
			_provider: &ProviderEntry,
			_account: &Account,
			_request: http::Request<Bytes>,
		) -> Result<HttpResponse, Error> {
			Ok(HttpResponse::new(200, Bytes::from_static(br#"{"models":[{"name":"fresh:latest"}]}"#)))
		}
	}

	#[tokio::test]
	async fn refresh_surfaces_a_newly_discovered_local_model() {
		let providers = load_providers(
			r#"
[providers.ollama]
transport = "open-ai-chat"
base_url = "http://127.0.0.1:11434"
auth = { type = "none" }
facets = ["chat"]
"#,
		)
		.expect("provider fixture");
		let credentials = Arc::new(Credentials(BTreeMap::from([(
			Str::new_static("ollama"),
			Availability::Available,
		)])));
		let mut registry = Registry::from_cards(&[], credentials);
		registry.configure_discovery(providers.clone(), Discovery::new(Arc::new(OllamaHttp)));
		let response = DiscoveryService::new(registry, providers)
			.refresh_models(Request::new(pb::RefreshModelsRequest { provider: "ollama".to_owned() }))
			.await
			.expect("refresh must succeed")
			.into_inner();
		assert_eq!(response.models.len(), 1);
		assert_eq!(response.models[0].id, "ollama/fresh:latest");
	}

	struct BlockingHttp {
		started: Arc<Notify>,
	}

	#[async_trait]
	impl DiscoveryHttp for BlockingHttp {
		async fn execute(
			&self,
			_provider: &ProviderEntry,
			_account: &Account,
			_request: http::Request<Bytes>,
		) -> Result<HttpResponse, Error> {
			self.started.notify_one();
			pending().await
		}
	}

	#[tokio::test]
	async fn cancelling_refresh_discards_the_uncommitted_batch() {
		let providers = load_providers(
			r#"
[providers.ollama]
transport = "open-ai-chat"
base_url = "http://127.0.0.1:11434"
auth = { type = "none" }
facets = ["chat"]
"#,
		)
		.expect("provider fixture");
		let started = Arc::new(Notify::new());
		let mut registry = Registry::from_cards(
			&[card("ollama", "fallback", &["chat"])],
			Arc::new(Credentials(BTreeMap::from([(
				Str::new_static("ollama"),
				Availability::Available,
			)]))),
		);
		registry.configure_discovery(
			providers.clone(),
			Discovery::new(Arc::new(BlockingHttp { started: Arc::clone(&started) })),
		);
		let service = DiscoveryService::new(registry, providers);
		let refresh = {
			let service = service.clone();
			tokio::spawn(async move {
				service
					.refresh_models(Request::new(pb::RefreshModelsRequest {
						provider: "ollama".to_owned(),
					}))
					.await
			})
		};
		started.notified().await;
		refresh.abort();
		assert!(refresh.await.is_err());

		let listed = service
			.list_models(Request::new(pb::ListModelsRequest::default()))
			.await
			.expect("list")
			.into_inner();
		assert_eq!(
			listed
				.models
				.iter()
				.map(|model| model.id.as_str())
				.collect::<Vec<_>>(),
			["ollama/fallback",]
		);
	}

	#[test]
	fn serialized_card_leaks_no_routing_internals() {
		let value = serde_json::to_value(super::model_to_proto(card("alpha", "mini", &["chat"])))
			.expect("wire card serializes");
		let object = value.as_object().expect("card is an object");
		for forbidden in
			["base_url", "baseUrl", "headers", "transport", "compat", "fallback_base_urls"]
		{
			assert!(!object.contains_key(forbidden), "wire card leaked {forbidden}");
		}
	}
}
