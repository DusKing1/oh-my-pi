//! Joined model registry with resumable, epoch-bearing deltas.
//!
//! The registry is the single client-facing join of bundled/configured model
//! cards, live discovery, and broker-supplied credential availability. It owns
//! no credentials: callers inject [`CredentialView`].

use std::{
	collections::{BTreeMap, BTreeSet, VecDeque},
	sync::Arc,
	time::{SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use futures::{StreamExt, stream::BoxStream};
use omp_core::{Str, fmts};
use omp_llm_types::Effort;
use parking_lot::RwLock;
use smallvec::SmallVec;
use tokio::sync::broadcast;
use ulid::Ulid;

use super::{
	discovery::{self, Discovery},
	models::{Availability, ModelCard, ModelCatalog, Source},
	provider::{Facet, ProviderCatalog, ProviderEntry},
};

/// Number of model deltas retained for cursor resumption.
///
/// A cursor older than the event immediately preceding this 256-event window
/// receives [`ModelEvent::Reset`] rather than a partial replay.
pub const DEFAULT_DELTA_RETENTION: usize = 256;
/// Time a successful live source remains authoritative without reconnecting.
pub const DEFAULT_SOURCE_TTL_MS: u64 = 5 * 60 * 1_000;

/// Broker-owned credential state as observed by the model registry.
///
/// The narrow synchronous view keeps `omp-llm-catalog` independent of
/// `omp-llm-broker`; the broker triggers [`Registry::rebuild`] when this view
/// changes. Account-aware implementations override [`Self::availability_for`]
/// so one unavailable credential never hides models supplied by another.
pub trait CredentialView: Send + Sync {
	/// Returns the current provider-wide availability.
	fn availability(&self, provider: &str) -> Availability;

	/// Returns availability for an opaque discovery account source.
	fn availability_for(&self, provider: &str, _account: &str) -> Availability {
		self.availability(provider)
	}
}

/// A resumable position in one registry epoch.
#[derive(Clone, Debug, Eq, PartialEq, bon::Builder)]
#[non_exhaustive]
pub struct Cursor {
	/// Opaque registry-instance token.
	pub epoch:      Bytes,
	/// Monotone event generation within `epoch`.
	pub generation: u64,
}

/// Filter accepted by [`Registry::list`].
#[derive(Clone, Debug, Default, Eq, PartialEq, bon::Builder)]
#[non_exhaustive]
pub struct ListFilter {
	/// Provider id, or `None` for every provider.
	pub provider:       Option<Str>,
	/// Required facet, or `None` for every facet.
	pub facet:          Option<Facet>,
	/// Whether to omit models that cannot currently serve a request.
	pub available_only: bool,
}
/// A list snapshot including the role bindings exposed by the wire response.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ListSnapshot {
	/// Cards matching the request filter.
	pub models: Vec<ModelCard>,
	/// Atomic resume position for `models`.
	pub cursor: Cursor,
	/// Resolved gateway role to canonical model-id bindings.
	pub roles:  BTreeMap<Str, Str>,
}

/// One change in the model registry.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ModelEvent {
	/// A card was added or replaced.
	Upserted {
		/// Position after this event.
		cursor: Cursor,
		/// Current card value.
		card:   Box<ModelCard>,
	},
	/// A card vanished from the effective registry.
	Removed {
		/// Position after this event.
		cursor: Cursor,
		/// Canonical card id.
		id:     Str,
	},
	/// Replay is impossible; the client must re-list before applying deltas.
	Reset {
		/// Position at which the reset was observed.
		cursor: Cursor,
	},
}

impl ModelEvent {
	/// Returns the position after this event.
	#[must_use]
	pub const fn cursor(&self) -> &Cursor {
		match self {
			Self::Upserted { cursor, .. } | Self::Removed { cursor, .. } | Self::Reset { cursor } => {
				cursor
			},
		}
	}
}

/// Ordered candidate lists for gateway model roles.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RoleConfig {
	candidates: BTreeMap<Str, SmallVec<Str, 8>>,
}

impl RoleConfig {
	/// Creates an empty role configuration.
	#[must_use]
	pub const fn empty() -> Self {
		Self { candidates: BTreeMap::new() }
	}

	/// Replaces a role's candidate list while preserving caller order.
	pub fn set<I, S>(&mut self, role: impl Into<Str>, candidates: I)
	where
		I: IntoIterator<Item = S>,
		S: Into<Str>,
	{
		self
			.candidates
			.insert(role.into(), candidates.into_iter().map(Into::into).collect());
	}

	/// Prepends configured candidates before the role's built-in fallbacks.
	pub fn prepend<I, S>(&mut self, role: impl Into<Str>, configured: I)
	where
		I: IntoIterator<Item = S>,
		S: Into<Str>,
	{
		let role = role.into();
		let mut combined: SmallVec<Str, 8> = configured.into_iter().map(Into::into).collect();
		if let Some(fallbacks) = self.candidates.remove(&role) {
			combined.extend(fallbacks);
		}
		self.candidates.insert(role, combined);
	}

	fn get(&self, role: &str) -> Option<&[Str]> {
		self.candidates.get(role).map(AsRef::as_ref)
	}
}

impl Default for RoleConfig {
	fn default() -> Self {
		let mut roles = Self::empty();
		let smol = [
			"cerebras/zai-glm-4.7",
			"cerebras/zai-glm-4.6",
			"cerebras/zai-glm",
			"google-antigravity/gemini-3.1-flash-lite",
			"google-gemini-cli/gemini-3.1-flash-lite",
			"gemini-3.1-flash-lite",
			"gemini-3-1-flash-lite",
			"flash-lite",
			"google-antigravity/gemini-3.5-flash",
			"google-antigravity/gemini-3-flash",
			"google-gemini-cli/gemini-3.5-flash",
			"google-gemini-cli/gemini-3-flash",
			"gemini-3.5-flash",
			"gemini-3-5-flash",
			"gemini-3-flash",
			"haiku-4-5",
			"haiku-4.5",
			"haiku",
			"flash",
			"mini",
		];
		roles.set("tiny", smol);
		roles.set("smol", smol);
		roles.set("slow", [
			"openai-codex/gpt-5.5",
			"openai-codex/gpt-5.4",
			"openai-codex/gpt-5.3-codex",
			"gpt-5.5",
			"gpt-5.4",
			"gpt-5.3-codex",
			"gpt-5.3",
			"gpt-5.2-codex",
			"gpt-5.2",
			"gpt-5.1-codex",
			"gpt-5.1",
			"codex",
			"opus-4.8",
			"opus-4-8",
			"opus-4.7",
			"opus-4-7",
			"opus-4.6",
			"opus-4-6",
			"opus-4.5",
			"opus-4-5",
			"opus-4.1",
			"opus-4-1",
			"pro",
		]);
		roles.set("designer", [
			"google-gemini-cli/gemini-3.1-pro",
			"google-gemini-cli/gemini-3-pro",
			"gemini-3.1-pro",
			"gemini-3-1-pro",
			"gemini-3-pro",
			"gemini-3",
			"google-gemini-cli/gemini-3.5-flash",
			"gemini-3.5-flash",
			"gemini-3-5-flash",
		]);
		roles
	}
}

struct WatchState {
	epoch:      Bytes,
	generation: u64,
	deltas:     VecDeque<ModelEvent>,
}

struct WatchHub {
	state:     RwLock<WatchState>,
	sender:    broadcast::Sender<ModelEvent>,
	retention: usize,
}

impl WatchHub {
	fn new(retention: usize) -> Self {
		let retention = retention.max(1);
		let (sender, _) = broadcast::channel(retention);
		Self {
			state: RwLock::new(WatchState {
				epoch:      mint_epoch(),
				generation: 0,
				deltas:     VecDeque::with_capacity(retention),
			}),
			sender,
			retention,
		}
	}

	fn cursor(&self) -> Cursor {
		let state = self.state.read();
		Cursor { epoch: state.epoch.clone(), generation: state.generation }
	}

	fn emit(&self, event: impl FnOnce(Cursor) -> ModelEvent) -> Cursor {
		let mut state = self.state.write();
		state.generation = state.generation.saturating_add(1);
		let cursor = Cursor { epoch: state.epoch.clone(), generation: state.generation };
		let event = event(cursor.clone());
		if state.deltas.len() == self.retention {
			state.deltas.pop_front();
		}
		state.deltas.push_back(event.clone());
		drop(state);
		let _no_receivers = self.sender.send(event);
		cursor
	}

	fn rotate(&self) -> Cursor {
		let mut state = self.state.write();
		state.epoch = mint_epoch();
		state.generation = 0;
		state.deltas.clear();
		let cursor = Cursor { epoch: state.epoch.clone(), generation: 0 };
		drop(state);
		let _no_receivers = self
			.sender
			.send(ModelEvent::Reset { cursor: cursor.clone() });
		cursor
	}
}

#[derive(Clone, Copy)]
enum AvailabilityAuthority {
	LocalCredentials,
	Upstream,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SourceKey {
	provider: Str,
	account:  Str,
}

#[derive(Clone)]
struct LiveSource {
	cards:         BTreeMap<Str, ModelCard>,
	authority:     AvailabilityAuthority,
	authoritative: bool,
	expires_at_ms: u64,
}

/// Catalog, discovery, and credential join exposed to gateway model RPCs.
#[non_exhaustive]
pub struct Registry {
	static_cards:  BTreeMap<Str, ModelCard>,
	live_sources:  BTreeMap<SourceKey, LiveSource>,
	cards:         BTreeMap<Str, ModelCard>,
	credentials:   Arc<dyn CredentialView>,
	roles:         RoleConfig,
	providers:     ProviderCatalog,
	discovery:     Option<Discovery>,
	source_ttl_ms: u64,
	watch:         Arc<WatchHub>,
}

impl Registry {
	/// Creates a registry from the bundled/static catalog.
	#[must_use]
	pub fn new(catalog: &ModelCatalog, credentials: Arc<dyn CredentialView>) -> Self {
		Self::from_cards_with_retention(catalog.models(), credentials, DEFAULT_DELTA_RETENTION)
	}

	/// Creates a registry from an explicit card slice.
	///
	/// This is useful for user/project catalogs and deterministic embedding.
	#[must_use]
	pub fn from_cards(cards: &[ModelCard], credentials: Arc<dyn CredentialView>) -> Self {
		Self::from_cards_with_retention(cards, credentials, DEFAULT_DELTA_RETENTION)
	}

	/// Creates a registry with a caller-selected retained-delta capacity.
	#[must_use]
	pub fn from_cards_with_retention(
		cards: &[ModelCard],
		credentials: Arc<dyn CredentialView>,
		retention: usize,
	) -> Self {
		let static_cards = cards
			.iter()
			.cloned()
			.map(|card| (card.id.clone(), card))
			.collect();
		let mut registry = Self {
			static_cards,
			live_sources: BTreeMap::new(),
			cards: BTreeMap::new(),
			credentials,
			roles: RoleConfig::default(),
			providers: ProviderCatalog::new(),
			discovery: None,
			source_ttl_ms: DEFAULT_SOURCE_TTL_MS,
			watch: Arc::new(WatchHub::new(retention)),
		};
		registry.rejoin();
		registry
	}

	/// Installs provider data and the discovery stack used by
	/// [`refresh`](Self::refresh).
	pub fn configure_discovery(&mut self, providers: ProviderCatalog, discovery: Discovery) {
		self.providers = providers;
		self.discovery = Some(discovery);
	}

	/// Returns the configured discovery stack, if production discovery is
	/// enabled for this registry.
	#[must_use]
	pub fn discovery(&self) -> Option<Discovery> {
		self.discovery.clone()
	}

	/// Returns the configured source freshness window.
	#[must_use]
	pub const fn source_ttl_ms(&self) -> u64 {
		self.source_ttl_ms
	}

	/// Replaces role candidate lists and rotates the epoch.
	pub fn set_roles(&mut self, roles: RoleConfig) {
		self.roles = roles;
		self.rebuild();
	}

	/// Rebuilds the catalog × discovery × credentials join.
	///
	/// Rebuilds rotate the epoch even when card values are unchanged. Existing
	/// watchers receive an in-band reset, and cursors from the previous epoch
	/// can never silently resume.
	pub fn rebuild(&mut self) {
		self.rejoin();
		self.watch.rotate();
	}

	/// Replaces the static catalog and performs an epoch-rotating rebuild.
	pub fn replace_catalog(&mut self, catalog: &ModelCatalog) {
		self.static_cards = catalog
			.models()
			.iter()
			.cloned()
			.map(|card| (card.id.clone(), card))
			.collect();
		self.rebuild();
	}

	/// Lists effective cards and the atomic resume position for that snapshot.
	#[must_use]
	pub fn list(&self, filter: &ListFilter) -> (Vec<ModelCard>, Cursor) {
		let cards = self
			.cards
			.values()
			.filter(|card| {
				filter
					.provider
					.as_ref()
					.is_none_or(|provider| provider == &card.provider)
					&& filter
						.facet
						.as_ref()
						.is_none_or(|facet| card.facets.contains(facet))
					&& (!filter.available_only || card.availability == Availability::Available)
			})
			.cloned()
			.collect();
		(cards, self.watch.cursor())
	}

	/// Lists effective cards together with bindings for a complete
	/// `ListModelsResponse`.
	#[must_use]
	pub fn list_snapshot(&self, filter: &ListFilter) -> ListSnapshot {
		let (models, cursor) = self.list(filter);
		ListSnapshot { models, cursor, roles: self.role_bindings() }
	}

	/// Watches changes after `since`.
	///
	/// Absence, a dead epoch, a future cursor, or a cursor older than the
	/// retained delta window yields [`ModelEvent::Reset`] first. The broadcast
	/// subscription is opened before the replay snapshot, and duplicate events
	/// at or below the last emitted cursor are discarded. Consequently a client
	/// may re-list after reset, discard stream events at or below the list
	/// cursor, and lose or duplicate nothing.
	#[must_use = "the stream must be polled to observe model changes"]
	pub fn watch(&self, since: Option<Cursor>) -> BoxStream<'static, ModelEvent> {
		let mut receiver = self.watch.sender.subscribe();
		let (initial, mut last) = {
			let state = self.watch.state.read();
			let current = Cursor { epoch: state.epoch.clone(), generation: state.generation };
			let resumable = since.as_ref().is_some_and(|cursor| {
				let earliest = state
					.deltas
					.front()
					.map_or(state.generation, |event| event.cursor().generation.saturating_sub(1));
				cursor.epoch == state.epoch
					&& cursor.generation <= state.generation
					&& cursor.generation >= earliest
			});
			if resumable {
				let cursor = since.expect("resumable cursor must be present");
				let events = state
					.deltas
					.iter()
					.filter(|event| event.cursor().generation > cursor.generation)
					.cloned()
					.collect::<Vec<_>>();
				(events, cursor)
			} else {
				(vec![ModelEvent::Reset { cursor: current.clone() }], current)
			}
		};
		let watch = Arc::clone(&self.watch);

		async_stream::stream! {
			for event in initial {
				last = event.cursor().clone();
				yield event;
			}
			loop {
				match receiver.recv().await {
					Ok(event) => {
						let cursor = event.cursor();
						let newer = cursor.epoch != last.epoch || cursor.generation > last.generation;
						if newer {
							last = cursor.clone();
							yield event;
						}
					},
					Err(broadcast::error::RecvError::Lagged(_)) => {
						let cursor = watch.cursor();
						if cursor != last {
							last = cursor.clone();
							yield ModelEvent::Reset { cursor };
						}
					},
					Err(broadcast::error::RecvError::Closed) => {
						break;
					},
				}
			}
		}
		.boxed()
	}

	/// Refreshes one provider, or every discovery-capable provider when
	/// `provider` is `None`, and returns the cursor after emitted deltas.
	///
	/// Accounts are enumerated and fetched independently. A failed account
	/// retains its last successful snapshot until source expiry; successful
	/// siblings are still published. A provider with no live account keeps its
	/// deterministic static fallback.
	///
	/// # Errors
	///
	/// Returns an error when discovery is unconfigured, the requested provider
	/// is unknown/unsupported, or every source for an explicitly requested
	/// provider fails.
	pub async fn refresh(&mut self, provider: Option<&str>) -> Result<Cursor, discovery::Error> {
		let discovery = self.discovery.clone().ok_or_else(|| {
			discovery::Error::Transport(Str::from("model discovery is not configured"))
		})?;
		let targets: Vec<ProviderEntry> = match provider {
			Some(id) if !id.is_empty() => {
				let entry = self
					.providers
					.get(id)
					.cloned()
					.ok_or_else(|| discovery::Error::UnsupportedProvider(Str::from(id)))?;
				if !discovery::supports(&entry) {
					return Err(discovery::Error::UnsupportedProvider(entry.id));
				}
				vec![entry]
			},
			None | Some(_) => self
				.providers
				.values()
				.filter(|entry| discovery::supports(entry))
				.cloned()
				.collect(),
		};

		let now_ms = unix_now_ms();
		let explicitly_requested = provider.is_some_and(|value| !value.is_empty());
		let mut first_error = None;
		let mut enumerated = Vec::new();
		let mut batch = Vec::new();
		for entry in targets {
			let mut accounts = match discovery.accounts(&entry).await {
				Ok(accounts) => accounts,
				Err(error) => {
					first_error.get_or_insert(error);
					continue;
				},
			};
			accounts.sort();
			enumerated.push((
				entry.id.clone(),
				accounts
					.iter()
					.map(|account| account.key.clone())
					.collect::<BTreeSet<_>>(),
			));
			let authoritative = entry
				.discovery
				.as_ref()
				.is_some_and(|spec| spec.authoritative);
			for account in accounts {
				match discovery.discover(&entry, &account).await {
					Ok(cards) => {
						batch.push((entry.id.clone(), account.key, cards, authoritative));
					},
					Err(error) => {
						first_error.get_or_insert(error);
					},
				}
			}
		}
		if explicitly_requested
			&& batch.is_empty()
			&& let Some(error) = first_error
		{
			return Err(error);
		}

		self.expire_stale(now_ms);
		for (provider, accounts) in enumerated {
			self.retain_discovered_accounts(provider.as_str(), &accounts);
		}
		for (provider, account, cards, authoritative) in batch {
			self.apply_discovered_account(
				provider.as_str(),
				account.as_str(),
				cards,
				authoritative,
				now_ms,
				self.source_ttl_ms,
			);
		}
		Ok(self.watch.cursor())
	}

	/// Configures the freshness window used by subsequent successful refreshes.
	pub fn set_source_ttl_ms(&mut self, source_ttl_ms: u64) {
		self.source_ttl_ms = source_ttl_ms.max(1);
	}

	/// Merges one provider's complete locally-discovered default source.
	pub fn apply_discovered(&mut self, provider: &str, cards: Vec<ModelCard>) {
		let authoritative = self
			.providers
			.get(provider)
			.and_then(|entry| entry.discovery.as_ref())
			.is_some_and(|discovery| discovery.authoritative);
		self.apply_discovered_account(
			provider,
			"",
			cards,
			authoritative,
			unix_now_ms(),
			self.source_ttl_ms,
		);
	}

	/// Merges one credential-isolated discovery snapshot.
	///
	/// `account` is an opaque, non-secret key. `ttl_ms` controls how long this
	/// source can hide static fallback rows without reconnecting.
	pub fn apply_discovered_account(
		&mut self,
		provider: &str,
		account: &str,
		mut cards: Vec<ModelCard>,
		authoritative: bool,
		refreshed_at_ms: u64,
		ttl_ms: u64,
	) {
		for card in &mut cards {
			card.provider = Str::from(provider);
			card.id = fmts!("{provider}/{}", card.model);
			card.source = Source::Discovered;
		}
		let key = SourceKey { provider: provider.into(), account: account.into() };
		let cards = cards
			.into_iter()
			.map(|card| (card.id.clone(), card))
			.collect();
		self.live_sources.insert(key, LiveSource {
			cards,
			authority: AvailabilityAuthority::LocalCredentials,
			authoritative,
			expires_at_ms: refreshed_at_ms.saturating_add(ttl_ms.max(1)),
		});
		self.publish_join();
	}

	/// Merges one federated provider's complete snapshot.
	///
	/// Upstream availability is preserved because local credentials cannot
	/// repair an upstream route. Federated snapshots do not expire here; their
	/// owning federation transport controls their lifecycle.
	pub fn apply_federated(&mut self, provider: &str, mut cards: Vec<ModelCard>) {
		for card in &mut cards {
			card.provider = Str::from(provider);
			card.id = fmts!("{provider}/{}", card.model);
			card.source = Source::Discovered;
		}
		let key = SourceKey { provider: provider.into(), account: "__federated".into() };
		let cards = cards
			.into_iter()
			.map(|card| (card.id.clone(), card))
			.collect();
		self.live_sources.insert(key, LiveSource {
			cards,
			authority: AvailabilityAuthority::Upstream,
			authoritative: true,
			expires_at_ms: u64::MAX,
		});
		self.publish_join();
	}

	/// Expires disconnected sources and republishes deterministic static
	/// fallback rows. Returns whether the effective registry changed.
	pub fn expire_stale(&mut self, now_ms: u64) -> bool {
		let before = self.live_sources.len();
		self
			.live_sources
			.retain(|_, source| source.expires_at_ms > now_ms);
		if self.live_sources.len() == before {
			return false;
		}
		self.publish_join();
		true
	}

	/// Removes snapshots for accounts no longer reported by an authoritative
	/// account enumeration while retaining failed/reconnecting sources.
	pub fn retain_discovered_accounts(&mut self, provider: &str, accounts: &BTreeSet<Str>) {
		let before = self.live_sources.len();
		self.live_sources.retain(|key, _| {
			key.provider != provider || key.account == "__federated" || accounts.contains(&key.account)
		});
		if self.live_sources.len() != before {
			self.publish_join();
		}
	}

	/// Resolves a role to the first currently available candidate.
	///
	/// Candidates first match canonical ids, then provider-local ids, then the
	/// substring patterns used by the historical `priority.json` fallback list.
	#[must_use]
	pub fn resolve_role(&self, role: &str) -> Option<&ModelCard> {
		self
			.roles
			.get(role)?
			.iter()
			.find_map(|candidate| self.resolve_candidate(candidate))
	}

	/// Returns resolved role bindings suitable for `ListModelsResponse.roles`.
	#[must_use]
	pub fn role_bindings(&self) -> BTreeMap<Str, Str> {
		self
			.roles
			.candidates
			.keys()
			.filter_map(|role| {
				self
					.resolve_role(role)
					.map(|card| (role.clone(), card.id.clone()))
			})
			.collect()
	}

	fn resolve_candidate(&self, candidate: &str) -> Option<&ModelCard> {
		self
			.cards
			.get(candidate)
			.filter(|card| card.availability == Availability::Available)
			.or_else(|| {
				self
					.cards
					.values()
					.filter(|card| {
						card.availability == Availability::Available
							&& card.model.eq_ignore_ascii_case(candidate)
					})
					.min_by(|left, right| {
						(left.behavior.priority.unwrap_or(u32::MAX), &left.id)
							.cmp(&(right.behavior.priority.unwrap_or(u32::MAX), &right.id))
					})
			})
			.or_else(|| {
				self
					.cards
					.values()
					.filter(|card| {
						card.availability == Availability::Available
							&& card.model.as_str().contains(candidate)
					})
					.min_by(|left, right| {
						(left.behavior.priority.unwrap_or(u32::MAX), &left.id)
							.cmp(&(right.behavior.priority.unwrap_or(u32::MAX), &right.id))
					})
			})
	}

	fn rejoin(&mut self) {
		self.cards = self.joined_cards();
	}

	fn publish_join(&mut self) {
		let joined = self.joined_cards();
		let ids: BTreeSet<Str> = self.cards.keys().chain(joined.keys()).cloned().collect();
		for id in ids {
			match (self.cards.get(&id), joined.get(&id)) {
				(Some(previous), Some(card)) if previous == card => {},
				(_, Some(card)) => {
					self
						.watch
						.emit(|cursor| ModelEvent::Upserted { cursor, card: Box::new(card.clone()) });
				},
				(Some(_), None) => {
					self
						.watch
						.emit(|cursor| ModelEvent::Removed { cursor, id: id.clone() });
				},
				(None, None) => {},
			}
		}
		self.cards = joined;
	}

	fn joined_cards(&self) -> BTreeMap<Str, ModelCard> {
		let authoritative: BTreeSet<&str> = self
			.live_sources
			.iter()
			.filter(|(_, source)| source.authoritative)
			.map(|(key, _)| key.provider.as_str())
			.collect();
		let mut joined: BTreeMap<Str, ModelCard> = self
			.static_cards
			.values()
			.filter(|card| !authoritative.contains(card.provider.as_str()))
			.cloned()
			.map(|card| {
				let card = self.with_local_availability(card);
				(card.id.clone(), card)
			})
			.collect();
		for (key, source) in &self.live_sources {
			for (id, discovered) in &source.cards {
				let mut card = self.static_cards.get(id).map_or_else(
					|| {
						derived_base(&self.static_cards, discovered).map_or_else(
							|| discovered.clone(),
							|base| overlay_derived_card(base, discovered),
						)
					},
					|base| overlay_card(base, discovered),
				);
				card.availability = match source.authority {
					AvailabilityAuthority::LocalCredentials => self
						.credentials
						.availability_for(key.provider.as_str(), key.account.as_str()),
					AvailabilityAuthority::Upstream => discovered.availability,
				};
				match joined.get(id) {
					Some(existing)
						if existing.source == Source::Discovered
							&& availability_rank(existing.availability)
								>= availability_rank(card.availability) => {},
					_ => {
						joined.insert(id.clone(), card);
					},
				}
			}
		}
		joined
	}

	fn with_local_availability(&self, mut card: ModelCard) -> ModelCard {
		card.availability = self.credentials.availability(card.provider.as_str());
		card
	}
}
fn overlay_card(base: &ModelCard, discovered: &ModelCard) -> ModelCard {
	let mut card = base.clone();
	card.name = discovered.name.clone();
	if !discovered.family.is_empty() {
		card.family = discovered.family.clone();
	}
	if !discovered.facets.is_empty() {
		card.facets = discovered.facets.clone();
	}
	if !discovered.inputs.is_empty() {
		card.inputs = discovered.inputs.clone();
	}
	if !discovered.outputs.is_empty() {
		card.outputs = discovered.outputs.clone();
	}
	card.reasoning |= discovered.reasoning;
	if !discovered.efforts.is_empty() {
		card.efforts = discovered.efforts.clone();
	}
	if discovered.context_window != 0 {
		card.context_window = discovered.context_window;
	}
	if discovered.max_output_tokens != 0 {
		card.max_output_tokens = discovered.max_output_tokens;
	}
	if !discovered.pricing.is_empty() {
		for &price in &discovered.pricing {
			if let Some(existing) = card
				.pricing
				.iter_mut()
				.find(|existing| existing.unit == price.unit)
			{
				*existing = price;
			} else {
				card.pricing.push(price);
			}
		}
	}
	card.availability = discovered.availability;
	card.source = Source::Discovered;
	card.blocked_until_ms = discovered.blocked_until_ms;
	card.deprecated = discovered.deprecated;
	card.updated_at_ms = discovered.updated_at_ms;
	if discovered.props != Default::default() {
		card.props = discovered.props.clone();
	}
	if !discovered.effort_routing.is_empty() {
		card.effort_routing = discovered.effort_routing.clone();
	}
	card
}

fn derived_base<'a>(
	static_cards: &'a BTreeMap<Str, ModelCard>,
	discovered: &ModelCard,
) -> Option<&'a ModelCard> {
	let wire_model = discovered.effort_routing.get(&Effort::Off)?;
	let id = fmts!("{}/{}", discovered.provider, wire_model);
	static_cards.get(&id)
}

fn overlay_derived_card(base: &ModelCard, discovered: &ModelCard) -> ModelCard {
	let mut card = overlay_card(base, discovered);
	card.id.clone_from(&discovered.id);
	card.provider.clone_from(&discovered.provider);
	card.model.clone_from(&discovered.model);
	let mut routes = base.effort_routing.clone();
	routes.extend(discovered.effort_routing.clone());
	card.effort_routing = routes;
	card
}

const fn availability_rank(availability: Availability) -> u8 {
	match availability {
		Availability::Available => 5,
		Availability::Blocked => 4,
		Availability::LoginRequired => 3,
		Availability::Unspecified => 2,
		Availability::Disabled => 1,
	}
}

fn unix_now_ms() -> u64 {
	let millis = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis();
	u64::try_from(millis).unwrap_or(u64::MAX)
}

fn mint_epoch() -> Bytes {
	Bytes::copy_from_slice(&Ulid::generate().to_bytes())
}

#[cfg(test)]
mod tests {
	use std::{collections::BTreeMap, sync::Arc};

	use futures::StreamExt;
	use omp_llm_types::Props;
	use smallvec::SmallVec;

	use super::*;
	use crate::models::{Modality, Price, PriceUnit};

	struct Credentials(BTreeMap<Str, Availability>);

	impl CredentialView for Credentials {
		fn availability(&self, provider: &str) -> Availability {
			self
				.0
				.get(provider)
				.copied()
				.unwrap_or(Availability::Available)
		}
	}

	fn registry(cards: &[ModelCard], retention: usize) -> Registry {
		Registry::from_cards_with_retention(cards, Arc::new(Credentials(BTreeMap::new())), retention)
	}

	fn registry_with_credentials(entries: &[(&str, Availability)], retention: usize) -> Registry {
		let credentials = entries
			.iter()
			.map(|(provider, availability)| (Str::from(*provider), *availability))
			.collect();
		Registry::from_cards_with_retention(&[], Arc::new(Credentials(credentials)), retention)
	}

	fn listed_card(registry: &Registry, id: &str) -> ModelCard {
		registry
			.list(&ListFilter::default())
			.0
			.into_iter()
			.find(|card| card.id == id)
			.expect("card must be listed")
	}

	fn card(provider: &str, model: &str) -> ModelCard {
		let mut facets = SmallVec::new();
		facets.push(Facet::Chat);
		let mut inputs = SmallVec::new();
		inputs.push(Modality::Text);
		let mut outputs = SmallVec::new();
		outputs.push(Modality::Text);
		ModelCard {
			id: Str::from(format!("{provider}/{model}")),
			provider: Str::from(provider),
			model: Str::from(model),
			name: Str::from(model),
			family: Str::from(model),
			facets,
			inputs,
			outputs,
			reasoning: false,
			efforts: SmallVec::new(),
			context_window: 0,
			max_output_tokens: 0,
			pricing: SmallVec::new(),
			availability: Availability::Available,
			source: Source::Discovered,
			blocked_until_ms: 0,
			deprecated: false,
			updated_at_ms: 0,
			props: Props::default(),
			effort_routing: BTreeMap::new(),
			behavior: crate::models::ModelBehavior::default(),
			wire: None,
		}
	}

	#[test]
	fn role_candidate_ties_use_catalog_priority_then_canonical_id() {
		let mut low_priority = card("z-provider", "shared-model");
		low_priority.behavior.priority = Some(20);
		let mut preferred = card("z-provider", "shared-model-preview");
		preferred.behavior.priority = Some(3);
		let mut same_priority = card("a-provider", "shared-model-beta");
		same_priority.behavior.priority = Some(3);
		let mut registry = registry(&[low_priority, preferred, same_priority], 16);
		registry.roles.set("test-role", ["shared"]);

		assert_eq!(
			registry
				.resolve_role("test-role")
				.map(|card| card.id.as_str()),
			Some("a-provider/shared-model-beta"),
		);
	}

	#[test]
	fn derived_wire_alias_inherits_static_prices_and_effort_routes() {
		let mut bundled = card("provider", "stable");
		bundled.source = Source::Bundled;
		bundled
			.pricing
			.push(Price { unit: PriceUnit::MtokInput, nanos_usd: 1 });
		bundled
			.pricing
			.push(Price { unit: PriceUnit::MtokCacheWrite, nanos_usd: 9 });
		bundled
			.effort_routing
			.insert(Effort::Medium, Str::from("stable-medium"));
		let mut registry = registry(&[bundled], 16);

		let mut variant = card("provider", "stable-1m");
		variant.context_window = 1_000_000;
		variant
			.pricing
			.push(Price { unit: PriceUnit::MtokInput, nanos_usd: 2 });
		variant
			.effort_routing
			.insert(Effort::Off, Str::from("stable"));
		registry.apply_discovered("provider", vec![variant]);

		let variant = listed_card(&registry, "provider/stable-1m");
		assert_eq!(variant.model, "stable-1m");
		assert_eq!(variant.context_window, 1_000_000);
		assert_eq!(
			variant
				.pricing
				.iter()
				.find(|price| price.unit == PriceUnit::MtokInput)
				.map(|price| price.nanos_usd),
			Some(2)
		);
		assert_eq!(
			variant
				.pricing
				.iter()
				.find(|price| price.unit == PriceUnit::MtokCacheWrite)
				.map(|price| price.nanos_usd),
			Some(9)
		);
		assert_eq!(variant.effort_routing[&Effort::Off], "stable");
		assert_eq!(variant.effort_routing[&Effort::Medium], "stable-medium");
	}

	#[test]
	fn federated_availability_is_preserved_in_both_directions() {
		let mut registry = registry_with_credentials(
			&[
				("upstream-login", Availability::Available),
				("upstream-ready", Availability::LoginRequired),
			],
			8,
		);
		let mut login_required = card("ignored", "login");
		login_required.availability = Availability::LoginRequired;
		registry.apply_federated("upstream-login", vec![login_required]);
		let mut available = card("ignored", "ready");
		available.availability = Availability::Available;
		registry.apply_federated("upstream-ready", vec![available]);

		assert_eq!(
			listed_card(&registry, "upstream-login/login").availability,
			Availability::LoginRequired
		);
		assert_eq!(
			listed_card(&registry, "upstream-ready/ready").availability,
			Availability::Available
		);
		assert_eq!(listed_card(&registry, "upstream-login/login").source, Source::Discovered);

		registry.rebuild();
		assert_eq!(
			listed_card(&registry, "upstream-login/login").availability,
			Availability::LoginRequired
		);
		assert_eq!(
			listed_card(&registry, "upstream-ready/ready").availability,
			Availability::Available
		);
	}

	#[test]
	fn discovered_availability_comes_from_local_credentials() {
		let mut registry = registry_with_credentials(&[("local", Availability::LoginRequired)], 8);
		let mut discovered = card("ignored", "model");
		discovered.availability = Availability::Available;
		registry.apply_discovered("local", vec![discovered]);

		let listed = listed_card(&registry, "local/model");
		assert_eq!(listed.availability, Availability::LoginRequired);
		assert_eq!(listed.source, Source::Discovered);
	}

	#[tokio::test]
	async fn both_paths_bump_generation_and_emit_events() {
		let mut registry = registry(&[], 8);
		let (_, start) = registry.list(&ListFilter::default());
		registry.apply_discovered("local", vec![card("local", "one")]);
		registry.apply_federated("remote", vec![card("remote", "two")]);

		let (_, end) = registry.list(&ListFilter::default());
		assert_eq!(end.epoch, start.epoch);
		assert_eq!(end.generation, start.generation + 2);

		let mut watch = registry.watch(Some(start));
		let local = watch.next().await.expect("local discovery event");
		let federated = watch.next().await.expect("federated discovery event");
		assert_eq!(local.cursor().generation + 1, federated.cursor().generation);
		assert!(matches!(local, ModelEvent::Upserted { card, .. } if card.id == "local/one"));
		assert!(matches!(federated, ModelEvent::Upserted { card, .. } if card.id == "remote/two"));
	}

	#[tokio::test]
	async fn watcher_has_consistent_sequence_across_local_and_federated_updates() {
		let mut registry = registry(&[], 8);
		let (_, start) = registry.list(&ListFilter::default());
		let mut watch = registry.watch(Some(start.clone()));

		registry.apply_discovered("local", vec![card("local", "one")]);
		registry.apply_federated("remote", vec![card("remote", "two")]);
		registry.apply_discovered("local", Vec::new());
		let mut blocked = card("remote", "two");
		blocked.availability = Availability::Blocked;
		registry.apply_federated("remote", vec![blocked]);

		let mut events = Vec::new();
		for _ in 0..4 {
			events.push(watch.next().await.expect("contiguous delta"));
		}
		let generations = events
			.iter()
			.map(|event| event.cursor().generation)
			.collect::<Vec<_>>();
		assert_eq!(generations, (start.generation + 1..=start.generation + 4).collect::<Vec<_>>());
		assert!(matches!(&events[0], ModelEvent::Upserted { card, .. } if card.id == "local/one"));
		assert!(matches!(&events[1], ModelEvent::Upserted { card, .. } if card.id == "remote/two"));
		assert!(matches!(&events[2], ModelEvent::Removed { id, .. } if id == "local/one"));
		assert!(matches!(&events[3], ModelEvent::Upserted { card, .. }
				if card.id == "remote/two" && card.availability == Availability::Blocked));
	}

	#[tokio::test]
	async fn live_cursor_replays_only_newer_deltas() {
		let mut registry = registry(&[], 8);
		registry.apply_discovered("local", vec![card("local", "one")]);
		let (_, cursor) = registry.list(&ListFilter::default());
		registry.apply_discovered("local", vec![card("local", "one"), card("local", "two")]);

		let mut watch = registry.watch(Some(cursor));
		let event = watch.next().await.expect("new delta");
		assert!(matches!(event, ModelEvent::Upserted { ref card, .. } if card.model == "two"));
	}

	#[tokio::test]
	async fn stale_epoch_resets_first() {
		let registry = registry(&[], 8);
		let stale = Cursor { epoch: Bytes::from_static(b"dead epoch"), generation: 0 };
		let mut watch = registry.watch(Some(stale));
		assert!(matches!(watch.next().await, Some(ModelEvent::Reset { .. })));
	}

	#[tokio::test]
	async fn cursor_beyond_retained_window_resets_first() {
		let mut registry = registry(&[], 2);
		let (_, old) = registry.list(&ListFilter::default());
		registry.apply_discovered("one", vec![card("one", "a")]);
		registry.apply_discovered("two", vec![card("two", "b")]);
		registry.apply_discovered("three", vec![card("three", "c")]);

		let mut watch = registry.watch(Some(old));
		assert!(matches!(watch.next().await, Some(ModelEvent::Reset { .. })));
	}

	#[tokio::test]
	async fn relist_after_reset_drops_old_deltas_without_gap() {
		let mut registry = registry(&[], 8);
		let stale = Cursor { epoch: Bytes::from_static(b"dead epoch"), generation: 99 };
		let mut watch = registry.watch(Some(stale));
		registry.apply_discovered("local", vec![card("local", "before-list")]);
		let reset = watch.next().await.expect("reset");
		assert!(matches!(reset, ModelEvent::Reset { .. }));
		let (listed, list_cursor) = registry.list(&ListFilter::default());
		assert_eq!(listed.len(), 1);
		registry.apply_discovered("local", vec![
			card("local", "before-list"),
			card("local", "after-list"),
		]);

		let mut applied = Vec::new();
		while applied.is_empty() {
			let event = watch.next().await.expect("queued delta");
			if event.cursor().epoch == list_cursor.epoch
				&& event.cursor().generation > list_cursor.generation
				&& let ModelEvent::Upserted { card, .. } = event
			{
				applied.push(card.model);
			}
		}
		assert_eq!(applied, [Str::from("after-list")]);
	}

	#[test]
	fn authoritative_account_snapshot_overlays_then_expires_to_static_fallback() {
		let mut bundled = card("provider", "stable");
		bundled.name = "Bundled name".into();
		bundled.context_window = 128_000;
		let mut registry = registry(&[bundled], 16);
		let mut live = card("provider", "live");
		live.name = "Live name".into();
		registry.apply_discovered_account("provider", "account-1", vec![live], true, 100, 50);

		let (models, _) = registry.list(&ListFilter::default());
		assert_eq!(
			models
				.iter()
				.map(|card| card.id.as_str())
				.collect::<Vec<_>>(),
			["provider/live"]
		);
		assert!(!registry.expire_stale(149));
		assert!(registry.expire_stale(150));
		let fallback = listed_card(&registry, "provider/stable");
		assert_eq!(fallback.name, "Bundled name");
		assert_eq!(fallback.context_window, 128_000);
	}

	#[test]
	fn account_sources_merge_without_duplicate_models_and_prefer_available() {
		struct Accounts;
		impl CredentialView for Accounts {
			fn availability(&self, _provider: &str) -> Availability {
				Availability::LoginRequired
			}

			fn availability_for(&self, _provider: &str, account: &str) -> Availability {
				if account == "healthy" {
					Availability::Available
				} else {
					Availability::Blocked
				}
			}
		}

		let mut registry = Registry::from_cards(&[], Arc::new(Accounts));
		registry.apply_discovered_account(
			"provider",
			"blocked",
			vec![card("provider", "shared"), card("provider", "blocked-only")],
			false,
			0,
			100,
		);
		registry.apply_discovered_account(
			"provider",
			"healthy",
			vec![card("provider", "shared"), card("provider", "healthy-only")],
			false,
			0,
			100,
		);

		let (models, _) = registry.list(&ListFilter::default());
		assert_eq!(models.len(), 3);
		assert_eq!(listed_card(&registry, "provider/shared").availability, Availability::Available);
	}

	#[test]
	fn role_resolution_falls_through_unavailable_candidate() {
		let cards = [card("first", "fast"), card("second", "steady")];
		let credentials = Credentials(BTreeMap::from([
			(Str::from("first"), Availability::LoginRequired),
			(Str::from("second"), Availability::Available),
		]));
		let mut registry = Registry::from_cards(&cards, Arc::new(credentials));
		let mut roles = RoleConfig::empty();
		roles.set("smol", ["first/fast", "second/steady"]);
		registry.set_roles(roles);

		assert_eq!(registry.resolve_role("smol").map(|card| card.id.as_str()), Some("second/steady"));
		assert_eq!(registry.role_bindings().get("smol").map(Str::as_str), Some("second/steady"));
	}
}
