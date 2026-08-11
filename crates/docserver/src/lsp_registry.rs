//! Project-scoped LSP binding selection, document lifecycle, revision
//! admission, inbound revision tagging, and transaction formatting
//! coordination.

use std::{
	collections::{HashMap, HashSet, VecDeque},
	future::Future,
	sync::Arc,
};

use bytes::Bytes;
use globset::{Glob, GlobMatcher};
use omp_core::Str;
use parking_lot::Mutex;
use serde_json::Value;
use thiserror::Error;
use tokio::sync::{Mutex as AsyncMutex, broadcast};
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::{
	DocumentEvent, DocumentHead, DocumentId, DocumentKind, DocumentPresence, DocumentSnapshot,
	DocumentStore, LanguageId, LeaseId, ReadBody, ReadSelection, Revision, TransactionId,
	lsp::{LspDocument, LspError, LspResponse, LspServer, LspTransportError, SyncPolicy},
	transaction::{
		FormatCoordinator, FormatRequest, FormatResult, PublishedDocument, RevertedDocument,
	},
};
const PUBLIC_VERSION_LIMIT: usize = 32;
const LSP_EVENT_BUS_CAPACITY: usize = 256;
const DOCUMENT_EVENT_FORWARD_CAPACITY: usize = 64;

/// Stable process-local identity assigned to an LSP binding.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LspBindingId(u64);

impl LspBindingId {
	/// Reconstructs a binding identity from its registry-local integer
	/// representation.
	#[must_use]
	pub const fn from_u64(value: u64) -> Self {
		Self(value)
	}

	/// Returns the registry-local integer representation.
	#[must_use]
	pub const fn get(self) -> u64 {
		self.0
	}
}

/// Generation-bound identity for callbacks originating from one concrete LSP
/// server lane.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LspBindingHandle {
	binding_id: LspBindingId,
	generation: u64,
}

impl LspBindingHandle {
	/// Returns the stable binding identity.
	#[must_use]
	pub const fn binding_id(self) -> LspBindingId {
		self.binding_id
	}
}

/// A compiled language, URI-scheme, and URI-path binding selector.
#[derive(Clone)]
pub struct LspSelector {
	languages:     Vec<LanguageId>,
	schemes:       Vec<Str>,
	path_patterns: Vec<Str>,
	path_matchers: Vec<GlobMatcher>,
}

impl std::fmt::Debug for LspSelector {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("LspSelector")
			.field("languages", &self.languages)
			.field("schemes", &self.schemes)
			.field("path_patterns", &self.path_patterns)
			.finish()
	}
}

impl LspSelector {
	/// Compiles a selector. An empty dimension matches every value in that
	/// dimension.
	pub fn new(
		languages: Vec<LanguageId>,
		schemes: Vec<Str>,
		path_patterns: Vec<Str>,
	) -> Result<Self, LspRegistryError> {
		let mut path_matchers = Vec::with_capacity(path_patterns.len());
		for pattern in &path_patterns {
			let matcher = Glob::new(pattern.as_str())
				.map_err(|error| LspRegistryError::InvalidSelector {
					reason: Str::new(error.to_string()),
				})?
				.compile_matcher();
			path_matchers.push(matcher);
		}
		Ok(Self { languages, schemes, path_patterns, path_matchers })
	}

	/// Creates a selector matching every document.
	#[must_use]
	pub const fn all() -> Self {
		Self {
			languages:     Vec::new(),
			schemes:       Vec::new(),
			path_patterns: Vec::new(),
			path_matchers: Vec::new(),
		}
	}

	/// Returns the language restrictions in declaration order.
	#[must_use]
	pub fn languages(&self) -> &[LanguageId] {
		&self.languages
	}

	/// Returns the URI-scheme restrictions in declaration order.
	#[must_use]
	pub fn schemes(&self) -> &[Str] {
		&self.schemes
	}

	/// Returns the path glob restrictions in declaration order.
	#[must_use]
	pub fn path_patterns(&self) -> &[Str] {
		&self.path_patterns
	}

	/// Reports whether this selector accepts a URI and language classification.
	#[must_use]
	pub fn matches(&self, uri: &Url, language: Option<&LanguageId>) -> bool {
		let language_matches = self.languages.is_empty()
			|| language.is_some_and(|language| self.languages.iter().any(|item| item == language));
		let scheme_matches = self.schemes.is_empty()
			|| self
				.schemes
				.iter()
				.any(|scheme| scheme.as_str() == uri.scheme());
		let path_matches = self.path_matchers.is_empty()
			|| self
				.path_matchers
				.iter()
				.any(|matcher| matcher.is_match(uri.path()));
		language_matches && scheme_matches && path_matches
	}
}

/// Declaration used when installing a named server binding.
#[derive(Clone, Debug)]
pub struct LspBindingSpec {
	name:     Str,
	priority: i32,
	selector: LspSelector,
}

impl LspBindingSpec {
	/// Creates a binding declaration.
	pub fn new(
		name: impl AsRef<str>,
		priority: i32,
		selector: LspSelector,
	) -> Result<Self, LspRegistryError> {
		let name = name.as_ref();
		if name.is_empty() {
			return Err(LspRegistryError::InvalidBindingName);
		}
		Ok(Self { name: Str::new(name), priority, selector })
	}

	/// Returns the unique binding name.
	#[must_use]
	pub fn name(&self) -> &str {
		self.name.as_str()
	}

	/// Returns the deterministic selection priority. Higher values run first.
	#[must_use]
	pub const fn priority(&self) -> i32 {
		self.priority
	}

	/// Returns the binding selector.
	#[must_use]
	pub const fn selector(&self) -> &LspSelector {
		&self.selector
	}
}

/// Immutable public description of an installed binding.
#[derive(Clone, Debug)]
pub struct LspBindingInfo {
	id:   LspBindingId,
	spec: LspBindingSpec,
}

impl LspBindingInfo {
	/// Returns the binding identity.
	#[must_use]
	pub const fn id(&self) -> LspBindingId {
		self.id
	}

	/// Returns the binding declaration.
	#[must_use]
	pub const fn spec(&self) -> &LspBindingSpec {
		&self.spec
	}
}

/// Current synchronization policy and capabilities for a selected lease
/// binding.
#[derive(Clone, Debug)]
pub struct LspLeaseBinding {
	info:              LspBindingInfo,
	sync_policy:       SyncPolicy,
	capabilities_json: Bytes,
}

impl LspLeaseBinding {
	/// Returns the installed binding description.
	#[must_use]
	pub const fn info(&self) -> &LspBindingInfo {
		&self.info
	}

	/// Returns the selector-resolved synchronization policy.
	#[must_use]
	pub const fn sync_policy(&self) -> &SyncPolicy {
		&self.sync_policy
	}

	/// Returns exact `InitializeResult` capability JSON.
	#[must_use]
	pub const fn capabilities_json(&self) -> &Bytes {
		&self.capabilities_json
	}
}

/// Requested stale-response policy for semantic operations.
///
/// Semantic parameters are opaque raw JSON, so both policies reject a stale
/// admission or completion rather than replaying position-bearing parameters
/// against different text.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum StaleResponsePolicy {
	/// Reject admission or a response whenever the requested head is no longer
	/// current.
	#[default]
	ContentModified,
	/// Retained for protocol compatibility; opaque parameters are not retried.
	RetryOnce,
}

/// An inbound server event tagged with a provable daemon revision when
/// available.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaggedLspEvent {
	binding_id:        LspBindingId,
	binding_name:      Str,
	method:            Str,
	params_json:       Bytes,
	revision:          Option<Revision>,
	document_identity: Option<(DocumentId, Url)>,
}

impl TaggedLspEvent {
	/// Returns the server binding that emitted the event.
	#[must_use]
	pub const fn binding_id(&self) -> LspBindingId {
		self.binding_id
	}

	/// Returns the server binding name captured with the event.
	#[must_use]
	pub fn binding_name(&self) -> &str {
		self.binding_name.as_str()
	}

	/// Returns the inbound LSP method.
	#[must_use]
	pub fn method(&self) -> &str {
		self.method.as_str()
	}

	/// Returns the exact inbound JSON parameters.
	#[must_use]
	pub const fn params_json(&self) -> &Bytes {
		&self.params_json
	}

	/// Returns the daemon revision proven by a URI/version pair, if any.
	#[must_use]
	pub const fn revision(&self) -> Option<Revision> {
		self.revision
	}

	/// Returns the document identity proven by the binding's public version
	/// history, if the notification names one unambiguously.
	#[must_use]
	pub const fn document_identity(&self) -> Option<&(DocumentId, Url)> {
		self.document_identity.as_ref()
	}

	/// Returns the document identity proven for this notification, if any.
	#[must_use]
	pub fn document_id(&self) -> Option<DocumentId> {
		self
			.document_identity
			.as_ref()
			.map(|(document_id, _)| *document_id)
	}

	/// Returns the document URI proven for this notification, if any.
	#[must_use]
	pub fn document_uri(&self) -> Option<&Url> {
		self.document_identity.as_ref().map(|(_, uri)| uri)
	}
}

/// A registry event that connection-local subscribers can forward without
/// translating native LSP or document identities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LspRegistryEvent {
	/// An inbound server notification with the exact parameters received.
	Inbound(Box<TaggedLspEvent>),
	/// A binding lifecycle or synchronization-policy change.
	Binding(LspBindingEvent),
}

/// The kind of an installed binding change.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LspBindingEventKind {
	/// The binding has been installed and is ready for requests.
	Ready,
	/// Dynamic registration changed the binding's policy for an open document.
	PolicyChanged,
	/// The binding's server lane has been replaced successfully.
	Restarted,
	/// The binding has been removed after its documents were released.
	Stopped,
}

/// A binding lifecycle or policy event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LspBindingEvent {
	binding_id:  LspBindingId,
	document_id: Option<DocumentId>,
	kind:        LspBindingEventKind,
}

impl LspBindingEvent {
	/// Returns the binding affected by this change.
	#[must_use]
	pub const fn binding_id(&self) -> LspBindingId {
		self.binding_id
	}

	/// Returns the affected open document for document-scoped policy changes.
	#[must_use]
	pub const fn document_id(&self) -> Option<DocumentId> {
		self.document_id
	}

	/// Returns the lifecycle or policy transition.
	#[must_use]
	pub const fn kind(&self) -> LspBindingEventKind {
		self.kind
	}
}

/// Terminal failure for a lease's bounded committed-document event stream.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DocumentEventStreamError {
	/// The actor produced more events than the registry could synchronize.
	#[error("document event stream lagged by {skipped} events")]
	Lagged {
		/// Number of overwritten events.
		skipped: u64,
	},
	/// An event could not be synchronized to every selected LSP binding.
	#[error("document event synchronization failed: {message}")]
	Synchronization {
		/// Registry or LSP failure.
		message: Str,
	},
	/// The document actor stopped while the lease remained open.
	#[error("document event stream closed unexpectedly")]
	Closed,
}

/// An active document lease owned by the registry and its initial committed
/// head.
#[derive(Debug)]
pub struct LspDocumentLease {
	lease_id:    LeaseId,
	head:        DocumentHead,
	binding_ids: Vec<LspBindingId>,
	events:      flume::Receiver<Result<DocumentEvent, DocumentEventStreamError>>,
}

impl LspDocumentLease {
	/// Returns the underlying document-store lease identity.
	#[must_use]
	pub const fn lease_id(&self) -> LeaseId {
		self.lease_id
	}

	/// Returns the committed head admitted by the open operation.
	#[must_use]
	pub const fn head(&self) -> &DocumentHead {
		&self.head
	}

	/// Returns selected bindings in deterministic priority order.
	#[must_use]
	pub fn binding_ids(&self) -> &[LspBindingId] {
		&self.binding_ids
	}

	/// Returns this lease's ordered committed-document event stream.
	#[must_use]
	pub const fn events(&self) -> &flume::Receiver<Result<DocumentEvent, DocumentEventStreamError>> {
		&self.events
	}

	/// Splits the lease into its identity, initial head, selected bindings, and
	/// event stream.
	#[must_use]
	pub fn into_parts(
		self,
	) -> (
		LeaseId,
		DocumentHead,
		Vec<LspBindingId>,
		flume::Receiver<Result<DocumentEvent, DocumentEventStreamError>>,
	) {
		(self.lease_id, self.head, self.binding_ids, self.events)
	}
}

#[derive(Clone)]
struct Binding {
	id:         LspBindingId,
	spec:       LspBindingSpec,
	server:     LspServer,
	generation: u64,
}

struct FormatBindingLease {
	binding:  Binding,
	existing: usize,
}

struct FormatLeaseSet {
	bindings:         Vec<FormatBindingLease>,
	base_uri:         Url,
	base_language_id: Option<LanguageId>,
}

struct RefreshProgress {
	binding:          Binding,
	original_count:   usize,
	opened:           bool,
	retained:         usize,
	released:         usize,
	current_language: Option<LanguageId>,
}

#[derive(Clone)]
struct LeaseRecord {
	document_id:   DocumentId,
	language_id:   Option<LanguageId>,
	binding_ids:   Vec<LspBindingId>,
	cancel_events: CancellationToken,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ProvisionalLeaseKey {
	binding_id:  LspBindingId,
	document_id: DocumentId,
	transaction: TransactionId,
}

#[derive(Default)]
struct RegistryState {
	next_binding_id:    u64,
	bindings:           HashMap<LspBindingId, Binding>,
	binding_names:      HashMap<Str, LspBindingId>,
	leases:             HashMap<LeaseId, LeaseRecord>,
	public_versions:    HashMap<(LspBindingId, DocumentId), VecDeque<(Str, i32)>>,
	provisional_leases: HashSet<ProvisionalLeaseKey>,
	publication_gates:  HashMap<TransactionId, CancellationToken>,
}

struct RegistryInner {
	store:    DocumentStore,
	events:   broadcast::Sender<LspRegistryEvent>,
	mutation: AsyncMutex<()>,
	state:    Mutex<RegistryState>,
}

/// Project-scoped owner of selected, ordered LSP server bindings.
#[derive(Clone)]
pub struct LspRegistry {
	inner: Arc<RegistryInner>,
}

/// Releases actor-event publication for one committed inbound LSP transaction.
pub(crate) struct LspPublicationBarrier {
	registry:       LspRegistry,
	transaction_id: TransactionId,
}

impl LspPublicationBarrier {
	/// Releases every document event blocked on this transaction.
	pub(crate) fn release(self) {
		drop(self);
	}
}

impl Drop for LspPublicationBarrier {
	fn drop(&mut self) {
		let gate = {
			self
				.registry
				.inner
				.state
				.lock()
				.publication_gates
				.remove(&self.transaction_id)
		};
		if let Some(gate) = gate {
			gate.cancel();
		}
	}
}

impl std::fmt::Debug for LspRegistry {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("LspRegistry")
			.finish_non_exhaustive()
	}
}

impl LspRegistry {
	/// Creates an empty registry above a document store.
	#[must_use]
	pub fn new(store: DocumentStore) -> Self {
		Self {
			inner: Arc::new(RegistryInner {
				store,
				events: broadcast::channel(LSP_EVENT_BUS_CAPACITY).0,
				mutation: AsyncMutex::new(()),
				state: Mutex::new(RegistryState { next_binding_id: 1, ..RegistryState::default() }),
			}),
		}
	}

	/// Returns the project document store used for revision admission.
	#[must_use]
	pub fn document_store(&self) -> &DocumentStore {
		&self.inner.store
	}

	/// Subscribes to the bounded registry event stream.
	///
	/// Receivers observe [`broadcast::error::RecvError::Lagged`] when they fall
	/// behind instead of silently losing notifications.
	#[must_use]
	pub fn subscribe_events(&self) -> broadcast::Receiver<LspRegistryEvent> {
		self.inner.events.subscribe()
	}

	/// Defers actor-event publication until an inbound LSP response is written.
	pub(crate) fn defer_transaction_publication(
		&self,
		transaction_id: TransactionId,
	) -> LspPublicationBarrier {
		let displaced = self
			.inner
			.state
			.lock()
			.publication_gates
			.insert(transaction_id, CancellationToken::new());
		assert!(displaced.is_none(), "transaction publication barrier is unique");
		LspPublicationBarrier { registry: self.clone(), transaction_id }
	}

	async fn await_transaction_publication(&self, transaction_id: Option<TransactionId>) {
		let gate = transaction_id.and_then(|transaction_id| {
			self
				.inner
				.state
				.lock()
				.publication_gates
				.get(&transaction_id)
				.cloned()
		});
		if let Some(gate) = gate {
			gate.cancelled().await;
		}
	}

	/// Installs a named server and synchronizes every already-open matching
	/// document.
	pub async fn add_binding(
		&self,
		spec: LspBindingSpec,
		server: LspServer,
		cancel: CancellationToken,
	) -> Result<LspBindingId, LspRegistryError> {
		let _mutation = self.inner.mutation.lock().await;
		{
			let state = self.inner.state.lock();
			if state.binding_names.contains_key(spec.name.as_str()) {
				return Err(LspRegistryError::DuplicateBinding { name: spec.name.clone() });
			}
		}
		let lease_records = self.lease_records().await;
		let mut documents =
			HashMap::<DocumentId, (Arc<DocumentSnapshot>, Url, Option<LanguageId>, usize)>::new();
		for (_, record) in &lease_records {
			let (snapshot, uri) = self.current_snapshot(record.document_id).await?;
			if !spec.selector.matches(&uri, record.language_id.as_ref()) {
				continue;
			}
			documents
				.entry(record.document_id)
				.and_modify(|entry| entry.3 += 1)
				.or_insert_with(|| (snapshot, uri, record.language_id.clone(), 1));
		}
		let mut installed = Vec::new();
		for (document_id, (snapshot, uri, language_id, count)) in &documents {
			if let Err(error) = server
				.synchronize(lsp_document(snapshot, uri, language_id.as_ref()), cancel.child_token())
				.await
			{
				for installed_id in installed {
					let _ = server
						.release_document(installed_id, CancellationToken::new())
						.await;
				}
				return Err(error.into());
			}
			installed.push(*document_id);
			for _ in 1..*count {
				if let Err(error) = server.retain_document(*document_id).await {
					for installed_id in installed {
						let _ = server
							.release_document(installed_id, CancellationToken::new())
							.await;
					}
					return Err(error.into());
				}
				installed.push(*document_id);
			}
		}
		let id = {
			let mut state = self.inner.state.lock();
			let id = LspBindingId(state.next_binding_id);
			state.next_binding_id = state
				.next_binding_id
				.checked_add(1)
				.ok_or(LspRegistryError::BindingIdOverflow)?;
			state.binding_names.insert(spec.name.clone(), id);
			state
				.bindings
				.insert(id, Binding { id, spec, server: server.clone(), generation: 0 });
			for (lease_id, record) in &lease_records {
				if documents.contains_key(&record.document_id) {
					state
						.leases
						.get_mut(lease_id)
						.expect("lease captured under mutation gate")
						.binding_ids
						.push(id);
				}
			}
			id
		};
		for (document_id, (_, uri, ..)) in &documents {
			if let Some((version, _)) = server.tracked_version_revision(*document_id) {
				self
					.mark_public_version(id, *document_id, uri, version)
					.await;
			}
		}
		self.publish_binding_event(id, None, LspBindingEventKind::Ready);
		Ok(id)
	}

	/// Removes a binding after balancing all document leases and advertised
	/// closes.
	pub async fn remove_binding(
		&self,
		binding_id: LspBindingId,
		cancel: CancellationToken,
	) -> Result<(), LspRegistryError> {
		let _mutation = self.inner.mutation.lock().await;
		if self
			.inner
			.state
			.lock()
			.provisional_leases
			.iter()
			.any(|lease| lease.binding_id == binding_id)
		{
			return Err(LspRegistryError::BindingBusy { binding_id });
		}
		let binding = self.binding(binding_id).await?;
		loop {
			let selected = self
				.inner
				.state
				.lock()
				.leases
				.iter()
				.find_map(|(lease_id, record)| {
					record
						.binding_ids
						.contains(&binding_id)
						.then_some((*lease_id, record.document_id))
				});
			let Some((lease_id, document_id)) = selected else {
				break;
			};
			binding
				.server
				.release_document(document_id, cancel.child_token())
				.await?;
			self
				.inner
				.state
				.lock()
				.leases
				.get_mut(&lease_id)
				.expect("lease retained under mutation gate")
				.binding_ids
				.retain(|id| *id != binding_id);
		}
		{
			let mut state = self.inner.state.lock();
			state.bindings.remove(&binding_id);
			state.binding_names.remove(binding.spec.name.as_str());
			state.public_versions.retain(|(id, _), _| *id != binding_id);
		}
		self.publish_binding_event(binding_id, None, LspBindingEventKind::Stopped);
		Ok(())
	}

	/// Returns installed bindings in deterministic selection order.
	pub async fn bindings(&self) -> Vec<LspBindingInfo> {
		let mut bindings = self
			.inner
			.state
			.lock()
			.bindings
			.values()
			.map(|binding| LspBindingInfo { id: binding.id, spec: binding.spec.clone() })
			.collect::<Vec<_>>();
		bindings.sort_by(binding_info_order);
		bindings
	}

	/// Resolves a binding identity by its unique name.
	pub async fn binding_id(&self, name: &str) -> Option<LspBindingId> {
		self.inner.state.lock().binding_names.get(name).copied()
	}

	/// Captures a generation-bound handle for callbacks installed on the
	/// binding's current server lane.
	pub async fn binding_handle(
		&self,
		binding_id: LspBindingId,
	) -> Result<LspBindingHandle, LspRegistryError> {
		let binding = self.binding(binding_id).await?;
		Ok(LspBindingHandle { binding_id, generation: binding.generation })
	}

	/// Resolves synchronization policy for the concrete server generation that
	/// originated an inbound request.
	pub async fn sync_policy_for_handle(
		&self,
		handle: LspBindingHandle,
		uri: &Url,
		language_id: Option<&LanguageId>,
	) -> Result<SyncPolicy, LspRegistryError> {
		let binding = self.binding_for_handle(handle).await?;
		Ok(binding.server.sync_policy(uri, language_id))
	}

	/// Resolves one server-visible text document version to its daemon revision.
	pub async fn revision_for_version(
		&self,
		handle: LspBindingHandle,
		uri: &Url,
		version: i32,
	) -> Result<Option<Revision>, LspRegistryError> {
		let binding = self.binding_for_handle(handle).await?;
		Ok(binding.server.revision_for_version(uri, version))
	}

	/// Returns current policy and capabilities for bindings selected by one
	/// lease.
	pub async fn lease_bindings(
		&self,
		lease_id: LeaseId,
	) -> Result<Vec<LspLeaseBinding>, LspRegistryError> {
		let (document_id, language_id, bindings) = {
			let state = self.inner.state.lock();
			let record = state
				.leases
				.get(&lease_id)
				.ok_or(LspRegistryError::UnknownLease { lease_id })?;
			let bindings = record
				.binding_ids
				.iter()
				.filter_map(|id| state.bindings.get(id).cloned())
				.collect::<Vec<_>>();
			(record.document_id, record.language_id.clone(), bindings)
		};
		let uri = self.document_uri(document_id).await?;
		let mut selected = Vec::with_capacity(bindings.len());
		for binding in bindings {
			selected.push(LspLeaseBinding {
				info:              LspBindingInfo { id: binding.id, spec: binding.spec },
				sync_policy:       binding.server.sync_policy(&uri, language_id.as_ref()),
				capabilities_json: binding.server.capabilities_json(),
			});
		}
		selected.sort_by(|left, right| binding_info_order(&left.info, &right.info));
		Ok(selected)
	}

	/// Opens a store lease, selects matching bindings, and begins automatic head
	/// publication.
	pub async fn open_document(
		&self,
		locator: impl Into<crate::DocumentLocator>,
		language_id: Option<LanguageId>,
		cancel: CancellationToken,
	) -> Result<LspDocumentLease, LspRegistryError> {
		let _mutation = self.inner.mutation.lock().await;
		let opened = self.inner.store.open(locator).await?;
		let (lease_id, head, mut events) = opened.into_parts();
		let snapshot = match self.snapshot_from_store(lease_id, None).await {
			Ok(snapshot) => snapshot,
			Err(error) => {
				let _ = self.inner.store.close(lease_id).await;
				return Err(error);
			},
		};
		let uri = match self.document_uri(head.document_id()).await {
			Ok(uri) => uri,
			Err(error) => {
				let _ = self.inner.store.close(lease_id).await;
				return Err(error);
			},
		};
		let bindings = self.matching_bindings(&uri, language_id.as_ref()).await;
		let mut installed: Vec<Binding> = Vec::new();
		for binding in &bindings {
			let existing = self
				.binding_document_count(binding.id, head.document_id())
				.await;
			let mut acquired = false;
			let result = if existing == 0 {
				let result = binding
					.server
					.synchronize(
						lsp_document(&snapshot, &uri, language_id.as_ref()),
						cancel.child_token(),
					)
					.await;
				acquired = result.is_ok();
				result
			} else {
				match binding.server.retain_document(head.document_id()).await {
					Ok(()) => {
						acquired = true;
						binding
							.server
							.synchronize(
								lsp_document(&snapshot, &uri, language_id.as_ref()),
								cancel.child_token(),
							)
							.await
					},
					Err(error) => Err(error),
				}
			};
			let version = match result {
				Ok(version) => version,
				Err(error) => {
					if acquired {
						let _ = binding
							.server
							.release_document(head.document_id(), CancellationToken::new())
							.await;
					}
					for installed_binding in installed {
						let _ = installed_binding
							.server
							.release_document(head.document_id(), CancellationToken::new())
							.await;
					}
					let _ = self.inner.store.close(lease_id).await;
					return Err(error.into());
				},
			};
			self
				.mark_public_version(binding.id, head.document_id(), &uri, version)
				.await;
			installed.push(binding.clone());
		}
		let binding_ids = bindings
			.iter()
			.map(|binding| binding.id)
			.collect::<Vec<_>>();
		let cancel_events = CancellationToken::new();
		self
			.inner
			.state
			.lock()
			.leases
			.insert(lease_id, LeaseRecord {
				document_id: head.document_id(),
				language_id,
				binding_ids: binding_ids.clone(),
				cancel_events: cancel_events.clone(),
			});
		let registry = self.clone();
		let (client_events_sender, client_events) = flume::bounded(DOCUMENT_EVENT_FORWARD_CAPACITY);
		tokio::spawn(async move {
			loop {
				tokio::select! {
					() = cancel_events.cancelled() => break,
					event = events.recv() => match event {
						Ok(event) => {
							tokio::select! {
								() = cancel_events.cancelled() => break,
								() = registry.await_transaction_publication(event.transaction_id()) => {},
							}
							let document_id = event.head().document_id();
							if let Err(error) =
								registry.publish_head(document_id, CancellationToken::new()).await
							{
								let _ = client_events_sender
									.send_async(Err(DocumentEventStreamError::Synchronization {
										message: Str::new(error.to_string()),
									}))
									.await;
								break;
							}
							if client_events_sender.send_async(Ok(event)).await.is_err() {
								break;
							}
						},
						Err(broadcast::error::RecvError::Lagged(skipped)) => {
							let _ = client_events_sender
								.send_async(Err(DocumentEventStreamError::Lagged { skipped }))
								.await;
							break;
						},
						Err(broadcast::error::RecvError::Closed) => {
							let _ = client_events_sender
								.send_async(Err(DocumentEventStreamError::Closed))
								.await;
							break;
						},
					},
				}
			}
		});
		Ok(LspDocumentLease { lease_id, head, binding_ids, events: client_events })
	}

	/// Releases a registry lease and balances every selected server lease.
	pub async fn close_document(
		&self,
		lease_id: LeaseId,
		cancel: CancellationToken,
	) -> Result<(), LspRegistryError> {
		let _mutation = self.inner.mutation.lock().await;
		let record = self
			.inner
			.state
			.lock()
			.leases
			.get(&lease_id)
			.cloned()
			.ok_or(LspRegistryError::UnknownLease { lease_id })?;
		record.cancel_events.cancel();

		let mut first_error = None;
		for binding_id in &record.binding_ids {
			match self.binding(*binding_id).await {
				Ok(binding) => {
					let result = {
						let release = binding
							.server
							.release_document(record.document_id, cancel.child_token());
						tokio::pin!(release);
						tokio::select! {
							biased;
							result = &mut release => result,
							() = cancel.cancelled() => Err(LspTransportError::Cancelled.into()),
						}
					};
					if let Err(error) = result {
						binding
							.server
							.abandon_document_lease(record.document_id)
							.await;
						if first_error.is_none() {
							first_error = Some(LspRegistryError::from(error));
						}
					}
				},
				Err(error) if first_error.is_none() => first_error = Some(error),
				Err(_) => {},
			}
		}
		if let Err(error) = self.inner.store.close(lease_id).await
			&& first_error.is_none()
		{
			first_error = Some(LspRegistryError::from(error));
		}
		let mut state = self.inner.state.lock();
		state.leases.remove(&lease_id);
		if !state
			.leases
			.values()
			.any(|lease| lease.document_id == record.document_id)
		{
			state
				.public_versions
				.retain(|(_, document_id), _| *document_id != record.document_id);
		}
		drop(state);
		match first_error {
			Some(error) => Err(error),
			None => Ok(()),
		}
	}

	/// Synchronizes a current committed or external head to every selected
	/// binding.
	pub async fn publish_head(
		&self,
		document_id: DocumentId,
		cancel: CancellationToken,
	) -> Result<(), LspRegistryError> {
		let _mutation = self.inner.mutation.lock().await;
		self.refresh_document(document_id, cancel).await
	}

	/// Applies dynamic registrations and schedules document reconciliation after
	/// the registration request can be acknowledged on the server lane.
	pub async fn register_capabilities(
		&self,
		handle: LspBindingHandle,
		params_json: Bytes,
		cancel: CancellationToken,
	) -> Result<(), LspRegistryError> {
		let mutation = self.inner.mutation.lock().await;
		let binding = self.binding_for_handle(handle).await?;
		let affected = self.binding_document_ids(binding.id).await;
		binding.server.register_capabilities(params_json)?;
		drop(mutation);
		self.schedule_policy_reconciliation(binding, affected, cancel);
		Ok(())
	}

	/// Applies dynamic unregistrations and schedules document reconciliation
	/// after the unregister request can be acknowledged on the server lane.
	pub async fn unregister_capabilities(
		&self,
		handle: LspBindingHandle,
		params_json: Bytes,
		cancel: CancellationToken,
	) -> Result<(), LspRegistryError> {
		let mutation = self.inner.mutation.lock().await;
		let binding = self.binding_for_handle(handle).await?;
		let affected = self.binding_document_ids(binding.id).await;
		binding.server.unregister_capabilities(params_json)?;
		drop(mutation);
		self.schedule_policy_reconciliation(binding, affected, cancel);
		Ok(())
	}

	/// Replaces a restarted server lane only after its complete document state
	/// has been staged successfully.
	pub async fn restart_binding(
		&self,
		binding_id: LspBindingId,
		server: LspServer,
		cancel: CancellationToken,
	) -> Result<(), LspRegistryError> {
		let _mutation = self.inner.mutation.lock().await;
		if self
			.inner
			.state
			.lock()
			.provisional_leases
			.iter()
			.any(|lease| lease.binding_id == binding_id)
		{
			return Err(LspRegistryError::BindingBusy { binding_id });
		}
		let binding = self.binding(binding_id).await?;
		let generation = binding
			.generation
			.checked_add(1)
			.ok_or(LspRegistryError::BindingGenerationOverflow { binding_id })?;
		let records = self.lease_records().await;
		let mut documents =
			HashMap::<DocumentId, (Arc<DocumentSnapshot>, Url, Option<LanguageId>, usize)>::new();
		for (_, record) in records
			.iter()
			.filter(|(_, record)| record.binding_ids.contains(&binding_id))
		{
			let (snapshot, uri) = self.current_snapshot(record.document_id).await?;
			documents
				.entry(record.document_id)
				.and_modify(|entry| entry.3 += 1)
				.or_insert_with(|| (snapshot, uri, record.language_id.clone(), 1));
		}

		let mut acquired = Vec::<(DocumentId, usize)>::new();
		let mut public_versions = Vec::with_capacity(documents.len());
		for (document_id, (snapshot, uri, language_id, count)) in documents {
			let version = match server
				.synchronize(lsp_document(&snapshot, &uri, language_id.as_ref()), cancel.child_token())
				.await
			{
				Ok(version) => version,
				Err(error) => {
					self.cleanup_replacement_leases(&server, &acquired).await;
					return Err(error.into());
				},
			};
			acquired.push((document_id, 1));
			public_versions.push((document_id, Str::new(uri.as_str()), version));
			for _ in 1..count {
				if let Err(error) = server.retain_document(document_id).await {
					self.cleanup_replacement_leases(&server, &acquired).await;
					return Err(error.into());
				}
				acquired
					.last_mut()
					.expect("replacement document was staged")
					.1 += 1;
			}
		}

		{
			let mut state = self.inner.state.lock();
			state.public_versions.retain(|(id, _), _| *id != binding_id);
			for (document_id, uri, version) in public_versions {
				state
					.public_versions
					.entry((binding_id, document_id))
					.or_default()
					.push_back((uri, version));
			}
			state
				.bindings
				.insert(binding_id, Binding { server, generation, ..binding });
		}
		self.publish_binding_event(binding_id, None, LspBindingEventKind::Restarted);
		Ok(())
	}

	/// Sends an exact raw workspace request without document revision tagging.
	pub async fn workspace_request(
		&self,
		binding_id: LspBindingId,
		method: &str,
		params_json: Bytes,
		cancel: CancellationToken,
	) -> Result<LspResponse, LspRegistryError> {
		let binding = self.binding(binding_id).await?;
		let response = binding
			.server
			.request(method, params_json, None, cancel)
			.await?;
		self.ensure_binding_generation(binding.id, binding.generation)?;
		Ok(response)
	}

	/// Sends an exact raw non-lifecycle notification through the binding's
	/// ordered lane.
	pub async fn notification(
		&self,
		binding_id: LspBindingId,
		method: &str,
		params_json: Bytes,
		cancel: CancellationToken,
	) -> Result<(), LspRegistryError> {
		let binding = self.binding(binding_id).await?;
		binding
			.server
			.notification(method, params_json, cancel)
			.await?;
		self.ensure_binding_generation(binding.id, binding.generation)
	}

	/// Admits opaque semantic parameters only against their exact requested
	/// revision. Stale raw position parameters are never replayed at a newer
	/// revision.
	pub async fn semantic_request(
		&self,
		binding_id: LspBindingId,
		method: &str,
		params_json: Bytes,
		lease_id: LeaseId,
		requested_revision: Revision,
		_stale_policy: StaleResponsePolicy,
		cancel: CancellationToken,
	) -> Result<LspResponse, LspRegistryError> {
		let binding = self.binding(binding_id).await?;
		let record = self.lease_record(lease_id).await?;
		if !record.binding_ids.contains(&binding_id) {
			return Err(LspRegistryError::BindingNotSelected {
				binding_id,
				document_id: record.document_id,
			});
		}
		let snapshot = self.snapshot_from_store(lease_id, None).await?;
		let current = snapshot.head().revision();
		if current != requested_revision {
			return Err(LspRegistryError::ContentModified { requested: requested_revision, current });
		}
		let uri = self.document_uri(record.document_id).await?;
		let version = binding
			.server
			.synchronize(
				lsp_document(&snapshot, &uri, record.language_id.as_ref()),
				cancel.child_token(),
			)
			.await?;
		if !self.mark_public_version_if_current(&binding, record.document_id, &uri, version) {
			return Err(LspRegistryError::BindingRestarted { binding_id });
		}
		let response = binding
			.server
			.request(
				method,
				params_json,
				Some(lsp_document(&snapshot, &uri, record.language_id.as_ref())),
				cancel.child_token(),
			)
			.await?;
		self.ensure_binding_generation(binding.id, binding.generation)?;
		let newest = self
			.snapshot_from_store(lease_id, None)
			.await?
			.head()
			.revision();
		if newest == requested_revision && response.revision == Some(requested_revision) {
			return Ok(response);
		}
		Err(LspRegistryError::ContentModified { requested: requested_revision, current: newest })
	}

	/// Tags an inbound event, resolving versioned diagnostics when the mapping
	/// is provable.
	pub async fn tag_inbound_event(
		&self,
		handle: LspBindingHandle,
		method: impl AsRef<str>,
		params_json: Bytes,
	) -> Result<TaggedLspEvent, LspRegistryError> {
		let binding = self.binding_for_handle(handle).await?;
		let binding_id = binding.id;
		let method = method.as_ref();
		let value: Value = serde_json::from_slice(&params_json).map_err(|error| {
			LspRegistryError::InvalidInboundJson { reason: Str::new(error.to_string()) }
		})?;
		let uri = value
			.get("uri")
			.or_else(|| value.pointer("/textDocument/uri"))
			.and_then(Value::as_str)
			.and_then(|uri| Url::parse(uri).ok());
		let version = (method == "textDocument/publishDiagnostics")
			.then(|| value.get("version"))
			.flatten()
			.and_then(Value::as_i64)
			.and_then(|version| i32::try_from(version).ok());
		let document_identity = if let Some(uri) = uri {
			let state = self.inner.state.lock();
			let mut document_id = None;
			let mut ambiguous = false;
			for ((entry_binding_id, entry_document_id), entries) in &state.public_versions {
				if *entry_binding_id != binding_id
					|| !entries
						.iter()
						.any(|(entry_uri, _)| entry_uri.as_str() == uri.as_str())
				{
					continue;
				}
				if document_id.is_some_and(|known| known != *entry_document_id) {
					ambiguous = true;
					break;
				}
				document_id = Some(*entry_document_id);
			}
			(!ambiguous)
				.then_some(document_id)
				.flatten()
				.map(|document_id| (document_id, uri))
		} else {
			None
		};
		let revision = match (&document_identity, version) {
			(Some((document_id, uri)), Some(version)) => {
				let is_public = self
					.inner
					.state
					.lock()
					.public_versions
					.get(&(binding_id, *document_id))
					.is_some_and(|entries| {
						entries.iter().any(|(entry_uri, entry_version)| {
							entry_uri.as_str() == uri.as_str() && *entry_version == version
						})
					});
				if is_public {
					binding.server.revision_for_version(uri, version)
				} else {
					None
				}
			},
			_ => None,
		};
		self.ensure_binding_generation(binding.id, binding.generation)?;
		Ok(TaggedLspEvent {
			binding_id,
			binding_name: binding.spec.name,
			method: Str::new(method),
			params_json,
			revision,
			document_identity,
		})
	}

	/// Tags and publishes an inbound server notification.
	///
	/// The exact parameter bytes are retained in both the returned event and
	/// the clone delivered to every current subscriber.
	pub async fn publish_inbound_event(
		&self,
		handle: LspBindingHandle,
		method: impl AsRef<str>,
		params_json: Bytes,
	) -> Result<TaggedLspEvent, LspRegistryError> {
		let event = self.tag_inbound_event(handle, method, params_json).await?;
		let _ = self
			.inner
			.events
			.send(LspRegistryEvent::Inbound(Box::new(event.clone())));
		Ok(event)
	}

	async fn binding_document_ids(&self, binding_id: LspBindingId) -> HashSet<DocumentId> {
		self
			.inner
			.state
			.lock()
			.leases
			.values()
			.filter(|record| record.binding_ids.contains(&binding_id))
			.map(|record| record.document_id)
			.collect()
	}

	fn schedule_policy_reconciliation(
		&self,
		binding: Binding,
		affected: HashSet<DocumentId>,
		cancel: CancellationToken,
	) {
		let registry = self.clone();
		tokio::spawn(async move {
			let _mutation = registry.inner.mutation.lock().await;
			let is_current = registry
				.inner
				.state
				.lock()
				.bindings
				.get(&binding.id)
				.is_some_and(|current| current.generation == binding.generation);
			if !is_current || registry.refresh_all_documents(cancel).await.is_err() {
				return;
			}
			for document_id in affected {
				registry.publish_binding_event(
					binding.id,
					Some(document_id),
					LspBindingEventKind::PolicyChanged,
				);
			}
		});
	}

	fn publish_binding_event(
		&self,
		binding_id: LspBindingId,
		document_id: Option<DocumentId>,
		kind: LspBindingEventKind,
	) {
		let _ = self
			.inner
			.events
			.send(LspRegistryEvent::Binding(LspBindingEvent { binding_id, document_id, kind }));
	}

	async fn refresh_all_documents(
		&self,
		cancel: CancellationToken,
	) -> Result<(), LspRegistryError> {
		let ids = self
			.inner
			.state
			.lock()
			.leases
			.values()
			.map(|record| record.document_id)
			.collect::<HashSet<_>>();
		for document_id in ids {
			self
				.refresh_document(document_id, cancel.child_token())
				.await?;
		}
		Ok(())
	}

	async fn refresh_document(
		&self,
		document_id: DocumentId,
		cancel: CancellationToken,
	) -> Result<(), LspRegistryError> {
		let records = self
			.lease_records()
			.await
			.into_iter()
			.filter(|(_, record)| record.document_id == document_id)
			.collect::<Vec<_>>();
		if records.is_empty() {
			return Ok(());
		}
		let (snapshot, uri) = self.current_snapshot(document_id).await?;
		let bindings = self.sorted_bindings().await;
		let mut desired_by_lease = HashMap::new();
		let mut desired_counts = HashMap::<LspBindingId, usize>::new();
		let mut desired_languages = HashMap::<LspBindingId, Option<LanguageId>>::new();
		let mut current_counts = HashMap::<LspBindingId, usize>::new();
		let mut current_languages = HashMap::<LspBindingId, Option<LanguageId>>::new();
		for (lease_id, record) in &records {
			for id in &record.binding_ids {
				*current_counts.entry(*id).or_default() += 1;
				current_languages
					.entry(*id)
					.or_insert_with(|| record.language_id.clone());
			}
			let desired = bindings
				.iter()
				.filter(|binding| {
					binding
						.spec
						.selector
						.matches(&uri, record.language_id.as_ref())
				})
				.map(|binding| binding.id)
				.collect::<Vec<_>>();
			for id in &desired {
				*desired_counts.entry(*id).or_default() += 1;
				desired_languages
					.entry(*id)
					.or_insert_with(|| record.language_id.clone());
			}
			desired_by_lease.insert(*lease_id, desired);
		}
		let mut provisional_counts = HashMap::<LspBindingId, usize>::new();
		for lease in &self.inner.state.lock().provisional_leases {
			if lease.document_id == document_id {
				*provisional_counts.entry(lease.binding_id).or_default() += 1;
			}
		}

		let mut progress = Vec::new();
		for binding in bindings {
			let binding_id = binding.id;
			let server = binding.server.clone();
			let provisional = provisional_counts.get(&binding_id).copied().unwrap_or(0);
			let current_committed = current_counts.get(&binding_id).copied().unwrap_or(0);
			let desired_committed = desired_counts.get(&binding_id).copied().unwrap_or(0);
			let current = current_committed + provisional;
			let desired = desired_committed + provisional;
			let current_language = current_languages.get(&binding_id).cloned().flatten();
			let index = progress.len();
			progress.push(RefreshProgress {
				binding,
				original_count: current,
				opened: false,
				retained: 0,
				released: 0,
				current_language,
			});
			if desired_committed > 0 {
				let version = match server
					.synchronize(
						lsp_document(
							&snapshot,
							&uri,
							desired_languages.get(&binding_id).and_then(Option::as_ref),
						),
						cancel.child_token(),
					)
					.await
				{
					Ok(version) => version,
					Err(error) => {
						self
							.compensate_refresh(document_id, &snapshot, &uri, &progress)
							.await;
						return Err(error.into());
					},
				};
				progress[index].opened = current == 0;
				self
					.mark_public_version(binding_id, document_id, &uri, version)
					.await;
				for _ in current.max(1)..desired {
					if let Err(error) = server.retain_document(document_id).await {
						self
							.compensate_refresh(document_id, &snapshot, &uri, &progress)
							.await;
						return Err(error.into());
					}
					progress[index].retained += 1;
				}
			}
			if current > desired {
				for _ in desired..current {
					if let Err(error) = server
						.release_document(document_id, cancel.child_token())
						.await
					{
						self
							.compensate_refresh(document_id, &snapshot, &uri, &progress)
							.await;
						return Err(error.into());
					}
					progress[index].released += 1;
				}
			}
		}

		let mut state = self.inner.state.lock();
		for (lease_id, binding_ids) in desired_by_lease {
			state
				.leases
				.get_mut(&lease_id)
				.expect("lease retained under mutation gate")
				.binding_ids = binding_ids;
		}
		for binding_id in current_counts.keys().chain(desired_counts.keys()) {
			let desired = desired_counts.get(binding_id).copied().unwrap_or(0)
				+ provisional_counts.get(binding_id).copied().unwrap_or(0);
			if desired == 0 {
				state.public_versions.remove(&(*binding_id, document_id));
			}
		}
		Ok(())
	}

	async fn compensate_refresh(
		&self,
		document_id: DocumentId,
		snapshot: &DocumentSnapshot,
		uri: &Url,
		progress: &[RefreshProgress],
	) {
		for item in progress.iter().rev() {
			if item.released > 0 {
				let remaining = item.original_count.saturating_sub(item.released);
				if remaining == 0 {
					if let Ok(version) = item
						.binding
						.server
						.synchronize(
							lsp_document(snapshot, uri, item.current_language.as_ref()),
							CancellationToken::new(),
						)
						.await
					{
						self
							.mark_public_version(item.binding.id, document_id, uri, version)
							.await;
						for _ in 1..item.released {
							let _ = item.binding.server.retain_document(document_id).await;
						}
					}
				} else {
					for _ in 0..item.released {
						let _ = item.binding.server.retain_document(document_id).await;
					}
				}
			}
			for _ in 0..item.retained {
				if item
					.binding
					.server
					.release_document(document_id, CancellationToken::new())
					.await
					.is_err()
				{
					item
						.binding
						.server
						.abandon_document_lease(document_id)
						.await;
				}
			}
			if item.opened {
				if item
					.binding
					.server
					.release_document(document_id, CancellationToken::new())
					.await
					.is_err()
				{
					item
						.binding
						.server
						.abandon_document_lease(document_id)
						.await;
				}
				self
					.inner
					.state
					.lock()
					.public_versions
					.remove(&(item.binding.id, document_id));
			}
		}
	}

	async fn sorted_bindings(&self) -> Vec<Binding> {
		let mut bindings = self
			.inner
			.state
			.lock()
			.bindings
			.values()
			.cloned()
			.collect::<Vec<_>>();
		bindings.sort_by(binding_order);
		bindings
	}

	async fn matching_bindings(&self, uri: &Url, language_id: Option<&LanguageId>) -> Vec<Binding> {
		self
			.sorted_bindings()
			.await
			.into_iter()
			.filter(|binding| binding.spec.selector.matches(uri, language_id))
			.collect()
	}

	async fn binding(&self, binding_id: LspBindingId) -> Result<Binding, LspRegistryError> {
		self
			.inner
			.state
			.lock()
			.bindings
			.get(&binding_id)
			.cloned()
			.ok_or(LspRegistryError::UnknownBinding { binding_id })
	}

	async fn binding_for_handle(
		&self,
		handle: LspBindingHandle,
	) -> Result<Binding, LspRegistryError> {
		let binding = self.binding(handle.binding_id).await?;
		if binding.generation != handle.generation {
			return Err(LspRegistryError::BindingRestarted { binding_id: handle.binding_id });
		}
		Ok(binding)
	}

	async fn cleanup_replacement_leases(
		&self,
		server: &LspServer,
		acquired: &[(DocumentId, usize)],
	) {
		for (document_id, count) in acquired.iter().rev() {
			for _ in 0..*count {
				if server
					.release_document(*document_id, CancellationToken::new())
					.await
					.is_err()
				{
					server.abandon_document_lease(*document_id).await;
				}
			}
		}
	}

	async fn lease_record(&self, lease_id: LeaseId) -> Result<LeaseRecord, LspRegistryError> {
		self
			.inner
			.state
			.lock()
			.leases
			.get(&lease_id)
			.cloned()
			.ok_or(LspRegistryError::UnknownLease { lease_id })
	}

	async fn lease_records(&self) -> Vec<(LeaseId, LeaseRecord)> {
		self
			.inner
			.state
			.lock()
			.leases
			.iter()
			.map(|(id, record)| (*id, record.clone()))
			.collect()
	}

	async fn binding_document_count(
		&self,
		binding_id: LspBindingId,
		document_id: DocumentId,
	) -> usize {
		let state = self.inner.state.lock();
		let committed = state
			.leases
			.values()
			.filter(|record| {
				record.document_id == document_id && record.binding_ids.contains(&binding_id)
			})
			.count();
		let provisional = state
			.provisional_leases
			.iter()
			.filter(|lease| lease.binding_id == binding_id && lease.document_id == document_id)
			.count();
		committed + provisional
	}

	fn committed_binding_document_count(
		&self,
		binding_id: LspBindingId,
		document_id: DocumentId,
	) -> usize {
		self
			.inner
			.state
			.lock()
			.leases
			.values()
			.filter(|record| {
				record.document_id == document_id && record.binding_ids.contains(&binding_id)
			})
			.count()
	}

	async fn current_snapshot(
		&self,
		document_id: DocumentId,
	) -> Result<(Arc<DocumentSnapshot>, Url), LspRegistryError> {
		let state = self
			.inner
			.store
			.actor_handle(document_id)?
			.ready_state()
			.await?;
		let snapshot = state
			.head
			.ok_or(LspRegistryError::DocumentNotActivated { document_id })?;
		let uri = Url::from_file_path(&state.path)
			.map_err(|()| LspRegistryError::PathCannotBeUri { path: state.path })?;
		Ok((snapshot, uri))
	}

	async fn document_uri(&self, document_id: DocumentId) -> Result<Url, LspRegistryError> {
		Ok(self.current_snapshot(document_id).await?.1)
	}

	async fn snapshot_from_store(
		&self,
		lease_id: LeaseId,
		revision: Option<Revision>,
	) -> Result<Arc<DocumentSnapshot>, LspRegistryError> {
		let read = self
			.inner
			.store
			.read(lease_id, revision, ReadSelection::Whole)
			.await?;
		let content = match read.body() {
			ReadBody::Whole(content) => content.clone(),
			ReadBody::Slices(_) => unreachable!("whole selection returns whole bytes"),
		};
		Ok(Arc::new(DocumentSnapshot::new(read.head().clone(), content)?))
	}

	async fn mark_private_version(
		&self,
		binding_id: LspBindingId,
		document_id: DocumentId,
		uri: &Url,
		version: i32,
	) {
		if let Some(entries) = self
			.inner
			.state
			.lock()
			.public_versions
			.get_mut(&(binding_id, document_id))
		{
			entries.retain(|(entry_uri, entry_version)| {
				entry_uri.as_str() != uri.as_str() || *entry_version != version
			});
		}
	}

	fn ensure_binding_generation(
		&self,
		binding_id: LspBindingId,
		generation: u64,
	) -> Result<(), LspRegistryError> {
		if self
			.inner
			.state
			.lock()
			.bindings
			.get(&binding_id)
			.is_some_and(|binding| binding.generation == generation)
		{
			Ok(())
		} else {
			Err(LspRegistryError::BindingRestarted { binding_id })
		}
	}

	fn mark_public_version_if_current(
		&self,
		binding: &Binding,
		document_id: DocumentId,
		uri: &Url,
		version: i32,
	) -> bool {
		let mut state = self.inner.state.lock();
		if state
			.bindings
			.get(&binding.id)
			.is_none_or(|current| current.generation != binding.generation)
		{
			return false;
		}
		record_public_version(&mut state, binding.id, document_id, uri, version);
		true
	}

	async fn mark_public_version(
		&self,
		binding_id: LspBindingId,
		document_id: DocumentId,
		uri: &Url,
		version: i32,
	) {
		let mut state = self.inner.state.lock();
		record_public_version(&mut state, binding_id, document_id, uri, version);
	}

	async fn acquire_format_leases(
		&self,
		request: &FormatRequest,
		cancel: CancellationToken,
	) -> Result<FormatLeaseSet, LspRegistryError> {
		let _mutation = self.inner.mutation.lock().await;
		let document_id = request.base().head().document_id();
		let base_language_id = language_for_head(request.base().head()).cloned();
		let all_bindings = self.sorted_bindings().await;
		let base_uri = match self.current_snapshot(document_id).await {
			Ok((_, uri)) => uri,
			Err(_error)
				if all_bindings.iter().all(|binding| {
					self.committed_binding_document_count(binding.id, document_id) == 0
				}) =>
			{
				request.uri().clone()
			},
			Err(error) => return Err(error),
		};
		let bindings = all_bindings
			.into_iter()
			.filter(|binding| {
				binding
					.spec
					.selector
					.matches(&base_uri, base_language_id.as_ref())
			})
			.collect::<Vec<_>>();
		let snapshot = provisional_snapshot(request.base(), request.candidate().clone())?;
		let mut acquired = Vec::new();
		for binding in &bindings {
			let key = ProvisionalLeaseKey {
				binding_id: binding.id,
				document_id,
				transaction: request.transaction_id(),
			};
			let existing = self.committed_binding_document_count(binding.id, document_id);
			if self.inner.state.lock().provisional_leases.contains(&key) {
				acquired.push(FormatBindingLease { binding: binding.clone(), existing });
				continue;
			}
			let retained = existing > 0;
			if retained && let Err(error) = binding.server.retain_document(document_id).await {
				self
					.rollback_format_acquisition(
						request,
						&base_uri,
						base_language_id.as_ref(),
						&mut acquired,
					)
					.await;
				return Err(error.into());
			}
			let version = match binding
				.server
				.synchronize(
					lsp_document(&snapshot, &base_uri, base_language_id.as_ref()),
					cancel.child_token(),
				)
				.await
			{
				Ok(version) => version,
				Err(error) => {
					if retained
						&& let Ok(version) = binding
							.server
							.synchronize(
								lsp_document(request.base(), &base_uri, base_language_id.as_ref()),
								CancellationToken::new(),
							)
							.await
					{
						self
							.mark_public_version(binding.id, document_id, &base_uri, version)
							.await;
					}
					if retained
						&& binding
							.server
							.release_document(document_id, CancellationToken::new())
							.await
							.is_err()
					{
						binding.server.abandon_document_lease(document_id).await;
					}
					self
						.rollback_format_acquisition(
							request,
							&base_uri,
							base_language_id.as_ref(),
							&mut acquired,
						)
						.await;
					return Err(error.into());
				},
			};
			self
				.mark_private_version(binding.id, document_id, &base_uri, version)
				.await;
			self.inner.state.lock().provisional_leases.insert(key);
			acquired.push(FormatBindingLease { binding: binding.clone(), existing });
		}
		Ok(FormatLeaseSet { bindings: acquired, base_uri, base_language_id })
	}

	async fn rollback_format_acquisition(
		&self,
		request: &FormatRequest,
		base_uri: &Url,
		base_language_id: Option<&LanguageId>,
		acquired: &mut Vec<FormatBindingLease>,
	) {
		let document_id = request.base().head().document_id();
		for lease in acquired.drain(..).rev() {
			let key = ProvisionalLeaseKey {
				binding_id: lease.binding.id,
				document_id,
				transaction: request.transaction_id(),
			};
			if !self.inner.state.lock().provisional_leases.remove(&key) {
				continue;
			}
			if lease.existing > 0
				&& let Ok(version) = lease
					.binding
					.server
					.synchronize(
						lsp_document(request.base(), base_uri, base_language_id),
						CancellationToken::new(),
					)
					.await
			{
				self
					.mark_public_version(lease.binding.id, document_id, base_uri, version)
					.await;
			}
			if lease
				.binding
				.server
				.release_document(document_id, CancellationToken::new())
				.await
				.is_err()
			{
				lease
					.binding
					.server
					.abandon_document_lease(document_id)
					.await;
			}
			if lease.existing == 0 {
				self
					.inner
					.state
					.lock()
					.public_versions
					.remove(&(lease.binding.id, document_id));
			}
		}
	}

	async fn release_provisional_in_gate(
		&self,
		bindings: &[Binding],
		document_id: DocumentId,
		transaction_id: TransactionId,
		cancel: CancellationToken,
	) -> Result<(), LspRegistryError> {
		let mut first_error = None;
		for binding in bindings {
			let key = ProvisionalLeaseKey {
				binding_id: binding.id,
				document_id,
				transaction: transaction_id,
			};
			if self.inner.state.lock().provisional_leases.contains(&key) {
				if let Err(error) = binding
					.server
					.release_document(document_id, cancel.child_token())
					.await && first_error.is_none()
				{
					first_error = Some(LspRegistryError::from(error));
				}
				self.inner.state.lock().provisional_leases.remove(&key);
				if self.binding_document_count(binding.id, document_id).await == 0 {
					self
						.inner
						.state
						.lock()
						.public_versions
						.remove(&(binding.id, document_id));
				}
			}
		}
		match first_error {
			Some(error) => Err(error),
			None => Ok(()),
		}
	}

	async fn format_candidate_inner(
		&self,
		request: &FormatRequest,
		leases: &FormatLeaseSet,
		cancel: CancellationToken,
	) -> Result<Bytes, LspRegistryError> {
		if leases.bindings.is_empty() {
			return Err(LspRegistryError::FormattingUnavailable);
		}
		let bindings = &leases.bindings;
		let uri = &leases.base_uri;
		let language_id = leases.base_language_id.as_ref();
		let mut content = request.candidate().clone();
		let mut performed = false;
		for binding in bindings {
			let binding = &binding.binding;
			let mut snapshot = provisional_snapshot(request.base(), content.clone())?;
			let version = binding
				.server
				.synchronize(lsp_document(&snapshot, uri, language_id), cancel.child_token())
				.await?;
			self
				.mark_private_version(binding.id, request.base().head().document_id(), uri, version)
				.await;
			let policy = binding.server.sync_policy(uri, language_id);
			if policy.will_save {
				binding
					.server
					.will_save(lsp_document(&snapshot, uri, language_id), 1, cancel.child_token())
					.await?;
			}
			if policy.will_save_wait_until {
				content = binding
					.server
					.will_save_wait_until(
						lsp_document(&snapshot, uri, language_id),
						1,
						cancel.child_token(),
					)
					.await?;
				performed = true;
				snapshot = provisional_snapshot(request.base(), content.clone())?;
				let version = binding
					.server
					.synchronize(lsp_document(&snapshot, uri, language_id), cancel.child_token())
					.await?;
				self
					.mark_private_version(binding.id, request.base().head().document_id(), uri, version)
					.await;
			}
			if binding.server.supports_formatting(uri, language_id) {
				content = binding
					.server
					.format_document(
						lsp_document(&snapshot, uri, language_id),
						Bytes::from_static(br#"{"tabSize":4,"insertSpaces":true}"#),
						cancel.child_token(),
					)
					.await?;
				performed = true;
				snapshot = provisional_snapshot(request.base(), content.clone())?;
				let version = binding
					.server
					.synchronize(lsp_document(&snapshot, uri, language_id), cancel.child_token())
					.await?;
				self
					.mark_private_version(binding.id, request.base().head().document_id(), uri, version)
					.await;
			}
		}
		if !performed {
			return Err(LspRegistryError::FormattingUnavailable);
		}
		let snapshot = provisional_snapshot(request.base(), content.clone())?;
		for binding in bindings {
			let binding = &binding.binding;
			let version = binding
				.server
				.synchronize(lsp_document(&snapshot, uri, language_id), cancel.child_token())
				.await?;
			self
				.mark_private_version(binding.id, request.base().head().document_id(), uri, version)
				.await;
		}
		Ok(content)
	}

	async fn rollback_candidate(&self, request: &FormatRequest, leases: &FormatLeaseSet) {
		for lease in &leases.bindings {
			if lease.existing == 0 {
				continue;
			}
			let binding = &lease.binding;
			if let Ok(version) = binding
				.server
				.synchronize(
					lsp_document(request.base(), &leases.base_uri, leases.base_language_id.as_ref()),
					CancellationToken::new(),
				)
				.await
			{
				self
					.mark_public_version(
						binding.id,
						request.base().head().document_id(),
						&leases.base_uri,
						version,
					)
					.await;
			}
		}
	}

	async fn publish_committed_inner(
		&self,
		document: &PublishedDocument,
		bindings: &[Binding],
		cancel: CancellationToken,
	) -> Result<(), LspRegistryError> {
		let snapshot = DocumentSnapshot::new(document.head().clone(), document.content().clone())?;
		let language_id = language_for_head(document.head());
		for binding in bindings {
			let version = binding
				.server
				.synchronize(lsp_document(&snapshot, document.uri(), language_id), cancel.child_token())
				.await?;
			self
				.mark_public_version(binding.id, document.head().document_id(), document.uri(), version)
				.await;
			if binding.server.sync_policy(document.uri(), language_id).save {
				binding
					.server
					.did_save(lsp_document(&snapshot, document.uri(), language_id), cancel.child_token())
					.await?;
			}
		}
		Ok(())
	}
}

impl FormatCoordinator for LspRegistry {
	fn format_candidate(
		&self,
		request: FormatRequest,
		cancel: CancellationToken,
	) -> impl Future<Output = crate::Result<FormatResult>> + Send + '_ {
		async move {
			let leases = self
				.acquire_format_leases(&request, cancel.child_token())
				.await
				.map_err(registry_protocol_error)?;
			match self.format_candidate_inner(&request, &leases, cancel).await {
				Ok(content) => Ok(FormatResult::new(content)),
				Err(error) => {
					let _mutation = self.inner.mutation.lock().await;
					self.rollback_candidate(&request, &leases).await;
					let bindings = leases
						.bindings
						.iter()
						.map(|lease| lease.binding.clone())
						.collect::<Vec<_>>();
					let _ = self
						.release_provisional_in_gate(
							&bindings,
							request.base().head().document_id(),
							request.transaction_id(),
							CancellationToken::new(),
						)
						.await;
					Err(registry_protocol_error(error))
				},
			}
		}
	}

	fn publish_committed(
		&self,
		document: PublishedDocument,
		cancel: CancellationToken,
	) -> impl Future<Output = crate::Result<()>> + Send + '_ {
		async move {
			let _mutation = self.inner.mutation.lock().await;
			let refresh_result = if self
				.inner
				.state
				.lock()
				.leases
				.values()
				.any(|record| record.document_id == document.head().document_id())
			{
				self
					.refresh_document(document.head().document_id(), cancel.child_token())
					.await
			} else {
				Ok(())
			};
			let mut bindings = Vec::new();
			let mut included = HashSet::new();
			for binding in self
				.matching_bindings(document.uri(), language_for_head(document.head()))
				.await
			{
				if self
					.binding_document_count(binding.id, document.head().document_id())
					.await > 0
				{
					included.insert(binding.id);
					bindings.push(binding);
				}
			}
			let provisional_ids = self
				.inner
				.state
				.lock()
				.provisional_leases
				.iter()
				.filter(|key| {
					key.document_id == document.head().document_id()
						&& key.transaction == document.transaction_id()
				})
				.map(|key| key.binding_id)
				.collect::<Vec<_>>();
			for binding_id in provisional_ids {
				if included.insert(binding_id) {
					bindings.push(
						self
							.binding(binding_id)
							.await
							.map_err(registry_protocol_error)?,
					);
				}
			}
			let publish_result = match refresh_result {
				Ok(()) => {
					self
						.publish_committed_inner(&document, &bindings, cancel.child_token())
						.await
				},
				Err(error) => Err(error),
			};
			let release_result = self
				.release_provisional_in_gate(
					&bindings,
					document.head().document_id(),
					document.transaction_id(),
					cancel,
				)
				.await;
			publish_result
				.and(release_result)
				.map_err(registry_protocol_error)
		}
	}

	fn revert_uncommitted(
		&self,
		document: RevertedDocument,
		cancel: CancellationToken,
	) -> impl Future<Output = crate::Result<()>> + Send + '_ {
		async move {
			let _mutation = self.inner.mutation.lock().await;
			let snapshot = document.snapshot();
			let keys = self
				.inner
				.state
				.lock()
				.provisional_leases
				.iter()
				.copied()
				.filter(|key| {
					key.document_id == snapshot.head().document_id()
						&& key.transaction == document.transaction_id()
				})
				.collect::<Vec<_>>();
			let mut bindings = Vec::new();
			let mut first_error = None;
			for key in keys {
				let binding = match self.binding(key.binding_id).await {
					Ok(binding) => binding,
					Err(error) => {
						self.inner.state.lock().provisional_leases.remove(&key);
						if first_error.is_none() {
							first_error = Some(error);
						}
						continue;
					},
				};
				match binding
					.server
					.synchronize(
						lsp_document(snapshot, document.uri(), document.language_id()),
						cancel.child_token(),
					)
					.await
				{
					Ok(version) => {
						self
							.mark_public_version(
								binding.id,
								snapshot.head().document_id(),
								document.uri(),
								version,
							)
							.await;
					},
					Err(error) if first_error.is_none() => {
						first_error = Some(LspRegistryError::from(error));
					},
					Err(_) => {},
				}
				bindings.push(binding);
			}
			let release_result = self
				.release_provisional_in_gate(
					&bindings,
					snapshot.head().document_id(),
					document.transaction_id(),
					cancel,
				)
				.await;
			match first_error {
				Some(error) => Err(registry_protocol_error(error)),
				None => release_result.map_err(registry_protocol_error),
			}
		}
	}
}

fn record_public_version(
	state: &mut RegistryState,
	binding_id: LspBindingId,
	document_id: DocumentId,
	uri: &Url,
	version: i32,
) {
	let entries = state
		.public_versions
		.entry((binding_id, document_id))
		.or_default();
	entries.retain(|(entry_uri, entry_version)| {
		entry_uri.as_str() != uri.as_str() || *entry_version != version
	});
	if entries.len() == PUBLIC_VERSION_LIMIT {
		entries.pop_front();
	}
	entries.push_back((Str::new(uri.as_str()), version));
}

fn provisional_snapshot(
	base: &DocumentSnapshot,
	content: Bytes,
) -> crate::Result<DocumentSnapshot> {
	let sequence = base
		.head()
		.revision()
		.sequence()
		.checked_add(1)
		.unwrap_or_else(|| base.head().revision().sequence());
	let revision = Revision::for_content(sequence, &content);
	let presence = DocumentPresence::Present;
	let head = DocumentHead::new(
		base.head().document_id(),
		revision,
		presence,
		base.head().kind().clone(),
		content.len() as u64,
	)?;
	DocumentSnapshot::new(head, content)
}

const fn language_for_head(head: &DocumentHead) -> Option<&LanguageId> {
	match head.kind() {
		DocumentKind::Text(language_id) => language_id.as_ref(),
		DocumentKind::Binary => None,
	}
}

fn registry_protocol_error(error: LspRegistryError) -> crate::Error {
	crate::Error::Protocol { reason: Str::new(error.to_string()) }
}

const fn lsp_document<'a>(
	snapshot: &'a DocumentSnapshot,
	uri: &'a Url,
	language_id: Option<&'a LanguageId>,
) -> LspDocument<'a> {
	LspDocument { snapshot, uri, language_id }
}

fn binding_order(left: &Binding, right: &Binding) -> std::cmp::Ordering {
	right
		.spec
		.priority
		.cmp(&left.spec.priority)
		.then_with(|| left.spec.name.cmp(&right.spec.name))
		.then_with(|| left.id.cmp(&right.id))
}

fn binding_info_order(left: &LspBindingInfo, right: &LspBindingInfo) -> std::cmp::Ordering {
	right
		.spec
		.priority
		.cmp(&left.spec.priority)
		.then_with(|| left.spec.name.cmp(&right.spec.name))
		.then_with(|| left.id.cmp(&right.id))
}

/// A binding, selection, revision-admission, or delegated LSP failure.
#[derive(Debug, Error)]
pub enum LspRegistryError {
	/// A binding name was empty.
	#[error("LSP binding name must not be empty")]
	InvalidBindingName,
	/// A binding name is already installed.
	#[error("LSP binding {name} already exists")]
	DuplicateBinding {
		/// Duplicate binding name.
		name: Str,
	},
	/// A selector glob could not be compiled.
	#[error("invalid LSP selector: {reason}")]
	InvalidSelector {
		/// Selector diagnostic.
		reason: Str,
	},
	/// No installed binding has this identity.
	#[error("unknown LSP binding {}", binding_id.get())]
	UnknownBinding {
		/// Missing binding identity.
		binding_id: LspBindingId,
	},
	/// A topology operation cannot replace a binding while it owns provisional
	/// text.
	#[error("LSP binding {} has an active provisional document lease", binding_id.get())]
	BindingBusy {
		/// Busy binding identity.
		binding_id: LspBindingId,
	},
	/// An in-flight operation completed on a server generation that has been
	/// replaced.
	#[error("LSP binding {} restarted while the operation was in flight", binding_id.get())]
	BindingRestarted {
		/// Replaced binding identity.
		binding_id: LspBindingId,
	},
	/// No open registry lease has this identity.
	#[error("unknown LSP registry lease {lease_id}")]
	UnknownLease {
		/// Missing lease identity.
		lease_id: LeaseId,
	},
	/// The selected server is not bound to this document lease.
	#[error("LSP binding {} is not selected for document {document_id}", binding_id.get())]
	BindingNotSelected {
		/// Unselected binding identity.
		binding_id:  LspBindingId,
		/// Requested document identity.
		document_id: DocumentId,
	},
	/// A semantic request or response raced a different current head.
	#[error("document content modified: requested {requested}, current {current}")]
	ContentModified {
		/// Revision against which the operation was admitted.
		requested: Revision,
		/// Newest revision observed at rejection.
		current:   Revision,
	},
	/// A document actor had no activated immutable head.
	#[error("document {document_id} is not activated")]
	DocumentNotActivated {
		/// Document identity.
		document_id: DocumentId,
	},
	/// A canonical document path could not be represented as a file URI.
	#[error("document path cannot be represented as a file URI: {path:?}")]
	PathCannotBeUri {
		/// Canonical path.
		path: std::path::PathBuf,
	},
	/// Binding identities exhausted their integer representation.
	#[error("LSP binding identity overflow")]
	BindingIdOverflow,
	/// A binding's restart generation exhausted its integer representation.
	#[error("LSP binding {} restart generation overflow", binding_id.get())]
	BindingGenerationOverflow {
		/// Binding whose generation could not advance.
		binding_id: LspBindingId,
	},
	/// Inbound parameters were not exact valid JSON.
	#[error("invalid inbound LSP JSON: {reason}")]
	InvalidInboundJson {
		/// JSON diagnostic.
		reason: Str,
	},
	/// No selected server advertised an operation capable of formatting bytes.
	#[error("no selected LSP binding provides formatting")]
	FormattingUnavailable,
	/// The document store rejected the operation.
	#[error(transparent)]
	Store(#[from] crate::Error),
	/// The selected LSP lane rejected the operation.
	#[error(transparent)]
	Lsp(#[from] LspError),
}

#[cfg(test)]
mod tests {
	use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

	use async_trait::async_trait;
	use tokio::sync::Notify;

	use super::*;
	use crate::{
		DocumentPresence, ServerConfig, TransactionId,
		lsp::{LspTransport, LspTransportError},
	};
	struct NullTransport;

	#[async_trait]
	impl LspTransport for NullTransport {
		async fn request(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<Bytes, LspTransportError> {
			Ok(Bytes::from_static(b"null"))
		}

		async fn notify(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<(), LspTransportError> {
			Ok(())
		}
	}

	struct HangingCloseTransport;

	#[async_trait]
	impl LspTransport for HangingCloseTransport {
		async fn request(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<Bytes, LspTransportError> {
			Ok(Bytes::from_static(b"null"))
		}

		async fn notify(
			&self,
			method: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<(), LspTransportError> {
			if method == "textDocument/didClose" {
				std::future::pending().await
			} else {
				Ok(())
			}
		}
	}

	struct PendingFormatTransport {
		started: Notify,
		release: Notify,
	}

	#[async_trait]
	impl LspTransport for PendingFormatTransport {
		async fn request(
			&self,
			method: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<Bytes, LspTransportError> {
			if method == "textDocument/formatting" {
				self.started.notify_one();
				self.release.notified().await;
				Ok(Bytes::from_static(b"[]"))
			} else {
				Ok(Bytes::from_static(b"null"))
			}
		}

		async fn notify(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<(), LspTransportError> {
			Ok(())
		}
	}
	struct CountingTransport {
		messages: AtomicU64,
	}

	#[async_trait]
	impl LspTransport for CountingTransport {
		async fn request(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<Bytes, LspTransportError> {
			self.messages.fetch_add(1, Ordering::Relaxed);
			Ok(Bytes::from_static(b"null"))
		}

		async fn notify(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<(), LspTransportError> {
			self.messages.fetch_add(1, Ordering::Relaxed);
			Ok(())
		}
	}

	struct PendingRequestTransport {
		started: Notify,
		release: Notify,
	}

	#[async_trait]
	impl LspTransport for PendingRequestTransport {
		async fn request(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<Bytes, LspTransportError> {
			self.started.notify_one();
			self.release.notified().await;
			Ok(Bytes::from_static(b"null"))
		}

		async fn notify(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<(), LspTransportError> {
			Ok(())
		}
	}

	struct RequestCountingTransport {
		requests: AtomicU64,
	}

	#[async_trait]
	impl LspTransport for RequestCountingTransport {
		async fn request(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<Bytes, LspTransportError> {
			self.requests.fetch_add(1, Ordering::Relaxed);
			Ok(Bytes::from_static(b"null"))
		}

		async fn notify(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<(), LspTransportError> {
			Ok(())
		}
	}

	struct FailingNotifyTransport;

	#[async_trait]
	impl LspTransport for FailingNotifyTransport {
		async fn request(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<Bytes, LspTransportError> {
			Ok(Bytes::from_static(b"null"))
		}

		async fn notify(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<(), LspTransportError> {
			Err(LspTransportError::Closed { message: Str::new_static("injected failure") })
		}
	}

	struct ToggleNotifyTransport {
		fail:   AtomicBool,
		params: Mutex<Vec<Bytes>>,
	}

	#[async_trait]
	impl LspTransport for ToggleNotifyTransport {
		async fn request(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<Bytes, LspTransportError> {
			Ok(Bytes::from_static(b"null"))
		}

		async fn notify(
			&self,
			_: &str,
			params: Bytes,
			_: CancellationToken,
		) -> Result<(), LspTransportError> {
			self.params.lock().push(params);
			if self.fail.load(Ordering::Relaxed) {
				Err(LspTransportError::Closed { message: Str::new_static("injected failure") })
			} else {
				Ok(())
			}
		}
	}

	struct FailSecondNotifyTransport {
		notifications: AtomicU64,
	}

	#[async_trait]
	impl LspTransport for FailSecondNotifyTransport {
		async fn request(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<Bytes, LspTransportError> {
			Ok(Bytes::from_static(b"null"))
		}

		async fn notify(
			&self,
			_: &str,
			_: Bytes,
			_: CancellationToken,
		) -> Result<(), LspTransportError> {
			if self.notifications.fetch_add(1, Ordering::Relaxed) == 1 {
				Err(LspTransportError::Closed {
					message: Str::new_static("injected second notification failure"),
				})
			} else {
				Ok(())
			}
		}
	}

	fn server() -> LspServer {
		LspServer::new(
			Arc::new(NullTransport),
			Bytes::from_static(br#"{"textDocumentSync":{"openClose":true,"change":1}}"#),
		)
		.unwrap()
	}

	fn binding(id: u64, name: &str, priority: i32) -> Binding {
		Binding {
			id:         LspBindingId(id),
			spec:       LspBindingSpec::new(name, priority, LspSelector::all()).unwrap(),
			server:     server(),
			generation: 0,
		}
	}

	#[tokio::test]
	async fn inbound_publication_preserves_bytes_and_proven_revision_identity() {
		static ROOT_SEQUENCE: AtomicU64 = AtomicU64::new(20_000);
		let root = std::env::temp_dir().join(format!(
			"omp-lsp-registry-events-{}-{}",
			std::process::id(),
			ROOT_SEQUENCE.fetch_add(1, Ordering::Relaxed),
		));
		std::fs::create_dir_all(&root).unwrap();
		let registry =
			LspRegistry::new(DocumentStore::new(ServerConfig::new(&root).unwrap()).unwrap());
		let binding_id = registry
			.add_binding(
				LspBindingSpec::new("events", 0, LspSelector::all()).unwrap(),
				server(),
				CancellationToken::new(),
			)
			.await
			.unwrap();
		let handle = registry.binding_handle(binding_id).await.unwrap();
		let document_id = DocumentId::from_bytes([7; 16]);
		let content = Bytes::from_static(b"published");
		let revision = Revision::for_content(1, &content);
		let head = DocumentHead::new(
			document_id,
			revision,
			DocumentPresence::Present,
			DocumentKind::Text(None),
			content.len() as u64,
		)
		.unwrap();
		let snapshot = DocumentSnapshot::new(head, content).unwrap();
		let uri = Url::from_file_path(root.join("published.txt")).unwrap();
		let bound = registry.binding(binding_id).await.unwrap();
		let version = bound
			.server
			.synchronize(lsp_document(&snapshot, &uri, None), CancellationToken::new())
			.await
			.unwrap();
		registry
			.mark_public_version(binding_id, document_id, &uri, version)
			.await;
		let params_json =
			Bytes::from(format!(r#"{{"uri":"{uri}","version":{version},"diagnostics":[]}}"#));
		let mut events = registry.subscribe_events();

		let tagged = registry
			.publish_inbound_event(handle, "textDocument/publishDiagnostics", params_json.clone())
			.await
			.unwrap();

		assert_eq!(tagged.params_json(), &params_json);
		assert_eq!(tagged.revision(), Some(revision));
		assert_eq!(tagged.document_id(), Some(document_id));
		assert_eq!(tagged.document_uri(), Some(&uri));
		assert_eq!(events.recv().await.unwrap(), LspRegistryEvent::Inbound(Box::new(tagged)));
		std::fs::remove_dir_all(root).unwrap();
	}

	#[test]
	fn selector_requires_every_declared_dimension() {
		let selector = LspSelector::new(
			vec![LanguageId::new("rust").unwrap()],
			vec![Str::new_static("file")],
			vec![Str::new_static("**/*.rs")],
		)
		.unwrap();
		let rust = LanguageId::new("rust").unwrap();
		let python = LanguageId::new("python").unwrap();
		assert!(selector.matches(&Url::parse("file:///project/src/lib.rs").unwrap(), Some(&rust)));
		assert!(!selector.matches(&Url::parse("file:///project/src/lib.py").unwrap(), Some(&rust)));
		assert!(!selector.matches(&Url::parse("file:///project/src/lib.rs").unwrap(), Some(&python)));
		assert!(
			!selector.matches(&Url::parse("untitled:///project/src/lib.rs").unwrap(), Some(&rust))
		);
	}

	#[test]
	fn bindings_order_by_priority_name_then_identity() {
		let mut bindings = [binding(3, "zeta", 10), binding(2, "alpha", 10), binding(1, "low", 1)];
		bindings.sort_by(binding_order);
		assert_eq!(
			bindings
				.iter()
				.map(|binding| binding.id.get())
				.collect::<Vec<_>>(),
			vec![2, 3, 1],
		);
	}

	#[test]
	fn provisional_snapshot_never_mutates_the_committed_base() {
		let content = Bytes::from_static(b"base");
		let revision = Revision::for_content(4, &content);
		let head = DocumentHead::new(
			DocumentId::from_bytes([9; 16]),
			revision,
			DocumentPresence::Present,
			DocumentKind::Text(None),
			content.len() as u64,
		)
		.unwrap();
		let base = DocumentSnapshot::new(head, content).unwrap();
		let provisional = provisional_snapshot(&base, Bytes::from_static(b"candidate")).unwrap();
		assert_eq!(base.content(), &Bytes::from_static(b"base"));
		assert_eq!(provisional.content(), &Bytes::from_static(b"candidate"));
		assert_ne!(base.head().revision(), provisional.head().revision());
	}
	#[test]
	fn empty_provisional_text_remains_present() {
		let content = Bytes::new();
		let revision = Revision::for_content(1, &content);
		let head = DocumentHead::new(
			DocumentId::from_bytes([4; 16]),
			revision,
			DocumentPresence::Missing,
			DocumentKind::Text(None),
			0,
		)
		.unwrap();
		let base = DocumentSnapshot::new(head, content).unwrap();
		let provisional = provisional_snapshot(&base, Bytes::new()).unwrap();
		assert_eq!(provisional.head().presence(), DocumentPresence::Present);
	}

	#[tokio::test]
	async fn dynamic_registration_completes_while_formatting_is_pending() {
		static ROOT_SEQUENCE: AtomicU64 = AtomicU64::new(1);
		let root = std::env::temp_dir().join(format!(
			"omp-lsp-registry-{}-{}",
			std::process::id(),
			ROOT_SEQUENCE.fetch_add(1, Ordering::Relaxed),
		));
		std::fs::create_dir_all(&root).unwrap();
		let store = DocumentStore::new(ServerConfig::new(&root).unwrap()).unwrap();
		let registry = LspRegistry::new(store);
		let transport =
			Arc::new(PendingFormatTransport { started: Notify::new(), release: Notify::new() });
		let server = LspServer::new(
			transport.clone(),
			Bytes::from_static(
				br#"{"documentFormattingProvider":true,"textDocumentSync":{"openClose":true,"change":1}}"#,
			),
		).unwrap();
		let binding_id = registry
			.add_binding(
				LspBindingSpec::new("formatter", 0, LspSelector::all()).unwrap(),
				server,
				CancellationToken::new(),
			)
			.await
			.unwrap();
		let binding_handle = registry.binding_handle(binding_id).await.unwrap();
		let content = Bytes::from_static(b"base");
		let revision = Revision::for_content(1, &content);
		let head = DocumentHead::new(
			DocumentId::from_bytes([5; 16]),
			revision,
			DocumentPresence::Present,
			DocumentKind::Text(None),
			content.len() as u64,
		)
		.unwrap();
		let base = Arc::new(DocumentSnapshot::new(head, content).unwrap());
		let uri = Url::from_file_path(root.join("file.txt")).unwrap();
		let transaction_id = TransactionId::from_bytes([6; 16]);
		let rollback_base = base.clone();
		let rollback_uri = uri.clone();
		let request = FormatRequest::new(
			transaction_id,
			0,
			base,
			uri.clone(),
			None,
			Bytes::from_static(b"candidate"),
		);
		let formatting_registry = registry.clone();
		let formatting = tokio::spawn(async move {
			formatting_registry
				.format_candidate(request, CancellationToken::new())
				.await
		});
		transport.started.notified().await;
		tokio::time::timeout(
			std::time::Duration::from_secs(1),
			registry.register_capabilities(
				binding_handle,
				Bytes::from_static(
					br#"{"registrations":[{"id":"save","method":"textDocument/didSave"}]}"#,
				),
				CancellationToken::new(),
			),
		)
		.await
		.unwrap()
		.unwrap();
		transport.release.notify_one();
		assert_eq!(formatting.await.unwrap().unwrap().content(), &Bytes::from_static(b"candidate"),);
		let second_request = FormatRequest::new(
			transaction_id,
			1,
			rollback_base.clone(),
			uri.clone(),
			None,
			Bytes::from_static(b"candidate-two"),
		);
		let second_registry = registry.clone();
		let second_format = tokio::spawn(async move {
			second_registry
				.format_candidate(second_request, CancellationToken::new())
				.await
		});
		transport.started.notified().await;
		transport.release.notify_one();
		assert_eq!(
			second_format.await.unwrap().unwrap().content(),
			&Bytes::from_static(b"candidate-two"),
		);
		let bound_server = registry.binding(binding_id).await.unwrap().server;
		let (version, _) = bound_server
			.tracked_version_revision(DocumentId::from_bytes([5; 16]))
			.unwrap();
		let diagnostics =
			Bytes::from(format!(r#"{{"uri":"{uri}","version":{version},"diagnostics":[]}}"#));
		assert_eq!(
			registry
				.tag_inbound_event(
					binding_handle,
					"textDocument/publishDiagnostics",
					diagnostics.clone(),
				)
				.await
				.unwrap()
				.revision(),
			None,
		);
		registry
			.revert_uncommitted(
				RevertedDocument::new(transaction_id, 0, rollback_base, rollback_uri, None),
				CancellationToken::new(),
			)
			.await
			.unwrap();
		assert_eq!(bound_server.tracked_version_revision(DocumentId::from_bytes([5; 16])), None,);
		assert_eq!(
			registry
				.tag_inbound_event(binding_handle, "textDocument/publishDiagnostics", diagnostics,)
				.await
				.unwrap()
				.revision(),
			None,
		);
		std::fs::remove_dir_all(root).unwrap();
	}
	#[tokio::test]
	async fn unopened_unformatted_publication_emits_no_lifecycle() {
		static ROOT_SEQUENCE: AtomicU64 = AtomicU64::new(10_000);
		let root = std::env::temp_dir().join(format!(
			"omp-lsp-registry-publish-{}-{}",
			std::process::id(),
			ROOT_SEQUENCE.fetch_add(1, Ordering::Relaxed),
		));
		std::fs::create_dir_all(&root).unwrap();
		let registry =
			LspRegistry::new(DocumentStore::new(ServerConfig::new(&root).unwrap()).unwrap());
		let transport = Arc::new(CountingTransport { messages: AtomicU64::new(0) });
		let server = LspServer::new(
			transport.clone(),
			Bytes::from_static(br#"{"textDocumentSync":{"openClose":true,"change":1,"save":true}}"#),
		)
		.unwrap();
		registry
			.add_binding(
				LspBindingSpec::new("unopened", 0, LspSelector::all()).unwrap(),
				server,
				CancellationToken::new(),
			)
			.await
			.unwrap();
		let content = Bytes::from_static(b"committed");
		let revision = Revision::for_content(1, &content);
		let head = DocumentHead::new(
			DocumentId::from_bytes([8; 16]),
			revision,
			DocumentPresence::Present,
			DocumentKind::Text(None),
			content.len() as u64,
		)
		.unwrap();
		registry
			.publish_committed(
				PublishedDocument::new(
					TransactionId::from_bytes([7; 16]),
					0,
					head,
					content,
					Url::from_file_path(root.join("unopened.txt")).unwrap(),
					None,
				),
				CancellationToken::new(),
			)
			.await
			.unwrap();
		assert_eq!(transport.messages.load(Ordering::Relaxed), 0);
		std::fs::remove_dir_all(root).unwrap();
	}
	#[tokio::test]
	async fn restarted_binding_rejects_old_request_completion() {
		let registry = LspRegistry::new(
			DocumentStore::new(ServerConfig::new(std::env::temp_dir()).unwrap()).unwrap(),
		);
		let transport =
			Arc::new(PendingRequestTransport { started: Notify::new(), release: Notify::new() });
		let binding_id = registry
			.add_binding(
				LspBindingSpec::new("pending", 0, LspSelector::all()).unwrap(),
				LspServer::new(transport.clone(), Bytes::from_static(b"{}")).unwrap(),
				CancellationToken::new(),
			)
			.await
			.unwrap();
		let old_handle = registry.binding_handle(binding_id).await.unwrap();
		let requesting_registry = registry.clone();
		let request = tokio::spawn(async move {
			requesting_registry
				.workspace_request(
					binding_id,
					"workspace/symbol",
					Bytes::from_static(br#"{"query":"x"}"#),
					CancellationToken::new(),
				)
				.await
		});
		transport.started.notified().await;
		registry
			.restart_binding(binding_id, server(), CancellationToken::new())
			.await
			.unwrap();
		transport.release.notify_one();
		assert!(matches!(
			request.await.unwrap(),
			Err(LspRegistryError::BindingRestarted { binding_id: rejected })
				if rejected == binding_id
		));
		assert!(matches!(
			registry
				.publish_inbound_event(
					old_handle,
					"window/logMessage",
					Bytes::from_static(br#"{"type":3,"message":"late"}"#),
				)
				.await,
			Err(LspRegistryError::BindingRestarted { binding_id: rejected })
				if rejected == binding_id
		));
	}

	#[tokio::test]
	async fn opaque_stale_semantic_params_are_not_retried() {
		static ROOT_SEQUENCE: AtomicU64 = AtomicU64::new(30_000);
		let root = std::env::temp_dir().join(format!(
			"omp-lsp-registry-stale-{}-{}",
			std::process::id(),
			ROOT_SEQUENCE.fetch_add(1, Ordering::Relaxed),
		));
		std::fs::create_dir_all(&root).unwrap();
		let path = root.join("file.txt");
		std::fs::write(&path, b"current").unwrap();
		let registry =
			LspRegistry::new(DocumentStore::new(ServerConfig::new(&root).unwrap()).unwrap());
		let transport = Arc::new(RequestCountingTransport { requests: AtomicU64::new(0) });
		let binding_id = registry
			.add_binding(
				LspBindingSpec::new("semantic", 0, LspSelector::all()).unwrap(),
				LspServer::new(
					transport.clone(),
					Bytes::from_static(br#"{"textDocumentSync":{"openClose":true,"change":1}}"#),
				)
				.unwrap(),
				CancellationToken::new(),
			)
			.await
			.unwrap();
		let lease = registry
			.open_document(path, None, CancellationToken::new())
			.await
			.unwrap();
		let stale =
			Revision::for_content(lease.head().revision().sequence(), &Bytes::from_static(b"stale"));
		assert!(matches!(
			registry
				.semantic_request(
					binding_id,
					"textDocument/hover",
					Bytes::from_static(
						br#"{"textDocument":{"uri":"file:///file.txt"},"position":{"line":0,"character":0}}"#,
					),
					lease.lease_id(),
					stale,
					StaleResponsePolicy::RetryOnce,
					CancellationToken::new(),
				)
				.await,
			Err(LspRegistryError::ContentModified { .. })
		));
		assert_eq!(transport.requests.load(Ordering::Relaxed), 0);
		registry
			.close_document(lease.lease_id(), CancellationToken::new())
			.await
			.unwrap();
		std::fs::remove_dir_all(root).unwrap();
	}

	#[tokio::test]
	async fn refresh_failure_compensates_a_prior_open() {
		static ROOT_SEQUENCE: AtomicU64 = AtomicU64::new(40_000);
		let root = std::env::temp_dir().join(format!(
			"omp-lsp-registry-refresh-{}-{}",
			std::process::id(),
			ROOT_SEQUENCE.fetch_add(1, Ordering::Relaxed),
		));
		std::fs::create_dir_all(&root).unwrap();
		let path = root.join("file.txt");
		std::fs::write(&path, b"current").unwrap();
		let registry =
			LspRegistry::new(DocumentStore::new(ServerConfig::new(&root).unwrap()).unwrap());
		let lease = registry
			.open_document(path, None, CancellationToken::new())
			.await
			.unwrap();
		let successful = server();
		let failing = LspServer::new(
			Arc::new(FailingNotifyTransport),
			Bytes::from_static(br#"{"textDocumentSync":{"openClose":true,"change":1}}"#),
		)
		.unwrap();
		{
			let mut state = registry.inner.state.lock();
			state.bindings.insert(LspBindingId(2), Binding {
				id:         LspBindingId(2),
				spec:       LspBindingSpec::new("successful", 10, LspSelector::all()).unwrap(),
				server:     successful.clone(),
				generation: 0,
			});
			state.bindings.insert(LspBindingId(3), Binding {
				id:         LspBindingId(3),
				spec:       LspBindingSpec::new("failing", 5, LspSelector::all()).unwrap(),
				server:     failing,
				generation: 0,
			});
		}
		assert!(
			registry
				.publish_head(lease.head().document_id(), CancellationToken::new())
				.await
				.is_err()
		);
		assert_eq!(
			registry
				.inner
				.state
				.lock()
				.leases
				.get(&lease.lease_id())
				.unwrap()
				.binding_ids
				.len(),
			0
		);
		registry
			.close_document(lease.lease_id(), CancellationToken::new())
			.await
			.unwrap();
		std::fs::remove_dir_all(root).unwrap();
	}

	#[tokio::test]
	async fn failed_restart_keeps_old_binding_and_cleans_staged_leases() {
		static ROOT_SEQUENCE: AtomicU64 = AtomicU64::new(45_000);
		let root = std::env::temp_dir().join(format!(
			"omp-lsp-registry-restart-{}-{}",
			std::process::id(),
			ROOT_SEQUENCE.fetch_add(1, Ordering::Relaxed),
		));
		std::fs::create_dir_all(&root).unwrap();
		let first_path = root.join("first.txt");
		let second_path = root.join("second.txt");
		std::fs::write(&first_path, b"first").unwrap();
		std::fs::write(&second_path, b"second").unwrap();
		let registry =
			LspRegistry::new(DocumentStore::new(ServerConfig::new(&root).unwrap()).unwrap());
		let binding_id = registry
			.add_binding(
				LspBindingSpec::new("restart", 0, LspSelector::all()).unwrap(),
				server(),
				CancellationToken::new(),
			)
			.await
			.unwrap();
		let first = registry
			.open_document(first_path, None, CancellationToken::new())
			.await
			.unwrap();
		let second = registry
			.open_document(second_path, None, CancellationToken::new())
			.await
			.unwrap();
		let old_versions = registry.inner.state.lock().public_versions.clone();
		let replacement = LspServer::new(
			Arc::new(FailSecondNotifyTransport { notifications: AtomicU64::new(0) }),
			Bytes::from_static(br#"{"textDocumentSync":{"openClose":true,"change":1}}"#),
		)
		.unwrap();
		assert!(
			registry
				.restart_binding(binding_id, replacement.clone(), CancellationToken::new(),)
				.await
				.is_err()
		);
		assert_eq!(registry.binding(binding_id).await.unwrap().generation, 0);
		assert_eq!(registry.inner.state.lock().public_versions, old_versions);
		assert_eq!(replacement.tracked_version_revision(first.head().document_id()), None);
		assert_eq!(replacement.tracked_version_revision(second.head().document_id()), None);
		registry
			.close_document(first.lease_id(), CancellationToken::new())
			.await
			.unwrap();
		registry
			.close_document(second.lease_id(), CancellationToken::new())
			.await
			.unwrap();
		std::fs::remove_dir_all(root).unwrap();
	}

	#[tokio::test]
	async fn formatting_acquisition_failure_restores_base_identity_and_mapping() {
		static ROOT_SEQUENCE: AtomicU64 = AtomicU64::new(50_000);
		let root = std::env::temp_dir().join(format!(
			"omp-lsp-registry-format-rollback-{}-{}",
			std::process::id(),
			ROOT_SEQUENCE.fetch_add(1, Ordering::Relaxed),
		));
		std::fs::create_dir_all(&root).unwrap();
		let path = root.join("base.txt");
		std::fs::write(&path, b"base").unwrap();
		let registry =
			LspRegistry::new(DocumentStore::new(ServerConfig::new(&root).unwrap()).unwrap());
		let first_transport = Arc::new(ToggleNotifyTransport {
			fail:   AtomicBool::new(false),
			params: Mutex::new(Vec::new()),
		});
		let second_transport = Arc::new(ToggleNotifyTransport {
			fail:   AtomicBool::new(false),
			params: Mutex::new(Vec::new()),
		});
		let capabilities = Bytes::from_static(
			br#"{"documentFormattingProvider":true,"textDocumentSync":{"openClose":true,"change":1}}"#,
		);
		let first_id = registry
			.add_binding(
				LspBindingSpec::new("first", 10, LspSelector::all()).unwrap(),
				LspServer::new(first_transport.clone(), capabilities.clone()).unwrap(),
				CancellationToken::new(),
			)
			.await
			.unwrap();
		registry
			.add_binding(
				LspBindingSpec::new("second", 5, LspSelector::all()).unwrap(),
				LspServer::new(second_transport.clone(), capabilities).unwrap(),
				CancellationToken::new(),
			)
			.await
			.unwrap();
		let lease = registry
			.open_document(path.clone(), None, CancellationToken::new())
			.await
			.unwrap();
		let base = registry
			.snapshot_from_store(lease.lease_id(), None)
			.await
			.unwrap();
		let base_uri = registry
			.document_uri(lease.head().document_id())
			.await
			.unwrap();
		let candidate_uri = Url::from_file_path(root.join("candidate.rs")).unwrap();
		first_transport.params.lock().clear();
		second_transport.params.lock().clear();
		second_transport.fail.store(true, Ordering::Relaxed);
		let result = registry
			.format_candidate(
				FormatRequest::new(
					TransactionId::from_bytes([51; 16]),
					0,
					base.clone(),
					candidate_uri.clone(),
					Some(LanguageId::new("rust").unwrap()),
					Bytes::from_static(b"candidate"),
				),
				CancellationToken::new(),
			)
			.await;
		assert!(result.is_err());
		let first = registry.binding(first_id).await.unwrap();
		assert_eq!(
			first
				.server
				.tracked_version_revision(lease.head().document_id())
				.unwrap()
				.1,
			base.head().revision()
		);
		{
			let state = registry.inner.state.lock();
			assert!(
				state
					.public_versions
					.get(&(first_id, lease.head().document_id()))
					.unwrap()
					.iter()
					.all(|(uri, _)| uri.as_str() == base_uri.as_str())
			);
		}
		let candidate = candidate_uri.as_str().as_bytes();
		let candidate_absent = {
			let params = first_transport.params.lock();
			params.iter().all(|params| {
				!params
					.windows(candidate.len())
					.any(|window| window == candidate)
			})
		};
		assert!(candidate_absent);
		second_transport.fail.store(false, Ordering::Relaxed);
		registry
			.close_document(lease.lease_id(), CancellationToken::new())
			.await
			.unwrap();
		std::fs::remove_dir_all(root).unwrap();
	}
	#[tokio::test]
	async fn cancelled_close_abandons_a_transport_that_ignores_cancellation() {
		let root = tempfile::tempdir().expect("temporary directory");
		let path = root.path().join("close.txt");
		std::fs::write(&path, b"content").expect("write fixture");
		let store =
			DocumentStore::new(ServerConfig::new(root.path()).expect("server config")).expect("store");
		let registry = LspRegistry::new(store);
		let server = LspServer::new(
			Arc::new(HangingCloseTransport),
			Bytes::from_static(br#"{"textDocumentSync":{"openClose":true,"change":1}}"#),
		)
		.expect("LSP server");
		registry
			.add_binding(
				LspBindingSpec::new("hanging-close", 0, LspSelector::all()).expect("binding"),
				server,
				CancellationToken::new(),
			)
			.await
			.expect("install binding");
		let lease = registry
			.open_document(path, None, CancellationToken::new())
			.await
			.expect("open document");
		let lease_id = lease.lease_id();
		let cancellation = CancellationToken::new();
		let close_registry = registry.clone();
		let close_cancellation = cancellation.clone();
		let close = tokio::spawn(async move {
			close_registry
				.close_document(lease_id, close_cancellation)
				.await
		});
		tokio::task::yield_now().await;
		cancellation.cancel();
		let result = tokio::time::timeout(std::time::Duration::from_secs(1), close)
			.await
			.expect("bounded close")
			.expect("close task");
		assert!(matches!(
			result,
			Err(LspRegistryError::Lsp(LspError::Transport(LspTransportError::Cancelled)))
		));
		assert!(matches!(
			registry
				.close_document(lease_id, CancellationToken::new())
				.await,
			Err(LspRegistryError::UnknownLease { .. })
		));
	}
}
