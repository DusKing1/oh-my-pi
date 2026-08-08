//! Project-scoped document authority and connection-local session state.

use std::{collections::HashMap, sync::Arc};

use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::{
	DocumentId, DocumentStore, EditAdapterRegistry, LeaseId, PathService, Result, ServerConfig,
	lsp_registry::LspRegistry, summary::SummaryService, transaction::TransactionCoordinator,
};

/// Project-scoped document, transaction, path, summary, and LSP authority.
#[derive(Clone)]
pub struct Environment {
	inner: Arc<EnvironmentInner>,
}

struct EnvironmentInner {
	store:        DocumentStore,
	lsp:          LspRegistry,
	transactions: TransactionCoordinator<LspRegistry>,
	paths:        PathService,
	summaries:    SummaryService,
	root_uri:     Url,
	workspace_id: [u8; 16],
	server_epoch: [u8; 16],
}

impl std::fmt::Debug for Environment {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("Environment")
			.field("root_uri", &self.inner.root_uri)
			.finish_non_exhaustive()
	}
}

impl Environment {
	/// Creates one authority for a canonical project root.
	pub fn new(config: ServerConfig) -> Result<Self> {
		let root_uri = config.file_uri(config.environment_root())?;
		let store = DocumentStore::new(config)?;
		let lsp = LspRegistry::new(store.clone());
		let workspace_id = rand::random();
		let server_epoch = rand::random();
		let transactions =
			TransactionCoordinator::with_formatter(store.clone(), server_epoch, lsp.clone());
		let paths = PathService::new(store.clone(), transactions.clone());
		Ok(Self {
			inner: Arc::new(EnvironmentInner {
				store,
				lsp,
				transactions,
				paths,
				summaries: SummaryService::new(),
				root_uri,
				workspace_id,
				server_epoch,
			}),
		})
	}

	/// Starts isolated edit provenance and lease ownership for one connection.
	#[must_use]
	pub fn session(&self) -> EnvironmentSession {
		EnvironmentSession {
			inner: Arc::new(SessionInner {
				environment: self.clone(),
				adapters:    EditAdapterRegistry::with_built_ins(),
				leases:      Mutex::new(HashMap::new()),
			}),
		}
	}

	/// Returns the shared immutable document store.
	#[must_use]
	pub fn store(&self) -> &DocumentStore {
		&self.inner.store
	}

	/// Returns the project-scoped LSP binding registry.
	#[must_use]
	pub fn lsp(&self) -> &LspRegistry {
		&self.inner.lsp
	}

	/// Returns the revisioned transaction coordinator.
	#[must_use]
	pub fn transactions(&self) -> &TransactionCoordinator<LspRegistry> {
		&self.inner.transactions
	}

	/// Returns the actor-aware path service.
	#[must_use]
	pub fn paths(&self) -> &PathService {
		&self.inner.paths
	}

	/// Returns the structural summary service.
	#[must_use]
	pub fn summaries(&self) -> &SummaryService {
		&self.inner.summaries
	}

	/// Returns the canonical project root URI.
	#[must_use]
	pub fn root_uri(&self) -> &Url {
		&self.inner.root_uri
	}

	/// Returns the stable identity of this running project authority.
	#[must_use]
	pub fn workspace_id(&self) -> &[u8; 16] {
		&self.inner.workspace_id
	}

	/// Returns the identity scoping the in-memory transaction outcome ledger.
	#[must_use]
	pub fn server_epoch(&self) -> &[u8; 16] {
		&self.inner.server_epoch
	}

	/// Stops every active document actor after connection sessions are closed.
	pub async fn shutdown(&self) {
		self.inner.store.shutdown().await;
	}
}

/// Connection-local edit provenance, open leases, and cancellation ownership.
#[derive(Clone)]
pub struct EnvironmentSession {
	inner: Arc<SessionInner>,
}

struct SessionInner {
	environment: Environment,
	adapters:    EditAdapterRegistry,
	leases:      Mutex<HashMap<LeaseId, OwnedLease>>,
}

struct OwnedLease {
	document_id:  DocumentId,
	cancellation: CancellationToken,
	events_ready: CancellationToken,
}

impl std::fmt::Debug for EnvironmentSession {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("EnvironmentSession")
			.finish_non_exhaustive()
	}
}

impl EnvironmentSession {
	/// Returns the project authority shared by this connection.
	#[must_use]
	pub fn environment(&self) -> &Environment {
		&self.inner.environment
	}

	/// Returns this connection's isolated edit-format registry.
	#[must_use]
	pub fn edit_adapters(&self) -> &EditAdapterRegistry {
		&self.inner.adapters
	}

	pub(crate) fn own_lease(
		&self,
		lease_id: LeaseId,
		document_id: DocumentId,
		cancellation: CancellationToken,
		events_ready: CancellationToken,
	) {
		self.inner.leases.lock().insert(lease_id, OwnedLease {
			document_id,
			cancellation,
			events_ready,
		});
	}

	pub(crate) fn owns_lease(&self, lease_id: LeaseId) -> bool {
		self.inner.leases.lock().contains_key(&lease_id)
	}

	pub(crate) fn start_lease_events(&self, lease_id: LeaseId) -> bool {
		let leases = self.inner.leases.lock();
		let Some(lease) = leases.get(&lease_id) else {
			return false;
		};
		lease.events_ready.cancel();
		true
	}

	pub(crate) fn release_lease(&self, lease_id: LeaseId) -> bool {
		self
			.inner
			.leases
			.lock()
			.remove(&lease_id)
			.map(|lease| {
				lease.cancellation.cancel();
				lease.events_ready.cancel();
			})
			.is_some()
	}

	pub(crate) fn lease_for_document(&self, document_id: DocumentId) -> Option<LeaseId> {
		self
			.inner
			.leases
			.lock()
			.iter()
			.find_map(|(lease_id, lease)| {
				(lease.document_id == document_id && lease.events_ready.is_cancelled())
					.then_some(*lease_id)
			})
	}

	pub(crate) fn take_leases(&self) -> Vec<LeaseId> {
		let mut leases = self.inner.leases.lock();
		let lease_ids = leases.keys().copied().collect();
		for lease in leases.values() {
			lease.cancellation.cancel();
			lease.events_ready.cancel();
		}
		leases.clear();
		lease_ids
	}
}

#[cfg(test)]
mod tests {
	use tempfile::TempDir;

	use super::*;

	#[test]
	fn lease_events_become_visible_only_after_the_open_response_is_enqueued() {
		let root = TempDir::new().expect("temporary directory");
		let session = Environment::new(ServerConfig::new(root.path()).expect("server config"))
			.expect("environment")
			.session();
		let lease_id = LeaseId::from_bytes([1; 16]);
		let document_id = DocumentId::from_bytes([2; 16]);
		let forwarder = CancellationToken::new();
		let ready = CancellationToken::new();
		session.own_lease(lease_id, document_id, forwarder, ready);

		assert_eq!(session.lease_for_document(document_id), None);
		assert!(session.start_lease_events(lease_id));
		assert_eq!(session.lease_for_document(document_id), Some(lease_id));
	}
}
