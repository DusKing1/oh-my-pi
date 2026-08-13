//! Pluggable conversation persistence with in-memory and SQLite
//! implementations.

use std::{
	hash::{DefaultHasher, Hash, Hasher},
	path::Path,
	sync::Arc,
};

use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Serialize, de::DeserializeOwned};

use super::{
	binding::{PendingServerStateBinding, ServerStateBinding},
	conversation::{
		ConversationError, ConversationState, RevisionNode, SharedState, TurnDraft, delta,
		is_ancestor, revision,
	},
	revision::{CommittedRevision, HistoryDelta},
};
use crate::id::{ConversationId, Revision, TurnId};

/// Append/fork conversation persistence contract.
pub trait ConversationStore<I>: Send + Sync {
	/// Private draft type used to stage an atomic turn.
	type Draft;

	/// Creates an empty conversation and its committed root revision.
	fn create(&self) -> Result<CommittedRevision<I>, ConversationError>;
	/// Returns the current committed head.
	fn head(&self, conversation: &ConversationId)
	-> Result<CommittedRevision<I>, ConversationError>;
	/// Returns one immutable committed revision.
	fn revision(&self, revision: &Revision) -> Result<CommittedRevision<I>, ConversationError>;
	/// Creates a new branch whose head references an existing committed
	/// revision.
	fn fork(&self, at: &Revision) -> Result<ConversationId, ConversationError>;
	/// Privately stages an append at the exact branch head.
	fn begin(
		&self,
		conversation: &ConversationId,
		at: &Revision,
		turn: TurnId,
		append: Arc<[I]>,
	) -> Result<Self::Draft, ConversationError>;
	/// Extracts immutable items after `base`, or a complete replay from the
	/// root.
	fn delta(
		&self,
		base: Option<&Revision>,
		head: &Revision,
	) -> Result<HistoryDelta<I>, ConversationError>;
	/// Returns whether one committed revision is an ancestor of another.
	fn is_ancestor(
		&self,
		ancestor: &Revision,
		descendant: &Revision,
	) -> Result<bool, ConversationError>;
	/// Returns provider-side state atomically attached to the branch's last
	/// successful turn.
	fn server_state(
		&self,
		conversation: &ConversationId,
	) -> Result<Option<ServerStateBinding>, ConversationError>;
}

/// Lock-efficient in-memory append-only conversation store.
pub struct InMemoryConversationStore<I> {
	state: SharedState<I>,
}

impl<I> Default for InMemoryConversationStore<I> {
	fn default() -> Self {
		Self { state: Arc::new(Mutex::new(ConversationState::default())) }
	}
}

impl<I> InMemoryConversationStore<I> {
	/// Creates an empty in-memory store.
	pub fn new() -> Self {
		Self::default()
	}

	/// Returns the number of live private drafts for rollback-law verification
	/// and diagnostics.
	pub fn active_drafts(&self) -> usize {
		self.state.lock().drafts.len()
	}
}

impl<I: PartialEq + Send + Sync + 'static> ConversationStore<I> for InMemoryConversationStore<I> {
	type Draft = TurnDraft<I>;

	fn create(&self) -> Result<CommittedRevision<I>, ConversationError> {
		let mut state = self.state.lock();
		let conversation = state.allocate_conversation();
		let revision = state.revision_for(&conversation, None, None);
		state.revisions.insert(revision.clone(), RevisionNode {
			conversation: conversation.clone(),
			parent:       None,
			turn:         None,
			items:        Arc::from([]),
		});
		state.heads.insert(conversation.clone(), revision.clone());
		Ok(CommittedRevision::new(conversation, revision, None, None, Arc::from([])))
	}

	fn head(
		&self,
		conversation: &ConversationId,
	) -> Result<CommittedRevision<I>, ConversationError> {
		let state = self.state.lock();
		let head = state
			.heads
			.get(conversation)
			.ok_or_else(|| ConversationError::UnknownConversation(conversation.clone()))?;
		revision(&state, head)
	}

	fn revision(&self, id: &Revision) -> Result<CommittedRevision<I>, ConversationError> {
		revision(&self.state.lock(), id)
	}

	fn fork(&self, at: &Revision) -> Result<ConversationId, ConversationError> {
		let mut state = self.state.lock();
		if !state.revisions.contains_key(at) {
			return Err(ConversationError::UnknownRevision(at.clone()));
		}
		let conversation = state.allocate_conversation();
		state.heads.insert(conversation.clone(), at.clone());
		Ok(conversation)
	}

	fn begin(
		&self,
		conversation: &ConversationId,
		at: &Revision,
		turn: TurnId,
		append: Arc<[I]>,
	) -> Result<Self::Draft, ConversationError> {
		let mut state = self.state.lock();
		let head = state
			.heads
			.get(conversation)
			.ok_or_else(|| ConversationError::UnknownConversation(conversation.clone()))?;
		let existing = state
			.turns
			.contains_key(&(conversation.clone(), turn.clone()));
		if head != at && !existing {
			return Err(ConversationError::RevisionConflict {
				expected: head.clone(),
				actual:   at.clone(),
			});
		}
		let draft = state.next_draft;
		state.next_draft += 1;
		state.drafts.insert(draft);
		drop(state);
		Ok(TurnDraft {
			state: Arc::clone(&self.state),
			draft,
			conversation: conversation.clone(),
			base: at.clone(),
			turn,
			items: Some(append),
			binding: None,
		})
	}

	fn delta(
		&self,
		base: Option<&Revision>,
		head: &Revision,
	) -> Result<HistoryDelta<I>, ConversationError> {
		delta(&self.state.lock(), base, head)
	}

	fn is_ancestor(
		&self,
		ancestor: &Revision,
		descendant: &Revision,
	) -> Result<bool, ConversationError> {
		let state = self.state.lock();
		if !state.revisions.contains_key(ancestor) {
			return Err(ConversationError::UnknownRevision(ancestor.clone()));
		}
		if !state.revisions.contains_key(descendant) {
			return Err(ConversationError::UnknownRevision(descendant.clone()));
		}
		Ok(is_ancestor(&state, ancestor, descendant))
	}

	fn server_state(
		&self,
		conversation: &ConversationId,
	) -> Result<Option<ServerStateBinding>, ConversationError> {
		let state = self.state.lock();
		if !state.heads.contains_key(conversation) {
			return Err(ConversationError::UnknownConversation(conversation.clone()));
		}
		Ok(state.bindings.get(conversation).cloned())
	}
}

/// SQLite-backed append-only conversation store suitable for process restarts.
pub struct SqliteConversationStore<I> {
	connection: Arc<Mutex<Connection>>,
	marker:     std::marker::PhantomData<fn() -> I>,
}

impl<I> SqliteConversationStore<I> {
	/// Opens or creates a SQLite conversation database and validates its schema.
	pub fn open(path: impl AsRef<Path>) -> Result<Self, ConversationError> {
		let connection = Connection::open(path).map_err(|_| ConversationError::Persistence)?;
		Self::from_connection(connection)
	}

	/// Creates an isolated SQLite conversation store in memory.
	pub fn open_in_memory() -> Result<Self, ConversationError> {
		let connection = Connection::open_in_memory().map_err(|_| ConversationError::Persistence)?;
		Self::from_connection(connection)
	}

	fn from_connection(connection: Connection) -> Result<Self, ConversationError> {
		connection
			.execute_batch(
				"PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;
			CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, head TEXT NOT NULL);
			CREATE TABLE IF NOT EXISTS revisions (id TEXT PRIMARY KEY, conversation TEXT NOT NULL, parent \
				 TEXT, turn TEXT, items BLOB NOT NULL);
			CREATE UNIQUE INDEX IF NOT EXISTS revisions_turn ON revisions(conversation, turn) WHERE turn IS \
				 NOT NULL;
			CREATE TABLE IF NOT EXISTS bindings (conversation TEXT PRIMARY KEY, binding BLOB NOT NULL);",
			)
			.map_err(|_| ConversationError::Persistence)?;
		Ok(Self {
			connection: Arc::new(Mutex::new(connection)),
			marker:     std::marker::PhantomData,
		})
	}
}

/// A private SQLite turn transaction staged entirely outside committed tables.
#[must_use = "dropping a SQLite turn draft rolls it back"]
pub struct SqliteTurnDraft<I> {
	connection:   Arc<Mutex<Connection>>,
	conversation: ConversationId,
	base:         Revision,
	turn:         TurnId,
	items:        Option<Arc<[I]>>,
	binding:      Option<PendingServerStateBinding>,
}

impl<I> SqliteTurnDraft<I> {
	/// Associates a successful provider-state capture with this turn's atomic
	/// commit.
	pub fn capture_server_state(
		&mut self,
		binding: PendingServerStateBinding,
	) -> Result<(), ConversationError> {
		if binding.conversation != self.conversation {
			return Err(ConversationError::CorruptStore);
		}
		self.binding = Some(binding);
		Ok(())
	}

	/// Appends additional private items before atomic commit.
	pub fn append(&mut self, items: Arc<[I]>)
	where
		I: Clone,
	{
		let current = self.items.take().unwrap_or_default();
		self.items = Some(
			current
				.iter()
				.cloned()
				.chain(items.iter().cloned())
				.collect::<Vec<_>>()
				.into(),
		);
	}
}

impl<I: Serialize + DeserializeOwned> SqliteTurnDraft<I> {
	/// Atomically commits a successful turn and its captured provider state.
	pub fn commit_successful_turn(
		mut self,
		binding: PendingServerStateBinding,
	) -> Result<CommittedRevision<I>, ConversationError>
	where
		I: Serialize + DeserializeOwned,
	{
		self.capture_server_state(binding)?;
		self.commit()
	}

	/// Atomically commits history and provider state, idempotently by
	/// conversation and turn ID.
	pub fn commit(mut self) -> Result<CommittedRevision<I>, ConversationError> {
		let items = self.items.take().ok_or(ConversationError::CorruptStore)?;
		let encoded =
			postcard::to_allocvec(items.as_ref()).map_err(|_| ConversationError::CorruptStore)?;
		let mut connection = self.connection.lock();
		let transaction = connection
			.transaction_with_behavior(TransactionBehavior::Immediate)
			.map_err(|_| ConversationError::Persistence)?;
		if let Some(existing) = transaction
			.query_row(
				"SELECT id,parent,items FROM revisions WHERE conversation=?1 AND turn=?2",
				params![self.conversation.as_str(), self.turn.as_str()],
				|row| {
					Ok((
						row.get::<_, String>(0)?,
						row.get::<_, Option<String>>(1)?,
						row.get::<_, Vec<u8>>(2)?,
					))
				},
			)
			.optional()
			.map_err(|_| ConversationError::Persistence)?
		{
			if existing.1.as_deref() != Some(self.base.as_str()) || existing.2 != encoded {
				return Err(ConversationError::TurnConflict(self.turn));
			}
			let decoded: Arc<[I]> =
				postcard::from_bytes(&existing.2).map_err(|_| ConversationError::CorruptStore)?;
			transaction
				.commit()
				.map_err(|_| ConversationError::Persistence)?;
			return Ok(CommittedRevision::new(
				self.conversation,
				Revision::new(existing.0),
				Some(self.base),
				Some(self.turn),
				decoded,
			));
		}
		let current: String = transaction
			.query_row(
				"SELECT head FROM conversations WHERE id=?1",
				[self.conversation.as_str()],
				|row| row.get(0),
			)
			.map_err(|_| ConversationError::UnknownConversation(self.conversation.clone()))?;
		if current != self.base.as_str() {
			return Err(ConversationError::RevisionConflict {
				expected: Revision::new(current),
				actual:   self.base,
			});
		}
		let revision = sqlite_revision_id(&self.conversation, Some(&self.base), Some(&self.turn));
		transaction
			.execute(
				"INSERT INTO revisions(id,conversation,parent,turn,items) VALUES (?1,?2,?3,?4,?5)",
				params![
					revision.as_str(),
					self.conversation.as_str(),
					self.base.as_str(),
					self.turn.as_str(),
					encoded
				],
			)
			.map_err(|_| ConversationError::Persistence)?;
		transaction
			.execute("UPDATE conversations SET head=?1 WHERE id=?2 AND head=?3", params![
				revision.as_str(),
				self.conversation.as_str(),
				self.base.as_str()
			])
			.map_err(|_| ConversationError::Persistence)?;
		if let Some(binding) = self.binding.take() {
			let binding = binding.commit(revision.clone());
			let encoded =
				postcard::to_allocvec(&binding).map_err(|_| ConversationError::CorruptStore)?;
			transaction
				.execute(
					"INSERT INTO bindings(conversation,binding) VALUES (?1,?2) ON \
					 CONFLICT(conversation) DO UPDATE SET binding=excluded.binding",
					params![self.conversation.as_str(), encoded],
				)
				.map_err(|_| ConversationError::Persistence)?;
		}
		transaction
			.commit()
			.map_err(|_| ConversationError::Persistence)?;
		Ok(CommittedRevision::new(
			self.conversation,
			revision,
			Some(self.base),
			Some(self.turn),
			items,
		))
	}
}

fn sqlite_next_conversation_id(connection: &Connection) -> Result<i64, ConversationError> {
	connection
		.query_row("SELECT COALESCE(MAX(rowid),0)+1 FROM conversations", [], |row| row.get(0))
		.map_err(|_| ConversationError::Persistence)
}

fn sqlite_revision_id(
	conversation: &ConversationId,
	parent: Option<&Revision>,
	turn: Option<&TurnId>,
) -> Revision {
	let mut hasher = DefaultHasher::new();
	conversation.hash(&mut hasher);
	parent.hash(&mut hasher);
	turn.hash(&mut hasher);
	Revision::new(format!("revision-{:016x}", hasher.finish()))
}

impl<I: Serialize + DeserializeOwned + Send + Sync + 'static> ConversationStore<I>
	for SqliteConversationStore<I>
{
	type Draft = SqliteTurnDraft<I>;

	fn create(&self) -> Result<CommittedRevision<I>, ConversationError> {
		let mut connection = self.connection.lock();
		let transaction = connection
			.transaction_with_behavior(TransactionBehavior::Immediate)
			.map_err(|_| ConversationError::Persistence)?;
		let conversation = ConversationId::new(format!(
			"conversation-{}",
			sqlite_next_conversation_id(&transaction)?
		));
		let revision = sqlite_revision_id(&conversation, None, None);
		let empty: Arc<[I]> = Arc::from([]);
		let items =
			postcard::to_allocvec(empty.as_ref()).map_err(|_| ConversationError::CorruptStore)?;
		transaction
			.execute(
				"INSERT INTO revisions(id,conversation,parent,turn,items) VALUES (?1,?2,NULL,NULL,?3)",
				params![revision.as_str(), conversation.as_str(), items],
			)
			.map_err(|_| ConversationError::Persistence)?;
		transaction
			.execute("INSERT INTO conversations(id,head) VALUES (?1,?2)", params![
				conversation.as_str(),
				revision.as_str()
			])
			.map_err(|_| ConversationError::Persistence)?;
		transaction
			.commit()
			.map_err(|_| ConversationError::Persistence)?;
		Ok(CommittedRevision::new(conversation, revision, None, None, empty))
	}

	fn head(
		&self,
		conversation: &ConversationId,
	) -> Result<CommittedRevision<I>, ConversationError> {
		let connection = self.connection.lock();
		let head: String = connection
			.query_row("SELECT head FROM conversations WHERE id=?1", [conversation.as_str()], |row| {
				row.get(0)
			})
			.map_err(|_| ConversationError::UnknownConversation(conversation.clone()))?;
		sqlite_revision(&connection, &Revision::new(head))
	}

	fn revision(&self, revision: &Revision) -> Result<CommittedRevision<I>, ConversationError> {
		sqlite_revision(&self.connection.lock(), revision)
	}

	fn fork(&self, at: &Revision) -> Result<ConversationId, ConversationError> {
		let connection = self.connection.lock();
		let exists: bool = connection
			.query_row("SELECT EXISTS(SELECT 1 FROM revisions WHERE id=?1)", [at.as_str()], |row| {
				row.get(0)
			})
			.map_err(|_| ConversationError::Persistence)?;
		if !exists {
			return Err(ConversationError::UnknownRevision(at.clone()));
		}
		let conversation =
			ConversationId::new(format!("conversation-{}", sqlite_next_conversation_id(&connection)?));
		connection
			.execute("INSERT INTO conversations(id,head) VALUES (?1,?2)", params![
				conversation.as_str(),
				at.as_str()
			])
			.map_err(|_| ConversationError::Persistence)?;
		Ok(conversation)
	}

	fn begin(
		&self,
		conversation: &ConversationId,
		at: &Revision,
		turn: TurnId,
		append: Arc<[I]>,
	) -> Result<Self::Draft, ConversationError> {
		let connection = self.connection.lock();
		let head: String = connection
			.query_row("SELECT head FROM conversations WHERE id=?1", [conversation.as_str()], |row| {
				row.get(0)
			})
			.map_err(|_| ConversationError::UnknownConversation(conversation.clone()))?;
		let existing: bool = connection
			.query_row(
				"SELECT EXISTS(SELECT 1 FROM revisions WHERE conversation=?1 AND turn=?2)",
				params![conversation.as_str(), turn.as_str()],
				|row| row.get(0),
			)
			.map_err(|_| ConversationError::Persistence)?;
		if head != at.as_str() && !existing {
			return Err(ConversationError::RevisionConflict {
				expected: Revision::new(head),
				actual:   at.clone(),
			});
		}
		drop(connection);
		Ok(SqliteTurnDraft {
			connection: Arc::clone(&self.connection),
			conversation: conversation.clone(),
			base: at.clone(),
			turn,
			items: Some(append),
			binding: None,
		})
	}

	fn delta(
		&self,
		base: Option<&Revision>,
		head: &Revision,
	) -> Result<HistoryDelta<I>, ConversationError> {
		let connection = self.connection.lock();
		let mut cursor = Some(head.clone());
		let mut segments = Vec::new();
		while let Some(revision) = cursor.as_ref() {
			if base == Some(revision) {
				break;
			}
			let committed: CommittedRevision<I> = sqlite_revision(&connection, revision)?;
			segments.push(committed.shared_items());
			cursor = committed.parent().cloned();
		}
		if let (Some(base), None) = (base, cursor.as_ref()) {
			return Err(ConversationError::RevisionConflict {
				expected: head.clone(),
				actual:   base.clone(),
			});
		}
		segments.reverse();
		Ok(HistoryDelta::new(base.cloned(), head.clone(), segments))
	}

	fn is_ancestor(
		&self,
		ancestor: &Revision,
		descendant: &Revision,
	) -> Result<bool, ConversationError> {
		let connection = self.connection.lock();
		let mut cursor = Some(descendant.clone());
		while let Some(revision) = cursor {
			if &revision == ancestor {
				return Ok(true);
			}
			cursor = sqlite_revision::<I>(&connection, &revision)?
				.parent()
				.cloned();
		}
		Ok(false)
	}

	fn server_state(
		&self,
		conversation: &ConversationId,
	) -> Result<Option<ServerStateBinding>, ConversationError> {
		let connection = self.connection.lock();
		let bytes: Option<Vec<u8>> = connection
			.query_row(
				"SELECT binding FROM bindings WHERE conversation=?1",
				[conversation.as_str()],
				|row| row.get(0),
			)
			.optional()
			.map_err(|_| ConversationError::Persistence)?;
		bytes
			.map(|bytes| postcard::from_bytes(&bytes).map_err(|_| ConversationError::CorruptStore))
			.transpose()
	}
}

fn sqlite_revision<I: DeserializeOwned>(
	connection: &Connection,
	revision: &Revision,
) -> Result<CommittedRevision<I>, ConversationError> {
	let row = connection
		.query_row(
			"SELECT conversation,parent,turn,items FROM revisions WHERE id=?1",
			[revision.as_str()],
			|row| {
				Ok((
					row.get::<_, String>(0)?,
					row.get::<_, Option<String>>(1)?,
					row.get::<_, Option<String>>(2)?,
					row.get::<_, Vec<u8>>(3)?,
				))
			},
		)
		.optional()
		.map_err(|_| ConversationError::Persistence)?
		.ok_or_else(|| ConversationError::UnknownRevision(revision.clone()))?;
	let items = postcard::from_bytes(&row.3).map_err(|_| ConversationError::CorruptStore)?;
	Ok(CommittedRevision::new(
		ConversationId::new(row.0),
		revision.clone(),
		row.1.map(Revision::new),
		row.2.map(TurnId::new),
		items,
	))
}
