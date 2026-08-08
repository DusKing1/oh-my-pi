//! Atomic, optimistic-concurrency storage for gateway-held conversations.
//!
//! Precondition, mutation intent, and turn idempotency deliberately remain
//! separate: a revision says whether the caller saw this exact history, a
//! [`ThreadDelta`] says how that history should change, and a `turn_id` says
//! whether the logical operation has already started or completed.

use std::{
	collections::{BTreeMap, VecDeque},
	mem,
	sync::{Arc, Weak},
	time::{Duration, Instant},
};

use blake3::Hasher;
use bytes::Bytes;
use omp_core::SmolStr;
use omp_llm_types::{
	Accuracy, BlobPart, ChatOutcome, ContextRef, Item, ItemKind, Message, Part, Props, Revision,
	Role, StopReason, Thinking, Thread, ThreadDelta, ToolCall, ToolResult, TurnError, TurnErrorKind,
	TurnEvent, Unsupported, UnsupportedAction, Usage, ids::CallId,
};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use serde_json::Value;
use smallvec::SmallVec;
use thiserror::Error;

const DEFAULT_CONTEXT_TTL: Duration = Duration::from_mins(30);
const DEFAULT_CONTEXT_CAPACITY: usize = 4_096;
const DEFAULT_DEDUP_CAPACITY: usize = 8_192;

/// Resource bounds for a [`ContextStore`].
#[derive(Clone, Copy, Debug)]
pub struct ContextStoreConfig {
	/// Idle lifetime of a retained context. In-flight contexts are never
	/// TTL-evicted.
	pub context_ttl:      Duration,
	/// Maximum number of retained contexts.
	pub context_capacity: usize,
	/// Hard bound on in-flight plus recently committed `turn_id` records.
	///
	/// Once a committed record falls out of this window, another use of that id
	/// is treated as a new logical turn. Callers therefore retry promptly and do
	/// not use the deduplication window as durable outcome storage.
	pub dedup_capacity:   usize,
}

impl Default for ContextStoreConfig {
	fn default() -> Self {
		Self {
			context_ttl:      DEFAULT_CONTEXT_TTL,
			context_capacity: DEFAULT_CONTEXT_CAPACITY,
			dedup_capacity:   DEFAULT_DEDUP_CAPACITY,
		}
	}
}

/// Stateful or stateless input admitted by [`ContextStore::begin`].
#[derive(Clone, Debug, PartialEq)]
pub enum BeginInput {
	/// A delta guarded by an exact revision precondition.
	Incremental {
		/// Existing held context and the caller's exact view of it.
		context: ContextRef,
		/// Explicit mutation intent, applied only if the turn commits.
		delta:   ThreadDelta,
	},
	/// A full thread, optionally retained under a new context id on commit.
	Seed {
		/// Context to create, or `None` for a stateless turn.
		context_id: Option<SmolStr>,
		/// Complete prompt thread.
		thread:     Thread,
	},
}

/// Result of idempotent turn admission.
pub enum Begin {
	/// This caller owns newly admitted work.
	Started(TurnGuard),
	/// The same logical turn is already running; consume its existing stream.
	Attached(TurnAttachment),
	/// The same logical turn committed within the bounded deduplication window.
	/// The caller emits `Accepted { replay: true }` and replays this outcome.
	Replay {
		/// Authoritative committed outcome, without applying the delta again.
		outcome:  ChatOutcome,
		/// BLAKE3 digest of the authoritative outcome.
		digest:   [u8; 32],
		/// Post-commit context revision, absent for a stateless turn.
		revision: Option<Revision>,
	},
}

/// A lossless attachment to an already-running turn event stream.
pub struct TurnAttachment {
	pending:  VecDeque<TurnEvent>,
	receiver: flume::Receiver<TurnEvent>,
}

impl TurnAttachment {
	/// Receives the next event, first replaying events emitted before
	/// attachment.
	pub async fn recv(&mut self) -> Option<TurnEvent> {
		if let Some(event) = self.pending.pop_front() {
			return Some(event);
		}
		self.receiver.recv_async().await.ok()
	}
}

/// Per-session upstream state committed atomically with a successful turn.
///
/// Production credential identity is committed transactionally by the
/// selection stack under `prompt_cache_key`; every dispatch resolves a fresh
/// canonical lease generation. It is deliberately not duplicated here, where
/// a cancelled turn could otherwise retain a stale generation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SessionAffinity {
	/// Canonical provider/model route that owns every field below.
	pub route_id:                     Option<SmolStr>,
	/// Durable credential selected for a legacy credential-bound route.
	pub credential_id:                Option<SmolStr>,
	/// Provider prompt-cache identity associated with the committed prefix.
	pub prompt_cache_key:             Option<SmolStr>,
	/// Last authoritative `OpenAI` Responses id used for stateful chaining.
	pub previous_response_id:         Option<SmolStr>,
	/// Number of canonical items committed into `previous_response_id`.
	///
	/// This is an input boundary, not a truncated thread: the gateway retains
	/// the complete canonical history and uses the count only to shape the next
	/// Responses request as a delta.
	pub previous_response_item_count: Option<usize>,
	/// Opaque replay-safe Codex transport state.
	pub codex_state:                  Bytes,
}

/// Failures produced by context admission and atomic commit.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ContextError {
	/// The caller's expected head or integrity token did not exactly match.
	#[error("context revision conflict")]
	Conflict {
		/// Actual server revision returned to the caller.
		actual: Revision,
	},
	/// The context is unknown or was evicted and must be reseeded in full.
	#[error("context unknown or evicted; full thread required")]
	NeedFull,
	/// Another turn currently owns this context's serialization slot.
	#[error("context already has an in-flight turn")]
	Busy,
	/// A full seed or fork tried to replace a live context.
	#[error("context already exists")]
	AlreadyExists,
	/// The explicit truncation point was beyond the preconditioned head.
	#[error("truncate target {at} is beyond head {head}")]
	InvalidTruncate {
		/// Requested branch or truncation sequence.
		at:   u64,
		/// Current preconditioned head.
		head: u64,
	},
	/// A `turn_id` was reused for different context input.
	#[error("turn_id was reused with different input")]
	TurnIdReuse,
	/// Every bounded deduplication slot is occupied by an in-flight turn.
	#[error("turn_id deduplication window is full")]
	DedupWindowFull,
	/// No idle context could be evicted to respect the configured capacity.
	#[error("context capacity is full")]
	ContextCapacity,
	/// Output was supplied both incrementally and in the terminal outcome.
	#[error("turn output was supplied twice")]
	OutputAlreadyAccumulated,
	/// The guard no longer owns a live admitted turn.
	#[error("turn guard is no longer active")]
	InactiveGuard,
}

impl From<ContextError> for TurnError {
	fn from(error: ContextError) -> Self {
		let (kind, detail, actual) = match error {
			ContextError::Conflict { actual } => {
				(TurnErrorKind::Conflict, SmolStr::from("context revision conflict"), Some(actual))
			},
			ContextError::NeedFull => (
				TurnErrorKind::NeedFull,
				SmolStr::from("context unknown or evicted; full thread required"),
				None,
			),
			ContextError::Busy => (
				TurnErrorKind::Overloaded,
				SmolStr::from("context already has an in-flight turn"),
				None,
			),
			ContextError::AlreadyExists => {
				(TurnErrorKind::Conflict, SmolStr::from("context already exists"), None)
			},
			ContextError::InvalidTruncate { .. } => (
				TurnErrorKind::Conflict,
				SmolStr::from("truncate target is beyond the preconditioned head"),
				None,
			),
			ContextError::TurnIdReuse => (
				TurnErrorKind::Conflict,
				SmolStr::from("turn_id was reused with different input"),
				None,
			),
			ContextError::DedupWindowFull => {
				(TurnErrorKind::Overloaded, SmolStr::from("turn_id deduplication window is full"), None)
			},
			ContextError::ContextCapacity => {
				(TurnErrorKind::Overloaded, SmolStr::from("context capacity is full"), None)
			},
			ContextError::OutputAlreadyAccumulated => {
				(TurnErrorKind::Upstream, SmolStr::from("turn output was supplied twice"), None)
			},
			ContextError::InactiveGuard => {
				(TurnErrorKind::Upstream, SmolStr::from("turn guard is no longer active"), None)
			},
		};
		Self::builder()
			.kind(kind)
			.detail(detail)
			.maybe_actual(actual)
			.unsupported(Vec::new())
			.retry_after_ms(0)
			.build()
	}
}

/// In-memory context storage with persistent-history copy-on-write branches.
#[derive(Clone)]
pub struct ContextStore {
	inner: Arc<Inner>,
}

struct Inner {
	config: ContextStoreConfig,
	state:  Mutex<StoreState>,
}

#[derive(Default)]
struct StoreState {
	contexts:         FxHashMap<SmolStr, HeldContext>,
	pending_contexts: FxHashMap<SmolStr, SmolStr>,
	turns:            FxHashMap<SmolStr, TurnRecord>,
	committed_order:  VecDeque<SmolStr>,
}

struct HeldContext {
	tip:        History,
	affinities: BTreeMap<SmolStr, SessionAffinity>,
	in_flight:  Option<SmolStr>,
	last_used:  Instant,
}

type History = Option<Arc<HistoryNode>>;

struct HistoryNode {
	previous: History,
	item:     Item,
	revision: Revision,
}

struct LiveTurn {
	stream: Mutex<LiveStream>,
}

#[derive(Default)]
struct LiveStream {
	events:      Vec<TurnEvent>,
	subscribers: Vec<flume::Sender<TurnEvent>>,
}

impl LiveTurn {
	fn new() -> Self {
		Self { stream: Mutex::new(LiveStream::default()) }
	}

	fn attach(&self) -> TurnAttachment {
		let (sender, receiver) = flume::unbounded();
		let mut stream = self.stream.lock();
		let pending = stream.events.iter().cloned().collect();
		stream.subscribers.push(sender);
		TurnAttachment { pending, receiver }
	}

	fn publish(&self, event: TurnEvent) {
		let mut stream = self.stream.lock();
		stream.events.push(event.clone());
		stream
			.subscribers
			.retain(|subscriber| subscriber.send(event.clone()).is_ok());
	}

	fn close(&self) {
		self.stream.lock().subscribers.clear();
	}
}

enum TurnRecord {
	InFlight {
		input: BeginInput,
		live:  Arc<LiveTurn>,
	},
	Committed {
		input:    BeginInput,
		outcome:  Box<ChatOutcome>,
		digest:   [u8; 32],
		revision: Option<Revision>,
	},
}

/// Drop-guard owning an admitted turn's uncommitted working snapshot.
///
/// The held context is untouched until [`TurnGuard::commit`]. Dropping this
/// value releases the serialization slot and idempotency record, which makes
/// cancellation and every early error a structural rollback rather than an
/// optional cleanup step.
pub struct TurnGuard {
	store:       Weak<Inner>,
	turn_id:     SmolStr,
	context_id:  Option<SmolStr>,
	base:        History,
	thread:      Thread,
	input_start: usize,
	affinities:  BTreeMap<SmolStr, SessionAffinity>,
	output:      Vec<Item>,
	live:        Arc<LiveTurn>,
	retained:    bool,
	active:      bool,
}

impl TurnGuard {
	/// Returns the post-delta working thread used to construct the upstream
	/// request. It is a private snapshot; mutating the store remains impossible
	/// before commit.
	#[must_use]
	pub const fn thread(&self) -> &Thread {
		&self.thread
	}

	/// Returns the retained-prefix item count before this turn's append.
	///
	/// Continuation affinity is valid only when its committed boundary does not
	/// extend past this prefix (for example, after a branch truncation).
	#[must_use]
	pub(crate) const fn input_start(&self) -> usize {
		self.input_start
	}

	/// Returns mutable session affinity staged for atomic commit.
	pub fn affinity(&mut self, session_key: &str) -> &mut SessionAffinity {
		self.affinities.entry(session_key.into()).or_default()
	}

	/// Publishes a non-terminal event to this turn and all idempotent
	/// attachments.
	pub fn publish(&self, event: TurnEvent) {
		self.live.publish(event);
	}

	/// Adds one canonical output item to the pending atomic commit.
	pub fn push_output(&mut self, item: Item) {
		self.output.push(item);
	}

	/// Adds canonical output items to the pending atomic commit.
	pub fn extend_output(&mut self, items: impl IntoIterator<Item = Item>) {
		self.output.extend(items);
	}

	/// Atomically applies truncate, input append, and model output, then caches
	/// the outcome.
	///
	/// Output may be accumulated through [`Self::push_output`] or supplied in
	/// `outcome.output`, but not both. The returned outcome contains
	/// gateway-assigned sequences and the authoritative post-commit revision.
	///
	/// # Errors
	///
	/// Returns [`ContextError::InactiveGuard`] if the context was explicitly
	/// dropped, or another context error if a configured resource bound
	/// prevents commit.
	pub fn commit(self, outcome: ChatOutcome) -> Result<ChatOutcome, ContextError> {
		let Some(store) = self.store.upgrade() else {
			return Err(ContextError::InactiveGuard);
		};
		commit_guard(&store, self, outcome)
	}
}

impl Drop for TurnGuard {
	fn drop(&mut self) {
		if !self.active {
			return;
		}
		let Some(store) = self.store.upgrade() else {
			return;
		};
		let mut state = store.state.lock();
		remove_live_turn(&mut state, &self.turn_id, &self.live);
		if let Some(context_id) = &self.context_id {
			if self.retained {
				if let Some(context) = state.contexts.get_mut(context_id)
					&& context.in_flight.as_ref() == Some(&self.turn_id)
				{
					context.in_flight = None;
				}
			} else if state.pending_contexts.get(context_id) == Some(&self.turn_id) {
				state.pending_contexts.remove(context_id);
			}
		}
	}
}

impl Default for ContextStore {
	fn default() -> Self {
		Self::new(ContextStoreConfig::default())
	}
}

impl ContextStore {
	/// Creates an empty store with explicit TTL and bounded retention windows.
	#[must_use]
	pub fn new(config: ContextStoreConfig) -> Self {
		Self { inner: Arc::new(Inner { config, state: Mutex::new(StoreState::default()) }) }
	}

	/// Seeds a held context outside a turn, assigning dense sequences and a
	/// revision chain.
	///
	/// This is the administrative counterpart of a successful full-seed turn and
	/// is useful when the caller has already established the complete
	/// authoritative state.
	///
	/// # Errors
	///
	/// Returns [`ContextError::AlreadyExists`] for a live id or
	/// [`ContextError::ContextCapacity`] when no idle context can be evicted.
	pub fn seed(
		&self,
		context_id: impl Into<SmolStr>,
		thread: Thread,
	) -> Result<Revision, ContextError> {
		let context_id = context_id.into();
		let mut state = self.inner.state.lock();
		prune_expired(&mut state, self.inner.config.context_ttl, Instant::now());
		if state.contexts.contains_key(&context_id)
			|| state.pending_contexts.contains_key(&context_id)
		{
			return Err(ContextError::AlreadyExists);
		}
		make_context_room(&mut state, self.inner.config.context_capacity)?;
		let tip = append_items(None, thread.items).0;
		let revision = history_revision(&tip);
		state.contexts.insert(context_id, HeldContext {
			tip,
			affinities: BTreeMap::new(),
			in_flight: None,
			last_used: Instant::now(),
		});
		Ok(revision)
	}

	/// Admits a turn after idempotency lookup and, for incremental input, an
	/// exact precondition.
	///
	/// Existing `turn_id` records are checked before the context precondition
	/// because a committed retry necessarily carries the old pre-commit
	/// revision. For a new id, head and token are compared before `truncate_to`
	/// is read or validated.
	///
	/// # Errors
	///
	/// Returns [`ContextError::Conflict`] for a stale revision,
	/// [`ContextError::NeedFull`] for unknown state, or a resource/serialization
	/// error.
	pub fn begin(&self, turn_id: SmolStr, input: BeginInput) -> Result<Begin, ContextError> {
		let mut state = self.inner.state.lock();
		if let Some(record) = state.turns.get(&turn_id) {
			return match record {
				TurnRecord::InFlight { input: original, live } if original == &input => {
					Ok(Begin::Attached(live.attach()))
				},
				TurnRecord::Committed { input: original, outcome, digest, revision }
					if original == &input =>
				{
					Ok(Begin::Replay {
						outcome:  (**outcome).clone(),
						digest:   *digest,
						revision: revision.clone(),
					})
				},
				_ => Err(ContextError::TurnIdReuse),
			};
		}

		let now = Instant::now();
		prune_expired(&mut state, self.inner.config.context_ttl, now);
		let live = Arc::new(LiveTurn::new());
		let guard = match &input {
			BeginInput::Incremental { context: context_ref, delta } => {
				let Some(context) = state.contexts.get(&context_ref.context_id) else {
					return Err(ContextError::NeedFull);
				};
				let actual = history_revision(&context.tip);
				if context_ref.expected != actual {
					return Err(ContextError::Conflict { actual });
				}
				let truncate_to = delta.truncate_to.unwrap_or(actual.head);
				if truncate_to > actual.head {
					return Err(ContextError::InvalidTruncate { at: truncate_to, head: actual.head });
				}
				if context.in_flight.is_some() {
					return Err(ContextError::Busy);
				}
				ensure_turn_room(&mut state, self.inner.config.dedup_capacity)?;
				let context = state
					.contexts
					.get_mut(&context_ref.context_id)
					.expect("context checked above");
				let base = history_at(&context.tip, truncate_to);
				let mut thread = materialize(&base);
				let input_start = thread.items.len();
				assign_for_working_thread(&mut thread.items, delta.append.clone());
				context.in_flight = Some(turn_id.clone());
				context.last_used = now;
				TurnGuard {
					store: Arc::downgrade(&self.inner),
					turn_id: turn_id.clone(),
					context_id: Some(context_ref.context_id.clone()),
					base,
					thread,
					input_start,
					affinities: context.affinities.clone(),
					output: Vec::new(),
					live: live.clone(),
					retained: true,
					active: true,
				}
			},
			BeginInput::Seed { context_id, thread } => {
				if let Some(context_id) = context_id {
					if let Some(context) = state.contexts.get(context_id) {
						return Err(ContextError::Conflict { actual: history_revision(&context.tip) });
					}
					if state.pending_contexts.contains_key(context_id) {
						return Err(ContextError::Busy);
					}
				}
				ensure_turn_room(&mut state, self.inner.config.dedup_capacity)?;
				let mut working = Thread::default();
				assign_for_working_thread(&mut working.items, thread.items.clone());
				if let Some(context_id) = context_id {
					state
						.pending_contexts
						.insert(context_id.clone(), turn_id.clone());
				}
				TurnGuard {
					store:       Arc::downgrade(&self.inner),
					turn_id:     turn_id.clone(),
					context_id:  context_id.clone(),
					base:        None,
					thread:      working,
					input_start: 0,
					affinities:  BTreeMap::new(),
					output:      Vec::new(),
					live:        live.clone(),
					retained:    false,
					active:      true,
				}
			},
		};
		state
			.turns
			.insert(turn_id, TurnRecord::InFlight { input, live });
		Ok(Begin::Started(guard))
	}

	/// Commits a guard owned by this store.
	///
	/// This is equivalent to [`TurnGuard::commit`] and is provided for engines
	/// that keep storage ownership explicit at their commit boundary.
	///
	/// # Errors
	///
	/// Returns a context error if the guard is inactive or output is duplicated.
	pub fn commit(
		&self,
		guard: TurnGuard,
		outcome: ChatOutcome,
	) -> Result<ChatOutcome, ContextError> {
		let Some(owner) = guard.store.upgrade() else {
			return Err(ContextError::InactiveGuard);
		};
		if !Arc::ptr_eq(&owner, &self.inner) {
			return Err(ContextError::InactiveGuard);
		}
		commit_guard(&self.inner, guard, outcome)
	}

	/// Returns an exact-precondition snapshot of a held context.
	///
	/// # Errors
	///
	/// Returns [`ContextError::NeedFull`] if absent and
	/// [`ContextError::Conflict`] when the supplied revision is stale.
	pub fn snapshot(&self, context_ref: &ContextRef) -> Result<Thread, ContextError> {
		let mut state = self.inner.state.lock();
		prune_expired(&mut state, self.inner.config.context_ttl, Instant::now());
		let Some(context) = state.contexts.get_mut(&context_ref.context_id) else {
			return Err(ContextError::NeedFull);
		};
		let actual = history_revision(&context.tip);
		if actual != context_ref.expected {
			return Err(ContextError::Conflict { actual });
		}
		context.last_used = Instant::now();
		Ok(materialize(&context.tip))
	}

	/// Returns the current revision without imposing a caller precondition.
	///
	/// # Errors
	///
	/// Returns [`ContextError::NeedFull`] for an unknown or expired context.
	pub fn revision(&self, context_id: &str) -> Result<Revision, ContextError> {
		let mut state = self.inner.state.lock();
		prune_expired(&mut state, self.inner.config.context_ttl, Instant::now());
		let Some(context) = state.contexts.get_mut(context_id) else {
			return Err(ContextError::NeedFull);
		};
		context.last_used = Instant::now();
		Ok(history_revision(&context.tip))
	}

	/// Creates a copy-on-write branch at `at`, or at the parent's head when
	/// absent.
	///
	/// The parent's exact precondition is checked before the requested branch
	/// point. Persistent history nodes are shared, so later commits to either
	/// context cannot alter the other branch.
	///
	/// # Errors
	///
	/// Returns conflict, need-full, invalid-branch, serialization, or capacity
	/// errors.
	pub fn fork(
		&self,
		parent: &ContextRef,
		at: Option<u64>,
		context_id: impl Into<SmolStr>,
	) -> Result<Revision, ContextError> {
		let context_id = context_id.into();
		let mut state = self.inner.state.lock();
		prune_expired(&mut state, self.inner.config.context_ttl, Instant::now());
		let Some(parent_context) = state.contexts.get(&parent.context_id) else {
			return Err(ContextError::NeedFull);
		};
		let actual = history_revision(&parent_context.tip);
		if actual != parent.expected {
			return Err(ContextError::Conflict { actual });
		}
		let at = at.unwrap_or(actual.head);
		if at > actual.head {
			return Err(ContextError::InvalidTruncate { at, head: actual.head });
		}
		if parent_context.in_flight.is_some() {
			return Err(ContextError::Busy);
		}
		if state.contexts.contains_key(&context_id)
			|| state.pending_contexts.contains_key(&context_id)
		{
			return Err(ContextError::AlreadyExists);
		}
		let tip = history_at(&parent_context.tip, at);
		let affinities = if at == actual.head {
			parent_context.affinities.clone()
		} else {
			BTreeMap::new()
		};
		make_context_room_except(
			&mut state,
			self.inner.config.context_capacity,
			Some(parent.context_id.as_str()),
		)?;
		let revision = history_revision(&tip);
		state.contexts.insert(context_id, HeldContext {
			tip,
			affinities,
			in_flight: None,
			last_used: Instant::now(),
		});
		Ok(revision)
	}

	/// Releases a context ahead of its TTL.
	///
	/// An active guard is invalidated and cannot commit after an explicit drop.
	#[must_use]
	pub fn drop_context(&self, context_id: &str) -> bool {
		let mut state = self.inner.state.lock();
		let mut removed = false;
		if let Some(context) = state.contexts.remove(context_id) {
			removed = true;
			if let Some(turn_id) = context.in_flight {
				close_turn(&mut state, &turn_id);
			}
		}
		if let Some(turn_id) = state.pending_contexts.remove(context_id) {
			removed = true;
			close_turn(&mut state, &turn_id);
		}
		removed
	}

	/// Evicts one idle context, primarily for explicit lifecycle management.
	#[must_use]
	pub fn evict(&self, context_id: &str) -> bool {
		let mut state = self.inner.state.lock();
		if state
			.contexts
			.get(context_id)
			.is_some_and(|context| context.in_flight.is_some())
		{
			return false;
		}
		state.contexts.remove(context_id).is_some()
	}

	/// Removes every expired idle context and returns the number evicted.
	pub fn evict_expired(&self) -> usize {
		let mut state = self.inner.state.lock();
		prune_expired(&mut state, self.inner.config.context_ttl, Instant::now())
	}
}

fn commit_guard(
	store: &Arc<Inner>,
	mut guard: TurnGuard,
	mut outcome: ChatOutcome,
) -> Result<ChatOutcome, ContextError> {
	if !guard.active {
		return Err(ContextError::InactiveGuard);
	}
	if guard.output.is_empty() {
		guard.output = mem::take(&mut outcome.output);
	} else if !outcome.output.is_empty() {
		return Err(ContextError::OutputAlreadyAccumulated);
	}

	let mut state = store.state.lock();
	let owns_record = matches!(
		state.turns.get(&guard.turn_id),
		Some(TurnRecord::InFlight { live, .. }) if Arc::ptr_eq(live, &guard.live)
	);
	if !owns_record {
		return Err(ContextError::InactiveGuard);
	}

	let retained_revision = if let Some(context_id) = &guard.context_id {
		if guard.retained {
			let Some(context) = state.contexts.get(context_id) else {
				return Err(ContextError::NeedFull);
			};
			if context.in_flight.as_ref() != Some(&guard.turn_id) {
				return Err(ContextError::InactiveGuard);
			}
		} else if state.pending_contexts.get(context_id) != Some(&guard.turn_id) {
			return Err(ContextError::InactiveGuard);
		}

		let input = guard.thread.items.split_off(guard.input_start);
		let (tip, _) = append_items(guard.base.clone(), input);
		let (tip, committed_output) = append_items(tip, mem::take(&mut guard.output));
		let revision = history_revision(&tip);
		outcome.output = committed_output;
		outcome.revision = Some(revision.clone());
		if guard.retained {
			let context = state
				.contexts
				.get_mut(context_id)
				.expect("retained context checked above");
			context.tip = tip;
			context.affinities = mem::take(&mut guard.affinities);
			context.in_flight = None;
			context.last_used = Instant::now();
		} else {
			make_context_room(&mut state, store.config.context_capacity)?;
			state.pending_contexts.remove(context_id);
			state.contexts.insert(context_id.clone(), HeldContext {
				tip,
				affinities: mem::take(&mut guard.affinities),
				in_flight: None,
				last_used: Instant::now(),
			});
		}
		Some(revision)
	} else {
		let mut next_seq = guard.thread.items.last().map_or(0, |item| item.seq);
		for item in &mut guard.output {
			next_seq = next_seq.saturating_add(1);
			item.seq = next_seq;
		}
		outcome.output = mem::take(&mut guard.output);
		outcome.revision = None;
		None
	};

	let digest = digest_outcome(&outcome);
	let Some(TurnRecord::InFlight { input, .. }) = state.turns.remove(&guard.turn_id) else {
		return Err(ContextError::InactiveGuard);
	};
	state
		.turns
		.insert(guard.turn_id.clone(), TurnRecord::Committed {
			input,
			outcome: Box::new(outcome.clone()),
			digest,
			revision: retained_revision,
		});
	state.committed_order.push_back(guard.turn_id.clone());
	guard.active = false;
	drop(state);
	guard.live.publish(TurnEvent::Outcome(outcome.clone()));
	Ok(outcome)
}

fn remove_live_turn(state: &mut StoreState, turn_id: &str, live: &Arc<LiveTurn>) {
	let remove = matches!(
		state.turns.get(turn_id),
		Some(TurnRecord::InFlight { live: current, .. }) if Arc::ptr_eq(current, live)
	);
	if remove {
		state.turns.remove(turn_id);
	}
}

fn close_turn(state: &mut StoreState, turn_id: &str) {
	if let Some(TurnRecord::InFlight { live, .. }) = state.turns.remove(turn_id) {
		live.close();
	}
}

fn ensure_turn_room(state: &mut StoreState, capacity: usize) -> Result<(), ContextError> {
	while state.turns.len() >= capacity {
		let Some(oldest) = state.committed_order.pop_front() else {
			return Err(ContextError::DedupWindowFull);
		};
		if matches!(state.turns.get(&oldest), Some(TurnRecord::Committed { .. })) {
			state.turns.remove(&oldest);
		}
	}
	Ok(())
}

fn make_context_room(state: &mut StoreState, capacity: usize) -> Result<(), ContextError> {
	make_context_room_except(state, capacity, None)
}

fn make_context_room_except(
	state: &mut StoreState,
	capacity: usize,
	protected: Option<&str>,
) -> Result<(), ContextError> {
	while state.contexts.len() >= capacity {
		let candidate = state
			.contexts
			.iter()
			.filter(|(id, context)| {
				context.in_flight.is_none()
					&& protected.is_none_or(|protected| id.as_str() != protected)
			})
			.min_by_key(|(_, context)| context.last_used)
			.map(|(id, _)| id.clone());
		let Some(candidate) = candidate else {
			return Err(ContextError::ContextCapacity);
		};
		state.contexts.remove(&candidate);
	}
	Ok(())
}

fn prune_expired(state: &mut StoreState, ttl: Duration, now: Instant) -> usize {
	let before = state.contexts.len();
	state.contexts.retain(|_, context| {
		context.in_flight.is_some() || now.saturating_duration_since(context.last_used) < ttl
	});
	before - state.contexts.len()
}

fn history_revision(history: &History) -> Revision {
	history
		.as_ref()
		.map_or_else(genesis_revision, |node| node.revision.clone())
}

fn genesis_revision() -> Revision {
	Revision::builder()
		.head(0)
		.token(Bytes::copy_from_slice(blake3::hash(&[]).as_bytes()))
		.build()
}

fn history_at(history: &History, at: u64) -> History {
	let mut cursor = history.clone();
	while cursor.as_ref().is_some_and(|node| node.item.seq > at) {
		cursor = cursor.and_then(|node| node.previous.clone());
	}
	cursor
}

fn materialize(history: &History) -> Thread {
	let mut reversed = Vec::new();
	let mut cursor = history.clone();
	while let Some(node) = &cursor {
		reversed.push(node.item.clone());
		let previous = node.previous.clone();
		cursor = previous;
	}
	reversed.reverse();
	Thread::builder().items(reversed).build()
}

fn assign_for_working_thread(target: &mut Vec<Item>, items: Vec<Item>) {
	let mut next_seq = target.last().map_or(0, |item| item.seq);
	target.reserve(items.len());
	for mut item in items {
		next_seq = next_seq.saturating_add(1);
		item.seq = next_seq;
		target.push(item);
	}
}

fn append_items(mut history: History, items: Vec<Item>) -> (History, Vec<Item>) {
	let mut committed = Vec::with_capacity(items.len());
	let mut next_seq = history_revision(&history).head;
	for mut item in items {
		next_seq = next_seq.saturating_add(1);
		item.seq = next_seq;
		let previous = history_revision(&history);
		let revision = Revision::builder()
			.head(next_seq)
			.token(hash_item(&previous.token, &item))
			.build();
		history = Some(Arc::new(HistoryNode { previous: history, item: item.clone(), revision }));
		committed.push(item);
	}
	(history, committed)
}

/// Computes `BLAKE3(previous_token || canonical_bytes(item))`.
///
/// A bare counter is insufficient because two divergent histories can reach
/// the same length (the classic ABA shape). The chained token binds every
/// revision to its exact ordered prefix, making equal-head divergence visible.
fn hash_item(previous_token: &Bytes, item: &Item) -> Bytes {
	let mut hasher = Hasher::new();
	hasher.update(previous_token);
	canonical_item(&mut hasher, item);
	Bytes::copy_from_slice(hasher.finalize().as_bytes())
}

fn canonical_item(hasher: &mut Hasher, item: &Item) {
	put_bytes(hasher, b"omp.thread.v1.Item");
	put_u64(hasher, item.seq);
	match &item.kind {
		ItemKind::Message(message) => {
			put_u8(hasher, 1);
			canonical_message(hasher, message);
		},
		ItemKind::ToolCall(call) => {
			put_u8(hasher, 2);
			canonical_tool_call(hasher, call);
		},
		ItemKind::ToolResult(result) => {
			put_u8(hasher, 3);
			canonical_tool_result(hasher, result);
		},
		_ => put_u8(hasher, u8::MAX),
	}
	canonical_props(hasher, &item.props);
}

fn canonical_message(hasher: &mut Hasher, message: &Message) {
	put_u8(hasher, match message.role {
		Role::System => 1,
		Role::User => 2,
		Role::Assistant => 3,
		_ => u8::MAX,
	});
	put_len(hasher, message.parts.len());
	for part in &message.parts {
		canonical_part(hasher, part);
	}
}

fn canonical_part(hasher: &mut Hasher, part: &Part) {
	match part {
		Part::Text(text) => {
			put_u8(hasher, 1);
			put_bytes(hasher, text.as_bytes());
		},
		Part::Thinking(thinking) => {
			put_u8(hasher, 2);
			canonical_thinking(hasher, thinking);
		},
		Part::Blob(blob) => {
			put_u8(hasher, 3);
			canonical_blob(hasher, blob);
		},
		_ => put_u8(hasher, u8::MAX),
	}
}

fn canonical_thinking(hasher: &mut Hasher, thinking: &Thinking) {
	put_bytes(hasher, thinking.text.as_bytes());
	put_bytes(hasher, &thinking.signature);
	put_bool(hasher, thinking.redacted);
}

fn canonical_blob(hasher: &mut Hasher, blob: &BlobPart) {
	hasher.update(&blob.hash);
	put_bytes(hasher, blob.mime.as_bytes());
	put_u64(hasher, blob.size);
	put_bytes(hasher, &blob.inline);
}

fn canonical_tool_call(hasher: &mut Hasher, call: &ToolCall) {
	canonical_call_id(hasher, &call.id);
	put_bytes(hasher, call.name.as_bytes());
	put_bytes(hasher, &call.args_json);
	put_bytes(hasher, &call.thought_signature);
}

fn canonical_tool_result(hasher: &mut Hasher, result: &ToolResult) {
	canonical_call_id(hasher, &result.call_id);
	put_len(hasher, result.parts.len());
	for part in &result.parts {
		canonical_part(hasher, part);
	}
	put_bool(hasher, result.is_error);
}

fn canonical_call_id(hasher: &mut Hasher, id: &CallId) {
	put_bytes(hasher, &id.as_ulid().to_bytes());
}

fn canonical_props(hasher: &mut Hasher, props: &Props) {
	put_len(hasher, props.0.len());
	for (key, value) in &props.0 {
		put_bytes(hasher, key.as_bytes());
		canonical_json(hasher, value);
	}
}

fn canonical_json(hasher: &mut Hasher, value: &Value) {
	match value {
		Value::Null => put_u8(hasher, 0),
		Value::Bool(value) => {
			put_u8(hasher, 1);
			put_bool(hasher, *value);
		},
		Value::Number(value) => {
			put_u8(hasher, 2);
			put_bytes(hasher, value.to_string().as_bytes());
		},
		Value::String(value) => {
			put_u8(hasher, 3);
			put_bytes(hasher, value.as_bytes());
		},
		Value::Array(values) => {
			put_u8(hasher, 4);
			put_len(hasher, values.len());
			for value in values {
				canonical_json(hasher, value);
			}
		},
		Value::Object(values) => {
			put_u8(hasher, 5);
			put_len(hasher, values.len());
			let mut fields: SmallVec<(&str, &Value), 8> = values
				.iter()
				.map(|(key, value)| (key.as_str(), value))
				.collect();
			fields.sort_unstable_by_key(|(left, _)| *left);
			for (key, value) in fields {
				put_bytes(hasher, key.as_bytes());
				canonical_json(hasher, value);
			}
		},
	}
}

fn digest_outcome(outcome: &ChatOutcome) -> [u8; 32] {
	let mut hasher = Hasher::new();
	put_bytes(&mut hasher, b"omp.inference.v1.Outcome");
	put_len(&mut hasher, outcome.output.len());
	for item in &outcome.output {
		canonical_item(&mut hasher, item);
	}
	put_u8(&mut hasher, match outcome.stop {
		StopReason::EndTurn => 1,
		StopReason::ToolUse => 2,
		StopReason::MaxTokens => 3,
		StopReason::ContentFilter => 4,
		_ => u8::MAX,
	});
	canonical_optional_usage(&mut hasher, outcome.usage.as_ref());
	match outcome.cost {
		Some(cost) => {
			put_u8(&mut hasher, 1);
			put_u64(&mut hasher, cost.nanos_usd);
			put_bool(&mut hasher, cost.estimated);
		},
		None => put_u8(&mut hasher, 0),
	}
	put_len(&mut hasher, outcome.unsupported.len());
	for unsupported in &outcome.unsupported {
		canonical_unsupported(&mut hasher, unsupported);
	}
	match &outcome.revision {
		Some(revision) => {
			put_u8(&mut hasher, 1);
			put_u64(&mut hasher, revision.head);
			put_bytes(&mut hasher, &revision.token);
		},
		None => put_u8(&mut hasher, 0),
	}
	put_bytes(&mut hasher, outcome.provider.as_bytes());
	put_bytes(&mut hasher, outcome.model.as_bytes());
	canonical_props(&mut hasher, &outcome.props);
	*hasher.finalize().as_bytes()
}

fn canonical_optional_usage(hasher: &mut Hasher, usage: Option<&Usage>) {
	let Some(usage) = usage else {
		put_u8(hasher, 0);
		return;
	};
	put_u8(hasher, 1);
	put_u64(hasher, usage.input_tokens);
	put_u64(hasher, usage.output_tokens);
	put_u64(hasher, usage.cache_read_tokens);
	put_u64(hasher, usage.cache_write_tokens);
	put_u8(hasher, match usage.accuracy {
		Accuracy::Exact => 1,
		Accuracy::Estimated => 2,
		_ => u8::MAX,
	});
	canonical_props(hasher, &usage.detail);
}

fn canonical_unsupported(hasher: &mut Hasher, unsupported: &Unsupported) {
	put_bytes(hasher, unsupported.what.as_bytes());
	put_bytes(hasher, unsupported.detail.as_bytes());
	put_u8(hasher, match unsupported.action {
		UnsupportedAction::Dropped => 1,
		UnsupportedAction::Emulated => 2,
		UnsupportedAction::Clamped => 3,
		_ => u8::MAX,
	});
}

fn put_u8(hasher: &mut Hasher, value: u8) {
	hasher.update(&[value]);
}

fn put_bool(hasher: &mut Hasher, value: bool) {
	put_u8(hasher, u8::from(value));
}

fn put_u64(hasher: &mut Hasher, value: u64) {
	hasher.update(&value.to_le_bytes());
}

fn put_len(hasher: &mut Hasher, value: usize) {
	put_u64(hasher, u64::try_from(value).unwrap_or(u64::MAX));
}

fn put_bytes(hasher: &mut Hasher, value: &[u8]) {
	put_len(hasher, value.len());
	hasher.update(value);
}

#[cfg(test)]
mod tests {
	use super::*;

	fn text_item(text: &str) -> Item {
		Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Text(text.into())])
					.build(),
			))
			.props(Props::default())
			.build()
	}

	fn context_ref(context_id: &str, expected: Revision) -> ContextRef {
		ContextRef::builder()
			.context_id(SmolStr::from(context_id))
			.expected(expected)
			.build()
	}

	fn delta(append: &[&str], truncate_to: Option<u64>) -> ThreadDelta {
		let mut delta = ThreadDelta::default();
		delta.append = append.iter().map(|text| text_item(text)).collect();
		delta.truncate_to = truncate_to;
		delta
	}

	fn outcome(output: &[&str]) -> ChatOutcome {
		ChatOutcome::builder()
			.output(output.iter().map(|text| text_item(text)).collect())
			.stop(StopReason::EndTurn)
			.unsupported(Vec::new())
			.provider(SmolStr::from("test"))
			.model(SmolStr::from("test/model"))
			.props(Props::default())
			.build()
	}

	fn started(store: &ContextStore, turn_id: &str, input: BeginInput) -> TurnGuard {
		match store.begin(turn_id.into(), input).unwrap() {
			Begin::Started(guard) => guard,
			_ => panic!("expected a newly started turn"),
		}
	}

	fn incremental(context_id: &str, expected: Revision, delta: ThreadDelta) -> BeginInput {
		BeginInput::Incremental { context: context_ref(context_id, expected), delta }
	}

	#[test]
	fn stale_revision_never_truncates() {
		let store = ContextStore::default();
		let first = store
			.seed(
				"ctx",
				Thread::builder()
					.items(vec![text_item("a"), text_item("b")])
					.build(),
			)
			.unwrap();
		let guard =
			started(&store, "advance", incremental("ctx", first.clone(), delta(&["c"], None)));
		let committed = guard.commit(outcome(&[])).unwrap();
		let actual = committed.revision.unwrap();

		let error =
			store.begin("stale".into(), incremental("ctx", first, delta(&["replacement"], Some(0))));
		assert_eq!(error.err().unwrap(), ContextError::Conflict { actual: actual.clone() });
		let snapshot = store.snapshot(&context_ref("ctx", actual)).unwrap();
		assert_eq!(snapshot.items.len(), 3);
	}

	#[test]
	fn correct_precondition_advances_head_and_token() {
		let store = ContextStore::default();
		let before = store.seed("ctx", Thread::default()).unwrap();
		let guard =
			started(&store, "turn", incremental("ctx", before.clone(), delta(&["input"], None)));
		let committed = guard.commit(outcome(&["output"])).unwrap();
		let after = committed.revision.unwrap();
		assert_eq!(after.head, 2);
		assert_ne!(after.token, before.token);
	}

	#[test]
	fn aba_same_head_with_different_token_is_rejected() {
		let store = ContextStore::default();
		let revision_a = store
			.seed("a", Thread::builder().items(vec![text_item("alpha")]).build())
			.unwrap();
		let revision_b = store
			.seed("b", Thread::builder().items(vec![text_item("beta")]).build())
			.unwrap();
		assert_eq!(revision_a.head, revision_b.head);
		assert_ne!(revision_a.token, revision_b.token);

		let error = store.begin("turn".into(), incremental("b", revision_a, delta(&[], None)));
		assert_eq!(error.err().unwrap(), ContextError::Conflict { actual: revision_b });
	}

	#[test]
	fn committed_turn_id_replays_without_reapplying_delta() {
		let store = ContextStore::default();
		let before = store.seed("ctx", Thread::default()).unwrap();
		let input = incremental("ctx", before, delta(&["input"], None));
		let guard = started(&store, "turn", input.clone());
		let committed = guard.commit(outcome(&["output"])).unwrap();
		let revision = committed.revision.clone().unwrap();

		match store.begin("turn".into(), input).unwrap() {
			Begin::Replay { outcome: replay, revision: replay_revision, .. } => {
				assert_eq!(replay, committed);
				assert_eq!(replay_revision, Some(revision.clone()));
			},
			_ => panic!("expected committed replay"),
		}
		assert_eq!(
			store
				.snapshot(&context_ref("ctx", revision))
				.unwrap()
				.items
				.len(),
			2
		);
	}

	#[tokio::test]
	async fn in_flight_turn_id_attaches() {
		let store = ContextStore::default();
		let before = store.seed("ctx", Thread::default()).unwrap();
		let input = incremental("ctx", before, delta(&["input"], None));
		let guard = started(&store, "turn", input.clone());
		guard.publish(TurnEvent::Accepted { replay: false });
		let Begin::Attached(mut attachment) = store.begin("turn".into(), input).unwrap() else {
			panic!("expected in-flight attachment");
		};
		assert_eq!(attachment.recv().await, Some(TurnEvent::Accepted { replay: false }));
		drop(guard);
		assert_eq!(attachment.recv().await, None);
	}

	#[test]
	fn drop_before_commit_rolls_back() {
		let store = ContextStore::default();
		let before = store
			.seed("ctx", Thread::builder().items(vec![text_item("stable")]).build())
			.unwrap();
		let guard = started(
			&store,
			"cancel",
			incremental("ctx", before.clone(), delta(&["uncommitted"], Some(0))),
		);
		drop(guard);
		let snapshot = store.snapshot(&context_ref("ctx", before)).unwrap();
		assert_eq!(snapshot.items.len(), 1);
	}

	#[test]
	fn fork_isolated_from_parent_later_commits() {
		let store = ContextStore::default();
		let parent = store
			.seed("parent", Thread::builder().items(vec![text_item("root")]).build())
			.unwrap();
		let branch = store
			.fork(&context_ref("parent", parent.clone()), None, "branch")
			.unwrap();
		let guard =
			started(&store, "parent-turn", incremental("parent", parent, delta(&["later"], None)));
		let parent_after = guard.commit(outcome(&[])).unwrap().revision.unwrap();
		assert_eq!(
			store
				.snapshot(&context_ref("branch", branch))
				.unwrap()
				.items
				.len(),
			1
		);
		assert_eq!(
			store
				.snapshot(&context_ref("parent", parent_after))
				.unwrap()
				.items
				.len(),
			2
		);
	}

	#[test]
	fn evicted_context_requires_full_seed() {
		let store = ContextStore::default();
		let revision = store.seed("ctx", Thread::default()).unwrap();
		assert!(store.evict("ctx"));
		let error = store.begin("turn".into(), incremental("ctx", revision, delta(&[], None)));
		assert_eq!(error.err().unwrap(), ContextError::NeedFull);
	}
}
