//! Append-only conversation history, context planning, and provider-side state.

pub mod binding;
pub mod conversation;
pub mod revision;
pub mod store;

use std::{
	collections::{BTreeMap, HashMap},
	sync::Arc,
	time::SystemTime,
};

pub use binding::{
	BindingContext, BindingKey, BindingValidity, CredentialGenerationPolicy,
	PendingServerStateBinding, ProviderExpiryDecision, ReseedReason, ReseedState,
	ServerStateBinding, SessionExpiryError, StoredProviderStateEvent,
};
use bytes::Bytes;
pub use conversation::{
	ConversationError, MessagePersistenceError, StoredCacheRetention, StoredContent, StoredMedia,
	StoredMessage, StoredProof, StoredRole, StoredToolResult, TurnDraft,
};
use omp_core::Str;
use parking_lot::Mutex;
pub use revision::{CommittedRevision, HistoryDelta};
pub use store::{
	ConversationStore, InMemoryConversationStore, SqliteConversationStore, SqliteTurnDraft,
};

use crate::{
	answer::ArtifactBody,
	call::{ContextStrategy, Message, OperationCall},
	codec::ProviderStateEvent,
	error::{Error, ErrorKind, ErrorPhase, RetryAction},
	event::{BlockKind, ChatEvent},
	id::{RequestId, Revision},
	layer::{
		ExecutionContext, SessionAffinity,
		session::{SessionAction, SessionCompletion, SessionPlanner},
	},
	receipt::{ExecutionReceipt, ReasonId, RecoveryKind, RecoveryRecord},
};

/// Stable prefix-cache identity derived solely from immutable history and
/// policy scope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixCacheIdentity {
	/// Revision whose canonical prefix is cached.
	pub revision: Revision,
	/// Deterministic opaque cache key.
	pub key:      Str,
}

/// Exact canonical context selected for one attempt.
#[derive(Clone, Debug)]
pub enum ContextPlan<I> {
	/// Send complete canonical history.
	Replay {
		history:              HistoryDelta<I>,
		capture_server_state: bool,
		reason:               Option<ReseedReason>,
	},
	/// Send complete history with a revision-derived provider cache identity.
	PrefixCache { history: HistoryDelta<I>, cache: PrefixCacheIdentity },
	/// Send an opaque compatible handle and only items after its committed base.
	ServerState { binding: ServerStateBinding, delta: HistoryDelta<I> },
}

/// Plans replay, prefix-cache, or provider-side delta context without vendor
/// heuristics.
pub fn plan_context<I, S>(
	store: &S,
	strategy: &ContextStrategy,
	head: &Revision,
	binding: Option<&ServerStateBinding>,
	context: Option<&BindingContext<'_>>,
) -> Result<ContextPlan<I>, ConversationError>
where
	S: ConversationStore<I>,
{
	match strategy {
		ContextStrategy::Replay => Ok(ContextPlan::Replay {
			history:              store.delta(None, head)?,
			capture_server_state: false,
			reason:               None,
		}),
		ContextStrategy::PrefixCache(_) => {
			let history = store.delta(None, head)?;
			let key = Str::from(format!("prefix:{}", head.as_str()));
			Ok(ContextPlan::PrefixCache {
				history,
				cache: PrefixCacheIdentity { revision: head.clone(), key },
			})
		},
		ContextStrategy::ServerState(policy) => {
			let Some(binding) = binding else {
				return Ok(ContextPlan::Replay {
					history:              store.delta(None, head)?,
					capture_server_state: true,
					reason:               Some(ReseedReason::FirstTurn),
				});
			};
			let context = context.ok_or(ConversationError::CorruptStore)?;
			let ancestor = store.is_ancestor(&binding.key.base_revision, head)?;
			let mut scoped = context.clone();
			scoped.max_age = match (context.max_age, policy.max_age) {
				(Some(context), Some(policy)) => Some(context.min(policy)),
				(None, policy) => policy,
				(context, None) => context,
			};
			match binding.validity(&scoped, ancestor) {
				BindingValidity::Compatible => Ok(ContextPlan::ServerState {
					binding: binding.clone(),
					delta:   store.delta(Some(&binding.key.base_revision), head)?,
				}),
				BindingValidity::Reseed(reason) if policy.allow_reseed => Ok(ContextPlan::Replay {
					history:              store.delta(None, head)?,
					capture_server_state: true,
					reason:               Some(reason),
				}),
				BindingValidity::Reseed(_) => Err(ConversationError::RevisionConflict {
					expected: binding.key.base_revision.clone(),
					actual:   head.clone(),
				}),
			}
		},
	}
}

#[derive(Clone)]
struct PreparedTurn {
	request:           crate::id::RequestId,
	session:           crate::call::SessionRequest,
	input:             Arc<[StoredMessage]>,
	provider:          crate::catalog::ProviderId,
	codec:             crate::catalog::CodecId,
	route:             crate::catalog::RouteId,
	model:             crate::catalog::ModelKey,
	trust_domain:      crate::catalog::TrustDomain,
	credential_policy: CredentialGenerationPolicy,
}

#[derive(Clone)]
enum PlannerStore {
	Sqlite(Arc<SqliteConversationStore<StoredMessage>>),
	Memory(Arc<InMemoryConversationStore<StoredMessage>>),
}

enum PlannerDraft {
	Sqlite(SqliteTurnDraft<StoredMessage>),
	Memory(TurnDraft<StoredMessage>),
}

impl PlannerStore {
	fn server_state(
		&self,
		conversation: &crate::id::ConversationId,
	) -> Result<Option<ServerStateBinding>, ConversationError> {
		match self {
			Self::Sqlite(store) => store.server_state(conversation),
			Self::Memory(store) => store.server_state(conversation),
		}
	}

	fn delta(
		&self,
		base: Option<&Revision>,
		head: &Revision,
	) -> Result<HistoryDelta<StoredMessage>, ConversationError> {
		match self {
			Self::Sqlite(store) => store.delta(base, head),
			Self::Memory(store) => store.delta(base, head),
		}
	}

	fn plan(
		&self,
		strategy: &ContextStrategy,
		head: &Revision,
		binding: Option<&ServerStateBinding>,
		context: Option<&BindingContext<'_>>,
	) -> Result<ContextPlan<StoredMessage>, ConversationError> {
		match self {
			Self::Sqlite(store) => plan_context(store.as_ref(), strategy, head, binding, context),
			Self::Memory(store) => plan_context(store.as_ref(), strategy, head, binding, context),
		}
	}

	fn begin(
		&self,
		conversation: &crate::id::ConversationId,
		revision: &Revision,
		turn: crate::id::TurnId,
		input: Arc<[StoredMessage]>,
	) -> Result<PlannerDraft, ConversationError> {
		match self {
			Self::Sqlite(store) => store
				.begin(conversation, revision, turn, input)
				.map(PlannerDraft::Sqlite),
			Self::Memory(store) => store
				.begin(conversation, revision, turn, input)
				.map(PlannerDraft::Memory),
		}
	}
}

impl PlannerDraft {
	fn append(&mut self, items: Arc<[StoredMessage]>) {
		match self {
			Self::Sqlite(draft) => draft.append(items),
			Self::Memory(draft) => draft.append(items),
		}
	}

	fn commit(self) -> Result<CommittedRevision<StoredMessage>, ConversationError> {
		match self {
			Self::Sqlite(draft) => draft.commit(),
			Self::Memory(draft) => draft.commit(),
		}
	}

	fn commit_successful_turn(
		self,
		binding: PendingServerStateBinding,
	) -> Result<CommittedRevision<StoredMessage>, ConversationError> {
		match self {
			Self::Sqlite(draft) => draft.commit_successful_turn(binding),
			Self::Memory(draft) => draft.commit_successful_turn(binding),
		}
	}
}

/// Clone-cheap durable conversation planner shared by every production route
/// stack.
#[derive(Clone)]
pub struct ConversationSessionPlanner {
	store:    PlannerStore,
	catalog:  Arc<crate::catalog::snapshot::Catalog>,
	prepared: Arc<Mutex<HashMap<RequestId, PreparedTurn>>>,
}

impl ConversationSessionPlanner {
	/// Creates a planner backed by an explicitly injected durable SQLite store
	/// and catalog.
	pub fn new(
		store: Arc<SqliteConversationStore<StoredMessage>>,
		catalog: Arc<crate::catalog::snapshot::Catalog>,
	) -> Self {
		Self {
			store: PlannerStore::Sqlite(store),
			catalog,
			prepared: Arc::new(Mutex::new(HashMap::new())),
		}
	}

	/// Creates a planner over an explicitly injected in-memory store for
	/// deterministic tests.
	pub fn with_in_memory(
		store: Arc<InMemoryConversationStore<StoredMessage>>,
		catalog: Arc<crate::catalog::snapshot::Catalog>,
	) -> Self {
		Self {
			store: PlannerStore::Memory(store),
			catalog,
			prepared: Arc::new(Mutex::new(HashMap::new())),
		}
	}

	/// Opens a durable SQLite store at `path` and constructs a planner over it.
	pub fn open(
		path: impl AsRef<std::path::Path>,
		catalog: Arc<crate::catalog::snapshot::Catalog>,
	) -> Result<Self, ConversationError> {
		Ok(Self::new(Arc::new(SqliteConversationStore::open(path)?), catalog))
	}

	fn prepare_inner(
		&self,
		call: &mut crate::call::Call,
		context: &ExecutionContext,
		force_replay: bool,
		input_override: Option<Arc<[StoredMessage]>>,
	) -> Result<SessionAction, Error> {
		let Some(session) = call.session.clone() else {
			context.set_session_affinity(None);
			context.set_session_state(None);
			return Ok(SessionAction::None);
		};
		let plan = call
			.execution
			.as_ref()
			.ok_or_else(|| session_error(context, ErrorKind::InvalidRequest, RetryAction::Never))?;
		let model = plan.model.clone().ok_or_else(|| {
			session_error(context, ErrorKind::CapabilityMismatch, RetryAction::Never)
		})?;
		let route = self
			.catalog
			.route(&plan.route)
			.ok_or_else(|| session_error(context, ErrorKind::SessionConflict, RetryAction::Never))?;
		let OperationCall::Chat(request) = &call.operation else {
			return Err(session_error(context, ErrorKind::InvalidRequest, RetryAction::Never));
		};
		let input = match input_override {
			Some(input) => input,
			None => request
				.messages
				.iter()
				.map(StoredMessage::try_from)
				.collect::<Result<Vec<_>, _>>()
				.map_err(|_| session_error(context, ErrorKind::InvalidRequest, RetryAction::Never))?
				.into(),
		};
		let binding = if force_replay {
			None
		} else {
			self
				.store
				.server_state(&session.conversation)
				.map_err(|_| session_error(context, ErrorKind::SessionConflict, RetryAction::Never))?
		};
		let context_plan = if force_replay {
			ContextPlan::Replay {
				history:              self.store.delta(None, &session.revision).map_err(|_| {
					session_error(context, ErrorKind::SessionConflict, RetryAction::Never)
				})?,
				capture_server_state: matches!(session.strategy, ContextStrategy::ServerState(_)),
				reason:               Some(ReseedReason::ProviderExpired),
			}
		} else if let Some(binding) = binding.as_ref() {
			let scope = BindingContext {
				conversation:          &session.conversation,
				route:                 &plan.route,
				model:                 &model,
				principal:             &binding.key.principal,
				account_change:        None,
				trust_domain:          &route.trust_domain,
				credential_generation: binding.key.credential_generation,
				now:                   SystemTime::now(),
				max_age:               match &session.strategy {
					ContextStrategy::ServerState(policy) => policy.max_age,
					_ => None,
				},
			};
			self
				.store
				.plan(&session.strategy, &session.revision, Some(binding), Some(&scope))
				.map_err(|_| session_error(context, ErrorKind::SessionConflict, RetryAction::Never))?
		} else {
			self
				.store
				.plan(&session.strategy, &session.revision, None, None)
				.map_err(|_| session_error(context, ErrorKind::SessionConflict, RetryAction::Never))?
		};
		let (history, action, selected_binding) = match context_plan {
			ContextPlan::Replay { history, reason, .. } => {
				if reason.is_some_and(|reason| reason != ReseedReason::FirstTurn) {
					context.with_receipt(|receipt| {
						receipt.recoveries.push(RecoveryRecord {
							attempt:     context.attempts(),
							kind:        RecoveryKind::SessionReseed,
							rule:        ReasonId(Str::from(format!("{reason:?}"))),
							input_bytes: 0,
							steps:       1,
						})
					});
				}
				(
					history,
					if reason.is_some() {
						SessionAction::Reseed
					} else {
						SessionAction::Replay
					},
					None,
				)
			},
			ContextPlan::PrefixCache { history, .. } => (history, SessionAction::Replay, None),
			ContextPlan::ServerState { binding, delta } => {
				(delta, SessionAction::Reuse, Some(binding))
			},
		};
		let mut messages = history
			.items()
			.cloned()
			.map(Message::try_from)
			.collect::<Result<Vec<_>, _>>()
			.map_err(|_| session_error(context, ErrorKind::SessionConflict, RetryAction::Never))?;
		messages.extend(
			input
				.iter()
				.cloned()
				.map(Message::try_from)
				.collect::<Result<Vec<_>, _>>()
				.map_err(|_| session_error(context, ErrorKind::SessionConflict, RetryAction::Never))?,
		);
		let mut rewritten = (**request).clone();
		rewritten.messages = messages.into();
		call.operation = OperationCall::Chat(Arc::new(rewritten));
		if let Some(binding) = selected_binding.as_ref() {
			context.set_session_affinity(Some(SessionAffinity {
				principal:             binding.key.principal.clone(),
				credential_generation: binding.key.credential_generation,
				credential_policy:     binding.key.credential_policy,
			}));
		} else {
			context.set_session_affinity(None);
		}
		context.set_session_state(selected_binding);
		let credential_policy = match plan.policy_model.as_ref().map(|model| model.context) {
			Some(crate::catalog::model::ContextStrategy::ServerState(policy))
				if policy.credential_generation_bound =>
			{
				CredentialGenerationPolicy::CredentialGenerationBound
			},
			_ => CredentialGenerationPolicy::PrincipalBound,
		};
		self.prepared.lock().insert(call.id.clone(), PreparedTurn {
			request: call.id.clone(),
			session,
			input,
			provider: plan.provider.clone(),
			codec: plan.codec.clone(),
			route: plan.route.clone(),
			model,
			trust_domain: route.trust_domain.clone(),
			credential_policy,
		});
		Ok(action)
	}
}

impl SessionPlanner for ConversationSessionPlanner {
	fn prepare(
		&self,
		call: &mut crate::call::Call,
		context: &ExecutionContext,
	) -> Result<SessionAction, Error> {
		self.prepare_inner(call, context, false, None)
	}

	fn reseed(&self, call: &mut crate::call::Call, context: &ExecutionContext) -> Result<(), Error> {
		let input = self
			.prepared
			.lock()
			.remove(&call.id)
			.map(|prepared| prepared.input);
		context.set_session_affinity(None);
		context.set_session_state(None);
		self.prepare_inner(call, context, true, input).map(|_| ())
	}

	fn completion(
		&self,
		call: &crate::call::Call,
		context: &ExecutionContext,
	) -> Result<Option<Arc<dyn SessionCompletion>>, Error> {
		let Some(prepared) = self.prepared.lock().remove(&call.id) else {
			return Ok(None);
		};
		let draft = match self.store.begin(
			&prepared.session.conversation,
			&prepared.session.revision,
			prepared.session.turn.clone(),
			prepared.input.clone(),
		) {
			Ok(draft) => draft,
			Err(_) => {
				return Err(session_error(context, ErrorKind::SessionConflict, RetryAction::Never));
			},
		};
		Ok(Some(Arc::new(DurableCompletion {
			draft: Mutex::new(Some(draft)),
			blocks: Mutex::new(BTreeMap::new()),
			prepared,
			prepared_turns: Arc::clone(&self.prepared),
		})))
	}
}

enum AssistantBlock {
	Text(String),
	Reasoning(String),
	Tool(StoredContent),
}

struct DurableCompletion {
	draft:          Mutex<Option<PlannerDraft>>,
	blocks:         Mutex<BTreeMap<u32, AssistantBlock>>,
	prepared:       PreparedTurn,
	prepared_turns: Arc<Mutex<HashMap<crate::id::RequestId, PreparedTurn>>>,
}

impl SessionCompletion for DurableCompletion {
	fn record_chat_event(&self, event: &ChatEvent, context: &ExecutionContext) -> Result<(), Error> {
		let mut blocks = self.blocks.lock();
		match event {
			ChatEvent::BlockStarted { index, kind: BlockKind::Text } => {
				blocks
					.entry(*index)
					.or_insert_with(|| AssistantBlock::Text(String::new()));
			},
			ChatEvent::BlockStarted { index, kind: BlockKind::Thinking } => {
				blocks
					.entry(*index)
					.or_insert_with(|| AssistantBlock::Reasoning(String::new()));
			},
			ChatEvent::BlockStarted { .. }
			| ChatEvent::Started(_)
			| ChatEvent::Usage(_)
			| ChatEvent::Completed(_) => {},
			ChatEvent::TextDelta { index, text } => match blocks
				.entry(*index)
				.or_insert_with(|| AssistantBlock::Text(String::new()))
			{
				AssistantBlock::Text(output) => output.push_str(text.as_str()),
				_ => {
					return Err(session_error(context, ErrorKind::SessionConflict, RetryAction::Never));
				},
			},
			ChatEvent::ThinkingDelta { index, text } => match blocks
				.entry(*index)
				.or_insert_with(|| AssistantBlock::Reasoning(String::new()))
			{
				AssistantBlock::Reasoning(output) => output.push_str(text.as_str()),
				_ => {
					return Err(session_error(context, ErrorKind::SessionConflict, RetryAction::Never));
				},
			},
			ChatEvent::ToolCallReady { index, call } => {
				blocks.insert(
					*index,
					AssistantBlock::Tool(StoredContent::ToolCall {
						call:      call.id.clone(),
						name:      call.name.clone(),
						arguments: serde_json::to_vec(call.arguments.as_value())
							.map(Bytes::from)
							.map_err(|_| {
								session_error(context, ErrorKind::MalformedModelOutput, RetryAction::Never)
							})?,
						proof:     None,
					}),
				);
			},
			ChatEvent::ToolCallStarted { .. } | ChatEvent::ToolArgumentsDelta { .. } => {},
			ChatEvent::Artifact { index, artifact } => {
				let media = match &artifact.body {
					ArtifactBody::Bytes(data) => StoredMedia::Bytes {
						media_type: artifact.media_type.clone(),
						data:       data.clone(),
					},
					ArtifactBody::Stored(reference) => StoredMedia::Artifact {
						store:    reference.store.clone(),
						id:       reference.id.clone(),
						revision: reference.revision.clone(),
					},
					ArtifactBody::Stream(_) => {
						return Err(session_error(
							context,
							ErrorKind::InvalidRequest,
							RetryAction::Never,
						));
					},
				};
				let content = if artifact.media_type.as_str().starts_with("image/") {
					StoredContent::Image(media)
				} else if artifact.media_type.as_str().starts_with("audio/") {
					StoredContent::Audio(media)
				} else {
					StoredContent::Document(media)
				};
				blocks.insert(*index, AssistantBlock::Tool(content));
			},
		}
		Ok(())
	}

	fn commit(
		&self,
		provider_state: Vec<ProviderStateEvent>,
		_: &ExecutionReceipt,
		context: &ExecutionContext,
	) -> Result<(), Error> {
		let proof = |value: Bytes| StoredProof {
			provider: self.prepared.provider.clone(),
			codec: self.prepared.codec.clone(),
			value,
		};
		let mut blocks = self.blocks.lock();
		for event in &provider_state {
			match event {
				ProviderStateEvent::ReasoningSignature { index, signature } => {
					if let Some(block) = blocks.remove(index) {
						let block = match block {
							AssistantBlock::Reasoning(text) => {
								AssistantBlock::Tool(StoredContent::Reasoning {
									text:  Str::from(text),
									proof: Some(proof(signature.clone())),
								})
							},
							other => other,
						};
						blocks.insert(*index, block);
					}
				},
				ProviderStateEvent::HistoryBlock { index, data } => {
					if let Some(block) = blocks.remove(index) {
						let block = match block {
							AssistantBlock::Text(text) => AssistantBlock::Tool(StoredContent::Text {
								text:  Str::from(text),
								proof: Some(proof(data.clone())),
							}),
							AssistantBlock::Reasoning(text) => {
								AssistantBlock::Tool(StoredContent::Reasoning {
									text:  Str::from(text),
									proof: Some(proof(data.clone())),
								})
							},
							AssistantBlock::Tool(StoredContent::ToolCall {
								call,
								name,
								arguments,
								..
							}) => AssistantBlock::Tool(StoredContent::ToolCall {
								call,
								name,
								arguments,
								proof: Some(proof(data.clone())),
							}),
							other => other,
						};
						blocks.insert(*index, block);
					}
				},
				ProviderStateEvent::ToolCallProof { index, value } => {
					if let Some(AssistantBlock::Tool(StoredContent::ToolCall { proof: slot, .. })) =
						blocks.get_mut(index)
					{
						*slot = Some(proof(value.clone()));
					}
				},
				_ => {},
			}
		}
		let content = std::mem::take(&mut *blocks)
			.into_values()
			.map(|block| match block {
				AssistantBlock::Text(text) => {
					StoredContent::Text { text: Str::from(text), proof: None }
				},
				AssistantBlock::Reasoning(text) => {
					StoredContent::Reasoning { text: Str::from(text), proof: None }
				},
				AssistantBlock::Tool(content) => content,
			})
			.collect::<Vec<_>>();
		let assistant =
			StoredMessage { role: StoredRole::Assistant, content: content.into(), name: None };
		let mut draft =
			self.draft.lock().take().ok_or_else(|| {
				session_error(context, ErrorKind::SessionConflict, RetryAction::Never)
			})?;
		draft.append(Arc::from([assistant]));
		let capture_binding =
			matches!(self.prepared.session.strategy, ContextStrategy::ServerState(_))
				&& provider_state.iter().any(|event| {
					matches!(
						event,
						ProviderStateEvent::Continuation { .. } | ProviderStateEvent::Checkpoint { .. }
					)
				});
		if !capture_binding {
			draft
				.commit()
				.map_err(|_| session_error(context, ErrorKind::SessionConflict, RetryAction::Never))?;
		} else {
			let account = context.account_routing().ok_or_else(|| {
				session_error(context, ErrorKind::SessionConflict, RetryAction::Never)
			})?;
			let principal = account.principal.ok_or_else(|| {
				session_error(context, ErrorKind::SessionConflict, RetryAction::Never)
			})?;
			let generation = account.credential_generation.ok_or_else(|| {
				session_error(context, ErrorKind::SessionConflict, RetryAction::Never)
			})?;
			let handle = postcard::to_allocvec(
				&provider_state
					.into_iter()
					.map(StoredProviderStateEvent::from)
					.collect::<Vec<_>>(),
			)
			.map(Bytes::from)
			.map_err(|_| session_error(context, ErrorKind::SessionConflict, RetryAction::Never))?;
			draft
				.commit_successful_turn(PendingServerStateBinding {
					conversation: self.prepared.session.conversation.clone(),
					route: self.prepared.route.clone(),
					model: self.prepared.model.clone(),
					principal,
					trust_domain: self.prepared.trust_domain.clone(),
					credential_generation: generation,
					credential_policy: self.prepared.credential_policy,
					created_at: SystemTime::now(),
					expires_at: None,
					handle,
				})
				.map_err(|_| session_error(context, ErrorKind::SessionConflict, RetryAction::Never))?;
		}
		self.prepared_turns.lock().remove(&self.prepared.request);
		Ok(())
	}

	fn abort(&self, retain_preparation: bool) {
		self.draft.lock().take();
		if retain_preparation {
			self
				.prepared_turns
				.lock()
				.insert(self.prepared.request.clone(), self.prepared.clone());
		} else {
			self.prepared_turns.lock().remove(&self.prepared.request);
		}
	}
}

fn session_error(context: &ExecutionContext, kind: ErrorKind, action: RetryAction) -> Error {
	Error::new(kind, ErrorPhase::Session, action, context.receipt())
}

#[cfg(test)]
mod tests {
	use std::{
		collections::{BTreeMap, HashMap},
		sync::{Arc, Barrier},
		thread,
		time::{Duration, SystemTime},
	};

	use bytes::Bytes;
	use omp_core::Str;
	use omp_llm_catalog::{
		id::{CodecId, ModelKey, ProviderId, RouteId},
		provider::{RedirectTrust, TrustDomain},
	};

	use super::{
		ContextPlan,
		binding::{
			BindingContext, BindingValidity, CredentialGenerationPolicy, PendingServerStateBinding,
			ProviderExpiryDecision, ReseedReason, ReseedState,
		},
		conversation::MessagePersistenceError,
		plan_context,
		store::{ConversationStore, InMemoryConversationStore},
	};
	use crate::{
		account::AccountChangeEvidence,
		body::{AttemptBodyEvidence, BodySource, Replayability, RetryDecision, RetryDecisionReason},
		call::{
			ContentPart, ContextStrategy, MediaInput, Message, ProviderProof, Role, ServerStatePolicy,
		},
		id::{AccountId, PrincipalId, RequestId, TurnId},
	};

	fn trust(origin: &str) -> TrustDomain {
		TrustDomain {
			origin:          Str::from(origin),
			redirects:       RedirectTrust::SameOrigin,
			allow_plaintext: false,
		}
	}

	fn pending(
		conversation: crate::id::ConversationId,
		route: &str,
		model: &str,
		principal: &str,
		created_at: SystemTime,
	) -> PendingServerStateBinding {
		PendingServerStateBinding {
			conversation,
			route: RouteId::new(route),
			model: ModelKey::new(model),
			principal: PrincipalId::new(principal),
			trust_domain: trust("https://route.test"),
			credential_generation: 1,
			credential_policy: CredentialGenerationPolicy::PrincipalBound,
			created_at,
			expires_at: None,
			handle: Bytes::from_static(b"opaque"),
		}
	}

	fn context<'a>(
		conversation: &'a crate::id::ConversationId,
		route: &'a RouteId,
		model: &'a ModelKey,
		principal: &'a PrincipalId,
		trust_domain: &'a TrustDomain,
		now: SystemTime,
	) -> BindingContext<'a> {
		BindingContext {
			conversation,
			route,
			model,
			principal,
			account_change: None,
			trust_domain,
			credential_generation: 2,
			now,
			max_age: Some(Duration::from_secs(3600)),
		}
	}

	#[test]
	fn durable_message_round_trip_preserves_provider_scoped_proof() {
		let message = Message {
			role:    Role::Assistant,
			content: Arc::from([ContentPart::Text {
				text:  Str::from("answer"),
				proof: Some(ProviderProof {
					provider: ProviderId::new("provider"),
					codec:    CodecId::new("codec"),
					value:    Bytes::from_static(b"signed"),
				}),
			}]),
			name:    None,
		};
		let stored = super::StoredMessage::try_from(&message).unwrap();
		let bytes = postcard::to_allocvec(&stored).unwrap();
		let decoded: super::StoredMessage = postcard::from_bytes(&bytes).unwrap();
		let restored = Message::try_from(decoded).unwrap();
		match &restored.content[0] {
			ContentPart::Text { text, proof: Some(proof) } => {
				assert_eq!(text.as_str(), "answer");
				assert_eq!(proof.provider.as_str(), "provider");
				assert_eq!(proof.codec.as_str(), "codec");
				assert_eq!(proof.value, Bytes::from_static(b"signed"));
			},
			_ => panic!("durable content changed shape"),
		}
	}

	#[test]
	fn durable_message_rejects_multipart_wire_body() {
		let message = Message {
			role:    Role::User,
			content: Arc::from([ContentPart::Image(MediaInput::Body {
				media_type: Str::from("image/png"),
				body:       BodySource::multipart(Arc::from([BodySource::bytes(Bytes::from_static(
					b"wire",
				))])),
				name:       Some(Str::from("input.png")),
			})]),
			name:    None,
		};
		assert_eq!(
			super::StoredMessage::try_from(&message),
			Err(MessagePersistenceError::UnstagedBody)
		);
	}
	#[test]
	fn reseed_abort_reopens_fresh_root_draft_and_commits_binding_to_new_revision() {
		let store = Arc::new(InMemoryConversationStore::new());
		let root = store.create().unwrap();
		let input = Arc::from([super::StoredMessage {
			role:    super::StoredRole::User,
			content: Arc::from([]),
			name:    None,
		}]);
		let prepared = super::PreparedTurn {
			request:           RequestId::new("request"),
			session:           crate::call::SessionRequest {
				conversation: root.conversation().clone(),
				revision:     root.revision().clone(),
				turn:         TurnId::new("turn"),
				strategy:     server_strategy(),
			},
			input:             Arc::clone(&input),
			provider:          ProviderId::new("provider"),
			codec:             CodecId::new("codec"),
			route:             RouteId::new("route"),
			model:             ModelKey::new("model"),
			trust_domain:      trust("https://route.test"),
			credential_policy: CredentialGenerationPolicy::PrincipalBound,
		};
		let planner_store = super::PlannerStore::Memory(Arc::clone(&store));
		let draft = planner_store
			.begin(
				&prepared.session.conversation,
				&prepared.session.revision,
				prepared.session.turn.clone(),
				Arc::clone(&input),
			)
			.unwrap();
		let prepared_turns = Arc::new(parking_lot::Mutex::new(HashMap::new()));
		let completion = super::DurableCompletion {
			draft:          parking_lot::Mutex::new(Some(draft)),
			blocks:         parking_lot::Mutex::new(BTreeMap::new()),
			prepared:       prepared.clone(),
			prepared_turns: Arc::clone(&prepared_turns),
		};
		crate::layer::session::SessionCompletion::abort(&completion, true);
		assert_eq!(store.active_drafts(), 0);
		let restored = prepared_turns.lock().remove(&prepared.request).unwrap();
		let fresh = planner_store
			.begin(
				&restored.session.conversation,
				&restored.session.revision,
				restored.session.turn,
				restored.input,
			)
			.unwrap();
		let committed = fresh
			.commit_successful_turn(pending(
				root.conversation().clone(),
				"route",
				"model",
				"principal",
				SystemTime::UNIX_EPOCH,
			))
			.unwrap();
		assert_eq!(committed.parent(), Some(root.revision()));
		assert_eq!(
			store
				.server_state(root.conversation())
				.unwrap()
				.unwrap()
				.key
				.base_revision,
			committed.revision().clone(),
		);
	}

	fn server_strategy() -> ContextStrategy {
		ContextStrategy::ServerState(ServerStatePolicy {
			allow_reseed: true,
			max_age:      Some(Duration::from_secs(3600)),
		})
	}

	#[test]
	fn first_server_turn_replays_then_compatible_turn_sends_only_delta() {
		let store = InMemoryConversationStore::new();
		let root = store.create().unwrap();
		let first = store
			.begin(root.conversation(), root.revision(), TurnId::new("one"), Arc::from([1]))
			.unwrap();
		let first = first.commit().unwrap();
		let initial =
			plan_context::<i32, _>(&store, &server_strategy(), first.revision(), None, None).unwrap();
		assert!(matches!(initial, ContextPlan::Replay {
			capture_server_state: true,
			reason: Some(ReseedReason::FirstTurn),
			..
		}));

		let now = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
		let mut captured = store
			.begin(first.conversation(), first.revision(), TurnId::new("capture"), Arc::from([2]))
			.unwrap();
		captured
			.capture_server_state(pending(
				first.conversation().clone(),
				"route",
				"model",
				"principal",
				now,
			))
			.unwrap();
		let captured = captured.commit().unwrap();
		let binding = store.server_state(first.conversation()).unwrap().unwrap();
		assert_eq!(binding.key.base_revision, *captured.revision());

		let next = store
			.begin(first.conversation(), captured.revision(), TurnId::new("next"), Arc::from([3]))
			.unwrap()
			.commit()
			.unwrap();
		let route = RouteId::new("route");
		let model = ModelKey::new("model");
		let principal = PrincipalId::new("principal");
		let domain = trust("https://route.test");
		let scope = context(first.conversation(), &route, &model, &principal, &domain, now);
		let plan =
			plan_context(&store, &server_strategy(), next.revision(), Some(&binding), Some(&scope))
				.unwrap();
		match plan {
			ContextPlan::ServerState { delta, .. } => {
				assert_eq!(delta.items().copied().collect::<Vec<_>>(), vec![3])
			},
			other => panic!("expected compatible delta, got {other:?}"),
		}
	}

	#[test]
	fn fork_reseeds_once_then_resumes_deltas() {
		let store = InMemoryConversationStore::new();
		let root = store.create().unwrap();
		let now = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
		let mut original = store
			.begin(root.conversation(), root.revision(), TurnId::new("one"), Arc::from([1]))
			.unwrap();
		original
			.capture_server_state(pending(
				root.conversation().clone(),
				"route",
				"model",
				"principal",
				now,
			))
			.unwrap();
		let original = original.commit().unwrap();
		let old_binding = store.server_state(root.conversation()).unwrap().unwrap();
		let fork = store.fork(original.revision()).unwrap();
		let route = RouteId::new("route");
		let model = ModelKey::new("model");
		let principal = PrincipalId::new("principal");
		let domain = trust("https://route.test");
		let fork_scope = context(&fork, &route, &model, &principal, &domain, now);
		assert!(matches!(
			plan_context::<i32, _>(
				&store,
				&server_strategy(),
				original.revision(),
				Some(&old_binding),
				Some(&fork_scope)
			)
			.unwrap(),
			ContextPlan::Replay { reason: Some(ReseedReason::Fork), .. }
		));

		let mut reseed = store
			.begin(&fork, original.revision(), TurnId::new("fork-reseed"), Arc::from([2]))
			.unwrap();
		reseed
			.capture_server_state(pending(fork.clone(), "route", "model", "principal", now))
			.unwrap();
		let reseed = reseed.commit().unwrap();
		let fork_binding = store.server_state(&fork).unwrap().unwrap();
		let next = store
			.begin(&fork, reseed.revision(), TurnId::new("fork-next"), Arc::from([3]))
			.unwrap()
			.commit()
			.unwrap();
		assert!(matches!(
			plan_context(
				&store,
				&server_strategy(),
				next.revision(),
				Some(&fork_binding),
				Some(&fork_scope)
			)
			.unwrap(),
			ContextPlan::ServerState { .. }
		));
	}

	#[test]
	fn binding_scope_changes_have_deterministic_reseed_reasons() {
		let now = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
		let pending = pending(
			crate::id::ConversationId::new("conversation"),
			"route",
			"model",
			"principal",
			now,
		);
		let binding = pending.commit(crate::id::Revision::new("revision"));
		let conversation = crate::id::ConversationId::new("conversation");
		let route = RouteId::new("route");
		let model = ModelKey::new("model");
		let principal = PrincipalId::new("principal");
		let domain = trust("https://route.test");
		let mut scope = context(&conversation, &route, &model, &principal, &domain, now);
		assert_eq!(binding.validity(&scope, true), BindingValidity::Compatible);

		let changed_route = RouteId::new("other-route");
		scope.route = &changed_route;
		assert_eq!(
			binding.validity(&scope, true),
			BindingValidity::Reseed(ReseedReason::RouteChanged)
		);
		scope.route = &route;
		let changed_model = ModelKey::new("other-model");
		scope.model = &changed_model;
		assert_eq!(
			binding.validity(&scope, true),
			BindingValidity::Reseed(ReseedReason::ModelChanged)
		);
		scope.model = &model;
		let changed_domain = trust("https://other.test");
		scope.trust_domain = &changed_domain;
		assert_eq!(
			binding.validity(&scope, true),
			BindingValidity::Reseed(ReseedReason::TrustDomainChanged)
		);
		scope.trust_domain = &domain;
		let changed_principal = PrincipalId::new("other-principal");
		scope.principal = &changed_principal;
		assert_eq!(
			binding.validity(&scope, true),
			BindingValidity::Reseed(ReseedReason::PrincipalChanged)
		);
		scope.principal = &principal;
		let account_change = AccountChangeEvidence::new(
			Some(AccountId::new("old")),
			Some(principal.clone()),
			AccountId::new("new"),
			principal.clone(),
			now,
		);
		scope.account_change = Some(&account_change);
		assert_eq!(
			binding.validity(&scope, true),
			BindingValidity::Reseed(ReseedReason::AccountChanged)
		);
	}

	#[test]
	fn ordinary_same_principal_refresh_preserves_principal_bound_state() {
		let now = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
		let binding = pending(
			crate::id::ConversationId::new("conversation"),
			"route",
			"model",
			"principal",
			now,
		)
		.commit(crate::id::Revision::new("revision"));
		let conversation = crate::id::ConversationId::new("conversation");
		let route = RouteId::new("route");
		let model = ModelKey::new("model");
		let principal = PrincipalId::new("principal");
		let domain = trust("https://route.test");
		let scope = context(&conversation, &route, &model, &principal, &domain, now);
		assert_eq!(binding.validity(&scope, true), BindingValidity::Compatible);

		let mut generation_bound = binding.clone();
		generation_bound.key.credential_policy =
			CredentialGenerationPolicy::CredentialGenerationBound;
		assert_eq!(
			generation_bound.validity(&scope, true),
			BindingValidity::Reseed(ReseedReason::CredentialGenerationChanged)
		);
	}

	#[test]
	fn expired_binding_is_classified_before_attempt_as_provider_expiry() {
		let created = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
		let mut pending = pending(
			crate::id::ConversationId::new("conversation"),
			"route",
			"model",
			"principal",
			created,
		);
		pending.expires_at = Some(created + Duration::from_secs(10));
		let binding = pending.commit(crate::id::Revision::new("revision"));
		let conversation = crate::id::ConversationId::new("conversation");
		let route = RouteId::new("route");
		let model = ModelKey::new("model");
		let principal = PrincipalId::new("principal");
		let domain = trust("https://route.test");
		let scope = context(
			&conversation,
			&route,
			&model,
			&principal,
			&domain,
			created + Duration::from_secs(10),
		);
		assert_eq!(
			binding.validity(&scope, true),
			BindingValidity::Reseed(ReseedReason::ProviderExpired),
		);
	}

	fn replayable() -> AttemptBodyEvidence {
		AttemptBodyEvidence {
			opened:         true,
			consumed:       true,
			replayability:  Replayability::Replayable,
			retry_decision: RetryDecision::Allow,
			reason:         RetryDecisionReason::ReplayableSource,
		}
	}

	#[test]
	fn provider_expiry_reseeds_once_precommit_and_is_partial_postcommit() {
		let mut precommit = ReseedState::default();
		assert_eq!(
			precommit.on_provider_expiry(true, &replayable()),
			ProviderExpiryDecision::ReseedOnce
		);
		assert_eq!(
			precommit.on_provider_expiry(true, &replayable()),
			ProviderExpiryDecision::FailUncommitted
		);

		let mut postcommit = ReseedState::default();
		postcommit.mark_committed();
		assert_eq!(
			postcommit.on_provider_expiry(true, &replayable()),
			ProviderExpiryDecision::FailPartial
		);
	}

	#[test]
	fn consumed_one_shot_body_suppresses_precommit_reseed() {
		let consumed = AttemptBodyEvidence {
			opened:         true,
			consumed:       true,
			replayability:  Replayability::OneShot,
			retry_decision: RetryDecision::Suppress,
			reason:         RetryDecisionReason::ConsumedOneShot,
		};
		assert_eq!(
			ReseedState::default().on_provider_expiry(true, &consumed),
			ProviderExpiryDecision::FailUncommitted,
		);
	}

	#[test]
	fn draft_drop_rolls_back_and_concurrent_same_turn_commit_is_idempotent() {
		let store = Arc::new(InMemoryConversationStore::new());
		let root = store.create().unwrap();
		let dropped = store
			.begin(root.conversation(), root.revision(), TurnId::new("drop"), Arc::from([0]))
			.unwrap();
		assert_eq!(store.active_drafts(), 1);
		drop(dropped);
		assert_eq!(store.active_drafts(), 0);
		assert_eq!(store.head(root.conversation()).unwrap().revision(), root.revision());

		let first = store
			.begin(root.conversation(), root.revision(), TurnId::new("same"), Arc::from([1]))
			.unwrap();
		let second = store
			.begin(root.conversation(), root.revision(), TurnId::new("same"), Arc::from([1]))
			.unwrap();
		let barrier = Arc::new(Barrier::new(2));
		let left_barrier = Arc::clone(&barrier);
		let left = thread::spawn(move || {
			left_barrier.wait();
			first.commit().unwrap()
		});
		let right_barrier = Arc::clone(&barrier);
		let right = thread::spawn(move || {
			right_barrier.wait();
			second.commit().unwrap()
		});
		let left = left.join().unwrap();
		let right = right.join().unwrap();
		assert_eq!(left.revision(), right.revision());
		assert_eq!(
			store
				.delta(Some(root.revision()), left.revision())
				.unwrap()
				.items()
				.copied()
				.collect::<Vec<_>>(),
			vec![1]
		);
	}

	#[test]
	fn sqlite_store_obeys_commit_fork_delta_and_idempotency_laws() {
		let store = super::store::SqliteConversationStore::open_in_memory().unwrap();
		let root = store.create().unwrap();
		let first = store
			.begin(root.conversation(), root.revision(), TurnId::new("same"), Arc::from([1_i32]))
			.unwrap()
			.commit()
			.unwrap();
		let repeated = store
			.begin(root.conversation(), root.revision(), TurnId::new("same"), Arc::from([1_i32]))
			.unwrap()
			.commit()
			.unwrap();
		assert_eq!(first.revision(), repeated.revision());
		let fork = store.fork(first.revision()).unwrap();
		let next = store
			.begin(&fork, first.revision(), TurnId::new("fork"), Arc::from([2_i32]))
			.unwrap()
			.commit()
			.unwrap();
		assert_eq!(
			store
				.delta(Some(first.revision()), next.revision())
				.unwrap()
				.items()
				.copied()
				.collect::<Vec<_>>(),
			vec![2]
		);
	}
}
