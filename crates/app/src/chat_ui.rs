pub mod input;

use std::{
	collections::{HashMap, HashSet},
	future::pending,
	path::PathBuf,
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use omp_agent::{
	Agent, AgentEvent, AgentPhase, AgentState, Interrupt, InterruptClass, InterruptSource,
	RewindTarget, TurnClient,
};
use omp_chat_ui::{
	Attachment, BackendEvent, Chat, Intent, ModelRow, RewindTargetRow, SessionRow, StatusFacts,
	SubmitMode,
	host::{HostExit, HostOptions},
};
use omp_core::{Str, fmts};
use omp_llm_catalog::{
	ModelKey, ModelSpec, PriceUnit, ProviderDef, ProviderId, provider::AuthSpecKind,
	snapshot::Catalog,
};
use omp_llm_inference::{call::AuthInput, id::TurnId};
use omp_proto::{
	inference::v1::{part_start, turn_event::Event, value},
	thread::v1::{Blob, Item, Message, Part, Role, blob, item, part},
};
use omp_tui::{UiContext, components::AttachmentContent, detect};
use secrecy::SecretString;
use xutf::IntoAnsiStripped as _;

use crate::{
	chat_ui::input::{ChatCommand, commands, help_text, parse_input},
	settings::Settings,
};

pub const CREDENTIAL_STORAGE_LOCKED_MESSAGE: &str =
	"Credential storage is locked (no OS keychain). Set OMP_LLM_KEYCHAIN=1 or run interactively.";
const GATEWAY_LOGIN_MESSAGE: &str = "Provider login is unavailable through a remote gateway; run \
                                     `omp auth login <provider>` on the gateway host.";
const MAX_ATTACHMENT_BYTES: usize = 8 * 1024 * 1024;

/// Kind of caller response requested by an authentication provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthPromptKind {
	/// Static API key.
	ApiKey,
	/// OAuth authorization code.
	AuthorizationCode,
	/// Provider session token.
	SessionToken,
	/// Visible plain text, including an empty default selection.
	PlainText,
	/// Optional secret text for which an empty response means skip.
	OptionalSecret,
	/// Confirmation that an external device step is complete.
	Confirmation,
}

/// User-visible progress from the asynchronous provider login worker.
#[derive(Debug, Eq, PartialEq)]
pub enum ChatAuthEvent {
	/// Public browser authorization URL.
	Url(Str),
	/// Short-lived device code and public verification URL.
	DeviceCode { code: Str, url: Str },
	/// Private input requested by the provider.
	Prompt { message: Str, kind: AuthPromptKind },
	/// Public login instructions or waiting state.
	Notice(Str),
	/// Login completed and credentials are available to later turns.
	Complete(Str),
	/// Login could not persist credentials because no OS keychain is available.
	CredentialStorageLocked,
	/// Login stopped with a secret-free diagnostic.
	Failed(Str),
}

/// Commands serialized into the authentication worker's single mailbox.
pub enum ChatAuthCommand {
	/// Starts a new provider flow.
	Start(Str),
	/// Answers the current private-input prompt.
	Answer(AuthInput),
	/// Cancels the active flow regardless of its current provider event.
	Cancel,
}

/// Non-blocking command and event channels for provider authentication.
pub struct ChatAuth {
	commands: flume::Sender<ChatAuthCommand>,
	events:   flume::Receiver<ChatAuthEvent>,
	active:   Arc<AtomicBool>,
}

impl ChatAuth {
	/// Creates a UI handle over an application-owned authentication worker.
	pub(crate) const fn new(
		commands: flume::Sender<ChatAuthCommand>,
		events: flume::Receiver<ChatAuthEvent>,
		active: Arc<AtomicBool>,
	) -> Self {
		Self { commands, events, active }
	}

	/// Starts one provider login unless another flow is already active.
	pub(crate) fn start(&self, provider: Str) -> Result<(), &'static str> {
		if self
			.active
			.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
			.is_err()
		{
			return Err("authentication is already in progress");
		}
		if self
			.commands
			.try_send(ChatAuthCommand::Start(provider))
			.is_err()
		{
			self.active.store(false, Ordering::Release);
			return Err("authentication worker is unavailable");
		}
		Ok(())
	}

	/// Answers the active provider prompt without exposing its secret to UI
	/// events.
	pub(crate) fn answer(&self, input: AuthInput) -> Result<(), &'static str> {
		match input {
			AuthInput::Cancel => self.cancel(),
			input => self
				.commands
				.try_send(ChatAuthCommand::Answer(input))
				.map_err(|_| "authentication worker is not waiting for input"),
		}
	}

	/// Cancels the active flow even while it is waiting on an external provider.
	pub(crate) fn cancel(&self) -> Result<(), &'static str> {
		self
			.commands
			.try_send(ChatAuthCommand::Cancel)
			.map_err(|_| "authentication worker is unavailable")
	}

	/// Reports whether the worker currently owns a login flow.
	pub(crate) fn is_active(&self) -> bool {
		self.active.load(Ordering::Acquire)
	}

	/// Receives the next secret-free worker event.
	pub(crate) async fn next_event(&self) -> Option<ChatAuthEvent> {
		self.events.recv_async().await.ok()
	}
}

/// One project-local durable session shown by the welcome and resume pickers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResumeChoice {
	/// Stable session identity submitted by the picker.
	pub id:     Str,
	/// Human-readable session name.
	pub label:  Str,
	/// Recency and identity details shown beneath the name.
	pub detail: Str,
}

/// Durable session facts required to initialize the designed chat scene.
pub struct ChatUiSession {
	/// Stable session identifier displayed by the status line.
	pub session_id:     Str,
	/// Canonical history replayed before live events.
	pub initial_items:  Vec<Item>,
	/// Selected model's total token window, when known by the catalog.
	pub context_window: Option<u64>,
}

enum UiCmd {
	Submit(Item),
	ListRewind { reply: flume::Sender<Result<Vec<RewindTarget>, String>> },
	Rewind { to: Option<u64>, reply: flume::Sender<Result<Vec<Item>, String>> },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SubmitAck {
	Done,
	Interrupted,
}

#[derive(Debug)]
struct ToolDisplay {
	name:           Str,
	args:           omp_slopjson::Value,
	started:        bool,
	output_bytes:   Vec<u8>,
	emitted_output: String,
	preview:        String,
}

struct BridgeState {
	model:             String,
	context_window:    Option<u64>,
	context_tokens:    u64,
	cost_nanos:        u64,
	queued:            usize,
	jobs:              HashSet<Str>,
	attempt:           u32,
	turn_started:      Option<Instant>,
	submit_pending:    bool,
	part_serial:       u64,
	active_parts:      HashMap<u32, Str>,
	tools:             HashMap<Str, ToolDisplay>,
	rewind_targets:    Vec<RewindTarget>,
	pending_auth_kind: Option<AuthPromptKind>,
	replaying_turn:    bool,
	settings:          Settings,
}

/// Runs the designed terminal chat scene bridged to a real durable agent.
#[expect(
	clippy::future_not_send,
	reason = "the terminal scene and its bridge stay on one event-loop thread"
)]
pub async fn run<'a, C, R>(
	mut agent: Agent<C>,
	session: ChatUiSession,
	auth: Option<&'a ChatAuth>,
	data_dir: PathBuf,
	mut list_sessions: R,
	welcome: bool,
) -> anyhow::Result<HostExit>
where
	C: TurnClient + 'static,
	R: FnMut() -> anyhow::Result<Vec<ResumeChoice>> + 'a,
{
	let bus = agent.events().clone();
	let mailbox = agent.mailbox();
	let agent_events = bus.subscribe_ui(256);
	let agent_state = agent.state().clone();
	let abort = agent.abort_handle();
	let startup_pending = startup_recovery_needed(
		agent.journal().pending_turn().is_some(),
		agent.journal().pending_input_submission().is_some(),
	);

	let (ui_tx, ui_rx) = flume::bounded::<UiCmd>(1);
	let (error_tx, error_rx) = flume::unbounded::<String>();
	let (ack_tx, ack_rx) = flume::bounded::<SubmitAck>(1);
	let mut agent_task = tokio::spawn(async move {
		if startup_pending {
			let turn_id = TurnId::new(ulid::Ulid::generate().to_string());
			let ack = match agent.submit(Vec::new(), turn_id).await {
				Ok(summary) if summary.interrupted => SubmitAck::Interrupted,
				Ok(_) => SubmitAck::Done,
				Err(error) => {
					let _ = error_tx.send(format!("Startup resume error: {error}"));
					SubmitAck::Done
				},
			};
			let _ = ack_tx.send(ack);
		}
		while let Ok(command) = ui_rx.recv_async().await {
			match command {
				UiCmd::Submit(item) => {
					let turn_id = TurnId::new(ulid::Ulid::generate().to_string());
					let ack = match agent.submit([item], turn_id).await {
						Ok(summary) if summary.interrupted => SubmitAck::Interrupted,
						Ok(_) => SubmitAck::Done,
						Err(error) => {
							let _ = error_tx.send(format!("Submit error: {error}"));
							SubmitAck::Done
						},
					};
					let _ = ack_tx.send(ack);
				},
				UiCmd::ListRewind { reply } => {
					let result = agent.rewind_targets().map_err(|error| error.to_string());
					let _ = reply.send(result);
				},
				UiCmd::Rewind { to, reply } => {
					let result = agent.rewind(to).map_err(|error| error.to_string());
					let _ = reply.send(result);
				},
			}
		}
	});

	let caps = detect();
	let ctx = UiContext::default().with_terminal_caps(&caps);
	let mut chat = Chat::new(&ctx);
	chat.set_slash_commands(commands());
	let (backend_tx, backend_rx) = flume::unbounded();
	let (intent_tx, intent_rx) = flume::unbounded();
	let snapshot = agent_state.snapshot();
	let model = snapshot.turn.params.model.clone();
	drop(snapshot);
	let mut state = BridgeState {
		model,
		context_window: session.context_window,
		context_tokens: 0,
		cost_nanos: 0,
		queued: 0,
		jobs: HashSet::new(),
		attempt: 0,
		turn_started: startup_pending.then(Instant::now),
		submit_pending: startup_pending,
		part_serial: 0,
		active_parts: HashMap::new(),
		tools: HashMap::new(),
		rewind_targets: Vec::new(),
		pending_auth_kind: None,
		replaying_turn: false,
		settings: Settings::load(&data_dir),
	};

	send_backend(&backend_tx, BackendEvent::SessionTitle(session.session_id));
	send_backend(&backend_tx, BackendEvent::ModelsUpdated {
		rows:    model_rows(Catalog::embedded()),
		current: current_model_index(Catalog::embedded(), &state.model),
	});
	if welcome {
		match list_sessions() {
			Ok(choices) => send_backend(&backend_tx, BackendEvent::Sessions(session_rows(choices))),
			Err(error) => send_backend(
				&backend_tx,
				BackendEvent::Error(fmts!("Could not list sessions: {error}")),
			),
		}
	}
	replay_items(&backend_tx, &session.initial_items, &mut state.tools, &mut state.part_serial);
	send_status(&backend_tx, &state, &bus, agent_events.dropped());

	let bridge = async move {
		loop {
			tokio::select! {
				intent = intent_rx.recv_async() => {
					let Ok(intent) = intent else { break };
					if handle_intent(
						intent,
						&backend_tx,
						&ui_tx,
						&mailbox,
						&abort,
						&agent_state,
						auth,
						&data_dir,
						&mut list_sessions,
						&bus,
						agent_events.dropped(),
						&mut state,
					).await? {
						break;
					}
				},
				Ok(message) = error_rx.recv_async() => {
					send_backend(&backend_tx, BackendEvent::Error(Str::from(message)));
				},
				Ok(ack) = ack_rx.recv_async() => {
					state.submit_pending = false;
					state.turn_started = None;
					state.queued = 0;
					send_backend(&backend_tx, BackendEvent::Ack {
						interrupted: ack == SubmitAck::Interrupted,
					});
					send_status(&backend_tx, &state, &bus, agent_events.dropped());
				},
				Some(event) = next_auth_event(auth) => {
					handle_auth_event(&backend_tx, &mut state, event);
				},
				Ok(event) = agent_events.recv() => {
					handle_agent_event(
						&backend_tx,
						&mut state,
						&event,
						&bus,
						agent_events.dropped(),
					);
				},
			}
		}
		Ok::<(), anyhow::Error>(())
	};

	let host = omp_chat_ui::host::run_with_options(chat, ctx, backend_rx, intent_tx, HostOptions {
		welcome,
		exit_on_session_change: true,
	});
	let (host_result, bridge_result) = tokio::join!(host, bridge);
	if tokio::time::timeout(Duration::from_secs(3), &mut agent_task)
		.await
		.is_err()
	{
		agent_task.abort();
		let _ = agent_task.await;
	}
	bridge_result?;
	host_result.map_err(Into::into)
}

#[allow(clippy::too_many_arguments, reason = "the bridge owns one explicit production seam")]
async fn handle_intent<R>(
	intent: Intent,
	backend: &flume::Sender<BackendEvent>,
	commands_tx: &flume::Sender<UiCmd>,
	mailbox: &omp_agent::MailboxSender,
	abort: &omp_agent::AbortHandle,
	agent_state: &AgentState,
	auth: Option<&ChatAuth>,
	data_dir: &std::path::Path,
	list_sessions: &mut R,
	bus: &omp_agent::EventBus,
	dropped: u64,
	state: &mut BridgeState,
) -> anyhow::Result<bool>
where
	R: FnMut() -> anyhow::Result<Vec<ResumeChoice>>,
{
	match intent {
		Intent::Submit { text, attachments, mode } => match parse_input(&text) {
			Ok(ChatCommand::Nothing) => {
				if should_abort_empty(chat_active(state.submit_pending, bus.phase()), state.queued) {
					abort.abort();
				}
			},
			Ok(ChatCommand::Help) => {
				send_backend(backend, BackendEvent::Notice(Str::from(help_text())));
			},
			Ok(ChatCommand::Login(provider)) => {
				if chat_active(state.submit_pending, bus.phase()) {
					send_backend(
						backend,
						BackendEvent::Error(Str::new_static(
							"Wait for the active turn to finish before logging in.",
						)),
					);
				} else {
					handle_login(backend, auth, provider, state);
				}
			},
			Ok(ChatCommand::Model(selector)) => {
				switch_model(backend, agent_state, data_dir, selector.as_str(), state);
			},
			Ok(ChatCommand::ModelPicker) => send_open_models(backend, state),
			Ok(ChatCommand::Resume) => {
				if chat_active(state.submit_pending, bus.phase()) {
					send_backend(
						backend,
						BackendEvent::Error(Str::new_static(
							"Wait for the active turn to finish before resuming another session.",
						)),
					);
				} else {
					match list_sessions() {
						Ok(choices) => {
							send_backend(backend, BackendEvent::Sessions(session_rows(choices)));
						},
						Err(error) => send_backend(
							backend,
							BackendEvent::Error(fmts!("Could not list sessions: {error}")),
						),
					}
				}
			},
			Ok(ChatCommand::Quit) => {
				if chat_active(state.submit_pending, bus.phase()) {
					abort.abort();
				}
				return Ok(true);
			},
			Ok(ChatCommand::Submit(item)) => {
				if auth.is_some_and(ChatAuth::is_active) {
					send_backend(
						backend,
						BackendEvent::Error(Str::new_static(
							"Wait for provider authentication to finish before submitting.",
						)),
					);
				} else {
					let mut item = *item;
					let chips = lower_attachments(&mut item, attachments, |message| {
						send_backend(backend, BackendEvent::Error(message));
					});
					let active = chat_active(state.submit_pending, bus.phase());
					let delivered = if active {
						mailbox
							.try_enqueue(Interrupt {
								class: active_submit_class(mode),
								item,
								source: InterruptSource::Producer(Str::new_static("user")),
							})
							.is_ok()
					} else {
						state.submit_pending = true;
						commands_tx.send_async(UiCmd::Submit(item)).await.is_ok()
					};
					if delivered {
						send_backend(backend, BackendEvent::UserReplayed {
							text: Str::from(text),
							chips,
						});
						if active {
							state.queued = state.queued.saturating_add(1);
						} else {
							state.turn_started.get_or_insert_with(Instant::now);
						}
					} else {
						state.submit_pending = false;
						send_backend(
							backend,
							BackendEvent::Error(Str::new_static("Agent input channel is closed.")),
						);
					}
				}
			},
			Err(error) => send_backend(backend, BackendEvent::Error(Str::from(error.to_string()))),
		},
		Intent::Abort => {
			if chat_active(state.submit_pending, bus.phase()) {
				abort.abort();
			}
		},
		Intent::RewindRequest => {
			if chat_active(state.submit_pending, bus.phase()) {
				send_backend(
					backend,
					BackendEvent::Error(Str::new_static(
						"Wait for the active turn to finish before rewinding.",
					)),
				);
			} else {
				let (reply_tx, reply_rx) = flume::bounded(1);
				if commands_tx
					.send_async(UiCmd::ListRewind { reply: reply_tx })
					.await
					.is_err()
				{
					send_backend(
						backend,
						BackendEvent::Error(Str::new_static("Agent input channel is closed.")),
					);
				} else {
					match reply_rx.recv_async().await {
						Ok(Ok(targets)) => {
							state.rewind_targets = targets;
							send_backend(
								backend,
								BackendEvent::RewindTargets(
									state
										.rewind_targets
										.iter()
										.map(|target| RewindTargetRow {
											event: target.event,
											text:  target.text.clone(),
										})
										.collect(),
								),
							);
						},
						Ok(Err(error)) => send_backend(backend, BackendEvent::Error(Str::from(error))),
						Err(_) => send_backend(
							backend,
							BackendEvent::Error(Str::new_static("Agent rewind reply channel is closed.")),
						),
					}
				}
			}
		},
		Intent::Rewind { event } => {
			let target = state
				.rewind_targets
				.iter()
				.find(|target| target.event == event)
				.cloned();
			if let Some(target) = target {
				let (reply_tx, reply_rx) = flume::bounded(1);
				if commands_tx
					.send_async(UiCmd::Rewind { to: target.keep, reply: reply_tx })
					.await
					.is_err()
				{
					send_backend(
						backend,
						BackendEvent::Error(Str::new_static("Agent input channel is closed.")),
					);
				} else {
					match reply_rx.recv_async().await {
						Ok(Ok(items)) => {
							state.tools.clear();
							send_backend(backend, BackendEvent::HistoryCleared);
							replay_items(backend, &items, &mut state.tools, &mut state.part_serial);
							state.rewind_targets.clear();
						},
						Ok(Err(error)) => send_backend(backend, BackendEvent::Error(Str::from(error))),
						Err(_) => send_backend(
							backend,
							BackendEvent::Error(Str::new_static("Agent rewind reply channel is closed.")),
						),
					}
				}
			} else {
				send_backend(
					backend,
					BackendEvent::Error(Str::new_static(
						"The selected rewind target is no longer available.",
					)),
				);
			}
		},
		Intent::SwitchModel(model) => {
			switch_model(backend, agent_state, data_dir, model.as_str(), state);
		},
		Intent::Login(provider) => {
			if chat_active(state.submit_pending, bus.phase()) {
				send_backend(
					backend,
					BackendEvent::Error(Str::new_static(
						"Wait for the active turn to finish before logging in.",
					)),
				);
			} else {
				handle_login(backend, auth, provider, state);
			}
		},
		Intent::Resume(None) => {
			if chat_active(state.submit_pending, bus.phase()) {
				send_backend(
					backend,
					BackendEvent::Error(Str::new_static(
						"Wait for the active turn to finish before resuming another session.",
					)),
				);
			} else {
				match list_sessions() {
					Ok(choices) => {
						send_backend(backend, BackendEvent::Sessions(session_rows(choices)));
					},
					Err(error) => send_backend(
						backend,
						BackendEvent::Error(fmts!("Could not list sessions: {error}")),
					),
				}
			}
		},
		Intent::Resume(Some(_)) | Intent::NewSession => {},
		Intent::AuthAnswer { value } => {
			if let (Some(auth), Some(kind)) = (auth, state.pending_auth_kind.take()) {
				if let Err(error) = auth.answer(auth_input(kind, value)) {
					send_backend(backend, BackendEvent::Error(Str::from(error)));
				}
			} else {
				send_backend(
					backend,
					BackendEvent::Error(Str::new_static("No authentication prompt is active.")),
				);
			}
		},
		Intent::AuthCancel => {
			state.pending_auth_kind = None;
			if let Some(auth) = auth
				&& let Err(error) = auth.cancel()
			{
				send_backend(backend, BackendEvent::Error(Str::from(error)));
			}
		},
		Intent::Help => send_backend(backend, BackendEvent::Notice(Str::from(help_text()))),
		Intent::Quit => {
			if chat_active(state.submit_pending, bus.phase()) {
				abort.abort();
			}
			return Ok(true);
		},
	}
	send_status(backend, state, bus, dropped);
	Ok(false)
}

fn handle_login(
	backend: &flume::Sender<BackendEvent>,
	auth: Option<&ChatAuth>,
	requested: Option<Str>,
	state: &BridgeState,
) {
	let Some(auth) = auth else {
		send_backend(backend, BackendEvent::Error(Str::new_static(GATEWAY_LOGIN_MESSAGE)));
		return;
	};
	if let Some(requested) = requested {
		match resolve_login_provider(Catalog::embedded(), &requested) {
			Ok(provider) => match auth.start(provider.clone()) {
				Ok(()) => send_backend(
					backend,
					BackendEvent::Notice(fmts!("Starting authentication for `{provider}`…")),
				),
				Err(error) => send_backend(backend, BackendEvent::Error(Str::from(error))),
			},
			Err(error) => send_backend(backend, BackendEvent::Error(error)),
		}
	} else {
		let current = model_provider(Catalog::embedded(), &state.model);
		send_backend(
			backend,
			BackendEvent::LoginProviders(provider_rows(Catalog::embedded(), current.as_deref())),
		);
	}
}

fn handle_auth_event(
	backend: &flume::Sender<BackendEvent>,
	state: &mut BridgeState,
	event: ChatAuthEvent,
) {
	match event {
		ChatAuthEvent::Url(url) => {
			send_backend(backend, BackendEvent::Notice(fmts!("[open to authorize]({url})")));
		},
		ChatAuthEvent::DeviceCode { code, url } => {
			send_backend(
				backend,
				BackendEvent::Notice(fmts!("Enter code `{code}` at [{url}]({url})")),
			);
		},
		ChatAuthEvent::Prompt { message, kind } => {
			state.pending_auth_kind = Some(kind);
			send_backend(backend, BackendEvent::AuthPrompt {
				message,
				masked: prompt_masks_input(kind),
			});
		},
		ChatAuthEvent::Notice(message) => send_backend(backend, BackendEvent::Notice(message)),
		ChatAuthEvent::Complete(message) => {
			state.pending_auth_kind = None;
			send_backend(backend, BackendEvent::AuthPromptClose);
			send_backend(backend, BackendEvent::Notice(message));
		},
		ChatAuthEvent::CredentialStorageLocked => {
			state.pending_auth_kind = None;
			send_backend(backend, BackendEvent::AuthPromptClose);
			send_backend(
				backend,
				BackendEvent::Error(Str::new_static(CREDENTIAL_STORAGE_LOCKED_MESSAGE)),
			);
		},
		ChatAuthEvent::Failed(message) => {
			state.pending_auth_kind = None;
			send_backend(backend, BackendEvent::AuthPromptClose);
			send_backend(backend, BackendEvent::Error(message));
		},
	}
}

fn handle_agent_event(
	backend: &flume::Sender<BackendEvent>,
	state: &mut BridgeState,
	event: &AgentEvent,
	bus: &omp_agent::EventBus,
	dropped: u64,
) {
	match event {
		AgentEvent::Turn { event, .. } => match &event.event {
			Some(Event::Accepted(accepted)) => state.replaying_turn = accepted.replay,
			Some(Event::Outcome(outcome)) => {
				if state.replaying_turn {
					replay_items(backend, &outcome.output, &mut state.tools, &mut state.part_serial);
					state.replaying_turn = false;
				}
				state.queued = 0;
				state.model.clone_from(&outcome.model);
				state.context_window = resolve_model(Catalog::embedded(), &outcome.model)
					.and_then(|spec| spec.limits.context_window);
				if let Some(cost) = &outcome.cost {
					state.cost_nanos = state.cost_nanos.saturating_add(cost.nanos_usd);
				}
				if let Some(snapshot) = &outcome.context_snapshot {
					state.context_tokens = snapshot.prompt_tokens;
				}
				for (_, id) in state.active_parts.drain() {
					send_backend(backend, BackendEvent::AssistantEnd { id });
				}
			},
			Some(Event::Attempt(attempt)) => state.attempt = attempt.number,
			Some(Event::PartStart(start)) => {
				let prefix = match part_start::Kind::try_from(start.kind) {
					Ok(part_start::Kind::Text) => Some(None),
					Ok(part_start::Kind::Thinking) => Some(Some("*Thinking:* ")),
					_ => None,
				};
				if let Some(prefix) = prefix {
					state.part_serial = state.part_serial.saturating_add(1);
					let id = Str::from(format!("assistant-{}", state.part_serial));
					send_backend(backend, BackendEvent::AssistantBegin { id: id.clone() });
					if let Some(prefix) = prefix {
						send_backend(backend, BackendEvent::AssistantDelta {
							id:   id.clone(),
							text: Str::new_static(prefix),
						});
					}
					state.active_parts.insert(start.index, id);
				}
			},
			Some(Event::PartDelta(delta)) => {
				if let Some(id) = state.active_parts.get(&delta.index)
					&& let Ok(fragment) = std::str::from_utf8(&delta.chunk)
				{
					send_backend(backend, BackendEvent::AssistantDelta {
						id:   id.clone(),
						text: Str::from(fragment),
					});
				}
			},
			Some(Event::PartEnd(end)) => {
				if let Some(id) = state.active_parts.remove(&end.index) {
					send_backend(backend, BackendEvent::AssistantEnd { id });
				}
			},
			_ => {},
		},
		AgentEvent::ToolOpened { call_id, name, .. } => {
			state.tools.insert(call_id.clone(), ToolDisplay {
				name:           name.clone(),
				args:           omp_slopjson::Value::Object(omp_slopjson::Object::new()),
				started:        false,
				output_bytes:   Vec::new(),
				emitted_output: String::new(),
				preview:        String::new(),
			});
		},
		AgentEvent::ToolArgs { call_id, view, .. } => {
			if let Some(tool) = state.tools.get_mut(call_id.as_str()) {
				tool.args = view.clone();
				ensure_tool_started(backend, call_id, tool, false);
			}
		},
		AgentEvent::ToolUpdate { call_id, json } => {
			if let Ok(update) = serde_json::from_slice::<serde_json::Value>(json)
				&& let Some(tool) = state.tools.get_mut(call_id.as_str())
			{
				ensure_tool_started(backend, call_id, tool, true);
				if let Some(chunk) = tool_update_text(tool, &update) {
					send_backend(backend, BackendEvent::ToolOutput { id: call_id.clone(), chunk });
				}
			}
		},
		AgentEvent::ToolFinished { call_id, item } => {
			let mut tool = state.tools.remove(call_id.as_str());
			let name = tool
				.as_ref()
				.map_or_else(|| tool_result_name(item), |tool| tool.name.clone());
			if let Some(tool) = tool.as_mut() {
				ensure_tool_started(backend, call_id, tool, true);
			} else {
				send_backend(backend, BackendEvent::ToolStarted {
					id:    call_id.clone(),
					name:  name.clone(),
					title: name.clone(),
				});
			}
			send_tool_result_output(backend, call_id, item);
			let ok = matches!(&item.kind, Some(item::Kind::ToolResult(result)) if !result.is_error);
			send_backend(backend, BackendEvent::ToolFinished {
				id: call_id.clone(),
				ok,
				summary: tool_summary(&name, item),
			});
		},
		AgentEvent::JobRegistered { job_id } => {
			state.jobs.insert(job_id.clone());
		},
		AgentEvent::JobSettled { job_id } => {
			state.jobs.remove(job_id);
		},
		AgentEvent::Failed { message, .. } => {
			send_backend(backend, BackendEvent::Error(fmts!("Agent error: {message}")));
		},
		AgentEvent::Snapshot(_) | AgentEvent::PhaseChanged { .. } => {},
	}
	send_status(backend, state, bus, dropped);
}

fn replay_items(
	backend: &flume::Sender<BackendEvent>,
	items: &[Item],
	tools: &mut HashMap<Str, ToolDisplay>,
	serial: &mut u64,
) {
	for item in items {
		match &item.kind {
			Some(item::Kind::Message(message)) => replay_message(backend, message, serial),
			Some(item::Kind::ToolCall(call)) => {
				let id = Str::from(call.id.as_str());
				let args = std::str::from_utf8(&call.args_json).map_or_else(
					|_| omp_slopjson::Value::Object(omp_slopjson::Object::new()),
					omp_slopjson::parse_streaming,
				);
				let name = Str::from(call.name.as_str());
				let title = call
					.intent
					.as_deref()
					.map_or_else(|| tool_title(&name, &args), Str::from);
				send_backend(backend, BackendEvent::ToolStarted {
					id: id.clone(),
					name: name.clone(),
					title,
				});
				tools.insert(id, ToolDisplay {
					name,
					args,
					started: true,
					output_bytes: Vec::new(),
					emitted_output: String::new(),
					preview: String::new(),
				});
			},
			Some(item::Kind::ToolResult(result)) => {
				let id = Str::from(result.call_id.as_str());
				let tool = tools.remove(id.as_str());
				let name = tool
					.as_ref()
					.map_or_else(|| Str::from(result.name.as_str()), |tool| tool.name.clone());
				if tool.is_none() {
					send_backend(backend, BackendEvent::ToolStarted {
						id:    id.clone(),
						name:  name.clone(),
						title: name.clone(),
					});
				}
				send_tool_result_output(backend, &id, item);
				send_backend(backend, BackendEvent::ToolFinished {
					id,
					ok: !result.is_error,
					summary: tool_summary(&name, item),
				});
			},
			_ => {},
		}
	}
}

fn ensure_tool_started(
	backend: &flume::Sender<BackendEvent>,
	call_id: &Str,
	tool: &mut ToolDisplay,
	force: bool,
) {
	if tool.started {
		return;
	}
	let title = tool_title(&tool.name, &tool.args);
	if !force && title == tool.name {
		return;
	}
	send_backend(backend, BackendEvent::ToolStarted {
		id: call_id.clone(),
		name: tool.name.clone(),
		title,
	});
	tool.started = true;
}

fn replay_message(backend: &flume::Sender<BackendEvent>, message: &Message, serial: &mut u64) {
	let mut text_parts = Vec::new();
	let mut chips = Vec::new();
	for part in &message.parts {
		match &part.kind {
			Some(part::Kind::Text(text)) => {
				if let Some(attachment) = text
					.strip_prefix("<attachment>")
					.and_then(|text| text.strip_suffix("</attachment>"))
				{
					let lines = attachment.bytes().filter(|byte| *byte == b'\n').count() + 1;
					chips.push(fmts!("paste · {lines} lines"));
				} else {
					text_parts.push(text.as_str());
				}
			},
			Some(part::Kind::Blob(blob)) => chips.push(blob_label(blob)),
			_ => {},
		}
	}
	let text = text_parts.join("\n");
	match Role::try_from(message.role) {
		Ok(Role::User) => {
			send_backend(backend, BackendEvent::UserReplayed { text: Str::from(text), chips });
		},
		Ok(Role::System) => {
			if !text.is_empty() {
				send_backend(backend, BackendEvent::Notice(Str::from(text)));
			}
		},
		_ if !text.is_empty() => {
			*serial = serial.saturating_add(1);
			let id = Str::from(format!("history-assistant-{serial}"));
			send_backend(backend, BackendEvent::AssistantBegin { id: id.clone() });
			send_backend(backend, BackendEvent::AssistantDelta {
				id:   id.clone(),
				text: Str::from(text),
			});
			send_backend(backend, BackendEvent::AssistantEnd { id });
		},
		_ => {},
	}
}

fn lower_attachments(
	item: &mut Item,
	attachments: Vec<Attachment>,
	mut report: impl FnMut(Str),
) -> Vec<Str> {
	let mut parts = Vec::with_capacity(attachments.len());
	let mut chips = Vec::with_capacity(attachments.len());
	for attachment in attachments {
		match attachment.content {
			AttachmentContent::Image { source, .. } => {
				let bytes = match std::fs::read(source.as_str()) {
					Ok(bytes) => bytes,
					Err(error) => {
						report(fmts!("Could not attach image `{source}`: {error}"));
						continue;
					},
				};
				if bytes.len() > MAX_ATTACHMENT_BYTES {
					report(fmts!(
						"Image `{source}` is larger than the 8 MiB attachment limit and was skipped."
					));
					continue;
				}
				let Some(mime) = image_mime(source.as_str()) else {
					report(fmts!("Image `{source}` has an unsupported file type and was skipped."));
					continue;
				};
				let size = bytes.len() as u64;
				let hash = Bytes::copy_from_slice(blake3::hash(&bytes).as_bytes());
				let blob = Blob {
					hash,
					mime: mime.to_owned(),
					size,
					inline: Bytes::from(bytes),
					detail: blob::Detail::Auto as i32,
				};
				chips.push(blob_label(&blob));
				parts.push(Part { kind: Some(part::Kind::Blob(blob)) });
			},
			AttachmentContent::Text { text, lines, .. } => {
				chips.push(fmts!("paste · {lines} lines"));
				parts.push(Part {
					kind: Some(part::Kind::Text(format!("<attachment>{text}</attachment>"))),
				});
			},
		}
	}
	if let Some(item::Kind::Message(message)) = item.kind.as_mut() {
		message.parts.extend(parts);
	}
	chips
}

fn image_mime(path: &str) -> Option<&'static str> {
	let extension = std::path::Path::new(path).extension()?.to_str()?;
	if extension.eq_ignore_ascii_case("png") {
		Some("image/png")
	} else if extension.eq_ignore_ascii_case("jpg") || extension.eq_ignore_ascii_case("jpeg") {
		Some("image/jpeg")
	} else if extension.eq_ignore_ascii_case("gif") {
		Some("image/gif")
	} else if extension.eq_ignore_ascii_case("webp") {
		Some("image/webp")
	} else {
		None
	}
}

fn blob_label(blob: &Blob) -> Str {
	fmts!("image {} · {} KB", blob.mime, blob.size.div_ceil(1024))
}

fn tool_update_text(tool: &mut ToolDisplay, update: &serde_json::Value) -> Option<Str> {
	if let Some(preview) = update.get("preview").and_then(serde_json::Value::as_str) {
		let chunk = preview
			.strip_prefix(&tool.preview)
			.map_or_else(|| fmts!("\n{preview}"), Str::from);
		tool.preview.clear();
		tool.preview.push_str(preview);
		return (!chunk.is_empty()).then_some(chunk);
	}
	if let Some(text) = update.get("text").and_then(serde_json::Value::as_str) {
		tool.output_bytes.extend_from_slice(text.as_bytes());
	} else {
		let bytes = update
			.get("data")?
			.as_array()?
			.iter()
			.map(|value| value.as_u64().and_then(|byte| u8::try_from(byte).ok()))
			.collect::<Option<Vec<_>>>()?;
		tool.output_bytes.extend_from_slice(&bytes);
	}
	let rendered = match std::str::from_utf8(&tool.output_bytes) {
		Ok(text) => text.to_owned(),
		Err(error) if error.error_len().is_none() => return None,
		Err(_) => String::from_utf8_lossy(&tool.output_bytes).into_owned(),
	}
	.into_ansi_stripped();
	let chunk = rendered.strip_prefix(&tool.emitted_output)?;
	let chunk = (!chunk.is_empty()).then(|| Str::from(chunk))?;
	tool.emitted_output = rendered;
	Some(chunk)
}

fn tool_title(name: &Str, args: &omp_slopjson::Value) -> Str {
	let detail = if name.as_str() == "edit" {
		args
			.get("input")
			.and_then(|value| value.as_str())
			.and_then(edit_input_path)
	} else {
		["title", "path", "command", "pattern", "query"]
			.into_iter()
			.find_map(|key| args.get(key).and_then(|value| value.as_str()))
			.and_then(|text| text.lines().next())
	};
	detail.map_or_else(|| name.clone(), |detail| fmts!("{name} · {detail}"))
}

fn edit_input_path(input: &str) -> Option<&str> {
	input.lines().find_map(|line| {
		let body = line.trim().strip_prefix('[')?.strip_suffix(']')?;
		let (path, tag) = body.rsplit_once('#')?;
		(!path.is_empty() && !tag.is_empty()).then_some(path)
	})
}

fn tool_result_name(item: &Item) -> Str {
	match &item.kind {
		Some(item::Kind::ToolResult(result)) => Str::from(result.name.as_str()),
		_ => Str::new_static("tool"),
	}
}

fn send_tool_result_output(backend: &flume::Sender<BackendEvent>, call_id: &Str, item: &Item) {
	let Some(item::Kind::ToolResult(result)) = &item.kind else {
		return;
	};
	let mut has_output = false;
	for part in &result.parts {
		let chunk = match &part.kind {
			Some(part::Kind::Text(text)) if !text.is_empty() => Str::from(text.as_str()),
			Some(part::Kind::Blob(blob)) => {
				if let Some(source) = persist_tool_image(blob) {
					// The scene renders persisted PNG payloads inline via the
					// terminal graphics tiers (pi UI-06/UI-20).
					send_backend(backend, BackendEvent::ToolImage { id: call_id.clone(), source });
					continue;
				}
				blob_label(blob)
			},
			_ => continue,
		};
		if has_output {
			send_backend(backend, BackendEvent::ToolOutput {
				id:    call_id.clone(),
				chunk: Str::new_static("\n"),
			});
		}
		send_backend(backend, BackendEvent::ToolOutput { id: call_id.clone(), chunk });
		has_output = true;
	}
}

/// Persists an inline PNG tool-result payload to a content-addressed temp
/// file for inline terminal rendering, returning its path. Non-PNG payloads
/// and by-reference blobs keep their text label: the terminal graphics
/// tiers transmit PNG only.
fn persist_tool_image(blob: &Blob) -> Option<Str> {
	if blob.mime != "image/png" || blob.inline.is_empty() {
		return None;
	}
	let name = if blob.hash.is_empty() {
		format!("omp-tool-image-{}.png", ulid::Ulid::generate())
	} else {
		let hex: String = blob
			.hash
			.iter()
			.take(16)
			.map(|byte| format!("{byte:02x}"))
			.collect();
		format!("omp-tool-image-{hex}.png")
	};
	let path = std::env::temp_dir().join(name);
	if !path.exists() {
		std::fs::write(&path, &blob.inline).ok()?;
	}
	Some(Str::from(path.to_string_lossy().as_ref()))
}

fn tool_summary(name: &Str, item: &Item) -> Vec<Str> {
	let Some(item::Kind::ToolResult(result)) = &item.kind else {
		return vec![fmts!("{name} finished without a tool result")];
	};
	let details = result.details.as_ref().and_then(proto_to_json);
	let mut lines = details
		.as_ref()
		.map_or_else(Vec::new, |details| specialized_tool_summary(name.as_str(), details));
	if lines.is_empty()
		&& result.parts.is_empty()
		&& let Some(details) = details
	{
		let mut preferred = Vec::new();
		collect_summary_strings(&details, &mut preferred);
		lines.extend(
			preferred
				.into_iter()
				.flat_map(|text| text.lines().take(12).map(Str::from).collect::<Vec<_>>()),
		);
	}
	if lines.is_empty() {
		lines.push(if result.is_error {
			fmts!("{name} failed")
		} else {
			fmts!("{name} completed")
		});
	}
	lines.truncate(12);
	lines
}

fn specialized_tool_summary(name: &str, details: &serde_json::Value) -> Vec<Str> {
	let kind = details.get("kind").and_then(serde_json::Value::as_str);
	let value = details.get("value").unwrap_or(details);
	if kind.is_some_and(|kind| kind != "ok") {
		let mut messages = Vec::new();
		collect_summary_strings(value, &mut messages);
		return messages
			.into_iter()
			.flat_map(|message| message.lines().take(6).map(Str::from).collect::<Vec<_>>())
			.take(6)
			.collect();
	}
	match name {
		"edit" => edit_summary(value),
		"grep" => match_summary(value, "matches", "files"),
		"glob" => match_summary(value, "paths", "matches"),
		"shell" | "eval" => status_summary(value),
		"write" => write_summary(value),
		_ => Vec::new(),
	}
}

fn edit_summary(value: &serde_json::Value) -> Vec<Str> {
	let Some(sections) = value.get("sections").and_then(serde_json::Value::as_array) else {
		return Vec::new();
	};
	let (mut added, mut removed) = (0usize, 0usize);
	for line in sections
		.iter()
		.filter_map(|section| section.get("diff").and_then(serde_json::Value::as_str))
		.flat_map(str::lines)
	{
		added += usize::from(line.starts_with('+') && !line.starts_with("+++"));
		removed += usize::from(line.starts_with('-') && !line.starts_with("---"));
	}
	let mut lines = vec![fmts!("{} files changed · +{added} -{removed}", sections.len())];
	lines.extend(sections.iter().take(5).filter_map(|section| {
		let path = section.get("path")?.as_str()?;
		let op = section
			.get("op")
			.and_then(serde_json::Value::as_str)
			.unwrap_or("updated");
		Some(fmts!("{op} {path}"))
	}));
	lines
}

fn match_summary(value: &serde_json::Value, noun: &str, field: &str) -> Vec<Str> {
	let Some(entries) = value.get(field).and_then(serde_json::Value::as_array) else {
		return Vec::new();
	};
	let count = if field == "files" {
		entries
			.iter()
			.filter_map(|entry| entry.get("matches").and_then(serde_json::Value::as_array))
			.map(Vec::len)
			.sum()
	} else {
		entries.len()
	};
	vec![fmts!("{count} {noun}")]
}

fn status_summary(value: &serde_json::Value) -> Vec<Str> {
	let Some(status) = value.get("status") else {
		return Vec::new();
	};
	let outcome = status
		.get("outcome")
		.and_then(serde_json::Value::as_str)
		.unwrap_or("finished");
	let mut lines = vec![
		status
			.get("exit_code")
			.and_then(serde_json::Value::as_i64)
			.map_or_else(|| Str::from(outcome), |code| fmts!("{outcome} · exit {code}")),
	];
	if let Some(exception) = status.get("exception")
		&& let Some(message) = exception.get("message").and_then(serde_json::Value::as_str)
	{
		lines.push(Str::from(message));
	}
	lines
}

fn write_summary(value: &serde_json::Value) -> Vec<Str> {
	let Some(path) = value
		.get("display_path")
		.and_then(serde_json::Value::as_str)
	else {
		return Vec::new();
	};
	let disposition = value
		.get("disposition")
		.and_then(serde_json::Value::as_str)
		.unwrap_or("wrote");
	let bytes = value
		.get("reported_len")
		.and_then(serde_json::Value::as_u64);
	vec![bytes.map_or_else(
		|| fmts!("{disposition} {path}"),
		|bytes| fmts!("{disposition} {path} · {bytes} bytes"),
	)]
}

fn collect_summary_strings(value: &serde_json::Value, out: &mut Vec<String>) {
	match value {
		serde_json::Value::Object(map) => {
			for key in ["message", "reason", "diagnostic", "text", "display_path", "preview"] {
				if let Some(text) = map.get(key).and_then(serde_json::Value::as_str) {
					out.push(text.to_owned());
				}
			}
			for value in map.values() {
				if out.len() >= 12 {
					break;
				}
				collect_summary_strings(value, out);
			}
		},
		serde_json::Value::Array(values) => {
			for value in values {
				if out.len() >= 12 {
					break;
				}
				collect_summary_strings(value, out);
			}
		},
		_ => {},
	}
}

fn proto_to_json(value: &omp_proto::inference::v1::Value) -> Option<serde_json::Value> {
	match value.kind.as_ref()? {
		value::Kind::Null(_) => Some(serde_json::Value::Null),
		value::Kind::Int(number) => Some((*number).into()),
		value::Kind::Uint(number) => Some((*number).into()),
		value::Kind::Double(number) => serde_json::Number::from_f64(*number).map(Into::into),
		value::Kind::Bool(boolean) => Some((*boolean).into()),
		value::Kind::String(string) => Some(string.clone().into()),
		value::Kind::List(list) => list
			.values
			.iter()
			.map(proto_to_json)
			.collect::<Option<Vec<_>>>()
			.map(Into::into),
		value::Kind::Map(map) => {
			let mut object = serde_json::Map::with_capacity(map.fields.len());
			for (key, value) in &map.fields {
				object.insert(key.clone(), proto_to_json(value)?);
			}
			Some(serde_json::Value::Object(object))
		},
	}
}

fn model_rows(catalog: &Catalog) -> Vec<ModelRow> {
	catalog
		.models()
		.iter()
		.map(|model| {
			let (provider_id, provider) = model
				.routes
				.first()
				.and_then(|route| catalog.route(route))
				.map(|route| {
					let name = catalog
						.provider(&route.provider)
						.map_or_else(|| route.provider.to_string(), |provider| provider.name.to_string());
					(Str::from(route.provider.as_str()), Str::from(name))
				})
				.unwrap_or_default();
			let price = |unit| {
				model
					.pricing
					.components
					.iter()
					.find(|price| price.unit == unit)
					.map(|price| price.nanos_usd as f64 / 1_000_000_000.0)
			};
			ModelRow {
				key: Str::from(model.key.to_string()),
				name: model.display_name.clone(),
				provider_id,
				provider,
				context: model.limits.context_window,
				input_mtok: price(PriceUnit::MtokInput),
				output_mtok: price(PriceUnit::MtokOutput),
			}
		})
		.collect()
}

fn current_model_index(catalog: &Catalog, current: &str) -> usize {
	catalog
		.models()
		.iter()
		.position(|model| model.key.as_str() == current)
		.unwrap_or_default()
}

fn send_open_models(backend: &flume::Sender<BackendEvent>, state: &BridgeState) {
	send_backend(backend, BackendEvent::OpenModelPicker {
		rows:    model_rows(Catalog::embedded()),
		current: current_model_index(Catalog::embedded(), &state.model),
	});
}

fn send_models_updated(backend: &flume::Sender<BackendEvent>, state: &BridgeState) {
	send_backend(backend, BackendEvent::ModelsUpdated {
		rows:    model_rows(Catalog::embedded()),
		current: current_model_index(Catalog::embedded(), &state.model),
	});
}

fn provider_rows(catalog: &Catalog, current: Option<&str>) -> Vec<SessionRow> {
	let mut providers = catalog
		.providers()
		.iter()
		.filter(|provider| provider_supports_login(catalog, provider))
		.map(|provider| {
			let oauth = provider_uses_oauth(catalog, provider);
			(provider, oauth, current == Some(provider.id.as_str()))
		})
		.collect::<Vec<_>>();
	providers.sort_by_key(|(_, oauth, current)| (!*current, !*oauth));
	providers
		.into_iter()
		.map(|(provider, oauth, _)| SessionRow {
			id:     Str::from(provider.id.as_str()),
			label:  provider.name.clone(),
			detail: Str::new_static(if oauth { "OAuth" } else { "API key" }),
		})
		.collect()
}

fn provider_supports_login(catalog: &Catalog, provider: &ProviderDef) -> bool {
	provider
		.auth
		.iter()
		.filter_map(|auth_id| catalog.auth_spec(auth_id))
		.any(|auth| auth.kind != AuthSpecKind::None)
}

fn provider_uses_oauth(catalog: &Catalog, provider: &ProviderDef) -> bool {
	provider.auth.iter().any(|auth_id| {
		catalog
			.auth_spec(auth_id)
			.and_then(|auth| auth.oauth.as_ref())
			.is_some_and(|oauth_id| catalog.oauth_spec(oauth_id).is_some())
	})
}

fn session_rows(choices: Vec<ResumeChoice>) -> Vec<SessionRow> {
	choices
		.into_iter()
		.map(|choice| SessionRow { id: choice.id, label: choice.label, detail: choice.detail })
		.collect()
}

fn switch_model(
	backend: &flume::Sender<BackendEvent>,
	state_handle: &AgentState,
	data_dir: &std::path::Path,
	selector: &str,
	state: &mut BridgeState,
) {
	match select_model(state_handle, Catalog::embedded(), selector) {
		Some(spec) => {
			state.model = spec.key.to_string();
			state.context_window = spec.limits.context_window;
			state.settings.default_model = Some(state.model.clone());
			if let Err(error) = state.settings.save(data_dir) {
				send_backend(
					backend,
					BackendEvent::Error(fmts!("Could not save the default model: {error}")),
				);
			}
			send_models_updated(backend, state);
		},
		None => send_backend(backend, BackendEvent::Error(fmts!("Unknown model: {selector}"))),
	}
}

fn select_model<'a>(
	state: &AgentState,
	catalog: &'a Catalog,
	selector: &str,
) -> Option<&'a ModelSpec> {
	let spec = resolve_model(catalog, selector)?;
	let key = spec.key.to_string();
	state.update(|snapshot| snapshot.turn.params.model.clone_from(&key));
	Some(spec)
}

fn resolve_model<'a>(catalog: &'a Catalog, selector: &str) -> Option<&'a ModelSpec> {
	catalog
		.model(&ModelKey::from(selector))
		.or_else(|| catalog.resolve_alias(selector))
}

fn model_provider(catalog: &Catalog, selector: &str) -> Option<Str> {
	let model = resolve_model(catalog, selector)?;
	let route = catalog.route(model.routes.first()?)?;
	Some(Str::from(route.provider.as_str()))
}

fn resolve_login_provider(catalog: &Catalog, requested: &Str) -> Result<Str, Str> {
	let provider_id = ProviderId::from(requested.as_str());
	let Some(provider) = catalog.provider(&provider_id) else {
		return Err(fmts!(
			"Unknown provider `{requested}`. Use `/login` to choose an available provider."
		));
	};
	if !provider_supports_login(catalog, provider) {
		return Err(fmts!(
			"Provider `{}` does not support interactive authentication. Use `/login` to choose \
			 another provider.",
			provider.id
		));
	}
	Ok(Str::from(provider.id.as_str()))
}

fn send_status(
	backend: &flume::Sender<BackendEvent>,
	state: &BridgeState,
	bus: &omp_agent::EventBus,
	dropped: u64,
) {
	send_backend(
		backend,
		BackendEvent::Status(StatusFacts {
			model: Str::from(state.model.as_str()),
			working: chat_active(state.submit_pending, bus.phase()),
			turn_started: state.turn_started,
			context_tokens: state.context_tokens,
			context_window: state.context_window,
			cost_nanos: state.cost_nanos,
			queued: state.queued,
			jobs: state.jobs.len(),
			attempt: state.attempt,
			dropped,
			git: None,
		}),
	);
}

fn send_backend(sender: &flume::Sender<BackendEvent>, event: BackendEvent) {
	let _ = sender.send(event);
}

fn chat_active(submit_pending: bool, phase: AgentPhase) -> bool {
	submit_pending || phase != AgentPhase::Idle
}
const fn should_abort_empty(active: bool, queued: usize) -> bool {
	active && queued > 0
}

/// Interrupt class delivering a submission into an active turn: Enter
/// steers immediately, Alt+Enter queues an idle follow-up.
const fn active_submit_class(mode: SubmitMode) -> InterruptClass {
	match mode {
		SubmitMode::Steer => InterruptClass::Immediate,
		SubmitMode::FollowUp => InterruptClass::Idle,
	}
}

const fn startup_recovery_needed(pending_turn: bool, pending_input_submission: bool) -> bool {
	pending_turn || pending_input_submission
}

/// Returns whether an authentication prompt must suppress terminal echo.
pub const fn prompt_masks_input(kind: AuthPromptKind) -> bool {
	!matches!(kind, AuthPromptKind::Confirmation | AuthPromptKind::PlainText)
}

/// Converts the scene's prompt answer to the inference authentication input.
pub fn auth_input(kind: AuthPromptKind, value: String) -> AuthInput {
	match kind {
		AuthPromptKind::ApiKey => AuthInput::ApiKey(SecretString::from(value)),
		AuthPromptKind::AuthorizationCode => AuthInput::AuthorizationCode(SecretString::from(value)),
		AuthPromptKind::SessionToken => AuthInput::SessionToken(SecretString::from(value)),
		AuthPromptKind::PlainText => AuthInput::PlainText(Str::from(value)),
		AuthPromptKind::OptionalSecret => AuthInput::OptionalSecret(SecretString::from(value)),
		AuthPromptKind::Confirmation => AuthInput::DeviceConfirmed,
	}
}

async fn next_auth_event(auth: Option<&ChatAuth>) -> Option<ChatAuthEvent> {
	match auth {
		Some(auth) => auth.next_event().await,
		None => pending().await,
	}
}

/// Current Unix time in milliseconds for canonical user items.
pub fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis() as u64
}

#[cfg(test)]
mod tests {
	use omp_tui::{Color, components::AttachmentContent};

	use super::*;

	#[test]
	fn blank_submission_interrupts_only_with_queued_work() {
		assert!(!should_abort_empty(false, 0));
		assert!(!should_abort_empty(true, 0));
		assert!(!should_abort_empty(false, 1));
		assert!(should_abort_empty(true, 1));
	}

	#[test]
	fn active_submissions_map_enter_to_steer_and_follow_up_to_idle() {
		assert_eq!(active_submit_class(SubmitMode::Steer), InterruptClass::Immediate);
		assert_eq!(active_submit_class(SubmitMode::FollowUp), InterruptClass::Idle);
	}

	#[test]
	fn authentication_prompt_masking_matches_input_kind() {
		assert!(prompt_masks_input(AuthPromptKind::ApiKey));
		assert!(prompt_masks_input(AuthPromptKind::OptionalSecret));
		assert!(!prompt_masks_input(AuthPromptKind::PlainText));
		assert!(!prompt_masks_input(AuthPromptKind::Confirmation));
	}

	#[test]
	fn authentication_answers_preserve_prompt_kind() {
		assert!(matches!(
			auth_input(AuthPromptKind::ApiKey, "secret".to_owned()),
			AuthInput::ApiKey(_)
		));
		assert!(matches!(
			auth_input(AuthPromptKind::PlainText, "visible".to_owned()),
			AuthInput::PlainText(value) if value.as_str() == "visible"
		));
		assert!(matches!(
			auth_input(AuthPromptKind::Confirmation, String::new()),
			AuthInput::DeviceConfirmed
		));
	}

	#[test]
	fn text_attachment_lowers_after_typed_text() {
		let mut item = input::user_message("typed");
		let attachment = Attachment {
			content: AttachmentContent::Text {
				text:    Str::from("pasted"),
				snippet: Str::from("pasted"),
				lines:   1,
				chars:   6,
			},
			marker:  1,
			color:   Color::Default,
		};
		let chips = lower_attachments(&mut item, vec![attachment], |_| {});
		let Some(item::Kind::Message(message)) = item.kind else {
			panic!("message")
		};
		assert_eq!(message.parts.len(), 2);
		assert!(matches!(
			&message.parts[1].kind,
			Some(part::Kind::Text(text)) if text == "<attachment>pasted</attachment>"
		));
		assert_eq!(chips[0].as_str(), "paste · 1 lines");
	}

	#[test]
	fn image_attachment_lowers_to_inline_hashed_blob() {
		let path =
			std::env::temp_dir().join(format!("omp-chat-attachment-{}.png", ulid::Ulid::generate()));
		let bytes = b"not-a-decoded-image";
		std::fs::write(&path, bytes).expect("write attachment fixture");
		let mut item = input::user_message("inspect");
		let attachment = Attachment {
			content: AttachmentContent::Image {
				source:     Str::from(path.to_string_lossy().as_ref()),
				dimensions: None,
			},
			marker:  1,
			color:   Color::Default,
		};
		let mut errors = Vec::new();
		let chips = lower_attachments(&mut item, vec![attachment], |error| errors.push(error));
		std::fs::remove_file(path).expect("remove attachment fixture");
		assert!(errors.is_empty());
		let Some(item::Kind::Message(message)) = item.kind else {
			panic!("message")
		};
		let Some(part::Kind::Blob(blob)) = &message.parts[1].kind else {
			panic!("blob")
		};
		assert_eq!(blob.mime, "image/png");
		assert_eq!(blob.inline.as_ref(), bytes);
		assert_eq!(blob.hash.as_ref(), blake3::hash(bytes).as_bytes());
		assert_eq!(chips.len(), 1);
	}

	#[test]
	fn png_tool_result_blobs_surface_as_inline_image_events() {
		let (tx, rx) = flume::unbounded();
		let png: &[u8] = b"\x89PNG\r\n\x1a\nfake";
		let item = Item {
			kind: Some(item::Kind::ToolResult(omp_proto::thread::v1::ToolResult {
				call_id: "call-1".to_owned(),
				name: "read".to_owned(),
				parts: vec![
					Part { kind: Some(part::Kind::Text("rendered page 1".to_owned())) },
					Part {
						kind: Some(part::Kind::Blob(Blob {
							hash:   Bytes::from_static(b"0123456789abcdef0123456789abcdef"),
							mime:   "image/png".to_owned(),
							size:   png.len() as u64,
							inline: Bytes::from_static(png),
							detail: blob::Detail::Original as i32,
						})),
					},
				],
				..Default::default()
			})),
			..Default::default()
		};
		send_tool_result_output(&tx, &Str::from("call-1"), &item);
		let events: Vec<_> = rx.drain().collect();
		assert!(matches!(
			&events[0],
			BackendEvent::ToolOutput { chunk, .. } if chunk.as_str() == "rendered page 1"
		));
		let Some(BackendEvent::ToolImage { id, source }) = events.get(1) else {
			panic!("PNG blob produces a ToolImage event");
		};
		assert_eq!(id.as_str(), "call-1");
		let persisted = std::fs::read(source.as_str()).expect("persisted image payload");
		assert_eq!(persisted, png);
		assert_eq!(events.len(), 2, "the image replaces the blob text label");
		std::fs::remove_file(source.as_str()).ok();
	}

	#[test]
	fn non_png_tool_result_blobs_keep_their_text_label() {
		let (tx, rx) = flume::unbounded();
		let item = Item {
			kind: Some(item::Kind::ToolResult(omp_proto::thread::v1::ToolResult {
				call_id: "call-2".to_owned(),
				name: "read".to_owned(),
				parts: vec![Part {
					kind: Some(part::Kind::Blob(Blob {
						hash:   Bytes::new(),
						mime:   "image/jpeg".to_owned(),
						size:   4,
						inline: Bytes::from_static(b"jpeg"),
						detail: blob::Detail::Original as i32,
					})),
				}],
				..Default::default()
			})),
			..Default::default()
		};
		send_tool_result_output(&tx, &Str::from("call-2"), &item);
		let events: Vec<_> = rx.drain().collect();
		assert_eq!(events.len(), 1);
		assert!(matches!(
			&events[0],
			BackendEvent::ToolOutput { chunk, .. } if chunk.contains("image/jpeg")
		));
	}

	#[test]
	fn startup_recovery_covers_both_durable_crash_windows() {
		assert!(!startup_recovery_needed(false, false));
		assert!(startup_recovery_needed(true, false));
		assert!(startup_recovery_needed(false, true));
		assert!(startup_recovery_needed(true, true));
	}

	#[test]
	fn tool_updates_join_split_utf8_and_strip_terminal_escapes() {
		let mut tool = ToolDisplay {
			name:           Str::from("shell"),
			args:           omp_slopjson::Value::Object(omp_slopjson::Object::new()),
			started:        true,
			output_bytes:   Vec::new(),
			emitted_output: String::new(),
			preview:        String::new(),
		};
		assert!(tool_update_text(&mut tool, &serde_json::json!({ "data": [231, 149] })).is_none());
		let chunk = tool_update_text(
			&mut tool,
			&serde_json::json!({
				"data": [140, 27, 91, 51, 49, 109, 111, 107, 27, 91, 48, 109]
			}),
		)
		.expect("completed UTF-8 update");
		assert_eq!(chunk.as_str(), "界ok");
	}
}
