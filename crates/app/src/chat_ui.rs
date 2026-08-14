pub mod input;
pub(crate) mod login;
pub(crate) mod models;
pub mod renderers;

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
use omp_core::{Str, StrMut, fmts};
use omp_llm_catalog::{ModelKey, ModelSpec, ProviderId, provider::AuthSpecKind, snapshot::Catalog};
use omp_llm_inference::{call::AuthInput, id::TurnId};
use omp_proto::{
	inference::v1::{part_start, turn_event::Event, value},
	thread::v1::{Blob, Item, Message, Part, Role, blob, item, part},
};
use omp_tool::{Rev, TOOL_REV_PROP};
use omp_tui::{
	App, AppEvent, AppOptions, Border, Dim, Key, OverlayAnchor, OverlayMargin, OverlayOptions, Prop,
	Size, SlashCommands, Ui,
	components::{
		AttachmentContent, Attachments, Boxed, Col, EditorPane, Input, Markdown, Segment, Select,
		SelectOption, Status, TextLeaf, ToolCard, ToolState, TranscriptView,
	},
	dom,
};
use secrecy::SecretString;

use crate::{
	chat_ui::{
		input::{ChatCommand, commands, help_text, parse_input, user_message},
		login::{PROVIDER_SELECT_ID, show_provider_picker_for},
		models::{MODEL_SELECT_ID, show_model_picker},
		renderers::{RendererRegistry, ToolFold},
	},
	settings::Settings,
};

const RESUME_SELECT_ID: &str = "resume-session";
const REWIND_SELECT_ID: &str = "rewind-target";

/// Kind of caller response requested by an authentication provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AuthPromptKind {
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
pub(crate) enum ChatAuthEvent {
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
	/// Login stopped with a secret-free diagnostic.
	Failed(Str),
}

/// Non-blocking command and event channels for provider authentication.
pub(crate) struct ChatAuth {
	requests: flume::Sender<Str>,
	answers:  flume::Sender<AuthInput>,
	events:   flume::Receiver<ChatAuthEvent>,
	active:   Arc<AtomicBool>,
}

impl ChatAuth {
	/// Creates a UI handle over an application-owned authentication worker.
	pub(crate) const fn new(
		requests: flume::Sender<Str>,
		answers: flume::Sender<AuthInput>,
		events: flume::Receiver<ChatAuthEvent>,
		active: Arc<AtomicBool>,
	) -> Self {
		Self { requests, answers, events, active }
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
		if self.requests.try_send(provider).is_err() {
			self.active.store(false, Ordering::Release);
			return Err("authentication worker is unavailable");
		}
		Ok(())
	}

	/// Answers the active provider prompt without exposing its secret to UI
	/// events.
	pub(crate) fn answer(&self, input: AuthInput) -> Result<(), &'static str> {
		self
			.answers
			.try_send(input)
			.map_err(|_| "authentication worker is not waiting for input")
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

/// One project-local durable session shown by the resume picker.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResumeChoice {
	/// Stable session identity submitted by the picker.
	pub id:     Str,
	/// Human-readable session name.
	pub label:  Str,
	/// Recency and identity details shown beneath the name.
	pub detail: Str,
}

/// Terminal-shell disposition returned to the chat composition owner.
#[derive(Debug, Eq, PartialEq)]
pub enum ChatUiExit {
	/// End the interactive chat process.
	Quit,
	/// Reload another durable session in the existing shell.
	Resume(Str),
}

/// Durable session facts required to initialize the inline chat shell.
pub struct ChatUiSession {
	/// Stable session identifier displayed by the status line.
	pub session_id:     Str,
	/// Canonical history replayed into the transcript before live events.
	pub initial_items:  Vec<Item>,
	/// Selected model's total token window, when known by the catalog.
	pub context_window: Option<u64>,
}

struct ActivePart {
	id:     Str,
	text:   StrMut,
	prefix: &'static str,
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

/// Starts the retained inline chat host shared across session reloads.
pub async fn start() -> anyhow::Result<App> {
	let mut app = AppOptions::new()
		.keep_on_cancel()
		.start(|env| {
			let root = dom! {
				<col>
					{ TranscriptView::new().with(Prop::Id, "transcript") }
					<editor id="input" submit>
						<status id="status" />
					</editor>
				</col>
			};
			Ui::from_root(root, env.viewport.width, env.ctx)
		})
		.await?;
	app.ui_mut().focus_first();
	app.ui_mut()
		.update_component::<EditorPane>("input", |pane| {
			pane.set_completion(Box::new(SlashCommands::new(commands())));
			true
		});
	Ok(app)
}

/// Drives one durable session inside an existing inline chat host.
#[expect(
	clippy::future_not_send,
	reason = "omp_tui::App is deliberately confined to its terminal event-loop thread"
)]
pub async fn run<'a, C, R>(
	app: &'a mut App,
	mut agent: Agent<C>,
	session: ChatUiSession,
	auth: Option<&'a ChatAuth>,
	data_dir: PathBuf,
	mut list_sessions: R,
) -> anyhow::Result<ChatUiExit>
where
	C: TurnClient + 'static,
	R: FnMut() -> anyhow::Result<Vec<ResumeChoice>> + 'a,
{
	let bus = agent.events().clone();
	let mailbox = agent.mailbox();
	let events = bus.subscribe_ui(256);
	let agent_state = agent.state().clone();

	let replacing_session = app.ui().has_overlay();

	while app.ui_mut().close_top_overlay().is_some() {}
	app.ui_mut()
		.update_component::<TranscriptView>("transcript", |view| {
			view.clear();
			true
		});
	app.ui_mut().set_text("input", "");
	app.ui_mut().focus_first();
	let mut attachments = None;
	app.ui_mut()
		.update_component::<EditorPane>("input", |pane| {
			attachments = Some(pane.attachments());
			false
		});
	let attachments: Attachments =
		attachments.expect("the chat composer is an EditorPane with attachments");

	let renderers = RendererRegistry::new();
	let mut tool_folds = HashMap::new();
	render_history(app.ui_mut(), &session.initial_items, &renderers, &mut tool_folds);
	if replacing_session {
		app.rebuild_history();
	}

	let mut session_model = agent_state.snapshot().turn.params.model.clone();
	let mut settings = Settings::load(&data_dir);
	let mut context_window = session.context_window;
	let mut session_cost_nanos = 0_u64;
	let mut live_jobs = HashSet::new();
	let mut attempt_indicator = 0;
	let mut context_tokens = 0_u64;
	let mut queued = 0_usize;
	let mut last_esc = None;
	let mut rewind_targets: Vec<RewindTarget> = Vec::new();
	let mut auth_prompt = None;
	let mut submit_pending = startup_recovery_needed(
		agent.journal().pending_turn().is_some(),
		agent.journal().pending_input_submission().is_some(),
	);
	let mut active_parts: HashMap<u32, ActivePart> = HashMap::new();
	let mut replaying_turn = false;
	let mut part_serial = 0_u64;

	update_status(
		app.ui_mut(),
		&session.session_id,
		&session_model,
		attempt_indicator,
		live_jobs.len(),
		session_cost_nanos,
		context_tokens,
		context_window,
		queued,
		events.dropped(),
	);

	let (tx, rx) = flume::bounded::<UiCmd>(1);
	let (err_tx, err_rx) = flume::unbounded::<String>();
	let (submit_ack_tx, submit_ack_rx) = flume::bounded::<SubmitAck>(1);
	let abort = agent.abort_handle();
	let mut agent_task = tokio::spawn(async move {
		if startup_recovery_needed(
			agent.journal().pending_turn().is_some(),
			agent.journal().pending_input_submission().is_some(),
		) {
			let resume_turn_id = TurnId::new(ulid::Ulid::generate().to_string());
			let ack = match agent.submit(Vec::new(), resume_turn_id).await {
				Ok(summary) if summary.interrupted => SubmitAck::Interrupted,
				Ok(_) => SubmitAck::Done,
				Err(error) => {
					let _ = err_tx.send(format!("**Startup resume error:** {error}"));
					SubmitAck::Done
				},
			};
			let _ = submit_ack_tx.send(ack);
		}
		while let Ok(command) = rx.recv_async().await {
			match command {
				UiCmd::Submit(item) => {
					let turn_id = TurnId::new(ulid::Ulid::generate().to_string());
					let ack = match agent.submit([item], turn_id).await {
						Ok(summary) if summary.interrupted => SubmitAck::Interrupted,
						Ok(_) => SubmitAck::Done,
						Err(error) => {
							let _ = err_tx.send(format!("**Submit error:** {error}"));
							SubmitAck::Done
						},
					};
					let _ = submit_ack_tx.send(ack);
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
	let mut exit = ChatUiExit::Quit;
	'ui: loop {
		tokio::select! {
			event = app.next() => {
				let mut received_ack = false;
				while let Ok(ack) = submit_ack_rx.try_recv() {
					submit_pending = false;
					queued = 0;
					received_ack = true;
					handle_submit_ack(app.ui_mut(), ack, &mut active_parts);
				}
				if received_ack {
					update_status(
						app.ui_mut(),
						&session.session_id,
						&session_model,
						attempt_indicator,
						live_jobs.len(),
						session_cost_nanos,
						context_tokens,
						context_window,
						queued,
						events.dropped(),
					);
				}
				let is_active = chat_active(submit_pending, bus.phase());
				if matches!(&event, Ok(Some(AppEvent::Submitted | AppEvent::Updated)))
					|| matches!(
						&event,
						Ok(Some(AppEvent::Key(key))) if *key != Key::Esc
					) {
					last_esc = None;
				}
				match event {
				Ok(Some(trigger @ (AppEvent::Submitted | AppEvent::Key(Key::FollowUp)))) => {
					if matches!(trigger, AppEvent::Submitted)
						&& let Some(kind) = auth_prompt.take()
					{
						let value = app.ui().values()["auth-secret"]
							.as_str()
							.unwrap_or("")
							.to_owned();
						let _ = app.ui_mut().close_top_overlay();
						if let Some(auth) = auth
							&& let Err(error) = auth.answer(auth_input(kind, value))
						{
							push_error(app.ui_mut(), error);
						}
						continue 'ui;
					}
					let text = app.ui().values()["input"].as_str().unwrap_or("").to_owned();
					app.ui_mut().set_text("input", "");
					match parse_input(&text) {
						Ok(ChatCommand::Nothing) => {
							if is_active && queued > 0 {
								abort.abort();
							}
						},
						Ok(ChatCommand::Help) => push_notice(app.ui_mut(), help_text()),
						Ok(ChatCommand::Login(requested)) => {
							if is_active {
								push_error(
									app.ui_mut(),
									"Wait for the active turn to finish before logging in.",
								);
							} else if let Some(auth) = auth {
								if let Some(requested) = requested {
									match resolve_login_provider(Catalog::embedded(), &requested) {
										Ok(provider) => {
											start_provider_login(app.ui_mut(), auth, provider);
										},
										Err(error) => push_error(app.ui_mut(), error),
									}
								} else {
									let current =
										model_provider(Catalog::embedded(), &session_model);
									show_provider_picker_for(
										app.ui_mut(),
										Catalog::embedded(),
										current.as_deref(),
									);
								}
							} else {
								push_error(
									app.ui_mut(),
									"Provider login is unavailable through a remote gateway; run `omp auth login <provider>` on the gateway host.",
								);
							}
						},
						Ok(ChatCommand::Model(requested)) => {
							match select_model(&agent_state, Catalog::embedded(), &requested) {
								Some(spec) => {
									session_model = spec.key.to_string();
									context_window = spec.limits.context_window;
									update_status(
										app.ui_mut(),
										&session.session_id,
										&session_model,
										attempt_indicator,
										live_jobs.len(),
										session_cost_nanos,
										context_tokens,
										context_window,
										queued,
										events.dropped(),
									);
								},
								None => push_error(app.ui_mut(), format!("Unknown model: {requested}")),
							}
						},
						Ok(ChatCommand::ModelPicker) => {
							show_model_picker(app.ui_mut(), Catalog::embedded(), &session_model);
						},
						Ok(ChatCommand::Resume) => {
							if is_active {
								push_error(app.ui_mut(), "Wait for the active turn to finish before resuming another session.");
							} else {
								match list_sessions() {
									Ok(choices) => show_resume_picker(app.ui_mut(), &choices),
									Err(error) => {
										push_error(app.ui_mut(), format!("Could not list sessions: {error}"));
									},
								}
							}
						},
						Ok(ChatCommand::Quit) => {
							if is_active {
								abort.abort();
							}
							break 'ui;
						},
						Ok(ChatCommand::Submit(item)) => {
							if auth.is_some_and(ChatAuth::is_active) {
								push_error(
									app.ui_mut(),
									"Wait for provider authentication to finish before submitting.",
								);
							} else {
								let mut item = *item;
								let mut attachment_parts = Vec::new();
								for attachment in attachments.take() {
									match attachment.content {
										AttachmentContent::Image { source, .. } => {
											let bytes = match std::fs::read(source.as_str()) {
												Ok(bytes) => bytes,
												Err(error) => {
													push_error(
														app.ui_mut(),
														format!("Could not attach image `{source}`: {error}"),
													);
													continue;
												},
											};
											if bytes.len() > 8 * 1024 * 1024 {
												push_error(
													app.ui_mut(),
													format!(
														"Image `{source}` is larger than the 8 MiB attachment limit and was skipped."
													),
												);
												continue;
											}
											let extension = std::path::Path::new(source.as_str())
												.extension()
												.and_then(std::ffi::OsStr::to_str)
												.unwrap_or_default();
											let mime = if extension.eq_ignore_ascii_case("png") {
												"image/png"
											} else if extension.eq_ignore_ascii_case("jpg")
												|| extension.eq_ignore_ascii_case("jpeg")
											{
												"image/jpeg"
											} else if extension.eq_ignore_ascii_case("gif") {
												"image/gif"
											} else if extension.eq_ignore_ascii_case("webp") {
												"image/webp"
											} else {
												push_error(
													app.ui_mut(),
													format!(
														"Image `{source}` has an unsupported file type and was skipped."
													),
												);
												continue;
											};
											let size = bytes.len() as u64;
											let hash =
												Bytes::copy_from_slice(blake3::hash(&bytes).as_bytes());
											attachment_parts.push(Part {
												kind: Some(part::Kind::Blob(Blob {
													hash,
													mime: mime.to_owned(),
													size,
													inline: Bytes::from(bytes),
													detail: blob::Detail::Auto as i32,
												})),
											});
										},
										AttachmentContent::Text { text, .. } => {
											attachment_parts.push(Part {
												kind: Some(part::Kind::Text(format!(
													"<attachment>{text}</attachment>"
												))),
											});
										},
									}
								}
								if let Some(item::Kind::Message(message)) = item.kind.as_mut() {
									message.parts.extend(attachment_parts);
								}

								if is_active {
									let class = if matches!(trigger, AppEvent::Key(Key::FollowUp)) {
										InterruptClass::Idle
									} else {
										InterruptClass::Immediate
									};
									let enqueued = render_then_deliver(
										item,
										|item| render_submitted_item(app.ui_mut(), item),
										|item| {
											mailbox
												.try_enqueue(Interrupt {
													class,
													item,
													source: InterruptSource::Producer(Str::new_static("user")),
												})
												.is_ok()
										},
									);
									if enqueued {
										queued = queued.saturating_add(1);
										update_status(
											app.ui_mut(),
											&session.session_id,
											&session_model,
											attempt_indicator,
											live_jobs.len(),
											session_cost_nanos,
											context_tokens,
											context_window,
											queued,
											events.dropped(),
										);
									}
								} else {
									let sent = render_then_deliver(
										item,
										|item| render_submitted_item(app.ui_mut(), item),
										|item| {
											submit_pending = true;
											tx.send(UiCmd::Submit(item)).is_ok()
										},
									);
									if !sent {
										submit_pending = false;
										push_error(app.ui_mut(), "Agent input channel is closed.");
									}
								}
							}
						},
						Err(error) => push_error(app.ui_mut(), error.to_string()),
					}
				},
				Ok(Some(AppEvent::Changed { id, value }))
					if id.as_str() == PROVIDER_SELECT_ID =>
				{
					let _ = app.ui_mut().close_top_overlay();
					if let Some(auth) = auth {
						start_provider_login(app.ui_mut(), auth, value);
					}
				},
				Ok(Some(AppEvent::Changed { id, value })) if id.as_str() == REWIND_SELECT_ID => {
					let selected_event = value.parse::<u64>().ok();
					let target = rewind_targets
						.iter()
						.find(|target| Some(target.event) == selected_event)
						.cloned();
					if let Some(target) = target {
						let (reply_tx, reply_rx) = flume::bounded(1);
						if tx
							.send_async(UiCmd::Rewind { to: target.keep, reply: reply_tx })
							.await
							.is_err()
						{
							push_error(app.ui_mut(), "Agent input channel is closed.");
						} else {
							match reply_rx.recv_async().await {
								Ok(Ok(items)) => {
									app.ui_mut().update_component::<TranscriptView>(
										"transcript",
										|view| {
											view.clear();
											true
										},
									);
									tool_folds.clear();
									render_history(
										app.ui_mut(),
										&items,
										&renderers,
										&mut tool_folds,
									);
									app.rebuild_history();
									app.ui_mut().set_text("input", target.text);
									let _ = app.ui_mut().close_top_overlay();
									rewind_targets.clear();
								},
								Ok(Err(error)) => push_error(app.ui_mut(), error),
								Err(_) => push_error(app.ui_mut(), "Agent rewind reply channel is closed."),
							}
						}
					} else {
						push_error(app.ui_mut(), "The selected rewind target is no longer available.");
					}
				},
				Ok(Some(AppEvent::Changed { id, value })) if id.as_str() == MODEL_SELECT_ID => {
					let _ = app.ui_mut().close_top_overlay();
					match select_model(&agent_state, Catalog::embedded(), &value) {
						Some(spec) => {
							session_model = spec.key.to_string();
							context_window = spec.limits.context_window;
							update_status(
								app.ui_mut(),
								&session.session_id,
								&session_model,
								attempt_indicator,
								live_jobs.len(),
								session_cost_nanos,
								context_tokens,
								context_window,
								queued,
								events.dropped(),
							);
							settings.default_model = Some(session_model.clone());
							if let Err(error) = settings.save(&data_dir) {
								push_error(
									app.ui_mut(),
									format!("Could not save the default model: {error}"),
								);
							}
						},
						None => push_error(app.ui_mut(), format!("Unknown model: {value}")),
					}
				},
				Ok(Some(AppEvent::Changed { id, value })) if id.as_str() == RESUME_SELECT_ID => {
					exit = ChatUiExit::Resume(value);
					break 'ui;
				},
				Ok(Some(AppEvent::OverlayClosed(_))) if auth_prompt.take().is_some() => {
					if let Some(auth) = auth
						&& let Err(error) = auth.answer(AuthInput::Cancel)
					{
						push_error(app.ui_mut(), error);
					}
				},
				Ok(Some(AppEvent::Key(Key::Esc))) => {
					if is_active {
						last_esc = None;
						let item = user_message("User interrupted via Esc.");
						let enqueued = render_then_deliver(
							item,
							|item| render_submitted_item(app.ui_mut(), item),
							|item| {
								mailbox
									.try_enqueue(Interrupt {
										class: InterruptClass::Immediate,
										item,
										source: InterruptSource::Producer(Str::new_static("user")),
									})
									.is_ok()
							},
						);
						abort.abort();
						if enqueued {
							queued = queued.saturating_add(1);
							update_status(
								app.ui_mut(),
								&session.session_id,
								&session_model,
								attempt_indicator,
								live_jobs.len(),
								session_cost_nanos,
								context_tokens,
								context_window,
								queued,
								events.dropped(),
							);
						}
					} else if app.ui().values()["input"].as_str().unwrap_or("").is_empty() {
						let now = Instant::now();
						let is_double = last_esc.is_some_and(|previous| {
							now.saturating_duration_since(previous) <= Duration::from_millis(500)
						});
						last_esc = Some(now);
						if is_double {
							last_esc = None;
							let (reply_tx, reply_rx) = flume::bounded(1);
							if tx
								.send_async(UiCmd::ListRewind { reply: reply_tx })
								.await
								.is_err()
							{
								push_error(app.ui_mut(), "Agent input channel is closed.");
							} else {
								match reply_rx.recv_async().await {
									Ok(Ok(targets)) => {
										rewind_targets = targets;
										show_rewind_picker(app.ui_mut(), &rewind_targets);
									},
									Ok(Err(error)) => push_error(app.ui_mut(), error),
									Err(_) => push_error(app.ui_mut(), "Agent rewind reply channel is closed."),
								}
							}
						}
					} else {
						last_esc = None;
					}
				},
				Ok(Some(_)) => {},
				Ok(None) | Err(_) => {
					if is_active {
						abort.abort();
					}
					break 'ui;
				},
				}
			},
			Ok(message) = err_rx.recv_async() => push_error(app.ui_mut(), message),
			Ok(ack) = submit_ack_rx.recv_async() => {
				submit_pending = false;
				queued = 0;
				handle_submit_ack(app.ui_mut(), ack, &mut active_parts);
				update_status(
					app.ui_mut(),
					&session.session_id,
					&session_model,
					attempt_indicator,
					live_jobs.len(),
					session_cost_nanos,
					context_tokens,
					context_window,
					queued,
					events.dropped(),
				);
			},
			Some(auth_event) = next_auth_event(auth) => match auth_event {
				ChatAuthEvent::Url(url) => {
					push_notice(app.ui_mut(), fmts!("[open to authorize]({url})"));
				},
				ChatAuthEvent::DeviceCode { code, url } => {
					push_notice(
						app.ui_mut(),
						fmts!("Enter code `{code}` at [{url}]({url})"),
					);
				},
				ChatAuthEvent::Prompt { message, kind } => {
					auth_prompt = Some(kind);
					show_auth_prompt(app.ui_mut(), message, kind);
				},
				ChatAuthEvent::Notice(message) => push_notice(app.ui_mut(), message),
				ChatAuthEvent::Complete(message) => {
					if auth_prompt.take().is_some() {
						let _ = app.ui_mut().close_top_overlay();
					}
					push_notice(app.ui_mut(), message);
				},
				ChatAuthEvent::Failed(message) => {
					if auth_prompt.take().is_some() {
						let _ = app.ui_mut().close_top_overlay();
					}
					push_error(app.ui_mut(), message);
				},
			},
			Ok(agent_event) = events.recv() => {
				match &*agent_event {
					AgentEvent::Turn { event: turn_event, .. } => match &turn_event.event {
						Some(Event::Accepted(accepted)) => replaying_turn = accepted.replay,
						Some(Event::Outcome(outcome)) => {
							if replaying_turn {
								render_history(
									app.ui_mut(),
									&outcome.output,
									&renderers,
									&mut tool_folds,
								);
								replaying_turn = false;
							}
							queued = 0;
							session_model.clone_from(&outcome.model);
							if let Some(spec) = resolve_model(Catalog::embedded(), &outcome.model) {
								context_window = spec.limits.context_window;
							}
							if let Some(cost) = &outcome.cost {
								session_cost_nanos = session_cost_nanos.saturating_add(cost.nanos_usd);
							}
							if let Some(snapshot) = &outcome.context_snapshot {
								context_tokens = snapshot.prompt_tokens;
							}
							for active in active_parts.values() {
								app.ui_mut().set_prop(active.id.as_str(), Prop::Partial, false);
							}
							active_parts.clear();
						},
						Some(Event::Attempt(attempt)) => attempt_indicator = attempt.number,
						Some(Event::PartStart(start)) => {
							let prefix = match part_start::Kind::try_from(start.kind) {
								Ok(part_start::Kind::Text) => Some("**Assistant:** "),
								Ok(part_start::Kind::Thinking) => Some("**Thinking:** "),
								_ => None,
							};
							if let Some(prefix) = prefix {
								part_serial = part_serial.saturating_add(1);
								let id = fmts!("part-{part_serial}");
								app.ui_mut().update_component::<TranscriptView>("transcript", |view| {
									view.push(
										Markdown::new()
											.with(Prop::Id, id.as_str())
											.with(Prop::Partial, true),
									);
									true
								});
								active_parts.insert(
									start.index,
									ActivePart { id, text: StrMut::new_inline(""), prefix },
								);
							}
						},
						Some(Event::PartDelta(delta)) => {
							if let Some(active) = active_parts.get_mut(&delta.index)
								&& let Ok(fragment) = std::str::from_utf8(&delta.chunk)
							{
								active.text.push_str(fragment);
								let rendered = fmts!("{}{}", active.prefix, active.text.as_str());
								app.ui_mut().set_text(active.id.as_str(), rendered);
							}
						},
						Some(Event::PartEnd(end)) => {
							if let Some(active) = active_parts.remove(&end.index) {
								app.ui_mut().set_prop(active.id.as_str(), Prop::Partial, false);
							}
						},
						_ => {},
					},
					AgentEvent::ToolOpened { call_id, name, rev } => {
						let fold = ToolFold::new(call_id.clone(), name.clone(), rev.clone());
						tool_folds.insert(call_id.clone(), fold);
						push_tool_card(app.ui_mut(), call_id);
					},
					AgentEvent::ToolArgs { call_id, view, .. } => {
						if let Some(fold) = tool_folds.get_mut(call_id.as_str()) {
							fold.set_args_view(view.clone());
							renderers.update(app.ui_mut(), fold);
						}
					},
					AgentEvent::ToolUpdate { call_id, json } => {
						if let Some(fold) = tool_folds.get_mut(call_id.as_str()) {
							fold.push_update(json.clone());
							renderers.update(app.ui_mut(), fold);
						}
					},
					AgentEvent::ToolFinished { call_id, item } => {
						if let Some(fold) = tool_folds.get_mut(call_id.as_str()) {
							fold.item = Some(item.clone());
							fold.state = match &item.kind {
								Some(item::Kind::ToolResult(result)) if result.is_error => ToolState::Failure,
								Some(item::Kind::ToolResult(_)) => ToolState::Success,
								_ => {
									push_error(app.ui_mut(), format!("Tool {call_id} finished without a tool result."));
									ToolState::Failure
								},
							};
							renderers.update(app.ui_mut(), fold);
						}
					},
					AgentEvent::JobRegistered { job_id } => { live_jobs.insert(job_id.clone()); },
					AgentEvent::JobSettled { job_id } => { live_jobs.remove(job_id); },
					AgentEvent::Failed { message, .. } => push_error(app.ui_mut(), format!("Agent error: {message}")),
					_ => {},
				}
				update_status(
					app.ui_mut(),
					&session.session_id,
					&session_model,
					attempt_indicator,
					live_jobs.len(),
					session_cost_nanos,
					context_tokens,
					context_window,
					queued,
					events.dropped(),
				);
			},
		}
	}

	drop(tx);
	if tokio::time::timeout(Duration::from_secs(3), &mut agent_task)
		.await
		.is_err()
	{
		agent_task.abort();
		let _ = agent_task.await;
	}
	Ok(exit)
}

fn show_resume_picker(ui: &mut Ui, choices: &[ResumeChoice]) {
	if choices.is_empty() {
		push_error(ui, "No resumable sessions found in this project.");
		return;
	}

	let rows = u16::try_from(choices.len())
		.unwrap_or(u16::MAX)
		.min(12)
		.saturating_add(1);
	let mut select = Select::new()
		.with(Prop::Id, RESUME_SELECT_ID)
		.with(Prop::Filter, true)
		.with(Prop::H, rows);
	for choice in choices {
		select = select.option(
			SelectOption::new()
				.with(Prop::Value, choice.id.clone())
				.with(Prop::Desc, choice.detail.clone())
				.label(choice.label.clone()),
		);
	}
	let content = Col::new().child(select).child(
		TextLeaf::new()
			.with(Prop::Dim, true)
			.text("Type to filter · Enter resume · Esc cancel"),
	);
	let picker = Boxed::new()
		.with(Prop::Border, Border::Round)
		.with(Prop::Title, "Resume Session")
		.with(Prop::PadX, 1_u16)
		.child(content);
	ui.show_overlay(
		picker,
		OverlayOptions::default()
			.anchor(OverlayAnchor::Center)
			.width(Dim::Pct(80))
			.min_width(48)
			.max_height(Dim::Pct(75))
			.margin(OverlayMargin::uniform(1))
			.min_viewport(Size::new(24, 6)),
	);
}

fn show_rewind_picker(ui: &mut Ui, targets: &[RewindTarget]) {
	if targets.is_empty() {
		push_error(ui, "No user messages are available to rewind.");
		return;
	}

	let rows = u16::try_from(targets.len())
		.unwrap_or(u16::MAX)
		.min(12)
		.saturating_add(1);
	let total = targets.len();
	let mut select = Select::new()
		.with(Prop::Id, REWIND_SELECT_ID)
		.with(Prop::Filter, true)
		.with(Prop::H, rows);
	for (index, target) in targets.iter().enumerate().rev() {
		let label = target
			.text
			.lines()
			.next()
			.filter(|line| !line.is_empty())
			.unwrap_or("(empty message)");
		select = select.option(
			SelectOption::new()
				.with(Prop::Value, target.event.to_string())
				.with(Prop::Desc, format!("Turn {} of {total}", index + 1))
				.label(label),
		);
	}
	let content = Col::new().child(select).child(
		TextLeaf::new()
			.with(Prop::Dim, true)
			.text("Type to filter · Enter rewind · Esc cancel"),
	);
	let picker = Boxed::new()
		.with(Prop::Border, Border::Round)
		.with(Prop::Title, "Rewind History")
		.with(Prop::PadX, 1_u16)
		.child(content);
	ui.show_overlay(
		picker,
		OverlayOptions::default()
			.anchor(OverlayAnchor::Center)
			.width(Dim::Pct(80))
			.min_width(48)
			.max_height(Dim::Pct(75))
			.margin(OverlayMargin::uniform(1))
			.min_viewport(Size::new(24, 6)),
	);
}

fn select_model<'a>(
	state: &AgentState,
	catalog: &'a Catalog,
	requested: &Str,
) -> Option<&'a ModelSpec> {
	let spec = resolve_model(catalog, requested.as_str())?;
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
	let supports_login = provider
		.auth
		.iter()
		.filter_map(|auth_id| catalog.auth_spec(auth_id))
		.any(|auth| auth.kind != AuthSpecKind::None);
	if !supports_login {
		return Err(fmts!(
			"Provider `{}` does not support interactive authentication. Use `/login` to choose \
			 another provider.",
			provider.id
		));
	}
	Ok(Str::from(provider.id.as_str()))
}

fn start_provider_login(ui: &mut Ui, auth: &ChatAuth, provider: Str) {
	match auth.start(provider.clone()) {
		Ok(()) => push_notice(ui, fmts!("Starting authentication for `{provider}`…")),
		Err(error) => push_error(
			ui,
			fmts!(
				"Could not start authentication for provider `{provider}`: {error}. Use `/login \
				 {provider}` to try again."
			),
		),
	}
}

pub(crate) fn auth_input(kind: AuthPromptKind, value: String) -> AuthInput {
	match kind {
		AuthPromptKind::ApiKey => AuthInput::ApiKey(SecretString::from(value)),
		AuthPromptKind::AuthorizationCode => AuthInput::AuthorizationCode(SecretString::from(value)),
		AuthPromptKind::SessionToken => AuthInput::SessionToken(SecretString::from(value)),
		AuthPromptKind::PlainText => AuthInput::PlainText(Str::from(value)),
		AuthPromptKind::OptionalSecret => AuthInput::OptionalSecret(SecretString::from(value)),
		AuthPromptKind::Confirmation => AuthInput::DeviceConfirmed,
	}
}

pub(crate) fn show_auth_prompt(ui: &mut Ui, message: Str, kind: AuthPromptKind) {
	let placeholder = match kind {
		AuthPromptKind::Confirmation => "Press Enter to confirm",
		AuthPromptKind::OptionalSecret => "Enter optional provider response or press Enter to skip",
		_ => "Enter provider response",
	};
	let input = Input::new()
		.with(Prop::Id, "auth-secret")
		.with(Prop::Placeholder, placeholder)
		.with(Prop::Mask, login::prompt_masks_input(kind))
		.with(Prop::Submit, true);
	let content = Col::new()
		.with(Prop::Gap, 1_u16)
		.child(TextLeaf::new().text(message))
		.child(input)
		.child(
			TextLeaf::new()
				.with(Prop::Dim, true)
				.text("Enter submit · Esc cancel"),
		);
	let prompt = Boxed::new()
		.with(Prop::Border, Border::Round)
		.with(Prop::Title, "Provider Authentication")
		.with(Prop::PadX, 1_u16)
		.child(content);
	ui.show_overlay(
		prompt,
		OverlayOptions::default()
			.anchor(OverlayAnchor::Center)
			.width(Dim::Pct(70))
			.min_width(40)
			.max_height(Dim::Pct(50))
			.margin(OverlayMargin::uniform(1))
			.min_viewport(Size::new(24, 6)),
	);
}

async fn next_auth_event(auth: Option<&ChatAuth>) -> Option<ChatAuthEvent> {
	match auth {
		Some(auth) => auth.events.recv_async().await.ok(),
		None => pending().await,
	}
}

fn render_history(
	ui: &mut Ui,
	items: &[Item],
	renderers: &RendererRegistry,
	folds: &mut HashMap<Str, ToolFold>,
) {
	for item in items {
		match &item.kind {
			Some(item::Kind::Message(message)) => render_message(ui, message),
			Some(item::Kind::ToolCall(call)) => {
				let call_id = Str::from(call.id.as_str());
				let mut fold = ToolFold::new(
					call_id.clone(),
					Str::from(call.name.as_str()),
					tool_revision(item).unwrap_or(Rev { family: Str::new(""), n: 0 }),
				);
				if let Ok(args) = std::str::from_utf8(&call.args_json) {
					fold.set_args_view(omp_slopjson::parse_streaming(args));
				}
				push_tool_card(ui, &call_id);
				renderers.update(ui, &fold);
				folds.insert(call_id, fold);
			},
			Some(item::Kind::ToolResult(result)) => {
				let call_id = Str::from(result.call_id.as_str());
				if !folds.contains_key(call_id.as_str()) {
					let fold = ToolFold::new(
						call_id.clone(),
						Str::from(result.name.as_str()),
						tool_revision(item).unwrap_or(Rev { family: Str::new(""), n: 0 }),
					);
					push_tool_card(ui, &call_id);
					folds.insert(call_id.clone(), fold);
				}
				if let Some(fold) = folds.get_mut(call_id.as_str()) {
					fold.item = Some(item.clone());
					fold.state = if result.is_error {
						ToolState::Failure
					} else {
						ToolState::Success
					};
					renderers.update(ui, fold);
				}
			},
			_ => {},
		}
	}
}

fn render_then_deliver<R>(
	item: Item,
	render: impl FnOnce(&Item),
	deliver: impl FnOnce(Item) -> R,
) -> R {
	render(&item);
	deliver(item)
}

fn render_submitted_item(ui: &mut Ui, submitted: &Item) {
	if let Some(item::Kind::Message(message)) = &submitted.kind {
		render_message(ui, message);
	}
}

fn render_message(ui: &mut Ui, message: &Message) {
	let text = message
		.parts
		.iter()
		.filter_map(|part| match &part.kind {
			Some(part::Kind::Text(text)) => Some(text.clone()),
			Some(part::Kind::Blob(blob)) => {
				Some(format!("`[image {}, {} KB]`", blob.mime, blob.size.div_ceil(1024)))
			},
			_ => None,
		})
		.collect::<Vec<_>>()
		.join("\n");
	if text.is_empty() {
		return;
	}
	let label = match Role::try_from(message.role) {
		Ok(Role::User) => "User",
		Ok(Role::System) => "System",
		_ => "Assistant",
	};
	let rendered = format!("**{label}:** {text}");
	ui.update_component::<TranscriptView>("transcript", |view| {
		view.push(dom! { <markdown>{rendered}</markdown> });
		true
	});
}

fn tool_revision(item: &Item) -> Option<Rev> {
	let value = item.props.as_ref()?.fields.get(TOOL_REV_PROP)?;
	let value::Kind::String(revision) = value.kind.as_ref()? else {
		return None;
	};
	let (family, number) = revision
		.rsplit_once('.')
		.map_or(("", revision.as_str()), |(family, number)| (family, number));
	Some(Rev { family: Str::from(family), n: number.parse().ok()? })
}

fn push_tool_card(ui: &mut Ui, call_id: &Str) {
	ui.update_component::<TranscriptView>("transcript", |view| {
		view.push(ToolCard::new().with(Prop::Id, call_id.as_str()));
		true
	});
}

fn push_notice(ui: &mut Ui, message: impl Into<Str>) {
	let message = message.into();
	ui.update_component::<TranscriptView>("transcript", |view| {
		view.push(dom! { <markdown>{message}</markdown> });
		true
	});
}

fn handle_submit_ack(ui: &mut Ui, ack: SubmitAck, active_parts: &mut HashMap<u32, ActivePart>) {
	if ack != SubmitAck::Interrupted {
		return;
	}
	for active in active_parts.values() {
		ui.set_prop(active.id.as_str(), Prop::Partial, false);
	}
	active_parts.clear();
	ui.update_component::<TranscriptView>("transcript", |view| {
		view.push(Markdown::text_of("*Interrupted.*").with(Prop::Dim, true));
		true
	});
}

fn push_error(ui: &mut Ui, message: impl std::fmt::Display) {
	let rendered = format!("**Error:** {message}");
	ui.update_component::<TranscriptView>("transcript", |view| {
		view.push(dom! { <markdown>{rendered}</markdown> });
		true
	});
}

const fn startup_recovery_needed(pending_turn: bool, pending_input_submission: bool) -> bool {
	pending_turn || pending_input_submission
}

fn chat_active(submit_pending: bool, phase: AgentPhase) -> bool {
	submit_pending || phase != AgentPhase::Idle
}

pub fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis()
		.try_into()
		.unwrap_or(u64::MAX)
}

#[allow(clippy::too_many_arguments, reason = "status facts are independent display values")]
fn update_status(
	ui: &mut Ui,
	session_id: &Str,
	model: &str,
	attempt: u32,
	job_count: usize,
	cost_nanos: u64,
	context_tokens: u64,
	context_window: Option<u64>,
	queued: usize,
	dropped: u64,
) -> bool {
	ui.update_component::<Status>("status", |status| {
		let cost = (cost_nanos > 0).then(|| {
			let dollars = cost_nanos / 1_000_000_000;
			let fraction = cost_nanos % 1_000_000_000 / 100_000;
			Segment::new().label(format!("Cost: ${dollars}.{fraction:04}"))
		});
		let context = (context_tokens > 0).then(|| {
			let label = context_window.filter(|limit| *limit > 0).map_or_else(
				|| format!("Ctx: {context_tokens} tk"),
				|limit| {
					let percent = context_tokens
						.saturating_mul(100)
						.checked_div(limit)
						.unwrap_or(100)
						.min(100);
					format!("Ctx: {percent}%")
				},
			);
			Segment::new().label(label)
		});
		status.set_segments(
			[
				Some(Segment::new().label(format!("Session: {session_id}"))),
				(!model.is_empty()).then(|| Segment::new().label(model)),
				(attempt > 1).then(|| Segment::new().label(format!("Attempt: {attempt}"))),
				(job_count > 0).then(|| Segment::new().label(format!("Jobs: {job_count}"))),
				(queued > 0).then(|| Segment::new().label(format!("queued {queued}"))),
				cost,
				context,
				(dropped > 0).then(|| Segment::new().label(format!("Dropped: {dropped}"))),
			]
			.into_iter()
			.flatten(),
		);
		true
	})
}

#[cfg(test)]
mod tests {
	use std::cell::RefCell;

	use omp_tui::{UiContext, UiEvent};

	use super::*;

	#[test]
	fn submission_is_rendered_once_before_delivery() {
		let observations = RefCell::new(Vec::new());
		let ChatCommand::Submit(item) = parse_input("visible prompt").unwrap() else {
			panic!("plain input must be a submission");
		};
		render_then_deliver(
			*item,
			|item| {
				let Some(item::Kind::Message(message)) = &item.kind else {
					panic!("submission must be a message");
				};
				observations
					.borrow_mut()
					.push(("render", message.parts.len()));
			},
			|item| {
				let Some(item::Kind::Message(message)) = item.kind else {
					panic!("delivery must receive the message");
				};
				observations
					.borrow_mut()
					.push(("deliver", message.parts.len()));
			},
		);
		assert_eq!(&*observations.borrow(), &[("render", 1), ("deliver", 1)]);
	}

	#[test]
	fn startup_recovery_covers_both_durable_crash_windows() {
		assert!(!startup_recovery_needed(false, false));
		assert!(startup_recovery_needed(true, false));
		assert!(startup_recovery_needed(false, true));
		assert!(startup_recovery_needed(true, true));
	}
	#[test]
	fn chat_is_active_when_pending_or_phase_turning() {
		assert!(!chat_active(false, AgentPhase::Idle));
		assert!(chat_active(true, AgentPhase::Idle));
		assert!(chat_active(false, AgentPhase::Turning));
		assert!(chat_active(true, AgentPhase::Projecting));
	}

	#[test]
	fn selected_model_resolves_its_login_provider() {
		assert_eq!(model_provider(Catalog::embedded(), "kimi-code/k3").as_deref(), Some("kimi-code"));
	}
	#[test]
	fn explicit_login_provider_resolves_through_catalog_auth_specs() {
		assert_eq!(
			resolve_login_provider(Catalog::embedded(), &Str::from("kimi-code"))
				.expect("Kimi provider with interactive auth")
				.as_str(),
			"kimi-code"
		);
		let error =
			resolve_login_provider(Catalog::embedded(), &Str::from("provider-that-is-not-cataloged"))
				.expect_err("unknown provider");
		assert!(error.contains("Unknown provider `provider-that-is-not-cataloged`"));
		assert!(error.contains("`/login`"));
	}

	#[test]
	fn auth_handle_rejects_overlapping_logins() {
		let (request_tx, request_rx) = flume::bounded(1);
		let (answer_tx, _answer_rx) = flume::bounded(1);
		let (_event_tx, event_rx) = flume::unbounded();
		let active = Arc::new(AtomicBool::new(false));
		let auth = ChatAuth::new(request_tx, answer_tx, event_rx, Arc::clone(&active));
		auth.start(Str::from("kimi-code")).expect("start login");
		assert_eq!(request_rx.try_recv().expect("login request").as_str(), "kimi-code");
		assert_eq!(auth.start(Str::from("openai")), Err("authentication is already in progress"));
		active.store(false, Ordering::Release);
		assert!(!auth.is_active());
	}

	#[tokio::test]
	async fn auth_handle_delivers_device_prompt_url_and_terminal_progress() {
		let (request_tx, _request_rx) = flume::bounded(1);
		let (answer_tx, _answer_rx) = flume::bounded(1);
		let (event_tx, event_rx) = flume::unbounded();
		let auth = ChatAuth::new(request_tx, answer_tx, event_rx, Arc::new(AtomicBool::new(true)));
		let progress = [
			ChatAuthEvent::Notice(Str::from("Waiting for provider authorization…")),
			ChatAuthEvent::DeviceCode {
				code: Str::from("ABCD-1234"),
				url:  Str::from("https://kimi.example/device"),
			},
			ChatAuthEvent::Url(Str::from("https://kimi.example/authorize")),
			ChatAuthEvent::Prompt {
				message: Str::from("Confirm authorization"),
				kind:    AuthPromptKind::Confirmation,
			},
			ChatAuthEvent::Complete(Str::from("Authenticated Kimi.")),
			ChatAuthEvent::Failed(Str::from(
				"Authentication failed for provider `kimi-code`. Use `/login kimi-code`.",
			)),
		];
		for event in progress {
			event_tx.send(event).expect("auth progress receiver");
		}

		assert!(matches!(
			auth.next_event().await,
			Some(ChatAuthEvent::Notice(message)) if message.contains("Waiting")
		));
		assert!(matches!(
			auth.next_event().await,
			Some(ChatAuthEvent::DeviceCode { code, url })
				if code == "ABCD-1234" && url.contains("device")
		));
		assert!(matches!(
			auth.next_event().await,
			Some(ChatAuthEvent::Url(url)) if url.contains("authorize")
		));
		assert!(matches!(
			auth.next_event().await,
			Some(ChatAuthEvent::Prompt { kind: AuthPromptKind::Confirmation, .. })
		));
		assert!(matches!(auth.next_event().await, Some(ChatAuthEvent::Complete(_))));
		assert!(matches!(
			auth.next_event().await,
			Some(ChatAuthEvent::Failed(message)) if message.contains("/login kimi-code")
		));
	}
	#[test]
	fn status_updates_preserve_identity_and_replace_all_metrics() {
		let root = Status::new().with(Prop::Id, "status");
		let mut ui = Ui::from_root(root, 120, UiContext::default());
		assert!(update_status(
			&mut ui,
			&Str::from("test1"),
			"gpt-4o",
			2,
			3,
			1_500_000_000,
			450,
			Some(1000),
			4,
			5,
		));
		let mut queued_renderer = omp_tui::Renderer::new(Vec::new());
		ui.present(&mut queued_renderer, 10, 0).unwrap();
		let queued_painted = omp_tui::test_support::frame_row_text(ui.frame(), 0);
		assert!(queued_painted.contains("queued 4"));

		assert!(update_status(
			&mut ui,
			&Str::from("test2"),
			"claude-3",
			1,
			0,
			2_000_000_000,
			200,
			None,
			0,
			0,
		));

		let mut renderer = omp_tui::Renderer::new(Vec::new());
		ui.present(&mut renderer, 10, 0).unwrap();
		let painted = omp_tui::test_support::frame_row_text(ui.frame(), 0);

		assert!(painted.contains("test2"));
		assert!(painted.contains("claude-3"));
		assert!(painted.contains("Cost: $2.0000"));
		assert!(painted.contains("Ctx: 200 tk"));
		assert!(!painted.contains("Attempt:"));
		assert!(!painted.contains("Jobs:"));
		assert!(!painted.contains("queued"));
		assert!(!painted.contains("Dropped:"));
		assert!(!painted.contains("gpt-4o"));
		assert!(!painted.contains("test1"));
	}

	#[test]
	fn resume_picker_filters_and_submits_the_session_identity() {
		let mut ui = Ui::from_root(TextLeaf::new().text("chat"), 80, UiContext::default());
		let choices = [
			ResumeChoice {
				id:     Str::from("first"),
				label:  Str::from("Alpha session"),
				detail: Str::from("1h ago"),
			},
			ResumeChoice {
				id:     Str::from("second"),
				label:  Str::from("Beta session"),
				detail: Str::from("2h ago"),
			},
		];
		show_resume_picker(&mut ui, &choices);

		assert_eq!(ui.handle_key(Key::Char('b')), UiEvent::Filtered {
			id:    Str::from(RESUME_SELECT_ID),
			query: Str::from("b"),
			value: Some(Str::from("second")),
		});
		assert_eq!(ui.handle_key(Key::Enter), UiEvent::Changed {
			id:    Str::from(RESUME_SELECT_ID),
			value: Str::from("second"),
		});
	}

	#[test]
	fn rewind_picker_lists_newest_user_message_first() {
		let mut ui = Ui::from_root(TextLeaf::new().text("chat"), 80, UiContext::default());
		let targets = [
			RewindTarget { event: 11, keep: None, text: Str::from("first message\nextra detail") },
			RewindTarget { event: 22, keep: Some(11), text: Str::from("latest message") },
		];
		show_rewind_picker(&mut ui, &targets);

		assert_eq!(ui.handle_key(Key::Enter), UiEvent::Changed {
			id:    Str::from(REWIND_SELECT_ID),
			value: Str::from("22"),
		});
	}

	#[test]
	fn root_transcript_view_is_typed_and_updates_successfully() {
		let root = dom! {
			<col>
				{ TranscriptView::new().with(Prop::Id, "transcript") }
			</col>
		};
		let mut ui = Ui::from_root(root, 120, UiContext::default());
		let updated = ui.update_component::<TranscriptView>("transcript", |view| {
			view.push(dom! { <markdown>"Test Item"</markdown> });
			true
		});
		assert!(updated, "TranscriptView resolves to concrete type and accepts children");
		let mut renderer = omp_tui::Renderer::new(Vec::new());
		ui.present(&mut renderer, 10, 0).unwrap();
		let text = omp_tui::test_support::frame_row_text(ui.frame(), 0);
		assert!(text.contains("Test Item"));
	}
}
