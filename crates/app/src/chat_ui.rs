pub mod input;
pub mod renderers;

use std::{
	collections::{HashMap, HashSet},
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use omp_agent::{
	Agent, AgentEvent, AgentPhase, AgentState, Interrupt, InterruptClass, InterruptSource,
	MailboxSender, TurnClient,
};
use omp_core::{Str, StrMut, fmts};
use omp_llm_catalog::{ModelKey, ModelSpec, snapshot::Catalog};
use omp_llm_inference::id::TurnId;
use omp_proto::{
	inference::v1::{part_start, turn_event::Event, value},
	thread::v1::{Item, Message, Part, Role, item, part},
};
use omp_tool::{Rev, TOOL_REV_PROP};
use omp_tui::{
	App, AppEvent, AppOptions, Border, Dim, Key, OverlayAnchor, OverlayMargin, OverlayOptions, Prop,
	Size, Ui,
	components::{
		Boxed, Col, Markdown, Segment, Select, SelectOption, Status, TextLeaf, ToolCard, ToolState,
		TranscriptView,
	},
	dom,
};

use crate::chat_ui::{
	input::{ChatCommand, parse_input},
	renderers::{RendererRegistry, ToolFold},
};

const RESUME_SELECT_ID: &str = "resume-session";

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
	Ok(app)
}

/// Drives one durable session inside an existing inline chat host.
pub fn run<'a, C, R>(
	app: &'a mut App,
	mut agent: Agent<C>,
	session: ChatUiSession,
	mut list_sessions: R,
) -> impl Future<Output = anyhow::Result<ChatUiExit>> + 'a
where
	C: TurnClient + 'static,
	R: FnMut() -> anyhow::Result<Vec<ResumeChoice>> + 'a,
{
	async move {
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

		let renderers = RendererRegistry::new();
		let mut tool_folds = HashMap::new();
		render_history(app.ui_mut(), &session.initial_items, &renderers, &mut tool_folds);
		if replacing_session {
			app.rebuild_history();
		}

		let mut session_model = agent_state.snapshot().turn.params.model.clone();
		let mut context_window = session.context_window;
		let mut session_cost_nanos = 0_u64;
		let mut live_jobs = HashSet::new();
		let mut attempt_indicator = 0;
		let mut context_tokens = 0_u64;
		let mut submit_pending = startup_recovery_needed(
			agent.journal().pending_turn().is_some(),
			agent.journal().pending_input_submission().is_some(),
		);
		let mut active_parts: HashMap<u32, ActivePart> = HashMap::new();
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
			events.dropped(),
		);

		let (tx, rx) = flume::bounded::<Item>(1);
		let (err_tx, err_rx) = flume::unbounded::<String>();
		let (submit_ack_tx, submit_ack_rx) = flume::bounded::<()>(1);
		let mut agent_task = tokio::spawn(async move {
			if startup_recovery_needed(
				agent.journal().pending_turn().is_some(),
				agent.journal().pending_input_submission().is_some(),
			) {
				let resume_turn_id = TurnId::new(ulid::Ulid::generate().to_string());
				if let Err(error) = agent.submit(Vec::new(), resume_turn_id).await {
					let _ = err_tx.send(format!("**Startup resume error:** {error}"));
				}
				let _ = submit_ack_tx.send(());
			}
			while let Ok(item) = rx.recv_async().await {
				let turn_id = TurnId::new(ulid::Ulid::generate().to_string());
				if let Err(error) = agent.submit([item], turn_id).await {
					let _ = err_tx.send(format!("**Submit error:** {error}"));
				}
				let _ = submit_ack_tx.send(());
			}
		});
		let mut exit = ChatUiExit::Quit;
		'ui: loop {
			tokio::select! {
				event = app.next() => {
					while submit_ack_rx.try_recv() == Ok(()) {
						submit_pending = false;
					}
					let is_active = chat_active(submit_pending, bus.phase());
					match event {
					Ok(Some(trigger @ (AppEvent::Submitted | AppEvent::Key(Key::FollowUp)))) => {
						let text = app.ui().values()["input"].as_str().unwrap_or("").to_owned();
						app.ui_mut().set_text("input", "");
						match parse_input(&text) {
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
											events.dropped(),
										);
									},
									None => push_error(app.ui_mut(), format!("Unknown model: {requested}")),
								}
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
								enqueue_shutdown_interrupt(&mailbox, is_active);
								break 'ui;
							},
							Ok(ChatCommand::Submit(item)) => {
								if is_active {
									let class = if matches!(trigger, AppEvent::Key(Key::FollowUp)) {
										InterruptClass::Idle
									} else {
										InterruptClass::Immediate
									};
									render_then_deliver(
										*item,
										|item| render_submitted_item(app.ui_mut(), item),
										|item| {
											let _ = mailbox.try_enqueue(Interrupt {
												class,
												item,
												source: InterruptSource::Producer(Str::new_static("user")),
											});
										},
									);
								} else {
									let sent = render_then_deliver(
										*item,
										|item| render_submitted_item(app.ui_mut(), item),
										|item| {
											submit_pending = true;
											tx.send(item).is_ok()
										},
									);
									if !sent {
										submit_pending = false;
										push_error(app.ui_mut(), "Agent input channel is closed.");
									}
								}
							},
							Err(error) => push_error(app.ui_mut(), error.to_string()),
						}
					},
					Ok(Some(AppEvent::Changed { id, value })) if id.as_str() == RESUME_SELECT_ID => {
						exit = ChatUiExit::Resume(value);
						break 'ui;
					},
					Ok(Some(AppEvent::Key(Key::Esc))) => {
						if is_active {
							let _ = mailbox.try_enqueue(Interrupt {
								class: InterruptClass::Immediate,
								item: interrupt_item("User interrupted via Esc."),
								source: InterruptSource::Producer(Str::new_static("user")),
							});
						}
					},
					Ok(Some(_)) => {},
					Ok(None) | Err(_) => {
						enqueue_shutdown_interrupt(&mailbox, is_active);
						break 'ui;
					},
					}
				},
				Ok(message) = err_rx.recv_async() => push_error(app.ui_mut(), message),
				Ok(()) = submit_ack_rx.recv_async() => {
					submit_pending = false;
				},
				Ok(agent_event) = events.recv() => {
					match &*agent_event {
						AgentEvent::Turn { event: turn_event, .. } => match &turn_event.event {
							Some(Event::Outcome(outcome)) => {
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
			Some(part::Kind::Text(text)) => Some(text.as_str()),
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

fn enqueue_shutdown_interrupt(mailbox: &MailboxSender, is_active: bool) {
	if is_active {
		let _ = mailbox.try_enqueue(Interrupt {
			class:  InterruptClass::Immediate,
			item:   interrupt_item("User quit the active chat."),
			source: InterruptSource::Producer(Str::new_static("user")),
		});
	}
}

fn interrupt_item(text: &str) -> Item {
	Item {
		seq:           0,
		created_at_ms: now_ms(),
		kind:          Some(item::Kind::Message(Message {
			role:  i32::from(Role::User),
			parts: vec![Part { kind: Some(part::Kind::Text(text.to_owned())) }],
		})),
		props:         None,
	}
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
	dropped: u64,
) -> bool {
	ui.update_component::<Status>("status", |status| {
		let mut next = Status::new().segment(Segment::new().label(format!("Session: {session_id}")));
		if !model.is_empty() {
			next = next.segment(Segment::new().label(model));
		}
		if attempt > 1 {
			next = next.segment(Segment::new().label(format!("Attempt: {attempt}")));
		}
		if job_count > 0 {
			next = next.segment(Segment::new().label(format!("Jobs: {job_count}")));
		}
		if cost_nanos > 0 {
			let dollars = cost_nanos / 1_000_000_000;
			let fraction = cost_nanos % 1_000_000_000 / 100_000;
			next = next.segment(Segment::new().label(format!("Cost: ${dollars}.{fraction:04}")));
		}
		if context_tokens > 0 {
			let context = context_window.filter(|limit| *limit > 0).map_or_else(
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
			next = next.segment(Segment::new().label(context));
		}
		if dropped > 0 {
			next = next.segment(Segment::new().label(format!("Dropped: {dropped}")));
		}
		*status = next.with(Prop::Id, "status");
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
		let item = interrupt_item("visible prompt");
		render_then_deliver(
			item,
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
			5,
		));
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
