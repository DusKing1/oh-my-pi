pub mod input;
pub mod renderers;

use std::{
	collections::{HashMap, HashSet},
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use omp_agent::{
	Agent, AgentEvent, AgentPhase, AgentState, Interrupt, InterruptClass, InterruptSource, TurnClient,
};
use omp_core::Str;
use omp_llm_catalog::{ModelKey, ModelSpec, snapshot::Catalog};
use omp_llm_inference::id::TurnId;
use omp_proto::{
	inference::v1::{part_start, turn_event::Event, value},
	thread::v1::{Item, Message, Part, Role, item, part},
};
use omp_tool::{Rev, TOOL_REV_PROP};
use omp_tui::{
	AppEvent, AppOptions, Key, Prop, Ui, dom,
	components::{Markdown, Segment, Status, ToolCard, ToolState, TranscriptView},
};

use crate::chat_ui::{
	input::{ChatCommand, parse_input},
	renderers::{RendererRegistry, ToolFold},
};

pub struct ChatUiSession {
	pub session_id:    Str,
	pub initial_items: Vec<Item>,
	pub context_window: Option<u64>,
}

struct ActivePart {
	id:     String,
	text:   String,
	prefix: &'static str,
}

pub async fn run<C: TurnClient + 'static>(
	mut agent: Agent<C>,
	session: ChatUiSession,
) -> anyhow::Result<()> {
	let bus = agent.events().clone();
	let mailbox = agent.mailbox();
	let events = bus.subscribe_ui(256);
	// Subscribe before the submit task can publish its first transition. This
	// feed is lossless because phase controls input routing, not decoration.
	let phase_events = bus.subscribe_lossless();
	let agent_state = agent.state().clone();

	let mut app = AppOptions::new()
		.start(|env| {
			let root = dom! {
				<col>
					<TranscriptView id="transcript" />
					<row>
						<input id="input" />
						<status id="status" />
					</row>
				</col>
			};
			Ui::from_root(root, env.viewport.width, env.ctx)
		})
		.await?;
	app.ui_mut().focus_first();

	let renderers = RendererRegistry::new();
	let mut tool_folds = HashMap::new();
	render_history(
		app.ui_mut(),
		&session.initial_items,
		&renderers,
		&mut tool_folds,
	);

	let mut session_model = agent_state.snapshot().turn.params.model.clone();
	let mut context_window = session.context_window;
	let mut session_cost_nanos = 0_u64;
	let mut live_jobs = HashSet::new();
	let mut attempt_indicator = 0;
	let mut context_tokens = 0_u64;
	let mut phase = AgentPhase::Idle;
	let mut active_parts: HashMap<u32, ActivePart> = HashMap::new();
	let mut part_serial = 0_u64;

	let (tx, rx) = flume::unbounded::<Item>();
	let (err_tx, err_rx) = flume::unbounded::<String>();
	let mut agent_task = tokio::spawn(async move {
		if agent.journal().pending_turn().is_some() {
			let resume_turn_id = TurnId::new(ulid::Ulid::generate().to_string());
			if let Err(error) = agent.submit(Vec::new(), resume_turn_id).await {
				let _ = err_tx.send(format!("**Startup resume error:** {error}"));
			}
		}
		while let Ok(item) = rx.recv_async().await {
			let turn_id = TurnId::new(ulid::Ulid::generate().to_string());
			if let Err(error) = agent.submit([item], turn_id).await {
				let _ = err_tx.send(format!("**Submit error:** {error}"));
			}
		}
	});

	'ui: loop {
		tokio::select! {
			event = app.next() => match event {
				Ok(Some(AppEvent::Submitted)) => {
					let text = app.ui().values()["input"].as_str().unwrap_or("").to_owned();
					app.ui_mut().set_text("input", "");
					match parse_input(&text) {
						Ok(ChatCommand::Model(requested)) => {
							match select_model(&agent_state, Catalog::embedded(), &requested) {
								Some(spec) => {
									session_model = spec.key.to_string();
									context_window = spec.limits.context_window;
								},
								None => push_error(app.ui_mut(), format!("Unknown model: {requested}")),
							}
						},
						Ok(ChatCommand::Resume) => push_error(
							app.ui_mut(),
							"Session already active; select another session with `omp chat --resume <id>`.",
						),
						Ok(ChatCommand::Quit) => break 'ui,
						Ok(ChatCommand::Submit(item)) if phase == AgentPhase::Idle => {
							if tx.send(item).is_err() {
								push_error(app.ui_mut(), "Agent input channel is closed.");
							}
						},
						Ok(ChatCommand::Submit(item)) => {
							let _ = mailbox.try_enqueue(Interrupt {
								class: InterruptClass::Immediate,
								item,
								source: InterruptSource::Producer(Str::new_static("user")),
							});
						},
						Err(error) => push_error(app.ui_mut(), error.to_string()),
					}
				},
				Ok(Some(AppEvent::Key(Key::Esc))) => {
					let _ = mailbox.try_enqueue(Interrupt {
						class: InterruptClass::Immediate,
						item: interrupt_item(),
						source: InterruptSource::Producer(Str::new_static("user")),
					});
				},
				Ok(Some(_)) => {},
				Ok(None) | Err(_) => break 'ui,
			},
			Ok(message) = err_rx.recv_async() => push_error(app.ui_mut(), message),
			Ok(phase_event) = phase_events.recv() => {
				if let AgentEvent::PhaseChanged { to, .. } = &*phase_event {
					phase = *to;
				}
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
								let id = format!("part-{part_serial}");
								app.ui_mut().update_component::<TranscriptView>("transcript", |view| {
									view.push(Markdown::new().with(Prop::Id, id.as_str()));
									true
								});
								active_parts.insert(start.index, ActivePart { id, text: String::new(), prefix });
							}
						},
						Some(Event::PartDelta(delta)) => {
							if let Some(active) = active_parts.get_mut(&delta.index)
								&& let Ok(fragment) = std::str::from_utf8(&delta.chunk)
							{
								active.text.push_str(fragment);
								app.ui_mut().set_text(&active.id, format!("{}{}", active.prefix, active.text));
							}
						},
						Some(Event::PartEnd(end)) => {
							active_parts.remove(&end.index);
						},
						_ => {},
					},
					AgentEvent::ToolOpened { call_id, name, rev } => {
						let fold = ToolFold::new(call_id.clone(), name.clone(), rev.clone());
						tool_folds.insert(call_id.clone(), fold);
						push_tool_card(app.ui_mut(), call_id);
					},
					AgentEvent::ToolArgs { call_id, fragment } => {
						if let Some(fold) = tool_folds.get_mut(call_id.as_str()) {
							if let Ok(fragment) = std::str::from_utf8(fragment) {
								fold.push_args(fragment);
							}
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
	if tokio::time::timeout(Duration::from_secs(3), &mut agent_task).await.is_err() {
		agent_task.abort();
		let _ = agent_task.await;
	}
	Ok(())
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
					fold.push_args(args);
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
					fold.state = if result.is_error { ToolState::Failure } else { ToolState::Success };
					renderers.update(ui, fold);
				}
			},
			_ => {},
		}
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
	let value::Kind::String(revision) = value.kind.as_ref()? else { return None };
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

fn interrupt_item() -> Item {
	Item {
		seq: 0,
		created_at_ms: now_ms(),
		kind: Some(item::Kind::Message(Message {
			role: i32::from(Role::User),
			parts: vec![Part {
				kind: Some(part::Kind::Text("User interrupted via Esc.".to_owned())),
			}],
		})),
		props: None,
	}
}

fn now_ms() -> u64 {
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
) {
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
					let percent = context_tokens.saturating_mul(100).checked_div(limit).unwrap_or(100).min(100);
					format!("Ctx: {percent}%")
				},
			);
			next = next.segment(Segment::new().label(context));
		}
		if dropped > 0 {
			next = next.segment(Segment::new().label(format!("Dropped: {dropped}")));
		}
		*status = next;
		true
	});
}

#[cfg(test)]
mod tests {
	use super::*;
	use omp_tui::UiContext;

	#[test]
	fn test_status_update() {
		let mut ui = Ui::from_root(dom! { <status id="status" /> }, 80, UiContext::default());
		update_status(&mut ui, &Str::from("test"), "gpt-4o", 2, 3, 1_500_000_000, 450, Some(1000), 5);
		let status = ui.values()["status"].as_array().unwrap().clone();
		assert!(status.iter().any(|v| v.as_str().unwrap().contains("Session: test")));
		assert!(status.iter().any(|v| v.as_str().unwrap().contains("gpt-4o")));
		assert!(status.iter().any(|v| v.as_str().unwrap().contains("Attempt: 2")));
		assert!(status.iter().any(|v| v.as_str().unwrap().contains("Jobs: 3")));
		assert!(status.iter().any(|v| v.as_str().unwrap().contains("Cost: $1.5000")));
		assert!(status.iter().any(|v| v.as_str().unwrap().contains("Ctx: 45%")));
		assert!(status.iter().any(|v| v.as_str().unwrap().contains("Dropped: 5")));
	}

	#[test]
	fn test_history_and_error_cards() {
		let mut ui = Ui::from_root(dom! { <TranscriptView id="transcript" /> }, 80, UiContext::default());
		
		// Error card
		push_error(&mut ui, "Test failure");
		assert!(ui.values()["transcript"].as_array().unwrap()[0].as_str().unwrap().contains("Error: Test failure"));
		
		// Tool slot preservation
		push_tool_card(&mut ui, &Str::from("tool-1"));
		ui.update_component::<TranscriptView>("transcript", |v| {
			v.push(Markdown::new().with(Prop::Id, "part-1"));
			true
		});
		ui.set_text("part-1", "Streaming text");
		
		let vals = ui.values()["transcript"].as_array().unwrap().clone();
		assert_eq!(vals.len(), 3); // Error, ToolCard, Markdown
	}
}
