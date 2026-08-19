//! Immediate-mode chat scene with append-only transcript and mutable tail
//! chrome.

use std::{
	cell::RefCell,
	fmt::Write as _,
	rc::Rc,
	time::{Duration, Instant},
};

use omp_core::{Str, StrMut, fmts};
use omp_tui::{
	Border, Charset, Color, Command, Component, Decor, DecorKind, Frame, Icon, Key, MouseReport,
	PaintCtx, Prop, Props, Rect, Size, SlashCommands, Slot, Style, Theme, Ui, UiContext, UiEvent,
	anim::{Easing, Shimmer, Tween},
	components::{Attachment, Attachments, EditorPane, Segment, Status},
	next_slot,
};
use smallvec::SmallVec;

use crate::{BackendEvent, StatusFacts, SubmitMode};

const LIVE_PANEL_ROWS: u16 = 12;
/// Column cap for inline tool-result images inside committed cards.
const TOOL_IMAGE_MAX_COLS: u16 = 64;
/// Row cap for inline tool-result images inside committed cards.
const TOOL_IMAGE_MAX_ROWS: u16 = 12;
const SHIMMER_PERIOD: Duration = Duration::from_millis(1900);
const BRAND_FADE: Duration = Duration::from_millis(450);
const FADE_FRAME: Duration = Duration::from_millis(40);
const STATUS_ID: &str = "status";
const INPUT_ID: &str = "input";

/// One retained chat document update and its exact repainted row ranges.
pub struct RenderedFrame<'a> {
	/// Complete logical document frame.
	pub frame:       &'a Frame,
	/// Final transcript prefix safe for native scrollback commits.
	pub stable_rows: u16,
	/// Half-open logical row ranges changed since the previous render.
	pub damage:      SmallVec<(u16, u16), 4>,
}

/// Result of routing one key through the focused composer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatKey {
	/// The composer handled the key.
	Consumed,
	/// The composer did not handle the key.
	Ignored,
	/// The scene requested host shutdown.
	Quit,
}

/// Presentation selected for a committed tool invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolKind {
	/// A command/output card.
	Command,
	/// A file-edit card.
	Edit,
}

#[derive(Clone, Copy)]
struct Span<'a> {
	text:  &'a str,
	style: Style,
}

impl<'a> Span<'a> {
	const fn new(text: &'a str, style: Style) -> Self {
		Self { text, style }
	}
}

struct RichText {
	text:  String,
	width: u16,
	view:  Option<Ui>,
}

impl RichText {
	fn new(text: impl Into<String>, width: u16, ctx: &UiContext) -> Self {
		let text = text.into();
		let view = Self::view(&text, width, ctx);
		Self { text, width, view }
	}

	fn view(text: &str, width: u16, ctx: &UiContext) -> Option<Ui> {
		(!text.contains("</md>"))
			.then(|| Ui::from_markup(format!("<md>{text}</md>"), width, ctx.clone()).ok())
			.flatten()
	}

	fn resize(&mut self, width: u16, ctx: &UiContext) {
		if self.width != width {
			self.width = width;
			self.view = Self::view(&self.text, width, ctx);
		}
	}

	fn height(&self) -> u16 {
		self
			.view
			.as_ref()
			.map_or_else(|| explicit_line_count(&self.text), Ui::height)
	}
}

struct UserEntry {
	body:  RichText,
	chips: Vec<Str>,
}

/// One persisted tool-result image with its probed pixel dimensions.
struct ToolImageEntry {
	source: Str,
	px:     omp_tui::imagefmt::ImageDimensions,
}

struct ToolEntry {
	title:   Str,
	kind:    ToolKind,
	ok:      bool,
	output:  Str,
	summary: Vec<Str>,
	images:  Vec<ToolImageEntry>,
}

struct LiveAssistant {
	id:   Str,
	text: StrMut,
}

struct LiveTool {
	id:     Str,
	name:   Str,
	title:  Str,
	output: StrMut,
	images: Vec<ToolImageEntry>,
}

enum Entry {
	User(UserEntry),
	Assistant(RichText),
	Tool(ToolEntry),
	Notice { text: Str, error: bool },
}

enum PreviewEntry<'a> {
	User(RichText, &'a [Str]),
	Assistant(RichText),
	Other(&'a Entry),
}

impl<'a> PreviewEntry<'a> {
	fn new(entry: &'a Entry, width: u16, ctx: &UiContext) -> Self {
		match entry {
			Entry::User(user) => Self::User(
				RichText::new(user.body.text.as_str(), Chat::message_width(width), ctx),
				&user.chips,
			),
			Entry::Assistant(body) => {
				Self::Assistant(RichText::new(body.text.as_str(), width.max(1), ctx))
			},
			Entry::Tool(_) | Entry::Notice { .. } => Self::Other(entry),
		}
	}

	fn height(&self, width: u16) -> u16 {
		match self {
			Self::User(body, chips) => body
				.height()
				.saturating_add(u16::from(!chips.is_empty()))
				.saturating_add(1),
			Self::Assistant(body) => body.height().saturating_add(1),
			Self::Other(entry) => Chat::entry_height(entry, width),
		}
	}

	fn draw(&self, frame: &mut Frame, y: u16, width: u16, ctx: &UiContext) {
		match self {
			Self::User(body, chips) => {
				draw_user_body(frame, y, body, chips, ctx);
			},
			Self::Assistant(body) => {
				draw_rich(frame, y, body, 0, width, ctx.theme);
			},
			Self::Other(entry) => {
				Chat::draw_entry(frame, entry, y, width, ctx);
			},
		}
	}
}

struct WorkState {
	facts: StatusFacts,
	fade:  Tween<Color>,
}

struct ChatStatus {
	props:   Props,
	slot:    Slot,
	work:    Rc<RefCell<WorkState>>,
	charset: Charset,
	theme:   Theme,
}

impl ChatStatus {
	fn new(work: Rc<RefCell<WorkState>>, charset: Charset, theme: Theme) -> Self {
		let mut props = Props::new();
		props.set(Prop::Id, STATUS_ID);
		props.set(Prop::NoSelect, true);
		Self { props, slot: next_slot(), work, charset, theme }
	}

	fn group(&self) -> Status {
		Status::new()
			.with(Prop::Bg, self.theme.panel)
			.with(Prop::Fg, self.theme.fg)
	}

	fn brand_segment(&self, now: Duration) -> Segment {
		let work = self.work.borrow();
		let label = if work.facts.working {
			let elapsed = work
				.facts
				.turn_started
				.map_or(Duration::ZERO, |started| Instant::now().saturating_duration_since(started));
			fmts!("{} {}", self.charset.spinner().at(now), elapsed_label(elapsed))
		} else {
			fmts!("{} omp", self.charset.icon(Icon::Omp))
		};
		Segment::new()
			.label(label)
			.with(Prop::Fg, work.fade.sample(now))
	}

	fn left_group(&self, now: Duration) -> Status {
		let model = self.work.borrow().facts.model.clone();
		self.group().segment(self.brand_segment(now)).segment(
			Segment::new()
				.label(fmts!("{} {model}", self.charset.icon(Icon::Model)))
				.with(Prop::Fg, self.theme.ok),
		)
	}

	fn right_group(&self) -> Status {
		let work = self.work.borrow();
		let facts = &work.facts;
		let mut status = self.group().with_str(Prop::Align, "right");
		if let Some(git) = &facts.git {
			let mut label =
				StrMut::new(fmts!("{} {}", self.charset.icon(Icon::Branch), git.branch).as_str());
			if git.dirty > 0 {
				let _ = write!(label, " *{}", git.dirty);
			}
			if git.staged > 0 {
				let _ = write!(label, " +{}", git.staged);
			}
			status = status.segment(Segment::new().label(label).with(Prop::Fg, self.theme.info));
		}
		if facts.context_tokens > 0 || facts.context_window.is_some() {
			let label = match facts.context_window {
				Some(window) if window > 0 => fmts!(
					"{} {:.1}%/{}",
					self.charset.icon(Icon::Context),
					facts.context_tokens as f64 * 100.0 / window as f64,
					compact_count(window)
				),
				_ => {
					fmts!("{} {}", self.charset.icon(Icon::Context), compact_count(facts.context_tokens))
				},
			};
			status = status.segment(Segment::new().label(label).with(Prop::Fg, self.theme.warn));
		}
		if facts.cost_nanos > 0 {
			status = status.segment(
				Segment::new()
					.label(fmts!("${:.4}", facts.cost_nanos as f64 / 1_000_000_000.0))
					.with(Prop::Fg, self.theme.secondary),
			);
		}
		if facts.queued > 0 {
			status = status.segment(
				Segment::new()
					.label(fmts!("queued {}", facts.queued))
					.with(Prop::Fg, self.theme.warn),
			);
		}
		if facts.jobs > 0 {
			status = status.segment(
				Segment::new()
					.label(fmts!("jobs {}", facts.jobs))
					.with(Prop::Fg, self.theme.info),
			);
		}
		if facts.attempt > 0 {
			status = status.segment(
				Segment::new()
					.label(fmts!("retry {}", facts.attempt))
					.with(Prop::Fg, self.theme.warn),
			);
		}
		if facts.dropped > 0 {
			status = status.segment(
				Segment::new()
					.label(fmts!("dropped {}", facts.dropped))
					.with(Prop::Fg, self.theme.err),
			);
		}
		status
	}
}

impl Component for ChatStatus {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn measure(&mut self, ctx: &UiContext) -> (u16, u16) {
		let mut left = self.left_group(Duration::ZERO);
		left.measure(ctx)
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		1
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		let mut left = self.left_group(pc.now);
		let mut right = self.right_group();
		let (_, left_width) = left.measure(pc.ctx);
		let (_, right_width) = right.measure(pc.ctx);
		if left_width.saturating_add(2).saturating_add(right_width) <= rect.width {
			left.paint(pc, Rect::new(rect.x, rect.y, left_width, 1));
			let dock = rect
				.x
				.saturating_add(rect.width)
				.saturating_sub(right_width);
			right.paint(pc, Rect::new(dock, rect.y, right_width, 1));
		} else {
			let mut combined = self.left_group(pc.now);
			let has_more = {
				let work = self.work.borrow();
				work.facts.git.is_some() || work.facts.context_tokens > 0 || work.facts.cost_nanos > 0
			};
			if has_more {
				combined = combined.segment(Segment::new().label("…").with(Prop::Fg, self.theme.muted));
			}
			combined.paint(pc, rect);
		}
		let work = self.work.borrow();
		let fade_frame = work
			.fade
			.settles_at()
			.min(pc.now.saturating_add(FADE_FRAME));
		let deadline = match (work.facts.working, work.fade.is_settled(pc.now)) {
			(true, true) => Some(pc.ctx.charset.spinner().next_change(pc.now)),
			(true, false) => Some(pc.ctx.charset.spinner().next_change(pc.now).min(fade_frame)),
			(false, false) => Some(fade_frame),
			(false, true) => None,
		};
		if let Some(at) = deadline {
			pc.wake(self.slot, at);
		}
	}

	fn paints_background(&self) -> bool {
		false
	}
}

/// Immediate-mode designed chat scene driven entirely by host data.
pub struct Chat {
	started_at:      Instant,
	ctx:             UiContext,
	editor_ui:       Ui,
	attachments:     Attachments,
	pending_submit:  Option<(String, Vec<Attachment>, SubmitMode)>,
	copied:          Option<Str>,
	work:            Rc<RefCell<WorkState>>,
	session_title:   Str,
	transcript:      Vec<Entry>,
	drawn_entries:   usize,
	transcript_rows: u16,
	last_viewport:   Size,
	height_floor:    u16,
	frame:           Frame,
	live_assistant:  Option<LiveAssistant>,
	live_tools:      Vec<LiveTool>,
	live_revision:   u64,
	drawn_live:      u64,
	last_working:    bool,
	right_inset:     u16,
}

impl Chat {
	/// Creates an empty scene using the host's detected presentation context.
	pub fn new(ctx: &UiContext) -> Self {
		let work = Rc::new(RefCell::new(WorkState {
			facts: StatusFacts::default(),
			fade:  Tween::settled(ctx.theme.muted),
		}));
		let pane = EditorPane::new()
			.with(Prop::Id, INPUT_ID)
			.with(Prop::Submit, true)
			.with(Prop::Placeholder, "Ask anything…")
			.status(ChatStatus::new(Rc::clone(&work), ctx.charset, ctx.theme));
		let attachments = pane.attachments();
		let mut editor_ui = Ui::from_root(pane, 0, ctx.clone());
		editor_ui.focus_first();
		Self {
			started_at: Instant::now(),
			ctx: ctx.clone(),
			editor_ui,
			attachments,
			pending_submit: None,
			copied: None,
			work,
			session_title: Str::default(),
			transcript: Vec::new(),
			drawn_entries: 0,
			transcript_rows: 0,
			last_viewport: Size::new(0, 0),
			height_floor: 0,
			frame: Frame::new(Size::new(0, 0)),
			live_assistant: None,
			live_tools: Vec::new(),
			live_revision: 0,
			drawn_live: 0,
			last_working: false,
			right_inset: 0,
		}
	}

	/// Routes a key through the composer.
	pub fn handle_key(&mut self, key: Key) -> ChatKey {
		if key == Key::Enter && self.composer_empty() && self.is_working() {
			self.pending_submit = Some((String::new(), Vec::new(), SubmitMode::Steer));
			return ChatKey::Consumed;
		}
		if key == Key::FollowUp {
			self.stage_submission(SubmitMode::FollowUp);
			return ChatKey::Consumed;
		}
		match self.editor_ui.handle_key(key) {
			UiEvent::Submit => {
				self.stage_submission(SubmitMode::Steer);
				ChatKey::Consumed
			},
			UiEvent::Copied(text) => {
				self.copied = Some(text);
				ChatKey::Consumed
			},
			UiEvent::None if key == Key::Ctrl('c') => ChatKey::Quit,
			UiEvent::None if key == Key::Esc => ChatKey::Ignored,
			UiEvent::None => ChatKey::Consumed,
			_ => ChatKey::Consumed,
		}
	}

	/// Stages the composer's non-empty text as a pending submission and
	/// clears the input; staged attachments ride along unless the text's
	/// slash command preserves them.
	fn stage_submission(&mut self, mode: SubmitMode) {
		let text = self.composer_text();
		if text.trim().is_empty() {
			return;
		}
		let attachments = if preserves_attachments(&text) {
			Vec::new()
		} else {
			self.attachments.take()
		};
		self.pending_submit = Some((text, attachments, mode));
		self.editor_ui.set_text(INPUT_ID, "");
		self.refresh_composer();
	}

	/// Routes sanitized bracketed-paste text through the composer.
	pub fn handle_paste(&mut self, text: &str) {
		let _ = self.editor_ui.handle_paste(text);
		self.refresh_composer();
	}

	/// Routes clipboard text verbatim, bypassing attachment staging.
	pub fn handle_paste_raw(&mut self, text: &str) {
		let _ = self.editor_ui.handle_paste_raw(text);
		self.refresh_composer();
	}

	/// Routes a document-space mouse report into the composer.
	pub fn handle_mouse(&mut self, report: &MouseReport) {
		let rows = self.composer_rows();
		let y = self.frame.size().height.saturating_sub(rows);
		if report.row >= y && report.row < y.saturating_add(rows) {
			let _ = self
				.editor_ui
				.handle_mouse(report.col, report.row - y, report.kind);
		}
	}

	/// Takes text copied or cut by the composer.
	pub fn take_copied(&mut self) -> Option<Str> {
		self.copied.take()
	}

	/// Takes the next composer submission: its text, staged attachments,
	/// and active-turn delivery mode.
	pub fn take_submission(&mut self) -> Option<(String, Vec<Attachment>, SubmitMode)> {
		self.pending_submit.take()
	}

	/// Returns whether the composer contains no non-whitespace text.
	pub fn composer_empty(&self) -> bool {
		self.composer_text().trim().is_empty()
	}

	/// Replaces composer text, preserving staged attachments.
	pub fn set_composer_text(&mut self, text: &str) {
		self.editor_ui.set_text(INPUT_ID, text);
		self.refresh_composer();
	}

	/// Returns the composer block height used for pointer hit testing.
	pub const fn composer_rows(&self) -> u16 {
		self.editor_ui.height()
	}

	/// Returns whether the latest status snapshot says a turn is active.
	pub fn is_working(&self) -> bool {
		self.work.borrow().facts.working
	}

	/// Returns a copy of the latest status snapshot.
	pub fn status(&self) -> StatusFacts {
		self.work.borrow().facts.clone()
	}

	/// Replaces slash-command completion data.
	pub fn set_slash_commands(&mut self, commands: Vec<Command>) {
		self
			.editor_ui
			.update_component::<EditorPane>(INPUT_ID, |pane| {
				pane.set_completion(Box::new(SlashCommands::new(commands)));
				true
			});
	}

	/// Reserves right-edge columns for host-composited chrome.
	pub const fn set_right_inset(&mut self, cols: u16) {
		self.right_inset = cols;
	}

	/// Appends a committed user message.
	pub fn push_user(&mut self, text: impl Into<String>, chips: Vec<Str>) {
		self.transcript.push(Entry::User(UserEntry {
			body: RichText::new(text, Self::message_width(self.last_viewport.width), &self.ctx),
			chips,
		}));
	}

	/// Begins a live assistant message.
	pub fn begin_assistant(&mut self, id: impl Into<Str>) {
		self.live_assistant = Some(LiveAssistant { id: id.into(), text: StrMut::new("") });
		self.bump_live();
	}

	/// Appends a delta to a matching live assistant message.
	pub fn append_assistant(&mut self, id: &str, text: &str) {
		if let Some(message) = &mut self.live_assistant
			&& message.id.as_str() == id
		{
			message.text.push_str(text);
			self.bump_live();
		}
	}

	/// Commits a matching live assistant message into stable transcript rows.
	pub fn end_assistant(&mut self, id: &str) {
		if self
			.live_assistant
			.as_ref()
			.is_some_and(|message| message.id.as_str() == id)
		{
			let message = self
				.live_assistant
				.take()
				.expect("matching live assistant exists");
			self.transcript.push(Entry::Assistant(RichText::new(
				message.text.as_str(),
				Self::message_width(self.last_viewport.width),
				&self.ctx,
			)));
			self.bump_live();
		}
	}

	/// Begins a live tool card.
	pub fn tool_started(&mut self, id: impl Into<Str>, name: impl Into<Str>, title: impl Into<Str>) {
		self.live_tools.push(LiveTool {
			id:     id.into(),
			name:   name.into(),
			title:  title.into(),
			output: StrMut::new(""),
			images: Vec::new(),
		});
		self.bump_live();
	}

	/// Appends output to a matching live tool card.
	pub fn tool_output(&mut self, id: &str, chunk: &str) {
		if let Some(tool) = self
			.live_tools
			.iter_mut()
			.find(|tool| tool.id.as_str() == id)
		{
			tool.output.push_str(chunk);
			self.bump_live();
		}
	}

	/// Attaches a persisted PNG to a matching live tool card; the committed
	/// card renders it inline. Sources whose headers fail to probe are
	/// ignored, keeping the text fallback.
	pub fn tool_image(&mut self, id: &str, source: impl Into<Str>) {
		let source = source.into();
		let Some(px) = std::fs::read(source.as_str())
			.ok()
			.and_then(|bytes| omp_tui::imagefmt::dimensions(&bytes))
		else {
			return;
		};
		if let Some(tool) = self
			.live_tools
			.iter_mut()
			.find(|tool| tool.id.as_str() == id)
		{
			tool.images.push(ToolImageEntry { source, px });
			self.bump_live();
		}
	}

	/// Commits a matching live tool card with its terminal state.
	pub fn tool_finished(&mut self, id: &str, ok: bool, summary: Vec<Str>) {
		if let Some(index) = self
			.live_tools
			.iter()
			.position(|tool| tool.id.as_str() == id)
		{
			let tool = self.live_tools.remove(index);
			self.transcript.push(Entry::Tool(ToolEntry {
				kind: tool_kind(&tool.name),
				title: tool.title,
				ok,
				output: tool.output.freeze(),
				summary,
				images: tool.images,
			}));
			self.bump_live();
		}
	}

	/// Appends an informational transcript notice.
	pub fn push_notice(&mut self, text: impl Into<Str>) {
		self
			.transcript
			.push(Entry::Notice { text: text.into(), error: false });
	}

	/// Appends an error transcript notice.
	pub fn push_error(&mut self, text: impl Into<Str>) {
		self
			.transcript
			.push(Entry::Notice { text: text.into(), error: true });
	}

	/// Replaces the complete status snapshot.
	pub fn set_status(&mut self, facts: StatusFacts) {
		let now = self.started_at.elapsed();
		let mut work = self.work.borrow_mut();
		if work.facts.working != facts.working {
			work.fade.retarget(
				now,
				if facts.working {
					self.ctx.theme.ok
				} else {
					self.ctx.theme.muted
				},
				BRAND_FADE,
				Easing::EaseInOut,
			);
		}
		work.facts = facts;
		drop(work);
		self.editor_ui.invalidate(STATUS_ID);
		self.bump_live();
	}

	/// Replaces the session title shown in the air row.
	pub fn set_session_title(&mut self, title: impl Into<Str>) {
		self.session_title = title.into();
		self.bump_live();
	}

	/// Removes committed and live transcript content.
	pub fn clear_history(&mut self) {
		self.transcript.clear();
		self.live_assistant = None;
		self.live_tools.clear();
		self.drawn_entries = 0;
		self.transcript_rows = 0;
		self.height_floor = 0;
		self.last_viewport = Size::new(0, 0);
		self.bump_live();
	}

	/// Applies scene-owned backend mutations and returns events owned by host
	/// overlays.
	#[must_use]
	pub fn apply_backend_event(&mut self, event: BackendEvent) -> Option<BackendEvent> {
		match event {
			BackendEvent::UserReplayed { text, chips } => self.push_user(text.as_str(), chips),
			BackendEvent::AssistantBegin { id } => self.begin_assistant(id),
			BackendEvent::AssistantDelta { id, text } => {
				self.append_assistant(id.as_str(), text.as_str())
			},
			BackendEvent::AssistantEnd { id } => self.end_assistant(id.as_str()),
			BackendEvent::ToolStarted { id, name, title } => self.tool_started(id, name, title),
			BackendEvent::ToolOutput { id, chunk } => self.tool_output(id.as_str(), chunk.as_str()),
			BackendEvent::ToolImage { id, source } => self.tool_image(id.as_str(), source),
			BackendEvent::ToolFinished { id, ok, summary } => {
				self.tool_finished(id.as_str(), ok, summary)
			},
			BackendEvent::Notice(text) => self.push_notice(text),
			BackendEvent::Error(text) => self.push_error(text),
			BackendEvent::Status(facts) => self.set_status(facts),
			BackendEvent::SessionTitle(title) => self.set_session_title(title),
			BackendEvent::HistoryCleared => self.clear_history(),
			BackendEvent::Ack { interrupted } => {
				if interrupted {
					self.push_notice("Interrupted.");
				}
			},
			event @ (BackendEvent::OpenModelPicker { .. }
			| BackendEvent::ModelsUpdated { .. }
			| BackendEvent::Sessions(_)
			| BackendEvent::LoginProviders(_)
			| BackendEvent::RewindTargets(_)
			| BackendEvent::AuthPrompt { .. }
			| BackendEvent::AuthPromptClose) => return Some(event),
		}
		None
	}

	/// Updates the retained logical document and reports exact changed rows.
	pub fn render(&mut self, viewport: Size) -> RenderedFrame<'_> {
		self.render_at(viewport, self.started_at.elapsed())
	}

	/// Produces one throwaway viewport during an active resize gesture.
	pub fn render_resize_preview(&mut self, viewport: Size) -> Frame {
		let elapsed = self.started_at.elapsed();
		let mut frame = Frame::new(viewport);
		if viewport.width == 0 || viewport.height == 0 {
			return frame;
		}
		frame.fill(Rect::new(0, 0, viewport.width, viewport.height), base_style(self.ctx.theme));
		let composer_width = self.composer_width(viewport);
		if self.editor_ui.frame().size().width != composer_width {
			self.editor_ui.resize(composer_width);
		}
		self.editor_ui.tick(elapsed);
		let editor_height = self.composer_rows();
		let editor_y = viewport.height.saturating_sub(editor_height);
		let title_y = editor_y.saturating_sub(1);
		let working_y = title_y.saturating_sub(1);
		let panel_y = working_y
			.saturating_sub(1)
			.saturating_sub(LIVE_PANEL_ROWS + 2);
		self.draw_live_panel(
			&mut frame,
			Rect::new(0, panel_y, viewport.width, LIVE_PANEL_ROWS + 2),
			elapsed,
		);
		if self.is_working() {
			self.draw_working(&mut frame, working_y, elapsed);
		}
		self.draw_session_title(&mut frame, title_y);
		frame.blit(self.editor_ui.frame(), 0, editor_height, 0, editor_y);
		let mut remaining = panel_y;
		for entry in self.transcript.iter().rev() {
			if remaining == 0 {
				break;
			}
			let preview = PreviewEntry::new(entry, viewport.width, &self.ctx);
			let height = preview.height(viewport.width);
			if height <= remaining {
				remaining -= height;
				preview.draw(&mut frame, remaining, viewport.width, &self.ctx);
			} else {
				let mut scratch = Frame::new(Size::new(viewport.width, height));
				preview.draw(&mut scratch, 0, viewport.width, &self.ctx);
				frame.blit(&scratch, height - remaining, remaining, 0, 0);
				remaining = 0;
			}
		}
		frame
	}

	fn composer_text(&self) -> String {
		self.editor_ui.values()[INPUT_ID]
			.as_str()
			.unwrap_or_default()
			.to_owned()
	}

	fn refresh_composer(&mut self) {
		let width = self.editor_ui.frame().size().width;
		if width > 0 {
			self.editor_ui.resize(width);
		}
	}

	fn bump_live(&mut self) {
		self.live_revision = self.live_revision.wrapping_add(1);
	}

	fn composer_width(&self, viewport: Size) -> u16 {
		viewport.width.saturating_sub(self.right_inset).max(1)
	}

	fn render_at(&mut self, viewport: Size, elapsed: Duration) -> RenderedFrame<'_> {
		if viewport.width == 0 || viewport.height == 0 {
			self.last_viewport = viewport;
			self.height_floor = 0;
			self.drawn_entries = 0;
			self.transcript_rows = 0;
			self.frame = Frame::new(viewport);
			return RenderedFrame {
				frame:       &self.frame,
				stable_rows: 0,
				damage:      SmallVec::new(),
			};
		}
		let composer_width = self.composer_width(viewport);
		if self.editor_ui.frame().size().width != composer_width {
			self.editor_ui.resize(composer_width);
		}
		self.editor_ui.tick(elapsed);
		let editor_changed = self.editor_ui.take_frame_damage();
		let rebuild = self.last_viewport != viewport;
		if rebuild {
			self.last_viewport = viewport;
			self.height_floor = 0;
			self.drawn_entries = 0;
			self.transcript_rows = 0;
			for entry in &mut self.transcript {
				Self::resize_entry(entry, viewport.width, &self.ctx);
			}
		}
		let new_rows = self.transcript[self.drawn_entries..]
			.iter()
			.fold(0_u16, |rows, entry| rows.saturating_add(Self::entry_height(entry, viewport.width)));
		let transcript_rows = self.transcript_rows.saturating_add(new_rows);
		let editor_height = self.composer_rows();
		let natural_height = transcript_rows.saturating_add(Self::band_height(editor_height));
		self.height_floor = self.height_floor.max(natural_height);
		let document_height = self.height_floor.max(viewport.height);
		let transcript_damage_start = if rebuild { 0 } else { self.transcript_rows };
		let editor_y = document_height.saturating_sub(editor_height);
		let title_y = editor_y.saturating_sub(1);
		let working_y = title_y.saturating_sub(1);
		let panel_y = working_y
			.saturating_sub(1)
			.saturating_sub(LIVE_PANEL_ROWS + 2);
		let panel = Rect::new(0, panel_y, viewport.width, LIVE_PANEL_ROWS + 2);
		let repaint_suffix = rebuild || new_rows > 0;
		if rebuild {
			self.frame = Frame::new(Size::new(viewport.width, document_height));
		} else {
			self
				.frame
				.resize_height(document_height, base_style(self.ctx.theme));
		}
		if repaint_suffix {
			self.frame.fill(
				Rect::new(
					0,
					transcript_damage_start,
					viewport.width,
					document_height.saturating_sub(transcript_damage_start),
				),
				base_style(self.ctx.theme),
			);
		}
		let mut y = self.transcript_rows;
		for index in self.drawn_entries..self.transcript.len() {
			let used = Self::draw_entry(
				&mut self.frame,
				&self.transcript[index],
				y,
				viewport.width,
				&self.ctx,
			);
			y = y.saturating_add(used);
		}
		self.drawn_entries = self.transcript.len();
		self.transcript_rows = y;
		let spinner_active = !self.live_tools.is_empty();
		let live_changed = repaint_suffix || self.drawn_live != self.live_revision || spinner_active;
		if live_changed {
			self.draw_live_panel_owned(panel, elapsed);
		}
		let working = self.is_working();
		let working_changed = working != self.last_working;
		if !repaint_suffix && self.last_working && !working {
			self
				.frame
				.fill(Rect::new(0, working_y, viewport.width, 1), base_style(self.ctx.theme));
		}
		if working {
			self.draw_working_owned(working_y, elapsed);
		}
		if repaint_suffix || live_changed {
			self.draw_session_title_owned(title_y);
		}
		let hud = Rect::new(0, working_y, viewport.width, 2);
		if !self.frame.noselect().contains(&hud) {
			self.frame.push_noselect(hud);
		}
		if repaint_suffix || editor_changed {
			self
				.frame
				.blit(self.editor_ui.frame(), 0, editor_height, 0, editor_y);
		}
		let mut damage = SmallVec::new();
		if repaint_suffix {
			damage.push((transcript_damage_start, document_height));
		} else {
			if live_changed {
				damage.push((panel_y, panel_y.saturating_add(LIVE_PANEL_ROWS + 2)));
			}
			if working || working_changed {
				damage.push((working_y, working_y.saturating_add(1)));
			}
			if editor_changed {
				damage.push((editor_y, document_height));
			}
		}
		self.drawn_live = self.live_revision;
		self.last_working = working;
		RenderedFrame { frame: &self.frame, stable_rows: self.transcript_rows, damage }
	}

	const fn band_height(editor_height: u16) -> u16 {
		LIVE_PANEL_ROWS + 5 + editor_height
	}

	const fn message_width(width: u16) -> u16 {
		let narrowed = width.saturating_sub(3);
		if narrowed == 0 { 1 } else { narrowed }
	}

	fn resize_entry(entry: &mut Entry, width: u16, ctx: &UiContext) {
		let message_width = Self::message_width(width);
		match entry {
			Entry::User(user) => user.body.resize(message_width, ctx),
			Entry::Assistant(body) => body.resize(width.max(1), ctx),
			Entry::Tool(_) | Entry::Notice { .. } => {},
		}
	}

	fn entry_height(entry: &Entry, width: u16) -> u16 {
		match entry {
			Entry::User(user) => user
				.body
				.height()
				.saturating_add(u16::from(!user.chips.is_empty()))
				.saturating_add(1),
			Entry::Assistant(body) => body.height().saturating_add(1),
			Entry::Tool(tool) => tool_height(tool, width).saturating_add(1),
			Entry::Notice { text, .. } => flowed_height(text, width).saturating_add(1),
		}
	}

	fn draw_entry(frame: &mut Frame, entry: &Entry, y: u16, width: u16, ctx: &UiContext) -> u16 {
		match entry {
			Entry::User(user) => draw_user(frame, y, user, ctx),
			Entry::Assistant(body) => draw_rich(frame, y, body, 0, width, ctx.theme).saturating_add(1),
			Entry::Tool(tool) => draw_tool(frame, y, width, tool, ctx).saturating_add(1),
			Entry::Notice { text, error } => {
				let style = if *error {
					ink(ctx.theme.err)
				} else {
					ink(ctx.theme.muted).italic()
				};
				draw_flowed(
					frame,
					Rect::new(1, y, width.saturating_sub(2), frame.size().height.saturating_sub(y)),
					&[Span::new(text, style)],
				)
				.saturating_add(1)
			},
		}
	}

	fn draw_live_panel_owned(&mut self, rect: Rect, elapsed: Duration) {
		let ctx = self.ctx.clone();
		draw_live_panel_impl(
			&mut self.frame,
			rect,
			&self.live_assistant,
			&self.live_tools,
			&ctx,
			elapsed,
		);
	}

	fn draw_live_panel(&self, frame: &mut Frame, rect: Rect, elapsed: Duration) {
		draw_live_panel_impl(frame, rect, &self.live_assistant, &self.live_tools, &self.ctx, elapsed);
	}

	fn draw_working_owned(&mut self, y: u16, elapsed: Duration) {
		draw_working_impl(
			&mut self.frame,
			y,
			elapsed,
			self.ctx.charset.icon(Icon::Cancellable),
			self.ctx.native_decor,
			self.ctx.theme,
		);
	}

	fn draw_working(&self, frame: &mut Frame, y: u16, elapsed: Duration) {
		draw_working_impl(
			frame,
			y,
			elapsed,
			self.ctx.charset.icon(Icon::Cancellable),
			self.ctx.native_decor,
			self.ctx.theme,
		);
	}

	fn draw_session_title_owned(&mut self, y: u16) {
		draw_session_title_impl(
			&mut self.frame,
			y,
			self.right_inset,
			&self.session_title,
			self.ctx.theme,
		);
	}

	fn draw_session_title(&self, frame: &mut Frame, y: u16) {
		draw_session_title_impl(frame, y, self.right_inset, &self.session_title, self.ctx.theme);
	}
}

fn draw_live_panel_impl(
	frame: &mut Frame,
	rect: Rect,
	assistant: &Option<LiveAssistant>,
	tools: &[LiveTool],
	ctx: &UiContext,
	elapsed: Duration,
) {
	frame.fill(rect, base_style(ctx.theme));
	if assistant.is_none() && tools.is_empty() {
		return;
	}
	draw_box(
		frame,
		rect,
		ink(ctx.theme.border),
		panel_style(ctx.theme),
		ctx.charset,
		ctx.native_decor,
	);
	let mut y = rect.y.saturating_add(1);
	let bottom = rect.y.saturating_add(rect.height).saturating_sub(1);
	if let Some(message) = assistant {
		let used = draw_flowed(
			frame,
			Rect::new(
				rect.x.saturating_add(1),
				y,
				rect.width.saturating_sub(2),
				bottom.saturating_sub(y),
			),
			&[Span::new(message.text.as_str(), prose_style(ctx.theme))],
		);
		y = y.saturating_add(used).min(bottom);
	}
	for tool in tools {
		if y >= bottom {
			break;
		}
		let prefix = fmts!("{} {}", ctx.charset.spinner().at(elapsed), tool.title);
		draw_line(frame, rect.x.saturating_add(1), y, rect.width.saturating_sub(2), &[Span::new(
			&prefix,
			ink(ctx.theme.info),
		)]);
		y = y.saturating_add(1);
		for line in tool
			.output
			.as_str()
			.lines()
			.rev()
			.take(2)
			.collect::<Vec<_>>()
			.into_iter()
			.rev()
		{
			if y >= bottom {
				break;
			}
			draw_line(frame, rect.x.saturating_add(3), y, rect.width.saturating_sub(4), &[Span::new(
				line,
				ink(ctx.theme.muted),
			)]);
			y = y.saturating_add(1);
		}
	}
}

fn draw_working_impl(
	frame: &mut Frame,
	y: u16,
	elapsed: Duration,
	hint: &str,
	native: bool,
	theme: Theme,
) {
	if y >= frame.size().height || frame.size().width < 4 {
		return;
	}
	let label = "Working";
	let start = u16::from(frame.size().width >= 50);
	let mut column = start;
	let length = visible_width(hint)
		.saturating_add(visible_width(label))
		.saturating_add(1);
	let shimmer = Shimmer::new(elapsed, SHIMMER_PERIOD, length);
	let right = frame.size().width.saturating_sub(1);
	for (text, high) in [(hint, theme.info), (" ", theme.ok), (label, theme.ok)] {
		for grapheme in xutf::graphemes_str(text) {
			if column >= right {
				break;
			}
			let style = if native {
				ink(high)
			} else {
				shimmer.pick(column - start, ink(theme.border), ink(theme.muted), ink(high))
			};
			let next = frame.put(column, y, grapheme, style);
			if next == column {
				break;
			}
			column = next;
		}
	}
	if native {
		frame.push_decor(Decor {
			rect: Rect::new(start, y, column.saturating_sub(start), 1),
			kind: DecorKind::Shimmer { period: SHIMMER_PERIOD },
		});
	}
}

fn draw_session_title_impl(frame: &mut Frame, y: u16, right_inset: u16, title: &str, theme: Theme) {
	if title.is_empty() {
		return;
	}
	let width = frame.size().width.saturating_sub(right_inset);
	let title_width = visible_width(title);
	if y < frame.size().height && width >= title_width.saturating_add(2) {
		let x = width.saturating_sub(title_width.saturating_add(1));
		draw_line(frame, x, y, title_width, &[Span::new(title, ink(theme.border).italic())]);
	}
}

fn draw_user(frame: &mut Frame, y: u16, user: &UserEntry, ctx: &UiContext) -> u16 {
	draw_user_body(frame, y, &user.body, &user.chips, ctx)
}

fn draw_user_body(
	frame: &mut Frame,
	y: u16,
	body: &RichText,
	chips: &[Str],
	ctx: &UiContext,
) -> u16 {
	let mut at = y;
	if !chips.is_empty() {
		let mut x = frame.put(1, at, ctx.charset.icon(Icon::Image), ink(ctx.theme.warn));
		for chip in chips {
			x = frame.put(x, at, " ", ink(ctx.theme.muted));
			x = frame.put(x, at, chip, ink(ctx.theme.warn).bold());
		}
		at = at.saturating_add(1);
	}
	let gutter = fmts!("{} ", ctx.charset.cursor());
	frame.put(0, at, &gutter, ink(ctx.theme.ok));
	let used = draw_rich(frame, at, body, 3, body.width, ctx.theme);
	at.saturating_sub(y).saturating_add(used).saturating_add(1)
}

fn draw_rich(frame: &mut Frame, y: u16, body: &RichText, x: u16, width: u16, theme: Theme) -> u16 {
	if let Some(view) = &body.view {
		let height = view.height();
		frame.blit(view.frame(), 0, height, x, y);
		height
	} else {
		draw_flowed(frame, Rect::new(x, y, width, frame.size().height.saturating_sub(y)), &[
			Span::new(&body.text, prose_style(theme)),
		])
	}
}

fn draw_tool(frame: &mut Frame, y: u16, width: u16, tool: &ToolEntry, ctx: &UiContext) -> u16 {
	let margin = u16::from(width >= 50);
	let height = tool_height(tool, width);
	let rect = Rect::new(margin, y, width.saturating_sub(margin * 2), height);
	let state = if tool.ok { ctx.theme.ok } else { ctx.theme.err };
	draw_box(frame, rect, ink(state), panel_style(ctx.theme), ctx.charset, ctx.native_decor);
	let icon = if tool.ok {
		ctx.charset.check()
	} else {
		ctx.charset.icon(Icon::Error)
	};
	let title = fmts!("{icon} {}", tool.title);
	draw_line(frame, rect.x.saturating_add(1), y, rect.width.saturating_sub(2), &[Span::new(
		&title,
		ink(state).bold(),
	)]);
	let mut row = y.saturating_add(1);
	let bottom = y.saturating_add(height).saturating_sub(1);
	let lines: Vec<&str> = if tool.summary.is_empty() {
		tool.output.lines().collect()
	} else {
		tool.summary.iter().map(Str::as_str).collect()
	};
	for line in lines.into_iter().take(3) {
		if row >= bottom {
			break;
		}
		let color = if tool.kind == ToolKind::Edit {
			ctx.theme.info
		} else {
			ctx.theme.muted
		};
		draw_line(frame, rect.x.saturating_add(2), row, rect.width.saturating_sub(4), &[Span::new(
			line,
			ink(color),
		)]);
		row = row.saturating_add(1);
	}
	for image in &tool.images {
		let (cols, rows) = tool_image_box(image, width);
		if rows == 0 || row.saturating_add(rows) > bottom {
			break;
		}
		omp_tui::components::draw_image_inline(
			frame,
			ctx,
			rect.x.saturating_add(2),
			row,
			image.source.as_str(),
			cols,
			rows,
		);
		row = row.saturating_add(rows);
	}
	height
}

/// Aspect-fit cell box for one tool image inside a card of `width` columns.
fn tool_image_box(image: &ToolImageEntry, width: u16) -> (u16, u16) {
	let margin = u16::from(width >= 50);
	let interior = width
		.saturating_sub(margin * 2)
		.saturating_sub(4)
		.min(TOOL_IMAGE_MAX_COLS);
	if interior == 0 {
		return (0, 0);
	}
	omp_tui::components::image_cell_box(image.px, interior, TOOL_IMAGE_MAX_ROWS)
}

fn tool_height(tool: &ToolEntry, width: u16) -> u16 {
	let lines = if tool.summary.is_empty() {
		tool.output.lines().count()
	} else {
		tool.summary.len()
	};
	let image_rows = tool
		.images
		.iter()
		.fold(0_u16, |rows, image| rows.saturating_add(tool_image_box(image, width).1));
	u16::try_from(lines.min(3))
		.unwrap_or(3)
		.saturating_add(image_rows)
		.saturating_add(2)
		.max(3)
}

fn tool_kind(name: &str) -> ToolKind {
	let name = name.to_ascii_lowercase();
	if name.contains("edit") || name.contains("write") || name.contains("patch") {
		ToolKind::Edit
	} else {
		ToolKind::Command
	}
}

fn preserves_attachments(text: &str) -> bool {
	let first = text.trim().split_whitespace().next().unwrap_or_default();
	first.starts_with('/') && first.get(1..).is_some_and(|command| !command.contains('/'))
}

fn draw_box(
	frame: &mut Frame,
	rect: Rect,
	border: Style,
	fill: Style,
	charset: Charset,
	native: bool,
) {
	if rect.width < 2 || rect.height < 2 {
		return;
	}
	let (tl, tr, bl, br, h, v) = charset.border(Border::Round);
	let mut glyph = [0_u8; 4];
	frame.fill(rect, fill);
	frame.put(rect.x, rect.y, tl.encode_utf8(&mut glyph), border);
	frame.put(rect.x + rect.width - 1, rect.y, tr.encode_utf8(&mut glyph), border);
	frame.put(rect.x, rect.y + rect.height - 1, bl.encode_utf8(&mut glyph), border);
	frame.put(rect.x + rect.width - 1, rect.y + rect.height - 1, br.encode_utf8(&mut glyph), border);
	for x in rect.x + 1..rect.x + rect.width - 1 {
		frame.put(x, rect.y, h.encode_utf8(&mut glyph), border);
		frame.put(x, rect.y + rect.height - 1, h.encode_utf8(&mut glyph), border);
	}
	for y in rect.y + 1..rect.y + rect.height - 1 {
		frame.put(rect.x, y, v.encode_utf8(&mut glyph), border);
		frame.put(rect.x + rect.width - 1, y, v.encode_utf8(&mut glyph), border);
	}
	if native {
		frame.push_noselect(rect);
	}
}

fn draw_line(frame: &mut Frame, x: u16, y: u16, width: u16, spans: &[Span<'_>]) -> u16 {
	let right = x.saturating_add(width);
	let mut at = x;
	for span in spans {
		for grapheme in xutf::graphemes_str(span.text) {
			if at >= right {
				return at;
			}
			let next = frame.put(at, y, grapheme, span.style);
			if next == at {
				return at;
			}
			at = next;
		}
	}
	at
}

fn draw_flowed(frame: &mut Frame, rect: Rect, spans: &[Span<'_>]) -> u16 {
	if rect.width == 0 || rect.height == 0 {
		return 0;
	}
	let mut x = rect.x;
	let mut y = rect.y;
	let right = rect.x.saturating_add(rect.width);
	let bottom = rect.y.saturating_add(rect.height);
	for span in spans {
		for grapheme in xutf::graphemes_str(span.text) {
			if grapheme == "\n" {
				x = rect.x;
				y = y.saturating_add(1);
				if y >= bottom {
					return y.saturating_sub(rect.y);
				}
				continue;
			}
			let width = visible_width(grapheme);
			if x > rect.x && x.saturating_add(width) > right {
				frame.set_soft_wrap(y);
				x = rect.x;
				y = y.saturating_add(1);
			}
			if y >= bottom {
				return y.saturating_sub(rect.y);
			}
			x = frame.put(x, y, grapheme, span.style);
		}
	}
	y.saturating_sub(rect.y).saturating_add(1)
}

fn flowed_height(text: &str, width: u16) -> u16 {
	if width == 0 {
		return 0;
	}
	let mut rows = 1_u16;
	let mut column = 0_u16;
	for grapheme in xutf::graphemes_str(text) {
		if grapheme == "\n" {
			rows = rows.saturating_add(1);
			column = 0;
			continue;
		}
		let size = visible_width(grapheme);
		if column > 0 && column.saturating_add(size) > width {
			rows = rows.saturating_add(1);
			column = 0;
		}
		column = column.saturating_add(size);
	}
	rows
}

fn explicit_line_count(text: &str) -> u16 {
	u16::try_from(text.lines().count().max(1)).unwrap_or(u16::MAX)
}

fn elapsed_label(elapsed: Duration) -> Str {
	let seconds = elapsed.as_secs();
	if seconds < 60 {
		fmts!("{seconds}s")
	} else if seconds < 3_600 {
		fmts!("{}m", seconds / 60)
	} else {
		fmts!("{}h", (seconds / 3_600).min(99))
	}
}

fn compact_count(value: u64) -> Str {
	if value >= 1_000_000 {
		fmts!("{:.1}m", value as f64 / 1_000_000.0)
	} else if value >= 1_000 {
		fmts!("{:.0}k", value as f64 / 1_000.0)
	} else {
		fmts!("{value}")
	}
}

fn visible_width(text: &str) -> u16 {
	u16::try_from(xutf::width_str(text)).unwrap_or(u16::MAX)
}
const fn base_style(theme: Theme) -> Style {
	Style::new().fg(theme.fg)
}
const fn panel_style(theme: Theme) -> Style {
	Style::new().fg(theme.fg).bg(theme.panel)
}
const fn ink(color: Color) -> Style {
	Style::new().fg(color)
}
const fn prose_style(theme: Theme) -> Style {
	Style::new().fg(theme.muted).italic()
}

#[cfg(test)]
mod tests {
	use super::*;

	fn ctx() -> UiContext {
		UiContext::default()
	}

	fn row_text(frame: &Frame, row: u16) -> String {
		omp_tui::test_support::frame_row_text(frame, row)
	}

	#[test]
	fn mutation_api_commits_stable_rows_and_keeps_tail_anchored() {
		let mut chat = Chat::new(&ctx());
		chat.set_status(StatusFacts { model: Str::from("model-a"), ..StatusFacts::default() });
		chat.push_user("hello", vec![]);
		chat.begin_assistant("a");
		chat.append_assistant("a", "world");
		let before = chat.render(Size::new(80, 24)).stable_rows;
		chat.end_assistant("a");
		let composer_rows = chat.composer_rows();
		let rendered = chat.render(Size::new(80, 24));
		assert!(rendered.stable_rows > before);
		assert!(row_text(rendered.frame, rendered.stable_rows - 2).contains("world"));
		let bottom = rendered.frame.size().height;
		assert!(
			(bottom - composer_rows..bottom)
				.any(|row| row_text(rendered.frame, row).contains("model-a"))
		);
	}

	#[test]
	fn live_delta_damages_only_mutable_band() {
		let mut chat = Chat::new(&ctx());
		chat.push_user("stable", vec![]);
		let stable = chat.render(Size::new(80, 30)).stable_rows;
		chat.begin_assistant("a");
		chat.append_assistant("a", "stream");
		let rendered = chat.render(Size::new(80, 30));
		assert_eq!(rendered.stable_rows, stable);
		assert!(rendered.damage.iter().all(|(start, _)| *start >= stable));
	}

	#[test]
	fn resize_preview_does_not_mutate_retained_geometry() {
		let mut chat = Chat::new(&ctx());
		chat.push_user("a line that wraps when narrow", vec![]);
		let original = chat.render(Size::new(80, 24)).frame.size();
		let original_width = match &chat.transcript[0] {
			Entry::User(user) => user.body.width,
			_ => unreachable!(),
		};
		let preview = chat.render_resize_preview(Size::new(30, 12));
		assert_eq!(preview.size(), Size::new(30, 12));
		assert_eq!(chat.frame.size(), original);
		let retained_width = match &chat.transcript[0] {
			Entry::User(user) => user.body.width,
			_ => unreachable!(),
		};
		assert_eq!(retained_width, original_width);
	}

	#[test]
	fn status_uses_only_supplied_facts_and_git_is_optional() {
		let mut chat = Chat::new(&ctx());
		chat.set_status(StatusFacts { model: Str::from("real/model"), ..StatusFacts::default() });
		let composer_rows = chat.composer_rows();
		let frame = chat.render(Size::new(100, 24)).frame;
		let bottom = frame.size().height;
		let status = (bottom - composer_rows..bottom)
			.map(|row| row_text(frame, row))
			.collect::<Vec<_>>()
			.join(" ");
		assert!(status.contains("real/model"));
		assert!(!status.contains("main"));
	}

	#[test]
	fn chips_are_rendered_from_public_user_mutation() {
		let mut chat = Chat::new(&ctx());
		chat.push_user("inspect", vec![Str::from("image.png")]);
		let frame = chat.render(Size::new(80, 24)).frame;
		assert!((0..frame.size().height).any(|row| row_text(frame, row).contains("image.png")));
	}

	#[test]
	fn composer_selection_restyles_xml_without_losing_selection_background() {
		let context = ctx();
		let mut chat = Chat::new(&context);
		let viewport = Size::new(80, 24);
		let _ = chat.render(viewport);
		for character in "<tag>value</tag>".chars() {
			assert_eq!(chat.handle_key(Key::Char(character)), ChatKey::Consumed);
		}
		assert_eq!(chat.handle_key(Key::SelectAll), ChatKey::Consumed);
		let composer_rows = chat.composer_rows();
		let frame = chat.render(viewport).frame;
		let input_y = frame
			.size()
			.height
			.saturating_sub(composer_rows)
			.saturating_add(1);
		assert_eq!(frame.cell(2, input_y).style().background_color(), context.theme.selection);
	}

	#[test]
	fn slash_detours_preserve_staged_attachments_but_paths_submit_them() {
		let mut chat = Chat::new(&ctx());
		chat.attachments.push_text("payload");
		chat.set_composer_text("/models");
		assert_eq!(chat.handle_key(Key::Enter), ChatKey::Consumed);
		let (_, submitted, _) = chat.take_submission().expect("slash command submitted");
		assert!(submitted.is_empty());
		assert_eq!(chat.attachments.take().len(), 1);

		chat.attachments.push_text("payload");
		chat.set_composer_text("/wat");
		assert_eq!(chat.handle_key(Key::Enter), ChatKey::Consumed);
		let (_, submitted, _) = chat
			.take_submission()
			.expect("unknown slash command submitted");
		assert!(submitted.is_empty());
		assert_eq!(chat.attachments.take().len(), 1);

		chat.attachments.push_text("payload");
		chat.set_composer_text("/tmp/input.txt");
		assert_eq!(chat.handle_key(Key::Enter), ChatKey::Consumed);
		let (_, submitted, _) = chat.take_submission().expect("path submitted");
		assert_eq!(submitted.len(), 1);
	}

	#[test]
	fn empty_enter_while_working_emits_abort_signal_without_draining_chips() {
		let mut chat = Chat::new(&ctx());
		chat.attachments.push_text("payload");
		chat.set_status(StatusFacts { working: true, ..StatusFacts::default() });
		assert_eq!(chat.handle_key(Key::Enter), ChatKey::Consumed);
		let (text, submitted, mode) = chat
			.take_submission()
			.expect("working empty enter submitted");
		assert!(text.is_empty());
		assert!(submitted.is_empty());
		assert_eq!(mode, SubmitMode::Steer);
		assert_eq!(chat.attachments.take().len(), 1);
	}

	#[test]
	fn enter_steers_and_follow_up_queues() {
		let mut chat = Chat::new(&ctx());
		chat.set_composer_text("steer this");
		assert_eq!(chat.handle_key(Key::Enter), ChatKey::Consumed);
		let (text, _, mode) = chat.take_submission().expect("enter submits");
		assert_eq!(text, "steer this");
		assert_eq!(mode, SubmitMode::Steer);

		chat.set_composer_text("later please");
		assert_eq!(chat.handle_key(Key::FollowUp), ChatKey::Consumed);
		let (text, _, mode) = chat.take_submission().expect("follow-up submits");
		assert_eq!(text, "later please");
		assert_eq!(mode, SubmitMode::FollowUp);

		assert_eq!(chat.handle_key(Key::FollowUp), ChatKey::Consumed);
		assert!(chat.take_submission().is_none(), "empty follow-up stages nothing");
	}

	#[test]
	fn raw_scene_chrome_uses_the_supplied_theme() {
		let mut context = ctx();
		context.theme = Theme::for_appearance(omp_tui::Appearance::Light);
		let mut frame = Frame::new(Size::new(20, 5));
		let assistant = Some(LiveAssistant { id: Str::from("a"), text: StrMut::new("stream") });
		draw_live_panel_impl(
			&mut frame,
			Rect::new(0, 0, 20, 5),
			&assistant,
			&[],
			&context,
			Duration::ZERO,
		);

		let border = frame.cell(0, 0).style();
		assert_eq!(border.foreground_color(), context.theme.border);
		assert_eq!(frame.cell(10, 3).style().background_color(), context.theme.panel);
		assert_eq!(frame.cell(2, 1).style().foreground_color(), context.theme.muted);
	}

	#[test]
	fn clear_history_resets_stable_prefix() {
		let mut chat = Chat::new(&ctx());
		chat.push_notice("notice");
		assert!(chat.render(Size::new(60, 20)).stable_rows > 0);
		chat.clear_history();
		assert_eq!(chat.render(Size::new(60, 20)).stable_rows, 0);
	}

	#[test]
	fn tool_result_images_render_inline_in_committed_cards() {
		// pi UI-06/UI-20: image payloads returned by tools (including PDF
		// page screenshots) render inline in the committed card instead of
		// only appearing as a text label.
		let path =
			std::env::temp_dir().join(format!("omp-chat-tool-image-{}.png", std::process::id()));
		omp_tui::test_support::write_test_png(&path, 8, 8, [255, 0, 0]);
		let source = Str::from(path.to_string_lossy().as_ref());

		let mut chat = Chat::new(&ctx());
		chat.tool_started("t1", "read", "read page.pdf:p1.png");
		chat.tool_image("t1", source);
		chat.tool_finished("t1", true, vec![Str::from("rendered page 1")]);
		let frame = chat.render(Size::new(80, 40)).frame;
		std::fs::remove_file(&path).ok();
		assert!(
			(0..frame.size().height).any(|row| row_text(frame, row).contains('▀')),
			"committed tool card renders half-block image rows"
		);
		assert!(
			(0..frame.size().height).any(|row| row_text(frame, row).contains("rendered page 1")),
			"summary text stays alongside the inline image"
		);
	}

	#[test]
	fn undecodable_tool_image_keeps_the_text_card() {
		let mut chat = Chat::new(&ctx());
		chat.tool_started("t1", "shell", "shell ls");
		chat.tool_image("t1", "/nonexistent/omp-tool-image.png");
		chat.tool_finished("t1", true, vec![Str::from("done")]);
		let frame = chat.render(Size::new(80, 24)).frame;
		assert!((0..frame.size().height).any(|row| row_text(frame, row).contains("done")));
		assert!((0..frame.size().height).all(|row| !row_text(frame, row).contains('▀')));
	}
}
