use std::{
	cell::RefCell,
	fmt::Write as _,
	rc::Rc,
	time::{Duration, Instant},
};

use omp_core::{Str, StrMut, fmts};
use omp_tui::{
	Border, Charset, Color, Command, Component, Decor, DecorFill, DecorKind, EditOutcome, Editor,
	EditorOptions, EventCtx, Flow, Frame, Hit, HitTag, Icon, Key, Mouse, MouseReport, PaintCtx,
	Prop, Props, Rect, Size, SlashCommands, Slot, Style, SuggestionDisplay, Theme, Ui, UiContext,
	anim::{Easing, Shimmer, Tween},
	components::{
		Attachment, Attachments, EditorPane, Segment, Status, attachment_color, chip_label,
	},
	next_slot,
	syntax::{SyntaxRun, highlight_xml},
};
use smallvec::SmallVec;
use xutf::Text;

/// Only panel boxes paint a background; the rest of the chrome is
/// transparent so the terminal's own backdrop shows through.
const PANEL: Color = Color::Rgb(12, 15, 18);
const TEXT: Color = Color::Rgb(194, 198, 204);
const MUTED: Color = Color::Rgb(110, 116, 124);
const FAINT: Color = Color::Rgb(72, 78, 86);
const GREEN: Color = Color::Rgb(81, 196, 112);
const CYAN: Color = Color::Rgb(62, 190, 203);
const PURPLE: Color = Color::Rgb(171, 119, 230);
const GOLD: Color = Color::Rgb(210, 167, 86);

const EDIT_BOX_HEIGHT: u16 = 4;
const MESSAGE_INTERVAL: Duration = Duration::from_millis(430);
const EMIT_INTERVAL: Duration = Duration::from_millis(700);
const LIVE_SHARD_ROWS: u16 = 12;
/// One crest sweep over the working line's ~57-cell padded track,
/// matching the classic 30 cells/second pace.
const SHIMMER_PERIOD: Duration = Duration::from_millis(1900);
const WORKING_MESSAGE: &str = "Implementing immutable seam commits";
/// Mock session title: the named task — distinct from the working
/// narration — resting right-aligned in the air row above the band.
const SESSION_TITLE: &str = "Immutable Seam Commits & Status Bar Rework";
/// Nerd-tier cancel hint; tests assert against this exact resolution.
#[cfg(test)]
const CANCEL_HINT: &str = Charset::NerdFont.icon(Icon::Cancellable);
/// Nerd/Unicode-tier composer prompt; tests assert this exact shape.
#[cfg(test)]
const INPUT_PROMPT: &str = "╰─";
/// How long the brand segment fades between the spinner and the omp brand.
const BRAND_FADE: Duration = Duration::from_millis(450);
/// Repaint cadence while the brand fade is in flight.
const FADE_FRAME: Duration = Duration::from_millis(40);
const STATUS_ID: &str = "status";

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

/// Previous mutable-row text and placement, retained for byte-run updates.
struct LiveRowCache {
	label:          StrMut,
	label_x:        u16,
	label_shard:    u16,
	label_progress: u64,
	label_valid:    bool,
	prefix_shard:   u16,
	prefix_phase:   u8,
	prefix_valid:   bool,
}
impl LiveRowCache {
	fn new() -> Self {
		Self {
			label:          StrMut::with_capacity(40),
			label_x:        0,
			label_shard:    0,
			label_progress: 0,
			label_valid:    false,
			prefix_shard:   0,
			prefix_phase:   0,
			prefix_valid:   false,
		}
	}
}

/// Paints `text` under the working-line crest, advancing `column`.
/// `start` anchors cell zero so every segment rides one sweep.
#[allow(clippy::too_many_arguments, reason = "immediate-mode painter threading frame state")]
fn draw_shimmer(
	frame: &mut Frame,
	column: &mut u16,
	start: u16,
	y: u16,
	right: u16,
	text: &str,
	shimmer: Shimmer,
	high: Style,
	native: bool,
) {
	for grapheme in xutf::graphemes_str(text) {
		if *column >= right {
			return;
		}
		let style = if native {
			high
		} else {
			shimmer.pick(*column - start, ink(FAINT), ink(MUTED), high)
		};
		let next = frame.put(*column, y, grapheme, style);
		if next == *column {
			return;
		}
		*column = next;
	}
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

/// The chat demo's slash-command palette. Editor completion is generic;
/// this list is the demo application's own — the `Ctrl+K` command palette
/// surfaces the same entries.
pub fn demo_commands() -> Vec<Command> {
	vec![
		Command::new(
			"security",
			"Plan, run, inspect, import, and compare OMP-native security scans",
			&[],
		)
		.with_args(&[
			("plan", "Draft a scan plan for this workspace", "[focus]"),
			("run", "Execute the current scan plan", ""),
			("inspect", "Browse findings from the last scan", "[finding-id]"),
			("import", "Import an external scan report", "<path>"),
			("compare", "Diff two scan runs", "<run-a> <run-b>"),
		])
		.with_hint("plan|run|inspect|import|compare"),
		Command::new("attach", "Stage an image attachment on the composer", &[]).with_hint("<path>"),
		Command::new("settings", "Open settings menu", &[]),
		Command::new("setup", "Open provider setup", &["providers"]).with_hint("[provider]"),
		Command::new("plan", "Toggle plan mode (agent plans before executing)", &[]),
		Command::new("plan-review", "Re-open the plan review for the latest plan", &[]),
		Command::new("vibe", "Toggle persistent fast worker sessions", &[]),
		Command::new("goal", "Toggle an autonomous objective for this session", &[])
			.with_hint("<objective>"),
		Command::new("guided-goal", "Interview you in chat, then set up goal mode", &[]),
		Command::new("queue", "Queue a message for after the agent yields", &[]),
		Command::new("switch", "Switch model for this session (same as alt+p)", &[]),
		Command::new("fast", "Toggle priority service tier", &[]),
		Command::new("computer", "Toggle the native computer-use tool", &[]),
		Command::new("vision", "Control inspect_image vision delegation", &[]),
		Command::new("prewalk", "Switch to a fast model at the next action", &[]),
		Command::new("advisor", "Toggle the second-model advisor", &[]),
		Command::new("export", "Export session to an HTML file", &[]),
		Command::new("dump", "Copy the session transcript to clipboard", &[]),
		Command::new("share", "Share session via an encrypted link", &[]),
		Command::new("collab", "Share this session live via a relay", &[]),
		Command::new("join", "Join a shared collab session", &[]),
		Command::new("leave", "Leave the collab session", &[]),
		Command::new("browser", "Toggle browser headless vs visible mode", &[]),
		Command::new("copy", "Pick conversation text or code to copy", &[]),
		Command::new("todo", "View or modify the agent's todo list", &[]),
		Command::new("session", "Session management commands", &[]),
		Command::new("jobs", "Show async background jobs status", &[]),
		Command::new("usage", "Show provider usage and limits", &[]),
		Command::new("stats", "Launch the local stats dashboard", &[]),
		Command::new("changelog", "Show changelog entries", &[]),
		Command::new("hotkeys", "Show all keyboard shortcuts", &[]),
		Command::new("tools", "Show tools currently visible to the agent", &[]),
		Command::new("context", "Show estimated context usage breakdown", &[]),
		Command::new("agents", "Open Agent Control Center dashboard", &[]),
		Command::new("branch", "Create a new branch from a previous message", &[]),
		Command::new("fork", "Create a new fork from a previous message", &[]),
		Command::new("tree", "Navigate the session tree", &[]),
		Command::new("login", "Login with an OAuth provider", &[]),
		Command::new("logout", "Logout from an OAuth provider", &[]),
		Command::new("mcp", "Manage MCP servers", &[]),
		Command::new("ssh", "Manage SSH hosts", &[]),
		Command::new("new", "Start a new session", &[]),
		Command::new("fresh", "Reset provider state without changing the transcript", &[]),
		Command::new("clear", "Clear conversation context, keeping the session", &[]),
		Command::new("drop", "Delete the current session and start a new one", &[]),
		Command::new("compact", "Manually compact the session context", &[]),
		Command::new("shake", "Drop heavy content from context", &[]),
		Command::new("handoff", "Hand off context to a new session", &[]),
		Command::new("resume", "Resume a different session", &[]),
		Command::new("btw", "Ask an ephemeral side question", &[]),
		Command::new("tan", "Run a background agent on tangential work", &[]),
		Command::new("omfg", "Forge a rule from a recurring complaint", &[]),
		Command::new("retry", "Retry the last failed agent turn", &[]),
		Command::new("debug", "Open the debug tools selector", &[]),
		Command::new("memory", "Inspect and operate memory maintenance", &[]),
		Command::new("rename", "Rename the current session", &[]),
		Command::new("move", "Move the session to a different directory", &[]),
		Command::new("add-dir", "Add a workspace directory", &[]),
		Command::new("remove-dir", "Remove a workspace directory", &[]),
		Command::new("dirs", "List this session's workspace directories", &[]),
		Command::new("marketplace", "Manage marketplace plugins", &[]),
		Command::new("plugins", "View and manage installed plugins", &[]),
		Command::new("reload-plugins", "Reload skills, commands, hooks, tools, and agents", &[]),
		Command::new("force", "Force the next turn to use a specific tool", &["force:"]),
		Command::new("live", "Start Codex-backed realtime voice mode", &[]),
		Command::new("pause", "Freeze all agents until resumed", &[]),
		Command::new("quit", "Quit the application", &["q"]),
	]
}

/// A submitted message rendered as Markdown (with embedded markup), cached
/// until the text or the content width changes.
struct Submission {
	text:  String,
	width: u16,
	/// `None` when the text can't be a markdown document (a literal
	/// `</md>`, or embedded interactive markup) — painted verbatim instead.
	view:  Option<Ui>,
}

impl Submission {
	fn new(text: String, width: u16, ctx: &UiContext) -> Self {
		let view = Self::view(&text, width, ctx);
		Self { text, width, view }
	}

	fn view(text: &str, width: u16, ctx: &UiContext) -> Option<Ui> {
		(!text.contains("</md>") && next_ref_tag(text).is_none())
			.then(|| Ui::from_markup(format!("<md>{text}</md>"), width, ctx.clone()).ok())
			.flatten()
	}

	fn resize(&mut self, width: u16, ctx: &UiContext) {
		if self.width != width {
			self.width = width;
			self.view = Self::view(&self.text, width, ctx);
		}
	}

	/// Rendered row count, including the fallback's own line count.
	fn height(&self) -> u16 {
		self
			.view
			.as_ref()
			.map_or_else(|| explicit_line_count(&self.text), Ui::height)
	}
}

/// One append-only transcript entry. The log is retained so a geometry
/// rebuild can replay every entry at the new width; between rebuilds each
/// entry is measured and painted exactly once, then never touched again.
enum Entry {
	/// The closed command box that opens the session.
	Command,
	/// The n-th scripted narration message.
	Message(usize),
	/// A finished shard's permanent result line.
	ShardDone(u16),
	/// A message submitted through the composer.
	Submitted(Box<Submission>),
}

/// Demo-specific focused editor leaf with completion and syntax rendering.
struct DemoInput {
	props:      Props,
	slot:       Slot,
	editor:     Rc<RefCell<Editor>>,
	outcome:    Rc<RefCell<Option<EditOutcome>>>,
	last_click: Option<(Instant, (u16, u16))>,
}

impl DemoInput {
	fn new(editor: Rc<RefCell<Editor>>, outcome: Rc<RefCell<Option<EditOutcome>>>) -> Self {
		Self { props: Props::new(), slot: next_slot(), editor, outcome, last_click: None }
	}

	/// Cells before the editor text: the two-cell prompt plus a gap —
	/// identical on every tier.
	const fn input_offset() -> u16 {
		3
	}

	/// The `╰─` composer prompt composed from the tier's round border.
	fn input_prompt(charset: Charset) -> Str {
		let (_, _, bl, _, horizontal, _) = charset.border(Border::Round);
		fmts!("{bl}{horizontal}")
	}

	const fn input_width(width: u16) -> u16 {
		width.saturating_sub(Self::input_offset()).saturating_sub(1)
	}

	fn paint_picker(pc: &mut PaintCtx<'_>, rect: Rect, y: u16, editor: &Editor) {
		let Some(picker) = editor.picker() else {
			return;
		};
		let (start, suggestions) = picker.visible_suggestions();
		let overflow = picker.len() > suggestions.len();
		let row_right = rect
			.x
			.saturating_add(rect.width.saturating_sub(u16::from(overflow)));
		let primary_width = suggestions
			.iter()
			.filter_map(|suggestion| match suggestion.display() {
				SuggestionDisplay::Text(name) => Some(visible_width(name).saturating_add(2)),
				SuggestionDisplay::Emoji { .. } => None,
			})
			.max()
			.unwrap_or(12)
			.clamp(12, 32);

		for (offset, suggestion) in suggestions.iter().enumerate() {
			let Ok(offset) = u16::try_from(offset) else {
				break;
			};
			let row = y.saturating_add(offset);
			if row >= pc.clip {
				break;
			}
			let selected = start + usize::from(offset) == picker.selected();
			let label = if selected { ink(GREEN) } else { ink(TEXT) };
			let description = if selected { ink(GREEN) } else { ink(MUTED) };
			pc.frame.put(
				rect.x,
				row,
				if selected {
					pc.ctx.charset.cursor()
				} else {
					"  "
				},
				label,
			);
			match suggestion.display() {
				SuggestionDisplay::Text(name) => {
					draw_line(
						pc.frame,
						rect.x.saturating_add(2),
						row,
						row_right.saturating_sub(rect.x.saturating_add(2)),
						&[Span::new(name, label)],
					);
					if let Some(text) = suggestion.description()
						&& rect.width > 40
					{
						let description_x = rect
							.x
							.saturating_add(2)
							.saturating_add(primary_width)
							.min(row_right);
						draw_line(
							pc.frame,
							description_x,
							row,
							row_right.saturating_sub(description_x),
							&[Span::new(text, description)],
						);
					}
				},
				SuggestionDisplay::Emoji { emoji, shortcode } => {
					let mut column = pc.frame.put(rect.x.saturating_add(2), row, emoji, label);
					column = pc.frame.put(column, row, "  ", label);
					if shortcode.starts_with(':') {
						pc.frame.put(column, row, shortcode, label);
					} else {
						column = pc.frame.put(column, row, ":", label);
						column = pc.frame.put(column, row, shortcode, label);
						pc.frame.put(column, row, ":", label);
					}
				},
			}
		}

		if overflow && !suggestions.is_empty() {
			let (track, thumb_glyph) = pc.ctx.charset.scrollbar();
			let track_x = rect.x.saturating_add(rect.width.saturating_sub(1));
			for offset in 0..suggestions.len() {
				let Ok(offset) = u16::try_from(offset) else {
					break;
				};
				pc.frame
					.put(track_x, y.saturating_add(offset), track, ink(FAINT));
			}
			let thumb = picker
				.selected()
				.saturating_mul(suggestions.len().saturating_sub(1))
				/ picker.len().saturating_sub(1);
			pc.frame.put(
				track_x,
				y.saturating_add(u16::try_from(thumb).unwrap_or(u16::MAX)),
				thumb_glyph,
				ink(GREEN),
			);
		}
	}
}

impl Component for DemoInput {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn measure(&mut self, _ctx: &UiContext) -> (u16, u16) {
		(6, 40)
	}

	fn height(&mut self, _ctx: &UiContext, width: u16) -> u16 {
		let editor = self.editor.borrow();
		editor
			.input_height_for(Self::input_width(width))
			.saturating_add(editor.picker_height())
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		pc.hits
			.push(Hit { rect, slot: self.slot, tag: HitTag::Press });
		let editor = self.editor.borrow();
		let input_x = rect.x.saturating_add(Self::input_offset());
		let input_width = Self::input_width(rect.width);
		let input_height = editor.input_height_for(input_width);
		let theme = Theme::default();
		let mut in_comment = false;
		for (offset, row) in editor.view(input_width).iter().enumerate() {
			let row_y = rect
				.y
				.saturating_add(u16::try_from(offset).unwrap_or(u16::MAX));
			if row_y >= pc.clip {
				break;
			}
			if offset == 0 {
				pc.frame
					.put(rect.x, row_y, &Self::input_prompt(pc.ctx.charset), ink(FAINT));
			}
			let mut spans: SmallVec<Span<'_>, 16> = SmallVec::new();
			if editor.options().xml {
				let (runs, next) = highlight_xml(row.text, &theme, in_comment);
				in_comment = next;
				push_row_spans(
					&editor,
					row.text,
					&runs,
					editor.selection_span(row),
					pc.ctx.theme.selection,
					&mut spans,
				);
			} else {
				push_row_spans(
					&editor,
					row.text,
					&[],
					editor.selection_span(row),
					pc.ctx.theme.selection,
					&mut spans,
				);
			}
			draw_line(pc.frame, input_x, row_y, input_width, &spans);
			if let Some(cursor_column) = row.cursor_column {
				if cursor_column >= visible_width(row.text)
					&& let Some(hint) = editor.inline_hint()
				{
					let hint_x = input_x.saturating_add(cursor_column).saturating_add(1);
					let width = input_width.saturating_sub(cursor_column.saturating_add(1));
					draw_line(pc.frame, hint_x, row_y, width, &[Span::new(
						hint.as_str(),
						ink(MUTED).dim(),
					)]);
				}
				pc.frame.set_cursor(
					input_x
						.saturating_add(cursor_column)
						.min(rect.x.saturating_add(rect.width.saturating_sub(2))),
					row_y,
				);
			}
		}
		Self::paint_picker(pc, rect, rect.y.saturating_add(input_height), &editor);
	}

	fn focusable(&self) -> bool {
		true
	}

	fn key(&mut self, _ec: &mut EventCtx<'_>, key: Key) -> Flow {
		let outcome = self.editor.borrow_mut().handle(key);
		*self.outcome.borrow_mut() = Some(outcome);
		// The editor owns every key while focused. In particular, an ignored
		// picker key must not escape into `Ui`'s focus-ring navigation; the
		// demo applies its quit policy from the recorded `EditOutcome`.
		Flow::Consumed
	}

	fn mouse(
		&mut self,
		_ec: &mut EventCtx<'_>,
		_tag: HitTag,
		at: (u16, u16),
		rect: Rect,
		mouse: Mouse,
	) -> Flow {
		let width = Self::input_width(rect.width);
		let row = usize::from(at.1.saturating_sub(rect.y));
		let column = at
			.0
			.saturating_sub(rect.x.saturating_add(Self::input_offset()));
		match mouse {
			Mouse::Click => {
				let now = Instant::now();
				let position = (at.0, at.1);
				let double_click = self
					.last_click
					.is_some_and(|(previous, previous_position)| {
						previous_position == position
							&& now.saturating_duration_since(previous) <= Duration::from_millis(400)
					});
				if double_click {
					self
						.editor
						.borrow_mut()
						.select_word_visual_row(row, column, width);
					self.last_click = None;
				} else {
					self
						.editor
						.borrow_mut()
						.set_cursor_visual_row(row, column, width);
					self.last_click = Some((now, position));
				}
				Flow::Consumed
			},
			Mouse::Drag => {
				self
					.editor
					.borrow_mut()
					.extend_selection_visual_row(row, column, width);
				self.last_click = None;
				Flow::Consumed
			},
			Mouse::Release => Flow::Consumed,
			Mouse::WheelUp | Mouse::WheelDown => {
				let delta = if mouse == Mouse::WheelUp { -1 } else { 1 };
				if self
					.editor
					.borrow()
					.scroll_rows(delta, width, usize::from(rect.height))
				{
					Flow::Consumed
				} else {
					Flow::Skip
				}
			},
			_ => Flow::Skip,
		}
	}

	fn paste(&mut self, _ec: &mut EventCtx<'_>, text: &str) -> Flow {
		if matches!(self.editor.borrow_mut().insert_text(text), EditOutcome::Changed) {
			Flow::Consumed
		} else {
			Flow::Skip
		}
	}
}
/// Whether the demo is working, and how the status bar's brand segment
/// blends between its two states.
struct WorkState {
	working: bool,
	/// When the current mode began; the working timer counts from here.
	since:   Duration,
	/// Brand foreground: [`GREEN`] while working, [`MUTED`] at rest.
	fade:    Tween<Color>,
}

/// Powerline status split into a left brand group — spinner and session
/// timer while working, the omp brand at rest, the foreground tweening
/// between the two so neither swap ever snaps — and a right-docked
/// session group (branch, context, cost). Panes too narrow for both
/// groups fall back to one left-anchored band that sheds from the tail.
struct DemoStatus {
	props:   Props,
	slot:    Slot,
	work:    Rc<RefCell<WorkState>>,
	model:   Rc<RefCell<Str>>,
	charset: Charset,
	right:   Status,
}

impl DemoStatus {
	fn new(work: Rc<RefCell<WorkState>>, model: Rc<RefCell<Str>>, charset: Charset) -> Self {
		let mut props = Props::new();
		props.set(Prop::Id, STATUS_ID);
		// HUD chrome: never part of host text selection.
		props.set(Prop::NoSelect, true);
		let right = Self::right_group(charset);
		Self { props, slot: next_slot(), work, model, charset, right }
	}

	/// One styled band-group shell on the shared dark backdrop.
	fn group() -> Status {
		Status::new()
			.with(Prop::Bg, Color::Rgb(18, 18, 18))
			.with(Prop::Fg, TEXT)
	}

	/// The brand segment at `now`: spinner plus session timer while
	/// working, the omp badge at rest, foreground riding the work fade.
	fn brand_segment(&self, now: Duration) -> Segment {
		let work = self.work.borrow();
		let brand = if work.working {
			fmts!(
				"{} {}",
				self.charset.spinner().at(now),
				elapsed_label(now.saturating_sub(work.since))
			)
		} else {
			fmts!("{} omp", self.charset.icon(Icon::Omp))
		};
		Segment::new()
			.label(brand)
			.with(Prop::Fg, work.fade.sample(now))
	}

	fn model_segment(&self) -> Segment {
		Segment::new()
			.label(fmts!("{} {}", self.charset.icon(Icon::Model), self.model.borrow()))
			.with(Prop::Fg, GREEN)
	}

	fn git_segment(charset: Charset) -> Segment {
		Segment::new()
			.label(fmts!("{} main *5 +9", charset.icon(Icon::Branch)))
			.with(Prop::Fg, CYAN)
	}

	fn context_segment(charset: Charset) -> Segment {
		Segment::new()
			.label(fmts!("{} 39.1%/1M", charset.icon(Icon::Context)))
			.with(Prop::Fg, GOLD)
	}

	fn cost_segment() -> Segment {
		Segment::new()
			.label("$60.07 (sub) + $8.65 (adv)")
			.with(Prop::Fg, PURPLE)
	}

	/// The left band group: brand and model.
	fn left_group(&self, now: Duration) -> Status {
		Self::group()
			.segment(self.brand_segment(now))
			.segment(self.model_segment())
	}

	/// The right band group: branch, context, and cost.
	fn right_group(charset: Charset) -> Status {
		Self::group()
			.with_str(Prop::Align, "right")
			.segment(Self::git_segment(charset))
			.segment(Self::context_segment(charset))
			.segment(Self::cost_segment())
	}

	/// Every segment in one band, for panes too narrow to split.
	fn combined(&self, now: Duration) -> Status {
		Self::group()
			.segment(self.brand_segment(now))
			.segment(self.model_segment())
			.segment(Self::git_segment(self.charset))
			.segment(Self::context_segment(self.charset))
			.segment(Self::cost_segment())
	}
}

impl Component for DemoStatus {
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
		self.combined(Duration::ZERO).measure(ctx)
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		1
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		let mut left = self.left_group(pc.now);
		let (_, left_width) = left.measure(pc.ctx);
		let (_, right_width) = self.right.measure(pc.ctx);
		if left_width.saturating_add(2).saturating_add(right_width) <= rect.width {
			left.paint(pc, Rect::new(rect.x, rect.y, left_width, 1));
			let dock = rect
				.x
				.saturating_add(rect.width)
				.saturating_sub(right_width);
			self
				.right
				.paint(pc, Rect::new(dock, rect.y, right_width, 1));
		} else {
			let mut combined = self.combined(pc.now);
			combined.paint(pc, rect);
		}
		let work = self.work.borrow();
		let fade_frame = work
			.fade
			.settles_at()
			.min(pc.now.saturating_add(FADE_FRAME));
		let deadline = match (work.working, work.fade.is_settled(pc.now)) {
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

/// One retained chat document update and its exact repainted row ranges.
pub struct RenderedFrame<'a> {
	pub(crate) frame:       &'a Frame,
	pub(crate) stable_rows: u16,
	pub(crate) damage:      SmallVec<(u16, u16), 4>,
}

/// Result of routing one key through the focused composer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DemoKey {
	/// The composer handled the key.
	Consumed,
	/// The composer did not handle the key.
	Ignored,
	/// The demo requested application exit.
	Quit,
}

/// Produces the animated transcript, work indicator, editor, and status
/// line demo.
pub struct Demo {
	started_at:         Instant,
	/// Detected presentation context shared by every retained subtree.
	ctx:                UiContext,
	/// Cancel hint resolved once through the context's charset.
	cancel_hint:        &'static str,
	editor_ui:          Ui,
	editor:             Rc<RefCell<Editor>>,
	edit_outcome:       Rc<RefCell<Option<EditOutcome>>>,
	work:               Rc<RefCell<WorkState>>,
	last_working:       bool,
	model:              Rc<RefCell<Str>>,
	/// Images staged on the composer, previewed above the status line.
	attachments:        Attachments,
	/// Append-only transcript log, replayed in full on geometry rebuilds.
	transcript:         Vec<Entry>,
	/// Entries already painted into the retained frame.
	drawn_entries:      usize,
	/// Rows covered by the drawn entries; doubles as `stable_rows`.
	transcript_rows:    u16,
	appended_messages:  usize,
	emitted_shards:     u16,
	last_viewport:      Size,
	height_floor:       u16,
	frame:              Frame,
	/// Geometry of the retained live panel chrome.
	live_panel:         Option<Rect>,
	/// Reusable text and placement state for the animated shard rows.
	live_rows:          [LiveRowCache; LIVE_SHARD_ROWS as usize],
	/// One build buffer rotated through the row caches without reallocating.
	live_label_scratch: StrMut,
	/// Columns reserved at the right edge for a composited rail; the
	/// editor and title dock against the remaining visible width.
	right_inset:        u16,
	/// The composer submitted `/switch`; the host opens the model picker.
	switch_requested:   bool,
}

impl Demo {
	/// Starts the demo's animation clock, presenting through the host's
	/// detected context.
	pub fn new(ctx: &UiContext) -> Self {
		let editor = Rc::new(RefCell::new({
			let mut editor = Editor::new(EditorOptions::default());
			editor.set_completion(Box::new(SlashCommands::new(demo_commands())));
			editor
		}));
		let edit_outcome = Rc::new(RefCell::new(None));
		let work = Rc::new(RefCell::new(WorkState {
			working: true,
			since:   Duration::ZERO,
			fade:    Tween::settled(GREEN),
		}));
		let model = Rc::new(RefCell::new(Str::new_static("Fable 5++")));
		let pane = EditorPane::new()
			.input(DemoInput::new(Rc::clone(&editor), Rc::clone(&edit_outcome)))
			.status(DemoStatus::new(Rc::clone(&work), Rc::clone(&model), ctx.charset));
		let attachments = pane.attachments();
		let editor_ui = Ui::from_root(pane, 0, ctx.clone());
		Self {
			started_at: Instant::now(),
			ctx: ctx.clone(),
			cancel_hint: ctx.charset.icon(Icon::Cancellable),
			editor_ui,
			editor,
			edit_outcome,
			work,
			last_working: true,
			model,
			attachments,
			transcript: vec![Entry::Command],
			drawn_entries: 0,
			transcript_rows: 0,
			appended_messages: 0,
			emitted_shards: 0,
			last_viewport: Size::new(0, 0),
			height_floor: 0,
			frame: Frame::new(Size::new(0, 0)),
			live_panel: None,
			live_rows: std::array::from_fn(|_| LiveRowCache::new()),
			live_label_scratch: StrMut::with_capacity(40),
			right_inset: 0,
			switch_requested: false,
		}
	}

	/// Routes a key through the editor and reports whether it was consumed
	/// or requests exit. Quit policy lives here, not in the editor: once
	/// the editor reports a key unused, `esc` first cancels running work and
	/// only quits at rest; `ctrl-c` always quits.
	pub fn handle_key(&mut self, key: Key) -> DemoKey {
		*self.edit_outcome.borrow_mut() = None;
		let _ = self.editor_ui.handle_key(key);
		let outcome = self
			.edit_outcome
			.borrow_mut()
			.take()
			.unwrap_or(EditOutcome::Ignored);
		match outcome {
			EditOutcome::Submitted(text) => {
				let trimmed = text.trim();
				if trimmed == "/switch" {
					self.switch_requested = true;
					return DemoKey::Consumed;
				}
				if let Some(path) = trimmed
					.strip_prefix("/attach")
					.filter(|rest| rest.is_empty() || rest.starts_with(' '))
				{
					let path = path.trim().to_string();
					if !path.is_empty() {
						self.attach_image(&path);
					}
					return DemoKey::Consumed;
				}
				let _ = self.attachments.take();
				self.refresh_composer();
				self
					.transcript
					.push(Entry::Submitted(Box::new(Submission::new(
						text,
						Self::message_width(self.last_viewport.width),
						&self.ctx,
					))));
				self.set_working(true, self.started_at.elapsed());
				DemoKey::Consumed
			},
			EditOutcome::Changed => {
				self.reconcile_attachments();
				DemoKey::Consumed
			},
			EditOutcome::Ignored => {
				if key == Key::Ctrl('c') {
					return DemoKey::Quit;
				}
				if key == Key::SelectAll || key == Key::Copy || key == Key::Cut {
					return DemoKey::Consumed;
				}
				if key != Key::Esc {
					return DemoKey::Ignored;
				}
				if self.work.borrow().working {
					self.set_working(false, self.started_at.elapsed());
					return DemoKey::Consumed;
				}
				DemoKey::Quit
			},
		}
	}

	/// Consumes a pending `/switch` request submitted through the composer.
	pub fn take_switch_request(&mut self) -> bool {
		std::mem::take(&mut self.switch_requested)
	}

	/// Takes text the composer copied or cut; the host owns the clipboard
	/// write (OSC 52 on the terminal, a detached native write in the GUI).
	pub fn take_copied(&self) -> Option<Str> {
		self.editor.borrow_mut().take_copied()
	}

	/// Height in rows of the composer block at the document tail; the GUI
	/// host routes plain pointer gestures there to the scene.
	pub fn composer_rows(&self) -> u16 {
		self.editor_ui.height()
	}

	/// Routes a document-space mouse report into the editor UI.
	pub fn handle_mouse(&mut self, report: &MouseReport) {
		let editor_height = self.editor_ui.height();
		let editor_y = self.frame.size().height.saturating_sub(editor_height);
		let editor_bottom = editor_y.saturating_add(editor_height);
		if report.row < editor_y || report.row >= editor_bottom {
			return;
		}
		let _ = self
			.editor_ui
			.handle_mouse(report.col, report.row - editor_y, report.kind);
	}

	/// Switches the work state and retargets the brand fade. The status bar
	/// repaints immediately and the fade departs from whatever color is on
	/// screen, so rapid cancel/resume never snaps.
	fn set_working(&mut self, working: bool, now: Duration) {
		{
			let mut work = self.work.borrow_mut();
			if work.working == working {
				return;
			}
			work.working = working;
			work.since = now;
			let target = if working { GREEN } else { MUTED };
			work
				.fade
				.retarget(now, target, BRAND_FADE, Easing::EaseInOut);
		}
		self.editor_ui.invalidate(STATUS_ID);
	}

	/// Reflects a session model switch in the status bar's model segment.
	pub fn set_model(&mut self, name: &str) {
		*self.model.borrow_mut() = Str::from(name);
		self.editor_ui.invalidate(STATUS_ID);
	}

	/// Routes sanitized bracketed paste text through the editor. Dropped
	/// paths to existing image files (quoted, escaped, `file://`, or
	/// multi-file) and any large paste collapse into composer attachment
	/// chips instead of raw text.
	pub fn handle_paste(&mut self, text: &str) {
		let paths = omp_tui::paste::dropped_paths(text);
		if !paths.is_empty()
			&& paths.iter().all(|path| {
				omp_tui::paste::is_image_path(path) && std::path::Path::new(path.as_str()).is_file()
			}) {
			for path in &paths {
				self.attach_image(path);
			}
			return;
		}
		if text.lines().count() > 10 || text.len() > 1000 {
			self.attach_paste(text);
			return;
		}
		let _ = self.editor_ui.handle_paste(text);
	}

	/// Routes Ctrl+Shift+V clipboard text into the composer verbatim: no
	/// attachment staging, no large-paste collapse — the text stays inline
	/// and editable.
	pub fn handle_paste_raw(&mut self, text: &str) {
		let _ = self.editor_ui.handle_paste_raw(text);
	}

	/// Stages `path` on the composer and mentions it in the prompt as an
	/// atomic `<icon> #N` chip expanding to `<ref image=N/>` on submit.
	fn attach_image(&mut self, path: &str) {
		let attachment = self.attachments.push_image(path);
		let payload = format!("<ref image={}/>", attachment.marker);
		self.insert_chip(&attachment, &payload);
	}

	/// Collapses a large paste into a staged attachment card and an atomic
	/// composer chip expanding back to the pasted text on submit.
	fn attach_paste(&mut self, text: &str) {
		let attachment = self.attachments.push_text(text);
		self.insert_chip(&attachment, text);
	}

	/// Inserts one attachment chip as an atomic editor reference.
	fn insert_chip(&mut self, attachment: &Attachment, payload: &str) {
		let chip = chip_label(attachment, self.ctx.charset);
		{
			let mut editor = self.editor.borrow_mut();
			let _ = editor.insert_reference(&chip, payload);
			let _ = editor.insert_text(" ");
		}
		self.refresh_composer();
	}

	/// Hides staged attachments whose chip the user deleted from the
	/// composer (and re-shows them after an undo). Presence is derived
	/// from the buffer's atomic ranges, never from text matching.
	fn reconcile_attachments(&mut self) {
		let charset = self.ctx.charset;
		let changed = {
			let editor = self.editor.borrow();
			let text = editor.text();
			let ranges = editor.atom_ranges();
			self.attachments.set_visible(|attachment| {
				let chip = chip_label(attachment, charset);
				ranges
					.iter()
					.any(|&(start, end)| text.get(start..end) == Some(chip.as_str()))
			})
		};
		if changed {
			self.refresh_composer();
		}
	}

	/// Relayouts the composer after out-of-band state changed its height.
	fn refresh_composer(&mut self) {
		let width = self.editor_ui.frame().size().width;
		if width > 0 {
			self.editor_ui.resize(width);
		}
	}

	/// Reserves `cols` at the right edge for a composited rail, so the
	/// composer's right-docked chrome stays visible beside it. The next
	/// render relayouts the editor at the narrowed width.
	pub const fn set_right_inset(&mut self, cols: u16) {
		self.right_inset = cols;
	}

	/// The width the composer may actually occupy at `viewport`.
	fn composer_width(&self, viewport: Size) -> u16 {
		viewport.width.saturating_sub(self.right_inset).max(1)
	}

	/// Updates the retained logical document and reports its repainted rows.
	pub fn render(&mut self, viewport: Size) -> RenderedFrame<'_> {
		self.render_at(viewport, self.started_at.elapsed())
	}

	fn render_at(&mut self, viewport: Size, elapsed: Duration) -> RenderedFrame<'_> {
		if viewport.width == 0 || viewport.height == 0 {
			self.last_viewport = viewport;
			self.height_floor = 0;
			self.drawn_entries = 0;
			self.transcript_rows = 0;
			self.live_panel = None;
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
		// Fires due animation wakes (the status bar's spinner and brand
		// fade) so the blit below picks up fresh retained pixels.
		self.editor_ui.tick(elapsed);
		let editor_changed = self.editor_ui.take_frame_damage();

		// A viewport change starts a fresh renderer session: replay the
		// whole transcript log at the new width. Between rebuilds the log
		// is append-only and every drawn row is final, so selections over
		// transcript text stay anchored to it in every terminal.
		let rebuild = self.last_viewport != viewport;
		if rebuild {
			self.last_viewport = viewport;
			self.height_floor = 0;
			self.drawn_entries = 0;
			self.transcript_rows = 0;
			let message_width = Self::message_width(viewport.width);
			for entry in &mut self.transcript {
				if let Entry::Submitted(submission) = entry {
					submission.resize(message_width, &self.ctx);
				}
			}
		}
		while self.appended_messages < Self::visible_messages(elapsed) {
			self.transcript.push(Entry::Message(self.appended_messages));
			self.appended_messages += 1;
		}
		while self.emitted_shards < Self::finished_shards(elapsed) {
			self.emitted_shards += 1;
			self.transcript.push(Entry::ShardDone(self.emitted_shards));
		}

		let mut new_rows = 0_u16;
		for entry in &self.transcript[self.drawn_entries..] {
			new_rows = new_rows.saturating_add(Self::entry_height(entry, viewport.width, &self.ctx));
		}
		let transcript_rows = self.transcript_rows.saturating_add(new_rows);
		let editor_height = self.editor_ui.height();
		// Native scrollback is append-only, so the logical document may
		// never shrink while the seam is live: band rows that close again
		// (extra input lines) become blank padding that heals as the
		// transcript grows.
		let natural_height = transcript_rows.saturating_add(Self::band_height(editor_height));
		self.height_floor = self.height_floor.max(natural_height);
		let document_height = self.height_floor.max(viewport.height);
		let transcript_damage_start = if rebuild { 0 } else { self.transcript_rows };
		let margin = u16::from(viewport.width >= 50);
		let content_width = viewport.width.saturating_sub(margin * 2);
		let editor_y = document_height.saturating_sub(editor_height);
		let title_y = editor_y.saturating_sub(1);
		let working_y = title_y.saturating_sub(1);
		let panel_height = LIVE_SHARD_ROWS + 2;
		let panel_y = working_y.saturating_sub(1).saturating_sub(panel_height);
		let panel = Rect::new(margin, panel_y, content_width, panel_height);
		let repaint_suffix = rebuild || new_rows > 0 || self.live_panel != Some(panel);
		if rebuild {
			self.frame = Frame::new(Size::new(viewport.width, document_height));
		} else {
			self.frame.resize_height(document_height, base_style());
		}
		if repaint_suffix {
			self.frame.fill(
				Rect::new(
					0,
					transcript_damage_start,
					viewport.width,
					document_height.saturating_sub(transcript_damage_start),
				),
				base_style(),
			);
		}

		// Paint the new transcript entries; rows above `transcript_rows`
		// are final and never repainted.
		let mut y = self.transcript_rows;
		for index in self.drawn_entries..self.transcript.len() {
			let used = self.draw_entry_at(index, y, viewport.width);
			y = y.saturating_add(used);
		}
		self.drawn_entries = self.transcript.len();
		self.transcript_rows = y;

		// The live band repaints in place at the bottom of the document.
		let animation_frame = Self::animation_frame(elapsed);
		let panel_changed = draw_live_panel(
			&mut self.frame,
			&mut self.live_rows,
			&mut self.live_label_scratch,
			panel,
			repaint_suffix,
			self.emitted_shards,
			animation_frame,
			self.ctx.charset,
			self.ctx.native_decor,
		);
		let working = self.work.borrow().working;
		let working_changed = self.last_working != working;
		if !repaint_suffix && self.last_working && !working {
			self
				.frame
				.fill(Rect::new(0, working_y, viewport.width, 1), base_style());
		}
		if working {
			Self::draw_working(
				&mut self.frame,
				working_y,
				elapsed,
				self.cancel_hint,
				self.ctx.native_decor,
			);
		}
		Self::draw_session_title(&mut self.frame, title_y, self.right_inset);
		// The activity row and ghost title are HUD chrome: excluded from
		// host text selection. Re-pushed only when a suffix repaint (which
		// shifts these rows) dropped the old mark.
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
			if panel_changed {
				damage.push((panel_y, panel_y.saturating_add(panel_height)));
			}
			if working || working_changed {
				damage.push((working_y, working_y.saturating_add(1)));
			}
			if editor_changed {
				damage.push((editor_y, document_height));
			}
		}
		self.last_working = working;
		self.live_panel = Some(panel);

		RenderedFrame { frame: &self.frame, stable_rows: self.transcript_rows, damage }
	}

	fn generation(elapsed: Duration) -> u64 {
		u64::try_from(elapsed.as_millis() / EMIT_INTERVAL.as_millis()).unwrap_or(u64::MAX)
	}

	fn animation_frame(elapsed: Duration) -> u64 {
		u64::try_from(elapsed.as_millis() / 80).unwrap_or(u64::MAX)
	}

	fn visible_messages(elapsed: Duration) -> usize {
		let interval = MESSAGE_INTERVAL.as_millis();
		usize::try_from(elapsed.as_millis() / interval + 1)
			.unwrap_or(usize::MAX)
			.min(4)
	}

	/// Shards whose permanent result line has been appended by `elapsed`:
	/// two per emit tick, capped well inside `u16` document heights.
	fn finished_shards(elapsed: Duration) -> u16 {
		u16::try_from(Self::generation(elapsed).saturating_mul(2).min(60_000))
			.expect("finished shard count is clamped")
	}

	/// Rows the bottom live band occupies: the shard panel, a blank
	/// separator, the activity row, the title air row, and the editor
	/// block.
	const fn band_height(editor_height: u16) -> u16 {
		LIVE_SHARD_ROWS + 2 + 3 + editor_height
	}

	/// Rows `entry` will occupy at `width`, including its trailing blank.
	fn entry_height(entry: &Entry, width: u16, ctx: &UiContext) -> u16 {
		match entry {
			Entry::Command => 5,
			Entry::Message(message) => {
				let mut scratch = Frame::new(Size::new(width, 48));
				Self::draw_message(&mut scratch, 0, *message, width, ctx.charset, ctx.native_decor)
			},
			Entry::ShardDone(_) => 1,
			Entry::Submitted(submission) => submission.height().saturating_add(1),
		}
	}

	const fn message_width(width: u16) -> u16 {
		let narrowed = width.saturating_sub(3);
		if narrowed == 0 { 1 } else { narrowed }
	}

	/// Paints one transcript entry at `y` and returns the rows it used.
	fn draw_entry_at(&mut self, index: usize, y: u16, width: u16) -> u16 {
		Self::draw_entry(&mut self.frame, &self.transcript[index], y, width, &self.ctx)
	}

	/// Paints `entry` into any frame at `y` and returns the rows it used,
	/// including the trailing blank.
	fn draw_entry(frame: &mut Frame, entry: &Entry, y: u16, width: u16, ctx: &UiContext) -> u16 {
		let margin = u16::from(width >= 50);
		let content_width = width.saturating_sub(margin * 2);
		match entry {
			Entry::Command => {
				draw_command_box(
					frame,
					Rect::new(margin, y, content_width, 4),
					ctx.charset,
					ctx.native_decor,
				);
				5
			},
			Entry::Message(message) => {
				Self::draw_message(frame, y, *message, width, ctx.charset, ctx.native_decor)
			},
			Entry::ShardDone(shard) => {
				Self::draw_shard_done(frame, y, *shard, width, ctx.charset);
				1
			},
			Entry::Submitted(submission) => {
				draw_submission(frame, y, submission, ctx.charset);
				submission.height().saturating_add(1)
			},
		}
	}

	/// Composes exactly one viewport of throwaway resize-drag content at the
	/// new geometry: the live band anchors to the bottom, then transcript
	/// entries are walked backward and rewrapped at `viewport.width` until
	/// the screen is full — O(viewport) work per drag frame, with the
	/// topmost entry sliced when it only partially fits. Retained transcript
	/// state is untouched, so the settle rebuild replays full history
	/// exactly once.
	pub fn render_resize_preview(&mut self, viewport: Size) -> Frame {
		let elapsed = self.started_at.elapsed();
		let mut frame = Frame::new(viewport);
		if viewport.width == 0 || viewport.height == 0 {
			return frame;
		}
		frame.fill(Rect::new(0, 0, viewport.width, viewport.height), base_style());
		let composer_width = self.composer_width(viewport);
		if self.editor_ui.frame().size().width != composer_width {
			self.editor_ui.resize(composer_width);
		}
		self.editor_ui.tick(elapsed);

		// The live band, laid out exactly like the retained document's.
		let margin = u16::from(viewport.width >= 50);
		let content_width = viewport.width.saturating_sub(margin * 2);
		let editor_height = self.editor_ui.height();
		let editor_y = viewport.height.saturating_sub(editor_height);
		let title_y = editor_y.saturating_sub(1);
		let working_y = title_y.saturating_sub(1);
		let panel_height = LIVE_SHARD_ROWS + 2;
		let panel_y = working_y.saturating_sub(1).saturating_sub(panel_height);
		draw_live_panel(
			&mut frame,
			&mut self.live_rows,
			&mut self.live_label_scratch,
			Rect::new(margin, panel_y, content_width, panel_height),
			true,
			self.emitted_shards,
			Self::animation_frame(elapsed),
			self.ctx.charset,
			self.ctx.native_decor,
		);
		if self.work.borrow().working {
			Self::draw_working(
				&mut frame,
				working_y,
				elapsed,
				self.cancel_hint,
				self.ctx.native_decor,
			);
		}
		Self::draw_session_title(&mut frame, title_y, self.right_inset);
		frame.blit(self.editor_ui.frame(), 0, editor_height, 0, editor_y);

		// Transcript tail, bottom-up above the band.
		let mut remaining = panel_y;
		for entry in self.transcript.iter().rev() {
			if remaining == 0 {
				break;
			}
			let height = Self::entry_height(entry, viewport.width, &self.ctx);
			if height == 0 {
				continue;
			}
			if height <= remaining {
				remaining -= height;
				Self::draw_entry(&mut frame, entry, remaining, viewport.width, &self.ctx);
			} else {
				// Slice the bottom rows of the partially visible entry.
				let mut scratch = Frame::new(Size::new(viewport.width, height));
				scratch.fill(Rect::new(0, 0, viewport.width, height), base_style());
				Self::draw_entry(&mut scratch, entry, 0, viewport.width, &self.ctx);
				frame.blit(&scratch, height - remaining, remaining, 0, 0);
				remaining = 0;
			}
		}
		frame
	}

	/// Paints the n-th scripted message and returns rows used including
	/// the trailing blank. Measurement draws into a scratch frame.
	fn draw_message(
		frame: &mut Frame,
		y: u16,
		message: usize,
		width: u16,
		charset: Charset,
		native: bool,
	) -> u16 {
		let margin = u16::from(width >= 50);
		let content_width = width.saturating_sub(margin * 2);
		if message == 2 {
			draw_edit_box(
				frame,
				Rect::new(margin, y, content_width, EDIT_BOX_HEIGHT),
				charset,
				native,
			);
			return EDIT_BOX_HEIGHT + 1;
		}
		let bottom = frame.size().height;
		let spans = Self::message_spans(message);
		// Prose flows edge-to-edge grapheme-exact — no side pads — so every
		// wrapped row re-joins byte-for-byte in native selection.
		let used = draw_flowed(frame, Rect::new(0, y, width, bottom.saturating_sub(y)), &spans);
		used.saturating_add(1)
	}

	fn message_spans(message: usize) -> SmallVec<Span<'static>, 3> {
		let mut spans = SmallVec::new();
		match message {
			0 => {
				spans.push(Span::new("Transcript rows are ", prose_style()));
				spans.push(Span::new("append-only", code_style()));
				spans.push(Span::new(
					": every line is painted once, becomes stable, and rides into native scrollback \
					 with any selection anchored to it.",
					prose_style(),
				));
			},
			1 => {
				spans.push(Span::new(
					"Only the bottom band repaints in place — the live shard panel, the activity \
					 shimmer, and the composer. Rows above it are never rewritten.",
					prose_style(),
				));
			},
			_ => {
				spans.push(Span::new(
					"On terminals that move margin-scrolled rows into scrollback, commits scroll only \
					 the stable transcript through a ",
					prose_style(),
				));
				spans.push(Span::new("DECSTBM top region", code_style()));
				spans.push(Span::new(", so the live band never shifts on screen.", prose_style()));
			},
		}
		spans
	}

	/// Appends a finished shard's permanent one-line result.
	fn draw_shard_done(frame: &mut Frame, y: u16, shard: u16, width: u16, charset: Charset) {
		let margin = u16::from(width >= 50);
		let prefix = fmts!(" {} shard {shard:03} passed", charset.check());
		let detail = fmts!("  workspace-{shard:03}.test.ts  [100%]");
		draw_line(frame, margin + 1, y, width.saturating_sub(margin * 2).saturating_sub(2), &[
			Span::new(prefix.as_str(), ink(GREEN)),
			Span::new(detail.as_str(), ink(MUTED)),
		]);
	}

	/// Shimmering activity line above the editor. The spinner and timer
	/// live in the status bar's brand segment; this row only narrates.
	fn draw_working(frame: &mut Frame, y: u16, elapsed: Duration, hint: &str, native: bool) {
		if y >= frame.size().height || frame.size().width < 4 {
			return;
		}
		let start = u16::from(frame.size().width >= 50);
		let mut column = start;
		let length = xutf::graphemes_str(WORKING_MESSAGE)
			.count()
			.saturating_add(xutf::graphemes_str(hint).count())
			.saturating_add(1);
		let length = u16::try_from(length).unwrap_or(u16::MAX);
		let shimmer = Shimmer::new(elapsed, SHIMMER_PERIOD, length);
		let right = frame.size().width.saturating_sub(1);
		if native {
			frame.fill(Rect::new(start, y, right.saturating_sub(start), 1), base_style());
		}
		draw_shimmer(frame, &mut column, start, y, right, hint, shimmer, ink(CYAN), native);
		draw_shimmer(frame, &mut column, start, y, right, " ", shimmer, ink(GREEN), native);
		draw_shimmer(
			frame,
			&mut column,
			start,
			y,
			right,
			WORKING_MESSAGE,
			shimmer,
			ink(GREEN),
			native,
		);
		if native {
			frame.push_decor(Decor {
				rect: Rect::new(start, y, column.saturating_sub(start), 1),
				kind: DecorKind::Shimmer { period: SHIMMER_PERIOD },
			});
		}
	}

	/// The session title resting right-aligned in the air row between
	/// the working narration and the status band — against the visible
	/// right bound, inside any rail reservation — so the gap reads as
	/// session identity instead of dead space.
	fn draw_session_title(frame: &mut Frame, y: u16, right_inset: u16) {
		let width = frame.size().width.saturating_sub(right_inset);
		let title_width = visible_width(SESSION_TITLE);
		if y >= frame.size().height || width < title_width.saturating_add(2) {
			return;
		}
		let x = width.saturating_sub(title_width.saturating_add(1));
		draw_line(frame, x, y, title_width, &[Span::new(SESSION_TITLE, ink(FAINT).italic())]);
	}
}

/// The closed four-row command box that opens the transcript.
fn draw_command_box(frame: &mut Frame, rect: Rect, charset: Charset, native: bool) {
	draw_box(frame, rect, ink(FAINT), panel_style(), charset, native);
	if rect.width < 4 || rect.height < 4 {
		return;
	}

	let content_x = rect.x + 2;
	let content_width = rect.width.saturating_sub(4);
	let header = [
		Span::new(" PARALLEL TEST RUN ", panel_ink(GREEN).bold()),
		Span::new("results append below · live rows in the bottom panel", panel_ink(MUTED)),
	];
	draw_line(frame, content_x, rect.y + 1, content_width, &header);
	let command = [
		Span::new("$ ", panel_ink(MUTED)),
		Span::new("bun test --parallel=8", panel_ink(CYAN)),
		Span::new(" --timeout=30000 --all-workspaces", panel_ink(TEXT)),
	];
	draw_line(frame, content_x, rect.y + 2, content_width, &command);
}

/// The live band's shard panel: twelve mutable rows that repaint in place
/// every frame and never enter native scrollback.
fn draw_live_panel(
	frame: &mut Frame,
	rows: &mut [LiveRowCache; LIVE_SHARD_ROWS as usize],
	label_scratch: &mut StrMut,
	rect: Rect,
	repaint_chrome: bool,
	emitted_shards: u16,
	animation_frame: u64,
	charset: Charset,
	native: bool,
) -> bool {
	let mut changed = repaint_chrome;
	if repaint_chrome {
		draw_box(frame, rect, ink(FAINT), panel_style(), charset, native);
	}
	if rect.width < 4 || rect.height < 3 {
		return changed;
	}

	if repaint_chrome {
		let title = [
			Span::new(" LIVE SHARDS ", panel_ink(GREEN).bold()),
			Span::new("mutable rows repaint in place ", panel_ink(MUTED)),
		];
		draw_line(frame, rect.x + 2, rect.y, rect.width.saturating_sub(4), &title);
	}
	let content_x = rect.x + 2;
	let content_width = rect.width.saturating_sub(4);
	for row in 0..rect.height.saturating_sub(2) {
		let shard = emitted_shards.saturating_add(row).saturating_add(1);
		let phase = (u64::from(row) + animation_frame) % 11;
		let (prefix_phase, symbol, state, state_style, progress) = match phase {
			0 => (
				0,
				"⠼",
				"running",
				panel_ink(GREEN).bold(),
				(u64::from(row) * 17 + animation_frame * 7) % 100,
			),
			1..=7 => {
				(1, "·", "working", panel_ink(MUTED), (u64::from(row) * 17 + animation_frame * 7) % 100)
			},
			_ => (2, "·", "queued ", panel_ink(FAINT), 0),
		};
		let row_y = rect.y + 1 + row;
		let right = content_x
			.saturating_add(content_width)
			.min(frame.size().width);
		let cache = &mut rows[usize::from(row)];
		let prefix_changed = repaint_chrome
			|| !cache.prefix_valid
			|| cache.prefix_shard != shard
			|| cache.prefix_phase != prefix_phase;
		changed |= prefix_changed;
		let label_x = if prefix_changed {
			let prefix = fmts!(" {symbol} shard {shard:03} ");
			let prefix_width = prefix.len().saturating_sub(symbol.len()).saturating_add(1);
			let next_x = content_x
				.saturating_add(u16::try_from(prefix_width).unwrap_or(u16::MAX))
				.saturating_add(u16::try_from(state.len()).unwrap_or(u16::MAX))
				.saturating_add(2)
				.min(right);
			if !repaint_chrome && cache.label_x != next_x {
				clear_cached_label(frame, cache, row_y, right);
			}
			let next_x = draw_line(frame, content_x, row_y, content_width, &[
				Span::new(prefix.as_str(), state_style),
				Span::new(state, state_style),
				Span::new("  ", panel_ink(FAINT)),
			]);
			cache.prefix_shard = shard;
			cache.prefix_phase = prefix_phase;
			cache.prefix_valid = true;
			next_x
		} else {
			cache.label_x
		};
		let moved = cache.label_x != label_x;
		let label_changed = repaint_chrome
			|| moved
			|| !cache.label_valid
			|| cache.label_shard != shard
			|| cache.label_progress != progress;
		changed |= label_changed;
		if label_changed {
			label_scratch.truncate(0);
			write!(label_scratch, "workspace-{shard:03}.test.ts  [{progress:>3}%]")
				.expect("shard label formatting is infallible");
			let resized = cache.label.len() != label_scratch.len();
			if !repaint_chrome && resized && !moved {
				clear_cached_label(frame, cache, row_y, right);
			}
			let width = right.saturating_sub(label_x);
			if repaint_chrome || moved || resized {
				frame.put_clipped(label_x, row_y, width, label_scratch.as_str(), panel_ink(MUTED));
			} else {
				draw_ascii_changes(
					frame,
					label_x,
					row_y,
					width,
					cache.label.as_str(),
					label_scratch.as_str(),
					panel_ink(MUTED),
				);
			}
			std::mem::swap(&mut cache.label, label_scratch);
			cache.label_shard = shard;
			cache.label_progress = progress;
			cache.label_valid = true;
		}
		cache.label_x = label_x;
	}
	changed
}

fn clear_cached_label(frame: &mut Frame, cache: &LiveRowCache, y: u16, right: u16) {
	if cache.label.is_empty() {
		return;
	}
	let width = u16::try_from(cache.label.len())
		.unwrap_or(u16::MAX)
		.min(right.saturating_sub(cache.label_x));
	frame.fill(Rect::new(cache.label_x, y, width, 1), panel_style());
}

/// Repaints only changed byte runs within an equal-length ASCII label.
fn draw_ascii_changes(
	frame: &mut Frame,
	x: u16,
	y: u16,
	width: u16,
	previous: &str,
	next: &str,
	style: Style,
) {
	if width == 0 || previous == next {
		return;
	}
	if previous.len() != next.len() || !previous.is_ascii() || !next.is_ascii() {
		frame.put_clipped(x, y, width, next, style);
		return;
	}
	let previous = previous.as_bytes();
	let next_bytes = next.as_bytes();
	let limit = previous.len().min(usize::from(width));
	let mut index = 0;
	while index < limit {
		while index < limit && previous[index] == next_bytes[index] {
			index += 1;
		}
		let start = index;
		while index < limit && previous[index] != next_bytes[index] {
			index += 1;
		}
		if start < index {
			let offset = u16::try_from(start).unwrap_or(u16::MAX);
			frame.put_clipped(
				x.saturating_add(offset),
				y,
				u16::try_from(index - start).unwrap_or(u16::MAX),
				&next[start..index],
				style,
			);
		}
	}
}

fn draw_edit_box(frame: &mut Frame, rect: Rect, charset: Charset, native: bool) {
	draw_box(frame, rect, ink(FAINT), panel_style(), charset, native);
	if rect.width < 8 || rect.height < EDIT_BOX_HEIGHT {
		return;
	}

	let title = [
		Span::new(" Live ", panel_ink(GREEN).bold()),
		Span::new("band · selection semantics ", panel_ink(CYAN)),
	];
	draw_line(frame, rect.x + 2, rect.y, rect.width.saturating_sub(4), &title);
	draw_line(frame, rect.x + 2, rect.y + 1, rect.width.saturating_sub(4), &[
		Span::new(charset.check(), panel_ink(GREEN).bold()),
		Span::new(" ", panel_ink(GREEN)),
		Span::new("Transcript selections ride with the text", panel_ink(TEXT)),
	]);
	draw_line(frame, rect.x + 2, rect.y + 2, rect.width.saturating_sub(4), &[Span::new(
		"  margin commits pin the band on kitty-class terminals",
		panel_ink(MUTED),
	)]);
}

/// Paints a submitted message: the prompt gutter, then the rendered
/// Markdown document blitted beside it (or the raw lines when the text
/// isn't renderable as Markdown).
fn draw_submission(frame: &mut Frame, y: u16, submission: &Submission, charset: Charset) {
	if frame.size().width < 4 {
		return;
	}
	let Some(view) = &submission.view else {
		for (offset, line) in submission.text.split('\n').enumerate() {
			let Ok(offset) = u16::try_from(offset) else {
				break;
			};
			let row = y.saturating_add(offset);
			if row >= frame.size().height {
				break;
			}
			let prompt = if offset == 0 { charset.cursor() } else { "  " };
			let text_x = frame.put(1, row, prompt, ink(GREEN).bold());
			let width = frame
				.size()
				.width
				.saturating_sub(1)
				.saturating_sub(text_x.saturating_sub(1));
			draw_submission_text(frame, text_x, row, width, line, charset);
		}
		return;
	};
	frame.put(1, y, charset.cursor(), ink(GREEN).bold());
	frame.blit(view.frame(), 0, view.height(), 3, y);
}
fn explicit_line_count(text: &str) -> u16 {
	u16::try_from(
		text
			.bytes()
			.filter(|byte| *byte == b'\n')
			.count()
			.saturating_add(1),
	)
	.unwrap_or(u16::MAX)
}

/// Paints a rounded panel box through the tier's border glyphs.
fn draw_box(
	frame: &mut Frame,
	rect: Rect,
	border: Style,
	fill: Style,
	charset: Charset,
	native: bool,
) {
	if rect.width == 0 || rect.height == 0 {
		return;
	}
	if native {
		frame.push_decor(Decor {
			rect,
			kind: DecorKind::Fill {
				fill:    DecorFill::Solid(fill.background_color()),
				rounded: true,
			},
		});
		frame.push_decor(Decor {
			rect,
			kind: DecorKind::Border {
				border: Border::Round,
				ink:    DecorFill::Solid(border.foreground_color()),
				glow:   None,
			},
		});
		return;
	}
	let (tl, tr, _, _, horizontal, vertical) = charset.border(Border::Round);
	frame.fill(rect, fill);
	let mut glyph = [0_u8; 4];
	if rect.width == 1 {
		frame.put(rect.x, rect.y, vertical.encode_utf8(&mut glyph), border);
		return;
	}

	let right = rect.x + rect.width - 1;
	let bottom = rect.y + rect.height - 1;
	frame.put(rect.x, rect.y, tl.encode_utf8(&mut glyph), border);
	frame.put(right, rect.y, tr.encode_utf8(&mut glyph), border);
	for x in rect.x + 1..right {
		frame.put(x, rect.y, horizontal.encode_utf8(&mut glyph), border);
	}

	if rect.height > 1 {
		draw_box_bottom(frame, rect, border, charset, native);
	}
	for row in rect.y + 1..bottom {
		frame.put(rect.x, row, vertical.encode_utf8(&mut glyph), border);
		frame.put(right, row, vertical.encode_utf8(&mut glyph), border);
	}
}

fn draw_box_bottom(frame: &mut Frame, rect: Rect, border: Style, charset: Charset, native: bool) {
	if rect.width < 2 || rect.height < 2 {
		return;
	}
	if native {
		frame.push_decor(Decor {
			rect,
			kind: DecorKind::Fill { fill: DecorFill::Solid(PANEL), rounded: true },
		});
		frame.push_decor(Decor {
			rect,
			kind: DecorKind::Border {
				border: Border::Round,
				ink:    DecorFill::Solid(border.foreground_color()),
				glow:   None,
			},
		});
		return;
	}
	let (_, _, bl, br, horizontal, _) = charset.border(Border::Round);
	let mut glyph = [0_u8; 4];
	let right = rect.x + rect.width - 1;
	let bottom = rect.y + rect.height - 1;
	frame.put(rect.x, bottom, bl.encode_utf8(&mut glyph), border);
	frame.put(right, bottom, br.encode_utf8(&mut glyph), border);
	for x in rect.x + 1..right {
		frame.put(x, bottom, horizontal.encode_utf8(&mut glyph), border);
	}
}

fn draw_line(frame: &mut Frame, x: u16, y: u16, width: u16, spans: &[Span<'_>]) -> u16 {
	let right = x.saturating_add(width).min(frame.size().width);
	let mut column = x;
	for span in spans {
		column = frame.put_clipped(column, y, right.saturating_sub(column), span.text, span.style);
		if column >= right {
			break;
		}
	}
	column
}

/// Flows `spans` grapheme-exact across the rect like a bare terminal,
/// preserving all whitespace and flagging each exactly-filled row boundary
/// soft so native selection copies the paragraph as one unbroken line.
/// Returns the rows used.
fn draw_flowed(frame: &mut Frame, rect: Rect, spans: &[Span<'_>]) -> u16 {
	if rect.width == 0 || rect.height == 0 {
		return 0;
	}
	let full_row = rect.x == 0 && rect.width == frame.size().width;
	let mut row = 0_u16;
	let mut column = 0_u16;
	let mut drew_anything = false;

	for span in spans {
		for grapheme in xutf::graphemes_str(span.text) {
			let grapheme_width = visible_width(grapheme);
			if grapheme_width == 0 || grapheme_width > rect.width {
				continue;
			}
			if column.saturating_add(grapheme_width) > rect.width {
				// Only an exactly-filled row is byte-joinable by autowrap.
				if full_row && column == rect.width {
					frame.set_soft_wrap(rect.y.saturating_add(row));
				}
				row += 1;
				column = 0;
			}
			if row >= rect.height {
				return rect.height;
			}
			frame.put(rect.x + column, rect.y + row, grapheme, span.style);
			column += grapheme_width;
			drew_anything = true;
		}
	}

	if drew_anything { row + 1 } else { 0 }
}

fn visible_width(text: &str) -> u16 {
	u16::try_from(xutf::width_str(text)).unwrap_or(u16::MAX)
}

/// Finds the first `<ref image=N/>` tag in submitted text: its byte range
/// plus the marker number `N`.
fn next_ref_tag(text: &str) -> Option<(usize, usize, usize)> {
	const HEAD: &str = "<ref image=";
	let mut from = 0;
	while let Some(at) = text[from..].find(HEAD) {
		let start = from + at;
		let body = &text[start + HEAD.len()..];
		let digits = body.bytes().take_while(u8::is_ascii_digit).count();
		if digits > 0 && body[digits..].starts_with("/>") {
			let marker = body[..digits].parse().unwrap_or(usize::MAX);
			return Some((start, start + HEAD.len() + digits + 2, marker));
		}
		from = start + HEAD.len();
	}
	None
}

/// Splits one composer input row into base-styled text and attachment
/// chips, chip styling winning over any overlapping XML run.
///
/// Chips are located through the buffer's atomic ranges — like the XML
/// pass, styling happens at paint time, and typed lookalike text is never
/// recolored.
fn push_row_spans<'a>(
	editor: &Editor,
	row: &'a str,
	runs: &[SyntaxRun],
	selection: Option<(u16, u16)>,
	selection_bg: Color,
	spans: &mut SmallVec<Span<'a>, 16>,
) {
	let text = editor.text();
	let buffer_start = text.as_ptr() as usize;
	let row_start = (row.as_ptr() as usize).saturating_sub(buffer_start);
	let row_end = row_start + row.len();
	// Chip segments clipped to this row; style derives from the FULL atom
	// text, so a chip wrapped across rows keeps its color on every row.
	let mut chips: SmallVec<(usize, usize, Style), 4> = SmallVec::new();
	for (start, end) in editor.atom_ranges() {
		let from = start.max(row_start);
		let to = end.min(row_end);
		if from < to {
			chips.push((from - row_start, to - row_start, chip_style(&text[start..end])));
		}
	}
	// The style and extent of the base segment covering `at`: its XML run,
	// or plain text up to the next run.
	let base = |at: usize| {
		runs
			.iter()
			.find(|run| run.start <= at && at < run.end)
			.map_or_else(
				|| {
					let next = runs
						.iter()
						.map(|run| run.start)
						.filter(|start| *start > at)
						.min()
						.unwrap_or(row.len());
					(next, base_style())
				},
				|run| (run.end, run.style),
			)
	};
	fn emit<'a>(
		row: &'a str,
		base: &impl Fn(usize) -> (usize, Style),
		from: usize,
		to: usize,
		spans: &mut SmallVec<Span<'a>, 16>,
	) {
		let mut at = from;
		while at < to {
			let (run_end, style) = base(at);
			let end = run_end.min(to);
			spans.push(Span::new(&row[at..end], style));
			at = end;
		}
	}
	let mut at = 0;
	for (start, end, style) in chips {
		emit(row, &base, at, start, spans);
		spans.push(Span::new(&row[start..end], style));
		at = end;
	}
	emit(row, &base, at, row.len(), spans);
	if let Some((start, end)) = selection {
		restyle_selection(row, start, end, selection_bg, spans);
	}
}

/// Layers a background onto the selected display columns without disturbing
/// syntax foregrounds or attachment-chip emphasis.
fn restyle_selection<'a>(
	row: &'a str,
	start_column: u16,
	end_column: u16,
	background: Color,
	spans: &mut SmallVec<Span<'a>, 16>,
) {
	if start_column >= end_column {
		return;
	}
	let start = byte_at_display_column(row, start_column);
	let end = byte_at_display_column(row, end_column);
	if start >= end {
		return;
	}
	let source = std::mem::take(spans);
	let mut at = 0;
	for span in source {
		let span_end = at + span.text.len();
		let selected_start = start.clamp(at, span_end);
		let selected_end = end.clamp(at, span_end);
		if at < selected_start {
			spans.push(Span::new(&span.text[..selected_start - at], span.style));
		}
		if selected_start < selected_end {
			spans.push(Span::new(
				&span.text[selected_start - at..selected_end - at],
				span.style.bg(background),
			));
		}
		if selected_end < span_end {
			spans.push(Span::new(&span.text[selected_end - at..], span.style));
		}
		at = span_end;
	}
}

fn byte_at_display_column(text: &str, column: u16) -> usize {
	let mut byte = 0;
	let mut width: u16 = 0;
	for grapheme in text.graphemes() {
		let next = width.saturating_add(visible_width(grapheme));
		if next > column {
			break;
		}
		byte += grapheme.len();
		width = next;
	}
	byte
}

/// Style for one atomic chip: a trailing `#N` selects the marker's
/// identity color; other atoms stay plain.
fn chip_style(chip: &str) -> Style {
	let Some(hash) = chip.rfind('#') else {
		return base_style();
	};
	let digits = &chip[hash + 1..];
	if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
		return base_style();
	}
	match digits.parse::<usize>() {
		Ok(marker) if marker > 0 => ink(attachment_color(marker)).bold(),
		_ => base_style(),
	}
}

/// Paints one transcript line, rendering each `<ref image=N/>` tag as a
/// compact `<icon> #N` pill filled with the attachment's identity color.
fn draw_submission_text(
	frame: &mut Frame,
	x: u16,
	y: u16,
	width: u16,
	line: &str,
	charset: Charset,
) {
	let icon = charset.icon(Icon::Image);
	let mut chips: SmallVec<(usize, usize, String, usize), 4> = SmallVec::new();
	let mut base = 0;
	while let Some((start, end, marker)) = next_ref_tag(&line[base..]) {
		chips.push((base + start, base + end, format!("{icon} #{marker}"), marker));
		base += end;
	}
	let mut spans: SmallVec<Span<'_>, 8> = SmallVec::new();
	let mut at = 0;
	for (start, end, label, marker) in &chips {
		if *start > at {
			spans.push(Span::new(&line[at..*start], ink(TEXT)));
		}
		spans.push(Span::new(label, Style::new().fg(PANEL).bg(attachment_color(*marker)).bold()));
		at = *end;
	}
	if at < line.len() {
		spans.push(Span::new(&line[at..], ink(TEXT)));
	}
	draw_line(frame, x, y, width, &spans);
}

/// Chrome outside a panel is transparent: no `bg`, so the terminal's own
/// background (and any image or blur behind it) shows through. Only the
/// panel boxes below opt into a fill.
const fn base_style() -> Style {
	Style::new().fg(TEXT)
}

const fn panel_style() -> Style {
	Style::new().fg(TEXT).bg(PANEL)
}

const fn ink(color: Color) -> Style {
	Style::new().fg(color)
}

const fn panel_ink(color: Color) -> Style {
	Style::new().fg(color).bg(PANEL)
}

const fn prose_style() -> Style {
	Style::new().fg(MUTED).italic()
}

const fn code_style() -> Style {
	Style::new().fg(GREEN)
}

#[cfg(test)]
mod tests {
	use std::{hint::black_box, str, time::Instant};

	use omp_tui::{
		CellContent, Key, Mods, Mouse, MouseButton, MouseReport, Renderer,
		test_support::{TerminalModel, frame_row_text},
	};

	use super::{
		BRAND_FADE, CANCEL_HINT, Charset, Color, DecorFill, DecorKind, Demo, DemoInput, DemoKey,
		Duration, FAINT, Frame, GREEN, INPUT_PROMPT, MUTED, PANEL, Rect, RenderedFrame,
		SHIMMER_PERIOD, Shimmer, Size, Style, UiContext, WORKING_MESSAGE, draw_box, elapsed_label,
		ink, panel_style,
	};

	/// The nerd-tier context every rendering assertion in this module
	/// expects (glyph fixtures are authored for it).
	fn test_ctx() -> UiContext {
		UiContext { charset: Charset::NerdFont, ..UiContext::default() }
	}
	#[test]
	fn native_box_uses_decor_without_inking_cells() {
		let mut frame = Frame::new(Size::new(8, 4));
		let rect = Rect::new(1, 1, 6, 2);

		draw_box(&mut frame, rect, ink(FAINT), panel_style(), Charset::NerdFont, true);

		for y in rect.y..rect.y + rect.height {
			for x in rect.x..rect.x + rect.width {
				let cell = frame.cell(x, y);
				assert!(matches!(cell.content(), CellContent::Blank));
				assert_eq!(cell.style(), Style::default());
			}
		}
		assert!(matches!(
			frame.decors(),
			[
				super::Decor {
					rect: border_rect,
					kind: DecorKind::Fill {
						fill: DecorFill::Solid(PANEL),
						rounded: true,
					},
				},
				super::Decor {
					rect: ink_rect,
					kind: DecorKind::Border {
						border: super::Border::Round,
						ink: DecorFill::Solid(FAINT),
						glow: None,
					},
				},
			] if *border_rect == rect && *ink_rect == rect
		));
	}

	#[test]
	fn native_working_line_uses_one_plain_shimmer_decor() {
		let mut frame = Frame::new(Size::new(80, 1));

		Demo::draw_working(&mut frame, 0, Duration::from_millis(500), "esc", true);

		assert!(matches!(frame.decors(), [super::Decor {
			kind: DecorKind::Shimmer { period: SHIMMER_PERIOD },
			..
		}]));
		assert_eq!(frame.cell(1, 0).style().foreground_color(), Color::Rgb(62, 190, 203));
	}

	fn present<W: std::io::Write>(
		renderer: &mut Renderer<W>,
		rendered: RenderedFrame<'_>,
		viewport: Size,
	) -> std::io::Result<omp_tui::PaintStats> {
		renderer.present_damaged(
			rendered.frame,
			rendered.damage.as_slice(),
			viewport.height,
			rendered.stable_rows,
		)
	}

	/// Renders one step of an interactive session and asserts the emitted
	/// ANSI leaves the terminal exactly matching the frame's live window.
	fn replay_step(
		demo: &mut Demo,
		renderer: &mut Renderer<Vec<u8>>,
		terminal: &mut TerminalModel,
		viewport: Size,
		elapsed_ms: u64,
		key: Option<char>,
	) {
		if let Some(character) = key {
			demo.handle_key(Key::Char(character));
		}
		let rendered = demo.render_at(viewport, Duration::from_millis(elapsed_ms));
		let height = rendered.frame.size().height;
		let rows: Vec<String> = (0..height)
			.map(|row| frame_row_text(rendered.frame, row))
			.collect();
		present(renderer, rendered, viewport).expect("replay paint succeeds");
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).expect("ANSI is UTF-8");
		terminal.apply(&output);
		let window_top = renderer
			.committed_rows()
			.max(height.saturating_sub(viewport.height));
		let expected: Vec<String> = (0..viewport.height)
			.map(|row| rows[usize::from(window_top.saturating_add(row))].clone())
			.collect();
		let actual = terminal.visible_rows();
		for (row, (have, want)) in actual.iter().zip(&expected).enumerate() {
			assert_eq!(
				have, want,
				"terminal row {row} diverged from the frame at {elapsed_ms}ms after key {key:?}"
			);
		}
	}

	#[test]
	fn interactive_picker_session_replays_cell_for_cell() {
		let viewport = Size::new(157, 46);
		let mut demo = Demo::new(&test_ctx());
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(157, 46);

		let script: &[(u64, Option<char>)] = &[
			(0, None),
			(700, None),
			(2_600, Some(':')),
			(2_650, Some('e')),
			(3_500, None),
			(4_000, Some('m')),
			(4_400, Some('q')),
			(4_700, None),
			(5_400, None),
			(6_100, None),
			(6_800, None),
			(7_500, None),
			(8_200, None),
		];
		for &(elapsed_ms, key) in script {
			replay_step(&mut demo, &mut renderer, &mut terminal, viewport, elapsed_ms, key);
		}
	}

	/// The same class of interactive session committed through DECSTBM
	/// margins: pinned band rows and native history must replay
	/// cell-for-cell on the margin-scrollback terminal model.
	#[test]
	fn interactive_session_replays_cell_for_cell_with_margin_commits() {
		let viewport = Size::new(157, 46);
		let mut demo = Demo::new(&test_ctx());
		let mut renderer = Renderer::new(Vec::new());
		renderer.set_margin_scrollback(true);
		let mut terminal = TerminalModel::new(157, 46);

		let script: &[(u64, Option<char>)] = &[
			(0, None),
			(700, None),
			(2_600, Some(':')),
			(2_650, Some('e')),
			(3_500, None),
			(4_400, Some('q')),
			(5_400, None),
			(6_100, None),
			(6_800, None),
			(7_500, None),
			(8_200, None),
		];
		for &(elapsed_ms, key) in script {
			replay_step(&mut demo, &mut renderer, &mut terminal, viewport, elapsed_ms, key);
		}
	}
	fn has_csi_command(output: &str, command: u8) -> bool {
		let bytes = output.as_bytes();
		let mut index = 0;
		while index + 1 < bytes.len() {
			if bytes[index] != 0x1b || bytes[index + 1] != b'[' {
				index += 1;
				continue;
			}
			index += 2;
			while let Some(byte) = bytes.get(index) {
				if (0x40..=0x7e).contains(byte) {
					if *byte == command {
						return true;
					}
					break;
				}
				index += 1;
			}
			index += 1;
		}
		false
	}

	#[test]
	fn classic_shimmer_moves_a_bright_crest_across_the_message() {
		let low = ink(FAINT);
		let high = ink(GREEN).bold();
		let period = Duration::from_secs(1);
		// One second sweeps the 50-cell padded track, so 200ms puts the
		// crest ten cells in — exactly over the first glyph.
		let before_band = Shimmer::new(Duration::ZERO, period, 30);
		let at_crest = Shimmer::new(Duration::from_millis(200), period, 30);

		assert_eq!(before_band.pick(0, low, ink(MUTED), high), low);
		assert_eq!(at_crest.pick(0, low, ink(MUTED), high), high);
	}

	#[test]
	fn elapsed_label_stays_compact_across_units() {
		assert_eq!(elapsed_label(Duration::from_secs(20)), "20s");
		assert_eq!(elapsed_label(Duration::from_secs(60)), "1m");
		assert_eq!(elapsed_label(Duration::from_mins(15)), "15m");
		assert_eq!(elapsed_label(Duration::from_hours(1)), "1h");
		assert_eq!(elapsed_label(Duration::from_hours(10)), "10h");
		assert_eq!(elapsed_label(Duration::from_hours(100)), "99h");
	}

	#[test]
	fn status_brand_swaps_spinner_for_omp_across_work_states() {
		let viewport = Size::new(120, 32);
		let mut demo = Demo::new(&test_ctx());
		let rows_at = |demo: &mut Demo, elapsed| {
			let rendered = demo.render_at(viewport, elapsed);
			(0..rendered.frame.size().height)
				.map(|row| frame_row_text(rendered.frame, row))
				.collect::<Vec<_>>()
		};
		fn status_of(rows: &[String]) -> &str {
			rows
				.iter()
				.find(|row| row.contains("Fable 5++"))
				.expect("status row must be present")
		}

		let rows = rows_at(&mut demo, Duration::ZERO);
		assert!(status_of(&rows).starts_with("\u{e0b6} ⠋ 0s"), "{}", status_of(&rows));
		let activity = rows
			.iter()
			.find(|row| row.contains(WORKING_MESSAGE.trim()))
			.expect("activity row narrates while working");
		assert!(
			activity.trim_start().starts_with(CANCEL_HINT),
			"the cancel hint leads the activity row"
		);

		let rows = rows_at(&mut demo, Duration::from_millis(80));
		assert!(status_of(&rows).starts_with("\u{e0b6} ⠙ 0s"), "the ticked spinner advances");
		let rows = rows_at(&mut demo, Duration::from_secs(65));
		assert!(status_of(&rows).contains(" 1m"), "the session timer stays compact");

		demo.set_working(false, Duration::from_secs(66));
		let rows = rows_at(&mut demo, Duration::from_secs(67));
		assert!(status_of(&rows).starts_with("\u{e0b6} 󰵗 omp"), "{}", status_of(&rows));
		assert!(
			rows.iter().all(|row| !row.contains(WORKING_MESSAGE.trim())),
			"the activity row clears at rest"
		);

		demo.set_working(true, Duration::from_secs(70));
		let rows = rows_at(&mut demo, Duration::from_secs(75));
		assert!(status_of(&rows).contains(" 5s"), "resuming restarts the session timer");
	}

	#[test]
	fn esc_rests_work_then_quits_and_submit_resumes() {
		let mut demo = Demo::new(&test_ctx());
		let esc = Key::Esc;

		assert_eq!(
			demo.handle_key(esc),
			DemoKey::Consumed,
			"the first esc only cancels the running work"
		);
		assert!(!demo.work.borrow().working);

		for character in "go".chars() {
			demo.handle_key(Key::Char(character));
		}
		assert_eq!(demo.handle_key(Key::Enter), DemoKey::Consumed);
		assert!(demo.work.borrow().working, "submitting a message resumes work");

		assert_eq!(demo.handle_key(esc), DemoKey::Consumed, "esc cancels the resumed work");
		assert_eq!(demo.handle_key(esc), DemoKey::Quit, "esc at rest quits");

		let mut fresh = Demo::new(&test_ctx());
		assert_eq!(
			fresh.handle_key(Key::Ctrl('c')),
			DemoKey::Quit,
			"ctrl-c quits even while working"
		);
	}

	fn mouse_report(kind: Mouse, col: u16, row: u16, button: MouseButton) -> MouseReport {
		MouseReport { kind, col, row, button, mods: Mods::default(), pressed: true }
	}

	#[test]
	fn editor_click_translates_document_row_and_moves_the_caret() {
		let viewport = Size::new(80, 24);
		let mut demo = Demo::new(&test_ctx());
		for character in "abcdef".chars() {
			demo.handle_key(Key::Char(character));
		}
		let document_height = demo.render_at(viewport, Duration::ZERO).frame.size().height;
		let editor_y = document_height.saturating_sub(demo.editor_ui.height());

		demo.handle_mouse(&mouse_report(
			Mouse::Click,
			DemoInput::input_offset() + 2,
			editor_y + 1,
			MouseButton::Left,
		));

		let editor = demo.editor.borrow();
		assert_eq!(editor.text(), "abcdef");
		assert_eq!(editor.view(DemoInput::input_width(viewport.width))[0].cursor_column, Some(2));
	}

	#[test]
	fn editor_drag_and_double_click_select_and_paint() {
		let viewport = Size::new(80, 24);
		let ctx = test_ctx();
		let mut demo = Demo::new(&ctx);
		for character in "alpha beta".chars() {
			demo.handle_key(Key::Char(character));
		}
		let document_height = demo.render_at(viewport, Duration::ZERO).frame.size().height;
		let editor_y = document_height.saturating_sub(demo.editor_ui.height());
		let input_x = DemoInput::input_offset();

		demo.handle_mouse(&mouse_report(Mouse::Click, input_x + 1, editor_y + 1, MouseButton::Left));
		demo.handle_mouse(&mouse_report(Mouse::Drag, input_x + 5, editor_y + 1, MouseButton::Left));
		{
			let editor = demo.editor.borrow();
			let rows = editor.view(DemoInput::input_width(viewport.width));
			assert_eq!(editor.selection_span(&rows[0]), Some((1, 5)));
		}
		let frame = demo.render_at(viewport, Duration::ZERO).frame;
		assert_eq!(
			frame
				.cell(input_x + 2, editor_y + 1)
				.style()
				.background_color(),
			ctx.theme.selection
		);

		demo.handle_mouse(&mouse_report(Mouse::Click, input_x + 2, editor_y + 1, MouseButton::Left));
		demo.handle_mouse(&mouse_report(Mouse::Click, input_x + 2, editor_y + 1, MouseButton::Left));
		let editor = demo.editor.borrow();
		let rows = editor.view(DemoInput::input_width(viewport.width));
		assert_eq!(editor.selection_span(&rows[0]), Some((0, 5)));
	}

	#[test]
	fn click_above_editor_block_is_ignored() {
		let viewport = Size::new(80, 24);
		let mut demo = Demo::new(&test_ctx());
		for character in "abcdef".chars() {
			demo.handle_key(Key::Char(character));
		}
		let document_height = demo.render_at(viewport, Duration::ZERO).frame.size().height;
		let editor_y = document_height.saturating_sub(demo.editor_ui.height());
		let before = demo
			.editor
			.borrow()
			.view(DemoInput::input_width(viewport.width))[0]
			.cursor_column;

		demo.handle_mouse(&mouse_report(
			Mouse::Click,
			DemoInput::input_offset(),
			editor_y.saturating_sub(1),
			MouseButton::Left,
		));

		assert_eq!(
			demo
				.editor
				.borrow()
				.view(DemoInput::input_width(viewport.width))[0]
				.cursor_column,
			before
		);
	}

	#[test]
	fn wheel_over_editor_preserves_submitted_transcript() {
		let viewport = Size::new(80, 24);
		let mut demo = Demo::new(&test_ctx());
		for character in "keep this".chars() {
			demo.handle_key(Key::Char(character));
		}
		demo.handle_key(Key::Enter);
		let submitted_row = |demo: &mut Demo| {
			let rendered = demo.render_at(viewport, Duration::ZERO);
			(0..rendered.frame.size().height)
				.find(|&row| frame_row_text(rendered.frame, row).contains("keep this"))
		};
		let before = submitted_row(&mut demo).expect("submission appended to the transcript");
		let document_height = demo.frame.size().height;
		let editor_y = document_height.saturating_sub(demo.editor_ui.height());

		demo.handle_mouse(&mouse_report(
			Mouse::WheelDown,
			DemoInput::input_offset(),
			editor_y + 1,
			MouseButton::WheelDown,
		));

		assert_eq!(
			submitted_row(&mut demo),
			Some(before),
			"the submitted transcript row must survive wheel input over the editor"
		);
	}

	#[test]
	fn brand_fade_is_continuous_across_rapid_cancel_and_resume() {
		let mut demo = Demo::new(&test_ctx());
		let cancel_at = Duration::from_secs(1);
		demo.set_working(false, cancel_at);
		let midway = cancel_at + BRAND_FADE / 2;
		let color = demo.work.borrow().fade.sample(midway);
		assert!(color != GREEN && color != MUTED, "midway the brand sits between its endpoints");

		demo.set_working(true, midway);
		assert_eq!(
			demo.work.borrow().fade.sample(midway),
			color,
			"resuming mid-fade departs from the color already on screen"
		);
	}

	#[test]
	fn growing_tick_commits_new_rows_without_repainting_transcript() {
		let viewport = Size::new(80, 24);
		let mut demo = Demo::new(&test_ctx());
		let mut renderer = Renderer::new(Vec::new());
		let rendered = demo.render_at(viewport, Duration::from_millis(1_500));
		let initial_height = rendered.frame.size().height;
		present(&mut renderer, rendered, viewport).expect("warm demo paint succeeds");
		renderer.writer_mut().clear();

		let rendered = demo.render_at(viewport, Duration::from_millis(2_100));
		assert_eq!(
			rendered.frame.size().height,
			initial_height.saturating_add(2),
			"one emit tick appends two shard result rows"
		);
		let stats = present(&mut renderer, rendered, viewport).expect("growth tick paints");
		let output = String::from_utf8(renderer.into_inner()).expect("renderer output is UTF-8");

		assert_eq!(stats.committed_rows, 2, "appended rows commit into native scrollback");
		assert_eq!(output.matches("\r\n").count(), 2);
		assert!(!has_csi_command(&output, b'H'));
		assert!(
			!output.contains("append-only") && !output.contains("PARALLEL TEST RUN"),
			"stable transcript rows must never be re-emitted"
		);
	}

	#[test]
	fn active_picker_stays_open_while_the_transcript_commits() {
		let viewport = Size::new(120, 32);
		let mut demo = Demo::new(&test_ctx());
		for character in ":joy".chars() {
			assert_eq!(demo.handle_key(Key::Char(character)), DemoKey::Consumed);
		}

		let mut renderer = Renderer::new(Vec::new());
		let rendered = demo.render_at(viewport, Duration::from_millis(100));
		present(&mut renderer, rendered, viewport).expect("initial picker paint succeeds");
		renderer.writer_mut().clear();

		let rendered = demo.render_at(viewport, Duration::from_millis(800));
		let stats =
			present(&mut renderer, rendered, viewport).expect("growing picker frame succeeds");
		let output = String::from_utf8(renderer.into_inner()).expect("renderer output is UTF-8");

		assert_eq!(stats.committed_rows, 3, "the transcript keeps committing under the picker");
		assert!(!has_csi_command(&output, b'H'));
		assert_eq!(output.matches("\r\n").count(), 3);
	}

	#[test]
	fn closing_the_picker_never_shrinks_the_document_mid_stream() {
		// A large viewport keeps `committed == window_top`, so every tick
		// commits rows into native scrollback. Transient picker rows must not
		// strand that ratchet when they close (issue: bottom UI crept down as
		// the transcript regrew through the leftover blank strip).
		let viewport = Size::new(157, 46);
		let mut demo = Demo::new(&test_ctx());
		let mut renderer = Renderer::new(Vec::new());

		let rendered = demo.render_at(viewport, Duration::ZERO);
		present(&mut renderer, rendered, viewport).expect("initial paint succeeds");

		for character in ":e".chars() {
			demo.handle_key(Key::Char(character));
		}
		let rendered = demo.render_at(viewport, Duration::from_millis(100));
		let open_height = rendered.frame.size().height;
		present(&mut renderer, rendered, viewport).expect("open picker paints");

		demo.handle_key(Key::Esc);
		let rendered = demo.render_at(viewport, Duration::from_millis(200));
		assert_eq!(
			rendered.frame.size().height,
			open_height,
			"closing the picker must not shrink the frame"
		);
		present(&mut renderer, rendered, viewport)
			.expect("closed picker paints without violating committed history");

		for elapsed_ms in [700, 1_400, 2_100, 2_800] {
			let rendered = demo.render_at(viewport, Duration::from_millis(elapsed_ms));
			assert!(rendered.frame.size().height >= open_height);
			present(&mut renderer, rendered, viewport)
				.expect("streaming after picker close stays monotonic");
		}
	}

	#[test]
	fn editor_chrome_contains_the_rounded_status_and_soft_prompt_without_a_border() {
		let viewport = Size::new(120, 32);
		let mut demo = Demo::new(&test_ctx());
		// Rest the brand so the chrome shows the omp badge, not the spinner.
		demo.set_working(false, Duration::ZERO);
		let rendered = demo.render_at(viewport, Duration::ZERO);
		let status_y = (0..rendered.frame.size().height)
			.find(|&row| frame_row_text(rendered.frame, row).contains("󰵗 omp"))
			.expect("status row must be present");
		let status_row = frame_row_text(rendered.frame, status_y);
		let input_row = frame_row_text(rendered.frame, status_y.saturating_add(1));
		for row in status_y..rendered.frame.size().height {
			let text = frame_row_text(rendered.frame, row);
			let text = text.strip_prefix(INPUT_PROMPT).unwrap_or(&text);
			assert!(
				!text
					.chars()
					.any(|glyph| matches!(glyph, '╭' | '╮' | '╰' | '╯' | '│' | '─')),
				"unexpected editor border on row {row}: {text}",
			);
		}
		assert_eq!(input_row, INPUT_PROMPT);
		let mut renderer = Renderer::new(Vec::new());
		present(&mut renderer, rendered, viewport).expect("editor frame paints");
		let output = String::from_utf8(renderer.into_inner()).expect("renderer output is UTF-8");

		let model_segment = format!("{} Fable 5++", Charset::NerdFont.icon(super::Icon::Model));
		for segment in
			["󰵗 omp", model_segment.as_str(), " main *5 +9", " 39.1%/1M", "$60.07 (sub) + $8.65 (adv)"]
		{
			assert!(output.contains(segment), "missing status segment: {segment}");
		}
		assert!(status_row.starts_with("\u{e0b6} 󰵗 omp"));
		assert!(
			status_row.contains('\u{e0b2}'),
			"the right group opens with a mirrored cap: {status_row}"
		);
		assert!(
			status_row.ends_with("(adv)"),
			"the right group ends flat against the margin: {status_row}"
		);
		assert!(!status_row.contains('─'), "status row must not retain the editor border");
		assert!(
			output.contains("\r\x1b[3C\x1b[?25h"),
			"focused editor caret must sit beyond the continuation prompt"
		);
		assert!(output.contains("48;2;18;18;18"), "status group must paint its dark background band");
	}

	#[test]
	fn right_docked_chrome_reserves_the_rail_inset() {
		let viewport = Size::new(140, 40);
		let mut demo = Demo::new(&test_ctx());
		demo.set_right_inset(30);
		let rendered = demo.render_at(viewport, Duration::ZERO);
		let rows: Vec<String> = (0..rendered.frame.size().height)
			.map(|row| frame_row_text(rendered.frame, row))
			.collect();
		let visible = viewport.width - 30;
		let band = rows
			.iter()
			.find(|row| row.contains("(adv)"))
			.expect("split band row");
		let title = rows
			.iter()
			.find(|row| row.contains(super::SESSION_TITLE))
			.expect("session title row");
		for (label, row) in [("band", band), ("title", title)] {
			assert!(
				super::visible_width(row.trim_end()) <= visible,
				"{label} must dock inside the rail reservation: {row}"
			);
		}
		assert!(
			super::visible_width(band.trim_end()) > visible.saturating_sub(2),
			"the band still docks flush against the visible bound: {band}"
		);
	}

	#[test]
	fn shifted_text_and_picker_keys_stay_owned_by_the_editor() {
		let viewport = Size::new(120, 32);
		let mut demo = Demo::new(&test_ctx());
		for character in "Hello World!".chars() {
			assert_eq!(demo.handle_key(Key::Char(character)), DemoKey::Consumed);
		}
		let mut renderer = Renderer::new(Vec::new());
		let rendered = demo.render_at(viewport, Duration::ZERO);
		present(&mut renderer, rendered, viewport).expect("shifted text paints");
		let output =
			str::from_utf8(renderer.writer_mut().as_slice()).expect("renderer output is UTF-8");
		assert!(output.contains("Hello World!"));
		assert_eq!(demo.handle_key(Key::Enter), DemoKey::Consumed);

		assert_eq!(demo.handle_key(Key::Char('/')), DemoKey::Consumed);
		assert_eq!(demo.handle_key(Key::Char('s')), DemoKey::Consumed);
		assert_eq!(demo.handle_key(Key::Char('e')), DemoKey::Consumed);
		assert_eq!(demo.handle_key(Key::BackTab), DemoKey::Ignored);
		renderer.writer_mut().clear();
		let rendered = demo.render_at(viewport, Duration::ZERO);
		present(&mut renderer, rendered, viewport).expect("picker paints after back-tab");
		let picker_output =
			str::from_utf8(renderer.writer_mut().as_slice()).expect("renderer output is UTF-8");
		assert!(picker_output.contains("security"), "picker remains open after back-tab");

		assert_eq!(demo.handle_key(Key::Tab), DemoKey::Consumed);
		renderer.writer_mut().clear();
		let rendered = demo.render_at(viewport, Duration::ZERO);
		let accepted_in_frame = (0..rendered.frame.size().height)
			.any(|row| frame_row_text(rendered.frame, row).contains("/security"));
		present(&mut renderer, rendered, viewport).expect("accepted completion paints");
		assert!(accepted_in_frame, "tab accepts the selected slash command");

		assert_eq!(demo.handle_key(Key::Esc), DemoKey::Consumed);
		assert_eq!(
			demo.handle_key(Key::Esc),
			DemoKey::Consumed,
			"the demo-level esc cancels the running work first"
		);
		assert_eq!(demo.handle_key(Key::Esc), DemoKey::Quit);
	}

	#[test]
	fn multiline_input_grows_inside_the_editor_and_keeps_the_caret_visible() {
		let viewport = Size::new(120, 32);
		let mut demo = Demo::new(&test_ctx());
		demo.set_working(false, Duration::ZERO);
		let (initial_height, initial_status_y) = {
			let rendered = demo.render_at(viewport, Duration::ZERO);
			let height = rendered.frame.size().height;
			let status_y = (0..height)
				.find(|&row| frame_row_text(rendered.frame, row).contains("󰵗 omp"))
				.expect("initial editor status row");
			(height, status_y)
		};
		for text in ["first", "second", "third"] {
			for character in text.chars() {
				demo.handle_key(Key::Char(character));
			}
			if text != "third" {
				demo.handle_key(Key::ShiftEnter);
			}
		}

		let mut renderer = Renderer::new(Vec::new());
		let rendered = demo.render_at(viewport, Duration::ZERO);
		let height = rendered.frame.size().height;
		let rows: Vec<String> = (0..height)
			.map(|row| frame_row_text(rendered.frame, row))
			.collect();
		let status_y = rows
			.iter()
			.position(|row| row.contains("󰵗 omp"))
			.and_then(|row| u16::try_from(row).ok())
			.expect("grown editor status row");
		let input_rows: Vec<u16> = ["first", "second", "third"]
			.into_iter()
			.map(|text| {
				rows
					.iter()
					.position(|row| row.contains(text))
					.and_then(|row| u16::try_from(row).ok())
					.unwrap_or_else(|| panic!("missing input row {text:?}"))
			})
			.collect();

		assert_eq!(height, initial_height, "editor growth is absorbed by the blank band padding");
		assert_eq!(
			status_y,
			initial_status_y.saturating_sub(2),
			"the editor chrome rises as it grows"
		);
		assert_eq!(
			input_rows,
			[status_y + 1, status_y + 2, status_y + 3],
			"input lines must occupy distinct rows below the status chrome",
		);
		assert_eq!(input_rows[2].saturating_add(1), height, "third line stays inside the document");
		assert!(
			["first", "second", "third"]
				.into_iter()
				.all(|text| !rows[usize::from(status_y)].contains(text)),
			"input must not overwrite the status chrome",
		);
		assert!(rows[usize::from(input_rows[0])].starts_with("╰─ first"));
		assert!(rows[usize::from(input_rows[1])].starts_with("   second"));
		assert!(rows[usize::from(input_rows[2])].starts_with("   third"));

		present(&mut renderer, rendered, viewport).expect("multiline editor paints");
		let editor_output =
			str::from_utf8(renderer.writer_mut().as_slice()).expect("renderer output is UTF-8");
		assert!(
			editor_output.contains("\r\x1b[8C\x1b[?25h"),
			"presented caret must account for the prompt before `third`",
		);
		renderer.writer_mut().clear();

		assert_eq!(demo.handle_key(Key::Enter), DemoKey::Consumed);
		let rendered = demo.render_at(viewport, Duration::ZERO);
		present(&mut renderer, rendered, viewport)
			.expect("multiline submission preserves the immutable seam");
		let submission_output =
			String::from_utf8(renderer.into_inner()).expect("renderer output is UTF-8");
		for text in ["first", "second", "third"] {
			assert!(submission_output.contains(text), "submission lost {text:?}");
		}
	}

	#[test]
	fn transcript_header_is_emitted_once_across_commits() {
		let viewport = Size::new(80, 24);
		let mut renderer = Renderer::new(Vec::new());
		let mut demo = Demo::new(&test_ctx());

		for elapsed_ms in [0, 80, 699, 700, 780, 1_400, 2_100] {
			let rendered = demo.render_at(viewport, Duration::from_millis(elapsed_ms));
			present(&mut renderer, rendered, viewport)
				.expect("demo frame satisfies the immutable seam contract");
		}

		let output = String::from_utf8(renderer.into_inner()).expect("renderer output is UTF-8");
		assert_eq!(output.matches("PARALLEL TEST RUN").count(), 1);
		assert!(!output.contains("\x1b[3J"));
	}
	#[test]
	fn ten_minute_session_repaints_only_the_live_suffix() {
		let viewport = Size::new(120, 32);
		let mut demo = Demo::new(&test_ctx());
		let warm = demo.render_at(viewport, Duration::from_millis(599_900));
		let previous_stable_rows = warm.stable_rows;

		let rendered = demo.render_at(viewport, Duration::from_mins(10));

		assert_eq!(rendered.damage.first().map(|range| range.0), Some(previous_stable_rows));
		let damaged_rows: u16 = rendered
			.damage
			.iter()
			.map(|&(start, end)| end.saturating_sub(start))
			.sum();
		assert!(
			damaged_rows <= viewport.height + 8,
			"damaged rows must stay bounded to the live suffix: {:?}",
			rendered.damage
		);
	}

	/// Steady rendering and presentation must stay independent of immutable
	/// transcript size. Run with `cargo test -p omp-tui --release --example
	/// chat -- --ignored perf --nocapture`.
	#[test]
	#[ignore = "release-mode perf smoke, run explicitly"]
	fn perf_render_cost_does_not_grow_with_history() {
		let viewport = Size::new(120, 32);
		let frame_cost = |elapsed| {
			let mut demo = Demo::new(&test_ctx());
			let mut renderer = Renderer::new(std::io::sink());
			let rendered = demo.render_at(viewport, elapsed);
			present(&mut renderer, rendered, viewport).expect("warm-up presentation succeeds");
			const FRAMES: u32 = 100;
			let started_at = Instant::now();
			for frame in 0..FRAMES {
				let rendered =
					demo.render_at(viewport, elapsed + Duration::from_millis(u64::from(frame)));
				black_box(
					present(&mut renderer, rendered, viewport).expect("steady presentation succeeds"),
				);
			}
			started_at.elapsed() / FRAMES
		};

		let short = frame_cost(Duration::from_secs(30));
		let long = frame_cost(Duration::from_mins(10));
		println!("steady frame: {short:?} at 30s vs {long:?} at 10m");
		assert!(
			long.as_nanos() < short.as_nanos() * 3,
			"frame cost still scales with immutable history: {short:?} -> {long:?}"
		);
	}

	fn rows_of(demo: &mut Demo, viewport: Size) -> Vec<String> {
		let rendered = demo.render_at(viewport, Duration::ZERO);
		(0..rendered.frame.size().height)
			.map(|row| frame_row_text(rendered.frame, row))
			.collect()
	}

	#[test]
	fn attach_command_stages_a_framed_preview_and_a_colored_chip() {
		let dir = std::env::temp_dir().join(format!("omp-chat-attach-cmd-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let path = dir.join("shot.png");
		let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
		png.extend(528_u32.to_be_bytes());
		png.extend(200_u32.to_be_bytes());
		std::fs::write(&path, png).unwrap();

		let viewport = Size::new(120, 40);
		let mut demo = Demo::new(&test_ctx());
		for character in format!("/attach {}", path.display()).chars() {
			assert_eq!(demo.handle_key(Key::Char(character)), DemoKey::Consumed);
		}
		assert_eq!(demo.handle_key(Key::Enter), DemoKey::Consumed);
		let rows = rows_of(&mut demo, viewport);
		let caption_row = rows
			.iter()
			.position(|row| row.contains("#1"))
			.expect("preview frame caption above the composer");
		assert!(
			rows.iter().any(|row| row.contains("528x200")),
			"the frame's bottom edge captions the probed resolution"
		);
		let status_row = rows
			.iter()
			.position(|row| row.contains("Fable 5++"))
			.expect("status row");
		assert!(caption_row < status_row, "the preview band sits above the status line");
		assert_eq!(
			rows.iter().filter(|row| row.contains("#1")).count(),
			2,
			"the chip is mentioned in the prompt as well as the frame caption"
		);

		// The chip paints in the attachment's identity color.
		let mut renderer = Renderer::new(Vec::new());
		let rendered = demo.render_at(viewport, Duration::ZERO);
		present(&mut renderer, rendered, viewport).expect("chip paints");
		let output =
			str::from_utf8(renderer.writer_mut().as_slice()).expect("renderer output is UTF-8");
		assert!(output.contains("38;2;255;179;102"), "composer chip and frame use identity color #1");

		for character in "ship it".chars() {
			demo.handle_key(Key::Char(character));
		}
		assert_eq!(demo.handle_key(Key::Enter), DemoKey::Consumed);
		let rows = rows_of(&mut demo, viewport);
		assert!(
			rows.iter().any(|row| row.contains("#1 ship it")),
			"the transcript renders the ref tag as a compact chip pill"
		);
		assert!(
			!rows.iter().any(|row| row.contains("<ref image=1/>")),
			"the raw ref tag never renders"
		);
		assert_eq!(
			rows.iter().filter(|row| row.contains("#1")).count(),
			1,
			"the preview band collapsed after submit"
		);
		std::fs::remove_dir_all(&dir).ok();
	}

	#[test]
	fn deleting_a_chip_hides_its_card_and_undo_restores_it() {
		let dir = std::env::temp_dir().join(format!("omp-chat-attach-del-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let path = dir.join("gone.png");
		std::fs::write(&path, b"\x89PNG\r\n\x1a\n").unwrap();

		let viewport = Size::new(120, 40);
		let mut demo = Demo::new(&test_ctx());
		demo.handle_paste(path.to_str().expect("temp path is UTF-8"));
		assert!(
			rows_of(&mut demo, viewport)
				.iter()
				.any(|row| row.contains("#1"))
		);

		// Backspace over the trailing space, then the chip: one unit.
		demo.handle_key(Key::Backspace);
		demo.handle_key(Key::Backspace);
		let rows = rows_of(&mut demo, viewport);
		assert!(
			!rows.iter().any(|row| row.contains("#1")),
			"deleting the chip removes the card from the band"
		);

		// Undo brings the chip and its card back.
		demo.handle_key(Key::Ctrl('_'));
		assert!(
			rows_of(&mut demo, viewport)
				.iter()
				.any(|row| row.contains("#1"))
		);

		// Deleted again and submitted: the image must not reach the chat.
		demo.handle_key(Key::Ctrl('_'));
		demo.handle_key(Key::Backspace);
		demo.handle_key(Key::Backspace);
		for character in "done".chars() {
			demo.handle_key(Key::Char(character));
		}
		demo.handle_key(Key::Enter);
		let rows = rows_of(&mut demo, viewport);
		assert!(rows.iter().any(|row| row.contains("done")));
		assert!(
			!rows
				.iter()
				.any(|row| row.contains("#1") || row.contains("<ref image=1/>")),
			"a deleted attachment never reaches the transcript"
		);
		std::fs::remove_dir_all(&dir).ok();
	}

	#[test]
	fn large_pastes_collapse_into_text_cards_and_expand_on_submit() {
		let viewport = Size::new(120, 44);
		let mut demo = Demo::new(&test_ctx());
		let paste = (0..12)
			.map(|n| format!("line{n}"))
			.collect::<Vec<_>>()
			.join("\n");
		demo.handle_paste(&paste);
		let rows = rows_of(&mut demo, viewport);
		assert!(rows.iter().any(|row| row.contains("#1")), "paste stages a numbered card");
		assert!(rows.iter().any(|row| row.contains("+12 lines")), "the card captions the paste size");
		assert!(
			rows
				.iter()
				.any(|row| row.contains("line0") && !row.contains("line10")),
			"the card previews the leading paste text"
		);

		demo.handle_key(Key::Enter);
		let rows = rows_of(&mut demo, viewport);
		assert!(
			rows.iter().any(|row| row.contains("line11")),
			"the submitted transcript expands the full paste"
		);
		assert!(!rows.iter().any(|row| row.contains("+12 lines")), "the card collapsed");
	}

	#[test]
	fn pasting_an_image_path_stages_an_attachment_instead_of_inserting() {
		let dir = std::env::temp_dir().join(format!("omp-chat attach-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let path = dir.join("paste drop.png");
		let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
		png.extend([0; 16]);
		std::fs::write(&path, png).unwrap();

		let viewport = Size::new(120, 40);
		let mut demo = Demo::new(&test_ctx());
		// Finder-style drop: quoted because the directory and file name
		// both contain spaces.
		demo.handle_paste(&format!("'{}'", path.display()));
		let rows = rows_of(&mut demo, viewport);
		assert!(
			rows.iter().any(|row| row.contains("#1")),
			"pasted image is staged as a framed preview"
		);
		let input_row = rows
			.iter()
			.rev()
			.find(|row| row.contains(INPUT_PROMPT))
			.expect("composer prompt row");
		assert!(input_row.contains("#1"), "the attachment is mentioned in the prompt: {input_row}");
		assert!(
			!input_row.contains("drop.png"),
			"the path must not be inserted into the input: {input_row}"
		);
		std::fs::remove_dir_all(&dir).ok();
	}
}
