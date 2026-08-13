//! Command palette: a VS Code-style quick command overlay (`Ctrl+K`).
//!
//! A top-anchored `<select filter>` lists the demo's executable actions —
//! switch model, toggle the sidebar, quit — followed by every slash
//! command the composer completes; picking a slash entry stages `/name `
//! in the composer instead of pretending to run it. The core widget owns
//! the query editor, fuzzy ranking, cursor movement, windowed scrolling,
//! hover, wheel, and click activation — the palette only routes the
//! surfaced [`UiEvent`]s, exactly like the model picker.

use omp_core::{Str, fmts};
use omp_tui::{
	Color, Dim, Key, Layer, Mouse, OverlayAnchor, OverlayOptions, Prop, Size, Ui, UiContext,
	UiEvent, dom,
};

use crate::demo::demo_commands;

const CYAN: Color = Color::Rgb(62, 190, 203);
const TEXT: Color = Color::Rgb(194, 198, 204);
const DIM: Color = Color::Rgb(110, 116, 124);

const HINT: &str = "↑/↓ commands · Enter run · type to search · Esc close";

/// Rows the palette occupies beyond the list: box borders, the select's
/// query row, and the hint bar.
const FRAME_ROWS: u16 = 4;

/// What a routed input event did to the palette.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PaletteEvent {
	/// The event was handled; the palette stays open.
	Consumed,
	/// The palette dismissed without running anything.
	Close,
	/// An entry was activated; the host executes it and closes.
	Run(PaletteAction),
}

/// One executable palette entry.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PaletteAction {
	/// Open the model picker (`Ctrl+P`).
	SwitchModel,
	/// Toggle the session rail (`Ctrl+B`).
	ToggleSidebar,
	/// Exit the demo (`Ctrl+C`).
	Quit,
	/// Stage this text in the composer (slash commands).
	Insert(Str),
}

/// Sentinel values for the built-in actions; slash entries carry their
/// `/name` spelling as the value, so bare names never collide.
const SWITCH_MODEL: &str = "switch-model";
const TOGGLE_SIDEBAR: &str = "toggle-sidebar";
const QUIT: &str = "quit";

/// Retained palette overlay: one `Ui` for the whole entry list, rebuilt
/// only on width changes; everything else is core select state.
pub struct CommandPalette {
	ui:      Ui,
	ctx:     UiContext,
	options: OverlayOptions,
	/// Query carried across width rebuilds.
	query:   Str,
	/// List rows granted by the last viewport.
	rows:    u16,
}

impl CommandPalette {
	/// Opens the palette, presenting through the host's detected context.
	pub fn open(ctx: &UiContext) -> Self {
		let options = OverlayOptions::default()
			.anchor(OverlayAnchor::Top)
			.offset_y(1)
			.z(10);
		Self { ui: build("", 8, 100, ctx), ctx: ctx.clone(), options, query: Str::default(), rows: 8 }
	}

	/// Routes a key through the retained tree and maps the surfaced event.
	pub fn handle_key(&mut self, key: Key) -> PaletteEvent {
		let event = self.ui.handle_key(key);
		self.route(event)
	}

	/// Routes pasted text into the select's query editor.
	pub fn handle_paste(&mut self, text: &str) -> PaletteEvent {
		let event = self.ui.handle_paste(text);
		self.route(event)
	}

	/// Routes a mouse report through the compositor's own band; a click
	/// outside the layer dismisses the palette.
	pub fn handle_mouse(&mut self, col: u16, row: u16, kind: Mouse, viewport: Size) -> PaletteEvent {
		match self
			.ui
			.handle_mouse_as_layer(&self.options, viewport, col, row, kind)
		{
			Some(event) => self.route(event),
			None if kind == Mouse::Click => PaletteEvent::Close,
			None => PaletteEvent::Consumed,
		}
	}

	/// The composited layer for this frame: top-anchored, 60% wide (at
	/// least 48 cells), at most half the viewport tall.
	pub fn layer(&mut self, viewport: Size) -> Layer<'_> {
		let width = (viewport.width * 3 / 5).max(48).min(viewport.width);
		let rows = (viewport.height / 2).saturating_sub(FRAME_ROWS).max(5);
		if rows != self.rows {
			self.rows = rows;
			// One query row plus the windowed list.
			self
				.ui
				.set_prop("commands", Prop::H, rows.saturating_add(1));
		}
		if self.ui.frame().size().width != width {
			self.ui = build(&self.query, self.rows, width, &self.ctx);
		}
		self.options = self.options.width(Dim::Cells(width));
		Layer { frame: self.ui.frame(), options: &self.options, active: true }
	}

	/// Applies one surfaced [`UiEvent`] to palette state.
	fn route(&mut self, event: UiEvent) -> PaletteEvent {
		match event {
			UiEvent::Cancel => PaletteEvent::Close,
			UiEvent::Changed { value, .. } => match value.as_str() {
				SWITCH_MODEL => PaletteEvent::Run(PaletteAction::SwitchModel),
				TOGGLE_SIDEBAR => PaletteEvent::Run(PaletteAction::ToggleSidebar),
				QUIT => PaletteEvent::Run(PaletteAction::Quit),
				slash => PaletteEvent::Run(PaletteAction::Insert(fmts!("{slash} "))),
			},
			UiEvent::Filtered { query, .. } => {
				self.query = query;
				PaletteEvent::Consumed
			},
			UiEvent::None
			| UiEvent::Submit
			| UiEvent::Highlighted { .. }
			| UiEvent::Pressed(_)
			| UiEvent::Copied(_) => PaletteEvent::Consumed,
		}
	}
}

/// One option row's static content.
struct EntrySpec {
	value:   Str,
	label:   Str,
	name:    Str,
	name_fg: Color,
	detail:  Str,
	/// Right-aligned keybinding column; empty for slash entries.
	key:     Str,
}

impl EntrySpec {
	const fn action(
		value: &'static str,
		name: &'static str,
		detail: &'static str,
		key: &'static str,
	) -> Self {
		Self {
			value:   Str::new_static(value),
			label:   Str::new_static(name),
			name:    Str::new_static(name),
			name_fg: TEXT,
			detail:  Str::new_static(detail),
			key:     Str::new_static(key),
		}
	}
}

/// The full entry list: built-in actions first, slash commands after,
/// mirroring VS Code's palette ordering (commands, then everything else).
fn entries() -> Vec<EntrySpec> {
	let commands = demo_commands();
	let mut list = Vec::with_capacity(commands.len() + 3);
	list.push(EntrySpec::action(
		SWITCH_MODEL,
		"Switch Model",
		"Pick the model for this session",
		"ctrl+p",
	));
	list.push(EntrySpec::action(
		TOGGLE_SIDEBAR,
		"Toggle Sidebar",
		"Show or hide the session rail",
		"ctrl+b",
	));
	list.push(EntrySpec::action(QUIT, "Quit", "Exit the demo", "ctrl+c"));
	list.extend(commands.iter().map(|command| {
		let name = fmts!("/{}", command.name());
		EntrySpec {
			value: name.clone(),
			label: name.clone(),
			name,
			name_fg: CYAN,
			detail: Str::from(command.description()),
			key: Str::default(),
		}
	}));
	list
}

/// Builds the retained overlay tree.
fn build(query: &str, rows: u16, width: u16, ctx: &UiContext) -> Ui {
	let list = entries();
	let seed = Str::from(query);
	let height = rows.saturating_add(1);
	Ui::from_root(
		dom! {
			<box border=round title="Commands" pad-x=1>
				<col>
					<select id="commands" filter={seed} h={height}>
						for entry in list {
							<option value={entry.value} label={entry.label}>
								<td><pre fg={entry.name_fg}>{entry.name}</pre></td>
								<td truncate grow><pre fg={DIM}>{entry.detail}</pre></td>
								if !entry.key.is_empty() {
									<td align=end><pre fg={DIM}>{entry.key}</pre></td>
								}
							</option>
						}
					</select>
					<text dim truncate>{HINT}</text>
				</col>
			</box>
		},
		width,
		ctx.clone(),
	)
}
