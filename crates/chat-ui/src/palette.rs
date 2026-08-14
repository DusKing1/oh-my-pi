//! Command palette whose rows and actions are injected by the host.

use omp_core::{Str, fmts};
use omp_tui::{
	Color, Dim, Key, Layer, Mouse, OverlayAnchor, OverlayOptions, Prop, Size, Ui, UiContext,
	UiEvent, dom,
};

use crate::Intent;

const HINT: &str = "↑/↓ commands · Enter run · type to search · Esc close";
const FRAME_ROWS: u16 = 4;

/// Action carried by one palette row.
#[derive(Clone)]
pub enum PaletteAction {
	/// Forward this intent to the backend.
	Intent(Intent),
	/// Open the model picker locally.
	OpenModelPicker,
	/// Toggle host-owned sidebar chrome.
	ToggleSidebar,
	/// Insert text into the composer without submitting it.
	Insert(Str),
}

/// Host-supplied command-palette row.
#[derive(Clone)]
pub struct PaletteEntry {
	label:  Str,
	detail: Str,
	key:    Str,
	action: PaletteAction,
	accent: bool,
}

impl PaletteEntry {
	/// Creates an executable palette row.
	pub fn new(label: impl Into<Str>, detail: impl Into<Str>, action: PaletteAction) -> Self {
		Self {
			label: label.into(),
			detail: detail.into(),
			key: Str::default(),
			action,
			accent: false,
		}
	}

	/// Adds a right-aligned shortcut hint.
	#[must_use]
	pub fn key(mut self, key: impl Into<Str>) -> Self {
		self.key = key.into();
		self
	}

	/// Uses the accent color, appropriate for slash-command insertion rows.
	#[must_use]
	pub const fn accent(mut self, accent: bool) -> Self {
		self.accent = accent;
		self
	}

	/// Returns the primary label.
	pub fn label(&self) -> &str {
		&self.label
	}

	/// Returns the secondary detail.
	pub fn detail(&self) -> &str {
		&self.detail
	}
}

/// Result of routing an input event through the command palette.
pub enum PaletteEvent {
	/// Event consumed while the palette remains open.
	Consumed,
	/// Palette dismissed without an action.
	Close,
	/// Activated action for the host to execute.
	Run(PaletteAction),
}

/// Retained filterable command-palette overlay.
pub struct CommandPalette {
	ui:      Ui,
	ctx:     UiContext,
	options: OverlayOptions,
	entries: Vec<PaletteEntry>,
	query:   Str,
	rows:    u16,
}

impl CommandPalette {
	/// Opens a palette over the supplied executable rows.
	pub fn open(entries: Vec<PaletteEntry>, ctx: &UiContext) -> Self {
		Self {
			ui: build(&entries, "", 8, 100, ctx),
			ctx: ctx.clone(),
			options: OverlayOptions::default()
				.anchor(OverlayAnchor::Top)
				.offset_y(1)
				.z(10),
			entries,
			query: Str::default(),
			rows: 8,
		}
	}

	/// Converts slash completion commands into insertion entries and appends
	/// them to `builtins`.
	pub fn with_slash_commands(
		mut builtins: Vec<PaletteEntry>,
		commands: &[omp_tui::Command],
	) -> Vec<PaletteEntry> {
		builtins.extend(commands.iter().map(|command| {
			let slash = fmts!("/{}", command.name());
			PaletteEntry::new(
				slash.clone(),
				command.description(),
				PaletteAction::Insert(fmts!("{slash} ")),
			)
			.accent(true)
		}));
		builtins
	}

	/// Routes a key into the filter and list.
	pub fn handle_key(&mut self, key: Key) -> PaletteEvent {
		let event = self.ui.handle_key(key);
		self.route(event)
	}

	/// Routes pasted query text into the filter.
	pub fn handle_paste(&mut self, text: &str) -> PaletteEvent {
		let event = self.ui.handle_paste(text);
		self.route(event)
	}

	/// Routes a pointer event; clicking outside dismisses the palette.
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

	/// Returns the top-anchored composited layer for this frame.
	pub fn layer(&mut self, viewport: Size) -> Layer<'_> {
		let width = (viewport.width * 3 / 5).max(48).min(viewport.width);
		let rows = (viewport.height / 2).saturating_sub(FRAME_ROWS).max(5);
		if rows != self.rows {
			self.rows = rows;
			self
				.ui
				.set_prop("commands", Prop::H, rows.saturating_add(1));
		}
		if self.ui.frame().size().width != width {
			self.ui = build(&self.entries, &self.query, self.rows, width, &self.ctx);
		}
		self.options = self.options.width(Dim::Cells(width));
		Layer { frame: self.ui.frame(), options: &self.options, active: true }
	}

	fn route(&mut self, event: UiEvent) -> PaletteEvent {
		match event {
			UiEvent::Cancel => PaletteEvent::Close,
			UiEvent::Changed { value, .. } => value
				.as_str()
				.parse::<usize>()
				.ok()
				.and_then(|index| self.entries.get(index))
				.map_or(PaletteEvent::Consumed, |entry| PaletteEvent::Run(entry.action.clone())),
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

struct DisplayEntry {
	value:  Str,
	label:  Str,
	detail: Str,
	key:    Str,
	color:  Color,
}

fn build(entries: &[PaletteEntry], query: &str, rows: u16, width: u16, ctx: &UiContext) -> Ui {
	let display: Vec<_> = entries
		.iter()
		.enumerate()
		.map(|(index, entry)| DisplayEntry {
			value:  fmts!("{index}"),
			label:  entry.label.clone(),
			detail: entry.detail.clone(),
			key:    entry.key.clone(),
			color:  if entry.accent {
				ctx.theme.info
			} else {
				ctx.theme.fg
			},
		})
		.collect();
	let seed = Str::from(query);
	let height = rows.saturating_add(1);
	Ui::from_root(
		dom! {
			<box border=round title="Commands" pad-x=1>
				<col>
					<select id="commands" filter={seed} h={height}>
						for entry in display {
							<option value={entry.value} label={entry.label.clone()}>
								<td><pre fg={entry.color}>{entry.label}</pre></td>
								<td truncate grow><pre fg=muted>{entry.detail}</pre></td>
								if !entry.key.is_empty() { <td align=end><pre fg=muted>{entry.key}</pre></td> }
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

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn slash_entries_are_injected_not_built_in() {
		let commands = vec![omp_tui::Command::new("help", "Show help", &[])];
		let rows = CommandPalette::with_slash_commands(Vec::new(), &commands);
		assert_eq!(rows.len(), 1);
		assert_eq!(rows[0].label(), "/help");
	}
}
