//! Provider login picker: a reflowing grid of focusable logo cards with
//! incremental type-to-filter, modeled on the `companies` roster example.

use omp_core::{Str, fmts};
use omp_tui::{
	Dim, Key, Layer, Mouse, OverlayAnchor, OverlayOptions, Size, Ui, UiContext, UiEvent, dom,
};

use crate::PickerEvent;

const HINT: &str = "type to filter · ↹/←→/↑↓ pick · ↵ login · Esc close";
/// Card cell: rounded border (2) + 1-cell side padding around a 12-wide body
/// that fits the 4-cell logo and every shortened provider name.
const CARD_W: u16 = 16;

/// Grid picker over login providers; Enter or click on a card picks it.
pub struct ProviderPicker {
	ui:        Ui,
	rows:      Vec<crate::SessionRow>,
	query:     String,
	ctx:       UiContext,
	options:   OverlayOptions,
	grid_rows: u16,
	width:     u16,
}

impl ProviderPicker {
	/// Opens the card grid over host-supplied provider rows; `row.id` is the
	/// provider key and `row.label` the display name.
	pub fn open(rows: Vec<crate::SessionRow>, ctx: &UiContext) -> Self {
		let mut picker = Self {
			ui: Ui::from_root(dom! { <text>{""}</text> }, 1, ctx.clone()),
			rows,
			query: String::new(),
			ctx: ctx.clone(),
			options: OverlayOptions::default()
				.anchor(OverlayAnchor::Center)
				.width(Dim::Cells(72))
				.z(10),
			grid_rows: 12,
			width: 72,
		};
		picker.rebuild();
		picker
	}

	/// Routes a key: printable characters narrow the filter, Esc clears it
	/// before closing, everything else drives card focus and selection.
	pub fn handle_key(&mut self, key: Key) -> PickerEvent {
		match key {
			Key::Char(ch) if !ch.is_control() => {
				self.query.push(ch);
				self.rebuild();
				PickerEvent::Consumed
			},
			Key::Backspace if !self.query.is_empty() => {
				self.query.pop();
				self.rebuild();
				PickerEvent::Consumed
			},
			Key::Esc if !self.query.is_empty() => {
				self.query.clear();
				self.rebuild();
				PickerEvent::Consumed
			},
			key => {
				let event = self.ui.handle_key(key);
				Self::route(event)
			},
		}
	}

	/// Routes pasted text into the filter.
	pub fn handle_paste(&mut self, text: &str) -> PickerEvent {
		let mut changed = false;
		for ch in text.chars().filter(|ch| !ch.is_control()) {
			self.query.push(ch);
			changed = true;
		}
		if changed {
			self.rebuild();
		}
		PickerEvent::Consumed
	}

	/// Routes a pointer event; clicking outside dismisses the picker.
	pub fn handle_mouse(&mut self, col: u16, row: u16, kind: Mouse, viewport: Size) -> PickerEvent {
		match self
			.ui
			.handle_mouse_as_layer(&self.options, viewport, col, row, kind)
		{
			Some(event) => Self::route(event),
			None if kind == Mouse::Click => PickerEvent::Close,
			None => PickerEvent::Consumed,
		}
	}

	/// Returns a centered, viewport-responsive composited layer.
	pub fn layer(&mut self, viewport: Size) -> Layer<'_> {
		let width = viewport.width.saturating_sub(4).clamp(24, 76);
		let rows = (viewport.height.saturating_sub(8)).max(6);
		if width != self.width || rows != self.grid_rows {
			self.width = width;
			self.grid_rows = rows;
			self.rebuild();
		}
		self.options = self.options.width(Dim::Cells(width));
		Layer { frame: self.ui.frame(), options: &self.options, active: true }
	}

	/// Returns the provider key for a picked row index.
	pub fn key(&self, index: usize) -> Option<&Str> {
		self.rows.get(index).map(|row| &row.id)
	}

	fn matches(&self, row: &crate::SessionRow) -> bool {
		if self.query.is_empty() {
			return true;
		}
		let query = self.query.to_lowercase();
		row.id.as_str().to_lowercase().contains(&query)
			|| row.label.as_str().to_lowercase().contains(&query)
	}

	fn rebuild(&mut self) {
		let cards: Vec<ProviderCard> = self
			.rows
			.iter()
			.enumerate()
			.filter(|(_, row)| self.matches(row))
			.map(|(index, row)| ProviderCard {
				press_id:    fmts!("{index}"),
				provider_id: row.id.clone(),
				label:       row.label.clone(),
			})
			.collect();
		let shown = cards.len();
		let total = self.rows.len();
		let counter = if self.query.is_empty() {
			fmts!("{total} providers")
		} else {
			fmts!("{shown}/{total} · filter: {}", self.query)
		};
		self.ui = Ui::from_root(
			dom! {
				<box border=round title="Provider Login" pad-x=1>
					{provider_card_grid(cards, counter, HINT, self.grid_rows)}
				</box>
			},
			self.width,
			self.ctx.clone(),
		);
	}

	fn route(event: UiEvent) -> PickerEvent {
		match event {
			UiEvent::Cancel => PickerEvent::Close,
			UiEvent::Pressed(id) => id
				.as_str()
				.parse()
				.map_or(PickerEvent::Consumed, PickerEvent::Pick),
			UiEvent::None
			| UiEvent::Submit
			| UiEvent::Changed { .. }
			| UiEvent::Filtered { .. }
			| UiEvent::Highlighted { .. }
			| UiEvent::Copied(_) => PickerEvent::Consumed,
		}
	}
}

/// One card in the shared provider grid.
#[derive(Clone, Debug)]
pub struct ProviderCard {
	/// Id emitted through `Pressed` when the card is picked.
	pub press_id:    Str,
	/// Catalog provider id; selects the packaged logo or monogram.
	pub provider_id: Str,
	/// Display label under the logo.
	pub label:       Str,
}

/// Builds the reflowing provider card grid shared by the chat overlay and
/// the setup wizard.
///
/// Focusable bordered logo cards flow in a wrapping, centered row inside a
/// scroll region; Enter or click presses a card's `press_id`.
pub fn provider_card_grid(
	cards: Vec<ProviderCard>,
	counter: Str,
	hint: &'static str,
	grid_rows: u16,
) -> impl omp_tui::IntoComponent {
	dom! {
		<col gap=1>
			<row gap=1>
				<i:log-in/>
				<text bold fg="accent..info">{"Choose a provider"}</text>
				<text dim truncate>{counter}</text>
			</row>
			<scroll id="login-provider-grid" h={grid_rows}>
				<row wrap gap=1 justify=center>
					for card in cards {
						<box focus id={card.press_id} w={CARD_W} border=round bc="muted..muted"
							hover="accent..info" lift=1 anim=220 ease=in-out
							align=center pad-x=1>
							<logo id={card.provider_id.as_str()} w=4 h=2/>
							<text bold truncate align=center>{card.label}</text>
						</box>
					}
				</row>
			</scroll>
			<text dim truncate>{hint}</text>
		</col>
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::SessionRow;

	fn rows() -> Vec<SessionRow> {
		["anthropic", "github-copilot", "deepseek"]
			.into_iter()
			.map(|id| SessionRow {
				id:     Str::new_static(id),
				label:  Str::new_static(id),
				detail: Str::default(),
			})
			.collect()
	}

	#[test]
	fn filter_narrows_without_stacking_and_esc_clears_before_close() {
		let mut picker = ProviderPicker::open(rows(), &UiContext::default());
		for ch in "cop".chars() {
			assert_eq!(picker.handle_key(Key::Char(ch)), PickerEvent::Consumed);
		}
		assert_eq!(picker.query, "cop");
		assert_eq!(picker.handle_key(Key::Esc), PickerEvent::Consumed);
		assert!(picker.query.is_empty());
		assert_eq!(picker.handle_key(Key::Esc), PickerEvent::Close);
	}

	#[test]
	fn picked_index_resolves_the_provider_key() {
		let picker = ProviderPicker::open(rows(), &UiContext::default());
		assert_eq!(picker.key(1).map(Str::as_str), Some("github-copilot"));
		assert_eq!(picker.key(9), None);
	}
}
