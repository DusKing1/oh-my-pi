use omp_core::{Str, fmts};
use omp_tui::{
	Dim, Key, Layer, Mouse, OverlayAnchor, OverlayOptions, Prop, Size, Ui, UiContext, UiEvent, dom,
};

use crate::PickerEvent;

const LIST_HINT: &str = "↑/↓ choose · Enter select · type to search · Esc close";
const PROMPT_HINT: &str = "Enter submit · Esc cancel";

/// One host-supplied row for [`ListPicker`].
#[derive(Clone, Debug)]
pub struct ListRow {
	/// Stable value associated with this row.
	pub key:    Str,
	/// Primary visible label.
	pub label:  Str,
	/// Secondary visible detail.
	pub detail: Str,
}

/// Single-column filterable picker for sessions, rewind targets, or providers.
pub struct ListPicker {
	ui:        Ui,
	title:     Str,
	rows:      Vec<ListRow>,
	current:   usize,
	ctx:       UiContext,
	options:   OverlayOptions,
	query:     Str,
	list_rows: u16,
}

impl ListPicker {
	/// Opens a titled picker over host-supplied rows.
	pub fn open(title: impl Into<Str>, rows: &[ListRow], current: usize, ctx: &UiContext) -> Self {
		let title = title.into();
		let rows = rows.to_vec();
		let current = current.min(rows.len().saturating_sub(1));
		Self {
			ui: build_list(&title, &rows, current, "", 7, 64, ctx),
			title,
			rows,
			current,
			ctx: ctx.clone(),
			options: OverlayOptions::default()
				.anchor(OverlayAnchor::Center)
				.width(Dim::Cells(64))
				.z(10),
			query: Str::default(),
			list_rows: 7,
		}
	}

	/// Routes a key into the filter and list.
	pub fn handle_key(&mut self, key: Key) -> PickerEvent {
		let event = self.ui.handle_key(key);
		self.route(event)
	}

	/// Routes pasted query text into the filter.
	pub fn handle_paste(&mut self, text: &str) -> PickerEvent {
		let event = self.ui.handle_paste(text);
		self.route(event)
	}

	/// Routes a pointer event; clicking outside dismisses the picker.
	pub fn handle_mouse(&mut self, col: u16, row: u16, kind: Mouse, viewport: Size) -> PickerEvent {
		match self
			.ui
			.handle_mouse_as_layer(&self.options, viewport, col, row, kind)
		{
			Some(event) => self.route(event),
			None if kind == Mouse::Click => PickerEvent::Close,
			None => PickerEvent::Consumed,
		}
	}

	/// Returns a centered, viewport-responsive composited layer.
	pub fn layer(&mut self, viewport: Size) -> Layer<'_> {
		let width = viewport.width.saturating_sub(4).min(72).max(1);
		let rows = (viewport.height / 2).saturating_sub(4).max(5);
		if rows != self.list_rows {
			self.list_rows = rows;
			self
				.ui
				.set_prop("list-picker", Prop::H, rows.saturating_add(1));
		}
		if self.ui.frame().size().width != width {
			self.ui = build_list(
				&self.title,
				&self.rows,
				self.current,
				&self.query,
				self.list_rows,
				width,
				&self.ctx,
			);
		}
		self.options = self.options.width(Dim::Cells(width));
		Layer { frame: self.ui.frame(), options: &self.options, active: true }
	}

	/// Returns the stable key for a picked row index.
	pub fn key(&self, index: usize) -> Option<&Str> {
		self.rows.get(index).map(|row| &row.key)
	}

	fn route(&mut self, event: UiEvent) -> PickerEvent {
		match event {
			UiEvent::Cancel => PickerEvent::Close,
			UiEvent::Changed { value, .. } => value
				.as_str()
				.parse()
				.map_or(PickerEvent::Consumed, PickerEvent::Pick),
			UiEvent::Filtered { query, .. } => {
				self.query = query;
				PickerEvent::Consumed
			},
			UiEvent::None
			| UiEvent::Submit
			| UiEvent::Highlighted { .. }
			| UiEvent::Pressed(_)
			| UiEvent::Copied(_) => PickerEvent::Consumed,
		}
	}
}

fn build_list(
	title: &str,
	rows: &[ListRow],
	current: usize,
	query: &str,
	list_rows: u16,
	width: u16,
	ctx: &UiContext,
) -> Ui {
	let display: Vec<_> = rows
		.iter()
		.enumerate()
		.map(|(index, row)| {
			(
				fmts!("{index}"),
				fmts!("{} {}", row.label, row.detail),
				row.label.clone(),
				row.detail.clone(),
				index == current,
			)
		})
		.collect();
	let title = Str::from(title);
	let seed = Str::from(query);
	let height = list_rows.saturating_add(1);
	Ui::from_root(
		dom! {
			<box border=round title={title} pad-x=1>
				<col>
					<select id="list-picker" filter={seed} h={height}>
						for (value, haystack, label, detail, selected) in display {
							<option value={value} label={haystack} recommended={selected}>
								<td truncate><pre fg=fg>{label}</pre></td>
								<td truncate grow><pre fg=muted>{detail}</pre></td>
							</option>
						}
					</select>
					<text dim truncate>{LIST_HINT}</text>
				</col>
			</box>
		},
		width,
		ctx.clone(),
	)
}

/// Result of routing input through a [`PromptOverlay`].
pub enum PromptEvent {
	/// Event consumed while the prompt remains open.
	Consumed,
	/// Prompt cancelled without a value.
	Cancel,
	/// Prompt submitted with the unmasked value.
	Submit(Str),
}

/// Small rounded-box input overlay for backend authentication prompts.
pub struct PromptOverlay {
	ui:      Ui,
	title:   Str,
	masked:  bool,
	ctx:     UiContext,
	options: OverlayOptions,
}

impl PromptOverlay {
	/// Opens a plain or masked prompt and focuses its input.
	pub fn open(title: impl Into<Str>, masked: bool, ctx: &UiContext) -> Self {
		let title = title.into();
		let mut ui = build_prompt(&title, masked, 56, ctx);
		ui.focus_first();
		Self {
			ui,
			title,
			masked,
			ctx: ctx.clone(),
			options: OverlayOptions::default()
				.anchor(OverlayAnchor::Center)
				.width(Dim::Cells(56))
				.z(20),
		}
	}

	/// Routes a key into the prompt.
	pub fn handle_key(&mut self, key: Key) -> PromptEvent {
		if key == Key::Esc {
			return PromptEvent::Cancel;
		}
		let event = self.ui.handle_key(key);
		self.route(event)
	}

	/// Routes pasted text into the prompt input.
	pub fn handle_paste(&mut self, text: &str) -> PromptEvent {
		let event = self.ui.handle_paste(text);
		self.route(event)
	}

	/// Routes a pointer event; clicking outside cancels the prompt.
	pub fn handle_mouse(&mut self, col: u16, row: u16, kind: Mouse, viewport: Size) -> PromptEvent {
		match self
			.ui
			.handle_mouse_as_layer(&self.options, viewport, col, row, kind)
		{
			Some(event) => self.route(event),
			None if kind == Mouse::Click => PromptEvent::Cancel,
			None => PromptEvent::Consumed,
		}
	}

	/// Returns a centered rounded-box composited layer.
	pub fn layer(&mut self, viewport: Size) -> Layer<'_> {
		let width = viewport.width.saturating_sub(4).min(56).max(1);
		if self.ui.frame().size().width != width {
			let value = self.value();
			self.ui = build_prompt(&self.title, self.masked, width, &self.ctx);
			self.ui.set_text("prompt-input", value);
			self.ui.focus_first();
		}
		self.options = self.options.width(Dim::Cells(width));
		Layer { frame: self.ui.frame(), options: &self.options, active: true }
	}

	fn route(&self, event: UiEvent) -> PromptEvent {
		match event {
			UiEvent::Cancel => PromptEvent::Cancel,
			UiEvent::Submit => PromptEvent::Submit(Str::from(self.value())),
			UiEvent::None
			| UiEvent::Changed { .. }
			| UiEvent::Highlighted { .. }
			| UiEvent::Filtered { .. }
			| UiEvent::Pressed(_)
			| UiEvent::Copied(_) => PromptEvent::Consumed,
		}
	}

	fn value(&self) -> String {
		self.ui.values()["prompt-input"]
			.as_str()
			.unwrap_or_default()
			.to_owned()
	}
}

fn build_prompt(title: &str, masked: bool, width: u16, ctx: &UiContext) -> Ui {
	let title = Str::from(title);
	Ui::from_root(
		dom! {
			<box border=round title={title} pad-x=1 pad-y=1>
				<col>
					<input id="prompt-input" submit mask={masked} placeholder="Enter value"/>
					<spacer h=1/>
					<text dim truncate>{PROMPT_HINT}</text>
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
	fn list_picker_keeps_host_keys_out_of_option_values() {
		let rows = vec![ListRow {
			key:    Str::from("session/opaque"),
			label:  Str::from("Session"),
			detail: Str::from("today"),
		}];
		let picker = ListPicker::open("Resume", &rows, 0, &UiContext::default());
		assert_eq!(picker.key(0).map(Str::as_str), Some("session/opaque"));
	}

	#[test]
	fn masked_prompt_returns_original_value() {
		let mut prompt = PromptOverlay::open("Token", true, &UiContext::default());
		for ch in "secret".chars() {
			assert!(matches!(prompt.handle_key(Key::Char(ch)), PromptEvent::Consumed));
		}
		match prompt.handle_key(Key::Enter) {
			PromptEvent::Submit(value) => assert_eq!(value.as_str(), "secret"),
			_ => panic!("prompt did not submit"),
		}
	}
}
