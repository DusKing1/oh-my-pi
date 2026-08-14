//! Filterable model picker backed exclusively by host-supplied catalog rows.

use omp_core::{Str, StrMut, fmts};
use omp_tui::{
	Dim, IntoComponent as _, Key, Layer, Mouse, OverlayAnchor, OverlayOptions, Prop, Size, Ui,
	UiContext, UiEvent, assets::provider_logo, dom,
};

use crate::ModelRow;

const HINT: &str = "↑/↓ models · Enter switch · type to search · Esc close";
const FRAME_ROWS: u16 = 6;
const CONTEXT_WIDTH: u16 = 62;
const INPUT_PRICE_WIDTH: u16 = 76;
const OUTPUT_PRICE_WIDTH: u16 = 88;

/// What a routed picker event did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PickerEvent {
	/// The picker consumed the event and remains open.
	Consumed,
	/// Close without choosing a row.
	Close,
	/// Choose the row at this index.
	Pick(usize),
}

/// Retained filterable model-picker overlay.
pub struct ModelPicker {
	ui:        Ui,
	rows:      Vec<ModelRow>,
	current:   usize,
	ctx:       UiContext,
	options:   OverlayOptions,
	query:     Str,
	list_rows: u16,
}

impl ModelPicker {
	/// Opens the picker over host-supplied rows with `current` preselected.
	pub fn open(rows: &[ModelRow], current: usize, ctx: &UiContext) -> Self {
		let rows = rows.to_vec();
		let current = current.min(rows.len().saturating_sub(1));
		let ui = build(&rows, current, "", 6, 100, ctx);
		let mut picker = Self {
			ui,
			rows,
			current,
			ctx: ctx.clone(),
			options: OverlayOptions::default()
				.anchor(OverlayAnchor::Bottom)
				.width(Dim::Pct(100))
				.z(10),
			query: Str::default(),
			list_rows: 6,
		};
		picker.show_detail((!picker.rows.is_empty()).then_some(current));
		picker
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

	/// Routes a pointer event; clicking outside dismisses the overlay.
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

	/// Returns the bottom-anchored composited layer for this frame.
	pub fn layer(&mut self, viewport: Size) -> Layer<'_> {
		let rows = (viewport.height * 2 / 5).saturating_sub(FRAME_ROWS).max(5);
		if rows != self.list_rows {
			self.list_rows = rows;
			self.ui.set_prop("models", Prop::H, rows.saturating_add(1));
		}
		if self.ui.frame().size().width != viewport.width {
			self.rebuild(viewport.width);
		}
		Layer { frame: self.ui.frame(), options: &self.options, active: true }
	}

	fn route(&mut self, event: UiEvent) -> PickerEvent {
		match event {
			UiEvent::Cancel => PickerEvent::Close,
			UiEvent::Changed { value, .. } => value
				.as_str()
				.parse()
				.map_or(PickerEvent::Consumed, PickerEvent::Pick),
			UiEvent::Highlighted { value, .. } => {
				self.show_detail(value.as_str().parse().ok());
				PickerEvent::Consumed
			},
			UiEvent::Filtered { query, value, .. } => {
				self.query = query;
				self.show_detail(value.and_then(|value| value.as_str().parse().ok()));
				PickerEvent::Consumed
			},
			UiEvent::None | UiEvent::Submit | UiEvent::Pressed(_) | UiEvent::Copied(_) => {
				PickerEvent::Consumed
			},
		}
	}

	fn rebuild(&mut self, width: u16) {
		self.ui = build(&self.rows, self.current, &self.query, self.list_rows, width, &self.ctx);
		self.show_detail((!self.rows.is_empty()).then_some(self.current));
	}

	fn show_detail(&mut self, model: Option<usize>) {
		let text = model
			.and_then(|index| self.rows.get(index))
			.map_or_else(|| Str::new_static(" "), facts);
		self.ui.set_text("model-facts", text);
	}
}

struct DisplayRow {
	value:    Str,
	label:    Str,
	logo_src: Option<Str>,
	provider: Str,
	name:     Str,
	current:  bool,
	context:  Str,
	input:    Str,
	output:   Str,
}

fn build(
	rows: &[ModelRow],
	current: usize,
	query: &str,
	list_rows: u16,
	width: u16,
	ctx: &UiContext,
) -> Ui {
	let show_context = width >= CONTEXT_WIDTH && rows.iter().any(|row| row.context.is_some());
	let show_input = width >= INPUT_PRICE_WIDTH && rows.iter().any(|row| row.input_mtok.is_some());
	let show_output =
		width >= OUTPUT_PRICE_WIDTH && rows.iter().any(|row| row.output_mtok.is_some());
	let display: Vec<_> = rows
		.iter()
		.enumerate()
		.map(|(index, row)| DisplayRow {
			value:    fmts!("{index}"),
			label:    fmts!("{} {} {}", row.provider, row.name, row.key),
			logo_src: provider_logo(row.provider_id.as_str())
				.is_some()
				.then(|| fmts!("asset://login/{}", row.provider_id)),
			provider: if row.provider.is_empty() {
				row.provider_id.clone()
			} else {
				row.provider.clone()
			},
			name:     if row.name.is_empty() {
				row.key.clone()
			} else {
				row.name.clone()
			},
			current:  index == current,
			context:  row
				.context
				.map_or_else(Str::default, |tokens| fmts!("{} ctx", compact_count(tokens))),
			input:    row
				.input_mtok
				.map_or_else(Str::default, |cost| fmts!("${cost} in")),
			output:   row
				.output_mtok
				.map_or_else(Str::default, |cost| fmts!("${cost} out")),
		})
		.collect();
	let seed = Str::from(query);
	let current_mark = Str::new_static(" current");
	let height = list_rows.saturating_add(1);
	Ui::from_root(
		dom! {
			<box border=round title="Switch Model" pad-x=1>
				<col>
					<select id="models" filter={seed} h={height}>
						for row in display {
							<option value={row.value} label={row.label} recommended={row.current}>
								<td>
									if let Some(src) = row.logo_src.clone() { <img src={src} w=2 h=1/> }
								</td>
								<td truncate>
									<pre fg=fg bg=border>{" "}{row.provider}{" "}</pre>
								</td>
								<td truncate=start grow>
									<pre fg=fg>{row.name}</pre>
									if row.current { <pre fg=ok>{current_mark.clone()}</pre> }
								</td>
								if show_context { <td align=end><pre fg=muted>{row.context}</pre></td> }
								if show_input { <td align=end><pre fg=muted>{row.input}</pre></td> }
								if show_output { <td align=end><pre fg=muted>{row.output}</pre></td> }
							</option>
						}
					</select>
					<spacer h=1/>
					<text id="model-facts" fg=muted truncate>{" "}</text>
					<text fg=muted truncate>{HINT}</text>
				</col>
			</box>
		}
		.into_component(),
		width,
		ctx.clone(),
	)
}

fn facts(row: &ModelRow) -> Str {
	let mut line = StrMut::with_capacity(96);
	push_fact(
		&mut line,
		if row.name.is_empty() {
			&row.key
		} else {
			&row.name
		},
	);
	push_fact(&mut line, &row.provider);
	if let Some(context) = row.context {
		push_fact(&mut line, &fmts!("{} context", compact_count(context)));
	}
	if let Some(price) = price(row) {
		push_fact(&mut line, &fmts!("{price} per Mtok"));
	}
	line.freeze()
}

fn price(row: &ModelRow) -> Option<Str> {
	match (row.input_mtok, row.output_mtok) {
		(Some(input), Some(output)) => Some(fmts!("${input}/${output}")),
		(Some(input), None) => Some(fmts!("${input} in")),
		(None, Some(output)) => Some(fmts!("${output} out")),
		(None, None) => None,
	}
}

fn push_fact(line: &mut StrMut, text: &str) {
	if !line.is_empty() {
		line.push_str(" · ");
	}
	line.push_str(text);
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

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn absent_facts_are_omitted() {
		let row = ModelRow {
			key:         Str::from("p/m"),
			name:        Str::from("Model"),
			provider_id: Str::from("p"),
			provider:    Str::from("Provider"),
			context:     None,
			input_mtok:  None,
			output_mtok: None,
		};
		let facts = facts(&row);
		assert!(!facts.contains("ctx"));
		assert!(!facts.contains('$'));
	}
}
