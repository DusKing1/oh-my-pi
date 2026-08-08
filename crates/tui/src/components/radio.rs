use omp_core::SmolStr;
use smallvec::SmallVec;

use crate::{
	component::{Component, EventCtx, Flow, Hit, HitTag, PaintCtx, Slot, next_slot},
	context::{Theme, UiContext},
	frame::{Color, Frame, Rect, Style},
	input::{Key, Mouse},
	props::{Prop, PropValue, Props},
	rich::cell_width,
};

#[derive(Default)]
struct RadioState {
	options: SmallVec<SmolStr, 8>,
	idx:     u16,
	spans:   SmallVec<(u16, u16), 8>,
}

/// A compact single-choice row of chips backing the `<radio>` markup tag.
pub struct Radio {
	props: Props,
	slot:  Slot,
	state: RadioState,
}

impl Radio {
	/// Creates an empty radio group.
	pub fn new() -> Self {
		Self { props: Props::new(), slot: next_slot(), state: RadioState::default() }
	}

	/// Sets one radio property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		if matches!(prop, Prop::Options | Prop::Value) {
			self.sync_options();
		}
		self
	}

	/// Sets one radio property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		if matches!(prop, Prop::Options | Prop::Value) {
			self.sync_options();
		}
		self
	}

	fn sync_options(&mut self) {
		self.state.options = self
			.props
			.str_of(Prop::Options)
			.map(|options| {
				options
					.split_whitespace()
					.map(|word| options.slice_ref(word))
					.collect()
			})
			.unwrap_or_default();
		self.state.idx = self
			.props
			.str_of(Prop::Value)
			.and_then(|value| self.state.options.iter().position(|option| option == value))
			.unwrap_or(0) as u16;
	}
}

impl Default for Radio {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Radio {
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
		let total = self
			.state
			.options
			.iter()
			.map(|option| cell_width(option).saturating_add(3))
			.fold(2u16, u16::saturating_add);
		(total, total)
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		1
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		if rect.y >= pc.clip {
			return;
		}
		let focused = pc.focus == Some(self.slot);
		let hover_chip = match pc.hover {
			Some((slot, HitTag::Chip(index))) if slot == self.slot => Some(index),
			_ => None,
		};
		self.state.spans.clear();
		let mut x = pc.frame.put(
			rect.x,
			rect.y,
			if focused {
				pc.ctx.charset.cursor()
			} else {
				"  "
			},
			Style::new().fg(pc.ctx.theme.accent),
		);
		for (index, option) in self.state.options.iter().enumerate() {
			let start = x.saturating_sub(rect.x);
			let active = index as u16 == self.state.idx;
			let hovered = hover_chip == Some(index as u16);
			if active {
				x = pill(
					pc.frame,
					x,
					rect.y,
					option,
					pc.ctx.theme.accent,
					pc.ctx.theme.contrast,
					pc.ctx.charset.pill_caps(),
					focused || hovered,
				);
			} else {
				let mut style = Style::new().fg(if hovered {
					pc.ctx.theme.fg
				} else {
					pc.ctx.theme.muted
				});
				if hovered {
					style = style.underline();
				}
				x = pc.frame.put(x, rect.y, " ", base(&pc.ctx.theme));
				x = pc.frame.put(x, rect.y, option, style);
				x = pc.frame.put(x, rect.y, " ", base(&pc.ctx.theme));
			}
			let end = x.saturating_sub(rect.x);
			self.state.spans.push((start, end));
			pc.hits.push(Hit {
				rect: Rect::new(rect.x.saturating_add(start), rect.y, end.saturating_sub(start), 1),
				slot: self.slot,
				tag:  HitTag::Chip(index as u16),
			});
			x = pc.frame.put(x, rect.y, " ", base(&pc.ctx.theme));
		}
	}

	fn focusable(&self) -> bool {
		true
	}

	fn key(&mut self, _ec: &mut EventCtx<'_>, key: Key) -> Flow {
		let len = self.state.options.len() as u16;
		match key {
			Key::Left if len > 0 => {
				self.state.idx = (self.state.idx + len - 1) % len;
				Flow::Consumed
			},
			Key::Right if len > 0 => {
				self.state.idx = (self.state.idx + 1) % len;
				Flow::Consumed
			},
			_ => Flow::Skip,
		}
	}

	fn mouse(
		&mut self,
		_ec: &mut EventCtx<'_>,
		tag: HitTag,
		_at: (u16, u16),
		_rect: Rect,
		mouse: Mouse,
	) -> Flow {
		match mouse {
			Mouse::Click
				if let HitTag::Chip(index) = tag
					&& usize::from(index) < self.state.options.len() =>
			{
				self.state.idx = index;
				Flow::Consumed
			},
			Mouse::Click
			| Mouse::RightClick
			| Mouse::MiddleClick
			| Mouse::Move
			| Mouse::Drag
			| Mouse::Release
			| Mouse::WheelUp
			| Mouse::WheelDown
			| Mouse::WheelLeft
			| Mouse::WheelRight => Flow::Skip,
		}
	}

	fn value(&self, out: &mut serde_json::Map<String, serde_json::Value>) {
		let Some(id) = self.props.id() else {
			return;
		};
		let value = self
			.state
			.options
			.get(usize::from(self.state.idx))
			.map_or(serde_json::Value::Null, |option| serde_json::Value::String(option.to_string()));
		out.insert(id.to_string(), value);
	}
}

pub(super) fn pill(
	frame: &mut Frame,
	x: u16,
	y: u16,
	label: &str,
	background: Color,
	foreground: Color,
	caps: (&str, &str),
	highlight: bool,
) -> u16 {
	let background = if highlight {
		brighten(background)
	} else {
		background
	};
	let cap = Style::new().fg(background);
	let body = Style::new().fg(foreground).bg(background).bold();
	let mut x = frame.put(x, y, caps.0, cap);
	x = frame.put(x, y, label, body);
	frame.put(x, y, caps.1, cap)
}

fn brighten(color: Color) -> Color {
	match color {
		Color::Rgb(red, green, blue) => Color::Rgb(
			red.saturating_add((255 - u16::from(red)) as u8 / 5),
			green.saturating_add((255 - u16::from(green)) as u8 / 5),
			blue.saturating_add((255 - u16::from(blue)) as u8 / 5),
		),
		other => other,
	}
}

const fn base(theme: &Theme) -> Style {
	Style::new().fg(theme.fg)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Frame, Size, test_support::frame_row_text};

	fn event_ctx(ctx: &UiContext) -> EventCtx<'_> {
		EventCtx::new(ctx, 40, 1)
	}

	#[test]
	fn left_and_right_cycle_and_export_value() {
		let mut radio = Radio::new()
			.with(Prop::Id, "mode")
			.with(Prop::Options, "one two three");
		let ctx = UiContext::default();
		assert_eq!(radio.key(&mut event_ctx(&ctx), Key::Left), Flow::Consumed);
		let mut values = serde_json::Map::new();
		radio.value(&mut values);
		assert_eq!(values["mode"], serde_json::json!("three"));
		assert_eq!(radio.key(&mut event_ctx(&ctx), Key::Right), Flow::Consumed);
	}

	#[test]
	fn paint_draws_chips_and_hits() {
		let mut radio = Radio::new().with(Prop::Options, "one two");
		let ctx = UiContext::default();
		let mut frame = Frame::new(Size::new(32, 1));
		let mut hits = Vec::new();
		let slot = radio.slot();
		let mut wakes = Vec::new();
		let mut pc = PaintCtx::new(&mut frame, &ctx, &mut hits, &mut wakes);
		pc.focus = Some(slot);
		radio.paint(&mut pc, Rect::new(0, 0, 32, 1));
		assert!(frame_row_text(&frame, 0).contains("one"));
		assert_eq!(hits.len(), 2);
	}
}
