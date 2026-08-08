use omp_core::SmolStr;
use smallvec::SmallVec;

use super::Col;
use crate::{
	component::{
		Cached, Component, EventCtx, Flow, Hit, HitTag, IntoChildren, PaintCtx, Slot, next_slot,
	},
	context::UiContext,
	frame::{Color, Rect, Style},
	input::{Key, Mouse},
	props::{Prop, PropValue, Props},
	rich::cell_width,
};

#[derive(Default)]
struct TabsState {
	titles: SmallVec<SmolStr, 6>,
	panes:  Vec<Cached>,
	idx:    u16,
	spans:  SmallVec<(u16, u16), 6>,
	rule:   String,
}

/// A switchable pane set backing the `<tabs>` markup tag.
pub struct Tabs {
	props: Props,
	slot:  Slot,
	state: TabsState,
}

impl Tabs {
	/// Creates an empty tab set.
	pub fn new() -> Self {
		Self { props: Props::new(), slot: next_slot(), state: TabsState::default() }
	}

	/// Sets one tab-set property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one tab-set property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Appends an untitled pane.
	pub fn child(self, children: impl IntoChildren) -> Self {
		self.pane("tab", children)
	}

	/// Appends a pane with the supplied title.
	pub fn pane(mut self, title: impl Into<SmolStr>, children: impl IntoChildren) -> Self {
		let mut pane = Vec::new();
		children.extend_children(&mut pane);
		let pane = if pane.len() == 1 {
			pane.pop().expect("one pane child")
		} else {
			Cached::new(Box::new(Col::new().child(pane)))
		};
		self.state.titles.push(title.into());
		self.state.panes.push(pane);
		self
	}

	fn active(&self) -> Option<usize> {
		let index = usize::from(self.state.idx);
		(index < self.state.panes.len()).then_some(index)
	}
}

impl Default for Tabs {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Tabs {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn children(&self) -> &[Cached] {
		&self.state.panes
	}

	fn children_mut(&mut self) -> &mut [Cached] {
		&mut self.state.panes
	}

	fn measure(&mut self, ctx: &UiContext) -> (u16, u16) {
		let bar = self
			.state
			.titles
			.iter()
			.fold(2u16, |width, title| width.saturating_add(cell_width(title).saturating_add(4)));
		let mut nat = bar;
		for pane in &mut self.state.panes {
			nat = nat.max(pane.measure(ctx).1);
		}
		(bar.min(24), nat)
	}

	fn height(&mut self, ctx: &UiContext, width: u16) -> u16 {
		let pane_height = self
			.active()
			.and_then(|index| self.state.panes.get_mut(index))
			.filter(|pane| pane.visible)
			.map_or(0, |pane| pane.height(ctx, width));
		pane_height.saturating_add(2)
	}

	fn place(&mut self, ctx: &UiContext, content: Rect) {
		let Some(index) = self.active() else {
			return;
		};
		let pane = &mut self.state.panes[index];
		if !pane.visible {
			return;
		}
		let width = content.width;
		let height = pane.height(ctx, width);
		pane.place(ctx, Rect::new(content.x, content.y.saturating_add(2), width, height));
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		let focused = pc.focus == Some(self.slot);
		let hover_chip = match pc.hover {
			Some((slot, HitTag::Chip(index))) if slot == self.slot => Some(index),
			_ => None,
		};
		self.state.spans.clear();
		if rect.y < pc.clip {
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
			for (index, title) in self.state.titles.iter().enumerate() {
				let start = x.saturating_sub(rect.x);
				let index = index as u16;
				let active = index == self.state.idx;
				let hovered = hover_chip == Some(index);
				if active {
					x = pill(
						pc.frame,
						x,
						rect.y,
						title,
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
					x = pc
						.frame
						.put(x, rect.y, " ", Style::new().fg(pc.ctx.theme.fg));
					x = pc.frame.put(x, rect.y, title, style);
					x = pc
						.frame
						.put(x, rect.y, " ", Style::new().fg(pc.ctx.theme.fg));
				}
				let end = x.saturating_sub(rect.x);
				self.state.spans.push((start, end));
				pc.hits.push(Hit {
					rect: Rect::new(rect.x.saturating_add(start), rect.y, end.saturating_sub(start), 1),
					slot: self.slot,
					tag:  HitTag::Chip(index),
				});
				x = pc
					.frame
					.put(x, rect.y, "  ", Style::new().fg(pc.ctx.theme.fg));
			}
		}
		if rect.y.saturating_add(1) < pc.clip {
			self.state.rule.clear();
			for _ in 0..rect.width {
				self.state.rule.push(pc.ctx.charset.rule());
			}
			pc.frame.put(
				rect.x,
				rect.y.saturating_add(1),
				&self.state.rule,
				Style::new().fg(pc.ctx.theme.muted),
			);
		}
		let Some(index) = self.active() else {
			return;
		};
		let pane = &mut self.state.panes[index];
		if pane.visible {
			pane.paint(pc);
		}
	}

	fn focusable(&self) -> bool {
		true
	}

	fn ring(&self, out: &mut Vec<Slot>) {
		out.push(self.slot);
		if let Some(index) = self.active()
			&& let Some(pane) = self.state.panes.get(index)
			&& pane.visible
		{
			pane.comp().ring(out);
		}
	}

	fn key(&mut self, _ec: &mut EventCtx<'_>, key: Key) -> Flow {
		let len = self.state.titles.len() as u16;
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
			Mouse::Click => {
				let HitTag::Chip(index) = tag else {
					return Flow::Skip;
				};
				if usize::from(index) >= self.state.titles.len() {
					return Flow::Skip;
				}
				self.state.idx = index;
				Flow::Consumed
			},
			Mouse::RightClick
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
			.titles
			.get(usize::from(self.state.idx))
			.map_or(serde_json::Value::Null, |title| serde_json::Value::String(title.to_string()));
		out.insert(id.to_string(), value);
	}
}

fn pill(
	frame: &mut crate::Frame,
	x: u16,
	y: u16,
	label: &str,
	bg: Color,
	fg: Color,
	caps: (&str, &str),
	highlight: bool,
) -> u16 {
	let bg = if highlight { brighten(bg) } else { bg };
	let cap = Style::new().fg(bg);
	let body = Style::new().fg(fg).bg(bg).bold();
	let mut x = frame.put(x, y, caps.0, cap);
	x = frame.put(x, y, label, body);
	frame.put(x, y, caps.1, cap)
}

fn brighten(color: Color) -> Color {
	match color {
		Color::Rgb(r, g, b) => Color::Rgb(
			r.saturating_add((255 - u16::from(r)) as u8 / 5),
			g.saturating_add((255 - u16::from(g)) as u8 / 5),
			b.saturating_add((255 - u16::from(b)) as u8 / 5),
		),
		other => other,
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Frame, Size, component::Component, components::Pre, test_support::frame_row_text};

	struct FocusProbe {
		props: Props,
		slot:  Slot,
		text:  &'static str,
	}

	impl FocusProbe {
		fn new(text: &'static str) -> Self {
			Self { props: Props::new(), slot: next_slot(), text }
		}
	}

	impl Component for FocusProbe {
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
			(3, 3)
		}

		fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
			1
		}

		fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
			pc.frame.put(rect.x, rect.y, self.text, Style::default());
		}

		fn focusable(&self) -> bool {
			true
		}
	}

	#[test]
	fn switching_panes_changes_paint_value_and_ring() {
		let ctx = UiContext::default();
		let first = FocusProbe::new("one");
		let first_slot = first.slot;
		let second = FocusProbe::new("two");
		let second_slot = second.slot;
		let mut tabs = Tabs::new()
			.with(Prop::Id, "tab-id")
			.pane("First", first)
			.pane("Second", second);
		let tabs_slot = tabs.slot;
		tabs.place(&ctx, Rect::new(0, 0, 24, 3));
		let mut ring = Vec::new();
		tabs.ring(&mut ring);
		assert_eq!(ring, vec![tabs_slot, first_slot]);

		let mut frame = Frame::new(Size::new(24, 3));
		let mut hits = Vec::new();
		let mut wakes = Vec::new();
		let mut pc = PaintCtx::new(&mut frame, &ctx, &mut hits, &mut wakes);
		tabs.paint(&mut pc, Rect::new(0, 0, 24, 3));
		assert_eq!(frame_row_text(pc.frame, 2), "one");

		let mut ec = EventCtx::new(&ctx, 24, 3);
		assert_eq!(tabs.key(&mut ec, Key::Right), Flow::Consumed);
		tabs.place(&ctx, Rect::new(0, 0, 24, 3));
		pc.frame.clear(Style::default());
		pc.hits.clear();
		tabs.paint(&mut pc, Rect::new(0, 0, 24, 3));
		assert_eq!(frame_row_text(pc.frame, 2), "two");
		ring.clear();
		tabs.ring(&mut ring);
		assert_eq!(ring, vec![tabs_slot, second_slot]);
		let mut values = serde_json::Map::new();
		tabs.value(&mut values);
		assert_eq!(values["tab-id"], serde_json::json!("Second"));
	}

	#[test]
	fn pane_accepts_multiple_children() {
		let tabs = Tabs::new().pane("many", vec![Pre::new().text("a"), Pre::new().text("b")]);
		assert_eq!(tabs.state.panes.len(), 1);
		assert_eq!(tabs.state.panes[0].comp().children().len(), 2);
	}
}
