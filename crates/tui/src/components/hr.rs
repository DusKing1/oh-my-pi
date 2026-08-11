use crate::{
	component::{Component, PaintCtx, Slot, next_slot},
	context::UiContext,
	frame::Rect,
	markup::Border,
	props::{Prop, PropValue, Props},
};

/// A horizontal or vertical divider backing the `<hr>` markup tag.
pub struct Hr {
	props: Props,
	slot:  Slot,
	bar:   String,
}

impl Hr {
	/// Creates a divider with default styling.
	pub fn new() -> Self {
		Self { props: Props::new(), slot: next_slot(), bar: String::new() }
	}

	/// Sets one divider property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one divider property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self
	}
}

impl Default for Hr {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Hr {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn paints_border(&self) -> bool {
		false
	}

	fn stretch_in_row(&self) -> bool {
		true
	}

	fn measure(&mut self, _ctx: &UiContext) -> (u16, u16) {
		if self.props.flag(Prop::Vertical) {
			(1, 1)
		} else {
			(1, 4)
		}
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		1
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		let border = self.props.border().unwrap_or(Border::Square);
		let (.., horizontal, vertical) = pc.ctx.charset.border(border);
		let style = self.props.style(&pc.ctx.theme);
		// An unstyled rule takes the theme's border tone; `fg=`/`bc=` win.
		let line = self.props.edge(&pc.ctx.theme).map_or_else(
			|| {
				if !self.props.contains(Prop::Fg) {
					style.fg(pc.ctx.theme.border)
				} else {
					style.dim()
				}
			},
			|color| style.fg(color),
		);
		if self.props.flag(Prop::Vertical) {
			self.bar.clear();
			repeated_char(&mut self.bar, vertical, 1);
			for row in 0..rect.height {
				let y = rect.y.saturating_add(row);
				if y >= pc.clip {
					break;
				}
				pc.frame.put(rect.x, y, &self.bar, line);
			}
			return;
		}
		if rect.y >= pc.clip {
			return;
		}
		self.bar.clear();
		repeated_char(&mut self.bar, horizontal, usize::from(rect.width));
		pc.frame.put(rect.x, rect.y, &self.bar, line);
		if let Some(title) = self.props.title() {
			pc.frame.put(rect.x.saturating_add(2), rect.y, " ", style);
			let end = pc
				.frame
				.put(rect.x.saturating_add(3), rect.y, title, style.bold());
			pc.frame.put(end, rect.y, " ", style);
		}
	}
}

/// Flexible blank space backing the `<spacer>` markup tag.
pub struct Spacer {
	props: Props,
	slot:  Slot,
}

impl Spacer {
	/// Creates an empty spacer.
	pub fn new() -> Self {
		Self { props: Props::new(), slot: next_slot() }
	}

	/// Sets one spacer property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one spacer property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self
	}
}

impl Default for Spacer {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Spacer {
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
		(1, 4)
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		1
	}

	fn paint(&mut self, _pc: &mut PaintCtx<'_>, _rect: Rect) {}
}

fn repeated_char(output: &mut String, character: char, count: usize) {
	output.reserve(count.saturating_mul(character.len_utf8()));
	output.extend(std::iter::repeat_n(character, count));
}

#[cfg(test)]
mod tests {
	use super::Hr;
	use crate::{
		component::{Cached, Component, PaintCtx},
		context::UiContext,
		frame::{Frame, Rect, Size},
		props::Prop,
		test_support::frame_row_text,
	};

	#[test]
	fn fills_horizontal_rule_with_charset_glyph() {
		let ctx = UiContext::default();
		let mut hr = Cached::new(Box::new(Hr::new()));
		assert_eq!(hr.measure(&ctx), (1, 4));
		hr.place(&ctx, Rect::new(0, 0, 6, 1));
		let mut frame = Frame::new(Size::new(6, 1));
		let mut hits = Vec::new();
		hr.paint(&mut PaintCtx::new(&mut frame, &ctx, &mut hits, &mut Vec::new()));
		assert_eq!(frame_row_text(&frame, 0), "──────");
	}

	#[test]
	fn vertical_rule_uses_one_column_and_fills_height() {
		let ctx = UiContext::default();
		let mut hr = Hr::new().with(Prop::Vertical, true);
		assert_eq!(hr.measure(&ctx), (1, 1));
		let mut frame = Frame::new(Size::new(1, 3));
		let mut hits = Vec::new();
		hr.paint(
			&mut PaintCtx::new(&mut frame, &ctx, &mut hits, &mut Vec::new()),
			Rect::new(0, 0, 1, 3),
		);
		assert_eq!(frame_row_text(&frame, 0), "│");
		assert_eq!(frame_row_text(&frame, 1), "│");
		assert_eq!(frame_row_text(&frame, 2), "│");
	}
}
