use std::fmt::Write as _;

use crate::{
	component::{Component, PaintCtx, Slot, next_slot},
	context::UiContext,
	frame::{Rect, Style},
	props::{Prop, PropValue, Props},
	rich::cell_width,
};

/// A determinate progress bar backing the `<progress>` markup tag.
pub struct Progress {
	props:   Props,
	slot:    Slot,
	scratch: String,
}

impl Progress {
	/// Creates a progress bar at its default value.
	pub fn new() -> Self {
		Self { props: Props::new(), slot: next_slot(), scratch: String::new() }
	}

	/// Sets one progress-bar property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one progress-bar property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self
	}

	fn amount(&self) -> (u64, u64) {
		let number = |prop| match self.props.get(prop) {
			Some(PropValue::U16(value)) => Some(u64::from(*value)),
			Some(PropValue::I64(value)) => Some((*value).max(0) as u64),
			Some(PropValue::F32(value)) => Some(value.max(0.0) as u64),
			Some(PropValue::Str(value)) => value.parse().ok(),
			_ => None,
		};
		let maximum = number(Prop::Max).unwrap_or(100).max(1);
		(number(Prop::Value).unwrap_or(0).min(maximum), maximum)
	}
}

impl Default for Progress {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Progress {
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
		(16, 40)
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		1
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		if rect.y >= pc.clip {
			return;
		}
		let (value, maximum) = self.amount();
		let percent = value.saturating_mul(100) / maximum;
		let mut x = rect.x;
		if let Some(label) = self.props.str_of(Prop::Label) {
			x = pc
				.frame
				.put(x, rect.y, label, Style::new().fg(pc.ctx.theme.fg));
			x = pc
				.frame
				.put(x, rect.y, " ", Style::new().fg(pc.ctx.theme.fg));
		}
		self.scratch.clear();
		let _ = write!(self.scratch, " {percent}%");
		let bar_width = rect
			.x
			.saturating_add(rect.width)
			.saturating_sub(x)
			.saturating_sub(cell_width(&self.scratch))
			.max(4);
		let fill =
			u16::try_from(u64::from(bar_width).saturating_mul(value) / maximum).unwrap_or(bar_width);
		for index in 0..bar_width {
			let (glyph, style) = if index < fill {
				(pc.ctx.charset.progress().0, Style::new().fg(pc.ctx.theme.accent))
			} else {
				(pc.ctx.charset.progress().1, Style::new().fg(pc.ctx.theme.muted))
			};
			x = pc.frame.put(x, rect.y, glyph, style);
		}
		pc.frame
			.put(x, rect.y, &self.scratch, Style::new().fg(pc.ctx.theme.muted));
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		component::PaintCtx,
		frame::{Frame, Size},
		test_support::frame_row_text,
	};

	#[test]
	fn fill_uses_value_over_maximum() {
		let mut progress = Progress::new()
			.with(Prop::Value, "3")
			.with(Prop::Max, 4_u16);
		let ctx = UiContext::default();
		let mut frame = Frame::new(Size::new(12, 1));
		let mut hits = Vec::new();
		progress.paint(
			&mut PaintCtx::new(&mut frame, &ctx, &mut hits, &mut Vec::new()),
			Rect::new(0, 0, 12, 1),
		);
		let row = frame_row_text(&frame, 0);
		assert!(row.starts_with("██████░░"), "{row:?}");
		assert!(row.ends_with(" 75%"), "{row:?}");
	}
}
