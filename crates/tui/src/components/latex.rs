use omp_core::Str;

use super::text::{append, paint_rich, truncate_rich};
use crate::{
	component::{Component, MemoKey, PaintCtx, Slot, next_slot},
	frame::Rect,
	props::{Prop, PropValue, Props},
	rich::{Measure, RichText},
};

/// Typeset mathematical text backing the `<latex>` markup tag.
pub struct Latex {
	props:        Props,
	slot:         Slot,
	text:         Str,
	rich:         RichText,
	version:      u64,
	cached_width: u16,
	cached:       Option<MemoKey>,
}

impl Latex {
	/// Creates an empty LaTeX block.
	pub fn new() -> Self {
		Self {
			props:        Props::new(),
			slot:         next_slot(),
			text:         Str::default(),
			rich:         RichText::default(),
			version:      1,
			cached_width: 0,
			cached:       None,
		}
	}

	/// Sets one LaTeX property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self.version = self.version.wrapping_add(1);
		self
	}

	/// Sets one LaTeX property from a string.
	pub fn with_str(self, prop: Prop, value: &str) -> Self {
		self.with(prop, value)
	}

	/// Appends LaTeX source text.
	pub fn text(mut self, text: impl Into<Str>) -> Self {
		append(&mut self.text, text.into());
		self.version = self.version.wrapping_add(1);
		self
	}

	fn render(&mut self, ctx: &crate::UiContext, width: u16) {
		let width = width.max(1);
		let key = MemoKey::new(self.version, ctx);
		if self.cached_width == width && self.cached == Some(key) {
			return;
		}
		let style = self.props.style(&ctx.theme);
		self.rich.clear();
		if !crate::latex::latex_block(&self.text, style, &mut self.rich) {
			crate::latex::latex_inline(&self.text, style, &mut self.rich);
		}
		truncate_rich(&mut self.rich, width, style, self.props.truncate());
		self.cached_width = width;
		self.cached = Some(key);
	}
}

impl Default for Latex {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Latex {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn measure(&mut self, ctx: &crate::UiContext) -> (u16, u16) {
		let style = self.props.style(&ctx.theme);
		let mut measure = Measure::default();
		if !crate::latex::latex_block(&self.text, style, &mut measure) {
			crate::latex::latex_inline(&self.text, style, &mut measure);
		}
		let width = measure.widest.max(1);
		(width, width)
	}

	fn height(&mut self, ctx: &crate::UiContext, width: u16) -> u16 {
		self.render(ctx, width);
		RichText::rows(&self.rich)
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		self.render(pc.ctx, rect.width);
		paint_rich(pc, rect, &self.rich, self.props.align());
	}

	fn set_text(&mut self, _ctx: &crate::UiContext, text: Str) -> bool {
		if self.text == text {
			return false;
		}
		self.text = text;
		self.version = self.version.wrapping_add(1);
		true
	}
}
