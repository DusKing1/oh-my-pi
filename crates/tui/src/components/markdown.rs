use omp_core::Str;

use super::text::{append, paint_rich, truncate_rich};
use crate::{
	component::{Cached, Component, IntoChildren, MemoKey, PaintCtx, Slot, next_slot},
	frame::Rect,
	markdown::MdTheme,
	props::{Prop, PropValue, Props},
	rich::{Measure, RichText},
};

/// Rendered Markdown content backing the `<markdown>` markup tag.
pub struct Markdown {
	props:        Props,
	slot:         Slot,
	text:         Str,
	source:       Str,
	rich:         RichText,
	embedded:     Vec<Cached>,
	version:      u64,
	cached_width: u16,
	cached:       Option<MemoKey>,
}

impl Markdown {
	/// Creates an empty Markdown block.
	pub fn new() -> Self {
		Self {
			props:        Props::new(),
			slot:         next_slot(),
			text:         Str::default(),
			source:       Str::default(),
			rich:         RichText::default(),
			embedded:     Vec::new(),
			version:      1,
			cached_width: 0,
			cached:       None,
		}
	}

	/// Creates a Markdown block containing the supplied source.
	pub fn text_of(text: impl Into<Str>) -> Self {
		Self::new().text(text)
	}

	/// Sets one Markdown property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self.version = self.version.wrapping_add(1);
		self
	}

	/// Sets one Markdown property from a string.
	pub fn with_str(self, prop: Prop, value: &str) -> Self {
		self.with(prop, value)
	}

	/// Appends Markdown source text.
	pub fn text(mut self, text: impl Into<Str>) -> Self {
		let text = text.into();
		append(&mut self.source, text.clone());
		append(&mut self.text, text);
		self.version = self.version.wrapping_add(1);
		self
	}

	/// Appends embedded components referenced by the Markdown source.
	pub fn child(mut self, child: impl IntoChildren) -> Self {
		child.extend_children(&mut self.embedded);
		self
	}

	fn theme(&self, ctx: &crate::UiContext) -> MdTheme {
		MdTheme::from_context(ctx).cascade(self.props.style(&ctx.theme))
	}

	fn render(&mut self, ctx: &crate::UiContext, width: u16) {
		let width = width.max(1);
		let key = MemoKey::new(self.version, ctx);
		if self.cached_width == width && self.cached == Some(key) {
			return;
		}
		let theme = self.theme(ctx);
		let style = self.props.style(&ctx.theme);
		self.rich.clear();
		crate::markdown::render(&self.text, width, &theme, &mut self.rich);
		truncate_rich(&mut self.rich, width, style, self.props.truncate());
		self.cached_width = width;
		self.cached = Some(key);
	}
}

impl Default for Markdown {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Markdown {
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
		&self.embedded
	}

	fn children_mut(&mut self) -> &mut [Cached] {
		&mut self.embedded
	}

	fn measure(&mut self, ctx: &crate::UiContext) -> (u16, u16) {
		let theme = self.theme(ctx);
		let mut natural = Measure::default();
		crate::markdown::render(&self.text, u16::MAX, &theme, &mut natural);
		let mut min = natural.widest.clamp(1, 12);
		let mut nat = natural.widest.max(min);
		for child in &mut self.embedded {
			if child.visible {
				let (child_min, child_nat) = child.measure(ctx);
				min = min.max(child_min);
				nat = nat.max(child_nat);
			}
		}
		(min, nat)
	}

	fn height(&mut self, ctx: &crate::UiContext, width: u16) -> u16 {
		self.render(ctx, width);
		let mut height = if self.text.is_empty() {
			0
		} else {
			RichText::rows(&self.rich)
		};
		let mut placed = !self.text.is_empty();
		for child in &mut self.embedded {
			if !child.visible {
				continue;
			}
			if placed {
				height = height.saturating_add(1);
			}
			height = height.saturating_add(child.height(ctx, width));
			placed = true;
		}
		height
	}

	fn place(&mut self, ctx: &crate::UiContext, content: Rect) {
		self.render(ctx, content.width);
		let mut cursor = content.y;
		let mut placed = if self.text.is_empty() {
			false
		} else {
			cursor = cursor.saturating_add(RichText::rows(&self.rich));
			true
		};
		for child in &mut self.embedded {
			if !child.visible {
				continue;
			}
			if placed {
				cursor = cursor.saturating_add(1);
			}
			let height = child.height(ctx, content.width);
			child.place(ctx, Rect::new(content.x, cursor, content.width, height));
			cursor = cursor.saturating_add(height);
			placed = true;
		}
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		if !self.text.is_empty() {
			self.render(pc.ctx, rect.width);
			let own = Rect::new(rect.x, rect.y, rect.width, RichText::rows(&self.rich));
			paint_rich(pc, own, &self.rich, self.props.align());
		}
		for child in &mut self.embedded {
			if child.visible {
				child.paint(pc);
			}
		}
	}

	fn set_text(&mut self, ctx: &crate::UiContext, text: Str) -> bool {
		if self.source == text {
			return false;
		}
		self.source = text.clone();
		if crate::markup::md_embeds_markup(&text) {
			if let Ok(children) = crate::markup::parse_md_fragment_inheriting(&text, ctx, &self.props)
			{
				self.text = Str::default();
				self.embedded = children;
			} else {
				self.text = text;
				self.embedded.clear();
			}
		} else {
			self.text = text;
			self.embedded.clear();
		}
		self.version = self.version.wrapping_add(1);
		true
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::context::UiContext;

	#[test]
	fn set_text_regrafts_static_markup_in_document_order() {
		let ctx = UiContext::default();
		let mut markdown = Markdown::text_of("old");
		assert!(markdown.set_text(&ctx, Str::new("before\n<box><text>inside</text></box>\nafter"),));
		assert!(markdown.text.is_empty());
		assert_eq!(markdown.embedded.len(), 3);
	}

	#[test]
	fn set_text_degrades_rejected_dynamic_markup_to_literal_text() {
		let ctx = UiContext::default();
		for source in ["<input/>", "<box id=duplicate/>", "<box when=\"x == y\"/>", "</md>"] {
			let mut markdown = Markdown::text_of("old");
			assert!(markdown.set_text(&ctx, Str::new(source)));
			assert_eq!(markdown.text, source);
			assert!(markdown.embedded.is_empty());
		}
	}
}
