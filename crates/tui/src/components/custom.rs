use std::mem;

use omp_core::Str;

use super::layout::{stack_height, stack_measure, stack_place};
use crate::{
	component::{
		Cached, Component, EventCtx, Flow, HitTag, IntoChildren, PaintCtx, Slot, next_slot,
	},
	context::UiContext,
	frame::Rect,
	input::{Key, Mouse},
	markup::Align,
	props::{Prop, PropValue, Props},
};

/// A late-bound component backing a registry-defined markup tag.
pub struct CustomElement {
	props:    Props,
	slot:     Slot,
	name:     Str,
	children: Vec<Cached>,
	resolved: Option<Cached>,
	tried:    bool,
}

impl CustomElement {
	/// Creates an unresolved custom element for the supplied tag name.
	pub fn new(name: impl Into<Str>) -> Self {
		Self {
			props:    Props::new(),
			slot:     next_slot(),
			name:     name.into(),
			children: Vec::new(),
			resolved: None,
			tried:    false,
		}
	}

	/// Sets one custom-element property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one custom-element property from a string.
	pub fn with_str(self, prop: Prop, value: &str) -> Self {
		self.with(prop, value)
	}

	/// Appends fallback or factory-provided child content.
	pub fn child(mut self, children: impl IntoChildren) -> Self {
		children.extend_children(&mut self.children);
		self
	}

	fn resolve(&mut self, ctx: &UiContext) {
		if self.tried {
			return;
		}
		self.tried = true;
		let Some(factory) = ctx.elements.get(&self.name) else {
			return;
		};
		let component =
			factory.build(&self.name, mem::take(&mut self.props), mem::take(&mut self.children));
		self.resolved = Some(Cached::new(component));
	}
}

impl Component for CustomElement {
	fn props(&self) -> &Props {
		self
			.resolved
			.as_ref()
			.map_or(&self.props, |resolved| resolved.comp().props())
	}

	fn props_mut(&mut self) -> &mut Props {
		self
			.resolved
			.as_mut()
			.map_or(&mut self.props, |resolved| resolved.comp_mut().props_mut())
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn children(&self) -> &[Cached] {
		if let Some(resolved) = &self.resolved {
			std::slice::from_ref(resolved)
		} else {
			&self.children
		}
	}

	fn children_mut(&mut self) -> &mut [Cached] {
		if let Some(resolved) = &mut self.resolved {
			std::slice::from_mut(resolved)
		} else {
			&mut self.children
		}
	}

	fn measure(&mut self, ctx: &UiContext) -> (u16, u16) {
		self.resolve(ctx);
		if let Some(resolved) = &mut self.resolved {
			resolved.comp_mut().measure(ctx)
		} else {
			stack_measure(ctx, &mut self.children)
		}
	}

	fn height(&mut self, ctx: &UiContext, width: u16) -> u16 {
		self.resolve(ctx);
		if let Some(resolved) = &mut self.resolved {
			resolved.comp_mut().height(ctx, width)
		} else {
			stack_height(ctx, &mut self.children, width, 0)
		}
	}

	fn place(&mut self, ctx: &UiContext, content: Rect) {
		self.resolve(ctx);
		if let Some(resolved) = &mut self.resolved {
			resolved.rect = content;
			resolved.comp_mut().place(ctx, content);
		} else {
			stack_place(ctx, &mut self.children, content, 0, None, Align::Start);
		}
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		self.resolve(pc.ctx);
		if let Some(resolved) = &mut self.resolved {
			resolved.rect = rect;
			resolved.comp_mut().paint(pc, rect);
		} else {
			for child in self.children.iter_mut().filter(|child| child.visible) {
				child.paint(pc);
			}
		}
	}

	fn focusable(&self) -> bool {
		self
			.resolved
			.as_ref()
			.is_some_and(|resolved| resolved.comp().focusable())
	}

	fn enter(&mut self, forward: bool) {
		if let Some(resolved) = &mut self.resolved {
			resolved.comp_mut().enter(forward);
		}
	}

	fn ring(&self, out: &mut Vec<Slot>) {
		if let Some(resolved) = &self.resolved {
			resolved.comp().ring(out);
		} else {
			for child in self.children.iter().filter(|child| child.visible) {
				child.comp().ring(out);
			}
		}
	}

	fn key(&mut self, ec: &mut EventCtx<'_>, key: Key) -> Flow {
		self.resolve(ec.ctx);
		self
			.resolved
			.as_mut()
			.map_or(Flow::Skip, |resolved| resolved.comp_mut().key(ec, key))
	}

	fn mouse(
		&mut self,
		ec: &mut EventCtx<'_>,
		tag: HitTag,
		at: (u16, u16),
		rect: Rect,
		mouse: Mouse,
	) -> Flow {
		self.resolve(ec.ctx);
		self
			.resolved
			.as_mut()
			.map_or(Flow::Skip, |resolved| resolved.comp_mut().mouse(ec, tag, at, rect, mouse))
	}

	fn paste(&mut self, ec: &mut EventCtx<'_>, text: &str) -> Flow {
		self.resolve(ec.ctx);
		self
			.resolved
			.as_mut()
			.map_or(Flow::Skip, |resolved| resolved.comp_mut().paste(ec, text))
	}

	fn paste_raw(&mut self, ec: &mut EventCtx<'_>, text: &str) -> Flow {
		self.resolve(ec.ctx);
		self
			.resolved
			.as_mut()
			.map_or(Flow::Skip, |resolved| resolved.comp_mut().paste_raw(ec, text))
	}

	fn value(&self, out: &mut serde_json::Map<String, serde_json::Value>) {
		if let Some(resolved) = &self.resolved {
			resolved.comp().value(out);
		}
	}

	fn set_text(&mut self, ctx: &UiContext, text: Str) -> bool {
		self.resolve(ctx);
		self
			.resolved
			.as_mut()
			.is_some_and(|resolved| resolved.comp_mut().set_text(ctx, text))
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		component::{Elements, IntoComponent},
		components::TextLeaf,
		frame::{Frame, Size},
		test_support::frame_row_text,
	};

	fn paint(root: &mut Cached, ctx: &UiContext, width: u16, height: u16) -> Frame {
		root.place(ctx, Rect::new(0, 0, width, height));
		let mut frame = Frame::new(Size::new(width, height));
		let mut hits = Vec::new();
		root.paint(&mut PaintCtx::new(&mut frame, ctx, &mut hits, &mut Vec::new()));
		frame
	}

	#[test]
	fn registered_factory_resolves_and_paints() {
		let ctx = UiContext {
			elements: Elements::builder()
				.with("badge", |_name: &str, _props: Props, _children: Vec<Cached>| {
					TextLeaf::new().text("resolved").into_component()
				})
				.build(),
			..UiContext::default()
		};
		let mut root = Cached::new(CustomElement::new("badge").into_component());
		let height = root.height(&ctx, 12);
		assert_eq!(height, 1);
		assert_eq!(root.comp().children().len(), 1);
		let frame = paint(&mut root, &ctx, 12, height);
		assert_eq!(frame_row_text(&frame, 0), "resolved");
	}

	#[test]
	fn unregistered_element_is_a_zero_gap_div() {
		let ctx = UiContext::default();
		let mut root = Cached::new(
			CustomElement::new("unknown")
				.child(TextLeaf::new().text("one"))
				.child(TextLeaf::new().text("two"))
				.into_component(),
		);
		let height = root.height(&ctx, 8);
		assert_eq!(height, 2);
		let frame = paint(&mut root, &ctx, 8, height);
		assert_eq!(frame_row_text(&frame, 0), "one");
		assert_eq!(frame_row_text(&frame, 1), "two");

		let mut empty = CustomElement::new("empty");
		assert_eq!(empty.height(&ctx, 8), 0);
	}
}
