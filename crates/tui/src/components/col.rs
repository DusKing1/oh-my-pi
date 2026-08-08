use super::layout::{stack_height, stack_measure, stack_place};
use crate::{
	component::{Cached, Component, IntoChildren, PaintCtx, ResizeTail, Slot, next_slot},
	context::UiContext,
	frame::Rect,
	props::{Prop, PropValue, Props},
};

/// A vertical child stack backing the `<col>` markup tag.
pub struct Col {
	props:    Props,
	slot:     Slot,
	children: Vec<Cached>,
}

impl Col {
	/// Creates an empty column.
	pub fn new() -> Self {
		Self { props: Props::new(), slot: next_slot(), children: Vec::new() }
	}

	/// Sets one column property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one column property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Appends child components to the column.
	pub fn child(mut self, children: impl IntoChildren) -> Self {
		children.extend_children(&mut self.children);
		self
	}
}

impl Default for Col {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Col {
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
		&self.children
	}

	fn children_mut(&mut self) -> &mut [Cached] {
		&mut self.children
	}

	fn resize_tail(&mut self) -> Option<ResizeTail<'_>> {
		// Only a chrome-neutral flow decomposes into drag-frame tails;
		// styled columns render whole so borders, backgrounds, and
		// alignment keep full fidelity during a resize.
		if self.props.border().is_some()
			|| self.props.valign().is_some()
			|| self.props.get(Prop::Bg).is_some()
			|| self.props.get(Prop::On).is_some()
		{
			return None;
		}
		Some(ResizeTail { children: &mut self.children, gap: self.props.gap() })
	}

	fn measure(&mut self, ctx: &UiContext) -> (u16, u16) {
		stack_measure(ctx, &mut self.children)
	}

	fn height(&mut self, ctx: &UiContext, width: u16) -> u16 {
		stack_height(ctx, &mut self.children, width, self.props.gap())
	}

	fn place(&mut self, ctx: &UiContext, content: Rect) {
		stack_place(
			ctx,
			&mut self.children,
			content,
			self.props.gap(),
			self.props.valign(),
			self.props.align(),
		);
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, _rect: Rect) {
		for child in self.children.iter_mut().filter(|child| child.visible) {
			child.paint(pc);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::Col;
	use crate::{
		component::{Cached, PaintCtx},
		components::TextLeaf,
		context::UiContext,
		frame::{Frame, Rect, Size},
		props::Prop,
		test_support::frame_row_text,
	};

	#[test]
	fn stacks_children_with_gap() {
		let ctx = UiContext::default();
		let mut root = Cached::new(Box::new(
			Col::new()
				.with(Prop::Gap, 1_u16)
				.child(TextLeaf::new().text("first"))
				.child(TextLeaf::new().text("second")),
		));
		let height = root.height(&ctx, 12);
		assert_eq!(height, 3);
		root.place(&ctx, Rect::new(0, 0, 12, height));
		let mut frame = Frame::new(Size::new(12, height));
		let mut hits = Vec::new();
		root.paint(&mut PaintCtx::new(&mut frame, &ctx, &mut hits, &mut Vec::new()));
		assert_eq!(frame_row_text(&frame, 0), "first");
		assert_eq!(frame_row_text(&frame, 1), "");
		assert_eq!(frame_row_text(&frame, 2), "second");
	}
}
