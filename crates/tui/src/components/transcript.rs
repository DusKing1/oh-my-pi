use super::layout::{stack_height, stack_measure, stack_place};
use crate::{
	component::{Cached, Component, IntoChildren, PaintCtx, ResizeTail, Slot, next_slot},
	context::UiContext,
	frame::Rect,
	props::{Prop, PropValue, Props},
};

/// An inline `TranscriptView` component for a growing sequence of cached child
/// components.
pub struct TranscriptView {
	props:    Props,
	slot:     Slot,
	children: Vec<Cached>,
}

impl TranscriptView {
	/// Creates an empty transcript view.
	pub fn new() -> Self {
		Self { props: Props::new(), slot: next_slot(), children: Vec::new() }
	}

	/// Sets one property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Appends child components to the transcript.
	pub fn push(&mut self, child: impl IntoChildren) {
		child.extend_children(&mut self.children);
	}

	/// Replaces the last child component. Does not rebuild earlier children.
	pub fn replace_tail(&mut self, child: impl IntoChildren) {
		let mut replacement = Vec::new();
		child.extend_children(&mut replacement);
		if !self.children.is_empty() {
			self.children.pop();
		}
		self.children.extend(replacement);
	}

	/// Clears the transcript.
	pub fn clear(&mut self) {
		self.children.clear();
	}
}

impl Default for TranscriptView {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for TranscriptView {
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
		if self.props.border().is_some()
			|| self.props.valign().is_some()
			|| self.props.contains(Prop::Bg)
			|| self.props.contains(Prop::On)
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
			if child.rect.y >= pc.clip {
				// Viewport-local paint: skip traversing children that start below the
				// paint region, as this is a strict vertical stack.
				break;
			}
			child.paint(pc);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::TranscriptView;
	use crate::{
		component::{Cached, Component},
		components,
		context::UiContext,
		frame::Rect,
		props::Props,
	};

	struct MockChild {
		props: Props,
		slot:  u32,
	}
	impl MockChild {
		fn new() -> Self {
			Self { props: Props::new(), slot: crate::component::next_slot() }
		}
	}
	impl Component for MockChild {
		fn props(&self) -> &Props {
			&self.props
		}

		fn props_mut(&mut self) -> &mut Props {
			&mut self.props
		}

		fn slot(&self) -> u32 {
			self.slot
		}

		fn measure(&mut self, _ctx: &UiContext) -> (u16, u16) {
			(10, 10)
		}

		fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
			1
		}

		fn place(&mut self, _ctx: &UiContext, _content: Rect) {}

		fn paint(&mut self, _pc: &mut crate::component::PaintCtx<'_>, _rect: Rect) {}
	}

	#[test]
	fn test_transcript_push_and_clear() {
		let mut view = TranscriptView::new();
		assert_eq!(view.children().len(), 0);

		view.push(Cached::new(Box::new(MockChild::new())));
		assert_eq!(view.children().len(), 1);

		view.clear();
		assert_eq!(view.children().len(), 0);
	}

	#[test]
	fn test_transcript_replace_tail() {
		let mut view = TranscriptView::new();
		let first = Cached::new(Box::new(MockChild::new()));
		let second = Cached::new(Box::new(MockChild::new()));

		let first_slot = first.comp().slot();
		view.push(first);
		view.push(second);
		assert_eq!(view.children().len(), 2);

		let replacement = Cached::new(Box::new(MockChild::new()));
		let replacement_slot = replacement.comp().slot();
		view.replace_tail(replacement);

		assert_eq!(view.children().len(), 2);
		assert_eq!(view.children()[0].comp().slot(), first_slot, "Earlier child remained stable");
		assert_eq!(view.children()[1].comp().slot(), replacement_slot, "Tail was replaced");
	}
}
