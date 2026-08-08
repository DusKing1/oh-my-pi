use omp_core::SmolStr;

use super::radio::pill;
use crate::{
	component::{Component, EventCtx, Flow, Hit, HitTag, PaintCtx, Slot, next_slot},
	context::UiContext,
	frame::Rect,
	input::{Key, Mouse, UiEvent},
	props::{Prop, PropValue, Props},
	rich::cell_width,
};

#[derive(Default)]
struct ButtonState {
	armed: bool,
}

/// A pressable action button.
pub struct Button {
	props: Props,
	slot:  Slot,
	state: ButtonState,
	label: SmolStr,
}

impl Button {
	/// Creates an unlabeled button.
	pub fn new() -> Self {
		Self {
			props: Props::new(),
			slot:  next_slot(),
			state: ButtonState::default(),
			label: SmolStr::default(),
		}
	}

	/// Sets one button property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one button property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets the button's text child.
	pub fn child(mut self, label: impl Into<SmolStr>) -> Self {
		let label = label.into();
		if self.label.is_empty() {
			self.label = label;
		} else {
			self.label = SmolStr::from(format!("{}{}", self.label, label));
		}
		self
	}

	fn label(&self) -> &str {
		if !self.label.is_empty() {
			&self.label
		} else if let Some(label) = self.props.str_of(Prop::Label) {
			label
		} else if let Some(id) = self.props.id() {
			id
		} else {
			"ok"
		}
	}

	fn press(&mut self) -> Flow {
		if self.props.flag(Prop::Confirm) && !self.state.armed {
			self.state.armed = true;
			return Flow::Consumed;
		}
		self.state.armed = false;
		if self.props.flag(Prop::Cancel) {
			Flow::Event(UiEvent::Cancel)
		} else if self.props.flag(Prop::Submit) {
			Flow::Event(UiEvent::Submit)
		} else if let Some(id) = self.props.id() {
			Flow::Event(UiEvent::Pressed(id.clone()))
		} else {
			Flow::Event(UiEvent::Submit)
		}
	}
}

impl Default for Button {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Button {
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
		let width = cell_width(self.label()).saturating_add(4);
		(width, width)
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		1
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		if rect.y >= pc.clip {
			return;
		}
		pc.hits
			.push(Hit { rect, slot: self.slot, tag: HitTag::Press });
		let focused = pc.focus == Some(self.slot);
		let hovered = matches!(pc.hover, Some((slot, _)) if slot == self.slot);
		let accent = self.props.flag(Prop::Accent) || self.props.flag(Prop::Submit);
		let (text, background, foreground) = if self.state.armed {
			("sure?", pc.ctx.theme.warn, pc.ctx.theme.contrast)
		} else if accent {
			(self.label(), pc.ctx.theme.accent, pc.ctx.theme.contrast)
		} else {
			(self.label(), pc.ctx.theme.surface, pc.ctx.theme.fg)
		};
		pill(
			pc.frame,
			rect.x,
			rect.y,
			text,
			background,
			foreground,
			pc.ctx.charset.pill_caps(),
			focused || hovered,
		);
	}

	fn focusable(&self) -> bool {
		true
	}

	fn key(&mut self, _ec: &mut EventCtx<'_>, key: Key) -> Flow {
		match key {
			Key::Enter | Key::Space => self.press(),
			_ => {
				self.state.armed = false;
				Flow::Skip
			},
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
			Mouse::Click if tag == HitTag::Press => self.press(),
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
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Frame, Size, test_support::frame_row_text};

	fn event_ctx(ctx: &UiContext) -> EventCtx<'_> {
		EventCtx::new(ctx, 16, 1)
	}

	#[test]
	fn enter_emits_matching_application_event() {
		let ctx = UiContext::default();
		let mut submit = Button::new().with(Prop::Submit, true).child("Go");
		assert_eq!(submit.key(&mut event_ctx(&ctx), Key::Enter), Flow::Event(UiEvent::Submit));
		let mut plain = Button::new().with(Prop::Id, "again").child("Again");
		assert_eq!(
			plain.key(&mut event_ctx(&ctx), Key::Enter),
			Flow::Event(UiEvent::Pressed(SmolStr::from("again")))
		);
	}

	#[test]
	fn confirm_arms_before_emitting() {
		let ctx = UiContext::default();
		let mut button = Button::new().with(Prop::Confirm, true);
		assert_eq!(button.key(&mut event_ctx(&ctx), Key::Enter), Flow::Consumed);
		assert_eq!(button.key(&mut event_ctx(&ctx), Key::Enter), Flow::Event(UiEvent::Submit));
	}

	#[test]
	fn paint_draws_label_and_press_hit() {
		let mut button = Button::new().child("Continue");
		let ctx = UiContext::default();
		let mut frame = Frame::new(Size::new(24, 1));
		let mut hits = Vec::new();
		let slot = button.slot();
		let mut wakes = Vec::new();
		let mut pc = PaintCtx::new(&mut frame, &ctx, &mut hits, &mut wakes);
		pc.focus = Some(slot);
		button.paint(&mut pc, Rect::new(0, 0, 24, 1));
		assert!(frame_row_text(&frame, 0).contains("Continue"));
		assert_eq!(hits[0].tag, HitTag::Press);
	}
}
