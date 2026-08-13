use omp_core::Str;

use crate::{
	component::{
		Cached, Component, PaintCtx, Slot, next_slot, IntoChildren,
	},
	context::UiContext,
	frame::{Color, Rect, Style},
	markup::{Align, Border, VAlign},
	props::{Prop, PropValue, Props},
	rich::cell_width,
	components::layout::{stack_height, stack_measure, stack_place},
};

/// Public state vocabulary for a tool call.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum ToolState {
	/// The tool is currently running/streaming output.
	#[default]
	Streaming,
	/// The tool completed successfully.
	Success,
	/// The tool failed.
	Failure,
}

/// A themed card component representing one tool call across its lifecycle.
pub struct ToolCard {
	props:    Props,
	slot:     Slot,
	state:    ToolState,
	name:     Str,
	intent:   Str,
	badge:    Str,
	folded:   bool,
	children: Vec<Cached>,
}

impl ToolCard {
	/// Creates a new tool card in the streaming state, unfolded by default.
	pub fn new() -> Self {
		Self {
			props:    Props::new(),
			slot:     next_slot(),
			state:    ToolState::Streaming,
			name:     Str::default(),
			intent:   Str::default(),
			badge:    Str::default(),
			folded:   false,
			children: Vec::new(),
		}
	}

	/// Sets one generic property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// In-place update: tool name.
	pub fn set_name(&mut self, name: impl Into<Str>) -> bool {
		let name = name.into();
		if self.name == name { return false; }
		self.name = name;
		true
	}

	/// Sets the tool name (e.g. `read`).
	pub fn name(mut self, name: impl Into<Str>) -> Self {
		self.set_name(name);
		self
	}

	/// In-place update: tool state.
	pub fn set_state(&mut self, state: ToolState) -> bool {
		if self.state == state { return false; }
		self.state = state;
		true
	}

	/// Sets the tool state.
	pub fn state(mut self, state: ToolState) -> Self {
		self.set_state(state);
		self
	}

	/// In-place update: intent/summary text.
	pub fn set_intent(&mut self, intent: impl Into<Str>) -> bool {
		let intent = intent.into();
		if self.intent == intent { return false; }
		self.intent = intent;
		true
	}

	/// Sets the intent or summary text.
	pub fn intent(mut self, intent: impl Into<Str>) -> Self {
		self.set_intent(intent);
		self
	}

	/// In-place update: badge text.
	pub fn set_badge(&mut self, badge: impl Into<Str>) -> bool {
		let badge = badge.into();
		if self.badge == badge { return false; }
		self.badge = badge;
		true
	}

	/// Sets the right-aligned badge text (e.g. elapsed time).
	pub fn badge(mut self, badge: impl Into<Str>) -> Self {
		self.set_badge(badge);
		self
	}

	/// In-place update: fold state.
	pub fn set_folded(&mut self, folded: bool) -> bool {
		if self.folded == folded { return false; }
		self.folded = folded;
		true
	}

	/// Sets whether the card is folded (hides children).
	pub fn folded(mut self, folded: bool) -> Self {
		self.set_folded(folded);
		self
	}

	/// Replaces the card body children.
	pub fn replace_body(&mut self, children: impl IntoChildren) -> bool {
		self.children.clear();
		children.extend_children(&mut self.children);
		true
	}

	/// Appends child components to the card's body.
	pub fn child(mut self, children: impl IntoChildren) -> Self {
		children.extend_children(&mut self.children);
		self
	}

	fn header_color(&self, ctx: &UiContext) -> Color {
		match self.state {
			ToolState::Streaming => ctx.theme.accent,
			ToolState::Success => ctx.theme.ok,
			ToolState::Failure => ctx.theme.err,
		}
	}
}

impl Default for ToolCard {
    fn default() -> Self {
        Self::new()
    }
}

impl Component for ToolCard {
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

	fn measure(&mut self, ctx: &UiContext) -> (u16, u16) {
		let name_len = cell_width(&self.name);
		let intent_len = cell_width(&self.intent);
		let badge_len = cell_width(&self.badge);
		let header_min = 5 + name_len + intent_len + badge_len;
        let header_nat = header_min.saturating_add(if badge_len > 0 { 2 } else { 0 });

		if self.folded || self.children.is_empty() {
			(header_min, header_nat.max(30))
		} else {
			let (child_min, child_nat) = stack_measure(ctx, &mut self.children);
			(header_min.max(child_min.saturating_add(2)), header_nat.max(child_nat.saturating_add(2)).max(30))
		}
	}

	fn height(&mut self, ctx: &UiContext, width: u16) -> u16 {
		if self.folded || self.children.is_empty() {
			1
		} else {
			let child_width = width.saturating_sub(2);
			let child_h = stack_height(ctx, &mut self.children, child_width, 0);
			1 + child_h + 1
		}
	}

	fn place(&mut self, ctx: &UiContext, content: Rect) {
		if !self.folded && !self.children.is_empty() {
			let child_rect = Rect::new(
				content.x.saturating_add(2),
				content.y.saturating_add(1),
				content.width.saturating_sub(2),
				content.height.saturating_sub(2),
			);
			stack_place(ctx, &mut self.children, child_rect, 0, Some(VAlign::Start), Align::Start);
		}
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		if rect.y >= pc.clip || rect.width == 0 {
			return;
		}

		let expander = pc.ctx.charset.expander(!self.folded && !self.children.is_empty());
		let header_color = self.header_color(pc.ctx);
		let header_style = Style::new().fg(header_color);
        let normal_style = Style::new().fg(pc.ctx.theme.fg);
        let muted_style = Style::new().fg(pc.ctx.theme.muted);

		let mut x = rect.x;
		let y = rect.y;

		x = pc.frame.put(x, y, expander, header_style);

		match self.state {
			ToolState::Streaming => {
				let frames = pc.ctx.charset.spinner();
				x = pc.frame.put(x, y, frames.at(pc.now), header_style);
				pc.wake(self.slot, frames.next_change(pc.now));
			}
			ToolState::Success => {
				x = pc.frame.put(x, y, pc.ctx.charset.check(), header_style);
			}
			ToolState::Failure => {
				x = pc.frame.put(x, y, pc.ctx.charset.icon(crate::Icon::Error), header_style);
			}
		}
		x = pc.frame.put(x, y, " ", header_style);

		if !self.name.is_empty() {
			x = pc.frame.put(x, y, &self.name, header_style.bold());
            x = pc.frame.put(x, y, " ", normal_style);
		}

		if !self.intent.is_empty() {
            let badge_width = cell_width(&self.badge);
            let mut available = rect.x.saturating_add(rect.width).saturating_sub(x);
            if !self.badge.is_empty() {
                available = available.saturating_sub(badge_width + 1);
            }
            
            let text = &self.intent[..];
            x = pc.frame.put_clipped(x, y, available, text, normal_style);
		}

        if !self.badge.is_empty() {
            let badge_start = rect.x.saturating_add(rect.width).saturating_sub(cell_width(&self.badge));
            let badge_x = x.max(badge_start);
            pc.frame.put(badge_x, y, &self.badge, muted_style);
        }

		if self.folded || self.children.is_empty() {
			return;
		}

		for child in self.children.iter_mut().filter(|child| child.visible) {
			child.paint(pc);
		}

		let (_, last, rail) = pc.ctx.charset.guides(Border::Round);
		
        let child_h = rect.height.saturating_sub(2);
        for row in 0..child_h {
            let cy = y + 1 + row;
            if cy < pc.clip {
                pc.frame.put(rect.x, cy, rail, header_style);
            }
        }

        let bottom_y = y + 1 + child_h;
        if bottom_y < pc.clip {
            let mut bx = pc.frame.put(rect.x, bottom_y, last, header_style);
            let width = rect.width.saturating_sub(2);
            let mut buf = [0; 4];
            let rule = pc.ctx.charset.rule().encode_utf8(&mut buf);
            for _ in 0..width {
                bx = pc.frame.put(bx, bottom_y, rule, header_style);
            }
        }
	}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        components::TextLeaf,
        test_support::frame_row_text,
        ui::Ui,
    };

    #[test]
    fn formats_streaming_state_and_badge() {
        let card = ToolCard::new()
            .name("read")
            .intent("src/lib.rs")
            .badge("12ms")
            .state(ToolState::Streaming);

        let ui = Ui::from_root(card, 40, UiContext::default());
        assert!(frame_row_text(ui.frame(), 0).contains("read src/lib.rs"));
        assert!(frame_row_text(ui.frame(), 0).contains("12ms"));
    }

    #[test]
    fn formats_success_and_failure_states() {
        let ui = Ui::from_root(
            ToolCard::new().name("grep").intent("foo").state(ToolState::Success),
            20,
            UiContext::default(),
        );
        let row_success = frame_row_text(ui.frame(), 0);
        assert!(row_success.contains("grep foo"));

        let ui_fail = Ui::from_root(
            ToolCard::new().name("fail").intent("bar").state(ToolState::Failure),
            20,
            UiContext::default(),
        );
        let row_fail = frame_row_text(ui_fail.frame(), 0);
        assert!(row_fail.contains("fail bar"));
    }

    #[test]
    fn truncates_intent_narrow_width_without_panic() {
        let card = ToolCard::new()
            .name("long")
            .intent("this is a very long intent with 🚀 emoji")
            .badge("ok");
        
        let ui = Ui::from_root(card, 20, UiContext::default());
        let row = frame_row_text(ui.frame(), 0);
        assert!(row.contains("long"));
        assert!(row.contains("ok"));
        // check it has graphemes
        assert!(row.chars().count() > 10); 
    }

    #[test]
    fn renders_open_children_with_rails() {
        let card = ToolCard::new()
            .name("bash")
            .child(TextLeaf::new().text("echo ok"))
            .state(ToolState::Success);
            
        let ui = Ui::from_root(card, 20, UiContext::default());
        let row_0 = frame_row_text(ui.frame(), 0);
        assert!(row_0.contains("bash"));
        assert_eq!(frame_row_text(ui.frame(), 1), "│ echo ok");
        assert_eq!(frame_row_text(ui.frame(), 2), "╰───────────────────");
    }

    #[test]
    fn mutable_transitions_and_narrow_rendering() {
        let card = ToolCard::new()
            .with(Prop::Id, "t1")
            .name("edit")
            .intent("src/file.txt")
            .state(ToolState::Streaming);

        let mut ui = Ui::from_root(card, 15, UiContext::default()); // narrow 15
        assert!(frame_row_text(ui.frame(), 0).contains("edit src/"));

        // Transition to success with children
        let changed = ui.update_component::<ToolCard>("t1", |card| {
            let mut dirty = false;
            dirty |= card.set_state(ToolState::Success);
            dirty |= card.set_badge("1s");
            dirty |= card.replace_body(TextLeaf::new().text("done"));
            dirty
        });
        assert!(changed);
        
        let row_0 = frame_row_text(ui.frame(), 0);
        assert!(row_0.contains("edit"));
        assert!(row_0.contains("1s"));
        assert_eq!(frame_row_text(ui.frame(), 1), "│ done");

        // Transition to folded failure
        let changed = ui.update_component::<ToolCard>("t1", |card| {
            let mut dirty = false;
            dirty |= card.set_state(ToolState::Failure);
            dirty |= card.set_folded(true);
            dirty
        });
        assert!(changed);

        let row_0_fail = frame_row_text(ui.frame(), 0);
        assert!(row_0_fail.contains("edit")); // name is still edit
        assert_eq!(ui.frame().size().height, 1); // folded means height is 1
    }
}