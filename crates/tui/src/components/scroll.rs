use crate::{
	component::{
		Cached, Component, EventCtx, Flow, Hit, HitTag, IntoChildren, PaintCtx, Slot, next_slot,
	},
	context::UiContext,
	frame::{Frame, Rect, Size, Style},
	input::{Key, Mouse},
	props::{Prop, PropValue, Props},
};

#[derive(Clone, Debug, Default)]
struct ScrollState {
	off:       u16,
	content_h: u16,
	scratch:   Option<Frame>,
}

/// A vertically scrollable child stack backing the `<scroll>` markup tag.
pub struct Scroll {
	props:    Props,
	slot:     Slot,
	children: Vec<Cached>,
	state:    ScrollState,
}

impl Scroll {
	/// Creates an empty scrolling region.
	pub fn new() -> Self {
		Self {
			props:    Props::new(),
			slot:     next_slot(),
			children: Vec::new(),
			state:    ScrollState::default(),
		}
	}

	/// Sets one scrolling-region property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one scrolling-region property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Appends child components to the scrolling region.
	pub fn child(mut self, children: impl IntoChildren) -> Self {
		children.extend_children(&mut self.children);
		self
	}

	pub(crate) fn chase(&mut self, descendant_rect: Rect, view_rows: u16) -> bool {
		let previous = self.state.off;
		let view_bottom = previous.saturating_add(view_rows);
		let descendant_bottom = descendant_rect.y.saturating_add(descendant_rect.height);
		let target = if descendant_rect.y < previous {
			descendant_rect.y
		} else if descendant_bottom > view_bottom {
			descendant_bottom.saturating_sub(view_rows)
		} else {
			previous
		};
		self.state.off = target.min(self.state.content_h.saturating_sub(view_rows));
		self.state.off != previous
	}

	fn scroll_by(&mut self, delta: i32, view_rows: u16) -> bool {
		let max_off = self.state.content_h.saturating_sub(view_rows);
		let next = (i64::from(self.state.off) + i64::from(delta)).clamp(0, i64::from(max_off)) as u16;
		let changed = next != self.state.off;
		self.state.off = next;
		changed
	}
}

impl Default for Scroll {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Scroll {
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
		let mut min = 0;
		let mut nat = 0;
		for child in self.children.iter_mut().filter(|child| child.visible) {
			let (child_min, child_nat) = child.measure(ctx);
			min = min.max(child_min);
			nat = nat.max(child_nat);
		}
		(min.saturating_add(1), nat.saturating_add(1))
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		8
	}

	fn place(&mut self, ctx: &UiContext, content: Rect) {
		let inner = content.width.saturating_sub(1).max(1);
		let mut y = 0u16;
		for child in self.children.iter_mut().filter(|child| child.visible) {
			let height = child.height(ctx, inner);
			child.place(ctx, Rect::new(0, y, inner, height));
			y = y.saturating_add(height);
		}
		self.state.content_h = y;
		self.state.off = self.state.off.min(y.saturating_sub(content.height));
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		pc.hits
			.push(Hit { rect, slot: self.slot, tag: HitTag::Wheel });
		let child_hits = pc.hits.len();
		let inner = rect.width.saturating_sub(1).max(1);
		let content_h = self.state.content_h;
		let window_rows = rect.height.min(pc.clip.saturating_sub(rect.y));
		let off = self.state.off;
		let mut scratch = self
			.state
			.scratch
			.take()
			.filter(|frame| frame.size() == Size::new(inner, content_h))
			.unwrap_or_else(|| Frame::new(Size::new(inner, content_h)));
		scratch.clear(Style::default());
		// The pointer moves into scratch coordinates with the children;
		// pointers outside the window vanish.
		let pointer = pc.pointer.and_then(|(x, y)| {
			let column = x.checked_sub(rect.x)?;
			let row = y.checked_sub(rect.y)?;
			(column < inner && row < window_rows).then(|| (column, row.saturating_add(off)))
		});
		{
			let mut child_pc = pc.nested(&mut scratch, content_h);
			child_pc.pointer = pointer;
			for child in self.children.iter_mut().filter(|child| child.visible) {
				child.paint(&mut child_pc);
			}
		}

		pc.frame
			.blit(&scratch, self.state.off, window_rows, rect.x, rect.y);
		translate_hits(pc.hits, child_hits, rect, self.state.off, window_rows);
		self.state.scratch = Some(scratch);

		if rect.width == 0 || content_h <= rect.height {
			return;
		}
		let bar_x = rect.x.saturating_add(rect.width - 1);
		let thumb_h = (rect.height.saturating_mul(rect.height) / content_h).max(1);
		let denom = content_h - rect.height;
		let thumb_top = rect
			.height
			.saturating_sub(thumb_h)
			.saturating_mul(self.state.off)
			.checked_div(denom)
			.unwrap_or(0);
		for row in 0..window_rows {
			let (glyph, style) = if row >= thumb_top && row < thumb_top.saturating_add(thumb_h) {
				(pc.ctx.charset.scrollbar().1, Style::new().fg(pc.ctx.theme.accent))
			} else {
				(pc.ctx.charset.scrollbar().0, Style::new().fg(pc.ctx.theme.muted))
			};
			pc.frame
				.put(bar_x, rect.y.saturating_add(row), glyph, style);
		}
		pc.hits.push(Hit {
			rect: Rect::new(bar_x, rect.y, 1, window_rows),
			slot: self.slot,
			tag:  HitTag::Scrollbar,
		});
	}

	fn focusable(&self) -> bool {
		true
	}

	fn key(&mut self, ec: &mut EventCtx<'_>, key: Key) -> Flow {
		let view_rows = ec.view_rows;
		let delta = match key {
			Key::Up => -1,
			Key::Down => 1,
			Key::PageUp => -i32::from(view_rows),
			Key::PageDown => i32::from(view_rows),
			Key::Home => -i32::from(self.state.content_h),
			Key::End => i32::from(self.state.content_h),
			Key::Ctrl('u') => -i32::from(view_rows / 2).max(1),
			Key::Ctrl('d') => i32::from(view_rows / 2).max(1),
			_ => return Flow::Skip,
		};
		let changed = self.scroll_by(delta, view_rows);
		if !changed && matches!(key, Key::Up | Key::Down) {
			Flow::Skip
		} else {
			Flow::Consumed
		}
	}

	fn mouse(
		&mut self,
		ec: &mut EventCtx<'_>,
		tag: HitTag,
		at: (u16, u16),
		rect: Rect,
		mouse: Mouse,
	) -> Flow {
		match tag {
			HitTag::Scrollbar => match mouse {
				// Click or drag on the bar centers the thumb on the pointer
				// row — the inverse of the thumb placement painted above.
				Mouse::Click | Mouse::Drag => {
					let track = rect.height;
					let content_h = self.state.content_h;
					if track == 0 || content_h <= track {
						return Flow::Consumed;
					}
					let thumb_h = (track.saturating_mul(track) / content_h).max(1);
					let span = track - thumb_h;
					if span == 0 {
						return Flow::Consumed;
					}
					let row = at.1.saturating_sub(rect.y).min(track - 1);
					let grab = row.saturating_sub(thumb_h / 2).min(span);
					let range = u32::from(content_h - track);
					let target =
						((u32::from(grab) * range + u32::from(span / 2)) / u32::from(span)) as u16;
					self.state.off = target;
					Flow::Consumed
				},
				// Swallow everything else on the bar so a release or stray
				// click never falls through to occluded content.
				Mouse::RightClick | Mouse::MiddleClick | Mouse::Release => Flow::Consumed,
				_ => Flow::Skip,
			},
			HitTag::Wheel => match mouse {
				Mouse::WheelUp | Mouse::WheelDown => {
					let delta = if mouse == Mouse::WheelUp { -1 } else { 1 };
					if self.scroll_by(delta, ec.view_rows) {
						Flow::Consumed
					} else {
						Flow::Skip
					}
				},
				// Scroll has no horizontal offset. Swallow horizontal wheels
				// at the viewport so an underlying or ancestor widget cannot
				// act on them.
				Mouse::WheelLeft | Mouse::WheelRight => Flow::Consumed,
				_ => Flow::Skip,
			},
			_ => Flow::Skip,
		}
	}
}

fn translate_hits(hits: &mut Vec<Hit>, start: usize, viewport: Rect, off: u16, rows: u16) {
	let clip = Rect::new(viewport.x, viewport.y, viewport.width, rows);
	let mut write = start;
	for read in start..hits.len() {
		let mut hit = hits[read];
		let x = i32::from(viewport.x) + i32::from(hit.rect.x);
		let y = i32::from(viewport.y) + i32::from(hit.rect.y) - i32::from(off);
		let Some(rect) = translated_intersection(hit.rect, x, y, clip) else {
			continue;
		};
		hit.rect = rect;
		hits[write] = hit;
		write += 1;
	}
	hits.truncate(write);
}

fn translated_intersection(source: Rect, x: i32, y: i32, clip: Rect) -> Option<Rect> {
	let left = x.max(i32::from(clip.x));
	let top = y.max(i32::from(clip.y));
	let right = x
		.saturating_add(i32::from(source.width))
		.min(i32::from(clip.x) + i32::from(clip.width));
	let bottom = y
		.saturating_add(i32::from(source.height))
		.min(i32::from(clip.y) + i32::from(clip.height));
	if left >= right || top >= bottom {
		return None;
	}
	Some(Rect::new(left as u16, top as u16, (right - left) as u16, (bottom - top) as u16))
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{components::Pre, test_support::frame_row_text};

	#[test]
	fn scroll_clamps_and_blits_from_scratch() {
		let ctx = UiContext::default();
		let mut scroll = Scroll::new().child(Pre::new().text("one\ntwo\nthree"));
		scroll.place(&ctx, Rect::new(0, 0, 8, 2));
		let mut ec = EventCtx::new(&ctx, 8, 2);
		assert_eq!(scroll.key(&mut ec, Key::Up), Flow::Skip);
		assert_eq!(scroll.key(&mut ec, Key::Down), Flow::Consumed);
		assert_eq!(scroll.state.off, 1);
		assert_eq!(scroll.key(&mut ec, Key::Down), Flow::Skip);
		assert_eq!(
			scroll.mouse(&mut ec, HitTag::Wheel, (0, 0), Rect::new(0, 0, 8, 2), Mouse::WheelDown),
			Flow::Skip,
		);
		assert_eq!(scroll.state.off, 1);
		assert_eq!(
			scroll.mouse(&mut ec, HitTag::Wheel, (0, 0), Rect::new(0, 0, 8, 2), Mouse::WheelLeft),
			Flow::Consumed,
		);
		assert_eq!(
			scroll.mouse(&mut ec, HitTag::Wheel, (0, 0), Rect::new(0, 0, 8, 2), Mouse::WheelRight),
			Flow::Consumed,
		);
		assert_eq!(scroll.state.off, 1);

		let mut frame = Frame::new(Size::new(8, 2));
		let mut hits = Vec::new();
		let mut wakes = Vec::new();
		let mut pc = PaintCtx::new(&mut frame, &ctx, &mut hits, &mut wakes);
		scroll.paint(&mut pc, Rect::new(0, 0, 8, 2));
		assert!(frame_row_text(&frame, 0).starts_with("two"));
		assert!(frame_row_text(&frame, 1).starts_with("three"));
	}

	#[test]
	fn focus_chase_moves_only_as_far_as_needed() {
		let ctx = UiContext::default();
		let mut scroll = Scroll::new().child(Pre::new().text("one\ntwo\nthree\nfour"));
		scroll.place(&ctx, Rect::new(0, 0, 8, 2));
		assert!(scroll.chase(Rect::new(0, 3, 4, 1), 2));
		assert_eq!(scroll.state.off, 2);
		assert!(!scroll.chase(Rect::new(0, 2, 4, 1), 2));
		assert!(scroll.chase(Rect::new(0, 0, 4, 1), 2));
		assert_eq!(scroll.state.off, 0);
	}

	#[test]
	fn child_hits_translate_and_clip_to_the_viewport() {
		let mut hits = vec![Hit { rect: Rect::new(0, 0, 3, 1), slot: 7, tag: HitTag::Press }, Hit {
			rect: Rect::new(1, 2, 4, 2),
			slot: 8,
			tag:  HitTag::Row(0),
		}];
		translate_hits(&mut hits, 0, Rect::new(10, 5, 6, 2), 1, 2);
		assert_eq!(hits.len(), 1);
		assert_eq!(hits[0].slot, 8);
		assert_eq!(hits[0].rect, Rect::new(11, 6, 4, 1));
	}

	#[test]
	fn scrollbar_clicks_jump_and_drags_track_the_pointer() {
		let ctx = UiContext::default();
		let mut scroll =
			Scroll::new().child(Pre::new().text("l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\nl11\nl12"));
		scroll.place(&ctx, Rect::new(0, 0, 10, 4));
		let mut ec = EventCtx::new(&ctx, 10, 4);
		let bar = Rect::new(9, 0, 1, 4);

		// content 12, track 4 → thumb 1, span 3, range 8.
		let flow = scroll.mouse(&mut ec, HitTag::Scrollbar, (9, 3), bar, Mouse::Click);
		assert_eq!(flow, Flow::Consumed);
		assert_eq!(scroll.state.off, 8, "bottom of the track is the maximum offset");

		// A drag row maps proportionally, even when the pointer leaves the
		// bar column or the rectangle vertically.
		scroll.mouse(&mut ec, HitTag::Scrollbar, (3, 1), bar, Mouse::Drag);
		assert_eq!(scroll.state.off, 3);
		scroll.mouse(&mut ec, HitTag::Scrollbar, (9, 60), bar, Mouse::Drag);
		assert_eq!(scroll.state.off, 8);
		scroll.mouse(&mut ec, HitTag::Scrollbar, (9, 0), bar, Mouse::Drag);
		assert_eq!(scroll.state.off, 0);

		// Releases and stray clicks on the bar are swallowed, not forwarded.
		assert_eq!(
			scroll.mouse(&mut ec, HitTag::Scrollbar, (9, 0), bar, Mouse::Release),
			Flow::Consumed
		);
	}

	#[test]
	fn scrollbar_hit_zone_routes_through_ui_mouse_handling() {
		let mut ui = crate::Ui::from_markup(
			"<scroll h=4><pre>l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\nl11\nl12</pre></scroll>",
			10,
			crate::UiContext::default(),
		)
		.unwrap();
		assert!(crate::test_support::frame_row_text(ui.frame(), 0).starts_with("l1"));

		// Click the bottom of the bar column: jump to the end.
		ui.handle_mouse(9, 3, Mouse::Click);
		assert!(crate::test_support::frame_row_text(ui.frame(), 0).starts_with("l9"));

		// Drag capture keeps routing to the bar even off-column.
		ui.handle_mouse(2, 1, Mouse::Drag);
		assert!(crate::test_support::frame_row_text(ui.frame(), 0).starts_with("l4"));
		ui.handle_mouse(2, 0, Mouse::Drag);
		assert!(crate::test_support::frame_row_text(ui.frame(), 0).starts_with("l1"));
		ui.handle_mouse(2, 0, Mouse::Release);
	}
}
