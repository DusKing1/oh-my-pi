use smallvec::SmallVec;

use super::layout::{Track, distribute};
use crate::{
	component::{Cached, Component, IntoChildren, PaintCtx, Slot, next_slot},
	context::UiContext,
	frame::Rect,
	markup::{Align, Dim, Justify, VAlign},
	props::{Prop, PropValue, Props},
};

/// A horizontal child stack backing the `<row>` markup tag.
///
/// With the `wrap` flag set the row flows children into as many lines as
/// the width allows; each line is solved and justified independently, so a
/// wrapping row of fixed-width children behaves as a responsive grid.
pub struct Row {
	props:    Props,
	slot:     Slot,
	children: Vec<Cached>,
}

impl Row {
	/// Creates an empty row.
	pub fn new() -> Self {
		Self { props: Props::new(), slot: next_slot(), children: Vec::new() }
	}

	/// Sets one row property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one row property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Appends child components to the row.
	pub fn child(mut self, children: impl IntoChildren) -> Self {
		let first = self.children.len();
		children.extend_children(&mut self.children);
		for child in &mut self.children[first..] {
			child.comp_mut().props_mut().set(Prop::Vertical, true);
			child.invalidate();
		}
		self
	}

	fn visible(&self) -> SmallVec<usize, 8> {
		self
			.children
			.iter()
			.enumerate()
			.filter_map(|(index, child)| child.visible.then_some(index))
			.collect()
	}

	/// Width a child occupies when packing wrap lines: the requested or
	/// natural width clamped to `min`/`max`. Grow children pack at their
	/// minimum and expand once their line is solved.
	fn pack_width(&mut self, ctx: &UiContext, index: usize, width: u16) -> u16 {
		let (measured_min, measured_natural) = self.children[index].measure(ctx);
		let request = self.children[index].w(ctx);
		let props = self.children[index].comp().props();
		let minimum = measured_min.max(props.min().unwrap_or(0));
		let cap = props.max().unwrap_or(u16::MAX);
		let base = match request {
			Some(Dim::Pct(percent)) => ((u32::from(width) * u32::from(percent)) / 100).max(1) as u16,
			Some(Dim::Cells(cells)) => cells,
			None if props.grow().is_some() => minimum,
			None => measured_natural.min(width),
		};
		base.min(cap).max(minimum)
	}

	/// Greedily packs visible children into flow-wrap lines fitting `width`,
	/// returning each line's exclusive end position within `visible`.
	fn wrap_lines(
		&mut self,
		ctx: &UiContext,
		visible: &[usize],
		width: u16,
		gap: u16,
	) -> SmallVec<usize, 8> {
		let mut ends: SmallVec<usize, 8> = SmallVec::new();
		let mut used = 0_u16;
		let mut count = 0_usize;
		for (position, &index) in visible.iter().enumerate() {
			let pack = self.pack_width(ctx, index, width);
			let extended = used.saturating_add(gap).saturating_add(pack);
			if count > 0 && extended > width {
				ends.push(position);
				used = pack;
				count = 1;
			} else {
				used = if count == 0 { pack } else { extended };
				count += 1;
			}
		}
		if count > 0 {
			ends.push(visible.len());
		}
		ends
	}

	/// Height of one solved line: its tallest child at that child's width.
	fn line_height(&mut self, ctx: &UiContext, line: &[usize], width: u16, gap: u16) -> u16 {
		let widths = self.solve_row(ctx, line, width, gap);
		line
			.iter()
			.zip(widths)
			.map(|(&index, child_width)| self.children[index].height(ctx, child_width))
			.max()
			.unwrap_or(0)
	}

	fn solve_row(
		&mut self,
		ctx: &UiContext,
		visible: &[usize],
		available: u16,
		gap: u16,
	) -> SmallVec<u16, 8> {
		let count = visible.len();
		let room = available.saturating_sub(
			gap.saturating_mul(u16::try_from(count.saturating_sub(1)).unwrap_or(u16::MAX)),
		);
		let mut tracks: SmallVec<Track, 8> = SmallVec::new();
		for &index in visible {
			let (measured_min, measured_natural) = self.children[index].measure(ctx);
			let width_request = self.children[index].w(ctx);
			let props = self.children[index].comp().props();
			let mut track = Track {
				base:     0,
				min:      measured_min.max(props.min().unwrap_or(0)),
				cap:      props.max().unwrap_or(u16::MAX),
				grow:     None,
				flexible: false,
			};
			track.base = match width_request {
				Some(Dim::Pct(percent)) => {
					track.flexible = true;
					(u32::from(room) * u32::from(percent) / 100).max(1) as u16
				},
				Some(Dim::Cells(cells)) => cells,
				None => {
					if let Some(weight) = props.grow() {
						track.grow = Some(weight);
						track.min
					} else {
						track.flexible = true;
						measured_natural.min(room)
					}
				},
			};
			track.base = track.base.min(track.cap).max(track.min);
			tracks.push(track);
		}
		distribute(&mut tracks, room);
		tracks.iter().map(|track| track.base).collect()
	}

	fn align_cross_axis(
		&mut self,
		ctx: &UiContext,
		visible: &[usize],
		row: Option<VAlign>,
		top: u16,
		tallest: u16,
	) {
		for &index in visible {
			let mode = if self.children[index].comp().stretch_in_row() {
				VAlign::Stretch
			} else {
				row.unwrap_or(VAlign::Stretch)
			};
			let mut rect = self.children[index].rect;
			let slack = tallest.saturating_sub(rect.height);
			if slack == 0 {
				continue;
			}
			match mode {
				VAlign::Start => {},
				VAlign::Center => {
					rect.y = top.saturating_add(slack / 2);
					self.children[index].place(ctx, rect);
				},
				VAlign::End => {
					rect.y = top.saturating_add(slack);
					self.children[index].place(ctx, rect);
				},
				VAlign::Stretch => {
					rect.height = tallest;
					self.children[index].place(ctx, rect);
				},
			}
		}
	}

	/// Solves, justifies, places, and cross-axis-aligns one line of children,
	/// returning the line's height.
	fn place_line(
		&mut self,
		ctx: &UiContext,
		line: &[usize],
		x: u16,
		y: u16,
		width: u16,
		gap: u16,
	) -> u16 {
		let widths = self.solve_row(ctx, line, width, gap);
		let used = widths
			.iter()
			.copied()
			.fold(0_u16, u16::saturating_add)
			.saturating_add(
				gap.saturating_mul(u16::try_from(line.len().saturating_sub(1)).unwrap_or(0)),
			);
		let slack = width.saturating_sub(used);
		let justify = match self.props.get(Prop::Justify) {
			Some(PropValue::Justify(value)) => *value,
			_ => Justify::Start,
		};
		let mut cursor = x.saturating_add(match justify {
			Justify::Between => 0,
			Justify::Center => slack / 2,
			Justify::End => slack,
			Justify::Start => match self.props.align() {
				Align::Start => 0,
				Align::Center => slack / 2,
				Align::End => slack,
			},
		});
		let between = u16::try_from(line.len().saturating_sub(1)).unwrap_or(0);
		let (gap_extra, gap_remainder) = if justify == Justify::Between && between > 0 {
			(slack / between, slack % between)
		} else {
			(0, 0)
		};
		let mut tallest = 0_u16;
		for (position, (&index, child_width)) in line.iter().zip(widths).enumerate() {
			let height = self.children[index].height(ctx, child_width);
			self.children[index].place(ctx, Rect::new(cursor, y, child_width, height));
			tallest = tallest.max(height);
			let remainder = u16::from(u16::try_from(position).unwrap_or(u16::MAX) < gap_remainder);
			cursor = cursor
				.saturating_add(child_width)
				.saturating_add(gap)
				.saturating_add(gap_extra)
				.saturating_add(remainder);
		}
		self.align_cross_axis(ctx, line, self.props.valign(), y, tallest);
		tallest
	}
}

impl Default for Row {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Row {
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
		let visible = self.visible();
		let gaps = self
			.props
			.gap()
			.saturating_mul(u16::try_from(visible.len().saturating_sub(1)).unwrap_or(u16::MAX));
		let wraps = self.props.flag(Prop::Wrap);
		let mut minimum = if wraps { 0 } else { gaps };
		let mut natural = gaps;
		for index in visible {
			let (child_minimum, child_natural) = self.children[index].measure(ctx);
			let child_minimum =
				child_minimum.max(self.children[index].comp().props().min().unwrap_or(0));
			if wraps {
				minimum = minimum.max(child_minimum);
			} else {
				minimum = minimum.saturating_add(child_minimum);
			}
			natural = natural.saturating_add(child_natural);
		}
		(minimum, natural)
	}

	fn height(&mut self, ctx: &UiContext, width: u16) -> u16 {
		let visible = self.visible();
		let gap = self.props.gap();
		if self.props.flag(Prop::Wrap) {
			let ends = self.wrap_lines(ctx, &visible, width, gap);
			let mut total = 0_u16;
			let mut start = 0_usize;
			for &end in &ends {
				total = total.saturating_add(self.line_height(ctx, &visible[start..end], width, gap));
				start = end;
			}
			return total;
		}
		self.line_height(ctx, &visible, width, gap)
	}

	fn place(&mut self, ctx: &UiContext, content: Rect) {
		let visible = self.visible();
		let gap = self.props.gap();
		if self.props.flag(Prop::Wrap) {
			let ends = self.wrap_lines(ctx, &visible, content.width, gap);
			let mut top = content.y;
			let mut start = 0_usize;
			for &end in &ends {
				let tallest =
					self.place_line(ctx, &visible[start..end], content.x, top, content.width, gap);
				top = top.saturating_add(tallest);
				start = end;
			}
			return;
		}
		self.place_line(ctx, &visible, content.x, content.y, content.width, gap);
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, _rect: Rect) {
		for child in self.children.iter_mut().filter(|child| child.visible) {
			child.paint(pc);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::Row;
	use crate::{
		component::Component, components::TextLeaf, context::UiContext, frame::Rect, markup::Dim,
		props::Prop,
	};

	#[test]
	fn solves_percent_and_grow_widths_without_heap_scratch_for_small_rows() {
		let ctx = UiContext::default();
		let mut row = Row::new()
			.child(TextLeaf::new().text("pct").with(Prop::W, Dim::Pct(50)))
			.child(TextLeaf::new().text("grow").with(Prop::Grow, 1.0_f32));
		assert_eq!(row.measure(&ctx), (7, 7));
		row.place(&ctx, Rect::new(0, 0, 20, 1));
		assert_eq!(row.children()[0].rect, Rect::new(0, 0, 10, 1));
		assert_eq!(row.children()[1].rect, Rect::new(10, 0, 10, 1));
	}
}
