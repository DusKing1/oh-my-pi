use omp_core::SmolStr;
use smallvec::SmallVec;

use crate::{
	component::{Component, PaintCtx, Slot, next_slot},
	context::{Charset, UiContext},
	frame::{Color, Rect},
	markup::Align,
	props::{Prop, PropValue, Props},
	rich::cell_width,
};

/// Declarative segment data backing the `<segment>` markup tag.
pub struct Segment {
	props: Props,
	label: SmolStr,
}

impl Segment {
	/// Creates an empty status segment.
	pub fn new() -> Self {
		Self { props: Props::new(), label: SmolStr::default() }
	}

	/// Appends label text.
	pub fn label(mut self, label: impl Into<SmolStr>) -> Self {
		let label = label.into();
		if self.label.is_empty() {
			self.label = label;
		} else {
			self.label = SmolStr::from(format!("{}{}", self.label, label));
		}
		self
	}

	/// Sets one segment property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one custom segment property.
	pub fn with_custom(mut self, name: impl Into<SmolStr>, value: impl Into<PropValue>) -> Self {
		self.props.set_custom(name, value);
		self
	}
}

impl Default for Segment {
	fn default() -> Self {
		Self::new()
	}
}

/// A one-line powerline-style status group backing the `<status>` markup tag.
///
/// `align=end` (`right`) mirrors the caps for a band docked against the right
/// edge: the opening cap points into the background and the closing edge sits
/// solid on the margin.
pub struct Status {
	props:       Props,
	slot:        Slot,
	segments:    SmallVec<Segment, 8>,
	text_widths: SmallVec<u16, 8>,
}

impl Status {
	/// Creates an empty status group.
	pub fn new() -> Self {
		Self {
			props:       Props::new(),
			slot:        next_slot(),
			segments:    SmallVec::new(),
			text_widths: SmallVec::new(),
		}
	}

	/// Sets one status property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one status property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Appends a segment to the group.
	pub fn segment(mut self, segment: Segment) -> Self {
		let width = self
			.text_widths
			.last()
			.copied()
			.unwrap_or(0)
			.saturating_add(cell_width(&segment.label));
		self.segments.push(segment);
		self.text_widths.push(width);
		self
	}

	/// Band chrome for this group's dock side.
	fn chrome(&self, charset: Charset) -> (&'static str, &'static str, &'static str) {
		match self.props.align() {
			Align::End => charset.status_band_end(),
			Align::Start | Align::Center => charset.status_band(),
		}
	}

	fn group_width(&self, count: usize, charset: Charset) -> u16 {
		let (left_cap, separator, cap) = self.chrome(charset);
		let text = count
			.checked_sub(1)
			.and_then(|index| self.text_widths.get(index))
			.copied()
			.unwrap_or(0);
		let separators = u16::try_from(count.saturating_sub(1))
			.unwrap_or(u16::MAX)
			.saturating_mul(cell_width(separator).saturating_add(2));
		text
			.saturating_add(separators)
			.saturating_add(cell_width(left_cap))
			.saturating_add(2)
			.saturating_add(cell_width(cap))
	}
}

impl Default for Status {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Status {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn measure(&mut self, ctx: &UiContext) -> (u16, u16) {
		let min = self.group_width(self.segments.len().min(1), ctx.charset);
		let natural = self.group_width(self.segments.len(), ctx.charset);
		(min, natural)
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		1
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		if rect.y >= pc.clip || rect.width == 0 {
			return;
		}
		let mut visible = self.segments.len();
		while visible > 1 && self.group_width(visible, pc.ctx.charset) > rect.width {
			visible -= 1;
		}
		let style = self.props.style(&pc.ctx.theme);
		let (left_cap, separator, cap) = self.chrome(pc.ctx.charset);
		let edge_style = crate::Style::new().fg(style.background_color());
		let mut column = pc.frame.put(rect.x, rect.y, left_cap, edge_style);
		column = pc.frame.put(column, rect.y, " ", style);
		for (index, segment) in self.segments[..visible].iter().enumerate() {
			if index > 0 {
				column = pc.frame.put(column, rect.y, " ", style.dim());
				column = pc.frame.put(column, rect.y, separator, style.dim());
				column = pc.frame.put(column, rect.y, " ", style.dim());
			}
			let mut segment_style = segment.props.style(&pc.ctx.theme).inherit(style);
			if segment_style.background_color() == Color::Default {
				segment_style = segment_style.bg(style.background_color());
			}
			column = pc.frame.put(column, rect.y, &segment.label, segment_style);
		}
		column = pc.frame.put(column, rect.y, " ", style);
		pc.frame.put(column, rect.y, cap, edge_style);
	}

	fn paints_background(&self) -> bool {
		false
	}
}

#[cfg(test)]
mod tests {
	use super::{Segment, Status};
	use crate::{
		Charset, Color, Prop, Ui, UiContext,
		component::{Cached, Hit, PaintCtx},
		dom,
		frame::{Frame, Rect, Size},
		test_support::frame_row_text,
	};

	fn paint(status: Status, width: u16) -> (Frame, Vec<Hit>) {
		paint_with_charset(status, width, Charset::default())
	}

	fn paint_with_charset(status: Status, width: u16, charset: Charset) -> (Frame, Vec<Hit>) {
		let ctx = UiContext { charset, ..UiContext::default() };
		let mut status = Cached::new(Box::new(status));
		status.place(&ctx, Rect::new(0, 0, width, 1));
		let mut frame = Frame::new(Size::new(width, 1));
		let mut hits = Vec::new();
		status.paint(&mut PaintCtx::new(&mut frame, &ctx, &mut hits, &mut Vec::new()));
		(frame, hits)
	}

	#[test]
	fn status_paints_segments_and_styles() {
		let status = Status::new()
			.with(Prop::Bg, "yellow")
			.segment(Segment::new().label("alpha").with(Prop::Fg, "red"))
			.segment(
				Segment::new()
					.label("beta")
					.with(Prop::Fg, "green")
					.with(Prop::Bg, "blue"),
			)
			.segment(Segment::new().label("gamma").with(Prop::Fg, "blue"));
		let (frame, hits) = paint(status, 40);
		assert_eq!(frame_row_text(&frame, 0), " alpha › beta › gamma ›");
		assert_eq!(frame.cell(1, 0).style.foreground_color(), Color::Rgb(255, 0, 0));
		assert_eq!(frame.cell(9, 0).style.foreground_color(), Color::Rgb(0, 128, 0));
		assert_eq!(frame.cell(16, 0).style.foreground_color(), Color::Rgb(0, 0, 255));
		assert_eq!(frame.cell(1, 0).style.background_color(), Color::Rgb(255, 255, 0),);
		assert_eq!(frame.cell(9, 0).style.background_color(), Color::Rgb(0, 0, 255));
		assert_eq!(
			frame.cell(22, 0).style.foreground_color(),
			Color::Rgb(255, 255, 0),
			"the cap uses the band's background as its foreground",
		);
		assert_eq!(
			frame.cell(22, 0).style.background_color(),
			Color::Default,
			"the cap transitions onto the surrounding background",
		);
		assert_eq!(
			frame.cell(23, 0).style.background_color(),
			Color::Default,
			"the band stops after the rendered group",
		);
		assert!(hits.is_empty());
	}

	#[test]
	fn nerd_font_edges_use_band_background_as_foreground() {
		let status = Status::new()
			.with(Prop::Bg, "yellow")
			.segment(Segment::new().label("chip"));
		let (frame, _) = paint_with_charset(status, 20, Charset::NerdFont);

		assert_eq!(frame_row_text(&frame, 0), "\u{e0b6} chip \u{e0b0}");
		for column in [0, 7] {
			assert_eq!(frame.cell(column, 0).style.foreground_color(), Color::Rgb(255, 255, 0),);
			assert_eq!(frame.cell(column, 0).style.background_color(), Color::Default);
		}
		assert_eq!(frame.cell(8, 0).style.background_color(), Color::Default);
	}

	#[test]
	fn align_end_mirrors_the_caps_for_a_right_docked_band() {
		let status = Status::new()
			.with_str(Prop::Align, "right")
			.with(Prop::Bg, "yellow")
			.segment(Segment::new().label("chip"));
		let (frame, _) = paint_with_charset(status, 20, Charset::NerdFont);
		assert_eq!(frame_row_text(&frame, 0), "\u{e0b2} chip");
		assert_eq!(
			frame.cell(6, 0).style.background_color(),
			Color::Rgb(255, 255, 0),
			"the flat closing edge keeps the band background through its pad cell",
		);
		let (frame, _) = paint(
			Status::new()
				.with_str(Prop::Align, "right")
				.segment(Segment::new().label("alpha"))
				.segment(Segment::new().label("beta")),
			20,
		);
		assert_eq!(frame_row_text(&frame, 0), "‹ alpha › beta");
	}

	#[test]
	fn status_narrow_width_drops_whole_trailing_segments() {
		let status = Status::new()
			.segment(Segment::new().label("alpha"))
			.segment(Segment::new().label("beta"))
			.segment(Segment::new().label("gamma"));
		let (frame, _) = paint(status, 10);
		let painted = frame_row_text(&frame, 0);
		assert_eq!(painted, " alpha ›");
		assert!(!painted.contains("beta"));
	}

	#[test]
	fn status_markup_paints_segment_labels() {
		let ui = Ui::from_markup(
			"<status><segment fg=green>alpha</segment><segment>beta</segment></status>",
			40,
			UiContext::default(),
		)
		.expect("status markup should parse");
		let painted = frame_row_text(ui.frame(), 0);
		assert!(painted.contains("alpha › beta"));
	}

	#[test]
	fn status_markup_rejects_orphan_segment() {
		let error = Ui::from_markup("<segment>alpha</segment>", 40, UiContext::default())
			.err()
			.expect("orphan segment must fail");
		assert!(
			error
				.message
				.contains("<segment> is not allowed directly inside")
		);
	}

	#[test]
	fn status_macro_paints_segment_label() {
		let ui = Ui::from_root(
			dom! { <status><segment fg=green>{"alpha"}</segment></status> },
			40,
			UiContext::default(),
		);
		assert!(frame_row_text(ui.frame(), 0).contains("alpha"));
	}
}
