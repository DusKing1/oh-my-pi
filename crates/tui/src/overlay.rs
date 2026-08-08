//! Viewport-anchored sizing and placement for z-ordered layers.
//!
//! Geometry is resolved for each presentation, so layer cells never enter
//! native terminal scrollback. Retained stacks use [`crate::Ui::show_overlay`],
//! while raw-frame hosts pass [`Layer`]s to
//! [`crate::Renderer::present_overlaid`].

use crate::{
	frame::{Frame, Size},
	markup::Dim,
};

/// A viewport edge or corner used to position an overlay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OverlayAnchor {
	/// Centers the overlay horizontally and vertically.
	#[default]
	Center,
	/// Places the overlay at the top-left corner.
	TopLeft,
	/// Centers the overlay along the top edge.
	Top,
	/// Places the overlay at the top-right corner.
	TopRight,
	/// Centers the overlay along the right edge.
	Right,
	/// Places the overlay at the bottom-right corner.
	BottomRight,
	/// Centers the overlay along the bottom edge.
	Bottom,
	/// Places the overlay at the bottom-left corner.
	BottomLeft,
	/// Centers the overlay along the left edge.
	Left,
}

/// Insets that keep an overlay away from viewport edges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OverlayMargin {
	/// Inset from the top edge.
	pub top:    u16,
	/// Inset from the right edge.
	pub right:  u16,
	/// Inset from the bottom edge.
	pub bottom: u16,
	/// Inset from the left edge.
	pub left:   u16,
}

impl OverlayMargin {
	/// Creates equal insets on all four sides.
	pub const fn uniform(n: u16) -> Self {
		Self { top: n, right: n, bottom: n, left: n }
	}
}

/// Declarative sizing, placement, and visibility options for an overlay.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlayOptions {
	/// Requested width, resolved against the full viewport width.
	pub width:        Option<Dim>,
	/// Minimum width applied before fitting the overlay to its margins.
	pub min_width:    Option<u16>,
	/// Maximum height, resolved against the full viewport height.
	pub max_height:   Option<Dim>,
	/// Viewport position used when no explicit row or column is supplied.
	pub anchor:       OverlayAnchor,
	/// Horizontal displacement from the resolved position.
	pub offset_x:     i16,
	/// Vertical displacement from the resolved position.
	pub offset_y:     i16,
	/// Explicit absolute or percentage row position.
	pub row:          Option<Dim>,
	/// Explicit absolute or percentage column position.
	pub col:          Option<Dim>,
	/// Insets from viewport edges.
	pub margin:       OverlayMargin,
	/// Stacking height: higher layers composite above lower ones; ties stack by
	/// creation order.
	pub z:            i16,
	/// Smallest viewport in which the overlay is visible.
	pub min_viewport: Size,
	/// Whether the layer captures the keyboard while visible (the default).
	///
	/// A non-modal layer — a sidebar, a status rail — leaves keys and paste
	/// with the base tree unless explicitly focused through
	/// [`crate::Ui::focus_overlay`], and never triggers the [`crate::App`]
	/// alternate-screen hold: the document keeps committing to native
	/// scrollback beneath it while the layer rides the live viewport.
	pub modal:        bool,
	/// Stretches a retained overlay tree to the full available viewport
	/// height (after `margin` and `max_height`) instead of its content
	/// height, so `grow`/`valign` fill the band like a full-height rail.
	///
	/// Raw-frame [`Layer`] hosts control their frame height directly; the
	/// band there always follows the frame.
	pub fill_height:  bool,
}

impl Default for OverlayOptions {
	fn default() -> Self {
		Self {
			width:        None,
			min_width:    None,
			max_height:   None,
			anchor:       OverlayAnchor::Center,
			offset_x:     0,
			offset_y:     0,
			row:          None,
			col:          None,
			margin:       OverlayMargin::default(),
			z:            0,
			min_viewport: Size::new(0, 0),
			modal:        true,
			fill_height:  false,
		}
	}
}

impl OverlayOptions {
	/// Sets the requested width.
	#[must_use]
	pub const fn width(mut self, width: Dim) -> Self {
		self.width = Some(width);
		self
	}

	/// Sets the minimum width in cells.
	#[must_use]
	pub const fn min_width(mut self, min_width: u16) -> Self {
		self.min_width = Some(min_width);
		self
	}

	/// Sets the maximum height.
	#[must_use]
	pub const fn max_height(mut self, max_height: Dim) -> Self {
		self.max_height = Some(max_height);
		self
	}

	/// Sets the stacking height.
	#[must_use]
	pub const fn z(mut self, z: i16) -> Self {
		self.z = z;
		self
	}

	/// Sets the fallback anchor position.
	#[must_use]
	pub const fn anchor(mut self, anchor: OverlayAnchor) -> Self {
		self.anchor = anchor;
		self
	}

	/// Sets the horizontal offset in cells.
	#[must_use]
	pub const fn offset_x(mut self, offset_x: i16) -> Self {
		self.offset_x = offset_x;
		self
	}

	/// Sets the vertical offset in cells.
	#[must_use]
	pub const fn offset_y(mut self, offset_y: i16) -> Self {
		self.offset_y = offset_y;
		self
	}

	/// Sets an explicit absolute or percentage row.
	#[must_use]
	pub const fn row(mut self, row: Dim) -> Self {
		self.row = Some(row);
		self
	}

	/// Sets an explicit absolute or percentage column.
	#[must_use]
	pub const fn col(mut self, col: Dim) -> Self {
		self.col = Some(col);
		self
	}

	/// Sets the viewport-edge insets.
	#[must_use]
	pub const fn margin(mut self, margin: OverlayMargin) -> Self {
		self.margin = margin;
		self
	}

	/// Sets the minimum viewport required for visibility.
	#[must_use]
	pub const fn min_viewport(mut self, min_viewport: Size) -> Self {
		self.min_viewport = min_viewport;
		self
	}

	/// Leaves keys and paste with the base tree while the layer is visible.
	///
	/// [`crate::Ui::focus_overlay`] hands the keyboard to the layer on
	/// demand; a click inside its band does the same, and a click outside
	/// (or an unconsumed `Esc`) returns it.
	#[must_use]
	pub const fn non_modal(mut self) -> Self {
		self.modal = false;
		self
	}

	/// Stretches a retained overlay tree to the full available viewport
	/// height.
	#[must_use]
	pub const fn fill_height(mut self) -> Self {
		self.fill_height = true;
		self
	}
}

/// Identity handle returned by [`crate::Ui::show_overlay`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OverlayId(pub(crate) u32);

/// Resolved overlay dimensions before content-height clipping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlayExtent {
	pub width:      u16,
	pub max_height: u16,
}

/// Resolved viewport band of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlayBand {
	/// Leftmost viewport column of the band.
	pub x:       u16,
	/// Topmost viewport row of the band.
	pub y:       u16,
	/// First source-frame row composited into the band.
	pub src_top: u16,
	/// Number of source rows in the band; `0` means not composited.
	pub rows:    u16,
}

/// One z-ordered viewport layer: a rendered frame placed declaratively at
/// present time.
///
/// Stacking follows [`OverlayOptions::z`], then slice order.
pub struct Layer<'a> {
	/// Source frame containing the layer cells.
	pub frame:   &'a Frame,
	/// Placement, sizing, z, and visibility, resolved per present.
	pub options: &'a OverlayOptions,
	/// Whether this layer owns the keyboard — and with it the hardware
	/// cursor: an active layer's frame cursor places the caret (no frame
	/// cursor suppresses it), while passive layers let the base document's
	/// caret show through. At most one layer should be active per present;
	/// among several, the topmost wins.
	pub active:  bool,
}

impl Layer<'_> {
	/// Resolved viewport band for `viewport`; `rows == 0` means gated or empty.
	pub fn band(&self, viewport: Size) -> OverlayBand {
		if !visible_at(self.options, viewport) {
			return OverlayBand { x: 0, y: 0, src_top: 0, rows: 0 };
		}
		resolve_band(self.options, viewport, self.frame.size().width, self.frame.size().height)
	}
}

fn resolve_dim(dim: Dim, reference: u16) -> u16 {
	match dim {
		Dim::Cells(cells) => cells,
		Dim::Pct(percent) => {
			(u32::from(reference) * u32::from(percent) / 100).min(u32::from(u16::MAX)) as u16
		},
	}
}

fn offset(value: u16, amount: i16) -> u16 {
	(i32::from(value) + i32::from(amount)).clamp(0, i32::from(u16::MAX)) as u16
}

pub fn resolve_extent(options: &OverlayOptions, viewport: Size) -> OverlayExtent {
	let avail_width = viewport
		.width
		.saturating_sub(options.margin.left)
		.saturating_sub(options.margin.right)
		.max(1);
	let avail_height = viewport
		.height
		.saturating_sub(options.margin.top)
		.saturating_sub(options.margin.bottom)
		.max(1);

	let mut width = options
		.width
		.map_or_else(|| avail_width.min(80), |dim| resolve_dim(dim, viewport.width));
	if let Some(min_width) = options.min_width {
		width = width.max(min_width);
	}
	width = width.clamp(1, avail_width);

	let max_height = options
		.max_height
		.map_or(avail_height, |dim| resolve_dim(dim, viewport.height))
		.clamp(1, avail_height);

	OverlayExtent { width, max_height }
}

pub fn resolve_band(
	options: &OverlayOptions,
	viewport: Size,
	width: u16,
	content_height: u16,
) -> OverlayBand {
	let extent = resolve_extent(options, viewport);
	let effective_rows = content_height.min(extent.max_height);
	let avail_width = viewport
		.width
		.saturating_sub(options.margin.left)
		.saturating_sub(options.margin.right)
		.max(1);
	let avail_height = viewport
		.height
		.saturating_sub(options.margin.top)
		.saturating_sub(options.margin.bottom)
		.max(1);
	let row_span = avail_height.saturating_sub(effective_rows);
	let col_span = avail_width.saturating_sub(width);

	let row = match options.row {
		Some(Dim::Cells(row)) => row,
		Some(Dim::Pct(percent)) => options.margin.top.saturating_add(
			(u32::from(row_span) * u32::from(percent) / 100).min(u32::from(u16::MAX)) as u16,
		),
		None => match options.anchor {
			OverlayAnchor::TopLeft | OverlayAnchor::Top | OverlayAnchor::TopRight => {
				options.margin.top
			},
			OverlayAnchor::BottomLeft | OverlayAnchor::Bottom | OverlayAnchor::BottomRight => {
				options.margin.top.saturating_add(row_span)
			},
			OverlayAnchor::Center | OverlayAnchor::Left | OverlayAnchor::Right => {
				options.margin.top.saturating_add(row_span / 2)
			},
		},
	};
	let col = match options.col {
		Some(Dim::Cells(col)) => col,
		Some(Dim::Pct(percent)) => options.margin.left.saturating_add(
			(u32::from(col_span) * u32::from(percent) / 100).min(u32::from(u16::MAX)) as u16,
		),
		None => match options.anchor {
			OverlayAnchor::TopLeft | OverlayAnchor::Left | OverlayAnchor::BottomLeft => {
				options.margin.left
			},
			OverlayAnchor::TopRight | OverlayAnchor::Right | OverlayAnchor::BottomRight => {
				options.margin.left.saturating_add(col_span)
			},
			OverlayAnchor::Center | OverlayAnchor::Top | OverlayAnchor::Bottom => {
				options.margin.left.saturating_add(col_span / 2)
			},
		},
	};

	let max_row = viewport
		.height
		.saturating_sub(options.margin.bottom)
		.saturating_sub(effective_rows);
	let max_col = viewport
		.width
		.saturating_sub(options.margin.right)
		.saturating_sub(width);
	let y = offset(row, options.offset_y)
		.min(max_row)
		.max(options.margin.top);
	let x = offset(col, options.offset_x)
		.min(max_col)
		.max(options.margin.left);
	let rows = effective_rows.min(viewport.height.saturating_sub(y));
	let src_top = if content_height > effective_rows
		&& matches!(
			options.anchor,
			OverlayAnchor::BottomLeft | OverlayAnchor::Bottom | OverlayAnchor::BottomRight
		) {
		content_height - rows
	} else {
		0
	};

	OverlayBand { x, y, src_top, rows }
}

pub const fn visible_at(options: &OverlayOptions, viewport: Size) -> bool {
	viewport.width >= options.min_viewport.width && viewport.height >= options.min_viewport.height
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn defaults_to_centered_eighty_column_overlay() {
		let options = OverlayOptions::default();
		let viewport = Size::new(120, 40);

		assert_eq!(resolve_extent(&options, viewport), OverlayExtent {
			width:      80,
			max_height: 40,
		});
		assert_eq!(resolve_band(&options, viewport, 80, 10), OverlayBand {
			x:       20,
			y:       15,
			src_top: 0,
			rows:    10,
		});
	}

	#[test]
	fn percentages_resolve_against_full_viewport() {
		let options = OverlayOptions::default()
			.width(Dim::Pct(50))
			.max_height(Dim::Pct(25))
			.margin(OverlayMargin::uniform(3));

		assert_eq!(resolve_extent(&options, Size::new(120, 40)), OverlayExtent {
			width:      60,
			max_height: 10,
		});
	}

	#[test]
	fn margins_position_all_corner_anchors() {
		let viewport = Size::new(100, 40);
		let margin = OverlayMargin { top: 2, right: 3, bottom: 4, left: 5 };
		let cases = [
			(OverlayAnchor::TopLeft, (5, 2)),
			(OverlayAnchor::TopRight, (77, 2)),
			(OverlayAnchor::BottomLeft, (5, 26)),
			(OverlayAnchor::BottomRight, (77, 26)),
		];

		for (anchor, (x, y)) in cases {
			let options = OverlayOptions::default().anchor(anchor).margin(margin);
			let band = resolve_band(&options, viewport, 20, 10);
			assert_eq!((band.x, band.y), (x, y));
		}
	}

	#[test]
	fn explicit_percent_positions_use_remaining_space() {
		let options = OverlayOptions::default()
			.row(Dim::Pct(25))
			.col(Dim::Pct(50))
			.margin(OverlayMargin { top: 2, right: 3, bottom: 4, left: 5 });

		let band = resolve_band(&options, Size::new(100, 40), 20, 10);
		assert_eq!((band.x, band.y), (41, 8));
	}

	#[test]
	fn offsets_clamp_at_margin_edges() {
		let viewport = Size::new(100, 40);
		let margin = OverlayMargin::uniform(2);
		let top_left = OverlayOptions::default()
			.anchor(OverlayAnchor::TopLeft)
			.margin(margin)
			.offset_x(i16::MIN)
			.offset_y(i16::MIN);
		let bottom_right = OverlayOptions::default()
			.anchor(OverlayAnchor::BottomRight)
			.margin(margin)
			.offset_x(i16::MAX)
			.offset_y(i16::MAX);

		let top_left = resolve_band(&top_left, viewport, 20, 10);
		let bottom_right = resolve_band(&bottom_right, viewport, 20, 10);
		assert_eq!((top_left.x, top_left.y), (2, 2));
		assert_eq!((bottom_right.x, bottom_right.y), (78, 28));
	}

	#[test]
	fn bottom_anchor_slices_content_tail() {
		let bottom = OverlayOptions::default()
			.anchor(OverlayAnchor::Bottom)
			.max_height(Dim::Cells(10));
		let top = OverlayOptions::default()
			.anchor(OverlayAnchor::Top)
			.max_height(Dim::Cells(10));

		assert_eq!(resolve_band(&bottom, Size::new(100, 40), 20, 30).src_top, 20);
		assert_eq!(resolve_band(&top, Size::new(100, 40), 20, 30).src_top, 0);
	}

	#[test]
	fn minimum_viewport_gates_visibility() {
		let options = OverlayOptions::default().min_viewport(Size::new(80, 24));

		assert!(visible_at(&options, Size::new(80, 24)));
		assert!(!visible_at(&options, Size::new(79, 24)));
		assert!(!visible_at(&options, Size::new(80, 23)));
	}

	#[test]
	fn one_cell_viewport_clamps_without_panicking() {
		let options = OverlayOptions::default()
			.width(Dim::Pct(100))
			.max_height(Dim::Pct(100))
			.offset_x(i16::MAX)
			.offset_y(i16::MAX);
		let viewport = Size::new(1, 1);

		assert_eq!(resolve_extent(&options, viewport), OverlayExtent {
			width:      1,
			max_height: 1,
		});
		assert_eq!(resolve_band(&options, viewport, 1, 10), OverlayBand {
			x:       0,
			y:       0,
			src_top: 0,
			rows:    1,
		});
	}
}
