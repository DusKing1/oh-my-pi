use std::{
	collections::BTreeMap,
	fmt::Write as _,
	io::{self, Write},
	ops::Range,
};

use omp_core::CowBytes;
use smallvec::SmallVec;

use crate::{
	Graphics, TerminalCaps,
	escape::esc,
	frame::{Cell, CellContent, Color, Frame, LinkId, Size, Style, with_link_url},
	iterm2::{Iterm2Image, Iterm2Viewport, iterm2_output},
	kitty::{
		DirectPlacement, append_delete_image, append_direct_placement, append_placement,
		append_tmux_passthrough, append_transmission, placeholder_cell,
	},
	overlay::Layer,
	sixel::SixelImage,
	terminal::terminal_write_all,
};

const RESET_STYLE: &str = esc!(style_reset);
const CLEAR_VIEWPORT: &str = esc!(erase_display, cursor_home);
const SCREEN_TO_SCROLLBACK: &str = esc!(screen_to_scrollback);
const REBUILD_HISTORY: &str = esc!(cursor_home, erase_scrollback);
const SYNC_OUTPUT_BEGIN: &str = esc!(sync_output);
const SYNC_OUTPUT_END: &str = esc!(!sync_output);
const HIDE_CURSOR: &str = esc!(!cursor_visible);
const SHOW_CURSOR: &str = esc!(cursor_visible);
// CUD clamps at the bottom without changing the user's scrollback viewport,
// unlike an absolute CUP address.
const VIEWPORT_BOTTOM: &str = esc!(viewport_bottom);
const DEFAULT_CELL_PIXEL_WIDTH: u16 = 9;
const DEFAULT_CELL_PIXEL_HEIGHT: u16 = 18;
#[cfg(any(windows, target_os = "linux", test))]
const MAX_CONPTY_WRITE_CHUNK_BYTES: usize = 16 * 1024;
const MAX_OUTPUT_BACKLOG_BYTES: usize = 64 * 1024 * 1024;

/// Health of the renderer's bounded terminal output queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputState {
	/// Output is still being accepted.
	Connected,
	/// Pending terminal output exceeded the safety limit.
	Disconnected,
}

#[derive(Default)]
struct OutputBacklogGuard {
	bytes: usize,
}

impl OutputBacklogGuard {
	const fn queue(&mut self, bytes: usize) -> bool {
		self.bytes = self.bytes.saturating_add(bytes);
		self.bytes > MAX_OUTPUT_BACKLOG_BYTES
	}

	const fn flushed(&mut self) {
		self.bytes = 0;
	}
}

/// Measurements from one native-scrollback paint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PaintStats {
	/// Whether this replaced the complete viewport.
	pub full_repaint:   bool,
	/// Number of changed cells emitted.
	pub changed_cells:  usize,
	/// Number of changed runs or complete rows emitted.
	pub runs:           usize,
	/// Number of newly finalized rows committed to native scrollback.
	pub committed_rows: u16,
	/// Number of uncommitted logical rows clipped above the live viewport.
	pub clipped_rows:   u16,
	/// Number of bytes written to the terminal.
	pub bytes:          usize,
}

/// A layer with its band already resolved, ready to composite.
#[derive(Clone, Copy)]
pub struct ResolvedLayer<'a> {
	/// Source frame containing the layer cells.
	pub(crate) frame:   &'a Frame,
	/// Viewport column of the band's left edge.
	pub(crate) x:       u16,
	/// Viewport row of the band's top edge.
	pub(crate) y:       u16,
	/// First source-frame row in the band.
	pub(crate) src_top: u16,
	/// Number of source rows in the band.
	pub(crate) rows:    u16,
	/// Whether this layer owns the keyboard and hardware cursor.
	pub(crate) active:  bool,
}

struct StoredLayer {
	frame:           Frame,
	x:               u16,
	document_y:      u16,
	src_top:         u16,
	rows:            u16,
	active:          bool,
	source_address:  usize,
	source_id:       u64,
	source_revision: u64,
}

impl StoredLayer {
	#[inline(always)]
	const fn contains(&self, y: u16, x: u16) -> bool {
		y >= self.document_y
			&& y < self.document_y.saturating_add(self.rows)
			&& x >= self.x
			&& x < self.x.saturating_add(self.frame.size().width)
			&& y - self.document_y + self.src_top < self.frame.size().height
	}

	#[inline(always)]
	const fn same_cells_and_placement(&self, other: &Self) -> bool {
		self.x == other.x
			&& self.document_y == other.document_y
			&& self.src_top == other.src_top
			&& self.rows == other.rows
			&& self.source_address == other.source_address
			&& self.source_id == other.source_id
			&& self.source_revision == other.source_revision
	}
}

struct ComposedFrame<'a> {
	base:   &'a Frame,
	layers: &'a [StoredLayer],
}

impl ComposedFrame<'_> {
	#[inline(always)]
	fn cell_or<'b>(&'b self, y: u16, x: u16, blank: &'b Cell) -> &'b Cell {
		if self.layers.is_empty() {
			return self.base.cell_or(y, x, blank);
		}
		let layer = self.layer_at(y, x);
		let cell = match layer {
			Some(index) => {
				let layer = &self.layers[index];
				layer
					.frame
					.cell_or(y - layer.document_y + layer.src_top, x - layer.x, blank)
			},
			None => self.base.cell_or(y, x, blank),
		};
		match &cell.content {
			CellContent::Grapheme { width, .. } if *width > 1 => {
				let right = x.saturating_add(*width);
				if right > self.base.size().width
					|| (x..right).any(|column| self.layer_at(y, column) != layer)
				{
					blank
				} else {
					cell
				}
			},
			CellContent::Continuation => {
				let Some((head_x, width)) = self.grapheme_head(layer, y, x) else {
					return blank;
				};
				let right = head_x.saturating_add(width);
				if right > self.base.size().width
					|| (head_x..right).any(|column| self.layer_at(y, column) != layer)
				{
					blank
				} else {
					cell
				}
			},
			_ => cell,
		}
	}

	#[inline(always)]
	fn layer_at(&self, y: u16, x: u16) -> Option<usize> {
		match self.layers {
			[] => None,
			[layer] => layer.contains(y, x).then_some(0),
			layers => layers.iter().rposition(|layer| layer.contains(y, x)),
		}
	}

	fn grapheme_head(&self, layer: Option<usize>, y: u16, x: u16) -> Option<(u16, u16)> {
		let (frame, row, left) = match layer {
			Some(index) => {
				let layer = &self.layers[index];
				(&layer.frame, y - layer.document_y + layer.src_top, layer.x)
			},
			None => (self.base, y, 0),
		};
		let mut column = x;
		while column > left {
			column -= 1;
			let source_x = column - left;
			match &frame.cell(source_x, row).content {
				CellContent::Continuation => {},
				CellContent::Blank => return None,
				CellContent::Grapheme { width, .. } => return Some((column, *width)),
				CellContent::Image { .. } => return None,
			}
		}
		None
	}
}

#[derive(Clone, Copy)]
struct Layout {
	stable_limit: u16,
	window_top:   u16,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Window {
	top:    u16,
	height: u16,
}

#[derive(Clone, Copy)]
struct Run {
	document_y: u16,
	screen_y:   u16,
	start:      u16,
	end:        u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScreenCursor {
	row: u16,
	col: u16,
}

struct RegisteredImage {
	png:            CowBytes<'static>,
	uploaded:       bool,
	/// Cell boxes (`rows`, `cols`) already given a virtual placement.
	placed:         SmallVec<(u16, u16), 2>,
	sixel:          Option<SixelImage>,
	sixel_decoded:  bool,
	direct_visible: bool,
}

impl RegisteredImage {
	const fn new(png: CowBytes<'static>) -> Self {
		Self {
			png,
			uploaded: false,
			placed: SmallVec::new(),
			sixel: None,
			sixel_decoded: false,
			direct_visible: false,
		}
	}
}

/// Renders an immutable document prefix and a mutable viewport-local suffix.
///
/// `stable_rows` declares the leading rows that will never change again. Stable
/// rows enter native scrollback only when they leave the visible top edge.
/// Already-clipped stable rows remain protected but deferred until a rebuild,
/// avoiding a viewport replay that would displace native text selections.
/// A committed or previously declared-stable mutation is rejected before any
/// terminal output, because native scrollback has no addressable cells. The
/// retained physical screen model composes the raw previous frame with its
/// stored viewport layers.
pub struct Renderer<W: Write> {
	writer:               W,
	previous:             Option<Frame>,
	layers:               SmallVec<StoredLayer, 4>,
	layer_scratch:        SmallVec<StoredLayer, 4>,
	/// Throwaway-screen baseline retained separately from normal-buffer history.
	preview_previous:     Option<Frame>,
	preview_layers:       SmallVec<StoredLayer, 4>,
	preview_window:       Option<Window>,
	preview_cursor:       Option<ScreenCursor>,
	/// Reused ANSI cell-diff assembly buffer for steady-state paints.
	paint_scratch:        String,
	/// Reused terminal output assembly buffer for steady-state paints.
	output_scratch:       String,
	viewport_height:      u16,
	window_top:           u16,
	committed_rows:       u16,
	stable_rows:          u16,
	cursor:               Option<ScreenCursor>,
	poisoned:             bool,
	output_state:         OutputState,
	backlog:              OutputBacklogGuard,
	#[cfg(any(windows, target_os = "linux"))]
	conpty_hosted:        bool,
	images:               BTreeMap<u32, RegisteredImage>,
	alt_screen:           bool,
	graphics:             Graphics,
	cell_pixel_width:     u16,
	cell_pixel_height:    u16,
	tmux_passthrough:     bool,
	sync_output:          bool,
	screen_to_scrollback: bool,
	hyperlinks:           bool,
	margin_scrollback:    bool,
}

impl<W: Write> Renderer<W> {
	/// Creates a renderer whose first document clears only the visible viewport.
	pub fn new(writer: W) -> Self {
		Self {
			writer,
			previous: None,
			viewport_height: 0,
			layers: SmallVec::new(),
			layer_scratch: SmallVec::new(),
			preview_previous: None,
			preview_layers: SmallVec::new(),
			preview_window: None,
			preview_cursor: None,
			paint_scratch: String::new(),
			output_scratch: String::new(),
			window_top: 0,
			committed_rows: 0,
			stable_rows: 0,
			cursor: None,
			poisoned: false,
			output_state: OutputState::Connected,
			backlog: OutputBacklogGuard::default(),
			#[cfg(any(windows, target_os = "linux"))]
			conpty_hosted: is_conpty_hosted(),
			images: BTreeMap::new(),
			alt_screen: crate::terminal::alt_screen_active(),
			graphics: Graphics::KittyPlaceholders,
			cell_pixel_width: DEFAULT_CELL_PIXEL_WIDTH,
			cell_pixel_height: DEFAULT_CELL_PIXEL_HEIGHT,
			tmux_passthrough: false,
			sync_output: true,
			screen_to_scrollback: false,
			margin_scrollback: false,
			hyperlinks: false,
		}
	}

	/// Configures every capability-driven renderer option from resolved caps.
	///
	/// # Errors
	///
	/// Rejects zero cell-pixel dimensions.
	pub fn apply_caps(&mut self, caps: &TerminalCaps) -> io::Result<()> {
		self.set_graphics(caps.graphics);
		self.set_sync_output(caps.sync_output);
		self.set_screen_to_scrollback(caps.screen_to_scrollback);
		self.set_margin_scrollback(caps.margin_scrollback);
		self.set_hyperlinks(caps.hyperlinks);
		self.set_tmux_passthrough(caps.inside_tmux);
		if let Some((width, height)) = caps.cell_px {
			self.set_cell_pixel_size(width, height)?;
		}
		Ok(())
	}

	/// Registers PNG bytes for a typed terminal image ID.
	///
	/// Protocol encoding is deferred until a presented frame references the
	/// ID. Re-registering an ID replaces its bytes and protocol cache.
	///
	/// # Errors
	///
	/// Rejects ID zero and IDs wider than Kitty's 24-bit placeholder encoding.
	pub fn register_image(
		&mut self,
		id: u32,
		png_bytes: impl Into<CowBytes<'static>>,
	) -> io::Result<()> {
		if id == 0 || id > 0x00ff_ffff {
			return Err(io::Error::new(
				io::ErrorKind::InvalidInput,
				"terminal image ID must fit in 24 bits",
			));
		}
		self
			.images
			.insert(id, RegisteredImage::new(png_bytes.into()));
		Ok(())
	}

	/// Selects how typed image cells are materialized.
	///
	/// Set this before the first presentation. [`Graphics::Cells`],
	/// [`Graphics::Sixel`], and [`Graphics::KittyDirect`] materialize typed
	/// cells as ordinary blanks; [`Graphics::KittyPlaceholders`] uses Unicode
	/// placeholders.
	pub const fn set_graphics(&mut self, graphics: Graphics) {
		self.graphics = graphics;
	}

	/// Enables or disables DEC synchronized-output wrapping.
	///
	/// Wrapping is enabled by default to preserve the renderer's historical
	/// behavior. Capability detection should disable it for unsupported
	/// terminals.
	pub const fn set_sync_output(&mut self, enabled: bool) {
		self.sync_output = enabled;
	}

	/// Enables or disables moving cleared viewport content to native scrollback.
	///
	/// When enabled, a full viewport clear first emits Kitty's `CSI 22 J`
	/// extension. It is disabled by default.
	pub const fn set_screen_to_scrollback(&mut self, enabled: bool) {
		self.screen_to_scrollback = enabled;
	}

	/// Enables committing scrolled-out rows through a top-anchored DECSTBM
	/// region instead of a whole-screen scroll.
	///
	/// Screen rows below the region never move during a commit, and native
	/// scrollback receives exactly the same history as a whole-screen
	/// scroll. Whether a terminal-native text selection over the pinned
	/// rows survives is a separate, terminal-specific property: kitty and
	/// Alacritty transform selections correctly on region scrolls; ghostty,
	/// iTerm2, and xterm.js leave them anchored to pre-scroll storage rows,
	/// so they drift upward — matching what a whole-screen scroll does to
	/// selections over stationary live content repainted back into place;
	/// `WezTerm` clears them. Enable this only for terminals that move rows
	/// scrolled out of a top-anchored region into native scrollback (see
	/// `TerminalCaps::margin_scrollback`); it is disabled by default.
	pub const fn set_margin_scrollback(&mut self, enabled: bool) {
		self.margin_scrollback = enabled;
	}

	/// Enables or disables OSC 8 hyperlink materialization.
	///
	/// Link identities remain attached to frame cells while disabled, but output
	/// stays byte-for-byte identical to ordinary styled text.
	pub const fn set_hyperlinks(&mut self, enabled: bool) {
		self.hyperlinks = enabled;
	}

	/// Sets the terminal cell size used to scale sixel placements.
	///
	/// The default is 9 by 18 pixels per cell, matching pi's nominal terminal
	/// metrics. Detection code may override it before presentation.
	///
	/// # Errors
	///
	/// Rejects a zero pixel dimension.
	pub fn set_cell_pixel_size(&mut self, width: u16, height: u16) -> io::Result<()> {
		if width == 0 || height == 0 {
			return Err(io::Error::new(
				io::ErrorKind::InvalidInput,
				"cell pixel dimensions must be non-zero",
			));
		}
		self.cell_pixel_width = width;
		self.cell_pixel_height = height;
		Ok(())
	}

	/// Enables tmux DCS passthrough for Kitty and sixel graphics sequences.
	///
	/// Cursor movement, synchronized output, and ordinary text styling remain
	/// direct terminal output.
	pub const fn set_tmux_passthrough(&mut self, enabled: bool) {
		self.tmux_passthrough = enabled;
	}

	/// Paints a logical document with an immutable leading-row boundary.
	///
	/// The caller must disable terminal autowrap and keep terminal geometry
	/// fixed while the renderer is active; the renderer itself re-enables
	/// DECAWM transiently to join flagged soft-wrap boundaries (see
	/// [`Frame::set_soft_wrap`]) so native selection and scrollback copy
	/// them as one unbroken line. Advancing `stable_rows` is permanent,
	/// and committed history makes the document height a ratchet: between
	/// rebuilds the document may only grow, so transient rows (pickers, extra
	/// input lines) must be absorbed by the caller rather than shrinking the
	/// frame.
	///
	/// # Errors
	///
	/// Rejects zero or changed geometry, a retreating stable boundary, mutation
	/// within the prior stable prefix, or a document whose tail shrank below
	/// committed history. Writer failure poisons the renderer because its
	/// physical state is unknown.
	pub fn present(
		&mut self,
		next: Frame,
		viewport_height: u16,
		stable_rows: u16,
	) -> io::Result<PaintStats> {
		self.forget_preview();
		self.validate_input(&next, viewport_height, stable_rows)?;
		let stats = if self.previous.is_none() {
			self.initial_paint(next, viewport_height, stable_rows)?
		} else {
			let stats = self.paint_next(&next, viewport_height, stable_rows)?;
			self.previous = Some(next);
			stats
		};
		self.publish_debug_screen();
		Ok(stats)
	}

	/// [`Renderer::present`] without taking the frame: diffs against the
	/// retained previous frame, then `clone_from`s the borrowed one into
	/// it — reusing the existing cell allocation instead of copying a
	/// whole frame per paint. Cost is still O(grid) cell clones per call;
	/// retained callers that track their own damage should prefer
	/// [`Renderer::present_damaged`].
	///
	/// # Errors
	/// Same contract as [`Renderer::present`].
	pub fn present_ref(
		&mut self,
		next: &Frame,
		viewport_height: u16,
		stable_rows: u16,
	) -> io::Result<PaintStats> {
		self.forget_preview();
		self.validate_input(next, viewport_height, stable_rows)?;
		let stats = if self.previous.is_none() {
			self.initial_paint(next.clone(), viewport_height, stable_rows)?
		} else {
			let stats = self.paint_next(next, viewport_height, stable_rows)?;
			self
				.previous
				.as_mut()
				.expect("initial-paint branch checked previous above")
				.clone_from(next);
			stats
		};
		self.publish_debug_screen();
		Ok(stats)
	}

	/// Paints a damaged raw document with declarative viewport-anchored layers.
	///
	/// `damaged` follows [`Renderer::present_damaged`]. Layers composite only
	/// into the live viewport while history commits keep flowing: a row
	/// leaving the window is repainted from the raw document before it
	/// scrolls into native scrollback, so layer cells never reach history.
	/// Direct-drawn sixel, Kitty-direct, and iTerm2 images remain raw and are
	/// not occluded; Kitty placeholder cells participate in composition.
	///
	/// # Errors
	/// Same contract as [`Renderer::present`].
	pub fn present_overlaid(
		&mut self,
		next: &Frame,
		damaged: &[(u16, u16)],
		viewport_height: u16,
		stable_rows: u16,
		layers: &[Layer<'_>],
	) -> io::Result<PaintStats> {
		let viewport = Size::new(next.size().width, viewport_height);
		let resolved = resolve_layers(layers, viewport);
		self.present_resolved(next, damaged, viewport_height, stable_rows, &resolved)
	}

	/// Paints layers whose viewport bands have already been resolved.
	pub(crate) fn present_resolved(
		&mut self,
		next: &Frame,
		damaged: &[(u16, u16)],
		viewport_height: u16,
		stable_rows: u16,
		layers: &[ResolvedLayer<'_>],
	) -> io::Result<PaintStats> {
		self.forget_preview();
		self.validate_input(next, viewport_height, stable_rows)?;
		let stats = if self.previous.is_none() {
			self.initial_paint_overlaid(next.clone(), viewport_height, stable_rows, layers)?
		} else {
			self.validate_damaged_stable_prefix(next, damaged)?;
			let stats =
				self.paint_validated_next(next, viewport_height, stable_rows, Some(damaged), layers)?;
			let previous = self
				.previous
				.as_mut()
				.expect("initial-paint branch checked previous above");
			previous.resize_height(next.size().height, Style::default());
			for &(start, end) in damaged {
				for row in start..end.min(next.size().height) {
					previous.copy_row_from(next, row);
				}
			}
			previous.sync_soft_wraps(next);
			stats
		};
		self.publish_debug_screen();
		Ok(stats)
	}

	/// [`Renderer::present_ref`] with a caller-supplied damage list: only rows
	/// inside `damaged` `(start, end)` ranges are validated and snapshotted.
	/// The caller guarantees every changed row is covered; the full grid is
	/// copied only on the initial paint.
	///
	/// # Errors
	/// Same contract as [`Renderer::present`].
	pub fn present_damaged(
		&mut self,
		next: &Frame,
		damaged: &[(u16, u16)],
		viewport_height: u16,
		stable_rows: u16,
	) -> io::Result<PaintStats> {
		self.present_resolved(next, damaged, viewport_height, stable_rows, &[])
	}

	/// Repaints every composited viewport-layer band from the raw document
	/// and drops the stored layers.
	///
	/// The final inline screen persists into native scrollback once the
	/// host exits and the shell resumes scrolling, so teardown must not
	/// leave layer cells composited — [`crate::App`] does this
	/// automatically, and manual hosts call it before dropping their
	/// [`crate::Terminal`]. Call it on the main screen (release any
	/// alternate-screen hold first); with no stored layers, or while the
	/// alternate screen is active, nothing is written.
	///
	/// # Errors
	/// Propagates writer failures, which poison the renderer.
	pub fn clear_layers(&mut self) -> io::Result<()> {
		if self.poisoned || self.layers.is_empty() || crate::terminal::alt_screen_active() {
			return Ok(());
		}
		if self.previous.is_none() {
			self.layers.clear();
			return Ok(());
		}
		self.sync_screen_buffer();
		let layers = std::mem::take(&mut self.layers);
		let window = Window { top: self.window_top, height: self.viewport_height };
		let mut stats = PaintStats::default();
		let (output, next_cursor) = {
			let previous = self
				.previous
				.as_ref()
				.expect("layer-clearing checked previous above");
			let previous_view = ComposedFrame { base: previous, layers: &layers };
			let next_view = ComposedFrame { base: previous, layers: &[] };
			let mut paint = String::new();
			emit_window_diff(
				&mut paint,
				&previous_view,
				window,
				&next_view,
				window,
				0,
				self.viewport_height,
				self.graphics,
				self.hyperlinks,
				&mut stats,
			);
			let next_cursor = frame_cursor(previous, window);
			let mut output = String::with_capacity(paint.len().saturating_add(64));
			if stats.runs > 0 || next_cursor != self.cursor {
				if self.sync_output {
					output.push_str(SYNC_OUTPUT_BEGIN);
				}
				output.push_str(HIDE_CURSOR);
				output.push_str(VIEWPORT_BOTTOM);
				output.push_str(&paint);
				place_cursor(&mut output, next_cursor, self.viewport_height);
				if self.sync_output {
					output.push_str(SYNC_OUTPUT_END);
				}
			}
			(output, next_cursor)
		};
		self.write(&output)?;
		self.cursor = next_cursor;
		Ok(())
	}

	fn paint_next(
		&mut self,
		next: &Frame,
		viewport_height: u16,
		stable_rows: u16,
	) -> io::Result<PaintStats> {
		self.validate_stable_prefix(next)?;
		self.paint_validated_next(next, viewport_height, stable_rows, None, &[])
	}

	fn validate_stable_prefix(&self, next: &Frame) -> io::Result<()> {
		let previous = self
			.previous
			.as_ref()
			.expect("callers checked previous before painting");
		if (0..self.stable_rows).any(|row| !previous.row_equals(row, next, row)) {
			return Err(Self::stable_mutation_error());
		}
		Ok(())
	}

	fn validate_damaged_stable_prefix(
		&self,
		next: &Frame,
		damaged: &[(u16, u16)],
	) -> io::Result<()> {
		let previous = self
			.previous
			.as_ref()
			.expect("callers checked previous before painting");
		for &(start, end) in damaged {
			let end = end.min(self.stable_rows);
			if (start.min(end)..end).any(|row| !previous.row_equals(row, next, row)) {
				return Err(Self::stable_mutation_error());
			}
		}
		Ok(())
	}

	fn stable_mutation_error() -> io::Error {
		contract_error("a previously declared-stable row changed; native history was left untouched")
	}

	fn paint_validated_next(
		&mut self,
		next: &Frame,
		viewport_height: u16,
		stable_rows: u16,
		damaged: Option<&[(u16, u16)]>,
		layers: &[ResolvedLayer<'_>],
	) -> io::Result<PaintStats> {
		self.sync_screen_buffer();
		let image_prefix = self.image_prefix(next, layers);
		let mut output = std::mem::take(&mut self.output_scratch);
		output.clear();
		output.push_str(&image_prefix);
		self.prepare_sixels(next);
		let previous = self
			.previous
			.as_ref()
			.expect("callers checked previous before painting");
		let layout = layout(next.size().height, viewport_height, stable_rows, self.committed_rows);
		let previous_window = Window { top: self.window_top, height: viewport_height };
		let next_window = Window { top: layout.window_top, height: viewport_height };
		let mut incoming = std::mem::take(&mut self.layer_scratch);
		store_layers_into(layers, next_window, next.size().width, &mut incoming);
		let commit_to =
			scroll_append_to(previous_window, next_window, self.committed_rows, layout.stable_limit);
		let newly_committed = commit_to - self.committed_rows;
		let margin_rows = if self.margin_scrollback {
			stable_rows
				.saturating_sub(layout.window_top)
				.max(newly_committed)
				.max(2)
		} else {
			viewport_height
		};
		let mut stats = PaintStats {
			committed_rows: newly_committed,
			clipped_rows: layout.window_top.saturating_sub(commit_to),
			..PaintStats::default()
		};
		// Direct-drawn image protocols consume the raw document. Only Kitty
		// placeholder cells are occluded by the composed cell view below.
		let sixels = self.sixel_output(
			next,
			next_window,
			Some((previous, previous_window)),
			damaged,
			previous_window.top != next_window.top,
		);
		let kitty_direct = kitty_direct_output(
			self.graphics,
			&mut self.images,
			next,
			next_window,
			Some((previous, previous_window)),
			damaged,
			false,
			self.cell_pixel_width,
			self.cell_pixel_height,
			self.tmux_passthrough,
		);
		let iterm2 = iterm2_output(
			self.graphics,
			self
				.images
				.iter()
				.map(|(&id, image)| Iterm2Image { id, png: &image.png }),
			next,
			Iterm2Viewport { top: next_window.top, height: next_window.height },
			Some((previous, Iterm2Viewport {
				top:    previous_window.top,
				height: previous_window.height,
			})),
			damaged,
			false,
			self.tmux_passthrough,
		);

		let dirty_rows = damaged.and_then(|damaged| {
			(previous_window.top == next_window.top && previous_window.height == next_window.height)
				.then(|| changed_screen_rows(damaged, &self.layers, &incoming, next_window))
		});
		let previous_view = ComposedFrame { base: previous, layers: &self.layers };
		let next_view = ComposedFrame { base: next, layers: &incoming };
		let capacity = usize::from(next.size().width).saturating_mul(usize::from(viewport_height));
		let mut paint = std::mem::take(&mut self.paint_scratch);
		paint.clear();
		paint.reserve(capacity);
		if newly_committed > 0 {
			// Scroll only the visible stable rows through a DECSTBM region so
			// the live window below stays physically pinned. The region must
			// cover the scrolled rows and span the two rows DECSTBM requires;
			// a seam at or below the screen bottom leaves nothing to pin and
			// falls back to the whole-screen scroll.
			if margin_rows < viewport_height {
				emit_margin_scroll_append(
					&mut paint,
					&previous_view,
					previous_window,
					&next_view,
					next_window,
					margin_rows,
					self.graphics,
					self.hyperlinks,
					&mut stats,
				);
			} else {
				emit_scroll_append(
					&mut paint,
					&previous_view,
					previous_window,
					&next_view,
					next_window,
					self.graphics,
					self.hyperlinks,
					&mut stats,
				);
			}
		} else {
			emit_window_diff_rows(
				&mut paint,
				&previous_view,
				previous_window,
				&next_view,
				next_window,
				0,
				viewport_height,
				dirty_rows.as_deref(),
				self.graphics,
				self.hyperlinks,
				&mut stats,
			);
		}
		// Wrap-boundary metadata has no in-place VT rewrite: boundaries
		// whose hard/soft state changed are re-emitted surgically — never
		// via a viewport clear, which scrollback-pushing terminals would
		// turn into duplicated history.
		reconcile_wrap_boundaries(
			&mut paint,
			&previous_view,
			previous_window,
			&next_view,
			next_window,
			newly_committed,
			margin_rows.min(viewport_height),
			self.graphics,
			self.hyperlinks,
			&mut stats,
		);

		let next_cursor = compose_cursor(next, &incoming, next_window, next.size().width);
		output.reserve(
			paint
				.len()
				.saturating_add(sixels.len())
				.saturating_add(kitty_direct.len())
				.saturating_add(iterm2.len())
				.saturating_add(64),
		);
		if stats.runs > 0
			|| !sixels.is_empty()
			|| !kitty_direct.is_empty()
			|| !iterm2.is_empty()
			|| next_cursor != self.cursor
		{
			if self.sync_output {
				output.push_str(SYNC_OUTPUT_BEGIN);
			}
			output.push_str(HIDE_CURSOR);
			output.push_str(VIEWPORT_BOTTOM);
			output.push_str(&paint);
			output.push_str(&sixels);
			output.push_str(&kitty_direct);
			output.push_str(&iterm2);
			place_cursor(&mut output, next_cursor, viewport_height);
			if self.sync_output {
				output.push_str(SYNC_OUTPUT_END);
			}
		}
		let bytes = output.len();
		let write_result = self.write(&output);
		self.paint_scratch = paint;
		self.output_scratch = output;
		write_result?;

		self.window_top = layout.window_top;
		self.committed_rows = commit_to;
		self.stable_rows = stable_rows;
		self.cursor = next_cursor;
		stats.bytes = bytes;
		let previous_layers = std::mem::replace(&mut self.layers, incoming);
		self.layer_scratch = previous_layers;
		Ok(stats)
	}

	/// Paints only the current raw document tail without changing committed
	/// state.
	///
	/// Resize handlers use this on an alternate buffer while normal-buffer
	/// history remains untouched. Stored overlay layers are deliberately
	/// ignored; `leading_sequence` is emitted inside the synchronized update,
	/// before the viewport paint. Overlays go through
	/// [`Renderer::preview_overlaid`].
	///
	/// # Errors
	///
	/// Rejects zero geometry. Writer failure poisons the renderer because its
	/// physical state is unknown.
	pub fn preview(
		&mut self,
		next: &Frame,
		viewport_height: u16,
		leading_sequence: &str,
	) -> io::Result<PaintStats> {
		self.preview_resolved(next, &[], viewport_height, leading_sequence)
	}

	/// [`Renderer::preview`] with declarative viewport-anchored layers.
	///
	/// The document tail and every visible layer composite into one throwaway
	/// synchronized paint while committed history and stored layers stay
	/// untouched. Alternate-screen holders — fullscreen scenes and modal
	/// overlays — repaint with this on damage or geometry change;
	/// [`Renderer::present_overlaid`] is the normal-buffer counterpart.
	///
	/// # Errors
	///
	/// Same contract as [`Renderer::preview`].
	pub fn preview_overlaid(
		&mut self,
		next: &Frame,
		layers: &[Layer<'_>],
		viewport_height: u16,
		leading_sequence: &str,
	) -> io::Result<PaintStats> {
		let viewport = Size::new(next.size().width, viewport_height);
		let resolved = resolve_layers(layers, viewport);
		self.preview_resolved(next, &resolved, viewport_height, leading_sequence)
	}

	/// Paints the viewport with pre-resolved layer bands, state-isolated.
	pub(crate) fn preview_resolved(
		&mut self,
		next: &Frame,
		layers: &[ResolvedLayer<'_>],
		viewport_height: u16,
		leading_sequence: &str,
	) -> io::Result<PaintStats> {
		self.validate_frame(next, viewport_height)?;
		if !leading_sequence.is_empty() {
			self.forget_preview();
		}
		self.sync_screen_buffer();

		let paint_cells = usize::from(next.size().width).saturating_mul(usize::from(viewport_height));
		let window = Window {
			top:    next.size().height.saturating_sub(viewport_height),
			height: viewport_height,
		};
		let can_diff = leading_sequence.is_empty()
			&& self.preview_window == Some(window)
			&& self
				.preview_previous
				.as_ref()
				.is_some_and(|previous| previous.size() == next.size());
		let composited = store_layers(layers, window, next.size().width);
		let images = self.image_prefix(next, layers);
		self.prepare_sixels(next);
		let sixels = self.sixel_output(next, window, None, None, true);
		let kitty_direct = kitty_direct_output(
			self.graphics,
			&mut self.images,
			next,
			window,
			None,
			None,
			true,
			self.cell_pixel_width,
			self.cell_pixel_height,
			self.tmux_passthrough,
		);
		let iterm2 = iterm2_output(
			self.graphics,
			self
				.images
				.iter()
				.map(|(&id, image)| Iterm2Image { id, png: &image.png }),
			next,
			Iterm2Viewport { top: window.top, height: window.height },
			None,
			None,
			true,
			self.tmux_passthrough,
		);
		let raw = ComposedFrame { base: next, layers: &composited };
		let cursor = compose_cursor(next, &composited, window, next.size().width);
		let mut stats = PaintStats::default();
		let mut paint = String::new();
		if can_diff {
			let previous = ComposedFrame {
				base:   self
					.preview_previous
					.as_ref()
					.expect("preview geometry checked above"),
				layers: &self.preview_layers,
			};
			emit_window_diff(
				&mut paint,
				&previous,
				window,
				&raw,
				window,
				0,
				viewport_height,
				self.graphics,
				self.hyperlinks,
				&mut stats,
			);
		}

		let auxiliary =
			!images.is_empty() || !sixels.is_empty() || !kitty_direct.is_empty() || !iterm2.is_empty();
		let full_repaint = !can_diff;
		let mut output = String::with_capacity(
			if full_repaint {
				paint_cells.saturating_mul(2)
			} else {
				paint.len()
			}
			.saturating_add(images.len())
			.saturating_add(sixels.len())
			.saturating_add(kitty_direct.len())
			.saturating_add(iterm2.len())
			.saturating_add(64),
		);
		if full_repaint {
			if self.sync_output {
				output.push_str(SYNC_OUTPUT_BEGIN);
			}
			output.push_str(HIDE_CURSOR);
			output.push_str(leading_sequence);
			// Kitty traffic must follow the staged buffer switch: per-screen
			// image stores only keep bytes transmitted on the active screen.
			output.push_str(&images);
			output.push_str(RESET_STYLE);
			output.push_str(esc!(cursor_home));
			emit_rows(&mut output, &raw, 0..0, window, self.graphics, self.hyperlinks);
			output.push_str(RESET_STYLE);
			output.push('\r');
			output.push_str(&sixels);
			output.push_str(&kitty_direct);
			output.push_str(&iterm2);
			place_cursor(&mut output, cursor, viewport_height);
			if self.sync_output {
				output.push_str(SYNC_OUTPUT_END);
			}
			stats.full_repaint = true;
			stats.changed_cells = paint_cells;
			stats.runs = usize::from(viewport_height);
		} else if stats.runs > 0 || cursor != self.preview_cursor || auxiliary {
			if self.sync_output {
				output.push_str(SYNC_OUTPUT_BEGIN);
			}
			output.push_str(HIDE_CURSOR);
			output.push_str(&images);
			output.push_str(VIEWPORT_BOTTOM);
			output.push_str(&paint);
			output.push_str(&sixels);
			output.push_str(&kitty_direct);
			output.push_str(&iterm2);
			place_cursor(&mut output, cursor, viewport_height);
			if self.sync_output {
				output.push_str(SYNC_OUTPUT_END);
			}
		}

		stats.bytes = output.len();
		self.write(&output)?;
		if crate::debug::publishing() {
			// A preview is what the terminal shows right now (alternate
			// screen or drag frame); publish its composition, not the
			// committed main-screen model.
			crate::debug::publish_screen(crate::debug::ScreenSnapshot {
				lines:      stored_text(next, &composited, window.top, viewport_height),
				cursor:     cursor.map(|cursor| (cursor.row, cursor.col)),
				window_top: window.top,
				cols:       next.size().width,
				rows:       viewport_height,
				doc_height: next.size().height,
				overlay:    !composited.is_empty(),
			});
		}
		match &mut self.preview_previous {
			Some(previous) => previous.clone_from(next),
			None => self.preview_previous = Some(next.clone()),
		}
		self.preview_layers = composited;
		self.preview_window = Some(window);
		self.preview_cursor = cursor;
		Ok(stats)
	}

	/// Clears and reconstructs native history at new terminal geometry.
	///
	/// The synchronized update emits `leading_sequence`, clears scrollback once,
	/// then writes the stable prefix and current viewport. The reconstructed
	/// frame becomes the baseline for subsequent [`Self::present`] calls.
	///
	/// # Errors
	///
	/// Rejects zero geometry or a stable boundary beyond the document. Writer
	/// failure poisons the renderer because history may be partially rebuilt.
	pub fn rebuild(
		&mut self,
		next: Frame,
		viewport_height: u16,
		stable_rows: u16,
		leading_sequence: &str,
	) -> io::Result<PaintStats> {
		self.forget_preview();
		self.validate_frame(&next, viewport_height)?;
		if stable_rows > next.size().height {
			return Err(contract_error("stable_rows exceeds the document height"));
		}
		let stats =
			self.full_paint(next, viewport_height, stable_rows, leading_sequence, REBUILD_HISTORY)?;
		self.publish_debug_screen();
		Ok(stats)
	}

	/// Publishes the committed screen to the shared debug snapshot when a
	/// stream-served `OMP_TUI_DEBUG` host is listening; no-op otherwise.
	fn publish_debug_screen(&self) {
		if !crate::debug::publishing() {
			return;
		}
		let Some(previous) = &self.previous else {
			return;
		};
		crate::debug::publish_screen(crate::debug::ScreenSnapshot {
			lines:      self.screen_text(),
			cursor:     self.screen_cursor(),
			window_top: self.window_top,
			cols:       previous.size().width,
			rows:       self.viewport_height,
			doc_height: previous.size().height,
			overlay:    !self.layers.is_empty(),
		});
	}

	/// Returns the number of finalized rows physically stored above the
	/// viewport.
	pub const fn committed_rows(&self) -> u16 {
		self.committed_rows
	}

	/// Returns the document row currently shown at the viewport top.
	pub const fn window_top(&self) -> u16 {
		self.window_top
	}

	/// Renders the retained physical screen model — the committed frame
	/// composed with its stored viewport layers — as visible text, one
	/// right-trimmed string per viewport row.
	///
	/// This is what the terminal currently shows, driving the `OMP_TUI_DEBUG`
	/// `text` op. Empty before the first present or rebuild.
	pub fn screen_text(&self) -> Vec<String> {
		match &self.previous {
			Some(previous) => {
				stored_text(previous, &self.layers, self.window_top, self.viewport_height)
			},
			None => Vec::new(),
		}
	}

	/// Screen coordinates (row, column) of the visible hardware cursor, when
	/// one was placed by the last present.
	pub const fn screen_cursor(&self) -> Option<(u16, u16)> {
		match self.cursor {
			Some(cursor) => Some((cursor.row, cursor.col)),
			None => None,
		}
	}

	/// Returns whether terminal output is connected or was abandoned after its
	/// unflushed backlog crossed the safety limit.
	pub const fn output_state(&self) -> OutputState {
		self.output_state
	}

	/// Borrows the output writer for terminal session teardown.
	pub const fn writer_mut(&mut self) -> &mut W {
		&mut self.writer
	}

	/// Returns the output writer after the renderer is no longer needed.
	pub fn into_inner(self) -> W {
		self.writer
	}

	fn validate_frame(&self, next: &Frame, viewport_height: u16) -> io::Result<()> {
		if self.poisoned {
			return Err(io::Error::other(
				"renderer state is unknown after a partial write; restart the terminal session",
			));
		}
		if next.size().width == 0 || viewport_height == 0 {
			return Err(io::Error::new(
				io::ErrorKind::InvalidInput,
				"document width and viewport height must be non-zero",
			));
		}
		Ok(())
	}

	fn validate_input(
		&self,
		next: &Frame,
		viewport_height: u16,
		stable_rows: u16,
	) -> io::Result<()> {
		self.validate_frame(next, viewport_height)?;
		if stable_rows > next.size().height {
			return Err(contract_error("stable_rows exceeds the document height"));
		}
		if stable_rows < self.stable_rows {
			return Err(contract_error("stable_rows cannot retreat"));
		}
		if next.size().height < self.committed_rows {
			return Err(contract_error(
				"document is shorter than rows already committed to native history",
			));
		}
		if next.size().height.saturating_sub(viewport_height) < self.committed_rows {
			return Err(contract_error(
				"document tail shrank below committed history; document height must stay monotonic \
				 between rebuilds",
			));
		}
		if let Some(previous) = &self.previous
			&& (previous.size().width != next.size().width || self.viewport_height != viewport_height)
		{
			return Err(io::Error::new(
				io::ErrorKind::InvalidInput,
				"terminal geometry changed; preserving native history requires a new renderer session",
			));
		}
		Ok(())
	}

	fn initial_paint(
		&mut self,
		next: Frame,
		viewport_height: u16,
		stable_rows: u16,
	) -> io::Result<PaintStats> {
		self.full_paint(next, viewport_height, stable_rows, "", CLEAR_VIEWPORT)
	}

	fn initial_paint_overlaid(
		&mut self,
		next: Frame,
		viewport_height: u16,
		stable_rows: u16,
		layers: &[ResolvedLayer<'_>],
	) -> io::Result<PaintStats> {
		self.paint_full(next, viewport_height, stable_rows, "", CLEAR_VIEWPORT, layers)
	}

	fn full_paint(
		&mut self,
		next: Frame,
		viewport_height: u16,
		stable_rows: u16,
		leading_sequence: &str,
		clear_sequence: &str,
	) -> io::Result<PaintStats> {
		self.paint_full(next, viewport_height, stable_rows, leading_sequence, clear_sequence, &[])
	}

	fn paint_full(
		&mut self,
		next: Frame,
		viewport_height: u16,
		stable_rows: u16,
		leading_sequence: &str,
		clear_sequence: &str,
		layers: &[ResolvedLayer<'_>],
	) -> io::Result<PaintStats> {
		let layout = layout(next.size().height, viewport_height, stable_rows, 0);
		let paint_rows = layout.stable_limit.saturating_add(viewport_height);
		let paint_cells = usize::from(next.size().width).saturating_mul(usize::from(paint_rows));
		let window = Window { top: layout.window_top, height: viewport_height };
		let stored_layers = store_layers(layers, window, next.size().width);
		let next_cursor = compose_cursor(&next, &stored_layers, window, next.size().width);
		self.sync_screen_buffer();
		let images = self.image_prefix(&next, layers);
		self.prepare_sixels(&next);
		let sixels = self.sixel_output(&next, window, None, None, true);
		let kitty_direct = kitty_direct_output(
			self.graphics,
			&mut self.images,
			&next,
			window,
			None,
			None,
			true,
			self.cell_pixel_width,
			self.cell_pixel_height,
			self.tmux_passthrough,
		);
		let iterm2 = iterm2_output(
			self.graphics,
			self
				.images
				.iter()
				.map(|(&id, image)| Iterm2Image { id, png: &image.png }),
			&next,
			Iterm2Viewport { top: window.top, height: window.height },
			None,
			None,
			true,
			self.tmux_passthrough,
		);
		let mut output = String::with_capacity(
			paint_cells
				.saturating_mul(2)
				.saturating_add(images.len())
				.saturating_add(sixels.len())
				.saturating_add(kitty_direct.len())
				.saturating_add(iterm2.len()),
		);
		if self.sync_output {
			output.push_str(SYNC_OUTPUT_BEGIN);
		}
		output.push_str(HIDE_CURSOR);
		output.push_str(leading_sequence);
		output.push_str(RESET_STYLE);
		if self.screen_to_scrollback && clear_sequence == CLEAR_VIEWPORT {
			output.push_str(SCREEN_TO_SCROLLBACK);
		}
		output.push_str(clear_sequence);
		// Kitty traffic must follow both the staged buffer switch (per-screen
		// image stores only keep what arrives on the active screen) and the
		// clear, which may drop placements on some implementations.
		output.push_str(&images);
		let composed = ComposedFrame { base: &next, layers: &stored_layers };
		emit_rows(
			&mut output,
			&composed,
			0..layout.stable_limit,
			window,
			self.graphics,
			self.hyperlinks,
		);
		output.push_str(RESET_STYLE);
		output.push('\r');
		output.push_str(&sixels);
		output.push_str(&kitty_direct);
		output.push_str(&iterm2);
		place_cursor(&mut output, next_cursor, viewport_height);
		if self.sync_output {
			output.push_str(SYNC_OUTPUT_END);
		}

		let bytes = output.len();
		self.write(&output)?;
		let stats = PaintStats {
			full_repaint: true,
			changed_cells: paint_cells,
			runs: usize::from(paint_rows),
			committed_rows: layout.stable_limit,
			clipped_rows: layout.window_top.saturating_sub(layout.stable_limit),
			bytes,
		};
		self.previous = Some(next);
		self.viewport_height = viewport_height;
		self.window_top = layout.window_top;
		self.committed_rows = layout.stable_limit;
		self.stable_rows = stable_rows;
		self.cursor = next_cursor;
		self.layers = stored_layers;
		Ok(stats)
	}

	fn forget_preview(&mut self) {
		self.preview_previous = None;
		self.preview_layers.clear();
		self.preview_window = None;
		self.preview_cursor = None;
	}

	/// Reconciles graphics caches with the terminal's current screen buffer.
	fn sync_screen_buffer(&mut self) {
		self.set_screen_buffer(crate::terminal::alt_screen_active());
	}

	/// Records which screen buffer subsequent paints target.
	///
	/// A change drops all terminal-side Kitty graphics state — transmissions,
	/// virtual placements, direct placements — because terminals with
	/// per-screen image storage (ghostty) do not share them between the main
	/// and alternate buffers; the next paint retransmits and re-places.
	fn set_screen_buffer(&mut self, alt_screen: bool) {
		if alt_screen == self.alt_screen {
			return;
		}
		self.forget_preview();
		self.alt_screen = alt_screen;
		for image in self.images.values_mut() {
			image.uploaded = false;
			image.placed.clear();
			image.direct_visible = false;
		}
	}

	/// Emits Kitty transmissions and virtual placements for every image
	/// referenced by the document or by a composited overlay layer band.
	///
	/// Each distinct cell box of an image gets its own placement, keyed by
	/// [`crate::kitty::placement_id`], so repeated sizes replace instead of
	/// accumulating and placeholder cells always resolve their exact grid.
	/// IDs unknown to [`Renderer::register_image`] are resolved from the
	/// process-wide `<img src>` registry.
	fn image_prefix(&mut self, frame: &Frame, layers: &[ResolvedLayer<'_>]) -> String {
		if self.graphics != Graphics::KittyPlaceholders
			|| (!frame.may_have_images() && layers.iter().all(|layer| !layer.frame.may_have_images()))
		{
			return String::new();
		}
		let mut needed: SmallVec<(u32, u16, u16), 8> = SmallVec::new();
		let mut collect = |frame: &Frame, y0: u16, y1: u16| {
			for y in y0..y1.min(frame.size().height) {
				for x in 0..frame.size().width {
					if let CellContent::Image { id, rows, cols, .. } = frame.cell(x, y).content
						&& rows > 0 && cols > 0
						&& !needed.contains(&(id, rows, cols))
					{
						needed.push((id, rows, cols));
					}
				}
			}
		};
		collect(frame, 0, frame.size().height);
		for layer in layers {
			collect(layer.frame, layer.src_top, layer.src_top.saturating_add(layer.rows));
		}
		let mut output = String::new();
		for (id, rows, cols) in needed {
			let image = match self.images.entry(id) {
				std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
				std::collections::btree_map::Entry::Vacant(entry) => {
					let Some(png) = crate::imagereg::bytes(id) else {
						continue;
					};
					entry.insert(RegisteredImage::new(png))
				},
			};
			if !image.uploaded {
				append_transmission(&mut output, id, &image.png, self.tmux_passthrough);
				image.uploaded = true;
			}
			if !image.placed.contains(&(rows, cols)) {
				append_placement(&mut output, id, rows, cols, self.tmux_passthrough);
				image.placed.push((rows, cols));
			}
		}
		output
	}

	fn prepare_sixels(&mut self, frame: &Frame) {
		if self.graphics != Graphics::Sixel {
			return;
		}
		for y in 0..frame.size().height {
			for x in 0..frame.size().width {
				let CellContent::Image { id, .. } = frame.cell(x, y).content else {
					continue;
				};
				let Some(image) = self.images.get_mut(&id) else {
					continue;
				};
				if !image.sixel_decoded {
					image.sixel = SixelImage::from_png(&image.png);
					image.sixel_decoded = true;
				}
			}
		}
	}

	fn sixel_output(
		&self,
		frame: &Frame,
		window: Window,
		previous: Option<(&Frame, Window)>,
		damaged: Option<&[(u16, u16)]>,
		force: bool,
	) -> String {
		if self.graphics != Graphics::Sixel {
			return String::new();
		}
		let mut output = String::new();
		let mut cursor_row = window.height - 1;
		for (&id, registered) in &self.images {
			let Some(image) = &registered.sixel else {
				continue;
			};
			let Some((top, left, rows, cols)) = image_placement(frame, id) else {
				continue;
			};
			let visible_top = top.max(window.top);
			let visible_bottom = top
				.saturating_add(rows)
				.min(window.top.saturating_add(window.height))
				.min(frame.size().height);
			if visible_top >= visible_bottom {
				continue;
			}
			let needs_emit = force
				|| match damaged {
					Some(ranges) => ranges
						.iter()
						.any(|&(start, end)| start < visible_bottom && end > visible_top),
					None => match previous {
						None => true,
						Some((previous, previous_window)) => {
							previous_window.top != window.top
								|| (visible_top..visible_bottom)
									.any(|row| !previous.row_equals(row, frame, row))
						},
					},
				};
			if !needs_emit {
				continue;
			}
			let target_width = usize::from(cols).saturating_mul(usize::from(self.cell_pixel_width));
			let target_height = usize::from(rows).saturating_mul(usize::from(self.cell_pixel_height));
			let y0 = usize::from(visible_top - top).saturating_mul(target_height) / usize::from(rows);
			let y1 =
				usize::from(visible_bottom - top).saturating_mul(target_height) / usize::from(rows);
			let sixel = image.encode_band(target_width, target_height, y0, y1);
			if sixel.is_empty() {
				continue;
			}
			move_cursor_row(&mut output, &mut cursor_row, visible_top - window.top);
			output.push('\r');
			if left > 0 {
				let _ = write!(output, esc!(cursor_forward), left);
			}
			if self.tmux_passthrough {
				append_tmux_passthrough(&mut output, &sixel);
			} else {
				output.push_str(&sixel);
			}
		}
		if !output.is_empty() {
			move_cursor_row(&mut output, &mut cursor_row, window.height - 1);
			output.push('\r');
		}
		output
	}

	fn write(&mut self, output: &str) -> io::Result<()> {
		if output.is_empty() {
			return Ok(());
		}
		if self.output_state == OutputState::Disconnected || self.backlog.queue(output.len()) {
			self.output_state = OutputState::Disconnected;
			self.poisoned = true;
			return Err(io::Error::new(
				io::ErrorKind::BrokenPipe,
				"terminal output backlog exceeded 64 MiB; terminal is disconnected",
			));
		}
		let result = self
			.write_output(output.as_bytes())
			.and_then(|()| self.writer.flush());
		if let Err(error) = result {
			self.poisoned = true;
			return Err(error);
		}
		self.backlog.flushed();
		Ok(())
	}

	fn write_output(&mut self, output: &[u8]) -> io::Result<()> {
		#[cfg(any(windows, target_os = "linux"))]
		if self.conpty_hosted && output.len() > MAX_CONPTY_WRITE_CHUNK_BYTES {
			for chunk in ConptyChunks::new(output, MAX_CONPTY_WRITE_CHUNK_BYTES) {
				terminal_write_all(&mut self.writer, chunk)?;
			}
			return Ok(());
		}
		terminal_write_all(&mut self.writer, output)
	}
}

#[cfg(any(windows, target_os = "linux", test))]
struct ConptyChunks<'a> {
	bytes: &'a [u8],
	pos:   usize,
	max:   usize,
}

#[cfg(any(windows, target_os = "linux", test))]
impl<'a> ConptyChunks<'a> {
	fn new(bytes: &'a [u8], max: usize) -> Self {
		debug_assert!(max > 0);
		Self { bytes, pos: 0, max }
	}
}

#[cfg(any(windows, target_os = "linux", test))]
impl<'a> Iterator for ConptyChunks<'a> {
	type Item = &'a [u8];

	fn next(&mut self) -> Option<Self::Item> {
		if self.pos == self.bytes.len() {
			return None;
		}
		let start = self.pos;
		if self.bytes.len() - start <= self.max {
			self.pos = self.bytes.len();
			return Some(&self.bytes[start..]);
		}

		let mut window_end = start + self.max;
		while self.bytes[window_end] & 0xc0 == 0x80 {
			window_end -= 1;
		}
		let mut search_end = window_end;
		let cut = loop {
			let newline = self.bytes[start..search_end]
				.iter()
				.rposition(|byte| *byte == b'\n')
				.map(|index| start + index + 1);
			let Some(newline) = newline else {
				break escape_end_crossing(self.bytes, start, window_end).unwrap_or(window_end);
			};
			if escape_end_crossing(self.bytes, start, newline).is_none() {
				break newline;
			}
			search_end = newline - 1;
		};
		self.pos = cut;
		Some(&self.bytes[start..cut])
	}
}

#[cfg(any(windows, target_os = "linux", test))]
fn escape_end_crossing(bytes: &[u8], start: usize, cut: usize) -> Option<usize> {
	let mut index = start;
	while index < cut {
		if bytes[index] != b'\x1b' {
			index += 1;
			continue;
		}
		let end = escape_sequence_end(bytes, index);
		if end > cut {
			return Some(end);
		}
		index = end.max(index + 1);
	}
	None
}

#[cfg(any(windows, target_os = "linux", test))]
fn escape_sequence_end(bytes: &[u8], start: usize) -> usize {
	let Some(&kind) = bytes.get(start + 1) else {
		return bytes.len();
	};
	match kind {
		b'[' => {
			for (offset, byte) in bytes[start + 2..].iter().enumerate() {
				if (0x40..=0x7e).contains(byte) {
					return start + 3 + offset;
				}
			}
			bytes.len()
		},
		b']' => string_escape_end(bytes, start + 2, true),
		b'P' | b'X' | b'^' | b'_' => string_escape_end(bytes, start + 2, false),
		0x20..=0x2f => {
			for (offset, byte) in bytes[start + 2..].iter().enumerate() {
				if (0x30..=0x7e).contains(byte) {
					return start + 3 + offset;
				}
			}
			bytes.len()
		},
		_ => (start + 2).min(bytes.len()),
	}
}

#[cfg(any(windows, target_os = "linux", test))]
fn string_escape_end(bytes: &[u8], start: usize, bell_terminated: bool) -> usize {
	let mut index = start;
	while index < bytes.len() {
		if bell_terminated && bytes[index] == b'\x07' {
			return index + 1;
		}
		if bytes[index] == b'\x1b' && bytes.get(index + 1) == Some(&b'\\') {
			return index + 2;
		}
		index += 1;
	}
	bytes.len()
}

#[cfg(windows)]
const fn is_conpty_hosted() -> bool {
	true
}

#[cfg(target_os = "linux")]
fn is_conpty_hosted() -> bool {
	std::env::var_os("WSL_DISTRO_NAME").is_some() || std::env::var_os("WSL_INTEROP").is_some()
}

#[allow(clippy::too_many_arguments, reason = "rendering inputs are independent frame state")]
fn kitty_direct_output(
	graphics: Graphics,
	images: &mut BTreeMap<u32, RegisteredImage>,
	frame: &Frame,
	window: Window,
	previous: Option<(&Frame, Window)>,
	damaged: Option<&[(u16, u16)]>,
	force: bool,
	cell_pixel_width: u16,
	cell_pixel_height: u16,
	tmux_passthrough: bool,
) -> String {
	if graphics != Graphics::KittyDirect {
		return String::new();
	}
	let mut output = String::new();
	let mut cursor_row = window.height - 1;
	for (&id, image) in images {
		let placement = image_placement(frame, id);
		let visible = placement.and_then(|(top, left, rows, cols)| {
			let visible_top = top.max(window.top);
			let visible_bottom = top
				.saturating_add(rows)
				.min(window.top.saturating_add(window.height))
				.min(frame.size().height);
			(visible_top < visible_bottom).then_some((
				top,
				left,
				rows,
				cols,
				visible_top,
				visible_bottom,
			))
		});
		let Some((top, left, rows, cols, visible_top, visible_bottom)) = visible else {
			if image.direct_visible {
				append_delete_image(&mut output, id, tmux_passthrough);
				image.uploaded = false;
				image.direct_visible = false;
			}
			continue;
		};

		let moved = previous.is_none_or(|(previous_frame, previous_window)| {
			image_placement(previous_frame, id) != placement || previous_window.top != window.top
		});
		let intersects_damage = damaged.is_some_and(|ranges| {
			ranges
				.iter()
				.any(|&(start, end)| start < visible_bottom && end > visible_top)
		});
		let changed = damaged.is_none()
			&& previous.is_some_and(|(previous_frame, _)| {
				(visible_top..visible_bottom).any(|row| !previous_frame.row_equals(row, frame, row))
			});
		let needs_emit =
			force || !image.uploaded || !image.direct_visible || moved || intersects_damage || changed;
		image.direct_visible = true;
		if !needs_emit {
			continue;
		}
		if !image.uploaded {
			append_transmission(&mut output, id, &image.png, tmux_passthrough);
			image.uploaded = true;
		}

		let fallback_width = u32::from(cols)
			.saturating_mul(u32::from(cell_pixel_width))
			.max(1);
		let fallback_height = u32::from(rows)
			.saturating_mul(u32::from(cell_pixel_height))
			.max(1);
		let (source_width, source_height) =
			png_dimensions(&image.png).unwrap_or((fallback_width, fallback_height));
		let row_offset = u64::from(visible_top - top);
		let row_end = u64::from(visible_bottom - top);
		let source_y = (row_offset.saturating_mul(u64::from(source_height)) / u64::from(rows)) as u32;
		let source_bottom =
			(row_end.saturating_mul(u64::from(source_height)) / u64::from(rows)) as u32;
		let source_height = source_bottom.saturating_sub(source_y).max(1);

		move_cursor_row(&mut output, &mut cursor_row, visible_top - window.top);
		output.push('\r');
		if left > 0 {
			let _ = write!(output, esc!(cursor_forward), left);
		}
		append_direct_placement(
			&mut output,
			id,
			DirectPlacement {
				source_x: 0,
				source_y,
				source_width,
				source_height,
				rows: visible_bottom - visible_top,
				cols,
			},
			tmux_passthrough,
		);
	}
	if !output.is_empty() {
		move_cursor_row(&mut output, &mut cursor_row, window.height - 1);
		output.push('\r');
	}
	output
}

fn png_dimensions(png: &[u8]) -> Option<(u32, u32)> {
	const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
	if png.get(..8) != Some(SIGNATURE) || png.get(12..16) != Some(b"IHDR") {
		return None;
	}
	let width = u32::from_be_bytes(png.get(16..20)?.try_into().ok()?);
	let height = u32::from_be_bytes(png.get(20..24)?.try_into().ok()?);
	(width > 0 && height > 0).then_some((width, height))
}

pub fn image_placement(frame: &Frame, id: u32) -> Option<(u16, u16, u16, u16)> {
	for y in 0..frame.size().height {
		for x in 0..frame.size().width {
			if let CellContent::Image { id: cell_id, row, col, rows, cols } = frame.cell(x, y).content
				&& cell_id == id
				&& rows > 0
				&& cols > 0
			{
				return Some((y.saturating_sub(row), x.saturating_sub(col), rows, cols));
			}
		}
	}
	None
}

fn contract_error(message: &'static str) -> io::Error {
	io::Error::new(io::ErrorKind::InvalidData, message)
}

fn layout(
	document_height: u16,
	viewport_height: u16,
	stable_rows: u16,
	committed_rows: u16,
) -> Layout {
	let natural_top = document_height.saturating_sub(viewport_height);
	let stable_limit = committed_rows.max(stable_rows.min(natural_top));
	let window_top = committed_rows.max(natural_top);
	Layout { stable_limit, window_top }
}

/// Resolves declarative layers into z-ordered viewport bands.
fn resolve_layers<'a>(layers: &'a [Layer<'_>], viewport: Size) -> SmallVec<ResolvedLayer<'a>, 4> {
	let mut ordered: SmallVec<(i16, ResolvedLayer<'a>), 4> = layers
		.iter()
		.filter_map(|layer| {
			let band = layer.band(viewport);
			(band.rows > 0).then_some((layer.options.z, ResolvedLayer {
				frame:   layer.frame,
				x:       band.x,
				y:       band.y,
				src_top: band.src_top,
				rows:    band.rows,
				active:  layer.active,
			}))
		})
		.collect();
	ordered.sort_by_key(|(z, _)| *z);
	ordered.into_iter().map(|(_, layer)| layer).collect()
}

fn store_layers(
	layers: &[ResolvedLayer<'_>],
	window: Window,
	document_width: u16,
) -> SmallVec<StoredLayer, 4> {
	let mut stored = SmallVec::new();
	store_layers_into(layers, window, document_width, &mut stored);
	stored
}

fn store_layers_into(
	layers: &[ResolvedLayer<'_>],
	window: Window,
	document_width: u16,
	stored: &mut SmallVec<StoredLayer, 4>,
) {
	let mut len = 0;
	for layer in layers {
		if layer.y >= window.height
			|| layer.x >= document_width
			|| layer.src_top >= layer.frame.size().height
			|| layer.frame.size().width == 0
		{
			continue;
		}
		let rows = layer
			.rows
			.min(window.height - layer.y)
			.min(layer.frame.size().height - layer.src_top);
		if rows == 0 {
			continue;
		}
		let source_address = std::ptr::from_ref(layer.frame).addr();
		let (source_id, source_revision) = layer.frame.source_stamp();
		if let Some(slot) = stored.get_mut(len) {
			let source_unchanged = slot.source_address == source_address
				&& slot.source_id == source_id
				&& slot.source_revision == source_revision;
			if !source_unchanged && !slot.frame.same_grid(layer.frame) {
				slot.frame.clone_from(layer.frame);
			}
			slot.x = layer.x;
			slot.document_y = window.top.saturating_add(layer.y);
			slot.src_top = layer.src_top;
			slot.rows = rows;
			slot.active = layer.active;
			slot.source_address = source_address;
			slot.source_id = source_id;
			slot.source_revision = source_revision;
		} else {
			stored.push(StoredLayer {
				frame: layer.frame.clone(),
				x: layer.x,
				document_y: window.top.saturating_add(layer.y),
				src_top: layer.src_top,
				rows,
				active: layer.active,
				source_address,
				source_id,
				source_revision,
			});
		}
		len += 1;
	}
	stored.truncate(len);
}

fn changed_screen_rows(
	damaged: &[(u16, u16)],
	previous_layers: &[StoredLayer],
	next_layers: &[StoredLayer],
	window: Window,
) -> SmallVec<(u16, u16), 12> {
	let mut rows = SmallVec::new();
	let window_end = window.top.saturating_add(window.height);
	let mut push_document_rows = |start: u16, end: u16| {
		let start = start.max(window.top);
		let end = end.min(window_end);
		if start < end {
			rows.push((start - window.top, end - window.top));
		}
	};
	for &(start, end) in damaged {
		push_document_rows(start, end);
	}
	for index in 0..previous_layers.len().max(next_layers.len()) {
		let previous = previous_layers.get(index);
		let next = next_layers.get(index);
		if previous
			.zip(next)
			.is_some_and(|(previous, next)| previous.same_cells_and_placement(next))
		{
			continue;
		}
		if let Some(layer) = previous {
			push_document_rows(layer.document_y, layer.document_y.saturating_add(layer.rows));
		}
		if let Some(layer) = next {
			push_document_rows(layer.document_y, layer.document_y.saturating_add(layer.rows));
		}
	}
	rows
}

/// One right-trimmed text row per viewport line of `base` under `layers`.
fn stored_text(base: &Frame, layers: &[StoredLayer], top: u16, height: u16) -> Vec<String> {
	let composed = ComposedFrame { base, layers };
	let blank = Cell::blank(Style::default());
	let width = base.size().width;
	let mut rows = Vec::with_capacity(usize::from(height));
	for offset in 0..height {
		let y = top.saturating_add(offset);
		let mut text = String::new();
		for x in 0..width {
			match &composed.cell_or(y, x, &blank).content {
				CellContent::Blank => text.push(' '),
				CellContent::Grapheme { text: glyph, .. } => text.push_str(glyph),
				CellContent::Image { .. } => text.push(' '),
				CellContent::Continuation => {},
			}
		}
		text.truncate(text.trim_end().len());
		rows.push(text);
	}
	rows
}

/// Hardware-cursor choice for a composited screen: the layer owning the
/// keyboard places — or, without a frame cursor, suppresses — the caret;
/// with no active layer the base document's caret shows through passive
/// layers.
fn compose_cursor(
	base: &Frame,
	layers: &[StoredLayer],
	window: Window,
	document_width: u16,
) -> Option<ScreenCursor> {
	match layers.iter().rev().find(|layer| layer.active) {
		Some(layer) => layer_cursor(layer, window, document_width),
		None => frame_cursor(base, window),
	}
}

/// Translates a layer frame's cursor into screen coordinates.
fn layer_cursor(layer: &StoredLayer, window: Window, document_width: u16) -> Option<ScreenCursor> {
	let (col, row) = layer.frame.cursor()?;
	if col >= layer.frame.size().width
		|| row < layer.src_top
		|| row >= layer.src_top.saturating_add(layer.rows)
	{
		return None;
	}
	let screen_row = layer
		.document_y
		.saturating_sub(window.top)
		.saturating_add(row - layer.src_top);
	let screen_col = layer.x.saturating_add(col);
	(screen_row < window.height && screen_col < document_width)
		.then_some(ScreenCursor { row: screen_row, col: screen_col })
}

fn frame_cursor(frame: &Frame, window: Window) -> Option<ScreenCursor> {
	let (col, document_row) = frame.cursor()?;
	if col >= frame.size().width
		|| document_row < window.top
		|| document_row >= window.top.saturating_add(window.height)
	{
		return None;
	}
	Some(ScreenCursor { row: document_row - window.top, col })
}

fn place_cursor(output: &mut String, cursor: Option<ScreenCursor>, viewport_height: u16) {
	let Some(cursor) = cursor else {
		return;
	};
	let mut row = viewport_height - 1;
	move_cursor_row(output, &mut row, cursor.row);
	output.push('\r');
	if cursor.col > 0 {
		let _ = write!(output, esc!(cursor_forward), cursor.col);
	}
	output.push_str(SHOW_CURSOR);
}

const fn scroll_append_to(
	previous_window: Window,
	next_window: Window,
	committed_rows: u16,
	stable_limit: u16,
) -> u16 {
	if committed_rows != previous_window.top || next_window.top <= previous_window.top {
		return committed_rows;
	}
	let scroll = next_window.top - previous_window.top;
	if scroll >= previous_window.height || next_window.top > stable_limit {
		return committed_rows;
	}
	next_window.top
}

/// Re-emits live wrap boundaries whose hard/soft state changed since the
/// previous paint. VT has no in-place line-attribute rewrite: a boundary
/// turning soft re-arms the pending wrap and re-prints its continuation
/// row through autowrap; one turning hard erases and re-prints both rows
/// (EL resets the attribute on mainstream terminals; frames that may hold
/// direct-drawn images overprint without erasing so placements survive).
/// Boundaries the commit loop emitted this paint are skipped; `scroll` is
/// the number of newly committed rows and `region` the scrolled zone
/// height (the full viewport without margin scrollback). The cursor is
/// expected on — and is re-parked at — the viewport's bottom row.
#[allow(clippy::too_many_arguments, reason = "diff inputs describe two composed viewport slices")]
fn reconcile_wrap_boundaries(
	output: &mut String,
	previous: &ComposedFrame<'_>,
	previous_window: Window,
	next: &ComposedFrame<'_>,
	next_window: Window,
	scroll: u16,
	region: u16,
	graphics: Graphics,
	hyperlinks: bool,
	stats: &mut PaintStats,
) {
	let height = next_window.height;
	let erase = !next.base.may_have_images();
	let mut cursor_row = height - 1;
	let mut emitted = false;
	for boundary in 0..height.saturating_sub(1) {
		let row = next_window.top.saturating_add(boundary);
		let wanted = wrap_joinable(next, row);
		let painted = if scroll == 0 {
			wrap_joinable(previous, previous_window.top.saturating_add(boundary))
		} else if boundary.saturating_add(1) < region.saturating_sub(scroll) {
			// Retained rows scrolled up with their line attributes intact.
			wrap_joinable(previous, next_window.top.saturating_add(boundary))
		} else if boundary.saturating_add(1) == region {
			// The commit scroll created the region's bottom line fresh.
			false
		} else if boundary >= region {
			// Pinned rows below a margin region never moved.
			wrap_joinable(previous, previous_window.top.saturating_add(boundary))
		} else {
			// The commit loop emits this boundary in its desired state.
			continue;
		};
		if painted == wanted {
			continue;
		}
		emitted = true;
		stats.runs += 1;
		stats.changed_cells = stats
			.changed_cells
			.saturating_add(usize::from(next.base.size().width).saturating_mul(2));
		move_cursor_row(output, &mut cursor_row, boundary);
		if wanted {
			output.push_str(esc!(autowrap));
			arm_wrap_boundary(output, next, row, graphics, hyperlinks);
			// The continuation row's first glyph rides the pending wrap;
			// re-printing it whole keeps the screen byte-identical.
			encode_frame_row(output, next, row.saturating_add(1), graphics, hyperlinks);
			output.push_str(esc!(!autowrap));
			cursor_row = boundary + 1;
		} else {
			output.push('\r');
			if erase {
				output.push_str(esc!(erase_line));
			}
			encode_frame_row(output, next, row, graphics, hyperlinks);
			move_cursor_row(output, &mut cursor_row, boundary + 1);
			output.push('\r');
			if erase {
				output.push_str(esc!(erase_line));
			}
			encode_frame_row(output, next, row.saturating_add(1), graphics, hyperlinks);
		}
	}
	if emitted {
		output.push_str(RESET_STYLE);
		move_cursor_row(output, &mut cursor_row, height - 1);
		output.push('\r');
	}
}
fn emit_scroll_append(
	output: &mut String,
	previous: &ComposedFrame<'_>,
	previous_window: Window,
	next: &ComposedFrame<'_>,
	next_window: Window,
	graphics: Graphics,
	hyperlinks: bool,
	stats: &mut PaintStats,
) {
	let scroll = next_window.top - previous_window.top;
	emit_window_diff(
		output,
		previous,
		Window { top: previous_window.top, height: scroll },
		next,
		Window { top: next_window.top - scroll, height: scroll },
		0,
		next_window.height,
		graphics,
		hyperlinks,
		stats,
	);
	output.push_str(VIEWPORT_BOTTOM);
	let first_new = next_window.height - scroll;
	let any_join = (first_new..next_window.height).any(|screen_y| {
		let row = next_window.top.saturating_add(screen_y);
		row > 0 && wrap_joinable(next, row - 1)
	});
	if any_join {
		output.push_str(esc!(autowrap));
	}
	for screen_y in first_new..next_window.height {
		let row = next_window.top.saturating_add(screen_y);
		if row > 0 && wrap_joinable(next, row - 1) {
			// The first joined row rides a freshly armed pending wrap: the
			// bottom line still shows last frame's paint, so its trailing
			// glyph is re-printed under DECAWM. Every further full-width
			// row printed below arms the pending wrap itself.
			if screen_y == first_new {
				arm_wrap_boundary(output, next, row - 1, graphics, hyperlinks);
			}
		} else {
			output.push_str("\r\n");
		}
		encode_frame_row(output, next, row, graphics, hyperlinks);
	}
	if any_join {
		output.push_str(esc!(!autowrap));
	}
	stats.runs += usize::from(scroll);
	stats.changed_cells += usize::from(next.base.size().width).saturating_mul(usize::from(scroll));

	let retained_rows = next_window.height - scroll;
	emit_window_diff(
		output,
		previous,
		Window { top: previous_window.top.saturating_add(scroll), height: retained_rows },
		next,
		Window { top: next_window.top, height: retained_rows },
		0,
		next_window.height,
		graphics,
		hyperlinks,
		stats,
	);
}

/// Commits rows like [`emit_scroll_append`] but scrolls only the top
/// `region_rows` screen rows through a top-anchored DECSTBM margin, leaving
/// the rows below physically pinned.
///
/// On terminals that move margin-scrolled rows into native scrollback this
/// keeps history identical to a whole-screen scroll while the pinned live
/// rows never move on screen; whether a native selection over them survives
/// is the terminal's selection-transform property (see
/// [`Renderer::set_margin_scrollback`]). Changed pinned cells (spinners,
/// streaming text) are diffed in place. The caller guarantees
/// `scroll <= region_rows < viewport height`.
fn emit_margin_scroll_append(
	output: &mut String,
	previous: &ComposedFrame<'_>,
	previous_window: Window,
	next: &ComposedFrame<'_>,
	next_window: Window,
	region_rows: u16,
	graphics: Graphics,
	hyperlinks: bool,
	stats: &mut PaintStats,
) {
	let scroll = next_window.top - previous_window.top;
	// Finalize the outgoing rows in place so native scrollback receives
	// their committed content.
	emit_window_diff(
		output,
		previous,
		Window { top: previous_window.top, height: scroll },
		next,
		Window { top: next_window.top - scroll, height: scroll },
		0,
		next_window.height,
		graphics,
		hyperlinks,
		stats,
	);
	// DECSTBM homes the cursor into the region; CUD then parks on the
	// bottom margin, where each newline commits the region's top row.
	let _ = write!(output, esc!(scroll_region, cursor_down), region_rows, region_rows - 1);
	let first_new = region_rows - scroll;
	let any_join = (first_new..region_rows).any(|screen_y| {
		let row = next_window.top.saturating_add(screen_y);
		row > 0 && wrap_joinable(next, row - 1)
	});
	if any_join {
		output.push_str(esc!(autowrap));
	}
	for screen_y in first_new..region_rows {
		let row = next_window.top.saturating_add(screen_y);
		if row > 0 && wrap_joinable(next, row - 1) {
			if screen_y == first_new {
				arm_wrap_boundary(output, next, row - 1, graphics, hyperlinks);
			}
		} else {
			output.push_str("\r\n");
		}
		encode_frame_row(output, next, row, graphics, hyperlinks);
	}
	if any_join {
		output.push_str(esc!(!autowrap));
	}
	stats.runs += usize::from(scroll);
	stats.changed_cells += usize::from(next.base.size().width).saturating_mul(usize::from(scroll));
	// Reset the margins (homing the cursor again) and re-park at the
	// viewport bottom for the retained-row diff.
	output.push_str(esc!(margins_reset));
	output.push_str(VIEWPORT_BOTTOM);
	let shifted = region_rows - scroll;
	emit_window_diff(
		output,
		previous,
		Window { top: previous_window.top.saturating_add(scroll), height: shifted },
		next,
		Window { top: next_window.top, height: shifted },
		0,
		next_window.height,
		graphics,
		hyperlinks,
		stats,
	);
	// The pinned live rows never moved; repaint their changed cells in place.
	let pinned = next_window.height - region_rows;
	emit_window_diff(
		output,
		previous,
		Window { top: previous_window.top.saturating_add(region_rows), height: pinned },
		next,
		Window { top: next_window.top.saturating_add(region_rows), height: pinned },
		region_rows,
		next_window.height,
		graphics,
		hyperlinks,
		stats,
	);
}

/// Emits `prefix` document rows then the window sequentially from the
/// cursor's current line. Hard boundaries advance with `\r\n`; a joinable
/// boundary between consecutive document rows is left to terminal
/// autowrap, marking the pair as one soft-wrapped line for native copy.
fn emit_rows(
	output: &mut String,
	frame: &ComposedFrame<'_>,
	prefix: Range<u16>,
	window: Window,
	graphics: Graphics,
	hyperlinks: bool,
) {
	let mut any_join = false;
	let mut previous: Option<u16> = None;
	for row in prefix
		.clone()
		.chain((0..window.height).map(|screen_y| window.top.saturating_add(screen_y)))
	{
		if previous.is_some_and(|p| row == p.saturating_add(1) && wrap_joinable(frame, p)) {
			any_join = true;
			break;
		}
		previous = Some(row);
	}
	if any_join {
		output.push_str(esc!(autowrap));
	}
	let mut previous: Option<u16> = None;
	for row in prefix.chain((0..window.height).map(|screen_y| window.top.saturating_add(screen_y))) {
		if let Some(p) = previous
			&& !(row == p.saturating_add(1) && wrap_joinable(frame, p))
		{
			output.push_str("\r\n");
		}
		encode_frame_row(output, frame, row, graphics, hyperlinks);
		previous = Some(row);
	}
	if any_join {
		output.push_str(esc!(!autowrap));
	}
}

#[inline(always)]
fn cells_equal(previous: &Cell, next: &Cell, hyperlinks: bool) -> bool {
	previous.content == next.content
		&& (previous.style == next.style
			|| (!hyperlinks && previous.style.without_link() == next.style.without_link()))
}

#[inline]
fn emit_window_diff(
	output: &mut String,
	previous: &ComposedFrame<'_>,
	previous_window: Window,
	next: &ComposedFrame<'_>,
	next_window: Window,
	screen_top: u16,
	screen_height: u16,
	graphics: Graphics,
	hyperlinks: bool,
	stats: &mut PaintStats,
) {
	emit_window_diff_rows(
		output,
		previous,
		previous_window,
		next,
		next_window,
		screen_top,
		screen_height,
		None,
		graphics,
		hyperlinks,
		stats,
	);
}

#[allow(clippy::too_many_arguments, reason = "diff inputs describe two composed viewport slices")]
fn emit_window_diff_rows(
	output: &mut String,
	previous: &ComposedFrame<'_>,
	previous_window: Window,
	next: &ComposedFrame<'_>,
	next_window: Window,
	screen_top: u16,
	screen_height: u16,
	dirty_rows: Option<&[(u16, u16)]>,
	graphics: Graphics,
	hyperlinks: bool,
	stats: &mut PaintStats,
) {
	let blank = Cell::blank(Style::default());
	let width = next.base.size().width;
	let mut active_style = Style::default();
	let mut cursor_row = screen_height - 1;

	for screen_y in 0..next_window.height {
		if let Some(rows) = dirty_rows
			&& !rows
				.iter()
				.any(|&(start, end)| start <= screen_y && screen_y < end)
		{
			continue;
		}
		let previous_y = previous_window.top.saturating_add(screen_y);
		let next_y = next_window.top.saturating_add(screen_y);
		let mut x = 0;
		while x < width {
			if cells_equal(
				previous.cell_or(previous_y, x, &blank),
				next.cell_or(next_y, x, &blank),
				hyperlinks,
			) {
				x += 1;
				continue;
			}

			let mut start = x;
			while start > 0
				&& matches!(next.cell_or(next_y, start, &blank).content, CellContent::Continuation)
			{
				start -= 1;
			}

			let mut end = x + 1;
			stats.changed_cells += 1;
			while end < width {
				let previous_cell = previous.cell_or(previous_y, end, &blank);
				let next_cell = next.cell_or(next_y, end, &blank);
				if cells_equal(previous_cell, next_cell, hyperlinks) {
					break;
				}
				end += 1;
				stats.changed_cells += 1;
			}
			while end < width
				&& matches!(next.cell_or(next_y, end, &blank).content, CellContent::Continuation)
			{
				end += 1;
			}

			emit_run(
				output,
				next,
				Run { document_y: next_y, screen_y: screen_top.saturating_add(screen_y), start, end },
				&blank,
				&mut active_style,
				&mut cursor_row,
				graphics,
				hyperlinks,
			);
			stats.runs += 1;
			x = end;
		}
	}

	if stats.runs > 0 {
		output.push_str(RESET_STYLE);
		move_cursor_row(output, &mut cursor_row, screen_height - 1);
		output.push('\r');
	}
}

pub fn move_cursor_row(output: &mut String, current: &mut u16, target: u16) {
	if target < *current {
		let _ = write!(output, esc!(cursor_up), *current - target);
	} else if target > *current {
		let _ = write!(output, esc!(cursor_down), target - *current);
	}
	*current = target;
}

fn emit_run(
	output: &mut String,
	frame: &ComposedFrame<'_>,
	run: Run,
	blank: &Cell,
	active_style: &mut Style,
	cursor_row: &mut u16,
	graphics: Graphics,
	hyperlinks: bool,
) {
	move_cursor_row(output, cursor_row, run.screen_y);
	output.push('\r');
	if run.start > 0 {
		let _ = write!(output, esc!(cursor_forward), run.start);
	}
	let mut x = run.start;

	while x < run.end {
		let cell = frame.cell_or(run.document_y, x, blank);
		match &cell.content {
			CellContent::Blank => {
				emit_cell_style(output, cell.style, active_style, hyperlinks);
				output.push(' ');
				x += 1;
			},
			CellContent::Grapheme { text, width } => {
				emit_cell_style(output, cell.style, active_style, hyperlinks);
				output.push_str(text);
				x = x.saturating_add(*width);
			},
			CellContent::Image { id, row, col, rows, cols } => {
				emit_image_cell(
					output,
					*id,
					*row,
					*col,
					*rows,
					*cols,
					active_style,
					graphics,
					hyperlinks,
				);
				x += 1;
			},
			CellContent::Continuation => x += 1,
		}
	}
	close_active_link(output, active_style, hyperlinks);
}
/// Whether the boundary between document rows `row` and `row + 1` may be
/// joined by terminal autowrap. The flag is the certification: painters
/// only set it for rows whose source content exactly fills the width (see
/// [`Frame::set_soft_wrap`]), which the renderer cannot re-verify — a real
/// trailing space and a padding cell are both stored as blanks.
///
/// Deliberately a pure document property: overlay layers composite on top
/// without changing it, so band movement never flips boundaries (which
/// would force viewport repaints), and the line attribute stays correct
/// for the raw rows an overlay only transiently covers.
#[inline]
fn wrap_joinable(frame: &ComposedFrame<'_>, row: u16) -> bool {
	frame.base.soft_wrap(row)
}

/// Re-prints the composed cell covering the final column of document row
/// `row` on the cursor's current line, arming the terminal's pending-wrap
/// state so the next printed glyph soft-wraps onto the following line.
/// Emitting the composed view keeps overlay layers intact. Requires DECAWM
/// to be enabled.
fn arm_wrap_boundary(
	output: &mut String,
	frame: &ComposedFrame<'_>,
	row: u16,
	graphics: Graphics,
	hyperlinks: bool,
) {
	let width = frame.base.size().width;
	let Some(last) = width.checked_sub(1) else {
		return;
	};
	let blank = Cell::blank(Style::default());
	// Walk left over continuation cells so a wide glyph is re-printed
	// whole from its head instead of being clobbered mid-cell.
	let mut x = last;
	let cell = loop {
		let cell = frame.cell_or(row, x, &blank);
		match &cell.content {
			CellContent::Continuation if x > 0 => x -= 1,
			_ => break cell,
		}
	};
	output.push('\r');
	if x > 0 {
		let _ = write!(output, esc!(cursor_forward), x);
	}
	output.push_str(RESET_STYLE);
	let mut active = Style::default();
	match &cell.content {
		CellContent::Grapheme { text, width: glyph }
			if x.saturating_add(*glyph) == width && *glyph > 0 =>
		{
			emit_cell_style(output, cell.style, &mut active, hyperlinks);
			output.push_str(text);
		},
		CellContent::Image { id, row: img_row, col, rows, cols } if x == last => {
			emit_image_cell(
				output,
				*id,
				*img_row,
				*col,
				*rows,
				*cols,
				&mut active,
				graphics,
				hyperlinks,
			);
		},
		_ => {
			// Blanks (or anything unprintable) still fill through the
			// final column, which is all the pending wrap needs.
			emit_cell_style(output, cell.style, &mut active, hyperlinks);
			for _ in x..width {
				output.push(' ');
			}
		},
	}
	close_active_link(output, &mut active, hyperlinks);
}
fn encode_frame_row(
	output: &mut String,
	frame: &ComposedFrame<'_>,
	row: u16,
	graphics: Graphics,
	hyperlinks: bool,
) {
	if row < frame.base.size().height {
		encode_row(output, frame, row, graphics, hyperlinks);
	} else {
		encode_blank_row(output, frame.base.size().width);
	}
}

fn encode_row(
	output: &mut String,
	frame: &ComposedFrame<'_>,
	row: u16,
	graphics: Graphics,
	hyperlinks: bool,
) {
	output.push_str(RESET_STYLE);
	let blank = Cell::blank(Style::default());
	let mut active_style = Style::default();
	let mut x = 0;
	while x < frame.base.size().width {
		let cell = frame.cell_or(row, x, &blank);
		match &cell.content {
			CellContent::Blank => {
				emit_cell_style(output, cell.style, &mut active_style, hyperlinks);
				output.push(' ');
				x += 1;
			},
			CellContent::Grapheme { text, width } => {
				emit_cell_style(output, cell.style, &mut active_style, hyperlinks);
				output.push_str(text);
				x = x.saturating_add(*width);
			},
			CellContent::Image { id, row, col, rows, cols } => {
				emit_image_cell(
					output,
					*id,
					*row,
					*col,
					*rows,
					*cols,
					&mut active_style,
					graphics,
					hyperlinks,
				);
				x += 1;
			},
			CellContent::Continuation => x += 1,
		}
	}
	close_active_link(output, &mut active_style, hyperlinks);
}

#[allow(clippy::too_many_arguments, reason = "flat cell emission hot path")]
fn emit_image_cell(
	output: &mut String,
	id: u32,
	row: u16,
	col: u16,
	rows: u16,
	cols: u16,
	active_style: &mut Style,
	graphics: Graphics,
	hyperlinks: bool,
) {
	if graphics != Graphics::KittyPlaceholders {
		emit_cell_style(output, Style::default(), active_style, hyperlinks);
		output.push(' ');
		return;
	}
	let (placeholder, style) = placeholder_cell(id, row, col, rows, cols);
	emit_cell_style(output, style, active_style, hyperlinks);
	output.push_str(&placeholder);
}

fn encode_blank_row(output: &mut String, width: u16) {
	output.push_str(RESET_STYLE);
	for _ in 0..width {
		output.push(' ');
	}
}

fn emit_cell_style(output: &mut String, style: Style, active_style: &mut Style, hyperlinks: bool) {
	let link_changed = hyperlinks && active_style.link != style.link;
	if link_changed && active_style.link.is_some() {
		output.push_str(esc!(osc, "8;;", st));
	}
	let visual = style.without_link();
	if active_style.without_link() != visual {
		emit_style(output, visual);
	}
	if link_changed && let Some(id) = style.link {
		emit_link_open(output, id);
	}
	*active_style = style;
}

fn close_active_link(output: &mut String, active_style: &mut Style, hyperlinks: bool) {
	if hyperlinks && active_style.link.is_some() {
		output.push_str(esc!(osc, "8;;", st));
	}
	active_style.link = None;
}

fn emit_link_open(output: &mut String, id: LinkId) {
	let _ = with_link_url(id, |url| {
		let _ = write!(output, esc!(osc, "8;id={};"), id.get());
		for ch in url.chars().filter(|ch| !matches!(ch, '\x1b' | '\x07')) {
			output.push(ch);
		}
		output.push_str(esc!(st));
	});
}

fn emit_style(output: &mut String, style: Style) {
	let style = style.without_link();
	output.push_str(RESET_STYLE);
	if style == Style::default() {
		return;
	}

	output.push_str(esc!(csi));
	let mut first = true;
	push_style_parameters(output, style, &mut first);
	output.push('m');
}

/// Appends the renderer's canonical non-reset SGR parameters.
pub fn push_style_parameters(output: &mut String, style: Style, first: &mut bool) {
	if style.bold {
		push_parameter(output, first, "1");
	}
	if style.dim {
		push_parameter(output, first, "2");
	}
	if style.italic {
		push_parameter(output, first, "3");
	}
	if style.underline {
		push_parameter(output, first, "4");
	}
	match style.underline_color {
		Color::Default => {},
		color => {
			if !*first {
				output.push(';');
			}
			*first = false;
			// Colon sub-parameter form per kitty; ghostty accepts both forms.
			match color {
				Color::Indexed(index) => {
					let _ = write!(output, "58:5:{index}");
				},
				Color::Rgb(red, green, blue) => {
					let _ = write!(output, "58:2::{red}:{green}:{blue}");
				},
				Color::Default => unreachable!("matched above"),
			}
		},
	}
	if style.reverse {
		push_parameter(output, first, "7");
	}
	if style.strikethrough {
		push_parameter(output, first, "9");
	}
	push_color_code(output, first, style.foreground, false);
	push_color_code(output, first, style.background, true);
}

fn push_parameter(output: &mut String, first: &mut bool, parameter: &str) {
	if !*first {
		output.push(';');
	}
	output.push_str(parameter);
	*first = false;
}

fn push_color_code(output: &mut String, first: &mut bool, color: Color, background: bool) {
	if color == Color::Default {
		return;
	}
	if !*first {
		output.push(';');
	}
	*first = false;

	let prefix = if background { 48 } else { 38 };
	match color {
		Color::Default => unreachable!("default colors returned before emission"),
		Color::Indexed(index) => {
			let _ = write!(output, "{prefix};5;{index}");
		},
		Color::Rgb(red, green, blue) => {
			let _ = write!(output, "{prefix};2;{red};{green};{blue}");
		},
	}
}

#[cfg(test)]
mod tests {
	use std::io::ErrorKind;

	use super::{
		ConptyChunks, MAX_CONPTY_WRITE_CHUNK_BYTES, MAX_OUTPUT_BACKLOG_BYTES, OutputBacklogGuard,
		REBUILD_HISTORY, ResolvedLayer, SYNC_OUTPUT_BEGIN, SYNC_OUTPUT_END, VIEWPORT_BOTTOM,
	};
	use crate::{
		Color, Frame, Graphics, Renderer, Size, Style,
		overlay::{Layer, OverlayAnchor, OverlayOptions},
		test_support::TerminalModel,
	};

	fn document(lines: &[&str]) -> Frame {
		let mut frame = Frame::new(Size::new(8, u16::try_from(lines.len()).expect("small fixture")));
		for (row, line) in lines.iter().enumerate() {
			frame.put(0, u16::try_from(row).expect("small fixture"), line, Style::default());
		}
		frame
	}
	/// [`document`] with soft-wrap flags on the given boundary rows.
	fn soft_document(lines: &[&str], soft_after: &[u16]) -> Frame {
		let mut frame = document(lines);
		for &row in soft_after {
			frame.set_soft_wrap(row);
		}
		frame
	}

	fn apply_paint(renderer: &mut Renderer<Vec<u8>>, terminal: &mut TerminalModel) {
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).expect("ANSI is UTF-8");
		terminal.apply(&output);
	}

	fn without_sync_markers(output: &str) -> String {
		output
			.replace(SYNC_OUTPUT_BEGIN, "")
			.replace(SYNC_OUTPUT_END, "")
	}

	#[test]
	fn layer_appears_once_at_viewport_coordinates() {
		let base = document(&["base000", "base111"]);
		let mut overlay = Frame::new(Size::new(2, 1));
		overlay.put(0, 0, "OV", Style::default());
		let layer = ResolvedLayer {
			frame:   &overlay,
			x:       2,
			y:       1,
			src_top: 0,
			rows:    1,
			active:  false,
		};
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 2);

		let first = renderer
			.present_resolved(&base, &[], 2, 0, &[layer])
			.expect("overlay paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert!(first.runs > 0);
		assert_eq!(terminal.visible_rows(), ["base000", "baOV111"]);

		let second = renderer
			.present_resolved(&base, &[], 2, 0, &[ResolvedLayer {
				frame:   &overlay,
				x:       2,
				y:       1,
				src_top: 0,
				rows:    1,
				active:  false,
			}])
			.expect("identical overlay paint succeeds");
		assert_eq!(second.runs, 0);
		assert_eq!(second.bytes, 0);
	}

	#[test]
	fn declarative_layer_resolves_options_to_a_band() {
		let base = document(&["base000", "base111"]);
		let mut overlay = Frame::new(Size::new(2, 1));
		overlay.put(0, 0, "OV", Style::default());
		let options = OverlayOptions::default()
			.anchor(OverlayAnchor::TopLeft)
			.offset_x(2)
			.offset_y(1);
		let layer = Layer { frame: &overlay, options: &options, active: false };
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 2);

		renderer
			.present_overlaid(&base, &[], 2, 0, &[layer])
			.expect("declarative layer paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(terminal.visible_rows(), ["base000", "baOV111"]);
	}

	#[test]
	fn clearing_overlay_repaints_document_cells() {
		let base = document(&["base000", "base111"]);
		let mut overlay = Frame::new(Size::new(2, 1));
		overlay.put(0, 0, "OV", Style::default());
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 2);
		renderer
			.present(base.clone(), 2, 0)
			.expect("base paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		renderer
			.present_resolved(&base, &[], 2, 0, &[ResolvedLayer {
				frame:   &overlay,
				x:       2,
				y:       1,
				src_top: 0,
				rows:    1,
				active:  false,
			}])
			.expect("overlay paint succeeds");
		apply_paint(&mut renderer, &mut terminal);

		let stats = renderer
			.present_ref(&base, 2, 0)
			.expect("clearing paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert!(stats.runs > 0);
		assert_eq!(terminal.visible_rows(), ["base000", "base111"]);
	}

	#[test]
	fn document_growth_scrolls_raw_rows_to_history_under_open_overlay() {
		fn resolved_layer(overlay: &Frame) -> ResolvedLayer<'_> {
			ResolvedLayer {
				frame:   overlay,
				x:       0,
				y:       0,
				src_top: 0,
				rows:    1,
				active:  false,
			}
		}
		let mut overlay = Frame::new(Size::new(2, 1));
		overlay.put(0, 0, "OV", Style::default());
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 2);
		renderer
			.present(document(&["row00", "row01"]), 2, 2)
			.expect("initial paint succeeds");
		apply_paint(&mut renderer, &mut terminal);

		let first = renderer
			.present_resolved(&document(&["row00", "row01", "row02"]), &[(2, 3)], 2, 3, &[
				resolved_layer(&overlay),
			])
			.expect("first growth under the layer succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(first.committed_rows, 1, "commits keep flowing under an open layer");
		assert_eq!(terminal.history, ["row00"]);
		assert_eq!(
			terminal.visible_rows(),
			["OVw01", "row02"],
			"the layer stays viewport-anchored after the scroll"
		);

		let second = renderer
			.present_resolved(&document(&["row00", "row01", "row02", "row03"]), &[(3, 4)], 2, 4, &[
				resolved_layer(&overlay),
			])
			.expect("second growth under the layer succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(second.committed_rows, 1);
		assert_eq!(
			terminal.history,
			["row00", "row01"],
			"the row physically under the layer is restored before it scrolls out"
		);
		assert!(terminal.history.iter().all(|row| !row.contains("OV")));
		assert_eq!(terminal.visible_rows(), ["OVw02", "row03"]);

		renderer
			.present_damaged(&document(&["row00", "row01", "row02", "row03"]), &[], 2, 4)
			.expect("clearing the layer succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(terminal.visible_rows(), ["row02", "row03"]);
		assert_eq!(terminal.history, ["row00", "row01"], "clearing commits nothing extra");
	}

	#[test]
	fn clear_layers_restores_raw_cells_without_committing() {
		let mut overlay = Frame::new(Size::new(2, 1));
		overlay.put(0, 0, "OV", Style::default());
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 2);
		renderer
			.present(document(&["row00", "row01", "row02"]), 2, 3)
			.expect("initial paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		renderer
			.present_resolved(&document(&["row00", "row01", "row02"]), &[], 2, 3, &[ResolvedLayer {
				frame:   &overlay,
				x:       0,
				y:       0,
				src_top: 0,
				rows:    1,
				active:  false,
			}])
			.expect("layered paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(terminal.visible_rows(), ["OVw01", "row02"]);

		renderer.clear_layers().expect("teardown scrub succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(
			terminal.visible_rows(),
			["row01", "row02"],
			"bands repaint from the raw document"
		);
		assert_eq!(terminal.history, ["row00"], "the scrub commits nothing");

		renderer.clear_layers().expect("layer-free scrub succeeds");
		assert!(renderer.writer_mut().is_empty(), "a layer-free scrub writes nothing");
	}

	#[test]
	fn cursor_follows_the_active_layer_and_base_shows_through_passive_ones() {
		let mut base = document(&["base000", "base111", "base222"]);
		base.set_cursor(0, 0);
		let mut overlay = Frame::new(Size::new(2, 2));
		overlay.put(0, 0, "aa", Style::default());
		overlay.put(0, 1, "bb", Style::default());
		overlay.set_cursor(1, 1);
		let layer =
			|active| ResolvedLayer { frame: &overlay, x: 3, y: 0, src_top: 0, rows: 2, active };
		let mut renderer = Renderer::new(Vec::new());
		renderer
			.present(base.clone(), 3, 0)
			.expect("base paint succeeds");
		renderer.writer_mut().clear();

		// An active layer owns the caret, translated to screen coordinates.
		renderer
			.present_resolved(&base, &[], 3, 0, &[layer(true)])
			.expect("active layer paint succeeds");
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).expect("UTF-8");
		assert!(output.contains("\x1b[1A\r\x1b[4C\x1b[?25h"), "{output:?}");

		// A passive layer lets the base document's caret show through.
		renderer
			.present_resolved(&base, &[], 3, 0, &[layer(false)])
			.expect("passive layer paint succeeds");
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).expect("UTF-8");
		assert!(output.contains("\x1b[2A\r\x1b[?25h"), "base caret at (0,0): {output:?}");

		// An active layer without a frame cursor suppresses the base caret.
		let mut blank = Frame::new(Size::new(2, 2));
		blank.put(0, 0, "cc", Style::default());
		renderer
			.present_resolved(&base, &[], 3, 0, &[ResolvedLayer {
				frame:   &blank,
				x:       3,
				y:       0,
				src_top: 0,
				rows:    2,
				active:  true,
			}])
			.expect("cursorless active paint succeeds");
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).expect("UTF-8");
		assert!(!output.contains("\x1b[?25h"), "no caret may show: {output:?}");
	}

	#[test]
	fn wide_document_grapheme_cut_by_overlay_is_blank_not_torn() {
		let mut base = Frame::new(Size::new(4, 1));
		base.put(0, 0, "界x", Style::default());
		let mut overlay = Frame::new(Size::new(1, 1));
		overlay.put(0, 0, "O", Style::default());
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(4, 1);
		renderer
			.present(base.clone(), 1, 0)
			.expect("base paint succeeds");
		apply_paint(&mut renderer, &mut terminal);

		renderer
			.present_resolved(&base, &[], 1, 0, &[ResolvedLayer {
				frame:   &overlay,
				x:       1,
				y:       0,
				src_top: 0,
				rows:    1,
				active:  false,
			}])
			.expect("wide overlay paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(terminal.visible_rows(), [" Ox"]);
	}

	#[test]
	fn damaged_present_diffs_away_stored_overlay() {
		let base = document(&["base000", "base111"]);
		let mut overlay = Frame::new(Size::new(2, 1));
		overlay.put(0, 0, "OV", Style::default());
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 2);
		renderer
			.present(base.clone(), 2, 0)
			.expect("base paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		renderer
			.present_resolved(&base, &[], 2, 0, &[ResolvedLayer {
				frame:   &overlay,
				x:       2,
				y:       1,
				src_top: 0,
				rows:    1,
				active:  false,
			}])
			.expect("overlay paint succeeds");
		apply_paint(&mut renderer, &mut terminal);

		let stats = renderer
			.present_damaged(&base, &[], 2, 0)
			.expect("damaged clearing paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert!(stats.runs > 0);
		assert_eq!(terminal.visible_rows(), ["base000", "base111"]);
	}

	#[test]
	fn layer_only_kitty_image_still_transmits_and_places() {
		let base = document(&["base000", "base111"]);
		let mut overlay = Frame::new(Size::new(3, 1));
		for col in 0..2 {
			overlay.put_image_cell(col, 0, 7, 0, col, 1, 2);
		}
		let mut renderer = Renderer::new(Vec::new());
		renderer.set_graphics(Graphics::KittyPlaceholders);
		renderer
			.register_image(7, b"\x89PNG\r\n\x1a\nsmall".to_vec())
			.expect("image registration succeeds");
		renderer
			.present(base.clone(), 2, 0)
			.expect("base paint succeeds");
		renderer.writer_mut().clear();

		renderer
			.present_resolved(&base, &[], 2, 0, &[ResolvedLayer {
				frame:   &overlay,
				x:       0,
				y:       0,
				src_top: 0,
				rows:    1,
				active:  false,
			}])
			.expect("overlay image paint succeeds");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");
		assert!(
			output.contains("\x1b_G"),
			"an image referenced only by a layer still uploads: {output:?}"
		);

		renderer.writer_mut().clear();
		renderer
			.present_resolved(&base, &[], 2, 0, &[ResolvedLayer {
				frame:   &overlay,
				x:       0,
				y:       0,
				src_top: 0,
				rows:    1,
				active:  false,
			}])
			.expect("steady overlay paint succeeds");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");
		assert!(!output.contains("\x1b_G"), "uploads happen once: {output:?}");
	}

	#[test]
	fn conpty_chunker_prefers_newlines_within_sixteen_kibibytes() {
		let line = "x".repeat(8 * 1024 - 1) + "\n";
		let payload = line.repeat(5);
		assert_eq!(payload.len(), 40 * 1024);
		let chunks =
			ConptyChunks::new(payload.as_bytes(), MAX_CONPTY_WRITE_CHUNK_BYTES).collect::<Vec<_>>();
		assert!(chunks.len() > 1);
		assert!(
			chunks
				.iter()
				.all(|chunk| chunk.len() <= MAX_CONPTY_WRITE_CHUNK_BYTES)
		);
		assert!(
			chunks[..chunks.len() - 1]
				.iter()
				.all(|chunk| chunk.ends_with(b"\n"))
		);
		assert_eq!(chunks.concat(), payload.as_bytes());
	}

	#[test]
	fn conpty_chunker_extends_past_an_escape_sequence_without_newlines() {
		let mut payload = vec![b'x'; MAX_CONPTY_WRITE_CHUNK_BYTES - 2];
		payload.extend_from_slice(b"\x1b]8;;https://example.test/a-very-long-link\x1b\\");
		payload.extend(std::iter::repeat_n(b'y', MAX_CONPTY_WRITE_CHUNK_BYTES));
		let chunks = ConptyChunks::new(&payload, MAX_CONPTY_WRITE_CHUNK_BYTES).collect::<Vec<_>>();
		assert!(chunks[0].len() > MAX_CONPTY_WRITE_CHUNK_BYTES);
		assert!(chunks[0].ends_with(b"\x1b\\"));
		assert_eq!(chunks.concat(), payload);
	}

	#[test]
	fn backlog_disconnects_only_after_sixty_four_mibibytes() {
		let mut guard = OutputBacklogGuard::default();
		assert!(!guard.queue(MAX_OUTPUT_BACKLOG_BYTES - 1));
		assert!(!guard.queue(1));
		assert!(guard.queue(1));
		guard.flushed();
		assert!(!guard.queue(1));
	}
	#[test]
	fn hyperlink_capability_materializes_only_the_link_label() {
		let target = "https://example.test/docs";
		let link_style = Style::new().underline().link(target);
		let id = link_style.link.expect("non-empty URL is interned").get();
		let mut frame = Frame::new(Size::new(12, 1));
		frame.put(0, 0, "go ", Style::new());
		frame.put(3, 0, "label", link_style);
		frame.put(8, 0, " end", Style::new());

		let mut renderer = Renderer::new(Vec::new());
		renderer.set_hyperlinks(true);
		renderer
			.present(frame, 1, 0)
			.expect("hyperlinked frame paints");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");
		let linked = format!("\x1b]8;id={id};{target}\x1b\\label\x1b]8;;\x1b\\");
		assert!(output.contains(&linked), "{output:?}");
		assert_eq!(output.matches("\x1b]8;id=").count(), 1);
		assert_eq!(output.matches("\x1b]8;;\x1b\\").count(), 1);
	}

	#[test]
	fn disabled_hyperlinks_are_byte_identical_to_plain_styled_cells() {
		let mut linked = Frame::new(Size::new(8, 1));
		linked.put(0, 0, "label", Style::new().underline().link("https://example.test"));
		let mut plain = Frame::new(Size::new(8, 1));
		plain.put(0, 0, "label", Style::new().underline());

		let mut linked_renderer = Renderer::new(Vec::new());
		let mut plain_renderer = Renderer::new(Vec::new());
		linked_renderer
			.present(linked, 1, 0)
			.expect("disabled hyperlink frame paints");
		plain_renderer
			.present(plain, 1, 0)
			.expect("plain frame paints");
		assert_eq!(linked_renderer.writer_mut(), plain_renderer.writer_mut());
	}

	#[test]
	fn iterm2_graphics_dispatches_registered_png_post_pass() {
		let mut frame = Frame::new(Size::new(2, 1));
		for col in 0..2 {
			frame.put_image_cell(col, 0, 7, 0, col, 1, 2);
		}
		let mut renderer = Renderer::new(Vec::new());
		renderer.set_graphics(Graphics::Iterm2);
		renderer
			.register_image(7, b"\x89PNG\r\n\x1a\nsmall".to_vec())
			.expect("image registration succeeds");
		renderer.present(frame, 1, 0).expect("iTerm2 frame paints");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");
		assert!(output.contains("\x1b]1337;File=inline=1;"));
	}

	#[test]
	fn disabled_synchronized_output_only_removes_wrappers_from_every_paint_path() {
		let initial = document(&["one", "two", "three"]);
		let changed = document(&["one", "TWO", "three"]);
		let mut synchronized = Renderer::new(Vec::new());
		let mut plain = Renderer::new(Vec::new());
		plain.set_sync_output(false);

		synchronized
			.present(initial.clone(), 2, 1)
			.expect("synchronized full paint succeeds");
		plain
			.present(initial, 2, 1)
			.expect("plain full paint succeeds");
		let synchronized_output =
			String::from_utf8(std::mem::take(synchronized.writer_mut())).expect("ANSI is UTF-8");
		let plain_output =
			String::from_utf8(std::mem::take(plain.writer_mut())).expect("ANSI is UTF-8");
		assert_eq!(without_sync_markers(&synchronized_output), plain_output);
		assert!(!plain_output.contains(SYNC_OUTPUT_BEGIN));
		assert!(!plain_output.contains(SYNC_OUTPUT_END));

		synchronized.writer_mut().clear();
		plain.writer_mut().clear();
		synchronized
			.present(changed.clone(), 2, 1)
			.expect("synchronized incremental paint succeeds");
		plain
			.present(changed.clone(), 2, 1)
			.expect("plain incremental paint succeeds");
		let synchronized_output =
			String::from_utf8(synchronized.writer_mut().clone()).expect("ANSI is UTF-8");
		let plain_output = String::from_utf8(plain.writer_mut().clone()).expect("ANSI is UTF-8");
		assert_eq!(without_sync_markers(&synchronized_output), plain_output);
		assert!(!plain_output.contains(SYNC_OUTPUT_BEGIN));
		assert!(!plain_output.contains(SYNC_OUTPUT_END));

		synchronized.writer_mut().clear();
		plain.writer_mut().clear();
		synchronized
			.preview(&changed, 2, "\x1b[?1049h")
			.expect("synchronized preview succeeds");
		plain
			.preview(&changed, 2, "\x1b[?1049h")
			.expect("plain preview succeeds");
		let synchronized_output =
			String::from_utf8(synchronized.writer_mut().clone()).expect("ANSI is UTF-8");
		let plain_output = String::from_utf8(plain.writer_mut().clone()).expect("ANSI is UTF-8");
		assert_eq!(without_sync_markers(&synchronized_output), plain_output);
		assert!(!plain_output.contains(SYNC_OUTPUT_BEGIN));
		assert!(!plain_output.contains(SYNC_OUTPUT_END));
	}

	#[test]
	fn screen_to_scrollback_precedes_viewport_clear_only_when_enabled() {
		let mut ordinary = Renderer::new(Vec::new());
		ordinary
			.present(document(&["one", "two"]), 2, 0)
			.expect("ordinary paint succeeds");
		let ordinary_output =
			String::from_utf8(ordinary.writer_mut().clone()).expect("ANSI is UTF-8");
		assert!(!ordinary_output.contains("\x1b[22J"));

		let mut preserving = Renderer::new(Vec::new());
		preserving.set_screen_to_scrollback(true);
		preserving
			.present(document(&["one", "two"]), 2, 0)
			.expect("scrollback-preserving paint succeeds");
		let preserving_output =
			String::from_utf8(preserving.writer_mut().clone()).expect("ANSI is UTF-8");
		assert!(preserving_output.contains("\x1b[22J\x1b[2J\x1b[H"));
	}

	#[test]
	fn soft_wrapped_rows_join_on_screen_and_in_history() {
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 3);

		// "abcdefgh" fills the row exactly and continues mid-word on "ij".
		renderer
			.present(soft_document(&["abcdefgh", "ij", "tail"], &[0]), 3, 0)
			.expect("initial paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert!(terminal.row_wrapped(1), "the continuation row carries the wrap attribute");
		assert_eq!(terminal.visible_rows(), ["abcdefgh", "ij", "tail"]);

		// Scrolling the pair into native scrollback keeps the join: copy
		// reads one unbroken line.
		renderer
			.present(soft_document(&["abcdefgh", "ij", "tail", "x", "y"], &[0]), 3, 4)
			.expect("growth paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(terminal.history, ["abcdefghij"]);
		assert_eq!(terminal.visible_rows(), ["tail", "x", "y"]);
	}

	#[test]
	fn scroll_append_arms_joins_for_committed_pairs() {
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 3);
		renderer
			.present(document(&["one", "two", "three"]), 3, 3)
			.expect("initial paint succeeds");
		apply_paint(&mut renderer, &mut terminal);

		renderer
			.present(soft_document(&["one", "two", "three", "abcdefgh", "ij"], &[3]), 3, 5)
			.expect("scroll paint succeeds");
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).expect("UTF-8");
		terminal.apply(&output);
		assert!(output.contains("\x1b[?7h"), "committing a joined pair enables autowrap");
		assert!(output.contains("\x1b[?7l"), "autowrap is restored after the commit");
		assert_eq!(terminal.history, ["one", "two"]);
		assert_eq!(terminal.visible_rows(), ["three", "abcdefgh", "ij"]);
		assert!(terminal.row_wrapped(2), "the committed continuation soft-wraps on screen");

		renderer
			.present(soft_document(&["one", "two", "three", "abcdefgh", "ij", "z", "w"], &[3]), 3, 7)
			.expect("second growth succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(terminal.history, ["one", "two", "three", "abcdefgh"]);

		renderer
			.present(
				soft_document(&["one", "two", "three", "abcdefgh", "ij", "z", "w", "v", "u"], &[3]),
				3,
				9,
			)
			.expect("third growth succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(
			terminal.history,
			["one", "two", "three", "abcdefghij", "z"],
			"the soft pair merges as it scrolls into history"
		);
		assert_eq!(terminal.visible_rows(), ["w", "v", "u"]);
	}

	#[test]
	fn wrap_boundary_flips_reconcile_in_place() {
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 2);
		renderer
			.present(soft_document(&["abcdefgh", "ij"], &[0]), 2, 0)
			.expect("initial paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert!(terminal.row_wrapped(1));

		// Same cells, hard boundary: both rows are erased and re-printed
		// in place — never through a viewport clear, which would push
		// duplicated history on scrollback-preserving terminals.
		renderer
			.present(document(&["abcdefgh", "ij"]), 2, 0)
			.expect("hardening paint succeeds");
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).expect("UTF-8");
		terminal.apply(&output);
		assert!(!output.contains("\x1b[H"), "reconciliation never clears the viewport");
		assert!(output.contains("\x1b[2K"), "the stale soft rows are erased in place");
		assert!(!terminal.row_wrapped(1), "the boundary reads hard again");
		assert_eq!(terminal.visible_rows(), ["abcdefgh", "ij"]);

		// Flagging it again re-arms the join without a repaint of the
		// rest of the viewport.
		renderer
			.present(soft_document(&["abcdefgh", "ij"], &[0]), 2, 0)
			.expect("softening paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert!(terminal.row_wrapped(1), "the boundary soft-wraps again");
		assert_eq!(terminal.visible_rows(), ["abcdefgh", "ij"]);
	}

	#[test]
	fn char_wrapped_source_spaces_survive_history_joins() {
		use crate::{
			UiContext,
			component::{Component, PaintCtx},
			components::TextLeaf,
			frame::Rect,
			props::Prop,
		};
		let ctx = UiContext::default();
		// The real space inside "ab cdef" lands in the final column of a
		// width-3 row and is stored as a blank cell — the flag, not the
		// cells, certifies the row as exactly full.
		let mut leaf = TextLeaf::new().with(Prop::Wrap, "char").text("ab cdef");
		let mut paint_into = |leaf: &mut TextLeaf, frame: &mut Frame| {
			let mut hits = Vec::new();
			let mut wakes = Vec::new();
			let mut pc = PaintCtx::new(frame, &ctx, &mut hits, &mut wakes);
			leaf.paint(&mut pc, Rect::new(0, 0, 3, 3));
		};

		let mut first = Frame::new(Size::new(3, 3));
		paint_into(&mut leaf, &mut first);
		assert!(first.soft_wrap(0) && first.soft_wrap(1));

		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(3, 3);
		renderer
			.present(first, 3, 0)
			.expect("initial paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert!(terminal.row_wrapped(1) && terminal.row_wrapped(2));

		let mut grown = Frame::new(Size::new(3, 5));
		paint_into(&mut leaf, &mut grown);
		grown.put(0, 3, "x", Style::default());
		grown.put(0, 4, "y", Style::default());
		renderer
			.present(grown, 3, 5)
			.expect("growth paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(terminal.history, ["ab cde"], "the join keeps the written space");

		let mut taller = Frame::new(Size::new(3, 7));
		paint_into(&mut leaf, &mut taller);
		for (offset, line) in ["x", "y", "z", "w"].iter().enumerate() {
			taller.put(0, 3 + u16::try_from(offset).expect("small fixture"), line, Style::default());
		}
		renderer
			.present(taller, 3, 7)
			.expect("second growth succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(
			terminal.history,
			["ab cdef", "x"],
			"copying the committed paragraph reproduces the source bytes"
		);
	}
	#[test]
	fn repeated_seam_advances_preserve_one_ordered_history_copy() {
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 3);

		renderer
			.present(document(&["row00", "row01", "work02", "work03", "footer"]), 3, 2)
			.expect("initial paint succeeds");
		apply_paint(&mut renderer, &mut terminal);

		renderer
			.present(document(&["row00", "row01", "row02", "work03", "work04", "footer"]), 3, 3)
			.expect("first seam advance succeeds");
		apply_paint(&mut renderer, &mut terminal);

		renderer
			.present(
				document(&["row00", "row01", "row02", "row03", "work04", "work05", "footer"]),
				3,
				4,
			)
			.expect("second seam advance succeeds");
		apply_paint(&mut renderer, &mut terminal);

		renderer
			.present(
				document(&["row00", "row01", "row02", "row03", "row04", "work05", "work06", "footer"]),
				3,
				5,
			)
			.expect("third seam advance succeeds");
		apply_paint(&mut renderer, &mut terminal);

		assert_eq!(terminal.history, ["row00", "row01", "row02", "row03", "row04"]);
		assert_eq!(terminal.visible_rows(), ["work05", "work06", "footer"]);
	}

	#[test]
	fn initial_paint_commits_only_stable_overflow() {
		let mut renderer = Renderer::new(Vec::new());

		let stats = renderer
			.present(document(&["one", "two", "three", "four"]), 2, 1)
			.expect("paint succeeds");

		assert!(stats.full_repaint);
		assert_eq!(stats.committed_rows, 1);
		assert_eq!(stats.clipped_rows, 1);
		assert_eq!(renderer.committed_rows(), 1);
	}

	#[test]
	fn clipped_stable_growth_is_deferred_without_replay() {
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 2);
		renderer
			.present(document(&["one", "two", "three", "four", "five"]), 2, 1)
			.expect("initial paint succeeds");
		apply_paint(&mut renderer, &mut terminal);

		let stats = renderer
			.present(document(&["one", "two", "three", "FOUR", "five", "six", "seven"]), 2, 2)
			.expect("clipped stable growth succeeds");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");
		terminal.apply(&output);

		assert_eq!(stats.committed_rows, 0);
		assert_eq!(stats.clipped_rows, 4);
		assert_eq!(renderer.committed_rows(), 1);
		assert_eq!(output.matches("\r\n").count(), 0);
		assert_eq!(terminal.history, ["one"]);
		assert_eq!(terminal.visible_rows(), ["six", "seven"]);

		renderer.writer_mut().clear();
		let error = renderer
			.present(document(&["one", "TWO", "three", "FOUR", "five", "six", "seven"]), 2, 2)
			.expect_err("deferred stable rows remain immutable");
		assert_eq!(error.kind(), ErrorKind::InvalidData);
		assert_eq!(renderer.writer_mut().as_slice(), &[] as &[u8]);
	}

	#[test]
	fn visible_mutation_uses_relative_cursor_without_scrolling() {
		let mut renderer = Renderer::new(Vec::new());
		renderer
			.present(document(&["one", "two", "three", "four"]), 2, 2)
			.expect("first paint succeeds");
		renderer.writer_mut().clear();

		let stats = renderer
			.present(document(&["one", "two", "THREE", "four"]), 2, 2)
			.expect("diff succeeds");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");

		assert_eq!(stats.committed_rows, 0);
		assert!(stats.changed_cells > 0);
		assert!(output.contains("\x1b[1A\r"));
		assert!(output.contains("\x1b[1B\r"));
		assert!(!output.contains("\r\n"));
		assert!(!output.contains("\x1b[1;"));
		assert!(!output.contains("\x1b[2;"));
		assert!(!output.contains("\x1b[2J"));
		assert!(!output.contains("\x1b[3J"));
	}

	#[test]
	fn committed_mutation_is_rejected_without_output() {
		let mut renderer = Renderer::new(Vec::new());
		renderer
			.present(document(&["one", "two", "three", "four"]), 2, 2)
			.expect("first paint succeeds");
		renderer.writer_mut().clear();

		let error = renderer
			.present(document(&["ONE", "two", "three", "four"]), 2, 2)
			.expect_err("stable mutation must fail");

		assert_eq!(error.kind(), ErrorKind::InvalidData);
		assert_eq!(renderer.writer_mut().as_slice(), &[] as &[u8]);
		assert_eq!(renderer.committed_rows(), 2);
	}

	#[test]
	fn damaged_stable_mutation_is_rejected_without_output() {
		let mut renderer = Renderer::new(Vec::new());
		let initial = document(&["one", "two", "three", "four"]);
		renderer
			.present_damaged(&initial, &[(0, 4)], 2, 2)
			.expect("first paint succeeds");
		renderer.writer_mut().clear();

		let changed = document(&["ONE", "two", "three", "four"]);
		let error = renderer
			.present_damaged(&changed, &[(0, 1)], 2, 2)
			.expect_err("reported stable mutation must fail");

		assert_eq!(error.kind(), ErrorKind::InvalidData);
		assert_eq!(renderer.writer_mut().as_slice(), &[] as &[u8]);
		assert_eq!(renderer.committed_rows(), 2);
	}

	#[test]
	fn seam_advance_corrects_final_row_then_scrolls_once() {
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 2);
		renderer
			.present(document(&["one", "two", "three", "four"]), 2, 2)
			.expect("first paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		let stats = renderer
			.present(document(&["one", "two", "THREE", "four", "five"]), 2, 3)
			.expect("seam commit succeeds");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");

		assert_eq!(stats.committed_rows, 1);
		assert_eq!(renderer.committed_rows(), 3);
		assert_eq!(output.matches("\r\n").count(), 1);
		assert!(output.contains(VIEWPORT_BOTTOM));
		assert!(output.contains("THREE"));
		assert!(output.contains("five"));
		terminal.apply(&output);
		assert_eq!(terminal.history, ["one", "two", "THREE"]);
		assert_eq!(terminal.visible_rows(), ["four", "five"]);
		assert!(!output.contains("\x1b[2J"));
		assert!(!output.contains("\x1b[3J"));
	}

	#[test]
	fn scroll_append_keeps_adjacent_box_edges_separate() {
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 5);
		renderer
			.present(
				document(&["old0", "old1", "╰ live─╯", "", "╭──────╮", "│ body │", "footer"]),
				5,
				2,
			)
			.expect("initial paint succeeds");
		apply_paint(&mut renderer, &mut terminal);

		renderer
			.present(
				document(&["old0", "old1", "╰──────╯", "", "╭──────╮", "│ body │", "new", "footer"]),
				5,
				3,
			)
			.expect("box boundary commit succeeds");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");
		terminal.apply(&output);

		assert_eq!(output.matches("\r\n").count(), 1);
		assert!(output.contains(VIEWPORT_BOTTOM));
		assert_eq!(terminal.history, ["old0", "old1", "╰──────╯"]);
		assert_eq!(terminal.visible_rows(), ["", "╭──────╮", "│ body │", "new", "footer"]);
	}

	#[test]
	fn margin_commit_scrolls_history_without_touching_pinned_rows() {
		let mut renderer = Renderer::new(Vec::new());
		renderer.set_margin_scrollback(true);
		let mut terminal = TerminalModel::new(8, 4);
		renderer
			.present(document(&["old0", "old1", "work2", "editor", "footer"]), 4, 2)
			.expect("initial paint succeeds");
		apply_paint(&mut renderer, &mut terminal);

		let stats = renderer
			.present(document(&["old0", "old1", "row2", "row3", "editor", "footer"]), 4, 4)
			.expect("margin commit succeeds");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");
		terminal.apply(&output);

		assert_eq!(stats.committed_rows, 1);
		assert!(output.contains("\x1b[1;2r"));
		assert!(output.contains("\x1b[r"));
		assert_eq!(output.matches("\r\n").count(), 1);
		assert!(
			!output.contains("editor") && !output.contains("footer"),
			"pinned rows must never be re-emitted"
		);
		assert_eq!(terminal.history, ["old0", "old1"]);
		assert_eq!(terminal.visible_rows(), ["row2", "row3", "editor", "footer"]);
	}

	#[test]
	fn margin_commit_matches_whole_screen_scroll_end_state() {
		let mut margin = Renderer::new(Vec::new());
		margin.set_margin_scrollback(true);
		let mut plain = Renderer::new(Vec::new());
		let mut margin_terminal = TerminalModel::new(8, 4);
		let mut plain_terminal = TerminalModel::new(8, 4);

		for (renderer, terminal) in
			[(&mut margin, &mut margin_terminal), (&mut plain, &mut plain_terminal)]
		{
			renderer
				.present(document(&["old0", "old1", "work2", "editor", "footer"]), 4, 2)
				.expect("initial paint succeeds");
			apply_paint(renderer, terminal);
			renderer
				.present(document(&["old0", "old1", "row2", "row3", "editor", "footer"]), 4, 4)
				.expect("commit succeeds");
			apply_paint(renderer, terminal);
		}

		assert_eq!(margin_terminal.history, plain_terminal.history);
		assert_eq!(margin_terminal.visible_rows(), plain_terminal.visible_rows());
	}

	#[test]
	fn margin_commit_repaints_changed_live_rows_in_place() {
		let mut renderer = Renderer::new(Vec::new());
		renderer.set_margin_scrollback(true);
		let mut terminal = TerminalModel::new(8, 4);
		renderer
			.present(document(&["old0", "old1", "work2", "spin0", "footer"]), 4, 2)
			.expect("initial paint succeeds");
		apply_paint(&mut renderer, &mut terminal);

		renderer
			.present(document(&["old0", "old1", "row2", "row3", "pulse", "footer"]), 4, 4)
			.expect("margin commit succeeds");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");
		terminal.apply(&output);

		assert!(output.contains("\x1b[1;2r"), "animated live rows must not shrink the pin");
		assert!(output.contains("pulse"), "changed live cells repaint in place");
		assert!(!output.contains("footer"), "unchanged pinned rows stay untouched");
		assert_eq!(terminal.history, ["old0", "old1"]);
		assert_eq!(terminal.visible_rows(), ["row2", "row3", "pulse", "footer"]);
	}

	#[test]
	fn margin_commit_falls_back_when_stable_seam_reaches_screen_bottom() {
		let mut renderer = Renderer::new(Vec::new());
		renderer.set_margin_scrollback(true);
		let mut terminal = TerminalModel::new(8, 4);
		renderer
			.present(document(&["old0", "old1", "work2", "editor", "footer"]), 4, 2)
			.expect("initial paint succeeds");
		apply_paint(&mut renderer, &mut terminal);

		renderer
			.present(document(&["old0", "old1", "row2", "row3", "editor", "footer"]), 4, 6)
			.expect("fully stable commit succeeds");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");
		terminal.apply(&output);

		assert!(
			!output.contains("\x1b[1;"),
			"a seam at the screen bottom must use the whole-screen scroll"
		);
		assert_eq!(terminal.history, ["old0", "old1"]);
		assert_eq!(terminal.visible_rows(), ["row2", "row3", "editor", "footer"]);
	}

	#[test]
	fn growing_mutable_suffix_is_clipped_without_committing_snapshots() {
		let mut renderer = Renderer::new(Vec::new());
		renderer
			.present(document(&["one", "two", "three", "four"]), 2, 2)
			.expect("first paint succeeds");
		renderer.writer_mut().clear();

		let stats = renderer
			.present(document(&["one", "two", "three", "four", "five"]), 2, 2)
			.expect("virtual shift succeeds");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");

		assert_eq!(stats.committed_rows, 0);
		assert_eq!(stats.clipped_rows, 1);
		assert_eq!(renderer.committed_rows(), 2);
		assert!(stats.changed_cells > 0);
		assert!(output.contains("\x1b[1A\r"));
		assert!(output.contains("\x1b[1C"));
		assert!(!output.contains("\x1b[1;"));
		assert!(!output.contains("\x1b[2;"));
		assert!(!output.contains("\r\n"));
	}

	#[test]
	fn live_collapse_below_history_is_rejected_until_rebuild() {
		let mut renderer = Renderer::new(Vec::new());
		renderer
			.present(document(&["one", "two", "three", "four"]), 2, 2)
			.expect("first paint succeeds");
		renderer.writer_mut().clear();

		let error = renderer
			.present(document(&["one", "two", "three"]), 2, 2)
			.expect_err("a document tail shorter than committed history must be rejected");
		assert_eq!(error.kind(), ErrorKind::InvalidData);
		assert_eq!(renderer.writer_mut().as_slice(), &[] as &[u8]);

		renderer
			.rebuild(document(&["one", "two", "three"]), 2, 2, "")
			.expect("rebuild accepts the shorter document");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");
		assert!(output.contains("\x1b[3J"));
		assert_eq!(renderer.committed_rows(), 1);
	}

	#[test]
	fn stable_boundary_cannot_retreat() {
		let mut renderer = Renderer::new(Vec::new());
		renderer
			.present(document(&["one", "two", "three"]), 2, 2)
			.expect("first paint succeeds");
		renderer.writer_mut().clear();

		let error = renderer
			.present(document(&["one", "two", "three"]), 2, 1)
			.expect_err("retreat must fail");

		assert_eq!(error.kind(), ErrorKind::InvalidData);
		assert_eq!(renderer.writer_mut().as_slice(), &[] as &[u8]);
	}

	#[test]
	fn resize_preview_leaves_normal_renderer_state_untouched() {
		let mut renderer = Renderer::new(Vec::new());
		renderer
			.present(document(&["row00", "row01", "work02", "footer"]), 2, 2)
			.expect("initial paint succeeds");
		renderer.writer_mut().clear();

		let preview = document(&["new00", "new01", "live02", "live03", "footer"]);
		let stats = renderer
			.preview(&preview, 3, "\x1b[?1049h")
			.expect("alternate viewport preview succeeds");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");

		assert!(stats.full_repaint);
		assert_eq!(renderer.committed_rows(), 2);
		assert!(output.contains("\x1b[?1049h"));
		assert!(output.contains("live02"));
		assert!(output.contains("footer"));
		assert!(!output.contains("new00"));
		assert!(!output.contains("\x1b[3J"));

		renderer.writer_mut().clear();
		let stats = renderer
			.present(document(&["row00", "row01", "work02", "footer"]), 2, 2)
			.expect("normal state still matches its pre-preview frame");
		assert_eq!(stats.bytes, 0);
	}

	#[test]
	fn settled_resize_clears_and_rebuilds_history_once() {
		let mut renderer = Renderer::new(Vec::new());
		let mut terminal = TerminalModel::new(8, 3);
		renderer
			.present(document(&["old00", "old01", "old02", "old03", "old04"]), 3, 4)
			.expect("initial paint succeeds");
		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(terminal.history, ["old00", "old01"]);

		terminal.resize(8, 2);
		let stats = renderer
			.rebuild(document(&["new00", "new01", "new02", "live03", "footer"]), 2, 3, "\x1b[?1049l")
			.expect("settled resize rebuild succeeds");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");

		let sync = output.find(SYNC_OUTPUT_BEGIN).expect("synchronized paint");
		let alt_exit = output.find("\x1b[?1049l").expect("alternate-buffer exit");
		let clear = output.find(REBUILD_HISTORY).expect("history clear");
		assert!(sync < alt_exit && alt_exit < clear);
		assert_eq!(output.matches("\x1b[3J").count(), 1);
		assert!(!output.contains("\x1b[2J"));
		assert!(stats.full_repaint);
		assert_eq!(stats.committed_rows, 3);

		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(terminal.history, ["new00", "new01", "new02"]);
		assert_eq!(terminal.visible_rows(), ["live03", "footer"]);

		let stats = renderer
			.present(document(&["new00", "new01", "new02", "live03", "footer"]), 2, 3)
			.expect("incremental rendering resumes from rebuilt state");
		assert_eq!(stats.bytes, 0);

		let stats = renderer
			.present(document(&["new00", "new01", "new02", "new03", "live04", "footer"]), 2, 4)
			.expect("immutable seam advances after the rebuild");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");
		assert_eq!(stats.committed_rows, 1);
		assert!(!output.contains("\x1b[3J"));

		apply_paint(&mut renderer, &mut terminal);
		assert_eq!(terminal.history, ["new00", "new01", "new02", "new03"]);
		assert_eq!(terminal.visible_rows(), ["live04", "footer"]);
	}

	#[test]
	fn hardware_cursor_moves_without_repainting_cells() {
		let mut renderer = Renderer::new(Vec::new());
		let mut first = document(&["one", "two"]);
		first.set_cursor(1, 1);
		renderer.present(first, 2, 2).expect("first paint succeeds");
		renderer.writer_mut().clear();

		let mut second = document(&["one", "two"]);
		second.set_cursor(4, 0);
		let stats = renderer
			.present(second, 2, 2)
			.expect("cursor move succeeds");
		let output = String::from_utf8(renderer.writer_mut().clone()).expect("ANSI is UTF-8");

		assert_eq!(stats.changed_cells, 0);
		assert_eq!(stats.runs, 0);
		assert!(stats.bytes > 0);
		assert!(output.contains("\x1b[?25l"));
		assert!(output.contains("\x1b[1A\r\x1b[4C\x1b[?25h"));
		assert!(!output.contains("\r\n"));
		assert!(!output.contains("\x1b[H"));
	}

	#[test]
	fn identical_document_writes_nothing() {
		let mut renderer = Renderer::new(Vec::new());
		renderer
			.present(document(&["one", "two"]), 2, 2)
			.expect("first paint succeeds");
		renderer.writer_mut().clear();

		let stats = renderer
			.present(document(&["one", "two"]), 2, 2)
			.expect("second paint succeeds");

		assert_eq!(stats.bytes, 0);
		assert_eq!(renderer.writer_mut().as_slice(), &[] as &[u8]);
	}

	#[test]
	fn kitty_images_upload_place_and_materialize_typed_cells() {
		let mut frame = Frame::new(Size::new(3, 2));
		frame.put_image_cell(0, 0, 0x12_34_56, 0, 0, 2, 3);
		frame.put_image_cell(2, 1, 0x12_34_56, 1, 2, 2, 3);
		let mut renderer = Renderer::new(Vec::new());
		renderer
			.register_image(0x12_34_56, vec![0x5a; 3073])
			.unwrap();
		renderer.present(frame, 2, 0).unwrap();
		let output = String::from_utf8(renderer.into_inner()).unwrap();

		// Transmission rides the synchronized paint, after the cursor hide,
		// so a staged buffer switch in the leading sequence precedes it.
		assert!(output.contains("\x1b_Gf=100,t=d,a=t,i=1193046,q=2,m=1;"));
		assert!(output.contains("\x1b_Gm=0;"));
		assert!(output.contains("\x1b_Ga=p,U=1,i=1193046,p=1027,r=2,c=3,q=2\x1b\\"));
		assert!(output.contains("\u{10eeee}\u{0305}\u{0305}"));
		assert!(output.contains("\u{10eeee}\u{030d}\u{030e}"));
		// Image ID in the foreground, placement ID (2<<9|3 = 1027) in the
		// underline color.
		assert!(output.contains("38;2;18;52;86m"));
		assert!(output.contains("58:2::0:4:3"));

		let packets = output
			.split("\x1b\\")
			.filter_map(|piece| piece.find("\x1b_G").map(|start| &piece[start..]))
			.filter(|packet| packet.contains(';'))
			.collect::<Vec<_>>();
		assert_eq!(packets.len(), 2);
		assert!(
			packets
				.iter()
				.all(|packet| packet.split_once(';').unwrap().1.len() <= 4096)
		);
	}

	#[test]
	fn clipped_kitty_image_uses_full_placement_without_scroll_rescaling() {
		fn clipped_image(first_row: u16) -> Frame {
			let mut frame = Frame::new(Size::new(8, 4));
			for row in first_row..4 {
				for col in 0..8 {
					frame.put_image_cell(col, row, 7, row, col, 4, 8);
				}
			}
			frame
		}

		let mut renderer = Renderer::new(Vec::new());
		renderer.register_image(7, vec![1, 2, 3]).unwrap();
		renderer.present(clipped_image(3), 4, 0).unwrap();
		let initial = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		assert!(initial.contains("\x1b_Ga=p,U=1,i=7,p=2056,r=4,c=8,q=2\x1b\\"));

		for first_row in (0..3).rev() {
			renderer.present(clipped_image(first_row), 4, 0).unwrap();
			let output = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
			assert!(
				!output.contains("\x1b_Ga=p"),
				"declared 4x8 dimensions must not be re-placed when row {first_row} enters"
			);
		}
	}

	#[test]
	fn distinct_cell_boxes_of_one_image_place_once_each_with_stable_ids() {
		// Two boxes of image 7 in one frame: a 1x2 thumbnail and a 2x4 card.
		let mut frame = Frame::new(Size::new(8, 3));
		for col in 0..2 {
			frame.put_image_cell(col, 0, 7, 0, col, 1, 2);
		}
		for row in 0..2 {
			for col in 0..4 {
				frame.put_image_cell(col, 1 + row, 7, row, col, 2, 4);
			}
		}
		let mut renderer = Renderer::new(Vec::new());
		renderer
			.register_image(7, b"\x89PNG\r\n\x1a\nsmall".to_vec())
			.unwrap();
		renderer.present(frame.clone(), 3, 0).unwrap();
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		assert_eq!(output.matches("a=t").count(), 1, "one upload serves every box");
		assert!(output.contains("\x1b_Ga=p,U=1,i=7,p=514,r=1,c=2,q=2\x1b\\"));
		assert!(output.contains("\x1b_Ga=p,U=1,i=7,p=1028,r=2,c=4,q=2\x1b\\"));
		// Placeholder cells reference their box's placement via the
		// underline color: 514 = 0:2:2, 1028 = 0:4:4.
		assert!(output.contains("58:2::0:2:2"));
		assert!(output.contains("58:2::0:4:4"));

		// Identical re-present: placements are session-cached, never re-sent.
		renderer.present(frame, 3, 0).unwrap();
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		assert!(!output.contains("\x1b_G"), "no image traffic on a settled frame: {output:?}");
	}

	#[test]
	fn screen_buffer_switch_retransmits_and_replaces_images() {
		// Ghostty stores Kitty images per screen: transmissions and virtual
		// placements made on one buffer do not exist on the other. Paints
		// resync against the process flag (main under tests), so flipping the
		// tracked buffer makes the next paint observe a switch.
		let mut frame = Frame::new(Size::new(8, 3));
		for col in 0..2 {
			frame.put_image_cell(col, 0, 7, 0, col, 1, 2);
		}
		let mut renderer = Renderer::new(Vec::new());
		renderer
			.register_image(7, b"\x89PNG\r\n\x1a\nsmall".to_vec())
			.unwrap();
		renderer.present(frame.clone(), 3, 0).unwrap();
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		assert_eq!(output.matches("a=t").count(), 1, "first paint uploads: {output:?}");

		renderer.preview(&frame, 3, "").unwrap();
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		assert!(!output.contains("\x1b_G"), "no retransmit within one buffer: {output:?}");

		renderer.set_screen_buffer(true);
		renderer.preview(&frame, 3, "").unwrap();
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		assert_eq!(
			output.matches("a=t").count(),
			1,
			"a buffer switch re-uploads to the new screen's store: {output:?}"
		);
		assert!(
			output.contains("\x1b_Ga=p,U=1,i=7,p=514,r=1,c=2,q=2\x1b\\"),
			"virtual placements are re-created after a switch: {output:?}"
		);

		renderer.preview(&frame, 3, "").unwrap();
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		assert!(!output.contains("\x1b_G"), "caches hold once settled: {output:?}");
	}

	#[test]
	fn alt_entry_preview_uploads_overlay_only_images_after_the_switch() {
		// The model picker's logos live only in its overlay layer, and its
		// alt-screen entry rides the preview's leading sequence: the images
		// must be collected from the layers and their Kitty traffic must land
		// after `?1049h`, or a per-screen store (ghostty) files them under
		// the buffer being left.
		let base = document(&["base000", "base111"]);
		let mut overlay = Frame::new(Size::new(3, 1));
		for col in 0..2 {
			overlay.put_image_cell(col, 0, 7, 0, col, 1, 2);
		}
		let layer = ResolvedLayer {
			frame:   &overlay,
			x:       0,
			y:       0,
			src_top: 0,
			rows:    1,
			active:  false,
		};
		let mut renderer = Renderer::new(Vec::new());
		renderer
			.register_image(7, b"\x89PNG\r\n\x1a\nsmall".to_vec())
			.unwrap();

		renderer
			.preview_resolved(&base, &[layer], 2, "\x1b[?1049h")
			.unwrap();
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		let switch = output
			.find("\x1b[?1049h")
			.expect("staged alt entry is emitted");
		let upload = output
			.find("\x1b_Gf=100,t=d,a=t")
			.expect("overlay-only image uploads");
		assert!(switch < upload, "upload must follow the buffer switch: {output:?}");
		assert!(
			output.contains("\x1b_Ga=p,U=1,i=7,p=514,r=1,c=2,q=2\x1b\\"),
			"overlay-only image is placed: {output:?}"
		);

		renderer.preview_resolved(&base, &[layer], 2, "").unwrap();
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		assert!(!output.contains("\x1b_G"), "steady overlay preview stays quiet: {output:?}");

		// Leaving the hold flips the tracked buffer again: the exit
		// retransmission must follow `?1049l` for the same reason.
		renderer.set_screen_buffer(true);
		renderer
			.preview_resolved(&base, &[layer], 2, "\x1b[?1049l")
			.unwrap();
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		let switch = output
			.find("\x1b[?1049l")
			.expect("staged alt exit is emitted");
		let upload = output
			.find("\x1b_Gf=100,t=d,a=t")
			.expect("the other buffer needs its own upload");
		assert!(switch < upload, "re-upload must follow the buffer switch: {output:?}");
	}

	#[test]
	fn staged_buffer_switch_precedes_image_uploads() {
		// Per-screen image stores only keep what arrives on the active
		// screen, so a staged `1049h`/`1049l` must hit the wire before any
		// Kitty upload in the same paint.
		let mut frame = Frame::new(Size::new(8, 3));
		for col in 0..2 {
			frame.put_image_cell(col, 0, 7, 0, col, 1, 2);
		}
		let mut renderer = Renderer::new(Vec::new());
		renderer
			.register_image(7, b"\x89PNG\r\n\x1a\nsmall".to_vec())
			.unwrap();
		renderer.present(frame.clone(), 3, 0).unwrap();
		renderer.writer_mut().clear();

		renderer.set_screen_buffer(true);
		renderer.preview(&frame, 3, "\x1b[?1049h").unwrap();
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		let switch = output.find("\x1b[?1049h").expect("staged entry is emitted");
		let upload = output.find("a=t").expect("the new screen needs an upload");
		assert!(switch < upload, "upload lands on the freshly entered screen: {output:?}");

		// Paints resync the tracked buffer to the process flag (main under
		// tests), so a fresh flip is needed to observe the exit switch.
		renderer.set_screen_buffer(true);
		renderer.rebuild(frame, 3, 0, "\x1b[?1049l").unwrap();
		let output = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		let switch = output.find("\x1b[?1049l").expect("staged exit is emitted");
		let upload = output
			.find("a=t")
			.expect("the restored screen needs an upload");
		assert!(switch < upload, "upload lands on the restored screen: {output:?}");
		let clear = output
			.find(REBUILD_HISTORY)
			.expect("history clear is emitted");
		assert!(clear < upload, "upload must follow the history clear: {output:?}");
	}
	#[test]
	fn kitty_direct_crops_replaces_without_retransmit_and_deletes_offscreen() {
		fn png_fixture() -> Vec<u8> {
			let mut bytes = Vec::new();
			{
				let mut encoder = png::Encoder::new(&mut bytes, 2, 4);
				encoder.set_color(png::ColorType::Rgb);
				encoder.set_depth(png::BitDepth::Eight);
				let mut writer = encoder.write_header().unwrap();
				writer.write_image_data(&[0x7f; 24]).unwrap();
			}
			bytes
		}

		fn image_frame(top: u16) -> Frame {
			let mut frame = Frame::new(Size::new(4, 6));
			for row in 0..4 {
				let y = top + row;
				if y >= frame.size().height {
					break;
				}
				frame.put(0, y, "L", Style::default());
				for col in 0..2 {
					frame.put_image_cell(1 + col, y, 7, row, col, 4, 2);
				}
				frame.put(3, y, "R", Style::default());
			}
			frame
		}

		let mut renderer = Renderer::new(Vec::new());
		renderer.set_graphics(Graphics::KittyDirect);
		renderer.set_cell_pixel_size(1, 1).unwrap();
		renderer.register_image(7, png_fixture()).unwrap();

		renderer.present(image_frame(0), 4, 0).unwrap();
		let clipped = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		assert_eq!(clipped.matches("a=t").count(), 1);
		assert!(
			clipped
				.contains("\x1b[3A\r\x1b[1C\x1b_Ga=p,q=2,C=1,i=7,p=7,x=0,y=2,w=2,h=2,c=2,r=2\x1b\\")
		);
		assert!(clipped.contains("L  R"));
		assert!(!clipped.contains("\u{10eeee}"));

		renderer.present(image_frame(2), 4, 0).unwrap();
		let fully_visible = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		assert!(!fully_visible.contains("a=t"));
		assert!(
			fully_visible
				.contains("\x1b[3A\r\x1b[1C\x1b_Ga=p,q=2,C=1,i=7,p=7,x=0,y=0,w=2,h=4,c=2,r=4\x1b\\")
		);

		renderer.present(Frame::new(Size::new(4, 6)), 4, 0).unwrap();
		let offscreen = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		assert!(offscreen.contains("\x1b_Ga=d,d=I,i=7,q=2\x1b\\"));
	}

	#[test]
	fn tmux_passthrough_wraps_kitty_packets_but_not_text_sgr() {
		let mut frame = Frame::new(Size::new(2, 1));
		frame.put_image_cell(0, 0, 7, 0, 0, 1, 1);
		frame.put(1, 0, "X", Style::new().fg(Color::Rgb(1, 2, 3)));
		let mut renderer = Renderer::new(Vec::new());
		renderer.set_graphics(Graphics::KittyDirect);
		renderer.set_tmux_passthrough(true);
		renderer.set_cell_pixel_size(1, 1).unwrap();
		renderer.register_image(7, [1, 2, 3]).unwrap();

		renderer.present(frame, 1, 0).unwrap();
		let output = String::from_utf8(renderer.into_inner()).unwrap();
		assert!(
			output.contains("\x1bPtmux;\x1b\x1b_Gf=100,t=d,a=t,i=7,q=2,m=0;AQID\x1b\x1b\\\x1b\\")
		);
		assert!(output.contains(
			"\x1bPtmux;\x1b\x1b_Ga=p,q=2,C=1,i=7,p=7,x=0,y=0,w=1,h=1,c=1,r=1\x1b\x1b\\\x1b\\"
		));
		assert_eq!(output.matches("\x1bPtmux;").count(), output.matches("_G").count());
		assert!(output.contains("\x1b[38;2;1;2;3mX"));
		assert!(!output.contains("\x1bPtmux;\x1b\x1b[38;2;1;2;3m"));
	}

	#[test]
	fn sixel_images_crop_reemit_and_leave_text_cells_intact() {
		fn png_fixture() -> Vec<u8> {
			let mut bytes = Vec::new();
			{
				let mut encoder = png::Encoder::new(&mut bytes, 4, 2);
				encoder.set_color(png::ColorType::Rgb);
				encoder.set_depth(png::BitDepth::Eight);
				let mut writer = encoder.write_header().unwrap();
				writer
					.write_image_data(&[
						255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 0, 255, 0,
						0, 255,
					])
					.unwrap();
			}
			bytes
		}

		fn image_frame(top: u16, height: u16) -> Frame {
			let mut frame = Frame::new(Size::new(10, height));
			for row in 0..4 {
				let y = top + row;
				if y >= frame.size().height {
					break;
				}
				frame.put(0, y, "L", Style::default());
				for col in 0..8 {
					frame.put_image_cell(1 + col, y, 7, row, col, 4, 8);
				}
				frame.put(9, y, "R", Style::default());
			}
			frame
		}

		let mut renderer = Renderer::new(Vec::new());
		renderer.set_graphics(Graphics::Sixel);
		renderer.set_cell_pixel_size(1, 1).unwrap();
		renderer.register_image(7, png_fixture()).unwrap();

		renderer.present(image_frame(0, 6), 4, 0).unwrap();
		let clipped = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		assert!(clipped.contains("\x1b[3A\r\x1b[1C\x1bP0;1;0q"));
		assert!(clipped.contains("\"1;1;8;2"));
		assert!(!clipped.contains("\x1b_G"));
		assert!(clipped.contains("L        R"));

		let fully_visible = image_frame(4, 8);
		renderer.present(fully_visible.clone(), 4, 0).unwrap();
		let moved = String::from_utf8(std::mem::take(renderer.writer_mut())).unwrap();
		assert!(moved.contains("\x1b[3A\r\x1b[1C\x1bP0;1;0q"));
		assert!(moved.contains("\"1;1;8;4"));
		assert!(!moved.contains("\x1b_G"));
		assert!(moved.contains("L        R"));

		let stats = renderer.present(fully_visible, 4, 0).unwrap();
		assert_eq!(stats.bytes, 0);
		assert_eq!(renderer.writer_mut().as_slice(), &[] as &[u8]);

		let mut tmux = Renderer::new(Vec::new());
		tmux.set_graphics(Graphics::Sixel);
		tmux.set_tmux_passthrough(true);
		tmux.set_cell_pixel_size(1, 1).unwrap();
		tmux.register_image(7, png_fixture()).unwrap();
		tmux.present(image_frame(0, 4), 4, 0).unwrap();
		let wrapped = String::from_utf8(tmux.into_inner()).unwrap();
		assert!(wrapped.contains("\x1bPtmux;\x1b\x1bP0;1;0q"));
		assert!(wrapped.contains("\x1b\x1b\\\x1b\\"));
		assert!(!wrapped.contains("\x1b_G"));
	}
}
