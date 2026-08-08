use omp_core::smolstr::IntoSmolStr;

use crate::{
	component::{Component, PaintCtx, Slot, next_slot},
	context::{Graphics, UiContext},
	frame::{Color, Rect, Style},
	imagefmt::{self, ImageDimensions},
	kitty::PLACEHOLDER_LIMIT,
	markup::{Border, Dim},
	props::{Prop, PropValue, Props},
};

type Rgb = [u8; 3];
type CellColors = (Option<Rgb>, Option<Rgb>);

#[derive(Clone, Copy, Default)]
enum AutoBox {
	#[default]
	Unresolved,
	Resolved(Option<(u32, u16, u16)>),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Load {
	#[default]
	Unloaded,
	Loading,
	Ready,
	Boxed,
}

#[derive(Default)]
pub struct ImgState {
	/// Half-block colors per cell; `None` halves are transparent and leave
	/// the underlying background visible.
	cells: Box<[CellColors]>,
	width: u16,
	rows:  u16,
	phase: Load,
}

impl ImgState {
	fn row(&self, index: u16) -> &[CellColors] {
		let stride = usize::from(self.width);
		let start = usize::from(index) * stride;
		&self.cells[start..start + stride]
	}
}

/// A terminal-rendered image backing the `<img>` markup tag.
///
/// On the Kitty-placeholder graphics tier a PNG `src` renders as real pixels
/// with no further setup: the source is interned process-wide, uploaded by
/// the renderer on first reference, and placed in the cell box derived from
/// `w`/`h` (aspect-derived when `h` is omitted). On every other tier, PNG
/// and binary PPM sources decode to colored half-block cells. JPEG, GIF,
/// and WebP sources are header-probed only: the component reserves their
/// aspect-correct cell box and paints a themed placeholder. The `trim` flag
/// crops fully transparent margins before half-block sampling (terminal
/// compositors always show the full source), so padded logo sources stay
/// visible even as tiny thumbnails.
pub struct Img {
	props:  Props,
	slot:   Slot,
	state:  ImgState,
	kitty:  Option<(u32, u16, u16)>,
	/// Cached `src`-interned placeholder box, resolved at most once.
	auto:   AutoBox,
	top:    String,
	bottom: String,
}

impl Img {
	/// Creates an image with no source.
	pub fn new() -> Self {
		Self {
			props:  Props::new(),
			slot:   next_slot(),
			state:  ImgState::default(),
			kitty:  None,
			auto:   AutoBox::Unresolved,
			top:    String::new(),
			bottom: String::new(),
		}
	}

	/// Sets one image property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one image property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Uses a renderer-registered image ID in a fixed cell box on every
	/// pixel-capable graphics tier, overriding the `src` placeholder path.
	///
	/// Pair with [`crate::Renderer::register_image`]. Rebuild the component
	/// with new dimensions after a resize. Dimensions beyond Kitty's
	/// 297-entry coordinate table leave the component on its cell fallback.
	pub const fn kitty(mut self, id: u32, rows: u16, cols: u16) -> Self {
		if rows > 0 && cols > 0 && rows <= PLACEHOLDER_LIMIT && cols <= PLACEHOLDER_LIMIT {
			self.kitty = Some((id, rows, cols));
		}
		self
	}

	/// The typed-cell box for this context: the explicit [`Img::kitty`] box
	/// on any pixel tier, else a `src`-interned placeholder box on the
	/// Kitty-placeholder tier. `None` selects the half-block/box fallback.
	fn cell_box(&mut self, ctx: &UiContext) -> Option<(u32, u16, u16)> {
		if ctx.graphics == Graphics::Cells {
			return None;
		}
		if self.kitty.is_some() {
			return self.kitty;
		}
		if ctx.graphics != Graphics::KittyPlaceholders {
			return None;
		}
		if matches!(self.auto, AutoBox::Unresolved) {
			self.auto = AutoBox::Resolved(resolve_placeholder_box(&self.props));
		}
		let AutoBox::Resolved(cell_box) = self.auto else {
			unreachable!("placeholder resolution was initialized");
		};
		cell_box
	}

	fn requested_width(&self, available: u16) -> u16 {
		match self.props.w() {
			Some(Dim::Cells(cells)) => cells,
			Some(Dim::Pct(percent)) => (u32::from(available) * u32::from(percent) / 100).max(1) as u16,
			None => 24,
		}
		.min(available.max(1))
	}

	fn ensure_decoded(&mut self, ctx: &UiContext, available: u16) {
		if self.state.phase != Load::Unloaded {
			return;
		}
		let source = self
			.props
			.str_of(Prop::Src)
			.map_or("", |value| value.as_str());
		let width = self.requested_width(available);
		let trim = self.props.flag(Prop::Trim);
		if let Some(loader) = &ctx.loader {
			loader.request(self.slot, source.to_smolstr(), width, self.props.h(), trim);
			self.state.phase = Load::Loading;
			self.state.width = width;
			self.state.rows = 3;
		} else {
			self.state = decode_source(source, width, self.props.h(), trim);
		}
	}

	/// Installs an off-thread decode result; ignores stale deliveries after
	/// the state already settled.
	pub(crate) fn apply_decoded(&mut self, state: ImgState) {
		if self.state.phase == Load::Loading {
			self.state = state;
		}
	}
}

/// Resolves an interned placeholder box from `src`, `w`, and `h` props:
/// PNG-only, fixed-cell widths only, aspect-derived rows when `h` is
/// omitted, bounded by Kitty's diacritic table.
fn resolve_placeholder_box(props: &Props) -> Option<(u32, u16, u16)> {
	let source = props.str_of(Prop::Src)?;
	let interned = crate::imagereg::intern(source.as_str())?;
	let cols = match props.w() {
		Some(Dim::Cells(cells)) => cells,
		Some(Dim::Pct(_)) => return None,
		None => 24,
	};
	let rows = props.h().unwrap_or_else(|| {
		let scaled = u64::from(cols) * u64::from(interned.dimensions.height);
		let denominator = u64::from(interned.dimensions.width) * 2;
		((scaled + denominator / 2) / denominator)
			.max(1)
			.min(u64::from(PLACEHOLDER_LIMIT)) as u16
	});
	(rows > 0 && cols > 0 && rows <= PLACEHOLDER_LIMIT && cols <= PLACEHOLDER_LIMIT).then_some((
		interned.id,
		rows,
		cols,
	))
}

impl Default for Img {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Img {
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
		if let Some((_, rows, cols)) = self.cell_box(ctx) {
			return (cols, rows);
		}
		let width = match self.props.w() {
			Some(Dim::Cells(width)) => width,
			_ => 24,
		};
		(width, width)
	}

	fn height(&mut self, ctx: &UiContext, width: u16) -> u16 {
		if let Some((_, rows, _)) = self.cell_box(ctx) {
			return rows;
		}
		self.ensure_decoded(ctx, width);
		self.state.rows
	}

	fn place(&mut self, ctx: &UiContext, content: Rect) {
		if self.cell_box(ctx).is_none() {
			self.ensure_decoded(ctx, content.width);
		}
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		if let Some((id, rows, cols)) = self.cell_box(pc.ctx) {
			for row in 0..rows.min(rect.height) {
				let y = rect.y + row;
				if y >= pc.clip {
					break;
				}
				for col in 0..cols.min(rect.width) {
					pc.frame
						.put_image_cell(rect.x + col, y, id, row, col, rows, cols);
				}
			}
			return;
		}
		self.ensure_decoded(pc.ctx, rect.width);
		if self.state.phase != Load::Ready {
			let source = self
				.props
				.str_of(Prop::Src)
				.map_or("", |value| value.as_str());
			let name = source.rsplit('/').next().unwrap_or("image");
			let width = self.state.width.min(rect.width);
			let rows = self.state.rows.min(rect.height);
			if width == 0 || rows == 0 {
				return;
			}
			let (tl, tr, bl, br, horizontal, _) = pc.ctx.charset.border(Border::Square);
			self.top.clear();
			self.bottom.clear();
			self.top.reserve(usize::from(width));
			self.bottom.reserve(usize::from(width));
			self.top.push(tl);
			self.bottom.push(bl);
			for _ in 0..width.saturating_sub(2) {
				self.top.push(horizontal);
				self.bottom.push(horizontal);
			}
			if width > 1 {
				self.top.push(tr);
				self.bottom.push(br);
			}
			let style = Style::new().fg(pc.ctx.theme.muted);
			for row in 0..rows {
				let y = rect.y + row;
				if y >= pc.clip {
					break;
				}
				if row == 0 {
					pc.frame.put(rect.x, y, &self.top, style);
				} else if row + 1 == rows {
					pc.frame.put(rect.x, y, &self.bottom, style);
				} else {
					let rail = pc.ctx.charset.icon(crate::Icon::PlaceholderRail);
					pc.frame.put(rect.x, y, rail, style);
					if row == rows / 2 && width > 4 {
						let mut x = pc.frame.put(rect.x + 2, y, "[img: ", style);
						x = pc.frame.put(x, y, name, style);
						pc.frame.put(x, y, "]", style);
					}
					if width > 1 {
						pc.frame.put(rect.x + width - 1, y, rail, style);
					}
				}
			}
			return;
		}
		for row_index in 0..self.state.rows {
			let y = rect.y + row_index;
			if y >= pc.clip {
				break;
			}
			let mut x = rect.x;
			for (upper, lower) in self.state.row(row_index) {
				// Transparent halves stay unpainted so the terminal or
				// container background shows through.
				let cell = match (upper, lower) {
					(Some(upper), Some(lower)) => Some((
						crate::Icon::UpperHalf,
						Style::new()
							.fg(Color::Rgb(upper[0], upper[1], upper[2]))
							.bg(Color::Rgb(lower[0], lower[1], lower[2])),
					)),
					(Some(upper), None) => Some((
						crate::Icon::UpperHalf,
						Style::new().fg(Color::Rgb(upper[0], upper[1], upper[2])),
					)),
					(None, Some(lower)) => Some((
						crate::Icon::LowerHalf,
						Style::new().fg(Color::Rgb(lower[0], lower[1], lower[2])),
					)),
					(None, None) => None,
				};
				x = match cell {
					Some((icon, style)) => pc.frame.put(x, y, pc.ctx.charset.icon(icon), style),
					None => x.saturating_add(1),
				};
			}
		}
	}
}

enum DecodedImage {
	Pixels(Vec<Vec<[u8; 4]>>),
	Placeholder(ImageDimensions),
}
/// Reads, decodes, and cell-samples `source` at `width_cells`. Never
/// panics; a settled no-pixel outcome (failure or probe-only format)
/// returns a [`Load::Boxed`] state.
pub fn decode_source(
	source: &str,
	width_cells: u16,
	height_cells: Option<u16>,
	trim: bool,
) -> ImgState {
	match decode_image(source) {
		Some(DecodedImage::Pixels(mut pixels))
			if !pixels.is_empty() && pixels.first().is_some_and(|row| !row.is_empty()) =>
		{
			if trim {
				pixels = trim_transparent(pixels);
			}
			let gate = if trim { TRIMMED_PAINT_GATE } else { PAINT_GATE };
			sample_cells(&pixels, width_cells, height_cells, gate)
		},
		Some(DecodedImage::Placeholder(dimensions)) => {
			placeholder_state(dimensions, width_cells, height_cells)
		},
		_ => {
			ImgState { cells: Box::default(), width: width_cells.max(1), rows: 3, phase: Load::Boxed }
		},
	}
}

/// Crops rows and columns whose pixels are all (nearly) transparent, so a
/// padded logo fills its cell box instead of averaging away. A fully
/// transparent image is returned unchanged.
fn trim_transparent(pixels: Vec<Vec<[u8; 4]>>) -> Vec<Vec<[u8; 4]>> {
	const VISIBLE: u8 = 8;
	let mut top = None;
	let mut bottom = 0_usize;
	let mut left = usize::MAX;
	let mut right = 0_usize;
	for (y, row) in pixels.iter().enumerate() {
		for (x, pixel) in row.iter().enumerate() {
			if pixel[3] >= VISIBLE {
				top.get_or_insert(y);
				bottom = y;
				left = left.min(x);
				right = right.max(x);
			}
		}
	}
	let Some(top) = top else {
		return pixels;
	};
	pixels[top..=bottom]
		.iter()
		.map(|row| row[left.min(row.len() - 1)..=right.min(row.len() - 1)].to_vec())
		.collect()
}

fn decode_image(path: &str) -> Option<DecodedImage> {
	let bytes = std::fs::read(path).ok()?;
	if bytes.starts_with(b"P6") {
		return decode_ppm(&bytes).map(DecodedImage::Pixels);
	}
	let dimensions = imagefmt::dimensions(&bytes)?;
	if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
		return Some(DecodedImage::Placeholder(dimensions));
	}
	decode_png(&bytes)
		.map(DecodedImage::Pixels)
		.or(Some(DecodedImage::Placeholder(dimensions)))
}

fn decode_png(bytes: &[u8]) -> Option<Vec<Vec<[u8; 4]>>> {
	let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
	// Official logos frequently ship indexed palettes (with tRNS alpha) or
	// 16-bit channels; normalize so `samples()` below always describes
	// plain 8-bit gray/RGB(A) output instead of palette indices.
	decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
	let mut reader = decoder.read_info().ok()?;
	let mut buffer = vec![0_u8; reader.output_buffer_size()?];
	let info = reader.next_frame(&mut buffer).ok()?;
	let (width, height) = (info.width as usize, info.height as usize);
	let stride = info.color_type.samples();
	let mut rows = Vec::with_capacity(height);
	for y in 0..height {
		let mut row = Vec::with_capacity(width);
		for x in 0..width {
			let at = y * width * stride + x * stride;
			row.push(match stride {
				1 => [buffer[at], buffer[at], buffer[at], 255],
				2 => [buffer[at], buffer[at], buffer[at], buffer[at + 1]],
				3 => [buffer[at], buffer[at + 1], buffer[at + 2], 255],
				_ => [buffer[at], buffer[at + 1], buffer[at + 2], buffer[at + 3]],
			});
		}
		rows.push(row);
	}
	Some(rows)
}

fn decode_ppm(bytes: &[u8]) -> Option<Vec<Vec<[u8; 4]>>> {
	let mut fields = Vec::new();
	let mut at = 2_usize;
	while fields.len() < 3 && at < bytes.len() {
		while at < bytes.len() && bytes[at].is_ascii_whitespace() {
			at += 1;
		}
		if bytes.get(at) == Some(&b'#') {
			while at < bytes.len() && bytes[at] != b'\n' {
				at += 1;
			}
			continue;
		}
		let start = at;
		while at < bytes.len() && bytes[at].is_ascii_digit() {
			at += 1;
		}
		fields.push(
			std::str::from_utf8(&bytes[start..at])
				.ok()?
				.parse::<usize>()
				.ok()?,
		);
	}
	at += 1;
	let (&width, &height) = (fields.first()?, fields.get(1)?);
	let data = bytes.get(at..at + width * height * 3)?;
	Some(
		data
			.chunks_exact(width * 3)
			.map(|row| {
				row.as_chunks::<3>()
					.0
					.iter()
					.map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
					.collect()
			})
			.collect(),
	)
}

fn placeholder_state(
	dimensions: ImageDimensions,
	width_cells: u16,
	height_cells: Option<u16>,
) -> ImgState {
	let rows = height_cells.unwrap_or_else(|| {
		let scaled = u64::from(width_cells) * u64::from(dimensions.height);
		let denominator = u64::from(dimensions.width) * 2;
		((scaled + denominator / 2) / denominator)
			.max(1)
			.min(u64::from(u16::MAX)) as u16
	});
	ImgState { cells: Box::default(), width: width_cells.max(1), rows, phase: Load::Boxed }
}

/// Paint gate for untrimmed sources: a half-cell must be at least half
/// covered, so transparent logo padding never paints.
const PAINT_GATE: u64 = 128;
/// Paint gate for trimmed thumbnails: the crop already removed padding, so
/// any half-cell with meaningful coverage (≥ 12.5%) keeps its glyph color.
const TRIMMED_PAINT_GATE: u64 = 32;

fn sample_cells(
	pixels: &[Vec<[u8; 4]>],
	width_cells: u16,
	height_cells: Option<u16>,
	gate: u64,
) -> ImgState {
	let source_height = pixels.len();
	let source_width = pixels[0].len();
	let width = usize::from(width_cells.max(1));
	let height = usize::from(height_cells.unwrap_or_else(|| {
		let ratio = source_height as f32 / source_width as f32;
		((f32::from(width_cells) * ratio) / 2.0).round().max(1.0) as u16
	}));
	let mut cells = Vec::with_capacity(width * height);
	for cell_y in 0..height {
		let upper_y0 = cell_y * 2 * source_height / (height * 2);
		let upper_y1 = ((cell_y * 2 + 1) * source_height / (height * 2)).max(upper_y0 + 1);
		let lower_y0 = (cell_y * 2 + 1) * source_height / (height * 2);
		let lower_y1 = ((cell_y * 2 + 2) * source_height / (height * 2)).max(lower_y0 + 1);
		for cell_x in 0..width {
			let x0 = cell_x * source_width / width;
			let x1 = ((cell_x + 1) * source_width / width).max(x0 + 1);
			cells.push((
				average_pixels(pixels, x0, x1, upper_y0, upper_y1, gate),
				average_pixels(pixels, x0, x1, lower_y0, lower_y1, gate),
			));
		}
	}
	ImgState {
		cells: cells.into_boxed_slice(),
		width: width as u16,
		rows:  height as u16,
		phase: Load::Ready,
	}
}

/// Alpha-weighted mean of one half-cell's source block; `None` when the
/// block is mostly transparent, so logo padding never paints.
fn average_pixels(
	pixels: &[Vec<[u8; 4]>],
	x0: usize,
	x1: usize,
	y0: usize,
	y1: usize,
	gate: u64,
) -> Option<[u8; 3]> {
	let mut color = [0_u64; 3];
	let mut alpha = 0_u64;
	let mut count = 0_u64;
	for row in &pixels[y0.min(pixels.len() - 1)..y1.min(pixels.len())] {
		for pixel in &row[x0.min(row.len() - 1)..x1.min(row.len())] {
			let weight = u64::from(pixel[3]);
			color[0] += u64::from(pixel[0]) * weight;
			color[1] += u64::from(pixel[1]) * weight;
			color[2] += u64::from(pixel[2]) * weight;
			alpha += weight;
			count += 1;
		}
	}
	if count == 0 || alpha < count * gate {
		return None;
	}
	Some([(color[0] / alpha) as u8, (color[1] / alpha) as u8, (color[2] / alpha) as u8])
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		component::PaintCtx,
		frame::{CellContent, Frame, Size},
		test_support::frame_row_text,
	};

	#[test]
	fn invalid_inline_base64_source_paints_placeholder_without_panicking() {
		let mut image = Img::new()
			.with(Prop::Src, "data:image/png;base64,AAAA")
			.with(Prop::W, 12_u16);
		let ctx = UiContext::default();
		assert_eq!(image.height(&ctx, 12), 3);
		let mut frame = Frame::new(Size::new(20, 3));
		let mut hits = Vec::new();
		image.paint(
			&mut PaintCtx::new(&mut frame, &ctx, &mut hits, &mut Vec::new()),
			Rect::new(0, 0, 12, 3),
		);
		assert!(frame_row_text(&frame, 1).contains("[img:"));
	}

	#[test]
	fn indexed_palette_with_trns_expands_to_rgba() {
		let mut bytes = Vec::new();
		{
			let mut encoder = png::Encoder::new(&mut bytes, 2, 2);
			encoder.set_color(png::ColorType::Indexed);
			encoder.set_depth(png::BitDepth::Eight);
			encoder.set_palette(vec![255, 0, 0, 0, 0, 255]);
			encoder.set_trns(vec![255, 0]);
			let mut writer = encoder.write_header().unwrap();
			writer.write_image_data(&[0, 0, 1, 1]).unwrap();
		}
		let pixels = decode_png(&bytes).unwrap();
		assert_eq!(pixels[0][0], [255, 0, 0, 255], "palette index 0 is opaque red");
		assert_eq!(pixels[1][1], [0, 0, 255, 0], "palette index 1 is transparent blue");
	}

	#[test]
	fn transparent_pixels_sample_to_unpainted_halves() {
		// 4x2 source at two cells: one pixel block per half-cell.
		let red = [255_u8, 0, 0, 255];
		let clear = [0_u8, 0, 0, 0];
		let pixels = vec![vec![red, red, clear, clear], vec![clear, clear, clear, clear]];
		let state = sample_cells(&pixels, 2, None, PAINT_GATE);
		assert_eq!(state.rows, 1);
		assert_eq!(&*state.cells, &[(Some([255, 0, 0]), None), (None, None)]);
	}

	#[test]
	fn trim_recovers_padded_logos_at_thumbnail_sizes() {
		// A 2x2 opaque glyph centered in an 8x8 transparent canvas: at one
		// cell, every half-block averages under the alpha threshold.
		let blue = [0_u8, 0, 255, 255];
		let clear = [0_u8, 0, 0, 0];
		let mut pixels = vec![vec![clear; 8]; 8];
		for row in pixels.iter_mut().skip(3).take(2) {
			for pixel in row.iter_mut().skip(3).take(2) {
				*pixel = blue;
			}
		}
		let padded = sample_cells(&pixels, 1, None, PAINT_GATE);
		assert_eq!(&*padded.cells, &[(None, None)], "padding averages the glyph away");

		let trimmed = sample_cells(&trim_transparent(pixels), 1, None, TRIMMED_PAINT_GATE);
		assert_eq!(
			&*trimmed.cells,
			&[(Some([0, 0, 255]), Some([0, 0, 255]))],
			"trimming crops to the glyph before sampling"
		);

		let empty = vec![vec![clear; 4]; 4];
		assert_eq!(trim_transparent(empty).len(), 4, "fully transparent stays unchanged");
	}

	#[test]
	fn alpha_background_stays_unpainted_in_cells_mode() {
		let mut bytes = Vec::new();
		{
			let mut encoder = png::Encoder::new(&mut bytes, 4, 2);
			encoder.set_color(png::ColorType::Rgba);
			encoder.set_depth(png::BitDepth::Eight);
			let mut writer = encoder.write_header().unwrap();
			let mut data = vec![0_u8; 4 * 2 * 4];
			// Opaque red in the top-left pixel block; everything else clear.
			for x in 0..2 {
				data[x * 4] = 255;
				data[x * 4 + 3] = 255;
			}
			writer.write_image_data(&data).unwrap();
		}
		let path = std::env::temp_dir().join(format!("omp-tui-img-alpha-{}.png", std::process::id()));
		std::fs::write(&path, bytes).unwrap();
		let mut image = Img::new()
			.with(Prop::Src, path.to_string_lossy().as_ref())
			.with(Prop::W, 2_u16);
		let ctx = UiContext::default();
		assert_eq!(image.height(&ctx, 2), 1);
		std::fs::remove_file(path).unwrap();

		let mut frame = Frame::new(Size::new(3, 1));
		image.paint(
			&mut PaintCtx::new(&mut frame, &ctx, &mut Vec::new(), &mut Vec::new()),
			Rect::new(0, 0, 2, 1),
		);
		// Opaque top half paints a foreground-only half block …
		let painted = frame.cell(0, 0);
		assert_eq!(painted.style.foreground_color(), Color::Rgb(255, 0, 0));
		assert_eq!(painted.style.background_color(), Color::Default);
		// … and the fully transparent cell is never touched.
		assert_eq!(frame_row_text(&frame, 0), "▀");
	}

	#[test]
	fn jpeg_header_reserves_aspect_correct_placeholder() {
		let path = std::env::temp_dir().join(format!("omp-tui-img-jpeg-{}.jpg", std::process::id()));
		let jpeg = [0xff, 0xd8, 0xff, 0xc0, 0x00, 0x08, 8, 0x00, 80, 0x00, 160, 1];
		std::fs::write(&path, jpeg).unwrap();
		let mut image = Img::new()
			.with(Prop::Src, path.to_string_lossy().as_ref())
			.with(Prop::W, 20_u16);
		let ctx = UiContext::default();
		assert_eq!(image.height(&ctx, 20), 5);
		std::fs::remove_file(path).unwrap();

		let mut frame = Frame::new(Size::new(20, 5));
		image.paint(
			&mut PaintCtx::new(&mut frame, &ctx, &mut Vec::new(), &mut Vec::new()),
			Rect::new(0, 0, 20, 5),
		);
		assert!(frame_row_text(&frame, 2).contains("[img:"));
		assert_ne!(frame_row_text(&frame, 4).trim(), "");
	}

	#[test]
	fn kitty_mode_paints_typed_image_cells() {
		let mut image = Img::new().kitty(0x12_34_56, 2, 3);
		let ctx = UiContext { graphics: Graphics::KittyPlaceholders, ..UiContext::default() };
		assert_eq!(image.height(&ctx, 20), 2);
		let mut frame = Frame::new(Size::new(3, 2));
		image.paint(
			&mut PaintCtx::new(&mut frame, &ctx, &mut Vec::new(), &mut Vec::new()),
			Rect::new(0, 0, 3, 2),
		);
		assert!(matches!(frame.cell(2, 1).content, CellContent::Image {
			id:   0x12_34_56,
			row:  1,
			col:  2,
			rows: 2,
			cols: 3,
		}));
	}

	#[test]
	fn png_src_auto_interns_placeholder_cells_without_registration() {
		let logo = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets/login/anthropic.png");
		let mut image = Img::new()
			.with(Prop::Src, logo)
			.with(Prop::W, 2_u16)
			.with(Prop::H, 1_u16);
		let ctx = UiContext { graphics: Graphics::KittyPlaceholders, ..UiContext::default() };
		assert_eq!(image.measure(&ctx), (2, 1));
		let mut frame = Frame::new(Size::new(3, 1));
		image.paint(
			&mut PaintCtx::new(&mut frame, &ctx, &mut Vec::new(), &mut Vec::new()),
			Rect::new(0, 0, 2, 1),
		);
		let CellContent::Image { id, row: 0, col: 1, rows: 1, cols: 2 } = frame.cell(1, 0).content
		else {
			panic!("src image paints typed placeholder cells: {:?}", frame.cell(1, 0).content);
		};
		assert!(id > 0x00f0_0000, "registry IDs allocate from the top of the 24-bit range");

		// The same source in a second component shares the interned ID.
		let mut sibling = Img::new()
			.with(Prop::Src, logo)
			.with(Prop::W, 2_u16)
			.with(Prop::H, 1_u16);
		let mut second = Frame::new(Size::new(3, 1));
		sibling.paint(
			&mut PaintCtx::new(&mut second, &ctx, &mut Vec::new(), &mut Vec::new()),
			Rect::new(0, 0, 2, 1),
		);
		assert!(
			matches!(second.cell(0, 0).content, CellContent::Image { id: other, .. } if other == id)
		);

		// Cells tier ignores the interned box and keeps the half-block path.
		let cells_ctx = UiContext::default();
		let mut fallback = Img::new()
			.with(Prop::Src, logo)
			.with(Prop::W, 2_u16)
			.with(Prop::H, 1_u16);
		assert_eq!(fallback.height(&cells_ctx, 2), 1);
	}
}
