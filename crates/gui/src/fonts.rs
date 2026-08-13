//! Font discovery, cluster shaping, glyph rasterization, and the dual atlas.
//!
//! System faces are discovered through fontdb (SF Mono / JetBrains Mono /
//! Menlo chains), cell clusters are shaped with rustybuzz (ligatures, ZWJ
//! emoji, combining marks), rasterized by swash with hinting and subpixel-x
//! positioning, and packed into two etagere atlases: R8 coverage for outlined
//! glyphs, RGBA for color bitmap emoji. Fallback faces are resolved by
//! coverage — a face is accepted only when it shapes the whole cluster
//! without `.notdef` — never by maintained codepoint range tables.

use std::{collections::HashMap, sync::Arc};

use omp_core::Str;
use rustybuzz::Face as RustyFace;
use smallvec::SmallVec;
use swash::{FontRef, scale};
use thiserror::Error;

use crate::gpu::{ATLAS_SIZE, AtlasRegion};

/// Errors bringing up the font system.
#[derive(Debug, Error)]
pub enum FontError {
	/// No monospace face from the discovery chain exists on this system.
	#[error("no monospace font found on this system")]
	NoMonoFont,
}

/// Vertical metrics plus monospace advance for the primary face at a size.
#[derive(Clone, Copy, Debug)]
pub struct LineMetrics {
	/// Advance of one monospace cell, px.
	pub advance:     f32,
	/// Baseline to top of the em box, px.
	pub ascent:      f32,
	/// Baseline to bottom of the em box, px (positive).
	pub descent:     f32,
	/// Vertical advance between line tops, px.
	pub line_height: f32,
}

/// One rasterized glyph of a shaped cluster, quad-ready.
#[derive(Clone, Copy)]
pub struct CachedGlyph {
	/// Top-left texel in its atlas.
	pub uv:     [f32; 2],
	/// Quad size, px (bitmap emoji are normalized to the em box).
	pub size:   [f32; 2],
	/// Quad top-left relative to (origin, baseline): x right, y up.
	pub offset: [f32; 2],
	/// Whether the glyph lives in the RGBA (color bitmap) atlas.
	pub color:  bool,
}

/// A shaped, rasterized cell cluster: glyphs positioned relative to a
/// snapped pen origin the caller computes per placement (the subpixel
/// fraction is baked into the rasters, so cache entries are absolute-x
/// independent).
#[derive(Clone)]
pub struct CachedCluster {
	/// Shaped glyphs in draw order.
	pub glyphs:  SmallVec<CachedGlyph, 4>,
	/// Total shaped advance, px.
	pub advance: f32,
}

/// Subpixel bin for horizontal glyph positioning (cosmic-text / egui
/// scheme): each cluster caches up to four rasters at fractional x offsets,
/// so inter-cell spacing stays even without per-position rasterization.
/// Vertical stays integer-snapped.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SubpixelBin {
	Zero,
	One,
	Two,
	Three,
}

impl SubpixelBin {
	/// Bins a fractional position; returns the integral coordinate and bin.
	fn new(pos: f32) -> (i32, Self) {
		let trunc = pos as i32;
		let fract = pos - trunc as f32;
		if fract < 0.125 {
			(trunc, Self::Zero)
		} else if fract < 0.375 {
			(trunc, Self::One)
		} else if fract < 0.625 {
			(trunc, Self::Two)
		} else if fract < 0.875 {
			(trunc, Self::Three)
		} else {
			(trunc + 1, Self::Zero)
		}
	}

	fn offset(self) -> f32 {
		match self {
			Self::Zero => 0.0,
			Self::One => 0.25,
			Self::Two => 0.5,
			Self::Three => 0.75,
		}
	}
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct ClusterKey {
	text:   Str,
	px8:    u32,
	bold:   bool,
	italic: bool,
	bin:    SubpixelBin,
}

/// Coverage -> alpha transfer for light-on-dark text
/// (`alpha = 2c - c²`, epaint's `TwoCoverageMinusCoverageSq` dark-mode
/// default: fuller antialiased edges than linear, sharper than gamma 0.5).
fn coverage_lut() -> [u8; 256] {
	let mut lut = [0_u8; 256];
	let mut i = 0;
	while i < 256 {
		let c = i as f32 / 255.0;
		lut[i] = ((2.0 * c - c * c) * 255.0 + 0.5) as u8;
		i += 1;
	}
	lut
}

struct FaceData {
	data:  Arc<Vec<u8>>,
	index: u32,
}

impl FaceData {
	fn font(&self) -> Option<FontRef<'_>> {
		FontRef::from_index(&self.data, self.index as usize)
	}
}

self_cell::self_cell!(
	/// A rustybuzz face owning its font bytes.
	struct BuzzFace {
		owner:     Arc<Vec<u8>>,
		#[covariant]
		dependent: RustyFace,
	}
);

const MONO_CHAIN: &[&str] = &[
	"SF Mono",
	"JetBrains Mono",
	"JetBrainsMono Nerd Font",
	"Cascadia Code",
	"Fira Code",
	"Menlo",
	"Consolas",
	"DejaVu Sans Mono",
];

/// Fallback families probed first for uncovered base scalars (CJK, symbols).
const FALLBACK_CHAIN: &[&str] = &[
	"Hiragino Sans",
	"PingFang SC",
	"Apple SD Gothic Neo",
	"Apple Symbols",
	"Noto Sans",
	"Arial Unicode MS",
	"Zapf Dingbats",
];

/// Color bitmap emoji faces, leading candidates for VS16 clusters.
const EMOJI_CHAIN: &[&str] = &["Apple Color Emoji", "Noto Color Emoji", "Segoe UI Emoji"];

/// Nerd Font families; their presence upgrades the icon charset.
const NERD_CHAIN: &[&str] =
	&["Symbols Nerd Font Mono", "JetBrainsMono Nerd Font", "Nerd Fonts Symbols Only"];

/// Rasterization sources, tried in order: COLR layers, embedded bitmaps,
/// then plain outlines.
const SOURCES: &[scale::Source] = &[
	scale::Source::ColorOutline(0),
	scale::Source::ColorBitmap(scale::StrikeWith::BestFit),
	scale::Source::Outline,
];

/// The font system. One instance per window; not thread-safe by design
/// (all rasterization happens on the paint path).
pub struct Fonts {
	db:                  fontdb::Database,
	faces:               Vec<FaceData>,
	by_db_id:            HashMap<fontdb::ID, u16>,
	primary:             u16,
	primary_bold:        u16,
	emoji:               Option<u16>,
	nerd:                bool,
	primary_italic:      Option<u16>,
	primary_bold_italic: Option<u16>,
	/// Nerd-family faces, the only sanctioned Private-Use-Area fallbacks.
	nerd_faces:          SmallVec<u16, 2>,
	buzz:                HashMap<u16, BuzzFace>,
	fallback:            HashMap<char, Option<u16>>,
	cx:                  scale::ScaleContext,
	clusters:            HashMap<ClusterKey, CachedCluster>,
	metrics:             HashMap<u32, LineMetrics>,
	images:              HashMap<(u32, u32, u32), Option<([f32; 2], [f32; 2])>>,
	mask_alloc:          etagere::AtlasAllocator,
	color_alloc:         etagere::AtlasAllocator,
	pending_mask:        Vec<AtlasRegion>,
	pending_color:       Vec<AtlasRegion>,
	atlas_exhausted:     bool,
	/// Resolved primary family name, for status display.
	family:              String,
}

impl Fonts {
	/// Discovers system fonts and builds the (empty) atlases. Fails only
	/// when no monospace face exists at all.
	pub fn new() -> Result<Fonts, FontError> {
		let mut db = fontdb::Database::new();
		db.load_system_fonts();
		let mut fonts = Fonts {
			db,
			faces: Vec::new(),
			by_db_id: HashMap::new(),
			primary: 0,
			primary_bold: 0,
			emoji: None,
			nerd: false,
			primary_italic: None,
			primary_bold_italic: None,
			nerd_faces: SmallVec::new(),
			buzz: HashMap::new(),
			fallback: HashMap::new(),
			cx: scale::ScaleContext::new(),
			clusters: HashMap::new(),
			metrics: HashMap::new(),
			images: HashMap::new(),
			mask_alloc: etagere::AtlasAllocator::new(etagere::size2(
				ATLAS_SIZE as i32,
				ATLAS_SIZE as i32,
			)),
			color_alloc: etagere::AtlasAllocator::new(etagere::size2(
				ATLAS_SIZE as i32,
				ATLAS_SIZE as i32,
			)),
			pending_mask: Vec::new(),
			pending_color: Vec::new(),
			atlas_exhausted: false,
			family: String::new(),
		};
		let (primary, family) = fonts
			.resolve_chain(MONO_CHAIN, fontdb::Family::Monospace, fontdb::Weight::NORMAL)
			.ok_or(FontError::NoMonoFont)?;
		fonts.primary_bold = fonts
			.resolve_family(&family, fontdb::Weight::BOLD, fontdb::Style::Normal)
			.unwrap_or(primary);
		fonts.primary_italic =
			fonts.resolve_family(&family, fontdb::Weight::NORMAL, fontdb::Style::Italic);
		fonts.primary_bold_italic =
			fonts.resolve_family(&family, fontdb::Weight::BOLD, fontdb::Style::Italic);
		fonts.primary = primary;
		fonts.emoji = EMOJI_CHAIN.iter().find_map(|name| {
			fonts.resolve_family(name, fontdb::Weight::NORMAL, fontdb::Style::Normal)
		});
		fonts.nerd_faces = NERD_CHAIN
			.iter()
			.filter_map(|name| {
				fonts.resolve_family(name, fontdb::Weight::NORMAL, fontdb::Style::Normal)
			})
			.collect();
		fonts.nerd = !fonts.nerd_faces.is_empty()
			|| fonts.faces[primary as usize]
				.font()
				.is_some_and(|font| font.charmap().map('\u{e0b0}') != 0);
		fonts.family = family;
		Ok(fonts)
	}

	/// Whether a Nerd Font face is available (primary or fallback), gating
	/// the `Charset::NerdFont` icon tier.
	pub fn has_nerd_font(&self) -> bool {
		self.nerd
	}

	/// Whether the primary family ships a real italic face.
	pub fn has_italic(&self) -> bool {
		self.primary_italic.is_some()
	}

	/// The resolved primary family name.
	pub fn family(&self) -> &str {
		&self.family
	}

	/// Vertical metrics and the monospace cell advance at `px` (cached).
	pub fn cell_metrics(&mut self, px: f32) -> LineMetrics {
		let key = Self::quant(px);
		if let Some(metrics) = self.metrics.get(&key) {
			return *metrics;
		}
		let metrics = self.faces[self.primary as usize]
			.font()
			.map(|font| {
				let sm = font.metrics(&[]).scale(px);
				let advance = font
					.charmap()
					.map('0')
					.checked_sub(0)
					.map(|gid| font.glyph_metrics(&[]).scale(px).advance_width(gid as u16))
					.filter(|advance| *advance > 0.0)
					.unwrap_or(px * 0.6);
				LineMetrics {
					advance,
					ascent: sm.ascent,
					descent: sm.descent,
					// Terminal cell height: the bare em span. Mono fonts draw
					// box-drawing glyphs to fill exactly this, so vertical
					// seams connect; editor-style leading would break them.
					line_height: (sm.ascent + sm.descent).ceil(),
				}
			})
			.unwrap_or(LineMetrics {
				advance:     px * 0.6,
				ascent:      px * 0.8,
				descent:     px * 0.2,
				line_height: (px * 1.2).ceil(),
			});
		self.metrics.insert(key, metrics);
		metrics
	}

	fn quant(px: f32) -> u32 {
		(px * 8.0).round().max(1.0) as u32
	}

	/// Drains dirty atlas regions for the GPU upload.
	pub fn take_uploads(&mut self) -> (Vec<AtlasRegion>, Vec<AtlasRegion>) {
		(std::mem::take(&mut self.pending_mask), std::mem::take(&mut self.pending_color))
	}

	/// Caches one registered image resized into the RGBA atlas.
	pub fn image_region(
		&mut self,
		id: u32,
		width: u32,
		height: u32,
	) -> Option<([f32; 2], [f32; 2])> {
		let key = (id, width, height);
		if let Some(region) = self.images.get(&key) {
			return *region;
		}
		let region = self.build_image_region(id, width, height);
		self.images.insert(key, region);
		region
	}

	fn build_image_region(
		&mut self,
		id: u32,
		width: u32,
		height: u32,
	) -> Option<([f32; 2], [f32; 2])> {
		if width == 0 || height == 0 {
			return None;
		}
		let bytes = omp_tui::image_bytes(id)?;
		let (source, source_width, source_height) = decode_png_rgba(bytes.as_ref())?;
		let pixels = resize_bilinear(&source, source_width, source_height, width, height)?;
		let allocation = self
			.color_alloc
			.allocate(etagere::size2(width as i32 + 2, height as i32 + 2))?;
		let point = allocation.rectangle.min;
		let (uvx, uvy) = (point.x as u32 + 1, point.y as u32 + 1);
		let row = width as usize * 4;
		let padded_row = (width as usize + 2) * 4;
		let mut data = vec![0_u8; padded_row * (height as usize + 2)];
		for y in 0..height as usize {
			let dst = (y + 1) * padded_row + 4;
			let src = y * row;
			data[dst..dst + row].copy_from_slice(&pixels[src..src + row]);
		}
		self.pending_color.push(AtlasRegion {
			x: uvx - 1,
			y: uvy - 1,
			width: width + 2,
			height: height + 2,
			data,
		});
		Some(([uvx as f32, uvy as f32], [width as f32, height as f32]))
	}

	/// Drops every raster cache after a scale-factor or size change; the
	/// painter's atlas textures must be recreated alongside.
	pub fn clear_caches(&mut self) {
		self.clusters.clear();
		self.images.clear();
		self.metrics.clear();
		self.fallback.clear();
		self.mask_alloc =
			etagere::AtlasAllocator::new(etagere::size2(ATLAS_SIZE as i32, ATLAS_SIZE as i32));
		self.color_alloc =
			etagere::AtlasAllocator::new(etagere::size2(ATLAS_SIZE as i32, ATLAS_SIZE as i32));
		self.pending_mask.clear();
		self.pending_color.clear();
		self.atlas_exhausted = false;
	}

	/// Shapes and rasterizes one cell cluster, returning the cached quads
	/// plus the snapped pen origin for *this* `pen_x`. `box_width` (the
	/// cell span in px) centers glyphs narrower than their box.
	pub fn cluster(
		&mut self,
		text: &str,
		bold: bool,
		italic: bool,
		px: f32,
		pen_x: f32,
		box_width: f32,
	) -> (&CachedCluster, i32) {
		let (origin, bin) = SubpixelBin::new(pen_x);
		let key = ClusterKey { text: Str::new(text), px8: Self::quant(px), bold, italic, bin };
		if !self.clusters.contains_key(&key) {
			let cluster = self.build_cluster(text, bold, italic, px, bin, box_width);
			self.clusters.insert(key.clone(), cluster);
		}
		(&self.clusters[&key], origin)
	}

	fn build_cluster(
		&mut self,
		text: &str,
		bold: bool,
		italic: bool,
		px: f32,
		bin: SubpixelBin,
		box_width: f32,
	) -> CachedCluster {
		let empty = CachedCluster { glyphs: SmallVec::new(), advance: 0.0 };
		let Some(first) = text.chars().next() else {
			return empty;
		};
		let face = self.face_for_cluster(text, first, bold, italic);
		let Some(face) = face else {
			return empty;
		};
		let shaped = self.shape(face, text, px);
		if shaped.is_empty() {
			return empty;
		}

		let lut = coverage_lut();
		let mut glyphs: SmallVec<CachedGlyph, 4> = SmallVec::new();
		let mut pen = 0.0_f32;
		for (gid, advance) in &shaped {
			if *gid == 0 {
				pen += advance;
				continue;
			}
			if let Some(glyph) = self.rasterize(face, *gid, px, bin, pen, &lut) {
				glyphs.push(glyph);
			}
			pen += advance;
		}
		if glyphs.is_empty() {
			return empty;
		}
		// Center a cluster narrower than its cell box (emoji in two cells).
		let advance = pen.max(0.0);
		let slack = box_width - advance;
		if slack > 0.5 {
			let shift = slack * 0.5;
			for glyph in &mut glyphs {
				glyph.offset[0] += shift;
			}
		}
		CachedCluster { glyphs, advance }
	}

	/// Picks the face shaping `text`. Style resolution prefers the real
	/// bold/italic cuts of the primary family; Private-Use-Area scalars
	/// (nerd icons) resolve only through nerd families — a random system
	/// font covering a PUA codepoint is never the intended glyph. The
	/// color emoji face leads only for explicit VS16 (U+FE0F) clusters —
	/// everywhere else it is just another coverage candidate, so symbols
	/// keep text presentation unless nothing else covers them.
	fn face_for_cluster(
		&mut self,
		text: &str,
		first: char,
		bold: bool,
		italic: bool,
	) -> Option<u16> {
		let primary = match (bold, italic) {
			(true, true) => self.primary_bold_italic.unwrap_or(self.primary_bold),
			(false, true) => self.primary_italic.unwrap_or(self.primary),
			(true, false) => self.primary_bold,
			(false, false) => self.primary,
		};
		if is_pua(first) {
			let mut candidates: SmallVec<u16, 4> = SmallVec::new();
			candidates.push(primary);
			candidates.extend(self.nerd_faces.iter().copied());
			return candidates
				.into_iter()
				.find(|face| self.shapes_cleanly(*face, text));
		}
		let prefer_emoji = self.emoji.is_some() && text.contains('\u{fe0f}');
		let mut candidates: SmallVec<u16, 4> = SmallVec::new();
		if prefer_emoji && let Some(emoji) = self.emoji {
			candidates.push(emoji);
		}
		candidates.push(primary);
		if let Some(fallback) = self.resolve_fallback(first) {
			candidates.push(fallback);
		}
		if !prefer_emoji && let Some(emoji) = self.emoji {
			candidates.push(emoji);
		}
		candidates
			.into_iter()
			.find(|face| self.shapes_cleanly(*face, text))
	}

	/// Whether `face` covers the first scalar and shapes the whole cluster.
	fn shapes_cleanly(&mut self, face: u16, text: &str) -> bool {
		let Some(font) = self.faces[face as usize].font() else {
			return false;
		};
		let mut chars = text.chars();
		let Some(first) = chars.next() else {
			return false;
		};
		if font.charmap().map(first) == 0 {
			return false;
		}
		if chars.next().is_none() {
			return true;
		}
		let shaped = self.shape(face, text, 16.0);
		!shaped.is_empty() && shaped.iter().all(|(gid, _)| *gid != 0)
	}

	/// Shapes the cluster, returning `(glyph id, advance px)` pairs.
	fn shape(&mut self, face: u16, text: &str, px: f32) -> SmallVec<(u16, f32), 4> {
		let mut chars = text.chars();
		if let (Some(c), None) = (chars.next(), chars.next()) {
			let gid = self.faces[face as usize]
				.font()
				.map(|f| f.charmap().map(c))
				.unwrap_or(0);
			if gid == 0 {
				return SmallVec::new();
			}
			let advance = self.faces[face as usize]
				.font()
				.map(|f| f.glyph_metrics(&[]).scale(px).advance_width(gid))
				.unwrap_or(px * 0.5);
			let mut shaped = SmallVec::new();
			shaped.push((gid, advance));
			return shaped;
		}
		let data = Arc::clone(&self.faces[face as usize].data);
		let index = self.faces[face as usize].index;
		let buzz = self.buzz.entry(face).or_insert_with(|| {
			BuzzFace::new(data, |bytes| {
				rustybuzz::Face::from_slice(bytes, index).expect("face validated at load")
			})
		});
		let mut buffer = rustybuzz::UnicodeBuffer::new();
		buffer.push_str(text);
		let buffer = rustybuzz::shape(buzz.borrow_dependent(), &[], buffer);
		let upem = buzz.borrow_dependent().units_per_em().max(1) as f32;
		let scale = px / upem;
		rustybuzz::GlyphBuffer::glyph_infos(&buffer)
			.iter()
			.zip(rustybuzz::GlyphBuffer::glyph_positions(&buffer))
			.map(|(info, pos)| (info.glyph_id as u16, pos.x_advance as f32 * scale))
			.collect()
	}

	/// Resolves the fallback face covering `ch`, cached per scalar.
	fn resolve_fallback(&mut self, ch: char) -> Option<u16> {
		if let Some(cached) = self.fallback.get(&ch) {
			return *cached;
		}
		let found = self.scan_fallback(ch);
		self.fallback.insert(ch, found);
		found
	}

	fn scan_fallback(&mut self, ch: char) -> Option<u16> {
		for name in FALLBACK_CHAIN {
			if let Some(face) =
				self.resolve_family(name, fontdb::Weight::NORMAL, fontdb::Style::Normal)
				&& let Some(font) = self.faces[face as usize].font()
				&& font.charmap().map(ch) != 0
			{
				return Some(face);
			}
		}
		// Full scan as last resort; cached per scalar so this runs once.
		let ids: Vec<fontdb::ID> = self.db.faces().map(|face| face.id).collect();
		for id in ids {
			let covers = self
				.db
				.with_face_data(id, |data, index| {
					FontRef::from_index(data, index as usize)
						.map(|font| font.charmap().map(ch) != 0)
						.unwrap_or(false)
				})
				.unwrap_or(false);
			if covers && let Some(face) = self.load_face(id) {
				return Some(face);
			}
		}
		None
	}

	/// Rasterizes one glyph into its atlas; `pen` is the shaped pen offset
	/// within the cluster. Returns quad placement relative to (origin,
	/// baseline).
	fn rasterize(
		&mut self,
		face: u16,
		gid: u16,
		px: f32,
		bin: SubpixelBin,
		pen: f32,
		lut: &[u8; 256],
	) -> Option<CachedGlyph> {
		let font = self.faces[face as usize].font()?;
		let mut scaler = self.cx.builder(font).size(px).hint(true).build();
		let mut render = scale::Render::new(SOURCES);
		render
			.format(swash::zeno::Format::Alpha)
			.offset(swash::zeno::Point::new(bin.offset(), 0.0));
		let image = render.render(&mut scaler, gid)?;
		let (width, height) = (image.placement.width, image.placement.height);
		if width == 0 || height == 0 {
			return None;
		}
		let color = image.content != scale::image::Content::Mask;
		// Bitmap strikes come at their fixed size (Apple ships 160px);
		// normalize the quad to the em box, scaling bearings alongside.
		let (quad_w, quad_h, left, top) =
			if matches!(image.source, scale::Source::ColorBitmap(_) | scale::Source::Bitmap(_)) {
				let factor = px / height.max(1) as f32;
				(
					width as f32 * factor,
					height as f32 * factor,
					image.placement.left as f32 * factor,
					image.placement.top as f32 * factor,
				)
			} else {
				(width as f32, height as f32, image.placement.left as f32, image.placement.top as f32)
			};
		let (alloc, pending) = if color {
			(&mut self.color_alloc, &mut self.pending_color)
		} else {
			(&mut self.mask_alloc, &mut self.pending_mask)
		};
		// A one-texel transparent gutter around every glyph: linear
		// sampling at quad edges must never read a neighboring glyph's
		// texels out of the shared atlas.
		let allocation = alloc
			.allocate(etagere::size2(width as i32 + 2, height as i32 + 2))
			.or_else(|| {
				if !self.atlas_exhausted {
					self.atlas_exhausted = true;
					eprintln!("omp-gui: glyph atlas exhausted; new glyphs will be dropped");
				}
				None
			})?;
		let point = allocation.rectangle.min;
		let (uvx, uvy) = (point.x as u32 + 1, point.y as u32 + 1);
		let bpp = if color { 4_usize } else { 1 };
		let row = width as usize * bpp;
		let padded_row = (width as usize + 2) * bpp;
		let mut data = vec![0_u8; padded_row * (height as usize + 2)];
		for y in 0..height as usize {
			let dst = (y + 1) * padded_row + bpp;
			let src = y * row;
			data[dst..dst + row].copy_from_slice(&image.data[src..src + row]);
			if !color {
				for value in &mut data[dst..dst + row] {
					*value = lut[*value as usize];
				}
			}
		}
		pending.push(AtlasRegion {
			x: uvx - 1,
			y: uvy - 1,
			width: width + 2,
			height: height + 2,
			data,
		});
		Some(CachedGlyph {
			uv: [uvx as f32, uvy as f32],
			size: [quad_w, quad_h],
			offset: [pen + left, top],
			color,
		})
	}

	fn resolve_chain(
		&mut self,
		chain: &[&str],
		generic: fontdb::Family,
		weight: fontdb::Weight,
	) -> Option<(u16, String)> {
		for name in chain {
			if let Some(face) = self.resolve_family(name, weight, fontdb::Style::Normal) {
				return Some((face, (*name).to_string()));
			}
		}
		let query = fontdb::Query { families: &[generic], weight, ..Default::default() };
		let id = self.db.query(&query)?;
		let name = self
			.db
			.face(id)
			.and_then(|face| face.families.first().map(|(name, _)| name.clone()))
			.unwrap_or_default();
		self.load_face(id).map(|face| (face, name))
	}

	/// Resolves one family at an exact weight/style; fontdb's fuzzy query
	/// is post-checked so a missing italic cut reports `None` instead of
	/// silently returning the regular face.
	fn resolve_family(
		&mut self,
		name: &str,
		weight: fontdb::Weight,
		style: fontdb::Style,
	) -> Option<u16> {
		let query = fontdb::Query {
			families: &[fontdb::Family::Name(name)],
			weight,
			style,
			..Default::default()
		};
		let id = self.db.query(&query)?;
		let info = self.db.face(id)?;
		if info.style != style || info.weight != weight {
			return None;
		}
		self.load_face(id)
	}

	fn load_face(&mut self, id: fontdb::ID) -> Option<u16> {
		if let Some(face) = self.by_db_id.get(&id) {
			return Some(*face);
		}
		let (data, index) = self
			.db
			.with_face_data(id, |data, index| (Arc::new(data.to_vec()), index))?;
		let face = FaceData { data, index };
		face.font()?;
		let idx = u16::try_from(self.faces.len()).ok()?;
		self.faces.push(face);
		self.by_db_id.insert(id, idx);
		Some(idx)
	}
}

fn decode_png_rgba(bytes: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
	let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
	decoder.set_transformations(png::Transformations::EXPAND);
	let mut reader = decoder.read_info().ok()?;
	let mut buffer = vec![0_u8; reader.output_buffer_size()?];
	let info = reader.next_frame(&mut buffer).ok()?;
	if info.bit_depth != png::BitDepth::Eight {
		return None;
	}
	let source = &buffer[..info.buffer_size()];
	let pixels = usize::try_from(info.width.checked_mul(info.height)?).ok()?;
	let mut rgba = Vec::with_capacity(pixels.checked_mul(4)?);
	for pixel in source.chunks_exact(info.color_type.samples()) {
		let value = match info.color_type {
			png::ColorType::Grayscale => [pixel[0], pixel[0], pixel[0], 255],
			png::ColorType::GrayscaleAlpha => [pixel[0], pixel[0], pixel[0], pixel[1]],
			png::ColorType::Rgb => [pixel[0], pixel[1], pixel[2], 255],
			png::ColorType::Rgba => [pixel[0], pixel[1], pixel[2], pixel[3]],
			png::ColorType::Indexed => return None,
		};
		rgba.extend_from_slice(&value);
	}
	(rgba.len() == pixels * 4).then_some((rgba, info.width, info.height))
}

fn resize_bilinear(
	source: &[u8],
	source_width: u32,
	source_height: u32,
	width: u32,
	height: u32,
) -> Option<Vec<u8>> {
	let len = usize::try_from(width.checked_mul(height)?.checked_mul(4)?).ok()?;
	let mut out = vec![0_u8; len];
	for y in 0..height {
		let sy = ((y as f32 + 0.5) * source_height as f32 / height as f32 - 0.5)
			.clamp(0.0, source_height.saturating_sub(1) as f32);
		let (y0, y1, fy) = (sy.floor() as u32, sy.ceil() as u32, sy.fract());
		for x in 0..width {
			let sx = ((x as f32 + 0.5) * source_width as f32 / width as f32 - 0.5)
				.clamp(0.0, source_width.saturating_sub(1) as f32);
			let (x0, x1, fx) = (sx.floor() as u32, sx.ceil() as u32, sx.fract());
			for channel in 0..4 {
				let at = |px, py| ((py * source_width + px) * 4 + channel) as usize;
				let top = f32::from(*source.get(at(x0, y0))?)
					.mul_add(1.0 - fx, f32::from(*source.get(at(x1, y0))?) * fx);
				let bottom = f32::from(*source.get(at(x0, y1))?)
					.mul_add(1.0 - fx, f32::from(*source.get(at(x1, y1))?) * fx);
				out[((y * width + x) * 4 + channel) as usize] =
					top.mul_add(1.0 - fy, bottom * fy).round() as u8;
			}
		}
	}
	Some(out)
}

/// Private Use Areas (BMP block plus planes 15–16): reserved for
/// font-specific icons, so only nerd families may serve them.
const fn is_pua(c: char) -> bool {
	matches!(c as u32, 0xe000..=0xf8ff | 0xf0000..=0xffffd | 0x100000..=0x10fffd)
}
