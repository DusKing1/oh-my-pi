//! CPU fragment-shader toolkit: a GPU-style program trait and a rasterizer
//! that packs per-pixel shading into half-block cells.
//!
//! The mental model is a fullscreen GPU pipeline at terminal resolution:
//! implement [`Program`] — a per-frame [`advance`](Program::advance), a
//! per-pixel [`fragment`](Program::fragment), and an optional point-sprite
//! [`particles`](Program::particles) pass — or use a plain
//! `Fn(f32, f32) -> (Vec3, f32)` closure for a still field. Then either
//! mount it in a retained tree with [`crate::components::Shader`] or paint
//! it into any cell grid with [`Surface::render`].
//!
//! One terminal cell is a 1×2 pixel column (`▀` foreground over
//! background), so a `cols × rows` viewport is a `cols × rows·2` pixel
//! target with square pixels on a typical 1:2 cell. Everything is a pure
//! function of the caller's clock, so animated effects stay deterministic
//! and testable.
//!
//! # Example
//! ```
//! use omp_tui::{
//! 	scene::{Vec3, vec3},
//! 	shader::Surface,
//! };
//!
//! // A vertical dusk gradient: any `Fn(x, y) -> (color, coverage)` closure.
//! let mut dusk = |_: f32, y: f32| (vec3(0.9, 0.4, 0.2).lerp(vec3(0.05, 0.05, 0.2), y / 16.0), 1.0);
//! let mut lit = 0;
//! Surface::new().render(&mut dusk, std::time::Duration::ZERO, 20, 8, |_, _, _, _, _| lit += 1);
//! assert_eq!(lit, 20 * 8);
//! ```

mod eclipse;

use std::time::Duration;

pub use eclipse::Eclipse;

use crate::{frame::Color, scene::Vec3};

/// Coverage at which a pixel lights; both halves dark leaves the cell
/// untouched, so whatever sits behind it shows through.
const LIT_THRESHOLD: f32 = 0.04;

/// A fullscreen effect: per-frame state plus a shader for every pixel.
///
/// [`fragment`](Self::fragment) returns a unit-range color and a coverage
/// alpha. Coverage decides whether a pixel lights; color, scaled by
/// coverage, is composited over black. [`particles`](Self::particles)
/// splats point sprites over the shaded field — the CPU stand-in for an
/// instanced sprite pass. Any `Fn(f32, f32) -> (Vec3, f32)` closure is a
/// still, particle-free program.
pub trait Program {
	/// Advances animation state to `now` for a `width` × `height` pixel
	/// target. Runs once per frame before any sampling.
	fn advance(&mut self, now: Duration, width: f32, height: f32) {
		let _ = (now, width, height);
	}

	/// Shades the pixel centered at `(x, y)`: `(color, coverage)`, both in
	/// unit range.
	fn fragment(&self, x: f32, y: f32) -> (Vec3, f32);

	/// Emits point sprites as `emit(x, y, color, alpha)`; each blends over
	/// the shaded field at its nearest pixel. Off-target sprites are
	/// ignored.
	fn particles(&self, emit: &mut dyn FnMut(f32, f32, Vec3, f32)) {
		let _ = emit;
	}
}

impl<F: Fn(f32, f32) -> (Vec3, f32)> Program for F {
	fn fragment(&self, x: f32, y: f32) -> (Vec3, f32) {
		self(x, y)
	}
}

/// Avalanching integer hash (lowbias32) — the seed mixer stipple and dust
/// shaders build on; identical to the WGSL `hash_u32` used on GPU ports.
pub const fn hash(value: u32) -> u32 {
	let mut mixed = value;
	mixed = (mixed ^ (mixed >> 16)).wrapping_mul(0x7feb_352d);
	mixed = (mixed ^ (mixed >> 15)).wrapping_mul(0x846c_a68b);
	mixed ^ (mixed >> 16)
}

/// A uniform `[0, 1)` sample from a hash seed — stable across frames, so
/// hash-derived geometry never flickers.
pub fn rand01(value: u32) -> f32 {
	(hash(value) >> 8) as f32 / 16_777_216.0
}

/// A reusable half-block render target.
///
/// Owns the pixel buffer between frames, so rendering allocates nothing
/// once warm. One instance per viewport; sizes are per-call.
#[derive(Default)]
pub struct Surface {
	/// Premultiplied linear color and accumulated coverage per pixel,
	/// row-major at `cols × rows·2`. A `Vec` is the right shape here: one
	/// large frame-sized buffer, reused across frames.
	pixels: Vec<(Vec3, f32)>,
}

impl Surface {
	/// Creates an empty target; the buffer grows on first render.
	pub const fn new() -> Self {
		Self { pixels: Vec::new() }
	}

	/// Renders one frame of `program` into a `cols` × `rows` grid of
	/// half-block cells.
	///
	/// Advances the program to `now`, shades every pixel, splats particles,
	/// then emits lit cells as `put(column, row, glyph, fg, bg)`: `▀` with
	/// both colors when both halves lit, `▀`/`▄` with `bg = None` when only
	/// one half is, nothing when neither.
	pub fn render<P: Program + ?Sized>(
		&mut self,
		program: &mut P,
		now: Duration,
		cols: u16,
		rows: u16,
		mut put: impl FnMut(u16, u16, char, Color, Option<Color>),
	) {
		if cols == 0 || rows == 0 {
			return;
		}
		let width = cols as usize;
		let height = rows as usize * 2;
		program.advance(now, width as f32, height as f32);

		self.pixels.clear();
		self.pixels.reserve(width * height);
		for py in 0..height {
			for px in 0..width {
				let (color, alpha) = program.fragment(px as f32 + 0.5, py as f32 + 0.5);
				let alpha = alpha.clamp(0.0, 1.0);
				self.pixels.push((color * alpha, alpha));
			}
		}
		program.particles(&mut |x: f32, y: f32, color: Vec3, alpha: f32| {
			if x < 0.0 || y < 0.0 || x >= width as f32 || y >= height as f32 {
				return;
			}
			let alpha = alpha.clamp(0.0, 1.0);
			let pixel = &mut self.pixels[y as usize * width + x as usize];
			pixel.0 = color * alpha + pixel.0 * (1.0 - alpha);
			pixel.1 = pixel.1.mul_add(1.0 - alpha, alpha);
		});

		for row in 0..rows as usize {
			for col in 0..width {
				let (top, top_cover) = self.pixels[row * 2 * width + col];
				let (bottom, bottom_cover) = self.pixels[(row * 2 + 1) * width + col];
				let (glyph, fg, bg) = match (top_cover >= LIT_THRESHOLD, bottom_cover >= LIT_THRESHOLD)
				{
					(true, true) => ('▀', top, Some(bottom)),
					(true, false) => ('▀', top, None),
					(false, true) => ('▄', bottom, None),
					(false, false) => continue,
				};
				put(col as u16, row as u16, glyph, Color::from(fg), bg.map(Color::from));
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::scene::vec3;

	fn cells<P: Program>(
		cols: u16,
		rows: u16,
		program: &mut P,
	) -> Vec<(u16, u16, char, Color, Option<Color>)> {
		let mut out = Vec::new();
		Surface::new().render(program, Duration::ZERO, cols, rows, |x, y, glyph, fg, bg| {
			out.push((x, y, glyph, fg, bg));
		});
		out
	}

	#[test]
	fn opaque_field_packs_pixel_pairs_into_upper_half_blocks() {
		// Top pixel row red, bottom row blue.
		let mut split = |_: f32, y: f32| {
			if y < 1.0 {
				(vec3(1.0, 0.0, 0.0), 1.0)
			} else {
				(vec3(0.0, 0.0, 1.0), 1.0)
			}
		};
		let cells = cells(4, 1, &mut split);
		assert_eq!(cells.len(), 4);
		assert!(cells.iter().all(|&(.., glyph, fg, bg)| {
			glyph == '▀' && fg == Color::Rgb(255, 0, 0) && bg == Some(Color::Rgb(0, 0, 255))
		}));
	}

	#[test]
	fn coverage_below_the_lit_threshold_stays_transparent() {
		let mut haze = |_: f32, _: f32| (vec3(1.0, 1.0, 1.0), 0.02);
		assert_eq!(cells(8, 4, &mut haze).len(), 0, "0.02 sits under the lit threshold");
	}

	#[test]
	fn a_half_lit_cell_picks_the_matching_half_block() {
		let mut top = |_: f32, y: f32| (vec3(1.0, 1.0, 1.0), if y < 1.0 { 1.0 } else { 0.0 });
		assert_eq!(cells(1, 1, &mut top), vec![(0, 0, '▀', Color::Rgb(255, 255, 255), None)]);
		let mut bottom = |_: f32, y: f32| (vec3(1.0, 1.0, 1.0), if y < 1.0 { 0.0 } else { 1.0 });
		assert_eq!(cells(1, 1, &mut bottom), vec![(0, 0, '▄', Color::Rgb(255, 255, 255), None)]);
	}

	#[test]
	fn particles_blend_over_the_field_and_ignore_off_target_splats() {
		struct Dust;
		impl Program for Dust {
			fn fragment(&self, _: f32, _: f32) -> (Vec3, f32) {
				(Vec3::ZERO, 1.0)
			}

			fn particles(&self, emit: &mut dyn FnMut(f32, f32, Vec3, f32)) {
				emit(0.5, 0.5, vec3(1.0, 1.0, 1.0), 0.5);
				emit(-3.0, 0.5, vec3(1.0, 1.0, 1.0), 1.0);
				emit(0.5, 99.0, vec3(1.0, 1.0, 1.0), 1.0);
			}
		}
		let cells = cells(2, 1, &mut Dust);
		// Half-alpha white over black is linear 0.5, encoded as sRGB 188.
		assert!(cells.contains(&(0, 0, '▀', Color::Rgb(188, 188, 188), Some(Color::Rgb(0, 0, 0)))));
		assert!(cells.contains(&(1, 0, '▀', Color::Rgb(0, 0, 0), Some(Color::Rgb(0, 0, 0)))));
	}

	#[test]
	fn rand01_is_deterministic_and_unit_range() {
		for seed in 0..1000_u32 {
			let sample = rand01(seed);
			assert!((0.0..1.0).contains(&sample));
			assert_eq!(sample, rand01(seed));
		}
	}

	#[test]
	fn advance_sees_the_pixel_resolution() {
		struct Probe(f32, f32);
		impl Program for Probe {
			fn advance(&mut self, _: Duration, width: f32, height: f32) {
				*self = Self(width, height);
			}

			fn fragment(&self, _: f32, _: f32) -> (Vec3, f32) {
				(Vec3::ZERO, 0.0)
			}
		}
		let mut probe = Probe(0.0, 0.0);
		Surface::new().render(&mut probe, Duration::ZERO, 10, 4, |_, _, _, _, _| {});
		assert_eq!((probe.0, probe.1), (10.0, 8.0), "rows double into pixel height");
	}
}
