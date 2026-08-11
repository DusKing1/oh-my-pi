//! The stippled-eclipse effect: a CPU port of the WebGPU shader behind
//! stencil.so's landing page, and the reference [`Program`] implementation.
#![allow(
	clippy::suboptimal_flops,
	clippy::imprecise_flops,
	clippy::manual_midpoint,
	reason = "kept literal to the WGSL original"
)]

use std::{f32::consts::TAU, time::Duration};

use crate::{
	scene::{Vec3, vec3},
	shader::{Program, hash, rand01},
};

/// Upper bound on dust sprites; the arc-area formula stays well under it.
const MAX_PARTICLES: f32 = 4000.0;
/// Reference viewport area: the web shader's 1920×1080 design pixels. The
/// scale factor derived from it drives every distance in the effect, so
/// the geometry survives the ~10× drop to terminal resolution.
const DESIGN_AREA: f32 = 1920.0 * 1080.0;
/// Full glimmer waves along the visible rim.
const GLIMMER_WAVES: f32 = 2.2;

/// Rim anchor caps, as viewport fractions. The legacy scale-based circle
/// decides where the arc enters the top edge and exits the right edge, but
/// each anchor is clamped: entry never past `RIM_TOP_X * width`, exit never
/// below `RIM_RIGHT_Y * height` (legacy exits through the bottom edge count
/// as below). The arc then bulges toward the lower-left by `RIM_BULGE *
/// chord` (sagitta).
const RIM_TOP_X: f32 = 0.405;
const RIM_RIGHT_Y: f32 = 0.585;
const RIM_BULGE: f32 = 0.062;

const SKY: Vec3 = Vec3::rgb(56, 189, 248);
const VIOLET: Vec3 = Vec3::rgb(192, 132, 252);
const PLUM: Vec3 = Vec3::rgb(70, 15, 85);
const SILVER: Vec3 = Vec3::rgb(250, 250, 252);
const BLACK: Vec3 = Vec3::rgb(9, 9, 11);

/// A stippled eclipse in the brand palette.
///
/// An analytic rim arc pinned to the viewport's top and right edges, a
/// sky/violet/plum corona regenerating on a 90-frame cycle, rim glimmer,
/// migrating silver dust, film grain, and a vignette.
///
/// Use it as a full-viewport backdrop — mount it with
/// [`crate::components::Shader`], or drive it through
/// [`Surface::render`](crate::shader::Surface::render) under hand-painted
/// chrome. The field is opaque everywhere, so it always paints the whole
/// target. Every distance derives from `sqrt(area / DESIGN_AREA)`, exactly
/// as on the web, so the geometry adapts to any viewport. All state is
/// recomputed in [`Program::advance`] — the CPU twin of the WGSL `Scene`
/// uniform — making frames a pure function of the clock.
///
/// The GPU original draws stipple sites at 2×2 physical pixels and adds a
/// barely-there monochrome grade pass; here every half-block pixel is a
/// stipple site and the grade is dropped — it sits below one terminal
/// color step.
#[derive(Default)]
pub struct Eclipse {
	width:          f32,
	height:         f32,
	/// Pixel row pitch for stipple/grain cell indices.
	pixel_width:    u32,
	/// `sqrt(area / DESIGN_AREA)`: every falloff, speed, and depth scales
	/// by it, exactly as on the web.
	scale:          f32,
	center:         (f32, f32),
	radius:         f32,
	/// Solid rim band width; the web's `5 * scale` floored at one pixel so
	/// the cyan arc survives coarse terminal pixels.
	rim_band:       f32,
	theta_min:      f32,
	theta_max:      f32,
	glimmer_k:      f32,
	dust_depth:     f32,
	particle_count: u32,
	/// Clock in web design frames (60 per second).
	time_frames:    f32,
	/// `floor(time_frames)`: the stipple regeneration counter.
	frame:          u32,
}

impl Eclipse {
	/// Radial brightness falloff toward the viewport corners.
	fn vignette(&self, x: f32, y: f32) -> f32 {
		let dx = x - self.width * 0.5;
		let dy = y - self.height * 0.5;
		let normalized = (dx * dx + dy * dy).sqrt() / (1600.0 * self.scale);
		(1.0 - normalized.powi(3)).max(0.2)
	}

	/// Slow brightness wave traveling along the rim.
	fn glimmer(&self, theta: f32) -> f32 {
		let phase = self.time_frames * (TAU * GLIMMER_WAVES / 720.0);
		1.0 + 0.16 * (theta * self.glimmer_k - phase).sin()
	}
}

impl Program for Eclipse {
	fn advance(&mut self, now: Duration, width: f32, height: f32) {
		self.time_frames = now.as_secs_f32() * 60.0;
		self.frame = self.time_frames as u32;
		self.width = width;
		self.height = height;
		self.pixel_width = width as u32;
		// The web clamps scale to 0.5..=2.0; a terminal target sits far
		// below the design area, so only guard the degenerate low end.
		let scale = ((width * height) / DESIGN_AREA).sqrt().max(0.01);
		self.scale = scale;
		self.rim_band = (5.0 * scale).max(1.0);

		// Legacy scale-based circle, used only to place the edge anchors.
		let legacy_cx = -724.0 * scale + (width - 1920.0 * scale) * 0.4;
		let legacy_cy = 2000.0 * scale;
		let legacy_r = 2500.0 * scale;
		let legacy_top_x = legacy_cx + (legacy_r * legacy_r - legacy_cy * legacy_cy).sqrt();
		let edge_dx = width - legacy_cx;
		// No right-edge crossing (arc would exit through the bottom) counts
		// as "below the cap", so the cap takes over on wide viewports.
		let legacy_right_y = if legacy_r > edge_dx {
			legacy_cy - (legacy_r * legacy_r - edge_dx * edge_dx).sqrt()
		} else {
			f32::INFINITY
		};
		// Circle through the two capped edge anchors with a chord-relative
		// bulge; the center sits on the plate side (down-left chord normal).
		let (p1x, p1y) = (legacy_top_x.min(width * RIM_TOP_X), 0.0);
		let (p2x, p2y) = (width, legacy_right_y.min(height * RIM_RIGHT_Y));
		let (dx, dy) = (p2x - p1x, p2y - p1y);
		let chord = (dx * dx + dy * dy).sqrt();
		let sagitta = chord * RIM_BULGE;
		let radius = chord * chord / (8.0 * sagitta) + sagitta * 0.5;
		self.center = (
			(p1x + p2x) * 0.5 - dy / chord * (radius - sagitta),
			(p1y + p2y) * 0.5 + dx / chord * (radius - sagitta),
		);
		self.radius = radius;
		self.dust_depth = 500.0 * scale;
		let (theta_min, theta_max) = visible_arc(width, height, self.center.0, self.center.1);
		self.theta_min = theta_min;
		self.theta_max = theta_max;
		self.glimmer_k = TAU * GLIMMER_WAVES / (theta_max - theta_min);
		self.particle_count = (0.10
			* 0.25 * (theta_max - theta_min)
			* (radius - self.dust_depth * 0.5)
			* self.dust_depth
			* 0.25)
			.round()
			.clamp(0.0, MAX_PARTICLES) as u32;
	}

	fn fragment(&self, x: f32, y: f32) -> (Vec3, f32) {
		let index = y as u32 * self.pixel_width + x as u32;
		let grain = (rand01(index ^ 0x9e37_79b9) - 0.5) * Vec3::rgb(24, 24, 24).x;
		let grain = vec3(grain, grain, grain);
		let base = (BLACK * self.vignette(x, y) + grain).clamp01();
		let (dx, dy) = (x - self.center.0, y - self.center.1);
		let rim_distance = (dx * dx + dy * dy).sqrt() - self.radius;

		// Inside the rim sits the featureless plate.
		if rim_distance < 0.0 {
			return (base, 1.0);
		}

		let shade = self.vignette(x, y);
		let theta = dy.atan2(dx);
		let scale = self.scale;
		let ink = if rim_distance < self.rim_band {
			Some(SKY * (shade * self.glimmer(theta)))
		} else {
			// Corona stipple: three exponential shells, each pixel rolling
			// against the summed density on a staggered 90-frame cycle.
			let sky = (-rim_distance / (14.0 * scale)).exp();
			let violet = (-rim_distance / (120.0 * scale)).exp()
				* (rim_distance / (10.0 * scale)).min(1.0)
				* 0.85;
			let plum = (-rim_distance / (600.0 * scale)).exp()
				* (rim_distance / (30.0 * scale)).min(1.0)
				* 0.7;
			let total = (sky + violet + plum).min(1.0);

			if total > 0.003 {
				let bucket = hash(index ^ 0x85eb_ca6b) % 90;
				let generation = (self.frame + 89 - bucket) / 90;
				let roll = rand01(index ^ generation.wrapping_mul(0xc2b2_ae35) ^ 0x2026_0712);
				if roll < sky {
					Some(SKY * (shade * self.glimmer(theta)))
				} else if roll < sky + violet {
					Some(VIOLET * shade)
				} else if roll < total {
					Some(PLUM * shade)
				} else {
					None
				}
			} else {
				None
			}
		};

		ink.map_or((base, 1.0), |ink| ((ink + grain).clamp01(), 1.0))
	}

	fn particles(&self, emit: &mut dyn FnMut(f32, f32, Vec3, f32)) {
		// Dust is pure hash-derived instancing, exactly like the GPU vertex
		// shader: each mote's orbit, speed, and wander come from its index.
		for instance in 0..self.particle_count {
			let seed = hash(instance ^ 0xa511_e9b3);
			let mix = rand01(seed);
			let theta_base = self.theta_min * (1.0 - mix) + self.theta_max * mix;
			let sampled_depth = self.dust_depth * (1.0 - rand01(seed ^ 0x63d8_3595).cbrt());
			let max_depth = sampled_depth.max(14.0 * self.scale);
			let speed = (0.14 + rand01(seed ^ 0x9e37_79b9) * 0.22) * self.scale;
			let offset = rand01(seed ^ 0xc2b2_ae35) * max_depth;
			let travel = self.time_frames * speed + offset;
			let depth = travel - (travel / max_depth).floor() * max_depth;
			let radius = self.radius - depth;
			let wave_frequency = 0.004 + rand01(seed ^ 0x27d4_eb2f) * 0.011;
			let wave_phase = rand01(seed ^ 0x1656_67b1) * TAU;
			let wander = -(0.06 * self.scale / wave_frequency)
				* (self.time_frames * wave_frequency + wave_phase).cos()
				/ radius.max(1.0);
			let theta = theta_base + wander;
			let x = self.center.0 + radius * theta.cos();
			let y = self.center.1 + radius * theta.sin();
			let fade_in = (depth / 3.0).min(1.0);
			let fade_out = (max_depth - depth) / (max_depth * 0.25);
			emit(x, y, SILVER * self.vignette(x, y), fade_in.min(fade_out).clamp(0.0, 1.0));
		}
	}
}

/// Angular span of the rim circle visible inside the viewport, padded a
/// few hundredths of a radian so dust never pops at the edges.
fn visible_arc(width: f32, height: f32, center_x: f32, center_y: f32) -> (f32, f32) {
	let mut theta_min = f32::INFINITY;
	let mut theta_max = f32::NEG_INFINITY;
	for step in 0..=32 {
		let ratio = step as f32 / 32.0;
		for (x, y) in [
			(width * ratio, 0.0),
			(width * ratio, height),
			(0.0, height * ratio),
			(width, height * ratio),
		] {
			let theta = (y - center_y).atan2(x - center_x);
			theta_min = theta_min.min(theta);
			theta_max = theta_max.max(theta);
		}
	}
	(theta_min - 0.04, theta_max + 0.04)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{frame::Color, shader::Surface};

	/// Collects one rendered frame as `(x, y, fg)` cells.
	fn frame_cells(cols: u16, rows: u16, at: Duration) -> Vec<(u16, u16, Color)> {
		let mut cells = Vec::new();
		let mut eclipse = Eclipse::default();
		Surface::new().render(&mut eclipse, at, cols, rows, |x, y, _, fg, _| {
			cells.push((x, y, fg));
		});
		cells
	}

	#[test]
	fn the_field_is_opaque_and_splits_plate_from_corona() {
		let cells = frame_cells(80, 24, Duration::ZERO);
		assert_eq!(cells.len(), 80 * 24, "the eclipse paints every cell");
		let bright = |color: &Color| match color {
			Color::Rgb(r, g, b) => u16::from(*r) + u16::from(*g) + u16::from(*b) > 120,
			_ => false,
		};
		let lit = |x: u16, y: u16| {
			cells
				.iter()
				.any(|&(cx, cy, fg)| cx == x && cy == y && bright(&fg))
		};
		// The stipple is sparse, so single corona cells may roll dark; the
		// solid rim band always lights something.
		assert!(cells.iter().any(|(.., fg)| bright(fg)), "the rim band lights cells");
		assert!(!lit(0, 23), "the plate keeps the bottom-left corner dark");
	}

	#[test]
	fn the_stipple_regenerates_over_time() {
		let start = frame_cells(60, 20, Duration::ZERO);
		let later = frame_cells(60, 20, Duration::from_secs(2));
		assert_ne!(start, later, "the 90-frame cycle re-rolls corona pixels");
	}
}
