//! Deterministic CPU ray tracing packed into braille terminal cells.
//!
//! [`PathTracer`] combines analytic [`Primitive`] geometry, an owning [`Bvh`],
//! principled [`Material`]s, analytic lights, and bounded indirect transport.
//! It traces shadows, GGX reflections, dielectric refraction, emissive
//! surfaces, and environment illumination without per-ray allocation.
//! Implement [`Trace`] directly when a procedural scene needs custom shading
//! or animation; [`rasterize`] accepts either form.
//!
//! # Example
//! ```
//! use omp_tui::scene::{self, Light, Material, Object, PathTracer, Sphere, Vec3, World, vec3};
//!
//! let world = World::new(vec![Object::new(
//! 	Sphere::new(Vec3::ZERO, 1.0),
//! 	Material::diffuse(Vec3::rgb(56, 189, 248)),
//! )])
//! .with_light(Light::directional(vec3(-1.0, -1.0, -1.0), Vec3::ONE, 2.0))
//! .with_environment(Vec3::rgb(5, 7, 12));
//! let tracer = PathTracer::new(world);
//! let mut lit = 0;
//! scene::rasterize(&tracer, &Default::default(), 20, 8, |_, _, _, _| lit += 1);
//! assert!(lit > 0);
//! ```
#![allow(clippy::suboptimal_flops, reason = "mul_add chains obscure the ray math")]

mod geometry;
mod integrator;
mod material;

use std::{
	ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub},
	time::Duration,
};

pub use geometry::{Aabb, Bvh, Disk, Geometry, GeometryHit, Hit, Object, Primitive, Quad, Sphere};
pub use integrator::{Integrator, Light, PathTracer, World};
pub use material::Material;

use crate::frame::Color;

/// A 3-component `f32` vector used for points, directions, and linear colors.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
	/// X component (red when used as a color).
	pub x: f32,
	/// Y component (green when used as a color).
	pub y: f32,
	/// Z component (blue when used as a color).
	pub z: f32,
}

/// Shorthand [`Vec3`] constructor.
pub const fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
	Vec3 { x, y, z }
}

const fn fifth_root(value: f32) -> f32 {
	let mut low = 0.0_f32;
	let mut high = 1.0_f32;
	let mut iteration = 0;
	while iteration < 24 {
		let middle = low.midpoint(high);
		let square = middle * middle;
		if square * square * middle < value {
			low = middle;
		} else {
			high = middle;
		}
		iteration += 1;
	}
	low.midpoint(high)
}

const fn srgb_to_linear(channel: u8) -> f32 {
	let encoded = channel as f32 / 255.0;
	if encoded <= 0.04045 {
		encoded / 12.92
	} else {
		let base = (encoded + 0.055) / 1.055;
		let square = base * base;
		square * fifth_root(square)
	}
}

impl Vec3 {
	/// The vector with every component set to one.
	pub const ONE: Self = vec3(1.0, 1.0, 1.0);
	/// The zero vector.
	pub const ZERO: Self = vec3(0.0, 0.0, 0.0);

	/// Creates a vector with every component set to `value`.
	pub const fn splat(value: f32) -> Self {
		vec3(value, value, value)
	}

	/// Decodes an sRGB byte triple into linear-light components.
	pub const fn rgb(red: u8, green: u8, blue: u8) -> Self {
		vec3(srgb_to_linear(red), srgb_to_linear(green), srgb_to_linear(blue))
	}

	/// Dot product.
	pub fn dot(self, other: Self) -> f32 {
		self.x * other.x + self.y * other.y + self.z * other.z
	}

	/// Squared vector length.
	pub fn length_squared(self) -> f32 {
		self.dot(self)
	}

	/// Vector length.
	pub fn length(self) -> f32 {
		self.length_squared().sqrt()
	}

	/// Cross product.
	pub fn cross(self, other: Self) -> Self {
		vec3(
			self.y * other.z - self.z * other.y,
			self.z * other.x - self.x * other.z,
			self.x * other.y - self.y * other.x,
		)
	}

	/// Unit-length copy; near-zero vectors stay finite.
	pub fn normalize(self) -> Self {
		self * (1.0 / self.dot(self).sqrt().max(1e-8))
	}

	/// Reflection of this direction around `normal`.
	pub fn reflect(self, normal: Self) -> Self {
		self - normal * (2.0 * self.dot(normal))
	}

	/// Refraction through `normal` at the incident/transmitted IOR ratio.
	///
	/// Returns `None` when total internal reflection prevents transmission.
	pub fn refract(self, normal: Self, eta: f32) -> Option<Self> {
		let cos_theta = (-self).dot(normal).min(1.0);
		let perpendicular = (self + normal * cos_theta) * eta;
		let parallel_squared = 1.0 - perpendicular.length_squared();
		if parallel_squared < 0.0 {
			None
		} else {
			Some(perpendicular - normal * parallel_squared.sqrt())
		}
	}

	/// Largest component.
	pub const fn max_component(self) -> f32 {
		self.x.max(self.y).max(self.z)
	}

	/// Whether every component is finite.
	pub const fn is_finite(self) -> bool {
		self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
	}

	/// Componentwise clamp to `0..=1`.
	pub const fn clamp01(self) -> Self {
		vec3(self.x.clamp(0.0, 1.0), self.y.clamp(0.0, 1.0), self.z.clamp(0.0, 1.0))
	}

	/// Linear interpolation toward `to` by `mix` (0 = self, 1 = to).
	pub fn lerp(self, to: Self, mix: f32) -> Self {
		self * (1.0 - mix) + to * mix
	}
}

impl Add for Vec3 {
	type Output = Self;

	fn add(self, other: Self) -> Self {
		vec3(self.x + other.x, self.y + other.y, self.z + other.z)
	}
}

impl Sub for Vec3 {
	type Output = Self;

	fn sub(self, other: Self) -> Self {
		vec3(self.x - other.x, self.y - other.y, self.z - other.z)
	}
}

impl Neg for Vec3 {
	type Output = Self;

	fn neg(self) -> Self {
		vec3(-self.x, -self.y, -self.z)
	}
}

impl AddAssign for Vec3 {
	fn add_assign(&mut self, other: Self) {
		*self = *self + other;
	}
}

impl Mul<f32> for Vec3 {
	type Output = Self;

	fn mul(self, factor: f32) -> Self {
		vec3(self.x * factor, self.y * factor, self.z * factor)
	}
}

impl Mul for Vec3 {
	type Output = Self;

	fn mul(self, other: Self) -> Self {
		vec3(self.x * other.x, self.y * other.y, self.z * other.z)
	}
}

impl Mul<Vec3> for f32 {
	type Output = Vec3;

	fn mul(self, vector: Vec3) -> Vec3 {
		vector * self
	}
}

impl MulAssign<f32> for Vec3 {
	fn mul_assign(&mut self, factor: f32) {
		*self = *self * factor;
	}
}

impl Div<f32> for Vec3 {
	type Output = Self;

	fn div(self, divisor: f32) -> Self {
		self * (1.0 / divisor)
	}
}

/// Encodes linear-light components through the sRGB transfer function.
impl From<Vec3> for Color {
	fn from(color: Vec3) -> Self {
		let channel = |linear: f32| {
			let linear = linear.clamp(0.0, 1.0);
			let encoded = if linear <= 0.003_130_8 {
				linear * 12.92
			} else {
				1.055 * linear.powf(1.0 / 2.4) - 0.055
			};
			(encoded * 255.0).round() as u8
		};
		Self::Rgb(channel(color.x), channel(color.y), channel(color.z))
	}
}

/// One camera ray: `origin + dir * t`, with `dir` unit length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
	/// Camera position.
	pub origin: Vec3,
	/// Unit direction of travel.
	pub dir:    Vec3,
}

impl Ray {
	/// Creates a ray and normalizes its travel direction.
	pub fn new(origin: Vec3, direction: Vec3) -> Self {
		Self { origin, dir: direction.normalize() }
	}

	/// Point reached after travelling `distance` along the ray.
	pub fn at(self, distance: f32) -> Vec3 {
		self.origin + self.dir * distance
	}
}

/// An orbit camera: a position on a sphere around `target`, looking at it.
///
/// The default is a gentle three-quarter view sized for small scenes near
/// the origin — closure scenes get it for free through [`Trace::advance`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
	/// Point the camera looks at.
	pub target:   Vec3,
	/// Rotation around the vertical axis, in radians.
	pub yaw:      f32,
	/// Elevation above the horizon, in radians.
	pub pitch:    f32,
	/// Distance from `target`.
	pub distance: f32,
	/// Extra vertical offset of the camera position; the camera keeps
	/// aiming at `target`, so lifting tilts the view.
	pub lift:     f32,
	/// Ray focal length: higher values narrow the field of view.
	pub focal:    f32,
}

impl Default for Camera {
	fn default() -> Self {
		Self {
			target:   Vec3::ZERO,
			yaw:      0.0,
			pitch:    0.44,
			distance: 4.2,
			lift:     0.0,
			focal:    2.7,
		}
	}
}

impl Camera {
	fn axes(&self) -> Axes {
		let origin = self.target
			+ vec3(
				self.distance * self.pitch.cos() * self.yaw.sin(),
				self.distance * self.pitch.sin() + self.lift,
				self.distance * self.pitch.cos() * self.yaw.cos(),
			);
		let forward = (self.target - origin).normalize();
		let right = forward.cross(vec3(0.0, 1.0, 0.0)).normalize();
		let up = right.cross(forward).normalize();
		Axes { origin, forward: forward * self.focal, right, up }
	}
}

/// Per-frame ray basis: the camera origin and a pre-scaled view frame.
struct Axes {
	origin:  Vec3,
	forward: Vec3,
	right:   Vec3,
	up:      Vec3,
}

impl Axes {
	fn ray(&self, x: f32, y: f32) -> Ray {
		Ray { origin: self.origin, dir: (self.forward + self.right * x + self.up * y).normalize() }
	}
}

/// A raytraced scene: per-frame state plus a shader for every ray.
///
/// [`shade`](Self::shade) returns a unit-range color and a coverage alpha.
/// Coverage decides which braille dots light; color, weighted by coverage,
/// decides each cell's tint. Any `Fn(Ray) -> (Vec3, f32)` closure is a still
/// scene viewed through [`Camera::default`].
pub trait Trace {
	/// Advances animation state to `now` and returns this frame's camera.
	fn advance(&mut self, now: Duration) -> Camera {
		let _ = now;
		Camera::default()
	}

	/// Shades one ray: `(color, coverage)`, both in unit range.
	fn shade(&self, ray: Ray) -> (Vec3, f32);
}

impl<F: Fn(Ray) -> (Vec3, f32)> Trace for F {
	fn shade(&self, ray: Ray) -> (Vec3, f32) {
		self(ray)
	}
}

/// Rays per braille-dot side; every dot averages the square of this.
const SUPERSAMPLE: usize = 2;
/// Coverage at which a braille dot lights.
const DOT_THRESHOLD: f32 = 0.24;
/// World half-height of the view plane at the focal distance.
const HALF_HEIGHT: f32 = 0.98;

/// Braille dot layout: (x, y) inside the 2x4 cell and the bit it sets.
const BRAILLE_DOTS: [(usize, usize, u32); 8] = [
	(0, 0, 0x01),
	(0, 1, 0x02),
	(0, 2, 0x04),
	(1, 0, 0x08),
	(1, 1, 0x10),
	(1, 2, 0x20),
	(0, 3, 0x40),
	(1, 3, 0x80),
];

/// Traces `scene` through `camera` into a `cols` × `rows` grid of braille
/// cells with 2× supersampling.
///
/// `put` runs once per lit cell as `(column, row, glyph, color)`. Cells with
/// no dot over the coverage threshold are skipped, so whatever sits behind
/// them shows through.
pub fn rasterize<T: Trace + ?Sized>(
	scene: &T,
	camera: &Camera,
	cols: u16,
	rows: u16,
	mut put: impl FnMut(u16, u16, char, Color),
) {
	if cols == 0 || rows == 0 {
		return;
	}
	let axes = camera.axes();
	let pixel_w = cols as usize * 2 * SUPERSAMPLE;
	let pixel_h = rows as usize * 4 * SUPERSAMPLE;
	// Braille dots are square on a typical 1:2 terminal cell, so the view
	// aspect is simply the raster aspect.
	let half_w = HALF_HEIGHT * pixel_w as f32 / pixel_h as f32;
	let step_x = 2.0 * half_w / pixel_w as f32;
	let step_y = 2.0 * HALF_HEIGHT / pixel_h as f32;
	for row in 0..rows as usize {
		for col in 0..cols as usize {
			let mut mask = 0_u32;
			let mut cell_color = Vec3::ZERO;
			let mut cell_weight = 0.0_f32;
			for &(dot_x, dot_y, bit) in &BRAILLE_DOTS {
				let mut color_sum = Vec3::ZERO;
				let mut coverage_sum = 0.0_f32;
				for sub_y in 0..SUPERSAMPLE {
					for sub_x in 0..SUPERSAMPLE {
						let px = ((col * 2 + dot_x) * SUPERSAMPLE + sub_x) as f32;
						let py = ((row * 4 + dot_y) * SUPERSAMPLE + sub_y) as f32;
						let (color, alpha) = scene.shade(
							axes.ray((px + 0.5) * step_x - half_w, HALF_HEIGHT - (py + 0.5) * step_y),
						);
						color_sum += color * alpha;
						coverage_sum += alpha;
					}
				}
				let coverage = coverage_sum / (SUPERSAMPLE * SUPERSAMPLE) as f32;
				if coverage >= DOT_THRESHOLD {
					mask |= bit;
					cell_color += color_sum * (coverage / coverage_sum.max(1e-6));
					cell_weight += coverage;
				}
			}
			if mask == 0 {
				continue;
			}
			let Some(glyph) = char::from_u32(0x2800 + mask) else {
				continue;
			};
			put(col as u16, row as u16, glyph, Color::from(cell_color * (1.0 / cell_weight)));
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A hard-edged white unit sphere at the origin.
	fn sphere(ray: Ray) -> (Vec3, f32) {
		let along = -ray.origin.dot(ray.dir);
		let nearest = ray.origin + ray.dir * along;
		if along > 0.0 && nearest.dot(nearest) <= 1.0 {
			(vec3(1.0, 1.0, 1.0), 1.0)
		} else {
			(Vec3::ZERO, 0.0)
		}
	}

	fn cells(cols: u16, rows: u16, scene: impl Trace) -> Vec<(u16, u16, char, Color)> {
		let mut out = Vec::new();
		rasterize(&scene, &Camera::default(), cols, rows, |x, y, glyph, color| {
			out.push((x, y, glyph, color));
		});
		out
	}

	#[test]
	fn srgb_inputs_round_trip_through_linear_light() {
		for channel in 0..=u8::MAX {
			assert_eq!(
				Color::from(Vec3::rgb(channel, channel, channel)),
				Color::Rgb(channel, channel, channel)
			);
		}
	}

	#[test]
	fn linear_midpoint_uses_the_srgb_transfer_curve() {
		assert_eq!(Color::from(Vec3::splat(0.5)), Color::Rgb(188, 188, 188));
	}

	#[test]
	fn sphere_lights_the_center_and_spares_the_corners() {
		let cells = cells(21, 9, sphere);
		assert!(
			cells
				.iter()
				.any(|&(x, y, glyph, color)| x == 10
					&& y == 4 && glyph == '\u{28ff}'
					&& color == Color::Rgb(255, 255, 255)),
			"the center cell is fully covered in white"
		);
		let corner = |x: u16, y: u16| cells.iter().any(|&(cx, cy, ..)| cx == x && cy == y);
		assert!(!corner(0, 0) && !corner(20, 0) && !corner(0, 8) && !corner(20, 8));
	}

	#[test]
	fn coverage_below_the_dot_threshold_stays_dark() {
		let haze = |_: Ray| (vec3(1.0, 1.0, 1.0), 0.2);
		assert_eq!(cells(8, 4, haze).len(), 0, "0.2 coverage sits under the 0.24 dot threshold");
	}

	#[test]
	fn full_coverage_lights_every_dot_of_every_cell() {
		let wall = |_: Ray| (vec3(1.0, 0.0, 0.0), 1.0);
		let cells = cells(8, 4, wall);
		assert_eq!(cells.len(), 8 * 4);
		assert!(
			cells
				.iter()
				.all(|&(.., glyph, color)| glyph == '\u{28ff}' && color == Color::Rgb(255, 0, 0))
		);
	}

	#[test]
	fn yaw_orbits_around_an_off_axis_scene() {
		// A sphere pushed off-center along +X lands on opposite sides of the
		// view when the camera makes a half-turn.
		let offset =
			|ray: Ray| sphere(Ray { origin: ray.origin - vec3(1.4, 0.0, 0.0), dir: ray.dir });
		let spots = |camera: &Camera| {
			let mut spots = Vec::new();
			rasterize(&offset, camera, 21, 9, |x, y, _, _| spots.push((x, y)));
			spots
		};
		let front = spots(&Camera::default());
		let back = spots(&Camera { yaw: std::f32::consts::PI, ..Camera::default() });
		assert_ne!(front.len(), 0);
		assert_ne!(back.len(), 0);
		assert_ne!(front, back, "orbiting the camera reframes the scene");
	}
}
