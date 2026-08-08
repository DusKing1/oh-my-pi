//! Raytraced braille viewport: the retained face of [`crate::scene`].

use crate::{
	anim::FRAME,
	component::{Component, PaintCtx, Slot, next_slot},
	context::UiContext,
	frame::Rect,
	props::{Prop, PropValue, Props},
	scene::{Trace, rasterize},
};

/// A live 3D viewport that rasterizes a [`Trace`] scene into braille cells.
///
/// The scene advances on the shared presentation clock and repaints at
/// [`FRAME`] cadence while presented; a [`still`](Self::still) scene paints
/// once and requests nothing. Unlit cells stay transparent, so set `bg` for
/// a backdrop. Mount it in `dom!` as an expression child, or register a
/// factory for a `<scene>` markup tag through [`crate::Elements`].
///
/// ```
/// use omp_tui::{
/// 	components::Scene,
/// 	dom,
/// 	scene::{Ray, Vec3, vec3},
/// };
///
/// // A still orb; animated scenes implement `scene::Trace` instead.
/// let orb = |ray: Ray| -> (Vec3, f32) {
/// 	let along = -ray.origin.dot(ray.dir);
/// 	let nearest = ray.origin + ray.dir * along;
/// 	let glow = 1.0 - nearest.dot(nearest);
/// 	(vec3(0.4, 0.8, 1.0) * (0.4 + 0.6 * glow), if glow > 0.0 { 1.0 } else { 0.0 })
/// };
/// let tree = dom! {
/// 	<col>
/// 		{Scene::new(orb).size(24, 8).still()}
/// 	</col>
/// };
/// # let _ = tree;
/// ```
pub struct Scene {
	props: Props,
	slot:  Slot,
	trace: Box<dyn Trace>,
	cols:  u16,
	rows:  u16,
	live:  bool,
}

impl Scene {
	/// Creates a 24×8-cell viewport over `scene`.
	pub fn new(scene: impl Trace + 'static) -> Self {
		Self {
			props: Props::new(),
			slot:  next_slot(),
			trace: Box::new(scene),
			cols:  24,
			rows:  8,
			live:  true,
		}
	}

	/// Sets the viewport size in terminal cells.
	pub const fn size(mut self, cols: u16, rows: u16) -> Self {
		self.cols = cols;
		self.rows = rows;
		self
	}

	/// Paints once instead of animating — for scenes that ignore the clock.
	pub const fn still(mut self) -> Self {
		self.live = false;
		self
	}

	/// Sets one scene property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}
}

impl Component for Scene {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn measure(&mut self, _ctx: &UiContext) -> (u16, u16) {
		(self.cols, self.cols)
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		self.rows
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		if rect.y >= pc.clip || rect.width == 0 || rect.height == 0 {
			return;
		}
		let camera = self.trace.advance(pc.now);
		let base = self.props.style(&pc.ctx.theme);
		let cols = self.cols.min(rect.width);
		let rows = self.rows.min(rect.height).min(pc.clip - rect.y);
		let frame = &mut *pc.frame;
		let mut buffer = [0_u8; 4];
		rasterize(&*self.trace, &camera, cols, rows, |x, y, glyph, color| {
			frame.put(rect.x + x, rect.y + y, glyph.encode_utf8(&mut buffer), base.fg(color));
		});
		if self.live {
			pc.wake(self.slot, pc.now + FRAME);
		}
	}
}

#[cfg(test)]
mod tests {
	use std::time::Duration;

	use super::*;
	use crate::{
		anim,
		scene::{Camera, Ray, Vec3, vec3},
		test_support::frame_row_text,
		ui::Ui,
	};

	fn orb(ray: Ray) -> (Vec3, f32) {
		let along = -ray.origin.dot(ray.dir);
		let nearest = ray.origin + ray.dir * along;
		if along > 0.0 && nearest.dot(nearest) <= 1.0 {
			(vec3(1.0, 1.0, 1.0), 1.0)
		} else {
			(Vec3::ZERO, 0.0)
		}
	}

	fn lit_braille(ui: &Ui) -> bool {
		(0..ui.frame().size().height).any(|row| {
			frame_row_text(ui.frame(), row)
				.chars()
				.any(|glyph| ('\u{2801}'..='\u{28ff}').contains(&glyph))
		})
	}

	#[test]
	fn live_scene_paints_braille_and_reschedules() {
		let mut ui = Ui::from_root(Scene::new(orb).size(12, 6), 12, UiContext::default());
		assert!(lit_braille(&ui), "the orb lights braille cells");
		assert_eq!(ui.next_wake(), Some(anim::FRAME));
		assert!(ui.tick(anim::FRAME), "the frame deadline repaints");
		assert_eq!(ui.next_wake(), Some(anim::FRAME + anim::FRAME));
	}

	#[test]
	fn still_scene_requests_no_wake() {
		let ui = Ui::from_root(Scene::new(orb).size(12, 6).still(), 12, UiContext::default());
		assert!(lit_braille(&ui));
		assert_eq!(ui.next_wake(), None);
	}

	#[test]
	fn animated_scene_reads_the_paint_clock() {
		/// Sweeps the camera a quarter turn per second.
		struct Turntable;
		impl Trace for Turntable {
			fn advance(&mut self, now: Duration) -> Camera {
				Camera { yaw: now.as_secs_f32() * std::f32::consts::FRAC_PI_2, ..Camera::default() }
			}

			fn shade(&self, ray: Ray) -> (Vec3, f32) {
				orb(Ray { origin: ray.origin - vec3(1.4, 0.0, 0.0), dir: ray.dir })
			}
		}

		let rows = |ui: &Ui| -> Vec<String> {
			(0..ui.frame().size().height)
				.map(|row| frame_row_text(ui.frame(), row))
				.collect()
		};
		let mut ui = Ui::from_root(Scene::new(Turntable).size(16, 8), 16, UiContext::default());
		let start = rows(&ui);
		ui.tick(Duration::from_millis(500));
		assert_ne!(start, rows(&ui), "the clock swings the camera");
	}
}
