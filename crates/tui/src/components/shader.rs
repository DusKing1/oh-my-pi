//! Half-block shader viewport: the retained face of [`crate::shader`].

use crate::{
	anim::FRAME,
	component::{Component, PaintCtx, Slot, next_slot},
	context::UiContext,
	frame::Rect,
	props::{Prop, PropValue, Props},
	shader::{Program, Surface},
};

/// A live effect viewport that renders a [`Program`] into half-block cells.
///
/// The program advances on the shared presentation clock and repaints at
/// [`FRAME`] cadence while presented; a [`still`](Self::still) program
/// paints once and requests nothing. Unlit cells stay transparent, so set
/// `bg` for a backdrop. Mount it in `dom!` as an expression child, or
/// register a factory for a `<shader>` markup tag through
/// [`crate::Elements`].
///
/// ```
/// use omp_tui::{components::Shader, dom, scene::vec3};
///
/// // A still radial glow; animated effects implement `shader::Program`.
/// let glow = |x: f32, y: f32| {
/// 	let (dx, dy) = (x - 12.0, y - 8.0);
/// 	let fall = 1.0 - ((dx * dx + dy * dy).sqrt() / 10.0).min(1.0);
/// 	(vec3(0.4, 0.8, 1.0) * fall, fall)
/// };
/// let tree = dom! {
/// 	<col>
/// 		{Shader::new(glow).size(24, 8).still()}
/// 	</col>
/// };
/// # let _ = tree;
/// ```
pub struct Shader {
	props:   Props,
	slot:    Slot,
	program: Box<dyn Program>,
	surface: Surface,
	cols:    u16,
	rows:    u16,
	live:    bool,
}

impl Shader {
	/// Creates a 24×8-cell viewport over `program`.
	pub fn new(program: impl Program + 'static) -> Self {
		Self {
			props:   Props::new(),
			slot:    next_slot(),
			program: Box::new(program),
			surface: Surface::new(),
			cols:    24,
			rows:    8,
			live:    true,
		}
	}

	/// Sets the viewport size in terminal cells.
	pub const fn size(mut self, cols: u16, rows: u16) -> Self {
		self.cols = cols;
		self.rows = rows;
		self
	}

	/// Paints once instead of animating — for programs that ignore the clock.
	pub const fn still(mut self) -> Self {
		self.live = false;
		self
	}

	/// Sets one shader property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}
}

impl Component for Shader {
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
		let base = self.props.style(&pc.ctx.theme);
		let cols = self.cols.min(rect.width);
		let rows = self.rows.min(rect.height).min(pc.clip - rect.y);
		let frame = &mut *pc.frame;
		let mut buffer = [0_u8; 4];
		self
			.surface
			.render(&mut *self.program, pc.now, cols, rows, |x, y, glyph, fg, bg| {
				let style = match bg {
					Some(bg) => base.fg(fg).bg(bg),
					None => base.fg(fg),
				};
				frame.put(rect.x + x, rect.y + y, glyph.encode_utf8(&mut buffer), style);
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
		scene::{Vec3, vec3},
		test_support::frame_row_text,
		ui::Ui,
	};

	fn wall(_: f32, _: f32) -> (Vec3, f32) {
		(vec3(1.0, 0.0, 0.0), 1.0)
	}

	fn lit_blocks(ui: &Ui) -> bool {
		(0..ui.frame().size().height).any(|row| {
			frame_row_text(ui.frame(), row)
				.chars()
				.any(|glyph| glyph == '▀')
		})
	}

	#[test]
	fn live_shader_paints_half_blocks_and_reschedules() {
		let mut ui = Ui::from_root(Shader::new(wall).size(12, 4), 12, UiContext::default());
		assert!(lit_blocks(&ui), "the wall lights half-block cells");
		assert_eq!(ui.next_wake(), Some(anim::FRAME));
		assert!(ui.tick(anim::FRAME), "the frame deadline repaints");
		assert_eq!(ui.next_wake(), Some(anim::FRAME + anim::FRAME));
	}

	#[test]
	fn still_shader_requests_no_wake() {
		let ui = Ui::from_root(Shader::new(wall).size(12, 4).still(), 12, UiContext::default());
		assert!(lit_blocks(&ui));
		assert_eq!(ui.next_wake(), None);
	}

	#[test]
	fn animated_shader_reads_the_paint_clock() {
		/// A vertical bar sweeping one column per second.
		struct Sweep {
			column: f32,
		}
		impl Program for Sweep {
			fn advance(&mut self, now: Duration, width: f32, _: f32) {
				self.column = now.as_secs_f32() % width;
			}

			fn fragment(&self, x: f32, _: f32) -> (Vec3, f32) {
				let lit = x >= self.column && x < self.column + 1.0;
				(vec3(1.0, 1.0, 1.0), if lit { 1.0 } else { 0.0 })
			}
		}

		let rows = |ui: &Ui| -> Vec<String> {
			(0..ui.frame().size().height)
				.map(|row| frame_row_text(ui.frame(), row))
				.collect()
		};
		let mut ui =
			Ui::from_root(Shader::new(Sweep { column: 0.0 }).size(16, 4), 16, UiContext::default());
		let start = rows(&ui);
		ui.tick(Duration::from_millis(2000));
		assert_ne!(start, rows(&ui), "the clock moves the bar");
	}
}
