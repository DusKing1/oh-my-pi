//! Time-driven animation primitives for retained and immediate paints.
//!
//! Everything here is a pure function of a caller-supplied clock — a
//! [`Duration`](std::time::Duration) since an arbitrary epoch — so animated
//! paints stay deterministic and testable. Retained components read the clock
//! from [`crate::PaintCtx::now`] and request a future repaint with
//! [`crate::PaintCtx::wake`]; immediate-mode painters call the same types
//! with their own elapsed time.
//!
//! Three shapes cover terminal animation:
//! - [`Frames`](crate::anim::Frames): periodic glyph cycles (spinners, pulses)
//!   that run while a state holds and never finish.
//! - [`Tween`](crate::anim::Tween): finite eased interpolations (color fades,
//!   progress smoothing) that settle, and can be retargeted mid-flight without
//!   a visual jump.
//! - [`Reveal`](crate::anim::Reveal): a paced cursor chasing a growing unit
//!   total (streamed text), catching up smoothly and settling once even.
//!
//! Components get all of this declaratively: the `anim`, `ease`, and `spin`
//! properties tween `fg`/`bg`/`bc` colors, gradient endpoints, `w`/`h`
//! sizes, and gradient rotation on any component without custom paint code
//! — see [`crate::Props::anim`], [`crate::Props::ease`], and
//! [`crate::Props::spin`]. The `shimmer` property sweeps a
//! [`Shimmer`](crate::anim::Shimmer) brightness crest across `<text>`, the
//! `reveal` property paces streamed `<text>` content through a [`Reveal`]
//! cursor, and the `hover` and `lift` properties ride the same clock:
//! pointer-driven border chrome and elevation ease through their declared
//! `anim`/`ease`, and snap without one.

use std::time::Duration;

use strum::{EnumString, IntoStaticStr};

use crate::frame::{Color, Style};

/// Repaint cadence for continuously changing values (mid-flight tweens,
/// gradient spins): fast enough to read as motion in a terminal, slow
/// enough to stay cheap. Pass `now + FRAME` to [`crate::PaintCtx::wake`].
pub const FRAME: Duration = Duration::from_millis(33);

/// Easing curve applied to a tween's normalized progress.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, EnumString, IntoStaticStr)]
pub enum Easing {
	/// Constant velocity.
	#[default]
	#[strum(serialize = "linear")]
	Linear,
	/// Cubic acceleration from rest.
	#[strum(serialize = "in")]
	EaseIn,
	/// Cubic deceleration to rest.
	#[strum(serialize = "out")]
	EaseOut,
	/// Cubic acceleration, then deceleration.
	#[strum(serialize = "in-out")]
	EaseInOut,
}

impl Easing {
	/// Maps linear progress onto the eased curve; both ends clamp to `[0, 1]`.
	#[must_use]
	pub fn apply(self, t: f32) -> f32 {
		let t = t.clamp(0.0, 1.0);
		match self {
			Self::Linear => t,
			Self::EaseIn => t * t * t,
			Self::EaseOut => {
				let inv = 1.0 - t;
				inv.mul_add(-inv * inv, 1.0)
			},
			Self::EaseInOut if t < 0.5 => 4.0 * t * t * t,
			Self::EaseInOut => {
				let inv = (-2.0f32).mul_add(t, 2.0);
				inv.mul_add(-inv * inv / 2.0, 1.0)
			},
		}
	}
}

/// Values a [`Tween`] can interpolate.
pub trait Lerp: Copy {
	/// Blends from `self` toward `to` at eased progress `t` in `[0, 1]`.
	#[must_use]
	fn lerp(self, to: Self, t: f32) -> Self;
}

impl Lerp for f32 {
	fn lerp(self, to: Self, t: f32) -> Self {
		(to - self).mul_add(t, self)
	}
}

impl Lerp for u8 {
	fn lerp(self, to: Self, t: f32) -> Self {
		f32::from(self).lerp(f32::from(to), t).round() as Self
	}
}

impl Lerp for u16 {
	fn lerp(self, to: Self, t: f32) -> Self {
		f32::from(self).lerp(f32::from(to), t).round() as Self
	}
}

/// RGB endpoints blend per channel; palette and default endpoints have no
/// interpolable space, so they snap at the halfway point.
impl Lerp for Color {
	fn lerp(self, to: Self, t: f32) -> Self {
		match (self, to) {
			(Self::Rgb(r0, g0, b0), Self::Rgb(r1, g1, b1)) => {
				Self::Rgb(r0.lerp(r1, t), g0.lerp(g1, t), b0.lerp(b1, t))
			},
			_ if t < 0.5 => self,
			_ => to,
		}
	}
}

/// Pairs blend componentwise on one shared eased clock — the natural shape
/// for two-stop color ramps.
impl<A: Lerp, B: Lerp> Lerp for (A, B) {
	fn lerp(self, to: Self, t: f32) -> Self {
		(self.0.lerp(to.0, t), self.1.lerp(to.1, t))
	}
}

/// A finite eased interpolation between two values on a caller-supplied
/// clock.
///
/// A tween never owns time: sampling and retargeting take `now`, so one
/// value drives retained repaints and immediate-mode paints alike.
/// [`Tween::retarget`] restarts from the current sample, so interrupting a
/// running transition never jumps.
///
/// # Example
/// ```
/// use std::time::Duration;
///
/// use omp_tui::anim::{Easing, Tween};
///
/// let mut fade = Tween::settled(0.0f32);
/// fade.retarget(Duration::ZERO, 1.0, Duration::from_millis(100), Easing::Linear);
/// assert_eq!(fade.sample(Duration::from_millis(50)), 0.5);
/// assert!(fade.is_settled(Duration::from_millis(100)));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Tween<T: Lerp> {
	from:     T,
	to:       T,
	start:    Duration,
	duration: Duration,
	easing:   Easing,
}

impl<T: Lerp> Tween<T> {
	/// A tween already settled at `value`.
	pub const fn settled(value: T) -> Self {
		Self {
			from:     value,
			to:       value,
			start:    Duration::ZERO,
			duration: Duration::ZERO,
			easing:   Easing::Linear,
		}
	}

	/// The value at time `now`.
	pub fn sample(&self, now: Duration) -> T {
		if self.duration.is_zero() {
			return self.to;
		}
		let t = now
			.saturating_sub(self.start)
			.div_duration_f32(self.duration);
		self.from.lerp(self.to, self.easing.apply(t))
	}

	/// The value the tween is heading toward.
	pub const fn target(&self) -> T {
		self.to
	}

	/// Whether the tween has reached its target at time `now`.
	pub fn is_settled(&self, now: Duration) -> bool {
		now >= self.settles_at()
	}

	/// When the tween reaches its target — the deadline for the final frame.
	pub const fn settles_at(&self) -> Duration {
		self.start.saturating_add(self.duration)
	}

	/// Redirects the tween toward `to` over `duration`, starting from the
	/// value currently on screen. A matching target is a no-op, so callers
	/// may retarget unconditionally on every state change.
	pub fn retarget(&mut self, now: Duration, to: T, duration: Duration, easing: Easing)
	where
		T: PartialEq,
	{
		if self.to == to {
			return;
		}
		self.from = self.sample(now);
		self.to = to;
		self.start = now;
		self.duration = duration;
		self.easing = easing;
	}
}

/// A periodic glyph cycle on a caller-supplied clock.
///
/// Pure phase arithmetic: the frame at `now` and the instant of the next
/// change derive from `now mod interval`, so cycles stay aligned no matter
/// when or how often they are sampled.
#[derive(Clone, Copy, Debug)]
pub struct Frames {
	frames:   &'static [&'static str],
	interval: Duration,
}

impl Frames {
	/// Braille spinner for Unicode-capable terminals.
	pub const SPINNER: Self =
		Self::new(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"], Duration::from_millis(80));
	/// Four-spoke spinner safe for 7-bit ASCII terminals.
	pub const SPINNER_ASCII: Self = Self::new(&["|", "/", "-", "\\"], Duration::from_millis(120));

	/// Creates a cycle stepping through `frames` every `interval`.
	///
	/// # Panics
	/// When `frames` is empty or `interval` is zero.
	pub const fn new(frames: &'static [&'static str], interval: Duration) -> Self {
		assert!(!frames.is_empty(), "a frame cycle needs at least one frame");
		assert!(!interval.is_zero(), "a frame cycle needs a nonzero interval");
		Self { frames, interval }
	}

	/// The glyph on screen at time `now`.
	pub const fn at(&self, now: Duration) -> &'static str {
		let step = now.as_nanos() / self.interval.as_nanos();
		self.frames[(step % self.frames.len() as u128) as usize]
	}

	/// When the glyph after `now` appears — the deadline to pass to
	/// [`crate::PaintCtx::wake`].
	pub const fn next_change(&self, now: Duration) -> Duration {
		let interval = self.interval.as_nanos();
		let remaining = interval - now.as_nanos() % interval;
		now.saturating_add(Duration::from_nanos(remaining as u64))
	}
}

/// A brightness crest sweeping across one line of cells.
///
/// Pure phase arithmetic on the shared clock, like a gradient `spin`: the
/// crest travels a track padded on both sides so it fully enters and exits
/// the text instead of popping at an edge, then wraps and sweeps again.
/// Cells away from the crest keep the authored style, its shoulders
/// brighten, and the peak lifts further and paints bold. `<text>` consumes
/// this declaratively through the `shimmer` property — see
/// [`crate::Props::shimmer`].
#[derive(Clone, Copy, Debug)]
pub struct Shimmer {
	/// Crest center in cells from the start of the padded track.
	position: f32,
}

impl Shimmer {
	/// Half-width of the cosine crest, in cells.
	const HALF_WIDTH: f32 = 6.0;
	/// Intensity at or above which a cell paints bold.
	const HIGH: f32 = 0.65;
	/// Intensity at or above which a cell keeps the base style.
	const MID: f32 = 0.22;
	/// Off-text runway on each side of the track, in cells.
	const PADDING: f32 = 10.0;

	/// Places the crest at `now`: one sweep across `length` cells (plus
	/// runway) per `period`.
	pub fn new(now: Duration, period: Duration, length: u16) -> Self {
		let track = Self::PADDING.mul_add(2.0, f32::from(length));
		let period = period.as_secs_f32().max(f32::EPSILON);
		let phase = (now.as_secs_f32() / period).fract();
		Self { position: phase * track }
	}

	/// Picks one of three values by crest intensity at `cell` (zero-based
	/// from the text start): `low` off the crest, `mid` on its shoulders,
	/// `high` at the peak. The seam for custom palettes — [`Shimmer::style_at`]
	/// is this with the dim/base/bold derivation.
	pub fn pick<T>(&self, cell: u16, low: T, mid: T, high: T) -> T {
		let distance = (f32::from(cell) + Self::PADDING - self.position).abs();
		if distance >= Self::HALF_WIDTH {
			return low;
		}
		let angle = std::f32::consts::PI * distance / Self::HALF_WIDTH;
		let intensity = f32::midpoint(1.0, angle.cos());
		if intensity >= Self::HIGH {
			high
		} else if intensity >= Self::MID {
			mid
		} else {
			low
		}
	}

	/// The style for `cell` (zero-based from the text start). Shimmer is
	/// additive: `base` rests unchanged off the crest, so a shimmering line
	/// matches its non-shimmering appearance except under the sweep. An RGB
	/// foreground lifts one-fifth toward white on the shoulders and
	/// two-fifths plus bold at the peak; a default or indexed foreground
	/// carries no channel data, so only the peak's bold shows.
	pub fn style_at(&self, cell: u16, base: Style) -> Style {
		let Color::Rgb(red, green, blue) = base.foreground_color() else {
			return self.pick(cell, base, base, base.bold());
		};
		let lift = |channel: u8, fifths: u16| {
			(u16::from(channel) + (255 - u16::from(channel)) * fifths / 5) as u8
		};
		let toward_white =
			|fifths: u16| Color::Rgb(lift(red, fifths), lift(green, fifths), lift(blue, fifths));
		self.pick(cell, base, base.fg(toward_white(1)), base.fg(toward_white(2)).bold())
	}
}

/// A paced cursor over streamed units — grapheme clusters, rows, blocks —
/// that chases a growing total and settles once even with it.
///
/// Unlike the phase-arithmetic shapes, this is an integrator: each
/// [`advance`](Reveal::advance) moves an internal cursor forward, in two
/// regimes. While the backlog exceeds one horizon's worth of the floor
/// rate it decays exponentially with e-folding time `horizon`, so a
/// bursty producer accelerates the reveal instead of queueing behind it;
/// the remainder drains linearly at [`Reveal::MIN_RATE`] units per
/// second, so the tail types out steadily rather than slowing
/// asymptotically.
///
/// Progress per sample is capped at one [`FRAME`] of elapsed time, so the
/// cursor follows frames actually observed — like a fixed-cadence
/// interval timer. Stale paint clocks, stalled hosts, idle gaps, and a
/// settled cursor awaiting more content all resume at the frame cadence
/// instead of jumping. `<text>` consumes this declaratively through the
/// `reveal` property — see [`crate::Props::reveal`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Reveal {
	/// Units currently revealed, fractional between frames.
	shown: f32,
	/// Previous sample instant; `None` while idle or settled.
	last:  Option<Duration>,
}

impl Reveal {
	/// Floor reveal rate in units per second (3 per 33 ms frame).
	pub const MIN_RATE: f32 = 90.0;

	/// A cursor with nothing revealed and the clock disarmed.
	pub const fn new() -> Self {
		Self { shown: 0.0, last: None }
	}

	/// Advances the cursor toward `total` at `now` and returns the whole
	/// units revealed. A zero `horizon` snaps to `total`; a `total` below
	/// the cursor clamps it down (content shrank in place).
	pub fn advance(&mut self, now: Duration, total: usize, horizon: Duration) -> usize {
		let target = total as f32;
		if self.shown >= target {
			self.shown = target;
			self.last = None;
			return total;
		}
		// One frame is the most a single sample may earn: the first sample
		// of a run arms the clock and earns nothing.
		let elapsed = self
			.last
			.map_or(Duration::ZERO, |prev| now.saturating_sub(prev).min(FRAME));
		self.last = Some(now);
		let horizon = horizon.as_secs_f32();
		if horizon <= 0.0 {
			self.shown = target;
			self.last = None;
			return total;
		}
		let mut backlog = target - self.shown;
		let mut dt = elapsed.as_secs_f32();
		// Catch-up regime: exact exponential decay while the implied rate
		// exceeds the floor, then hand the leftover time to the floor.
		let floor = Self::MIN_RATE * horizon;
		if backlog > floor {
			let cross = horizon * (backlog / floor).ln();
			if dt < cross {
				backlog *= (-dt / horizon).exp();
				dt = 0.0;
			} else {
				backlog = floor;
				dt -= cross;
			}
		}
		backlog = Self::MIN_RATE.mul_add(-dt, backlog).max(0.0);
		if backlog <= 0.0 {
			self.shown = target;
			self.last = None;
			return total;
		}
		self.shown = target - backlog;
		(self.shown as usize).min(total)
	}

	/// Restarts from nothing — the content was replaced, not extended.
	pub const fn reset(&mut self) {
		self.shown = 0.0;
		self.last = None;
	}

	/// Whether the cursor has caught up with `total`.
	pub fn is_settled(&self, total: usize) -> bool {
		self.shown >= total as f32
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn easing_curves_hit_both_endpoints_and_stay_ordered() {
		for easing in [Easing::Linear, Easing::EaseIn, Easing::EaseOut, Easing::EaseInOut] {
			assert_eq!(easing.apply(0.0), 0.0, "{easing:?} must start at rest");
			assert!((easing.apply(1.0) - 1.0).abs() < 1e-6, "{easing:?} must land on the target");
			assert!(easing.apply(-1.0) == 0.0 && (easing.apply(2.0) - 1.0).abs() < 1e-6);
		}
		assert!(Easing::EaseIn.apply(0.25) < 0.25 && Easing::EaseOut.apply(0.25) > 0.25);
	}

	#[test]
	fn color_lerp_blends_rgb_and_snaps_unblendable_endpoints() {
		let midpoint = Color::Rgb(0, 100, 200).lerp(Color::Rgb(100, 200, 0), 0.5);
		assert_eq!(midpoint, Color::Rgb(50, 150, 100));
		assert_eq!(Color::Indexed(1).lerp(Color::Rgb(9, 9, 9), 0.4), Color::Indexed(1));
		assert_eq!(Color::Indexed(1).lerp(Color::Rgb(9, 9, 9), 0.6), Color::Rgb(9, 9, 9));
	}

	#[test]
	fn retarget_resumes_from_the_current_sample_without_jumping() {
		let mut fade = Tween::settled(Color::Rgb(0, 0, 0));
		fade.retarget(
			Duration::ZERO,
			Color::Rgb(200, 200, 200),
			Duration::from_millis(400),
			Easing::Linear,
		);
		let now = Duration::from_millis(200);
		let midway = fade.sample(now);
		assert_eq!(midway, Color::Rgb(100, 100, 100));

		// Interrupt halfway and head back: the sample at the turn is unchanged.
		fade.retarget(now, Color::Rgb(0, 0, 0), Duration::from_millis(400), Easing::Linear);
		assert_eq!(fade.sample(now), midway);
		assert!(!fade.is_settled(Duration::from_millis(599)));
		assert_eq!(fade.sample(Duration::from_millis(600)), Color::Rgb(0, 0, 0));
		assert!(fade.is_settled(Duration::from_millis(600)));
	}

	#[test]
	fn retargeting_the_same_target_keeps_the_running_tween() {
		let mut fade = Tween::settled(0.0f32);
		fade.retarget(Duration::ZERO, 1.0, Duration::from_millis(100), Easing::Linear);
		fade.retarget(Duration::from_millis(50), 1.0, Duration::from_millis(100), Easing::Linear);
		assert_eq!(fade.sample(Duration::from_millis(50)), 0.5);
	}

	#[test]
	fn frame_cycles_wrap_and_predict_the_next_change() {
		let cycle = Frames::new(&["a", "b", "c"], Duration::from_millis(10));
		assert_eq!(cycle.at(Duration::ZERO), "a");
		assert_eq!(cycle.at(Duration::from_millis(19)), "b");
		assert_eq!(cycle.at(Duration::from_millis(35)), "a");
		assert_eq!(cycle.next_change(Duration::from_millis(19)), Duration::from_millis(20));
		assert_eq!(cycle.next_change(Duration::from_millis(20)), Duration::from_millis(30));
	}

	/// Steps the cursor at the frame cadence starting after `from`.
	fn drain(reveal: &mut Reveal, from: Duration, total: usize, horizon: Duration) -> u32 {
		let mut frames = 0;
		while !reveal.is_settled(total) {
			frames += 1;
			assert!(frames < 1000, "reveal never settled");
			reveal.advance(from + FRAME * frames, total, horizon);
		}
		frames
	}

	#[test]
	fn reveal_arms_on_first_sample_then_drains_at_the_floor_rate() {
		let mut reveal = Reveal::new();
		let horizon = Duration::from_millis(250);
		assert_eq!(reveal.advance(Duration::ZERO, 18, horizon), 0, "first sample only arms");
		// Each 33ms frame at the 90 units/s floor earns 2.97 units.
		assert_eq!(reveal.advance(FRAME, 18, horizon), 2);
		assert_eq!(reveal.advance(FRAME * 2, 18, horizon), 5);
		assert_eq!(reveal.advance(FRAME * 3, 18, horizon), 8);
		assert_eq!(drain(&mut reveal, FRAME * 3, 18, horizon), 4);
		assert!(reveal.is_settled(18));
	}

	#[test]
	fn reveal_catches_up_exponentially_then_settles_on_the_floor() {
		let mut reveal = Reveal::new();
		let horizon = Duration::from_millis(250);
		reveal.advance(Duration::ZERO, 1000, horizon);
		// One frame decays the backlog by e^(-FRAME/horizon) ≈ 12%.
		let shown = reveal.advance(FRAME, 1000, horizon);
		assert!((110..=135).contains(&shown), "one frame reveals ~123 units, got {shown}");
		// Exponential decay alone never lands; the floor finishes the tail
		// (~29 catch-up frames plus ~8 floor frames).
		let frames = drain(&mut reveal, FRAME, 1000, horizon);
		assert!((30..=45).contains(&frames), "settled after {frames} more frames");
	}

	#[test]
	fn reveal_never_earns_more_than_one_frame_per_sample() {
		let mut reveal = Reveal::new();
		let horizon = Duration::from_millis(250);
		// Armed on a stale clock, first sampled 400ms later: the gap counts
		// as one frame, not as banked catch-up time.
		reveal.advance(Duration::ZERO, 20, horizon);
		assert_eq!(reveal.advance(Duration::from_millis(400), 20, horizon), 2);

		// A settled cursor idles a minute before the stream appends; the
		// resume also earns at most one frame.
		let mut idle = Reveal::new();
		idle.advance(Duration::ZERO, 3, horizon);
		idle.advance(FRAME, 3, horizon);
		assert_eq!(idle.advance(FRAME * 2, 3, horizon), 3);
		assert!(idle.is_settled(3));
		assert_eq!(idle.advance(Duration::from_secs(60), 40, horizon), 3, "resume only arms");
		let resumed = idle.advance(Duration::from_secs(60) + FRAME, 40, horizon);
		assert!((4..=12).contains(&resumed), "one catch-up frame, not a jump: {resumed}");
	}

	#[test]
	fn reveal_zero_horizon_snaps_and_a_smaller_total_clamps() {
		let mut reveal = Reveal::new();
		assert_eq!(reveal.advance(Duration::ZERO, 12, Duration::ZERO), 12);
		assert_eq!(reveal.advance(Duration::from_secs(1), 5, Duration::from_millis(250)), 5);
		assert!(reveal.is_settled(5));
		reveal.reset();
		assert_eq!(reveal.advance(Duration::from_secs(5), 12, Duration::from_millis(250)), 0);
	}

	#[test]
	fn shimmer_bands_derive_from_an_rgb_foreground() {
		use crate::frame::Style;
		// 200ms into a 1s sweep over a 50-cell track puts the crest on cell 0.
		let shimmer = Shimmer::new(Duration::from_millis(200), Duration::from_secs(1), 30);
		let base = Style::new().fg(Color::Rgb(120, 120, 120));

		// Peak: lifted two-fifths toward white and bold.
		let peak = shimmer.style_at(0, base);
		assert_eq!(peak.foreground_color(), Color::Rgb(174, 174, 174));
		assert!(peak.bold && !peak.dim);
		// Shoulder: lifted one-fifth toward white, attributes untouched.
		let shoulder = shimmer.style_at(3, base);
		assert_eq!(shoulder.foreground_color(), Color::Rgb(147, 147, 147));
		assert!(!shoulder.bold && !shoulder.dim);
		// Rest: shimmer is additive — the authored style passes through.
		assert_eq!(shimmer.style_at(29, base), base);

		// No channel data: only the peak's bold shows, and nothing dims.
		let fallback = Style::new();
		assert!(shimmer.style_at(0, fallback).bold);
		assert_eq!(shimmer.style_at(29, fallback), fallback);
	}
}
