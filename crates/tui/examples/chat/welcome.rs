//! Welcome card shown while the chat demo "boots".
//!
//! The stippled-eclipse landing shader ([`omp_tui::shader::Eclipse`])
//! resolves out of black as a full-viewport backdrop — the same brand
//! palette as the card, so the corona frames it like a landing page. Over
//! it, a raytraced platter spins up, turns to glass, and settles into the
//! omp brand gradient. The wide layout connects it to a recency rail with
//! a moving lock-on beam and sparse depth field. The scene is traced per
//! frame through [`omp_tui::scene`]'s braille rasterizer, so it animates
//! inside an ordinary text grid. The card shrinks to a narrow
//! variant, then to the bare logo, as the viewport tightens.
//!
//! Hovering the card rides the same primitive the retained pipeline's
//! `hover` prop uses: an [`omp_tui::anim::Tween`] eases a pointer-tracking
//! glow that sweeps the brand gradient along the border.
#![allow(
	clippy::suboptimal_flops,
	reason = "straight port of the omp-welcome-term prototype; mul_add obscures the math"
)]

use std::{
	f32::consts::{PI, TAU},
	time::Duration,
};

use omp_core::{Str, fmts};
use omp_tui::{
	Border, Charset, Color, Frame, Icon, Rect, Size, Style,
	anim::{Easing, Lerp, Tween},
	scene::{self, Camera, Ray, Trace, Vec3, vec3},
	shader::{Eclipse, Surface},
};

// ── card palette (independent of the chat chrome) ──────────────────────────
const CARD_BG: Color = Color::Rgb(11, 14, 21);
const CARD_BORDER: Color = Color::Rgb(43, 51, 68);
const FOOTER_BG: Color = Color::Rgb(7, 10, 15);
const SELECTED_BG: Color = Color::Rgb(13, 21, 27);
const TEXT: Color = Color::Rgb(226, 232, 240);
const TEXT_STRONG: Color = Color::Rgb(248, 250, 252);
const MUTED: Color = Color::Rgb(100, 116, 139);
const FAINT: Color = Color::Rgb(51, 65, 85);
const CYAN: Color = Color::Rgb(56, 189, 248);
const INDIGO: Color = Color::Rgb(129, 140, 248);
const VIOLET: Color = Color::Rgb(192, 132, 252);
const GREEN: Color = Color::Rgb(52, 211, 153);
const AMBER: Color = Color::Rgb(251, 191, 36);

/// Wide card: logo on the left, recent sessions on the right.
const CARD_COLS: u16 = 90;
const CARD_ROWS: u16 = 15;
/// Narrow card: centered logo, condensed key hints.
const SMOL_COLS: u16 = 33;

/// Hover transition length; the same feel as the pipeline's `anim` default.
const HOVER_EASE: Duration = Duration::from_millis(160);
/// Ambient-scene cadence: preserve the boot reveal, then spend less of the
/// input thread on motion that remains legible at a lower rate.
const AMBIENT_REVEAL_INTERVAL: Duration = Duration::from_millis(40);
const LOGO_IDLE_INTERVAL: Duration = Duration::from_millis(120);
const BACKDROP_IDLE_INTERVAL: Duration = Duration::from_millis(80);
const AMBIENT_REVEAL_END: Duration = Duration::from_millis(1_200);
/// The eclipse's plate tone behind the bare logo, so braille composites
/// over the backdrop instead of punching a default-background hole.
const PLATE: Color = Color::Rgb(9, 9, 11);
/// Backdrop fade-in horizon (seconds): the eclipse resolves out of black
/// while the platter does its reveal orbit.
const BACKDROP_FADE: f32 = 1.2;

fn ambient_interval(clock: Duration, idle: Duration) -> Duration {
	if clock < AMBIENT_REVEAL_END {
		AMBIENT_REVEAL_INTERVAL
	} else {
		idle
	}
}

const SESSIONS: [(&str, &str); 4] = [
	("Optimize custom status widget rendering", "NOW"),
	("Check Unicode character display", "01m"),
	("Add ╰─ cursor shift", "02m"),
	("The Unicode Similarity", "18m"),
];

const DUST: [(u16, u16, f32); 9] = [
	(3, 3, 0.1),
	(6, 9, 1.7),
	(9, 2, 2.5),
	(12, 10, 4.4),
	(18, 2, 5.2),
	(21, 11, 3.1),
	(25, 3, 1.2),
	(28, 9, 4.8),
	(30, 6, 0.8),
];
const BEAM: [(u16, u16, &str); 8] = [
	(27, 7, "╱"),
	(28, 6, "╱"),
	(29, 6, "─"),
	(30, 5, "╱"),
	(31, 5, "─"),
	(32, 4, "╱"),
	(33, 4, "━"),
	(34, 4, "━"),
];
const HORIZON: &str = "· · · · · · · · · · · · · · · ";

// ── scene constants ─────────────────────────────────────────────────────────
const LOGO_COLS: usize = 25;
const LOGO_ROWS: usize = 10;

const DISK_RADIUS: f32 = 0.82;
const DISK_Y: f32 = 0.12;
const DISK_HALF_THICKNESS: f32 = 0.045;
const DISK_GLASS_OPACITY: f32 = 0.56;
const AXIS_RADIUS: f32 = 0.047;
const AXIS_BOTTOM: f32 = -0.78;
const AXIS_TOP: f32 = 1.14;
const FLOOR_Y: f32 = -0.84;

/// The prototype's orbit (xz radius 3.78, eye 3.78·sin 28° above target),
/// restated as the spherical orbit [`Camera`] expects.
const CAMERA_DISTANCE: f32 = 4.176;
const CAMERA_PITCH: f32 = 0.4391;
const CAMERA_FOCAL: f32 = 2.72;

/// Brand gradient around the disk: cyan → indigo → violet → cyan.
const DISK_STOPS: [Vec3; 4] = [
	Vec3::rgb(56, 189, 248),
	Vec3::rgb(129, 140, 248),
	Vec3::rgb(192, 132, 252),
	Vec3::rgb(56, 189, 248),
];
const BACKGROUND: Vec3 = Vec3::rgb(11, 14, 21);
const WHITE: Vec3 = Vec3::rgb(248, 250, 252);
const INK: Vec3 = Vec3::rgb(3, 4, 7);

type LogoGrid = [[Option<(char, Color)>; LOGO_COLS]; LOGO_ROWS];

/// The animated welcome screen: a retained full-viewport frame, the
/// pre-built title chip, and the pointer-orbit camera. [`crate::run_welcome`]
/// drives it until the user resumes into the chat demo.
pub struct Welcome {
	frame:          Frame,
	title:          Str,
	/// Detected glyph tier for the card chrome.
	charset:        Charset,
	camera:         (f32, f32),
	camera_target:  (f32, f32),
	last_elapsed:   f32,
	/// Top-left cell of the logo as last drawn; anchors pointer mapping.
	logo_origin:    (u16, u16),
	logo:           LogoGrid,
	logo_at:        Option<Duration>,
	backdrop_frame: Frame,
	backdrop_at:    Option<Duration>,
	/// Eclipse backdrop program and its reusable half-block render target.
	backdrop:       Eclipse,
	surface:        Surface,
	/// Last pointer cell reported by the host loop.
	pointer:        Option<(u16, u16)>,
	/// Eased hover amount driving the border glow.
	hover:          Tween<f32>,
}

impl Welcome {
	pub fn new(charset: Charset) -> Self {
		Self {
			charset,
			frame: Frame::new(Size::new(0, 0)),
			title: fmts!(" {} omp v{} ", charset.icon(Icon::Omp), env!("CARGO_PKG_VERSION")),
			camera: (0.0, 0.0),
			camera_target: (0.0, 0.0),
			last_elapsed: 0.0,
			logo_origin: (0, 0),
			logo: [[None; LOGO_COLS]; LOGO_ROWS],
			logo_at: None,
			backdrop_frame: Frame::new(Size::new(0, 0)),
			backdrop_at: None,
			backdrop: Eclipse::default(),
			surface: Surface::new(),
			pointer: None,
			hover: Tween::settled(0.0),
		}
	}

	/// Records the pointer (0-based cells) for the hover zone and retargets
	/// the camera: the pointer's offset from the logo center maps to camera
	/// lift and a full half-turn of yaw in each direction, matching the
	/// prototype.
	pub fn point_at(&mut self, column: u16, row: u16) {
		self.pointer = Some((column, row));
		self.logo_at = None;
		let center_x = f32::from(self.logo_origin.0) + LOGO_COLS as f32 / 2.0;
		let center_y = f32::from(self.logo_origin.1) + LOGO_ROWS as f32 / 2.0;
		let horizontal = ((f32::from(column) - center_x) / (LOGO_COLS as f32 / 2.0)).clamp(-1.0, 1.0);
		let vertical = ((f32::from(row) - center_y) / (LOGO_ROWS as f32 / 2.0)).clamp(-1.0, 1.0);
		self.camera_target = (-vertical * 0.42, -horizontal * PI);
	}

	/// Paints the card centered in `viewport` at `elapsed` since boot and
	/// returns the full-viewport frame (no stable rows, everything damaged).
	pub fn render(&mut self, viewport: Size, elapsed: Duration) -> &Frame {
		if self.frame.size() != viewport {
			self.frame = Frame::new(viewport);
			self.backdrop_frame = Frame::new(viewport);
			self.backdrop_at = None;
		}
		let clock = elapsed;
		let elapsed = elapsed.as_secs_f32();
		// Exponential pointer chase, frame-rate independent (~100ms lag).
		let delta = (elapsed - self.last_elapsed).max(0.0);
		self.last_elapsed = elapsed;
		let response = 1.0 - (-delta * 10.0).exp();
		self.camera.0 += (self.camera_target.0 - self.camera.0) * response;
		self.camera.1 += (self.camera_target.1 - self.camera.1) * response;
		self.draw_backdrop(viewport, clock, elapsed);

		let logo_interval = ambient_interval(clock, LOGO_IDLE_INTERVAL);
		if self
			.logo_at
			.is_none_or(|rendered_at| clock.saturating_sub(rendered_at) >= logo_interval)
		{
			self.logo = logo_cells(elapsed, self.camera);
			self.logo_at = Some(clock);
		}
		let cols = if viewport.width >= CARD_COLS && viewport.height >= CARD_ROWS {
			Some(CARD_COLS)
		} else if viewport.width >= SMOL_COLS && viewport.height >= CARD_ROWS {
			Some(SMOL_COLS)
		} else {
			None
		};
		let Some(cols) = cols else {
			let left = viewport.width.saturating_sub(LOGO_COLS as u16) / 2;
			let top = viewport.height.saturating_sub(LOGO_ROWS as u16) / 2;
			self.logo_origin = (left, top);
			blit_logo(&mut self.frame, &self.logo, left, top, PLATE);
			return &self.frame;
		};

		let left = (viewport.width - cols) / 2;
		let top = (viewport.height - CARD_ROWS) / 2;
		let hovered = self.pointer.is_some_and(|(x, y)| {
			(left..left + cols).contains(&x) && (top..top + CARD_ROWS).contains(&y)
		});
		self
			.hover
			.retarget(clock, if hovered { 1.0 } else { 0.0 }, HOVER_EASE, Easing::EaseOut);
		let hover = self.hover.sample(clock).clamp(0.0, 1.0);
		self.draw_card(cols, left, top, elapsed, hover);
		&self.frame
	}

	/// Paints the eclipse across the whole viewport, resolving out of
	/// black over the first [`BACKDROP_FADE`] seconds of boot.
	fn draw_backdrop(&mut self, viewport: Size, clock: Duration, elapsed: f32) {
		let interval = ambient_interval(clock, BACKDROP_IDLE_INTERVAL);
		if self
			.backdrop_at
			.is_none_or(|rendered_at| clock.saturating_sub(rendered_at) >= interval)
		{
			let fade = smooth((elapsed / BACKDROP_FADE).clamp(0.0, 1.0));
			self
				.backdrop_frame
				.fill(Rect::new(0, 0, viewport.width, viewport.height), Style::default());
			let frame = &mut self.backdrop_frame;
			let mut buffer = [0_u8; 4];
			let dim = |color: Color| Color::Rgb(0, 0, 0).lerp(color, fade);
			self.surface.render(
				&mut self.backdrop,
				clock,
				viewport.width,
				viewport.height,
				|x, y, glyph, fg, bg| {
					let style = Style::new().fg(dim(fg));
					let style = match bg {
						Some(bg) => style.bg(dim(bg)),
						None => style,
					};
					frame.put(x, y, glyph.encode_utf8(&mut buffer), style);
				},
			);
			self.backdrop_at = Some(clock);
		}
		self.frame.clone_from(&self.backdrop_frame);
	}

	fn draw_card(&mut self, cols: u16, left: u16, top: u16, elapsed: f32, hover: f32) {
		let full = cols == CARD_COLS;
		let logo_left = if full {
			left + 3
		} else {
			left + (cols - LOGO_COLS as u16) / 2
		};
		self.logo_origin = (logo_left, top + 2);
		// Pointer-tracking border glow: the brand gradient sampled by angle
		// around the card center (the disk's own palette), strongest near
		// the pointer, scaled by the eased hover amount.
		let pointer = self.pointer;
		let center =
			(f32::from(left) + f32::from(cols) / 2.0, f32::from(top) + f32::from(CARD_ROWS) / 2.0);
		let edge_at = move |x: u16, y: u16| -> Style {
			let Some((px, py)) = pointer.filter(|_| hover > 0.02) else {
				return on_card(CARD_BORDER);
			};
			let dx = (f32::from(x) - f32::from(px)) * 0.5;
			let dy = f32::from(y) - f32::from(py);
			let glow = hover * (-(dx * dx + dy * dy) / 34.0).exp();
			if glow < 0.02 {
				return on_card(CARD_BORDER);
			}
			let angle = (f32::from(y) - center.1).atan2((f32::from(x) - center.0) * 0.5);
			let brand = vec3_color(gradient(angle - elapsed * 0.5));
			on_card(CARD_BORDER.lerp(brand, glow))
		};
		let frame = &mut self.frame;
		frame.fill(Rect::new(left, top, cols, CARD_ROWS), on_card(TEXT));

		let right = left + cols - 1;
		let bottom = top + CARD_ROWS - 1;
		let divider = bottom - 2;
		let (tl, tr, bl, br, horizontal, vertical) = self.charset.border(Border::Round);
		let grid = self.charset.grid();
		let mut glyph = [0_u8; 4];
		frame.put(left, top, tl.encode_utf8(&mut glyph), edge_at(left, top));
		frame.put(right, top, tr.encode_utf8(&mut glyph), edge_at(right, top));
		frame.put(left, divider, grid.middle.0.encode_utf8(&mut glyph), edge_at(left, divider));
		frame.put(right, divider, grid.middle.2.encode_utf8(&mut glyph), edge_at(right, divider));
		frame.put(left, bottom, bl.encode_utf8(&mut glyph), edge_at(left, bottom));
		frame.put(right, bottom, br.encode_utf8(&mut glyph), edge_at(right, bottom));
		for x in left + 1..right {
			frame.put(x, top, horizontal.encode_utf8(&mut glyph), edge_at(x, top));
			frame.put(x, divider, horizontal.encode_utf8(&mut glyph), edge_at(x, divider));
			frame.put(x, bottom, horizontal.encode_utf8(&mut glyph), edge_at(x, bottom));
		}
		for y in top + 1..bottom {
			if y != divider {
				frame.put(left, y, vertical.encode_utf8(&mut glyph), edge_at(left, y));
				frame.put(right, y, vertical.encode_utf8(&mut glyph), edge_at(right, y));
			}
		}

		frame.put(left + 2, top, self.title.as_str(), on_card(TEXT_STRONG));
		if full {
			frame.put(left + 34, top, " SESSION ORBIT ", on_card(FAINT));
			draw_dust(frame, left, top, elapsed);
			draw_sessions(frame, left, top, self.charset);
			draw_beam(frame, left, top, elapsed);
		}

		blit_logo(frame, &self.logo, logo_left, top + 2, CARD_BG);

		let footer = divider + 1;
		frame.fill(Rect::new(left + 1, footer, cols - 2, 1), on_footer(TEXT));
		if full {
			frame.put(left + 3, divider, " SHORTCUTS ", on_card(FAINT));
			let dot = fmts!(" {} ", self.charset.icon(Icon::Enabled));
			let x = frame.put(left + cols - 21, top, &dot, on_card(GREEN));
			frame.put(x, top, "rust-analyzer ", on_card(MUTED));
			draw_full_hints(frame, left, footer);
		} else {
			draw_smol_hints(frame, left, cols, footer);
		}
	}
}

impl Default for Welcome {
	fn default() -> Self {
		Self::new(Charset::NerdFont)
	}
}

fn draw_dust(frame: &mut Frame, left: u16, top: u16, elapsed: f32) {
	for &(x, y, offset) in &DUST {
		let pulse = 0.5 + 0.5 * (elapsed * 1.4 + offset).sin();
		let color = FAINT.lerp(CYAN, pulse * 0.28);
		frame.put(left + x, top + y, "·", on_card(color));
	}
	frame.put(left + 1, top + 7, HORIZON, on_card(FAINT.lerp(INDIGO, 0.16)));
	frame.put(left + 14, top + 1, "+Z", on_card(FAINT));
}

fn draw_beam(frame: &mut Frame, left: u16, top: u16, elapsed: f32) {
	let phase = (elapsed * 9.0) as usize % BEAM.len();
	for (index, &(x, y, glyph)) in BEAM.iter().enumerate() {
		let direct = index.abs_diff(phase);
		let distance = direct.min(BEAM.len() - direct);
		let color = match distance {
			0 => TEXT_STRONG,
			1 => CYAN,
			_ => FAINT.lerp(INDIGO, 0.34),
		};
		frame.put(left + x, top + y, glyph, on_card(color));
	}
}

fn draw_sessions(frame: &mut Frame, left: u16, top: u16, charset: Charset) {
	let (_, _, _, _, _, vertical) = charset.border(Border::Round);
	let mut glyph = [0_u8; 4];
	let panel_x = left + 36;
	frame.put(panel_x, top + 2, "RECENT SESSIONS", on_card(MUTED));
	frame.put(left + CARD_COLS - 14, top + 2, "4 / LOCAL", on_card(FAINT));
	for y in top + 4..=top + 10 {
		frame.put(panel_x, y, vertical.encode_utf8(&mut glyph), on_card(FAINT.lerp(INDIGO, 0.18)));
	}
	for (index, (label, age)) in SESSIONS.iter().enumerate() {
		let y = top + 4 + index as u16 * 2;
		if index == 0 {
			frame.fill(Rect::new(panel_x - 2, y, CARD_COLS - 35, 1), on_selected(TEXT));
			frame.put(panel_x - 2, y, charset.rail(), on_selected(GREEN));
			frame.put(panel_x, y, charset.radio(true), on_selected(GREEN));
			frame.put(panel_x + 2, y, age, on_selected(GREEN));
			frame.put(panel_x + 7, y, label, on_selected(TEXT_STRONG));
		} else {
			frame.put(panel_x, y, charset.radio(false), on_card(FAINT));
			frame.put(panel_x + 2, y, age, on_card(FAINT));
			frame.put(panel_x + 7, y, label, on_card(MUTED));
		}
	}
}

fn draw_full_hints(frame: &mut Frame, left: u16, y: u16) {
	frame.put(left + 3, y, "#", on_footer(CYAN));
	frame.put(left + 5, y, "actions", on_footer(MUTED));
	frame.put(left + 14, y, "/", on_footer(GREEN));
	frame.put(left + 16, y, "commands", on_footer(MUTED));
	frame.put(left + 27, y, "!", on_footer(AMBER));
	frame.put(left + 29, y, "shell", on_footer(MUTED));
	frame.put(left + 37, y, "$", on_footer(VIOLET));
	frame.put(left + 39, y, "python", on_footer(MUTED));
	frame.put(left + CARD_COLS - 23, y, "↑↓ move", on_footer(FAINT));
	frame.put(left + CARD_COLS - 13, y, "↵ resume", on_footer(TEXT_STRONG));
}

fn draw_smol_hints(frame: &mut Frame, left: u16, cols: u16, y: u16) {
	frame.put(left + 3, y, "#", on_footer(CYAN).bold());
	frame.put(left + 5, y, "/", on_footer(CYAN).bold());
	frame.put(left + 7, y, "!", on_footer(AMBER).bold());
	frame.put(left + 9, y, "$", on_footer(GREEN).bold());
	frame.put(left + cols - 14, y, "enter", on_footer(FAINT));
	frame.put(left + cols - 8, y, "resume", on_footer(TEXT_STRONG).bold());
}

fn blit_logo(frame: &mut Frame, logo: &LogoGrid, left: u16, top: u16, background: Color) {
	let mut buffer = [0_u8; 4];
	for (row, cells) in logo.iter().enumerate() {
		for (column, cell) in cells.iter().enumerate() {
			let Some((glyph, color)) = cell else { continue };
			let style = Style::new().fg(*color).bg(background);
			frame.put(left + column as u16, top + row as u16, glyph.encode_utf8(&mut buffer), style);
		}
	}
}

const fn on_card(fg: Color) -> Style {
	Style::new().fg(fg).bg(CARD_BG)
}

const fn on_footer(fg: Color) -> Style {
	Style::new().fg(fg).bg(FOOTER_BG)
}

const fn on_selected(fg: Color) -> Style {
	Style::new().fg(fg).bg(SELECTED_BG)
}

// ── raytraced logo ───────────────────────────────────────────────────────────

fn smooth(edge: f32) -> f32 {
	edge * edge * (3.0 - 2.0 * edge)
}

fn ease_in(progress: f32) -> f32 {
	let clamped = progress.clamp(0.0, 1.0);
	clamped * clamped
}

/// Opaque platter → glass blend, driven by boot time.
fn color_mix(elapsed: f32) -> f32 {
	ease_in((elapsed - 0.18) / 0.58)
}

/// One decaying full camera orbit as the logo reveals.
fn reveal_orbit(elapsed: f32) -> f32 {
	let progress = ((elapsed - 0.18) / 0.64).clamp(0.0, 1.0);
	TAU * (1.0 - (1.0 - progress).powi(3))
}

/// Disk spin: quadratic ramp into a constant angular speed.
fn disk_rotation(elapsed: f32) -> f32 {
	let spinning = (elapsed - 0.18).max(0.0);
	let ramp = 0.40;
	let angular_speed = TAU / 2.4;
	if spinning < ramp {
		angular_speed * spinning * spinning / (2.0 * ramp)
	} else {
		angular_speed * (spinning - ramp / 2.0)
	}
}

/// Samples the brand gradient by angle around the disk.
fn gradient(angle: f32) -> Vec3 {
	let position = (angle / TAU + 0.5).rem_euclid(1.0) * 3.0;
	let index = (position as usize).min(2);
	DISK_STOPS[index].lerp(DISK_STOPS[index + 1], position - index as f32)
}

/// Lowers a unit-range color vector to a terminal cell color.
fn vec3_color(color: Vec3) -> Color {
	let channel = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
	Color::Rgb(channel(color.x), channel(color.y), channel(color.z))
}

/// Three soft light shafts crossing the stage in the sun's plane.
fn sun_rays(x: f32, z: f32) -> f32 {
	let length = 0.55_f32.hypot(0.38);
	let (hx, hz) = (-0.55 / length, 0.38 / length);
	let across = x * -hz + z * hx;
	let along = x * hx + z * hz;
	let rays = (-((across + 0.56) / 0.075).powi(2)).exp()
		+ (-((across + 0.04) / 0.055).powi(2)).exp()
		+ (-((across - 0.47) / 0.09).powi(2)).exp();
	let envelope = (-((along + 0.20) / 2.45).powi(4)).exp();
	(rays * envelope).clamp(0.0, 1.0)
}

fn sphere_depth(origin: Vec3, direction: Vec3, center: Vec3, radius: f32) -> f32 {
	let offset = origin - center;
	let projection = offset.dot(direction);
	let discriminant = projection * projection - (offset.dot(offset) - radius * radius);
	let root = -projection - discriminant.max(0.0).sqrt();
	if discriminant >= 0.0 && root > 0.0 {
		root
	} else {
		f32::INFINITY
	}
}

/// Central axis: a capped cylinder with a slightly bulged top sphere.
fn axis_hit(origin: Vec3, direction: Vec3) -> (f32, bool, Vec3) {
	let quadratic = direction.x * direction.x + direction.z * direction.z;
	let linear = 2.0 * (origin.x * direction.x + origin.z * direction.z);
	let constant = origin.x * origin.x + origin.z * origin.z - AXIS_RADIUS * AXIS_RADIUS;
	let discriminant = linear * linear - 4.0 * quadratic * constant;
	let root = (-linear - discriminant.max(0.0).sqrt()) / (2.0 * quadratic).max(1e-8);
	let hit_y = origin.y + direction.y * root;
	let cylinder = if discriminant >= 0.0 && root > 0.0 && (AXIS_BOTTOM..=AXIS_TOP).contains(&hit_y)
	{
		root
	} else {
		f32::INFINITY
	};
	let cap_center = vec3(0.0, AXIS_TOP, 0.0);
	let cap = sphere_depth(origin, direction, cap_center, AXIS_RADIUS * 1.75);
	let depth = cylinder.min(cap);
	if !depth.is_finite() {
		return (f32::INFINITY, false, vec3(0.0, 1.0, 0.0));
	}
	let point = origin + direction * depth;
	let normal = if cap < cylinder {
		(point - cap_center).normalize()
	} else {
		vec3(point.x, 0.0, point.z).normalize()
	};
	(depth, true, normal)
}

/// Terminal display lift: brightens traced colors so braille reads on a
/// dark card (the prototype's channel curve, applied per sample — the
/// rasterizer's coverage-weighted averaging commutes with an affine map).
fn tone(color: Vec3) -> Vec3 {
	color * 1.28 + vec3(4.0 / 255.0, 4.0 / 255.0, 4.0 / 255.0)
}

/// Per-frame scene state shared by every ray: sun direction, disk spin,
/// glass transition, and the pointer camera offsets.
struct Platter {
	sun:        Vec3,
	angle:      f32,
	transition: f32,
	/// Smoothed pointer camera from [`Welcome`]: (lift, yaw).
	pointer:    (f32, f32),
}

impl Platter {
	fn new(pointer: (f32, f32)) -> Self {
		Self { sun: vec3(-0.55, 1.0, 0.38).normalize(), angle: 0.0, transition: 0.0, pointer }
	}

	/// Floor glow: light shafts, the disk's shadow, tinted transmission
	/// through the glass, and a soft rim reflection.
	fn ground(&self, origin: Vec3, direction: Vec3) -> (Vec3, f32) {
		if direction.y >= 0.0 {
			return (BACKGROUND, 0.0);
		}
		let depth = (FLOOR_Y - origin.y) / direction.y;
		if depth <= 0.0 {
			return (BACKGROUND, 0.0);
		}
		let floor = origin + direction * depth;
		let stage =
			(-0.42 * ((floor.x.abs() / 2.30).powi(4) + ((floor.z + 0.15).abs() / 1.70).powi(4))).exp();

		let sun_depth = (DISK_Y - DISK_HALF_THICKNESS - FLOOR_Y) / self.sun.y;
		let sunlit = floor + self.sun * sun_depth;
		let shadow_radius = sunlit.x.hypot(sunlit.z);
		let occlusion = smooth(((DISK_RADIUS + 0.08 - shadow_radius) / 0.16).clamp(0.0, 1.0));
		let rays = sun_rays(floor.x, floor.z);

		let neutral_alpha = stage * rays * (1.0 - occlusion) * 0.36;
		let neutral = (BACKGROUND + vec3(0.36, 0.33, 0.27) * (stage * rays)).clamp01();
		let mut color = neutral * neutral_alpha + BACKGROUND * (1.0 - neutral_alpha);
		let mut alpha = neutral_alpha;

		let transmission_alpha =
			stage * sun_rays(sunlit.x, sunlit.z) * occlusion * self.transition * 0.64;
		let transmission =
			(BACKGROUND + gradient(sunlit.z.atan2(sunlit.x) - self.angle) * 0.86).clamp01();
		color = transmission * transmission_alpha + color * (1.0 - transmission_alpha);
		alpha = transmission_alpha + alpha * (1.0 - transmission_alpha);

		let mirrored = vec3(direction.x, -direction.y, direction.z);
		let reflected_depth = (DISK_Y - DISK_HALF_THICKNESS - FLOOR_Y) / mirrored.y.max(1e-6);
		let reflected = floor + mirrored * reflected_depth;
		let reflected_radius = reflected.x.hypot(reflected.z);
		if reflected_radius <= DISK_RADIUS {
			let edge = smooth(((DISK_RADIUS - reflected_radius) / 0.13).clamp(0.0, 1.0));
			let grazing = (1.0 + direction.y).clamp(0.0, 1.0);
			let reflection_alpha = self.transition * edge * (0.22 + 0.28 * grazing);
			let reflection = gradient(reflected.z.atan2(reflected.x) - self.angle);
			color = reflection * reflection_alpha + color * (1.0 - reflection_alpha);
			alpha = reflection_alpha + alpha * (1.0 - reflection_alpha);
		}
		(color, alpha)
	}
}

impl Trace for Platter {
	fn advance(&mut self, now: Duration) -> Camera {
		let elapsed = now.as_secs_f32();
		self.angle = disk_rotation(elapsed);
		self.transition = color_mix(elapsed);
		Camera {
			target:   vec3(0.0, 0.08, 0.0),
			yaw:      self.pointer.1 + reveal_orbit(elapsed) + (elapsed * 0.31).sin() * 0.018,
			pitch:    CAMERA_PITCH,
			distance: CAMERA_DISTANCE,
			lift:     self.pointer.0.clamp(-0.42, 0.42) + (elapsed * 0.55).sin() * 0.014,
			focal:    CAMERA_FOCAL,
		}
	}

	/// Shades one sample: returns the color composited over the card
	/// background and the coverage used for braille dot thresholds.
	fn shade(&self, ray: Ray) -> (Vec3, f32) {
		let Ray { origin, dir: direction } = ray;

		// Top surface of the disk.
		let mut disk_depth = f32::INFINITY;
		let mut disk_point = Vec3::ZERO;
		let mut disk_radial = 0.0;
		if direction.y < 0.0 {
			let depth = (DISK_Y + DISK_HALF_THICKNESS - origin.y) / direction.y;
			if depth > 0.0 {
				let point = origin + direction * depth;
				let radial = point.x.hypot(point.z);
				if radial <= DISK_RADIUS {
					disk_depth = depth;
					disk_point = point;
					disk_radial = radial;
				}
			}
		}
		let disk_visible = disk_depth.is_finite();
		let (axis_depth, axis_visible, axis_normal) = axis_hit(origin, direction);

		let (ground_color, ground_alpha) = self.ground(origin, direction);
		let mut color = ground_color * ground_alpha + BACKGROUND * (1.0 - ground_alpha);
		let mut alpha = ground_alpha;

		let view = direction * -1.0;
		let halfway = if disk_visible || axis_visible {
			(self.sun + view).normalize()
		} else {
			Vec3::ZERO
		};
		let axis_color = if axis_visible {
			let diffuse = axis_normal.dot(self.sun).max(0.0);
			let specular = axis_normal.dot(halfway).max(0.0).powi(44);
			(WHITE * (0.32 + 0.68 * diffuse + 0.48 * specular)).clamp01()
		} else {
			Vec3::ZERO
		};

		if axis_visible && axis_depth > disk_depth {
			color = axis_color;
			alpha = 1.0;
		}

		if disk_visible {
			let diffuse = self.sun.y.max(0.0);
			let specular = halfway.y.max(0.0).powi(72);
			let fresnel = (1.0 - view.y.max(0.0)).powi(4);
			let material_angle = disk_point.z.atan2(disk_point.x) - self.angle;
			let rim = smooth(((disk_radial - (DISK_RADIUS - 0.075)) / 0.055).clamp(0.0, 1.0));
			let streak_angle = (material_angle - 0.32 + PI).rem_euclid(TAU) - PI;
			let streak = (-(streak_angle / 0.19).powi(2)).exp();
			let incident = sun_rays(disk_point.x, disk_point.z);

			let opaque =
				(WHITE * (0.30 + 0.27 * diffuse + 0.36 * incident + 0.20 * specular)).clamp01();
			let glass = (gradient(material_angle) * (0.34 + 0.70 * diffuse)
				+ WHITE * (0.72 * specular + 0.16 * fresnel + 0.20 * streak))
				.clamp01();
			let border_strength = rim * self.transition;
			let mut disk_color = opaque
				.lerp(glass, self.transition)
				.lerp(WHITE, border_strength);

			// Orbiting index marker punched into the surface.
			let marker_x = 0.52 * self.angle.cos();
			let marker_z = 0.52 * self.angle.sin();
			let marker_distance = (disk_point.x - marker_x).hypot(disk_point.z - marker_z);
			let marker = smooth(((0.10 - marker_distance) / 0.035).clamp(0.0, 1.0));
			disk_color = disk_color.lerp(INK, marker);

			let disk_alpha = (1.0 - self.transition * (1.0 - DISK_GLASS_OPACITY))
				.max(border_strength * 0.96)
				.max(marker * 0.98);
			color = disk_color * disk_alpha + color * (1.0 - disk_alpha);
			alpha = disk_alpha + alpha * (1.0 - disk_alpha);
		}

		if axis_visible && axis_depth <= disk_depth {
			color = axis_color;
			alpha = 1.0;
		}
		(tone(color), alpha)
	}
}

/// Renders the scene at `elapsed` under the pointer camera and packs it
/// into braille cells through [`scene::rasterize`].
fn logo_cells(elapsed: f32, pointer: (f32, f32)) -> LogoGrid {
	let mut platter = Platter::new(pointer);
	let camera = platter.advance(Duration::from_secs_f32(elapsed));
	let mut grid: LogoGrid = [[None; LOGO_COLS]; LOGO_ROWS];
	scene::rasterize(&platter, &camera, LOGO_COLS as u16, LOGO_ROWS as u16, |x, y, glyph, color| {
		grid[y as usize][x as usize] = Some((glyph, color));
	});
	grid
}

#[cfg(test)]
mod tests {
	use omp_tui::test_support::{frame_cell_style, frame_row_text};

	use super::{CARD_BORDER, CARD_ROWS, Charset, Duration, Size, Welcome};

	fn rows(viewport: Size, elapsed_ms: u64) -> Vec<String> {
		let mut welcome = Welcome::new(Charset::NerdFont);
		let frame = welcome.render(viewport, Duration::from_millis(elapsed_ms));
		(0..frame.size().height)
			.map(|row| frame_row_text(frame, row))
			.collect()
	}

	fn has_lit_braille(rows: &[String]) -> bool {
		rows.iter().any(|row| {
			row.chars()
				.any(|glyph| ('\u{2801}'..='\u{28ff}').contains(&glyph))
		})
	}

	#[test]
	fn full_card_centers_sessions_and_hints() {
		let rows = rows(Size::new(100, 21), 2_000);
		// Card spans rows 3..=17 when centered in a 100x21 viewport.
		assert!(rows[3].contains("omp v"), "title chip in the top border: {}", rows[3]);
		assert!(rows[3].contains("rust-analyzer"), "LSP chip in the top border");
		assert!(rows[5].contains("RECENT SESSIONS") && rows[5].contains("4 / LOCAL"));
		assert!(rows[7].contains("Optimize custom status widget rendering"));
		assert!(rows[15].contains("SHORTCUTS"), "divider labels the command rail");
		assert!(rows[16].contains("actions") && rows[16].contains("resume"));
		assert!(rows[17].contains('╰'), "bottom border closes the card");
		assert!(has_lit_braille(&rows), "the disk renders lit braille cells");
	}

	#[test]
	fn smol_card_drops_the_session_panel() {
		let rows = rows(Size::new(40, CARD_ROWS), 2_000);
		assert!(rows[0].contains("omp v"));
		assert!(rows.iter().all(|row| !row.contains("RECENT SESSIONS")));
		assert!(rows[13].contains("resume"));
		assert!(has_lit_braille(&rows));
	}

	#[test]
	fn bare_logo_survives_a_tiny_viewport() {
		let rows = rows(Size::new(29, 11), 2_000);
		assert!(has_lit_braille(&rows), "the logo still renders without the card");
		assert!(rows.iter().all(|row| !row.contains('╭')), "no card chrome at this size");
	}

	#[test]
	fn eclipse_backdrop_fills_the_margins() {
		let rows = rows(Size::new(100, 21), 2_000);
		// Card spans rows 3..=17; everything outside is half-block pixels.
		assert!(rows[0].chars().all(|glyph| glyph == '▀'), "top margin is backdrop: {}", rows[0]);
		assert!(rows[20].chars().all(|glyph| glyph == '▀'), "bottom margin is backdrop");
	}

	#[test]
	fn pointer_orbit_changes_the_rendered_logo() {
		let viewport = Size::new(100, 21);
		let settle = Duration::from_millis(1_000);
		let sample = Duration::from_millis(2_000);

		let mut centered = Welcome::new(Charset::NerdFont);
		centered.render(viewport, settle);
		let baseline: Vec<String> = {
			let frame = centered.render(viewport, sample);
			(0..frame.size().height)
				.map(|row| frame_row_text(frame, row))
				.collect()
		};

		let mut orbited = Welcome::new(Charset::NerdFont);
		// First render fixes the logo origin the pointer maps against; the
		// second, a full second later, lets the exponential chase converge.
		orbited.render(viewport, settle);
		orbited.point_at(0, 0);
		let frame = orbited.render(viewport, sample);
		let pointed: Vec<String> = (0..frame.size().height)
			.map(|row| frame_row_text(frame, row))
			.collect();

		assert_ne!(baseline, pointed, "pointing away from the logo center orbits the camera");
	}

	#[test]
	fn hovering_the_card_glows_the_border_near_the_pointer() {
		let viewport = Size::new(100, 21);
		let mut welcome = Welcome::new(Charset::NerdFont);
		welcome.render(viewport, Duration::from_millis(1_000));

		// Hover just under a bare stretch of the top border (clear of the
		// title and panel labels): one frame starts the tween, a later
		// frame samples it settled.
		welcome.point_at(27, 4);
		welcome.render(viewport, Duration::from_millis(1_050));
		let frame = welcome.render(viewport, Duration::from_millis(2_000));
		let near_pointer = frame_cell_style(frame, 27, 3).foreground_color();
		let far_border = frame_cell_style(frame, 8, 17).foreground_color();
		assert!(frame_row_text(frame, 3).contains('╭'), "the card stays on row 3 while hovered");
		assert_ne!(near_pointer, CARD_BORDER, "border glows near the pointer");
		assert_eq!(far_border, CARD_BORDER, "the glow stays local to the pointer");
	}
}
