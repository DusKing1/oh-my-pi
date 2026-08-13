//! Window chrome theme derived from the scene's presentation context.

use omp_tui::{Color, UiContext};

/// Straight-alpha RGBA for a cell color, or `fallback` for
/// [`Color::Default`]; `None` means "paint nothing" (transparent backdrop).
pub fn color4(color: Color, fallback: Option<[f32; 4]>) -> Option<[f32; 4]> {
	match color {
		Color::Default => fallback,
		Color::Rgb(red, green, blue) => {
			Some([red as f32 / 255.0, green as f32 / 255.0, blue as f32 / 255.0, 1.0])
		},
		Color::Indexed(index) => Some(xterm256(index)),
	}
}

/// The xterm 256-color palette: 16 system colors, the 6×6×6 cube, grayscale.
pub fn xterm256(index: u8) -> [f32; 4] {
	const SYSTEM: [[u8; 3]; 16] = [
		[0x00, 0x00, 0x00],
		[0xcd, 0x00, 0x00],
		[0x00, 0xcd, 0x00],
		[0xcd, 0xcd, 0x00],
		[0x00, 0x00, 0xee],
		[0xcd, 0x00, 0xcd],
		[0x00, 0xcd, 0xcd],
		[0xe5, 0xe5, 0xe5],
		[0x7f, 0x7f, 0x7f],
		[0xff, 0x00, 0x00],
		[0x00, 0xff, 0x00],
		[0xff, 0xff, 0x00],
		[0x5c, 0x5c, 0xff],
		[0xff, 0x00, 0xff],
		[0x00, 0xff, 0xff],
		[0xff, 0xff, 0xff],
	];
	let [red, green, blue] = match index {
		0..=15 => SYSTEM[index as usize],
		16..=231 => {
			let index = index - 16;
			let cube = |v: u8| if v == 0 { 0 } else { 55 + 40 * v };
			[cube(index / 36), cube(index / 6 % 6), cube(index % 6)]
		},
		_ => {
			let gray = 8 + 10 * (index - 232);
			[gray, gray, gray]
		},
	};
	[red as f32 / 255.0, green as f32 / 255.0, blue as f32 / 255.0, 1.0]
}

/// Chrome colors for the window shell, resolved from the scene's theme.
#[derive(Clone, Copy, Debug)]
pub struct GuiTheme {
	/// Default cell foreground.
	pub fg:            [f32; 4],
	/// Secondary chrome ink (scrollbar track).
	pub muted:         [f32; 4],
	/// Accent (scrollbar thumb, links).
	pub accent:        [f32; 4],
	/// Cell selection overlay: accent RGB at 24% alpha.
	pub selection:     [f32; 4],
	/// Window backdrop; alpha carries the translucency.
	pub backdrop:      [f32; 4],
	/// Text caret.
	pub cursor:        [f32; 4],
	/// Backdrop corner radius, physical px.
	pub corner_radius: f32,
}

impl GuiTheme {
	/// Resolves chrome from the scene context; `opacity` is the backdrop
	/// alpha (0 = fully transparent, 1 = opaque).
	pub fn from_ctx(ctx: &UiContext, opacity: f32) -> Self {
		let theme = ctx.theme;
		let fg = color4(theme.fg, None).unwrap_or([0.78, 0.80, 0.83, 1.0]);
		let accent = color4(theme.accent, None).unwrap_or([0.38, 0.69, 0.94, 1.0]);
		let mut backdrop = color4(theme.contrast, None).unwrap_or([0.06, 0.07, 0.09, 1.0]);
		backdrop[3] = opacity.clamp(0.0, 1.0);
		Self {
			fg,
			muted: color4(theme.muted, None).unwrap_or([0.36, 0.39, 0.44, 1.0]),
			accent,
			backdrop,
			cursor: color4(theme.accent, None).unwrap_or([0.38, 0.69, 0.94, 0.95]),
			selection: [accent[0], accent[1], accent[2], 0.24],
			corner_radius: 12.0,
		}
	}
}
