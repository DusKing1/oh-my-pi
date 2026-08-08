//! CSS color parsing: every color form from CSS Color Module Level 4,
//! plus `hsv()`/`hsva()` as a non-CSS convenience.
//!
//! Parsing is context-free and lossless: [`CssColor`] preserves alpha,
//! `currentcolor`, and system-color keywords instead of guessing what an
//! opaque cell should show. Lowering to a terminal [`Color`] is a
//! separate, explicit step — [`CssColor::resolve`] with only a theme in
//! hand, [`CssColor::composite`] when the backdrop and current
//! foreground are known, or [`Color::parse`] as the documented
//! context-free shorthand.
//!
//! Each submodule owns one conversion family; this module owns the
//! grammar: function dispatch, component tokenizing ([`Components`]),
//! and the shared value readers (`<number>`, `<percentage>`, `<hue>`,
//! `<alpha-value>`).
//!
//! Supported forms, all case-insensitive:
//! - named colors, `transparent`, `currentcolor`, and the CSS system colors
//!   (current keywords plus the deprecated aliases)
//! - `#rgb`/`#rgba`/`#rrggbb`/`#rrggbbaa`
//! - `rgb()`/`rgba()`, `hsl()`/`hsla()`, `hwb()`, `hsv()`/`hsva()`
//! - `lab()`/`lch()`, `oklab()`/`oklch()`
//! - `color()` with the predefined RGB and XYZ color spaces
//!
//! Conversion follows the CSS Color 4 algorithms, including `OKLCh`
//! chroma-reduction gamut mapping (§13.2) for results outside sRGB.
//!
//! Deliberate deviations from the spec:
//! - Legacy comma syntax and modern features (`none`, mixed number/percentage
//!   components) combine freely: a lenient superset.
//! - `calc()` and relative color syntax are not supported; theme tokens play
//!   that role in markup.

mod convert;
mod hex;
mod hsl;
mod hsv;
mod hwb;
mod lab;
mod lch;
mod named;
mod oklab;
mod oklch;
mod rgb;
mod space;

pub use named::SystemColor;

use crate::{context::Theme, frame::Color};

/// A parsed CSS color value, before lowering to a terminal [`Color`].
///
/// Terminal cells are opaque and know nothing of CSS inheritance, so
/// translucency and contextual keywords survive parsing here and only
/// collapse when a caller lowers them with the context it actually has.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CssColor {
	/// An absolute color: sRGB channel bytes plus an alpha in `[0, 1]`.
	Rgba(u8, u8, u8, f32),
	/// The `currentcolor` keyword: the element's own foreground.
	Current,
	/// A system color keyword, resolved against the [`Theme`].
	System(SystemColor),
}

impl CssColor {
	/// Parses any supported CSS color; `None` when `value` is not one.
	pub fn parse(value: &str) -> Option<Self> {
		parse(value)
	}

	/// An opaque absolute color.
	pub const fn rgb(red: u8, green: u8, blue: u8) -> Self {
		Self::Rgba(red, green, blue, 1.0)
	}

	/// Lowers to a cell color with only a theme in hand: system colors
	/// read the theme, while `currentcolor` and fully transparent values
	/// become [`Color::Default`] — the terminal's own pass-through
	/// color is the nearest thing to "inherit" and "let the background
	/// show". Translucent values keep their color: without a backdrop
	/// there is nothing to blend with (see [`Self::composite`]).
	pub fn resolve(self, theme: &Theme) -> Color {
		match self {
			Self::Rgba(_, _, _, alpha) if alpha <= 0.0 => Color::Default,
			Self::Rgba(red, green, blue, _) => Color::Rgb(red, green, blue),
			Self::Current => Color::Default,
			Self::System(system) => system.resolve(theme),
		}
	}

	/// Lowers to a cell color with full context: `currentcolor` becomes
	/// `current`, system colors read the theme, and translucency is
	/// alpha-blended over `backdrop`. Blending needs concrete channels,
	/// so over a default or indexed backdrop full transparency yields
	/// the backdrop itself and partial translucency keeps the color.
	///
	/// # Example
	/// ```
	/// use omp_tui::{Color, CssColor, Theme};
	/// let red = CssColor::parse("rgb(255 0 0 / 50%)").unwrap();
	/// let lowered = red.composite(&Theme::default(), Color::Default, Color::Rgb(0, 0, 0));
	/// assert_eq!(lowered, Color::Rgb(128, 0, 0));
	/// ```
	pub fn composite(self, theme: &Theme, current: Color, backdrop: Color) -> Color {
		let (color, alpha) = match self {
			Self::Rgba(red, green, blue, alpha) => (Color::Rgb(red, green, blue), alpha),
			Self::Current => (current, 1.0),
			Self::System(system) => (system.resolve(theme), 1.0),
		};
		match (color, backdrop) {
			_ if alpha >= 1.0 => color,
			(Color::Rgb(red, green, blue), Color::Rgb(below_r, below_g, below_b)) => {
				let mix = |top: u8, below: u8| {
					f32::from(top)
						.mul_add(alpha, f32::from(below) * (1.0 - alpha))
						.round() as u8
				};
				Color::Rgb(mix(red, below_r), mix(green, below_g), mix(blue, below_b))
			},
			_ if alpha <= 0.0 => backdrop,
			_ => color,
		}
	}
}

/// Parses any supported CSS color; `None` when `value` is not one.
pub fn parse(value: &str) -> Option<CssColor> {
	let value = value.trim();
	if let Some(hex) = value.strip_prefix('#') {
		return hex::parse(hex);
	}
	// Every remaining form is ASCII; the guard also makes the byte
	// slicing below safe for arbitrary UTF-8 input.
	if !value.is_ascii() {
		return None;
	}
	if let Some(open) = value.find('(') {
		let body = value[open + 1..].strip_suffix(')')?;
		return function(&value[..open], body);
	}
	named::parse(value)
}

/// Dispatches one `name(body)` form to its family parser.
fn function(name: &str, body: &str) -> Option<CssColor> {
	type Family = fn(&str) -> Option<CssColor>;
	const FAMILIES: &[(&str, Family)] = &[
		("color", space::parse),
		("hsl", hsl::parse),
		("hsla", hsl::parse),
		("hsv", hsv::parse),
		("hsva", hsv::parse),
		("hwb", hwb::parse),
		("lab", lab::parse),
		("lch", lch::parse),
		("oklab", oklab::parse),
		("oklch", oklch::parse),
		("rgb", rgb::parse),
		("rgba", rgb::parse),
	];
	let (_, family) = FAMILIES
		.iter()
		.find(|(candidate, _)| name.eq_ignore_ascii_case(candidate))?;
	family(body)
}

/// Component tokens of one color-function body: up to four space- or
/// comma-separated values plus an optional `/`-separated alpha.
pub struct Components<'a> {
	parts:  [&'a str; 4],
	count:  usize,
	alpha:  Option<&'a str>,
	commas: bool,
}

impl<'a> Components<'a> {
	/// Tokenizes a function body. `None` on an empty list, empty or
	/// multi-token components, more than four components, or a
	/// malformed alpha tail.
	pub(super) fn split(body: &'a str) -> Option<Self> {
		let (left, alpha) = match body.split_once('/') {
			Some((left, alpha)) => {
				let alpha = alpha.trim();
				if alpha.is_empty()
					|| alpha.contains(['/', ','])
					|| alpha.contains(|c: char| c.is_ascii_whitespace())
				{
					return None;
				}
				(left, Some(alpha))
			},
			None => (body, None),
		};
		let commas = left.contains(',');
		let mut parts = [""; 4];
		let mut count = 0;
		let mut push = |token: &'a str| {
			if count == 4 {
				return None;
			}
			parts[count] = token;
			count += 1;
			Some(())
		};
		if commas {
			for part in left.split(',') {
				let part = part.trim();
				if part.is_empty() || part.contains(|c: char| c.is_ascii_whitespace()) {
					return None;
				}
				push(part)?;
			}
		} else {
			for part in left.split_ascii_whitespace() {
				push(part)?;
			}
		}
		(count > 0).then_some(Self { parts, count, alpha, commas })
	}

	/// Exactly three channels plus optional alpha; a fourth
	/// comma-separated component is legacy alpha (`rgba(r, g, b, a)`).
	pub(super) const fn three(&self) -> Option<([&'a str; 3], Option<&'a str>)> {
		let channels = [self.parts[0], self.parts[1], self.parts[2]];
		match (self.count, self.commas, self.alpha) {
			(3, false, alpha) => Some((channels, alpha)),
			(3, true, None) => Some((channels, None)),
			(4, true, None) => Some((channels, Some(self.parts[3]))),
			_ => None,
		}
	}

	/// Exactly three space-separated channels; comma syntax rejected
	/// (`hwb()` and the lab-family functions never had a legacy form).
	pub(super) fn modern3(&self) -> Option<([&'a str; 3], Option<&'a str>)> {
		(self.count == 3 && !self.commas)
			.then(|| ([self.parts[0], self.parts[1], self.parts[2]], self.alpha))
	}

	/// Every component plus alpha, for `color()`'s ident-led argument
	/// list; `commas` reports whether legacy separators were used.
	pub(super) fn all(&self) -> (&[&'a str], Option<&'a str>, bool) {
		(&self.parts[..self.count], self.alpha, self.commas)
	}
}

/// Parses a CSS `<number>`: float with optional sign and exponent.
/// Rejects NaN, infinities, and unit suffixes.
pub fn number(token: &str) -> Option<f32> {
	// Reject alphabetic forms Rust accepts but CSS does not ("inf",
	// "NaN") along with any stray unit; the exponent marker is the one
	// legal letter.
	if token
		.bytes()
		.any(|b| b.is_ascii_alphabetic() && !matches!(b, b'e' | b'E'))
	{
		return None;
	}
	let value: f32 = token.parse().ok()?;
	value.is_finite().then_some(value)
}

/// Parses a `<number>` (taken raw) or `<percentage>` (`100%` maps to
/// `scale`); the `none` keyword reads as zero.
pub fn number_or_percent(token: &str, scale: f32) -> Option<f32> {
	if token.eq_ignore_ascii_case("none") {
		return Some(0.0);
	}
	match token.strip_suffix('%') {
		Some(percent) => Some(number(percent)? / 100.0 * scale),
		None => number(token),
	}
}

/// Parses a CSS `<hue>`: a bare number in degrees or an angle with a
/// `deg`/`grad`/`rad`/`turn` unit; returns degrees normalized to
/// `[0, 360)`. The `none` keyword reads as zero.
pub fn hue(token: &str) -> Option<f32> {
	if token.eq_ignore_ascii_case("none") {
		return Some(0.0);
	}
	// "grad" must be tried before its suffix "rad".
	let (raw, factor) = if let Some(raw) = strip_unit(token, "grad") {
		(raw, 0.9)
	} else if let Some(raw) = strip_unit(token, "rad") {
		(raw, 180.0 / std::f32::consts::PI)
	} else if let Some(raw) = strip_unit(token, "deg") {
		(raw, 1.0)
	} else if let Some(raw) = strip_unit(token, "turn") {
		(raw, 360.0)
	} else {
		(token, 1.0)
	};
	Some((number(raw)? * factor).rem_euclid(360.0))
}

/// Parses an optional `<alpha-value>` clamped to `[0, 1]`: a number, a
/// percentage, or `none` (fully transparent). A missing component is
/// opaque.
pub fn alpha(token: Option<&str>) -> Option<f32> {
	match token {
		None => Some(1.0),
		Some(token) => Some(number_or_percent(token, 1.0)?.clamp(0.0, 1.0)),
	}
}

/// Case-insensitively strips a trailing unit from an ASCII token.
fn strip_unit<'a>(token: &'a str, unit: &str) -> Option<&'a str> {
	let split = token.len().checked_sub(unit.len())?;
	token[split..]
		.eq_ignore_ascii_case(unit)
		.then(|| &token[..split])
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn red_in_every_exact_notation() {
		let red = Some(CssColor::rgb(255, 0, 0));
		for form in [
			"red",
			"RED",
			" red ",
			"#f00",
			"#F00f",
			"#ff0000",
			"#ff0000ff",
			"rgb(255, 0, 0)",
			"RGB(255 0 0)",
			"rgb(100% 0% 0%)",
			"hsl(0 100% 50%)",
			"hsl(360deg, 100%, 50%)",
			"hsv(0 100% 100%)",
			"hwb(0 0% 0%)",
			"color(srgb 1 0 0)",
			"color(srgb-linear 1 0 0)",
		] {
			assert_eq!(parse(form), red, "{form}");
		}
	}

	#[test]
	fn alpha_survives_parsing_in_every_family() {
		assert_eq!(parse("rgba(255, 0, 0, 0.5)"), Some(CssColor::Rgba(255, 0, 0, 0.5)));
		assert_eq!(parse("rgb(255 0 0 / 25%)"), Some(CssColor::Rgba(255, 0, 0, 0.25)));
		assert_eq!(parse("#ff000080"), Some(CssColor::Rgba(255, 0, 0, 128.0 / 255.0)));
		assert_eq!(parse("hsl(0 100% 50% / 25%)"), Some(CssColor::Rgba(255, 0, 0, 0.25)));
		assert_eq!(parse("hsva(0, 100%, 100%, 0)"), Some(CssColor::Rgba(255, 0, 0, 0.0)));
		assert_eq!(parse("lab(100 0 0 / 0.5)"), Some(CssColor::Rgba(255, 255, 255, 0.5)));
		assert_eq!(parse("color(srgb 1 0 0 / 75%)"), Some(CssColor::Rgba(255, 0, 0, 0.75)));
	}

	#[test]
	fn keywords_keep_their_css_semantics() {
		assert_eq!(parse("transparent"), Some(CssColor::Rgba(0, 0, 0, 0.0)));
		assert_eq!(parse("currentcolor"), Some(CssColor::Current));
		assert_eq!(parse("CurrentColor"), Some(CssColor::Current));
		assert_eq!(parse("Canvas"), Some(CssColor::System(SystemColor::Canvas)));
		assert_eq!(parse("buttontext"), Some(CssColor::System(SystemColor::ButtonText)));
	}

	#[test]
	fn resolve_lowers_with_theme_context() {
		let theme = Theme::default();
		assert_eq!(CssColor::rgb(1, 2, 3).resolve(&theme), Color::Rgb(1, 2, 3));
		assert_eq!(CssColor::Rgba(9, 9, 9, 0.0).resolve(&theme), Color::Default);
		assert_eq!(CssColor::Rgba(9, 9, 9, 0.5).resolve(&theme), Color::Rgb(9, 9, 9));
		assert_eq!(CssColor::Current.resolve(&theme), Color::Default);
		assert_eq!(CssColor::System(SystemColor::CanvasText).resolve(&theme), theme.fg);
		assert_eq!(CssColor::System(SystemColor::Canvas).resolve(&theme), Color::Default);
	}

	#[test]
	fn composite_blends_translucency_over_a_concrete_backdrop() {
		let theme = Theme::default();
		let half_red = CssColor::Rgba(255, 0, 0, 0.5);
		assert_eq!(
			half_red.composite(&theme, Color::Default, Color::Rgb(0, 0, 255)),
			Color::Rgb(128, 0, 128)
		);
		// No concrete channels to blend with: keep the color.
		assert_eq!(half_red.composite(&theme, Color::Default, Color::Default), Color::Rgb(255, 0, 0));
		assert_eq!(
			CssColor::Current.composite(&theme, Color::Rgb(7, 7, 7), Color::Rgb(0, 0, 0)),
			Color::Rgb(7, 7, 7)
		);
		assert_eq!(
			CssColor::Rgba(1, 1, 1, 0.0).composite(&theme, Color::Default, Color::Indexed(3)),
			Color::Indexed(3)
		);
	}

	#[test]
	fn function_dispatch_rejects_malformed_shells() {
		for form in ["rgb(", "rgb)", "rgb 255 0 0", "rgb (255 0 0)", "nosuch(1 2 3)", "rgb(💥)"] {
			assert_eq!(parse(form), None, "{form}");
		}
	}

	#[test]
	fn number_rejects_css_invalid_floats() {
		assert_eq!(number("1e2"), Some(100.0));
		assert_eq!(number("-.5"), Some(-0.5));
		for token in ["nan", "inf", "infinity", "1px", "0x10", ""] {
			assert_eq!(number(token), None, "{token}");
		}
	}

	#[test]
	fn percent_maps_to_scale_and_number_stays_raw() {
		assert_eq!(number_or_percent("50%", 255.0), Some(127.5));
		assert_eq!(number_or_percent("50", 255.0), Some(50.0));
		assert_eq!(number_or_percent("100%", 0.4), Some(0.4));
		assert_eq!(number_or_percent("none", 255.0), Some(0.0));
		assert_eq!(number_or_percent("%", 255.0), None);
	}

	#[test]
	fn hue_units_convert_to_degrees() {
		assert_eq!(hue("90"), Some(90.0));
		assert_eq!(hue("90deg"), Some(90.0));
		assert_eq!(hue("100GRAD"), Some(90.0));
		assert_eq!(hue("0.25turn"), Some(90.0));
		let radians = hue("1.5707964rad").unwrap();
		assert!((radians - 90.0).abs() < 1e-3, "{radians}");
		assert_eq!(hue("none"), Some(0.0));
		assert_eq!(hue("90px"), None);
	}

	#[test]
	fn hue_normalizes_into_one_turn() {
		assert_eq!(hue("-90"), Some(270.0));
		assert_eq!(hue("450"), Some(90.0));
		assert_eq!(hue("-0.5turn"), Some(180.0));
	}

	#[test]
	fn alpha_reads_number_percent_and_none() {
		assert_eq!(alpha(None), Some(1.0));
		assert_eq!(alpha(Some("0.5")), Some(0.5));
		assert_eq!(alpha(Some("50%")), Some(0.5));
		assert_eq!(alpha(Some("none")), Some(0.0));
		assert_eq!(alpha(Some("300%")), Some(1.0));
		assert_eq!(alpha(Some("-1")), Some(0.0));
		assert_eq!(alpha(Some("half")), None);
	}

	#[test]
	fn components_split_legacy_and_modern() {
		let legacy = Components::split("1, 2, 3, 0.5").unwrap();
		assert_eq!(legacy.three(), Some((["1", "2", "3"], Some("0.5"))));
		let modern = Components::split("1 2 3 / 25%").unwrap();
		assert_eq!(modern.three(), Some((["1", "2", "3"], Some("25%"))));
		assert_eq!(modern.modern3(), Some((["1", "2", "3"], Some("25%"))));
		let bare = Components::split("1 2 3").unwrap();
		assert_eq!(bare.three(), Some((["1", "2", "3"], None)));
	}

	#[test]
	fn components_reject_mixed_and_overflowing_forms() {
		for body in ["", " ", "1,,3", "1 2, 3", "1,2,3,4,5", "1 2 3 4 5", "1,2,3 / 0.5 0.5"] {
			assert!(Components::split(body).is_none(), "{body}");
		}
		// Legacy commas combined with a slash alpha, or five components,
		// are rejected at interpretation time.
		assert_eq!(Components::split("1,2,3 / .5").unwrap().three(), None);
		assert_eq!(Components::split("1 2 3 4").unwrap().three(), None);
		assert_eq!(Components::split("1 2 3 4").unwrap().modern3(), None);
	}
}
