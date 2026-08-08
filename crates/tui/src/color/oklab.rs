use super::{Components, CssColor, alpha, convert, number_or_percent};

/// Parses CSS `oklab()` colors using the modern space-separated grammar.
pub(super) fn parse(body: &str) -> Option<CssColor> {
	let (channels, alpha_value) = Components::split(body)?.modern3()?;
	let alpha = alpha(alpha_value)?;
	let lightness = number_or_percent(channels[0], 1.0)?.clamp(0.0, 1.0);
	let a = number_or_percent(channels[1], 0.4)?;
	let b = number_or_percent(channels[2], 0.4)?;
	let (red, green, blue) = to_srgb(lightness, a, b);
	Some(CssColor::Rgba(red, green, blue, alpha))
}

/// Converts an `OKLab` triple to gamut-mapped sRGB channel bytes.
pub(super) fn to_srgb(lightness: f32, a: f32, b: f32) -> (u8, u8, u8) {
	let (red, green, blue) = convert::oklab_to_linear_srgb((lightness, a, b));
	convert::linear_srgb(red, green, blue)
}

#[cfg(test)]
mod tests {
	use super::parse;
	use crate::color::CssColor;

	fn assert_rgb_near(actual: Option<CssColor>, expected: (u8, u8, u8)) {
		let Some(CssColor::Rgba(red, green, blue, _)) = actual else {
			panic!("expected RGBA color, got {actual:?}");
		};
		assert!((i16::from(red) - i16::from(expected.0)).abs() <= 2, "red: {red}");
		assert!((i16::from(green) - i16::from(expected.1)).abs() <= 2, "green: {green}");
		assert!((i16::from(blue) - i16::from(expected.2)).abs() <= 2, "blue: {blue}");
	}

	#[test]
	fn achromatic_endpoints_are_black_and_white() {
		assert_eq!(parse("0 0 0"), Some(CssColor::rgb(0, 0, 0)));
		assert_eq!(parse("1 0 0"), Some(CssColor::rgb(255, 255, 255)));
	}

	#[test]
	fn percentage_lightness_and_axes_use_css_scales() {
		assert_eq!(parse("100% 0% 0%"), Some(CssColor::rgb(255, 255, 255)));
		assert_eq!(parse("0.5 100% 0"), parse("0.5 0.4 0"));
	}

	#[test]
	fn known_red_anchor_converts_to_srgb() {
		assert_rgb_near(parse("0.627955 0.224863 0.125846"), (255, 0, 0));
	}

	#[test]
	fn lightness_clamps_before_conversion() {
		assert_eq!(parse("2 0 0"), Some(CssColor::rgb(255, 255, 255)));
		assert_eq!(parse("-1 0 0"), Some(CssColor::rgb(0, 0, 0)));
	}

	#[test]
	fn comma_syntax_is_rejected() {
		assert_eq!(parse("0.5, 0.1, 0"), None);
	}

	#[test]
	fn slash_alpha_is_preserved_and_validated() {
		let Some(CssColor::Rgba(red, green, blue, _)) = parse("0.5 0.1 0") else {
			panic!("expected RGBA color");
		};
		assert_eq!(parse("0.5 0.1 0 / 25%"), Some(CssColor::Rgba(red, green, blue, 0.25)));
		assert_eq!(parse("0.5 0.1 0 / invalid"), None);
	}

	#[test]
	fn none_axes_are_achromatic() {
		assert_eq!(parse("0.5 none none"), parse("0.5 0 0"));
	}
}
