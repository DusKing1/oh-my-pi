use super::{Components, CssColor, alpha, hue, number_or_percent, oklab};

/// Parses CSS `oklch()` colors using polar `OKLab` coordinates.
pub(super) fn parse(body: &str) -> Option<CssColor> {
	let (channels, alpha_value) = Components::split(body)?.modern3()?;
	let alpha = alpha(alpha_value)?;
	let lightness = number_or_percent(channels[0], 1.0)?.clamp(0.0, 1.0);
	let chroma = number_or_percent(channels[1], 0.4)?.max(0.0);
	let hue = hue(channels[2])?.to_radians();
	let a = chroma * hue.cos();
	let b = chroma * hue.sin();
	let (red, green, blue) = oklab::to_srgb(lightness, a, b);
	Some(CssColor::Rgba(red, green, blue, alpha))
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
	fn known_red_anchor_converts_to_srgb() {
		assert_rgb_near(parse("0.627955 0.257683 29.2339"), (255, 0, 0));
	}

	#[test]
	fn out_of_gamut_color_is_mapped_without_losing_hue() {
		assert_rgb_near(parse("0.8 0.4 150"), (0, 231, 90));
	}

	#[test]
	fn equivalent_hue_units_match() {
		assert_eq!(parse("0.5 0.2 180deg"), parse("0.5 0.2 0.5turn"));
	}

	#[test]
	fn negative_chroma_clamps_to_achromatic() {
		assert_eq!(parse("0.5 -0.2 180"), parse("0.5 0 137"));
	}

	#[test]
	fn none_hue_is_valid_for_white() {
		assert_eq!(parse("1 0 none"), Some(CssColor::rgb(255, 255, 255)));
	}

	#[test]
	fn polar_coordinates_match_cartesian_oklab() {
		let chroma = 0.2_f32;
		let hue = 60.0_f32.to_radians();
		let (red, green, blue) = super::oklab::to_srgb(0.5, chroma * hue.cos(), chroma * hue.sin());
		assert_eq!(parse("0.5 0.2 60deg"), Some(CssColor::rgb(red, green, blue)));
	}

	#[test]
	fn percentage_channels_use_css_scales() {
		assert_eq!(parse("50% 50% 0"), parse("0.5 0.2 0"));
	}

	#[test]
	fn lightness_and_chroma_clamp_at_their_lower_bounds() {
		assert_eq!(parse("-1 0 90"), Some(CssColor::rgb(0, 0, 0)));
		assert_eq!(parse("0.5 -100% 20"), parse("0.5 0 0"));
	}

	#[test]
	fn comma_syntax_is_rejected() {
		assert_eq!(parse("0.5, 0.2, 30"), None);
	}

	#[test]
	fn slash_alpha_is_preserved_and_validated() {
		let Some(CssColor::Rgba(red, green, blue, _)) = parse("0.5 0.2 30") else {
			panic!("expected RGBA color");
		};
		assert_eq!(parse("0.5 0.2 30 / none"), Some(CssColor::Rgba(red, green, blue, 0.0)));
		assert_eq!(parse("0.5 0.2 30 / bad"), None);
	}
}
