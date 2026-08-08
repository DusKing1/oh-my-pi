//! HSV is a non-CSS extension provided as a convenient alternative to HSL.

use super::{Components, CssColor, alpha, convert, hue, number_or_percent};

/// Parses an HSV function body, including the extension's legacy alpha form.
pub(super) fn parse(body: &str) -> Option<CssColor> {
	let components = Components::split(body)?;
	let ([hue_token, saturation_token, value_token], alpha_token) = components.three()?;
	let hue = hue(hue_token)?;
	let saturation = number_or_percent(saturation_token, 100.0)?.clamp(0.0, 100.0) / 100.0;
	let value = number_or_percent(value_token, 100.0)?.clamp(0.0, 100.0) / 100.0;
	let alpha = alpha(alpha_token)?;

	let chroma = value * saturation;
	let x = chroma * (1.0 - ((hue / 60.0).rem_euclid(2.0) - 1.0).abs());
	let (red, green, blue) = match (hue / 60.0).floor() as u8 {
		0 => (chroma, x, 0.0),
		1 => (x, chroma, 0.0),
		2 => (0.0, chroma, x),
		3 => (0.0, x, chroma),
		4 => (x, 0.0, chroma),
		_ => (chroma, 0.0, x),
	};
	let minimum = value - chroma;
	let (red, green, blue) = convert::srgb(red + minimum, green + minimum, blue + minimum);
	Some(CssColor::Rgba(red, green, blue, alpha))
}

#[cfg(test)]
mod tests {
	use super::{CssColor, parse};

	#[test]
	fn primary_anchors_accept_modern_and_legacy_separators() {
		assert_eq!(parse("0 100% 100%"), Some(CssColor::rgb(255, 0, 0)));
		assert_eq!(parse("120, 100%, 100%"), Some(CssColor::rgb(0, 255, 0)));
		assert_eq!(parse("240 100% 100%"), Some(CssColor::rgb(0, 0, 255)));
	}

	#[test]
	fn zero_value_is_black() {
		assert_eq!(parse("75 100% 0%"), Some(CssColor::rgb(0, 0, 0)));
	}

	#[test]
	fn zero_saturation_scales_gray_by_value() {
		assert_eq!(parse("275 0% 40%"), Some(CssColor::rgb(102, 102, 102)));
	}

	#[test]
	fn legacy_alpha_is_preserved_and_validated() {
		assert_eq!(parse("0, 100%, 100%, 10%"), Some(CssColor::Rgba(255, 0, 0, 0.1)));
		assert_eq!(parse("0, 100%, 100%, opaque"), None);
	}

	#[test]
	fn hue_wrap_and_units_are_supported() {
		assert_eq!(parse("-240 100% 100%"), parse("120 100% 100%"));
		assert_eq!(parse("0.5turn 100% 100%"), parse("180 100% 100%"));
	}

	#[test]
	fn bare_channels_match_percent_scale_values() {
		assert_eq!(parse("30 60 40"), parse("30 60% 40%"));
	}

	#[test]
	fn channels_clamp_to_their_parsed_ranges() {
		assert_eq!(parse("0 200% 200%"), Some(CssColor::rgb(255, 0, 0)));
		assert_eq!(parse("0 -10% 50%"), Some(CssColor::rgb(128, 128, 128)));
	}

	#[test]
	fn slash_alpha_is_preserved_but_malformed_input_is_not() {
		assert_eq!(parse("60 100% 100% / none"), Some(CssColor::Rgba(255, 255, 0, 0.0)));
		assert_eq!(parse("60 100% / 1"), None);
	}
}
