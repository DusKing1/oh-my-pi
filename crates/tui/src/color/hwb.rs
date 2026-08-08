//! HWB parsing and conversion through gamma-encoded sRGB.

use super::{Components, CssColor, alpha, convert, hsl, hue, number_or_percent};

/// Parses a modern, space-separated HWB function body.
pub(super) fn parse(body: &str) -> Option<CssColor> {
	let components = Components::split(body)?;
	let ([hue_token, white_token, black_token], alpha_token) = components.modern3()?;
	let hue = hue(hue_token)?;
	let white = number_or_percent(white_token, 100.0)?.clamp(0.0, 100.0) / 100.0;
	let black = number_or_percent(black_token, 100.0)?.clamp(0.0, 100.0) / 100.0;
	let alpha = alpha(alpha_token)?;

	if white + black >= 1.0 {
		let gray = white / (white + black);
		let (red, green, blue) = convert::srgb(gray, gray, gray);
		return Some(CssColor::Rgba(red, green, blue, alpha));
	}

	let (red, green, blue) = hsl::to_srgb(hue, 100.0, 50.0);
	let scale = 1.0 - white - black;
	let (red, green, blue) = convert::srgb(
		red.mul_add(scale, white),
		green.mul_add(scale, white),
		blue.mul_add(scale, white),
	);
	Some(CssColor::Rgba(red, green, blue, alpha))
}

#[cfg(test)]
mod tests {
	use super::{CssColor, parse};

	#[test]
	fn zero_white_and_black_preserve_the_hue() {
		assert_eq!(parse("0 0% 0%"), Some(CssColor::rgb(255, 0, 0)));
	}

	#[test]
	fn full_white_is_white() {
		assert_eq!(parse("90 100% 0%"), Some(CssColor::rgb(255, 255, 255)));
	}

	#[test]
	fn full_black_is_black() {
		assert_eq!(parse("90 0% 100%"), Some(CssColor::rgb(0, 0, 0)));
	}

	#[test]
	fn excess_white_and_black_normalize_to_gray() {
		assert_eq!(parse("0 100% 100%"), Some(CssColor::rgb(128, 128, 128)));
	}

	#[test]
	fn partial_white_and_black_keep_blue_dominant() {
		let Some(CssColor::Rgba(red, green, blue, _)) = parse("240 30% 20%") else {
			panic!("expected an RGB color");
		};
		assert!(blue > red && blue > green);
		assert_eq!((red, green, blue), (77, 77, 204));
	}

	#[test]
	fn legacy_comma_separators_are_rejected() {
		assert_eq!(parse("0, 0%, 0%"), None);
	}

	#[test]
	fn slash_alpha_is_preserved_and_validated() {
		assert_eq!(parse("0 0% 0% / 0"), Some(CssColor::Rgba(255, 0, 0, 0.0)));
		assert_eq!(parse("0 0% 0% / opaque"), None);
	}

	#[test]
	fn bare_channels_clamp_like_percentages() {
		assert_eq!(parse("0 -20 120"), Some(CssColor::rgb(0, 0, 0)));
		assert_eq!(parse("0 25 25"), parse("0 25% 25%"));
	}

	#[test]
	fn hue_units_and_wrap_are_supported() {
		assert_eq!(parse("-120 0% 0%"), parse("240 0% 0%"));
		assert_eq!(parse("0.5turn 0% 0%"), parse("180 0% 0%"));
	}
}
