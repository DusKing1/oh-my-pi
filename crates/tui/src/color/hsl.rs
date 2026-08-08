//! HSL parsing and conversion to gamma-encoded sRGB.

use super::{Components, CssColor, alpha, convert, hue, number_or_percent};

/// Parses an HSL function body, including the legacy comma-separated alpha
/// form.
pub(super) fn parse(body: &str) -> Option<CssColor> {
	let components = Components::split(body)?;
	let ([hue_token, saturation_token, lightness_token], alpha_token) = components.three()?;
	let hue = hue(hue_token)?;
	let saturation = number_or_percent(saturation_token, 100.0)?.clamp(0.0, 100.0);
	let lightness = number_or_percent(lightness_token, 100.0)?.clamp(0.0, 100.0);
	let alpha = alpha(alpha_token)?;

	let (red, green, blue) = to_srgb(hue, saturation, lightness);
	let (red, green, blue) = convert::srgb(red, green, blue);
	Some(CssColor::Rgba(red, green, blue, alpha))
}

/// Converts HSL channels to the gamma-encoded sRGB triple shared with HWB.
pub(super) fn to_srgb(hue_degrees: f32, saturation: f32, lightness: f32) -> (f32, f32, f32) {
	// CSS Color 4 HSL-to-RGB algorithm.
	let saturation = saturation / 100.0;
	let lightness = lightness / 100.0;
	let a = saturation * lightness.min(1.0 - lightness);
	let channel = |n: f32| {
		let k = (n + hue_degrees / 30.0).rem_euclid(12.0);
		let distance = (k - 3.0).min(9.0 - k).clamp(-1.0, 1.0);
		a.mul_add(-distance, lightness)
	};
	(channel(0.0), channel(8.0), channel(4.0))
}

#[cfg(test)]
mod tests {
	use super::{CssColor, parse};

	#[test]
	fn primary_anchors_accept_modern_and_legacy_separators() {
		assert_eq!(parse("0 100% 50%"), Some(CssColor::rgb(255, 0, 0)));
		assert_eq!(parse("120, 100%, 50%"), Some(CssColor::rgb(0, 255, 0)));
		assert_eq!(parse("240 100% 50%"), Some(CssColor::rgb(0, 0, 255)));
	}

	#[test]
	fn zero_saturation_at_full_lightness_is_white() {
		assert_eq!(parse("0 0% 100%"), Some(CssColor::rgb(255, 255, 255)));
	}

	#[test]
	fn negative_hues_wrap() {
		assert_eq!(parse("-240 100% 50%"), parse("120 100% 50%"));
	}

	#[test]
	fn turn_units_match_degrees() {
		assert_eq!(parse("0.5turn 100% 50%"), parse("180 100% 50%"));
	}

	#[test]
	fn saturation_clamps_below_zero() {
		assert_eq!(parse("0 -20% 50%"), Some(CssColor::rgb(128, 128, 128)));
	}

	#[test]
	fn legacy_alpha_is_preserved_and_validated() {
		assert_eq!(parse("0, 100%, 50%, 0.25"), Some(CssColor::Rgba(255, 0, 0, 0.25)));
		assert_eq!(parse("0, 100%, 50%, opaque"), None);
	}

	#[test]
	fn bare_saturation_and_lightness_are_percent_scale_values() {
		assert_eq!(parse("30 60 40"), parse("30 60% 40%"));
	}

	#[test]
	fn angle_units_and_none_are_supported() {
		assert_eq!(parse("200grad 100% 50%"), parse("180deg 100% 50%"));
		assert_eq!(parse("none 100% 50%"), parse("0 100% 50%"));
	}

	#[test]
	fn malformed_component_counts_are_rejected() {
		assert_eq!(parse("0 100%"), None);
		assert_eq!(parse("0 100% 50% extra"), None);
	}
}
