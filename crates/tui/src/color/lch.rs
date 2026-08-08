//! Polar CIE `LCh` colors referenced to the D50 white point.

use super::{Components, CssColor, alpha, hue, lab, number_or_percent};

/// Parses a modern `lch()` body while preserving its semantic alpha.
pub(super) fn parse(body: &str) -> Option<CssColor> {
	let ([lightness, chroma, hue_angle], alpha_value) = Components::split(body)?.modern3()?;
	let alpha = alpha(alpha_value)?;
	let lightness = number_or_percent(lightness, 100.0)?.clamp(0.0, 100.0);
	let chroma = number_or_percent(chroma, 150.0)?.max(0.0);
	let hue_radians = hue(hue_angle)?.to_radians();
	let a = chroma * hue_radians.cos();
	let b = chroma * hue_radians.sin();
	let (red, green, blue) = lab::to_srgb(lightness, a, b);
	Some(CssColor::Rgba(red, green, blue, alpha))
}

#[cfg(test)]
mod tests {
	use super::{super::CssColor, parse};

	fn assert_achromatic(actual: Option<CssColor>, tolerance: u8) {
		let Some(CssColor::Rgba(red, green, blue, 1.0)) = actual else {
			panic!("expected an RGB color, got {actual:?}");
		};
		assert!(red.abs_diff(green) <= tolerance, "red {red} and green {green} differ");
		assert!(red.abs_diff(blue) <= tolerance, "red {red} and blue {blue} differ");
	}

	#[test]
	fn polar_channels_match_direct_lab_conversion() {
		let hue = 22.0_f32.to_radians();
		let expected = super::super::lab::to_srgb(52.0, 58.0 * hue.cos(), 58.0 * hue.sin());
		assert_eq!(parse("52 58 22"), Some(CssColor::Rgba(expected.0, expected.1, expected.2, 1.0)));
	}

	#[test]
	fn equivalent_hue_units_match() {
		assert_eq!(parse("50 30 90"), parse("50 30 0.25turn"));
		assert_eq!(parse("50 30 100grad"), parse("50 30 90deg"));
	}

	#[test]
	fn negative_chroma_clamps_to_zero() {
		assert_eq!(parse("50 -20 40"), parse("50 0 0"));
	}

	#[test]
	fn zero_chroma_is_achromatic_at_any_hue() {
		assert_achromatic(parse("70 0 200"), 1);
	}

	#[test]
	fn alpha_is_preserved_after_validation() {
		let Some(CssColor::Rgba(red, green, blue, 1.0)) = parse("52 58 22") else {
			panic!("expected an opaque RGB color");
		};
		assert_eq!(parse("52 58 22 / 50%"), Some(CssColor::Rgba(red, green, blue, 0.5)));
		assert_eq!(parse("52 58 22 / transparent"), None);
	}

	#[test]
	fn percentage_channels_use_css_scales() {
		assert_eq!(parse("50% 100% 22"), parse("50 150 22"));
	}

	#[test]
	fn none_hue_is_zero_degrees() {
		assert_eq!(parse("50 30 none"), parse("50 30 0"));
	}

	#[test]
	fn rejects_legacy_commas_and_missing_channels() {
		assert_eq!(parse("52, 58, 22"), None);
		assert_eq!(parse("52 58"), None);
	}
}
