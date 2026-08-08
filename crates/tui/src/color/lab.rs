//! CIE Lab colors referenced to the D50 white point.

use super::{Components, CssColor, alpha, convert, number_or_percent};

/// Parses a modern `lab()` body while preserving its semantic alpha.
pub(super) fn parse(body: &str) -> Option<CssColor> {
	let ([lightness, a, b], alpha_value) = Components::split(body)?.modern3()?;
	let alpha = alpha(alpha_value)?;
	let lightness = number_or_percent(lightness, 100.0)?.clamp(0.0, 100.0);
	let a = number_or_percent(a, 125.0)?;
	let b = number_or_percent(b, 125.0)?;
	let (red, green, blue) = to_srgb(lightness, a, b);
	Some(CssColor::Rgba(red, green, blue, alpha))
}

/// Converts D50-referenced CIE Lab coordinates to gamut-mapped sRGB bytes.
pub(super) fn to_srgb(lightness: f32, a: f32, b: f32) -> (u8, u8, u8) {
	// CSS Color 4's Lab-to-XYZ conversion constants and D50 white.
	const KAPPA: f32 = 24_389.0 / 27.0;
	const EPSILON: f32 = 216.0 / 24_389.0;
	const D50_X: f32 = 0.964_22;
	const D50_Z: f32 = 0.825_21;

	let fy = (lightness + 16.0) / 116.0;
	let fx = fy + a / 500.0;
	let fz = fy - b / 200.0;
	let fx3 = fx.powi(3);
	let fz3 = fz.powi(3);
	let xr = if fx3 > EPSILON {
		fx3
	} else {
		116.0f32.mul_add(fx, -16.0) / KAPPA
	};
	let yr = if lightness > KAPPA * EPSILON {
		fy.powi(3)
	} else {
		lightness / KAPPA
	};
	let zr = if fz3 > EPSILON {
		fz3
	} else {
		116.0f32.mul_add(fz, -16.0) / KAPPA
	};
	let xyz_d50 = (xr * D50_X, yr, zr * D50_Z);
	let xyz_d65 = convert::xyz_d50_to_d65(xyz_d50);
	let (red, green, blue) = convert::xyz_d65_to_linear_srgb(xyz_d65);
	convert::linear_srgb(red, green, blue)
}

#[cfg(test)]
mod tests {
	use super::{super::CssColor, parse};

	fn assert_rgb_near(actual: Option<CssColor>, expected: (u8, u8, u8), tolerance: u8) {
		let Some(CssColor::Rgba(red, green, blue, 1.0)) = actual else {
			panic!("expected an RGB color, got {actual:?}");
		};
		for (channel, expected) in [(red, expected.0), (green, expected.1), (blue, expected.2)] {
			assert!(
				channel.abs_diff(expected) <= tolerance,
				"{channel} was not within {tolerance} of {expected}"
			);
		}
	}

	#[test]
	fn reference_white_and_black_are_exact() {
		assert_eq!(parse("100 0 0"), Some(CssColor::rgb(255, 255, 255)));
		assert_eq!(parse("0 0 0"), Some(CssColor::rgb(0, 0, 0)));
	}

	#[test]
	fn none_channels_are_zero() {
		assert_eq!(parse("100% none none"), Some(CssColor::rgb(255, 255, 255)));
	}

	#[test]
	fn middle_lightness_is_perceptual_gray() {
		assert_rgb_near(parse("50 0 0"), (119, 119, 119), 1);
	}

	#[test]
	fn positive_a_pushes_toward_red() {
		let Some(CssColor::Rgba(red, _, blue, 1.0)) = parse("50 60 0") else {
			panic!("expected an RGB color");
		};
		assert!(red > 150 && red > blue);
	}

	#[test]
	fn negative_b_pushes_toward_blue() {
		let Some(CssColor::Rgba(red, green, blue, 1.0)) = parse("50 0 -60") else {
			panic!("expected an RGB color");
		};
		assert!(blue > red && blue > green);
	}

	#[test]
	fn lightness_clamps_to_the_css_range() {
		assert_eq!(parse("200 0 0"), Some(CssColor::rgb(255, 255, 255)));
		assert_eq!(parse("-20 0 0"), Some(CssColor::rgb(0, 0, 0)));
	}

	#[test]
	fn percentage_opponent_channels_use_css_scales() {
		assert_eq!(parse("50 100% 0"), parse("50 125 0"));
		assert_eq!(parse("50 0 -100%"), parse("50 0 -125"));
	}

	#[test]
	fn far_out_of_gamut_color_uses_css_gamut_mapping() {
		assert_rgb_near(parse("50 125 -125"), (187, 77, 255), 2);
	}

	#[test]
	fn rejects_legacy_commas_and_invalid_alpha() {
		assert_eq!(parse("50, 60, 0"), None);
		assert_eq!(parse("50 60 0 / opaque"), None);
	}
}
