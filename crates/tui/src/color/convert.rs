//! Shared color-space math from the CSS Color 4 conversion algorithms:
//! sRGB transfer functions, XYZ and `OKLab` conversions, Bradford
//! chromatic adaptation, and `OKLCh` gamut mapping.
//!
//! Family modules hand off at one of three points: gamma-encoded sRGB
//! ([`srgb`]), linear-light sRGB ([`linear_srgb`], which gamut-maps out
//! of range results), or CIE XYZ ([`xyz_d65_to_linear_srgb`], with
//! [`xyz_d50_to_d65`] for D50-referenced spaces).

/// Multiplies a row-major 3×3 matrix by a column vector.
pub(super) fn mat3(m: &[[f32; 3]; 3], (a, b, c): (f32, f32, f32)) -> (f32, f32, f32) {
	let row = |r: &[f32; 3]| c.mul_add(r[2], b.mul_add(r[1], a * r[0]));
	(row(&m[0]), row(&m[1]), row(&m[2]))
}

/// Quantizes a gamma-encoded sRGB triple in `[0, 1]` to channel bytes,
/// clamping stray values (CSS parsed-value clamping for the sRGB-native
/// functions).
pub(super) fn srgb(red: f32, green: f32, blue: f32) -> (u8, u8, u8) {
	let channel = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
	(channel(red), channel(green), channel(blue))
}

/// Encodes a linear-light sRGB triple to channel bytes, gamut-mapping
/// out-of-range colors per CSS Color 4 §13.2.
pub(super) fn linear_srgb(red: f32, green: f32, blue: f32) -> (u8, u8, u8) {
	let (red, green, blue) = gamut_map((red, green, blue));
	srgb(srgb_encode(red), srgb_encode(green), srgb_encode(blue))
}

/// sRGB opto-electronic encode: linear-light to gamma-encoded,
/// sign-preserving as CSS specifies for out-of-gamut values.
fn srgb_encode(value: f32) -> f32 {
	let magnitude = value.abs();
	let encoded = if magnitude > 0.003_130_8 {
		1.055f32.mul_add(magnitude.powf(1.0 / 2.4), -0.055)
	} else {
		12.92 * magnitude
	};
	encoded.copysign(value)
}

/// sRGB electro-optical decode: gamma-encoded to linear-light,
/// sign-preserving. Also the `display-p3` transfer function.
pub(super) fn srgb_decode(value: f32) -> f32 {
	let magnitude = value.abs();
	let decoded = if magnitude > 0.040_45 {
		((magnitude + 0.055) / 1.055).powf(2.4)
	} else {
		magnitude / 12.92
	};
	decoded.copysign(value)
}

/// CIE XYZ (D65 white point) to linear-light sRGB.
pub(super) fn xyz_d65_to_linear_srgb(xyz: (f32, f32, f32)) -> (f32, f32, f32) {
	const TO_SRGB: [[f32; 3]; 3] = [
		[3.240_97, -1.537_383_2, -0.498_610_76],
		[-0.969_243_65, 1.875_967_5, 0.041_555_06],
		[0.055_630_08, -0.203_976_96, 1.056_971_5],
	];
	mat3(&TO_SRGB, xyz)
}

/// Bradford chromatic adaptation from a D50 to a D65 white point.
pub(super) fn xyz_d50_to_d65(xyz: (f32, f32, f32)) -> (f32, f32, f32) {
	const ADAPT: [[f32; 3]; 3] = [
		[0.955_473_4, -0.023_098_536, 0.063_259_31],
		[-0.028_369_707, 1.009_995_5, 0.021_041_399],
		[0.012_314_002, -0.020_507_696, 1.330_365_9],
	];
	mat3(&ADAPT, xyz)
}

/// `OKLab` to linear-light sRGB (Ottosson's inverse transform, as used by
/// CSS Color 4).
pub(super) fn oklab_to_linear_srgb(lab: (f32, f32, f32)) -> (f32, f32, f32) {
	const OKLAB_TO_LMS_ROOT: [[f32; 3]; 3] = [
		[1.0, 0.396_337_78, 0.215_803_76],
		[1.0, -0.105_561_346, -0.063_854_17],
		[1.0, -0.089_484_18, -1.291_485_5],
	];
	const LMS_TO_LINEAR_SRGB: [[f32; 3]; 3] = [
		[4.076_741_7, -3.307_711_6, 0.230_969_94],
		[-1.268_438, 2.609_757_4, -0.341_319_38],
		[-0.004_196_086_4, -0.703_418_6, 1.707_614_7],
	];
	let (l, m, s) = mat3(&OKLAB_TO_LMS_ROOT, lab);
	mat3(&LMS_TO_LINEAR_SRGB, (l * l * l, m * m * m, s * s * s))
}

/// Linear-light sRGB to `OKLab` (Ottosson's forward transform).
fn linear_srgb_to_oklab(rgb: (f32, f32, f32)) -> (f32, f32, f32) {
	const LINEAR_SRGB_TO_LMS: [[f32; 3]; 3] = [
		[0.412_221_46, 0.536_332_55, 0.051_445_995],
		[0.211_903_5, 0.680_699_5, 0.107_396_96],
		[0.088_302_46, 0.281_718_85, 0.629_978_7],
	];
	const LMS_ROOT_TO_OKLAB: [[f32; 3]; 3] = [
		[0.210_454_26, 0.793_617_8, -0.004_072_047],
		[1.977_998_5, -2.428_592_2, 0.450_593_7],
		[0.025_904_037, 0.782_771_77, -0.808_675_77],
	];
	let (l, m, s) = mat3(&LINEAR_SRGB_TO_LMS, rgb);
	mat3(&LMS_ROOT_TO_OKLAB, (l.cbrt(), m.cbrt(), s.cbrt()))
}

/// Maps a possibly out-of-gamut linear sRGB color into gamut with the
/// CSS Color 4 §13.2 state machine (mirrored from the colorjs.io
/// reference): bisect `OKLCh` chroma at constant lightness and hue,
/// keeping the LARGEST chroma whose channel clip stays within a
/// just-noticeable ΔEOK — an under-JND clip only ends the search once
/// ΔE sits within `EPSILON` of the JND boundary; otherwise it raises
/// the lower bound and keeps looking. Preserves hue where naive
/// clipping would shift it.
fn gamut_map(rgb: (f32, f32, f32)) -> (f32, f32, f32) {
	const JND: f32 = 0.02;
	const EPSILON: f32 = 1e-4;
	if in_gamut(rgb) {
		return clip(rgb);
	}
	let (lightness, a, b) = linear_srgb_to_oklab(rgb);
	if lightness >= 1.0 {
		return (1.0, 1.0, 1.0);
	}
	if lightness <= 0.0 {
		return (0.0, 0.0, 0.0);
	}
	// Origins already within a JND of their clip take the fast path.
	let mut clipped = clip(rgb);
	if delta_eok(clipped, rgb) < JND {
		return clipped;
	}
	let (sin, cos) = b.atan2(a).sin_cos();
	let mut low = 0.0_f32;
	let mut high = a.hypot(b);
	let mut low_in_gamut = true;
	// Every branch halves the interval, so ~14 iterations reach the
	// epsilon; the cap keeps the loop finite for degenerate inputs.
	for _ in 0..64 {
		if high - low <= EPSILON {
			break;
		}
		let chroma = f32::midpoint(low, high);
		let current = oklab_to_linear_srgb((lightness, chroma * cos, chroma * sin));
		if low_in_gamut && in_gamut(current) {
			low = chroma;
			continue;
		}
		clipped = clip(current);
		let delta = delta_eok(clipped, current);
		if delta < JND {
			if JND - delta < EPSILON {
				return clipped;
			}
			low_in_gamut = false;
			low = chroma;
		} else {
			high = chroma;
		}
	}
	clipped
}

/// Whether every channel sits exactly inside sRGB — `toGamutCSS` in the
/// colorjs.io reference passes `epsilon: 0` to its gamut checks, unlike
/// the library's lenient default.
fn in_gamut((red, green, blue): (f32, f32, f32)) -> bool {
	let ok = |value: f32| (0.0..=1.0).contains(&value);
	ok(red) && ok(green) && ok(blue)
}

/// Clamps channels to the sRGB cube.
const fn clip((red, green, blue): (f32, f32, f32)) -> (f32, f32, f32) {
	(red.clamp(0.0, 1.0), green.clamp(0.0, 1.0), blue.clamp(0.0, 1.0))
}

/// Euclidean distance between two linear sRGB colors in `OKLab`.
fn delta_eok(left: (f32, f32, f32), right: (f32, f32, f32)) -> f32 {
	let (l1, a1, b1) = linear_srgb_to_oklab(left);
	let (l2, a2, b2) = linear_srgb_to_oklab(right);
	let (dl, da, db) = (l1 - l2, a1 - a2, b1 - b2);
	db.mul_add(db, da.mul_add(da, dl * dl)).sqrt()
}

#[cfg(test)]
mod tests {
	use super::*;

	/// D65 reference white in CIE XYZ.
	const D65_WHITE: (f32, f32, f32) = (0.950_47, 1.0, 1.088_83);
	/// D50 reference white in CIE XYZ.
	const D50_WHITE: (f32, f32, f32) = (0.964_22, 1.0, 0.825_21);

	#[test]
	fn d65_white_is_srgb_white() {
		let (r, g, b) = xyz_d65_to_linear_srgb(D65_WHITE);
		assert!((r - 1.0).abs() < 2e-3 && (g - 1.0).abs() < 2e-3 && (b - 1.0).abs() < 2e-3);
		assert_eq!(linear_srgb(r, g, b), (255, 255, 255));
	}

	#[test]
	fn srgb_green_primary_round_trips_from_xyz() {
		// Column two of the lin-sRGB-to-XYZ matrix is the green primary.
		let (r, g, b) = xyz_d65_to_linear_srgb((0.357_584_34, 0.715_168_68, 0.119_194_78));
		assert_eq!(linear_srgb(r, g, b), (0, 255, 0));
	}

	#[test]
	fn bradford_adaptation_maps_d50_white_to_d65_white() {
		let (r, g, b) = xyz_d65_to_linear_srgb(xyz_d50_to_d65(D50_WHITE));
		assert_eq!(linear_srgb(r, g, b), (255, 255, 255));
	}

	#[test]
	fn transfer_functions_invert_each_other() {
		for value in [0.0, 0.001, 0.04, 0.5, 1.0] {
			let round = srgb_decode(srgb_encode(value));
			assert!((round - value).abs() < 1e-5, "{value} -> {round}");
		}
	}

	#[test]
	fn oklab_transforms_invert_each_other() {
		let rgb = (0.3, 0.2, 0.6);
		let (r, g, b) = oklab_to_linear_srgb(linear_srgb_to_oklab(rgb));
		assert!((r - 0.3).abs() < 1e-4 && (g - 0.2).abs() < 1e-4 && (b - 0.6).abs() < 1e-4);
	}

	#[test]
	fn srgb_quantizer_clamps_stray_channels() {
		assert_eq!(srgb(1.2, -0.1, 0.5), (255, 0, 128));
	}

	#[test]
	fn in_gamut_colors_pass_through_unmapped() {
		// Linear 0.5 is gamma ~0.7354: the classic "50% gray is not 128".
		assert_eq!(linear_srgb(0.5, 0.5, 0.5), (188, 188, 188));
		assert_eq!(linear_srgb(1.0, 0.0, 0.0), (255, 0, 0));
	}

	#[test]
	fn gamut_mapping_preserves_hue_where_clipping_would_not() {
		// display-p3 red expressed in linear sRGB; naive clipping gives
		// (255, 0, 0) while CSS gamut mapping keeps a sliver of the
		// green/blue that preserves the OKLCh hue.
		let (r, g, b) = linear_srgb(1.224_94, -0.042_057, -0.019_638);
		assert_eq!(r, 255);
		assert!(g.abs_diff(11) <= 2, "green: {g}");
		assert!(b.abs_diff(12) <= 2, "blue: {b}");
	}

	type GamutVector = ((f32, f32, f32), (u8, u8, u8));
	#[test]
	fn gamut_mapping_matches_the_colorjs_reference() {
		// Oracle vectors from colorjs.io 0.7.1 `toGamut({ method: "css" })`,
		// the CSS WG reference implementation: linear sRGB origin to
		// gamut-mapped sRGB bytes. ±2 absorbs f32-vs-f64 bisection drift.
		#[rustfmt::skip]
		const VECTORS: &[GamutVector] = &[
			((1.224_94, -0.042_057, -0.019_638), (255, 11, 12)),   // display-p3 red
			((-0.224_94, 1.042_057, -0.078_636), (0, 251, 41)),    // display-p3 green
			((0.0, -0.0, 1.098_274), (0, 0, 255)),                 // display-p3 blue
			((1.0, 1.0, -0.098_274), (254, 255, 0)),               // display-p3 yellow
			((-0.177_121, 0.820_528, 0.802_875), (0, 229, 226)),   // display-p3 0 .9 .9
			((1.660_491, -0.124_55, -0.018_151), (255, 73, 79)),   // rec2020 red
			((-0.587_641, 1.132_9, -0.100_579), (0, 242, 114)),    // rec2020 green
			((-0.072_85, -0.008_349, 1.118_73), (0, 81, 147)),     // rec2020 blue
			((1.398_356, -0.0, 0.0), (255, 90, 72)),               // a98-rgb red
			((-0.727_636, 1.231_743, -0.153_267), (0, 245, 117)),  // prophoto green
			((0.673_088, -0.118_526, 1.969_633), (187, 77, 255)),  // lab(50 125 -125)
			((-0.140_498, 0.614_978, -0.034_898), (0, 199, 41)),   // lab(70 -90 80)
			((0.537_705, -0.080_673, 0.009_181), (156, 0, 49)),    // lab(30 100 40)
			((-0.372_064, 0.948_801, -0.073_975), (0, 231, 90)),   // oklch(0.8 0.4 150)
			((0.999_872, -0.000_049, -0.000_006), (255, 0, 0)),    // oklch sRGB red
			((-0.024_647, -0.007_729, 1.484_238), (0, 69, 254)),   // oklch(0.5 0.37 260)
			((1.147_387, 0.706_108, -0.249_376), (255, 223, 0)),   // oklch(0.9 0.3 100)
			((0.181_605, -0.029_117, 0.176_684), (104, 0, 101)),   // oklch(0.35 0.25 330)
			((1.518_808, -0.056_034, 0.050_831), (255, 84, 100)),  // oklch(0.7 0.32 20)
			((1.01, 0.5, 0.5), (255, 188, 188)),                   // barely out: high red
			((0.5, -0.01, 0.2), (188, 0, 124)),                    // barely out: low green
			((1.005, 1.002, 0.998), (255, 255, 255)),              // barely out: near white
			((0.0, 0.003, 1.02), (0, 10, 255)),                    // barely out: high blue
		];
		for &((red, green, blue), want) in VECTORS {
			let got = linear_srgb(red, green, blue);
			for (channel, (got, want)) in [(got.0, want.0), (got.1, want.1), (got.2, want.2)]
				.into_iter()
				.enumerate()
			{
				assert!(
					got.abs_diff(want) <= 2,
					"channel {channel}: {got} vs {want} for ({red}, {green}, {blue})"
				);
			}
		}
	}

	#[test]
	fn gamut_check_uses_exact_bounds_like_the_reference() {
		// Near-black boundary origin where the 12.92x linear segment
		// amplifies the branch difference: colorjs.io maps this to gamma
		// (0.0052830, 0.0006362, 0) -> (1, 0, 0) with the `epsilon: 0`
		// checks `toGamutCSS` uses. The library's lenient generic
		// in-gamut default (7.5e-5) would accept a higher-chroma
		// candidate and produce (2, 0, 0), so this assert is exact.
		assert_eq!(linear_srgb(0.001_268, -0.000_184, -0.000_563), (1, 0, 0));
	}

	#[test]
	fn gamut_mapping_pins_lightness_extremes() {
		// OKLab lightness beyond the endpoints snaps to black or white.
		assert_eq!(linear_srgb(1.8, 1.9, 2.0), (255, 255, 255));
		assert_eq!(linear_srgb(-0.4, -0.5, -0.2), (0, 0, 0));
	}
}
