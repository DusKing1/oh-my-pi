//! CSS `color()` parsing for the predefined RGB and XYZ spaces.

use super::{Components, CssColor, alpha, convert, number_or_percent};

// RGB-to-XYZ matrices from the CSS Color 4 sample conversion code.
const P3_TO_XYZ_D65: [[f32; 3]; 3] = [
	[0.486_570_95, 0.265_667_7, 0.198_217_29],
	[0.228_974_56, 0.691_738_5, 0.079_286_91],
	[0.0, 0.045_113_38, 1.043_944_4],
];
const A98_TO_XYZ_D65: [[f32; 3]; 3] = [
	[0.576_669_04, 0.185_558_24, 0.188_228_65],
	[0.297_344_98, 0.627_363_57, 0.075_291_46],
	[0.027_031_36, 0.070_688_85, 0.991_337_54],
];
const PROPHOTO_TO_XYZ_D50: [[f32; 3]; 3] = [
	[0.797_760_5, 0.135_185_84, 0.031_349_35],
	[0.288_071_13, 0.711_843_2, 0.000_085_653_96],
	[0.0, 0.0, 0.825_104_6],
];
const REC2020_TO_XYZ_D65: [[f32; 3]; 3] = [
	[0.636_958_05, 0.144_616_9, 0.168_880_98],
	[0.262_700_2, 0.677_998_07, 0.059_301_716],
	[0.0, 0.028_072_693, 1.060_985_1],
];

#[derive(Clone, Copy)]
enum Space {
	Srgb,
	SrgbLinear,
	DisplayP3,
	DisplayP3Linear,
	A98Rgb,
	ProphotoRgb,
	Rec2020,
	XyzD50,
	XyzD65,
}

impl Space {
	const fn from_ident(ident: &str) -> Option<Self> {
		if ident.eq_ignore_ascii_case("srgb") {
			Some(Self::Srgb)
		} else if ident.eq_ignore_ascii_case("srgb-linear") {
			Some(Self::SrgbLinear)
		} else if ident.eq_ignore_ascii_case("display-p3") {
			Some(Self::DisplayP3)
		} else if ident.eq_ignore_ascii_case("display-p3-linear") {
			Some(Self::DisplayP3Linear)
		} else if ident.eq_ignore_ascii_case("a98-rgb") {
			Some(Self::A98Rgb)
		} else if ident.eq_ignore_ascii_case("prophoto-rgb") {
			Some(Self::ProphotoRgb)
		} else if ident.eq_ignore_ascii_case("rec2020") {
			Some(Self::Rec2020)
		} else if ident.eq_ignore_ascii_case("xyz-d50") {
			Some(Self::XyzD50)
		} else if ident.eq_ignore_ascii_case("xyz") || ident.eq_ignore_ascii_case("xyz-d65") {
			Some(Self::XyzD65)
		} else {
			None
		}
	}
}

/// Parses a modern `color()` body, preserving alpha while converting to sRGB.
pub(super) fn parse(body: &str) -> Option<CssColor> {
	let components = Components::split(body)?;
	let (parts, alpha_token, commas) = components.all();
	if commas || parts.len() != 4 {
		return None;
	}
	let alpha = alpha(alpha_token)?;

	let space = Space::from_ident(parts[0])?;
	let channels = (
		number_or_percent(parts[1], 1.0)?,
		number_or_percent(parts[2], 1.0)?,
		number_or_percent(parts[3], 1.0)?,
	);
	let (red, green, blue) = to_srgb(space, channels);
	Some(CssColor::Rgba(red, green, blue, alpha))
}

fn to_srgb(space: Space, channels: (f32, f32, f32)) -> (u8, u8, u8) {
	let (red, green, blue) = channels;
	match space {
		Space::Srgb => convert::srgb(red, green, blue),
		Space::SrgbLinear => convert::linear_srgb(red, green, blue),
		Space::DisplayP3 => {
			let linear = map_channels(channels, convert::srgb_decode);
			from_xyz_d65(convert::mat3(&P3_TO_XYZ_D65, linear))
		},
		Space::DisplayP3Linear => from_xyz_d65(convert::mat3(&P3_TO_XYZ_D65, channels)),
		Space::A98Rgb => {
			let linear = map_channels(channels, decode_a98);
			from_xyz_d65(convert::mat3(&A98_TO_XYZ_D65, linear))
		},
		Space::ProphotoRgb => {
			let linear = map_channels(channels, decode_prophoto);
			let xyz_d50 = convert::mat3(&PROPHOTO_TO_XYZ_D50, linear);
			from_xyz_d65(convert::xyz_d50_to_d65(xyz_d50))
		},
		Space::Rec2020 => {
			let linear = map_channels(channels, decode_rec2020);
			from_xyz_d65(convert::mat3(&REC2020_TO_XYZ_D65, linear))
		},
		Space::XyzD50 => from_xyz_d65(convert::xyz_d50_to_d65(channels)),
		Space::XyzD65 => from_xyz_d65(channels),
	}
}

fn map_channels((red, green, blue): (f32, f32, f32), decode: fn(f32) -> f32) -> (f32, f32, f32) {
	(decode(red), decode(green), decode(blue))
}

fn from_xyz_d65(xyz: (f32, f32, f32)) -> (u8, u8, u8) {
	let (red, green, blue) = convert::xyz_d65_to_linear_srgb(xyz);
	convert::linear_srgb(red, green, blue)
}

fn decode_a98(value: f32) -> f32 {
	value.abs().powf(563.0 / 256.0).copysign(value)
}

fn decode_prophoto(value: f32) -> f32 {
	let magnitude = value.abs();
	let decoded = if magnitude <= 16.0 / 512.0 {
		magnitude / 16.0
	} else {
		magnitude.powf(1.8)
	};
	decoded.copysign(value)
}

fn decode_rec2020(value: f32) -> f32 {
	// Transfer constants from the CSS Color 4 sample conversion code.
	const ALPHA: f32 = 1.099_296_8;
	const BETA: f32 = 0.018_053_97;

	let magnitude = value.abs();
	let decoded = if magnitude < BETA * 4.5 {
		magnitude / 4.5
	} else {
		((magnitude + ALPHA - 1.0) / ALPHA).powf(1.0 / 0.45)
	};
	decoded.copysign(value)
}

#[cfg(test)]
mod tests {
	use super::{CssColor, parse};

	#[test]
	fn srgb_accepts_numbers_percentages_and_none() {
		assert_eq!(parse("srgb 1 0 0"), Some(CssColor::rgb(255, 0, 0)));
		assert_eq!(parse("srgb 50% 50% 50%"), Some(CssColor::rgb(128, 128, 128)));
		assert_eq!(parse("srgb none 0 0"), Some(CssColor::rgb(0, 0, 0)));
	}

	#[test]
	fn linear_srgb_encodes_before_emitting() {
		assert_eq!(parse("srgb-linear 0.5 0.5 0.5"), Some(CssColor::rgb(188, 188, 188)));
	}

	#[test]
	fn rgb_spaces_share_white_and_black_anchors() {
		for space in [
			"srgb",
			"srgb-linear",
			"display-p3",
			"display-p3-linear",
			"a98-rgb",
			"prophoto-rgb",
			"rec2020",
		] {
			assert_eq!(
				parse(&format!("{space} 1 1 1")),
				Some(CssColor::rgb(255, 255, 255)),
				"{space}"
			);
			assert_eq!(parse(&format!("{space} 0 0 0")), Some(CssColor::rgb(0, 0, 0)), "{space}");
		}
	}

	#[test]
	fn xyz_spaces_share_black_anchor() {
		for space in ["xyz", "xyz-d65", "xyz-d50"] {
			assert_eq!(parse(&format!("{space} 0 0 0")), Some(CssColor::rgb(0, 0, 0)), "{space}");
		}
	}

	#[test]
	fn display_p3_red_is_gamut_mapped_without_naive_clipping() {
		let Some(CssColor::Rgba(red, green, blue, _)) = parse("display-p3 1 0 0") else {
			panic!("expected an absolute color");
		};
		assert_eq!(red, 255);
		assert!((i16::from(green) - 11).abs() <= 2);
		assert!((i16::from(blue) - 12).abs() <= 2);
	}

	#[test]
	fn display_p3_linear_skips_the_transfer_function() {
		// P3 shares the D65 white point, so linear gray comes out as
		// linear sRGB gray — the gamma-encoded form lands at 128.
		assert_eq!(parse("display-p3-linear 0.5 0.5 0.5"), Some(CssColor::rgb(188, 188, 188)));
		assert_eq!(parse("display-p3 0.5 0.5 0.5"), Some(CssColor::rgb(128, 128, 128)));
		// Endpoint channels are unaffected by the transfer function.
		assert_eq!(parse("display-p3-linear 1 0 0"), parse("display-p3 1 0 0"));
	}

	#[test]
	fn display_p3_linear_matches_the_spec_equivalence_example() {
		// CSS Color 4 §10.5: "these are the same color" (the spec quotes
		// the linear form to four decimals, hence the one-bit slack).
		let Some(CssColor::Rgba(r1, g1, b1, _)) = parse("display-p3 0.591 0.123 0.264") else {
			panic!("expected an absolute color");
		};
		let Some(CssColor::Rgba(r2, g2, b2, _)) = parse("display-p3-linear 0.3081 0.014 0.0567")
		else {
			panic!("expected an absolute color");
		};
		assert!(r1.abs_diff(r2) <= 1, "red: {r1} vs {r2}");
		assert!(g1.abs_diff(g2) <= 1, "green: {g1} vs {g2}");
		assert!(b1.abs_diff(b2) <= 1, "blue: {b1} vs {b2}");
	}

	#[test]
	fn xyz_d65_red_primary_converts_with_rounding_tolerance() {
		let Some(CssColor::Rgba(red, green, blue, _)) = parse("xyz 0.4124 0.2126 0.0193") else {
			panic!("expected an absolute color");
		};
		assert!((i16::from(red) - 255).abs() <= 2);
		assert!(i16::from(green).abs() <= 2);
		assert!(i16::from(blue).abs() <= 2);
	}

	#[test]
	fn xyz_is_an_alias_for_xyz_d65() {
		assert_eq!(parse("xyz 0.4124 0.2126 0.0193"), parse("xyz-d65 0.4124 0.2126 0.0193"));
	}

	#[test]
	fn xyz_white_points_convert_to_white() {
		assert_eq!(parse("xyz-d65 0.95047 1 1.08883"), Some(CssColor::rgb(255, 255, 255)));
		assert_eq!(parse("xyz-d50 0.96422 1 0.82521"), Some(CssColor::rgb(255, 255, 255)));
	}

	#[test]
	fn ident_is_ascii_case_insensitive_and_alpha_is_preserved() {
		assert_eq!(parse("SRGB 0 1 0 / 25%"), Some(CssColor::Rgba(0, 255, 0, 0.25)));
		assert_eq!(parse("srgb 1 0 0 / 75%"), Some(CssColor::Rgba(255, 0, 0, 0.75)));
		assert_eq!(parse("srgb 1 0 0 / junk"), None);
	}

	#[test]
	fn malformed_component_lists_are_rejected() {
		for body in ["unknown 1 0 0", "1 0 0", "srgb 1 0", "srgb, 1, 0, 0"] {
			assert_eq!(parse(body), None, "{body}");
		}
	}
}
