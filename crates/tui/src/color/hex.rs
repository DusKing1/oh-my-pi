use super::CssColor;

/// Parses compact or full-width hexadecimal CSS colors.
pub(super) fn parse(hex: &str) -> Option<CssColor> {
	if !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
		return None;
	}

	let bytes = hex.as_bytes();
	let (red, green, blue, alpha) = match bytes.len() {
		3 => (nibble(bytes[0])? * 17, nibble(bytes[1])? * 17, nibble(bytes[2])? * 17, 1.0),
		4 => (
			nibble(bytes[0])? * 17,
			nibble(bytes[1])? * 17,
			nibble(bytes[2])? * 17,
			f32::from(nibble(bytes[3])? * 17) / 255.0,
		),
		6 => (pair(bytes[0], bytes[1])?, pair(bytes[2], bytes[3])?, pair(bytes[4], bytes[5])?, 1.0),
		8 => (
			pair(bytes[0], bytes[1])?,
			pair(bytes[2], bytes[3])?,
			pair(bytes[4], bytes[5])?,
			f32::from(pair(bytes[6], bytes[7])?) / 255.0,
		),
		_ => return None,
	};
	Some(CssColor::Rgba(red, green, blue, alpha))
}

const fn nibble(byte: u8) -> Option<u8> {
	match byte {
		b'0'..=b'9' => Some(byte - b'0'),
		b'a'..=b'f' => Some(byte - b'a' + 10),
		b'A'..=b'F' => Some(byte - b'A' + 10),
		_ => None,
	}
}

fn pair(high: u8, low: u8) -> Option<u8> {
	Some(nibble(high)? * 16 + nibble(low)?)
}

#[cfg(test)]
mod tests {
	use super::{CssColor, parse};

	#[test]
	fn parses_three_digit_form() {
		assert_eq!(parse("f00"), Some(CssColor::rgb(255, 0, 0)));
	}

	#[test]
	fn parses_four_digit_form_and_preserves_alpha() {
		assert_eq!(parse("1a2f"), Some(CssColor::Rgba(17, 170, 34, 1.0)));
		assert_eq!(parse("1a28"), Some(CssColor::Rgba(17, 170, 34, 136.0 / 255.0)));
	}

	#[test]
	fn parses_six_digit_form() {
		assert_eq!(parse("12abF0"), Some(CssColor::rgb(18, 171, 240)));
	}

	#[test]
	fn parses_eight_digit_form_and_preserves_alpha() {
		assert_eq!(parse("12ABf000"), Some(CssColor::Rgba(18, 171, 240, 0.0)));
		assert_eq!(parse("12ABf080"), Some(CssColor::Rgba(18, 171, 240, 128.0 / 255.0)));
	}

	#[test]
	fn accepts_mixed_case_digits() {
		assert_eq!(parse("aBc"), Some(CssColor::rgb(170, 187, 204)));
		assert_eq!(parse("aB12cD"), Some(CssColor::rgb(171, 18, 205)));
	}

	#[test]
	fn rejects_unsupported_lengths() {
		for hex in ["", "f", "ff", "12345", "1234567", "123456789"] {
			assert_eq!(parse(hex), None, "{hex}");
		}
	}

	#[test]
	fn rejects_non_hex_ascii_bytes() {
		for hex in ["ggg", "12-", "fffffg", "00 0000"] {
			assert_eq!(parse(hex), None, "{hex}");
		}
	}

	#[test]
	fn rejects_non_ascii_bytes() {
		for hex in ["é00", "💥", "１２３"] {
			assert_eq!(parse(hex), None, "{hex}");
		}
	}
}
