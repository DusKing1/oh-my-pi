use super::{Components, CssColor};

/// Parses legacy and modern CSS RGB component syntax.
pub(super) fn parse(body: &str) -> Option<CssColor> {
	let (channels, alpha) = Components::split(body)?.three()?;
	let red = channel(channels[0])?;
	let green = channel(channels[1])?;
	let blue = channel(channels[2])?;
	let alpha = super::alpha(alpha)?;
	Some(CssColor::Rgba(red, green, blue, alpha))
}

fn channel(token: &str) -> Option<u8> {
	Some(
		super::number_or_percent(token, 255.0)?
			.clamp(0.0, 255.0)
			.round() as u8,
	)
}

#[cfg(test)]
mod tests {
	use super::{CssColor, parse};

	#[test]
	fn parses_legacy_comma_syntax() {
		assert_eq!(parse("255, 0, 0"), Some(CssColor::rgb(255, 0, 0)));
	}

	#[test]
	fn parses_modern_space_syntax() {
		assert_eq!(parse("255 0 0"), Some(CssColor::rgb(255, 0, 0)));
	}

	#[test]
	fn parses_percentage_channels() {
		assert_eq!(parse("100% 0% 0%"), Some(CssColor::rgb(255, 0, 0)));
	}

	#[test]
	fn parses_mixed_number_and_percentage_channels() {
		assert_eq!(parse("50% 128 0"), Some(CssColor::rgb(128, 128, 0)));
	}

	#[test]
	fn none_channels_are_zero() {
		assert_eq!(parse("none none none"), Some(CssColor::rgb(0, 0, 0)));
	}

	#[test]
	fn accepts_and_preserves_legacy_alpha() {
		assert_eq!(parse("255, 0, 0, .25"), Some(CssColor::Rgba(255, 0, 0, 0.25)));
	}

	#[test]
	fn accepts_and_preserves_modern_alpha() {
		assert_eq!(parse("255 0 0 / 25%"), Some(CssColor::Rgba(255, 0, 0, 0.25)));
	}

	#[test]
	fn clamps_out_of_range_channels() {
		assert_eq!(parse("300 -20 0"), Some(CssColor::rgb(255, 0, 0)));
	}

	#[test]
	fn rejects_wrong_channel_counts_and_mixed_separators() {
		for body in ["1 2", "1 2 3 4 5", "1,2,3 / .5"] {
			assert_eq!(parse(body), None, "{body}");
		}
	}

	#[test]
	fn rejects_junk_tokens_and_alpha() {
		for body in ["red 0 0", "1 2 px", "1 2 3 / opaque"] {
			assert_eq!(parse(body), None, "{body}");
		}
	}
}
