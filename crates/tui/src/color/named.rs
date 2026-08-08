//! CSS color keywords: the full extended named-color set (CSS Color 4
//! `<named-color>`, including `rebeccapurple`), the `transparent` and
//! `currentcolor` specials, and the system colors with their deprecated
//! aliases.

use super::CssColor;
use crate::{context::Theme, frame::Color};

/// A CSS system color keyword, resolved against the [`Theme`]'s
/// semantic palette rather than a fixed RGB value.
///
/// Deprecated keywords (`ActiveBorder`, `WindowText`, ...) parse to the
/// modern keyword CSS Color 4 §6.5 maps them to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SystemColor {
	/// Accent fill for selected or activated interface parts.
	AccentColor,
	/// Text painted over [`Self::AccentColor`] fills.
	AccentColorText,
	/// Text of active links.
	ActiveText,
	/// Border of push buttons.
	ButtonBorder,
	/// Face of push buttons.
	ButtonFace,
	/// Text on push buttons.
	ButtonText,
	/// The application background: the terminal's own default.
	Canvas,
	/// Text on the application background.
	CanvasText,
	/// Background of input fields.
	Field,
	/// Text inside input fields.
	FieldText,
	/// Disabled or de-emphasized text.
	GrayText,
	/// Background of selected text.
	Highlight,
	/// Selected text.
	HighlightText,
	/// Text of unvisited links.
	LinkText,
	/// Background of highlighter marks.
	Mark,
	/// Text inside highlighter marks.
	MarkText,
	/// Background of chosen items.
	SelectedItem,
	/// Text of chosen items.
	SelectedItemText,
	/// Text of visited links.
	VisitedText,
}

impl SystemColor {
	/// Case-insensitively parses a system color keyword, including the
	/// deprecated aliases. Called by [`Theme::token`] so markup resolves
	/// system colors through the same deferred path as theme tokens.
	pub(crate) fn parse(name: &str) -> Option<Self> {
		let (_, system) = SYSTEM
			.iter()
			.find(|(keyword, _)| keyword.eq_ignore_ascii_case(name))?;
		Some(*system)
	}

	/// Resolves to the theme color playing this keyword's role.
	pub(crate) const fn resolve(self, theme: &Theme) -> Color {
		match self {
			Self::Canvas => Color::Default,
			Self::CanvasText | Self::ButtonText | Self::FieldText => theme.fg,
			Self::AccentColor
			| Self::ActiveText
			| Self::Highlight
			| Self::LinkText
			| Self::SelectedItem => theme.accent,
			Self::AccentColorText | Self::HighlightText | Self::MarkText | Self::SelectedItemText => {
				theme.contrast
			},
			Self::ButtonFace | Self::Field => theme.surface,
			Self::ButtonBorder | Self::GrayText | Self::VisitedText => theme.muted,
			Self::Mark => theme.warn,
		}
	}
}

/// Longest keyword across every table: `lightgoldenrodyellow`.
const LONGEST: usize = 20;

/// Resolves a color keyword, case-insensitively.
pub(super) fn parse(name: &str) -> Option<CssColor> {
	if name.is_empty() || name.len() > LONGEST || !name.is_ascii() {
		return None;
	}
	let mut lower = [0_u8; LONGEST];
	for (slot, byte) in lower.iter_mut().zip(name.bytes()) {
		*slot = byte.to_ascii_lowercase();
	}
	let needle = std::str::from_utf8(&lower[..name.len()]).ok()?;
	if needle == "transparent" {
		return Some(CssColor::Rgba(0, 0, 0, 0.0));
	}
	if needle == "currentcolor" {
		return Some(CssColor::Current);
	}
	if let Ok(index) = SYSTEM.binary_search_by(|(keyword, _)| keyword.cmp(&needle)) {
		return Some(CssColor::System(SYSTEM[index].1));
	}
	let index = NAMED
		.binary_search_by(|(candidate, _)| candidate.cmp(&needle))
		.ok()?;
	let rgb = NAMED[index].1;
	Some(CssColor::rgb((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8))
}

/// System color keywords (current and CSS Color 4 §6.5 deprecated
/// aliases), lowercase and sorted for binary search.
const SYSTEM: &[(&str, SystemColor)] = &[
	("accentcolor", SystemColor::AccentColor),
	("accentcolortext", SystemColor::AccentColorText),
	("activeborder", SystemColor::ButtonBorder),
	("activecaption", SystemColor::Canvas),
	("activetext", SystemColor::ActiveText),
	("appworkspace", SystemColor::Canvas),
	("background", SystemColor::Canvas),
	("buttonborder", SystemColor::ButtonBorder),
	("buttonface", SystemColor::ButtonFace),
	("buttonhighlight", SystemColor::ButtonFace),
	("buttonshadow", SystemColor::ButtonFace),
	("buttontext", SystemColor::ButtonText),
	("canvas", SystemColor::Canvas),
	("canvastext", SystemColor::CanvasText),
	("captiontext", SystemColor::CanvasText),
	("field", SystemColor::Field),
	("fieldtext", SystemColor::FieldText),
	("graytext", SystemColor::GrayText),
	("highlight", SystemColor::Highlight),
	("highlighttext", SystemColor::HighlightText),
	("inactiveborder", SystemColor::ButtonBorder),
	("inactivecaption", SystemColor::Canvas),
	("inactivecaptiontext", SystemColor::GrayText),
	("infobackground", SystemColor::Canvas),
	("infotext", SystemColor::CanvasText),
	("linktext", SystemColor::LinkText),
	("mark", SystemColor::Mark),
	("marktext", SystemColor::MarkText),
	("menu", SystemColor::Canvas),
	("menutext", SystemColor::CanvasText),
	("scrollbar", SystemColor::Canvas),
	("selecteditem", SystemColor::SelectedItem),
	("selecteditemtext", SystemColor::SelectedItemText),
	("threeddarkshadow", SystemColor::ButtonBorder),
	("threedface", SystemColor::ButtonFace),
	("threedhighlight", SystemColor::ButtonBorder),
	("threedlightshadow", SystemColor::ButtonBorder),
	("threedshadow", SystemColor::ButtonBorder),
	("visitedtext", SystemColor::VisitedText),
	("window", SystemColor::Canvas),
	("windowframe", SystemColor::ButtonBorder),
	("windowtext", SystemColor::CanvasText),
];

/// Every CSS/HTML named color (CSS Color Module extended keywords),
/// sorted for binary search.
const NAMED: &[(&str, u32)] = &[
	("aliceblue", 0x00f0_f8ff),
	("antiquewhite", 0x00fa_ebd7),
	("aqua", 0x0000_ffff),
	("aquamarine", 0x007f_ffd4),
	("azure", 0x00f0_ffff),
	("beige", 0x00f5_f5dc),
	("bisque", 0x00ff_e4c4),
	("black", 0x0000_0000),
	("blanchedalmond", 0x00ff_ebcd),
	("blue", 0x0000_00ff),
	("blueviolet", 0x008a_2be2),
	("brown", 0x00a5_2a2a),
	("burlywood", 0x00de_b887),
	("cadetblue", 0x005f_9ea0),
	("chartreuse", 0x007f_ff00),
	("chocolate", 0x00d2_691e),
	("coral", 0x00ff_7f50),
	("cornflowerblue", 0x0064_95ed),
	("cornsilk", 0x00ff_f8dc),
	("crimson", 0x00dc_143c),
	("cyan", 0x0000_ffff),
	("darkblue", 0x0000_008b),
	("darkcyan", 0x0000_8b8b),
	("darkgoldenrod", 0x00b8_860b),
	("darkgray", 0x00a9_a9a9),
	("darkgreen", 0x0000_6400),
	("darkgrey", 0x00a9_a9a9),
	("darkkhaki", 0x00bd_b76b),
	("darkmagenta", 0x008b_008b),
	("darkolivegreen", 0x0055_6b2f),
	("darkorange", 0x00ff_8c00),
	("darkorchid", 0x0099_32cc),
	("darkred", 0x008b_0000),
	("darksalmon", 0x00e9_967a),
	("darkseagreen", 0x008f_bc8f),
	("darkslateblue", 0x0048_3d8b),
	("darkslategray", 0x002f_4f4f),
	("darkslategrey", 0x002f_4f4f),
	("darkturquoise", 0x0000_ced1),
	("darkviolet", 0x0094_00d3),
	("deeppink", 0x00ff_1493),
	("deepskyblue", 0x0000_bfff),
	("dimgray", 0x0069_6969),
	("dimgrey", 0x0069_6969),
	("dodgerblue", 0x001e_90ff),
	("firebrick", 0x00b2_2222),
	("floralwhite", 0x00ff_faf0),
	("forestgreen", 0x0022_8b22),
	("fuchsia", 0x00ff_00ff),
	("gainsboro", 0x00dc_dcdc),
	("ghostwhite", 0x00f8_f8ff),
	("gold", 0x00ff_d700),
	("goldenrod", 0x00da_a520),
	("gray", 0x0080_8080),
	("green", 0x0000_8000),
	("greenyellow", 0x00ad_ff2f),
	("grey", 0x0080_8080),
	("honeydew", 0x00f0_fff0),
	("hotpink", 0x00ff_69b4),
	("indianred", 0x00cd_5c5c),
	("indigo", 0x004b_0082),
	("ivory", 0x00ff_fff0),
	("khaki", 0x00f0_e68c),
	("lavender", 0x00e6_e6fa),
	("lavenderblush", 0x00ff_f0f5),
	("lawngreen", 0x007c_fc00),
	("lemonchiffon", 0x00ff_facd),
	("lightblue", 0x00ad_d8e6),
	("lightcoral", 0x00f0_8080),
	("lightcyan", 0x00e0_ffff),
	("lightgoldenrodyellow", 0x00fa_fad2),
	("lightgray", 0x00d3_d3d3),
	("lightgreen", 0x0090_ee90),
	("lightgrey", 0x00d3_d3d3),
	("lightpink", 0x00ff_b6c1),
	("lightsalmon", 0x00ff_a07a),
	("lightseagreen", 0x0020_b2aa),
	("lightskyblue", 0x0087_cefa),
	("lightslategray", 0x0077_8899),
	("lightslategrey", 0x0077_8899),
	("lightsteelblue", 0x00b0_c4de),
	("lightyellow", 0x00ff_ffe0),
	("lime", 0x0000_ff00),
	("limegreen", 0x0032_cd32),
	("linen", 0x00fa_f0e6),
	("magenta", 0x00ff_00ff),
	("maroon", 0x0080_0000),
	("mediumaquamarine", 0x0066_cdaa),
	("mediumblue", 0x0000_00cd),
	("mediumorchid", 0x00ba_55d3),
	("mediumpurple", 0x0093_70db),
	("mediumseagreen", 0x003c_b371),
	("mediumslateblue", 0x007b_68ee),
	("mediumspringgreen", 0x0000_fa9a),
	("mediumturquoise", 0x0048_d1cc),
	("mediumvioletred", 0x00c7_1585),
	("midnightblue", 0x0019_1970),
	("mintcream", 0x00f5_fffa),
	("mistyrose", 0x00ff_e4e1),
	("moccasin", 0x00ff_e4b5),
	("navajowhite", 0x00ff_dead),
	("navy", 0x0000_0080),
	("oldlace", 0x00fd_f5e6),
	("olive", 0x0080_8000),
	("olivedrab", 0x006b_8e23),
	("orange", 0x00ff_a500),
	("orangered", 0x00ff_4500),
	("orchid", 0x00da_70d6),
	("palegoldenrod", 0x00ee_e8aa),
	("palegreen", 0x0098_fb98),
	("paleturquoise", 0x00af_eeee),
	("palevioletred", 0x00db_7093),
	("papayawhip", 0x00ff_efd5),
	("peachpuff", 0x00ff_dab9),
	("peru", 0x00cd_853f),
	("pink", 0x00ff_c0cb),
	("plum", 0x00dd_a0dd),
	("powderblue", 0x00b0_e0e6),
	("purple", 0x0080_0080),
	("rebeccapurple", 0x0066_3399),
	("red", 0x00ff_0000),
	("rosybrown", 0x00bc_8f8f),
	("royalblue", 0x0041_69e1),
	("saddlebrown", 0x008b_4513),
	("salmon", 0x00fa_8072),
	("sandybrown", 0x00f4_a460),
	("seagreen", 0x002e_8b57),
	("seashell", 0x00ff_f5ee),
	("sienna", 0x00a0_522d),
	("silver", 0x00c0_c0c0),
	("skyblue", 0x0087_ceeb),
	("slateblue", 0x006a_5acd),
	("slategray", 0x0070_8090),
	("slategrey", 0x0070_8090),
	("snow", 0x00ff_fafa),
	("springgreen", 0x0000_ff7f),
	("steelblue", 0x0046_82b4),
	("tan", 0x00d2_b48c),
	("teal", 0x0000_8080),
	("thistle", 0x00d8_bfd8),
	("tomato", 0x00ff_6347),
	("turquoise", 0x0040_e0d0),
	("violet", 0x00ee_82ee),
	("wheat", 0x00f5_deb3),
	("white", 0x00ff_ffff),
	("whitesmoke", 0x00f5_f5f5),
	("yellow", 0x00ff_ff00),
	("yellowgreen", 0x009a_cd32),
];

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn tables_are_sorted_for_binary_search() {
		assert!(NAMED.windows(2).all(|pair| pair[0].0 < pair[1].0));
		assert!(SYSTEM.windows(2).all(|pair| pair[0].0 < pair[1].0));
	}

	#[test]
	fn keywords_resolve_case_insensitively() {
		assert_eq!(parse("rebeccapurple"), Some(CssColor::rgb(0x66, 0x33, 0x99)));
		assert_eq!(parse("WHITE"), Some(CssColor::rgb(255, 255, 255)));
		assert_eq!(parse("LightGoldenrodYellow"), Some(CssColor::rgb(0xfa, 0xfa, 0xd2)));
	}

	#[test]
	fn special_keywords_keep_css_semantics() {
		assert_eq!(parse("transparent"), Some(CssColor::Rgba(0, 0, 0, 0.0)));
		assert_eq!(parse("CurrentColor"), Some(CssColor::Current));
	}

	#[test]
	fn system_colors_parse_including_deprecated_aliases() {
		assert_eq!(parse("Canvas"), Some(CssColor::System(SystemColor::Canvas)));
		assert_eq!(parse("HIGHLIGHT"), Some(CssColor::System(SystemColor::Highlight)));
		// Deprecated keywords map to their §6.5 modern equivalents.
		assert_eq!(parse("WindowText"), Some(CssColor::System(SystemColor::CanvasText)));
		assert_eq!(parse("ThreeDShadow"), Some(CssColor::System(SystemColor::ButtonBorder)));
		assert_eq!(SystemColor::parse("InfoBackground"), Some(SystemColor::Canvas));
	}

	#[test]
	fn system_colors_resolve_through_the_theme() {
		let theme = Theme::default();
		assert_eq!(SystemColor::Canvas.resolve(&theme), Color::Default);
		assert_eq!(SystemColor::CanvasText.resolve(&theme), theme.fg);
		assert_eq!(SystemColor::LinkText.resolve(&theme), theme.accent);
		assert_eq!(SystemColor::Mark.resolve(&theme), theme.warn);
		assert_eq!(SystemColor::GrayText.resolve(&theme), theme.muted);
		assert_eq!(SystemColor::HighlightText.resolve(&theme), theme.contrast);
		assert_eq!(SystemColor::Field.resolve(&theme), theme.surface);
	}

	#[test]
	fn junk_names_are_rejected() {
		assert_eq!(parse(""), None);
		assert_eq!(parse("nosuchcolorname"), None);
		assert_eq!(parse("lightgoldenrodyellowish"), None);
		assert_eq!(parse("wh\u{ef}te"), None);
	}
}
