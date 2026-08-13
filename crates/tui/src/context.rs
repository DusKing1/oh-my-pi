//! Presentation context: glyph charset and color theme.
//!
//! Widgets never hardcode glyphs or colors — they consult the [`UiContext`]
//! carried by [`crate::Ui`]. Agents author semantic tokens (`accent`,
//! `warn`, …) and structural markup; the context decides what a border,
//! cursor, or `warn` actually looks like on this terminal.

use crate::{color::SystemColor, component::Elements, frame::Color, markup::Border};
/// Terminal policy for Hangul Compatibility Jamo (`U+3131..=U+318E`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum JamoWidth {
	/// Follow the platform default: narrow on macOS, Unicode tables elsewhere.
	#[default]
	Platform,
	/// Use the Unicode width table without a terminal-specific correction.
	Unicode,
	/// Force visible Compatibility Jamo to one cell.
	Narrow,
	/// Force visible Compatibility Jamo to two cells.
	Wide,
}

impl JamoWidth {
	const fn from_caps(value: u8) -> Self {
		match value {
			1 => Self::Narrow,
			2 => Self::Wide,
			_ => Self::Platform,
		}
	}
}

/// Glyph capability tier, mirroring the `unicode | nerd | ascii` symbol
/// presets in the coding agent.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Charset {
	/// Full Unicode box drawing, geometric shapes, half blocks.
	#[default]
	Unicode,
	/// Unicode plus Nerd Font private-use glyphs where they read better.
	NerdFont,
	/// Pure 7-bit ASCII: every terminal, every font, every era.
	Ascii,
}

/// Table-grid glyph set resolved by [`Charset::grid`]: border rows as
/// `(left, junction, right)` triples plus the row-interior separators.
#[derive(Clone, Copy)]
pub struct Grid {
	/// Horizontal fill between junctions.
	pub fill:   char,
	/// Left edge of a content row.
	pub lead:   &'static str,
	/// Between-cells separator.
	pub mid:    &'static str,
	/// Right edge of a content row.
	pub tail:   &'static str,
	/// Top border row glyphs.
	pub top:    (char, char, char),
	/// Separator row glyphs.
	pub middle: (char, char, char),
	/// Bottom border row glyphs.
	pub bottom: (char, char, char),
}


/// Semantic prefixes for diff lines.
#[derive(Clone, Copy, Debug)]
pub struct DiffPrefixes {
	/// Header or file metadata prefix.
	pub header:  &'static str,
	/// Unchanged context line prefix.
	pub context: &'static str,
	/// Added line prefix.
	pub add:     &'static str,
	/// Removed line prefix.
	pub remove:  &'static str,
}

impl Charset {
	pub const fn diff_prefixes(self) -> DiffPrefixes {
		match self {
			Self::Ascii => DiffPrefixes {
				header:  "  ",
				context: "  ",
				add:     "+ ",
				remove:  "- ",
			},
			_ => DiffPrefixes {
				header:  "  ",
				context: "  ",
				add:     "+ ",
				remove:  "- ",
			},
		}
	}

	/// Resolves a semantic icon through this terminal's capability tier.
	pub const fn icon(self, icon: crate::Icon) -> &'static str {
		icon.glyph(self)
	}

	/// Resolves a short icon name or qualified compatibility alias.
	pub fn icon_named(self, name: &str) -> Option<&'static str> {
		crate::Icon::from_name(name).map(|icon| self.icon(icon))
	}

	/// Border glyph set for a box: `(tl, tr, bl, br, horizontal, vertical)`.
	/// Public so raw-frame hosts painting their own chrome share the
	/// widget tier policy instead of hardcoding box drawing.
	pub const fn border(self, border: Border) -> (char, char, char, char, char, char) {
		match self {
			Self::Ascii => ('+', '+', '+', '+', '-', '|'),
			_ => match border {
				Border::Square => ('┌', '┐', '└', '┘', '─', '│'),
				Border::Dash => ('┌', '┐', '└', '┘', '╌', '┆'),
				Border::Round => ('╭', '╮', '╰', '╯', '─', '│'),
				Border::Heavy => ('┏', '┓', '┗', '┛', '━', '┃'),
				Border::Double => ('╔', '╗', '╚', '╝', '═', '║'),
			},
		}
	}

	/// Focus cursor prefix, two cells wide.
	pub const fn cursor(self) -> &'static str {
		match self {
			Self::Unicode => "❯ ",
			Self::NerdFont => "\u{f054} ",
			Self::Ascii => "> ",
		}
	}

	/// Radio mark for `(selected)`.
	pub const fn radio(self, selected: bool) -> &'static str {
		match (self, selected) {
			(Self::Ascii, true) => "(o)",
			(Self::Ascii, false) => "( )",
			(Self::NerdFont, true) => "\u{f192}",
			(Self::NerdFont, false) => "\u{f10c}",
			(_, true) => "◉",
			(_, false) => "○",
		}
	}

	/// Checkbox mark for `(checked)`.
	pub(crate) const fn checkbox(self, checked: bool) -> &'static str {
		match (self, checked) {
			(Self::Ascii, true) => "[x]",
			(Self::Ascii, false) => "[ ]",
			(Self::NerdFont, true) => "\u{f14a}",
			(Self::NerdFont, false) => "\u{f096}",
			(_, true) => "☑",
			(_, false) => "☐",
		}
	}

	/// Tree expander for `(has_children, open)`.
	pub(crate) const fn expander(self, open: bool) -> &'static str {
		match (self, open) {
			(Self::Ascii, true) => "v ",
			(Self::Ascii, false) => "> ",
			(_, true) => "▾ ",
			(_, false) => "▸ ",
		}
	}

	/// Tree guide glyphs for a connector family: `(branch, last, continue)`.
	///
	/// Each is two cells wide; ASCII terminals collapse every family to the
	/// same 7-bit set.
	pub(crate) const fn guides(self, family: Border) -> (&'static str, &'static str, &'static str) {
		match self {
			Self::Ascii => ("|-", "`-", "| "),
			_ => match family {
				Border::Square => ("├─", "└─", "│ "),
				Border::Dash => ("├╌", "└╌", "┆ "),
				Border::Round => ("├─", "╰─", "│ "),
				Border::Heavy => ("┣━", "┗━", "┃ "),
				Border::Double => ("╠═", "╚═", "║ "),
			},
		}
	}

	/// Horizontal rule / divider fill character.
	pub(crate) const fn rule(self) -> char {
		match self {
			Self::Ascii => '-',
			_ => '─',
		}
	}

	/// A rule fill honoring this tier: non-ASCII requests (box-drawing,
	/// em-dashes) degrade to the plain [`Charset::rule`] character on
	/// ASCII terminals; ASCII requests pass through everywhere.
	pub(crate) const fn rule_fill(self, requested: char) -> char {
		if matches!(self, Self::Ascii) && !requested.is_ascii() {
			self.rule()
		} else {
			requested
		}
	}

	/// Blockquote rail prefix.
	pub(crate) const fn quote_rail(self) -> &'static str {
		match self {
			Self::Ascii => "| ",
			_ => "│ ",
		}
	}

	/// Grid chrome for cell-bordered tables: the square border strokes
	/// plus the tees and cross that [`Charset::border`] alone cannot
	/// provide.
	pub const fn grid(self) -> Grid {
		match self {
			Self::Ascii => Grid {
				fill:   '-',
				lead:   "| ",
				mid:    " | ",
				tail:   " |",
				top:    ('+', '+', '+'),
				middle: ('+', '+', '+'),
				bottom: ('+', '+', '+'),
			},
			_ => Grid {
				fill:   '─',
				lead:   "│ ",
				mid:    " │ ",
				tail:   " │",
				top:    ('┌', '┬', '┐'),
				middle: ('├', '┼', '┤'),
				bottom: ('└', '┴', '┘'),
			},
		}
	}

	/// Scrollbar `(track, thumb)`.
	pub const fn scrollbar(self) -> (&'static str, &'static str) {
		match self {
			Self::Ascii => ("|", "#"),
			_ => ("│", "█"),
		}
	}

	/// Progress bar `(filled, empty)`.
	pub(crate) const fn progress(self) -> (&'static str, &'static str) {
		match self {
			Self::Ascii => ("#", "."),
			_ => ("█", "░"),
		}
	}

	/// Pill chip caps `(left, right)`; empty in ASCII (flat chips).
	pub(crate) const fn pill_caps(self) -> (&'static str, &'static str) {
		match self {
			Self::Ascii => ("", ""),
			_ => ("▐", "▌"),
		}
	}

	/// Left rail glyph for editors and `<note>` callouts.
	pub const fn rail(self) -> &'static str {
		match self {
			Self::Ascii => "| ",
			_ => "▎ ",
		}
	}

	/// Status-band chrome: `(left cap, segment separator, right cap)`.
	pub(crate) const fn status_band(self) -> (&'static str, &'static str, &'static str) {
		match self {
			Self::Ascii => ("", ">", ">"),
			Self::Unicode => ("", "›", "›"),
			Self::NerdFont => ("\u{e0b6}", "\u{e0b1}", "\u{e0b0}"),
		}
	}

	/// Right-docked status-band chrome, [`Charset::status_band`] mirrored:
	/// the opening cap points left into the surrounding background and the
	/// closing edge ends flat, solid against the right margin.
	pub(crate) const fn status_band_end(self) -> (&'static str, &'static str, &'static str) {
		match self {
			Self::Ascii => ("<", ">", ""),
			Self::Unicode => ("‹", "›", ""),
			Self::NerdFont => ("\u{e0b2}", "\u{e0b1}", ""),
		}
	}

	/// Lift-shadow glyph under risen chrome; `None` skips the shadow —
	/// ASCII has no half blocks worth faking with punctuation.
	pub(crate) const fn shadow(self) -> Option<&'static str> {
		match self {
			Self::Ascii => None,
			_ => Some("▀"),
		}
	}

	/// Spinner animation frames for this tier.
	pub const fn spinner(self) -> crate::anim::Frames {
		match self {
			Self::Ascii => crate::anim::Frames::SPINNER_ASCII,
			_ => crate::anim::Frames::SPINNER,
		}
	}

	/// Text cursor beam shown in inline edit modes.
	pub(crate) const fn beam(self) -> &'static str {
		match self {
			Self::Ascii => "_",
			_ => "▏",
		}
	}

	/// Success / chosen mark.
	pub const fn check(self) -> &'static str {
		match self {
			Self::Ascii => "*",
			Self::NerdFont => "\u{f00c}",
			Self::Unicode => "✓",
		}
	}

	/// `<note>` header icon.
	pub(crate) const fn note_icon(self) -> &'static str {
		match self {
			Self::Ascii => "[i]",
			Self::NerdFont => "\u{f05a}",
			Self::Unicode => "ℹ",
		}
	}

	/// Enum-cycle affordance `(left, right)` arrows.
	pub(crate) const fn arrows(self) -> (&'static str, &'static str) {
		match self {
			Self::Ascii => ("<", ">"),
			_ => ("◂", "▸"),
		}
	}

	/// Dropdown-opens-here affordance.
	pub(crate) const fn dropdown(self) -> &'static str {
		match self {
			Self::Ascii => " v",
			_ => " ▾",
		}
	}
}

/// Terminal-reported background appearance.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Appearance {
	/// A background whose BT.601 luminance is below 0.5.
	#[default]
	Dark,
	/// A background whose BT.601 luminance is at least 0.5.
	Light,
}

impl Appearance {
	/// Classifies 16-bit RGB components using BT.601 luminance.
	pub const fn from_rgb16(red: u16, green: u16, blue: u16) -> Self {
		let weighted = 299 * red as u64 + 587 * green as u64 + 114 * blue as u64;
		if weighted < 500 * u16::MAX as u64 {
			Self::Dark
		} else {
			Self::Light
		}
	}

	/// Classifies 8-bit RGB components using BT.601 luminance.
	pub const fn from_rgb8(red: u8, green: u8, blue: u8) -> Self {
		Self::from_rgb16((red as u16) * 0x101, (green as u16) * 0x101, (blue as u16) * 0x101)
	}
}

/// Semantic color palette. Agents pick meanings; the theme picks colors —
/// no widget hardcodes an RGB value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Theme {
	/// Default foreground.
	pub fg:        Color,
	/// Primary interactive accent (focus, active controls, links).
	pub accent:    Color,
	/// Informational values.
	pub info:      Color,
	/// Success / enabled.
	pub ok:        Color,
	/// Caution / modified.
	pub warn:      Color,
	/// Errors / destructive.
	pub err:       Color,
	/// De-emphasized chrome and hints.
	pub muted:     Color,
	/// Container borders and rules; dimmer than `fg`, brighter than `surface`.
	pub border:    Color,
	/// Neutral chip / button fill.
	pub surface:   Color,
	/// Hover row tint.
	pub hover:     Color,
	/// Text-selection background tint.
	pub selection: Color,
	/// Drop-shadow tint painted under lifted (elevated) surfaces.
	pub shadow:    Color,
	/// Text painted on top of accent/warn fills.
	pub contrast:  Color,
}

impl Default for Theme {
	fn default() -> Self {
		Self {
			fg:        Color::Rgb(0xc8, 0xcc, 0xd4),
			accent:    Color::Rgb(0x61, 0xaf, 0xef),
			info:      Color::Rgb(0x56, 0xb6, 0xc2),
			ok:        Color::Rgb(0x98, 0xc3, 0x79),
			warn:      Color::Rgb(0xe5, 0xc0, 0x7b),
			err:       Color::Rgb(0xe0, 0x6c, 0x75),
			muted:     Color::Rgb(0x5c, 0x63, 0x70),
			border:    Color::Rgb(0x45, 0x4b, 0x58),
			surface:   Color::Rgb(0x3a, 0x3f, 0x4b),
			hover:     Color::Rgb(0x2c, 0x31, 0x3a),
			selection: Color::Rgb(0x36, 0x4c, 0x61),
			shadow:    Color::Rgb(0x05, 0x07, 0x0c),
			contrast:  Color::Rgb(0x10, 0x12, 0x16),
		}
	}
}

impl Theme {
	/// Returns the semantic palette for a terminal background appearance.
	pub const fn for_appearance(appearance: Appearance) -> Self {
		match appearance {
			Appearance::Dark => Self {
				fg:        Color::Rgb(0xc8, 0xcc, 0xd4),
				accent:    Color::Rgb(0x61, 0xaf, 0xef),
				info:      Color::Rgb(0x56, 0xb6, 0xc2),
				ok:        Color::Rgb(0x98, 0xc3, 0x79),
				warn:      Color::Rgb(0xe5, 0xc0, 0x7b),
				err:       Color::Rgb(0xe0, 0x6c, 0x75),
				muted:     Color::Rgb(0x5c, 0x63, 0x70),
				border:    Color::Rgb(0x45, 0x4b, 0x58),
				surface:   Color::Rgb(0x3a, 0x3f, 0x4b),
				hover:     Color::Rgb(0x2c, 0x31, 0x3a),
				selection: Color::Rgb(0x36, 0x4c, 0x61),
				shadow:    Color::Rgb(0x05, 0x07, 0x0c),
				contrast:  Color::Rgb(0x10, 0x12, 0x16),
			},
			Appearance::Light => Self {
				fg:        Color::Rgb(0x24, 0x28, 0x30),
				accent:    Color::Rgb(0x00, 0x5f, 0xaf),
				info:      Color::Rgb(0x00, 0x72, 0x7d),
				ok:        Color::Rgb(0x3f, 0x70, 0x19),
				warn:      Color::Rgb(0x8a, 0x5a, 0x00),
				err:       Color::Rgb(0xb0, 0x24, 0x32),
				muted:     Color::Rgb(0x6b, 0x70, 0x78),
				border:    Color::Rgb(0xd0, 0xd7, 0xde),
				surface:   Color::Rgb(0xe2, 0xe5, 0xea),
				hover:     Color::Rgb(0xed, 0xef, 0xf2),
				selection: Color::Rgb(0xc2, 0xda, 0xed),
				shadow:    Color::Rgb(0xb8, 0xbd, 0xc7),
				contrast:  Color::Rgb(0xff, 0xff, 0xff),
			},
		}
	}

	/// Resolves a semantic token name (`accent`, `warn`, …) or a CSS
	/// system color keyword (`Canvas`, `LinkText`, …) to its color.
	pub(crate) fn token(&self, name: &str) -> Option<Color> {
		Some(match name {
			"fg" => self.fg,
			"accent" => self.accent,
			"info" => self.info,
			"ok" => self.ok,
			"warn" => self.warn,
			"err" => self.err,
			"muted" => self.muted,
			"border" => self.border,
			"surface" => self.surface,
			"hover" => self.hover,
			"selection" => self.selection,
			"shadow" => self.shadow,
			"contrast" => self.contrast,
			_ => return SystemColor::parse(name).map(|system| system.resolve(self)),
		})
	}

	/// Whether `name` resolves via [`Self::token`] on every theme.
	pub(crate) fn is_token(name: &str) -> bool {
		matches!(
			name,
			"fg"
				| "accent"
				| "info" | "ok"
				| "warn" | "err"
				| "muted"
				| "border"
				| "surface"
				| "hover"
				| "selection"
				| "shadow"
				| "contrast"
		) || SystemColor::parse(name).is_some()
	}
}

/// Terminal image rendering capability.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Graphics {
	/// Render images as colored half-block text cells.
	#[default]
	Cells,
	/// Render registered images with the DEC sixel protocol.
	Sixel,
	/// Render registered images with cursor-positioned Kitty placements.
	KittyDirect,
	/// Render registered images with Kitty Unicode placeholders.
	KittyPlaceholders,
	/// Render registered images with the iTerm2 inline-image protocol.
	Iterm2,
}

/// Presentation context threaded through parse, layout, and paint.
#[derive(Clone, Debug)]
pub struct UiContext {
	/// Terminal-reported dark or light background appearance.
	pub appearance:   Appearance,
	/// Glyph capability tier.
	pub charset:      Charset,
	/// Terminal image rendering capability.
	pub graphics:     Graphics,
	/// Pixel-capable presenter: components emit [`Decor`](crate::Decor)
	/// primitives instead of border/fill glyphs.
	pub native_decor: bool,
	/// Hangul Compatibility Jamo width policy.
	///
	/// Prefer [`UiContext::set_jamo_width`] over direct assignment: the method
	/// also updates the process-wide hot-path setting and invalidates width
	/// caches.
	pub jamo_width:   JamoWidth,
	/// Semantic color palette.
	pub theme:        Theme,
	/// Custom element registry.
	pub elements:     Elements,
	/// Presentation clock of the pass in flight: [`crate::Ui::tick`] advances
	/// it so size transitions can be sampled during layout, where no
	/// [`crate::PaintCtx`] exists. Excluded from equality — a moving clock
	/// must never read as a context change.
	pub now:          std::time::Duration,
	/// Cache-invalidation revision, advanced by [`crate::Ui::set_context`]
	/// when a differing context is applied. Geometry and render memos fold
	/// it into their keys so output derived from the previous context is
	/// discarded. Excluded from equality, like the clock.
	pub revision:     u64,
	/// Off-thread image decoder. `None` decodes inline during layout for
	/// deterministic tests and bare synchronous hosts. [`crate::App`] installs
	/// one before building the [`crate::Ui`].
	pub loader:       Option<crate::ImageLoader>,
}

impl Default for UiContext {
	fn default() -> Self {
		Self {
			appearance:   Appearance::default(),
			charset:      Charset::default(),
			graphics:     Graphics::default(),
			native_decor: false,
			jamo_width:   crate::rich::jamo_width(),
			theme:        Theme::default(),
			elements:     Elements::default(),
			now:          std::time::Duration::default(),
			revision:     0,
			loader:       None,
		}
	}
}

impl UiContext {
	/// Applies a Hangul Compatibility Jamo policy process-wide.
	///
	/// Returns whether the effective configuration changed. Width-derived
	/// caches observe that change through [`crate::rich::width_config_epoch`].
	pub fn set_jamo_width(&mut self, width: JamoWidth) -> bool {
		self.jamo_width = width;
		crate::rich::set_jamo_width(width)
	}

	/// Applies the detected terminal's capabilities: graphics tier, glyph
	/// charset, Compatibility Jamo policy, and background appearance.
	///
	/// Capability values are `0` for platform default, `1` for narrow, and `2`
	/// for wide.
	pub fn apply_terminal_caps(&mut self, caps: &crate::TerminalCaps) -> bool {
		self.graphics = caps.graphics;
		let mut changed = self.charset != caps.charset;
		self.charset = caps.charset;
		changed |= self.set_jamo_width(JamoWidth::from_caps(caps.jamo_width));
		if let Some((red, green, blue)) = caps.background {
			let appearance = Appearance::from_rgb16(red, green, blue);
			if appearance != self.appearance {
				self.appearance = appearance;
				self.theme = Theme::for_appearance(appearance);
				changed = true;
			}
		}
		changed
	}

	/// Returns this context configured for the detected terminal.
	pub fn with_terminal_caps(mut self, caps: &crate::TerminalCaps) -> Self {
		self.apply_terminal_caps(caps);
		self
	}
}

impl PartialEq for UiContext {
	fn eq(&self, other: &Self) -> bool {
		self.charset == other.charset
			&& self.appearance == other.appearance
			&& self.graphics == other.graphics
			&& self.native_decor == other.native_decor
			&& self.jamo_width == other.jamo_width
			&& self.theme == other.theme
			&& self.elements.ptr_eq(&other.elements)
	}
}

impl Eq for UiContext {}

#[cfg(test)]
mod tests {
	use super::{Appearance, Theme};

	#[test]
	fn bt601_classifies_boundary_colors_at_both_component_depths() {
		assert_eq!(Appearance::from_rgb8(0, 0, 0), Appearance::Dark);
		assert_eq!(Appearance::from_rgb8(255, 255, 255), Appearance::Light);
		assert_eq!(Appearance::from_rgb8(127, 127, 127), Appearance::Dark);
		assert_eq!(Appearance::from_rgb8(128, 128, 128), Appearance::Light);
		assert_eq!(Appearance::from_rgb16(0, 0, 0), Appearance::Dark);
		assert_eq!(Appearance::from_rgb16(u16::MAX, u16::MAX, u16::MAX), Appearance::Light);
		assert_eq!(Appearance::from_rgb16(0x7fff, 0x7fff, 0x7fff), Appearance::Dark);
		assert_eq!(Appearance::from_rgb16(0x8000, 0x8000, 0x8000), Appearance::Light);
	}

	#[test]
	fn appearance_palettes_are_distinct_and_cover_every_token() {
		let dark = Theme::for_appearance(Appearance::Dark);
		let light = Theme::for_appearance(Appearance::Light);
		assert_ne!(dark, light);
		for token in [
			"fg", "accent", "info", "ok", "warn", "err", "muted", "surface", "hover", "shadow",
			"contrast",
		] {
			assert!(dark.token(token).is_some(), "dark palette misses {token}");
			assert!(light.token(token).is_some(), "light palette misses {token}");
		}
	}
}
