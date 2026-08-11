//! Typed component properties with allocation-free well-known slots.

use std::{fmt, time::Duration};

use omp_core::{SparseMap, Str, sparse_index::TrySparseIndex};
use strum::{Display, EnumIter, EnumString, FromRepr};

use crate::{
	anim::Easing,
	context::Theme,
	frame::{Color, Style},
	markup::{Align, Border, Dim, Justify, Truncate, VAlign},
};

/// A well-known component property.
///
/// The markup attribute name of every variant is its kebab-cased ident
/// (`PadX` ⇒ `pad-x`), parsed by [`Props::prop_of`] and emitted by
/// `Display`; `VAlign` alone overrides the derived `v-align` to keep the
/// established `valign`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Display, EnumIter, EnumString, FromRepr)]
#[strum(serialize_all = "kebab-case")]
pub enum Prop {
	/// Space between adjacent children.
	Gap,
	/// Shorthand padding applied to both axes.
	Pad,
	/// Horizontal inner padding.
	PadX,
	/// Vertical inner padding.
	PadY,
	/// Flexible share of remaining layout space.
	Grow,
	/// Preferred width in cells or percent.
	W,
	/// Minimum width or numeric field value.
	Min,
	/// Maximum width or numeric field value.
	Max,
	/// Preferred height in rows.
	H,
	/// Border glyph family.
	Border,
	/// Border color or gradient.
	Bc,
	/// Alternate name for the border color or gradient.
	Edge,
	/// Extends the background through border cells.
	Bleed,
	/// Display title for a container or step.
	Title,
	/// Horizontal placement of the border title.
	TitleAlign,
	/// Display footer on a framed container's bottom edge.
	Footer,
	/// Horizontal placement of the border footer.
	FooterAlign,
	/// Horizontal content alignment.
	Align,
	/// Vertical content alignment.
	#[strum(serialize = "valign")]
	VAlign,
	/// Distribution of children along the layout axis.
	Justify,
	/// Foreground color, theme token, or gradient.
	Fg,
	/// Background color, theme token, or gradient.
	Bg,
	/// Shorthand background color, theme token, or gradient.
	On,
	/// Enables bold text.
	Bold,
	/// Enables dim text.
	Dim,
	/// Enables italic text.
	Italic,
	/// Enables underlined text.
	Underline,
	/// Swaps foreground and background colors.
	Reverse,
	/// Enables struck-through text.
	Strike,
	/// Enables wrapping rows; on text, a value selects the wrapping mode.
	Wrap,
	/// Enables text truncation.
	Truncate,
	/// Crops transparent image margins before cell sampling.
	Trim,
	/// Stable identifier used by updates and conditions.
	Id,
	/// Visibility condition referencing another component value.
	When,
	/// Initial or submitted field value.
	Value,
	/// Space-delimited choices for a selection field.
	Options,
	/// User-facing field or item label.
	Label,
	/// Supporting description for an option.
	Desc,
	/// Field control kind.
	Kind,
	/// Numeric increment or wizard step metadata.
	Step,
	/// Enables multiple selection.
	Multi,
	/// Enables interactive option filtering.
	Filter,
	/// Allows values outside the listed options.
	Custom,
	/// Obscures input contents.
	Mask,
	/// Marks an option as the recommended default.
	Recommended,
	/// Expands a tree node initially.
	Open,
	/// Requires a nonempty field value.
	Required,
	/// Pattern that a field value must satisfy.
	Match,
	/// Image or external content source.
	Src,
	/// Leading icon name.
	Icon,
	/// Compact status label.
	Badge,
	/// Emits a submit event when activated.
	Submit,
	/// Emits a cancel event when activated.
	Cancel,
	/// Requires a second activation before committing.
	Confirm,
	/// Hint shown by an empty input.
	Placeholder,
	/// Gradient direction in screen degrees.
	Angle,
	/// Applies accent styling to an action.
	Accent,
	/// Selects vertical rendering where supported.
	Vertical,
	/// Transition duration for animatable properties.
	Anim,
	/// Easing curve applied to `anim` transitions.
	Ease,
	/// Gradient rotation period.
	Spin,
	/// Border color or gradient applied while the pointer rests on the
	/// component or one of its descendants.
	Hover,
	/// Rows the component rises toward while hovered.
	Lift,
	/// Opts the component into the keyboard focus ring.
	Focus,
	/// Tree guide connector family; a bare flag selects the square set.
	Guides,
	/// Task lifecycle state on a todo item.
	Status,
	/// Sweep period of the brightness crest across text content.
	Shimmer,
	/// Catch-up horizon for progressively revealed streamed text.
	Reveal,
}

/// Invalid numeric property discriminant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PropIndexError(usize);

impl fmt::Display for PropIndexError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "invalid property index {}", self.0)
	}
}
impl std::error::Error for PropIndexError {}

impl TrySparseIndex for Prop {
	type Error = PropIndexError;

	fn index(&self) -> usize {
		*self as usize
	}

	fn try_from_index(index: usize) -> Result<Self, Self::Error> {
		u8::try_from(index)
			.ok()
			.and_then(Self::from_repr)
			.ok_or(PropIndexError(index))
	}
}

/// A parsed component property value.
#[derive(Clone, Debug, PartialEq)]
pub enum PropValue {
	/// Boolean flag.
	Bool(bool),
	/// Unsigned cell count or angle.
	U16(u16),
	/// Floating-point layout weight.
	F32(f32),
	/// Signed numeric field value.
	I64(i64),
	/// Resolved terminal color.
	Color(Color),
	/// Theme color token resolved at render time.
	Token(Str),
	/// A validated `start..end` ramp resolved by the renderer's theme.
	Gradient(Str),
	/// Cell or percentage dimension.
	Dim(Dim),
	/// Border glyph family.
	Border(Border),
	/// Horizontal alignment.
	Align(Align),
	/// Vertical alignment.
	VAlign(VAlign),
	/// Child distribution along the layout axis.
	Justify(Justify),
	/// Uninterpreted textual value.
	Str(Str),
	/// Easing curve for `anim` transitions.
	Easing(Easing),
}

/// A property value rejected by the key-aware parser.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PropError {
	pub prop:  Prop,
	pub value: Str,
}

impl fmt::Display for PropError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "bad value {:?} for property {:?}", self.value, self.prop)
	}
}
impl std::error::Error for PropError {}

/// Typed component attributes.
#[derive(Clone, Debug, Default)]
pub struct Props {
	known:  SparseMap<Prop, PropValue>,
	custom: Vec<(Str, PropValue)>,
}

impl Props {
	/// Creates an empty property collection.
	pub fn new() -> Self {
		Self::default()
	}

	/// Returns this collection with a known property assigned.
	///
	/// # Panics
	///
	/// Panics when a textual value is invalid for the selected property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.set(prop, value);
		self
	}

	/// Assigns a known property.
	///
	/// # Panics
	///
	/// Panics when a textual value is invalid for the selected property.
	pub fn set(&mut self, prop: Prop, value: impl Into<PropValue>) {
		if let Err(error) = self.try_set(prop, value.into()) {
			panic!("{error}")
		}
	}

	/// Validates and assigns a known property.
	///
	/// # Errors
	///
	/// Returns `PropError` when a textual value cannot be parsed for the
	/// selected property.
	pub fn try_set(&mut self, prop: Prop, value: PropValue) -> Result<(), PropError> {
		if prop == Prop::Pad
			&& let PropValue::Str(value) = &value
		{
			let mut parts = value.split_whitespace();
			let y = parts
				.next()
				.unwrap_or("0")
				.parse()
				.map_err(|_| PropError { prop, value: value.clone() })?;
			let x = match parts.next() {
				Some(part) => part
					.parse()
					.map_err(|_| PropError { prop, value: value.clone() })?,
				None => y,
			};
			if parts.next().is_some() {
				return Err(PropError { prop, value: value.clone() });
			}
			self.known.insert(Prop::PadY, PropValue::U16(y));
			self.known.insert(Prop::PadX, PropValue::U16(x));
			return Ok(());
		}
		let value = match value {
			PropValue::Str(value) => parse_str(prop, value)?,
			value => value,
		};
		self.known.insert(prop, value);
		Ok(())
	}

	/// Returns the typed value assigned to a known property.
	pub fn get(&self, prop: Prop) -> Option<&PropValue> {
		self.known.get(prop)
	}

	/// Removes a known property, restoring its unset default.
	pub fn unset(&mut self, prop: Prop) {
		self.known.remove(prop);
	}

	/// Formats a known property value using its markup representation.
	pub fn get_str(&self, prop: Prop) -> Option<Str> {
		self.get(prop).map(display_value)
	}

	/// Returns this collection with a custom property assigned.
	pub fn with_custom(mut self, name: impl Into<Str>, value: impl Into<PropValue>) -> Self {
		self.set_custom(name, value);
		self
	}

	/// Assigns or replaces a custom property.
	pub fn set_custom(&mut self, name: impl Into<Str>, value: impl Into<PropValue>) {
		let name = name.into();
		let value = value.into();
		if let Some((_, stored)) = self.custom.iter_mut().find(|(key, _)| key == &name) {
			*stored = value;
		} else {
			self.custom.push((name, value));
		}
	}

	/// Returns a custom property by its literal name.
	pub fn custom(&self, name: &str) -> Option<&PropValue> {
		self
			.custom
			.iter()
			.find(|(key, _)| key == name)
			.map(|(_, value)| value)
	}

	/// Returns either a known or custom property by markup name.
	pub fn named(&self, name: &str) -> Option<&PropValue> {
		Self::prop_of(name)
			.and_then(|prop| self.get(prop))
			.or_else(|| self.custom(name))
	}

	/// Resolves a markup attribute name to its well-known property — the
	/// kebab-cased variant ident derived on [`Prop`].
	pub fn prop_of(name: &str) -> Option<Prop> {
		name.parse().ok()
	}

	/// Returns the inter-child spacing, defaulting to zero.
	pub fn gap(&self) -> u16 {
		self.u16(Prop::Gap).unwrap_or(0)
	}

	/// Returns vertical and horizontal padding, defaulting to zero.
	pub fn pad(&self) -> (u16, u16) {
		(self.u16(Prop::PadY).unwrap_or(0), self.u16(Prop::PadX).unwrap_or(0))
	}

	/// Returns the flexible growth weight, with a bare flag meaning one.
	pub fn grow(&self) -> Option<f32> {
		match self.get(Prop::Grow) {
			Some(PropValue::F32(value)) => Some(*value),
			Some(PropValue::Bool(true)) => Some(1.0),
			_ => None,
		}
	}

	/// Returns the preferred width as a cell or percentage dimension.
	pub fn w(&self) -> Option<Dim> {
		match self.get(Prop::W) {
			Some(PropValue::U16(value)) => Some(Dim::Cells(*value)),
			Some(PropValue::Dim(value)) => Some(*value),
			_ => None,
		}
	}

	/// Returns the minimum width or numeric value.
	pub fn min(&self) -> Option<u16> {
		self.u16(Prop::Min)
	}

	/// Returns the maximum width or numeric value.
	pub fn max(&self) -> Option<u16> {
		self.u16(Prop::Max)
	}

	/// Returns the preferred height in rows.
	pub fn h(&self) -> Option<u16> {
		self.u16(Prop::H)
	}

	/// Returns the truncation mode: a bare `truncate` flag clips the end,
	/// `truncate=start` clips the beginning; `None` disables truncation.
	pub fn truncate(&self) -> Option<Truncate> {
		match self.get(Prop::Truncate) {
			Some(PropValue::Bool(true)) => Some(Truncate::End),
			Some(PropValue::Str(side)) if side == "start" => Some(Truncate::Start),
			Some(PropValue::Str(_)) => Some(Truncate::End),
			_ => None,
		}
	}

	/// Whether text flows grapheme-exact to the width (`wrap=char`) like a
	/// bare terminal: every break is a byte-preserving soft wrap the
	/// renderer re-joins for native copy. Defaults to word wrapping.
	pub fn wrap_chars(&self) -> bool {
		matches!(self.get(Prop::Wrap), Some(PropValue::Str(mode)) if mode == "char")
	}

	/// Returns the selected border glyph family.
	pub fn border(&self) -> Option<Border> {
		match self.get(Prop::Border) {
			Some(PropValue::Border(value)) => Some(*value),
			_ => None,
		}
	}

	/// Returns the tree guide connector family; a bare flag means square.
	pub fn guides(&self) -> Option<Border> {
		match self.get(Prop::Guides) {
			Some(PropValue::Border(value)) => Some(*value),
			Some(PropValue::Bool(true)) => Some(Border::Square),
			_ => None,
		}
	}

	/// Reports whether the background extends through the border.
	pub fn bleed(&self) -> bool {
		self.flag(Prop::Bleed)
	}

	/// Returns horizontal alignment, defaulting to the start edge.
	pub fn align(&self) -> Align {
		self.align_slot(Prop::Align)
	}

	/// Returns the border-title placement, defaulting to the start edge.
	pub fn title_align(&self) -> Align {
		self.align_slot(Prop::TitleAlign)
	}

	/// Returns the border-footer placement, defaulting to the start edge.
	pub fn footer_align(&self) -> Align {
		self.align_slot(Prop::FooterAlign)
	}

	fn align_slot(&self, prop: Prop) -> Align {
		match self.get(prop) {
			Some(PropValue::Align(value)) => *value,
			_ => Align::Start,
		}
	}

	/// Returns the configured vertical alignment.
	pub fn valign(&self) -> Option<VAlign> {
		match self.get(Prop::VAlign) {
			Some(PropValue::VAlign(value)) => Some(*value),
			_ => None,
		}
	}

	/// Returns the stable component identifier.
	pub fn id(&self) -> Option<&Str> {
		self.str_of(Prop::Id)
	}

	/// Returns the user-facing component title.
	pub fn title(&self) -> Option<&Str> {
		self.str_of(Prop::Title)
	}

	/// Returns the footer shown on a framed container's bottom border.
	pub fn footer(&self) -> Option<&Str> {
		self.str_of(Prop::Footer)
	}

	/// Normalized gradient direction in screen degrees.
	pub fn angle(&self) -> u16 {
		self.u16(Prop::Angle).unwrap_or(0)
	}

	/// Transition duration for animatable properties, when `anim` is set.
	/// A bare `anim` flag selects 200ms.
	pub fn anim(&self) -> Option<Duration> {
		self.duration(Prop::Anim, 200)
	}

	/// Easing curve for `anim` transitions, defaulting to ease-out — the
	/// natural shape for state changes that should land softly.
	pub fn ease(&self) -> Easing {
		match self.get(Prop::Ease) {
			Some(PropValue::Easing(value)) => *value,
			_ => Easing::EaseOut,
		}
	}

	/// Gradient rotation period, when `spin` is set. A bare `spin` flag
	/// selects one revolution every 3 seconds.
	pub fn spin(&self) -> Option<Duration> {
		self.duration(Prop::Spin, 3000)
	}

	/// Brightness-crest sweep period, when `shimmer` is set. A bare
	/// `shimmer` flag selects one sweep every 2 seconds.
	pub fn shimmer(&self) -> Option<Duration> {
		self.duration(Prop::Shimmer, 2000)
	}

	/// Streamed-text reveal catch-up horizon, when `reveal` is set. A bare
	/// `reveal` flag selects 250ms; `reveal=0` shows new text immediately.
	pub fn reveal(&self) -> Option<Duration> {
		self.duration(Prop::Reveal, 250)
	}

	/// Rows of hover elevation, with a bare flag meaning one.
	pub fn lift(&self) -> u16 {
		match self.get(Prop::Lift) {
			Some(PropValue::U16(value)) => *value,
			Some(PropValue::Bool(true)) => 1,
			_ => 0,
		}
	}

	/// Whether hover styling or elevation is declared — the gate for
	/// registering a pointer zone and resolving hover chrome at paint time.
	pub(crate) fn hover_decorated(&self) -> bool {
		self.get(Prop::Hover).is_some() || self.lift() > 0
	}

	fn duration(&self, prop: Prop, default_ms: u64) -> Option<Duration> {
		match self.get(prop)? {
			PropValue::U16(ms) => Some(Duration::from_millis(u64::from(*ms))),
			PropValue::Bool(true) => Some(Duration::from_millis(default_ms)),
			_ => None,
		}
	}

	pub(crate) fn gradient_of(&self, prop: Prop) -> Option<&Str> {
		match self.get(prop) {
			Some(PropValue::Gradient(value)) => Some(value),
			_ => None,
		}
	}

	/// Reports whether a boolean property is enabled.
	pub fn flag(&self, prop: Prop) -> bool {
		matches!(self.get(prop), Some(PropValue::Bool(true)))
	}

	/// Returns the textual payload of a property.
	pub fn str_of(&self, prop: Prop) -> Option<&Str> {
		match self.get(prop) {
			Some(PropValue::Str(value)) => Some(value),
			_ => None,
		}
	}

	/// Resolves colors and text attributes into a render style.
	pub fn style(&self, theme: &Theme) -> Style {
		let mut style = Style::new();
		if let Some(color) = self.color(Prop::Fg, theme) {
			style = style.fg(color);
		}
		let background = if self.get(Prop::Bg).is_some() {
			self.color(Prop::Bg, theme)
		} else {
			self.color(Prop::On, theme)
		};
		if let Some(color) = background {
			style = style.bg(color);
		}
		if self.flag(Prop::Bold) {
			style = style.bold();
		}
		if self.flag(Prop::Dim) {
			style = style.dim();
		}
		if self.flag(Prop::Italic) {
			style = style.italic();
		}
		if self.flag(Prop::Underline) {
			style = style.underline();
		}
		if self.flag(Prop::Reverse) {
			style = style.reverse();
		}
		if self.flag(Prop::Strike) {
			style = style.strikethrough();
		}
		style
	}

	/// Resolves the border color from either supported attribute name.
	pub fn edge(&self, theme: &Theme) -> Option<Color> {
		self
			.color(Prop::Bc, theme)
			.or_else(|| self.color(Prop::Edge, theme))
	}

	fn u16(&self, prop: Prop) -> Option<u16> {
		match self.get(prop) {
			Some(PropValue::U16(value)) => Some(*value),
			_ => None,
		}
	}

	fn color(&self, prop: Prop, theme: &Theme) -> Option<Color> {
		match self.get(prop) {
			Some(PropValue::Color(value)) => Some(*value),
			Some(PropValue::Token(value)) => theme.token(value),
			_ => None,
		}
	}
}

fn parse_str(prop: Prop, value: Str) -> Result<PropValue, PropError> {
	let bad = || PropError { prop, value: value.clone() };
	Ok(match prop {
		Prop::Gap | Prop::PadX | Prop::PadY | Prop::Min | Prop::Max | Prop::H | Prop::Lift => {
			PropValue::U16(value.parse().map_err(|_| bad())?)
		},
		Prop::W => {
			if let Some(percent) = value.strip_suffix("%") {
				PropValue::Dim(Dim::Pct(percent.parse().map_err(|_| bad())?))
			} else {
				PropValue::U16(value.parse().map_err(|_| bad())?)
			}
		},
		Prop::Grow => PropValue::F32(value.parse().map_err(|_| bad())?),
		Prop::Step => PropValue::I64(value.parse().map_err(|_| bad())?),
		Prop::Border | Prop::Guides => PropValue::Border(match value.as_str() {
			"square" => Border::Square,
			"dash" => Border::Dash,
			"round" => Border::Round,
			"heavy" => Border::Heavy,
			"double" => Border::Double,
			_ => return Err(bad()),
		}),
		Prop::Align | Prop::TitleAlign | Prop::FooterAlign => {
			PropValue::Align(match value.as_str() {
				"start" | "left" => Align::Start,
				"center" | "middle" => Align::Center,
				"end" | "right" => Align::End,
				_ => return Err(bad()),
			})
		},
		Prop::VAlign => PropValue::VAlign(match value.as_str() {
			"start" | "top" => VAlign::Start,
			"center" | "middle" => VAlign::Center,
			"end" | "bottom" => VAlign::End,
			"stretch" | "fill" => VAlign::Stretch,
			_ => return Err(bad()),
		}),
		Prop::Justify => PropValue::Justify(match value.as_str() {
			"start" => Justify::Start,
			"center" => Justify::Center,
			"end" => Justify::End,
			"between" => Justify::Between,
			_ => return Err(bad()),
		}),
		Prop::Angle => PropValue::U16(parse_angle(&value).ok_or_else(bad)?),
		Prop::Anim | Prop::Spin | Prop::Shimmer | Prop::Reveal => {
			PropValue::U16(parse_duration_ms(&value).ok_or_else(bad)?)
		},
		// `truncate` keeps its bare-flag form (end clipping); a value picks
		// the clipped side.
		Prop::Truncate => match value.as_str() {
			"start" | "end" => PropValue::Str(value),
			_ => return Err(bad()),
		},
		// `wrap` stays a bare flag (wrapping rows); on text a value picks
		// the mode — `wrap=char` flows grapheme-exact like a bare terminal.
		Prop::Wrap => match value.as_str() {
			"char" | "word" => PropValue::Str(value),
			_ => return Err(bad()),
		},
		// A textual `filter` seeds the initial query besides enabling
		// filtering; the bare flag stays a boolean.
		Prop::Filter => PropValue::Str(value),
		Prop::Ease => PropValue::Easing(match value.as_str() {
			"linear" => Easing::Linear,
			"in" => Easing::EaseIn,
			"out" => Easing::EaseOut,
			"in-out" => Easing::EaseInOut,
			_ => return Err(bad()),
		}),
		Prop::Fg | Prop::Bg | Prop::On | Prop::Bc | Prop::Edge | Prop::Hover => {
			if is_gradient(&value) {
				PropValue::Gradient(value)
			} else if is_theme_token(&value) {
				PropValue::Token(value)
			} else if let Some(color) = Color::parse(&value) {
				PropValue::Color(color)
			} else {
				return Err(bad());
			}
		},
		Prop::Bold
		| Prop::Dim
		| Prop::Italic
		| Prop::Underline
		| Prop::Reverse
		| Prop::Strike
		| Prop::Trim
		| Prop::Bleed
		| Prop::Multi
		| Prop::Custom
		| Prop::Mask
		| Prop::Recommended
		| Prop::Open
		| Prop::Required
		| Prop::Submit
		| Prop::Cancel
		| Prop::Confirm
		| Prop::Accent
		| Prop::Vertical
		| Prop::Focus => PropValue::Bool(true),
		_ => PropValue::Str(value),
	})
}

fn is_theme_token(value: &str) -> bool {
	Theme::is_token(value)
}

fn is_gradient(value: &str) -> bool {
	let Some((start, end)) = value.split_once("..") else {
		return false;
	};
	is_color(start) && is_color(end)
}

fn is_color(value: &str) -> bool {
	is_theme_token(value) || Color::parse(value).is_some()
}

fn parse_angle(value: &str) -> Option<u16> {
	let value = value.trim();
	let value = value.strip_suffix("deg").unwrap_or(value);
	Some(value.parse::<i32>().ok()?.rem_euclid(360) as u16)
}

/// Parses `250`, `250ms`, or `0.4s` into whole milliseconds.
fn parse_duration_ms(value: &str) -> Option<u16> {
	let value = value.trim();
	if let Some(millis) = value.strip_suffix("ms") {
		return millis.trim().parse().ok();
	}
	if let Some(seconds) = value.strip_suffix('s') {
		let seconds: f32 = seconds.trim().parse().ok()?;
		if !(0.0..=65.0).contains(&seconds) {
			return None;
		}
		return Some((seconds * 1000.0).round() as u16);
	}
	value.parse().ok()
}

fn display_value(value: &PropValue) -> Str {
	match value {
		PropValue::Bool(value) => Str::new(if *value { "true" } else { "false" }),
		PropValue::U16(value) => Str::from(value.to_string()),
		PropValue::F32(value) => Str::from(value.to_string()),
		PropValue::I64(value) => Str::from(value.to_string()),
		PropValue::Color(Color::Default) => Str::new_static("default"),
		PropValue::Color(Color::Indexed(value)) => Str::from(value.to_string()),
		PropValue::Color(Color::Rgb(r, g, b)) => Str::from(format!("#{r:02x}{g:02x}{b:02x}")),
		PropValue::Token(value) | PropValue::Gradient(value) | PropValue::Str(value) => value.clone(),
		PropValue::Easing(value) => Str::new_static(match value {
			Easing::Linear => "linear",
			Easing::EaseIn => "in",
			Easing::EaseOut => "out",
			Easing::EaseInOut => "in-out",
		}),
		PropValue::Dim(Dim::Cells(value)) => Str::from(value.to_string()),
		PropValue::Dim(Dim::Pct(value)) => Str::from(format!("{value}%")),
		PropValue::Border(value) => Str::new_static(match value {
			Border::Square => "square",
			Border::Dash => "dash",
			Border::Round => "round",
			Border::Heavy => "heavy",
			Border::Double => "double",
		}),
		PropValue::Align(value) => Str::new_static(match value {
			Align::Start => "start",
			Align::Center => "center",
			Align::End => "end",
		}),
		PropValue::VAlign(value) => Str::new_static(match value {
			VAlign::Start => "start",
			VAlign::Center => "center",
			VAlign::End => "end",
			VAlign::Stretch => "stretch",
		}),
		PropValue::Justify(value) => Str::new_static(match value {
			Justify::Start => "start",
			Justify::Center => "center",
			Justify::End => "end",
			Justify::Between => "between",
		}),
	}
}

macro_rules! from_value {
	($type:ty, $variant:ident) => {
		impl From<$type> for PropValue {
			fn from(value: $type) -> Self {
				Self::$variant(value)
			}
		}
	};
}
from_value!(Color, Color);
from_value!(bool, Bool);
from_value!(u16, U16);
from_value!(f32, F32);
from_value!(i64, I64);
from_value!(Str, Str);
from_value!(Dim, Dim);
from_value!(Border, Border);
from_value!(Align, Align);
from_value!(VAlign, VAlign);
from_value!(Justify, Justify);
from_value!(Easing, Easing);
impl From<&str> for PropValue {
	fn from(value: &str) -> Self {
		Self::Str(Str::new(value))
	}
}
impl From<String> for PropValue {
	fn from(value: String) -> Self {
		Self::Str(value.into())
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn known_values_parse_at_set_time() {
		assert_eq!(
			Props::new().with(Prop::Fg, "blue").get(Prop::Fg),
			Some(&PropValue::Color(Color::Rgb(0, 0, 255)))
		);
		assert_eq!(
			Props::new().with(Prop::Fg, "accent").get(Prop::Fg),
			Some(&PropValue::Token(Str::new("accent")))
		);
		assert_eq!(
			Props::new().with(Prop::Title, "x").get(Prop::Title),
			Some(&PropValue::Str(Str::new("x")))
		);
	}

	#[test]
	fn gradients_and_angles_use_standard_color_properties() {
		let props = Props::new()
			.with(Prop::Bg, "accent..info")
			.with(Prop::Fg, "#000000..#ffffff")
			.with(Prop::Angle, "-90deg");
		assert_eq!(props.get(Prop::Bg), Some(&PropValue::Gradient(Str::new("accent..info"))));
		assert_eq!(props.get(Prop::Fg), Some(&PropValue::Gradient(Str::new("#000000..#ffffff"))));
		assert_eq!(props.angle(), 270);
		assert!(Props::prop_of("gradient").is_none());
		assert!(Props::prop_of("dir").is_none());
	}

	#[test]
	#[should_panic(expected = "nosuch")]
	fn invalid_known_value_panics() {
		let _ = Props::new().with(Prop::Fg, "nosuch");
	}

	#[test]
	fn invalid_known_value_is_fallible() {
		let mut props = Props::new();
		assert!(props.try_set(Prop::Fg, PropValue::from("nosuch")).is_err());
	}

	#[test]
	fn values_format_and_customs_round_trip() {
		let props = Props::new()
			.with(Prop::Gap, 2_u16)
			.with_custom("data-x", "1");
		assert_eq!(props.get_str(Prop::Gap).as_deref(), Some("2"));
		assert_eq!(props.custom("data-x"), Some(&PropValue::Str(Str::new("1"))));
		assert_eq!(props.named("data-x"), props.custom("data-x"));
	}

	#[test]
	fn style_resolves_tokens_at_read_time() {
		let theme = Theme { accent: Color::Rgb(1, 2, 3), ..Theme::default() };
		let props = Props::new().with(Prop::Fg, "accent").with(Prop::Bold, true);
		assert_eq!(props.style(&theme).foreground_color(), Color::Rgb(1, 2, 3));
		assert_eq!(props.get(Prop::Bold), Some(&PropValue::Bool(true)));
		assert!(props.flag(Prop::Bold));
		assert!(!Props::new().with(Prop::Bold, false).flag(Prop::Bold));
	}

	#[test]
	fn anim_props_parse_durations_and_easing() {
		let mut props = Props::new();
		props.set(Prop::Anim, "150ms");
		assert_eq!(props.anim(), Some(Duration::from_millis(150)));
		props.set(Prop::Anim, "0.4s");
		assert_eq!(props.anim(), Some(Duration::from_millis(400)));
		props.set(Prop::Anim, "250");
		assert_eq!(props.anim(), Some(Duration::from_millis(250)));
		props.set(Prop::Spin, "2s");
		assert_eq!(props.spin(), Some(Duration::from_millis(2000)));
		props.set(Prop::Shimmer, "1.5s");
		assert_eq!(props.shimmer(), Some(Duration::from_millis(1500)));
		props.set(Prop::Reveal, "500ms");
		assert_eq!(props.reveal(), Some(Duration::from_millis(500)));

		// Bare flags pick the documented defaults; absence disables.
		let bare = Props::new()
			.with(Prop::Anim, true)
			.with(Prop::Spin, true)
			.with(Prop::Shimmer, true)
			.with(Prop::Reveal, true);
		assert_eq!(bare.anim(), Some(Duration::from_millis(200)));
		assert_eq!(bare.spin(), Some(Duration::from_millis(3000)));
		assert_eq!(bare.shimmer(), Some(Duration::from_millis(2000)));
		assert_eq!(bare.reveal(), Some(Duration::from_millis(250)));
		assert_eq!(Props::new().reveal(), None);
		assert_eq!(Props::new().anim(), None);

		// Easing defaults to ease-out and parses every token.
		assert_eq!(props.ease(), Easing::EaseOut);
		props.set(Prop::Ease, "in-out");
		assert_eq!(props.ease(), Easing::EaseInOut);
		assert_eq!(props.get_str(Prop::Ease).as_deref(), Some("in-out"));
		assert!(
			props
				.try_set(Prop::Ease, PropValue::from("bouncy"))
				.is_err()
		);
		assert!(props.try_set(Prop::Anim, PropValue::from("fast")).is_err());
		assert!(props.try_set(Prop::Spin, PropValue::from("99s")).is_err());
	}

	#[test]
	fn prop_indices_round_trip_through_the_catalog() {
		use strum::IntoEnumIterator as _;
		for (index, prop) in Prop::iter().enumerate() {
			assert_eq!(prop as usize, index, "the catalog diverges from enum order at {prop:?}");
			assert_eq!(Prop::try_from_index(index), Ok(prop));
		}
	}
}
