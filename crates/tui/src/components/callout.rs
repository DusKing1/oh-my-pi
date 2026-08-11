use omp_core::Str;

use super::text::{append, put_clipped, truncate_rich};
use crate::{
	component::{Component, MemoKey, PaintCtx, Slot, next_slot},
	frame::{Color, Rect, Style},
	markdown::MdTheme,
	props::{Prop, PropValue, Props},
	rich::{RichText, cell_width},
};

/// A highlighted Markdown notice backing the `<callout>` markup tag.
pub struct Callout {
	props:        Props,
	slot:         Slot,
	text:         Str,
	rich:         RichText,
	version:      u64,
	cached_width: u16,
	cached:       Option<MemoKey>,
}

impl Callout {
	/// Creates an empty callout.
	pub fn new() -> Self {
		Self {
			props:        Props::new(),
			slot:         next_slot(),
			text:         Str::default(),
			rich:         RichText::default(),
			version:      1,
			cached_width: 0,
			cached:       None,
		}
	}

	/// Sets one callout property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self.version = self.version.wrapping_add(1);
		self
	}

	/// Sets one callout property from a string.
	pub fn with_str(self, prop: Prop, value: &str) -> Self {
		self.with(prop, value)
	}

	/// Appends Markdown source text.
	pub fn text(mut self, text: impl Into<Str>) -> Self {
		append(&mut self.text, text.into());
		self.version = self.version.wrapping_add(1);
		self
	}

	const fn has_header(&self) -> bool {
		self.props.title().is_some()
			|| self.props.str_of(Prop::Badge).is_some()
			|| self.props.str_of(Prop::Icon).is_some()
	}

	fn accent(&self, ctx: &crate::UiContext) -> Color {
		let color = self.props.style(&ctx.theme).foreground_color();
		if matches!(color, Color::Default) {
			ctx.theme.info
		} else {
			color
		}
	}

	fn icon<'a>(&'a self, ctx: &'a crate::UiContext) -> &'a str {
		self.props.str_of(Prop::Icon).map_or_else(
			|| ctx.charset.note_icon(),
			|name| ctx.charset.icon_named(name).unwrap_or(name),
		)
	}

	fn header_width(&self, ctx: &crate::UiContext) -> u16 {
		if !self.has_header() {
			return 0;
		}
		let icon = cell_width(self.icon(ctx)).saturating_add(1);
		let title = self.props.title().map_or(0, |title| cell_width(title));
		let badge = self
			.props
			.str_of(Prop::Badge)
			.map_or(0, |badge| cell_width(badge).saturating_add(1));
		icon.saturating_add(title).saturating_add(badge)
	}

	fn render(&mut self, ctx: &crate::UiContext, width: u16) {
		let width = width.saturating_sub(2).max(1);
		let key = MemoKey::new(self.version, ctx);
		if self.cached_width == width && self.cached == Some(key) {
			return;
		}
		let style = self.props.style(&ctx.theme);
		let theme = MdTheme::from_context(ctx).cascade(style);
		self.rich.clear();
		crate::markdown::render(&self.text, width, &theme, &mut self.rich);
		truncate_rich(&mut self.rich, width, style, self.props.truncate());
		self.cached_width = width;
		self.cached = Some(key);
	}
}

impl Default for Callout {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Callout {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn measure(&mut self, ctx: &crate::UiContext) -> (u16, u16) {
		let body = self
			.text
			.lines()
			.map(cell_width)
			.max()
			.unwrap_or(0)
			.saturating_add(2);
		(14, body.max(self.header_width(ctx)).max(16))
	}

	fn height(&mut self, ctx: &crate::UiContext, width: u16) -> u16 {
		self.render(ctx, width);
		u16::from(self.has_header()).saturating_add(RichText::rows(&self.rich))
	}

	fn place(&mut self, ctx: &crate::UiContext, content: Rect) {
		self.render(ctx, content.width);
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		self.render(pc.ctx, rect.width);
		let accent = self.accent(pc.ctx);
		let clip = pc.clip.min(rect.y.saturating_add(rect.height));
		let mut y = rect.y;
		if self.has_header() {
			if y < clip {
				let mut x = pc
					.frame
					.put(rect.x, y, self.icon(pc.ctx), Style::new().fg(accent));
				x = pc.frame.put(x, y, " ", Style::new().fg(pc.ctx.theme.fg));
				if let Some(title) = self.props.title() {
					x = pc.frame.put(x, y, title, Style::new().fg(accent).bold());
				}
				if let Some(badge) = self.props.str_of(Prop::Badge) {
					x = pc.frame.put(x, y, " ", Style::new().fg(pc.ctx.theme.fg));
					pc.frame
						.put(x, y, badge, Style::new().fg(pc.ctx.theme.muted));
				}
			}
			y = y.saturating_add(1);
		}
		let right = rect.x.saturating_add(rect.width);
		for row in 0..RichText::rows(&self.rich) {
			let line_y = y.saturating_add(row);
			if line_y >= clip {
				break;
			}
			let mut x = pc
				.frame
				.put(rect.x, line_y, pc.ctx.charset.rail(), Style::new().fg(accent));
			for (style, text) in self.rich.row_runs(row) {
				x = put_clipped(pc.frame, x, line_y, right, text, style);
				if x >= right {
					break;
				}
			}
		}
	}

	fn set_text(&mut self, _ctx: &crate::UiContext, text: Str) -> bool {
		if self.text == text {
			return false;
		}
		self.text = text;
		self.version = self.version.wrapping_add(1);
		true
	}
}
