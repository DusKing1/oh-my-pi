use xutf::Text;

use crate::{
	component::{Component, EventCtx, Flow, Hit, HitTag, PaintCtx, Slot, next_slot},
	context::{Theme, UiContext},
	frame::{Rect, Style},
	input::{
		Key, Mouse, byte_at_column, sanitize_paste, word_left_column, word_right_column,
		word_rubout_start,
	},
	props::{Prop, PropValue, Props},
	rich::cell_width,
};

#[derive(Default)]
struct InputState {
	text:   String,
	cursor: u16,
	mask:   bool,
	masked: String,
}

impl InputState {
	fn refresh_mask(&mut self) {
		self.masked.clear();
		if self.mask {
			self
				.masked
				.reserve(self.text.chars().count().saturating_mul('•'.len_utf8()));
			self
				.masked
				.extend(std::iter::repeat_n('•', self.text.chars().count()));
		}
	}
}

/// An editable, single-line text field.
pub struct Input {
	props: Props,
	slot:  Slot,
	state: InputState,
}

impl Input {
	/// Creates an empty input.
	pub fn new() -> Self {
		Self { props: Props::new(), slot: next_slot(), state: InputState::default() }
	}

	/// Sets one input property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self.sync_prop(prop);
		self
	}

	/// Sets one input property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self.sync_prop(prop);
		self
	}

	fn sync_prop(&mut self, prop: Prop) {
		match prop {
			Prop::Value => {
				self.state.text = self
					.props
					.str_of(Prop::Value)
					.map(ToString::to_string)
					.unwrap_or_default();
				self.state.cursor = cell_width(&self.state.text);
				self.state.refresh_mask();
			},
			Prop::Mask => {
				self.state.mask = self.props.flag(Prop::Mask);
				self.state.refresh_mask();
			},
			_ => {},
		}
	}

	fn edit(&mut self, key: Key) -> bool {
		match key {
			Key::Left | Key::Ctrl('b') => self.state.cursor = self.state.cursor.saturating_sub(1),
			Key::Right | Key::Ctrl('f') => {
				self.state.cursor = (self.state.cursor + 1).min(cell_width(&self.state.text));
			},
			Key::Home => self.state.cursor = 0,
			Key::End => self.state.cursor = cell_width(&self.state.text),
			Key::Backspace => {
				if self.state.cursor > 0 {
					let end = byte_at_column(&self.state.text, self.state.cursor);
					let start = self.state.text[..end]
						.grapheme_indices()
						.next_back()
						.map_or(0, |(offset, _)| offset);
					let removed = cell_width(&self.state.text[start..end]);
					self.state.text.replace_range(start..end, "");
					self.state.cursor = self.state.cursor.saturating_sub(removed);
				}
			},
			Key::Delete | Key::Ctrl('d') => {
				let start = byte_at_column(&self.state.text, self.state.cursor);
				if start < self.state.text.len() {
					let grapheme_len = self.state.text[start..]
						.graphemes()
						.next()
						.map_or(0, str::len);
					self
						.state
						.text
						.replace_range(start..start + grapheme_len, "");
				}
			},
			Key::Space => {
				let at = byte_at_column(&self.state.text, self.state.cursor);
				self.state.text.insert(at, ' ');
				self.state.cursor += 1;
			},
			Key::Char(character) => {
				let at = byte_at_column(&self.state.text, self.state.cursor);
				self.state.text.insert(at, character);
				self.state.cursor += cell_width(character.encode_utf8(&mut [0u8; 4]));
			},
			Key::Ctrl('a') => self.state.cursor = 0,
			Key::Ctrl('e') => self.state.cursor = cell_width(&self.state.text),
			Key::Ctrl('k') => {
				let at = byte_at_column(&self.state.text, self.state.cursor);
				self.state.text.truncate(at);
			},
			Key::Ctrl('u') => {
				let at = byte_at_column(&self.state.text, self.state.cursor);
				self.state.text.replace_range(..at, "");
				self.state.cursor = 0;
			},
			Key::Ctrl('w') => {
				let end = byte_at_column(&self.state.text, self.state.cursor);
				let start = word_rubout_start(&self.state.text, end);
				let removed = cell_width(&self.state.text[start..end]);
				self.state.text.replace_range(start..end, "");
				self.state.cursor = self.state.cursor.saturating_sub(removed);
			},
			Key::WordLeft => self.state.cursor = word_left_column(&self.state.text, self.state.cursor),
			Key::WordRight => {
				self.state.cursor = word_right_column(&self.state.text, self.state.cursor);
			},
			Key::WordDelete => {
				let start = byte_at_column(&self.state.text, self.state.cursor);
				let end_column = word_right_column(&self.state.text, self.state.cursor);
				let end = byte_at_column(&self.state.text, end_column);
				self.state.text.replace_range(start..end, "");
			},
			_ => return false,
		}
		self.state.refresh_mask();
		true
	}
}

impl Default for Input {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Input {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn measure(&mut self, _ctx: &UiContext) -> (u16, u16) {
		let placeholder = self
			.props
			.str_of(Prop::Placeholder)
			.map_or(0, |placeholder| cell_width(placeholder));
		(16, placeholder.saturating_add(3).max(30))
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		1
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		if rect.y >= pc.clip {
			return;
		}
		pc.hits
			.push(Hit { rect, slot: self.slot, tag: HitTag::Press });
		let focused = pc.focus == Some(self.slot);
		let shown = if self.state.mask {
			&self.state.masked
		} else {
			&self.state.text
		};
		let x = pc.frame.put(
			rect.x,
			rect.y,
			pc.ctx.charset.cursor(),
			if focused {
				Style::new().fg(pc.ctx.theme.accent)
			} else {
				dim(&pc.ctx.theme)
			},
		);
		if shown.is_empty() && !focused {
			if let Some(placeholder) = self.props.str_of(Prop::Placeholder) {
				pc.frame
					.put(x, rect.y, placeholder, dim(&pc.ctx.theme).italic());
			}
			return;
		}
		let available = rect.width.saturating_sub(3);
		let total = cell_width(shown);
		let left = if total > available {
			self
				.state
				.cursor
				.saturating_sub(available.saturating_sub(8))
		} else {
			0
		};
		let start = byte_at_column(shown, left);
		let visible = &shown[start..];
		if focused {
			// The real terminal cursor marks the insertion point — one
			// cursor treatment across every core single-line editor.
			let split = byte_at_column(visible, self.state.cursor.saturating_sub(left));
			pc.frame.put(x, rect.y, visible, base(&pc.ctx.theme));
			pc.frame
				.set_cursor(x.saturating_add(cell_width(&visible[..split])), rect.y);
		} else {
			pc.frame.put(x, rect.y, visible, base(&pc.ctx.theme));
		}
	}

	fn focusable(&self) -> bool {
		true
	}

	fn key(&mut self, _ec: &mut EventCtx<'_>, key: Key) -> Flow {
		if self.edit(key) {
			Flow::Consumed
		} else {
			Flow::Skip
		}
	}

	fn mouse(
		&mut self,
		_ec: &mut EventCtx<'_>,
		tag: HitTag,
		at: (u16, u16),
		rect: Rect,
		mouse: Mouse,
	) -> Flow {
		match mouse {
			Mouse::Click if tag == HitTag::Press => {
				let column = at.0.saturating_sub(rect.x.saturating_add(2));
				self.state.cursor = column.min(cell_width(&self.state.text));
				Flow::Consumed
			},
			Mouse::Click
			| Mouse::RightClick
			| Mouse::MiddleClick
			| Mouse::Move
			| Mouse::Drag
			| Mouse::Release
			| Mouse::WheelUp
			| Mouse::WheelDown
			| Mouse::WheelLeft
			| Mouse::WheelRight => Flow::Skip,
		}
	}

	fn paste(&mut self, _ec: &mut EventCtx<'_>, text: &str) -> Flow {
		let sanitized = sanitize_paste(text);
		if sanitized.is_empty() {
			return Flow::Skip;
		}
		let paste = sanitized.replace(['\n', '\t'], " ");
		let at = byte_at_column(&self.state.text, self.state.cursor);
		self.state.text.insert_str(at, &paste);
		self.state.cursor = self.state.cursor.saturating_add(cell_width(&paste));
		self.state.refresh_mask();
		Flow::Consumed
	}

	fn value(&self, out: &mut serde_json::Map<String, serde_json::Value>) {
		if let Some(id) = self.props.id() {
			out.insert(id.to_string(), serde_json::Value::String(self.state.text.clone()));
		}
	}
}

const fn base(theme: &Theme) -> Style {
	Style::new().fg(theme.fg)
}
const fn dim(theme: &Theme) -> Style {
	Style::new().fg(theme.muted)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Frame, Size, test_support::frame_row_text};

	fn event_ctx(ctx: &UiContext) -> EventCtx<'_> {
		EventCtx::new(ctx, 32, 1)
	}

	#[test]
	fn key_and_paste_edit_exported_text() {
		let mut input = Input::new().with(Prop::Id, "name");
		let ctx = UiContext::default();
		assert_eq!(input.key(&mut event_ctx(&ctx), Key::Char('a')), Flow::Consumed);
		assert_eq!(input.key(&mut event_ctx(&ctx), Key::Char('b')), Flow::Consumed);
		assert_eq!(input.paste(&mut event_ctx(&ctx), " c\td"), Flow::Consumed);
		let mut values = serde_json::Map::new();
		input.value(&mut values);
		assert_eq!(values["name"], serde_json::json!("ab c d"));
	}

	#[test]
	fn paint_draws_text_and_press_hit() {
		let mut input = Input::new().with(Prop::Value, "hello");
		let ctx = UiContext::default();
		let mut frame = Frame::new(Size::new(32, 1));
		let mut hits = Vec::new();
		let slot = input.slot();
		let mut wakes = Vec::new();
		let mut pc = PaintCtx::new(&mut frame, &ctx, &mut hits, &mut wakes);
		input.paint(&mut pc, Rect::new(0, 0, 32, 1));
		assert!(frame_row_text(&frame, 0).contains("hello"));
		assert_eq!(hits[0].slot, slot);
	}
}
