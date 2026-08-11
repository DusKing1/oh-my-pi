use std::fmt::Write as _;

use omp_core::Str;
use serde_json::{Map, Value};
use smallvec::SmallVec;

use crate::{
	component::{Component, EventCtx, Flow, Hit, HitTag, IntoChildren, PaintCtx, Slot, next_slot},
	context::{Theme, UiContext},
	frame::{Rect, Style},
	input::{Key, Mouse, sanitize_paste, word_rubout_start},
	props::{Prop, PropValue, Props},
	rich::cell_width,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FieldKind {
	Bool,
	Enum,
	Text,
	Select,
	Multi,
	Number,
}

#[derive(Clone, Debug)]
enum FieldValue {
	Bool(bool),
	Text(String),
	Choice(Str),
	Many(SmallVec<Str, 4>),
	Number(i64),
}

/// Declarative input metadata backing the `<field>` markup tag.
pub struct Field {
	props:    Props,
	label:    Str,
	children: Vec<crate::component::Cached>,
}

impl Field {
	/// Creates an empty field definition.
	pub fn new() -> Self {
		Self { props: Props::new(), label: Str::new(""), children: Vec::new() }
	}

	/// Sets one field property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one field property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets the field's visible label.
	pub fn label(mut self, label: impl Into<Str>) -> Self {
		self.label = label.into();
		self
	}

	/// Appends field content used by richer controls.
	pub fn child(mut self, children: impl IntoChildren) -> Self {
		children.extend_children(&mut self.children);
		self
	}
}

impl Default for Field {
	fn default() -> Self {
		Self::new()
	}
}

#[derive(Clone, Debug)]
struct FieldData {
	kind:     FieldKind,
	id:       Str,
	label:    Str,
	desc:     Option<Str>,
	options:  SmallVec<Str, 8>,
	value:    FieldValue,
	required: bool,
	pattern:  Option<Str>,
	min:      i64,
	max:      i64,
	step:     i64,
}

impl FieldData {
	fn from_field(field: Field) -> Self {
		let kind = match field.props.str_of(Prop::Kind).map(Str::as_str) {
			Some("bool") => FieldKind::Bool,
			Some("enum") => FieldKind::Enum,
			Some("select") => FieldKind::Select,
			Some("multi") => FieldKind::Multi,
			Some("number") => FieldKind::Number,
			_ => FieldKind::Text,
		};
		let options: SmallVec<Str, 8> = field
			.props
			.str_of(Prop::Options)
			.map(|options| options.split_whitespace().map(Str::new).collect())
			.unwrap_or_default();
		let raw = field.props.str_of(Prop::Value);
		let value = match kind {
			FieldKind::Bool => FieldValue::Bool(raw.is_some_and(|value| value == "true")),
			FieldKind::Enum | FieldKind::Select => FieldValue::Choice(
				raw.filter(|value| options.iter().any(|option| option == *value))
					.cloned()
					.or_else(|| options.first().cloned())
					.unwrap_or_default(),
			),
			FieldKind::Multi => FieldValue::Many(
				raw.map(|value| {
					options
						.iter()
						.filter(|option| value.split_whitespace().any(|part| *option == part))
						.cloned()
						.collect()
				})
				.unwrap_or_default(),
			),
			FieldKind::Number => {
				FieldValue::Number(raw.and_then(|value| value.parse().ok()).unwrap_or(0))
			},
			FieldKind::Text => FieldValue::Text(raw.map(ToString::to_string).unwrap_or_default()),
		};
		let i64_prop = |prop| match field.props.get(prop) {
			Some(PropValue::I64(value)) => Some(value),
			Some(PropValue::U16(value)) => Some(i64::from(value)),
			Some(PropValue::Str(value)) => value.parse().ok(),
			_ => None,
		};
		let id = field.props.id().cloned().unwrap_or_default();
		let label = if field.label.is_empty() {
			field
				.props
				.str_of(Prop::Label)
				.cloned()
				.unwrap_or_else(|| id.clone())
		} else {
			field.label
		};
		Self {
			kind,
			id,
			label,
			desc: field.props.str_of(Prop::Desc).cloned(),
			options,
			value,
			required: field.props.flag(Prop::Required),
			pattern: field.props.str_of(Prop::Match).cloned(),
			min: i64_prop(Prop::Min).unwrap_or(i64::MIN),
			max: i64_prop(Prop::Max).unwrap_or(i64::MAX),
			step: i64_prop(Prop::Step).unwrap_or(1),
		}
	}
}

/// An interactive collection of fields backing the `<form>` markup tag.
pub struct Form {
	props:      Props,
	slot:       Slot,
	fields:     Vec<FieldData>,
	cursor:     u16,
	editing:    bool,
	open:       Option<u16>,
	sub_cursor: u16,
	scratch:    String,
}

impl Form {
	/// Creates an empty form.
	pub fn new() -> Self {
		Self {
			props:      Props::new(),
			slot:       next_slot(),
			fields:     Vec::new(),
			cursor:     0,
			editing:    false,
			open:       None,
			sub_cursor: 0,
			scratch:    String::new(),
		}
	}

	/// Sets one form property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one form property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Appends a field definition.
	pub fn field(mut self, field: Field) -> Self {
		self.fields.push(FieldData::from_field(field));
		self
	}

	fn activate(&mut self) {
		let cursor = self.cursor;
		let Some(field) = self.fields.get_mut(usize::from(cursor)) else {
			return;
		};
		match field.kind {
			FieldKind::Bool => {
				if let FieldValue::Bool(value) = &mut field.value {
					*value = !*value;
				}
			},
			FieldKind::Enum => cycle_choice(field, true),
			FieldKind::Select | FieldKind::Multi => {
				self.open = Some(cursor);
				self.sub_cursor = match (&field.value, field.kind) {
					(FieldValue::Choice(choice), FieldKind::Select) => field
						.options
						.iter()
						.position(|option| option == choice)
						.unwrap_or(0) as u16,
					_ => 0,
				};
			},
			FieldKind::Text => self.editing = true,
			FieldKind::Number => {},
		}
	}

	fn click_row(&mut self, index: u16) {
		if usize::from(index) >= self.fields.len() {
			return;
		}
		self.cursor = index;
		if self.open.is_some() && self.open != Some(index) {
			self.open = None;
		}
		self.activate();
	}

	fn click_sub(&mut self, index: u16) {
		let Some(open) = self.open else { return };
		self.sub_cursor = index;
		let field = &mut self.fields[usize::from(open)];
		if field.kind == FieldKind::Multi {
			toggle_multi(field, index);
		} else {
			if let Some(option) = field.options.get(usize::from(index)) {
				field.value = FieldValue::Choice(option.clone());
			}
			self.open = None;
		}
	}
}

impl Default for Form {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Form {
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
		let natural = self
			.fields
			.iter()
			.map(|field| cell_width(&field.label) + 24)
			.max()
			.unwrap_or(24);
		(24, natural)
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		let mut height = self.fields.len() as u16;
		if let Some(open) = self.open {
			height += self.fields[usize::from(open)].options.len() as u16;
		}
		if self
			.fields
			.get(usize::from(self.cursor))
			.is_some_and(|field| field.desc.is_some())
		{
			height += 1;
		}
		height
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		let focused = pc.focus == Some(self.slot);
		let label_width = self
			.fields
			.iter()
			.map(|field| cell_width(&field.label))
			.max()
			.unwrap_or(8)
			+ 2;
		let mut hit_y = rect.y;
		for (index, field) in self.fields.iter().enumerate() {
			pc.hits.push(Hit {
				rect: Rect::new(rect.x, hit_y, rect.width, 1),
				slot: self.slot,
				tag:  HitTag::Row(index as u16),
			});
			hit_y = hit_y.saturating_add(1);
			if self.open == Some(index as u16) {
				for option_index in 0..field.options.len() as u16 {
					pc.hits.push(Hit {
						rect: Rect::new(rect.x, hit_y, rect.width, 1),
						slot: self.slot,
						tag:  HitTag::Sub(option_index),
					});
					hit_y = hit_y.saturating_add(1);
				}
			}
		}
		let mut y = rect.y;
		for (index, field) in self.fields.iter().enumerate() {
			if y >= pc.clip {
				return;
			}
			let here = focused && index as u16 == self.cursor;
			let hovered = matches!(pc.hover, Some((slot, HitTag::Row(row))) if slot == self.slot && row == index as u16);
			if hovered {
				pc.frame
					.fill(Rect::new(rect.x, y, rect.width, 1), Style::new().bg(pc.ctx.theme.hover));
			}
			let tint = |style: Style| {
				if hovered {
					style.bg(pc.ctx.theme.hover)
				} else {
					style
				}
			};
			let mut x = pc.frame.put(
				rect.x,
				y,
				if here { pc.ctx.charset.cursor() } else { "  " },
				tint(Style::new().fg(pc.ctx.theme.accent)),
			);
			let label_style = if here {
				tint(Style::new().fg(pc.ctx.theme.accent).bold())
			} else {
				tint(base(&pc.ctx.theme))
			};
			x = pc.frame.put(x, y, &field.label, label_style);
			for _ in cell_width(&field.label)..label_width {
				x = pc.frame.put(x, y, " ", tint(base(&pc.ctx.theme)));
			}
			x = paint_field_value(pc.ctx, pc.frame, x, y, field, here, tint, &mut self.scratch);
			if here && self.editing {
				pc.frame
					.put(x, y, pc.ctx.charset.beam(), tint(Style::new().fg(pc.ctx.theme.accent)));
			}
			y += 1;
			if self.open == Some(index as u16) {
				for (option_index, option) in field.options.iter().enumerate() {
					if y >= pc.clip {
						return;
					}
					let sub_here = option_index as u16 == self.sub_cursor;
					let picked = match &field.value {
						FieldValue::Choice(choice) => choice == option,
						FieldValue::Many(values) => values.contains(option),
						_ => false,
					};
					let mark = if field.kind == FieldKind::Multi {
						pc.ctx.charset.checkbox(picked)
					} else {
						pc.ctx.charset.radio(picked)
					};
					let mut sx = pc.frame.put(
						rect.x + 4,
						y,
						if sub_here {
							pc.ctx.charset.cursor()
						} else {
							"  "
						},
						Style::new().fg(pc.ctx.theme.accent),
					);
					sx = pc.frame.put(
						sx,
						y,
						mark,
						Style::new().fg(if picked {
							pc.ctx.theme.ok
						} else {
							pc.ctx.theme.muted
						}),
					);
					sx = pc.frame.put(sx, y, " ", base(&pc.ctx.theme));
					pc.frame.put(
						sx,
						y,
						option,
						if sub_here {
							Style::new().fg(pc.ctx.theme.accent).bold()
						} else {
							base(&pc.ctx.theme)
						},
					);
					y += 1;
				}
			}
		}
		if let Some(field) = self.fields.get(usize::from(self.cursor))
			&& let Some(desc) = &field.desc
			&& y < pc.clip
		{
			pc.frame.put(rect.x + 2, y, desc, dim(&pc.ctx.theme));
		}
	}

	fn focusable(&self) -> bool {
		true
	}

	fn enter(&mut self, forward: bool) {
		self.cursor = if forward {
			0
		} else {
			self.fields.len().saturating_sub(1) as u16
		};
	}

	fn key(&mut self, _ec: &mut EventCtx<'_>, key: Key) -> Flow {
		if self.fields.is_empty() {
			return Flow::Skip;
		}
		if let Some(open) = self.open {
			let field = &mut self.fields[usize::from(open)];
			let len = field.options.len() as u16;
			match key {
				Key::Up if len > 0 => self.sub_cursor = (self.sub_cursor + len - 1) % len,
				Key::Down if len > 0 => self.sub_cursor = (self.sub_cursor + 1) % len,
				Key::Space if field.kind == FieldKind::Multi => toggle_multi(field, self.sub_cursor),
				Key::Enter => {
					if field.kind != FieldKind::Multi
						&& let Some(option) = field.options.get(usize::from(self.sub_cursor))
					{
						field.value = FieldValue::Choice(option.clone());
					}
					self.open = None;
				},
				Key::Esc => self.open = None,
				_ => {},
			}
			return Flow::Consumed;
		}
		let field_count = self.fields.len() as u16;
		if self.editing {
			let field = &mut self.fields[usize::from(self.cursor)];
			let FieldValue::Text(text) = &mut field.value else {
				self.editing = false;
				return Flow::Consumed;
			};
			match key {
				Key::Enter | Key::Esc => self.editing = false,
				Key::Backspace => {
					text.pop();
				},
				Key::Space => text.push(' '),
				Key::Char(character) => text.push(character),
				Key::Ctrl('u') => text.clear(),
				Key::Ctrl('w') => text.truncate(word_rubout_start(text, text.len())),
				_ => {},
			}
			return Flow::Consumed;
		}
		let kind = self.fields[usize::from(self.cursor)].kind;
		match key {
			Key::Left | Key::Right if kind == FieldKind::Enum => {
				cycle_choice(&mut self.fields[usize::from(self.cursor)], key == Key::Right);
				Flow::Consumed
			},
			Key::Left | Key::Right if kind == FieldKind::Number => {
				let field = &mut self.fields[usize::from(self.cursor)];
				if let FieldValue::Number(value) = &mut field.value {
					let step = if key == Key::Right {
						field.step
					} else {
						field.step.saturating_neg()
					};
					*value = value.saturating_add(step).clamp(field.min, field.max);
				}
				Flow::Consumed
			},
			Key::Up if self.cursor > 0 => {
				self.cursor -= 1;
				Flow::Consumed
			},
			Key::Down if self.cursor + 1 < field_count => {
				self.cursor += 1;
				Flow::Consumed
			},
			Key::Enter | Key::Space => {
				self.activate();
				Flow::Consumed
			},
			_ => Flow::Skip,
		}
	}

	fn mouse(
		&mut self,
		_ec: &mut EventCtx<'_>,
		tag: HitTag,
		_at: (u16, u16),
		_rect: Rect,
		mouse: Mouse,
	) -> Flow {
		match mouse {
			Mouse::Click => {
				match tag {
					HitTag::Row(index) => self.click_row(index),
					HitTag::Sub(index) => self.click_sub(index),
					_ => return Flow::Skip,
				}
				Flow::Consumed
			},
			Mouse::RightClick
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
		if !self.editing {
			return Flow::Skip;
		}
		let sanitized = sanitize_paste(text);
		if sanitized.is_empty() {
			return Flow::Skip;
		}
		let Some(FieldData { value: FieldValue::Text(value), .. }) =
			self.fields.get_mut(usize::from(self.cursor))
		else {
			return Flow::Skip;
		};
		value.push_str(&sanitized.replace(['\n', '\t'], " "));
		Flow::Consumed
	}

	fn validation_error(&self) -> Option<String> {
		for field in &self.fields {
			let value = field_value(field);
			let text = super::wizard::display_value(&value);
			if field.required && text.trim().is_empty() {
				return Some(format!("{} is required", field.id));
			}
			if let Some(pattern) = &field.pattern
				&& !text.trim().is_empty()
				&& !super::wizard::match_simple(pattern, text.trim())
			{
				return Some(format!("{} must match {}", field.id, pattern));
			}
		}
		None
	}

	fn value(&self, out: &mut Map<String, Value>) {
		let Some(id) = self.props.id() else { return };
		let mut object = Map::new();
		for field in &self.fields {
			if !field.id.is_empty() {
				object.insert(field.id.to_string(), field_value(field));
			}
		}
		out.insert(id.to_string(), Value::Object(object));
	}
}

fn cycle_choice(field: &mut FieldData, forward: bool) {
	if field.options.is_empty() {
		return;
	}
	if let FieldValue::Choice(current) = &field.value {
		let len = field.options.len();
		let at = field
			.options
			.iter()
			.position(|option| option == current)
			.unwrap_or(0);
		let next = if forward {
			(at + 1) % len
		} else {
			(at + len - 1) % len
		};
		field.value = FieldValue::Choice(field.options[next].clone());
	}
}

fn toggle_multi(field: &mut FieldData, index: u16) {
	let Some(option) = field.options.get(usize::from(index)) else {
		return;
	};
	let option = option.clone();
	if let FieldValue::Many(values) = &mut field.value {
		if values.contains(&option) {
			values.retain(|value| *value != option);
		} else {
			values.push(option);
			values.sort_by_key(|value| field.options.iter().position(|option| option == value));
		}
	}
}

fn field_value(field: &FieldData) -> Value {
	match &field.value {
		FieldValue::Bool(value) => Value::Bool(*value),
		FieldValue::Text(value) => Value::String(value.clone()),
		FieldValue::Choice(value) => Value::String(value.to_string()),
		FieldValue::Many(values) => Value::Array(
			values
				.iter()
				.map(|value| Value::String(value.to_string()))
				.collect(),
		),
		FieldValue::Number(value) => Value::Number((*value).into()),
	}
}

const fn base(theme: &Theme) -> Style {
	Style::new().fg(theme.fg)
}
const fn dim(theme: &Theme) -> Style {
	Style::new().fg(theme.muted)
}

fn paint_field_value(
	ctx: &UiContext,
	frame: &mut crate::frame::Frame,
	x: u16,
	y: u16,
	field: &FieldData,
	here: bool,
	tint: impl Fn(Style) -> Style,
	scratch: &mut String,
) -> u16 {
	match (&field.value, field.kind) {
		(FieldValue::Bool(value), _) => frame.put(
			x,
			y,
			if *value { "true" } else { "false" },
			tint(Style::new().fg(if *value {
				ctx.theme.ok
			} else {
				ctx.theme.muted
			})),
		),
		(FieldValue::Choice(choice), FieldKind::Enum) => {
			let mut x = frame.put(x, y, choice, tint(Style::new().fg(ctx.theme.info)));
			if here {
				x = frame.put(x, y, "  ", tint(dim(&ctx.theme)));
				x = frame.put(x, y, ctx.charset.arrows().0, tint(dim(&ctx.theme)));
				x = frame.put(x, y, " ", tint(dim(&ctx.theme)));
				x = frame.put(x, y, ctx.charset.arrows().1, tint(dim(&ctx.theme)));
			}
			x
		},
		(FieldValue::Choice(choice), _) => {
			let x = frame.put(x, y, choice, tint(Style::new().fg(ctx.theme.info)));
			frame.put(x, y, ctx.charset.dropdown(), tint(dim(&ctx.theme)))
		},
		(FieldValue::Many(values), _) => {
			scratch.clear();
			if values.is_empty() {
				scratch.push('—');
			} else {
				for (index, value) in values.iter().enumerate() {
					if index > 0 {
						scratch.push_str(", ");
					}
					scratch.push_str(value);
				}
			}
			let x = frame.put(x, y, scratch, tint(Style::new().fg(ctx.theme.info)));
			frame.put(x, y, ctx.charset.dropdown(), tint(dim(&ctx.theme)))
		},
		(FieldValue::Number(value), _) => {
			let mut x = x;
			if here {
				x = frame.put(x, y, ctx.charset.arrows().0, tint(dim(&ctx.theme)));
				x = frame.put(x, y, " ", tint(dim(&ctx.theme)));
			}
			scratch.clear();
			let _ = write!(scratch, "{value}");
			x = frame.put(x, y, scratch, tint(Style::new().fg(ctx.theme.warn)));
			if here {
				x = frame.put(x, y, " ", tint(dim(&ctx.theme)));
				x = frame.put(x, y, ctx.charset.arrows().1, tint(dim(&ctx.theme)));
			}
			x
		},
		(FieldValue::Text(text), _) => frame.put(x, y, text, tint(base(&ctx.theme))),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn event_ctx(ctx: &UiContext) -> EventCtx<'_> {
		EventCtx::new(ctx, 40, 10)
	}

	#[test]
	fn navigation_edit_and_values_match_form_contract() {
		let mut form = Form::new()
			.with(Prop::Id, "settings")
			.field(
				Field::new()
					.with(Prop::Id, "name")
					.with(Prop::Kind, "text")
					.with(Prop::Value, "omp"),
			)
			.field(
				Field::new()
					.with(Prop::Id, "theme")
					.with(Prop::Kind, "select")
					.with(Prop::Options, "dark light")
					.with(Prop::Value, "dark"),
			);
		let ctx = UiContext::default();
		let mut ec = event_ctx(&ctx);
		assert_eq!(form.key(&mut ec, Key::Enter), Flow::Consumed);
		assert_eq!(form.key(&mut ec, Key::Char('!')), Flow::Consumed);
		assert_eq!(form.key(&mut ec, Key::Enter), Flow::Consumed);
		assert_eq!(form.key(&mut ec, Key::Down), Flow::Consumed);
		assert_eq!(form.key(&mut ec, Key::Enter), Flow::Consumed);
		assert_eq!(form.key(&mut ec, Key::Down), Flow::Consumed);
		assert_eq!(form.key(&mut ec, Key::Enter), Flow::Consumed);
		let mut values = Map::new();
		form.value(&mut values);
		assert_eq!(values["settings"], serde_json::json!({ "name": "omp!", "theme": "light" }));
	}

	#[test]
	fn validation_reports_the_first_invalid_field() {
		let mut form = Form::new()
			.field(
				Field::new()
					.with(Prop::Id, "name")
					.with(Prop::Required, true),
			)
			.field(
				Field::new()
					.with(Prop::Id, "slug")
					.with(Prop::Match, "[a-z-]+")
					.with(Prop::Value, "Bad Slug"),
			);

		assert_eq!(form.validation_error().as_deref(), Some("name is required"));
		form.fields[0].value = FieldValue::Text("OMP".into());
		assert_eq!(form.validation_error().as_deref(), Some("slug must match [a-z-]+"));
		form.fields[1].value = FieldValue::Text("valid-slug".into());
		assert_eq!(form.validation_error(), None);
	}
}
