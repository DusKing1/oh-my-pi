use std::{cmp::Ordering, fmt::Write};

use omp_core::SmolStr;
use serde_json::Value;

use crate::{
	component::{
		Cached, Component, EventCtx, Flow, Hit, HitTag, IntoChildren, IntoComponent, PaintCtx, Slot,
		next_slot,
	},
	context::UiContext,
	frame::{Color, Frame, Rect, Style},
	input::{Key, Mouse, UiEvent},
	props::{Prop, PropValue, Props},
	rich::cell_width,
};

#[derive(Default)]
struct WizardState {
	idx:     u16,
	error:   Option<String>,
	button:  u8,
	spans:   Vec<(u16, u16)>,
	rect:    Option<Rect>,
	scratch: String,
	rule:    String,
}

/// A validated sequence of step panes backing the `<wizard>` markup tag.
pub struct Wizard {
	props: Props,
	slot:  Slot,
	steps: Vec<Cached>,
	state: WizardState,
}

impl Wizard {
	/// Creates an empty wizard.
	pub fn new() -> Self {
		Self {
			props: Props::new(),
			slot:  next_slot(),
			steps: Vec::new(),
			state: WizardState::default(),
		}
	}

	/// Sets one wizard property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one wizard property from a string.
	pub fn with_str(self, prop: Prop, value: &str) -> Self {
		self.with(prop, value)
	}

	/// Appends an untitled step pane.
	pub fn child(mut self, children: impl IntoChildren) -> Self {
		children.extend_children(&mut self.steps);
		self
	}

	/// Appends a titled step pane.
	pub fn step(mut self, title: impl Into<SmolStr>, children: impl IntoChildren) -> Self {
		let pane = super::Col::new()
			.with(Prop::Title, title.into())
			.child(children);
		self.steps.push(Cached::new(pane.into_component()));
		self
	}

	#[allow(dead_code, reason = "acceptance-suite probe")]
	pub(crate) fn error(&self) -> Option<&str> {
		self.state.error.as_deref()
	}

	#[allow(dead_code, reason = "acceptance-suite probe")]
	pub(crate) fn step_index(&self) -> usize {
		usize::from(self.state.idx)
	}

	fn active(&self) -> Option<usize> {
		let index = usize::from(self.state.idx);
		(index < self.steps.len()).then_some(index)
	}

	fn validate_step(&self) -> Option<String> {
		let step = self.steps.get(self.active()?)?;
		validate_cached(step)
	}

	fn next(&mut self) -> UiEvent {
		if let Some(error) = self.validate_step() {
			self.state.error = Some(error);
			return UiEvent::None;
		}
		self.state.error = None;
		if usize::from(self.state.idx) + 1 < self.steps.len() {
			self.state.idx += 1;
			UiEvent::None
		} else if self.props.flag(Prop::Submit) {
			UiEvent::Submit
		} else {
			UiEvent::None
		}
	}

	fn back(&mut self) {
		self.state.error = None;
		self.state.idx = self.state.idx.saturating_sub(1);
	}
}

impl Default for Wizard {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Wizard {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn children(&self) -> &[Cached] {
		&self.steps
	}

	fn children_mut(&mut self) -> &mut [Cached] {
		&mut self.steps
	}

	fn measure(&mut self, ctx: &UiContext) -> (u16, u16) {
		let chips = u16::try_from(self.steps.len())
			.unwrap_or(u16::MAX)
			.saturating_mul(12);
		let mut natural = chips.max(24);
		for step in &mut self.steps {
			natural = natural.max(step.measure(ctx).1);
		}
		(24, natural)
	}

	fn height(&mut self, ctx: &UiContext, width: u16) -> u16 {
		let content = if let Some(active) = self.active() {
			self.steps[active].height(ctx, width)
		} else {
			0
		};
		content
			.saturating_add(3)
			.saturating_add(u16::from(self.state.error.is_some()))
	}

	fn place(&mut self, ctx: &UiContext, content: Rect) {
		self.state.rect = Some(content);
		if let Some(active) = self.active() {
			let height = self.steps[active].height(ctx, content.width);
			self.steps[active]
				.place(ctx, Rect::new(content.x, content.y.saturating_add(2), content.width, height));
		}
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		self.state.rect = Some(rect);
		let focused = pc.focus == Some(self.slot);
		let base = Style::new().fg(pc.ctx.theme.fg);
		self.state.spans.clear();
		if rect.y < pc.clip {
			let mut x = rect.x;
			for (index, step) in self.steps.iter().enumerate() {
				let title = step.comp().props().title().map_or("step", SmolStr::as_str);
				let start = x.saturating_sub(rect.x);
				match u16::try_from(index)
					.unwrap_or(u16::MAX)
					.cmp(&self.state.idx)
				{
					Ordering::Less => {
						x = pc.frame.put(
							x,
							rect.y,
							pc.ctx.charset.check(),
							Style::new().fg(pc.ctx.theme.ok),
						);
						x = pc
							.frame
							.put(x, rect.y, " ", Style::new().fg(pc.ctx.theme.ok));
						x = pc
							.frame
							.put(x, rect.y, title, Style::new().fg(pc.ctx.theme.ok));
					},
					Ordering::Equal => {
						self.state.scratch.clear();
						let _ = write!(self.state.scratch, " {} {} ", index + 1, title);
						x = pill(
							pc.frame,
							x,
							rect.y,
							&self.state.scratch,
							pc.ctx.theme.accent,
							pc.ctx.theme.contrast,
							pc.ctx.charset.pill_caps(),
							focused,
						);
					},
					Ordering::Greater => {
						self.state.scratch.clear();
						let _ = write!(self.state.scratch, "{} {}", index + 1, title);
						x = pc.frame.put(
							x,
							rect.y,
							&self.state.scratch,
							Style::new().fg(pc.ctx.theme.muted),
						);
					},
				}
				let end = x.saturating_sub(rect.x);
				self.state.spans.push((start, end));
				if end > start {
					pc.hits.push(Hit {
						rect: Rect::new(rect.x.saturating_add(start), rect.y, end - start, 1),
						slot: self.slot,
						tag:  HitTag::Chip(index as u16),
					});
				}
				x = pc.frame.put(x, rect.y, "  ", base);
			}
		}
		if rect.y.saturating_add(1) < pc.clip {
			self.state.rule.clear();
			for _ in 0..rect.width {
				self.state.rule.push(pc.ctx.charset.rule());
			}
			pc.frame.put(
				rect.x,
				rect.y.saturating_add(1),
				&self.state.rule,
				Style::new().fg(pc.ctx.theme.muted),
			);
		}
		if let Some(active) = self.active() {
			self.steps[active].paint(pc);
		}

		let y = rect.y.saturating_add(rect.height).saturating_sub(1);
		if let Some(error) = &self.state.error {
			let error_y = y.saturating_sub(1);
			if error_y < pc.clip {
				let style = Style::new().fg(pc.ctx.theme.warn);
				let mut x =
					pc.frame
						.put(rect.x, error_y, pc.ctx.charset.icon(crate::Icon::Warning), style);
				x = pc.frame.put(x, error_y, " ", style);
				pc.frame.put(x, error_y, error, style);
			}
		}
		if y < pc.clip {
			let last = usize::from(self.state.idx) + 1 == self.steps.len();
			let next_label = if last { "Finish" } else { "Next" };
			let back_x = rect.x.saturating_add(
				rect
					.width
					.saturating_sub(cell_width(next_label) + cell_width("Back") + 9),
			);
			let x = pill(
				pc.frame,
				back_x,
				y,
				" Back ",
				pc.ctx.theme.surface,
				pc.ctx.theme.fg,
				pc.ctx.charset.pill_caps(),
				focused && self.state.button == 0,
			);
			self.state.scratch.clear();
			let _ = write!(self.state.scratch, " {next_label} ");
			pill(
				pc.frame,
				x.saturating_add(2),
				y,
				&self.state.scratch,
				pc.ctx.theme.accent,
				pc.ctx.theme.contrast,
				pc.ctx.charset.pill_caps(),
				focused && self.state.button == 1,
			);
			pc.hits.push(Hit {
				rect: Rect::new(rect.x, y, rect.width, 1),
				slot: self.slot,
				tag:  HitTag::Press,
			});
		}
	}

	fn focusable(&self) -> bool {
		true
	}

	fn ring(&self, out: &mut Vec<Slot>) {
		if let Some(active) = self.active()
			&& self.steps[active].visible
		{
			self.steps[active].comp().ring(out);
		}
		out.push(self.slot);
	}

	fn key(&mut self, _ec: &mut EventCtx<'_>, key: Key) -> Flow {
		match key {
			Key::Left => {
				self.state.button = 0;
				Flow::Consumed
			},
			Key::Right => {
				self.state.button = 1;
				Flow::Consumed
			},
			Key::Enter | Key::Space if self.state.button == 0 => {
				self.back();
				Flow::Consumed
			},
			Key::Enter | Key::Space => match self.next() {
				UiEvent::None => Flow::Consumed,
				event => Flow::Event(event),
			},
			_ => Flow::Skip,
		}
	}

	fn mouse(
		&mut self,
		_ec: &mut EventCtx<'_>,
		tag: HitTag,
		at: (u16, u16),
		_rect: Rect,
		mouse: Mouse,
	) -> Flow {
		match mouse {
			Mouse::Click => match tag {
				HitTag::Chip(target) => {
					if target < self.state.idx {
						self.state.idx = target;
						self.state.error = None;
					}
					Flow::Consumed
				},
				HitTag::Press => {
					let Some(rect) = self.state.rect else {
						return Flow::Skip;
					};
					let midpoint = rect.x.saturating_add(rect.width.saturating_sub(9));
					if at.0 >= midpoint {
						self.state.button = 1;
						match self.next() {
							UiEvent::None => Flow::Consumed,
							event => Flow::Event(event),
						}
					} else {
						self.state.button = 0;
						self.back();
						Flow::Consumed
					}
				},
				_ => Flow::Skip,
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
}

/// Tiny anchored pattern matcher for `match=` validation. Supported
/// syntax: literals, `[a-z0-9-]` classes (ranges + literals, leading `^`
/// negation), and the postfix quantifiers `*` `+` `?` on the previous
/// atom. This is NOT a regex engine; unsupported constructs match
/// literally.
pub fn match_simple(pattern: &str, text: &str) -> bool {
	#[derive(Clone)]
	enum Atom {
		Literal(char),
		Class(Vec<(char, char)>, bool),
		Any,
	}
	fn atom_matches(atom: &Atom, c: char) -> bool {
		match atom {
			Atom::Literal(l) => *l == c,
			Atom::Any => true,
			Atom::Class(ranges, negated) => {
				let inside = ranges.iter().any(|&(lo, hi)| c >= lo && c <= hi);
				inside != *negated
			},
		}
	}
	// parse into (atom, quantifier) pairs
	let mut atoms: Vec<(Atom, char)> = Vec::new();
	let mut chars = pattern.chars().peekable();
	while let Some(c) = chars.next() {
		let atom = match c {
			'.' => Atom::Any,
			'[' => {
				let mut ranges = Vec::new();
				let negated = chars.peek() == Some(&'^');
				if negated {
					chars.next();
				}
				let mut prev: Option<char> = None;
				loop {
					match chars.next() {
						None | Some(']') => break,
						Some('-') => {
							let (Some(lo), Some(&hi)) = (prev, chars.peek()) else {
								ranges.push(('-', '-'));
								continue;
							};
							chars.next();
							ranges.pop();
							ranges.push((lo, hi));
							prev = None;
						},
						Some(other) => {
							ranges.push((other, other));
							prev = Some(other);
						},
					}
				}
				Atom::Class(ranges, negated)
			},
			'\\' => Atom::Literal(chars.next().unwrap_or('\\')),
			other => Atom::Literal(other),
		};
		let quantifier = match chars.peek() {
			Some(&q @ ('*' | '+' | '?')) => {
				chars.next();
				q
			},
			_ => ' ',
		};
		atoms.push((atom, quantifier));
	}
	// backtracking match, anchored both ends
	fn matches_at(atoms: &[(Atom, char)], text: &[char], ai: usize, ti: usize) -> bool {
		let Some((atom, quantifier)) = atoms.get(ai) else {
			return ti == text.len();
		};
		match quantifier {
			'*' | '+' => {
				let mut count = 0usize;
				let minimum = usize::from(*quantifier == '+');
				loop {
					if count >= minimum && matches_at(atoms, text, ai + 1, ti + count) {
						return true;
					}
					match text.get(ti + count) {
						Some(&c) if atom_matches(atom, c) => count += 1,
						_ => return false,
					}
				}
			},
			'?' => {
				if matches_at(atoms, text, ai + 1, ti) {
					return true;
				}
				matches!(text.get(ti), Some(&c) if atom_matches(atom, c))
					&& matches_at(atoms, text, ai + 1, ti + 1)
			},
			_ => {
				matches!(text.get(ti), Some(&c) if atom_matches(atom, c))
					&& matches_at(atoms, text, ai + 1, ti + 1)
			},
		}
	}
	let text: Vec<char> = text.chars().collect();
	matches_at(&atoms, &text, 0, 0)
}

fn validate_cached(cached: &Cached) -> Option<String> {
	if !cached.visible {
		return None;
	}
	let component = cached.comp();
	let props = component.props();
	if let Some(id) = props.id() {
		let mut values = serde_json::Map::new();
		component.value(&mut values);
		if let Some(value) = values.get(id.as_str()) {
			let text = display_value(value);
			if props.flag(Prop::Required) && text.trim().is_empty() {
				return Some(format!("{id} is required"));
			}
			if let Some(pattern) = props.str_of(Prop::Match)
				&& !text.trim().is_empty()
				&& !match_simple(pattern, text.trim())
			{
				return Some(format!("{id} must match {pattern}"));
			}
		}
	}
	if let Some(error) = component.validation_error() {
		return Some(error);
	}
	for child in component.children() {
		if let Some(error) = validate_cached(child) {
			return Some(error);
		}
	}
	None
}

pub(super) fn display_value(value: &Value) -> String {
	match value {
		Value::Null => String::new(),
		Value::String(value) => value.clone(),
		Value::Array(values) => values
			.iter()
			.map(display_value)
			.collect::<Vec<_>>()
			.join(" "),
		Value::Bool(value) => value.to_string(),
		Value::Number(value) => value.to_string(),
		Value::Object(_) => value.to_string(),
	}
}

fn pill(
	frame: &mut Frame,
	x: u16,
	y: u16,
	label: &str,
	background: Color,
	foreground: Color,
	caps: (&str, &str),
	highlight: bool,
) -> u16 {
	let background = if highlight {
		brighten(background)
	} else {
		background
	};
	let cap = Style::new().fg(background);
	let body = Style::new().fg(foreground).bg(background).bold();
	let mut x = frame.put(x, y, caps.0, cap);
	x = frame.put(x, y, label, body);
	frame.put(x, y, caps.1, cap)
}

fn brighten(color: Color) -> Color {
	match color {
		Color::Rgb(red, green, blue) => Color::Rgb(
			red.saturating_add((255 - u16::from(red)) as u8 / 5),
			green.saturating_add((255 - u16::from(green)) as u8 / 5),
			blue.saturating_add((255 - u16::from(blue)) as u8 / 5),
		),
		other => other,
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		component::EventCtx,
		components::{Field, Form, Input},
	};

	#[test]
	fn next_and_back_switch_the_active_ring_before_the_wizard_slot() {
		let ctx = UiContext::default();
		let first = Input::new();
		let first_slot = first.slot();
		let second = Input::new();
		let second_slot = second.slot();
		let mut wizard = Wizard::new().step("First", first).step("Second", second);
		let wizard_slot = wizard.slot();
		let mut ring = Vec::new();
		wizard.ring(&mut ring);
		assert_eq!(ring, vec![first_slot, wizard_slot]);

		let mut ec = EventCtx::new(&ctx, 30, 6);
		assert_eq!(wizard.key(&mut ec, Key::Right), Flow::Consumed);
		assert_eq!(wizard.key(&mut ec, Key::Enter), Flow::Consumed);
		ring.clear();
		wizard.ring(&mut ring);
		assert_eq!(ring, vec![second_slot, wizard_slot]);

		assert_eq!(wizard.key(&mut ec, Key::Left), Flow::Consumed);
		assert_eq!(wizard.key(&mut ec, Key::Enter), Flow::Consumed);
		ring.clear();
		wizard.ring(&mut ring);
		assert_eq!(ring, vec![first_slot, wizard_slot]);
	}

	#[test]
	fn required_value_blocks_next_step() {
		let ctx = UiContext::default();
		let required = Input::new()
			.with(Prop::Id, "name")
			.with(Prop::Required, true);
		let required_slot = required.slot();
		let second = Input::new();
		let second_slot = second.slot();
		let mut wizard = Wizard::new().step("Details", required).step("Done", second);
		let wizard_slot = wizard.slot();
		let mut ec = EventCtx::new(&ctx, 30, 6);
		wizard.key(&mut ec, Key::Right);
		assert_eq!(wizard.key(&mut ec, Key::Enter), Flow::Consumed);

		let mut ring = Vec::new();
		wizard.ring(&mut ring);
		assert_eq!(ring, vec![required_slot, wizard_slot]);
		assert!(!ring.contains(&second_slot));
		assert_eq!(wizard.state.error.as_deref(), Some("name is required"));
	}

	#[test]
	fn form_field_validation_blocks_and_allows_next_step() {
		let empty = Form::new().field(
			Field::new()
				.with(Prop::Id, "name")
				.with(Prop::Required, true),
		);
		let mut blocked = Wizard::new()
			.step("Details", empty)
			.step("Done", Input::new());
		assert_eq!(blocked.next(), UiEvent::None);
		assert_eq!(blocked.step_index(), 0);
		assert_eq!(blocked.error(), Some("name is required"));

		let filled = Form::new().field(
			Field::new()
				.with(Prop::Id, "name")
				.with(Prop::Required, true)
				.with(Prop::Value, "OMP"),
		);
		let mut allowed = Wizard::new()
			.step("Details", filled)
			.step("Done", Input::new());
		assert_eq!(allowed.next(), UiEvent::None);
		assert_eq!(allowed.step_index(), 1);
		assert_eq!(allowed.error(), None);
	}
}
