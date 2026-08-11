use omp_core::{Str, fmts};
use smallvec::SmallVec;
use strum::{EnumString, IntoStaticStr};

use crate::{
	component::{Component, PaintCtx, Slot, next_slot},
	context::UiContext,
	frame::{Rect, Style},
	markup::Border,
	props::{Prop, PropValue, Props},
	rich::cell_width,
};

/// Lifecycle state of a [`TodoTask`], mirroring the coding agent's todo
/// tracker: open work, the one active item, and the three closed shapes.
#[derive(Clone, Copy, Debug, Default, EnumString, Eq, IntoStaticStr, PartialEq)]
pub enum TaskStatus {
	/// Not started; renders dim with an empty checkbox.
	#[default]
	#[strum(to_string = "pending", serialize = "open")]
	Pending,
	/// Currently being worked; renders accent.
	#[strum(to_string = "active", serialize = "in-progress", serialize = "in_progress")]
	Active,
	/// Finished; renders ok with a checked box and struck label.
	#[strum(to_string = "done", serialize = "completed")]
	Done,
	/// Abandoned; renders err with a struck label.
	#[strum(to_string = "dropped", serialize = "abandoned")]
	Dropped,
	/// Waiting on something external; renders warn with the blocker note.
	#[strum(to_string = "blocked")]
	Blocked,
}

impl TaskStatus {
	/// Parses a markup `status=` value, accepting the agent-side aliases.
	pub fn parse(name: &str) -> Option<Self> {
		name.parse().ok()
	}
}

/// One row of a [`Todo`] list, backing the `<task>` markup tag.
///
/// A task with children renders as a group header with an automatic
/// `done/total` count over its direct children; a leaf renders a status
/// checkbox and its label. `status=` sets [`TaskStatus`]; `desc=` carries
/// the blocker note shown by [`TaskStatus::Blocked`].
pub struct TodoTask {
	props:    Props,
	label:    Str,
	children: Vec<Self>,
}

impl TodoTask {
	/// Creates a pending, empty task.
	pub fn new() -> Self {
		Self { props: Props::new(), label: Str::default(), children: Vec::new() }
	}

	/// Sets one task property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Appends label text.
	pub fn label(mut self, label: impl Into<Str>) -> Self {
		let suffix = label.into();
		if self.label.is_empty() {
			self.label = suffix;
		} else {
			self.label = fmts!("{}{}", self.label, suffix);
		}
		self
	}

	/// Sets the lifecycle state.
	pub fn status(mut self, status: TaskStatus) -> Self {
		let name: &'static str = status.into();
		self.props.set(Prop::Status, name);
		self
	}

	/// Appends a child task, turning this task into a group header.
	pub fn task(mut self, task: Self) -> Self {
		self.children.push(task);
		self
	}

	fn effective_label(&self) -> &str {
		if self.label.is_empty() {
			self.props.str_of(Prop::Label).map_or("", Str::as_str)
		} else {
			&self.label
		}
	}

	fn effective_status(&self) -> TaskStatus {
		self
			.props
			.str_of(Prop::Status)
			.and_then(|name| TaskStatus::parse(name))
			.unwrap_or_default()
	}
}

impl Default for TodoTask {
	fn default() -> Self {
		Self::new()
	}
}

/// A static todo list backing the `<todo>` markup tag.
///
/// Children are [`TodoTask`] records; nesting produces tree guides in the
/// family chosen by `guides=` (square by default). Unlike [`crate::Tree`],
/// the list is display-only: no focus, keys, or collapse state.
pub struct Todo {
	props: Props,
	slot:  Slot,
	tasks: Vec<TodoTask>,
}

impl Todo {
	/// Creates an empty todo list.
	pub fn new() -> Self {
		Self { props: Props::new(), slot: next_slot(), tasks: Vec::new() }
	}

	/// Sets one list property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Appends a root task.
	pub fn task(mut self, task: TodoTask) -> Self {
		self.tasks.push(task);
		self
	}

	/// Leaf `(done, total)` across the whole list, for host-built headers
	/// like `3/14 tasks`.
	pub fn counts(&self) -> (usize, usize) {
		leaf_counts(&self.tasks)
	}

	fn family(&self) -> Border {
		self.props.guides().unwrap_or(Border::Square)
	}

	fn row_count(tasks: &[TodoTask]) -> u16 {
		let mut rows = 0u16;
		for task in tasks {
			rows = rows
				.saturating_add(1)
				.saturating_add(Self::row_count(&task.children));
		}
		rows
	}

	fn max_width(tasks: &[TodoTask], depth: u16) -> u16 {
		let mut widest = 0u16;
		for task in tasks {
			// gutter columns + connector + checkbox/count slack
			let width = cell_width(task.effective_label())
				.saturating_add(depth.saturating_mul(2))
				.saturating_add(10);
			widest = widest
				.max(width)
				.max(Self::max_width(&task.children, depth + 1));
		}
		widest
	}
}
/// Leaf `(done, total)` under `tasks`; groups contribute their descendants.
fn leaf_counts(tasks: &[TodoTask]) -> (usize, usize) {
	let (mut done, mut total) = (0, 0);
	for task in tasks {
		if task.children.is_empty() {
			total += 1;
			done += usize::from(task.effective_status() == TaskStatus::Done);
		} else {
			let (child_done, child_total) = leaf_counts(&task.children);
			done += child_done;
			total += child_total;
		}
	}
	(done, total)
}

impl Default for Todo {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Todo {
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
		(12, Self::max_width(&self.tasks, 0).max(12))
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		Self::row_count(&self.tasks)
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		let (branch, last, cont) = pc.ctx.charset.guides(self.family());
		let glyphs = Glyphs {
			branch,
			last,
			cont,
			checked: pc.ctx.charset.checkbox(true),
			unchecked: pc.ctx.charset.checkbox(false),
		};
		let mut y = rect.y;
		let mut trail: SmallVec<bool, 8> = SmallVec::new();
		paint_tasks(pc, rect, &glyphs, &self.tasks, &mut trail, &mut y);
	}
}

struct Glyphs {
	branch:    &'static str,
	last:      &'static str,
	cont:      &'static str,
	checked:   &'static str,
	unchecked: &'static str,
}

fn paint_tasks(
	pc: &mut PaintCtx<'_>,
	rect: Rect,
	glyphs: &Glyphs,
	tasks: &[TodoTask],
	trail: &mut SmallVec<bool, 8>,
	y: &mut u16,
) {
	let bottom = rect.y.saturating_add(rect.height).min(pc.clip);
	let count = tasks.len();
	for (index, task) in tasks.iter().enumerate() {
		if *y >= bottom {
			return;
		}
		let is_last = index + 1 == count;
		let mut x = rect.x;
		let guide = Style::new().fg(pc.ctx.theme.muted);
		// Ancestor gutters, then this row's connector. Roots draw neither,
		// and the root level's continuation bit is never displayed.
		if !trail.is_empty() {
			for &more in &trail[1..] {
				x = pc
					.frame
					.put(x, *y, if more { glyphs.cont } else { "  " }, guide);
			}
			x = pc
				.frame
				.put(x, *y, if is_last { glyphs.last } else { glyphs.branch }, guide);
			x = pc.frame.put(x, *y, " ", guide);
		}
		let label = task.effective_label();
		if task.children.is_empty() {
			let theme = &pc.ctx.theme;
			let status = task.effective_status();
			let (glyph, style) = match status {
				TaskStatus::Done => (glyphs.checked, Style::new().fg(theme.ok)),
				TaskStatus::Active => (glyphs.unchecked, Style::new().fg(theme.accent)),
				TaskStatus::Dropped => (glyphs.unchecked, Style::new().fg(theme.err)),
				TaskStatus::Blocked => (glyphs.unchecked, Style::new().fg(theme.warn)),
				TaskStatus::Pending => (glyphs.unchecked, Style::new().dim()),
			};
			x = pc.frame.put(x, *y, glyph, style);
			x = pc.frame.put(x, *y, " ", style);
			let label_style = match status {
				TaskStatus::Done | TaskStatus::Dropped => style.strikethrough(),
				_ => style,
			};
			x = pc.frame.put(x, *y, label, label_style);
			if status == TaskStatus::Blocked {
				let note = task.props.str_of(Prop::Desc).map_or_else(
					|| Str::new_static(" (blocked)"),
					|reason| fmts!(" (blocked: {reason})"),
				);
				pc.frame.put(x, *y, &note, Style::new().dim());
			}
		} else {
			// Group header: bold label plus an automatic done/total count
			// over its descendant leaves.
			let (done, total) = leaf_counts(&task.children);
			x = pc
				.frame
				.put(x, *y, label, Style::new().fg(pc.ctx.theme.fg).bold());
			let counter = fmts!(" {done}/{total}");
			pc.frame.put(x, *y, &counter, Style::new().dim());
		}
		*y = y.saturating_add(1);
		if !task.children.is_empty() {
			trail.push(!is_last);
			paint_tasks(pc, rect, glyphs, &task.children, trail, y);
			trail.pop();
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn counts_walk_nested_leaves_only() {
		let todo = Todo::new()
			.task(
				TodoTask::new()
					.label("phase")
					.task(TodoTask::new().label("a").status(TaskStatus::Done))
					.task(TodoTask::new().label("b")),
			)
			.task(TodoTask::new().label("flat").status(TaskStatus::Done));
		assert_eq!(todo.counts(), (2, 3));
	}

	#[test]
	fn status_parse_accepts_agent_aliases_and_rejects_junk() {
		assert_eq!(TaskStatus::parse("in_progress"), Some(TaskStatus::Active));
		assert_eq!(TaskStatus::parse("completed"), Some(TaskStatus::Done));
		assert_eq!(TaskStatus::parse("abandoned"), Some(TaskStatus::Dropped));
		assert_eq!(TaskStatus::parse("nope"), None);
	}
}
