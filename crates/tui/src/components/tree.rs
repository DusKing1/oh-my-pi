use omp_core::{Str, StrMut};
use smallvec::SmallVec;

use crate::{
	component::{Component, EventCtx, Flow, Hit, HitTag, PaintCtx, Slot, next_slot},
	context::UiContext,
	frame::{Rect, Style},
	input::{Key, Mouse},
	props::{Prop, PropValue, Props},
	rich::cell_width,
};

/// A labeled branch or leaf backing the `<node>` markup tag.
pub struct TreeNode {
	props:    Props,
	slot:     Slot,
	label:    Str,
	children: Vec<Self>,
}

impl TreeNode {
	/// Creates an empty tree node.
	pub fn new() -> Self {
		Self {
			props:    Props::new(),
			slot:     next_slot(),
			label:    Str::default(),
			children: Vec::new(),
		}
	}

	/// Sets one node property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Appends node label text.
	pub fn label(mut self, label: impl Into<Str>) -> Self {
		append(&mut self.label, label.into());
		self
	}

	/// Appends a child node.
	pub fn node(mut self, node: Self) -> Self {
		self.children.push(node);
		self
	}

	fn effective_label(&self) -> &str {
		if self.label.is_empty() {
			self.props.str_of(Prop::Label).map_or("", Str::as_str)
		} else {
			&self.label
		}
	}
}

impl Default for TreeNode {
	fn default() -> Self {
		Self::new()
	}
}

#[derive(Clone, Debug, Default)]
struct TreeState {
	cursor: u16,
	chosen: Option<Str>,
	open:   SmallVec<Slot, 8>,
}

#[derive(Clone, Debug)]
struct TreeRow {
	node:         Slot,
	depth:        u16,
	path:         Str,
	label:        Str,
	has_children: bool,
	/// Continuation bits for ancestor levels below the root: `true` when
	/// that ancestor has further siblings, so its guide column keeps
	/// running through this row.
	gutters:      SmallVec<bool, 8>,
	/// Whether this row closes its sibling run (`└─` instead of `├─`).
	last:         bool,
}

/// An expandable hierarchy backing the `<tree>` markup tag.
pub struct Tree {
	props:      Props,
	slot:       Slot,
	nodes:      Vec<TreeNode>,
	state:      TreeState,
	rows:       Vec<TreeRow>,
	rows_dirty: bool,
}

impl Tree {
	/// Creates an empty tree.
	pub fn new() -> Self {
		Self {
			props:      Props::new(),
			slot:       next_slot(),
			nodes:      Vec::new(),
			state:      TreeState::default(),
			rows:       Vec::new(),
			rows_dirty: true,
		}
	}

	#[allow(dead_code, reason = "acceptance-suite probe")]
	pub(crate) const fn visible_rows_len(&self) -> usize {
		self.rows.len()
	}

	/// Sets one tree property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one tree property from a string.
	pub fn with_str(mut self, prop: Prop, value: &str) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Appends a root node.
	pub fn node(mut self, node: TreeNode) -> Self {
		collect_open(std::slice::from_ref(&node), &mut self.state.open);
		self.nodes.push(node);
		self.rows_dirty = true;
		self
	}

	fn rebuild_rows(&mut self) {
		if !self.rows_dirty {
			return;
		}
		self.rows.clear();
		let mut trail = SmallVec::new();
		walk_rows(&self.nodes, 0, "", &self.state.open, &mut trail, &mut self.rows);
		if self.rows.is_empty() {
			self.state.cursor = 0;
		} else {
			self.state.cursor = self.state.cursor.min(self.rows.len() as u16 - 1);
		}
		self.rows_dirty = false;
	}

	fn toggle(&mut self, slot: Slot) {
		if self.state.open.contains(&slot) {
			self.state.open.retain(|open| *open != slot);
		} else {
			self.state.open.push(slot);
		}
		self.rows_dirty = true;
	}
}

impl Default for Tree {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Tree {
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
		self.rebuild_rows();
		let nat = self
			.rows
			.iter()
			.map(|row| {
				cell_width(&row.label)
					.saturating_add(row.depth.saturating_mul(2))
					.saturating_add(6)
			})
			.max()
			.unwrap_or(16);
		(16, nat)
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		self.rebuild_rows();
		u16::try_from(self.rows.len()).unwrap_or(u16::MAX)
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		self.rebuild_rows();
		let focused = pc.focus == Some(self.slot);
		let hover_row = match pc.hover {
			Some((slot, HitTag::Row(index))) if slot == self.slot => Some(index),
			_ => None,
		};
		let bottom = rect.y.saturating_add(rect.height).min(pc.clip);
		for (index, row) in self.rows.iter().enumerate() {
			let index = index as u16;
			let y = rect.y.saturating_add(index);
			if y >= bottom {
				break;
			}
			let hovered = hover_row == Some(index);
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
			let here = focused && index == self.state.cursor;
			let mut x = pc.frame.put(
				rect.x,
				y,
				if here { pc.ctx.charset.cursor() } else { "  " },
				tint(Style::new().fg(pc.ctx.theme.accent)),
			);
			if let Some(family) = self.props.guides() {
				let (branch, last, cont) = pc.ctx.charset.guides(family);
				let guide = tint(Style::new().fg(pc.ctx.theme.muted));
				for &more in &row.gutters {
					x = pc.frame.put(x, y, if more { cont } else { "  " }, guide);
				}
				if row.depth > 0 {
					x = pc
						.frame
						.put(x, y, if row.last { last } else { branch }, guide);
					x = pc.frame.put(x, y, " ", guide);
				}
			} else {
				for _ in 0..row.depth {
					x = pc
						.frame
						.put(x, y, "  ", tint(Style::new().fg(pc.ctx.theme.fg)));
				}
			}
			let expander = if row.has_children {
				pc.ctx.charset.expander(self.state.open.contains(&row.node))
			} else if self.props.guides().is_some() {
				// Guide trees indent leaves with the connector alone.
				""
			} else {
				"  "
			};
			x = pc
				.frame
				.put(x, y, expander, tint(Style::new().fg(pc.ctx.theme.muted)));
			let style = if here {
				tint(Style::new().fg(pc.ctx.theme.accent).bold())
			} else {
				tint(Style::new().fg(pc.ctx.theme.fg))
			};
			x = pc.frame.put(x, y, &row.label, style);
			if self.state.chosen.as_ref() == Some(&row.path) {
				pc.frame.put(
					x.saturating_add(1),
					y,
					pc.ctx.charset.check(),
					tint(Style::new().fg(pc.ctx.theme.ok)),
				);
			}
			pc.hits.push(Hit {
				rect: Rect::new(rect.x, y, rect.width, 1),
				slot: self.slot,
				tag:  HitTag::Row(index),
			});
		}
	}

	fn focusable(&self) -> bool {
		true
	}

	fn enter(&mut self, forward: bool) {
		self.rebuild_rows();
		if self.rows.is_empty() {
			self.state.cursor = 0;
		} else {
			self.state.cursor = if forward {
				0
			} else {
				self.rows.len() as u16 - 1
			};
		}
	}

	fn key(&mut self, _ec: &mut EventCtx<'_>, key: Key) -> Flow {
		self.rebuild_rows();
		if self.rows.is_empty() {
			return Flow::Skip;
		}
		self.state.cursor = self.state.cursor.min(self.rows.len() as u16 - 1);
		let current = usize::from(self.state.cursor);
		let row = self.rows[current].clone();
		let is_open = self.state.open.contains(&row.node);
		match key {
			Key::Up => {
				if self.state.cursor == 0 {
					return Flow::Skip;
				}
				self.state.cursor -= 1;
			},
			Key::Down => {
				if self.state.cursor + 1 >= self.rows.len() as u16 {
					return Flow::Skip;
				}
				self.state.cursor += 1;
			},
			Key::Right if row.has_children && !is_open => self.toggle(row.node),
			Key::Left => {
				if row.has_children && is_open {
					self.toggle(row.node);
				} else if let Some(parent) = self.rows[..current]
					.iter()
					.rposition(|candidate| candidate.depth + 1 == row.depth)
				{
					self.state.cursor = parent as u16;
				} else {
					return Flow::Skip;
				}
			},
			Key::Enter | Key::Space => {
				if row.has_children {
					self.toggle(row.node);
				} else {
					self.state.chosen = Some(row.path);
				}
			},
			_ => return Flow::Skip,
		}
		Flow::Consumed
	}

	fn mouse(
		&mut self,
		_ec: &mut EventCtx<'_>,
		tag: HitTag,
		_at: (u16, u16),
		_rect: Rect,
		mouse: Mouse,
	) -> Flow {
		self.rebuild_rows();
		match mouse {
			Mouse::WheelUp | Mouse::WheelDown => {
				if self.rows.is_empty() {
					return Flow::Skip;
				}
				let delta = if mouse == Mouse::WheelUp { -1 } else { 1 };
				self.state.cursor =
					(i64::from(self.state.cursor) + delta).clamp(0, self.rows.len() as i64 - 1) as u16;
				Flow::Consumed
			},
			Mouse::Click => {
				let HitTag::Row(index) = tag else {
					return Flow::Skip;
				};
				let Some(row) = self.rows.get(usize::from(index)).cloned() else {
					return Flow::Skip;
				};
				self.state.cursor = index;
				if row.has_children {
					self.toggle(row.node);
				} else {
					self.state.chosen = Some(row.path);
				}
				Flow::Consumed
			},
			Mouse::RightClick
			| Mouse::MiddleClick
			| Mouse::Move
			| Mouse::Drag
			| Mouse::Release
			| Mouse::WheelLeft
			| Mouse::WheelRight => Flow::Skip,
		}
	}

	fn value(&self, out: &mut serde_json::Map<String, serde_json::Value>) {
		let Some(id) = self.props.id() else {
			return;
		};
		let value = self
			.state
			.chosen
			.as_ref()
			.map_or(serde_json::Value::Null, |path| serde_json::Value::String(path.to_string()));
		out.insert(id.to_string(), value);
	}
}

fn collect_open(nodes: &[TreeNode], open: &mut SmallVec<Slot, 8>) {
	for node in nodes {
		if node.props.flag(Prop::Open) {
			open.push(node.slot);
		}
		collect_open(&node.children, open);
	}
}

fn walk_rows(
	nodes: &[TreeNode],
	depth: u16,
	prefix: &str,
	open: &[Slot],
	trail: &mut SmallVec<bool, 8>,
	rows: &mut Vec<TreeRow>,
) {
	let count = nodes.len();
	for (index, node) in nodes.iter().enumerate() {
		let label = node.effective_label();
		let path = if prefix.is_empty() {
			Str::new(label)
		} else {
			let mut path =
				StrMut::with_capacity(prefix.len().saturating_add(label.len()).saturating_add(1));
			path.push_str(prefix);
			path.push('/');
			path.push_str(label);
			path.freeze()
		};
		let has_children = !node.children.is_empty();
		let last = index + 1 == count;
		// Gutter columns exist only for ancestor levels below the root: the
		// root level draws no connector column, so its continuation bit is
		// never displayed.
		let gutters = if depth > 1 {
			trail[1..].into()
		} else {
			SmallVec::new()
		};
		rows.push(TreeRow {
			node: node.slot,
			depth,
			path: path.clone(),
			label: Str::new(label),
			has_children,
			gutters,
			last,
		});
		if open.contains(&node.slot) {
			trail.push(!last);
			walk_rows(&node.children, depth.saturating_add(1), &path, open, trail, rows);
			trail.pop();
		}
	}
}

fn append(target: &mut Str, suffix: Str) {
	if target.is_empty() {
		*target = suffix;
		return;
	}
	let mut joined = StrMut::with_capacity(target.len().saturating_add(suffix.len()));
	joined.push_str(target);
	joined.push_str(&suffix);
	*target = joined.freeze();
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn expand_collapse_rebuilds_rows_and_values_use_paths() {
		let ctx = UiContext::default();
		let mut tree = Tree::new().with(Prop::Id, "tree-id").node(
			TreeNode::new()
				.label("root")
				.node(TreeNode::new().label("leaf")),
		);
		let mut ec = EventCtx::new(&ctx, 30, 4);
		assert_eq!(tree.height(&ctx, 30), 1);
		assert!(!tree.rows_dirty);
		assert_eq!(tree.key(&mut ec, Key::Right), Flow::Consumed);
		assert!(tree.rows_dirty);
		assert_eq!(tree.height(&ctx, 30), 2);
		assert_eq!(tree.rows[1].path, "root/leaf");
		assert_eq!(tree.key(&mut ec, Key::Down), Flow::Consumed);
		assert_eq!(tree.key(&mut ec, Key::Enter), Flow::Consumed);
		let mut values = serde_json::Map::new();
		tree.value(&mut values);
		assert_eq!(values["tree-id"], serde_json::json!("root/leaf"));
		assert_eq!(tree.key(&mut ec, Key::Left), Flow::Consumed);
		assert_eq!(tree.state.cursor, 0);
		assert_eq!(tree.key(&mut ec, Key::Left), Flow::Consumed);
		assert_eq!(tree.height(&ctx, 30), 1);
	}

	#[test]
	fn open_property_seeds_visible_rows() {
		let ctx = UiContext::default();
		let mut tree = Tree::new().node(
			TreeNode::new()
				.with(Prop::Label, "root")
				.with(Prop::Open, true)
				.node(TreeNode::new().label("leaf")),
		);
		assert_eq!(tree.height(&ctx, 20), 2);
		assert_eq!(tree.rows[0].label, "root");
	}
}
