//! Pane multiplexing: the binary split tree of one tab plus pure layout
//! math — partitioning, ratio clamping, and directional focus.
//!
//! Everything here is host-agnostic geometry: no winit, no wgpu. The host
//! owns pane state and calls [`layout`] whenever the tree or window changes.

use smallvec::SmallVec;

/// Minimum pane extent on a split axis, physical px.
pub(crate) const MIN_PANE: f32 = 80.0;

/// Stable identity of one pane within a window.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct PaneId(pub u32);

/// Split orientation: `X` places children side by side, `Y` stacks them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Axis {
	X,
	Y,
}

/// Directional focus target for pane navigation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Dir {
	Left,
	Right,
	Up,
	Down,
}

/// An axis-aligned rectangle in physical px.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct RectPx {
	pub x: f32,
	pub y: f32,
	pub w: f32,
	pub h: f32,
}

impl RectPx {
	/// Whether `point` falls inside the rectangle.
	pub fn contains(&self, point: [f32; 2]) -> bool {
		point[0] >= self.x
			&& point[0] < self.x + self.w
			&& point[1] >= self.y
			&& point[1] < self.y + self.h
	}
}

/// Path from the root to a split: child indices, 0 = first, 1 = second.
pub(crate) type Path = SmallVec<u8, 8>;

/// One tab's split tree.
pub(crate) enum Node {
	/// A pane fills this slot.
	Leaf(PaneId),
	/// Two children share this slot along `axis`; the first child takes
	/// `ratio` of the extent (gutter excluded).
	Split { axis: Axis, ratio: f32, children: Box<(Node, Node)> },
}

/// Outcome of [`Node::remove`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Removed {
	/// The leaf's split collapsed; focus moves to this surviving pane.
	Collapsed(PaneId),
	/// The tree was a single matching leaf; the caller drops the tab.
	Root,
	/// No such leaf.
	Missing,
}

/// The gutter band between the two children of one split.
pub(crate) struct Divider {
	/// The gutter rect itself, physical px.
	pub rect:   RectPx,
	/// The whole rect of the owning split, for ratio recomputation.
	pub region: RectPx,
	/// The owning split's axis.
	pub axis:   Axis,
	/// Path of the owning split from the root.
	pub path:   Path,
}

impl Node {
	/// Replaces `Leaf(at)` with a half-half split of `at` and `new`.
	pub fn split(&mut self, at: PaneId, axis: Axis, new: PaneId) -> bool {
		match self {
			Self::Leaf(id) if *id == at => {
				*self = Self::Split {
					axis,
					ratio: 0.5,
					children: Box::new((Self::Leaf(at), Self::Leaf(new))),
				};
				true
			},
			Self::Leaf(_) => false,
			Self::Split { children, .. } => {
				children.0.split(at, axis, new) || children.1.split(at, axis, new)
			},
		}
	}

	/// Deletes `Leaf(id)`, promoting its sibling subtree into the parent's
	/// slot; the returned pane is the sibling's first leaf, for refocus.
	pub fn remove(&mut self, id: PaneId) -> Removed {
		match self {
			Self::Leaf(leaf) => {
				if *leaf == id {
					Removed::Root
				} else {
					Removed::Missing
				}
			},
			Self::Split { children, .. } => {
				let keep = if matches!(children.0, Self::Leaf(leaf) if leaf == id) {
					Some(std::mem::replace(&mut children.1, Self::Leaf(id)))
				} else if matches!(children.1, Self::Leaf(leaf) if leaf == id) {
					Some(std::mem::replace(&mut children.0, Self::Leaf(id)))
				} else {
					None
				};
				if let Some(sibling) = keep {
					*self = sibling;
					return Removed::Collapsed(self.first_leaf());
				}
				match children.0.remove(id) {
					Removed::Missing => children.1.remove(id),
					removed => removed,
				}
			},
		}
	}

	/// The first pane in DFS order.
	pub fn first_leaf(&self) -> PaneId {
		match self {
			Self::Leaf(id) => *id,
			Self::Split { children, .. } => children.0.first_leaf(),
		}
	}

	/// Appends every pane in DFS order.
	pub fn leaves(&self, out: &mut SmallVec<PaneId, 8>) {
		match self {
			Self::Leaf(id) => out.push(*id),
			Self::Split { children, .. } => {
				children.0.leaves(out);
				children.1.leaves(out);
			},
		}
	}

	/// The ratio of the split at `path`, if it exists.
	pub fn ratio_mut(&mut self, path: &[u8]) -> Option<&mut f32> {
		match self {
			Self::Leaf(_) => None,
			Self::Split { ratio, children, .. } => match path.split_first() {
				None => Some(ratio),
				Some((0, rest)) => children.0.ratio_mut(rest),
				Some((_, rest)) => children.1.ratio_mut(rest),
			},
		}
	}

	/// Path of the nearest ancestor split with `axis` containing `id`,
	/// for keyboard resizing.
	pub fn resize_target(&self, id: PaneId, axis: Axis) -> Option<Path> {
		let mut path = Path::new();
		if !self.path_to(id, &mut path) {
			return None;
		}
		let mut node = self;
		let mut best = None;
		for (depth, &step) in path.iter().enumerate() {
			let Self::Split { axis: node_axis, children, .. } = node else {
				break;
			};
			if *node_axis == axis {
				best = Some(depth);
			}
			node = if step == 0 { &children.0 } else { &children.1 };
		}
		best.map(|depth| path[..depth].iter().copied().collect())
	}

	fn path_to(&self, id: PaneId, path: &mut Path) -> bool {
		match self {
			Self::Leaf(leaf) => *leaf == id,
			Self::Split { children, .. } => {
				path.push(0);
				if children.0.path_to(id, path) {
					return true;
				}
				path.pop();
				path.push(1);
				if children.1.path_to(id, path) {
					return true;
				}
				path.pop();
				false
			},
		}
	}
}

/// Partitions `rect` across the tree: every leaf gets its rect and every
/// split its gutter [`Divider`]. The first child's extent floors so pane
/// edges stay on whole pixels; the second child absorbs the remainder.
pub(crate) fn layout(
	node: &Node,
	rect: RectPx,
	gutter: f32,
	panes: &mut SmallVec<(PaneId, RectPx), 8>,
	dividers: &mut SmallVec<Divider, 8>,
) {
	let mut path = Path::new();
	partition(node, rect, gutter, panes, dividers, &mut path);
}

fn partition(
	node: &Node,
	rect: RectPx,
	gutter: f32,
	panes: &mut SmallVec<(PaneId, RectPx), 8>,
	dividers: &mut SmallVec<Divider, 8>,
	path: &mut Path,
) {
	match node {
		Node::Leaf(id) => panes.push((*id, rect)),
		Node::Split { axis, ratio, children } => {
			let (first, second, gap) = match axis {
				Axis::X => {
					let first_w = ((rect.w - gutter) * ratio).floor();
					(
						RectPx { w: first_w, ..rect },
						RectPx { x: rect.x + first_w + gutter, w: rect.w - gutter - first_w, ..rect },
						RectPx { x: rect.x + first_w, w: gutter, ..rect },
					)
				},
				Axis::Y => {
					let first_h = ((rect.h - gutter) * ratio).floor();
					(
						RectPx { h: first_h, ..rect },
						RectPx { y: rect.y + first_h + gutter, h: rect.h - gutter - first_h, ..rect },
						RectPx { y: rect.y + first_h, h: gutter, ..rect },
					)
				},
			};
			dividers.push(Divider { rect: gap, region: rect, axis: *axis, path: path.clone() });
			path.push(0);
			partition(&children.0, first, gutter, panes, dividers, path);
			path.pop();
			path.push(1);
			partition(&children.1, second, gutter, panes, dividers, path);
			path.pop();
		},
	}
}

/// Clamps `ratio` so both children keep at least [`MIN_PANE`] px; a region
/// too small for two minima passes the ratio through unchanged.
pub(crate) fn clamp_ratio(region: RectPx, axis: Axis, gutter: f32, ratio: f32) -> f32 {
	let extent = match axis {
		Axis::X => region.w,
		Axis::Y => region.h,
	} - gutter;
	if extent < MIN_PANE * 2.0 {
		return ratio;
	}
	let min = MIN_PANE / extent;
	ratio.clamp(min, 1.0 - min)
}

/// The pane adjacent to `from` in `dir`: among panes strictly past the
/// facing edge, the one with the largest perpendicular overlap, ties broken
/// by edge distance.
pub(crate) fn neighbor(rects: &[(PaneId, RectPx)], from: PaneId, dir: Dir) -> Option<PaneId> {
	let (_, origin) = rects.iter().find(|(id, _)| *id == from)?;
	let mut best: Option<(PaneId, f32, f32)> = None;
	for (id, rect) in rects {
		if *id == from {
			continue;
		}
		let distance = match dir {
			Dir::Left => origin.x - (rect.x + rect.w),
			Dir::Right => rect.x - (origin.x + origin.w),
			Dir::Up => origin.y - (rect.y + rect.h),
			Dir::Down => rect.y - (origin.y + origin.h),
		};
		if distance < 0.0 {
			continue;
		}
		let overlap = match dir {
			Dir::Left | Dir::Right => {
				(origin.y + origin.h).min(rect.y + rect.h) - origin.y.max(rect.y)
			},
			Dir::Up | Dir::Down => (origin.x + origin.w).min(rect.x + rect.w) - origin.x.max(rect.x),
		};
		let better = match best {
			None => true,
			Some((_, best_overlap, best_distance)) => {
				overlap > best_overlap || (overlap == best_overlap && distance < best_distance)
			},
		};
		if better {
			best = Some((*id, overlap, distance));
		}
	}
	best.map(|(id, ..)| id)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn rect(x: f32, y: f32, w: f32, h: f32) -> RectPx {
		RectPx { x, y, w, h }
	}

	#[test]
	fn split_then_remove_collapses_to_leaf_and_refocuses_survivor() {
		let mut tree = Node::Leaf(PaneId(0));
		assert!(tree.split(PaneId(0), Axis::X, PaneId(1)));
		assert!(tree.split(PaneId(1), Axis::Y, PaneId(2)));

		assert_eq!(tree.remove(PaneId(2)), Removed::Collapsed(PaneId(1)));
		assert_eq!(tree.remove(PaneId(0)), Removed::Collapsed(PaneId(1)));
		assert!(matches!(tree, Node::Leaf(PaneId(1))));
		assert_eq!(tree.remove(PaneId(0)), Removed::Missing);
		assert_eq!(tree.remove(PaneId(1)), Removed::Root);
	}

	#[test]
	fn layout_partitions_exactly_with_gutter() {
		let mut tree = Node::Leaf(PaneId(0));
		tree.split(PaneId(0), Axis::X, PaneId(1));
		tree.split(PaneId(1), Axis::Y, PaneId(2));

		let mut panes = SmallVec::new();
		let mut dividers = SmallVec::new();
		let region = rect(10.0, 20.0, 807.0, 605.0);
		layout(&tree, region, 8.0, &mut panes, &mut dividers);

		assert_eq!(panes.len(), 3);
		assert_eq!(dividers.len(), 2);
		let (_, a) = panes[0];
		let (_, b) = panes[1];
		let (_, c) = panes[2];
		// Horizontal split: widths plus gutter cover the region exactly.
		assert_eq!(a.w + 8.0 + b.w, region.w);
		assert_eq!(a.x, region.x);
		assert_eq!(b.x, a.x + a.w + 8.0);
		assert_eq!(b.x, c.x);
		// Vertical child split partitions the right column's height.
		assert_eq!(b.h + 8.0 + c.h, region.h);
		assert_eq!(c.y, b.y + b.h + 8.0);
		// No overlap: divider bands separate the panes.
		assert!(a.x + a.w <= b.x - 8.0 + 8.0);
		assert_eq!(dividers[0].path.as_slice(), &[] as &[u8]);
		assert_eq!(dividers[1].path.as_slice(), &[1]);
		assert_eq!(dividers[1].region, b_union_c(b, c));
	}

	fn b_union_c(b: RectPx, c: RectPx) -> RectPx {
		RectPx { x: b.x, y: b.y, w: b.w, h: b.h + 8.0 + c.h }
	}

	#[test]
	fn clamp_ratio_enforces_minimum_and_passes_degenerate_regions() {
		let region = rect(0.0, 0.0, 808.0, 600.0);
		let min = MIN_PANE / 800.0;
		assert_eq!(clamp_ratio(region, Axis::X, 8.0, 0.01), min);
		assert_eq!(clamp_ratio(region, Axis::X, 8.0, 0.99), 1.0 - min);
		assert_eq!(clamp_ratio(region, Axis::X, 8.0, 0.5), 0.5);
		// Too small for two minima: input unchanged.
		let tiny = rect(0.0, 0.0, 100.0, 100.0);
		assert_eq!(clamp_ratio(tiny, Axis::X, 8.0, 0.9), 0.9);
	}

	#[test]
	fn neighbor_resolves_all_directions_on_a_grid() {
		let rects = [
			(PaneId(0), rect(0.0, 0.0, 100.0, 100.0)),
			(PaneId(1), rect(108.0, 0.0, 100.0, 100.0)),
			(PaneId(2), rect(0.0, 108.0, 100.0, 100.0)),
			(PaneId(3), rect(108.0, 108.0, 100.0, 100.0)),
		];
		assert_eq!(neighbor(&rects, PaneId(0), Dir::Right), Some(PaneId(1)));
		assert_eq!(neighbor(&rects, PaneId(3), Dir::Left), Some(PaneId(2)));
		assert_eq!(neighbor(&rects, PaneId(3), Dir::Up), Some(PaneId(1)));
		assert_eq!(neighbor(&rects, PaneId(0), Dir::Down), Some(PaneId(2)));
		assert_eq!(neighbor(&rects, PaneId(0), Dir::Left), None);
		assert_eq!(neighbor(&rects, PaneId(1), Dir::Up), None);
	}
}
