//! Forward capsule capture and reverse native-item reconstruction.

use std::collections::BTreeMap;

use omp_core::SmolStr;
use serde_json::{Value, value::RawValue};
use smallvec::SmallVec;

use crate::transcript::{
	block::{Block, BlockKind, Replay},
	capsule::{
		DefaultCtx, Dialect, JoinMode, Oai, Rev, diff, insert_marker, overlay, split_markers,
	},
};

/// Renders one peek-free dialect default for every neutral block.
///
/// This is the forward renderer used when no native-item residue has yet been
/// captured. The same [`Dialect::default_item`] implementation is called by
/// [`capture`] and [`rebuild`].
#[must_use]
pub fn emit<D: Dialect + ?Sized>(blocks: &[Block], dialect: &D, rev: Rev) -> Vec<Value> {
	blocks
		.iter()
		.map(|block| dialect.default_item(DefaultCtx::single(&block.kind), rev))
		.collect()
}

/// Captures field-level replay capsules by diffing native items against
/// peek-free defaults.
///
/// Native items are matched to their neutral projections by dialect item kind
/// and default-diff size. This also discovers multi-part projections and
/// records `~ord` only when wire order and neutral block order differ. The
/// returned vector is parallel to `blocks`; grouped followers are represented
/// by `None` because their leading capsule owns the native item.
#[must_use]
pub fn capture<D: Dialect + ?Sized>(
	blocks: &[Block],
	items: &[Value],
	dialect: &D,
	rev: Rev,
) -> Vec<Option<Replay>> {
	let dialect_id = dialect.id();
	let dialect_name = dialect_id.0.as_str();
	let mut assigned = vec![false; blocks.len()];
	let mut captured = std::iter::repeat_with(|| None)
		.take(blocks.len())
		.collect::<Vec<_>>();
	let mut placements = Vec::with_capacity(items.len().min(blocks.len()));
	let mut previous_block_index = None;
	let mut order_changed = false;

	for (item_index, item) in items.iter().enumerate() {
		let expected = expected_kind(item, dialect_name);
		let mut best: Option<Candidate> = None;
		for (block_index, block) in blocks.iter().enumerate() {
			if assigned[block_index] || !kind_matches(&block.kind, expected) {
				continue;
			}
			let candidate = best_candidate(blocks, &assigned, block_index, item, dialect, rev);
			if best
				.as_ref()
				.is_none_or(|current| candidate.better_than(current))
			{
				best = Some(candidate);
			}
		}

		let Some(mut best) = best else {
			continue;
		};
		if best.np > 1 {
			insert_marker(&mut best.fields, "~np", &best.np);
		}
		if let Some(join) = best.join {
			let marker = match join {
				JoinMode::Split => "split",
				JoinMode::Jnn => "jnn",
				JoinMode::J => "j",
			};
			insert_marker(&mut best.fields, "~m", &marker);
		}
		let np = usize::try_from(best.np).unwrap_or(usize::MAX);
		let end = best.block_index.saturating_add(np).min(blocks.len());
		assigned[best.block_index..end].fill(true);
		order_changed |= previous_block_index.is_some_and(|previous| previous > best.block_index);
		previous_block_index = Some(best.block_index);
		captured[best.block_index] = Some(Replay { p: dialect_id.clone(), f: best.fields });
		placements.push((best.block_index, item_index));
	}

	if order_changed {
		for (block_index, item_index) in placements {
			if let Some(replay) = &mut captured[block_index] {
				let ord = u32::try_from(item_index).unwrap_or(u32::MAX);
				insert_marker(&mut replay.f, "~ord", &ord);
			}
		}
	}
	captured
}

/// Rebuilds same-provider native items with the normative replay algorithm.
///
/// A `~np: N` marker spans `N` consecutive same-kind blocks. `~m: "split"`
/// splits their text on `\n\n`; `"jnn"` joins block texts with `\n\n`; and
/// `"j"` joins them without a separator. Blocks without a capsule for `dialect`
/// yield nothing, and every reserved marker is consumed before the item reaches
/// wire output.
#[must_use]
pub fn rebuild<D: Dialect + ?Sized>(blocks: &[Block], dialect: &D, rev: Rev) -> Vec<Value> {
	let dialect_id = dialect.id();
	let dialect_name = dialect_id.0.as_str();
	let mut native = Vec::new();
	let mut index = 0;

	while index < blocks.len() {
		let block = &blocks[index];
		let Some(replay) = matching_replay(block, dialect_name) else {
			index += 1;
			continue;
		};
		let (markers, _) = split_markers(&replay.f);
		let source = u32::try_from(native.len()).unwrap_or(u32::MAX);

		match &block.kind {
			BlockKind::Text { .. } | BlockKind::Think { .. } => {
				let requested = usize::try_from(markers.np.unwrap_or(1).max(1)).unwrap_or(usize::MAX);
				let mut following = SmallVec::<&BlockKind, 4>::new();
				let mut cursor = index.saturating_add(1);
				while following.len().saturating_add(1) < requested && cursor < blocks.len() {
					let follower = &blocks[cursor];
					if follower.re.is_some() || !same_kind(&block.kind, &follower.kind) {
						break;
					}
					following.push(&follower.kind);
					cursor += 1;
				}
				let ctx = DefaultCtx::grouped(&block.kind, &following, markers.join);
				let default = dialect.default_item(ctx, rev);
				native.push(NativeItem {
					value: overlay(default, &replay.f),
					ord: markers.ord,
					source,
				});
				index = cursor;
			},
			BlockKind::Opaque => {
				native.push(NativeItem {
					value: overlay(Value::Object(serde_json::Map::new()), &replay.f),
					ord: markers.ord,
					source,
				});
				index += 1;
			},
			BlockKind::Tool { .. } | BlockKind::Image { .. } => {
				let default = dialect.default_item(DefaultCtx::single(&block.kind), rev);
				native.push(NativeItem {
					value: overlay(default, &replay.f),
					ord: markers.ord,
					source,
				});
				index += 1;
			},
		}
	}

	native.sort_by_key(|item| item.ord.unwrap_or(item.source));
	native.into_iter().map(|item| item.value).collect()
}

/// Re-encodes neutral blocks for a cross-provider handoff while ignoring every
/// capsule.
///
/// The neutral OpenAI-family encoding is used as the portable target. Opaque
/// blocks are dropped because they have no neutral projection; consequently
/// signatures and encrypted/model-bound content can never cross the provider
/// boundary.
#[must_use]
pub fn rebuild_cross(blocks: &[Block]) -> Vec<Value> {
	let dialect = Oai;
	blocks
		.iter()
		.filter(|block| !matches!(&block.kind, BlockKind::Opaque))
		.map(|block| {
			dialect.default_item(DefaultCtx::single(&block.kind), crate::transcript::capsule::REV_1)
		})
		.collect()
}

struct NativeItem {
	value:  Value,
	ord:    Option<u32>,
	source: u32,
}

#[derive(Clone, Copy)]
enum ExpectedKind {
	Text,
	Think,
	Tool,
	Image,
	Opaque,
}

struct Candidate {
	block_index: usize,
	np:          u32,
	join:        Option<JoinMode>,
	fields:      BTreeMap<SmolStr, Box<RawValue>>,
	score:       usize,
}

impl Candidate {
	fn better_than(&self, other: &Self) -> bool {
		(self.score, self.fields.len(), self.np, self.block_index)
			< (other.score, other.fields.len(), other.np, other.block_index)
	}
}

fn best_candidate<D: Dialect + ?Sized>(
	blocks: &[Block],
	assigned: &[bool],
	block_index: usize,
	item: &Value,
	dialect: &D,
	rev: Rev,
) -> Candidate {
	let block = &blocks[block_index];
	if matches!(&block.kind, BlockKind::Opaque | BlockKind::Tool { .. } | BlockKind::Image { .. }) {
		let default = dialect.default_item(DefaultCtx::single(&block.kind), rev);
		let fields = diff(&default, item);
		let score = capsule_score(&fields);
		return Candidate { block_index, np: 1, join: None, fields, score };
	}

	let mut count = 1;
	while block_index.saturating_add(count) < blocks.len()
		&& !assigned[block_index + count]
		&& same_kind(&block.kind, &blocks[block_index + count].kind)
	{
		count += 1;
	}
	let mut best = None;
	let mut following = SmallVec::<&BlockKind, 4>::new();
	for np in 1..=count {
		let modes: &[Option<JoinMode>] = if np == 1 {
			&[None, Some(JoinMode::Split)]
		} else {
			&[Some(JoinMode::Split), Some(JoinMode::Jnn), Some(JoinMode::J)]
		};
		for join in modes {
			let ctx = DefaultCtx::grouped(&block.kind, &following, *join);
			let default = dialect.default_item(ctx, rev);
			let fields = diff(&default, item);
			let score = capsule_score(&fields);
			let candidate = Candidate {
				block_index,
				np: u32::try_from(np).unwrap_or(u32::MAX),
				join: *join,
				fields,
				score,
			};
			if best
				.as_ref()
				.is_none_or(|current| candidate.better_than(current))
			{
				best = Some(candidate);
			}
		}
		if np < count {
			following.push(&blocks[block_index + np].kind);
		}
	}
	best.expect("every block has at least one replay candidate")
}

fn expected_kind(item: &Value, dialect: &str) -> ExpectedKind {
	let item_type = item
		.get(if dialect == "ant" { "t" } else { "type" })
		.and_then(Value::as_str);
	match (dialect, item_type) {
		("oai", Some("reasoning")) | ("ant", Some("think")) => ExpectedKind::Think,
		("oai", Some("message")) | ("ant", Some("text")) => ExpectedKind::Text,
		("oai", Some("function_call" | "custom_tool_call")) | ("ant", Some("tool")) => {
			ExpectedKind::Tool
		},
		("oai" | "ant", Some("image")) => ExpectedKind::Image,
		_ => ExpectedKind::Opaque,
	}
}

const fn kind_matches(kind: &BlockKind, expected: ExpectedKind) -> bool {
	matches!(
		(kind, expected),
		(BlockKind::Text { .. }, ExpectedKind::Text)
			| (BlockKind::Think { .. }, ExpectedKind::Think)
			| (BlockKind::Tool { .. }, ExpectedKind::Tool)
			| (BlockKind::Image { .. }, ExpectedKind::Image)
			| (BlockKind::Opaque, ExpectedKind::Opaque)
	)
}

const fn same_kind(left: &BlockKind, right: &BlockKind) -> bool {
	matches!(
		(left, right),
		(BlockKind::Text { .. }, BlockKind::Text { .. })
			| (BlockKind::Think { .. }, BlockKind::Think { .. })
			| (BlockKind::Tool { .. }, BlockKind::Tool { .. })
			| (BlockKind::Image { .. }, BlockKind::Image { .. })
			| (BlockKind::Opaque, BlockKind::Opaque)
	)
}

fn matching_replay<'a>(block: &'a Block, dialect: &str) -> Option<&'a Replay> {
	block
		.re
		.as_ref()
		.filter(|replay| replay.p.0.as_str() == dialect)
}

fn capsule_score(fields: &BTreeMap<SmolStr, Box<RawValue>>) -> usize {
	fields
		.iter()
		.filter(|(key, _)| !key.as_str().starts_with('~'))
		.map(|(key, value)| key.len().saturating_add(value.get().len()))
		.sum()
}
