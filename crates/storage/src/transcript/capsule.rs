//! Peek-free dialect defaults and replay-capsule overlays.

use std::{collections::BTreeMap, iter::FusedIterator};

use omp_core::Str;
use serde_json::{Map, Value, json, value::RawValue};
use smallvec::SmallVec;

use crate::transcript::block::{BlockKind, DialectId};

/// A revision of the deterministic dialect-default rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rev(
	/// The dialect rules revision number.
	pub u16,
);

/// The first revision of the dialect-default rules.
pub const REV_1: Rev = Rev(1);

/// The way text from a multi-block native item is recombined.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JoinMode {
	/// Split every block text on a blank line and render each part separately.
	Split,
	/// Join block texts with a blank line and render the result as one part.
	Jnn,
	/// Join block texts without a separator and render the result as one part.
	J,
}

impl JoinMode {
	fn parse(marker: &str) -> Option<Self> {
		match marker {
			"split" => Some(Self::Split),
			"jnn" => Some(Self::Jnn),
			"j" => Some(Self::J),
			_ => None,
		}
	}
}

/// Typed reconstruction metadata removed from a replay capsule before wire
/// emission.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Markers {
	/// Rendered-default fields which were absent from the native item.
	pub omit: SmallVec<Str, 4>,
	/// Number of consecutive same-kind blocks projected by this native item.
	pub np:   Option<u32>,
	/// Text recombination mode for a multi-block item.
	pub join: Option<JoinMode>,
	/// Explicit native item position used to recover a migrated ordering
	/// permutation.
	pub ord:  Option<u32>,
}

/// Peek-free input to a dialect's default renderer.
///
/// This type contains only neutral block content and join metadata. In
/// particular, it cannot carry a native item or a replay capsule, making the
/// no-peeking rule structural.
pub struct DefaultCtx<'a> {
	kind:      &'a BlockKind,
	following: &'a [&'a BlockKind],
	join:      Option<JoinMode>,
}

impl<'a> DefaultCtx<'a> {
	/// Builds context for a native item projected by one block.
	#[must_use]
	pub const fn single(kind: &'a BlockKind) -> Self {
		Self { kind, following: &[], join: None }
	}

	/// Builds context for an item spanning this block and same-kind followers.
	#[must_use]
	pub const fn grouped(
		kind: &'a BlockKind,
		following: &'a [&'a BlockKind],
		join: Option<JoinMode>,
	) -> Self {
		Self { kind, following, join }
	}

	/// Returns the leading neutral block kind.
	#[must_use]
	pub const fn kind(&self) -> &'a BlockKind {
		self.kind
	}

	/// Returns the number of neutral blocks projected by the item.
	#[must_use]
	pub fn np(&self) -> u32 {
		u32::try_from(self.following.len().saturating_add(1)).unwrap_or(u32::MAX)
	}

	/// Returns the requested multi-part join mode.
	#[must_use]
	pub const fn join(&self) -> Option<JoinMode> {
		self.join
	}
}

/// A provider dialect with deterministic, peek-free native-item defaults.
pub trait Dialect {
	/// Returns the stable dialect identifier stored in replay capsules.
	#[must_use]
	fn id(&self) -> DialectId;

	/// Renders a native-item default using only neutral block content and a
	/// rules revision.
	#[must_use]
	fn default_item(&self, ctx: DefaultCtx<'_>, rev: Rev) -> Value;
}

/// `OpenAI` Responses-family dialect defaults.
pub struct Oai;

impl Dialect for Oai {
	fn id(&self) -> DialectId {
		DialectId(Str::new_static("oai"))
	}

	fn default_item(&self, ctx: DefaultCtx<'_>, rev: Rev) -> Value {
		debug_assert_eq!(rev, REV_1, "unsupported OpenAI capsule revision");
		match ctx.kind {
			BlockKind::Think { .. } => {
				let summary = text_parts(&ctx, true)
					.into_iter()
					.map(|text| json!({ "type": "summary_text", "text": text }))
					.collect::<Vec<_>>();
				json!({ "type": "reasoning", "summary": summary })
			},
			BlockKind::Text { .. } => {
				let content = text_parts(&ctx, false)
					.into_iter()
					.map(|text| {
						json!({
							"type": "output_text",
							"text": text,
							"annotations": [],
							"logprobs": []
						})
					})
					.collect::<Vec<_>>();
				json!({
					"type": "message",
					"role": "assistant",
					"status": "completed",
					"phase": "final_answer",
					"content": content
				})
			},
			BlockKind::Tool { id, name, wire, args } => match wire {
				Some(wire) => json!({
					"type": "custom_tool_call",
					"status": "completed",
					"call_id": id.0,
					"name": wire,
					"input": args
				}),
				None => json!({
					"type": "function_call",
					"status": "completed",
					"call_id": id.0,
					"name": name,
					"arguments": args
				}),
			},
			BlockKind::Image { blob } => json!({ "type": "image", "blob": blob }),
			BlockKind::Opaque => Value::Object(Map::new()),
		}
	}
}

/// Anthropic block-wire dialect defaults.
pub struct Ant;

impl Dialect for Ant {
	fn id(&self) -> DialectId {
		DialectId(Str::new_static("ant"))
	}

	fn default_item(&self, ctx: DefaultCtx<'_>, rev: Rev) -> Value {
		debug_assert_eq!(rev, REV_1, "unsupported Anthropic capsule revision");
		match ctx.kind {
			BlockKind::Text { .. } => {
				json!({ "t": "text", "text": joined_text(&ctx) })
			},
			BlockKind::Think { .. } => {
				json!({ "t": "think", "text": joined_text(&ctx) })
			},
			BlockKind::Tool { id, name, wire, args } => json!({
				"t": "tool",
				"id": id.0,
				"name": name,
				"wire": wire,
				"args": args
			}),
			BlockKind::Image { blob } => json!({ "t": "image", "blob": blob }),
			BlockKind::Opaque => Value::Object(Map::new()),
		}
	}
}

/// Separates typed reconstruction markers from native wire fields.
///
/// Every key beginning with `~` is consumed, including unknown future markers,
/// so no reserved marker can accidentally reach provider wire output.
pub fn split_markers(
	f: &BTreeMap<Str, Box<RawValue>>,
) -> (Markers, impl Clone + DoubleEndedIterator<Item = (&Str, &RawValue)> + FusedIterator + '_)
{
	let omit = f
		.get("~omit")
		.and_then(|raw| serde_json::from_str(raw.get()).ok())
		.unwrap_or_default();
	let np = f
		.get("~np")
		.and_then(|raw| serde_json::from_str(raw.get()).ok());
	let join = f
		.get("~m")
		.and_then(|raw| serde_json::from_str::<&str>(raw.get()).ok())
		.and_then(JoinMode::parse);
	let ord = f
		.get("~ord")
		.and_then(|raw| serde_json::from_str(raw.get()).ok());
	let markers = Markers { omit, np, join, ord };
	(
		markers,
		f.iter()
			.filter(|(key, _)| !key.as_str().starts_with('~'))
			.map(|(key, raw)| (key, raw.as_ref())),
	)
}

/// Computes a whole-field capsule from a peek-free default and an actual native
/// item.
///
/// Fields are replaced atomically. Default fields absent from `actual` are
/// represented by the typed `~omit` marker.
#[must_use]
pub fn diff(default: &Value, actual: &Value) -> BTreeMap<Str, Box<RawValue>> {
	let mut fields = BTreeMap::new();
	let default = default.as_object();
	let actual = actual.as_object();

	if let Some(actual) = actual {
		for (key, value) in actual {
			if default.and_then(|object| object.get(key)) != Some(value) {
				fields.insert(Str::new(key), to_raw(value));
			}
		}
	}

	if let Some(default) = default {
		let omit = default
			.keys()
			.filter(|key| actual.is_none_or(|object| !object.contains_key(*key)))
			.map(Str::new)
			.collect::<SmallVec<Str, 4>>();
		if !omit.is_empty() {
			fields.insert(Str::new_static("~omit"), to_raw(&omit));
		}
	}
	fields
}

/// Applies whole-field capsule replacements and omissions to a rendered
/// default.
#[must_use]
pub fn overlay(default: Value, f: &BTreeMap<Str, Box<RawValue>>) -> Value {
	let mut object = match default {
		Value::Object(object) => object,
		_ => Map::new(),
	};
	let (markers, fields) = split_markers(f);
	for (key, raw) in fields {
		let value =
			serde_json::from_str(raw.get()).expect("RawValue always contains one valid JSON value");
		object.insert(key.as_str().to_owned(), value);
	}
	for key in markers.omit {
		object.remove(key.as_str());
	}
	Value::Object(object)
}

/// Inserts a writer-synthesized reconstruction marker as raw JSON.
pub(crate) fn insert_marker<T: serde::Serialize>(
	fields: &mut BTreeMap<Str, Box<RawValue>>,
	key: &'static str,
	value: &T,
) {
	fields.insert(Str::new_static(key), to_raw(value));
}

fn to_raw<T: serde::Serialize + ?Sized>(value: &T) -> Box<RawValue> {
	serde_json::value::to_raw_value(value).expect("serializing JSON values cannot fail")
}

fn block_text(kind: &BlockKind) -> Option<&str> {
	match kind {
		BlockKind::Text { text } | BlockKind::Think { text } => Some(text.as_str()),
		_ => None,
	}
}

fn source_texts<'a>(
	ctx: &DefaultCtx<'a>,
) -> impl Clone + DoubleEndedIterator<Item = &'a str> + FusedIterator + 'a {
	let kind = ctx.kind;
	let following = ctx.following;
	std::iter::once(kind)
		.chain(following.iter().copied())
		.filter_map(block_text)
}

fn text_parts(ctx: &DefaultCtx<'_>, empty_is_no_parts: bool) -> SmallVec<String, 4> {
	let source = source_texts(ctx);
	let mut parts = SmallVec::new();
	match ctx.join {
		Some(JoinMode::Split) => {
			for text in source {
				parts.extend(text.split("\n\n").map(str::to_owned));
			}
		},
		Some(JoinMode::Jnn) => parts.push(join_texts(source, "\n\n")),
		Some(JoinMode::J) => parts.push(join_texts(source, "")),
		None => parts.extend(source.map(str::to_owned)),
	}
	if empty_is_no_parts && parts.len() == 1 && parts[0].is_empty() {
		parts.clear();
	}
	parts
}

fn joined_text(ctx: &DefaultCtx<'_>) -> String {
	join_texts(
		source_texts(ctx),
		if matches!(ctx.join, Some(JoinMode::J)) {
			""
		} else {
			"\n\n"
		},
	)
}

fn join_texts<'a>(texts: impl Iterator<Item = &'a str> + Clone, separator: &str) -> String {
	let (text_len, count) = texts
		.clone()
		.fold((0_usize, 0_usize), |(len, count), text| {
			(len.saturating_add(text.len()), count.saturating_add(1))
		});
	let capacity = text_len.saturating_add(separator.len().saturating_mul(count.saturating_sub(1)));
	let mut joined = String::with_capacity(capacity);
	for (index, text) in texts.enumerate() {
		if index != 0 {
			joined.push_str(separator);
		}
		joined.push_str(text);
	}
	joined
}
