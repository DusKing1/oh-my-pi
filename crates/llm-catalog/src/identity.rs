//! Model lineage, reseller-reference lookup, and effort-variant collapse.

use std::{
	borrow::Cow,
	collections::{BTreeMap, BTreeSet},
	fmt,
	str::FromStr,
};

use omp_core::{Str, fmts, str::StrExt};
use omp_llm_types::Effort;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use strum::IntoStaticStr;
use thiserror::Error;

use crate::models::{ModelCard, ModelThinkingEffort, PriceUnit};

/// Returns a coarse vendor-lineage token for a model identifier.
///
/// Namespace prefixes and reseller decoration are ignored where possible. The
/// result is intentionally comparison-only: the vocabulary can grow as vendors
/// introduce new model families. Unknown identifiers return an empty token so
/// callers can fall back to their provider id.
#[must_use]
pub fn family_token(model_id: &str) -> Str {
	let lower = model_id.trim().to_ascii_lowercase_str();
	let bare = lower.rsplit('/').next().unwrap_or(lower.as_str());
	let family = if contains_segment(&lower, "claude") || lower.starts_with("anthropic/") {
		"anthropic"
	} else if lower.contains("gpt-oss") {
		"gpt-oss"
	} else if is_openai_id(&lower, bare) {
		"openai"
	} else if lower.contains("moonshotai/kimi") || has_vendor_fragment(&lower, "kimi") {
		"kimi"
	} else if lower.contains("qwen") {
		"qwen"
	} else if lower.contains("minimax") {
		"minimax"
	} else if lower.contains("deepseek") {
		"deepseek"
	} else if lower.contains("mimo") {
		"mimo"
	} else if starts_vendor_model(bare, "gemma") {
		"gemma"
	} else if starts_vendor_model(bare, "glm") {
		"glm"
	} else if starts_vendor_model(bare, "gemini") {
		"gemini"
	} else if starts_vendor_model(bare, "grok")
		|| lower.starts_with("x-ai/")
		|| lower.starts_with("xai/")
	{
		"xai"
	} else if starts_vendor_model(bare, "llama") || lower.starts_with("meta-llama/") {
		"meta"
	} else if starts_vendor_model(bare, "mistral") || starts_vendor_model(bare, "mixtral") {
		"mistral"
	} else if starts_vendor_model(bare, "command") || lower.starts_with("cohere/") {
		"cohere"
	} else if starts_vendor_model(bare, "jamba") || lower.starts_with("ai21/") {
		"ai21"
	} else if starts_vendor_model(bare, "nova") || starts_vendor_model(bare, "titan") {
		"amazon"
	} else if starts_vendor_model(bare, "doubao") {
		"bytedance"
	} else if starts_vendor_model(bare, "ernie") {
		"baidu"
	} else if starts_vendor_model(bare, "step") {
		"stepfun"
	} else {
		""
	};
	Str::new(family)
}
/// Environment variable used to override model-prompt dialect selection.
pub const DIALECT_ENV: &str = "OMP_DIALECT";

/// Safe fallback for model families without an owned dialect mapping.
pub const FALLBACK_DIALECT: Dialect = Dialect::Xml;

/// Model-authored prompt and in-band tool syntax owned by the dialect
/// subsystem.
#[derive(
	Clone,
	Copy,
	Debug,
	Default,
	Deserialize,
	Eq,
	Hash,
	IntoStaticStr,
	Ord,
	PartialEq,
	PartialOrd,
	Serialize,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "lowercase")]
pub enum Dialect {
	/// GLM XML-like tool syntax.
	Glm,
	/// Hermes tool-call syntax.
	Hermes,
	/// Kimi `ChatML` tool syntax.
	Kimi,
	/// Generic XML fallback syntax.
	#[default]
	Xml,
	/// Anthropic-style in-band function syntax.
	Anthropic,
	/// `DeepSeek` markup language syntax.
	#[serde(rename = "deepseek")]
	DeepSeek,
	/// `OpenAI` Harmony channel syntax.
	Harmony,
	/// Qwen 3 tool-call syntax.
	Qwen3,
	/// Gemini Python tool-code syntax.
	Gemini,
	/// Gemma control-token syntax.
	Gemma,
	/// `MiniMax` XML-like function syntax.
	#[serde(rename = "minimax")]
	MiniMax,
}

impl Dialect {
	/// Every owned dialect in stable display order.
	pub const ALL: [Self; 11] = [
		Self::Glm,
		Self::Hermes,
		Self::Kimi,
		Self::Xml,
		Self::Anthropic,
		Self::DeepSeek,
		Self::Harmony,
		Self::Qwen3,
		Self::Gemini,
		Self::Gemma,
		Self::MiniMax,
	];

	/// Converts this dialect to its canonical configuration spelling.
	#[must_use]
	pub const fn into_str(self) -> &'static str {
		match self {
			Self::Glm => "glm",
			Self::Hermes => "hermes",
			Self::Kimi => "kimi",
			Self::Xml => "xml",
			Self::Anthropic => "anthropic",
			Self::DeepSeek => "deepseek",
			Self::Harmony => "harmony",
			Self::Qwen3 => "qwen3",
			Self::Gemini => "gemini",
			Self::Gemma => "gemma",
			Self::MiniMax => "minimax",
		}
	}

	/// Returns the canonical configuration spelling.
	#[must_use]
	pub const fn as_str(self) -> &'static str {
		self.into_str()
	}
}

impl fmt::Display for Dialect {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(self.as_str())
	}
}

impl FromStr for Dialect {
	type Err = ParseDialectError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		parse_dialect(value).ok_or_else(|| ParseDialectError::new(value))
	}
}

/// Dialect choice made before a model request is rendered.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DialectSelection {
	/// Infer the dialect from the model family.
	#[default]
	Auto,
	/// Retain the provider's native tool and reasoning channels.
	Native,
	/// Force one owned model-prompt dialect.
	Explicit(Dialect),
}

impl DialectSelection {
	/// Resolves this choice for a model without consulting process state.
	///
	/// Native selection returns `None`; every owned choice returns its effective
	/// dialect.
	#[must_use]
	pub fn select(self, model_id: &str) -> Option<Dialect> {
		match self {
			Self::Auto => Some(preferred_dialect(model_id)),
			Self::Native => None,
			Self::Explicit(dialect) => Some(dialect),
		}
	}

	/// Applies an optional `OMP_DIALECT` value, then resolves for `model_id`.
	///
	/// Callers pass only the value of [`DIALECT_ENV`]. No legacy environment
	/// variable or dialect alias is recognized.
	pub fn resolve(
		self,
		model_id: &str,
		omp_dialect: Option<&str>,
	) -> Result<Option<Dialect>, ParseDialectError> {
		let selection = omp_dialect.map(str::parse).transpose()?.unwrap_or(self);
		Ok(selection.select(model_id))
	}
}

impl From<Dialect> for DialectSelection {
	fn from(value: Dialect) -> Self {
		Self::Explicit(value)
	}
}

impl fmt::Display for DialectSelection {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Auto => formatter.write_str("auto"),
			Self::Native => formatter.write_str("native"),
			Self::Explicit(dialect) => dialect.fmt(formatter),
		}
	}
}

impl FromStr for DialectSelection {
	type Err = ParseDialectError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		let value = value.trim();
		if value.eq_ignore_ascii_case("auto") {
			Ok(Self::Auto)
		} else if value.eq_ignore_ascii_case("native") {
			Ok(Self::Native)
		} else {
			value.parse().map(Self::Explicit)
		}
	}
}

/// Error returned for an unknown dialect or dialect selection spelling.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("unknown model-prompt dialect `{value}`")]
pub struct ParseDialectError {
	value: Str,
}

impl ParseDialectError {
	fn new(value: &str) -> Self {
		Self { value: Str::new(value.trim()) }
	}

	/// Returns the rejected configuration value.
	#[must_use]
	pub fn value(&self) -> &str {
		&self.value
	}
}

/// Returns the preferred owned dialect for a model family.
///
/// Hermes is intentionally not auto-selected. Unknown families use
/// [`FALLBACK_DIALECT`].
#[must_use]
pub fn preferred_dialect(model_id: &str) -> Dialect {
	match family_token(model_id).as_str() {
		"anthropic" => Dialect::Anthropic,
		"glm" => Dialect::Glm,
		"gemini" => Dialect::Gemini,
		"gemma" => Dialect::Gemma,
		"kimi" => Dialect::Kimi,
		"qwen" => Dialect::Qwen3,
		"deepseek" => Dialect::DeepSeek,
		"minimax" => Dialect::MiniMax,
		"openai" | "gpt-oss" => Dialect::Harmony,
		_ => FALLBACK_DIALECT,
	}
}

fn parse_dialect(value: &str) -> Option<Dialect> {
	let value = value.trim();
	Dialect::ALL
		.into_iter()
		.find(|dialect| value.eq_ignore_ascii_case(dialect.as_str()))
}

/// Lookup tables for resolving decorated reseller model ids to bundled cards.
#[derive(Clone, Debug, Default, bon::Builder)]
#[non_exhaustive]
pub struct ModelReferenceIndex<'models> {
	models:       &'models [ModelCard],
	retained:     usize,
	exact:        BTreeMap<Str, usize>,
	suffix_alias: BTreeMap<Str, usize>,
}

impl ModelReferenceIndex<'_> {
	/// Returns the number of candidate reference cards retained by the index.
	#[must_use]
	pub const fn len(&self) -> usize {
		self.retained
	}

	/// Returns whether the reference index is empty.
	#[must_use]
	pub const fn is_empty(&self) -> bool {
		self.retained == 0
	}
}

/// Builds a pure reseller-reference index from bundled model cards.
///
/// Duplicate ids prefer larger context/output limits, then complete prompt
/// cache pricing, then first-party `OpenAI`. Zero-cost `xai-oauth` subscription
/// rows are excluded because their inflated limits must not outrank public Grok
/// references.
#[must_use]
pub fn build_model_reference_index(models: &[ModelCard]) -> ModelReferenceIndex<'_> {
	let mut exact = BTreeMap::new();
	let mut retained = 0;
	for (index, candidate) in models.iter().enumerate() {
		if is_zero_cost_xai_oauth_reference(candidate) {
			continue;
		}
		retained += 1;
		for id in [&candidate.model, &candidate.id] {
			let key = normalize_reference_key(id);
			if should_replace_index(exact.get(&key).copied(), index, models) {
				exact.insert(key, index);
			}
		}
	}
	let mut suffix_alias = BTreeMap::new();
	for &index in exact.values() {
		let reference = &models[index];
		let suffix = reference
			.id
			.rsplit('/')
			.next()
			.unwrap_or(reference.id.as_str());
		let Some(alias) = longest_model_like_segment(suffix) else {
			continue;
		};
		let alias = normalize_reference_key(&alias);
		if should_replace_index(suffix_alias.get(&alias).copied(), index, models) {
			suffix_alias.insert(alias, index);
		}
	}
	ModelReferenceIndex { models, retained, exact, suffix_alias }
}

/// Resolves a decorated reseller id to its preferred bundled upstream card.
///
/// Portkey-style `@provider/model` ids are deliberately opaque: fuzzy matching
/// them can select an unrelated provider's similarly named model.
#[must_use]
pub fn resolve_model_reference<'models>(
	model_id: &str,
	index: &ModelReferenceIndex<'models>,
) -> Option<&'models ModelCard> {
	if model_id.starts_with('@') {
		return None;
	}
	for candidate in reference_candidate_ids(model_id) {
		let key = normalize_reference_key(&candidate);
		if let Some(&position) = index
			.exact
			.get(&key)
			.or_else(|| index.suffix_alias.get(&key))
		{
			return index.models.get(position);
		}
	}
	None
}

/// Inherits upstream pricing and unknown limits while preserving reseller
/// identity and provider-local behavior.
///
/// Discovered nonzero limits and price components win over bundled values.
/// Missing or explicit-zero price components are filled independently, matching
/// the TypeScript mapper's use of the bundled card as the metadata base.
/// Behavior and wire routing never cross providers: the reseller's request
/// model, headers, compatibility policy, and transport remain attached to its
/// own logical card.
pub fn inherit_model_reference(model: &mut ModelCard, reference: &ModelCard) {
	for &reference_price in &reference.pricing {
		if let Some(price) = model
			.pricing
			.iter_mut()
			.find(|price| price.unit == reference_price.unit)
		{
			if price.nanos_usd == 0 {
				*price = reference_price;
			}
		} else {
			model.pricing.push(reference_price);
		}
	}
	if model.context_window == 0 {
		model.context_window = reference.context_window;
	}
	if model.max_output_tokens == 0 {
		model.max_output_tokens = reference.max_output_tokens;
	}
	if model.family.is_empty() {
		model.family.clone_from(&reference.family);
	}
}

/// Resolves a reseller card and applies upstream pricing and limit inheritance.
///
/// Returns `true` when a bundled reference was found.
pub fn resolve_and_inherit_model_reference(
	model: &mut ModelCard,
	index: &ModelReferenceIndex<'_>,
) -> bool {
	let mut current_position: Option<usize> = None;
	let mut visited = SmallVec::<usize, 8>::new();
	loop {
		let next_position = if let Some(position) = current_position {
			let current = &index.models[position];
			resolve_model_reference_position_excluding_identity(
				current.model.as_str(),
				current.provider.as_str(),
				index,
			)
		} else {
			resolve_model_reference_position_excluding_identity(
				model.model.as_str(),
				model.provider.as_str(),
				index,
			)
		};
		let Some(position) = next_position else {
			break;
		};
		if visited.contains(&position) {
			break;
		}
		visited.push(position);
		inherit_model_reference(model, &index.models[position]);
		current_position = Some(position);
	}
	!visited.is_empty()
}

fn resolve_model_reference_position_excluding_identity(
	model_id: &str,
	provider: &str,
	index: &ModelReferenceIndex<'_>,
) -> Option<usize> {
	if model_id.starts_with('@') {
		return None;
	}
	for candidate in reference_candidate_ids(model_id) {
		let key = normalize_reference_key(&candidate);
		let Some(&position) = index
			.exact
			.get(&key)
			.or_else(|| index.suffix_alias.get(&key))
		else {
			continue;
		};
		let reference = index.models.get(position)?;
		if reference.provider != provider || reference.model != model_id {
			return Some(position);
		}
	}
	None
}

/// Collapses effort-tier and `-thinking` siblings into logical model cards.
///
/// Output order follows the first member of each family. For `X` plus
/// `X-thinking`, `Off` routes to `X` and every advertised nonzero effort routes
/// to the thinking wire id. Two or more `-minimal`/`-low`/`-medium`/`-high`/
/// `-max` siblings similarly become one card with a route per tier. Limits and
/// modality/facet sets are merged; routing always retains provider wire ids.
#[must_use]
pub fn collapse_effort_variants_across_providers(models: Vec<ModelCard>) -> Vec<ModelCard> {
	let by_identity: BTreeMap<(Str, Str), usize> = models
		.iter()
		.enumerate()
		.map(|(index, model)| ((model.provider.clone(), model.model.clone()), index))
		.collect();
	let mut thinking_variants: BTreeMap<(Str, Str), SmallVec<(usize, Str), 2>> = BTreeMap::new();
	type TierVariants = BTreeMap<(Str, Str), SmallVec<(usize, Effort, Str), 6>>;
	let mut tier_variants: TierVariants = BTreeMap::new();
	for (index, model) in models.iter().enumerate() {
		if let Some(base) = strip_thinking_variant_token(model.model.as_str()) {
			thinking_variants
				.entry((model.provider.clone(), base))
				.or_default()
				.push((index, model.model.clone()));
		}
		if let Some((base, effort)) = strip_effort_tier(model.model.as_str()) {
			tier_variants
				.entry((model.provider.clone(), base))
				.or_default()
				.push((index, effort, model.model.clone()));
		}
	}

	let mut replacements = BTreeMap::new();
	let mut consumed = BTreeSet::new();
	for ((provider, logical), variants) in thinking_variants {
		let Some(&base_index) = by_identity.get(&(provider, logical.clone())) else {
			continue;
		};
		let Some(&(thinking_index, ref wire_id)) = variants.first() else {
			continue;
		};
		if known_pricing_differs(&models[base_index], &models[thinking_index]) {
			continue;
		}
		let member_indices = [base_index, thinking_index];
		let first = base_index.min(thinking_index);
		let mut card = merge_family(&models, &member_indices, &logical, first);
		card.behavior.clone_from(&models[base_index].behavior);
		if let Some(thinking) = models[thinking_index].behavior.thinking.as_ref() {
			card.behavior.thinking = Some(thinking.clone());
		}
		card.name.clone_from(&models[base_index].name);
		let efforts: SmallVec<Effort, 6> = if models[thinking_index].efforts.is_empty() {
			[Effort::Minimal, Effort::Low, Effort::Medium, Effort::High]
				.into_iter()
				.collect()
		} else {
			models[thinking_index].efforts.clone()
		};
		card.efforts.clear();
		for effort in efforts {
			if effort != Effort::Off && !card.efforts.contains(&effort) {
				card.efforts.push(effort);
			}
		}
		card.effort_routing.clear();
		card
			.effort_routing
			.insert(Effort::Off, models[base_index].model.clone());
		for &effort in &card.efforts {
			card.effort_routing.insert(effort, wire_id.clone());
		}
		sync_behavior_effort_routing(&mut card, false);
		consumed.extend(member_indices);
		replacements.insert(first, card);
	}

	for ((provider, logical), mut variants) in tier_variants {
		if variants.len() < 2 {
			continue;
		}
		variants.sort_by_key(|(_, effort, _)| *effort);
		let mut member_indices: SmallVec<usize, 8> =
			variants.iter().map(|(index, ..)| *index).collect();
		let base_index = by_identity.get(&(provider, logical.clone())).copied();
		if let Some(index) = base_index {
			member_indices.push(index);
		}
		if member_indices.iter().any(|index| consumed.contains(index)) {
			continue;
		}
		let first = member_indices.iter().copied().min().unwrap_or_default();
		let mut card = merge_family(&models, &member_indices, &logical, first);
		card.efforts.clear();
		card.effort_routing.clear();
		if let Some(index) = base_index {
			card
				.effort_routing
				.insert(Effort::Off, models[index].model.clone());
		}
		for (_, effort, wire_id) in variants {
			if !card.efforts.contains(&effort) {
				card.efforts.push(effort);
			}
			card.effort_routing.insert(effort, wire_id);
		}
		sync_behavior_effort_routing(&mut card, true);
		consumed.extend(member_indices);
		replacements.insert(first, card);
	}

	let mut output =
		Vec::with_capacity(models.len().saturating_sub(consumed.len()) + replacements.len());
	for (index, model) in models.into_iter().enumerate() {
		if let Some(replacement) = replacements.remove(&index) {
			output.push(replacement);
		} else if !consumed.contains(&index) {
			output.push(model);
		}
	}
	output
}

fn merge_family(models: &[ModelCard], indices: &[usize], logical: &Str, first: usize) -> ModelCard {
	let mut card = models[first].clone();
	card.model.clone_from(logical);
	card.id = fmts!("{}/{}", card.provider, logical);
	card.reasoning = true;
	card.context_window = indices
		.iter()
		.map(|&index| models[index].context_window)
		.max()
		.unwrap_or_default();
	card.max_output_tokens = indices
		.iter()
		.map(|&index| models[index].max_output_tokens)
		.max()
		.unwrap_or_default();
	for &index in indices {
		merge_unique(&mut card.facets, &models[index].facets);
		merge_unique(&mut card.inputs, &models[index].inputs);
		merge_unique(&mut card.outputs, &models[index].outputs);
	}
	card
}

fn sync_behavior_effort_routing(card: &mut ModelCard, replace_efforts: bool) {
	let Some(thinking) = card.behavior.thinking.as_mut() else {
		return;
	};
	if replace_efforts {
		thinking.efforts = card.efforts.iter().copied().map(native_effort).collect();
	}
	thinking.effort_routing.clear();
	if let Some(route) = card.effort_routing.get(&Effort::Off) {
		thinking
			.effort_routing
			.insert(ModelThinkingEffort::Off, route.clone());
	}
	for &effort in &thinking.efforts {
		if let Some(route) = card.effort_routing.get(&effort.portable()) {
			thinking.effort_routing.insert(effort, route.clone());
		}
	}
}

const fn native_effort(effort: Effort) -> ModelThinkingEffort {
	match effort {
		Effort::Off => ModelThinkingEffort::Off,
		Effort::Minimal => ModelThinkingEffort::Minimal,
		Effort::Low => ModelThinkingEffort::Low,
		Effort::Medium => ModelThinkingEffort::Medium,
		Effort::High => ModelThinkingEffort::High,
		Effort::XHigh => ModelThinkingEffort::XHigh,
		Effort::Max => ModelThinkingEffort::Max,
		_ => ModelThinkingEffort::Max,
	}
}

fn merge_unique<T: Copy + PartialEq, const N: usize>(target: &mut SmallVec<T, N>, source: &[T]) {
	for &value in source {
		if !target.contains(&value) {
			target.push(value);
		}
	}
}

fn is_zero_cost_xai_oauth_reference(candidate: &ModelCard) -> bool {
	candidate.provider == "xai-oauth" && candidate.pricing.iter().all(|price| price.nanos_usd == 0)
}

fn should_replace_index(existing: Option<usize>, candidate: usize, models: &[ModelCard]) -> bool {
	let Some(existing) = existing else {
		return true;
	};
	let existing = &models[existing];
	let candidate = &models[candidate];
	if candidate.context_window != existing.context_window {
		return candidate.context_window > existing.context_window;
	}
	if candidate.max_output_tokens != existing.max_output_tokens {
		return candidate.max_output_tokens > existing.max_output_tokens;
	}
	let existing_cache = has_cache_pricing(existing);
	let candidate_cache = has_cache_pricing(candidate);
	if candidate_cache != existing_cache {
		return candidate_cache;
	}
	existing.provider != "openai" && candidate.provider == "openai"
}

fn has_cache_pricing(model: &ModelCard) -> bool {
	model.pricing.iter().any(|price| {
		matches!(price.unit, PriceUnit::MtokCacheRead | PriceUnit::MtokCacheWrite)
			&& price.nanos_usd > 0
	})
}

fn known_pricing_differs(left: &ModelCard, right: &ModelCard) -> bool {
	let left_priced =
		model_price(left, PriceUnit::MtokInput) > 0 || model_price(left, PriceUnit::MtokOutput) > 0;
	let right_priced =
		model_price(right, PriceUnit::MtokInput) > 0 || model_price(right, PriceUnit::MtokOutput) > 0;
	left_priced
		&& right_priced
		&& [
			PriceUnit::MtokInput,
			PriceUnit::MtokOutput,
			PriceUnit::MtokCacheRead,
			PriceUnit::MtokCacheWrite,
		]
		.into_iter()
		.any(|unit| model_price(left, unit) != model_price(right, unit))
}

fn model_price(model: &ModelCard, unit: PriceUnit) -> u64 {
	model
		.pricing
		.iter()
		.find(|price| price.unit == unit)
		.map_or(0, |price| price.nanos_usd)
}

fn normalize_reference_key(value: &str) -> Str {
	value.trim().to_ascii_lowercase_str()
}

fn reference_candidate_ids(model_id: &str) -> SmallVec<Str, 16> {
	let mut candidates = SmallVec::new();
	let mut queue = SmallVec::<Str, 16>::new();
	queue.push(Str::new(model_id));
	let mut next = 0;
	while let Some(queued) = queue.get(next) {
		let candidate = normalize_whitespace_str(queued);
		next += 1;
		if candidate.is_empty() || candidates.contains(&candidate) {
			continue;
		}
		candidates.push(candidate.clone());
		queue.extend(bracket_stripped_candidates(&candidate));
		queue.extend(model_like_segments(&candidate));
		let lower = candidate.as_str().to_ascii_lowercase_str();
		for suffix in [":cloud", "-cloud"] {
			if lower.ends_with(suffix) {
				queue.push(candidate.slice(..candidate.len() - suffix.len()));
			}
		}
		if let Some((_, suffix)) = candidate.rsplit_once('/') {
			queue.push(candidate.slice_ref(suffix));
		}
		if candidate.contains(':') {
			queue.push(Str::new(candidate.replace(':', "-")));
		}
		if lower != candidate {
			queue.push(lower);
		}
		if let Some(stripped) = strip_reference_trailing_marker(&candidate) {
			queue.push(stripped);
		}
	}
	candidates
}

fn bracket_stripped_candidates(value: &str) -> SmallVec<Str, 3> {
	if !value
		.chars()
		.any(|character| matches!(character, '[' | ']' | '【' | '】'))
	{
		return SmallVec::new();
	}
	let normalized = normalize_whitespace(value);
	let leading = strip_leading_brackets(&normalized);
	let trailing = strip_trailing_brackets(&normalized);
	let both = strip_trailing_brackets(leading);
	let mut output = SmallVec::new();
	for candidate in [both, leading, trailing] {
		let candidate = Str::new(normalize_whitespace(candidate));
		if !candidate.is_empty()
			&& candidate.as_str() != normalized.as_ref()
			&& !output.contains(&candidate)
		{
			output.push(candidate);
		}
	}
	output
}

fn strip_leading_brackets(value: &str) -> &str {
	let mut rest = value.trim_start();
	while let Some((close, close_len)) = (match rest.chars().next() {
		Some('[') => Some((']', 1)),
		Some('【') => Some(('】', '】'.len_utf8())),
		_ => None,
	}) && let Some(index) = rest.find(close)
	{
		rest = rest[index + close_len..].trim_start();
	}
	rest
}

fn strip_trailing_brackets(value: &str) -> &str {
	let mut rest = value.trim_end();
	while let Some((open, close_len)) = (match rest.chars().next_back() {
		Some(']') => Some(('[', 1)),
		Some('】') => Some(('【', '】'.len_utf8())),
		_ => None,
	}) && let Some(index) = rest[..rest.len() - close_len].rfind(open)
	{
		rest = rest[..index].trim_end();
	}
	rest
}

fn normalize_whitespace(value: &str) -> Cow<'_, str> {
	let trimmed = value.trim();
	let already_normalized = !trimmed
		.chars()
		.any(|character| character.is_whitespace() && character != ' ')
		&& !trimmed.as_bytes().windows(2).any(|pair| pair == b"  ");
	if already_normalized {
		return Cow::Borrowed(trimmed);
	}
	let mut words = trimmed.split_whitespace();
	let Some(first) = words.next() else {
		return Cow::Borrowed("");
	};
	let mut normalized = String::with_capacity(trimmed.len());
	normalized.push_str(first);
	for word in words {
		normalized.push(' ');
		normalized.push_str(word);
	}
	Cow::Owned(normalized)
}

fn normalize_whitespace_str(value: &Str) -> Str {
	match normalize_whitespace(value) {
		Cow::Borrowed(normalized) => value.slice_ref(normalized),
		Cow::Owned(normalized) => Str::new(normalized),
	}
}

fn model_like_segments(value: &str) -> SmallVec<Str, 8> {
	let lower = value.to_ascii_lowercase_str();
	let mut segments = SmallVec::<Str, 8>::new();
	let mut start = None;
	for (index, character) in lower.char_indices().chain([(lower.len(), ' ')]) {
		if character.is_ascii_alphanumeric() || matches!(character, '.' | ':' | '-') {
			start.get_or_insert(index);
		} else if let Some(begin) = start.take() {
			let segment = &lower[begin..index];
			if is_model_like_segment(segment) && !segments.iter().any(|seen| seen == segment) {
				segments.push(lower.slice(begin..index));
			}
		}
	}
	segments.sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
	segments
}

fn longest_model_like_segment(value: &str) -> Option<Str> {
	model_like_segments(value).into_iter().next()
}

fn is_model_like_segment(value: &str) -> bool {
	const PREFIXES: &[&str] = &[
		"claude", "gemini", "gpt", "grok", "glm", "qwen", "deepseek", "kimi", "mimo", "doubao",
		"ernie", "gpt-oss", "gemma", "minimax", "step", "command", "jamba", "llama", "o1", "o3",
		"o4", "o5",
	];
	value.bytes().any(|byte| byte.is_ascii_digit())
		&& PREFIXES.iter().any(|prefix| value.starts_with(prefix))
}

fn strip_reference_trailing_marker(value: &Str) -> Option<Str> {
	const MARKERS: &[&str] = &[
		"thinking",
		"customtools",
		"high",
		"low",
		"medium",
		"minimal",
		"xhigh",
		"free",
		"cloud",
		"exacto",
		"nitro",
		"original",
		"optimized",
		"nvfp4",
		"fp8",
		"fp4",
		"bf16",
		"int8",
		"int4",
		"search",
	];
	let lower = value.as_str().to_ascii_lowercase_str();
	for marker in MARKERS {
		let Some(prefix) = lower.strip_suffix(marker) else {
			continue;
		};
		if prefix.ends_with('-') || prefix.ends_with(':') {
			return Some(value.slice(..prefix.len().saturating_sub(1)));
		}
	}
	None
}

fn strip_thinking_variant_token(value: &str) -> Option<Str> {
	let lower = value.to_ascii_lowercase_str();
	for marker in ["-thinking", "-reasoner", "-reasoning"] {
		let mut offset = 0;
		while let Some(relative) = lower[offset..].find(marker) {
			let start = offset + relative;
			let end = start + marker.len();
			if lower[end..]
				.chars()
				.next()
				.is_none_or(|character| !character.is_ascii_alphanumeric())
			{
				return Some(fmts!("{}{}", &value[..start], &value[end..]));
			}
			offset = end;
		}
	}
	None
}

fn strip_effort_tier(value: &str) -> Option<(Str, Effort)> {
	let lower = value.to_ascii_lowercase_str();
	for (suffix, effort) in [
		("-minimal", Effort::Minimal),
		("-medium", Effort::Medium),
		("-xhigh", Effort::XHigh),
		("-high", Effort::High),
		("-low", Effort::Low),
		("-max", Effort::Max),
	] {
		if lower.ends_with(suffix) {
			return Some((Str::new(&value[..value.len() - suffix.len()]), effort));
		}
	}
	None
}

fn contains_segment(value: &str, needle: &str) -> bool {
	value
		.split(|character: char| !character.is_ascii_alphanumeric())
		.any(|segment| segment == needle)
}

fn has_vendor_fragment(value: &str, vendor: &str) -> bool {
	value.split('/').any(|segment| {
		segment
			.strip_prefix(vendor)
			.is_some_and(|suffix| suffix.starts_with('-') || suffix.starts_with('.'))
	})
}

fn starts_vendor_model(value: &str, vendor: &str) -> bool {
	value == vendor
		|| value.strip_prefix(vendor).is_some_and(|suffix| {
			suffix.starts_with('-')
				|| suffix.starts_with('.')
				|| suffix.starts_with(':')
				|| suffix.starts_with(|character: char| character.is_ascii_digit())
		})
}

fn is_openai_id(full: &str, bare: &str) -> bool {
	full.starts_with("openai/")
		|| ["gpt-", "chatgpt-", "codex-"]
			.iter()
			.any(|prefix| bare.starts_with(prefix))
		|| ["o1", "o3", "o4"].iter().any(|prefix| {
			bare == *prefix
				|| bare
					.strip_prefix(prefix)
					.is_some_and(|suffix| suffix.starts_with('-') || suffix.starts_with('.'))
		})
}

#[cfg(test)]
mod tests {
	use std::collections::BTreeMap;

	use super::*;
	use crate::{
		models::{Availability, Modality, Price, Source},
		provider::Facet,
	};

	fn card(provider: &str, model: &str) -> ModelCard {
		ModelCard {
			id:                fmts!("{provider}/{model}"),
			provider:          Str::new(provider),
			model:             Str::new(model),
			name:              Str::new(model),
			family:            family_token(model),
			facets:            std::iter::once(Facet::Chat).collect(),
			inputs:            std::iter::once(Modality::Text).collect(),
			outputs:           std::iter::once(Modality::Text).collect(),
			reasoning:         false,
			efforts:           SmallVec::new(),
			context_window:    0,
			max_output_tokens: 0,
			pricing:           SmallVec::new(),
			availability:      Availability::Unspecified,
			source:            Source::Bundled,
			blocked_until_ms:  0,
			deprecated:        false,
			updated_at_ms:     0,
			props:             omp_llm_types::Props::default(),
			effort_routing:    BTreeMap::new(),
			behavior:          crate::models::ModelBehavior::default(),
			wire:              None,
		}
	}

	#[test]
	fn family_tokens_normalize_vendor_lineages() {
		let cases = [
			("openrouter/anthropic/claude-opus-4.6", "anthropic"),
			("azure/gpt-5.4-codex", "openai"),
			("moonshotai/kimi-k2.6", "kimi"),
			("router/Qwen3-Coder", "qwen"),
			("zai/glm-5", "glm"),
			("together/deepseek-v3", "deepseek"),
			("google/gemini-3-pro", "gemini"),
			("meta-llama/llama-4-maverick", "meta"),
			("openrouter/openai/gpt-oss-120b", "gpt-oss"),
			("minimax/minimax-m2.5", "minimax"),
			("xiaomi/mimo-v2-flash", "mimo"),
			("google/gemma-3-27b-it", "gemma"),
			("xai/grok-4", "xai"),
			("mistralai/mistral-large-3", "mistral"),
			("cohere/command-r-plus", "cohere"),
			("ai21/jamba-large-1.7", "ai21"),
			("amazon/nova-pro", "amazon"),
			("volcengine/doubao-1.5-pro", "bytedance"),
			("baidu/ernie-4.5", "baidu"),
			("stepfun/step-3.5-flash", "stepfun"),
		];
		for (model, expected) in cases {
			assert_eq!(family_token(model), expected, "{model}");
		}
	}

	#[test]
	fn all_owned_dialects_round_trip_through_configuration_spelling() {
		assert_eq!(Dialect::ALL.len(), 11);
		for dialect in Dialect::ALL {
			let spelling = dialect.to_string();
			assert_eq!(spelling.parse::<Dialect>(), Ok(dialect));
			assert_eq!(spelling.parse::<DialectSelection>(), Ok(DialectSelection::Explicit(dialect)));
		}
		assert!("pi".parse::<DialectSelection>().is_err());
	}

	#[test]
	fn preferred_dialects_follow_model_family_identity() {
		let cases = [
			("anthropic/claude-opus-4.6", Dialect::Anthropic),
			("zai/glm-5", Dialect::Glm),
			("google/gemini-3-pro", Dialect::Gemini),
			("google/gemma-3-27b-it", Dialect::Gemma),
			("moonshotai/kimi-k2.6", Dialect::Kimi),
			("qwen/qwen3-coder", Dialect::Qwen3),
			("deepseek/deepseek-v3", Dialect::DeepSeek),
			("minimax/minimax-m2.5", Dialect::MiniMax),
			("openai/gpt-5.4", Dialect::Harmony),
			("openrouter/openai/gpt-oss-120b", Dialect::Harmony),
			("meta-llama/llama-4-maverick", Dialect::Xml),
		];
		for (model, expected) in cases {
			assert_eq!(preferred_dialect(model), expected, "{model}");
		}
	}

	#[test]
	fn selection_resolves_auto_native_and_omp_override() {
		assert_eq!(DialectSelection::Auto.resolve("zai/glm-5", None), Ok(Some(Dialect::Glm)));
		assert_eq!(DialectSelection::Native.resolve("zai/glm-5", None), Ok(None));
		assert_eq!(
			DialectSelection::Explicit(Dialect::Hermes).resolve("zai/glm-5", None),
			Ok(Some(Dialect::Hermes))
		);
		assert_eq!(
			DialectSelection::Auto.resolve("unknown/model", Some("hermes")),
			Ok(Some(Dialect::Hermes))
		);
		assert_eq!(DialectSelection::Auto.resolve("unknown/model", Some("native")), Ok(None));
		assert!(
			DialectSelection::Auto
				.resolve("unknown/model", Some("pi"))
				.is_err()
		);
	}

	#[test]
	fn reseller_reference_inherits_upstream_pricing_and_limits() {
		let mut upstream = card("anthropic", "claude-opus-4-8");
		upstream.context_window = 1_000_000;
		upstream.max_output_tokens = 128_000;
		upstream
			.pricing
			.push(Price { unit: PriceUnit::MtokInput, nanos_usd: 5_000_000_000 });
		upstream.behavior.supports_tools = Some(false);
		let mut reseller = card("proxy", "[Kiro] claude-opus-4-8-thinking");
		reseller.behavior.supports_tools = Some(true);
		let references = [upstream.clone(), reseller.clone()];
		let index = build_model_reference_index(&references);
		assert!(resolve_and_inherit_model_reference(&mut reseller, &index));
		assert_eq!(reseller.pricing, upstream.pricing);
		assert_eq!(reseller.context_window, 1_000_000);
		assert_eq!(reseller.max_output_tokens, 128_000);
		assert_eq!(reseller.provider, "proxy");
		assert_eq!(reseller.model, "[Kiro] claude-opus-4-8-thinking");
		assert_eq!(reseller.facets.as_slice(), &[Facet::Chat]);
		assert_eq!(reseller.behavior.supports_tools, Some(true));
	}

	#[test]
	fn effort_variants_collapse_with_wire_routing() {
		let low = card("devin", "gpt-5.6-luna-low");
		let medium = card("devin", "gpt-5.6-luna-medium");
		let high = card("devin", "gpt-5.6-luna-high");
		let collapsed = collapse_effort_variants_across_providers(vec![low, medium, high]);
		assert_eq!(collapsed.len(), 1);
		let model = &collapsed[0];
		assert_eq!(model.model, "gpt-5.6-luna");
		assert_eq!(model.efforts.as_slice(), &[Effort::Low, Effort::Medium, Effort::High]);
		assert_eq!(model.effort_routing[&Effort::Low], "gpt-5.6-luna-low");
		assert_eq!(model.effort_routing[&Effort::Medium], "gpt-5.6-luna-medium");
		assert_eq!(model.effort_routing[&Effort::High], "gpt-5.6-luna-high");
	}
}
