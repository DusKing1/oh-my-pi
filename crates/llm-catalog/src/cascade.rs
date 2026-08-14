//! KDL compat cascade: class/provider/model wire- and thinking-policy rules.
//!
//! Authoring format for the sparse per-model compatibility data that today
//! lives as flat enumerated profiles in
//! `fixtures/llm-oracle/catalog-policy/{compat,thinking}-profiles.json`.
//! Rules are conjunctions over three selector dimensions:
//!
//! - **class** — the centrally classified model class ([`crate::classify`]),
//! - **providers** — deployment hosts (`on "a" "b"` inside a class block, or
//!   the enclosing `provider` block),
//! - **models** — exact or `*`-glob, ASCII-case-insensitive, matched against
//!   the provider-relative model identifier.
//!
//! Axis ownership is semantic, not statistical: `classes/*.kdl` carry
//! model-lineage truths (the census keys them on model-class predicates —
//! dialect thinking markup, reasoning-content replay needs, reasoning control
//! ladders), while `providers/*.kdl` carry deployment wire contracts (role
//! and store support, token-field spelling, effort pass-through) plus
//! per-model residues the class stratum does not explain. Absence is never
//! inferred as "stripping": a rule only states what the census established,
//! scoped with `on` when a behavior is a class×host composition.
//!
//! ```kdl
//! class "deepseek" {
//!     models "deepseek-r1" "deepseek/deepseek-v3.2-exp" {
//!         requires-reasoning-content-for-all-assistant-turns #true
//!     }
//! }
//! provider "cursor" {
//!     models "gpt-5.1" { thinking-efforts "low" "high" }
//! }
//! ```
//!
//! Precedence is specificity-only: per axis, the matching rule with the
//! highest `(model-selector exactness, selector dimension count, priority)`
//! wins; two rules tying on all three while contesting one axis are rejected
//! at resolve time — declaration and file order are never semantic. Unknown
//! directives are rejected (`deny_unknown_fields` semantics). Thinking axes
//! describe the reasoning control surface and only apply to models the
//! catalog marks reasoning-capable; callers gate on that capability.
//! `tests/compat_cascade.rs` proves the bundled sources resolve to exactly
//! the frozen oracle for every catalog model.

use std::collections::BTreeMap;

use kdl::{KdlDocument, KdlNode, KdlValue};
use omp_core::{IntoStr, Str};
use serde_json::{Map, Value};
use thiserror::Error;

/// Value shape a directive accepts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AxisKind {
	/// Exactly one scalar argument.
	Scalar,
	/// One or more scalar arguments, resolved as a JSON array.
	Array,
	/// A children block of verbatim wire-JSON keys (possibly empty).
	Object,
}

/// Axis namespace: request-wire compatibility or thinking-control surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AxisSet {
	/// `wire/*` request compatibility overrides.
	Wire,
	/// Thinking/reasoning control-surface profile.
	Thinking,
}

/// Closed directive vocabulary: kebab directive, namespace, resolved key,
/// and accepted shape. Resolved keys use the frozen-oracle spellings.
///
/// Kept in lock-step with the oracle; `tests/compat_cascade.rs` fails when
/// either side drifts.
pub const KNOWN_AXES: &[(&str, AxisSet, &str, AxisKind)] = &[
	(
		"allows-synthetic-reasoning-content-for-tool-calls",
		AxisSet::Wire,
		"allows_synthetic_reasoning_content_for_tool_calls",
		AxisKind::Scalar,
	),
	("disable-adaptive-thinking", AxisSet::Wire, "disable_adaptive_thinking", AxisKind::Scalar),
	(
		"disable-reasoning-on-tool-choice",
		AxisSet::Wire,
		"disable_reasoning_on_tool_choice",
		AxisKind::Scalar,
	),
	("escape-builtin-tool-names", AxisSet::Wire, "escape_builtin_tool_names", AxisKind::Scalar),
	("extra-body", AxisSet::Wire, "extra_body", AxisKind::Object),
	("filter-reasoning-history", AxisSet::Wire, "filter_reasoning_history", AxisKind::Scalar),
	("include-encrypted-reasoning", AxisSet::Wire, "include_encrypted_reasoning", AxisKind::Scalar),
	("max-tokens-field", AxisSet::Wire, "max_tokens_field", AxisKind::Scalar),
	("official-endpoint", AxisSet::Wire, "official_endpoint", AxisKind::Scalar),
	("omit-reasoning-effort", AxisSet::Wire, "omit_reasoning_effort", AxisKind::Scalar),
	("reasoning-content-field", AxisSet::Wire, "reasoning_content_field", AxisKind::Scalar),
	("reasoning-disable-mode", AxisSet::Wire, "reasoning_disable_mode", AxisKind::Scalar),
	("reasoning-effort-map", AxisSet::Wire, "reasoning_effort_map", AxisKind::Object),
	("replay-unsigned-thinking", AxisSet::Wire, "replay_unsigned_thinking", AxisKind::Scalar),
	(
		"requires-assistant-content-for-tool-calls",
		AxisSet::Wire,
		"requires_assistant_content_for_tool_calls",
		AxisKind::Scalar,
	),
	(
		"requires-reasoning-content-for-all-assistant-turns",
		AxisSet::Wire,
		"requires_reasoning_content_for_all_assistant_turns",
		AxisKind::Scalar,
	),
	(
		"requires-reasoning-content-for-tool-calls",
		AxisSet::Wire,
		"requires_reasoning_content_for_tool_calls",
		AxisKind::Scalar,
	),
	("requires-thinking-enabled", AxisSet::Wire, "requires_thinking_enabled", AxisKind::Scalar),
	("requires-tool-result-id", AxisSet::Wire, "requires_tool_result_id", AxisKind::Scalar),
	("signing-endpoint", AxisSet::Wire, "signing_endpoint", AxisKind::Scalar),
	("stream-idle-timeout-ms", AxisSet::Wire, "stream_idle_timeout_ms", AxisKind::Scalar),
	("supports-developer-role", AxisSet::Wire, "supports_developer_role", AxisKind::Scalar),
	(
		"supports-eager-tool-input-streaming",
		AxisSet::Wire,
		"supports_eager_tool_input_streaming",
		AxisKind::Scalar,
	),
	("supports-forced-tool-choice", AxisSet::Wire, "supports_forced_tool_choice", AxisKind::Scalar),
	(
		"supports-image-detail-original",
		AxisSet::Wire,
		"supports_image_detail_original",
		AxisKind::Scalar,
	),
	(
		"supports-long-cache-retention",
		AxisSet::Wire,
		"supports_long_cache_retention",
		AxisKind::Scalar,
	),
	(
		"supports-mid-conversation-system",
		AxisSet::Wire,
		"supports_mid_conversation_system",
		AxisKind::Scalar,
	),
	("supports-reasoning-effort", AxisSet::Wire, "supports_reasoning_effort", AxisKind::Scalar),
	("supports-sampling-params", AxisSet::Wire, "supports_sampling_params", AxisKind::Scalar),
	("supports-store", AxisSet::Wire, "supports_store", AxisKind::Scalar),
	("supports-tool-choice", AxisSet::Wire, "supports_tool_choice", AxisKind::Scalar),
	("supports-usage-in-streaming", AxisSet::Wire, "supports_usage_in_streaming", AxisKind::Scalar),
	("thinking-format", AxisSet::Wire, "thinking_format", AxisKind::Scalar),
	("when-thinking", AxisSet::Wire, "when_thinking", AxisKind::Object),
	("thinking-default-level", AxisSet::Thinking, "defaultLevel", AxisKind::Scalar),
	("thinking-effort-budgets", AxisSet::Thinking, "effortBudgets", AxisKind::Object),
	("thinking-efforts", AxisSet::Thinking, "efforts", AxisKind::Array),
	("thinking-mode", AxisSet::Thinking, "mode", AxisKind::Scalar),
	("thinking-requires-effort", AxisSet::Thinking, "requiresEffort", AxisKind::Scalar),
	("thinking-suppress-when-off", AxisSet::Thinking, "suppressWhenOff", AxisKind::Scalar),
	("thinking-supports-display", AxisSet::Thinking, "supportsDisplay", AxisKind::Scalar),
];

macro_rules! sources {
	($($name:literal),+ $(,)?) => {
		&[$(($name, include_str!(concat!("../compat/", $name, ".kdl")))),+]
	};
}

/// Checked-in cascade sources: `classes/*` then `providers/*`.
///
/// `tests/compat_cascade.rs` asserts this list matches the on-disk `compat/`
/// tree so a new file cannot be silently dropped.
pub const BUNDLED_COMPAT: &[(&str, &str)] = sources![
	"classes/amazon",
	"classes/anthropic",
	"classes/baidu",
	"classes/bytedance",
	"classes/cohere",
	"classes/deepseek",
	"classes/gemini",
	"classes/gemma",
	"classes/glm",
	"classes/gpt-oss",
	"classes/kimi",
	"classes/meta",
	"classes/mimo",
	"classes/minimax",
	"classes/mistral",
	"classes/openai",
	"classes/qwen",
	"classes/stepfun",
	"classes/xai",
	"providers/agnes",
	"providers/agnes-plan",
	"providers/aiand",
	"providers/aimlapi",
	"providers/alibaba-coding-plan",
	"providers/alibaba-token-plan",
	"providers/amazon-bedrock",
	"providers/anthropic",
	"providers/azure",
	"providers/baseten",
	"providers/bedrock-mantle",
	"providers/cerebras",
	"providers/cloudflare-ai-gateway",
	"providers/cohere",
	"providers/coreweave",
	"providers/crofai",
	"providers/cursor",
	"providers/deepseek",
	"providers/firepass",
	"providers/fireworks",
	"providers/friendli",
	"providers/github-copilot",
	"providers/gitlab-duo",
	"providers/google",
	"providers/google-antigravity",
	"providers/google-gemini-cli",
	"providers/google-vertex",
	"providers/groq",
	"providers/huggingface",
	"providers/inception",
	"providers/kilo",
	"providers/kimi-code",
	"providers/meta",
	"providers/minimax",
	"providers/minimax-cn",
	"providers/minimax-code",
	"providers/minimax-code-cn",
	"providers/mistral",
	"providers/moonshot",
	"providers/nanogpt",
	"providers/novita",
	"providers/nvidia",
	"providers/ollama-cloud",
	"providers/openai",
	"providers/opencode-go",
	"providers/opencode-zen",
	"providers/openrouter",
	"providers/poolside",
	"providers/sakana",
	"providers/sarvam",
	"providers/scaleway",
	"providers/stepfun",
	"providers/stepfun-plan",
	"providers/synthetic",
	"providers/together",
	"providers/umans",
	"providers/venice",
	"providers/vercel-ai-gateway",
	"providers/wafer-serverless",
	"providers/xai-oauth",
	"providers/xiaomi",
	"providers/yandex",
	"providers/zai",
	"providers/zenmux",
	"providers/zhipu-coding-plan",
];

/// Resolved sparse axis assignments keyed by oracle-spelling axis name.
pub type AxisMap = BTreeMap<Str, Value>;

/// Wire and thinking assignments resolved for one model.
///
/// `thinking` describes the reasoning control surface and is only meaningful
/// for models the catalog marks reasoning-capable; callers apply that gate.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Resolved {
	/// `wire/*` request-compatibility overrides.
	pub wire:     AxisMap,
	/// Thinking-profile assignments (`mode`, `efforts`, …).
	pub thinking: AxisMap,
}

/// Cascade authoring or resolution failure.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CascadeError {
	/// A source file is not valid KDL.
	#[error("{file}: KDL parse failure: {message}")]
	Parse {
		/// Offending source file.
		file:    Str,
		/// Rendered parser diagnostic.
		message: Str,
	},
	/// A node appeared somewhere its kind is not allowed.
	#[error("{file}: unexpected node `{node}` under `{context}`")]
	UnexpectedNode {
		/// Offending source file.
		file:    Str,
		/// Node name found.
		node:    Str,
		/// Enclosing context.
		context: Str,
	},
	/// A directive is not in [`KNOWN_AXES`].
	#[error("{file}: unknown directive `{directive}`")]
	UnknownDirective {
		/// Offending source file.
		file:      Str,
		/// Kebab-case directive as written.
		directive: Str,
	},
	/// A directive has an argument shape its [`AxisKind`] rejects.
	#[error("{file}: directive `{directive}` has a malformed value")]
	MalformedDirective {
		/// Offending source file.
		file:      Str,
		/// Kebab-case directive as written.
		directive: Str,
	},
	/// The same axis was assigned twice within one rule block.
	#[error("{file}: axis `{axis}` assigned twice in one block")]
	DuplicateAxis {
		/// Offending source file.
		file: Str,
		/// Resolved axis name.
		axis: Str,
	},
	/// Two rules of equal specificity and priority set the same axis for one
	/// model. Declaration order is never a tiebreak; add `priority=N`.
	#[error(
		"ambiguous overlap for `{}/{}` on axis `{}`: rules `{}` and `{}` tie; add an explicit \
		 priority",
		.0.provider, .0.model, .0.axis, .0.first, .0.second
	)]
	AmbiguousOverlap(Box<OverlapDetails>),
}

/// Colliding-rule evidence for [`CascadeError::AmbiguousOverlap`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OverlapDetails {
	/// Provider whose rules collide.
	pub provider: Str,
	/// Provider-relative model identifier.
	pub model:    Str,
	/// Contested axis name.
	pub axis:     Str,
	/// First tied rule label.
	pub first:    Str,
	/// Second tied rule label.
	pub second:   Str,
}

/// One exact or glob model selector.
#[derive(Clone, Debug, Eq, PartialEq)]
enum Selector {
	/// Whole-identifier match (case-sensitive: identifiers are exact).
	Exact(Str),
	/// `*`-wildcard match (ASCII-case-insensitive: patterns span the chaotic
	/// aggregator spellings of one lineage).
	Glob(Str),
}

impl Selector {
	fn new(pattern: &str) -> Self {
		if pattern.contains('*') {
			Self::Glob(pattern.to_ascii_lowercase().to_str())
		} else {
			Self::Exact(pattern.to_str())
		}
	}

	fn matches(&self, model: &str, model_lower: &str) -> bool {
		match self {
			Self::Exact(id) => id.as_str() == model,
			Self::Glob(pattern) => glob_match(pattern.as_str(), model_lower),
		}
	}

	/// Exact selectors outrank globs; both outrank selector-free rules.
	const fn exactness(&self) -> u8 {
		match self {
			Self::Exact(_) => 2,
			Self::Glob(_) => 1,
		}
	}
}

/// Anchored `*`-wildcard match; both sides pre-lowercased.
fn glob_match(pattern: &str, value: &str) -> bool {
	let mut remainder = value;
	let mut segments = pattern.split('*');
	let Some(head) = segments.next() else {
		return value.is_empty();
	};
	let Some(stripped) = remainder.strip_prefix(head) else {
		return false;
	};
	remainder = stripped;
	let mut tail: Option<&str> = None;
	for segment in segments {
		if let Some(previous) = tail.take() {
			let Some(found) = remainder.find(previous) else {
				return false;
			};
			remainder = &remainder[found + previous.len()..];
		}
		tail = Some(segment);
	}
	match tail {
		// No `*` at all: the prefix strip must have consumed everything.
		None => remainder.is_empty(),
		Some("") => true,
		Some(last) => remainder.ends_with(last),
	}
}

/// One conjunction rule: every present dimension must match.
#[derive(Clone, Debug)]
struct Rule {
	class:     Option<Str>,
	providers: Option<Vec<Str>>,
	models:    Option<Vec<Selector>>,
	priority:  i64,
	wire:      AxisMap,
	thinking:  AxisMap,
	/// Human-readable origin for diagnostics.
	label:     Str,
}

impl Rule {
	/// Number of constrained selector dimensions.
	fn dimensions(&self) -> u8 {
		u8::from(self.class.is_some())
			+ u8::from(self.providers.is_some())
			+ u8::from(self.models.is_some())
	}

	/// `(exactness, dimensions, priority)` rank when the rule matches.
	fn rank(
		&self,
		provider: &str,
		class: &str,
		model: &str,
		model_lower: &str,
	) -> Option<(u8, u8, i64)> {
		if let Some(required) = &self.class
			&& required.as_str() != class
		{
			return None;
		}
		if let Some(providers) = &self.providers
			&& !providers
				.iter()
				.any(|candidate| candidate.as_str() == provider)
		{
			return None;
		}
		let exactness = match &self.models {
			None => 0,
			Some(selectors) => selectors
				.iter()
				.filter(|s| s.matches(model, model_lower))
				.map(Selector::exactness)
				.max()?,
		};
		Some((exactness, self.dimensions(), self.priority))
	}
}

/// Parsed, validated compat cascade over every source file.
#[derive(Clone, Debug, Default)]
pub struct CompatCascade {
	rules: Vec<Rule>,
}

impl CompatCascade {
	/// Parses and validates `(file name, KDL text)` sources.
	///
	/// # Errors
	/// Returns the first [`CascadeError`] encountered: invalid KDL, unknown or
	/// malformed directives, duplicate axes, or misplaced nodes.
	pub fn parse(sources: &[(&str, &str)]) -> Result<Self, CascadeError> {
		let mut rules = Vec::new();
		for &(file, text) in sources {
			let document: KdlDocument =
				text
					.parse()
					.map_err(|error: kdl::KdlError| CascadeError::Parse {
						file:    file.to_str(),
						message: error.to_string().to_str(),
					})?;
			for node in document.nodes() {
				match node.name().value() {
					"class" => parse_class(file, node, &mut rules)?,
					"provider" => parse_provider(file, node, &mut rules)?,
					other => {
						return Err(CascadeError::UnexpectedNode {
							file:    file.to_str(),
							node:    other.to_str(),
							context: "document root".to_str(),
						});
					},
				}
			}
		}
		Ok(Self { rules })
	}

	/// Parses the checked-in [`BUNDLED_COMPAT`] sources.
	///
	/// # Errors
	/// Propagates [`CascadeError`] from [`CompatCascade::parse`]; the bundled
	/// sources failing is a build defect.
	pub fn bundled() -> Result<Self, CascadeError> {
		Self::parse(BUNDLED_COMPAT)
	}

	/// Resolves wire and thinking assignments for one model.
	///
	/// `model` is the provider-relative identifier; `class` is the centrally
	/// classified class (`unknown` when unclassified); `reasoning` is the
	/// catalog's thinking capability for this model. When `false`, thinking
	/// axes are never evaluated, so class and `on` thinking rules cannot
	/// leak onto non-reasoning siblings. Unmatched models resolve to empty
	/// maps.
	///
	/// # Errors
	/// [`CascadeError::AmbiguousOverlap`] when two rules tying on
	/// `(exactness, dimensions, priority)` contest one axis.
	pub fn resolve(
		&self,
		provider: &str,
		class: &str,
		model: &str,
		reasoning: bool,
	) -> Result<Resolved, CascadeError> {
		let model_lower = model.to_ascii_lowercase();
		let mut wire: BTreeMap<&Str, ((u8, u8, i64), &Rule)> = BTreeMap::new();
		let mut thinking: BTreeMap<&Str, ((u8, u8, i64), &Rule)> = BTreeMap::new();
		for rule in &self.rules {
			if !reasoning && rule.wire.is_empty() {
				continue;
			}
			let Some(rank) = rule.rank(provider, class, model, &model_lower) else {
				continue;
			};
			contest(&mut wire, &rule.wire, rank, rule, provider, model)?;
			if reasoning {
				contest(&mut thinking, &rule.thinking, rank, rule, provider, model)?;
			}
		}
		let collect = |winners: BTreeMap<&Str, ((u8, u8, i64), &Rule)>,
		               pick: fn(&Rule) -> &AxisMap| {
			winners
				.into_iter()
				.map(|(axis, (_, rule))| (axis.clone(), pick(rule)[axis].clone()))
				.collect()
		};
		Ok(Resolved {
			wire:     collect(wire, |rule| &rule.wire),
			thinking: collect(thinking, |rule| &rule.thinking),
		})
	}
}

/// Ranks `rule` into the per-axis winner table; equal ranks are ambiguous.
fn contest<'cascade>(
	winners: &mut BTreeMap<&'cascade Str, ((u8, u8, i64), &'cascade Rule)>,
	axes: &'cascade AxisMap,
	rank: (u8, u8, i64),
	rule: &'cascade Rule,
	provider: &str,
	model: &str,
) -> Result<(), CascadeError> {
	for axis in axes.keys() {
		match winners.get(axis) {
			Some(&(held_rank, held)) if held_rank == rank => {
				return Err(CascadeError::AmbiguousOverlap(Box::new(OverlapDetails {
					provider: provider.to_str(),
					model:    model.to_str(),
					axis:     axis.clone(),
					first:    held.label.clone(),
					second:   rule.label.clone(),
				})));
			},
			Some(&(held_rank, _)) if held_rank > rank => {},
			_ => {
				winners.insert(axis, (rank, rule));
			},
		}
	}
	Ok(())
}

fn parse_class(file: &str, node: &KdlNode, rules: &mut Vec<Rule>) -> Result<(), CascadeError> {
	let name = single_string_argument(node).ok_or_else(|| CascadeError::MalformedDirective {
		file:      file.to_str(),
		directive: "class".to_str(),
	})?;
	let class = name.to_str();
	let mut direct = RuleAxes::default();
	if let Some(children) = node.children() {
		for child in children.nodes() {
			match child.name().value() {
				"on" => {
					let providers = string_arguments(child, file, "on")?;
					let (axes, models) = parse_rule_body(file, child, true)?;
					push_rule(rules, Some(class.clone()), Some(providers), models, child, axes, file);
				},
				"models" => {
					let selectors = selector_arguments(child, file)?;
					let (axes, nested) = parse_rule_body(file, child, false)?;
					debug_assert!(nested.is_none());
					push_rule(rules, Some(class.clone()), None, Some(selectors), child, axes, file);
				},
				_ => direct.collect(file, child)?,
			}
		}
	}
	if !direct.is_empty() {
		rules.push(Rule {
			class:     Some(class.clone()),
			providers: None,
			models:    None,
			priority:  node_priority(node),
			wire:      direct.wire,
			thinking:  direct.thinking,
			label:     fmt_label(file, &["class", class.as_str()]),
		});
	}
	Ok(())
}

fn parse_provider(file: &str, node: &KdlNode, rules: &mut Vec<Rule>) -> Result<(), CascadeError> {
	let name = single_string_argument(node).ok_or_else(|| CascadeError::MalformedDirective {
		file:      file.to_str(),
		directive: "provider".to_str(),
	})?;
	let provider = name.to_str();
	let mut direct = RuleAxes::default();
	if let Some(children) = node.children() {
		for child in children.nodes() {
			match child.name().value() {
				"class" => {
					let class = single_string_argument(child).ok_or_else(|| {
						CascadeError::MalformedDirective {
							file:      file.to_str(),
							directive: "class".to_str(),
						}
					})?;
					let (axes, nested) = parse_rule_body(file, child, false)?;
					debug_assert!(nested.is_none());
					push_rule(
						rules,
						Some(class.to_str()),
						Some(vec![provider.clone()]),
						None,
						child,
						axes,
						file,
					);
				},
				"models" => {
					let selectors = selector_arguments(child, file)?;
					let (axes, nested) = parse_rule_body(file, child, false)?;
					debug_assert!(nested.is_none());
					push_rule(
						rules,
						None,
						Some(vec![provider.clone()]),
						Some(selectors),
						child,
						axes,
						file,
					);
				},
				_ => direct.collect(file, child)?,
			}
		}
	}
	if !direct.is_empty() {
		rules.push(Rule {
			class:     None,
			providers: Some(vec![provider.clone()]),
			models:    None,
			priority:  node_priority(node),
			wire:      direct.wire,
			thinking:  direct.thinking,
			label:     fmt_label(file, &["provider", provider.as_str()]),
		});
	}
	Ok(())
}

/// Directives (and, for `on` blocks, one optional nested `models` rule body).
fn parse_rule_body(
	file: &str,
	node: &KdlNode,
	allow_models: bool,
) -> Result<(RuleAxes, Option<Vec<Selector>>), CascadeError> {
	let mut axes = RuleAxes::default();
	let mut models = None;
	if let Some(children) = node.children() {
		for child in children.nodes() {
			if allow_models && child.name().value() == "models" && models.is_none() {
				let selectors = selector_arguments(child, file)?;
				let (nested, _) = parse_rule_body(file, child, false)?;
				// Nested `models` inside `on` narrows the same rule; merge.
				axes.merge(file, nested)?;
				models = Some(selectors);
				continue;
			}
			axes.collect(file, child)?;
		}
	}
	Ok((axes, models))
}

fn push_rule(
	rules: &mut Vec<Rule>,
	class: Option<Str>,
	providers: Option<Vec<Str>>,
	models: Option<Vec<Selector>>,
	node: &KdlNode,
	axes: RuleAxes,
	file: &str,
) {
	if axes.is_empty() {
		return;
	}
	let label = fmt_label(file, &[node.name().value()]);
	rules.push(Rule {
		class,
		providers,
		models,
		priority: node_priority(node),
		wire: axes.wire,
		thinking: axes.thinking,
		label,
	});
}

/// Wire and thinking assignments collected from one rule block.
#[derive(Default)]
struct RuleAxes {
	wire:     AxisMap,
	thinking: AxisMap,
}

impl RuleAxes {
	fn is_empty(&self) -> bool {
		self.wire.is_empty() && self.thinking.is_empty()
	}

	fn collect(&mut self, file: &str, node: &KdlNode) -> Result<(), CascadeError> {
		let written = node.name().value();
		let Some(&(_, set, key, kind)) = KNOWN_AXES
			.iter()
			.find(|(directive, ..)| *directive == written)
		else {
			return Err(CascadeError::UnknownDirective {
				file:      file.to_str(),
				directive: written.to_str(),
			});
		};
		let value = node_value(node, kind).ok_or_else(|| CascadeError::MalformedDirective {
			file:      file.to_str(),
			directive: written.to_str(),
		})?;
		let map = match set {
			AxisSet::Wire => &mut self.wire,
			AxisSet::Thinking => &mut self.thinking,
		};
		if map.insert(key.to_str(), value).is_some() {
			return Err(CascadeError::DuplicateAxis { file: file.to_str(), axis: key.to_str() });
		}
		Ok(())
	}

	fn merge(&mut self, file: &str, other: Self) -> Result<(), CascadeError> {
		for (map, incoming) in [(&mut self.wire, other.wire), (&mut self.thinking, other.thinking)] {
			for (axis, value) in incoming {
				if map.insert(axis.clone(), value).is_some() {
					return Err(CascadeError::DuplicateAxis { file: file.to_str(), axis });
				}
			}
		}
		Ok(())
	}
}

fn node_priority(node: &KdlNode) -> i64 {
	node
		.get("priority")
		.and_then(KdlValue::as_integer)
		.and_then(|value| i64::try_from(value).ok())
		.unwrap_or(0)
}

fn fmt_label(file: &str, parts: &[&str]) -> Str {
	let mut label = String::with_capacity(file.len() + 16);
	label.push_str(file);
	for part in parts {
		label.push(':');
		label.push_str(part);
	}
	label.to_str()
}

/// Converts one directive node into wire JSON per its [`AxisKind`].
fn node_value(node: &KdlNode, kind: AxisKind) -> Option<Value> {
	let arguments: Vec<&KdlValue> = node
		.entries()
		.iter()
		.filter(|e| e.name().is_none())
		.map(kdl::KdlEntry::value)
		.collect();
	match kind {
		AxisKind::Scalar => match (arguments.as_slice(), node.children()) {
			([value], None) => scalar_value(value),
			_ => None,
		},
		AxisKind::Array => {
			if arguments.is_empty() || node.children().is_some() {
				return None;
			}
			arguments
				.iter()
				.map(|value| scalar_value(value))
				.collect::<Option<Vec<_>>>()
				.map(Value::from)
		},
		AxisKind::Object => match (arguments.as_slice(), node.children()) {
			([], Some(children)) => object_value(children),
			_ => None,
		},
	}
}

/// Nested payload node → JSON: verbatim keys, scalars or deeper objects.
fn object_value(children: &KdlDocument) -> Option<Value> {
	let mut object = Map::new();
	for child in children.nodes() {
		let arguments: Vec<&KdlValue> = child
			.entries()
			.iter()
			.filter(|entry| entry.name().is_none())
			.map(kdl::KdlEntry::value)
			.collect();
		let value = match (arguments.as_slice(), child.children()) {
			([value], None) => scalar_value(value)?,
			([], Some(nested)) => object_value(nested)?,
			_ => return None,
		};
		object.insert(child.name().value().into(), value);
	}
	Some(Value::Object(object))
}

fn scalar_value(value: &KdlValue) -> Option<Value> {
	match value {
		KdlValue::Bool(flag) => Some(Value::Bool(*flag)),
		KdlValue::Integer(integer) => i64::try_from(*integer).ok().map(Value::from),
		KdlValue::Float(float) => Some(Value::from(*float)),
		KdlValue::String(text) => Some(Value::from(text.as_str())),
		KdlValue::Null => None,
	}
}

fn single_string_argument(node: &KdlNode) -> Option<&str> {
	let mut arguments = node.entries().iter().filter(|entry| entry.name().is_none());
	let first = arguments.next()?;
	if arguments.next().is_some() {
		return None;
	}
	first.value().as_string()
}

fn string_arguments(node: &KdlNode, file: &str, directive: &str) -> Result<Vec<Str>, CascadeError> {
	let values: Option<Vec<Str>> = node
		.entries()
		.iter()
		.filter(|entry| entry.name().is_none())
		.map(|entry| entry.value().as_string().map(|text| text.to_str()))
		.collect();
	match values {
		Some(values) if !values.is_empty() => Ok(values),
		_ => Err(CascadeError::MalformedDirective {
			file:      file.to_str(),
			directive: directive.to_str(),
		}),
	}
}

fn selector_arguments(node: &KdlNode, file: &str) -> Result<Vec<Selector>, CascadeError> {
	let patterns = string_arguments(node, file, "models")?;
	Ok(patterns
		.iter()
		.map(|pattern| Selector::new(pattern.as_str()))
		.collect())
}

#[cfg(test)]
mod tests {
	use super::*;

	fn parse_one(text: &str) -> Result<CompatCascade, CascadeError> {
		CompatCascade::parse(&[("test.kdl", text)])
	}

	#[test]
	fn class_provider_and_model_rules_rank_by_specificity() {
		let cascade = parse_one(
			r#"
			class "deepseek" {
				models "r1-*" { requires-reasoning-content-for-all-assistant-turns #true }
				on "vendor" { thinking-mode "effort" }
			}
			provider "vendor" {
				supports-store #false
				models "r1-pro" { supports-store #true }
			}
			"#,
		)
		.expect("valid cascade");
		let base = cascade
			.resolve("vendor", "deepseek", "r1-mini", true)
			.expect("resolves");
		assert_eq!(base.wire["supports_store"], Value::Bool(false));
		assert_eq!(
			base.wire["requires_reasoning_content_for_all_assistant_turns"],
			Value::Bool(true)
		);
		assert_eq!(base.thinking["mode"], Value::from("effort"));
		let pro = cascade
			.resolve("vendor", "deepseek", "r1-pro", true)
			.expect("resolves");
		assert_eq!(pro.wire["supports_store"], Value::Bool(true), "model rule beats provider");
		let foreign = cascade
			.resolve("vendor", "qwen", "r1-mini", true)
			.expect("resolves");
		assert!(
			!foreign
				.wire
				.contains_key("requires_reasoning_content_for_all_assistant_turns")
		);
		assert!(foreign.thinking.is_empty());
		let elsewhere = cascade
			.resolve("other", "deepseek", "r1-mini", true)
			.expect("resolves");
		assert!(elsewhere.thinking.is_empty(), "`on` scopes the composition");
		assert_eq!(
			elsewhere.wire["requires_reasoning_content_for_all_assistant_turns"],
			Value::Bool(true),
			"selector-free dimensions do not scope"
		);
		let gated = cascade
			.resolve("vendor", "deepseek", "r1-mini", false)
			.expect("resolves");
		assert!(gated.thinking.is_empty(), "reasoning=false suppresses thinking axes");
		assert_eq!(gated.wire, base.wire, "wire axes are unaffected by the gate");
	}

	#[test]
	fn equal_rank_overlap_on_one_axis_is_rejected() {
		let cascade = parse_one(
			r#"provider "acme" {
				models "foo-*" { thinking-format "zai" }
				models "*-bar" { thinking-format "qwen" }
			}"#,
		)
		.expect("valid cascade");
		let error = cascade
			.resolve("acme", "unknown", "foo-bar", true)
			.expect_err("ambiguous");
		assert!(matches!(
			&error,
			CascadeError::AmbiguousOverlap(details) if details.axis.as_str() == "thinking_format"
		));
		assert!(cascade.resolve("acme", "unknown", "foo-only", true).is_ok());
	}

	#[test]
	fn disjoint_axes_overlap_resolves_both_values() {
		let cascade = parse_one(
			r#"provider "acme" {
				models "foo-*" { thinking-format "zai" }
				models "*-bar" { supports-store #false }
			}"#,
		)
		.expect("valid cascade");
		let resolved = cascade
			.resolve("acme", "unknown", "foo-bar", true)
			.expect("resolves");
		assert_eq!(resolved.wire["thinking_format"], Value::from("zai"));
		assert_eq!(resolved.wire["supports_store"], Value::Bool(false));
	}

	#[test]
	fn explicit_priority_breaks_ties_and_exact_beats_glob() {
		let cascade = parse_one(
			r#"provider "acme" {
				models "foo-*" priority=10 { thinking-format "zai" }
				models "*-bar" { thinking-format "qwen" }
				models "foo-exact" { thinking-format "kimi" }
			}"#,
		)
		.expect("valid cascade");
		let tied = cascade
			.resolve("acme", "unknown", "foo-bar", true)
			.expect("priority wins");
		assert_eq!(tied.wire["thinking_format"], Value::from("zai"));
		let exact = cascade
			.resolve("acme", "unknown", "foo-exact", true)
			.expect("resolves");
		assert_eq!(exact.wire["thinking_format"], Value::from("kimi"), "exact beats glob");
	}

	#[test]
	fn unknown_and_malformed_directives_are_rejected() {
		assert!(matches!(
			&parse_one(r#"provider "acme" { thinkign-format "zai" }"#),
			Err(CascadeError::UnknownDirective { directive, .. })
				if directive.as_str() == "thinkign-format"
		));
		assert!(matches!(
			parse_one(r#"provider "acme" { thinking-format "zai" "extra" }"#),
			Err(CascadeError::MalformedDirective { .. })
		));
		assert!(matches!(
			&parse_one(r#"provider "acme" { thinking-format "a"
				thinking-format "b" }"#),
			Err(CascadeError::DuplicateAxis { axis, .. }) if axis.as_str() == "thinking_format"
		));
	}

	#[test]
	fn nested_payloads_arrays_and_empty_maps_convert_to_wire_json() {
		let cascade = parse_one(
			r#"provider "acme" {
				extra-body { thinking { type "enabled" } }
				reasoning-effort-map {}
				stream-idle-timeout-ms 0
				thinking-efforts "low" "high" "max"
			}"#,
		)
		.expect("valid cascade");
		let resolved = cascade
			.resolve("acme", "unknown", "any", true)
			.expect("resolves");
		assert_eq!(
			resolved.wire["extra_body"],
			serde_json::json!({ "thinking": { "type": "enabled" } })
		);
		assert_eq!(resolved.wire["reasoning_effort_map"], serde_json::json!({}));
		assert_eq!(resolved.wire["stream_idle_timeout_ms"], Value::from(0));
		assert_eq!(resolved.thinking["efforts"], serde_json::json!(["low", "high", "max"]));
	}

	#[test]
	fn exact_selectors_are_case_sensitive_and_globs_are_not() {
		let cascade = parse_one(
			r#"class "glm" {
				models "zai-org/GLM-4.7" "glm-5.*" { thinking-format "zai" }
			}"#,
		)
		.expect("valid cascade");
		for id in ["zai-org/GLM-4.7", "GLM-5.2", "glm-5.2-fast"] {
			let resolved = cascade
				.resolve("anyhost", "glm", id, true)
				.expect("resolves");
			assert_eq!(resolved.wire["thinking_format"], Value::from("zai"), "{id}");
		}
		let miss = cascade
			.resolve("anyhost", "glm", "zai-org/glm-4.7", true)
			.expect("resolves");
		assert!(
			!miss.wire.contains_key("thinking_format"),
			"exact ids are distinct identifiers across case"
		);
	}

	#[test]
	fn glob_matching_is_anchored() {
		assert!(glob_match("foo-*", "foo-bar"));
		assert!(glob_match("*-bar", "foo-bar"));
		assert!(glob_match("foo-*-bar", "foo-x-bar"));
		assert!(glob_match("*", "anything"));
		assert!(!glob_match("foo-*", "xfoo-bar"));
		assert!(!glob_match("*-bar", "foo-barx"));
		assert!(!glob_match("foo", "foo-bar"));
	}
}
