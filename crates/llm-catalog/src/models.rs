//! Client-facing model cards and the lazily loaded bundled catalog.
//!
//! Pi denormalizes inference and provider routing metadata into every model
//! row. Translation retains `api` and per-model `baseUrl` as server-only
//! [`ModelWire`] data, and retains inference behavior and sparse compatibility
//! metadata in server-only [`ModelBehavior`]. Both fields are omitted from
//! serialized [`ModelCard`] values. Provider authentication and unrecognized
//! agent configuration remain outside the accepted input schema.
//!
//! The Rust importer in `src/bin/import_catalog.rs` is the only writer of the
//! checked-in payload. Its closed schema makes newly added source fields fail
//! generation instead of being silently discarded.

use std::{
	collections::{BTreeMap, btree_map::Entry},
	str::FromStr,
	sync::OnceLock,
};

use omp_core::Str;
use omp_llm_types::{
	ApplyPatchShape, Effort, Props, ResolvedModelCapabilities, ResolvedModelHeaders,
	ResolvedModelPolicy, ResolvedReasoningMode, ResolvedThinkingMode, ResolvedThinkingPolicy,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use serde_json::{Number, Value};
use smallvec::SmallVec;

use super::provider::{Facet, TransportId};

static MODELS_JSON_ZST: &[u8] = include_bytes!("../models.json.zst");

/// An input or output media modality.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
	/// No modality was specified.
	Unspecified,
	/// Text.
	Text,
	/// Image data.
	Image,
	/// Audio data.
	Audio,
	/// Video data.
	Video,
	/// PDF documents.
	Pdf,
}

/// Whether a model can currently serve requests.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Availability {
	/// Availability has not yet been joined with credential state.
	Unspecified,
	/// The model is ready to serve.
	Available,
	/// A provider login is required.
	LoginRequired,
	/// Every usable credential is temporarily blocked.
	Blocked,
	/// The model is disabled by configuration.
	Disabled,
}

/// Where a model card originated.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
	/// The source is unknown.
	Unspecified,
	/// The card came from the shipped catalog.
	Bundled,
	/// The card came from live provider discovery.
	Discovered,
	/// The card came from user or project configuration.
	Configured,
}

/// The billing unit for a price component.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriceUnit {
	/// One million input tokens.
	MtokInput,
	/// One million output tokens.
	MtokOutput,
	/// One million prompt-cache read tokens.
	MtokCacheRead,
	/// One million prompt-cache write tokens.
	MtokCacheWrite,
	/// One generated image.
	Image,
	/// One generated video second.
	VideoSecond,
	/// One generated or transcribed audio second.
	AudioSecond,
	/// One million input characters.
	McharInput,
	/// One request.
	Request,
}

/// One model pricing component, represented in billionths of a US dollar.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, bon::Builder)]
#[non_exhaustive]
pub struct Price {
	/// The quantity to which this price applies.
	pub unit:      PriceUnit,
	/// The price per unit in nanos of a US dollar.
	pub nanos_usd: u64,
}
/// A higher token-price schedule selected by total prompt size.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, bon::Builder)]
#[non_exhaustive]
pub struct PriceTier {
	/// The tier applies to the full request above this many prompt tokens.
	pub prompt_tokens_above: u64,
	/// Per-unit prices active in this tier.
	pub pricing:             SmallVec<Price, 4>,
}

/// Server-only wire routing retained from the native Pi model row.
///
/// This metadata is deliberately omitted from serialized [`ModelCard`] values:
/// clients select logical models, while the gateway owns endpoint selection.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize)]
pub struct ModelWire {
	/// Concrete codec/transport used by this model.
	pub transport: TransportId,
	/// Per-model endpoint override. `None` inherits the provider endpoint.
	pub base_url:  Option<Str>,
}

/// Native reasoning effort spelling retained by the model catalog.
///
/// Every spelling has a distinct portable [`Effort`] value, so routing and
/// request projection never fold `XHigh` into a provider-defined `Max`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
pub enum ModelThinkingEffort {
	/// Explicitly disable reasoning.
	Off,
	/// Minimal reasoning.
	Minimal,
	/// Low reasoning.
	Low,
	/// Medium reasoning.
	Medium,
	/// High reasoning.
	High,
	/// `OpenAI`'s extra-high reasoning tier.
	XHigh,
	/// A provider's maximum reasoning tier.
	Max,
}

impl ModelThinkingEffort {
	pub(crate) const fn portable(self) -> Effort {
		match self {
			Self::Off => Effort::Off,
			Self::Minimal => Effort::Minimal,
			Self::Low => Effort::Low,
			Self::Medium => Effort::Medium,
			Self::High => Effort::High,
			Self::XHigh => Effort::XHigh,
			Self::Max => Effort::Max,
		}
	}
}

/// Provider-native control used to select a reasoning effort.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "kebab-case")]
pub enum ModelThinkingMode {
	/// Send a named effort.
	Effort,
	/// Send a token budget.
	Budget,
	/// Send a Google thinking level.
	GoogleLevel,
	/// Use Anthropic adaptive thinking.
	AnthropicAdaptive,
	/// Use Anthropic budget thinking plus an effort.
	AnthropicBudgetEffort,
}

/// Server-only reasoning capability and routing metadata.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelThinking {
	/// Provider-native thinking control.
	pub mode:              ModelThinkingMode,
	/// Exact native effort spellings, ordered least to most intensive.
	pub efforts:           SmallVec<ModelThinkingEffort, 6>,
	/// Default effort selected by this model.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub default_level:     Option<ModelThinkingEffort>,
	/// Per-effort native value overrides.
	#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
	pub effort_map:        BTreeMap<ModelThinkingEffort, Str>,
	/// Per-effort wire model routing.
	#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
	pub effort_routing:    BTreeMap<ModelThinkingEffort, Str>,
	/// Per-effort thinking token budgets.
	#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
	pub effort_budgets:    BTreeMap<ModelThinkingEffort, u64>,
	/// Whether the native adaptive-thinking display control is supported.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub supports_display:  Option<bool>,
	/// Whether disabling reasoning must be explicit on the wire.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub suppress_when_off: Option<bool>,
	/// Whether the upstream requires a non-off reasoning effort.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub requires_effort:   Option<bool>,
}

/// `OpenAI` apply-patch tool encoding selected for a model.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
pub enum ApplyPatchToolType {
	/// Send the patch as an unwrapped custom-tool string.
	Freeform,
	/// Send the patch in a JSON function argument.
	Function,
}

/// `OpenAI` Responses reasoning serving mode.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
pub enum ModelReasoningMode {
	/// Use the provider's pro reasoning path.
	Pro,
}

/// An exact decimal premium-request multiplier at millionth precision.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub struct PremiumMultiplier {
	/// Multiplier scaled by 1,000,000 (`0.33` is `330_000`).
	pub millionths: u64,
}

impl PremiumMultiplier {
	const SCALE: u64 = 1_000_000;

	/// Constructs an exact multiplier from its millionth-scale integer.
	#[must_use]
	pub const fn from_millionths(millionths: u64) -> Self {
		Self { millionths }
	}

	fn from_number(number: &Number) -> Result<Self, &'static str> {
		let text = number.to_string();
		if text.starts_with('-') || text.contains('e') || text.contains('E') {
			return Err("premium multiplier must be a non-negative decimal without an exponent");
		}
		let (whole, fractional) = text.split_once('.').unwrap_or((text.as_str(), ""));
		if fractional.len() > 6 {
			return Err("premium multiplier supports at most six fractional digits");
		}
		let whole = whole
			.parse::<u64>()
			.map_err(|_| "premium multiplier is out of range")?;
		let fractional = if fractional.is_empty() {
			0
		} else {
			fractional
				.parse::<u64>()
				.map_err(|_| "premium multiplier has an invalid fractional part")?
				.checked_mul(10_u64.pow(6_u32.saturating_sub(fractional.len() as u32)))
				.ok_or("premium multiplier is out of range")?
		};
		let millionths = whole
			.checked_mul(Self::SCALE)
			.and_then(|value| value.checked_add(fractional))
			.ok_or("premium multiplier is out of range")?;
		Ok(Self { millionths })
	}

	fn as_number(self) -> Number {
		let whole = self.millionths / Self::SCALE;
		let fractional = self.millionths % Self::SCALE;
		if fractional == 0 {
			return Number::from(whole);
		}
		let mut text = format!("{whole}.{fractional:06}");
		while text.ends_with('0') {
			text.pop();
		}
		Number::from_str(&text).expect("fixed-point multiplier is valid JSON")
	}
}

impl Serialize for PremiumMultiplier {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: Serializer,
	{
		self.as_number().serialize(serializer)
	}
}

impl<'de> Deserialize<'de> for PremiumMultiplier {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		let number = Number::deserialize(deserializer)?;
		Self::from_number(&number).map_err(de::Error::custom)
	}
}

/// Provider-native compaction routing retained for a model.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct ModelRemoteCompaction {
	/// Explicit native enablement.
	pub enabled:              Option<bool>,
	/// Transport used by the native compaction endpoint.
	pub transport:            Option<TransportId>,
	/// Absolute V1 compaction endpoint override.
	pub endpoint:             Option<Str>,
	/// Enables Responses-stream V2 compaction.
	pub v2_streaming_enabled: Option<bool>,
	/// Absolute V2 endpoint override.
	pub v2_endpoint:          Option<Str>,
	/// Absolute streaming endpoint used by V2 compaction.
	pub streaming_endpoint:   Option<Str>,
	/// Model id sent to the compaction endpoint.
	pub model:                Option<Str>,
}

/// Inference-relevant metadata retained exclusively on the server.
///
/// The model resolver may convert this record into request policy, but it is
/// never serialized as part of [`ModelCard`] and therefore never crosses the
/// discovery/client security boundary. Sparse compatibility details use the
/// canonical `wire/*` namespace.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct ModelBehavior {
	/// Native reasoning configuration.
	pub thinking: Option<ModelThinking>,
	/// Explicit native tool-call support.
	pub supports_tools: Option<bool>,
	/// Effective computer-use support.
	pub supports_computer_use: Option<bool>,
	/// Explicit authored computer-use value, before inference.
	pub supports_computer_use_config: Option<bool>,
	/// Cursor premium max-mode flag.
	pub cursor_max_mode: Option<bool>,
	/// Suppress the native maximum-output-token field.
	pub omit_max_output_tokens: Option<bool>,
	/// `OpenAI` apply-patch encoding.
	pub apply_patch_tool_type: Option<ApplyPatchToolType>,
	/// Preferred logical model after context promotion.
	pub context_promotion_target: Option<Str>,
	/// Model id sent on the wire when it differs from the logical id.
	pub request_model_id: Option<Str>,
	/// Provider-native compaction routing.
	pub remote_compaction: Option<ModelRemoteCompaction>,
	/// Exact Copilot premium request multiplier.
	pub premium_multiplier: Option<PremiumMultiplier>,
	/// `OpenAI` Responses reasoning serving mode.
	pub reasoning_mode: Option<ModelReasoningMode>,
	/// Use the Codex Responses Lite request shape.
	pub use_responses_lite: Option<bool>,
	/// Prefer a websocket when the selected transport supports it.
	pub prefer_websockets: Option<bool>,
	/// Provider-assigned routing priority; lower values are preferred.
	pub priority: Option<u32>,
	/// Per-model request headers. Import retains these until provider metadata
	/// is available to prove an entry redundant.
	pub headers: BTreeMap<Str, Str>,
	/// Sparse canonical wire compatibility metadata.
	pub compat: Props,
}

impl ModelBehavior {
	/// Converts trusted catalog behavior into transport-neutral request policy.
	///
	/// Agent-only context promotion and remote compaction remain catalog-owned.
	/// Credential-bearing header names are filtered at this boundary; the
	/// returned value has no serialization implementation and is safe to attach
	/// only to native in-process requests.
	#[must_use]
	pub fn resolved_policy(&self) -> ResolvedModelPolicy {
		let thinking = self
			.thinking
			.as_ref()
			.map(|thinking| ResolvedThinkingPolicy {
				mode:              match thinking.mode {
					ModelThinkingMode::Effort => ResolvedThinkingMode::Effort,
					ModelThinkingMode::Budget => ResolvedThinkingMode::Budget,
					ModelThinkingMode::GoogleLevel => ResolvedThinkingMode::GoogleLevel,
					ModelThinkingMode::AnthropicAdaptive => ResolvedThinkingMode::AnthropicAdaptive,
					ModelThinkingMode::AnthropicBudgetEffort => {
						ResolvedThinkingMode::AnthropicBudgetEffort
					},
				},
				efforts:           thinking
					.efforts
					.iter()
					.map(|effort| effort.portable())
					.collect(),
				default_effort:    thinking.default_level.map(ModelThinkingEffort::portable),
				effort_map:        thinking
					.effort_map
					.iter()
					.map(|(effort, value)| (effort.portable(), value.clone()))
					.collect(),
				effort_routing:    thinking
					.effort_routing
					.iter()
					.map(|(effort, value)| (effort.portable(), value.clone()))
					.collect(),
				effort_budgets:    thinking
					.effort_budgets
					.iter()
					.map(|(effort, value)| (effort.portable(), *value))
					.collect(),
				supports_display:  thinking.supports_display,
				suppress_when_off: thinking.suppress_when_off,
				requires_effort:   thinking.requires_effort,
			});
		let headers = self
			.headers
			.iter()
			.filter(|(name, _)| safe_model_header(name))
			.map(|(name, value)| (name.clone(), value.clone()))
			.collect();
		ResolvedModelPolicy {
			request_model_id: self.request_model_id.clone(),
			thinking,
			capabilities: ResolvedModelCapabilities {
				tools:               self.supports_tools,
				computer_use:        self.supports_computer_use,
				computer_use_config: self.supports_computer_use_config,
			},
			cursor_max_mode: self.cursor_max_mode,
			omit_max_output_tokens: self.omit_max_output_tokens,
			apply_patch_shape: self.apply_patch_tool_type.map(|shape| match shape {
				ApplyPatchToolType::Freeform => ApplyPatchShape::Freeform,
				ApplyPatchToolType::Function => ApplyPatchShape::Function,
			}),
			premium_millionths: self.premium_multiplier.map(|value| value.millionths),
			reasoning_mode: self.reasoning_mode.map(|mode| match mode {
				ModelReasoningMode::Pro => ResolvedReasoningMode::Pro,
			}),
			use_responses_lite: self.use_responses_lite,
			prefer_websockets: self.prefer_websockets,
			headers: ResolvedModelHeaders(headers),
			compat: self.compat.clone(),
		}
	}
}

fn safe_model_header(name: &str) -> bool {
	![
		"authorization",
		"proxy-authorization",
		"cookie",
		"set-cookie",
		"x-api-key",
		"api-key",
		"x-goog-api-key",
		"host",
		"connection",
		"content-length",
		"transfer-encoding",
	]
	.iter()
	.any(|unsafe_name| name.eq_ignore_ascii_case(unsafe_name))
}

/// A client-facing description of one logical model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, bon::Builder)]
#[non_exhaustive]
pub struct ModelCard {
	/// Canonical `provider/model` selector.
	pub id:                Str,
	/// Provider identifier.
	pub provider:          Str,
	/// Provider-local logical model identifier.
	pub model:             Str,
	/// Display name.
	pub name:              Str,
	/// Coarse vendor-lineage token.
	pub family:            Str,
	/// Inference capabilities served by this model.
	pub facets:            SmallVec<Facet, 4>,
	/// Accepted input modalities.
	pub inputs:            SmallVec<Modality, 4>,
	/// Produced output modalities.
	pub outputs:           SmallVec<Modality, 4>,
	/// Whether the model performs reasoning.
	pub reasoning:         bool,
	/// Reasoning effort levels supported after variant collapse.
	pub efforts:           SmallVec<Effort, 6>,
	/// Maximum input context in tokens, or zero when unknown.
	pub context_window:    u64,
	/// Maximum generated tokens, or zero when unknown.
	pub max_output_tokens: u64,
	/// Pricing components.
	pub pricing:           SmallVec<Price, 4>,
	/// Higher price schedules selected by total prompt size.
	#[builder(default)]
	#[serde(default, skip_serializing_if = "SmallVec::is_empty")]
	pub pricing_tiers:      SmallVec<PriceTier, 1>,
	/// Current joined availability.
	pub availability:      Availability,
	/// Card origin.
	pub source:            Source,
	/// Earliest credential unblock time for blocked cards.
	pub blocked_until_ms:  u64,
	/// Whether the provider has deprecated this model.
	pub deprecated:        bool,
	/// Last registry update time.
	pub updated_at_ms:     u64,
	/// Namespaced vendor metadata suitable for clients.
	pub props:             Props,
	/// Gateway-internal effort-to-wire-model routing produced by variant
	/// collapse.
	#[serde(skip)]
	pub effort_routing:    BTreeMap<Effort, Str>,
	/// Server-only inference behavior retained from the native model row.
	#[builder(default)]
	#[serde(skip)]
	pub behavior:          ModelBehavior,
	/// Gateway-internal per-model transport and endpoint override.
	#[serde(skip)]
	pub wire:              Option<ModelWire>,
}

/// Token counters added by orchestration outside the provider turn.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, bon::Builder)]
#[non_exhaustive]
pub struct OrchestrationUsage {
	/// Additional input tokens.
	pub input:      u64,
	/// Additional output tokens.
	pub output:     u64,
	/// Additional cache-read tokens.
	pub cache_read: u64,
}

/// Provider-reported cache-write token buckets by retention TTL.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, bon::Builder)]
#[non_exhaustive]
pub struct CacheTtlUsage {
	/// Cache-write tokens retained for five minutes.
	pub ephemeral_5m: u64,
	/// Cache-write tokens retained for one hour.
	pub ephemeral_1h: u64,
}

/// Usage inputs needed for catalog cost calculation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, bon::Builder)]
#[non_exhaustive]
pub struct CostUsage {
	/// Provider input tokens.
	pub input:         u64,
	/// Provider output tokens.
	pub output:        u64,
	/// Provider cache-read tokens.
	pub cache_read:    u64,
	/// Provider cache-write tokens.
	pub cache_write:   u64,
	/// Optional agent-side token counts.
	pub orchestration: Option<OrchestrationUsage>,
	/// Optional provider cache TTL breakdown.
	pub cttl:          Option<CacheTtlUsage>,
}

/// Cost components in US dollars.
#[derive(Clone, Copy, Debug, Default, PartialEq, bon::Builder)]
#[non_exhaustive]
pub struct CostBreakdown {
	/// Input-token cost.
	pub input:       f64,
	/// Output-token cost.
	pub output:      f64,
	/// Cache-read cost.
	pub cache_read:  f64,
	/// Cache-write cost.
	pub cache_write: f64,
	/// Sum of all components.
	pub total:       f64,
}

/// Errors decoding a generated catalog.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CatalogError {
	/// The zstd payload could not be decompressed.
	#[error("could not decompress bundled model catalog: {0}")]
	Decompress(#[source] std::io::Error),
	/// A generated catalog could not be compressed.
	#[error("could not compress bundled model catalog: {0}")]
	Compress(#[source] std::io::Error),
	/// The decompressed JSON did not match the translated schema.
	#[error("could not parse bundled model catalog: {0}")]
	Json(#[from] serde_json::Error),
	/// One source model failed the closed importer schema.
	#[error("could not parse source model `{identity}`: {source}")]
	InvalidSourceModel {
		/// Source `provider/model` path.
		identity: String,
		/// Schema error at that model path.
		#[source]
		source:   serde_json::Error,
	},
	/// A nonempty source unexpectedly translated into no model cards.
	#[error("nonempty source catalog translated into an empty model catalog")]
	EmptyCatalog,
	/// A source provider or model key was empty after normalization.
	#[error("catalog contains an empty {kind} identity")]
	EmptyIdentity {
		/// The identity component that was empty.
		kind: &'static str,
	},
	/// Normalization made two source keys refer to the same identity.
	#[error("catalog normalization produced duplicate {kind} identity `{identity}`")]
	DuplicateIdentity {
		/// The identity component that collided.
		kind:     &'static str,
		/// The normalized identity.
		identity: String,
	},
	/// A model attempted to install a credential or framing-sensitive header.
	#[error("source model `{identity}` contains unsafe request header `{header}`")]
	UnsafeModelHeader {
		/// Source `provider/model` path.
		identity: String,
		/// Rejected header name.
		header:   String,
	},
}

/// An immutable translated model catalog with allocation-free keyed lookup.
#[derive(Clone, Debug, Default, bon::Builder)]
#[non_exhaustive]
pub struct ModelCatalog {
	models: Vec<ModelCard>,
	by_key: BTreeMap<Str, BTreeMap<Str, usize>>,
}

impl ModelCatalog {
	/// Constructs a catalog and its provider/model index.
	#[must_use]
	pub fn new(models: Vec<ModelCard>) -> Self {
		let mut by_key: BTreeMap<Str, BTreeMap<Str, usize>> = BTreeMap::new();
		for (index, model) in models.iter().enumerate() {
			by_key
				.entry(model.provider.clone())
				.or_default()
				.insert(model.model.clone(), index);
		}
		Self { models, by_key }
	}

	/// Returns all cards in stable catalog order.
	#[must_use]
	pub fn models(&self) -> &[ModelCard] {
		&self.models
	}

	/// Finds a card by provider and provider-local model id.
	#[must_use]
	pub fn get(&self, provider: &str, model: &str) -> Option<&ModelCard> {
		self
			.by_key
			.get(provider)?
			.get(model)
			.map(|&index| &self.models[index])
	}

	/// Returns whether the catalog has no cards.
	#[must_use]
	pub const fn is_empty(&self) -> bool {
		self.models.is_empty()
	}

	/// Returns the number of cards.
	#[must_use]
	pub const fn len(&self) -> usize {
		self.models.len()
	}
}

/// Computes the four token cost components using the TypeScript catalog rules.
#[must_use]
pub fn calculate_cost(model: &ModelCard, usage: &CostUsage) -> CostBreakdown {
	let orchestration = usage.orchestration.unwrap_or_default();
	let pricing = resolve_pricing(model, usage);
	let input =
		token_cost(pricing, PriceUnit::MtokInput, usage.input.saturating_add(orchestration.input));
	let output =
		token_cost(pricing, PriceUnit::MtokOutput, usage.output.saturating_add(orchestration.output));
	let cache_read = token_cost(
		pricing,
		PriceUnit::MtokCacheRead,
		usage.cache_read.saturating_add(orchestration.cache_read),
	);
	let cache_write = cache_write_cost_with_pricing(pricing, usage);
	CostBreakdown {
		input,
		output,
		cache_read,
		cache_write,
		total: input + output + cache_read + cache_write,
	}
}

/// Prices cache writes, preserving unattributed tokens and the one-hour rate.
#[must_use]
pub fn cache_write_cost(model: &ModelCard, usage: &CostUsage) -> f64 {
	cache_write_cost_with_pricing(resolve_pricing(model, usage), usage)
}

fn resolve_pricing<'a>(model: &'a ModelCard, usage: &CostUsage) -> &'a [Price] {
	let orchestration = usage.orchestration.unwrap_or_default();
	let prompt_tokens = usage
		.input
		.saturating_add(usage.cache_read)
		.saturating_add(usage.cache_write)
		.saturating_add(orchestration.input)
		.saturating_add(orchestration.cache_read);
	model
		.pricing_tiers
		.iter()
		.filter(|tier| prompt_tokens > tier.prompt_tokens_above)
		.max_by_key(|tier| tier.prompt_tokens_above)
		.map_or(model.pricing.as_slice(), |tier| tier.pricing.as_slice())
}

fn cache_write_cost_with_pricing(pricing: &[Price], usage: &CostUsage) -> f64 {
	let rate_5m = price_usd(pricing, PriceUnit::MtokCacheWrite) / 1_000_000.0;
	let Some(cttl) = usage.cttl else {
		return rate_5m * usage.cache_write as f64;
	};
	let residual = usage
		.cache_write
		.saturating_sub(cttl.ephemeral_5m.saturating_add(cttl.ephemeral_1h));
	let one_hour_rate = price_usd(pricing, PriceUnit::MtokInput) * 2.0 / 1_000_000.0;
	one_hour_rate.mul_add(
		cttl.ephemeral_1h as f64,
		rate_5m * cttl.ephemeral_5m.saturating_add(residual) as f64,
	)
}

/// Compares model identity by provider and provider-local id.
#[must_use]
pub fn models_are_equal(left: Option<&ModelCard>, right: Option<&ModelCard>) -> bool {
	matches!((left, right), (Some(left), Some(right)) if left.provider == right.provider && left.model == right.model)
}

/// Decodes and translates a zstd-compressed generated `models.json` payload.
pub fn load_catalog_zstd(bytes: &[u8]) -> Result<ModelCatalog, CatalogError> {
	let json = zstd::stream::decode_all(bytes).map_err(CatalogError::Decompress)?;
	load_catalog_json(&json)
}

/// Translates generated catalog JSON or the upstream Pi provider map into
/// normalized model cards.
pub fn load_catalog_json(bytes: &[u8]) -> Result<ModelCatalog, CatalogError> {
	let shape: Value = serde_json::from_slice(bytes)?;
	if shape
		.as_object()
		.is_some_and(|fields| fields.contains_key("models"))
	{
		let envelope: RawEnvelope = serde_json::from_slice(bytes)?;
		let mut intern = Interner::default();
		let models = envelope
			.models
			.into_iter()
			.map(|raw_model| {
				let outer_provider = raw_model.provider.clone().unwrap_or_default();
				let outer_model = raw_model.id.clone().unwrap_or_default();
				translate_model(&mut intern, &outer_provider, &outer_model, raw_model)
			})
			.collect();
		Ok(ModelCatalog::new(finalize_models(models)))
	} else {
		// Deserialize directly from serde_json's parser. An untagged enum
		// buffers arbitrary-precision numbers through serde's generic Content
		// representation, where their private number token no longer decodes as
		// the typed f64 fields in RawModel.
		let providers: BTreeMap<String, BTreeMap<String, RawModel>> = serde_json::from_slice(bytes)?;
		Ok(ModelCatalog::new(translate_provider_map(providers)))
	}
}

/// Converts a Pi catalog source snapshot into the deterministic checked-in
/// zstd payload consumed by [`embedded_catalog`].
///
/// Provider/model keys are trimmed. Typed `api`/`baseUrl` routing and inference
/// behavior are retained server-side, while authentication and unknown agent
/// fields are rejected by the closed input schema. Pi does not place `OpenAI`'s
/// embedding cards in its `openai` bucket, so the importer promotes
/// Pi's exact `ZenMux` copies (pricing and limits) to direct `OpenAI`
/// identities. Their native dimensions are `OpenAI`'s published 1,536/3,072
/// widths; OMP's curated `OpenAI` provider entry and remote embedding policy
/// are the capability sources for the direct route and custom dimensions. The
/// serialized provider map remains a source snapshot plus those two documented
/// promotions; loading it applies family detection, reseller-reference
/// inheritance, and effort-variant collapse. Zstd frames do not contain
/// filesystem timestamps.
pub fn import_catalog_zstd(bytes: &[u8]) -> Result<Vec<u8>, CatalogError> {
	let source_values: BTreeMap<String, BTreeMap<String, Value>> = serde_json::from_slice(bytes)?;
	for (provider, models) in &source_values {
		for (model, value) in models {
			let raw: RawModel = serde_json::from_value(value.clone()).map_err(|source| {
				CatalogError::InvalidSourceModel { identity: format!("{provider}/{model}"), source }
			})?;
			if let Some(header) = raw.headers.keys().find(|name| !safe_model_header(name)) {
				return Err(CatalogError::UnsafeModelHeader {
					identity: format!("{provider}/{model}"),
					header:   header.to_string(),
				});
			}
		}
	}
	let mut normalized = normalize_provider_values(source_values)?;
	promote_openai_embedding_values(&mut normalized);
	for models in normalized.values_mut() {
		for value in models.values_mut() {
			sort_value_keys(value);
		}
	}
	let json = serde_json::to_vec(&normalized)?;
	let catalog = load_catalog_json(&json)?;
	if !normalized.is_empty() && catalog.is_empty() {
		return Err(CatalogError::EmptyCatalog);
	}
	if catalog
		.models()
		.iter()
		.any(|model| model.id.is_empty() || model.provider.is_empty() || model.model.is_empty())
	{
		return Err(CatalogError::EmptyIdentity { kind: "normalized" });
	}
	zstd::bulk::compress(&json, 19).map_err(CatalogError::Compress)
}

/// Returns the process-wide bundled catalog, decompressing and parsing it only
/// on first access.
#[must_use]
pub fn embedded_catalog() -> &'static ModelCatalog {
	static CATALOG: OnceLock<ModelCatalog> = OnceLock::new();
	CATALOG.get_or_init(|| {
		load_catalog_zstd(MODELS_JSON_ZST).unwrap_or_else(|error| {
			tracing::warn!(%error, "bundled model catalog is unavailable");
			ModelCatalog::default()
		})
	})
}

fn token_cost(pricing: &[Price], unit: PriceUnit, tokens: u64) -> f64 {
	price_usd(pricing, unit) / 1_000_000.0 * tokens as f64
}

fn price_usd(pricing: &[Price], unit: PriceUnit) -> f64 {
	pricing
		.iter()
		.find(|price| price.unit == unit)
		.map_or(0.0, |price| price.nanos_usd as f64 / 1_000_000_000.0)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEnvelope {
	models: Vec<RawModel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawModel {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	id: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	name: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	provider: Option<String>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	api: Option<RawApi>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	base_url: Option<Str>,
	#[serde(default)]
	reasoning: bool,
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	input: Vec<RawModality>,
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	output: Vec<RawModality>,
	#[serde(default)]
	cost: RawCost,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	context_window: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	max_tokens: Option<u64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	thinking: Option<ModelThinking>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	embedding_dimensions: Option<u32>,
	#[serde(default, skip_serializing_if = "<&bool as std::ops::Not>::not")]
	deprecated: bool,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	supports_tools: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	supports_computer_use: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	supports_computer_use_config: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	cursor_max_mode: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	omit_max_output_tokens: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	apply_patch_tool_type: Option<ApplyPatchToolType>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	context_promotion_target: Option<Str>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	request_model_id: Option<Str>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	remote_compaction: Option<RawRemoteCompaction>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	premium_multiplier: Option<PremiumMultiplier>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	reasoning_mode: Option<ModelReasoningMode>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	use_responses_lite: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	prefer_websockets: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	priority: Option<u32>,
	#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
	headers: BTreeMap<Str, Str>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	compat: Option<RawCompat>,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum RawApi {
	#[serde(rename = "anthropic-messages")]
	AnthropicMessages,
	#[serde(rename = "apple-intelligence-api")]
	AppleIntelligenceApi,
	#[serde(rename = "azure-openai-responses")]
	AzureOpenAiResponses,
	#[serde(rename = "bedrock-converse-stream")]
	BedrockConverseStream,
	#[serde(rename = "cursor-agent")]
	CursorAgent,
	#[serde(rename = "devin-agent")]
	DevinAgent,
	#[serde(rename = "gitlab-duo-agent")]
	GitlabDuoAgent,
	#[serde(rename = "google-gemini-cli")]
	GoogleGeminiCli,
	#[serde(rename = "google-generative-ai")]
	GoogleGenerativeAi,
	#[serde(rename = "google-vertex")]
	GoogleVertex,
	#[serde(rename = "ollama-chat")]
	OllamaChat,
	#[serde(rename = "openai-codex-responses")]
	OpenAiCodexResponses,
	#[serde(rename = "openai-completions")]
	OpenAiCompletions,
	#[serde(rename = "openai-responses")]
	OpenAiResponses,
	#[serde(rename = "openrouter")]
	OpenRouter,
}

impl RawApi {
	const fn transport(self) -> TransportId {
		match self {
			Self::AnthropicMessages => TransportId::AnthropicMessages,
			Self::AppleIntelligenceApi => TransportId::Embedded,
			Self::AzureOpenAiResponses | Self::OpenAiResponses => TransportId::OpenAiResponses,
			Self::BedrockConverseStream => TransportId::BedrockConverse,
			Self::CursorAgent => TransportId::Cursor,
			Self::DevinAgent => TransportId::Devin,
			Self::GitlabDuoAgent => TransportId::GitLabDuoWorkflow,
			Self::GoogleGeminiCli => TransportId::GoogleCca,
			Self::GoogleGenerativeAi => TransportId::GoogleGenAi,
			Self::GoogleVertex => TransportId::GoogleVertex,
			Self::OllamaChat => TransportId::OllamaChat,
			Self::OpenAiCodexResponses => TransportId::OpenAiCodex,
			Self::OpenAiCompletions | Self::OpenRouter => TransportId::OpenAiChat,
		}
	}
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum RawModality {
	Text,
	Image,
	Audio,
	Video,
	Pdf,
}

impl From<RawModality> for Modality {
	fn from(value: RawModality) -> Self {
		match value {
			RawModality::Text => Self::Text,
			RawModality::Image => Self::Image,
			RawModality::Audio => Self::Audio,
			RawModality::Video => Self::Video,
			RawModality::Pdf => Self::Pdf,
		}
	}
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawCost {
	#[serde(default)]
	input:        f64,
	#[serde(default)]
	output:       f64,
	#[serde(default)]
	cache_read:   f64,
	#[serde(default)]
	cache_write:  f64,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	long_context: Option<RawLongContextCost>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawLongContextCost {
	input_threshold: u64,
	#[serde(default)]
	input:           f64,
	#[serde(default)]
	output:          f64,
	#[serde(default)]
	cache_read:      f64,
	#[serde(default)]
	cache_write:     f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawRemoteCompaction {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	enabled:              Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	api:                  Option<RawApi>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	endpoint:             Option<Str>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	v2_streaming_enabled: Option<bool>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	v2_endpoint:          Option<Str>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	streaming_endpoint:   Option<Str>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	model:                Option<Str>,
}

impl From<RawRemoteCompaction> for ModelRemoteCompaction {
	fn from(value: RawRemoteCompaction) -> Self {
		Self {
			enabled:              value.enabled,
			transport:            value.api.map(RawApi::transport),
			endpoint:             value.endpoint,
			v2_streaming_enabled: value.v2_streaming_enabled,
			v2_endpoint:          value.v2_endpoint,
			streaming_endpoint:   value.streaming_endpoint,
			model:                value.model,
		}
	}
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(transparent)]
struct RawCompat(BTreeMap<String, Value>);

impl<'de> Deserialize<'de> for RawCompat {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		let values = BTreeMap::<String, Value>::deserialize(deserializer)?;
		for key in values.keys() {
			if canonical_compat_key(key).is_none() {
				return Err(de::Error::custom(format!("unknown compatibility key `{key}`")));
			}
		}
		Ok(Self(values))
	}
}

fn canonical_compat_key(key: &str) -> Option<&'static str> {
	Some(match key {
		"allowsSyntheticReasoningContentForToolCalls" => {
			"allows_synthetic_reasoning_content_for_tool_calls"
		},
		"disableAdaptiveThinking" => "disable_adaptive_thinking",
		"disableReasoningOnToolChoice" => "disable_reasoning_on_tool_choice",
		"escapeBuiltinToolNames" => "escape_builtin_tool_names",
		"extraBody" => "extra_body",
		"filterReasoningHistory" => "filter_reasoning_history",
		"includeEncryptedReasoning" => "include_encrypted_reasoning",
		"maxTokensField" => "max_tokens_field",
		"officialEndpoint" => "official_endpoint",
		"omitReasoningEffort" => "omit_reasoning_effort",
		"reasoningContentField" => "reasoning_content_field",
		"reasoningEffortMap" => "reasoning_effort_map",
		"reasoningDisableMode" => "reasoning_disable_mode",
		"replayUnsignedThinking" => "replay_unsigned_thinking",
		"requiresAssistantContentForToolCalls" => "requires_assistant_content_for_tool_calls",
		"requiresReasoningContentForToolCalls" => "requires_reasoning_content_for_tool_calls",
		"requiresReasoningContentForAllAssistantTurns" => {
			"requires_reasoning_content_for_all_assistant_turns"
		},
		"requiresThinkingEnabled" => "requires_thinking_enabled",
		"requiresToolResultId" => "requires_tool_result_id",
		"signingEndpoint" => "signing_endpoint",
		"streamIdleTimeoutMs" => "stream_idle_timeout_ms",
		"supportsDeveloperRole" => "supports_developer_role",
		"supportsEagerToolInputStreaming" => "supports_eager_tool_input_streaming",
		"supportsForcedToolChoice" => "supports_forced_tool_choice",
		"supportsImageDetailOriginal" => "supports_image_detail_original",
		"supportsLongCacheRetention" => "supports_long_cache_retention",
		"supportsMidConversationSystem" => "supports_mid_conversation_system",
		"supportsReasoningEffort" => "supports_reasoning_effort",
		"supportsSamplingParams" => "supports_sampling_params",
		"supportsStore" => "supports_store",
		"supportsToolChoice" => "supports_tool_choice",
		"supportsUsageInStreaming" => "supports_usage_in_streaming",
		"thinkingFormat" => "thinking_format",
		"whenThinking" => "when_thinking",
		_ => return None,
	})
}

#[derive(Default)]
struct Interner {
	values: BTreeMap<Str, Str>,
}

impl Interner {
	fn intern(&mut self, value: &str) -> Str {
		match self.values.entry(Str::new(value)) {
			Entry::Occupied(entry) => entry.get().clone(),
			Entry::Vacant(entry) => {
				let value = entry.key().clone();
				entry.insert(value.clone());
				value
			},
		}
	}
}

fn normalize_provider_values(
	source: BTreeMap<String, BTreeMap<String, Value>>,
) -> Result<BTreeMap<String, BTreeMap<String, Value>>, CatalogError> {
	let mut providers = BTreeMap::new();
	for (provider_key, source_models) in source {
		let provider = provider_key.trim();
		if provider.is_empty() {
			return Err(CatalogError::EmptyIdentity { kind: "provider" });
		}
		let mut models = BTreeMap::new();
		for (model_key, mut value) in source_models {
			let model = model_key.trim();
			if model.is_empty() {
				return Err(CatalogError::EmptyIdentity { kind: "model" });
			}
			set_model_identity(&mut value, provider, model);
			if models.insert(model.to_owned(), value).is_some() {
				return Err(CatalogError::DuplicateIdentity {
					kind:     "model",
					identity: format!("{provider}/{model}"),
				});
			}
		}
		if providers.insert(provider.to_owned(), models).is_some() {
			return Err(CatalogError::DuplicateIdentity {
				kind:     "provider",
				identity: provider.to_owned(),
			});
		}
	}
	Ok(providers)
}

fn set_model_identity(value: &mut Value, provider: &str, model: &str) {
	let fields = value
		.as_object_mut()
		.expect("RawModel validation guarantees an object");
	fields.insert("id".to_owned(), Value::String(model.to_owned()));
	fields.insert("provider".to_owned(), Value::String(provider.to_owned()));
}

fn sort_value_keys(value: &mut Value) {
	match value {
		Value::Object(fields) => {
			for value in fields.values_mut() {
				sort_value_keys(value);
			}
			fields.sort_keys();
		},
		Value::Array(values) => {
			for value in values {
				sort_value_keys(value);
			}
		},
		_ => {},
	}
}

fn promote_openai_embedding_values(providers: &mut BTreeMap<String, BTreeMap<String, Value>>) {
	const PROMOTIONS: [(&str, &str, u32); 2] = [
		("openai/text-embedding-3-small", "text-embedding-3-small", 1_536),
		("openai/text-embedding-3-large", "text-embedding-3-large", 3_072),
	];
	let promoted: Vec<(&str, u32, Value)> = {
		let Some(zenmux) = providers.get("zenmux") else {
			return;
		};
		PROMOTIONS
			.into_iter()
			.filter_map(|(source, model, dimensions)| {
				zenmux
					.get(source)
					.cloned()
					.map(|value| (model, dimensions, value))
			})
			.collect()
	};
	let openai = providers.entry("openai".to_owned()).or_default();
	for (model, dimensions, mut value) in promoted {
		set_model_identity(&mut value, "openai", model);
		value
			.as_object_mut()
			.expect("RawModel validation guarantees an object")
			.insert("embeddingDimensions".to_owned(), Value::from(dimensions));
		openai.entry(model.to_owned()).or_insert(value);
	}
}

fn translate_provider_map(
	providers: BTreeMap<String, BTreeMap<String, RawModel>>,
) -> Vec<ModelCard> {
	let mut intern = Interner::default();
	let mut models = Vec::new();
	for (outer_provider, provider_models) in providers {
		for (outer_model, raw_model) in provider_models {
			models.push(translate_model(&mut intern, &outer_provider, &outer_model, raw_model));
		}
	}
	finalize_models(models)
}

fn finalize_models(mut models: Vec<ModelCard>) -> Vec<ModelCard> {
	let references = models.clone();
	let index = crate::identity::build_model_reference_index(&references);
	for model in &mut models {
		crate::identity::resolve_and_inherit_model_reference(model, &index);
	}
	let mut models = crate::identity::collapse_effort_variants_across_providers(models);
	models.sort_unstable_by(|left, right| {
		(&left.provider, &left.model).cmp(&(&right.provider, &right.model))
	});
	models
}

fn translate_model(
	intern: &mut Interner,
	outer_provider: &str,
	outer_model: &str,
	raw: RawModel,
) -> ModelCard {
	let provider_text = raw.provider.as_deref().unwrap_or(outer_provider);
	let model_text = raw.id.as_deref().unwrap_or(outer_model);
	let provider = intern.intern(provider_text);
	let model = intern.intern(model_text);
	let family = intern.intern(crate::identity::family_token(model_text).as_str());
	let is_deepseek_family = family.as_str() == "deepseek"
		|| raw
			.name
			.as_deref()
			.is_some_and(|name| crate::identity::family_token(name).as_str() == "deepseek");
	let is_openrouter = provider_text == "openrouter"
		|| raw
			.base_url
			.as_deref()
			.is_some_and(|url| url.contains("openrouter.ai"));
	let canonical = intern.intern(&format!("{provider_text}/{model_text}"));
	let name = intern.intern(raw.name.as_deref().unwrap_or(model_text));
	let facet = infer_model_facet(model_text);
	let mut facets = SmallVec::new();
	if let Some(facet) = facet {
		facets.push(facet);
	}
	let inputs = raw.input.into_iter().map(Modality::from).collect();
	let mut outputs: SmallVec<Modality, 4> = raw.output.into_iter().map(Modality::from).collect();
	if outputs.is_empty() && facet == Some(Facet::Chat) {
		outputs.push(Modality::Text);
	}
	let mut efforts = SmallVec::new();
	let mut effort_routing = BTreeMap::new();
	if facet == Some(Facet::Chat)
		&& let Some(thinking) = raw.thinking.as_ref()
	{
		for value in &thinking.efforts {
			let effort = value.portable();
			if !efforts.contains(&effort) {
				efforts.push(effort);
			}
		}
		for (&effort, route) in &thinking.effort_routing {
			effort_routing.insert(effort.portable(), intern.intern(route));
		}
	}
	let mut pricing = SmallVec::new();
	push_price(&mut pricing, PriceUnit::MtokInput, raw.cost.input);
	push_price(&mut pricing, PriceUnit::MtokOutput, raw.cost.output);
	push_price(&mut pricing, PriceUnit::MtokCacheRead, raw.cost.cache_read);
	push_price(&mut pricing, PriceUnit::MtokCacheWrite, raw.cost.cache_write);
	let mut pricing_tiers = SmallVec::new();
	if let Some(long_context) = raw.cost.long_context {
		let mut tier_pricing = SmallVec::new();
		push_price(&mut tier_pricing, PriceUnit::MtokInput, long_context.input);
		push_price(&mut tier_pricing, PriceUnit::MtokOutput, long_context.output);
		push_price(&mut tier_pricing, PriceUnit::MtokCacheRead, long_context.cache_read);
		push_price(&mut tier_pricing, PriceUnit::MtokCacheWrite, long_context.cache_write);
		pricing_tiers.push(PriceTier {
			prompt_tokens_above: long_context.input_threshold,
			pricing:             tier_pricing,
		});
	}
	let mut props = Props::default();
	if let Some(dimensions) = raw.embedding_dimensions {
		let _ =
			props.insert_ns("openai", "embedding_dimensions", serde_json::Value::from(dimensions));
	}
	let wire = raw
		.api
		.map(|api| ModelWire { transport: api.transport(), base_url: raw.base_url });
	let mut compat = Props::default();
	if let Some(raw_compat) = raw.compat {
		for (key, value) in raw_compat.0 {
			let canonical =
				canonical_compat_key(&key).expect("RawCompat rejects unknown compatibility keys");
			let _ = compat.insert_ns("wire", canonical, value);
		}
	}
	if raw.reasoning
		&& !is_openrouter
		&& is_deepseek_family
		&& compat
			.get_ns("wire", "requires_reasoning_content_for_all_assistant_turns")
			.is_none()
	{
		let _ = compat.insert_ns(
			"wire",
			"requires_reasoning_content_for_all_assistant_turns",
			Value::Bool(true),
		);
	}
	let behavior = ModelBehavior {
		thinking: raw.thinking,
		supports_tools: raw.supports_tools,
		supports_computer_use: raw.supports_computer_use,
		supports_computer_use_config: raw.supports_computer_use_config,
		cursor_max_mode: raw.cursor_max_mode,
		omit_max_output_tokens: raw.omit_max_output_tokens,
		apply_patch_tool_type: raw.apply_patch_tool_type,
		context_promotion_target: raw.context_promotion_target,
		request_model_id: raw.request_model_id,
		remote_compaction: raw.remote_compaction.map(ModelRemoteCompaction::from),
		premium_multiplier: raw.premium_multiplier,
		reasoning_mode: raw.reasoning_mode,
		use_responses_lite: raw.use_responses_lite,
		prefer_websockets: raw.prefer_websockets,
		priority: raw.priority,
		headers: raw.headers,
		compat,
	};
	ModelCard {
		id: canonical,
		provider,
		model,
		name,
		family,
		facets,
		inputs,
		outputs,
		reasoning: raw.reasoning && facet == Some(Facet::Chat),
		efforts,
		context_window: raw.context_window.unwrap_or_default(),
		max_output_tokens: raw.max_tokens.unwrap_or_default(),
		pricing,
		pricing_tiers,
		availability: Availability::Unspecified,
		source: Source::Bundled,
		blocked_until_ms: 0,
		deprecated: raw.deprecated,
		updated_at_ms: 0,
		props,
		effort_routing,
		behavior,
		wire,
	}
}

fn infer_model_facet(model: &str) -> Option<Facet> {
	let normalized = model.to_ascii_lowercase();
	if normalized.contains("rerank") {
		// Pi's sole reranker row is Fireworks qwen3-reranker-8b, but Pi has
		// no rerank request contract or provider dispatch. Keeping this empty
		// prevents its generic `openai-completions` tag from becoming fake chat.
		return None;
	}
	let embedding = normalized.contains("embed")
		|| normalized.starts_with("voyage-")
		|| normalized.starts_with("bge-")
		|| normalized.contains("/bge-");
	Some(if embedding {
		Facet::Embeddings
	} else {
		Facet::Chat
	})
}

fn push_price(pricing: &mut SmallVec<Price, 4>, unit: PriceUnit, dollars: f64) {
	if dollars.is_finite() && dollars >= 0.0 {
		pricing.push(Price { unit, nanos_usd: (dollars * 1_000_000_000.0).round() as u64 });
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn provider_map_bypasses_untagged_arbitrary_precision_buffering() {
		let json = zstd::stream::decode_all(MODELS_JSON_ZST).expect("snapshot decompresses");
		let providers: BTreeMap<String, BTreeMap<String, RawModel>> =
			serde_json::from_slice(&json).expect("provider map decodes directly");
		assert_eq!(providers.len(), 80);
		assert_eq!(providers.values().map(BTreeMap::len).sum::<usize>(), 4_293);
	}

	fn priced_model() -> ModelCard {
		ModelCard {
			id:                Str::new("test/model"),
			provider:          Str::new("test"),
			model:             Str::new("model"),
			name:              Str::new("Model"),
			family:            Str::new("test"),
			facets:            SmallVec::new(),
			inputs:            SmallVec::new(),
			outputs:           SmallVec::new(),
			reasoning:         false,
			efforts:           SmallVec::new(),
			context_window:    0,
			max_output_tokens: 0,
			pricing:           [
				Price { unit: PriceUnit::MtokInput, nanos_usd: 2_000_000_000 },
				Price { unit: PriceUnit::MtokOutput, nanos_usd: 4_000_000_000 },
				Price { unit: PriceUnit::MtokCacheRead, nanos_usd: 200_000_000 },
				Price { unit: PriceUnit::MtokCacheWrite, nanos_usd: 2_500_000_000 },
			]
			.into_iter()
			.collect(),
			pricing_tiers:     SmallVec::new(),
			availability:      Availability::Unspecified,
			source:            Source::Bundled,
			blocked_until_ms:  0,
			deprecated:        false,
			updated_at_ms:     0,
			props:             Props::default(),
			effort_routing:    BTreeMap::new(),
			behavior:          ModelBehavior::default(),
			wire:              None,
		}
	}

	fn close(left: f64, right: f64) {
		assert!((left - right).abs() < 1e-12, "{left} != {right}");
	}

	#[test]
	fn cache_write_without_ttl_uses_flat_rate() {
		let usage = CostUsage { cache_write: 100, ..CostUsage::default() };
		close(cache_write_cost(&priced_model(), &usage), 0.000_25);
	}

	#[test]
	fn cache_write_ttl_prices_residual_and_one_hour_separately() {
		let usage = CostUsage {
			cache_write: 100,
			cttl: Some(CacheTtlUsage { ephemeral_5m: 30, ephemeral_1h: 20 }),
			..CostUsage::default()
		};
		close(cache_write_cost(&priced_model(), &usage), 0.000_28);
	}

	#[test]
	fn cache_write_ttl_clamps_negative_residual() {
		let usage = CostUsage {
			cache_write: 40,
			cttl: Some(CacheTtlUsage { ephemeral_5m: 30, ephemeral_1h: 20 }),
			..CostUsage::default()
		};
		close(cache_write_cost(&priced_model(), &usage), 0.000_155);
	}

	#[test]
	fn calculate_cost_includes_orchestration() {
		let usage = CostUsage {
			input: 10,
			output: 5,
			cache_read: 20,
			orchestration: Some(OrchestrationUsage { input: 2, output: 3, cache_read: 4 }),
			..CostUsage::default()
		};
		let cost = calculate_cost(&priced_model(), &usage);
		close(cost.input, 0.000_024);
		close(cost.output, 0.000_032);
		close(cost.cache_read, 0.000_004_8);
		close(cost.total, 0.000_060_8);
	}

	#[test]
	fn long_context_tier_prices_full_request_above_prompt_threshold() {
		let mut model = priced_model();
		model.pricing_tiers.push(PriceTier {
			prompt_tokens_above: 272_000,
			pricing:             [
				Price { unit: PriceUnit::MtokInput, nanos_usd: 10_000_000_000 },
				Price { unit: PriceUnit::MtokOutput, nanos_usd: 45_000_000_000 },
				Price { unit: PriceUnit::MtokCacheRead, nanos_usd: 1_000_000_000 },
				Price { unit: PriceUnit::MtokCacheWrite, nanos_usd: 12_500_000_000 },
			]
			.into_iter()
			.collect(),
		});
		let short = calculate_cost(
			&model,
			&CostUsage { input: 270_000, cache_read: 2_000, output: 1_000, ..Default::default() },
		);
		close(short.input, 0.54);
		close(short.output, 0.004);
		let long = calculate_cost(
			&model,
			&CostUsage { input: 270_001, cache_read: 2_000, output: 1_000, ..Default::default() },
		);
		close(long.input, 2.700_01);
		close(long.output, 0.045);
		close(long.cache_read, 0.002);
	}

	#[test]
	fn translation_retains_behavior_but_serialization_omits_server_metadata() {
		let json = br#"{"proxy":{"model":{"id":"model","provider":"proxy","name":"Model","api":"openai-responses","baseUrl":"https://private.example","headers":{"x-private":"yes","authorization":"secret"},"contextPromotionTarget":"proxy/large","requestModelId":"wire-model","preferWebsockets":true,"supportsTools":false,"premiumMultiplier":0.33,"thinking":{"mode":"effort","efforts":["low","xhigh","max"],"defaultLevel":"xhigh","requiresEffort":true},"compat":{"thinkingFormat":"zai"}}}}"#;
		let catalog = load_catalog_json(json).expect("valid translated catalog");
		let model = catalog.get("proxy", "model").expect("translated model");
		assert_eq!(model.behavior.supports_tools, Some(false));
		assert_eq!(model.behavior.request_model_id.as_deref(), Some("wire-model"));
		assert_eq!(
			model.behavior.premium_multiplier,
			Some(PremiumMultiplier::from_millionths(330_000))
		);
		assert_eq!(
			model.behavior.compat.get_ns("wire", "thinking_format"),
			Some(&Value::String("zai".into()))
		);
		assert_eq!(model.behavior.headers["x-private"], "yes");
		let policy = model.behavior.resolved_policy();
		assert_eq!(policy.request_model_id.as_deref(), Some("wire-model"));
		assert_eq!(policy.premium_millionths, Some(330_000));
		assert_eq!(
			policy
				.thinking
				.as_ref()
				.map(|thinking| thinking.efforts.as_slice()),
			Some([Effort::Low, Effort::XHigh, Effort::Max].as_slice()),
		);
		assert_eq!(policy.headers.get("x-private"), Some("yes"));
		assert_eq!(policy.headers.get("authorization"), None);
		assert_eq!(
			model.wire.as_ref().map(|wire| wire.base_url.as_deref()),
			Some(Some("https://private.example"))
		);
		let mut serialized = serde_json::to_value(model).expect("model card serializes");
		assert!(serialized.get("behavior").is_none());
		assert!(serialized.get("wire").is_none());
		assert!(serialized.get("headers").is_none());
		assert!(serialized.get("base_url").is_none());
		let object = serialized.as_object_mut().expect("model object");
		object.insert("behavior".into(), serde_json::json!({"request_model_id":"injected"}));
		object.insert("effort_routing".into(), serde_json::json!({"high":"injected"}));
		object.insert(
			"wire".into(),
			serde_json::json!({
				"transport":"open_ai_responses",
				"base_url":"https://injected.example"
			}),
		);
		let foreign: ModelCard = serde_json::from_value(serialized).expect("foreign card decodes");
		assert_eq!(foreign.behavior, ModelBehavior::default());
		assert!(foreign.effort_routing.is_empty());
		assert!(foreign.wire.is_none());
	}

	#[test]
	fn importer_rejects_unknown_model_thinking_and_compat_keys_with_paths() {
		for (json, unknown) in [
			(br#"{"p":{"m":{"surprise":true}}}"#.as_slice(), "surprise"),
			(
				br#"{"p":{"m":{"thinking":{"mode":"effort","efforts":["low"],"surprise":true}}}}"#
					.as_slice(),
				"surprise",
			),
			(
				br#"{"p":{"m":{"compat":{"thinkingFormat":"zai","surprise":true}}}}"#.as_slice(),
				"surprise",
			),
		] {
			let error = import_catalog_zstd(json).expect_err("unknown metadata must fail import");
			assert!(
				error.to_string().contains("p/m") && error.to_string().contains(unknown),
				"error should name `{unknown}`: {error}"
			);
		}
	}

	#[test]
	fn non_chat_model_ids_receive_native_facets() {
		let json = br#"{"fireworks":{"qwen3-embedding-8b":{"input":["text"]},"qwen3-reranker-8b":{"input":["text"]},"qwen3-8b":{"input":["text"]},"voyage-code-2":{"input":["text"]},"baai/bge-m3":{"input":["text"]},"nvidia/nv-embedqa-e5-v5":{"input":["text"]}}}"#;
		let catalog = load_catalog_json(json).expect("valid translated catalog");
		assert_eq!(
			catalog
				.get("fireworks", "qwen3-embedding-8b")
				.unwrap()
				.facets
				.as_slice(),
			&[Facet::Embeddings]
		);
		assert!(
			catalog
				.get("fireworks", "qwen3-reranker-8b")
				.unwrap()
				.facets
				.is_empty()
		);
		for id in ["voyage-code-2", "baai/bge-m3", "nvidia/nv-embedqa-e5-v5"] {
			assert_eq!(catalog.get("fireworks", id).unwrap().facets.as_slice(), &[Facet::Embeddings]);
		}
		assert_eq!(
			catalog
				.get("fireworks", "qwen3-8b")
				.unwrap()
				.facets
				.as_slice(),
			&[Facet::Chat]
		);
	}

	#[test]
	fn embedded_catalog_is_lazily_shared_and_populated() {
		let first = embedded_catalog();
		let second = embedded_catalog();
		assert_eq!(first.len(), 4_218);
		assert!(std::ptr::eq(first, second));
	}
}
