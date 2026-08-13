//! Behavioral conformance against the frozen catalog oracle.

use std::{
	collections::{BTreeMap, BTreeSet},
	str::FromStr,
};

use omp_llm_catalog::{
	Availability, EmbeddingInputBits, EvidenceConfidence, ModalityBits, ModelAvailability,
	OperationKind, ProvenanceKind,
	classify::{ClassificationInput, ClassificationPhase, EffortTier, ModelVersion, classify},
	compile::{CompiledCatalog, compile_oracle},
	policy::{
		ApplyPatchWireKind, ComputerUseConfigSupport, ComputerUseWireSupport, ExtendedContextMode,
		MaxOutputTokensEmission, MaxTokensField, ReasoningDisableMode, ThinkingFormat, WirePolicy,
	},
	pricing::{PriceUnit, UsageDimensions},
	provider::{AuthSpecKind, CodexTransportPreference, OAuthRefreshBehavior},
	snapshot::{Catalog, SnapshotProvenance},
	thinking::{ThinkingMode, ThinkingPolicy},
};
use serde::Deserialize;
use serde_json::value::RawValue;

const FAMILY_CASES: &str =
	include_str!("../../../fixtures/llm-oracle/catalog-policy/family-classifier.json");
const VERSION_CASES: &str =
	include_str!("../../../fixtures/llm-oracle/catalog-policy/openai-version-aliases.json");
const EFFORT_CASES: &str =
	include_str!("../../../fixtures/llm-oracle/catalog-policy/effort-tier-classifier.json");
const THINKING_PROFILES: &str =
	include_str!("../../../fixtures/llm-oracle/catalog-policy/thinking-profiles.json");
const COMPAT_PROFILES: &str =
	include_str!("../../../fixtures/llm-oracle/catalog-policy/compat-profiles.json");
const CATALOG_POSTCARD: &[u8] = include_bytes!("../data/catalog.postcard");
const PROVIDERS: &str = include_str!("../../../fixtures/llm-oracle/catalog/providers.toml");
const OAUTH: &str = include_str!("../../../fixtures/llm-oracle/catalog/oauth.toml");
const MODELS_ZSTD: &[u8] = include_bytes!("../../../fixtures/llm-oracle/catalog/models.json.zst");
const PRICE_CASES: &str =
	include_str!("../../../fixtures/llm-oracle/catalog-policy/aliases-tiers-and-deepseek.json");
const EXACT_OVERRIDES: &str =
	include_str!("../../../fixtures/llm-oracle/catalog-policy/exact-model-overrides.json");
const QWEN_COLLAPSE: &str =
	include_str!("../../../fixtures/llm-oracle/catalog-policy/qwen-collapse-cases.json");
const NORMALIZED_MODELS: &str =
	include_str!("../../../fixtures/llm-oracle/catalog/models.normalized.json");
const CENSUS: &str = include_str!("../../../fixtures/llm-oracle/catalog/census.json");

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Census {
	schema_version:           u32,
	curated_provider_catalog: CuratedProviderCensus,
	normalized_catalog:       NormalizedCatalogCensus,
	raw_catalog:              RawCatalogCensus,
	transports:               TransportCensus,
	urls:                     UrlCensus,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CuratedProviderCensus {
	provider_count: usize,
	provider_keys:  Vec<String>,
	source:         String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NormalizedCatalogCensus {
	model_count:           usize,
	models_by_provider:    BTreeMap<String, usize>,
	sort_key:              Vec<String>,
	source:                String,
	unique_identity_count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCatalogCensus {
	provider_key_count: usize,
	provider_keys:      Vec<String>,
	row_count:          usize,
	rows_by_provider:   BTreeMap<String, usize>,
	source:             String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransportCensus {
	active:        Vec<String>,
	active_count:  usize,
	source:        String,
	variant_count: usize,
	variants:      Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UrlCensus {
	curated_provider_distinct_count: usize,
	distinct_count: usize,
	intersection_count: usize,
	normalized_model_distinct_count: usize,
	source: String,
	values: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NormalizedOracle {
	schema_version: u32,
	models:         Vec<OracleModel>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleModel {
	id:               String,
	provider:         String,
	model:            String,
	name:             String,
	family:           String,
	facets:           Vec<String>,
	modalities:       OracleModalities,
	reasoning:        bool,
	efforts:          Vec<FixtureEffort>,
	limits:           OracleLimits,
	pricing:          Vec<OraclePrice>,
	pricing_tiers:    Vec<OraclePriceTier>,
	availability:     String,
	source:           String,
	blocked_until_ms: u64,
	deprecated:       bool,
	updated_at_ms:    u64,
	props:            BTreeMap<String, u32>,
	effort_routing:   BTreeMap<FixtureEffort, String>,
	wire:             OracleWire,
	behavior:         Box<RawValue>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleModalities {
	inputs:  Vec<String>,
	outputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleLimits {
	context_window:    u64,
	max_output_tokens: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OraclePrice {
	unit:      PriceUnit,
	nanos_usd: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OraclePriceTier {
	prompt_tokens_above: u64,
	pricing:             Vec<OraclePrice>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleWire {
	transport: String,
	base_url:  String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct OracleBehaviorCapabilities {
	supports_tools: Option<bool>,
	supports_computer_use: Option<bool>,
	supports_computer_use_config: Option<bool>,
	omit_max_output_tokens: Option<bool>,
	apply_patch_tool_type: Option<String>,
	cursor_max_mode: Option<bool>,
	context_promotion_target: Option<String>,
	request_model_id: Option<String>,
	remote_compaction: Option<OracleRemoteCompaction>,
	premium_multiplier: Option<f64>,
	reasoning_mode: Option<String>,
	use_responses_lite: Option<bool>,
	prefer_websockets: Option<bool>,
	priority: Option<u32>,
	headers: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleRemoteCompaction {
	enabled:              bool,
	transport:            String,
	endpoint:             Option<String>,
	v2_streaming_enabled: bool,
	v2_endpoint:          Option<String>,
	streaming_endpoint:   Option<String>,
	model:                Option<String>,
}

type RawPricingOracle = BTreeMap<String, BTreeMap<String, RawPricingModel>>;

#[derive(Debug, Deserialize)]
struct RawPricingModel {
	#[serde(default)]
	cost: RawCost,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct RawCost {
	input:       f64,
	output:      f64,
	cache_read:  f64,
	cache_write: f64,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyCases {
	schema_version: u32,
	unknown_family: String,
	cases:          Vec<FamilyCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyCase {
	case_kind:       String,
	input:           String,
	expected_family: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VersionCases {
	schema_version:   u32,
	alias_provenance: String,
	cases:            Vec<VersionCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VersionCase {
	case_kind:        String,
	input:            String,
	expected_version: Option<ModelVersion>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EffortCases {
	schema_version: u32,
	collapse_minimum_tier_siblings: usize,
	synthetic_collapse: SyntheticCollapse,
	cases: Vec<EffortCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EffortCase {
	input:    String,
	expected: Option<ExpectedEffort>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedEffort {
	logical_model: String,
	tier:          FixtureEffort,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SyntheticCollapse {
	provider: String,
	inputs:   Vec<String>,
	expected: SyntheticCollapseExpected,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SyntheticCollapseExpected {
	logical_model:  String,
	efforts:        Vec<FixtureEffort>,
	effort_routing: std::collections::BTreeMap<FixtureEffort, String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(rename_all = "snake_case")]
enum FixtureEffort {
	Off,
	Minimal,
	Low,
	Medium,
	High,
	#[serde(alias = "xhigh")]
	XHigh,
	Max,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThinkingProfiles {
	schema_version: u32,
	profile_count:  usize,
	normalization:  String,
	profiles:       Vec<ThinkingProfileCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThinkingProfileCase {
	profile_id:  String,
	model_count: usize,
	models:      Vec<String>,
	shape:       ThinkingPolicy,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompatProfiles {
	schema_version: u32,
	profile_count:  usize,
	normalization:  String,
	profiles:       Vec<CompatProfileCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompatProfileCase {
	profile_id:  String,
	model_count: usize,
	models:      Vec<String>,
	shape:       CompatShape,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct CompatShape {
	#[serde(rename = "wire/allows_synthetic_reasoning_content_for_tool_calls")]
	allows_synthetic_reasoning_content_for_tool_calls: Option<bool>,
	#[serde(rename = "wire/disable_adaptive_thinking")]
	disable_adaptive_thinking: Option<bool>,
	#[serde(rename = "wire/disable_reasoning_on_tool_choice")]
	disable_reasoning_on_tool_choice: Option<bool>,
	#[serde(rename = "wire/escape_builtin_tool_names")]
	escape_builtin_tool_names: Option<bool>,
	#[serde(rename = "wire/filter_reasoning_history")]
	filter_reasoning_history: Option<bool>,
	#[serde(rename = "wire/include_encrypted_reasoning")]
	include_encrypted_reasoning: Option<bool>,
	#[serde(rename = "wire/max_tokens_field")]
	max_tokens_field: Option<String>,
	#[serde(rename = "wire/official_endpoint")]
	official_endpoint: Option<bool>,
	#[serde(rename = "wire/omit_reasoning_effort")]
	omit_reasoning_effort: Option<bool>,
	#[serde(rename = "wire/reasoning_content_field")]
	reasoning_content_field: Option<String>,
	#[serde(rename = "wire/reasoning_disable_mode")]
	reasoning_disable_mode: Option<String>,
	#[serde(rename = "wire/reasoning_effort_map")]
	reasoning_effort_map: BTreeMap<FixtureEffort, String>,
	#[serde(rename = "wire/replay_unsigned_thinking")]
	replay_unsigned_thinking: Option<bool>,
	#[serde(rename = "wire/requires_assistant_content_for_tool_calls")]
	requires_assistant_content_for_tool_calls: Option<bool>,
	#[serde(rename = "wire/requires_reasoning_content_for_all_assistant_turns")]
	requires_reasoning_content_for_all_assistant_turns: Option<bool>,
	#[serde(rename = "wire/requires_reasoning_content_for_tool_calls")]
	requires_reasoning_content_for_tool_calls: Option<bool>,
	#[serde(rename = "wire/requires_thinking_enabled")]
	requires_thinking_enabled: Option<bool>,
	#[serde(rename = "wire/requires_tool_result_id")]
	requires_tool_result_id: Option<bool>,
	#[serde(rename = "wire/signing_endpoint")]
	signing_endpoint: Option<bool>,
	#[serde(rename = "wire/stream_idle_timeout_ms")]
	stream_idle_timeout_ms: Option<u64>,
	#[serde(rename = "wire/supports_developer_role")]
	supports_developer_role: Option<bool>,
	#[serde(rename = "wire/supports_eager_tool_input_streaming")]
	supports_eager_tool_input_streaming: Option<bool>,
	#[serde(rename = "wire/supports_forced_tool_choice")]
	supports_forced_tool_choice: Option<bool>,
	#[serde(rename = "wire/supports_image_detail_original")]
	supports_image_detail_original: Option<bool>,
	#[serde(rename = "wire/supports_long_cache_retention")]
	supports_long_cache_retention: Option<bool>,
	#[serde(rename = "wire/supports_mid_conversation_system")]
	supports_mid_conversation_system: Option<bool>,
	#[serde(rename = "wire/supports_reasoning_effort")]
	supports_reasoning_effort: Option<bool>,
	#[serde(rename = "wire/supports_sampling_params")]
	supports_sampling_params: Option<bool>,
	#[serde(rename = "wire/supports_store")]
	supports_store: Option<bool>,
	#[serde(rename = "wire/supports_tool_choice")]
	supports_tool_choice: Option<bool>,
	#[serde(rename = "wire/supports_usage_in_streaming")]
	supports_usage_in_streaming: Option<bool>,
	#[serde(rename = "wire/thinking_format")]
	thinking_format: Option<String>,
	#[serde(rename = "wire/extra_body")]
	extra_body: Option<omp_llm_catalog::policy::ReasoningBodyOverride>,
	#[serde(rename = "wire/when_thinking")]
	when_thinking: Option<FixtureWhenThinking>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FixtureWhenThinking {
	extra_body:      omp_llm_catalog::policy::ReasoningBodyOverride,
	thinking_format: ThinkingFormat,
}

impl CompatShape {
	fn into_policy(self) -> WirePolicy {
		let mut policy = WirePolicy::overrides();
		policy.reasoning.allows_synthetic_content_for_tool_calls =
			self.allows_synthetic_reasoning_content_for_tool_calls;
		policy.reasoning.disable_adaptive = self.disable_adaptive_thinking;
		policy.tool.disable_reasoning_on_choice = self.disable_reasoning_on_tool_choice;
		policy.tool.escape_builtin_names = self.escape_builtin_tool_names;
		policy.reasoning.filter_history = self.filter_reasoning_history;
		policy.reasoning.include_encrypted = self.include_encrypted_reasoning;
		policy.context.max_tokens_field = self
			.max_tokens_field
			.map(|value| MaxTokensField::from_str(&value).expect("known fixture max-token field"));
		policy.reasoning.official_endpoint = self.official_endpoint;
		policy.reasoning.omit_effort = self.omit_reasoning_effort;
		policy.reasoning.content_field = self.reasoning_content_field.map(Into::into);
		policy.reasoning.disable_mode = self.reasoning_disable_mode.map(|value| {
			ReasoningDisableMode::from_str(&value).expect("known fixture reasoning disable mode")
		});
		policy.reasoning.effort_map = self
			.reasoning_effort_map
			.into_iter()
			.map(|(effort, value)| {
				(omp_llm_catalog::thinking::ThinkingEffort::from(effort), value.into())
			})
			.collect();
		policy.reasoning.replay_unsigned = self.replay_unsigned_thinking;
		policy.tool.requires_assistant_content = self.requires_assistant_content_for_tool_calls;
		policy.reasoning.requires_content_for_all_assistant_turns =
			self.requires_reasoning_content_for_all_assistant_turns;
		policy.reasoning.requires_content_for_tool_calls =
			self.requires_reasoning_content_for_tool_calls;
		policy.reasoning.requires_enabled = self.requires_thinking_enabled;
		policy.tool.requires_result_id = self.requires_tool_result_id;
		policy.reasoning.signing_endpoint = self.signing_endpoint;
		policy.streaming.watchdog =
			self
				.stream_idle_timeout_ms
				.map(|idle_ms| omp_llm_catalog::policy::StreamWatchdog {
					first_event_ms: None,
					idle_ms:        Some(idle_ms),
				});
		policy.role.supports_developer_role = self.supports_developer_role;
		policy.tool.eager_input_streaming = self.supports_eager_tool_input_streaming;
		policy.tool.forced_choice = self.supports_forced_tool_choice;
		policy.image.supports_detail_original = self.supports_image_detail_original;
		policy.cache.supports_long_retention = self.supports_long_cache_retention;
		policy.role.supports_mid_conversation_system = self.supports_mid_conversation_system;
		policy.reasoning.supports_effort = self.supports_reasoning_effort;
		policy.structured.sampling_params = self.supports_sampling_params;
		policy.context.supports_store = self.supports_store;
		policy.tool.supports_tool_choice = self.supports_tool_choice;
		policy.usage.in_streaming = self.supports_usage_in_streaming;
		policy.reasoning.thinking_format = self
			.thinking_format
			.map(|value| ThinkingFormat::from_str(&value).expect("known fixture thinking format"));
		policy.reasoning.extra_body = self.extra_body;
		policy.reasoning.when_thinking =
			self
				.when_thinking
				.map(|value| omp_llm_catalog::policy::WhenThinkingPolicy {
					extra_body:      value.extra_body,
					thinking_format: value.thinking_format,
				});
		policy
	}
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PriceCases {
	schema_version:     u32,
	daybreak:           Vec<PriceModelCase>,
	long_context_tiers: Vec<PriceTierCase>,
	deepseek_efforts:   Vec<DeepseekEffortCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PriceModelCase {
	model:             String,
	context_window:    u64,
	max_output_tokens: u64,
	openai_version:    ModelVersion,
	pricing:           Vec<FixturePrice>,
	pricing_tiers:     Vec<FixtureTier>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PriceTierCase {
	model:         String,
	pricing_tiers: Vec<FixtureTier>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureTier {
	prompt_tokens_above: u64,
	pricing:             Vec<FixturePrice>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactOverrides {
	schema_version:    u32,
	source_assertions: String,
	cases:             Vec<ExactOverrideCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactOverrideCase {
	model:      String,
	expected:   ExactExpected,
	rationale:  String,
	provenance: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QwenCases {
	schema_version: u32,
	cases:          Vec<QwenCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QwenCase {
	provider:              String,
	inputs:                Vec<String>,
	absent_after_collapse: String,
	expected_logical:      QwenLogical,
	rationale:             String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ExactExpected {
	apply_patch_tool_type: Option<String>,
	thinking: Option<ExactThinking>,
	headers: BTreeMap<String, String>,
	premium_multiplier: Option<f64>,
	compat: Option<ExactCompat>,
	supports_computer_use: Option<bool>,
	supports_tools: Option<bool>,
	context_promotion_target: Option<String>,
	prefer_websockets: Option<bool>,
	priority: Option<u32>,
	remote_compaction: Option<ExactRemoteCompaction>,
	request_model_id: Option<String>,
	omit_max_output_tokens: Option<bool>,
	supports_computer_use_config: Option<bool>,
	use_responses_lite: Option<bool>,
	reasoning_mode: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExactThinking {
	mode:              String,
	efforts:           Vec<FixtureEffort>,
	#[serde(default)]
	effort_budgets:    BTreeMap<FixtureEffort, u64>,
	#[serde(default)]
	effort_routing:    BTreeMap<FixtureEffort, String>,
	#[serde(default)]
	suppress_when_off: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ExactCompat {
	#[serde(rename = "wire/thinking_format")]
	thinking_format: Option<String>,
	#[serde(rename = "wire/requires_reasoning_content_for_all_assistant_turns")]
	requires_reasoning_content_for_all_assistant_turns: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactRemoteCompaction {
	enabled:              bool,
	transport:            String,
	v2_streaming_enabled: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QwenLogical {
	model:          String,
	efforts:        Vec<FixtureEffort>,
	effort_routing: BTreeMap<FixtureEffort, String>,
	thinking:       QwenThinking,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct QwenThinking {
	mode:            String,
	efforts:         Vec<FixtureEffort>,
	effort_routing:  BTreeMap<FixtureEffort, String>,
	requires_effort: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixturePrice {
	unit:      omp_llm_catalog::pricing::PriceUnit,
	nanos_usd: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeepseekEffortCase {
	model:   String,
	efforts: Vec<FixtureEffort>,
}

impl From<FixtureEffort> for EffortTier {
	fn from(value: FixtureEffort) -> Self {
		match value {
			FixtureEffort::Off => Self::Off,
			FixtureEffort::Minimal => Self::Minimal,
			FixtureEffort::Low => Self::Low,
			FixtureEffort::Medium => Self::Medium,
			FixtureEffort::High => Self::High,
			FixtureEffort::XHigh => Self::XHigh,
			FixtureEffort::Max => Self::Max,
		}
	}
}

impl From<FixtureEffort> for omp_llm_catalog::thinking::ThinkingEffort {
	fn from(value: FixtureEffort) -> Self {
		match value {
			FixtureEffort::Off => Self::Off,
			FixtureEffort::Minimal => Self::Minimal,
			FixtureEffort::Low => Self::Low,
			FixtureEffort::Medium => Self::Medium,
			FixtureEffort::High => Self::High,
			FixtureEffort::XHigh => Self::XHigh,
			FixtureEffort::Max => Self::Max,
		}
	}
}

fn compiler_classification(model: &str) -> omp_llm_catalog::classify::ModelClassification {
	classify(ClassificationInput {
		phase: ClassificationPhase::CatalogCompiler,
		provider: "fixture",
		model,
		observed_at_ms: None,
	})
}

#[test]
fn family_classifier_matches_all_canonical_and_adversarial_cases() {
	let fixture: FamilyCases = serde_json::from_str(FAMILY_CASES).expect("family fixture is valid");
	assert_eq!(fixture.schema_version, 1);
	assert_eq!(fixture.cases.len(), 44);
	assert!(
		fixture
			.cases
			.iter()
			.any(|case| case.case_kind == "canonical")
	);
	assert!(
		fixture
			.cases
			.iter()
			.any(|case| case.case_kind == "adversarial_near_match_or_unknown")
	);

	for case in fixture.cases {
		let actual = compiler_classification(&case.input);
		let expected = if case.expected_family == fixture.unknown_family {
			"unknown"
		} else {
			case.expected_family.as_str()
		};
		assert_eq!(actual.family.as_str(), expected, "family classification for {:?}", case.input);
	}
}

#[test]
fn openai_versions_match_alias_canonical_and_adversarial_cases() {
	let fixture: VersionCases =
		serde_json::from_str(VERSION_CASES).expect("version fixture is valid");
	assert_eq!(fixture.schema_version, 1);
	assert!(!fixture.alias_provenance.is_empty());
	assert_eq!(fixture.cases.len(), 16);

	for case in fixture.cases {
		let actual = compiler_classification(&case.input);
		assert_eq!(actual.version, case.expected_version, "{} case {:?}", case.case_kind, case.input);
	}
}

#[test]
fn effort_suffix_classification_matches_every_boundary_case() {
	let fixture: EffortCases = serde_json::from_str(EFFORT_CASES).expect("effort fixture is valid");
	assert_eq!(fixture.schema_version, 1);
	assert_eq!(fixture.collapse_minimum_tier_siblings, 2);
	assert_eq!(fixture.synthetic_collapse.inputs.len(), 3);
	assert_eq!(fixture.synthetic_collapse.expected.efforts.len(), 3);
	assert_eq!(fixture.synthetic_collapse.expected.effort_routing.len(), 3);
	assert_eq!(fixture.synthetic_collapse.provider, "devin");
	assert_eq!(fixture.synthetic_collapse.expected.logical_model, "gpt-5.6-luna");

	for case in fixture.cases {
		let actual = compiler_classification(&case.input);
		match case.expected {
			Some(expected) => {
				assert_eq!(
					actual.logical_model.as_str(),
					expected.logical_model,
					"logical model for {:?}",
					case.input
				);
				assert_eq!(
					actual.effort,
					Some(expected.tier.into()),
					"effort tier for {:?}",
					case.input
				);
			},
			None => assert_eq!(actual.effort, None, "unexpected effort tier for {:?}", case.input),
		}
	}
}

fn compile_frozen_oracle() -> CompiledCatalog {
	compile_oracle(PROVIDERS, MODELS_ZSTD, OAUTH).expect("frozen catalog oracle compiles")
}

fn all_operations() -> [OperationKind; 15] {
	[
		OperationKind::Chat,
		OperationKind::CountTokens,
		OperationKind::Tokenize,
		OperationKind::Detokenize,
		OperationKind::Embed,
		OperationKind::GenerateImage,
		OperationKind::GenerateVideo,
		OperationKind::Speak,
		OperationKind::Transcribe,
		OperationKind::Realtime,
		OperationKind::Search,
		OperationKind::Usage,
		OperationKind::DiscoverModels,
		OperationKind::Auth,
		OperationKind::Native,
	]
}

fn oracle_codec(transport: &str) -> &'static str {
	match transport {
		"anthropic-messages" => "anthropic",
		"bedrock-converse" => "bedrock-converse",
		"cursor" => "cursor",
		"devin" => "devin",
		"gitlab-duo-workflow" => "gitlab-duo",
		"google-cca" => "google-cca",
		"google-gen-ai" => "google-genai",
		"google-vertex" => "google-vertex",
		"embedded" => "local",
		"ollama-chat" => "ollama",
		"open-ai-chat" => "openai-chat",
		"open-ai-codex" => "openai-codex",
		"open-ai-responses" => "openai-responses",
		other => panic!("inactive or unknown normalized transport {other}"),
	}
}

fn oracle_modalities(values: &[String]) -> ModalityBits {
	let mut modalities = ModalityBits::empty();
	for value in values {
		match value.as_str() {
			"text" => modalities.insert(ModalityBits::TEXT),
			"image" => modalities.insert(ModalityBits::IMAGE),
			other => panic!("unknown normalized modality {other}"),
		}
	}
	modalities
}

fn with_model_behavior(
	mut policy: WirePolicy,
	behavior: &OracleBehaviorCapabilities,
) -> WirePolicy {
	if let Some(enabled) = behavior.cursor_max_mode {
		policy.context.extended_mode = Some(ExtendedContextMode::from_enabled(enabled));
	}
	if let Some(omit) = behavior.omit_max_output_tokens {
		policy.context.max_output_tokens = Some(if omit {
			MaxOutputTokensEmission::Omit
		} else {
			MaxOutputTokensEmission::Emit
		});
	}
	if let Some(kind) = behavior.apply_patch_tool_type.as_deref() {
		policy.tool.apply_patch = Some(
			kind
				.parse::<ApplyPatchWireKind>()
				.expect("known apply-patch wire kind"),
		);
	}
	if let Some(supported) = behavior.supports_computer_use {
		policy.tool.computer_use = Some(if supported {
			ComputerUseWireSupport::Native
		} else {
			ComputerUseWireSupport::Unsupported
		});
	}
	if let Some(supported) = behavior.supports_computer_use_config {
		policy.tool.computer_use_config = Some(if supported {
			ComputerUseConfigSupport::Supported
		} else {
			ComputerUseConfigSupport::Unsupported
		});
	}
	policy
}

fn raw_price_nanos(value: f64) -> i128 {
	(value * 1_000_000_000.0).round() as i128
}

fn raw_prices(cost: &RawCost) -> BTreeMap<PriceUnit, i128> {
	[
		(PriceUnit::MtokInput, raw_price_nanos(cost.input)),
		(PriceUnit::MtokOutput, raw_price_nanos(cost.output)),
		(PriceUnit::MtokCacheRead, raw_price_nanos(cost.cache_read)),
		(PriceUnit::MtokCacheWrite, raw_price_nanos(cost.cache_write)),
	]
	.into_iter()
	.collect()
}
#[test]
fn compiled_catalog_matches_the_complete_frozen_census() {
	let expected: Census = serde_json::from_str(CENSUS).expect("catalog census is valid");
	let compiled = compile_frozen_oracle();
	assert_eq!(expected.schema_version, 1);
	assert_eq!(compiled.schema_version, 1);
	assert_eq!(expected.curated_provider_catalog.provider_count, 94);
	assert_eq!(expected.normalized_catalog.model_count, 4_227);
	assert_eq!(expected.normalized_catalog.unique_identity_count, 4_227);
	assert_eq!(expected.raw_catalog.provider_key_count, 80);
	assert_eq!(expected.raw_catalog.row_count, 4_302);
	assert_eq!(expected.raw_catalog.row_count - expected.normalized_catalog.model_count, 75);
	assert_eq!(expected.transports.variant_count, 16);
	assert_eq!(expected.transports.active_count, 13);
	assert_eq!(expected.urls.distinct_count, 108);
	assert_eq!(compiled.providers.len(), expected.curated_provider_catalog.provider_count);
	assert_eq!(compiled.models.len(), expected.normalized_catalog.model_count);
	assert_eq!(compiled.routes.len(), 210, "frozen distinct route-shape census");
	assert_eq!(
		compiled
			.routes
			.iter()
			.map(|route| &route.id)
			.collect::<BTreeSet<_>>()
			.len(),
		compiled.routes.len(),
		"route identifiers must be unique"
	);
	assert!(
		compiled
			.routes
			.windows(2)
			.all(|routes| routes[0].id < routes[1].id)
	);
	assert!(
		compiled
			.auth_specs
			.windows(2)
			.all(|specs| specs[0].id < specs[1].id)
	);
	assert!(
		compiled
			.oauth_specs
			.windows(2)
			.all(|specs| specs[0].id < specs[1].id)
	);
	assert!(
		compiled
			.header_profiles
			.windows(2)
			.all(|profiles| profiles[0].id < profiles[1].id)
	);
	assert!(
		compiled
			.discovery_specs
			.windows(2)
			.all(|specs| specs[0].id < specs[1].id)
	);
	assert!(
		compiled
			.wire_policies
			.windows(2)
			.all(|policies| policies[0].content_id() < policies[1].content_id())
	);
	assert!(
		compiled
			.thinking_policies
			.windows(2)
			.all(|policies| policies[0].content_id() < policies[1].content_id())
	);
	for provider in &compiled.providers {
		assert!(
			provider
				.routes
				.windows(2)
				.all(|routes| routes[0] < routes[1]),
			"{} route order",
			provider.id
		);
	}

	let provider_ids = compiled
		.providers
		.iter()
		.map(|provider| provider.id.as_str().to_owned())
		.collect::<Vec<_>>();
	assert_eq!(provider_ids, expected.curated_provider_catalog.provider_keys);
	let normalized: NormalizedOracle =
		serde_json::from_str(NORMALIZED_MODELS).expect("normalized model fixture is valid");
	assert_eq!(
		compiled
			.models
			.iter()
			.map(|model| model.key.as_str())
			.collect::<Vec<_>>(),
		normalized
			.models
			.iter()
			.map(|model| model.id.as_str())
			.collect::<Vec<_>>(),
		"compiled models must preserve the frozen provider/model tuple order"
	);

	let codecs = compiled
		.routes
		.iter()
		.map(|route| route.codec.as_str().to_owned())
		.collect::<BTreeSet<_>>();
	let expected_codecs = [
		"anthropic",
		"bedrock-converse",
		"cursor",
		"devin",
		"gitlab-duo",
		"google-cca",
		"google-genai",
		"google-vertex",
		"local",
		"ollama",
		"openai-chat",
		"openai-codex",
		"openai-responses",
		"search-exa",
		"search-kagi",
		"search-parallel",
		"search-perplexity",
		"search-tavily",
	]
	.into_iter()
	.map(str::to_owned)
	.collect::<BTreeSet<_>>();
	assert_eq!(codecs, expected_codecs);
	assert_eq!(expected.transports.active.len(), expected.transports.active_count);
	assert_eq!(expected.transports.variants.len(), expected.transports.variant_count);
	let active_source_codecs = expected
		.transports
		.active
		.iter()
		.map(|transport| oracle_codec(transport).to_owned())
		.collect::<BTreeSet<_>>();
	assert_eq!(active_source_codecs.len(), expected.transports.active_count);
	assert!(
		active_source_codecs.is_subset(&codecs),
		"an active frozen transport has no compiled codec"
	);
	assert_eq!(
		expected
			.transports
			.variants
			.iter()
			.filter(|variant| expected.transports.active.contains(variant))
			.count(),
		expected.transports.active_count,
		"active transports must be drawn from the full variant census"
	);

	let urls = compiled
		.routes
		.iter()
		.map(|route| route.endpoint.base_url.as_str().to_owned())
		.collect::<BTreeSet<_>>();
	assert_eq!(compiled.aliases.len(), 96, "frozen alias census");
	assert!(
		compiled
			.aliases
			.windows(2)
			.all(|aliases| aliases[0].alias < aliases[1].alias)
	);
	for alias in &compiled.aliases {
		assert!(
			compiled
				.models
				.iter()
				.any(|model| model.key == alias.target),
			"alias target {}",
			alias.target
		);
		assert!(
			!compiled
				.models
				.iter()
				.any(|model| model.key.as_str() == alias.alias.as_str()),
			"alias duplicates model {}",
			alias.alias
		);
		assert!(!alias.rationale.is_empty(), "alias {} rationale", alias.alias);
		assert!(!alias.provenance.is_empty(), "alias {} provenance", alias.alias);
	}
	let mut expected_active_urls = expected.urls.values.into_iter().collect::<BTreeSet<_>>();
	assert!(
		expected_active_urls.remove("https://omp.sh/"),
		"inactive omp transport URL remains in the source census"
	);
	assert_eq!(urls, expected_active_urls);

	let mut models_by_provider = BTreeMap::<String, usize>::new();
	for model in &compiled.models {
		let first_route = model
			.routes
			.first()
			.expect("every model has at least one route");
		let route = compiled
			.routes
			.iter()
			.find(|route| route.id == *first_route)
			.expect("model route is indexed");
		*models_by_provider
			.entry(route.provider.as_str().to_owned())
			.or_default() += 1;
	}
	assert_eq!(models_by_provider, expected.normalized_catalog.models_by_provider);
	assert_eq!(
		expected
			.raw_catalog
			.rows_by_provider
			.values()
			.sum::<usize>(),
		expected.raw_catalog.row_count
	);
	assert_eq!(expected.raw_catalog.provider_keys.len(), expected.raw_catalog.provider_key_count);
	assert_eq!(expected.normalized_catalog.sort_key, [
		String::from("provider"),
		String::from("model")
	]);
	assert!(!expected.curated_provider_catalog.source.is_empty());
	assert!(!expected.normalized_catalog.source.is_empty());
	assert!(!expected.raw_catalog.source.is_empty());
	assert!(!expected.transports.source.is_empty());
	assert!(!expected.urls.source.is_empty());
	assert_eq!(expected.urls.curated_provider_distinct_count, 90);
	assert_eq!(expected.urls.normalized_model_distinct_count, 90);
	assert_eq!(expected.urls.intersection_count, 72);
}

#[test]
fn every_normalized_logical_model_matches_typed_semantic_oracle_fields() {
	let oracle: NormalizedOracle =
		serde_json::from_str(NORMALIZED_MODELS).expect("normalized model fixture is valid");
	let compiled = compile_frozen_oracle();
	assert_eq!(oracle.schema_version, 1);
	let raw_bytes = zstd::stream::decode_all(MODELS_ZSTD).expect("raw model oracle decompresses");
	let mut inherited_price_components = BTreeMap::<PriceUnit, usize>::new();
	let mut omitted_dynamic_price_models = 0usize;
	let raw: RawPricingOracle =
		serde_json::from_slice(&raw_bytes).expect("raw pricing projection is valid");
	let mut inherited_price_models = 0usize;
	assert_eq!(oracle.models.len(), 4_227);
	assert_eq!(compiled.models.len(), oracle.models.len());
	let actual_by_key = compiled
		.models
		.iter()
		.map(|model| (model.key.as_str(), model))
		.collect::<BTreeMap<_, _>>();
	assert_eq!(actual_by_key.len(), oracle.models.len());
	let mut base_price_mismatches = Vec::new();
	let mut limit_mismatches = Vec::new();

	for expected in &oracle.models {
		let actual = actual_by_key
			.get(expected.id.as_str())
			.unwrap_or_else(|| panic!("missing logical model {}", expected.id));
		assert_eq!(expected.id, format!("{}/{}", expected.provider, expected.model));
		assert_eq!(actual.display_name.as_str(), expected.name, "{} display name", expected.id);
		let expected_family = if expected.family.is_empty() {
			"unknown"
		} else {
			expected.family.as_str()
		};
		assert_eq!(actual.family.as_str(), expected_family, "{} family", expected.id);
		let expected_context =
			(expected.limits.context_window != 0).then_some(expected.limits.context_window);
		let expected_output =
			(expected.limits.max_output_tokens != 0).then_some(expected.limits.max_output_tokens);
		if actual.limits.context_window != expected_context
			|| actual.limits.maximum_output_tokens != expected_output
		{
			limit_mismatches.push(format!(
				"{}: actual context {:?}/output {:?}, expected context {:?}/output {:?}",
				expected.id,
				actual.limits.context_window,
				actual.limits.maximum_output_tokens,
				expected_context,
				expected_output
			));
		}
		assert_eq!(
			actual.availability,
			ModelAvailability::Unspecified,
			"{} availability",
			expected.id
		);
		let behavior: OracleBehaviorCapabilities = serde_json::from_str(expected.behavior.get())
			.expect("typed behavior capability projection");
		let wire_policy = compiled
			.wire_policies
			.iter()
			.find(|policy| policy.content_id() == actual.wire_policy)
			.expect("model wire policy is interned");
		assert_eq!(
			wire_policy.tool.apply_patch.map(|kind| kind.to_string()),
			behavior.apply_patch_tool_type,
			"{} apply-patch wire type",
			expected.id
		);
		assert_eq!(
			wire_policy
				.tool
				.computer_use
				.map(|support| { support == omp_llm_catalog::policy::ComputerUseWireSupport::Native }),
			behavior.supports_computer_use,
			"{} computer-use evidence",
			expected.id
		);
		assert_eq!(
			wire_policy.tool.computer_use_config.map(|support| {
				support == omp_llm_catalog::policy::ComputerUseConfigSupport::Supported
			}),
			behavior.supports_computer_use_config,
			"{} computer-use config evidence",
			expected.id
		);
		assert_eq!(
			wire_policy
				.context
				.max_output_tokens
				.map(|emission| { emission == omp_llm_catalog::policy::MaxOutputTokensEmission::Omit }),
			behavior.omit_max_output_tokens,
			"{} output-token field emission",
			expected.id
		);
		if let Some(chat) = &actual.capabilities.chat {
			match behavior.supports_tools {
				Some(true) => {
					assert!(chat.tools.constraints().is_some(), "{} tool support", expected.id)
				},
				Some(false) => {
					assert!(chat.tools.is_unsupported(), "{} explicit tool rejection", expected.id)
				},
				None => assert!(chat.tools.is_unknown(), "{} absent tool evidence", expected.id),
			}
		}
		assert_eq!(expected.availability, "unspecified");
		assert_eq!(expected.source, "bundled");
		assert!(
			actual.provenance.sources.iter().any(|source| {
				source.kind == ProvenanceKind::Bundled
					&& source.confidence == EvidenceConfidence::Declared
			}),
			"{} bundled provenance",
			expected.id
		);
		assert_eq!(actual.provenance.deprecated, expected.deprecated, "{} deprecation", expected.id);
		assert_eq!(
			actual.provenance.blocked_until_ms.unwrap_or(0),
			expected.blocked_until_ms,
			"{} blocked-until",
			expected.id
		);
		assert_eq!(
			actual.provenance.updated_at_ms.unwrap_or(0),
			expected.updated_at_ms,
			"{} updated-at",
			expected.id
		);
		assert!(!actual.provenance.sources.is_empty(), "{} provenance", expected.id);

		let expected_chat = expected.facets.iter().any(|facet| facet == "chat");
		let expected_embed = expected.facets.iter().any(|facet| facet == "embeddings");
		for operation in all_operations() {
			let expected_operation = match operation {
				OperationKind::Chat => expected_chat,
				OperationKind::Embed => expected_embed,
				_ => false,
			};
			assert_eq!(
				actual.capabilities.operations.contains_kind(operation),
				expected_operation,
				"{} {operation} operation",
				expected.id
			);
		}
		assert_eq!(
			actual
				.capabilities
				.operations
				.contains_kind(OperationKind::Chat),
			expected_chat,
			"{} chat facet",
			expected.id
		);
		assert_eq!(
			actual
				.capabilities
				.operations
				.contains_kind(OperationKind::Embed),
			expected_embed,
			"{} embed facet",
			expected.id
		);
		assert_eq!(expected.facets.len(), usize::from(expected_chat) + usize::from(expected_embed));
		let input_modalities = oracle_modalities(&expected.modalities.inputs);
		assert_eq!(expected.modalities.outputs.as_slice(), [String::from("text")]);
		if expected_chat {
			let chat = actual.capabilities.chat.as_ref().expect("chat constraints");
			assert_eq!(
				chat.input_modalities,
				Availability::Native(input_modalities),
				"{} input modalities",
				expected.id
			);
			assert_eq!(
				chat.reasoning.is_unsupported(),
				!expected.reasoning,
				"{} reasoning",
				expected.id
			);
		}
		if expected_embed {
			let embeddings = actual
				.capabilities
				.embeddings
				.as_ref()
				.expect("embedding constraints");
			assert_eq!(
				embeddings.input_modalities, input_modalities,
				"{} embedding inputs",
				expected.id
			);
			assert_eq!(
				embeddings.input_kinds,
				EmbeddingInputBits::TEXT,
				"{} embedding input kinds",
				expected.id
			);
			if let Some(dimension) = expected.props.get("openai/embedding_dimensions") {
				assert_eq!(
					embeddings
						.dimensions
						.constraints()
						.map(|range| (range.minimum, range.maximum)),
					Some((*dimension, *dimension)),
					"{} dimensions",
					expected.id
				);
			}
		}

		let actual_base_prices = actual
			.pricing
			.components
			.iter()
			.map(|price| (price.unit, price.nanos_usd))
			.collect::<BTreeMap<_, _>>();
		let expected_base_prices = expected
			.pricing
			.iter()
			.map(|price| (price.unit, price.nanos_usd))
			.collect::<BTreeMap<_, _>>();
		if actual_base_prices != expected_base_prices {
			base_price_mismatches.push(format!(
				"{}: actual {actual_base_prices:?}, expected {expected_base_prices:?}",
				expected.id
			));
		}
		assert_eq!(
			actual.pricing.tiers.len(),
			expected.pricing_tiers.len(),
			"{} price tiers",
			expected.id
		);
		for (actual_tier, expected_tier) in actual.pricing.tiers.iter().zip(&expected.pricing_tiers) {
			assert_eq!(
				actual_tier.prompt_tokens_above, expected_tier.prompt_tokens_above,
				"{} tier threshold",
				expected.id
			);
			assert_eq!(
				actual_tier
					.components
					.iter()
					.map(|price| (price.unit, price.nanos_usd))
					.collect::<Vec<_>>(),
				expected_tier
					.pricing
					.iter()
					.map(|price| (price.unit, price.nanos_usd))
					.collect::<Vec<_>>(),
				"{} tier prices",
				expected.id
			);
		}

		let route = compiled
			.routes
			.iter()
			.find(|route| {
				actual.routes.contains(&route.id)
					&& route.provider.as_str() == expected.provider
					&& route.codec.as_str() == oracle_codec(&expected.wire.transport)
					&& if expected.wire.base_url.is_empty() {
						!route.endpoint.base_url.as_str().is_empty()
					} else {
						route.endpoint.base_url.as_str() == expected.wire.base_url
					}
			})
			.unwrap_or_else(|| panic!("{} exact wire route is missing", expected.id));
		assert!(
			actual
				.wire_ids
				.iter()
				.any(|(route_id, _)| route_id == &route.id),
			"{} wire target",
			expected.id
		);
		let expected_wire_model = behavior
			.request_model_id
			.as_deref()
			.unwrap_or(expected.model.as_str());
		assert!(
			actual.wire_ids.iter().any(|(route_id, wire_model)| {
				route_id == &route.id && wire_model.as_str() == expected_wire_model
			}),
			"{} exact wire model",
			expected.id
		);
		assert_eq!(
			wire_policy.context.extended_mode,
			behavior
				.cursor_max_mode
				.map(ExtendedContextMode::from_enabled),
			"{} extended-context mode",
			expected.id
		);
		assert_eq!(
			actual
				.context_promotion_target
				.as_ref()
				.map(|key| key.as_str()),
			behavior.context_promotion_target.as_deref(),
			"{} context promotion",
			expected.id
		);
		assert_eq!(
			actual
				.premium_multiplier_millionths
				.map(|value| value.as_millionths()),
			behavior
				.premium_multiplier
				.map(|value| (value * 1_000_000.0).round() as u64),
			"{} premium multiplier",
			expected.id
		);
		assert_eq!(
			actual
				.thinking_routing
				.reasoning_mode
				.map(|mode| mode.to_string()),
			behavior.reasoning_mode.clone(),
			"{} reasoning mode",
			expected.id
		);
		match (&actual.remote_compaction, &behavior.remote_compaction) {
			(Some(actual), Some(expected_remote)) => {
				assert_eq!(
					actual.enabled,
					Some(expected_remote.enabled),
					"{} compaction enabled",
					expected.id
				);
				assert_eq!(
					actual.transport.as_ref().map(|codec| codec.as_str()),
					Some(oracle_codec(&expected_remote.transport)),
					"{} compaction transport",
					expected.id
				);
				assert_eq!(
					actual.endpoint.as_ref().map(|value| value.as_str()),
					expected_remote.endpoint.as_deref(),
					"{} compaction endpoint",
					expected.id
				);
				assert_eq!(
					actual.v2_streaming_enabled,
					Some(expected_remote.v2_streaming_enabled),
					"{} compaction v2",
					expected.id
				);
				assert_eq!(
					actual.v2_endpoint.as_ref().map(|value| value.as_str()),
					expected_remote.v2_endpoint.as_deref(),
					"{} compaction v2 endpoint",
					expected.id
				);
				assert_eq!(
					actual
						.streaming_endpoint
						.as_ref()
						.map(|value| value.as_str()),
					expected_remote.streaming_endpoint.as_deref(),
					"{} compaction streaming endpoint",
					expected.id
				);
				assert_eq!(
					actual.model.as_ref().map(|value| value.as_str()),
					expected_remote.model.as_deref(),
					"{} compaction model",
					expected.id
				);
			},
			(None, None) => {},
			_ => panic!("{} remote compaction mismatch", expected.id),
		}
		if let Some(lite) = behavior.use_responses_lite {
			assert_eq!(route.use_responses_lite, Some(lite), "{} responses-lite route", expected.id);
		}
		if let Some(prefer_websockets) = behavior.prefer_websockets {
			assert_eq!(
				route.codex_transport == CodexTransportPreference::WebsocketPreferred,
				prefer_websockets,
				"{} websocket preference",
				expected.id
			);
		}
		assert_eq!(route.priority, behavior.priority, "{} route priority", expected.id);
		if !behavior.headers.is_empty() {
			let headers = compiled
				.header_profiles
				.iter()
				.find(|profile| profile.id == route.headers)
				.expect("model header profile is interned");
			assert_eq!(
				headers
					.headers
					.iter()
					.map(|header| (header.name.as_str().to_owned(), header.value.as_str()))
					.collect::<BTreeMap<_, _>>(),
				behavior
					.headers
					.iter()
					.map(|(name, value)| (name.to_ascii_lowercase(), value.as_str()))
					.collect::<BTreeMap<_, _>>(),
				"{} route headers",
				expected.id
			);
		}
		assert!(!expected.behavior.get().is_empty());

		if let Some(raw_model) = raw
			.get(&expected.provider)
			.and_then(|provider| provider.get(&expected.model))
		{
			let direct = raw_prices(&raw_model.cost);
			let resolved = expected
				.pricing
				.iter()
				.map(|price| (price.unit, i128::from(price.nanos_usd)))
				.collect::<BTreeMap<_, _>>();
			if direct != resolved {
				inherited_price_models += 1;
				let sentinel = direct.values().any(|price| *price < 0);
				let expected_origin =
					sentinel.then_some("catalog-oracle:omit:dynamic-pricing-sentinel");
				assert!(
					actual.provenance.sources.iter().any(|source| {
						if source.confidence != EvidenceConfidence::Inferred {
							return false;
						}
						if let Some(expected_origin) = expected_origin {
							return source.origin.as_str() == expected_origin;
						}
						let Some(reference) = source
							.origin
							.as_str()
							.strip_prefix("catalog-oracle:inherit:")
						else {
							return false;
						};
						actual_by_key.contains_key(reference)
					}),
					"{} resolved pricing lacks canonical inheritance provenance",
					expected.id
				);
				for unit in [
					PriceUnit::MtokInput,
					PriceUnit::MtokOutput,
					PriceUnit::MtokCacheRead,
					PriceUnit::MtokCacheWrite,
				] {
					if direct.get(&unit) != resolved.get(&unit) {
						*inherited_price_components.entry(unit).or_default() += 1;
					}
				}
				omitted_dynamic_price_models += usize::from(sentinel);
			}
		}
	}
	assert!(
		base_price_mismatches.is_empty() && limit_mismatches.is_empty(),
		"semantic parity mismatches:\nbase prices:\n{}\nmodel limits:\n{}",
		base_price_mismatches.join("\n"),
		limit_mismatches.join("\n")
	);
	assert_eq!(inherited_price_models, 715, "resolved-price inheritance census");
	assert_eq!(
		inherited_price_components,
		BTreeMap::from([
			(PriceUnit::MtokInput, 458),
			(PriceUnit::MtokOutput, 458),
			(PriceUnit::MtokCacheRead, 575),
			(PriceUnit::MtokCacheWrite, 150),
		]),
		"resolved-price component census"
	);
	assert_eq!(omitted_dynamic_price_models, 2, "dynamic-price sentinel census");
}

#[test]
fn regeneration_is_structurally_and_byte_deterministic() {
	let first = compile_frozen_oracle();
	let second = compile_frozen_oracle();
	assert_eq!(first, second);
	assert_eq!(
		first.normalized_json().expect("first normalized output"),
		second.normalized_json().expect("second normalized output")
	);
}

#[test]
fn price_schedules_limits_and_long_context_tiers_match_exact_integer_oracle_values() {
	let fixture: PriceCases = serde_json::from_str(PRICE_CASES).expect("price fixture is valid");
	let compiled = compile_frozen_oracle();
	assert_eq!(fixture.schema_version, 1);
	assert_eq!(fixture.daybreak.len(), 3);
	assert_eq!(fixture.long_context_tiers.len(), 3);
	assert_eq!(fixture.deepseek_efforts.len(), 6);

	for case in fixture.daybreak {
		let key = format!("openai/{}", case.model);
		let model = compiled
			.models
			.iter()
			.find(|model| model.key.as_str() == key)
			.unwrap_or_else(|| panic!("missing price fixture model {key}"));
		assert_eq!(model.limits.context_window, Some(case.context_window), "{key}");
		assert_eq!(model.limits.maximum_output_tokens, Some(case.max_output_tokens), "{key}");
		assert_eq!(
			compiler_classification(&case.model).version,
			Some(case.openai_version),
			"{key} version"
		);
		assert_eq!(
			model
				.pricing
				.components
				.iter()
				.map(|price| (price.unit, price.nanos_usd))
				.collect::<Vec<_>>(),
			case
				.pricing
				.iter()
				.map(|price| (price.unit, price.nanos_usd))
				.collect::<Vec<_>>(),
			"{key} base prices"
		);
		assert_eq!(model.pricing.tiers.len(), case.pricing_tiers.len(), "{key} tiers");
	}

	for case in fixture.long_context_tiers {
		let key = format!("openai/{}", case.model);
		let model = compiled
			.models
			.iter()
			.find(|model| model.key.as_str() == key)
			.unwrap_or_else(|| panic!("missing tier fixture model {key}"));
		for tier in case.pricing_tiers {
			let actual = model
				.pricing
				.tiers
				.iter()
				.find(|candidate| candidate.prompt_tokens_above == tier.prompt_tokens_above)
				.unwrap_or_else(|| {
					panic!("missing {} threshold {}", case.model, tier.prompt_tokens_above)
				});
			assert_eq!(
				actual
					.components
					.iter()
					.map(|price| (price.unit, price.nanos_usd))
					.collect::<Vec<_>>(),
				tier
					.pricing
					.iter()
					.map(|price| (price.unit, price.nanos_usd))
					.collect::<Vec<_>>()
			);
			let at_threshold = model
				.pricing
				.cost(UsageDimensions {
					input_tokens: tier.prompt_tokens_above,
					..UsageDimensions::default()
				})
				.expect("threshold cost");
			let above_threshold = model
				.pricing
				.cost(UsageDimensions {
					input_tokens: tier.prompt_tokens_above + 1,
					..UsageDimensions::default()
				})
				.expect("tier cost");
			assert_ne!(at_threshold, above_threshold, "{} tier boundary", case.model);
		}
	}
	for case in fixture.deepseek_efforts {
		let key = format!("ollama-cloud/{}", case.model);
		let model = compiled
			.models
			.iter()
			.find(|model| model.key.as_str() == key)
			.unwrap_or_else(|| panic!("missing DeepSeek effort fixture model {key}"));
		let policy = model
			.thinking
			.as_ref()
			.and_then(|id| {
				compiled
					.thinking_policies
					.iter()
					.find(|policy| policy.content_id() == *id)
			})
			.unwrap_or_else(|| panic!("{key} has no interned thinking policy"));
		assert_eq!(
			policy.efforts.as_slice(),
			case
				.efforts
				.iter()
				.copied()
				.map(Into::into)
				.collect::<Vec<_>>(),
			"{key} efforts"
		);
	}
}

#[test]
fn exact_override_rows_and_qwen_collapses_remain_present_and_auditable() {
	let exact: ExactOverrides =
		serde_json::from_str(EXACT_OVERRIDES).expect("exact override fixture is valid");
	let qwen: QwenCases =
		serde_json::from_str(QWEN_COLLAPSE).expect("Qwen collapse fixture is valid");
	let compiled = compile_frozen_oracle();
	assert_eq!(exact.schema_version, 1);
	assert_eq!(exact.cases.len(), 9);
	assert!(!exact.source_assertions.is_empty());
	for case in exact.cases {
		let model = compiled
			.models
			.iter()
			.find(|model| model.key.as_str() == case.model)
			.unwrap_or_else(|| panic!("missing exact override {}", case.model));
		assert!(!case.rationale.is_empty(), "{} lacks rationale", case.model);
		assert!(!case.provenance.is_empty(), "{} lacks provenance", case.model);
		let expected_thinking = case
			.expected
			.thinking
			.as_ref()
			.expect("exact thinking behavior");
		let actual_thinking = model
			.thinking
			.as_ref()
			.and_then(|id| {
				compiled
					.thinking_policies
					.iter()
					.find(|policy| policy.content_id() == *id)
			})
			.unwrap_or_else(|| panic!("{} exact thinking policy", case.model));
		assert_eq!(
			actual_thinking.mode,
			ThinkingMode::from_str(&expected_thinking.mode).expect("known exact thinking mode"),
			"{} thinking mode",
			case.model
		);
		assert_eq!(
			actual_thinking.efforts.as_slice(),
			expected_thinking
				.efforts
				.iter()
				.copied()
				.map(Into::into)
				.collect::<Vec<_>>(),
			"{} thinking efforts",
			case.model
		);
		assert_eq!(
			actual_thinking.effort_budgets,
			expected_thinking
				.effort_budgets
				.iter()
				.map(|(effort, budget)| ((*effort).into(), *budget))
				.collect(),
			"{} thinking budgets",
			case.model
		);
		assert_eq!(
			actual_thinking.suppress_when_off, expected_thinking.suppress_when_off,
			"{} thinking-off suppression",
			case.model
		);
		assert_eq!(
			model
				.thinking_routing
				.effort_routing
				.iter()
				.map(|(effort, route)| (*effort, route.as_str()))
				.collect::<BTreeMap<_, _>>(),
			expected_thinking
				.effort_routing
				.iter()
				.map(|(effort, route)| ((*effort).into(), route.as_str()))
				.collect(),
			"{} thinking effort routing",
			case.model
		);
		if let Some(target) = &case.expected.context_promotion_target {
			assert_eq!(
				model
					.context_promotion_target
					.as_ref()
					.map(|key| key.as_str()),
				Some(target.as_str()),
				"{} context promotion target",
				case.model
			);
		}
		if let Some(multiplier) = case.expected.premium_multiplier {
			assert_eq!(
				model
					.premium_multiplier_millionths
					.map(|value| value.as_millionths()),
				Some((multiplier * 1_000_000.0).round() as u64),
				"{} premium multiplier",
				case.model
			);
		}
		if let Some(reasoning_mode) = &case.expected.reasoning_mode {
			assert_eq!(
				model
					.thinking_routing
					.reasoning_mode
					.map(|mode| mode.to_string()),
				Some(reasoning_mode.clone()),
				"{} reasoning mode",
				case.model
			);
		}
		if let Some(expected) = &case.expected.remote_compaction {
			let actual = model
				.remote_compaction
				.as_ref()
				.expect("exact remote compaction");
			assert_eq!(
				actual.enabled,
				Some(expected.enabled),
				"{} remote compaction enabled",
				case.model
			);
			assert_eq!(
				actual
					.transport
					.as_ref()
					.map(|transport| transport.as_str()),
				Some(oracle_codec(&expected.transport)),
				"{} remote compaction codec",
				case.model
			);
			assert_eq!(
				actual.v2_streaming_enabled,
				Some(expected.v2_streaming_enabled),
				"{} remote compaction v2 streaming",
				case.model
			);
		}
		if let Some(request_model) = &case.expected.request_model_id {
			assert!(
				model
					.wire_ids
					.iter()
					.any(|(_, wire_model)| wire_model.as_str() == request_model),
				"{} request model id",
				case.model
			);
		}
		if let Some(supports_tools) = case.expected.supports_tools {
			let chat = model
				.capabilities
				.chat
				.as_ref()
				.expect("exact chat capabilities");
			assert_eq!(
				chat.tools.constraints().is_some(),
				supports_tools,
				"{} tool support",
				case.model
			);
		}

		let routes = model
			.routes
			.iter()
			.map(|route_id| {
				compiled
					.routes
					.iter()
					.find(|route| route.id == *route_id)
					.expect("exact model route")
			})
			.collect::<Vec<_>>();
		if let Some(prefer_websockets) = case.expected.prefer_websockets {
			assert!(
				routes.iter().any(|route| {
					(route.codex_transport
						== omp_llm_catalog::provider::CodexTransportPreference::WebsocketPreferred)
						== prefer_websockets
				}),
				"{} websocket preference",
				case.model
			);
		}
		if let Some(use_responses_lite) = case.expected.use_responses_lite {
			assert!(
				routes
					.iter()
					.any(|route| route.use_responses_lite == Some(use_responses_lite)),
				"{} responses-lite",
				case.model
			);
		}
		if let Some(priority) = case.expected.priority {
			assert!(
				routes.iter().any(|route| route.priority == Some(priority)),
				"{} route priority",
				case.model
			);
		}
		if !case.expected.headers.is_empty() {
			let expected_headers = case
				.expected
				.headers
				.iter()
				.map(|(name, value)| (name.to_ascii_lowercase(), value.as_str()))
				.collect::<BTreeMap<_, _>>();
			assert!(
				routes.iter().any(|route| {
					compiled
						.header_profiles
						.iter()
						.find(|profile| profile.id == route.headers)
						.is_some_and(|profile| {
							profile
								.headers
								.iter()
								.map(|header| (header.name.as_str().to_owned(), header.value.as_str()))
								.collect::<BTreeMap<_, _>>()
								== expected_headers
						})
				}),
				"{} exact headers",
				case.model
			);
		}
	}

	assert_eq!(qwen.schema_version, 1);
	assert_eq!(qwen.cases.len(), 2);
	for case in qwen.cases {
		assert_eq!(case.inputs.len(), 2);
		assert!(!case.rationale.is_empty());
		let logical = case.expected_logical.model.as_str();
		assert_eq!(case.expected_logical.efforts.len(), 4);
		assert_eq!(case.expected_logical.effort_routing.len(), 5);
		assert_eq!(case.expected_logical.thinking.efforts.len(), 4);
		assert_eq!(case.expected_logical.thinking.effort_routing.len(), 5);
		assert!(case.expected_logical.thinking.requires_effort);
		assert_eq!(case.expected_logical.thinking.mode, "effort");
		let key = format!("{}/{}", case.provider, logical);
		let model = compiled
			.models
			.iter()
			.find(|model| model.key.as_str() == key)
			.unwrap_or_else(|| panic!("missing collapsed {key}"));
		let thinking = model
			.thinking
			.as_ref()
			.and_then(|id| {
				compiled
					.thinking_policies
					.iter()
					.find(|policy| policy.content_id() == *id)
			})
			.unwrap_or_else(|| panic!("{key} thinking policy is missing"));
		assert_eq!(
			thinking.mode,
			ThinkingMode::from_str(&case.expected_logical.thinking.mode)
				.expect("known Qwen thinking mode"),
			"{key} thinking mode"
		);
		assert_eq!(
			thinking.efforts.as_slice(),
			case
				.expected_logical
				.thinking
				.efforts
				.iter()
				.copied()
				.map(Into::into)
				.collect::<Vec<_>>(),
			"{key} thinking efforts"
		);
		assert_eq!(thinking.requires_effort, Some(true), "{key} required effort");
		assert_eq!(
			model
				.thinking_routing
				.effort_routing
				.iter()
				.map(|(effort, wire)| (*effort, wire.as_str()))
				.collect::<BTreeMap<_, _>>(),
			case
				.expected_logical
				.effort_routing
				.iter()
				.map(|(effort, wire)| ((*effort).into(), wire.as_str()))
				.collect(),
			"{key} effort routing"
		);
		assert_eq!(
			model
				.wire_ids
				.iter()
				.map(|(_, wire)| wire.as_str())
				.collect::<BTreeSet<_>>(),
			case.inputs.iter().map(String::as_str).collect(),
			"{key} collapsed wire inputs"
		);
		let absent = format!("{}/{}", case.provider, case.absent_after_collapse);
		assert!(
			!compiled
				.models
				.iter()
				.any(|model| model.key.as_str() == absent),
			"uncollapsed sibling {absent}"
		);
		let alias = compiled
			.aliases
			.iter()
			.find(|alias| alias.alias.as_str() == absent)
			.unwrap_or_else(|| panic!("collapsed sibling alias {absent} is missing"));
		assert_eq!(alias.target.as_str(), key, "{absent} alias target");
	}
}

#[test]
fn embedded_snapshot_and_all_indexes_match_a_fresh_deterministic_encoding() {
	let compiled = compile_frozen_oracle();
	let embedded = Catalog::embedded();
	assert_eq!(embedded.census(), compiled.census);
	assert_eq!(embedded.revision(), &compiled.revision);
	assert_eq!(embedded.providers(), &*compiled.providers);
	assert_eq!(embedded.routes(), &*compiled.routes);
	assert_eq!(embedded.models(), &*compiled.models);
	assert_eq!(embedded.auth_specs(), &*compiled.auth_specs);
	assert_eq!(embedded.oauth_specs(), &*compiled.oauth_specs);
	assert_eq!(embedded.header_profiles(), &*compiled.header_profiles);
	assert_eq!(embedded.discovery_specs(), &*compiled.discovery_specs);
	assert_eq!(embedded.aliases(), &*compiled.aliases);

	let regenerated = Catalog::encode(compiled.clone(), SnapshotProvenance {
		source_digest: *embedded.source_digest(),
	})
	.expect("fresh snapshot encoding");
	assert_eq!(regenerated.postcard, CATALOG_POSTCARD);
	assert_eq!(
		regenerated.normalized_json,
		compiled.normalized_json().expect("fresh normalized JSON")
	);

	for auth in embedded.auth_specs() {
		assert_eq!(embedded.auth_spec(&auth.id), Some(auth));
		if let Some(oauth) = &auth.oauth {
			assert!(embedded.oauth_spec(oauth).is_some());
		}
	}
	for oauth in embedded.oauth_specs() {
		assert_eq!(embedded.oauth_spec(&oauth.id), Some(oauth));
	}
	for headers in embedded.header_profiles() {
		assert_eq!(embedded.header_profile(&headers.id), Some(headers));
	}
	for discovery in embedded.discovery_specs() {
		assert_eq!(embedded.discovery_spec(&discovery.id), Some(discovery));
	}
	for policy in &compiled.wire_policies {
		assert_eq!(embedded.wire_policy(&policy.content_id()), Some(policy));
	}
	for policy in &compiled.thinking_policies {
		assert_eq!(embedded.thinking_policy(&policy.content_id()), Some(policy));
	}
	for provider in embedded.providers() {
		assert_eq!(embedded.provider(&provider.id), Some(provider));
	}
	for route in embedded.routes() {
		assert_eq!(embedded.route(&route.id), Some(route));
		assert!(embedded.auth_spec(&route.auth).is_some());
		assert!(embedded.header_profile(&route.headers).is_some());
		if let Some(discovery) = &route.discovery {
			assert!(embedded.discovery_spec(discovery).is_some());
		}
	}
	for model in embedded.models() {
		assert_eq!(embedded.model(&model.key), Some(model));
		assert!(embedded.wire_policy(&model.wire_policy).is_some());
		if let Some(thinking) = &model.thinking {
			assert!(embedded.thinking_policy(thinking).is_some());
		}
		let provider = model
			.routes
			.first()
			.and_then(|route| embedded.route(route))
			.map(|route| &route.provider)
			.expect("model route has a provider");
		assert_eq!(embedded.model_for_provider(provider, &model.key), Some(model));
	}
	for alias in embedded.aliases() {
		assert_eq!(embedded.resolve_alias(alias.alias.as_str()), embedded.model(&alias.target));
	}
}

#[test]
fn snapshot_corruption_and_source_mismatch_fail_loudly() {
	assert!(Catalog::decode(&[]).is_err());
	let mut corrupted = CATALOG_POSTCARD.to_vec();
	*corrupted.last_mut().expect("snapshot is nonempty") ^= 0x01;
	assert!(Catalog::decode(&corrupted).is_err());
	assert!(Catalog::decode_for_source(CATALOG_POSTCARD, [0xa5; 32]).is_err());
}

#[test]
fn every_thinking_profile_is_interned_and_attached_to_its_exact_model_set() {
	let fixture: ThinkingProfiles =
		serde_json::from_str(THINKING_PROFILES).expect("thinking profile fixture is valid");
	let compiled = compile_frozen_oracle();
	assert_eq!(fixture.schema_version, 1);
	assert_eq!(fixture.profile_count, 43);
	assert_eq!(fixture.profiles.len(), fixture.profile_count);
	assert!(!fixture.normalization.is_empty());

	let mut fixture_labels = BTreeSet::new();
	let mut expected_ids = BTreeSet::new();
	let mut expected_by_model = BTreeMap::new();
	for profile in fixture.profiles {
		assert!(
			fixture_labels.insert(profile.profile_id.clone()),
			"duplicate fixture profile {}",
			profile.profile_id
		);
		assert_eq!(profile.models.len(), profile.model_count, "{}", profile.profile_id);
		profile
			.shape
			.validate()
			.expect("fixture thinking policy is structurally valid");
		let expected_id = profile.shape.content_id();
		assert!(
			expected_ids.insert(expected_id.clone()),
			"{} is not structurally distinct",
			profile.profile_id
		);
		for key in profile.models {
			assert!(
				expected_by_model
					.insert(key.clone(), expected_id.clone())
					.is_none(),
				"{key} appears in more than one thinking profile"
			);
			let model = compiled
				.models
				.iter()
				.find(|model| model.key.as_str() == key)
				.unwrap_or_else(|| panic!("thinking profile references missing model {key}"));
			let actual_policy = model
				.thinking
				.as_ref()
				.and_then(|id| {
					compiled
						.thinking_policies
						.iter()
						.find(|policy| policy.content_id() == *id)
				})
				.unwrap_or_else(|| panic!("{key} thinking policy is not interned"));
			assert_eq!(actual_policy, &profile.shape, "{key} thinking policy shape");
			assert_eq!(model.thinking.as_ref(), Some(&expected_id), "{key} thinking policy");
		}
	}
	assert_eq!(expected_ids.len(), 43);
	let actual_ids = compiled
		.thinking_policies
		.iter()
		.map(ThinkingPolicy::content_id)
		.collect::<BTreeSet<_>>();
	assert_eq!(
		actual_ids.difference(&expected_ids).collect::<Vec<_>>(),
		Vec::<&omp_llm_catalog::ThinkingPolicyId>::new(),
		"unexpected compiled thinking policies"
	);
	assert_eq!(
		expected_ids.difference(&actual_ids).collect::<Vec<_>>(),
		Vec::<&omp_llm_catalog::ThinkingPolicyId>::new(),
		"missing compiled thinking policies"
	);
	for model in &compiled.models {
		assert_eq!(
			model.thinking.as_ref(),
			expected_by_model.get(model.key.as_str()),
			"{} exact thinking policy",
			model.key
		);
	}
}

#[test]
fn every_sparse_wire_profile_has_a_stable_distinct_content_id() {
	let fixture: CompatProfiles =
		serde_json::from_str(COMPAT_PROFILES).expect("wire profile fixture is valid");
	let compiled = compile_frozen_oracle();
	let normalized: NormalizedOracle =
		serde_json::from_str(NORMALIZED_MODELS).expect("normalized model fixture is valid");
	let behavior_by_model = normalized
		.models
		.into_iter()
		.map(|model| {
			let behavior = serde_json::from_str(model.behavior.get())
				.expect("typed behavior capability projection");
			(model.id, behavior)
		})
		.collect::<BTreeMap<_, OracleBehaviorCapabilities>>();
	assert_eq!(fixture.schema_version, 1);
	assert_eq!(fixture.profile_count, 35);
	assert_eq!(fixture.profiles.len(), fixture.profile_count);
	assert!(!fixture.normalization.is_empty());

	let mut fixture_labels = BTreeSet::new();
	let mut expected_ids = BTreeSet::new();
	let mut expected_by_model = BTreeMap::new();
	for profile in fixture.profiles {
		assert!(
			fixture_labels.insert(profile.profile_id.clone()),
			"duplicate fixture profile {}",
			profile.profile_id
		);
		assert_eq!(profile.models.len(), profile.model_count, "{}", profile.profile_id);
		let policy = profile.shape.into_policy();
		let expected_id = policy.content_id();
		assert!(
			expected_ids.insert(expected_id.clone()),
			"{} is not structurally distinct",
			profile.profile_id
		);
		assert!(
			compiled
				.wire_policies
				.iter()
				.any(|candidate| candidate.content_id() == expected_id),
			"{} wire policy is missing",
			profile.profile_id
		);
		for key in profile.models {
			let behavior = behavior_by_model
				.get(&key)
				.unwrap_or_else(|| panic!("{key} behavior fixture is missing"));
			let expected_policy = with_model_behavior(policy.clone(), behavior);
			let attached_id = expected_policy.content_id();
			assert!(
				expected_by_model
					.insert(key.clone(), attached_id.clone())
					.is_none(),
				"{key} appears in more than one wire profile"
			);
			let model = compiled
				.models
				.iter()
				.find(|model| model.key.as_str() == key)
				.unwrap_or_else(|| panic!("wire profile {} missing model {key}", profile.profile_id));
			let actual_policy = compiled
				.wire_policies
				.iter()
				.find(|candidate| candidate.content_id() == model.wire_policy)
				.unwrap_or_else(|| panic!("{key} wire policy is not interned"));
			assert_eq!(actual_policy, &expected_policy, "{key} wire compatibility shape");
			assert_eq!(model.wire_policy, attached_id, "{key} wire compatibility ID");
		}
	}
	assert_eq!(expected_ids.len(), 35);
	assert_eq!(expected_by_model.len(), 312);
	let baseline = WirePolicy::baseline();
	for model in &compiled.models {
		let expected = if let Some(expected) = expected_by_model.get(model.key.as_str()) {
			expected.clone()
		} else {
			let behavior = behavior_by_model
				.get(model.key.as_str())
				.unwrap_or_else(|| panic!("{} behavior fixture is missing", model.key));
			let expected_policy = with_model_behavior(baseline.clone(), behavior);
			let actual_policy = compiled
				.wire_policies
				.iter()
				.find(|candidate| candidate.content_id() == model.wire_policy)
				.unwrap_or_else(|| panic!("{} wire policy is not interned", model.key));
			assert_eq!(
				actual_policy, &expected_policy,
				"{} baseline wire compatibility shape",
				model.key
			);
			expected_policy.content_id()
		};
		assert_eq!(model.wire_policy, expected, "{} exact wire compatibility policy", model.key);
	}
}

#[test]
fn catalog_references_and_advertised_capabilities_are_internally_complete() {
	let compiled = compile_frozen_oracle();
	for provider in &compiled.providers {
		for route_id in &provider.routes {
			let route = compiled
				.routes
				.iter()
				.find(|route| route.id == *route_id)
				.expect("provider route exists");
			assert_eq!(route.provider, provider.id, "route owner for {}", route.id);
		}
		assert!(!provider.name.as_str().is_empty(), "{} has no display name", provider.id);
		assert!(!provider.auth.is_empty(), "{} has no authentication contract", provider.id);
		assert!(
			compiled
				.wire_policies
				.iter()
				.any(|policy| policy.content_id() == provider.wire_policy),
			"{} provider wire policy is missing",
			provider.id
		);
		for auth_id in &provider.auth {
			assert!(
				compiled.auth_specs.iter().any(|auth| auth.id == *auth_id),
				"{} references missing auth {auth_id}",
				provider.id
			);
		}
		for operation in all_operations() {
			if provider.management.operations.contains_kind(operation) {
				assert!(
					matches!(
						operation,
						OperationKind::Usage | OperationKind::DiscoverModels | OperationKind::Auth
					),
					"{} exposes model operation {operation} as management",
					provider.id
				);
			}
		}
		if provider.management.refresh {
			assert!(
				provider.auth.iter().any(|auth_id| {
					compiled
						.auth_specs
						.iter()
						.find(|auth| auth.id == *auth_id)
						.and_then(|auth| auth.oauth.as_ref())
						.and_then(|oauth_id| {
							compiled
								.oauth_specs
								.iter()
								.find(|oauth| oauth.id == *oauth_id)
						})
						.is_some_and(|oauth| oauth.refresh != OAuthRefreshBehavior::Unsupported)
				}),
				"{} advertises refresh without a refreshable credential flow",
				provider.id
			);
		}
	}

	for route in &compiled.routes {
		assert!(
			compiled
				.providers
				.iter()
				.any(|provider| provider.id == route.provider)
		);
		assert!(compiled.auth_specs.iter().any(|auth| auth.id == route.auth));
		assert!(
			compiled
				.header_profiles
				.iter()
				.any(|headers| headers.id == route.headers)
		);
		if let Some(discovery) = &route.discovery {
			assert!(
				compiled
					.discovery_specs
					.iter()
					.any(|spec| spec.id == *discovery)
			);
		}
		assert!(
			compiled
				.providers
				.iter()
				.find(|provider| provider.id == route.provider)
				.is_some_and(|provider| provider.auth.contains(&route.auth)),
			"{} auth is not owned by {}",
			route.id,
			route.provider
		);
		assert!(!route.endpoint.base_url.as_str().is_empty(), "{} has an empty endpoint", route.id);
		assert!(
			!route.trust_domain.origin.as_str().is_empty(),
			"{} has an empty trust origin",
			route.id
		);
	}

	for auth in &compiled.auth_specs {
		assert_eq!(auth.kind == AuthSpecKind::Oauth, auth.oauth.is_some(), "{} OAuth link", auth.id);
		if let Some(oauth_id) = &auth.oauth {
			assert!(
				compiled
					.oauth_specs
					.iter()
					.any(|oauth| oauth.id == *oauth_id),
				"{} missing OAuth flow {oauth_id}",
				auth.id
			);
		}
		let placement_count = usize::from(auth.header_name.is_some())
			+ usize::from(auth.query_parameter.is_some())
			+ usize::from(auth.sealed_body.is_some());
		assert!(placement_count <= 1, "{} has conflicting credential placements", auth.id);
	}

	for model in &compiled.models {
		assert!(!model.routes.is_empty(), "{} has no route", model.key);
		assert!(!model.wire_ids.is_empty(), "{} has no wire target", model.key);
		for route_id in &model.routes {
			assert!(
				compiled.routes.iter().any(|route| route.id == *route_id),
				"{} has missing route {route_id}",
				model.key
			);
			assert!(
				model
					.wire_ids
					.iter()
					.any(|(wire_route, _)| wire_route == route_id),
				"{} has no wire id for {route_id}",
				model.key
			);
		}
		for (wire_route, wire_model) in &model.wire_ids {
			assert!(
				model.routes.contains(wire_route),
				"{} wire route {wire_route} is ineligible",
				model.key
			);
			assert!(!wire_model.as_str().is_empty(), "{} has an empty wire model", model.key);
		}
		assert!(
			compiled
				.wire_policies
				.iter()
				.any(|policy| policy.content_id() == model.wire_policy)
		);
		if let Some(thinking) = &model.thinking {
			assert!(
				compiled
					.thinking_policies
					.iter()
					.any(|policy| policy.content_id() == *thinking)
			);
		}
		let capabilities = &model.capabilities;
		assert_eq!(
			capabilities.operations.contains_kind(OperationKind::Chat),
			capabilities.chat.is_some(),
			"{} chat capability",
			model.key
		);
		assert_eq!(
			capabilities.operations.contains_kind(OperationKind::Embed),
			capabilities.embeddings.is_some(),
			"{} embedding capability",
			model.key
		);
		assert_eq!(
			capabilities
				.operations
				.contains_kind(OperationKind::GenerateImage),
			capabilities.image.is_some(),
			"{} image capability",
			model.key
		);
		assert_eq!(
			capabilities
				.operations
				.contains_kind(OperationKind::GenerateVideo),
			capabilities.video.is_some(),
			"{} video capability",
			model.key
		);
		assert_eq!(
			capabilities.operations.contains_kind(OperationKind::Speak),
			capabilities.speech.is_some(),
			"{} speech capability",
			model.key
		);
		assert_eq!(
			capabilities
				.operations
				.contains_kind(OperationKind::Transcribe),
			capabilities.transcription.is_some(),
			"{} transcription capability",
			model.key
		);
		assert_eq!(
			capabilities
				.operations
				.contains_kind(OperationKind::Realtime),
			capabilities.realtime.is_some(),
			"{} realtime capability",
			model.key
		);
		assert_eq!(
			capabilities.operations.contains_kind(OperationKind::Search),
			capabilities.search.is_some(),
			"{} search capability",
			model.key
		);
		for operation in
			[OperationKind::CountTokens, OperationKind::Tokenize, OperationKind::Detokenize]
		{
			if capabilities.operations.contains_kind(operation) {
				assert!(
					capabilities.tokenization.is_some(),
					"{} advertises {operation} without constraints",
					model.key
				);
			}
		}

		for operation in all_operations() {
			if !capabilities.operations.contains_kind(operation) {
				continue;
			}
			assert!(
				model.routes.iter().any(|route_id| {
					compiled
						.routes
						.iter()
						.find(|route| route.id == *route_id)
						.is_some_and(|route| {
							route
								.capability_limits
								.operations
								.is_none_or(|allowed| allowed.contains_kind(operation))
						})
				}),
				"{} advertises {operation} without an eligible route",
				model.key
			);
		}
	}
}
