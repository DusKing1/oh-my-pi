//! Offline model identity classification for catalog compilation and discovery
//! normalization.

use omp_core::Str;
use serde::{Deserialize, Serialize};

use crate::id::FamilyId;

/// Source phase allowed to invoke identity classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassificationPhase {
	/// Checked-in source compilation.
	CatalogCompiler,
	/// Provider model-list normalization.
	DiscoveryNormalizer,
}

/// Ordered reasoning effort suffix recognized by the catalog compiler.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffortTier {
	/// Explicit non-reasoning route.
	Off,
	/// Minimal reasoning.
	Minimal,
	/// Low reasoning.
	Low,
	/// Medium reasoning.
	Medium,
	/// High reasoning.
	High,
	/// Extra-high reasoning.
	XHigh,
	/// Provider-defined maximum reasoning.
	Max,
}

/// Three-component model generation parsed from an identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ModelVersion {
	/// Major generation.
	pub major: u16,
	/// Minor generation.
	pub minor: u16,
	/// Patch generation.
	pub patch: u16,
}

/// Why a classification fact is present.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassificationMethod {
	/// An exact reviewed override supplied the result.
	ExactOverride,
	/// A bounded family rule supplied the result.
	FamilyRule,
	/// A structural suffix rule supplied the result.
	StructuralSuffix,
	/// No rule established the fact.
	Unknown,
}

/// Auditable evidence attached to compiler-produced identity facts.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ClassificationEvidence {
	/// Classification mechanism.
	pub method:        ClassificationMethod,
	/// Stable rule or override identifier.
	pub rule:          Str,
	/// Human-readable review rationale.
	pub rationale:     Str,
	/// Source path, document, or provider declaration supporting the fact.
	pub provenance:    Str,
	/// Optional Unix-millisecond expiry for temporary evidence.
	pub expires_at_ms: Option<u64>,
}

/// Borrowed input accepted only by compiler and discovery normalization code.
#[derive(Clone, Copy, Debug)]
pub struct ClassificationInput<'a> {
	/// Compiler or discovery phase invoking the classifier.
	pub phase:          ClassificationPhase,
	/// Provider source key, before provider alias resolution.
	pub provider:       &'a str,
	/// Opaque provider model identifier.
	pub model:          &'a str,
	/// Observation time used to reject expired overrides.
	pub observed_at_ms: Option<u64>,
}

/// Compiler-normalized identity and its evidence.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelClassification {
	/// Logical model identifier after effort sibling collapse.
	pub logical_model:    Str,
	/// Centrally classified model family; `unknown` is conservative.
	pub family:           FamilyId,
	/// Parsed family generation when the rule establishes one.
	pub version:          Option<ModelVersion>,
	/// Effort route represented by this source row.
	pub effort:           Option<EffortTier>,
	/// Whether this row is the reasoning sibling of an explicit off route.
	pub thinking_variant: bool,
	/// Evidence for family and identity normalization.
	pub evidence:         ClassificationEvidence,
}

/// Reviewed exact identity correction applied before general family rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactOverride {
	/// Stable override identifier.
	pub id:            &'static str,
	/// Optional exact provider source key.
	pub provider:      Option<&'static str>,
	/// Exact case-insensitive wire model identifier.
	pub model:         &'static str,
	/// Logical model identifier.
	pub logical_model: &'static str,
	/// Classified family.
	pub family:        &'static str,
	/// Optional pinned generation.
	pub version:       Option<ModelVersion>,
	/// Optional effort route.
	pub effort:        Option<EffortTier>,
	/// Review rationale.
	pub rationale:     &'static str,
	/// Evidence provenance.
	pub provenance:    &'static str,
	/// Optional Unix-millisecond expiry.
	pub expires_at_ms: Option<u64>,
}

const fn exact_family(
	id: &'static str,
	provider: &'static str,
	model: &'static str,
	logical_model: &'static str,
	family: &'static str,
	rationale: &'static str,
) -> ExactOverride {
	ExactOverride {
		id,
		provider: Some(provider),
		model,
		logical_model,
		family,
		version: None,
		effort: None,
		rationale,
		provenance: "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	}
}

const EXACT_OVERRIDES: &[ExactOverride] = &[
	ExactOverride {
		id:            "openai-daybreak-blue-2026-08",
		provider:      None,
		model:         "daybreak-blue-latest",
		logical_model: "daybreak-blue-latest",
		family:        "unknown",
		version:       Some(ModelVersion { major: 5, minor: 6, patch: 0 }),
		effort:        None,
		rationale:     "OpenAI rolling alias pins only the documented generation; its opaque \
		                product name supplies no family evidence",
		provenance:    "catalog-oracle:identity/daybreak-aliases",
		expires_at_ms: Some(1_799_712_000_000),
	},
	ExactOverride {
		id:            "openai-daybreak-red-2026-08",
		provider:      None,
		model:         "daybreak-red-latest",
		logical_model: "daybreak-red-latest",
		family:        "unknown",
		version:       Some(ModelVersion { major: 5, minor: 6, patch: 0 }),
		effort:        None,
		rationale:     "OpenAI rolling alias pins only the documented generation; its opaque \
		                product name supplies no family evidence",
		provenance:    "catalog-oracle:identity/daybreak-aliases",
		expires_at_ms: Some(1_799_712_000_000),
	},
	ExactOverride {
		id:            "openai-gpt-daybreak-blue-2026-08",
		provider:      None,
		model:         "gpt-daybreak-blue-latest",
		logical_model: "gpt-daybreak-blue-latest",
		family:        "openai",
		version:       Some(ModelVersion { major: 5, minor: 6, patch: 0 }),
		effort:        None,
		rationale:     "OpenAI rolling alias pinned to the generation declared by the source \
		                snapshot",
		provenance:    "catalog-oracle:identity/daybreak-aliases",
		expires_at_ms: Some(1_799_712_000_000),
	},
	ExactOverride {
		id:            "openai-gpt-daybreak-red-2026-08",
		provider:      None,
		model:         "gpt-daybreak-red-latest",
		logical_model: "gpt-daybreak-red-latest",
		family:        "openai",
		version:       Some(ModelVersion { major: 5, minor: 6, patch: 0 }),
		effort:        None,
		rationale:     "OpenAI rolling alias pinned to the generation declared by the source \
		                snapshot",
		provenance:    "catalog-oracle:identity/daybreak-aliases",
		expires_at_ms: Some(1_799_712_000_000),
	},
	ExactOverride {
		id:            "kilo-kimi-k2-thinking-product",
		provider:      Some("kilo"),
		model:         "kimi-k2-thinking",
		logical_model: "moonshotai/kimi-k2-thinking",
		family:        "kimi",
		version:       None,
		effort:        None,
		rationale:     "Oracle exposes this independently priced product rather than an effort route",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	ExactOverride {
		id:            "openrouter-kimi-k2-thinking-product",
		provider:      Some("openrouter"),
		model:         "kimi-k2-thinking",
		logical_model: "moonshotai/kimi-k2-thinking",
		family:        "kimi",
		version:       None,
		effort:        None,
		rationale:     "Oracle exposes this independently priced product rather than an effort route",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	ExactOverride {
		id:            "vercel-kimi-k2-thinking-product",
		provider:      Some("vercel-ai-gateway"),
		model:         "kimi-k2-thinking",
		logical_model: "moonshotai/kimi-k2-thinking",
		family:        "kimi",
		version:       None,
		effort:        None,
		rationale:     "Oracle exposes this independently priced product rather than an effort route",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	ExactOverride {
		id:            "vercel-deepseek-v3-2-thinking-product",
		provider:      Some("vercel-ai-gateway"),
		model:         "deepseek-v3.2-thinking",
		logical_model: "deepseek/deepseek-v3.2-thinking",
		family:        "deepseek",
		version:       None,
		effort:        None,
		rationale:     "Oracle exposes this independently priced product rather than an effort route",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	ExactOverride {
		id:            "umans-deepseek-v4-flash-family",
		provider:      Some("umans"),
		model:         "umans-deepseek-v4-flash-0731",
		logical_model: "umans-deepseek-v4-flash-0731",
		family:        "deepseek",
		version:       None,
		effort:        None,
		rationale:     "The opaque Umans product identifier is backed by the reviewed DeepSeek V4 \
		                Flash family",
		provenance:    "fixtures/llm-oracle/catalog-policy/compat-profiles.json:compat-04",
		expires_at_ms: None,
	},
	ExactOverride {
		id:            "umans-deepseek-v4-flash-lab-family",
		provider:      Some("umans"),
		model:         "umans-deepseek-v4-flash-0731-lab",
		logical_model: "umans-deepseek-v4-flash-0731-lab",
		family:        "deepseek",
		version:       None,
		effort:        None,
		rationale:     "The opaque Umans lab product identifier is backed by the reviewed DeepSeek \
		                V4 Flash family",
		provenance:    "fixtures/llm-oracle/catalog-policy/compat-profiles.json:compat-04",
		expires_at_ms: None,
	},
	ExactOverride {
		id:            "kilo-qwq-32b-family",
		provider:      Some("kilo"),
		model:         "qwq-32b",
		logical_model: "qwen/qwq-32b",
		family:        "qwen",
		version:       None,
		effort:        None,
		rationale:     "The reviewed Kilo QwQ deployment belongs to the Qwen family despite its \
		                opaque product spelling",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	ExactOverride {
		id:            "nanogpt-eva-qwen-2-5-family",
		provider:      Some("nanogpt"),
		model:         "EVA-Qwen2.5-32B-v0.2",
		logical_model: "EVA-UNIT-01/EVA-Qwen2.5-32B-v0.2",
		family:        "qwen",
		version:       None,
		effort:        None,
		rationale:     "The reviewed EVA deployment is a Qwen 2.5 derivative despite its opaque \
		                product namespace",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	ExactOverride {
		id:            "nanogpt-eva-qwen-2-5-72b-family",
		provider:      Some("nanogpt"),
		model:         "EVA-Qwen2.5-72B-v0.2",
		logical_model: "EVA-UNIT-01/EVA-Qwen2.5-72B-v0.2",
		family:        "qwen",
		version:       None,
		effort:        None,
		rationale:     "The reviewed EVA 72B deployment is a Qwen 2.5 derivative despite its opaque \
		                product namespace",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	ExactOverride {
		id:            "nanogpt-gemma-claude-opus-distill-family",
		provider:      Some("nanogpt"),
		model:         "Gemma-4-31B-Claude-4.6-Opus-Reasoning-Distilled",
		logical_model: "Gemma-4-31B-Claude-4.6-Opus-Reasoning-Distilled",
		family:        "anthropic",
		version:       None,
		effort:        None,
		rationale:     "The reviewed distillation lineage follows its Claude Opus teacher rather \
		                than the student architecture",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	ExactOverride {
		id:            "nanogpt-qwen-claude-opus-distill-family",
		provider:      Some("nanogpt"),
		model:         "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Derestricted",
		logical_model: "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Derestricted",
		family:        "anthropic",
		version:       None,
		effort:        None,
		rationale:     "The reviewed distillation lineage follows its Claude Opus teacher rather \
		                than the student architecture",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	ExactOverride {
		id:            "nanogpt-qwen-claude-opus-distill-lite-family",
		provider:      Some("nanogpt"),
		model:         "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Derestricted-Lite",
		logical_model: "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Derestricted-Lite",
		family:        "anthropic",
		version:       None,
		effort:        None,
		rationale:     "The reviewed lite distillation lineage follows its Claude Opus teacher \
		                rather than the student architecture",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	ExactOverride {
		id:            "nanogpt-qwenlong-l1-family",
		provider:      Some("nanogpt"),
		model:         "QwenLong-L1-32B",
		logical_model: "Tongyi-Zhiwen/QwenLong-L1-32B",
		family:        "qwen",
		version:       None,
		effort:        None,
		rationale:     "The reviewed Tongyi Zhiwen QwenLong deployment belongs to the Qwen family",
		provenance:    "fixtures/llm-oracle/catalog/models.normalized.json",
		expires_at_ms: None,
	},
	exact_family(
		"nanogpt-dolphin-qwen-family",
		"nanogpt",
		"dolphin-2.9.2-qwen2-72b",
		"cognitivecomputations/dolphin-2.9.2-qwen2-72b",
		"qwen",
		"The reviewed Dolphin deployment is a Qwen derivative",
	),
	exact_family(
		"nanogpt-cogito-qwen-family",
		"nanogpt",
		"cogito-v1-preview-qwen-32B",
		"deepcogito/cogito-v1-preview-qwen-32B",
		"qwen",
		"The reviewed Cogito deployment is a Qwen derivative",
	),
	exact_family(
		"nanogpt-qwq-preview-family",
		"nanogpt",
		"qwq-32b-preview",
		"qwen/qwq-32b-preview",
		"qwen",
		"The reviewed QwQ preview belongs to the Qwen family",
	),
	exact_family(
		"nanogpt-grayline-qwen-family",
		"nanogpt",
		"GrayLine-Qwen3-8B",
		"soob3123/GrayLine-Qwen3-8B",
		"qwen",
		"The reviewed GrayLine deployment is a Qwen derivative",
	),
	exact_family(
		"novita-r1-qwen-family",
		"novita",
		"deepseek-r1-0528-qwen3-8b",
		"deepseek/deepseek-r1-0528-qwen3-8b",
		"qwen",
		"The reviewed distillation lineage follows the Qwen student architecture",
	),
	exact_family(
		"openrouter-qwq-family",
		"openrouter",
		"qwq-32b",
		"qwen/qwq-32b",
		"qwen",
		"The reviewed QwQ deployment belongs to the Qwen family",
	),
	exact_family(
		"umans-qwen-3-6-family",
		"umans",
		"umans-qwen3.6-35b-a3b",
		"umans-qwen3.6-35b-a3b",
		"qwen",
		"The opaque Umans deployment is backed by the reviewed Qwen 3.6 family",
	),
	exact_family(
		"venice-e2ee-deepseek-family",
		"venice",
		"e2ee-deepseek-v4-flash",
		"e2ee-deepseek-v4-flash",
		"deepseek",
		"The opaque Venice E2EE deployment is backed by the reviewed DeepSeek family",
	),
	exact_family(
		"venice-e2ee-gpt-oss-120b-family",
		"venice",
		"e2ee-gpt-oss-120b-p",
		"e2ee-gpt-oss-120b-p",
		"gpt-oss",
		"The opaque Venice E2EE deployment is backed by the reviewed GPT OSS family",
	),
	exact_family(
		"venice-e2ee-gpt-oss-20b-family",
		"venice",
		"e2ee-gpt-oss-20b-p",
		"e2ee-gpt-oss-20b-p",
		"gpt-oss",
		"The opaque Venice E2EE deployment is backed by the reviewed GPT OSS family",
	),
	exact_family(
		"venice-e2ee-qwen-2-5-family",
		"venice",
		"e2ee-qwen-2-5-7b-p",
		"e2ee-qwen-2-5-7b-p",
		"qwen",
		"The opaque Venice E2EE deployment is backed by the reviewed Qwen family",
	),
	exact_family(
		"venice-e2ee-qwen-3-30b-family",
		"venice",
		"e2ee-qwen3-30b-a3b-p",
		"e2ee-qwen3-30b-a3b-p",
		"qwen",
		"The opaque Venice E2EE deployment is backed by the reviewed Qwen family",
	),
	exact_family(
		"venice-e2ee-qwen-3-5-family",
		"venice",
		"e2ee-qwen3-5-122b-a10b",
		"e2ee-qwen3-5-122b-a10b",
		"qwen",
		"The opaque Venice E2EE deployment is backed by the reviewed Qwen family",
	),
	exact_family(
		"venice-e2ee-qwen-3-6-27b-family",
		"venice",
		"e2ee-qwen3-6-27b",
		"e2ee-qwen3-6-27b",
		"qwen",
		"The opaque Venice E2EE deployment is backed by the reviewed Qwen family",
	),
	exact_family(
		"venice-e2ee-qwen-3-6-35b-family",
		"venice",
		"e2ee-qwen3-6-35b-a3b",
		"e2ee-qwen3-6-35b-a3b",
		"qwen",
		"The opaque Venice E2EE deployment is backed by the reviewed Qwen family",
	),
	exact_family(
		"venice-e2ee-qwen-3-6-uncensored-family",
		"venice",
		"e2ee-qwen3-6-35b-a3b-uncensored-p",
		"e2ee-qwen3-6-35b-a3b-uncensored-p",
		"qwen",
		"The opaque Venice E2EE deployment is backed by the reviewed Qwen family",
	),
	exact_family(
		"venice-e2ee-qwen-vl-family",
		"venice",
		"e2ee-qwen3-vl-30b-a3b-p",
		"e2ee-qwen3-vl-30b-a3b-p",
		"qwen",
		"The opaque Venice E2EE deployment is backed by the reviewed Qwen family",
	),
	exact_family(
		"venice-openai-gpt-oss-family",
		"venice",
		"openai-gpt-oss-120b",
		"openai-gpt-oss-120b",
		"gpt-oss",
		"The reviewed Venice deployment belongs to the GPT OSS family",
	),
	exact_family(
		"venice-xiaomi-mimo-family",
		"venice",
		"xiaomi-mimo-v2-5",
		"xiaomi-mimo-v2-5",
		"mimo",
		"The reviewed Venice deployment belongs to the Xiaomi MiMo family",
	),
];

/// Returns the reviewed built-in exact overrides in stable order.
pub const fn exact_overrides() -> &'static [ExactOverride] {
	EXACT_OVERRIDES
}

/// Classifies one source identity without consulting process state.
#[must_use]
pub fn classify(input: ClassificationInput<'_>) -> ModelClassification {
	let trimmed = input.model.trim();
	let bare = trimmed.rsplit('/').next().unwrap_or(trimmed);
	if let Some(override_) = EXACT_OVERRIDES.iter().find(|override_| {
		override_.model.eq_ignore_ascii_case(bare)
			&& override_
				.provider
				.is_none_or(|provider| provider.eq_ignore_ascii_case(input.provider))
			&& !is_expired(override_.expires_at_ms, input.observed_at_ms)
	}) {
		return ModelClassification {
			logical_model:    Str::from(override_.logical_model),
			family:           FamilyId::new(override_.family),
			version:          override_.version,
			effort:           override_.effort,
			thinking_variant: false,
			evidence:         ClassificationEvidence {
				method:        ClassificationMethod::ExactOverride,
				rule:          Str::from(override_.id),
				rationale:     Str::from(override_.rationale),
				provenance:    Str::from(override_.provenance),
				expires_at_ms: override_.expires_at_ms,
			},
		};
	}

	let (logical, effort, thinking_variant) = if trimmed.len() == input.model.len() {
		collapse_suffix(trimmed)
	} else {
		(trimmed, None, false)
	};
	let family = family(&logical);
	let version = parse_version(family.as_str(), &logical);
	let structural = effort.is_some() || thinking_variant;
	let method = if structural {
		ClassificationMethod::StructuralSuffix
	} else if family.as_str() == "unknown" {
		ClassificationMethod::Unknown
	} else {
		ClassificationMethod::FamilyRule
	};
	ModelClassification {
		logical_model: Str::from(logical),
		family,
		version,
		effort,
		thinking_variant,
		evidence: ClassificationEvidence {
			method,
			rule: Str::from(if structural {
				"effort-suffix-v1"
			} else {
				"family-segments-v1"
			}),
			rationale: Str::from(if structural {
				"provider row is a structurally named effort route of one logical model"
			} else {
				"bounded vendor and model-family segments establish lineage"
			}),
			provenance: Str::from(match input.phase {
				ClassificationPhase::CatalogCompiler => "catalog-compiler",
				ClassificationPhase::DiscoveryNormalizer => "provider-discovery",
			}),
			expires_at_ms: None,
		},
	}
}

fn is_expired(expiry: Option<u64>, observed: Option<u64>) -> bool {
	matches!((expiry, observed), (Some(expiry), Some(observed)) if observed >= expiry)
}

fn collapse_suffix(model: &str) -> (&str, Option<EffortTier>, bool) {
	let lower = model.to_ascii_lowercase();
	if lower.ends_with("-thinking") {
		return (&model[..model.len() - "-thinking".len()], None, true);
	}
	for (suffix, effort) in [
		("-minimal", EffortTier::Minimal),
		("-medium", EffortTier::Medium),
		("-xhigh", EffortTier::XHigh),
		("-high", EffortTier::High),
		("-low", EffortTier::Low),
		("-max", EffortTier::Max),
	] {
		if lower.ends_with(suffix) {
			// Qwen Max is a product tier (including dotted generations such as
			// `qwen3.8-max`), never a portable reasoning-effort suffix.
			let bare = lower.rsplit('/').next().unwrap_or(lower.as_str());
			if suffix == "-max" && bare.starts_with("qwen") {
				break;
			}
			return (&model[..model.len() - suffix.len()], Some(effort), false);
		}
	}
	(model, None, false)
}

fn family(model: &str) -> FamilyId {
	let lower = model.trim().to_ascii_lowercase();
	let segments: Vec<&str> = lower
		.split('/')
		.filter(|segment| !segment.is_empty())
		.collect();
	let bare = segments.last().copied().unwrap_or(lower.as_str());
	let selected =
		if namespaced(&lower, "anthropic") || bounded(bare, "anthropic") || bounded(bare, "claude") {
			"anthropic"
		} else if namespaced(&lower, "gpt-oss") || bounded(bare, "gpt-oss") {
			"gpt-oss"
		} else if segments.iter().any(|segment| *segment == "openai") || openai_family(bare) {
			"openai"
		} else if segments.iter().any(|segment| *segment == "moonshotai") || bounded(bare, "kimi") {
			"kimi"
		} else if bare.contains("distill-qwen") || bounded(bare, "qwen") {
			"qwen"
		} else if segments.iter().any(|segment| *segment == "minimax")
			|| bounded(bare, "minimax")
			|| bounded(bare, "hailuo")
		{
			"minimax"
		} else if namespaced(&lower, "deepseek") || bounded(bare, "deepseek") {
			"deepseek"
		} else if bounded(bare, "mimo") {
			"mimo"
		} else if bounded(bare, "gemma") {
			"gemma"
		} else if bounded(bare, "glm") {
			"glm"
		} else if bounded(bare, "gemini") {
			"gemini"
		} else if bounded(bare, "grok")
			|| segments
				.iter()
				.any(|segment| matches!(*segment, "x-ai" | "xai"))
		{
			"xai"
		} else if bounded(bare, "llama") || segments.iter().any(|segment| *segment == "meta-llama") {
			"meta"
		} else if bounded(bare, "mistral") || bounded(bare, "mixtral") {
			"mistral"
		} else if bounded(bare, "command") || segments.iter().any(|segment| *segment == "cohere") {
			"cohere"
		} else if bounded(bare, "jamba") || segments.iter().any(|segment| *segment == "ai21") {
			"ai21"
		} else if bounded(bare, "nova") || bounded(bare, "titan") {
			"amazon"
		} else if bounded(bare, "doubao") {
			"bytedance"
		} else if bounded(bare, "ernie") {
			"baidu"
		} else if bounded(bare, "step") {
			"stepfun"
		} else {
			"unknown"
		};
	FamilyId::new(selected)
}

fn namespaced(value: &str, namespace: &str) -> bool {
	value
		.split(['/', '.', ':'])
		.any(|segment| segment == namespace || bounded(segment, namespace))
}

fn bounded(value: &str, prefix: &str) -> bool {
	value == prefix
		|| value.strip_prefix(prefix).is_some_and(|rest| {
			rest
				.as_bytes()
				.first()
				.is_some_and(|byte| matches!(byte, b'-' | b'_' | b'.' | b':' | b'0'..=b'9'))
		})
}

fn openai_family(bare: &str) -> bool {
	bare.starts_with("gpt-")
		|| bare.starts_with("chatgpt-")
		|| bare.starts_with("codex-")
		|| ["o1", "o3", "o4"].iter().any(|prefix| {
			bare == *prefix
				|| bare
					.strip_prefix(prefix)
					.is_some_and(|suffix| suffix.starts_with('-') || suffix.starts_with('.'))
		})
}

fn parse_version(family: &str, model: &str) -> Option<ModelVersion> {
	let bare = model.rsplit('/').next()?;
	let lower = bare.to_ascii_lowercase();
	if family == "openai" && ["o1", "o3", "o4"].contains(&lower.as_str()) {
		return None;
	}
	let prefixes: &[&str] = match family {
		"openai" => &["chatgpt-", "gpt-", "o"],
		"gemini" => &["gemini-"],
		"anthropic" => &["claude-"],
		"qwen" => &["qwen"],
		"xai" => &["grok-"],
		_ => &[],
	};
	let tail = prefixes
		.iter()
		.find_map(|prefix| lower.strip_prefix(prefix))?;
	let start = tail.find(|character: char| character.is_ascii_digit())?;
	let numeric: String = tail[start..]
		.chars()
		.take_while(|character| character.is_ascii_digit() || *character == '.')
		.collect();
	let mut parts = numeric.split('.');
	Some(ModelVersion {
		major: parts.next()?.parse().ok()?,
		minor: parts
			.next()
			.filter(|part| !part.is_empty())
			.map(str::parse)
			.transpose()
			.ok()?
			.unwrap_or(0),
		patch: parts
			.next()
			.filter(|part| !part.is_empty())
			.map(str::parse)
			.transpose()
			.ok()?
			.unwrap_or(0),
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	fn compiler(model: &str) -> ModelClassification {
		classify(ClassificationInput {
			phase: ClassificationPhase::CatalogCompiler,
			provider: "test",
			model,
			observed_at_ms: None,
		})
	}

	#[test]
	fn classifies_bounded_families_without_substring_false_positives() {
		assert_eq!(compiler("openrouter/qwen/qwen3-coder").family.as_str(), "qwen");
		assert_eq!(compiler("acme/notqwen-model").family.as_str(), "unknown");
		assert_eq!(compiler("xai/grok-4.6").family.as_str(), "xai");
		assert_eq!(compiler("myxai/grokker").family.as_str(), "unknown");
	}

	#[test]
	fn preserves_qwen3_max_product_name_and_collapses_thinking_sibling() {
		let ordinary = compiler("qwen/qwen3-max");
		assert_eq!(ordinary.logical_model.as_str(), "qwen/qwen3-max");
		assert_eq!(ordinary.effort, None);
		let thinking = compiler("qwen/qwen3-max-thinking");
		assert_eq!(thinking.logical_model.as_str(), "qwen/qwen3-max");
		assert!(thinking.thinking_variant);
	}

	#[test]
	fn collapses_all_effort_tiers() {
		for (suffix, effort) in [
			("minimal", EffortTier::Minimal),
			("low", EffortTier::Low),
			("medium", EffortTier::Medium),
			("high", EffortTier::High),
			("xhigh", EffortTier::XHigh),
			("max", EffortTier::Max),
		] {
			let value = compiler(&format!("gpt-5-luna-{suffix}"));
			assert_eq!(value.logical_model.as_str(), "gpt-5-luna");
			assert_eq!(value.effort, Some(effort));
		}
	}

	#[test]
	fn expired_exact_override_falls_back_to_rules() {
		let before_expiry = classify(ClassificationInput {
			phase:          ClassificationPhase::CatalogCompiler,
			provider:       "openai",
			model:          "gpt-daybreak-blue-latest",
			observed_at_ms: Some(1_799_711_999_999),
		});
		assert_eq!(before_expiry.evidence.method, ClassificationMethod::ExactOverride);
		assert_eq!(before_expiry.version, Some(ModelVersion { major: 5, minor: 6, patch: 0 }));

		let at_expiry = classify(ClassificationInput {
			phase:          ClassificationPhase::CatalogCompiler,
			provider:       "openai",
			model:          "gpt-daybreak-blue-latest",
			observed_at_ms: Some(1_799_712_000_000),
		});
		assert_eq!(at_expiry.evidence.method, ClassificationMethod::FamilyRule);
		assert_eq!(at_expiry.version, None);
	}
}
