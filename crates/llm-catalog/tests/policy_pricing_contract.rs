//! Contract tests for catalog policy compilation and pricing behavior.

use std::collections::BTreeMap;

use omp_core::Str;
use omp_llm_catalog::{
	ApplyPatchWireKind, CatalogSource, ComputerUseConfigSupport, ComputerUseWireSupport,
	ExtendedContextMode, MaxOutputTokensEmission, NanoUsd, PremiumMultiplier, Price, PriceTier,
	PriceUnit, Pricing, ProvenanceKind, SourceModelRecord, SourceProviderRecord, ThinkingEffort,
	ThinkingMode, UsageDimensions, WirePolicy, compile,
};

fn source_provider() -> SourceProviderRecord {
	serde_json::from_str(
		r#"{
			"transport":"cursor",
			"base_url":"https://catalog-policy.example.test",
			"compat":{"supportsStore":false}
		}"#,
	)
	.expect("typed provider source")
}

fn source_model() -> SourceModelRecord {
	serde_json::from_str(
		r#"{
			"name":"Policy Fixture",
			"reasoning":true,
			"input":["text"],
			"output":["text"],
			"cost":{"input":0.000001,"output":0.000002},
			"contextWindow":1000000,
			"maxTokens":1000,
			"thinking":{
				"mode":"effort",
				"efforts":["low","high"],
				"effortRouting":{"low":"opaque-low","high":"opaque-high"}
			},
			"cursorMaxMode":true,
			"requestModelId":"opaque-default",
			"applyPatchToolType":"freeform",
			"supportsComputerUse":false,
			"supportsComputerUseConfig":false,
			"omitMaxOutputTokens":true
		}"#,
	)
	.expect("typed model source")
}

#[test]
fn compiler_preserves_policy_provenance_interning_extended_context_and_wire_routing() {
	let compiled = compile(CatalogSource {
		providers: BTreeMap::from([(Str::from("cursor"), source_provider())]),
		models:    BTreeMap::from([(
			Str::from("cursor"),
			BTreeMap::from([(Str::from("gpt-5.1"), source_model())]),
		)]),
	})
	.expect("fixture catalog compiles");

	let model = compiled.models.first().expect("compiled model");
	let policy = compiled
		.wire_policies
		.iter()
		.find(|policy| policy.content_id() == model.wire_policy)
		.expect("model wire policy is interned by its content id");
	assert_eq!(policy.context.supports_store, None);
	assert_eq!(policy.context.extended_mode, Some(ExtendedContextMode::Extended));
	assert_eq!(policy.context.max_output_tokens, Some(MaxOutputTokensEmission::Omit),);
	assert_eq!(policy.tool.apply_patch, Some(ApplyPatchWireKind::Freeform));
	assert_eq!(policy.tool.computer_use, Some(ComputerUseWireSupport::Unsupported),);
	assert_eq!(policy.tool.computer_use_config, Some(ComputerUseConfigSupport::Unsupported),);
	assert_eq!(model.wire_ids[0].1.as_str(), "opaque-default");
	assert_eq!(model.thinking_routing.effort_routing[&ThinkingEffort::Low].as_str(), "opaque-low",);
	assert_eq!(model.thinking_routing.effort_routing[&ThinkingEffort::High].as_str(), "opaque-high",);
	assert_eq!(model.provenance.sources[0].kind, ProvenanceKind::Bundled);
	assert_eq!(model.provenance.sources[0].origin.as_str(), "catalog-oracle/models.json.zst",);

	let provider_policy = compiled
		.wire_policies
		.iter()
		.find(|candidate| candidate.context.supports_store == Some(false))
		.expect("provider default policy remains independently interned");
	assert_eq!(
		compiled.providers[0].wire_policy,
		provider_policy.content_id(),
		"provider record carries its independently interned default policy id",
	);
	assert_ne!(provider_policy.content_id(), policy.content_id());
	let encoded = policy.canonical_bytes();
	let decoded: WirePolicy = serde_json::from_slice(&encoded).expect("canonical policy round-trip");
	assert_eq!(decoded.content_id(), policy.content_id());
}

#[test]
fn baseten_kimi_k3_exposes_reasoning_with_max_as_the_default_effort() {
	let model: SourceModelRecord = serde_json::from_str(
		r#"{
			"name":"Kimi K3",
			"reasoning":false,
			"input":["text","image"],
			"output":["text"],
			"contextWindow":1048576,
			"maxTokens":262144
		}"#,
	)
	.expect("typed Baseten model source");
	let compiled = compile(CatalogSource {
		providers: BTreeMap::from([(Str::from("baseten"), source_provider())]),
		models:    BTreeMap::from([(
			Str::from("baseten"),
			BTreeMap::from([(Str::from("moonshotai/Kimi-K3"), model)]),
		)]),
	})
	.expect("Baseten catalog compiles");
	let model = compiled.models.first().expect("compiled Kimi K3");
	let chat = model.capabilities.chat.as_ref().expect("Kimi K3 chat capability");
	assert!(!chat.reasoning.is_unsupported(), "Kimi K3 advertises reasoning");
	let thinking_id = model.thinking.as_ref().expect("Kimi K3 thinking policy");
	let thinking = compiled
		.thinking_policies
		.iter()
		.find(|policy| policy.content_id() == *thinking_id)
		.expect("Kimi K3 thinking policy is interned");
	assert_eq!(thinking.mode, ThinkingMode::Effort);
	assert_eq!(
		thinking.efforts.as_slice(),
		[ThinkingEffort::Low, ThinkingEffort::High, ThinkingEffort::Max],
	);
	assert_eq!(thinking.default_level, Some(ThinkingEffort::Max));
}

#[test]
fn opencode_go_deepseek_v4_omits_tool_choice_without_hiding_tools() {
	let source = |name: &str| {
		serde_json::from_value::<SourceModelRecord>(serde_json::json!({
			"name": name,
			"reasoning": true,
			"supportsTools": true,
			"input": ["text"],
			"output": ["text"],
			"thinking": {
				"mode": "effort",
				"efforts": ["high"]
			}
		}))
		.expect("typed OpenCode model source")
	};
	let compiled = compile(CatalogSource {
		providers: BTreeMap::from([(Str::from("opencode-go"), source_provider())]),
		models:    BTreeMap::from([(
			Str::from("opencode-go"),
			BTreeMap::from([
				(Str::from("deepseek-v4-flash"), source("DeepSeek V4 Flash")),
				(Str::from("deepseek-v4-pro"), source("DeepSeek V4 Pro")),
			]),
		)]),
	})
	.expect("OpenCode Go catalog compiles");

	for key in ["opencode-go/deepseek-v4-flash", "opencode-go/deepseek-v4-pro"] {
		let model = compiled
			.models
			.iter()
			.find(|model| model.key.as_str() == key)
			.unwrap_or_else(|| panic!("compiled {key}"));
		let policy = compiled
			.wire_policies
			.iter()
			.find(|policy| policy.content_id() == model.wire_policy)
			.unwrap_or_else(|| panic!("{key} wire policy"));
		assert_eq!(policy.tool.supports_tool_choice, Some(false), "{key} tool_choice");
		let chat = model.capabilities.chat.as_ref().expect("chat capability");
		assert!(chat.tools.constraints().is_some(), "{key} still advertises tools");
	}
}

#[test]
fn integer_nano_usd_cost_is_exact_at_micro_usd_and_tier_boundaries() {
	let pricing =
		Pricing::new(vec![Price { unit: PriceUnit::MtokInput, nanos_usd: 1_000 }], vec![PriceTier {
			prompt_tokens_above: 1_000_000,
			components:          Box::new([Price {
				unit:      PriceUnit::MtokInput,
				nanos_usd: 2_000,
			}]),
		}])
		.expect("canonical pricing");

	assert_eq!(
		pricing
			.cost(UsageDimensions { input_tokens: 1_000_000, ..UsageDimensions::default() })
			.expect("base boundary"),
		NanoUsd::from_nanos(1_000),
	);
	assert_eq!(
		pricing
			.cost(UsageDimensions { input_tokens: 1_000_001, ..UsageDimensions::default() })
			.expect("extended tier boundary"),
		NanoUsd::from_nanos(2_001),
	);
	assert_eq!(
		PremiumMultiplier::from_millionths(500_000)
			.apply(NanoUsd::from_nanos(1))
			.expect("sub-nano result rounds upward"),
		NanoUsd::from_nanos(1),
	);
}
