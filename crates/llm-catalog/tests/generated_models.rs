//! Determinism and invariant coverage for the generated model catalog.

use omp_llm_catalog::{
	Effort,
	models::{
		ApplyPatchToolType, Availability, Modality, ModelReasoningMode, ModelThinkingEffort,
		ModelThinkingMode, ModelWire, PremiumMultiplier, PriceUnit, Source, import_catalog_zstd,
		load_catalog_zstd,
	},
	provider::{Facet, TransportId},
};

const GENERATED: &[u8] = include_bytes!("../models.json.zst");

#[test]
fn importer_is_byte_deterministic() {
	let source = br#"{
		" provider ": {
			" model ": {
				"id": "ignored-wire-id",
				"name": "Example",
				"provider": "ignored-provider",
				"reasoning": true,
				"input": ["text", "image"],
				"cost": {"input": 1.25, "output": 5, "cacheRead": 0.125, "cacheWrite": 0},
				"contextWindow": 1234,
				"maxTokens": 321,
				"api": "openai-completions",
				"baseUrl": "https://wire.example/v1",
				"headers": {"x-model-test": "must-not-be-bundled"},
				"premiumMultiplier": 0.33
			}
		}
	}"#;
	let first = import_catalog_zstd(source).expect("first import");
	let second = import_catalog_zstd(source).expect("second import");
	assert_eq!(first, second);

	let catalog = load_catalog_zstd(&first).expect("generated payload loads");
	let model = catalog
		.get("provider", "model")
		.expect("normalized identity");
	assert_eq!(model.id.as_str(), "provider/model");
	assert_eq!(model.context_window, 1234);
	assert_eq!(model.max_output_tokens, 321);
	assert_eq!(model.inputs.as_slice(), &[Modality::Text, Modality::Image]);
	assert_eq!(
		model.wire,
		Some(ModelWire {
			transport: TransportId::OpenAiChat,
			base_url:  Some("https://wire.example/v1".into()),
		})
	);
	assert_eq!(model.behavior.premium_multiplier, Some(PremiumMultiplier::from_millionths(330_000)));
	let client = serde_json::to_value(model).expect("client model serialization");
	assert!(client.get("wire").is_none());
	assert!(client.get("behavior").is_none());
	assert!(!client.to_string().contains("must-not-be-bundled"));
	assert!(!client.to_string().contains("wire.example"));
}

#[test]
fn importer_preserves_nulls_and_exact_number_lexemes() {
	let source = br#"{"p":{"m":{"id":"ignored","provider":"ignored","contextWindow":null,"maxTokens":null,"cost":{"input":0.09999999999999999}}}}"#;
	let payload = import_catalog_zstd(source).expect("source imports");
	let json = zstd::stream::decode_all(payload.as_slice()).expect("payload decodes");
	assert_eq!(
		json,
		br#"{"p":{"m":{"contextWindow":null,"cost":{"input":0.09999999999999999},"id":"m","maxTokens":null,"provider":"p"}}}"#
	);
}

#[test]
fn malformed_provider_map_cannot_fall_through_to_an_empty_envelope() {
	let malformed = br#"{"provider":{"model":{"unknownPiField":true}}}"#;
	assert!(omp_llm_catalog::models::load_catalog_json(malformed).is_err());

	let valid = br#"{"provider":{"model":{"id":"model","provider":"provider"}}}"#;
	let payload = import_catalog_zstd(valid).expect("nonempty source imports");
	let catalog = load_catalog_zstd(&payload).expect("imported provider map loads");
	assert!(!catalog.is_empty());
}

#[test]
fn importer_rejects_credential_and_framing_headers_with_model_path() {
	for header in ["authorization", "host", "content-length", "transfer-encoding"] {
		let source = format!(r#"{{"provider":{{"model":{{"headers":{{"{header}":"unsafe"}}}}}}}}"#);
		let error = import_catalog_zstd(source.as_bytes()).expect_err("unsafe header rejected");
		assert!(error.to_string().contains("provider/model"));
		assert!(error.to_string().contains(header));
	}
}

#[test]
fn importer_rejects_unknown_pi_api_names() {
	let source = br#"{
		"provider": {
			"model": {
				"api": "plausible-but-unsupported",
				"baseUrl": "https://wire.example/v1"
			}
		}
	}"#;
	assert!(import_catalog_zstd(source).is_err());
}

#[test]
fn checked_in_snapshot_has_stable_counts_and_identities() {
	let json = zstd::stream::decode_all(GENERATED).expect("checked-in zstd frame");
	let providers: serde_json::Map<String, serde_json::Value> =
		serde_json::from_slice(&json).expect("checked-in provider map");
	assert_eq!(providers.len(), 80);
	let source_records: usize = providers
		.values()
		.map(|models| models.as_object().expect("provider model map").len())
		.sum();
	assert_eq!(source_records, 4_293);
	assert!(providers.keys().all(|provider| !provider.is_empty()));
	assert!(providers.values().all(|models| {
		models
			.as_object()
			.is_some_and(|models| models.keys().all(|model| !model.is_empty()))
	}));

	let catalog = load_catalog_zstd(GENERATED).expect("checked-in catalog loads");
	assert_eq!(catalog.len(), 4_218);
	assert_eq!(source_records - catalog.len(), 75);
	assert!(catalog.models().iter().all(|model| {
		!model.id.is_empty() && !model.provider.is_empty() && !model.model.is_empty()
	}));
	assert!(
		catalog
			.models()
			.windows(2)
			.all(|pair| { (&pair[0].provider, &pair[0].model) < (&pair[1].provider, &pair[1].model) })
	);
	assert!(catalog.models().iter().all(|model| {
		model.availability == Availability::Unspecified && model.source == Source::Bundled
	}));
}

#[test]
fn checked_in_snapshot_preserves_mixed_provider_wire_routes() {
	let catalog = load_catalog_zstd(GENERATED).expect("checked-in catalog loads");
	let cases = [
		(
			"github-copilot",
			"claude-sonnet-4.5",
			TransportId::AnthropicMessages,
			"https://api.githubcopilot.com",
		),
		("github-copilot", "gpt-4.1", TransportId::OpenAiChat, "https://api.githubcopilot.com"),
		("github-copilot", "gpt-5", TransportId::OpenAiResponses, "https://api.githubcopilot.com"),
		(
			"opencode-zen",
			"claude-sonnet-4-5",
			TransportId::AnthropicMessages,
			"https://opencode.ai/zen",
		),
		("opencode-zen", "gemini-3-flash", TransportId::GoogleGenAi, "https://opencode.ai/zen/v1"),
		("opencode-go", "deepseek-v4-pro", TransportId::OpenAiChat, "https://opencode.ai/zen/go/v1"),
		(
			"opencode-go",
			"deepseek-v4-flash",
			TransportId::OpenAiResponses,
			"https://opencode.ai/zen/go/v1",
		),
		("opencode-go", "minimax-m2.5", TransportId::AnthropicMessages, "https://opencode.ai/zen/go"),
	];
	for (provider, model, transport, base_url) in cases {
		let wire = catalog
			.get(provider, model)
			.and_then(|card| card.wire.as_ref())
			.expect("representative model has wire metadata");
		assert_eq!(wire.transport, transport, "{provider}/{model}");
		assert_eq!(wire.base_url.as_deref(), Some(base_url), "{provider}/{model}");
	}
}

#[test]
fn checked_in_snapshot_marks_deepseek_responses_reasoning_replay() {
	let catalog = load_catalog_zstd(GENERATED).expect("checked-in catalog loads");
	let deepseek = catalog
		.get("opencode-go", "deepseek-v4-flash")
		.expect("DeepSeek Responses model exists");
	assert_eq!(
		deepseek
			.behavior
			.compat
			.get_ns("wire", "requires_reasoning_content_for_all_assistant_turns")
			.and_then(serde_json::Value::as_bool),
		Some(true),
	);
	let openai = catalog
		.get("openai", "gpt-5-mini")
		.expect("OpenAI Responses model exists");
	assert!(
		openai
			.behavior
			.compat
			.get_ns("wire", "requires_reasoning_content_for_all_assistant_turns")
			.is_none()
	);
}

#[test]
fn inherited_prices_enable_qwen3_max_thinking_collapse() {
	let catalog = load_catalog_zstd(GENERATED).expect("checked-in catalog loads");
	for provider in ["kilo", "openrouter"] {
		let model = catalog
			.get(provider, "qwen/qwen3-max")
			.expect("collapsed Qwen3 Max card");
		assert!(catalog.get(provider, "qwen/qwen3-max-thinking").is_none());
		assert_eq!(model.efforts.as_slice(), &[
			Effort::Minimal,
			Effort::Low,
			Effort::Medium,
			Effort::High
		]);
		assert_eq!(model.effort_routing[&Effort::Off], "qwen/qwen3-max");
		for effort in [Effort::Minimal, Effort::Low, Effort::Medium, Effort::High] {
			assert_eq!(model.effort_routing[&effort], "qwen/qwen3-max-thinking");
		}
		let thinking = model
			.behavior
			.thinking
			.as_ref()
			.expect("thinking behavior follows collapsed logical card");
		assert_eq!(thinking.requires_effort, Some(true));
		assert_eq!(thinking.effort_routing[&ModelThinkingEffort::Off], "qwen/qwen3-max");
		for effort in [
			ModelThinkingEffort::Minimal,
			ModelThinkingEffort::Low,
			ModelThinkingEffort::Medium,
			ModelThinkingEffort::High,
		] {
			assert_eq!(thinking.effort_routing[&effort], "qwen/qwen3-max-thinking");
		}
	}
}

#[test]
fn checked_in_snapshot_preserves_model_facts() {
	let catalog = load_catalog_zstd(GENERATED).expect("checked-in catalog loads");
	let model = catalog
		.get("agnes", "agnes-1.5-flash")
		.expect("known source model");
	assert_eq!(model.name.as_str(), "Agnes 1.5 Flash");
	assert_eq!(model.context_window, 256_000);
	assert_eq!(model.max_output_tokens, 64_000);
	assert_eq!(model.inputs.as_slice(), &[Modality::Text, Modality::Image]);
	assert_eq!(model.outputs.as_slice(), &[Modality::Text]);
	assert!(!model.reasoning);
	assert_eq!(model.pricing.len(), 4);
	assert!(model.pricing.iter().all(|price| price.nanos_usd == 0));
	assert!(
		model
			.pricing
			.iter()
			.any(|price| price.unit == PriceUnit::MtokInput)
	);
}

#[test]
fn checked_in_snapshot_retains_representative_server_behavior() {
	let catalog = load_catalog_zstd(GENERATED).expect("checked-in catalog loads");

	let openai = catalog.get("openai", "gpt-5").expect("OpenAI GPT-5");
	assert_eq!(openai.behavior.apply_patch_tool_type, Some(ApplyPatchToolType::Freeform));

	let copilot = catalog
		.get("github-copilot", "claude-haiku-4.5")
		.expect("Copilot Haiku");
	assert_eq!(
		copilot.behavior.premium_multiplier,
		Some(PremiumMultiplier::from_millionths(330_000))
	);
	let copilot_thinking = copilot
		.behavior
		.thinking
		.as_ref()
		.expect("Copilot thinking metadata");
	assert_eq!(copilot_thinking.mode, ModelThinkingMode::Budget);
	assert_eq!(copilot_thinking.efforts.as_slice(), &[
		ModelThinkingEffort::Minimal,
		ModelThinkingEffort::Low,
		ModelThinkingEffort::Medium,
		ModelThinkingEffort::High,
		ModelThinkingEffort::XHigh,
	]);

	let agnes = catalog
		.get("agnes", "agnes-2.0-flash")
		.expect("Agnes 2.0 Flash");
	assert_eq!(agnes.behavior.supports_tools, Some(true));
	assert_eq!(agnes.behavior.supports_computer_use, Some(false));
	assert_eq!(
		agnes.behavior.compat.get_ns("wire", "thinking_format"),
		Some(&serde_json::json!("qwen-chat-template"))
	);

	let codex = catalog
		.get("openai-codex", "gpt-5.3-codex-spark")
		.expect("Codex Spark");
	assert_eq!(codex.behavior.prefer_websockets, Some(true));
	assert_eq!(codex.behavior.priority, Some(26));
	let remote = codex
		.behavior
		.remote_compaction
		.as_ref()
		.expect("remote compaction metadata");
	assert_eq!(remote.enabled, Some(true));
	assert_eq!(remote.transport, Some(TransportId::OpenAiCodex));
	assert_eq!(remote.v2_streaming_enabled, Some(true));

	let antigravity = catalog
		.get("google-antigravity", "gemini-3.1-pro")
		.expect("Antigravity Gemini");
	assert_eq!(antigravity.behavior.request_model_id.as_deref(), Some("gemini-3.1-pro-low"));
	let budgets = &antigravity
		.behavior
		.thinking
		.as_ref()
		.expect("Antigravity thinking metadata")
		.effort_budgets;
	assert_eq!(budgets[&ModelThinkingEffort::Low], 1_001);
	assert_eq!(budgets[&ModelThinkingEffort::High], 10_001);

	let ollama = catalog
		.get("ollama-cloud", "cogito-2.1:671b")
		.expect("Ollama omit-token model");
	assert_eq!(ollama.behavior.omit_max_output_tokens, Some(true));
	assert_eq!(ollama.behavior.supports_computer_use_config, Some(false));

	let responses_lite = catalog
		.get("openai-codex", "gpt-5.6-luna")
		.expect("Responses Lite Codex model");
	assert_eq!(responses_lite.behavior.use_responses_lite, Some(true));
	assert_eq!(responses_lite.behavior.reasoning_mode, None);

	let pro = catalog
		.get("openai", "gpt-5.6-luna-pro")
		.expect("OpenAI pro reasoning alias");
	assert_eq!(pro.behavior.reasoning_mode, Some(ModelReasoningMode::Pro));
	assert_eq!(pro.behavior.request_model_id.as_deref(), Some("gpt-5.6-luna"));
}

#[test]
fn checked_in_snapshot_has_daybreak_tiers_and_deepseek_contract() {
	let catalog = load_catalog_zstd(GENERATED).expect("checked-in catalog loads");
	for (model_id, context, input_nanos, output_nanos) in [
		("daybreak-blue-latest", 1_050_000, 5_000_000_000, 30_000_000_000),
		("daybreak-red-latest", 400_000, 12_500_000_000, 75_000_000_000),
		("gpt-5.6-cyber", 400_000, 12_500_000_000, 75_000_000_000),
	] {
		let model = catalog.get("openai", model_id).expect("Daybreak model");
		assert_eq!(model.context_window, context);
		assert_eq!(model.max_output_tokens, 128_000);
		assert_eq!(
			model
				.pricing
				.iter()
				.find(|price| price.unit == PriceUnit::MtokInput)
				.map(|price| price.nanos_usd),
			Some(input_nanos)
		);
		assert_eq!(
			model
				.pricing
				.iter()
				.find(|price| price.unit == PriceUnit::MtokOutput)
				.map(|price| price.nanos_usd),
			Some(output_nanos)
		);
	}

	for (model_id, input_nanos, output_nanos) in [
		("daybreak-blue-latest", 10_000_000_000, 45_000_000_000),
		("gpt-5.6-luna", 400_000_000, 1_800_000_000),
		("gpt-5.6-terra", 4_000_000_000, 18_000_000_000),
	] {
		let tier = &catalog.get("openai", model_id).expect("tiered model").pricing_tiers[0];
		assert_eq!(tier.prompt_tokens_above, 272_000);
		assert_eq!(
			tier
				.pricing
				.iter()
				.find(|price| price.unit == PriceUnit::MtokInput)
				.map(|price| price.nanos_usd),
			Some(input_nanos)
		);
		assert_eq!(
			tier
				.pricing
				.iter()
				.find(|price| price.unit == PriceUnit::MtokOutput)
				.map(|price| price.nanos_usd),
			Some(output_nanos)
		);
	}

	for model_id in ["deepseek-v4-flash", "deepseek-v4-flash:0731", "deepseek-v4-flash:preview"] {
		assert_eq!(
			catalog.get("ollama-cloud", model_id).expect("DeepSeek Flash").efforts.as_slice(),
			&[Effort::Low, Effort::High, Effort::Max]
		);
	}
	for model_id in ["deepseek-v3.1:671b", "deepseek-v3.2", "deepseek-v4-pro"] {
		assert_eq!(
			catalog.get("ollama-cloud", model_id).expect("DeepSeek reasoner").efforts.as_slice(),
			&[Effort::High, Effort::Max]
		);
	}
}

#[test]
fn checked_in_snapshot_promotes_official_openai_embeddings() {
	let catalog = load_catalog_zstd(GENERATED).expect("checked-in catalog loads");
	for (model_id, dimensions, input_nanos) in [
		("text-embedding-3-small", 1_536_u64, 20_000_000),
		("text-embedding-3-large", 3_072_u64, 130_000_000),
	] {
		let model = catalog
			.get("openai", model_id)
			.expect("direct OpenAI embedding card");
		assert_eq!(model.facets.as_slice(), &[Facet::Embeddings]);
		assert_eq!(model.inputs.as_slice(), &[Modality::Text]);
		assert!(model.outputs.is_empty());
		assert_eq!(model.context_window, 8_192);
		assert_eq!(model.max_output_tokens, 0);
		assert_eq!(
			model.props.get_ns("openai", "embedding_dimensions"),
			Some(&serde_json::Value::from(dimensions))
		);
		assert_eq!(
			model
				.pricing
				.iter()
				.find(|price| price.unit == PriceUnit::MtokInput)
				.map(|price| price.nanos_usd),
			Some(input_nanos)
		);
	}
}
