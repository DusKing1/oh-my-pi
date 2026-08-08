//! Generated parity assertions for the Pi provider registry snapshot.
//!
//! Regenerate this list when `packages/ai/src/registry/registry.ts` changes.

use std::collections::BTreeSet;

use omp_llm_catalog::{
	models::embedded_catalog,
	oauth_params::{load_embedded as load_oauth, validate_provider_links},
	provider::{
		AuthSpec, BaseUrlVars, CredentialPlacement, Facet, RegistryMapping, TransportId,
		expand_base_url, load_builtin,
	},
};

const PI_REGISTRY_SOURCE: &str = "packages/ai/src/registry/registry.ts";

// Order is the source `ALL` array. Keeping the complete generated snapshot here
// makes additions, removals, and renames visible as a single coverage failure.
const PI_REGISTRY_IDS: [&str; 94] = [
	"azure",
	"openai-codex",
	"anthropic",
	"zai",
	"zai-coding-plan",
	"kimi-code",
	"openrouter",
	"github-copilot",
	"cursor",
	"devin",
	"google-antigravity",
	"google-gemini-cli",
	"openai-codex-device",
	"xai",
	"xai-oauth",
	"gitlab-duo",
	"gitlab-duo-agent",
	"alibaba-coding-plan",
	"alibaba-token-plan",
	"agnes",
	"agnes-plan",
	"aiand",
	"aimlapi",
	"friendli",
	"inception",
	"ovhai",
	"crofai",
	"zhipu-coding-plan",
	"umans",
	"qwen-portal",
	"sakana",
	"minimax-code",
	"minimax-code-cn",
	"xiaomi",
	"xiaomi-token-plan-sgp",
	"xiaomi-token-plan-ams",
	"xiaomi-token-plan-cn",
	"firepass",
	"deepseek",
	"meta",
	"moonshot",
	"cerebras",
	"baseten",
	"fireworks",
	"together",
	"nvidia",
	"novita",
	"cohere",
	"deepinfra",
	"stepfun",
	"stepfun-plan",
	"poolside",
	"huggingface",
	"perplexity",
	"gigachat",
	"yandex",
	"sarvam",
	"scaleway",
	"qianfan",
	"venice",
	"siliconflow",
	"siliconflow-cn",
	"synthetic",
	"nanogpt",
	"wafer-serverless",
	"coreweave",
	"vercel-ai-gateway",
	"cloudflare-ai-gateway",
	"litellm",
	"kilo",
	"zenmux",
	"opencode",
	"opencode-zen",
	"opencode-go",
	"tavily",
	"kagi",
	"exa",
	"parallel",
	"apple-intelligence",
	"ollama",
	"ollama-cloud",
	"lm-studio",
	"llama.cpp",
	"vllm",
	"openai",
	"google",
	"google-vertex",
	"groq",
	"mistral",
	"minimax",
	"minimax-cn",
	"amazon-bedrock",
	"bedrock-mantle",
	"gmi-cloud",
];

const PI_OAUTH_FLOW_IDS: [&str; 18] = [
	"anthropic",
	"openai-codex",
	"google-gemini-cli",
	"google-antigravity",
	"github-copilot",
	"xai",
	"kimi",
	"gitlab-duo",
	"gitlab-duo-workflow",
	"cursor",
	"zai",
	"devin",
	"perplexity",
	"opencode",
	"minimax-code",
	"minimax-code-cn",
	"wafer",
	"xiaomi",
];

#[test]
fn pi_registry_snapshot_has_exact_provider_coverage() {
	let providers = load_builtin().expect("shipped providers.toml must parse");
	let expected: BTreeSet<_> = PI_REGISTRY_IDS.into_iter().collect();
	let actual: BTreeSet<_> = providers.keys().map(|id| id.as_str()).collect();
	assert_eq!(actual, expected, "{PI_REGISTRY_SOURCE} drifted from generated coverage");

	for provider in providers.values() {
		assert!(
			provider.oauth_auth.is_none() || provider.oauth_flow.is_some(),
			"{} has OAuth placement without a flow",
			provider.id
		);
		match &provider.mapping {
			RegistryMapping::Concrete => {},
			RegistryMapping::Alias { target, reason } => {
				assert!(!reason.is_empty(), "alias {} is undocumented", provider.id);
				assert!(
					providers.contains_key(target),
					"alias {} targets missing {target}",
					provider.id
				);
				let canonical = &providers[target];
				assert_eq!(provider.transport, canonical.transport);
				assert_eq!(provider.base_url, canonical.base_url);
				assert_eq!(provider.auth, canonical.auth);
				assert_eq!(provider.facets, canonical.facets);
				assert_ne!(&provider.id, target, "provider {} aliases itself", provider.id);
			},
			RegistryMapping::Replacement { component, reason } => {
				assert!(!component.is_empty() && !reason.is_empty());
				assert!(provider.facets.is_empty(), "replacement {} advertises inference", provider.id);
				assert!(
					provider.pending_facets.is_empty(),
					"replacement {} claims pending inference",
					provider.id
				);
			},
		}
	}
}

#[test]
fn apple_intelligence_is_an_active_embedded_chat_provider() {
	let providers = load_builtin().expect("shipped providers.toml must parse");
	let apple = &providers["apple-intelligence"];
	assert_eq!(apple.transport, TransportId::Embedded);
	assert_eq!(apple.facets.as_slice(), &[Facet::Chat]);
	assert!(apple.pending_facets.is_empty());
	assert!(apple.pending_transport.is_none());
}

#[test]
fn all_oauth_flows_join_to_canonical_provider_rows() {
	let providers = load_builtin().expect("shipped providers.toml must parse");
	let oauth = load_oauth().expect("shipped oauth.toml must parse");
	assert_eq!(oauth.len(), PI_OAUTH_FLOW_IDS.len());
	let expected: BTreeSet<_> = PI_OAUTH_FLOW_IDS.into_iter().collect();
	let actual: BTreeSet<_> = oauth.iter().map(|row| row.provider.as_str()).collect();
	assert_eq!(actual, expected, "Pi OAuth source drifted from generated coverage");
	validate_provider_links(&providers, &oauth).expect("every OAuth flow must have a provider join");
}

#[test]
fn advertised_facets_have_real_or_explicitly_pending_transport_support() {
	let providers = load_builtin().expect("shipped providers.toml must parse");
	for provider in providers.values() {
		let advertised: BTreeSet<_> = provider.facets.iter().copied().collect();
		let pending: BTreeSet<_> = provider.pending_facets.iter().copied().collect();
		assert!(
			advertised.is_disjoint(&pending),
			"{} marks a facet both ready and pending",
			provider.id
		);
		assert!(
			advertised.iter().all(|facet| matches!(
				facet,
				Facet::Chat
					| Facet::Embeddings
					| Facet::ImageGeneration
					| Facet::AudioSpeech
					| Facet::AudioTranscription
					| Facet::VideoGeneration
			)),
			"{} advertises a facet with no implemented service",
			provider.id
		);

		if pending.is_empty() {
			assert!(
				provider.pending_transport.is_none(),
				"{} names a pending wire without facets",
				provider.id
			);
		} else {
			assert!(
				pending
					.iter()
					.all(|facet| *facet == Facet::Chat || *facet == Facet::ImageGeneration),
				"{} has an unexplained pending facet",
				provider.id
			);
			assert!(
				provider
					.pending_transport
					.as_ref()
					.is_some_and(|wire| !wire.is_empty())
			);
		}

		if !advertised.is_empty() {
			assert!(
				matches!(
					provider.transport,
					TransportId::OpenAiChat
						| TransportId::OpenAiResponses
						| TransportId::OpenAiCodex
						| TransportId::AnthropicMessages
						| TransportId::AnthropicBedrock
						| TransportId::BedrockConverse
						| TransportId::AnthropicVertex
						| TransportId::GoogleGenAi
						| TransportId::GoogleVertex
						| TransportId::GoogleCca
						| TransportId::OllamaChat
						| TransportId::Embedded
						| TransportId::Cursor
						| TransportId::Devin
						| TransportId::GitLabDuoWorkflow
						| TransportId::Omp
				),
				"{} advertises a facet on an unavailable transport",
				provider.id
			);
		}
	}
}

#[test]
fn fallback_headers_discovery_and_model_snapshot_are_retained() {
	let providers = load_builtin().expect("shipped providers.toml must parse");
	assert!(embedded_catalog().len() >= 4_000, "generated Pi model snapshot unexpectedly shrank");

	let anthropic = &providers["anthropic"];
	let AuthSpec::Header { env, .. } = &anthropic.auth else {
		panic!("Anthropic auth changed")
	};
	assert_eq!(env.iter().map(|name| name.as_str()).collect::<Vec<_>>(), [
		"ANTHROPIC_FOUNDRY_API_KEY",
		"ANTHROPIC_OAUTH_TOKEN",
		"ANTHROPIC_API_KEY"
	]);
	assert_eq!(anthropic.headers["anthropic-version"], "2023-06-01");
	assert_eq!(anthropic.oauth_auth.as_ref(), Some(&CredentialPlacement::Bearer));

	let google = &providers["google"];
	let AuthSpec::Query { env, .. } = &google.auth else {
		panic!("Google auth changed")
	};
	assert_eq!(env.iter().map(|name| name.as_str()).collect::<Vec<_>>(), [
		"GEMINI_API_KEY",
		"GOOGLE_API_KEY"
	]);

	let vertex = &providers["google-vertex"];
	let AuthSpec::GoogleAdc { api_key_env, project_env, location_env } = &vertex.auth else {
		panic!("Vertex auth changed")
	};
	assert_eq!(
		api_key_env
			.iter()
			.map(|name| name.as_str())
			.collect::<Vec<_>>(),
		["GOOGLE_CLOUD_API_KEY"]
	);
	assert_eq!(
		project_env
			.iter()
			.map(|name| name.as_str())
			.collect::<Vec<_>>(),
		["GOOGLE_CLOUD_PROJECT", "GCP_PROJECT", "GCLOUD_PROJECT"]
	);
	assert_eq!(
		location_env
			.iter()
			.map(|name| name.as_str())
			.collect::<Vec<_>>(),
		["GOOGLE_VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION", "VERTEX_LOCATION"]
	);
	assert_eq!(
		vertex.transport,
		TransportId::GoogleVertex,
		"Gemini Vertex must not use Anthropic projection"
	);
	assert_eq!(providers["amazon-bedrock"].transport, TransportId::BedrockConverse);
	let azure = &providers["azure"];
	assert_eq!(azure.transport, TransportId::OpenAiChat);
	assert_eq!(azure.api_version.as_deref(), Some("2024-10-21"));

	let copilot = &providers["github-copilot"];
	assert_eq!(copilot.headers["X-GitHub-Api-Version"], "2026-06-01");
	assert!(
		providers
			.values()
			.filter(|row| row.discovery.is_some())
			.count()
			>= 75
	);
}

#[test]
fn bounded_provider_url_variables_cover_registry_templates() {
	let providers = load_builtin().expect("shipped providers.toml must parse");
	let cloudflare = &providers["cloudflare-ai-gateway"];
	let expanded = expand_base_url(
		&cloudflare.base_url,
		BaseUrlVars::builder()
			.account("account-7")
			.gateway("production")
			.build(),
	)
	.expect("supported account and gateway placeholders expand");
	assert_eq!(expanded, "https://gateway.ai.cloudflare.com/v1/account-7/production/anthropic");

	let vertex = &providers["google-vertex"];
	let expanded =
		expand_base_url(&vertex.base_url, BaseUrlVars::builder().location("us-central1").build())
			.expect("supported location placeholder expands");
	assert_eq!(expanded, "https://us-central1-aiplatform.googleapis.com/v1");
}
