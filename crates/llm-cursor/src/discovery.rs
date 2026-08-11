//! Cursor's authenticated live model-discovery protocol.

use async_trait::async_trait;
use http::{Method, Request, Version};
use omp_llm_catalog::{
	discovery::{Account, DiscoveryHttp, DiscoveryProtocol, Error, discovered_card},
	models::{Modality, ModelCard},
	provider::{ProviderEntry, TransportId},
};

use crate::DiscoveredModel;

/// Cursor's `GetUsableModels` discovery protocol.
pub struct CursorDiscovery;

#[async_trait]
impl DiscoveryProtocol for CursorDiscovery {
	fn transports(&self) -> &'static [TransportId] {
		&[TransportId::Cursor]
	}

	async fn discover(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		http: &dyn DiscoveryHttp,
	) -> Result<Vec<ModelCard>, Error> {
		let request = Request::builder()
			.method(Method::POST)
			.version(Version::HTTP_2)
			.uri(format!(
				"{}/agent.v1.AgentService/GetUsableModels",
				provider.base_url.trim_end_matches('/')
			))
			.header("content-type", "application/proto")
			.header("accept", "application/proto")
			.header("te", "trailers")
			.header("x-ghost-mode", "true")
			.header("x-cursor-client-version", "cli-2026.07.23-e383d2b")
			.header("x-cursor-client-type", "cli")
			.body(crate::model_discovery_request())
			.map_err(Error::transport)?;
		let response = http.execute(provider, account, request).await?;
		let body = response.ensure_success(provider)?;
		crate::decode_model_discovery(body)
			.map_err(Error::transport)
			.map(|models| models.iter().map(|model| card(provider, model)).collect())
	}
}

fn card(provider: &ProviderEntry, model: &DiscoveredModel) -> ModelCard {
	let family = model
		.id
		.as_str()
		.split(['/', '-'])
		.next()
		.unwrap_or(model.id.as_str());
	let mut card = discovered_card(provider, model.id.as_str(), model.name.as_str(), family);
	card.reasoning = model.reasoning;
	card.context_window = if model.max_mode { 1_000_000 } else { 200_000 };
	card.max_output_tokens = 64_000;
	if model.id.contains("claude")
		|| model.id.contains("gemini")
		|| model.id.contains("gpt-")
		|| model.id.contains("codex")
	{
		card.inputs.push(Modality::Image);
	}
	card
}

/// Discovery implementation registered by the application at daemon start-up.
pub static DISCOVERY: CursorDiscovery = CursorDiscovery;

#[cfg(test)]
mod tests {
	use omp_core::Str;
	use omp_llm_catalog::{
		compat::Compat,
		models::Modality,
		provider::{AuthSpec, Facet, ProviderEntry, TransportId},
	};
	use smallvec::SmallVec;

	use super::{DiscoveredModel, card};

	fn provider() -> ProviderEntry {
		ProviderEntry::builder()
			.id(Str::new_static("fixture"))
			.transport(TransportId::Cursor)
			.base_url(Str::new_static("https://example.invalid"))
			.fallback_base_urls(SmallVec::new())
			.auth(AuthSpec::None)
			.facets([Facet::Chat].into())
			.headers(Default::default())
			.compat(Compat::default())
			.build()
	}

	#[test]
	fn shapes_context_window_and_image_input() {
		let models = [
			DiscoveredModel {
				id:        Str::new_static("claude-sonnet"),
				name:      Str::new_static("Claude Sonnet"),
				reasoning: true,
				max_mode:  true,
			},
			DiscoveredModel {
				id:        Str::new_static("plain-model"),
				name:      Str::new_static("Plain Model"),
				reasoning: false,
				max_mode:  false,
			},
		];
		let cards: Vec<_> = models
			.iter()
			.map(|model| card(&provider(), model))
			.collect();

		assert_eq!(cards[0].context_window, 1_000_000);
		assert!(cards[0].inputs.contains(&Modality::Image));
		assert_eq!(cards[1].context_window, 200_000);
		assert!(!cards[1].inputs.contains(&Modality::Image));
	}
}
