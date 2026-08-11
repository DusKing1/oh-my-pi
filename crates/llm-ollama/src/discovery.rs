//! Authenticated Ollama Cloud discovery, the sibling of the bundled local
//! `ollama` row, which reaches the same payload through the catalog's generic
//! `DiscoveryKind::OllamaTags` path.

use async_trait::async_trait;
use omp_llm_catalog::{
	TransportId,
	discovery::{Account, DiscoveryHttp, DiscoveryProtocol, Error, parse_ollama_tags},
	models::ModelCard,
	provider::ProviderEntry,
};

/// Discovery for the authenticated Ollama Cloud native model-listing protocol.
pub struct OllamaCloudDiscovery;

#[async_trait]
impl DiscoveryProtocol for OllamaCloudDiscovery {
	fn transports(&self) -> &'static [TransportId] {
		&[TransportId::OllamaChat]
	}

	async fn discover(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		http: &dyn DiscoveryHttp,
	) -> Result<Vec<ModelCard>, Error> {
		let url = format!("{}/api/tags", provider.base_url.trim_end_matches('/'));
		let response = http.get(provider, account, &url).await?;
		parse_ollama_tags(provider, response.ensure_success(provider)?)
	}
}

/// Registered by the application at daemon start-up.
pub static DISCOVERY: OllamaCloudDiscovery = OllamaCloudDiscovery;

#[cfg(test)]
mod tests {
	use bytes::Bytes;
	use omp_core::Str;
	use omp_llm_catalog::{
		compat::Compat,
		discovery::HttpResponse,
		provider::{AuthSpec, Facet as ProviderFacet},
	};
	use parking_lot::Mutex;
	use smallvec::{SmallVec, smallvec};

	use super::*;

	struct FixtureHttp {
		requested: Mutex<Option<String>>,
	}

	#[async_trait]
	impl DiscoveryHttp for FixtureHttp {
		async fn execute(
			&self,
			_provider: &ProviderEntry,
			_account: &Account,
			request: http::Request<Bytes>,
		) -> Result<HttpResponse, Error> {
			*self.requested.lock() = Some(request.uri().to_string());
			Ok(HttpResponse::new(
				200,
				Bytes::from_static(
					br#"{"models":[{"name":"qwen3:8b","model":"qwen3:8b","details":{"family":"qwen3"}}]}"#,
				),
			))
		}
	}

	fn provider() -> ProviderEntry {
		ProviderEntry::builder()
			.id(Str::new_static("ollama-cloud"))
			.transport(TransportId::OllamaChat)
			.base_url(Str::new_static("https://ollama.com"))
			.fallback_base_urls(SmallVec::new())
			.auth(AuthSpec::Bearer { env: smallvec![Str::new_static("OLLAMA_CLOUD_API_KEY")] })
			.facets([ProviderFacet::Chat].into())
			.headers(Default::default())
			.compat(Compat::default())
			.build()
	}

	#[tokio::test]
	async fn discovers_cloud_tags_at_native_endpoint() {
		let provider = provider();
		let http = FixtureHttp { requested: Mutex::new(None) };

		let cards = DISCOVERY
			.discover(&provider, &Account::provider_default(), &http)
			.await
			.expect("Ollama Cloud tags should parse");

		assert_eq!(http.requested.lock().as_deref(), Some("https://ollama.com/api/tags"));
		assert_eq!(cards.len(), 1);
		assert_eq!(cards[0].id.as_str(), "ollama-cloud/qwen3:8b");
		assert_eq!(cards[0].family.as_str(), "qwen3");
	}
}
