//! On-device model discovery for specialized `Embedded` providers.
//!
//! [`TransportId::Embedded`] is also shared by the Perplexity, Tavily, Kagi,
//! Exa, and Parallel tool providers. Those rows do not declare specialized
//! discovery, so this implementation is reached only for rows that do.

use async_trait::async_trait;
use omp_llm_catalog::{
	discovery::{Account, DiscoveryHttp, DiscoveryProtocol, Error, discovered_card},
	models::ModelCard,
	provider::{ProviderEntry, TransportId},
};

use crate::{AppleFm, CONTEXT_SIZE};

/// Discovers Apple's on-device Foundation Models chat model when it is usable.
pub struct AppleFmDiscovery;

#[async_trait]
impl DiscoveryProtocol for AppleFmDiscovery {
	fn transports(&self) -> &'static [TransportId] {
		&[TransportId::Embedded]
	}

	async fn discover(
		&self,
		provider: &ProviderEntry,
		_account: &Account,
		_http: &dyn DiscoveryHttp,
	) -> Result<Vec<ModelCard>, Error> {
		let availability = AppleFm::availability().await.map_err(Error::transport)?;
		if !availability.available {
			return Ok(Vec::new());
		}

		let mut card = discovered_card(provider, "apple-on-device", "Apple Intelligence", "apple");
		card.context_window = u64::from(CONTEXT_SIZE);
		Ok(vec![card])
	}
}

/// Apple Foundation Models discovery registered by the application at daemon
/// start-up.
pub static DISCOVERY: AppleFmDiscovery = AppleFmDiscovery;

#[cfg(test)]
mod tests {
	use bytes::Bytes;
	use http::Request;
	use omp_llm_catalog::{discovery::HttpResponse, provider::load_builtin};

	use super::*;

	struct NoHttp;

	#[async_trait]
	impl DiscoveryHttp for NoHttp {
		async fn execute(
			&self,
			_provider: &ProviderEntry,
			_account: &Account,
			_request: Request<Bytes>,
		) -> Result<HttpResponse, Error> {
			panic!("Apple Foundation Models discovery must not use HTTP")
		}
	}

	#[tokio::test]
	async fn discovery_treats_host_availability_as_a_normal_result() {
		let providers = load_builtin().unwrap();
		let provider = providers
			.values()
			.find(|provider| {
				provider.transport == TransportId::Embedded && provider.discovery.is_some()
			})
			.unwrap();

		let result = DISCOVERY
			.discover(provider, &Account::provider_default(), &NoHttp)
			.await;
		assert!(result.is_ok());
		assert!(result.unwrap().len() <= 1);
	}
}
