//! Devin model discovery, whose protobuf body is sealed until credentialed
//! dispatch.
//!
//! The protocol marks its empty request with [`SealedBody`]. The runtime's
//! discovery executor then calls [`crate::model_discovery_request`] with the
//! account secret inside the broker credential boundary, so this module never
//! receives credential bytes.

use async_trait::async_trait;
use bytes::Bytes;
use http::{Method, Request, header::ACCEPT};
use omp_llm_catalog::{
	discovery::{Account, DiscoveryHttp, DiscoveryProtocol, Error, SealedBody, discovered_card},
	models::{Modality, ModelCard},
	provider::{ProviderEntry, TransportId},
};

use crate::DiscoveredModel;

/// Devin's protobuf model-listing protocol.
pub struct DevinDiscovery;

#[async_trait]
impl DiscoveryProtocol for DevinDiscovery {
	fn transports(&self) -> &'static [TransportId] {
		&[TransportId::Devin]
	}

	async fn discover(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		http: &dyn DiscoveryHttp,
	) -> Result<Vec<ModelCard>, Error> {
		let mut request = Request::builder()
			.method(Method::POST)
			.uri(format!(
				"{}/exa.api_server_pb.ApiServerService/GetCliModelConfigs",
				provider.base_url.trim_end_matches('/')
			))
			.header(ACCEPT, "*/*")
			.header("content-type", "application/proto")
			.header("connect-protocol-version", "1")
			.body(Bytes::new())
			.map_err(Error::transport)?;
		// The executor fills this empty body with model_discovery_request(secret)
		// inside its credential boundary.
		request.extensions_mut().insert(SealedBody);
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
		.split(['-', '_'])
		.next()
		.unwrap_or(model.id.as_str());
	let mut card = discovered_card(provider, model.id.as_str(), model.name.as_str(), family);
	card.reasoning = model.reasoning;
	card.context_window = model.context_window;
	card.max_output_tokens = model.max_output_tokens;
	if model.supports_images {
		card.inputs.push(Modality::Image);
	}
	card
}

/// Registered by the application at daemon start-up.
pub static DISCOVERY: DevinDiscovery = DevinDiscovery;

#[cfg(test)]
mod tests {
	use omp_core::Str;
	use omp_llm_catalog::{models::Modality, provider::load_builtin};

	use super::*;

	#[test]
	fn shapes_discovered_model_card() {
		let providers = load_builtin().expect("built-in provider catalog");
		let provider = providers
			.values()
			.find(|provider| provider.transport == TransportId::Devin)
			.expect("Devin transport provider");
		let model = DiscoveredModel {
			id:                Str::from("claude_sonnet-4"),
			name:              Str::from("Claude Sonnet 4"),
			supports_images:   true,
			reasoning:         true,
			context_window:    200_000,
			max_output_tokens: 64_000,
		};

		let card = card(provider, &model);

		assert_eq!(card.family, "claude");
		assert_eq!(card.context_window, 200_000);
		assert_eq!(card.max_output_tokens, 64_000);
		assert!(card.inputs.contains(&Modality::Image));
	}
}
