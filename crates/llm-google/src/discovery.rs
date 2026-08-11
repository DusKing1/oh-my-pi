//! Cloud Code Assist and Antigravity model discovery.

use std::collections::BTreeMap;

use async_trait::async_trait;
use bytes::Bytes;
use http::{
	Method, Request,
	header::{ACCEPT, CONTENT_TYPE},
};
use omp_llm_catalog::{
	discovery::{Account, DiscoveryHttp, DiscoveryProtocol, Error, discovered_card, infer_family},
	models::{Modality, ModelCard},
	provider::{ProviderEntry, TransportId},
};
use serde_json::Value;

/// Discovery protocol for Google Cloud Code Assist and Antigravity transports.
pub struct CcaDiscovery;

#[async_trait]
impl DiscoveryProtocol for CcaDiscovery {
	fn transports(&self) -> &'static [TransportId] {
		&[TransportId::GoogleCca]
	}

	async fn discover(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		http: &dyn DiscoveryHttp,
	) -> Result<Vec<ModelCard>, Error> {
		let mut endpoints = Vec::with_capacity(1 + provider.fallback_base_urls.len());
		endpoints.push(provider.base_url.as_str());
		endpoints.extend(
			provider
				.fallback_base_urls
				.iter()
				.map(|endpoint| endpoint.as_str()),
		);
		let mut last_error = None;
		for endpoint in endpoints {
			let url = format!("{}/v1internal:fetchAvailableModels", endpoint.trim_end_matches('/'));
			let request = Request::builder()
				.method(Method::POST)
				.uri(url)
				.header(ACCEPT, "application/json")
				.header(CONTENT_TYPE, "application/json")
				.body(Bytes::from_static(b"{}"))
				.map_err(Error::transport);
			let response = match request {
				Ok(request) => http.execute(provider, account, request).await,
				Err(error) => Err(error),
			};
			match response {
				Ok(response) if response.is_success() => {
					return parse_cca_models(provider, &response.body);
				},
				Ok(response) => last_error = Some(Error::status(provider, response.status)),
				Err(error) => last_error = Some(error),
			}
		}
		Err(last_error.unwrap_or_else(|| Error::transport("CCA discovery had no endpoint")))
	}
}

/// Registered by the application at daemon start-up.
pub static DISCOVERY: CcaDiscovery = CcaDiscovery;

/// Parses Cloud Code Assist/Antigravity `fetchAvailableModels` responses.
///
/// # Errors
///
/// Returns [`Error::InvalidPayload`] when the response lacks its model map.
pub fn parse_cca_models(provider: &ProviderEntry, body: &[u8]) -> Result<Vec<ModelCard>, Error> {
	let payload: Value =
		serde_json::from_slice(body).map_err(|error| Error::payload(provider, error))?;
	let models = payload
		.get("models")
		.and_then(Value::as_object)
		.ok_or_else(|| Error::payload(provider, "missing models object"))?;
	let mut cards = BTreeMap::new();
	for (id, entry) in models {
		if id.is_empty()
			|| matches!(id.as_str(), "chat_20706" | "chat_23310" | "gemini-2.5-pro")
			|| entry.get("isInternal").and_then(Value::as_bool) == Some(true)
		{
			continue;
		}
		let name = entry
			.get("displayName")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
			.unwrap_or(id);
		let mut card = discovered_card(provider, id, name, infer_family(id));
		card.reasoning = entry
			.get("supportsThinking")
			.and_then(Value::as_bool)
			.unwrap_or(false);
		card.context_window = entry
			.get("maxTokens")
			.and_then(Value::as_u64)
			.unwrap_or(200_000);
		card.max_output_tokens = entry
			.get("maxOutputTokens")
			.and_then(Value::as_u64)
			.unwrap_or(64_000);
		if entry.get("supportsImages").and_then(Value::as_bool) == Some(true) {
			card.inputs.push(Modality::Image);
		}

		cards.insert(card.id.clone(), card);
	}
	Ok(cards.into_values().collect())
}

#[cfg(test)]
mod tests {
	use std::collections::BTreeMap;

	use omp_core::Str;
	use omp_llm_catalog::{
		compat::Compat,
		provider::{AuthSpec, Facet as ProviderFacet},
	};
	use smallvec::SmallVec;

	use super::*;

	fn provider(id: &str, base_url: &str) -> ProviderEntry {
		ProviderEntry::builder()
			.id(Str::from(id))
			.transport(TransportId::GoogleCca)
			.base_url(Str::from(base_url))
			.fallback_base_urls(SmallVec::new())
			.auth(AuthSpec::None)
			.facets([ProviderFacet::Chat].into())
			.headers(BTreeMap::new())
			.compat(Compat::default())
			.build()
	}

	#[test]
	fn cca_parser_filters_internal_and_denylisted_rows() {
		let provider = provider("google-antigravity", "https://daily-cloudcode-pa.googleapis.com");
		let cards = parse_cca_models(
			&provider,
			br#"{"models":{
				"gemini-3.1-pro":{"displayName":"Gemini 3.1 Pro","supportsImages":true,
					"supportsThinking":true,"maxTokens":1000000,"maxOutputTokens":65536},
				"internal":{"isInternal":true},
				"chat_20706":{"displayName":"Denied"}
			}}"#,
		)
		.expect("CCA response");
		assert_eq!(cards.len(), 1);
		assert_eq!(cards[0].model, "gemini-3.1-pro");
		assert_eq!(cards[0].max_output_tokens, 65_536);
		assert!(cards[0].reasoning);
	}
}
