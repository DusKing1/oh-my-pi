//! Account-scoped model discovery for `OpenAI` Codex.

use std::collections::BTreeMap;

use async_trait::async_trait;
use bytes::Bytes;
use http::{Method, Request, header::ACCEPT};
use omp_llm_catalog::{
	codex::{CODEX_CLIENT_VERSION, CODEX_ORIGINATOR},
	discovery::{
		Account, DiscoveryHttp, DiscoveryProtocol, Error, discovered_card, find_model_array,
		infer_family,
	},
	models::{Modality, ModelCard},
	provider::{ProviderEntry, TransportId},
};
use serde_json::Value;

/// `OpenAI` Codex's account-scoped model discovery protocol.
pub struct CodexDiscovery;

#[async_trait]
impl DiscoveryProtocol for CodexDiscovery {
	fn transports(&self) -> &'static [TransportId] {
		&[TransportId::OpenAiCodex]
	}

	async fn discover(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		http: &dyn DiscoveryHttp,
	) -> Result<Vec<ModelCard>, Error> {
		let base = provider.base_url.trim_end_matches('/');
		let mut last_error = None;
		for path in [
			format!("/codex/models?client_version={CODEX_CLIENT_VERSION}"),
			format!("/models?client_version={CODEX_CLIENT_VERSION}"),
		] {
			let mut request = Request::builder()
				.method(Method::GET)
				.uri(format!("{base}{path}"))
				.header(ACCEPT, "application/json")
				.header("OpenAI-Beta", "responses=experimental")
				.header("originator", CODEX_ORIGINATOR)
				.header("version", CODEX_CLIENT_VERSION);
			if let Some(account_id) = account.account_id.as_deref() {
				request = request.header("chatgpt-account-id", account_id);
			}
			let request = request.body(Bytes::new()).map_err(Error::transport);
			let response = match request {
				Ok(request) => http.execute(provider, account, request).await,
				Err(error) => Err(error),
			};
			match response {
				Ok(response) if response.is_success() => {
					return parse_codex_models(provider, &response.body);
				},
				Ok(response) => last_error = Some(Error::status(provider, response.status)),
				Err(error) => last_error = Some(error),
			}
		}
		Err(last_error.unwrap_or_else(|| Error::transport("Codex discovery had no endpoint")))
	}
}

/// Parses `OpenAI` Codex's account-scoped model registry response.
///
/// # Errors
///
/// Returns [`Error::InvalidPayload`] when the response has no model array.
pub fn parse_codex_models(provider: &ProviderEntry, body: &[u8]) -> Result<Vec<ModelCard>, Error> {
	let payload: Value =
		serde_json::from_slice(body).map_err(|error| Error::payload(provider, error))?;
	let entries = find_model_array(&payload)
		.ok_or_else(|| Error::payload(provider, "missing models/data array"))?;
	let mut cards = BTreeMap::new();
	for entry in entries {
		let Some(model) = entry
			.get("slug")
			.or_else(|| entry.get("id"))
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
		else {
			continue;
		};
		if matches!(entry.get("visibility").and_then(Value::as_str), Some("hide" | "hidden")) {
			continue;
		}
		let name = entry
			.get("display_name")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
			.unwrap_or(model);
		let mut card = discovered_card(provider, model, name, infer_family(model));
		card.context_window = entry
			.get("context_window")
			.and_then(Value::as_u64)
			.unwrap_or(272_000);
		card.max_output_tokens = 128_000.min(card.context_window);
		card.reasoning = entry
			.get("default_reasoning_level")
			.and_then(Value::as_str)
			.is_some_and(|level| !matches!(level, "none" | "off"))
			|| entry
				.get("supported_reasoning_levels")
				.and_then(Value::as_array)
				.is_some_and(|levels| !levels.is_empty());
		if entry
			.get("input_modalities")
			.and_then(Value::as_array)
			.is_some_and(|modalities| {
				modalities
					.iter()
					.any(|value| value.as_str() == Some("image"))
			}) {
			card.inputs.push(Modality::Image);
		}
		cards.insert(card.id.clone(), card);
	}
	Ok(cards.into_values().collect())
}

/// `OpenAI` Codex discovery protocol registered by the application at daemon
/// start-up.
pub static DISCOVERY: CodexDiscovery = CodexDiscovery;

#[cfg(test)]
mod tests {
	use omp_core::Str;
	use omp_llm_catalog::{
		compat::Compat,
		models::Modality,
		provider::{AuthSpec, Facet as ProviderFacet},
	};
	use smallvec::SmallVec;

	use super::*;

	fn provider() -> ProviderEntry {
		ProviderEntry::builder()
			.id(Str::from("openai-codex"))
			.transport(TransportId::OpenAiCodex)
			.base_url(Str::from("https://chatgpt.com/backend-api"))
			.fallback_base_urls(SmallVec::new())
			.auth(AuthSpec::None)
			.facets([ProviderFacet::Chat].into())
			.headers(BTreeMap::new())
			.compat(Compat::default())
			.build()
	}

	#[test]
	fn codex_parser_filters_hidden_rows_and_keeps_native_limits() {
		let provider = provider();
		let cards = parse_codex_models(
			&provider,
			br#"{"models":[
				{"slug":"gpt-5.2-codex","display_name":"GPT-5.2 Codex",
				 "visibility":"visible","context_window":400000,
				 "default_reasoning_level":"medium","input_modalities":["text","image"]},
				{"slug":"hidden-model","visibility":"hidden"},
				{"id":"gpt-5.1-codex-mini","supported_reasoning_levels":["low","high"]}
			]}"#,
		)
		.expect("Codex model response");

		assert_eq!(cards.len(), 2);
		assert_eq!(cards[0].model.as_str(), "gpt-5.1-codex-mini");
		assert_eq!(cards[0].context_window, 272_000);
		assert_eq!(cards[0].max_output_tokens, 128_000);
		assert!(cards[0].reasoning);
		assert_eq!(cards[1].model.as_str(), "gpt-5.2-codex");
		assert_eq!(cards[1].context_window, 400_000);
		assert_eq!(cards[1].max_output_tokens, 128_000);
		assert!(cards[1].reasoning);
		assert!(cards[1].inputs.contains(&Modality::Image));
	}
}
