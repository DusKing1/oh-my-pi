//! Live model discovery behind the gateway's injected HTTP stack.
//!
//! `ollama`, `vllm`, `lm-studio`, and `litellm` are deliberately absent from
//! the bundled catalog: their model sets and localhost endpoints exist only at
//! runtime. Discovery is therefore required, rather than an optional catalog
//! enhancement. The same path also serves authenticated OpenAI-compatible
//! account listings; [`HttpClient`] receives the provider so the gateway can
//! inject a credential without this crate depending on the broker.

use std::collections::BTreeMap;

use async_trait::async_trait;
use bytes::Bytes;
use omp_core::Str;
use omp_llm_types::{Effort, Props};
use serde_json::Value;
use smallvec::SmallVec;
use url::Url;

use crate::{
	models::{Availability, Modality, ModelCard, Price, PriceUnit, Source},
	provider::{AuthSpec, DiscoveryKind, ProviderEntry, TransportId},
};

/// A successful HTTP response used by model discovery.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HttpResponse {
	/// HTTP status code.
	pub status: u16,
	/// Complete response body.
	pub body:   Bytes,
}

impl HttpResponse {
	/// Creates a complete discovery HTTP response.
	#[must_use]
	pub const fn new(status: u16, body: Bytes) -> Self {
		Self { status, body }
	}
}
/// Non-secret identity of one credential/account discovery source.
///
/// `key` is opaque to the catalog and is passed back to [`HttpClient`] for
/// credential selection. It must never contain credential material.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub struct Account {
	/// Stable, opaque source key.
	pub key:             Str,
	/// Client-safe account label used for diagnostics.
	pub label:           Str,
	/// Provider account id required by some account-scoped protocols.
	pub account_id:      Option<Str>,
	/// Provider organization or namespace id, when broker metadata supplies one.
	pub organization_id: Option<Str>,
	/// Provider project id, when broker metadata supplies one.
	pub project_id:      Option<Str>,
}

impl Account {
	/// Creates one client-safe account source.
	#[must_use]
	pub fn new(key: impl Into<Str>, label: impl Into<Str>) -> Self {
		Self {
			key:             key.into(),
			label:           label.into(),
			account_id:      None,
			organization_id: None,
			project_id:      None,
		}
	}

	/// Attaches a validated provider account id.
	#[must_use]
	pub fn with_account_id(mut self, account_id: Option<Str>) -> Self {
		self.account_id = account_id;
		self
	}

	/// Attaches client-safe organization and project scope.
	#[must_use]
	pub fn with_scope(
		mut self,
		organization_id: Option<Str>,
		project_id: Option<Str>,
	) -> Self {
		self.organization_id = organization_id;
		self.project_id = project_id;
		self
	}

	/// Returns the provider-wide source used when account enumeration is not
	/// supported by a transport.
	#[must_use]
	pub const fn provider_default() -> Self {
		Self {
			key:             Str::new_static(""),
			label:           Str::new_static("provider"),
			account_id:      None,
			organization_id: None,
			project_id:      None,
		}
	}
}

/// Minimal authenticated transport required by model discovery.
///
/// The provider and opaque account arguments are intentional: production
/// implementations select and inject a sealed credential without this crate
/// receiving or storing credential bytes. Existing provider-wide transports
/// need only implement [`Self::get`]; account-aware and specialized transports
/// override the other methods.
#[async_trait]
pub trait HttpClient: Send + Sync {
	/// Lists the client-safe credential/account sources visible to `provider`.
	async fn accounts(&self, _provider: &ProviderEntry) -> Result<Vec<Account>, Error> {
		Ok(vec![Account::provider_default()])
	}

	/// Performs an authenticated, cancellable `GET` for `provider`.
	async fn get(&self, provider: &ProviderEntry, url: &str) -> Result<HttpResponse, Error>;

	/// Performs a `GET` using exactly `account`, never another credential.
	async fn get_for_account(
		&self,
		provider: &ProviderEntry,
		_account: &Account,
		url: &str,
	) -> Result<HttpResponse, Error> {
		self.get(provider, url).await
	}

	/// Runs a provider-owned discovery protocol (for example Cursor Connect,
	/// Devin protobuf, Codex metadata, or Cloud Code Assist).
	async fn discover_specialized(
		&self,
		provider: &ProviderEntry,
		_account: &Account,
	) -> Result<Vec<ModelCard>, Error> {
		Err(Error::UnsupportedProvider(provider.id.clone()))
	}
}

/// Failure while fetching or decoding a provider's model listing.
#[derive(Clone, Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
	/// The provider has no model-listing convention known to this module.
	#[error("provider {0} does not support model discovery")]
	UnsupportedProvider(Str),
	/// The configured provider URL is invalid.
	#[error("invalid discovery URL for {provider}: {detail}")]
	InvalidUrl {
		/// Provider id.
		provider: Str,
		/// Parser detail.
		detail:   Str,
	},
	/// The injected transport failed.
	// TODO(errors): integrate transport evidence with omp_llm_error once it
	// exposes an owned provider-operation error type.
	#[error("model discovery transport failed: {0}")]
	Transport(Str),
	/// The provider returned a non-success status.
	#[error("model discovery returned HTTP {status} for {provider}")]
	HttpStatus {
		/// Provider id.
		provider: Str,
		/// HTTP status code.
		status:   u16,
	},
	/// The response was not a recognized model-list payload.
	#[error("invalid model listing for {provider}: {detail}")]
	InvalidPayload {
		/// Provider id.
		provider: Str,
		/// Decode detail.
		detail:   Str,
	},
}

/// Returns whether `provider` declares or infers a live-listing convention.
#[must_use]
pub fn supports(provider: &ProviderEntry) -> bool {
	provider.discovery.is_some()
		|| matches!(provider.id.as_str(), "ollama" | "vllm" | "lm-studio" | "litellm")
		|| ((is_openai_compatible(provider) || provider.transport == TransportId::GoogleGenAi)
			&& !matches!(&provider.auth, AuthSpec::None))
}

/// Discovers models through the provider-wide default credential source.
///
/// Account-aware runtimes should enumerate [`HttpClient::accounts`] and call
/// [`discover_account`] once per source.
///
/// # Errors
///
/// Returns an error for unsupported providers, invalid endpoints, transport or
/// HTTP failures, and malformed payloads.
pub async fn discover(
	provider: &ProviderEntry,
	http: &dyn HttpClient,
) -> Result<Vec<ModelCard>, Error> {
	discover_account(provider, &Account::provider_default(), http).await
}

/// Discovers the models visible to exactly one non-secret account source.
///
/// Specialized protocols are delegated to the injected production client;
/// shared HTTP protocols remain implemented and decoded in this crate.
///
/// # Errors
///
/// Returns an error when the protocol is unsupported or its transport/payload
/// fails.
pub async fn discover_account(
	provider: &ProviderEntry,
	account: &Account,
	http: &dyn HttpClient,
) -> Result<Vec<ModelCard>, Error> {
	let kind = endpoint_kind(provider)?;
	if matches!(kind, EndpointKind::Specialized) {
		return http.discover_specialized(provider, account).await;
	}
	if matches!(kind, EndpointKind::Google) {
		return discover_google_pages(provider, account, http).await;
	}
	let url = discovery_url(provider, kind)?;
	let response = http.get_for_account(provider, account, &url).await?;
	if !(200..300).contains(&response.status) {
		return Err(Error::HttpStatus { provider: provider.id.clone(), status: response.status });
	}

	match kind {
		EndpointKind::Ollama => parse_ollama(provider, &response.body),
		EndpointKind::OpenAi | EndpointKind::AccountModels => parse_openai(provider, &response.body),
		EndpointKind::Google => unreachable!("Google pagination returned above"),
		EndpointKind::Specialized => unreachable!("specialized discovery returned above"),
	}
}

#[derive(Clone, Copy)]
enum EndpointKind {
	Ollama,
	OpenAi,
	AccountModels,
	Google,
	Specialized,
}

fn endpoint_kind(provider: &ProviderEntry) -> Result<EndpointKind, Error> {
	if let Some(discovery) = &provider.discovery {
		return Ok(match discovery.kind {
			DiscoveryKind::OpenAiModels => EndpointKind::OpenAi,
			DiscoveryKind::GoogleModels => EndpointKind::Google,
			DiscoveryKind::OllamaTags => EndpointKind::Ollama,
			DiscoveryKind::AccountModels => EndpointKind::AccountModels,
			DiscoveryKind::Specialized => EndpointKind::Specialized,
		});
	}
	match provider.id.as_str() {
		"ollama" => Ok(EndpointKind::Ollama),
		"vllm" | "lm-studio" | "litellm" => Ok(EndpointKind::OpenAi),
		"openai-codex" => Ok(EndpointKind::AccountModels),
		_ if provider.transport == TransportId::GoogleGenAi
			&& !matches!(&provider.auth, AuthSpec::None) =>
		{
			Ok(EndpointKind::Google)
		},
		_ if is_openai_compatible(provider) && !matches!(&provider.auth, AuthSpec::None) => {
			Ok(EndpointKind::OpenAi)
		},
		_ => Err(Error::UnsupportedProvider(provider.id.clone())),
	}
}

const fn is_openai_compatible(provider: &ProviderEntry) -> bool {
	matches!(
		provider.transport,
		TransportId::OpenAiChat | TransportId::OpenAiResponses | TransportId::OpenAiCodex
	)
}

fn discovery_url(provider: &ProviderEntry, kind: EndpointKind) -> Result<String, Error> {
	let mut url = Url::parse(provider.base_url.as_str()).map_err(|error| Error::InvalidUrl {
		provider: provider.id.clone(),
		detail:   Str::from(error.to_string()),
	})?;
	url.set_query(None);
	url.set_fragment(None);

	match kind {
		EndpointKind::Ollama => url.set_path("/api/tags"),
		EndpointKind::OpenAi => {
			let base_path = url.path().trim_end_matches('/');
			let path = if base_path.ends_with("/v1") {
				format!("{base_path}/models")
			} else {
				format!("{base_path}/v1/models")
			};
			url.set_path(&path);
		},
		EndpointKind::AccountModels | EndpointKind::Google => {
			let base_path = url.path().trim_end_matches('/');
			url.set_path(&format!("{base_path}/models"));
		},
		EndpointKind::Specialized => {
			return Err(Error::UnsupportedProvider(provider.id.clone()));
		},
	}
	Ok(url.to_string())
}

fn parse_ollama(provider: &ProviderEntry, body: &[u8]) -> Result<Vec<ModelCard>, Error> {
	let payload: Value =
		serde_json::from_slice(body).map_err(|error| invalid_payload(provider, error))?;
	let entries = payload
		.get("models")
		.and_then(Value::as_array)
		.ok_or_else(|| invalid_payload(provider, "missing models array"))?;
	let mut cards = BTreeMap::new();
	for entry in entries {
		let Some(model) = entry
			.get("model")
			.or_else(|| entry.get("name"))
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
		else {
			continue;
		};
		let family = entry
			.get("details")
			.and_then(|details| details.get("family"))
			.and_then(Value::as_str)
			.unwrap_or_else(|| infer_family(model));
		let card = discovered_card(provider, model, model, family);
		cards.insert(card.id.clone(), card);
	}
	Ok(cards.into_values().collect())
}

fn parse_openai(provider: &ProviderEntry, body: &[u8]) -> Result<Vec<ModelCard>, Error> {
	let payload: Value =
		serde_json::from_slice(body).map_err(|error| invalid_payload(provider, error))?;
	let entries = find_model_array(&payload)
		.ok_or_else(|| invalid_payload(provider, "missing data/models/result/items array"))?;
	let copilot = provider.id == "github-copilot";
	let mut cards = BTreeMap::new();
	let mut variants = Vec::new();
	for entry in entries {
		if copilot
			&& entry
				.pointer("/capabilities/type")
				.and_then(Value::as_str)
				.is_some_and(|kind| kind != "chat")
		{
			continue;
		}
		let Some(model) = entry
			.get("id")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
		else {
			continue;
		};
		let name = entry
			.get("name")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
			.unwrap_or(model);
		let family = entry
			.get("owned_by")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
			.unwrap_or_else(|| infer_family(model));
		let mut card = discovered_card(provider, model, name, family);
		if copilot && let Some(variant) = apply_copilot_metadata(entry, &mut card) {
			variants.push(variant);
		}
		cards.insert(card.id.clone(), card);
	}
	for variant in variants {
		cards.entry(variant.id.clone()).or_insert(variant);
	}
	Ok(cards.into_values().collect())
}

#[derive(Clone, Copy, Default)]
struct CopilotTier {
	context_max:  Option<u64>,
	input_price:  Option<u64>,
	output_price: Option<u64>,
	cache_price:  Option<u64>,
}

fn apply_copilot_metadata(entry: &Value, card: &mut ModelCard) -> Option<ModelCard> {
	let full_context = positive_at(entry, &["capabilities", "limits", "max_context_window_tokens"])
		.or_else(|| positive_at(entry, &["context_length"]))
		.or_else(|| positive_at(entry, &["capabilities", "limits", "max_prompt_tokens"]));
	let max_output = positive_at(entry, &["capabilities", "limits", "max_output_tokens"])
		.or_else(|| positive_at(entry, &["max_completion_tokens"]))
		.or_else(|| {
			positive_at(entry, &["capabilities", "limits", "max_non_streaming_output_tokens"])
		});
	if let Some(context) = full_context {
		card.context_window = context;
	}
	if let Some(output) = max_output {
		card.max_output_tokens = output;
	}
	if entry
		.pointer("/capabilities/supports/vision")
		.and_then(Value::as_bool)
		== Some(true)
		&& !card.inputs.contains(&Modality::Image)
	{
		card.inputs.push(Modality::Image);
	}

	let default_tier = copilot_tier(entry, "default");
	if let Some(pricing) = copilot_pricing(default_tier) {
		card.pricing = pricing;
	}
	if let (Some(boundary), Some(context), Some(output)) =
		(default_tier.context_max, full_context, max_output)
		&& boundary != 0
	{
		card.context_window = context.min(boundary.saturating_add(output));
	}

	let long_tier = copilot_tier(entry, "long_context");
	let (Some(boundary), Some(context), Some(output)) =
		(long_tier.context_max, full_context, max_output)
	else {
		return None;
	};
	if boundary == 0 {
		return None;
	}
	let variant_context = context.min(boundary.saturating_add(output));
	if variant_context <= card.context_window {
		return None;
	}

	let mut variant = card.clone();
	let wire_model = card.model.clone();
	variant.model = omp_core::fmts!("{wire_model}-1m");
	variant.id = omp_core::fmts!("{}/{}", card.provider, variant.model);
	variant.name = omp_core::fmts!("{} (1M)", card.name);
	variant.context_window = variant_context;
	if let Some(pricing) = copilot_pricing(long_tier) {
		variant.pricing = pricing;
	}
	variant.effort_routing.insert(Effort::Off, wire_model);
	Some(variant)
}

fn positive_at(value: &Value, path: &[&str]) -> Option<u64> {
	path
		.iter()
		.try_fold(value, |value, field| value.get(*field))
		.and_then(Value::as_u64)
		.filter(|value| *value != 0)
}

fn copilot_tier(entry: &Value, name: &str) -> CopilotTier {
	let Some(tier) = entry
		.pointer("/billing/token_prices")
		.and_then(|prices| prices.get(name))
	else {
		return CopilotTier::default();
	};
	CopilotTier {
		context_max:  tier.get("context_max").and_then(Value::as_u64),
		input_price:  tier.get("input_price").and_then(Value::as_u64),
		output_price: tier.get("output_price").and_then(Value::as_u64),
		cache_price:  tier.get("cache_price").and_then(Value::as_u64),
	}
}

fn copilot_pricing(tier: CopilotTier) -> Option<SmallVec<Price, 4>> {
	let mut pricing = SmallVec::new();
	pricing.push(copilot_price(PriceUnit::MtokInput, tier.input_price?)?);
	pricing.push(copilot_price(PriceUnit::MtokOutput, tier.output_price?)?);
	if let Some(cache) = tier.cache_price {
		pricing.push(copilot_price(PriceUnit::MtokCacheRead, cache)?);
	}
	Some(pricing)
}

fn copilot_price(unit: PriceUnit, hundredths_usd: u64) -> Option<Price> {
	Some(Price { unit, nanos_usd: hundredths_usd.checked_mul(10_000_000)? })
}

fn parse_google(provider: &ProviderEntry, body: &[u8]) -> Result<Vec<ModelCard>, Error> {
	let payload: Value =
		serde_json::from_slice(body).map_err(|error| invalid_payload(provider, error))?;
	let entries = payload
		.get("models")
		.and_then(Value::as_array)
		.ok_or_else(|| invalid_payload(provider, "missing models array"))?;
	let mut cards = BTreeMap::new();
	for entry in entries {
		let Some(wire_name) = entry
			.get("name")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
		else {
			continue;
		};
		if entry
			.get("supportedGenerationMethods")
			.and_then(Value::as_array)
			.is_some_and(|methods| {
				!methods.iter().any(|method| {
					matches!(method.as_str(), Some("generateContent" | "streamGenerateContent"))
				})
			}) {
			continue;
		}
		let model = wire_name.strip_prefix("models/").unwrap_or(wire_name);
		let name = entry
			.get("displayName")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
			.unwrap_or(model);
		let mut card = discovered_card(provider, model, name, infer_family(model));
		card.context_window = entry
			.get("inputTokenLimit")
			.and_then(Value::as_u64)
			.unwrap_or(0);
		card.max_output_tokens = entry
			.get("outputTokenLimit")
			.and_then(Value::as_u64)
			.unwrap_or(0);
		cards.insert(card.id.clone(), card);
	}
	Ok(cards.into_values().collect())
}
async fn discover_google_pages(
	provider: &ProviderEntry,
	account: &Account,
	http: &dyn HttpClient,
) -> Result<Vec<ModelCard>, Error> {
	let mut url = Url::parse(&discovery_url(provider, EndpointKind::Google)?).map_err(|error| {
		Error::InvalidUrl {
			provider: provider.id.clone(),
			detail:   Str::from(error.to_string()),
		}
	})?;
	let mut cards = BTreeMap::new();
	let mut seen_tokens = std::collections::BTreeSet::new();
	let mut next_page: Option<String> = None;
	for _ in 0..25 {
		{
			let mut query = url.query_pairs_mut();
			query.clear().append_pair("pageSize", "100");
			if let Some(token) = &next_page {
				query.append_pair("pageToken", token);
			}
		}
		let response = http
			.get_for_account(provider, account, url.as_str())
			.await?;
		if !(200..300).contains(&response.status) {
			return Err(Error::HttpStatus {
				provider: provider.id.clone(),
				status:   response.status,
			});
		}
		for card in parse_google(provider, &response.body)? {
			cards.insert(card.id.clone(), card);
		}
		let payload: Value = serde_json::from_slice(&response.body)
			.map_err(|error| invalid_payload(provider, error))?;
		let token = payload
			.get("nextPageToken")
			.and_then(Value::as_str)
			.map(str::trim)
			.filter(|token| !token.is_empty());
		let Some(token) = token else {
			break;
		};
		if !seen_tokens.insert(token.to_owned()) {
			break;
		}
		next_page = Some(token.to_owned());
	}
	Ok(cards.into_values().collect())
}

fn find_model_array(payload: &Value) -> Option<&[Value]> {
	if let Some(entries) = payload.as_array() {
		return Some(entries);
	}

	for field in ["data", "models", "result", "items"] {
		let Some(candidate) = payload.get(field) else {
			continue;
		};
		if let Some(entries) = find_model_array(candidate) {
			return Some(entries);
		}
	}
	None
}
/// Parses OpenAI Codex's account-scoped model registry response.
///
/// # Errors
///
/// Returns [`Error::InvalidPayload`] when the response has no model array.
pub fn parse_codex_models(provider: &ProviderEntry, body: &[u8]) -> Result<Vec<ModelCard>, Error> {
	let payload: Value =
		serde_json::from_slice(body).map_err(|error| invalid_payload(provider, error))?;
	let entries = find_model_array(&payload)
		.ok_or_else(|| invalid_payload(provider, "missing models/data array"))?;
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

/// Parses Cloud Code Assist/Antigravity `fetchAvailableModels`.
///
/// # Errors
///
/// Returns [`Error::InvalidPayload`] when the response lacks its model map.
pub fn parse_cca_models(provider: &ProviderEntry, body: &[u8]) -> Result<Vec<ModelCard>, Error> {
	let payload: Value =
		serde_json::from_slice(body).map_err(|error| invalid_payload(provider, error))?;
	let models = payload
		.get("models")
		.and_then(Value::as_object)
		.ok_or_else(|| invalid_payload(provider, "missing models object"))?;
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
/// Parses GitLab Duo's `aiChatAvailableModels` GraphQL payload.
///
/// # Errors
/// Returns [`Error::InvalidPayload`] for malformed JSON. A successful GraphQL
/// response with no availability yields an empty source.
pub fn parse_gitlab_duo_models(
	provider: &ProviderEntry,
	body: &[u8],
) -> Result<Vec<ModelCard>, Error> {
	let payload: Value =
		serde_json::from_slice(body).map_err(|error| invalid_payload(provider, error))?;
	let Some(availability) = payload
		.pointer("/data/aiChatAvailableModels")
		.and_then(Value::as_object)
	else {
		return Ok(Vec::new());
	};
	let entries = availability
		.get("selectableModels")
		.and_then(Value::as_array)
		.into_iter()
		.flatten()
		.chain(availability.get("defaultModel"))
		.chain(availability.get("pinnedModel"));
	let mut models = BTreeMap::new();
	for entry in entries {
		let Some(model) = entry
			.get("ref")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
		else {
			continue;
		};
		let name = entry
			.get("name")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
			.unwrap_or(model);
		let mut card = discovered_card(provider, model, name, infer_family(model));
		card.context_window =
			if model.contains("opus") || model.contains("sonnet") || model.contains("gemini") {
				1_000_000
			} else if model.contains("gpt-5") {
				400_000
			} else {
				200_000
			};
		card.max_output_tokens = 0;
		models.insert(card.id.clone(), card);
	}
	Ok(models.into_values().collect())
}

/// Creates a normalized discovered card for a provider-owned protocol.
#[must_use]
pub fn discovered_card(
	provider: &ProviderEntry,
	model: &str,
	name: &str,
	family: &str,
) -> ModelCard {
	let facets = provider.facets.clone();
	let mut inputs = SmallVec::new();
	inputs.push(Modality::Text);
	let mut outputs = SmallVec::new();
	outputs.push(Modality::Text);

	ModelCard {
		id: Str::from(format!("{}/{model}", provider.id)),
		provider: provider.id.clone(),
		model: Str::from(model),
		name: Str::from(name),
		family: Str::from(family),
		facets,
		inputs,
		outputs,
		reasoning: false,
		efforts: SmallVec::new(),
		context_window: 0,
		max_output_tokens: 0,
		pricing: SmallVec::new(),
		availability: Availability::Available,
		source: Source::Discovered,
		blocked_until_ms: 0,
		deprecated: false,
		updated_at_ms: 0,
		props: Props::default(),
		effort_routing: BTreeMap::new(),
		behavior: crate::models::ModelBehavior::default(),
		wire: None,
	}
}

fn infer_family(model: &str) -> &str {
	model
		.split(['/', '-', ':'])
		.find(|part| !part.is_empty())
		.unwrap_or(model)
}

fn invalid_payload(provider: &ProviderEntry, detail: impl std::fmt::Display) -> Error {
	Error::InvalidPayload {
		provider: provider.id.clone(),
		detail:   Str::from(detail.to_string()),
	}
}

#[cfg(test)]
mod tests {
	use std::collections::BTreeMap;

	use parking_lot::Mutex;

	use super::*;
	use crate::{
		compat::Compat,
		provider::{Facet as ProviderFacet, load_providers},
	};

	struct FixtureHttp {
		body:      &'static str,
		requested: Mutex<Option<String>>,
	}

	#[async_trait]
	impl HttpClient for FixtureHttp {
		async fn get(&self, _provider: &ProviderEntry, url: &str) -> Result<HttpResponse, Error> {
			*self.requested.lock() = Some(url.to_owned());
			Ok(HttpResponse { status: 200, body: Bytes::from_static(self.body.as_bytes()) })
		}
	}

	fn provider(id: &str, base_url: &str) -> ProviderEntry {
		let mut facets = SmallVec::new();
		facets.push(ProviderFacet::Chat);
		ProviderEntry {
			id: Str::from(id),
			transport: TransportId::OpenAiChat,
			codex_transport: Default::default(),
			codex_responses_lite: false,
			base_url: Str::from(base_url),
			base_url_overridden: false,
			transport_overridden: false,
			api_version: None,
			fallback_base_urls: SmallVec::new(),
			auth: AuthSpec::None,
			facets,
			headers: BTreeMap::new(),
			compat: Compat::default(),
			mapping: Default::default(),
			oauth_flow: None,
			oauth_auth: None,
			discovery: None,
			pending_facets: SmallVec::new(),
			pending_transport: None,
		}
	}

	#[tokio::test]
	async fn parses_ollama_tags() {
		let provider = provider("ollama", "http://127.0.0.1:11434/v1");
		let http = FixtureHttp {
			body:      r#"{"models":[{"name":"qwen3:8b","model":"qwen3:8b","details":{"family":"qwen3"}}]}"#,
			requested: Mutex::new(None),
		};
		let cards = discover(&provider, &http)
			.await
			.expect("fixture should parse");
		assert_eq!(cards.len(), 1);
		assert_eq!(cards[0].id.as_str(), "ollama/qwen3:8b");
		assert_eq!(cards[0].family.as_str(), "qwen3");
		assert_eq!(cards[0].source, Source::Discovered);
		assert_eq!(http.requested.lock().as_deref(), Some("http://127.0.0.1:11434/api/tags"));
	}

	#[tokio::test]
	async fn parses_openai_models() {
		let provider = provider("vllm", "http://127.0.0.1:8000/v1");
		let http = FixtureHttp {
			body:      r#"{"object":"list","data":[{"id":"Qwen/Qwen3-8B","owned_by":"qwen"}]}"#,
			requested: Mutex::new(None),
		};
		let cards = discover(&provider, &http)
			.await
			.expect("fixture should parse");
		assert_eq!(cards.len(), 1);
		assert_eq!(cards[0].id.as_str(), "vllm/Qwen/Qwen3-8B");
		assert_eq!(cards[0].family.as_str(), "qwen");
		assert_eq!(http.requested.lock().as_deref(), Some("http://127.0.0.1:8000/v1/models"));
	}

	#[test]
	fn copilot_tiers_cap_the_base_and_add_a_priced_wire_alias() {
		let provider = provider("github-copilot", "https://api.githubcopilot.com");
		let cards = parse_openai(
			&provider,
			br#"{"data":[
				{"id":"claude-opus-4.7","name":"Claude Opus 4.7",
				 "capabilities":{"type":"chat","limits":{"max_context_window_tokens":1000000,
				 "max_output_tokens":64000},"supports":{"vision":true}},
				 "billing":{"token_prices":{
					"default":{"context_max":200000,"input_price":500,"output_price":2500,
					 "cache_price":50},
					"long_context":{"context_max":936000,"input_price":700,"output_price":3000,
					 "cache_price":70}}}},
				{"id":"text-embedding-3-small","capabilities":{"type":"embeddings"}}
			]}"#,
		)
		.expect("Copilot model response");
		assert_eq!(cards.len(), 2);
		let base = cards
			.iter()
			.find(|card| card.model == "claude-opus-4.7")
			.expect("base model");
		assert_eq!(base.context_window, 264_000);
		assert_eq!(base.max_output_tokens, 64_000);
		assert!(base.inputs.contains(&Modality::Image));
		assert_eq!(
			base
				.pricing
				.iter()
				.find(|price| price.unit == PriceUnit::MtokInput)
				.map(|price| price.nanos_usd),
			Some(5_000_000_000)
		);

		let variant = cards
			.iter()
			.find(|card| card.model == "claude-opus-4.7-1m")
			.expect("long-context variant");
		assert_eq!(variant.id, "github-copilot/claude-opus-4.7-1m");
		assert_eq!(variant.name, "Claude Opus 4.7 (1M)");
		assert_eq!(variant.context_window, 1_000_000);
		assert_eq!(
			variant
				.effort_routing
				.get(&Effort::Off)
				.map(Str::as_str),
			Some("claude-opus-4.7")
		);
		assert_eq!(
			variant
				.pricing
				.iter()
				.find(|price| price.unit == PriceUnit::MtokInput)
				.map(|price| price.nanos_usd),
			Some(7_000_000_000)
		);
	}

	#[test]
	fn copilot_real_model_id_wins_over_a_synthesized_variant() {
		let provider = provider("github-copilot", "https://api.githubcopilot.com");
		let cards = parse_openai(
			&provider,
			br#"{"data":[
				{"id":"claude-opus-4.6","capabilities":{"type":"chat","limits":{
				 "max_context_window_tokens":1000000,"max_output_tokens":64000}},
				 "billing":{"token_prices":{"default":{"context_max":200000},
				 "long_context":{"context_max":936000}}}},
				{"id":"claude-opus-4.6-1m","name":"Served 1M","capabilities":{"type":"chat",
				 "limits":{"max_context_window_tokens":999000,"max_output_tokens":64000}}}
			]}"#,
		)
		.expect("Copilot model response");
		let served = cards
			.iter()
			.find(|card| card.model == "claude-opus-4.6-1m")
			.expect("served model");
		assert_eq!(served.name, "Served 1M");
		assert_eq!(served.context_window, 999_000);
		assert!(served.effort_routing.is_empty());
	}
	#[test]
	fn codex_parser_filters_hidden_rows_and_keeps_native_limits() {
		let provider = provider("openai-codex", "https://chatgpt.com/backend-api");
		let cards = parse_codex_models(
			&provider,
			br#"{"models":[
				{"slug":"gpt-5.6","display_name":"GPT-5.6","context_window":372000,
				 "supported_reasoning_levels":[{"effort":"high"}],"input_modalities":["text","image"]},
				{"slug":"retired","visibility":"hidden"}
			]}"#,
		)
		.expect("Codex response");
		assert_eq!(cards.len(), 1);
		assert_eq!(cards[0].id, "openai-codex/gpt-5.6");
		assert_eq!(cards[0].context_window, 372_000);
		assert!(cards[0].reasoning);
		assert!(cards[0].inputs.contains(&Modality::Image));
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

	#[test]
	fn gitlab_parser_merges_default_selectable_and_pinned_refs() {
		let provider = provider("gitlab-duo-agent", "https://gitlab.com");
		let cards = parse_gitlab_duo_models(
			&provider,
			br#"{"data":{"aiChatAvailableModels":{
				"defaultModel":{"name":"Sonnet","ref":"claude_sonnet_4_6_vertex"},
				"selectableModels":[{"name":"Opus","ref":"claude_opus_4_8"}],
				"pinnedModel":{"name":"Sonnet pinned","ref":"claude_sonnet_4_6_vertex"}
			}}}"#,
		)
		.expect("GitLab GraphQL fixture");
		assert_eq!(cards.len(), 2);
		assert!(cards.iter().all(|card| card.context_window == 1_000_000));
	}

	#[test]
	fn every_declared_discovery_mode_has_dispatch() {
		let providers =
			load_providers(crate::provider::BUILTIN_PROVIDERS_TOML).expect("built-in providers");
		let mut kinds = BTreeMap::new();
		for provider in providers
			.values()
			.filter(|provider| provider.discovery.is_some())
		{
			assert!(supports(provider), "{} must be discoverable", provider.id);
			let kind = endpoint_kind(provider).expect("declared discovery must dispatch");
			let index = match kind {
				EndpointKind::Ollama => 0,
				EndpointKind::OpenAi => 1,
				EndpointKind::AccountModels => 2,
				EndpointKind::Google => 3,
				EndpointKind::Specialized => 4,
			};
			*kinds.entry(index).or_insert(0usize) += 1;
		}
		assert_eq!(kinds.keys().copied().collect::<Vec<_>>(), [0, 1, 3, 4]);
	}
}
