//! Live model discovery, split along the credential boundary.
//!
//! - [`DiscoveryHttp`] is implemented once by the runtime. It selects an
//!   account's credential and injects it during dispatch, so no discovery
//!   protocol ever sees credential bytes.
//! - [`DiscoveryProtocol`] is implemented by each provider's own `omp-llm-*`
//!   crate, keeping endpoint choice, headers, request bodies, and payload
//!   decoding beside the wire protocol they belong to.
//!
//! [`Discovery`] joins the two. Listing conventions shared by many providers
//! (`OpenAI` `GET /models`, Ollama tags, Google pagination) stay in this module
//! and are selected by [`DiscoveryKind`]; a provider declaring
//! [`DiscoveryKind::Specialized`] is dispatched to the [`DiscoveryProtocol`]
//! registered for its [`TransportId`].
//!
//! Neither trait is a "transport" in this workspace's sense: a [`TransportId`]
//! names a provider's wire protocol, and specialized discovery is keyed by one.
//!
//! `ollama`, `vllm`, `lm-studio`, and `litellm` are deliberately absent from
//! the bundled catalog: their model sets and localhost endpoints exist only at
//! runtime, so discovery is required rather than an optional enhancement.

use std::{
	collections::BTreeMap,
	fmt::{self, Display},
	sync::Arc,
	time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use http::{Method, Request, header::ACCEPT};
use omp_core::{Str, fmts};
use omp_llm_types::{Effort, Props};
use serde_json::Value;
use smallvec::SmallVec;
use url::Url;

use crate::{
	models::{Availability, Modality, ModelCard, Price, PriceUnit, Source},
	provider::{AuthSpec, DiscoveryKind, ProviderEntry, TransportId},
};

/// Hard deadline for an OpenAI-compatible `/models` request.
pub const DEFAULT_OPENAI_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);

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

	/// Returns whether the provider answered with a 2xx status.
	#[must_use]
	pub const fn is_success(&self) -> bool {
		self.status >= 200 && self.status < 300
	}

	/// Borrows the body of a successful response.
	///
	/// # Errors
	///
	/// Returns [`Error::HttpStatus`] for any non-2xx status.
	pub fn ensure_success(&self, provider: &ProviderEntry) -> Result<&Bytes, Error> {
		if self.is_success() {
			return Ok(&self.body);
		}
		Err(Error::status(provider, self.status))
	}
}
/// Non-secret identity of one credential/account discovery source.
///
/// `key` is opaque to the catalog and is passed back to [`Transport`] for
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
	/// Cloud region this source is scoped to, when the runtime resolved one.
	///
	/// Deployment facts like an AWS region are read by the runtime, never by a
	/// [`DiscoveryProtocol`], so protocols stay free of environment lookups.
	pub region:          Option<Str>,
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
			region:          None,
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
	pub fn with_scope(mut self, organization_id: Option<Str>, project_id: Option<Str>) -> Self {
		self.organization_id = organization_id;
		self.project_id = project_id;
		self
	}

	/// Attaches the cloud region the runtime resolved for this source.
	#[must_use]
	pub fn with_region(mut self, region: Option<Str>) -> Self {
		self.region = region;
		self
	}

	/// Returns the provider-wide source used when account enumeration is not
	/// supported by the discovery executor.
	#[must_use]
	pub const fn provider_default() -> Self {
		Self {
			key:             Str::new_static(""),
			label:           Str::new_static("provider"),
			account_id:      None,
			organization_id: None,
			project_id:      None,
			region:          None,
		}
	}
}

/// Authenticated HTTP execution for model discovery, owned by the runtime.
///
/// The provider and opaque account arguments are deliberate: implementations
/// select and inject a sealed credential, so neither this crate nor any
/// [`DiscoveryProtocol`] receives or stores credential bytes.
#[async_trait]
pub trait DiscoveryHttp: Send + Sync {
	/// Lists the client-safe credential/account sources visible to `provider`.
	///
	/// Transports without account enumeration keep the single provider-wide
	/// source.
	async fn accounts(&self, _provider: &ProviderEntry) -> Result<Vec<Account>, Error> {
		Ok(vec![Account::provider_default()])
	}

	/// Executes `request` with exactly `account`'s credential, never a
	/// sibling's.
	///
	/// Headers already present on `request` win over `provider.headers`. A
	/// request carrying the [`SealedBody`] extension has its body written
	/// inside its credential boundary.
	async fn execute(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		request: Request<Bytes>,
	) -> Result<HttpResponse, Error>;

	/// Executes a JSON `GET` for `account`.
	///
	/// # Errors
	///
	/// Returns [`Error::Transport`] for an invalid URL or a failed dispatch.
	async fn get(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		url: &str,
	) -> Result<HttpResponse, Error> {
		let request = Request::builder()
			.method(Method::GET)
			.uri(url)
			.header(ACCEPT, "application/json")
			.body(Bytes::new())
			.map_err(Error::transport)?;
		self.execute(provider, account, request).await
	}
}

/// Request extension marking a protocol that carries its credential inside the
/// request body rather than a header.
///
/// The protocol leaves the body empty and the [`Transport`] fills it within its
/// credential boundary. Devin's protobuf discovery is the only user today.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SealedBody;

/// A provider-owned model discovery protocol.
///
/// Implemented inside the provider's own `omp-llm-*` crate and registered with
/// [`Discovery::with`], so teaching the runtime a new listing protocol never
/// edits this crate or the application. Implementations issue requests through
/// the injected [`Transport`] and never handle credentials.
///
/// ```ignore
/// struct CursorDiscovery;
///
/// #[async_trait]
/// impl DiscoveryProtocol for CursorDiscovery {
///    fn transports(&self) -> &'static [TransportId] {
///       &[TransportId::Cursor]
///    }
///
///    async fn discover(
///       &self,
///       provider: &ProviderEntry,
///       account: &Account,
///       http: &dyn DiscoveryHttp,
///    ) -> Result<Vec<ModelCard>, Error> {
///       // build a request, hand it to `http`, decode the reply
///    }
/// }
/// ```
#[async_trait]
pub trait DiscoveryProtocol: Send + Sync {
	/// Transports whose listing protocol this implementation speaks.
	///
	/// Discovery is keyed by transport rather than provider id: every
	/// `providers.toml` row naming one of these transports and declaring
	/// [`DiscoveryKind::Specialized`] is served without a code change.
	fn transports(&self) -> &'static [TransportId];

	/// Lists the models visible to exactly one account.
	///
	/// # Errors
	///
	/// Returns the transport, status, or payload failure raised by the
	/// provider's own listing protocol.
	async fn discover(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		http: &dyn DiscoveryHttp,
	) -> Result<Vec<ModelCard>, Error>;
}

/// The runtime's discovery entry point: one authenticated [`Transport`] plus
/// the [`DiscoveryProtocol`] protocols registered for specialized providers.
#[derive(Clone)]
pub struct Discovery {
	http:        Arc<dyn DiscoveryHttp>,
	specialized: SmallVec<&'static dyn DiscoveryProtocol, 8>,
}

impl Discovery {
	/// Creates discovery over `transport` with no specialized protocols.
	#[must_use]
	pub fn new(http: Arc<dyn DiscoveryHttp>) -> Self {
		Self { http, specialized: SmallVec::new() }
	}

	/// Registers one provider-owned protocol.
	#[must_use]
	pub fn with(mut self, protocol: &'static dyn DiscoveryProtocol) -> Self {
		self.specialized.push(protocol);
		self
	}

	/// Returns the authenticated HTTP executor shared by every protocol.
	#[must_use]
	pub fn http(&self) -> &dyn DiscoveryHttp {
		self.http.as_ref()
	}

	/// Lists the client-safe account sources visible to `provider`.
	///
	/// # Errors
	///
	/// Returns the executor's account-enumeration failure.
	pub async fn accounts(&self, provider: &ProviderEntry) -> Result<Vec<Account>, Error> {
		self.http.accounts(provider).await
	}

	/// Discovers models through the provider-wide default credential source.
	///
	/// Account-aware runtimes should enumerate [`Self::accounts`] and call
	/// [`Self::discover`] once per source.
	///
	/// # Errors
	///
	/// Returns an error for unsupported providers, invalid endpoints, transport
	/// or HTTP failures, and malformed payloads.
	pub async fn discover_provider(
		&self,
		provider: &ProviderEntry,
	) -> Result<Vec<ModelCard>, Error> {
		self.discover(provider, &Account::provider_default()).await
	}

	/// Discovers the models visible to exactly one non-secret account source.
	///
	/// # Errors
	///
	/// Returns [`Error::UnregisteredProtocol`] when the provider declares
	/// [`DiscoveryKind::Specialized`] and no matching [`DiscoveryProtocol`] is
	/// registered, plus any transport, status, or payload failure.
	pub async fn discover(
		&self,
		provider: &ProviderEntry,
		account: &Account,
	) -> Result<Vec<ModelCard>, Error> {
		let http = self.http.as_ref();
		let kind = endpoint_kind(provider)?;
		match kind {
			EndpointKind::Specialized => {
				self
					.protocol(provider.transport)
					.ok_or_else(|| Error::UnregisteredProtocol {
						provider:  provider.id.clone(),
						transport: provider.transport,
					})?
					.discover(provider, account, http)
					.await
			},
			EndpointKind::Google => discover_google_pages(provider, account, http).await,
			EndpointKind::OpenAi => {
				discover_openai_with_timeout(
					provider,
					account,
					http,
					DEFAULT_OPENAI_DISCOVERY_TIMEOUT,
				)
				.await
			},
			EndpointKind::Ollama | EndpointKind::AccountModels => {
				let url = discovery_url(provider, kind)?;
				let response = http.get(provider, account, &url).await?;
				let body = response.ensure_success(provider)?;
				if matches!(kind, EndpointKind::Ollama) {
					parse_ollama_tags(provider, body)
				} else {
					parse_openai(provider, body)
				}
			},
		}
	}

	/// Returns whether a protocol is registered for `transport`.
	///
	/// A provider declaring [`DiscoveryKind::Specialized`] on an unregistered
	/// transport fails at listing time with [`Error::UnregisteredProtocol`];
	/// this lets a runtime assert its coverage up front instead.

	#[must_use]
	pub fn serves(&self, transport: TransportId) -> bool {
		self.protocol(transport).is_some()
	}

	fn protocol(&self, transport: TransportId) -> Option<&'static dyn DiscoveryProtocol> {
		self
			.specialized
			.iter()
			.copied()
			.find(|protocol| protocol.transports().contains(&transport))
	}
}
async fn discover_openai_with_timeout(
	provider: &ProviderEntry,
	account: &Account,
	http: &dyn DiscoveryHttp,
	timeout: Duration,
) -> Result<Vec<ModelCard>, Error> {
	let url = discovery_url(provider, EndpointKind::OpenAi)?;
	let response = tokio::time::timeout(timeout, http.get(provider, account, &url))
		.await
		.map_err(|_| Error::Timeout(provider.id.clone()))??;
	parse_openai(provider, response.ensure_success(provider)?)
}

impl fmt::Debug for Discovery {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("Discovery")
			.field(
				"transports",
				&self
					.specialized
					.iter()
					.flat_map(|protocol| protocol.transports().iter().copied())
					.collect::<Vec<_>>(),
			)
			.finish_non_exhaustive()
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
	/// The provider did not complete model discovery before the default deadline.
	#[error("model discovery timed out for {0}")]
	Timeout(Str),
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
	/// The provider declares a specialized protocol but the runtime registered
	/// no [`DiscoveryProtocol`] for its transport.
	#[error("no discovery protocol is registered for {provider} (transport {transport:?})")]
	UnregisteredProtocol {
		/// Provider id.
		provider:  Str,
		/// Transport the provider declares.
		transport: TransportId,
	},
}

impl Error {
	/// Wraps a transport, URI, or request-construction failure.
	#[must_use]
	pub fn transport(error: impl Display) -> Self {
		Self::Transport(Str::from(error.to_string()))
	}

	/// Reports a payload from `provider` that its protocol cannot decode.
	#[must_use]
	pub fn payload(provider: &ProviderEntry, detail: impl Display) -> Self {
		Self::InvalidPayload {
			provider: provider.id.clone(),
			detail:   Str::from(detail.to_string()),
		}
	}

	/// Reports a non-success discovery status from `provider`.
	#[must_use]
	pub fn status(provider: &ProviderEntry, status: u16) -> Self {
		Self::HttpStatus { provider: provider.id.clone(), status }
	}
}

/// Returns whether `provider` declares or infers a live-listing convention.
#[must_use]
pub fn supports(provider: &ProviderEntry) -> bool {
	provider.discovery.is_some()
		|| matches!(provider.id.as_str(), "ollama" | "vllm" | "lm-studio" | "litellm")
		|| ((is_openai_compatible(provider) || provider.transport == TransportId::GoogleGenAi)
			&& !matches!(&provider.auth, AuthSpec::None))
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

/// Parses Ollama's native `GET /api/tags` listing.
///
/// Shared by the bundled local-Ollama convention and by Ollama Cloud's
/// provider-owned protocol.
///
/// # Errors
///
/// Returns [`Error::InvalidPayload`] when the response has no models array.
pub fn parse_ollama_tags(provider: &ProviderEntry, body: &[u8]) -> Result<Vec<ModelCard>, Error> {
	let payload: Value =
		serde_json::from_slice(body).map_err(|error| Error::payload(provider, error))?;
	let entries = payload
		.get("models")
		.and_then(Value::as_array)
		.ok_or_else(|| Error::payload(provider, "missing models array"))?;
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
		serde_json::from_slice(body).map_err(|error| Error::payload(provider, error))?;
	let entries = find_model_array(&payload)
		.ok_or_else(|| Error::payload(provider, "missing data/models/result/items array"))?;
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
	variant.model = fmts!("{wire_model}-1m");
	variant.id = fmts!("{}/{}", card.provider, variant.model);
	variant.name = fmts!("{} (1M)", card.name);
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
		serde_json::from_slice(body).map_err(|error| Error::payload(provider, error))?;
	let entries = payload
		.get("models")
		.and_then(Value::as_array)
		.ok_or_else(|| Error::payload(provider, "missing models array"))?;
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
	http: &dyn DiscoveryHttp,
) -> Result<Vec<ModelCard>, Error> {
	let mut url = Url::parse(&discovery_url(provider, EndpointKind::Google)?).map_err(|error| {
		Error::InvalidUrl { provider: provider.id.clone(), detail: Str::from(error.to_string()) }
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
		let response = http.get(provider, account, url.as_str()).await?;
		for card in parse_google(provider, response.ensure_success(provider)?)? {
			cards.insert(card.id.clone(), card);
		}
		let payload: Value =
			serde_json::from_slice(&response.body).map_err(|error| Error::payload(provider, error))?;
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

/// Finds the first array of model entries in a provider payload.
///
/// Accepts a bare array or the common `data`/`models`/`result`/`items`
/// envelopes, searched recursively.
#[must_use]
pub fn find_model_array(payload: &Value) -> Option<&[Value]> {
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
		pricing_tiers: SmallVec::new(),
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

/// Infers a model family from a model id when the provider names none.
#[must_use]
pub fn infer_family(model: &str) -> &str {
	model
		.split(['/', '-', ':'])
		.find(|part| !part.is_empty())
		.unwrap_or(model)
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
	impl DiscoveryHttp for FixtureHttp {
		async fn execute(
			&self,
			_provider: &ProviderEntry,
			_account: &Account,
			request: Request<Bytes>,
		) -> Result<HttpResponse, Error> {
			*self.requested.lock() = Some(request.uri().to_string());
			Ok(HttpResponse::new(200, Bytes::from_static(self.body.as_bytes())))
		}
	}

	struct StallingHttp;

	#[async_trait]
	impl DiscoveryHttp for StallingHttp {
		async fn execute(
			&self,
			_provider: &ProviderEntry,
			_account: &Account,
			_request: Request<Bytes>,
		) -> Result<HttpResponse, Error> {
			std::future::pending().await
		}
	}

	/// Stands in for a provider crate's own `DiscoveryProtocol` implementation.
	struct FixtureProtocol;

	#[async_trait]
	impl DiscoveryProtocol for FixtureProtocol {
		fn transports(&self) -> &'static [TransportId] {
			&[TransportId::Cursor]
		}

		async fn discover(
			&self,
			provider: &ProviderEntry,
			_account: &Account,
			_http: &dyn DiscoveryHttp,
		) -> Result<Vec<ModelCard>, Error> {
			Ok(vec![discovered_card(provider, "only", "Only", "only")])
		}
	}

	static FIXTURE_PROTOCOL: FixtureProtocol = FixtureProtocol;

	fn specialized(id: &str, transport: TransportId) -> ProviderEntry {
		let mut entry = provider(id, "https://example.invalid");
		entry.transport = transport;
		entry.discovery = Some(crate::provider::DiscoverySpec {
			kind:          DiscoveryKind::Specialized,
			label:         Str::new_static("fixture"),
			authoritative: false,
		});
		entry
	}

	fn fixture_discovery() -> Discovery {
		Discovery::new(Arc::new(FixtureHttp { body: "{}", requested: Mutex::new(None) }))
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
	async fn openai_discovery_has_a_hard_deadline() {
		let provider = provider("openai-compatible", "https://stall.example/v1");
		let error = discover_openai_with_timeout(
			&provider,
			&Account::provider_default(),
			&StallingHttp,
			Duration::from_millis(1),
		)
		.await
		.expect_err("stalled discovery must time out");
		assert!(matches!(error, Error::Timeout(id) if id == provider.id));
		assert_eq!(DEFAULT_OPENAI_DISCOVERY_TIMEOUT, Duration::from_secs(10));
	}

	#[tokio::test]
	async fn parses_ollama_tags() {
		let provider = provider("ollama", "http://127.0.0.1:11434/v1");
		let http = Arc::new(FixtureHttp {
			body:      r#"{"models":[{"name":"qwen3:8b","model":"qwen3:8b","details":{"family":"qwen3"}}]}"#,
			requested: Mutex::new(None),
		});
		let cards = Discovery::new(http.clone())
			.discover_provider(&provider)
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
		let http = Arc::new(FixtureHttp {
			body:      r#"{"object":"list","data":[{"id":"Qwen/Qwen3-8B","owned_by":"qwen"}]}"#,
			requested: Mutex::new(None),
		});
		let cards = Discovery::new(http.clone())
			.discover_provider(&provider)
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
			variant.effort_routing.get(&Effort::Off).map(Str::as_str),
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

	#[tokio::test]
	async fn specialized_dispatch_selects_a_protocol_by_transport() {
		let discovery = fixture_discovery().with(&FIXTURE_PROTOCOL);
		let cards = discovery
			.discover_provider(&specialized("cursor", TransportId::Cursor))
			.await
			.expect("registered protocol should serve its transport");
		assert_eq!(cards.len(), 1);
		assert_eq!(cards[0].id.as_str(), "cursor/only");
	}

	#[tokio::test]
	async fn a_new_provider_row_reuses_a_registered_transport() {
		// Providers are data: a second row naming an already-registered
		// transport must discover without any code change.
		let discovery = fixture_discovery().with(&FIXTURE_PROTOCOL);
		let cards = discovery
			.discover_provider(&specialized("cursor-enterprise", TransportId::Cursor))
			.await
			.expect("an unseen provider id on a registered transport must dispatch");
		assert_eq!(cards[0].id.as_str(), "cursor-enterprise/only");
	}

	#[tokio::test]
	async fn an_unregistered_transport_names_itself() {
		let error = fixture_discovery()
			.discover_provider(&specialized("devin", TransportId::Devin))
			.await
			.expect_err("an unregistered specialized transport must not silently succeed");
		assert!(
			matches!(
				&error,
				Error::UnregisteredProtocol { provider, transport }
					if provider == "devin" && *transport == TransportId::Devin
			),
			"{error:?}"
		);
	}
}
