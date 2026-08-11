//! Provider rows, authentication descriptions, and curated TOML loading.

use std::collections::BTreeMap;

use bon::Builder;
use omp_core::Str;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use super::compat::Compat;

/// Curated provider TOML embedded in this crate.
pub const BUILTIN_PROVIDERS_TOML: &str = include_str!("../providers.toml");

/// Maximum expanded base-URL length accepted by [`expand_base_url`].
pub const MAX_EXPANDED_BASE_URL_LEN: usize = 8 * 1024;
const MAX_BASE_URL_EXPANSIONS: usize = 16;

/// How this row reconciles a source-registry identifier.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum RegistryMapping {
	/// The registry identifier has its own endpoint and authentication policy.
	#[default]
	Concrete,
	/// The identifier is an alternate login or spelling for another row.
	Alias {
		/// Canonical provider identifier.
		target: Str,
		/// Why the source registry keeps the alternate identifier.
		reason: Str,
	},
	/// The source entry moved to a non-inference subsystem.
	Replacement {
		/// Owning subsystem or component.
		component: Str,
		/// Why this is deliberately not selectable as inference.
		reason:    Str,
	},
}

/// Live model-listing convention advertised by a provider.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DiscoveryKind {
	/// OpenAI-compatible `GET /models`.
	OpenAiModels,
	/// Google Generative Language `GET /models`.
	GoogleModels,
	/// Ollama's native `GET /api/tags`.
	OllamaTags,
	/// A provider-specific account model listing.
	AccountModels,
	/// Discovery is owned by a specialized transport.
	Specialized,
}

/// Account model-discovery policy retained from the source registry.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoverySpec {
	/// Listing protocol.
	pub kind:          DiscoveryKind,
	/// Human-readable account source shown by model discovery.
	pub label:         Str,
	/// Whether a successful account listing replaces, rather than augments, the
	/// bundled model set.
	#[serde(default)]
	pub authoritative: bool,
}

/// Request placement used specifically for broker-minted login credentials.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum CredentialPlacement {
	/// `Authorization: Bearer`.
	Bearer,
	/// A provider-defined header.
	Header {
		/// Header name.
		name: Str,
	},
	/// A provider-defined query parameter.
	Query {
		/// Query parameter name.
		param: Str,
	},
}
/// Preferred wire path for ChatGPT Codex Responses requests.
///
/// This is catalog data rather than an application-provider name check. The
/// HTTP path remains the replay-safe baseline for rows that do not opt in.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CodexTransportPreference {
	/// Use HTTP SSE only.
	#[default]
	HttpOnly,
	/// Attempt Responses-Lite WebSocket execution before replay-safe HTTP SSE.
	WebsocketPreferred,
}

/// One provider endpoint and its data-selected wire behavior.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct ProviderEntry {
	/// Stable catalog identifier.
	pub id:                   Str,
	/// Wire transport implemented by the endpoint.
	pub transport:            TransportId,
	/// Preferred Codex wire path; ignored by non-Codex transports.
	#[builder(default)]
	pub codex_transport:      CodexTransportPreference,
	/// Provider-level default for the Codex Responses Lite request shape.
	#[builder(default)]
	pub codex_responses_lite: bool,
	/// Base URL, optionally containing the bounded placeholders accepted by
	/// [`BaseUrlVars`].
	pub base_url:             Str,
	/// Whether a user or project overlay explicitly selected [`Self::base_url`].
	///
	/// Server-only routing provenance; built-in model wire routes may override
	/// only fields that were not explicitly configured.
	#[builder(default)]
	pub base_url_overridden:  bool,
	/// Whether a user or project overlay explicitly selected
	/// [`Self::transport`].
	#[builder(default)]
	pub transport_overridden: bool,
	/// Default API version appended by transports that require a version query.
	pub api_version:          Option<Str>,
	/// Ordered failover base URLs attempted after [`Self::base_url`].
	///
	/// This is transport-agnostic catalog data; Cloud Code Assist is its first
	/// user.
	#[builder(default)]
	pub fallback_base_urls:   SmallVec<Str, 2>,
	/// Credential injection description.
	pub auth:                 AuthSpec,
	/// Facets exposed at this endpoint.
	pub facets:               SmallVec<Facet, 4>,
	/// Static request headers.
	pub headers:              BTreeMap<Str, Str>,
	/// Data-like deviations from the transport defaults.
	pub compat:               Compat,
	/// Source-registry reconciliation for this identifier.
	#[builder(default)]
	pub mapping:              RegistryMapping,
	/// Login flow usable in addition to the request authentication policy.
	///
	/// This is separate from [`AuthSpec::OAuth`] because providers such as
	/// Anthropic accept both environment API keys and broker-minted OAuth
	/// credentials.
	pub oauth_flow:           Option<Str>,
	/// Credential placement for [`Self::oauth_flow`] when it differs from
	/// [`Self::auth`].
	pub oauth_auth:           Option<CredentialPlacement>,
	/// Live account model-discovery policy.
	pub discovery:            Option<DiscoverySpec>,
	/// Facets known upstream but withheld until their distinct transport exists.
	#[builder(default)]
	pub pending_facets:       SmallVec<Facet, 2>,
	/// Upstream wire name for [`Self::pending_facets`].
	pub pending_transport:    Option<Str>,
}

/// Supported provider wire transports.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TransportId {
	/// Anthropic Messages API.
	AnthropicMessages,
	/// Anthropic Messages adapted to AWS Bedrock
	/// `InvokeModelWithResponseStream`/EventStream.
	AnthropicBedrock,
	/// Amazon Bedrock model-independent `ConverseStream` API over AWS
	/// EventStream.
	BedrockConverse,
	/// Anthropic Messages adapted to Vertex `streamRawPredict`.
	AnthropicVertex,
	/// `OpenAI` Chat Completions API.
	OpenAiChat,
	/// `OpenAI` Responses API.
	OpenAiResponses,
	/// ChatGPT subscription Codex Responses transport.
	OpenAiCodex,
	/// Public Google Generative Language API.
	GoogleGenAi,
	/// Google Vertex AI Gemini API.
	GoogleVertex,
	/// Google Cloud Code Assist API.
	GoogleCca,
	/// Ollama's native `/api/chat` NDJSON protocol.
	OllamaChat,
	/// Cursor's Connect/gRPC agent transport.
	Cursor,
	/// Devin's Connect server-streaming transport.
	Devin,
	/// GitLab Duo Workflow authenticated WebSocket agent tunnel.
	#[serde(rename = "gitlab-duo-workflow")]
	GitLabDuoWorkflow,
	/// OMP federation protocol.
	Omp,
	/// In-process embedded inference.
	Embedded,
}

/// How credentials are obtained and placed on an outbound request.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum AuthSpec {
	/// No credential is required.
	#[default]
	None,
	/// A bearer token read from the first populated environment variable.
	Bearer {
		/// Environment variables in priority order.
		env: SmallVec<Str, 2>,
	},
	/// An optional bearer token, used by local servers that accept authenticated
	/// and unauthenticated requests.
	OptionalBearer {
		/// Environment variables in priority order.
		env: SmallVec<Str, 2>,
	},
	/// A token sent in a custom header.
	Header {
		/// Header name.
		name: Str,
		/// Environment variables in priority order.
		env:  SmallVec<Str, 2>,
	},
	/// A token sent in a query parameter.
	Query {
		/// Query parameter name.
		param: Str,
		/// Environment variables in priority order.
		env:   SmallVec<Str, 2>,
	},
	/// AWS Signature Version 4 request signing.
	AwsSigV4,
	/// Google Application Default Credentials, with optional API-key and route
	/// metadata fallbacks.
	GoogleAdc {
		/// API-key environment variables in priority order.
		api_key_env:  SmallVec<Str, 2>,
		/// Project environment variables in priority order.
		project_env:  SmallVec<Str, 3>,
		/// Location environment variables in priority order.
		location_env: SmallVec<Str, 3>,
	},
	/// A broker-managed OAuth flow.
	#[serde(rename = "oauth")]
	OAuth {
		/// Flow identifier in the OAuth parameter catalog.
		flow: Str,
	},
}

/// Inference capabilities independently exposed by a provider.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Facet {
	/// Conversational generation.
	Chat,
	/// Text embeddings.
	Embeddings,
	/// Text reranking.
	Rerank,
	/// Speech synthesis.
	AudioSpeech,
	/// Audio transcription.
	AudioTranscription,
	/// Image generation.
	ImageGeneration,
	/// Asynchronous video generation.
	VideoGeneration,
}

/// Loaded provider rows indexed by stable provider id.
pub type ProviderCatalog = BTreeMap<Str, ProviderEntry>;

/// Failure while parsing a provider catalog.
#[derive(Debug, thiserror::Error)]
pub enum ProviderLoadError {
	/// TOML deserialization failed.
	#[error("failed to deserialize provider document: {0}")]
	Toml(#[from] toml::de::Error),
}

/// Values accepted by the deliberately small base-URL template expander.
#[non_exhaustive]
#[derive(Builder, Clone, Copy, Debug, Default)]
pub struct BaseUrlVars<'a> {
	/// Value for the `{region}` placeholder.
	pub region:     Option<&'a str>,
	/// Value for the `{location}` placeholder.
	pub location:   Option<&'a str>,
	/// Value for the `{project}` placeholder.
	pub project:    Option<&'a str>,
	/// Value for the `{deployment}` placeholder.
	pub deployment: Option<&'a str>,
	/// Value for the `{model}` placeholder.
	pub model:      Option<&'a str>,
	/// Value for the `{account}` placeholder.
	pub account:    Option<&'a str>,
	/// Value for the `{gateway}` placeholder.
	pub gateway:    Option<&'a str>,
}

/// Failure while expanding a bounded base-URL template.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum BaseUrlTemplateError {
	/// The template contains an unsupported placeholder syntax.
	#[error("unsupported placeholder syntax in template: {0}")]
	UnsupportedPlaceholder(Str),
	/// Unclosed placeholder delimiter `{`.
	#[error("unclosed placeholder bracket at byte index {0}")]
	UnclosedBracket(usize),
	/// A required variable was not supplied in [`BaseUrlVars`].
	#[error("missing required base-URL variable `{0}`")]
	MissingVar(&'static str),
	/// Maximum expansion pass limit exceeded.
	#[error("exceeded maximum template substitutions ({MAX_BASE_URL_EXPANSIONS})")]
	TooManyExpansions,
	/// Expanded URL exceeded maximum permitted size.
	#[error("expanded URL exceeds maximum length ({MAX_EXPANDED_BASE_URL_LEN} bytes)")]
	UrlTooLong,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderDocument {
	providers: BTreeMap<Str, ProviderConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderConfig {
	transport:            TransportId,
	base_url:             Str,
	#[serde(default)]
	api_version:          Option<Str>,
	#[serde(default)]
	codex_transport:      CodexTransportPreference,
	#[serde(default)]
	codex_responses_lite: bool,
	#[serde(default)]
	fallback_base_urls:   SmallVec<Str, 2>,
	#[serde(default)]
	auth:                 AuthSpec,
	#[serde(default)]
	facets:               SmallVec<Facet, 4>,
	#[serde(default)]
	headers:              BTreeMap<Str, Str>,
	#[serde(default)]
	compat:               Compat,
	#[serde(default)]
	mapping:              RegistryMapping,
	#[serde(default)]
	oauth_flow:           Option<Str>,
	#[serde(default)]
	oauth_auth:           Option<CredentialPlacement>,
	#[serde(default)]
	discovery:            Option<DiscoverySpec>,
	#[serde(default)]
	pending_facets:       SmallVec<Facet, 2>,
	#[serde(default)]
	pending_transport:    Option<Str>,
}

impl ProviderConfig {
	fn with_id(self, id: Str) -> ProviderEntry {
		ProviderEntry {
			id,
			transport: self.transport,
			codex_transport: self.codex_transport,
			codex_responses_lite: self.codex_responses_lite,
			base_url: self.base_url,
			base_url_overridden: false,
			transport_overridden: false,
			api_version: self.api_version,
			fallback_base_urls: self.fallback_base_urls,
			auth: self.auth,
			facets: self.facets,
			headers: self.headers,
			oauth_auth: self.oauth_auth,
			compat: self.compat,
			mapping: self.mapping,
			oauth_flow: self.oauth_flow,
			discovery: self.discovery,
			pending_facets: self.pending_facets,
			pending_transport: self.pending_transport,
		}
	}
}

/// Parses a complete provider TOML document.
///
/// Unknown keys at the document, provider, auth, and compat levels are rejected
/// by serde so configuration typos retain TOML's key path and source span.
pub fn load_providers(source: &str) -> Result<ProviderCatalog, ProviderLoadError> {
	let document: ProviderDocument = toml::from_str(source)?;
	Ok(document
		.providers
		.into_iter()
		.map(|(id, config)| (id.clone(), config.with_id(id)))
		.collect())
}

/// Loads the curated provider catalog embedded in this crate.
pub fn load_builtin() -> Result<ProviderCatalog, ProviderLoadError> {
	load_providers(BUILTIN_PROVIDERS_TOML)
}

/// Expands the seven permitted base-URL placeholders without a template engine.
///
/// Expansion is bounded to 16 substitutions and an 8 KiB output. Values are
/// inserted verbatim, so callers must provide URL path-safe identifiers. Any
/// other placeholder is rejected rather than silently passed through.
pub fn expand_base_url(
	template: &str,
	vars: BaseUrlVars<'_>,
) -> Result<Str, BaseUrlTemplateError> {
	if !template.contains('{') {
		if template.len() > MAX_EXPANDED_BASE_URL_LEN {
			return Err(BaseUrlTemplateError::UrlTooLong);
		}
		return Ok(Str::new(template));
	}

	let mut result = String::with_capacity(template.len() + 32);
	let mut cursor = 0;
	let bytes = template.as_bytes();
	let mut replacements = 0;

	while cursor < bytes.len() {
		let open = if let Some(pos) = bytes[cursor..].iter().position(|&b| b == b'{') {
			cursor + pos
		} else {
			result.push_str(&template[cursor..]);
			break;
		};

		result.push_str(&template[cursor..open]);

		let close = match bytes[open..].iter().position(|&b| b == b'}') {
			Some(pos) => open + pos,
			None => return Err(BaseUrlTemplateError::UnclosedBracket(open)),
		};

		let key = &template[open + 1..close];
		replacements += 1;
		if replacements > MAX_BASE_URL_EXPANSIONS {
			return Err(BaseUrlTemplateError::TooManyExpansions);
		}

		let val = match key {
			"region" => vars
				.region
				.ok_or(BaseUrlTemplateError::MissingVar("region"))?,
			"location" => vars
				.location
				.or(vars.region)
				.ok_or(BaseUrlTemplateError::MissingVar("location"))?,
			"project" => vars
				.project
				.ok_or(BaseUrlTemplateError::MissingVar("project"))?,
			"deployment" => vars
				.deployment
				.ok_or(BaseUrlTemplateError::MissingVar("deployment"))?,
			"model" => vars
				.model
				.ok_or(BaseUrlTemplateError::MissingVar("model"))?,
			"account" => vars
				.account
				.ok_or(BaseUrlTemplateError::MissingVar("account"))?,
			"gateway" => vars
				.gateway
				.ok_or(BaseUrlTemplateError::MissingVar("gateway"))?,
			_ => {
				return Err(BaseUrlTemplateError::UnsupportedPlaceholder(Str::new(
					&template[open..=close],
				)));
			},
		};

		result.push_str(val);
		if result.len() > MAX_EXPANDED_BASE_URL_LEN {
			return Err(BaseUrlTemplateError::UrlTooLong);
		}

		cursor = close + 1;
	}

	if result.len() > MAX_EXPANDED_BASE_URL_LEN {
		return Err(BaseUrlTemplateError::UrlTooLong);
	}

	Ok(Str::new(result))
}

#[cfg(test)]
mod tests {
	use omp_core::Str;
	use smallvec::smallvec;

	use super::{AuthSpec, CodexTransportPreference, load_builtin};

	#[test]
	fn auth_spec_wire_names_round_trip() {
		let cases = [
			(AuthSpec::None, "none"),
			(AuthSpec::Bearer { env: smallvec![Str::new_static("TOKEN")] }, "bearer"),
			(
				AuthSpec::OptionalBearer { env: smallvec![Str::new_static("TOKEN")] },
				"optional-bearer",
			),
			(
				AuthSpec::Header {
					name: Str::new_static("x-api-key"),
					env:  smallvec![Str::new_static("TOKEN")],
				},
				"header",
			),
			(
				AuthSpec::Query {
					param: Str::new_static("key"),
					env:   smallvec![Str::new_static("TOKEN")],
				},
				"query",
			),
			(AuthSpec::AwsSigV4, "aws-sig-v4"),
			(
				AuthSpec::GoogleAdc {
					api_key_env:  smallvec![Str::new_static("GOOGLE_CLOUD_API_KEY")],
					project_env:  smallvec![Str::new_static("GOOGLE_CLOUD_PROJECT")],
					location_env: smallvec![Str::new_static("GOOGLE_VERTEX_LOCATION")],
				},
				"google-adc",
			),
			(AuthSpec::OAuth { flow: Str::new_static("test-flow") }, "oauth"),
		];

		for (spec, wire_name) in cases {
			let encoded = serde_json::to_value(&spec).expect("auth spec serializes");
			assert_eq!(encoded["type"], wire_name);
			let decoded: AuthSpec =
				serde_json::from_value(encoded).expect("serialized auth spec deserializes");
			assert_eq!(decoded, spec);
		}
	}

	#[test]
	fn builtin_codex_rows_select_websocket_transport_with_http_default_elsewhere() {
		let providers = load_builtin().expect("built-in providers parse");
		assert_eq!(
			providers["openai-codex"].codex_transport,
			CodexTransportPreference::WebsocketPreferred
		);
		assert!(!providers["openai-codex"].codex_responses_lite);
		assert_eq!(providers["openai"].codex_transport, CodexTransportPreference::HttpOnly);
		assert!(!providers["openai"].codex_responses_lite);
	}
}
