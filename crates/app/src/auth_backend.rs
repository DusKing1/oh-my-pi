//! Production credential-broker construction for the command-line application.

use std::{
	fmt::Display,
	path::Path,
	sync::Arc,
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::BoxFuture;
use http::{Method, Request, Response, header::ACCEPT};
use http_body_util::{BodyExt as _, Full};
use hyper::body::Incoming;
use omp_core::SmolStr;
use omp_llm_broker::{
	BrokerCliBackend,
	oauth::{
		HttpClient as OAuthHttpClient, HttpError, HttpFuture, HttpRequest,
		HttpResponse as OAuthHttpResponse, OAuthEngine,
	},
	source::BrokerCredentialSource,
	store::{CredentialFilter, CredentialState, Store},
	usage::{UsageError, UsageHttp, UsageHttpResponse, UsageManager},
};
use omp_llm_catalog::{
	codex::{CODEX_CLIENT_VERSION, CODEX_ORIGINATOR},
	discovery::{
		Account, Error as DiscoveryError, HttpClient as DiscoveryClient, HttpResponse,
		discovered_card, parse_cca_models, parse_codex_models, parse_gitlab_duo_models,
	},
	models::{Modality, ModelCard},
	provider::{AuthSpec, ProviderEntry, RegistryMapping},
};
use omp_llm_egress::{
	auth_inject::{AuthContext, CredentialLease, CredentialMetadataSource},
	client::{Body, EgressClient},
};
use tower::{Service, ServiceExt as _};

const FIRST_BYTE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub(crate) struct BrokerHttp {
	client: EgressClient,
}

impl BrokerHttp {
	pub(crate) fn new() -> Self {
		Self::with_client(EgressClient::new(FIRST_BYTE_TIMEOUT))
	}

	pub(crate) const fn with_client(client: EgressClient) -> Self {
		Self { client }
	}
}

impl OAuthHttpClient for BrokerHttp {
	fn execute(&self, request: HttpRequest) -> HttpFuture<'_> {
		Box::pin(async move {
			let mut outbound = Request::new(Full::new(request.body));
			*outbound.method_mut() = request.method;
			*outbound.uri_mut() = request
				.url
				.as_str()
				.parse()
				.map_err(|error: http::uri::InvalidUri| oauth_error(error.to_string(), false))?;
			*outbound.headers_mut() = request.headers;
			let response = self
				.client
				.clone()
				.oneshot(outbound)
				.await
				.map_err(|error| oauth_error(error.to_string(), true))?;
			let status = response.status().as_u16();
			let body = response
				.into_body()
				.collect()
				.await
				.map_err(|error| oauth_error(error.to_string(), true))?
				.to_bytes();
			Ok(OAuthHttpResponse { status, body })
		})
	}
}

impl UsageHttp for BrokerHttp {
	fn send(&self, request: Request<Bytes>) -> BoxFuture<'_, Result<UsageHttpResponse, UsageError>> {
		Box::pin(async move {
			let (parts, body) = request.into_parts();
			let response = self
				.client
				.clone()
				.oneshot(Request::from_parts(parts, Full::new(body)))
				.await
				.map_err(usage_transport_error)?;
			let status = response.status();
			let body = response
				.into_body()
				.collect()
				.await
				.map_err(usage_transport_error)?
				.to_bytes();
			Ok(UsageHttpResponse { status, body })
		})
	}
}
/// Broker- and egress-backed production model discovery transport.
///
/// Account keys are non-secret database ids. Requests carry an exact
/// [`CredentialLease`] extension, so the shared auth-injection stack can never
/// substitute a sibling account.
pub(crate) struct CatalogDiscoveryHttp<S> {
	egress: S,
	store:  Arc<Store>,
	source: BrokerCredentialSource,
}

impl<S> CatalogDiscoveryHttp<S> {
	pub(crate) const fn new(egress: S, store: Arc<Store>, source: BrokerCredentialSource) -> Self {
		Self { egress, store, source }
	}
}

#[async_trait]
impl<S> DiscoveryClient for CatalogDiscoveryHttp<S>
where
	S: Service<Request<Body>, Response = Response<Incoming>> + Clone + Send + Sync + 'static,
	S::Future: Send,
	S::Error: Display + Send + Sync,
{
	async fn accounts(&self, provider: &ProviderEntry) -> Result<Vec<Account>, DiscoveryError> {
		let credential_provider = credential_provider(provider);
		let credentials = self
			.store
			.list_credentials(&CredentialFilter {
				provider: Some(credential_provider),
				states:   &[CredentialState::Active, CredentialState::Blocked],
				now_ms:   current_epoch_ms(),
			})
			.map_err(discovery_transport)?;
		if credentials.is_empty()
			&& matches!(&provider.auth, AuthSpec::None | AuthSpec::OptionalBearer { .. })
		{
			return Ok(vec![Account::provider_default()]);
		}
		let mut accounts = Vec::with_capacity(credentials.len());
		for credential in credentials {
			let lease = self
				.store
				.lease(credential.id)
				.map_err(discovery_transport)?
				.ok_or_else(|| DiscoveryError::Transport("discovery account lease is stale".into()))?;
			let metadata = self.source.metadata(&lease).map_err(discovery_transport)?;
			accounts.push(
				Account::new(credential.id.to_string(), credential.identity)
					.with_account_id(metadata.account_id)
					.with_scope(metadata.organization_id, metadata.project_id),
			);
		}
		Ok(accounts)
	}

	async fn get(
		&self,
		provider: &ProviderEntry,
		url: &str,
	) -> Result<HttpResponse, DiscoveryError> {
		self.send(provider, &Account::provider_default(), url).await
	}

	async fn get_for_account(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		url: &str,
	) -> Result<HttpResponse, DiscoveryError> {
		self.send(provider, account, url).await
	}

	async fn discover_specialized(
		&self,
		provider: &ProviderEntry,
		account: &Account,
	) -> Result<Vec<ModelCard>, DiscoveryError> {
		match provider.id.as_str() {
			"openai-codex" | "openai-codex-device" => {
				let base = provider.base_url.trim_end_matches('/');
				let mut headers = vec![
					("OpenAI-Beta", "responses=experimental"),
					("originator", CODEX_ORIGINATOR),
					("version", CODEX_CLIENT_VERSION),
				];
				if let Some(account_id) = account.account_id.as_deref() {
					headers.push(("chatgpt-account-id", account_id));
				}
				let mut last_error = None;
				for path in [
					format!("/codex/models?client_version={CODEX_CLIENT_VERSION}"),
					format!("/models?client_version={CODEX_CLIENT_VERSION}"),
				] {
					match self
						.send_request(
							provider,
							account,
							Method::GET,
							&format!("{base}{path}"),
							Bytes::new(),
							&headers,
						)
						.await
					{
						Ok(response) if (200..300).contains(&response.status) => {
							return parse_codex_models(provider, &response.body);
						},
						Ok(response) => {
							last_error = Some(DiscoveryError::HttpStatus {
								provider: provider.id.clone(),
								status:   response.status,
							});
						},
						Err(error) => last_error = Some(error),
					}
				}
				Err(last_error.unwrap_or_else(|| {
					DiscoveryError::Transport("Codex discovery had no endpoint".into())
				}))
			},
			"cursor" => {
				let response = self
					.send_request(
						provider,
						account,
						Method::POST,
						&format!(
							"{}/agent.v1.AgentService/GetUsableModels",
							provider.base_url.trim_end_matches('/')
						),
						omp_llm_cursor::model_discovery_request(),
						&[
							("content-type", "application/proto"),
							("accept", "application/proto"),
							("te", "trailers"),
							("x-ghost-mode", "true"),
							("x-cursor-client-version", "cli-2026.07.23-e383d2b"),
							("x-cursor-client-type", "cli"),
						],
					)
					.await?;
				if !(200..300).contains(&response.status) {
					return Err(DiscoveryError::HttpStatus {
						provider: provider.id.clone(),
						status:   response.status,
					});
				}
				omp_llm_cursor::decode_model_discovery(&response.body)
					.map_err(discovery_transport)
					.map(|models| {
						models
							.into_iter()
							.map(|model| {
								let family = model
									.id
									.as_str()
									.split(['/', '-'])
									.next()
									.unwrap_or(model.id.as_str());
								let mut card = discovered_card(
									provider,
									model.id.as_str(),
									model.name.as_str(),
									family,
								);
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
							})
							.collect()
					})
			},
			"devin" => {
				let response = self.send_devin_discovery(provider, account).await?;
				if !(200..300).contains(&response.status) {
					return Err(DiscoveryError::HttpStatus {
						provider: provider.id.clone(),
						status:   response.status,
					});
				}
				omp_llm_devin::decode_model_discovery(&response.body)
					.map_err(discovery_transport)
					.map(|models| {
						models
							.into_iter()
							.map(|model| {
								let mut card = discovered_card(
									provider,
									model.id.as_str(),
									model.name.as_str(),
									model
										.id
										.as_str()
										.split(['-', '_'])
										.next()
										.unwrap_or(model.id.as_str()),
								);
								card.reasoning = model.reasoning;
								card.context_window = model.context_window;
								card.max_output_tokens = model.max_output_tokens;
								if model.supports_images {
									card.inputs.push(Modality::Image);
								}
								card
							})
							.collect()
					})
			},
			"gitlab-duo-agent" => self.discover_gitlab_duo(provider, account).await,
			"google-antigravity" | "google-gemini-cli" => {
				let mut endpoints = Vec::with_capacity(1 + provider.fallback_base_urls.len());
				endpoints.push(provider.base_url.as_str());
				endpoints.extend(provider.fallback_base_urls.iter().map(SmolStr::as_str));
				let mut last_error = None;
				for endpoint in endpoints {
					match self
						.send_request(
							provider,
							account,
							Method::POST,
							&format!("{}/v1internal:fetchAvailableModels", endpoint.trim_end_matches('/')),
							Bytes::from_static(b"{}"),
							&[("content-type", "application/json")],
						)
						.await
					{
						Ok(response) if (200..300).contains(&response.status) => {
							return parse_cca_models(provider, &response.body);
						},
						Ok(response) => {
							last_error = Some(DiscoveryError::HttpStatus {
								provider: provider.id.clone(),
								status:   response.status,
							});
						},
						Err(error) => last_error = Some(error),
					}
				}
				Err(last_error.unwrap_or_else(|| {
					DiscoveryError::Transport("CCA discovery had no endpoint".into())
				}))
			},
			_ => Err(DiscoveryError::UnsupportedProvider(provider.id.clone())),
		}
	}
}

impl<S> CatalogDiscoveryHttp<S>
where
	S: Service<Request<Body>, Response = Response<Incoming>> + Clone + Send + Sync + 'static,
	S::Future: Send,
	S::Error: Display + Send + Sync,
{
	async fn send(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		url: &str,
	) -> Result<HttpResponse, DiscoveryError> {
		self
			.send_request(provider, account, Method::GET, url, Bytes::new(), &[])
			.await
	}

	async fn send_request(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		method: Method,
		url: &str,
		body: Bytes,
		extra_headers: &[(&str, &str)],
	) -> Result<HttpResponse, DiscoveryError> {
		let mut request = Request::builder()
			.method(method)
			.uri(url)
			.header(ACCEPT, "application/json")
			.body(Full::new(body))
			.map_err(discovery_transport)?;
		if provider.id == "cursor" {
			*request.version_mut() = http::Version::HTTP_2;
		}
		for (name, value) in &provider.headers {
			let name = http::HeaderName::try_from(name.as_str()).map_err(discovery_transport)?;
			let value = http::HeaderValue::try_from(value.as_str()).map_err(discovery_transport)?;
			request.headers_mut().insert(name, value);
		}
		for (name, value) in extra_headers {
			let name = http::HeaderName::try_from(*name).map_err(discovery_transport)?;
			let value = http::HeaderValue::try_from(*value).map_err(discovery_transport)?;
			request.headers_mut().insert(name, value);
		}
		if account.key.is_empty() {
			request
				.extensions_mut()
				.insert(AuthContext::new(credential_provider(provider)));
		} else {
			let id = account
				.key
				.parse::<u64>()
				.map_err(|_| DiscoveryError::Transport("invalid discovery account key".into()))?;
			let lease = self
				.store
				.lease(id)
				.map_err(discovery_transport)?
				.filter(|lease| lease.provider() == credential_provider(provider))
				.ok_or_else(|| DiscoveryError::Transport("discovery account lease is stale".into()))?;
			request.extensions_mut().insert::<CredentialLease>(lease);
		}
		let response = self
			.egress
			.clone()
			.oneshot(request)
			.await
			.map_err(discovery_transport)?;
		let status = response.status().as_u16();
		let body = response
			.into_body()
			.collect()
			.await
			.map_err(discovery_transport)?
			.to_bytes();
		Ok(HttpResponse::new(status, body))
	}

	async fn send_devin_discovery(
		&self,
		provider: &ProviderEntry,
		account: &Account,
	) -> Result<HttpResponse, DiscoveryError> {
		let id = account
			.key
			.parse::<u64>()
			.map_err(|_| DiscoveryError::Transport("invalid Devin discovery account key".into()))?;
		let lease = self
			.store
			.lease(id)
			.map_err(discovery_transport)?
			.filter(|lease| lease.provider() == credential_provider(provider))
			.ok_or_else(|| {
				DiscoveryError::Transport("Devin discovery account lease is stale".into())
			})?;
		let mut request = Request::builder()
			.method(Method::POST)
			.uri(format!(
				"{}/exa.api_server_pb.ApiServerService/GetCliModelConfigs",
				provider.base_url.trim_end_matches('/')
			))
			.header(ACCEPT, "*/*")
			.header("content-type", "application/proto")
			.header("connect-protocol-version", "1")
			.body(Full::new(Bytes::new()))
			.map_err(discovery_transport)?;
		self
			.source
			.apply_devin_discovery(&lease, &mut request)
			.map_err(discovery_transport)?;
		let response = self
			.egress
			.clone()
			.oneshot(request)
			.await
			.map_err(discovery_transport)?;
		let status = response.status().as_u16();
		let body = response
			.into_body()
			.collect()
			.await
			.map_err(discovery_transport)?
			.to_bytes();
		Ok(HttpResponse::new(status, body))
	}

	async fn discover_gitlab_duo(
		&self,
		provider: &ProviderEntry,
		account: &Account,
	) -> Result<Vec<ModelCard>, DiscoveryError> {
		if let Some(namespace) = account.organization_id.as_deref() {
			let cards = self
				.gitlab_namespace_models(provider, account, namespace)
				.await?;
			if !cards.is_empty() {
				return Ok(cards);
			}
		}
		if let Some(project) = account.project_id.as_deref() {
			if let Some(namespace) = self
				.gitlab_project_namespace(provider, account, project)
				.await?
			{
				let cards = self
					.gitlab_namespace_models(provider, account, &namespace)
					.await?;
				if !cards.is_empty() {
					return Ok(cards);
				}
			}
		}
		for page in 1..=50 {
			let url = format!(
				"{}/api/v4/groups?top_level_only=true&per_page=100&page={page}",
				provider.base_url.trim_end_matches('/')
			);
			let response = self
				.send_request(provider, account, Method::GET, &url, Bytes::new(), &[])
				.await?;
			if !(200..300).contains(&response.status) {
				return Err(DiscoveryError::HttpStatus {
					provider: provider.id.clone(),
					status:   response.status,
				});
			}
			let groups: Vec<serde_json::Value> =
				serde_json::from_slice(&response.body).map_err(discovery_transport)?;
			let count = groups.len();
			for group in groups {
				let Some(namespace) = group.get("id").and_then(serde_json::Value::as_u64) else {
					continue;
				};
				let cards = self
					.gitlab_namespace_models(provider, account, &namespace.to_string())
					.await?;
				if !cards.is_empty() {
					return Ok(cards);
				}
			}
			if count < 100 {
				break;
			}
		}
		Err(DiscoveryError::InvalidPayload {
			provider: provider.id.clone(),
			detail:   "no GitLab namespace exposes Duo models".into(),
		})
	}

	async fn gitlab_project_namespace(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		project: &str,
	) -> Result<Option<String>, DiscoveryError> {
		let encoded = url::form_urlencoded::byte_serialize(project.as_bytes()).collect::<String>();
		let url = format!("{}/api/v4/projects/{encoded}", provider.base_url.trim_end_matches('/'));
		let response = self
			.send_request(provider, account, Method::GET, &url, Bytes::new(), &[])
			.await?;
		if !(200..300).contains(&response.status) {
			return Err(DiscoveryError::HttpStatus {
				provider: provider.id.clone(),
				status:   response.status,
			});
		}
		let payload: serde_json::Value =
			serde_json::from_slice(&response.body).map_err(discovery_transport)?;
		Ok(payload
			.pointer("/namespace/root_ancestor/id")
			.or_else(|| payload.pointer("/namespace/id"))
			.and_then(|value| {
				value
					.as_str()
					.map(ToOwned::to_owned)
					.or_else(|| value.as_u64().map(|value| value.to_string()))
			}))
	}

	async fn gitlab_namespace_models(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		namespace: &str,
	) -> Result<Vec<ModelCard>, DiscoveryError> {
		const QUERY: &str = "query lsp_aiChatAvailableModels($rootNamespaceId: GroupID!) { \
		                     aiChatAvailableModels(rootNamespaceId: $rootNamespaceId) { \
		                     defaultModel { name ref } selectableModels { name ref } pinnedModel { \
		                     name ref } } }";
		let root_namespace_id = if namespace.bytes().all(|byte| byte.is_ascii_digit()) {
			format!("gid://gitlab/Group/{namespace}")
		} else {
			namespace.to_owned()
		};
		let body = serde_json::to_vec(&serde_json::json!({
			"query": QUERY,
			"variables": { "rootNamespaceId": root_namespace_id },
		}))
		.map_err(discovery_transport)?;
		let response = self
			.send_request(
				provider,
				account,
				Method::POST,
				&format!("{}/api/graphql", provider.base_url.trim_end_matches('/')),
				Bytes::from(body),
				&[("content-type", "application/json")],
			)
			.await?;
		if !(200..300).contains(&response.status) {
			return Err(DiscoveryError::HttpStatus {
				provider: provider.id.clone(),
				status:   response.status,
			});
		}
		parse_gitlab_duo_models(provider, &response.body)
	}
}
fn current_epoch_ms() -> u64 {
	let millis = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis();
	u64::try_from(millis).unwrap_or(u64::MAX)
}

fn credential_provider(provider: &ProviderEntry) -> &str {
	match &provider.mapping {
		RegistryMapping::Alias { target, .. } => target,
		RegistryMapping::Concrete | RegistryMapping::Replacement { .. } => provider.id.as_str(),
	}
}

fn discovery_transport(error: impl Display) -> DiscoveryError {
	DiscoveryError::Transport(SmolStr::from(error.to_string()))
}

fn oauth_error(error: String, transient: bool) -> HttpError {
	HttpError { detail: SmolStr::from(error), transient }
}

fn usage_transport_error(error: impl std::fmt::Display) -> UsageError {
	UsageError::InvalidResponse {
		provider: "transport".into(),
		message:  SmolStr::from(error.to_string()),
	}
}

/// Opens the durable broker store and constructs its production CLI backend.
///
/// OAuth and quota requests use the same pooled, proxy-aware egress client as
/// provider inference. The database parent is created before SQLite is opened.
pub fn open(path: &Path) -> anyhow::Result<BrokerCliBackend> {
	if let Some(parent) = path.parent()
		&& !parent.as_os_str().is_empty()
	{
		std::fs::create_dir_all(parent)?;
	}
	let store = Arc::new(Store::open(path)?);
	let http = Arc::new(BrokerHttp::new());
	let oauth = OAuthEngine::new(Arc::clone(&store), http.clone())?;
	let usage_http: Arc<dyn UsageHttp> = http;
	let usage = UsageManager::new(Arc::clone(&store), usage_http);
	Ok(BrokerCliBackend::new(store, Some(oauth), usage))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn transport_failures_are_client_safe() {
		let error = usage_transport_error("connection refused");
		assert!(error.to_string().contains("connection refused"));
		let oauth = oauth_error("timeout".to_owned(), true);
		assert!(oauth.transient);
		assert_eq!(oauth.detail, "timeout");
	}
}
