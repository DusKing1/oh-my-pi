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
use http::{Request, Response};
use http_body_util::{BodyExt as _, Full};
use hyper::body::Incoming;
use omp_core::Str;
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
	discovery::{Account, DiscoveryHttp, Error as DiscoveryError, HttpResponse, SealedBody},
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
			let (parts, body) = response.into_parts();
			let status = parts.status.as_u16();
			let body = body
				.collect()
				.await
				.map_err(|error| oauth_error(error.to_string(), true))?
				.to_bytes();
			Ok(OAuthHttpResponse { status, body, headers: parts.headers })
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
pub(crate) struct BrokerDiscoveryHttp<S> {
	egress: S,
	store:  Arc<Store>,
	source: BrokerCredentialSource,
}

impl<S> BrokerDiscoveryHttp<S> {
	pub(crate) const fn new(egress: S, store: Arc<Store>, source: BrokerCredentialSource) -> Self {
		Self { egress, store, source }
	}
}

#[async_trait]
impl<S> DiscoveryHttp for BrokerDiscoveryHttp<S>
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
			.map_err(DiscoveryError::transport)?;
		if credentials.is_empty()
			&& matches!(&provider.auth, AuthSpec::None | AuthSpec::OptionalBearer { .. })
		{
			return Ok(vec![Account::provider_default()]);
		}
		let region = deployment_region(provider);
		let mut accounts = Vec::with_capacity(credentials.len());
		for credential in credentials {
			let lease = self
				.store
				.lease(credential.id)
				.map_err(DiscoveryError::transport)?
				.ok_or_else(|| DiscoveryError::transport("discovery account lease is stale"))?;
			let metadata = self
				.source
				.metadata(&lease)
				.map_err(DiscoveryError::transport)?;
			accounts.push(
				Account::new(credential.id.to_string(), credential.identity)
					.with_account_id(metadata.account_id)
					.with_scope(metadata.organization_id, metadata.project_id)
					.with_region(region.clone()),
			);
		}
		Ok(accounts)
	}

	async fn execute(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		request: Request<Bytes>,
	) -> Result<HttpResponse, DiscoveryError> {
		let sealed_body = request.extensions().get::<SealedBody>().is_some();
		let (parts, body) = request.into_parts();
		let mut request = Request::from_parts(parts, Full::new(body));
		for (name, value) in &provider.headers {
			let name = http::HeaderName::try_from(name.as_str()).map_err(DiscoveryError::transport)?;
			if request.headers().contains_key(&name) {
				continue;
			}
			let value =
				http::HeaderValue::try_from(value.as_str()).map_err(DiscoveryError::transport)?;
			request.headers_mut().insert(name, value);
		}
		if sealed_body {
			// The protocol could not build its own body: the credential lives
			// inside the payload, so only the broker may write it.
			if account.key.is_empty() {
				return Err(DiscoveryError::transport(
					"sealed-body discovery requires an account credential",
				));
			}
			self
				.source
				.apply_sealed_discovery_body(&self.lease(provider, account)?, &mut request)
				.map_err(DiscoveryError::transport)?;
		} else if account.key.is_empty() {
			// No enumerated account: the shared auth stack selects the
			// provider-wide credential.
			request
				.extensions_mut()
				.insert(AuthContext::new(credential_provider(provider)));
		} else {
			request
				.extensions_mut()
				.insert::<CredentialLease>(self.lease(provider, account)?);
		}
		let response = self
			.egress
			.clone()
			.oneshot(request)
			.await
			.map_err(DiscoveryError::transport)?;
		let status = response.status().as_u16();
		let body = response
			.into_body()
			.collect()
			.await
			.map_err(DiscoveryError::transport)?
			.to_bytes();
		Ok(HttpResponse::new(status, body))
	}
}

impl<S> BrokerDiscoveryHttp<S> {
	/// Resolves the lease named by a non-secret discovery account key.
	///
	/// The key is a database id, never credential material. A lease belonging
	/// to a different provider is rejected rather than substituted, so a
	/// specialized protocol can never list a sibling account's models.
	fn lease(
		&self,
		provider: &ProviderEntry,
		account: &Account,
	) -> Result<CredentialLease, DiscoveryError> {
		let id = account
			.key
			.parse::<u64>()
			.map_err(|_| DiscoveryError::transport("invalid discovery account key"))?;
		self
			.store
			.lease(id)
			.map_err(DiscoveryError::transport)?
			.filter(|lease| lease.provider() == credential_provider(provider))
			.ok_or_else(|| DiscoveryError::transport("discovery account lease is stale"))
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

/// Reads the cloud region a provider's signing scheme needs from the process
/// environment.
///
/// Discovery protocols receive this through [`Account::region`] so they never
/// perform environment lookups of their own. Only AWS `SigV4` providers carry a
/// region today; every other scheme resolves its endpoint from catalog data.
fn deployment_region(provider: &ProviderEntry) -> Option<Str> {
	if !matches!(&provider.auth, AuthSpec::AwsSigV4) {
		return None;
	}
	["AWS_REGION", "AWS_DEFAULT_REGION"]
		.into_iter()
		.find_map(|name| std::env::var(name).ok())
		.filter(|region| !region.is_empty())
		.map(Str::from)
}

fn oauth_error(error: String, transient: bool) -> HttpError {
	HttpError { detail: Str::from(error), transient }
}

fn usage_transport_error(error: impl std::fmt::Display) -> UsageError {
	UsageError::InvalidResponse {
		provider: "transport".into(),
		message:  Str::from(error.to_string()),
	}
}

/// Opens the durable broker store and constructs its production CLI backend.
///
/// OAuth and quota requests use the same pooled, proxy-aware egress client as
/// provider inference. The database parent is created before SQLite is opened.
pub fn open(path: &Path) -> crate::Result<BrokerCliBackend> {
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
