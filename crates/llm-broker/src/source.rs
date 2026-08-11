//! Canonical egress credential leasing, redemption, and refresh.
//!
//! Secret bytes stay in [`Store`]. A lease carries only provider, database id,
//! and generation metadata; redemption validates all three while holding the
//! store lock and mutates the outbound request before releasing it.

use std::{
	error::Error as StdError,
	fmt,
	future::Future,
	sync::Arc,
	time::{SystemTime, UNIX_EPOCH},
};

use http::{
	Request,
	header::{HeaderMap, HeaderName},
};
use omp_core::Str;
use omp_llm_catalog::provider::{AuthSpec, ProviderCatalog};
use omp_llm_cursor::CursorAuth;
use omp_llm_devin::{
	DevinAuth,
	wire::{GetChatMessageRequest, GetUserJwtRequest, GetUserJwtResponse, Metadata},
};
use omp_llm_egress::{
	auth_inject::{CredentialLease, CredentialMetadata, CredentialMetadataSource, CredentialSource},
	client::Body,
};
use thiserror::Error;
use tonic::transport::{Channel, Endpoint};

use crate::{
	oauth::{OAuthEngine, OAuthError},
	store::{Store, StoreError},
};

/// Refresh capability used by [`BrokerCredentialSource`].
///
/// The trait uses a concrete returned future so the production OAuth path does
/// not box one future per refresh.
pub trait CredentialRefresher: Send + Sync + 'static {
	/// Refresh failure retained for callers but redacted by the broker source's
	/// own formatting.
	type Error: StdError + Send + Sync + 'static;

	/// Refreshes `credential_id`, persisting replacement access material before
	/// the returned future completes.
	fn refresh(
		&self,
		credential_id: u64,
		now_ms: u64,
	) -> impl Future<Output = Result<(), Self::Error>> + Send;
}

impl CredentialRefresher for OAuthEngine {
	type Error = OAuthError;

	async fn refresh(&self, credential_id: u64, now_ms: u64) -> Result<(), Self::Error> {
		self.refresh_credential(credential_id, now_ms).await?;
		Ok(())
	}
}

/// Failure while leasing, redeeming, or refreshing a broker credential.
#[derive(Error)]
pub enum BrokerCredentialSourceError<E: StdError + Send + Sync + 'static> {
	/// Credential persistence failed.
	#[error(transparent)]
	Store(#[from] StoreError),
	/// The lease references provider metadata absent from the catalog.
	#[error("credential provider `{0}` is not registered")]
	MissingProvider(String),
	/// The stored credential kind cannot satisfy the catalog authentication
	/// mode or requested lifecycle operation.
	#[error("credential provider `{provider}` uses unsupported authentication `{kind}`")]
	UnsupportedAuth {
		/// Catalog provider identifier.
		provider: String,
		/// Non-secret authentication kind.
		kind:     &'static str,
	},
	/// The credential id, provider, or generation changed before redemption.
	#[error("credential {credential_id} generation {generation} is stale for provider `{provider}`")]
	StaleGeneration {
		/// Catalog provider identifier.
		provider:      String,
		/// Stable database credential identifier.
		credential_id: u64,
		/// Generation claimed by the rejected lease.
		generation:    u64,
	},
	/// A catalog header name is invalid.
	#[error("provider `{provider}` has an invalid authentication header name")]
	InvalidHeaderName {
		/// Catalog provider identifier.
		provider: String,
	},
	/// Secret bytes cannot be represented by the configured header placement.
	#[error("provider `{provider}` credential is not a valid header value")]
	InvalidHeaderValue {
		/// Catalog provider identifier.
		provider: String,
	},
	/// AWS signing metadata was not attached by the provider transport.
	#[error("provider `{provider}` request has no AWS signing context")]
	MissingAwsContext {
		/// Catalog provider identifier.
		provider: String,
	},
	/// Sealed AWS signing failed without exposing credential details.
	#[error("provider `{provider}` request could not be AWS-signed")]
	AwsSigning {
		/// Catalog provider identifier.
		provider: String,
	},
	/// The injected OAuth machinery failed. Formatting never includes the
	/// underlying error because provider failures may originate at a secret
	/// boundary.
	#[error("credential refresh failed")]
	Refresh(E),
	/// Refresh completed without replacing the credential generation.
	#[error("credential {credential_id} refresh did not advance its generation")]
	RefreshDidNotRotate {
		/// Stable database credential identifier.
		credential_id: u64,
	},
	/// No active credential remained for a route selected during daemon startup.
	#[error("provider `{provider}` has no active credential")]
	MissingCredential {
		/// Catalog provider identifier.
		provider: String,
	},
	/// Devin's account authentication exchange failed or returned no JWT.
	#[error("provider `{provider}` could not establish Devin authentication")]
	DevinAuthentication {
		/// Catalog provider identifier.
		provider: String,
	},
	/// Devin returned an invalid account-specific API server URL.
	#[error("provider `{provider}` returned an invalid Devin endpoint")]
	InvalidDevinEndpoint {
		/// Catalog provider identifier.
		provider: String,
	},
}

impl<E: StdError + Send + Sync + 'static> fmt::Debug for BrokerCredentialSourceError<E> {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt::Display::fmt(self, formatter)
	}
}

/// Broker-backed implementation of the canonical egress credential source.
pub struct BrokerCredentialSource<R = OAuthEngine> {
	store:     Arc<Store>,
	providers: Arc<ProviderCatalog>,
	refresher: Arc<R>,
}

impl<R> Clone for BrokerCredentialSource<R> {
	fn clone(&self) -> Self {
		Self {
			store:     Arc::clone(&self.store),
			providers: Arc::clone(&self.providers),
			refresher: Arc::clone(&self.refresher),
		}
	}
}

impl<R> BrokerCredentialSource<R> {
	/// Constructs a credential source from the daemon store, catalog-owned
	/// provider metadata, and the existing OAuth refresh engine.
	#[must_use]
	pub const fn new(store: Arc<Store>, providers: Arc<ProviderCatalog>, refresher: Arc<R>) -> Self {
		Self { store, providers, refresher }
	}

	/// Returns the underlying daemon credential store.
	#[must_use]
	pub const fn store(&self) -> &Arc<Store> {
		&self.store
	}

	/// Leases the usage-ranked credential for `provider` at an injected clock.
	///
	/// The explicit clock keeps quota-window and cache-warm expiry decisions
	/// deterministic for daemon callers and focused tests. Secret material never
	/// leaves the store.
	pub fn lease_at(
		&self,
		provider: &str,
		now_ms: u64,
	) -> Result<Option<CredentialLease>, BrokerCredentialSourceError<R::Error>>
	where
		R: CredentialRefresher,
	{
		self.provider_auth(provider)?;
		Ok(self.store.lease_provider(provider, now_ms)?)
	}

	/// Fills a discovery request body whose credential is embedded in the
	/// payload.
	///
	/// The credential is redeemed and encoded entirely inside the broker
	/// boundary.
	pub fn apply_sealed_discovery_body(
		&self,
		lease: &CredentialLease,
		request: &mut Request<Body>,
	) -> Result<(), BrokerCredentialSourceError<R::Error>>
	where
		R: CredentialRefresher,
	{
		self.provider_auth(lease.provider())?;
		let applied = self.store.redeem_typed_with(
			lease.provider(),
			lease.credential_id(),
			lease.generation(),
			|_kind, auth| {
				auth.apply_sealed_discovery_body(request);
				Ok(())
			},
		)?;
		applied.unwrap_or_else(|| Err(Self::stale(lease)))
	}
}

impl<R> BrokerCredentialSource<R> {
	/// Applies a lease as Cursor bearer authentication without exposing the
	/// bearer value.
	pub fn apply_cursor_headers(
		&self,
		lease: &CredentialLease,
		headers: &mut HeaderMap,
	) -> Result<(), BrokerCredentialSourceError<R::Error>>
	where
		R: CredentialRefresher,
	{
		self.provider_auth(lease.provider())?;
		let applied = self.store.redeem_typed_with(
			lease.provider(),
			lease.credential_id(),
			lease.generation(),
			|kind, auth| {
				if kind == crate::store::CredentialKind::Aws {
					return Err(BrokerCredentialSourceError::UnsupportedAuth {
						provider: lease.provider().to_owned(),
						kind:     "aws-for-cursor",
					});
				}
				auth.apply_bearer_to_headers(headers).map_err(|_| {
					BrokerCredentialSourceError::InvalidHeaderValue {
						provider: lease.provider().to_owned(),
					}
				})
			},
		)?;
		applied.unwrap_or_else(|| Err(Self::stale(lease)))
	}

	/// Applies a lease and an account JWT directly to Devin request metadata.
	pub fn apply_devin_metadata(
		&self,
		lease: &CredentialLease,
		metadata: &mut Metadata,
		user_jwt: String,
	) -> Result<(), BrokerCredentialSourceError<R::Error>>
	where
		R: CredentialRefresher,
	{
		self.provider_auth(lease.provider())?;
		let applied = self.store.redeem_typed_with(
			lease.provider(),
			lease.credential_id(),
			lease.generation(),
			|kind, auth| {
				if kind == crate::store::CredentialKind::Aws {
					return Err(BrokerCredentialSourceError::UnsupportedAuth {
						provider: lease.provider().to_owned(),
						kind:     "aws-for-devin",
					});
				}
				auth.apply_to_devin_metadata(metadata, user_jwt);
				Ok(())
			},
		)?;
		applied.unwrap_or_else(|| Err(Self::stale(lease)))
	}
}

/// Broker-owned sealed authentication for one specialized provider route.
///
/// The adapter leases the currently usage-ranked credential for each turn and
/// redeems it only while mutating Cursor headers or Devin protobuf metadata.
/// It deliberately has no token accessor.
///
/// ```compile_fail
/// # fn cannot_observe<R>(auth: &omp_llm_broker::source::SpecializedCredentialAuth<R>) {
/// let _raw_token = auth.token();
/// # }
/// ```
#[derive(Clone)]
pub struct SpecializedCredentialAuth<R = OAuthEngine> {
	source:   BrokerCredentialSource<R>,
	provider: Str,
}

impl<R> SpecializedCredentialAuth<R> {
	/// Binds a broker credential source to one catalog provider id.
	#[must_use]
	pub fn new(source: BrokerCredentialSource<R>, provider: impl Into<Str>) -> Self {
		Self { source, provider: provider.into() }
	}
}

impl<R> CursorAuth for SpecializedCredentialAuth<R>
where
	R: CredentialRefresher,
{
	type Error = BrokerCredentialSourceError<R::Error>;

	fn apply(&self, headers: &mut HeaderMap) -> Result<(), Self::Error> {
		let lease = self.source.lease(self.provider.as_str())?.ok_or_else(|| {
			BrokerCredentialSourceError::MissingCredential { provider: self.provider.to_string() }
		})?;
		self.source.apply_cursor_headers(&lease, headers)
	}
}

impl<R> DevinAuth for SpecializedCredentialAuth<R>
where
	R: CredentialRefresher,
{
	type Error = BrokerCredentialSourceError<R::Error>;

	async fn apply(
		&self,
		channel: &mut Channel,
		request: &mut GetChatMessageRequest,
	) -> Result<(), Self::Error> {
		let lease = self.source.lease(self.provider.as_str())?.ok_or_else(|| {
			BrokerCredentialSourceError::MissingCredential { provider: self.provider.to_string() }
		})?;
		let mut auth_request = GetUserJwtRequest { metadata: Some(Metadata::default()) };
		self.source.apply_devin_metadata(
			&lease,
			auth_request
				.metadata
				.as_mut()
				.expect("metadata initialized"),
			String::new(),
		)?;
		let mut grpc = tonic::client::Grpc::new(channel.clone());
		grpc
			.ready()
			.await
			.map_err(|_| BrokerCredentialSourceError::DevinAuthentication {
				provider: self.provider.to_string(),
			})?;
		let path = http::uri::PathAndQuery::from_static("/exa.auth_pb.AuthService/GetUserJwt");
		let codec = tonic_prost::ProstCodec::<GetUserJwtRequest, GetUserJwtResponse>::default();
		let response = grpc
			.unary(tonic::Request::new(auth_request), path, codec)
			.await
			.map_err(|_| BrokerCredentialSourceError::DevinAuthentication {
				provider: self.provider.to_string(),
			})?
			.into_inner();
		if response.user_jwt.is_empty() {
			return Err(BrokerCredentialSourceError::DevinAuthentication {
				provider: self.provider.to_string(),
			});
		}
		if !response.custom_api_server_url.trim().is_empty() {
			let endpoint = Endpoint::from_shared(
				response
					.custom_api_server_url
					.trim()
					.trim_end_matches('/')
					.to_owned(),
			)
			.map_err(|_| BrokerCredentialSourceError::InvalidDevinEndpoint {
				provider: self.provider.to_string(),
			})?;
			*channel = endpoint.connect_lazy();
		}
		let metadata = request.metadata.get_or_insert_default();
		self
			.source
			.apply_devin_metadata(&lease, metadata, response.user_jwt)?;
		Ok(())
	}
}

impl<R> CredentialSource for BrokerCredentialSource<R>
where
	R: CredentialRefresher,
{
	type Error = BrokerCredentialSourceError<R::Error>;

	fn lease(&self, provider: &str) -> Result<Option<CredentialLease>, Self::Error> {
		self.lease_at(provider, now_ms())
	}

	fn apply(
		&self,
		lease: &CredentialLease,
		request: &mut Request<Body>,
	) -> Result<(), Self::Error> {
		let provider = lease.provider();
		let auth_spec = self.provider_auth(provider)?;
		let aws_context = request
			.extensions()
			.get::<omp_llm_egress::auth_inject::AwsSigV4Context>()
			.cloned();
		let applied = self.store.redeem_typed_with(
			provider,
			lease.credential_id(),
			lease.generation(),
			|kind, auth| {
				if kind == crate::store::CredentialKind::OAuth {
					if matches!(auth_spec, AuthSpec::AwsSigV4) {
						return Err(BrokerCredentialSourceError::UnsupportedAuth {
							provider: provider.to_owned(),
							kind:     "oauth-for-aws-sig-v4",
						});
					}
					if matches!(auth_spec, AuthSpec::None) {
						return Ok(());
					}
					return auth
						.apply_bearer_to_headers(request.headers_mut())
						.map_err(|_| BrokerCredentialSourceError::InvalidHeaderValue {
							provider: provider.to_owned(),
						});
				}
				if kind == crate::store::CredentialKind::Aws {
					if !matches!(auth_spec, AuthSpec::AwsSigV4) {
						return Err(BrokerCredentialSourceError::UnsupportedAuth {
							provider: provider.to_owned(),
							kind:     "aws-credential",
						});
					}
					let context = aws_context.as_ref().ok_or_else(|| {
						BrokerCredentialSourceError::MissingAwsContext { provider: provider.to_owned() }
					})?;
					return auth.aws_sigv4(context, request).map_err(|_| {
						BrokerCredentialSourceError::AwsSigning { provider: provider.to_owned() }
					});
				}
				match auth_spec {
					AuthSpec::None => Ok(()),
					AuthSpec::Bearer { .. }
					| AuthSpec::OptionalBearer { .. }
					| AuthSpec::OAuth { .. } => {
						auth
							.apply_bearer_to_headers(request.headers_mut())
							.map_err(|_| BrokerCredentialSourceError::InvalidHeaderValue {
								provider: provider.to_owned(),
							})
					},
					AuthSpec::Header { name, .. } => {
						let name = HeaderName::try_from(name.as_str()).map_err(|_| {
							BrokerCredentialSourceError::InvalidHeaderName {
								provider: provider.to_owned(),
							}
						})?;
						auth
							.apply_to_named_header(name, request.headers_mut())
							.map_err(|_| BrokerCredentialSourceError::InvalidHeaderValue {
								provider: provider.to_owned(),
							})
					},
					AuthSpec::Query { param, .. } => {
						auth.apply_to_sensitive_query(param.as_str(), request.extensions_mut());
						Ok(())
					},
					AuthSpec::AwsSigV4 => Err(BrokerCredentialSourceError::UnsupportedAuth {
						provider: provider.to_owned(),
						kind:     "aws-sig-v4-credential",
					}),
					AuthSpec::GoogleAdc { .. } => {
						auth.apply_to_sensitive_query("key", request.extensions_mut());
						Ok(())
					},
				}
			},
		)?;

		match applied {
			Some(result) => result,
			None => Err(Self::stale(lease)),
		}
	}

	fn refresh(
		&self,
		lease: CredentialLease,
	) -> impl Future<Output = Result<CredentialLease, Self::Error>> + Send + 'static {
		let source = self.clone();
		async move {
			source.provider_auth(lease.provider())?;
			let now_ms = now_ms();
			let meta = source
				.store
				.get_credential(lease.credential_id(), now_ms)?
				.filter(|meta| meta.provider.as_str() == lease.provider())
				.ok_or_else(|| Self::stale(&lease))?;
			if meta.kind != crate::store::CredentialKind::OAuth {
				return Err(BrokerCredentialSourceError::UnsupportedAuth {
					provider: lease.provider().to_owned(),
					kind:     "refresh",
				});
			}
			let current = source.store.lease(lease.credential_id())?;
			if current.as_ref() != Some(&lease) {
				return Err(Self::stale(&lease));
			}
			source
				.refresher
				.refresh(lease.credential_id(), now_ms)
				.await
				.map_err(BrokerCredentialSourceError::Refresh)?;
			let refreshed = source
				.store
				.lease(lease.credential_id())?
				.filter(|candidate| candidate.provider() == lease.provider())
				.ok_or_else(|| Self::stale(&lease))?;
			if refreshed.generation() == lease.generation() {
				return Err(BrokerCredentialSourceError::RefreshDidNotRotate {
					credential_id: lease.credential_id(),
				});
			}
			Ok(refreshed)
		}
	}
}

impl<R> CredentialMetadataSource for BrokerCredentialSource<R>
where
	R: CredentialRefresher,
{
	fn metadata(&self, lease: &CredentialLease) -> Result<CredentialMetadata, Self::Error> {
		self.provider_auth(lease.provider())?;
		self
			.store
			.credential_metadata(lease.provider(), lease.credential_id(), lease.generation())?
			.ok_or_else(|| Self::stale(lease))
	}
}

impl<R> BrokerCredentialSource<R>
where
	R: CredentialRefresher,
{
	fn provider_auth(
		&self,
		provider: &str,
	) -> Result<&AuthSpec, BrokerCredentialSourceError<R::Error>> {
		self
			.providers
			.get(provider)
			.map(|entry| &entry.auth)
			.ok_or_else(|| BrokerCredentialSourceError::MissingProvider(provider.to_owned()))
	}

	fn stale(lease: &CredentialLease) -> BrokerCredentialSourceError<R::Error> {
		BrokerCredentialSourceError::StaleGeneration {
			provider:      lease.provider().to_owned(),
			credential_id: lease.credential_id(),
			generation:    lease.generation(),
		}
	}
}

fn now_ms() -> u64 {
	let millis = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis();
	u64::try_from(millis).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
	use std::{
		convert::Infallible,
		future::Future,
		sync::Arc,
		time::{Duration, UNIX_EPOCH},
	};

	use bytes::Bytes;
	use http::{Request, header::AUTHORIZATION};
	use omp_llm_catalog::provider::{ProviderCatalog, load_providers};
	use omp_llm_egress::{
		auth_inject::{AwsSigV4Context, CredentialMetadataSource, CredentialSource, SensitiveQuery},
		client::Body,
	};
	use serde_json::json;
	use tempfile::tempdir;

	use super::{
		BrokerCredentialSource, BrokerCredentialSourceError, CredentialRefresher,
		SpecializedCredentialAuth,
	};
	use crate::store::Store;

	const PROVIDERS: &str = r#"
[providers.bearer]
transport = "open-ai-chat"
base_url = "https://example.test"
auth = { type = "bearer", env = ["TOKEN"] }

[providers.named]
transport = "anthropic-messages"
base_url = "https://example.test"
auth = { type = "header", name = "x-api-key", env = ["TOKEN"] }

[providers.query]
transport = "google-gen-ai"
base_url = "https://example.test"
auth = { type = "query", param = "key", env = ["TOKEN"] }

[providers.oauth]
transport = "open-ai-responses"
base_url = "https://example.test"
auth = { type = "oauth", flow = "oauth" }

[providers.noauth]
transport = "open-ai-chat"
base_url = "https://example.test"
auth = { type = "none" }

[providers.aws]
transport = "anthropic-messages"
base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
auth = { type = "aws-sig-v4" }

[providers.adc]
transport = "google-vertex"
base_url = "https://us-central1-aiplatform.googleapis.com"
auth = { type = "google-adc", api_key_env = ["GOOGLE_API_KEY"], project_env = ["GOOGLE_CLOUD_PROJECT"], location_env = ["GOOGLE_CLOUD_LOCATION"] }

[providers.cursor]
transport = "cursor"
base_url = "https://api2.cursor.sh"
auth = { type = "bearer", env = ["CURSOR_API_KEY"] }

[providers.devin]
transport = "devin"
base_url = "https://server.codeium.com"
auth = { type = "bearer", env = ["DEVIN_API_KEY"] }
"#;

	struct RotatingRefresher {
		store: Arc<Store>,
	}

	impl CredentialRefresher for RotatingRefresher {
		type Error = Infallible;

		fn refresh(
			&self,
			credential_id: u64,
			now_ms: u64,
		) -> impl Future<Output = Result<(), Self::Error>> + Send {
			let store = Arc::clone(&self.store);
			async move {
				let meta = store
					.get_credential(credential_id, now_ms)
					.expect("credential lookup")
					.expect("credential");
				store
					.upsert_oauth(
						meta.provider.as_str(),
						meta.identity.as_str(),
						b"new-token",
						now_ms + 1,
						now_ms,
					)
					.expect("rotate OAuth token");
				Ok(())
			}
		}
	}

	fn source() -> (tempfile::TempDir, Arc<Store>, BrokerCredentialSource<RotatingRefresher>) {
		let directory = tempdir().expect("tempdir");
		let store = Arc::new(Store::open(directory.path().join("broker.sqlite")).expect("store"));
		let providers: ProviderCatalog = load_providers(PROVIDERS).expect("providers");
		let source = BrokerCredentialSource::new(
			Arc::clone(&store),
			Arc::new(providers),
			Arc::new(RotatingRefresher { store: Arc::clone(&store) }),
		);
		(directory, store, source)
	}

	fn request(uri: &str) -> Request<Body> {
		Request::builder()
			.uri(uri)
			.header("x-existing", "kept")
			.body(Body::new(Bytes::new()))
			.expect("request")
	}

	#[test]
	fn apply_places_each_catalog_auth_shape_without_disturbing_request_data() {
		let (_directory, store, source) = source();
		for provider in ["bearer", "named", "query", "oauth"] {
			store
				.upsert_api_key(provider, "account", b"a b&c", 10)
				.expect("credential");
		}

		let mut bearer = request("/v1?existing=yes");
		source
			.apply(&source.lease("bearer").expect("lookup").expect("lease"), &mut bearer)
			.expect("bearer apply");
		assert_eq!(bearer.headers()[AUTHORIZATION], "Bearer a b&c");
		assert_eq!(bearer.headers()["x-existing"], "kept");
		assert_eq!(bearer.uri(), "/v1?existing=yes");

		let mut named = request("/v1?existing=yes");
		source
			.apply(&source.lease("named").expect("lookup").expect("lease"), &mut named)
			.expect("named apply");
		assert_eq!(named.headers()["x-api-key"], "a b&c");
		assert_eq!(named.headers()["x-existing"], "kept");

		let mut query = request("/v1?existing=yes");
		source
			.apply(&source.lease("query").expect("lookup").expect("lease"), &mut query)
			.expect("query apply");
		assert_eq!(query.uri(), "/v1?existing=yes");
		let sensitive = query
			.extensions()
			.get::<SensitiveQuery>()
			.expect("sensitive query");
		assert!(!format!("{sensitive:?}").contains("a b&c"));
		assert!(!format!("{query:?}").contains("a b&c"));
		assert_eq!(query.headers()["x-existing"], "kept");

		let mut oauth = request("/v1");
		source
			.apply(&source.lease("oauth").expect("lookup").expect("lease"), &mut oauth)
			.expect("OAuth bearer apply");
		assert_eq!(oauth.headers()[AUTHORIZATION], "Bearer a b&c");

		store
			.upsert_oauth("noauth", "account", b"unused-oauth", 1_000, 10)
			.expect("unused credential");
		let mut noauth = request("/v1");
		source
			.apply(&source.lease("noauth").expect("lookup").expect("lease"), &mut noauth)
			.expect("no-auth apply");
		assert!(!noauth.headers().contains_key(AUTHORIZATION));
	}

	#[test]
	fn oauth_credentials_use_bearer_even_when_api_keys_use_a_named_header() {
		let (_directory, store, source) = source();
		store
			.upsert_oauth("named", "account", b"oauth-token", 1_000, 10)
			.expect("OAuth credential");
		let mut request = request("/v1");
		source
			.apply(&source.lease("named").expect("lookup").expect("lease"), &mut request)
			.expect("OAuth bearer apply");
		assert_eq!(request.headers()[AUTHORIZATION], "Bearer oauth-token");
		assert!(!request.headers().contains_key("x-api-key"));
	}

	#[test]
	fn aws_and_google_adc_modes_redeem_without_exporting_credentials() {
		let (_directory, store, aws_source) = source();
		store
			.upsert_aws(
				"aws",
				"account",
				b"AKIDEXAMPLE",
				b"wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
				Some(b"session-token"),
				10,
			)
			.expect("AWS credential");
		let lease = aws_source.lease("aws").expect("lookup").expect("lease");
		let mut aws = Request::builder()
			.method("POST")
			.uri("https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke?x=1")
			.header("content-type", "application/json")
			.body(Body::new(Bytes::from_static(b"{}")))
			.expect("request");
		aws.extensions_mut().insert(AwsSigV4Context {
			service:   "bedrock".into(),
			region:    "us-east-1".into(),
			signed_at: UNIX_EPOCH + Duration::from_secs(1_704_164_645),
		});
		aws_source.apply(&lease, &mut aws).expect("AWS apply");
		assert_eq!(aws.headers()["x-amz-date"], "20240102T030405Z");
		assert!(
			aws.headers()[AUTHORIZATION]
				.to_str()
				.expect("authorization")
				.starts_with(
					"AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20240102/us-east-1/bedrock/aws4_request"
				)
		);
		assert!(!format!("{lease:?}").contains("EXAMPLEKEY"));

		let (_directory, store, api_key_source) = source();
		store
			.upsert_api_key("adc", "key", b"google-key", 10)
			.expect("API key");
		let mut api_key = request("/v1?existing=yes");
		api_key_source
			.apply(&api_key_source.lease("adc").expect("lookup").expect("lease"), &mut api_key)
			.expect("ADC API-key fallback");
		assert_eq!(api_key.uri(), "/v1?existing=yes");
		let sensitive = api_key
			.extensions()
			.get::<SensitiveQuery>()
			.expect("sensitive query");
		assert!(!format!("{sensitive:?}").contains("google-key"));
		assert!(!format!("{api_key:?}").contains("google-key"));

		let (_directory, store, adc_source) = source();
		store
			.upsert_oauth("adc", "adc-account", b"minted-adc-token", 1_000, 10)
			.expect("minted ADC");
		let mut adc = request("/v1");
		adc_source
			.apply(&adc_source.lease("adc").expect("lookup").expect("lease"), &mut adc)
			.expect("ADC bearer");
		assert_eq!(adc.headers()[AUTHORIZATION], "Bearer minted-adc-token");
	}

	#[test]
	fn aws_rotation_is_rejected_before_signature_headers_are_mutated() {
		let (_directory, store, source) = source();
		let first = store
			.upsert_aws("aws", "account", b"OLDACCESS", b"old-secret", None, 10)
			.expect("AWS credential");
		let lease = store.lease(first.id).expect("lease lookup").expect("lease");
		store
			.upsert_aws("aws", "account", b"NEWACCESS", b"new-secret", None, 11)
			.expect("rotation");
		let mut request = Request::builder()
			.uri("https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke")
			.body(Body::new(Bytes::new()))
			.expect("request");
		request.extensions_mut().insert(AwsSigV4Context {
			service:   "bedrock".into(),
			region:    "us-east-1".into(),
			signed_at: UNIX_EPOCH,
		});
		let error = source
			.apply(&lease, &mut request)
			.expect_err("stale AWS lease");
		assert!(matches!(error, BrokerCredentialSourceError::StaleGeneration { .. }));
		assert!(!request.headers().contains_key(AUTHORIZATION));
		assert!(!request.headers().contains_key("x-amz-date"));
	}

	#[test]
	fn rotation_rejects_stale_generation_before_request_mutation() {
		let (_directory, store, source) = source();
		store
			.upsert_api_key("bearer", "account", b"old", 10)
			.expect("credential");
		let lease = source.lease("bearer").expect("lookup").expect("lease");
		store
			.upsert_api_key("bearer", "account", b"new", 11)
			.expect("rotation");
		let mut request = request("/v1?existing=yes");

		let error = source.apply(&lease, &mut request).expect_err("stale lease");
		assert!(matches!(error, BrokerCredentialSourceError::StaleGeneration { .. }));
		assert!(!request.headers().contains_key(AUTHORIZATION));
		assert_eq!(request.headers()["x-existing"], "kept");
		assert_eq!(request.uri(), "/v1?existing=yes");
	}

	#[test]
	fn metadata_projects_only_known_non_secret_fields_and_rejects_rotation() {
		let (_directory, store, source) = source();
		let material = "must-never-project";
		let props = json!({
			"openai": {
				"account_id": "account-7",
				"organization_id": "org-9"
			},
			"antigravity": { "project_id": "project-3" },
			"access_token": material,
			"arbitrary": { "secret": material }
		});
		let meta = store
			.upsert_oauth_material(
				"oauth",
				"person@example.test",
				b"access-secret",
				Some(b"refresh-secret"),
				&props,
				1_000,
				10,
			)
			.expect("OAuth material");
		let lease = store.lease(meta.id).expect("lease lookup").expect("lease");
		let projected = source.metadata(&lease).expect("metadata");
		assert_eq!(projected.identity, "person@example.test");
		assert_eq!(projected.account_id.as_deref(), Some("account-7"));
		assert_eq!(projected.project_id.as_deref(), Some("project-3"));
		assert_eq!(projected.organization_id.as_deref(), Some("org-9"));
		let debug = format!("{projected:?}");
		assert!(!debug.contains(material));
		assert!(!debug.contains("access-secret"));
		assert!(!debug.contains("refresh-secret"));

		store
			.upsert_oauth("oauth", "person@example.test", b"rotated", 2_000, 11)
			.expect("rotation");
		let error = source.metadata(&lease).expect_err("stale metadata lease");
		assert!(matches!(error, BrokerCredentialSourceError::StaleGeneration { .. }));
	}

	#[test]
	fn credential_errors_and_debug_never_render_secret_material() {
		let (_directory, store, source) = source();
		let material = "super-secret-token";
		let mut invalid = material.as_bytes().to_vec();
		invalid.push(b'\n');
		store
			.upsert_api_key("bearer", "account", &invalid, 10)
			.expect("credential");
		let mut request = request("/v1");
		let error = source
			.apply(&source.lease("bearer").expect("lookup").expect("lease"), &mut request)
			.expect_err("invalid header");
		assert!(!format!("{error}").contains(material));
		assert!(!format!("{error:?}").contains(material));

		#[derive(Debug)]
		struct LeakyRefreshError;
		impl std::fmt::Display for LeakyRefreshError {
			fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
				formatter.write_str("super-secret-token")
			}
		}
		impl std::error::Error for LeakyRefreshError {}
		let refresh_error =
			BrokerCredentialSourceError::<LeakyRefreshError>::Refresh(LeakyRefreshError);
		assert!(!format!("{refresh_error}").contains(material));
		assert!(!format!("{refresh_error:?}").contains(material));
	}

	#[tokio::test]
	async fn refresh_returns_the_persisted_replacement_generation() {
		let (_directory, store, source) = source();
		store
			.upsert_oauth("oauth", "account", b"old-token", 1, 10)
			.expect("credential");
		let lease = source.lease("oauth").expect("lookup").expect("lease");
		let refreshed = source.refresh(lease.clone()).await.expect("refresh");
		assert_eq!(refreshed.provider(), "oauth");
		assert_eq!(refreshed.credential_id(), lease.credential_id());
		assert!(refreshed.generation() > lease.generation());
	}

	#[test]
	fn specialized_auth_mutates_wire_surfaces_and_rejects_stale_leases() {
		let (_directory, store, source) = source();
		store
			.upsert_api_key("cursor", "account", b"cursor-secret", 10)
			.expect("Cursor credential");
		store
			.upsert_api_key("devin", "account", b"devin-secret", 10)
			.expect("Devin credential");

		let cursor = SpecializedCredentialAuth::new(source.clone(), "cursor");
		let mut headers = http::HeaderMap::new();
		omp_llm_cursor::CursorAuth::apply(&cursor, &mut headers).expect("Cursor auth");
		assert_eq!(headers[AUTHORIZATION], "Bearer cursor-secret");

		let stale = source.lease("cursor").expect("lookup").expect("lease");
		store
			.upsert_api_key("cursor", "account", b"rotated", 11)
			.expect("rotate Cursor credential");
		assert!(matches!(
			source.apply_cursor_headers(&stale, &mut headers),
			Err(BrokerCredentialSourceError::StaleGeneration { .. })
		));

		let lease = source.lease("devin").expect("lookup").expect("lease");
		let mut metadata = omp_llm_devin::wire::Metadata::default();
		source
			.apply_devin_metadata(&lease, &mut metadata, "account-jwt".to_owned())
			.expect("Devin metadata auth");
		assert_eq!(metadata.api_key, "devin-session-token$devin-secret");
		assert_eq!(metadata.user_jwt, "account-jwt");
	}
}
