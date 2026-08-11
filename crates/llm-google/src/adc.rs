//! Google Application Default Credential discovery and OAuth token minting.
//!
//! The engine sends token requests through the shared egress client and
//! delivers minted credentials to a caller-owned sink. Access tokens are never
//! returned from this module or represented by a serializable/debuggable public
//! type.

use std::{
	collections::BTreeMap,
	error::Error as StdError,
	fmt,
	future::Future,
	path::{Path, PathBuf},
	sync::Arc,
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use base64::{Engine as _, engine::general_purpose};
use bytes::Bytes;
use http::{Request, Response, StatusCode, header};
use http_body_util::{BodyExt as _, Full};
use omp_llm_egress::client::{Body, EgressClient, EgressError};
use ring::{rand::SystemRandom, signature};
use serde::Deserialize;
use tokio::sync::Mutex;
use tower::ServiceExt as _;
use zeroize::Zeroizing;

const TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";
const METADATA_TOKEN_ENDPOINT: &str =
	"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";
const METADATA_PROJECT_ENDPOINT: &str =
	"http://metadata.google.internal/computeMetadata/v1/project/project-id";
const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const JWT_GRANT: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";
const DEFAULT_REFRESH_SKEW: Duration = Duration::from_secs(60);
const EXPLICIT_TOKEN_LIFETIME: Duration = Duration::from_secs(3600);

/// Receives a borrowed ADC access token at the one-way daemon ingress.
///
/// Implementations must copy the bytes directly into their protected credential
/// store. The borrowed slice is valid only for the duration of `accept`.
pub trait AdcTokenSink {
	/// Failure produced by the protected credential store.
	type Error: StdError + Send + Sync + 'static;

	/// Stores a bearer token and its absolute expiry without exposing it back to
	/// the caller.
	fn accept(&self, token: &[u8], expires_at: SystemTime) -> Result<(), Self::Error>;
}

/// HTTP boundary used for ADC token and metadata requests.
///
/// Production uses [`EgressClient`]. Tests may implement this trait with a
/// deterministic in-memory endpoint; no provider module creates a direct HTTP
/// client.
pub trait AdcEgress: Clone + Send + Sync + 'static {
	/// Dispatch failure.
	type Error: StdError + Send + Sync + 'static;

	/// Sends one fully buffered request and returns a fully buffered response.
	fn execute(
		&self,
		request: Request<Body>,
	) -> impl Future<Output = Result<Response<Bytes>, Self::Error>> + Send + 'static;
}

/// Error while buffering a shared-egress token response.
#[derive(Debug, thiserror::Error)]
pub enum SharedAdcEgressError {
	/// The shared HTTP egress rejected or failed the request.
	#[error("shared egress failed")]
	Dispatch(#[source] EgressError),
	/// The token response body failed while being collected.
	#[error("shared egress response body failed")]
	Body(#[source] hyper::Error),
}

impl AdcEgress for EgressClient {
	type Error = SharedAdcEgressError;

	fn execute(
		&self,
		request: Request<Body>,
	) -> impl Future<Output = Result<Response<Bytes>, Self::Error>> + Send + 'static {
		let service = self.clone();
		async move {
			let response = service
				.oneshot(request)
				.await
				.map_err(SharedAdcEgressError::Dispatch)?;
			let (parts, body) = response.into_parts();
			let body = body
				.collect()
				.await
				.map_err(SharedAdcEgressError::Body)?
				.to_bytes();
			Ok(Response::from_parts(parts, body))
		}
	}
}

/// Non-secret project and location selected for a Vertex deployment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdcRoute {
	/// Google Cloud project id.
	pub project:  String,
	/// Vertex location, such as `us-central1` or `global`.
	pub location: String,
}

/// Failure while discovering or minting Google ADC credentials.
#[derive(Debug, thiserror::Error)]
pub enum AdcError {
	/// A configured credential file could not be read.
	#[error("could not read Google credential file {path}")]
	Read {
		/// Credential file path.
		path:   PathBuf,
		/// Filesystem failure.
		#[source]
		source: std::io::Error,
	},
	/// A credential document was malformed or unsupported.
	#[error("malformed Google credentials: {0}")]
	Malformed(&'static str),
	/// No supported ADC source was available.
	#[error("Google Application Default Credentials were not found")]
	Missing,
	/// Vertex project discovery failed.
	#[error("Vertex project was not configured and could not be discovered")]
	MissingProject,
	/// Vertex location discovery failed.
	#[error("Vertex location was not configured")]
	MissingLocation,
	/// The shared egress service failed without retaining request material.
	#[error("Google credential egress failed")]
	Egress,
	/// Google rejected a token exchange.
	#[error("Google OAuth token exchange failed with HTTP {0}")]
	TokenStatus(StatusCode),
	/// A token response did not contain the required fields.
	#[error("malformed Google OAuth token response")]
	TokenResponse,
	/// An outbound URI or request could not be constructed.
	#[error("could not construct Google credential request")]
	Request,
	/// The service-account private key was not valid PKCS#8 RSA material.
	#[error("invalid Google service-account private key")]
	PrivateKey,
}

/// Failure to deliver an ADC token into a protected credential store.
#[derive(Debug, thiserror::Error)]
pub enum AdcIntoError<E: StdError + Send + Sync + 'static> {
	/// Credential discovery or token exchange failed.
	#[error(transparent)]
	Adc(#[from] AdcError),
	/// The protected credential store rejected the token.
	#[error("ADC token sink rejected the credential")]
	Sink(#[source] E),
}

#[derive(Clone)]
struct Settings {
	env: Arc<BTreeMap<String, String>>,
	application_default_path: PathBuf,
	token_endpoint: String,
	metadata_token_endpoint: String,
	metadata_project_endpoint: String,
	refresh_skew: Duration,
}

impl Settings {
	fn from_process() -> Self {
		let env = std::env::vars().collect::<BTreeMap<_, _>>();
		let application_default_path = env
			.get("CLOUDSDK_CONFIG")
			.map(PathBuf::from)
			.or_else(|| {
				std::env::var_os("HOME")
					.map(PathBuf::from)
					.map(|path| path.join(".config/gcloud"))
			})
			.unwrap_or_default()
			.join("application_default_credentials.json");
		let refresh_skew = env
			.get("GOOGLE_VERTEX_REFRESH_SKEW_MS")
			.and_then(|value| value.parse::<u64>().ok())
			.map_or(DEFAULT_REFRESH_SKEW, Duration::from_millis);
		Self {
			env: Arc::new(env),
			application_default_path,
			token_endpoint: TOKEN_ENDPOINT.to_owned(),
			metadata_token_endpoint: METADATA_TOKEN_ENDPOINT.to_owned(),
			metadata_project_endpoint: METADATA_PROJECT_ENDPOINT.to_owned(),
			refresh_skew,
		}
	}
}

struct CachedToken {
	source:     String,
	token:      Zeroizing<Vec<u8>>,
	expires_at: SystemTime,
}

/// ADC resolver and expiry-aware token minting engine.
///
/// Clones share one cache and one refresh critical section. Its custom `Debug`
/// implementation deliberately exposes neither environment values, credential
/// JSON, access tokens, nor request bodies.
#[derive(Clone)]
pub struct AdcEngine<E> {
	egress:   E,
	settings: Settings,
	cache:    Arc<Mutex<Option<CachedToken>>>,
}

impl<E> fmt::Debug for AdcEngine<E> {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("AdcEngine")
			.field("credentials", &"[REDACTED]")
			.finish_non_exhaustive()
	}
}

impl<E: AdcEgress> AdcEngine<E> {
	/// Constructs an engine using process ADC environment and the shared egress
	/// service supplied by the daemon.
	#[must_use]
	pub fn new(egress: E) -> Self {
		Self { egress, settings: Settings::from_process(), cache: Arc::new(Mutex::new(None)) }
	}

	/// Resolves a cached or newly minted token and sends it directly to `sink`.
	pub async fn authorize_into<S: AdcTokenSink>(
		&self,
		sink: &S,
	) -> Result<(), AdcIntoError<S::Error>> {
		self.deliver(false, sink).await
	}

	/// Forces a new token exchange and sends the replacement directly to `sink`.
	pub async fn refresh_into<S: AdcTokenSink>(
		&self,
		sink: &S,
	) -> Result<(), AdcIntoError<S::Error>> {
		self.deliver(true, sink).await
	}

	/// Discovers non-secret Vertex route values using explicit values,
	/// environment variables, credential project metadata, then the metadata
	/// server.
	pub async fn resolve_route(
		&self,
		project: Option<&str>,
		location: Option<&str>,
	) -> Result<AdcRoute, AdcError> {
		let location = first_nonempty(location, &self.settings.env, &[
			"GOOGLE_VERTEX_LOCATION",
			"GOOGLE_CLOUD_LOCATION",
			"VERTEX_LOCATION",
		])
		.ok_or(AdcError::MissingLocation)?;
		let project = if let Some(project) = first_nonempty(project, &self.settings.env, &[
			"GOOGLE_CLOUD_PROJECT",
			"GCP_PROJECT",
			"GCLOUD_PROJECT",
		]) {
			project
		} else if let Some(project) = self.project_from_file().await? {
			project
		} else {
			self
				.metadata_project()
				.await?
				.ok_or(AdcError::MissingProject)?
		};
		Ok(AdcRoute { project, location })
	}

	async fn deliver<S: AdcTokenSink>(
		&self,
		force: bool,
		sink: &S,
	) -> Result<(), AdcIntoError<S::Error>> {
		let source = self.load_source().await?;
		let source_key = source.key().to_owned();
		let now = SystemTime::now();
		let mut cache = self.cache.lock().await;
		let fresh = !force
			&& cache.as_ref().is_some_and(|cached| {
				cached.source == source_key
					&& cached
						.expires_at
						.duration_since(now)
						.is_ok_and(|remaining| remaining > self.settings.refresh_skew)
			});
		if !fresh {
			*cache = Some(self.exchange(source).await?);
		}
		let cached = cache.as_ref().ok_or(AdcError::TokenResponse)?;
		sink
			.accept(cached.token.as_slice(), cached.expires_at)
			.map_err(AdcIntoError::Sink)
	}

	async fn load_source(&self) -> Result<CredentialSource, AdcError> {
		if let Some(token) =
			env_first(&self.settings.env, &["GOOGLE_CLOUD_ACCESS_TOKEN", "CLOUDSDK_AUTH_ACCESS_TOKEN"])
		{
			return Ok(CredentialSource::Explicit(Zeroizing::new(token.as_bytes().to_vec())));
		}
		if let Some(document) = self.settings.env.get("GOOGLE_SERVICE_ACCOUNT_JSON") {
			return parse_credentials(document.as_bytes(), "env:GOOGLE_SERVICE_ACCOUNT_JSON");
		}
		if let Some(path) = self.settings.env.get("GOOGLE_APPLICATION_CREDENTIALS") {
			return self.read_credentials(Path::new(path), "env-file").await;
		}
		match self
			.read_optional_credentials(&self.settings.application_default_path, "application-default")
			.await?
		{
			Some(source) => Ok(source),
			None => Ok(CredentialSource::Metadata),
		}
	}

	async fn project_from_file(&self) -> Result<Option<String>, AdcError> {
		if let Some(document) = self.settings.env.get("GOOGLE_SERVICE_ACCOUNT_JSON") {
			let source = parse_credentials(document.as_bytes(), "env:GOOGLE_SERVICE_ACCOUNT_JSON")?;
			return Ok(source.project().map(ToOwned::to_owned));
		}
		let path = self
			.settings
			.env
			.get("GOOGLE_APPLICATION_CREDENTIALS").map_or_else(|| self.settings.application_default_path.clone(), PathBuf::from);
		let Some(source) = self
			.read_optional_credentials(&path, "project-discovery")
			.await?
		else {
			return Ok(None);
		};
		Ok(source.project().map(ToOwned::to_owned))
	}

	async fn read_credentials(
		&self,
		path: &Path,
		label: &str,
	) -> Result<CredentialSource, AdcError> {
		let bytes = tokio::fs::read(path)
			.await
			.map_err(|source| AdcError::Read { path: path.to_owned(), source })?;
		parse_credentials(&bytes, &format!("{label}:{}", path.display()))
	}

	async fn read_optional_credentials(
		&self,
		path: &Path,
		label: &str,
	) -> Result<Option<CredentialSource>, AdcError> {
		match tokio::fs::read(path).await {
			Ok(bytes) => parse_credentials(&bytes, &format!("{label}:{}", path.display())).map(Some),
			Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
			Err(source) => Err(AdcError::Read { path: path.to_owned(), source }),
		}
	}

	async fn exchange(&self, source: CredentialSource) -> Result<CachedToken, AdcError> {
		match source {
			CredentialSource::Explicit(token) => Ok(CachedToken {
				source: "environment-access-token".to_owned(),
				token,
				expires_at: SystemTime::now() + EXPLICIT_TOKEN_LIFETIME,
			}),
			CredentialSource::ServiceAccount { source, credentials } => {
				let assertion = sign_assertion(&credentials, &self.settings.token_endpoint)?;
				let form = url::form_urlencoded::Serializer::new(String::new())
					.append_pair("grant_type", JWT_GRANT)
					.append_pair("assertion", &assertion)
					.finish();
				self.exchange_form(source, form).await
			},
			CredentialSource::AuthorizedUser { source, credentials } => {
				let form = url::form_urlencoded::Serializer::new(String::new())
					.append_pair("client_id", &credentials.client_id)
					.append_pair("client_secret", &credentials.client_secret)
					.append_pair("refresh_token", &credentials.refresh_token)
					.append_pair("grant_type", "refresh_token")
					.finish();
				self.exchange_form(source, form).await
			},
			CredentialSource::Metadata => {
				let request = Request::get(&self.settings.metadata_token_endpoint)
					.header("Metadata-Flavor", "Google")
					.body(Full::new(Bytes::new()))
					.map_err(|_| AdcError::Request)?;
				let response = self.send(request).await?;
				if !response.status().is_success() {
					return Err(if response.status() == StatusCode::NOT_FOUND {
						AdcError::Missing
					} else {
						AdcError::TokenStatus(response.status())
					});
				}
				parse_token_response("metadata".to_owned(), response.body())
			},
		}
	}

	async fn exchange_form(&self, source: String, form: String) -> Result<CachedToken, AdcError> {
		let request = Request::post(&self.settings.token_endpoint)
			.header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
			.body(Full::new(Bytes::from(form)))
			.map_err(|_| AdcError::Request)?;
		let response = self.send(request).await?;
		if !response.status().is_success() {
			return Err(AdcError::TokenStatus(response.status()));
		}
		parse_token_response(source, response.body())
	}

	async fn metadata_project(&self) -> Result<Option<String>, AdcError> {
		let request = Request::get(&self.settings.metadata_project_endpoint)
			.header("Metadata-Flavor", "Google")
			.body(Full::new(Bytes::new()))
			.map_err(|_| AdcError::Request)?;
		let response = self.send(request).await?;
		if response.status() == StatusCode::NOT_FOUND {
			return Ok(None);
		}
		if !response.status().is_success() {
			return Err(AdcError::TokenStatus(response.status()));
		}
		let project = std::str::from_utf8(response.body())
			.map_err(|_| AdcError::Malformed("metadata project id was not UTF-8"))?
			.trim();
		Ok((!project.is_empty()).then(|| project.to_owned()))
	}

	async fn send(&self, request: Request<Body>) -> Result<Response<Bytes>, AdcError> {
		self
			.egress
			.execute(request)
			.await
			.map_err(|_| AdcError::Egress)
	}
}

struct ServiceAccount {
	client_email:   String,
	private_key:    Zeroizing<String>,
	private_key_id: Option<String>,
	project_id:     Option<String>,
}

#[derive(Deserialize)]
struct WireServiceAccount {
	client_email:   String,
	private_key:    String,
	private_key_id: Option<String>,
	project_id:     Option<String>,
}

struct AuthorizedUser {
	client_id:        String,
	client_secret:    Zeroizing<String>,
	refresh_token:    Zeroizing<String>,
	quota_project_id: Option<String>,
}

#[derive(Deserialize)]
struct WireAuthorizedUser {
	client_id:        String,
	client_secret:    String,
	refresh_token:    String,
	quota_project_id: Option<String>,
}

enum CredentialSource {
	Explicit(Zeroizing<Vec<u8>>),
	ServiceAccount { source: String, credentials: ServiceAccount },
	AuthorizedUser { source: String, credentials: AuthorizedUser },
	Metadata,
}

impl CredentialSource {
	fn key(&self) -> &str {
		match self {
			Self::Explicit(_) => "environment-access-token",
			Self::ServiceAccount { source, .. } | Self::AuthorizedUser { source, .. } => source,
			Self::Metadata => "metadata",
		}
	}

	fn project(&self) -> Option<&str> {
		match self {
			Self::ServiceAccount { credentials, .. } => credentials.project_id.as_deref(),
			Self::AuthorizedUser { credentials, .. } => credentials.quota_project_id.as_deref(),
			Self::Explicit(_) | Self::Metadata => None,
		}
	}
}

fn parse_credentials(bytes: &[u8], source: &str) -> Result<CredentialSource, AdcError> {
	let value: serde_json::Value = serde_json::from_slice(bytes)
		.map_err(|_| AdcError::Malformed("credential document was not valid JSON"))?;
	match value.get("type").and_then(serde_json::Value::as_str) {
		Some("service_account") => {
			let wire: WireServiceAccount = serde_json::from_value(value)
				.map_err(|_| AdcError::Malformed("service_account fields were missing or invalid"))?;
			if wire.client_email.is_empty() || wire.private_key.is_empty() {
				return Err(AdcError::Malformed("service_account fields were empty"));
			}
			let credentials = ServiceAccount {
				client_email:   wire.client_email,
				private_key:    Zeroizing::new(wire.private_key),
				private_key_id: wire.private_key_id,
				project_id:     wire.project_id,
			};
			Ok(CredentialSource::ServiceAccount { source: source.to_owned(), credentials })
		},
		Some("authorized_user") => {
			let wire: WireAuthorizedUser = serde_json::from_value(value)
				.map_err(|_| AdcError::Malformed("authorized_user fields were missing or invalid"))?;
			if wire.client_id.is_empty()
				|| wire.client_secret.is_empty()
				|| wire.refresh_token.is_empty()
			{
				return Err(AdcError::Malformed("authorized_user fields were empty"));
			}
			let credentials = AuthorizedUser {
				client_id:        wire.client_id,
				client_secret:    Zeroizing::new(wire.client_secret),
				refresh_token:    Zeroizing::new(wire.refresh_token),
				quota_project_id: wire.quota_project_id,
			};
			Ok(CredentialSource::AuthorizedUser { source: source.to_owned(), credentials })
		},
		Some(_) => Err(AdcError::Malformed("credential type is unsupported")),
		None => Err(AdcError::Malformed("credential type was missing")),
	}
}

fn sign_assertion(credentials: &ServiceAccount, audience: &str) -> Result<String, AdcError> {
	let now = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_secs();
	let mut header = serde_json::json!({ "alg": "RS256", "typ": "JWT" });
	if let Some(key_id) = &credentials.private_key_id {
		header["kid"] = serde_json::Value::String(key_id.clone());
	}
	let claims = serde_json::json!({
		"iss": credentials.client_email,
		"scope": CLOUD_PLATFORM_SCOPE,
		"aud": audience,
		"iat": now,
		"exp": now.saturating_add(3600),
	});
	let encoded_header = general_purpose::URL_SAFE_NO_PAD
		.encode(serde_json::to_vec(&header).map_err(|_| AdcError::PrivateKey)?);
	let encoded_claims = general_purpose::URL_SAFE_NO_PAD
		.encode(serde_json::to_vec(&claims).map_err(|_| AdcError::PrivateKey)?);
	let signing_input = format!("{encoded_header}.{encoded_claims}");
	let der = decode_private_key(&credentials.private_key)?;
	let key = signature::RsaKeyPair::from_pkcs8(&der).map_err(|_| AdcError::PrivateKey)?;
	let mut signature_bytes = Zeroizing::new(vec![0_u8; key.public().modulus_len()]);
	key.sign(
		&signature::RSA_PKCS1_SHA256,
		&SystemRandom::new(),
		signing_input.as_bytes(),
		&mut signature_bytes,
	)
	.map_err(|_| AdcError::PrivateKey)?;
	Ok(format!(
		"{signing_input}.{}",
		general_purpose::URL_SAFE_NO_PAD.encode(signature_bytes.as_slice())
	))
}

fn decode_private_key(pem: &str) -> Result<Zeroizing<Vec<u8>>, AdcError> {
	let body = pem
		.lines()
		.filter(|line| !line.starts_with("-----"))
		.collect::<String>();
	if body.is_empty() {
		return Err(AdcError::PrivateKey);
	}
	general_purpose::STANDARD
		.decode(body)
		.map(Zeroizing::new)
		.map_err(|_| AdcError::PrivateKey)
}

#[derive(Deserialize)]
struct WireTokenResponse {
	access_token: String,
	expires_in:   u64,
}

fn parse_token_response(source: String, body: &[u8]) -> Result<CachedToken, AdcError> {
	let response: WireTokenResponse =
		serde_json::from_slice(body).map_err(|_| AdcError::TokenResponse)?;
	if response.access_token.is_empty() || response.expires_in == 0 {
		return Err(AdcError::TokenResponse);
	}
	let expires_at = SystemTime::now()
		.checked_add(Duration::from_secs(response.expires_in))
		.ok_or(AdcError::TokenResponse)?;
	Ok(CachedToken { source, token: Zeroizing::new(response.access_token.into_bytes()), expires_at })
}

fn env_first<'a>(env: &'a BTreeMap<String, String>, names: &[&str]) -> Option<&'a str> {
	names
		.iter()
		.find_map(|name| env.get(*name).map(String::as_str))
		.filter(|value| !value.is_empty())
}

fn first_nonempty(
	explicit: Option<&str>,
	env: &BTreeMap<String, String>,
	names: &[&str],
) -> Option<String> {
	explicit
		.filter(|value| !value.is_empty())
		.map(ToOwned::to_owned)
		.or_else(|| env_first(env, names).map(ToOwned::to_owned))
}

#[cfg(test)]
mod tests {
	use std::{collections::VecDeque, convert::Infallible, sync::Arc};

	use http::HeaderMap;
	use parking_lot::Mutex as StdMutex;
	use tempfile::TempDir;

	use super::*;

	const TEST_RSA_KEY: &str = r"-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQClLQ9oJWuxPRVr
xVrG1tCi2GEE/2oee5KtJA4REC4fI2bnmU9MJusfUx0k1g4XkiYJ+4CErJIZDr7T
j9bclMyLx5jII5nzBeRtg4aA65NfZVeQ1qgd9Z3QO8dFVJVC7asaA0ydBpxx2ZvR
YJOb2AVY9jamYpOrwri12nWYg6mMtmCk8G/UaMT6/1o1rA+8UkvrPttwSkYaoYcY
EWbqMH3dLiRJaGwqO812hXSqJEV5nRfbAUqkKDmeJ5VcGKaxpZz8YCx0LbdVvAfG
2lO4B3bTihRyn8cqvgB/iBASOlcPIKreQ0qDUjBEUmUoZv03jHCCIZFa+GxvgEV+
58D2b2GnAgMBAAECggEAFljaVMTQpSdr1oDaSedA+Ec+GRGqpyUg2xFIYJadJtQA
qs+FzaUWT86hisegcHVIIDbz/v86D5nRx4hrvBGFpa5YxVB2a6LIcj3xMfqtSE11
uMAnTqZZtj/gM0kWSKUkbl3eklVqpRyZMDgDayT21D+7dRdb00ks+a2Xa2L6IByf
VhBApM2jYTyvadbhtnSKiIbe1poUrbp6QWrCqfUALWO3OXG1SlyTqs0rZvTI98MI
PmTLLBlImwsTSrQdnuQY1gnnj0BLK+Q5FWEijQZgW09BGaNjInFFp1aT1dSzXfzZ
eITyxTatG8xTMPiYm2qrYd6g7E/FCSOUVQxbMf/yQQKBgQDkCZwIa5qd4keCA66R
5onwK7YoQp55ajxPPX9y8wflvEzS0OBCqtkkDfJxGYzbNvf8UNZoFXkOUyGW7LxK
bekRVjl/1sLDyiOB7U+4Y/qFGhvLcSH+WWC1vmgSVyCNKhhIbWKWW7ryLopclKGu
TFfiDzt9Af865Wyp38CT9LYtswKBgQC5biWhPR+HSupVMlju3CV9oe2nx41yYeMt
p8OmOIjtXfoNuPKaLJCHeQbJFdwr60wG8Qm7poARYCrEcmFuu6QlkoQO+loY18JD
W+cTpMn1mi7euv2br5drJo5Ohc50lhv0zXfmIykTg4gBQ5bWFCXO/c/m6gHa0dDZ
96ysegyKPQKBgQDjEt6ZU+1HQshKIzh2eMbqrdxaAtyjsrIThf2fjXpTvkoRs4Vd
XZuUV38QOI0WzYnrauPWCWveY9GS5HIq+3+Wj/H55vVS2bq56oHz7zrLx8/dqe5b
xMyUreIcQT5c04oStTny16009Ds7LZZCZistJFXsiUyKbWLjVbgCnS+8GQKBgQCR
zT0DYjdHPy2wXc01y54jAc8HjM34cWWbAX3CVlO8KJe0cIc5mO7vxscCGBEt6261
SpP3m7y5bN9T5ggcdKhl7qWtzUZIoGYcZsf0Vy+B0YEnGurMnq21z/Q3Y9jpLRrA
S0sKhv0GXfbz33xbyi3MayAtFjTtJOtOaAO6/qCblQKBgFruLcu0g17WzSf6pFN6
El3lKjmLHAEhMJYEG29rWkHPVDzyQtstiCYJEu7zPrIdm0PwYEv3zLTiz1kBHfjC
4cRPorNhVA8fFhXKyiPiWy5SWm7cXo6lo2ZiNQBW947RmjPfj4+D7W7KSkK3zDBT
SoiVr2PnTe/N9FN9ZNov5wmt
-----END PRIVATE KEY-----";

	#[derive(Clone, Default)]
	struct MockEgress {
		requests:  Arc<StdMutex<Vec<(String, HeaderMap, Bytes)>>>,
		responses: Arc<StdMutex<VecDeque<Response<Bytes>>>>,
	}

	impl MockEgress {
		fn response(self, status: StatusCode, body: impl Into<Bytes>) -> Self {
			self.responses.lock().push_back(
				Response::builder()
					.status(status)
					.body(body.into())
					.unwrap(),
			);
			self
		}
	}

	impl AdcEgress for MockEgress {
		type Error = Infallible;

		fn execute(
			&self,
			request: Request<Body>,
		) -> impl Future<Output = Result<Response<Bytes>, Self::Error>> + Send + 'static {
			let requests = Arc::clone(&self.requests);
			let responses = Arc::clone(&self.responses);
			async move {
				let (parts, body) = request.into_parts();
				let bytes = body.collect().await.unwrap().to_bytes();
				requests
					.lock()
					.push((parts.uri.to_string(), parts.headers, bytes));
				Ok(responses.lock().pop_front().unwrap())
			}
		}
	}

	#[derive(Default)]
	struct Sink(StdMutex<Vec<(Vec<u8>, SystemTime)>>);

	impl AdcTokenSink for Sink {
		type Error = Infallible;

		fn accept(&self, token: &[u8], expires_at: SystemTime) -> Result<(), Self::Error> {
			self.0.lock().push((token.to_vec(), expires_at));
			Ok(())
		}
	}

	#[derive(Default)]
	struct BrokerSink(StdMutex<Vec<u8>>);

	impl AdcTokenSink for BrokerSink {
		type Error = Infallible;

		fn accept(&self, token: &[u8], _expires_at: SystemTime) -> Result<(), Self::Error> {
			self.0.lock().extend_from_slice(token);
			Ok(())
		}
	}

	impl BrokerSink {
		fn inject(&self, request: &mut Request<Body>) {
			let token = self.0.lock();
			let mut value = http::HeaderValue::from_bytes(
				[b"Bearer ".as_slice(), token.as_slice()]
					.concat()
					.as_slice(),
			)
			.unwrap();
			value.set_sensitive(true);
			request.headers_mut().insert(header::AUTHORIZATION, value);
		}
	}

	fn engine(
		egress: MockEgress,
		env: BTreeMap<String, String>,
		adc: PathBuf,
	) -> AdcEngine<MockEgress> {
		AdcEngine {
			egress,
			settings: Settings {
				env: Arc::new(env),
				application_default_path: adc,
				token_endpoint: "https://oauth.test/token".to_owned(),
				metadata_token_endpoint: "http://metadata.test/token".to_owned(),
				metadata_project_endpoint: "http://metadata.test/project".to_owned(),
				refresh_skew: DEFAULT_REFRESH_SKEW,
			},
			cache: Arc::new(Mutex::new(None)),
		}
	}

	fn authorized_user(secret: &str, refresh: &str, project: &str) -> String {
		serde_json::json!({
			"type": "authorized_user",
			"client_id": "client-id",
			"client_secret": secret,
			"refresh_token": refresh,
			"quota_project_id": project,
		})
		.to_string()
	}
	#[tokio::test]
	async fn service_account_signs_rs256_jwt_and_exchanges_it_through_egress() {
		let mut env = BTreeMap::new();
		env.insert(
			"GOOGLE_SERVICE_ACCOUNT_JSON".to_owned(),
			serde_json::json!({
				"type": "service_account",
				"client_email": "service@example.iam.gserviceaccount.com",
				"private_key": TEST_RSA_KEY,
				"private_key_id": "key-id",
				"project_id": "service-project",
			})
			.to_string(),
		);
		let egress = MockEgress::default()
			.response(StatusCode::OK, r#"{"access_token":"service-token","expires_in":3600}"#);
		let engine = engine(egress.clone(), env, PathBuf::new());
		engine.authorize_into(&Sink::default()).await.unwrap();

		let requests = egress.requests.lock();
		assert_eq!(requests.len(), 1);
		assert_eq!(requests[0].0, "https://oauth.test/token");
		let assertion = url::form_urlencoded::parse(&requests[0].2)
			.find(|(key, _)| key == "assertion")
			.map(|(_, value)| value.into_owned())
			.unwrap();
		let segments = assertion.split('.').collect::<Vec<_>>();
		assert_eq!(segments.len(), 3);
		let claims: serde_json::Value = serde_json::from_slice(
			&general_purpose::URL_SAFE_NO_PAD
				.decode(segments[1])
				.unwrap(),
		)
		.unwrap();
		assert_eq!(claims["iss"], "service@example.iam.gserviceaccount.com");
		assert_eq!(claims["scope"], CLOUD_PLATFORM_SCOPE);
		assert_eq!(claims["aud"], "https://oauth.test/token");
	}

	#[tokio::test]
	async fn explicit_environment_token_precedes_service_account_and_is_redacted() {
		let tmp = TempDir::new().unwrap();
		let credential_path = tmp.path().join("service.json");
		std::fs::write(&credential_path, "not-json").unwrap();
		let mut env = BTreeMap::new();
		env.insert("GOOGLE_CLOUD_ACCESS_TOKEN".to_owned(), "environment-secret".to_owned());
		env.insert(
			"GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
			credential_path.display().to_string(),
		);
		let egress = MockEgress::default();
		let engine = engine(egress.clone(), env, tmp.path().join("adc.json"));
		let sink = Sink::default();
		engine.authorize_into(&sink).await.unwrap();
		assert_eq!(sink.0.lock()[0].0, b"environment-secret");
		assert!(egress.requests.lock().is_empty());
		let debug = format!("{engine:?}");
		assert!(!debug.contains("environment-secret"));
		assert!(debug.contains("[REDACTED]"));
	}

	#[tokio::test]
	async fn application_default_refreshes_at_expiry_and_never_returns_token() {
		let tmp = TempDir::new().unwrap();
		let adc = tmp.path().join("application_default_credentials.json");
		std::fs::write(&adc, authorized_user("client-secret", "refresh-secret", "adc-project"))
			.unwrap();
		let egress = MockEgress::default()
			.response(StatusCode::OK, r#"{"access_token":"first-token","expires_in":30}"#)
			.response(StatusCode::OK, r#"{"access_token":"second-token","expires_in":3600}"#);
		let engine = engine(egress.clone(), BTreeMap::new(), adc);
		let sink = Sink::default();
		engine.authorize_into(&sink).await.unwrap();
		engine.authorize_into(&sink).await.unwrap();
		let tokens = sink
			.0
			.lock()
			.iter()
			.map(|entry| entry.0.clone())
			.collect::<Vec<_>>();
		assert_eq!(tokens, [b"first-token".to_vec(), b"second-token".to_vec()]);
		let requests = egress.requests.lock();
		assert_eq!(requests.len(), 2);
		let form = std::str::from_utf8(&requests[0].2).unwrap();
		assert!(form.contains("grant_type=refresh_token"));
		assert!(form.contains("refresh_token=refresh-secret"));
	}

	#[tokio::test]
	async fn token_and_model_mock_endpoints_prove_broker_header_injection() {
		let tmp = TempDir::new().unwrap();
		let adc = tmp.path().join("adc.json");
		std::fs::write(&adc, authorized_user("secret", "refresh", "project")).unwrap();
		let egress = MockEgress::default()
			.response(StatusCode::OK, r#"{"access_token":"broker-token","expires_in":3600}"#)
			.response(StatusCode::OK, Bytes::from_static(b"model-ok"));
		let engine = engine(egress.clone(), BTreeMap::new(), adc);
		let sink = BrokerSink::default();
		engine.authorize_into(&sink).await.unwrap();

		let endpoint = crate::vertex::VertexDeployment::new("project", "us-central1")
			.unwrap()
			.stream_endpoint("gemini-2.5-pro")
			.unwrap();
		let mut request = Request::post(endpoint.as_str())
			.body(Full::new(Bytes::from_static(b"{}")))
			.unwrap();
		sink.inject(&mut request);
		egress.execute(request).await.unwrap();

		let requests = egress.requests.lock();
		assert_eq!(requests[0].0, "https://oauth.test/token");
		assert_eq!(
			requests[1].0,
			"https://us-central1-aiplatform.googleapis.com/v1/projects/project/locations/us-central1/publishers/google/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
		);
		let authorization = requests[1].1.get(header::AUTHORIZATION).unwrap();
		assert_eq!(authorization, "Bearer broker-token");
		assert!(authorization.is_sensitive());
	}

	#[tokio::test]
	async fn explicit_project_location_precede_environment_and_adc_project() {
		let tmp = TempDir::new().unwrap();
		let adc_path = tmp.path().join("adc.json");
		std::fs::write(&adc_path, authorized_user("secret", "refresh", "adc-project")).unwrap();
		let mut env = BTreeMap::new();
		env.insert("GOOGLE_CLOUD_PROJECT".to_owned(), "env-project".to_owned());
		env.insert("GOOGLE_VERTEX_LOCATION".to_owned(), "env-location".to_owned());
		let route_engine = engine(MockEgress::default(), env, adc_path);
		assert_eq!(
			route_engine
				.resolve_route(Some("explicit-project"), Some("global"))
				.await
				.unwrap(),
			AdcRoute { project: "explicit-project".to_owned(), location: "global".to_owned() }
		);

		let mut adc_env = BTreeMap::new();
		adc_env.insert("GOOGLE_VERTEX_LOCATION".to_owned(), "europe-west4".to_owned());
		let adc_engine = engine(MockEgress::default(), adc_env, tmp.path().join("adc.json"));
		assert_eq!(adc_engine.resolve_route(None, None).await.unwrap(), AdcRoute {
			project:  "adc-project".to_owned(),
			location: "europe-west4".to_owned(),
		});
	}

	#[tokio::test]
	async fn malformed_credentials_fail_before_egress_without_echoing_secrets() {
		let tmp = TempDir::new().unwrap();
		let adc = tmp.path().join("adc.json");
		std::fs::write(&adc, r#"{"type":"authorized_user","client_secret":"do-not-echo"}"#).unwrap();
		let egress = MockEgress::default();
		let engine = engine(egress.clone(), BTreeMap::new(), adc);
		let error = engine.authorize_into(&Sink::default()).await.unwrap_err();
		assert!(!error.to_string().contains("do-not-echo"));
		assert!(egress.requests.lock().is_empty());
	}
}
