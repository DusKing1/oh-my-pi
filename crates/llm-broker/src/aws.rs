//! AWS shared-config credential discovery and STS role resolution.
//!
//! Credential bytes remain inside the broker: callers provide a sink that copies
//! resolved material directly into the protected store. STS requests use the
//! workspace egress client and the broker's sealed SigV4 implementation.

use std::{
	collections::{BTreeMap, HashSet},
	error::Error as StdError,
	fmt,
	future::Future,
	path::{Path, PathBuf},
	sync::Arc,
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use http::{Request, Response, StatusCode, header};
use http_body_util::{BodyExt as _, Full};
use jiff::Timestamp;
use omp_core::Str;
use omp_llm_egress::{
	auth_inject::AwsSigV4Context,
	client::{Body, EgressClient, EgressError},
};
use serde::Deserialize;
use tokio::sync::Mutex;
use tower::ServiceExt as _;
use url::Url;

use crate::sealed::{AppliedAuth, AwsSigV4Error, Secret};

const REFRESH_SKEW_MS: u64 = 60_000;
const FILE_SESSION_CREDS_TTL_MS: u64 = 5 * 60_000;
const IMDS_TIMEOUT: Duration = Duration::from_secs(1);
const ECS_TASK_CREDENTIALS_BASE_URL: &str = "http://169.254.170.2/";
const IMDS_IPV4_BASE_URL: &str = "http://169.254.169.254/";
const IMDS_IPV6_BASE_URL: &str = "http://[fd00:ec2::254]/";

/// Receives resolved AWS key material at the broker's one-way secret ingress.
pub trait AwsCredentialSink {
	/// Failure produced by the protected credential store.
	type Error: StdError + Send + Sync + 'static;

	/// Copies resolved key material and its absolute expiry into protected storage.
	fn accept(
		&self,
		access_key_id: &[u8],
		secret_access_key: &[u8],
		session_token: Option<&[u8]>,
		expires_at_ms: u64,
	) -> Result<(), Self::Error>;
}

/// HTTP boundary used for STS and metadata requests.
pub trait AwsEgress: Clone + Send + Sync + 'static {
	/// Dispatch failure.
	type Error: StdError + Send + Sync + 'static;

	/// Sends one fully buffered request and returns a fully buffered response.
	fn execute(
		&self,
		request: Request<Body>,
	) -> impl Future<Output = Result<Response<Bytes>, Self::Error>> + Send + 'static;
}

/// Error while buffering a shared-egress AWS response.
#[derive(Debug, thiserror::Error)]
pub enum SharedAwsEgressError {
	/// The shared HTTP egress rejected or failed the request.
	#[error("shared egress failed")]
	Dispatch(#[source] EgressError),
	/// The response body failed while being collected.
	#[error("shared egress response body failed")]
	Body(#[source] hyper::Error),
}

impl AwsEgress for EgressClient {
	type Error = SharedAwsEgressError;

	fn execute(
		&self,
		request: Request<Body>,
	) -> impl Future<Output = Result<Response<Bytes>, Self::Error>> + Send + 'static {
		let service = self.clone();
		async move {
			let response = service
				.oneshot(request)
				.await
				.map_err(SharedAwsEgressError::Dispatch)?;
			let (parts, body) = response.into_parts();
			let body = body
				.collect()
				.await
				.map_err(SharedAwsEgressError::Body)?
				.to_bytes();
			Ok(Response::from_parts(parts, body))
		}
	}
}

/// Failure while discovering or minting AWS credentials.
#[derive(Debug, thiserror::Error)]
pub enum AwsError {
	/// A configured shared-credentials file could not be read.
	#[error("could not read AWS credential file {path}")]
	Read {
		/// Path that could not be read.
		path: PathBuf,
		/// Filesystem failure.
		#[source]
		source: std::io::Error,
	},
	/// No supported AWS credential source was available.
	#[error("AWS credentials were not found")]
	Missing,
	/// A role profile did not name a usable base source.
	#[error("AWS profile `{0}` has no usable role credential source")]
	MissingRoleSource(Str),
	/// A source profile does not contain supported credentials.
	#[error("AWS source profile `{0}` has no usable credentials")]
	MissingSourceProfile(Str),
	/// A source-profile role chain contains a cycle.
	#[error("AWS profile role chain contains a cycle at `{0}`")]
	ProfileCycle(Str),
	/// The named credential source is not supported.
	#[error("unsupported AWS credential_source `{0}`")]
	UnsupportedCredentialSource(Str),
	/// An MFA-gated role cannot be resolved non-interactively.
	#[error("AWS profile `{0}` requires unsupported interactive MFA")]
	MfaRequired(Str),
	/// A configured token or authorization file was empty or unreadable.
	#[error("AWS credential token file is unavailable")]
	TokenFile,
	/// A metadata endpoint URI or request was invalid.
	#[error("invalid AWS credential endpoint or request")]
	Request,
	/// The shared egress service failed without retaining credential material.
	#[error("AWS credential egress failed")]
	Egress,
	/// An AWS credential endpoint rejected the request.
	#[error("AWS credential endpoint returned HTTP {0}")]
	Status(StatusCode),
	/// An STS or metadata response omitted required credential fields or expiry.
	#[error("malformed AWS credential response")]
	Response,
	/// The broker could not sign an STS AssumeRole request.
	#[error("could not sign AWS AssumeRole request")]
	Signing(#[from] AwsSigV4Error),
}

/// Failure to deliver resolved AWS credentials into protected storage.
#[derive(Debug, thiserror::Error)]
pub enum AwsIntoError<E: Send + Sync + 'static> {
	/// Credential discovery or STS exchange failed.
	#[error(transparent)]
	Aws(#[from] AwsError),
	/// The protected credential store rejected the credential.
	#[error("AWS credential sink rejected the credential")]
	Sink(#[source] E),
}

#[derive(Clone)]
struct Settings {
	env:              Arc<BTreeMap<Str, Str>>,
	credentials_path: PathBuf,
	config_path:      PathBuf,
	dmi_root:         PathBuf,
}

impl Settings {
	fn from_process() -> Self {
		let env = std::env::vars()
			.map(|(name, value)| (Str::from(name), Str::from(value)))
			.collect::<BTreeMap<_, _>>();
		let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
		let credentials_path = env
			.get("AWS_SHARED_CREDENTIALS_FILE")
			.map_or_else(|| home.join(".aws/credentials"), |path| PathBuf::from(path.as_str()));
		let config_path = env
			.get("AWS_CONFIG_FILE")
			.map_or_else(|| home.join(".aws/config"), |path| PathBuf::from(path.as_str()));
		Self { env: Arc::new(env), credentials_path, config_path, dmi_root: PathBuf::from("/") }
	}

	fn env(&self, name: &str) -> Option<&str> {
		self.env.get(name).map(Str::as_str).filter(|value| !value.is_empty())
	}

	fn profile(&self) -> &str {
		self.env("AWS_PROFILE").unwrap_or("default")
	}

	fn loads_config(&self) -> bool {
		self.env("AWS_PROFILE").is_some()
			|| self
				.env("AWS_SDK_LOAD_CONFIG")
				.is_some_and(|value| value.eq_ignore_ascii_case("true") || value == "1")
	}
}

struct Credentials {
	access_key_id:     Secret,
	secret_access_key: Secret,
	session_token:     Option<Secret>,
	expires_at_ms:     Option<u64>,
}

impl Credentials {
	fn from_values(
		access_key_id: &[u8],
		secret_access_key: &[u8],
		session_token: Option<&[u8]>,
		expires_at_ms: Option<u64>,
	) -> Self {
		Self {
			access_key_id:     Secret::new(access_key_id),
			secret_access_key: Secret::new(secret_access_key),
			session_token:     session_token.map(Secret::new),
			expires_at_ms,
		}
	}

	fn is_fresh(&self, now_ms: u64) -> bool {
		self.expires_at_ms
			.is_none_or(|expires| expires.saturating_sub(REFRESH_SKEW_MS) > now_ms)
	}

	fn deliver<S: AwsCredentialSink>(&self, sink: &S) -> Result<(), S::Error> {
		sink.accept(
			self.access_key_id.expose(),
			self.secret_access_key.expose(),
			self.session_token.as_ref().map(Secret::expose),
			self.expires_at_ms.unwrap_or(0),
		)
	}
}

struct CacheEntry {
	profile:     Str,
	region:      Str,
	credentials: Credentials,
}

/// Expiry-aware AWS shared-config and STS credential resolver.
///
/// Clones share a single-flight cache. Temporary credentials are refreshed 60
/// seconds before their required expiration; static credentials remain cached.
#[derive(Clone)]
pub struct AwsEngine<E> {
	egress:   E,
	settings: Settings,
	cache:    Arc<Mutex<Option<CacheEntry>>>,
}

impl<E> fmt::Debug for AwsEngine<E> {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("AwsEngine")
			.field("credentials", &"[REDACTED]")
			.finish_non_exhaustive()
	}
}

impl<E: AwsEgress> AwsEngine<E> {
	/// Constructs an engine using process AWS environment and shared-config paths.
	#[must_use]
	pub fn new(egress: E) -> Self {
		Self { egress, settings: Settings::from_process(), cache: Arc::new(Mutex::new(None)) }
	}
	/// Resolves the AWS region from environment, selected profile, then the SDK
	/// default used by STS and Bedrock.
	pub fn region(&self) -> Result<Str, AwsError> {
		self.resolve_region().map(Str::from)
	}


	/// Reports whether the selected AWS chain has a source that is ready to try.
	///
	/// `credential_source` profiles are gated by the named source's environment
	/// readiness, matching the resolver rather than merely trusting the directive.
	#[must_use]
	pub fn has_credential_source(&self) -> bool {
		self.has_env_credentials()
			|| self.has_environment_web_identity()
			|| self.has_configured_profile()
			|| self.has_container_source()
			|| self.has_instance_source()
	}

	/// Resolves a cached or newly minted credential and sends it to `sink`.
	pub async fn authorize_into<S: AwsCredentialSink>(
		&self,
		sink: &S,
	) -> Result<(), AwsIntoError<S::Error>> {
		self.deliver(false, sink).await
	}

	/// Invalidates the cached chain, resolves it again, and sends it to `sink`.
	pub async fn refresh_into<S: AwsCredentialSink>(
		&self,
		sink: &S,
	) -> Result<(), AwsIntoError<S::Error>> {
		self.deliver(true, sink).await
	}

	async fn deliver<S: AwsCredentialSink>(
		&self,
		force: bool,
		sink: &S,
	) -> Result<(), AwsIntoError<S::Error>> {
		let profile = Str::new(self.settings.profile());
		let region = Str::from(self.resolve_region()?);
		let now = now_ms();
		let mut cache = self.cache.lock().await;
		if !force
			&& let Some(hit) = cache.as_ref()
			&& hit.profile == profile
			&& hit.region == region
			&& hit.credentials.is_fresh(now)
		{
			return hit.credentials.deliver(sink).map_err(AwsIntoError::Sink);
		}
		let credentials = self.resolve_fresh(profile.as_str(), region.as_str()).await?;
		credentials.deliver(sink).map_err(AwsIntoError::Sink)?;
		*cache = Some(CacheEntry { profile, region, credentials });
		Ok(())
	}

	async fn resolve_fresh(&self, profile: &str, region: &str) -> Result<Credentials, AwsError> {
		if let Some(credentials) = self.env_credentials() {
			return Ok(credentials);
		}
		if self.has_environment_web_identity() {
			return self
				.assume_role_with_web_identity(
					self.settings.env("AWS_ROLE_ARN").ok_or(AwsError::Missing)?,
					self.settings
						.env("AWS_WEB_IDENTITY_TOKEN_FILE")
						.ok_or(AwsError::Missing)?,
					self.settings.env("AWS_ROLE_SESSION_NAME"),
					region,
				)
				.await;
		}
		if let Some(credentials) = self.resolve_profile(profile, region).await? {
			return Ok(credentials);
		}
		if let Some(credentials) = self.container_credentials().await? {
			return Ok(credentials);
		}
		if !self.metadata_disabled()
			&& let Some(credentials) = self.imds_credentials().await
		{
			return Ok(credentials);
		}
		Err(AwsError::Missing)
	}

	fn resolve_region(&self) -> Result<String, AwsError> {
		if let Some(region) = self
			.settings
			.env("AWS_REGION")
			.or_else(|| self.settings.env("AWS_DEFAULT_REGION"))
		{
			return Ok(region.to_owned());
		}
		if self.settings.loads_config()
			&& let Some(config) = read_ini_file(&self.settings.config_path).ok().flatten()
			&& let Some(region) = config
				.get(self.settings.profile())
				.and_then(|profile| ini_value(profile, "region"))
		{
			return Ok(region.to_string());
		}
		Ok("us-east-1".to_owned())
	}

	fn has_env_credentials(&self) -> bool {
		self.settings.env("AWS_ACCESS_KEY_ID").is_some()
			&& self.settings.env("AWS_SECRET_ACCESS_KEY").is_some()
	}

	fn env_credentials(&self) -> Option<Credentials> {
		let access = self.settings.env("AWS_ACCESS_KEY_ID")?;
		let secret = self.settings.env("AWS_SECRET_ACCESS_KEY")?;
		Some(Credentials::from_values(
			access.as_bytes(),
			secret.as_bytes(),
			self.settings.env("AWS_SESSION_TOKEN").map(str::as_bytes),
			None,
		))
	}

	fn has_environment_web_identity(&self) -> bool {
		self.settings.env("AWS_WEB_IDENTITY_TOKEN_FILE").is_some()
			&& self.settings.env("AWS_ROLE_ARN").is_some()
	}

	fn has_container_source(&self) -> bool {
		self.settings.env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").is_some()
			|| self.settings.env("AWS_CONTAINER_CREDENTIALS_FULL_URI").is_some()
	}

	fn metadata_disabled(&self) -> bool {
		self.settings
			.env("AWS_EC2_METADATA_DISABLED")
			.is_some_and(|value| value.eq_ignore_ascii_case("true"))
	}

	fn has_instance_source(&self) -> bool {
		!self.metadata_disabled()
			&& (self.settings.env("AWS_EC2_METADATA_SERVICE_ENDPOINT").is_some()
				|| is_ec2_host(&self.settings.dmi_root))
	}

	fn has_configured_profile(&self) -> bool {
		let credentials = read_ini_file(&self.settings.credentials_path).ok().flatten();
		let config = self
			.settings
			.loads_config()
			.then(|| read_ini_file(&self.settings.config_path).ok().flatten())
			.flatten();
		profile_has_source(
			self.settings.profile(),
			credentials.as_ref(),
			config.as_ref(),
			&self.settings,
			&mut HashSet::new(),
		)
	}

	async fn resolve_profile(
		&self,
		profile: &str,
		region: &str,
	) -> Result<Option<Credentials>, AwsError> {
		let credentials_ini = read_ini_file(&self.settings.credentials_path)?;
		let config_ini = if self.settings.loads_config() {
			read_ini_file(&self.settings.config_path)?
		} else {
			None
		};
		let mut current = Str::new(profile);
		let mut seen = HashSet::new();
		let mut roles = Vec::new();
		let base = loop {
			if !seen.insert(current.clone()) {
				return Err(AwsError::ProfileCycle(current));
			}
			let Some(merged) = merged_profile(
				current.as_str(),
				credentials_ini.as_ref(),
				config_ini.as_ref(),
			) else {
				if roles.is_empty() {
					return Ok(None);
				}
				return Err(AwsError::MissingSourceProfile(current));
			};

			if let Some(role_arn) = ini_value(&merged, "role_arn") {
				if let Some(token_file) = ini_value(&merged, "web_identity_token_file") {
					break self
						.assume_role_with_web_identity(
							role_arn.as_str(),
							token_file.as_str(),
							ini_value(&merged, "role_session_name").map(Str::as_str),
							region,
						)
						.await?;
				}
				if ini_value(&merged, "mfa_serial").is_some() {
					return Err(AwsError::MfaRequired(current));
				}
				let role = RoleSpec {
					role_arn:         role_arn.clone(),
					session_name:     ini_value(&merged, "role_session_name").cloned(),
					duration_seconds: ini_value(&merged, "duration_seconds").cloned(),
					external_id:      ini_value(&merged, "external_id").cloned(),
				};
				if let Some(source_profile) = ini_value(&merged, "source_profile") {
					roles.push(role);
					current = source_profile.clone();
					continue;
				}
				if let Some(source) = ini_value(&merged, "credential_source") {
					roles.push(role);
					break self.resolve_credential_source(source.as_str()).await?.ok_or_else(|| {
						AwsError::MissingRoleSource(current.clone())
					})?;
				}
				return Err(AwsError::MissingRoleSource(current));
			}

			if let (Some(access), Some(secret)) = (
				ini_value(&merged, "aws_access_key_id"),
				ini_value(&merged, "aws_secret_access_key"),
			) {
				let token = ini_value(&merged, "aws_session_token");
				break Credentials::from_values(
					access.as_bytes(),
					secret.as_bytes(),
					token.map(Str::as_bytes),
					token.map(|_| now_ms().saturating_add(FILE_SESSION_CREDS_TTL_MS)),
				);
			}
			if roles.is_empty() {
				return Ok(None);
			}
			return Err(AwsError::MissingSourceProfile(current));
		};

		let mut credentials = base;
		for role in roles.into_iter().rev() {
			credentials = self.assume_role(credentials, &role, region).await?;
		}
		Ok(Some(credentials))
	}

	async fn resolve_credential_source(
		&self,
		source: &str,
	) -> Result<Option<Credentials>, AwsError> {
		match source {
			"Environment" => Ok(self.env_credentials()),
			"Ec2InstanceMetadata" if self.metadata_disabled() => Ok(None),
			"Ec2InstanceMetadata" => Ok(self.imds_credentials().await),
			"EcsContainer" => self.container_credentials().await,
			_ => Err(AwsError::UnsupportedCredentialSource(Str::new(source))),
		}
	}

	async fn assume_role_with_web_identity(
		&self,
		role_arn: &str,
		token_file: &str,
		session_name: Option<&str>,
		region: &str,
	) -> Result<Credentials, AwsError> {
		let token = std::fs::read_to_string(token_file).map_err(|_| AwsError::TokenFile)?;
		let token = token.trim();
		if token.is_empty() {
			return Err(AwsError::TokenFile);
		}
		let session = session_name
			.map(str::to_owned)
			.unwrap_or_else(|| format!("omp-{}", std::process::id()));
		let endpoint = sts_endpoint(region);
		let body = query_body(&[
			("Action", "AssumeRoleWithWebIdentity"),
			("Version", "2011-06-15"),
			("RoleArn", role_arn),
			("RoleSessionName", session.as_str()),
			("WebIdentityToken", token),
		]);
		let request = Request::builder()
			.method("POST")
			.uri(endpoint.as_str())
			.header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
			.body(Full::new(Bytes::from(body)))
			.map_err(|_| AwsError::Request)?;
		let response = self.egress.execute(request).await.map_err(|_| AwsError::Egress)?;
		parse_sts_response(response)
	}

	async fn assume_role(
		&self,
		base: Credentials,
		role: &RoleSpec,
		region: &str,
	) -> Result<Credentials, AwsError> {
		let session = role
			.session_name
			.as_ref()
			.map_or_else(|| format!("omp-{}", std::process::id()), ToString::to_string);
		let mut fields = vec![
			("Action", "AssumeRole"),
			("Version", "2011-06-15"),
			("RoleArn", role.role_arn.as_str()),
			("RoleSessionName", session.as_str()),
		];
		if let Some(duration) = role.duration_seconds.as_ref() {
			fields.push(("DurationSeconds", duration.as_str()));
		}
		if let Some(external_id) = role.external_id.as_ref() {
			fields.push(("ExternalId", external_id.as_str()));
		}
		let body = query_body(&fields);
		let endpoint = sts_endpoint(region);
		let mut request = Request::builder()
			.method("POST")
			.uri(endpoint.as_str())
			.header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
			.body(Full::new(Bytes::from(body)))
			.map_err(|_| AwsError::Request)?;
		let auth = AppliedAuth::aws(
			base.secret_access_key,
			base.access_key_id,
			base.session_token,
		);
		auth.aws_sigv4(
			&AwsSigV4Context {
				service:   "sts".into(),
				region:    region.into(),
				signed_at: SystemTime::now(),
			},
			&mut request,
		)?;
		let response = self.egress.execute(request).await.map_err(|_| AwsError::Egress)?;
		parse_sts_response(response)
	}

	async fn container_credentials(&self) -> Result<Option<Credentials>, AwsError> {
		let relative = self.settings.env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
		let full = self.settings.env("AWS_CONTAINER_CREDENTIALS_FULL_URI");
		if relative.is_none() && full.is_none() {
			return Ok(None);
		}
		let endpoint = if let Some(relative) = relative {
			if !relative.starts_with('/') || relative.starts_with("//") {
				return Err(AwsError::Request);
			}
			Url::parse(ECS_TASK_CREDENTIALS_BASE_URL)
				.and_then(|base| base.join(relative.trim_start_matches('/')))
				.map_err(|_| AwsError::Request)?
		} else {
			let endpoint = Url::parse(full.ok_or(AwsError::Request)?)
				.map_err(|_| AwsError::Request)?;
			if endpoint.scheme() != "https" && !is_local_metadata_host(&endpoint) {
				return Err(AwsError::Request);
			}
			endpoint
		};
		let authorization = if let Some(token) = self.settings.env("AWS_CONTAINER_AUTHORIZATION_TOKEN") {
			Some(token.to_owned())
		} else if let Some(path) = self.settings.env("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE") {
			Some(
				std::fs::read_to_string(path)
					.map_err(|_| AwsError::TokenFile)?
					.trim()
					.to_owned(),
			)
		} else {
			None
		};
		let mut builder = Request::builder().uri(endpoint.as_str());
		if let Some(token) = authorization.as_ref().filter(|token| !token.is_empty()) {
			let mut token =
				http::HeaderValue::from_str(token).map_err(|_| AwsError::Request)?;
			token.set_sensitive(true);
			builder = builder.header(header::AUTHORIZATION, token);
		}
		let request = builder
			.body(Full::new(Bytes::new()))
			.map_err(|_| AwsError::Request)?;
		let response = self.egress.execute(request).await.map_err(|_| AwsError::Egress)?;
		if !response.status().is_success() {
			return Err(AwsError::Status(response.status()));
		}
		let body: MetadataCredentials =
			serde_json::from_slice(response.body()).map_err(|_| AwsError::Response)?;
		Ok(Some(body.into_credentials()?))
	}

	async fn imds_credentials(&self) -> Option<Credentials> {
		let endpoint = self.imds_base_url().ok()?;
		let token_request = Request::builder()
			.method("PUT")
			.uri(endpoint.join("latest/api/token").ok()?.as_str())
			.header("x-aws-ec2-metadata-token-ttl-seconds", "21600")
			.body(Full::new(Bytes::new()))
			.ok()?;
		let token_response = tokio::time::timeout(IMDS_TIMEOUT, self.egress.execute(token_request))
			.await
			.ok()?
			.ok()?;
		if !token_response.status().is_success() {
			return None;
		}
		let mut token = http::HeaderValue::from_bytes(token_response.body()).ok()?;
		token.set_sensitive(true);
		let role_request = Request::builder()
			.uri(
				endpoint
					.join("latest/meta-data/iam/security-credentials/")
					.ok()?
					.as_str(),
			)
			.header("x-aws-ec2-metadata-token", token.clone())
			.body(Full::new(Bytes::new()))
			.ok()?;
		let role_response = tokio::time::timeout(IMDS_TIMEOUT, self.egress.execute(role_request))
			.await
			.ok()?
			.ok()?;
		if !role_response.status().is_success() {
			return None;
		}
		let role = std::str::from_utf8(role_response.body()).ok()?.trim();
		if role.is_empty() {
			return None;
		}
		let encoded_role: String = url::form_urlencoded::byte_serialize(role.as_bytes()).collect();
		let credentials_url = endpoint
			.join(&format!("latest/meta-data/iam/security-credentials/{encoded_role}"))
			.ok()?;
		let credentials_request = Request::builder()
			.uri(credentials_url.as_str())
			.header("x-aws-ec2-metadata-token", token)
			.body(Full::new(Bytes::new()))
			.ok()?;
		let response = tokio::time::timeout(IMDS_TIMEOUT, self.egress.execute(credentials_request))
			.await
			.ok()?
			.ok()?;
		if !response.status().is_success() {
			return None;
		}
		serde_json::from_slice::<MetadataCredentials>(response.body())
			.ok()?
			.into_credentials()
			.ok()
	}

	fn imds_base_url(&self) -> Result<Url, AwsError> {
		let fallback = if self
			.settings
			.env("AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE")
			.is_some_and(|mode| mode.eq_ignore_ascii_case("ipv6"))
		{
			IMDS_IPV6_BASE_URL
		} else {
			IMDS_IPV4_BASE_URL
		};
		let mut endpoint = Url::parse(
			self.settings
				.env("AWS_EC2_METADATA_SERVICE_ENDPOINT")
				.unwrap_or(fallback),
		)
		.map_err(|_| AwsError::Request)?;
		if !endpoint.path().ends_with('/') {
			let mut path = endpoint.path().to_owned();
			path.push('/');
			endpoint.set_path(&path);
		}
		Ok(endpoint)
	}
}

#[derive(Clone)]
struct RoleSpec {
	role_arn:         Str,
	session_name:     Option<Str>,
	duration_seconds: Option<Str>,
	external_id:      Option<Str>,
}

type Ini = BTreeMap<Str, BTreeMap<Str, Str>>;

fn parse_ini(text: &str) -> Ini {
	let mut result = BTreeMap::new();
	let mut current: Option<Str> = None;
	for raw in text.lines() {
		let line = raw.trim();
		if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
			continue;
		}
		if let Some(section) = line.strip_prefix('[').and_then(|line| line.strip_suffix(']')) {
			let section = section.trim().strip_prefix("profile ").unwrap_or(section.trim()).trim();
			let section = Str::new(section);
			result.entry(section.clone()).or_insert_with(BTreeMap::new);
			current = Some(section);
			continue;
		}
		let Some(section) = current.as_ref() else {
			continue;
		};
		let Some((key, value)) = line.split_once('=') else {
			continue;
		};
		result
			.get_mut(section)
			.expect("current INI section exists")
			.insert(Str::new(key.trim()), Str::new(value.trim()));
	}
	result
}

fn read_ini_file(path: &Path) -> Result<Option<Ini>, AwsError> {
	match std::fs::read_to_string(path) {
		Ok(text) => Ok(Some(parse_ini(&text))),
		Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
		Err(source) => Err(AwsError::Read { path: path.to_owned(), source }),
	}
}

fn merged_profile(profile: &str, credentials: Option<&Ini>, config: Option<&Ini>) -> Option<BTreeMap<Str, Str>> {
	let mut merged = config.and_then(|ini| ini.get(profile)).cloned().unwrap_or_default();
	if let Some(values) = credentials.and_then(|ini| ini.get(profile)) {
		for (key, value) in values {
			merged.insert(key.clone(), value.clone());
		}
	}
	(!merged.is_empty()).then_some(merged)
}

fn ini_value<'a>(profile: &'a BTreeMap<Str, Str>, key: &str) -> Option<&'a Str> {
	profile.get(key).filter(|value| !value.is_empty())
}

fn profile_has_source(
	profile: &str,
	credentials: Option<&Ini>,
	config: Option<&Ini>,
	settings: &Settings,
	seen: &mut HashSet<Str>,
) -> bool {
	if !seen.insert(Str::new(profile)) {
		return false;
	}
	let Some(merged) = merged_profile(profile, credentials, config) else {
		return false;
	};
	if ini_value(&merged, "role_arn").is_some() {
		if ini_value(&merged, "web_identity_token_file").is_some() {
			return true;
		}
		if ini_value(&merged, "mfa_serial").is_some() {
			return false;
		}
		if let Some(source) = ini_value(&merged, "credential_source") {
			return match source.as_str() {
				"Environment" => {
					settings.env("AWS_ACCESS_KEY_ID").is_some()
						&& settings.env("AWS_SECRET_ACCESS_KEY").is_some()
				},
				"EcsContainer" => {
					settings.env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").is_some()
						|| settings.env("AWS_CONTAINER_CREDENTIALS_FULL_URI").is_some()
				},
				"Ec2InstanceMetadata" => !settings
					.env("AWS_EC2_METADATA_DISABLED")
					.is_some_and(|value| value.eq_ignore_ascii_case("true")),
				_ => false,
			};
		}
		return ini_value(&merged, "source_profile").is_some_and(|source| {
			profile_has_source(source.as_str(), credentials, config, settings, seen)
		});
	}
	ini_value(&merged, "aws_access_key_id").is_some()
		&& ini_value(&merged, "aws_secret_access_key").is_some()
}

fn query_body(fields: &[(&str, &str)]) -> String {
	let mut serializer = url::form_urlencoded::Serializer::new(String::new());
	for (key, value) in fields {
		serializer.append_pair(key, value);
	}
	serializer.finish()
}

fn sts_endpoint(region: &str) -> String {
	let suffix = if region.starts_with("cn-") { "amazonaws.com.cn" } else { "amazonaws.com" };
	format!("https://sts.{region}.{suffix}/")
}

fn parse_sts_response(response: Response<Bytes>) -> Result<Credentials, AwsError> {
	if !response.status().is_success() {
		return Err(AwsError::Status(response.status()));
	}
	let xml = std::str::from_utf8(response.body()).map_err(|_| AwsError::Response)?;
	let access = xml_tag(xml, "AccessKeyId").ok_or(AwsError::Response)?;
	let secret = xml_tag(xml, "SecretAccessKey").ok_or(AwsError::Response)?;
	let token = xml_tag(xml, "SessionToken").ok_or(AwsError::Response)?;
	let expiration = xml_tag(xml, "Expiration").ok_or(AwsError::Response)?;
	let expires_at_ms = expiration
		.parse::<Timestamp>()
		.ok()
		.and_then(|timestamp| timestamp.as_millisecond().try_into().ok())
		.ok_or(AwsError::Response)?;
	Ok(Credentials::from_values(
		access.as_bytes(),
		secret.as_bytes(),
		Some(token.as_bytes()),
		Some(expires_at_ms),
	))
}

fn xml_tag(xml: &str, tag: &str) -> Option<String> {
	let mut opening = String::with_capacity(tag.len() + 2);
	opening.push('<');
	opening.push_str(tag);
	opening.push('>');
	let mut closing = String::with_capacity(tag.len() + 3);
	closing.push_str("</");
	closing.push_str(tag);
	closing.push('>');
	let start = xml.find(&opening)? + opening.len();
	let end = xml[start..].find(&closing)? + start;
	Some(
		xml[start..end]
			.replace("&amp;", "&")
			.replace("&lt;", "<")
			.replace("&gt;", ">")
			.replace("&quot;", "\"")
			.replace("&apos;", "'"),
	)
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct MetadataCredentials {
	access_key_id:     Option<String>,
	secret_access_key: Option<String>,
	token:             Option<String>,
	expiration:        Option<String>,
}

impl MetadataCredentials {
	fn into_credentials(self) -> Result<Credentials, AwsError> {
		let access = self.access_key_id.ok_or(AwsError::Response)?;
		let secret = self.secret_access_key.ok_or(AwsError::Response)?;
		let token = self.token.ok_or(AwsError::Response)?;
		let expiration = self.expiration.ok_or(AwsError::Response)?;
		let expires_at_ms = expiration
			.parse::<Timestamp>()
			.ok()
			.and_then(|timestamp| timestamp.as_millisecond().try_into().ok())
			.ok_or(AwsError::Response)?;
		Ok(Credentials::from_values(
			access.as_bytes(),
			secret.as_bytes(),
			Some(token.as_bytes()),
			Some(expires_at_ms),
		))
	}
}

fn is_local_metadata_host(endpoint: &Url) -> bool {
	let Some(host) = endpoint.host_str() else {
		return false;
	};
	host.eq_ignore_ascii_case("localhost")
		|| host == "169.254.170.2"
		|| host == "169.254.169.254"
		|| host.eq_ignore_ascii_case("fd00:ec2::23")
		|| host.parse::<std::net::IpAddr>().is_ok_and(|address| address.is_loopback())
}

fn is_ec2_host(root: &Path) -> bool {
	[
		("sys/hypervisor/uuid", DmiMarker::Ec2Prefix),
		("sys/devices/virtual/dmi/id/product_uuid", DmiMarker::Ec2Prefix),
		("sys/devices/virtual/dmi/id/board_asset_tag", DmiMarker::BoardAsset),
		("sys/devices/virtual/dmi/id/sys_vendor", DmiMarker::AmazonVendor),
		("sys/devices/virtual/dmi/id/bios_vendor", DmiMarker::AmazonVendor),
	]
	.into_iter()
	.any(|(path, marker)| {
		std::fs::read_to_string(root.join(path))
			.ok()
			.is_some_and(|value| marker.matches(value.trim()))
	})
}

#[derive(Clone, Copy)]
enum DmiMarker {
	Ec2Prefix,
	BoardAsset,
	AmazonVendor,
}

impl DmiMarker {
	fn matches(self, value: &str) -> bool {
		let value = value.to_ascii_lowercase();
		match self {
			Self::Ec2Prefix => value.starts_with("ec2"),
			Self::BoardAsset => value.starts_with("ec2") || value.starts_with("i-"),
			Self::AmazonVendor => value.contains("amazon ec2"),
		}
	}
}

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis()
		.try_into()
		.unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
	use super::*;
	use parking_lot::Mutex as ParkingMutex;
	use tempfile::TempDir;

	#[derive(Clone, Default)]
	struct MockEgress {
		requests: Arc<ParkingMutex<Vec<Request<Body>>>>,
	}

	#[derive(Debug, thiserror::Error)]
	#[error("mock egress failed")]
	struct MockError;

	impl AwsEgress for MockEgress {
		type Error = MockError;

		fn execute(
			&self,
			request: Request<Body>,
		) -> impl Future<Output = Result<Response<Bytes>, Self::Error>> + Send + 'static {
			let requests = Arc::clone(&self.requests);
			async move {
				let is_container = request.uri().host() == Some("169.254.170.2");
				let body = request.body().clone().into_inner().unwrap_or_default();
				let params = url::form_urlencoded::parse(body.as_ref())
					.into_owned()
					.collect::<BTreeMap<_, _>>();
				let access = if params.get("Action").is_some_and(|action| action == "AssumeRole") {
					"AKIAFINAL"
				} else {
					"AKIABASE"
				};
				requests.lock().push(request);
				if is_container {
					return Ok(Response::new(Bytes::from_static(
						br#"{"AccessKeyId":"AKIACONTAINER","SecretAccessKey":"container-secret","Token":"container-token","Expiration":"2099-01-01T00:00:00Z"}"#,
					)));
				}
				Ok(Response::new(Bytes::from(format!(
					"<Credentials><AccessKeyId>{access}</AccessKeyId><SecretAccessKey>{access}-secret</SecretAccessKey><SessionToken>{access}-token</SessionToken><Expiration>2099-01-01T00:00:00Z</Expiration></Credentials>"
				))))
			}
		}
	}

	#[derive(Default)]
	struct CaptureSink(ParkingMutex<Option<(Vec<u8>, Vec<u8>, Option<Vec<u8>>, u64)>>);

	impl AwsCredentialSink for CaptureSink {
		type Error = std::convert::Infallible;

		fn accept(
			&self,
			access_key_id: &[u8],
			secret_access_key: &[u8],
			session_token: Option<&[u8]>,
			expires_at_ms: u64,
		) -> Result<(), Self::Error> {
			*self.0.lock() = Some((
				access_key_id.to_vec(),
				secret_access_key.to_vec(),
				session_token.map(<[u8]>::to_vec),
				expires_at_ms,
			));
			Ok(())
		}
	}

	fn fixture_settings(temp: &TempDir, config: &str, env: &[(&str, &str)]) -> Settings {
		let credentials_path = temp.path().join("credentials");
		let config_path = temp.path().join("config");
		std::fs::write(&credentials_path, "").expect("credentials fixture");
		std::fs::write(&config_path, config).expect("config fixture");
		let mut values = env
			.iter()
			.map(|(key, value)| (Str::new(key), Str::new(value)))
			.collect::<BTreeMap<_, _>>();
		values.insert("AWS_PROFILE".into(), "default".into());
		values.entry("AWS_EC2_METADATA_DISABLED".into()).or_insert_with(|| "true".into());
		Settings {
			env: Arc::new(values),
			credentials_path,
			config_path,
			dmi_root: temp.path().to_owned(),
		}
	}

	#[tokio::test]
	async fn resolves_web_identity_source_profile_chain_and_signs_outer_role() {
		let temp = TempDir::new().expect("tempdir");
		let token_path = temp.path().join("token");
		std::fs::write(&token_path, "irsa-jwt\n").expect("token fixture");
		let config = format!(
			"[profile irsa]\nrole_arn = arn:aws:iam::111122223333:role/workspace\nweb_identity_token_file = {}\n\n[default]\nrole_arn = arn:aws:iam::111122223333:role/user\nrole_session_name = someone@example.com\nsource_profile = irsa\nexternal_id = ext-1\nduration_seconds = 1800\n",
			token_path.display()
		);
		let egress = MockEgress::default();
		let engine = AwsEngine {
			egress: egress.clone(),
			settings: fixture_settings(&temp, &config, &[("AWS_REGION", "us-east-1")]),
			cache: Arc::new(Mutex::new(None)),
		};
		let sink = CaptureSink::default();

		engine.authorize_into(&sink).await.expect("role chain");

		let requests = egress.requests.lock();
		assert_eq!(requests.len(), 2);
		let first = requests[0].body().clone().into_inner().unwrap_or_default();
		let first = url::form_urlencoded::parse(first.as_ref())
			.into_owned()
			.collect::<BTreeMap<_, _>>();
		assert_eq!(first.get("Action").map(String::as_str), Some("AssumeRoleWithWebIdentity"));
		assert_eq!(first.get("WebIdentityToken").map(String::as_str), Some("irsa-jwt"));
		let second = requests[1].body().clone().into_inner().unwrap_or_default();
		let second = url::form_urlencoded::parse(second.as_ref())
			.into_owned()
			.collect::<BTreeMap<_, _>>();
		assert_eq!(second.get("RoleSessionName").map(String::as_str), Some("someone@example.com"));
		assert_eq!(second.get("ExternalId").map(String::as_str), Some("ext-1"));
		assert_eq!(second.get("DurationSeconds").map(String::as_str), Some("1800"));
		assert!(
			requests[1]
				.headers()
				.get(header::AUTHORIZATION)
				.and_then(|value| value.to_str().ok())
				.is_some_and(|value| value.starts_with("AWS4-HMAC-SHA256 Credential=AKIABASE/"))
		);
		drop(requests);
		let captured = sink.0.lock();
		assert_eq!(captured.as_ref().map(|value| value.0.as_slice()), Some(b"AKIAFINAL".as_slice()));
		assert_ne!(captured.as_ref().map(|value| value.3), Some(0));
		drop(captured);
		engine.authorize_into(&sink).await.expect("cached role chain");
		assert_eq!(egress.requests.lock().len(), 2);
		engine.refresh_into(&sink).await.expect("forced role refresh");
		assert_eq!(egress.requests.lock().len(), 4);
	}

	#[tokio::test]
	async fn source_profile_cycle_is_rejected_without_sts() {
		let temp = TempDir::new().expect("tempdir");
		let config = "[default]\nrole_arn = arn:aws:iam::1:role/a\nsource_profile = b\n\n[profile b]\nrole_arn = arn:aws:iam::1:role/b\nsource_profile = default\n";
		let egress = MockEgress::default();
		let engine = AwsEngine {
			egress: egress.clone(),
			settings: fixture_settings(&temp, config, &[("AWS_REGION", "us-east-1")]),
			cache: Arc::new(Mutex::new(None)),
		};
		let error = engine
			.authorize_into(&CaptureSink::default())
			.await
			.expect_err("cycle must fail");
		assert!(matches!(error, AwsIntoError::Aws(AwsError::ProfileCycle(_))));
		assert!(egress.requests.lock().is_empty());
	}

	#[tokio::test]
	async fn role_without_base_source_is_not_advertised_or_resolved() {
		let temp = TempDir::new().expect("tempdir");
		let settings = fixture_settings(
			&temp,
			"[default]\nrole_arn = arn:aws:iam::111122223333:role/user\n",
			&[("AWS_REGION", "us-east-1")],
		);
		let engine = AwsEngine {
			egress: MockEgress::default(),
			settings,
			cache: Arc::new(Mutex::new(None)),
		};

		assert!(!engine.has_configured_profile());
		let error = engine
			.authorize_into(&CaptureSink::default())
			.await
			.expect_err("orphan role must fail");
		assert!(matches!(error, AwsIntoError::Aws(AwsError::MissingRoleSource(_))));
	}
	#[tokio::test]
	async fn ecs_credential_source_is_exchanged_for_profile_role() {
		let temp = TempDir::new().expect("tempdir");
		let settings = fixture_settings(
			&temp,
			"[default]\nrole_arn = arn:aws:iam::111122223333:role/user\ncredential_source = EcsContainer\n",
			&[
				("AWS_REGION", "us-east-1"),
				("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/test"),
			],
		);
		let egress = MockEgress::default();
		let engine = AwsEngine {
			egress: egress.clone(),
			settings,
			cache: Arc::new(Mutex::new(None)),
		};
		let sink = CaptureSink::default();
		engine.authorize_into(&sink).await.expect("ECS role chain");
		let requests = egress.requests.lock();
		assert_eq!(requests.len(), 2);
		assert_eq!(requests[0].uri().host(), Some("169.254.170.2"));
		assert!(
			requests[1]
				.headers()
				.get(header::AUTHORIZATION)
				.and_then(|value| value.to_str().ok())
				.is_some_and(|value| value.starts_with("AWS4-HMAC-SHA256 Credential=AKIACONTAINER/"))
		);
	}

	#[test]
	fn credential_source_availability_is_gated_by_named_source() {
		let temp = TempDir::new().expect("tempdir");
		for (source, env, expected) in [
			("Environment", vec![], false),
			("Environment", vec![("AWS_ACCESS_KEY_ID", "AKIA"), ("AWS_SECRET_ACCESS_KEY", "secret")], true),
			("EcsContainer", vec![], false),
			("EcsContainer", vec![("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/test")], true),
			("Ec2InstanceMetadata", vec![], false),
			("Ec2InstanceMetadata", vec![("AWS_EC2_METADATA_DISABLED", "false")], true),
			("Unknown", vec![], false),
		] {
			let config = format!(
				"[default]\nrole_arn = arn:aws:iam::111122223333:role/user\ncredential_source = {source}\n"
			);
			let settings = fixture_settings(&temp, &config, &env);
			let engine = AwsEngine {
				egress: MockEgress::default(),
				settings,
				cache: Arc::new(Mutex::new(None)),
			};
			assert_eq!(engine.has_configured_profile(), expected, "source {source}");
		}
	}

	#[test]
	fn nitro_dmi_markers_are_recognized() {
		let temp = TempDir::new().expect("tempdir");
		let dmi = temp.path().join("sys/devices/virtual/dmi/id");
		std::fs::create_dir_all(&dmi).expect("DMI fixture directory");
		std::fs::write(dmi.join("board_asset_tag"), "i-0123456789abcdef0\n")
			.expect("board asset fixture");
		assert!(is_ec2_host(temp.path()));
		std::fs::remove_file(dmi.join("board_asset_tag")).expect("remove board fixture");
		std::fs::write(dmi.join("sys_vendor"), "Amazon EC2\n").expect("vendor fixture");
		assert!(is_ec2_host(temp.path()));
	}
}
