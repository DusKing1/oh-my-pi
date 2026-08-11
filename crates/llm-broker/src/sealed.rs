//! Sealed provider-authentication values.
//!
//! Provider token bytes never cross a process boundary except from the daemon
//! to the provider. The previous implementation's snapshot export shipped API
//! keys and OAuth access tokens to every client. Keeping secret construction
//! and lease redemption inside the broker makes that failure mode
//! unrepresentable.

use std::{
	collections::BTreeMap,
	fmt,
	sync::atomic::{Ordering, compiler_fence},
	time::{SystemTime, UNIX_EPOCH},
};

use hmac::{Hmac, Mac};
use http::{
	Extensions, HeaderValue, Request,
	header::{AUTHORIZATION, HOST, HeaderMap, HeaderName},
	request::Builder,
};
use omp_llm_egress::{
	auth_inject::{AwsSigV4Context, SensitiveQuery},
	client::Body,
};
use sha2::{Digest, Sha256};

/// Provider token bytes owned by the broker.
///
/// Construction is crate-private, the backing allocation is wiped on drop,
/// and formatting is always redacted.
///
/// ```compile_fail
/// use omp_llm_broker::sealed::Secret;
///
/// // The field and constructor are not public API.
/// let secret = Secret(b"must not compile".to_vec().into_boxed_slice());
/// ```
///
/// ```compile_fail
/// # fn cannot_serialize(secret: &omp_llm_broker::sealed::Secret) {
/// let _ = serde_json::to_vec(secret);
/// # }
/// ```
///
/// `Secret` deliberately does not implement `serde::Serialize`, so it cannot
/// accidentally become part of an RPC or snapshot payload.
pub struct Secret(Box<[u8]>);

impl Secret {
	/// Copies bytes arriving at the broker's one-way secret-ingress boundary.
	pub(crate) fn new(bytes: &[u8]) -> Self {
		Self(bytes.into())
	}

	/// Takes ownership of a database buffer without leaving a copied token.
	pub(crate) fn from_vec(bytes: Vec<u8>) -> Self {
		Self(bytes.into_boxed_slice())
	}

	pub(crate) fn expose(&self) -> &[u8] {
		&self.0
	}

	fn zeroize(&mut self) {
		// Volatile writes prevent dead-store elimination without requiring the
		// secret to use a generally serializable container type.
		for byte in &mut self.0 {
			// SAFETY: `byte` is a valid, uniquely borrowed byte in the owned allocation.
			unsafe { std::ptr::write_volatile(byte, 0) };
		}
		compiler_fence(Ordering::SeqCst);
	}
}

impl fmt::Debug for Secret {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str("Secret([redacted])")
	}
}

impl fmt::Display for Secret {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str("Secret([redacted])")
	}
}

impl Drop for Secret {
	fn drop(&mut self) {
		self.zeroize();
	}
}

/// Failure to sign a request using sealed AWS credential material.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum AwsSigV4Error {
	/// The selected store row does not contain complete AWS key material.
	#[error("AWS credential material is incomplete")]
	MissingMaterial,
	/// The injectable signing time precedes the Unix epoch.
	#[error("AWS signing time is invalid")]
	InvalidTime,
	/// Neither the URI nor headers identify the upstream host.
	#[error("AWS request has no host")]
	MissingHost,
	/// A generated signing header was not representable by HTTP.
	#[error("AWS signing produced an invalid header")]
	InvalidHeader,
}

/// Sealed authentication ready to be applied to one provider request.
pub struct AppliedAuth {
	secret:            Secret,
	aws_access_key:    Option<Secret>,
	aws_session_token: Option<Secret>,
}

impl AppliedAuth {
	pub(crate) const fn bearer(secret: Secret) -> Self {
		Self { secret, aws_access_key: None, aws_session_token: None }
	}

	pub(crate) const fn aws(
		secret_access_key: Secret,
		access_key_id: Secret,
		session_token: Option<Secret>,
	) -> Self {
		Self {
			secret:            secret_access_key,
			aws_access_key:    Some(access_key_id),
			aws_session_token: session_token,
		}
	}

	/// Applies bearer authentication to a request builder without returning
	/// token bytes to the caller.
	pub fn apply(self, builder: Builder) -> Result<Builder, http::header::InvalidHeaderValue> {
		let value = self.bearer_value()?;
		Ok(builder.header(AUTHORIZATION, value))
	}

	/// Applies bearer authentication directly to a header map without returning
	/// token bytes to the caller.
	pub fn apply_bearer_to_headers(
		&self,
		headers: &mut HeaderMap,
	) -> Result<(), http::header::InvalidHeaderValue> {
		headers.insert(AUTHORIZATION, self.bearer_value()?);
		Ok(())
	}

	/// Fills a discovery request body whose credential is embedded in the
	/// payload, without exposing that credential to the caller.
	pub fn apply_sealed_discovery_body(self, request: &mut Request<Body>) {
		*request.body_mut() =
			http_body_util::Full::new(omp_llm_devin::model_discovery_request(self.secret.expose()));
	}

	/// Applies Devin's session token and account JWT directly to protobuf
	/// metadata without returning either value to the caller.
	pub fn apply_to_devin_metadata(
		&self,
		metadata: &mut omp_llm_devin::wire::Metadata,
		user_jwt: String,
	) {
		const PREFIX: &str = "devin-session-token$";
		let token = String::from_utf8_lossy(self.secret.expose());
		metadata.api_key = if token.starts_with(PREFIX) {
			token.into_owned()
		} else {
			let mut prefixed = String::with_capacity(PREFIX.len() + token.len());
			prefixed.push_str(PREFIX);
			prefixed.push_str(&token);
			prefixed
		};
		metadata.user_jwt = user_jwt;
		metadata.ide_name = "windsurf".to_owned();
		metadata.ide_version = "3.2.23".to_owned();
		metadata.extension_name = "windsurf".to_owned();
		metadata.extension_version = "1.48.2".to_owned();
		metadata.locale = "en".to_owned();
	}

	/// Applies the credential as the complete `Authorization` value.
	pub fn apply_to_authorization(
		&self,
		headers: &mut HeaderMap,
	) -> Result<(), http::header::InvalidHeaderValue> {
		self.apply_to_named_header(AUTHORIZATION, headers)
	}

	/// Applies the credential as the complete value of a named header.
	pub fn apply_to_named_header(
		&self,
		name: HeaderName,
		headers: &mut HeaderMap,
	) -> Result<(), http::header::InvalidHeaderValue> {
		let mut value = HeaderValue::from_bytes(self.secret.expose())?;
		value.set_sensitive(true);
		headers.insert(name, value);
		Ok(())
	}

	/// Applies Cursor's session-token cookie without leaving an unwiped
	/// intermediate token buffer.
	pub fn apply_to_cursor_cookie(
		&self,
		headers: &mut HeaderMap,
	) -> Result<(), http::header::InvalidHeaderValue> {
		let cookie_name = b"WorkosCursorSessionToken";
		let mut value = Vec::with_capacity(cookie_name.len() + 1 + self.secret.expose().len());
		value.extend_from_slice(cookie_name);
		value.push(b'=');
		value.extend_from_slice(self.secret.expose());
		let value = Secret::from_vec(value);
		let mut header = HeaderValue::from_bytes(value.expose())?;
		header.set_sensitive(true);
		headers.insert(http::header::COOKIE, header);
		Ok(())
	}

	/// Defers query-credential placement to the egress connector.
	///
	/// The opaque extension has redacted formatting and owns a zeroizing copy;
	/// the request URI remains non-secret until final wire serialization.
	pub fn apply_to_sensitive_query(&self, parameter: &str, extensions: &mut Extensions) {
		extensions.insert(SensitiveQuery::new(parameter, self.secret.expose()));
	}

	/// Signs the complete buffered request in place using AWS Signature Version
	/// 4 without returning any key material.
	pub fn aws_sigv4(
		&self,
		context: &AwsSigV4Context,
		request: &mut Request<Body>,
	) -> Result<(), AwsSigV4Error> {
		let mut signed = Request::new(request.body().clone());
		*signed.method_mut() = request.method().clone();
		*signed.uri_mut() = request.uri().clone();
		*signed.version_mut() = request.version();
		*signed.headers_mut() = request.headers().clone();
		self.aws_sigv4_in_place(context, &mut signed)?;
		*request.headers_mut() = std::mem::take(signed.headers_mut());
		Ok(())
	}

	fn aws_sigv4_in_place(
		&self,
		context: &AwsSigV4Context,
		request: &mut Request<Body>,
	) -> Result<(), AwsSigV4Error> {
		let access_key = self
			.aws_access_key
			.as_ref()
			.ok_or(AwsSigV4Error::MissingMaterial)?;
		let (amz_date, short_date) = aws_dates(context.signed_at)?;
		let host = request
			.headers()
			.get(HOST)
			.cloned()
			.or_else(|| {
				request
					.uri()
					.authority()
					.and_then(|authority| HeaderValue::from_str(authority.as_str()).ok())
			})
			.ok_or(AwsSigV4Error::MissingHost)?;
		request.headers_mut().insert(HOST, host);
		request.headers_mut().insert(
			HeaderName::from_static("x-amz-date"),
			HeaderValue::from_str(&amz_date).map_err(|_| AwsSigV4Error::InvalidHeader)?,
		);
		if let Some(session_token) = self.aws_session_token.as_ref() {
			let mut value = HeaderValue::from_bytes(session_token.expose())
				.map_err(|_| AwsSigV4Error::InvalidHeader)?;
			value.set_sensitive(true);
			request
				.headers_mut()
				.insert(HeaderName::from_static("x-amz-security-token"), value);
		}

		let body = request.body().clone().into_inner().unwrap_or_default();
		let payload_hash = hex(&Sha256::digest(body.as_ref()));
		let (canonical_hash, signed_headers) = canonical_request(request, &payload_hash)?;
		let scope = format!("{short_date}/{}/{}/aws4_request", context.region, context.service);
		let string_to_sign =
			format!("AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}", hex(&canonical_hash));

		let mut initial = Vec::with_capacity(4 + self.secret.expose().len());
		initial.extend_from_slice(b"AWS4");
		initial.extend_from_slice(self.secret.expose());
		let initial = Secret::from_vec(initial);
		let mut date_key = hmac_sha256(initial.expose(), short_date.as_bytes());
		let mut region_key = hmac_sha256(&date_key, context.region.as_bytes());
		let mut service_key = hmac_sha256(&region_key, context.service.as_bytes());
		let mut signing_key = hmac_sha256(&service_key, b"aws4_request");
		let signature = hmac_sha256(&signing_key, string_to_sign.as_bytes());
		zeroize_bytes(&mut date_key);
		zeroize_bytes(&mut region_key);
		zeroize_bytes(&mut service_key);
		zeroize_bytes(&mut signing_key);

		let mut authorization = Vec::new();
		authorization.extend_from_slice(b"AWS4-HMAC-SHA256 Credential=");
		authorization.extend_from_slice(access_key.expose());
		authorization.push(b'/');
		authorization.extend_from_slice(scope.as_bytes());
		authorization.extend_from_slice(b", SignedHeaders=");
		authorization.extend_from_slice(signed_headers.as_bytes());
		authorization.extend_from_slice(b", Signature=");
		authorization.extend_from_slice(hex(&signature).as_bytes());
		let authorization = Secret::from_vec(authorization);
		let mut value = HeaderValue::from_bytes(authorization.expose())
			.map_err(|_| AwsSigV4Error::InvalidHeader)?;
		value.set_sensitive(true);
		request.headers_mut().insert(AUTHORIZATION, value);
		Ok(())
	}

	fn bearer_value(&self) -> Result<HeaderValue, http::header::InvalidHeaderValue> {
		let mut value = Vec::with_capacity(7 + self.secret.expose().len());
		value.extend_from_slice(b"Bearer ");
		value.extend_from_slice(self.secret.expose());
		let value = Secret::from_vec(value);
		let mut header = HeaderValue::from_bytes(value.expose())?;
		header.set_sensitive(true);
		Ok(header)
	}
}

type HmacSha256 = Hmac<Sha256>;

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
	let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts every key length");
	mac.update(message);
	mac.finalize().into_bytes().into()
}

fn canonical_request(
	request: &Request<Body>,
	payload_hash: &str,
) -> Result<([u8; 32], String), AwsSigV4Error> {
	let mut headers: BTreeMap<&str, Vec<&HeaderValue>> = BTreeMap::new();
	for (name, value) in request.headers() {
		if matches!(
			name.as_str(),
			"authorization"
				| "connection"
				| "expect"
				| "transfer-encoding"
				| "user-agent"
				| "x-amzn-trace-id"
		) {
			continue;
		}
		headers.entry(name.as_str()).or_default().push(value);
	}
	let signed_headers = headers.keys().copied().collect::<Vec<_>>().join(";");
	let mut canonical = Vec::new();
	canonical.extend_from_slice(request.method().as_str().as_bytes());
	canonical.push(b'\n');
	canonical.extend_from_slice(request.uri().path().as_bytes());
	canonical.push(b'\n');
	let mut query = request
		.uri()
		.query()
		.unwrap_or_default()
		.split('&')
		.filter(|part| !part.is_empty())
		.collect::<Vec<_>>();
	query.sort_unstable();
	for (index, part) in query.into_iter().enumerate() {
		if index != 0 {
			canonical.push(b'&');
		}
		canonical.extend_from_slice(part.as_bytes());
	}
	canonical.push(b'\n');
	for (name, values) in headers {
		canonical.extend_from_slice(name.as_bytes());
		canonical.push(b':');
		for (index, value) in values.into_iter().enumerate() {
			if index != 0 {
				canonical.push(b',');
			}
			let value = value.to_str().map_err(|_| AwsSigV4Error::InvalidHeader)?;
			append_normalized_header(&mut canonical, value);
		}
		canonical.push(b'\n');
	}
	canonical.push(b'\n');
	canonical.extend_from_slice(signed_headers.as_bytes());
	canonical.push(b'\n');
	canonical.extend_from_slice(payload_hash.as_bytes());
	let canonical = Secret::from_vec(canonical);
	Ok((Sha256::digest(canonical.expose()).into(), signed_headers))
}

fn append_normalized_header(output: &mut Vec<u8>, value: &str) {
	for (index, part) in value.split_ascii_whitespace().enumerate() {
		if index != 0 {
			output.push(b' ');
		}
		output.extend_from_slice(part.as_bytes());
	}
}

fn aws_dates(time: SystemTime) -> Result<(String, String), AwsSigV4Error> {
	let seconds = time
		.duration_since(UNIX_EPOCH)
		.map_err(|_| AwsSigV4Error::InvalidTime)?
		.as_secs();
	let days = i64::try_from(seconds / 86_400).map_err(|_| AwsSigV4Error::InvalidTime)?;
	let seconds_of_day = seconds % 86_400;
	let (year, month, day) = civil_from_days(days);
	let hour = seconds_of_day / 3_600;
	let minute = seconds_of_day % 3_600 / 60;
	let second = seconds_of_day % 60;
	let short = format!("{year:04}{month:02}{day:02}");
	Ok((format!("{short}T{hour:02}{minute:02}{second:02}Z"), short))
}

fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
	let days = days_since_epoch + 719_468;
	let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
	let day_of_era = days - era * 146_097;
	let year_of_era =
		(day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
	let mut year = year_of_era + era * 400;
	let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
	let month_prime = (5 * day_of_year + 2) / 153;
	let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
	let month = month_prime + if month_prime < 10 { 3 } else { -9 };
	year += i64::from(month <= 2);
	(year, month, day)
}

fn hex(bytes: &[u8]) -> String {
	const HEX: &[u8; 16] = b"0123456789abcdef";
	let mut encoded = String::with_capacity(bytes.len() * 2);
	for byte in bytes {
		encoded.push(char::from(HEX[usize::from(byte >> 4)]));
		encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
	}
	encoded
}

fn zeroize_bytes(bytes: &mut [u8]) {
	for byte in bytes {
		// SAFETY: `byte` is a valid, uniquely borrowed byte.
		unsafe { std::ptr::write_volatile(byte, 0) };
	}
	compiler_fence(Ordering::SeqCst);
}

impl fmt::Debug for AppliedAuth {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str("AppliedAuth([redacted])")
	}
}

#[cfg(test)]
mod tests {
	use std::time::{Duration, UNIX_EPOCH};

	use bytes::Bytes;
	use http::header::{AUTHORIZATION, COOKIE};
	use omp_llm_egress::{
		auth_inject::{AwsSigV4Context, SensitiveQuery},
		client::Body,
	};

	use super::{AppliedAuth, Secret};

	#[test]
	fn secret_debug_and_display_are_redacted() {
		let material = "super-secret-token";
		let secret = Secret::new(material.as_bytes());
		let debug = format!("{secret:?}");
		let display = format!("{secret}");

		assert_eq!(debug, "Secret([redacted])");
		assert_eq!(display, "Secret([redacted])");
		assert!(!debug.contains(material));
		assert!(!display.contains(material));
	}

	#[test]
	fn secret_zeroize_clears_the_owned_buffer() {
		let mut secret = Secret::new(b"super-secret-token");
		secret.zeroize();
		assert!(secret.expose().iter().all(|byte| *byte == 0));
	}

	#[test]
	fn direct_request_placement_exposes_material_only_in_request_parts() {
		let material = "super-secret-token";
		let auth = AppliedAuth::bearer(Secret::new(material.as_bytes()));
		let mut headers = http::HeaderMap::new();
		auth.apply_bearer_to_headers(&mut headers).expect("bearer");
		assert_eq!(headers[AUTHORIZATION], "Bearer super-secret-token");
		auth
			.apply_to_authorization(&mut headers)
			.expect("authorization");
		assert_eq!(headers[AUTHORIZATION], material);
		auth
			.apply_to_named_header(http::HeaderName::from_static("x-api-key"), &mut headers)
			.expect("named header");
		assert_eq!(headers["x-api-key"], material);
		auth
			.apply_to_cursor_cookie(&mut headers)
			.expect("Cursor cookie");
		assert_eq!(headers[COOKIE], "WorkosCursorSessionToken=super-secret-token");
		let headers_debug = format!("{headers:?}");
		assert!(!headers_debug.contains(material));
		let mut extensions = http::Extensions::new();
		auth.apply_to_sensitive_query("key", &mut extensions);
		let query = extensions.get::<SensitiveQuery>().expect("sensitive query");
		assert!(!format!("{query:?}").contains(material));
		assert!(!format!("{auth:?}").contains(material));
	}

	#[test]
	fn aws_sigv4_matches_canonical_request_fixture_without_formatting_secrets() {
		let secret = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
		let session = "session-token";
		let auth = AppliedAuth::aws(
			Secret::new(secret.as_bytes()),
			Secret::new(b"AKIDEXAMPLE"),
			Some(Secret::new(session.as_bytes())),
		);
		let mut request = http::Request::builder()
			.method("POST")
			.uri("https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke?x=1")
			.header("content-type", "application/json")
			.body(Body::new(Bytes::from_static(b"{}")))
			.expect("request");
		auth
			.aws_sigv4(
				&AwsSigV4Context {
					service:   "bedrock".into(),
					region:    "us-east-1".into(),
					signed_at: UNIX_EPOCH + Duration::from_secs(1_704_164_645),
				},
				&mut request,
			)
			.expect("signature");
		assert_eq!(request.headers()["x-amz-date"], "20240102T030405Z");
		assert_eq!(request.headers()["x-amz-security-token"], session);
		assert_eq!(
			request.headers()[AUTHORIZATION],
			"AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20240102/us-east-1/bedrock/aws4_request, \
			 SignedHeaders=content-type;host;x-amz-date;x-amz-security-token, \
			 Signature=b3d1eb71036f1bc84ecc771586e0218c87cd321b6b484b244702cb01fc3914a0"
		);
		let request_debug = format!("{request:?}");
		assert!(!request_debug.contains(secret));
		assert!(!request_debug.contains(session));
		let debug = format!("{auth:?}");
		assert!(!debug.contains(secret));
		assert!(!debug.contains(session));
	}
}
