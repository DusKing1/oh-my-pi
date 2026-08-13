//! AWS Signature Version 4 over finalized request bytes.

use std::{
	collections::BTreeMap,
	time::{SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use hmac::{Hmac, Mac};
use http::{
	HeaderValue, Request,
	header::{AUTHORIZATION, HOST, HeaderName},
};
use omp_core::hex;
use secrecy::{ExposeSecret, SecretString};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

use super::spec::SigV4Spec;

type HmacSha256 = Hmac<Sha256>;

/// Sealed AWS access material accepted only by a credential lease.
pub(crate) struct AwsCredential {
	pub(crate) access_key_id:     SecretString,
	pub(crate) secret_access_key: SecretString,
	pub(crate) session_token:     Option<SecretString>,
}

impl AwsCredential {
	pub(crate) fn new(
		access_key_id: SecretString,
		secret_access_key: SecretString,
		session_token: Option<SecretString>,
	) -> Self {
		Self { access_key_id, secret_access_key, session_token }
	}
}

impl Clone for AwsCredential {
	fn clone(&self) -> Self {
		Self {
			access_key_id:     self.access_key_id.clone(),
			secret_access_key: self.secret_access_key.clone(),
			session_token:     self.session_token.clone(),
		}
	}
}

impl std::fmt::Debug for AwsCredential {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter.write_str("AwsCredential([REDACTED])")
	}
}

/// Failure while signing a finalized request.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SigV4Error {
	/// The injectable signing time predates the Unix epoch or cannot be
	/// represented.
	#[error("SigV4 signing time is invalid")]
	InvalidTime,
	/// Neither the URI nor request headers identify a host.
	#[error("SigV4 request has no host")]
	MissingHost,
	/// A request header cannot be represented canonically.
	#[error("SigV4 request contains an invalid header")]
	InvalidHeader,
	/// The catalog signing specification is structurally incomplete.
	#[error("SigV4 signing specification is incomplete")]
	InvalidSpec,
}

/// Signs the exact method, URI, headers, and buffered body in place.
///
/// This function is crate-private so AWS key material cannot be used outside a
/// [`super::lease::CredentialLease`].
pub(crate) fn sign_request(
	credential: &AwsCredential,
	spec: &SigV4Spec,
	signed_at: SystemTime,
	request: &mut Request<Bytes>,
) -> Result<(), SigV4Error> {
	if spec.service.is_empty() || spec.region.is_empty() {
		return Err(SigV4Error::InvalidSpec);
	}
	let (amz_date, short_date) = aws_dates(signed_at)?;
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
		.ok_or(SigV4Error::MissingHost)?;
	request.headers_mut().insert(HOST, host);
	request.headers_mut().insert(
		HeaderName::from_static("x-amz-date"),
		HeaderValue::from_str(&amz_date).map_err(|_| SigV4Error::InvalidHeader)?,
	);
	let payload_hash = Sha256::digest(request.body());
	let payload_hash = hex::encode(&payload_hash).into_string();
	if let Some(token) = &credential.session_token {
		let mut value =
			HeaderValue::from_str(token.expose_secret()).map_err(|_| SigV4Error::InvalidHeader)?;
		value.set_sensitive(true);
		request
			.headers_mut()
			.insert(HeaderName::from_static("x-amz-security-token"), value);
	}

	let (canonical_hash, signed_headers) = canonical_request(request, &payload_hash, spec)?;
	let scope = format!("{short_date}/{}/{}/aws4_request", spec.region, spec.service);
	let string_to_sign = format!(
		"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
		hex::encode(&canonical_hash).into_string()
	);

	let mut initial =
		Zeroizing::new(Vec::with_capacity(4 + credential.secret_access_key.expose_secret().len()));
	initial.extend_from_slice(b"AWS4");
	initial.extend_from_slice(credential.secret_access_key.expose_secret().as_bytes());
	let date_key = Zeroizing::new(hmac_sha256(&initial, short_date.as_bytes()));
	let region_key = Zeroizing::new(hmac_sha256(&date_key[..], spec.region.as_bytes()));
	let service_key = Zeroizing::new(hmac_sha256(&region_key[..], spec.service.as_bytes()));
	let signing_key = Zeroizing::new(hmac_sha256(&service_key[..], b"aws4_request"));
	let mut signature = Zeroizing::new(hmac_sha256(&signing_key[..], string_to_sign.as_bytes()));
	let signature_hex = hex::encode(&signature[..]).into_string();

	let mut authorization = Zeroizing::new(Vec::with_capacity(
		credential.access_key_id.expose_secret().len()
			+ scope.len()
			+ signed_headers.len()
			+ signature_hex.len()
			+ 64,
	));
	authorization.extend_from_slice(b"AWS4-HMAC-SHA256 Credential=");
	authorization.extend_from_slice(credential.access_key_id.expose_secret().as_bytes());
	authorization.push(b'/');
	authorization.extend_from_slice(scope.as_bytes());
	authorization.extend_from_slice(b", SignedHeaders=");
	authorization.extend_from_slice(signed_headers.as_bytes());
	authorization.extend_from_slice(b", Signature=");
	authorization.extend_from_slice(signature_hex.as_bytes());
	let mut value =
		HeaderValue::from_bytes(&authorization).map_err(|_| SigV4Error::InvalidHeader)?;
	value.set_sensitive(true);
	request.headers_mut().insert(AUTHORIZATION, value);
	signature.zeroize();
	Ok(())
}

fn canonical_request(
	request: &Request<Bytes>,
	payload_hash: &str,
	spec: &SigV4Spec,
) -> Result<([u8; 32], String), SigV4Error> {
	let mut headers: BTreeMap<&str, Vec<&HeaderValue>> = BTreeMap::new();
	for (name, value) in request.headers() {
		let name = name.as_str();
		if default_unsigned_header(name)
			|| spec
				.unsigned_headers
				.iter()
				.any(|excluded| excluded == name)
		{
			continue;
		}
		headers.entry(name).or_default().push(value);
	}
	let signed_headers = headers.keys().copied().collect::<Vec<_>>().join(";");
	let mut canonical = Zeroizing::new(Vec::new());
	canonical.extend_from_slice(request.method().as_str().as_bytes());
	canonical.push(b'\n');
	let canonical_uri = canonical_uri(request.uri().path(), spec.service.as_str());
	canonical.extend_from_slice(canonical_uri.as_bytes());
	canonical.push(b'\n');
	let canonical_query = canonical_query(request.uri().query().unwrap_or_default());
	canonical.extend_from_slice(canonical_query.as_bytes());
	canonical.push(b'\n');
	for (name, values) in headers {
		canonical.extend_from_slice(name.as_bytes());
		canonical.push(b':');
		for (index, value) in values.into_iter().enumerate() {
			if index != 0 {
				canonical.push(b',');
			}
			let value = value.to_str().map_err(|_| SigV4Error::InvalidHeader)?;
			append_normalized_header(&mut canonical, value);
		}
		canonical.push(b'\n');
	}
	canonical.push(b'\n');
	canonical.extend_from_slice(signed_headers.as_bytes());
	canonical.push(b'\n');
	canonical.extend_from_slice(payload_hash.as_bytes());
	Ok((Sha256::digest(&canonical[..]).into(), signed_headers))
}

fn canonical_uri(path: &str, service: &str) -> String {
	let normalized = if service == "s3" {
		path.to_owned()
	} else {
		normalize_path(path)
	};
	encode_path(&normalized, service != "s3")
}

fn normalize_path(path: &str) -> String {
	let trailing_slash = path.ends_with('/');
	let mut segments = Vec::new();
	for segment in path.split('/') {
		match segment {
			"" | "." => {},
			".." => {
				segments.pop();
			},
			value => segments.push(value),
		}
	}
	let mut output = String::from("/");
	output.push_str(&segments.join("/"));
	if trailing_slash && output.len() > 1 {
		output.push('/');
	}
	output
}

fn encode_path(path: &str, double_encode_percent: bool) -> String {
	let bytes = path.as_bytes();
	let mut output = String::with_capacity(bytes.len());
	let mut index = 0;
	while index < bytes.len() {
		let byte = bytes[index];
		if byte == b'/' {
			output.push('/');
		} else if byte == b'%'
			&& index + 2 < bytes.len()
			&& hex_value(bytes[index + 1]).is_some()
			&& hex_value(bytes[index + 2]).is_some()
		{
			if double_encode_percent {
				output.push_str("%25");
				output.push(hex_digit(hex_value(bytes[index + 1]).expect("checked")));
				output.push(hex_digit(hex_value(bytes[index + 2]).expect("checked")));
			} else {
				output.push('%');
				output.push(hex_digit(hex_value(bytes[index + 1]).expect("checked")));
				output.push(hex_digit(hex_value(bytes[index + 2]).expect("checked")));
			}
			index += 2;
		} else {
			append_uri_byte(&mut output, byte);
		}
		index += 1;
	}
	if output.is_empty() {
		"/".to_owned()
	} else {
		output
	}
}

fn canonical_query(query: &str) -> String {
	if query.is_empty() {
		return String::new();
	}
	let mut parameters = query
		.split('&')
		.map(|parameter| {
			let (name, value) = parameter.split_once('=').unwrap_or((parameter, ""));
			(encode_query_component(name), encode_query_component(value))
		})
		.collect::<Vec<_>>();
	parameters.sort_unstable();
	let mut output = String::new();
	for (index, (name, value)) in parameters.into_iter().enumerate() {
		if index != 0 {
			output.push('&');
		}
		output.push_str(&name);
		output.push('=');
		output.push_str(&value);
	}
	output
}

fn encode_query_component(value: &str) -> String {
	let bytes = value.as_bytes();
	let mut output = String::with_capacity(bytes.len());
	let mut index = 0;
	while index < bytes.len() {
		if bytes[index] == b'%'
			&& index + 2 < bytes.len()
			&& let (Some(high), Some(low)) = (hex_value(bytes[index + 1]), hex_value(bytes[index + 2]))
		{
			append_uri_byte(&mut output, high * 16 + low);
			index += 3;
		} else {
			append_uri_byte(&mut output, bytes[index]);
			index += 1;
		}
	}
	output
}

fn append_uri_byte(output: &mut String, byte: u8) {
	if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
		output.push(char::from(byte));
	} else {
		output.push('%');
		output.push(hex_digit(byte >> 4));
		output.push(hex_digit(byte & 0x0f));
	}
}

const fn hex_value(byte: u8) -> Option<u8> {
	match byte {
		b'0'..=b'9' => Some(byte - b'0'),
		b'a'..=b'f' => Some(byte - b'a' + 10),
		b'A'..=b'F' => Some(byte - b'A' + 10),
		_ => None,
	}
}

const fn hex_digit(value: u8) -> char {
	match value {
		0..=9 => (b'0' + value) as char,
		_ => (b'A' + value - 10) as char,
	}
}

const fn default_unsigned_header(name: &str) -> bool {
	matches!(
		name.as_bytes(),
		b"authorization"
			| b"connection"
			| b"expect"
			| b"transfer-encoding"
			| b"user-agent"
			| b"x-amzn-trace-id"
	)
}

fn append_normalized_header(output: &mut Vec<u8>, value: &str) {
	for (index, part) in value.split_ascii_whitespace().enumerate() {
		if index != 0 {
			output.push(b' ');
		}
		output.extend_from_slice(part.as_bytes());
	}
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
	let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts every key length");
	mac.update(message);
	mac.finalize().into_bytes().into()
}

fn aws_dates(time: SystemTime) -> Result<(String, String), SigV4Error> {
	let seconds = time
		.duration_since(UNIX_EPOCH)
		.map_err(|_| SigV4Error::InvalidTime)?
		.as_secs();
	let days = i64::try_from(seconds / 86_400).map_err(|_| SigV4Error::InvalidTime)?;
	let seconds_of_day = seconds % 86_400;
	let (year, month, day) = civil_from_days(days);
	let hour = seconds_of_day / 3_600;
	let minute = seconds_of_day % 3_600 / 60;
	let second = seconds_of_day % 60;
	let short = format!("{year:04}{month:02}{day:02}");
	Ok((format!("{short}T{hour:02}{minute:02}{second:02}Z"), short))
}

const fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
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
	year += if month <= 2 { 1 } else { 0 };
	(year, month, day)
}

#[cfg(test)]
mod tests {
	use std::time::Duration;

	use super::*;

	#[test]
	fn bedrock_golden_signs_the_final_request_and_redacts_material() {
		let secret = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
		let session = "session-token";
		let credential = AwsCredential::new(
			SecretString::from("AKIDEXAMPLE".to_owned()),
			SecretString::from(secret.to_owned()),
			Some(SecretString::from(session.to_owned())),
		);
		let spec = SigV4Spec {
			service:          "bedrock".into(),
			region:           "us-east-1".into(),
			unsigned_headers: Vec::new(),
		};
		let mut request = Request::builder()
			.method("POST")
			.uri("https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke?x=1")
			.header("content-type", "application/json")
			.body(Bytes::from_static(b"{}"))
			.expect("request");
		sign_request(
			&credential,
			&spec,
			UNIX_EPOCH + Duration::from_secs(1_704_164_645),
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
		let debug = format!("{credential:?} {request:?}");
		assert!(!debug.contains(secret));
		assert!(!debug.contains(session));
	}

	#[test]
	fn canonical_query_uses_aws_rfc3986_encoding_and_encoded_pair_order() {
		assert_eq!(
			canonical_query("z=last&a=+&a=/&empty&colon=:&lower=%2f&upper=%2F&a="),
			"a=&a=%2B&a=%2F&colon=%3A&empty=&lower=%2F&upper=%2F&z=last"
		);
	}

	#[test]
	fn canonical_path_normalizes_service_paths_without_confusing_encoded_slashes() {
		assert_eq!(
			canonical_uri("/a//b/./c/../d:+/%2f/%2F", "execute-api"),
			"/a/b/d%3A%2B/%252F/%252F"
		);
		assert_eq!(canonical_uri("/a//b/./c/../d:+/%2f/%2F", "s3"), "/a//b/./c/../d%3A%2B/%2F/%2F");
	}

	#[test]
	fn aws_published_iam_canonical_components_are_stable() {
		assert_eq!(canonical_uri("/", "iam"), "/");
		assert_eq!(
			canonical_query("Version=2010-05-08&Action=ListUsers"),
			"Action=ListUsers&Version=2010-05-08"
		);
	}
}
