//! Data-driven proxy selection and connection establishment.

use std::{
	collections::BTreeMap,
	fmt,
	future::Future,
	io,
	net::{IpAddr, Ipv4Addr, Ipv6Addr},
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
};

use http::{Request, Uri, header::PROXY_AUTHORIZATION};
use hyper::rt::{Read as HyperRead, ReadBufCursor, Write as HyperWrite};
use hyper_util::{
	client::legacy::connect::{Connected, Connection},
	rt::TokioIo,
};
use omp_core::{Str, base64};
use thiserror::Error;
use tokio::{
	io::{AsyncReadExt, AsyncWriteExt},
	net::TcpStream,
};
use tower::Service;
use url::Url;
use zeroize::Zeroizing;

/// Hosts which must never be sent to an egress proxy.
///
/// Proxying cloud metadata endpoints breaks ADC and instance credentials and
/// may leak credential probes outside the instance. The numeric ranges below
/// additionally retain pi's local/private-network bypass behavior.
const ALWAYS_DIRECT_HOSTS: &[&str] =
	&["localhost", "169.254.169.254", "metadata.google.internal", "fd00:ec2::254"];

/// Network rules which are intrinsically direct, ported from pi's proxy layer.
const ALWAYS_DIRECT_NETWORKS: &[&str] = &[
	"0.0.0.0/8",
	"10.0.0.0/8",
	"127.0.0.0/8",
	"169.254.0.0/16",
	"172.16.0.0/12",
	"192.168.0.0/16",
	"::/128",
	"::1/128",
	"fc00::/7",
	"fe80::/10",
];

/// Credentials embedded in a proxy URL.
#[derive(Clone, Eq, PartialEq)]
#[non_exhaustive]
pub struct ProxyAuth {
	/// Proxy user name.
	pub username: Str,
	/// Proxy password.
	pub password: Str,
}

impl fmt::Debug for ProxyAuth {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("ProxyAuth")
			.field("username", &"[REDACTED]")
			.field("password", &"[REDACTED]")
			.finish()
	}
}

/// The resolved route for one outbound URL.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ProxyDecision {
	/// Connect directly to the origin.
	Direct,
	/// Use an HTTP proxy, CONNECT-tunnelling secure origins.
	Http {
		/// Parsed proxy endpoint.
		url:  Url,
		/// Optional HTTP Basic proxy credentials.
		auth: Option<ProxyAuth>,
	},
	/// Use a SOCKS5 proxy.
	Socks5 {
		/// Parsed proxy endpoint.
		url:  Url,
		/// Optional RFC 1929 user/password credentials.
		auth: Option<ProxyAuth>,
	},
}

/// An immutable environment snapshot used to resolve proxy policy.
///
/// Snapshotting avoids environment races and makes the precedence rules
/// directly testable. Provider variables are `OMP_PROXY_<PROVIDER>` (with
/// non-alphanumeric characters replaced by `_`). No legacy prefix is read.
#[derive(Clone, Default)]
pub struct ProxyResolver {
	environment: BTreeMap<Str, Str>,
}

impl ProxyResolver {
	/// Captures the process environment once.
	#[must_use]
	pub fn from_env() -> Self {
		Self::from_pairs(std::env::vars())
	}

	/// Builds a resolver from explicit key/value pairs.
	///
	/// This is useful to construct provider-scoped policy without mutating the
	/// process environment, including in concurrent tests.
	#[must_use]
	pub fn from_pairs<I, K, V>(pairs: I) -> Self
	where
		I: IntoIterator<Item = (K, V)>,
		K: AsRef<str>,
		V: AsRef<str>,
	{
		let environment = pairs
			.into_iter()
			.map(|(key, value)| (Str::new(key), Str::new(value)))
			.collect();
		Self { environment }
	}

	/// Resolves a target according to bypass, provider, protocol, and fallback
	/// rules.
	///
	/// Invalid or unsupported proxy values safely produce
	/// [`ProxyDecision::Direct`].
	#[must_use]
	pub fn resolve(&self, url: &Url, provider: Option<&str>) -> ProxyDecision {
		if self.should_bypass(url) {
			return ProxyDecision::Direct;
		}

		let configured = provider
			.and_then(|provider| self.provider_proxy(provider))
			.or_else(|| match url.scheme() {
				"https" | "wss" => self.first(&["HTTPS_PROXY", "https_proxy"]),
				_ => self.first(&["HTTP_PROXY", "http_proxy"]),
			})
			.or_else(|| self.first(&["ALL_PROXY", "all_proxy"]));
		configured
			.and_then(parse_proxy)
			.unwrap_or(ProxyDecision::Direct)
	}

	fn provider_proxy(&self, provider: &str) -> Option<&str> {
		let normalized: String = provider
			.chars()
			.map(|character| {
				if character.is_ascii_alphanumeric() {
					character.to_ascii_uppercase()
				} else {
					'_'
				}
			})
			.collect();
		let key = format!("OMP_PROXY_{normalized}");
		if let Some(value) = self.environment.get(key.as_str()) {
			return Some(value.as_str());
		}
		self.first(&["OMP_PROXY"])
	}

	fn first(&self, keys: &[&str]) -> Option<&str> {
		keys.iter().find_map(|key| {
			self
				.environment
				.get(*key)
				.map(Str::as_str)
				.filter(|value| !value.is_empty())
		})
	}

	fn should_bypass(&self, url: &Url) -> bool {
		let Some(host) = url.host_str() else {
			return true;
		};
		if is_always_direct(host) {
			return true;
		}
		let Some(rules) = self.first(&["NO_PROXY", "no_proxy"]) else {
			return false;
		};
		let port = url.port_or_known_default();
		rules
			.split(|character: char| character == ',' || character.is_ascii_whitespace())
			.filter(|rule| !rule.is_empty())
			.any(|rule| no_proxy_matches(rule, host, port))
	}
}

/// Connection establishment failure after a proxy decision.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProxyError {
	/// The target or proxy URL omitted a host or usable port.
	#[error("URL has no connectable host or port")]
	InvalidEndpoint,
	/// A Hyper URI could not be converted to a URL.
	#[error("invalid outbound URI")]
	InvalidUri,
	/// The proxy returned a non-successful CONNECT response.
	#[error("HTTP proxy rejected CONNECT with status {0}")]
	ConnectRejected(u16),
	/// The proxy produced a malformed or excessively large handshake.
	#[error("malformed proxy handshake: {0}")]
	MalformedHandshake(&'static str),
	/// SOCKS authentication or connection was rejected.
	#[error("SOCKS5 proxy rejected request with code {0}")]
	SocksRejected(u8),
	/// A proxy credential does not fit the wire protocol.
	#[error("proxy credential or host is too long")]
	ValueTooLong,
	/// Socket I/O failed. The underlying diagnostic is discarded because
	/// platform errors may include a credential-bearing endpoint.
	#[error("proxy socket I/O failed")]
	Io,
}

impl From<io::Error> for ProxyError {
	fn from(_error: io::Error) -> Self {
		Self::Io
	}
}

/// A transport connector which applies [`ProxyResolver`] decisions.
///
/// HTTPS through an HTTP proxy is CONNECT-tunnelled before `hyper-rustls`
/// performs origin TLS. Plain HTTP proxy sockets report Hyper's proxy bit so
/// its HTTP/1 encoder retains the required absolute-form request target.
#[derive(Clone)]
pub struct ProxyConnector {
	resolver: Arc<ProxyResolver>,
	provider: Option<Str>,
}

impl ProxyConnector {
	/// Constructs a connector from an immutable policy snapshot.
	#[must_use]
	pub fn new(resolver: ProxyResolver) -> Self {
		Self { resolver: Arc::new(resolver), provider: None }
	}

	/// Constructs a connector whose requests use one provider override.
	#[must_use]
	pub fn for_provider(resolver: ProxyResolver, provider: impl AsRef<str>) -> Self {
		Self { resolver: Arc::new(resolver), provider: Some(Str::new(provider)) }
	}

	pub(crate) fn inject_proxy_auth<B>(&self, request: &mut Request<B>) {
		if !matches!(request.uri().scheme_str(), Some("http")) {
			return;
		}
		let Ok(target) = url_from_uri_origin(request.uri()) else {
			return;
		};
		let provider = self.provider.as_ref().map(Str::as_str);
		let ProxyDecision::Http { auth: Some(auth), .. } = self.resolver.resolve(&target, provider)
		else {
			return;
		};
		let credentials = Zeroizing::new(format!("{}:{}", auth.username, auth.password));
		let encoded = Zeroizing::new(base64::encode(credentials.as_bytes()).into_string());
		let mut bytes = Zeroizing::new(Vec::with_capacity(6 + encoded.len()));
		bytes.extend_from_slice(b"Basic ");
		bytes.extend_from_slice(encoded.as_bytes());
		let Ok(value) = http::HeaderValue::from_bytes(&bytes) else {
			return;
		};
		request.headers_mut().insert(PROXY_AUTHORIZATION, value);
	}

	/// Resolves and opens the transport for one target.
	///
	/// # Errors
	///
	/// Returns [`ProxyError`] when endpoint parsing, connection, authentication,
	/// CONNECT, or SOCKS negotiation fails.
	pub async fn connect(
		&self,
		target: &Url,
		provider: Option<&str>,
	) -> Result<ProxyStream, ProxyError> {
		match self.resolver.resolve(target, provider) {
			ProxyDecision::Direct => connect_url(target)
				.await
				.map(|inner| ProxyStream::new(inner, false)),
			ProxyDecision::Http { url, auth } => {
				let mut stream = connect_url(&url).await?;
				let tunneled = matches!(target.scheme(), "https" | "wss");
				if tunneled {
					http_connect(&mut stream, target, auth.as_ref()).await?;
				}
				Ok(ProxyStream::new(stream, !tunneled))
			},
			ProxyDecision::Socks5 { url, auth } => {
				let mut stream = connect_url(&url).await?;
				socks5_connect(&mut stream, target, auth.as_ref()).await?;
				Ok(ProxyStream::new(stream, false))
			},
		}
	}
}

/// Boxed future used for connection establishment at Hyper's cold connector
/// boundary.
pub type ConnectFuture =
	Pin<Box<dyn Future<Output = Result<ProxyStream, ProxyError>> + Send + 'static>>;

impl Service<Uri> for ProxyConnector {
	type Error = ProxyError;
	type Future = ConnectFuture;
	type Response = ProxyStream;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, uri: Uri) -> Self::Future {
		let connector = self.clone();
		Box::pin(async move {
			let target = url_from_uri_origin(&uri)?;
			let provider = connector.provider.as_ref().map(Str::as_str);
			connector.connect(&target, provider).await
		})
	}
}

/// A connected origin or proxy socket returned to Hyper.
#[derive(Debug)]
pub struct ProxyStream {
	inner:        TokioIo<TcpStream>,
	proxied_http: bool,
}

impl ProxyStream {
	fn new(inner: TcpStream, proxied_http: bool) -> Self {
		Self { inner: TokioIo::new(inner), proxied_http }
	}
}

impl HyperRead for ProxyStream {
	fn poll_read(
		self: Pin<&mut Self>,
		cx: &mut Context<'_>,
		buffer: ReadBufCursor<'_>,
	) -> Poll<io::Result<()>> {
		HyperRead::poll_read(Pin::new(&mut self.get_mut().inner), cx, buffer)
	}
}

impl HyperWrite for ProxyStream {
	fn poll_write(
		self: Pin<&mut Self>,
		cx: &mut Context<'_>,
		buffer: &[u8],
	) -> Poll<Result<usize, io::Error>> {
		HyperWrite::poll_write(Pin::new(&mut self.get_mut().inner), cx, buffer)
	}

	fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> {
		HyperWrite::poll_flush(Pin::new(&mut self.get_mut().inner), cx)
	}

	fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> {
		HyperWrite::poll_shutdown(Pin::new(&mut self.get_mut().inner), cx)
	}

	fn is_write_vectored(&self) -> bool {
		HyperWrite::is_write_vectored(&self.inner)
	}

	fn poll_write_vectored(
		self: Pin<&mut Self>,
		cx: &mut Context<'_>,
		buffers: &[io::IoSlice<'_>],
	) -> Poll<Result<usize, io::Error>> {
		HyperWrite::poll_write_vectored(Pin::new(&mut self.get_mut().inner), cx, buffers)
	}
}

impl Connection for ProxyStream {
	fn connected(&self) -> Connected {
		Connected::new().proxy(self.proxied_http)
	}
}

async fn connect_url(url: &Url) -> Result<TcpStream, ProxyError> {
	let host = url.host_str().ok_or(ProxyError::InvalidEndpoint)?;
	let port = url
		.port_or_known_default()
		.or_else(|| matches!(url.scheme(), "socks5" | "socks5h").then_some(1080))
		.ok_or(ProxyError::InvalidEndpoint)?;
	Ok(TcpStream::connect((host, port)).await?)
}

async fn http_connect(
	stream: &mut TcpStream,
	target: &Url,
	auth: Option<&ProxyAuth>,
) -> Result<(), ProxyError> {
	let host = target.host_str().ok_or(ProxyError::InvalidEndpoint)?;
	let port = target
		.port_or_known_default()
		.ok_or(ProxyError::InvalidEndpoint)?;
	let authority = format_authority(host, port);
	let mut request =
		Zeroizing::new(format!("CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n"));
	if let Some(auth) = auth {
		let credentials = Zeroizing::new(format!("{}:{}", auth.username, auth.password));
		let encoded = Zeroizing::new(base64::encode(credentials.as_bytes()).into_string());
		request.push_str("Proxy-Authorization: Basic ");
		request.push_str(&encoded);
		request.push_str("\r\n");
	}
	request.push_str("\r\n");
	stream.write_all(request.as_bytes()).await?;

	let mut response = Zeroizing::new(Vec::with_capacity(512));
	loop {
		if response.len() == 16 * 1024 {
			return Err(ProxyError::MalformedHandshake("CONNECT headers exceed 16 KiB"));
		}
		let byte = stream.read_u8().await?;
		response.push(byte);
		if response.ends_with(b"\r\n\r\n") {
			break;
		}
	}
	let first_line = response
		.split(|byte| *byte == b'\n')
		.next()
		.and_then(|line| std::str::from_utf8(line).ok())
		.ok_or(ProxyError::MalformedHandshake("invalid CONNECT status line"))?;
	let status = first_line
		.split_ascii_whitespace()
		.nth(1)
		.and_then(|value| value.parse::<u16>().ok())
		.ok_or(ProxyError::MalformedHandshake("invalid CONNECT status"))?;
	if !(200..300).contains(&status) {
		return Err(ProxyError::ConnectRejected(status));
	}
	Ok(())
}

async fn socks5_connect(
	stream: &mut TcpStream,
	target: &Url,
	auth: Option<&ProxyAuth>,
) -> Result<(), ProxyError> {
	let methods: &[u8] = if auth.is_some() {
		&[5, 2, 0, 2]
	} else {
		&[5, 1, 0]
	};
	stream.write_all(methods).await?;
	let mut selection = [0; 2];
	stream.read_exact(&mut selection).await?;
	if selection[0] != 5 || selection[1] == 0xff {
		return Err(ProxyError::SocksRejected(selection[1]));
	}
	if selection[1] == 2 {
		let auth = auth.ok_or(ProxyError::SocksRejected(2))?;
		let username = auth.username.as_bytes();
		let password = auth.password.as_bytes();
		let user_len = u8::try_from(username.len()).map_err(|_| ProxyError::ValueTooLong)?;
		let password_len = u8::try_from(password.len()).map_err(|_| ProxyError::ValueTooLong)?;
		let mut packet = Zeroizing::new(Vec::with_capacity(3 + username.len() + password.len()));
		packet.extend_from_slice(&[1, user_len]);
		packet.extend_from_slice(username);
		packet.push(password_len);
		packet.extend_from_slice(password);
		stream.write_all(&packet).await?;
		stream.read_exact(&mut selection).await?;
		if selection != [1, 0] {
			return Err(ProxyError::SocksRejected(selection[1]));
		}
	} else if selection[1] != 0 {
		return Err(ProxyError::SocksRejected(selection[1]));
	}

	let host = target.host_str().ok_or(ProxyError::InvalidEndpoint)?;
	let port = target
		.port_or_known_default()
		.ok_or(ProxyError::InvalidEndpoint)?;
	let mut packet = Vec::with_capacity(host.len() + 7);
	packet.extend_from_slice(&[5, 1, 0]);
	match host.parse::<IpAddr>() {
		Ok(IpAddr::V4(address)) => {
			packet.push(1);
			packet.extend_from_slice(&address.octets());
		},
		Ok(IpAddr::V6(address)) => {
			packet.push(4);
			packet.extend_from_slice(&address.octets());
		},
		Err(_) => {
			let len = u8::try_from(host.len()).map_err(|_| ProxyError::ValueTooLong)?;
			packet.extend_from_slice(&[3, len]);
			packet.extend_from_slice(host.as_bytes());
		},
	}
	packet.extend_from_slice(&port.to_be_bytes());
	stream.write_all(&packet).await?;

	let mut header = [0; 4];
	stream.read_exact(&mut header).await?;
	if header[0] != 5 || header[1] != 0 {
		return Err(ProxyError::SocksRejected(header[1]));
	}
	let address_len = match header[3] {
		1 => 4,
		4 => 16,
		3 => usize::from(stream.read_u8().await?),
		_ => return Err(ProxyError::MalformedHandshake("invalid SOCKS address type")),
	};
	let mut discard = vec![0; address_len + 2];
	stream.read_exact(&mut discard).await?;
	Ok(())
}

fn url_from_uri_origin(uri: &Uri) -> Result<Url, ProxyError> {
	let scheme = uri.scheme_str().ok_or(ProxyError::InvalidUri)?;
	let authority = uri.authority().ok_or(ProxyError::InvalidUri)?.as_str();
	if authority.contains('@') {
		return Err(ProxyError::InvalidUri);
	}
	let origin = Zeroizing::new(format!("{scheme}://{authority}"));
	Url::parse(&origin).map_err(|_| ProxyError::InvalidUri)
}

fn format_authority(host: &str, port: u16) -> String {
	if host.contains(':') {
		format!("[{host}]:{port}")
	} else {
		format!("{host}:{port}")
	}
}

fn parse_proxy(value: &str) -> Option<ProxyDecision> {
	let mut parsed = if value.contains("://") {
		Url::parse(value).ok()?
	} else {
		Url::parse(&format!("http://{value}")).ok()?
	};
	let auth = (!parsed.username().is_empty()).then(|| ProxyAuth {
		username: decode_url_component(parsed.username()),
		password: decode_url_component(parsed.password().unwrap_or_default()),
	});
	if auth.is_some() {
		parsed.set_username("").ok()?;
		parsed.set_password(None).ok()?;
	}
	match parsed.scheme() {
		"http" => Some(ProxyDecision::Http { url: parsed, auth }),
		"socks5" | "socks5h" => Some(ProxyDecision::Socks5 { url: parsed, auth }),
		_ => None,
	}
}

fn decode_url_component(value: &str) -> Str {
	let source = value.as_bytes();
	let mut decoded = Zeroizing::new(Vec::with_capacity(source.len()));
	let mut cursor = 0;
	while cursor < source.len() {
		if source[cursor] == b'%'
			&& cursor + 2 < source.len()
			&& let (Some(high), Some(low)) =
				(hex_digit(source[cursor + 1]), hex_digit(source[cursor + 2]))
		{
			decoded.push((high << 4) | low);
			cursor += 3;
		} else {
			decoded.push(source[cursor]);
			cursor += 1;
		}
	}
	Str::new(String::from_utf8_lossy(&decoded))
}

const fn hex_digit(value: u8) -> Option<u8> {
	match value {
		b'0'..=b'9' => Some(value - b'0'),
		b'a'..=b'f' => Some(value - b'a' + 10),
		b'A'..=b'F' => Some(value - b'A' + 10),
		_ => None,
	}
}

fn is_always_direct(host: &str) -> bool {
	let host = host.trim_matches(['[', ']']).to_ascii_lowercase();
	if ALWAYS_DIRECT_HOSTS.contains(&host.as_str()) || host.ends_with(".localhost") {
		return true;
	}
	let Ok(address) = host.parse::<IpAddr>() else {
		return false;
	};
	ALWAYS_DIRECT_NETWORKS
		.iter()
		.any(|network| cidr_matches(network, address))
}

fn no_proxy_matches(rule: &str, host: &str, target_port: Option<u16>) -> bool {
	if rule == "*" {
		return true;
	}
	let (rule_host, rule_port) = split_rule_port(rule);
	if rule_port.is_some() && rule_port != target_port {
		return false;
	}
	let target = host.trim_matches(['[', ']']).to_ascii_lowercase();
	let candidate = rule_host.trim_matches(['[', ']']).to_ascii_lowercase();
	if let Ok(address) = target.parse::<IpAddr>()
		&& candidate.contains('/')
	{
		return cidr_matches(&candidate, address);
	}
	if let Some(suffix) = candidate.strip_prefix('.') {
		target == suffix || target.ends_with(&candidate)
	} else {
		target == candidate || target.ends_with(&format!(".{candidate}"))
	}
}

fn split_rule_port(rule: &str) -> (&str, Option<u16>) {
	if let Some(bracket) = rule.strip_prefix('[')
		&& let Some((host, port)) = bracket.split_once("]:")
	{
		return (host, port.parse().ok());
	}
	if rule.parse::<IpAddr>().is_ok() || rule.contains('/') || rule.matches(':').count() != 1 {
		return (rule, None);
	}
	let Some((host, port)) = rule.rsplit_once(':') else {
		return (rule, None);
	};
	port.parse().map_or((rule, None), |port| (host, Some(port)))
}

fn cidr_matches(network: &str, address: IpAddr) -> bool {
	let Some((base, prefix)) = network.split_once('/') else {
		return false;
	};
	let Ok(prefix) = prefix.parse::<u8>() else {
		return false;
	};
	match (base.parse::<IpAddr>(), address) {
		(Ok(IpAddr::V4(base)), IpAddr::V4(address)) if prefix <= 32 => {
			masked_v4(base, prefix) == masked_v4(address, prefix)
		},
		(Ok(IpAddr::V6(base)), IpAddr::V6(address)) if prefix <= 128 => {
			masked_v6(base, prefix) == masked_v6(address, prefix)
		},
		_ => false,
	}
}

fn masked_v4(address: Ipv4Addr, prefix: u8) -> u32 {
	let mask = u32::MAX.checked_shl(u32::from(32 - prefix)).unwrap_or(0);
	u32::from(address) & mask
}

fn masked_v6(address: Ipv6Addr, prefix: u8) -> u128 {
	let mask = u128::MAX.checked_shl(u32::from(128 - prefix)).unwrap_or(0);
	u128::from(address) & mask
}

#[cfg(test)]
mod tests {
	use super::*;

	fn resolver(no_proxy: &str) -> ProxyResolver {
		ProxyResolver::from_pairs([("ALL_PROXY", "http://proxy.test:8080"), ("NO_PROXY", no_proxy)])
	}

	#[test]
	fn no_proxy_matching_table() {
		let cases = [
			("api.example.com", "https://api.example.com", true),
			("api.example.com", "https://other.example.com", false),
			(".example.com", "https://example.com", true),
			(".example.com", "https://deep.api.example.com", true),
			(".example.com", "https://notexample.com", false),
			("*", "https://anywhere.invalid", true),
			("203.0.113.0/24", "http://203.0.113.8", true),
			("203.0.113.0/24", "http://203.0.114.8", false),
			("example.com:8443", "https://example.com:8443", true),
			("example.com:8443", "https://example.com:443", false),
		];
		for (rule, target, expected_direct) in cases {
			let decision = resolver(rule).resolve(&Url::parse(target).unwrap(), None);
			assert_eq!(
				matches!(decision, ProxyDecision::Direct),
				expected_direct,
				"{rule} against {target}"
			);
		}
	}

	#[test]
	fn metadata_endpoints_always_bypass_all_proxy() {
		let resolver = resolver("");
		for target in [
			"http://169.254.169.254/latest/meta-data",
			"http://metadata.google.internal/computeMetadata/v1",
			"http://[fd00:ec2::254]/latest/meta-data",
		] {
			assert_eq!(resolver.resolve(&Url::parse(target).unwrap(), None), ProxyDecision::Direct);
		}
	}

	#[test]
	fn provider_override_beats_environment_proxy() {
		let resolver = ProxyResolver::from_pairs([
			("OMP_PROXY_OPENAI_COMPAT", "http://provider-proxy.test:9000"),
			("HTTPS_PROXY", "http://environment-proxy.test:8080"),
		]);
		let decision =
			resolver.resolve(&Url::parse("https://api.example.test").unwrap(), Some("openai-compat"));
		let ProxyDecision::Http { url, .. } = decision else {
			panic!("expected HTTP proxy");
		};
		assert_eq!(url.host_str(), Some("provider-proxy.test"));
	}

	#[test]
	fn legacy_pi_proxy_names_are_ignored() {
		let resolver = ProxyResolver::from_pairs([
			("PI_PROXY_OPENAI_COMPAT", "http://legacy-provider.test:9000"),
			("PI_PROXY", "http://legacy-global.test:9001"),
		]);
		let target = Url::parse("https://api.example.test").unwrap();
		assert_eq!(resolver.resolve(&target, Some("openai-compat")), ProxyDecision::Direct);
	}

	#[test]
	fn parses_proxy_urls_with_and_without_credentials() {
		let without = parse_proxy("proxy.test:3128").unwrap();
		assert!(matches!(without, ProxyDecision::Http { auth: None, .. }));

		let with = parse_proxy("http://agent:secret@proxy.test:8080").unwrap();
		let ProxyDecision::Http { auth: Some(auth), .. } = with else {
			panic!("expected proxy credentials");
		};
		assert_eq!(auth.username.as_str(), "agent");
		assert_eq!(auth.password.as_str(), "secret");
	}

	#[test]
	fn connector_origin_and_errors_never_retain_query_credentials() {
		const CANARY: &str = "canary-query-credential";
		let uri: Uri = format!("https://provider.test/v1?key={CANARY}")
			.parse()
			.unwrap();
		let origin = url_from_uri_origin(&uri).unwrap();
		assert_eq!(origin.as_str(), "https://provider.test/");
		assert!(!origin.as_str().contains(CANARY));

		for error in [
			ProxyError::InvalidUri,
			ProxyError::InvalidEndpoint,
			ProxyError::from(std::io::Error::other(CANARY)),
		] {
			assert!(!error.to_string().contains(CANARY));
			assert!(!format!("{error:?}").contains(CANARY));
		}
	}
}
