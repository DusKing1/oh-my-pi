//! Shared request, response, and rendered-content contracts for web reads.

use std::future::Future;

use bytes::Bytes;
use omp_core::Str;
use smallvec::SmallVec;
use xutf::{TextBuf as _, Utf8};

/// Maximum downloaded response size accepted by the web reader.
pub const MAX_BYTES: usize = 50 * 1024 * 1024;
/// Maximum rendered scraper output, measured in Unicode scalar values.
pub const MAX_OUTPUT_CHARS: usize = 500_000;

/// User agents tried by the transport, in retry order.
pub const USER_AGENTS: [&str; 3] = [
	"curl/8.0",
	"Mozilla/5.0 (compatible; TextBot/1.0)",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \
	 Chrome/131.0.0.0 Safari/537.36",
];

/// A bounded HTTP GET request issued by the web reader or a scraper.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpRequest {
	/// Absolute request URL.
	pub url:       Str,
	/// Additional request headers.
	pub headers:   SmallVec<(Str, Str), 4>,
	/// Maximum response-body size accepted by the caller.
	pub max_bytes: usize,
}

impl HttpRequest {
	/// Creates a GET request with the web reader's default size limit.
	#[must_use]
	pub fn new(url: impl Into<Str>) -> Self {
		Self { url: url.into(), headers: SmallVec::new(), max_bytes: MAX_BYTES }
	}

	/// Adds a request header.
	#[must_use]
	pub fn with_header(mut self, name: impl Into<Str>, value: impl Into<Str>) -> Self {
		self.headers.push((name.into(), value.into()));
		self
	}

	/// Overrides the maximum response-body size.
	#[must_use]
	pub const fn with_max_bytes(mut self, max_bytes: usize) -> Self {
		self.max_bytes = max_bytes;
		self
	}
}

/// A fully buffered HTTP response returned by the application transport.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpResponse {
	/// URL after redirects.
	pub final_url:    Str,
	/// HTTP status code.
	pub status:       u16,
	/// Lowercased MIME type without parameters, when supplied.
	pub content_type: Option<Str>,
	/// Response headers.
	pub headers:      SmallVec<(Str, Str), 12>,
	/// Buffered response body.
	pub body:         Bytes,
}

impl HttpResponse {
	/// Returns whether the status is in the successful 2xx range.
	#[must_use]
	pub const fn is_success(&self) -> bool {
		self.status >= 200 && self.status < 300
	}

	/// Looks up a response header using ASCII case-insensitive matching.
	#[must_use]
	pub fn header(&self, name: &str) -> Option<&str> {
		self
			.headers
			.iter()
			.find_map(|(key, value)| key.eq_ignore_ascii_case(name).then_some(value.as_ref()))
	}

	/// Decodes the body as UTF-8, replacing malformed sequences.
	#[must_use]
	pub fn text(&self) -> Str {
		let units = xutf::transcode::<Utf8, Utf8>(&self.body);
		String::from_units(units).into()
	}
}

/// Site-specific content ready for the common URL output framing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderResult {
	/// Rendered text or markdown.
	pub content:      Str,
	/// MIME type of the rendered content, when known.
	pub content_type: Option<Str>,
	/// Human-readable extraction method.
	pub method:       Str,
	/// Ordered extraction notes.
	pub notes:        SmallVec<Str, 4>,
}

impl RenderResult {
	/// Builds a markdown result and applies the shared cleanup and size cap.
	#[must_use]
	pub fn markdown(content: &str, method: impl Into<Str>) -> Self {
		let (content, truncated) = finalize_output(content);
		let mut notes = SmallVec::new();
		if truncated {
			notes.push("Output truncated to 500000 characters".into());
		}
		Self { content, content_type: Some("text/markdown".into()), method: method.into(), notes }
	}
}

/// Failure while fetching, decoding, or rendering web content.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum WebError {
	/// The authored target is not a valid supported URL.
	#[error("invalid URL: {0}")]
	InvalidUrl(Str),
	/// The HTTP transport failed before producing a response.
	#[error("HTTP request failed: {0}")]
	Request(Str),
	/// A server returned a non-successful status required by a scraper.
	#[error("HTTP {status} for {url}")]
	HttpStatus {
		/// Final request URL.
		url:    Str,
		/// HTTP status code.
		status: u16,
	},
	/// A response exceeded its caller-provided byte limit.
	#[error("response exceeds {max_bytes} bytes")]
	ResponseTooLarge {
		/// Configured response limit.
		max_bytes: usize,
	},
	/// Response text or structured data could not be decoded.
	#[error("failed to decode web response: {0}")]
	Decode(Str),
	/// Site-specific extraction failed.
	#[error("failed to render web content: {0}")]
	Render(Str),
}

impl WebError {
	/// Creates a transport error from a displayable message.
	pub fn request(message: impl Into<Str>) -> Self {
		Self::Request(message.into())
	}

	/// Creates a response-decoding error from a displayable message.
	pub fn decode(message: impl Into<Str>) -> Self {
		Self::Decode(message.into())
	}

	/// Creates a site-rendering error from a displayable message.
	pub fn render(message: impl Into<Str>) -> Self {
		Self::Render(message.into())
	}

	/// Returns the stable model-facing error message.
	#[must_use]
	pub fn message(&self) -> Str {
		self.to_string().into()
	}
}

/// Application-provided HTTP transport used by the pure web pipeline.
pub trait HttpClient {
	/// Performs a bounded GET request without allocating a boxed future.
	fn get(
		&self,
		request: HttpRequest,
	) -> impl Future<Output = Result<HttpResponse, WebError>> + Send + '_;
}

/// Cleans repeated blank lines and caps rendered output.
#[must_use]
pub fn finalize_output(content: &str) -> (Str, bool) {
	let mut cleaned = String::with_capacity(content.len());
	let mut newline_run = 0_u8;
	for character in content.trim().chars() {
		if character == '\n' {
			newline_run = newline_run.saturating_add(1);
			if newline_run <= 2 {
				cleaned.push(character);
			}
		} else {
			newline_run = 0;
			cleaned.push(character);
		}
	}

	let mut end = cleaned.len();
	let mut count = 0_usize;
	for (index, _) in cleaned.char_indices() {
		if count == MAX_OUTPUT_CHARS {
			end = index;
			break;
		}
		count += 1;
	}
	let truncated = count == MAX_OUTPUT_CHARS && end < cleaned.len();
	if truncated {
		cleaned.truncate(end);
	}
	(cleaned.into(), truncated)
}

/// Returns whether a response is a recognizable bot-block page worth retrying.
#[must_use]
pub fn is_bot_blocked(status: u16, content: &str) -> bool {
	if status != 403 && status != 503 {
		return false;
	}
	let lower = content.to_ascii_lowercase();
	["cloudflare", "captcha", "challenge", "blocked", "access denied", "bot detection"]
		.iter()
		.any(|marker| lower.contains(marker))
}
