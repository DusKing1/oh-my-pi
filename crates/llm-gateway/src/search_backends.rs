//! Production HTTP adapters for Pi-compatible web-search providers.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use http::{Method, Request, header};
use http_body_util::{BodyExt as _, Full};
use omp_core::Str;
use omp_llm_egress::{
	auth_inject::{AuthContext, AuthInjectLayer, CredentialSource},
	client::{Body, EgressClient},
};
use omp_llm_types::{SafeSearch, SearchCitation, SearchRecency, SearchRequest, SearchSource};
use serde_json::{Value, json};
use tower::{Layer as _, Service};

use crate::search::{
	CredentialRequirement, EnginePayload, EngineResult, EngineSpec, SearchAttemptError,
	SearchCredentials, SearchEngineBackend, SearchProviderError, SearchProviderErrorKind,
};

const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

/// Credential admission backed by non-secret broker lease lookup.
pub struct LeaseSearchCredentials<C> {
	source: C,
}

impl<C> LeaseSearchCredentials<C> {
	/// Wraps the canonical sealed egress credential source.
	#[must_use]
	pub const fn new(source: C) -> Self {
		Self { source }
	}
}

impl<C: CredentialSource> SearchCredentials for LeaseSearchCredentials<C> {
	fn has_credential(&self, provider: &str) -> bool {
		self
			.source
			.lease(provider)
			.is_ok_and(|lease| lease.is_some())
	}
}

#[async_trait]
trait SearchHttp: Send + Sync {
	async fn send(&self, request: Request<Body>) -> Result<http::Response<Bytes>, Str>;
}

struct PlainHttp(EgressClient);

#[async_trait]
impl SearchHttp for PlainHttp {
	async fn send(&self, request: Request<Body>) -> Result<http::Response<Bytes>, Str> {
		send_buffered(self.0.clone(), request).await
	}
}

struct AuthenticatedHttp<C: CredentialSource>(
	omp_llm_egress::auth_inject::AuthInject<C, EgressClient>,
);

#[async_trait]
impl<C: CredentialSource> SearchHttp for AuthenticatedHttp<C> {
	async fn send(&self, request: Request<Body>) -> Result<http::Response<Bytes>, Str> {
		send_buffered(self.0.clone(), request).await
	}
}

async fn send_buffered<S>(
	mut service: S,
	request: Request<Body>,
) -> Result<http::Response<Bytes>, Str>
where
	S: Service<Request<Body>, Response = http::Response<hyper::body::Incoming>> + Send,
	S::Future: Send,
	S::Error: std::fmt::Display,
{
	let response = service
		.call(request)
		.await
		.map_err(|error| Str::from(error.to_string()))?;
	let (parts, body) = response.into_parts();
	let bytes = body
		.collect()
		.await
		.map_err(|error| Str::from(error.to_string()))?
		.to_bytes();
	Ok(http::Response::from_parts(parts, bytes))
}

/// Real, pooled HTTP implementation of every engine in the search registry.
#[derive(Clone)]
pub struct ProductionSearchBackend {
	client:    Arc<dyn SearchHttp>,
	endpoints: Arc<BTreeMap<Str, Str>>,
}

impl ProductionSearchBackend {
	/// Creates adapters without broker injection, for credential-free engines
	/// and deterministic transport tests.
	#[must_use]
	pub fn new(client: Arc<EgressClient>) -> Self {
		Self {
			client:    Arc::new(PlainHttp(client.as_ref().clone())),
			endpoints: Arc::new(BTreeMap::new()),
		}
	}

	/// Creates adapters whose requests redeem sealed broker leases inside the
	/// canonical authentication layer.
	#[must_use]
	pub fn authenticated<C: CredentialSource>(client: Arc<EgressClient>, source: C) -> Self {
		let client = AuthInjectLayer::new(source).layer(client.as_ref().clone());
		Self { client: Arc::new(AuthenticatedHttp(client)), endpoints: Arc::new(BTreeMap::new()) }
	}

	/// Overrides one provider endpoint, primarily for private `SearXNG` and
	/// Firecrawl deployments and deterministic integration tests.
	#[must_use]
	pub fn with_endpoint(mut self, provider: impl Into<Str>, endpoint: impl Into<Str>) -> Self {
		Arc::make_mut(&mut self.endpoints).insert(provider.into(), endpoint.into());
		self
	}

	fn endpoint(&self, id: &str) -> &'_ str {
		self
			.endpoints
			.get(id)
			.map_or_else(|| default_endpoint(id), Str::as_str)
	}
}

#[async_trait]
impl SearchEngineBackend for ProductionSearchBackend {
	async fn search(
		&self,
		engine: &'static EngineSpec,
		request: &SearchRequest,
		_timeout: Duration,
	) -> Result<EngineResult, SearchAttemptError> {
		if engine.id == "public" {
			// Pi's Public Web engine fans out to every credential-free scraper.
			// `join_all` retains engine order, while failed/challenged scrapers do
			// not discard partial results returned by their peers.
			let attempts = ["duckduckgo", "google", "ecosia", "startpage", "mojeek"]
				.into_iter()
				.map(|id| {
					let spec = crate::search::ENGINES
						.iter()
						.find(|candidate| candidate.id == id)
						.expect("registered public engine");
					self.execute(spec, request)
				});
			let mut sources = Vec::new();
			for result in futures::future::join_all(attempts)
				.await
				.into_iter()
				.flatten()
			{
				let EnginePayload::Raw { sources: found } = result.payload else {
					continue;
				};
				for source in found {
					if !sources
						.iter()
						.any(|known: &SearchSource| known.url == source.url)
					{
						sources.push(source);
					}
				}
			}
			if !sources.is_empty() {
				return Ok(EngineResult::new(EnginePayload::Raw { sources }));
			}
			return Err(provider_error(
				engine.id,
				SearchProviderErrorKind::Parse,
				"public search engines returned no results",
			));
		}
		self.execute(engine, request).await
	}
}

impl ProductionSearchBackend {
	async fn execute(
		&self,
		engine: &'static EngineSpec,
		request: &SearchRequest,
	) -> Result<EngineResult, SearchAttemptError> {
		let wire = build_request(engine, self.endpoint(engine.id), request)?;
		let response = self.client.send(wire).await.map_err(|error| {
			provider_error(engine.id, SearchProviderErrorKind::Transport, error.as_str())
		})?;
		let status = response.status();
		let request_id = response
			.headers()
			.get("x-request-id")
			.and_then(|value| value.to_str().ok())
			.map(Str::from);
		let bytes = response.into_body();
		if bytes.len() > MAX_RESPONSE_BYTES {
			return Err(provider_error(
				engine.id,
				SearchProviderErrorKind::Parse,
				"search response exceeded 2 MiB",
			));
		}
		if !status.is_success() {
			let message = String::from_utf8_lossy(&bytes);
			return Err(provider_error(
				engine.id,
				SearchProviderErrorKind::Status(status.as_u16()),
				message.trim(),
			));
		}
		let mut result = if is_scraper(engine.id) {
			parse_html(engine.id, &bytes, request.limit)
		} else {
			parse_json(engine.id, &bytes, request.limit)
		}?;
		if let Some(request_id) = request_id {
			result
				.props
				.insert_ns(engine.id, "request_id", Value::String(request_id.to_string()));
		}
		Ok(result)
	}
}

fn build_request(
	engine: &EngineSpec,
	endpoint: &str,
	request: &SearchRequest,
) -> Result<Request<Full<Bytes>>, SearchAttemptError> {
	let id = engine.id;
	let limit = request.limit.clamp(1, 20);
	let mut method = Method::POST;
	let mut url = endpoint.to_owned();
	let mut body = json!({"query": request.query.as_str(), "limit": limit});
	match id {
		"brave" => {
			method = Method::GET;
			url = query_url(endpoint, &[
				("q", request.query.as_str()),
				("count", &limit.to_string()),
				("extra_snippets", "true"),
				("text_decorations", "false"),
				("safesearch", safe_search(request.safesearch)),
			]);
		},
		"jina" => {
			method = Method::GET;
			url = query_url(endpoint, &[("q", request.query.as_str())]);
		},
		"tinyfish" => {
			method = Method::GET;
			url = query_url(endpoint, &[("q", request.query.as_str()), ("limit", &limit.to_string())]);
		},
		"searxng" => {
			method = Method::GET;
			url = query_url(endpoint, &[("q", request.query.as_str()), ("format", "json")]);
		},
		"duckduckgo" => {
			body = json!({"q": request.query.as_str()});
		},
		"google" | "ecosia" | "mojeek" => {
			method = Method::GET;
			url = query_url(endpoint, &[("q", request.query.as_str())]);
		},
		"startpage" => {
			body = json!({"query": request.query.as_str()});
		},
		"perplexity" => {
			body = json!({"model":"sonar", "messages":[{"role":"user","content":request.query.as_str()}], "search_recency_filter": recency(request.recency)});
		},
		"gemini" => {
			body = json!({"contents":[{"parts":[{"text":request.query.as_str()}]}],"tools":[{"google_search":{}}]});
		},
		"anthropic" => {
			body = json!({"model":"claude-sonnet-4-20250514","max_tokens":4096,"messages":[{"role":"user","content":request.query.as_str()}],"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":limit}]});
		},
		"codex" => {
			body = json!({"model":"gpt-5.2-codex","input":request.query.as_str(),"tools":[{"type":"web_search_preview"}]});
		},
		"xai" => {
			body = json!({"model":"grok-4.5","input":request.query.as_str(),"tools":[{"type":"web_search"}]});
		},
		"zai" => {
			body = json!({"jsonrpc":"2.0","id":"omp-search","method":"tools/call","params":{"name":"web_search_prime","arguments":{"query":request.query.as_str(),"count":limit}}});
		},
		"kimi" => {
			body = json!({"text_query":request.query.as_str(),"limit":limit,"enable_page_crawling":false,"timeout_seconds":30});
		},
		"exa" => {
			body = json!({"query":request.query.as_str(),"numResults":limit,"contents":{"text":true},"includeDomains":request.allowed_domains,"excludeDomains":request.excluded_domains,"startPublishedDate":nonempty(&request.after),"endPublishedDate":nonempty(&request.before)});
		},
		"tavily" => {
			body = json!({"query":request.query.as_str(),"max_results":limit,"search_depth":"basic","include_answer":"advanced","include_domains":request.allowed_domains,"exclude_domains":request.excluded_domains});
		},
		"firecrawl" => body = json!({"query":request.query.as_str(),"limit":limit}),
		"parallel" => {
			body = json!({"objective":request.query.as_str(),"search_queries":[request.query.as_str()],"max_results":limit});
		},
		"synthetic" => body = json!({"query":request.query.as_str()}),
		"kagi" => body = json!({"query":request.query.as_str(),"workflow":"search","limit":limit}),
		_ => {},
	}
	let encoded = if matches!(id, "duckduckgo" | "startpage") {
		let key = if id == "duckduckgo" { "q" } else { "query" };
		format!("{key}={}", percent_encode(request.query.as_str())).into_bytes()
	} else if method == Method::GET {
		Vec::new()
	} else {
		serde_json::to_vec(&body).expect("JSON value serializes")
	};
	let mut builder = Request::builder()
		.method(method)
		.uri(url)
		.header(
			header::ACCEPT,
			if is_scraper(id) {
				"text/html,application/xhtml+xml"
			} else {
				"application/json"
			},
		)
		.header(
			header::USER_AGENT,
			"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/149 \
			 Safari/537.36",
		);
	if !encoded.is_empty() {
		builder = builder.header(
			header::CONTENT_TYPE,
			if matches!(id, "duckduckgo" | "startpage") {
				"application/x-www-form-urlencoded"
			} else {
				"application/json"
			},
		);
	}
	builder = match id {
		"anthropic" => builder.header("anthropic-version", "2023-06-01"),
		"parallel" => builder.header("parallel-beta", "search-extract-2025-10-10"),
		"jina" => builder.header("x-respond-with", "no-content"),
		_ => builder,
	};
	let mut request = builder
		.body(Full::new(Bytes::from(encoded)))
		.map_err(|error| provider_error(id, SearchProviderErrorKind::Parse, &error.to_string()))?;
	if let CredentialRequirement::ApiKey { provider }
	| CredentialRequirement::OptionalApiKey { provider } = engine.credential
	{
		request.extensions_mut().insert(AuthContext::new(provider));
	}
	Ok(request)
}

fn parse_json(id: &str, bytes: &[u8], limit: u32) -> Result<EngineResult, SearchAttemptError> {
	let value: Value = serde_json::from_slice(bytes).map_err(|_| {
		provider_error(id, SearchProviderErrorKind::Parse, "search provider returned invalid JSON")
	})?;
	let mut sources = Vec::new();
	for path in result_arrays(id) {
		if let Some(items) = pointer(&value, path).and_then(Value::as_array) {
			for item in items {
				if let Some(source) = source_from_value(item) {
					sources.push(source);
				}
				if sources.len() >= limit.max(1) as usize {
					break;
				}
			}
			if !sources.is_empty() {
				break;
			}
		}
	}
	let answer = first_string(&value, &[
		"/answer",
		"/output_text",
		"/choices/0/message/content",
		"/candidates/0/content/parts/0/text",
		"/content/0/text",
	]);
	let citations = pointer(&value, "/citations")
		.and_then(Value::as_array)
		.map(|items| items.iter().filter_map(citation_from_value).collect())
		.unwrap_or_default();
	let payload = if answer.is_empty() {
		EnginePayload::Raw { sources }
	} else if sources.is_empty() {
		EnginePayload::Synthesized { answer, sources, citations }
	} else {
		EnginePayload::Hybrid { answer, sources, citations }
	};
	Ok(EngineResult::new(payload))
}

fn parse_html(id: &str, bytes: &[u8], limit: u32) -> Result<EngineResult, SearchAttemptError> {
	let html = String::from_utf8_lossy(bytes);
	let lowered = html.to_ascii_lowercase();
	if lowered.contains("captcha")
		|| lowered.contains("unusual traffic")
		|| lowered.contains("automated queries")
	{
		return Err(provider_error(
			id,
			SearchProviderErrorKind::Status(429),
			"search engine challenge page",
		));
	}
	let mut sources = Vec::new();
	let mut cursor = html.as_ref();
	while let Some(anchor) = cursor.find("<a") {
		cursor = &cursor[anchor + 2..];
		let Some(tag_end) = cursor.find('>') else {
			break;
		};
		let tag = &cursor[..tag_end];
		let Some(href) = attribute(tag, "href") else {
			cursor = &cursor[tag_end + 1..];
			continue;
		};
		let rest = &cursor[tag_end + 1..];
		let Some(close) = rest.find("</a>") else {
			break;
		};
		let title = strip_tags(&rest[..close]);
		let url = unwrap_search_url(href);
		if (url.starts_with("https://") || url.starts_with("http://")) && !title.is_empty() {
			sources.push(
				SearchSource::builder()
					.url(Str::from(url))
					.title(Str::from(title))
					.snippet(Str::default())
					.published_at(Str::default())
					.author(Str::default())
					.build(),
			);
			if sources.len() >= limit.max(1) as usize {
				break;
			}
		}
		cursor = &rest[close + 4..];
	}
	if sources.is_empty() {
		return Err(provider_error(
			id,
			SearchProviderErrorKind::Parse,
			"search engine returned no parseable results",
		));
	}
	Ok(EngineResult::new(EnginePayload::Raw { sources }))
}

fn source_from_value(value: &Value) -> Option<SearchSource> {
	let url = field_string(value, &["url", "link", "id"])?;
	if !(url.starts_with("http://") || url.starts_with("https://")) {
		return None;
	}
	let title = field_string(value, &["title", "name"]).unwrap_or(url);
	let snippet =
		field_string(value, &["description", "snippet", "text", "content", "summary"]).unwrap_or("");
	let published = field_string(value, &[
		"publishedDate",
		"published_date",
		"published_at",
		"date",
		"time",
		"age",
	])
	.unwrap_or("");
	let author = field_string(value, &["author", "publisher", "site_name"]).unwrap_or("");
	let score = value.get("score").and_then(Value::as_f64);
	Some(
		SearchSource::builder()
			.url(Str::from(url))
			.title(Str::from(strip_tags(title)))
			.snippet(Str::from(strip_tags(snippet)))
			.published_at(Str::from(published))
			.author(Str::from(author))
			.maybe_score(score)
			.build(),
	)
}

fn citation_from_value(value: &Value) -> Option<SearchCitation> {
	let url = value
		.as_str()
		.or_else(|| field_string(value, &["url", "uri"]))?;
	Some(
		SearchCitation::builder()
			.url(Str::from(url))
			.title(Str::from(field_string(value, &["title"]).unwrap_or("")))
			.cited_text(Str::from(field_string(value, &["text", "cited_text"]).unwrap_or("")))
			.build(),
	)
}

fn result_arrays(id: &str) -> &'static [&'static str] {
	match id {
		"brave" => &["/web/results"],
		"exa" | "tavily" | "synthetic" | "tinyfish" | "searxng" => &["/results"],
		"firecrawl" => &["/data", "/data/web"],
		"parallel" => &["/results", "/search_results"],
		"kimi" => &["/search_results"],
		"kagi" => &["/data/search", "/data/news", "/data/video", "/data/infobox"],
		"jina" => &["/data", ""],
		_ => &["/results", "/data/results", "/sources", "/citations"],
	}
}

fn default_endpoint(id: &str) -> &'static str {
	match id {
		"perplexity" => "https://api.perplexity.ai/chat/completions",
		"gemini" => {
			"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
		},
		"anthropic" => "https://api.anthropic.com/v1/messages",
		"codex" => "https://api.openai.com/v1/responses",
		"xai" => "https://api.x.ai/v1/responses",
		"zai" => "https://api.z.ai/api/mcp/web_search_prime/mcp",
		"exa" => "https://api.exa.ai/search",
		"tinyfish" => "https://api.search.tinyfish.ai/search",
		"jina" => "https://s.jina.ai/",
		"kagi" => "https://kagi.com/api/v1/search",
		"tavily" => "https://api.tavily.com/search",
		"firecrawl" => "https://api.firecrawl.dev/v2/search",
		"brave" => "https://api.search.brave.com/res/v1/web/search",
		"kimi" => "https://api.kimi.com/coding/v1/search",
		"parallel" => "https://api.parallel.ai/v1beta/search",
		"synthetic" => "https://api.synthetic.new/v2/search",
		"searxng" => "https://search.bus-hit.me/search",
		"duckduckgo" => "https://html.duckduckgo.com/html/",
		"google" => "https://www.google.com/search",
		"ecosia" => "https://www.ecosia.org/search",
		"startpage" => "https://www.startpage.com/sp/search",
		"mojeek" => "https://www.mojeek.com/search",
		_ => "",
	}
}

fn is_scraper(id: &str) -> bool {
	matches!(id, "duckduckgo" | "google" | "ecosia" | "startpage" | "mojeek")
}
fn pointer<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
	if path.is_empty() {
		Some(value)
	} else {
		value.pointer(path)
	}
}
fn first_string(value: &Value, paths: &[&str]) -> Str {
	paths
		.iter()
		.find_map(|path| value.pointer(path).and_then(Value::as_str))
		.unwrap_or("")
		.into()
}
fn field_string<'a>(value: &'a Value, fields: &[&str]) -> Option<&'a str> {
	fields
		.iter()
		.find_map(|field| value.get(*field).and_then(Value::as_str))
}
fn nonempty(value: &Str) -> Option<&str> {
	(!value.is_empty()).then_some(value.as_str())
}
fn recency(value: Option<SearchRecency>) -> Option<&'static str> {
	value.map(|value| match value {
		SearchRecency::Day => "day",
		SearchRecency::Week => "week",
		SearchRecency::Month => "month",
		SearchRecency::Year => "year",
		_ => "month",
	})
}
const fn safe_search(value: Option<SafeSearch>) -> &'static str {
	match value {
		Some(SafeSearch::Off) => "off",
		Some(SafeSearch::Strict) => "strict",
		_ => "moderate",
	}
}
fn query_url(base: &str, pairs: &[(&str, &str)]) -> String {
	let separator = if base.contains('?') { '&' } else { '?' };
	let query = pairs
		.iter()
		.map(|(key, value)| format!("{key}={}", percent_encode(value)))
		.collect::<Vec<_>>()
		.join("&");
	format!("{base}{separator}{query}")
}
fn percent_encode(value: &str) -> String {
	let mut output = String::with_capacity(value.len());
	for byte in value.bytes() {
		if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
			output.push(char::from(byte));
		} else {
			use std::fmt::Write as _;
			let _ = write!(output, "%{byte:02X}");
		}
	}
	output
}
fn attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
	let start = tag.find(&format!("{name}=\""))? + name.len() + 2;
	let end = tag[start..].find('"')?;
	Some(&tag[start..start + end])
}
fn strip_tags(value: &str) -> String {
	let mut output = String::with_capacity(value.len());
	let mut tag = false;
	for character in value.chars() {
		match character {
			'<' => tag = true,
			'>' => tag = false,
			_ if !tag => output.push(character),
			_ => {},
		}
	}
	let normalized = output
		.replace("&amp;", "&")
		.replace("&quot;", "\"")
		.replace("&#39;", "'")
		.split_whitespace()
		.collect::<Vec<_>>()
		.join(" ");
	let (kept, _) = xutf::truncate_measured_str(&normalized, 8_000);
	if kept.len() == normalized.len() {
 		normalized
 	} else {
 		format!("{kept}…")
 	}
}
fn unwrap_search_url(value: &str) -> &str {
	value
		.split("uddg=")
		.nth(1)
		.and_then(|value| value.split('&').next())
		.unwrap_or(value)
}
fn provider_error(id: &str, kind: SearchProviderErrorKind, message: &str) -> SearchAttemptError {
	SearchProviderError {
		engine: id.into(),
		kind,
		message: if message.is_empty() {
			"search provider request failed".into()
		} else {
			message.into()
		},
	}
	.into()
}
