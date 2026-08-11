//! Registry policy and fallback execution for the web-search facet.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use omp_core::Str;
use omp_llm_types::{
	Cost, Props, SearchCitation, SearchRequest, SearchResponse, SearchSource, Unsupported, Usage,
	facet,
};
use smallvec::SmallVec;

/// Default deadline applied to one engine attempt.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
/// Largest deadline accepted from a request.
pub const MAX_TIMEOUT: Duration = Duration::from_secs(300);

/// The credential an engine needs before it may enter the automatic chain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CredentialRequirement {
	/// No secret is required.
	None,
	/// A provider API key is required.
	ApiKey {
		/// Broker/provider key used to resolve the secret.
		provider: &'static str,
	},
	/// An API key improves the engine but its public endpoint also works without
	/// one.
	OptionalApiKey {
		/// Broker/provider key used to resolve the optional secret.
		provider: &'static str,
	},
}

/// Shape of content an engine returns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchParadigm {
	/// Ranked links and snippets.
	Raw,
	/// A generated answer with citations.
	Synthesized,
	/// Both a generated answer and ranked results.
	Hybrid,
}

/// Query controls an engine can enforce before post-filtering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchCapabilities {
	/// Supports domain allowlists.
	pub domains:    bool,
	/// Supports publication date bounds.
	pub dates:      bool,
	/// Supports geographic bias.
	pub location:   bool,
	/// Supports safe-search controls.
	pub safesearch: bool,
}

/// An exceptional adapter selected by registry data rather than embedded in
/// policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineHook {
	/// Fan out to the credential-free scraper set and deduplicate its hits.
	PublicScrapers,
}

/// Static metadata for one search engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineSpec {
	/// Stable request and configuration id.
	pub id:           &'static str,
	/// Human-readable engine name.
	pub label:        &'static str,
	/// Credential admission rule.
	pub credential:   CredentialRequirement,
	/// Response paradigm.
	pub paradigm:     SearchParadigm,
	/// Native query capabilities.
	pub capabilities: SearchCapabilities,
	/// Isolated exceptional adapter, if any.
	pub hook:         Option<EngineHook>,
}

const RAW_FULL: SearchCapabilities =
	SearchCapabilities { domains: true, dates: true, location: true, safesearch: true };
const RAW_LIMITED: SearchCapabilities = SearchCapabilities {
	domains:    false,
	dates:      false,
	location:   false,
	safesearch: false,
};
const HYBRID: SearchCapabilities =
	SearchCapabilities { domains: true, dates: true, location: false, safesearch: false };

/// Built-in engines in Pi's default fallback order.
///
/// The registry is intentionally complete: credential admission may skip an
/// unavailable provider, but every registered Pi search id has a real wire
/// adapter in [`crate::search_backends`].
pub static ENGINES: [EngineSpec; 23] = [
	engine_spec(
		"perplexity",
		"Perplexity",
		api_key("perplexity"),
		SearchParadigm::Synthesized,
		HYBRID,
	),
	engine_spec("gemini", "Gemini", api_key("google"), SearchParadigm::Synthesized, HYBRID),
	engine_spec("anthropic", "Anthropic", api_key("anthropic"), SearchParadigm::Synthesized, HYBRID),
	engine_spec("codex", "Codex", api_key("openai-codex"), SearchParadigm::Synthesized, HYBRID),
	engine_spec("xai", "xAI", api_key("xai"), SearchParadigm::Synthesized, HYBRID),
	engine_spec("zai", "Z.ai", api_key("zai"), SearchParadigm::Synthesized, HYBRID),
	engine_spec("exa", "Exa", api_key("exa"), SearchParadigm::Hybrid, HYBRID),
	engine_spec("tinyfish", "Tinyfish", api_key("tinyfish"), SearchParadigm::Raw, HYBRID),
	engine_spec("jina", "Jina", api_key("jina"), SearchParadigm::Raw, RAW_LIMITED),
	engine_spec("kagi", "Kagi", api_key("kagi"), SearchParadigm::Raw, RAW_FULL),
	engine_spec("tavily", "Tavily", api_key("tavily"), SearchParadigm::Hybrid, HYBRID),
	engine_spec(
		"firecrawl",
		"Firecrawl",
		CredentialRequirement::OptionalApiKey { provider: "firecrawl" },
		SearchParadigm::Raw,
		HYBRID,
	),
	engine_spec("brave", "Brave", api_key("brave"), SearchParadigm::Raw, RAW_FULL),
	engine_spec("kimi", "Kimi", api_key("kimi-code"), SearchParadigm::Raw, HYBRID),
	engine_spec("parallel", "Parallel", api_key("parallel"), SearchParadigm::Hybrid, HYBRID),
	engine_spec("synthetic", "Synthetic", api_key("synthetic"), SearchParadigm::Raw, HYBRID),
	engine_spec("searxng", "SearXNG", CredentialRequirement::None, SearchParadigm::Raw, RAW_FULL),
	engine_spec(
		"duckduckgo",
		"DuckDuckGo",
		CredentialRequirement::None,
		SearchParadigm::Raw,
		RAW_LIMITED,
	),
	engine_spec("google", "Google", CredentialRequirement::None, SearchParadigm::Raw, RAW_FULL),
	engine_spec("ecosia", "Ecosia", CredentialRequirement::None, SearchParadigm::Raw, RAW_LIMITED),
	engine_spec(
		"startpage",
		"Startpage",
		CredentialRequirement::None,
		SearchParadigm::Raw,
		RAW_LIMITED,
	),
	engine_spec("mojeek", "Mojeek", CredentialRequirement::None, SearchParadigm::Raw, RAW_LIMITED),
	EngineSpec {
		id:           "public",
		label:        "Public Web",
		credential:   CredentialRequirement::None,
		paradigm:     SearchParadigm::Raw,
		capabilities: RAW_FULL,
		hook:         Some(EngineHook::PublicScrapers),
	},
];

const fn api_key(provider: &'static str) -> CredentialRequirement {
	CredentialRequirement::ApiKey { provider }
}

const fn engine_spec(
	id: &'static str,
	label: &'static str,
	credential: CredentialRequirement,
	paradigm: SearchParadigm,
	capabilities: SearchCapabilities,
) -> EngineSpec {
	EngineSpec { id, label, credential, paradigm, capabilities, hook: None }
}

/// Non-secret credential availability used by candidate admission.
pub trait SearchCredentials: Send + Sync {
	/// Returns whether a provider lease can currently be issued.
	fn has_credential(&self, provider: &str) -> bool;
}

/// Provider-specific response content before it enters the unified envelope.
#[derive(Clone, Debug, PartialEq)]
pub enum EnginePayload {
	/// Raw ranked hits.
	Raw {
		/// Retrieved sources.
		sources: Vec<SearchSource>,
	},
	/// Generated answer and its evidence.
	Synthesized {
		/// Generated narrative.
		answer:    Str,
		/// Retrieved evidence, when exposed by the engine.
		sources:   Vec<SearchSource>,
		/// Answer anchors.
		citations: Vec<SearchCitation>,
	},
	/// Generated answer plus first-class ranked hits.
	Hybrid {
		/// Generated narrative.
		answer:    Str,
		/// Ranked hits.
		sources:   Vec<SearchSource>,
		/// Answer anchors.
		citations: Vec<SearchCitation>,
	},
}

/// Provider output shared across the three response paradigms.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineResult {
	/// Paradigm-specific content.
	pub payload:        EnginePayload,
	/// Queries generated by the provider.
	pub search_queries: Vec<Str>,
	/// Suggested follow-up queries.
	pub related:        Vec<Str>,
	/// Provider diagnostics.
	pub warnings:       Vec<Str>,
	/// Provider usage, when available.
	pub usage:          Option<Usage>,
	/// Metered cost, when available.
	pub cost:           Option<Cost>,
	/// Controls the provider could not honor.
	pub unsupported:    Vec<Unsupported>,
	/// Namespaced provider metadata.
	pub props:          Props,
}

impl EngineResult {
	/// Builds a result with empty common metadata.
	#[must_use]
	pub fn new(payload: EnginePayload) -> Self {
		Self {
			payload,
			search_queries: Vec::new(),
			related: Vec::new(),
			warnings: Vec::new(),
			usage: None,
			cost: None,
			unsupported: Vec::new(),
			props: Props::default(),
		}
	}
}

/// Provider failure category used to decide whether fallback is safe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchProviderErrorKind {
	/// HTTP response status.
	Status(u16),
	/// The provider response could not be decoded.
	Parse,
	/// The provider exceeded its per-attempt deadline.
	Timeout,
	/// A transport failure for which replay safety is unknown.
	Transport,
}

/// One named provider failure.
#[derive(Clone, Debug, thiserror::Error)]
#[error("{engine}: {message}")]
pub struct SearchProviderError {
	/// Engine that failed.
	pub engine:  Str,
	/// Failure category.
	pub kind:    SearchProviderErrorKind,
	/// Diagnostic safe to return to a caller.
	pub message: Str,
}

impl SearchProviderError {
	/// Returns whether policy may advance to another engine.
	#[must_use]
	pub fn permits_fallback(&self) -> bool {
		match self.kind {
			SearchProviderErrorKind::Status(status) => {
				matches!(status, 204 | 401 | 403 | 404 | 429) || (500..600).contains(&status)
			},
			SearchProviderErrorKind::Parse | SearchProviderErrorKind::Timeout => true,
			SearchProviderErrorKind::Transport => false,
		}
	}
}

/// Attempt outcome returned by an engine adapter.
#[derive(Clone, Debug, thiserror::Error)]
pub enum SearchAttemptError {
	/// The caller cancelled the operation; policy must not try another engine.
	#[error("search cancelled")]
	Cancelled,
	/// The engine failed.
	#[error(transparent)]
	Provider(#[from] SearchProviderError),
}

/// Adapter boundary for engine-specific wire implementations.
#[async_trait]
pub trait SearchEngineBackend: Send + Sync {
	/// Executes one engine attempt.
	async fn search(
		&self,
		engine: &'static EngineSpec,
		request: &SearchRequest,
		timeout: Duration,
	) -> Result<EngineResult, SearchAttemptError>;
}

/// Failure from registry policy or all attempted providers.
#[derive(Clone, Debug, thiserror::Error)]
pub enum SearchRegistryError {
	/// A pinned engine id is not registered.
	#[error("unknown search engine: {0}")]
	UnknownEngine(Str),
	/// No engine has usable credentials.
	#[error("no configured search engine")]
	NoAvailableEngine,
	/// The caller cancelled the request.
	#[error("search cancelled")]
	Cancelled,
	/// A non-fallback-safe provider failure terminated execution.
	#[error(transparent)]
	Provider(SearchProviderError),
	/// Every available engine failed in fallback-safe ways.
	#[error("all search engines failed")]
	AllProvidersFailed {
		/// Failures in attempted order.
		failures: Box<SmallVec<SearchProviderError, 10>>,
	},
}

/// Data-driven engine registry and fallback policy.
pub struct SearchRegistry {
	credentials:      Arc<dyn SearchCredentials>,
	backend:          Arc<dyn SearchEngineBackend>,
	configured_order: SmallVec<Str, 10>,
}

impl SearchRegistry {
	/// Creates a registry using the built-in order.
	#[must_use]
	pub fn new(
		credentials: Arc<dyn SearchCredentials>,
		backend: Arc<dyn SearchEngineBackend>,
	) -> Self {
		Self { credentials, backend, configured_order: SmallVec::new() }
	}

	/// Creates the production registry backed by broker-sealed credential
	/// leases, the shared pooled egress client, and Pi-compatible adapters.
	#[must_use]
	pub fn production<C>(source: C, client: Arc<omp_llm_egress::client::EgressClient>) -> Self
	where
		C: omp_llm_egress::auth_inject::CredentialSource,
	{
		Self::new(
			Arc::new(crate::search_backends::LeaseSearchCredentials::new(source.clone())),
			Arc::new(crate::search_backends::ProductionSearchBackend::authenticated(client, source)),
		)
	}

	/// Replaces the configured priority prefix; unknown ids are ignored.
	#[must_use]
	pub fn with_configured_order<I, S>(mut self, order: I) -> Self
	where
		I: IntoIterator<Item = S>,
		S: Into<Str>,
	{
		for id in order {
			let id = id.into();
			if engine(id.as_str()).is_some() && !self.configured_order.contains(&id) {
				self.configured_order.push(id);
			}
		}
		self
	}

	/// Resolves the credentialed attempt order for a request.
	///
	/// A non-empty request engine is an explicit, terminal override. Automatic
	/// selection uses the configured priority prefix followed by Pi's built-in
	/// order.
	pub fn candidates(
		&self,
		pinned: &str,
	) -> Result<SmallVec<&'static EngineSpec, 12>, SearchRegistryError> {
		if !pinned.is_empty() {
			let spec =
				engine(pinned).ok_or_else(|| SearchRegistryError::UnknownEngine(pinned.into()))?;
			return Ok(self
				.has_credential(spec)
				.then_some(spec)
				.into_iter()
				.collect());
		}
		let mut ordered: SmallVec<&'static EngineSpec, 12> = SmallVec::new();
		for id in &self.configured_order {
			push_unique(&mut ordered, engine(id.as_str()));
		}
		for spec in &ENGINES {
			push_unique(&mut ordered, Some(spec));
		}
		Ok(ordered
			.into_iter()
			.filter(|spec| self.has_credential(spec))
			.collect())
	}

	/// Executes a search and applies the fallback and lenient-filter policies.
	pub async fn execute(
		&self,
		request: SearchRequest,
	) -> Result<SearchResponse, SearchRegistryError> {
		let candidates = self.candidates(request.engine.as_str())?;
		if candidates.is_empty() {
			return Err(SearchRegistryError::NoAvailableEngine);
		}
		let timeout = request_timeout(request.timeout_ms);
		let query = ParsedQuery::parse(request.query.as_str(), &request);
		let mut failures: SmallVec<SearchProviderError, 10> = SmallVec::new();
		let mut attempted = false;
		for spec in candidates {
			if !self.has_credential(spec) {
				continue;
			}
			attempted = true;
			let attempt =
				tokio::time::timeout(timeout, self.backend.search(spec, &request, timeout)).await;
			let result = match attempt {
				Err(_) => Err(SearchAttemptError::Provider(SearchProviderError {
					engine:  spec.id.into(),
					kind:    SearchProviderErrorKind::Timeout,
					message: "search attempt timed out".into(),
				})),
				Ok(result) => result,
			};
			match result {
				Ok(result) => return Ok(unify(spec, result, &query)),
				Err(SearchAttemptError::Cancelled) => return Err(SearchRegistryError::Cancelled),
				Err(SearchAttemptError::Provider(error)) if error.permits_fallback() => {
					failures.push(error);
				},
				Err(SearchAttemptError::Provider(error)) => {
					return Err(SearchRegistryError::Provider(error));
				},
			}
		}
		if !attempted {
			return Err(SearchRegistryError::NoAvailableEngine);
		}
		Err(SearchRegistryError::AllProvidersFailed { failures: Box::new(failures) })
	}

	fn has_credential(&self, spec: &EngineSpec) -> bool {
		match spec.credential {
			CredentialRequirement::None | CredentialRequirement::OptionalApiKey { .. } => true,
			CredentialRequirement::ApiKey { provider } => self.credentials.has_credential(provider),
		}
	}
}

#[async_trait]
impl facet::Search for SearchRegistry {
	async fn search(&self, request: SearchRequest) -> Result<SearchResponse, facet::Error> {
		self
			.execute(request)
			.await
			.map_err(|error| facet::Error::Provider(error.to_string().into()))
	}
}

fn engine(id: &str) -> Option<&'static EngineSpec> {
	ENGINES.iter().find(|spec| spec.id == id)
}

fn push_unique(ordered: &mut SmallVec<&'static EngineSpec, 12>, spec: Option<&'static EngineSpec>) {
	if let Some(spec) = spec
		&& !ordered.iter().any(|candidate| candidate.id == spec.id)
	{
		ordered.push(spec);
	}
}

fn request_timeout(timeout_ms: u32) -> Duration {
	if timeout_ms == 0 {
		DEFAULT_TIMEOUT
	} else {
		Duration::from_millis(u64::from(timeout_ms)).min(MAX_TIMEOUT)
	}
}

fn unify(spec: &EngineSpec, result: EngineResult, query: &ParsedQuery) -> SearchResponse {
	let (answer, sources, citations) = match result.payload {
		EnginePayload::Raw { sources } => (Str::default(), sources, Vec::new()),
		EnginePayload::Synthesized { answer, sources, citations }
		| EnginePayload::Hybrid { answer, sources, citations } => (answer, sources, citations),
	};
	let filtered = apply_query_constraints(sources, query);
	let mut warnings = result.warnings;
	warnings.extend(
		filtered.relaxed.into_iter().map(|label| {
			Str::from(format!("no results matched `{label}`; the constraint was relaxed"))
		}),
	);
	SearchResponse::builder()
		.engine(Str::from(spec.id))
		.answer(answer)
		.sources(filtered.sources)
		.citations(citations)
		.search_queries(result.search_queries)
		.related(result.related)
		.warnings(warnings)
		.maybe_usage(result.usage)
		.maybe_cost(result.cost)
		.unsupported(result.unsupported)
		.props(result.props)
		.build()
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct QueryTerm {
	text:    Str,
	quoted:  bool,
	negated: bool,
	group:   Option<u16>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ParsedQuery {
	sites:          Vec<Str>,
	excluded_sites: Vec<Str>,
	in_url:         Vec<Str>,
	in_title:       Vec<Str>,
	filetypes:      Vec<Str>,
	after:          Option<Str>,
	before:         Option<Str>,
	terms:          Vec<QueryTerm>,
}

impl ParsedQuery {
	fn parse(raw: &str, request: &SearchRequest) -> Self {
		let mut parsed = Self {
			sites: request
				.allowed_domains
				.iter()
				.map(|site| normalize_site(site))
				.filter(|site| !site.is_empty())
				.collect(),
			excluded_sites: request
				.excluded_domains
				.iter()
				.map(|site| normalize_site(site))
				.filter(|site| !site.is_empty())
				.collect(),
			after: (!request.after.is_empty()).then(|| request.after.clone()),
			before: (!request.before.is_empty()).then(|| request.before.clone()),
			..Self::default()
		};
		let tokens = tokenize(raw);
		let mut group = 0_u16;
		let mut or_pending = false;
		for token in tokens {
			if token.text.eq_ignore_ascii_case("OR") {
				or_pending = true;
				continue;
			}
			let (negative, text) = token
				.text
				.strip_prefix('-')
				.map_or((false, token.text.as_str()), |text| (true, text));
			if let Some((name, value)) = text.split_once(':') {
				let value = value.trim_matches('"');
				if !value.is_empty() && parsed.push_directive(name, value, negative) {
					or_pending = false;
					continue;
				}
			}
			if text.is_empty() {
				continue;
			}
			let mut term = QueryTerm {
				text:    text.into(),
				quoted:  token.quoted,
				negated: negative,
				group:   None,
			};
			if or_pending && let Some(previous) = parsed.terms.last_mut() {
				if previous.group.is_none() {
					group = group.saturating_add(1);
					previous.group = Some(group);
				}
				term.group = previous.group;
			}
			or_pending = false;
			parsed.terms.push(term);
		}
		parsed
	}

	fn push_directive(&mut self, name: &str, value: &str, negative: bool) -> bool {
		match name.to_ascii_lowercase().as_str() {
			"site" | "domain" | "host" => {
				let value = normalize_site(value);
				if negative {
					self.excluded_sites.push(value);
				} else {
					self.sites.push(value);
				}
			},
			"inurl" | "url" => {
				if negative {
					return false;
				}
				self.in_url.push(value.to_ascii_lowercase().into());
			},
			"intitle" | "title" => {
				if negative {
					return false;
				}
				self.in_title.push(value.to_ascii_lowercase().into());
			},
			"filetype" | "ext" => {
				if negative {
					return false;
				}
				self
					.filetypes
					.push(value.trim_start_matches('.').to_ascii_lowercase().into());
			},
			"after" if iso_date(value) => self.after = Some(value.into()),
			"before" if iso_date(value) => self.before = Some(value.into()),
			_ => return false,
		}
		true
	}
}

#[derive(Clone, Debug, PartialEq)]
struct Token {
	text:   String,
	quoted: bool,
}

fn tokenize(raw: &str) -> Vec<Token> {
	let mut tokens = Vec::new();
	let mut chars = raw.chars().peekable();
	while chars.peek().is_some() {
		while chars
			.next_if(|character| character.is_whitespace())
			.is_some()
		{}
		if chars.peek().is_none() {
			break;
		}
		let mut text = String::new();
		let mut quoted = false;
		let mut quote = false;
		for character in chars.by_ref() {
			if character == '"' || character == '\u{201c}' || character == '\u{201d}' {
				quoted = true;
				quote = !quote;
				continue;
			}
			if character.is_whitespace() && !quote {
				break;
			}
			text.push(character);
		}
		if !text.is_empty() {
			tokens.push(Token { text, quoted });
		}
	}
	tokens
}

fn normalize_site(value: &str) -> Str {
	let value = value
		.trim()
		.trim_start_matches("https://")
		.trim_start_matches("http://")
		.trim_start_matches("www.")
		.trim_end_matches('/');
	value.to_ascii_lowercase().into()
}

fn iso_date(value: &str) -> bool {
	let bytes = value.as_bytes();
	bytes.len() == 10
		&& bytes[4] == b'-'
		&& bytes[7] == b'-'
		&& bytes
			.iter()
			.enumerate()
			.all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit())
}

struct FilterResult {
	sources: Vec<SearchSource>,
	relaxed: Vec<Str>,
}

fn apply_query_constraints(sources: Vec<SearchSource>, query: &ParsedQuery) -> FilterResult {
	let mut current = sources;
	let mut relaxed = Vec::new();
	if current.is_empty() {
		return FilterResult { sources: current, relaxed };
	}
	apply_dimension(
		&mut current,
		&mut relaxed,
		query
			.sites
			.iter()
			.map(|site| format!("site:{site}"))
			.collect::<Vec<_>>()
			.join(" OR "),
		!query.sites.is_empty(),
		|source| {
			query
				.sites
				.iter()
				.any(|site| matches_site(source.url.as_str(), site))
		},
	);
	apply_dimension(
		&mut current,
		&mut relaxed,
		query
			.excluded_sites
			.iter()
			.map(|site| format!("-site:{site}"))
			.collect::<Vec<_>>()
			.join(" "),
		!query.excluded_sites.is_empty(),
		|source| {
			!query
				.excluded_sites
				.iter()
				.any(|site| matches_site(source.url.as_str(), site))
		},
	);
	for value in &query.in_url {
		apply_dimension(&mut current, &mut relaxed, format!("inurl:{value}"), true, |source| {
			contains_ascii_case(source.url.as_str(), value)
		});
	}
	for value in &query.in_title {
		apply_dimension(&mut current, &mut relaxed, format!("intitle:{value}"), true, |source| {
			contains_ascii_case(source.title.as_str(), value)
		});
	}
	apply_dimension(
		&mut current,
		&mut relaxed,
		query
			.filetypes
			.iter()
			.map(|kind| format!("filetype:{kind}"))
			.collect::<Vec<_>>()
			.join(" OR "),
		!query.filetypes.is_empty(),
		|source| {
			query
				.filetypes
				.iter()
				.any(|kind| matches_filetype(source.url.as_str(), kind))
		},
	);
	if query.after.is_some() || query.before.is_some() {
		let label = [
			query.after.as_ref().map(|date| format!("after:{date}")),
			query.before.as_ref().map(|date| format!("before:{date}")),
		]
		.into_iter()
		.flatten()
		.collect::<Vec<_>>()
		.join(" ");
		apply_dimension(&mut current, &mut relaxed, label, true, |source| {
			let Some(date) = source.published_at.get(..10).filter(|date| iso_date(date)) else {
				return true;
			};
			query
				.after
				.as_ref()
				.is_none_or(|after| date >= after.as_str())
				&& query
					.before
					.as_ref()
					.is_none_or(|before| date < before.as_str())
		});
	}
	let mut groups: SmallVec<u16, 4> = SmallVec::new();
	for term in &query.terms {
		if let Some(group) = term.group
			&& !term.negated
			&& !groups.contains(&group)
		{
			groups.push(group);
			let alternatives: Vec<_> = query
				.terms
				.iter()
				.filter(|candidate| candidate.group == Some(group) && !candidate.negated)
				.collect();
			let label = alternatives
				.iter()
				.map(|candidate| candidate.text.as_str())
				.collect::<Vec<_>>()
				.join(" OR ");
			apply_dimension(&mut current, &mut relaxed, label, true, |source| {
				alternatives
					.iter()
					.any(|candidate| source_contains(source, candidate.text.as_str()))
			});
		} else if term.negated {
			apply_dimension(&mut current, &mut relaxed, format!("-{}", term.text), true, |source| {
				!source_contains(source, term.text.as_str())
			});
		} else if term.quoted && term.group.is_none() {
			apply_dimension(
				&mut current,
				&mut relaxed,
				format!("\"{}\"", term.text),
				true,
				|source| source_contains(source, term.text.as_str()),
			);
		}
	}
	FilterResult { sources: current, relaxed }
}

fn apply_dimension(
	current: &mut Vec<SearchSource>,
	relaxed: &mut Vec<Str>,
	label: String,
	enabled: bool,
	predicate: impl Fn(&SearchSource) -> bool,
) {
	if !enabled {
		return;
	}
	if current.iter().any(&predicate) {
		current.retain(predicate);
	} else {
		relaxed.push(label.into());
	}
}

fn source_contains(source: &SearchSource, needle: &str) -> bool {
	contains_ascii_case(source.title.as_str(), needle)
		|| contains_ascii_case(source.snippet.as_str(), needle)
		|| contains_ascii_case(source.url.as_str(), needle)
}

fn contains_ascii_case(haystack: &str, needle: &str) -> bool {
	needle.is_empty()
		|| haystack
			.as_bytes()
			.windows(needle.len())
			.any(|window| window.eq_ignore_ascii_case(needle.as_bytes()))
}

fn matches_site(url: &str, site: &str) -> bool {
	let normalized = url
		.split_once("://")
		.map_or(url, |(_, rest)| rest)
		.trim_start_matches("www.");
	let host = normalized.split(['/', '?', '#']).next().unwrap_or_default();
	let site_host = site.split('/').next().unwrap_or_default();
	host.eq_ignore_ascii_case(site_host)
		|| host
			.get(host.len().saturating_sub(site_host.len())..)
			.is_some_and(|suffix| {
				suffix.eq_ignore_ascii_case(site_host)
					&& host.as_bytes().get(host.len() - site_host.len() - 1) == Some(&b'.')
			})
}

fn matches_filetype(url: &str, kind: &str) -> bool {
	url.split(['?', '#'])
		.next()
		.and_then(|path| path.rsplit_once('.'))
		.is_some_and(|(_, extension)| extension.eq_ignore_ascii_case(kind))
}

#[cfg(test)]
mod tests {
	use std::collections::{HashMap, VecDeque};

	use parking_lot::Mutex;

	use super::*;

	#[derive(Default)]
	struct Credentials(HashMap<Str, Str>);

	impl SearchCredentials for Credentials {
		fn has_credential(&self, provider: &str) -> bool {
			self.0.contains_key(provider)
		}
	}

	struct Backend {
		results: Mutex<VecDeque<Result<EngineResult, SearchAttemptError>>>,
		calls:   Mutex<Vec<Str>>,
	}

	#[async_trait]
	impl SearchEngineBackend for Backend {
		async fn search(
			&self,
			engine: &'static EngineSpec,
			_request: &SearchRequest,
			_timeout: Duration,
		) -> Result<EngineResult, SearchAttemptError> {
			self.calls.lock().push(engine.id.into());
			self.results.lock().pop_front().unwrap()
		}
	}

	fn source(url: &str, title: &str) -> SearchSource {
		SearchSource::builder()
			.url(Str::new(url))
			.title(Str::new(title))
			.snippet(Str::new_static("snippet"))
			.published_at(Str::new_static("2025-01-02"))
			.author(Str::default())
			.build()
	}

	fn request(query: &str, engine: &str) -> SearchRequest {
		SearchRequest::builder()
			.query(Str::new(query))
			.limit(0)
			.after(Str::default())
			.before(Str::default())
			.allowed_domains(Vec::new())
			.excluded_domains(Vec::new())
			.country(Str::default())
			.language(Str::default())
			.engine(Str::new(engine))
			.timeout_ms(0)
			.props(Props::default())
			.build()
	}

	fn registry(
		keys: &[&str],
		results: Vec<Result<EngineResult, SearchAttemptError>>,
	) -> (SearchRegistry, Arc<Backend>) {
		let credentials = Credentials(
			keys
				.iter()
				.map(|key| ((*key).into(), "key".into()))
				.collect(),
		);
		let backend =
			Arc::new(Backend { results: Mutex::new(results.into()), calls: Mutex::new(Vec::new()) });
		(SearchRegistry::new(Arc::new(credentials), backend.clone()), backend)
	}

	fn raw() -> EngineResult {
		EngineResult::new(EnginePayload::Raw {
			sources: vec![source("https://example.com/a", "Example")],
		})
	}

	#[test]
	fn explicit_candidate_is_terminal() {
		let (registry, _) = registry(&["brave", "exa", "kagi"], Vec::new());
		let ids: Vec<_> = registry
			.with_configured_order(["kagi", "brave"])
			.candidates("exa")
			.unwrap()
			.into_iter()
			.map(|spec| spec.id)
			.collect();
		assert_eq!(ids, ["exa"]);
	}

	#[test]
	fn configured_order_precedes_default_order() {
		let (registry, _) = registry(&["brave", "exa"], Vec::new());
		let ids: Vec<_> = registry
			.with_configured_order(["brave"])
			.candidates("")
			.unwrap()
			.into_iter()
			.map(|spec| spec.id)
			.collect();
		assert_eq!(ids[..2], ["brave", "exa"]);
	}

	#[tokio::test]
	async fn rate_limit_falls_through_but_cancellation_does_not() {
		let limited = SearchProviderError {
			engine:  "exa".into(),
			kind:    SearchProviderErrorKind::Status(429),
			message: "limited".into(),
		};
		let (configured, backend) = registry(&["exa", "brave"], vec![Err(limited.into()), Ok(raw())]);
		let configured = configured.with_configured_order(["exa", "brave"]);
		let response = configured.execute(request("query", "")).await.unwrap();
		assert_eq!(response.engine, "brave");
		assert_eq!(*backend.calls.lock(), ["exa", "brave"]);

		let (registry, backend) =
			registry(&["exa", "brave"], vec![Err(SearchAttemptError::Cancelled), Ok(raw())]);
		let registry = registry.with_configured_order(["exa", "brave"]);
		assert!(matches!(
			registry.execute(request("query", "")).await,
			Err(SearchRegistryError::Cancelled)
		));
		assert_eq!(*backend.calls.lock(), ["exa"]);
	}

	#[tokio::test]
	async fn all_three_paradigms_map_to_one_shape() {
		let citation = SearchCitation::builder()
			.url(Str::new_static("https://example.com"))
			.title(Str::new_static("Example"))
			.cited_text(Str::new_static("evidence"))
			.build();
		let cases = [
			EnginePayload::Raw { sources: vec![source("https://example.com/raw", "Raw")] },
			EnginePayload::Synthesized {
				answer:    "answer".into(),
				sources:   vec![source("https://example.com/synth", "Synth")],
				citations: vec![citation.clone()],
			},
			EnginePayload::Hybrid {
				answer:    "answer".into(),
				sources:   vec![source("https://example.com/hybrid", "Hybrid")],
				citations: vec![citation],
			},
		];
		for payload in cases {
			let (registry, _) = registry(&["exa"], vec![Ok(EngineResult::new(payload.clone()))]);
			let response = registry.execute(request("query", "exa")).await.unwrap();
			assert_eq!(response.sources.len(), 1);
			match payload {
				EnginePayload::Raw { .. } => {
					assert!(response.answer.is_empty());
					assert_eq!(response.citations, [] as [omp_llm_types::SearchCitation; 0]);
				},
				EnginePayload::Synthesized { .. } | EnginePayload::Hybrid { .. } => {
					assert_eq!(response.answer, "answer");
					assert_eq!(response.citations.len(), 1);
				},
			}
		}
	}

	#[tokio::test]
	async fn lenient_constraints_enforce_matches_and_warn_on_zero() {
		let payload = EnginePayload::Raw {
			sources: vec![
				source("https://docs.example.com/guide.pdf", "Rust Guide"),
				source("https://other.test/page", "Other"),
			],
		};
		let (registry, _) = registry(&["brave"], vec![Ok(EngineResult::new(payload))]);
		let response = registry
			.execute(request("site:example.com intitle:missing filetype:pdf", "brave"))
			.await
			.unwrap();
		assert_eq!(response.sources.len(), 1);
		assert!(response.sources[0].url.contains("example.com"));
		assert_eq!(response.warnings.len(), 1);
		assert!(response.warnings[0].contains("intitle:missing"));
	}
}
