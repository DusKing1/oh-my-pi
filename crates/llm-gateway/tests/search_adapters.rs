//! Integration coverage for production search provider adapters.

use std::{collections::VecDeque, sync::Arc, time::Duration};

use async_trait::async_trait;
use omp_core::SmolStr;
use omp_llm_egress::client::EgressClient;
use omp_llm_gateway::{
	search::{
		EnginePayload, EngineResult, EngineSpec, SearchAttemptError, SearchCredentials,
		SearchEngineBackend, SearchProviderError, SearchProviderErrorKind, SearchRegistry,
		SearchRegistryError,
	},
	search_backends::ProductionSearchBackend,
};
use omp_llm_types::{Props, SearchRequest, SearchSource};
use parking_lot::Mutex;
use wiremock::{
	Mock, MockServer, ResponseTemplate,
	matchers::{method, path},
};

struct Credentials;

impl SearchCredentials for Credentials {
	fn has_credential(&self, provider: &str) -> bool {
		matches!(provider, "exa" | "brave")
	}
}

struct ScriptedBackend {
	results: Mutex<VecDeque<Result<EngineResult, SearchAttemptError>>>,
	calls:   Mutex<Vec<&'static str>>,
}

#[async_trait]
impl SearchEngineBackend for ScriptedBackend {
	async fn search(
		&self,
		engine: &'static EngineSpec,
		_: &SearchRequest,
		_: Duration,
	) -> Result<EngineResult, SearchAttemptError> {
		self.calls.lock().push(engine.id);
		self.results.lock().pop_front().expect("scripted result")
	}
}

fn request(engine: &str) -> SearchRequest {
	SearchRequest::builder()
		.query(SmolStr::new_static("rust async"))
		.limit(10)
		.after(SmolStr::default())
		.before(SmolStr::default())
		.allowed_domains(Vec::new())
		.excluded_domains(Vec::new())
		.country(SmolStr::default())
		.language(SmolStr::default())
		.engine(SmolStr::from(engine))
		.timeout_ms(0)
		.props(Props::default())
		.build()
}

fn successful_result() -> EngineResult {
	EngineResult::new(EnginePayload::Raw {
		sources: vec![
			SearchSource::builder()
				.url(SmolStr::new_static("https://example.test"))
				.title(SmolStr::new_static("Example"))
				.snippet(SmolStr::default())
				.published_at(SmolStr::default())
				.author(SmolStr::default())
				.build(),
		],
	})
}

#[tokio::test]
async fn ordered_fallback_and_cancellation_hard_stop() {
	let limited = SearchProviderError {
		engine:  "exa".into(),
		kind:    SearchProviderErrorKind::Status(429),
		message: "limited".into(),
	};
	let backend = Arc::new(ScriptedBackend {
		results: Mutex::new(VecDeque::from([Err(limited.into()), Ok(successful_result())])),
		calls:   Mutex::new(Vec::new()),
	});
	let registry = SearchRegistry::new(Arc::new(Credentials), backend.clone())
		.with_configured_order(["exa", "brave"]);
	let response = registry.execute(request("")).await.expect("Brave fallback");
	assert_eq!(response.engine, "brave");
	assert_eq!(*backend.calls.lock(), ["exa", "brave"]);

	let backend = Arc::new(ScriptedBackend {
		results: Mutex::new(VecDeque::from([
			Err(SearchAttemptError::Cancelled),
			Ok(successful_result()),
		])),
		calls:   Mutex::new(Vec::new()),
	});
	let registry = SearchRegistry::new(Arc::new(Credentials), backend.clone())
		.with_configured_order(["exa", "brave"]);
	assert!(matches!(registry.execute(request("")).await, Err(SearchRegistryError::Cancelled)));
	assert_eq!(*backend.calls.lock(), ["exa"]);
}

#[tokio::test]
async fn brave_parser_preserves_valid_partial_results() {
	let server = MockServer::start().await;
	Mock::given(method("GET"))
		.and(path("/search"))
		.respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
			"web": {"results": [
				{"title":"missing URL and must be ignored"},
				{"url":"https://one.example/","title":"One","description":"first <b>snippet</b>"},
				{"url":"https://two.example/","title":"Two","extra_snippets":["second"]}
			]}
		})))
		.mount(&server)
		.await;
	let backend = ProductionSearchBackend::new(Arc::new(EgressClient::new(Duration::from_secs(5))))
		.with_endpoint("brave", format!("{}/search", server.uri()));
	let registry = SearchRegistry::new(Arc::new(Credentials), Arc::new(backend));
	let response = registry
		.execute(request("brave"))
		.await
		.expect("parsed Brave response");
	assert_eq!(response.sources.len(), 2);
	assert_eq!(response.sources[0].title, "One");
	assert_eq!(response.sources[0].snippet, "first snippet");
	assert_eq!(response.sources[1].url, "https://two.example/");
}
