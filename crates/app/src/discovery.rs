//! Registration of the provider-owned model-discovery protocols.
//!
//! Every `omp-llm-*` crate that speaks a provider's own listing protocol
//! implements
//! [`DiscoveryProtocol`](omp_llm_catalog::discovery::DiscoveryProtocol) next to
//! that protocol's wire code and exports it as a `DISCOVERY` static.
//! This module is the single place those statics are wired into the runtime.

use std::sync::Arc;

use omp_llm_catalog::discovery::{Discovery, DiscoveryHttp};

/// Builds the discovery stack over `http`, registering every protocol
/// shipped with the CLI.
///
/// Dispatch is keyed by
/// [`TransportId`](omp_llm_catalog::provider::TransportId), so a new
/// `providers.toml` row naming an already-registered transport is discovered
/// without touching this list. Add an entry here only when a crate introduces a
/// genuinely new wire protocol.
pub(crate) fn register(http: Arc<dyn DiscoveryHttp>) -> Discovery {
	Discovery::new(http)
		.with(&omp_llm_openai::discovery::DISCOVERY)
		.with(&omp_llm_google::discovery::DISCOVERY)
		.with(&omp_llm_cursor::discovery::DISCOVERY)
		.with(&omp_llm_devin::discovery::DISCOVERY)
		.with(&omp_llm_gitlab::discovery::DISCOVERY)
		.with(&omp_llm_fm::discovery::DISCOVERY)
		.with(&omp_llm_ollama::discovery::DISCOVERY)
		.with(&omp_llm_bedrock::discovery::DISCOVERY)
}

#[cfg(test)]
mod tests {
	use async_trait::async_trait;
	use bytes::Bytes;
	use http::{Method, Request, Version};
	use omp_llm_catalog::{
		codex::{CODEX_CLIENT_VERSION, CODEX_ORIGINATOR},
		discovery::{Account, Error, HttpResponse, SealedBody},
		provider::{BUILTIN_PROVIDERS_TOML, DiscoveryKind, ProviderEntry, load_providers},
	};
	use omp_llm_egress::auth_inject::AwsSigV4Context;
	use parking_lot::Mutex;

	use super::*;

	/// `RDiscoveryHttpon` coverage is a pure question about the provider table,
	/// so the stack never dispatches a request.
	struct NeverCalled;

	#[async_trait]
	impl DiscoveryHttp for NeverCalled {
		async fn execute(
			&self,
			_provider: &ProviderEntry,
			_account: &Account,
			_request: Request<Bytes>,
		) -> Result<HttpResponse, Error> {
			unreachable!("registration coverage never dispatches")
		}
	}

	#[test]
	fn every_specialized_provider_row_has_a_registered_protocol() {
		let providers = load_providers(BUILTIN_PROVIDERS_TOML).expect("built-in providers");
		let discovery = register(Arc::new(NeverCalled));
		let mut covered = 0usize;
		for provider in providers.values().filter(|provider| {
			provider
				.discovery
				.as_ref()
				.is_some_and(|spec| spec.kind == DiscoveryKind::Specialized)
		}) {
			assert!(
				discovery.serves(provider.transport),
				"provider {} declares specialized discovery but no protocol is registered for \
				 transport {:?}",
				provider.id,
				provider.transport
			);
			covered += 1;
		}
		assert!(covered >= 10, "expected every shipped specialized row to be covered, saw {covered}");
	}

	/// One outbound discovery request, captured before credential injection.
	#[derive(Debug)]
	struct Seen {
		method:  Method,
		uri:     String,
		version: Version,
		headers: Vec<(String, String)>,
		body:    Bytes,
		sealed:  bool,
		aws:     Option<AwsSigV4Context>,
	}

	impl Seen {
		fn header(&self, name: &str) -> Option<&str> {
			self
				.headers
				.iter()
				.find(|(key, _)| key == name)
				.map(|(_, value)| value.as_str())
		}
	}

	/// Answers every request with one canned reply and records what was sent.
	struct Recorder {
		status: u16,
		body:   &'static str,
		seen:   Mutex<Vec<Seen>>,
	}

	impl Recorder {
		fn new(status: u16, body: &'static str) -> Arc<Self> {
			Arc::new(Self { status, body, seen: Mutex::new(Vec::new()) })
		}
	}

	#[async_trait]
	impl DiscoveryHttp for Recorder {
		async fn execute(
			&self,
			_provider: &ProviderEntry,
			_account: &Account,
			request: Request<Bytes>,
		) -> Result<HttpResponse, Error> {
			self.seen.lock().push(Seen {
				method:  request.method().clone(),
				uri:     request.uri().to_string(),
				version: request.version(),
				headers: request
					.headers()
					.iter()
					.map(|(name, value)| {
						(name.as_str().to_owned(), value.to_str().unwrap_or_default().to_owned())
					})
					.collect(),
				sealed:  request.extensions().get::<SealedBody>().is_some(),
				aws:     request.extensions().get::<AwsSigV4Context>().cloned(),
				body:    request.into_body(),
			});
			Ok(HttpResponse::new(self.status, Bytes::from_static(self.body.as_bytes())))
		}
	}

	fn entry(id: &str) -> ProviderEntry {
		load_providers(BUILTIN_PROVIDERS_TOML)
			.expect("built-in providers")
			.get(id)
			.unwrap_or_else(|| panic!("{id} must exist in providers.toml"))
			.clone()
	}

	/// Drives one provider's registered protocol and returns what it sent.
	async fn drive(id: &str, account: &Account, http: Arc<Recorder>) -> Vec<Seen> {
		let _ = register(http.clone()).discover(&entry(id), account).await;
		std::mem::take(&mut *http.seen.lock())
	}

	fn account() -> Account {
		Account::new("7", "fixture")
	}

	#[tokio::test]
	async fn cursor_posts_protobuf_over_http2() {
		let sent = drive("cursor", &account(), Recorder::new(500, "")).await;
		let [request] = sent.as_slice() else {
			panic!("Cursor discovery makes exactly one call, saw {}", sent.len());
		};
		assert_eq!(request.method, Method::POST);
		assert_eq!(request.uri, "https://api2.cursor.sh/agent.v1.AgentService/GetUsableModels");
		// Connect-over-proto needs trailers, so the protocol pins HTTP/2 itself
		// instead of the transport special-casing the provider id.
		assert_eq!(request.version, Version::HTTP_2);
		assert_eq!(request.header("content-type"), Some("application/proto"));
		assert_eq!(request.header("te"), Some("trailers"));
		assert_eq!(request.header("x-cursor-client-type"), Some("cli"));
		// Unlike Devin, Cursor's protocol owns its own body: an empty
		// `GetUsableModels` message, which protobuf encodes to zero bytes.
		assert!(!request.sealed);
		assert_eq!(request.body, omp_llm_cursor::model_discovery_request());
	}

	#[tokio::test]
	async fn codex_falls_back_to_the_bare_models_endpoint() {
		let sent = drive("openai-codex", &account(), Recorder::new(404, "")).await;
		let uris: Vec<&str> = sent.iter().map(|request| request.uri.as_str()).collect();
		assert_eq!(uris.len(), 2, "both Codex endpoints are attempted: {uris:?}");
		assert!(uris[0].contains("/codex/models?client_version="), "{uris:?}");
		assert!(
			uris[1].contains("/models?client_version=") && !uris[1].contains("/codex/"),
			"{uris:?}"
		);
		// Upstream branding is stripped on port, so the pinned originator is ours.
		assert_eq!(sent[0].header("originator"), Some(CODEX_ORIGINATOR));
		assert_eq!(sent[0].header("version"), Some(CODEX_CLIENT_VERSION));
	}

	#[tokio::test]
	async fn devin_defers_its_body_to_the_credential_boundary() {
		let sent = drive("devin", &account(), Recorder::new(500, "")).await;
		let [request] = sent.as_slice() else {
			panic!("Devin discovery makes exactly one call, saw {}", sent.len());
		};
		assert_eq!(request.method, Method::POST);
		assert_eq!(
			request.uri,
			"https://server.codeium.com/exa.api_server_pb.ApiServerService/GetCliModelConfigs"
		);
		assert!(request.sealed, "Devin embeds its credential in the body");
		assert!(request.body.is_empty(), "the protocol must not attempt to build the sealed body");
	}

	#[tokio::test]
	async fn cloud_code_assist_walks_its_fallback_endpoints() {
		let sent = drive("google-antigravity", &account(), Recorder::new(503, "")).await;
		let uris: Vec<&str> = sent.iter().map(|request| request.uri.as_str()).collect();
		assert_eq!(
			uris,
			[
				"https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels",
				"https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:fetchAvailableModels",
			],
			"the fallback_base_urls column must be walked in order"
		);
		assert_eq!(sent[0].body.as_ref(), b"{}");
	}

	#[tokio::test]
	async fn ollama_cloud_lists_native_tags() {
		let http = Recorder::new(200, r#"{"models":[{"name":"kimi-k2:1t","model":"kimi-k2:1t"}]}"#);
		let cards = register(http.clone())
			.discover(&entry("ollama-cloud"), &account())
			.await
			.expect("canned tags payload");
		let sent = std::mem::take(&mut *http.seen.lock());
		assert_eq!(sent[0].method, Method::GET);
		assert_eq!(sent[0].uri, "https://ollama.com/api/tags");
		assert_eq!(cards.len(), 1);
		assert_eq!(cards[0].id.as_str(), "ollama-cloud/kimi-k2:1t");
	}

	#[tokio::test]
	async fn gitlab_duo_queries_the_account_namespace_first() {
		let http = Recorder::new(
			200,
			r#"{"data":{"aiChatAvailableModels":{
				"defaultModel":{"name":"Sonnet","ref":"claude_sonnet_4_6_vertex"},
				"selectableModels":[],"pinnedModel":null}}}"#,
		);
		let account = Account::new("7", "fixture").with_scope(Some("42".into()), None);
		let cards = register(http.clone())
			.discover(&entry("gitlab-duo-agent"), &account)
			.await
			.expect("canned GraphQL payload");
		let sent = std::mem::take(&mut *http.seen.lock());
		assert_eq!(sent.len(), 1, "a scoped account skips the group crawl: {sent:?}");
		assert_eq!(sent[0].method, Method::POST);
		assert_eq!(sent[0].uri, "https://gitlab.com/api/graphql");
		let body = String::from_utf8_lossy(&sent[0].body);
		assert!(body.contains("gid://gitlab/Group/42"), "numeric namespaces are wrapped: {body}");
		assert_eq!(cards.len(), 1);
		assert_eq!(cards[0].model.as_str(), "claude_sonnet_4_6_vertex");
	}

	#[tokio::test]
	async fn bedrock_lists_from_the_control_plane_with_signing_context() {
		let http = Recorder::new(
			200,
			r#"{"modelSummaries":[{"modelId":"anthropic.claude-3-5-sonnet-20241022-v2:0",
				"modelName":"Claude 3.5 Sonnet v2","providerName":"Anthropic",
				"inputModalities":["TEXT","IMAGE"],"outputModalities":["TEXT"],
				"responseStreamingSupported":true,
				"inferenceTypesSupported":["ON_DEMAND"],
				"modelLifecycle":{"status":"ACTIVE"}}]}"#,
		);
		let account = Account::new("7", "aws").with_region(Some("eu-central-1".into()));
		let cards = register(http.clone())
			.discover(&entry("amazon-bedrock"), &account)
			.await
			.expect("canned ListFoundationModels payload");
		let sent = std::mem::take(&mut *http.seen.lock());
		let [request] = sent.as_slice() else {
			panic!("ListFoundationModels is a single unpaginated call, saw {}", sent.len());
		};
		assert_eq!(request.method, Method::GET);
		// The control plane is a different service from the `bedrock-runtime`
		// host the provider row carries for inference.
		assert_eq!(request.uri, "https://bedrock.eu-central-1.amazonaws.com/foundation-models");
		// The protocol never signs: it attaches non-secret scope and the broker
		// signs during credential redemption.
		let aws = request
			.aws
			.as_ref()
			.expect("SigV4 context must reach the broker");
		assert_eq!(aws.service, "bedrock");
		assert_eq!(aws.region, "eu-central-1");
		assert_eq!(cards.len(), 1);
		assert_eq!(cards[0].id.as_str(), "amazon-bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0");
	}
}
