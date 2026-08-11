//! Remote embedding route and provider behavior.

use std::{collections::BTreeMap, convert::Infallible, future::pending, sync::Arc};

use bytes::Bytes;
use http::{Request, Response, header};
use http_body_util::{BodyExt, Full};
use omp_core::Str;
use omp_llm_catalog::{
	compat::Compat,
	models::{
		Availability, Modality, ModelCard, ModelCatalog, Price, PriceUnit, Source, embedded_catalog,
	},
	provider::{
		AuthSpec, Facet as CatalogFacet, ProviderCatalog, ProviderEntry, TransportId, load_builtin,
	},
};
use omp_llm_egress::{auth_inject::AuthContext, client::Body};
use omp_llm_tower::{
	provider::ProviderRoute,
	stack::routing::{EmbedRouter, RemoteEmbed, RemoteEmbedBuildError, remote_embed_route},
};
use omp_llm_types::{Accuracy, Embed, EmbedRequest, Error, Props};
use parking_lot::Mutex;
use serde_json::{Value, json};
use smallvec::{SmallVec, smallvec};
use tokio::sync::oneshot;
use tower::service_fn;

fn provider(id: &str, transport: TransportId, facet: CatalogFacet) -> ProviderEntry {
	ProviderEntry::builder()
		.id(Str::from(id))
		.transport(transport)
		.base_url(Str::from("https://example.test/v1"))
		.fallback_base_urls(SmallVec::new())
		.auth(AuthSpec::default())
		.facets(smallvec![facet])
		.headers(BTreeMap::new())
		.compat(Compat::default())
		.build()
}

fn model(provider: &str, wire_model: &str, pricing: SmallVec<Price, 4>) -> ModelCard {
	ModelCard::builder()
		.id(Str::from(format!("{provider}/{wire_model}")))
		.provider(Str::from(provider))
		.model(Str::from(wire_model))
		.name(Str::from("Embedding model"))
		.family(Str::from("embedding"))
		.facets(smallvec![CatalogFacet::Embeddings])
		.inputs(smallvec![Modality::Text])
		.outputs(SmallVec::new())
		.reasoning(false)
		.efforts(SmallVec::new())
		.context_window(8_192)
		.max_output_tokens(0)
		.pricing(pricing)
		.availability(Availability::Available)
		.source(Source::Bundled)
		.blocked_until_ms(0)
		.deprecated(false)
		.updated_at_ms(0)
		.props(Props::default())
		.effort_routing(BTreeMap::new())
		.build()
}

fn router(
	model: ModelCard,
	provider: ProviderEntry,
	route: omp_llm_tower::stack::routing::EmbedRoute,
) -> EmbedRouter {
	let provider_id = provider.id.clone();
	let mut providers = ProviderCatalog::new();
	providers.insert(provider.id.clone(), provider);
	let mut router = EmbedRouter::new(Arc::new(ModelCatalog::new(vec![model])), Arc::new(providers));
	router.insert_route(provider_id, route);
	router
}

#[tokio::test]
async fn openai_batches_restores_indexes_and_accounts_exact_usage_and_cost() {
	let captured = Arc::new(Mutex::new(Vec::<Value>::new()));
	let captured_requests = Arc::clone(&captured);
	let service = service_fn(move |request: Request<Body>| {
		let captured = Arc::clone(&captured_requests);
		async move {
			assert!(request.headers().get(header::AUTHORIZATION).is_none());
			assert_eq!(
				request
					.extensions()
					.get::<AuthContext>()
					.map(AuthContext::provider),
				Some("openai")
			);
			let value: Value =
				serde_json::from_slice(&request.into_body().collect().await.unwrap().to_bytes())
					.unwrap();
			let input = value["input"].as_array().unwrap();
			let input_len = input.len();
			let mut data = input
				.iter()
				.enumerate()
				.map(|(index, text)| {
					json!({
						"object": "embedding",
						"index": index,
						"embedding": [text.as_str().unwrap().parse::<f32>().unwrap()]
					})
				})
				.collect::<Vec<_>>();
			data.reverse();
			captured.lock().push(value);
			let body = json!({
				"object": "list",
				"data": data,
				"usage": { "prompt_tokens": input_len, "total_tokens": input_len }
			});
			Ok::<_, Infallible>(Response::new(Full::new(Bytes::from(body.to_string()))))
		}
	});
	let provider = provider("openai", TransportId::OpenAiResponses, CatalogFacet::Embeddings);
	let route = remote_embed_route(provider.clone(), ProviderRoute::default(), service).unwrap();
	let price = Price::builder()
		.unit(PriceUnit::MtokInput)
		.nanos_usd(1_000_000)
		.build();
	let router =
		router(model("openai", "text-embedding-3-small", smallvec![price]), provider, route);
	let response = router
		.embed(
			EmbedRequest::builder()
				.model(Str::from("openai/text-embedding-3-small"))
				.texts(
					(0..2_049)
						.map(|index| Str::from(index.to_string()))
						.collect(),
				)
				.dimensions(1)
				.props(Props::default())
				.build(),
		)
		.await
		.unwrap();

	let requests = captured.lock();
	assert_eq!(
		requests
			.iter()
			.map(|body| body["input"].as_array().unwrap().len())
			.collect::<Vec<_>>(),
		[2_048, 1]
	);
	assert!(
		requests
			.iter()
			.all(|body| body["model"] == "text-embedding-3-small")
	);
	assert!(requests.iter().all(|body| body["dimensions"] == 1));
	assert_eq!(response.vectors.first().unwrap().values, [0.0]);
	assert_eq!(response.vectors.last().unwrap().values, [2_048.0]);
	let usage = response.usage.unwrap();
	assert_eq!((usage.input_tokens, usage.accuracy), (2_049, Accuracy::Exact));
	assert_eq!(usage.detail.get_ns("omp", "cost_nanos_usd"), Some(&Value::from(2_049)));
	assert_eq!(usage.detail.get_ns("omp", "cost_estimated"), Some(&Value::Bool(true)));
}

#[tokio::test]
async fn google_batch_wire_preserves_native_order_and_estimates_missing_usage() {
	let captured = Arc::new(Mutex::new(None::<(String, Value)>));
	let inspect = Arc::clone(&captured);
	let service = service_fn(move |request: Request<Body>| {
		let inspect = Arc::clone(&inspect);
		async move {
			let uri = request.uri().to_string();
			let body: Value =
				serde_json::from_slice(&request.into_body().collect().await.unwrap().to_bytes())
					.unwrap();
			*inspect.lock() = Some((uri, body));
			Ok::<_, Infallible>(Response::new(Full::new(Bytes::from_static(
				br#"{"embeddings":[{"values":[1.0,2.0]},{"values":[3.0,4.0]}]}"#,
			))))
		}
	});
	let provider = provider("google", TransportId::GoogleGenAi, CatalogFacet::Embeddings);
	let route = remote_embed_route(provider.clone(), ProviderRoute::default(), service).unwrap();
	let router = router(model("google", "gemini-embedding-001", SmallVec::new()), provider, route);
	let response = router
		.embed(
			EmbedRequest::builder()
				.model(Str::from("google/gemini-embedding-001"))
				.texts(vec![Str::from("abcd"), Str::from("abcdefgh")])
				.dimensions(2)
				.props(Props::default())
				.build(),
		)
		.await
		.unwrap();
	let (uri, body) = captured.lock().clone().unwrap();
	assert_eq!(uri, "https://example.test/v1/models/gemini-embedding-001:batchEmbedContents");
	assert_eq!(body["requests"][0]["model"], "models/gemini-embedding-001");
	assert_eq!(body["requests"][0]["outputDimensionality"], 2);
	assert_eq!(response.vectors[1].values, [3.0, 4.0]);
	assert_eq!(
		response
			.usage
			.map(|usage| (usage.input_tokens, usage.accuracy)),
		Some((3, Accuracy::Estimated))
	);
}

#[tokio::test]
async fn vertex_predict_wire_uses_route_metadata_and_exact_statistics() {
	let captured = Arc::new(Mutex::new(None::<(String, Value)>));
	let inspect = Arc::clone(&captured);
	let service = service_fn(move |request: Request<Body>| {
		let inspect = Arc::clone(&inspect);
		async move {
			assert_eq!(
				request
					.extensions()
					.get::<AuthContext>()
					.map(AuthContext::provider),
				Some("google-vertex")
			);
			let uri = request.uri().to_string();
			let body: Value =
				serde_json::from_slice(&request.into_body().collect().await.unwrap().to_bytes())
					.unwrap();
			*inspect.lock() = Some((uri, body));
			Ok::<_, Infallible>(Response::new(Full::new(Bytes::from_static(
				br#"{
				"predictions":[
					{"embeddings":{"values":[1.0,2.0],"statistics":{"token_count":3}}},
					{"embeddings":{"values":[3.0,4.0],"statistics":{"token_count":5}}}
				]
			}"#,
			))))
		}
	});
	let mut provider =
		provider("google-vertex", TransportId::GoogleVertex, CatalogFacet::Embeddings);
	provider.base_url = Str::from("https://{location}-aiplatform.googleapis.com/v1");
	let route_meta = ProviderRoute {
		project: Str::from("project-a"),
		region: Str::from("us-central1"),
		..ProviderRoute::default()
	};
	let route = remote_embed_route(provider.clone(), route_meta, service).unwrap();
	let router =
		router(model("google-vertex", "text-embedding-005", SmallVec::new()), provider, route);
	let response = router
		.embed(
			EmbedRequest::builder()
				.model(Str::from("google-vertex/text-embedding-005"))
				.texts(vec![Str::from("first"), Str::from("second")])
				.dimensions(2)
				.props(Props::default())
				.build(),
		)
		.await
		.unwrap();
	let (uri, body) = captured.lock().clone().unwrap();
	assert_eq!(uri, "https://us-central1-aiplatform.googleapis.com/v1/projects/project-a/locations/us-central1/publishers/google/models/text-embedding-005:predict");
	assert_eq!(body["instances"][0]["content"], "first");
	assert_eq!(body["parameters"]["outputDimensionality"], 2);
	assert_eq!(
		response
			.usage
			.map(|usage| (usage.input_tokens, usage.accuracy)),
		Some((8, Accuracy::Exact))
	);
}

#[tokio::test]
async fn status_errors_are_structured_and_fixed_dimension_providers_reject_before_egress() {
	let calls = Arc::new(Mutex::new(0usize));
	let observed = Arc::clone(&calls);
	let service = service_fn(move |_request: Request<Body>| {
		*observed.lock() += 1;
		async {
			let mut response = Response::new(Full::new(Bytes::from_static(
				br#"{"error":{"message":"rate limited"}}"#,
			)));
			*response.status_mut() = http::StatusCode::TOO_MANY_REQUESTS;
			Ok::<_, Infallible>(response)
		}
	});
	let provider = provider("mistral", TransportId::OpenAiChat, CatalogFacet::Embeddings);
	let remote = RemoteEmbed::new(provider, ProviderRoute::default(), service).unwrap();
	let unsupported = remote
		.embed(
			EmbedRequest::builder()
				.model(Str::from("mistral-embed"))
				.texts(vec![Str::from("text")])
				.dimensions(128)
				.props(Props::default())
				.build(),
		)
		.await
		.unwrap_err();
	assert!(matches!(unsupported, Error::Unsupported(records) if records[0].what == "dimensions"));
	assert_eq!(*calls.lock(), 0);

	let failed = remote
		.embed(
			EmbedRequest::builder()
				.model(Str::from("mistral-embed"))
				.texts(vec![Str::from("text")])
				.props(Props::default())
				.build(),
		)
		.await
		.unwrap_err();
	assert!(
		matches!(failed, Error::Provider(message) if message.contains("HTTP 429") && message.contains("rate limited"))
	);
}

struct DropSignal(Option<oneshot::Sender<()>>);
impl Drop for DropSignal {
	fn drop(&mut self) {
		if let Some(signal) = self.0.take() {
			let _ = signal.send(());
		}
	}
}

#[tokio::test]
async fn dropping_embedding_future_cancels_egress_without_fallback() {
	let (started_tx, started_rx) = oneshot::channel();
	let (dropped_tx, dropped_rx) = oneshot::channel();
	let started_tx = Arc::new(Mutex::new(Some(started_tx)));
	let dropped_tx = Arc::new(Mutex::new(Some(dropped_tx)));
	let service = service_fn(move |_request: Request<Body>| {
		let started = started_tx.lock().take();
		let dropped = dropped_tx.lock().take();
		async move {
			if let Some(started) = started {
				let _ = started.send(());
			}
			let _guard = DropSignal(dropped);
			pending::<Result<Response<Full<Bytes>>, Infallible>>().await
		}
	});
	let provider = provider("openai", TransportId::OpenAiResponses, CatalogFacet::Embeddings);
	let remote = Arc::new(RemoteEmbed::new(provider, ProviderRoute::default(), service).unwrap());
	let task = tokio::spawn({
		let remote = Arc::clone(&remote);
		async move {
			remote
				.embed(
					EmbedRequest::builder()
						.model(Str::from("text-embedding-3-small"))
						.texts(vec![Str::from("cancel")])
						.props(Props::default())
						.build(),
				)
				.await
		}
	});
	started_rx.await.unwrap();
	task.abort();
	let _ = task.await;
	tokio::time::timeout(std::time::Duration::from_secs(1), dropped_rx)
		.await
		.unwrap()
		.unwrap();
}

#[test]
fn every_advertised_remote_embedding_route_constructs_and_rerank_stays_unadvertised() {
	let providers = load_builtin().unwrap();
	for provider in providers
		.values()
		.filter(|provider| provider.facets.contains(&CatalogFacet::Embeddings))
	{
		let route = ProviderRoute {
			project: Str::from("project"),
			region: Str::from("us-central1"),
			account: Str::from("account"),
			gateway: Str::from("gateway"),
			..ProviderRoute::default()
		};
		let service = service_fn(|_request: Request<Body>| async {
			Ok::<_, Infallible>(Response::new(Full::new(Bytes::new())))
		});
		let built = remote_embed_route(provider.clone(), route, service)
			.unwrap_or_else(|error| panic!("{} failed to construct: {error}", provider.id));
		let expected = match provider.id.as_str() {
			"openai" | "litellm" | "vllm" => (2_048, true),
			"mistral" => (32, false),
			"google" => (100, true),
			"google-vertex" => (250, true),
			"fireworks" | "together" | "nvidia" | "ollama" | "lm-studio" => (1, false),
			id => panic!("advertised embedding provider has no expected policy: {id}"),
		};
		assert_eq!(
			(built.max_batch_size, built.supports_dimensions),
			expected,
			"{} policy",
			provider.id
		);
	}
	assert!(
		providers
			.values()
			.all(|provider| !provider.facets.contains(&CatalogFacet::Rerank))
	);
	let models = embedded_catalog();
	assert!(
		models
			.get("fireworks", "qwen3-embedding-8b")
			.expect("Pi Fireworks embedding row")
			.facets
			.contains(&CatalogFacet::Embeddings)
	);
	assert!(
		models
			.get("fireworks", "qwen3-reranker-8b")
			.expect("Pi Fireworks reranker row")
			.facets
			.is_empty(),
		"Pi has no rerank dispatch, so the generic OpenAI tag must not become chat"
	);
	assert!(
		models
			.get("nvidia", "baai/bge-m3")
			.expect("Pi NVIDIA embedding row")
			.facets
			.contains(&CatalogFacet::Embeddings)
	);

	let fake = provider("anthropic", TransportId::AnthropicMessages, CatalogFacet::Embeddings);
	let error = match RemoteEmbed::new(fake, ProviderRoute::default(), ()) {
		Ok(_) => panic!("Anthropic must not construct an embedding adapter"),
		Err(error) => error,
	};
	assert_eq!(error, RemoteEmbedBuildError::UnsupportedTransport(TransportId::AnthropicMessages));
}
