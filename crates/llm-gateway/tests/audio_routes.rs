//! Catalog registration coverage for production remote audio facets.

use std::convert::Infallible;

use bytes::Bytes;
use http::{Request, Response};
use http_body_util::Full;
use omp_llm_catalog::provider::load_builtin;
use omp_llm_egress::client::Body;
use omp_llm_gateway::audio_backends::register_production_audio_routes;
use omp_llm_tower::provider::ProviderRoute;
use tower::service_fn;

#[test]
fn every_advertised_audio_row_has_a_production_codec() {
	let providers = load_builtin().expect("built-in providers");
	let egress = service_fn(|_request: Request<Body>| async move {
		Ok::<_, Infallible>(Response::new(Full::new(Bytes::new())))
	});
	let facets =
		register_production_audio_routes(providers.values(), egress, |_| ProviderRoute::default())
			.expect("every advertised audio transport must have an adapter");
	assert_eq!(facets.speech_routes, 3, "Azure, LiteLLM, and OpenAI speech");
	assert_eq!(facets.transcription_routes, 4, "Azure, LiteLLM, OpenAI, and Groq transcription");
	assert!(facets.speak.is_some());
	assert!(facets.transcribe.is_some());
}
