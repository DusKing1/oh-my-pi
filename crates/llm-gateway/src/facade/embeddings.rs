//! OpenAI-compatible embeddings facade.

use std::{fmt::Display, sync::Arc};

use bytes::Bytes;
use http::{Request, StatusCode};
use hyper::body::Body;
use omp_core::Str;
use omp_llm_types::{EmbedRequest, Props};
use serde::{Deserialize, Serialize};

use super::{
	FacadeError, FacadeResponse, FacadeState, Vendor, error_response, json_response, read_json,
};

#[derive(Deserialize)]
struct EmbeddingsRequest {
	model:      Str,
	input:      EmbeddingInput,
	#[serde(default)]
	dimensions: Option<u32>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
	One(Str),
	Many(Vec<Str>),
}

#[derive(Serialize)]
struct EmbeddingsResponse {
	object: &'static str,
	data:   Vec<EmbeddingData>,
	model:  Str,
	usage:  EmbeddingUsage,
}

#[derive(Serialize)]
struct EmbeddingData {
	object:    &'static str,
	embedding: Vec<f32>,
	index:     usize,
}

#[derive(Serialize)]
struct EmbeddingUsage {
	prompt_tokens: u64,
	total_tokens:  u64,
}

pub(crate) async fn handle<B>(request: Request<B>, state: Arc<FacadeState>) -> FacadeResponse
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
{
	let wire: EmbeddingsRequest = match read_json(request, Vendor::OpenAi).await {
		Ok(wire) => wire,
		Err(response) => return *response,
	};
	let texts = match wire.input {
		EmbeddingInput::One(text) => vec![text],
		EmbeddingInput::Many(texts) if texts.is_empty() => {
			return error_response(
				Vendor::OpenAi,
				FacadeError::Invalid(Str::new("input must contain at least one string")),
			);
		},
		EmbeddingInput::Many(texts) => texts,
	};
	let Some(embed) = &state.facets.embed else {
		return error_response(
			Vendor::OpenAi,
			FacadeError::Invalid(Str::new("embedding is not available")),
		);
	};
	let request = EmbedRequest::builder()
		.model(wire.model.clone())
		.texts(texts)
		.maybe_dimensions(wire.dimensions)
		.props(Props::default())
		.build();
	let canonical = match embed.embed(request).await {
		Ok(response) => response,
		Err(error) => return error_response(Vendor::OpenAi, FacadeError::Facet(error)),
	};
	let usage = canonical
		.usage
		.as_ref()
		.map_or(0, |usage| usage.input_tokens);
	let data = canonical
		.vectors
		.into_iter()
		.enumerate()
		.map(|(index, vector)| EmbeddingData { object: "embedding", embedding: vector.values, index })
		.collect();
	json_response(StatusCode::OK, &EmbeddingsResponse {
		object: "list",
		data,
		model: wire.model,
		usage: EmbeddingUsage { prompt_tokens: usage, total_tokens: usage },
	})
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use async_trait::async_trait;
	use http_body_util::{BodyExt, Full};
	use omp_llm_catalog::{
		models::Availability,
		registry::{CredentialView, Registry},
	};
	use omp_llm_types::{
		Accuracy, EmbedResponse, EmbeddingVector, Props, Usage,
		facet::{Embed, Error, Facets},
	};
	use omp_storage::blob::BlobStore;

	use super::*;

	struct Credentials;

	impl CredentialView for Credentials {
		fn availability(&self, _provider: &str) -> Availability {
			Availability::Available
		}
	}

	struct FakeEmbed;

	#[async_trait]
	impl Embed for FakeEmbed {
		async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, Error> {
			assert_eq!(request.texts, ["first", "second"]);
			Ok(EmbedResponse::builder()
				.vectors(vec![
					EmbeddingVector::builder().values(vec![1.0, 2.0]).build(),
					EmbeddingVector::builder().values(vec![3.0, 4.0]).build(),
				])
				.usage(
					Usage::builder()
						.input_tokens(7)
						.output_tokens(0)
						.cache_read_tokens(0)
						.cache_write_tokens(0)
						.accuracy(Accuracy::Exact)
						.detail(Props::default())
						.build(),
				)
				.build())
		}
	}

	#[tokio::test]
	async fn embeds_array_input_in_order() {
		let facets = Facets { embed: Some(Arc::new(FakeEmbed)), ..Facets::default() };
		let directory = tempfile::tempdir().expect("temporary directory");
		let state = Arc::new(FacadeState {
			facets:   Arc::new(facets),
			registry: Arc::new(parking_lot::RwLock::new(Registry::from_cards(
				&[],
				Arc::new(Credentials),
			))),
			blobs:    Arc::new(BlobStore::open(directory.path()).expect("blob store")),
			auth:     super::super::FacadeAuth::new("token"),
			config:   super::super::FacadeConfig::default(),
		});
		let request = Request::post("/v1/embeddings")
			.body(Full::<Bytes>::new(Bytes::from_static(
				br#"{"model":"embed","input":["first","second"]}"#,
			)))
			.expect("request");
		let response = handle(request, state).await;
		assert_eq!(response.status(), StatusCode::OK);
		let body = response
			.into_body()
			.collect()
			.await
			.expect("infallible")
			.to_bytes();
		let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON response");
		assert_eq!(value["object"], "list");
		assert_eq!(value["data"][1]["embedding"], serde_json::json!([3.0, 4.0]));
		assert_eq!(value["usage"]["total_tokens"], 7);
	}
}
