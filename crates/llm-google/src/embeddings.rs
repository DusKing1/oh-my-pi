//! Google Generative Language and Vertex text-embedding wire codecs.

use bytes::Bytes;
use omp_core::{SmolStr, format_smol};
use omp_llm_types::{
	Accuracy, EmbedRequest, EmbedResponse, EmbeddingVector, Error, Props, Unsupported,
	UnsupportedAction, Usage,
};
use serde_json::{Map, Value};

/// Google embedding endpoint family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GoogleEmbeddingVariant {
	/// Public Generative Language `batchEmbedContents`.
	GenAi,
	/// Vertex publisher-model `predict`.
	Vertex,
}

/// Encodes a Google embedding request for the selected endpoint family.
pub fn encode(request: &EmbedRequest, variant: GoogleEmbeddingVariant) -> Result<Bytes, Error> {
	reject_props(&request.props)?;
	let body = match variant {
		GoogleEmbeddingVariant::GenAi => encode_gen_ai(request),
		GoogleEmbeddingVariant::Vertex => encode_vertex(request),
	};
	serde_json::to_vec(&body)
		.map(Bytes::from)
		.map_err(|error| Error::Provider(format_smol!("failed to encode embedding request: {error}")))
}

/// Decodes a Google embedding response.
///
/// Vertex token statistics are exact. Public `batchEmbedContents` does not
/// return token accounting, so `estimated_input_tokens` is retained with
/// [`Accuracy::Estimated`].
pub fn decode(
	body: &[u8],
	variant: GoogleEmbeddingVariant,
	estimated_input_tokens: u64,
) -> Result<EmbedResponse, Error> {
	let value: Value = serde_json::from_slice(body)
		.map_err(|error| Error::Provider(format_smol!("invalid embedding response JSON: {error}")))?;
	if let Some(message) = error_message(&value) {
		return Err(Error::Provider(message));
	}
	match variant {
		GoogleEmbeddingVariant::GenAi => decode_gen_ai(&value, estimated_input_tokens),
		GoogleEmbeddingVariant::Vertex => decode_vertex(&value, estimated_input_tokens),
	}
}

/// Extracts the safe message from Google's structured error envelope.
#[must_use]
pub fn error_message(value: &Value) -> Option<SmolStr> {
	let error = value.get("error")?;
	if let Some(message) = error.get("message").and_then(Value::as_str) {
		return Some(SmolStr::from(message));
	}
	error.as_str().map(SmolStr::from)
}

fn encode_gen_ai(request: &EmbedRequest) -> Value {
	let model = if request.model.starts_with("models/") {
		request.model.to_string()
	} else {
		format!("models/{}", request.model)
	};
	let requests: Vec<Value> = request
		.texts
		.iter()
		.map(|text| {
			let mut item = Map::new();
			item.insert("model".into(), Value::String(model.clone()));
			item.insert("content".into(), serde_json::json!({ "parts": [{ "text": text.as_str() }] }));
			if let Some(dimensions) = request.dimensions {
				item.insert("outputDimensionality".into(), Value::from(dimensions));
			}
			Value::Object(item)
		})
		.collect();
	serde_json::json!({ "requests": requests })
}

fn encode_vertex(request: &EmbedRequest) -> Value {
	let instances = request
		.texts
		.iter()
		.map(|text| serde_json::json!({ "content": text.as_str() }))
		.collect::<Vec<_>>();
	let mut body = Map::new();
	body.insert("instances".into(), Value::Array(instances));
	if let Some(dimensions) = request.dimensions {
		body.insert("parameters".into(), serde_json::json!({ "outputDimensionality": dimensions }));
	}
	Value::Object(body)
}

fn decode_gen_ai(value: &Value, estimated_input_tokens: u64) -> Result<EmbedResponse, Error> {
	let embeddings = value
		.get("embeddings")
		.and_then(Value::as_array)
		.ok_or_else(|| {
			Error::Provider(SmolStr::from("Google embedding response is missing embeddings"))
		})?;
	let vectors = embeddings
		.iter()
		.map(|embedding| decode_vector(embedding.get("values")))
		.collect::<Result<Vec<_>, _>>()?;
	Ok(EmbedResponse::builder()
		.vectors(vectors)
		.usage(usage(estimated_input_tokens, Accuracy::Estimated))
		.build())
}

fn decode_vertex(value: &Value, estimated_input_tokens: u64) -> Result<EmbedResponse, Error> {
	let predictions = value
		.get("predictions")
		.and_then(Value::as_array)
		.ok_or_else(|| {
			Error::Provider(SmolStr::from("Vertex embedding response is missing predictions"))
		})?;
	let mut exact_tokens = Some(0u64);
	let vectors = predictions
		.iter()
		.map(|prediction| {
			let embedding = prediction.get("embeddings").unwrap_or(prediction);
			let token_count = embedding
				.get("statistics")
				.and_then(|statistics| {
					statistics
						.get("token_count")
						.or_else(|| statistics.get("tokenCount"))
				})
				.and_then(Value::as_u64);
			exact_tokens = exact_tokens
				.zip(token_count)
				.map(|(total, next)| total.saturating_add(next));
			decode_vector(embedding.get("values"))
		})
		.collect::<Result<Vec<_>, _>>()?;
	let (tokens, accuracy) = exact_tokens
		.map_or((estimated_input_tokens, Accuracy::Estimated), |tokens| (tokens, Accuracy::Exact));
	Ok(EmbedResponse::builder()
		.vectors(vectors)
		.usage(usage(tokens, accuracy))
		.build())
}

fn decode_vector(values: Option<&Value>) -> Result<EmbeddingVector, Error> {
	let values = values
		.and_then(Value::as_array)
		.ok_or_else(|| Error::Provider(SmolStr::from("embedding response item has no vector")))?;
	let values = values
		.iter()
		.map(|value| {
			let number = value.as_f64().ok_or_else(|| {
				Error::Provider(SmolStr::from("embedding vector contains a non-number"))
			})? as f32;
			if !number.is_finite() {
				return Err(Error::Provider(SmolStr::from(
					"embedding vector contains a non-finite component",
				)));
			}
			Ok(number)
		})
		.collect::<Result<Vec<_>, _>>()?;
	Ok(EmbeddingVector::builder().values(values).build())
}

fn usage(input_tokens: u64, accuracy: Accuracy) -> Usage {
	Usage::builder()
		.input_tokens(input_tokens)
		.output_tokens(0)
		.cache_read_tokens(0)
		.cache_write_tokens(0)
		.accuracy(accuracy)
		.detail(Props::default())
		.build()
}

fn reject_props(props: &Props) -> Result<(), Error> {
	if props.is_empty() {
		return Ok(());
	}
	Err(Error::Unsupported(
		props
			.0
			.keys()
			.map(|key| {
				Unsupported::builder()
					.what(key.clone())
					.detail(SmolStr::from("the Google embeddings wire has no mapping for this property"))
					.action(UnsupportedAction::Dropped)
					.build()
			})
			.collect(),
	))
}
