//! OpenAI-compatible unary embeddings wire codec.

use bytes::Bytes;
use omp_core::{Str, fmts};
use omp_llm_types::{
	Accuracy, EmbedRequest, EmbedResponse, EmbeddingVector, Error, Props, Unsupported,
	UnsupportedAction, Usage,
};
use serde_json::{Map, Value};

/// Encodes the OpenAI-compatible `POST /embeddings` request body.
///
/// Provider-specific properties are rejected rather than silently omitted.
pub fn encode(request: &EmbedRequest) -> Result<Bytes, Error> {
	reject_props(&request.props)?;
	let mut body = Map::new();
	body.insert("model".into(), Value::String(request.model.to_string()));
	body.insert(
		"input".into(),
		Value::Array(
			request
				.texts
				.iter()
				.map(|text| Value::String(text.to_string()))
				.collect(),
		),
	);
	body.insert("encoding_format".into(), Value::String("float".into()));
	if let Some(dimensions) = request.dimensions {
		body.insert("dimensions".into(), Value::from(dimensions));
	}
	serde_json::to_vec(&Value::Object(body))
		.map(Bytes::from)
		.map_err(|error| Error::Provider(fmts!("failed to encode embedding request: {error}")))
}

/// Decodes an OpenAI-compatible embeddings response and restores input order
/// from each response item's `index` field.
///
/// `estimated_input_tokens` is used only when the endpoint omits its `usage`
/// object, as some otherwise compatible local servers do.
pub fn decode(body: &[u8], estimated_input_tokens: u64) -> Result<EmbedResponse, Error> {
	let value: Value = serde_json::from_slice(body)
		.map_err(|error| Error::Provider(fmts!("invalid embedding response JSON: {error}")))?;
	if let Some(message) = error_message(&value) {
		return Err(Error::Provider(message));
	}
	let data = value
		.get("data")
		.and_then(Value::as_array)
		.ok_or_else(|| Error::Provider(Str::from("embedding response is missing data")))?;
	let mut ordered = vec![None; data.len()];
	for item in data {
		let index = item
			.get("index")
			.and_then(Value::as_u64)
			.and_then(|index| usize::try_from(index).ok())
			.ok_or_else(|| {
				Error::Provider(Str::from("embedding response item has no valid index"))
			})?;
		if index >= ordered.len() || ordered[index].is_some() {
			return Err(Error::Provider(Str::from(
				"embedding response contains an out-of-range or duplicate index",
			)));
		}
		let embedding = item
			.get("embedding")
			.and_then(Value::as_array)
			.ok_or_else(|| Error::Provider(Str::from("embedding response item has no vector")))?;
		ordered[index] = Some(
			EmbeddingVector::builder()
				.values(vector(embedding)?)
				.build(),
		);
	}
	let vectors = ordered
		.into_iter()
		.collect::<Option<Vec<_>>>()
		.ok_or_else(|| Error::Provider(Str::from("embedding response has a missing index")))?;
	let exact = value
		.get("usage")
		.and_then(|usage| usage.get("prompt_tokens"))
		.and_then(Value::as_u64);
	let usage = usage(exact.unwrap_or(estimated_input_tokens), exact.is_some());
	Ok(EmbedResponse::builder()
		.vectors(vectors)
		.usage(usage)
		.build())
}

/// Extracts the safe provider message from a successful or error envelope.
#[must_use]
pub fn error_message(value: &Value) -> Option<Str> {
	let error = value.get("error")?;
	if let Some(message) = error.get("message").and_then(Value::as_str) {
		return Some(Str::from(message));
	}
	error.as_str().map(Str::from)
}

fn vector(values: &[Value]) -> Result<Vec<f32>, Error> {
	values
		.iter()
		.map(|value| {
			let number = value.as_f64().ok_or_else(|| {
				Error::Provider(Str::from("embedding vector contains a non-number"))
			})?;
			let number = number as f32;
			if !number.is_finite() {
				return Err(Error::Provider(Str::from(
					"embedding vector contains a non-finite component",
				)));
			}
			Ok(number)
		})
		.collect()
}

fn usage(input_tokens: u64, exact: bool) -> Usage {
	Usage::builder()
		.input_tokens(input_tokens)
		.output_tokens(0)
		.cache_read_tokens(0)
		.cache_write_tokens(0)
		.accuracy(if exact {
			Accuracy::Exact
		} else {
			Accuracy::Estimated
		})
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
					.detail(Str::from(
						"the OpenAI-compatible embeddings wire has no mapping for this property",
					))
					.action(UnsupportedAction::Dropped)
					.build()
			})
			.collect(),
	))
}
