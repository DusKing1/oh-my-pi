use bon::Builder;
use omp_core::SmolStr;

use crate::{Accuracy, ContextRef, Props, Thread, ToolDef, Usage};

/// Input whose prompt tokens should be counted.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum CountInput {
	/// Count a server-held conversation without uploading it again.
	Context(ContextRef),
	/// Count a complete inline conversation.
	Thread(Thread),
}

/// Request for prompt token accounting, including tool definitions that
/// contribute to the prompt.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct CountRequest {
	/// Catalog model whose tokenizer and projection rules apply.
	pub model: SmolStr,
	/// Held or inline conversation to count.
	pub input: CountInput,
	/// Tool schemas included in the projected prompt.
	pub tools: Vec<ToolDef>,
}

/// Token count together with whether it came from an exact source or a
/// heuristic.
#[non_exhaustive]
#[derive(Builder, Clone, Copy, Debug, Eq, PartialEq)]
pub struct CountResponse {
	/// Total projected prompt tokens.
	pub tokens:   u64,
	/// Provenance of the count.
	pub accuracy: Accuracy,
}

/// Request for one embedding vector per input text.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct EmbedRequest {
	/// Catalog model used for embedding.
	pub model:      SmolStr,
	/// Ordered texts, preserving one-to-one response correspondence.
	pub texts:      Vec<SmolStr>,
	/// Requested vector width where the model supports dimensionality reduction.
	pub dimensions: Option<u32>,
	/// Namespaced provider-specific embedding controls.
	pub props:      Props,
}

/// One embedding vector.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct EmbeddingVector {
	/// Components in model-defined embedding order.
	pub values: Vec<f32>,
}

/// Ordered embeddings and their token accounting.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct EmbedResponse {
	/// One vector for each request text in the same order.
	pub vectors: Vec<EmbeddingVector>,
	/// Prompt token usage when reported.
	pub usage:   Option<Usage>,
}
