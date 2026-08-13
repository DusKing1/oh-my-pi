//! FastEmbed-backed local text embeddings.

use std::{path::PathBuf, sync::Arc, time::Duration};

use fastembed::{TextEmbedding, TextInitOptions};

use super::runtime::{
	LocalCancellation, LocalError, LocalErrorKind, LocalExecutionReceipt, LocalResult, LocalRuntime,
	MemoryPool,
};

/// Configuration for a real FastEmbed model.
#[derive(Clone, Debug)]
pub struct EmbeddingConfig {
	/// Model from FastEmbed's typed catalog.
	pub model:           fastembed::EmbeddingModel,
	/// Hugging Face cache used by FastEmbed.
	pub cache_dir:       PathBuf,
	/// Maximum tokenized input length.
	pub max_length:      usize,
	/// Estimated resident bytes charged before loading.
	pub resident_bytes:  usize,
	/// Admission limit; currently must be one because FastEmbed access is
	/// serialized.
	pub max_concurrency: usize,
	/// Duration after which an explicit idle sweep unloads the model.
	pub idle_timeout:    Duration,
}

/// Per-call embedding controls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddingOptions {
	/// Optional FastEmbed batch size.
	pub batch_size: Option<usize>,
	/// Whether to L2-normalize every vector.
	pub normalize:  bool,
}

impl Default for EmbeddingOptions {
	fn default() -> Self {
		Self { batch_size: None, normalize: true }
	}
}

/// Embedding result with lifecycle/isolation evidence.
#[derive(Debug)]
pub struct EmbeddingOutput {
	/// One vector per input, in input order.
	pub embeddings: Vec<Vec<f32>>,
	/// Local runtime execution receipt.
	pub receipt:    LocalExecutionReceipt,
}

/// Lazy, bounded adapter over FastEmbed's ONNX runtime.
#[derive(Clone)]
pub struct EmbeddingAdapter {
	runtime: LocalRuntime<TextEmbedding>,
}

impl EmbeddingAdapter {
	/// Creates a lazy adapter without downloading or loading until first use.
	pub fn new(config: EmbeddingConfig, memory: Arc<MemoryPool>) -> LocalResult<Self> {
		if config.max_length == 0 {
			return Err(LocalError::new(
				LocalErrorKind::InvalidInput,
				"embedding maximum length must be non-zero",
			));
		}
		let resident_bytes = config.resident_bytes;
		let max_concurrency = config.max_concurrency;
		let idle_timeout = config.idle_timeout;
		let runtime = LocalRuntime::new(
			move || {
				let options = TextInitOptions::new(config.model.clone())
					.with_cache_dir(config.cache_dir.clone())
					.with_max_length(config.max_length)
					.with_show_download_progress(false);
				TextEmbedding::try_new(options).map_err(|error| {
					LocalError::new(LocalErrorKind::Backend, format!("FastEmbed load failed: {error}"))
				})
			},
			memory,
			resident_bytes,
			max_concurrency,
			idle_timeout,
		)?;
		Ok(Self { runtime })
	}

	/// Embeds an owned batch using the real ONNX model.
	pub fn embed(
		&self,
		texts: Vec<String>,
		options: EmbeddingOptions,
		cancel: &LocalCancellation,
	) -> LocalResult<EmbeddingOutput> {
		if texts.is_empty() {
			return Err(LocalError::new(
				LocalErrorKind::InvalidInput,
				"embedding requires at least one input",
			));
		}
		if options.batch_size == Some(0) {
			return Err(LocalError::new(
				LocalErrorKind::InvalidInput,
				"embedding batch size must be non-zero",
			));
		}
		let lease = self.runtime.acquire(cancel)?;
		let receipt = lease.receipt();
		let mut embeddings = lease.with_engine(|model| {
			model.embed(texts, options.batch_size).map_err(|error| {
				LocalError::new(LocalErrorKind::Backend, format!("FastEmbed inference failed: {error}"))
			})
		})?;
		if cancel.is_cancelled() {
			return Err(LocalError::cancelled());
		}
		if options.normalize {
			for embedding in &mut embeddings {
				normalize(embedding);
			}
		}
		Ok(EmbeddingOutput { embeddings, receipt })
	}

	/// Borrows the shared lifecycle runtime for idle sweeps and diagnostics.
	pub const fn runtime(&self) -> &LocalRuntime<TextEmbedding> {
		&self.runtime
	}
}

fn normalize(embedding: &mut [f32]) {
	let norm = embedding
		.iter()
		.map(|value| value * value)
		.sum::<f32>()
		.sqrt();
	if norm > 0.0 && norm.is_finite() {
		for value in embedding {
			*value /= norm;
		}
	}
}
