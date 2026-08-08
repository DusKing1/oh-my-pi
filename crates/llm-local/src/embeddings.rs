use fastembed::{TextEmbedding, TextInitOptions};
use ort::ep::ExecutionProviderDispatch;
use tokio_util::sync::CancellationToken;

use crate::{Accelerator, DevicePreference, Error, Hub, Result, worker::Worker};

/// Curated embedding models plus an escape hatch to the complete fastembed
/// catalog.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub enum EmbeddingModel {
	/// `BAAI/bge-small-en-v1.5`, matching the prior local-memory default.
	#[default]
	BgeSmallEnglish,
	/// Quantized `BAAI/bge-small-en-v1.5` for a smaller CPU footprint.
	BgeSmallEnglishQuantized,
	/// `BAAI/bge-base-en-v1.5`.
	BgeBaseEnglish,
	/// Multilingual `BAAI/bge-m3`.
	BgeM3,
	/// `intfloat/multilingual-e5-small`.
	MultilingualE5Small,
	/// `nomic-ai/nomic-embed-text-v1.5`.
	NomicEmbedTextV15,
	/// `sentence-transformers/all-MiniLM-L6-v2`.
	AllMiniLmL6V2,
	/// Quantized `onnx-community/embeddinggemma-300m-ONNX`.
	EmbeddingGemma300MQ4,
	/// Any model supported by the linked fastembed release.
	Other(fastembed::EmbeddingModel),
}

impl EmbeddingModel {
	fn runtime(&self) -> fastembed::EmbeddingModel {
		match self {
			Self::BgeSmallEnglish => fastembed::EmbeddingModel::BGESmallENV15,
			Self::BgeSmallEnglishQuantized => fastembed::EmbeddingModel::BGESmallENV15Q,
			Self::BgeBaseEnglish => fastembed::EmbeddingModel::BGEBaseENV15,
			Self::BgeM3 => fastembed::EmbeddingModel::BGEM3,
			Self::MultilingualE5Small => fastembed::EmbeddingModel::MultilingualE5Small,
			Self::NomicEmbedTextV15 => fastembed::EmbeddingModel::NomicEmbedTextV15,
			Self::AllMiniLmL6V2 => fastembed::EmbeddingModel::AllMiniLML6V2,
			Self::EmbeddingGemma300MQ4 => fastembed::EmbeddingModel::EmbeddingGemma300MQ4,
			Self::Other(model) => model.clone(),
		}
	}
}

/// Per-call embedding controls.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EmbeddingOptions {
	/// Number of inputs evaluated together; the runtime chooses when absent.
	pub batch_size: Option<usize>,
	/// L2-normalize each result for direct cosine or dot-product search.
	pub normalize:  bool,
}

impl Default for EmbeddingOptions {
	fn default() -> Self {
		Self { batch_size: None, normalize: true }
	}
}

/// Configures an asynchronously hosted text embedding model.
#[derive(Clone)]
pub struct EmbedderBuilder {
	model:                  EmbeddingModel,
	hub:                    Option<Hub>,
	device:                 DevicePreference,
	max_length:             usize,
	show_download_progress: bool,
}

impl EmbedderBuilder {
	/// Creates a builder for a curated or fastembed model.
	pub const fn new(model: EmbeddingModel) -> Self {
		Self {
			model,
			hub: None,
			device: DevicePreference::Auto,
			max_length: 512,
			show_download_progress: false,
		}
	}

	/// Uses the shared cache directory; fastembed reads endpoint and credentials
	/// from HF environment variables.
	pub fn hub(mut self, hub: Hub) -> Self {
		self.hub = Some(hub);
		self
	}

	/// Selects CPU, Core ML, or CUDA execution.
	pub const fn device(mut self, device: DevicePreference) -> Self {
		self.device = device;
		self
	}

	/// Truncates tokenized inputs at this model sequence length.
	pub const fn max_length(mut self, max_length: usize) -> Self {
		self.max_length = max_length;
		self
	}

	/// Enables fastembed's terminal download progress display.
	pub const fn show_download_progress(mut self, show: bool) -> Self {
		self.show_download_progress = show;
		self
	}

	/// Downloads and loads the model on a dedicated worker thread.
	pub async fn build(self) -> Result<Embedder> {
		if self.max_length == 0 {
			return Err(Error::invalid("embedding max length must be non-zero"));
		}
		let hub = match self.hub {
			Some(hub) => hub,
			None => Hub::new()?,
		};
		if hub.offline() {
			return Err(Error::invalid(
				"fastembed cannot enforce cache-only loading; build embeddings without offline Hub \
				 policy",
			));
		}
		let (execution_providers, requested_accelerator) = execution_providers(self.device)?;
		let model = self.model.runtime();
		let cache_dir = hub.cache_dir().to_path_buf();
		let max_length = self.max_length;
		let show_download_progress = self.show_download_progress;
		let preference = self.device;
		let worker = Worker::spawn("omp-llm-local-embeddings", move || {
			let loaded = load_embedding(
				model.clone(),
				cache_dir.clone(),
				max_length,
				show_download_progress,
				execution_providers,
			);
			match loaded {
				Ok(model) => Ok(EmbeddingRuntime { model, accelerator: requested_accelerator }),
				Err(_)
					if preference == DevicePreference::Auto
						&& requested_accelerator != Accelerator::Cpu =>
				{
					let model =
						load_embedding(model, cache_dir, max_length, show_download_progress, Vec::new())?;
					Ok(EmbeddingRuntime { model, accelerator: Accelerator::Cpu })
				},
				Err(error) => Err(error),
			}
		})
		.await?;
		let accelerator = worker
			.run_uncancelled(|runtime| Ok(runtime.accelerator))
			.await?;
		Ok(Embedder { worker, accelerator })
	}
}

impl Default for EmbedderBuilder {
	fn default() -> Self {
		Self::new(EmbeddingModel::default())
	}
}

struct EmbeddingRuntime {
	model:       TextEmbedding,
	accelerator: Accelerator,
}

/// Asynchronous text embedding model serialized on a dedicated runtime thread.
#[derive(Clone)]
pub struct Embedder {
	worker:      Worker<EmbeddingRuntime>,
	accelerator: Accelerator,
}

impl Embedder {
	/// Starts a builder using `BAAI/bge-small-en-v1.5`.
	pub fn builder() -> EmbedderBuilder {
		EmbedderBuilder::default()
	}

	/// Backend requested when the model was loaded.
	pub const fn accelerator(&self) -> Accelerator {
		self.accelerator
	}

	/// Embeds an owned batch, allowing inference to outlive the caller's input
	/// buffers.
	pub async fn embed(
		&self,
		texts: Vec<String>,
		options: EmbeddingOptions,
		cancel: CancellationToken,
	) -> Result<Vec<Vec<f32>>> {
		self
			.worker
			.run(cancel, move |runtime, cancel| {
				let mut embeddings = runtime
					.model
					.embed(texts, options.batch_size)
					.map_err(|error| Error::backend("fastembed", error))?;
				if cancel.is_cancelled() {
					return Err(Error::Cancelled);
				}
				if options.normalize {
					for embedding in &mut embeddings {
						normalize(embedding);
					}
				}
				Ok(embeddings)
			})
			.await
	}

	/// Embeds one string with normalized default options.
	pub async fn embed_one(&self, text: impl Into<String>) -> Result<Vec<f32>> {
		let mut embeddings = self
			.embed(vec![text.into()], EmbeddingOptions::default(), CancellationToken::new())
			.await?;
		embeddings
			.pop()
			.ok_or_else(|| Error::backend("fastembed", "model returned no embedding"))
	}

	/// Stops the worker after all earlier requests have drained.
	pub async fn shutdown(&self) -> Result<()> {
		self.worker.shutdown().await
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

fn load_embedding(
	model: fastembed::EmbeddingModel,
	cache_dir: std::path::PathBuf,
	max_length: usize,
	show_download_progress: bool,
	execution_providers: Vec<ExecutionProviderDispatch>,
) -> Result<TextEmbedding> {
	let options = TextInitOptions::new(model)
		.with_cache_dir(cache_dir)
		.with_max_length(max_length)
		.with_show_download_progress(show_download_progress)
		.with_execution_providers(execution_providers);
	TextEmbedding::try_new(options).map_err(|error| Error::backend("fastembed", error))
}

fn execution_providers(
	preference: DevicePreference,
) -> Result<(Vec<ExecutionProviderDispatch>, Accelerator)> {
	match preference {
		DevicePreference::Cpu => Ok((Vec::new(), Accelerator::Cpu)),
		DevicePreference::Metal => {
			#[cfg(target_os = "macos")]
			{
				Ok(core_ml_providers())
			}
			#[cfg(not(target_os = "macos"))]
			{
				core_ml_providers()
			}
		},
		DevicePreference::Cuda => cuda_providers(),
		DevicePreference::Gpu => {
			#[cfg(target_os = "macos")]
			{
				Ok(native_gpu_providers())
			}
			#[cfg(not(target_os = "macos"))]
			{
				native_gpu_providers()
			}
		},
		DevicePreference::Auto => {
			#[cfg(target_os = "macos")]
			{
				Ok(native_gpu_providers())
			}
			#[cfg(not(target_os = "macos"))]
			{
				native_gpu_providers().or_else(|_| Ok((Vec::new(), Accelerator::Cpu)))
			}
		},
	}
}

#[cfg(target_os = "macos")]
fn core_ml_providers() -> (Vec<ExecutionProviderDispatch>, Accelerator) {
	use ort::ep::CoreML;
	(vec![CoreML::default().build().error_on_failure()], Accelerator::CoreMl)
}

#[cfg(not(target_os = "macos"))]
fn core_ml_providers() -> Result<(Vec<ExecutionProviderDispatch>, Accelerator)> {
	Err(Error::unavailable("Core ML is available only on macOS"))
}

#[cfg(feature = "cuda")]
fn cuda_providers() -> Result<(Vec<ExecutionProviderDispatch>, Accelerator)> {
	use ort::ep::CUDA;
	Ok((vec![CUDA::default().build().error_on_failure()], Accelerator::Cuda))
}

#[cfg(not(feature = "cuda"))]
fn cuda_providers() -> Result<(Vec<ExecutionProviderDispatch>, Accelerator)> {
	Err(Error::unavailable("enable the omp-llm-local `cuda` feature"))
}

#[cfg(target_os = "macos")]
fn native_gpu_providers() -> (Vec<ExecutionProviderDispatch>, Accelerator) {
	core_ml_providers()
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
fn native_gpu_providers() -> Result<(Vec<ExecutionProviderDispatch>, Accelerator)> {
	cuda_providers()
}

#[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
fn native_gpu_providers() -> Result<(Vec<ExecutionProviderDispatch>, Accelerator)> {
	Err(Error::unavailable("no embedding GPU backend is enabled"))
}
