use std::{
	collections::HashMap,
	num::NonZeroU32,
	path::PathBuf,
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
};

use futures::{Stream, channel::mpsc};
use llama_cpp_2::{
	TokenToStringError,
	context::params::LlamaContextParams,
	llama_backend::LlamaBackend,
	llama_batch::LlamaBatch,
	model::{AddBos, LlamaChatMessage, LlamaModel, params::LlamaModelParams},
	sampling::LlamaSampler,
	token::LlamaToken,
};
use omp_core::SmolStr;
use parking_lot::Mutex;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;
use xutf::{Encoding, Utf8};

use crate::{Accelerator, DevicePreference, Error, Hub, ModelRepo, Result, worker::Worker};

static LLAMA_WORKER: OnceCell<Worker<LlamaState>> = OnceCell::const_new();

/// Purpose-built local language-model presets used by title generation,
/// classification, and memory work.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum SmallModel {
	/// 350M-parameter LFM2 Q4 model for latency-sensitive labels and titles.
	Lfm2_350M,
	/// 700M-parameter LFM2 Q4 model balancing latency and instruction following.
	#[default]
	Lfm2_700M,
	/// 1.2B-parameter LFM2 Q4 model for higher-quality extraction and
	/// consolidation.
	Lfm2_1_2B,
	/// 600M-parameter Qwen3 Q8 model with strong multilingual instruction
	/// following.
	Qwen3_600M,
	/// 500M-parameter Qwen2.5 Q4 model without Qwen3's visible reasoning
	/// convention.
	Qwen2_5_500M,
}

/// A GGUF model stored locally or in a Hugging Face model repository.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextModel {
	/// A curated small GGUF model.
	Small(SmallModel),
	/// One or more GGUF shards from a Hugging Face repository.
	HuggingFace {
		/// Repository containing the GGUF file or shards.
		repo:  ModelRepo,
		/// Repository-relative GGUF filenames in shard order.
		files: Vec<SmolStr>,
	},
	/// A GGUF file already present on disk; split shards are discovered beside
	/// it.
	Local(PathBuf),
}

impl From<SmallModel> for TextModel {
	fn from(model: SmallModel) -> Self {
		Self::Small(model)
	}
}

impl SmallModel {
	fn source(self) -> (ModelRepo, SmolStr) {
		match self {
			Self::Lfm2_350M => {
				(ModelRepo::new("LiquidAI/LFM2-350M-GGUF"), "LFM2-350M-Q4_K_M.gguf".into())
			},
			Self::Lfm2_700M => {
				(ModelRepo::new("LiquidAI/LFM2-700M-GGUF"), "LFM2-700M-Q4_K_M.gguf".into())
			},
			Self::Lfm2_1_2B => {
				(ModelRepo::new("LiquidAI/LFM2-1.2B-GGUF"), "LFM2-1.2B-Q4_K_M.gguf".into())
			},
			Self::Qwen3_600M => {
				(ModelRepo::new("Qwen/Qwen3-0.6B-GGUF"), "Qwen3-0.6B-Q8_0.gguf".into())
			},
			Self::Qwen2_5_500M => (
				ModelRepo::new("Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
				"qwen2.5-0.5b-instruct-q4_k_m.gguf".into(),
			),
		}
	}
}

/// Role of one local chat message.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ChatRole {
	/// Instructions governing the assistant.
	System,
	/// Human or calling-agent input.
	User,
	/// Prior model output included as conversational context.
	Assistant,
}

/// Owned chat message suitable for queued asynchronous generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatMessage {
	/// Message role used by the model's embedded chat template.
	pub role:    ChatRole,
	/// Message text.
	pub content: SmolStr,
}

impl ChatMessage {
	/// Creates a system instruction.
	pub fn system(content: impl Into<SmolStr>) -> Self {
		Self { role: ChatRole::System, content: content.into() }
	}

	/// Creates a user message.
	pub fn user(content: impl Into<SmolStr>) -> Self {
		Self { role: ChatRole::User, content: content.into() }
	}

	/// Creates an assistant-history message.
	pub fn assistant(content: impl Into<SmolStr>) -> Self {
		Self { role: ChatRole::Assistant, content: content.into() }
	}
}

/// Sampling controls for one local generation.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationOptions {
	/// Maximum number of newly generated tokens.
	pub max_tokens:         usize,
	/// Sampling temperature; `None` or `0.0` performs greedy decoding.
	pub temperature:        Option<f32>,
	/// Nucleus-sampling probability cutoff.
	pub top_p:              Option<f32>,
	/// Restricts sampling to the highest-probability `k` tokens.
	pub top_k:              Option<u32>,
	/// Discards tokens below this fraction of the leading token's probability.
	pub min_p:              Option<f32>,
	/// Penalty applied to tokens already present in the prompt or generated
	/// text.
	pub repetition_penalty: Option<f32>,
	/// Penalty scaled by how often a token already appears.
	pub frequency_penalty:  Option<f32>,
	/// Penalty applied when a token has appeared at least once.
	pub presence_penalty:   Option<f32>,
	/// Seed used by non-greedy sampling.
	pub seed:               u32,
	/// Text sequences that stop generation and are withheld from the result.
	pub stop:               Vec<SmolStr>,
}

impl Default for GenerationOptions {
	fn default() -> Self {
		Self {
			max_tokens:         256,
			temperature:        None,
			top_p:              None,
			top_k:              None,
			min_p:              None,
			repetition_penalty: None,
			frequency_penalty:  None,
			presence_penalty:   None,
			seed:               0,
			stop:               Vec::new(),
		}
	}
}

/// Configures a GGUF model's source, context, CPU threads, and GPU offload.
#[derive(Clone)]
pub struct TextGeneratorBuilder {
	model:        TextModel,
	hub:          Option<Hub>,
	device:       DevicePreference,
	context_size: u32,
	threads:      Option<usize>,
	gpu_layers:   u32,
}

impl TextGeneratorBuilder {
	/// Creates a builder from a curated, remote, or local GGUF model.
	pub fn new(model: impl Into<TextModel>) -> Self {
		Self {
			model:        model.into(),
			hub:          None,
			device:       DevicePreference::Auto,
			context_size: 4096,
			threads:      None,
			gpu_layers:   u32::MAX,
		}
	}

	/// Shares Hugging Face cache, token, endpoint, and offline policy.
	pub fn hub(mut self, hub: Hub) -> Self {
		self.hub = Some(hub);
		self
	}

	/// Selects CPU, Metal, or CUDA execution.
	pub const fn device(mut self, device: DevicePreference) -> Self {
		self.device = device;
		self
	}

	/// Sets the model context window allocated for each generation.
	pub const fn context_size(mut self, tokens: u32) -> Self {
		self.context_size = tokens;
		self
	}

	/// Sets llama.cpp's prompt-processing and generation thread count.
	pub const fn threads(mut self, threads: usize) -> Self {
		self.threads = Some(threads);
		self
	}

	/// Limits transformer layers offloaded when GPU execution is selected.
	pub const fn gpu_layers(mut self, layers: u32) -> Self {
		self.gpu_layers = layers;
		self
	}

	/// Downloads and memory-maps the model on the process-wide llama.cpp worker.
	pub async fn build(self) -> Result<TextGenerator> {
		let context_size = NonZeroU32::new(self.context_size)
			.ok_or_else(|| Error::invalid("text context size must be non-zero"))?;
		if self.threads == Some(0) {
			return Err(Error::invalid("text-generation thread count must be non-zero"));
		}
		let hub = match self.hub {
			Some(hub) => hub,
			None => Hub::new()?,
		};
		let model_path = resolve_model(self.model, &hub).await?;
		let worker = llama_worker().await?;
		let device = self.device;
		let gpu_layers = self.gpu_layers;
		let threads = self.threads.unwrap_or_else(|| {
			std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
		});
		let (id, accelerator) = worker
			.run_uncancelled(move |state| {
				let (use_gpu, accelerator) = select_accelerator(device, &state.backend)?;
				let mut params = LlamaModelParams::default();
				if use_gpu {
					params = params.with_n_gpu_layers(gpu_layers);
				}
				let model = LlamaModel::load_from_file(&state.backend, model_path, &params)
					.map_err(|error| Error::backend("llama.cpp", error))?;
				let id = state.next_id;
				state.next_id = state
					.next_id
					.checked_add(1)
					.ok_or_else(|| Error::backend("llama.cpp", "model identifier space exhausted"))?;
				state
					.models
					.insert(id, LoadedModel { model, context_size, threads });
				Ok((id, accelerator))
			})
			.await?;
		Ok(TextGenerator { handle: Arc::new(ModelHandle { id, worker }), accelerator })
	}
}

struct LlamaState {
	backend: LlamaBackend,
	models:  HashMap<u64, LoadedModel>,
	next_id: u64,
}

struct LoadedModel {
	model:        LlamaModel,
	context_size: NonZeroU32,
	threads:      usize,
}

struct ModelHandle {
	id:     u64,
	worker: Worker<LlamaState>,
}

impl Drop for ModelHandle {
	fn drop(&mut self) {
		let id = self.id;
		let _ = self.worker.dispatch(move |state| {
			state.models.remove(&id);
		});
	}
}

/// Concurrent asynchronous facade over a serialized process-wide llama.cpp
/// worker.
#[derive(Clone)]
pub struct TextGenerator {
	handle:      Arc<ModelHandle>,
	accelerator: Accelerator,
}

impl TextGenerator {
	/// Starts a builder for a curated small model.
	pub fn small(model: SmallModel) -> TextGeneratorBuilder {
		TextGeneratorBuilder::new(model)
	}

	/// Backend selected when the model was loaded.
	pub const fn accelerator(&self) -> Accelerator {
		self.accelerator
	}

	/// Generates one complete assistant message.
	pub async fn generate(
		&self,
		messages: &[ChatMessage],
		options: GenerationOptions,
		cancel: CancellationToken,
	) -> Result<SmolStr> {
		validate_request(messages, &options)?;
		let messages = messages.to_vec();
		let id = self.handle.id;
		self
			.handle
			.worker
			.run(cancel, move |state, cancel| {
				let loaded = state
					.models
					.get(&id)
					.ok_or_else(|| Error::backend("llama.cpp", "model was unloaded"))?;
				let mut output = String::new();
				generate(&state.backend, loaded, &messages, options, cancel, &mut |chunk| {
					output.push_str(chunk.as_str());
					true
				})?;
				Ok(output.into())
			})
			.await
	}

	/// Counts the tokens the chat template produces for `messages`, using the
	/// model's own tokenizer.
	pub async fn count_tokens(&self, messages: &[ChatMessage]) -> Result<u64> {
		if messages.is_empty() {
			return Err(Error::invalid("at least one chat message is required"));
		}
		let messages = messages.to_vec();
		let id = self.handle.id;
		self
			.handle
			.worker
			.run_uncancelled(move |state| {
				let loaded = state
					.models
					.get(&id)
					.ok_or_else(|| Error::backend("llama.cpp", "model was unloaded"))?;
				let tokens = chat_tokens(loaded, &messages)?;
				Ok(tokens.len() as u64)
			})
			.await
	}

	/// Counts raw text tokens without applying the chat template.
	pub async fn count_text_tokens(&self, text: &str) -> Result<u64> {
		let text = text.to_owned();
		let id = self.handle.id;
		self
			.handle
			.worker
			.run_uncancelled(move |state| {
				let loaded = state
					.models
					.get(&id)
					.ok_or_else(|| Error::backend("llama.cpp", "model was unloaded"))?;
				let tokens = loaded
					.model
					.str_to_token(&text, AddBos::Always)
					.map_err(|error| Error::backend("llama.cpp tokenizer", error))?;
				Ok(tokens.len() as u64)
			})
			.await
	}

	/// Generates from one user prompt with greedy default sampling.
	pub async fn complete(&self, prompt: impl Into<SmolStr>) -> Result<SmolStr> {
		self
			.generate(
				&[ChatMessage::user(prompt)],
				GenerationOptions::default(),
				CancellationToken::new(),
			)
			.await
	}

	/// Starts incremental token streaming for an owned chat request.
	pub fn stream(
		&self,
		messages: Vec<ChatMessage>,
		options: GenerationOptions,
	) -> Result<GenerationStream> {
		validate_request(&messages, &options)?;
		let id = self.handle.id;
		let worker = self.handle.worker.clone();
		let cancel = CancellationToken::new();
		let task_cancel = cancel.clone();
		let summary = Arc::new(Mutex::new(None));
		let task_summary = Arc::clone(&summary);
		let (tx, rx) = mpsc::unbounded();
		tokio::spawn(async move {
			let chunk_tx = tx.clone();
			let result = worker
				.run(task_cancel.clone(), move |state, cancel| {
					let loaded = state
						.models
						.get(&id)
						.ok_or_else(|| Error::backend("llama.cpp", "model was unloaded"))?;
					generate(&state.backend, loaded, &messages, options, cancel, &mut |chunk| {
						chunk_tx.unbounded_send(Ok(chunk)).is_ok()
					})
				})
				.await;
			match result {
				Ok(report) => {
					*task_summary.lock() = Some(report);
				},
				Err(error) if !matches!(error, Error::Cancelled) => {
					let _ = tx.unbounded_send(Err(error));
				},
				Err(_) => {},
			}
		});
		Ok(GenerationStream { rx, cancel, summary })
	}

	/// Unloads this model immediately; all clones become unusable.
	pub async fn shutdown(&self) -> Result<()> {
		let id = self.handle.id;
		self
			.handle
			.worker
			.run_uncancelled(move |state| {
				state.models.remove(&id);
				Ok(())
			})
			.await
	}
}

/// Why one generation stopped producing tokens.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GenerationStop {
	/// The model emitted its end-of-generation token or the receiver went away.
	EndTurn,
	/// The configured `max_tokens` budget was exhausted.
	MaxTokens,
	/// A configured stop sequence was reached and withheld from the output.
	StopSequence,
}

/// Accounting produced by one completed generation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GenerationSummary {
	/// Tokens in the chat-templated prompt.
	pub prompt_tokens: u64,
	/// Tokens sampled during generation, including any withheld by a stop
	/// sequence.
	pub output_tokens: u64,
	/// Why generation stopped.
	pub stop:          GenerationStop,
}

/// Token stream produced by [`TextGenerator::stream`].
pub struct GenerationStream {
	rx:      mpsc::UnboundedReceiver<Result<SmolStr>>,
	cancel:  CancellationToken,
	summary: Arc<Mutex<Option<GenerationSummary>>>,
}

impl GenerationStream {
	/// Cancels generation and closes the stream after the active llama.cpp
	/// decode pass.
	pub fn cancel(&self) {
		self.cancel.cancel();
	}

	/// Token accounting and stop reason, available once the stream has ended
	/// without an error.
	pub fn summary(&self) -> Option<GenerationSummary> {
		*self.summary.lock()
	}
}

impl Stream for GenerationStream {
	type Item = Result<SmolStr>;

	fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		Pin::new(&mut self.rx).poll_next(context)
	}
}

impl Drop for GenerationStream {
	fn drop(&mut self) {
		self.cancel.cancel();
	}
}

async fn llama_worker() -> Result<Worker<LlamaState>> {
	let worker = LLAMA_WORKER
		.get_or_try_init(|| async {
			Worker::spawn("omp-llm-local-llama", || {
				let mut backend =
					LlamaBackend::init().map_err(|error| Error::backend("llama.cpp", error))?;
				backend.void_logs();
				Ok(LlamaState { backend, models: HashMap::new(), next_id: 1 })
			})
			.await
		})
		.await?;
	Ok(worker.clone())
}

async fn resolve_model(model: TextModel, hub: &Hub) -> Result<PathBuf> {
	match model {
		TextModel::Small(model) => {
			let (repo, filename) = model.source();
			hub.file(&repo, filename.as_str()).await
		},
		TextModel::HuggingFace { repo, files } => {
			if files.is_empty() {
				return Err(Error::invalid("at least one GGUF file is required"));
			}
			let paths = hub.files(&repo, files).await?;
			paths
				.into_iter()
				.next()
				.ok_or_else(|| Error::backend("hugging face", "model download returned no files"))
		},
		TextModel::Local(path) => Ok(path),
	}
}

fn select_accelerator(
	preference: DevicePreference,
	backend: &LlamaBackend,
) -> Result<(bool, Accelerator)> {
	let gpu_available = backend.supports_gpu_offload();
	match preference {
		DevicePreference::Cpu => Ok((false, Accelerator::Cpu)),
		DevicePreference::Auto if !gpu_available => Ok((false, Accelerator::Cpu)),
		DevicePreference::Auto | DevicePreference::Gpu => {
			if gpu_available {
				Ok((true, native_gpu_accelerator()))
			} else {
				Err(Error::unavailable("llama.cpp found no usable GPU backend"))
			}
		},
		DevicePreference::Metal => require_metal(gpu_available),
		DevicePreference::Cuda => require_cuda(gpu_available),
	}
}

#[cfg(target_os = "macos")]
const fn native_gpu_accelerator() -> Accelerator {
	Accelerator::Metal
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
fn native_gpu_accelerator() -> Accelerator {
	Accelerator::Cuda
}

#[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
fn native_gpu_accelerator() -> Accelerator {
	Accelerator::Cpu
}

#[cfg(target_os = "macos")]
fn require_metal(available: bool) -> Result<(bool, Accelerator)> {
	if available {
		Ok((true, Accelerator::Metal))
	} else {
		Err(Error::unavailable("llama.cpp Metal backend is unavailable"))
	}
}

#[cfg(not(target_os = "macos"))]
fn require_metal(_available: bool) -> Result<(bool, Accelerator)> {
	Err(Error::unavailable("this target was not compiled with Metal"))
}

#[cfg(feature = "cuda")]
fn require_cuda(available: bool) -> Result<(bool, Accelerator)> {
	if available {
		Ok((true, Accelerator::Cuda))
	} else {
		Err(Error::unavailable("llama.cpp CUDA backend is unavailable"))
	}
}

#[cfg(not(feature = "cuda"))]
fn require_cuda(_available: bool) -> Result<(bool, Accelerator)> {
	Err(Error::unavailable("enable the omp-llm-local `cuda` feature"))
}

#[derive(Default)]
struct TokenUtf8Decoder {
	pending:     [u8; 4],
	pending_len: usize,
	output:      String,
}

impl TokenUtf8Decoder {
	fn push(&mut self, mut bytes: &[u8]) -> &str {
		self.output.clear();
		if self.pending_len != 0 {
			let sequence_len = Utf8::run_length(self.pending[0]);
			let copied = (sequence_len - self.pending_len).min(bytes.len());
			self.pending[self.pending_len..self.pending_len + copied]
				.copy_from_slice(&bytes[..copied]);
			self.pending_len += copied;
			bytes = &bytes[copied..];
			if self.pending_len < sequence_len {
				return &self.output;
			}
			let pending = self.pending;
			self.pending_len = 0;
			Self::decode(
				&mut self.pending,
				&mut self.pending_len,
				&pending[..sequence_len],
				&mut self.output,
			);
		}
		Self::decode(&mut self.pending, &mut self.pending_len, bytes, &mut self.output);
		&self.output
	}

	fn decode(pending: &mut [u8; 4], pending_len: &mut usize, mut bytes: &[u8], text: &mut String) {
		while !bytes.is_empty() {
			match std::str::from_utf8(bytes) {
				Ok(valid) => {
					text.push_str(valid);
					return;
				},
				Err(error) => {
					let valid_len = error.valid_up_to();
					let invalid_len = error.error_len();
					text.push_str(
						std::str::from_utf8(&bytes[..valid_len])
							.expect("UTF-8 validator reported an invalid prefix"),
					);
					bytes = &bytes[valid_len..];
					if let Some(invalid_len) = invalid_len {
						text.push(char::REPLACEMENT_CHARACTER);
						bytes = &bytes[invalid_len..];
					} else {
						pending[..bytes.len()].copy_from_slice(bytes);
						*pending_len = bytes.len();
						return;
					}
				},
			}
		}
	}
}

fn token_piece(model: &LlamaModel, token: LlamaToken) -> Result<Vec<u8>> {
	let bytes = match model.token_to_piece_bytes(token, 8, true, None) {
		Err(TokenToStringError::InsufficientBufferSpace(size)) => model.token_to_piece_bytes(
			token,
			usize::try_from(size.unsigned_abs())
				.expect("llama.cpp reported a token size unsupported by usize"),
			true,
			None,
		),
		result => result,
	};
	bytes.map_err(|error| Error::backend("llama.cpp tokenizer", error))
}

fn chat_tokens(loaded: &LoadedModel, messages: &[ChatMessage]) -> Result<Vec<LlamaToken>> {
	let chat = messages
		.iter()
		.map(|message| {
			let role = match message.role {
				ChatRole::System => "system",
				ChatRole::User => "user",
				ChatRole::Assistant => "assistant",
			};
			LlamaChatMessage::new(role.to_owned(), message.content.to_string())
				.map_err(|error| Error::backend("llama.cpp chat", error))
		})
		.collect::<Result<Vec<_>>>()?;
	let template = loaded
		.model
		.chat_template(None)
		.map_err(|error| Error::backend("llama.cpp chat template", error))?;
	let prompt = loaded
		.model
		.apply_chat_template(&template, &chat, true)
		.map_err(|error| Error::backend("llama.cpp chat template", error))?;
	let tokens = loaded
		.model
		.str_to_token(&prompt, AddBos::Always)
		.map_err(|error| Error::backend("llama.cpp tokenizer", error))?;
	if tokens.is_empty() {
		return Err(Error::backend("llama.cpp tokenizer", "chat template produced no tokens"));
	}
	Ok(tokens)
}

fn generate(
	backend: &LlamaBackend,
	loaded: &LoadedModel,
	messages: &[ChatMessage],
	options: GenerationOptions,
	cancel: &CancellationToken,
	emit: &mut dyn FnMut(SmolStr) -> bool,
) -> Result<GenerationSummary> {
	let tokens = chat_tokens(loaded, messages)?;
	if tokens.len().saturating_add(options.max_tokens) > loaded.context_size.get() as usize {
		return Err(Error::invalid("prompt and requested output exceed the configured context"));
	}

	let threads = i32::try_from(loaded.threads)
		.map_err(|_| Error::invalid("text-generation thread count exceeds i32"))?;
	let batch_size = loaded.context_size.get().min(2048);
	let params = LlamaContextParams::default()
		.with_n_ctx(Some(loaded.context_size))
		.with_n_batch(batch_size)
		.with_n_threads(threads)
		.with_n_threads_batch(threads);
	let mut context = loaded
		.model
		.new_context(backend, params)
		.map_err(|error| Error::backend("llama.cpp context", error))?;
	let mut batch = LlamaBatch::new(batch_size as usize, 1);
	let last_token = tokens.len() - 1;
	for (chunk_index, chunk) in tokens.chunks(batch_size as usize).enumerate() {
		batch.clear();
		let offset = chunk_index * batch_size as usize;
		for (index, token) in chunk.iter().enumerate() {
			let absolute = offset + index;
			let position = i32::try_from(absolute)
				.map_err(|_| Error::invalid("prompt position exceeds llama.cpp limits"))?;
			batch
				.add(*token, position, &[0], absolute == last_token)
				.map_err(|error| Error::backend("llama.cpp batch", error))?;
		}
		context
			.decode(&mut batch)
			.map_err(|error| Error::backend("llama.cpp", error))?;
	}

	let mut sampler = sampler(&options)?;
	sampler.accept_many(&tokens);
	let mut decoder = TokenUtf8Decoder::default();
	let mut position = i32::try_from(tokens.len())
		.map_err(|_| Error::invalid("prompt length exceeds llama.cpp limits"))?;
	let mut filter = StopFilter::new(options.stop);
	let prompt_tokens = tokens.len() as u64;
	let mut output_tokens = 0_u64;
	let mut stop = GenerationStop::EndTurn;
	for generated in 0..options.max_tokens {
		if cancel.is_cancelled() {
			return Err(Error::Cancelled);
		}
		let token = sampler.sample(&context, batch.n_tokens() - 1);
		if loaded.model.is_eog_token(token) {
			break;
		}
		output_tokens += 1;
		let piece = decoder.push(&token_piece(&loaded.model, token)?);
		if let Some(chunk) = filter.push(piece) {
			if !chunk.is_empty() && !emit(chunk) {
				return Ok(GenerationSummary { prompt_tokens, output_tokens, stop });
			}
			if filter.stopped() {
				stop = GenerationStop::StopSequence;
				return Ok(GenerationSummary { prompt_tokens, output_tokens, stop });
			}
		}
		if generated + 1 == options.max_tokens {
			stop = GenerationStop::MaxTokens;
			break;
		}
		batch.clear();
		batch
			.add(token, position, &[0], true)
			.map_err(|error| Error::backend("llama.cpp batch", error))?;
		position = position
			.checked_add(1)
			.ok_or_else(|| Error::invalid("generated position exceeds llama.cpp limits"))?;
		context
			.decode(&mut batch)
			.map_err(|error| Error::backend("llama.cpp", error))?;
	}
	if let Some(chunk) = filter.finish()
		&& !chunk.is_empty()
	{
		let _ = emit(chunk);
	}
	Ok(GenerationSummary { prompt_tokens, output_tokens, stop })
}

fn sampler(options: &GenerationOptions) -> Result<LlamaSampler> {
	let mut samplers = Vec::with_capacity(6);
	if options.repetition_penalty.is_some()
		|| options.frequency_penalty.is_some()
		|| options.presence_penalty.is_some()
	{
		samplers.push(LlamaSampler::penalties(
			-1,
			options.repetition_penalty.unwrap_or(1.0),
			options.frequency_penalty.unwrap_or(0.0),
			options.presence_penalty.unwrap_or(0.0),
		));
	}
	if let Some(top_k) = options.top_k {
		let top_k = i32::try_from(top_k).map_err(|_| Error::invalid("top-k exceeds i32"))?;
		samplers.push(LlamaSampler::top_k(top_k));
	}
	if let Some(top_p) = options.top_p {
		samplers.push(LlamaSampler::top_p(top_p, 1));
	}
	if let Some(min_p) = options.min_p {
		samplers.push(LlamaSampler::min_p(min_p, 1));
	}
	if let Some(temperature) = options.temperature.filter(|value| *value > 0.0) {
		samplers.push(LlamaSampler::temp(temperature));
		samplers.push(LlamaSampler::dist(options.seed));
	} else {
		samplers.push(LlamaSampler::greedy());
	}
	Ok(LlamaSampler::chain_simple(samplers))
}

fn validate_request(messages: &[ChatMessage], options: &GenerationOptions) -> Result<()> {
	if messages.is_empty() {
		return Err(Error::invalid("at least one chat message is required"));
	}
	if options.max_tokens == 0 {
		return Err(Error::invalid("maximum generated tokens must be non-zero"));
	}
	if options
		.temperature
		.is_some_and(|value| !value.is_finite() || value < 0.0)
	{
		return Err(Error::invalid("temperature must be finite and non-negative"));
	}
	if options
		.top_p
		.is_some_and(|value| !value.is_finite() || value <= 0.0 || value > 1.0)
	{
		return Err(Error::invalid("top-p must be in (0, 1]"));
	}
	if options.top_k == Some(0) {
		return Err(Error::invalid("top-k must be non-zero"));
	}
	if options
		.min_p
		.is_some_and(|value| !value.is_finite() || value < 0.0 || value > 1.0)
	{
		return Err(Error::invalid("min-p must be in [0, 1]"));
	}
	if options
		.repetition_penalty
		.is_some_and(|value| !value.is_finite() || value <= 0.0)
	{
		return Err(Error::invalid("repetition penalty must be finite and greater than zero"));
	}
	if options
		.frequency_penalty
		.is_some_and(|value| !value.is_finite())
	{
		return Err(Error::invalid("frequency penalty must be finite"));
	}
	if options
		.presence_penalty
		.is_some_and(|value| !value.is_finite())
	{
		return Err(Error::invalid("presence penalty must be finite"));
	}
	if options.stop.iter().any(|stop| stop.is_empty()) {
		return Err(Error::invalid("stop sequences must not be empty"));
	}
	Ok(())
}

struct StopFilter {
	pending: String,
	stops:   Vec<SmolStr>,
	stopped: bool,
}

impl StopFilter {
	const fn new(stops: Vec<SmolStr>) -> Self {
		Self { pending: String::new(), stops, stopped: false }
	}

	fn push(&mut self, piece: &str) -> Option<SmolStr> {
		self.pending.push_str(piece);
		let earliest = self
			.stops
			.iter()
			.filter_map(|stop| self.pending.find(stop.as_str()))
			.min();
		if let Some(index) = earliest {
			self.stopped = true;
			let output: SmolStr = self.pending[..index].into();
			self.pending.clear();
			return Some(output);
		}
		let keep = self
			.pending
			.char_indices()
			.map(|(index, _)| &self.pending[index..])
			.filter(|suffix| self.stops.iter().any(|stop| stop.starts_with(suffix)))
			.map(str::len)
			.max()
			.unwrap_or(0);
		let emit_len = self.pending.len() - keep;
		if emit_len == 0 {
			return None;
		}
		let output: SmolStr = self.pending[..emit_len].into();
		self.pending.drain(..emit_len);
		Some(output)
	}

	const fn stopped(&self) -> bool {
		self.stopped
	}

	fn finish(&mut self) -> Option<SmolStr> {
		(!self.pending.is_empty()).then(|| std::mem::take(&mut self.pending).into())
	}
}

#[cfg(test)]
mod tests {
	use super::{ChatMessage, GenerationOptions, StopFilter, TokenUtf8Decoder, validate_request};
	use crate::Error;

	#[test]
	fn token_decoder_preserves_codepoints_split_across_tokens() {
		let mut decoder = TokenUtf8Decoder::default();
		assert_eq!(decoder.push(b"before \xf0\x9f\x98"), "before ");
		assert_eq!(decoder.push(b"\x80 after"), "😀 after");
	}

	#[test]
	fn token_decoder_replaces_malformed_sequences_without_losing_text() {
		let mut decoder = TokenUtf8Decoder::default();
		assert_eq!(decoder.push(b"\xe2"), "");
		assert_eq!(decoder.push(b"(ok"), "\u{fffd}(ok");
	}

	#[test]
	fn stop_filter_withholds_markers_split_across_tokens() {
		let mut filter = StopFilter::new(vec!["</title>".into()]);
		assert_eq!(filter.push("Result</ti").as_deref(), Some("Result"));
		assert!(
			filter
				.push("tle>ignored")
				.is_some_and(|chunk| chunk.is_empty())
		);
		assert!(filter.stopped());
	}

	#[test]
	fn stop_filter_flushes_partial_nonmatch() {
		let mut filter = StopFilter::new(vec!["stop".into()]);
		assert_eq!(filter.push("hello st").as_deref(), Some("hello "));
		assert_eq!(filter.push("art").as_deref(), Some("start"));
		assert_eq!(filter.finish(), None);
	}

	#[test]
	fn generation_validation_rejects_ambiguous_or_unbounded_requests() {
		let mut options = GenerationOptions::default();
		assert!(matches!(validate_request(&[], &options), Err(Error::InvalidInput(_))));
		options.max_tokens = 0;
		assert!(matches!(
			validate_request(&[ChatMessage::user("hello")], &options),
			Err(Error::InvalidInput(_))
		));
		options.max_tokens = 1;
		options.stop.push("".into());
		assert!(matches!(
			validate_request(&[ChatMessage::user("hello")], &options),
			Err(Error::InvalidInput(_))
		));
	}
}
