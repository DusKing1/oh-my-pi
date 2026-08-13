//! llama.cpp-backed local text generation.

use std::{num::NonZeroU32, path::PathBuf, sync::Arc, time::Duration};

use llama_cpp_2::{
	TokenToStringError,
	context::params::LlamaContextParams,
	llama_backend::LlamaBackend,
	llama_batch::LlamaBatch,
	model::{AddBos, LlamaChatMessage, LlamaModel, params::LlamaModelParams},
	sampling::LlamaSampler,
	token::LlamaToken,
};
use omp_core::Str;
use xutf::{Encoding, Utf8};

use super::runtime::{
	LocalCancellation, LocalError, LocalErrorKind, LocalExecutionReceipt, LocalResult, LocalRuntime,
	MemoryPool,
};

/// Role of one local chat message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatRole {
	/// System instruction.
	System,
	/// User input.
	User,
	/// Prior assistant output.
	Assistant,
}

/// Owned message passed to llama.cpp's embedded chat template.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatMessage {
	/// Message role.
	pub role:    ChatRole,
	/// Message content.
	pub content: Str,
}

/// Local GGUF model and lifecycle configuration.
#[derive(Clone, Debug)]
pub struct TextConfig {
	/// Local GGUF model path.
	pub model_path:      PathBuf,
	/// Context allocation in tokens.
	pub context_tokens:  u32,
	/// llama.cpp CPU thread count.
	pub threads:         usize,
	/// Transformer layers offloaded to the compiled GPU backend.
	pub gpu_layers:      u32,
	/// Estimated resident bytes charged before loading.
	pub resident_bytes:  usize,
	/// Admission limit; currently must be one because llama.cpp access is
	/// serialized.
	pub max_concurrency: usize,
	/// Explicit idle-unload interval.
	pub idle_timeout:    Duration,
}

/// Sampling controls implemented by the llama.cpp adapter.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationOptions {
	/// Maximum newly generated tokens.
	pub max_tokens:  usize,
	/// Optional non-negative temperature; zero is greedy.
	pub temperature: Option<f32>,
	/// Sampling seed.
	pub seed:        u32,
	/// Stop sequences withheld from output.
	pub stop:        Vec<Str>,
}

impl Default for GenerationOptions {
	fn default() -> Self {
		Self { max_tokens: 256, temperature: None, seed: 0, stop: Vec::new() }
	}
}

/// Honest feature evidence for this adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextCapabilities {
	/// Token deltas can be delivered incrementally.
	pub streaming:             bool,
	/// Native tool definitions and calls are supported.
	pub tools:                 bool,
	/// Native schema-constrained generation is supported.
	pub structured_generation: bool,
}

/// Text generation result and usage estimates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextGeneration {
	/// Complete generated text.
	pub content:       Str,
	/// Exact prompt token count from llama.cpp.
	pub prompt_tokens: u64,
	/// Exact generated token count from llama.cpp.
	pub output_tokens: u64,
	/// Runtime isolation evidence.
	pub receipt:       LocalExecutionReceipt,
}

struct LlamaEngine {
	backend:        LlamaBackend,
	model:          LlamaModel,
	context_tokens: NonZeroU32,
	threads:        usize,
}

/// Lazy, bounded GGUF text adapter using llama.cpp in-process.
#[derive(Clone)]
pub struct TextAdapter {
	runtime: LocalRuntime<LlamaEngine>,
}

impl TextAdapter {
	/// Creates a lazy adapter for a local GGUF artifact.
	pub fn new(config: TextConfig, memory: Arc<MemoryPool>) -> LocalResult<Self> {
		let context_tokens = NonZeroU32::new(config.context_tokens).ok_or_else(|| {
			LocalError::new(LocalErrorKind::InvalidInput, "text context must be non-zero")
		})?;
		if config.threads == 0 {
			return Err(LocalError::new(
				LocalErrorKind::InvalidInput,
				"text thread count must be non-zero",
			));
		}
		let resident = config.resident_bytes;
		let concurrency = config.max_concurrency;
		let idle = config.idle_timeout;
		let runtime = LocalRuntime::new(
			move || {
				let backend = LlamaBackend::init().map_err(|error| {
					LocalError::new(LocalErrorKind::Backend, format!("llama.cpp init failed: {error}"))
				})?;
				let params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
				let model = LlamaModel::load_from_file(&backend, &config.model_path, &params).map_err(
					|error| {
						LocalError::new(LocalErrorKind::Backend, format!("GGUF load failed: {error}"))
					},
				)?;
				Ok(LlamaEngine { backend, model, context_tokens, threads: config.threads })
			},
			memory,
			resident,
			concurrency,
			idle,
		)?;
		Ok(Self { runtime })
	}

	/// Reports exactly the features implemented by this adapter.
	pub const fn capabilities() -> TextCapabilities {
		TextCapabilities {
			streaming:             true,
			tools:                 false,
			structured_generation: false,
		}
	}

	/// Generates text and calls `on_delta` synchronously for backpressured
	/// delivery.
	pub fn generate(
		&self,
		messages: &[ChatMessage],
		options: GenerationOptions,
		cancel: &LocalCancellation,
		mut on_delta: impl FnMut(&str) -> bool,
	) -> LocalResult<TextGeneration> {
		validate_request(messages, &options)?;
		let lease = self.runtime.acquire(cancel)?;
		let receipt = lease.receipt();
		let mut content = String::new();
		let (prompt_tokens, output_tokens) = lease.with_engine(|engine| {
			generate(engine, messages, &options, cancel, &mut |delta| {
				content.push_str(delta);
				on_delta(delta)
			})
		})?;
		Ok(TextGeneration { content: content.into(), prompt_tokens, output_tokens, receipt })
	}

	/// Unloads the model when no call is active and its idle interval elapsed.
	pub fn unload_if_idle(&self, now: std::time::Instant) -> bool {
		self.runtime.unload_if_idle(now)
	}

	/// Returns whether the GGUF model is resident.
	pub fn is_loaded(&self) -> bool {
		self.runtime.is_loaded()
	}
}

fn generate(
	engine: &mut LlamaEngine,
	messages: &[ChatMessage],
	options: &GenerationOptions,
	cancel: &LocalCancellation,
	emit: &mut dyn FnMut(&str) -> bool,
) -> LocalResult<(u64, u64)> {
	let chat = messages
		.iter()
		.map(|message| {
			let role = match message.role {
				ChatRole::System => "system",
				ChatRole::User => "user",
				ChatRole::Assistant => "assistant",
			};
			LlamaChatMessage::new(role.to_owned(), message.content.to_string()).map_err(|error| {
				LocalError::new(LocalErrorKind::InvalidInput, format!("chat message failed: {error}"))
			})
		})
		.collect::<LocalResult<Vec<_>>>()?;
	let template = engine.model.chat_template(None).map_err(|error| {
		LocalError::new(LocalErrorKind::Backend, format!("chat template failed: {error}"))
	})?;
	let prompt = engine
		.model
		.apply_chat_template(&template, &chat, true)
		.map_err(|error| {
			LocalError::new(LocalErrorKind::Backend, format!("chat rendering failed: {error}"))
		})?;
	let tokens = engine
		.model
		.str_to_token(&prompt, AddBos::Always)
		.map_err(|error| {
			LocalError::new(LocalErrorKind::Backend, format!("tokenization failed: {error}"))
		})?;
	if tokens.is_empty()
		|| tokens.len().saturating_add(options.max_tokens) > engine.context_tokens.get() as usize
	{
		return Err(LocalError::new(
			LocalErrorKind::InvalidInput,
			"prompt and output exceed the configured context",
		));
	}
	let threads = i32::try_from(engine.threads).map_err(|_| {
		LocalError::new(LocalErrorKind::InvalidInput, "text thread count exceeds i32")
	})?;
	let batch_size = engine.context_tokens.get().min(2048);
	let params = LlamaContextParams::default()
		.with_n_ctx(Some(engine.context_tokens))
		.with_n_batch(batch_size)
		.with_n_threads(threads)
		.with_n_threads_batch(threads);
	let mut context = engine
		.model
		.new_context(&engine.backend, params)
		.map_err(|error| {
			LocalError::new(LocalErrorKind::Backend, format!("llama context failed: {error}"))
		})?;
	let mut batch = LlamaBatch::new(batch_size as usize, 1);
	let last = tokens.len() - 1;
	for (chunk_index, chunk) in tokens.chunks(batch_size as usize).enumerate() {
		batch.clear();
		let offset = chunk_index * batch_size as usize;
		for (index, token) in chunk.iter().enumerate() {
			let absolute = offset + index;
			batch
				.add(*token, absolute as i32, &[0], absolute == last)
				.map_err(|error| {
					LocalError::new(LocalErrorKind::Backend, format!("llama batch failed: {error}"))
				})?;
		}
		context.decode(&mut batch).map_err(|error| {
			LocalError::new(LocalErrorKind::Backend, format!("prompt decode failed: {error}"))
		})?;
	}
	let mut sampler = if options.temperature.is_some_and(|value| value > 0.0) {
		LlamaSampler::chain_simple(vec![
			LlamaSampler::temp(options.temperature.unwrap_or_default()),
			LlamaSampler::dist(options.seed),
		])
	} else {
		LlamaSampler::greedy()
	};
	sampler.accept_many(&tokens);
	let mut position = tokens.len() as i32;
	let mut output_tokens = 0_u64;
	let mut decoder = TokenUtf8Decoder::default();
	let mut pending = String::new();
	for generated in 0..options.max_tokens {
		if cancel.is_cancelled() {
			return Err(LocalError::cancelled());
		}
		let token = sampler.sample(&context, batch.n_tokens() - 1);
		if engine.model.is_eog_token(token) {
			break;
		}
		output_tokens += 1;
		pending.push_str(decoder.push(&token_piece(&engine.model, token)?));
		if let Some(stop) = options
			.stop
			.iter()
			.filter_map(|stop| pending.find(stop.as_str()))
			.min()
		{
			let visible = &pending[..stop];
			if !visible.is_empty() {
				let _ = emit(visible);
			}
			return Ok((tokens.len() as u64, output_tokens));
		}
		let keep = options
			.stop
			.iter()
			.map(|stop| stop.len().min(pending.len()))
			.max()
			.unwrap_or(0);
		if pending.len() > keep {
			let split = pending.len() - keep;
			if pending.is_char_boundary(split) {
				let visible = pending[..split].to_owned();
				pending.drain(..split);
				if !visible.is_empty() && !emit(&visible) {
					return Ok((tokens.len() as u64, output_tokens));
				}
			}
		}
		if generated + 1 == options.max_tokens {
			break;
		}
		batch.clear();
		batch.add(token, position, &[0], true).map_err(|error| {
			LocalError::new(LocalErrorKind::Backend, format!("generation batch failed: {error}"))
		})?;
		position += 1;
		context.decode(&mut batch).map_err(|error| {
			LocalError::new(LocalErrorKind::Backend, format!("generation decode failed: {error}"))
		})?;
	}
	if !pending.is_empty() {
		let _ = emit(&pending);
	}
	Ok((tokens.len() as u64, output_tokens))
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

	fn decode(
		pending: &mut [u8; 4],
		pending_len: &mut usize,
		mut bytes: &[u8],
		output: &mut String,
	) {
		while !bytes.is_empty() {
			match std::str::from_utf8(bytes) {
				Ok(valid) => {
					output.push_str(valid);
					return;
				},
				Err(error) => {
					let valid_len = error.valid_up_to();
					output.push_str(
						std::str::from_utf8(&bytes[..valid_len])
							.expect("UTF-8 validator reported an invalid prefix"),
					);
					bytes = &bytes[valid_len..];
					if let Some(invalid_len) = error.error_len() {
						output.push(char::REPLACEMENT_CHARACTER);
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

fn token_piece(model: &LlamaModel, token: LlamaToken) -> LocalResult<Vec<u8>> {
	let bytes = match model.token_to_piece_bytes(token, 8, true, None) {
		Err(TokenToStringError::InsufficientBufferSpace(size)) => model.token_to_piece_bytes(
			token,
			usize::try_from(size.unsigned_abs()).unwrap_or(usize::MAX),
			true,
			None,
		),
		result => result,
	};
	bytes.map_err(|error| {
		LocalError::new(LocalErrorKind::Backend, format!("token decoding failed: {error}"))
	})
}

fn validate_request(messages: &[ChatMessage], options: &GenerationOptions) -> LocalResult<()> {
	if messages.is_empty() || options.max_tokens == 0 || options.stop.iter().any(Str::is_empty) {
		return Err(LocalError::new(
			LocalErrorKind::InvalidInput,
			"generation requires messages, tokens, and non-empty stop sequences",
		));
	}
	if options
		.temperature
		.is_some_and(|value| !value.is_finite() || value < 0.0)
	{
		return Err(LocalError::new(
			LocalErrorKind::InvalidInput,
			"temperature must be finite and non-negative",
		));
	}
	Ok(())
}
