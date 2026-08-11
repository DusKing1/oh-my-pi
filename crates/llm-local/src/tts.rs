use std::{
	collections::HashMap,
	path::PathBuf,
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use futures::{Stream, channel::mpsc};
use omp_core::Str;
use parking_lot::RwLock;
use tokio_util::sync::CancellationToken;
use voice_kokoro::{KModel, ModelConfig, SynthesisMode};

use crate::{
	Accelerator, Audio, DevicePreference, Error, Hub, ModelRepo, Result, device::candle_device,
	worker::Worker,
};

const KOKORO_REPO: &str = "prince-canuma/Kokoro-82M";
const KOKORO_CONFIG: &str = "config.json";
const KOKORO_WEIGHTS: &str = "kokoro-v1_0.safetensors";

/// Curated Kokoro voices available in the default model repository.
pub const KOKORO_VOICES: &[&str] = &[
	"af_alloy",
	"af_heart",
	"af_bella",
	"af_nicole",
	"af_jessica",
	"af_aoede",
	"af_kore",
	"af_sarah",
	"af_nova",
	"af_sky",
	"af_river",
	"am_adam",
	"am_michael",
	"am_fenrir",
	"am_puck",
	"am_echo",
	"am_eric",
	"am_liam",
	"am_onyx",
	"am_santa",
	"bf_emma",
	"bf_isabella",
	"bf_alice",
	"bf_lily",
	"bm_george",
	"bm_lewis",
	"bm_daniel",
	"bm_fable",
];

/// Voice pack name within a Kokoro repository.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct KokoroVoice(Str);

impl KokoroVoice {
	/// Validates a repository voice name such as `af_heart`.
	pub fn new(name: impl Into<Str>) -> Result<Self> {
		let name = name.into();
		if name.is_empty()
			|| !name
				.bytes()
				.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
		{
			return Err(Error::invalid(
				"Kokoro voice names may contain only lowercase ASCII letters, digits, and underscores",
			));
		}
		Ok(Self(name))
	}

	/// Repository voice identifier without the `voices/` prefix or extension.
	pub fn name(&self) -> &str {
		self.0.as_str()
	}
}

impl Default for KokoroVoice {
	fn default() -> Self {
		Self("af_heart".into())
	}
}

impl TryFrom<&str> for KokoroVoice {
	type Error = Error;

	fn try_from(value: &str) -> Result<Self> {
		Self::new(value)
	}
}

/// Controls one Kokoro synthesis request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynthesisOptions {
	/// Speaking-rate multiplier; `1.0` is the model's natural rate.
	pub speed:           f32,
	/// Maximum approximate characters per model pass before text is split at
	/// word boundaries.
	pub max_chunk_chars: usize,
	/// Removes decoder noise so repeated calls produce byte-stable audio.
	pub deterministic:   bool,
}

impl Default for SynthesisOptions {
	fn default() -> Self {
		Self { speed: 1.0, max_chunk_chars: 400, deterministic: false }
	}
}

/// Configures a Kokoro model, model source, and accelerator.
#[derive(Clone)]
pub struct KokoroBuilder {
	hub:          Option<Hub>,
	repo:         ModelRepo,
	config_file:  Str,
	weights_file: Str,
	device:       DevicePreference,
}

impl KokoroBuilder {
	/// Creates a builder using Kokoro-82M v1.0 from Hugging Face.
	pub fn new() -> Self {
		Self {
			hub:          None,
			repo:         ModelRepo::new(KOKORO_REPO),
			config_file:  KOKORO_CONFIG.into(),
			weights_file: KOKORO_WEIGHTS.into(),
			device:       DevicePreference::Auto,
		}
	}

	/// Shares Hugging Face cache, authentication, endpoint, and offline policy.
	pub fn hub(mut self, hub: Hub) -> Self {
		self.hub = Some(hub);
		self
	}

	/// Overrides the model repository and its config and safetensors filenames.
	pub fn source(
		mut self,
		repo: ModelRepo,
		config_file: impl Into<Str>,
		weights_file: impl Into<Str>,
	) -> Self {
		self.repo = repo;
		self.config_file = config_file.into();
		self.weights_file = weights_file.into();
		self
	}

	/// Selects CPU, Metal, or CUDA execution.
	pub const fn device(mut self, device: DevicePreference) -> Self {
		self.device = device;
		self
	}

	/// Fetches and memory-maps Kokoro on a dedicated native thread.
	pub async fn build(self) -> Result<Kokoro> {
		let hub = match self.hub {
			Some(hub) => hub,
			None => Hub::new()?,
		};
		let paths = hub
			.files(&self.repo, [self.config_file.clone(), self.weights_file.clone()])
			.await?;
		let [config_path, weights_path]: [PathBuf; 2] = paths
			.try_into()
			.map_err(|_| Error::backend("kokoro", "model download returned the wrong file count"))?;
		let (device, accelerator) = candle_device(self.device)?;
		let worker = Worker::spawn("omp-llm-local-kokoro", move || {
			let config_bytes = std::fs::read(config_path)?;
			let config: ModelConfig = serde_json::from_slice(&config_bytes)
				.map_err(|error| Error::backend("kokoro config", error))?;
			// SAFETY: Candle owns the mappings for the lifetime of every tensor built from
			// this builder.
			let variables =
				unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device) }
					.map_err(|error| Error::backend("kokoro weights", error))?;
			let model =
				KModel::load(&config, variables).map_err(|error| Error::backend("kokoro", error))?;
			let g2p = voice_g2p::G2P::new();
			Ok(KokoroRuntime { model, config, device, voices: HashMap::new(), g2p })
		})
		.await?;
		Ok(Kokoro {
			worker,
			hub,
			repo: self.repo,
			voice_paths: Arc::new(RwLock::new(HashMap::new())),
			accelerator,
		})
	}
}

impl Default for KokoroBuilder {
	fn default() -> Self {
		Self::new()
	}
}

struct KokoroRuntime {
	model:  KModel,
	config: ModelConfig,
	device: Device,
	voices: HashMap<Str, Tensor>,
	g2p:    voice_g2p::G2P,
}

/// Asynchronous, multi-voice Kokoro-82M text-to-speech model.
#[derive(Clone)]
pub struct Kokoro {
	worker:      Worker<KokoroRuntime>,
	hub:         Hub,
	repo:        ModelRepo,
	voice_paths: Arc<RwLock<HashMap<Str, PathBuf>>>,
	accelerator: Accelerator,
}

impl Kokoro {
	/// Starts a builder for the default Kokoro-82M repository.
	pub fn builder() -> KokoroBuilder {
		KokoroBuilder::new()
	}

	/// Backend selected for synthesis.
	pub const fn accelerator(&self) -> Accelerator {
		self.accelerator
	}

	/// Synthesizes one text buffer into mono 24 kHz PCM.
	pub async fn synthesize(
		&self,
		text: impl Into<String>,
		voice: &KokoroVoice,
		options: SynthesisOptions,
		cancel: CancellationToken,
	) -> Result<Audio> {
		validate_synthesis(&options)?;
		let text = text.into();
		if text.trim().is_empty() {
			return Err(Error::invalid("cannot synthesize empty text"));
		}
		let voice_path = self.voice_path(voice).await?;
		let voice_name = voice.0.clone();
		let (samples, sample_rate) = self
			.worker
			.run(cancel, move |runtime, cancel| {
				if !runtime.voices.contains_key(&voice_name) {
					let tensors = candle_core::safetensors::load(&voice_path, &runtime.device)
						.map_err(|error| Error::backend("kokoro voice", error))?;
					let tensor = tensors.into_values().next().ok_or_else(|| {
						Error::backend("kokoro voice", "voice pack contains no tensors")
					})?;
					runtime.voices.insert(voice_name.clone(), tensor);
				}
				let voice = runtime
					.voices
					.get(&voice_name)
					.ok_or_else(|| Error::backend("kokoro voice", "loaded voice disappeared"))?
					.clone();
				let samples = synthesize_text(runtime, &voice, &text, options, cancel)?;
				Ok((samples, runtime.config.sample_rate))
			})
			.await?;
		Ok(Audio::mono(samples, sample_rate))
	}

	/// Synthesizes one buffer with the default `af_heart` voice and natural
	/// speed.
	pub async fn speak(&self, text: impl Into<String>) -> Result<Audio> {
		self
			.synthesize(
				text,
				&KokoroVoice::default(),
				SynthesisOptions::default(),
				CancellationToken::new(),
			)
			.await
	}

	/// Synthesizes already-segmented text in order and yields each audio chunk
	/// as soon as it is ready.
	pub fn stream<I, S>(
		&self,
		segments: I,
		voice: KokoroVoice,
		options: SynthesisOptions,
	) -> SynthesisStream
	where
		I: IntoIterator<Item = S>,
		I::IntoIter: Send + 'static,
		S: Into<String> + Send + 'static,
	{
		let segments = segments.into_iter();
		let model = self.clone();
		let cancel = CancellationToken::new();
		let task_cancel = cancel.clone();
		let (tx, rx) = mpsc::unbounded();
		tokio::spawn(async move {
			for segment in segments {
				if task_cancel.is_cancelled() {
					break;
				}
				let result = model
					.synthesize(segment.into(), &voice, options, task_cancel.clone())
					.await;
				let failed = result.is_err();
				if tx.unbounded_send(result).is_err() || failed {
					break;
				}
			}
		});
		SynthesisStream { rx, cancel }
	}

	/// Stops the shared model worker; all clones become unusable after this
	/// call.
	pub async fn shutdown(&self) -> Result<()> {
		self.worker.shutdown().await
	}

	async fn voice_path(&self, voice: &KokoroVoice) -> Result<PathBuf> {
		let cached_path = self.voice_paths.read().get(&voice.0).cloned();
		if let Some(path) = cached_path {
			return Ok(path);
		}
		let filename = format!("voices/{}.safetensors", voice.name());
		let path = self.hub.file(&self.repo, filename).await?;
		self
			.voice_paths
			.write()
			.insert(voice.0.clone(), path.clone());
		Ok(path)
	}
}

/// Stream of independently playable mono 24 kHz synthesis chunks.
pub struct SynthesisStream {
	rx:     mpsc::UnboundedReceiver<Result<Audio>>,
	cancel: CancellationToken,
}

impl SynthesisStream {
	/// Cancels the active model pass and discards queued text segments.
	pub fn cancel(&self) {
		self.cancel.cancel();
	}
}

impl Stream for SynthesisStream {
	type Item = Result<Audio>;

	fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		Pin::new(&mut self.rx).poll_next(context)
	}
}

impl Drop for SynthesisStream {
	fn drop(&mut self) {
		self.cancel.cancel();
	}
}

fn validate_synthesis(options: &SynthesisOptions) -> Result<()> {
	if !options.speed.is_finite() || options.speed <= 0.0 {
		return Err(Error::invalid("synthesis speed must be finite and greater than zero"));
	}
	if options.max_chunk_chars == 0 {
		return Err(Error::invalid("synthesis chunk size must be non-zero"));
	}
	Ok(())
}

fn synthesize_text(
	runtime: &KokoroRuntime,
	voice: &Tensor,
	text: &str,
	options: SynthesisOptions,
	cancel: &CancellationToken,
) -> Result<Vec<f32>> {
	let mut output = Vec::new();
	for_each_text_chunk(text, options.max_chunk_chars, |chunk| {
		if cancel.is_cancelled() {
			return Err(Error::Cancelled);
		}
		let phonemes = runtime
			.g2p
			.convert(chunk)
			.map_err(|error| Error::backend("g2p", error))?;
		let mut token_ids = Vec::with_capacity(phonemes.chars().count());
		let mut encoded = [0_u8; 4];
		token_ids.extend(phonemes.chars().filter_map(|character| {
			runtime
				.config
				.vocab
				.get(character.encode_utf8(&mut encoded))
				.copied()
		}));
		if token_ids.is_empty() {
			return Ok(());
		}
		let pack_len = voice
			.dim(0)
			.map_err(|error| Error::backend("kokoro voice", error))?;
		if pack_len == 0 {
			return Err(Error::backend("kokoro voice", "voice pack contains no styles"));
		}
		let style = voice
			.i((token_ids.len() - 1).min(pack_len - 1))
			.and_then(|tensor| tensor.squeeze(0))
			.and_then(|tensor| tensor.unsqueeze(0))
			.map_err(|error| Error::backend("kokoro voice", error))?;
		let mode = if options.deterministic {
			SynthesisMode::Deterministic
		} else {
			SynthesisMode::Stochastic
		};
		let audio = runtime
			.model
			.forward_with_mode(&token_ids, &style, options.speed, &runtime.device, mode)
			.and_then(|tensor| tensor.to_vec1::<f32>())
			.map_err(|error| Error::backend("kokoro", error))?;
		output.extend(audio);
		Ok(())
	})?;
	if output.is_empty() {
		return Err(Error::backend("kokoro", "text produced no supported phonemes"));
	}
	Ok(output)
}

fn for_each_text_chunk<E>(
	text: &str,
	max_chars: usize,
	mut emit: impl FnMut(&str) -> std::result::Result<(), E>,
) -> std::result::Result<(), E> {
	let mut current = String::new();
	let mut current_chars = 0;
	for word in text.split_whitespace() {
		let word_chars = word.chars().count();
		if current_chars > 0 && current_chars + 1 + word_chars > max_chars {
			emit(&current)?;
			current.clear();
			current_chars = 0;
		}
		if word_chars > max_chars {
			for character in word.chars() {
				if current_chars == max_chars {
					emit(&current)?;
					current.clear();
					current_chars = 0;
				}
				current.push(character);
				current_chars += 1;
			}
			continue;
		}
		if current_chars > 0 {
			current.push(' ');
			current_chars += 1;
		}
		current.push_str(word);
		current_chars += word_chars;
	}
	if !current.is_empty() {
		emit(&current)?;
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::{KOKORO_VOICES, KokoroVoice, for_each_text_chunk};

	#[test]
	fn chunking_preserves_words_and_unicode_boundaries() {
		let mut chunks = Vec::new();
		for_each_text_chunk("one two superlongword café", 5, |chunk| {
			chunks.push(chunk.to_owned());
			Ok::<_, std::convert::Infallible>(())
		})
		.unwrap();
		assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 5));
		assert_eq!(
			chunks.join(" ").split_whitespace().collect::<String>(),
			"onetwosuperlongwordcafé"
		);
	}

	#[test]
	fn default_catalog_exposes_all_english_voice_packs() {
		assert_eq!(KOKORO_VOICES.len(), 28);
		assert!(
			KOKORO_VOICES
				.iter()
				.all(|voice| KokoroVoice::new(*voice).is_ok())
		);
	}
}
