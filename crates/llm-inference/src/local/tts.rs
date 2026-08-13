//! Kokoro-82M-backed local speech synthesis.

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use omp_core::Str;
use omp_voice_kokoro::{KModel, ModelConfig, SynthesisMode};

use super::runtime::{
	LocalCancellation, LocalError, LocalErrorKind, LocalExecutionReceipt, LocalResult, LocalRuntime,
	MemoryPool,
};

/// Accelerator requested for Kokoro.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KokoroDevice {
	/// Portable CPU execution.
	Cpu,
	/// Apple Metal execution.
	Metal,
}

/// Local files and lifecycle bounds for Kokoro-82M.
#[derive(Clone, Debug)]
pub struct KokoroConfig {
	/// Model JSON configuration path.
	pub config_path:     PathBuf,
	/// Model safetensors path.
	pub weights_path:    PathBuf,
	/// Voice-pack safetensors paths keyed by voice name.
	pub voices:          HashMap<Str, PathBuf>,
	/// Requested accelerator.
	pub device:          KokoroDevice,
	/// Estimated resident bytes charged before loading.
	pub resident_bytes:  usize,
	/// Admission limit; currently must be one because Kokoro access is
	/// serialized.
	pub max_concurrency: usize,
	/// Explicit idle-unload interval.
	pub idle_timeout:    Duration,
}

/// Controls one synthesis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynthesisOptions {
	/// Speaking-rate multiplier.
	pub speed:           f32,
	/// Maximum approximate characters per model pass.
	pub max_chunk_chars: usize,
	/// Removes decoder noise for repeatable output.
	pub deterministic:   bool,
}

impl Default for SynthesisOptions {
	fn default() -> Self {
		Self { speed: 1.0, max_chunk_chars: 400, deterministic: false }
	}
}

/// Complete mono PCM synthesis with evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct SynthesisOutput {
	/// Mono floating-point PCM samples.
	pub samples:     Vec<f32>,
	/// Sample rate declared by the model.
	pub sample_rate: u32,
	/// Local runtime receipt.
	pub receipt:     LocalExecutionReceipt,
}

struct KokoroEngine {
	model:       KModel,
	config:      ModelConfig,
	device:      Device,
	voices:      HashMap<Str, Tensor>,
	voice_paths: HashMap<Str, PathBuf>,
	g2p:         voice_g2p::G2P,
}

/// Lazy, bounded adapter over the workspace Kokoro engine.
#[derive(Clone)]
pub struct KokoroAdapter {
	runtime: LocalRuntime<KokoroEngine>,
}

impl KokoroAdapter {
	/// Creates a lazy adapter from local model and voice artifacts.
	pub fn new(config: KokoroConfig, memory: Arc<MemoryPool>) -> LocalResult<Self> {
		let resident = config.resident_bytes;
		let concurrency = config.max_concurrency;
		let idle = config.idle_timeout;
		let runtime = LocalRuntime::new(
			move || {
				let device = kokoro_device(config.device)?;
				let config_bytes = std::fs::read(&config.config_path).map_err(|error| {
					LocalError::new(
						LocalErrorKind::Artifact,
						format!("Kokoro config read failed: {error}"),
					)
				})?;
				let model_config: ModelConfig =
					serde_json::from_slice(&config_bytes).map_err(|error| {
						LocalError::new(
							LocalErrorKind::Artifact,
							format!("Kokoro config decode failed: {error}"),
						)
					})?;
				// SAFETY: Candle owns each mapping for the lifetime of tensors built from it.
				let variables = unsafe {
					VarBuilder::from_mmaped_safetensors(&[&config.weights_path], DType::F32, &device)
				}
				.map_err(|error| {
					LocalError::new(LocalErrorKind::Backend, format!("Kokoro weights failed: {error}"))
				})?;
				let model = KModel::load(&model_config, variables).map_err(|error| {
					LocalError::new(LocalErrorKind::Backend, format!("Kokoro load failed: {error}"))
				})?;
				Ok(KokoroEngine {
					model,
					config: model_config,
					device,
					voices: HashMap::new(),
					voice_paths: config.voices.clone(),
					g2p: voice_g2p::G2P::new(),
				})
			},
			memory,
			resident,
			concurrency,
			idle,
		)?;
		Ok(Self { runtime })
	}

	/// Synthesizes text into mono PCM with the selected real voice pack.
	pub fn synthesize(
		&self,
		text: &str,
		voice: &str,
		options: SynthesisOptions,
		cancel: &LocalCancellation,
	) -> LocalResult<SynthesisOutput> {
		validate_synthesis(text, voice, options)?;
		let lease = self.runtime.acquire(cancel)?;
		let receipt = lease.receipt();
		let (samples, sample_rate) = lease.with_engine(|engine| {
			if !engine.voices.contains_key(voice) {
				let path = engine.voice_paths.get(voice).ok_or_else(|| {
					LocalError::new(LocalErrorKind::InvalidInput, "unknown Kokoro voice")
				})?;
				let tensors =
					candle_core::safetensors::load(path, &engine.device).map_err(|error| {
						LocalError::new(
							LocalErrorKind::Artifact,
							format!("Kokoro voice load failed: {error}"),
						)
					})?;
				let tensor = tensors.into_values().next().ok_or_else(|| {
					LocalError::new(LocalErrorKind::Artifact, "Kokoro voice pack contains no tensors")
				})?;
				engine.voices.insert(voice.into(), tensor);
			}
			let voice_tensor = engine.voices.get(voice).expect("inserted above").clone();
			let samples = synthesize_text(engine, &voice_tensor, text, options, cancel)?;
			Ok((samples, engine.config.sample_rate))
		})?;
		Ok(SynthesisOutput { samples, sample_rate, receipt })
	}

	/// Unloads Kokoro when inactive for its configured interval.
	pub fn unload_if_idle(&self, now: std::time::Instant) -> bool {
		self.runtime.unload_if_idle(now)
	}

	/// Returns whether Kokoro is resident.
	pub fn is_loaded(&self) -> bool {
		self.runtime.is_loaded()
	}
}

fn kokoro_device(requested: KokoroDevice) -> LocalResult<Device> {
	match requested {
		KokoroDevice::Cpu => Ok(Device::Cpu),
		KokoroDevice::Metal => {
			#[cfg(target_os = "macos")]
			{
				Device::new_metal(0).map_err(|error| {
					LocalError::new(
						LocalErrorKind::Unsupported,
						format!("Metal is unavailable: {error}"),
					)
				})
			}
			#[cfg(not(target_os = "macos"))]
			{
				Err(LocalError::new(LocalErrorKind::Unsupported, "Metal requires macOS"))
			}
		},
	}
}

fn validate_synthesis(text: &str, voice: &str, options: SynthesisOptions) -> LocalResult<()> {
	if text.trim().is_empty() || voice.is_empty() {
		return Err(LocalError::new(
			LocalErrorKind::InvalidInput,
			"speech synthesis requires text and a voice",
		));
	}
	if !options.speed.is_finite() || options.speed <= 0.0 || options.max_chunk_chars == 0 {
		return Err(LocalError::new(
			LocalErrorKind::InvalidInput,
			"synthesis speed and chunk size must be positive",
		));
	}
	Ok(())
}

fn synthesize_text(
	engine: &KokoroEngine,
	voice: &Tensor,
	text: &str,
	options: SynthesisOptions,
	cancel: &LocalCancellation,
) -> LocalResult<Vec<f32>> {
	let mut output = Vec::new();
	for_each_text_chunk(text, options.max_chunk_chars, |chunk| {
		if cancel.is_cancelled() {
			return Err(LocalError::cancelled());
		}
		let phonemes = engine.g2p.convert(chunk).map_err(|error| {
			LocalError::new(LocalErrorKind::Backend, format!("Kokoro G2P failed: {error}"))
		})?;
		let mut encoded = [0_u8; 4];
		let token_ids: Vec<_> = phonemes
			.chars()
			.filter_map(|character| {
				engine
					.config
					.vocab
					.get(character.encode_utf8(&mut encoded))
					.copied()
			})
			.collect();
		if token_ids.is_empty() {
			return Ok(());
		}
		let pack_len = voice.dim(0).map_err(|error| {
			LocalError::new(LocalErrorKind::Artifact, format!("Kokoro voice shape failed: {error}"))
		})?;
		if pack_len == 0 {
			return Err(LocalError::new(LocalErrorKind::Artifact, "empty Kokoro voice pack"));
		}
		let style = voice
			.i((token_ids.len() - 1).min(pack_len - 1))
			.and_then(|tensor| tensor.squeeze(0))
			.and_then(|tensor| tensor.unsqueeze(0))
			.map_err(|error| {
				LocalError::new(LocalErrorKind::Artifact, format!("Kokoro voice style failed: {error}"))
			})?;
		let mode = if options.deterministic {
			SynthesisMode::Deterministic
		} else {
			SynthesisMode::Stochastic
		};
		let audio = engine
			.model
			.forward_with_mode(&token_ids, &style, options.speed, &engine.device, mode)
			.and_then(|tensor| tensor.to_vec1::<f32>())
			.map_err(|error| {
				LocalError::new(LocalErrorKind::Backend, format!("Kokoro inference failed: {error}"))
			})?;
		output.extend(audio);
		Ok(())
	})?;
	if output.is_empty() {
		return Err(LocalError::new(LocalErrorKind::Backend, "text produced no supported phonemes"));
	}
	Ok(output)
}

fn for_each_text_chunk<E>(
	text: &str,
	max_chars: usize,
	mut emit: impl FnMut(&str) -> Result<(), E>,
) -> Result<(), E> {
	let mut current = String::new();
	let mut count = 0;
	for word in text.split_whitespace() {
		let chars = word.chars().count();
		if count > 0 && count + 1 + chars > max_chars {
			emit(&current)?;
			current.clear();
			count = 0;
		}
		if chars > max_chars {
			for character in word.chars() {
				if count == max_chars {
					emit(&current)?;
					current.clear();
					count = 0;
				}
				current.push(character);
				count += 1;
			}
			continue;
		}
		if count > 0 {
			current.push(' ');
			count += 1;
		}
		current.push_str(word);
		count += chars;
	}
	if !current.trim().is_empty() {
		emit(current.trim())?;
	}
	Ok(())
}
