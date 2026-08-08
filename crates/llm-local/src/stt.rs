use std::{path::PathBuf, time::Duration};

use omp_core::SmolStr;
use tokio_util::sync::CancellationToken;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::{
	Accelerator, Audio, DevicePreference, Error, Hub, ModelRepo, Result,
	device::whisper_accelerator, worker::Worker,
};

const WHISPER_REPO: &str = "ggerganov/whisper.cpp";
const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Whisper checkpoints available from the canonical `whisper.cpp` repository.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum SttModel {
	/// Fast English-only tiny model.
	TinyEnglish,
	/// English-only base model.
	BaseEnglish,
	/// English-only small model.
	SmallEnglish,
	/// Multilingual small model.
	Small,
	/// Multilingual medium model.
	Medium,
	/// Highest-accuracy multilingual Whisper v3 model.
	LargeV3,
	/// Faster distilled Whisper v3 model and the default local recognizer.
	#[default]
	LargeV3Turbo,
	/// A model file from an arbitrary Hugging Face repository.
	HuggingFace {
		/// Repository containing a GGML whisper.cpp checkpoint.
		repo:     ModelRepo,
		/// Repository-relative GGML filename.
		filename: SmolStr,
	},
	/// A GGML whisper.cpp checkpoint already on disk.
	Local(PathBuf),
}

impl SttModel {
	fn remote(&self) -> Option<(ModelRepo, SmolStr)> {
		let filename = match self {
			Self::TinyEnglish => "ggml-tiny.en.bin",
			Self::BaseEnglish => "ggml-base.en.bin",
			Self::SmallEnglish => "ggml-small.en.bin",
			Self::Small => "ggml-small.bin",
			Self::Medium => "ggml-medium.bin",
			Self::LargeV3 => "ggml-large-v3.bin",
			Self::LargeV3Turbo => "ggml-large-v3-turbo.bin",
			Self::HuggingFace { repo, filename } => return Some((repo.clone(), filename.clone())),
			Self::Local(_) => return None,
		};
		Some((ModelRepo::new(WHISPER_REPO), filename.into()))
	}
}

/// Controls language, timestamps, and decoding for one transcription.
#[derive(Clone, Debug)]
pub struct TranscriptionOptions {
	/// ISO-639-1 language code; `None` enables automatic detection.
	pub language:       Option<SmolStr>,
	/// Translate recognized speech to English instead of transcribing it.
	pub translate:      bool,
	/// Preserve per-segment timestamps in the result.
	pub timestamps:     bool,
	/// Prompt that biases names, vocabulary, or continuity from earlier audio.
	pub initial_prompt: Option<SmolStr>,
	/// Decoding temperature; `None` uses the recognizer's default greedy
	/// decoding. Whisper accepts values in `[0, 1]`; Parakeet rejects any
	/// value.
	pub temperature:    Option<f32>,
}

impl Default for TranscriptionOptions {
	fn default() -> Self {
		Self {
			language:       None,
			translate:      false,
			timestamps:     true,
			initial_prompt: None,
			temperature:    None,
		}
	}
}

/// Timestamped phrase produced by the recognizer.
#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptionSegment {
	/// Recognized text for this segment.
	pub text:                  SmolStr,
	/// Offset from the start of the submitted audio.
	pub start:                 Duration,
	/// End offset from the start of the submitted audio.
	pub end:                   Duration,
	/// Model probability that this interval contains no speech, when reported by
	/// the backend.
	pub no_speech_probability: Option<f32>,
}

/// Complete speech-recognition result.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcription {
	/// Concatenated, trimmed transcript.
	pub text:     SmolStr,
	/// Timestamped model segments, empty when timestamps were disabled.
	pub segments: Vec<TranscriptionSegment>,
	/// ISO-639-1 language code selected by the recognizer, when reported.
	pub language: Option<SmolStr>,
}

/// Configures a Whisper recognizer hosted on a dedicated native thread.
#[derive(Clone)]
pub struct WhisperBuilder {
	model:      SttModel,
	hub:        Option<Hub>,
	device:     DevicePreference,
	threads:    Option<usize>,
	flash_attn: bool,
}

impl WhisperBuilder {
	/// Creates a builder for a local or remote Whisper checkpoint.
	pub const fn new(model: SttModel) -> Self {
		Self { model, hub: None, device: DevicePreference::Auto, threads: None, flash_attn: true }
	}

	/// Shares Hugging Face cache, authentication, and offline policy.
	pub fn hub(mut self, hub: Hub) -> Self {
		self.hub = Some(hub);
		self
	}

	/// Selects CPU, Metal, or CUDA decoding.
	pub const fn device(mut self, device: DevicePreference) -> Self {
		self.device = device;
		self
	}

	/// Sets whisper.cpp's CPU worker count; the host's available parallelism is
	/// used by default.
	pub const fn threads(mut self, threads: usize) -> Self {
		self.threads = Some(threads);
		self
	}

	/// Enables fused attention on compatible whisper.cpp GPU backends.
	pub const fn flash_attention(mut self, enabled: bool) -> Self {
		self.flash_attn = enabled;
		self
	}

	/// Fetches the checkpoint and initializes whisper.cpp without blocking
	/// Tokio.
	pub async fn build(self) -> Result<Whisper> {
		if self.threads == Some(0) {
			return Err(Error::invalid("Whisper thread count must be non-zero"));
		}
		let default_language = matches!(
			&self.model,
			SttModel::TinyEnglish | SttModel::BaseEnglish | SttModel::SmallEnglish
		)
		.then_some("en");
		let hub = match self.hub {
			Some(hub) => hub,
			None => Hub::new()?,
		};
		let model_path = match &self.model {
			SttModel::Local(path) => path.clone(),
			model => {
				let (repo, filename) = model
					.remote()
					.ok_or_else(|| Error::invalid("remote Whisper model is missing its source"))?;
				hub.file(&repo, filename.as_str()).await?
			},
		};
		let preferred_accelerator = whisper_accelerator(self.device)?;
		let threads = self.threads.unwrap_or_else(|| {
			std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
		});
		let flash_attn = self.flash_attn;
		let preference = self.device;
		let worker = Worker::spawn("omp-llm-local-whisper", move || {
			let loaded =
				load_whisper(&model_path, preferred_accelerator, threads, flash_attn, default_language);
			match loaded {
				Ok(runtime) => Ok(runtime),
				Err(_) if preference == DevicePreference::Auto => {
					load_whisper(&model_path, Accelerator::Cpu, threads, false, default_language)
				},
				Err(error) => Err(error),
			}
		})
		.await?;
		let accelerator = worker
			.run_uncancelled(|runtime| Ok(runtime.accelerator))
			.await?;
		Ok(Whisper { worker, accelerator })
	}
}

impl Default for WhisperBuilder {
	fn default() -> Self {
		Self::new(SttModel::default())
	}
}

struct WhisperRuntime {
	context:          WhisperContext,
	threads:          usize,
	accelerator:      Accelerator,
	default_language: Option<&'static str>,
}

/// Fully asynchronous Whisper speech recognizer.
#[derive(Clone)]
pub struct Whisper {
	worker:      Worker<WhisperRuntime>,
	accelerator: Accelerator,
}

impl Whisper {
	/// Starts a builder using Whisper large-v3-turbo.
	pub fn builder() -> WhisperBuilder {
		WhisperBuilder::default()
	}

	/// Backend selected for recognition.
	pub const fn accelerator(&self) -> Accelerator {
		self.accelerator
	}

	/// Consumes PCM audio, converts it to mono 16 kHz, and transcribes it
	/// asynchronously.
	pub async fn transcribe(
		&self,
		audio: Audio,
		options: TranscriptionOptions,
		cancel: CancellationToken,
	) -> Result<Transcription> {
		let samples = tokio::task::spawn_blocking(move || audio.into_mono_at(WHISPER_SAMPLE_RATE))
			.await
			.map_err(|error| Error::backend("audio resampler", error))??;
		transcribe(&self.worker, samples, options, cancel).await
	}

	/// Opens an incremental input session whose buffered audio is moved into one
	/// final decode.
	pub fn session(&self, options: TranscriptionOptions) -> TranscriptionSession {
		TranscriptionSession { worker: self.worker.clone(), options, samples: Vec::new() }
	}

	/// Stops the recognizer after requests already in its queue have drained.
	pub async fn shutdown(&self) -> Result<()> {
		self.worker.shutdown().await
	}
}

/// Incremental audio-input session for live capture pipelines.
pub struct TranscriptionSession {
	worker:  Worker<WhisperRuntime>,
	options: TranscriptionOptions,
	samples: Vec<f32>,
}

impl TranscriptionSession {
	/// Appends one captured PCM chunk after downmixing and resampling it to
	/// Whisper's rate.
	pub fn push(&mut self, audio: Audio) -> Result<()> {
		self
			.samples
			.extend(audio.into_mono_at(WHISPER_SAMPLE_RATE)?);
		Ok(())
	}

	/// Moves all accumulated audio into the model and closes the session.
	pub async fn finish(self, cancel: CancellationToken) -> Result<Transcription> {
		transcribe(&self.worker, self.samples, self.options, cancel).await
	}
}

fn load_whisper(
	model_path: &std::path::Path,
	accelerator: Accelerator,
	threads: usize,
	flash_attn: bool,
	default_language: Option<&'static str>,
) -> Result<WhisperRuntime> {
	whisper_rs::install_logging_hooks();
	let mut parameters = WhisperContextParameters::new();
	parameters.use_gpu(accelerator != Accelerator::Cpu);
	parameters.gpu_device(0);
	parameters.flash_attn(flash_attn);
	let context = WhisperContext::new_with_params(model_path, parameters)
		.map_err(|error| Error::backend("whisper", error))?;
	Ok(WhisperRuntime { context, threads, accelerator, default_language })
}

unsafe extern "C" fn whisper_abort(user_data: *mut std::ffi::c_void) -> bool {
	if user_data.is_null() {
		return false;
	}
	// SAFETY: `transcribe` keeps this CancellationToken alive until whisper.cpp
	// returns.
	let cancel = unsafe { &*user_data.cast::<CancellationToken>() };
	cancel.is_cancelled()
}

async fn transcribe(
	worker: &Worker<WhisperRuntime>,
	samples: Vec<f32>,
	options: TranscriptionOptions,
	cancel: CancellationToken,
) -> Result<Transcription> {
	worker
		.run(cancel, move |runtime, cancel| {
			let mut state = runtime
				.context
				.create_state()
				.map_err(|error| Error::backend("whisper", error))?;
			let mut parameters = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
			parameters.set_n_threads(runtime.threads.min(i32::MAX as usize) as i32);
			parameters.set_translate(options.translate);
			parameters.set_no_timestamps(!options.timestamps);
			parameters.set_print_progress(false);
			parameters.set_print_realtime(false);
			parameters.set_print_timestamps(false);
			parameters.set_language(options.language.as_deref().or(runtime.default_language));
			if let Some(temperature) = options.temperature {
				if !temperature.is_finite() || !(0.0..=1.0).contains(&temperature) {
					return Err(Error::invalid("transcription temperature must be in [0, 1]"));
				}
				parameters.set_temperature(temperature);
			}
			if let Some(prompt) = options.initial_prompt.as_deref() {
				parameters.set_initial_prompt(prompt);
			}
			let abort = cancel.clone();
			// SAFETY: whisper.cpp invokes the callback only during `full`; `abort` has a
			// stable address for that call and CancellationToken supports concurrent
			// shared access.
			unsafe {
				parameters.set_abort_callback(Some(whisper_abort));
				parameters.set_abort_callback_user_data(std::ptr::from_ref(&abort).cast_mut().cast());
			}
			state
				.full(parameters, &samples)
				.map_err(|error| Error::backend("whisper", error))?;
			if cancel.is_cancelled() {
				return Err(Error::Cancelled);
			}

			let mut text = String::new();
			let mut segments = Vec::with_capacity(state.full_n_segments().max(0) as usize);
			for segment in state.as_iter() {
				let segment_text = segment
					.to_str()
					.map_err(|error| Error::backend("whisper", error))?;
				text.push_str(segment_text);
				if options.timestamps {
					segments.push(TranscriptionSegment {
						text:                  segment_text.into(),
						start:                 whisper_timestamp(segment.start_timestamp()),
						end:                   whisper_timestamp(segment.end_timestamp()),
						no_speech_probability: Some(segment.no_speech_probability()),
					});
				}
			}
			let language = whisper_rs::get_lang_str(state.full_lang_id_from_state()).map(Into::into);
			Ok(Transcription { text: text.trim().into(), segments, language })
		})
		.await
}

fn whisper_timestamp(timestamp: i64) -> Duration {
	Duration::from_millis(timestamp.max(0) as u64 * 10)
}
