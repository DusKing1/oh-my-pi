//! Whisper.cpp-backed local speech recognition.

use std::{path::PathBuf, sync::Arc, time::Duration};

use omp_core::Str;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use super::runtime::{
	LocalCancellation, LocalError, LocalErrorKind, LocalExecutionReceipt, LocalResult, LocalRuntime,
	MemoryPool,
};

/// Configuration for a verified whisper.cpp checkpoint.
#[derive(Clone, Debug)]
pub struct WhisperConfig {
	/// Path to a ggml Whisper checkpoint.
	pub model_path:      PathBuf,
	/// CPU worker count used by decoding.
	pub threads:         usize,
	/// Whether whisper.cpp may use its compiled GPU backend.
	pub use_gpu:         bool,
	/// Whether fused attention is enabled.
	pub flash_attention: bool,
	/// Estimated resident bytes charged before loading.
	pub resident_bytes:  usize,
	/// Admission limit; currently must be one because Whisper access is
	/// serialized.
	pub max_concurrency: usize,
	/// Explicit idle-unload interval.
	pub idle_timeout:    Duration,
}

/// Controls one transcription.
#[derive(Clone, Debug, Default)]
pub struct TranscriptionOptions {
	/// Optional ISO-639-1 language code; absent enables detection.
	pub language:       Option<Str>,
	/// Translate recognized speech to English.
	pub translate:      bool,
	/// Include segment timestamps.
	pub timestamps:     bool,
	/// Optional initial decoder prompt.
	pub initial_prompt: Option<Str>,
	/// Sampling temperature in `[0, 1]`.
	pub temperature:    Option<f32>,
}

/// One timestamped transcription segment.
#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptionSegment {
	/// Recognized text.
	pub text:                  Str,
	/// Start offset from the audio beginning.
	pub start:                 Duration,
	/// End offset from the audio beginning.
	pub end:                   Duration,
	/// Model probability that the interval contains no speech.
	pub no_speech_probability: f32,
}

/// Complete transcription and local execution evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcription {
	/// Concatenated recognized text.
	pub text:     Str,
	/// Timestamped segments, empty when timestamps were disabled.
	pub segments: Vec<TranscriptionSegment>,
	/// Detected or requested language.
	pub language: Option<Str>,
	/// Local runtime receipt.
	pub receipt:  LocalExecutionReceipt,
}

struct WhisperEngine {
	context: WhisperContext,
	threads: usize,
}

/// Lazy, bounded adapter over whisper.cpp.
#[derive(Clone)]
pub struct WhisperAdapter {
	runtime: LocalRuntime<WhisperEngine>,
}

unsafe extern "C" fn whisper_abort(user_data: *mut std::ffi::c_void) -> bool {
	if user_data.is_null() {
		return false;
	}
	// SAFETY: transcribe keeps this cancellation token alive until whisper.cpp
	// returns.
	let cancel = unsafe { &*user_data.cast::<LocalCancellation>() };
	cancel.is_cancelled()
}

impl WhisperAdapter {
	/// Creates a lazy adapter for a local checkpoint.
	pub fn new(config: WhisperConfig, memory: Arc<MemoryPool>) -> LocalResult<Self> {
		if config.threads == 0 {
			return Err(LocalError::new(
				LocalErrorKind::InvalidInput,
				"Whisper thread count must be non-zero",
			));
		}
		let resident_bytes = config.resident_bytes;
		let concurrency = config.max_concurrency;
		let idle = config.idle_timeout;
		let runtime = LocalRuntime::new(
			move || {
				whisper_rs::install_logging_hooks();
				let mut parameters = WhisperContextParameters::new();
				parameters.use_gpu(config.use_gpu);
				parameters.gpu_device(0);
				parameters.flash_attn(config.flash_attention);
				let context = WhisperContext::new_with_params(&config.model_path, parameters).map_err(
					|error| {
						LocalError::new(LocalErrorKind::Backend, format!("Whisper load failed: {error}"))
					},
				)?;
				Ok(WhisperEngine { context, threads: config.threads })
			},
			memory,
			resident_bytes,
			concurrency,
			idle,
		)?;
		Ok(Self { runtime })
	}

	/// Transcribes mono 16 kHz floating-point PCM using whisper.cpp.
	pub fn transcribe_mono_16khz(
		&self,
		samples: &[f32],
		options: &TranscriptionOptions,
		cancel: &LocalCancellation,
	) -> LocalResult<Transcription> {
		if samples.is_empty() {
			return Err(LocalError::new(
				LocalErrorKind::InvalidInput,
				"transcription requires audio samples",
			));
		}
		if options
			.temperature
			.is_some_and(|temperature| !temperature.is_finite() || !(0.0..=1.0).contains(&temperature))
		{
			return Err(LocalError::new(
				LocalErrorKind::InvalidInput,
				"transcription temperature must be in [0, 1]",
			));
		}
		let lease = self.runtime.acquire(cancel)?;
		let receipt = lease.receipt();
		let (text, segments, language) = lease.with_engine(|engine| {
			let mut state = engine.context.create_state().map_err(|error| {
				LocalError::new(LocalErrorKind::Backend, format!("Whisper state failed: {error}"))
			})?;
			let mut parameters = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
			parameters.set_n_threads(engine.threads.min(i32::MAX as usize) as i32);
			parameters.set_translate(options.translate);
			parameters.set_no_timestamps(!options.timestamps);
			parameters.set_print_progress(false);
			parameters.set_print_realtime(false);
			parameters.set_print_timestamps(false);
			parameters.set_language(options.language.as_ref().map(Str::as_str));
			if let Some(temperature) = options.temperature {
				parameters.set_temperature(temperature);
			}
			if let Some(prompt) = options.initial_prompt.as_ref() {
				parameters.set_initial_prompt(prompt.as_str());
			}
			// SAFETY: whisper.cpp invokes the callback only during `full`; `cancel`
			// remains at a stable address for that synchronous call.
			unsafe {
				parameters.set_abort_callback(Some(whisper_abort));
				parameters.set_abort_callback_user_data(std::ptr::from_ref(cancel).cast_mut().cast());
			}
			state.full(parameters, samples).map_err(|error| {
				LocalError::new(LocalErrorKind::Backend, format!("Whisper inference failed: {error}"))
			})?;
			if cancel.is_cancelled() {
				return Err(LocalError::cancelled());
			}
			let mut text = String::new();
			let mut segments = Vec::with_capacity(state.full_n_segments().max(0) as usize);
			for segment in state.as_iter() {
				let segment_text = segment.to_str().map_err(|error| {
					LocalError::new(LocalErrorKind::Backend, format!("Whisper text failed: {error}"))
				})?;
				text.push_str(segment_text);
				if options.timestamps {
					segments.push(TranscriptionSegment {
						text:                  segment_text.into(),
						start:                 whisper_timestamp(segment.start_timestamp()),
						end:                   whisper_timestamp(segment.end_timestamp()),
						no_speech_probability: segment.no_speech_probability(),
					});
				}
			}
			let language = whisper_rs::get_lang_str(state.full_lang_id_from_state()).map(Into::into);
			Ok((Str::from(text.trim()), segments, language))
		})?;
		Ok(Transcription { text, segments, language, receipt })
	}

	/// Unloads the checkpoint when inactive for its configured interval.
	pub fn unload_if_idle(&self, now: std::time::Instant) -> bool {
		self.runtime.unload_if_idle(now)
	}

	/// Returns whether the Whisper checkpoint is resident.
	pub fn is_loaded(&self) -> bool {
		self.runtime.is_loaded()
	}
}

fn whisper_timestamp(timestamp: i64) -> Duration {
	Duration::from_millis(timestamp.max(0) as u64 * 10)
}
