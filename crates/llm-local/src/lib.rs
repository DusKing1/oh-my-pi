//! Async, hardware-accelerated local inference for text, embeddings, Apple
//! Foundation Models, speech recognition, and speech synthesis.
//!
//! [`Inference`] is the unified engine entry point. It implements the canonical
//! [`LocalEngine`](omp_llm_transport::embedded::LocalEngine) contract and can
//! be mounted with [`Embedded`] so local and remote providers traverse
//! identical catalog routing. Individual engines ([`TextGenerator`],
//! [`Embedder`], [`Kokoro`], [`Whisper`], [`Parakeet`]) remain available for
//! direct use.
//!
//! Model downloads are resolved through [`Hub`] and cached in the standard
//! Hugging Face cache. CPU-bound runtimes execute on dedicated worker threads
//! so inference never blocks a Tokio executor.

mod audio;
mod device;
mod embeddings;
mod error;
mod hub;
mod inference;
mod parakeet;
mod stt;
mod text;
mod tts;
mod worker;

pub use audio::Audio;
pub use device::{Accelerator, DevicePreference};
pub use embeddings::{Embedder, EmbedderBuilder, EmbeddingModel, EmbeddingOptions};
pub use error::{Error, ErrorKind, Result};
pub use hub::{FetchOptions, Hub, HubBuilder, ModelRepo, SnapshotOptions};
pub use inference::{Inference, InferenceBuilder, SttSelection, TextSelection};
pub use omp_llm_fm::{
	AppleFm, AppleFmAvailability, AppleFmError, AppleFmErrorCode, AppleFmEvent, AppleFmGeneration,
	AppleFmOptions, AppleFmStream,
};
/// In-process transport wrapper for mounting an [`Inference`] engine.
pub use omp_llm_transport::embedded::Embedded;
/// Canonical request, event, and facet contracts used by the embedded bridge.
pub use omp_llm_types as types;
pub use parakeet::{Parakeet, ParakeetBuilder, ParakeetFiles, ParakeetSession};
pub use stt::{
	SttModel, Transcription, TranscriptionOptions, TranscriptionSegment, TranscriptionSession,
	Whisper, WhisperBuilder,
};
pub use text::{
	ChatMessage, ChatRole, GenerationOptions, GenerationStop, GenerationStream, GenerationSummary,
	SmallModel, TextGenerator, TextGeneratorBuilder, TextModel,
};
pub use tokio_util::sync::CancellationToken;
pub use tts::{
	KOKORO_VOICES, Kokoro, KokoroBuilder, KokoroVoice, SynthesisOptions, SynthesisStream,
};
