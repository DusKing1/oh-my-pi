//! In-process inference with shared bounded lifecycle and verified artifacts.

/// Apple Foundation Models dynamic runtime.
#[cfg(feature = "local-applefm")]
pub mod applefm;
/// Verified, root-confined model artifacts.
pub mod artifact;
/// FastEmbed local embeddings.
#[cfg(feature = "local-embedding")]
pub mod embedding;
/// Shared admission, memory, cancellation, and idle-unload lifecycle.
pub mod runtime;
/// Whisper.cpp speech recognition.
#[cfg(feature = "local-stt")]
pub mod stt;
/// llama.cpp GGUF text generation.
#[cfg(feature = "local-text")]
pub mod text;
/// Kokoro-82M speech synthesis.
#[cfg(feature = "local-tts")]
pub mod tts;

pub use artifact::{ArtifactReceipt, ArtifactSpec, ArtifactStore, VerifiedArtifact};
pub use runtime::{
	AdmissionControl, AvailabilityEvidence, LocalCancellation, LocalError, LocalErrorKind,
	LocalExecutionReceipt, LocalResult, LocalRuntime, MemoryPool, MemoryReservation, RuntimeLease,
};
