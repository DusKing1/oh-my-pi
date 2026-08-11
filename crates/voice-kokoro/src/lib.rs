//! Neural text-to-speech synthesis with the Kokoro model.
//!
//! The public modules expose the model architecture and checkpoint configuration
//! used by [`KModel`] to generate audio samples.

#![allow(
	clippy::too_many_arguments,
	reason = "Model-layer constructors mirror the fixed checkpoint tensor interfaces."
)]
#![allow(
	clippy::vec_init_then_push,
	reason = "Layer lists are assembled conditionally while loading model architecture."
)]

/// Checkpoint-compatible ALBERT text encoder layers.
pub mod albert;
/// Bidirectional LSTM layers used by the duration predictor.
pub mod bilstm;
/// Model configuration deserialized from Kokoro checkpoint metadata.
pub mod config;
pub mod istftnet;
pub mod model;
pub mod modules;

pub use config::ModelConfig;
pub use istftnet::SynthesisMode;
pub use model::KModel;
