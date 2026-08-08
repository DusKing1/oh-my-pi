# omp-llm-local

`omp-llm-local` runs local text generation, embeddings, speech recognition, speech synthesis, and Apple Foundation Models behind the canonical `omp-llm-types` facet contracts. `Inference` implements the embedded `LocalEngine` boundary, while the individual engines remain available for direct use.

## Structure

- `inference` adapts canonical chat, embedding, transcription, and speech requests directly to the local engines.
- `text` and `embeddings` provide local text generation and embedding engines.
- `stt` and `parakeet` implement speech recognition; `tts` implements Kokoro speech synthesis, with `audio` holding shared audio types.
- `hub` resolves and caches model artifacts through Hugging Face, while `device` selects available accelerators.
- `worker` keeps CPU-bound runtimes off Tokio executor threads.
- `error` defines the crate's shared error model, and `omp-llm-fm` supplies the re-exported Apple Foundation Models integration.

## Philosophy

The crate keeps local inference behind the same canonical contracts as remote providers without hiding engine-specific APIs. Model acquisition and accelerator selection are centralized, stream ownership provides cancellation, and blocking inference is isolated on dedicated workers so callers can compose local models with Tokio applications without blocking the executor.
