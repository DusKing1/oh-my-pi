# omp-voice-kokoro

`omp-voice-kokoro` runs Kokoro-82M text-to-speech inference on [candle](https://github.com/huggingface/candle), with Metal acceleration on macOS. It is a vendored copy of the MIT-licensed `voice-kokoro` crate from [rgbkrk/voice](https://github.com/rgbkrk/voice) (Copyright Kyle Kelley, see `LICENSE`), published under the `omp-` prefix for use by OMP's local inference backends.

## Structure

`KModel` (in `model.rs`) is the full synthesis pipeline: an ALBERT text encoder (`albert.rs`), duration/prosody predictors with bidirectional LSTMs (`bilstm.rs`, `modules.rs`), and an iSTFTNet vocoder (`istftnet.rs`). `ModelConfig` (`config.rs`) deserializes the upstream `config.json`; weights load from safetensors through a candle `VarBuilder`.

## Philosophy

Pure inference, no I/O: the crate never downloads models or touches the network — callers fetch weights and voice embeddings themselves and hand them over as candle tensors. Upstream numerical-parity tests against the Python reference implementation live in the source repository.
