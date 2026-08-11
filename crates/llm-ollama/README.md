# omp-llm-ollama

`omp-llm-ollama` adapts OMP's provider-neutral LLM types to Ollama's native `POST /api/chat` protocol. It encodes chat requests — messages, images, tools, structured-output schemas, and reasoning options — into Ollama request JSON, decodes the NDJSON response stream into canonical turn events, outcomes, usage, and stop reasons, and enumerates available models in `discovery`.

## Structure

The crate is implemented in `src/lib.rs` around `OllamaChatCodec`, which implements the shared transport interface. Request projection flattens message parts into Ollama's text-plus-images shape, sanitizes JSON schemas to the subset Ollama accepts, maps reasoning effort per model policy, and reports unsupported features explicitly. The response path decodes one NDJSON line at a time with per-stream state, opening and closing text and thinking parts as the stream interleaves them.

## Philosophy

Keep Ollama-specific wire details at the transport boundary and expose the rest of OMP to shared request and event types. Request conversion reports compatibility losses instead of silently dropping them, and streaming decode preserves part ordering and treats a truncated stream as an upstream error rather than a successful response.
