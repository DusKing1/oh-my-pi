# omp-llm-transport

`omp-llm-transport` defines the shared boundary between provider-neutral LLM requests and provider wire formats. Its `Transport` trait pairs request encoding with incremental response decoding, while `Frame` represents the SSE, JSON, raw-byte, and error inputs presented to provider codecs. The crate also supplies reusable framing and first-party transports for local inference and OMP federation.

## Structure

- `lib.rs` defines `Transport`, `Frame`, per-turn `DecodeState`, and shared stop-reason precedence.
- `sse.rs` incrementally assembles Server-Sent Events, including multi-line data and the terminal `[DONE]` sentinel.
- `ndjson.rs` splits byte chunks into complete newline-delimited JSON records while retaining partial input.
- `normalize.rs` adapts tool schemas to provider-selected compatibility rules and reports unsupported behavior.
- `embedded.rs` exposes local inference engines through the same transport-facing facets used by remote providers.
- `omp.rs` implements the native `omp.inference.v1` federation client over gRPC.

## Philosophy

Transport mechanics stay separate from provider semantics: framing turns arbitrary byte chunks into stable inputs, and provider codecs own their decoding state. Encoding returns unsupported-feature reports alongside wire bodies rather than silently dropping requested behavior. Incremental decoders retain only incomplete input and favor shared byte slices where possible. Local engines and federated gateways use the same routing-facing abstractions so callers do not need transport-specific special cases.
