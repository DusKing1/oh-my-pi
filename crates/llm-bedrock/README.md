# omp-llm-bedrock

`omp-llm-bedrock` adapts OMP's provider-neutral LLM types to Amazon Bedrock's model-independent `ConverseStream` API. It encodes chat requests into Converse JSON — messages, tools, inference settings, cache points, and reasoning configuration — decodes the streamed event payloads into canonical turn events, outcomes, usage, and stop reasons, and lists available models via the `ListFoundationModels` control plane in `discovery`.

## Structure

The crate is implemented in `src/lib.rs` around `BedrockConverseCodec`, which implements the shared transport interface. Request projection builds Converse message and content blocks, reports unsupported features explicitly, and handles Bedrock quirks such as the `toolConfig` requirement for historical tool blocks (via a reserved sentinel tool) and per-model reasoning budgets for Claude variants. The response path parses `ConverseStream` JSON events with per-stream state to assemble text, tool-use, and reasoning parts deterministically. AWS `EventStream` framing and `SigV4` signing intentionally stay in the shared Bedrock infrastructure; `discovery` attaches non-secret signing context but never signs.

## Philosophy

Keep Bedrock-specific wire details at the transport boundary and expose the rest of OMP to shared request and event types. Request conversion reports compatibility losses instead of silently dropping them, and streaming decode preserves ordering and treats an incomplete stream as an upstream error rather than a successful response.
