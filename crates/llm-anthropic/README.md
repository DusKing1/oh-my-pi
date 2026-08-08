# omp-llm-anthropic

`omp-llm-anthropic` adapts OMP's provider-neutral LLM types to Anthropic's Messages API. It encodes chat and token-counting requests, decodes typed server-sent events, and turns Anthropic responses into canonical turn events, outcomes, usage, stop reasons, and errors.

## Structure

The crate is implemented in `src/lib.rs` around `AnthropicCodec`, which implements the shared transport interface for `AnthropicMessages`. Request projection builds Anthropic wire messages and content blocks for text, images, tools, reasoning, cache control, and related options while reporting unsupported features. The response path parses message, content-block, delta, usage, and error events, then uses per-stream state to assemble canonical output. `encode_count` and `decode_count` handle the `/v1/messages/count_tokens` endpoint.

## Philosophy

Keep Anthropic-specific wire details at the transport boundary and expose the rest of OMP to shared request and event types. Request conversion should report compatibility losses explicitly, while streaming decode should preserve ordering, accumulate partial blocks deterministically, and treat an incomplete stream as an upstream error rather than a successful response.
