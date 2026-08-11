# omp-llm-openai

OpenAI transport codecs for omp's shared LLM request and event types. The crate supports both the Chat Completions (`/v1/chat/completions`) protocol and the Responses (`/v1/responses`) item and typed-event protocol.

## Structure

- `openai_chat` provides `OpenAiChatCodec`, which encodes chat messages, tools, and reasoning options and decodes streaming chunks into turn events.
- `openai_responses` provides `OpenAiResponsesCodec`, which encodes Responses API input and decodes typed streaming events, output items, usage, errors, and terminal outcomes.
- `lib.rs` exposes both modules and re-exports the two codec types.

## Philosophy

Keep OpenAI-specific wire formats at the transport boundary while using the workspace's shared chat and event model everywhere else. Compatibility differences and unsupported features are reported explicitly rather than silently changing request meaning. Responses conversation state remains gateway-owned: the codec accepts an authoritative `previous_response_id` when supplied and never discovers or invents one.
