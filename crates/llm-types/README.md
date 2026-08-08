# omp-llm-types

`omp-llm-types` defines the canonical, provider-independent values and object-safe capability traits used by LLM integrations. It is the in-process contract for chat, token counting, embeddings, media generation, speech, transcription, search, quota inspection, and provider-initiated invocations; protobuf is treated as a process-boundary transport binding.

## Structure

- `request`, `thread`, and `event` model requests, portable conversations, streamed turn events, outcomes, usage, and errors.
- `text`, `media`, and `search` contain operation-specific values for completion, extraction, embeddings, audio, image and video generation, transcription, and web search.
- `facet` declares the provider-neutral async capability traits and their streaming or job-shaped contracts.
- `accumulator` reconstructs canonical turn events from streamed parts, while `convert` bridges canonical values to `omp-proto` messages.
- `ids` supplies stable identifiers, and `props` carries extensible, namespaced provider properties without expanding the portable type surface.

## Philosophy

Canonical values stay independent of any provider or wire format so gateway, middleware, and provider implementations share one semantic contract. Portable features carry explicit fallback policy, and outcomes record unsupported, omitted, or emulated behavior rather than hiding semantic changes. Provider-specific data belongs in namespaced properties or edge conversions; fields with portable meaning remain typed. Public enums and records are generally non-exhaustive so the contract can grow without requiring consumers to assume a closed provider capability set.
