# omp-llm-google

Google provider wire integration for OMP. The crate translates shared chat requests into Google `GenerateContent` payloads and projects streamed responses back into OMP turn events for the public Generative Language API, Vertex AI, and Cloud Code Assist.

## Structure

- `lib.rs` defines `GoogleCodec`, the public GenAI and Vertex variants, request encoding, streamed response decoding, and Vertex publisher-model URL construction.
- `cca.rs` adapts the same wire model to Cloud Code Assist: it wraps requests and responses in CCA envelopes, builds catalog-ordered endpoint plans, records the served endpoint, and decides when a pre-response failure permits endpoint fallback.

## Philosophy

Keep provider-specific wire details at the transport boundary while preserving OMP's shared chat and event types. Endpoint differences are represented as small data-driven variants, and Cloud Code Assist routing remains explicit: credentials and onboarding stay outside the codec, endpoint priority comes from the provider catalog, and fallback is limited to transient failures before response content begins.
