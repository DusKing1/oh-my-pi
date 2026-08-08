# omp-llm-gateway

`omp-llm-gateway` is the server-side gateway for OMP model inference. It owns conversation context and turn orchestration, exposes native inference services, and adapts vendor-compatible HTTP requests to the same internal egress stack.

## Structure

- `context` provides bounded, optimistic-concurrency conversation storage, turn idempotency, and atomic commit or rollback.
- `turn` implements the bidirectional chat-turn protocol, resolves credential-bound chat routes, drives client tool execution, and records turn telemetry.
- `discovery` translates joined inference-registry snapshots and events into model and provider discovery RPCs.
- `federation` discovers catalogs from other OMP gateways and forwards turns without moving provider credentials between hosts.
- `media` orchestrates image, speech, transcription, and video facets while storing reusable media through content-addressed blobs.
- `search` applies registry policy, deadlines, and fallback execution for web search.
- `listener` serves native gRPC and foreign vendor HTTP APIs over one transport boundary.
- `local` parses and connects platform-native local endpoints: owner-only Unix sockets and local-user-only Windows named pipes.
- `facade` contains authenticated OpenAI- and Anthropic-compatible routes for chat, responses, messages, models, embeddings, audio, images, and videos.

## Philosophy

All provider traffic passes through one registered gateway path so authentication, proxying, retries, limits, and metering remain consistent. Turns stage context and provider affinity together, committing only after success; cancellation and errors leave the prior revision intact. Protocol adapters stay at the boundary: they translate native or vendor wire formats while provider credentials and routing policy remain inside the server-side egress stack.
