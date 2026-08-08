# omp-llm-devin

`omp-llm-devin` integrates Devin Cascade with OMP's common LLM interfaces. It translates chat requests into Cascade's Connect-compatible protobuf API and decodes the server-streamed responses into turn events.

## Structure

- `src/lib.rs` provides `DevinChat`, implements the shared `Chat` interface, builds Cascade requests, and manages streamed response decoding. Its transport-independent `State`, `decode_response`, and `finish` entry points also support replaying recorded frames without a live connection.
- `src/wire.rs` contains the hand-written `prost` request, response, configuration, tool, usage, and service types used by the Cascade endpoint.

## Philosophy

The crate keeps provider-specific protocol details behind the shared LLM types and reports unsupported request features explicitly. Request transport and streaming are owned by the client, while decoding remains transport-independent and stateful. Tool calls complete the current turn and are answered by a later request rather than through an in-turn invocation channel.
