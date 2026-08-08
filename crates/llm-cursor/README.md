# omp-llm-cursor

`omp-llm-cursor` integrates Cursor's agent service with OMP's canonical chat and tool-execution interfaces. It owns Cursor's pinned protobuf and Connect framing, translates chat requests and streamed responses, and bridges provider-initiated shell invocations to an OMP executor.

## Structure

- `lib.rs` implements `CursorChat`, request assembly, incremental Connect envelope decoding, streamed interaction translation, and the in-turn invocation bridge. `CursorDecodeState`, `ConnectDecoder`, and `InvocationFramer` expose the stateful protocol boundaries used for frame replay and conversion.
- `wire.rs` contains the pinned Cursor protobuf message model used by the transport and execution paths.

## Philosophy

Keep Cursor-specific protocol details at the provider edge. The crate converts wire messages into canonical OMP events and invocation values, while preserving the live service's bidirectional streaming behavior even where the pinned schema describes a unary method. Decoding is incremental and avoids copying complete payloads; invocation output is framed on UTF-8 and ANSI-safe boundaries. Cancellation, timeout, and heartbeat behavior remain explicit so an executor cannot emit late frames after an aborted invocation.
