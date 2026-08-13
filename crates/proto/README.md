# omp-proto

`omp-proto` owns the workspace's wire-level Protobuf definitions and the Rust bindings generated from them. Message bindings are always available; the optional `tonic` feature also generates gRPC clients and servers.

## Structure

- `proto/omp/thread/v1` defines the canonical conversation AST.
- `proto/omp/inference/v1` defines inference turns, models, media, search, and shared inference messages.
- `proto/omp/auth/v1`, `gateway/v1`, `blob/v1`, and `document/v1` define authentication, connection negotiation, blob transfer, and document synchronization protocols.
- `proto/omp/env/v1` defines the environment boundary as three multiplexed planes: streaming tool invocations, exec sessions plus named processes, and content-addressed blobs. The same varint-framed messages are used in process, over owner-only UDS, and remotely.
- `proto/omp/toolhost/v1` defines the varint-framed stdio protocol between the environment host and a supervised Python worker.
- `build.rs` recursively compiles the schemas with the pure-Rust `protox` compiler and `tonic-prost-build`, writing one generated Rust file per Protobuf package to `OUT_DIR`.
- `src/lib.rs` includes those generated package files, re-exports their modules, and exposes the wire-visible `SCHEMA_REV`.

## Philosophy

This crate is the process-boundary representation; owning crates retain their native in-process types. Code generation avoids a system `protoc` dependency, uses `Bytes` for inexpensive byte-field clones and slices, and uses `BTreeMap` for deterministic map serialization. Generated messages derive Serde using Rust-native enum and field representations rather than the proto3 JSON mapping. Service bindings remain feature-gated so consumers that only need messages do not pull in the gRPC runtime.
