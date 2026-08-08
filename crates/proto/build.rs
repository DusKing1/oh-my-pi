//! Compiles every `.proto` under `proto/` with protox + tonic-prost-build.
//!
//! protox is a pure-Rust protobuf compiler, so no system `protoc` install is
//! required. Generated Rust lands in `OUT_DIR` as one file per protobuf
//! package (`omp.thread.v1.rs`, ...) and is pulled in by `include!` in
//! `src/lib.rs`.
//!
//! Codegen choices:
//! - `bytes` fields decode into `Bytes` (O(1) clone, zero-copy slices).
//! - Maps are `BTreeMap` for deterministic serialization.
//! - Every type derives serde; well-known types (`google.protobuf.Struct`) are
//!   compiled locally instead of mapped to `prost-types` so the derives reach
//!   them too. This is Rust-native serde, not the proto3 JSON mapping (enums as
//!   ints, `snake_case` fields).
//!
//! Message bindings are always generated. The `tonic` feature additionally
//! emits gRPC client and server bindings into the same package files.

use std::{
	fs,
	path::{Path, PathBuf},
};

fn main() {
	let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("proto");
	println!("cargo::rerun-if-changed={}", root.display());

	let mut protos = Vec::new();
	collect(&root, &mut protos);
	protos.sort();
	// `src/lib.rs` includes `google.protobuf.rs` unconditionally so the serde
	// derives reach the well-known types; protox serves this file from its
	// embedded descriptor set even though it is not under `proto/`.
	protos.push(PathBuf::from("google/protobuf/struct.proto"));

	let fds = protox::compile(&protos, [&root]).expect("protox failed to compile .proto sources");
	let generate_services = std::env::var_os("CARGO_FEATURE_TONIC").is_some();
	tonic_prost_build::configure()
		.build_client(generate_services)
		.build_server(generate_services)
		.bytes(".")
		.btree_map(".")
		.compile_well_known_types(true)
		.type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
		.message_attribute(".", "#[serde(default)]")
		.compile_fds(fds)
		.expect("tonic-prost-build failed to generate Rust code");
}

/// Recursively gathers `.proto` files under `dir`.
fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
	for entry in fs::read_dir(dir).expect("proto/ directory missing") {
		let path = entry.expect("unreadable dir entry").path();
		if path.is_dir() {
			collect(&path, out);
		} else if path.extension().is_some_and(|ext| ext == "proto") {
			out.push(path);
		}
	}
}
