//! Applies omp-py's final-link requirements to tool integration tests.

use std::path::PathBuf;

fn main() {
	println!("cargo::rerun-if-env-changed=TARGET");
	let target = std::env::var("TARGET").expect("Cargo must provide TARGET to omp-tools/build.rs");
	if target == "aarch64-apple-darwin" {
		let shim = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../py/scripts/ld64.lld");
		println!("cargo::rerun-if-changed={}", shim.display());
		assert!(
			shim.is_file(),
			"omp-tools tests require omp-py's ld64.lld shim at {}; restore crates/py/scripts/ld64.lld",
			shim.display()
		);
		println!("cargo::rustc-link-arg=--ld-path={}", shim.display());
	}
	println!("cargo::rustc-link-arg=-Wl,-export_dynamic");
}
