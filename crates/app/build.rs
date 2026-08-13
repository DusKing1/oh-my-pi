//! Applies final-link requirements that cannot propagate from `omp-py`'s
//! library build script to `omp-app` binaries, examples, and test executables.
//!
//! On aarch64 macOS the vendored CPython archives contain LLVM LTO bitcode,
//! so the final link must use `omp-py`'s ld64-to-lld shim. A missing shim is a
//! checkout/build-input error: this script fails immediately with the expected
//! path rather than allowing the final linker to emit misleading archive
//! errors. Every target also retains and exports CPython's global C API so
//! native wheels can resolve code and data symbols when they are loaded.

use std::path::PathBuf;

fn main() {
	println!("cargo::rerun-if-env-changed=TARGET");

	let target = std::env::var("TARGET").expect("Cargo must provide TARGET to omp-app/build.rs");
	if target == "aarch64-apple-darwin" {
		let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
		let shim = manifest.join("../py/scripts/ld64.lld");
		println!("cargo::rerun-if-changed={}", shim.display());
		assert!(
			shim.is_file(),
			"omp's aarch64 macOS link requires omp-py's ld64.lld shim at {}; restore \
			 crates/py/scripts/ld64.lld",
			shim.display()
		);
		println!("cargo::rustc-link-arg=--ld-path={}", shim.display());
	}

	println!("cargo::rustc-link-arg=-Wl,-export_dynamic");
}
