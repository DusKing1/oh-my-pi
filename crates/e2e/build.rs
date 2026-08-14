//! Applies omp-py's final-link requirements to the executable acceptance host.

use std::path::{Path, PathBuf};

fn main() {
	println!("cargo::rerun-if-env-changed=PYO3_CONFIG_FILE");

	let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
	let vendor = std::env::var_os("PYO3_CONFIG_FILE")
		.map(PathBuf::from)
		.and_then(|p| {
			p.canonicalize()
				.ok()
				.or_else(|| manifest.join("../..").join(&p).canonicalize().ok())
		})
		.and_then(|p| p.parent().map(Path::to_path_buf));

	if let Some(vendor_dir) = &vendor {
		let marker = vendor_dir.join("needs-lld");
		println!("cargo::rerun-if-changed={}", marker.display());
		if marker.is_file() {
			let shim = manifest.join("../py/scripts/ld64.lld");
			println!("cargo::rerun-if-changed={}", shim.display());
			assert!(
				shim.is_file(),
				"omp-e2e's release macOS link requires omp-py's ld64.lld shim at {}; restore \
				 crates/py/scripts/ld64.lld",
				shim.display()
			);
			println!("cargo::rustc-link-arg=--ld-path={}", shim.display());
		}
	}
	println!("cargo::rustc-link-arg=-Wl,-export_dynamic");
}
