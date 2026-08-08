//! Links the vendored static `CPython`'s dependencies into the library and
//! packs the repo-provided Python modules (`crates/py/python`) plus the
//! pure-Python packages pinned in `requirements.txt` into a frozen-modules
//! blob.
//!
//! pyo3 links the libpython archive itself (via `PYO3_CONFIG_FILE`); that
//! archive already contains the interpreter core plus every builtin stdlib C
//! extension. This script supplies what the archive expects from outside:
//! dependency archives (OpenSSL, sqlite, ...), macOS frameworks, and system
//! libraries, all enumerated in `PYTHON.json`. Everything is emitted as
//! `rustc-link-search`/`rustc-link-lib` so it propagates to downstream
//! crates that depend on omp-py; only the final-link flags (`--ld-path`,
//! `-export_dynamic`) are per-binary and MUST be replicated by consumer
//! build scripts (see crates/py-smoke).

use std::{
	collections::BTreeSet,
	path::{Path, PathBuf},
	process::Command,
};

/// System libraries only `_tkinter` needs; it is not in the static inittab
/// (pbs ships it as a shared module), so its archive member is never pulled.
const TCL_LIBS: [&str; 2] = ["tcl9.0", "tcl9tk9.0"];

fn main() {
	let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
	let vendor = root.join("vendor/python");
	assert!(
		vendor.join("PYTHON.json").is_file(),
		"vendor/python missing — run scripts/fetch-python.sh first"
	);
	println!("cargo::rerun-if-changed={}", vendor.join("PYTHON.json").display());
	println!("cargo::rerun-if-changed={}", vendor.join("stdlib.bin").display());

	// pyo3-ffi configures itself from PYO3_CONFIG_FILE *before* this script
	// has any say; if it is unset or points elsewhere, pyo3 silently
	// introspects a host Python and links the wrong runtime. This repo's
	// .cargo/config.toml sets it for workspace members; external consumers
	// must set it themselves (environment or their own `[env]` section).
	println!("cargo::rerun-if-env-changed=PYO3_CONFIG_FILE");
	let expected = vendor.join("pyo3-config.txt").canonicalize().unwrap();
	let actual =
		std::env::var_os("PYO3_CONFIG_FILE").and_then(|v| PathBuf::from(v).canonicalize().ok());
	assert!(
		actual.as_deref() == Some(&expected),
		"PYO3_CONFIG_FILE must point at {} (found {:?}); set it before cargo runs — e.g. in the \
		 consumer's .cargo/config.toml: [env] PYO3_CONFIG_FILE = \"...\"",
		expected.display(),
		actual,
	);

	let json: serde_json::Value =
		serde_json::from_str(&std::fs::read_to_string(vendor.join("PYTHON.json")).unwrap()).unwrap();

	// compiler-rt: rustc links with -nodefaultlibs, so clang's runtime for
	// `@available` checks (___isPlatformVersionAtLeast) must come from here.
	let mut static_libs = BTreeSet::from(["clang_rt.osx".to_owned()]);
	let mut frameworks = BTreeSet::new();
	let mut system_libs = BTreeSet::new();
	let extensions = json["build_info"]["extensions"].as_object().unwrap();
	for variant in extensions.values().flat_map(|v| v.as_array().unwrap()) {
		for link in variant["links"].as_array().unwrap() {
			let name = link["name"].as_str().unwrap();
			if link["path_static"].is_string() {
				static_libs.insert(name.to_owned());
			} else if link["framework"].as_bool() == Some(true) {
				frameworks.insert(name);
			} else if !TCL_LIBS.contains(&name) {
				system_libs.insert(name);
			}
		}
	}

	// The vendored archives are LLVM-22 LTO bitcode, which Xcode's ld64 cannot
	// read; scripts/ld64.lld routes the link through a matching Homebrew lld.
	// Emitted here, not in .cargo/config.toml, so no absolute checkout path is
	// baked in and only this binary pays for the shim.
	if std::env::var("TARGET").as_deref() == Ok("aarch64-apple-darwin") {
		println!("cargo::rustc-link-arg=--ld-path={}", root.join("scripts/ld64.lld").display());
	}

	// Wheels' native extensions (.so) resolve CPython symbols from this
	// executable at dlopen; keep every global (code AND data like PyExc_*)
	// through dead-strip so the full C-API surface stays exported.
	println!("cargo::rustc-link-arg=-Wl,-export_dynamic");

	// Static archives propagate transitively: they bundle into this crate's
	// rlib and reach any downstream binary. Unreferenced members cost
	// nothing — they are only pulled when they resolve a symbol.
	println!("cargo::rustc-link-search=native={}", vendor.join("build/lib").display());
	for lib in &static_libs {
		println!("cargo::rustc-link-lib=static={lib}");
	}
	for framework in &frameworks {
		println!("cargo::rustc-link-lib=framework={framework}");
	}
	for lib in &system_libs {
		println!("cargo::rustc-link-lib={lib}");
	}

	// Repo-provided Python modules (omp_remote, ...) and the third-party
	// packages pinned in requirements.txt are marshalled by the vendored
	// interpreter — guaranteeing a matching bytecode format — into a second
	// frozen blob, registered next to the stdlib by lib.rs.
	let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
	let py_src = manifest.join("python");
	let requirements = manifest.join("requirements.txt");
	let packer = root.join("scripts/pack-pymodules.py");
	println!("cargo::rerun-if-changed={}", py_src.display());
	println!("cargo::rerun-if-changed={}", requirements.display());
	println!("cargo::rerun-if-changed={}", packer.display());
	let interpreter = ["python3.14td", "python3.14t"]
		.iter()
		.map(|name| vendor.join("install/bin").join(name))
		.find(|p| p.is_file())
		.expect("vendored interpreter missing — run scripts/fetch-python.sh first");
	let bundled = bundled_packages(&requirements, &vendor);
	let modules_blob = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("omp_modules.bin");
	let mut pack = Command::new(interpreter);
	pack.arg(&packer).arg(&py_src);
	if let Some(dir) = &bundled {
		pack.arg(dir);
	}
	let status = pack
		.arg(&modules_blob)
		.args(["--prefix", "<omp-py>"])
		.args(["--exclude", "*.dist-info", "__pycache__", "bin"])
		.status()
		.expect("failed to run pack-pymodules.py");
	assert!(status.success(), "pack-pymodules.py failed");

	// Baked into the library by lib.rs.
	println!("cargo::rustc-env=OMP_STDLIB_BLOB={}", vendor.join("stdlib.bin").display());
	println!("cargo::rustc-env=OMP_PY_MODULES_BLOB={}", modules_blob.display());
}

/// Locates the bundled third-party packages (pinned in `requirements.txt`,
/// fetched by scripts/fetch-python.sh into `vendor/python/bundled`), or
/// `None` when the manifest lists nothing. Build time does no network I/O
/// and never writes outside `OUT_DIR`: this only checks that the cached
/// tree's stamp matches the manifest text and fails with a pointer to the
/// fetch script when it is missing or stale.
fn bundled_packages(requirements: &Path, vendor: &Path) -> Option<PathBuf> {
	let spec = std::fs::read_to_string(requirements).unwrap_or_default();
	let listed = spec
		.lines()
		.map(str::trim)
		.any(|l| !l.is_empty() && !l.starts_with('#'));
	if !listed {
		return None;
	}
	let dir = vendor.join("bundled");
	let stamp = dir.join(".requirements.stamp");
	println!("cargo::rerun-if-changed={}", stamp.display());
	assert!(
		std::fs::read_to_string(&stamp).ok().as_deref() == Some(&spec),
		"vendor/python/bundled missing or stale vs {} — run scripts/fetch-python.sh",
		requirements.display()
	);
	Some(dir)
}
