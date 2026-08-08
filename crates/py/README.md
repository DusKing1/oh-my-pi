# omp-py

`omp-py` embeds a statically linked, free-threaded CPython 3.14 interpreter in a Rust process. It freezes the Python standard library, repository-provided modules, and pinned pure-Python packages into the binary, while allowing native wheels to be loaded from a configured site-packages directory.

## Structure

- `src/lib.rs` exposes `Engine` and `Builder`, installs the frozen-module tables, boots CPython in isolated mode, and provides the default site-packages location.
- `build.rs` links the vendored interpreter's native dependencies and packs project modules and bundled packages into frozen-module blobs without network access.
- `python/` contains repository-provided Python modules, including `omp_remote`; `requirements.txt` pins bundled pure-Python packages.
- `src/bin/demo.rs` is the crate's `omp-demo` binary.
- `THIRD-PARTY-NOTICES.txt` records notices for bundled Python packages and is also exposed through `THIRD_PARTY_LICENSES`.

## Philosophy

Keep the interpreter self-contained and deterministic: Python boots once per process in isolated mode, and frozen modules remain available to subinterpreters without relying on a host Python installation or ordinary filesystem imports. The frozen data is stored uncompressed so CPython can point directly into static binary data and the operating system can avoid paging unused modules.

The deliberate filesystem exception is site-packages, because native extension modules must be loaded from disk. Binaries supporting native wheels must export CPython's C API at final link time (for example, with `-Wl,-export_dynamic`); this crate applies the flag to its own binaries, while downstream binaries must apply the equivalent final-link configuration themselves.
