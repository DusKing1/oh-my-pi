# omp-shell

`omp-shell` is the batteries-included facade for the workspace's shell implementation. It presents the parser and execution API from `omp-shell-engine` alongside the utility and process builtin registries from `omp-shell-builtins`, and provides the standalone `omp-sh` executable.

## Structure

- `src/lib.rs` re-exports the engine API and the `utility_builtins` and `process_builtins` registry constructors.
- `src/bin/omp-sh.rs` composes the engine's default builtins with those registries and runs commands supplied through `-c`, a script path, or standard input.
- `tests/exec.rs` contains integration coverage for the composed shell.

## Philosophy

Keep this crate as a thin composition boundary: parsing and execution belong in `omp-shell-engine`, builtin implementations belong in `omp-shell-builtins`, and this package joins them into a convenient library and executable. The executable deliberately constructs a non-interactive shell without loading profile or rc files so invocation behavior remains explicit.

`omp-shell-engine` incorporates and adapts source from `brush-core` 0.5.0 and `brush-parser` 0.4.0 by Reuben Olinsky under the MIT License. See `LICENSE` for the attribution, license terms, and notes about the local adaptations.
