# omp-grep

`omp-grep` provides a synchronous, binding-free search engine for in-memory bytes and workspace files. It uses ripgrep's regex engine with PCRE2 fallback, bounded leading-window reads for large files, binary detection, context collection, and deterministic result aggregation.

## Structure

- `src/lib.rs` defines the public options, matches, results, and typed errors; compiles matchers; searches byte slices; discovers files through `omp-walker`; and executes file searches with walker-owned parallel workers.

## Philosophy

The crate owns matching and counting, while callers own tool schemas and presentation. Filesystem discovery stays delegated to `omp-walker`, retained output uses `omp-core` strings and inline context storage, and bounded reads prevent a search from materializing arbitrarily large files.
