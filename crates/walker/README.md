# omp-walker

`omp-walker` provides reusable filesystem traversal and file-candidate discovery for globbing, search, AST scans, and shell builtins. It supports collected and streaming walks with configurable ordering, metadata detail, ignore handling, symbolic-link policy, filtering, ranking, cancellation heartbeats, and serial or parallel execution.

## Structure

- `src/lib.rs` defines walk requests and options, entry and candidate types, filters and predicates, visitor APIs, collection and streaming entry points, traversal ordering, and platform-specific directory-reading backends for macOS, Linux, Windows, and other targets.
- `src/cache.rs` owns the shared scan cache, path normalization and classification helpers, cache invalidation, and the centralized Rayon worker pool used by traversal and candidate processing.

## Philosophy

The crate keeps filesystem discovery independent of higher-level bindings: callers supply visitors, predicates, sinks, and heartbeats through plain Rust interfaces. Traversal policy is explicit, platform-native directory reads are used where supported, and shared caching and bounded parallel work avoid repeating expensive scans while preserving serial paths for smaller workloads and restrictive walk options.
