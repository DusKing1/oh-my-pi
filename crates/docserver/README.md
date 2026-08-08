# omp-docserver

`omp-docserver` is the local document authority for an OMP environment. It coordinates document state with filesystem access, revisions and rebased edits, transactions, file watching, and language-server sessions, and exposes the service over framed protocol connections.

## Structure

- `actor`, `types`, and `environment` define the document store, authority model, shared environment, and connection-local sessions.
- `fs` and `path_ops` provide portable filesystem values and actor-aware path operations; `watch` classifies and tracks file changes.
- `transaction`, `rebase`, `summary`, and `edit_adapter` validate and apply edits, manage transactions, summarize content, and lower supported edit formats.
- `lsp`, `lsp_registry`, and `position` manage language-server lifecycle and synchronization, server registration, and checked position/text-edit conversion.
- `protocol`, `connection`, `wire`, and `daemon` implement protobuf request handling, concurrent connections, bounded length-delimited framing, and the long-lived server process.

## Philosophy

The crate keeps one project-scoped authority over document state while isolating connection-specific behavior in sessions. Filesystem concepts use portable value types, edit application is revision-aware and validated before mutation, and protocol boundaries use explicit bounded framing. Language-server and edit-format integrations are adapters around the document authority rather than independent sources of state.
