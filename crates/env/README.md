# omp-env

`omp-env` is the typed client boundary for OMP's `env/v1` protocol. It correlates invocation, command, session, named-process, and blob requests over a bidirectional frame transport while exposing server events as asynchronous request streams.

The crate deliberately owns no world resources. Files, processes, document leases, workspace search, and blob storage remain behind the environment service. In-process and remote deployments feed the same frame client; per-invocation and per-command `RunGuard`s provide nonblocking, request-scoped cancellation without ending server-owned sessions. Detached work must relinquish its guard explicitly.
