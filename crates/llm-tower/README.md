# omp-llm-tower

Tower middleware for the internal `omp.inference.v1` provider-attempt boundary. It wraps a `Service<TurnRequest>` that yields a stream of `TurnEvent`s below the turn coordinator's commit and idempotency machinery; it is not middleware for the public bidirectional `Turn` RPC.

## Structure

- `preflight`, `recovery`, `select`, and `refresh` handle quota admission, classified terminal-error recovery, credential routing, and OAuth refresh.
- `learn` and `resample` implement sticky capability fallback and pre-commit re-sampling for empty completions or thinking loops.
- `admission` and `timeout` bound concurrent attempts and enforce connect, first-event, and idle deadlines.
- `tap` observes requests and frames without changing them.
- `stack` contains typed facet middleware for capability checks, routing, credential rotation, usage metering, transport encoding, and post-commit stream combinators.
- `testing` provides the crate's hidden scripted attempt-service and frame helpers.

## Philosophy

Middleware is ordered by where policy is safe to apply. Retry, suppression, and re-dispatch stay below commit, while post-commit adapters operate only on canonical response streams. Errors and pre-wire rejections remain typed rather than inferred from provider prose, credential identity travels beside the protobuf payload, and admission permits live for the full response stream. Short-circuit responses use concrete streams to avoid allocation, and diagnostics remain observational with redaction left to the sink that exports data.
