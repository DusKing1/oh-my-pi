# omp-llm-broker

`omp-llm-broker` is the daemon-owned authentication broker for LLM provider credentials. It keeps provider secrets inside the broker while exposing credential metadata, login flows, usage information, and scoped token operations through the `omp.auth.v1` tonic service.

## Structure

- `service` implements the tonic authentication API and translates between broker domain types and protocol messages.
- `store` owns SQLite persistence for credentials, mutation deltas, usage history, and per-client accounting, including in-process single-flight refresh coordination.
- `sealed` contains redacted, zeroizing secret values plus credential leases and request-ready authentication.
- `oauth` runs catalog-driven PKCE, device-code, and custom OAuth exchanges through injected HTTP, browser, and time capabilities.
- `usage` fetches and normalizes provider quota data, coordinates its durable cache, and supports declarative and provider-specific flows.
- `cli` provides the mountable `omp auth` command tree, migration readers, and human- or JSON-oriented rendering.

## Philosophy

The crate boundary is the credential security boundary: clients receive metadata rather than serializable token bytes, and secret construction and redemption remain broker-internal. A single daemon owns persistence and refresh coordination, avoiding cross-process lease and compare-and-swap machinery. Provider differences are represented in shared catalog data or small explicit hooks, while injected transports and clocks keep network and timing behavior separate from orchestration.
