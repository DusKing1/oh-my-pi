# omp-llm-egress

`omp-llm-egress` provides the HTTP egress policy and transport primitives used to send LLM provider requests. It separates replayable, pre-commit work from committed response streams so routing, credentials, admission control, and retries remain explicit at the Tower service boundary.

## Structure

- `auth_inject` selects and redeems credential leases for each attempt, including coordinated refresh after an unauthorized response without exposing secret bytes.
- `client` supplies the pooled Hyper and rustls transport, buffered request bodies, and the response-header timeout layer.
- `limits` applies provider-and-credential keyed queue, concurrency, rate, and block policy.
- `proxy` resolves direct, HTTP, and SOCKS5 routes from an environment snapshot, including required bypasses for local and cloud metadata endpoints.
  Its boxed connector future is the sanctioned cold connection-establishment boundary; request middleware futures remain unboxed.
- `retry` models replayable requests, committed responses, pre-commit failures, and jittered retry policy.

## Philosophy

Egress policy should be data-driven, testable, and scoped to the credential and provider actually making an attempt. Requests remain buffered and replayable only until the first meaningful provider event has decoded and validated. After that commit point, failures belong to the response stream rather than the retry error channel. Narrow interfaces keep broker-owned credentials and block persistence outside the transport crate while preserving the dependency boundary.
