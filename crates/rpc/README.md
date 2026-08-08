# omp-rpc

`omp-rpc` provides the transport and protocol-negotiation plumbing for Oh My Pi's gRPC services. It supports owner-only local Unix-domain sockets and TCP connections secured with mutual TLS, performs a schema-aware hello handshake, and exposes standard gRPC health reporting.

## Structure

- `health` wraps gRPC liveness and per-service readiness reporting.
- `hello` implements the initial peer handshake and schema-revision compatibility checks.
- `tls` builds client and server TLS configuration.
- `uds` listens for and connects to Unix-domain socket transports.
- The crate-level `Error` type unifies I/O, transport, RPC, TLS, schema-negotiation, and unsupported-transport failures.

## Philosophy

Transport concerns stay separate from service behavior while local and network clients share the same protocol. Connections negotiate compatibility before exchanging application data so protobuf unknown-field behavior cannot silently discard data from a newer client. Health reporting uses the standard `grpc.health.v1` protocol rather than a project-specific alternative.
