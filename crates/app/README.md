# omp-app

`omp-app` is the production application boundary for OMP inference. It ships the `omp` binary for serving the gateway, running remote or local inference, managing broker credentials, and importing the model catalog.

## Structure

- `main` owns process startup, telemetry initialization, and error exit status.
- `cli` defines and dispatches the public command tree.
- `auth_backend` adapts the shared production HTTP egress client to the credential broker.
- `daemon` composes and supervises the production gateway runtime.

## Philosophy

The application crate contains composition, not alternate implementations. Provider traffic uses the shared egress stack, authentication delegates to the broker, local generation uses the local inference facade, and serving delegates to the gateway daemon assembly. Commands reject incomplete configurations rather than simulating success.
