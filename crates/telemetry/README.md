# omp-telemetry

`omp-telemetry` provides OpenTelemetry instrumentation for OMP's agent loop. It preserves the established telemetry wire contract, including span names, attribute keys, metric instruments, log-record shapes, environment-variable controls, and the existing `pi.gen_ai.*` and `pi.omp.*` extensions.

## Structure

- `attrs` and `semconv` define the telemetry vocabulary, span names, enum values, and provider normalization.
- `span` and `content` manage span lifecycles and opt-in content capture.
- `metrics` and `collector` define instruments and aggregate data for each run.
- `config`, `export`, and `redact` handle configuration, OTLP setup, and sensitive-data scrubbing.

## Philosophy

Wire compatibility is the primary constraint: existing collectors, dashboards, and alerts should continue to work across the Rust rewrite. The crate keeps vocabulary, instrumentation, aggregation, export, and redaction in distinct modules, and content capture remains explicit rather than automatic.
