//! OpenTelemetry instrumentation for the agent loop, ported 1:1 from the
//! previous TypeScript implementation.
//!
//! Wire compatibility is the contract: span names, attribute keys, metric
//! instruments, log-record shapes, and environment-variable knobs are
//! **identical** to `pi`'s, so existing dashboards, collectors, and alerts
//! keep working across the rewrite. Where `pi` extends the OpenTelemetry
//! `GenAI` semantic conventions it does so under the `pi.gen_ai.*` /
//! `pi.omp.*` prefixes; those prefixes are preserved verbatim rather than
//! renamed.
//!
//! Layering mirrors the original split:
//! - [`attrs`] / [`semconv`] — the constant vocabulary (attribute keys, span
//!   names, enum values, provider normalization).
//! - [`span`] / [`content`] — span lifecycle and opt-in content capture.
//! - [`metrics`] / [`collector`] — instruments and per-run aggregation.
//! - [`export`] / [`redact`] — OTLP bootstrap, configuration, and scrubbing.

pub mod attrs;
pub mod collector;
pub mod config;
pub mod content;
pub mod export;
pub mod metrics;
pub mod redact;
pub mod semconv;
pub mod span;
