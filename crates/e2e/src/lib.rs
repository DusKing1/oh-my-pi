//! Shared bounded authority harness for OMP's executable acceptance proofs.
//!
//! Scenario bodies live in integration-test targets. This crate owns only the
//! reusable lifecycle, transport, fixture, and canonical-data support they use.

/// Reusable acceptance-test infrastructure.
pub mod support;
