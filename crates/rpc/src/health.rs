//! Standard gRPC health service integration.

use tonic_health::pb::health_server::{Health, HealthServer};
pub use tonic_health::server::HealthReporter;

/// Create a standard `grpc.health.v1` reporter and service.
///
/// Liveness is exposed through `grpc.health.v1.Health/Check`. Servers set
/// global or per-service serving status through the returned
/// [`HealthReporter`].
pub fn health_service() -> (HealthReporter, HealthServer<impl Health>) {
	tonic_health::server::health_reporter()
}
