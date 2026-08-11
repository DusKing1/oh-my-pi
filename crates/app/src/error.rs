//! Production application error types.

use std::path::PathBuf;

use omp_core::Str;
use omp_llm_gateway::local::LocalEndpoint;
use thiserror::Error;

/// Failures that can occur during command dispatch or application execution.
#[derive(Debug, Error)]
pub enum AppError {
	/// Daemon startup or serving failure.
	#[error(transparent)]
	Daemon(#[from] crate::daemon::DaemonError),

	/// RPC communication failure.
	#[error(transparent)]
	Rpc(#[from] omp_rpc::Error),

	/// gRPC status error.
	#[error(transparent)]
	Status(#[from] tonic::Status),

	/// Local filesystem or I/O failure.
	#[error(transparent)]
	Io(#[from] std::io::Error),

	/// Broker durable store error.
	#[error(transparent)]
	BrokerStore(#[from] omp_llm_broker::store::StoreError),
	/// Common LLM types or validation error.
	#[error(transparent)]
	LlmTypes(#[from] omp_llm_types::Error),

	/// Broker OAuth engine error.
	#[error(transparent)]
	BrokerOAuth(#[from] omp_llm_broker::oauth::OAuthError),

	/// Broker CLI command failure.
	#[error(transparent)]
	BrokerCli(#[from] omp_llm_broker::cli::CliError),

	/// Model catalog error.
	#[error(transparent)]
	Catalog(#[from] omp_llm_catalog::models::CatalogError),

	/// In-process local inference error.
	#[error(transparent)]
	LocalInference(#[from] omp_llm_local::Error),

	/// Inference gateway returned an error event.
	#[error("inference failed: {detail}")]
	InferenceFailed {
		/// Error detail string from gateway.
		detail: Str,
	},

	/// Inference stream ended before emitting a terminal outcome.
	#[error("inference stream ended without a terminal outcome")]
	InferenceStreamUnterminated,

	/// Local inference stream ended before emitting a terminal outcome.
	#[error("local inference stream ended without a terminal outcome")]
	LocalInferenceStreamUnterminated,

	/// `auth migrate` was invoked without providing `--sqlite` or `--json-file`.
	#[error("auth migrate requires --sqlite or --json-file")]
	AuthMigrateArgsRequired,

	/// Neither `HOME` nor `OMP_DATA_DIR` is set in the environment.
	#[error("HOME or OMP_DATA_DIR must be set")]
	DataDirNotConfigured,

	/// Catalog import source and destination paths are identical.
	#[error("catalog source and destination must be different files")]
	SameCatalogSourceAndDestination,

	/// Could not read the catalog import source file.
	#[error("could not read {path:?}: {source}")]
	ReadCatalogSource {
		/// Source file path.
		path:   PathBuf,
		/// Underlying I/O error.
		#[source]
		source: std::io::Error,
	},

	/// Could not write the catalog import destination file.
	#[error("could not write {path:?}: {source}")]
	WriteCatalogDestination {
		/// Destination file path.
		path:   PathBuf,
		/// Underlying I/O error.
		#[source]
		source: std::io::Error,
	},

	/// Failed to connect to the local gateway endpoint.
	#[error("could not connect to {endpoint}: {source}")]
	ConnectGateway {
		/// Local endpoint attempted.
		endpoint: LocalEndpoint,
		/// RPC error.
		#[source]
		source:   omp_rpc::Error,
	},
}

/// An application result.
pub type Result<T, E = AppError> = std::result::Result<T, E>;
