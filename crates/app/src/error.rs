//! Production application error types.

use thiserror::Error;

use crate::endpoint::LocalEndpoint;

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
	/// Typed inference planning or execution failure.
	#[error(transparent)]
	Inference(#[from] omp_llm_inference::Error),
	/// Catalog compilation failure.
	#[error(transparent)]
	CatalogCompile(#[from] omp_llm_catalog::compile::CompileError),
	/// Embedded catalog loading failure.
	#[error("embedded catalog is unavailable")]
	CatalogSnapshot,
	/// Credential persistence failure.
	#[error(transparent)]
	CredentialStore(#[from] omp_llm_inference::auth::StoreError),
	/// Credential key-source failure.
	#[error(transparent)]
	CredentialKey(#[from] omp_llm_inference::auth::KeyError),
	#[cfg(feature = "local-applefm")]
	/// Apple Foundation Models failure.
	#[error(transparent)]
	AppleFm(#[from] omp_llm_inference::local::applefm::AppleFmError),
	/// Inference stream ended before its terminal completion event.
	#[error("inference stream ended without completion")]
	InferenceStreamUnterminated,
	/// Local inference stream ended before its terminal completion event.
	#[error("local inference stream ended without completion")]
	LocalInferenceStreamUnterminated,
	/// Local Apple Foundation Models support was not compiled in.
	#[error("local inference requires the `local-applefm` feature")]
	LocalFeatureDisabled,
	/// Neither `HOME` nor `OMP_DATA_DIR` is set in the environment.
	#[error("HOME or OMP_DATA_DIR must be set")]
	DataDirNotConfigured,
	/// Catalog compiler output would overwrite an input.
	#[error("catalog inputs and destination must be different files")]
	SameCatalogSourceAndDestination,
	/// Failed to connect to the local daemon endpoint.
	#[error("could not connect to {endpoint}: {source}")]
	ConnectGateway {
		/// Local endpoint attempted.
		endpoint: LocalEndpoint,
		/// RPC error.
		#[source]
		source:   omp_rpc::Error,
	},
	/// Requested auth operation is not supported by the active auth engine.
	#[error("authentication operation is unavailable: {0}")]
	AuthUnavailable(&'static str),
}

impl From<&'static omp_llm_catalog::snapshot::SnapshotError> for AppError {
	fn from(_: &'static omp_llm_catalog::snapshot::SnapshotError) -> Self {
		Self::CatalogSnapshot
	}
}

/// An application result.
pub type Result<T, E = AppError> = std::result::Result<T, E>;
