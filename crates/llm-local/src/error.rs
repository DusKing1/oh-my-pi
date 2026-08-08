use omp_core::SmolStr;

/// Error returned by local model acquisition or inference.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	/// The operation was cancelled by its caller.
	#[error("operation cancelled")]
	Cancelled,
	/// A backend rejected model data or failed during inference.
	#[error("{backend}: {message}")]
	Backend {
		/// Runtime that produced the error.
		backend: &'static str,
		/// Runtime error message.
		message: SmolStr,
	},
	/// An input violated a model contract.
	#[error("invalid input: {0}")]
	InvalidInput(SmolStr),
	/// The requested accelerator is unavailable in this build or on this host.
	#[error("accelerator unavailable: {0}")]
	Unavailable(SmolStr),
	/// A Hugging Face operation failed.
	#[error(transparent)]
	Hub(#[from] hf_hub::HFError),
	/// A filesystem operation failed.
	#[error(transparent)]
	Io(#[from] std::io::Error),
	/// The model worker stopped before completing the request.
	#[error("model worker stopped")]
	WorkerStopped,
}

impl Error {
	/// Stable machine-readable category, mirroring the `TurnError.Kind`
	/// vocabulary of `omp.inference.v1` for callers bridging this crate onto
	/// the inference protocol.
	pub const fn kind(&self) -> ErrorKind {
		match self {
			Self::Cancelled => ErrorKind::Cancelled,
			Self::Backend { .. } | Self::Hub(_) | Self::Io(_) | Self::WorkerStopped => {
				ErrorKind::Upstream
			},
			Self::InvalidInput(_) => ErrorKind::InvalidInput,
			Self::Unavailable(_) => ErrorKind::Unsupported,
		}
	}

	#[cold]
	pub(crate) fn backend(backend: &'static str, error: impl std::fmt::Display) -> Self {
		Self::Backend { backend, message: error.to_string().into() }
	}

	#[cold]
	pub(crate) fn invalid(message: impl Into<SmolStr>) -> Self {
		Self::InvalidInput(message.into())
	}

	#[cold]
	pub(crate) fn unavailable(message: impl Into<SmolStr>) -> Self {
		Self::Unavailable(message.into())
	}
}

/// Stable category of an [`Error`], mirroring
/// `omp.inference.v1.TurnError.Kind`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ErrorKind {
	/// The operation was cancelled by its caller.
	Cancelled,
	/// An input violated a model contract.
	InvalidInput,
	/// The requested capability is unavailable in this build or on this host.
	Unsupported,
	/// A backend, download, or worker failed while serving the request.
	Upstream,
}

/// Result type used throughout `omp-llm-local`.
pub type Result<T> = std::result::Result<T, Error>;
