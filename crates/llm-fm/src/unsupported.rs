use omp_core::SmolStr;
use tokio_util::sync::CancellationToken;

use crate::{
	AppleFmAvailability, AppleFmError, AppleFmErrorCode, AppleFmGeneration, AppleFmOptions, Result,
};

const PLATFORM_MESSAGE: &str =
	"Apple Foundation Models requires macOS 26 or later on an eligible Apple Silicon Mac";

pub(super) fn availability() -> AppleFmAvailability {
	AppleFmAvailability { available: false, reason: Some("macOS Apple Silicon only".into()) }
}

pub(super) fn generate(
	_options: AppleFmOptions,
	_on_delta: impl FnMut(SmolStr) -> bool,
	_cancel: &CancellationToken,
) -> Result<AppleFmGeneration> {
	Err(AppleFmError::new(AppleFmErrorCode::ModelUnavailable, PLATFORM_MESSAGE))
}
