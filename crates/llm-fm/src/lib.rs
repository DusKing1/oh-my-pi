//! Async Rust bindings for Apple's on-device Foundation Models runtime.
//!
//! The framework is loaded dynamically, so this crate builds on every platform
//! while generation remains available only on eligible Apple Silicon Macs with
//! Apple Intelligence enabled.
//!
//! # Example
//!
//! ```no_run
//! use omp_llm_fm::AppleFm;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = AppleFm::load().await?;
//! let response = model
//! 	.complete("Summarize on-device inference in one sentence.")
//! 	.await?;
//! println!("{response}");
//! # Ok(())
//! # }
//! ```

use std::{
	fmt,
	pin::Pin,
	task::{Context, Poll},
	time::Duration,
};

use futures::{Stream, channel::mpsc};
use omp_core::Str;
use tokio::task::JoinError;
use tokio_util::sync::CancellationToken;

#[cfg(target_os = "macos")]
mod abi;
mod chat;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(not(target_os = "macos"))]
mod unsupported;

pub use chat::{APPLE_INTELLIGENCE_SYSTEM_PROMPT, AppleFmChat, AppleFmEngine};
#[cfg(target_os = "macos")]
use macos as platform;
#[cfg(not(target_os = "macos"))]
use unsupported as platform;

const RUN_TIMEOUT: Duration = Duration::from_secs(30);

/// Stable category attached to an [`AppleFmError`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AppleFmErrorCode {
	/// The caller supplied an invalid request option.
	InvalidInput,
	/// The caller cancelled generation.
	Cancelled,
	/// Generation exceeded the runtime's request deadline.
	TimedOut,
	/// The device or system model cannot currently run the request.
	ModelUnavailable,
	/// This hardware is not eligible for Apple Intelligence.
	DeviceNotEligible,
	/// Apple Intelligence is disabled in System Settings.
	AppleIntelligenceNotEnabled,
	/// The on-device model has not finished downloading or preparing.
	ModelNotReady,
	/// The prompt exceeded the system model's context window.
	ContextOverflow,
	/// Apple's safety policy rejected the request.
	GuardrailBlocked,
	/// Guided generation is unsupported for this request.
	UnsupportedGuide,
	/// The current language or locale is unsupported.
	UnsupportedLocale,
	/// The framework could not decode a response.
	DecodingFailure,
	/// The system model rate-limited the request.
	RateLimited,
	/// Another process-local request is already active.
	ConcurrentRequests,
	/// The Foundation Models or Swift runtime failed unexpectedly.
	Runtime,
}

impl fmt::Display for AppleFmErrorCode {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(match self {
			Self::InvalidInput => "invalid_input",
			Self::Cancelled => "cancelled",
			Self::TimedOut => "timed_out",
			Self::ModelUnavailable => "model_unavailable",
			Self::DeviceNotEligible => "device_not_eligible",
			Self::AppleIntelligenceNotEnabled => "apple_intelligence_not_enabled",
			Self::ModelNotReady => "model_not_ready",
			Self::ContextOverflow => "context_overflow",
			Self::GuardrailBlocked => "guardrail_blocked",
			Self::UnsupportedGuide => "unsupported_guide",
			Self::UnsupportedLocale => "unsupported_locale",
			Self::DecodingFailure => "decoding_failure",
			Self::RateLimited => "rate_limited",
			Self::ConcurrentRequests => "concurrent_requests",
			Self::Runtime => "runtime_error",
		})
	}
}

/// Error returned by Apple Foundation Models availability checks or generation.
#[derive(Clone, Debug, thiserror::Error)]
#[error("{message}")]
pub struct AppleFmError {
	code:    AppleFmErrorCode,
	message: Str,
}

impl AppleFmError {
	/// Stable machine-readable error category.
	pub const fn code(&self) -> AppleFmErrorCode {
		self.code
	}

	/// Native or platform diagnostic suitable for logs and user-facing errors.
	pub fn message(&self) -> &str {
		self.message.as_str()
	}

	fn new(code: AppleFmErrorCode, message: impl Into<Str>) -> Self {
		Self { code, message: message.into() }
	}

	fn cancelled() -> Self {
		Self::new(AppleFmErrorCode::Cancelled, "Apple Foundation Models generation was cancelled")
	}

	fn timed_out() -> Self {
		Self::new(
			AppleFmErrorCode::TimedOut,
			"Apple Foundation Models generation exceeded 30 seconds",
		)
	}

	fn runtime(message: impl Into<Str>) -> Self {
		Self::new(AppleFmErrorCode::Runtime, message)
	}
}

/// Result type used by `omp-llm-fm` operations and streams.
pub type Result<T, E = AppleFmError> = std::result::Result<T, E>;

/// Current usability of Apple's on-device system language model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppleFmAvailability {
	/// Whether the system model can generate responses now.
	pub available: bool,
	/// Stable unavailability reason or native loading diagnostic.
	pub reason:    Option<Str>,
}

/// Controls one Apple Foundation Models request.
#[derive(Clone, Debug, PartialEq)]
pub struct AppleFmOptions {
	/// User prompt sent to the on-device model.
	pub prompt:        Str,
	/// Optional instructions applied to the model session.
	pub system_prompt: Option<Str>,
	/// Enables Apple's permissive content-transformations guardrail mode.
	pub permissive:    bool,
	/// Optional sampling temperature.
	pub temperature:   Option<f64>,
	/// Optional maximum number of response tokens.
	pub max_tokens:    Option<u32>,
}

impl AppleFmOptions {
	/// Creates a request with the framework's default guardrails and sampling.
	pub fn new(prompt: impl Into<Str>) -> Self {
		Self {
			prompt:        prompt.into(),
			system_prompt: None,
			permissive:    false,
			temperature:   None,
			max_tokens:    None,
		}
	}

	/// Applies instructions to the model session.
	pub fn system_prompt(mut self, prompt: impl Into<Str>) -> Self {
		self.system_prompt = Some(prompt.into());
		self
	}

	/// Selects Apple's permissive content-transformations guardrail mode.
	pub const fn permissive(mut self, permissive: bool) -> Self {
		self.permissive = permissive;
		self
	}

	/// Sets the sampling temperature.
	pub const fn temperature(mut self, temperature: f64) -> Self {
		self.temperature = Some(temperature);
		self
	}

	/// Limits the number of response tokens.
	pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
		self.max_tokens = Some(max_tokens);
		self
	}
}

/// Completed response and byte-derived token estimates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppleFmGeneration {
	/// Complete generated response.
	pub content:                     Str,
	/// Approximate prompt token count because the framework exposes no
	/// tokenizer.
	pub prompt_tokens_estimated:     u32,
	/// Approximate completion token count because the framework exposes no
	/// tokenizer.
	pub completion_tokens_estimated: u32,
	/// Apple's documented on-device context budget from TN3193.
	pub context_size_documented:     u32,
}

/// Incremental event produced by [`AppleFmStream`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AppleFmEvent {
	/// Newly generated text not present in the prior snapshot.
	Delta(Str),
	/// Canonical completed response and usage estimates.
	Finished(AppleFmGeneration),
}

/// Handle to Apple's process-local on-device system language model.
#[derive(Clone, Copy, Debug, Default)]
pub struct AppleFm;

impl AppleFm {
	/// Loads the system framework and verifies that the model is ready.
	pub async fn load() -> Result<Self> {
		let availability = Self::availability().await?;
		if availability.available {
			return Ok(Self);
		}
		let reason = availability
			.reason
			.unwrap_or_else(|| "model_unavailable".into());
		Err(AppleFmError::new(availability_error_code(reason.as_str()), reason))
	}

	/// Checks whether Apple Foundation Models can generate on this machine.
	pub async fn availability() -> Result<AppleFmAvailability> {
		tokio::task::spawn_blocking(platform::availability)
			.await
			.map_err(join_error)
	}

	/// Generates one complete response, respecting cancellation and the
	/// 30-second deadline.
	pub async fn generate(
		&self,
		options: AppleFmOptions,
		cancel: CancellationToken,
	) -> Result<AppleFmGeneration> {
		run_generation(options, cancel, |_| true).await
	}

	/// Generates one complete response with the framework's default request
	/// settings.
	pub async fn complete(&self, prompt: impl Into<Str>) -> Result<Str> {
		self
			.generate(AppleFmOptions::new(prompt), CancellationToken::new())
			.await
			.map(|generation| generation.content)
	}

	/// Starts a cancellable stream of response deltas followed by one completed
	/// response.
	pub fn stream(&self, options: AppleFmOptions) -> Result<AppleFmStream> {
		validate_options(&options)?;
		let cancel = CancellationToken::new();
		let task_cancel = cancel.clone();
		let (tx, rx) = mpsc::unbounded();
		tokio::spawn(async move {
			let delta_tx = tx.clone();
			let result = run_generation(options, task_cancel, move |delta| {
				delta_tx
					.unbounded_send(Ok(AppleFmEvent::Delta(delta)))
					.is_ok()
			})
			.await;
			match result {
				Ok(generation) => {
					let _ = tx.unbounded_send(Ok(AppleFmEvent::Finished(generation)));
				},
				Err(error) if error.code() == AppleFmErrorCode::Cancelled && tx.is_closed() => {},
				Err(error) => {
					let _ = tx.unbounded_send(Err(error));
				},
			}
		});
		Ok(AppleFmStream { rx, cancel })
	}
}

/// Asynchronous event stream returned by [`AppleFm::stream`].
pub struct AppleFmStream {
	rx:     mpsc::UnboundedReceiver<Result<AppleFmEvent>>,
	cancel: CancellationToken,
}

impl AppleFmStream {
	/// Requests cancellation of the active Foundation Models task.
	pub fn cancel(&self) {
		self.cancel.cancel();
	}
}

impl Stream for AppleFmStream {
	type Item = Result<AppleFmEvent>;

	fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		Pin::new(&mut self.rx).poll_next(context)
	}
}

impl Drop for AppleFmStream {
	fn drop(&mut self) {
		self.cancel.cancel();
	}
}

async fn run_generation(
	options: AppleFmOptions,
	cancel: CancellationToken,
	on_delta: impl FnMut(Str) -> bool + Send + 'static,
) -> Result<AppleFmGeneration> {
	validate_options(&options)?;
	if cancel.is_cancelled() {
		return Err(AppleFmError::cancelled());
	}
	let work_cancel = cancel.child_token();
	let blocking_cancel = work_cancel.clone();
	let mut task =
		tokio::task::spawn_blocking(move || platform::generate(options, on_delta, &blocking_cancel));
	let outcome = tokio::select! {
		biased;
		result = &mut task => return result.map_err(join_error)?,
		() = cancel.cancelled() => AppleFmError::cancelled(),
		() = tokio::time::sleep(RUN_TIMEOUT) => AppleFmError::timed_out(),
	};
	work_cancel.cancel();
	let _ = task.await;
	Err(outcome)
}

fn validate_options(options: &AppleFmOptions) -> Result<()> {
	if options.prompt.trim().is_empty() {
		return Err(AppleFmError::new(
			AppleFmErrorCode::InvalidInput,
			"Apple Foundation Models requires a non-empty prompt",
		));
	}
	if options
		.temperature
		.is_some_and(|value| !value.is_finite() || value < 0.0)
	{
		return Err(AppleFmError::new(
			AppleFmErrorCode::InvalidInput,
			"temperature must be finite and non-negative",
		));
	}
	if options.max_tokens == Some(0) {
		return Err(AppleFmError::new(
			AppleFmErrorCode::InvalidInput,
			"maximum response tokens must be non-zero",
		));
	}
	Ok(())
}

fn availability_error_code(reason: &str) -> AppleFmErrorCode {
	match reason {
		"device_not_eligible" => AppleFmErrorCode::DeviceNotEligible,
		"apple_intelligence_not_enabled" => AppleFmErrorCode::AppleIntelligenceNotEnabled,
		"model_not_ready" => AppleFmErrorCode::ModelNotReady,
		"model_unavailable" | "macOS Apple Silicon only" => AppleFmErrorCode::ModelUnavailable,
		_ => AppleFmErrorCode::Runtime,
	}
}

fn join_error(error: JoinError) -> AppleFmError {
	AppleFmError::runtime(format!("Apple Foundation Models worker failed: {error}"))
}

#[cfg(test)]
mod tests {
	use tokio_util::sync::CancellationToken;

	use super::{AppleFm, AppleFmErrorCode, AppleFmOptions, validate_options};

	#[test]
	fn request_validation_rejects_empty_or_invalid_limits() {
		let empty = AppleFmOptions::new("  ");
		assert!(validate_options(&empty).is_err());

		let invalid_temperature = AppleFmOptions::new("hello").temperature(f64::NAN);
		assert!(validate_options(&invalid_temperature).is_err());

		let no_tokens = AppleFmOptions::new("hello").max_tokens(0);
		assert!(validate_options(&no_tokens).is_err());
	}

	#[test]
	fn error_codes_have_stable_wire_names() {
		assert_eq!(AppleFmErrorCode::GuardrailBlocked.to_string(), "guardrail_blocked");
		assert_eq!(AppleFmErrorCode::Runtime.to_string(), "runtime_error");
	}

	#[tokio::test]
	async fn generation_honors_preexisting_cancellation() {
		let cancel = CancellationToken::new();
		cancel.cancel();
		let error = AppleFm
			.generate(AppleFmOptions::new("hello"), cancel)
			.await
			.unwrap_err();
		assert_eq!(error.code(), AppleFmErrorCode::Cancelled);
	}
}
