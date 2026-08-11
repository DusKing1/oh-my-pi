//! `DeviceCheck` attestation envelope encoding for `ChatGPT` Codex.
//!
//! The native attestor mints a fresh single-use token for each eligible
//! `ChatGPT` OAuth request. The deterministic encoder never accepts bearer
//! credentials, and all token/envelope buffers zeroize on drop.

use std::fmt;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::sync::LazyLock;

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use omp_core::Str;
use zeroize::Zeroizing;

use crate::CodexAttestation;

const CHATGPT_BUNDLE_ID: &str = "com.openai.codex";

/// Opaque platform-issued `DeviceCheck` token.
#[derive(Clone, Eq, PartialEq)]
pub struct CodexDeviceToken(Zeroizing<Vec<u8>>);

impl CodexDeviceToken {
	/// Wraps a non-empty base64 token returned by the native `DeviceCheck` API.
	#[must_use]
	pub fn new(token_base64: impl AsRef<[u8]>) -> Option<Self> {
		let token = token_base64.as_ref();
		(!token.is_empty()).then(|| Self(Zeroizing::new(token.to_vec())))
	}

	fn as_bytes(&self) -> &[u8] {
		&self.0
	}
}

impl fmt::Debug for CodexDeviceToken {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str("CodexDeviceToken([redacted])")
	}
}

/// Native `DeviceCheck` result projected into the Codex attestation payload.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CodexDeviceCheckResult {
	/// Whether `DeviceCheck` is supported on the current platform.
	pub supported:  bool,
	/// Platform token, when `DeviceCheck` minted one successfully.
	pub token:      Option<CodexDeviceToken>,
	/// Native token-generation latency in milliseconds.
	pub latency_ms: Option<f64>,
}

/// Non-secret process and locale signals included in Codex attestation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodexAttestationSignals<'a> {
	/// Resolved locale, truncated to 64 Unicode scalar values.
	pub locale:     &'a str,
	/// Resolved IANA timezone, truncated to 64 Unicode scalar values.
	pub timezone:   &'a str,
	/// Process-stable random application session id, truncated to 128 scalars.
	pub session_id: &'a str,
}

/// Attestation encoding failure.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CodexAttestationError {
	/// A CBOR collection or string exceeded the supported 32-bit length.
	#[error("Codex attestation field is too large")]
	FieldTooLarge,
	/// The native `DeviceCheck` token was not UTF-8 text.
	#[error("Codex DeviceCheck token is not UTF-8")]
	InvalidToken,
}

/// Just-in-time platform attestor for `ChatGPT` Codex requests.
///
/// The enum is concrete so the request hot path never allocates a boxed
/// callback future. Unsupported platforms select [`Self::Unavailable`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodexAttestor {
	/// Apple `DeviceCheck` on macOS arm64.
	DeviceCheck,
	/// No platform attestation implementation is available.
	Unavailable,
}

impl Default for CodexAttestor {
	fn default() -> Self {
		if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
			Self::DeviceCheck
		} else {
			Self::Unavailable
		}
	}
}

impl CodexAttestor {
	/// Mints one fresh attestation envelope.
	///
	/// Unavailable platform integration produces `None`, matching Pi's
	/// omission rule. A supported `DeviceCheck` call that returns no token is a
	/// valid attestation with error code 4. OAuth material is never observed.
	pub async fn generate(self) -> Option<CodexAttestation> {
		match self {
			Self::Unavailable => None,
			Self::DeviceCheck => platform::generate().await,
		}
	}
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn process_session_id() -> &'static str {
	static SESSION_ID: LazyLock<Str> = LazyLock::new(random_uuid);
	SESSION_ID.as_str()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn random_uuid() -> Str {
	let mut bytes: [u8; 16] = rand::random();
	bytes[6] = (bytes[6] & 0x0f) | 0x40;
	bytes[8] = (bytes[8] & 0x3f) | 0x80;
	format!(
		"{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:\
		 02x}{:02x}",
		bytes[0],
		bytes[1],
		bytes[2],
		bytes[3],
		bytes[4],
		bytes[5],
		bytes[6],
		bytes[7],
		bytes[8],
		bytes[9],
		bytes[10],
		bytes[11],
		bytes[12],
		bytes[13],
		bytes[14],
		bytes[15],
	)
	.into()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod platform {
	use std::{
		sync::{Arc, Mutex},
		time::{Duration, Instant},
	};

	use base64::{Engine as _, engine::general_purpose::STANDARD};
	use block2::RcBlock;
	use objc2_device_check::DCDevice;
	use objc2_foundation::{NSData, NSError, NSLocale, NSTimeZone};
	use tokio::sync::oneshot;
	use zeroize::Zeroizing;

	use super::{
		CodexAttestation, CodexAttestationSignals, CodexDeviceCheckResult, CodexDeviceToken,
		build_codex_attestation, process_session_id,
	};

	pub(super) async fn generate() -> Option<CodexAttestation> {
		let started = Instant::now();
		let receiver = {
			// SAFETY: `currentDevice` and `isSupported` are framework singleton
			// accessors with no caller-side lifetime requirements.
			let device = unsafe { DCDevice::currentDevice() };
			if !unsafe { device.isSupported() } {
				return build(false, None, started);
			}

			let (sender, receiver) = oneshot::channel();
			let sender = Arc::new(Mutex::new(Some(sender)));
			let callback_sender = Arc::clone(&sender);
			let callback = RcBlock::new(move |token: *mut NSData, _error: *mut NSError| {
				// SAFETY: DeviceCheck promises both callback pointers remain valid
				// for the duration of this invocation. `to_vec` copies immediately.
				let token = unsafe { token.as_ref() }.map(NSData::to_vec);
				if let Some(sender) = callback_sender.lock().ok().and_then(|mut slot| slot.take()) {
					let _ = sender.send(token);
				}
			});
			// SAFETY: the heap block is retained by DeviceCheck for asynchronous
			// invocation and its typed signature matches the framework declaration.
			unsafe { device.generateTokenWithCompletionHandler(&callback) };
			receiver
		};
		let token = match tokio::time::timeout(Duration::from_secs(1), receiver).await {
			Ok(Ok(token)) => token,
			Ok(Err(_)) | Err(_) => None,
		};
		build(true, token, started)
	}

	fn build(supported: bool, token: Option<Vec<u8>>, started: Instant) -> Option<CodexAttestation> {
		let token = token
			.map(Zeroizing::new)
			.map(|bytes| Zeroizing::new(STANDARD.encode(bytes.as_slice())))
			.and_then(|value| CodexDeviceToken::new(value.as_bytes()));
		let locale = NSLocale::currentLocale()
			.localeIdentifier()
			.to_string()
			.split('@')
			.next()
			.unwrap_or("en_US")
			.replace('_', "-");
		let timezone = NSTimeZone::localTimeZone().name().to_string();
		build_codex_attestation(
			&CodexDeviceCheckResult {
				supported,
				token,
				latency_ms: Some(started.elapsed().as_secs_f64() * 1_000.0),
			},
			&CodexAttestationSignals {
				locale:     &locale,
				timezone:   &timezone,
				session_id: process_session_id(),
			},
		)
		.ok()
	}
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod platform {
	use super::CodexAttestation;

	pub(super) async fn generate() -> Option<CodexAttestation> {
		None
	}
}

/// Builds the complete `x-oai-attestation` JSON envelope.
///
/// Unsupported `DeviceCheck` is encoded with error code 3; supported `DeviceCheck`
/// without a token uses error code 4. A token is never included in debug
/// output.
pub fn build_codex_attestation(
	result: &CodexDeviceCheckResult,
	signals: &CodexAttestationSignals<'_>,
) -> Result<CodexAttestation, CodexAttestationError> {
	let locale = truncate_scalars(signals.locale, 64);
	let timezone = truncate_scalars(signals.timezone, 64);
	let session_id = truncate_scalars(signals.session_id, 128);

	let mut signal_entries = Vec::with_capacity(7);
	signal_entries.push((cbor_unsigned(0), cbor_unsigned(1)));
	signal_entries.push((cbor_unsigned(1), cbor_array(vec![cbor_text(locale.as_bytes())?])?));
	signal_entries.push((cbor_unsigned(2), cbor_text(locale.as_bytes())?));
	signal_entries.push((cbor_unsigned(3), cbor_text(timezone.as_bytes())?));
	signal_entries.push((cbor_unsigned(4), cbor_unsigned(0)));
	signal_entries.push((cbor_unsigned(5), cbor_unsigned(1)));
	signal_entries.push((cbor_unsigned(6), cbor_text(session_id.as_bytes())?));
	let signal_bytes = cbor_map(signal_entries)?;

	let mut entries = Vec::with_capacity(4);
	if result.supported {
		if let Some(token) = &result.token {
			let token = std::str::from_utf8(token.as_bytes())
				.map_err(|_| CodexAttestationError::InvalidToken)?;
			entries.push((cbor_text(b"token")?, cbor_text(token.as_bytes())?));
		} else {
			entries.push((cbor_text(b"error_code")?, cbor_unsigned(4)));
		}
	} else {
		entries.push((cbor_text(b"error_code")?, cbor_unsigned(3)));
	}
	entries.push((cbor_text(b"bundle_id")?, cbor_text(CHATGPT_BUNDLE_ID.as_bytes())?));
	entries.push((cbor_text(b"f")?, cbor_bytes(&signal_bytes)?));
	if let Some(latency_ms) = result.latency_ms {
		let mut value = Vec::with_capacity(9);
		value.push(0xfb);
		value.extend_from_slice(&latency_ms.to_be_bytes());
		entries.push((cbor_text(b"t")?, value));
	}
	let encoded = Zeroizing::new(cbor_map(entries)?);
	let client_attestation =
		Zeroizing::new(format!("v1.{}", URL_SAFE_NO_PAD.encode(encoded.as_slice())));
	let envelope = Zeroizing::new(format!(
		r#"{{"v":1,"s":0,"t":{}}}"#,
		serde_json::to_string(client_attestation.as_str())
			.expect("attestation token is valid JSON text")
	));
	CodexAttestation::new(envelope.as_bytes()).ok_or(CodexAttestationError::FieldTooLarge)
}

fn truncate_scalars(value: &str, maximum: usize) -> Str {
	if value.chars().count() <= maximum {
		return Str::new(value);
	}
	Str::from(value.chars().take(maximum).collect::<String>())
}

fn cbor_unsigned(value: u32) -> Vec<u8> {
	cbor_header(0x00, value).expect("u32 is always a valid CBOR length")
}

fn cbor_text(value: &[u8]) -> Result<Vec<u8>, CodexAttestationError> {
	let mut out = cbor_header(0x60, length(value.len())?)?;
	out.extend_from_slice(value);
	Ok(out)
}

fn cbor_bytes(value: &[u8]) -> Result<Vec<u8>, CodexAttestationError> {
	let mut out = cbor_header(0x40, length(value.len())?)?;
	out.extend_from_slice(value);
	Ok(out)
}

fn cbor_array(values: Vec<Vec<u8>>) -> Result<Vec<u8>, CodexAttestationError> {
	let mut out = cbor_header(0x80, length(values.len())?)?;
	for value in values {
		out.extend(value);
	}
	Ok(out)
}

fn cbor_map(entries: Vec<(Vec<u8>, Vec<u8>)>) -> Result<Vec<u8>, CodexAttestationError> {
	let mut out = cbor_header(0xa0, length(entries.len())?)?;
	for (key, value) in entries {
		out.extend(key);
		out.extend(value);
	}
	Ok(out)
}

fn length(value: usize) -> Result<u32, CodexAttestationError> {
	u32::try_from(value).map_err(|_| CodexAttestationError::FieldTooLarge)
}

fn cbor_header(base: u8, value: u32) -> Result<Vec<u8>, CodexAttestationError> {
	let out = if value < 24 {
		vec![base + value as u8]
	} else if let Ok(value) = u8::try_from(value) {
		vec![base + 24, value]
	} else if let Ok(value) = u16::try_from(value) {
		let mut out = vec![base + 25];
		out.extend_from_slice(&value.to_be_bytes());
		out
	} else {
		let mut out = vec![base + 26];
		out.extend_from_slice(&value.to_be_bytes());
		out
	};
	Ok(out)
}
