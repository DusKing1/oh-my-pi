/// Errors that can occur during base-N encoding/decoding operations.
#[derive(Debug, thiserror::Error, PartialEq, Eq, PartialOrd, Ord)]
pub enum DecodeError {
	/// An invalid character was encountered during decoding.
	#[error("invalid character: 0x{0:02x}")]
	InvalidCharacter(u8),
	/// The input length is invalid for the expected padded format.
	#[error("invalid length for padded base-n input")]
	InvalidLength,
	/// The buffer is too small.
	#[error("buffer too small: {0} bytes")]
	BufferTooSmall(usize),
	/// The input is too short for the expected length.
	#[error("input length too short for expected length")]
	InputTooShort,
}

/// Result type for base-N operations.
pub type Result<T, E = DecodeError> = std::result::Result<T, E>;
