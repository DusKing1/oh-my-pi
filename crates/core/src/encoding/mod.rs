//! Generic base-N encoding/decoding with compile-time dictionaries.
//!
//! Provides a unified interface for binary-to-text encodings (hex, base64,
//! base32) with optimized runtime implementations and const-friendly fallbacks.
//!
//! # Features
//!
//! - **Const evaluation**: All encoding/decoding logic works in const contexts
//! - **Zero-copy iterators**: Streaming encoders and decoders with exact size
//!   hints
//! - **Optimized paths**: Hand-tuned implementations for Base64/Base32
//! - **Flexible dictionaries**: Support for custom alphabets and padding
//!   schemes
//! - **Stack allocation**: Fixed-size [`ArrayStr`] and [`Array`] wrappers
//!
//! # Submodules
//!
//! - [`hex`]: Hexadecimal encoding with upper/lowercase support
//! - [`base64`]: Standard and URL-safe Base64 encodings
//! - [`base64_url`]: URL-safe Base64 encodings
//! - [`base32`]: RFC 4648 Base32, Base32-Hex, and Base32-DNS variants
//! - [`base32_hex`]: RFC 4648 Base32-Hex
//! - [`base32_dns`]: RFC 4648 Base32-DNS
//!
//! # Examples
//! ```
//! use omp_core::{base64, hex};
//!
//! // Hex encoding
//! let encoded = hex::encode(b"Hello").into_string();
//! assert_eq!(encoded, "48656c6c6f");
//!
//! // Base64 encoding
//! let encoded = base64::encode(b"Hello").into_string();
//! assert_eq!(encoded, "SGVsbG8=");
//!
//! // Const evaluation
//! const HEX: hex::ArrayStr<5> = hex::encode_n(b"Hello");
//! assert_eq!(&*HEX, "48656c6c6f");
//! ```

mod fixed_arr;
pub mod hex;
mod opt;
pub use fixed_arr::{Array, ArrayStr};
pub(crate) use fixed_arr::{ascii_to_str, ascii_to_str_owned, format_with_precision};

mod error;
pub use error::{DecodeError, Result};

mod base_n;
pub use base_n::{
	DecodeWriter, Decoder, EncodeWriter, Encoder, Encoding, base32, base32_dns, base32_hex, base64,
	base64_url,
};
