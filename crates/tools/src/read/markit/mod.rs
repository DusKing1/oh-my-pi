//! In-memory document-to-Markdown conversion.

use std::{fmt, path::Path};

use omp_core::Str;

mod docx;
mod epub;
mod ooxml;
mod pdf;
mod pptx;
mod xlsx;

/// Markdown produced from a supported document.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Conversion {
	/// Converted document text.
	pub text:  Str,
	/// Optional model-facing qualification of the converted text.
	pub note:  Option<Str>,
	/// Optional title supplied by document metadata.
	///
	/// Metadata stays separate from `text`, preserving the converter's source
	/// order and model-facing Markdown.
	pub title: Option<Str>,
}

/// A typed document conversion failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MarkitError {
	/// A converter accepted the document but could not produce Markdown.
	Conversion {
		/// Stable converter name.
		format:  &'static str,
		/// Converter-specific failure detail.
		message: Str,
	},
}

impl MarkitError {
	/// Build a failure reported by a specific document converter.
	pub fn conversion(format: &'static str, message: impl Into<Str>) -> Self {
		Self::Conversion { format, message: message.into() }
	}

	/// Stable name of the converter that failed.
	pub const fn format(&self) -> &'static str {
		match self {
			Self::Conversion { format, .. } => format,
		}
	}

	/// Converter-specific failure detail.
	pub fn message(&self) -> &str {
		match self {
			Self::Conversion { message, .. } => message.as_ref(),
		}
	}
}

impl fmt::Display for MarkitError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(formatter, "{} conversion failed: {}", self.format(), self.message())
	}
}

impl std::error::Error for MarkitError {}

/// Convert one of the approved document formats to Markdown.
///
/// Unsupported extensions return `Ok(None)`. Once an extension is recognized,
/// converter failures remain typed so the caller can truthfully render the
/// original binary size rather than treating the bytes as text.
pub fn convert(path: &Path, bytes: &[u8]) -> Result<Option<Conversion>, MarkitError> {
	let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
		return Ok(None);
	};
	let extension = extension.to_ascii_lowercase();

	let conversion = match extension.as_str() {
		"pdf" => pdf::convert(bytes)?,
		"docx" => Conversion { text: docx::convert(bytes)?, note: None, title: None },
		"xlsx" => Conversion { text: xlsx::convert(bytes)?, note: None, title: None },
		"pptx" => Conversion { text: pptx::convert(bytes)?, note: None, title: None },
		"epub" => {
			let (text, title) = epub::convert(bytes)?;
			Conversion { text, note: None, title }
		},
		_ => return Ok(None),
	};

	Ok(Some(conversion))
}
