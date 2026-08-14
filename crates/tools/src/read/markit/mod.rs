//! In-memory document-to-Markdown conversion.

use std::{fmt, path::Path};

use omp_core::Str;
use strum::EnumString;

mod doc;
mod docx;
mod epub;
mod odf;
mod odp;
mod ods;
mod odt;
mod ooxml;
mod pdf;
mod ppt;
mod pptx;
mod rtf;
mod xls;
mod xlsx;

#[derive(Clone, Copy, EnumString)]
#[strum(ascii_case_insensitive, serialize_all = "lowercase")]
enum Format {
	Pdf,
	Doc,
	#[strum(serialize = "docx", serialize = "docm")]
	Docx,
	Xls,
	#[strum(serialize = "xlsx", serialize = "xlsm")]
	Xlsx,
	Odt,
	Ods,
	Odp,
	Ppt,
	Pptx,
	Rtf,
	Epub,
}

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

impl Conversion {
	fn plain(text: Str) -> Self {
		Self { text, note: None, title: None }
	}
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

fn convert_with_anydoc(
	bytes: &[u8],
	format: anydoc::Format,
	format_name: &'static str,
) -> Result<Str, MarkitError> {
	anydoc::to_markdown_bytes(bytes, format)
		.map(Str::from)
		.map_err(|error| MarkitError::conversion(format_name, error.to_string()))
}

fn format_from_extension(extension: &str) -> Option<Format> {
	extension.trim_start_matches('.').parse().ok()
}

/// Whether a path names a supported in-memory document format.
pub(crate) fn supports_path(path: &Path) -> bool {
	path
		.extension()
		.and_then(|extension| extension.to_str())
		.and_then(format_from_extension)
		.is_some()
}

/// Whether an extension names a supported in-memory document format.
///
/// Both `docx` and `.docx` forms are accepted.
pub(crate) fn supports_extension(extension: &str) -> bool {
	format_from_extension(extension).is_some()
}

/// Convert one of the approved document formats to Markdown.
///
/// Unsupported extensions return `Ok(None)`. Once an extension is recognized,
/// converter failures remain typed so the caller can truthfully render the
/// original binary size rather than treating the bytes as text.
pub fn convert(path: &Path, bytes: &[u8]) -> Result<Option<Conversion>, MarkitError> {
	let Some(format) = path
		.extension()
		.and_then(|extension| extension.to_str())
		.and_then(format_from_extension)
	else {
		return Ok(None);
	};

	let conversion = match format {
		Format::Pdf => pdf::convert(bytes)?,
		Format::Doc => Conversion::plain(doc::convert(bytes)?),
		Format::Docx => Conversion::plain(docx::convert(bytes)?),
		Format::Xls => Conversion::plain(xls::convert(bytes)?),
		Format::Xlsx => Conversion::plain(xlsx::convert(bytes)?),
		Format::Odt => Conversion::plain(odt::convert(bytes)?),
		Format::Ods => Conversion::plain(ods::convert(bytes)?),
		Format::Odp => Conversion::plain(odp::convert(bytes)?),
		Format::Ppt => Conversion::plain(ppt::convert(bytes)?),
		Format::Pptx => Conversion::plain(pptx::convert(bytes)?),
		Format::Rtf => Conversion::plain(rtf::convert(bytes)?),
		Format::Epub => {
			let (text, title) = epub::convert(bytes)?;
			Conversion { text, note: None, title }
		},
	};

	Ok(Some(conversion))
}
