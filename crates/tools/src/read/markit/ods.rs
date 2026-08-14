//! OpenDocument Spreadsheet conversion.

use omp_core::Str;

use super::{MarkitError, odf};

const MIME_TYPE: &str = "application/vnd.oasis.opendocument.spreadsheet";

/// Converts a bounded ODS package to deterministic Markdown tables.
pub(super) fn convert(bytes: &[u8]) -> Result<Str, MarkitError> {
	odf::convert(bytes, anydoc::Format::Ods, "ods", MIME_TYPE)
}
