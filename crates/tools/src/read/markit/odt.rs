//! OpenDocument Text to Markdown conversion.

use omp_core::Str;

use super::{MarkitError, odf};

const FORMAT: &str = "odt";
const MIMETYPE: &str = "application/vnd.oasis.opendocument.text";

/// Converts an ODT package to deterministic Markdown without executing or
/// interpreting any embedded object or macro payload.
pub(super) fn convert(bytes: &[u8]) -> Result<Str, MarkitError> {
	odf::convert(bytes, anydoc::Format::Odt, FORMAT, MIMETYPE)
}
