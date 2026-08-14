//! OpenDocument Presentation conversion.

use omp_core::Str;

use super::{MarkitError, odf};

const MIMETYPE: &str = "application/vnd.oasis.opendocument.presentation";

/// Converts a bounded ODP package to deterministic slide-ordered Markdown.
pub(super) fn convert(bytes: &[u8]) -> Result<Str, MarkitError> {
	odf::convert(bytes, anydoc::Format::Odp, "odp", MIMETYPE)
}
