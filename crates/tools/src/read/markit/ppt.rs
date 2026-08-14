//! PowerPoint 97-2003 binary presentation conversion through `anydoc`.

use omp_core::Str;

use super::{MarkitError, convert_with_anydoc};

/// Converts a legacy binary PowerPoint presentation to deterministic Markdown.
///
/// `anydoc` validates the OLE container, resolves the persist directory, and
/// traverses bounded PowerPoint records in-process. Embedded code is never
/// executed.
pub(super) fn convert(bytes: &[u8]) -> Result<Str, MarkitError> {
	convert_with_anydoc(bytes, anydoc::Format::Ppt, "ppt")
}
