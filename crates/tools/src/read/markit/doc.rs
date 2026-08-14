//! Word 97-2003 binary document conversion through `anydoc`.

use omp_core::Str;

use super::{MarkitError, convert_with_anydoc};

/// Convert a legacy binary Word document to deterministic Markdown.
///
/// `anydoc` parses the OLE container and Word binary streams in-process. It
/// treats encryption as an error and only reads supported content; embedded
/// objects and macro payloads are never executed.
pub(super) fn convert(bytes: &[u8]) -> Result<Str, MarkitError> {
	convert_with_anydoc(bytes, anydoc::Format::Doc, "doc")
}
