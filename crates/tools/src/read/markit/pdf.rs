//! PDF-to-Markdown conversion through `pdf-inspector`.

use omp_core::Str;
use pdf_inspector::{MarkdownOptions, PdfOptions, PdfType, process_pdf_mem_with_options};

use super::{Conversion, MarkitError};

const TEXT_LAYER_NOTE: &str = "This PDF is scanned or image-based and has no usable text layer. \
                               OCR is required to extract its text.";

/// Convert PDF bytes, preserving page order, metadata, and any text-layer
/// qualification reported by `pdf-inspector`.
pub(super) fn convert(bytes: &[u8]) -> Result<Conversion, MarkitError> {
	let mut markdown = MarkdownOptions::default();
	// pi's PDF projection identifies every extracted page. `pdf-inspector`
	// emits these as `<!-- Page N -->` before that page's content.
	markdown.include_page_numbers = true;
	let options = PdfOptions::new().markdown(markdown);
	let result = process_pdf_mem_with_options(bytes, options)
		.map_err(|error| MarkitError::conversion("pdf", error.to_string()))?;

	let scanned = matches!(result.pdf_type, PdfType::Scanned | PdfType::ImageBased);
	let note = scanned.then(|| Str::from(TEXT_LAYER_NOTE));
	let text = match result.markdown {
		Some(text) if !text.is_empty() => Str::from(text),
		// `pdf-inspector` intentionally has no Markdown for an image-only
		// document. Preserve that typed classification as a successful empty
		// conversion so read can explain that OCR is required.
		None if scanned => Str::default(),
		_ => return Err(MarkitError::conversion("pdf", "Conversion produced no output")),
	};

	Ok(Conversion { text, note, title: result.title.map(Str::from) })
}
