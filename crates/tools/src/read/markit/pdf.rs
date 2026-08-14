//! PDF-to-Markdown conversion through `pdf-inspector`.

use omp_core::Str;
use pdf_inspector::{MarkdownOptions, PdfOptions, PdfType, process_pdf_mem_with_options};

use super::{Conversion, MarkitError};

const TEXT_LAYER_NOTE: &str = "This PDF is scanned or image-based and has no usable text layer. \
                               OCR is required to extract its text.";
const ENCODING_NOTE: &str =
	"Broken PDF font encodings were detected; extracted text may be garbled.";

/// Convert PDF bytes, preserving page order, metadata, and any text-layer
/// qualification reported by `pdf-inspector`.
pub(super) fn convert(bytes: &[u8]) -> Result<Conversion, MarkitError> {
	let markdown = MarkdownOptions { include_page_numbers: true, ..MarkdownOptions::default() };
	let options = PdfOptions::new().markdown(markdown);
	let result = process_pdf_mem_with_options(bytes, options)
		.map_err(|error| MarkitError::conversion("pdf", error.to_string()))?;

	let scanned = matches!(result.pdf_type, PdfType::Scanned | PdfType::ImageBased);
	let note = if scanned {
		Some(Str::new_static(TEXT_LAYER_NOTE))
	} else {
		extraction_note(result.pages_needing_ocr.len(), result.page_count, result.has_encoding_issues)
	};
	let text = match result.markdown {
		Some(mut text) if has_markdown_content(&text) => {
			if !text.ends_with('\n') {
				text.push('\n');
			}
			Str::from(text)
		},
		// `pdf-inspector` intentionally has no Markdown for an image-only
		// document. Preserve that typed classification as a successful empty
		// conversion so read can explain that OCR is required.
		_ if scanned => Str::default(),
		_ => {
			return Err(MarkitError::conversion(
				"pdf",
				format!(
					"PDF has no extractable text ({:?}, {} pages): OCR is required",
					result.pdf_type, result.page_count
				),
			));
		},
	};

	Ok(Conversion { text, note, title: result.title.map(Str::from) })
}

fn extraction_note(
	pages_needing_ocr: usize,
	page_count: u32,
	has_encoding_issues: bool,
) -> Option<Str> {
	match (pages_needing_ocr, has_encoding_issues) {
		(0, false) => None,
		(0, true) => Some(Str::new_static(ENCODING_NOTE)),
		(count, false) => Some(Str::from(format!(
			"{count} of {page_count} PDF pages may need OCR; extracted text may be incomplete."
		))),
		(count, true) => Some(Str::from(format!(
			"{count} of {page_count} PDF pages may need OCR, and broken font encodings were \
			 detected; extracted text may be incomplete or garbled."
		))),
	}
}

/// `include_page_numbers` can make an otherwise empty extraction look
/// non-empty. Ignore those generated markers when deciding whether the PDF
/// actually yielded content.
fn has_markdown_content(markdown: &str) -> bool {
	markdown.lines().any(|line| {
		let line = line.trim();
		!line.is_empty() && !is_page_marker(line)
	})
}

fn is_page_marker(line: &str) -> bool {
	line
		.strip_prefix("<!-- Page ")
		.and_then(|line| line.strip_suffix(" -->"))
		.is_some_and(|page| !page.is_empty() && page.bytes().all(|byte| byte.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
	use super::{ENCODING_NOTE, extraction_note, has_markdown_content};

	#[test]
	fn generated_page_markers_are_not_extractable_content() {
		assert!(!has_markdown_content("\n<!-- Page 1 -->\n\n<!-- Page 27 -->\n"));
		assert!(has_markdown_content("<!-- Page 1 -->\n\nActual text\n"));
	}

	#[test]
	fn encoding_qualification_does_not_require_an_ocr_page_list() {
		assert_eq!(extraction_note(0, 1, true).as_deref(), Some(ENCODING_NOTE));
	}
}
