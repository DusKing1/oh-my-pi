//! RTF to Markdown conversion.
//!
//! `anydoc` supplies the in-process, byte-oriented RTF lexer and state machine:
//! group-scoped formatting, destination suppression, code-page and `\u`
//! fallback decoding, fields, lists, and tables all pass through its shared
//! document model and deterministic Markdown renderer. Binary and object
//! payloads are parsed as inert data and never executed.

use omp_core::Str;

use super::{MarkitError, convert_with_anydoc};

const FORMAT: &str = "rtf";

/// Converts Rich Text Format bytes to deterministic Markdown.
pub(super) fn convert(bytes: &[u8]) -> Result<Str, MarkitError> {
	convert_with_anydoc(bytes, anydoc::Format::Rtf, FORMAT)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn markdown(source: &[u8]) -> String {
		convert(source)
			.expect("RTF conversion succeeds")
			.to_string()
	}

	#[test]
	fn rtf_extension_dispatches_case_insensitively() {
		let conversion =
			super::super::convert(std::path::Path::new("note.RTF"), br"{\rtf1\ansi Dispatched\par}")
				.expect("dispatch succeeds")
				.expect("RTF is supported");
		assert_eq!(conversion.text.as_str(), "Dispatched\n");
		assert_eq!(conversion.note, None);
		assert_eq!(conversion.title, None);
	}

	#[test]
	fn nested_formatting_and_fields_render_as_markdown() {
		let rendered = markdown(
			br#"{\rtf1\ansi {\b outer {\i inner} tail}\par {\strike struck}\par {\field{\*\fldinst HYPERLINK "https://example.com/a"}{\fldrslt Example}}\par}"#,
		);

		assert!(rendered.contains("**outer** ***inner*** **tail**"), "{rendered}");
		assert!(rendered.contains("~~struck~~"), "{rendered}");
		assert!(rendered.contains("[Example](https://example.com/a)"), "{rendered}");
	}

	#[test]
	fn ansi_codepage_unicode_fallback_and_surrogates_decode_once() {
		let rendered = markdown(
			br"{\rtf1\ansi\ansicpg1251 \'cf\'f0\'e8\'e2\'e5\'f2 \uc1\u20013? \u55357?\u56842?\par}",
		);

		assert_eq!(rendered, "Привет 中 😊\n");

		let font_override = markdown(
			br"{\rtf1\ansi\ansicpg1252{\fonttbl{\f0\fcharset0 Latin;}{\f1\fcharset204 Cyrillic;}}\f1\'cf\'f0\'e8\'e2\'e5\'f2\par}",
		);
		assert_eq!(font_override, "Привет\n");
	}

	#[test]
	fn escaped_controls_and_ignorable_destinations_do_not_leak() {
		let rendered = markdown(
			br"{\rtf1\ansi Slash \\ braces \{x\} hard\_hyphen soft\-hyphen nbsp\~space {\*\mystery hidden \b secret} {\object\objdata 414243{\result Preview}}\par}",
		);

		assert!(rendered.contains("Slash \\\\ braces {x}"), "{rendered}");
		assert!(rendered.contains("hard-hyphen softhyphen"), "{rendered}");
		assert!(rendered.contains("nbsp space"), "{rendered}");
		assert!(rendered.contains("Preview"), "{rendered}");
		assert!(!rendered.contains("mystery"), "{rendered}");
		assert!(!rendered.contains("hidden"), "{rendered}");
		assert!(!rendered.contains("objdata"), "{rendered}");
		assert!(!rendered.contains("414243"), "{rendered}");
	}

	#[test]
	fn binary_payload_cannot_change_group_structure_or_allocate_from_its_count() {
		let rendered = markdown(br"{\rtf1 before\bin5 }}{\\after\par}");
		assert_eq!(rendered, "beforeafter\n");

		let truncated = markdown(br"{\rtf1 kept\bin2147483647 tiny}");
		assert_eq!(truncated, "kept\n");
	}

	#[test]
	fn list_and_table_structure_survive_rendering() {
		let list = markdown(
			br"{\rtf1\ansi\pard{\listtext \bullet\tab}One\par\pard{\listtext \bullet\tab}Two\par}",
		);
		assert!(list.contains("One"), "{list}");
		assert!(list.contains("Two"), "{list}");
		assert!(list.lines().filter(|line| line.starts_with("- ")).count() >= 2, "{list}");

		let table = markdown(br"{\rtf1\ansi\trowd\cellx1000\cellx2000\intbl A\cell B\cell\row}");
		assert!(table.contains("| A | B |"), "{table}");
	}

	#[test]
	fn malformed_groups_recover_without_exposing_control_words() {
		let rendered = markdown(br"{\rtf1 alpha} stray } {\b beta");
		assert!(rendered.contains("alpha"), "{rendered}");
		assert!(rendered.contains("beta"), "{rendered}");
		assert!(!rendered.contains("rtf1"), "{rendered}");
		assert!(!rendered.contains("\\b"), "{rendered}");
	}

	#[test]
	fn non_rtf_bytes_keep_a_typed_failure() {
		let error = convert(b"plain text").expect_err("non-RTF input is malformed");
		assert_eq!(error.format(), FORMAT);
	}
}
