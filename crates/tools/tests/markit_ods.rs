//! Behavioral OpenDocument Spreadsheet conversion fixtures.

use std::path::Path;

use omp_ar::zip::Writer;
use omp_tools::read::markit;

const ODS_MIMETYPE: &str = "application/vnd.oasis.opendocument.spreadsheet";
const NS: &str = r#"xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
	xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
	xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
	xmlns:xlink="http://www.w3.org/1999/xlink""#;

fn zip(entries: &[(&str, &str)]) -> Vec<u8> {
	let mut writer = Writer::new(Vec::new());
	for (path, content) in entries {
		writer
			.add_file(path, content.as_bytes())
			.expect("fixture member adds");
	}
	writer.finish().expect("fixture archive finishes")
}

fn ods(content: &str) -> Vec<u8> {
	zip(&[("mimetype", ODS_MIMETYPE), ("content.xml", content)])
}

#[test]
fn ods_renders_declared_sheets_typed_display_values_and_structure() {
	// Extraction keeps non-empty hidden sheets: visibility is a spreadsheet UI
	// state, while the declared sheet order remains the source order.
	let content = format!(
		r##"<?xml version="1.0"?>
<office:document-content {NS}>
<office:body><office:spreadsheet>
<table:table table:name="Visible &amp; Data">
<table:table-header-rows><table:table-row>
<table:table-cell><text:p>Shown</text:p></table:table-cell>
<table:table-cell><text:p>Formula</text:p></table:table-cell>
<table:table-cell><text:p>Date</text:p></table:table-cell>
<table:table-cell><text:p>Time</text:p></table:table-cell>
<table:table-cell><text:p>Bool</text:p></table:table-cell>
<table:table-cell><text:p>Error</text:p></table:table-cell>
<table:table-cell><text:p>Link</text:p></table:table-cell>
<table:table-cell><text:p>Escaped</text:p></table:table-cell>
</table:table-row></table:table-header-rows>
<table:table-row>
<table:table-cell office:value-type="currency" office:value="1.25" office:currency="EUR"><text:p>Displayed 1,25 €</text:p></table:table-cell>
<table:table-cell table:formula="of:=SUM([.C2:.D2])" office:value-type="float" office:value="3"><text:p>3.00 shown</text:p></table:table-cell>
<table:table-cell office:value-type="date" office:date-value="2026-08-14"/>
<table:table-cell office:value-type="time" office:time-value="PT1H30M"/>
<table:table-cell office:value-type="boolean" office:boolean-value="false"/>
<table:table-cell table:formula="of:=1/0" office:value-type="string" office:string-value="#DIV/0!"><text:p>#DIV/0!</text:p></table:table-cell>
<table:table-cell office:value-type="string"><text:p><text:a xlink:href="https://example.com/a|b">site | docs</text:a></text:p></table:table-cell>
<table:table-cell office:value-type="string" office:string-value="raw | string"/>
</table:table-row>
<table:table-row>
<table:table-cell office:value-type="percentage" office:value="0.125"/>
<table:table-cell office:value-type="currency" office:value="1.25" office:currency="EUR"/>
<table:table-cell office:value-type="float" office:value="42.5"/>
<table:table-cell office:value-type="string" office:string-value="typed text"/>
</table:table-row>
</table:table>
<table:table table:name="Empty"><table:table-row table:number-rows-repeated="999999999999"><table:table-cell/></table:table-row></table:table>
<table:table table:name="Hidden Sheet" table:display="false">
<table:table-header-rows><table:table-row>
<table:table-cell table:number-columns-spanned="2"><text:p>Merged | head</text:p></table:table-cell>
<table:covered-table-cell/>
<table:table-cell table:number-columns-repeated="2"><text:p>Repeat</text:p></table:table-cell>
</table:table-row></table:table-header-rows>
<table:table-row table:number-rows-repeated="2">
<table:table-cell office:value-type="string" office:string-value="A"/>
<table:table-cell><text:p>B</text:p></table:table-cell>
<table:table-cell table:number-columns-repeated="2" office:value-type="float" office:value="7"/>
</table:table-row>
</table:table>
</office:spreadsheet></office:body>
</office:document-content>"##
	);

	let conversion = markit::convert(Path::new("ordered.ods"), &ods(&content))
		.expect("ODS conversion succeeds")
		.expect("ODS is supported");
	assert_eq!(conversion.note, None);
	assert_eq!(conversion.title, None);
	assert_eq!(
		conversion.text.as_str(),
		"## Visible & Data\n\n| Shown | Formula | Date | Time | Bool | Error | Link | Escaped |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n| Displayed 1,25 € | 3.00 shown | 2026-08-14 | 1:30:00 | FALSE | #DIV/0! | [site \\| docs](https://example.com/a%7Cb) | raw \\| string |\n| 12.5% | 1.25 EUR | 42.5 | typed text |  |  |  |  |\n\n## Hidden Sheet\n\n| Merged \\| head |  | Repeat | Repeat |\n| --- | --- | --- | --- |\n| A | B | 7 | 7 |\n| A | B | 7 | 7 |\n"
	);
	assert!(!conversion.text.contains("SUM("), "formula source is not display text");
}

#[test]
fn ods_elides_attacker_declared_trailing_sparse_ranges() {
	let content = format!(
		r#"<office:document-content {NS}><office:body><office:spreadsheet>
<table:table table:name="Sparse"><table:table-row>
<table:table-cell><text:p>only</text:p></table:table-cell>
<table:table-cell table:number-columns-repeated="18446744073709551615"/>
</table:table-row>
<table:table-row table:number-rows-repeated="18446744073709551615"><table:table-cell/></table:table-row>
</table:table></office:spreadsheet></office:body></office:document-content>"#
	);

	let conversion = markit::convert(Path::new("sparse.ods"), &ods(&content))
		.expect("trailing sparse dimensions are elided")
		.expect("ODS is supported");
	assert_eq!(conversion.text.as_str(), "|  |\n| --- |\n| only |\n");
}

#[test]
fn ods_rejects_repeat_and_internal_sparse_expansion_before_materializing_it() {
	let content = format!(
		r#"<office:document-content {NS}><office:body><office:spreadsheet>
<table:table table:name="Bomb"><table:table-row table:number-rows-repeated="4000002">
<table:table-cell><text:p>x</text:p></table:table-cell>
</table:table-row></table:table>
</office:spreadsheet></office:body></office:document-content>"#
	);

	let error = markit::convert(Path::new("repeat.ods"), &ods(&content))
		.expect_err("content-bearing repeats exceed the fixed expansion budget");
	assert_eq!(error.format(), "ods");
	assert!(error.message().contains("max_expansion"), "unexpected error: {error}");
	let sparse = format!(
		r#"<office:document-content {NS}><office:body><office:spreadsheet>
<table:table table:name="Sparse"><table:table-row>
<table:table-cell><text:p>left</text:p></table:table-cell>
<table:table-cell table:number-columns-repeated="4000001"/>
<table:table-cell><text:p>right</text:p></table:table-cell>
</table:table-row></table:table>
</office:spreadsheet></office:body></office:document-content>"#
	);
	let sparse = markit::convert(Path::new("internal-sparse.ods"), &ods(&sparse))
		.expect_err("an interior sparse range cannot allocate past the fixed budget");
	assert_eq!(sparse.format(), "ods");
	assert!(sparse.message().contains("max_expansion"), "unexpected error: {sparse}");
}

#[test]
fn ods_reports_malformed_and_encrypted_packages_without_panicking() {
	let invalid_zip = markit::convert(Path::new("not-a-zip.ods"), b"not a ZIP package")
		.expect_err("a malformed ODS container is rejected");
	assert_eq!(invalid_zip.format(), "ods");

	let malformed = markit::convert(Path::new("malformed.ods"), &ods("<not-odf/>"))
		.expect_err("an ODF package without an office body is malformed");
	assert_eq!(malformed.format(), "ods");
	assert!(malformed.message().contains("OpenDocument body"), "unexpected error: {malformed}");

	let encrypted_content = format!(
		r#"<office:document-content {NS}><office:body><office:spreadsheet/></office:body></office:document-content>"#
	);
	let manifest = r#"<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"><manifest:file-entry manifest:full-path="content.xml"><manifest:encryption-data/></manifest:file-entry></manifest:manifest>"#;
	let encrypted = zip(&[
		("mimetype", ODS_MIMETYPE),
		("META-INF/manifest.xml", manifest),
		("content.xml", &encrypted_content),
	]);
	let encrypted = markit::convert(Path::new("encrypted.ods"), &encrypted)
		.expect_err("encrypted ODS is rejected rather than interpreted");
	assert_eq!(encrypted.format(), "ods");
	assert_eq!(encrypted.message(), "document is encrypted");
}
