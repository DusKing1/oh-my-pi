//! Focused synthetic XLSX conversion fixtures.

use std::path::Path;

use omp_ar::zip::Writer;
use omp_tools::read::markit;

fn zip(entries: &[(&str, &str)]) -> Vec<u8> {
	let mut writer = Writer::new(Vec::new());
	for (path, content) in entries {
		writer
			.add_file(path, content.as_bytes())
			.expect("fixture member adds");
	}
	writer.finish().expect("fixture archive finishes")
}

fn workbook(sheet: &str, styles: Option<&str>, shared: Option<&str>) -> Vec<u8> {
	let mut entries = vec![
		(
			"xl/workbook.xml",
			r#"<workbook xmlns:r="r"><workbookPr date1904="1"/><sheets><sheet name="Data" state="hidden" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
		),
		(
			"xl/_rels/workbook.xml.rels",
			r#"<Relationships><Relationship Id="external" TargetMode="External" Target="https://example.com/evil.xml"/><Relationship Id="rId1" Target="worksheets/../worksheets/sheet1.xml"/></Relationships>"#,
		),
		("xl/worksheets/sheet1.xml", sheet),
	];
	if let Some(styles) = styles {
		entries.push(("xl/styles.xml", styles));
	}
	if let Some(shared) = shared {
		entries.push(("xl/sharedStrings.xml", shared));
	}
	zip(&entries)
}

#[test]
fn xlsx_preserves_sparse_coordinates_and_formats_typed_values() {
	let bytes = workbook(
		r#"<worksheet><sheetData>
<row r="5"><c r="C5" t="inlineStr"><is><r><t>Name|Label</t></r></is></c><c r="E5" t="str"><f>"Formula"</f><v>Formula</v></c></row>
<row r="7"><c r="C7" t="s"><v>0</v></c><c r="D7" t="b"><v>1</v></c><c r="E7" s="1"><v>0</v></c></row>
<row r="8"><c r="C8" t="e"><v>#DIV/0!</v></c><c r="D8" s="2"><v>0.5</v></c><c r="E8" s="3"><v>1.10434027777778</v></c></row>
<row r="9"><c r="C9" t="inlineStr"><is><t>Merged</t></is></c><c r="D9" t="str"><v>covered</v></c><c r="E9" t="inlineStr"><is><t>Tail</t></is></c></row>
<row r="10"><c r="C10"><v>3554.7000000000003</v></c><c r="E10"><v>0.0000004</v></c></row>
</sheetData><mergeCells><mergeCell ref="C9:D9"/></mergeCells></worksheet>"#,
		Some(
			r#"<styleSheet><numFmts count="1"><numFmt numFmtId="165" formatCode="[h]:mm:ss"/></numFmts><cellXfs count="4"><xf numFmtId="0"/><xf numFmtId="14"/><xf numFmtId="20"/><xf numFmtId="165"/></cellXfs></styleSheet>"#,
		),
		Some(
			r#"<sst><si><r><t>Shared </t></r><r><t>rich &amp;copy;</t></r><rPh><t>phonetic</t></rPh></si></sst>"#,
		),
	);

	let conversion = markit::convert(Path::new("typed.xlsx"), &bytes)
		.expect("XLSX conversion succeeds")
		.expect("XLSX is supported");
	assert_eq!(
		conversion.text.as_str(),
		"## Data\n\n| Name\\|Label |  | Formula |\n| --- | --- | --- |\n|  |  |  |\n| Shared rich \
		 &amp;copy; | TRUE | 1904-01-01 |\n| #DIV/0! | 12:00:00 | 26:30:15 |\n| Merged |  | Tail \
		 |\n| 3554.7 |  | 0.0000004 |"
	);
}

#[test]
fn xlsx_rejects_sparse_dimensions_that_would_force_dense_allocation() {
	let bytes = workbook(
		r#"<worksheet><sheetData><row r="1"><c r="A1"><v>1</v></c></row><row r="1048576"><c r="XFD1048576"><v>2</v></c></row></sheetData></worksheet>"#,
		None,
		None,
	);

	let error = markit::convert(Path::new("sparse.xlsx"), &bytes)
		.expect_err("adversarial sparse dimensions must be bounded");
	assert!(
		error
			.to_string()
			.contains("sparse worksheet would require 17179869184 rendered cells (limit 1000000)"),
		"unexpected error: {error}"
	);
}

#[test]
fn xlsx_rejects_out_of_range_and_invalid_shared_string_references() {
	let out_of_range = workbook(
		r#"<worksheet><sheetData><row><c r="XFE1"><v>1</v></c></row></sheetData></worksheet>"#,
		None,
		None,
	);
	let error = markit::convert(Path::new("column.xlsx"), &out_of_range)
		.expect_err("columns beyond XFD are invalid");
	assert!(error.to_string().contains("invalid cell reference 'XFE1'"));

	let bad_shared = workbook(
		r#"<worksheet><sheetData><row><c r="A1" t="s"><v>3</v></c></row></sheetData></worksheet>"#,
		None,
		Some("<sst><si><t>only</t></si></sst>"),
	);
	let error = markit::convert(Path::new("shared.xlsx"), &bad_shared)
		.expect_err("shared-string indexes must be checked");
	assert!(
		error
			.to_string()
			.contains("shared-string index 3 is out of bounds")
	);

	let incomplete = workbook("<worksheet><sheetData>", None, None);
	let error = markit::convert(Path::new("incomplete.xlsx"), &incomplete)
		.expect_err("truncated worksheet XML must be rejected");
	assert!(
		error
			.to_string()
			.contains("missing or incomplete sheetData")
	);
}

#[test]
fn xlsx_keeps_invalid_booleans_truthful_and_escapes_markdown_syntax() {
	let bytes = workbook(
		r#"<worksheet><sheetData><row><c r="A1" t="b"><v>2</v></c><c r="B1" t="inlineStr"><is><t>*literal*</t></is></c></row></sheetData></worksheet>"#,
		None,
		None,
	);
	let conversion = markit::convert(Path::new("boolean.xlsx"), &bytes)
		.expect("XLSX conversion succeeds")
		.expect("XLSX is supported");
	assert_eq!(conversion.text.as_str(), "## Data\n\n| 2 | \\*literal\\* |\n| --- | --- |");
}
