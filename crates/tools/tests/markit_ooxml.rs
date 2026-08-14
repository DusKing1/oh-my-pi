//! Exact-output OOXML conversion fixtures.

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

fn zip_bytes(entries: &[(&str, &[u8])]) -> Vec<u8> {
	let mut writer = Writer::new(Vec::new());
	for (path, content) in entries {
		writer.add_file(path, content).expect("fixture member adds");
	}
	writer.finish().expect("fixture archive finishes")
}

#[test]
fn port_xlsx_preserves_declared_sheet_order_and_cell_display() {
	let bytes = zip(&[
		(
			"xl/workbook.xml",
			r#"<workbook xmlns:r="r"><sheets><sheet name="First &amp; Main" sheetId="2" r:id="rId2"/><sheet name="Second" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
		),
		(
			"xl/_rels/workbook.xml.rels",
			r#"<Relationships><Relationship Id="rId1" Target="/xl/worksheets/sheet1.xml"/><Relationship Id="rId2" Target="worksheets/sheet2.xml"/></Relationships>"#,
		),
		(
			"xl/sharedStrings.xml",
			r"<sst><si><t>A &amp; B</t></si><si><r><t>Rich </t></r><r><t>text</t></r></si></sst>",
		),
		(
			"xl/worksheets/sheet2.xml",
			r#"<worksheet><sheetData><row><c t="s"><v>0</v></c><c t="s"><v>1</v></c></row><row><c><f>1+2</f><v>3</v></c><c t="b"><v>1</v></c></row></sheetData></worksheet>"#,
		),
		(
			"xl/worksheets/sheet1.xml",
			r#"<worksheet><sheetData><row><c t="inlineStr"><is><r><t>Inline</t></r><r><t> rich</t></r></is></c><c t="str"><v>Status</v></c></row><row><c t="str"><f>TEXT(1)</f><v>done &amp; ready</v></c><c t="b"><v>0</v></c></row></sheetData></worksheet>"#,
		),
	]);

	let conversion = markit::convert(Path::new("ordered.xlsx"), &bytes)
		.expect("XLSX conversion succeeds")
		.expect("XLSX is supported");
	assert_eq!(
		conversion.text.as_str(),
		"## First & Main\n\n| A & B | Rich text |\n| --- | --- |\n| 3 | TRUE |\n\n## Second\n\n| \
		 Inline rich | Status |\n| --- | --- |\n| done & ready | FALSE |"
	);
}

#[test]
fn port_xlsx_reports_a_malformed_archive_exactly() {
	let error =
		markit::convert(Path::new("broken.xlsx"), &zip(&[("xl/sharedStrings.xml", "<sst/>")]))
			.expect_err("workbook-less XLSX is malformed");
	assert_eq!(error.to_string(), "xlsx conversion failed: Invalid XLSX: missing workbook.xml");
}

#[test]
fn xlsm_matches_xlsx_and_ignores_macro_and_external_link_parts() {
	let workbook = r#"<workbook xmlns:r="r"><sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets><externalReferences><externalReference r:id="rId3"/></externalReferences></workbook>"#;
	let worksheet = r#"<worksheet><sheetData><row><c t="inlineStr"><is><t>Name</t></is></c><c t="inlineStr"><is><t>Value</t></is></c></row><row><c t="inlineStr"><is><t>answer</t></is></c><c><v>42</v></c></row></sheetData></worksheet>"#;
	let xlsx = zip(&[
		("xl/workbook.xml", workbook),
		(
			"xl/_rels/workbook.xml.rels",
			r#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#,
		),
		("xl/worksheets/sheet1.xml", worksheet),
	]);
	let xlsm = zip(&[
		(
			"[Content_Types].xml",
			r#"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="bin" ContentType="application/vnd.ms-office.vbaProject"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.ms-excel.sheet.macroEnabled.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/externalLinks/externalLink1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.externalLink+xml"/></Types>"#,
		),
		("xl/workbook.xml", workbook),
		(
			"xl/_rels/workbook.xml.rels",
			r#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.microsoft.com/office/2006/relationships/vbaProject" Target="vbaProject.bin"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/externalLink" Target="externalLinks/externalLink1.xml"/></Relationships>"#,
		),
		("xl/worksheets/sheet1.xml", worksheet),
		("xl/vbaProject.bin", "macro payload must not become worksheet text or execute"),
		(
			"xl/externalLinks/externalLink1.xml",
			r#"<externalLink><externalBook><sheetDataSet><sheetData><row><c><v>external secret</v></c></row></sheetData></sheetDataSet></externalBook></externalLink>"#,
		),
	]);

	let expected = markit::convert(Path::new("equivalent.xlsx"), &xlsx)
		.expect("XLSX conversion succeeds")
		.expect("XLSX is supported");
	let actual = markit::convert(Path::new("macro.xlsm"), &xlsm)
		.expect("XLSM conversion succeeds")
		.expect("XLSM is supported");
	assert_eq!(actual, expected);
	assert_eq!(actual.text.as_str(), "## Data\n\n| Name | Value |\n| --- | --- |\n| answer | 42 |");
	assert!(!actual.text.contains("macro payload"));
	assert!(!actual.text.contains("external secret"));
}

#[test]
fn xlsm_reports_malformed_workbooks_through_the_xlsx_converter() {
	let error =
		markit::convert(Path::new("broken.xlsm"), &zip(&[("xl/vbaProject.bin", "not a workbook")]))
			.expect_err("workbook-less XLSM is malformed");
	assert_eq!(error.to_string(), "xlsx conversion failed: Invalid XLSX: missing workbook.xml");
}

#[test]
fn docm_matches_docx_and_ignores_the_vba_project() {
	const ROOT_RELS: &[u8] = br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>"#;
	const DOCUMENT: &[u8] = br#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Macro-safe body</w:t></w:r></w:p></w:body></w:document>"#;
	const EMPTY_RELS: &[u8] =
		br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"#;
	const MACRO_RELS: &[u8] = br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdVba" Type="http://schemas.microsoft.com/office/2006/relationships/vbaProject" Target="vbaProject.bin"/></Relationships>"#;
	const CONTENT_TYPES: &[u8] = br#"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="bin" ContentType="application/vnd.ms-office.vbaProject"/><Override PartName="/word/document.xml" ContentType="application/vnd.ms-word.document.macroEnabled.main+xml"/></Types>"#;
	const VBA_PROJECT: &[u8] =
		b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1\x00\xffmacro payload is not document text";

	let docx = zip_bytes(&[
		("_rels/.rels", ROOT_RELS),
		("word/document.xml", DOCUMENT),
		("word/_rels/document.xml.rels", EMPTY_RELS),
	]);
	let docm = zip_bytes(&[
		("[Content_Types].xml", CONTENT_TYPES),
		("_rels/.rels", ROOT_RELS),
		("word/document.xml", DOCUMENT),
		("word/_rels/document.xml.rels", MACRO_RELS),
		("word/vbaProject.bin", VBA_PROJECT),
	]);

	let expected = markit::convert(Path::new("equivalent.docx"), &docx)
		.expect("DOCX conversion succeeds")
		.expect("DOCX is supported");
	let actual = markit::convert(Path::new("macro.docm"), &docm)
		.expect("DOCM conversion succeeds")
		.expect("DOCM is supported");
	assert_eq!(actual, expected);
	assert_eq!(actual.text.as_str(), "Macro-safe body");
	assert!(!actual.text.contains("macro payload"));
}

#[test]
fn docm_reports_malformed_packages_through_the_docx_converter() {
	let bytes =
		zip_bytes(&[("word/vbaProject.bin", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1not a document")]);
	let error = markit::convert(Path::new("broken.docm"), &bytes)
		.expect_err("document-less DOCM is malformed");
	assert_eq!(error.format(), "docx");
	assert!(error.message().contains("missing word/document.xml"));
}

#[test]
fn port_pptx_preserves_slide_content_and_speaker_note_order() {
	let bytes = zip(&[
		(
			"ppt/presentation.xml",
			r#"<p:presentation xmlns:p="p" xmlns:r="r"><p:sldIdLst><p:sldId r:id="rId2"/><p:sldId r:id="rId1"/></p:sldIdLst></p:presentation>"#,
		),
		(
			"ppt/_rels/presentation.xml.rels",
			r#"<Relationships><Relationship Id="rId1" Target="slides/slide1.xml"/><Relationship Id="rId2" Target="slides/slide2.xml"/></Relationships>"#,
		),
		(
			"ppt/slides/slide2.xml",
			r#"<p:sld xmlns:p="p" xmlns:a="a" xmlns:r="r"><p:cSld><p:spTree>
<p:graphicFrame><a:graphic><a:graphicData><a:tbl><a:tr><a:tc><a:txBody><a:p><a:r><a:t>Key</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>Value &amp; More</a:t></a:r></a:p></a:txBody></a:tc></a:tr><a:tr><a:tc><a:txBody><a:p><a:r><a:t>A</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>two</a:t></a:r><a:r><a:t>parts</a:t></a:r></a:p></a:txBody></a:tc></a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>
<p:sp><p:txBody><a:p><a:r><a:t>Plan &amp; Scope</a:t></a:r></a:p></p:txBody></p:sp>
<p:pic><p:nvPicPr><p:cNvPr name="Diagram &amp; Flow"/></p:nvPicPr><p:blipFill><a:blip r:embed="rImg1"/></p:blipFill></p:pic>
<p:sp><p:txBody><a:p><a:r><a:t>Rich</a:t></a:r><a:r><a:t> text</a:t></a:r></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>"#,
		),
		(
			"ppt/slides/_rels/slide2.xml.rels",
			r#"<Relationships><Relationship Id="rImg1" Target="../media/image1.png"/></Relationships>"#,
		),
		("ppt/media/image1.png", "image"),
		(
			"ppt/notesSlides/notesSlide2.xml",
			r#"<p:notes xmlns:p="p" xmlns:a="a"><p:cSld><p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>skip me</a:t></a:r></a:p></p:txBody></p:sp><p:sp><p:txBody><a:p><a:r><a:t>First &amp; note</a:t></a:r></a:p></p:txBody></p:sp><p:sp><p:txBody><a:p><a:r><a:t>Second note</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>"#,
		),
		(
			"ppt/slides/slide1.xml",
			r#"<p:sld xmlns:p="p" xmlns:a="a"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Appendix</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"#,
		),
	]);

	let conversion = markit::convert(Path::new("ordered.pptx"), &bytes)
		.expect("PPTX conversion succeeds")
		.expect("PPTX is supported");
	assert_eq!(
		conversion.text.as_str(),
		"<!-- Slide 1 -->\n| Key | Value & More |\n| --- | --- |\n| A | twoparts |\n# Plan & \
		 Scope\n<!-- image: Diagram & Flow (slide 1) -->\nRich text\n\n### Notes:\nFirst & \
		 note\nSecond note\n\n<!-- Slide 2 -->\n# Appendix"
	);
}

#[test]
fn port_pptx_reports_a_malformed_archive_exactly() {
	let error =
		markit::convert(Path::new("broken.pptx"), &zip(&[("ppt/slides/slide1.xml", "<p:sld/>")]))
			.expect_err("presentation-less PPTX is malformed");
	assert_eq!(error.to_string(), "pptx conversion failed: Invalid PPTX: missing presentation.xml");
}
