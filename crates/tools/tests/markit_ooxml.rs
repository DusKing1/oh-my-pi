//! Exact-output workbook and presentation conversion fixtures.

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
		"<!-- Slide 1 -->\n# Plan & Scope\nRich text\n<!-- image: Diagram & Flow (slide 1) -->\n| \
		 Key | Value & More |\n| --- | --- |\n| A | two parts |\n\n### Notes:\nFirst & note\nSecond \
		 note\n\n<!-- Slide 2 -->\n# Appendix"
	);
}

#[test]
fn port_pptx_reports_a_malformed_archive_exactly() {
	let error =
		markit::convert(Path::new("broken.pptx"), &zip(&[("ppt/slides/slide1.xml", "<p:sld/>")]))
			.expect_err("presentation-less PPTX is malformed");
	assert_eq!(error.to_string(), "pptx conversion failed: Invalid PPTX: missing presentation.xml");
}
