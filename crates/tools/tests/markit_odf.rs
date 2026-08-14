//! Compact, synthetic OpenDocument conversion fixtures.

use std::path::Path;

use omp_ar::zip::Writer;
use omp_tools::read::markit;

const ODT_MIMETYPE: &str = "application/vnd.oasis.opendocument.text";

fn zip(entries: &[(&str, &str)]) -> Vec<u8> {
	let mut writer = Writer::new(Vec::new());
	for (path, content) in entries {
		writer
			.add_file(path, content.as_bytes())
			.expect("fixture member adds");
	}
	writer.finish().expect("fixture archive finishes")
}

fn structured_odt() -> Vec<u8> {
	let styles = r#"<office:document-styles
		xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
		xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
		xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
		xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0">
		<office:styles>
			<style:style style:name="Bold" style:family="text"><style:text-properties fo:font-weight="bold"/></style:style>
			<style:style style:name="Italic" style:family="text"><style:text-properties fo:font-style="italic"/></style:style>
			<text:list-style style:name="Bullets"><text:list-level-style-bullet text:level="1" text:bullet-char="•"/></text:list-style>
		</office:styles>
	</office:document-styles>"#;
	// Deliberately uses non-canonical namespace prefixes. Namespace URIs, not
	// producer-chosen prefixes, define ODF elements.
	let content = r##"<o:document-content
		xmlns:o="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
		xmlns:t="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
		xmlns:tb="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
		xmlns:dr="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
		xmlns:xl="http://www.w3.org/1999/xlink"
		xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0">
		<o:body><o:text>
			<t:tracked-changes><t:changed-region><t:deletion><t:p>deleted secret</t:p></t:deletion></t:changed-region></t:tracked-changes>
			<t:h t:outline-level="2">A heading</t:h>
			<t:p>Plain<t:s t:c="2"/>text<t:tab/>with<t:line-break/>break and <t:span t:style-name="Bold">bold</t:span>, <t:span t:style-name="Italic">italic</t:span>.</t:p>
			<t:p><t:bookmark-start t:name="place"/><t:a xl:href="https://example.test/path">Example link</t:a><t:note t:id="fn1"><t:note-citation>1</t:note-citation><t:note-body><t:p>Footnote body</t:p></t:note-body></t:note></t:p>
			<t:list t:style-name="Bullets"><t:list-item><t:p>First item</t:p></t:list-item><t:list-item><t:p>Second item</t:p></t:list-item></t:list>
			<tb:table><tb:table-row><tb:table-cell tb:number-columns-spanned="2"><t:p>Merged heading</t:p></tb:table-cell><tb:covered-table-cell/></tb:table-row><tb:table-row><tb:table-cell><t:p>Left</t:p></tb:table-cell><tb:table-cell><t:p>Right</t:p></tb:table-cell></tb:table-row></tb:table>
			<t:p><dr:frame><svg:title>Architecture diagram</svg:title></dr:frame></t:p>
		</o:text></o:body>
	</o:document-content>"##;
	let manifest = format!(
		r#"<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"><manifest:file-entry manifest:full-path="/" manifest:media-type="{ODT_MIMETYPE}"/><manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/><manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/><manifest:file-entry manifest:full-path="meta.xml" manifest:media-type="text/xml"/></manifest:manifest>"#
	);
	zip(&[
		("mimetype", ODT_MIMETYPE),
		("META-INF/manifest.xml", &manifest),
		("styles.xml", styles),
		("content.xml", content),
		("meta.xml", "<metadata>must not leak</metadata>"),
		("Scripts/macro.bin", "must not execute or leak"),
	])
}

#[test]
fn odt_preserves_document_structure_and_ignores_non_content_parts() {
	let conversion = markit::convert(Path::new("structured.odt"), &structured_odt())
		.expect("ODT conversion succeeds")
		.expect("ODT is supported");
	let markdown = conversion.text.as_str();

	assert!(markdown.contains("## A heading"));
	assert!(markdown.contains("Plain  text with"));
	assert!(markdown.contains("break and **bold**, *italic*."));
	assert!(markdown.contains("[Example link](https://example.test/path)"));
	assert!(markdown.contains("First item"));
	assert!(markdown.contains("Second item"));
	assert!(markdown.contains("Merged heading"));
	assert!(markdown.contains("Left"));
	assert!(markdown.contains("Right"));
	assert!(markdown.contains("Footnote body"));
	assert!(markdown.contains("Architecture diagram"));
	assert!(!markdown.contains("deleted secret"));
	assert!(!markdown.contains("must not leak"));
	assert!(!markdown.contains("must not execute"));
	assert_eq!(conversion.note, None);
	assert_eq!(conversion.title, None);
}

#[test]
fn odt_reports_missing_or_malformed_content_without_panicking() {
	let missing = zip(&[("mimetype", ODT_MIMETYPE)]);
	let error =
		markit::convert(Path::new("missing.odt"), &missing).expect_err("content.xml is required");
	assert_eq!(error.format(), "odt");
	assert!(error.message().contains("missing content.xml"));

	let malformed = zip(&[("mimetype", ODT_MIMETYPE), ("content.xml", "<broken>")]);
	let error = markit::convert(Path::new("malformed.odt"), &malformed)
		.expect_err("malformed XML is rejected");
	assert_eq!(error.format(), "odt");
	assert!(!error.message().is_empty());

	let error = markit::convert(Path::new("not-a-zip.odt"), b"not a ZIP archive")
		.expect_err("malformed ZIP is rejected");
	assert_eq!(error.format(), "odt");
	assert!(error.message().contains("ZIP"));
}

#[test]
fn odt_rejects_conflicting_package_identity_and_encryption() {
	let wrong_kind = zip(&[
		("mimetype", "application/vnd.oasis.opendocument.spreadsheet"),
		("content.xml", "<document/>"),
	]);
	let error = markit::convert(Path::new("wrong.odt"), &wrong_kind)
		.expect_err("conflicting mimetype is rejected");
	assert_eq!(error.format(), "odt");
	assert!(error.message().contains("unexpected OpenDocument mimetype"));

	let encrypted = zip(&[
		("mimetype", ODT_MIMETYPE),
		("content.xml", "encrypted bytes are not XML"),
		(
			"META-INF/manifest.xml",
			r#"<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"><manifest:file-entry manifest:full-path="content.xml"><manifest:encryption-data/></manifest:file-entry></manifest:manifest>"#,
		),
	]);
	let error = markit::convert(Path::new("encrypted.odt"), &encrypted)
		.expect_err("encrypted ODT is rejected");
	assert_eq!(error.format(), "odt");
	assert!(error.message().to_ascii_lowercase().contains("encrypt"));
}

#[test]
fn odt_uses_manifest_identity_when_mimetype_is_absent() {
	let content = r#"<office:document-content
		xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
		xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
		<office:body><office:text><text:p>Recovered ODT</text:p></office:text></office:body>
	</office:document-content>"#;
	let manifest = format!(
		r#"<m:manifest xmlns:m="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"><m:file-entry m:full-path="/" m:media-type="{ODT_MIMETYPE}"/></m:manifest>"#
	);
	let bytes = zip(&[("META-INF/manifest.xml", &manifest), ("content.xml", content)]);
	let conversion = markit::convert(Path::new("manifest-only.odt"), &bytes)
		.expect("manifest-identified ODT converts")
		.expect("ODT is supported");
	assert_eq!(conversion.text.as_str(), "Recovered ODT\n");

	let wrong_manifest = r#"<m:manifest xmlns:m="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"><m:file-entry m:full-path="/" m:media-type="application/vnd.oasis.opendocument.presentation"/></m:manifest>"#;
	let bytes = zip(&[("META-INF/manifest.xml", wrong_manifest), ("content.xml", content)]);
	let error = markit::convert(Path::new("wrong-manifest.odt"), &bytes)
		.expect_err("conflicting manifest identity is rejected");
	assert!(error.message().contains("unexpected OpenDocument mimetype"));
}
