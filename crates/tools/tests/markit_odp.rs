//! OpenDocument Presentation conversion fixtures.

use std::path::Path;

use omp_ar::zip::Writer;
use omp_tools::read::markit;

const ODP_MIME: &str = "application/vnd.oasis.opendocument.presentation";

fn zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
	let mut writer = Writer::new(Vec::new());
	for (path, content) in entries {
		writer.add_file(path, content).expect("fixture member adds");
	}
	writer.finish().expect("fixture archive finishes")
}

fn valid_odp() -> Vec<u8> {
	let content = br#"<office:document-content
		xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
		xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
		xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
		xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
		xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
		xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
		xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
		xmlns:xlink="http://www.w3.org/1999/xlink"
		xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0">
		<office:automatic-styles>
			<style:style style:name="Strong" style:family="text">
				<style:text-properties fo:font-weight="bold"/>
			</style:style>
		</office:automatic-styles>
		<office:body><office:presentation>
			<draw:page draw:name="slide-1">
				<draw:frame presentation:class="outline"><draw:text-box>
					<text:p>Read the <text:a xlink:href="https://example.com/docs">external docs</text:a> for <text:span text:style-name="Strong">important</text:span> details.</text:p>
					<text:list><text:list-item><text:p>First point</text:p></text:list-item><text:list-item><text:p>Second point</text:p></text:list-item></text:list>
				</draw:text-box></draw:frame>
				<draw:frame><svg:title>Architecture diagram</svg:title><draw:image xlink:href="Pictures/diagram.png"/></draw:frame>
				<draw:frame presentation:class="title"><draw:text-box><text:p>Opening</text:p></draw:text-box></draw:frame>
				<presentation:notes><draw:frame><draw:text-box><text:p>Presenter note one.</text:p></draw:text-box></draw:frame></presentation:notes>
			</draw:page>
			<draw:page draw:name="slide-2">
				<draw:frame presentation:class="title"><draw:text-box><text:p>Results</text:p></draw:text-box></draw:frame>
				<draw:frame><table:table>
					<table:table-row><table:table-cell><text:p>Region</text:p></table:table-cell><table:table-cell><text:p>Total</text:p></table:table-cell></table:table-row>
					<table:table-row table:number-rows-repeated="2"><table:table-cell><text:p>North</text:p></table:table-cell><table:table-cell><text:p>42</text:p></table:table-cell></table:table-row>
				</table:table></draw:frame>
				<presentation:notes><draw:frame><draw:text-box><text:p>Presenter note two.</text:p></draw:text-box></draw:frame></presentation:notes>
			</draw:page>
		</office:presentation></office:body>
	</office:document-content>"#;
	zip(&[
		("mimetype", ODP_MIME.as_bytes()),
		("content.xml", content),
		("Pictures/diagram.png", b"embedded image payload"),
	])
}

#[test]
fn odp_preserves_slide_structure_content_and_notes() {
	let conversion = markit::convert(Path::new("deck.odp"), &valid_odp())
		.expect("ODP conversion succeeds")
		.expect("ODP is supported");
	let text = conversion.text.as_str();

	let opening = text.find("Opening").expect("first slide title is rendered");
	let first_point = text
		.find("First point")
		.expect("first slide body is rendered");
	let results = text
		.find("Results")
		.expect("second slide title is rendered");
	assert!(opening < first_point, "title frames render before body frames");
	assert!(first_point < results, "package page order is preserved");
	assert!(text.contains("## Opening"));
	assert!(text.contains("## Results"));
	assert!(text.contains("[external docs](https://example.com/docs)"));
	assert!(text.contains("**important**"));
	assert!(text.contains("- First point"));
	assert!(text.contains("Architecture diagram"));
	assert!(!text.contains("embedded image payload"));
	assert!(text.contains("| Region | Total |"));
	assert_eq!(text.matches("| North | 42 |").count(), 2, "small repeated rows expand");
	assert!(text.contains("> Presenter note one."));
	assert!(text.contains("> Presenter note two."));
	assert_eq!(conversion.note, None);
	assert_eq!(conversion.title, None);
}

#[test]
fn odp_rejects_malformed_and_missing_content_without_panicking() {
	let missing_content = zip(&[("mimetype", ODP_MIME.as_bytes())]);
	for bytes in [b"not a zip archive".as_slice(), missing_content.as_slice()] {
		let error =
			markit::convert(Path::new("broken.odp"), bytes).expect_err("malformed ODP is rejected");
		assert_eq!(error.format(), "odp");
	}
}

#[test]
fn odp_rejects_wrong_package_identity_and_encryption() {
	let wrong_type = zip(&[
		("mimetype", b"application/vnd.oasis.opendocument.text"),
		("content.xml", b"<content/>"),
	]);
	let error = markit::convert(Path::new("wrong.odp"), &wrong_type)
		.expect_err("a text package must not be accepted as ODP");
	assert_eq!(error.format(), "odp");

	let wrong_manifest = br#"<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"><manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.spreadsheet"/></manifest:manifest>"#;
	let wrong_type =
		zip(&[("META-INF/manifest.xml", wrong_manifest), ("content.xml", b"<content/>")]);
	let error = markit::convert(Path::new("wrong-manifest.odp"), &wrong_type)
		.expect_err("a conflicting manifest must not be accepted as ODP");
	assert_eq!(error.format(), "odp");
	assert!(error.message().contains("unexpected OpenDocument mimetype"));

	let manifest = br#"<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"><manifest:file-entry manifest:full-path="content.xml"><manifest:encryption-data/></manifest:file-entry></manifest:manifest>"#;
	let encrypted = zip(&[
		("mimetype", ODP_MIME.as_bytes()),
		("META-INF/manifest.xml", manifest),
		("content.xml", b"<content/>"),
	]);
	let error = markit::convert(Path::new("encrypted.odp"), &encrypted)
		.expect_err("encrypted ODP is rejected");
	assert_eq!(error.format(), "odp");
	assert!(error.message().to_ascii_lowercase().contains("encrypt"));
}
