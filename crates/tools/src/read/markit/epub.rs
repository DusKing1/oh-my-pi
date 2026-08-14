//! EPUB to deterministic Markdown conversion.

use std::collections::BTreeMap;

use html_to_markdown_rs::{ConversionOptions, TierStrategy};
use omp_core::Str;
use quick_xml::{Reader, XmlVersion, events::Event};

use super::{
	MarkitError,
	ooxml::{Archive, decode_xml_bytes},
};

const FORMAT: &str = "epub";

#[derive(Clone, Debug, Default)]
struct Node {
	name:     String,
	attrs:    BTreeMap<String, String>,
	children: Vec<Content>,
}

#[derive(Clone, Debug)]
enum Content {
	Element(Node),
	Text(String),
}

impl Node {
	fn attr(&self, name: &str) -> Option<&str> {
		self
			.attrs
			.get(name)
			.or_else(|| self.attrs.get(local(name)))
			.map(String::as_str)
	}

	fn elements(&self) -> impl Iterator<Item = &Node> {
		self.children.iter().filter_map(|child| match child {
			Content::Element(node) => Some(node),
			Content::Text(_) => None,
		})
	}

	fn text(&self) -> String {
		let mut text = String::new();
		collect_text(self, &mut text);
		collapse_whitespace(&text)
	}
}

#[derive(Clone, Debug, Default)]
struct Metadata {
	title:       Option<String>,
	creators:    Vec<String>,
	language:    Option<String>,
	publisher:   Option<String>,
	date:        Option<String>,
	description: Option<String>,
}

#[derive(Debug)]
struct ManifestItem {
	path: String,
}

#[derive(Debug, Default)]
struct Package {
	metadata: Metadata,
	manifest: BTreeMap<String, ManifestItem>,
	spine:    Vec<String>,
}

/// Converts an EPUB package to Markdown in its declared reading order.
pub(super) fn convert(bytes: &[u8]) -> Result<(Str, Option<Str>), MarkitError> {
	let mut archive = Archive::open(bytes).map_err(failure)?;
	let container = read_xml_member(&mut archive, "META-INF/container.xml")?
		.ok_or_else(|| failure("Invalid EPUB: missing container.xml"))?;
	let opf_path = parse_container(&container)?;
	let opf_xml = read_xml_member(&mut archive, &opf_path)?
		.ok_or_else(|| failure("Invalid EPUB: missing content.opf"))?;
	let package = parse_package(&opf_xml, &opf_path)?;
	let title = package.metadata.title.clone().map(Str::from);

	let mut sections = metadata_sections(&package.metadata);
	let options = epub_html_options();
	for idref in &package.spine {
		let Some(item) = package.manifest.get(idref) else {
			continue;
		};
		let Some(xhtml) = read_xml_member(&mut archive, &item.path)? else {
			continue;
		};
		let cleaned = normalize_table_cells(&strip_script_and_style(&xhtml));
		let converted = html_to_markdown_rs::convert(&cleaned, options.clone())
			.map_err(|error| failure(format!("{}: {error}", item.path)))?;
		let markdown = normalize_converted_markdown(&converted.content.unwrap_or_default());
		let markdown = markdown.trim();
		if !markdown.is_empty() {
			sections.push(markdown.to_owned());
		}
	}

	Ok((Str::from(sections.join("\n\n").trim().to_owned()), title))
}

fn epub_html_options() -> ConversionOptions {
	let mut options = ConversionOptions::default();
	options.bullets = "-".to_owned();
	options.autolinks = false;
	options.compact_tables = true;
	options.escape_asterisks = true;
	options.escape_underscores = true;
	options.extract_metadata = false;
	options.preprocessing.enabled = false;
	options.tier_strategy = TierStrategy::Tier2;
	options
}

fn parse_container(xml: &str) -> Result<String, MarkitError> {
	let root = parse_xml(xml)?;
	let mut rootfiles = Vec::new();
	descendants(&root, "rootfile", &mut rootfiles);
	rootfiles
		.first()
		.and_then(|node| node.attr("full-path"))
		.map(str::to_owned)
		.ok_or_else(|| failure("Invalid EPUB: missing rootfile path"))
}

fn parse_package(xml: &str, opf_path: &str) -> Result<Package, MarkitError> {
	let root = parse_xml(xml)?;
	let Some(package_node) = descendant(&root, "package") else {
		return Ok(Package::default());
	};
	let metadata = child(package_node, "metadata")
		.map(parse_metadata)
		.unwrap_or_default();

	let mut manifest = BTreeMap::new();
	if let Some(manifest_node) = child(package_node, "manifest") {
		for item in manifest_node
			.elements()
			.filter(|node| local(&node.name) == "item")
		{
			let (Some(id), Some(href)) = (item.attr("id"), item.attr("href")) else {
				continue;
			};
			manifest.insert(id.to_owned(), ManifestItem { path: package_member_path(opf_path, href) });
		}
	}

	let mut spine = Vec::new();
	if let Some(spine_node) = child(package_node, "spine") {
		for itemref in spine_node
			.elements()
			.filter(|node| local(&node.name) == "itemref")
		{
			if let Some(idref) = itemref.attr("idref") {
				spine.push(idref.to_owned());
			}
		}
	}

	Ok(Package { metadata, manifest, spine })
}

fn parse_metadata(metadata: &Node) -> Metadata {
	let values = |name: &str| {
		metadata
			.elements()
			.filter(|node| local(&node.name) == name)
			.map(Node::text)
			.filter(|value| !value.is_empty())
			.collect::<Vec<_>>()
	};
	let first = |name: &str| values(name).into_iter().next();
	Metadata {
		title:       first("title"),
		creators:    values("creator"),
		language:    first("language"),
		publisher:   first("publisher"),
		date:        first("date"),
		description: first("description"),
	}
}

fn package_member_path(opf_path: &str, href: &str) -> String {
	let base = opf_path
		.rsplit_once('/')
		.map_or("", |(directory, _)| directory);
	if base.is_empty() {
		href.to_owned()
	} else {
		format!("{base}/{href}")
	}
}

fn strip_script_and_style(html: &str) -> String {
	let mut output = String::with_capacity(html.len());
	let mut rest = html;
	loop {
		let script = find_ascii_case_insensitive(rest, "<script").map(|index| (index, "script"));
		let style = find_ascii_case_insensitive(rest, "<style").map(|index| (index, "style"));
		let next = match (script, style) {
			(Some(script), Some(style)) => Some(if script.0 <= style.0 { script } else { style }),
			(Some(script), None) => Some(script),
			(None, Some(style)) => Some(style),
			(None, None) => None,
		};
		let Some((start, element)) = next else {
			output.push_str(rest);
			break;
		};
		let closing = format!("</{element}>");
		let Some(close) = find_ascii_case_insensitive(&rest[start..], &closing) else {
			output.push_str(rest);
			break;
		};
		output.push_str(&rest[..start]);
		rest = &rest[start + close + closing.len()..];
	}
	output
}

fn find_ascii_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
	haystack
		.as_bytes()
		.windows(needle.len())
		.position(|window| window.eq_ignore_ascii_case(needle.as_bytes()))
}

fn normalize_table_cells(html: &str) -> String {
	let mut output = String::with_capacity(html.len());
	let mut rest = html;
	loop {
		let td = find_cell_start(rest, "td").map(|index| (index, "td"));
		let th = find_cell_start(rest, "th").map(|index| (index, "th"));
		let next = match (td, th) {
			(Some(td), Some(th)) => Some(if td.0 <= th.0 { td } else { th }),
			(Some(td), None) => Some(td),
			(None, Some(th)) => Some(th),
			(None, None) => None,
		};
		let Some((start, tag)) = next else {
			output.push_str(rest);
			break;
		};
		let Some(open_end) = rest[start..].find('>').map(|index| start + index + 1) else {
			output.push_str(rest);
			break;
		};
		let closing = format!("</{tag}>");
		let Some(close_start) =
			find_ascii_case_insensitive(&rest[open_end..], &closing).map(|index| open_end + index)
		else {
			output.push_str(rest);
			break;
		};
		output.push_str(&rest[..open_end]);
		output.push_str(&join_cell_paragraphs(&rest[open_end..close_start]));
		output.push_str(&rest[close_start..close_start + closing.len()]);
		rest = &rest[close_start + closing.len()..];
	}
	output
}

fn find_cell_start(html: &str, tag: &str) -> Option<usize> {
	let needle = format!("<{tag}");
	let mut offset = 0usize;
	while let Some(relative) = find_ascii_case_insensitive(&html[offset..], &needle) {
		let index = offset + relative;
		let boundary = html.as_bytes().get(index + needle.len()).copied();
		if boundary.is_some_and(|byte| byte == b'>' || byte.is_ascii_whitespace()) {
			return Some(index);
		}
		offset = index + needle.len();
	}
	None
}

fn join_cell_paragraphs(inner: &str) -> String {
	let leading = inner.len() - inner.trim_start_matches(char::is_whitespace).len();
	let after_leading = &inner[leading..];
	let mut body = inner;
	if after_leading
		.get(..3)
		.is_some_and(|tag| tag.eq_ignore_ascii_case("<p>"))
	{
		body = &after_leading[3..];
	}
	let trimmed_end = body.trim_end_matches(char::is_whitespace);
	if trimmed_end
		.get(trimmed_end.len().saturating_sub(4)..)
		.is_some_and(|tag| tag.eq_ignore_ascii_case("</p>"))
	{
		body = &trimmed_end[..trimmed_end.len() - 4];
	}

	let mut output = String::with_capacity(body.len());
	let mut rest = body;
	while let Some(close) = find_ascii_case_insensitive(rest, "</p>") {
		let after_close = &rest[close + 4..];
		let whitespace =
			after_close.len() - after_close.trim_start_matches(char::is_whitespace).len();
		let after_whitespace = &after_close[whitespace..];
		if after_whitespace
			.get(..3)
			.is_some_and(|tag| tag.eq_ignore_ascii_case("<p>"))
		{
			output.push_str(&rest[..close]);
			output.push(' ');
			rest = &after_whitespace[3..];
		} else {
			output.push_str(&rest[..close + 4]);
			rest = after_close;
		}
	}
	output.push_str(rest);
	output
}

fn normalize_converted_markdown(markdown: &str) -> String {
	markdown
		.lines()
		.map(|line| {
			if line.trim_start().starts_with('|') {
				line.replace("\\|", "|")
			} else {
				line.to_owned()
			}
		})
		.collect::<Vec<_>>()
		.join("\n")
}

fn metadata_sections(metadata: &Metadata) -> Vec<String> {
	let mut lines = Vec::new();
	if let Some(title) = metadata.title.as_deref() {
		lines.push(format!("**Title:** {title}"));
	}
	if !metadata.creators.is_empty() {
		lines.push(format!("**Authors:** {}", metadata.creators.join(", ")));
	}
	for (label, value) in [
		("Language", metadata.language.as_deref()),
		("Publisher", metadata.publisher.as_deref()),
		("Date", metadata.date.as_deref()),
		("Description", metadata.description.as_deref()),
	] {
		if let Some(value) = value {
			lines.push(format!("**{label}:** {value}"));
		}
	}
	if lines.is_empty() {
		Vec::new()
	} else {
		vec![lines.join("\n")]
	}
}

fn parse_xml(xml: &str) -> Result<Node, MarkitError> {
	let mut reader = Reader::from_str(xml);
	reader.config_mut().trim_text(false);
	let mut stack = vec![Node { name: "root".to_owned(), ..Node::default() }];
	loop {
		match reader.read_event() {
			Ok(Event::Start(event)) => stack.push(event_node(&event, &reader)?),
			Ok(Event::Empty(event)) => {
				let node = event_node(&event, &reader)?;
				stack
					.last_mut()
					.expect("XML root exists")
					.children
					.push(Content::Element(node));
			},
			Ok(Event::End(_)) => {
				if stack.len() == 1 {
					return Err(failure("unexpected XML closing tag"));
				}
				let node = stack.pop().expect("XML element exists");
				stack
					.last_mut()
					.expect("XML root exists")
					.children
					.push(Content::Element(node));
			},
			Ok(Event::Text(event)) => {
				let decoded = event
					.xml_content(XmlVersion::Implicit1_0)
					.map_err(|error| failure(error.to_string()))?;
				let text =
					quick_xml::escape::unescape(&decoded).map_err(|error| failure(error.to_string()))?;
				stack
					.last_mut()
					.expect("XML root exists")
					.children
					.push(Content::Text(text.into_owned()));
			},
			Ok(Event::GeneralRef(event)) => {
				let text = if let Some(character) = event
					.resolve_char_ref()
					.map_err(|error| failure(error.to_string()))?
				{
					character.to_string()
				} else {
					let name = event.decode().map_err(|error| failure(error.to_string()))?;
					quick_xml::escape::resolve_predefined_entity(&name)
						.map_or_else(|| format!("&{name};"), str::to_owned)
				};
				stack
					.last_mut()
					.expect("XML root exists")
					.children
					.push(Content::Text(text));
			},
			Ok(Event::CData(event)) => {
				let text = event.decode().map_err(|error| failure(error.to_string()))?;
				stack
					.last_mut()
					.expect("XML root exists")
					.children
					.push(Content::Text(text.into_owned()));
			},
			Ok(Event::Eof) => break,
			Ok(_) => {},
			Err(error) => return Err(failure(error.to_string())),
		}
	}
	if stack.len() != 1 {
		return Err(failure("unterminated XML element"));
	}
	Ok(stack.pop().expect("XML root exists"))
}

fn event_node(
	event: &quick_xml::events::BytesStart<'_>,
	reader: &Reader<&[u8]>,
) -> Result<Node, MarkitError> {
	let name = String::from_utf8_lossy(event.name().as_ref()).into_owned();
	let mut attrs = BTreeMap::new();
	for attribute in event.attributes().with_checks(false) {
		let attribute = attribute.map_err(|error| failure(error.to_string()))?;
		let key = String::from_utf8_lossy(attribute.key.as_ref()).into_owned();
		let value = attribute
			.decoded_and_normalized_value(XmlVersion::Implicit1_0, reader.decoder())
			.map_err(|error| failure(error.to_string()))?
			.into_owned();
		attrs.insert(key.clone(), value.clone());
		attrs.entry(local(&key).to_owned()).or_insert(value);
	}
	Ok(Node { name, attrs, children: Vec::new() })
}

fn read_xml_member(archive: &mut Archive<'_>, path: &str) -> Result<Option<String>, MarkitError> {
	archive
		.read_xml(path)
		.map_err(failure)?
		.map(|bytes| {
			decode_xml_bytes(&bytes)
				.map_err(|error| failure(format!("'{path}' is not valid UTF text: {error}")))
		})
		.transpose()
}

fn collect_text(node: &Node, output: &mut String) {
	for child in &node.children {
		match child {
			Content::Text(text) => {
				output.push_str(text);
				output.push(' ');
			},
			Content::Element(node) => collect_text(node, output),
		}
	}
}

fn collapse_whitespace(value: &str) -> String {
	value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn child<'a>(node: &'a Node, name: &str) -> Option<&'a Node> {
	node.elements().find(|child| local(&child.name) == name)
}

fn descendant<'a>(node: &'a Node, name: &str) -> Option<&'a Node> {
	if local(&node.name) == name {
		return Some(node);
	}
	node.elements().find_map(|child| descendant(child, name))
}

fn descendants<'a>(node: &'a Node, name: &str, output: &mut Vec<&'a Node>) {
	for child in node.elements() {
		if local(&child.name) == name {
			output.push(child);
		}
		descendants(child, name, output);
	}
}

fn local(name: &str) -> &str {
	name.rsplit_once(':').map_or(name, |(_, local)| local)
}

fn failure(message: impl Into<Str>) -> MarkitError {
	MarkitError::conversion(FORMAT, message)
}
