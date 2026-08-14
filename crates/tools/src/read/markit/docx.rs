//! DOCX to Markdown conversion.

use std::collections::{HashMap, HashSet};

use omp_core::Str;
use quick_xml::{Reader, XmlVersion, events::Event};

use super::{
	MarkitError,
	ooxml::{Archive, decode_xml_bytes},
};

const FORMAT: &str = "docx";

#[derive(Clone, Debug, Default)]
struct Node {
	name:     String,
	attrs:    HashMap<String, String>,
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

	fn child(&self, name: &str) -> Option<&Node> {
		self.children.iter().find_map(|child| match child {
			Content::Element(node) if local(&node.name) == name => Some(node),
			_ => None,
		})
	}

	fn elements(&self) -> impl Iterator<Item = &Node> {
		self.children.iter().filter_map(|child| match child {
			Content::Element(node) => Some(node),
			Content::Text(_) => None,
		})
	}
}

#[derive(Clone, Debug, Default)]
struct ParagraphStyle {
	name:      String,
	based_on:  Option<String>,
	numbering: Option<(String, usize)>,
}

#[derive(Clone, Copy, Debug, Default)]
struct NumberLevel {
	ordered: bool,
}

struct Context {
	relationships: HashMap<String, String>,
	styles:        HashMap<String, ParagraphStyle>,
	numbering:     HashMap<(String, usize), NumberLevel>,
	counters:      HashMap<(bool, usize), usize>,
}

/// Converts an Office Open XML word-processing document to deterministic
/// Markdown.
pub(super) fn convert(bytes: &[u8]) -> Result<Str, MarkitError> {
	let mut archive =
		Archive::open(bytes).map_err(|error| MarkitError::conversion(FORMAT, error))?;
	let document = read_member(&mut archive, "word/document.xml")?
		.ok_or_else(|| MarkitError::conversion(FORMAT, "Invalid DOCX: missing word/document.xml"))?;
	let relationships = read_member(&mut archive, "word/_rels/document.xml.rels")?
		.map(|xml| parse_relationships(&xml))
		.transpose()?
		.unwrap_or_default();
	let styles = read_member(&mut archive, "word/styles.xml")?
		.map(|xml| parse_styles(&xml))
		.transpose()?
		.unwrap_or_default();
	let numbering = read_member(&mut archive, "word/numbering.xml")?
		.map(|xml| parse_numbering(&xml))
		.transpose()?
		.unwrap_or_default();

	let root = parse_xml(&document)?;
	let body = descendant(&root, "body")
		.ok_or_else(|| MarkitError::conversion(FORMAT, "Invalid DOCX: missing document body"))?;
	let mut context = Context { relationships, styles, numbering, counters: HashMap::new() };
	let mut blocks = Vec::new();
	render_block_children(body, &mut context, &mut blocks);
	Ok(Str::from(blocks.join("\n\n").trim().to_owned()))
}

fn read_member(archive: &mut Archive<'_>, path: &str) -> Result<Option<String>, MarkitError> {
	let Some(bytes) = archive
		.read_xml(path)
		.map_err(|error| MarkitError::conversion(FORMAT, error))?
	else {
		return Ok(None);
	};
	let text = decode_xml_bytes(&bytes).map_err(|error| {
		MarkitError::conversion(FORMAT, format!("{path} is not valid UTF text: {error}"))
	})?;
	Ok(Some(text))
}

fn parse_xml(xml: &str) -> Result<Node, MarkitError> {
	let mut reader = Reader::from_str(xml);
	reader.config_mut().trim_text(false);
	let mut stack = vec![Node { name: "root".into(), ..Node::default() }];
	loop {
		match reader.read_event() {
			Ok(Event::Start(event)) => {
				stack.push(event_node(&event, &reader)?);
			},
			Ok(Event::Empty(event)) => {
				let node = event_node(&event, &reader)?;
				stack
					.last_mut()
					.expect("root exists")
					.children
					.push(Content::Element(node));
			},
			Ok(Event::End(_)) => {
				if stack.len() == 1 {
					return Err(MarkitError::conversion(FORMAT, "unexpected XML closing tag"));
				}
				let node = stack.pop().expect("length checked");
				stack
					.last_mut()
					.expect("root exists")
					.children
					.push(Content::Element(node));
			},
			Ok(Event::Text(event)) => {
				let decoded = event
					.xml_content(XmlVersion::Implicit1_0)
					.map_err(|error| MarkitError::conversion(FORMAT, error.to_string()))?;
				let text = quick_xml::escape::unescape(&decoded)
					.map_err(|error| MarkitError::conversion(FORMAT, error.to_string()))?;
				stack
					.last_mut()
					.expect("root exists")
					.children
					.push(Content::Text(text.into_owned()));
			},
			Ok(Event::CData(event)) => {
				let text = event
					.decode()
					.map_err(|error| MarkitError::conversion(FORMAT, error.to_string()))?;
				stack
					.last_mut()
					.expect("root exists")
					.children
					.push(Content::Text(text.into_owned()));
			},
			Ok(Event::Eof) => break,
			Ok(_) => {},
			Err(error) => return Err(MarkitError::conversion(FORMAT, error.to_string())),
		}
	}
	if stack.len() != 1 {
		return Err(MarkitError::conversion(FORMAT, "unterminated XML element"));
	}
	Ok(stack.pop().expect("root exists"))
}

fn event_node(
	event: &quick_xml::events::BytesStart<'_>,
	reader: &Reader<&[u8]>,
) -> Result<Node, MarkitError> {
	let name = String::from_utf8_lossy(event.name().as_ref()).into_owned();
	let mut attrs = HashMap::new();
	for attribute in event.attributes().with_checks(false) {
		let attribute =
			attribute.map_err(|error| MarkitError::conversion(FORMAT, error.to_string()))?;
		let key = String::from_utf8_lossy(attribute.key.as_ref()).into_owned();
		let value = attribute
			.decoded_and_normalized_value(XmlVersion::Implicit1_0, reader.decoder())
			.map_err(|error| MarkitError::conversion(FORMAT, error.to_string()))?
			.into_owned();
		attrs.insert(key.clone(), value.clone());
		attrs.entry(local(&key).to_owned()).or_insert(value);
	}
	Ok(Node { name, attrs, children: Vec::new() })
}

fn local(name: &str) -> &str {
	name.rsplit_once(':').map_or(name, |(_, local)| local)
}

fn descendant<'a>(node: &'a Node, name: &str) -> Option<&'a Node> {
	if local(&node.name) == name {
		return Some(node);
	}
	node.elements().find_map(|child| descendant(child, name))
}

fn descendants<'a>(node: &'a Node, name: &'a str, output: &mut Vec<&'a Node>) {
	for child in node.elements() {
		if local(&child.name) == name {
			output.push(child);
		}
		descendants(child, name, output);
	}
}

fn parse_relationships(xml: &str) -> Result<HashMap<String, String>, MarkitError> {
	let root = parse_xml(xml)?;
	let mut nodes = Vec::new();
	descendants(&root, "Relationship", &mut nodes);
	Ok(nodes
		.into_iter()
		.filter_map(|node| Some((node.attr("Id")?.to_owned(), node.attr("Target")?.to_owned())))
		.collect())
}

fn parse_styles(xml: &str) -> Result<HashMap<String, ParagraphStyle>, MarkitError> {
	let root = parse_xml(xml)?;
	let mut nodes = Vec::new();
	descendants(&root, "style", &mut nodes);
	let mut styles = HashMap::new();
	for node in nodes {
		if node.attr("type") != Some("paragraph") {
			continue;
		}
		let Some(id) = node.attr("styleId") else {
			continue;
		};
		let name = node
			.child("name")
			.and_then(|node| node.attr("val"))
			.unwrap_or(id)
			.to_owned();
		styles.insert(id.to_owned(), ParagraphStyle {
			name,
			based_on: node
				.child("basedOn")
				.and_then(|node| node.attr("val"))
				.map(str::to_owned),
			numbering: numbering_reference(node.child("pPr")),
		});
	}
	Ok(styles)
}

fn parse_numbering(xml: &str) -> Result<HashMap<(String, usize), NumberLevel>, MarkitError> {
	let root = parse_xml(xml)?;
	let mut abstracts = HashMap::<String, HashMap<usize, NumberLevel>>::new();
	let mut abstract_nodes = Vec::new();
	descendants(&root, "abstractNum", &mut abstract_nodes);
	for node in abstract_nodes {
		let Some(id) = node.attr("abstractNumId") else {
			continue;
		};
		let mut levels = HashMap::new();
		for level in node.elements().filter(|node| local(&node.name) == "lvl") {
			let index = level
				.attr("ilvl")
				.and_then(|value| value.parse().ok())
				.unwrap_or(0);
			let ordered = level.child("numFmt").and_then(|node| node.attr("val")) != Some("bullet");
			levels.insert(index, NumberLevel { ordered });
		}
		abstracts.insert(id.to_owned(), levels);
	}
	let mut result = HashMap::new();
	let mut number_nodes = Vec::new();
	descendants(&root, "num", &mut number_nodes);
	for node in number_nodes {
		let (Some(id), Some(abstract_id)) = (
			node.attr("numId"),
			node
				.child("abstractNumId")
				.and_then(|node| node.attr("val")),
		) else {
			continue;
		};
		if let Some(levels) = abstracts.get(abstract_id) {
			for (&level, &kind) in levels {
				result.insert((id.to_owned(), level), kind);
			}
		}
	}
	Ok(result)
}

fn numbering_reference(properties: Option<&Node>) -> Option<(String, usize)> {
	let num = properties?.child("numPr")?;
	let id = num.child("numId")?.attr("val")?;
	if id == "0" {
		return None;
	}
	let level = num
		.child("ilvl")
		.and_then(|node| node.attr("val"))
		.and_then(|value| value.parse().ok())
		.unwrap_or(0);
	Some((id.to_owned(), level))
}

fn render_block_children(node: &Node, context: &mut Context, output: &mut Vec<String>) {
	let mut list_block: Option<(bool, usize)> = None;
	let mut previous_list: Option<(bool, usize)> = None;
	for child in node.elements() {
		match local(&child.name) {
			"p" => {
				let list = paragraph_numbering(child, context).map(|(id, level)| {
					let ordered = context
						.numbering
						.get(&(id, level))
						.copied()
						.unwrap_or(NumberLevel { ordered: true })
						.ordered;
					(ordered, level)
				});
				let continues_list = match (list_block, list) {
					(Some((root_kind, root_level)), Some((kind, level))) => {
						level > root_level || (level == root_level && kind == root_kind)
					},
					_ => false,
				};
				if let Some((_, level)) = list {
					if !continues_list {
						context.counters.clear();
						list_block = list;
					} else if previous_list.is_some_and(|(_, previous_level)| previous_level >= level) {
						context.counters.retain(|(_, depth), _| *depth <= level);
					}
				} else {
					context.counters.clear();
					list_block = None;
				}
				if let Some(paragraph) = render_paragraph(child, context, false) {
					if continues_list {
						let previous = output.last_mut().expect("a preceding list block exists");
						previous.push('\n');
						previous.push_str(&paragraph);
					} else {
						output.push(paragraph);
					}
				}
				previous_list = list;
			},
			"tbl" => {
				context.counters.clear();
				let table = render_table(child, context);
				if !table.is_empty() {
					output.push(table);
				}
				list_block = None;
				previous_list = None;
			},
			"sdt" | "sdtContent" | "customXml" => {
				context.counters.clear();
				render_block_children(child, context, output);
				list_block = None;
				previous_list = None;
			},
			_ => {},
		}
	}
}

fn paragraph_numbering(node: &Node, context: &Context) -> Option<(String, usize)> {
	let properties = node.child("pPr");
	let style_id = properties
		.and_then(|node| node.child("pStyle"))
		.and_then(|node| node.attr("val"));
	numbering_reference(properties).or_else(|| {
		resolve_style_numbering(style_id.and_then(|id| context.styles.get(id)), &context.styles)
	})
}

fn resolve_style_numbering(
	style: Option<&ParagraphStyle>,
	styles: &HashMap<String, ParagraphStyle>,
) -> Option<(String, usize)> {
	let mut style = style;
	let mut visited = std::collections::HashSet::new();
	while let Some(current) = style {
		if let Some(numbering) = &current.numbering {
			return Some(numbering.clone());
		}
		let Some(base) = current.based_on.as_deref() else {
			break;
		};
		if !visited.insert(base) {
			break;
		}
		style = styles.get(base);
	}
	None
}

fn render_paragraph(node: &Node, context: &mut Context, in_table: bool) -> Option<String> {
	let properties = node.child("pPr");
	let style_id = properties
		.and_then(|node| node.child("pStyle"))
		.and_then(|node| node.attr("val"));
	let style = style_id.and_then(|id| context.styles.get(id));
	let numbering = paragraph_numbering(node, context);
	let text = render_inline(node, context).trim().to_owned();
	if text.is_empty() && numbering.is_none() {
		return None;
	}
	if let Some((id, level)) = numbering {
		let ordered = context
			.numbering
			.get(&(id.clone(), level))
			.copied()
			.unwrap_or(NumberLevel { ordered: true })
			.ordered;
		let marker = if ordered {
			let counter = context.counters.entry((ordered, level)).or_insert(0);
			*counter += 1;
			format!("{}.", counter)
		} else {
			"-".to_owned()
		};
		context.counters.retain(|(_, depth), _| *depth <= level);
		return Some(format!("{}{} {}", "  ".repeat(level), marker, text));
	}
	context.counters.clear();
	if in_table {
		return Some(text);
	}
	let heading = style
		.and_then(|style| heading_level(&style.name))
		.or_else(|| style_id.and_then(heading_level));
	Some(match heading {
		Some(level) => format!("{} {}", "#".repeat(level), text.replace("\\.", ".")),
		None => text,
	})
}

fn heading_level(name: &str) -> Option<usize> {
	let compact = name
		.chars()
		.filter(|ch| !ch.is_whitespace())
		.flat_map(char::to_lowercase)
		.collect::<String>();
	compact
		.strip_prefix("heading")?
		.parse::<usize>()
		.ok()
		.filter(|level| (1..=6).contains(level))
}

fn render_inline(node: &Node, context: &Context) -> String {
	let mut output = String::new();
	for child in node.elements() {
		let piece = match local(&child.name) {
			"pPr" | "rPr" | "del" => continue,
			"r" => render_run(child),
			"hyperlink" => {
				let label = render_inline(child, context);
				let relationship_target = child
					.attr("id")
					.and_then(|id| context.relationships.get(id).map(String::as_str));
				let anchor_target = child.attr("anchor").map(|anchor| format!("#{anchor}"));
				let target = relationship_target.map(str::to_owned).or(anchor_target);
				if let Some(target) = target {
					format!("[{label}]({})", escape_link_destination(&target))
				} else {
					label
				}
			},
			"fldSimple" | "ins" | "smartTag" | "sdt" | "sdtContent" => render_inline(child, context),
			_ => render_inline(child, context),
		};
		append_inline_piece(&mut output, &piece);
	}
	output
}

fn append_inline_piece(output: &mut String, piece: &str) {
	let piece = if !piece.starts_with("  \n")
		&& output.chars().next_back().is_some_and(char::is_whitespace)
	{
		piece.trim_start_matches(char::is_whitespace)
	} else {
		piece
	};
	output.push_str(piece);
}

fn render_run(run: &Node) -> String {
	let properties = run.child("rPr");
	let bold = property_enabled(properties.and_then(|node| node.child("b")));
	let italic = property_enabled(properties.and_then(|node| node.child("i")));
	let strike = property_enabled(properties.and_then(|node| node.child("strike")))
		|| property_enabled(properties.and_then(|node| node.child("dstrike")));
	let mut text = String::new();
	for child in run.elements() {
		match local(&child.name) {
			"t" | "instrText" => append_text(child, &mut text),
			"tab" => text.push(' '),
			"br" if child.attr("type") != Some("page") => text.push_str("  \n"),
			"noBreakHyphen" => text.push('‑'),
			"softHyphen" => text.push('\u{ad}'),
			"drawing" | "pict" => {
				let alt = descendant(child, "docPr")
					.and_then(|node| node.attr("descr").or_else(|| node.attr("title")))
					.unwrap_or("");
				text.push_str("<!-- image");
				if !alt.is_empty() {
					text.push_str(": ");
					text.push_str(&escape_markdown(alt));
				}
				text.push_str(" -->");
			},
			_ => {},
		}
	}
	let normalized = normalize_run_whitespace(&text);
	let mut rendered = escape_markdown_preserving_breaks(&normalized);
	if strike && !rendered.trim().is_empty() {
		rendered = wrap_inline(rendered, "~~");
	}
	if italic && !rendered.trim().is_empty() {
		rendered = wrap_inline(rendered, "*");
	}
	if bold && !rendered.trim().is_empty() {
		rendered = wrap_inline(rendered, "**");
	}
	rendered
}

fn wrap_inline(value: String, delimiter: &str) -> String {
	let start = value.len() - value.trim_start_matches(char::is_whitespace).len();
	let end = value.trim_end_matches(char::is_whitespace).len();
	if start >= end {
		return value;
	}
	format!("{}{delimiter}{}{delimiter}{}", &value[..start], &value[start..end], &value[end..])
}

fn property_enabled(node: Option<&Node>) -> bool {
	node.is_some_and(|node| {
		!matches!(
			node.attr("val").map(str::to_ascii_lowercase).as_deref(),
			Some("0" | "false" | "off" | "none")
		)
	})
}

fn append_text(node: &Node, output: &mut String) {
	for child in &node.children {
		match child {
			Content::Text(text) => output.push_str(text),
			Content::Element(node) => append_text(node, output),
		}
	}
}

fn render_table(table: &Node, context: &mut Context) -> String {
	let mut rows = Vec::<Vec<String>>::new();
	let mut active_merges = HashSet::new();
	for row in table.elements().filter(|node| local(&node.name) == "tr") {
		let mut cells = Vec::new();
		let mut next_merges = HashSet::new();
		let mut column = 0usize;
		for cell in row.elements().filter(|node| local(&node.name) == "tc") {
			let properties = cell.child("tcPr");
			let span = properties
				.and_then(|node| node.child("gridSpan"))
				.and_then(|node| node.attr("val"))
				.and_then(|value| value.parse::<usize>().ok())
				.unwrap_or(1)
				.max(1);
			let vertical_merge = properties.and_then(|node| node.child("vMerge"));
			let merge = vertical_merge.map(|node| {
				if node.attr("val") == Some("restart") {
					"restart"
				} else {
					"continue"
				}
			});
			if merge == Some("continue") && active_merges.contains(&column) {
				next_merges.insert(column);
				column = column.saturating_add(span);
				continue;
			}

			let mut paragraphs = Vec::new();
			for paragraph in cell.elements().filter(|node| local(&node.name) == "p") {
				if let Some(text) = render_paragraph(paragraph, context, true) {
					paragraphs.push(text);
				}
			}
			cells.push(paragraphs.join(" "));
			if merge == Some("restart") {
				next_merges.insert(column);
			}
			column = column.saturating_add(span);
		}
		rows.push(cells);
		active_merges = next_merges;
	}
	let width = rows.first().map(Vec::len).unwrap_or(0);
	if width == 0 {
		return String::new();
	}
	let mut lines = Vec::with_capacity(rows.len() + 1);
	lines.push(format!("| {} |", rows[0].join(" | ")));
	lines.push(format!(
		"| {} |",
		std::iter::repeat_n("---", width)
			.collect::<Vec<_>>()
			.join(" | ")
	));
	for row in rows.iter().skip(1) {
		lines.push(format!("| {} |", row.join(" | ")));
	}
	lines.join("\n")
}

fn escape_markdown_preserving_breaks(value: &str) -> String {
	value
		.split("  \n")
		.map(escape_markdown)
		.collect::<Vec<_>>()
		.join("  \n")
}

fn normalize_run_whitespace(value: &str) -> String {
	value
		.split("  \n")
		.map(|segment| {
			let mut normalized = String::with_capacity(segment.len());
			let mut whitespace = false;
			for character in segment.chars() {
				if character.is_whitespace() {
					if !whitespace {
						normalized.push(' ');
						whitespace = true;
					}
				} else {
					normalized.push(character);
					whitespace = false;
				}
			}
			normalized
		})
		.collect::<Vec<_>>()
		.join("  \n")
}

fn escape_markdown(value: &str) -> String {
	let mut output = value
		.replace('\\', "\\\\")
		.replace('*', "\\*")
		.replace('_', "\\_")
		.replace('[', "\\[")
		.replace(']', "\\]");
	let leading_whitespace = output.len() - output.trim_start_matches(char::is_whitespace).len();
	let escape_at = {
		let content = &output[leading_whitespace..];
		if let Some(dot) = content.find('.')
			&& !content[..dot].is_empty()
			&& content[..dot]
				.chars()
				.all(|character| character.is_ascii_digit())
			&& content[dot + 1..]
				.chars()
				.next()
				.is_some_and(char::is_whitespace)
		{
			Some(leading_whitespace + dot)
		} else if ["-", "+", ">"]
			.into_iter()
			.any(|marker| marker_followed_by_whitespace(content, marker))
		{
			Some(leading_whitespace)
		} else if content.starts_with('#') {
			let count = content
				.chars()
				.take_while(|character| *character == '#')
				.count();
			((1..=6).contains(&count)
				&& content[count..]
					.chars()
					.next()
					.is_some_and(char::is_whitespace))
			.then_some(leading_whitespace)
		} else {
			None
		}
	};
	if let Some(index) = escape_at {
		output.insert(index, '\\');
	}
	output
}

fn marker_followed_by_whitespace(content: &str, marker: &str) -> bool {
	content
		.strip_prefix(marker)
		.and_then(|rest| rest.chars().next())
		.is_some_and(char::is_whitespace)
}

fn escape_link_destination(value: &str) -> String {
	let mut escaped = String::with_capacity(value.len());
	for character in value.chars() {
		if matches!(character, '(' | ')' | '<' | '>') {
			escaped.push('\\');
		}
		escaped.push(character);
	}
	escaped
}

#[cfg(test)]
mod tests {
	use omp_ar::zip::Writer;

	use super::convert;

	fn docx(parts: &[(&str, &str)]) -> Vec<u8> {
		let mut archive = Writer::new(Vec::new());
		for (name, contents) in parts {
			archive.add_file(name, contents.as_bytes()).unwrap();
		}
		archive.finish().unwrap()
	}

	#[test]
	fn renders_headings_lists_links_breaks_and_tables_in_document_order() {
		let bytes = docx(&[
			(
				"word/styles.xml",
				r#"<w:styles xmlns:w="w"><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/></w:style><w:style w:type="paragraph" w:styleId="BaseList"><w:name w:val="Base List"/><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="7"/></w:numPr></w:pPr></w:style><w:style w:type="paragraph" w:styleId="DerivedList"><w:name w:val="Derived List"/><w:basedOn w:val="BaseList"/></w:style></w:styles>"#,
			),
			(
				"word/numbering.xml",
				r#"<w:numbering xmlns:w="w"><w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/></w:lvl></w:abstractNum><w:num w:numId="7"><w:abstractNumId w:val="0"/></w:num></w:numbering>"#,
			),
			(
				"word/_rels/document.xml.rels",
				r#"<Relationships><Relationship Id="rId1" Target="https://example.com/a b" TargetMode="External"/></Relationships>"#,
			),
			(
				"word/document.xml",
				r#"<w:document xmlns:w="w" xmlns:r="r"><w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>1. Title &amp; * one</w:t></w:r></w:p><w:p><w:r><w:t>- prose</w:t></w:r></w:p><w:p><w:pPr><w:pStyle w:val="DerivedList"/></w:pPr><w:r><w:t>Item</w:t></w:r></w:p><w:p><w:hyperlink r:id="rId1"><w:r><w:t>Example</w:t><w:br/><w:t>site</w:t></w:r></w:hyperlink></w:p><w:p><w:hyperlink w:anchor="bookmark"><w:r><w:t>Jump</w:t></w:r></w:hyperlink></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc></w:tr><w:tr><w:tc><w:p><w:r><w:t>x|y</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>2</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>"#,
			),
		]);
		let markdown = convert(&bytes).unwrap();
		assert_eq!(
			markdown.as_str(),
			"# 1. Title & \\* one\n\n\\- prose\n\n- Item\n\n[Example  \nsite](https://example.com/a \
			 b)\n\n[Jump](#bookmark)\n\n| A | B |\n| --- | --- |\n| x|y | 2 |"
		);
	}

	#[test]
	fn matches_pi_turndown_escape_contract() {
		for (input, expected) in [
			("\\", "\\\\"),
			("*", "\\*"),
			("- dash", "\\- dash"),
			("-dash", "-dash"),
			("+ plus", "\\+ plus"),
			("=== title", "=== title"),
			("# heading", "\\# heading"),
			("`code`", "`code`"),
			("~~~lang", "~~~lang"),
			("[link]", "\\[link\\]"),
			("> quote", "\\> quote"),
			("_word_", "\\_word\\_"),
			("12. item", "12\\. item"),
			("mid-dash", "mid-dash"),
		] {
			assert_eq!(super::escape_markdown(input), expected);
		}
	}
}
