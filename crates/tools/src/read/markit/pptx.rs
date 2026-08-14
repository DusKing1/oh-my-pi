//! PowerPoint Open XML to deterministic Markdown conversion.

use std::collections::HashMap;

use omp_core::Str;
use quick_xml::{Reader, events::Event};

use super::{
	MarkitError,
	ooxml::{Archive, attribute, decode_text, local_name, xml_reader},
};

const FORMAT: &str = "pptx";

/// Converts a PPTX document to Markdown in presentation order.
pub(super) fn convert(bytes: &[u8]) -> Result<Str, MarkitError> {
	convert_inner(bytes).map(Str::from).map_err(failure)
}

fn convert_inner(bytes: &[u8]) -> Result<String, String> {
	let mut archive = Archive::open(bytes)?;
	let presentation = archive
		.read_xml("ppt/presentation.xml")?
		.ok_or_else(|| "Invalid PPTX: missing presentation.xml".to_owned())?;
	let slide_ids = presentation_slide_ids(&presentation)?;
	let relationships = archive
		.read_xml("ppt/_rels/presentation.xml.rels")?
		.map(|xml| parse_relationships(&xml))
		.transpose()?
		.unwrap_or_default();

	let mut slide_paths = slide_ids
		.iter()
		.filter_map(|id| relationships.get(id))
		.map(|target| format!("ppt/{target}"))
		.collect::<Vec<_>>();
	if slide_paths.is_empty() {
		slide_paths = fallback_slide_paths(&archive);
	}

	let mut sections = Vec::with_capacity(slide_paths.len());
	let mut image_count = 0usize;
	for (index, slide_path) in slide_paths.iter().enumerate() {
		let Some(slide_xml) = archive.read_xml(slide_path)? else {
			continue;
		};
		let slide = parse_slide(&slide_xml)?;
		if !slide.has_shape_tree {
			continue;
		}

		let slide_relationships_path =
			format!("{}.rels", slide_path.replace("slides/slide", "slides/_rels/slide"));
		let slide_relationships = archive
			.read_xml(&slide_relationships_path)?
			.map(|xml| parse_relationships(&xml))
			.transpose()?
			.unwrap_or_default();

		let mut lines = vec![format!("<!-- Slide {} -->", index + 1)];
		let mut is_title = true;
		for shape in slide.shapes {
			if shape.text.is_empty() {
				continue;
			}
			if is_title {
				lines.push(format!("# {}", shape.text));
				is_title = false;
			} else {
				lines.push(shape.text);
			}
		}

		for picture in slide.pictures {
			let Some(relationship_id) = picture.relationship_id else {
				continue;
			};
			let Some(target) = slide_relationships.get(&relationship_id) else {
				continue;
			};
			let image_path = if let Some(target) = target.strip_prefix('/') {
				target.to_owned()
			} else {
				format!("ppt/slides/{target}")
			};
			let normalized_path = normalize_image_path(&image_path);
			if !archive.contains(&normalized_path) {
				continue;
			}
			image_count += 1;
			let name = picture
				.name
				.unwrap_or_else(|| format!("image_{image_count}"));
			lines.push(format!("<!-- image: {name} (slide {}) -->", index + 1));
		}

		lines.extend(slide.tables.into_iter().map(render_table));

		let notes_path = slide_path.replace("slides/slide", "notesSlides/notesSlide");
		if let Some(notes_xml) = archive.read_xml(&notes_path)? {
			let notes = parse_notes(&notes_xml)?;
			if !notes.is_empty() {
				lines.push("\n### Notes:".to_owned());
				lines.push(notes.join("\n"));
			}
		}

		sections.push(lines.join("\n"));
	}

	Ok(sections.join("\n\n").trim().to_owned())
}

fn fallback_slide_paths(archive: &Archive<'_>) -> Vec<String> {
	let mut slides = archive
		.paths()
		.filter_map(|name| {
			let number = name
				.strip_prefix("ppt/slides/slide")?
				.strip_suffix(".xml")?
				.parse::<u64>()
				.ok()?;
			Some((number, name.to_owned()))
		})
		.collect::<Vec<_>>();
	slides.sort_unstable_by_key(|(number, _)| *number);
	slides.into_iter().map(|(_, path)| path).collect()
}

fn presentation_slide_ids(xml: &[u8]) -> Result<Vec<String>, String> {
	let mut reader = xml_reader(xml);
	let mut buffer = Vec::new();
	let mut ids = Vec::new();
	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(element) | Event::Empty(element)
				if local_name(element.name().as_ref()) == b"sldId" =>
			{
				if let Some(id) = attribute(&reader, &element, b"id")? {
					ids.push(id);
				}
			},
			Event::Eof => break,
			_ => {},
		}
		buffer.clear();
	}
	Ok(ids)
}

fn parse_relationships(xml: &[u8]) -> Result<HashMap<String, String>, String> {
	let mut reader = xml_reader(xml);
	let mut buffer = Vec::new();
	let mut relationships = HashMap::new();
	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(element) | Event::Empty(element)
				if local_name(element.name().as_ref()) == b"Relationship" =>
			{
				if let (Some(id), Some(target)) =
					(attribute(&reader, &element, b"Id")?, attribute(&reader, &element, b"Target")?)
				{
					relationships.insert(id, target);
				}
			},
			Event::Eof => break,
			_ => {},
		}
		buffer.clear();
	}
	Ok(relationships)
}

#[derive(Default)]
struct ParsedSlide {
	has_shape_tree: bool,
	shapes:         Vec<Shape>,
	pictures:       Vec<Picture>,
	tables:         Vec<Vec<Vec<String>>>,
}

struct Shape {
	text:                    String,
	slide_image_placeholder: bool,
}

struct Picture {
	relationship_id: Option<String>,
	name:            Option<String>,
}

fn parse_slide(xml: &[u8]) -> Result<ParsedSlide, String> {
	let mut reader = xml_reader(xml);
	let mut buffer = Vec::new();
	let mut slide = ParsedSlide::default();
	let mut in_shape_tree = 0usize;

	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(element) if local_name(element.name().as_ref()) == b"spTree" => {
				slide.has_shape_tree = true;
				in_shape_tree += 1;
			},
			Event::End(element) if local_name(element.name().as_ref()) == b"spTree" => {
				in_shape_tree = in_shape_tree.saturating_sub(1);
			},
			Event::Start(element)
				if in_shape_tree > 0 && local_name(element.name().as_ref()) == b"sp" =>
			{
				slide.shapes.push(parse_shape(&mut reader)?);
			},
			Event::Start(element)
				if in_shape_tree > 0 && local_name(element.name().as_ref()) == b"pic" =>
			{
				slide.pictures.push(parse_picture(&mut reader)?);
			},
			Event::Start(element)
				if in_shape_tree > 0 && local_name(element.name().as_ref()) == b"graphicFrame" =>
			{
				if let Some(table) = parse_graphic_frame(&mut reader)? {
					slide.tables.push(table);
				}
			},
			Event::Eof => break,
			_ => {},
		}
		buffer.clear();
	}
	Ok(slide)
}

fn parse_notes(xml: &[u8]) -> Result<Vec<String>, String> {
	let mut reader = xml_reader(xml);
	let mut buffer = Vec::new();
	let mut notes = Vec::new();
	let mut in_shape_tree = 0usize;
	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(element) if local_name(element.name().as_ref()) == b"spTree" => {
				in_shape_tree += 1;
			},
			Event::End(element) if local_name(element.name().as_ref()) == b"spTree" => {
				in_shape_tree = in_shape_tree.saturating_sub(1);
			},
			Event::Start(element)
				if in_shape_tree > 0 && local_name(element.name().as_ref()) == b"sp" =>
			{
				let shape = parse_shape(&mut reader)?;
				if !shape.slide_image_placeholder && !shape.text.is_empty() {
					notes.push(shape.text);
				}
			},
			Event::Eof => break,
			_ => {},
		}
		buffer.clear();
	}
	Ok(notes)
}

fn parse_shape(reader: &mut Reader<&[u8]>) -> Result<Shape, String> {
	let mut buffer = Vec::new();
	let mut depth = 1usize;
	let mut text_body_depth = None;
	let mut paragraph_depth = None;
	let mut run_depth = None;
	let mut paragraph_parts = Vec::new();
	let mut lines = Vec::new();
	let mut slide_image_placeholder = false;

	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(element) => {
				depth += 1;
				let qualified_name = element.name();
				let name = local_name(qualified_name.as_ref());
				if name == b"ph" && attribute(reader, &element, b"type")?.as_deref() == Some("sldImg") {
					slide_image_placeholder = true;
				}
				if name == b"txBody" {
					text_body_depth = Some(depth);
				} else if text_body_depth.is_some() && name == b"p" {
					paragraph_depth = Some(depth);
					paragraph_parts.clear();
				} else if paragraph_depth.is_some() && name == b"r" {
					run_depth = Some(depth);
				} else if run_depth.is_some() && name == b"t" {
					paragraph_parts.push(read_element_text(reader)?);
					depth -= 1;
				}
			},
			Event::Empty(element) => {
				if local_name(element.name().as_ref()) == b"ph"
					&& attribute(reader, &element, b"type")?.as_deref() == Some("sldImg")
				{
					slide_image_placeholder = true;
				}
			},
			Event::End(element) => {
				let qualified_name = element.name();
				let name = local_name(qualified_name.as_ref());
				if name == b"p" && paragraph_depth == Some(depth) {
					if !paragraph_parts.is_empty() {
						lines.push(paragraph_parts.concat());
					}
					paragraph_depth = None;
					run_depth = None;
				} else if name == b"r" && run_depth == Some(depth) {
					run_depth = None;
				} else if name == b"txBody" && text_body_depth == Some(depth) {
					text_body_depth = None;
				}
				if depth == 1 {
					break;
				}
				depth -= 1;
			},
			Event::Eof => return Err("unexpected end of XML inside shape".to_owned()),
			_ => {},
		}
		buffer.clear();
	}

	Ok(Shape { text: lines.join("\n").trim().to_owned(), slide_image_placeholder })
}

fn parse_picture(reader: &mut Reader<&[u8]>) -> Result<Picture, String> {
	let mut buffer = Vec::new();
	let mut depth = 1usize;
	let mut relationship_id = None;
	let mut name = None;
	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(element) => {
				depth += 1;
				let qualified_name = element.name();
				let element_name = local_name(qualified_name.as_ref());
				if element_name == b"blip" {
					relationship_id = attribute(reader, &element, b"embed")?;
				} else if element_name == b"cNvPr" && name.is_none() {
					name = attribute(reader, &element, b"name")?;
				}
			},
			Event::Empty(element) => {
				let qualified_name = element.name();
				let element_name = local_name(qualified_name.as_ref());
				if element_name == b"blip" {
					relationship_id = attribute(reader, &element, b"embed")?;
				} else if element_name == b"cNvPr" && name.is_none() {
					name = attribute(reader, &element, b"name")?;
				}
			},
			Event::End(_) => {
				if depth == 1 {
					break;
				}
				depth -= 1;
			},
			Event::Eof => return Err("unexpected end of XML inside picture".to_owned()),
			_ => {},
		}
		buffer.clear();
	}
	Ok(Picture { relationship_id, name })
}

fn parse_graphic_frame(reader: &mut Reader<&[u8]>) -> Result<Option<Vec<Vec<String>>>, String> {
	let mut buffer = Vec::new();
	let mut depth = 1usize;
	let mut table = None;
	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(element) => {
				depth += 1;
				if local_name(element.name().as_ref()) == b"tbl" {
					table = Some(parse_table(reader)?);
					depth -= 1;
				}
			},
			Event::End(_) => {
				if depth == 1 {
					break;
				}
				depth -= 1;
			},
			Event::Eof => return Err("unexpected end of XML inside graphic frame".to_owned()),
			_ => {},
		}
		buffer.clear();
	}
	Ok(table.filter(|rows| !rows.is_empty()))
}

fn parse_table(reader: &mut Reader<&[u8]>) -> Result<Vec<Vec<String>>, String> {
	let mut buffer = Vec::new();
	let mut depth = 1usize;
	let mut rows = Vec::new();
	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(element) => {
				depth += 1;
				if local_name(element.name().as_ref()) == b"tr" {
					rows.push(parse_table_row(reader)?);
					depth -= 1;
				}
			},
			Event::End(_) => {
				if depth == 1 {
					break;
				}
				depth -= 1;
			},
			Event::Eof => return Err("unexpected end of XML inside table".to_owned()),
			_ => {},
		}
		buffer.clear();
	}
	Ok(rows)
}

fn parse_table_row(reader: &mut Reader<&[u8]>) -> Result<Vec<String>, String> {
	let mut buffer = Vec::new();
	let mut depth = 1usize;
	let mut cells = Vec::new();
	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(element) => {
				depth += 1;
				if local_name(element.name().as_ref()) == b"tc" {
					cells.push(parse_table_cell(reader)?);
					depth -= 1;
				}
			},
			Event::End(_) => {
				if depth == 1 {
					break;
				}
				depth -= 1;
			},
			Event::Eof => return Err("unexpected end of XML inside table row".to_owned()),
			_ => {},
		}
		buffer.clear();
	}
	Ok(cells)
}

fn parse_table_cell(reader: &mut Reader<&[u8]>) -> Result<String, String> {
	let mut buffer = Vec::new();
	let mut depth = 1usize;
	let mut text_body_depth = None;
	let mut run_depth = None;
	let mut parts = Vec::new();
	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(element) => {
				depth += 1;
				let qualified_name = element.name();
				let name = local_name(qualified_name.as_ref());
				if name == b"txBody" {
					text_body_depth = Some(depth);
				} else if text_body_depth.is_some() && name == b"r" {
					run_depth = Some(depth);
				} else if run_depth.is_some() && name == b"t" {
					parts.push(read_element_text(reader)?);
					depth -= 1;
				}
			},
			Event::End(element) => {
				let qualified_name = element.name();
				let name = local_name(qualified_name.as_ref());
				if name == b"r" && run_depth == Some(depth) {
					run_depth = None;
				} else if name == b"txBody" && text_body_depth == Some(depth) {
					text_body_depth = None;
				}
				if depth == 1 {
					break;
				}
				depth -= 1;
			},
			Event::Eof => return Err("unexpected end of XML inside table cell".to_owned()),
			_ => {},
		}
		buffer.clear();
	}
	Ok(parts.join(" "))
}

fn render_table(mut rows: Vec<Vec<String>>) -> String {
	let Some(header) = rows.first().cloned() else {
		return String::new();
	};
	let mut lines = Vec::with_capacity(rows.len() + 1);
	lines.push(format!("| {} |", header.join(" | ")));
	lines.push(format!("| {} |", vec!["---"; header.len()].join(" | ")));
	for row in rows.iter_mut().skip(1) {
		if row.len() < header.len() {
			row.resize(header.len(), String::new());
		}
		lines.push(format!("| {} |", row.join(" | ")));
	}
	lines.join("\n")
}

fn read_element_text(reader: &mut Reader<&[u8]>) -> Result<String, String> {
	let mut buffer = Vec::new();
	let mut text = String::new();
	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Text(part) => text.push_str(&decode_text(&part)?),
			Event::CData(part) => text.push_str(&part.decode().map_err(xml_error)?),
			Event::End(_) => break,
			Event::Eof => return Err("unexpected end of XML inside text element".to_owned()),
			_ => {},
		}
		buffer.clear();
	}
	Ok(text)
}

fn normalize_image_path(path: &str) -> String {
	path
		.split('/')
		.fold(Vec::new(), |mut parts, segment| {
			if segment == ".." {
				parts.pop();
			} else {
				parts.push(segment);
			}
			parts
		})
		.join("/")
}

fn xml_error(error: impl std::fmt::Display) -> String {
	format!("invalid PPTX XML: {error}")
}

fn failure(error: impl Into<Str>) -> MarkitError {
	MarkitError::conversion(FORMAT, error)
}
