//! Excel Open XML to deterministic Markdown conversion.

use std::collections::HashMap;

use omp_core::Str;
use quick_xml::events::Event;

use super::{
	MarkitError,
	ooxml::{
		Archive, attribute, decode_reference, decode_text, local_name, render_markdown_table,
		xml_reader,
	},
};

const FORMAT: &str = "xlsx";

#[derive(Debug)]
struct Sheet {
	name:            String,
	relationship_id: String,
}

#[derive(Debug, Default)]
struct Cell {
	kind:   Option<String>,
	value:  String,
	inline: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CellText {
	None,
	Value,
	Inline,
}

/// Converts an XLSX workbook into one Markdown table per non-empty worksheet.
pub(super) fn convert(bytes: &[u8]) -> Result<Str, MarkitError> {
	convert_inner(bytes).map(Str::from).map_err(failure)
}

fn convert_inner(bytes: &[u8]) -> Result<String, String> {
	let mut archive = Archive::open(bytes)?;
	let workbook = archive
		.read_xml("xl/workbook.xml")?
		.ok_or_else(|| "Invalid XLSX: missing workbook.xml".to_owned())?;
	let sheets = parse_workbook(&workbook)?;
	let relationships = archive
		.read_xml("xl/_rels/workbook.xml.rels")?
		.map(|xml| parse_relationships(&xml))
		.transpose()?
		.unwrap_or_default();
	let shared = archive
		.read_xml("xl/sharedStrings.xml")?
		.map(|xml| parse_shared_strings(&xml))
		.transpose()?
		.unwrap_or_default();

	let mut sections = Vec::new();
	for sheet in sheets {
		let Some(target) = relationships.get(&sheet.relationship_id) else {
			continue;
		};
		let path = if let Some(target) = target.strip_prefix('/') {
			target.to_owned()
		} else {
			format!("xl/{target}")
		};
		let Some(xml) = archive.read_xml(&path)? else {
			continue;
		};
		let rows = parse_worksheet(&xml, &shared)?;
		if rows.is_empty() {
			continue;
		}
		let table = render_markdown_table(rows);
		sections.push(format!("## {}\n\n{table}", sheet.name));
	}

	Ok(sections.join("\n\n"))
}

fn parse_workbook(xml: &[u8]) -> Result<Vec<Sheet>, String> {
	let mut reader = xml_reader(xml);
	let mut buffer = Vec::new();
	let mut sheets = Vec::new();
	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(start) | Event::Empty(start)
				if local_name(start.name().as_ref()) == b"sheet" =>
			{
				let name = attribute(&reader, &start, b"name")?.unwrap_or_default();
				let relationship_id = attribute(&reader, &start, b"id")?.unwrap_or_default();
				if !relationship_id.is_empty() {
					sheets.push(Sheet { name, relationship_id });
				}
			},
			Event::Eof => break,
			_ => {},
		}
		buffer.clear();
	}
	Ok(sheets)
}

fn parse_relationships(xml: &[u8]) -> Result<HashMap<String, String>, String> {
	let mut reader = xml_reader(xml);
	let mut buffer = Vec::new();
	let mut relationships = HashMap::new();
	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(start) | Event::Empty(start)
				if local_name(start.name().as_ref()) == b"Relationship" =>
			{
				if let (Some(id), Some(target)) =
					(attribute(&reader, &start, b"Id")?, attribute(&reader, &start, b"Target")?)
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

fn parse_shared_strings(xml: &[u8]) -> Result<Vec<String>, String> {
	let mut reader = xml_reader(xml);
	let mut buffer = Vec::new();
	let mut strings = Vec::new();
	let mut item: Option<String> = None;
	let mut in_text = false;
	let mut phonetic_depth = 0usize;
	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(start) if local_name(start.name().as_ref()) == b"si" => {
				item = Some(String::new());
			},
			Event::End(end) if local_name(end.name().as_ref()) == b"si" => {
				strings.push(item.take().unwrap_or_default());
			},
			Event::Start(start) if local_name(start.name().as_ref()) == b"rPh" && item.is_some() => {
				phonetic_depth += 1;
			},
			Event::End(end) if local_name(end.name().as_ref()) == b"rPh" => {
				phonetic_depth = phonetic_depth.saturating_sub(1);
			},
			Event::Start(start)
				if local_name(start.name().as_ref()) == b"t"
					&& item.is_some()
					&& phonetic_depth == 0 =>
			{
				in_text = true;
			},
			Event::End(end) if local_name(end.name().as_ref()) == b"t" => in_text = false,
			Event::Text(text) if in_text => {
				if let Some(item) = item.as_mut() {
					item.push_str(&decode_text(&text)?);
				}
			},
			Event::GeneralRef(reference) if in_text => {
				if let Some(item) = item.as_mut() {
					item.push_str(&decode_reference(&reference)?);
				}
			},
			Event::CData(text) if in_text => {
				if let Some(item) = item.as_mut() {
					item.push_str(&text.decode().map_err(xml_error)?);
				}
			},
			Event::Eof => break,
			_ => {},
		}
		buffer.clear();
	}
	Ok(strings)
}

fn parse_worksheet(xml: &[u8], shared: &[String]) -> Result<Vec<Vec<String>>, String> {
	let mut reader = xml_reader(xml);
	let mut buffer = Vec::new();
	let mut rows = Vec::new();
	let mut row: Option<Vec<String>> = None;
	let mut cell: Option<Cell> = None;
	let mut cell_text = CellText::None;
	let mut phonetic_depth = 0usize;

	loop {
		match reader.read_event_into(&mut buffer).map_err(xml_error)? {
			Event::Start(start) if local_name(start.name().as_ref()) == b"row" => {
				row = Some(Vec::new());
			},
			Event::Empty(start) if local_name(start.name().as_ref()) == b"row" => {
				rows.push(Vec::new());
			},
			Event::End(end) if local_name(end.name().as_ref()) == b"row" => {
				rows.push(row.take().unwrap_or_default());
			},
			Event::Start(start) if local_name(start.name().as_ref()) == b"c" => {
				cell = Some(Cell { kind: attribute(&reader, &start, b"t")?, ..Cell::default() });
			},
			Event::Empty(start) if local_name(start.name().as_ref()) == b"c" => {
				if let Some(row) = row.as_mut() {
					let cell = Cell { kind: attribute(&reader, &start, b"t")?, ..Cell::default() };
					row.push(cell_value(&cell, shared));
				}
			},
			Event::End(end) if local_name(end.name().as_ref()) == b"c" => {
				if let (Some(row), Some(cell)) = (row.as_mut(), cell.take()) {
					row.push(cell_value(&cell, shared));
				}
				cell_text = CellText::None;
				phonetic_depth = 0;
			},
			Event::Start(start) if local_name(start.name().as_ref()) == b"v" && cell.is_some() => {
				cell_text = CellText::Value;
			},
			Event::End(end) if local_name(end.name().as_ref()) == b"v" => {
				cell_text = CellText::None;
			},
			Event::Start(start) if local_name(start.name().as_ref()) == b"rPh" && cell.is_some() => {
				phonetic_depth += 1;
			},
			Event::End(end) if local_name(end.name().as_ref()) == b"rPh" => {
				phonetic_depth = phonetic_depth.saturating_sub(1);
			},
			Event::Start(start)
				if local_name(start.name().as_ref()) == b"t"
					&& cell.as_ref().and_then(|cell| cell.kind.as_deref()) == Some("inlineStr")
					&& phonetic_depth == 0 =>
			{
				cell_text = CellText::Inline;
			},
			Event::End(end) if local_name(end.name().as_ref()) == b"t" => {
				cell_text = CellText::None;
			},
			Event::Text(text) if cell_text != CellText::None => {
				let text = decode_text(&text)?;
				if let Some(cell) = cell.as_mut() {
					match cell_text {
						CellText::Value => cell.value.push_str(&text),
						CellText::Inline => cell.inline.push_str(&text),
						CellText::None => {},
					}
				}
			},
			Event::GeneralRef(reference) if cell_text != CellText::None => {
				let text = decode_reference(&reference)?;
				if let Some(cell) = cell.as_mut() {
					match cell_text {
						CellText::Value => cell.value.push_str(&text),
						CellText::Inline => cell.inline.push_str(&text),
						CellText::None => {},
					}
				}
			},
			Event::CData(text) if cell_text != CellText::None => {
				let text = text.decode().map_err(xml_error)?;
				if let Some(cell) = cell.as_mut() {
					match cell_text {
						CellText::Value => cell.value.push_str(&text),
						CellText::Inline => cell.inline.push_str(&text),
						CellText::None => {},
					}
				}
			},
			Event::Eof => break,
			_ => {},
		}
		buffer.clear();
	}
	Ok(rows)
}

fn cell_value(cell: &Cell, shared: &[String]) -> String {
	match cell.kind.as_deref() {
		Some("s") => cell
			.value
			.trim()
			.parse::<usize>()
			.ok()
			.and_then(|index| shared.get(index))
			.cloned()
			.unwrap_or_default(),
		Some("inlineStr") => cell.inline.clone(),
		Some("b") => {
			if cell.value.trim() == "1" {
				"TRUE".to_owned()
			} else {
				"FALSE".to_owned()
			}
		},
		_ => cell.value.clone(),
	}
}

fn xml_error(error: impl std::fmt::Display) -> String {
	format!("invalid XML: {error}")
}

fn failure(error: impl Into<Str>) -> MarkitError {
	MarkitError::conversion(FORMAT, error)
}
