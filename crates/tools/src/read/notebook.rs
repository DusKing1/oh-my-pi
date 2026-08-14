//! Jupyter notebook conversion into pi's editable virtual text.

use std::{borrow::Cow, fmt};

use omp_core::Str;
use serde_json::Value;

/// A supported Jupyter notebook cell kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NotebookCellType {
	/// An executable code cell.
	Code,
	/// A Markdown cell.
	Markdown,
	/// A raw cell.
	Raw,
}

impl NotebookCellType {
	const fn marker_name(self) -> &'static str {
		match self {
			Self::Code => "code",
			Self::Markdown => "markdown",
			Self::Raw => "raw",
		}
	}

	fn parse(value: &Value) -> Option<Self> {
		match value.as_str()? {
			"code" => Some(Self::Code),
			"markdown" => Some(Self::Markdown),
			"raw" => Some(Self::Raw),
			_ => None,
		}
	}
}

/// The virtual-text location of one original notebook cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NotebookCellMapping {
	/// Zero-based index of the cell in the notebook JSON.
	pub original_index: usize,
	/// Cell kind encoded in the marker.
	pub cell_type:      NotebookCellType,
	/// One-based line containing the cell marker.
	pub marker_line:    u64,
	/// Inclusive one-based source-line bounds, absent for an empty cell.
	pub source_lines:   Option<(u64, u64)>,
}

/// Editable notebook text and its original-cell locations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderedNotebook {
	/// Virtual cell-marked text consumed by the standard read formatter.
	pub text:  String,
	/// Original cell indices and their locations in `text`.
	pub cells: Vec<NotebookCellMapping>,
}

/// A malformed notebook error with pi-compatible model-facing text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NotebookError(Str);

impl NotebookError {
	fn new(message: impl Into<Str>) -> Self {
		Self(message.into())
	}

	/// Model-facing error text.
	pub fn message(&self) -> &str {
		self.0.as_ref()
	}
}

impl fmt::Display for NotebookError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(self.message())
	}
}

impl std::error::Error for NotebookError {}

struct PreparedCell<'a> {
	cell_type: NotebookCellType,
	source:    Cow<'a, str>,
}

/// Parse notebook JSON bytes and render the editable cell-marked text.
///
/// Notebook and cell metadata, execution counts, and outputs are intentionally
/// not projected into the virtual text. Cell markers retain the original index
/// so notebook-aware edits can preserve those fields when writing JSON back.
pub fn render(bytes: &[u8], display_path: &str) -> Result<RenderedNotebook, NotebookError> {
	let notebook: Value = serde_json::from_slice(bytes)
		.map_err(|_| NotebookError::new(format!("Invalid JSON in notebook: {display_path}")))?;
	let object = notebook.as_object().ok_or_else(|| {
		NotebookError::new(format!("Invalid notebook structure (expected object): {display_path}"))
	})?;
	let cells = object
		.get("cells")
		.and_then(Value::as_array)
		.ok_or_else(|| {
			NotebookError::new(format!(
				"Invalid notebook structure (missing cells array): {display_path}"
			))
		})?;

	let mut prepared = Vec::with_capacity(cells.len());
	let mut text_capacity = 0usize;
	for (index, value) in cells.iter().enumerate() {
		let Some(cell) = value.as_object() else {
			return Err(invalid_cell(index, display_path));
		};
		let Some(cell_type) = cell.get("cell_type").and_then(NotebookCellType::parse) else {
			return Err(invalid_cell(index, display_path));
		};
		let source = match cell.get("source") {
			None => Cow::Borrowed(""),
			Some(Value::String(source)) => Cow::Borrowed(source.as_str()),
			Some(Value::Array(lines)) => {
				let mut length = 0usize;
				for line in lines {
					let Some(line) = line.as_str() else {
						return Err(invalid_cell(index, display_path));
					};
					length = length.saturating_add(line.len());
				}
				let mut source = String::with_capacity(length);
				for line in lines {
					source.push_str(line.as_str().expect("source entries were validated"));
				}
				Cow::Owned(source)
			},
			Some(_) => return Err(invalid_cell(index, display_path)),
		};
		text_capacity = text_capacity
			.saturating_add(24)
			.saturating_add(decimal_digits(index))
			.saturating_add(source.len());
		prepared.push(PreparedCell { cell_type, source });
	}
	text_capacity = text_capacity.saturating_add(prepared.len().saturating_sub(1));

	let mut text = String::with_capacity(text_capacity);
	let mut mappings = Vec::with_capacity(prepared.len());
	let mut marker_line = 1u64;
	for (index, cell) in prepared.into_iter().enumerate() {
		if index != 0 {
			text.push('\n');
		}
		use std::fmt::Write as _;
		write!(text, "# %% [{}] cell:{index}", cell.cell_type.marker_name())
			.expect("writing to a String cannot fail");

		let source_lines = if cell.source.is_empty() {
			None
		} else {
			text.push('\n');
			push_escaped_source(&mut text, &cell.source);
			let count = cell.source.bytes().filter(|byte| *byte == b'\n').count() as u64 + 1;
			Some((marker_line + 1, marker_line.saturating_add(count)))
		};
		mappings.push(NotebookCellMapping {
			original_index: index,
			cell_type: cell.cell_type,
			marker_line,
			source_lines,
		});
		marker_line = marker_line
			.saturating_add(cell.source.bytes().filter(|byte| *byte == b'\n').count() as u64)
			.saturating_add(u64::from(!cell.source.is_empty()))
			.saturating_add(1);
	}

	Ok(RenderedNotebook { text, cells: mappings })
}

fn invalid_cell(index: usize, display_path: &str) -> NotebookError {
	NotebookError::new(format!("Invalid notebook cell {index} in {display_path}"))
}

const fn decimal_digits(mut value: usize) -> usize {
	let mut digits = 1;
	while value >= 10 {
		value /= 10;
		digits += 1;
	}
	digits
}

fn push_escaped_source(output: &mut String, source: &str) {
	if !source.contains("# %%") {
		output.push_str(source);
		return;
	}
	for segment in source.split_inclusive('\n') {
		if let Some(line) = segment.strip_suffix('\n') {
			push_escaped_line(output, line);
			output.push('\n');
		} else {
			push_escaped_line(output, segment);
		}
	}
}

fn push_escaped_line(output: &mut String, line: &str) {
	if is_marker_like_source_line(line) {
		output.push_str("# %%");
		output.push_str(&line[3..]);
	} else {
		output.push_str(line);
	}
}

fn is_marker_like_source_line(line: &str) -> bool {
	let Some(after_prefix) = line.strip_prefix("# ") else {
		return false;
	};
	let percent_count = after_prefix
		.bytes()
		.take_while(|byte| *byte == b'%')
		.count();
	if percent_count < 2 {
		return false;
	}
	let suffix = &after_prefix[percent_count..];
	for marker in [" [code]", " [markdown]", " [raw]"] {
		if suffix == marker {
			return true;
		}
		if let Some(index) = suffix
			.strip_prefix(marker)
			.and_then(|rest| rest.strip_prefix(" cell:"))
		{
			return !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit());
		}
	}
	false
}
