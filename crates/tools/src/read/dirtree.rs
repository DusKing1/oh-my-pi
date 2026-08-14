//! Pure rendering for directory reads.
//!
//! The application owns traversal and supplies [`DirEntry`] values. This
//! module only assembles, caps, formats, and slices those values.

use std::{collections::HashMap, fmt::Write as _};

use omp_core::Str;

/// Maximum directory depth rendered below the root.
pub const MAX_DEPTH: usize = 2;
/// Maximum retained children for each non-root directory.
pub const CHILD_LIMIT: usize = 12;

/// Filesystem metadata collected by the application for one directory entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirEntry {
	/// Slash-separated path relative to the listed directory.
	pub relative_path: Str,
	/// Whether this entry is a directory.
	pub is_dir:        bool,
	/// File size in bytes. Directory sizes are not rendered.
	pub size:          u64,
	/// Modification time as milliseconds since the Unix epoch.
	pub modified_ms:   u64,
}

/// Fully formatted result of one directory read.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectoryRender {
	/// Model-facing listing, including selector continuation diagnostics.
	pub text:        Str,
	/// Number of rows in the unsliced listing.
	pub total_lines: usize,
	/// Whether traversal or a per-directory child cap omitted entries.
	pub truncated:   bool,
	/// Resolved path represented by this listing.
	pub root_path:   Str,
}

#[derive(Clone, Copy)]
struct EntryRef<'a> {
	entry: &'a DirEntry,
	name:  &'a str,
	depth: usize,
}

struct RenderedLine {
	label: String,
	size:  Option<String>,
	age:   Option<String>,
}

/// Render application-supplied directory metadata using read's tree layout.
///
/// `now_ms` is supplied by the caller rather than sampled here, keeping this
/// pure and making relative ages deterministic. `scan_truncated` preserves an
/// incomplete traversal reported by the application. `offset` is one-based
/// and `limit` is a row count; both are applied after the complete tree has
/// been aligned and rendered.
pub fn render_directory(
	root_path: impl Into<Str>,
	entries: &[DirEntry],
	scan_truncated: bool,
	now_ms: u64,
	offset: Option<usize>,
	limit: Option<usize>,
) -> DirectoryRender {
	let root_path = root_path.into();
	let mut by_parent: HashMap<&str, Vec<EntryRef<'_>>> = HashMap::new();
	for entry in entries {
		let path = entry.relative_path.as_str().trim_matches('/');
		if path.is_empty() {
			continue;
		}
		let depth = path.bytes().filter(|byte| *byte == b'/').count() + 1;
		if depth > MAX_DEPTH {
			continue;
		}
		let (parent, name) = path.rsplit_once('/').unwrap_or(("", path));
		by_parent
			.entry(parent)
			.or_default()
			.push(EntryRef { entry, name, depth });
	}
	for children in by_parent.values_mut() {
		children.sort_unstable_by(|a, b| {
			b.entry
				.modified_ms
				.cmp(&a.entry.modified_ms)
				.then_with(|| a.name.cmp(b.name))
		});
	}

	let mut rows = vec![RenderedLine { label: ".".into(), size: None, age: None }];
	let mut truncated = scan_truncated;
	render_children("", 0, &by_parent, now_ms, &mut rows, &mut truncated);
	let formatted = format_lines(&rows);
	let base = if rows.len() <= 1 {
		"(empty directory)".to_owned()
	} else {
		formatted
	};
	let all_lines: Vec<&str> = base.split('\n').collect();
	let total_lines = all_lines.len();

	if offset.is_none() && limit.is_none() {
		return DirectoryRender { root_path, text: base.into(), total_lines, truncated };
	}

	let start = offset.unwrap_or(1).saturating_sub(1);
	if start >= total_lines {
		let suggestion = if total_lines == 0 {
			"The listing is empty.".to_owned()
		} else {
			format!("Use :1 to read from the start, or :{total_lines} to read the last line.")
		};
		return DirectoryRender {
			root_path,
			text: format!(
				"Line {} is beyond end of listing ({total_lines} lines total). {suggestion}",
				start + 1
			)
			.into(),
			total_lines,
			truncated,
		};
	}
	let end = limit.map_or(total_lines, |count| start.saturating_add(count).min(total_lines));
	let mut text = all_lines[start..end].join("\n");
	if end < total_lines {
		let remaining = total_lines - end;
		let _ = write!(text, "\n\n[{remaining} more lines in listing. Use :{} to continue]", end + 1);
	}
	DirectoryRender { root_path, text: text.into(), total_lines, truncated }
}

fn render_children<'a>(
	parent: &str,
	parent_depth: usize,
	by_parent: &HashMap<&'a str, Vec<EntryRef<'a>>>,
	now_ms: u64,
	rows: &mut Vec<RenderedLine>,
	truncated: &mut bool,
) {
	let Some(all) = by_parent.get(parent) else {
		return;
	};
	let capped = parent_depth > 0 && all.len() > CHILD_LIMIT;
	let recent_len = if capped { CHILD_LIMIT - 1 } else { all.len() };
	for child in &all[..recent_len] {
		render_entry(*child, parent, by_parent, now_ms, rows, truncated);
	}
	if !capped {
		return;
	}

	*truncated = true;
	let omitted = all.len() - CHILD_LIMIT;
	rows.push(RenderedLine {
		label: format!("{}- … {omitted} more", "  ".repeat(parent_depth + 1)),
		size:  None,
		age:   None,
	});
	if let Some(oldest) = all.last() {
		render_entry(*oldest, parent, by_parent, now_ms, rows, truncated);
	}
}

fn render_entry<'a>(
	node: EntryRef<'a>,
	parent: &str,
	by_parent: &HashMap<&'a str, Vec<EntryRef<'a>>>,
	now_ms: u64,
	rows: &mut Vec<RenderedLine>,
	truncated: &mut bool,
) {
	let suffix = if node.entry.is_dir { "/" } else { "" };
	rows.push(RenderedLine {
		label: format!("{}- {}{suffix}", "  ".repeat(node.depth), node.name),
		size:  (!node.entry.is_dir).then(|| format_bytes(node.entry.size)),
		age:   format_age(now_ms.saturating_sub(node.entry.modified_ms) / 1_000),
	});
	if !node.entry.is_dir || node.depth >= MAX_DEPTH {
		return;
	}
	let child_path = if parent.is_empty() {
		node.name.to_owned()
	} else {
		format!("{parent}/{}", node.name)
	};
	render_children(&child_path, node.depth, by_parent, now_ms, rows, truncated);
}

fn format_lines(rows: &[RenderedLine]) -> String {
	let max_label_len = rows
		.iter()
		.map(|row| row.label.encode_utf16().count())
		.max()
		.unwrap_or(0);
	let mut output = String::new();
	for (index, row) in rows.iter().enumerate() {
		if index > 0 {
			output.push('\n');
		}
		let Some(age) = &row.age else {
			output.push_str(&row.label);
			continue;
		};
		output.push_str(&row.label);
		output.extend(std::iter::repeat_n(' ', max_label_len - row.label.encode_utf16().count() + 2));
		let size = row.size.as_deref().unwrap_or("");
		output.push_str(size);
		output.extend(std::iter::repeat_n(' ', 8usize.saturating_sub(size.len())));
		output.push_str("  ");
		output.push_str(age);
	}
	output
}
fn format_bytes(bytes: u64) -> String {
	const KB: f64 = 1024.0;
	const MB: f64 = 1024.0 * 1024.0;
	const GB: f64 = 1024.0 * 1024.0 * 1024.0;
	match bytes {
		0..=1023 => format!("{bytes}B"),
		1024..=1_048_575 => format!("{:.1}KB", bytes as f64 / KB),
		1_048_576..=1_073_741_823 => format!("{:.1}MB", bytes as f64 / MB),
		_ => format!("{:.1}GB", bytes as f64 / GB),
	}
}

fn format_age(seconds: u64) -> Option<String> {
	if seconds == 0 {
		return None;
	}
	let minutes = seconds / 60;
	let hours = minutes / 60;
	let days = hours / 24;
	let weeks = days / 7;
	let months = days / 30;
	Some(if months > 0 {
		format!("{months}mo ago")
	} else if weeks > 0 {
		format!("{weeks}w ago")
	} else if days > 0 {
		format!("{days}d ago")
	} else if hours > 0 {
		format!("{hours}h ago")
	} else if minutes > 0 {
		format!("{minutes}m ago")
	} else {
		"just now".to_owned()
	})
}
