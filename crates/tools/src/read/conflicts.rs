//! Pure detection and rendering of unresolved git conflict markers.

const OURS_PREFIX: &str = "<<<<<<<";
const BASE_PREFIX: &str = "|||||||";
const SEPARATOR: &str = "=======";
const THEIRS_PREFIX: &str = ">>>>>>>";
const PREVIEW_SIDE_LINES: usize = 6;

/// One complete unresolved conflict block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConflictBlock {
	/// One-based line containing the `<<<<<<<` marker.
	pub start_line:     usize,
	/// One-based line containing the `=======` marker.
	pub separator_line: usize,
	/// One-based line containing the `>>>>>>>` marker.
	pub end_line:       usize,
	/// One-based line containing the optional `|||||||` marker.
	pub base_line:      Option<usize>,
	/// Label following the opening marker.
	pub ours_label:     Option<String>,
	/// Label following the optional base marker.
	pub base_label:     Option<String>,
	/// Label following the closing marker.
	pub theirs_label:   Option<String>,
	/// Lines in the ours section, excluding markers.
	pub ours_lines:     Vec<String>,
	/// Lines in the base section for a three-way conflict, excluding markers.
	pub base_lines:     Option<Vec<String>>,
	/// Lines in the theirs section, excluding markers.
	pub theirs_lines:   Vec<String>,
}

/// A conflict block with the stable identifier used by conflict renderers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConflictEntry {
	/// Identifier shown to the model.
	pub id:    usize,
	/// Captured marker block.
	pub block: ConflictBlock,
}

impl ConflictEntry {
	/// Attaches a renderer-visible identifier to a captured block.
	pub const fn new(id: usize, block: ConflictBlock) -> Self {
		Self { id, block }
	}
}

/// Rendered conflict text together with its unresolved-region count.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderedConflicts {
	/// Model-facing text.
	pub text:  String,
	/// Number of complete unresolved regions represented by `text`.
	pub count: usize,
}

/// Options controlling the warning appended to an ordinary file read.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConflictWarningOptions<'a> {
	/// Total conflicts in the file when `entries` only covers a read window.
	pub total_in_file:  Option<usize>,
	/// Display path used in the `:conflicts` hint for a partial window.
	pub display_path:   Option<&'a str>,
	/// Whether the whole-file scan stopped at its byte cap.
	pub scan_truncated: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
	Idle,
	Ours,
	Base,
	Theirs,
}

#[derive(Debug)]
struct PartialConflict {
	start_line:     usize,
	ours_label:     Option<String>,
	ours_lines:     Vec<String>,
	base_line:      Option<usize>,
	base_label:     Option<String>,
	base_lines:     Option<Vec<String>>,
	separator_line: Option<usize>,
	theirs_lines:   Option<Vec<String>>,
}

/// Scans already-collected lines for complete unresolved conflict blocks.
///
/// `first_line_number` is the one-based number of the first input line. Only
/// strict column-zero markers are recognized. Incomplete or malformed blocks
/// are omitted, while a new valid opener abandons any partial preceding block.
pub fn scan_conflict_lines<'a>(
	lines: impl IntoIterator<Item = &'a str>,
	first_line_number: usize,
) -> Vec<ConflictBlock> {
	let mut blocks = Vec::new();
	let mut phase = Phase::Idle;
	let mut partial: Option<PartialConflict> = None;

	for (offset, raw_line) in lines.into_iter().enumerate() {
		let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
		let line_number = first_line_number + offset;

		if let Some(label) = match_marker(line, OURS_PREFIX) {
			partial = Some(PartialConflict {
				start_line:     line_number,
				ours_label:     nonempty_label(label),
				ours_lines:     Vec::new(),
				base_line:      None,
				base_label:     None,
				base_lines:     None,
				separator_line: None,
				theirs_lines:   None,
			});
			phase = Phase::Ours;
			continue;
		}

		let Some(current) = partial.as_mut() else {
			continue;
		};

		if let Some(label) = match_marker(line, BASE_PREFIX) {
			if phase != Phase::Ours {
				partial = None;
				phase = Phase::Idle;
				continue;
			}
			current.base_line = Some(line_number);
			current.base_label = nonempty_label(label);
			current.base_lines = Some(Vec::new());
			phase = Phase::Base;
			continue;
		}

		if line == SEPARATOR {
			if matches!(phase, Phase::Ours | Phase::Base) {
				current.separator_line = Some(line_number);
				current.theirs_lines = Some(Vec::new());
				phase = Phase::Theirs;
			} else {
				partial = None;
				phase = Phase::Idle;
			}
			continue;
		}

		if let Some(label) = match_marker(line, THEIRS_PREFIX) {
			if phase == Phase::Theirs {
				let completed = partial.take().expect("partial checked above");
				if let (Some(separator_line), Some(theirs_lines)) =
					(completed.separator_line, completed.theirs_lines)
				{
					blocks.push(ConflictBlock {
						start_line: completed.start_line,
						separator_line,
						end_line: line_number,
						base_line: completed.base_line,
						ours_label: completed.ours_label,
						base_label: completed.base_label,
						theirs_label: nonempty_label(label),
						ours_lines: completed.ours_lines,
						base_lines: completed.base_lines,
						theirs_lines,
					});
				}
			} else {
				partial = None;
			}
			phase = Phase::Idle;
			continue;
		}

		match phase {
			Phase::Ours => current.ours_lines.push(line.to_owned()),
			Phase::Base => {
				if let Some(lines) = current.base_lines.as_mut() {
					lines.push(line.to_owned());
				}
			},
			Phase::Theirs => {
				if let Some(lines) = current.theirs_lines.as_mut() {
					lines.push(line.to_owned());
				}
			},
			Phase::Idle => {},
		}
	}

	blocks
}

/// Scans a complete UTF-8 text buffer from line one.
pub fn scan_conflicts(input: &str) -> Vec<ConflictBlock> {
	scan_conflict_lines(input.split('\n'), 1)
}

/// Renders the one-line index row used for one conflict region.
pub fn render_conflict_region(entry: &ConflictEntry, id_width: usize) -> String {
	let block = &entry.block;
	let range = if block.start_line == block.end_line {
		format!("L{}", block.start_line)
	} else {
		format!("L{}-{}", block.start_line, block.end_line)
	};
	let kind = if block.base_lines.is_some() {
		"  (3-way)"
	} else {
		""
	};
	format!("#{:>width$}  {range}{kind}", entry.id, width = id_width)
}

/// Formats the `<path>:conflicts` selector result.
pub fn format_conflict_summary(
	entries: &[ConflictEntry],
	display_path: &str,
	scan_truncated: bool,
) -> String {
	let mut lines = Vec::new();
	let total = entries.len();
	let word = if total == 1 { "conflict" } else { "conflicts" };
	let display_path = if display_path.is_empty() {
		"<file>"
	} else {
		display_path
	};
	lines.push(format!("⚠ {total} unresolved {word} in {display_path}"));
	if scan_truncated {
		lines.push(
			"- note: file scan hit the byte cap; additional conflicts may exist beyond the scanned \
			 prefix."
				.to_owned(),
		);
	}
	if let Some(label) = pick_label(entries, |block| block.ours_label.as_deref()) {
		lines.push(format!("- ours = {label}"));
	}
	if let Some(label) = pick_label(entries, |block| block.theirs_label.as_deref()) {
		lines.push(format!("- theirs = {label}"));
	}
	let any_base = entries.iter().any(|entry| entry.block.base_lines.is_some());
	if any_base {
		let label =
			pick_label(entries, |block| block.base_lines.as_ref().and(block.base_label.as_deref()));
		lines.push(format!("- base = {}", label.unwrap_or("(no label)")));
	}
	lines.push(conflict_resolution_guidance(display_path));
	lines.push(String::new());
	let id_width = entries.last().map_or(1, |entry| entry.id.to_string().len());
	lines.extend(
		entries
			.iter()
			.map(|entry| render_conflict_region(entry, id_width)),
	);
	lines.join("\n")
}

/// Scans and formats a one-line-per-region conflict index for `<file>`.
pub fn render_conflicts(input: &str) -> RenderedConflicts {
	render_conflicts_for_path(input, "<file>", false)
}

/// Scans and formats a one-line-per-region conflict index for a display path.
pub fn render_conflicts_for_path(
	input: &str,
	display_path: &str,
	scan_truncated: bool,
) -> RenderedConflicts {
	let entries = numbered_entries(scan_conflicts(input));
	RenderedConflicts {
		text:  format_conflict_summary(&entries, display_path, scan_truncated),
		count: entries.len(),
	}
}

/// Formats the complete warning appended after an ordinary read.
pub fn format_conflict_warning(
	entries: &[ConflictEntry],
	options: ConflictWarningOptions<'_>,
) -> String {
	if entries.is_empty() {
		return String::new();
	}
	let total = options.total_in_file.unwrap_or(entries.len());
	let partial = total > entries.len();
	let word = if total == 1 { "conflict" } else { "conflicts" };
	let guidance_path = options.display_path.unwrap_or("path");
	let mut out = vec![String::new()];
	if partial {
		out.push(format!(
			"⚠ {} of {total} unresolved {word} visible in this window (read \
			 `{guidance_path}:conflicts` for the full list).",
			entries.len()
		));
	} else {
		out.push(format!("⚠ {total} unresolved {word} detected"));
	}
	if options.scan_truncated {
		out.push(
			"- note: file scan hit the byte cap; additional conflicts may exist beyond the scanned \
			 prefix."
				.to_owned(),
		);
	}
	if let Some(label) = pick_label(entries, |block| block.ours_label.as_deref()) {
		out.push(format!("- ours = {label}"));
	}
	if let Some(label) = pick_label(entries, |block| block.theirs_label.as_deref()) {
		out.push(format!("- theirs = {label}"));
	}
	let any_base = entries.iter().any(|entry| entry.block.base_lines.is_some());
	if any_base {
		let label =
			pick_label(entries, |block| block.base_lines.as_ref().and(block.base_label.as_deref()));
		out.push(format!("- base = {}", label.unwrap_or("(no label)")));
	}
	out.push(conflict_resolution_guidance(guidance_path));

	for entry in entries {
		let block = &entry.block;
		let range = if block.start_line == block.end_line {
			format!("L{}", block.start_line)
		} else {
			format!("L{}-{}", block.start_line, block.end_line)
		};
		out.push(String::new());
		out.push(format!("──── #{}  {range} ────", entry.id));
		let base_equals_ours = block
			.base_lines
			.as_ref()
			.is_some_and(|base| base == &block.ours_lines);
		let base_equals_theirs = block
			.base_lines
			.as_ref()
			.is_some_and(|base| base == &block.theirs_lines);
		let theirs_equals_ours = block.theirs_lines == block.ours_lines;

		out.push("<<< ours".to_owned());
		append_body(&mut out, &block.ours_lines);
		if let Some(base) = block.base_lines.as_ref() {
			if base_equals_ours {
				out.push("=== base ≡ ours".to_owned());
			} else if base_equals_theirs {
				out.push("=== base ≡ theirs".to_owned());
			} else {
				out.push("=== base".to_owned());
				append_body(&mut out, base);
			}
		}
		if theirs_equals_ours {
			out.push(">>> theirs ≡ ours".to_owned());
		} else {
			out.push(">>> theirs".to_owned());
			append_body(&mut out, &block.theirs_lines);
		}
	}
	out.join("\n")
}

fn conflict_resolution_guidance(display_path: &str) -> String {
	format!(
		"NOTICE: Read `{display_path}:conflicts` for the conflict index, then read the affected \
		 source ranges to obtain their `[{display_path}#TAG]` header and numbered marker lines. \
		 Resolve each complete marker block with the hashline `edit` tool, using `PUT N.=M:` from \
		 `<<<<<<<` through `>>>>>>>`; preserve the intended side(s), and re-read \
		 `{display_path}:conflicts` to verify."
	)
}

/// Returns the exact warning header for a known unresolved-conflict count.
pub fn conflict_warning(count: usize) -> String {
	if count == 0 {
		return String::new();
	}
	let word = if count == 1 { "conflict" } else { "conflicts" };
	format!("\n⚠ {count} unresolved {word} detected")
}

/// Scans a complete file and formats its ordinary-read warning and count.
pub fn render_conflict_warning(input: &str) -> RenderedConflicts {
	let entries = numbered_entries(scan_conflicts(input));
	RenderedConflicts {
		text:  format_conflict_warning(&entries, ConflictWarningOptions::default()),
		count: entries.len(),
	}
}

fn numbered_entries(blocks: Vec<ConflictBlock>) -> Vec<ConflictEntry> {
	blocks
		.into_iter()
		.enumerate()
		.map(|(index, block)| ConflictEntry::new(index + 1, block))
		.collect()
}

fn match_marker<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
	let rest = line.strip_prefix(prefix)?;
	if rest.is_empty() {
		return Some("");
	}
	rest.strip_prefix(' ')
}

fn nonempty_label(label: &str) -> Option<String> {
	(!label.is_empty()).then(|| label.to_owned())
}

fn pick_label<'a>(
	entries: &'a [ConflictEntry],
	get: impl Fn(&'a ConflictBlock) -> Option<&'a str>,
) -> Option<&'a str> {
	entries
		.iter()
		.filter_map(|entry| get(&entry.block))
		.find(|label| !label.trim().is_empty())
}

fn append_body(out: &mut Vec<String>, section: &[String]) {
	if section.is_empty() {
		out.push("(empty)".to_owned());
		return;
	}
	out.extend(section.iter().take(PREVIEW_SIDE_LINES).cloned());
	let hidden = section.len().saturating_sub(PREVIEW_SIDE_LINES);
	if hidden > 0 {
		let suffix = if hidden == 1 { "" } else { "s" };
		out.push(format!("… ({hidden} more line{suffix})"));
	}
}
