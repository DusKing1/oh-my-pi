//! Path-embedded selectors and read-target resolution primitives.

use std::{
	borrow::Cow,
	collections::HashMap,
	fmt, io,
	path::{Path, PathBuf},
};

use omp_core::Str;

/// One inclusive, one-based line range in a path selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LineRange {
	/// First selected line.
	pub start_line: u64,
	/// Last selected line, or `None` for a range extending to end-of-file.
	pub end_line:   Option<u64>,
}

/// A parsed read selector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParsedSelector {
	/// No recognized read selector was present.
	None,
	/// Return the resource verbatim.
	Raw,
	/// Summarize unresolved conflict regions.
	Conflicts,
	/// Return one or more line ranges, optionally verbatim.
	Lines {
		/// Sorted, merged ranges.
		ranges: Box<[LineRange]>,
		/// Whether numbering and hashline framing are disabled.
		raw:    bool,
	},
}

impl ParsedSelector {
	/// Whether this selector requests verbatim output.
	pub const fn is_raw(&self) -> bool {
		matches!(self, Self::Raw | Self::Lines { raw: true, .. })
	}

	/// Whether this selector contains more than one disjoint line range.
	pub fn is_multi_range(&self) -> bool {
		matches!(self, Self::Lines { ranges, .. } if ranges.len() > 1)
	}

	/// Convert the first range to the offset and optional limit used by paged
	/// readers.
	pub fn offset_limit(&self) -> (Option<u64>, Option<u64>) {
		match self {
			Self::Lines { ranges, .. } => {
				let Some(first) = ranges.first().copied() else {
					return (None, None);
				};
				let limit = first.end_line.map(|end| end - first.start_line + 1);
				(Some(first.start_line), limit)
			},
			_ => (None, None),
		}
	}
}

/// A selector syntax or bounds error suitable for a model-facing tool fault.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectorError(Str);

impl SelectorError {
	fn new(message: impl Into<Str>) -> Self {
		Self(message.into())
	}

	/// Model-facing error text.
	pub fn message(&self) -> &str {
		self.0.as_ref()
	}
}

impl fmt::Display for SelectorError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(self.message())
	}
}

impl std::error::Error for SelectorError {}

/// Parse one `N`, `N-M`, `N-`, `N+K`, `N..M`, or `N..` range chunk.
pub fn parse_line_range_chunk(input: &str) -> Result<Option<LineRange>, SelectorError> {
	let input = input
		.strip_prefix('L')
		.or_else(|| input.strip_prefix('l'))
		.unwrap_or(input);
	let digit_end = input.bytes().take_while(u8::is_ascii_digit).count();
	if digit_end == 0 {
		return Ok(None);
	}
	let start = parse_u64(&input[..digit_end])?;
	if start == 0 {
		return Err(SelectorError::new("Line selector 0 is invalid; lines are 1-indexed. Use :1."));
	}
	let rest = &input[digit_end..];
	if rest.is_empty() {
		return Ok(Some(LineRange { start_line: start, end_line: None }));
	}
	let (separator, rhs) = if let Some(rhs) = rest.strip_prefix("..") {
		('-', rhs)
	} else if let Some(rhs) = rest.strip_prefix('-') {
		('-', rhs)
	} else if let Some(rhs) = rest.strip_prefix('+') {
		('+', rhs)
	} else {
		return Ok(None);
	};
	let rhs = rhs
		.strip_prefix('L')
		.or_else(|| rhs.strip_prefix('l'))
		.unwrap_or(rhs);
	if rhs.bytes().any(|byte| !byte.is_ascii_digit()) {
		return Ok(None);
	}
	if separator == '-' && rhs.is_empty() {
		return Ok(Some(LineRange { start_line: start, end_line: None }));
	}
	if rhs.is_empty() {
		return Ok(None);
	}
	let value = parse_u64(rhs)?;
	if separator == '+' {
		if value == 0 {
			return Err(SelectorError::new(format!("Invalid range {start}+0: count must be >= 1.")));
		}
		let end = start.checked_add(value - 1).ok_or_else(|| {
			SelectorError::new(format!("Invalid range {start}+{value}: count is too large."))
		})?;
		return Ok(Some(LineRange { start_line: start, end_line: Some(end) }));
	}
	if value < start {
		return Err(SelectorError::new(format!(
			"Invalid range {start}-{value}: end must be >= start."
		)));
	}
	Ok(Some(LineRange { start_line: start, end_line: Some(value) }))
}

fn parse_u64(input: &str) -> Result<u64, SelectorError> {
	input
		.parse()
		.map_err(|_| SelectorError::new(format!("Line selector '{input}' is too large.")))
}

/// Parse, sort, and merge a comma-separated list of line ranges.
pub fn parse_line_ranges(input: &str) -> Result<Option<Box<[LineRange]>>, SelectorError> {
	let mut ranges = Vec::new();
	for chunk in input.split(',') {
		let Some(range) = parse_line_range_chunk(chunk)? else {
			return Ok(None);
		};
		ranges.push(range);
	}
	if ranges.is_empty() {
		return Ok(None);
	}
	ranges.sort_unstable_by_key(|range| range.start_line);
	let mut merged: Vec<LineRange> = Vec::with_capacity(ranges.len());
	for current in ranges {
		let Some(last) = merged.last_mut() else {
			merged.push(current);
			continue;
		};
		let Some(last_end) = last.end_line else {
			continue;
		};
		if current.start_line <= last_end.saturating_add(1) {
			match current.end_line {
				None => last.end_line = None,
				Some(end) if end > last_end => last.end_line = Some(end),
				Some(_) => {},
			}
		} else {
			merged.push(current);
		}
	}
	Ok(Some(merged.into_boxed_slice()))
}

/// Extract line ranges from a selector while ignoring raw/conflict display
/// chunks.
pub fn selector_line_ranges(
	selector: Option<&str>,
) -> Result<Option<Box<[LineRange]>>, SelectorError> {
	let Some(selector) = selector else {
		return Ok(None);
	};
	for chunk in selector.split(':') {
		if chunk.eq_ignore_ascii_case("raw") || chunk.eq_ignore_ascii_case("conflicts") {
			continue;
		}
		if let Some(ranges) = parse_line_ranges(chunk)? {
			return Ok(Some(ranges));
		}
	}
	Ok(None)
}

/// Whether a one-based line number falls in any supplied range.
pub fn line_is_in_ranges(line_number: u64, ranges: &[LineRange]) -> bool {
	ranges.iter().any(|range| {
		line_number >= range.start_line && range.end_line.is_none_or(|end| line_number <= end)
	})
}

/// Parse a selector suffix, preserving unrecognized suffixes for archive,
/// SQLite, and URL dispatch.
pub fn parse_selector(input: Option<&str>) -> Result<ParsedSelector, SelectorError> {
	let Some(input) = input.filter(|value| !value.is_empty()) else {
		return Ok(ParsedSelector::None);
	};
	if input.contains(':') {
		let mut chunks = input.split(':');
		let first = chunks.next().unwrap_or_default();
		let second = chunks.next();
		if let Some(second) = second.filter(|_| chunks.next().is_none()) {
			let range = if first.eq_ignore_ascii_case("raw") {
				Some(second)
			} else if second.eq_ignore_ascii_case("raw") {
				Some(first)
			} else {
				None
			};
			if let Some(ranges) = range.map(parse_line_ranges).transpose()?.flatten() {
				return Ok(ParsedSelector::Lines { ranges, raw: true });
			}
		}
		let mut all_read_like = true;
		for chunk in input.split(':') {
			if !selector_chunk_looks_read_like(chunk)? {
				all_read_like = false;
				break;
			}
		}
		if all_read_like {
			return Err(invalid_selector(input));
		}
		return Ok(ParsedSelector::None);
	}
	if input.eq_ignore_ascii_case("raw") {
		return Ok(ParsedSelector::Raw);
	}
	if input.eq_ignore_ascii_case("conflicts") {
		return Ok(ParsedSelector::Conflicts);
	}
	Ok(match parse_line_ranges(input)? {
		Some(ranges) => ParsedSelector::Lines { ranges, raw: false },
		None => ParsedSelector::None,
	})
}

fn selector_chunk_looks_read_like(input: &str) -> Result<bool, SelectorError> {
	if input.eq_ignore_ascii_case("raw") || input.eq_ignore_ascii_case("conflicts") {
		return Ok(true);
	}
	if parse_line_ranges(input)?.is_some() {
		return Ok(true);
	}
	let Some(rest) = input.strip_prefix('-') else {
		return Ok(false);
	};
	let digit_end = rest.bytes().take_while(u8::is_ascii_digit).count();
	if digit_end == 0 {
		return Ok(false);
	}
	let tail = &rest[digit_end..];
	if tail.is_empty() {
		return Ok(true);
	}
	let Some(rhs) = tail.strip_prefix('-').or_else(|| tail.strip_prefix('+')) else {
		return Ok(false);
	};
	Ok(!rhs.is_empty() && rhs.bytes().all(|byte| byte.is_ascii_digit()))
}

fn invalid_selector(input: &str) -> SelectorError {
	SelectorError::new(format!(
		"Invalid selector ':{input}'. Use :N, :N-M, :N+K, :N- (open-ended), a comma-separated list \
		 of ranges, :raw, or a range combined with raw (e.g. :raw:50-100)."
	))
}

/// Borrowed result of separating a path from a recognized trailing selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitPath<'a> {
	/// Resource path without the selector.
	pub path:     &'a str,
	/// Selector text without its leading colon.
	pub selector: Option<&'a str>,
}

/// Peel a strict filesystem-path selector from the end of `raw_path`.
pub fn split_path_and_selector(raw_path: &str) -> SplitPath<'_> {
	let Some(colon) = raw_path.rfind(':').filter(|colon| *colon > 0) else {
		return SplitPath { path: raw_path, selector: None };
	};
	let candidate = &raw_path[colon + 1..];
	if !is_simple_selector(candidate) {
		return SplitPath { path: raw_path, selector: None };
	}
	let mut path = &raw_path[..colon];
	let mut selector = candidate;
	if let Some(inner_colon) = path.rfind(':').filter(|colon| *colon > 0) {
		let inner = &path[inner_colon + 1..];
		let compound = (inner.eq_ignore_ascii_case("raw") && is_range_list(candidate))
			|| (is_range_list(inner) && candidate.eq_ignore_ascii_case("raw"));
		if compound {
			path = &path[..inner_colon];
			selector = &raw_path[inner_colon + 1..];
		}
	}
	SplitPath { path, selector: Some(selector) }
}

fn is_simple_selector(input: &str) -> bool {
	input.eq_ignore_ascii_case("raw")
		|| input.eq_ignore_ascii_case("conflicts")
		|| is_range_list(input)
}

fn is_range_list(input: &str) -> bool {
	!input.is_empty() && input.split(',').all(is_range_chunk_syntax)
}

fn is_range_chunk_syntax(input: &str) -> bool {
	let input = input
		.strip_prefix('L')
		.or_else(|| input.strip_prefix('l'))
		.unwrap_or(input);
	let digit_end = input.bytes().take_while(u8::is_ascii_digit).count();
	if digit_end == 0 {
		return false;
	}
	let rest = &input[digit_end..];
	if rest.is_empty() || rest == "-" || rest == ".." {
		return true;
	}
	let Some(rhs) = rest
		.strip_prefix('-')
		.or_else(|| rest.strip_prefix('+'))
		.or_else(|| rest.strip_prefix(".."))
	else {
		return false;
	};
	let rhs = rhs
		.strip_prefix('L')
		.or_else(|| rhs.strip_prefix('l'))
		.unwrap_or(rhs);
	!rhs.is_empty() && rhs.bytes().all(|byte| byte.is_ascii_digit())
}

/// Result of probing whether an exact literal path exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LiteralPathProbe {
	/// The exact entry exists, including a dangling symlink.
	Exists,
	/// The exact entry definitively does not exist.
	Missing,
	/// An access or transient error makes existence ambiguous.
	Unknown,
}

/// Probe an exact path with symlink metadata so dangling symlinks count as
/// existing.
pub fn probe_literal_path(raw_path: &str, cwd: &Path) -> LiteralPathProbe {
	let expanded = expand_tilde(raw_path, None);
	let resolved = if expanded.is_absolute() {
		expanded
	} else {
		cwd.join(expanded)
	};
	match std::fs::symlink_metadata(resolved) {
		Ok(_) => LiteralPathProbe::Exists,
		Err(error)
			if matches!(error.kind(), io::ErrorKind::NotFound | io::ErrorKind::NotADirectory) =>
		{
			LiteralPathProbe::Missing
		},
		Err(_) => LiteralPathProbe::Unknown,
	}
}

/// Split a selector only when the exact literal path is definitively missing.
pub fn split_path_and_selector_preferring_literal(
	raw_path: &str,
	mut probe: impl FnMut(&str) -> LiteralPathProbe,
) -> SplitPath<'_> {
	let strict = split_path_and_selector(raw_path);
	if strict.selector.is_none() || probe(raw_path) == LiteralPathProbe::Missing {
		strict
	} else {
		SplitPath { path: raw_path, selector: None }
	}
}

/// Expand a leading tilde using `home`, or the process home directory when
/// omitted.
pub fn expand_tilde(path: &str, home: Option<&Path>) -> PathBuf {
	if !path.starts_with('~') {
		return PathBuf::from(path);
	}
	let home = home.map(Path::to_path_buf).or_else(home_dir);
	let Some(mut home) = home else {
		return PathBuf::from(path);
	};
	if path == "~" {
		return home;
	}
	let tail = path
		.strip_prefix("~/")
		.or_else(|| path.strip_prefix("~\\"))
		.unwrap_or_else(|| &path[1..]);
	home.push(tail);
	home
}

fn home_dir() -> Option<PathBuf> {
	std::env::var_os("HOME")
		.or_else(|| std::env::var_os("USERPROFILE"))
		.map(PathBuf::from)
}

/// Split documented semicolon-delimited targets after trimming whitespace and
/// outer double quotes.
pub fn split_semicolon_targets(input: &str) -> Vec<Str> {
	input
		.split(';')
		.map(normalize_path_input)
		.filter(|part| !part.is_empty())
		.map(Str::from)
		.collect()
}

/// Split semicolon targets unless the entire input is an existing or ambiguous
/// literal path.
pub fn split_semicolon_targets_preferring_literal(
	input: &str,
	mut probe: impl FnMut(&str) -> LiteralPathProbe,
) -> Vec<Str> {
	if !input.contains(';') || probe(input) != LiteralPathProbe::Missing {
		return vec![Str::from(normalize_path_input(input))];
	}
	split_semicolon_targets(input)
}

fn normalize_path_input(input: &str) -> &str {
	let trimmed = input.trim();
	if trimmed.len() > 1 && trimmed.starts_with('"') && trimmed.ends_with('"') {
		&trimmed[1..trimmed.len() - 1]
	} else {
		trimmed
	}
}

/// Percent-encode literal URI/member delimiters so they cannot be parsed as
/// selectors or queries.
pub fn percent_encode_member_delimiters(input: &str) -> Cow<'_, str> {
	if !input.bytes().any(|byte| matches!(byte, b':' | b'?' | b'#')) {
		return Cow::Borrowed(input);
	}
	let mut encoded = String::with_capacity(input.len() + 6);
	for character in input.chars() {
		match character {
			':' => encoded.push_str("%3A"),
			'?' => encoded.push_str("%3F"),
			'#' => encoded.push_str("%23"),
			_ => encoded.push(character),
		}
	}
	Cow::Owned(encoded)
}

/// URI-scheme classification used before local path dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UriTarget<'a> {
	/// No leading `scheme://` shape was found.
	LocalOrOther,
	/// An HTTP(S) URL, handled by the web reader.
	Http {
		/// `http` or `https`, retaining the caller's spelling.
		scheme: &'a str,
	},
	/// A syntactically valid but currently unsupported URI scheme.
	Unsupported {
		/// Scheme text before `://`.
		scheme: &'a str,
	},
}

impl UriTarget<'_> {
	/// Return the exact unsupported-target fault text, if this is an unsupported
	/// scheme.
	pub fn unsupported_message(self) -> Option<Str> {
		match self {
			Self::Unsupported { scheme } => Some(unsupported_uri_message(scheme)),
			_ => None,
		}
	}
}

/// Classify `scheme://` paths while keeping HTTP(S) distinct from unsupported
/// internal schemes.
pub fn classify_uri_target(input: &str) -> UriTarget<'_> {
	let Some(separator) = input.find("://") else {
		return UriTarget::LocalOrOther;
	};
	let scheme = &input[..separator];
	if !valid_uri_scheme(scheme) {
		return UriTarget::LocalOrOther;
	}
	if scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https") {
		UriTarget::Http { scheme }
	} else {
		UriTarget::Unsupported { scheme }
	}
}

fn valid_uri_scheme(scheme: &str) -> bool {
	let mut bytes = scheme.bytes();
	matches!(bytes.next(), Some(byte) if byte.is_ascii_alphabetic())
		&& bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'.' | b'-'))
}

/// Build the exact unsupported-target seam message for a URI scheme.
pub fn unsupported_uri_message(scheme: &str) -> Str {
	Str::from(format!("{}:// targets are not supported yet", scheme.to_ascii_lowercase()))
}

/// Split selectors from internal URLs whose resource grammar permits them.
pub fn split_internal_uri_selector(raw_path: &str) -> SplitPath<'_> {
	let UriTarget::Unsupported { scheme } = classify_uri_target(raw_path) else {
		return SplitPath { path: raw_path, selector: None };
	};
	if scheme.eq_ignore_ascii_case("mcp") || !internal_scheme_accepts_selectors(scheme) {
		return SplitPath { path: raw_path, selector: None };
	}
	let scheme_end = scheme.len() + 3;
	if scheme.eq_ignore_ascii_case("ssh") && !raw_path[scheme_end..].contains('/') {
		return SplitPath { path: raw_path, selector: None };
	}
	let mut path = raw_path;
	let mut first_selector_start = None;
	while let Some(colon) = path.rfind(':').filter(|colon| *colon >= scheme_end) {
		if !internal_selector_chunk(&path[colon + 1..]) {
			break;
		}
		first_selector_start = Some(colon + 1);
		path = &raw_path[..colon];
	}
	match first_selector_start {
		Some(start) => SplitPath { path, selector: Some(&raw_path[start..]) },
		None => SplitPath { path: raw_path, selector: None },
	}
}

fn internal_scheme_accepts_selectors(scheme: &str) -> bool {
	[
		"agent", "artifact", "issue", "history", "local", "memory", "omp", "pr", "rule", "security",
		"skill", "ssh", "vault",
	]
	.iter()
	.any(|candidate| scheme.eq_ignore_ascii_case(candidate))
}

fn internal_selector_chunk(input: &str) -> bool {
	is_simple_selector(input) || selector_chunk_looks_read_like(input).unwrap_or(true)
}

/// A unique suffix resolution candidate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuffixMatch {
	/// Absolute filesystem path selected for the read.
	pub absolute_path: PathBuf,
	/// Workspace-relative path shown to the model.
	pub display_path:  Str,
}

/// Select the sole candidate whose complete trailing path matches the authored
/// missing path.
pub fn unique_suffix_match<'a>(
	raw_path: &str,
	cwd: &Path,
	candidates: impl IntoIterator<Item = &'a Path>,
) -> Option<SuffixMatch> {
	let normalized = normalize_suffix(raw_path)?;
	let mut found: Option<SuffixMatch> = None;
	for candidate in candidates {
		let display = candidate.strip_prefix(cwd).unwrap_or(candidate);
		let candidate_normalized = display.to_string_lossy().replace('\\', "/");
		if candidate_normalized != normalized
			&& !candidate_normalized.ends_with(&format!("/{normalized}"))
		{
			continue;
		}
		let next = SuffixMatch {
			absolute_path: if candidate.is_absolute() {
				candidate.to_path_buf()
			} else {
				cwd.join(candidate)
			},
			display_path:  Str::from(candidate_normalized),
		};
		if found
			.as_ref()
			.is_some_and(|prior| prior.absolute_path != next.absolute_path)
		{
			return None;
		}
		found = Some(next);
	}
	found
}

fn normalize_suffix(raw_path: &str) -> Option<String> {
	let normalized = raw_path
		.replace('\\', "/")
		.trim_start_matches("./")
		.trim_end_matches('/')
		.to_owned();
	(!normalized.is_empty()).then_some(normalized)
}

/// Per-execution memo for suffix lookups; `None` records a confirmed miss or
/// ambiguity.
#[derive(Debug, Default)]
pub struct SuffixMatchCache(HashMap<Str, Option<SuffixMatch>>);

impl SuffixMatchCache {
	/// Return a cached lookup when this authored path has already been scanned.
	pub fn get(&self, raw_path: &str) -> Option<&Option<SuffixMatch>> {
		self.0.get(raw_path)
	}

	/// Record and return a suffix lookup result.
	pub fn insert(&mut self, raw_path: impl Into<Str>, result: Option<SuffixMatch>) {
		self.0.insert(raw_path.into(), result);
	}
}

/// Prefix rendered output with pi's exact suffix-resolution notice.
pub fn prepend_suffix_resolution_notice(text: &str, from: &str, resolved: &SuffixMatch) -> String {
	let notice = format!(
		"[Path '{from}' not found; resolved to '{}' via suffix match]",
		resolved.display_path
	);
	if text.is_empty() {
		notice
	} else {
		format!("{notice}\n{text}")
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_and_merges_ranges() {
		let parsed = parse_selector(Some("9-10,5+4,20-")).unwrap();
		assert_eq!(parsed, ParsedSelector::Lines {
			ranges: Box::from([LineRange { start_line: 5, end_line: Some(10) }, LineRange {
				start_line: 20,
				end_line:   None,
			},]),
			raw:    false,
		});
	}

	#[test]
	fn accepts_raw_compounds_and_rejects_selector_like_compounds() {
		assert!(matches!(parse_selector(Some("raw:50-100")).unwrap(), ParsedSelector::Lines {
			raw: true,
			..
		}));
		assert_eq!(
			parse_selector(Some("raw:conflicts"))
				.unwrap_err()
				.to_string(),
			"Invalid selector ':raw:conflicts'. Use :N, :N-M, :N+K, :N- (open-ended), a \
			 comma-separated list of ranges, :raw, or a range combined with raw (e.g. :raw:50-100)."
		);
		assert_eq!(parse_selector(Some("table:key")).unwrap(), ParsedSelector::None);
	}

	#[test]
	fn literal_path_probe_can_override_selector_splitting() {
		let strict = split_path_and_selector("foo:1");
		assert_eq!(strict, SplitPath { path: "foo", selector: Some("1") });
		let literal =
			split_path_and_selector_preferring_literal("foo:1", |_| LiteralPathProbe::Exists);
		assert_eq!(literal, SplitPath { path: "foo:1", selector: None });
	}

	#[test]
	fn classifies_http_separately_and_formats_unsupported_seam() {
		assert!(matches!(classify_uri_target("https://example.com"), UriTarget::Http { .. }));
		let UriTarget::Unsupported { scheme } = classify_uri_target("Skill://name") else {
			panic!("expected unsupported URI");
		};
		assert_eq!(&*unsupported_uri_message(scheme), "skill:// targets are not supported yet");
	}

	#[test]
	fn encodes_only_member_delimiters_without_damaging_unicode() {
		assert_eq!(percent_encode_member_delimiters("café:a?b#c"), "café%3Aa%3Fb%23c");
	}

	#[test]
	fn suffix_selection_requires_uniqueness() {
		let cwd = Path::new("/workspace");
		let one = [Path::new("/workspace/src/foo.rs")];
		let matched = unique_suffix_match("src/foo.rs", cwd, one).unwrap();
		assert_eq!(&*matched.display_path, "src/foo.rs");
		let ambiguous = [Path::new("/workspace/a/src/foo.rs"), Path::new("/workspace/b/src/foo.rs")];
		assert!(unique_suffix_match("src/foo.rs", cwd, ambiguous).is_none());
	}
}
