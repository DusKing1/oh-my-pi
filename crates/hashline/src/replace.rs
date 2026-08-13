//! Pure exact and fuzzy text replacement over immutable byte snapshots.

use bytes::{Bytes, BytesMut};
use omp_core::Str;
use xutf::IntoUnicodeNormalized as _;

use crate::normalize::{LineEnding, detect_line_ending, normalize_to_lf, restore_line_endings};

/// Default similarity threshold for fuzzy replacement matching.
pub const DEFAULT_FUZZY_THRESHOLD: f64 = 0.95;
const SEQUENCE_FUZZY_THRESHOLD: f64 = 0.92;
const FALLBACK_THRESHOLD: f64 = 0.8;
const CONTEXT_FUZZY_THRESHOLD: f64 = 0.8;
const PARTIAL_MATCH_MIN_LENGTH: usize = 6;
const PARTIAL_MATCH_MIN_RATIO: f64 = 0.3;
const OCCURRENCE_PREVIEW_CONTEXT: usize = 5;
const OCCURRENCE_PREVIEW_MAX_LEN: usize = 80;
const MAX_RECORDED_MATCHES: usize = 5;
const DOMINANT_FUZZY_MIN_CONFIDENCE: f64 = 0.97;
const DOMINANT_FUZZY_DELTA: f64 = 0.08;

/// A byte range hidden from exact and fuzzy matching.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExcludedRange {
	/// Inclusive UTF-8 byte start in the normalized text.
	pub start: usize,
	/// Exclusive UTF-8 byte end in the normalized text.
	pub end:   usize,
}

/// Options controlling character-level match selection.
#[derive(Clone, Copy, Debug)]
pub struct MatchOptions<'a> {
	/// Whether a sufficiently confident fuzzy match may be selected.
	pub allow_fuzzy:     bool,
	/// Similarity threshold, defaulting to [`DEFAULT_FUZZY_THRESHOLD`].
	pub threshold:       Option<f64>,
	/// Normalized byte ranges which must remain invisible to matching.
	pub excluded_ranges: &'a [ExcludedRange],
}

impl Default for MatchOptions<'_> {
	fn default() -> Self {
		Self { allow_fuzzy: true, threshold: None, excluded_ranges: &[] }
	}
}

/// A selected or diagnostic fuzzy match.
#[derive(Clone, Debug, PartialEq)]
pub struct FuzzyMatch {
	/// Exact normalized bytes covered by the candidate.
	pub actual_text: Bytes,
	/// UTF-8 byte offset in the normalized content.
	pub start:       usize,
	/// One-based source line.
	pub start_line:  usize,
	/// Similarity in the inclusive range zero through one.
	pub confidence:  f64,
}

impl FuzzyMatch {
	const fn end(&self) -> usize {
		self.start + self.actual_text.len()
	}
}

/// Exact/fuzzy matching result with ambiguity and closest-match diagnostics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MatchOutcome {
	/// Safely selected unique or dominant match.
	pub matched:             Option<FuzzyMatch>,
	/// Highest-scoring candidate even when it could not be selected.
	pub closest:             Option<FuzzyMatch>,
	/// Count of non-overlapping exact occurrences.
	pub occurrences:         Option<usize>,
	/// One-based lines for the first retained exact occurrences.
	pub occurrence_lines:    Vec<usize>,
	/// Context previews for the first retained exact occurrences.
	pub occurrence_previews: Vec<Str>,
	/// Number of fuzzy windows at or above the threshold.
	pub fuzzy_matches:       Option<usize>,
	/// Whether selection used the dominant-match exception.
	pub dominant_fuzzy:      bool,
}

/// Progressive strategy used by [`seek_sequence`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequenceMatchStrategy {
	/// Byte-for-byte line equality.
	Exact,
	/// Equality after trimming trailing whitespace.
	TrimTrailing,
	/// Equality after trimming both ends.
	Trim,
	/// Equality after stripping a common comment marker.
	CommentPrefix,
	/// Equality after Unicode punctuation normalization.
	Unicode,
	/// Normalized pattern lines are prefixes of source lines.
	Prefix,
	/// Normalized pattern lines are significant substrings of source lines.
	Substring,
	/// Line-wise fuzzy similarity.
	Fuzzy,
	/// A uniquely dominant line-wise fuzzy candidate.
	FuzzyDominant,
	/// Character-level fuzzy fallback.
	Character,
}

/// Result of progressive line-sequence matching.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceSearchResult {
	/// Zero-based matching line, if any.
	pub index:         Option<usize>,
	/// Match confidence or best rejected confidence.
	pub confidence:    f64,
	/// Number of candidates at the selected strategy.
	pub match_count:   Option<usize>,
	/// First retained candidate line indices.
	pub match_indices: Vec<usize>,
	/// Strategy which produced the result.
	pub strategy:      Option<SequenceMatchStrategy>,
}

/// Progressive strategy used by [`find_context_line`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContextMatchStrategy {
	/// Byte-for-byte equality.
	Exact,
	/// Equality after trimming.
	Trim,
	/// Equality after Unicode punctuation normalization.
	Unicode,
	/// Normalized context is a source-line prefix.
	Prefix,
	/// Normalized context is a source-line substring.
	Substring,
	/// Character similarity.
	Fuzzy,
}

/// Result of progressive single-context-line matching.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextLineResult {
	/// Zero-based matching line, if any.
	pub index:         Option<usize>,
	/// Match confidence or best rejected confidence.
	pub confidence:    f64,
	/// Number of candidates at the selected strategy.
	pub match_count:   Option<usize>,
	/// First retained candidate line indices.
	pub match_indices: Vec<usize>,
	/// Strategy which produced the result.
	pub strategy:      Option<ContextMatchStrategy>,
}

/// Controls exact/fuzzy replacement over a byte snapshot.
#[derive(Clone, Copy, Debug)]
pub struct ReplaceOptions {
	/// Replace every safe occurrence rather than requiring one occurrence.
	pub replace_all: bool,
	/// Permit fuzzy matching after exact matching fails.
	pub allow_fuzzy: bool,
	/// Similarity threshold for fuzzy matching.
	pub threshold:   f64,
}

impl Default for ReplaceOptions {
	fn default() -> Self {
		Self { replace_all: false, allow_fuzzy: true, threshold: DEFAULT_FUZZY_THRESHOLD }
	}
}

/// A canonical edit in coordinates of the exact input byte snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplaceEdit {
	/// Inclusive byte start in the exact base snapshot.
	pub start:       usize,
	/// Exclusive byte end in the exact base snapshot.
	pub end:         usize,
	/// Exact replacement bytes, including restored line endings.
	pub replacement: Bytes,
}

/// Successful disk-free replacement output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplaceResult {
	/// Sorted, non-overlapping edits in exact base byte coordinates.
	pub edits:       Vec<ReplaceEdit>,
	/// Exact final bytes produced by applying [`Self::edits`] to the base.
	pub final_bytes: Bytes,
	/// Number of replacements.
	pub count:       usize,
}

/// Failure from pure replacement preparation.
#[derive(Clone, Debug, PartialEq, thiserror::Error, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ReplaceError {
	/// The exact base snapshot is not UTF-8 text.
	#[error("base snapshot is not valid UTF-8 and cannot be edited as text")]
	InvalidUtf8,
	/// Empty search text is not a valid replacement operation.
	#[error("old text must not be empty; provide the exact text to replace")]
	EmptyOldText,
	/// Exact matching found no candidate or useful near-match.
	#[error("could not find the exact text; old text must match including whitespace and newlines")]
	ExactNotFound,
	/// Exact matching failed but produced a useful near-match.
	#[error(
		"could not find the exact text; closest match was {similarity_percent:.0}% similar at line \
		 {line}:\n  - {expected_line}\n  + {actual_line}\nfuzzy matching is disabled; copy the \
		 exact text from the file or enable fuzzy matching"
	)]
	ExactMismatch {
		/// Similarity of the closest candidate as a percentage.
		similarity_percent: f64,
		/// One-based line containing the closest candidate.
		line:               usize,
		/// First requested line that differs from the candidate.
		expected_line:      Str,
		/// First candidate line that differs from the request.
		actual_line:        Str,
	},
	/// Fuzzy matching found no candidate worth reporting.
	#[error(
		"could not find a close enough match above the {threshold_percent:.0}% threshold; copy more \
		 exact text from the file"
	)]
	NoCloseMatch {
		/// Effective fuzzy threshold as a percentage.
		threshold_percent: f64,
	},
	/// The closest fuzzy candidate did not meet the configured threshold.
	#[error(
		"closest match was {similarity_percent:.0}% similar at line {line}:\n  - {expected_line}\n  \
		 + {actual_line}\nclosest match was below the {threshold_percent:.0}% threshold; copy more \
		 exact text or lower the threshold"
	)]
	FuzzyBelowThreshold {
		/// Similarity of the closest candidate as a percentage.
		similarity_percent: f64,
		/// Effective fuzzy threshold as a percentage.
		threshold_percent:  f64,
		/// One-based line containing the closest candidate.
		line:               usize,
		/// First requested line that differs from the candidate.
		expected_line:      Str,
		/// First candidate line that differs from the request.
		actual_line:        Str,
	},
	/// A non-all operation found several exact occurrences.
	#[error("found {occurrences} exact occurrences; provide more context or enable replace-all")]
	AmbiguousExact {
		/// Total occurrence count.
		occurrences: usize,
		/// One-based lines for retained occurrences.
		lines:       Vec<usize>,
		/// Context previews for retained occurrences.
		previews:    Vec<Str>,
	},
	/// Fuzzy matching found an ambiguous candidate group.
	#[error(
		"found {matches} high-confidence fuzzy matches; closest was {similarity_percent:.0}% \
		 similar at line {line}:\n  - {expected_line}\n  + {actual_line}\nprovide more unchanged \
		 context"
	)]
	AmbiguousFuzzy {
		/// Number of candidates above the threshold.
		matches:            usize,
		/// Similarity of the closest candidate as a percentage.
		similarity_percent: f64,
		/// One-based line containing the closest candidate.
		line:               usize,
		/// First requested line that differs from the candidate.
		expected_line:      Str,
		/// First candidate line that differs from the request.
		actual_line:        Str,
	},
	/// The requested replacement would leave the bytes unchanged.
	#[error("replacement resulted in no changes; choose replacement text that differs")]
	NoChanges,
}

impl ReplaceError {
	/// Returns the stable machine-readable diagnostic code.
	#[must_use]
	pub fn code(&self) -> &'static str {
		self.into()
	}
}

fn first_different_line<'old, 'actual>(
	old: &'old str,
	actual: &'actual str,
) -> (&'old str, &'actual str) {
	let mut old_lines = old.split('\n');
	let mut actual_lines = actual.split('\n');
	loop {
		match (old_lines.next(), actual_lines.next()) {
			(None, None) => {
				return (old.split('\n').next().unwrap_or(""), actual.split('\n').next().unwrap_or(""));
			},
			(old_line, actual_line) => {
				let old_line = old_line.unwrap_or("");
				let actual_line = actual_line.unwrap_or("");
				if old_line != actual_line {
					return (old_line, actual_line);
				}
			},
		}
	}
}

fn no_match_error(
	outcome: &MatchOutcome,
	search_text: &str,
	options: ReplaceOptions,
) -> ReplaceError {
	let Some(closest) = &outcome.closest else {
		return if options.allow_fuzzy {
			ReplaceError::NoCloseMatch { threshold_percent: options.threshold * 100.0 }
		} else {
			ReplaceError::ExactNotFound
		};
	};
	let actual_text =
		std::str::from_utf8(&closest.actual_text).expect("match candidates are valid UTF-8");
	let (expected_line, actual_line) = first_different_line(search_text, actual_text);
	let similarity_percent = closest.confidence * 100.0;
	if !options.allow_fuzzy {
		ReplaceError::ExactMismatch {
			similarity_percent,
			line: closest.start_line,
			expected_line: expected_line.into(),
			actual_line: actual_line.into(),
		}
	} else if let Some(matches) = outcome.fuzzy_matches.filter(|count| *count > 1) {
		ReplaceError::AmbiguousFuzzy {
			matches,
			similarity_percent,
			line: closest.start_line,
			expected_line: expected_line.into(),
			actual_line: actual_line.into(),
		}
	} else {
		ReplaceError::FuzzyBelowThreshold {
			similarity_percent,
			threshold_percent: options.threshold * 100.0,
			line: closest.start_line,
			expected_line: expected_line.into(),
			actual_line: actual_line.into(),
		}
	}
}

#[derive(Clone, Debug)]
struct IndexedMatches {
	first:   Option<usize>,
	count:   usize,
	indices: Vec<usize>,
}

fn collect_indexed_matches(
	mut from: usize,
	to: usize,
	mut predicate: impl FnMut(usize) -> bool,
) -> IndexedMatches {
	let mut matches = IndexedMatches { first: None, count: 0, indices: Vec::new() };
	if from > to {
		return matches;
	}
	while from <= to {
		if predicate(from) {
			matches.first.get_or_insert(from);
			matches.count += 1;
			if matches.indices.len() < MAX_RECORDED_MATCHES {
				matches.indices.push(from);
			}
		}
		from += 1;
	}
	matches
}

fn overlaps_excluded(start: usize, end: usize, excluded: &[ExcludedRange]) -> bool {
	excluded
		.iter()
		.any(|range| start < range.end && end > range.start)
}

fn format_preview_window(lines: &[&str], center: usize) -> Str {
	let start = center.saturating_sub(OCCURRENCE_PREVIEW_CONTEXT);
	let end = lines.len().min(center + OCCURRENCE_PREVIEW_CONTEXT + 1);
	let mut result = String::new();
	for (offset, line) in lines[start..end].iter().enumerate() {
		if !result.is_empty() {
			result.push('\n');
		}
		let truncated = truncate_utf16(line, OCCURRENCE_PREVIEW_MAX_LEN);
		use std::fmt::Write as _;
		let _ = write!(result, "  {} | {truncated}", start + offset + 1);
	}
	Str::new(result)
}

fn truncate_utf16(text: &str, max_units: usize) -> String {
	if utf16_len(text) <= max_units {
		return text.to_owned();
	}
	let kept_units = max_units.saturating_sub(1);
	let mut used = 0;
	let mut end = 0;
	for (index, ch) in text.char_indices() {
		let width = ch.len_utf16();
		if used + width > kept_units {
			break;
		}
		used += width;
		end = index + ch.len_utf8();
	}
	let mut result = text[..end].to_owned();
	result.push('…');
	result
}

fn exact_match_outcome(
	content: &str,
	target: &str,
	excluded: &[ExcludedRange],
) -> Option<MatchOutcome> {
	let mut first = None;
	let mut occurrences = 0;
	let mut recorded = Vec::new();
	let mut search_start = 0;
	while search_start <= content.len().saturating_sub(target.len()) {
		let Some(relative) = content[search_start..].find(target) else {
			break;
		};
		let index = search_start + relative;
		let end = index + target.len();
		if !overlaps_excluded(index, end, excluded) {
			first.get_or_insert(index);
			occurrences += 1;
			if recorded.len() < MAX_RECORDED_MATCHES {
				recorded.push(index);
			}
		}
		search_start = end;
	}
	let first = first?;
	if occurrences > 1 {
		let lines = content.split('\n').collect::<Vec<_>>();
		let occurrence_lines = recorded
			.iter()
			.map(|index| {
				content[..*index]
					.bytes()
					.filter(|byte| *byte == b'\n')
					.count() + 1
			})
			.collect::<Vec<_>>();
		let occurrence_previews = occurrence_lines
			.iter()
			.map(|line| format_preview_window(&lines, line - 1))
			.collect();
		return Some(MatchOutcome {
			occurrences: Some(occurrences),
			occurrence_lines,
			occurrence_previews,
			..MatchOutcome::default()
		});
	}
	let start_line = content[..first]
		.bytes()
		.filter(|byte| *byte == b'\n')
		.count()
		+ 1;
	Some(MatchOutcome {
		matched: Some(FuzzyMatch {
			actual_text: Bytes::copy_from_slice(target.as_bytes()),
			start: first,
			start_line,
			confidence: 1.0,
		}),
		..MatchOutcome::default()
	})
}

/// Computes Levenshtein distance using JavaScript UTF-16 code units.
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
	if a == b {
		return 0;
	}
	let a = a.encode_utf16().collect::<Vec<_>>();
	let b = b.encode_utf16().collect::<Vec<_>>();
	if a.is_empty() {
		return b.len();
	}
	if b.is_empty() {
		return a.len();
	}
	let mut previous = (0..=b.len()).collect::<Vec<_>>();
	let mut current = vec![0; b.len() + 1];
	for (i, a_code) in a.iter().enumerate() {
		current[0] = i + 1;
		for (j, b_code) in b.iter().enumerate() {
			let cost = usize::from(a_code != b_code);
			current[j + 1] = (previous[j + 1] + 1)
				.min(current[j] + 1)
				.min(previous[j] + cost);
		}
		std::mem::swap(&mut previous, &mut current);
	}
	previous[b.len()]
}

/// Computes zero-to-one similarity using JavaScript UTF-16 code-unit lengths.
pub fn similarity(a: &str, b: &str) -> f64 {
	let max_len = utf16_len(a).max(utf16_len(b));
	if max_len == 0 {
		return 1.0;
	}
	1.0 - levenshtein_distance(a, b) as f64 / max_len as f64
}

fn utf16_len(text: &str) -> usize {
	text.encode_utf16().count()
}

/// Counts leading ASCII spaces and tabs.
pub fn count_leading_whitespace(line: &str) -> usize {
	line
		.bytes()
		.take_while(|byte| matches!(byte, b' ' | b'\t'))
		.count()
}

/// Normalizes punctuation and odd spacing for permissive comparisons.
pub fn normalize_unicode(text: &str) -> String {
	let mut normalized = String::with_capacity(text.len());
	for ch in text.trim().chars() {
		match ch {
			'\u{2010}'..='\u{2015}' | '\u{2212}' => normalized.push('-'),
			'\u{2018}'..='\u{201b}' => normalized.push('\''),
			'\u{201c}'..='\u{201f}' => normalized.push('"'),
			'\u{00a0}' | '\u{2002}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}' => {
				normalized.push(' ');
			},
			'\u{2260}' => normalized.push_str("!="),
			'\u{00bd}' => normalized.push_str("1/2"),
			'\u{200b}'..='\u{200d}' | '\u{feff}' => {},
			_ => normalized.push(ch),
		}
	}
	normalized.into_nfc()
}

/// Normalizes a line by trimming, collapsing spaces/tabs, and folding
/// punctuation.
pub fn normalize_for_fuzzy(line: &str) -> String {
	let line = line.trim();
	if line.is_empty() {
		return String::new();
	}
	let mut result = String::with_capacity(line.len());
	let mut pending_space = false;
	for ch in line.chars() {
		let folded = match ch {
			'“' | '”' | '„' | '‟' | '«' | '»' => '"',
			'‘' | '’' | '‚' | '‛' | '`' | '´' => '\'',
			'‐' | '‑' | '‒' | '–' | '—' | '−' => '-',
			' ' | '\t' => {
				pending_space = true;
				continue;
			},
			_ => ch,
		};
		if pending_space && !result.is_empty() {
			result.push(' ');
		}
		pending_space = false;
		result.push(folded);
	}
	result
}

fn compute_relative_indent_depths(lines: &[&str]) -> Vec<usize> {
	let indents = lines
		.iter()
		.map(|line| count_leading_whitespace(line))
		.collect::<Vec<_>>();
	let non_empty = lines
		.iter()
		.zip(&indents)
		.filter_map(|(line, indent)| (!line.trim().is_empty()).then_some(*indent))
		.collect::<Vec<_>>();
	let minimum = non_empty.iter().copied().min().unwrap_or(0);
	let unit = non_empty
		.iter()
		.filter_map(|indent| indent.checked_sub(minimum).filter(|step| *step > 0))
		.min()
		.unwrap_or(1);
	lines
		.iter()
		.zip(indents)
		.map(|(line, indent)| {
			if line.trim().is_empty() || unit == 0 {
				0
			} else {
				((indent - minimum) as f64 / unit as f64).round() as usize
			}
		})
		.collect()
}

fn normalize_lines(lines: &[&str], include_depth: bool) -> Vec<String> {
	let depths = include_depth.then(|| compute_relative_indent_depths(lines));
	lines
		.iter()
		.enumerate()
		.map(|(index, line)| {
			let mut result = match &depths {
				Some(depths) => format!("{}|", depths[index]),
				None => "|".to_owned(),
			};
			let trimmed = line.trim();
			if !trimmed.is_empty() {
				result.push_str(&normalize_for_fuzzy(trimmed));
			}
			result
		})
		.collect()
}

fn line_offsets(lines: &[&str]) -> Vec<usize> {
	let mut result = Vec::with_capacity(lines.len());
	let mut offset = 0;
	for (index, line) in lines.iter().enumerate() {
		result.push(offset);
		offset += line.len() + usize::from(index + 1 < lines.len());
	}
	result
}

#[derive(Debug)]
struct BestFuzzyMatch {
	best:              Option<FuzzyMatch>,
	above_threshold:   usize,
	second_best_score: f64,
}

fn best_fuzzy_core(
	content_lines: &[&str],
	target_lines: &[&str],
	offsets: &[usize],
	threshold: f64,
	include_depth: bool,
	excluded: &[ExcludedRange],
) -> BestFuzzyMatch {
	let target_normalized = normalize_lines(target_lines, include_depth);
	let mut best = None;
	let mut best_score = -1.0;
	let mut second_best_score = -1.0;
	let mut above_threshold = 0;
	for start in 0..=content_lines.len() - target_lines.len() {
		let start_index = offsets[start];
		let end_line = start + target_lines.len() - 1;
		let end_index = (offsets[end_line] + content_lines[end_line].len()).max(start_index + 1);
		if overlaps_excluded(start_index, end_index, excluded) {
			continue;
		}
		let window = &content_lines[start..start + target_lines.len()];
		let normalized = normalize_lines(window, include_depth);
		let score = target_normalized
			.iter()
			.zip(&normalized)
			.map(|(target, actual)| similarity(target, actual))
			.sum::<f64>()
			/ target_lines.len() as f64;
		if score >= threshold {
			above_threshold += 1;
		}
		if score > best_score {
			second_best_score = best_score;
			best_score = score;
			best = Some(FuzzyMatch {
				actual_text: Bytes::copy_from_slice(window.join("\n").as_bytes()),
				start:       start_index,
				start_line:  start + 1,
				confidence:  score,
			});
		} else if score > second_best_score {
			second_best_score = score;
		}
	}
	BestFuzzyMatch { best, above_threshold, second_best_score }
}

fn best_fuzzy_match(
	content: &str,
	target: &str,
	threshold: f64,
	excluded: &[ExcludedRange],
) -> BestFuzzyMatch {
	let content_lines = content.split('\n').collect::<Vec<_>>();
	let target_lines = target.split('\n').collect::<Vec<_>>();
	if target.is_empty() || target_lines.len() > content_lines.len() {
		return BestFuzzyMatch {
			best:              None,
			above_threshold:   0,
			second_best_score: 0.0,
		};
	}
	let offsets = line_offsets(&content_lines);
	let mut result =
		best_fuzzy_core(&content_lines, &target_lines, &offsets, threshold, true, excluded);
	if result
		.best
		.as_ref()
		.is_some_and(|best| best.confidence < threshold && best.confidence >= FALLBACK_THRESHOLD)
	{
		let without_depth =
			best_fuzzy_core(&content_lines, &target_lines, &offsets, threshold, false, excluded);
		if without_depth
			.best
			.as_ref()
			.map_or(-1.0, |best| best.confidence)
			> result.best.as_ref().map_or(-1.0, |best| best.confidence)
		{
			result = without_depth;
		}
	}
	result
}

/// Finds an exact, unique fuzzy, or uniquely dominant match in normalized LF
/// text.
pub fn find_match(content: &str, target: &str, options: MatchOptions<'_>) -> MatchOutcome {
	if target.is_empty() {
		return MatchOutcome::default();
	}
	if let Some(exact) = exact_match_outcome(content, target, options.excluded_ranges) {
		return exact;
	}
	let threshold = options.threshold.unwrap_or(DEFAULT_FUZZY_THRESHOLD);
	let fuzzy = best_fuzzy_match(content, target, threshold, options.excluded_ranges);
	let Some(best) = fuzzy.best else {
		return MatchOutcome::default();
	};
	if options.allow_fuzzy && best.confidence >= threshold {
		if fuzzy.above_threshold == 1 {
			return MatchOutcome {
				matched: Some(best.clone()),
				closest: Some(best),
				..MatchOutcome::default()
			};
		}
		if fuzzy.above_threshold > 1
			&& best.confidence >= DOMINANT_FUZZY_MIN_CONFIDENCE
			&& best.confidence - fuzzy.second_best_score >= DOMINANT_FUZZY_DELTA
		{
			return MatchOutcome {
				matched: Some(best.clone()),
				closest: Some(best),
				fuzzy_matches: Some(fuzzy.above_threshold),
				dominant_fuzzy: true,
				..MatchOutcome::default()
			};
		}
	}
	MatchOutcome {
		closest: Some(best),
		fuzzy_matches: Some(fuzzy.above_threshold),
		..MatchOutcome::default()
	}
}

fn matches_at(
	lines: &[&str],
	pattern: &[&str],
	index: usize,
	compare: impl Fn(&str, &str) -> bool,
) -> bool {
	pattern
		.iter()
		.enumerate()
		.all(|(offset, expected)| compare(lines[index + offset], expected))
}

fn fuzzy_score_at(lines: &[String], pattern: &[String], index: usize, minimum: f64) -> f64 {
	let count = pattern.len();
	let mut total = 0.0;
	for offset in 0..count {
		let line = &lines[index + offset];
		let expected = &pattern[offset];
		if line == expected {
			total += 1.0;
			continue;
		}
		let remaining = count - offset - 1;
		let max_len = utf16_len(line).max(utf16_len(expected));
		let upper_bound = if max_len == 0 {
			1.0
		} else {
			1.0 - utf16_len(line).abs_diff(utf16_len(expected)) as f64 / max_len as f64
		};
		if (total + upper_bound + remaining as f64) / (count as f64) < minimum {
			return total / count as f64;
		}
		if upper_bound > 0.0 {
			total += similarity(line, expected);
		}
		if (total + remaining as f64) / (count as f64) < minimum {
			return total / count as f64;
		}
	}
	total / count as f64
}

fn normalized_starts_with(line: &str, pattern: &str) -> bool {
	if pattern.is_empty() {
		line.is_empty()
	} else {
		line.starts_with(pattern)
	}
}

fn normalized_includes(line: &str, pattern: &str) -> bool {
	if pattern.is_empty() {
		return line.is_empty();
	}
	let pattern_len = utf16_len(pattern);
	pattern_len >= PARTIAL_MATCH_MIN_LENGTH
		&& line.contains(pattern)
		&& pattern_len as f64 / utf16_len(line).max(1) as f64 >= PARTIAL_MATCH_MIN_RATIO
}

fn strip_comment_prefix(line: &str) -> &str {
	let trimmed = line.trim_start();
	let stripped = if let Some(rest) = trimmed.strip_prefix("/*") {
		rest
	} else if let Some(rest) = trimmed.strip_prefix("*/") {
		rest
	} else if let Some(rest) = trimmed.strip_prefix("//") {
		rest
	} else if let Some(rest) = trimmed.strip_prefix('*') {
		rest
	} else if let Some(rest) = trimmed.strip_prefix('#') {
		rest
	} else if let Some(rest) = trimmed.strip_prefix(';') {
		rest
	} else if trimmed.starts_with("/ ") {
		&trimmed[1..]
	} else {
		trimmed
	};
	stripped.trim_start()
}

const fn sequence_result(
	index: usize,
	confidence: f64,
	strategy: SequenceMatchStrategy,
) -> SequenceSearchResult {
	SequenceSearchResult {
		index: Some(index),
		confidence,
		match_count: None,
		match_indices: Vec::new(),
		strategy: Some(strategy),
	}
}

fn sequence_ambiguous(
	matches: IndexedMatches,
	confidence: f64,
	strategy: SequenceMatchStrategy,
) -> Option<SequenceSearchResult> {
	Some(SequenceSearchResult {
		index: Some(matches.first?),
		confidence,
		match_count: Some(matches.count),
		match_indices: matches.indices,
		strategy: Some(strategy),
	})
}

type SequencePass = (fn(&str, &str) -> bool, f64, SequenceMatchStrategy);

fn sequence_exact_passes(
	lines: &[&str],
	pattern: &[&str],
	from: usize,
	to: usize,
	allow_fuzzy: bool,
	lines_normalized: &[String],
	pattern_normalized: &[String],
) -> Option<SequenceSearchResult> {
	let passes: &[SequencePass] = &[
		(|a, b| a == b, 1.0, SequenceMatchStrategy::Exact),
		(|a, b| a.trim_end() == b.trim_end(), 0.99, SequenceMatchStrategy::TrimTrailing),
		(|a, b| a.trim() == b.trim(), 0.98, SequenceMatchStrategy::Trim),
		(
			|a, b| strip_comment_prefix(a) == strip_comment_prefix(b),
			0.975,
			SequenceMatchStrategy::CommentPrefix,
		),
		(|a, b| normalize_unicode(a) == normalize_unicode(b), 0.97, SequenceMatchStrategy::Unicode),
	];
	for (compare, confidence, strategy) in passes {
		let matches =
			collect_indexed_matches(from, to, |index| matches_at(lines, pattern, index, *compare));
		if let Some(index) = matches.first {
			return Some(sequence_result(index, *confidence, *strategy));
		}
	}
	if !allow_fuzzy {
		return None;
	}
	let partial: &[SequencePass] = &[
		(normalized_starts_with, 0.965, SequenceMatchStrategy::Prefix),
		(normalized_includes, 0.94, SequenceMatchStrategy::Substring),
	];
	for (compare, confidence, strategy) in partial {
		let matches = collect_indexed_matches(from, to, |index| {
			pattern_normalized
				.iter()
				.enumerate()
				.all(|(offset, expected)| compare(&lines_normalized[index + offset], expected))
		});
		if matches.first.is_some() {
			return sequence_ambiguous(matches, *confidence, *strategy);
		}
	}
	None
}

/// Finds a line sequence with exact, normalized, partial, and fuzzy strategies.
pub fn seek_sequence(
	lines: &[&str],
	pattern: &[&str],
	start: usize,
	eof: bool,
	allow_fuzzy: bool,
) -> SequenceSearchResult {
	if pattern.is_empty() {
		return sequence_result(start, 1.0, SequenceMatchStrategy::Exact);
	}
	if pattern.len() > lines.len() {
		return SequenceSearchResult {
			index:         None,
			confidence:    0.0,
			match_count:   None,
			match_indices: Vec::new(),
			strategy:      None,
		};
	}
	let maximum = lines.len() - pattern.len();
	let search_start = if eof { maximum } else { start };
	let lines_normalized = lines
		.iter()
		.map(|line| normalize_for_fuzzy(line))
		.collect::<Vec<_>>();
	let pattern_normalized = pattern
		.iter()
		.map(|line| normalize_for_fuzzy(line))
		.collect::<Vec<_>>();
	if let Some(result) = sequence_exact_passes(
		lines,
		pattern,
		search_start,
		maximum,
		allow_fuzzy,
		&lines_normalized,
		&pattern_normalized,
	) {
		return result;
	}
	if eof
		&& search_start > start
		&& let Some(result) = sequence_exact_passes(
			lines,
			pattern,
			start,
			maximum,
			allow_fuzzy,
			&lines_normalized,
			&pattern_normalized,
		) {
		return result;
	}
	if !allow_fuzzy {
		return SequenceSearchResult {
			index:         None,
			confidence:    0.0,
			match_count:   None,
			match_indices: Vec::new(),
			strategy:      None,
		};
	}
	let bail = SEQUENCE_FUZZY_THRESHOLD - DOMINANT_FUZZY_DELTA;
	let mut best_score = 0.0;
	let mut second_best = 0.0;
	let mut best_index = None;
	let mut fuzzy = IndexedMatches { first: None, count: 0, indices: Vec::new() };
	let mut score_range = |from: usize, to: usize| {
		if from > to {
			return;
		}
		for index in from..=to {
			let score = fuzzy_score_at(&lines_normalized, &pattern_normalized, index, bail);
			if score >= SEQUENCE_FUZZY_THRESHOLD {
				fuzzy.first.get_or_insert(index);
				fuzzy.count += 1;
				if fuzzy.indices.len() < MAX_RECORDED_MATCHES {
					fuzzy.indices.push(index);
				}
			}
			if score > best_score {
				second_best = best_score;
				best_score = score;
				best_index = Some(index);
			} else if score > second_best {
				second_best = score;
			}
		}
	};
	score_range(search_start, maximum);
	if eof && search_start > start {
		score_range(start, search_start - 1);
	}
	if let Some(index) = best_index.filter(|_| best_score >= SEQUENCE_FUZZY_THRESHOLD) {
		if fuzzy.count > 1
			&& best_score >= DOMINANT_FUZZY_MIN_CONFIDENCE
			&& best_score - second_best >= DOMINANT_FUZZY_DELTA
		{
			return SequenceSearchResult {
				index:         Some(index),
				confidence:    best_score,
				match_count:   Some(1),
				match_indices: fuzzy.indices,
				strategy:      Some(SequenceMatchStrategy::FuzzyDominant),
			};
		}
		return SequenceSearchResult {
			index:         Some(index),
			confidence:    best_score,
			match_count:   Some(fuzzy.count),
			match_indices: fuzzy.indices,
			strategy:      Some(SequenceMatchStrategy::Fuzzy),
		};
	}
	let content_text = lines.get(start..).unwrap_or_default().join("\n");
	let pattern_text = pattern.join("\n");
	let outcome = find_match(&content_text, &pattern_text, MatchOptions {
		allow_fuzzy:     true,
		threshold:       Some(0.92),
		excluded_ranges: &[],
	});
	if let Some(found) = outcome.matched {
		let line_index = start
			+ content_text[..found.start]
				.bytes()
				.filter(|byte| *byte == b'\n')
				.count();
		return SequenceSearchResult {
			index:         Some(line_index),
			confidence:    found.confidence,
			match_count:   Some(outcome.occurrences.or(outcome.fuzzy_matches).unwrap_or(1)),
			match_indices: Vec::new(),
			strategy:      Some(SequenceMatchStrategy::Character),
		};
	}
	SequenceSearchResult {
		index:         None,
		confidence:    best_score,
		match_count:   outcome.occurrences.or(outcome.fuzzy_matches),
		match_indices: Vec::new(),
		strategy:      None,
	}
}

/// Finds the closest line-sequence candidate without applying an acceptance
/// threshold.
pub fn find_closest_sequence_match(
	lines: &[&str],
	pattern: &[&str],
	start: usize,
	eof: bool,
) -> SequenceSearchResult {
	if pattern.is_empty() {
		return sequence_result(start, 1.0, SequenceMatchStrategy::Exact);
	}
	if pattern.len() > lines.len() {
		return SequenceSearchResult {
			index:         None,
			confidence:    0.0,
			match_count:   None,
			match_indices: Vec::new(),
			strategy:      Some(SequenceMatchStrategy::Fuzzy),
		};
	}
	let maximum = lines.len() - pattern.len();
	let search_start = if eof { maximum } else { start };
	let lines_normalized = lines
		.iter()
		.map(|line| normalize_for_fuzzy(line))
		.collect::<Vec<_>>();
	let pattern_normalized = pattern
		.iter()
		.map(|line| normalize_for_fuzzy(line))
		.collect::<Vec<_>>();
	let mut best_index = None;
	let mut best_score = 0.0;
	let mut score_range = |from: usize, to: usize| {
		if from > to {
			return;
		}
		for index in from..=to {
			let score = fuzzy_score_at(&lines_normalized, &pattern_normalized, index, best_score);
			if score > best_score {
				best_score = score;
				best_index = Some(index);
			}
		}
	};
	score_range(search_start, maximum);
	if eof && search_start > start {
		score_range(start, search_start - 1);
	}
	SequenceSearchResult {
		index:         best_index,
		confidence:    best_score,
		match_count:   None,
		match_indices: Vec::new(),
		strategy:      Some(SequenceMatchStrategy::Fuzzy),
	}
}

fn context_result(
	matches: IndexedMatches,
	confidence: f64,
	strategy: ContextMatchStrategy,
) -> Option<ContextLineResult> {
	Some(ContextLineResult {
		index: Some(matches.first?),
		confidence,
		match_count: Some(matches.count),
		match_indices: matches.indices,
		strategy: Some(strategy),
	})
}

/// Finds a context line using progressively less strict matching strategies.
pub fn find_context_line(
	lines: &[&str],
	context: &str,
	start: usize,
	allow_fuzzy: bool,
) -> ContextLineResult {
	find_context_line_inner(lines, context, start, allow_fuzzy, false)
}

fn find_context_line_inner(
	lines: &[&str],
	context: &str,
	start: usize,
	allow_fuzzy: bool,
	skip_function_fallback: bool,
) -> ContextLineResult {
	let empty = || ContextLineResult {
		index:         None,
		confidence:    0.0,
		match_count:   None,
		match_indices: Vec::new(),
		strategy:      None,
	};
	if start >= lines.len() {
		return empty();
	}
	let end = lines.len() - 1;
	let trimmed = context.trim();
	let exact = collect_indexed_matches(start, end, |index| lines[index] == context);
	if let Some(result) = context_result(exact, 1.0, ContextMatchStrategy::Exact) {
		return result;
	}
	let trimmed_matches =
		collect_indexed_matches(start, end, |index| lines[index].trim() == trimmed);
	if let Some(result) = context_result(trimmed_matches, 0.99, ContextMatchStrategy::Trim) {
		return result;
	}
	let normalized = normalize_unicode(context);
	let unicode_matches =
		collect_indexed_matches(start, end, |index| normalize_unicode(lines[index]) == normalized);
	if let Some(result) = context_result(unicode_matches, 0.98, ContextMatchStrategy::Unicode) {
		return result;
	}
	if !allow_fuzzy {
		return empty();
	}
	let context_normalized = normalize_for_fuzzy(context);
	if !context_normalized.is_empty() {
		let prefix = collect_indexed_matches(start, end, |index| {
			normalize_for_fuzzy(lines[index]).starts_with(&context_normalized)
		});
		if let Some(result) = context_result(prefix, 0.96, ContextMatchStrategy::Prefix) {
			return result;
		}
	}
	if utf16_len(&context_normalized) >= PARTIAL_MATCH_MIN_LENGTH {
		let substring_matches = (start..lines.len())
			.filter_map(|index| {
				let line = normalize_for_fuzzy(lines[index]);
				line.contains(&context_normalized).then_some((
					index,
					utf16_len(&context_normalized) as f64 / utf16_len(&line).max(1) as f64,
				))
			})
			.collect::<Vec<_>>();
		let retained = substring_matches
			.iter()
			.take(MAX_RECORDED_MATCHES)
			.map(|(index, _)| *index)
			.collect::<Vec<_>>();
		if substring_matches.len() == 1 {
			return ContextLineResult {
				index:         Some(substring_matches[0].0),
				confidence:    0.94,
				match_count:   Some(1),
				match_indices: retained,
				strategy:      Some(ContextMatchStrategy::Substring),
			};
		}
		let accepted = substring_matches
			.iter()
			.filter(|(_, ratio)| *ratio >= PARTIAL_MATCH_MIN_RATIO)
			.collect::<Vec<_>>();
		if !accepted.is_empty() {
			return ContextLineResult {
				index:         Some(accepted[0].0),
				confidence:    0.94,
				match_count:   Some(accepted.len()),
				match_indices: retained,
				strategy:      Some(ContextMatchStrategy::Substring),
			};
		}
		if substring_matches.len() > 1 {
			return ContextLineResult {
				index:         Some(substring_matches[0].0),
				confidence:    0.94,
				match_count:   Some(substring_matches.len()),
				match_indices: retained,
				strategy:      Some(ContextMatchStrategy::Substring),
			};
		}
	}
	let mut best_index = None;
	let mut best_score = 0.0;
	let mut fuzzy = IndexedMatches { first: None, count: 0, indices: Vec::new() };
	for (index, line) in lines.iter().enumerate().skip(start) {
		let score = similarity(&normalize_for_fuzzy(line), &context_normalized);
		if score >= CONTEXT_FUZZY_THRESHOLD {
			fuzzy.first.get_or_insert(index);
			fuzzy.count += 1;
			if fuzzy.indices.len() < MAX_RECORDED_MATCHES {
				fuzzy.indices.push(index);
			}
		}
		if score > best_score {
			best_score = score;
			best_index = Some(index);
		}
	}
	if best_score >= CONTEXT_FUZZY_THRESHOLD {
		return ContextLineResult {
			index:         best_index,
			confidence:    best_score,
			match_count:   Some(fuzzy.count),
			match_indices: fuzzy.indices,
			strategy:      Some(ContextMatchStrategy::Fuzzy),
		};
	}
	if !skip_function_fallback && trimmed.ends_with("()") {
		let stem = trimmed.strip_suffix("()").unwrap_or(trimmed);
		let with_parenthesis = format!("{stem}(");
		let result = find_context_line_inner(lines, &with_parenthesis, start, allow_fuzzy, true);
		if result.index.is_some() || result.match_count.unwrap_or(0) > 0 {
			return result;
		}
		return find_context_line_inner(lines, stem, start, allow_fuzzy, true);
	}
	ContextLineResult { confidence: best_score, ..empty() }
}

#[derive(Debug)]
struct IndentProfile<'a> {
	lines:      Vec<&'a str>,
	character:  Option<u8>,
	space_only: bool,
	tab_only:   bool,
	mixed:      bool,
	unit:       usize,
	non_empty:  usize,
}

fn leading_whitespace(line: &str) -> &str {
	&line[..count_leading_whitespace(line)]
}

const fn gcd(mut left: usize, mut right: usize) -> usize {
	while right != 0 {
		(left, right) = (right, left % right);
	}
	left
}

fn indent_profile(text: &str) -> IndentProfile<'_> {
	let lines = text.split('\n').collect::<Vec<_>>();
	let mut character = None;
	let mut space_only = true;
	let mut tab_only = true;
	let mut mixed = false;
	let mut non_empty = 0;
	let mut unit = 0;
	for line in &lines {
		if line.trim().is_empty() {
			continue;
		}
		non_empty += 1;
		let indent = leading_whitespace(line);
		let has_space = indent.as_bytes().contains(&b' ');
		let has_tab = indent.as_bytes().contains(&b'\t');
		if has_space {
			tab_only = false;
		}
		if has_tab {
			space_only = false;
		}
		if has_space && has_tab {
			mixed = true;
		}
		if let Some(first) = indent.as_bytes().first().copied() {
			match character {
				None => character = Some(first),
				Some(existing) if existing != first => mixed = true,
				_ => {},
			}
		}
		if space_only && !indent.is_empty() {
			unit = if unit == 0 {
				indent.len()
			} else {
				gcd(unit, indent.len())
			};
		}
	}
	if tab_only && non_empty > 0 {
		unit = 1;
	}
	IndentProfile { lines, character, space_only, tab_only, mixed, unit, non_empty }
}

fn indentation_only_rewrite(old: &str, new: &str) -> bool {
	old.split('\n')
		.map(str::trim)
		.eq(new.split('\n').map(str::trim))
}

fn convert_leading_tabs_to_spaces(text: &str, spaces_per_tab: usize) -> String {
	if spaces_per_tab == 0 {
		return text.to_owned();
	}
	let mut output = String::with_capacity(text.len());
	for (index, line) in text.split('\n').enumerate() {
		if index > 0 {
			output.push('\n');
		}
		let trimmed = line.trim_start();
		let leading = leading_whitespace(line);
		if trimmed.is_empty() || !leading.contains('\t') || leading.contains(' ') {
			output.push_str(line);
		} else {
			output.extend(std::iter::repeat_n(' ', leading.len() * spaces_per_tab));
			output.push_str(trimmed);
		}
	}
	output
}

fn maybe_convert_tabs(
	old: &IndentProfile<'_>,
	actual: &IndentProfile<'_>,
	new: &IndentProfile<'_>,
	new_text: &str,
) -> Option<String> {
	if !actual.space_only || !old.tab_only || !new.tab_only || actual.unit == 0 {
		return None;
	}
	for (old_line, actual_line) in old.lines.iter().zip(&actual.lines) {
		if old_line.trim().is_empty() || actual_line.trim().is_empty() {
			continue;
		}
		let old_indent = count_leading_whitespace(old_line);
		if old_indent > 0 && count_leading_whitespace(actual_line) != old_indent * actual.unit {
			return None;
		}
	}
	Some(convert_leading_tabs_to_spaces(new_text, actual.unit))
}

fn uniform_indent_delta(old: &IndentProfile<'_>, actual: &IndentProfile<'_>) -> Option<isize> {
	let mut delta = None;
	for (old_line, actual_line) in old.lines.iter().zip(&actual.lines) {
		if old_line.trim().is_empty() || actual_line.trim().is_empty() {
			continue;
		}
		let current = count_leading_whitespace(actual_line) as isize
			- count_leading_whitespace(old_line) as isize;
		match delta {
			None => delta = Some(current),
			Some(existing) if existing != current => return None,
			_ => {},
		}
	}
	delta
}

fn apply_indent_delta(text: &str, delta: isize, indent: u8) -> String {
	let mut output = String::with_capacity(text.len().saturating_add(delta.max(0) as usize));
	for (index, line) in text.split('\n').enumerate() {
		if index > 0 {
			output.push('\n');
		}
		if line.trim().is_empty() {
			output.push_str(line);
		} else if delta > 0 {
			output.extend(std::iter::repeat_n(indent as char, delta as usize));
			output.push_str(line);
		} else {
			let remove = (-delta) as usize;
			output.push_str(&line[count_leading_whitespace(line).min(remove)..]);
		}
	}
	output
}

/// Adjusts replacement indentation by the uniform delta between requested and
/// matched text.
pub fn adjust_indentation(old_text: &str, actual_text: &str, new_text: &str) -> String {
	if old_text == actual_text || indentation_only_rewrite(old_text, new_text) {
		return new_text.to_owned();
	}
	let old = indent_profile(old_text);
	let actual = indent_profile(actual_text);
	let new = indent_profile(new_text);
	if old.non_empty == 0
		|| actual.non_empty == 0
		|| new.non_empty == 0
		|| old.mixed
		|| actual.mixed
		|| new.mixed
	{
		return new_text.to_owned();
	}
	if old.character.is_some() && actual.character.is_some() && old.character != actual.character {
		return maybe_convert_tabs(&old, &actual, &new, new_text)
			.unwrap_or_else(|| new_text.to_owned());
	}
	let Some(delta) = uniform_indent_delta(&old, &actual).filter(|delta| *delta != 0) else {
		return new_text.to_owned();
	};
	if new.character.is_some() && actual.character.is_some() && new.character != actual.character {
		return new_text.to_owned();
	}
	let indent = actual.character.or(old.character).unwrap_or(b' ');
	apply_indent_delta(new_text, delta, indent)
}

struct NormalizedBase {
	text:             String,
	boundary_to_base: Vec<usize>,
	ending:           LineEnding,
}

fn normalize_base(base: &[u8]) -> Result<NormalizedBase, ReplaceError> {
	let exact = std::str::from_utf8(base).map_err(|_| ReplaceError::InvalidUtf8)?;
	let bom_len = usize::from(exact.starts_with('\u{feff}')) * '\u{feff}'.len_utf8();
	let body = &exact[bom_len..];
	let ending = detect_line_ending(body);
	let mut text = String::with_capacity(body.len());
	let mut boundaries = Vec::with_capacity(body.len() + 1);
	boundaries.push(bom_len);
	let bytes = body.as_bytes();
	let mut index = 0;
	while index < bytes.len() {
		if bytes[index] == b'\r' {
			let width = if bytes.get(index + 1) == Some(&b'\n') {
				2
			} else {
				1
			};
			text.push('\n');
			index += width;
			boundaries.push(bom_len + index);
		} else {
			let ch = body[index..].chars().next().expect("valid UTF-8 boundary");
			text.push(ch);
			index += ch.len_utf8();
			let previous = *boundaries.last().expect("initial boundary");
			for offset in 1..=ch.len_utf8() {
				boundaries.push(if offset == ch.len_utf8() {
					bom_len + index
				} else {
					previous + offset
				});
			}
		}
	}
	Ok(NormalizedBase { text, boundary_to_base: boundaries, ending })
}

fn replacement_bytes(text: &str, ending: LineEnding) -> Bytes {
	Bytes::copy_from_slice(restore_line_endings(text, ending).as_bytes())
}

fn apply_canonical_edits(base: &[u8], edits: &[ReplaceEdit]) -> Bytes {
	let removed = edits
		.iter()
		.map(|edit| edit.end - edit.start)
		.sum::<usize>();
	let inserted = edits
		.iter()
		.map(|edit| edit.replacement.len())
		.sum::<usize>();
	let mut result = BytesMut::with_capacity(base.len() - removed + inserted);
	let mut source = 0;
	for edit in edits {
		result.extend_from_slice(&base[source..edit.start]);
		result.extend_from_slice(&edit.replacement);
		source = edit.end;
	}
	result.extend_from_slice(&base[source..]);
	result.freeze()
}

fn canonical_edit(
	normalized: &NormalizedBase,
	start: usize,
	end: usize,
	replacement: Bytes,
) -> ReplaceEdit {
	ReplaceEdit {
		start: normalized.boundary_to_base[start],
		end: normalized.boundary_to_base[end],
		replacement,
	}
}

fn non_overlapping_exact_indices(content: &str, target: &str) -> Vec<usize> {
	let mut result = Vec::new();
	let mut start = 0;
	while start <= content.len().saturating_sub(target.len()) {
		let Some(relative) = content[start..].find(target) else {
			break;
		};
		let index = start + relative;
		result.push(index);
		start = index + target.len();
	}
	result
}

/// Applies replacement semantics to exact base bytes without filesystem access.
///
/// Matching uses LF-normalized text, while returned edits address the original
/// bytes and retain the original BOM and line-ending convention. Applying the
/// returned canonical edits always reproduces [`ReplaceResult::final_bytes`].
pub fn apply_replace(
	base: &[u8],
	old_text: &str,
	new_text: &str,
	options: ReplaceOptions,
) -> Result<ReplaceResult, ReplaceError> {
	if old_text.is_empty() {
		return Err(ReplaceError::EmptyOldText);
	}
	let normalized = normalize_base(base)?;
	let old = normalize_to_lf(old_text);
	let new = normalize_to_lf(new_text);
	let old = old.as_ref();
	let new = new.as_ref();
	if old.is_empty() {
		return Err(ReplaceError::EmptyOldText);
	}
	let exact_indices = non_overlapping_exact_indices(&normalized.text, old);
	let mut normalized_edits: Vec<(usize, usize, String)> = Vec::new();
	if options.replace_all && !exact_indices.is_empty() {
		normalized_edits.extend(
			exact_indices
				.into_iter()
				.map(|start| (start, start + old.len(), new.to_owned())),
		);
	} else if options.replace_all {
		let mut excluded = Vec::new();
		loop {
			let outcome = find_match(&normalized.text, old, MatchOptions {
				allow_fuzzy:     options.allow_fuzzy,
				threshold:       Some(options.threshold),
				excluded_ranges: &excluded,
			});
			if outcome.matched.is_none() && outcome.fuzzy_matches.is_some_and(|count| count > 1) {
				return Err(no_match_error(&outcome, old, options));
			}
			if outcome.matched.is_none() {
				if normalized_edits.is_empty() {
					return Err(no_match_error(&outcome, old, options));
				}
				break;
			}
			let found = outcome.matched.expect("match presence checked");
			let actual = std::str::from_utf8(&found.actual_text).expect("match slices valid UTF-8");
			let adjusted = adjust_indentation(old, actual, new);
			if adjusted.as_bytes() == found.actual_text.as_ref() {
				break;
			}
			let end = found.end().max(found.start + 1);
			excluded.push(ExcludedRange { start: found.start, end });
			normalized_edits.push((found.start, end, adjusted));
		}
	} else {
		let outcome = find_match(&normalized.text, old, MatchOptions {
			allow_fuzzy:     options.allow_fuzzy,
			threshold:       Some(options.threshold),
			excluded_ranges: &[],
		});
		if outcome.occurrences.unwrap_or(0) > 1 {
			return Err(ReplaceError::AmbiguousExact {
				occurrences: outcome.occurrences.unwrap_or(0),
				lines:       outcome.occurrence_lines,
				previews:    outcome.occurrence_previews,
			});
		}
		if outcome.matched.is_none() {
			return Err(no_match_error(&outcome, old, options));
		}
		let found = outcome.matched.expect("match presence checked");
		let actual = std::str::from_utf8(&found.actual_text).expect("match slices valid UTF-8");
		let adjusted = adjust_indentation(old, actual, new);
		normalized_edits.push((found.start, found.end(), adjusted));
	}
	normalized_edits.sort_by_key(|edit| edit.0);
	let edits = normalized_edits
		.into_iter()
		.map(|(start, end, replacement)| {
			canonical_edit(&normalized, start, end, replacement_bytes(&replacement, normalized.ending))
		})
		.collect::<Vec<_>>();
	let final_bytes = apply_canonical_edits(base, &edits);
	if final_bytes.as_ref() == base {
		return Err(ReplaceError::NoChanges);
	}
	Ok(ReplaceResult { count: edits.len(), edits, final_bytes })
}

#[cfg(test)]
mod tests {
	use super::*;

	fn options(allow_fuzzy: bool) -> MatchOptions<'static> {
		MatchOptions { allow_fuzzy, threshold: None, excluded_ranges: &[] }
	}

	#[test]
	fn unicode_normalization_canonicalizes_equivalent_text() {
		assert_eq!(normalize_unicode("  cafe\u{301}  "), "café");
	}

	#[test]
	fn exact_matching_and_ambiguity() {
		let found = find_match("line1\nline2\nline3", "line2", options(false));
		assert_eq!(found.matched.as_ref().unwrap().confidence, 1.0);
		assert_eq!(found.matched.as_ref().unwrap().start_line, 2);
		let ambiguous = find_match("foo\nbar\nfoo", "foo", options(false));
		assert!(ambiguous.matched.is_none());
		assert_eq!(ambiguous.occurrences, Some(2));
		let missing = find_match("line1\nline2", "notfound", options(false));
		assert!(missing.matched.is_none());
		assert_eq!(missing.occurrences, None);
	}

	#[test]
	fn tab_and_space_normalization() {
		for (content, target) in [
			("\tfoo\n\t\tbar\n\tbaz", "  foo\n    bar\n  baz"),
			("  foo\n    bar\n  baz", "\tfoo\n\t\tbar\n\tbaz"),
			("   foo\n      bar\n   baz", "  foo\n    bar\n  baz"),
			("prefix\n\t\t\t\"value\",\nsuffix", "          \"value\","),
		] {
			assert!(
				find_match(content, target, options(true))
					.matched
					.unwrap()
					.confidence
					>= DEFAULT_FUZZY_THRESHOLD
			);
		}
	}

	#[test]
	fn inconsistent_indentation_fallback() {
		let cases = [
			(
				"\t\t\tline1\n\t\t\tline2\n\t\tline3\n\t\t\tline4",
				"      line1\n      line2\n      line3\n      line4",
			),
			("  a\n    b\n   c\n    d", "  a\n    b\n    c\n    d"),
		];
		for (content, target) in cases {
			assert!(find_match(content, target, options(true)).matched.is_some());
		}
	}

	#[test]
	fn fuzzy_content_and_thresholds() {
		assert!(
			find_match("foo   bar    baz", "foo bar baz", options(true))
				.matched
				.is_some()
		);
		assert!(
			find_match("line1  \nline2\t", "line1\nline2", options(true))
				.matched
				.is_some()
		);
		let strict = find_match("function foo() {}", "function bar() {}", MatchOptions {
			allow_fuzzy:     true,
			threshold:       Some(0.99),
			excluded_ranges: &[],
		});
		assert!(strict.matched.is_none());
		let lenient = find_match("function foo() {}", "function bar() {}", MatchOptions {
			allow_fuzzy:     true,
			threshold:       Some(0.7),
			excluded_ranges: &[],
		});
		assert!(lenient.matched.is_some());
		let repeated = find_match("  item1\n  item2\n  item3", "  itemX", MatchOptions {
			allow_fuzzy:     true,
			threshold:       Some(0.7),
			excluded_ranges: &[],
		});
		assert!(repeated.fuzzy_matches.unwrap() > 1);
	}

	#[test]
	fn find_match_edge_cases() {
		assert_eq!(find_match("some content", "", options(true)), MatchOutcome::default());
		assert!(
			find_match("line1\n\nline3", "line1\n\nline3", options(false))
				.matched
				.is_some()
		);
		assert!(
			find_match("short", "this is much longer than the content", options(true))
				.matched
				.is_none()
		);
		assert!(
			find_match("😀a", "😀b", MatchOptions {
				allow_fuzzy:     true,
				threshold:       Some(2.0 / 3.0),
				excluded_ranges: &[],
			})
			.matched
			.is_some()
		);
	}

	#[test]
	fn indentation_adjustment() {
		assert_eq!(
			adjust_indentation("foo\nbar", "    foo\n    bar", "foo\nbaz\nbar"),
			"    foo\n    baz\n    bar"
		);
		assert_eq!(
			adjust_indentation(
				"        foo\n        bar",
				"    foo\n    bar",
				"        foo\n        baz"
			),
			"    foo\n    baz"
		);
		assert_eq!(
			adjust_indentation("foo\n\nbar", "    foo\n\n    bar", "foo\n\nbaz"),
			"    foo\n\n    baz"
		);
		assert_eq!(adjust_indentation("    foo", "    foo", "    bar"), "    bar");
		assert_eq!(adjust_indentation("foo", "\t\tfoo", "bar"), "\t\tbar");
		assert_eq!(
			adjust_indentation(
				"if (x) {\n  return y;\n}",
				"    if (x) {\n      return y;\n    }",
				"if (x) {\n  return z;\n}"
			),
			"    if (x) {\n      return z;\n    }"
		);
		assert_eq!(adjust_indentation("    foo", "foo", "  bar"), "bar");
	}

	#[test]
	fn sequence_strategies() {
		assert_eq!(
			seek_sequence(&["foo", "bar", "baz"], &["bar", "baz"], 0, false, true).index,
			Some(1)
		);
		assert_eq!(seek_sequence(&["foo", "bar"], &[], 5, false, true).index, Some(5));
		assert_eq!(
			seek_sequence(&["a", "b", "c", "d", "e"], &["d", "e"], 0, true, true).index,
			Some(3)
		);
		let comment = seek_sequence(
			&["// local import - avoids top-level dep"],
			&["local import - avoids top-level dep"],
			0,
			false,
			true,
		);
		assert_eq!(comment.strategy, Some(SequenceMatchStrategy::CommentPrefix));
		let disabled = seek_sequence(&["foo value"], &["foo"], 0, false, false);
		assert!(disabled.index.is_none());

		assert_eq!(
			seek_sequence(&["foo   ", "bar\t\t"], &["foo", "bar"], 0, false, true).index,
			Some(0)
		);
		assert_eq!(
			seek_sequence(&["    foo   ", "   bar\t"], &["foo", "bar"], 0, false, true).index,
			Some(0)
		);
		assert!(
			seek_sequence(&["just one line"], &["too", "many", "lines"], 0, false, true)
				.index
				.is_none()
		);
		let unicode = seek_sequence(
			&["import asyncio  # local import – avoids top‑level dep"],
			&["import asyncio  # local import - avoids top-level dep"],
			0,
			false,
			true,
		);
		assert_eq!(unicode.strategy, Some(SequenceMatchStrategy::Unicode));
		let minor = seek_sequence(
			&["function greet() {", "  console.log(\"Hello!\");", "}"],
			&["function greet() {", "  console.log(\"Hello!\")  ", "}"],
			0,
			false,
			true,
		);
		assert_eq!(minor.index, Some(0));
		assert!(minor.confidence >= SEQUENCE_FUZZY_THRESHOLD);
		let character = seek_sequence(
			&[
				"function calculateTotal(items) {",
				"  let sum = 0;",
				"  for (const item of items) {",
				"    sum += item.price * item.quantity;",
				"  }",
			],
			&["  for (const item of items)  {", "    sum += item.price*item.quantity;"],
			0,
			false,
			true,
		);
		assert_eq!(character.index, Some(2));
		assert!(character.confidence > 0.9);
		assert_eq!(
			seek_sequence(
				&["  function   foo()  {", "    return   42;", "  }"],
				&["function foo() {", "return 42;"],
				0,
				false,
				true,
			)
			.index,
			Some(0),
		);
	}

	#[test]
	fn context_strategies() {
		assert_eq!(
			find_context_line(&["function foo() {", "}"], "function foo() {", 0, true).index,
			Some(0)
		);
		assert_eq!(
			find_context_line(&["  function foo()  {"], "function foo()  {", 0, true).strategy,
			Some(ContextMatchStrategy::Trim)
		);
		assert_eq!(
			find_context_line(
				&["const msg = \"Hello – World\";"],
				"const msg = \"Hello - World\";",
				0,
				true
			)
			.strategy,
			Some(ContextMatchStrategy::Unicode)
		);
		assert_eq!(
			find_context_line(
				&["function calculateTotalWithTax(items) {"],
				"function calculateTotalWithTax(items",
				0,
				true
			)
			.strategy,
			Some(ContextMatchStrategy::Prefix)
		);
		assert_eq!(
			find_context_line(
				&["// comment: calculateTotal here", "function foo() {}"],
				"calculateTotal",
				0,
				true
			)
			.strategy,
			Some(ContextMatchStrategy::Substring)
		);
		assert_eq!(
			find_context_line(
				&["function calculteTotal(items) {"],
				"function calculateTotal(items) {",
				0,
				true
			)
			.strategy,
			Some(ContextMatchStrategy::Fuzzy)
		);
		let whitespace = find_context_line(
			&["  function foo()  {", "  return 1;", "}"],
			"function foo() {",
			0,
			true,
		);
		assert_eq!(whitespace.index, Some(0));
		assert!(whitespace.confidence > 0.9);
		let typo = find_context_line(
			&["functoin calclateTotal(itms) {", "  return 0;", "}"],
			"function calculateTotal(items) {",
			0,
			true,
		);
		assert_eq!(typo.index, Some(0));
		assert!(typo.confidence > 0.8);
	}

	#[test]
	fn exact_replace_rules_and_canonical_edits() {
		let base = Bytes::from_static(b"foo bar foo baz foo");
		assert!(matches!(
			apply_replace(&base, "foo", "qux", ReplaceOptions::default()),
			Err(ReplaceError::AmbiguousExact { occurrences: 3, .. })
		));
		let result = apply_replace(&base, "foo", "qux", ReplaceOptions {
			replace_all: true,
			..ReplaceOptions::default()
		})
		.unwrap();
		assert_eq!(result.count, 3);
		assert_eq!(&result.final_bytes[..], b"qux bar qux baz qux");
		assert_eq!(apply_canonical_edits(b"foo bar foo baz foo", &result.edits), result.final_bytes);
	}

	#[test]
	fn replace_all_multiline_single_and_missing() {
		let result = apply_replace(
			b"start\nfoo\nbar\nend\nstart\nfoo\nbar\nend",
			"foo\nbar",
			"replaced",
			ReplaceOptions { replace_all: true, ..ReplaceOptions::default() },
		)
		.unwrap();
		assert_eq!(&result.final_bytes[..], b"start\nreplaced\nend\nstart\nreplaced\nend");
		let single = apply_replace(b"hello world", "world", "universe", ReplaceOptions {
			replace_all: true,
			..ReplaceOptions::default()
		})
		.unwrap();
		assert_eq!(single.count, 1);
		let error = apply_replace(b"hello world", "missing", "x", ReplaceOptions {
			replace_all: true,
			..ReplaceOptions::default()
		})
		.unwrap_err();
		assert!(matches!(
			error,
			ReplaceError::FuzzyBelowThreshold { .. } | ReplaceError::NoCloseMatch { .. }
		));
	}

	#[test]
	fn ambiguous_fuzzy_replace_all_is_rejected() {
		let base = Bytes::from_static(b"function a() {\n  if (x) {\n    doThing();\n  }\n}\nfunction b() {\n    if (x) {\n        doThing();\n    }\n}\n");
		let result = apply_replace(
			&base,
			"if (x) {\n  doThing();\n}",
			"if (y) {\n  doOther();\n}",
			ReplaceOptions {
				replace_all: true,
				allow_fuzzy: true,
				threshold:   DEFAULT_FUZZY_THRESHOLD,
			},
		);
		assert!(matches!(result, Err(ReplaceError::AmbiguousFuzzy { matches: 2, .. })));
	}

	#[test]
	fn replace_all_does_not_rematch_inserted_text() {
		let old = "a".repeat(50);
		let first = format!("{}b", "a".repeat(49));
		let second = format!("{}cccccc", "a".repeat(44));
		let replacement = format!("{old}\nexpanded");
		let base = Bytes::from(format!("{first}\n{second}"));
		let result = apply_replace(&base, &old, &replacement, ReplaceOptions {
			replace_all: true,
			allow_fuzzy: true,
			threshold:   0.8,
		})
		.unwrap();
		assert_eq!(result.count, 2);
		assert_eq!(result.final_bytes, Bytes::from(format!("{replacement}\n{replacement}")));
	}

	#[test]
	fn preserves_bom_crlf_and_utf8_byte_coordinates() {
		let base = Bytes::from_static(b"\xef\xbb\xbffirst\r\nsecond \xf0\x9f\x98\x80\r\nthird\r\n");
		let result = apply_replace(&base, "second 😀\n", "REPLACED\n", ReplaceOptions {
			allow_fuzzy: false,
			..ReplaceOptions::default()
		})
		.unwrap();
		assert_eq!(&result.final_bytes[..], b"\xef\xbb\xbffirst\r\nREPLACED\r\nthird\r\n");
		assert_eq!(result.edits[0].start, 10);
		assert_eq!(apply_canonical_edits(&base, &result.edits), result.final_bytes);
		let lf = apply_replace(b"first\nsecond\nthird\n", "second\n", "REPLACED\n", ReplaceOptions {
			allow_fuzzy: false,
			..ReplaceOptions::default()
		})
		.unwrap();
		assert_eq!(lf.final_bytes.as_ref(), b"first\nREPLACED\nthird\n");

		let mixed = apply_replace(
			b"hello\r\nworld\r\n---\r\nhello\nworld\n",
			"hello\nworld\n",
			"replaced\n",
			ReplaceOptions { allow_fuzzy: false, ..ReplaceOptions::default() },
		);
		assert!(matches!(mixed, Err(ReplaceError::AmbiguousExact { occurrences: 2, .. })));
	}

	#[test]
	fn exact_mode_and_no_op_are_strict() {
		let error = apply_replace(b"foo   bar", "foo bar", "x", ReplaceOptions {
			allow_fuzzy: false,
			..ReplaceOptions::default()
		})
		.unwrap_err();
		assert_eq!(error.code(), "exact_mismatch");
		assert!(matches!(
			apply_replace(b"same", "same", "same", ReplaceOptions::default()),
			Err(ReplaceError::NoChanges)
		));
	}
}
