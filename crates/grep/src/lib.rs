//! Binding-free regex search over in-memory bytes and workspace files.
//!
//! Matching is implemented with ripgrep's regex engine and falls back to PCRE2
//! for look-around, backreferences, and other constructs unsupported by Rust's
//! regex automata. Filesystem traversal and parallel worker ownership remain in
//! [`omp_walker`].

use std::{
	borrow::Cow,
	fmt,
	fs::File,
	io::{self, Read},
	path::{Path, PathBuf},
	sync::{
		LazyLock,
		atomic::{AtomicU64, Ordering},
	},
	time::{Duration, Instant},
};

use grep_matcher::Matcher;
use grep_pcre2::{RegexMatcher as PcreMatcher, RegexMatcherBuilder as PcreMatcherBuilder};
use grep_regex::{RegexMatcher, RegexMatcherBuilder};
use grep_searcher::{
	BinaryDetection, Searcher, SearcherBuilder, Sink, SinkContext, SinkContextKind, SinkMatch,
};
use omp_core::Str;
use omp_walker::{
	CompiledWalkGlob, DirectoryErrorMode, FileCandidate, FollowLinks, SizeHintPolicy, WalkDetail,
	WalkFilter, WalkOrder, WalkRequest, execute_candidates_init,
};
use parking_lot::Mutex;
use smallvec::SmallVec;
use thiserror::Error;

/// Maximum number of bytes searched from any one file.
pub const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;

const FILE_CLASSIFICATION_READ_BYTES: u64 = MAX_FILE_BYTES + 1;

/// Whether PCRE2 JIT is enabled for fallback matchers.
///
/// `OMP_PCRE2_JIT=0` and `OMP_PCRE2_JIT=false` disable JIT. Any other non-empty
/// value enables it. When unset, JIT is enabled except on macOS, where PCRE2's
/// executable allocator is not reliable in every host process.
static PCRE2_JIT_ENABLED: LazyLock<bool> = LazyLock::new(|| match std::env::var("OMP_PCRE2_JIT") {
	Ok(value) if !value.is_empty() => value != "0" && !value.eq_ignore_ascii_case("false"),
	_ => !cfg!(target_os = "macos"),
});

/// Output mode used by [`search`] and [`grep`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GrepOutputMode {
	/// Return matched lines and requested context.
	#[default]
	Content,
	/// Return one result entry per matching file and count every match.
	Count,
	/// Return one result entry per matching file.
	FilesWithMatches,
}

/// Options shared by in-memory and filesystem searches.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrepOptions {
	/// Regex pattern to search for.
	pub pattern:            Str,
	/// File or directory to search, or the display path for [`search`].
	pub path:               Str,
	/// Optional recursive glob filter for filesystem searches.
	pub glob:               Option<Str>,
	/// Match without regard to case.
	pub ignore_case:        bool,
	/// Enable multiline matching; multiline-looking patterns enable it as well.
	pub multiline:          bool,
	/// Include dot-prefixed files and directories.
	pub hidden:             bool,
	/// Respect ignore files and repository excludes.
	pub gitignore:          bool,
	/// Maximum number of returned matches across all files.
	pub max_count:          Option<u32>,
	/// Maximum number of returned content matches from each file.
	pub max_count_per_file: Option<u32>,
	/// Number of context lines to retain before each match.
	pub context_before:     u32,
	/// Number of context lines to retain after each match.
	pub context_after:      u32,
	/// Maximum line length in UTF-8 bytes, including a three-byte ellipsis.
	pub max_columns:        Option<u32>,
	/// Shape of returned match entries.
	pub mode:               GrepOutputMode,
	/// Deadline in milliseconds from the start of the operation.
	pub timeout_ms:         Option<u32>,
}

impl Default for GrepOptions {
	fn default() -> Self {
		Self {
			pattern:            Str::new(""),
			path:               Str::new("."),
			glob:               None,
			ignore_case:        false,
			multiline:          false,
			hidden:             true,
			gitignore:          true,
			max_count:          None,
			max_count_per_file: None,
			context_before:     0,
			context_after:      0,
			max_columns:        None,
			mode:               GrepOutputMode::Content,
			timeout_ms:         Some(30_000),
		}
	}
}

/// One source line retained around a match.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextLine {
	/// One-indexed line number in the source.
	pub line_number: u32,
	/// Source text with its line ending removed.
	pub line:        Str,
}

/// One content match or one file marker in a non-content output mode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrepMatch {
	/// Search-root-relative path, or the caller-provided path for [`search`].
	pub path:           Str,
	/// One-indexed source line, or zero for non-content output modes.
	pub line_number:    u32,
	/// Matched line text, or an empty string for non-content output modes.
	pub line:           Str,
	/// Whether `line` was shortened to `GrepOptions::max_columns`.
	pub truncated:      bool,
	/// Context retained before the match.
	pub context_before: SmallVec<ContextLine, 8>,
	/// Context retained after the match.
	pub context_after:  SmallVec<ContextLine, 8>,
}

/// Aggregated result of an in-memory or filesystem search.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GrepResult {
	/// Returned content matches or matching-file markers.
	pub matches:            Vec<GrepMatch>,
	/// Matches observed across searched files before the global output cap.
	pub total_matches:      u32,
	/// Number of searched files containing at least one match.
	pub files_with_matches: u32,
	/// Number of files successfully read and searched.
	pub files_searched:     u32,
	/// Whether a global or per-file output cap omitted matches.
	pub limit_reached:      bool,
	/// Oversized files whose leading window could not be read.
	pub skipped_oversized:  u32,
}

/// Failure from matcher compilation, traversal, or searching.
#[derive(Debug, Error)]
pub enum GrepError {
	/// Both the Rust regex engine and PCRE2 rejected the pattern.
	#[error("invalid regex: {regex}; PCRE2 fallback: {pcre2}")]
	InvalidRegex {
		/// Rust regex compilation diagnostic.
		regex: Str,
		/// PCRE2 compilation diagnostic.
		pcre2: Str,
	},
	/// The filesystem target does not exist.
	#[error("path not found: {path}")]
	PathNotFound {
		/// Caller-provided path.
		path: Str,
	},
	/// A filename glob was invalid.
	#[error("invalid glob pattern: {message}")]
	InvalidGlob {
		/// Glob compiler diagnostic.
		message: Str,
	},
	/// Workspace traversal failed.
	#[error("filesystem traversal failed: {message}")]
	Walk {
		/// Walker diagnostic.
		message: Str,
	},
	/// A readable input could not be searched.
	#[error("search failed: {message}")]
	Search {
		/// Searcher diagnostic.
		message: Str,
	},
	/// The configured operation deadline elapsed.
	#[error("grep timed out after {timeout_ms}ms")]
	Timeout {
		/// Configured timeout.
		timeout_ms: u32,
	},
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SearchParams {
	context_before:     u32,
	context_after:      u32,
	max_columns:        Option<u32>,
	mode:               GrepOutputMode,
	max_count:          Option<u64>,
	max_count_per_file: Option<u64>,
	multiline:          bool,
}

#[derive(Debug)]
struct CollectedMatch {
	line_number:    u64,
	line:           Str,
	context_before: SmallVec<ContextLine, 8>,
	context_after:  SmallVec<ContextLine, 8>,
	truncated:      bool,
}

struct SearchResultInternal {
	matches:       Vec<CollectedMatch>,
	match_count:   u64,
	limit_reached: bool,
}

#[derive(Debug)]
struct FileSearchResult {
	relative_path: Str,
	matches:       Vec<CollectedMatch>,
	match_count:   u64,
	limit_reached: bool,
}

struct SearchWorker {
	searcher: Searcher,
	buffer:   Vec<u8>,
}

impl SearchWorker {
	fn new(params: SearchParams) -> Self {
		Self { searcher: build_searcher(params), buffer: Vec::new() }
	}
}

struct MatchCollector {
	matches:         Vec<CollectedMatch>,
	match_count:     u64,
	collected_count: u64,
	max_count:       Option<u64>,
	limit_reached:   bool,
	max_columns:     Option<usize>,
	collect_matches: bool,
	context_before:  SmallVec<ContextLine, 8>,
}

impl MatchCollector {
	fn new(max_count: Option<u64>, max_columns: Option<usize>, collect_matches: bool) -> Self {
		Self {
			matches: Vec::new(),
			match_count: 0,
			collected_count: 0,
			max_count,
			limit_reached: false,
			max_columns,
			collect_matches,
			context_before: SmallVec::new(),
		}
	}
}

impl Sink for MatchCollector {
	type Error = io::Error;

	fn matched(
		&mut self,
		_searcher: &Searcher,
		matched: &SinkMatch<'_>,
	) -> Result<bool, Self::Error> {
		self.match_count = self.match_count.saturating_add(1);
		if self.limit_reached {
			return Ok(false);
		}

		if self.collect_matches {
			let (line, truncated) =
				truncate_line(bytes_to_trimmed_str(matched.bytes()), self.max_columns);
			self.matches.push(CollectedMatch {
				line_number: matched.line_number().unwrap_or(0),
				line,
				context_before: std::mem::take(&mut self.context_before),
				context_after: SmallVec::new(),
				truncated,
			});
		} else {
			self.context_before.clear();
		}

		self.collected_count = self.collected_count.saturating_add(1);
		if self
			.max_count
			.is_some_and(|max| self.collected_count >= max)
		{
			self.limit_reached = true;
		}
		Ok(true)
	}

	fn context(
		&mut self,
		_searcher: &Searcher,
		context: &SinkContext<'_>,
	) -> Result<bool, Self::Error> {
		if !self.collect_matches {
			return Ok(true);
		}
		let kind = context.kind();
		let (line, _) = truncate_line(bytes_to_trimmed_str(context.bytes()), self.max_columns);
		let context_line =
			ContextLine { line_number: clamp_u32(context.line_number().unwrap_or(0)), line };
		match kind {
			SinkContextKind::Before => self.context_before.push(context_line),
			SinkContextKind::After => {
				if let Some(last_match) = self.matches.last_mut() {
					last_match.context_after.push(context_line);
				}
			},
			SinkContextKind::Other => {},
		}
		Ok(true)
	}
}

#[derive(Clone, Copy)]
struct Deadline {
	started:    Instant,
	timeout_ms: Option<u32>,
}

impl Deadline {
	fn new(timeout_ms: Option<u32>) -> Self {
		Self { started: Instant::now(), timeout_ms }
	}

	fn check(self) -> Result<(), GrepError> {
		let Some(timeout_ms) = self.timeout_ms else {
			return Ok(());
		};
		if self.started.elapsed() >= Duration::from_millis(u64::from(timeout_ms)) {
			return Err(GrepError::Timeout { timeout_ms });
		}
		Ok(())
	}

	fn expired(self) -> bool {
		self.timeout_ms.is_some_and(|timeout_ms| {
			self.started.elapsed() >= Duration::from_millis(u64::from(timeout_ms))
		})
	}
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ReadPolicy {
	Full,
	Prefix,
}

enum ReadFile {
	Read,
	Oversized,
	Skipped,
}

enum FileOutcome {
	Searched(SearchResultInternal),
	Defer,
	SkippedOversized,
	Skipped,
}

#[derive(Default)]
struct PassState {
	results:           Mutex<Vec<FileSearchResult>>,
	deferred:          Mutex<Vec<FileCandidate>>,
	files_searched:    AtomicU64,
	skipped_oversized: AtomicU64,
}

enum CompiledMatcher {
	Rust(RegexMatcher),
	Pcre2(PcreMatcher),
}

#[derive(Debug)]
enum CompiledMatcherError {
	Rust(grep_matcher::NoError),
	Pcre2(grep_pcre2::Error),
}

impl fmt::Display for CompiledMatcherError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Rust(error) => error.fmt(formatter),
			Self::Pcre2(error) => error.fmt(formatter),
		}
	}
}

impl Matcher for CompiledMatcher {
	type Captures = grep_matcher::NoCaptures;
	type Error = CompiledMatcherError;

	fn find_at(
		&self,
		haystack: &[u8],
		at: usize,
	) -> Result<Option<grep_matcher::Match>, Self::Error> {
		match self {
			Self::Rust(matcher) => matcher
				.find_at(haystack, at)
				.map_err(CompiledMatcherError::Rust),
			Self::Pcre2(matcher) => matcher
				.find_at(haystack, at)
				.map_err(CompiledMatcherError::Pcre2),
		}
	}

	fn new_captures(&self) -> Result<Self::Captures, Self::Error> {
		Ok(grep_matcher::NoCaptures::new())
	}
}

/// Search an in-memory byte slice.
///
/// The `path` option is copied into returned matches as their display path.
pub fn search(content: &[u8], options: &GrepOptions) -> Result<GrepResult, GrepError> {
	let deadline = Deadline::new(options.timeout_ms);
	deadline.check()?;
	let multiline = infer_multiline(options.pattern.as_str(), options.multiline);
	let matcher = build_matcher(options.pattern.as_str(), options.ignore_case, multiline)?;
	let params = search_params(options, multiline);
	let result = run_search(&matcher, content, params).map_err(search_error)?;
	deadline.check()?;
	Ok(aggregate_single(result, options.path.clone(), params, 0))
}

/// Search a file or directory synchronously.
///
/// Directory searches discover candidates through [`WalkRequest`] and execute
/// file work through [`execute_candidates_init`]. Directory match paths are
/// normalized and relative to the requested root.
pub fn grep(options: &GrepOptions) -> Result<GrepResult, GrepError> {
	let deadline = Deadline::new(options.timeout_ms);
	deadline.check()?;
	let target = resolve_search_path(options.path.as_str())?;
	let metadata = std::fs::metadata(&target)
		.map_err(|_| GrepError::PathNotFound { path: options.path.clone() })?;
	let multiline = infer_multiline(options.pattern.as_str(), options.multiline);
	let matcher = build_matcher(options.pattern.as_str(), options.ignore_case, multiline)?;
	let params = search_params(options, multiline);

	if metadata.is_file() {
		return grep_file(&target, options.path.clone(), &matcher, params, deadline);
	}
	if !metadata.is_dir() {
		return Ok(GrepResult::default());
	}

	let request = build_walk_request(&target, options)?;
	let candidates = request
		.collect_file_candidates_with_heartbeat(|| deadline.check())
		.map_err(|error| {
			if deadline.expired() {
				GrepError::Timeout { timeout_ms: options.timeout_ms.unwrap_or(0) }
			} else {
				GrepError::Walk { message: Str::from(error.to_string()) }
			}
		})?;
	let (results, skipped_oversized, files_searched) =
		process_candidates(candidates, &matcher, params, deadline)?;
	deadline.check()?;
	Ok(aggregate_results(results, params, files_searched, skipped_oversized))
}

fn search_params(options: &GrepOptions, multiline: bool) -> SearchParams {
	let content_mode = options.mode == GrepOutputMode::Content;
	SearchParams {
		context_before: if content_mode {
			options.context_before
		} else {
			0
		},
		context_after: if content_mode {
			options.context_after
		} else {
			0
		},
		max_columns: options.max_columns,
		mode: options.mode,
		max_count: options.max_count.map(u64::from),
		max_count_per_file: options.max_count_per_file.map(u64::from),
		multiline,
	}
}

fn infer_multiline(pattern: &str, requested: bool) -> bool {
	requested || pattern.contains('\n') || pattern.contains("\\n")
}

fn resolve_search_path(path: &str) -> Result<PathBuf, GrepError> {
	let path = PathBuf::from(path);
	if path.is_absolute() {
		return Ok(path);
	}
	std::env::current_dir()
		.map(|cwd| cwd.join(path))
		.map_err(|error| GrepError::Walk { message: Str::from(error.to_string()) })
}

fn build_walk_request(target: &Path, options: &GrepOptions) -> Result<WalkRequest, GrepError> {
	let mut filter = WalkFilter::files_only();
	if let Some(glob) = options
		.glob
		.as_ref()
		.map(Str::as_str)
		.map(str::trim)
		.filter(|glob| !glob.is_empty())
	{
		let pattern = normalize_recursive_glob(glob);
		let compiled = CompiledWalkGlob::new([pattern])
			.map_err(|error| GrepError::InvalidGlob { message: Str::from(error.to_string()) })?;
		filter = filter.glob(compiled);
	}
	let mentions_node_modules = options
		.glob
		.as_ref()
		.is_some_and(|glob| glob.as_str().contains("node_modules"));
	Ok(WalkRequest::new(target)
		.hidden(options.hidden)
		.gitignore(options.gitignore)
		.skip_git(true)
		.skip_node_modules(!mentions_node_modules)
		.follow_links(FollowLinks::Never)
		.detail(WalkDetail::Minimal)
		.size_hints(SizeHintPolicy::Always)
		.order(WalkOrder::Path)
		.emit_root(false)
		.depth(1, usize::MAX)
		.directory_errors(DirectoryErrorMode::SkipSkippable)
		.cache(false)
		.filter(filter))
}

fn normalize_recursive_glob(glob: &str) -> String {
	let normalized = glob.replace('\\', "/");
	let mut pattern = if normalized.contains('/')
		|| normalized.starts_with("**")
		|| is_exact_brace_union(&normalized)
	{
		normalized
	} else {
		format!("**/{normalized}")
	};
	let opens = pattern.bytes().filter(|byte| *byte == b'{').count();
	let closes = pattern.bytes().filter(|byte| *byte == b'}').count();
	if opens > closes {
		pattern.extend(std::iter::repeat_n('}', opens - closes));
	}
	pattern
}

fn is_exact_brace_union(pattern: &str) -> bool {
	if !(pattern.starts_with('{') && pattern.ends_with('}')) {
		return false;
	}
	let inner = &pattern[1..pattern.len() - 1];
	!inner.is_empty()
		&& !inner
			.chars()
			.any(|character| matches!(character, '*' | '?' | '[' | ']' | '{' | '}'))
}

fn build_regex_matcher(
	pattern: &str,
	ignore_case: bool,
	multiline: bool,
) -> Result<RegexMatcher, grep_regex::Error> {
	let build = |line_terminated| {
		let mut builder = RegexMatcherBuilder::new();
		builder.case_insensitive(ignore_case).multi_line(multiline);
		if line_terminated {
			builder.line_terminator(Some(b'\n'));
		}
		builder.build(pattern)
	};
	if !multiline && let Ok(matcher) = build(true) {
		return Ok(matcher);
	}
	build(false)
}

fn build_pcre_matcher(
	pattern: &str,
	ignore_case: bool,
	multiline: bool,
) -> Result<PcreMatcher, grep_pcre2::Error> {
	let mut builder = PcreMatcherBuilder::new();
	builder
		.caseless(ignore_case)
		.multi_line(multiline)
		.utf(true)
		.ucp(true)
		.jit_if_available(*PCRE2_JIT_ENABLED);
	builder.build(pattern)
}

fn build_matcher(
	pattern: &str,
	ignore_case: bool,
	multiline: bool,
) -> Result<CompiledMatcher, GrepError> {
	let sanitized = sanitize_braces(pattern);
	let regex_error = match build_regex_matcher(sanitized.as_ref(), ignore_case, multiline) {
		Ok(matcher) => return Ok(CompiledMatcher::Rust(matcher)),
		Err(error) => error,
	};
	match build_pcre_matcher(sanitized.as_ref(), ignore_case, multiline) {
		Ok(matcher) => Ok(CompiledMatcher::Pcre2(matcher)),
		Err(pcre2_error) => Err(GrepError::InvalidRegex {
			regex: Str::from(regex_error.to_string()),
			pcre2: Str::from(pcre2_error.to_string()),
		}),
	}
}

fn sanitize_braces(pattern: &str) -> Cow<'_, str> {
	let bytes = pattern.as_bytes();
	if !bytes.contains(&b'{') && !bytes.contains(&b'}') {
		return Cow::Borrowed(pattern);
	}
	let mut output = String::with_capacity(pattern.len() + 8);
	let mut modified = false;
	let mut index = 0;
	while index < bytes.len() {
		if bytes[index] == b'\\' && index + 1 < bytes.len() {
			output.push('\\');
			index += 1;
			let character = pattern[index..].chars().next().expect("non-empty suffix");
			output.push(character);
			index += character.len_utf8();
			if matches!(character, 'p' | 'P' | 'x' | 'u')
				&& index < bytes.len()
				&& bytes[index] == b'{'
			{
				if let Some(end) = find_braced_escape_end(bytes, index) {
					output.push_str(&pattern[index..=end]);
					index = end + 1;
				} else {
					output.push_str(&pattern[index..]);
					index = bytes.len();
				}
			}
			continue;
		}
		if bytes[index] == b'{' {
			if let Some(end) = find_valid_repetition(bytes, index) {
				output.push_str(&pattern[index..=end]);
				index = end + 1;
				continue;
			}
			output.push_str("\\{");
			index += 1;
			modified = true;
			continue;
		}
		if bytes[index] == b'}' {
			output.push_str("\\}");
			index += 1;
			modified = true;
			continue;
		}
		let character = pattern[index..].chars().next().expect("non-empty suffix");
		output.push(character);
		index += character.len_utf8();
	}
	if modified {
		Cow::Owned(output)
	} else {
		Cow::Borrowed(pattern)
	}
}

const fn find_valid_repetition(bytes: &[u8], start: usize) -> Option<usize> {
	let mut index = start + 1;
	if index >= bytes.len() || !bytes[index].is_ascii_digit() {
		return None;
	}
	while index < bytes.len() && bytes[index].is_ascii_digit() {
		index += 1;
	}
	if index >= bytes.len() {
		return None;
	}
	if bytes[index] == b'}' {
		return Some(index);
	}
	if bytes[index] != b',' {
		return None;
	}
	index += 1;
	while index < bytes.len() && bytes[index].is_ascii_digit() {
		index += 1;
	}
	if index < bytes.len() && bytes[index] == b'}' {
		Some(index)
	} else {
		None
	}
}

const fn find_braced_escape_end(bytes: &[u8], start: usize) -> Option<usize> {
	let mut index = start + 1;
	while index < bytes.len() {
		if bytes[index] == b'}' {
			return Some(index);
		}
		index += 1;
	}
	None
}

fn build_searcher(params: SearchParams) -> Searcher {
	let content_mode = params.mode == GrepOutputMode::Content;
	SearcherBuilder::new()
		.binary_detection(BinaryDetection::quit(b'\0'))
		.line_number(content_mode)
		.multi_line(params.multiline)
		.before_context(if content_mode {
			params.context_before as usize
		} else {
			0
		})
		.after_context(if content_mode {
			params.context_after as usize
		} else {
			0
		})
		.build()
}

fn per_file_params(params: SearchParams) -> SearchParams {
	let max_count = match params.mode {
		GrepOutputMode::Content => match (params.max_count, params.max_count_per_file) {
			(Some(global), Some(per_file)) => Some(global.min(per_file)),
			(global, per_file) => global.or(per_file),
		},
		GrepOutputMode::Count => None,
		GrepOutputMode::FilesWithMatches => Some(1),
	};
	SearchParams { max_count, ..params }
}

fn run_search(
	matcher: &CompiledMatcher,
	content: &[u8],
	params: SearchParams,
) -> io::Result<SearchResultInternal> {
	let mut searcher = build_searcher(params);
	run_search_slice(&mut searcher, matcher, content, params)
}

fn run_search_slice(
	searcher: &mut Searcher,
	matcher: &CompiledMatcher,
	content: &[u8],
	params: SearchParams,
) -> io::Result<SearchResultInternal> {
	let mut collector = MatchCollector::new(
		params.max_count,
		params.max_columns.map(|columns| columns as usize),
		params.mode == GrepOutputMode::Content,
	);
	searcher.search_slice(matcher, content, &mut collector)?;
	Ok(SearchResultInternal {
		matches:       collector.matches,
		match_count:   collector.match_count,
		limit_reached: collector.limit_reached,
	})
}

fn grep_file(
	path: &Path,
	display_path: Str,
	matcher: &CompiledMatcher,
	params: SearchParams,
	deadline: Deadline,
) -> Result<GrepResult, GrepError> {
	let mut worker = SearchWorker::new(params);
	let candidate = FileCandidate {
		path:     path.to_path_buf(),
		relative: display_path.as_str().to_owned(),
		mtime:    None,
		size:     std::fs::metadata(path)
			.ok()
			.map(|metadata| metadata.len() as f64),
	};
	let (search, skipped_oversized) =
		match search_one_file(&mut worker, matcher, &candidate, params, ReadPolicy::Full) {
			FileOutcome::Searched(search) => (Some(search), 0),
			FileOutcome::Defer => {
				match search_one_file(&mut worker, matcher, &candidate, params, ReadPolicy::Prefix) {
					FileOutcome::Searched(search) => (Some(search), 0),
					_ => (None, 1),
				}
			},
			_ => (None, 0),
		};
	deadline.check()?;
	Ok(search.map_or_else(
		|| GrepResult { skipped_oversized, ..GrepResult::default() },
		|search| aggregate_single(search, display_path, params, skipped_oversized),
	))
}

fn process_candidates(
	candidates: Vec<FileCandidate>,
	matcher: &CompiledMatcher,
	params: SearchParams,
	deadline: Deadline,
) -> Result<(Vec<FileSearchResult>, u64, u64), GrepError> {
	let file_params = per_file_params(params);
	let state = PassState::default();
	let (normal, oversized): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|candidate| {
		file_size_hint(candidate.size).is_none_or(|size| size <= MAX_FILE_BYTES)
	});
	state.deferred.lock().extend(oversized);

	let mut results = run_pass(&normal, matcher, file_params, ReadPolicy::Full, &state, deadline)?;
	let deferred = std::mem::take(&mut *state.deferred.lock());
	let content_budget_satisfied = params.mode == GrepOutputMode::Content
		&& params.max_count.is_some_and(|maximum| {
			results
				.iter()
				.map(|result| result.matches.len() as u64)
				.sum::<u64>()
				>= maximum
		});
	if !deferred.is_empty() && !content_budget_satisfied {
		results.extend(run_pass(
			&deferred,
			matcher,
			file_params,
			ReadPolicy::Prefix,
			&state,
			deadline,
		)?);
	}
	Ok((
		results,
		state.skipped_oversized.load(Ordering::Relaxed),
		state.files_searched.load(Ordering::Relaxed),
	))
}

fn run_pass(
	candidates: &[FileCandidate],
	matcher: &CompiledMatcher,
	params: SearchParams,
	policy: ReadPolicy,
	state: &PassState,
	deadline: Deadline,
) -> Result<Vec<FileSearchResult>, GrepError> {
	execute_candidates_init(
		candidates,
		|| SearchWorker::new(params),
		|worker, candidate| {
			deadline.check()?;
			handle_file(worker, matcher, candidate, params, policy, state);
			deadline.check()
		},
	)?;
	let mut results = std::mem::take(&mut *state.results.lock());
	results.sort_unstable_by(|left, right| left.relative_path.cmp(&right.relative_path));
	Ok(results)
}

fn handle_file(
	worker: &mut SearchWorker,
	matcher: &CompiledMatcher,
	candidate: &FileCandidate,
	params: SearchParams,
	policy: ReadPolicy,
	state: &PassState,
) {
	match search_one_file(worker, matcher, candidate, params, policy) {
		FileOutcome::Defer => state.deferred.lock().push(candidate.clone()),
		FileOutcome::SkippedOversized => {
			state.skipped_oversized.fetch_add(1, Ordering::Relaxed);
		},
		FileOutcome::Skipped => {},
		FileOutcome::Searched(search) => {
			state.files_searched.fetch_add(1, Ordering::Relaxed);
			if search.match_count > 0 {
				state.results.lock().push(FileSearchResult {
					relative_path: Str::from(candidate.relative.as_str()),
					matches:       search.matches,
					match_count:   search.match_count,
					limit_reached: search.limit_reached,
				});
			}
		},
	}
}

fn search_one_file(
	worker: &mut SearchWorker,
	matcher: &CompiledMatcher,
	candidate: &FileCandidate,
	params: SearchParams,
	policy: ReadPolicy,
) -> FileOutcome {
	let read = match policy {
		ReadPolicy::Full => read_file_bytes_with_size(
			&candidate.path,
			file_size_hint(candidate.size),
			&mut worker.buffer,
		),
		ReadPolicy::Prefix => read_file_prefix(&candidate.path, &mut worker.buffer),
	};
	match read {
		Ok(ReadFile::Read) => {},
		Ok(ReadFile::Oversized) => return FileOutcome::Defer,
		Ok(ReadFile::Skipped) => {
			return if policy == ReadPolicy::Prefix {
				FileOutcome::SkippedOversized
			} else {
				FileOutcome::Skipped
			};
		},
		Err(_) => {
			return if policy == ReadPolicy::Prefix {
				FileOutcome::SkippedOversized
			} else {
				FileOutcome::Skipped
			};
		},
	}
	let search = run_search_slice(&mut worker.searcher, matcher, &worker.buffer, params).unwrap_or(
		SearchResultInternal { matches: Vec::new(), match_count: 0, limit_reached: false },
	);
	FileOutcome::Searched(search)
}

fn read_file_bytes_with_size(
	path: &Path,
	size_hint: Option<u64>,
	buffer: &mut Vec<u8>,
) -> io::Result<ReadFile> {
	let file = match File::open(path) {
		Ok(file) => file,
		Err(error)
			if matches!(error.kind(), io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied) =>
		{
			return Ok(ReadFile::Skipped);
		},
		Err(error) => return Err(error),
	};
	let size = if let Some(size) = size_hint {
		size
	} else {
		let metadata = file.metadata()?;
		if !metadata.is_file() {
			return Ok(ReadFile::Skipped);
		}
		metadata.len()
	};
	if size > MAX_FILE_BYTES {
		return Ok(ReadFile::Oversized);
	}
	read_owned_prefix(file, FILE_CLASSIFICATION_READ_BYTES, size, buffer)?;
	if u64::try_from(buffer.len()).map_or(true, |length| length > MAX_FILE_BYTES) {
		return Ok(ReadFile::Oversized);
	}
	Ok(ReadFile::Read)
}

fn read_file_prefix(path: &Path, buffer: &mut Vec<u8>) -> io::Result<ReadFile> {
	let file = match File::open(path) {
		Ok(file) => file,
		Err(error)
			if matches!(error.kind(), io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied) =>
		{
			return Ok(ReadFile::Skipped);
		},
		Err(error) => return Err(error),
	};
	let metadata = file.metadata()?;
	if !metadata.is_file() {
		return Ok(ReadFile::Skipped);
	}
	let window = metadata.len().min(MAX_FILE_BYTES);
	read_owned_prefix(file, window, window, buffer)?;
	Ok(ReadFile::Read)
}

fn read_owned_prefix(
	mut file: File,
	limit: u64,
	capacity_hint: u64,
	buffer: &mut Vec<u8>,
) -> io::Result<()> {
	buffer.clear();
	let capacity = usize::try_from(capacity_hint.min(limit)).expect("bounded capacity fits usize");
	buffer.reserve(capacity);
	file.by_ref().take(limit).read_to_end(buffer)?;
	Ok(())
}

fn file_size_hint(size: Option<f64>) -> Option<u64> {
	size
		.filter(|size| size.is_finite() && *size >= 0.0 && *size <= u64::MAX as f64)
		.map(|size| size as u64)
}

fn aggregate_single(
	search: SearchResultInternal,
	path: Str,
	params: SearchParams,
	skipped_oversized: u32,
) -> GrepResult {
	let files_with_matches = u32::from(search.match_count > 0);
	let matches = match params.mode {
		GrepOutputMode::Content => search
			.matches
			.into_iter()
			.take(params.max_count.map_or(usize::MAX, |max| max as usize))
			.map(|matched| public_match(path.clone(), matched))
			.collect(),
		GrepOutputMode::Count | GrepOutputMode::FilesWithMatches if search.match_count > 0 => {
			vec![file_marker(path)]
		},
		GrepOutputMode::Count | GrepOutputMode::FilesWithMatches => Vec::new(),
	};
	let global_limit = params
		.max_count
		.is_some_and(|max| u64::try_from(matches.len()).unwrap_or(u64::MAX) >= max);
	GrepResult {
		matches,
		total_matches: clamp_u32(search.match_count),
		files_with_matches,
		files_searched: 1,
		limit_reached: search.limit_reached || global_limit,
		skipped_oversized,
	}
}

fn aggregate_results(
	results: Vec<FileSearchResult>,
	params: SearchParams,
	files_searched: u64,
	skipped_oversized: u64,
) -> GrepResult {
	let mut matches = Vec::new();
	let mut total_matches = 0u64;
	let mut files_with_matches = 0u64;
	let mut emitted = 0u64;
	let mut limit_reached = false;

	for result in results {
		if result.match_count == 0 {
			continue;
		}
		total_matches = total_matches.saturating_add(result.match_count);
		files_with_matches = files_with_matches.saturating_add(1);
		match params.mode {
			GrepOutputMode::Content => {
				for matched in result.matches {
					if params.max_count.is_some_and(|max| emitted >= max) {
						limit_reached = true;
						break;
					}
					matches.push(public_match(result.relative_path.clone(), matched));
					emitted = emitted.saturating_add(1);
				}
				limit_reached |= result.limit_reached;
			},
			GrepOutputMode::Count | GrepOutputMode::FilesWithMatches => {
				if params.max_count.is_some_and(|max| emitted >= max) {
					limit_reached = true;
					continue;
				}
				matches.push(file_marker(result.relative_path));
				emitted = emitted.saturating_add(match params.mode {
					GrepOutputMode::Count => result.match_count,
					GrepOutputMode::FilesWithMatches => 1,
					GrepOutputMode::Content => unreachable!(),
				});
			},
		}
	}
	if params.max_count.is_some_and(|max| emitted >= max) {
		limit_reached = true;
	}
	GrepResult {
		matches,
		total_matches: clamp_u32(total_matches),
		files_with_matches: clamp_u32(files_with_matches),
		files_searched: clamp_u32(files_searched),
		limit_reached,
		skipped_oversized: clamp_u32(skipped_oversized),
	}
}

fn public_match(path: Str, matched: CollectedMatch) -> GrepMatch {
	GrepMatch {
		path,
		line_number: clamp_u32(matched.line_number),
		line: matched.line,
		truncated: matched.truncated,
		context_before: matched.context_before,
		context_after: matched.context_after,
	}
}

fn file_marker(path: Str) -> GrepMatch {
	GrepMatch {
		path,
		line_number: 0,
		line: Str::new(""),
		truncated: false,
		context_before: SmallVec::new(),
		context_after: SmallVec::new(),
	}
}

fn truncate_line(line: Str, max_columns: Option<usize>) -> (Str, bool) {
	match max_columns {
		Some(maximum) if line.len() > maximum => {
			let mut boundary = maximum.saturating_sub(3).min(line.len());
			while !line.as_str().is_char_boundary(boundary) {
				boundary -= 1;
			}
			(Str::from(format!("{}...", &line.as_str()[..boundary])), true)
		},
		_ => (line, false),
	}
}

fn bytes_to_trimmed_str(bytes: &[u8]) -> Str {
	match std::str::from_utf8(bytes) {
		Ok(text) => Str::from(text.trim_end()),
		Err(_) => Str::from(String::from_utf8_lossy(bytes).trim_end()),
	}
}

const fn clamp_u32(value: u64) -> u32 {
	if value > u32::MAX as u64 {
		u32::MAX
	} else {
		value as u32
	}
}

fn search_error(error: io::Error) -> GrepError {
	GrepError::Search { message: Str::from(error.to_string()) }
}

#[cfg(test)]
mod tests {
	use std::{
		fs,
		sync::atomic::{AtomicU64, Ordering},
	};

	use super::*;

	fn options(pattern: &str) -> GrepOptions {
		GrepOptions { pattern: Str::from(pattern), timeout_ms: None, ..GrepOptions::default() }
	}

	struct TempDir(PathBuf);

	impl TempDir {
		fn new() -> Self {
			static NEXT: AtomicU64 = AtomicU64::new(0);
			let path = std::env::temp_dir().join(format!(
				"omp-grep-{}-{}",
				std::process::id(),
				NEXT.fetch_add(1, Ordering::Relaxed)
			));
			fs::create_dir_all(&path).expect("create temp directory");
			Self(path)
		}
	}

	impl Drop for TempDir {
		fn drop(&mut self) {
			let _ = fs::remove_dir_all(&self.0);
		}
	}

	#[test]
	fn in_memory_search_collects_context_and_truncates() {
		let mut options = options("needle");
		options.path = Str::new("memory");
		options.context_before = 1;
		options.context_after = 1;
		options.max_columns = Some(9);
		let result = search(b"before line\nneedle payload\nafter\n", &options).unwrap();
		assert_eq!(result.total_matches, 1);
		assert_eq!(result.matches[0].line.as_str(), "needle...");
		assert!(result.matches[0].truncated);
		assert_eq!(result.matches[0].context_before[0].line.as_str(), "before...");
		assert_eq!(result.matches[0].context_after[0].line.as_str(), "after");
	}

	#[test]
	fn pcre2_fallback_handles_lookaround() {
		let matcher = build_matcher(r"foo(?=bar)", false, false).unwrap();
		assert!(matches!(matcher, CompiledMatcher::Pcre2(_)));
		let result = search(b"foobar\nfoobaz\n", &options(r"foo(?=bar)")).unwrap();
		assert_eq!(result.total_matches, 1);
	}

	#[test]
	fn invalid_pattern_preserves_both_engine_diagnostics() {
		let error = search(b"text", &options("(")).unwrap_err();
		assert!(matches!(error, GrepError::InvalidRegex { .. }));
	}

	#[test]
	fn directory_search_counts_normal_and_oversized_files() {
		let root = TempDir::new();
		fs::write(root.0.join("small.txt"), "needle\n").unwrap();
		let mut large = Vec::with_capacity(MAX_FILE_BYTES as usize + 1);
		large.extend_from_slice(b"needle in prefix\n");
		large.resize(MAX_FILE_BYTES as usize + 1, b'x');
		fs::write(root.0.join("large.txt"), large).unwrap();
		let mut options = options("needle");
		options.path = Str::from(root.0.to_string_lossy());
		let result = grep(&options).unwrap();
		assert_eq!(result.files_searched, 2);
		assert_eq!(result.files_with_matches, 2);
		assert_eq!(result.skipped_oversized, 0);
		assert_eq!(result.matches[0].path.as_str(), "small.txt");
		assert_eq!(result.matches[1].path.as_str(), "large.txt");
	}

	#[test]
	fn nul_marks_binary_content() {
		let result = search(b"needle\0needle\n", &options("needle")).unwrap();
		assert_eq!(result.total_matches, 0);
		assert!(result.matches.is_empty());
	}

	#[test]
	fn literal_backslash_n_infers_multiline_mode() {
		let result = search(b"alpha\nbeta\n", &options(r"alpha\nbeta")).unwrap();
		assert_eq!(result.total_matches, 1);
	}

	#[test]
	fn directory_caps_are_applied_after_path_ordering() {
		let root = TempDir::new();
		fs::write(root.0.join("b.txt"), "needle\nneedle\n").unwrap();
		fs::write(root.0.join("a.txt"), "needle\nneedle\n").unwrap();
		let mut options = options("needle");
		options.path = Str::from(root.0.to_string_lossy());
		options.max_count_per_file = Some(1);
		options.max_count = Some(2);
		let result = grep(&options).unwrap();
		assert_eq!(result.matches.len(), 2);
		assert_eq!(result.matches[0].path.as_str(), "a.txt");
		assert_eq!(result.matches[1].path.as_str(), "b.txt");
		assert!(result.limit_reached);
	}

	#[test]
	fn satisfied_budget_skips_the_oversized_pass() {
		let root = TempDir::new();
		fs::write(root.0.join("small.txt"), "needle\n").unwrap();
		let mut large = Vec::with_capacity(MAX_FILE_BYTES as usize + 1);
		large.extend_from_slice(b"needle in prefix\n");
		large.resize(MAX_FILE_BYTES as usize + 1, b'x');
		fs::write(root.0.join("large.txt"), large).unwrap();
		let mut options = options("needle");
		options.path = Str::from(root.0.to_string_lossy());
		options.max_count = Some(1);
		let result = grep(&options).unwrap();
		assert_eq!(result.files_searched, 1);
		assert_eq!(result.matches.len(), 1);
		assert_eq!(result.matches[0].path.as_str(), "small.txt");
	}

	#[test]
	fn zero_timeout_is_typed() {
		let mut options = options("needle");
		options.timeout_ms = Some(0);
		assert!(matches!(search(b"needle", &options), Err(GrepError::Timeout { timeout_ms: 0 })));
	}
}
