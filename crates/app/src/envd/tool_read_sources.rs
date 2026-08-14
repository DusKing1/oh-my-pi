//! Resource-owned local and special-source I/O for `read@1`.

use std::{
	borrow::Cow,
	io,
	path::{Component, Path, PathBuf},
	time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bytes::{Bytes, BytesMut};
use futures::StreamExt as _;
use http::{
	HeaderMap, HeaderName, HeaderValue, StatusCode,
	header::{ACCEPT, ACCEPT_ENCODING, ACCEPT_LANGUAGE, CONTENT_TYPE, RETRY_AFTER, USER_AGENT},
};
use omp_core::Str;
use omp_hashline::RevisionToken;
use omp_tools::read::{
	DirectoryEntry, DirectorySource, Fault, ReadLease, ReadSources, SNAPSHOT_MAX_BYTES,
	SnapshotRecord, SourceKind, SourceStat,
	web::types::{HttpClient, HttpRequest, HttpResponse, MAX_BYTES, USER_AGENTS, WebError},
};
use omp_walker::{FileType, WalkDetail, WalkOrder, WalkRequest};
use tokio::io::AsyncReadExt as _;
use tokio_util::sync::CancellationToken;
use url::Url;

use super::{
	docs::{DocumentHost, DocumentLease},
	tool_document::{read_document_metadata, read_whole, resolve_read_document},
	workspace::WorkspaceHost,
};

const MAX_REDIRECTS: usize = 20;
const MAX_RETRY_AFTER: Duration = Duration::from_secs(10);
const DEFAULT_RETRY_AFTER: Duration = Duration::from_secs(1);
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
struct SystemHttpClient {
	inner: wreq::Client,
}

impl SystemHttpClient {
	fn new() -> Self {
		let inner = wreq::Client::builder()
			.redirect(wreq::redirect::Policy::limited(MAX_REDIRECTS))
			.referer(false)
			.build()
			.expect("build read HTTP client");
		Self { inner }
	}

	async fn request(&self, request: HttpRequest) -> Result<HttpResponse, WebError> {
		let mut authored_url = Url::parse(&request.url)
			.map_err(|error| WebError::InvalidUrl(error.to_string().into()))?;
		validate_http_url(&authored_url)?;
		authored_url.set_fragment(None);
		tokio::time::timeout(HTTP_TIMEOUT, self.request_with_retries(authored_url, request))
			.await
			.map_err(|_| WebError::request("request timed out after 30s"))?
	}

	async fn request_with_retries(
		&self,
		authored_url: Url,
		request: HttpRequest,
	) -> Result<HttpResponse, WebError> {
		let max_bytes = request.max_bytes.min(MAX_BYTES);
		let caller_headers = parse_request_headers(&request.headers)?;
		let mut retried_429 = false;
		let mut last_error = None;

		for (attempt, user_agent) in USER_AGENTS.iter().enumerate() {
			loop {
				let response = match self
					.inner
					.get(authored_url.as_str())
					.headers(request_headers(user_agent, &caller_headers))
					.send()
					.await
				{
					Ok(response) => response,
					Err(error) => {
						last_error = Some(WebError::request(error.to_string()));
						break;
					},
				};
				if response.status() == StatusCode::TOO_MANY_REQUESTS && !retried_429 {
					retried_429 = true;
					let delay = retry_after(response.headers().get(RETRY_AFTER));
					drop(response);
					tokio::time::sleep(delay).await;
					continue;
				}

				let final_url = Str::from(response.uri().to_string());
				let status = response.status().as_u16();
				let headers = response.headers().clone();
				let body = match read_bounded(response, max_bytes).await {
					Ok(body) => body,
					Err(error @ WebError::ResponseTooLarge { .. }) => return Err(error),
					Err(error) => {
						last_error = Some(error);
						break;
					},
				};
				if is_bot_blocked(status, &headers, &body) && attempt + 1 < USER_AGENTS.len() {
					break;
				}
				return Ok(build_http_response(final_url, status, headers, body));
			}
		}

		Err(last_error.unwrap_or_else(|| WebError::request("HTTP request failed")))
	}
}

impl std::fmt::Debug for SystemHttpClient {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter.write_str("SystemHttpClient(..)")
	}
}

impl Default for SystemHttpClient {
	fn default() -> Self {
		Self::new()
	}
}

impl HttpClient for SystemHttpClient {
	async fn get(&self, request: HttpRequest) -> Result<HttpResponse, WebError> {
		self.request(request).await
	}
}

fn validate_http_url(url: &Url) -> Result<(), WebError> {
	if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
		return Err(WebError::InvalidUrl(url.as_str().into()));
	}
	Ok(())
}

fn parse_request_headers(headers: &[(Str, Str)]) -> Result<HeaderMap, WebError> {
	let mut parsed = HeaderMap::with_capacity(headers.len());
	for (name, value) in headers {
		let name = HeaderName::from_bytes(name.as_bytes())
			.map_err(|error| WebError::request(format!("invalid request header '{name}': {error}")))?;
		let value = HeaderValue::from_str(value)
			.map_err(|error| WebError::request(format!("invalid request header value: {error}")))?;
		parsed.insert(name, value);
	}
	Ok(parsed)
}

fn request_headers(user_agent: &'static str, caller: &HeaderMap) -> HeaderMap {
	let mut headers = HeaderMap::with_capacity(caller.len() + 4);
	headers.insert(USER_AGENT, HeaderValue::from_static(user_agent));
	headers.insert(
		ACCEPT,
		HeaderValue::from_static("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
	);
	headers.insert(ACCEPT_LANGUAGE, HeaderValue::from_static("en-US,en;q=0.5"));
	headers.insert(ACCEPT_ENCODING, HeaderValue::from_static("identity"));
	for (name, value) in caller {
		headers.insert(name.clone(), value.clone());
	}
	headers
}

async fn read_bounded(response: wreq::Response, max_bytes: usize) -> Result<Bytes, WebError> {
	let content_length = response.content_length();
	if content_length.is_some_and(|length| length > u64::try_from(max_bytes).unwrap_or(u64::MAX)) {
		return Err(WebError::ResponseTooLarge { max_bytes });
	}
	let initial_capacity = content_length
		.and_then(|length| usize::try_from(length).ok())
		.unwrap_or_default()
		.min(max_bytes);
	let mut bytes = BytesMut::with_capacity(initial_capacity);
	let mut stream = response.bytes_stream();
	while let Some(chunk) = stream.next().await {
		let chunk = chunk.map_err(|error| WebError::request(error.to_string()))?;
		if bytes.len().saturating_add(chunk.len()) > max_bytes {
			return Err(WebError::ResponseTooLarge { max_bytes });
		}
		bytes.extend_from_slice(&chunk);
	}
	Ok(bytes.freeze())
}

fn build_http_response(
	final_url: Str,
	status: u16,
	headers: HeaderMap,
	body: Bytes,
) -> HttpResponse {
	let content_type = headers
		.get(CONTENT_TYPE)
		.and_then(|value| value.to_str().ok())
		.and_then(|value| value.split(';').next())
		.map(str::trim)
		.filter(|value| !value.is_empty())
		.map(|value| value.to_ascii_lowercase().into());
	let headers = headers
		.iter()
		.map(|(name, value)| {
			(
				Str::from(name.as_str()),
				Str::from(String::from_utf8_lossy(value.as_bytes()).into_owned()),
			)
		})
		.collect();
	HttpResponse { final_url, status, content_type, headers, body }
}

fn is_bot_blocked(status: u16, headers: &HeaderMap, body: &[u8]) -> bool {
	if status != 403 && status != 503 {
		return false;
	}
	let content = decode_response_text(headers, body);
	["cloudflare", "captcha", "challenge", "blocked", "access denied", "bot detection"]
		.iter()
		.any(|marker| contains_ascii_case_insensitive(content.as_bytes(), marker.as_bytes()))
}

fn decode_response_text<'a>(headers: &HeaderMap, body: &'a [u8]) -> Cow<'a, str> {
	let label = headers
		.get(CONTENT_TYPE)
		.and_then(|value| value.to_str().ok())
		.and_then(charset_from_content_type)
		.or_else(|| charset_from_meta(body));
	let encoding = label
		.as_deref()
		.and_then(|label| encoding_rs::Encoding::for_label(label.as_bytes()))
		.unwrap_or(encoding_rs::UTF_8);
	encoding.decode(body).0
}

fn charset_from_content_type(content_type: &str) -> Option<String> {
	content_type.split(';').skip(1).find_map(|parameter| {
		let (name, value) = parameter.split_once('=')?;
		name.trim().eq_ignore_ascii_case("charset").then(|| {
			value
				.trim()
				.trim_matches(|character| character == '"' || character == '\'')
				.to_owned()
		})
	})
}

fn charset_from_meta(body: &[u8]) -> Option<String> {
	let prefix = &body[..body.len().min(2048)];
	let lower = prefix
		.iter()
		.map(u8::to_ascii_lowercase)
		.collect::<Vec<_>>();
	let mut offset = 0;
	while let Some(relative) = find_bytes(&lower[offset..], b"<meta") {
		let start = offset + relative + 5;
		let end = lower[start..]
			.iter()
			.position(|byte| *byte == b'>')
			.map_or(lower.len(), |relative| start + relative);
		if let Some(relative) = find_bytes(&lower[start..end], b"charset") {
			let mut cursor = start + relative + b"charset".len();
			while lower.get(cursor).is_some_and(u8::is_ascii_whitespace) {
				cursor += 1;
			}
			if lower.get(cursor) != Some(&b'=') {
				offset = end.saturating_add(1);
				continue;
			}
			cursor += 1;
			while lower.get(cursor).is_some_and(u8::is_ascii_whitespace) {
				cursor += 1;
			}
			if matches!(lower.get(cursor), Some(b'"' | b'\'')) {
				cursor += 1;
			}
			let label_start = cursor;
			while lower
				.get(cursor)
				.is_some_and(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
			{
				cursor += 1;
			}
			if cursor > label_start {
				return String::from_utf8(lower[label_start..cursor].to_vec()).ok();
			}
		}
		offset = end.saturating_add(1);
	}
	None
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
	haystack
		.windows(needle.len())
		.position(|window| window == needle)
}

fn contains_ascii_case_insensitive(haystack: &[u8], needle: &[u8]) -> bool {
	haystack
		.windows(needle.len())
		.any(|window| window.eq_ignore_ascii_case(needle))
}

fn retry_after(value: Option<&HeaderValue>) -> Duration {
	let Some(value) = value.and_then(|value| value.to_str().ok()) else {
		return DEFAULT_RETRY_AFTER;
	};
	if let Ok(seconds) = value.trim().parse::<f64>()
		&& seconds.is_finite()
	{
		let seconds = seconds.clamp(0.0, MAX_RETRY_AFTER.as_secs_f64());
		return Duration::from_secs_f64(seconds);
	}
	let Some(time) = parse_http_date(value) else {
		return DEFAULT_RETRY_AFTER;
	};
	time
		.duration_since(SystemTime::now())
		.unwrap_or(Duration::ZERO)
		.min(MAX_RETRY_AFTER)
}

fn parse_http_date(value: &str) -> Option<SystemTime> {
	let fields = value.split_ascii_whitespace().collect::<Vec<_>>();
	if fields.len() != 6 || !fields[0].ends_with(',') || !fields[5].eq_ignore_ascii_case("GMT") {
		return None;
	}
	let day = fields[1].parse::<u32>().ok()?;
	let month = match fields[2].to_ascii_lowercase().as_str() {
		"jan" => 1,
		"feb" => 2,
		"mar" => 3,
		"apr" => 4,
		"may" => 5,
		"jun" => 6,
		"jul" => 7,
		"aug" => 8,
		"sep" => 9,
		"oct" => 10,
		"nov" => 11,
		"dec" => 12,
		_ => return None,
	};
	let year = fields[3].parse::<i64>().ok()?;
	let mut time = fields[4].split(':');
	let hour = time.next()?.parse::<u32>().ok()?;
	let minute = time.next()?.parse::<u32>().ok()?;
	let second = time.next()?.parse::<u32>().ok()?;
	let max_day = match month {
		2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
		2 => 28,
		4 | 6 | 9 | 11 => 30,
		_ => 31,
	};
	if time.next().is_some()
		|| !(1601..=9999).contains(&year)
		|| !(1..=max_day).contains(&day)
		|| hour > 23
		|| minute > 59
		|| second > 60
	{
		return None;
	}
	let days = days_from_civil(year, month, day);
	let seconds = days
		.checked_mul(86_400)?
		.checked_add(i64::from(hour) * 3_600)?
		.checked_add(i64::from(minute) * 60)?
		.checked_add(i64::from(second.min(59)))?;
	(seconds >= 0).then(|| UNIX_EPOCH + Duration::from_secs(seconds as u64))
}

fn days_from_civil(mut year: i64, month: u32, day: u32) -> i64 {
	if month <= 2 {
		year -= 1;
	}
	let era = year.div_euclid(400);
	let year_of_era = year - era * 400;
	let adjusted_month = i64::from(month) + if month > 2 { -3 } else { 9 };
	let day_of_year = (153 * adjusted_month + 2) / 5 + i64::from(day) - 1;
	let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
	era * 146_097 + day_of_era - 719_468
}

/// App-owned source adapter joining document leases and canonical workspace
/// I/O.
#[derive(Clone, Debug)]
pub struct ReadSourceAdapter {
	documents: DocumentHost,
	workspace: WorkspaceHost,
	http:      SystemHttpClient,
}

impl ReadSourceAdapter {
	/// Creates a source adapter over one project environment's shared resources.
	pub(crate) fn new(documents: DocumentHost, workspace: WorkspaceHost) -> Self {
		Self { documents, workspace, http: SystemHttpClient::new() }
	}

	async fn stat_path(&self, authored: &str) -> Result<SourceStat, Fault> {
		let candidate = resolve_authored_path(self.workspace.root(), authored);
		let authored_metadata = tokio::fs::symlink_metadata(&candidate)
			.await
			.map_err(|error| source_io("stat", authored, error))?;
		let canonical = tokio::fs::canonicalize(&candidate)
			.await
			.map_err(|error| source_io("canonicalize", authored, error))?;
		let metadata = tokio::fs::metadata(&canonical)
			.await
			.map_err(|error| source_io("stat", authored, error))?;
		let kind = if authored_metadata.file_type().is_symlink() {
			SourceKind::Symlink
		} else if metadata.is_dir() {
			SourceKind::Directory
		} else {
			SourceKind::File
		};
		let canonical_path = utf8_path(&canonical)?;
		let display_path = display_path(self.workspace.root(), &canonical)?;
		Ok(SourceStat {
			canonical_path,
			display_path,
			kind,
			byte_len: metadata.len(),
			modified_ms: modified_ms(&metadata),
		})
	}
}

impl HttpClient for ReadSourceAdapter {
	async fn get(&self, request: HttpRequest) -> Result<HttpResponse, WebError> {
		self.http.get(request).await
	}
}

/// One app-owned lease whose bytes remain stable until drop.
#[derive(Debug)]
pub struct ReadDocumentLease {
	backing:        ReadLeaseBacking,
	revision:       Str,
	canonical_path: Str,
}

#[derive(Debug)]
enum ReadLeaseBacking {
	Document { host: DocumentHost, lease: DocumentLease },
	File(Bytes),
}

impl ReadLease for ReadDocumentLease {
	fn revision(&self) -> &Str {
		&self.revision
	}

	fn canonical_path(&self) -> &Str {
		&self.canonical_path
	}

	async fn read_all(&self) -> Result<Bytes, Fault> {
		match &self.backing {
			ReadLeaseBacking::Document { host, lease } => read_whole(host, lease)
				.await
				.map_err(|error| Fault::source(error.to_string())),
			ReadLeaseBacking::File(bytes) => Ok(bytes.clone()),
		}
	}
}

async fn open_filesystem_lease(canonical_path: Str) -> Result<ReadDocumentLease, Fault> {
	let bytes = tokio::fs::read(canonical_path.as_str())
		.await
		.map(Bytes::from)
		.map_err(|error| source_io("read", &canonical_path, error))?;
	let revision = Str::from(format!("fs:{}", blake3::hash(&bytes).to_hex()));
	Ok(ReadDocumentLease { backing: ReadLeaseBacking::File(bytes), revision, canonical_path })
}

impl ReadSources for ReadSourceAdapter {
	type Lease = ReadDocumentLease;

	async fn stat(&self, path: Str) -> Result<SourceStat, Fault> {
		self.stat_path(&path).await
	}

	async fn resolve_suffix(&self, path: Str) -> Result<Option<SourceStat>, Fault> {
		let Some(suffix) = normalized_suffix(&path) else {
			return Ok(None);
		};
		let request = self
			.workspace
			.request()
			.hidden(true)
			.gitignore(true)
			.skip_git(true)
			.skip_node_modules(true)
			.detail(WalkDetail::Minimal)
			.order(WalkOrder::Path)
			.depth(1, usize::MAX);
		let deadline = Instant::now() + Duration::from_secs(5);
		let Ok(outcome) = request.collect_with_heartbeat(|| {
			(Instant::now() < deadline)
				.then_some(())
				.ok_or("suffix resolution timed out")
		}) else {
			return Ok(None);
		};
		let mut matched = None;
		for entry in outcome.entries {
			if path_has_suffix(&entry.path, &suffix) {
				if matched.is_some() {
					return Ok(None);
				}
				matched = Some(entry.path);
			}
		}
		let Some(relative) = matched else {
			return Ok(None);
		};
		let absolute = self.workspace.root().join(&relative);
		let absolute = utf8_path(&absolute)?;
		let Ok(mut stat) = self.stat_path(&absolute).await else {
			return Ok(None);
		};
		stat.display_path = Str::from(relative);
		Ok(Some(stat))
	}

	async fn open(&self, path: Str) -> Result<Self::Lease, Fault> {
		let stat = self.stat_path(&path).await?;
		if !Path::new(stat.canonical_path.as_str()).starts_with(self.workspace.root()) {
			return open_filesystem_lease(stat.canonical_path).await;
		}
		let resolved =
			resolve_read_document(&self.documents, &stat.canonical_path).map_err(Fault::source)?;
		let cancel = CancellationToken::new();
		let lease = DocumentHost::open(&self.documents, resolved.uri, None, &cancel)
			.await
			.map_err(|error| Fault::source(error.to_string()))?;
		let (revision, canonical_path) =
			read_document_metadata(lease.head()).map_err(Fault::source)?;
		Ok(ReadDocumentLease {
			backing: ReadLeaseBacking::Document { host: self.documents.clone(), lease },
			revision,
			canonical_path,
		})
	}

	async fn read_bytes(&self, path: Str) -> Result<Bytes, Fault> {
		tokio::fs::read(path.as_str())
			.await
			.map(Bytes::from)
			.map_err(|error| source_io("read", &path, error))
	}

	async fn read_prefix(&self, path: Str, max_bytes: usize) -> Result<Bytes, Fault> {
		if max_bytes == 0 {
			return Ok(Bytes::new());
		}
		let file = tokio::fs::File::open(path.as_str())
			.await
			.map_err(|error| source_io("read", &path, error))?;
		let mut prefix = Vec::with_capacity(max_bytes);
		file
			.take(u64::try_from(max_bytes).unwrap_or(u64::MAX))
			.read_to_end(&mut prefix)
			.await
			.map_err(|error| source_io("read", &path, error))?;
		Ok(Bytes::from(prefix))
	}

	async fn list_directory(&self, path: Str, max_depth: usize) -> Result<DirectorySource, Fault> {
		let root = PathBuf::from(path.as_str());
		let request = WalkRequest::new(root.clone())
			.hidden(true)
			.gitignore(false)
			.skip_git(true)
			.skip_node_modules(true)
			.detail(WalkDetail::Full)
			.order(WalkOrder::Path)
			.emit_root(false)
			.depth(1, max_depth);
		let outcome = if root.starts_with(self.workspace.root()) {
			self
				.workspace
				.walk(&request, &CancellationToken::new())
				.map_err(|error| Fault::source(format!("Cannot read directory: {error}")))?
		} else {
			request
				.collect()
				.map_err(|error| Fault::source(format!("Cannot read directory: {error}")))?
		};
		let entries = outcome
			.entries
			.into_iter()
			.map(|entry| DirectoryEntry {
				path:        Str::from(entry.path),
				kind:        walker_kind(entry.file_type),
				byte_len:    entry.size.map_or(0, float_to_u64),
				modified_ms: entry.mtime.map(float_to_u64),
			})
			.collect();
		Ok(DirectorySource {
			root: utf8_path(&root)?,
			entries,
			truncated: outcome.stats.limited_entries != 0,
		})
	}

	fn record_snapshot(&self, record: SnapshotRecord) -> Result<Option<Str>, Fault> {
		if record.bytes.len() > SNAPSHOT_MAX_BYTES {
			return Ok(None);
		}
		let revision = RevisionToken::new(record.revision.as_bytes());
		let seen = record
			.seen
			.into_iter()
			.flat_map(|range| range.start_line..=range.end_line)
			.filter_map(|line| usize::try_from(line).ok());
		Ok(self
			.documents
			.snapshot_store()
			.lock()
			.record(record.path, revision, record.bytes, seen)
			.ok())
	}
}

fn resolve_authored_path(root: &Path, authored: &str) -> PathBuf {
	let expanded = if authored == "~" {
		std::env::var_os("HOME").map(PathBuf::from)
	} else if let Some(rest) = authored.strip_prefix("~/") {
		std::env::var_os("HOME").map(|home| PathBuf::from(home).join(rest))
	} else {
		None
	};
	let path = expanded.unwrap_or_else(|| PathBuf::from(authored));
	if path.is_absolute() {
		path
	} else {
		root.join(path)
	}
}

fn display_path(root: &Path, canonical: &Path) -> Result<Str, Fault> {
	if let Ok(relative) = canonical.strip_prefix(root) {
		return if relative.as_os_str().is_empty() {
			Ok(Str::new_static("."))
		} else {
			utf8_slash_path(relative)
		};
	}
	if let Some(home) = std::env::var_os("HOME").map(PathBuf::from)
		&& let Ok(relative) = canonical.strip_prefix(home)
	{
		let suffix = utf8_slash_path(relative)?;
		return Ok(if suffix.is_empty() {
			Str::new_static("~")
		} else {
			Str::from(format!("~/{suffix}"))
		});
	}
	utf8_path(canonical)
}

fn utf8_path(path: &Path) -> Result<Str, Fault> {
	path
		.to_str()
		.map(Str::new)
		.ok_or_else(|| Fault::source("Local path is not valid UTF-8"))
}

fn utf8_slash_path(path: &Path) -> Result<Str, Fault> {
	let mut output = String::new();
	for component in path.components() {
		let value = match component {
			Component::Normal(value) => value
				.to_str()
				.ok_or_else(|| Fault::source("Local path is not valid UTF-8"))?,
			Component::CurDir => ".",
			Component::ParentDir => "..",
			Component::RootDir | Component::Prefix(_) => continue,
		};
		if !output.is_empty() {
			output.push('/');
		}
		output.push_str(value);
	}
	Ok(Str::from(output))
}

fn normalized_suffix(path: &str) -> Option<String> {
	let normalized = path.replace('\\', "/");
	let normalized = normalized
		.strip_prefix("./")
		.unwrap_or(&normalized)
		.trim_end_matches('/')
		.to_owned();
	(!normalized.is_empty() && !Path::new(&normalized).is_absolute()).then_some(normalized)
}

fn path_has_suffix(candidate: &str, suffix: &str) -> bool {
	candidate == suffix
		|| candidate
			.strip_suffix(suffix)
			.is_some_and(|prefix| prefix.ends_with('/'))
}

fn modified_ms(metadata: &std::fs::Metadata) -> Option<u64> {
	metadata
		.modified()
		.ok()?
		.duration_since(UNIX_EPOCH)
		.ok()
		.and_then(|duration| u64::try_from(duration.as_millis()).ok())
}

const fn walker_kind(kind: FileType) -> SourceKind {
	match kind {
		FileType::File => SourceKind::File,
		FileType::Dir => SourceKind::Directory,
		FileType::Symlink => SourceKind::Symlink,
	}
}

fn float_to_u64(value: f64) -> u64 {
	if value.is_finite() && value > 0.0 {
		value.min(u64::MAX as f64) as u64
	} else {
		0
	}
}

fn source_io(action: &str, path: &str, error: io::Error) -> Fault {
	Fault::source(format!("Cannot {action} '{path}': {error}"))
}

#[cfg(test)]
#[path = "tool_read_sources_tests.rs"]
mod tests;

#[cfg(test)]
mod external_path_tests {
	use super::*;

	#[tokio::test]
	async fn absolute_filesystem_lease_keeps_opened_bytes_pinned() {
		let sandbox = tempfile::tempdir().expect("sandbox");
		let path = sandbox.path().join("plain.txt");
		std::fs::write(&path, b"before").expect("write file");
		let canonical = std::fs::canonicalize(&path).expect("canonical file");
		let lease = open_filesystem_lease(utf8_path(&canonical).expect("UTF-8 path"))
			.await
			.expect("open external lease");
		std::fs::write(&path, b"after").expect("replace file");
		assert_eq!(lease.read_all().await.expect("read pinned bytes"), Bytes::from_static(b"before"));
	}

	#[tokio::test]
	async fn parent_relative_external_path_resolves_to_filesystem_lease() {
		let sandbox = tempfile::tempdir().expect("sandbox");
		let root = sandbox.path().join("root");
		std::fs::create_dir(&root).expect("workspace root");
		let path = sandbox.path().join("outside.txt");
		std::fs::write(&path, b"outside").expect("write file");
		let authored = resolve_authored_path(&root, "../outside.txt");
		let canonical = std::fs::canonicalize(authored).expect("canonical file");
		let lease = open_filesystem_lease(utf8_path(&canonical).expect("UTF-8 path"))
			.await
			.expect("open parent-relative lease");
		assert_eq!(
			lease.read_all().await.expect("read pinned bytes"),
			Bytes::from_static(b"outside")
		);
	}
}
