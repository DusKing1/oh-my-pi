//! Model-facing behavioral contracts for pi-compatible `read@1`.

use std::{
	collections::{HashMap, VecDeque},
	future::{Future, ready},
	path::{Path, PathBuf},
	sync::{
		Arc,
		atomic::{AtomicU64, Ordering},
	},
};

use bytes::Bytes;
use futures::StreamExt as _;
use omp_core::Str;
use omp_tool::{BlobRef, Ev, IncomingParams, Outcome, Part, PromptCaps, Tool};
use serde_json::json;
use omp_tools::read::{
	self, DirectoryEntry, DirectorySource, Fault, ReadBlobs, ReadLease, ReadSources, SnapshotRecord,
	SourceKind, SourceStat,
	web::types::{HttpClient, HttpRequest, HttpResponse, WebError},
};
use parking_lot::Mutex;

#[derive(Clone)]
struct FileSource {
	stat:     SourceStat,
	bytes:    Bytes,
	revision: Str,
}

#[derive(Clone, Default)]
struct Sources {
	files:     Arc<Mutex<HashMap<String, FileSource>>>,
	dirs:      Arc<Mutex<HashMap<String, (SourceStat, DirectorySource)>>>,
	suffixes:  Arc<Mutex<HashMap<String, SourceStat>>>,
	snapshots: Arc<Mutex<Vec<SnapshotRecord>>>,
	responses: Arc<Mutex<VecDeque<Result<HttpResponse, WebError>>>>,
}

#[derive(Clone)]
struct Lease {
	canonical_path: Str,
	revision:       Str,
	bytes:          Bytes,
}

impl ReadLease for Lease {
	fn revision(&self) -> &Str {
		&self.revision
	}

	fn canonical_path(&self) -> &Str {
		&self.canonical_path
	}

	fn read_all(&self) -> impl Future<Output = Result<Bytes, Fault>> + Send + '_ {
		ready(Ok(self.bytes.clone()))
	}
}

impl ReadSources for Sources {
	type Lease = Lease;

	fn stat(&self, path: Str) -> impl Future<Output = Result<SourceStat, Fault>> + Send + '_ {
		let result = self
			.files
			.lock()
			.get(path.as_str())
			.map(|source| source.stat.clone())
			.or_else(|| {
				self
					.dirs
					.lock()
					.get(path.as_str())
					.map(|(stat, _)| stat.clone())
			})
			.ok_or_else(|| Fault::source(format!("Path '{}' not found", path)));
		ready(result)
	}

	fn resolve_suffix(
		&self,
		path: Str,
	) -> impl Future<Output = Result<Option<SourceStat>, Fault>> + Send + '_ {
		ready(Ok(self.suffixes.lock().get(path.as_str()).cloned()))
	}

	fn open(&self, path: Str) -> impl Future<Output = Result<Self::Lease, Fault>> + Send + '_ {
		let result = self
			.files
			.lock()
			.get(path.as_str())
			.cloned()
			.map(|source| Lease {
				canonical_path: source.stat.canonical_path,
				revision:       source.revision,
				bytes:          source.bytes,
			})
			.ok_or_else(|| Fault::source(format!("Path '{}' not found", path)));
		ready(result)
	}

	fn read_bytes(&self, path: Str) -> impl Future<Output = Result<Bytes, Fault>> + Send + '_ {
		let result = self
			.files
			.lock()
			.get(path.as_str())
			.map(|source| source.bytes.clone())
			.ok_or_else(|| Fault::source(format!("Path '{}' not found", path)));
		ready(result)
	}

	fn list_directory(
		&self,
		path: Str,
		_max_depth: usize,
	) -> impl Future<Output = Result<DirectorySource, Fault>> + Send + '_ {
		let result = self
			.dirs
			.lock()
			.get(path.as_str())
			.map(|(_, source)| source.clone())
			.ok_or_else(|| Fault::source(format!("Path '{}' not found", path)));
		ready(result)
	}

	fn record_snapshot(&self, record: SnapshotRecord) -> Result<Option<Str>, Fault> {
		self.snapshots.lock().push(record);
		Ok(Some(Str::new_static("A1B2")))
	}
}

impl HttpClient for Sources {
	fn get(
		&self,
		_request: HttpRequest,
	) -> impl Future<Output = Result<HttpResponse, WebError>> + Send + '_ {
		ready(
			self
				.responses
				.lock()
				.pop_front()
				.unwrap_or_else(|| Err(WebError::request("web fixture not configured"))),
		)
	}
}

#[derive(Clone, Default)]
struct Blobs {
	stored: Arc<Mutex<Vec<(Bytes, Str)>>>,
}

impl ReadBlobs for Blobs {
	fn store(
		&self,
		bytes: Bytes,
		media_type: Str,
	) -> impl Future<Output = Result<BlobRef, Fault>> + Send + '_ {
		self.stored.lock().push((bytes.clone(), media_type.clone()));
		ready(Ok(BlobRef {
			hash: Str::new_static("blob-hash"),
			media_type,
			byte_len: bytes.len() as u64,
		}))
	}
}

impl Sources {
	fn file(&self, path: &str, bytes: impl Into<Bytes>) {
		self.file_as(path, path, path, bytes);
	}

	fn file_as(&self, authored: &str, canonical: &str, display: &str, bytes: impl Into<Bytes>) {
		let bytes = bytes.into();
		let source = FileSource {
			stat: SourceStat {
				canonical_path: Str::from(canonical),
				display_path:   Str::from(display),
				kind:           SourceKind::File,
				byte_len:       bytes.len() as u64,
				modified_ms:    Some(u64::MAX),
			},
			bytes,
			revision: Str::new_static("revision-7"),
		};
		let mut files = self.files.lock();
		files.insert(authored.to_owned(), source.clone());
		files.insert(canonical.to_owned(), source);
	}

	fn directory(&self, path: &str, entries: Vec<DirectoryEntry>) {
		let stat = SourceStat {
			canonical_path: Str::from(path),
			display_path:   Str::from(path),
			kind:           SourceKind::Directory,
			byte_len:       0,
			modified_ms:    Some(u64::MAX),
		};
		self.dirs.lock().insert(
			path.to_owned(),
			(stat, DirectorySource { root: Str::from(path), entries, truncated: false }),
		);
	}

	fn suffix(&self, authored: &str, resolved: &str) {
		let stat = self
			.files
			.lock()
			.get(resolved)
			.expect("resolved fixture exists")
			.stat
			.clone();
		self.suffixes.lock().insert(authored.to_owned(), stat);
	}
}

async fn project(sources: Sources, blobs: Blobs, raw: &str, media: bool) -> Vec<Part> {
	let tool = read::tool(sources, blobs);
	let (feed, params) = IncomingParams::channel();
	feed
		.args_committed(Str::from(raw))
		.expect("read invocation remains live");
	let events = tool.call(params).collect::<Vec<_>>().await;
	let [Ev::Done(Outcome::Done { result, .. })] = events.as_slice() else {
		panic!("expected one terminal read event: {events:?}");
	};
	tool.prompt(result.as_ref(), &PromptCaps {
		maximum_parts: 16,
		maximum_text_bytes: u32::MAX,
		media,
	})
}

async fn text(sources: Sources, raw: &str) -> String {
	let parts = project(sources, Blobs::default(), raw, false).await;
	let [Part::Text { text }] = parts.as_slice() else {
		panic!("expected exactly one model-facing text part: {parts:?}");
	};
	text.to_string()
}

fn numbered_lines(count: usize) -> String {
	(1..=count)
		.map(|line| format!("line {line}"))
		.collect::<Vec<_>>()
		.join("\n")
}

const FIXTURE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/special-sources");

fn fixture_path(relative: &str) -> PathBuf {
	Path::new(FIXTURE_ROOT).join(relative)
}

#[test]
fn schema_is_exactly_the_pi_read_schema() {
	let tool = read::tool(Sources::default(), Blobs::default());
	let actual: serde_json::Value =
		serde_json::from_slice(&tool.spec().schema).expect("schema JSON");
	assert_eq!(
		actual,
		json!({
			"type": "object",
			"additionalProperties": false,
			"required": ["path"],
			"properties": {
				"path": {
					"type": "string",
					"description": "Local path, internal URI (e.g. skill://), or URL. Inline selectors are supported."
				}
			}
		})
	);
}

#[test]
fn special_source_fixture_workspace_is_complete_and_self_contained() {
	let manifest: serde_json::Value = serde_json::from_slice(
		&std::fs::read(fixture_path("manifest.json")).expect("special-source manifest"),
	)
	.expect("valid special-source manifest");
	let expected_groups = [
		"plain",
		"directory",
		"archives",
		"database",
		"images",
		"documents",
		"notebooks",
		"profiles",
		"conflicts",
		"web",
	];
	for group in expected_groups {
		let paths = manifest[group]
			.as_array()
			.unwrap_or_else(|| panic!("fixture manifest group '{group}'"));
		assert!(!paths.is_empty(), "fixture manifest group '{group}' is empty");
		for path in paths {
			let relative = path.as_str().expect("fixture path string");
			assert!(fixture_path(relative).is_file(), "missing fixture '{relative}'");
		}
	}
	let large = std::fs::read(fixture_path("plain/large-utf8.txt")).expect("large UTF-8 fixture");
	assert!(large.len() > 50 * 1024);
	assert!(std::str::from_utf8(&large).is_ok());
	assert_eq!(
		&std::fs::read(fixture_path("database/catalog.sqlite")).expect("SQLite fixture")[..16],
		b"SQLite format 3\0"
	);
	assert_eq!(
		&std::fs::read(fixture_path("images/pixel.png")).expect("PNG fixture")[..8],
		b"\x89PNG\r\n\x1a\n"
	);
}

#[tokio::test]
async fn directory_listing_is_depth_two_and_elides_nested_children() {
	let sources = Sources::default();
	let mut entries = vec![DirectoryEntry {
		path:        Str::new_static("dir"),
		kind:        SourceKind::Directory,
		byte_len:    0,
		modified_ms: Some(u64::MAX),
	}];
	entries.extend((0..14).map(|index| DirectoryEntry {
		path:        Str::from(format!("dir/child-{index:02}.txt")),
		kind:        SourceKind::File,
		byte_len:    index,
		modified_ms: Some(u64::MAX),
	}));
	entries.push(DirectoryEntry {
		path:        Str::new_static("dir/nested/too-deep.txt"),
		kind:        SourceKind::File,
		byte_len:    1,
		modified_ms: Some(u64::MAX),
	});
	sources.directory("tree", entries);
	assert_eq!(
		text(sources, r#"{"path":"tree"}"#).await,
		concat!(
			".\n",
			"  - dir/\n",
			"    - child-00.txt\n",
			"    - child-01.txt\n",
			"    - child-02.txt\n",
			"    - child-03.txt\n",
			"    - child-04.txt\n",
			"    - child-05.txt\n",
			"    - child-06.txt\n",
			"    - child-07.txt\n",
			"    - child-08.txt\n",
			"    - child-09.txt\n",
			"    - child-10.txt\n",
			"    - … 2 more\n",
			"    - child-13.txt",
		)
	);
}

#[tokio::test]
async fn line_range_adds_context_header_and_records_the_exposed_snapshot() {
	let sources = Sources::default();
	sources.file("file.txt", numbered_lines(12));
	assert_eq!(
		text(sources.clone(), r#"{"path":"file.txt:5-8"}"#).await,
		concat!(
			"[file.txt#A1B2]\n",
			"4:line 4\n5:line 5\n6:line 6\n7:line 7\n8:line 8\n9:line 9\n10:line 10\n11:line 11\n\n",
			"[1 more lines in file. Use :12 to continue]",
		)
	);
	let snapshots = sources.snapshots.lock();
	let [snapshot] = snapshots.as_slice() else {
		panic!("one snapshot must be recorded")
	};
	assert_eq!(snapshot.path, "file.txt");
	assert_eq!(snapshot.revision, "revision-7");
	assert_eq!(
		snapshot
			.seen
			.iter()
			.map(|span| (span.start_line, span.end_line))
			.collect::<Vec<_>>(),
		vec![(4, 11)]
	);
}

#[tokio::test]
async fn raw_is_verbatim_and_multi_range_uses_one_hashline_header_and_ellipsis() {
	let sources = Sources::default();
	sources.file("file.txt", numbered_lines(10));
	assert_eq!(
		text(sources.clone(), r#"{"path":"file.txt:raw:5-8"}"#).await,
		"line 5\nline 6\nline 7\nline 8"
	);
	assert_eq!(
		text(sources, r#"{"path":"file.txt:2-3,8-9"}"#).await,
		concat!("[file.txt#A1B2]\n2:line 2\n3:line 3\n…\n8:line 8\n9:line 9",)
	);
}

#[tokio::test]
async fn standard_text_truncation_spills_the_complete_numbered_projection() {
	let sources = Sources::default();
	sources.file("large.txt", numbered_lines(4000));
	let blobs = Blobs::default();
	let parts = project(sources, blobs.clone(), r#"{"path":"large.txt"}"#, false).await;
	let [Part::Text { text }] = parts.as_slice() else {
		panic!("expected one truncated text projection: {parts:?}");
	};

	let mut full = String::from("[large.txt#A1B2]\n");
	for line in 1..=4000 {
		if line > 1 {
			full.push('\n');
		}
		full.push_str(&format!("{line}:line {line}"));
	}
	let visible = full.lines().take(3000).collect::<Vec<_>>().join("\n");
	assert_eq!(
		text.as_str(),
		format!("{visible}\n\n[truncated: 3000 of 4001 lines shown; full output in blob blob-hash]")
	);

	let stored = blobs.stored.lock();
	let [(bytes, media_type)] = stored.as_slice() else {
		panic!("truncated text must spill exactly one blob: {stored:?}");
	};
	assert_eq!(bytes.as_ref(), full.as_bytes());
	assert_eq!(media_type.as_str(), "text/plain; charset=utf-8");
}

#[tokio::test]
async fn structural_summary_has_a_concrete_recovery_footer() {
	let sources = Sources::default();
	let mut body = String::from("pub fn giant() {\n");
	for line in 0..120 {
		body.push_str(&format!("    let value_{line} = {line};\n"));
	}
	body.push_str("}\n");
	sources.file("big.rs", body);
	assert_eq!(
		text(sources, r#"{"path":"big.rs"}"#).await,
		concat!(
			"[big.rs#A1B2]\n",
			"1-122:pub fn giant() { … }\n\n",
			"[…120ln elided; re-read needed ranges with big.rs:1-122]",
		)
	);
}

#[tokio::test]
async fn files_over_twenty_thousand_lines_skip_structural_summary() {
	let sources = Sources::default();
	let mut body = String::from("pub fn too_many_lines() {\n");
	for line in 0..20_001 {
		body.push_str(&format!("\tlet value_{line} = {line};\n"));
	}
	body.push_str("}\n");
	sources.file("too-many.rs", body);
	let output = text(sources, r#"{"path":"too-many.rs"}"#).await;
	assert!(output.starts_with("[too-many.rs#A1B2]\n1:pub fn too_many_lines() {\n"), "{output}");
	assert!(!output.contains("ln elided; re-read needed ranges"), "{output}");
}

struct TempDb(PathBuf);
impl Drop for TempDb {
	fn drop(&mut self) {
		let _ = std::fs::remove_file(&self.0);
	}
}

fn sqlite_fixture() -> TempDb {
	static NEXT: AtomicU64 = AtomicU64::new(0);
	let path = std::env::temp_dir().join(format!(
		"omp-read-golden-{}-{}.sqlite",
		std::process::id(),
		NEXT.fetch_add(1, Ordering::Relaxed),
	));
	std::fs::write(&path, include_bytes!("fixtures/special-sources/database/catalog.sqlite"))
		.expect("copy checked-in SQLite fixture");
	TempDb(path)
}

#[tokio::test]
async fn sqlite_root_table_key_where_and_forbidden_where_are_model_text() {
	let db = sqlite_fixture();
	let sources = Sources::default();
	sources.file_as(
		"data.sqlite",
		db.0.to_str().unwrap(),
		"data.sqlite",
		std::fs::read(&db.0).expect("read SQLite fixture bytes"),
	);
	assert_eq!(
		text(sources.clone(), r#"{"path":"data.sqlite"}"#).await,
		"packages (2 rows)\npeople (3 rows)"
	);
	assert_eq!(
		text(sources.clone(), r#"{"path":"data.sqlite:people:2"}"#).await,
		"id: 2\nname: Grace\nscore: 20"
	);
	assert_eq!(
		text(sources.clone(), r#"{"path":"data.sqlite:people?where=score%3E10&limit=2"}"#).await,
		concat!(
			"| id  | name  | score |\n",
			"| --- | ----- | ----- |\n",
			"| 2   | Grace | 20    |\n",
			"| 3   | Linus | 30    |",
		)
	);
	let schema = text(sources.clone(), r#"{"path":"data.sqlite:people"}"#).await;
	assert_eq!(
		schema,
		concat!(
			"CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT, score INTEGER)\n\n",
			"Sample rows:\n",
			"| id  | name  | score |\n",
			"| --- | ----- | ----- |\n",
			"| 1   | Ada   | 10    |\n",
			"| 2   | Grace | 20    |\n",
			"| 3   | Linus | 30    |",
		)
	);
	assert_eq!(
		text(sources, r#"{"path":"data.sqlite:people?where=score%3E0%20LIMIT%201"}"#).await,
		"SQLite 'where' clause must not contain \
		 LIMIT/OFFSET/UNION/INTERSECT/EXCEPT/ATTACH/DETACH/PRAGMA; use '?q=SELECT ...' for raw SQL"
	);
}

fn zip_fixture() -> Bytes {
	Bytes::from_static(include_bytes!("fixtures/special-sources/archives/bundle.zip"))
}

fn tar_fixture() -> Bytes {
	Bytes::from_static(include_bytes!("fixtures/special-sources/archives/bundle.tar.gz"))
}

#[tokio::test]
async fn zip_and_tar_root_member_and_member_range_use_standard_text_formatting() {
	let sources = Sources::default();
	sources.file("bundle.zip", zip_fixture());
	sources.file("bundle.tar.gz", tar_fixture());
	assert_eq!(
		text(sources.clone(), r#"{"path":"bundle.zip"}"#).await,
		"binary.bin (4B)\ndir/\nroot.txt (18B)"
	);
	assert_eq!(
		text(sources.clone(), r#"{"path":"bundle.zip:dir/member.txt"}"#).await,
		"1:one\n2:two\n3:three\n4:four\n5:five"
	);
	assert_eq!(
		text(sources.clone(), r#"{"path":"bundle.zip:dir/member.txt:2-3"}"#).await,
		"1:one\n2:two\n3:three\n4:four\n5:five"
	);
	assert_eq!(
		text(sources.clone(), r#"{"path":"bundle.tar.gz"}"#).await,
		"binary.bin (4B)\ndir/\nroot.txt (18B)"
	);
	assert_eq!(
		text(sources.clone(), r#"{"path":"bundle.tar.gz:dir/member.txt"}"#).await,
		"1:one\n2:two\n3:three\n4:four\n5:five"
	);
	assert_eq!(
		text(sources, r#"{"path":"bundle.tar.gz:dir/member.txt:raw:2-3"}"#).await,
		"two\nthree"
	);
}

#[tokio::test]
async fn notebook_cells_are_projected_with_editable_markers() {
	let sources = Sources::default();
	let notebook = include_str!("fixtures/special-sources/notebooks/book.ipynb");
	sources.file("book.ipynb", notebook);
	assert_eq!(
		text(sources, r#"{"path":"book.ipynb"}"#).await,
		concat!(
			"1:# %% [markdown] cell:0\n2:# Fixture notebook\n3:Unicode: café 東京\n4:\n",
			"5:# %% [code] cell:1\n6:value = 42\n7:print(value)",
		)
	);
}

#[tokio::test]
async fn conflicted_notebook_selector_runs_before_notebook_json_conversion() {
	let sources = Sources::default();
	sources.file("merge.ipynb", include_str!("fixtures/special-sources/conflicts/merge.ipynb"));
	let output = text(sources, r#"{"path":"merge.ipynb:conflicts"}"#).await;
	assert!(
		output.starts_with(
			"⚠ 1 unresolved conflict in merge.ipynb\n- ours = HEAD\n- theirs = feature/notebook\n- \
			 base = base\n"
		),
		"{output}"
	);
	assert!(output.ends_with("(3-way)"), "{output}");
}

#[tokio::test]
async fn document_raw_selector_returns_converted_markdown_without_line_projection() {
	let sources = Sources::default();
	sources.file(
		"report.docx",
		Bytes::from_static(include_bytes!("fixtures/special-sources/documents/report.docx")),
	);
	assert_eq!(
		text(sources, r#"{"path":"report.docx:raw"}"#).await,
		"Fixture document\n\nConverted café."
	);
}

const CONFLICTED: &str = include_str!("fixtures/special-sources/conflicts/merge.txt");

#[tokio::test]
async fn conflict_selector_is_a_compact_index_and_normal_read_appends_warning() {
	let sources = Sources::default();
	sources.file("conflicted.txt", CONFLICTED);
	let summary = text(sources.clone(), r#"{"path":"conflicted.txt:conflicts"}"#).await;
	assert!(
		summary.starts_with(
			"⚠ 1 unresolved conflict in conflicted.txt\n- ours = HEAD\n- theirs = feature/source\n- \
			 base = base\n"
		),
		"{summary}"
	);
	assert!(summary.ends_with("\n\n#1  L2-8  (3-way)"), "{summary}");
	let ordinary = text(sources, r#"{"path":"conflicted.txt"}"#).await;
	assert!(ordinary.starts_with("[conflicted.txt#A1B2]\n1:before\n2:<<<<<<< HEAD"), "{ordinary}");
	assert!(
		ordinary.contains(
			"\n⚠ 1 unresolved conflict detected\n- ours = HEAD\n- theirs = feature/source\n"
		),
		"{ordinary}"
	);
}

fn png_fixture() -> Bytes {
	Bytes::from_static(include_bytes!("fixtures/special-sources/images/pixel.png"))
}

#[tokio::test]
async fn image_read_emits_description_and_blob_and_rejects_over_twenty_mibibytes() {
	let sources = Sources::default();
	sources.file("pixel.png", png_fixture());
	let blobs = Blobs::default();
	let parts = project(sources.clone(), blobs.clone(), r#"{"path":"pixel.png"}"#, true).await;
	let [Part::Text { text: description }, Part::Blob { blob, alt }] = parts.as_slice() else {
		panic!("image read must emit text plus blob: {parts:?}");
	};
	assert_eq!(description, "Read image file [image/png]");
	assert_eq!(blob.hash, "blob-hash");
	assert_eq!(blob.media_type, "image/png");
	assert_eq!(alt.as_deref(), Some("Read image file [image/png]"));
	assert_eq!(blobs.stored.lock().len(), 1);

	sources.file("huge.png", Bytes::from(vec![0; 20 * 1024 * 1024 + 1]));
	assert_eq!(
		text(sources, r#"{"path":"huge.png"}"#).await,
		"Image file too large: 20.0MB exceeds 20.0MB limit."
	);
}

#[tokio::test]
async fn cpu_profile_is_summarized_instead_of_dumping_json() {
	let sources = Sources::default();
	let profile = include_str!("fixtures/special-sources/profiles/run.cpuprofile");
	sources.file("run.cpuprofile", profile);
	assert_eq!(
		text(sources, r#"{"path":"run.cpuprofile"}"#).await,
		concat!(
			"1:V8 CPU profile: 1.00 s wall clock, 4 samples (avg interval 250000 µs)\n",
			"2:On-CPU total: 1.00 s (100.0% of wall clock). Values below are on-CPU milliseconds \
			 (idle time excluded).\n",
			"3:\n4:## Hot paths\n5:  1000.0 100.0%  work (/src/work.js:5)\n",
			"6:\n7:## Top functions by self time (idle time excluded)\n8:  1000.0 100.0%  work \
			 (/src/work.js:5)\n",
			"9:\n10:[Summarized view of a V8 .cpuprofile. Use ':raw' to read the original JSON.]",
		)
	);
}

#[tokio::test]
async fn macos_sample_profile_uses_the_checked_in_call_tree_fixture() {
	let sources = Sources::default();
	sources
		.file("trace.sample.txt", include_str!("fixtures/special-sources/profiles/trace.sample.txt"));
	let output = text(sources, r#"{"path":"trace.sample.txt"}"#).await;
	assert!(
		output.starts_with("1:macOS sample profile: fixture (pid 123), sampled every 1 ms\n"),
		"{output}"
	);
	assert!(output.contains("work (fixture)") || output.contains("work  (in fixture)"), "{output}");
	assert!(
		output.ends_with(
			"[Summarized view of a macOS `sample` call-tree report. Use ':raw' to read the original \
			 file.]"
		),
		"{output}"
	);
}

#[tokio::test]
async fn checked_in_url_mock_drives_the_network_free_html_pipeline() {
	let sources = Sources::default();
	sources.responses.lock().push_back(Ok(HttpResponse {
		final_url:    Str::new_static("https://fixture.invalid/final"),
		status:       200,
		content_type: Some(Str::new_static("text/html")),
		headers:      vec![(
			Str::new_static("content-type"),
			Str::new_static("text/html; charset=utf-8"),
		)]
		.into(),
		body:         Bytes::from_static(include_bytes!("fixtures/special-sources/web/page.html")),
	}));
	let output = text(sources, r#"{"path":"https://fixture.invalid/page"}"#).await;
	assert!(
		output.starts_with(
			"URL: https://fixture.invalid/final\nContent-Type: text/html\nMethod: native\n\n---\n\n"
		),
		"{output}"
	);
	assert!(output.contains("# Fixture page"), "{output}");
	assert!(output.contains("Network-free café content."), "{output}");
	assert!(!output.contains("Skip navigation"), "{output}");
}

#[tokio::test]
async fn unsupported_uri_suffix_recovery_and_semicolon_sections_are_exact() {
	let sources = Sources::default();
	assert_eq!(
		text(sources.clone(), r#"{"path":"skill://react"}"#).await,
		"skill:// targets are not supported yet"
	);

	sources.file("nested/lost.txt", "found");
	sources.suffix("lost.txt", "nested/lost.txt");
	assert_eq!(
		text(sources.clone(), r#"{"path":"lost.txt:raw"}"#).await,
		concat!("[Path 'lost.txt' not found; resolved to 'nested/lost.txt' via suffix match]\nfound",)
	);

	sources.file("one.txt", "alpha");
	sources.file("two.txt", "beta");
	assert_eq!(
		text(sources, r#"{"path":"one.txt:raw;two.txt:raw"}"#).await,
		concat!("Note: interpreted as 2 paths: one.txt:raw, two.txt:raw\n\nalpha\n\nbeta",)
	);
}
