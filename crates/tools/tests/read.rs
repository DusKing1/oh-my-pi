//! Model-facing behavioral contracts for pi-compatible `read@1`.

use std::{
	collections::{HashMap, VecDeque},
	fmt::Write as _,
	future::{Future, ready},
	path::{Path, PathBuf},
	sync::{
		Arc,
		atomic::{AtomicU64, Ordering},
	},
	time::Duration,
};

use bytes::Bytes;
use futures::StreamExt as _;
use omp_core::Str;
use omp_tool::{Abort, BlobRef, Ev, IncomingParams, Interrupt, Outcome, Part, PromptCaps, Tool};
use omp_tools::read::{
	self, DirectoryEntry, DirectorySource, Fault, ReadBlobs, ReadLease, ReadSources, SnapshotRecord,
	SourceKind, SourceStat,
	web::types::{HttpClient, HttpRequest, HttpResponse, WebError},
};
use parking_lot::Mutex;
use serde_json::json;

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
			.ok_or_else(|| Fault::source(format!("Path '{path}' not found")));
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
			.ok_or_else(|| Fault::source(format!("Path '{path}' not found")));
		ready(result)
	}

	fn read_bytes(&self, path: Str) -> impl Future<Output = Result<Bytes, Fault>> + Send + '_ {
		let result = self
			.files
			.lock()
			.get(path.as_str())
			.map(|source| source.bytes.clone())
			.ok_or_else(|| Fault::source(format!("Path '{path}' not found")));
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
			.ok_or_else(|| Fault::source(format!("Path '{path}' not found")));
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

	fn directory_symlink(&self, authored: &str, target: &str) {
		let target_stat = self
			.dirs
			.lock()
			.get(target)
			.unwrap_or_else(|| panic!("directory symlink target '{target}' exists"))
			.0
			.clone();
		self.files.lock().insert(authored.to_owned(), FileSource {
			stat:     SourceStat { kind: SourceKind::Symlink, ..target_stat },
			bytes:    Bytes::new(),
			revision: Str::new_static("symlink"),
		});
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

async fn assert_truncated_text_spill(sources: Sources, raw: &str, expected: &str) {
	let blobs = Blobs::default();
	let parts = project(sources, blobs.clone(), raw, false).await;
	let [Part::Text { text }] = parts.as_slice() else {
		panic!("expected one truncated text projection: {parts:?}");
	};
	let marker = "\n\n[truncated: ";
	let (visible, footer) = text
		.rsplit_once(marker)
		.unwrap_or_else(|| panic!("missing truthful truncation footer: {text}"));
	let shown_lines = if visible.is_empty() {
		0
	} else {
		visible.bytes().filter(|byte| *byte == b'\n').count() + 1
	};
	let total_lines = expected.bytes().filter(|byte| *byte == b'\n').count() + 1;
	assert_eq!(
		format!("[truncated: {footer}"),
		format!(
			"[truncated: {shown_lines} of {total_lines} lines shown; full output in blob blob-hash]"
		)
	);
	assert_eq!(
		visible,
		expected
			.lines()
			.take(shown_lines)
			.collect::<Vec<_>>()
			.join("\n"),
		"the visible prefix must contain only complete output lines"
	);
	let stored = blobs.stored.lock();
	let [(bytes, media_type)] = stored.as_slice() else {
		panic!("truncated text must spill exactly one blob: {stored:?}");
	};
	assert_eq!(bytes.as_ref(), expected.as_bytes());
	assert_eq!(media_type.as_str(), "text/plain; charset=utf-8");
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
fn generated_schema_is_semantically_the_pi_read_schema() {
	let tool = read::tool(Sources::default(), Blobs::default());
	let actual: serde_json::Value =
		serde_json::from_slice(&tool.spec().schema).expect("schema JSON");
	assert_eq!(
		tool.spec().schema.as_ref(),
		omp_tool::schema::<read::Params>().as_ref(),
		"tool schema must be generated directly from Params",
	);
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
	for legacy in [
		json!({"path": "src/lib.rs", "ranges": [[1, 2]]}),
		json!({"path": "src/lib.rs", "structural": true}),
	] {
		assert!(
			serde_json::from_value::<read::Params>(legacy).is_err(),
			"read params must reject legacy fields"
		);
	}
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
async fn oversized_directory_listing_spills_the_complete_rendered_tree() {
	let sources = Sources::default();
	let mut entries = Vec::with_capacity(4_000);
	let mut expected = String::from(".");
	for index in 0..4_000 {
		let name = format!("entry-{index:04}-abcdefghijklmnop.txt");
		entries.push(DirectoryEntry {
			path:        Str::from(name.clone()),
			kind:        SourceKind::File,
			byte_len:    1,
			modified_ms: Some(u64::MAX),
		});
		write!(expected, "\n  - {name}").expect("writing expected directory listing");
	}
	sources.directory("large-tree", entries);

	assert_truncated_text_spill(sources, r#"{"path":"large-tree"}"#, &expected).await;
}

#[tokio::test]
async fn directory_symlink_is_reclassified_before_special_dispatch() {
	let sources = Sources::default();
	sources.directory("tree", vec![DirectoryEntry {
		path:        Str::new_static("leaf.txt"),
		kind:        SourceKind::File,
		byte_len:    4,
		modified_ms: Some(u64::MAX),
	}]);
	sources.directory_symlink("tree-link", "tree");

	assert_eq!(text(sources, r#"{"path":"tree-link"}"#).await, ".\n  - leaf.txt");
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
		"[file.txt#A1B2]\n2:line 2\n3:line 3\n…\n8:line 8\n9:line 9"
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
		write!(full, "{line}:line {line}").expect("writing to string");
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
async fn final_projection_only_authorizes_source_lines_that_survive_the_shared_cap() {
	let sources = Sources::default();
	sources.file("large.txt", numbered_lines(4000));
	let _ = project(sources.clone(), Blobs::default(), r#"{"path":"large.txt"}"#, false).await;
	let snapshots = sources.snapshots.lock();
	let [snapshot] = snapshots.as_slice() else {
		panic!("one snapshot must be recorded")
	};
	assert_eq!(
		snapshot
			.seen
			.iter()
			.map(|span| (span.start_line, span.end_line))
			.collect::<Vec<_>>(),
		vec![(1, 2999)],
		"the header consumes one of the 3000 retained projection lines"
	);
}

#[tokio::test]
async fn structural_summary_has_a_concrete_recovery_footer() {
	let sources = Sources::default();
	let mut body = String::from("pub fn giant() {\n");
	for line in 0..120 {
		writeln!(body, "    let value_{line} = {line};").expect("writing to string");
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
		writeln!(body, "\tlet value_{line} = {line}").expect("writing to string");
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

#[tokio::test]
async fn oversized_sqlite_output_spills_the_complete_rendered_table() {
	let db = sqlite_fixture();
	{
		let mut connection = rusqlite::Connection::open(&db.0).expect("open SQLite spill fixture");
		connection
			.execute_batch("CREATE TABLE wide(id INTEGER PRIMARY KEY, alpha TEXT, beta TEXT);")
			.expect("create wide SQLite table");
		let transaction = connection
			.transaction()
			.expect("start SQLite spill fixture transaction");
		let cell = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
		for id in 1..=1_000_i64 {
			transaction
				.execute("INSERT INTO wide(id, alpha, beta) VALUES (?1, ?2, ?3)", (id, cell, cell))
				.expect("insert wide SQLite row");
		}
		transaction.commit().expect("commit SQLite spill fixture");
	}
	let authored = "wide.sqlite?q=SELECT%20id,alpha,beta%20FROM%20wide%20ORDER%20BY%20id";
	let expected =
		read::sqlite::read(&db.0, authored).expect("render complete oversized SQLite output");
	assert!(expected.len() > 50 * 1024, "SQLite fixture must exceed the shared byte limit");

	let sources = Sources::default();
	sources.file_as(
		"wide.sqlite",
		db.0.to_str().unwrap(),
		"wide.sqlite",
		std::fs::read(&db.0).expect("read oversized SQLite fixture bytes"),
	);
	assert_truncated_text_spill(
		sources,
		r#"{"path":"wide.sqlite?q=SELECT%20id,alpha,beta%20FROM%20wide%20ORDER%20BY%20id"}"#,
		&expected,
	)
	.await;
}

#[tokio::test]
async fn suffix_resolved_sqlite_container_dispatches_with_exact_notice() {
	let db = sqlite_fixture();
	let sources = Sources::default();
	sources.file_as(
		"resolved/data.sqlite",
		db.0.to_str().unwrap(),
		"resolved/data.sqlite",
		std::fs::read(&db.0).expect("read SQLite fixture bytes"),
	);
	sources.suffix("missing/data.sqlite", "resolved/data.sqlite");

	assert_eq!(
		text(sources, r#"{"path":"missing/data.sqlite"}"#).await,
		concat!(
			"[Path 'missing/data.sqlite' not found; resolved to 'resolved/data.sqlite' via suffix \
			 match]\n",
			"packages (2 rows)\npeople (3 rows)",
		)
	);
}

#[tokio::test]
async fn sqlite_extensions_without_magic_are_read_as_ordinary_text() {
	let sources = Sources::default();
	sources.file("notes.db", "not a database");
	sources.file("notes.sqlite", "also plain text");

	assert_eq!(
		text(sources.clone(), r#"{"path":"notes.db"}"#).await,
		"[notes.db#A1B2]\n1:not a database"
	);
	assert_eq!(
		text(sources, r#"{"path":"notes.sqlite"}"#).await,
		"[notes.sqlite#A1B2]\n1:also plain text"
	);
}

#[tokio::test(flavor = "current_thread")]
async fn long_sqlite_query_is_interrupted_without_blocking_the_runtime() {
	let db = sqlite_fixture();
	let sources = Sources::default();
	sources.file_as(
		"data.sqlite",
		db.0.to_str().unwrap(),
		"data.sqlite",
		std::fs::read(&db.0).expect("read SQLite fixture bytes"),
	);
	let tool = read::tool(sources, Blobs::default());
	let (feed, params) = IncomingParams::channel();
	feed
		.args_committed(Str::new_static(
			r#"{"path":"data.sqlite?q=WITH%20RECURSIVE%20count(x)%20AS%20(VALUES(0)%20UNION%20ALL%20SELECT%20x%2B1%20FROM%20count)%20SELECT%20sum(x)%20FROM%20count"}"#,
		))
		.expect("read invocation remains live");
	let events = tool.call(params).collect::<Vec<_>>();
	tokio::pin!(events);

	tokio::select! {
		result = &mut events => panic!("unbounded SQLite query completed unexpectedly: {result:?}"),
		() = tokio::time::sleep(Duration::from_millis(50)) => {},
	}
	feed
		.interrupt(Interrupt {
			class:  Str::new_static("deadline"),
			reason: Str::new_static("test deadline exceeded"),
		})
		.expect("read invocation accepts its deadline interrupt");
	let events = tokio::time::timeout(Duration::from_secs(1), &mut events)
		.await
		.expect("SQLite query stops within the cancellation bound");
	assert!(
		matches!(
			events.as_slice(),
			[Ev::Aborted(Abort::Interrupted { reason })] if reason == "test deadline exceeded"
		),
		"deadline remains structured abort truth: {events:?}"
	);
}

const fn zip_fixture() -> Bytes {
	Bytes::from_static(include_bytes!("fixtures/special-sources/archives/bundle.zip"))
}

const fn tar_fixture() -> Bytes {
	Bytes::from_static(include_bytes!("fixtures/special-sources/archives/bundle.tar.gz"))
}

fn encoded_zip(entries: &[(&str, &str)]) -> Bytes {
	Bytes::from(
		omp_ar::zip::encode(
			entries
				.iter()
				.map(|&(path, contents)| (path, contents.as_bytes())),
		)
		.expect("encode ZIP fixture"),
	)
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
async fn oversized_archive_listing_spills_every_complete_entry_line() {
	let mut writer = omp_ar::zip::Writer::new(Vec::new());
	let mut expected_lines = Vec::with_capacity(read::archive::DEFAULT_ARCHIVE_LIST_LIMIT);
	for index in 0..read::archive::DEFAULT_ARCHIVE_LIST_LIMIT {
		let name = format!("entry-{index:03}-{}.txt", "x".repeat(120));
		writer
			.add_file(&name, b"x")
			.expect("add oversized archive listing entry");
		expected_lines.push(format!("{name} (1B)"));
	}
	let archive = Bytes::from(writer.finish().expect("finish oversized archive fixture"));
	let expected = expected_lines.join("\n");
	assert!(expected.len() > 50 * 1024, "archive fixture must exceed the shared byte limit");

	let sources = Sources::default();
	sources.file("large-listing.zip", archive);
	assert_truncated_text_spill(sources, r#"{"path":"large-listing.zip"}"#, &expected).await;
}

#[tokio::test]
async fn selector_shaped_archive_members_win_over_selector_interpretation() {
	let sources = Sources::default();
	sources.file(
		"selectors.zip",
		encoded_zip(&[
			("50", "literal numeric member\nsecond line"),
			("raw", "literal raw member\nsecond line"),
		]),
	);

	assert_eq!(
		text(sources.clone(), r#"{"path":"selectors.zip:50"}"#).await,
		"1:literal numeric member\n2:second line"
	);
	assert_eq!(
		text(sources, r#"{"path":"selectors.zip:raw"}"#).await,
		"1:literal raw member\n2:second line"
	);
}

#[tokio::test]
async fn absent_selector_shaped_members_fall_back_to_text_selectors() {
	let sources = Sources::default();
	let member = numbered_lines(60);
	let root_sources = Sources::default();
	root_sources
		.file("root-fallback.zip", encoded_zip(&[("a.txt", "a"), ("b.txt", "b"), ("c.txt", "c")]));
	assert_eq!(
		text(root_sources.clone(), r#"{"path":"root-fallback.zip:raw"}"#).await,
		"a.txt (1B)\nb.txt (1B)\nc.txt (1B)"
	);
	assert_eq!(
		text(root_sources.clone(), r#"{"path":"root-fallback.zip:50"}"#).await,
		"(empty archive directory)"
	);
	assert_eq!(
		text(root_sources, r#"{"path":"root-fallback.zip:2-3"}"#).await,
		"b.txt (1B)\nc.txt (1B)"
	);
	sources.file("fallback.zip", encoded_zip(&[("member.txt", member.as_str())]));

	assert_eq!(text(sources.clone(), r#"{"path":"fallback.zip:member.txt:raw"}"#).await, member);
	assert_eq!(
		text(sources.clone(), r#"{"path":"fallback.zip:member.txt:50"}"#).await,
		concat!(
			"49:line 49\n50:line 50\n51:line 51\n52:line 52\n53:line 53\n54:line 54\n",
			"55:line 55\n56:line 56\n57:line 57\n58:line 58\n59:line 59\n60:line 60",
		)
	);
	assert_eq!(
		text(sources, r#"{"path":"fallback.zip:member.txt:10-12"}"#).await,
		concat!(
			"9:line 9\n10:line 10\n11:line 11\n12:line 12\n13:line 13\n14:line 14\n",
			"15:line 15\n\n[45 more lines in file. Use :16 to continue]",
		)
	);
}

#[tokio::test]
async fn suffix_resolved_archive_container_dispatches_with_exact_notice() {
	let sources = Sources::default();
	sources.file_as(
		"resolved/bundle.zip",
		"resolved/bundle.zip",
		"resolved/bundle.zip",
		zip_fixture(),
	);
	sources.suffix("missing/bundle.zip", "resolved/bundle.zip");

	assert_eq!(
		text(sources, r#"{"path":"missing/bundle.zip:dir/member.txt"}"#).await,
		concat!(
			"[Path 'missing/bundle.zip' not found; resolved to 'resolved/bundle.zip' via suffix \
			 match]\n",
			"1:one\n2:two\n3:three\n4:four\n5:five",
		)
	);
}

#[tokio::test]
async fn notebook_cells_are_projected_with_editable_markers() {
	let sources = Sources::default();
	let notebook = include_str!("fixtures/special-sources/notebooks/book.ipynb");
	sources.file("book.ipynb", notebook);
	assert_eq!(
		text(sources.clone(), r#"{"path":"book.ipynb"}"#).await,
		concat!(
			"[book.ipynb#A1B2]\n",
			"1:# %% [markdown] cell:0\n2:# Fixture notebook\n3:Unicode: café 東京\n4:\n",
			"5:# %% [code] cell:1\n6:value = 42\n7:print(value)",
		)
	);
	let snapshots = sources.snapshots.lock();
	let [snapshot] = snapshots.as_slice() else {
		panic!("one notebook snapshot must be recorded")
	};
	assert_eq!(
		snapshot.bytes.as_ref(),
		read::notebook::render(notebook.as_bytes(), "book.ipynb")
			.unwrap()
			.text
			.as_bytes()
	);
	assert_eq!(
		snapshot
			.seen
			.iter()
			.map(|span| (span.start_line, span.end_line))
			.collect::<Vec<_>>(),
		vec![(1, 7)]
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
		text(sources.clone(), r#"{"path":"report.docx:raw"}"#).await,
		"Fixture document\n\nConverted café."
	);
	assert!(sources.snapshots.lock().is_empty());
}

const CONFLICTED: &str = include_str!("fixtures/special-sources/conflicts/merge.txt");

#[tokio::test]
async fn conflict_selector_is_a_compact_index_and_normal_read_appends_warning() {
	let sources = Sources::default();
	sources.file("conflicted.txt", CONFLICTED);
	let summary = text(sources.clone(), r#"{"path":"conflicted.txt:conflicts"}"#).await;
	assert_eq!(
		summary,
		concat!(
			"⚠ 1 unresolved conflict in conflicted.txt\n",
			"- ours = HEAD\n",
			"- theirs = feature/source\n",
			"- base = base\n",
			"NOTICE: Read `conflicted.txt:conflicts` for the conflict index, then read the affected ",
			"source ranges to obtain their `[conflicted.txt#TAG]` header and numbered marker lines. ",
			"Resolve each complete marker block with the hashline `edit` tool, using `PUT N.=M:` \
			 from ",
			"`<<<<<<<` through `>>>>>>>`; preserve the intended side(s), and re-read ",
			"`conflicted.txt:conflicts` to verify.\n\n",
			"#1  L2-8  (3-way)",
		)
	);
	assert!(!summary.contains("conflict://"));
	let warning = read::conflicts::render_conflict_warning(CONFLICTED);
	assert_eq!(
		warning.text,
		concat!(
			"\n⚠ 1 unresolved conflict detected\n",
			"- ours = HEAD\n",
			"- theirs = feature/source\n",
			"- base = base\n",
			"NOTICE: Read `path:conflicts` for the conflict index, then read the affected source ",
			"ranges to obtain their `[path#TAG]` header and numbered marker lines. Resolve each ",
			"complete marker block with the hashline `edit` tool, using `PUT N.=M:` from ",
			"`<<<<<<<` through `>>>>>>>`; preserve the intended side(s), and re-read ",
			"`path:conflicts` to verify.\n\n",
			"──── #1  L2-8 ────\n",
			"<<< ours\n",
			"ours\n",
			"=== base\n",
			"ancestor\n",
			">>> theirs\n",
			"theirs",
		)
	);
	assert!(!warning.text.contains("conflict://"));
	let ordinary = text(sources, r#"{"path":"conflicted.txt"}"#).await;
	assert!(ordinary.starts_with("[conflicted.txt#A1B2]\n1:before\n2:<<<<<<< HEAD"), "{ordinary}");
	assert!(ordinary.ends_with(warning.text.as_str()), "{ordinary}");
}

#[tokio::test]
async fn oversized_conflict_index_spills_every_complete_summary_line() {
	let mut source = String::new();
	for index in 1..=3_100 {
		writeln!(source, "<<<<<<< HEAD\nours {index}\n=======\ntheirs {index}\n>>>>>>> feature")
			.expect("writing conflict fixture");
	}
	let expected =
		read::conflicts::render_conflicts_for_path(&source, "many-conflicts.txt", false).text;
	assert!(expected.lines().count() > 3_000, "conflict fixture must exceed the shared line limit");

	let sources = Sources::default();
	sources.file("many-conflicts.txt", source);
	assert_truncated_text_spill(sources, r#"{"path":"many-conflicts.txt:conflicts"}"#, &expected)
		.await;
}

#[tokio::test]
async fn ordinary_conflict_warning_requires_a_complete_emitted_marker_block() {
	const SOURCE: &str = concat!(
		"before\n",
		"<<<<<<< HEAD\n",
		"ours\n",
		"||||||| base\n",
		"ancestor\n",
		"=======\n",
		"theirs\n",
		">>>>>>> feature\n",
		"after\n",
		"far away\n",
	);
	let sources = Sources::default();
	sources.file("window.txt", SOURCE);

	let hidden = text(sources.clone(), r#"{"path":"window.txt:10"}"#).await;
	assert!(!hidden.contains("unresolved conflict"), "{hidden}");

	let visible = text(sources, r#"{"path":"window.txt:3-7"}"#).await;
	assert!(visible.contains("\n⚠ 1 unresolved conflict detected"), "{visible}");
}

const fn png_fixture() -> Bytes {
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
	let expected = concat!(
		"Read image file [image/jpeg]\n",
		"[Image: original 8x6, displayed at 267x200. Multiply coordinates by 0.03 to map to \
		 original image.]",
	);
	assert_eq!(description, expected);
	assert_eq!(blob.hash, "blob-hash");
	assert_eq!(blob.media_type, "image/jpeg");
	assert_eq!(alt.as_deref(), Some(expected));
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
		text(sources.clone(), r#"{"path":"run.cpuprofile"}"#).await,
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
	assert!(sources.snapshots.lock().is_empty());
}

#[tokio::test]
async fn macos_sample_profile_uses_the_checked_in_call_tree_fixture() {
	let sources = Sources::default();
	sources
		.file("trace.sample.txt", include_str!("fixtures/special-sources/profiles/trace.sample.txt"));
	let output = text(sources.clone(), r#"{"path":"trace.sample.txt"}"#).await;
	assert!(
		output.starts_with("1:macOS sample profile: fixture (pid 123), sampled every 1 ms\n"),
		"{output}"
	);
	assert!(output.contains("800  80.0%    work"), "{output}");
	assert!(
		output.ends_with(
			"[Summarized view of a macOS `sample` call-tree report. Use ':raw' to read the original \
			 file.]"
		),
		"{output}"
	);
	assert!(sources.snapshots.lock().is_empty());
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
		"[Path 'lost.txt' not found; resolved to 'nested/lost.txt' via suffix match]\nfound"
	);

	sources.file("one.txt", "alpha");
	sources.file("two.txt", "beta");
	assert_eq!(
		text(sources, r#"{"path":"one.txt:raw;two.txt:raw"}"#).await,
		"Note: interpreted as 2 paths: one.txt:raw, two.txt:raw\n\nalpha\n\nbeta"
	);
}
