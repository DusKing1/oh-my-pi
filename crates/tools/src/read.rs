//! Pi-compatible reads of local and special resources.

use std::{future::Future, path::Path};

use async_stream::stream;
use bytes::Bytes;
use futures::{FutureExt as _, Stream, pin_mut, select_biased};
use omp_core::Str;
use omp_tool::{
	Abort, ArgIssue, ArgIssueKind, BlobRef, CommitError, Constraint, Ev, IncomingParams, Outcome,
	ParamError, Part, PromptCaps, Rev, Tool, ToolSpec,
};
use serde::{Deserialize, Serialize};

use crate::render::{
	TextProjection,
	truncate::{TruncationOptions, append_blob_truncation_notice, truncate_head},
};

pub mod archive;
pub mod conflicts;
pub mod dirtree;
pub mod format;
pub mod image;
pub mod markit;
pub mod notebook;
pub mod profile;
pub mod selector;
pub mod sqlite;
pub mod web;

const SCHEMA: &[u8] = br#"{"type":"object","additionalProperties":false,"required":["path"],"properties":{"path":{"type":"string","description":"Local path, internal URI (e.g. skill://), or URL. Inline selectors are supported."}}}"#;
const DESCRIPTION: &str = r#"Read files, directories, archives, SQLite, images, documents, and web URLs via `path`.

<instruction>
- SHOULD parallelize independent reads.
- SHOULD use `read` (not browser) for web content; browser only when `read` can't deliver.
</instruction>

## Selectors — append `:<sel>` to `path` (e.g. `src/foo.ts:50-200`, `src/foo.ts:raw`, `db.sqlite:users:42`)
- `:50` / `:50-` — from line 50 | `:50-200` — inclusive | `:50+150` — 150 lines from 50 | `:5-16,960-973` — multiple ranges
- `:raw` — verbatim, no anchors/prefixes | `:2-4:raw` / `:raw:2-4` — range + verbatim
- `:conflicts` — one line per unresolved git merge conflict block

## Source kinds
- Parseable code, no selector → structural summary (declarations only, body elided). Footer names recovery selector — re-issue ONLY those ranges.
- File + selector → `[foo.ts#1A2B]` snapshot header + numbered lines. Copy `[FILENAME#TAG]` for anchored edits; NEVER fabricate the tag.
- Directory → depth-limited dirent listing.
- SQLite (`.sqlite`, `.sqlite3`, `.db`, `.db3`): `file.db` (tables), `file.db:table` (schema+rows), `file.db:table:key` (by PK), `?limit=`/`?where=`/`?q=SELECT`.
- Archives (`.tar`, `.tar.gz`, `.tgz`, `.zip`, plus ZIP-based `.jar`/`.war`/`.ear`/`.apk`): `archive.ext:path/inside/archive` reads a member.
- Documents → extracted text. Notebooks → editable cells. Images → decoded inline. `:raw` bypasses converters.
- URLs → reader-mode clean text/markdown; `:raw` → untouched HTML. Bare `host:port` needs trailing slash.
- Literal `:`, `?`, `#` in URI-like member paths → percent-encode (`%3A`/`%3F`/`%23`).

<critical>
Summary footer names elided ranges? Re-issue ONLY those ranges. NEVER guess `..`/`…` content.
</critical>"#;
const MAX_SUMMARY_BYTES: u64 = 2 * 1024 * 1024;
const MIN_SUMMARY_LINES: usize = 100;
const MAX_SUMMARY_LINES: usize = 20_000;
/// Maximum editable whole-file snapshot retained across read and write.
pub const SNAPSHOT_MAX_BYTES: usize = 4 * 1024 * 1024;

/// Arguments accepted by `read@1`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
	/// Local path, internal URI, or URL, optionally ending in a selector.
	pub path: Str,
}

/// Ephemeral read progress.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Update {
	/// Progress phase description.
	pub phase: Str,
}

/// A local source's filesystem classification.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
	/// A regular file.
	File,
	/// A directory.
	Directory,
	/// A symbolic link whose target is classified by the resource owner.
	Symlink,
}

/// Metadata resolved by the app-owned source adapter.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SourceStat {
	/// Canonical local path used for subsequent resource calls.
	pub canonical_path: Str,
	/// Stable model-facing path relative to the workspace when possible.
	pub display_path:   Str,
	/// Source classification.
	pub kind:           SourceKind,
	/// Exact byte length for files.
	pub byte_len:       u64,
	/// Milliseconds since the Unix epoch, when available.
	pub modified_ms:    Option<u64>,
}

/// One recursive directory entry supplied to the pure directory renderer.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DirectoryEntry {
	/// Path relative to the listed directory.
	pub path:        Str,
	/// Entry classification.
	pub kind:        SourceKind,
	/// Exact file byte length, or zero for directories.
	pub byte_len:    u64,
	/// Milliseconds since the Unix epoch, when available.
	pub modified_ms: Option<u64>,
}

/// Depth-bounded directory metadata returned by the resource owner.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DirectorySource {
	/// Canonical listed root.
	pub root:      Str,
	/// Entries at depth one or two, relative to `root`.
	pub entries:   Vec<DirectoryEntry>,
	/// Whether the resource owner stopped before visiting every entry.
	pub truncated: bool,
}

/// One inclusive span whose text was shown to the model.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SeenRange {
	/// First shown one-based line.
	pub start_line: u64,
	/// Last shown one-based line.
	pub end_line:   u64,
}

/// Snapshot information recorded alongside a revision-pinned plain-file read.
#[derive(Clone, Debug)]
pub struct SnapshotRecord {
	/// Canonical path key.
	pub path:     Str,
	/// Pinned document revision.
	pub revision: Str,
	/// Complete pinned bytes used to compute the hashline tag.
	pub bytes:    Bytes,
	/// Exact source line spans exposed by this result.
	pub seen:     Vec<SeenRange>,
}

/// An opaque revision-pinned lease for one plain file.
pub trait ReadLease: Send + Sync {
	/// Returns the pinned revision identity.
	fn revision(&self) -> &Str;
	/// Returns the canonical path represented by the lease.
	fn canonical_path(&self) -> &Str;
	/// Reads the complete pinned file bytes.
	fn read_all(&self) -> impl Future<Output = Result<Bytes, Fault>> + Send + '_;
}

/// App-owned local-source I/O boundary.
///
/// Rendering, document conversion, web fetching policy, and dispatch remain in
/// `omp-tools`; implementations provide local resources plus the low-level HTTP
/// transport inherited from [`web::types::HttpClient`].
pub trait ReadSources: web::types::HttpClient + Send + Sync + 'static {
	/// Revision-pinned plain-file lease type.
	type Lease: ReadLease;

	/// Stats an authored or canonical local path.
	fn stat(&self, path: Str) -> impl Future<Output = Result<SourceStat, Fault>> + Send + '_;
	/// Attempts unique workspace-suffix recovery for a missing authored path.
	fn resolve_suffix(
		&self,
		path: Str,
	) -> impl Future<Output = Result<Option<SourceStat>, Fault>> + Send + '_;
	/// Opens a revision-pinned lease for a plain file.
	fn open(&self, path: Str) -> impl Future<Output = Result<Self::Lease, Fault>> + Send + '_;
	/// Reads complete bytes for a special local source.
	fn read_bytes(&self, path: Str) -> impl Future<Output = Result<Bytes, Fault>> + Send + '_;
	/// Lists a directory recursively to the requested maximum depth.
	fn list_directory(
		&self,
		path: Str,
		max_depth: usize,
	) -> impl Future<Output = Result<DirectorySource, Fault>> + Send + '_;
	/// Reads a bounded prefix for magic-byte classification.
	fn read_prefix(
		&self,
		path: Str,
		max_bytes: usize,
	) -> impl Future<Output = Result<Bytes, Fault>> + Send + '_ {
		async move {
			let bytes = self.read_bytes(path).await?;
			Ok(bytes.slice(..bytes.len().min(max_bytes)))
		}
	}
	/// Records a hashline snapshot and its exposed line spans.
	fn record_snapshot(&self, record: SnapshotRecord) -> Result<Option<Str>, Fault>;
}

/// Stores binary bytes in the durable environment blob namespace.
pub trait ReadBlobs: Send + Sync + 'static {
	/// Stores bytes and returns a durable blob reference.
	fn store(
		&self,
		bytes: Bytes,
		media_type: Str,
	) -> impl Future<Output = Result<BlobRef, Fault>> + Send + '_;
}

/// One deterministic read result part.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PayloadPart {
	/// Model-visible UTF-8 text.
	Text {
		/// Exact text after read-level formatting and truncation.
		text: Str,
	},
	/// Durable binary media with a textual fallback.
	Blob {
		/// Stored media bytes.
		blob: BlobRef,
		/// Model-facing fallback and media description.
		alt:  Str,
	},
}

/// Durable, deterministic read truth.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Payload {
	/// Ordered text and blob parts.
	pub parts: Vec<PayloadPart>,
}

/// Typed read failure with an exact model-facing message.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Fault {
	/// Invalid selector or target syntax.
	Invalid { message: Str },
	/// Missing or unreadable local source.
	Source { message: Str },
	/// Unsupported internal resource seam.
	Unsupported { message: Str },
	/// Web transport, decoding, or rendering failure.
	Web { message: Str },
	/// Durable blob storage failure.
	Blob { message: Str },
}

impl Fault {
	/// Constructs a source failure.
	pub fn source(message: impl Into<Str>) -> Self {
		Self::Source { message: message.into() }
	}

	/// Returns the exact model-facing diagnostic.
	pub fn message(&self) -> &Str {
		match self {
			Self::Invalid { message }
			| Self::Source { message }
			| Self::Unsupported { message }
			| Self::Web { message }
			| Self::Blob { message } => message,
		}
	}
}

/// `read@1` executor over unboxed app resource adapters.
pub struct ReadTool<S, B> {
	sources: S,
	blobs:   B,
	spec:    ToolSpec,
}

/// Constructs the Pi-compatible `read@1` tool.
pub fn tool<S: ReadSources, B: ReadBlobs>(sources: S, blobs: B) -> ReadTool<S, B> {
	ReadTool {
		sources,
		blobs,
		spec: ToolSpec {
			name:        Str::new_static("read"),
			rev:         Rev { family: Str::new_static(""), n: 1 },
			description: Str::new_static(DESCRIPTION),
			schema:      Bytes::from_static(SCHEMA),
			constraint:  Constraint::Schema { priority: 10 },
		},
	}
}

impl<S: ReadSources, B: ReadBlobs> Tool for ReadTool<S, B> {
	type Fault = Fault;
	type Params = Params;
	type Payload = Payload;
	type Update = Update;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'c>(
		&'c self,
		mut incoming: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<Update, Payload, Fault>> + Send + 'c {
		stream! {
			let pulled = incoming.pull(|mut document| async move {
				let mut root = document.json().object();
				let _path = root.key("path").string().finish().await?;
				root.collect().await.map(|value| value.to_string())
			}).await;
			let raw = match pulled {
				Ok(value) => value,
				Err(ParamError::Args(issue)) if issue.kind == ArgIssueKind::Aborted => { yield Ev::Aborted(Abort::InputDropped); return; },
				Err(ParamError::Args(issue)) => { yield Ev::Args(issue); return; },
				Err(ParamError::Interrupted(interrupt)) => { yield Ev::Aborted(Abort::Interrupted { reason: interrupt.reason }); return; },
				Err(ParamError::Protocol(reason)) => { yield Ev::Args(protocol_issue(reason)); return; },
			};
			let params: Params = match serde_json::from_str(&raw) {
				Ok(value) => value,
				Err(_) => { yield Ev::Args(args_issue()); return; },
			};
			match incoming.committed().await {
				Ok(_) => {},
				Err(CommitError::Aborted) => { yield Ev::Aborted(Abort::InputDropped); return; },
				Err(CommitError::Interrupted(interrupt)) => { yield Ev::Aborted(Abort::Interrupted { reason: interrupt.reason }); return; },
				Err(CommitError::Protocol(reason)) => { yield Ev::Args(protocol_issue(reason)); return; },
			}
			let work = self.execute(params.path).fuse();
			let cancel = incoming.next_interrupt().fuse();
			pin_mut!(work, cancel);
			let result = select_biased! {
				value = work => value,
				interrupt = cancel => {
					let reason = interrupt.map(|value| value.reason).unwrap_or_else(|_| Str::new_static("invocation owner dropped"));
					yield Ev::Aborted(Abort::Interrupted { reason });
					return;
				},
			};
			yield done(result);
		}
	}

	fn prompt(&self, view: Result<&Payload, &Fault>, caps: &PromptCaps) -> Vec<Part> {
		let payload = match view {
			Ok(payload) => payload,
			Err(fault) => {
				let Some(mut projection) = TextProjection::new(caps) else {
					return Vec::new();
				};
				projection.push(fault.message());
				return projection.finish();
			},
		};
		let mut output = Vec::new();
		let mut remaining_text = caps.maximum_text_bytes as usize;
		for part in &payload.parts {
			if output.len() >= usize::from(caps.maximum_parts) {
				break;
			}
			match part {
				PayloadPart::Text { text } if remaining_text != 0 => {
					let mut end = text.len().min(remaining_text);
					while end != 0 && !text.is_char_boundary(end) {
						end -= 1;
					}
					if end != 0 {
						output.push(Part::Text { text: Str::from(&text[..end]) });
						remaining_text -= end;
					}
				},
				PayloadPart::Blob { blob, alt } if caps.media => {
					output.push(Part::Blob { blob: blob.clone(), alt: Some(alt.clone()) });
				},
				PayloadPart::Blob { alt, .. } if remaining_text != 0 => {
					let mut end = alt.len().min(remaining_text);
					while end != 0 && !alt.is_char_boundary(end) {
						end -= 1;
					}
					if end != 0 {
						output.push(Part::Text { text: Str::from(&alt[..end]) });
						remaining_text -= end;
					}
				},
				_ => {},
			}
		}
		output
	}
}

impl<S: ReadSources, B: ReadBlobs> ReadTool<S, B> {
	async fn execute(&self, authored: Str) -> Result<Payload, Fault> {
		let targets = self.split_targets(&authored).await?;
		let multiple = targets.len() > 1;
		let mut parts = Vec::new();
		if multiple {
			let names = targets
				.iter()
				.map(Str::as_str)
				.collect::<Vec<_>>()
				.join(", ");
			push_payload_part(&mut parts, PayloadPart::Text {
				text: Str::from(format!("Note: interpreted as {} paths: {}", targets.len(), names)),
			});
		}
		for target in &targets {
			match self.execute_target(target).await {
				Ok(section) => {
					for part in section {
						push_payload_part(&mut parts, part);
					}
				},
				Err(fault) if multiple => {
					push_payload_part(&mut parts, PayloadPart::Text {
						text: Str::from(format!("[Could not read {}: {}]", target, fault.message())),
					});
				},
				Err(fault) => return Err(fault),
			}
		}
		Ok(Payload { parts })
	}

	async fn split_targets(&self, authored: &str) -> Result<Vec<Str>, Fault> {
		if !authored.contains(';')
			|| !matches!(web::parse_target(authored), Ok(None))
			|| self.sources.stat(Str::from(authored)).await.is_ok()
		{
			return Ok(vec![Str::from(authored)]);
		}
		let targets = selector::split_semicolon_targets(authored);
		if targets.is_empty() {
			return Err(Fault::Invalid { message: Str::new_static("Path must not be empty") });
		}
		Ok(targets)
	}

	async fn execute_target(&self, authored: &str) -> Result<Vec<PayloadPart>, Fault> {
		if let Some(target) = web::parse_target(authored).map_err(|error| match error {
			web::types::WebError::InvalidUrl(message) => Fault::Invalid { message },
			other => Fault::Web { message: other.message() },
		})? {
			return self.read_web(target).await;
		}
		if let Some(message) = selector::classify_uri_target(authored).unsupported_message() {
			return Err(Fault::Unsupported { message });
		}

		let literal = self.sources.stat(Str::from(authored)).await.ok();
		let parsed_split = selector::split_path_and_selector(authored);
		let literal_wins = literal.is_some() && parsed_split.selector.is_some();
		let split = if literal_wins {
			selector::SplitPath { path: authored, selector: None }
		} else {
			parsed_split
		};
		let parsed = selector::parse_selector(split.selector)
			.map_err(|error| Fault::Invalid { message: Str::from(error.to_string()) })?;

		if !literal_wins {
			for candidate in archive::parse_archive_path_candidates(split.path) {
				if self
					.sources
					.stat(Str::from(candidate.archive_path.as_str()))
					.await
					.is_ok()
				{
					return self
						.read_archive(authored, &candidate.archive_path, &candidate.sub_path, &parsed)
						.await;
				}
			}
			for candidate in sqlite::parse_path_candidates(authored) {
				let database = candidate.sqlite_path.to_string_lossy();
				if let Ok(stat) = self.sources.stat(Str::from(database.as_ref())).await {
					return self.read_sqlite(authored, &stat).await;
				}
			}
			if let Some(stat) = literal
				.as_ref()
				.filter(|stat| stat.kind == SourceKind::File)
			{
				let prefix = self
					.sources
					.read_prefix(stat.canonical_path.clone(), 16)
					.await?;
				if sqlite::is_sqlite_target(&stat.display_path, &prefix) {
					return self.read_sqlite(authored, stat).await;
				}
			}
			if let Some(pdf) = pdf_image_member(split.path) {
				return Err(Fault::Unsupported {
					message: Str::from(format!(
						"PDF page-image members are not supported by the pdf-inspector backend; read {} \
						 for the extracted text",
						pdf
					)),
				});
			}
		}

		let mut recovered_from = None;
		let mut stat = if literal_wins {
			literal.expect("literal path was checked above")
		} else {
			match self.sources.stat(Str::from(split.path)).await {
				Ok(stat) => stat,
				Err(_) => {
					let stat = self
						.sources
						.resolve_suffix(Str::from(split.path))
						.await?
						.ok_or_else(|| Fault::source(format!("Path '{}' not found", split.path)))?;
					recovered_from = Some(split.path);
					stat
				},
			}
		};
		let suffix_from = recovered_from;

		if stat.kind == SourceKind::Directory {
			return self.read_directory(&stat, &parsed, suffix_from).await;
		}
		if stat.kind == SourceKind::Symlink {
			stat = self.sources.stat(stat.canonical_path.clone()).await?;
		}
		if matches!(parsed, selector::ParsedSelector::Conflicts) {
			let bytes = self.sources.read_bytes(stat.canonical_path.clone()).await?;
			let text = String::from_utf8_lossy(&bytes);
			let rendered = conflicts::render_conflicts_for_path(&text, &stat.display_path, false);
			let text = format::prepend_suffix_resolution_notice(
				&rendered.text,
				suffix_from.map(|from| format::SuffixResolution { from, to: &stat.display_path }),
			);
			return Ok(vec![PayloadPart::Text { text: Str::from(text) }]);
		}

		let raw = parsed.is_raw();
		let path = Path::new(stat.canonical_path.as_str());
		if !raw
			&& stat.byte_len <= profile::MAX_PROFILE_SUMMARY_BYTES
			&& (profile::is_cpu_profile_path(path) || profile::is_sample_profile_path(path))
		{
			let bytes = self.sources.read_bytes(stat.canonical_path.clone()).await?;
			if let Ok(text) = std::str::from_utf8(&bytes)
				&& let Some(summary) = profile::render_profile(path, text)
			{
				let lease = self.sources.open(stat.canonical_path.clone()).await?;
				let snapshot = Bytes::copy_from_slice(summary.as_bytes());
				return self
					.text_parts(
						&stat,
						&summary,
						&parsed,
						Some((lease.canonical_path(), lease.revision(), &snapshot)),
						suffix_from,
					)
					.await;
			}
		}
		let image_by_extension = image::is_supported_extension(path);
		let image_by_magic = if image_by_extension {
			true
		} else {
			let prefix = self
				.sources
				.read_prefix(stat.canonical_path.clone(), 256 * 1024)
				.await?;
			image::sniff_metadata(&prefix).is_some()
		};
		if image_by_magic && let Some(parts) = self.read_image(&stat).await? {
			return Ok(parts);
		}
		if !raw
			&& path
				.extension()
				.is_some_and(|ext| ext.eq_ignore_ascii_case("ipynb"))
		{
			let bytes = self.sources.read_bytes(stat.canonical_path.clone()).await?;
			let rendered = notebook::render(&bytes, &stat.display_path)
				.map_err(|error| Fault::Source { message: Str::from(error.message().to_owned()) })?;
			let lease = self.sources.open(stat.canonical_path.clone()).await?;
			let snapshot = Bytes::copy_from_slice(rendered.text.as_bytes());
			return self
				.text_parts(
					&stat,
					&rendered.text,
					&parsed,
					Some((lease.canonical_path(), lease.revision(), &snapshot)),
					suffix_from,
				)
				.await;
		}
		if is_document_path(path) {
			let bytes = self.sources.read_bytes(stat.canonical_path.clone()).await?;
			match markit::convert(path, &bytes) {
				Ok(Some(converted)) => {
					let mut text = converted.text.to_string();
					if let Some(note) = converted.note {
						text = format!("{note}\n{text}");
					}
					let lease = self.sources.open(stat.canonical_path.clone()).await?;
					let snapshot = Bytes::copy_from_slice(text.as_bytes());
					return self
						.text_parts(
							&stat,
							&text,
							&parsed,
							Some((lease.canonical_path(), lease.revision(), &snapshot)),
							suffix_from,
						)
						.await;
				},
				Ok(None) => {},
				Err(_) => {
					let notice = binary_notice(&stat);
					let text = format::prepend_suffix_resolution_notice(
						&notice,
						suffix_from.map(|from| format::SuffixResolution { from, to: &stat.display_path }),
					);
					return Ok(vec![PayloadPart::Text { text: Str::from(text) }]);
				},
			}
		}

		let lease = self.sources.open(stat.canonical_path.clone()).await?;
		let bytes = lease.read_all().await?;
		let text = match String::from_utf8(bytes.to_vec()) {
			Ok(text) => text,
			Err(error) if raw => String::from_utf8_lossy(error.as_bytes()).into_owned(),
			Err(_) => {
				return Err(Fault::Source { message: Str::from(binary_notice(&stat)) });
			},
		};
		if !raw
			&& matches!(parsed, selector::ParsedSelector::None)
			&& stat.byte_len <= MAX_SUMMARY_BYTES
			&& (MIN_SUMMARY_LINES..=MAX_SUMMARY_LINES).contains(&text.lines().count())
			&& let Some(summary) = structural_summary(&stat.display_path, &text)
		{
			return self
				.structural_parts(
					&stat,
					summary,
					lease.canonical_path(),
					lease.revision(),
					&bytes,
					suffix_from,
				)
				.await;
		}
		self
			.text_parts(
				&stat,
				&text,
				&parsed,
				Some((lease.canonical_path(), lease.revision(), &bytes)),
				suffix_from,
			)
			.await
	}

	async fn read_web(&self, target: web::ParsedTarget) -> Result<Vec<PayloadPart>, Fault> {
		let fetched = web::read_resource(&self.sources, &target.url, target.selector.is_raw())
			.await
			.map_err(|error| Fault::Web { message: error.message() })?;
		let notes = if fetched.render.notes.is_empty() {
			String::new()
		} else {
			format!(
				"Notes: {}\n",
				fetched
					.render
					.notes
					.iter()
					.map(Str::as_str)
					.collect::<Vec<_>>()
					.join("; ")
			)
		};
		let framed = format!(
			"URL: {}\nContent-Type: {}\nMethod: {}\n{}\n---\n\n{}",
			fetched.final_url,
			fetched.render.content_type.as_deref().unwrap_or("unknown"),
			fetched.render.method,
			notes,
			fetched.render.content
		);
		let mut parts = if matches!(
			&target.selector,
			selector::ParsedSelector::None | selector::ParsedSelector::Raw
		) {
			self.truncate_text(framed).await?
		} else {
			self.virtual_text_parts(&framed, &target.selector).await?
		};
		if let Some(image) = fetched.image {
			let blob = self.blobs.store(image.data, image.media_type).await?;
			parts.push(PayloadPart::Blob { blob, alt: image.description });
		}
		Ok(parts)
	}

	async fn read_directory(
		&self,
		stat: &SourceStat,
		parsed: &selector::ParsedSelector,
		suffix_from: Option<&str>,
	) -> Result<Vec<PayloadPart>, Fault> {
		if parsed.is_multi_range() {
			return Err(Fault::Invalid {
				message: Str::new_static(
					"Multi-range line selectors are not supported for directory listings.",
				),
			});
		}
		let source = self
			.sources
			.list_directory(stat.canonical_path.clone(), dirtree::MAX_DEPTH)
			.await?;
		let entries = source
			.entries
			.iter()
			.map(|entry| dirtree::DirEntry {
				relative_path: entry.path.clone(),
				is_dir:        entry.kind == SourceKind::Directory,
				size:          entry.byte_len,
				modified_ms:   entry.modified_ms.unwrap_or(0),
			})
			.collect::<Vec<_>>();
		let (offset, limit) = parsed.offset_limit();
		let offset = offset.and_then(|value| usize::try_from(value).ok());
		let limit = limit.and_then(|value| usize::try_from(value).ok());
		let now_ms = std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.unwrap_or_default()
			.as_millis() as u64;
		let rendered = dirtree::render_directory(
			stat.display_path.clone(),
			&entries,
			source.truncated,
			now_ms,
			offset,
			limit,
		);
		let mut text = rendered.text.to_string();
		if let Some(from) = suffix_from {
			text = format::prepend_suffix_resolution_notice(
				&text,
				Some(format::SuffixResolution { from, to: &stat.display_path }),
			);
		}
		Ok(vec![PayloadPart::Text { text: Str::from(text) }])
	}

	async fn read_archive(
		&self,
		_authored: &str,
		archive_path: &str,
		member: &str,
		parsed: &selector::ParsedSelector,
	) -> Result<Vec<PayloadPart>, Fault> {
		let stat = self.sources.stat(Str::from(archive_path)).await?;
		let bytes = self.sources.read_bytes(stat.canonical_path.clone()).await?;
		let archive_format = archive::archive_format_from_path(archive_path)
			.or_else(|| archive::sniff_archive_format(&bytes))
			.ok_or_else(|| Fault::source(format!("Unsupported archive format: {}", archive_path)))?;
		let result = archive::read_archive_bytes(bytes, archive_format, member, parsed.clone())
			.map_err(|error| Fault::Source { message: Str::from(error.to_string()) })?;
		match result.content {
			archive::ArchiveContent::Directory(listing) => {
				Ok(vec![PayloadPart::Text { text: Str::from(listing.render()) }])
			},
			archive::ArchiveContent::Text(member_text) => {
				let display_path = if member_text.node.path.is_empty() {
					stat.display_path.clone()
				} else {
					Str::from(format!("{}:{}", stat.display_path, member_text.node.path))
				};
				let member_stat = SourceStat { display_path, ..stat };
				self
					.text_parts(&member_stat, &member_text.text, &result.selector, None, None)
					.await
			},
			archive::ArchiveContent::Binary(member_binary) => {
				Ok(vec![PayloadPart::Text { text: Str::from(member_binary.notice) }])
			},
		}
	}

	async fn read_sqlite(
		&self,
		authored: &str,
		stat: &SourceStat,
	) -> Result<Vec<PayloadPart>, Fault> {
		let rendered = sqlite::read(Path::new(stat.canonical_path.as_str()), authored)
			.map_err(|error| Fault::Source { message: Str::from(error.to_string()) })?;
		Ok(vec![PayloadPart::Text { text: Str::from(rendered) }])
	}

	async fn read_image(&self, stat: &SourceStat) -> Result<Option<Vec<PayloadPart>>, Fault> {
		let bytes = self.sources.read_bytes(stat.canonical_path.clone()).await?;
		let Some(loaded) = image::process_image(bytes)
			.map_err(|error| Fault::Source { message: Str::from(error.message().to_owned()) })?
		else {
			return Ok(None);
		};
		let blob = self
			.blobs
			.store(loaded.data, loaded.media_type.clone())
			.await?;
		Ok(Some(vec![PayloadPart::Text { text: loaded.description.clone() }, PayloadPart::Blob {
			blob,
			alt: loaded.description,
		}]))
	}

	async fn structural_parts(
		&self,
		stat: &SourceStat,
		mut summary: StructuralRender,
		path: &Str,
		revision: &Str,
		bytes: &Bytes,
		suffix_from: Option<&str>,
	) -> Result<Vec<PayloadPart>, Fault> {
		let tag = self.sources.record_snapshot(SnapshotRecord {
			path:     path.clone(),
			revision: revision.clone(),
			bytes:    bytes.clone(),
			seen:     summary.seen,
		})?;
		if let Some(tag) = tag {
			let header = format::format_read_hashline_header(&stat.display_path, &tag);
			summary.text = format!("{}\n{}", header, summary.text);
		}
		if let Some(from) = suffix_from {
			summary.text = format::prepend_suffix_resolution_notice(
				&summary.text,
				Some(format::SuffixResolution { from, to: &stat.display_path }),
			);
		}
		self.truncate_text(summary.text).await
	}

	async fn text_parts(
		&self,
		stat: &SourceStat,
		text: &str,
		parsed: &selector::ParsedSelector,
		pinned: Option<(&Str, &Str, &Bytes)>,
		suffix_from: Option<&str>,
	) -> Result<Vec<PayloadPart>, Fault> {
		let mut options = format::TextFormatOptions::new("file");
		options.block_context =
			format::BlockContextSource { path: Some(&stat.display_path), language: None };
		let preformatted = format::format_text(text, parsed, options);
		let tag = if let Some((path, revision, bytes)) = pinned {
			self.sources.record_snapshot(SnapshotRecord {
				path:     path.clone(),
				revision: revision.clone(),
				bytes:    bytes.clone(),
				seen:     seen_ranges(&preformatted.seen_lines),
			})?
		} else {
			None
		};
		let mut formatted = if !parsed.is_raw() {
			if let Some(tag) = tag.as_deref() {
				options.snapshot = Some(format::SnapshotHeader { anchor: &stat.display_path, tag });
				format::format_text(text, parsed, options)
			} else {
				preformatted
			}
		} else {
			preformatted
		};
		if let Some(from) = suffix_from {
			formatted.prepend_suffix_resolution_notice(from, &stat.display_path);
		}
		if !parsed.is_raw() {
			let warning = conflicts::render_conflict_warning(text);
			formatted.append_conflict_warning(&warning.text);
		}
		let projection = formatted.full_text.unwrap_or(formatted.text);
		self.truncate_text(projection).await
	}

	async fn virtual_text_parts(
		&self,
		text: &str,
		parsed: &selector::ParsedSelector,
	) -> Result<Vec<PayloadPart>, Fault> {
		let formatted =
			format::format_text(text, parsed, format::TextFormatOptions::new("URL output"));
		let projection = formatted.full_text.unwrap_or(formatted.text);
		self.truncate_text(projection).await
	}

	async fn truncate_text(&self, text: String) -> Result<Vec<PayloadPart>, Fault> {
		let truncated = truncate_head(&text, TruncationOptions::default());
		if !truncated.truncated {
			return Ok(vec![PayloadPart::Text { text: Str::from(text) }]);
		}
		let blob = self
			.blobs
			.store(
				Bytes::copy_from_slice(text.as_bytes()),
				Str::new_static("text/plain; charset=utf-8"),
			)
			.await?;
		let mut visible = truncated.content.to_owned();
		append_blob_truncation_notice(&mut visible, &truncated, &blob.hash);
		Ok(vec![PayloadPart::Text { text: Str::from(visible) }])
	}
}

fn pdf_image_member(input: &str) -> Option<&str> {
	let lower = input.to_ascii_lowercase();
	let index = lower.find(".pdf:")?;
	let member = &lower[index + 5..];
	[".png", ".jpg", ".jpeg", ".webp"]
		.iter()
		.any(|extension| member.ends_with(extension))
		.then_some(&input[..index + 4])
}

fn is_document_path(path: &Path) -> bool {
	path
		.extension()
		.and_then(|value| value.to_str())
		.is_some_and(|extension| {
			matches!(
				extension.to_ascii_lowercase().as_str(),
				"pdf" | "docx" | "xlsx" | "pptx" | "epub"
			)
		})
}

fn binary_notice(stat: &SourceStat) -> String {
	format!(
		"[Cannot read binary file '{}' ({}); not valid UTF-8 text. Use ':raw' to read bytes \
		 verbatim.]",
		stat.display_path,
		format::format_bytes(stat.byte_len),
	)
}

struct StructuralRender {
	text: String,
	seen: Vec<SeenRange>,
}

fn structural_summary(path: &str, text: &str) -> Option<StructuralRender> {
	enum Unit {
		Line { number: usize, text: String },
		Elided { start: usize, end: usize },
	}

	let summary = omp_ast::summary::summarize_source(text, omp_ast::summary::SummarySettings {
		path: Some(path),
		..Default::default()
	})
	.ok()?;
	if !summary.parsed || !summary.elided {
		return None;
	}
	let mut units = Vec::new();
	for segment in summary.segments {
		let start = segment.start_line as usize;
		let end = segment.end_line as usize;
		if segment.kind == "kept" {
			for (offset, line) in segment.text.unwrap_or_default().lines().enumerate() {
				units.push(Unit::Line { number: start + offset, text: line.to_owned() });
			}
		} else {
			units.push(Unit::Elided { start, end });
		}
	}

	let mut rows = Vec::new();
	let mut seen = Vec::new();
	let mut elided = Vec::new();
	let mut elided_lines = 0;
	let mut index = 0;
	while index < units.len() {
		if let (
			Some(Unit::Line { number: start, text: head }),
			Some(Unit::Elided { .. }),
			Some(Unit::Line { number: end, text: tail }),
		) = (units.get(index), units.get(index + 1), units.get(index + 2))
			&& format::can_merge_brace_pair(head, tail)
		{
			rows.push(format::format_merged_brace_line(*start, *end, head, tail).model);
			seen.push(SeenRange { start_line: *start as u64, end_line: *start as u64 });
			seen.push(SeenRange { start_line: *end as u64, end_line: *end as u64 });
			elided.push(format::ElidedRange { start: *start, end: *end });
			elided_lines += end.saturating_sub(*start).saturating_sub(1);
			index += 3;
			continue;
		}
		match &units[index] {
			Unit::Line { number, text } => {
				rows.push(format!("{}:{}", number, text));
				seen.push(SeenRange { start_line: *number as u64, end_line: *number as u64 });
			},
			Unit::Elided { start, end } => {
				rows.push("…".to_owned());
				elided.push(format::ElidedRange { start: *start, end: *end });
				elided_lines += end - start + 1;
			},
		}
		index += 1;
	}
	let footer = format::format_summary_elision_footer(path, &elided, elided_lines);
	let mut output = rows.join("\n");
	if !footer.is_empty() {
		output.push_str("\n\n");
		output.push_str(&footer);
	}
	Some(StructuralRender { text: output, seen })
}

fn seen_ranges(lines: &[usize]) -> Vec<SeenRange> {
	let mut ranges = Vec::new();
	let mut iter = lines.iter().copied();
	let Some(mut start) = iter.next() else {
		return ranges;
	};
	let mut end = start;
	for line in iter {
		if line == end.saturating_add(1) {
			end = line;
		} else {
			ranges.push(SeenRange { start_line: start as u64, end_line: end as u64 });
			start = line;
			end = line;
		}
	}
	ranges.push(SeenRange { start_line: start as u64, end_line: end as u64 });
	ranges
}

fn push_payload_part(parts: &mut Vec<PayloadPart>, part: PayloadPart) {
	match (parts.last_mut(), part) {
		(Some(PayloadPart::Text { text: previous }), PayloadPart::Text { text }) => {
			let mut combined = String::with_capacity(previous.len() + text.len() + 2);
			combined.push_str(previous);
			combined.push_str("\n\n");
			combined.push_str(&text);
			*previous = Str::from(combined);
		},
		(_, part) => parts.push(part),
	}
}

fn done(result: Result<Payload, Fault>) -> Ev<Update, Payload, Fault> {
	Ev::Done(Outcome::Done { result, useless: false })
}

fn args_issue() -> ArgIssue {
	ArgIssue {
		path:     Vec::new(),
		expected: Str::new_static("read@1 arguments"),
		kind:     ArgIssueKind::Malformed,
		example:  None,
		found:    None,
	}
}

fn protocol_issue(reason: Str) -> ArgIssue {
	ArgIssue {
		path:     Vec::new(),
		expected: Str::new_static("linear invocation frames"),
		kind:     ArgIssueKind::Protocol,
		example:  Some(reason),
		found:    None,
	}
}
