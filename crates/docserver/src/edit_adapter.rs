//! Session-scoped lowering of opaque text-edit intents into canonical byte
//! edits.

use std::{collections::HashMap, path::Path, sync::Arc};

use bytes::Bytes;
use omp_core::Str;
use omp_hashline::{
	ApplyMode, ApplyOptions, Clipboard, Patch, RecoveryEdit, ReplaceOptions, RevisionToken,
	SnapshotStore, apply_parsed_patch, apply_replace, loop_guard::NoopLoopGuard, recover_exact,
	recovery::ByteRange as RecoveryByteRange,
};
use parking_lot::{Mutex, RwLock};
use serde::Deserialize;
use smallvec::SmallVec;

use crate::{ByteEdit, ByteRange, DocumentSnapshot, Error, ReadSelection, Result, validate_edits};

/// Built-in hashline edit-intent format name.
pub const HASHLINE_EDIT_FORMAT: &str = "omp.hashline";
/// Built-in exact/fuzzy replacement edit-intent format name.
pub const REPLACE_EDIT_FORMAT: &str = "omp.replace";

/// A disk-free lowering strategy for one opaque edit format.
///
/// Implementations receive the exact immutable transaction base. They must not
/// consult ambient filesystem state, and their edits must use coordinates from
/// that base. The registry validates this contract before returning an output.
pub trait TextEditAdapter: Send + Sync {
	/// Records which part of an exact snapshot was returned to this session.
	///
	/// Stateful adapters may retain the shared snapshot and derive authorization
	/// provenance from `selection`. Stateless adapters need not override this.
	fn record_snapshot(
		&self,
		_path: &Path,
		_snapshot: Arc<DocumentSnapshot>,
		_selection: &ReadSelection,
	) -> Result<()> {
		Ok(())
	}

	/// Lowers an opaque payload into sorted base-coordinate byte edits.
	fn lower(
		&self,
		path: &Path,
		base_snapshot: Arc<DocumentSnapshot>,
		payload: Bytes,
		options_json: Bytes,
	) -> Result<Vec<ByteEdit>>;
}

#[derive(Clone)]
enum Adapter {
	Hashline(Arc<HashlineAdapter>),
	Replace(Arc<ReplaceAdapter>),
	Boxed(Arc<dyn TextEditAdapter>),
}

impl Adapter {
	fn record_snapshot(
		&self,
		path: &Path,
		snapshot: Arc<DocumentSnapshot>,
		selection: &ReadSelection,
	) -> Result<()> {
		match self {
			Self::Hashline(adapter) => adapter.record_snapshot(path, snapshot, selection),
			Self::Replace(adapter) => adapter.record_snapshot(path, snapshot, selection),
			Self::Boxed(adapter) => adapter.record_snapshot(path, snapshot, selection),
		}
	}

	fn lower(
		&self,
		path: &Path,
		base_snapshot: Arc<DocumentSnapshot>,
		payload: Bytes,
		options_json: Bytes,
	) -> Result<Vec<ByteEdit>> {
		match self {
			Self::Hashline(adapter) => adapter.lower(path, base_snapshot, payload, options_json),
			Self::Replace(adapter) => adapter.lower(path, base_snapshot, payload, options_json),
			Self::Boxed(adapter) => adapter.lower(path, base_snapshot, payload, options_json),
		}
	}
}

/// A connection-local registry of opaque text-edit adapters.
///
/// Construct one registry per connection or session. In particular, sharing a
/// registry would also share hashline read provenance, retained snapshots,
/// clipboard registers, and no-op counters.
pub struct EditAdapterRegistry {
	adapters: RwLock<HashMap<Str, Adapter>>,
}

impl Default for EditAdapterRegistry {
	fn default() -> Self {
		Self::new()
	}
}
impl EditAdapterRegistry {
	/// Creates an empty session-scoped registry.
	#[must_use]
	pub fn new() -> Self {
		Self { adapters: RwLock::new(HashMap::new()) }
	}

	/// Creates a session-scoped registry containing `omp.hashline` and
	/// `omp.replace`.
	///
	/// `omp.hashline` accepts the raw UTF-8 hashline patch as its payload and
	/// either empty options or `{}`. `omp.replace` accepts payload JSON
	/// `{ \"old_text\": string, \"new_text\": string }`; its optional JSON
	/// options are `replace_all: bool`, `allow_fuzzy: bool`, and
	/// `threshold: number`, with omitted fields inheriting [`ReplaceOptions`]
	/// defaults.
	#[must_use]
	pub fn with_built_ins() -> Self {
		let mut adapters = HashMap::new();
		adapters.insert(
			HASHLINE_EDIT_FORMAT.into(),
			Adapter::Hashline(Arc::new(HashlineAdapter::default())),
		);
		adapters.insert(REPLACE_EDIT_FORMAT.into(), Adapter::Replace(Arc::new(ReplaceAdapter)));
		Self { adapters: RwLock::new(adapters) }
	}

	/// Registers one format for this session, rejecting empty or duplicate
	/// names.
	pub fn register(
		&self,
		format: impl Into<Str>,
		adapter: Arc<dyn TextEditAdapter>,
	) -> Result<()> {
		let format = format.into();
		if format.is_empty() {
			return Err(Error::InvalidTarget {
				target: format,
				reason: Str::new_static("edit format must not be empty"),
			});
		}
		let mut adapters = self.adapters.write();
		if adapters.contains_key(&format) {
			return Err(Error::InvalidTarget {
				target: format,
				reason: Str::new_static("edit format is already registered in this session"),
			});
		}
		adapters.insert(format, Adapter::Boxed(adapter));
		Ok(())
	}

	/// Records an exact read and its selection with every adapter currently
	/// registered in this session.
	pub fn record_snapshot(
		&self,
		path: &Path,
		snapshot: Arc<DocumentSnapshot>,
		selection: &ReadSelection,
	) -> Result<()> {
		let adapters = self
			.adapters
			.read()
			.values()
			.cloned()
			.collect::<SmallVec<Adapter, 4>>();
		for adapter in adapters {
			adapter.record_snapshot(path, snapshot.clone(), selection)?;
		}
		Ok(())
	}

	/// Lowers one opaque intent and validates its sorted base-coordinate edits.
	pub fn lower(
		&self,
		format: &str,
		path: &Path,
		base_snapshot: Arc<DocumentSnapshot>,
		payload: Bytes,
		options_json: Bytes,
	) -> Result<Vec<ByteEdit>> {
		let adapter =
			self
				.adapters
				.read()
				.get(format)
				.cloned()
				.ok_or_else(|| Error::InvalidTarget {
					target: Str::new(format),
					reason: Str::new_static("unknown edit format"),
				})?;
		let base_len =
			u64::try_from(base_snapshot.content().len()).map_err(|_| Error::InvalidContent {
				reason: Str::new_static("base snapshot is too large for byte coordinates"),
			})?;
		let edits = adapter.lower(path, base_snapshot, payload, options_json)?;
		validate_edits(base_len, &edits)?;
		Ok(edits)
	}
}

#[derive(Default)]
struct HashlineAdapter {
	state: Mutex<HashlineState>,
}

#[derive(Default)]
struct HashlineState {
	snapshots: SnapshotStore,
	clipboard: Clipboard,
	noops:     NoopLoopGuard,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HashlineOptions {}

impl TextEditAdapter for HashlineAdapter {
	fn record_snapshot(
		&self,
		path: &Path,
		snapshot: Arc<DocumentSnapshot>,
		selection: &ReadSelection,
	) -> Result<()> {
		let path = path_key(path)?;
		let seen_lines = selected_lines(snapshot.content(), selection)?;
		let revision = revision_token(&snapshot);
		self
			.state
			.lock()
			.snapshots
			.record(path, revision, snapshot.content().clone(), seen_lines)
			.map_err(|error| Error::Protocol {
				reason: Str::new(format!("could not retain hashline read snapshot: {error}")),
			})?;
		Ok(())
	}

	fn lower(
		&self,
		path: &Path,
		base_snapshot: Arc<DocumentSnapshot>,
		payload: Bytes,
		options_json: Bytes,
	) -> Result<Vec<ByteEdit>> {
		parse_hashline_options(&options_json)?;
		let path = path_key(path)?;
		let text = std::str::from_utf8(&payload).map_err(|error| Error::Protocol {
			reason: Str::new(format!("omp.hashline payload is not UTF-8: {error}")),
		})?;
		let patch = Patch::parse_default(text).map_err(hashline_content_error)?;
		if patch.sections.len() != 1 {
			return Err(Error::InvalidContent {
				reason: Str::new_static(
					"omp.hashline payload must contain exactly one file section",
				),
			});
		}
		let section = &patch.sections[0];
		if section.path != path {
			return Err(Error::InvalidTarget {
				target: section.path.clone(),
				reason: Str::new(format!(
					"hashline section path does not match transaction path {path}"
				)),
			});
		}
		let tag = section
			.file_hash
			.as_deref()
			.ok_or_else(|| Error::InvalidContent {
				reason: Str::new_static("omp.hashline section must include an exact snapshot tag"),
			})?;
		let parsed = section.parse().map_err(hashline_content_error)?;
		if parsed.file_op.is_some() {
			return Err(Error::InvalidContent {
				reason: Str::new_static(
					"omp.hashline text intents cannot contain filesystem operations",
				),
			});
		}
		let anchor_lines = section
			.collect_anchor_lines()
			.map_err(hashline_content_error)?;

		let mut state = self.state.lock();
		let retained = state
			.snapshots
			.resolve(&path, tag, None)
			.map_err(hashline_content_error)?;
		if let Some(unseen) = anchor_lines
			.iter()
			.find(|line| !retained.seen_lines().contains(line))
		{
			return Err(Error::InvalidTarget {
				target: path.clone(),
				reason: Str::new(format!(
					"hashline line {unseen} was not present in this session's read of {path}#{tag}"
				)),
			});
		}

		let mut clipboard = state.clipboard.start_batch();
		let applied =
			apply_parsed_patch(retained.bytes().clone(), &parsed, &mut clipboard, ApplyOptions {
				mode: ApplyMode::Strict,
				path: Some(&path),
			})
			.map_err(hashline_content_error)?;
		let authored = applied
			.edits
			.iter()
			.map(|edit| {
				let start = u64::try_from(edit.start).map_err(|_| Error::InvalidContent {
					reason: Str::new_static("hashline edit start exceeds byte coordinates"),
				})?;
				let end = u64::try_from(edit.end).map_err(|_| Error::InvalidContent {
					reason: Str::new_static("hashline edit end exceeds byte coordinates"),
				})?;
				let range = RecoveryByteRange::new(start, end).map_err(hashline_content_error)?;
				Ok(RecoveryEdit::new(range, edit.replacement.clone()))
			})
			.collect::<Result<Vec<_>>>()?;
		let recovered = recover_exact(retained.bytes(), base_snapshot.content(), &authored)
			.map_err(hashline_content_error)?;
		let edits = recovered
			.canonical_edits()
			.iter()
			.map(|edit| {
				let range = ByteRange::new(edit.range().start(), edit.range().end())?;
				Ok(ByteEdit::new(range, edit.replacement().clone()))
			})
			.collect::<Result<Vec<_>>>()?;

		state.clipboard.commit_named_from(&clipboard);
		if edits.is_empty() {
			let noop = state.noops.record_noop(path, payload);
			if noop.should_escalate() {
				return Err(Error::InvalidContent { reason: Str::new(noop.diagnostic()) });
			}
		} else {
			state.noops.reset(&path);
		}
		Ok(edits)
	}
}

struct ReplaceAdapter;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplacePayload {
	old_text: String,
	new_text: String,
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ReplaceAdapterOptions {
	replace_all: bool,
	allow_fuzzy: bool,
	threshold:   f64,
}

impl Default for ReplaceAdapterOptions {
	fn default() -> Self {
		let options = ReplaceOptions::default();
		Self {
			replace_all: options.replace_all,
			allow_fuzzy: options.allow_fuzzy,
			threshold:   options.threshold,
		}
	}
}

impl TextEditAdapter for ReplaceAdapter {
	/// `payload` is JSON `{ "old_text": string, "new_text": string }`.
	/// `options_json` is an optional JSON object with `replace_all`,
	/// `allow_fuzzy`, and `threshold`; omitted fields use [`ReplaceOptions`]
	/// defaults.
	fn lower(
		&self,
		_path: &Path,
		base_snapshot: Arc<DocumentSnapshot>,
		payload: Bytes,
		options_json: Bytes,
	) -> Result<Vec<ByteEdit>> {
		let payload: ReplacePayload =
			serde_json::from_slice(&payload).map_err(|error| Error::Protocol {
				reason: Str::new(format!("malformed omp.replace payload JSON: {error}")),
			})?;
		let options = if options_json.is_empty() {
			ReplaceAdapterOptions::default()
		} else {
			serde_json::from_slice(&options_json).map_err(|error| Error::Protocol {
				reason: Str::new(format!("malformed omp.replace options JSON: {error}")),
			})?
		};
		let result = apply_replace(
			base_snapshot.content(),
			&payload.old_text,
			&payload.new_text,
			ReplaceOptions {
				replace_all: options.replace_all,
				allow_fuzzy: options.allow_fuzzy,
				threshold:   options.threshold,
			},
		)
		.map_err(|error| Error::InvalidContent {
			reason: Str::new(format!("omp.replace could not be applied: {error}")),
		})?;
		result
			.edits
			.into_iter()
			.map(|edit| {
				let start = u64::try_from(edit.start).map_err(|_| Error::InvalidContent {
					reason: Str::new_static("replace edit start exceeds byte coordinates"),
				})?;
				let end = u64::try_from(edit.end).map_err(|_| Error::InvalidContent {
					reason: Str::new_static("replace edit end exceeds byte coordinates"),
				})?;
				Ok(ByteEdit::new(ByteRange::new(start, end)?, edit.replacement))
			})
			.collect()
	}
}

fn parse_hashline_options(options: &[u8]) -> Result<()> {
	if options.is_empty() {
		return Ok(());
	}
	serde_json::from_slice::<HashlineOptions>(options)
		.map(|_| ())
		.map_err(|error| Error::Protocol {
			reason: Str::new(format!("malformed omp.hashline options JSON: {error}")),
		})
}

fn path_key(path: &Path) -> Result<Str> {
	path
		.to_str()
		.map(Str::new)
		.ok_or_else(|| Error::InvalidTarget {
			target: Str::new(path.to_string_lossy()),
			reason: Str::new_static("edit paths must be valid UTF-8"),
		})
}

fn revision_token(snapshot: &DocumentSnapshot) -> RevisionToken {
	let revision = snapshot.head().revision();
	let mut token = [0u8; 40];
	token[..8].copy_from_slice(&revision.sequence().to_be_bytes());
	token[8..].copy_from_slice(revision.content_hash());
	RevisionToken::new(token)
}

fn selected_lines(content: &Bytes, selection: &ReadSelection) -> Result<Vec<usize>> {
	let line_count = if content.is_empty() {
		0
	} else {
		String::from_utf8_lossy(content).lines().count()
	};
	match selection {
		ReadSelection::Whole => Ok((1..=line_count).collect()),
		ReadSelection::Bytes(_) => Ok(Vec::new()),
		ReadSelection::Lines(ranges) => {
			let upper_bound = u64::try_from(line_count).map_err(|_| Error::InvalidContent {
				reason: Str::new_static("snapshot has too many lines"),
			})?;
			let mut lines = Vec::new();
			for range in ranges {
				let range = range.validate(upper_bound)?;
				lines.extend((range.start() + 1..=range.end()).map(|line| line as usize));
			}
			lines.sort_unstable();
			lines.dedup();
			Ok(lines)
		},
	}
}

fn hashline_content_error(error: impl std::fmt::Display) -> Error {
	Error::InvalidContent {
		reason: Str::new(format!("omp.hashline could not be applied: {error}")),
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{DocumentHead, DocumentId, DocumentKind, DocumentPresence, LineRange, Revision};

	fn snapshot(sequence: u64, content: &'static [u8]) -> Arc<DocumentSnapshot> {
		let content = Bytes::from_static(content);
		let head = DocumentHead::new(
			DocumentId::from_bytes([7; 16]),
			Revision::for_content(sequence, &content),
			DocumentPresence::Present,
			DocumentKind::Text(None),
			content.len() as u64,
		)
		.expect("head");
		Arc::new(DocumentSnapshot::new(head, content).expect("snapshot"))
	}

	#[test]
	fn unknown_format_is_rejected() {
		let registry = EditAdapterRegistry::with_built_ins();
		let error = registry
			.lower(
				"example.unknown",
				Path::new("file.txt"),
				snapshot(1, b"text\n"),
				Bytes::new(),
				Bytes::new(),
			)
			.expect_err("unknown format");
		assert!(matches!(error, Error::InvalidTarget { .. }));
	}

	#[test]
	fn replace_returns_exact_base_coordinate_edit() {
		let registry = EditAdapterRegistry::with_built_ins();
		let edits = registry
			.lower(
				REPLACE_EDIT_FORMAT,
				Path::new("file.txt"),
				snapshot(1, b"before needle after\n"),
				Bytes::from_static(br#"{"old_text":"needle","new_text":"thread"}"#),
				Bytes::from_static(br#"{"allow_fuzzy":false}"#),
			)
			.expect("replace");
		assert_eq!(edits.len(), 1);
		assert_eq!(edits[0].range(), ByteRange::new(7, 13).expect("range"));
		assert_eq!(edits[0].replacement().as_ref(), b"thread");
	}

	#[test]
	fn hashline_recovers_stale_edit_onto_current_snapshot() {
		let registry = EditAdapterRegistry::with_built_ins();
		let old = snapshot(1, b"alpha\nbeta\ngamma\n");
		registry
			.record_snapshot(
				Path::new("file.txt"),
				old.clone(),
				&ReadSelection::Lines(vec![LineRange::new(1, 2).expect("lines")]),
			)
			.expect("record");
		let tag = omp_hashline::compute_snapshot_tag(old.content());
		let patch = Bytes::from(format!("[file.txt#{tag}]\nPUT 2.=2:\n+BETA\n"));
		let edits = registry
			.lower(
				HASHLINE_EDIT_FORMAT,
				Path::new("file.txt"),
				snapshot(2, b"alpha\nbeta\ngamma\nsuffix\n"),
				patch,
				Bytes::new(),
			)
			.expect("recover");
		assert_eq!(edits.len(), 1);
		assert_eq!(edits[0].range(), ByteRange::new(6, 10).expect("range"));
		assert_eq!(edits[0].replacement().as_ref(), b"BETA");
	}

	#[test]
	fn hashline_seen_lines_do_not_cross_sessions() {
		let first = EditAdapterRegistry::with_built_ins();
		let second = EditAdapterRegistry::with_built_ins();
		let read = snapshot(1, b"alpha\nbeta\ngamma\n");
		first
			.record_snapshot(
				Path::new("file.txt"),
				read.clone(),
				&ReadSelection::Lines(vec![LineRange::new(1, 2).expect("lines")]),
			)
			.expect("first record");
		second
			.record_snapshot(
				Path::new("file.txt"),
				read.clone(),
				&ReadSelection::Lines(vec![LineRange::new(0, 1).expect("lines")]),
			)
			.expect("second record");
		let tag = omp_hashline::compute_snapshot_tag(read.content());
		let patch = Bytes::from(format!("[file.txt#{tag}]\nPUT 2.=2:\n+BETA\n"));
		first
			.lower(
				HASHLINE_EDIT_FORMAT,
				Path::new("file.txt"),
				read.clone(),
				patch.clone(),
				Bytes::new(),
			)
			.expect("authorized session");
		let error = second
			.lower(HASHLINE_EDIT_FORMAT, Path::new("file.txt"), read, patch, Bytes::new())
			.expect_err("isolated provenance");
		assert!(matches!(error, Error::InvalidTarget { .. }));
	}
}
