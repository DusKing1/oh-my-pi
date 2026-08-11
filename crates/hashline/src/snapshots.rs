//! Exact-byte snapshot retention for hashline reads and stale-edit recovery.

use std::{
	collections::{BTreeSet, HashMap, VecDeque},
	error::Error,
	fmt,
	sync::Arc,
};

use bytes::Bytes;
use omp_core::{Str, fmts};

use crate::format::normalized_file_xxh32;

const DEFAULT_MAX_PATHS: usize = 30;
const DEFAULT_MAX_REVISIONS_PER_PATH: usize = 4;
const DEFAULT_MAX_TOTAL_BYTES: usize = 64 * 1024 * 1024;

/// An opaque daemon revision identity supplied by the snapshot producer.
///
/// The token, rather than the four-hex display tag, is the concurrency
/// identity.
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct RevisionToken(Bytes);

impl RevisionToken {
	/// Copies an opaque token into shared immutable storage.
	#[must_use]
	pub fn new(token: impl AsRef<[u8]>) -> Self {
		Self(Bytes::copy_from_slice(token.as_ref()))
	}

	/// Wraps an already shared opaque token without copying.
	#[must_use]
	pub const fn from_bytes(token: Bytes) -> Self {
		Self(token)
	}

	/// Returns the opaque token bytes for round-tripping to its caller.
	#[must_use]
	pub const fn as_bytes(&self) -> &Bytes {
		&self.0
	}
}

impl fmt::Debug for RevisionToken {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str("RevisionToken(<opaque>)")
	}
}

/// One immutable exact-byte file revision retained for a canonical path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Snapshot {
	path:        Str,
	bytes:       Bytes,
	tag:         Str,
	revision:    RevisionToken,
	recorded_at: u64,
	seen_lines:  Arc<BTreeSet<usize>>,
}

impl Snapshot {
	/// Returns the canonical path associated with this retained revision.
	#[must_use]
	pub fn path(&self) -> &str {
		&self.path
	}

	/// Returns the exact shared bytes, including any BOM and original newlines.
	#[must_use]
	pub const fn bytes(&self) -> &Bytes {
		&self.bytes
	}

	/// Returns the four-hex representation tag displayed in hashline headers.
	#[must_use]
	pub fn tag(&self) -> &str {
		&self.tag
	}

	/// Returns the exact opaque revision identity.
	#[must_use]
	pub const fn revision(&self) -> &RevisionToken {
		&self.revision
	}

	/// Returns the store-local observation sequence for LRU ordering.
	#[must_use]
	pub const fn recorded_at(&self) -> u64 {
		self.recorded_at
	}

	/// Returns one-indexed lines displayed from exactly this revision.
	#[must_use]
	pub fn seen_lines(&self) -> &BTreeSet<usize> {
		&self.seen_lines
	}
}

/// Bounds for the in-memory per-path and per-revision LRU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotStoreOptions {
	/// Maximum retained canonical paths.
	pub max_paths:              usize,
	/// Maximum retained revisions for each path.
	pub max_revisions_per_path: usize,
	/// Maximum summed exact-byte length across retained records.
	pub max_total_bytes:        usize,
}

impl Default for SnapshotStoreOptions {
	fn default() -> Self {
		Self {
			max_paths:              DEFAULT_MAX_PATHS,
			max_revisions_per_path: DEFAULT_MAX_REVISIONS_PER_PATH,
			max_total_bytes:        DEFAULT_MAX_TOTAL_BYTES,
		}
	}
}

/// A snapshot-store mutation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SnapshotStoreError {
	/// A configured retention bound was zero.
	ZeroCapacity,
	/// One revision token was reused with different exact bytes.
	RevisionConflict {
		/// Canonical path whose token was reused.
		path: Str,
	},
}

impl fmt::Display for SnapshotStoreError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::ZeroCapacity => formatter.write_str("snapshot retention capacities must be nonzero"),
			Self::RevisionConflict { path } => {
				write!(formatter, "revision token identifies conflicting bytes at {path}")
			},
		}
	}
}

impl Error for SnapshotStoreError {}

/// A collision-aware snapshot lookup failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SnapshotLookupError {
	/// No retained revision matches the requested path and identity.
	Missing {
		/// Canonical path requested by the caller.
		path: Str,
		/// Four-hex representation tag requested by the caller.
		tag:  Str,
	},
	/// Several exact revisions share the requested representation tag.
	Ambiguous {
		/// Canonical path containing the colliding revisions.
		path:       Str,
		/// Colliding four-hex representation tag.
		tag:        Str,
		/// Number of independently retained matching revisions.
		candidates: usize,
	},
}

impl fmt::Display for SnapshotLookupError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Missing { path, tag } => write!(formatter, "no retained snapshot for {path}#{tag}"),
			Self::Ambiguous { path, tag, candidates } => write!(
				formatter,
				"snapshot tag {path}#{tag} is ambiguous across {candidates} retained revisions"
			),
		}
	}
}

impl Error for SnapshotLookupError {}

#[derive(Debug)]
struct PathHistory {
	revisions:  VecDeque<Arc<Snapshot>>,
	touched_at: u64,
}

/// Bounded in-memory snapshot records keyed by canonical path and exact
/// revision.
#[derive(Debug)]
pub struct SnapshotStore {
	options:     SnapshotStoreOptions,
	histories:   HashMap<Str, PathHistory>,
	clock:       u64,
	total_bytes: usize,
}

impl Default for SnapshotStore {
	fn default() -> Self {
		Self::new(SnapshotStoreOptions::default()).expect("default snapshot capacities are nonzero")
	}
}

impl SnapshotStore {
	/// Creates an empty bounded store.
	pub fn new(options: SnapshotStoreOptions) -> Result<Self, SnapshotStoreError> {
		if options.max_paths == 0
			|| options.max_revisions_per_path == 0
			|| options.max_total_bytes == 0
		{
			return Err(SnapshotStoreError::ZeroCapacity);
		}
		Ok(Self { options, histories: HashMap::new(), clock: 0, total_bytes: 0 })
	}

	/// Records exact bytes for one caller-supplied revision and returns its
	/// display tag.
	pub fn record<I>(
		&mut self,
		path: impl Into<Str>,
		revision: RevisionToken,
		bytes: Bytes,
		seen_lines: I,
	) -> Result<Str, SnapshotStoreError>
	where
		I: IntoIterator<Item = usize>,
	{
		let path = path.into();
		let tag = compute_snapshot_tag(&bytes);
		let observed: BTreeSet<_> = seen_lines.into_iter().filter(|line| *line != 0).collect();
		let now = self.tick();

		if let Some(history) = self.histories.get_mut(&path)
			&& let Some(index) = history
				.revisions
				.iter()
				.position(|item| item.revision == revision)
		{
			let existing = &history.revisions[index];
			if existing.bytes != bytes {
				return Err(SnapshotStoreError::RevisionConflict { path });
			}
			let mut merged = existing.seen_lines.as_ref().clone();
			merged.extend(observed);
			let refreshed = Arc::new(Snapshot {
				recorded_at: now,
				seen_lines: Arc::new(merged),
				..existing.as_ref().clone()
			});
			history.revisions.remove(index);
			history.revisions.push_front(refreshed);
			history.touched_at = now;
			return Ok(tag);
		}

		let snapshot = Arc::new(Snapshot {
			path: path.clone(),
			bytes,
			tag: tag.clone(),
			revision,
			recorded_at: now,
			seen_lines: Arc::new(observed),
		});
		self.total_bytes = self.total_bytes.saturating_add(snapshot.bytes.len());
		let history = self
			.histories
			.entry(path)
			.or_insert_with(|| PathHistory { revisions: VecDeque::new(), touched_at: now });
		history.touched_at = now;
		history.revisions.push_front(snapshot);
		while history.revisions.len() > self.options.max_revisions_per_path {
			if let Some(evicted) = history.revisions.pop_back() {
				self.total_bytes = self.total_bytes.saturating_sub(evicted.bytes.len());
			}
		}
		self.enforce_bounds();
		Ok(tag)
	}

	/// Returns the most recently observed retained revision for a path.
	pub fn head(&mut self, path: &str) -> Option<Arc<Snapshot>> {
		let now = self.tick();
		let history = self.histories.get_mut(path)?;
		history.touched_at = now;
		history.revisions.front().cloned()
	}

	/// Returns the retained record carrying an exact revision token.
	pub fn by_revision(&mut self, path: &str, revision: &RevisionToken) -> Option<Arc<Snapshot>> {
		let now = self.tick();
		let history = self.histories.get_mut(path)?;
		let index = history
			.revisions
			.iter()
			.position(|item| &item.revision == revision)?;
		history.touched_at = now;
		history.revisions.get(index).cloned()
	}

	/// Returns the retained record whose exact bytes equal `bytes`.
	pub fn by_content(&mut self, path: &str, bytes: &[u8]) -> Option<Arc<Snapshot>> {
		let now = self.tick();
		let history = self.histories.get_mut(path)?;
		let index = history
			.revisions
			.iter()
			.position(|item| item.bytes.as_ref() == bytes)?;
		history.touched_at = now;
		history.revisions.get(index).cloned()
	}

	/// Enumerates every independently retained candidate for a path and tag.
	pub fn tag_candidates(&mut self, path: &str, tag: &str) -> Vec<Arc<Snapshot>> {
		let now = self.tick();
		let Some(history) = self.histories.get_mut(path) else {
			return Vec::new();
		};
		history.touched_at = now;
		history
			.revisions
			.iter()
			.filter(|item| item.tag == tag)
			.cloned()
			.collect()
	}

	/// Enumerates every retained candidate for a tag across all paths.
	#[must_use]
	pub fn find_by_tag(&self, tag: &str) -> Vec<Arc<Snapshot>> {
		self
			.histories
			.values()
			.flat_map(|history| history.revisions.iter())
			.filter(|item| item.tag == tag)
			.cloned()
			.collect()
	}

	/// Resolves an exact token, or otherwise the sole retained candidate for a
	/// tag.
	pub fn resolve(
		&mut self,
		path: &str,
		tag: &str,
		revision: Option<&RevisionToken>,
	) -> Result<Arc<Snapshot>, SnapshotLookupError> {
		if let Some(revision) = revision {
			return self
				.by_revision(path, revision)
				.filter(|snapshot| snapshot.tag == tag)
				.ok_or_else(|| SnapshotLookupError::Missing { path: path.into(), tag: tag.into() });
		}
		let mut candidates = self.tag_candidates(path, tag);
		match candidates.len() {
			0 => Err(SnapshotLookupError::Missing { path: path.into(), tag: tag.into() }),
			1 => Ok(candidates.pop().expect("length checked")),
			count => Err(SnapshotLookupError::Ambiguous {
				path:       path.into(),
				tag:        tag.into(),
				candidates: count,
			}),
		}
	}

	/// Adds displayed-line provenance to exactly one retained revision.
	pub fn record_seen_lines<I>(&mut self, path: &str, revision: &RevisionToken, lines: I) -> bool
	where
		I: IntoIterator<Item = usize>,
	{
		let now = self.tick();
		let Some(history) = self.histories.get_mut(path) else {
			return false;
		};
		let Some(index) = history
			.revisions
			.iter()
			.position(|item| &item.revision == revision)
		else {
			return false;
		};
		let existing = &history.revisions[index];
		let mut merged = existing.seen_lines.as_ref().clone();
		merged.extend(lines.into_iter().filter(|line| *line != 0));
		let refreshed =
			Arc::new(Snapshot { seen_lines: Arc::new(merged), ..existing.as_ref().clone() });
		history.revisions[index] = refreshed;
		history.touched_at = now;
		true
	}

	/// Drops all retained revisions for one canonical path.
	pub fn invalidate(&mut self, path: &str) {
		if let Some(history) = self.histories.remove(path) {
			self.remove_history_bytes(&history);
		}
	}

	/// Moves retained history and provenance to another canonical path.
	pub fn relocate(&mut self, from: &str, to: impl Into<Str>) -> Result<(), SnapshotStoreError> {
		let to = to.into();
		if from == to.as_str() {
			return Ok(());
		}
		let Some(source) = self.histories.remove(from) else {
			return Ok(());
		};
		let now = self.tick();
		let conflict = self.histories.get(&to).is_some_and(|destination| {
			source.revisions.iter().any(|source_item| {
				destination
					.revisions
					.iter()
					.find(|item| item.revision == source_item.revision)
					.is_some_and(|destination_item| destination_item.bytes != source_item.bytes)
			})
		});
		if conflict {
			self.histories.insert(from.into(), source);
			return Err(SnapshotStoreError::RevisionConflict { path: to });
		}

		let mut merged = VecDeque::new();
		let destination = self.histories.remove(&to);
		for item in source.revisions.into_iter().chain(
			destination
				.into_iter()
				.flat_map(|history| history.revisions),
		) {
			if merged
				.iter()
				.any(|retained: &Arc<Snapshot>| retained.revision == item.revision)
			{
				self.total_bytes = self.total_bytes.saturating_sub(item.bytes.len());
				continue;
			}
			let relocated = Arc::new(Snapshot { path: to.clone(), ..item.as_ref().clone() });
			merged.push_back(relocated);
		}
		while merged.len() > self.options.max_revisions_per_path {
			if let Some(evicted) = merged.pop_back() {
				self.total_bytes = self.total_bytes.saturating_sub(evicted.bytes.len());
			}
		}
		self
			.histories
			.insert(to, PathHistory { revisions: merged, touched_at: now });
		self.enforce_bounds();
		Ok(())
	}

	/// Drops every retained snapshot record.
	pub fn clear(&mut self) {
		self.histories.clear();
		self.total_bytes = 0;
	}

	/// Returns the number of retained canonical paths.
	#[must_use]
	pub fn path_count(&self) -> usize {
		self.histories.len()
	}

	/// Returns the summed byte lengths used for retention accounting.
	#[must_use]
	pub const fn retained_bytes(&self) -> usize {
		self.total_bytes
	}

	const fn tick(&mut self) -> u64 {
		self.clock = self.clock.wrapping_add(1);
		self.clock
	}

	fn remove_history_bytes(&mut self, history: &PathHistory) {
		let removed = history
			.revisions
			.iter()
			.map(|snapshot| snapshot.bytes.len())
			.sum::<usize>();
		self.total_bytes = self.total_bytes.saturating_sub(removed);
	}

	fn enforce_bounds(&mut self) {
		while self.histories.len() > self.options.max_paths
			|| self.total_bytes > self.options.max_total_bytes
		{
			let Some(cold_path) = self
				.histories
				.iter()
				.min_by_key(|(_, history)| history.touched_at)
				.map(|(path, _)| path.clone())
			else {
				break;
			};
			if let Some(history) = self.histories.remove(&cold_path) {
				self.remove_history_bytes(&history);
			}
		}
	}
}

/// Computes the four-hex compatibility tag from exact file bytes.
///
/// A UTF-8 BOM is excluded, and spaces, tabs, and carriage returns immediately
/// before LF or EOF are ignored, matching hashline's textual tag normalization.
#[must_use]
pub fn compute_snapshot_tag(exact: &[u8]) -> Str {
	let exact = exact.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(exact);
	fmts!("{:04X}", normalized_file_xxh32(exact) & 0xffff)
}

#[cfg(test)]
mod tests {
	use super::*;

	const PATH: &str = "/tmp/hashline-snapshots.rs";
	const COLLIDE_A: &[u8] = b"line one 263\nline two 4471\n";
	const COLLIDE_B: &[u8] = b"line one 410\nline two 6970\n";

	fn token(value: &str) -> RevisionToken {
		RevisionToken::new(value)
	}

	#[test]
	fn colliding_tags_remain_independent_and_require_identity() {
		assert_eq!(compute_snapshot_tag(COLLIDE_A), "1D84");
		assert_eq!(compute_snapshot_tag(COLLIDE_B), "1D84");
		let mut store = SnapshotStore::default();
		store
			.record(PATH, token("a"), Bytes::from_static(COLLIDE_A), [1])
			.unwrap();
		store
			.record(PATH, token("b"), Bytes::from_static(COLLIDE_B), [2])
			.unwrap();

		assert!(matches!(
			store.resolve(PATH, "1D84", None),
			Err(SnapshotLookupError::Ambiguous { candidates: 2, .. })
		));
		let first = store.resolve(PATH, "1D84", Some(&token("a"))).unwrap();
		let second = store.resolve(PATH, "1D84", Some(&token("b"))).unwrap();
		assert_eq!(first.bytes(), &Bytes::from_static(COLLIDE_A));
		assert_eq!(second.bytes(), &Bytes::from_static(COLLIDE_B));
		assert_eq!(first.seen_lines(), &BTreeSet::from([1]));
		assert_eq!(second.seen_lines(), &BTreeSet::from([2]));
	}

	#[test]
	fn seen_lines_union_only_within_an_exact_revision() {
		let mut store = SnapshotStore::default();
		store
			.record(PATH, token("one"), Bytes::from_static(b"same\n"), [1])
			.unwrap();
		store
			.record(PATH, token("two"), Bytes::from_static(b"same\n"), [2])
			.unwrap();
		assert!(store.record_seen_lines(PATH, &token("one"), [3]));
		assert_eq!(
			store.by_revision(PATH, &token("one")).unwrap().seen_lines(),
			&BTreeSet::from([1, 3])
		);
		assert_eq!(
			store.by_revision(PATH, &token("two")).unwrap().seen_lines(),
			&BTreeSet::from([2])
		);
	}

	#[test]
	fn lru_bounds_revisions_paths_and_total_bytes() {
		let mut store = SnapshotStore::new(SnapshotStoreOptions {
			max_paths:              2,
			max_revisions_per_path: 2,
			max_total_bytes:        6,
		})
		.unwrap();
		store
			.record("a", token("a1"), Bytes::from_static(b"aa"), [])
			.unwrap();
		store
			.record("a", token("a2"), Bytes::from_static(b"bb"), [])
			.unwrap();
		store
			.record("a", token("a3"), Bytes::from_static(b"cc"), [])
			.unwrap();
		assert!(store.by_revision("a", &token("a1")).is_none());
		store
			.record("b", token("b1"), Bytes::from_static(b"dd"), [])
			.unwrap();
		store
			.record("c", token("c1"), Bytes::from_static(b"ee"), [])
			.unwrap();
		assert!(store.path_count() <= 2);
		assert!(store.retained_bytes() <= 6);
	}

	#[test]
	fn relocate_preserves_exact_revision_provenance() {
		let mut store = SnapshotStore::default();
		store
			.record(PATH, token("r"), Bytes::from_static(b"A\r\n"), [1])
			.unwrap();
		store.relocate(PATH, "dest").unwrap();
		assert!(store.by_revision(PATH, &token("r")).is_none());
		let moved = store.by_revision("dest", &token("r")).unwrap();
		assert_eq!(moved.bytes(), &Bytes::from_static(b"A\r\n"));
		assert_eq!(moved.seen_lines(), &BTreeSet::from([1]));
	}

	#[test]
	fn tags_ignore_bom_and_preserve_exact_record_bytes() {
		let exact = Bytes::from_static(b"\xef\xbb\xbfline  \r\n");
		assert_eq!(compute_snapshot_tag(&exact), compute_snapshot_tag(b"line\n"));
		let mut store = SnapshotStore::default();
		store
			.record(PATH, token("bom"), exact.clone(), [1])
			.unwrap();
		assert_eq!(store.by_revision(PATH, &token("bom")).unwrap().bytes(), &exact);
	}
}
