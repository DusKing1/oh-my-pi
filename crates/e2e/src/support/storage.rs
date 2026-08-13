use std::path::Path;

use omp_agent::{Journal, JournalError};
use omp_storage::transcript::{self, Log};

/// Reopens the physical transcript, preserving tombstones and exact event indexes.
pub fn reopen_transcript(path: &Path) -> Result<Log, transcript::Error> {
	transcript::load(path)
}

/// Reopens the durable agent journal and rebuilds its terminal-turn index.
pub fn reopen_journal(path: &Path) -> Result<Journal, JournalError> {
	Journal::open(path)
}
