//! ZIP archive reading and deterministic ordinary-ZIP writing.

pub(crate) mod reader;
mod spec;
mod writer;
pub(crate) use reader::{has_eocd, read_entries, read_entry_to};
pub(crate) const SNIFF_TAIL_SIZE: usize = spec::EOCD_LEN + spec::MAX_COMMENT_LEN;
pub use writer::{Writer, encode};

pub use crate::entry::CompressionMethod;
