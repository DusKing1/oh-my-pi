//! ZIP archive reading and deterministic ordinary-ZIP writing.

pub(crate) mod reader;
mod spec;
mod writer;
pub(crate) use reader::{read_entries, read_entry_to};
pub use writer::{Writer, encode};

pub use crate::entry::CompressionMethod;
