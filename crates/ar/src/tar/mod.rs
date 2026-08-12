//! TAR and TAR.GZ reading and deterministic writing.

pub(crate) mod reader;
mod spec;
mod writer;

pub(crate) use reader::{read_entries, read_entry_to, resolve_alias_path};
pub use writer::{Writer, encode, encode_gzip};
