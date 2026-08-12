//! Bounded ZIP, TAR, and TAR.GZ reading with deterministic archive writing.
//!
//! [`Archive`] indexes a seekable source without materializing ordinary ZIP or
//! TAR member payloads. TAR.GZ is decompressed once under [`Limits`] because
//! random member reads require the decoded TAR byte stream. Format-specific
//! writers live in [`zip`] and [`tar`].
//!
//! # Example
//!
//! ```
//! use omp_ar::{Archive, Format, tar};
//!
//! let encoded = tar::encode([("hello.txt", b"hello".as_slice())])?;
//! let mut archive = Archive::from_bytes_with_format(&encoded, Format::Tar)?;
//! assert_eq!(archive.read("hello.txt")?, b"hello");
//! # Ok::<(), omp_ar::Error>(())
//! ```

mod archive;
mod entry;
mod error;
mod path;
pub mod tar;
pub mod zip;

pub use archive::{Archive, Files, Format, Limits, unpack, unpack_with_format};
pub use entry::Entry;
pub use error::{Error, Result};
