//! Transcript v4's append-only event journal.
//!
//! Line zero is the identity header and every later physical line is an event,
//! so an event index is always its line number minus one. Malformed lines
//! remain as tombstones rather than being dropped; otherwise every later
//! reference would shift. Corrections and navigation are later events, never
//! edits to old bytes. Replay capsules follow the complementary storage rule
//! that every byte exists in exactly one place: neutral data in blocks,
//! provider-only residue in capsules, and large payloads
//! in the content-addressed blob store.

pub mod block;
pub mod capsule;
pub mod codec;
pub mod event;
pub mod msg;
pub mod patch;
mod raweq;
pub mod reader;
pub mod replay;
pub mod types;
pub mod writer;

pub use block::{Block, BlockKind, Replay};
pub use codec::{Error, Header, read_header, read_line, write_header, write_line};
pub use event::{Event, ItemRecord, Kind, TurnReceipt};
pub use msg::{Content, Msg, UserBlock};
pub use patch::Patch;
pub use reader::{Entry, Log, load};
pub use types::*;
pub use writer::Writer;
