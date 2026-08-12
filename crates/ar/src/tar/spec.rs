//! Typed TAR wire records.

use std::mem::size_of;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

pub const BLOCK_SIZE: usize = 512;

/// POSIX ustar header block.
#[derive(FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
#[repr(C)]
pub struct UstarHeader {
	pub name:         [u8; 100],
	pub mode:         [u8; 8],
	pub uid:          [u8; 8],
	pub gid:          [u8; 8],
	pub size:         [u8; 12],
	pub mtime:        [u8; 12],
	pub checksum:     [u8; 8],
	pub typeflag:     u8,
	pub link_name:    [u8; 100],
	pub magic:        [u8; 6],
	pub version:      [u8; 2],
	pub owner_name:   [u8; 32],
	pub group_name:   [u8; 32],
	pub device_major: [u8; 8],
	pub device_minor: [u8; 8],
	pub prefix:       [u8; 155],
	pub padding:      [u8; 12],
}

/// One old-GNU sparse map tuple.
#[derive(FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
#[repr(C)]
pub struct GnuSparseEntry {
	pub offset:    [u8; 12],
	pub num_bytes: [u8; 12],
}

/// Old-GNU header block. Its fields after `link_name` intentionally differ
/// from the POSIX prefix layout.
#[derive(FromBytes, Immutable, KnownLayout, Unaligned)]
#[repr(C)]
pub struct OldGnuHeader {
	pub name:         [u8; 100],
	pub mode:         [u8; 8],
	pub uid:          [u8; 8],
	pub gid:          [u8; 8],
	pub size:         [u8; 12],
	pub mtime:        [u8; 12],
	pub checksum:     [u8; 8],
	pub typeflag:     u8,
	pub link_name:    [u8; 100],
	pub magic:        [u8; 8],
	pub owner_name:   [u8; 32],
	pub group_name:   [u8; 32],
	pub device_major: [u8; 8],
	pub device_minor: [u8; 8],
	pub accessed_at:  [u8; 12],
	pub changed_at:   [u8; 12],
	pub offset:       [u8; 12],
	pub long_names:   [u8; 4],
	pub unused:       u8,
	pub sparse:       [GnuSparseEntry; 4],
	pub is_extended:  u8,
	pub real_size:    [u8; 12],
	pub padding:      [u8; 17],
}

/// Old-GNU sparse map continuation block.
#[derive(FromBytes, Immutable, KnownLayout, Unaligned)]
#[repr(C)]
pub struct GnuSparseContinuation {
	pub sparse:      [GnuSparseEntry; 21],
	pub is_extended: u8,
	pub padding:     [u8; 7],
}

const _: [(); BLOCK_SIZE] = [(); size_of::<UstarHeader>()];
const _: [(); BLOCK_SIZE] = [(); size_of::<OldGnuHeader>()];
const _: [(); BLOCK_SIZE] = [(); size_of::<GnuSparseContinuation>()];
