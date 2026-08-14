//! Typed ZIP wire records and filename decoding.

use std::{mem::size_of, string::String};

use omp_core::{Str, StrMut};
use xutf::{TextBuf as _, Utf8};
use zerocopy::{
	FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned,
	byteorder::little_endian::{U16, U32, U64},
};

pub const LOCAL_HEADER_SIGNATURE: u32 = 0x0403_4b50;
pub const CENTRAL_HEADER_SIGNATURE: u32 = 0x0201_4b50;
pub const ZIP64_EOCD_SIGNATURE: u32 = 0x0606_4b50;
pub const ZIP64_LOCATOR_SIGNATURE: u32 = 0x0706_4b50;
pub const EOCD_SIGNATURE: u32 = 0x0605_4b50;
pub const MAX_COMMENT_LEN: usize = u16::MAX as usize;
pub const UTF8_FLAG: u16 = 0x0800;
pub const ENCRYPTED_FLAG: u16 = 0x0001;
pub const U16_SENTINEL: u16 = u16::MAX;
pub const U32_SENTINEL: u32 = u32::MAX;

#[derive(FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
#[repr(C)]
pub struct LocalFileHeader {
	pub signature:         U32,
	pub version_needed:    U16,
	pub flags:             U16,
	pub method:            U16,
	pub modified_time:     U16,
	pub modified_date:     U16,
	pub crc32:             U32,
	pub compressed_size:   U32,
	pub uncompressed_size: U32,
	pub name_len:          U16,
	pub extra_len:         U16,
}

#[derive(FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
#[repr(C)]
pub struct CentralDirectoryHeader {
	pub signature:           U32,
	pub version_made_by:     U16,
	pub version_needed:      U16,
	pub flags:               U16,
	pub method:              U16,
	pub modified_time:       U16,
	pub modified_date:       U16,
	pub crc32:               U32,
	pub compressed_size:     U32,
	pub uncompressed_size:   U32,
	pub name_len:            U16,
	pub extra_len:           U16,
	pub comment_len:         U16,
	pub disk_start:          U16,
	pub internal_attributes: U16,
	pub external_attributes: U32,
	pub local_header_offset: U32,
}

#[derive(FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
#[repr(C)]
pub struct EndOfCentralDirectory {
	pub signature:        U32,
	pub disk:             U16,
	pub directory_disk:   U16,
	pub entries_on_disk:  U16,
	pub entries:          U16,
	pub directory_size:   U32,
	pub directory_offset: U32,
	pub comment_len:      U16,
}

#[derive(FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
#[repr(C)]
pub struct Zip64EndOfCentralDirectoryLocator {
	pub signature:     U32,
	pub record_disk:   U32,
	pub record_offset: U64,
	pub disks:         U32,
}

#[derive(FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
#[repr(C)]
pub struct Zip64EndOfCentralDirectory {
	pub signature:        U32,
	pub record_size:      U64,
	pub version_made_by:  U16,
	pub version_needed:   U16,
	pub disk:             U32,
	pub directory_disk:   U32,
	pub entries_on_disk:  U64,
	pub entries:          U64,
	pub directory_size:   U64,
	pub directory_offset: U64,
}

#[derive(FromBytes, Immutable, KnownLayout, Unaligned)]
#[repr(C)]
pub struct ExtraFieldHeader {
	pub id:       U16,
	pub data_len: U16,
}

pub const LOCAL_HEADER_LEN: usize = size_of::<LocalFileHeader>();
pub const CENTRAL_HEADER_LEN: usize = size_of::<CentralDirectoryHeader>();
pub const EOCD_LEN: usize = size_of::<EndOfCentralDirectory>();
pub const ZIP64_LOCATOR_LEN: u64 = size_of::<Zip64EndOfCentralDirectoryLocator>() as u64;
pub const ZIP64_EOCD_LEN: usize = size_of::<Zip64EndOfCentralDirectory>();

const _: [(); 30] = [(); LOCAL_HEADER_LEN];
const _: [(); 46] = [(); CENTRAL_HEADER_LEN];
const _: [(); 22] = [(); EOCD_LEN];
const _: [(); 20] = [(); ZIP64_LOCATOR_LEN as usize];
const _: [(); 56] = [(); ZIP64_EOCD_LEN];

pub fn decode_name(bytes: &[u8], utf8: bool) -> Str {
	if utf8 {
		let units = xutf::transcode::<Utf8, Utf8>(bytes);
		return String::from_units(units).into();
	}
	decode_cp437(bytes)
}

fn decode_cp437(bytes: &[u8]) -> Str {
	const HIGH: [u16; 128] = [
		0x00c7, 0x00fc, 0x00e9, 0x00e2, 0x00e4, 0x00e0, 0x00e5, 0x00e7, 0x00ea, 0x00eb, 0x00e8,
		0x00ef, 0x00ee, 0x00ec, 0x00c4, 0x00c5, 0x00c9, 0x00e6, 0x00c6, 0x00f4, 0x00f6, 0x00f2,
		0x00fb, 0x00f9, 0x00ff, 0x00d6, 0x00dc, 0x00a2, 0x00a3, 0x00a5, 0x20a7, 0x0192, 0x00e1,
		0x00ed, 0x00f3, 0x00fa, 0x00f1, 0x00d1, 0x00aa, 0x00ba, 0x00bf, 0x2310, 0x00ac, 0x00bd,
		0x00bc, 0x00a1, 0x00ab, 0x00bb, 0x2591, 0x2592, 0x2593, 0x2502, 0x2524, 0x2561, 0x2562,
		0x2556, 0x2555, 0x2563, 0x2551, 0x2557, 0x255d, 0x255c, 0x255b, 0x2510, 0x2514, 0x2534,
		0x252c, 0x251c, 0x2500, 0x253c, 0x255e, 0x255f, 0x255a, 0x2554, 0x2569, 0x2566, 0x2560,
		0x2550, 0x256c, 0x2567, 0x2568, 0x2564, 0x2565, 0x2559, 0x2558, 0x2552, 0x2553, 0x256b,
		0x256a, 0x2518, 0x250c, 0x2588, 0x2584, 0x258c, 0x2590, 0x2580, 0x03b1, 0x00df, 0x0393,
		0x03c0, 0x03a3, 0x03c3, 0x00b5, 0x03c4, 0x03a6, 0x0398, 0x03a9, 0x03b4, 0x221e, 0x03c6,
		0x03b5, 0x2229, 0x2261, 0x00b1, 0x2265, 0x2264, 0x2320, 0x2321, 0x00f7, 0x2248, 0x00b0,
		0x2219, 0x00b7, 0x221a, 0x207f, 0x00b2, 0x25a0, 0x00a0,
	];

	let mut decoded = StrMut::with_capacity(bytes.len());
	for &byte in bytes {
		let scalar = if byte < 0x80 {
			u32::from(byte)
		} else {
			u32::from(HIGH[usize::from(byte - 0x80)])
		};
		decoded.push(char::from_u32(scalar).expect("CP437 table contains Unicode scalars"));
	}
	decoded.freeze()
}
