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
	decode_windows_1252(bytes)
}

fn decode_windows_1252(bytes: &[u8]) -> Str {
	const C1: [u16; 32] = [
		0x20ac, 0xfffd, 0x201a, 0x0192, 0x201e, 0x2026, 0x2020, 0x2021, 0x02c6, 0x2030, 0x0160,
		0x2039, 0x0152, 0xfffd, 0x017d, 0xfffd, 0xfffd, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022,
		0x2013, 0x2014, 0x02dc, 0x2122, 0x0161, 0x203a, 0x0153, 0xfffd, 0x017e, 0x0178,
	];

	let mut decoded = StrMut::with_capacity(bytes.len());
	for &byte in bytes {
		let scalar = if (0x80..=0x9f).contains(&byte) {
			u32::from(C1[usize::from(byte - 0x80)])
		} else {
			u32::from(byte)
		};
		decoded.push(char::from_u32(scalar).unwrap_or(char::REPLACEMENT_CHARACTER));
	}
	decoded.freeze()
}
