//! Incremental, deterministic ordinary-ZIP writing.

use std::{collections::HashSet, io::Write};

use flate2::{Compress, Compression, FlushCompress, Status};
use omp_core::{Str, StrMut};
use zerocopy::{
	IntoBytes,
	byteorder::little_endian::{U16, U32},
};

use super::spec::{
	CENTRAL_HEADER_LEN, CENTRAL_HEADER_SIGNATURE, CentralDirectoryHeader, EOCD_LEN, EOCD_SIGNATURE,
	EndOfCentralDirectory, LOCAL_HEADER_LEN, LOCAL_HEADER_SIGNATURE, LocalFileHeader, U16_SENTINEL,
	U32_SENTINEL, UTF8_FLAG,
};
use crate::{
	Error, Limits, Result,
	path::{is_directory_name, normalize_bounded},
};

const VERSION: u16 = 20;
const STORED: u16 = 0;
const DEFLATE: u16 = 8;
const DOS_TIME: u16 = 0;
const DOS_DATE: u16 = 0x0021;
const DIRECTORY_ATTRIBUTE: u32 = 0x10;

#[derive(Debug)]
struct CentralEntry {
	name:                Str,
	crc32:               u32,
	compressed_size:     u32,
	uncompressed_size:   u32,
	method:              u16,
	local_header_offset: u32,
	directory:           bool,
}

/// A non-seeking ZIP writer.
///
/// Local records are emitted when entries are added. Only central-directory
/// metadata and normalized paths are retained until [`finish`](Self::finish).
pub struct Writer<W: Write> {
	inner:   W,
	entries: Vec<CentralEntry>,
	paths:   HashSet<Str>,
	offset:  u64,
}

impl<W: Write> Writer<W> {
	/// Creates an empty ZIP archive that writes to `inner`.
	pub fn new(inner: W) -> Self {
		Self { inner, entries: Vec::new(), paths: HashSet::new(), offset: 0 }
	}

	/// Adds a file at `path` with the supplied uncompressed bytes.
	pub fn add_file(&mut self, path: &str, data: &[u8]) -> Result<()> {
		if is_directory_name(path) {
			return Err(Error::UnsafePath(path.into()));
		}
		let path = normalize_bounded(path, Limits::DEFAULT)?;
		self.add_entry(path, false, data)
	}

	/// Adds an explicit directory entry at `path`.
	pub fn add_directory(&mut self, path: &str) -> Result<()> {
		let path = normalize_bounded(path, Limits::DEFAULT)?;
		self.add_entry(path, true, &[])
	}

	/// Writes the central directory and returns the wrapped writer.
	pub fn finish(mut self) -> Result<W> {
		let entry_count = self.entries.len();
		if entry_count >= usize::from(U16_SENTINEL) || self.offset >= u64::from(U32_SENTINEL) {
			return Err(Error::Zip64Required);
		}

		let central_size = self.entries.iter().try_fold(0_u64, |size, entry| {
			size
				.checked_add(CENTRAL_HEADER_LEN as u64 + entry.name.len() as u64)
				.ok_or(Error::Zip64Required)
		})?;
		if central_size >= u64::from(U32_SENTINEL) {
			return Err(Error::Zip64Required);
		}
		let archive_end = self
			.offset
			.checked_add(central_size)
			.and_then(|offset| offset.checked_add(EOCD_LEN as u64))
			.ok_or(Error::Zip64Required)?;
		if archive_end >= u64::from(U32_SENTINEL) {
			return Err(Error::Zip64Required);
		}

		let central_offset = self.offset as u32;
		for entry in &self.entries {
			write_central_header(&mut self.inner, entry)?;
		}

		let entry_count = entry_count as u16;
		let eocd = EndOfCentralDirectory {
			signature:        U32::new(EOCD_SIGNATURE),
			disk:             U16::new(0),
			directory_disk:   U16::new(0),
			entries_on_disk:  U16::new(entry_count),
			entries:          U16::new(entry_count),
			directory_size:   U32::new(central_size as u32),
			directory_offset: U32::new(central_offset),
			comment_len:      U16::new(0),
		};
		self.inner.write_all(eocd.as_bytes())?;
		Ok(self.inner)
	}

	fn add_entry(&mut self, path: Str, directory: bool, data: &[u8]) -> Result<()> {
		if self.paths.contains(path.as_str()) {
			return Err(Error::DuplicatePath(path));
		}
		if self.entries.len() + 1 >= usize::from(U16_SENTINEL) || data.len() >= U32_SENTINEL as usize
		{
			return Err(Error::Zip64Required);
		}

		let name = if directory {
			let mut name = StrMut::with_capacity(path.len() + 1);
			name.push_str(path.as_str());
			name.push('/');
			name.freeze()
		} else {
			path.clone()
		};
		if name.len() > usize::from(U16_SENTINEL) {
			return Err(Error::Zip64Required);
		}

		let compressed = if directory {
			None
		} else {
			compress_if_smaller(data)?
		};
		let (method, payload) = compressed
			.as_deref()
			.map_or((STORED, data), |bytes| (DEFLATE, bytes));
		let local_end = self
			.offset
			.checked_add(LOCAL_HEADER_LEN as u64)
			.and_then(|offset| offset.checked_add(name.len() as u64))
			.and_then(|offset| offset.checked_add(payload.len() as u64))
			.ok_or(Error::Zip64Required)?;
		if self.offset >= u64::from(U32_SENTINEL) || local_end >= u64::from(U32_SENTINEL) {
			return Err(Error::Zip64Required);
		}

		let entry = CentralEntry {
			name,
			crc32: crc32fast::hash(data),
			compressed_size: payload.len() as u32,
			uncompressed_size: data.len() as u32,
			method,
			local_header_offset: self.offset as u32,
			directory,
		};
		write_local_record(&mut self.inner, &entry, payload)?;
		self.offset = local_end;
		self.paths.insert(path);
		self.entries.push(entry);
		Ok(())
	}
}

/// Builds an in-memory ZIP archive from `(path, bytes)` files in iterator
/// order.
pub fn encode<I, P, D>(entries: I) -> Result<Vec<u8>>
where
	I: IntoIterator<Item = (P, D)>,
	P: AsRef<str>,
	D: AsRef<[u8]>,
{
	let mut writer = Writer::new(Vec::new());
	for (path, data) in entries {
		writer.add_file(path.as_ref(), data.as_ref())?;
	}
	writer.finish()
}

fn compress_if_smaller(data: &[u8]) -> Result<Option<Vec<u8>>> {
	if data.is_empty() {
		return Ok(None);
	}

	let mut compressor = Compress::new(Compression::default(), false);
	let mut compressed = Vec::with_capacity(data.len().min(64 * 1024));
	let mut input_offset = 0;
	let mut chunk = [0_u8; 8192];
	loop {
		let input_before = compressor.total_in();
		let output_before = compressor.total_out();
		let status = compressor
			.compress(&data[input_offset..], &mut chunk, FlushCompress::Finish)
			.map_err(Error::Compression)?;
		let consumed = (compressor.total_in() - input_before) as usize;
		let produced = (compressor.total_out() - output_before) as usize;
		if produced >= data.len() - compressed.len() {
			return Ok(None);
		}
		compressed.extend_from_slice(&chunk[..produced]);
		input_offset += consumed;
		if status == Status::StreamEnd {
			return Ok(Some(compressed));
		}
		if consumed == 0 && produced == 0 {
			return Ok(None);
		}
	}
}

fn write_local_record<W: Write>(writer: &mut W, entry: &CentralEntry, data: &[u8]) -> Result<()> {
	let header = LocalFileHeader {
		signature:         U32::new(LOCAL_HEADER_SIGNATURE),
		version_needed:    U16::new(VERSION),
		flags:             U16::new(UTF8_FLAG),
		method:            U16::new(entry.method),
		modified_time:     U16::new(DOS_TIME),
		modified_date:     U16::new(DOS_DATE),
		crc32:             U32::new(entry.crc32),
		compressed_size:   U32::new(entry.compressed_size),
		uncompressed_size: U32::new(entry.uncompressed_size),
		name_len:          U16::new(entry.name.len() as u16),
		extra_len:         U16::new(0),
	};
	writer.write_all(header.as_bytes())?;
	writer.write_all(entry.name.as_bytes())?;
	writer.write_all(data)?;
	Ok(())
}

fn write_central_header<W: Write>(writer: &mut W, entry: &CentralEntry) -> Result<()> {
	let header = CentralDirectoryHeader {
		signature:           U32::new(CENTRAL_HEADER_SIGNATURE),
		version_made_by:     U16::new(VERSION),
		version_needed:      U16::new(VERSION),
		flags:               U16::new(UTF8_FLAG),
		method:              U16::new(entry.method),
		modified_time:       U16::new(DOS_TIME),
		modified_date:       U16::new(DOS_DATE),
		crc32:               U32::new(entry.crc32),
		compressed_size:     U32::new(entry.compressed_size),
		uncompressed_size:   U32::new(entry.uncompressed_size),
		name_len:            U16::new(entry.name.len() as u16),
		extra_len:           U16::new(0),
		comment_len:         U16::new(0),
		disk_start:          U16::new(0),
		internal_attributes: U16::new(0),
		external_attributes: U32::new(if entry.directory {
			DIRECTORY_ATTRIBUTE
		} else {
			0
		}),
		local_header_offset: U32::new(entry.local_header_offset),
	};
	writer.write_all(header.as_bytes())?;
	writer.write_all(entry.name.as_bytes())?;
	Ok(())
}
