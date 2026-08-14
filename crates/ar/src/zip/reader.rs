//! Lazy ZIP central-directory indexing and bounded member decoding.

use std::{
	cmp,
	io::{self, Read, Seek, SeekFrom, Write},
};

use crc32fast::Hasher;
use flate2::{Decompress, FlushDecompress, Status};
use omp_core::Str;
use zerocopy::{
	FromBytes,
	byteorder::little_endian::{U32, U64},
};

use super::spec::{
	CENTRAL_HEADER_LEN, CENTRAL_HEADER_SIGNATURE, CentralDirectoryHeader, ENCRYPTED_FLAG, EOCD_LEN,
	EOCD_SIGNATURE, EndOfCentralDirectory, ExtraFieldHeader, LOCAL_HEADER_LEN,
	LOCAL_HEADER_SIGNATURE, LocalFileHeader, MAX_COMMENT_LEN, U16_SENTINEL, U32_SENTINEL, UTF8_FLAG,
	ZIP64_EOCD_LEN, ZIP64_EOCD_SIGNATURE, ZIP64_LOCATOR_LEN, ZIP64_LOCATOR_SIGNATURE,
	Zip64EndOfCentralDirectory, Zip64EndOfCentralDirectoryLocator, decode_name,
};
use crate::{
	Entry, Error, Limits, Result,
	entry::{CompressionMethod, Storage},
	path::{is_directory_name, normalize},
};

const IO_CHUNK_SIZE: usize = 16 * 1024;

#[derive(Debug, Clone, Copy)]
struct DirectoryInfo {
	entries:        u64,
	offset:         u64,
	size:           u64,
	archive_offset: u64,
}

#[derive(Debug, Clone, Copy)]
struct Zip64Values {
	compressed_size:     u64,
	uncompressed_size:   u64,
	local_header_offset: u64,
	disk_start:          u32,
}

#[derive(Debug, Clone, Copy)]
struct Zip64Placeholders {
	compressed_size:     bool,
	uncompressed_size:   bool,
	local_header_offset: bool,
	disk_start:          bool,
}

pub fn read_entries<R: Read + Seek>(
	source: &mut R,
	file_size: u64,
	limits: Limits,
) -> Result<Vec<Entry>> {
	if file_size < EOCD_LEN as u64 {
		return Err(Error::InvalidArchive("missing end of central directory"));
	}
	let tail_len = cmp::min(file_size, (EOCD_LEN + MAX_COMMENT_LEN) as u64);
	let tail_start = file_size - tail_len;
	let tail = read_vec_at(source, tail_start, tail_len)?;
	let last_offset = tail
		.len()
		.checked_sub(EOCD_LEN)
		.ok_or(Error::InvalidArchive("missing end of central directory"))?;
	let mut best_error = None;

	for eocd_index in (0..=last_offset).rev() {
		let Ok((eocd, comment)) = EndOfCentralDirectory::ref_from_prefix(&tail[eocd_index..]) else {
			continue;
		};
		if !valid_eocd(eocd, comment) {
			continue;
		}
		let eocd_offset = tail_start + eocd_index as u64;
		let result = directory_info_for_eocd(source, eocd_offset, eocd, limits).and_then(|info| {
			if info.entries > limits.entries {
				return Err(Error::TooManyEntries { actual: info.entries, limit: limits.entries });
			}
			if info.size > limits.index_size {
				return Err(Error::IndexTooLarge { actual: info.size, limit: limits.index_size });
			}
			let end = info
				.offset
				.checked_add(info.size)
				.ok_or(Error::InvalidArchive("central-directory range overflows"))?;
			if end > file_size {
				return Err(Error::InvalidArchive("central directory exceeds archive size"));
			}
			let directory = read_vec_at(source, info.offset, info.size)?;
			parse_directory(&directory, info.entries, info.archive_offset, limits)
		});
		match result {
			Ok(entries) => return Ok(entries),
			Err(error) if best_error.is_none() => best_error = Some(error),
			Err(_) => {},
		}
	}
	Err(best_error.unwrap_or(Error::InvalidArchive("missing end of central directory")))
}

pub fn read_entry_to<R: Read + Seek, W: Write>(
	source: &mut R,
	entry: &Entry,
	limits: Limits,
	output: &mut W,
) -> Result<u64> {
	let (compressed_size, crc32, method, flags, local_header_offset) = match &entry.storage {
		Storage::Zip { compressed_size, crc32, method, flags, local_header_offset } => {
			(*compressed_size, *crc32, *method, *flags, *local_header_offset)
		},
		_ => return Err(Error::InvalidArchive("ZIP reader received a non-ZIP entry")),
	};

	let actual_size = cmp::max(entry.size, compressed_size);
	if actual_size > limits.member_size {
		return Err(Error::MemberTooLarge {
			path:   entry.path.clone(),
			actual: actual_size,
			limit:  limits.member_size,
		});
	}
	if flags & ENCRYPTED_FLAG != 0 {
		return Err(Error::Encrypted(entry.path.clone()));
	}
	if let CompressionMethod::Unsupported(method) = method {
		return Err(Error::UnsupportedCompression { path: entry.path.clone(), method });
	}

	let header_bytes = read_array_at::<_, LOCAL_HEADER_LEN>(source, local_header_offset)?;
	let header = LocalFileHeader::ref_from_bytes(&header_bytes)
		.expect("fixed-size buffer matches local-file header");
	if header.signature.get() != LOCAL_HEADER_SIGNATURE {
		return Err(Error::InvalidArchive("malformed local file header"));
	}
	if header.flags.get() & ENCRYPTED_FLAG != 0 {
		return Err(Error::Encrypted(entry.path.clone()));
	}
	if header.method.get() != method.code() {
		return Err(Error::InvalidArchive("local and central compression methods disagree"));
	}

	let data_offset = local_header_offset
		.checked_add(LOCAL_HEADER_LEN as u64)
		.and_then(|offset| offset.checked_add(u64::from(header.name_len.get())))
		.and_then(|offset| offset.checked_add(u64::from(header.extra_len.get())))
		.ok_or(Error::InvalidArchive("member data offset overflows"))?;
	source.seek(SeekFrom::Start(data_offset))?;

	let mut crc = Hasher::new();
	let actual = match method {
		CompressionMethod::Stored => {
			if compressed_size != entry.size {
				return Err(Error::SizeMismatch {
					path:     entry.path.clone(),
					expected: entry.size,
					actual:   compressed_size,
				});
			}
			copy_stored(source, compressed_size, output, &mut crc)?
		},
		CompressionMethod::Deflate => {
			inflate(source, compressed_size, entry.size, &entry.path, output, &mut crc)?
		},
		CompressionMethod::Unsupported(_) => unreachable!("checked above"),
	};

	if actual != entry.size {
		return Err(Error::SizeMismatch { path: entry.path.clone(), expected: entry.size, actual });
	}
	let actual_crc = crc.finalize();
	if actual_crc != crc32 {
		return Err(Error::ChecksumMismatch {
			path:     entry.path.clone(),
			expected: crc32,
			actual:   actual_crc,
		});
	}
	Ok(actual)
}

fn directory_info_for_eocd<R: Read + Seek>(
	source: &mut R,
	eocd_offset: u64,
	eocd: &EndOfCentralDirectory,
	limits: Limits,
) -> Result<DirectoryInfo> {
	if eocd.disk.get() != 0 || eocd.directory_disk.get() != 0 {
		return Err(Error::InvalidArchive("multi-disk ZIP archives are not supported"));
	}
	let entries_on_disk = eocd.entries_on_disk.get();
	let total_entries = eocd.entries.get();
	if entries_on_disk != U16_SENTINEL
		&& total_entries != U16_SENTINEL
		&& entries_on_disk != total_entries
	{
		return Err(Error::InvalidArchive("multi-disk ZIP archives are not supported"));
	}

	let legacy = DirectoryInfo {
		entries:        u64::from(total_entries),
		size:           u64::from(eocd.directory_size.get()),
		offset:         u64::from(eocd.directory_offset.get()),
		archive_offset: 0,
	};
	let needs_zip64 = entries_on_disk == U16_SENTINEL
		|| total_entries == U16_SENTINEL
		|| legacy.size == u64::from(U32_SENTINEL)
		|| legacy.offset == u64::from(U32_SENTINEL);
	if let Some(zip64) = read_zip64_info(source, eocd_offset, limits.index_size)? {
		if zip64.entries > limits.entries {
			return Err(Error::TooManyEntries { actual: zip64.entries, limit: limits.entries });
		}
		if zip64.size > limits.index_size {
			return Err(Error::IndexTooLarge { actual: zip64.size, limit: limits.index_size });
		}
		return Ok(zip64);
	}
	if needs_zip64 {
		return Err(Error::InvalidArchive("missing ZIP64 central-directory metadata"));
	}
	if !needs_zip64 {
		if legacy.entries > limits.entries {
			return Err(Error::TooManyEntries { actual: legacy.entries, limit: limits.entries });
		}
		if legacy.size > limits.index_size {
			return Err(Error::IndexTooLarge { actual: legacy.size, limit: limits.index_size });
		}
	}
	let actual_offset = if total_entries == 0 {
		legacy.offset
	} else {
		find_central_header(
			source,
			legacy.offset,
			legacy.size,
			eocd_offset,
			u64::from(total_entries),
			limits.index_size,
		)?
	};
	let archive_offset = actual_offset
		.checked_sub(legacy.offset)
		.ok_or(Error::InvalidArchive("central-directory offset exceeds its actual position"))?;
	Ok(DirectoryInfo { offset: actual_offset, archive_offset, ..legacy })
}

fn find_eocd(tail: &[u8]) -> Result<(usize, &EndOfCentralDirectory)> {
	let Some(last_offset) = tail.len().checked_sub(EOCD_LEN) else {
		return Err(Error::InvalidArchive("missing end of central directory"));
	};
	for offset in (0..=last_offset).rev() {
		let Ok((eocd, comment)) = EndOfCentralDirectory::ref_from_prefix(&tail[offset..]) else {
			continue;
		};
		if valid_eocd(eocd, comment) {
			return Ok((offset, eocd));
		}
	}
	Err(Error::InvalidArchive("missing end of central directory"))
}

fn valid_eocd(eocd: &EndOfCentralDirectory, comment: &[u8]) -> bool {
	eocd.signature.get() == EOCD_SIGNATURE
		&& usize::from(eocd.comment_len.get()) == comment.len()
		&& eocd.disk.get() == 0
		&& eocd.directory_disk.get() == 0
		&& eocd.entries_on_disk.get() == eocd.entries.get()
		&& (eocd.entries.get() != 0 || eocd.directory_size.get() == 0)
}

pub fn has_eocd(bytes: &[u8]) -> bool {
	find_eocd(bytes).is_ok()
}

fn find_central_header<R: Read + Seek>(
	source: &mut R,
	declared_offset: u64,
	directory_size: u64,
	eocd_offset: u64,
	entries: u64,
	search_limit: u64,
) -> Result<u64> {
	if declared_offset >= eocd_offset {
		return Err(Error::InvalidArchive("central-directory offset exceeds end record"));
	}
	if let Some(inferred_offset) = eocd_offset.checked_sub(directory_size)
		&& inferred_offset >= declared_offset
		&& central_layout_at(source, inferred_offset, directory_size, entries, eocd_offset)?
	{
		return Ok(inferred_offset);
	}
	if central_layout_at(source, declared_offset, directory_size, entries, eocd_offset)? {
		return Ok(declared_offset);
	}

	let search_end = cmp::min(eocd_offset, declared_offset.saturating_add(search_limit));
	let mut start = declared_offset;
	let scratch_len =
		usize::try_from(cmp::min(search_end.saturating_sub(start), IO_CHUNK_SIZE as u64))
			.expect("scratch length is bounded by IO_CHUNK_SIZE");
	let mut scratch = vec![0_u8; scratch_len];
	while start < search_end {
		let len = usize::try_from(cmp::min(search_end - start, IO_CHUNK_SIZE as u64))
			.expect("chunk length is bounded by IO_CHUNK_SIZE");
		source.seek(SeekFrom::Start(start))?;
		read_exact_archive(source, &mut scratch[..len])?;
		for (index, _) in scratch[..len]
			.windows(4)
			.enumerate()
			.filter(|(_, bytes)| *bytes == CENTRAL_HEADER_SIGNATURE.to_le_bytes())
		{
			let candidate = start + index as u64;
			if central_layout_at(source, candidate, directory_size, entries, eocd_offset)? {
				return Ok(candidate);
			}
		}
		let end = start + len as u64;
		if end == search_end {
			break;
		}
		start = end - 3;
	}
	if search_end < eocd_offset {
		return Err(Error::IndexTooLarge {
			actual: eocd_offset - declared_offset,
			limit:  search_limit,
		});
	}
	Err(Error::InvalidArchive("missing central-directory header"))
}

fn central_layout_at<R: Read + Seek>(
	source: &mut R,
	offset: u64,
	directory_size: u64,
	entries: u64,
	eocd_offset: u64,
) -> Result<bool> {
	let Some(directory_end) = offset.checked_add(directory_size) else {
		return Ok(false);
	};
	if directory_end > eocd_offset {
		return Ok(false);
	}

	source.seek(SeekFrom::Start(offset))?;
	let mut reader =
		io::BufReader::with_capacity(IO_CHUNK_SIZE, (&mut *source).take(directory_size));
	let mut consumed = 0_u64;
	let mut skip = [0_u8; IO_CHUNK_SIZE];
	for _ in 0..entries {
		let Some(header_end) = consumed.checked_add(CENTRAL_HEADER_LEN as u64) else {
			return Ok(false);
		};
		if header_end > directory_size {
			return Ok(false);
		}
		let mut header_bytes = [0_u8; CENTRAL_HEADER_LEN];
		read_exact_archive(&mut reader, &mut header_bytes)?;
		let header = CentralDirectoryHeader::ref_from_bytes(&header_bytes)
			.expect("fixed-size buffer matches central header");
		if header.signature.get() != CENTRAL_HEADER_SIGNATURE {
			return Ok(false);
		}
		let variable_size = u64::from(header.name_len.get())
			+ u64::from(header.extra_len.get())
			+ u64::from(header.comment_len.get());
		let Some(record_end) = header_end.checked_add(variable_size) else {
			return Ok(false);
		};
		if record_end > directory_size {
			return Ok(false);
		}
		let mut remaining = variable_size;
		while remaining != 0 {
			let len = usize::try_from(cmp::min(remaining, IO_CHUNK_SIZE as u64))
				.expect("skip length is bounded by IO_CHUNK_SIZE");
			read_exact_archive(&mut reader, &mut skip[..len])?;
			remaining -= len as u64;
		}
		consumed = record_end;
	}
	Ok(consumed == directory_size)
}

fn read_zip64_info<R: Read + Seek>(
	source: &mut R,
	eocd_offset: u64,
	search_limit: u64,
) -> Result<Option<DirectoryInfo>> {
	let Some(locator_offset) = eocd_offset.checked_sub(ZIP64_LOCATOR_LEN) else {
		return Ok(None);
	};
	let locator_bytes = read_array_at::<_, { ZIP64_LOCATOR_LEN as usize }>(source, locator_offset)?;
	let locator = Zip64EndOfCentralDirectoryLocator::ref_from_bytes(&locator_bytes)
		.expect("fixed-size buffer matches ZIP64 locator");
	if locator.signature.get() != ZIP64_LOCATOR_SIGNATURE {
		return Ok(None);
	}
	if locator.record_disk.get() != 0 || locator.disks.get() != 1 {
		return Err(Error::InvalidArchive("multi-disk ZIP archives are not supported"));
	}

	let declared_offset = locator.record_offset.get();
	let Some((mut info, actual_record_offset)) =
		find_zip64_record(source, locator_offset, declared_offset, search_limit)?
	else {
		return Err(Error::InvalidArchive("missing ZIP64 end of central directory"));
	};
	let archive_offset = actual_record_offset
		.checked_sub(declared_offset)
		.ok_or(Error::InvalidArchive("ZIP64 record precedes its declared offset"))?;
	info.offset = info
		.offset
		.checked_add(archive_offset)
		.ok_or(Error::InvalidArchive("ZIP64 central-directory offset overflows"))?;
	info.archive_offset = archive_offset;
	Ok(Some(info))
}

fn find_zip64_record<R: Read + Seek>(
	source: &mut R,
	locator_offset: u64,
	declared_offset: u64,
	search_limit: u64,
) -> Result<Option<(DirectoryInfo, u64)>> {
	let lower_bound = locator_offset.saturating_sub(search_limit);
	let mut end = locator_offset;
	let scratch_len =
		usize::try_from(cmp::min(end.saturating_sub(lower_bound), IO_CHUNK_SIZE as u64))
			.expect("scratch length is bounded by IO_CHUNK_SIZE");
	let mut scratch = vec![0_u8; scratch_len];
	while end > lower_bound {
		let start = cmp::max(lower_bound, end.saturating_sub(IO_CHUNK_SIZE as u64));
		let len = usize::try_from(end - start).expect("chunk length is bounded by IO_CHUNK_SIZE");
		source.seek(SeekFrom::Start(start))?;
		read_exact_archive(source, &mut scratch[..len])?;
		for index in (0..len.saturating_sub(3)).rev() {
			if scratch[index..index + 4] == ZIP64_EOCD_SIGNATURE.to_le_bytes() {
				let offset = start + index as u64;
				if let Some(info) = zip64_record_at(source, offset, locator_offset)? {
					return Ok(Some((info, offset)));
				}
			}
		}
		if start == lower_bound {
			break;
		}
		end = start + 3;
	}
	if declared_offset < lower_bound {
		return Err(Error::IndexTooLarge {
			actual: locator_offset - declared_offset,
			limit:  search_limit,
		});
	}
	Ok(None)
}

fn zip64_record_at<R: Read + Seek>(
	source: &mut R,
	offset: u64,
	locator_offset: u64,
) -> Result<Option<DirectoryInfo>> {
	let Some(fixed_end) = offset.checked_add(ZIP64_EOCD_LEN as u64) else {
		return Ok(None);
	};
	if fixed_end > locator_offset {
		return Ok(None);
	}
	let record_bytes = read_array_at::<_, ZIP64_EOCD_LEN>(source, offset)?;
	let record = Zip64EndOfCentralDirectory::ref_from_bytes(&record_bytes)
		.expect("fixed-size buffer matches ZIP64 end record");
	if record.signature.get() != ZIP64_EOCD_SIGNATURE || record.record_size.get() < 44 {
		return Ok(None);
	}
	let Some(record_end) = offset
		.checked_add(12)
		.and_then(|start| start.checked_add(record.record_size.get()))
	else {
		return Ok(None);
	};
	if record_end != locator_offset {
		return Ok(None);
	}
	if record.disk.get() != 0 || record.directory_disk.get() != 0 {
		return Err(Error::InvalidArchive("multi-disk ZIP archives are not supported"));
	}
	let entries = record.entries.get();
	if record.entries_on_disk.get() != entries {
		return Err(Error::InvalidArchive("multi-disk ZIP archives are not supported"));
	}
	Ok(Some(DirectoryInfo {
		entries,
		size: record.directory_size.get(),
		offset: record.directory_offset.get(),
		archive_offset: 0,
	}))
}

fn parse_directory(
	directory: &[u8],
	expected_entries: u64,
	archive_offset: u64,
	limits: Limits,
) -> Result<Vec<Entry>> {
	if expected_entries > (directory.len() / CENTRAL_HEADER_LEN) as u64 {
		return Err(Error::InvalidArchive("truncated central directory"));
	}
	let capacity = usize::try_from(expected_entries)
		.map_err(|_| Error::InvalidArchive("entry count does not fit this platform"))?;
	let mut indexed = Vec::with_capacity(capacity);
	let mut remaining = directory;

	for _ in 0..expected_entries {
		let (header, entry_data) = CentralDirectoryHeader::ref_from_prefix(remaining)
			.map_err(|_| Error::InvalidArchive("truncated central directory"))?;
		if header.signature.get() != CENTRAL_HEADER_SIGNATURE {
			return Err(Error::InvalidArchive("malformed central directory"));
		}

		let flags = header.flags.get();
		let compressed_raw = header.compressed_size.get();
		let uncompressed_raw = header.uncompressed_size.get();
		let disk_start_raw = header.disk_start.get();
		let local_offset_raw = header.local_header_offset.get();
		let (raw_name, entry_data) = entry_data
			.split_at_checked(usize::from(header.name_len.get()))
			.ok_or(Error::InvalidArchive("truncated central-directory entry"))?;
		let (extra, entry_data) = entry_data
			.split_at_checked(usize::from(header.extra_len.get()))
			.ok_or(Error::InvalidArchive("truncated central-directory entry"))?;
		let (_, next) = entry_data
			.split_at_checked(usize::from(header.comment_len.get()))
			.ok_or(Error::InvalidArchive("truncated central-directory entry"))?;

		if raw_name.len() as u64 > limits.path_size {
			return Err(Error::PathTooLong {
				actual: raw_name.len() as u64,
				limit:  limits.path_size,
			});
		}
		let (decoded, modified_unix_seconds) =
			decode_entry_metadata(raw_name, extra, flags & UTF8_FLAG != 0)?;
		if decoded.len() as u64 > limits.path_size {
			return Err(Error::PathTooLong { actual: decoded.len() as u64, limit: limits.path_size });
		}
		if let Some(path) = normalize(decoded.as_str(), false) {
			let depth = path.bytes().filter(|byte| *byte == b'/').count() as u64 + 1;
			if depth > limits.path_depth {
				return Err(Error::PathTooDeep { actual: depth, limit: limits.path_depth });
			}
			let values = read_zip64_values(
				extra,
				Zip64Placeholders {
					compressed_size:     compressed_raw == U32_SENTINEL,
					uncompressed_size:   uncompressed_raw == U32_SENTINEL,
					local_header_offset: local_offset_raw == U32_SENTINEL,
					disk_start:          disk_start_raw == U16_SENTINEL,
				},
				Zip64Values {
					compressed_size:     u64::from(compressed_raw),
					uncompressed_size:   u64::from(uncompressed_raw),
					local_header_offset: u64::from(local_offset_raw),
					disk_start:          u32::from(disk_start_raw),
				},
			)?;
			if values.disk_start != 0 {
				return Err(Error::InvalidArchive("multi-disk ZIP archives are not supported"));
			}
			let directory_entry = raw_name
				.last()
				.is_some_and(|byte| *byte == b'/' || *byte == b'\\')
				|| is_directory_name(decoded.as_str());
			indexed.push(Entry {
				path,
				directory: directory_entry,
				size: if directory_entry {
					0
				} else {
					values.uncompressed_size
				},
				modified_unix_seconds,
				storage: Storage::Zip {
					compressed_size: if directory_entry {
						0
					} else {
						values.compressed_size
					},
					crc32: header.crc32.get(),
					method: CompressionMethod::from_code(header.method.get()),
					flags,
					local_header_offset: values
						.local_header_offset
						.checked_add(archive_offset)
						.ok_or(Error::InvalidArchive("local-file header offset overflows"))?,
				},
			});
		}
		remaining = next;
	}
	Ok(indexed)
}

fn decode_entry_metadata(raw_name: &[u8], extra: &[u8], utf8: bool) -> Result<(Str, Option<u64>)> {
	let mut fields = extra;
	let mut unicode_name = None;
	let mut modified_unix_seconds = None;
	while !fields.is_empty() {
		if fields.len() < 2 {
			break;
		}
		let id = u16::from_le_bytes([fields[0], fields[1]]);
		if fields.len() < 4 {
			if matches!(id, 0x0001 | 0x5455 | 0x7075) {
				return Err(Error::InvalidArchive("truncated ZIP extra-field header"));
			}
			break;
		}
		let data_len = usize::from(u16::from_le_bytes([fields[2], fields[3]]));
		let (data, next) = fields[4..]
			.split_at_checked(data_len)
			.ok_or(Error::InvalidArchive("malformed ZIP extra field"))?;
		if id == 0x7075 {
			if data.len() < 5 {
				return Err(Error::InvalidArchive("Unicode path extra field is too small"));
			}
			let expected_crc = u32::from_le_bytes(
				data[1..5]
					.try_into()
					.expect("Unicode-path CRC slice has fixed length"),
			);
			if data[0] == 1
				&& expected_crc == crc32fast::hash(raw_name)
				&& let Ok(name) = std::str::from_utf8(&data[5..])
			{
				unicode_name = Some(name);
			}
		} else if id == 0x5455 {
			let (&timestamp_flags, timestamps) = data
				.split_first()
				.ok_or(Error::InvalidArchive("extended timestamp extra field is too small"))?;
			if timestamp_flags & 1 != 0 {
				let timestamp = timestamps
					.get(..4)
					.ok_or(Error::InvalidArchive("extended timestamp extra field is too small"))?;
				let signed = i64::from(i32::from_le_bytes(timestamp.try_into().expect("fixed length")));
				modified_unix_seconds = u64::try_from(signed).ok();
			}
		}
		fields = next;
	}
	Ok((
		unicode_name.map_or_else(|| decode_name(raw_name, utf8), |name| name.into()),
		modified_unix_seconds,
	))
}

fn read_zip64_values(
	extra: &[u8],
	placeholders: Zip64Placeholders,
	current: Zip64Values,
) -> Result<Zip64Values> {
	if !placeholders.compressed_size
		&& !placeholders.uncompressed_size
		&& !placeholders.local_header_offset
		&& !placeholders.disk_start
	{
		return Ok(current);
	}

	let mut fields = extra;
	while let Ok((header, remaining)) = ExtraFieldHeader::ref_from_prefix(fields) {
		let (data, next) = remaining
			.split_at_checked(usize::from(header.data_len.get()))
			.ok_or(Error::InvalidArchive("malformed ZIP extra field"))?;
		if header.id.get() == 0x0001 {
			let mut data = data;
			let mut values = current;
			if placeholders.uncompressed_size {
				values.uncompressed_size = take_zip64_u64(&mut data)?;
			}
			if placeholders.compressed_size {
				values.compressed_size = take_zip64_u64(&mut data)?;
			}
			if placeholders.local_header_offset {
				values.local_header_offset = take_zip64_u64(&mut data)?;
			}
			if placeholders.disk_start {
				values.disk_start = take_zip64_u32(&mut data)?;
			}
			return Ok(values);
		}
		fields = next;
	}
	Err(Error::InvalidArchive("missing ZIP64 extra field"))
}

fn take_zip64_u64(bytes: &mut &[u8]) -> Result<u64> {
	let (value, remaining) = U64::read_from_prefix(bytes)
		.map_err(|_| Error::InvalidArchive("malformed ZIP64 extra field"))?;
	*bytes = remaining;
	Ok(value.get())
}

fn take_zip64_u32(bytes: &mut &[u8]) -> Result<u32> {
	let (value, remaining) = U32::read_from_prefix(bytes)
		.map_err(|_| Error::InvalidArchive("malformed ZIP64 extra field"))?;
	*bytes = remaining;
	Ok(value.get())
}

fn copy_stored<R: Read, W: Write>(
	source: &mut R,
	size: u64,
	output: &mut W,
	crc: &mut Hasher,
) -> Result<u64> {
	let mut remaining = size;
	let mut buffer = [0_u8; IO_CHUNK_SIZE];
	while remaining != 0 {
		let wanted =
			usize::try_from(cmp::min(remaining, buffer.len() as u64)).unwrap_or(buffer.len());
		let read = source.read(&mut buffer[..wanted])?;
		if read == 0 {
			return Err(Error::InvalidArchive("truncated ZIP member data"));
		}
		output.write_all(&buffer[..read])?;
		crc.update(&buffer[..read]);
		remaining -= read as u64;
	}
	Ok(size)
}

fn inflate<R: Read, W: Write>(
	source: &mut R,
	compressed_size: u64,
	expected_size: u64,
	path: &Str,
	output: &mut W,
	crc: &mut Hasher,
) -> Result<u64> {
	let mut bounded = source.take(compressed_size);
	let mut decoder = Decompress::new(false);
	let mut input = [0_u8; IO_CHUNK_SIZE];
	let mut input_start = 0_usize;
	let mut input_end = 0_usize;
	let mut decoded = [0_u8; IO_CHUNK_SIZE];
	let mut total = 0_u64;

	loop {
		if input_start == input_end && bounded.limit() != 0 {
			let read = bounded.read(&mut input)?;
			if read == 0 {
				return Err(Error::InvalidArchive("truncated ZIP member data"));
			}
			input_start = 0;
			input_end = read;
		}

		let remaining = expected_size.saturating_sub(total);
		let output_len = usize::try_from(cmp::min(remaining, (decoded.len() - 1) as u64) + 1)
			.unwrap_or(decoded.len());
		let before_in = decoder.total_in();
		let before_out = decoder.total_out();
		let flush = if input_start == input_end && bounded.limit() == 0 {
			FlushDecompress::Finish
		} else {
			FlushDecompress::None
		};
		let status = decoder
			.decompress(&input[input_start..input_end], &mut decoded[..output_len], flush)
			.map_err(|source| Error::Decompression { path: path.clone(), source })?;
		let consumed = usize::try_from(decoder.total_in() - before_in)
			.map_err(|_| Error::InvalidArchive("DEFLATE input count overflows"))?;
		let produced = usize::try_from(decoder.total_out() - before_out)
			.map_err(|_| Error::InvalidArchive("DEFLATE output count overflows"))?;
		input_start += consumed;

		let actual = total
			.checked_add(produced as u64)
			.ok_or(Error::SizeMismatch {
				path:     path.clone(),
				expected: expected_size,
				actual:   u64::MAX,
			})?;
		if actual > expected_size {
			return Err(Error::SizeMismatch { path: path.clone(), expected: expected_size, actual });
		}
		if produced != 0 {
			output.write_all(&decoded[..produced])?;
			crc.update(&decoded[..produced]);
			total = actual;
		}

		if status == Status::StreamEnd {
			if input_start != input_end || bounded.limit() != 0 {
				return Err(Error::InvalidArchive("DEFLATE member has trailing compressed data"));
			}
			return Ok(total);
		}
		if consumed == 0 && produced == 0 {
			return Err(Error::InvalidArchive("truncated or stalled DEFLATE stream"));
		}
	}
}

fn read_vec_at<R: Read + Seek>(source: &mut R, offset: u64, len: u64) -> Result<Vec<u8>> {
	let len = usize::try_from(len)
		.map_err(|_| Error::InvalidArchive("ZIP range does not fit this platform"))?;
	let mut bytes = vec![0_u8; len];
	source.seek(SeekFrom::Start(offset))?;
	read_exact_archive(source, &mut bytes)?;
	Ok(bytes)
}

fn read_array_at<R: Read + Seek, const N: usize>(source: &mut R, offset: u64) -> Result<[u8; N]> {
	let mut bytes = [0_u8; N];
	source.seek(SeekFrom::Start(offset))?;
	read_exact_archive(source, &mut bytes)?;
	Ok(bytes)
}

fn read_exact_archive(reader: &mut impl Read, bytes: &mut [u8]) -> Result<()> {
	match reader.read_exact(bytes) {
		Ok(()) => Ok(()),
		Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
			Err(Error::InvalidArchive("truncated ZIP archive"))
		},
		Err(error) => Err(Error::Io(error)),
	}
}
