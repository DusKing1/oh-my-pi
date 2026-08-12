//! Streaming TAR indexing and lazy member reads.

use std::{
	collections::{HashMap, HashSet, VecDeque},
	io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write},
	mem,
	string::String,
};

use omp_core::{Str, StrMut};
use smallvec::SmallVec;
use xutf::{TextBuf as _, Utf8};
use zerocopy::FromBytes;

use super::spec::{BLOCK_SIZE, GnuSparseContinuation, OldGnuHeader, UstarHeader};
use crate::{
	Entry, Error, Limits, Result,
	entry::Storage,
	path::{is_directory_name, normalize, parent, validate},
};

const PAX_KEY_BUFFER_SIZE: usize = 32;
const IO_BUFFER_SIZE: usize = 16 * 1024;
const GNU_SPARSE_PREFIX: &[u8] = b"GNU.sparse.";
const OLD_GNU_RENAME_PREFIX: &[u8] = b"Rename ";
const OLD_GNU_RENAME_SEPARATOR: &[u8] = b" to ";

#[derive(Clone, Copy)]
enum PaxNumber {
	Delete,
	Value(u64),
}

#[derive(Default)]
struct PaxState {
	path:             Option<Str>,
	link_path:        Option<Str>,
	size:             Option<PaxNumber>,
	sparse:           Option<bool>,
	sparse_name:      Option<Str>,
	sparse_real_size: Option<PaxNumber>,
}
fn apply_global_pax(global: &mut PaxState, update: PaxState) {
	apply_global_text(&mut global.path, update.path);
	apply_global_text(&mut global.link_path, update.link_path);
	apply_global_number(&mut global.size, update.size);
	if let Some(sparse) = update.sparse {
		global.sparse = sparse.then_some(true);
	}
	apply_global_text(&mut global.sparse_name, update.sparse_name);
	apply_global_number(&mut global.sparse_real_size, update.sparse_real_size);
}

fn apply_global_text(current: &mut Option<Str>, update: Option<Str>) {
	if let Some(value) = update {
		*current = (!value.is_empty()).then_some(value);
	}
}

const fn apply_global_number(current: &mut Option<PaxNumber>, update: Option<PaxNumber>) {
	if let Some(value) = update {
		*current = match value {
			PaxNumber::Delete => None,
			PaxNumber::Value(_) => Some(value),
		};
	}
}

const fn pax_number(value: Option<PaxNumber>, error: &'static str) -> Result<Option<u64>> {
	match value {
		None => Ok(None),
		Some(PaxNumber::Value(value)) => Ok(Some(value)),
		Some(PaxNumber::Delete) => Err(Error::InvalidArchive(error)),
	}
}

#[derive(Clone, Copy)]
enum LinkKind {
	Hard,
	Symbolic,
}

struct PendingLink {
	kind:   LinkKind,
	target: Str,
}

/// Indexes a seekable TAR stream without materializing ordinary file data.
pub fn read_entries<R: Read + Seek>(
	source: &mut R,
	file_size: u64,
	limits: Limits,
) -> Result<Vec<Entry>> {
	if file_size > limits.archive_size {
		return Err(Error::ArchiveTooLarge { actual: file_size, limit: limits.archive_size });
	}

	let mut entries = Vec::new();
	let mut entries_by_path = HashMap::<Str, usize>::new();
	let mut pending = Vec::<Option<PendingLink>>::new();
	let mut offset = 0_u64;
	let mut long_name = None;
	let mut long_link = None;
	let mut local_pax = None;
	let mut global_pax = PaxState::default();
	let mut saw_terminator = false;

	while offset < file_size {
		if file_size - offset < BLOCK_SIZE as u64 {
			return Err(Error::InvalidArchive("not a valid TAR archive: truncated header"));
		}
		let block = read_block_at(source, offset)?;
		if block.iter().all(|&byte| byte == 0) {
			if long_name.is_some() || long_link.is_some() || local_pax.is_some() {
				return Err(Error::InvalidArchive("orphaned TAR extended header"));
			}
			saw_terminator = true;
			break;
		}
		validate_checksum(&block)?;

		let header =
			UstarHeader::ref_from_bytes(&block).expect("a TAR header is exactly one typed wire block");
		let old_gnu = OldGnuHeader::ref_from_bytes(&block)
			.expect("an old-GNU header is exactly one typed wire block");
		let typeflag = header.typeflag;
		let header_size = parse_number(&header.size)?;
		let mtime = parse_number(&header.mtime)?;
		offset = checked_add(offset, BLOCK_SIZE as u64, "TAR offset overflow")?;

		let metadata_end = if matches!(typeflag, b'L' | b'K' | b'N' | b'x' | b'X' | b'g') {
			let end = padded_end(offset, header_size)?;
			if end > file_size {
				return Err(Error::InvalidArchive("truncated TAR member data"));
			}
			Some(end)
		} else {
			None
		};

		match typeflag {
			b'L' => {
				long_name = Some(read_long_text(source, offset, header_size, limits.path_size)?);
				offset = metadata_end.expect("GNU long name is a metadata record");
				continue;
			},
			b'K' => {
				long_link = Some(read_long_text(source, offset, header_size, limits.path_size)?);
				offset = metadata_end.expect("GNU long link is a metadata record");
				continue;
			},
			// Obsolete GNUTYPE_NAMES records rename members indexed before this header.
			b'N' => {
				apply_old_gnu_names(
					source,
					offset,
					header_size,
					&mut entries,
					&mut entries_by_path,
					&mut pending,
					limits,
				)?;
				offset = metadata_end.expect("old-GNU names are a metadata record");
				continue;
			},
			b'x' | b'X' => {
				local_pax = Some(parse_pax(source, offset, header_size, limits)?);
				offset = metadata_end.expect("PAX header is a metadata record");
				continue;
			},
			b'g' => {
				let update = parse_pax(source, offset, header_size, limits)?;
				apply_global_pax(&mut global_pax, update);
				offset = metadata_end.expect("global PAX header is a metadata record");
				continue;
			},
			_ => {},
		}

		let mut name = decode_header_name(header, &block, limits)?;
		let mut link_name = decode_field(&header.link_name);
		if let Some(value) = long_name.take() {
			name = value;
		}
		if let Some(value) = long_link.take() {
			link_name = value;
		}

		let local_pax = local_pax.take();
		if let Some(value) = local_pax
			.as_ref()
			.and_then(|attributes| attributes.path.as_ref())
			.or(global_pax.path.as_ref())
		{
			name = value.clone();
		}
		if let Some(value) = local_pax
			.as_ref()
			.and_then(|attributes| attributes.link_path.as_ref())
			.or(global_pax.link_path.as_ref())
		{
			link_name = value.clone();
		}

		let mut stored_size = header_size;
		let mut display_size = header_size;
		let pax_size = local_pax
			.as_ref()
			.and_then(|attributes| attributes.size)
			.or(global_pax.size);
		if let Some(value) = pax_number(pax_size, "invalid PAX member size")? {
			stored_size = value;
			display_size = value;
		}
		if let Some(value) = local_pax
			.as_ref()
			.and_then(|attributes| attributes.sparse_name.as_ref())
			.or(global_pax.sparse_name.as_ref())
		{
			name = value.clone();
		}
		let sparse_size = local_pax
			.as_ref()
			.and_then(|attributes| attributes.sparse_real_size)
			.or(global_pax.sparse_real_size);
		if let Some(value) = pax_number(sparse_size, "invalid PAX sparse real size")? {
			display_size = value;
		}
		let mut sparse = typeflag == b'S';
		sparse |= local_pax
			.as_ref()
			.and_then(|attributes| attributes.sparse)
			.or(global_pax.sparse)
			.unwrap_or(false);

		if typeflag == b'S' && old_gnu.is_extended != 0 {
			loop {
				if file_size.saturating_sub(offset) < BLOCK_SIZE as u64 {
					return Err(Error::InvalidArchive("truncated GNU sparse continuation"));
				}
				let continuation_bytes = read_block_at(source, offset)?;
				let continuation = GnuSparseContinuation::ref_from_bytes(&continuation_bytes)
					.expect("a GNU sparse continuation is exactly one typed wire block");
				offset = checked_add(offset, BLOCK_SIZE as u64, "TAR offset overflow")?;
				if continuation.is_extended == 0 {
					break;
				}
			}
		}

		let data_offset = offset;
		let data_end = padded_end(data_offset, stored_size)?;
		if data_end > file_size {
			return Err(Error::InvalidArchive("truncated TAR member data"));
		}
		offset = data_end;

		let raw_directory = typeflag == b'5' || is_directory_name(name.as_str());
		let Some(path) = normalize_member_path(name, limits)? else {
			continue;
		};
		let modified_unix_seconds = (mtime > 0).then_some(mtime);

		let (entry, pending_link) = match typeflag {
			b'5' => (
				Entry {
					path,
					directory: true,
					size: 0,
					modified_unix_seconds,
					storage: Storage::Synthetic,
				},
				None,
			),
			b'1' | b'2' => make_link_entry(path, link_name, typeflag, modified_unix_seconds, limits)?,
			b'0' | 0 | b'7' | b'S' => {
				let actual = stored_size.max(display_size);
				if actual > limits.member_size {
					return Err(Error::MemberTooLarge { path, actual, limit: limits.member_size });
				}
				(
					Entry {
						path,
						directory: raw_directory,
						size: if raw_directory { 0 } else { display_size },
						modified_unix_seconds,
						storage: if raw_directory {
							Storage::Synthetic
						} else {
							Storage::Tar { data_offset, stored_size, sparse }
						},
					},
					None,
				)
			},
			_ if raw_directory => (
				Entry {
					path,
					directory: true,
					size: 0,
					modified_unix_seconds,
					storage: Storage::Synthetic,
				},
				None,
			),
			_ => continue,
		};

		upsert_entry(
			&mut entries,
			&mut entries_by_path,
			&mut pending,
			entry,
			pending_link,
			limits.entries,
		)?;
	}

	if !saw_terminator {
		return Err(Error::InvalidArchive("missing terminating TAR zero block"));
	}
	resolve_pending_links(&mut entries, &entries_by_path, &pending, limits)?;
	entries.retain(|entry| !entry.path.is_empty());
	Ok(entries)
}

/// Streams one previously indexed TAR member to `output`.
pub fn read_entry_to<R: Read + Seek, W: Write>(
	source: &mut R,
	entry: &Entry,
	output: &mut W,
) -> Result<u64> {
	match &entry.storage {
		Storage::Tar { data_offset, stored_size, sparse } => {
			if *sparse {
				return Err(Error::SparseMember(entry.path.clone()));
			}
			if *stored_size < entry.size {
				return Err(Error::InvalidArchive("truncated TAR member data"));
			}
			source.seek(SeekFrom::Start(*data_offset))?;
			let mut data = source.take(entry.size);
			let copied = io::copy(&mut data, output)?;
			if copied != entry.size {
				return Err(Error::InvalidArchive("truncated TAR member data"));
			}
			Ok(copied)
		},
		Storage::TarLink { target_path } => {
			Err(Error::UnreadableLink { path: entry.path.clone(), target: target_path.clone() })
		},
		_ => Err(Error::InvalidArchive("entry is not a TAR member")),
	}
}

/// Rewrites a normalized lookup through directory symlink aliases.
///
/// `entries` must be sorted by path. A chain of exactly `limits.link_depth`
/// aliases is accepted; only the next required rewrite fails.
pub fn resolve_alias_path(entries: &[Entry], path: Str, limits: Limits) -> Result<Str> {
	validate(&path, limits)?;
	let original = path.clone();
	let mut resolved = path;
	let mut rewrites = 0_u64;

	loop {
		let Some((end, target)) = find_directory_alias(entries, resolved.as_str()) else {
			return Ok(resolved);
		};
		if rewrites == limits.link_depth {
			return Err(Error::LinkResolutionDepth { path: original, limit: limits.link_depth });
		}
		rewrites += 1;
		let suffix = resolved.get(end..).unwrap_or("").trim_start_matches('/');
		let replacement = join_alias_target(target.as_str(), suffix, limits)?;
		validate(&replacement, limits)?;
		resolved = replacement;
	}
}

fn read_block_at<R: Read + Seek>(source: &mut R, offset: u64) -> Result<[u8; BLOCK_SIZE]> {
	let mut block = [0_u8; BLOCK_SIZE];
	source.seek(SeekFrom::Start(offset))?;
	source.read_exact(&mut block)?;
	Ok(block)
}

fn validate_checksum(block: &[u8; BLOCK_SIZE]) -> Result<()> {
	let header =
		UstarHeader::ref_from_bytes(block).expect("a TAR header is exactly one typed wire block");
	let stored = parse_number(&header.checksum)?;
	let mut unsigned = 0_u64;
	let mut signed = 0_i64;
	for (index, &byte) in block.iter().enumerate() {
		let value = if (148..156).contains(&index) {
			b' '
		} else {
			byte
		};
		unsigned += u64::from(value);
		signed += i64::from(value as i8);
	}
	if stored == unsigned || signed >= 0 && stored == signed as u64 {
		Ok(())
	} else {
		Err(Error::InvalidArchive("invalid TAR header checksum"))
	}
}

fn parse_number(field: &[u8]) -> Result<u64> {
	if field.first().is_some_and(|byte| byte & 0x80 != 0) {
		if field[0] & 0x40 != 0 {
			return Err(Error::InvalidArchive("negative TAR numeric field"));
		}
		let mut value = u64::from(field[0] & 0x7f);
		for &byte in &field[1..] {
			value = value
				.checked_mul(256)
				.and_then(|value| value.checked_add(u64::from(byte)))
				.ok_or(Error::InvalidArchive("TAR numeric field overflow"))?;
		}
		return Ok(value);
	}

	let mut value = 0_u64;
	let mut saw_digit = false;
	let mut terminated = false;
	for &byte in field {
		match byte {
			b'0'..=b'7' if !terminated => {
				value = value
					.checked_mul(8)
					.and_then(|value| value.checked_add(u64::from(byte - b'0')))
					.ok_or(Error::InvalidArchive("TAR numeric field overflow"))?;
				saw_digit = true;
			},
			0 if !saw_digit => terminated = true,
			0 | b' ' => terminated |= saw_digit,
			_ => return Err(Error::InvalidArchive("invalid TAR numeric field")),
		}
	}
	Ok(value)
}

fn decode_field(field: &[u8]) -> Str {
	let end = field
		.iter()
		.position(|&byte| byte == 0)
		.unwrap_or(field.len());
	decode_text(&field[..end])
}

fn decode_text(bytes: &[u8]) -> Str {
	let units = xutf::transcode::<Utf8, Utf8>(bytes);
	String::from_units(units).into()
}

fn decode_header_name(
	header: &UstarHeader,
	block: &[u8; BLOCK_SIZE],
	limits: Limits,
) -> Result<Str> {
	let name_end = header
		.name
		.iter()
		.position(|&byte| byte == 0)
		.unwrap_or(header.name.len());
	let old_gnu = header.typeflag == b'S' || &block[257..265] == b"ustar  \0";
	let prefix_end = if old_gnu {
		0
	} else {
		header
			.prefix
			.iter()
			.position(|&byte| byte == 0)
			.unwrap_or(header.prefix.len())
	};
	let separator = usize::from(prefix_end > 0 && name_end > 0);
	let length = prefix_end
		.checked_add(separator)
		.and_then(|length| length.checked_add(name_end))
		.ok_or(Error::InvalidArchive("TAR path length overflow"))?;
	check_path_size(length as u64, limits.path_size)?;

	let mut bytes = [0_u8; 256];
	let mut cursor = 0;
	bytes[..prefix_end].copy_from_slice(&header.prefix[..prefix_end]);
	cursor += prefix_end;
	if separator != 0 {
		bytes[cursor] = b'/';
		cursor += 1;
	}
	bytes[cursor..length].copy_from_slice(&header.name[..name_end]);
	Ok(decode_text(&bytes[..length]))
}
fn apply_old_gnu_names<R: Read + Seek>(
	source: &mut R,
	offset: u64,
	size: u64,
	entries: &mut [Entry],
	entries_by_path: &mut HashMap<Str, usize>,
	pending: &mut [Option<PendingLink>],
	limits: Limits,
) -> Result<()> {
	source.seek(SeekFrom::Start(offset))?;
	let mut input = BufReader::with_capacity(IO_BUFFER_SIZE, source.take(size));
	let max_line = limits
		.path_size
		.checked_mul(2)
		.and_then(|length| length.checked_add(OLD_GNU_RENAME_PREFIX.len() as u64))
		.and_then(|length| length.checked_add(OLD_GNU_RENAME_SEPARATOR.len() as u64 + 1))
		.and_then(|length| usize::try_from(length).ok())
		.ok_or(Error::InvalidArchive("old-GNU rename length overflows"))?;
	let mut line = Vec::with_capacity(max_line.min(1024));
	let mut remaining = size;
	let mut candidate = true;
	let mut terminated = false;

	while remaining != 0 && !terminated {
		let available = input.fill_buf()?;
		if available.is_empty() {
			return Err(Error::InvalidArchive("truncated old-GNU name record"));
		}
		let count = available
			.len()
			.min(usize::try_from(remaining).unwrap_or(usize::MAX));
		for &byte in &available[..count] {
			if byte == 0 || byte == b'\n' {
				if candidate && !line.is_empty() {
					apply_old_gnu_name_line(&line, entries, entries_by_path, pending, limits)?;
				}
				line.clear();
				candidate = true;
				if byte == 0 {
					terminated = true;
					break;
				}
				continue;
			}
			if !candidate {
				continue;
			}
			if line.len() < OLD_GNU_RENAME_PREFIX.len() && byte != OLD_GNU_RENAME_PREFIX[line.len()] {
				line.clear();
				candidate = false;
				continue;
			}
			if line.len() == max_line {
				return Err(Error::InvalidArchive("old-GNU rename record is too long"));
			}
			line.push(byte);
		}
		input.consume(count);
		remaining -= count as u64;
	}
	if !terminated && candidate && !line.is_empty() {
		apply_old_gnu_name_line(&line, entries, entries_by_path, pending, limits)?;
	}
	Ok(())
}

fn apply_old_gnu_name_line(
	line: &[u8],
	entries: &mut [Entry],
	entries_by_path: &mut HashMap<Str, usize>,
	pending: &mut [Option<PendingLink>],
	limits: Limits,
) -> Result<()> {
	let Some(record) = line.strip_prefix(OLD_GNU_RENAME_PREFIX) else {
		return Ok(());
	};
	let separator = record
		.windows(OLD_GNU_RENAME_SEPARATOR.len())
		.position(|window| window == OLD_GNU_RENAME_SEPARATOR)
		.ok_or(Error::InvalidArchive("malformed old-GNU rename record"))?;
	let source = normalize_old_gnu_name(&record[..separator], limits)?;
	let mut target = &record[separator + OLD_GNU_RENAME_SEPARATOR.len()..];
	if target.last() == Some(&b'/') {
		target = &target[..target.len() - 1];
	}
	let unquoted = unquote_old_gnu_target(target);
	let target = normalize_old_gnu_name(unquoted.as_deref().unwrap_or(target), limits)?;
	rename_old_gnu_entries(
		entries,
		entries_by_path,
		pending,
		source.as_str(),
		target.as_str(),
		limits,
	)
}
fn unquote_old_gnu_target(raw: &[u8]) -> Option<Vec<u8>> {
	let first_escape = raw.iter().position(|&byte| byte == b'\\')?;
	let mut unquoted = Vec::with_capacity(raw.len());
	unquoted.extend_from_slice(&raw[..first_escape]);
	let mut cursor = first_escape;

	while cursor < raw.len() {
		if raw[cursor] != b'\\' {
			unquoted.push(raw[cursor]);
			cursor += 1;
			continue;
		}
		cursor += 1;
		let Some(&escape) = raw.get(cursor) else {
			unquoted.push(b'\\');
			break;
		};
		match escape {
			b'\\' => unquoted.push(b'\\'),
			b'n' => unquoted.push(b'\n'),
			b't' => unquoted.push(b'\t'),
			b'f' => unquoted.push(0x0c),
			b'b' => unquoted.push(0x08),
			b'r' => unquoted.push(b'\r'),
			b'?' => unquoted.push(0x7f),
			b'0'..=b'7' => {
				let mut value = escape - b'0';
				for _ in 0..2 {
					let Some(&digit) = raw.get(cursor + 1) else {
						break;
					};
					if !(b'0'..=b'7').contains(&digit) {
						break;
					}
					cursor += 1;
					value = value.wrapping_mul(8).wrapping_add(digit - b'0');
				}
				unquoted.push(value);
			},
			_ => {
				unquoted.push(b'\\');
				unquoted.push(escape);
			},
		}
		cursor += 1;
	}
	Some(unquoted)
}

fn normalize_old_gnu_name(raw: &[u8], limits: Limits) -> Result<Str> {
	check_path_size(raw.len() as u64, limits.path_size)?;
	let decoded = portable_path(decode_text(raw));
	if decoded.starts_with('/') {
		return Err(Error::InvalidArchive("invalid old-GNU rename path"));
	}
	let path = normalize(decoded.as_str(), false)
		.ok_or(Error::InvalidArchive("invalid old-GNU rename path"))?;
	validate(&path, limits)?;
	Ok(path)
}

fn rename_old_gnu_entries(
	entries: &mut [Entry],
	entries_by_path: &mut HashMap<Str, usize>,
	pending: &mut [Option<PendingLink>],
	source: &str,
	target: &str,
	limits: Limits,
) -> Result<()> {
	let moved: Vec<_> = entries
		.iter()
		.enumerate()
		.filter_map(|(index, entry)| {
			(!entry.path.is_empty() && has_path_prefix(entry.path.as_str(), source)).then_some(index)
		})
		.collect();
	for &index in &moved {
		entries_by_path.remove(entries[index].path.as_str());
	}
	for index in moved {
		let suffix = entries[index]
			.path
			.as_str()
			.strip_prefix(source)
			.expect("selected old-GNU path has the source prefix");
		let renamed = join_renamed_path(target, suffix, limits)?;
		entries[index].path = renamed.clone();
		if let Some(replaced) = entries_by_path.insert(renamed, index)
			&& replaced != index
		{
			entries[replaced] = Entry::synthetic_directory(Str::new(""));
			pending[replaced] = None;
		}
	}
	for link in pending.iter_mut().flatten() {
		if matches!(link.kind, LinkKind::Hard) && has_path_prefix(link.target.as_str(), source) {
			let suffix = link
				.target
				.as_str()
				.strip_prefix(source)
				.expect("selected old-GNU hard-link target has the source prefix");
			link.target = join_renamed_path(target, suffix, limits)?;
		}
	}
	Ok(())
}

fn has_path_prefix(path: &str, prefix: &str) -> bool {
	path == prefix
		|| path
			.strip_prefix(prefix)
			.is_some_and(|suffix| suffix.starts_with('/'))
}

fn join_renamed_path(prefix: &str, suffix: &str, limits: Limits) -> Result<Str> {
	let length = prefix
		.len()
		.checked_add(suffix.len())
		.ok_or(Error::InvalidArchive("old-GNU rename path overflows"))?;
	check_path_size(length as u64, limits.path_size)?;
	let mut path = StrMut::with_capacity(length);
	path.push_str(prefix);
	path.push_str(suffix);
	let path = path.freeze();
	validate(&path, limits)?;
	Ok(path)
}

fn portable_path(path: Str) -> Str {
	if !path.contains('\\') {
		return path;
	}
	let mut portable = StrMut::with_capacity(path.len());
	for (index, component) in path.as_str().split('\\').enumerate() {
		if index != 0 {
			portable.push('/');
		}
		portable.push_str(component);
	}
	portable.freeze()
}

fn read_long_text<R: Read + Seek>(
	source: &mut R,
	offset: u64,
	size: u64,
	path_limit: u64,
) -> Result<Str> {
	let effective_size = size.saturating_sub(1);
	check_path_size(effective_size, path_limit)?;
	let allocation = usize::try_from(size)
		.map_err(|_| Error::InvalidArchive("TAR metadata length does not fit memory"))?;
	let mut bytes = vec![0_u8; allocation];
	source.seek(SeekFrom::Start(offset))?;
	source.read_exact(&mut bytes)?;
	let end = bytes
		.iter()
		.position(|&byte| byte == 0)
		.unwrap_or(bytes.len());
	check_path_size(end as u64, path_limit)?;
	Ok(decode_text(&bytes[..end]))
}

fn parse_pax<R: Read + Seek>(
	source: &mut R,
	offset: u64,
	size: u64,
	limits: Limits,
) -> Result<PaxState> {
	source.seek(SeekFrom::Start(offset))?;
	let mut input = BufReader::with_capacity(IO_BUFFER_SIZE, source.take(size));
	let mut consumed = 0_u64;
	let mut state = PaxState::default();

	while consumed < size {
		let record_start = consumed;
		let mut record_len = 0_u64;
		let mut digits = 0_u32;
		loop {
			let byte = read_pax_byte(&mut input, &mut consumed)?;
			if byte == b' ' && digits > 0 {
				break;
			}
			if !byte.is_ascii_digit() || digits >= 20 {
				return Err(Error::InvalidArchive("malformed PAX record length"));
			}
			record_len = record_len
				.checked_mul(10)
				.and_then(|value| value.checked_add(u64::from(byte - b'0')))
				.ok_or(Error::InvalidArchive("PAX record length overflow"))?;
			digits += 1;
		}
		let prefix_len = consumed - record_start;
		if record_len <= prefix_len + 2 || record_len > size - record_start {
			return Err(Error::InvalidArchive("malformed PAX record"));
		}

		let mut remaining = record_len - prefix_len;
		let mut key = [0_u8; PAX_KEY_BUFFER_SIZE];
		let mut key_len = 0_usize;
		let mut sparse_prefix = true;
		loop {
			if remaining <= 1 {
				return Err(Error::InvalidArchive("malformed PAX key"));
			}
			let byte = read_pax_byte(&mut input, &mut consumed)?;
			remaining -= 1;
			if byte == b'=' {
				break;
			}
			if key_len < key.len() {
				key[key_len] = byte;
			}
			if key_len < GNU_SPARSE_PREFIX.len() && byte != GNU_SPARSE_PREFIX[key_len] {
				sparse_prefix = false;
			}
			key_len += 1;
		}

		let value_len = remaining - 1;
		let retained_key = if key_len <= key.len() {
			Some(&key[..key_len])
		} else {
			None
		};
		let sparse_key = sparse_prefix && key_len > GNU_SPARSE_PREFIX.len();
		if sparse_key {
			state.sparse = Some(value_len != 0);
		}

		match retained_key {
			Some(b"path") => {
				state.path =
					Some(read_pax_text(&mut input, &mut consumed, value_len, limits.path_size)?);
			},
			Some(b"linkpath") => {
				state.link_path =
					Some(read_pax_text(&mut input, &mut consumed, value_len, limits.path_size)?);
			},
			Some(b"size") => {
				state.size = Some(read_pax_decimal(&mut input, &mut consumed, value_len)?);
			},
			Some(b"GNU.sparse.name") => {
				state.sparse_name =
					Some(read_pax_text(&mut input, &mut consumed, value_len, limits.path_size)?);
			},
			Some(b"GNU.sparse.realsize") => {
				state.sparse_real_size = Some(read_pax_decimal(&mut input, &mut consumed, value_len)?);
			},
			_ => skip_exact(&mut input, value_len, &mut consumed)?,
		}
		if read_pax_byte(&mut input, &mut consumed)? != b'\n' {
			return Err(Error::InvalidArchive("PAX record is not newline terminated"));
		}
		if consumed != record_start + record_len {
			return Err(Error::InvalidArchive("malformed PAX record length"));
		}
	}
	Ok(state)
}

fn read_pax_text<R: Read>(
	input: &mut R,
	consumed: &mut u64,
	length: u64,
	path_limit: u64,
) -> Result<Str> {
	check_path_size(length, path_limit)?;
	let allocation = usize::try_from(length)
		.map_err(|_| Error::InvalidArchive("PAX value length does not fit memory"))?;
	let mut bytes = vec![0_u8; allocation];
	input.read_exact(&mut bytes)?;
	*consumed += length;
	Ok(decode_text(&bytes))
}

fn read_pax_decimal<R: Read>(input: &mut R, consumed: &mut u64, length: u64) -> Result<PaxNumber> {
	if length == 0 {
		return Ok(PaxNumber::Delete);
	}
	if length > 20 {
		return Err(Error::InvalidArchive("PAX numeric value is too long"));
	}
	let mut value = 0_u64;
	for _ in 0..length {
		let byte = read_pax_byte(input, consumed)?;
		if !byte.is_ascii_digit() {
			return Err(Error::InvalidArchive("invalid PAX numeric value"));
		}
		value = value
			.checked_mul(10)
			.and_then(|value| value.checked_add(u64::from(byte - b'0')))
			.ok_or(Error::InvalidArchive("PAX numeric value overflows"))?;
	}
	Ok(PaxNumber::Value(value))
}

fn read_pax_byte<R: Read>(input: &mut R, consumed: &mut u64) -> Result<u8> {
	let mut byte = [0_u8; 1];
	input.read_exact(&mut byte).map_err(|error| {
		if error.kind() == io::ErrorKind::UnexpectedEof {
			Error::InvalidArchive("truncated PAX record")
		} else {
			Error::Io(error)
		}
	})?;
	*consumed += 1;
	Ok(byte[0])
}

fn skip_exact<R: Read>(input: &mut R, length: u64, consumed: &mut u64) -> Result<()> {
	let copied = io::copy(&mut input.by_ref().take(length), &mut io::sink())?;
	if copied != length {
		return Err(Error::InvalidArchive("truncated PAX record"));
	}
	*consumed += length;
	Ok(())
}

fn normalize_member_path(raw: Str, limits: Limits) -> Result<Option<Str>> {
	check_path_size(raw.len() as u64, limits.path_size)?;
	let Some(path) = normalize(raw.as_str(), false) else {
		return Ok(None);
	};
	validate(&path, limits)?;
	Ok(Some(path))
}

const fn check_path_size(actual: u64, limit: u64) -> Result<()> {
	if actual > limit {
		Err(Error::PathTooLong { actual, limit })
	} else {
		Ok(())
	}
}

fn make_link_entry(
	path: Str,
	link_name: Str,
	typeflag: u8,
	modified_unix_seconds: Option<u64>,
	limits: Limits,
) -> Result<(Entry, Option<PendingLink>)> {
	check_path_size(link_name.len() as u64, limits.path_size)?;
	let portable = portable_path(link_name);
	let kind = if typeflag == b'1' {
		LinkKind::Hard
	} else {
		LinkKind::Symbolic
	};
	let target = match kind {
		LinkKind::Hard => match normalize(portable.as_str(), false) {
			Some(target) => {
				validate(&target, limits)?;
				target
			},
			None => {
				return Err(Error::InvalidHardLink {
					path,
					target: portable,
					reason: "invalid target",
				});
			},
		},
		LinkKind::Symbolic => {
			match normalize_symbolic_target(parent(path.as_str()), portable.as_str(), limits)? {
				Some(target) => target,
				None => {
					return Ok((
						Entry {
							path,
							directory: false,
							size: 0,
							modified_unix_seconds,
							storage: Storage::TarLink { target_path: portable },
						},
						None,
					));
				},
			}
		},
	};

	Ok((
		Entry {
			path,
			directory: false,
			size: 0,
			modified_unix_seconds,
			storage: Storage::TarLink { target_path: target.clone() },
		},
		Some(PendingLink { kind, target }),
	))
}

fn normalize_symbolic_target(base: &str, target: &str, limits: Limits) -> Result<Option<Str>> {
	if target.starts_with('/') || target.starts_with('\\') {
		return Ok(None);
	}
	if target
		.split(['/', '\\'])
		.find(|component| !component.is_empty() && *component != ".")
		.is_some_and(is_windows_drive)
	{
		return Ok(None);
	}
	let mut components: SmallVec<&str, 16> = SmallVec::new();
	components.extend(base.split('/').filter(|component| !component.is_empty()));
	for component in target.split(['/', '\\']) {
		match component {
			"" | "." => {},
			".." => {
				if components.pop().is_none() {
					return Ok(None);
				}
			},
			component => {
				components.push(component);
				if components.len() as u64 > limits.path_depth {
					return Err(Error::PathTooDeep {
						actual: components.len() as u64,
						limit:  limits.path_depth,
					});
				}
			},
		}
	}
	let length = components
		.iter()
		.map(|component| component.len())
		.sum::<usize>()
		.checked_add(components.len().saturating_sub(1))
		.ok_or(Error::InvalidArchive("TAR link target length overflows"))?;
	check_path_size(length as u64, limits.path_size)?;
	let mut normalized = StrMut::with_capacity(length);
	for (index, component) in components.into_iter().enumerate() {
		if index != 0 {
			normalized.push('/');
		}
		normalized.push_str(component);
	}
	Ok(Some(normalized.freeze()))
}

const fn is_windows_drive(component: &str) -> bool {
	let bytes = component.as_bytes();
	bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

fn upsert_entry(
	entries: &mut Vec<Entry>,
	entries_by_path: &mut HashMap<Str, usize>,
	pending: &mut Vec<Option<PendingLink>>,
	entry: Entry,
	pending_link: Option<PendingLink>,
	limit: u64,
) -> Result<()> {
	if let Some(&index) = entries_by_path.get(entry.path.as_str()) {
		entries[index] = entry;
		pending[index] = pending_link;
		return Ok(());
	}
	let actual = entries.len() as u64 + 1;
	if actual > limit {
		return Err(Error::TooManyEntries { actual, limit });
	}
	let index = entries.len();
	entries_by_path.insert(entry.path.clone(), index);
	entries.push(entry);
	pending.push(pending_link);
	Ok(())
}

fn resolve_pending_links(
	entries: &mut [Entry],
	entries_by_path: &HashMap<Str, usize>,
	pending: &[Option<PendingLink>],
	limits: Limits,
) -> Result<()> {
	let mut directory_prefixes = HashSet::<Str>::new();
	for entry in entries.iter() {
		let mut prefix = parent(entry.path.as_str());
		while !prefix.is_empty() {
			if !directory_prefixes.insert(prefix.into()) {
				break;
			}
			prefix = parent(prefix);
		}
	}

	let mut unresolved: Vec<bool> = pending.iter().map(Option::is_some).collect();
	let mut dependents: Vec<Vec<usize>> = (0..entries.len()).map(|_| Vec::new()).collect();
	let mut queue: VecDeque<usize> = unresolved
		.iter()
		.enumerate()
		.filter_map(|(index, &is_unresolved)| is_unresolved.then_some(index))
		.collect();

	while let Some(index) = queue.pop_front() {
		if !unresolved[index] {
			continue;
		}
		let link = pending[index]
			.as_ref()
			.expect("unresolved entries retain link state");
		let mut target_path = link.target.clone();
		let mut blocker = find_unresolved_blocker(entries_by_path, &unresolved, target_path.as_str());
		if blocker.is_none()
			&& let Ok(rewritten) =
				resolve_alias_path_map(entries, entries_by_path, target_path.clone(), limits)
		{
			target_path = rewritten;
			blocker = find_unresolved_blocker(entries_by_path, &unresolved, target_path.as_str());
		}

		if let Some(blocker_index) = blocker
			&& blocker_index != index
		{
			dependents[blocker_index].push(index);
			continue;
		}

		unresolved[index] = false;
		for dependent in mem::take(&mut dependents[index]) {
			queue.push_back(dependent);
		}

		if blocker == Some(index) {
			match link.kind {
				LinkKind::Hard => {
					return Err(Error::InvalidHardLink {
						path:   entries[index].path.clone(),
						target: link.target.clone(),
						reason: "cyclic target",
					});
				},
				LinkKind::Symbolic => {
					entries[index].storage = Storage::TarLink { target_path: link.target.clone() };
					continue;
				},
			}
		}

		if let Some(&target_index) = entries_by_path.get(target_path.as_str())
			&& !entries[target_index].directory
			&& !unresolved[target_index]
		{
			let target_size = entries[target_index].size;
			let target_storage = entries[target_index].storage.clone();
			entries[index].size = target_size;
			entries[index].storage = target_storage;
			continue;
		}

		let target_is_directory = target_path.is_empty()
			|| entries_by_path
				.get(target_path.as_str())
				.is_some_and(|&target_index| entries[target_index].directory)
			|| directory_prefixes.contains(target_path.as_str());
		if target_is_directory {
			match link.kind {
				LinkKind::Hard => {
					return Err(Error::InvalidHardLink {
						path:   entries[index].path.clone(),
						target: link.target.clone(),
						reason: "target is a directory",
					});
				},
				LinkKind::Symbolic => {
					entries[index].directory = true;
					entries[index].storage = Storage::TarLink { target_path: link.target.clone() };
					continue;
				},
			}
		}

		match link.kind {
			LinkKind::Symbolic => {
				entries[index].storage = Storage::TarLink { target_path: link.target.clone() };
			},
			LinkKind::Hard => {
				let reason = if entries_by_path.contains_key(target_path.as_str()) {
					"target is unreadable"
				} else {
					"target is missing"
				};
				return Err(Error::InvalidHardLink {
					path: entries[index].path.clone(),
					target: link.target.clone(),
					reason,
				});
			},
		}
	}

	if unresolved.iter().any(|&value| value) {
		return Err(Error::CyclicLinks);
	}
	Ok(())
}

fn find_unresolved_blocker(
	entries_by_path: &HashMap<Str, usize>,
	unresolved: &[bool],
	path: &str,
) -> Option<usize> {
	let mut end = path.len();
	while end > 0 {
		if let Some(&index) = entries_by_path.get(&path[..end])
			&& unresolved[index]
		{
			return Some(index);
		}
		end = path[..end].rfind('/').unwrap_or(0);
	}
	None
}

fn resolve_alias_path_map(
	entries: &[Entry],
	entries_by_path: &HashMap<Str, usize>,
	path: Str,
	limits: Limits,
) -> Result<Str> {
	let original = path.clone();
	let mut resolved = path;
	let mut rewrites = 0_u64;
	loop {
		let mut end = resolved.len();
		let mut alias = None;
		while end > 0 {
			if let Some(&index) = entries_by_path.get(&resolved[..end])
				&& entries[index].directory
				&& let Storage::TarLink { target_path } = &entries[index].storage
			{
				alias = Some((end, target_path.clone()));
				break;
			}
			end = resolved[..end].rfind('/').unwrap_or(0);
		}
		let Some((end, target)) = alias else {
			return Ok(resolved);
		};
		if rewrites == limits.link_depth {
			return Err(Error::LinkResolutionDepth { path: original, limit: limits.link_depth });
		}
		rewrites += 1;
		let suffix = resolved.get(end..).unwrap_or("").trim_start_matches('/');
		let replacement = join_alias_target(target.as_str(), suffix, limits)?;
		validate(&replacement, limits)?;
		resolved = replacement;
	}
}

fn find_directory_alias<'a>(entries: &'a [Entry], path: &str) -> Option<(usize, &'a Str)> {
	let mut end = path.len();
	while end > 0 {
		if let Ok(index) = entries.binary_search_by(|entry| entry.path.as_str().cmp(&path[..end])) {
			let entry = &entries[index];
			if entry.directory
				&& let Storage::TarLink { target_path } = &entry.storage
			{
				return Some((end, target_path));
			}
		}
		end = path[..end].rfind('/').unwrap_or(0);
	}
	None
}

fn join_alias_target(target: &str, suffix: &str, limits: Limits) -> Result<Str> {
	let separator = usize::from(!target.is_empty() && !suffix.is_empty());
	let length = target
		.len()
		.checked_add(separator)
		.and_then(|length| length.checked_add(suffix.len()))
		.ok_or(Error::InvalidArchive("TAR alias path length overflow"))?;
	check_path_size(length as u64, limits.path_size)?;
	if target.is_empty() {
		return Ok(suffix.into());
	}
	if suffix.is_empty() {
		return Ok(target.into());
	}
	let mut joined = StrMut::with_capacity(length);
	joined.push_str(target);
	joined.push('/');
	joined.push_str(suffix);
	Ok(joined.freeze())
}

fn padded_end(offset: u64, size: u64) -> Result<u64> {
	let padded = size
		.checked_add(BLOCK_SIZE as u64 - 1)
		.map(|value| value / BLOCK_SIZE as u64 * BLOCK_SIZE as u64)
		.ok_or(Error::InvalidArchive("TAR member size overflow"))?;
	checked_add(offset, padded, "TAR offset overflow")
}

fn checked_add(left: u64, right: u64, reason: &'static str) -> Result<u64> {
	left.checked_add(right).ok_or(Error::InvalidArchive(reason))
}
