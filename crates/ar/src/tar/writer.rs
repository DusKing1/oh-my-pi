//! Incremental deterministic TAR and TAR.GZ writing.

use std::{borrow::Cow, collections::HashSet, io::Write};

use flate2::{Compression, GzBuilder};
use omp_core::Str;
use zerocopy::IntoBytes;

use super::spec::{BLOCK_SIZE, UstarHeader};
use crate::{
	Error, Limits, Result,
	path::{is_directory_name, normalize_bounded},
};

const GNU_LONG_NAME: &str = "././@LongLink";
const ZERO_BLOCK: [u8; BLOCK_SIZE] = [0; BLOCK_SIZE];

/// A non-seeking, deterministic USTAR writer.
///
/// Records are written as entries are added. Paths that cannot be represented
/// by the USTAR name and prefix fields are emitted using a GNU long-name
/// metadata record.
pub struct Writer<W: Write> {
	inner: W,
	paths: HashSet<Str>,
}

impl<W: Write> Writer<W> {
	/// Creates an empty TAR archive that writes to `inner`.
	pub fn new(inner: W) -> Self {
		Self { inner, paths: HashSet::new() }
	}

	/// Adds a regular file at `path` with the supplied bytes.
	pub fn add_file(&mut self, path: &str, data: &[u8]) -> Result<()> {
		if is_directory_name(path) {
			return Err(Error::UnsafePath(path.into()));
		}
		let path = normalize_bounded(path, Limits::DEFAULT)?;
		self.add_entry(path, false, data)
	}

	/// Adds an explicit directory entry at `path`.
	///
	/// The emitted member name has exactly one trailing `/`.
	pub fn add_directory(&mut self, path: &str) -> Result<()> {
		let path = normalize_bounded(path, Limits::DEFAULT)?;
		self.add_entry(path, true, &[])
	}

	/// Adds a symbolic-link entry with the supplied link text.
	///
	/// Relative `..` components and absolute targets are retained because they
	/// are meaningful symbolic-link text. Capability-scoped extraction still
	/// refuses targets that cannot be resolved inside the archive.
	pub fn add_symlink(&mut self, path: &str, target: &str) -> Result<()> {
		if is_directory_name(path) {
			return Err(Error::UnsafePath(path.into()));
		}
		let path = normalize_bounded(path, Limits::DEFAULT)?;
		let target = portable_link_target(target)?;
		self.add_link(path, target.as_bytes(), b'2')
	}

	/// Adds a hard-link entry targeting another archive-relative member path.
	pub fn add_hard_link(&mut self, path: &str, target: &str) -> Result<()> {
		if is_directory_name(path) {
			return Err(Error::UnsafePath(path.into()));
		}
		let path = normalize_bounded(path, Limits::DEFAULT)?;
		let target = normalize_bounded(target, Limits::DEFAULT)?;
		self.add_link(path, target.as_bytes(), b'1')
	}

	/// Writes the two terminating zero blocks and returns the wrapped writer.
	pub fn finish(mut self) -> Result<W> {
		self.inner.write_all(&ZERO_BLOCK)?;
		self.inner.write_all(&ZERO_BLOCK)?;
		Ok(self.inner)
	}

	fn add_entry(&mut self, path: Str, directory: bool, data: &[u8]) -> Result<()> {
		if self.paths.contains(path.as_str()) {
			return Err(Error::DuplicatePath(path));
		}
		let emitted_name = if directory {
			let mut name = String::with_capacity(path.len() + 1);
			name.push_str(path.as_str());
			name.push('/');
			Cow::Owned(name)
		} else {
			Cow::Borrowed(path.as_str())
		};
		let size = u64::try_from(data.len()).map_err(|_| Error::TarFieldOverflow("size"))?;
		ensure_octal_fits(size, 12, "size")?;

		let split = split_ustar_name(emitted_name.as_bytes());
		if split.is_none() {
			self.write_long_text(emitted_name.as_bytes(), b'L', "GNU long name size")?;
		}
		let (name, prefix) = split.unwrap_or((GNU_LONG_NAME.as_bytes(), &[]));
		let header = make_header(name, prefix, size, if directory { b'5' } else { b'0' }, &[])?;
		self.inner.write_all(header.as_bytes())?;
		self.inner.write_all(data)?;
		write_padding(&mut self.inner, data.len())?;
		self.paths.insert(path);
		Ok(())
	}

	fn add_link(&mut self, path: Str, target: &[u8], typeflag: u8) -> Result<()> {
		if self.paths.contains(path.as_str()) {
			return Err(Error::DuplicatePath(path));
		}
		let split = split_ustar_name(path.as_bytes());
		if split.is_none() {
			self.write_long_text(path.as_bytes(), b'L', "GNU long name size")?;
		}
		if target.len() > 100 {
			self.write_long_text(target, b'K', "GNU long link size")?;
		}
		let (name, prefix) = split.unwrap_or((GNU_LONG_NAME.as_bytes(), &[]));
		let link_name = if target.len() <= 100 { target } else { &[] };
		let header = make_header(name, prefix, 0, typeflag, link_name)?;
		self.inner.write_all(header.as_bytes())?;
		self.paths.insert(path);
		Ok(())
	}

	fn write_long_text(
		&mut self,
		value: &[u8],
		typeflag: u8,
		overflow_field: &'static str,
	) -> Result<()> {
		let payload_len = value
			.len()
			.checked_add(1)
			.ok_or(Error::TarFieldOverflow(overflow_field))?;
		let size = u64::try_from(payload_len).map_err(|_| Error::TarFieldOverflow(overflow_field))?;
		ensure_octal_fits(size, 12, overflow_field)?;
		let header = make_header(GNU_LONG_NAME.as_bytes(), &[], size, typeflag, &[])?;
		self.inner.write_all(header.as_bytes())?;
		self.inner.write_all(value)?;
		self.inner.write_all(&[0])?;
		write_padding(&mut self.inner, payload_len)
	}
}

/// Encodes an iterator of `(path, bytes)` files as an in-memory TAR archive.
///
/// Entries are emitted in iterator order.
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

/// Encodes files as a deterministic gzip-compressed TAR archive.
///
/// The gzip header has an mtime of zero and contains no host-dependent name.
pub fn encode_gzip<I, P, D>(entries: I) -> Result<Vec<u8>>
where
	I: IntoIterator<Item = (P, D)>,
	P: AsRef<str>,
	D: AsRef<[u8]>,
{
	let encoder = GzBuilder::new()
		.mtime(0)
		.write(Vec::new(), Compression::default());
	let mut writer = Writer::new(encoder);
	for (path, data) in entries {
		writer.add_file(path.as_ref(), data.as_ref())?;
	}
	let encoder = writer.finish()?;
	Ok(encoder.finish()?)
}

fn split_ustar_name(path: &[u8]) -> Option<(&[u8], &[u8])> {
	if path.len() <= 100 {
		return Some((path, &[]));
	}
	path
		.iter()
		.enumerate()
		.rev()
		.find(|&(index, byte)| {
			*byte == b'/' && index + 1 < path.len() && index <= 155 && path.len() - index - 1 <= 100
		})
		.map(|(index, _)| (&path[index + 1..], &path[..index]))
}

fn make_header(
	name: &[u8],
	prefix: &[u8],
	size: u64,
	typeflag: u8,
	link_name: &[u8],
) -> Result<UstarHeader> {
	if name.len() > 100 || prefix.len() > 155 {
		return Err(Error::TarFieldOverflow("path"));
	}
	if link_name.len() > 100 {
		return Err(Error::TarFieldOverflow("link name"));
	}
	let mut header = UstarHeader {
		name: [0; 100],
		mode: [0; 8],
		uid: [0; 8],
		gid: [0; 8],
		size: [0; 12],
		mtime: [0; 12],
		checksum: [b' '; 8],
		typeflag,
		link_name: [0; 100],
		magic: *b"ustar\0",
		version: *b"00",
		owner_name: [0; 32],
		group_name: [0; 32],
		device_major: [0; 8],
		device_minor: [0; 8],
		prefix: [0; 155],
		padding: [0; 12],
	};
	header.name[..name.len()].copy_from_slice(name);
	header.prefix[..prefix.len()].copy_from_slice(prefix);
	header.link_name[..link_name.len()].copy_from_slice(link_name);
	write_octal(&mut header.mode, if typeflag == b'5' { 0o755 } else { 0o644 }, "mode")?;
	write_octal(&mut header.uid, 0, "uid")?;
	write_octal(&mut header.gid, 0, "gid")?;
	write_octal(&mut header.size, size, "size")?;
	write_octal(&mut header.mtime, 0, "mtime")?;
	write_octal(&mut header.device_major, 0, "device major")?;
	write_octal(&mut header.device_minor, 0, "device minor")?;
	let checksum = header.as_bytes().iter().map(|&byte| u64::from(byte)).sum();
	write_checksum(&mut header.checksum, checksum)?;
	Ok(header)
}

fn portable_link_target(target: &str) -> Result<Cow<'_, str>> {
	if target.is_empty() || target.contains('\0') {
		return Err(Error::UnsafePath(target.into()));
	}
	let limit = Limits::DEFAULT.max_path_size();
	if target.len() as u64 > limit {
		return Err(Error::PathTooLong { actual: target.len() as u64, limit });
	}
	if target.contains('\\') {
		Ok(Cow::Owned(target.replace('\\', "/")))
	} else {
		Ok(Cow::Borrowed(target))
	}
}

fn ensure_octal_fits(value: u64, width: usize, field: &'static str) -> Result<()> {
	let digits = width.checked_sub(1).ok_or(Error::TarFieldOverflow(field))?;
	let bits = digits
		.checked_mul(3)
		.ok_or(Error::TarFieldOverflow(field))?;
	if bits < 64 && value >= (1_u64 << bits) {
		return Err(Error::TarFieldOverflow(field));
	}
	Ok(())
}

fn write_octal(field: &mut [u8], mut value: u64, name: &'static str) -> Result<()> {
	ensure_octal_fits(value, field.len(), name)?;
	field.fill(b'0');
	let terminator = field.len() - 1;
	field[terminator] = 0;
	for byte in field[..terminator].iter_mut().rev() {
		*byte = b'0' + (value & 7) as u8;
		value >>= 3;
	}
	Ok(())
}

fn write_checksum(field: &mut [u8; 8], mut value: u64) -> Result<()> {
	ensure_octal_fits(value, 7, "checksum")?;
	field.fill(b'0');
	field[6] = 0;
	field[7] = b' ';
	for byte in field[..6].iter_mut().rev() {
		*byte = b'0' + (value & 7) as u8;
		value >>= 3;
	}
	Ok(())
}

fn write_padding<W: Write>(writer: &mut W, size: usize) -> Result<()> {
	let remainder = size % BLOCK_SIZE;
	if remainder != 0 {
		writer.write_all(&ZERO_BLOCK[..BLOCK_SIZE - remainder])?;
	}
	Ok(())
}
