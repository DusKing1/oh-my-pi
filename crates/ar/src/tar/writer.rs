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
			self.write_long_name(emitted_name.as_bytes())?;
		}
		let (name, prefix) = split.unwrap_or((GNU_LONG_NAME.as_bytes(), &[]));
		let header = make_header(name, prefix, size, if directory { b'5' } else { b'0' })?;
		self.inner.write_all(header.as_bytes())?;
		self.inner.write_all(data)?;
		write_padding(&mut self.inner, data.len())?;
		self.paths.insert(path);
		Ok(())
	}

	fn write_long_name(&mut self, name: &[u8]) -> Result<()> {
		let payload_len = name
			.len()
			.checked_add(1)
			.ok_or(Error::TarFieldOverflow("GNU long name size"))?;
		let size =
			u64::try_from(payload_len).map_err(|_| Error::TarFieldOverflow("GNU long name size"))?;
		ensure_octal_fits(size, 12, "GNU long name size")?;
		let header = make_header(GNU_LONG_NAME.as_bytes(), &[], size, b'L')?;
		self.inner.write_all(header.as_bytes())?;
		self.inner.write_all(name)?;
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

fn make_header(name: &[u8], prefix: &[u8], size: u64, typeflag: u8) -> Result<UstarHeader> {
	if name.len() > 100 || prefix.len() > 155 {
		return Err(Error::TarFieldOverflow("path"));
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
