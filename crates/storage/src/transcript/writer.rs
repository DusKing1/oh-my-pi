//! Append-only transcript file writer.

use std::{
	fs::{File, OpenOptions},
	io::{Read, Seek, SeekFrom, Write},
	path::Path,
};

use bytes::BytesMut;
use serde_json::value::RawValue;

use super::{
	codec::{Error, Header, read_header, write_header, write_line},
	event::{Event, Kind},
};

/// A transcript writer that owns the single header and appends event lines.
pub struct Writer {
	file:       File,
	next_index: u64,
	line:       BytesMut,
}

impl Writer {
	/// Creates a new transcript and writes its line-zero header.
	///
	/// Creation fails when the path already exists so an append-only journal is
	/// never overwritten.
	pub fn create(path: &Path, header: &Header) -> Result<Self, Error> {
		if header.v != 4 {
			return Err(Error::InvalidHeaderVersion(header.v));
		}
		let mut file = OpenOptions::new()
			.read(true)
			.write(true)
			.create_new(true)
			.open(path)?;
		let mut line = BytesMut::new();
		write_header(header, &mut line)?;
		line.extend_from_slice(b"\n");
		file.write_all(&line)?;
		line.clear();
		Ok(Self { file, next_index: 0, line })
	}

	/// Opens an existing transcript for append, repairing a torn final line.
	///
	/// Complete malformed lines remain in place as tombstones. Only an
	/// unterminated final fragment that is not valid JSON is truncated.
	pub fn open_append(path: &Path) -> Result<Self, Error> {
		let mut file = OpenOptions::new().read(true).write(true).open(path)?;
		let mut bytes = Vec::new();
		file.read_to_end(&mut bytes)?;
		if bytes.is_empty() {
			return Err(Error::MissingHeader);
		}

		let Some(header_end) = bytes.iter().position(|byte| *byte == b'\n') else {
			read_header(&bytes)?;
			file.seek(SeekFrom::End(0))?;
			file.write_all(b"\n")?;
			return Ok(Self { file, next_index: 0, line: BytesMut::new() });
		};
		read_header(&bytes[..header_end])?;

		let mut next_index = 0_u64;
		let mut start = header_end + 1;
		while let Some(relative_end) = bytes[start..].iter().position(|byte| *byte == b'\n') {
			let end = start + relative_end;
			if serde_json::from_slice::<Header>(&bytes[start..end]).is_ok() {
				return Err(Error::DuplicateHeader);
			}
			next_index = next_index.saturating_add(1);
			start = end + 1;
		}

		if start < bytes.len() {
			let tail = &bytes[start..];
			if serde_json::from_slice::<Header>(tail).is_ok() {
				return Err(Error::DuplicateHeader);
			}
			if serde_json::from_slice::<Box<RawValue>>(tail).is_ok() {
				next_index = next_index.saturating_add(1);
				file.seek(SeekFrom::End(0))?;
				file.write_all(b"\n")?;
			} else {
				file.set_len(u64::try_from(start).expect("file offsets fit in u64"))?;
			}
		}
		file.seek(SeekFrom::End(0))?;
		Ok(Self { file, next_index, line: BytesMut::new() })
	}

	/// Rejects an attempt to write another header to this transcript.
	pub const fn write_header(&mut self, _header: &Header) -> Result<(), Error> {
		Err(Error::DuplicateHeader)
	}

	/// Appends an event and returns its assigned event index.
	///
	/// The event index is the physical line number minus one. Empty inference
	/// patches are rejected because they encode no state transition.
	pub fn append(&mut self, event: &Event) -> Result<u64, Error> {
		if let Kind::Infer { thinking, model, tier, cred_pin } = &event.kind
			&& thinking.is_unchanged()
			&& model.is_unchanged()
			&& tier.is_unchanged()
			&& cred_pin.is_unchanged()
		{
			return Err(Error::EmptyInfer);
		}
		if let Kind::Unknown(raw) = &event.kind
			&& serde_json::from_str::<Header>(raw.get()).is_ok()
		{
			return Err(Error::DuplicateHeader);
		}

		self.line.clear();
		write_line(event, &mut self.line)?;
		self.line.extend_from_slice(b"\n");
		self.file.write_all(&self.line)?;
		self.line.clear();
		let index = self.next_index;
		self.next_index = self.next_index.saturating_add(1);
		Ok(index)
	}
}
