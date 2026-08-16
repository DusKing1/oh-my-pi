//! Append-only transcript file writer.

use std::{
	fs::{File, OpenOptions},
	io::{Seek, SeekFrom, Write},
	path::Path,
};

use bytes::BytesMut;

use super::{
	codec::{Error, Header, read_header, read_line, write_header, write_line},
	event::{Event, Kind},
};

/// A transcript writer that owns the single header and appends event lines.
pub struct Writer {
	file:       File,
	next_index: u64,
	line:       BytesMut,
}

trait AppendTarget: Write {
	fn append_len(&self) -> std::io::Result<u64>;
	fn rollback_to(&mut self, len: u64) -> std::io::Result<()>;
}

impl AppendTarget for File {
	fn append_len(&self) -> std::io::Result<u64> {
		Ok(self.metadata()?.len())
	}

	fn rollback_to(&mut self, len: u64) -> std::io::Result<()> {
		self.set_len(len)?;
		self.seek(SeekFrom::Start(len))?;
		Ok(())
	}
}

fn append_all(target: &mut impl AppendTarget, bytes: &[u8]) -> Result<(), Error> {
	let original_len = target.append_len()?;
	if let Err(write) = target.write_all(bytes) {
		return match target.rollback_to(original_len) {
			Ok(()) => Err(Error::Io(write)),
			Err(rollback) => Err(Error::AppendRollback { write, rollback }),
		};
	}
	Ok(())
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

	/// Opens an existing transcript for append, repairing malformed trailing records.
	///
	/// Complete malformed lines in the middle remain in place as tombstones so
	/// physical event indexes stay stable. A malformed trailing run cannot be
	/// referenced by a later event and is truncated before appending resumes.
	pub fn open_append(path: &Path) -> Result<Self, Error> {
		let mut file = OpenOptions::new().read(true).write(true).open(path)?;
		let mut bytes = Vec::new();
		std::io::Read::read_to_end(&mut file, &mut bytes)?;
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
		let mut malformed_tail = None;
		let mut start = header_end + 1;
		while let Some(relative_end) = bytes[start..].iter().position(|byte| *byte == b'\n') {
			let end = start + relative_end;
			let line = &bytes[start..end];
			if serde_json::from_slice::<Header>(line).is_ok() {
				return Err(Error::DuplicateHeader);
			}
			if read_line(line).is_ok() {
				malformed_tail = None;
			} else if malformed_tail.is_none() {
				malformed_tail = Some((
					u64::try_from(start).expect("file offsets fit in u64"),
					next_index,
				));
			}
			next_index = next_index.saturating_add(1);
			start = end + 1;
		}

		if start < bytes.len() {
			let tail = &bytes[start..];
			if serde_json::from_slice::<Header>(tail).is_ok() {
				return Err(Error::DuplicateHeader);
			}
			if read_line(tail).is_ok() {
				malformed_tail = None;
				next_index = next_index.saturating_add(1);
				file.seek(SeekFrom::End(0))?;
				file.write_all(b"\n")?;
			} else if malformed_tail.is_none() {
				malformed_tail = Some((
					u64::try_from(start).expect("file offsets fit in u64"),
					next_index,
				));
			}
		}
		if let Some((offset, repaired_next_index)) = malformed_tail {
			file.set_len(offset)?;
			next_index = repaired_next_index;
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
		append_all(&mut self.file, &self.line)?;
		self.line.clear();
		let index = self.next_index;
		self.next_index = self.next_index.saturating_add(1);
		Ok(index)
	}
}

#[cfg(test)]
mod tests {
	use std::io::{self, Write};

	use super::{AppendTarget, append_all};

	struct PartialTarget {
		bytes:      Vec<u8>,
		write_left: Option<usize>,
	}

	impl Write for PartialTarget {
		fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
			let Some(left) = self.write_left else {
				self.bytes.extend_from_slice(bytes);
				return Ok(bytes.len());
			};
			if left == 0 {
				return Err(io::Error::new(io::ErrorKind::StorageFull, "injected full device"));
			}
			let written = left.min(bytes.len());
			self.bytes.extend_from_slice(&bytes[..written]);
			self.write_left = Some(left - written);
			Ok(written)
		}

		fn flush(&mut self) -> io::Result<()> {
			Ok(())
		}
	}

	impl AppendTarget for PartialTarget {
		fn append_len(&self) -> io::Result<u64> {
			Ok(u64::try_from(self.bytes.len()).expect("test buffer length fits in u64"))
		}

		fn rollback_to(&mut self, len: u64) -> io::Result<()> {
			self
				.bytes
				.truncate(usize::try_from(len).expect("test buffer length fits in usize"));
			Ok(())
		}
	}

	#[test]
	fn partial_append_rolls_back_and_target_remains_retryable() {
		let mut target = PartialTarget { bytes: b"complete\n".to_vec(), write_left: Some(5) };
		assert!(append_all(&mut target, b"{\"torn\":true}\n").is_err());
		assert_eq!(target.bytes, b"complete\n");

		target.write_left = None;
		append_all(&mut target, b"{\"complete\":true}\n").expect("retry succeeds");
		assert_eq!(target.bytes, b"complete\n{\"complete\":true}\n");
	}
}
