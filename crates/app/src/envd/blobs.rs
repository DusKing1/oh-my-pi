//! Content-addressed env blob storage and hash-only result references.

use std::{io, path::Path};

use bytes::Bytes;
use omp_core::Str;
use omp_proto::{blob::v1 as blob_pb, thread::v1 as thread_pb};
use omp_storage::blob::{BlobRef, BlobStore};
use thiserror::Error;

/// Stable content identity returned by blob host operations.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BlobId {
	/// Raw BLAKE3-256 content digest.
	pub hash: [u8; 32],
	/// Exact byte length of the content.
	pub size: u64,
}

impl From<BlobRef> for BlobId {
	fn from(reference: BlobRef) -> Self {
		Self { hash: reference.hash, size: reference.size }
	}
}

impl From<BlobId> for BlobRef {
	fn from(id: BlobId) -> Self {
		Self { hash: id.hash, size: id.size }
	}
}

/// A complete or ranged blob read without text encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlobRead {
	/// Identity of the complete stored content.
	pub id:   BlobId,
	/// Requested complete or ranged content bytes.
	pub data: Bytes,
}

/// A blob request or backing-store operation failed.
#[derive(Debug, Error)]
pub enum BlobError {
	#[error(transparent)]
	Store(#[from] omp_storage::blob::Error),
	#[error("blob hash must be exactly 32 bytes")]
	InvalidHash,
	#[error("uploaded blob digest differs from the expected digest")]
	HashMismatch,
	#[error("uploaded blob size differs from expected {expected} bytes (received {actual})")]
	SizeMismatch { expected: u64, actual: u64 },
	#[error("blob range starts after the end of the content")]
	InvalidRange,
	#[error("blob length cannot be represented on this host")]
	LengthOverflow,
	#[error("blob removal failed: {0}")]
	Remove(#[source] io::Error),
}

/// Concrete env-side owner of a filesystem-backed content-addressed store.
#[derive(Clone, Debug)]
pub struct BlobHost {
	store: BlobStore,
}

impl BlobHost {
	/// Opens or creates a content-addressed store beneath `root`.
	pub fn open(root: impl AsRef<Path>) -> Result<Self, BlobError> {
		Ok(Self { store: BlobStore::open(root.as_ref())? })
	}

	/// Takes ownership of an already-open store.
	pub const fn from_store(store: BlobStore) -> Self {
		Self { store }
	}

	/// Stores exact bytes and returns their BLAKE3-derived identity.
	pub fn put(&self, data: &[u8]) -> Result<BlobId, BlobError> {
		self
			.store
			.put(data)
			.map(BlobId::from)
			.map_err(BlobError::from)
	}

	/// Stores bytes while validating optional upload-stream preconditions.
	pub fn put_checked(
		&self,
		data: &[u8],
		expected_hash: Option<&[u8]>,
		expected_size: Option<u64>,
	) -> Result<BlobId, BlobError> {
		let expected_hash = expected_hash.map(parse_hash).transpose()?;
		let actual_size = u64::try_from(data.len()).map_err(|_| BlobError::LengthOverflow)?;
		if let Some(expected) = expected_size
			&& expected != actual_size
		{
			return Err(BlobError::SizeMismatch { expected, actual: actual_size });
		}
		if expected_hash.is_some_and(|expected| expected != *blake3::hash(data).as_bytes()) {
			return Err(BlobError::HashMismatch);
		}
		self.put(data)
	}

	/// Stores exact bytes and returns the env wire response.
	pub fn put_response(&self, data: &[u8]) -> Result<blob_pb::PutResponse, BlobError> {
		let id = self.put(data)?;
		Ok(blob_pb::PutResponse { hash: Bytes::copy_from_slice(&id.hash), size: id.size })
	}

	/// Returns presence and size for a raw BLAKE3 digest.
	pub fn stat(&self, hash: &[u8]) -> Result<blob_pb::StatResponse, BlobError> {
		let hash = parse_hash(hash)?;
		let probe = BlobRef { hash, size: 0 };
		match std::fs::metadata(self.store.path(&probe)) {
			Ok(metadata) if metadata.is_file() => {
				Ok(blob_pb::StatResponse { present: true, size: metadata.len() })
			},
			Ok(_) => Ok(blob_pb::StatResponse { present: false, size: 0 }),
			Err(error) if error.kind() == io::ErrorKind::NotFound => {
				Ok(blob_pb::StatResponse { present: false, size: 0 })
			},
			Err(error) => Err(BlobError::Store(error.into())),
		}
	}

	/// Reads a complete blob by content identity.
	pub fn get(&self, id: BlobId) -> Result<Bytes, BlobError> {
		self.store.get(&id.into()).map_err(BlobError::from)
	}

	/// Reads the env wire range without base64 or another text projection.
	pub fn get_request(&self, request: &blob_pb::GetRequest) -> Result<BlobRead, BlobError> {
		let hash = parse_hash(&request.hash)?;
		let stat = self.stat(&request.hash)?;
		if !stat.present {
			return Err(BlobError::Store(omp_storage::blob::Error::NotFound));
		}
		if request.offset > stat.size {
			return Err(BlobError::InvalidRange);
		}
		let available = stat.size - request.offset;
		let length = if request.length == 0 {
			available
		} else {
			request.length.min(available)
		};
		let end = request
			.offset
			.checked_add(length)
			.ok_or(BlobError::InvalidRange)?;
		let start = usize::try_from(request.offset).map_err(|_| BlobError::LengthOverflow)?;
		let end = usize::try_from(end).map_err(|_| BlobError::LengthOverflow)?;
		let id = BlobId { hash, size: stat.size };
		let data = self.get(id)?.slice(start..end);
		Ok(BlobRead { id, data })
	}

	/// Removes a raw digest and reports whether content existed.
	pub fn delete(&self, hash: &[u8]) -> Result<blob_pb::DeleteResponse, BlobError> {
		let hash = parse_hash(hash)?;
		let probe = BlobRef { hash, size: 0 };
		match std::fs::remove_file(self.store.path(&probe)) {
			Ok(()) => Ok(blob_pb::DeleteResponse { deleted: true }),
			Err(error) if error.kind() == io::ErrorKind::NotFound => {
				Ok(blob_pb::DeleteResponse { deleted: false })
			},
			Err(error) => Err(BlobError::Remove(error)),
		}
	}

	/// Creates the canonical hash-only media/result shape used by thread parts.
	pub fn reference(
		&self,
		id: BlobId,
		mime: Str,
		detail: thread_pb::blob::Detail,
	) -> thread_pb::Blob {
		thread_pb::Blob {
			hash:   Bytes::copy_from_slice(&id.hash),
			mime:   mime.into(),
			size:   id.size,
			inline: Bytes::new(),
			detail: detail.into(),
		}
	}

	/// Stores media/result bytes and returns their canonical hash-only shape.
	pub fn put_reference(
		&self,
		data: &[u8],
		mime: Str,
		detail: thread_pb::blob::Detail,
	) -> Result<thread_pb::Blob, BlobError> {
		let id = self.put(data)?;
		Ok(self.reference(id, mime, detail))
	}
}

fn parse_hash(hash: &[u8]) -> Result<[u8; 32], BlobError> {
	hash.try_into().map_err(|_| BlobError::InvalidHash)
}
