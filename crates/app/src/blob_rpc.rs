//! Tonic projection of the daemon's content-addressed blob store.

use std::{path::PathBuf, sync::Arc};

use bytes::Bytes;
use futures::{Stream, StreamExt as _};
use omp_proto::omp::blob::v1 as pb;
use omp_storage::blob::{BlobRef, BlobStore};
use tonic::{Request, Response, Status};

const CHUNK_SIZE: usize = 64 * 1024;
const MAX_UPLOAD_BYTES: usize = 64 * 1024 * 1024;
type BlobStream = std::pin::Pin<Box<dyn Stream<Item = Result<pb::Chunk, Status>> + Send + 'static>>;

/// RPC server backed by one daemon-owned content-addressed store.
#[derive(Clone)]
pub struct BlobRpc {
	store: Arc<BlobStore>,
}

impl BlobRpc {
	/// Wraps a daemon-owned blob store.
	#[must_use]
	pub fn new(store: Arc<BlobStore>) -> Self {
		Self { store }
	}
}

#[tonic::async_trait]
impl pb::blob_server::Blob for BlobRpc {
	type GetStream = BlobStream;

	async fn stat(
		&self,
		request: Request<pb::StatRequest>,
	) -> Result<Response<pb::StatResponse>, Status> {
		let hash = parse_hash(&request.into_inner().hash)?;
		let reference = BlobRef { hash, size: 0 };
		let path = self.store.path(&reference);
		let metadata = tokio::task::spawn_blocking(move || std::fs::metadata(path))
			.await
			.map_err(join_status)?;
		match metadata {
			Ok(metadata) if metadata.is_file() => {
				Ok(Response::new(pb::StatResponse { present: true, size: metadata.len() }))
			},
			Ok(_) => Ok(Response::new(pb::StatResponse { present: false, size: 0 })),
			Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
				Ok(Response::new(pb::StatResponse { present: false, size: 0 }))
			},
			Err(error) => Err(io_status(error)),
		}
	}

	async fn get(
		&self,
		request: Request<pb::GetRequest>,
	) -> Result<Response<Self::GetStream>, Status> {
		let request = request.into_inner();
		let hash = parse_hash(&request.hash)?;
		let store = Arc::clone(&self.store);
		let bytes = tokio::task::spawn_blocking(move || {
			let path = store.path(&BlobRef { hash, size: 0 });
			let size = std::fs::metadata(&path)?.len();
			let bytes = std::fs::read(path)?;
			Ok::<_, std::io::Error>((size, Bytes::from(bytes)))
		})
		.await
		.map_err(join_status)?
		.map_err(io_status)?;
		let (size, bytes) = bytes;
		if request.offset > size {
			return Err(Status::out_of_range("blob range offset exceeds stored size"));
		}
		let end = if request.length == 0 {
			size
		} else {
			request
				.offset
				.checked_add(request.length)
				.unwrap_or(u64::MAX)
				.min(size)
		};
		let start = usize::try_from(request.offset)
			.map_err(|_| Status::out_of_range("blob offset exceeds platform limits"))?;
		let end = usize::try_from(end)
			.map_err(|_| Status::out_of_range("blob range exceeds platform limits"))?;
		let ranged = bytes.slice(start..end);
		let stream = async_stream::try_stream! {
			if ranged.is_empty() {
				yield pb::Chunk { data: Bytes::new(), hash: Bytes::copy_from_slice(&hash), size: Some(size) };
			} else {
				for (index, chunk) in ranged.chunks(CHUNK_SIZE).enumerate() {
					yield pb::Chunk {
						data: Bytes::copy_from_slice(chunk),
						hash: if index == 0 { Bytes::copy_from_slice(&hash) } else { Bytes::new() },
						size: (index == 0).then_some(size),
					};
				}
			}
		};
		Ok(Response::new(Box::pin(stream)))
	}

	async fn put(
		&self,
		request: Request<tonic::Streaming<pb::Chunk>>,
	) -> Result<Response<pb::PutResponse>, Status> {
		let mut incoming = request.into_inner();
		let mut bytes = Vec::new();
		let mut expected_hash = None;
		let mut expected_size = None;
		let mut first = true;
		while let Some(chunk) = incoming.next().await {
			let chunk = chunk?;
			if first {
				expected_hash = (!chunk.hash.is_empty())
					.then(|| parse_hash(&chunk.hash))
					.transpose()?;
				expected_size = chunk.size;
				first = false;
			} else if !chunk.hash.is_empty() || chunk.size.is_some() {
				return Err(Status::invalid_argument(
					"blob hash and declared size are permitted only on the first upload chunk",
				));
			}
			let next_len = bytes
				.len()
				.checked_add(chunk.data.len())
				.ok_or_else(|| Status::resource_exhausted("blob exceeds supported size"))?;
			if next_len > MAX_UPLOAD_BYTES {
				return Err(Status::resource_exhausted("blob exceeds the 64 MiB RPC upload limit"));
			}
			bytes.extend_from_slice(&chunk.data);
		}
		let actual_size = u64::try_from(bytes.len())
			.map_err(|_| Status::resource_exhausted("blob exceeds supported size"))?;
		if expected_size.is_some_and(|expected| expected != actual_size) {
			return Err(Status::invalid_argument("uploaded blob size does not match declared size"));
		}
		let store = Arc::clone(&self.store);
		let reference = tokio::task::spawn_blocking(move || store.put(&bytes))
			.await
			.map_err(join_status)?
			.map_err(storage_status)?;
		if expected_hash.is_some_and(|expected| expected != reference.hash) {
			return Err(Status::invalid_argument("uploaded blob hash does not match declared digest"));
		}
		Ok(Response::new(pb::PutResponse {
			hash: Bytes::copy_from_slice(&reference.hash),
			size: reference.size,
		}))
	}

	async fn delete(
		&self,
		request: Request<pb::DeleteRequest>,
	) -> Result<Response<pb::DeleteResponse>, Status> {
		let hash = parse_hash(&request.into_inner().hash)?;
		let path: PathBuf = self.store.path(&BlobRef { hash, size: 0 });
		let deleted = tokio::task::spawn_blocking(move || match std::fs::remove_file(path) {
			Ok(()) => Ok(true),
			Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
			Err(error) => Err(error),
		})
		.await
		.map_err(join_status)?
		.map_err(io_status)?;
		Ok(Response::new(pb::DeleteResponse { deleted }))
	}
}

fn parse_hash(bytes: &[u8]) -> Result<[u8; 32], Status> {
	bytes
		.try_into()
		.map_err(|_| Status::invalid_argument("blob hash must be exactly 32 bytes"))
}

fn join_status(error: tokio::task::JoinError) -> Status {
	Status::internal(format!("blob worker failed: {error}"))
}

fn io_status(error: std::io::Error) -> Status {
	if error.kind() == std::io::ErrorKind::NotFound {
		Status::not_found("blob not found")
	} else {
		Status::internal(format!("blob store I/O failed: {error}"))
	}
}

fn storage_status(error: omp_storage::blob::Error) -> Status {
	match error {
		omp_storage::blob::Error::NotFound => Status::not_found("blob not found"),
		other => Status::internal(format!("blob store failed: {other}")),
	}
}
