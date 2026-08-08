//! Remote content-addressed blob transfer.
//!
//! The listener supplies the security boundary: owner-only Unix sockets are
//! trusted locally, while remote listeners require mTLS or their configured
//! bearer token before dispatch reaches this service.

use std::{
	io::{self, Read},
	pin::Pin,
	sync::Arc,
};

use bytes::{Buf, Bytes};
use futures::Stream;
use omp_proto::blob::v1::{
	Chunk, DeleteRequest, DeleteResponse, GetRequest, PutResponse, StatRequest, StatResponse,
	blob_server::Blob,
};
use omp_storage::blob::{BlobRef, BlobStore};
use tokio::{
	io::{AsyncReadExt, AsyncSeekExt},
	sync::mpsc,
};
use tonic::{Request, Response, Status, Streaming};

const TRANSFER_CHUNK_SIZE: usize = 64 * 1024;
/// Default maximum accepted upload size (256 MiB).
pub const DEFAULT_MAX_UPLOAD_SIZE: u64 = 256 * 1024 * 1024;
/// Server stream returned by [`BlobService`] downloads.
pub type BlobGetStream = Pin<Box<dyn Stream<Item = Result<Chunk, Status>> + Send + 'static>>;

/// Native gRPC access to a shared [`BlobStore`].
#[derive(Clone, Debug)]
pub struct BlobService {
	store:           Arc<BlobStore>,
	max_upload_size: u64,
}

impl BlobService {
	/// Creates a service using the production upload limit.
	#[must_use]
	pub const fn new(store: Arc<BlobStore>) -> Self {
		Self { store, max_upload_size: DEFAULT_MAX_UPLOAD_SIZE }
	}

	/// Overrides the upload limit, primarily for constrained deployments.
	#[must_use]
	pub const fn with_max_upload_size(mut self, bytes: u64) -> Self {
		self.max_upload_size = bytes;
		self
	}
}

#[tonic::async_trait]
impl Blob for BlobService {
	type GetStream = BlobGetStream;

	async fn stat(&self, request: Request<StatRequest>) -> Result<Response<StatResponse>, Status> {
		let hash = parse_hash(&request.into_inner().hash)?;
		let reference = BlobRef { hash, size: 0 };
		match tokio::fs::metadata(self.store.path(&reference)).await {
			Ok(metadata) if metadata.is_file() => {
				Ok(Response::new(StatResponse { present: true, size: metadata.len() }))
			},
			Ok(_) => Ok(Response::new(StatResponse { present: false, size: 0 })),
			Err(error) if error.kind() == io::ErrorKind::NotFound => {
				Ok(Response::new(StatResponse { present: false, size: 0 }))
			},
			Err(_) => Err(Status::internal("blob metadata is unavailable")),
		}
	}

	async fn get(&self, request: Request<GetRequest>) -> Result<Response<Self::GetStream>, Status> {
		let request = request.into_inner();
		let hash = parse_hash(&request.hash)?;
		let reference = BlobRef { hash, size: 0 };
		let path = self.store.path(&reference);
		let mut file = tokio::fs::File::open(path).await.map_err(map_open_error)?;
		let size = file
			.metadata()
			.await
			.map_err(|_| Status::internal("blob metadata is unavailable"))?
			.len();
		if request.offset > size {
			return Err(Status::out_of_range("range offset exceeds blob size"));
		}
		let available = size - request.offset;
		let remaining = if request.length == 0 {
			available
		} else {
			request.length.min(available)
		};
		file
			.seek(io::SeekFrom::Start(request.offset))
			.await
			.map_err(|_| Status::internal("blob range is unavailable"))?;

		let stream = async_stream::try_stream! {
			let mut remaining = remaining;
			let mut first = true;
			let mut buffer = vec![0_u8; TRANSFER_CHUNK_SIZE];
			if remaining == 0 {
				yield Chunk { data: Bytes::new(), hash: Bytes::copy_from_slice(&hash), size: Some(size) };
				return;
			}
			while remaining != 0 {
				let limit = usize::try_from(remaining.min(TRANSFER_CHUNK_SIZE as u64))
					.expect("transfer chunk length fits usize");
				let read = file.read(&mut buffer[..limit]).await
					.map_err(|_| Status::internal("blob read failed"))?;
				if read == 0 {
					Err(Status::data_loss("blob was truncated during transfer"))?;
				}
				remaining -= read as u64;
				yield Chunk {
					data: Bytes::copy_from_slice(&buffer[..read]),
					hash: if first { Bytes::copy_from_slice(&hash) } else { Bytes::new() },
					size: first.then_some(size),
				};
				first = false;
			}
		};
		Ok(Response::new(Box::pin(stream)))
	}

	async fn put(
		&self,
		request: Request<Streaming<Chunk>>,
	) -> Result<Response<PutResponse>, Status> {
		let mut stream = request.into_inner();
		let (sender, receiver) = mpsc::channel::<UploadMessage>(2);
		let store = Arc::clone(&self.store);
		let writer =
			tokio::task::spawn_blocking(move || store.put_reader(ChunkReader::new(receiver)));
		let mut sender = Some(sender);
		let mut received = 0_u64;
		let mut hasher = blake3::Hasher::new();
		let mut expected_hash = None;
		let mut expected_size = None;
		let mut first = true;

		loop {
			let chunk = match stream.message().await {
				Ok(Some(chunk)) => chunk,
				Ok(None) => break,
				Err(_) => {
					return abort_upload(
						sender.take(),
						writer,
						Status::cancelled("upload stream failed"),
					)
					.await;
				},
			};
			if first {
				if !chunk.hash.is_empty() {
					expected_hash = Some(parse_hash(&chunk.hash)?);
				}
				expected_size = chunk.size;
				first = false;
			} else if !chunk.hash.is_empty() || chunk.size.is_some() {
				return abort_upload(
					sender.take(),
					writer,
					Status::invalid_argument("upload metadata is only allowed on the first chunk"),
				)
				.await;
			}
			let chunk_size = u64::try_from(chunk.data.len())
				.map_err(|_| Status::resource_exhausted("upload is too large"))?;
			received = match received.checked_add(chunk_size) {
				Some(size) if size <= self.max_upload_size => size,
				_ => {
					return abort_upload(
						sender.take(),
						writer,
						Status::resource_exhausted("upload exceeds the configured size limit"),
					)
					.await;
				},
			};
			hasher.update(&chunk.data);
			if sender
				.as_ref()
				.expect("upload sender is present")
				.send(UploadMessage::Data(chunk.data))
				.await
				.is_err()
			{
				return Err(Status::internal("blob upload writer stopped"));
			}
		}

		let actual_hash = *hasher.finalize().as_bytes();
		if expected_size.is_some_and(|size| size != received) {
			return abort_upload(
				sender.take(),
				writer,
				Status::invalid_argument("uploaded size does not match the declared size"),
			)
			.await;
		}
		if expected_hash.is_some_and(|hash| hash != actual_hash) {
			return abort_upload(
				sender.take(),
				writer,
				Status::invalid_argument("uploaded content does not match the declared hash"),
			)
			.await;
		}
		sender
			.take()
			.expect("upload sender is present")
			.send(UploadMessage::End)
			.await
			.map_err(|_| Status::internal("blob upload writer stopped"))?;
		let reference = writer
			.await
			.map_err(|_| Status::internal("blob upload writer failed"))?
			.map_err(|_| Status::internal("blob could not be stored"))?;
		Ok(Response::new(PutResponse {
			hash: Bytes::copy_from_slice(&reference.hash),
			size: reference.size,
		}))
	}

	async fn delete(
		&self,
		request: Request<DeleteRequest>,
	) -> Result<Response<DeleteResponse>, Status> {
		let hash = parse_hash(&request.into_inner().hash)?;
		let reference = BlobRef { hash, size: 0 };
		match tokio::fs::remove_file(self.store.path(&reference)).await {
			Ok(()) => Ok(Response::new(DeleteResponse { deleted: true })),
			Err(error) if error.kind() == io::ErrorKind::NotFound => {
				Ok(Response::new(DeleteResponse { deleted: false }))
			},
			Err(_) => Err(Status::internal("blob could not be deleted")),
		}
	}
}

async fn abort_upload(
	sender: Option<mpsc::Sender<UploadMessage>>,
	writer: tokio::task::JoinHandle<Result<BlobRef, omp_storage::blob::Error>>,
	status: Status,
) -> Result<Response<PutResponse>, Status> {
	drop(sender);
	let _ = writer.await;
	Err(status)
}

enum UploadMessage {
	Data(Bytes),
	End,
}

struct ChunkReader {
	receiver: mpsc::Receiver<UploadMessage>,
	chunk:    Bytes,
	ended:    bool,
}

impl ChunkReader {
	fn new(receiver: mpsc::Receiver<UploadMessage>) -> Self {
		Self { receiver, chunk: Bytes::new(), ended: false }
	}
}

impl Read for ChunkReader {
	fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
		while self.chunk.is_empty() && !self.ended {
			match self.receiver.blocking_recv() {
				Some(UploadMessage::Data(chunk)) => self.chunk = chunk,
				Some(UploadMessage::End) => self.ended = true,
				None => return Err(io::Error::new(io::ErrorKind::BrokenPipe, "upload cancelled")),
			}
		}
		if self.ended {
			return Ok(0);
		}
		let length = output.len().min(self.chunk.len());
		output[..length].copy_from_slice(&self.chunk[..length]);
		self.chunk.advance(length);
		Ok(length)
	}
}

fn parse_hash(hash: &[u8]) -> Result<[u8; 32], Status> {
	hash
		.try_into()
		.map_err(|_| Status::invalid_argument("BLAKE3 hash must be exactly 32 bytes"))
}

fn map_open_error(error: io::Error) -> Status {
	if error.kind() == io::ErrorKind::NotFound {
		Status::not_found("blob not found")
	} else {
		Status::internal("blob is unavailable")
	}
}
