//! Real-socket proof of the native content-addressed blob transfer service.

use std::{sync::Arc, time::Duration};

use bytes::Bytes;
use omp_llm_gateway::blob::BlobService;
use omp_proto::blob::v1::{
	Chunk, DeleteRequest, GetRequest, StatRequest, blob_client::BlobClient, blob_server::BlobServer,
};
use omp_storage::blob::{BlobRef, BlobStore};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{
	Code, Request,
	transport::{Channel, Endpoint, Server},
};

const MAX_UPLOAD: u64 = 512 * 1024;

#[tokio::test]
async fn remote_chunked_blob_lifecycle_ranges_limits_and_cancellation() {
	let directory = tempfile::tempdir().expect("temporary blob directory");
	let store = Arc::new(BlobStore::open(directory.path()).expect("blob store"));
	let listener = TcpListener::bind("127.0.0.1:0")
		.await
		.expect("bind TCP listener");
	let address = listener.local_addr().expect("listener address");
	let server_store = Arc::clone(&store);
	let server = tokio::spawn(async move {
		Server::builder()
			.add_service(BlobServer::new(
				BlobService::new(server_store).with_max_upload_size(MAX_UPLOAD),
			))
			.serve_with_incoming(TcpListenerStream::new(listener))
			.await
			.expect("serve blob service");
	});
	let channel = Endpoint::from_shared(format!("http://{address}"))
		.expect("endpoint")
		.connect()
		.await
		.expect("connect blob client");
	let cancellation_channel = channel.clone();
	let mut client = BlobClient::new(channel);

	let payload = (0..160 * 1024)
		.map(|index| (index % 251) as u8)
		.collect::<Vec<_>>();
	let expected_hash = *blake3::hash(&payload).as_bytes();
	let payload_size = payload.len() as u64;
	let chunks = payload
		.chunks(19 * 1024)
		.enumerate()
		.map(move |(index, data)| Chunk {
			data: Bytes::copy_from_slice(data),
			hash: if index == 0 {
				Bytes::copy_from_slice(&expected_hash)
			} else {
				Bytes::new()
			},
			size: (index == 0).then_some(payload_size),
		})
		.collect::<Vec<_>>();
	let uploaded = client
		.put(tokio_stream::iter(chunks))
		.await
		.expect("chunked upload")
		.into_inner();
	assert_eq!(&uploaded.hash[..], &expected_hash);
	assert_eq!(uploaded.size, payload.len() as u64);

	let stat = client
		.stat(StatRequest { hash: uploaded.hash.clone() })
		.await
		.expect("stat")
		.into_inner();
	assert!(stat.present);
	assert_eq!(stat.size, payload.len() as u64);

	let full = fetch(&mut client, uploaded.hash.clone(), 0, 0).await;
	assert_eq!(full, payload);
	let ranged = fetch(&mut client, uploaded.hash.clone(), 997, 12_345).await;
	assert_eq!(ranged, payload[997..997 + 12_345]);

	// This is the same store passed to media/facade state by the daemon. A
	// hash-only media reference therefore resolves without copying inline data.
	let reference = BlobRef { hash: expected_hash, size: payload.len() as u64 };
	assert_eq!(store.get(&reference).expect("media hash reference"), payload.as_slice());

	let invalid = client
		.stat(StatRequest { hash: Bytes::from_static(b"not-a-digest") })
		.await
		.expect_err("invalid hash must fail");
	assert_eq!(invalid.code(), Code::InvalidArgument);

	let oversize = vec![0_u8; MAX_UPLOAD as usize + 1];
	let error = client
		.put(tokio_stream::iter([Chunk {
			data: Bytes::from(oversize),
			hash: Bytes::new(),
			size: None,
		}]))
		.await
		.expect_err("oversize upload must fail");
	assert_eq!(error.code(), Code::ResourceExhausted);

	let (sender, receiver) = flume::bounded(1);
	let upload_stream = futures::stream::unfold(receiver, |receiver| async move {
		receiver
			.recv_async()
			.await
			.ok()
			.map(|chunk| (chunk, receiver))
	});
	let cancellation_client = BlobClient::new(cancellation_channel);
	let upload = tokio::spawn(async move {
		let mut cancellation_client = cancellation_client;
		cancellation_client.put(Request::new(upload_stream)).await
	});
	sender
		.send_async(Chunk {
			data: Bytes::from(vec![7_u8; 128 * 1024]),
			hash: Bytes::new(),
			size: None,
		})
		.await
		.expect("send partial upload");
	tokio::time::sleep(Duration::from_millis(20)).await;
	upload.abort();
	let _ = upload.await;
	drop(sender);
	for _ in 0..20 {
		if std::fs::read_dir(directory.path().join("tmp"))
			.expect("temporary directory")
			.next()
			.is_none()
		{
			break;
		}
		tokio::time::sleep(Duration::from_millis(10)).await;
	}
	assert!(
		std::fs::read_dir(directory.path().join("tmp"))
			.expect("temporary directory")
			.next()
			.is_none()
	);

	let mut client = BlobClient::new(connect(address).await);
	let deleted = client
		.delete(DeleteRequest { hash: Bytes::copy_from_slice(&expected_hash) })
		.await
		.expect("delete")
		.into_inner();
	assert!(deleted.deleted);
	let absent = client
		.stat(StatRequest { hash: Bytes::copy_from_slice(&expected_hash) })
		.await
		.expect("stat deleted blob")
		.into_inner();
	assert!(!absent.present);

	server.abort();
}

async fn connect(address: std::net::SocketAddr) -> Channel {
	Endpoint::from_shared(format!("http://{address}"))
		.expect("endpoint")
		.connect()
		.await
		.expect("connect blob client")
}

async fn fetch(client: &mut BlobClient<Channel>, hash: Bytes, offset: u64, length: u64) -> Vec<u8> {
	let mut stream = client
		.get(GetRequest { hash, offset, length })
		.await
		.expect("get blob")
		.into_inner();
	let mut output = Vec::new();
	let mut first = true;
	while let Some(chunk) = stream.message().await.expect("read blob chunk") {
		if first {
			assert!(chunk.size.is_some());
			assert_eq!(chunk.hash.len(), 32);
			first = false;
		}
		output.extend_from_slice(&chunk.data);
	}
	output
}
