//! RPC and in-memory tonic implementations of the turn seam.

use std::{future::Future, io, sync::Arc};

use flume::Sender;
use hyper_util::rt::TokioIo;
use omp_llm_inference::TurnId;
use omp_proto::inference::v1::{
	TurnEvent, TurnFrame,
	inference_client::InferenceClient,
	inference_server::{Inference, InferenceServer},
};
use parking_lot::Mutex;
use tokio::{io::DuplexStream, task::JoinHandle};
use tonic::transport::{Channel, Endpoint, Server};
use tower::service_fn;

use crate::turn::{
	Error, EventStream, InvokeFrame, TurnClient, TurnInput, TurnOptions, TurnSession,
	invocation_stream, open_frame,
};

const INVOCATION_FRAME_CAPACITY: usize = 32;
const INPROC_IO_CAPACITY: usize = 64 * 1024;

/// Turn client for an existing tonic channel.
///
/// The channel may be backed by an owner-only UDS or a mutually authenticated
/// TLS connection configured by `omp-rpc`; this type adds no transport policy.
#[derive(Clone, Debug)]
pub struct RpcTurnClient {
	client: InferenceClient<Channel>,
}

impl RpcTurnClient {
	/// Wraps an established tonic channel.
	#[inline]
	pub fn new(channel: Channel) -> Self {
		Self { client: InferenceClient::new(channel) }
	}

	/// Wraps a preconfigured generated inference client.
	#[inline]
	pub fn from_client(client: InferenceClient<Channel>) -> Self {
		Self { client }
	}
}

impl TurnClient for RpcTurnClient {
	type Session<'client> = RpcTurnSession;

	fn turn<'client>(
		&'client self,
		turn_id: TurnId,
		input: TurnInput,
		options: &'client TurnOptions,
	) -> impl Future<Output = Result<Self::Session<'client>, Error>> + Send + 'client {
		async move {
			let open = open_frame(&turn_id, input, options)?;
			let (sender, receiver) = flume::bounded(INVOCATION_FRAME_CAPACITY);
			sender.send_async(open).await.map_err(|_| Error::Closed)?;

			let mut client = self.client.clone();
			let stream = client.turn(invocation_stream(receiver)).await?.into_inner();
			Ok(RpcTurnSession {
				sender: options.executor.is_some().then_some(sender),
				stream,
				terminal: false,
				_server: None,
			})
		}
	}
}

/// One live tonic turn stream.
pub struct RpcTurnSession {
	sender:   Option<Sender<TurnFrame>>,
	stream:   tonic::Streaming<TurnEvent>,
	terminal: bool,
	_server:  Option<Arc<ServerTask>>,
}

impl TurnSession for RpcTurnSession {
	fn events(
		&mut self,
	) -> impl futures::Stream<Item = Result<TurnEvent, Error>> + Send + Unpin + '_ {
		EventStream::new(&mut self.stream, &mut self.terminal)
	}

	fn submit(&mut self, frame: InvokeFrame) -> impl Future<Output = Result<(), Error>> + Send + '_ {
		async move {
			if self.terminal {
				return Err(Error::Closed);
			}
			let sender = self
				.sender
				.as_ref()
				.ok_or(Error::Invalid("turn did not declare an in-turn executor"))?;
			sender
				.send_async(TurnFrame::from(frame))
				.await
				.map_err(|_| Error::Closed)
		}
	}
}

/// Turn client that serves the generated tonic [`Inference`] implementation
/// over a process-local duplex stream.
///
/// This intentionally runs the supplied production service itself instead of
/// reimplementing its context, commit, invocation, or replay behavior. The only
/// difference from [`RpcTurnClient`] is that bytes remain in memory.
#[derive(Clone)]
pub struct InProcTurnClient {
	client: RpcTurnClient,
	server: Arc<ServerTask>,
}

impl InProcTurnClient {
	/// Injects one generated inference service into an in-memory tonic
	/// transport.
	pub async fn new<S>(service: S) -> Result<Self, Error>
	where
		S: Inference,
	{
		let (client_io, server_io) = tokio::io::duplex(INPROC_IO_CAPACITY);
		let incoming = tokio_stream::once(Ok::<DuplexStream, io::Error>(server_io));
		let server = tokio::spawn(
			Server::builder()
				.add_service(InferenceServer::new(service))
				.serve_with_incoming(incoming),
		);

		let slot = Arc::new(Mutex::new(Some(client_io)));
		let connector = service_fn(move |_| {
			let stream = slot.lock().take();
			async move {
				stream.map(TokioIo::new).ok_or_else(|| {
					io::Error::new(io::ErrorKind::NotConnected, "in-process channel already connected")
				})
			}
		});
		let channel = match Endpoint::from_static("http://omp.in-process")
			.connect_with_connector(connector)
			.await
		{
			Ok(channel) => channel,
			Err(error) => {
				server.abort();
				return Err(Error::Connect(error));
			},
		};
		Ok(Self {
			client: RpcTurnClient::new(channel),
			server: Arc::new(ServerTask { handle: server }),
		})
	}
}

struct ServerTask {
	handle: JoinHandle<Result<(), tonic::transport::Error>>,
}

impl Drop for ServerTask {
	fn drop(&mut self) {
		self.handle.abort();
	}
}

impl TurnClient for InProcTurnClient {
	type Session<'client> = RpcTurnSession;

	fn turn<'client>(
		&'client self,
		turn_id: TurnId,
		input: TurnInput,
		options: &'client TurnOptions,
	) -> impl Future<Output = Result<Self::Session<'client>, Error>> + Send + 'client {
		async move {
			let mut session = self.client.turn(turn_id, input, options).await?;
			session._server = Some(Arc::clone(&self.server));
			Ok(session)
		}
	}
}
