use std::{
	sync::Arc,
	thread::{self, JoinHandle},
};

use parking_lot::Mutex;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::{Error, Result};

type Job<T> = Box<dyn FnOnce(&mut T) + Send + 'static>;

enum Message<T> {
	Run(Job<T>),
	Shutdown,
}

struct Inner<T> {
	tx:   flume::Sender<Message<T>>,
	join: Mutex<Option<JoinHandle<()>>>,
}

impl<T> Drop for Inner<T> {
	fn drop(&mut self) {
		let _ = self.tx.try_send(Message::Shutdown);
	}
}

pub struct Worker<T> {
	inner: Arc<Inner<T>>,
}

impl<T> Clone for Worker<T> {
	fn clone(&self) -> Self {
		Self { inner: Arc::clone(&self.inner) }
	}
}

impl<T: 'static> Worker<T> {
	pub(crate) async fn spawn(
		name: &'static str,
		initialize: impl FnOnce() -> Result<T> + Send + 'static,
	) -> Result<Self> {
		let (tx, rx) = flume::unbounded();
		let (ready_tx, ready_rx) = oneshot::channel();
		let join = thread::Builder::new().name(name.into()).spawn(move || {
			let mut state = match initialize() {
				Ok(state) => {
					let _ = ready_tx.send(Ok(()));
					state
				},
				Err(error) => {
					let _ = ready_tx.send(Err(error));
					return;
				},
			};
			while let Ok(message) = rx.recv() {
				match message {
					Message::Run(job) => job(&mut state),
					Message::Shutdown => break,
				}
			}
		})?;

		ready_rx.await.map_err(|_| Error::WorkerStopped)??;
		Ok(Self { inner: Arc::new(Inner { tx, join: Mutex::new(Some(join)) }) })
	}

	pub(crate) async fn run<R>(
		&self,
		cancel: CancellationToken,
		operation: impl FnOnce(&mut T, &CancellationToken) -> Result<R> + Send + 'static,
	) -> Result<R>
	where
		R: Send + 'static,
	{
		if cancel.is_cancelled() {
			return Err(Error::Cancelled);
		}
		let (result_tx, result_rx) = oneshot::channel();
		let worker_cancel = cancel.clone();
		self
			.inner
			.tx
			.send_async(Message::Run(Box::new(move |state| {
				let result = if worker_cancel.is_cancelled() {
					Err(Error::Cancelled)
				} else {
					operation(state, &worker_cancel)
				};
				let _ = result_tx.send(result);
			})))
			.await
			.map_err(|_| Error::WorkerStopped)?;

		tokio::select! {
			result = result_rx => result.map_err(|_| Error::WorkerStopped)?,
			() = cancel.cancelled() => Err(Error::Cancelled),
		}
	}

	pub(crate) async fn run_uncancelled<R>(
		&self,
		operation: impl FnOnce(&mut T) -> Result<R> + Send + 'static,
	) -> Result<R>
	where
		R: Send + 'static,
	{
		self
			.run(CancellationToken::new(), move |state, _| operation(state))
			.await
	}

	pub(crate) fn dispatch(&self, operation: impl FnOnce(&mut T) + Send + 'static) -> Result<()> {
		self
			.inner
			.tx
			.send(Message::Run(Box::new(operation)))
			.map_err(|_| Error::WorkerStopped)
	}

	pub(crate) async fn shutdown(&self) -> Result<()> {
		let Some(join) = self.inner.join.lock().take() else {
			return Ok(());
		};
		let _ = self.inner.tx.send_async(Message::Shutdown).await;
		tokio::task::spawn_blocking(move || join.join())
			.await
			.map_err(|error| Error::backend("worker", error))?
			.map_err(|_| Error::backend("worker", "worker thread panicked"))?;
		Ok(())
	}
}
