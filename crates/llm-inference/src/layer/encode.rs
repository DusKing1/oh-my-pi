//! Pure canonical-to-wire encoding after rate reservation and before
//! credentials.

use std::{
	task::{Context, Poll},
	time::SystemTime,
};

use omp_core::Str;
use tower::{Layer, Service};

use crate::{
	auth::{AuthSpec, CredentialLease},
	codec::{Cancellation, TransportRequest},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	layer::{ExecutionContext, LayerCall, auth::Authorized},
	receipt::ReasonId,
};

/// Pure construction-time codec binding for a planned request.
pub trait AttemptEncoder<R>: Clone + Send + 'static {
	/// Encodes with a fresh body source and decoder; it must not acquire
	/// credentials or perform I/O.
	fn encode(
		&self,
		request: &R,
		execution: &crate::layer::ExecutionContext,
		attempt: u32,
		provisional: bool,
		cancel: Cancellation,
	) -> Result<TransportRequest, Error>;
}

/// Fully encoded transport request paired with the still-opaque credential
/// lease.
pub struct EncodedAttempt<A, L> {
	/// Non-secret selected account metadata.
	pub account:      A,
	/// Secret-free encoded transport request.
	pub transport:    TransportRequest,
	/// Opaque lease consumed only by credential application.
	pub(crate) lease: L,
}

/// Adds pure codec lowering.
#[derive(Clone, Debug)]
pub struct EncodeLayer<E> {
	encoder:     E,
	provisional: bool,
}
impl<E> EncodeLayer<E> {
	/// Creates an encoding layer for visible or transactionally provisional
	/// attempts.
	pub const fn new(encoder: E, provisional: bool) -> Self {
		Self { encoder, provisional }
	}
}
/// Encoding service.
#[derive(Clone, Debug)]
pub struct EncodeService<S, E> {
	inner:       S,
	encoder:     E,
	provisional: bool,
}
impl<S, E: Clone> Layer<S> for EncodeLayer<E> {
	type Service = EncodeService<S, E>;

	fn layer(&self, inner: S) -> Self::Service {
		EncodeService { inner, encoder: self.encoder.clone(), provisional: self.provisional }
	}
}
impl<S, E, R, A, L> Service<LayerCall<Authorized<R, A, L>>> for EncodeService<S, E>
where
	E: AttemptEncoder<R>,
	S: Service<LayerCall<EncodedAttempt<A, L>>, Error = Error>,
{
	type Error = Error;
	type Response = S::Response;

	type Future = impl Future<Output = Result<S::Response, Error>>;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, request: LayerCall<Authorized<R, A, L>>) -> Self::Future {
		let Authorized { request: planned, account, lease } = request.payload;
		let attempt = request.context.attempts().saturating_sub(1);
		let cancel = Cancellation::default();
		request.context.register_transport_cancel(cancel.clone());
		let encoded = request
			.context
			.checkpoint(ErrorPhase::Encoding)
			.and_then(|()| {
				self
					.encoder
					.encode(&planned, &request.context, attempt, self.provisional, cancel)
			});
		let next = encoded.map(|transport| LayerCall {
			payload: EncodedAttempt { account, transport, lease },
			context: request.context,
		});
		let future = next.map(|request| self.inner.call(request));
		async move { future?.await }
	}
}

/// Applies an opaque lease to a fully encoded request without exposing secret
/// bytes.
pub trait CredentialApplier<A, L>: Clone + Send + 'static {
	/// Applies headers/query/signing metadata or returns a typed authentication
	/// error.
	fn apply(
		&self,
		account: &A,
		lease: L,
		request: &mut TransportRequest,
		context: &ExecutionContext,
	) -> Result<(), Error>;
}

/// Attaches an auth-owned opaque credential envelope without materializing
/// secrets.
#[derive(Clone, Copy, Debug, Default)]
pub struct AttachCredentials;
impl<A> CredentialApplier<A, crate::auth::lease::AppliedCredentials> for AttachCredentials {
	fn apply(
		&self,
		_: &A,
		credentials: crate::auth::lease::AppliedCredentials,
		request: &mut TransportRequest,
		_: &ExecutionContext,
	) -> Result<(), Error> {
		request.credentials = Some(credentials);
		Ok(())
	}
}

/// Prepares a raw opaque lease against one route's exact authentication spec.
#[derive(Clone, Debug)]
pub struct PrepareCredentials {
	spec: AuthSpec,
}

impl PrepareCredentials {
	/// Creates a route-scoped credential adapter.
	#[must_use]
	pub const fn new(spec: AuthSpec) -> Self {
		Self { spec }
	}
}

impl<A> CredentialApplier<A, CredentialLease> for PrepareCredentials {
	fn apply(
		&self,
		_: &A,
		lease: CredentialLease,
		request: &mut TransportRequest,
		context: &ExecutionContext,
	) -> Result<(), Error> {
		let credentials = lease
			.prepare(&self.spec, SystemTime::now())
			.map_err(|_| credential_prepare_error(context))?;
		request.credentials = Some(credentials);
		Ok(())
	}
}

fn credential_prepare_error(context: &ExecutionContext) -> Error {
	let mut error = Error::new(
		ErrorKind::Authentication,
		ErrorPhase::Authentication,
		RetryAction::Never,
		context.receipt(),
	);
	error.detail = Some(ErrorDetail::Protocol {
		reason: ReasonId(Str::new_static("credential-application-contract")),
	});
	error
}

/// Adds credential application at the last boundary before wire transport.
#[derive(Clone, Debug)]
pub struct CredentialApplyLayer<P> {
	applier: P,
}
impl<P> CredentialApplyLayer<P> {
	/// Creates a credential application layer.
	pub const fn new(applier: P) -> Self {
		Self { applier }
	}
}
/// Credential-finalizing service.
#[derive(Clone, Debug)]
pub struct CredentialApplyService<S, P> {
	inner:   S,
	applier: P,
}
impl<S, P: Clone> Layer<S> for CredentialApplyLayer<P> {
	type Service = CredentialApplyService<S, P>;

	fn layer(&self, inner: S) -> Self::Service {
		CredentialApplyService { inner, applier: self.applier.clone() }
	}
}
impl<S, P, A, L> Service<LayerCall<EncodedAttempt<A, L>>> for CredentialApplyService<S, P>
where
	P: CredentialApplier<A, L>,
	S: Service<TransportRequest, Error = Error>,
{
	type Error = Error;
	type Response = S::Response;

	type Future = impl Future<Output = Result<S::Response, Error>>;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, request: LayerCall<EncodedAttempt<A, L>>) -> Self::Future {
		let EncodedAttempt { account, mut transport, lease } = request.payload;
		let applied = request
			.context
			.checkpoint(ErrorPhase::Authentication)
			.and_then(|()| {
				self
					.applier
					.apply(&account, lease, &mut transport, &request.context)
			});
		let future = applied.map(|()| self.inner.call(transport));
		async move { future?.await }
	}
}

#[cfg(test)]
mod tests {
	use std::{
		sync::Arc,
		task::{Context, Poll},
	};

	use bytes::Bytes;
	use futures::future::{Ready, ready};
	use parking_lot::Mutex;
	use tower::Service;

	use super::{AttemptEncoder, CredentialApplier, CredentialApplyService, EncodeService};
	use crate::{
		body::BodySource,
		codec::{
			Cancellation, Decoder, EncodedRequest, RawEvent, RequestMethod, SizeBounds,
			TransportAttempt, TransportRequest,
		},
		error::Error,
		layer::{ExecutionContext, LayerCall, auth::Authorized},
		receipt::ExecutionBudget,
		transport::{Frame, FramingProtocol},
	};

	struct EmptyDecoder;
	impl Decoder for EmptyDecoder {
		fn push(&mut self, _: Frame, _: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
			Ok(())
		}

		fn finish(&mut self, _: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
			Ok(())
		}
	}
	fn transport() -> TransportRequest {
		TransportRequest {
			encoded:     EncodedRequest {
				operation:   omp_llm_catalog::OperationKind::Chat,
				method:      RequestMethod::Post,
				uri:         "https://example.invalid".into(),
				headers:     Box::new([]),
				body:        BodySource::Bytes(Bytes::new()),
				framing:     FramingProtocol::Raw,
				bounds:      SizeBounds { request_body: 1, frame: 1, response: 1 },
				sealed_body: None,
			},
			credentials: None,
			decoder:     Some(Box::new(EmptyDecoder)),
			realtime:    None,
			cancel:      Cancellation::default(),
			attempt:     TransportAttempt {
				request_id:    crate::id::RequestId::from("request"),
				provider:      omp_llm_catalog::ProviderId::from("provider"),
				route:         omp_llm_catalog::RouteId::from("route"),
				account:       None,
				principal:     None,
				index:         0,
				provisional:   false,
				capture_limit: 0,
				timeout:       std::time::Duration::from_secs(1),
			},
		}
	}
	#[derive(Clone)]
	struct Encoder(Arc<Mutex<Vec<&'static str>>>);
	impl AttemptEncoder<()> for Encoder {
		fn encode(
			&self,
			_: &(),
			_: &ExecutionContext,
			_: u32,
			_: bool,
			_: Cancellation,
		) -> Result<TransportRequest, Error> {
			self.0.lock().push("encode");
			Ok(transport())
		}
	}
	#[derive(Clone)]
	struct Applier(Arc<Mutex<Vec<&'static str>>>);
	impl CredentialApplier<(), u8> for Applier {
		fn apply(
			&self,
			_: &(),
			_: u8,
			_: &mut TransportRequest,
			_: &ExecutionContext,
		) -> Result<(), Error> {
			self.0.lock().push("credential");
			Ok(())
		}
	}
	#[derive(Clone)]
	struct Wire(Arc<Mutex<Vec<&'static str>>>);
	impl Service<TransportRequest> for Wire {
		type Error = Error;
		type Future = Ready<Result<(), Error>>;
		type Response = ();

		fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Error>> {
			Poll::Ready(Ok(()))
		}

		fn call(&mut self, _: TransportRequest) -> Self::Future {
			self.0.lock().push("wire");
			ready(Ok(()))
		}
	}
	#[tokio::test]
	async fn credentials_are_applied_only_after_encoding_and_immediately_before_transport() {
		let trace = Arc::new(Mutex::new(Vec::new()));
		let credential =
			CredentialApplyService { inner: Wire(trace.clone()), applier: Applier(trace.clone()) };
		let mut service = EncodeService {
			inner:       credential,
			encoder:     Encoder(trace.clone()),
			provisional: false,
		};
		futures::future::poll_fn(|cx| service.poll_ready(cx))
			.await
			.unwrap();
		service
			.call(LayerCall {
				payload: Authorized { request: (), account: (), lease: 7 },
				context: ExecutionContext::new(ExecutionBudget::default()),
			})
			.await
			.unwrap();
		assert_eq!(&*trace.lock(), &["encode", "credential", "wire"]);
	}
}
