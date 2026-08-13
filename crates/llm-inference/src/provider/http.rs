//! Data-defined HTTP provider service composition.

use std::task::{Context, Poll};

use tower::Service;

use crate::{
	answer::Answer,
	call::Call,
	catalog::{RouteDef, TransportKind},
	error::{Error, ErrorDetail, ErrorKind},
	layer::{LayerCall, stack::RouteProviderService},
	receipt::{ExecutionReceipt, ReasonId},
};

/// A route-definition-bound HTTP service that delegates to one preconstructed
/// stack.
#[derive(Clone)]
pub(crate) struct HttpRouteService {
	route: RouteDef,
	inner: RouteProviderService,
}

impl HttpRouteService {
	/// Binds a constructed codec/transport stack to its immutable route
	/// definition.
	pub(crate) fn new(route: RouteDef, inner: RouteProviderService) -> Result<Self, Error> {
		if !matches!(
			route.transport,
			TransportKind::Http | TransportKind::AwsEventStream | TransportKind::Connect
		) {
			return Err(Error::planning(
				ErrorKind::RouteUnavailable,
				ErrorDetail::Capability {
					feature: "http-route-transport".into(),
					reason:  ReasonId("route-is-not-http-borne".into()),
				},
				ExecutionReceipt::default(),
			));
		}
		Ok(Self { route, inner })
	}

	/// Borrows the declarative route bound to this service.
	pub(crate) fn route(&self) -> &RouteDef {
		&self.route
	}

	/// Erases the already-built route stack once at registry construction.
	pub(crate) fn boxed(self) -> RouteProviderService {
		RouteProviderService::new(self)
	}
}

impl Service<LayerCall<Call>> for HttpRouteService {
	type Error = Error;
	type Future = <RouteProviderService as Service<LayerCall<Call>>>::Future;
	type Response = Answer;

	fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(context)
	}

	fn call(&mut self, call: LayerCall<Call>) -> Self::Future {
		self.inner.call(call)
	}
}
