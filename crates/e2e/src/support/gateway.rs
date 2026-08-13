#![cfg(unix)]

use std::{path::{Path, PathBuf}, sync::Arc, task::{Context, Poll}, time::Duration};

use anyhow::{Context as _, Result};
use omp_agent::RpcTurnClient;
use omp_app::{daemon::{DaemonConfig, DaemonError, DaemonHandle}, endpoint::LocalEndpoint};
use omp_core::Str;
use omp_llm_catalog::{CompiledCatalog, ManagementCapabilities, OperationBits, OperationKind, snapshot::{Catalog, SnapshotProvenance}};
use omp_llm_inference::{
	Answer, Error as InferenceError, Registry,
	call::Call,
	event::WorkflowResponse,
	layer::{LayerCall, stack::RouteProviderService},
	provider::fake::{CapturedCall, FakeProvider, FakeScript},
	receipt::ReasonId,
	registry::RouteUnavailable,
	session::ConversationSessionPlanner,
};
use omp_tool::Registry as ToolRegistry;
use tokio::task::JoinHandle;
use tower::Service;

use super::{DEFAULT_TIMEOUT, Scratch, within};

#[derive(Clone)]
struct FakeRoute(FakeProvider);

impl Service<LayerCall<Call>> for FakeRoute {
	type Error = InferenceError;
	type Future = <FakeProvider as Service<Call>>::Future;
	type Response = Answer;

	fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		<FakeProvider as Service<Call>>::poll_ready(&mut self.0, context)
	}

	fn call(&mut self, request: LayerCall<Call>) -> Self::Future {
		<FakeProvider as Service<Call>>::call(&mut self.0, request.payload)
	}
}

/// Production inference RPC/context/session authority backed by a deterministic provider.
pub struct ScriptedGateway {
	socket: PathBuf,
	data_dir: PathBuf,
	session_db: PathBuf,
	model: Str,
	registry: Registry,
	tools: Arc<ToolRegistry>,
	provider: FakeProvider,
	live_responses: flume::Sender<WorkflowResponse>,
	responses: flume::Receiver<WorkflowResponse>,
	shutdown: Option<flume::Sender<()>>,
	actor: Option<JoinHandle<Result<(), DaemonError>>>,
}

impl ScriptedGateway {
	/// Starts the real daemon RPC adapter and persistent session store over a fake provider route.
	pub async fn spawn(
		scratch: &Scratch,
		scripts: impl IntoIterator<Item = FakeScript>,
		tools: Arc<ToolRegistry>,
	) -> Result<Self> {
		let socket = scratch.socket("gateway.sock");
		let data_dir = scratch.state().join("gateway");
		let session_db = scratch.state().join("gateway-sessions.db");
		let (registry, provider, model, catalog) = scripted_registry(scripts)?;
		let sessions = ConversationSessionPlanner::open(&session_db, catalog)
			.context("opening persistent gateway sessions")?;
		let (live_responses, responses) = flume::bounded(64);
		let (shutdown, actor) = start_daemon(
			&socket,
			&data_dir,
			registry.clone(),
			sessions,
			Arc::clone(&tools),
			live_responses.clone(),
		).await?;
		Ok(Self {
			socket,
			data_dir,
			session_db,
			model,
			registry,
			tools,
			provider,
			live_responses,
			responses,
			shutdown: Some(shutdown),
			actor: Some(actor),
		})
	}

	/// Returns the model key routed to the scripted provider.
	#[must_use]
	pub fn model(&self) -> &str {
		self.model.as_str()
	}

	/// Returns the owner-local gateway endpoint accepted by `omp chat --gateway`.
	#[must_use]
	pub fn endpoint(&self) -> &Path {
		&self.socket
	}

	/// Connects a real turn-protocol client to the gateway.
	pub async fn client(&self) -> Result<RpcTurnClient> {
		let channel = within("gateway connection", DEFAULT_TIMEOUT, omp_rpc::uds::connect(&self.socket))
			.await??;
		Ok(RpcTurnClient::new(channel))
	}

	/// Appends deterministic provider interactions without replacing the real gateway authority.
	pub fn push(&self, script: FakeScript) {
		self.provider.push(script);
	}

	/// Returns secret-safe canonical calls observed by the provider seam.
	#[must_use]
	pub fn calls(&self) -> Vec<CapturedCall> {
		self.provider.calls()
	}

	/// Receives one client-to-provider duplex response within `limit`.
	pub async fn next_workflow_response(&self, limit: Duration) -> Result<WorkflowResponse> {
		let response = within("workflow response", limit, self.responses.recv_async()).await??;
		Ok(response)
	}

	/// Restarts the real RPC/context authority over the same SQLite conversation store.
	pub async fn restart(&mut self) -> Result<()> {
		self.stop_actor().await?;
		let catalog = Arc::new(self.registry.catalog().clone());
		let sessions = ConversationSessionPlanner::open(&self.session_db, catalog)
			.context("reopening persistent gateway sessions")?;
		let (shutdown, actor) = start_daemon(
			&self.socket,
			&self.data_dir,
			self.registry.clone(),
			sessions,
			Arc::clone(&self.tools),
			self.live_responses.clone(),
		).await?;
		self.shutdown = Some(shutdown);
		self.actor = Some(actor);
		Ok(())
	}

	/// Stops the real gateway and removes its endpoint.
	pub async fn shutdown(mut self) -> Result<()> {
		self.stop_actor().await
	}

	async fn stop_actor(&mut self) -> Result<()> {
		if let Some(shutdown) = self.shutdown.take() {
			let _ = shutdown.send_async(()).await;
		}
		if let Some(actor) = self.actor.take() {
			within("gateway shutdown", DEFAULT_TIMEOUT, actor).await??
				.context("gateway stopped with an error")?;
		}
		Ok(())
	}
}

impl Drop for ScriptedGateway {
	fn drop(&mut self) {
		if let Some(shutdown) = self.shutdown.take() {
			let _ = shutdown.send(());
		}
	}
}

async fn start_daemon(
	socket: &Path,
	data_dir: &Path,
	registry: Registry,
	sessions: ConversationSessionPlanner,
	tools: Arc<ToolRegistry>,
	live_responses: flume::Sender<WorkflowResponse>,
) -> Result<(flume::Sender<()>, JoinHandle<Result<(), DaemonError>>)> {
	let daemon = DaemonHandle::start_for_test(
		DaemonConfig::local(LocalEndpoint::from(socket.to_owned())).with_data_dir(data_dir.to_owned()),
		registry,
		sessions,
		tools,
		live_responses,
	).await.context("starting scripted production gateway")?;
	let (shutdown, request) = flume::bounded(1);
	let actor = tokio::spawn(async move {
		let _ = request.recv_async().await;
		daemon.shutdown().await
	});
	Ok((shutdown, actor))
}

fn scripted_registry(
	scripts: impl IntoIterator<Item = FakeScript>,
) -> Result<(Registry, FakeProvider, Str, Arc<Catalog>)> {
	let mut compiled: CompiledCatalog = serde_json::from_str(include_str!(
		"../../../llm-catalog/data/catalog.normalized.json"
	)).context("decoding normalized test catalog")?;
	for provider in &mut compiled.providers {
		provider.management = ManagementCapabilities {
			operations: OperationBits::empty(),
			multiple_accounts: false,
			refresh: false,
			principal_quota: false,
		};
	}
	let artifacts = Catalog::encode(compiled, SnapshotProvenance { source_digest: [0; 32] })
		.context("encoding deterministic test catalog")?;
	let catalog = Arc::new(Catalog::decode(&artifacts.postcard).context("decoding test catalog")?);
	let model = catalog.models().iter().find(|model| {
		model.capabilities.operations.contains_kind(OperationKind::Chat)
	}).context("catalog has no chat model")?;
	let route_id = model.routes.first().context("chat model has no route")?.clone();
	let route = catalog.route(&route_id).context("chat route is absent")?;
	let provider = FakeProvider::new(route.provider.clone(), route_id.clone());
	provider.extend(scripts);
	let service = RouteProviderService::new(FakeRoute(provider.clone()));
	let mut builder = Registry::builder(Arc::clone(&catalog));
	for candidate in catalog.routes() {
		builder = if candidate.id == route_id {
			builder.register_route(candidate.id.clone(), service.clone())?
		} else {
			builder.register_unavailable(RouteUnavailable {
				route: candidate.id.clone(),
				reason: ReasonId(Str::from("e2e-route-unavailable")),
				operation: None,
			})?
		};
	}
	let model = Str::from(model.key.as_str());
	Ok((builder.build()?, provider, model, catalog))
}

