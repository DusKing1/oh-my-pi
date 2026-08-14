use std::{
	collections::{BTreeMap, BTreeSet},
	ffi::CString,
	fmt,
	sync::{
		Arc, OnceLock,
		atomic::{AtomicU64, Ordering},
	},
};

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt as _;
use omp_core::Str;
use omp_tool::{
	ErasedEv, ErasedOutcome, IncomingParams, Part, PromptCaps, Registry, ToolIdentity, ToolRoute,
};
use omp_tools::eval::{idle_timeout::TimeoutHandle, kernel::NamespaceInstaller};
use parking_lot::Mutex;
use pyo3::{
	exceptions::PyRuntimeError,
	prelude::*,
	types::{PyAny, PyDict, PyModule},
};
use secrecy::{ExposeSecret as _, SecretString};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::runtime::Handle;
use tokio_util::sync::CancellationToken;
use ulid::Ulid;

use super::PYTHON_PRELUDE;

const COMPLETION: &str = "__completion__";
const AGENT: &str = "__agent__";
const CONCURRENCY: &str = "__concurrency__";
const BUDGET: &str = "__budget__";

/// Capabilities granted to one eval cell. Tool names are explicit: possessing a
/// bridge grant never implies access to every tool registered in the session.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct BridgeCapabilities {
	tools:       BTreeSet<Str>,
	completion:  bool,
	agent:       bool,
	concurrency: bool,
	budget:      bool,
}

impl BridgeCapabilities {
	pub(crate) fn new(tools: impl IntoIterator<Item = Str>) -> Self {
		Self { tools: tools.into_iter().collect(), ..Self::default() }
	}

	pub(crate) fn with_completion(mut self) -> Self {
		self.completion = true;
		self
	}

	pub(crate) fn with_agent(mut self) -> Self {
		self.agent = true;
		self
	}

	pub(crate) fn with_concurrency(mut self) -> Self {
		self.concurrency = true;
		self
	}

	pub(crate) fn with_budget(mut self) -> Self {
		self.budget = true;
		self
	}

	fn allows(&self, name: &str) -> bool {
		match name {
			COMPLETION => self.completion,
			AGENT => self.agent,
			CONCURRENCY => self.concurrency,
			BUDGET => self.budget,
			_ => self.tools.iter().any(|tool| tool.as_str() == name),
		}
	}
}

#[derive(Clone)]
pub(crate) struct BridgeGrant {
	session:    Str,
	run:        Str,
	token:      SecretString,
	generation: Ulid,
}

impl fmt::Debug for BridgeGrant {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("BridgeGrant")
			.field("session", &self.session)
			.field("run", &self.run)
			.field("token", &"[REDACTED]")
			.field("generation", &self.generation)
			.finish()
	}
}

#[derive(Debug, Error)]
pub(crate) enum BridgeHostError {
	#[error("{0}")]
	Message(Str),
}

impl BridgeHostError {
	pub(crate) fn message(message: impl Into<Str>) -> Self {
		Self::Message(message.into())
	}
}

/// Real host-side boundary used for ordinary tools and the privileged eval
/// completion/agent/budget operations. Implementations receive only calls that
/// passed grant authentication and capability checks.
#[async_trait]
pub(crate) trait BridgeHost: Send + Sync {
	async fn call(&self, name: &str, args: Value) -> Result<Value, BridgeHostError>;
}

struct Registration {
	grant:        BridgeGrant,
	capabilities: BridgeCapabilities,
	host:         Arc<dyn BridgeHost>,
	timeout:      TimeoutHandle,
}

struct DispatcherInner {
	registrations: Mutex<BTreeMap<(Str, Str), Registration>>,
}

#[derive(Clone)]
pub(crate) struct BridgeDispatcher {
	inner: Arc<DispatcherInner>,
}

impl BridgeDispatcher {
	pub(crate) fn new() -> Self {
		Self { inner: Arc::new(DispatcherInner { registrations: Mutex::new(BTreeMap::new()) }) }
	}

	pub(crate) fn register(
		&self,
		session: Str,
		run: Str,
		capabilities: BridgeCapabilities,
		host: Arc<dyn BridgeHost>,
		timeout: TimeoutHandle,
	) -> Result<BridgeRegistration, BridgeCallError> {
		if session.is_empty() || run.is_empty() {
			return Err(BridgeCallError::InvalidRegistration);
		}
		let key = (session.clone(), run.clone());
		let mut registrations = self.inner.registrations.lock();
		if registrations.contains_key(&key) {
			return Err(BridgeCallError::AlreadyRegistered { session, run });
		}
		let grant = BridgeGrant {
			session,
			run,
			token: SecretString::from(Ulid::generate().to_string()),
			generation: Ulid::generate(),
		};
		registrations.insert(key, Registration { grant: grant.clone(), capabilities, host, timeout });
		drop(registrations);
		Ok(BridgeRegistration {
			lease: Arc::new(RegistrationLease { dispatcher: self.clone(), grant }),
		})
	}

	async fn dispatch(
		&self,
		grant: &BridgeGrant,
		name: &str,
		args: Value,
	) -> Result<Value, BridgeCallError> {
		let (host, timeout) = {
			let registrations = self.inner.registrations.lock();
			let entry = registrations
				.get(&(grant.session.clone(), grant.run.clone()))
				.ok_or_else(|| BridgeCallError::NoActiveSession {
					session: grant.session.clone(),
					run:     grant.run.clone(),
				})?;
			if entry.grant.generation != grant.generation
				|| entry.grant.token.expose_secret() != grant.token.expose_secret()
			{
				return Err(BridgeCallError::AuthenticationFailed);
			}
			if !entry.capabilities.allows(name) {
				return Err(BridgeCallError::CapabilityDenied { name: Str::from(name) });
			}
			(Arc::clone(&entry.host), entry.timeout.clone())
		};

		timeout
			.host_wait(host.call(name, args))
			.await
			.map_err(|error| BridgeCallError::Host { message: Str::from(error.to_string()) })
	}

	fn unregister(&self, grant: &BridgeGrant) {
		let key = (grant.session.clone(), grant.run.clone());
		let mut registrations = self.inner.registrations.lock();
		if registrations
			.get(&key)
			.is_some_and(|entry| entry.grant.generation == grant.generation)
		{
			registrations.remove(&key);
		}
	}
}

struct RegistrationLease {
	dispatcher: BridgeDispatcher,
	grant:      BridgeGrant,
}

impl Drop for RegistrationLease {
	fn drop(&mut self) {
		self.dispatcher.unregister(&self.grant);
	}
}

pub(crate) struct BridgeRegistration {
	lease: Arc<RegistrationLease>,
}

impl BridgeRegistration {
	pub(crate) fn client(&self) -> BridgeClient {
		BridgeClient {
			dispatcher: self.lease.dispatcher.clone(),
			grant:      self.lease.grant.clone(),
			abort:      None,
			_lease:     Arc::clone(&self.lease),
		}
	}
}

#[derive(Default)]
struct CellAbort {
	active: Mutex<Option<(Bytes, CancellationToken)>>,
}

impl CellAbort {
	fn begin(&self, cell_id: &Bytes) {
		if let Some((_, stale)) = self
			.active
			.lock()
			.replace((cell_id.clone(), CancellationToken::new()))
		{
			stale.cancel();
		}
	}

	fn end(&self, cell_id: &Bytes) {
		let mut active = self.active.lock();
		if active
			.as_ref()
			.is_some_and(|(current, _)| current == cell_id)
			&& let Some((_, token)) = active.take()
		{
			token.cancel();
		}
	}

	fn cancel(&self, cell_id: &Bytes) {
		if let Some((_, token)) = self
			.active
			.lock()
			.as_ref()
			.filter(|(current, _)| current == cell_id)
		{
			token.cancel();
		}
	}

	fn token(&self) -> Option<CancellationToken> {
		self.active.lock().as_ref().map(|(_, token)| token.clone())
	}
}

#[derive(Clone)]
pub(crate) struct BridgeClient {
	dispatcher: BridgeDispatcher,
	grant:      BridgeGrant,
	abort:      Option<Arc<CellAbort>>,
	_lease:     Arc<RegistrationLease>,
}

impl BridgeClient {
	pub(crate) async fn call(&self, name: &str, args: Value) -> Result<Value, BridgeCallError> {
		let Some(abort) = &self.abort else {
			return self.dispatcher.dispatch(&self.grant, name, args).await;
		};
		let token = abort.token().ok_or(BridgeCallError::NoActiveCell)?;
		tokio::select! {
			result = self.dispatcher.dispatch(&self.grant, name, args) => result,
			() = token.cancelled() => Err(BridgeCallError::CellCancelled),
		}
	}

	fn with_abort(mut self, abort: Arc<CellAbort>) -> Self {
		self.abort = Some(abort);
		self
	}

	fn session(&self) -> &str {
		self.grant.session.as_str()
	}
}

#[derive(Debug, Error, Eq, PartialEq)]
pub(crate) enum BridgeCallError {
	#[error("eval bridge registration requires non-empty session and run ids")]
	InvalidRegistration,
	#[error("eval bridge session is already registered: {session}:{run}")]
	AlreadyRegistered { session: Str, run: Str },
	#[error("No active Python tool bridge session: {session}:{run}")]
	NoActiveSession { session: Str, run: Str },
	#[error("eval bridge authentication failed")]
	AuthenticationFailed,
	#[error("eval bridge call has no active cell")]
	NoActiveCell,
	#[error("eval bridge host call cancelled")]
	CellCancelled,
	#[error("bridge capability denied: {name}")]
	CapabilityDenied { name: Str },
	#[error("{message}")]
	Host { message: Str },
}

/// Adapter for the native tools in the exact environment registry. Privileged
/// names such as `__agent__` remain separate host capabilities and are never
/// silently translated into ordinary registry tools.
pub(crate) struct RegistryBridgeHost {
	registry: Arc<Registry>,
}

impl RegistryBridgeHost {
	pub(crate) fn new(registry: Arc<Registry>) -> Self {
		Self { registry }
	}
}

#[async_trait]
impl BridgeHost for RegistryBridgeHost {
	async fn call(&self, name: &str, mut args: Value) -> Result<Value, BridgeHostError> {
		let Some((live_name, revision)) = self.registry.live_identity(name) else {
			return Err(BridgeHostError::message(format!("Unknown tool from py runtime: {name}")));
		};
		if self.registry.route(name).map_err(registry_error)? != ToolRoute::Native {
			return Err(BridgeHostError::message(format!(
				"Tool from py runtime is not available through the native eval bridge: {name}"
			)));
		}
		let identity = ToolIdentity { name: live_name.clone(), rev: revision.clone() };
		if let Some(object) = args.as_object_mut() {
			object.remove("i");
		}
		let raw = serde_json::to_string(&args).map_err(|error| {
			BridgeHostError::message(format!("bridge arguments are not JSON: {error}"))
		})?;
		let (feed, params) = IncomingParams::channel();
		feed
			.arg_text(Str::from(raw.as_str()))
			.map_err(|error| BridgeHostError::message(error.to_string()))?;
		feed
			.args_committed(Str::from(raw))
			.map_err(|error| BridgeHostError::message(error.to_string()))?;
		let mut events = self.registry.invoke(name, params).map_err(registry_error)?;
		let mut updates = Vec::new();
		while let Some(event) = events.next().await {
			match event.map_err(registry_error)? {
				ErasedEv::Update(update) => {
					updates.push(serde_json::from_slice(&update).map_err(|error| {
						BridgeHostError::message(format!(
							"tool {name} returned invalid update JSON: {error}"
						))
					})?);
				},
				ErasedEv::Done(ErasedOutcome::Detached(job)) => {
					let value = serde_json::to_value(job)
						.map_err(|error| BridgeHostError::message(error.to_string()))?;
					return Ok(bridge_envelope(value, updates));
				},
				ErasedEv::Done(ErasedOutcome::Done { verdict, .. }) => {
					let projected = self
						.registry
						.project_verdict(&identity, &verdict, false, &PromptCaps {
							maximum_parts:      u16::MAX,
							maximum_text_bytes: u32::MAX,
							media:              true,
						})
						.map_err(registry_error)?;
					let mut value = projected_parts(projected.parts)?;
					if projected.is_error {
						match &mut value {
							Value::Object(object) => {
								object.insert("hasError".to_owned(), Value::Bool(true));
							},
							_ => value = json!({ "text": value, "hasError": true }),
						}
					}
					return Ok(bridge_envelope(value, updates));
				},
			}
		}
		Err(BridgeHostError::message(format!("tool {name} ended without a terminal result")))
	}
}
fn bridge_envelope(value: Value, updates: Vec<Value>) -> Value {
	json!({
		"__omp_bridge_value__": value,
		"__omp_bridge_updates__": updates,
	})
}

/// Optional capabilities owned by the live parent agent session.
///
/// The environment never synthesizes these operations. A session composition
/// that can perform them supplies this authenticated callback; otherwise the
/// corresponding bridge names are omitted from the grant.
#[async_trait]
pub(crate) trait ParentSessionHost: Send + Sync {
	async fn completion(&self, args: Value) -> Result<Value, BridgeHostError>;
	async fn agent(&self, args: Value) -> Result<Value, BridgeHostError>;
	async fn concurrency(&self, args: Value) -> Result<Value, BridgeHostError>;
	async fn budget(&self, args: Value) -> Result<Value, BridgeHostError>;
}

#[derive(Clone)]
pub(crate) struct EvalSessionConfig {
	pub(crate) local_roots_json: Str,
	pub(crate) artifacts_dir:    Str,
	pub(crate) session_file:     Str,
}

/// Late-bound host used to break the registry/eval construction cycle.
///
/// Binding is one-shot: the namespace installer cannot be redirected to a
/// different project registry after a grant has been minted.
pub(crate) struct SessionBridgeHost {
	registry: OnceLock<Arc<Registry>>,
	parent:   OnceLock<Arc<dyn ParentSessionHost>>,
	config:   Mutex<Option<EvalSessionConfig>>,
}

impl SessionBridgeHost {
	pub(crate) fn new() -> Self {
		Self { registry: OnceLock::new(), parent: OnceLock::new(), config: Mutex::new(None) }
	}

	pub(crate) fn bind_registry(&self, registry: Arc<Registry>) -> Result<(), BridgeHostError> {
		self
			.registry
			.set(registry)
			.map_err(|_| BridgeHostError::message("eval bridge registry is already bound"))
	}

	pub(crate) fn bind_parent(
		&self,
		parent: Arc<dyn ParentSessionHost>,
	) -> Result<(), BridgeHostError> {
		self
			.parent
			.set(parent)
			.map_err(|_| BridgeHostError::message("eval bridge parent session is already bound"))
	}

	pub(crate) fn set_session_config(&self, config: EvalSessionConfig) {
		*self.config.lock() = Some(config);
	}

	fn session_config(&self) -> Option<EvalSessionConfig> {
		self.config.lock().clone()
	}

	fn capabilities(&self) -> Result<BridgeCapabilities, BridgeHostError> {
		let registry = self
			.registry
			.get()
			.ok_or_else(|| BridgeHostError::message("eval bridge registry is not bound"))?;
		let tools = registry.live_identities().filter_map(|(name, _)| {
			(name.as_str() != "eval" && registry.route(name.as_str()).ok() == Some(ToolRoute::Native))
				.then(|| name.clone())
		});
		let capabilities = BridgeCapabilities::new(tools);
		Ok(if self.parent.get().is_some() {
			capabilities
				.with_completion()
				.with_agent()
				.with_concurrency()
				.with_budget()
		} else {
			capabilities
		})
	}
}

#[async_trait]
impl BridgeHost for SessionBridgeHost {
	async fn call(&self, name: &str, args: Value) -> Result<Value, BridgeHostError> {
		match name {
			COMPLETION => {
				self
					.parent
					.get()
					.ok_or_else(|| {
						BridgeHostError::message("completion is unavailable in this session")
					})?
					.completion(args)
					.await
			},
			AGENT => {
				self
					.parent
					.get()
					.ok_or_else(|| {
						BridgeHostError::message("subagents are unavailable in this session")
					})?
					.agent(args)
					.await
			},
			CONCURRENCY => {
				self
					.parent
					.get()
					.ok_or_else(|| {
						BridgeHostError::message("concurrency control is unavailable in this session")
					})?
					.concurrency(args)
					.await
			},
			BUDGET => {
				self
					.parent
					.get()
					.ok_or_else(|| BridgeHostError::message("budget is unavailable in this session"))?
					.budget(args)
					.await
			},
			_ => {
				let registry = self
					.registry
					.get()
					.ok_or_else(|| BridgeHostError::message("eval bridge registry is not bound"))?;
				RegistryBridgeHost::new(Arc::clone(registry))
					.call(name, args)
					.await
			},
		}
	}
}

/// Installs namespace-local bridge grants and tracks cancellation per active
/// cell.
pub(crate) struct BridgeNamespaceInstaller {
	dispatcher:       BridgeDispatcher,
	host:             Arc<SessionBridgeHost>,
	runtime:          Handle,
	session:          Str,
	next_run:         AtomicU64,
	namespace_aborts: Mutex<BTreeMap<i64, Arc<CellAbort>>>,
	cells:            Mutex<BTreeMap<Bytes, Arc<CellAbort>>>,
}

impl BridgeNamespaceInstaller {
	pub(crate) fn new(host: Arc<SessionBridgeHost>, runtime: Handle) -> Self {
		Self {
			dispatcher: BridgeDispatcher::new(),
			host,
			runtime,
			session: Str::from(format!("envd-eval-{}", Ulid::generate())),
			next_run: AtomicU64::new(1),
			namespace_aborts: Mutex::new(BTreeMap::new()),
			cells: Mutex::new(BTreeMap::new()),
		}
	}

	fn inject_config(&self, globals: &Bound<'_, PyDict>) -> PyResult<()> {
		if let Some(config) = self.host.session_config() {
			globals.set_item("OMP_EVAL_LOCAL_ROOTS", config.local_roots_json.as_str())?;
			globals.set_item("OMP_ARTIFACTS_DIR", config.artifacts_dir.as_str())?;
			globals.set_item("OMP_SESSION_FILE", config.session_file.as_str())?;
		}
		Ok(())
	}
}

impl NamespaceInstaller for BridgeNamespaceInstaller {
	fn install(&self, py: Python<'_>, globals: &Bound<'_, PyDict>) -> PyResult<()> {
		let run = Str::from(format!("namespace-{}", self.next_run.fetch_add(1, Ordering::Relaxed)));
		let capabilities = self
			.host
			.capabilities()
			.map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
		let host: Arc<dyn BridgeHost> = self.host.clone();
		let registration = self
			.dispatcher
			.register(self.session.clone(), run, capabilities, host, TimeoutHandle::new(None))
			.map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
		let abort = Arc::new(CellAbort::default());
		self.inject_config(globals)?;
		install_python_bridge(
			py,
			globals,
			registration.client().with_abort(Arc::clone(&abort)),
			self.runtime.clone(),
		)?;
		install_python_prelude(py, globals)?;
		self
			.namespace_aborts
			.lock()
			.insert(python_thread_id(py)?, abort);
		Ok(())
	}

	fn begin_cell(
		&self,
		py: Python<'_>,
		globals: &Bound<'_, PyDict>,
		cell_id: &Bytes,
		_timeout: Option<std::time::Duration>,
	) -> PyResult<()> {
		self.inject_config(globals)?;
		let abort = self
			.namespace_aborts
			.lock()
			.get(&python_thread_id(py)?)
			.cloned()
			.ok_or_else(|| {
				PyRuntimeError::new_err("eval namespace has no bridge cancellation scope")
			})?;
		abort.begin(cell_id);
		if let Some(stale) = self
			.cells
			.lock()
			.insert(cell_id.clone(), Arc::clone(&abort))
		{
			stale.cancel(cell_id);
		}
		Ok(())
	}

	fn end_cell(
		&self,
		_py: Python<'_>,
		_globals: &Bound<'_, PyDict>,
		cell_id: &Bytes,
	) -> PyResult<()> {
		if let Some(abort) = self.cells.lock().remove(cell_id) {
			abort.end(cell_id);
		}
		Ok(())
	}

	fn cancel_cell(&self, cell_id: &Bytes) {
		if let Some(abort) = self.cells.lock().get(cell_id).cloned() {
			abort.cancel(cell_id);
		}
	}
}

fn python_thread_id(py: Python<'_>) -> PyResult<i64> {
	PyModule::import(py, "threading")?
		.call_method0("get_ident")?
		.extract()
}

fn registry_error(error: impl fmt::Display) -> BridgeHostError {
	BridgeHostError::message(error.to_string())
}

fn projected_parts(parts: Vec<Part>) -> Result<Value, BridgeHostError> {
	let mut text = String::new();
	let mut json_parts: Vec<Value> = Vec::new();
	let mut blobs: Vec<Value> = Vec::new();
	for part in parts {
		match part {
			Part::Text { text: value } => text.push_str(value.as_str()),
			Part::Json { json } => {
				json_parts.push(serde_json::from_slice(&json).map_err(|error| {
					BridgeHostError::message(format!("tool returned invalid JSON: {error}"))
				})?)
			},
			Part::Blob { blob, alt } => blobs.push(json!({ "blob": blob, "alt": alt })),
		}
	}
	if json_parts.is_empty() && blobs.is_empty() {
		return Ok(Value::String(text));
	}
	Ok(json!({ "text": text, "json": json_parts, "blobs": blobs }))
}

#[pyclass(frozen)]
struct PythonBridgeCallable {
	client:  BridgeClient,
	runtime: Handle,
}

#[pymethods]
impl PythonBridgeCallable {
	fn __call__<'py>(
		&self,
		py: Python<'py>,
		name: &str,
		args: &Bound<'py, PyAny>,
	) -> PyResult<Py<PyAny>> {
		let json_module = PyModule::import(py, "json")?;
		let encoded: String = json_module.call_method1("dumps", (args,))?.extract()?;
		let args = serde_json::from_str(&encoded).map_err(|error| {
			PyRuntimeError::new_err(format!("bridge arguments are not JSON: {error}"))
		})?;
		let value = self
			.runtime
			.block_on(self.client.call(name, args))
			.map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
		let encoded = serde_json::to_string(&value)
			.map_err(|error| PyRuntimeError::new_err(format!("bridge result is not JSON: {error}")))?;
		Ok(json_module.call_method1("loads", (encoded,))?.unbind())
	}
}

/// Installs the authenticated direct bridge callable in one persistent Python
/// namespace. The callable owns a single session/run grant; Python code cannot
/// supply or swap credentials.
pub(crate) fn install_python_bridge(
	py: Python<'_>,
	globals: &Bound<'_, PyDict>,
	client: BridgeClient,
	runtime: Handle,
) -> PyResult<()> {
	globals.set_item("__omp_bridge_session__", client.session())?;
	globals.set_item("__omp_bridge_call__", Py::new(py, PythonBridgeCallable { client, runtime })?)
}

/// Loads the normative helper prelude once into a persistent namespace.
pub(crate) fn install_python_prelude(py: Python<'_>, globals: &Bound<'_, PyDict>) -> PyResult<()> {
	let source = CString::new(PYTHON_PRELUDE)
		.map_err(|_| PyRuntimeError::new_err("embedded Python prelude contains a NUL byte"))?;
	py.run(source.as_c_str(), Some(globals), Some(globals))
}

#[cfg(test)]
mod tests {
	use std::sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	};

	use async_stream::stream;
	use futures::Stream;
	use omp_tool::{Constraint, Ev, Outcome, Rev, Tool, ToolSpec};
	use serde::{Deserialize, Deserializer, Serialize};

	use super::*;

	struct RecordingHost {
		calls: AtomicUsize,
		fail:  bool,
	}

	#[async_trait]
	impl BridgeHost for RecordingHost {
		async fn call(&self, name: &str, args: Value) -> Result<Value, BridgeHostError> {
			self.calls.fetch_add(1, Ordering::Relaxed);
			if self.fail {
				return Err(BridgeHostError::message("host exploded"));
			}
			Ok(json!({ "name": name, "args": args }))
		}
	}

	struct RecordingParent {
		calls: AtomicUsize,
	}

	impl RecordingParent {
		fn response(&self, operation: &str, args: Value) -> Value {
			self.calls.fetch_add(1, Ordering::Relaxed);
			json!({ "operation": operation, "args": args })
		}
	}

	#[async_trait]
	impl ParentSessionHost for RecordingParent {
		async fn completion(&self, args: Value) -> Result<Value, BridgeHostError> {
			Ok(self.response("completion", args))
		}

		async fn agent(&self, args: Value) -> Result<Value, BridgeHostError> {
			Ok(self.response("agent", args))
		}

		async fn concurrency(&self, args: Value) -> Result<Value, BridgeHostError> {
			Ok(self.response("concurrency", args))
		}

		async fn budget(&self, args: Value) -> Result<Value, BridgeHostError> {
			Ok(self.response("budget", args))
		}
	}

	enum ProbeUpdate {
		Value(Value),
		Invalid,
	}

	impl Serialize for ProbeUpdate {
		fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
			match self {
				Self::Value(value) => value.serialize(serializer),
				Self::Invalid => Err(<S::Error as serde::ser::Error>::custom("invalid probe update")),
			}
		}
	}

	impl<'de> Deserialize<'de> for ProbeUpdate {
		fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
			Value::deserialize(deserializer).map(Self::Value)
		}
	}

	struct StreamingProbe {
		spec:    ToolSpec,
		invalid: bool,
	}

	impl StreamingProbe {
		fn new(name: &'static str, invalid: bool) -> Self {
			Self {
				spec: ToolSpec {
					name:        Str::new_static(name),
					rev:         Rev { family: Str::default(), n: 1 },
					description: Str::new_static("eval bridge update probe"),
					schema:      Bytes::from_static(
						br#"{"type":"object","additionalProperties":false}"#,
					),
					constraint:  Constraint::None,
				},
				invalid,
			}
		}
	}

	impl Tool for StreamingProbe {
		type Fault = Value;
		type Params = Value;
		type Payload = Value;
		type Update = ProbeUpdate;

		fn spec(&self) -> &ToolSpec {
			&self.spec
		}

		fn call<'c>(
			&'c self,
			mut params: IncomingParams<'c>,
		) -> impl Stream<Item = Ev<Self::Update, Self::Payload, Self::Fault>> + Send + 'c {
			stream! {
				params.whole::<Value>().await.expect("probe arguments");
				params.committed().await.expect("probe commitment");
				if self.invalid {
					yield Ev::Update(ProbeUpdate::Invalid);
				} else {
					yield Ev::Update(ProbeUpdate::Value(json!({"step": 1})));
					yield Ev::Update(ProbeUpdate::Value(json!({"step": 2})));
				}
				yield Ev::Done(Outcome::Done {
					result: Ok(json!({"terminal": "done"})),
					useless: false,
				});
			}
		}

		fn prompt(
			&self,
			view: Result<&Self::Payload, &Self::Fault>,
			_caps: &PromptCaps,
		) -> Vec<Part> {
			let value = match view {
				Ok(value) => value["terminal"].as_str().unwrap_or("missing"),
				Err(_) => "fault",
			};
			vec![Part::Text { text: Str::from(value) }]
		}
	}

	fn dispatcher() -> BridgeDispatcher {
		BridgeDispatcher::new()
	}

	#[tokio::test]
	async fn authenticates_and_scopes_calls() {
		let dispatcher = dispatcher();
		let host = Arc::new(RecordingHost { calls: AtomicUsize::new(0), fail: false });
		let registration = dispatcher
			.register(
				Str::new_static("session"),
				Str::new_static("run"),
				BridgeCapabilities::new([Str::new_static("read")]).with_budget(),
				host.clone(),
				TimeoutHandle::new(None),
			)
			.expect("register bridge");
		let client = registration.client();
		assert_eq!(
			client
				.call("read", json!({ "path": "x" }))
				.await
				.expect("allowed call"),
			json!({ "name": "read", "args": { "path": "x" } })
		);
		assert_eq!(host.calls.load(Ordering::Relaxed), 1);
		assert_eq!(
			client.call("write", json!({})).await,
			Err(BridgeCallError::CapabilityDenied { name: Str::new_static("write") })
		);
		assert_eq!(host.calls.load(Ordering::Relaxed), 1, "denied calls never reach host");
	}

	#[tokio::test]
	async fn rejects_tampered_grants_and_keeps_registration_alive_with_the_client() {
		let dispatcher = dispatcher();
		let host = Arc::new(RecordingHost { calls: AtomicUsize::new(0), fail: false });
		let registration = dispatcher
			.register(
				Str::new_static("session"),
				Str::new_static("run"),
				BridgeCapabilities::new([Str::new_static("read")]),
				host,
				TimeoutHandle::new(None),
			)
			.expect("register bridge");
		let client = registration.client();
		let mut forged = client.clone();
		forged.grant.token = SecretString::from("wrong".to_owned());
		assert_eq!(forged.call("read", json!({})).await, Err(BridgeCallError::AuthenticationFailed));
		drop(registration);
		assert!(client.call("read", json!({})).await.is_ok());
		assert_eq!(dispatcher.inner.registrations.lock().len(), 1);
		drop(client);
		drop(forged);
		assert!(dispatcher.inner.registrations.lock().is_empty());
	}

	#[tokio::test]
	async fn cancelling_one_worker_cell_does_not_cancel_another_namespace() {
		let installer =
			BridgeNamespaceInstaller::new(Arc::new(SessionBridgeHost::new()), Handle::current());
		let first_id = Bytes::from_static(b"session-a:cell-1");
		let second_id = Bytes::from_static(b"session-b:cell-2");
		let first = Arc::new(CellAbort::default());
		let second = Arc::new(CellAbort::default());
		first.begin(&first_id);
		second.begin(&second_id);
		let first_token = first.token().expect("first token");
		let second_token = second.token().expect("second token");
		installer.cells.lock().insert(first_id.clone(), first);
		installer.cells.lock().insert(second_id.clone(), second);

		NamespaceInstaller::cancel_cell(&installer, &first_id);
		assert!(first_token.is_cancelled());
		assert!(!second_token.is_cancelled(), "another worker's host call remains active");
		NamespaceInstaller::cancel_cell(&installer, &second_id);
		assert!(second_token.is_cancelled());
	}

	#[tokio::test]
	async fn propagates_host_errors() {
		let dispatcher = dispatcher();
		let registration = dispatcher
			.register(
				Str::new_static("session"),
				Str::new_static("run"),
				BridgeCapabilities::new([Str::new_static("read")]),
				Arc::new(RecordingHost { calls: AtomicUsize::new(0), fail: true }),
				TimeoutHandle::new(None),
			)
			.expect("register bridge");
		assert_eq!(
			registration.client().call("read", json!({})).await,
			Err(BridgeCallError::Host { message: Str::new_static("host exploded") })
		);
	}

	#[tokio::test]
	async fn parent_helpers_use_only_the_bound_session_host() {
		let parent = Arc::new(RecordingParent { calls: AtomicUsize::new(0) });
		let host = Arc::new(SessionBridgeHost::new());
		host
			.bind_registry(Arc::new(Registry::new()))
			.expect("bind registry");
		host.bind_parent(parent.clone()).expect("bind parent");
		let capabilities = host.capabilities().expect("bound capabilities");
		let registration = dispatcher()
			.register(
				Str::new_static("owner"),
				Str::new_static("cell"),
				capabilities,
				host,
				TimeoutHandle::new(None),
			)
			.expect("register owner");
		let client = registration.client();
		for (name, operation) in [
			(COMPLETION, "completion"),
			(AGENT, "agent"),
			(CONCURRENCY, "concurrency"),
			(BUDGET, "budget"),
		] {
			assert_eq!(
				client
					.call(name, json!({ "marker": operation }))
					.await
					.expect("parent call"),
				json!({ "operation": operation, "args": { "marker": operation } })
			);
		}
		assert_eq!(parent.calls.load(Ordering::Relaxed), 4);
	}

	#[tokio::test]
	async fn absent_parent_capabilities_are_typed_denials() {
		let host = Arc::new(SessionBridgeHost::new());
		host
			.bind_registry(Arc::new(Registry::new()))
			.expect("bind registry");
		let capabilities = host.capabilities().expect("bound capabilities");
		let registration = dispatcher()
			.register(
				Str::new_static("owner"),
				Str::new_static("cell"),
				capabilities,
				host,
				TimeoutHandle::new(None),
			)
			.expect("register owner");
		assert_eq!(
			registration.client().call(COMPLETION, json!({})).await,
			Err(BridgeCallError::CapabilityDenied { name: Str::new_static(COMPLETION) })
		);
	}

	#[tokio::test]
	async fn registry_bridge_preserves_ordered_updates_in_its_private_envelope() {
		let mut registry = Registry::new();
		registry
			.register(StreamingProbe::new("update_probe", false))
			.expect("register update probe");
		let host = RegistryBridgeHost::new(Arc::new(registry));
		assert_eq!(
			host
				.call("update_probe", json!({"i":"py prelude"}))
				.await
				.expect("bridge probe call"),
			json!({
				"__omp_bridge_value__": "done",
				"__omp_bridge_updates__": [
					{"step": 1},
					{"step": 2}
				]
			})
		);
	}

	#[tokio::test]
	async fn registry_bridge_surfaces_invalid_update_serialization_as_a_host_fault() {
		let mut registry = Registry::new();
		registry
			.register(StreamingProbe::new("invalid_update_probe", true))
			.expect("register invalid update probe");
		let host = RegistryBridgeHost::new(Arc::new(registry));
		let error = host
			.call("invalid_update_probe", json!({}))
			.await
			.expect_err("invalid update serialization unexpectedly succeeded");
		assert_eq!(error.to_string(), "tool value serialization failed: invalid probe update");
	}
}
