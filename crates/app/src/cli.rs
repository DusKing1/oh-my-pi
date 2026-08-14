//! Command parsing and production dispatch for the `omp` executable.

use std::{
	path::{Path, PathBuf},
	sync::Arc,
	time::{SystemTime, UNIX_EPOCH},
};

use clap::{Args, Parser, Subcommand};
use futures::StreamExt as _;
use miette::{IntoDiagnostic as _, miette};
use omp_core::Str;
use omp_llm_catalog::{ModelKey, compile::compile_oracle};
#[cfg(feature = "local-applefm")]
use omp_llm_inference::local::applefm::{AppleFm, AppleFmEvent, AppleFmOptions};
use omp_llm_inference::{
	Client,
	call::{
		CallMeta, ChatRequest, ContentPart, Message, NegotiationPolicy, Role, Sampling, Setting,
		Target,
	},
	event::ChatEvent,
	id::RequestId,
	receipt::ExecutionBudget,
};
use tokio::io::AsyncWriteExt as _;

use crate::{
	daemon::{DaemonConfig, DaemonHandle},
	endpoint::LocalEndpoint,
};

/// Top-level parser for the production `omp` executable.
#[derive(Clone, Debug, Parser)]
#[command(name = "omp", version, about = "OMP inference and credential management")]
pub struct OmpCli {
	/// Operation to run.
	#[command(subcommand)]
	pub command: Command,
}

/// Production application commands.
#[derive(Clone, Debug, Subcommand)]
pub enum Command {
	/// Start the inference gateway on a platform-native local endpoint.
	Serve(ServeArgs),
	/// Start the project environment daemon.
	Envd(EnvdArgs),
	/// Start an interactive project agent session.
	Chat(ChatArgs),
	/// Run one typed operation in process.
	Infer(InferArgs),
	/// Manage provider credentials.
	Auth(AuthArgs),
	/// Manage generated model-catalog data.
	Catalog(CatalogArgs),
	/// Run hardware-accelerated local inference.
	Local(LocalArgs),
}

/// Gateway serving options.
#[derive(Clone, Debug, Args)]
pub struct ServeArgs {
	/// Platform-local endpoint: a Unix socket path or Windows named-pipe name.
	#[arg(long = "endpoint", visible_aliases = ["uds", "pipe"], value_name = "LOCAL_ENDPOINT")]
	pub endpoint: LocalEndpoint,
	/// Override the directory containing daemon state.
	#[arg(long, value_name = "PATH")]
	pub data_dir: Option<PathBuf>,
}
/// Project environment-daemon options.
#[derive(Clone, Debug, Args)]
pub struct EnvdArgs {
	/// Workspace root exposed by the environment.
	#[arg(long, value_name = "PATH", default_value = ".")]
	pub root:             PathBuf,
	/// Owner-only environment socket. Defaults to `<state-dir>/env.sock`.
	#[arg(long, value_name = "PATH")]
	pub socket:           Option<PathBuf>,
	/// Document-server socket. Defaults to `<state-dir>/docserver.sock`.
	#[arg(long, value_name = "PATH")]
	pub docserver_socket: Option<PathBuf>,
	/// Environment state directory. Defaults to a project-keyed directory under
	/// `OMP_DATA_DIR`.
	#[arg(long, value_name = "PATH")]
	pub state_dir:        Option<PathBuf>,
	/// Enable the built-in Python expression-evaluation tool.
	///
	/// This executes Python inside the environment owner's process sandbox and
	/// is disabled unless explicitly requested.
	#[arg(long)]
	pub py_eval:          bool,
}
/// Interactive project-chat options.
#[derive(Clone, Debug, Args)]
pub struct ChatArgs {
	/// Catalog model key, alias, or role.
	#[arg(long)]
	pub model:   Str,
	/// Project root whose environment and durable sessions are used.
	#[arg(long, value_name = "PATH", default_value = ".")]
	pub project: PathBuf,
	/// Existing inference gateway endpoint. Omit to run inference in process.
	#[arg(long, value_name = "LOCAL_ENDPOINT")]
	pub gateway: Option<LocalEndpoint>,
	/// Existing ULID session to reopen strictly.
	#[arg(long, value_name = "ULID")]
	pub resume:  Option<Str>,
	/// Enable the environment-owned Python expression-evaluation tool.
	#[arg(long)]
	pub py_eval: bool,
}

/// Direct typed inference options.
#[derive(Clone, Debug, Args)]
pub struct InferArgs {
	/// Catalog model key.
	#[arg(long)]
	pub model:  Str,
	/// User prompt.
	#[arg(long)]
	pub prompt: Str,
}

/// Authentication command options.
#[derive(Clone, Debug, Args)]
pub struct AuthArgs {
	/// OMP data directory containing `credentials.db`.
	#[arg(long, value_name = "PATH")]
	pub data_dir: Option<PathBuf>,
	/// Authentication operation.
	#[command(subcommand)]
	pub command:  AuthCommand,
}

/// Typed authentication commands.
#[derive(Clone, Debug, Subcommand)]
pub enum AuthCommand {
	/// Begin an interactive provider login.
	Login {
		/// Target provider identifier.
		provider: Str,
	},
	/// List non-secret account summaries.
	List {
		/// Optional provider filter.
		#[arg(long)]
		provider: Option<Str>,
	},
	/// Refresh one account.
	Refresh {
		/// Target account identifier.
		account: Str,
	},
	/// Remove one account.
	Logout {
		/// Target account identifier.
		account: Str,
	},
}

/// Model-catalog command tree.
#[derive(Clone, Debug, Args)]
pub struct CatalogArgs {
	/// Catalog operation.
	#[command(subcommand)]
	pub command: CatalogCommand,
}

/// Model-catalog operations.
#[derive(Clone, Debug, Subcommand)]
pub enum CatalogCommand {
	/// Import catalog sources into normalized JSON.
	Import(CatalogImportArgs),
}

/// Catalog compiler inputs and normalized output.
#[derive(Clone, Debug, Args)]
pub struct CatalogImportArgs {
	/// Provider manifest TOML.
	#[arg(long, value_name = "TOML")]
	pub providers:   PathBuf,
	/// Secret-free OAuth flow manifest TOML.
	#[arg(long, value_name = "TOML")]
	pub oauth:       PathBuf,
	/// Compressed oracle model rows.
	#[arg(long, value_name = "ZST")]
	pub models:      PathBuf,
	/// Destination normalized JSON.
	#[arg(long, value_name = "JSON")]
	pub destination: PathBuf,
}

/// In-process local inference command tree.
#[derive(Clone, Debug, Args)]
pub struct LocalArgs {
	/// Local inference operation.
	#[command(subcommand)]
	pub command: LocalCommand,
}

/// Local inference operations.
#[derive(Clone, Debug, Subcommand)]
pub enum LocalCommand {
	/// Run local in-process inference.
	Infer(LocalInferArgs),
}

/// In-process Apple Foundation Models options.
#[derive(Clone, Debug, Args)]
pub struct LocalInferArgs {
	/// User prompt.
	#[arg(long)]
	pub prompt: Str,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DispatchTarget {
	Serve,
	Envd,
	Chat,
	Infer,
	Auth,
	CatalogImport,
	LocalInfer,
}

#[cfg(test)]
const fn dispatch_target(command: &Command) -> DispatchTarget {
	match command {
		Command::Serve(_) => DispatchTarget::Serve,
		Command::Envd(_) => DispatchTarget::Envd,
		Command::Chat(_) => DispatchTarget::Chat,
		Command::Infer(_) => DispatchTarget::Infer,
		Command::Auth(_) => DispatchTarget::Auth,
		Command::Catalog(CatalogArgs { command: CatalogCommand::Import(_) }) => {
			DispatchTarget::CatalogImport
		},
		Command::Local(LocalArgs { command: LocalCommand::Infer(_) }) => DispatchTarget::LocalInfer,
	}
}

/// Dispatches one parsed command to its production implementation.
#[expect(
	clippy::future_not_send,
	reason = "chat dispatch preserves the thread-confined omp_tui::App future"
)]
pub async fn dispatch(cli: OmpCli) -> miette::Result<()> {
	match cli.command {
		Command::Serve(args) => serve(args).await,
		Command::Envd(args) => crate::envd::run(args).await,
		Command::Chat(args) => Box::pin(crate::chat::run(args)).await,
		Command::Infer(args) => infer(args).await,
		Command::Auth(args) => auth(args).await,
		Command::Catalog(CatalogArgs { command: CatalogCommand::Import(args) }) => {
			catalog_import(&args)
		},
		Command::Local(LocalArgs { command: LocalCommand::Infer(args) }) => local_infer(args).await,
	}
}

async fn serve(args: ServeArgs) -> miette::Result<()> {
	let config = args.data_dir.map_or_else(
		|| DaemonConfig::local(args.endpoint.clone()),
		|dir| DaemonConfig::local(args.endpoint.clone()).with_data_dir(dir),
	);
	let handle = DaemonHandle::start(config).await.into_diagnostic()?;
	handle.wait().await.into_diagnostic()?;
	Ok(())
}

async fn infer(args: InferArgs) -> miette::Result<()> {
	let data_dir = data_dir(None)?;
	let store =
		crate::daemon::open_credential_store(data_dir.join("credentials.db")).into_diagnostic()?;
	let registry = crate::daemon::production_registry(&data_dir, store)
		.await
		.into_diagnostic()?;
	let planner =
		omp_llm_inference::router::Router::new(registry.clone(), std::time::Duration::from_secs(30));
	let meta = CallMeta {
		id:       RequestId::from(turn_id()),
		target:   Target::Model(ModelKey::from(args.model)),
		deadline: None,
		budget:   ExecutionBudget::default(),
		session:  None,
	};
	let mut client = Client::new(registry.service(), planner, meta);
	let mut events = client
		.execute(chat_request(args.prompt))
		.await
		.into_diagnostic()?;
	let mut completed = false;
	let mut stdout = tokio::io::stdout();
	while let Some(event) = events.next().await {
		match event.into_diagnostic()? {
			ChatEvent::TextDelta { text, .. } => {
				stdout.write_all(text.as_bytes()).await.into_diagnostic()?;
			},
			ChatEvent::Completed(_) => completed = true,
			_ => {},
		}
	}
	if !completed {
		return Err(miette!("inference stream ended without completion"));
	}
	stdout.write_all(b"\n").await.into_diagnostic()?;
	stdout.flush().await.into_diagnostic()?;
	Ok(())
}

fn chat_request(prompt: Str) -> ChatRequest {
	ChatRequest {
		messages:          Arc::from([Message {
			role:    Role::User,
			content: Arc::from([ContentPart::Text { text: prompt, proof: None }]),
			name:    None,
		}]),
		tools:             Arc::from([]),
		hosted_tools:      Arc::from([]),
		tool_choice:       Setting::Unset,
		output:            Setting::Unset,
		reasoning:         Setting::Unset,
		verbosity:         Setting::Unset,
		cache_retention:   Setting::Unset,
		service_tier:      Setting::Unset,
		sampling:          Sampling::default(),
		max_output_tokens: None,
		top_logprobs:      None,
		safety:            Arc::from([]),
		negotiation:       NegotiationPolicy::default(),
	}
}

fn turn_id() -> String {
	let now = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_nanos();
	format!("omp-cli-{}-{now}", std::process::id())
}

async fn auth(args: AuthArgs) -> miette::Result<()> {
	let data = data_dir(args.data_dir)?;
	crate::auth_backend::run(data.join("credentials.db"), args.command).await
}

pub(crate) fn data_dir(explicit: Option<PathBuf>) -> miette::Result<PathBuf> {
	if let Some(path) = explicit {
		return Ok(path);
	}
	if let Some(path) = std::env::var_os("OMP_DATA_DIR") {
		return Ok(path.into());
	}
	let home =
		std::env::var_os("HOME").ok_or_else(|| miette!("HOME or OMP_DATA_DIR must be set"))?;
	Ok(PathBuf::from(home).join(".local/share/omp"))
}

fn catalog_import(args: &CatalogImportArgs) -> miette::Result<()> {
	if same_path(&args.providers, &args.destination)
		|| same_path(&args.oauth, &args.destination)
		|| same_path(&args.models, &args.destination)
	{
		return Err(miette!("catalog inputs and destination must be different files"));
	}
	let providers = std::fs::read_to_string(&args.providers).into_diagnostic()?;
	let oauth = std::fs::read_to_string(&args.oauth).into_diagnostic()?;
	let models = std::fs::read(&args.models).into_diagnostic()?;
	let payload = compile_oracle(&providers, &models, &oauth)
		.into_diagnostic()?
		.normalized_json()
		.into_diagnostic()?;
	if let Some(parent) = args
		.destination
		.parent()
		.filter(|path| !path.as_os_str().is_empty())
	{
		std::fs::create_dir_all(parent).into_diagnostic()?;
	}
	std::fs::write(&args.destination, payload).into_diagnostic()?;
	Ok(())
}

fn same_path(left: &Path, right: &Path) -> bool {
	left == right
		|| left
			.canonicalize()
			.ok()
			.zip(right.canonicalize().ok())
			.is_some_and(|(left, right)| left == right)
}

#[cfg(feature = "local-applefm")]
async fn local_infer(args: LocalInferArgs) -> miette::Result<()> {
	let model = AppleFm::load().await.into_diagnostic()?;
	let mut events = model
		.stream(AppleFmOptions::new(args.prompt))
		.into_diagnostic()?;
	let mut completed = false;
	let mut stdout = tokio::io::stdout();
	while let Some(event) = events.next().await {
		match event.into_diagnostic()? {
			AppleFmEvent::Delta(text) => stdout.write_all(text.as_bytes()).await.into_diagnostic()?,
			AppleFmEvent::Finished(_) => completed = true,
		}
	}
	if !completed {
		return Err(miette!("local inference stream ended without completion"));
	}
	stdout.write_all(b"\n").await.into_diagnostic()?;
	stdout.flush().await.into_diagnostic()?;
	Ok(())
}

#[cfg(not(feature = "local-applefm"))]
fn local_infer(_args: LocalInferArgs) -> std::future::Ready<miette::Result<()>> {
	std::future::ready(Err(miette!("local inference requires the `local-applefm` feature")))
}

#[cfg(test)]
mod tests {
	use clap::error::ErrorKind;

	use super::*;

	fn parse(arguments: &[&str]) -> OmpCli {
		OmpCli::try_parse_from(arguments).expect("valid command")
	}

	#[cfg(unix)]
	const TEST_ENDPOINT: &str = "/tmp/omp.sock";
	#[cfg(windows)]
	const TEST_ENDPOINT: &str = r"\\.\pipe\omp-cli-test";

	#[test]
	fn parses_every_dispatch_branch() {
		let cases = [
			(&["omp", "serve", "--endpoint", TEST_ENDPOINT][..], DispatchTarget::Serve),
			(&["omp", "envd"][..], DispatchTarget::Envd),
			(
				&["omp", "chat", "--model", "provider/model", "--project", "."][..],
				DispatchTarget::Chat,
			),
			(
				&["omp", "infer", "--model", "provider/model", "--prompt", "hello"][..],
				DispatchTarget::Infer,
			),
			(&["omp", "auth", "list"][..], DispatchTarget::Auth),
			(
				&[
					"omp",
					"catalog",
					"import",
					"--providers",
					"providers.toml",
					"--oauth",
					"oauth.toml",
					"--models",
					"models.json.zst",
					"--destination",
					"catalog.json",
				][..],
				DispatchTarget::CatalogImport,
			),
			(&["omp", "local", "infer", "--prompt", "hello"][..], DispatchTarget::LocalInfer),
		];
		for (arguments, expected) in cases {
			assert_eq!(dispatch_target(&parse(arguments).command), expected);
		}
	}
	#[test]
	fn parses_chat_composition_options() {
		let Command::Chat(args) = parse(&[
			"omp",
			"chat",
			"--model",
			"provider/model",
			"--project",
			"workspace",
			"--gateway",
			TEST_ENDPOINT,
			"--resume",
			"01ARZ3NDEKTSV4RRFFQ69G5FAV",
			"--py-eval",
		])
		.command
		else {
			panic!("chat command");
		};
		assert_eq!(args.model, Str::from("provider/model"));
		assert_eq!(args.project, PathBuf::from("workspace"));
		assert_eq!(args.gateway.as_ref().map(LocalEndpoint::as_path), Some(Path::new(TEST_ENDPOINT)));
		assert_eq!(args.resume, Some(Str::from("01ARZ3NDEKTSV4RRFFQ69G5FAV")));
		assert!(args.py_eval);
	}

	#[test]
	fn parses_every_auth_branch() {
		assert!(matches!(
			parse(&["omp", "auth", "login", "provider"]).command,
			Command::Auth(AuthArgs { command: AuthCommand::Login { .. }, .. })
		));
		assert!(matches!(
			parse(&["omp", "auth", "list", "--provider", "provider"]).command,
			Command::Auth(AuthArgs { command: AuthCommand::List { provider: Some(_) }, .. })
		));
		assert!(matches!(
			parse(&["omp", "auth", "refresh", "account"]).command,
			Command::Auth(AuthArgs { command: AuthCommand::Refresh { .. }, .. })
		));
		assert!(matches!(
			parse(&["omp", "auth", "logout", "account"]).command,
			Command::Auth(AuthArgs { command: AuthCommand::Logout { .. }, .. })
		));
	}

	#[test]
	fn rejects_incomplete_commands() {
		for arguments in [
			&["omp", "serve"][..],
			&["omp", "chat"][..],
			&["omp", "infer", "--model", "provider/model"][..],
			&["omp", "local", "infer"][..],
			&["omp", "catalog", "import", "--providers", "providers.toml", "--oauth", "oauth.toml"][..],
			&["omp", "auth", "login"][..],
		] {
			assert_eq!(
				OmpCli::try_parse_from(arguments)
					.expect_err("command must be rejected")
					.kind(),
				ErrorKind::MissingRequiredArgument
			);
		}
	}
}
