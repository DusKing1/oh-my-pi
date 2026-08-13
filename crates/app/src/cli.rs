//! Command parsing and production dispatch for the `omp` executable.

use std::{
	path::{Path, PathBuf},
	sync::Arc,
	time::{SystemTime, UNIX_EPOCH},
};

use clap::{Args, Parser, Subcommand};
use futures::StreamExt as _;
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
	error::AppError,
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
	Login { provider: Str },
	/// List non-secret account summaries.
	List {
		#[arg(long)]
		provider: Option<Str>,
	},
	/// Refresh one account.
	Refresh { account: Str },
	/// Remove one account.
	Logout { account: Str },
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
	Infer,
	Auth,
	CatalogImport,
	LocalInfer,
}

#[cfg(test)]
const fn dispatch_target(command: &Command) -> DispatchTarget {
	match command {
		Command::Serve(_) => DispatchTarget::Serve,
		Command::Infer(_) => DispatchTarget::Infer,
		Command::Auth(_) => DispatchTarget::Auth,
		Command::Catalog(CatalogArgs { command: CatalogCommand::Import(_) }) => {
			DispatchTarget::CatalogImport
		},
		Command::Local(LocalArgs { command: LocalCommand::Infer(_) }) => DispatchTarget::LocalInfer,
	}
}

/// Dispatches one parsed command to its production implementation.
pub async fn dispatch(cli: OmpCli) -> crate::Result<()> {
	match cli.command {
		Command::Serve(args) => serve(args).await,
		Command::Infer(args) => infer(args).await,
		Command::Auth(args) => auth(args).await,
		Command::Catalog(CatalogArgs { command: CatalogCommand::Import(args) }) => {
			catalog_import(&args)
		},
		Command::Local(LocalArgs { command: LocalCommand::Infer(args) }) => local_infer(args).await,
	}
}

async fn serve(args: ServeArgs) -> crate::Result<()> {
	let config = args.data_dir.map_or_else(
		|| DaemonConfig::local(args.endpoint.clone()),
		|dir| DaemonConfig::local(args.endpoint.clone()).with_data_dir(dir),
	);
	DaemonHandle::start(config).await?.wait().await?;
	Ok(())
}

async fn infer(args: InferArgs) -> crate::Result<()> {
	let data_dir = data_dir(None)?;
	let store = crate::daemon::open_credential_store(data_dir.join("credentials.db"))?;
	let registry = crate::daemon::production_registry(&data_dir, store).await?;
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
	let mut events = client.execute(chat_request(args.prompt)).await?;
	let mut completed = false;
	let mut stdout = tokio::io::stdout();
	while let Some(event) = events.next().await {
		match event? {
			ChatEvent::TextDelta { text, .. } => stdout.write_all(text.as_bytes()).await?,
			ChatEvent::Completed(_) => completed = true,
			_ => {},
		}
	}
	if !completed {
		return Err(AppError::InferenceStreamUnterminated);
	}
	stdout.write_all(b"\n").await?;
	stdout.flush().await?;
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

async fn auth(args: AuthArgs) -> crate::Result<()> {
	let data = data_dir(args.data_dir)?;
	crate::auth_backend::run(data.join("credentials.db"), args.command).await
}

fn data_dir(explicit: Option<PathBuf>) -> crate::Result<PathBuf> {
	if let Some(path) = explicit {
		return Ok(path);
	}
	if let Some(path) = std::env::var_os("OMP_DATA_DIR") {
		return Ok(path.into());
	}
	let home = std::env::var_os("HOME").ok_or(AppError::DataDirNotConfigured)?;
	Ok(PathBuf::from(home).join(".local/share/omp"))
}

fn catalog_import(args: &CatalogImportArgs) -> crate::Result<()> {
	if same_path(&args.providers, &args.destination)
		|| same_path(&args.oauth, &args.destination)
		|| same_path(&args.models, &args.destination)
	{
		return Err(AppError::SameCatalogSourceAndDestination);
	}
	let providers = std::fs::read_to_string(&args.providers)?;
	let oauth = std::fs::read_to_string(&args.oauth)?;
	let models = std::fs::read(&args.models)?;
	let payload = compile_oracle(&providers, &models, &oauth)?.normalized_json()?;
	if let Some(parent) = args
		.destination
		.parent()
		.filter(|path| !path.as_os_str().is_empty())
	{
		std::fs::create_dir_all(parent)?;
	}
	std::fs::write(&args.destination, payload)?;
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
async fn local_infer(args: LocalInferArgs) -> crate::Result<()> {
	let model = AppleFm::load().await?;
	let mut events = model.stream(AppleFmOptions::new(args.prompt))?;
	let mut completed = false;
	let mut stdout = tokio::io::stdout();
	while let Some(event) = events.next().await {
		match event? {
			AppleFmEvent::Delta(text) => stdout.write_all(text.as_bytes()).await?,
			AppleFmEvent::Finished(_) => completed = true,
		}
	}
	if !completed {
		return Err(AppError::LocalInferenceStreamUnterminated);
	}
	stdout.write_all(b"\n").await?;
	stdout.flush().await?;
	Ok(())
}

#[cfg(not(feature = "local-applefm"))]
async fn local_infer(_args: LocalInferArgs) -> crate::Result<()> {
	Err(AppError::LocalFeatureDisabled)
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
