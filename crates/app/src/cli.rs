//! Command parsing and production dispatch for the `omp` executable.

use std::{
	collections::BTreeSet,
	path::{Path, PathBuf},
	sync::Arc,
	time::{SystemTime, UNIX_EPOCH},
};

use clap::{Args, Parser, Subcommand};
use futures::{StreamExt as _, stream};
use omp_core::Str;
use omp_llm_broker::cli::{self as broker_cli, AuthCli, AuthCommand};
use omp_llm_catalog::models::import_catalog_zstd;
use omp_llm_gateway::local::LocalEndpoint;
use omp_llm_local::{Embedded, Inference, TextModel, TextSelection};
use omp_llm_types::{
	Chat, ChatRequest, Item, ItemKind, Message, Part, Props, Role, StreamPartKind, Thread, TurnEvent,
};
use omp_proto::inference::v1::{
	TurnFrame, inference_client::InferenceClient, turn_event, turn_frame,
};
use tokio::io::AsyncWriteExt as _;

use crate::{
	auth_backend,
	daemon::{DaemonConfig, DaemonHandle},
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
	/// Run one stateless turn through a running gateway.
	Infer(InferArgs),
	/// Manage provider credentials in the local broker store.
	Auth(AuthArgs),
	/// Manage generated model-catalog data.
	Catalog(CatalogArgs),
	/// Run hardware-accelerated inference in this process.
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

/// Direct gateway inference options.
#[derive(Clone, Debug, Args)]
pub struct InferArgs {
	/// Platform-local endpoint of the running gateway.
	#[arg(long = "endpoint", visible_aliases = ["uds", "pipe"], value_name = "LOCAL_ENDPOINT")]
	pub endpoint: LocalEndpoint,
	/// Catalog model id, alias, or role.
	#[arg(long)]
	pub model:    Str,
	/// User prompt for the stateless turn.
	#[arg(long)]
	pub prompt:   Str,
}

/// Broker command and durable-state options.
#[derive(Clone, Debug, Args)]
pub struct AuthArgs {
	/// OMP data directory containing `broker.db`.
	#[arg(long, value_name = "PATH")]
	pub data_dir: Option<PathBuf>,
	/// Authentication operation.
	#[command(subcommand)]
	pub command:  AuthCommand,
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
	/// Normalize and compress a generated source catalog.
	Import(CatalogImportArgs),
}

/// Catalog import paths.
#[derive(Clone, Debug, Args)]
pub struct CatalogImportArgs {
	/// Generated source JSON.
	#[arg(long, value_name = "JSON")]
	pub source:      PathBuf,
	/// Destination zstd catalog payload.
	#[arg(long, value_name = "ZST")]
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
	/// Generate one complete local response.
	Infer(LocalInferArgs),
}

/// In-process text generation options.
#[derive(Clone, Debug, Args)]
pub struct LocalInferArgs {
	/// Backend selection: `auto`, `foundation`, or a local GGUF path.
	#[arg(long, default_value = "auto", value_name = "BACKEND_OR_GGUF")]
	pub model:  Str,
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
	let config = match args.data_dir {
		Some(data_dir) => DaemonConfig::local(args.endpoint).with_data_dir(data_dir),
		None => DaemonConfig::local(args.endpoint),
	};
	let handle = DaemonHandle::start(config).await?;
	handle.wait().await?;
	Ok(())
}

async fn infer(args: InferArgs) -> crate::Result<()> {
	let channel = omp_llm_gateway::local::connect(&args.endpoint)
		.await
		.map_err(|source| AppError::ConnectGateway { endpoint: args.endpoint.clone(), source })?;
	omp_rpc::handshake(channel.clone(), "omp-cli", &["inference"]).await?;
	let request = ChatRequest::builder()
		.model(args.model)
		.thread(
			Thread::builder()
				.items(vec![user_item(args.prompt)])
				.build(),
		)
		.tools(Vec::new())
		.build();
	let mut open: omp_proto::inference::v1::TurnRequest = request.into();
	open.turn_id = turn_id();
	let frame = TurnFrame { frame: Some(turn_frame::Frame::Open(open)) };
	let mut events = InferenceClient::new(channel)
		.turn(stream::iter([frame]))
		.await?
		.into_inner();
	let mut terminal = false;
	let mut stdout = tokio::io::stdout();
	while let Some(event) = events.next().await {
		match event?.event {
			Some(turn_event::Event::PartDelta(delta)) => stdout.write_all(&delta.chunk).await?,
			Some(turn_event::Event::Outcome(_)) => terminal = true,
			Some(turn_event::Event::Error(error)) => {
				return Err(AppError::InferenceFailed { detail: Str::from(error.detail) });
			},
			_ => {},
		}
	}
	if !terminal {
		return Err(AppError::InferenceStreamUnterminated);
	}
	stdout.write_all(b"\n").await?;
	stdout.flush().await?;
	Ok(())
}

fn user_item(prompt: Str) -> Item {
	Item::builder()
		.seq(0)
		.kind(ItemKind::Message(
			Message::builder()
				.role(Role::User)
				.parts(vec![Part::Text(prompt)])
				.build(),
		))
		.props(Props::default())
		.build()
}

fn turn_id() -> String {
	let now = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_nanos();
	format!("cli-{}-{now}", std::process::id())
}

async fn auth(args: AuthArgs) -> crate::Result<()> {
	validate_auth(&args.command)?;
	let database = data_dir(args.data_dir)?.join("broker.db");
	let backend = auth_backend::open(&database)?;
	let output = broker_cli::run(&backend, &AuthCli { command: args.command }).await?;
	println!("{output}");
	Ok(())
}

const fn validate_auth(command: &AuthCommand) -> crate::Result<()> {
	if let AuthCommand::Migrate(args) = command
		&& args.sqlite.is_none()
		&& args.json_file.is_none()
	{
		return Err(AppError::AuthMigrateArgsRequired);
	}
	Ok(())
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
	if same_path(&args.source, &args.destination) {
		return Err(AppError::SameCatalogSourceAndDestination);
	}
	let input = std::fs::read(&args.source)
		.map_err(|source| AppError::ReadCatalogSource { path: args.source.clone(), source })?;
	let payload = import_catalog_zstd(&input)?;
	if let Some(parent) = args.destination.parent()
		&& !parent.as_os_str().is_empty()
	{
		std::fs::create_dir_all(parent)?;
	}
	std::fs::write(&args.destination, payload).map_err(|source| {
		AppError::WriteCatalogDestination { path: args.destination.clone(), source }
	})?;
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

async fn local_infer(args: LocalInferArgs) -> crate::Result<()> {
	let selection = match args.model.as_str() {
		"auto" => TextSelection::Auto,
		"foundation" => TextSelection::FoundationModels,
		path => TextSelection::Gguf(TextModel::Local(PathBuf::from(path))),
	};
	let inference = Arc::new(Inference::builder().text(selection).build().await?);
	let embedded = Embedded::new(Arc::clone(&inference));
	let request = ChatRequest::builder()
		.model(Str::new_static("local/default"))
		.thread(
			Thread::builder()
				.items(vec![user_item(args.prompt)])
				.build(),
		)
		.tools(Vec::new())
		.build();
	let mut events = embedded.turn(request, None).await?;
	let mut text_parts = BTreeSet::new();
	let mut terminal = false;
	let mut stdout = tokio::io::stdout();
	while let Some(event) = events.next().await {
		match event {
			TurnEvent::PartStart { index, kind: StreamPartKind::Text, .. } => {
				text_parts.insert(index);
			},
			TurnEvent::PartDelta { index, chunk } if text_parts.contains(&index) => {
				stdout.write_all(&chunk).await?;
			},
			TurnEvent::PartEnd { index, .. } => {
				text_parts.remove(&index);
			},
			TurnEvent::Outcome(_) => terminal = true,
			TurnEvent::Error(error) => {
				return Err(AppError::InferenceFailed { detail: error.detail });
			},
			_ => {},
		}
	}
	if !terminal {
		return Err(AppError::LocalInferenceStreamUnterminated);
	}
	stdout.write_all(b"\n").await?;
	stdout.flush().await?;
	inference.shutdown().await?;
	Ok(())
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
				&["omp", "infer", "--endpoint", TEST_ENDPOINT, "--model", "slow", "--prompt", "hello"]
					[..],
				DispatchTarget::Infer,
			),
			(&["omp", "auth", "list"][..], DispatchTarget::Auth),
			(
				&[
					"omp",
					"catalog",
					"import",
					"--source",
					"models.json",
					"--destination",
					"models.json.zst",
				][..],
				DispatchTarget::CatalogImport,
			),
			(&["omp", "local", "infer", "--prompt", "hello"][..], DispatchTarget::LocalInfer),
		];
		for (arguments, expected) in cases {
			assert_eq!(dispatch_target(&parse(arguments).command), expected);
		}
	}

	#[cfg(windows)]
	#[test]
	fn parses_windows_named_pipe_alias_and_uri() {
		for arguments in [
			&["omp", "serve", "--pipe", r"\\.\pipe\omp-cli-test"][..],
			&["omp", "serve", "--endpoint", "npipe://./pipe/omp-cli-test"][..],
		] {
			let Command::Serve(args) = parse(arguments).command else {
				panic!("serve command");
			};
			assert_eq!(args.endpoint.as_path(), Path::new(r"\\.\pipe\omp-cli-test"));
		}
	}

	#[test]
	fn rejects_incomplete_and_conflicting_commands() {
		for arguments in [
			&["omp", "serve"][..],
			&["omp", "infer", "--model", "slow", "--prompt", "hello"][..],
			&["omp", "local", "infer"][..],
			&["omp", "catalog", "import", "--source", "models.json"][..],
		] {
			assert_eq!(
				OmpCli::try_parse_from(arguments)
					.expect_err("command must be rejected")
					.kind(),
				ErrorKind::MissingRequiredArgument
			);
		}
	}

	#[test]
	fn catalog_import_rejects_overwriting_its_source() {
		let args =
			CatalogImportArgs { source: "models.json".into(), destination: "models.json".into() };
		assert!(catalog_import(&args).is_err());
	}

	#[test]
	fn auth_migrate_rejects_an_empty_source_set() {
		let command = parse(&["omp", "auth", "migrate"]).command;
		let Command::Auth(args) = command else {
			panic!("auth command");
		};
		assert!(validate_auth(&args.command).is_err());
	}
}
