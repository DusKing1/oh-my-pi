//! Durable project-chat composition.

use std::{
	path::{Path, PathBuf},
	sync::Arc,
	time::{SystemTime, UNIX_EPOCH},
};

use omp_agent::{
	Agent, AgentSnapshot, AgentState, InProcTurnClient, Journal, RpcTurnClient, TurnClient,
	TurnOptions, WorkspaceInput, project_journal,
};
use omp_core::Str;
use omp_llm_catalog::GrammarBits;
use omp_llm_inference::ToolInputConstraint;
use omp_proto::inference::v1 as inference_pb;
use omp_storage::transcript::{Header, SessionId};
use omp_tool::{LoweringCaps, PromptCaps, Registry};
use thiserror::Error;

use crate::{chat_ui::{self, ChatUiSession}, cli::ChatArgs};

const PROMPT_CAPS: PromptCaps =
	PromptCaps { maximum_parts: 1, maximum_text_bytes: 64 * 1024, media: false };

/// Failures while resolving or running one durable project-chat session.
#[derive(Debug, Error)]
pub enum ChatError {
	/// The requested project root could not be canonicalized.
	#[error("could not resolve project root {path}")]
	Project {
		/// Project path supplied by the caller.
		path: PathBuf,
		/// Filesystem failure.
		#[source]
		source: std::io::Error,
	},
	/// The canonical project path is not a directory.
	#[error("project root is not a directory: {0}")]
	ProjectNotDirectory(PathBuf),
	/// Project-local state failed its owner-only invariant.
	#[error("project state is not owner-only: {path}")]
	InsecureState {
		/// State path that failed validation.
		path: PathBuf,
		/// Filesystem or ownership failure.
		#[source]
		source: std::io::Error,
	},
	/// The requested resume identity is not a canonical ULID.
	#[error("invalid chat session id: {0}")]
	InvalidResume(Str),
	/// The requested durable session does not exist.
	#[error("chat session does not exist: {0}")]
	MissingResume(Str),
	/// The journal header did not match the requested session.
	#[error("chat journal identity does not match session {0}")]
	SessionMismatch(Str),
	/// The journal belongs to a different canonical project root.
	#[error("chat session {session} belongs to a different project")]
	SessionProjectMismatch {
		/// Requested session identity.
		session: Str,
	},
	/// Durable transcript state failed to open, create, or project.
	#[error(transparent)]
	Journal(#[from] omp_agent::JournalError),
	/// A durable transcript could not be projected into canonical replay items.
	#[error(transparent)]
	Projection(#[from] omp_agent::ProjectionError),
	/// The project environment authority failed to start or connect.
	#[error(transparent)]
	Environment(#[from] crate::envd::EnvdError),
	/// The in-process turn authority could not be constructed.
	#[error(transparent)]
	TurnClient(#[from] omp_agent::Error),
	/// A live tool declaration could not be represented on the turn protocol.
	#[error("tool {0} uses a grammar input unsupported by the turn protocol")]
	GrammarTool(Str),
	/// A tool schema could not be encoded for the turn protocol.
	#[error("could not encode tool schema")]
	ToolSchema(#[source] serde_json::Error),
	/// The interactive terminal shell failed.
	#[error("interactive chat shell failed")]
	Ui(#[source] anyhow::Error),
	/// The platform cannot enforce the Phase 3 owner-local environment contract.
	#[error("interactive chat requires Unix owner-local project authorities")]
	UnsupportedPlatform,
}

struct Session {
	id:            Str,
	journal:       Journal,
	initial_items: Vec<omp_proto::thread::v1::Item>,
}

/// Runs one interactive durable project-chat session.
#[cfg(unix)]
pub async fn run(args: ChatArgs) -> crate::Result<()> {
	let root = canonical_project(&args.project)?;
	let state_dir = root.join(".omp");
	let sessions_dir = state_dir.join("sessions");
	secure_owner_directory(&state_dir)?;
	secure_owner_directory(&sessions_dir)?;

	let environment = crate::envd::ProjectEnvironment::connect_or_start(
		&root,
		&state_dir,
		&state_dir.join("env.sock"),
		&state_dir.join("docserver.sock"),
		args.py_eval,
	)
	.await?;
	let env = environment.client().clone();

	let registry = environment.registry();
	let catalog = omp_llm_catalog::snapshot::Catalog::try_embedded()?;
	let session = open_session(&root, &sessions_dir, args.resume.as_ref(), registry.as_ref())?;
	let snapshot = agent_snapshot(&args.model, &root, &session.id, Arc::clone(&registry))?;
	let context_window = model_context_window(catalog, &args.model);
	let state = AgentState::new(snapshot);

	if let Some(endpoint) = args.gateway {
		let channel = omp_rpc::uds::connect(endpoint.as_path()).await.map_err(|source| {
			crate::AppError::ConnectGateway { endpoint: endpoint.clone(), source }
		})?;
		run_ui(RpcTurnClient::new(channel), env, state, session, context_window).await?;
	} else {
		let data_dir = crate::cli::data_dir(None)?;
		let (_, inference) =
			crate::daemon::production_inference(&data_dir, Arc::clone(&registry)).await?;
		let client = InProcTurnClient::new(inference).await.map_err(ChatError::from)?;
		run_ui(client, env, state, session, context_window).await?;
	}

	// `environment` is deliberately retained until the agent and UI have been
	// dropped. Its Drop implementation only stops authorities this process
	// autostarted; pre-existing project daemons remain untouched.
	drop(environment);
	Ok(())
}

/// Reports the platform limitation before touching project state.
#[cfg(not(unix))]
pub async fn run(_args: ChatArgs) -> crate::Result<()> {
	Err(ChatError::UnsupportedPlatform.into())
}

async fn run_ui<C: TurnClient + 'static>(
	client: C,
	env: omp_env::EnvClient,
	state: AgentState,
	session: Session,
	context_window: Option<u64>,
) -> Result<(), ChatError> {
	let agent = Agent::new(client, env, state, session.journal, PROMPT_CAPS);
	chat_ui::run(
		agent,
		ChatUiSession {
			session_id: session.id,
			initial_items: session.initial_items,
			context_window,
		},
	)
	.await
	.map_err(ChatError::Ui)
}

fn model_context_window(
	catalog: &omp_llm_catalog::snapshot::Catalog,
	model: &Str,
) -> Option<u64> {
	let key = omp_llm_catalog::ModelKey::from(model.as_str());
	catalog
		.model(&key)
		.or_else(|| catalog.resolve_alias(model.as_str()))
		.and_then(|spec| spec.limits.context_window)
}


fn canonical_project(path: &Path) -> Result<PathBuf, ChatError> {
	let root = std::fs::canonicalize(path)
		.map_err(|source| ChatError::Project { path: path.to_owned(), source })?;
	if !root.is_dir() {
		return Err(ChatError::ProjectNotDirectory(root));
	}
	Ok(root)
}

fn open_session(
	root: &Path,
	sessions_dir: &Path,
	resume: Option<&Str>,
	registry: &Registry,
) -> Result<Session, ChatError> {
	let id = match resume {
		Some(id) => strict_session_id(id)?,
		None => Str::from(ulid::Ulid::generate().to_string()),
	};
	let path = sessions_dir.join(format!("{}.jsonl", id.as_str()));
	let journal = if resume.is_some() {
		secure_session_file(&path).map_err(|source| {
			if source.kind() == std::io::ErrorKind::NotFound {
				ChatError::MissingResume(id.clone())
			} else {
				ChatError::InsecureState { path: path.clone(), source }
			}
		})?;
		let journal = Journal::open(&path)?;
		let log = journal.load()?;
		if log.header().id.0 != id {
			return Err(ChatError::SessionMismatch(id));
		}
		if log.header().cwd != root {
			return Err(ChatError::SessionProjectMismatch { session: id });
		}
		journal
	} else {
		let journal = Journal::create(
			&path,
			&Header {
				v: 4,
				id: SessionId(id.clone()),
				created: now_ms(),
				cwd: root.to_owned(),
			},
		)?;
		if let Err(source) = set_owner_file_permissions(&path) {
			drop(journal);
			let _ = std::fs::remove_file(&path);
			return Err(ChatError::InsecureState { path, source });
		}
		journal
	};
	let initial_items = project_journal(&journal.load()?, registry, &PROMPT_CAPS)?.items;
	Ok(Session { id, journal, initial_items })
}

fn strict_session_id(id: &Str) -> Result<Str, ChatError> {
	let parsed = id
		.as_str()
		.parse::<ulid::Ulid>()
		.map_err(|_| ChatError::InvalidResume(id.clone()))?;
	if parsed.to_string() != id.as_str() {
		return Err(ChatError::InvalidResume(id.clone()));
	}
	Ok(id.clone())
}

fn agent_snapshot(
	model: &Str,
	root: &Path,
	session_id: &Str,
	registry: Arc<Registry>,
) -> Result<AgentSnapshot, ChatError> {
	let advertised = registry.advertise(LoweringCaps {
		strict_schema: true,
		grammar: GrammarBits::LARK | GrammarBits::REGEX | GrammarBits::EBNF,
	});
	let mut enabled_tools = Vec::with_capacity(advertised.len());
	let mut tools = Vec::with_capacity(advertised.len());
	for tool in advertised {
		enabled_tools.push(tool.identity.name.clone());
		let (schema_json, strict) = match tool.definition.input {
			ToolInputConstraint::JsonSchema { parameters, strict } => (
				serde_json::to_vec(parameters.as_value()).map_err(ChatError::ToolSchema)?,
				strict,
			),
			ToolInputConstraint::Grammar(_) => {
				return Err(ChatError::GrammarTool(tool.identity.name));
			},
		};
		tools.push(inference_pb::ToolDef {
			name: tool.definition.name.to_string(),
			description: tool.definition.description.map_or_else(String::new, |value| value.to_string()),
			schema_json: schema_json.into(),
			strict: Some(strict),
		});
	}
	let turn = TurnOptions {
		context_id: Some(session_id.clone()),
		params: inference_pb::ChatParams {
			model: model.to_string(),
			tools,
			..inference_pb::ChatParams::default()
		},
		..TurnOptions::default()
	};
	let mut snapshot = AgentSnapshot::new(turn, WorkspaceInput::new(root, Arc::from([])), registry);
	snapshot.enabled_tools = enabled_tools.into();
	Ok(snapshot)
}

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis()
		.try_into()
		.unwrap_or(u64::MAX)
}

#[cfg(unix)]
fn secure_owner_directory(path: &Path) -> Result<(), ChatError> {
	use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

	match std::fs::create_dir(path) {
		Ok(()) => std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
			.map_err(|source| ChatError::InsecureState { path: path.to_owned(), source })?,
		Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {},
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
			let parent = path.parent().ok_or_else(|| ChatError::InsecureState {
				path: path.to_owned(),
				source: std::io::Error::new(std::io::ErrorKind::InvalidInput, "state path has no parent"),
			})?;
			secure_owner_directory(parent)?;
			std::fs::create_dir(path).map_err(|source| ChatError::InsecureState {
				path: path.to_owned(),
				source,
			})?;
			std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).map_err(
				|source| ChatError::InsecureState { path: path.to_owned(), source },
			)?;
		},
		Err(source) => return Err(ChatError::InsecureState { path: path.to_owned(), source }),
	}
	let metadata = std::fs::symlink_metadata(path)
		.map_err(|source| ChatError::InsecureState { path: path.to_owned(), source })?;
	if !metadata.is_dir()
		|| metadata.uid() != nix::unistd::geteuid().as_raw()
		|| metadata.mode() & 0o077 != 0
	{
		return Err(ChatError::InsecureState {
			path: path.to_owned(),
			source: std::io::Error::new(
				std::io::ErrorKind::PermissionDenied,
				"state directory must be owner-only and owned by the current user",
			),
		});
	}
	Ok(())
}

#[cfg(unix)]
fn set_owner_file_permissions(path: &Path) -> std::io::Result<()> {
	use std::os::unix::fs::PermissionsExt as _;

	std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
}

#[cfg(unix)]
fn secure_session_file(path: &Path) -> std::io::Result<()> {
	use std::os::unix::fs::MetadataExt as _;

	let metadata = std::fs::symlink_metadata(path)?;
	if !metadata.is_file()
		|| metadata.uid() != nix::unistd::geteuid().as_raw()
		|| metadata.mode() & 0o077 != 0
	{
		return Err(std::io::Error::new(
			std::io::ErrorKind::PermissionDenied,
			"session journal must be an owner-only regular file",
		));
	}
	Ok(())
}
