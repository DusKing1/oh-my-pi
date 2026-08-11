//! Mountable `omp auth` command parser and rendering.

use std::{
	fmt::Write as _,
	path::{Path, PathBuf},
	sync::Arc,
	time::{SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use clap::{Args, Parser, Subcommand};
use futures::future::BoxFuture;
use jiff::Timestamp;
use omp_core::Str;
use rusqlite::Connection;
use serde_json::{Value, json};
use thiserror::Error;

use crate::{
	oauth::{LoginPrompt, OAuthEngine},
	store::{CredentialFilter, CredentialKind, CredentialMeta, CredentialState, Store, UsageReport},
	usage::UsageManager,
};

/// Parser for the daemon-mounted `omp auth` command tree.
#[derive(Clone, Debug, Parser)]
#[command(name = "auth", about = "Manage provider credentials")]
pub struct AuthCli {
	/// Authentication operation.
	#[command(subcommand)]
	pub command: AuthCommand,
}

/// Authentication subcommands.
#[derive(Clone, Debug, Subcommand)]
pub enum AuthCommand {
	/// Start an interactive provider login.
	Login(LoginArgs),
	/// Remove credentials by id or provider.
	Logout(LogoutArgs),
	/// List credential metadata.
	List(ListArgs),
	/// Refresh one OAuth credential.
	Refresh(RefreshArgs),
	/// Import credentials from previous installations.
	Migrate(MigrateArgs),
	/// Show provider quota usage.
	Usage(UsageArgs),
}

/// Arguments for `auth login`.
#[derive(Clone, Debug, Args)]
pub struct LoginArgs {
	/// Provider to log into; omitted to show the daemon's provider chooser.
	pub provider: Option<Str>,
}

/// Arguments for `auth logout`.
#[derive(Clone, Debug, Args)]
pub struct LogoutArgs {
	/// Numeric credential id or provider identifier.
	pub selector: Str,
}

/// Arguments for `auth list`.
#[derive(Clone, Debug, Args)]
pub struct ListArgs {
	/// Restrict output to one provider.
	#[arg(long)]
	pub provider: Option<Str>,
	/// Emit structured JSON.
	#[arg(long)]
	pub json:     bool,
}

/// Arguments for `auth refresh`.
#[derive(Clone, Debug, Args)]
pub struct RefreshArgs {
	/// Credential database id.
	pub id: u64,
}

/// Arguments for `auth migrate`.
#[derive(Clone, Debug, Default, Args)]
pub struct MigrateArgs {
	/// Previous implementation's `agent.db` path.
	#[arg(long, value_name = "PATH")]
	pub sqlite:    Option<PathBuf>,
	/// CLIProxyAPI-style JSON credential file.
	#[arg(long = "json-file", value_name = "PATH")]
	pub json_file: Option<PathBuf>,
}

/// Arguments for `auth usage`.
#[derive(Clone, Debug, Args)]
pub struct UsageArgs {
	/// Restrict output to one provider.
	#[arg(long)]
	pub provider: Option<Str>,
	/// Emit structured JSON.
	#[arg(long)]
	pub json:     bool,
	/// Read durable history instead of the latest snapshots.
	#[arg(long)]
	pub history:  bool,
	/// History lookback in days.
	#[arg(long, default_value_t = 7)]
	pub days:     u32,
}

/// Secret-ingress record produced by migration readers.
///
/// This type intentionally implements neither `Debug` nor `Serialize`, so
/// secret material cannot accidentally enter CLI output.
pub enum CredentialImport {
	/// A provider API key.
	ApiKey {
		/// Canonical provider catalog id.
		provider: Str,
		/// Stable non-secret account identity.
		identity: Str,
		/// API-key bytes.
		secret:   Bytes,
	},
	/// An OAuth credential and its provider account metadata.
	OAuth {
		/// Canonical provider catalog id.
		provider:      Str,
		/// Stable non-secret account identity.
		identity:      Str,
		/// OAuth access token bytes.
		access_token:  Bytes,
		/// OAuth refresh token bytes when supplied by the source.
		refresh_token: Bytes,
		/// Non-secret provider account properties retained across refreshes.
		props:         Value,
		/// Absolute access-token expiry, or zero when unknown.
		expires_at_ms: u64,
	},
}

/// Daemon operations consumed by the mountable CLI.
pub trait AuthCliBackend: Send + Sync {
	/// Begins login and returns client-safe instructions.
	fn login<'a>(&'a self, provider: Option<&'a str>) -> BoxFuture<'a, Result<Str, CliError>>;
	/// Deletes credentials selected by numeric id or provider.
	fn logout<'a>(&'a self, selector: &'a str) -> BoxFuture<'a, Result<usize, CliError>>;
	/// Lists client-safe credential metadata.
	fn list<'a>(
		&'a self,
		provider: Option<&'a str>,
	) -> BoxFuture<'a, Result<Vec<CredentialMeta>, CliError>>;
	/// Refreshes one OAuth credential and returns its metadata.
	fn refresh(&self, id: u64) -> BoxFuture<'_, Result<CredentialMeta, CliError>>;
	/// Imports one credential through the daemon's one-way secret-ingress
	/// boundary.
	fn import_credential(
		&self,
		import: CredentialImport,
	) -> BoxFuture<'_, Result<CredentialMeta, CliError>>;
	/// Returns current snapshots or durable history reports.
	fn usage<'a>(
		&'a self,
		provider: Option<&'a str>,
		history: bool,
		since_ms: u64,
	) -> BoxFuture<'a, Result<Vec<UsageReport>, CliError>>;
	/// Supplies the daemon clock for stable command behavior and tests.
	fn now_ms(&self) -> u64;
}

/// Concrete in-process CLI backend over the daemon's broker state.
pub struct BrokerCliBackend {
	store: Arc<Store>,
	oauth: Option<Arc<OAuthEngine>>,
	usage: UsageManager,
}

impl BrokerCliBackend {
	/// Creates a backend and takes ownership of the configured OAuth engine.
	#[must_use]
	pub fn new(store: Arc<Store>, oauth: Option<OAuthEngine>, usage: UsageManager) -> Self {
		Self { store, oauth: oauth.map(Arc::new), usage }
	}

	/// Creates a backend sharing the daemon's OAuth engine.
	#[must_use]
	pub const fn with_shared_oauth(
		store: Arc<Store>,
		oauth: Option<Arc<OAuthEngine>>,
		usage: UsageManager,
	) -> Self {
		Self { store, oauth, usage }
	}

	/// Returns the shared OAuth engine handle, when OAuth is configured.
	#[must_use]
	pub fn oauth(&self) -> Option<Arc<OAuthEngine>> {
		self.oauth.clone()
	}

	fn oauth_ref(&self) -> Result<&OAuthEngine, CliError> {
		self
			.oauth
			.as_deref()
			.ok_or_else(|| CliError::Backend("OAuth is not configured".into()))
	}
}

impl AuthCliBackend for BrokerCliBackend {
	fn login<'a>(&'a self, provider: Option<&'a str>) -> BoxFuture<'a, Result<Str, CliError>> {
		Box::pin(async move {
			let oauth = self.oauth_ref()?;
			let Some(provider) = provider else {
				let mut picker = String::from("Select a login provider:\n");
				for (index, provider) in oauth.providers().enumerate() {
					let _ = writeln!(picker, "  {:>2}) {provider}", index + 1);
				}
				picker.push_str("Choose one with `omp auth login <provider-or-number>`.");
				return Ok(picker.into());
			};
			let provider = if let Ok(choice) = provider.parse::<usize>() {
				if choice == 0 {
					return Err(CliError::Backend("login choice must be at least 1".into()));
				}
				oauth
					.providers()
					.nth(choice - 1)
					.ok_or_else(|| CliError::Backend("login choice is out of range".into()))?
			} else {
				provider
			};
			let start = oauth
				.begin_login(provider, self.now_ms())
				.await
				.map_err(backend_error)?;
			let prompt = match start.prompt {
				LoginPrompt::Browse { url, loopback } => {
					format!("flow={} open={} loopback={loopback}", start.flow_id, url)
				},
				LoginPrompt::Device { user_code, verification_url, interval_secs } => format!(
					"flow={} open={} code={} poll={}s",
					start.flow_id, verification_url, user_code, interval_secs
				),
				LoginPrompt::Paste { url } if start.provider == "perplexity" => format!(
					"flow={} open={} then submit sealed JSON {{\"email\":\"…\",\"otp\":\"…\"}}",
					start.flow_id, url
				),
				LoginPrompt::Paste { url } => {
					format!("flow={} open={} then submit the returned value", start.flow_id, url)
				},
			};
			Ok(prompt.into())
		})
	}

	fn logout<'a>(&'a self, selector: &'a str) -> BoxFuture<'a, Result<usize, CliError>> {
		Box::pin(async move {
			let now_ms = self.now_ms();
			if let Ok(id) = selector.parse::<u64>() {
				return self
					.store
					.delete_credential(id, now_ms)
					.map(usize::from)
					.map_err(backend_error);
			}
			let credentials = self
				.store
				.list_credentials(&CredentialFilter {
					provider: Some(selector),
					now_ms,
					..CredentialFilter::default()
				})
				.map_err(backend_error)?;
			let mut removed = 0;
			for credential in credentials {
				if self
					.store
					.delete_credential(credential.id, now_ms)
					.map_err(backend_error)?
				{
					removed += 1;
				}
			}
			Ok(removed)
		})
	}

	fn list<'a>(
		&'a self,
		provider: Option<&'a str>,
	) -> BoxFuture<'a, Result<Vec<CredentialMeta>, CliError>> {
		Box::pin(async move {
			self
				.store
				.list_credentials(&CredentialFilter {
					provider,
					now_ms: self.now_ms(),
					..CredentialFilter::default()
				})
				.map_err(backend_error)
		})
	}

	fn refresh(&self, id: u64) -> BoxFuture<'_, Result<CredentialMeta, CliError>> {
		Box::pin(async move {
			self
				.oauth_ref()?
				.refresh_credential(id, self.now_ms())
				.await
				.map_err(backend_error)
		})
	}

	fn import_credential(
		&self,
		import: CredentialImport,
	) -> BoxFuture<'_, Result<CredentialMeta, CliError>> {
		Box::pin(async move {
			match import {
				CredentialImport::ApiKey { provider, identity, secret } => self
					.store
					.import_api_key(&provider, &identity, &secret, self.now_ms())
					.map_err(backend_error),
				CredentialImport::OAuth {
					provider,
					identity,
					access_token,
					refresh_token,
					props,
					expires_at_ms,
				} => self
					.store
					.import_oauth_material(
						&provider,
						&identity,
						&access_token,
						(!refresh_token.is_empty()).then_some(refresh_token.as_ref()),
						&props,
						expires_at_ms,
						self.now_ms(),
					)
					.map_err(backend_error),
			}
		})
	}

	fn usage<'a>(
		&'a self,
		provider: Option<&'a str>,
		history: bool,
		since_ms: u64,
	) -> BoxFuture<'a, Result<Vec<UsageReport>, CliError>> {
		Box::pin(async move {
			if !history {
				return self
					.usage
					.get_usage(provider, None, false, self.now_ms())
					.await
					.map_err(backend_error);
			}
			let credentials = self
				.store
				.list_credentials(&CredentialFilter {
					provider,
					now_ms: self.now_ms(),
					..CredentialFilter::default()
				})
				.map_err(backend_error)?;
			let mut reports = Vec::new();
			for credential in credentials {
				reports.extend(
					self
						.store
						.usage_history(credential.id, since_ms, 0)
						.map_err(backend_error)?
						.into_iter()
						.map(|entry| entry.report),
				);
			}
			Ok(reports)
		})
	}

	fn now_ms(&self) -> u64 {
		SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.unwrap_or_default()
			.as_millis()
			.try_into()
			.unwrap_or(u64::MAX)
	}
}

fn backend_error(error: impl std::fmt::Display) -> CliError {
	CliError::Backend(error.to_string().into())
}

/// CLI execution failure.
#[derive(Debug, Error)]
pub enum CliError {
	/// A migration file could not be read.
	#[error("migration I/O failed: {0}")]
	Io(#[from] std::io::Error),
	/// A legacy SQLite store could not be read.
	#[error("migration database failed: {0}")]
	Sqlite(#[from] rusqlite::Error),
	/// A JSON document was invalid.
	#[error("migration JSON failed: {0}")]
	Json(#[from] serde_json::Error),
	/// A migration record omitted the secret required by its credential kind.
	#[error("{kind} record for {provider} has no secret")]
	MissingSecret {
		/// Canonical provider id.
		provider: Str,
		/// Human-readable credential kind.
		kind:     &'static str,
	},
	/// The mounted daemon operation failed.
	#[error("{0}")]
	Backend(Str),
}

/// Runs a parsed command against a daemon adapter and returns printable output.
pub async fn run(backend: &dyn AuthCliBackend, cli: &AuthCli) -> Result<String, CliError> {
	match &cli.command {
		AuthCommand::Login(args) => Ok(backend.login(args.provider.as_deref()).await?.to_string()),
		AuthCommand::Logout(args) => {
			let removed = backend.logout(&args.selector).await?;
			Ok(format!("removed {removed} credential(s)"))
		},
		AuthCommand::List(args) => {
			let credentials = backend.list(args.provider.as_deref()).await?;
			if args.json {
				render_credentials_json(&credentials)
			} else {
				Ok(render_credentials_table(&credentials))
			}
		},
		AuthCommand::Refresh(args) => {
			let credential = backend.refresh(args.id).await?;
			Ok(render_credentials_table(&[credential]))
		},
		AuthCommand::Migrate(args) => {
			let imports = load_migration(args)?;
			let count = imports.len();
			for import in imports {
				backend.import_credential(import).await?;
			}
			Ok(format!("imported {count} credential(s)"))
		},
		AuthCommand::Usage(args) => {
			let day_ms = 86_400_000_u64;
			let since_ms = backend
				.now_ms()
				.saturating_sub(u64::from(args.days).saturating_mul(day_ms));
			let reports = backend
				.usage(args.provider.as_deref(), args.history, since_ms)
				.await?;
			if args.json {
				Ok(serde_json::to_string_pretty(&reports)?)
			} else {
				Ok(render_usage_table(&reports))
			}
		},
	}
}

fn render_credentials_json(credentials: &[CredentialMeta]) -> Result<String, CliError> {
	let values: Vec<Value> = credentials.iter().map(credential_json).collect();
	Ok(serde_json::to_string_pretty(&values)?)
}

fn credential_json(meta: &CredentialMeta) -> Value {
	json!({
		"id": meta.id,
		"provider": meta.provider,
		"kind": kind_name(meta.kind),
		"identity": meta.identity,
		"state": state_name(meta.state),
		"expires_at_ms": meta.expires_at_ms,
		"created_at_ms": meta.created_at_ms,
		"updated_at_ms": meta.updated_at_ms,
		"disabled_cause": meta.disabled_cause,
	})
}

fn render_credentials_table(credentials: &[CredentialMeta]) -> String {
	let mut output = String::from("ID  PROVIDER  KIND  STATE  IDENTITY\n");
	for meta in credentials {
		let _ = writeln!(
			output,
			"{}  {}  {}  {}  {}",
			meta.id,
			meta.provider,
			kind_name(meta.kind),
			state_name(meta.state),
			meta.identity
		);
	}
	output
}

fn render_usage_table(reports: &[UsageReport]) -> String {
	let mut output = String::from("ID  PROVIDER  PLAN  WINDOW  USED  RESETS\n");
	for report in reports {
		if report.windows.is_empty() {
			let _ = writeln!(
				output,
				"{}  {}  {}  -  -  -",
				report.credential_id, report.provider, report.plan
			);
		}
		for window in &report.windows {
			let _ = writeln!(
				output,
				"{}  {}  {}  {}  {:.1}%  {}",
				report.credential_id,
				report.provider,
				report.plan,
				window.label,
				window.used_percent,
				window.resets_at_ms
			);
		}
	}
	output
}

const fn kind_name(kind: CredentialKind) -> &'static str {
	match kind {
		CredentialKind::ApiKey => "api-key",
		CredentialKind::OAuth => "oauth",
		CredentialKind::Aws => "aws",
	}
}

const fn state_name(state: CredentialState) -> &'static str {
	match state {
		CredentialState::Active => "active",
		CredentialState::Blocked => "blocked",
		CredentialState::Disabled => "disabled",
		CredentialState::Expired => "expired",
	}
}

fn load_migration(args: &MigrateArgs) -> Result<Vec<CredentialImport>, CliError> {
	let mut imports = Vec::new();
	if let Some(path) = &args.sqlite {
		imports.extend(load_sqlite(path)?);
	}
	if let Some(path) = &args.json_file {
		imports.extend(load_json(path)?);
	}
	Ok(imports)
}

fn load_sqlite(path: &Path) -> Result<Vec<CredentialImport>, CliError> {
	let connection = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
	let columns: Vec<String> = {
		let mut statement = connection.prepare("PRAGMA table_info(auth_credentials)")?;
		let rows = statement.query_map([], |row| row.get(1))?;
		rows.collect::<rusqlite::Result<_>>()?
	};
	let active = if columns.iter().any(|column| column == "disabled_cause") {
		"disabled_cause IS NULL"
	} else if columns.iter().any(|column| column == "disabled") {
		"disabled = 0"
	} else {
		"1 = 1"
	};
	let identity = if columns.iter().any(|column| column == "identity_key") {
		"identity_key"
	} else {
		"NULL"
	};
	let sql = format!(
		"SELECT id, provider, credential_type, data, {identity} FROM auth_credentials WHERE \
		 credential_type IN ('oauth', 'api_key') AND {active} ORDER BY id"
	);
	let mut statement = connection.prepare(&sql)?;
	let rows = statement.query_map([], |row| {
		Ok((
			row.get::<_, u64>(0)?,
			row.get::<_, String>(1)?,
			row.get::<_, String>(2)?,
			row.get::<_, String>(3)?,
			row.get::<_, Option<String>>(4)?,
		))
	})?;
	let mut imports = Vec::new();
	for row in rows {
		let (id, provider, kind, data, identity) = row?;
		let fallback = identity.unwrap_or_else(|| format!("imported-{id}"));
		imports.push(import_from_value(
			&provider,
			Some(&kind),
			Some(&fallback),
			&serde_json::from_str(&data)?,
		)?);
	}
	Ok(imports)
}

fn load_json(path: &Path) -> Result<Vec<CredentialImport>, CliError> {
	if path.is_dir() {
		let mut paths: Vec<PathBuf> = std::fs::read_dir(path)?
			.filter_map(Result::ok)
			.map(|entry| entry.path())
			.filter(|entry| {
				entry
					.extension()
					.is_some_and(|extension| extension == "json")
			})
			.collect();
		paths.sort_unstable();
		let mut imports = Vec::new();
		for path in paths {
			let root: Value = serde_json::from_slice(&std::fs::read(path)?)?;
			imports.extend(parse_migration_json(&root)?);
		}
		return Ok(imports);
	}
	let root: Value = serde_json::from_slice(&std::fs::read(path)?)?;
	parse_migration_json(&root)
}

fn parse_migration_json(root: &Value) -> Result<Vec<CredentialImport>, CliError> {
	if is_disabled(root) {
		return Ok(Vec::new());
	}
	if let Some(provider) = root.get("provider").and_then(Value::as_str) {
		let kind = root
			.get("credential_type")
			.or_else(|| root.get("type"))
			.and_then(Value::as_str);
		return Ok(vec![import_from_value(
			provider,
			kind,
			migration_identity(root).as_deref(),
			root.get("data").unwrap_or(root),
		)?]);
	}
	if let Some(provider) = root.get("type").and_then(Value::as_str)
		&& !matches!(provider, "oauth" | "api_key")
	{
		return Ok(vec![import_from_value(
			provider,
			None,
			migration_identity(root).as_deref(),
			root,
		)?]);
	}
	let records = root.get("credentials").unwrap_or(root);
	let Some(providers) = records.as_object() else {
		return Ok(Vec::new());
	};
	let mut imports = Vec::new();
	for (provider, value) in providers {
		let items = value
			.as_array()
			.map_or_else(|| std::slice::from_ref(value), Vec::as_slice);
		for item in items {
			if is_disabled(item) {
				continue;
			}
			let kind = item
				.get("credential_type")
				.or_else(|| item.get("type"))
				.and_then(Value::as_str);
			imports.push(import_from_value(
				provider,
				kind,
				migration_identity(item).as_deref(),
				item.get("data").unwrap_or(item),
			)?);
		}
	}
	Ok(imports)
}

fn import_from_value(
	provider: &str,
	kind: Option<&str>,
	identity_hint: Option<&str>,
	value: &Value,
) -> Result<CredentialImport, CliError> {
	let canonical = canonical_provider(provider);
	let identity = identity_hint
		.map(ToOwned::to_owned)
		.or_else(|| {
			string_field(value, &[
				"email",
				"accountId",
				"account_id",
				"projectId",
				"project_id",
				"orgId",
				"org_id",
				"user",
			])
		})
		.unwrap_or_else(|| "imported".to_owned());
	if kind == Some("api_key")
		|| (kind.is_none()
			&& string_field(value, &["key", "apiKey", "api_key"]).is_some()
			&& string_field(value, &["accessToken", "access_token", "access"]).is_none())
	{
		let secret =
			string_field(value, &["key", "apiKey", "api_key", "token"]).ok_or_else(|| {
				CliError::MissingSecret { provider: Str::new(canonical), kind: "API-key" }
			})?;
		return Ok(CredentialImport::ApiKey {
			provider: Str::new(canonical),
			identity: Str::new(identity),
			secret:   Bytes::from(secret),
		});
	}
	let access =
		string_field(value, &["accessToken", "access_token", "access", "token"]).ok_or_else(
			|| CliError::MissingSecret { provider: Str::new(canonical), kind: "OAuth" },
		)?;
	let refresh =
		string_field(value, &["refreshToken", "refresh_token", "refresh"]).unwrap_or_default();
	let expires_at_ms =
		number_field(value, &["expiresAt", "expires_at", "expires_at_ms", "expires"])
			.or_else(|| {
				string_field(value, &["expired"])
					.and_then(|raw| raw.parse::<Timestamp>().ok())
					.and_then(|timestamp| timestamp.as_millisecond().try_into().ok())
			})
			.unwrap_or(0);
	Ok(CredentialImport::OAuth {
		provider: Str::new(canonical),
		identity: Str::new(identity),
		access_token: Bytes::from(access),
		refresh_token: Bytes::from(refresh),
		props: account_props(value),
		expires_at_ms,
	})
}
fn migration_identity(value: &Value) -> Option<String> {
	string_field(value, &["identity_key", "identity"]).or_else(|| {
		value
			.get("id")
			.and_then(Value::as_u64)
			.map(|id| format!("imported-{id}"))
	})
}

fn is_disabled(value: &Value) -> bool {
	value.get("disabled").and_then(Value::as_bool) == Some(true)
		|| value
			.get("disabled_cause")
			.is_some_and(|cause| !cause.is_null())
}

fn account_props(value: &Value) -> Value {
	const NAMES: &[&str] = &[
		"email",
		"accountId",
		"account_id",
		"projectId",
		"project_id",
		"orgId",
		"org_id",
		"orgName",
		"enterpriseUrl",
		"apiEndpoint",
	];
	let mut props = serde_json::Map::new();
	for name in NAMES {
		if let Some(item) = value.get(*name)
			&& (item.is_string() || item.is_number() || item.is_boolean())
		{
			props.insert((*name).to_owned(), item.clone());
		}
	}
	Value::Object(props)
}

fn string_field(value: &Value, names: &[&str]) -> Option<String> {
	names
		.iter()
		.find_map(|name| value.get(*name)?.as_str().map(ToOwned::to_owned))
}

fn number_field(value: &Value, names: &[&str]) -> Option<u64> {
	names.iter().find_map(|name| value.get(*name)?.as_u64())
}

fn canonical_provider(provider: &str) -> &str {
	match provider {
		"claude" => "anthropic",
		"codex" => "openai-codex",
		"gemini" => "google-gemini-cli",
		"antigravity" => "google-antigravity",
		"gemini-cli" => "google-gemini-cli",
		other => other,
	}
}

#[cfg(test)]
mod tests {
	use smallvec::SmallVec;

	use super::*;

	struct NoUsageHttp;

	impl crate::usage::UsageHttp for NoUsageHttp {
		fn send(
			&self,
			_request: http::Request<Bytes>,
		) -> BoxFuture<'_, Result<crate::usage::UsageHttpResponse, crate::usage::UsageError>> {
			panic!("credential migration must not perform usage HTTP requests")
		}
	}

	fn populated_meta() -> CredentialMeta {
		CredentialMeta {
			id:             9,
			provider:       Str::new("anthropic"),
			kind:           CredentialKind::OAuth,
			identity:       Str::new("person@example.test"),
			state:          CredentialState::Active,
			blocks:         SmallVec::new(),
			disabled_cause: Str::new(""),
			expires_at_ms:  123,
			created_at_ms:  1,
			updated_at_ms:  2,
		}
	}

	#[test]
	fn migration_maps_legacy_provider_names() {
		for (legacy, canonical) in [
			("claude", "anthropic"),
			("codex", "openai-codex"),
			("gemini", "google-gemini-cli"),
			("antigravity", "google-antigravity"),
		] {
			let root = json!({ "type": legacy, "access_token": "token", "refresh_token": "refresh" });
			let imports = parse_migration_json(&root).expect("imports");
			assert_eq!(imports.len(), 1);
			assert!(matches!(
				&imports[0],
				CredentialImport::OAuth { provider, .. } if provider == canonical
			));
		}
	}

	#[tokio::test]
	async fn mixed_pi_sqlite_migrates_through_production_backend_without_secret_egress() {
		let directory = tempfile::tempdir().expect("tempdir");
		let legacy_path = directory.path().join("agent.db");
		let connection = Connection::open(&legacy_path).expect("legacy database");
		connection
			.execute_batch(
				"CREATE TABLE auth_credentials (
					id INTEGER PRIMARY KEY,
					provider TEXT NOT NULL,
					credential_type TEXT NOT NULL,
					data TEXT NOT NULL,
					identity_key TEXT,
					disabled_cause TEXT
				);
				INSERT INTO auth_credentials VALUES
					(41, 'claude', 'oauth',
					 '{\"access\":\"oauth-secret\",\"refresh\":\"refresh-secret\",\"expires\":77,
					   \"email\":\"person@example.test\",\"accountId\":\"acct-7\",\"orgId\":\"org-2\"}',
					 'oauth:legacy-account-41', NULL),
					(73, 'openai', 'api_key',
					 '{\"key\":\"api-secret\",\"source\":\"login\"}', 'api:legacy-key-73', NULL),
					(99, 'xai', 'api_key', '{\"key\":\"disabled-secret\"}', 'api:disabled', 'logout');",
			)
			.expect("legacy fixture");
		drop(connection);

		let store = Arc::new(Store::open(directory.path().join("omp.db")).expect("OMP store"));
		let usage = UsageManager::new(Arc::clone(&store), Arc::new(NoUsageHttp));
		let backend = BrokerCliBackend::new(Arc::clone(&store), None, usage);
		let cli = AuthCli::parse_from([
			"auth",
			"migrate",
			"--sqlite",
			legacy_path.to_str().expect("UTF-8 fixture path"),
		]);

		let first_output = run(&backend, &cli).await.expect("production migration");
		assert_eq!(first_output, "imported 2 credential(s)");
		let first_cursor = store.cursor().expect("first cursor");
		let first = store
			.list_credentials(&CredentialFilter { now_ms: 1_000, ..CredentialFilter::default() })
			.expect("imported metadata");
		assert_eq!(first.len(), 2);
		let oauth = first
			.iter()
			.find(|meta| meta.kind == CredentialKind::OAuth)
			.expect("OAuth credential");
		let api_key = first
			.iter()
			.find(|meta| meta.kind == CredentialKind::ApiKey)
			.expect("API-key credential");
		assert_eq!(oauth.provider, "anthropic");
		assert_eq!(oauth.identity, "oauth:legacy-account-41");
		assert_eq!(oauth.expires_at_ms, 77);
		assert_eq!(api_key.provider, "openai");
		assert_eq!(api_key.identity, "api:legacy-key-73");
		assert_ne!(oauth.id, api_key.id);
		let oauth_generation = store
			.lease(oauth.id)
			.expect("OAuth lease")
			.expect("active OAuth");
		let api_key_generation = store
			.lease(api_key.id)
			.expect("API-key lease")
			.expect("active API key");
		assert_ne!(oauth_generation.generation(), api_key_generation.generation());
		let props = store
			.oauth_props(oauth.id)
			.expect("OAuth props")
			.expect("account props");
		assert_eq!(props["email"], "person@example.test");
		assert_eq!(props["accountId"], "acct-7");
		assert_eq!(props["orgId"], "org-2");

		let second_output = run(&backend, &cli)
			.await
			.expect("idempotent production migration");
		assert_eq!(second_output, first_output);
		assert_eq!(store.cursor().expect("second cursor").generation, first_cursor.generation);
		let second = store
			.list_credentials(&CredentialFilter { now_ms: 1_000, ..CredentialFilter::default() })
			.expect("metadata after rerun");
		assert_eq!(second, first);
		assert_eq!(
			store
				.lease(oauth.id)
				.expect("OAuth lease after rerun")
				.expect("active OAuth"),
			oauth_generation
		);
		assert_eq!(
			store
				.lease(api_key.id)
				.expect("API-key lease after rerun")
				.expect("active API key"),
			api_key_generation
		);

		let table_output = run(&backend, &AuthCli::parse_from(["auth", "list"]))
			.await
			.expect("client-safe table");
		let json_output = run(&backend, &AuthCli::parse_from(["auth", "list", "--json"]))
			.await
			.expect("client-safe JSON");
		for output in [&first_output, &second_output, &table_output, &json_output] {
			for secret in ["oauth-secret", "refresh-secret", "api-secret", "disabled-secret"] {
				assert!(!output.contains(secret), "secret leaked through CLI output");
			}
		}
	}

	#[test]
	fn mixed_pi_json_unwraps_serialized_rows_and_skips_disabled_secrets() {
		let root = json!({
			"credentials": {
				"openai": [
					{ "id": 1, "type": "api_key", "data": { "key": "api-secret" } },
					{ "id": 2, "type": "api_key", "disabled_cause": "logout",
					  "data": { "key": "disabled-secret" } }
				],
				"codex": [{
					"id": 3,
					"type": "oauth",
					"data": {
						"access": "oauth-secret",
						"refresh": "refresh-secret",
						"accountId": "acct-9",
						"email": "codex@example.test"
					}
				}]
			}
		});
		let imports = parse_migration_json(&root).expect("migration");
		assert_eq!(imports.len(), 2);
		assert!(imports.iter().any(|import| matches!(
			import,
			CredentialImport::ApiKey { provider, .. } if provider == "openai"
		)));
		assert!(imports.iter().any(|import| matches!(
			import,
			CredentialImport::OAuth { provider, identity, props, .. }
				if provider == "openai-codex"
					&& identity == "imported-3"
					&& props["accountId"] == "acct-9"
		)));
	}

	#[test]
	fn credential_rendering_never_leaks_tokens() {
		let secret = "token-bytes-must-not-appear";
		let credential = populated_meta();
		let table = render_credentials_table(std::slice::from_ref(&credential));
		let structured = render_credentials_json(&[credential]).expect("json");
		assert!(!table.contains(secret));
		assert!(!structured.contains(secret));
		assert!(!table.to_ascii_lowercase().contains("access_token"));
		assert!(!structured.to_ascii_lowercase().contains("access_token"));
	}

	#[test]
	fn parser_covers_all_subcommands() {
		for command in ["login", "logout", "list", "refresh", "migrate", "usage"] {
			let mut argv = vec!["auth", command];
			if matches!(command, "logout" | "refresh") {
				argv.push("1");
			}
			assert!(AuthCli::try_parse_from(argv).is_ok(), "{command}");
		}
	}
}
