//! Daemon-owned credential and usage persistence.
//!
//! One broker process owns the SQLite connection and every refresh. There are
//! intentionally no database lease tables: the old cross-process refresh
//! lease and CAS machinery existed only because every client could refresh.
//! [`Store::refresh_singleflight`] supplies the sole remaining coordination,
//! keyed in process by credential id.

use std::{collections::HashMap, path::Path, sync::Arc, time::Duration};

use omp_core::Str;
use omp_llm_egress::auth_inject::{CredentialAuthKind, CredentialLease, CredentialMetadata};
use parking_lot::Mutex;
use rusqlite::{
	Connection, OptionalExtension, Transaction, params,
	types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, Value as SqlValue, ValueRef},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use smallvec::SmallVec;
use thiserror::Error;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

use crate::sealed::{AppliedAuth, Secret};

const SCHEMA_VERSION: u64 = 5;
const MAX_DELTAS: u64 = 1_024;
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

const PRIMARY_WINDOW_HOT_PERCENT: f64 = 85.0;
const ANTHROPIC_CACHE_WARM_MS: u64 = 60 * 60_000;

#[derive(Clone, Copy)]
struct WarmSelection {
	credential_id: u64,
	last_used_ms:  u64,
}

struct RankedCredential {
	id:            u64,
	generation:    u64,
	blocked_until: Option<u64>,
	report:        Option<UsageReport>,
}

impl RankedCredential {
	const fn is_blocked(&self) -> bool {
		self.blocked_until.is_some()
	}

	fn window(&self, index: usize, model: Option<&str>, now_ms: u64) -> Option<&UsageWindow> {
		let report = self.report.as_ref()?;
		let scoped = model.is_some_and(|model| {
			report.windows.iter().any(|window| {
				window
					.label
					.strip_prefix(model)
					.is_some_and(|suffix| suffix.starts_with(':'))
			})
		});
		// OpenCode Go can fall through to balance after monthly exhaustion.
		let mut windows = report.windows.iter().filter(|window| {
			(!scoped
				|| model.is_some_and(|model| {
					window
						.label
						.strip_prefix(model)
						.is_some_and(|suffix| suffix.starts_with(':'))
				}))
				&& !(report.provider == "opencode-go" && window.label == "monthly")
				&& (window.resets_at_ms == 0 || window.resets_at_ms > now_ms)
		});
		windows.nth(index)
	}

	fn used(&self, index: usize, model: Option<&str>, now_ms: u64) -> f64 {
		self
			.window(index, model, now_ms)
			.map_or(50.0, |window| finite_percent(window.used_percent))
	}

	fn primary_used(&self, model: Option<&str>, now_ms: u64) -> f64 {
		self.used(0, model, now_ms)
	}

	fn measured(&self, model: Option<&str>, now_ms: u64) -> bool {
		self.window(0, model, now_ms).is_some() || self.window(1, model, now_ms).is_some()
	}

	fn drain(&self, index: usize, model: Option<&str>, now_ms: u64) -> f64 {
		let Some(window) = self.window(index, model, now_ms) else {
			return 0.0;
		};
		let headroom = (100.0 - finite_percent(window.used_percent)).max(0.0) / 100.0;
		let remaining_ms = if window.resets_at_ms == 0 {
			match index {
				0 => 5 * 60 * 60_000,
				_ => 7 * 24 * 60 * 60_000,
			}
		} else {
			window.resets_at_ms.saturating_sub(now_ms).max(60_000)
		};
		headroom / (remaining_ms as f64 / 3_600_000.0)
	}
}

const fn finite_percent(value: f64) -> f64 {
	if value.is_finite() {
		value.clamp(0.0, 100.0)
	} else {
		50.0
	}
}

fn account_matches(
	identity: &str,
	props: Option<&[u8]>,
	account: &str,
) -> Result<bool, StoreError> {
	if identity == account {
		return Ok(true);
	}
	let Some(props) = props else {
		return Ok(false);
	};
	let props: Value = serde_json::from_slice(props)?;
	Ok([
		props.get("accountId"),
		props.get("account_id"),
		props.get("email"),
		props.pointer("/openai/account_id"),
	]
	.into_iter()
	.flatten()
	.any(|value| value.as_str() == Some(account)))
}

fn compare_ranked(
	left: &RankedCredential,
	right: &RankedCredential,
	model: Option<&str>,
	now_ms: u64,
) -> std::cmp::Ordering {
	use std::cmp::Ordering;

	if left.is_blocked() != right.is_blocked() {
		return left.is_blocked().cmp(&right.is_blocked());
	}
	if left.is_blocked() {
		return left
			.blocked_until
			.cmp(&right.blocked_until)
			.then_with(|| left.id.cmp(&right.id));
	}
	let left_hot = left.primary_used(model, now_ms) >= PRIMARY_WINDOW_HOT_PERCENT;
	let right_hot = right.primary_used(model, now_ms) >= PRIMARY_WINDOW_HOT_PERCENT;
	if left_hot != right_hot {
		return left_hot.cmp(&right_hot);
	}
	if left.measured(model, now_ms) != right.measured(model, now_ms) {
		return right
			.measured(model, now_ms)
			.cmp(&left.measured(model, now_ms));
	}
	for ordering in [
		right
			.drain(1, model, now_ms)
			.total_cmp(&left.drain(1, model, now_ms)),
		left
			.used(1, model, now_ms)
			.total_cmp(&right.used(1, model, now_ms)),
		right
			.drain(0, model, now_ms)
			.total_cmp(&left.drain(0, model, now_ms)),
		left
			.primary_used(model, now_ms)
			.total_cmp(&right.primary_used(model, now_ms)),
	] {
		if ordering != Ordering::Equal {
			return ordering;
		}
	}
	left.id.cmp(&right.id)
}

/// Unsigned domain value stored at SQLite's signed-integer boundary.
///
/// Credential ids, generations, token counts, and epoch-millisecond values fit
/// in `i64` in practice. Values above `i64::MAX` are rejected rather than
/// silently wrapping; negative stored values are likewise rejected on read.
#[derive(Clone, Copy)]
struct SqlU64(u64);

impl ToSql for SqlU64 {
	fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
		let value = i64::try_from(self.0)
			.map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
		Ok(ToSqlOutput::Owned(SqlValue::Integer(value)))
	}
}

impl FromSql for SqlU64 {
	fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
		let value = i64::column_result(value)?;
		u64::try_from(value)
			.map(Self)
			.map_err(|_| FromSqlError::OutOfRange(value))
	}
}

/// Failures while opening or operating the broker store.
#[derive(Debug, Error)]
pub enum StoreError {
	/// SQLite rejected an operation.
	#[error("credential store database error: {0}")]
	Sqlite(#[from] rusqlite::Error),
	/// A stored or supplied usage payload was not valid JSON.
	#[error("credential store JSON error: {0}")]
	Json(#[from] serde_json::Error),
	/// The database was created by a newer broker.
	#[error("credential store schema {found} is newer than supported schema {supported}")]
	NewerSchema {
		/// Version found in the database.
		found:     u64,
		/// Latest version understood by this broker.
		supported: u64,
	},
	/// A generation counter could not be incremented.
	#[error("credential store generation exhausted")]
	GenerationExhausted,
	/// A terminal observation omitted its stable turn identity.
	#[error("terminal usage observation requires a non-empty turn id")]
	InvalidTurnId,
}

/// Stored credential category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CredentialKind {
	/// A provider API key.
	ApiKey,
	/// An OAuth or minted ADC bearer credential.
	OAuth,
	/// An AWS access-key credential used only for in-broker `SigV4` signing.
	Aws,
}
impl CredentialKind {
	const fn as_i64(self) -> i64 {
		match self {
			Self::ApiKey => 1,
			Self::OAuth => 2,
			Self::Aws => 3,
		}
	}

	const fn from_i64(value: i64) -> Self {
		match value {
			2 => Self::OAuth,
			3 => Self::Aws,
			_ => Self::ApiKey,
		}
	}
}

/// Credential lifecycle state exposed to clients.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CredentialState {
	/// Usable and not currently blocked.
	Active,
	/// OAuth access expired and could not be refreshed.
	Expired,
	/// At least one live scoped block exists.
	Blocked,
	/// Disabled by an operator or automatic policy.
	Disabled,
}

impl CredentialState {
	const fn as_i64(self) -> i64 {
		match self {
			Self::Active | Self::Blocked => 1,
			Self::Expired => 2,
			Self::Disabled => 4,
		}
	}

	const fn from_stored(value: i64) -> Self {
		match value {
			2 => Self::Expired,
			4 => Self::Disabled,
			_ => Self::Active,
		}
	}
}

/// One provider-defined rate-limit or quota block.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CredentialBlock {
	/// Provider meter scope such as `shared`, `chat`, or `spark`.
	pub scope:        Str,
	/// Provider endpoint/account discriminator.
	pub provider_key: Str,
	/// Unix epoch milliseconds at which this block expires.
	pub until_ms:     u64,
}

/// Client-safe credential metadata; it has no secret field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CredentialMeta {
	/// Stable database identifier.
	pub id:             u64,
	/// Provider catalog identifier.
	pub provider:       Str,
	/// Credential category.
	pub kind:           CredentialKind,
	/// Non-secret account label.
	pub identity:       Str,
	/// Current effective state.
	pub state:          CredentialState,
	/// Currently live scoped blocks.
	pub blocks:         SmallVec<CredentialBlock, 4>,
	/// Explanation for a disabled credential.
	pub disabled_cause: Str,
	/// OAuth or temporary AWS credential expiry, or zero for static credentials.
	pub expires_at_ms:  u64,
	/// Creation time in Unix epoch milliseconds.
	pub created_at_ms:  u64,
	/// Last mutation time in Unix epoch milliseconds.
	pub updated_at_ms:  u64,
}
/// Aggregate spend within one observed time window.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RollingSpend {
	/// Exact cumulative cost in billionths of a US dollar.
	pub nanos_usd:            u64,
	/// Earliest included observation, used to derive the rolling reset.
	pub first_observed_at_ms: Option<u64>,
}

/// Aggregate premium-request consumption within one observed time window.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PremiumConsumption {
	/// Exact request consumption in millionths of one premium request.
	pub millionths:           u64,
	/// Earliest included observation, used to derive the rolling reset.
	pub first_observed_at_ms: Option<u64>,
}

/// A resumable position in the credential delta stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cursor {
	/// Opaque 128-bit broker-process epoch.
	pub epoch:      [u8; 16],
	/// Store generation after the represented mutation.
	pub generation: u64,
}

/// The kind of one retained credential delta.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeltaKind {
	/// Credential metadata should be fetched and emitted as an upsert.
	Upserted,
	/// The credential was deleted.
	Deleted,
}

/// One entry in the bounded credential mutation log.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CredentialDelta {
	/// Cursor after this mutation.
	pub cursor:        Cursor,
	/// Affected credential id.
	pub credential_id: u64,
	/// Mutation kind.
	pub kind:          DeltaKind,
}

/// Result of attempting to resume the bounded delta log.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeltaReplay {
	/// The cursor is valid and all later retained deltas are returned.
	Deltas(Vec<CredentialDelta>),
	/// The epoch or generation is stale; clients must list again.
	Reset(Cursor),
}

/// One provider quota window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UsageWindow {
	/// Provider-defined window label.
	pub label:        Str,
	/// Percentage consumed; providers may report values above 100.
	pub used_percent: f64,
	/// Unix epoch milliseconds at which the window resets.
	pub resets_at_ms: u64,
}

/// Cached provider-side quota metadata for one credential.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UsageReport {
	/// Credential owning this quota report.
	pub credential_id: u64,
	/// Provider catalog identifier.
	pub provider:      Str,
	/// Provider plan or tier label.
	pub plan:          Str,
	/// Provider-defined quota windows.
	pub windows:       SmallVec<UsageWindow, 4>,
	/// Fetch time in Unix epoch milliseconds.
	pub fetched_at_ms: u64,
	/// Namespaced provider-specific metadata.
	pub detail:        Value,
}

/// Durable provider quota history entry.
#[derive(Clone, Debug, PartialEq)]
pub struct UsageHistoryEntry {
	/// Persistence time in Unix epoch milliseconds.
	pub at_ms:  u64,
	/// Report persisted at that time.
	pub report: UsageReport,
}

/// Per-client token and cost aggregate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClientUsage {
	/// Stable gateway client identifier.
	pub client_id:          Str,
	/// Human-readable client label.
	pub label:              Str,
	/// Total input tokens.
	pub input_tokens:       u64,
	/// Total output tokens.
	pub output_tokens:      u64,
	/// Total cache-read tokens.
	pub cache_read_tokens:  u64,
	/// Total cache-write tokens.
	pub cache_write_tokens: u64,
	/// Total cost in billionths of a US dollar.
	pub nanos_usd:          u64,
	/// Last observation time in Unix epoch milliseconds.
	pub last_seen_ms:       u64,
}

/// Filter applied by [`Store::list_credentials`].
#[derive(Clone, Debug, Default)]
pub struct CredentialFilter<'a> {
	/// Provider id, or `None` for every provider.
	pub provider: Option<&'a str>,
	/// Effective states, or an empty slice for every state.
	pub states:   &'a [CredentialState],
	/// Time used to determine live blocks.
	pub now_ms:   u64,
}

/// The daemon's sole SQLite credential-store owner.
pub struct Store {
	connection: Mutex<Connection>,
	refreshes:  Mutex<HashMap<u64, Arc<AsyncMutex<()>>>>,
	warm:       Mutex<HashMap<Str, WarmSelection>>,
	epoch:      [u8; 16],
}

impl Store {
	/// Opens a database, configures contention behavior, and migrates it.
	pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
		let connection = Connection::open(path)?;
		// These pragmas must precede all lock-taking DDL. The previous broker
		// created tables first and could crash when clients started concurrently.
		connection.busy_timeout(BUSY_TIMEOUT)?;
		connection.pragma_update(None, "journal_mode", "WAL")?;
		migrate(&connection)?;
		let epoch: [u8; 16] = rand::random();
		Ok(Self {
			connection: Mutex::new(connection),
			refreshes: Mutex::new(HashMap::new()),
			warm: Mutex::new(HashMap::new()),
			epoch,
		})
	}

	/// Returns the current stream cursor.
	pub fn cursor(&self) -> Result<Cursor, StoreError> {
		let generation = current_generation(&self.connection.lock())?;
		Ok(self.cursor_at(generation))
	}

	/// Serializes refresh work for one credential within this daemon.
	///
	/// Dropping the returned guard releases the single-flight. No SQLite lease
	/// row exists or is needed because no other process may refresh.
	pub async fn refresh_singleflight(&self, credential_id: u64) -> OwnedMutexGuard<()> {
		let flight = {
			let mut flights = self.refreshes.lock();
			Arc::clone(
				flights
					.entry(credential_id)
					.or_insert_with(|| Arc::new(AsyncMutex::new(()))),
			)
		};
		flight.lock_owned().await
	}

	/// Inserts or replaces an API key by provider and identity.
	pub fn upsert_api_key(
		&self,
		provider: &str,
		identity: &str,
		secret: &[u8],
		now_ms: u64,
	) -> Result<CredentialMeta, StoreError> {
		self.upsert_credential(
			provider,
			CredentialKind::ApiKey,
			identity,
			secret,
			None,
			None,
			None,
			None,
			0,
			false,
			now_ms,
		)
	}

	/// Idempotently imports an API key from an external credential store.
	pub(crate) fn import_api_key(
		&self,
		provider: &str,
		identity: &str,
		secret: &[u8],
		now_ms: u64,
	) -> Result<CredentialMeta, StoreError> {
		self.upsert_credential(
			provider,
			CredentialKind::ApiKey,
			identity,
			secret,
			None,
			None,
			None,
			None,
			0,
			true,
			now_ms,
		)
	}

	/// Inserts or replaces an OAuth access token by provider and identity.
	pub fn upsert_oauth(
		&self,
		provider: &str,
		identity: &str,
		access_token: &[u8],
		expires_at_ms: u64,
		now_ms: u64,
	) -> Result<CredentialMeta, StoreError> {
		self.upsert_credential(
			provider,
			CredentialKind::OAuth,
			identity,
			access_token,
			None,
			None,
			None,
			None,
			expires_at_ms,
			false,
			now_ms,
		)
	}

	/// Inserts or replaces a broker-minted bearer token with typed provider
	/// routing properties at a one-way secret ingress.
	///
	/// `props` must contain only non-secret routing identifiers under known
	/// provider namespaces; token material belongs exclusively in
	/// `access_token`.
	pub fn upsert_minted_bearer(
		&self,
		provider: &str,
		identity: &str,
		access_token: &[u8],
		expires_at_ms: u64,
		props: &Value,
		now_ms: u64,
	) -> Result<CredentialMeta, StoreError> {
		self.upsert_credential(
			provider,
			CredentialKind::OAuth,
			identity,
			access_token,
			None,
			Some(props),
			None,
			None,
			expires_at_ms,
			false,
			now_ms,
		)
	}

	/// Inserts or replaces OAuth access and refresh material received at the
	/// broker's one-way import or login boundary.
	pub(crate) fn upsert_oauth_material(
		&self,
		provider: &str,
		identity: &str,
		access_token: &[u8],
		refresh_token: Option<&[u8]>,
		props: &Value,
		expires_at_ms: u64,
		now_ms: u64,
	) -> Result<CredentialMeta, StoreError> {
		self.upsert_credential(
			provider,
			CredentialKind::OAuth,
			identity,
			access_token,
			refresh_token,
			Some(props),
			None,
			None,
			expires_at_ms,
			false,
			now_ms,
		)
	}

	/// Idempotently imports OAuth material from an external credential store.
	pub(crate) fn import_oauth_material(
		&self,
		provider: &str,
		identity: &str,
		access_token: &[u8],
		refresh_token: Option<&[u8]>,
		props: &Value,
		expires_at_ms: u64,
		now_ms: u64,
	) -> Result<CredentialMeta, StoreError> {
		self.upsert_credential(
			provider,
			CredentialKind::OAuth,
			identity,
			access_token,
			refresh_token,
			Some(props),
			None,
			None,
			expires_at_ms,
			true,
			now_ms,
		)
	}

	/// Inserts or replaces AWS `SigV4` access, secret, optional session, and
	/// expiry material at the broker's one-way secret ingress.
	pub fn upsert_aws(
		&self,
		provider: &str,
		identity: &str,
		access_key_id: &[u8],
		secret_access_key: &[u8],
		session_token: Option<&[u8]>,
		expires_at_ms: u64,
		now_ms: u64,
	) -> Result<CredentialMeta, StoreError> {
		self.upsert_credential(
			provider,
			CredentialKind::Aws,
			identity,
			secret_access_key,
			None,
			None,
			Some(access_key_id),
			session_token,
			expires_at_ms,
			false,
			now_ms,
		)
	}

	/// Lists client-safe metadata matching provider and effective-state filters.
	pub fn list_credentials(
		&self,
		filter: &CredentialFilter<'_>,
	) -> Result<Vec<CredentialMeta>, StoreError> {
		let connection = self.connection.lock();
		let mut statement = connection.prepare(
			"SELECT id, provider, kind, identity, state, disabled_cause, expires_at_ms, \
			 created_at_ms, updated_at_ms FROM credentials WHERE (?1 = '' OR provider = ?1) ORDER BY \
			 id",
		)?;
		let provider = filter.provider.unwrap_or("");
		let rows = statement.query_map([provider], metadata_row)?;
		let mut credentials = Vec::new();
		for row in rows {
			let mut meta = row?;
			load_blocks(&connection, &mut meta, filter.now_ms)?;
			if filter.states.is_empty() || filter.states.contains(&meta.state) {
				credentials.push(meta);
			}
		}
		Ok(credentials)
	}

	/// Gets client-safe metadata for one credential.
	pub fn get_credential(
		&self,
		id: u64,
		now_ms: u64,
	) -> Result<Option<CredentialMeta>, StoreError> {
		let connection = self.connection.lock();
		get_metadata(&connection, id, now_ms).map_err(Into::into)
	}

	/// Creates a canonical egress handle for the credential's current
	/// generation.
	pub fn lease(&self, id: u64) -> Result<Option<CredentialLease>, StoreError> {
		let connection = self.connection.lock();
		let row = connection
			.query_row(
				"SELECT provider, generation FROM credentials WHERE id = ?1 AND state = 1",
				[SqlU64(id)],
				|row| Ok((row_str(row, 0)?, row.get::<_, SqlU64>(1)?.0)),
			)
			.optional()?;
		Ok(row.map(|(provider, generation)| CredentialLease::new(provider, id, generation)))
	}

	/// Returns enabled credential ids in broker preference order.
	///
	/// Unblocked accounts precede live-blocked last resorts. Quota windows are
	/// scoped to `model` when the provider report carries model-qualified
	/// labels, and `account` restricts candidates to an exact non-secret
	/// credential identity. Anthropic's still-warm account is promoted without
	/// changing the deterministic ordering of genuine ties.
	pub fn ranked_credential_ids(
		&self,
		provider: &str,
		model: Option<&str>,
		account: Option<&str>,
		now_ms: u64,
	) -> Result<SmallVec<u64, 4>, StoreError> {
		Ok(self
			.ranked_credentials(provider, model, account, now_ms)?
			.into_iter()
			.map(|candidate| candidate.id)
			.collect())
	}

	/// Leases the highest-ranked enabled credential for `provider`.
	pub fn lease_provider(
		&self,
		provider: &str,
		now_ms: u64,
	) -> Result<Option<CredentialLease>, StoreError> {
		let candidates = self.ranked_credentials(provider, None, None, now_ms)?;
		let Some(selected) = candidates.first() else {
			return Ok(None);
		};
		let lease = CredentialLease::new(provider, selected.id, selected.generation);
		if provider == "anthropic" && !selected.is_blocked() {
			self.warm.lock().insert(Str::new(provider), WarmSelection {
				credential_id: selected.id,
				last_used_ms:  now_ms,
			});
		}
		Ok(Some(lease))
	}

	fn ranked_credentials(
		&self,
		provider: &str,
		model: Option<&str>,
		account: Option<&str>,
		now_ms: u64,
	) -> Result<Vec<RankedCredential>, StoreError> {
		let connection = self.connection.lock();
		let mut statement = connection.prepare(
			"SELECT c.id, c.generation,
			 (SELECT MAX(b.until_ms) FROM credential_blocks b
			  WHERE b.credential_id = c.id AND b.until_ms > ?2),
			 (SELECT u.report FROM usage_reports u
			  WHERE u.credential_id = c.id AND u.provider = c.provider AND u.stale = 0),
			 c.identity, c.props
			 FROM credentials c
			 WHERE c.provider = ?1 AND c.state = 1
			 ORDER BY c.id",
		)?;
		let rows = statement.query_map(params![provider, SqlU64(now_ms)], |row| {
			Ok((
				row.get::<_, SqlU64>(0)?.0,
				row.get::<_, SqlU64>(1)?.0,
				row.get::<_, Option<SqlU64>>(2)?.map(|value| value.0),
				row.get::<_, Option<Vec<u8>>>(3)?,
				row.get::<_, String>(4)?,
				row.get::<_, Option<Vec<u8>>>(5)?,
			))
		})?;
		let mut candidates = Vec::new();
		for row in rows {
			let (id, generation, blocked_until, report, identity, props) = row?;
			if let Some(account) = account
				&& !account_matches(&identity, props.as_deref(), account)?
			{
				continue;
			}
			candidates.push(RankedCredential {
				id,
				generation,
				blocked_until,
				report: report
					.map(|payload| serde_json::from_slice(&payload))
					.transpose()?,
			});
		}
		drop(statement);
		drop(connection);
		candidates.sort_by(|left, right| compare_ranked(left, right, model, now_ms));
		if provider == "anthropic"
			&& let Some(selection) = self.warm.lock().get(provider).copied()
			&& now_ms.saturating_sub(selection.last_used_ms) < ANTHROPIC_CACHE_WARM_MS
			&& let Some(position) = candidates.iter().position(|candidate| {
				candidate.id == selection.credential_id && !candidate.is_blocked()
			}) {
			candidates[..=position].rotate_right(1);
		}
		Ok(candidates)
	}

	/// Deletes a credential and its dependent blocks and usage records.
	pub fn delete_credential(&self, id: u64, now_ms: u64) -> Result<bool, StoreError> {
		let mut connection = self.connection.lock();
		let transaction = connection.transaction()?;
		let changed =
			transaction.execute("DELETE FROM credentials WHERE id = ?1", [SqlU64(id)])? != 0;
		if changed {
			let generation = bump_generation(&transaction)?;
			record_delta(&transaction, generation, id, DeltaKind::Deleted, now_ms)?;
		}
		transaction.commit()?;
		Ok(changed)
	}

	/// Disables a credential and records a client-safe cause.
	pub fn disable_credential(
		&self,
		id: u64,
		cause: &str,
		now_ms: u64,
	) -> Result<Option<CredentialMeta>, StoreError> {
		self.set_state(id, CredentialState::Disabled, cause, now_ms)
	}

	/// Re-enables a credential.
	pub fn enable_credential(
		&self,
		id: u64,
		now_ms: u64,
	) -> Result<Option<CredentialMeta>, StoreError> {
		self.set_state(id, CredentialState::Active, "", now_ms)
	}

	/// Marks a credential expired after an unrecoverable OAuth refresh failure.
	pub fn expire_credential(
		&self,
		id: u64,
		now_ms: u64,
	) -> Result<Option<CredentialMeta>, StoreError> {
		self.set_state(id, CredentialState::Expired, "", now_ms)
	}

	/// Adds or replaces one block without disturbing other scope/key pairs.
	pub fn report_block(
		&self,
		id: u64,
		block: &CredentialBlock,
		now_ms: u64,
	) -> Result<Option<CredentialMeta>, StoreError> {
		let mut connection = self.connection.lock();
		let transaction = connection.transaction()?;
		if !credential_exists(&transaction, id)? {
			return Ok(None);
		}
		transaction.execute(
			"INSERT INTO credential_blocks (credential_id, scope, provider_key, until_ms) VALUES \
			 (?1, ?2, ?3, ?4) ON CONFLICT(credential_id, scope, provider_key) DO UPDATE SET until_ms \
			 = excluded.until_ms",
			params![
				SqlU64(id),
				block.scope.as_str(),
				block.provider_key.as_str(),
				SqlU64(block.until_ms)
			],
		)?;
		let generation = bump_generation(&transaction)?;
		set_credential_generation(&transaction, id, generation, now_ms)?;
		record_delta(&transaction, generation, id, DeltaKind::Upserted, now_ms)?;
		transaction.commit()?;
		get_metadata(&connection, id, now_ms).map_err(Into::into)
	}

	/// Clears all blocks, or only the supplied scopes, for one credential.
	pub fn clear_blocks(
		&self,
		id: u64,
		scopes: &[Str],
		now_ms: u64,
	) -> Result<Option<CredentialMeta>, StoreError> {
		let mut connection = self.connection.lock();
		let transaction = connection.transaction()?;
		if !credential_exists(&transaction, id)? {
			return Ok(None);
		}
		if scopes.is_empty() {
			transaction
				.execute("DELETE FROM credential_blocks WHERE credential_id = ?1", [SqlU64(id)])?;
		} else {
			for scope in scopes {
				transaction.execute(
					"DELETE FROM credential_blocks WHERE credential_id = ?1 AND scope = ?2",
					params![SqlU64(id), scope.as_str()],
				)?;
			}
		}
		let generation = bump_generation(&transaction)?;
		set_credential_generation(&transaction, id, generation, now_ms)?;
		record_delta(&transaction, generation, id, DeltaKind::Upserted, now_ms)?;
		transaction.commit()?;
		get_metadata(&connection, id, now_ms).map_err(Into::into)
	}

	pub(crate) fn credential_metadata(
		&self,
		provider: &str,
		id: u64,
		generation: u64,
	) -> Result<Option<CredentialMetadata>, StoreError> {
		let connection = self.connection.lock();
		let row = connection
			.query_row(
				"SELECT identity, props, kind FROM credentials WHERE provider = ?1 AND id = ?2 AND \
				 generation = ?3 AND state = 1",
				params![provider, SqlU64(id), SqlU64(generation)],
				|row| {
					Ok((
						row_str(row, 0)?,
						row.get::<_, Option<Vec<u8>>>(1)?,
						CredentialKind::from_i64(row.get(2)?),
					))
				},
			)
			.optional()?;
		let Some((identity, props, kind)) = row else {
			return Ok(None);
		};
		let props = props
			.map(|props| {
				let props = Secret::from_vec(props);
				serde_json::from_slice::<MetadataProjection>(props.expose())
			})
			.transpose()?
			.unwrap_or_default();
		Ok(Some(CredentialMetadata {
			auth_kind: match kind {
				CredentialKind::ApiKey => CredentialAuthKind::ApiKey,
				CredentialKind::OAuth => CredentialAuthKind::OAuth,
				CredentialKind::Aws => CredentialAuthKind::Aws,
			},
			identity,
			account_id: first_metadata([
				props
					.openai
					.as_ref()
					.and_then(|props| props.account_id.as_ref()),
				props
					.codex
					.as_ref()
					.and_then(|props| props.account_id.as_ref()),
				props.account_id.as_ref(),
			]),
			project_id: first_metadata([
				props
					.google
					.as_ref()
					.and_then(|props| props.project_id.as_ref()),
				props
					.antigravity
					.as_ref()
					.and_then(|props| props.project_id.as_ref()),
				props.project_id.as_ref(),
				props.quota_project_id.as_ref(),
			]),
			organization_id: first_metadata([
				props
					.openai
					.as_ref()
					.and_then(|props| props.organization_id.as_ref()),
				props
					.zai
					.as_ref()
					.and_then(|props| props.organization_id.as_ref()),
				props.organization_id.as_ref(),
			]),
		}))
	}

	/// Stores the latest quota report and appends it to durable history.
	pub fn write_usage_report(&self, report: &UsageReport, at_ms: u64) -> Result<(), StoreError> {
		let payload = serde_json::to_vec(report)?;
		let mut connection = self.connection.lock();
		let transaction = connection.transaction()?;
		transaction.execute(
			"INSERT INTO usage_reports (credential_id, provider, report, fetched_at_ms, stale) \
			 VALUES (?1, ?2, ?3, ?4, 0) ON CONFLICT(credential_id) DO UPDATE SET provider = \
			 excluded.provider, report = excluded.report, fetched_at_ms = excluded.fetched_at_ms, \
			 stale = 0",
			params![
				SqlU64(report.credential_id),
				report.provider.as_str(),
				payload,
				SqlU64(report.fetched_at_ms)
			],
		)?;
		transaction.execute(
			"INSERT INTO usage_history (credential_id, at_ms, report) VALUES (?1, ?2, ?3)",
			params![SqlU64(report.credential_id), SqlU64(at_ms), payload],
		)?;
		transaction.commit()?;
		Ok(())
	}

	/// Reads cached quota reports, excluding reports marked stale.
	pub fn read_usage_reports(
		&self,
		provider: Option<&str>,
		credential_id: Option<u64>,
	) -> Result<Vec<UsageReport>, StoreError> {
		let connection = self.connection.lock();
		let mut statement = connection.prepare(
			"SELECT report FROM usage_reports WHERE stale = 0 AND (?1 = '' OR provider = ?1) AND (?2 \
			 = 0 OR credential_id = ?2) ORDER BY credential_id",
		)?;
		let rows = statement
			.query_map(params![provider.unwrap_or(""), SqlU64(credential_id.unwrap_or(0))], |row| {
				row.get::<_, Vec<u8>>(0)
			})?;
		let mut reports = Vec::new();
		for row in rows {
			reports.push(serde_json::from_slice(&row?)?);
		}
		Ok(reports)
	}

	/// Marks matching cached quota reports stale.
	pub fn mark_usage_stale(
		&self,
		provider: Option<&str>,
		credential_id: Option<u64>,
	) -> Result<usize, StoreError> {
		let connection = self.connection.lock();
		Ok(connection.execute(
			"UPDATE usage_reports SET stale = 1 WHERE (?1 = '' OR provider = ?1) AND (?2 = 0 OR \
			 credential_id = ?2)",
			params![provider.unwrap_or(""), SqlU64(credential_id.unwrap_or(0))],
		)?)
	}

	/// Reads durable quota history in the inclusive time range.
	pub fn usage_history(
		&self,
		credential_id: u64,
		since_ms: u64,
		until_ms: u64,
	) -> Result<Vec<UsageHistoryEntry>, StoreError> {
		let connection = self.connection.lock();
		let mut statement = connection.prepare(
			"SELECT at_ms, report FROM usage_history WHERE credential_id = ?1 AND at_ms >= ?2 AND \
			 (?3 = 0 OR at_ms <= ?3) ORDER BY at_ms",
		)?;
		let rows = statement
			.query_map(params![SqlU64(credential_id), SqlU64(since_ms), SqlU64(until_ms)], |row| {
				Ok((row.get::<_, SqlU64>(0)?.0, row.get::<_, Vec<u8>>(1)?))
			})?;
		let mut history = Vec::new();
		for row in rows {
			let (at_ms, payload) = row?;
			history.push(UsageHistoryEntry { at_ms, report: serde_json::from_slice(&payload)? });
		}
		Ok(history)
	}

	/// Adds one gateway observation to a client's durable aggregate.
	pub fn add_client_usage(&self, usage: &ClientUsage) -> Result<(), StoreError> {
		let connection = self.connection.lock();
		connection.execute(
			"INSERT INTO client_usage (client_id, label, input_tokens, output_tokens, \
			 cache_read_tokens, cache_write_tokens, nanos_usd, last_seen_ms) VALUES (?1, ?2, ?3, ?4, \
			 ?5, ?6, ?7, ?8) ON CONFLICT(client_id) DO UPDATE SET label = excluded.label, \
			 input_tokens = input_tokens + excluded.input_tokens, output_tokens = output_tokens + \
			 excluded.output_tokens, cache_read_tokens = cache_read_tokens + \
			 excluded.cache_read_tokens, cache_write_tokens = cache_write_tokens + \
			 excluded.cache_write_tokens, nanos_usd = nanos_usd + excluded.nanos_usd, last_seen_ms = \
			 MAX(last_seen_ms, excluded.last_seen_ms)",
			params![
				usage.client_id.as_str(),
				usage.label.as_str(),
				SqlU64(usage.input_tokens),
				SqlU64(usage.output_tokens),
				SqlU64(usage.cache_read_tokens),
				SqlU64(usage.cache_write_tokens),
				SqlU64(usage.nanos_usd),
				SqlU64(usage.last_seen_ms)
			],
		)?;
		Ok(())
	}

	/// Idempotently records one selected-credential terminal observation.
	///
	/// The credential account label and resolved catalog model multiplier are
	/// snapshotted inside the same transaction, and the client aggregate
	/// advances only when `turn_id` is newly inserted.
	pub fn record_terminal_usage(
		&self,
		lease: &CredentialLease,
		turn_id: &str,
		model: &str,
		premium_multiplier_millionths: u64,
		usage: &ClientUsage,
		observed_at_ms: u64,
	) -> Result<bool, StoreError> {
		if turn_id.is_empty() {
			return Err(StoreError::InvalidTurnId);
		}
		let mut connection = self.connection.lock();
		let transaction = connection.transaction()?;
		let account = transaction
			.query_row(
				"SELECT identity FROM credentials WHERE provider = ?1 AND id = ?2 AND generation = ?3",
				params![lease.provider(), SqlU64(lease.credential_id()), SqlU64(lease.generation())],
				|row| row.get::<_, String>(0),
			)
			.optional()?;
		let Some(account) = account else {
			return Ok(false);
		};
		let inserted = transaction.execute(
			"INSERT OR IGNORE INTO provider_usage_history (provider, credential_id, account, \
			 turn_id, model, premium_multiplier_millionths, input_tokens, output_tokens, \
			 cache_read_tokens, cache_write_tokens, nanos_usd, at_ms) VALUES (?1, ?2, ?3, ?4, ?5, \
			 ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
			params![
				lease.provider(),
				SqlU64(lease.credential_id()),
				account,
				turn_id,
				model,
				SqlU64(premium_multiplier_millionths),
				SqlU64(usage.input_tokens),
				SqlU64(usage.output_tokens),
				SqlU64(usage.cache_read_tokens),
				SqlU64(usage.cache_write_tokens),
				SqlU64(usage.nanos_usd),
				SqlU64(observed_at_ms),
			],
		)? == 1;
		if inserted {
			transaction.execute(
				"INSERT INTO client_usage (client_id, label, input_tokens, output_tokens, \
				 cache_read_tokens, cache_write_tokens, nanos_usd, last_seen_ms) VALUES (?1, ?2, ?3, \
				 ?4, ?5, ?6, ?7, ?8) ON CONFLICT(client_id) DO UPDATE SET label = excluded.label, \
				 input_tokens = input_tokens + excluded.input_tokens, output_tokens = output_tokens + \
				 excluded.output_tokens, cache_read_tokens = cache_read_tokens + \
				 excluded.cache_read_tokens, cache_write_tokens = cache_write_tokens + \
				 excluded.cache_write_tokens, nanos_usd = nanos_usd + excluded.nanos_usd, \
				 last_seen_ms = MAX(last_seen_ms, excluded.last_seen_ms)",
				params![
					usage.client_id.as_str(),
					usage.label.as_str(),
					SqlU64(usage.input_tokens),
					SqlU64(usage.output_tokens),
					SqlU64(usage.cache_read_tokens),
					SqlU64(usage.cache_write_tokens),
					SqlU64(usage.nanos_usd),
					SqlU64(usage.last_seen_ms),
				],
			)?;
			transaction.execute("UPDATE usage_reports SET stale = 1 WHERE credential_id = ?1", [
				SqlU64(lease.credential_id()),
			])?;
		}
		transaction.commit()?;
		Ok(inserted)
	}

	/// Returns exact observed spend for one provider/account window.
	pub fn rolling_spend(
		&self,
		provider: &str,
		credential_id: Option<u64>,
		account: Option<&str>,
		since_ms: u64,
		until_ms: u64,
	) -> Result<RollingSpend, StoreError> {
		let connection = self.connection.lock();
		let (nanos_usd, first_at) = connection.query_row(
			"SELECT SUM(nanos_usd), MIN(at_ms) FROM provider_usage_history WHERE provider = ?1 AND \
			 (?2 = 0 OR credential_id = ?2) AND (?3 IS NULL OR account = ?3) AND at_ms >= ?4 AND (?5 \
			 = 0 OR at_ms <= ?5)",
			params![
				provider,
				SqlU64(credential_id.unwrap_or(0)),
				account,
				SqlU64(since_ms),
				SqlU64(until_ms),
			],
			|row| {
				Ok((
					row.get::<_, Option<SqlU64>>(0)?.map_or(0, |value| value.0),
					row.get::<_, Option<SqlU64>>(1)?.map(|value| value.0),
				))
			},
		)?;
		Ok(RollingSpend { nanos_usd, first_observed_at_ms: first_at })
	}

	/// Returns exact premium-request consumption for one resolved route window.
	pub fn premium_consumption(
		&self,
		provider: &str,
		model: &str,
		credential_id: u64,
		account: Option<&str>,
		since_ms: u64,
		until_ms: u64,
	) -> Result<PremiumConsumption, StoreError> {
		let connection = self.connection.lock();
		let (millionths, first_observed_at_ms) = connection.query_row(
			"SELECT SUM(premium_multiplier_millionths), MIN(at_ms) FROM provider_usage_history WHERE \
			 provider = ?1 AND model = ?2 AND credential_id = ?3 AND (?4 IS NULL OR account = ?4) \
			 AND at_ms >= ?5 AND (?6 = 0 OR at_ms <= ?6)",
			params![
				provider,
				model,
				SqlU64(credential_id),
				account,
				SqlU64(since_ms),
				SqlU64(until_ms),
			],
			|row| {
				Ok((
					row.get::<_, Option<SqlU64>>(0)?.map_or(0, |value| value.0),
					row.get::<_, Option<SqlU64>>(1)?.map(|value| value.0),
				))
			},
		)?;
		Ok(PremiumConsumption { millionths, first_observed_at_ms })
	}

	/// Lists client usage aggregates observed at or after `since_ms`.
	pub fn client_usage(&self, since_ms: u64) -> Result<Vec<ClientUsage>, StoreError> {
		let connection = self.connection.lock();
		let mut statement = connection.prepare(
			"SELECT client_id, label, input_tokens, output_tokens, cache_read_tokens, \
			 cache_write_tokens, nanos_usd, last_seen_ms FROM client_usage WHERE last_seen_ms >= ?1 \
			 ORDER BY client_id",
		)?;
		let rows = statement.query_map([SqlU64(since_ms)], |row| {
			Ok(ClientUsage {
				client_id:          row_str(row, 0)?,
				label:              row_str(row, 1)?,
				input_tokens:       row.get::<_, SqlU64>(2)?.0,
				output_tokens:      row.get::<_, SqlU64>(3)?.0,
				cache_read_tokens:  row.get::<_, SqlU64>(4)?.0,
				cache_write_tokens: row.get::<_, SqlU64>(5)?.0,
				nanos_usd:          row.get::<_, SqlU64>(6)?.0,
				last_seen_ms:       row.get::<_, SqlU64>(7)?.0,
			})
		})?;
		rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
	}

	/// Replays retained deltas after a cursor or requests a full reset.
	pub fn deltas_since(&self, since: &Cursor) -> Result<DeltaReplay, StoreError> {
		let connection = self.connection.lock();
		let current = current_generation(&connection)?;
		if since.epoch != self.epoch || since.generation > current {
			return Ok(DeltaReplay::Reset(self.cursor_at(current)));
		}
		let oldest = connection
			.query_row("SELECT MIN(generation) FROM credential_deltas", [], |row| {
				row.get::<_, Option<SqlU64>>(0)
			})?
			.map(|value| value.0);
		if oldest.is_some_and(|oldest| since.generation.saturating_add(1) < oldest) {
			return Ok(DeltaReplay::Reset(self.cursor_at(current)));
		}
		let mut statement = connection.prepare(
			"SELECT generation, credential_id, kind FROM credential_deltas WHERE generation > ?1 \
			 ORDER BY generation",
		)?;
		let rows = statement.query_map([SqlU64(since.generation)], |row| {
			let generation = row.get::<_, SqlU64>(0)?.0;
			let kind = if row.get::<_, i64>(2)? == 2 {
				DeltaKind::Deleted
			} else {
				DeltaKind::Upserted
			};
			Ok(CredentialDelta {
				cursor: self.cursor_at(generation),
				credential_id: row.get::<_, SqlU64>(1)?.0,
				kind,
			})
		})?;
		Ok(DeltaReplay::Deltas(rows.collect::<Result<Vec<_>, _>>()?))
	}

	pub(crate) fn redeem_with<T, E>(
		&self,
		provider: &str,
		id: u64,
		generation: u64,
		apply: impl FnOnce(AppliedAuth) -> Result<T, E>,
	) -> Result<Option<Result<T, E>>, StoreError> {
		let redeemed = self.redeem_typed_with(provider, id, generation, |kind, auth| {
			Ok::<_, std::convert::Infallible>((kind, auth))
		})?;
		match redeemed {
			None => Ok(None),
			Some(Ok((CredentialKind::Aws, _))) => Ok(None),
			Some(Ok((_, auth))) => Ok(Some(apply(auth))),
			Some(Err(never)) => match never {},
		}
	}

	pub(crate) fn redeem_typed_with<T, E>(
		&self,
		provider: &str,
		id: u64,
		generation: u64,
		apply: impl FnOnce(CredentialKind, AppliedAuth) -> Result<T, E>,
	) -> Result<Option<Result<T, E>>, StoreError> {
		let connection = self.connection.lock();
		let row = connection
			.query_row(
				"SELECT kind, secret, aws_access_key, aws_session_token FROM credentials WHERE \
				 provider = ?1 AND id = ?2 AND generation = ?3 AND state = 1",
				params![provider, SqlU64(id), SqlU64(generation)],
				|row| {
					Ok((
						CredentialKind::from_i64(row.get(0)?),
						row.get::<_, Vec<u8>>(1)?,
						row.get::<_, Option<Vec<u8>>>(2)?,
						row.get::<_, Option<Vec<u8>>>(3)?,
					))
				},
			)
			.optional()?;
		Ok(row.map(|(kind, bytes, access_key, session_token)| {
			let auth = if kind == CredentialKind::Aws {
				match access_key {
					Some(access_key) => AppliedAuth::aws(
						Secret::from_vec(bytes),
						Secret::from_vec(access_key),
						session_token.map(Secret::from_vec),
					),
					None => AppliedAuth::bearer(Secret::from_vec(bytes)),
				}
			} else {
				AppliedAuth::bearer(Secret::from_vec(bytes))
			};
			apply(kind, auth)
		}))
	}

	pub(crate) fn redeem_refresh(&self, id: u64) -> Result<Option<Secret>, StoreError> {
		let connection = self.connection.lock();
		let bytes = connection
			.query_row(
				"SELECT refresh_secret FROM credentials WHERE id = ?1 AND kind = 2",
				[SqlU64(id)],
				|row| row.get::<_, Option<Vec<u8>>>(0),
			)
			.optional()?
			.flatten();
		Ok(bytes.map(Secret::from_vec))
	}

	pub(crate) fn oauth_props(&self, id: u64) -> Result<Option<Value>, StoreError> {
		let connection = self.connection.lock();
		let payload = connection
			.query_row(
				"SELECT props FROM credentials WHERE id = ?1 AND kind = 2",
				[SqlU64(id)],
				|row| row.get::<_, Option<Vec<u8>>>(0),
			)
			.optional()?
			.flatten();
		payload
			.map(|payload| serde_json::from_slice(&payload))
			.transpose()
			.map_err(Into::into)
	}

	fn upsert_credential(
		&self,
		provider: &str,
		kind: CredentialKind,
		identity: &str,
		secret: &[u8],
		refresh_token: Option<&[u8]>,
		props: Option<&Value>,
		aws_access_key: Option<&[u8]>,
		aws_session_token: Option<&[u8]>,
		expires_at_ms: u64,
		idempotent: bool,
		now_ms: u64,
	) -> Result<CredentialMeta, StoreError> {
		let props = props
			.map(serde_json::to_vec)
			.transpose()?
			.map(Secret::from_vec);
		let mut connection = self.connection.lock();
		let transaction = connection.transaction()?;
		let existing = transaction
			.query_row(
				"SELECT id, state, disabled_cause, expires_at_ms, secret, refresh_secret, props, \
				 aws_access_key, aws_session_token FROM credentials WHERE provider = ?1 AND kind = ?2 \
				 AND identity = ?3",
				params![provider, kind.as_i64(), identity],
				|row| {
					Ok((
						row.get::<_, SqlU64>(0)?.0,
						row.get::<_, i64>(1)?,
						row.get::<_, String>(2)?,
						row.get::<_, SqlU64>(3)?.0,
						row.get::<_, Vec<u8>>(4)?,
						row.get::<_, Option<Vec<u8>>>(5)?,
						row.get::<_, Option<Vec<u8>>>(6)?,
						row.get::<_, Option<Vec<u8>>>(7)?,
						row.get::<_, Option<Vec<u8>>>(8)?,
					))
				},
			)
			.optional()?;
		if idempotent
			&& let Some((
				id,
				state,
				disabled_cause,
				stored_expires_at_ms,
				stored_secret,
				stored_refresh,
				stored_props,
				stored_aws_access_key,
				stored_aws_session_token,
			)) = &existing
			&& *state == CredentialState::Active.as_i64()
			&& stored_secret.as_slice() == secret
			&& disabled_cause.is_empty()
			&& *stored_expires_at_ms == expires_at_ms
			&& refresh_token.is_none_or(|value| stored_refresh.as_deref() == Some(value))
			&& props
				.as_ref()
				.is_none_or(|value| stored_props.as_deref() == Some(value.expose()))
			&& stored_aws_access_key.as_deref() == aws_access_key
			&& stored_aws_session_token.as_deref() == aws_session_token
		{
			let id = *id;
			transaction.rollback()?;
			return get_metadata(&connection, id, now_ms)?
				.ok_or(rusqlite::Error::QueryReturnedNoRows.into());
		}
		let generation = bump_generation(&transaction)?;
		let id = if let Some((id, ..)) = existing {
			transaction.execute(
				"UPDATE credentials SET state = 1, disabled_cause = '', expires_at_ms = ?2, \
				 updated_at_ms = ?3, secret = ?4, refresh_secret = COALESCE(?5, refresh_secret), \
				 props = COALESCE(?6, props), aws_access_key = ?7, aws_session_token = ?8, generation \
				 = ?9 WHERE id = ?1",
				params![
					SqlU64(id),
					SqlU64(expires_at_ms),
					SqlU64(now_ms),
					secret,
					refresh_token,
					props.as_ref().map(Secret::expose),
					aws_access_key,
					aws_session_token,
					SqlU64(generation)
				],
			)?;
			id
		} else {
			transaction.execute(
				"INSERT INTO credentials (provider, kind, identity, state, disabled_cause, \
				 expires_at_ms, created_at_ms, updated_at_ms, secret, refresh_secret, props, \
				 aws_access_key, aws_session_token, generation) VALUES (?1, ?2, ?3, 1, '', ?4, ?5, \
				 ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
				params![
					provider,
					kind.as_i64(),
					identity,
					SqlU64(expires_at_ms),
					SqlU64(now_ms),
					secret,
					refresh_token,
					props.as_ref().map(Secret::expose),
					aws_access_key,
					aws_session_token,
					SqlU64(generation)
				],
			)?;
			u64::try_from(transaction.last_insert_rowid()).unwrap_or_default()
		};
		record_delta(&transaction, generation, id, DeltaKind::Upserted, now_ms)?;
		transaction.commit()?;
		get_metadata(&connection, id, now_ms)?.ok_or(rusqlite::Error::QueryReturnedNoRows.into())
	}

	fn set_state(
		&self,
		id: u64,
		state: CredentialState,
		cause: &str,
		now_ms: u64,
	) -> Result<Option<CredentialMeta>, StoreError> {
		let mut connection = self.connection.lock();
		let transaction = connection.transaction()?;
		if !credential_exists(&transaction, id)? {
			return Ok(None);
		}
		let generation = bump_generation(&transaction)?;
		transaction.execute(
			"UPDATE credentials SET state = ?2, disabled_cause = ?3, updated_at_ms = ?4, generation \
			 = ?5 WHERE id = ?1",
			params![SqlU64(id), state.as_i64(), cause, SqlU64(now_ms), SqlU64(generation)],
		)?;
		record_delta(&transaction, generation, id, DeltaKind::Upserted, now_ms)?;
		transaction.commit()?;
		get_metadata(&connection, id, now_ms).map_err(Into::into)
	}

	const fn cursor_at(&self, generation: u64) -> Cursor {
		Cursor { epoch: self.epoch, generation }
	}
}

fn migrate(connection: &Connection) -> Result<(), StoreError> {
	let transaction = connection.unchecked_transaction()?;
	transaction.execute_batch(
		"CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL); INSERT INTO \
		 schema_version (version) SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM schema_version);",
	)?;
	let version = transaction
		.query_row("SELECT version FROM schema_version LIMIT 1", [], |row| row.get::<_, SqlU64>(0))?
		.0;
	if version > SCHEMA_VERSION {
		return Err(StoreError::NewerSchema { found: version, supported: SCHEMA_VERSION });
	}
	if version == 0 {
		transaction.execute_batch(
			"CREATE TABLE credentials ( id INTEGER PRIMARY KEY, provider TEXT NOT NULL, kind INTEGER \
			 NOT NULL, identity TEXT NOT NULL, state INTEGER NOT NULL, disabled_cause TEXT NOT NULL, \
			 expires_at_ms INTEGER NOT NULL, created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER \
			 NOT NULL, secret BLOB NOT NULL, refresh_secret BLOB, props BLOB, aws_access_key BLOB, \
			 aws_session_token BLOB, generation INTEGER NOT NULL, UNIQUE(provider, kind, identity)); \
			 CREATE TABLE credential_blocks ( credential_id INTEGER NOT NULL REFERENCES \
			 credentials(id) ON DELETE CASCADE, scope TEXT NOT NULL, provider_key TEXT NOT NULL, \
			 until_ms INTEGER NOT NULL, PRIMARY KEY(credential_id, scope, provider_key)); CREATE \
			 TABLE usage_reports ( credential_id INTEGER PRIMARY KEY REFERENCES credentials(id) ON \
			 DELETE CASCADE, provider TEXT NOT NULL, report BLOB NOT NULL, fetched_at_ms INTEGER NOT \
			 NULL, stale INTEGER NOT NULL DEFAULT 0); CREATE TABLE usage_history ( id INTEGER \
			 PRIMARY KEY, credential_id INTEGER NOT NULL REFERENCES credentials(id) ON DELETE \
			 CASCADE, at_ms INTEGER NOT NULL, report BLOB NOT NULL); CREATE INDEX \
			 usage_history_lookup ON usage_history(credential_id, at_ms); CREATE TABLE client_usage \
			 ( client_id TEXT PRIMARY KEY, label TEXT NOT NULL, input_tokens INTEGER NOT NULL, \
			 output_tokens INTEGER NOT NULL, cache_read_tokens INTEGER NOT NULL, cache_write_tokens \
			 INTEGER NOT NULL, nanos_usd INTEGER NOT NULL, last_seen_ms INTEGER NOT NULL); CREATE \
			 TABLE store_meta (generation INTEGER NOT NULL); INSERT INTO store_meta (generation) \
			 VALUES (0); CREATE TABLE credential_deltas ( generation INTEGER PRIMARY KEY, \
			 credential_id INTEGER NOT NULL, kind INTEGER NOT NULL, at_ms INTEGER NOT NULL); CREATE \
			 TABLE provider_usage_history ( provider TEXT NOT NULL, credential_id INTEGER NOT NULL, \
			 account TEXT NOT NULL, turn_id TEXT NOT NULL, model TEXT NOT NULL, \
			 premium_multiplier_millionths INTEGER NOT NULL, input_tokens INTEGER NOT NULL, \
			 output_tokens INTEGER NOT NULL, cache_read_tokens INTEGER NOT NULL, cache_write_tokens \
			 INTEGER NOT NULL, nanos_usd INTEGER NOT NULL, at_ms INTEGER NOT NULL, PRIMARY \
			 KEY(turn_id)); CREATE INDEX provider_usage_window ON provider_usage_history(provider, \
			 credential_id, account, at_ms); UPDATE schema_version SET version = 5;",
		)?;
	} else {
		if version == 1 {
			transaction.execute_batch(
				"ALTER TABLE credentials ADD COLUMN refresh_secret BLOB; ALTER TABLE credentials ADD \
				 COLUMN props BLOB; UPDATE schema_version SET version = 2;",
			)?;
		}
		if version <= 2 {
			transaction.execute_batch(
				"ALTER TABLE credentials ADD COLUMN aws_access_key BLOB; ALTER TABLE credentials ADD \
				 COLUMN aws_session_token BLOB; UPDATE schema_version SET version = 3;",
			)?;
		}
		if version <= 3 {
			transaction.execute_batch(
				"CREATE TABLE provider_usage_history ( provider TEXT NOT NULL, credential_id INTEGER \
				 NOT NULL, account TEXT NOT NULL, turn_id TEXT NOT NULL, input_tokens INTEGER NOT \
				 NULL, output_tokens INTEGER NOT NULL, cache_read_tokens INTEGER NOT NULL, \
				 cache_write_tokens INTEGER NOT NULL, nanos_usd INTEGER NOT NULL, at_ms INTEGER NOT \
				 NULL, PRIMARY KEY(turn_id)); CREATE INDEX provider_usage_window ON \
				 provider_usage_history(provider, credential_id, account, at_ms); UPDATE \
				 schema_version SET version = 4;",
			)?;
		}
		if version <= 4 {
			transaction.execute_batch(
				"ALTER TABLE provider_usage_history ADD COLUMN model TEXT NOT NULL DEFAULT ''; ALTER \
				 TABLE provider_usage_history ADD COLUMN premium_multiplier_millionths INTEGER NOT \
				 NULL DEFAULT 0; UPDATE schema_version SET version = 5;",
			)?;
		}
	}
	transaction.commit()?;
	connection.pragma_update(None, "foreign_keys", "ON")?;
	Ok(())
}

fn current_generation(connection: &Connection) -> rusqlite::Result<u64> {
	connection
		.query_row("SELECT generation FROM store_meta", [], |row| row.get::<_, SqlU64>(0))
		.map(|value| value.0)
}

fn bump_generation(transaction: &Transaction<'_>) -> Result<u64, StoreError> {
	let current = current_generation(transaction)?;
	let generation = current
		.checked_add(1)
		.ok_or(StoreError::GenerationExhausted)?;
	transaction.execute("UPDATE store_meta SET generation = ?1", [SqlU64(generation)])?;
	Ok(generation)
}

fn record_delta(
	transaction: &Transaction<'_>,
	generation: u64,
	credential_id: u64,
	kind: DeltaKind,
	at_ms: u64,
) -> rusqlite::Result<()> {
	let kind = if kind == DeltaKind::Deleted { 2 } else { 1 };
	transaction.execute(
		"INSERT INTO credential_deltas (generation, credential_id, kind, at_ms) VALUES (?1, ?2, ?3, \
		 ?4)",
		params![SqlU64(generation), SqlU64(credential_id), kind, SqlU64(at_ms)],
	)?;
	transaction.execute("DELETE FROM credential_deltas WHERE generation <= ?1", [SqlU64(
		generation.saturating_sub(MAX_DELTAS),
	)])?;
	Ok(())
}

fn credential_exists(transaction: &Transaction<'_>, id: u64) -> rusqlite::Result<bool> {
	transaction.query_row(
		"SELECT EXISTS(SELECT 1 FROM credentials WHERE id = ?1)",
		[SqlU64(id)],
		|row| row.get(0),
	)
}

fn set_credential_generation(
	transaction: &Transaction<'_>,
	id: u64,
	generation: u64,
	now_ms: u64,
) -> rusqlite::Result<()> {
	transaction.execute(
		"UPDATE credentials SET generation = ?2, updated_at_ms = ?3 WHERE id = ?1",
		params![SqlU64(id), SqlU64(generation), SqlU64(now_ms)],
	)?;
	Ok(())
}

fn metadata_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<CredentialMeta> {
	Ok(CredentialMeta {
		id:             row.get::<_, SqlU64>(0)?.0,
		provider:       row_str(row, 1)?,
		kind:           CredentialKind::from_i64(row.get(2)?),
		identity:       row_str(row, 3)?,
		state:          CredentialState::from_stored(row.get(4)?),
		blocks:         SmallVec::new(),
		disabled_cause: row_str(row, 5)?,
		expires_at_ms:  row.get::<_, SqlU64>(6)?.0,
		created_at_ms:  row.get::<_, SqlU64>(7)?.0,
		updated_at_ms:  row.get::<_, SqlU64>(8)?.0,
	})
}

fn get_metadata(
	connection: &Connection,
	id: u64,
	now_ms: u64,
) -> rusqlite::Result<Option<CredentialMeta>> {
	let mut meta = connection
		.query_row(
			"SELECT id, provider, kind, identity, state, disabled_cause, expires_at_ms, \
			 created_at_ms, updated_at_ms FROM credentials WHERE id = ?1",
			[SqlU64(id)],
			metadata_row,
		)
		.optional()?;
	if let Some(meta) = &mut meta {
		load_blocks(connection, meta, now_ms)?;
	}
	Ok(meta)
}

fn load_blocks(
	connection: &Connection,
	meta: &mut CredentialMeta,
	now_ms: u64,
) -> rusqlite::Result<()> {
	let mut statement = connection.prepare(
		"SELECT scope, provider_key, until_ms FROM credential_blocks WHERE credential_id = ?1 AND \
		 until_ms > ?2 ORDER BY scope, provider_key",
	)?;
	meta.blocks = statement
		.query_map(params![SqlU64(meta.id), SqlU64(now_ms)], |row| {
			Ok(CredentialBlock {
				scope:        row_str(row, 0)?,
				provider_key: row_str(row, 1)?,
				until_ms:     row.get::<_, SqlU64>(2)?.0,
			})
		})?
		.collect::<Result<SmallVec<_, 4>, _>>()?;
	if meta.state == CredentialState::Active && !meta.blocks.is_empty() {
		meta.state = CredentialState::Blocked;
	}
	Ok(())
}

#[derive(Default, Deserialize)]
#[allow(
	clippy::struct_field_names,
	reason = "the id suffixes distinguish the provider metadata wire fields they deserialize"
)]
struct MetadataNamespace {
	#[serde(alias = "accountId")]
	account_id:      Option<Str>,
	#[serde(alias = "projectId")]
	project_id:      Option<Str>,
	#[serde(alias = "organizationId", alias = "org_id", alias = "orgId")]
	organization_id: Option<Str>,
}

#[derive(Default, Deserialize)]
struct MetadataProjection {
	#[serde(alias = "accountId")]
	account_id:       Option<Str>,
	#[serde(alias = "projectId")]
	project_id:       Option<Str>,
	#[serde(alias = "quotaProjectId")]
	quota_project_id: Option<Str>,
	#[serde(alias = "organizationId", alias = "org_id", alias = "orgId")]
	organization_id:  Option<Str>,
	openai:           Option<MetadataNamespace>,
	codex:            Option<MetadataNamespace>,
	google:           Option<MetadataNamespace>,
	antigravity:      Option<MetadataNamespace>,
	zai:              Option<MetadataNamespace>,
}

fn first_metadata<const N: usize>(values: [Option<&Str>; N]) -> Option<Str> {
	values
		.into_iter()
		.flatten()
		.find(|value| !value.is_empty())
		.cloned()
}

fn row_str(row: &rusqlite::Row<'_>, index: usize) -> rusqlite::Result<Str> {
	let value = row.get_ref(index)?;
	let text = value.as_str().map_err(|error| {
		rusqlite::Error::FromSqlConversionFailure(index, rusqlite::types::Type::Text, Box::new(error))
	})?;
	Ok(Str::new(text))
}

#[cfg(test)]
mod tests {
	use http::header::AUTHORIZATION;
	use omp_core::Str;
	use serde_json::json;
	use tempfile::tempdir;

	use super::{
		CredentialBlock, CredentialFilter, CredentialMeta, CredentialState, RankedCredential,
		SCHEMA_VERSION, SqlU64, Store, UsageReport, UsageWindow,
	};

	fn report(
		credential_id: u64,
		provider: &str,
		now_ms: u64,
		windows: &[(f64, u64)],
	) -> UsageReport {
		UsageReport {
			credential_id,
			provider: Str::new(provider),
			plan: Str::new("test"),
			windows: windows
				.iter()
				.enumerate()
				.map(|(index, (used_percent, resets_at_ms))| UsageWindow {
					label:        Str::new(if index == 0 { "primary" } else { "secondary" }),
					used_percent: *used_percent,
					resets_at_ms: *resets_at_ms,
				})
				.collect(),
			fetched_at_ms: now_ms,
			detail: json!({}),
		}
	}
	fn store() -> (tempfile::TempDir, Store) {
		let directory = tempdir().expect("tempdir");
		let store = Store::open(directory.path().join("broker.sqlite")).expect("open store");
		(directory, store)
	}

	#[test]
	fn opencode_go_monthly_window_is_display_only_for_ranking() {
		let mut usage = report(
			1,
			"opencode-go",
			10,
			&[(10.0, 200), (20.0, 50), (100.0, 200)],
		);
		usage.windows[0].label = Str::new("rolling-5h");
		usage.windows[1].label = Str::new("weekly");
		usage.windows[2].label = Str::new("monthly");
		let credential = RankedCredential {
			id: 1,
			generation: 1,
			blocked_until: None,
			report: Some(usage),
		};
		assert_eq!(credential.window(0, None, 100).expect("rolling window").label, "rolling-5h");
		assert!(credential.window(1, None, 100).is_none());
	}

	#[test]
	fn schema_creation_and_migration_are_idempotent() {
		let directory = tempdir().expect("tempdir");
		let path = directory.path().join("broker.sqlite");
		drop(Store::open(&path).expect("first open"));
		drop(Store::open(&path).expect("second open"));

		let connection = rusqlite::Connection::open(path).expect("inspect database");
		let version = connection
			.query_row("SELECT version FROM schema_version", [], |row| row.get::<_, SqlU64>(0))
			.expect("schema version")
			.0;
		assert_eq!(version, SCHEMA_VERSION);
		for table in [
			"credentials",
			"credential_blocks",
			"usage_reports",
			"usage_history",
			"client_usage",
			"provider_usage_history",
		] {
			let exists: bool = connection
				.query_row(
					"SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
					[table],
					|row| row.get(0),
				)
				.expect("table lookup");
			assert!(exists, "missing {table}");
		}
		let lease_tables = connection
			.query_row(
				"SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name LIKE '%lease%'",
				[],
				|row| row.get::<_, SqlU64>(0),
			)
			.expect("lease lookup")
			.0;
		assert_eq!(lease_tables, 0);
	}

	#[test]
	fn version_one_store_migrates_oauth_and_aws_columns() {
		let directory = tempdir().expect("tempdir");
		let path = directory.path().join("broker.sqlite");
		let connection = rusqlite::Connection::open(&path).expect("create version one store");
		connection
			.execute_batch(
				"CREATE TABLE schema_version (version INTEGER NOT NULL); INSERT INTO schema_version \
				 (version) VALUES (1); CREATE TABLE credentials ( id INTEGER PRIMARY KEY, provider \
				 TEXT NOT NULL, kind INTEGER NOT NULL, identity TEXT NOT NULL, state INTEGER NOT \
				 NULL, disabled_cause TEXT NOT NULL, expires_at_ms INTEGER NOT NULL, created_at_ms \
				 INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL, secret BLOB NOT NULL, generation \
				 INTEGER NOT NULL, UNIQUE(provider, kind, identity));",
			)
			.expect("create version one schema");
		drop(connection);

		drop(Store::open(&path).expect("migrate store"));
		let connection = rusqlite::Connection::open(path).expect("inspect migrated store");
		let version = connection
			.query_row("SELECT version FROM schema_version", [], |row| row.get::<_, SqlU64>(0))
			.expect("schema version")
			.0;
		assert_eq!(version, SCHEMA_VERSION);
		for column in ["refresh_secret", "props", "aws_access_key", "aws_session_token"] {
			let exists: bool = connection
				.query_row(
					"SELECT EXISTS(SELECT 1 FROM pragma_table_info('credentials') WHERE name = ?1)",
					[column],
					|row| row.get(0),
				)
				.expect("column lookup");
			assert!(exists, "missing {column}");
		}
		for column in ["model", "premium_multiplier_millionths"] {
			let exists: bool = connection
				.query_row(
					"SELECT EXISTS(SELECT 1 FROM pragma_table_info('provider_usage_history') WHERE \
					 name = ?1)",
					[column],
					|row| row.get(0),
				)
				.expect("history column lookup");
			assert!(exists, "missing {column}");
		}
	}

	#[test]
	fn secret_round_trips_only_through_lease_redemption() {
		let (_directory, store) = store();
		let material = "api-key-material";
		let meta = store
			.upsert_api_key("example", "account", material.as_bytes(), 10)
			.expect("upsert");
		let metadata_debug = format!("{meta:?}");
		assert!(!metadata_debug.contains(material));

		let lease = store.lease(meta.id).expect("lease lookup").expect("lease");
		let auth = store
			.redeem_with(
				lease.provider(),
				lease.credential_id(),
				lease.generation(),
				Ok::<_, std::convert::Infallible>,
			)
			.expect("redeem lookup")
			.expect("current generation")
			.expect("infallible apply");
		let mut headers = http::HeaderMap::new();
		auth
			.apply_bearer_to_headers(&mut headers)
			.expect("apply auth");
		assert_eq!(headers[AUTHORIZATION], "Bearer api-key-material");
	}

	#[test]
	fn oauth_refresh_and_props_remain_internal() {
		let (_directory, store) = store();
		let props = json!({ "openai": { "account_id": "account-7" } });
		let meta = store
			.upsert_oauth_material(
				"openai-codex",
				"account@example.test",
				b"access",
				Some(b"refresh"),
				&props,
				1_000,
				10,
			)
			.expect("upsert OAuth material");
		assert_eq!(
			store
				.redeem_refresh(meta.id)
				.expect("refresh lookup")
				.expect("refresh material")
				.expose(),
			b"refresh"
		);
		assert_eq!(store.oauth_props(meta.id).expect("props lookup"), Some(props));
		assert!(!format!("{meta:?}").contains("access"));
		assert!(!format!("{meta:?}").contains("refresh"));
	}

	#[test]
	fn aws_tuple_redeems_as_one_sealed_value_and_never_formats_material() {
		let (_directory, store) = store();
		let access = "AKIDEXAMPLE";
		let secret = "aws-secret-material";
		let session = "aws-session-material";
		let meta = store
			.upsert_aws(
				"bedrock",
				"account",
				access.as_bytes(),
				secret.as_bytes(),
				Some(session.as_bytes()),
				0,
				10,
			)
			.expect("upsert AWS");
		assert_eq!(meta.kind, super::CredentialKind::Aws);
		let debug = format!("{meta:?}");
		assert!(!debug.contains(access));
		assert!(!debug.contains(secret));
		assert!(!debug.contains(session));
		let lease = store.lease(meta.id).expect("lease lookup").expect("lease");
		let auth_debug = store
			.redeem_typed_with(
				lease.provider(),
				lease.credential_id(),
				lease.generation(),
				|kind, auth| {
					assert_eq!(kind, super::CredentialKind::Aws);
					Ok::<_, std::convert::Infallible>(format!("{auth:?}"))
				},
			)
			.expect("redeem lookup")
			.expect("AWS row")
			.expect("infallible");
		assert_eq!(auth_debug, "AppliedAuth([redacted])");
		assert!(!auth_debug.contains(access));
		assert!(!auth_debug.contains(secret));
		assert!(!auth_debug.contains(session));
	}

	#[test]
	fn rotated_credential_invalidates_old_lease() {
		let (_directory, store) = store();
		let first = store
			.upsert_api_key("example", "account", b"old", 10)
			.expect("first upsert");
		let stale = store.lease(first.id).expect("lease lookup").expect("lease");
		store
			.upsert_api_key("example", "account", b"new", 11)
			.expect("rotation");
		assert!(
			store
				.redeem_with(stale.provider(), stale.credential_id(), stale.generation(), |_| Ok::<
					_,
					std::convert::Infallible,
				>(()),)
				.expect("redeem lookup")
				.is_none()
		);
	}

	#[test]
	fn provider_lease_skips_live_blocks_and_preserves_stable_rank() {
		let (_directory, store) = store();
		let first = store
			.upsert_api_key("example", "first", b"one", 10)
			.expect("first credential");
		let second = store
			.upsert_api_key("example", "second", b"two", 11)
			.expect("second credential");
		assert_eq!(
			store
				.lease_provider("example", 20)
				.expect("provider lease")
				.expect("credential")
				.credential_id(),
			first.id
		);
		store
			.report_block(
				first.id,
				&CredentialBlock {
					scope:        "chat".into(),
					provider_key: "example".into(),
					until_ms:     100,
				},
				21,
			)
			.expect("block first");
		assert_eq!(
			store
				.lease_provider("example", 22)
				.expect("provider lease")
				.expect("credential")
				.credential_id(),
			second.id
		);
		assert_eq!(
			store
				.lease_provider("example", 101)
				.expect("provider lease")
				.expect("credential")
				.credential_id(),
			first.id
		);
	}

	#[test]
	fn provider_ranking_demotes_hot_primary_window_and_ignores_expired_windows() {
		let (_directory, store) = store();
		let hot = store
			.upsert_api_key("ranked", "hot", b"one", 1)
			.expect("hot");
		let cool = store
			.upsert_api_key("ranked", "cool", b"two", 2)
			.expect("cool");
		store
			.write_usage_report(&report(hot.id, "ranked", 10, &[(90.0, 10_000)]), 10)
			.expect("hot report");
		store
			.write_usage_report(&report(cool.id, "ranked", 10, &[(20.0, 10_000)]), 10)
			.expect("cool report");
		assert_eq!(
			store
				.lease_provider("ranked", 100)
				.expect("lease")
				.expect("candidate")
				.credential_id(),
			cool.id
		);

		store
			.write_usage_report(&report(hot.id, "ranked", 20, &[(90.0, 99)]), 20)
			.expect("expired report");
		store
			.mark_usage_stale(Some("ranked"), Some(cool.id))
			.expect("hide cool report");
		assert_eq!(
			store
				.lease_provider("ranked", 100)
				.expect("lease")
				.expect("candidate")
				.credential_id(),
			hot.id
		);
	}

	#[test]
	fn provider_ranking_chases_secondary_quota_with_greater_drain_urgency() {
		let (_directory, store) = store();
		let urgent = store
			.upsert_api_key("ranked", "urgent", b"one", 1)
			.expect("urgent");
		let relaxed = store
			.upsert_api_key("ranked", "relaxed", b"two", 2)
			.expect("relaxed");
		store
			.write_usage_report(
				&report(urgent.id, "ranked", 10, &[(20.0, 20_000), (20.0, 3_700_000)]),
				10,
			)
			.expect("urgent report");
		store
			.write_usage_report(
				&report(relaxed.id, "ranked", 10, &[(20.0, 20_000), (20.0, 604_800_100)]),
				10,
			)
			.expect("relaxed report");
		assert_eq!(
			store
				.lease_provider("ranked", 100)
				.expect("lease")
				.expect("candidate")
				.credential_id(),
			urgent.id
		);
	}

	#[test]
	fn provider_ranking_uses_stable_ties_and_never_crosses_provider_reports() {
		let (_directory, store) = store();
		let first = store
			.upsert_api_key("ranked", "first", b"one", 1)
			.expect("first");
		let second = store
			.upsert_api_key("ranked", "second", b"two", 2)
			.expect("second");
		let foreign = store
			.upsert_api_key("other", "foreign", b"three", 3)
			.expect("foreign");
		store
			.write_usage_report(&report(foreign.id, "other", 10, &[(99.0, 10_000)]), 10)
			.expect("foreign report");
		assert_eq!(
			store
				.lease_provider("ranked", 100)
				.expect("lease")
				.expect("candidate")
				.credential_id(),
			first.id
		);
		assert_ne!(first.id, second.id);
	}

	#[test]
	fn ranked_id_api_honors_model_and_account_scope_without_leaking_across_accounts() {
		let (_directory, store) = store();
		let first = store
			.upsert_api_key("ranked", "account-a", b"one", 1)
			.expect("first");
		let second = store
			.upsert_api_key("ranked", "account-b", b"two", 2)
			.expect("second");
		let mut first_report = report(first.id, "ranked", 10, &[(90.0, 10_000), (10.0, 10_000)]);
		first_report.windows[0].label = "model-a:5h".into();
		first_report.windows[1].label = "model-b:5h".into();
		let mut second_report = report(second.id, "ranked", 10, &[(10.0, 10_000), (90.0, 10_000)]);
		second_report.windows[0].label = "model-a:5h".into();
		second_report.windows[1].label = "model-b:5h".into();
		store
			.write_usage_report(&first_report, 10)
			.expect("first report");
		store
			.write_usage_report(&second_report, 10)
			.expect("second report");

		assert_eq!(
			store
				.ranked_credential_ids("ranked", Some("model-a"), None, 100)
				.expect("model-a rank")
				.as_slice(),
			[second.id, first.id]
		);
		assert_eq!(
			store
				.ranked_credential_ids("ranked", Some("model-b"), None, 100)
				.expect("model-b rank")
				.as_slice(),
			[first.id, second.id]
		);
		assert_eq!(
			store
				.ranked_credential_ids("ranked", Some("model-a"), Some("account-a"), 100)
				.expect("account rank")
				.as_slice(),
			[first.id]
		);
	}

	#[test]
	fn blocked_candidates_are_last_resort_then_earliest_unblocking_first() {
		let (_directory, store) = store();
		let first = store
			.upsert_api_key("ranked", "first", b"one", 1)
			.expect("first");
		let second = store
			.upsert_api_key("ranked", "second", b"two", 2)
			.expect("second");
		for (id, until_ms) in [(first.id, 500), (second.id, 300)] {
			store
				.report_block(
					id,
					&CredentialBlock { scope: "chat".into(), provider_key: "ranked".into(), until_ms },
					10,
				)
				.expect("block");
		}
		assert_eq!(
			store
				.lease_provider("ranked", 100)
				.expect("lease")
				.expect("candidate")
				.credential_id(),
			second.id
		);
	}

	#[test]
	fn anthropic_cache_warm_selection_sticks_then_expires() {
		let (_directory, store) = store();
		let first = store
			.upsert_api_key("anthropic", "first", b"one", 1)
			.expect("first");
		let second = store
			.upsert_api_key("anthropic", "second", b"two", 2)
			.expect("second");
		store
			.write_usage_report(&report(first.id, "anthropic", 10, &[(10.0, 10_000_000)]), 10)
			.expect("first report");
		assert_eq!(
			store
				.lease_provider("anthropic", 100)
				.expect("lease")
				.expect("candidate")
				.credential_id(),
			first.id
		);
		store
			.write_usage_report(&report(first.id, "anthropic", 200, &[(90.0, 10_000_000)]), 200)
			.expect("first hot");
		store
			.write_usage_report(&report(second.id, "anthropic", 200, &[(10.0, 10_000_000)]), 200)
			.expect("second cool");
		assert_eq!(
			store
				.lease_provider("anthropic", 300)
				.expect("lease")
				.expect("candidate")
				.credential_id(),
			first.id
		);
		assert_eq!(
			store
				.lease_provider("anthropic", 300 + super::ANTHROPIC_CACHE_WARM_MS)
				.expect("lease")
				.expect("candidate")
				.credential_id(),
			second.id
		);
	}

	#[test]
	fn scoped_blocks_coexist_and_expire_independently() {
		let (_directory, store) = store();
		let meta = store
			.upsert_api_key("example", "", b"key", 10)
			.expect("upsert");
		store
			.report_block(
				meta.id,
				&CredentialBlock {
					scope:        Str::new("chat"),
					provider_key: Str::new("primary"),
					until_ms:     100,
				},
				20,
			)
			.expect("chat block");
		store
			.report_block(
				meta.id,
				&CredentialBlock {
					scope:        Str::new("spark"),
					provider_key: Str::new("primary"),
					until_ms:     200,
				},
				21,
			)
			.expect("spark block");

		let at_50 = store
			.get_credential(meta.id, 50)
			.expect("get")
			.expect("meta");
		assert_eq!(at_50.state, CredentialState::Blocked);
		assert_eq!(at_50.blocks.len(), 2);
		let at_150 = store
			.get_credential(meta.id, 150)
			.expect("get")
			.expect("meta");
		assert_eq!(at_150.blocks.len(), 1);
		assert_eq!(at_150.blocks[0].scope, "spark");
		let at_250 = store
			.get_credential(meta.id, 250)
			.expect("get")
			.expect("meta");
		assert_eq!(at_250.state, CredentialState::Active);
		assert!(at_250.blocks.is_empty());
	}

	#[test]
	fn large_timestamp_round_trips_at_signed_sqlite_limit() {
		let (_directory, store) = store();
		let timestamp = i64::MAX as u64;
		let inserted = store
			.upsert_api_key("example", "large-timestamp", b"key", timestamp)
			.expect("upsert at SQLite integer limit");
		let loaded = store
			.get_credential(inserted.id, timestamp)
			.expect("get")
			.expect("metadata");
		assert_eq!(loaded.created_at_ms, timestamp);
		assert_eq!(loaded.updated_at_ms, timestamp);
	}

	#[test]
	fn unsigned_sql_value_above_signed_limit_is_rejected() {
		let (_directory, store) = store();
		let error = store
			.upsert_api_key("example", "overflow", b"key", i64::MAX as u64 + 1)
			.expect_err("out-of-range timestamp must fail");
		assert!(matches!(
			error,
			super::StoreError::Sqlite(rusqlite::Error::ToSqlConversionFailure(_))
		));
	}

	#[test]
	fn generation_is_monotone_across_mutations() {
		let (_directory, store) = store();
		let zero = store.cursor().expect("initial cursor").generation;
		let meta = store
			.upsert_api_key("example", "", b"key", 10)
			.expect("upsert");
		let one = store.cursor().expect("cursor one").generation;
		store
			.disable_credential(meta.id, "operator", 11)
			.expect("disable");
		let two = store.cursor().expect("cursor two").generation;
		store.enable_credential(meta.id, 12).expect("enable");
		let three = store.cursor().expect("cursor three").generation;
		store.delete_credential(meta.id, 13).expect("delete");
		let four = store.cursor().expect("cursor four").generation;
		assert!(zero < one && one < two && two < three && three < four);
		let credentials = store
			.list_credentials(&CredentialFilter { now_ms: 20, ..CredentialFilter::default() })
			.expect("list");
		assert_eq!(credentials, [] as [CredentialMeta; 0]);
	}
}
