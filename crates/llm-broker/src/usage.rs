//! Data-driven provider quota fetching and durable cache coordination.

use std::{
	sync::Arc,
	time::{SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use futures::future::BoxFuture;
use http::{
	Method, Request, StatusCode,
	header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderValue},
};
use jiff::{Timestamp, ToSpan, tz::TimeZone};
use omp_core::{Str, fmts};
use omp_llm_egress::{
	auth_inject::CredentialLease,
	limits::{BlockSink, CredentialBlock as EgressCredentialBlock},
};
use serde_json::Value;
use smallvec::SmallVec;
use thiserror::Error;

use crate::{
	sealed::AppliedAuth,
	store::{
		ClientUsage, CredentialBlock, CredentialFilter, CredentialMeta, RollingSpend, Store,
		StoreError, UsageReport, UsageWindow,
	},
};

const DEFAULT_CACHE_TTL_MS: u64 = 60_000;

/// Authentication placement for a provider usage request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthKind {
	/// `Authorization: Bearer <credential>`.
	Bearer,
	/// `Authorization: <credential>`.
	Authorization,
	/// Cursor's `WorkosCursorSessionToken` cookie.
	CursorCookie,
}

/// One quota window selected from a JSON response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowSpec {
	/// Stable display label.
	pub label:              &'static str,
	/// Dot-separated JSON path to a numeric used percentage.
	pub used_percent_path:  &'static str,
	/// Dot-separated JSON path to an epoch or RFC 3339 reset time.
	pub resets_at_path:     &'static str,
	/// Multiplier applied to the selected utilization value.
	pub used_percent_scale: f64,
	/// Whether the selected number is remaining rather than used.
	pub remaining:          bool,
}

/// Declarative description of a single-request provider usage endpoint.
#[derive(Clone, Copy, Debug)]
pub struct UsageFetcher {
	/// Catalog provider identifier.
	pub provider:     &'static str,
	/// Absolute endpoint template. `{base}` may be replaced by a caller
	/// override.
	pub url_template: &'static str,
	/// Credential placement on the request.
	pub auth_kind:    AuthKind,
	/// JSON path to the provider plan label.
	pub plan_path:    &'static str,
	/// Quota windows extracted from the response.
	pub windows:      &'static [WindowSpec],
}

const ANTHROPIC_WINDOWS: &[WindowSpec] = &[
	WindowSpec {
		label:              "5h",
		used_percent_path:  "five_hour.utilization",
		resets_at_path:     "five_hour.resets_at",
		used_percent_scale: 1.0,
		remaining:          false,
	},
	WindowSpec {
		label:              "week",
		used_percent_path:  "seven_day.utilization",
		resets_at_path:     "seven_day.resets_at",
		used_percent_scale: 1.0,
		remaining:          false,
	},
	WindowSpec {
		label:              "opus-week",
		used_percent_path:  "seven_day_opus.utilization",
		resets_at_path:     "seven_day_opus.resets_at",
		used_percent_scale: 1.0,
		remaining:          false,
	},
];
const CODEX_WINDOWS: &[WindowSpec] = &[
	WindowSpec {
		label:              "primary",
		used_percent_path:  "rate_limit.primary_window.used_percent",
		resets_at_path:     "rate_limit.primary_window.reset_at",
		used_percent_scale: 1.0,
		remaining:          false,
	},
	WindowSpec {
		label:              "secondary",
		used_percent_path:  "rate_limit.secondary_window.used_percent",
		resets_at_path:     "rate_limit.secondary_window.reset_at",
		used_percent_scale: 1.0,
		remaining:          false,
	},
];
const KIMI_WINDOWS: &[WindowSpec] = &[WindowSpec {
	label:              "usage",
	used_percent_path:  "usage.used_percent",
	resets_at_path:     "usage.reset_at",
	used_percent_scale: 1.0,
	remaining:          false,
}];
const ZAI_WINDOWS: &[WindowSpec] = &[WindowSpec {
	label:              "quota",
	used_percent_path:  "data.limits.0.percentage",
	resets_at_path:     "data.limits.0.nextResetTime",
	used_percent_scale: 1.0,
	remaining:          false,
}];
const COPILOT_WINDOWS: &[WindowSpec] = &[
	WindowSpec {
		label:              "chat",
		used_percent_path:  "quota_snapshots.chat.percent_remaining",
		resets_at_path:     "quota_reset_date",
		used_percent_scale: 1.0,
		remaining:          true,
	},
	WindowSpec {
		label:              "premium",
		used_percent_path:  "quota_snapshots.premium_interactions.percent_remaining",
		resets_at_path:     "quota_reset_date",
		used_percent_scale: 1.0,
		remaining:          true,
	},
];
const MINIMAX_WINDOWS: &[WindowSpec] = &[];

/// Single-request provider descriptors.
///
/// A const table is used rather than TOML because the paths are part of the
/// broker's typed wire contract and should be reviewed and compiled with code.
pub const FETCHERS: &[UsageFetcher] = &[
	UsageFetcher {
		provider:     "anthropic",
		url_template: "https://api.anthropic.com/api/oauth/usage",
		auth_kind:    AuthKind::Bearer,
		plan_path:    "plan",
		windows:      ANTHROPIC_WINDOWS,
	},
	UsageFetcher {
		provider:     "openai-codex",
		url_template: "https://chatgpt.com/backend-api/wham/usage",
		auth_kind:    AuthKind::Bearer,
		plan_path:    "plan_type",
		windows:      CODEX_WINDOWS,
	},
	UsageFetcher {
		provider:     "github-copilot",
		url_template: "https://api.github.com/copilot_internal/user",
		auth_kind:    AuthKind::Bearer,
		plan_path:    "copilot_plan",
		windows:      COPILOT_WINDOWS,
	},
	UsageFetcher {
		provider:     "zai",
		url_template: "https://api.z.ai/api/monitor/usage/quota/limit",
		auth_kind:    AuthKind::Authorization,
		plan_path:    "data.plan",
		windows:      ZAI_WINDOWS,
	},
	UsageFetcher {
		provider:     "kimi-code",
		url_template: "https://api.kimi.com/coding/v1/usages",
		auth_kind:    AuthKind::Bearer,
		plan_path:    "plan",
		windows:      KIMI_WINDOWS,
	},
	UsageFetcher {
		provider:     "minimax-code",
		url_template: "https://api.minimax.io/v1/token_plan/remains",
		auth_kind:    AuthKind::Bearer,
		plan_path:    "",
		windows:      MINIMAX_WINDOWS,
	},
	UsageFetcher {
		provider:     "minimax-code-cn",
		url_template: "https://api.minimaxi.com/v1/token_plan/remains",
		auth_kind:    AuthKind::Bearer,
		plan_path:    "",
		windows:      MINIMAX_WINDOWS,
	},
];

/// HTTP response needed by usage fetchers.
#[derive(Clone, Debug)]
pub struct UsageHttpResponse {
	/// Provider status code.
	pub status: StatusCode,
	/// Raw response bytes.
	pub body:   Bytes,
}

/// Injected HTTP transport; usage fetching never constructs a client.
pub trait UsageHttp: Send + Sync {
	/// Executes one fully authenticated request.
	fn send(&self, request: Request<Bytes>) -> BoxFuture<'_, Result<UsageHttpResponse, UsageError>>;
}

/// Provider-specific code hook for genuinely non-declarative usage flows.
pub trait UsageOverride: Send + Sync {
	/// Provider catalog identifier handled by this hook.
	fn provider(&self) -> &'static str;

	/// Fetches and normalizes provider quota data.
	fn fetch<'a>(
		&'a self,
		context: OverrideContext<'a>,
	) -> BoxFuture<'a, Result<UsageReport, UsageError>>;
}

/// Inputs available to a provider override.
pub struct OverrideContext<'a> {
	/// Credential metadata, which contains no secret bytes.
	pub credential: &'a CredentialMeta,
	/// Exact credential generation to redeem for each upstream step.
	pub lease:      CredentialLease,
	/// Store used to redeem the sealed lease.
	pub store:      &'a Store,
	/// Injected HTTP transport.
	pub http:       &'a dyn UsageHttp,
	/// Fetch timestamp.
	pub now_ms:     u64,
}

/// Usage fetching or normalization failure.
#[derive(Debug, Error)]
pub enum UsageError {
	/// Credential metadata or cache persistence failed.
	#[error(transparent)]
	Store(#[from] StoreError),
	/// The credential disappeared or rotated during the fetch.
	#[error("credential {0} is no longer redeemable")]
	StaleCredential(u64),
	/// No usage implementation exists for a provider.
	#[error("provider {0} has no usage fetcher")]
	UnsupportedProvider(Str),
	/// Request construction failed.
	#[error("invalid usage request: {0}")]
	Request(#[from] http::Error),
	/// Authentication bytes are not a valid HTTP header.
	#[error("invalid credential header: {0}")]
	Header(#[from] http::header::InvalidHeaderValue),
	/// Upstream rejected the usage request.
	#[error("usage endpoint returned HTTP {0}")]
	Http(StatusCode),
	/// Upstream returned malformed JSON.
	#[error("invalid usage response JSON: {0}")]
	Json(#[from] serde_json::Error),
	/// An override could not normalize the provider response.
	#[error("invalid {provider} usage response: {message}")]
	InvalidResponse {
		/// Provider whose response was invalid.
		provider: Str,
		/// Client-safe reason.
		message:  Str,
	},
}

/// Outcome returned by `OpenAI` Codex saved-reset consumption.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodexResetConsume {
	/// `true` only when the provider reports that a reset was applied.
	pub applied: bool,
	/// Provider business outcome (`reset`, `already_redeemed`, and so on).
	pub code:    Str,
	/// HTTP status returned by the consume route.
	pub status:  StatusCode,
}

/// Cloneable, non-secret sink for observations made by the inference service.
///
/// Callers retain provider usage and credential identity only; credential
/// material never crosses this boundary. A cancelled request has no terminal
/// observation and must be represented by `None`.
#[derive(Clone)]
pub struct BrokerObserver {
	store: Arc<Store>,
}

impl BrokerObserver {
	/// Creates an observer over the daemon-owned durable store.
	#[must_use]
	pub const fn new(store: Arc<Store>) -> Self {
		Self { store }
	}

	/// Idempotently records usage, exact cost, and resolved model consumption
	/// for one selected terminal turn.
	///
	/// `None` usage denotes cancellation before a terminal outcome and performs
	/// no write. A missing premium multiplier defaults to one request only for
	/// GitHub Copilot. Copilot agent continuations consume no premium request,
	/// while a cached free plan clamps user turns to at least one request.
	pub fn record_terminal_observation(
		&self,
		lease: &CredentialLease,
		turn_id: &str,
		model: &str,
		initiator: &str,
		premium_multiplier_millionths: Option<u64>,
		usage: Option<&ClientUsage>,
		observed_at_ms: u64,
	) -> Result<bool, StoreError> {
		let Some(usage) = usage else {
			return Ok(false);
		};
		let premium_multiplier_millionths =
			self.effective_premium_multiplier(lease, initiator, premium_multiplier_millionths)?;
		self.store.record_terminal_usage(
			lease,
			turn_id,
			model,
			premium_multiplier_millionths,
			usage,
			observed_at_ms,
		)
	}

	fn effective_premium_multiplier(
		&self,
		lease: &CredentialLease,
		initiator: &str,
		resolved: Option<u64>,
	) -> Result<u64, StoreError> {
		if lease.provider() != "github-copilot" {
			return Ok(resolved.unwrap_or(0));
		}
		if initiator.trim().eq_ignore_ascii_case("agent") {
			return Ok(0);
		}
		let mut multiplier = resolved.unwrap_or(1_000_000);
		let free_plan = self
			.store
			.read_usage_reports(Some("github-copilot"), Some(lease.credential_id()))?
			.into_iter()
			.any(|report| report.plan.as_str().trim().eq_ignore_ascii_case("free"));
		if free_plan {
			multiplier = multiplier.max(1_000_000);
		}
		Ok(multiplier)
	}

	/// Atomically adds usage and exact cost from one unkeyed terminal turn.
	///
	/// `None` denotes a request cancelled before a terminal outcome and performs
	/// no write. Prefer [`Self::record_terminal_observation`] when a selected
	/// credential and stable turn identity are available.
	pub fn record_terminal_usage(&self, usage: Option<&ClientUsage>) -> Result<(), StoreError> {
		if let Some(usage) = usage {
			self.store.add_client_usage(usage)?;
		}
		Ok(())
	}

	/// Stores a provider quota report as both the latest snapshot and history.
	pub fn record_quota_report(
		&self,
		report: &UsageReport,
		observed_at_ms: u64,
	) -> Result<(), StoreError> {
		self.store.write_usage_report(report, observed_at_ms)
	}

	/// Persists a block only for the exact credential selected by `lease`.
	pub fn record_block(
		&self,
		lease: &CredentialLease,
		block: &CredentialBlock,
		observed_at_ms: u64,
	) -> Result<Option<CredentialMeta>, StoreError> {
		self
			.store
			.report_block(lease.credential_id(), block, observed_at_ms)
	}
}

impl BlockSink for BrokerObserver {
	fn record_block(&self, observation: &EgressCredentialBlock) {
		let scope = match observation.status {
			StatusCode::UNAUTHORIZED => "auth",
			StatusCode::FORBIDDEN => "permission",
			StatusCode::TOO_MANY_REQUESTS => "rate-limit",
			_ => return,
		};
		let until_ms = observation
			.blocked_until
			.duration_since(UNIX_EPOCH)
			.unwrap_or_default()
			.as_millis()
			.try_into()
			.unwrap_or(u64::MAX);
		let observed_at_ms = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.unwrap_or_default()
			.as_millis()
			.try_into()
			.unwrap_or(u64::MAX);
		let block = CredentialBlock {
			scope: scope.into(),
			provider_key: observation.key.provider().into(),
			until_ms,
		};
		if let Err(error) =
			self
				.store
				.report_block(observation.key.credential_id(), &block, observed_at_ms)
		{
			tracing::warn!(
				credential_id = observation.key.credential_id(),
				provider = observation.key.provider(),
				%error,
				"failed to persist provider credential block"
			);
		}
	}
}

/// Cache-coordinated provider usage facade.
pub struct UsageManager {
	store:        Arc<Store>,
	http:         Arc<dyn UsageHttp>,
	overrides:    SmallVec<Arc<dyn UsageOverride>, 4>,
	cache_ttl_ms: u64,
}

impl UsageManager {
	/// Creates a manager with the built-in overrides and durable cache TTL.
	#[must_use]
	pub fn new(store: Arc<Store>, http: Arc<dyn UsageHttp>) -> Self {
		Self { store, http, overrides: built_in_overrides(), cache_ttl_ms: DEFAULT_CACHE_TTL_MS }
	}

	/// Replaces the cache TTL, primarily for daemon policy and deterministic
	/// tests.
	#[must_use]
	pub const fn with_cache_ttl_ms(mut self, cache_ttl_ms: u64) -> Self {
		self.cache_ttl_ms = cache_ttl_ms;
		self
	}

	/// Replaces the provider override registry.
	#[must_use]
	pub fn with_overrides(mut self, overrides: SmallVec<Arc<dyn UsageOverride>, 4>) -> Self {
		self.overrides = overrides;
		self
	}

	/// Gets matching reports, using fresh cache entries unless `refresh` is set.
	pub async fn get_usage(
		&self,
		provider: Option<&str>,
		credential_id: Option<u64>,
		refresh: bool,
		now_ms: u64,
	) -> Result<Vec<UsageReport>, UsageError> {
		let credentials = self.store.list_credentials(&CredentialFilter {
			provider,
			now_ms,
			..CredentialFilter::default()
		})?;
		let mut reports = Vec::new();
		for credential in credentials
			.into_iter()
			.filter(|meta| credential_id.is_none_or(|id| meta.id == id))
		{
			let cached = self
				.store
				.read_usage_reports(Some(credential.provider.as_str()), Some(credential.id))?
				.pop();
			let cached_before_at = cached.as_ref().map(|report| report.fetched_at_ms);
			if !refresh
				&& let Some(report) = cached.as_ref()
				&& now_ms.saturating_sub(report.fetched_at_ms) < self.cache_ttl_ms
			{
				reports.push(report.clone());
				continue;
			}
			let _flight = self.store.refresh_singleflight(credential.id).await;
			let cached = self
				.store
				.read_usage_reports(Some(credential.provider.as_str()), Some(credential.id))?
				.pop();
			if let Some(report) = cached
				&& ((!refresh && now_ms.saturating_sub(report.fetched_at_ms) < self.cache_ttl_ms)
					|| (refresh && cached_before_at.is_none_or(|before| report.fetched_at_ms > before)))
			{
				reports.push(report);
				continue;
			}
			let report = self.fetch(&credential, now_ms).await?;
			self.store.write_usage_report(&report, now_ms)?;
			reports.push(report);
		}
		Ok(reports)
	}

	/// Invalidates matching durable cache entries.
	pub fn mark_stale(
		&self,
		provider: Option<&str>,
		credential_id: Option<u64>,
	) -> Result<usize, UsageError> {
		Ok(self.store.mark_usage_stale(provider, credential_id)?)
	}

	/// Returns exact observed spend for one provider/account rolling window.
	pub fn rolling_spend(
		&self,
		provider: &str,
		credential_id: Option<u64>,
		account: Option<&str>,
		since_ms: u64,
		until_ms: u64,
	) -> Result<RollingSpend, UsageError> {
		Ok(self
			.store
			.rolling_spend(provider, credential_id, account, since_ms, until_ms)?)
	}

	/// Consumes one `OpenAI` Codex saved rate-limit reset credit.
	///
	/// `redeem_request_id` is the caller's idempotency key and must be reused
	/// when retrying an uncertain response. Secret bearer material remains
	/// sealed and is applied only to the outbound provider request.
	pub async fn consume_codex_reset_credit(
		&self,
		credential_id: u64,
		credit_id: &str,
		redeem_request_id: &str,
		now_ms: u64,
	) -> Result<CodexResetConsume, UsageError> {
		let credential = self
			.store
			.get_credential(credential_id, now_ms)?
			.filter(|meta| meta.provider == "openai-codex")
			.ok_or(UsageError::StaleCredential(credential_id))?;
		let lease = self
			.store
			.lease(credential.id)?
			.ok_or(UsageError::StaleCredential(credential_id))?;
		let props = self
			.store
			.oauth_props(credential_id)?
			.unwrap_or(Value::Null);
		let account_id = props
			.get("accountId")
			.or_else(|| props.get("account_id"))
			.or_else(|| json_path(&props, "openai.account_id"))
			.and_then(Value::as_str);
		let mut body = serde_json::Map::from_iter([
			("credit_id".to_owned(), Value::String(credit_id.to_owned())),
			("redeem_request_id".to_owned(), Value::String(redeem_request_id.to_owned())),
		]);
		if let Some(account_id) = account_id {
			body.insert("account_id".to_owned(), Value::String(account_id.to_owned()));
		}
		let mut request = authenticated_request(
			Method::POST,
			"https://chatgpt.com/backend-api/wham/rate-limit-reset-credits/consume",
			Bytes::from(serde_json::to_vec(&body)?),
			AuthKind::Bearer,
			&lease,
			&self.store,
		)?;
		request
			.headers_mut()
			.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
		if let Some(account_id) = account_id {
			request.headers_mut().insert(
				http::header::HeaderName::from_static("chatgpt-account-id"),
				HeaderValue::from_str(account_id)?,
			);
		}
		let response = self.http.send(request).await?;
		let payload: Value = serde_json::from_slice(&response.body).unwrap_or(Value::Null);
		let code = payload.get("code").and_then(Value::as_str).map_or_else(
			|| {
				if response.status.is_success() {
					"reset".to_owned()
				} else {
					format!("http_{}", response.status.as_u16())
				}
			},
			ToOwned::to_owned,
		);
		Ok(CodexResetConsume {
			applied: code == "reset",
			code:    Str::new(code),
			status:  response.status,
		})
	}

	async fn fetch(
		&self,
		credential: &CredentialMeta,
		now_ms: u64,
	) -> Result<UsageReport, UsageError> {
		let lease = self
			.store
			.lease(credential.id)?
			.ok_or(UsageError::StaleCredential(credential.id))?;
		if let Some(fetcher) = FETCHERS
			.iter()
			.find(|fetcher| fetcher.provider == credential.provider)
		{
			return fetch_table(fetcher, credential, lease, &self.store, self.http.as_ref(), now_ms)
				.await;
		}
		if let Some(provider_override) = self
			.overrides
			.iter()
			.find(|item| item.provider() == credential.provider)
		{
			return provider_override
				.fetch(OverrideContext {
					credential,
					lease,
					store: &self.store,
					http: self.http.as_ref(),
					now_ms,
				})
				.await;
		}
		Err(UsageError::UnsupportedProvider(credential.provider.clone()))
	}
}

async fn fetch_table(
	fetcher: &UsageFetcher,
	credential: &CredentialMeta,
	lease: CredentialLease,
	store: &Store,
	http: &dyn UsageHttp,
	now_ms: u64,
) -> Result<UsageReport, UsageError> {
	let request = authenticated_request(
		Method::GET,
		fetcher.url_template,
		Bytes::new(),
		fetcher.auth_kind,
		&lease,
		store,
	)?;
	let response = http.send(request).await?;
	if !response.status.is_success() {
		return Err(UsageError::Http(response.status));
	}
	let payload: Value = serde_json::from_slice(&response.body)?;
	if matches!(fetcher.provider, "minimax-code" | "minimax-code-cn")
		&& json_path(&payload, "base_resp.status_code").and_then(value_number) != Some(0.0)
	{
		return Err(UsageError::InvalidResponse {
			provider: Str::new(fetcher.provider),
			message:  Str::new("base_resp.status_code was not zero"),
		});
	}
	Ok(report_from_payload(fetcher, credential.id, now_ms, payload))
}

fn report_from_payload(
	fetcher: &UsageFetcher,
	credential_id: u64,
	now_ms: u64,
	payload: Value,
) -> UsageReport {
	if matches!(fetcher.provider, "minimax-code" | "minimax-code-cn") {
		return minimax_report(fetcher.provider, credential_id, now_ms, payload);
	}
	if fetcher.provider == "alibaba-token-plan" {
		return alibaba_token_plan_report(credential_id, now_ms, payload);
	}
	if fetcher.provider == "cursor" {
		return cursor_report(credential_id, now_ms, payload);
	}
	let mut windows = SmallVec::new();
	for spec in fetcher.windows {
		let Some(raw) = json_path(&payload, spec.used_percent_path).and_then(value_number) else {
			continue;
		};
		let scaled = raw * spec.used_percent_scale;
		windows.push(UsageWindow {
			label:        Str::new(spec.label),
			used_percent: if spec.remaining {
				100.0 - scaled
			} else {
				scaled
			},
			resets_at_ms: json_path(&payload, spec.resets_at_path)
				.and_then(timestamp_ms)
				.unwrap_or(0),
		});
	}
	let plan = json_path(&payload, fetcher.plan_path)
		.and_then(Value::as_str)
		.unwrap_or_default();
	UsageReport {
		credential_id,
		provider: Str::new(fetcher.provider),
		plan: Str::new(plan),
		windows,
		fetched_at_ms: now_ms,
		detail: payload,
	}
}

fn minimax_report(provider: &str, credential_id: u64, now_ms: u64, payload: Value) -> UsageReport {
	let mut windows = SmallVec::new();
	if json_path(&payload, "base_resp.status_code").and_then(value_number) == Some(0.0)
		&& let Some(buckets) = payload.get("model_remains").and_then(Value::as_array)
	{
		for bucket in buckets {
			let Some(model) = bucket.get("model_name").and_then(Value::as_str) else {
				continue;
			};
			let field = |name: &str| bucket.get(name).and_then(value_number);
			let unavailable = field("current_interval_total_count") == Some(0.0)
				&& field("current_weekly_total_count") == Some(0.0)
				&& field("current_interval_status") == Some(3.0)
				&& field("current_weekly_status") == Some(3.0);
			if unavailable {
				continue;
			}
			for (suffix, remaining_name, status_name, reset_name) in [
				(
					"interval",
					"current_interval_remaining_percent",
					"current_interval_status",
					"end_time",
				),
				("7d", "current_weekly_remaining_percent", "current_weekly_status", "weekly_end_time"),
			] {
				let used = if field(status_name) == Some(2.0) {
					Some(100.0)
				} else {
					field(remaining_name).map(|remaining| (100.0 - remaining).clamp(0.0, 100.0))
				};
				if let Some(used_percent) = used {
					windows.push(UsageWindow {
						label: fmts!("{model}:{suffix}"),
						used_percent,
						resets_at_ms: bucket.get(reset_name).and_then(timestamp_ms).unwrap_or(0),
					});
				}
			}
		}
	}
	UsageReport {
		credential_id,
		provider: Str::new(provider),
		plan: Str::new("token-plan"),
		windows,
		fetched_at_ms: now_ms,
		detail: payload,
	}
}

fn alibaba_token_plan_report(credential_id: u64, now_ms: u64, payload: Value) -> UsageReport {
	let mut normalized = payload
		.get("data")
		.cloned()
		.unwrap_or_else(|| payload.clone());
	if let Some(encoded) = normalized.get("Data").and_then(Value::as_str)
		&& let Ok(decoded) = serde_json::from_str(encoded)
	{
		normalized = decoded;
	}
	if let Some(data) = json_path(&normalized, "DataV2.data").cloned() {
		normalized = data;
	} else if let Some(data) = normalized.get("data").cloned() {
		normalized = data;
	}
	let mut windows = SmallVec::new();
	for (label, percent_name, reset_name) in [
		("5h", "per5HourPercentage", "per5HourResetTime"),
		("7d", "per1WeekPercentage", "per1WeekResetTime"),
	] {
		if let Some(raw) = normalized.get(percent_name).and_then(value_number) {
			let used_percent = if raw <= 1.0 { raw * 100.0 } else { raw }.clamp(0.0, 100.0);
			windows.push(UsageWindow {
				label: Str::new(label),
				used_percent,
				resets_at_ms: normalized
					.get(reset_name)
					.and_then(timestamp_ms)
					.unwrap_or(0),
			});
		}
	}
	UsageReport {
		credential_id,
		provider: Str::new("alibaba-token-plan"),
		plan: Str::new("token-plan"),
		windows,
		fetched_at_ms: now_ms,
		detail: payload,
	}
}
fn cursor_report(credential_id: u64, now_ms: u64, payload: Value) -> UsageReport {
	let reset_at_ms = cursor_reset_at_ms(&payload).unwrap_or(0);
	let individual = payload.get("individualUsage").and_then(Value::as_object);
	let mut windows = SmallVec::new();

	if let Some(individual) = individual {
		let used_overall = individual
			.get("overall")
			.and_then(Value::as_object)
			.and_then(cursor_cents_percent)
			.map(|used_percent| {
				windows.push(cursor_window("Personal Usage", used_percent, reset_at_ms));
			})
			.is_some();

		if !used_overall
			&& let Some(plan) = individual.get("plan").and_then(Value::as_object)
			&& plan.get("enabled") != Some(&Value::Bool(false))
		{
			let auto = plan.get("autoPercentUsed").and_then(value_number);
			let api = plan.get("apiPercentUsed").and_then(value_number);
			if let Some(used_percent) = auto {
				windows.push(cursor_window("Cursor Models", used_percent.max(0.0), reset_at_ms));
			}
			if let Some(used_percent) = api {
				windows.push(cursor_window("Other Models", used_percent.max(0.0), reset_at_ms));
			}
			if auto.is_none() && api.is_none() {
				let fallback = plan
					.get("totalPercentUsed")
					.and_then(value_number)
					.map(|percent| percent.max(0.0))
					.or_else(|| cursor_cents_percent(plan));
				if let Some(used_percent) = fallback {
					windows.push(cursor_window("Personal Usage", used_percent, reset_at_ms));
				}
			}
		}

		if let Some(used_percent) = individual
			.get("onDemand")
			.and_then(Value::as_object)
			.and_then(cursor_cents_percent)
		{
			windows.push(cursor_window("On-Demand Usage", used_percent, reset_at_ms));
		}
	}

	UsageReport {
		credential_id,
		provider: Str::new("cursor"),
		plan: payload
			.get("membershipType")
			.and_then(Value::as_str)
			.map_or_else(|| Str::new(""), Str::new),
		windows,
		fetched_at_ms: now_ms,
		detail: payload,
	}
}

fn cursor_cents_percent(bucket: &serde_json::Map<String, Value>) -> Option<f64> {
	if bucket.get("enabled") == Some(&Value::Bool(false)) {
		return None;
	}
	let limit = bucket.get("limit").and_then(value_number)?;
	if limit <= 0.0 {
		return None;
	}
	let reported_used = bucket.get("used").and_then(value_number);
	let reported_remaining = bucket.get("remaining").and_then(value_number);
	let used = if reported_used.is_some_and(|used| used > 0.0) {
		reported_used?
	} else if reported_remaining.is_some_and(|remaining| remaining >= 0.0 && remaining < limit) {
		(limit - reported_remaining?).max(0.0)
	} else {
		reported_used.filter(|used| *used >= 0.0)?
	};
	Some(used * 100.0 / limit)
}

fn cursor_window(label: &'static str, used_percent: f64, resets_at_ms: u64) -> UsageWindow {
	UsageWindow { label: Str::new(label), used_percent, resets_at_ms }
}

fn cursor_reset_at_ms(payload: &Value) -> Option<u64> {
	for key in ["billingCycleEnd", "endOfMonth", "resetsAt", "nextReset"] {
		if let Some(reset_at_ms) = payload.get(key).and_then(timestamp_ms) {
			return Some(reset_at_ms);
		}
	}
	for key in ["startOfMonth", "billingCycleStart", "startOfBillingCycle"] {
		let Some(start_at_ms) = payload.get(key).and_then(timestamp_ms) else {
			continue;
		};
		let start = Timestamp::from_millisecond(i64::try_from(start_at_ms).ok()?)
			.ok()?
			.to_zoned(TimeZone::UTC);
		let constrained = start.checked_add(1.month()).ok()?;
		let overflow_days = start.day() - constrained.day();
		let next_month = constrained
			.checked_add(i64::from(overflow_days).days())
			.ok()?;
		return next_month.timestamp().as_millisecond().try_into().ok();
	}
	None
}

fn value_number(value: &Value) -> Option<f64> {
	value
		.as_f64()
		.or_else(|| value.as_str()?.parse::<f64>().ok())
}

fn json_path<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
	path.split('.').try_fold(root, |value, segment| {
		segment
			.parse::<usize>()
			.map_or_else(|_| value.get(segment), |index| value.get(index))
	})
}

fn timestamp_ms(value: &Value) -> Option<u64> {
	if let Some(number) = value.as_u64() {
		return Some(if number < 10_000_000_000 {
			number.saturating_mul(1_000)
		} else {
			number
		});
	}
	let raw = value.as_str()?;
	if let Ok(number) = raw.parse::<u64>() {
		return Some(if number < 10_000_000_000 {
			number.saturating_mul(1_000)
		} else {
			number
		});
	}
	raw.parse::<Timestamp>()
		.ok()?
		.as_millisecond()
		.try_into()
		.ok()
}

fn authenticated_request(
	method: Method,
	url: &str,
	body: Bytes,
	auth_kind: AuthKind,
	lease: &CredentialLease,
	store: &Store,
) -> Result<Request<Bytes>, UsageError> {
	let credential_id = lease.credential_id();
	store
		.redeem_with(lease.provider(), credential_id, lease.generation(), |auth| {
			let mut headers = HeaderMap::new();
			headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
			apply_auth(auth, auth_kind, &mut headers)?;
			Ok(Request::builder()
				.method(method)
				.uri(url)
				.body(body)
				.map(|mut request| {
					*request.headers_mut() = headers;
					request
				})?)
		})?
		.ok_or(UsageError::StaleCredential(credential_id))?
}

fn apply_auth(
	auth: AppliedAuth,
	kind: AuthKind,
	headers: &mut HeaderMap,
) -> Result<(), http::header::InvalidHeaderValue> {
	match kind {
		AuthKind::Bearer => auth.apply_bearer_to_headers(headers)?,
		AuthKind::Authorization => {
			auth.apply_to_authorization(headers)?;
		},
		AuthKind::CursorCookie => {
			auth.apply_to_cursor_cookie(headers)?;
		},
	}
	Ok(())
}

struct MultiStepOverride {
	provider:  &'static str,
	steps:     &'static [OverrideStep],
	windows:   &'static [WindowSpec],
	plan_path: &'static str,
}

struct OverrideStep {
	method:    Method,
	url:       &'static str,
	auth_kind: AuthKind,
	body:      &'static [u8],
}

impl UsageOverride for MultiStepOverride {
	fn provider(&self) -> &'static str {
		self.provider
	}

	fn fetch<'a>(
		&'a self,
		context: OverrideContext<'a>,
	) -> BoxFuture<'a, Result<UsageReport, UsageError>> {
		Box::pin(async move {
			let mut payloads = Vec::with_capacity(self.steps.len());
			for step in self.steps {
				let mut request = authenticated_request(
					step.method.clone(),
					step.url,
					Bytes::from_static(step.body),
					step.auth_kind,
					&context.lease,
					context.store,
				)?;
				if !step.body.is_empty() {
					request
						.headers_mut()
						.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
				}
				let response = context.http.send(request).await?;
				if !response.status.is_success() {
					return Err(UsageError::Http(response.status));
				}
				payloads.push(serde_json::from_slice(&response.body)?);
			}
			let payload = if payloads.len() == 1 {
				payloads.pop().expect("one response was recorded")
			} else {
				serde_json::json!({ "responses": payloads })
			};
			let descriptor = UsageFetcher {
				provider:     self.provider,
				url_template: self.steps.last().map_or("", |step| step.url),
				auth_kind:    self
					.steps
					.last()
					.map_or(AuthKind::Bearer, |step| step.auth_kind),
				plan_path:    self.plan_path,
				windows:      self.windows,
			};
			Ok(report_from_payload(&descriptor, context.credential.id, context.now_ms, payload))
		})
	}
}

struct LocalUsageOverride {
	provider: &'static str,
	plan:     &'static str,
	windows:  &'static [(&'static str, f64)],
	note:     &'static str,
}

impl UsageOverride for LocalUsageOverride {
	fn provider(&self) -> &'static str {
		self.provider
	}

	fn fetch<'a>(
		&'a self,
		context: OverrideContext<'a>,
	) -> BoxFuture<'a, Result<UsageReport, UsageError>> {
		Box::pin(async move {
			if self.provider == "opencode-go" {
				let mut windows = SmallVec::new();
				let mut spend = serde_json::Map::new();
				for &(label, duration_ms, limit_nanos_usd) in OPENCODE_GO_SPEND_WINDOWS {
					let observed = context.store.rolling_spend(
						self.provider,
						Some(context.credential.id),
						Some(context.credential.identity.as_str()),
						context.now_ms.saturating_sub(duration_ms),
						context.now_ms,
					)?;
					windows.push(UsageWindow {
						label:        Str::new(label),
						used_percent: observed.nanos_usd as f64 * 100.0 / limit_nanos_usd as f64,
						resets_at_ms: observed
							.first_observed_at_ms
							.map_or(0, |first| first.saturating_add(duration_ms)),
					});
					spend.insert(label.to_owned(), Value::from(observed.nanos_usd));
				}
				return Ok(UsageReport {
					credential_id: context.credential.id,
					provider: Str::new(self.provider),
					plan: Str::new(self.plan),
					windows,
					fetched_at_ms: context.now_ms,
					detail: Value::Object(spend),
				});
			}
			Ok(UsageReport {
				credential_id: context.credential.id,
				provider:      Str::new(self.provider),
				plan:          Str::new(self.plan),
				windows:       self
					.windows
					.iter()
					.map(|(label, used_percent)| UsageWindow {
						label:        Str::new(label),
						used_percent: *used_percent,
						resets_at_ms: 0,
					})
					.collect(),
				fetched_at_ms: context.now_ms,
				detail:        serde_json::json!({ "note": self.note }),
			})
		})
	}
}

const OPENCODE_GO_LOCAL_WINDOWS: &[(&str, f64)] = &[];
const OPENCODE_GO_SPEND_WINDOWS: &[(&str, u64, u64)] = &[
	("rolling-5h", 5 * 60 * 60_000, 12_000_000_000),
	("weekly", 7 * 24 * 60 * 60_000, 30_000_000_000),
	("monthly", 30 * 24 * 60 * 60_000, 60_000_000_000),
];

const CURSOR_WINDOWS: &[WindowSpec] = &[];
const GEMINI_WINDOWS: &[WindowSpec] = &[WindowSpec {
	label:              "quota",
	used_percent_path:  "responses.1.buckets.0.remainingFraction",
	resets_at_path:     "responses.1.buckets.0.resetTime",
	used_percent_scale: 100.0,
	remaining:          true,
}];
const ANTIGRAVITY_WINDOWS: &[WindowSpec] = &[WindowSpec {
	label:              "quota",
	used_percent_path:  "models.0.quotaInfo.remainingFraction",
	resets_at_path:     "models.0.quotaInfo.resetTime",
	used_percent_scale: 100.0,
	remaining:          true,
}];
const XAI_WINDOWS: &[WindowSpec] = &[WindowSpec {
	label:              "week",
	used_percent_path:  "responses.0.creditUsagePercent",
	resets_at_path:     "responses.0.currentPeriod.end",
	used_percent_scale: 1.0,
	remaining:          false,
}];

fn built_in_overrides() -> SmallVec<Arc<dyn UsageOverride>, 4> {
	SmallVec::from_iter([
		Arc::new(MultiStepOverride {
			provider:  "cursor",
			steps:     &[OverrideStep {
				method:    Method::GET,
				url:       "https://cursor.com/api/usage-summary",
				auth_kind: AuthKind::CursorCookie,
				body:      b"",
			}],
			windows:   CURSOR_WINDOWS,
			plan_path: "membershipType",
		}) as Arc<dyn UsageOverride>,
		Arc::new(MultiStepOverride {
			provider:  "google-gemini-cli",
			steps:     &[
				OverrideStep {
					method:    Method::POST,
					url:       "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist",
					auth_kind: AuthKind::Bearer,
					body:      b"{}",
				},
				OverrideStep {
					method:    Method::POST,
					url:       "https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota",
					auth_kind: AuthKind::Bearer,
					body:      b"{}",
				},
			],
			windows:   GEMINI_WINDOWS,
			plan_path: "tier",
		}),
		Arc::new(MultiStepOverride {
			provider:  "google-antigravity",
			steps:     &[OverrideStep {
				method:    Method::POST,
				url:       "https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels",
				auth_kind: AuthKind::Bearer,
				body:      b"{}",
			}],
			windows:   ANTIGRAVITY_WINDOWS,
			plan_path: "",
		}),
		Arc::new(MultiStepOverride {
			provider:  "xai",
			steps:     &[
				OverrideStep {
					method:    Method::GET,
					url:       "https://grok.com/rest/billing/subscriptions",
					auth_kind: AuthKind::Bearer,
					body:      b"",
				},
				OverrideStep {
					method:    Method::GET,
					url:       "https://grok.com/rest/rate-limits",
					auth_kind: AuthKind::Bearer,
					body:      b"",
				},
			],
			windows:   XAI_WINDOWS,
			plan_path: "billingTier",
		}),
		Arc::new(LocalUsageOverride {
			provider: "opencode-go",
			plan:     "OpenCode Go",
			windows:  OPENCODE_GO_LOCAL_WINDOWS,
			note:     "OMP-observed spend only; no provider quota endpoint is available",
		}),
		Arc::new(LocalUsageOverride {
			provider: "ollama",
			plan:     "",
			windows:  &[],
			note:     "Ollama reports token usage per response and exposes no quota endpoint",
		}),
		Arc::new(LocalUsageOverride {
			provider: "ollama-cloud",
			plan:     "",
			windows:  &[],
			note:     "Ollama Cloud reports token usage per response and exposes no quota endpoint",
		}),
	])
}

#[cfg(test)]
mod tests {
	use std::sync::atomic::{AtomicUsize, Ordering};

	use tempfile::TempDir;

	use super::*;

	struct MockHttp {
		calls: AtomicUsize,
		body:  Bytes,
	}

	impl UsageHttp for MockHttp {
		fn send(
			&self,
			_request: Request<Bytes>,
		) -> BoxFuture<'_, Result<UsageHttpResponse, UsageError>> {
			self.calls.fetch_add(1, Ordering::SeqCst);
			let body = self.body.clone();
			Box::pin(async move { Ok(UsageHttpResponse { status: StatusCode::OK, body }) })
		}
	}

	struct CaptureHttp {
		request: parking_lot::Mutex<Option<Request<Bytes>>>,
		status:  StatusCode,
		body:    Bytes,
	}

	impl UsageHttp for CaptureHttp {
		fn send(
			&self,
			request: Request<Bytes>,
		) -> BoxFuture<'_, Result<UsageHttpResponse, UsageError>> {
			*self.request.lock() = Some(request);
			let status = self.status;
			let body = self.body.clone();
			Box::pin(async move { Ok(UsageHttpResponse { status, body }) })
		}
	}

	fn store() -> (TempDir, Arc<Store>, u64) {
		let directory = tempfile::tempdir().expect("tempdir");
		let store = Arc::new(Store::open(directory.path().join("broker.sqlite")).expect("store"));
		let id = store
			.upsert_oauth("openai-codex", "test", b"token", 0, 1)
			.expect("credential")
			.id;
		(directory, store, id)
	}

	#[test]
	fn table_fetcher_parses_recorded_payload() {
		let payload = serde_json::json!({
			"plan_type": "plus",
			"rate_limit": {
				"primary_window": { "used_percent": 31.5, "reset_at": 1_800_000_000 },
				"secondary_window": { "used_percent": 72.0, "reset_at": 1_900_000_000 }
			}
		});
		let report = report_from_payload(&FETCHERS[1], 7, 42, payload);
		assert_eq!(report.plan, "plus");
		assert_eq!(report.windows.len(), 2);
		assert_eq!(report.windows[0], UsageWindow {
			label:        Str::new("primary"),
			used_percent: 31.5,
			resets_at_ms: 1_800_000_000_000,
		});
	}

	#[test]
	fn cursor_plan_rails_and_on_demand_match_dashboard_percentages() {
		let reset = "2026-09-08T08:00:31.000Z";
		let report = cursor_report(
			7,
			42,
			serde_json::json!({
				"membershipType": "pro_plus",
				"individualUsage": {
					"plan": {
						"enabled": true,
						"used": 1504,
						"limit": 7000,
						"remaining": 5496,
						"autoPercentUsed": 1.85,
						"apiPercentUsed": 0,
						"totalPercentUsed": 1.63
					},
					"onDemand": {
						"enabled": true,
						"used": 0,
						"limit": 2000,
						"remaining": 2000
					}
				},
				"billingCycleEnd": reset
			}),
		);
		let reset_at_ms = timestamp_ms(&Value::from(reset)).expect("reset timestamp");
		assert_eq!(report.plan, "pro_plus");
		assert_eq!(report.windows, [
			UsageWindow {
				label:        Str::new("Cursor Models"),
				used_percent: 1.85,
				resets_at_ms: reset_at_ms,
			},
			UsageWindow {
				label:        Str::new("Other Models"),
				used_percent: 0.0,
				resets_at_ms: reset_at_ms,
			},
			UsageWindow {
				label:        Str::new("On-Demand Usage"),
				used_percent: 0.0,
				resets_at_ms: reset_at_ms,
			},
		]);
		assert_ne!(report.windows[0].used_percent, 1504.0 * 100.0 / 7000.0);
	}

	#[test]
	fn cursor_prefers_usable_overall_and_falls_through_disabled_overall() {
		let overall = cursor_report(
			7,
			42,
			serde_json::json!({
				"individualUsage": {
					"overall": { "enabled": true, "used": 100, "limit": 1000, "remaining": 900 },
					"plan": { "enabled": true, "autoPercentUsed": 1.85, "apiPercentUsed": 2.5 }
				}
			}),
		);
		assert_eq!(overall.windows, [UsageWindow {
			label:        Str::new("Personal Usage"),
			used_percent: 10.0,
			resets_at_ms: 0,
		}]);

		let plan = cursor_report(
			7,
			42,
			serde_json::json!({
				"individualUsage": {
					"overall": { "enabled": false, "used": 100, "limit": 1000 },
					"plan": { "enabled": true, "autoPercentUsed": 1.85, "apiPercentUsed": 0 }
				}
			}),
		);
		assert_eq!(plan.windows.len(), 2);
		assert_eq!(plan.windows[0].label, "Cursor Models");
		assert_eq!(plan.windows[0].used_percent, 1.85);
		assert_eq!(plan.windows[1].label, "Other Models");
		assert_eq!(plan.windows[1].used_percent, 0.0);
	}

	#[test]
	fn cursor_keeps_on_demand_when_plan_is_disabled() {
		let report = cursor_report(
			7,
			42,
			serde_json::json!({
				"individualUsage": {
					"plan": {
						"enabled": false,
						"limit": 7000,
						"autoPercentUsed": 1.85,
						"apiPercentUsed": 0
					},
					"onDemand": { "enabled": true, "used": 500, "limit": 2000, "remaining": 1500 }
				}
			}),
		);
		assert_eq!(report.windows, [UsageWindow {
			label:        Str::new("On-Demand Usage"),
			used_percent: 25.0,
			resets_at_ms: 0,
		}]);
	}

	#[test]
	fn cursor_plan_falls_back_to_total_percent_then_cents() {
		let total = cursor_report(
			7,
			42,
			serde_json::json!({
				"individualUsage": {
					"plan": { "enabled": true, "used": 1504, "limit": 7000, "totalPercentUsed": 1.63 }
				}
			}),
		);
		assert_eq!(total.windows[0].label, "Personal Usage");
		assert_eq!(total.windows[0].used_percent, 1.63);

		let cents = cursor_report(
			7,
			42,
			serde_json::json!({
				"individualUsage": {
					"plan": { "enabled": true, "used": 0, "limit": 7000, "remaining": 6076 }
				}
			}),
		);
		assert_eq!(cents.windows[0].label, "Personal Usage");
		assert_eq!(cents.windows[0].used_percent, 13.2);
	}

	#[test]
	fn cursor_derives_monthly_reset_from_billing_cycle_start() {
		let report = cursor_report(
			7,
			42,
			serde_json::json!({
				"individualUsage": {
					"plan": { "enabled": true, "autoPercentUsed": 1 }
				},
				"billingCycleStart": "2026-01-31T08:00:31Z"
			}),
		);
		assert_eq!(
			report.windows[0].resets_at_ms,
			timestamp_ms(&Value::from("2026-03-03T08:00:31Z")).expect("reset timestamp")
		);
	}

	#[test]
	fn minimax_token_plan_fixture_normalizes_remaining_status_and_resets() {
		let fetcher = FETCHERS
			.iter()
			.find(|fetcher| fetcher.provider == "minimax-code")
			.expect("MiniMax fetcher");
		let payload = serde_json::json!({
			"base_resp": { "status_code": 0 },
			"model_remains": [
				{
					"model_name": "general",
					"current_interval_remaining_percent": "90",
					"current_interval_status": 1,
					"current_interval_total_count": 1000,
					"end_time": 1_800_000_000,
					"current_weekly_remaining_percent": 35,
					"current_weekly_status": 1,
					"current_weekly_total_count": 5000,
					"weekly_end_time": "1900000000"
				},
				{
					"model_name": "video",
					"current_interval_remaining_percent": 100,
					"current_interval_status": 3,
					"current_interval_total_count": 0,
					"current_weekly_remaining_percent": 100,
					"current_weekly_status": 3,
					"current_weekly_total_count": 0
				}
			]
		});
		let report = report_from_payload(fetcher, 8, 42, payload);
		assert_eq!(report.plan, "token-plan");
		assert_eq!(report.windows.len(), 2);
		assert_eq!(report.windows[0], UsageWindow {
			label:        Str::new("general:interval"),
			used_percent: 10.0,
			resets_at_ms: 1_800_000_000_000,
		});
		assert_eq!(report.windows[1].used_percent, 65.0);
		assert_eq!(report.windows[1].resets_at_ms, 1_900_000_000_000);
	}

	#[test]
	fn alibaba_gateway_fixture_unwraps_encoded_data_and_fractional_percentages() {
		let descriptor = UsageFetcher {
			provider:     "alibaba-token-plan",
			url_template: "",
			auth_kind:    AuthKind::Bearer,
			plan_path:    "",
			windows:      &[],
		};
		let payload = serde_json::json!({
			"data": {
				"Data": "{\"DataV2\":{\"data\":{\"per5HourPercentage\":0.25,\
					\"per5HourResetTime\":1800000000,\"per1WeekPercentage\":\"80\",\
					\"per1WeekResetTime\":\"1900000000\"}}}"
			}
		});
		let report = report_from_payload(&descriptor, 9, 42, payload);
		assert_eq!(report.windows, [
			UsageWindow {
				label:        Str::new("5h"),
				used_percent: 25.0,
				resets_at_ms: 1_800_000_000_000,
			},
			UsageWindow {
				label:        Str::new("7d"),
				used_percent: 80.0,
				resets_at_ms: 1_900_000_000_000,
			},
		]);
	}

	#[tokio::test]
	async fn local_usage_sources_cover_opencode_go_and_ollama_without_fake_http() {
		for (provider, window_count) in [("opencode-go", 3_usize), ("ollama", 0), ("ollama-cloud", 0)]
		{
			let directory = tempfile::tempdir().expect("tempdir");
			let store = Arc::new(Store::open(directory.path().join("broker.sqlite")).expect("store"));
			let id = store
				.upsert_api_key(provider, "test", b"token", 1)
				.expect("credential")
				.id;
			if provider == "opencode-go" {
				let lease = store.lease(id).expect("lease query").expect("lease");
				store
					.record_terminal_usage(
						&lease,
						"turn-observed",
						"model",
						0,
						&ClientUsage {
							client_id:          "client".into(),
							label:              "client".into(),
							input_tokens:       10,
							output_tokens:      5,
							cache_read_tokens:  0,
							cache_write_tokens: 0,
							nanos_usd:          6_000_000_000,
							last_seen_ms:       90,
						},
						90,
					)
					.expect("record usage");
			}
			let http = Arc::new(MockHttp { calls: AtomicUsize::new(0), body: Bytes::new() });
			let reports = UsageManager::new(Arc::clone(&store), http.clone())
				.get_usage(Some(provider), Some(id), true, 100)
				.await
				.expect("local usage");
			assert_eq!(reports.len(), 1);
			assert_eq!(reports[0].provider, provider);
			assert_eq!(reports[0].windows.len(), window_count);
			if provider == "opencode-go" {
				assert_eq!(reports[0].windows[0].used_percent, 50.0);
				assert_eq!(reports[0].windows[0].resets_at_ms, 90 + 5 * 60 * 60_000);
				assert_eq!(reports[0].detail["rolling-5h"], 6_000_000_000_u64);
			}
			assert_eq!(http.calls.load(Ordering::SeqCst), 0);
		}
	}

	#[tokio::test]
	async fn codex_reset_credit_consumption_preserves_wire_outcome_and_idempotency_key() {
		let (_directory, store, id) = store();
		let updated = store
			.upsert_oauth_material(
				"openai-codex",
				"test",
				b"token",
				None,
				&serde_json::json!({ "accountId": "acct-7" }),
				0,
				2,
			)
			.expect("account metadata");
		assert_eq!(updated.id, id);
		let http = Arc::new(CaptureHttp {
			request: parking_lot::Mutex::new(None),
			status:  StatusCode::CONFLICT,
			body:    Bytes::from_static(br#"{"code":"already_redeemed"}"#),
		});
		let result = UsageManager::new(store, http.clone())
			.consume_codex_reset_credit(id, "credit-7", "request-9", 100)
			.await
			.expect("consume result");
		assert_eq!(result, CodexResetConsume {
			applied: false,
			code:    Str::new("already_redeemed"),
			status:  StatusCode::CONFLICT,
		});
		let request = http.request.lock().take().expect("captured request");
		assert_eq!(request.method(), Method::POST);
		assert_eq!(
			request.uri(),
			"https://chatgpt.com/backend-api/wham/rate-limit-reset-credits/consume"
		);
		assert_eq!(request.headers()[http::header::AUTHORIZATION], "Bearer token");
		assert_eq!(request.headers()["chatgpt-account-id"], "acct-7");
		assert_eq!(
			serde_json::from_slice::<Value>(request.body()).expect("request JSON"),
			serde_json::json!({
				"credit_id": "credit-7",
				"redeem_request_id": "request-9",
				"account_id": "acct-7",
			})
		);
	}

	#[test]
	fn override_hook_is_selected_for_outlier() {
		let overrides = built_in_overrides();
		assert!(FETCHERS.iter().all(|fetcher| fetcher.provider != "cursor"));
		assert_eq!(
			overrides
				.iter()
				.find(|item| item.provider() == "cursor")
				.map(|item| item.provider()),
			Some("cursor")
		);
	}

	#[tokio::test]
	async fn concurrent_fetches_coalesce() {
		let (_directory, store, id) = store();
		let http = Arc::new(MockHttp { calls: AtomicUsize::new(0), body: Bytes::from_static(br#"{"plan_type":"plus","rate_limit":{"primary_window":{"used_percent":1,"reset_at":1800000000}}}"#) });
		let manager = Arc::new(UsageManager::new(store, http.clone()));
		let (left, right) = tokio::join!(
			manager.get_usage(None, Some(id), false, 100),
			manager.get_usage(None, Some(id), false, 100),
		);
		assert!(left.is_ok() && right.is_ok());
		assert_eq!(http.calls.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn expired_cache_refetches() {
		let (_directory, store, id) = store();
		let http = Arc::new(MockHttp { calls: AtomicUsize::new(0), body: Bytes::from_static(br#"{"plan_type":"plus","rate_limit":{"primary_window":{"used_percent":1,"reset_at":1800000000}}}"#) });
		let manager = UsageManager::new(store, http.clone()).with_cache_ttl_ms(10);
		manager
			.get_usage(None, Some(id), false, 100)
			.await
			.expect("initial");
		manager
			.get_usage(None, Some(id), false, 111)
			.await
			.expect("expired");
		assert_eq!(http.calls.load(Ordering::SeqCst), 2);
	}
}
