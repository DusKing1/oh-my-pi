//! Account usage/quota service composition and typed window normalization.

use std::{
	future::Future,
	sync::Arc,
	task::{Context, Poll},
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use omp_core::Str;
use tower::Service;

use crate::{
	account::{QuotaProvenance, QuotaState, QuotaWindowId, RateState},
	answer::{Answer, AnswerBody, UsageReport, UsageWindow, UsageWindowKind},
	call::{OperationCall, UsageRequest, UsageScope},
	catalog::{OperationKind, ProviderId},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	id::{AccountId, PrincipalId},
	operation::{OperationRequest, OperationResponse},
	receipt::{ExecutionReceipt, ReasonId, UsageSource},
};

/// Configures stale-observation enforcement for a usage service.
#[derive(Clone, Copy, Debug)]
pub struct UsageServiceConfig {
	/// Largest acceptable age when the caller forbids stale observations.
	pub maximum_age: Duration,
	/// Injectable clock used for deterministic replay.
	pub clock:       fn() -> SystemTime,
}

impl UsageServiceConfig {
	/// Constructs a usage policy using the system wall clock.
	pub const fn new(maximum_age: Duration) -> Self {
		Self { maximum_age, clock: SystemTime::now }
	}
}

/// Concrete usage service over a constructed account/auth/codec backend.
#[derive(Clone, Debug)]
pub struct UsageService<S> {
	inner:  S,
	config: UsageServiceConfig,
}

impl<S> UsageService<S> {
	/// Wraps a route backend that returns typed, secret-free usage windows.
	pub const fn new(inner: S, config: UsageServiceConfig) -> Self {
		Self { inner, config }
	}
}

impl<S> Service<crate::call::Call> for UsageService<S>
where
	S: Service<
			OperationRequest<UsageRequest>,
			Response = OperationResponse<UsageReport>,
			Error = Error,
		>,
	S::Future: Send + 'static,
{
	type Error = Error;
	type Response = Answer;

	type Future = impl Future<Output = Result<Answer, Error>> + Send;

	fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(context)
	}

	fn call(&mut self, call: crate::call::Call) -> Self::Future {
		let request = match &call.operation {
			OperationCall::Usage(request) => {
				Some(OperationRequest::from_call(&call, Arc::clone(request)))
			},
			_ => None,
		};
		let pending = request
			.as_ref()
			.map(|request| self.inner.call(request.clone()));
		let config = self.config;
		async move {
			let Some(request) = request else {
				return Err(wrong_operation(&call));
			};
			let Some(pending) = pending else {
				return Err(protocol_error("usage_backend_not_called"));
			};
			let mut response = pending.await?;
			normalize_report(&mut response.output, &request.payload, config)?;
			Ok(response.into_answer(AnswerBody::Usage))
		}
	}
}

/// Validates selectors, freshness, arithmetic, and requested window scope in
/// place.
pub fn normalize_report(
	report: &mut UsageReport,
	request: &UsageRequest,
	config: UsageServiceConfig,
) -> Result<(), Error> {
	if request
		.provider
		.as_ref()
		.is_some_and(|provider| provider != &report.provider)
	{
		return Err(protocol_error("usage_provider_selector_mismatch"));
	}
	if request
		.account
		.as_ref()
		.is_some_and(|account| account != &report.account)
	{
		return Err(protocol_error("usage_account_selector_mismatch"));
	}
	let now = (config.clock)();
	for window in &report.windows {
		if window.dimension.is_empty() {
			return Err(protocol_error("usage_window_dimension_missing"));
		}
		if window.limit.is_some_and(|limit| {
			window.remaining.is_some_and(|remaining| remaining > limit)
				|| window.consumed.is_some_and(|consumed| consumed > limit)
		}) {
			return Err(protocol_error("usage_window_exceeds_limit"));
		}
		if !request.allow_stale
			&& now
				.duration_since(window.observed_at)
				.is_ok_and(|age| age > config.maximum_age)
		{
			return Err(stale_error(&window.dimension));
		}
	}
	report
		.windows
		.retain(|window| scope_includes(request.scope, window.kind));
	report.windows.sort_by(|left, right| {
		window_kind_rank(left.kind)
			.cmp(&window_kind_rank(right.kind))
			.then_with(|| left.dimension.cmp(&right.dimension))
			.then_with(|| left.observed_at.cmp(&right.observed_at))
	});
	Ok(())
}

/// Creates a usage report from shared account quota and rate state without
/// reading secrets.
pub fn report_from_account_state(
	provider: ProviderId,
	account: AccountId,
	principal: Option<PrincipalId>,
	quota: &QuotaState,
	rate: &RateState,
	quota_kinds: &[(QuotaWindowId, UsageWindowKind)],
) -> UsageReport {
	let mut windows = Vec::with_capacity(quota.windows().len() + rate.windows().len());
	for (id, window) in quota.windows() {
		let kind = quota_kinds
			.iter()
			.find_map(|(mapped, kind)| (mapped == id).then_some(*kind))
			.unwrap_or(UsageWindowKind::Quota);
		let observed_at = [
			window.consumed.map(|sample| sample.observed_at),
			window.remaining.map(|sample| sample.observed_at),
			window.limit.map(|sample| sample.observed_at),
			window.reset_at.map(|sample| sample.observed_at),
			window.exhausted.map(|sample| sample.observed_at),
		]
		.into_iter()
		.flatten()
		.max()
		.unwrap_or(UNIX_EPOCH);
		let source = window
			.receipts
			.last()
			.map_or(UsageSource::Unknown, |receipt| quota_source(receipt.provenance));
		windows.push(UsageWindow {
			kind,
			dimension: id.0.clone(),
			consumed: window.consumed.map(|sample| sample.value),
			remaining: window.remaining.map(|sample| sample.value),
			limit: window.limit.map(|sample| sample.value),
			resets_at: window.reset_at.map(|sample| sample.value),
			source,
			observed_at,
		});
	}
	for (id, window) in rate.windows() {
		let limit = window.limit.map(|sample| sample.value);
		let remaining = window.remaining.map(|sample| sample.value);
		let consumed = limit
			.zip(remaining)
			.map(|(limit, remaining)| limit.saturating_sub(remaining));
		let observed_at = [
			window.limit.map(|sample| sample.observed_at),
			window.remaining.map(|sample| sample.observed_at),
			window.reset_at.map(|sample| sample.observed_at),
			window.retry_at.map(|sample| sample.observed_at),
		]
		.into_iter()
		.flatten()
		.max()
		.unwrap_or(UNIX_EPOCH);
		windows.push(UsageWindow {
			kind: UsageWindowKind::RateLimit,
			dimension: id.0.clone(),
			consumed,
			remaining,
			limit,
			resets_at: window.reset_at.map(|sample| sample.value),
			source: UsageSource::Provider,
			observed_at,
		});
	}
	UsageReport { provider, account, principal, windows }
}

fn quota_source(source: QuotaProvenance) -> UsageSource {
	match source {
		QuotaProvenance::Provider | QuotaProvenance::Header | QuotaProvenance::Error => {
			UsageSource::Provider
		},
		QuotaProvenance::Measured => UsageSource::Measured,
	}
}

fn scope_includes(scope: UsageScope, kind: UsageWindowKind) -> bool {
	match scope {
		UsageScope::All => true,
		UsageScope::Current => matches!(kind, UsageWindowKind::RateLimit | UsageWindowKind::Quota),
		UsageScope::Billing => matches!(kind, UsageWindowKind::Billing | UsageWindowKind::Balance),
		UsageScope::RateLimit => kind == UsageWindowKind::RateLimit,
	}
}

const fn window_kind_rank(kind: UsageWindowKind) -> u8 {
	match kind {
		UsageWindowKind::RateLimit => 0,
		UsageWindowKind::Quota => 1,
		UsageWindowKind::Billing => 2,
		UsageWindowKind::Balance => 3,
	}
}

fn wrong_operation(call: &crate::call::Call) -> Error {
	let mut error = Error::new(
		ErrorKind::InternalInvariant,
		ErrorPhase::Internal,
		RetryAction::Never,
		ExecutionReceipt::default(),
	);
	error.request_id = Some(call.id.clone());
	error.detail = Some(ErrorDetail::Capability {
		feature: Str::from(OperationKind::Usage.to_string()),
		reason:  ReasonId(Str::from("operation_service_mismatch")),
	});
	error
}

fn stale_error(dimension: &str) -> Error {
	let mut error = Error::new(
		ErrorKind::ProviderContractMismatch,
		ErrorPhase::Discovery,
		RetryAction::Never,
		ExecutionReceipt::default(),
	);
	error.detail = Some(ErrorDetail::Protocol {
		reason: ReasonId(Str::from(format!("stale_usage_window:{dimension}"))),
	});
	error
}

fn protocol_error(reason: &'static str) -> Error {
	let mut error = Error::new(
		ErrorKind::ProviderContractMismatch,
		ErrorPhase::Discovery,
		RetryAction::Never,
		ExecutionReceipt::default(),
	);
	error.detail = Some(ErrorDetail::Protocol { reason: ReasonId(Str::from(reason)) });
	error
}

#[cfg(test)]
mod tests {
	use std::time::{Duration, UNIX_EPOCH};

	use super::{UsageServiceConfig, normalize_report, report_from_account_state};
	use crate::{
		account::{QuotaObservation, QuotaProvenance, QuotaState, QuotaWindowId, RateState},
		answer::UsageWindowKind,
		call::{UsageRequest, UsageScope},
		catalog::ProviderId,
		id::AccountId,
	};

	fn now() -> std::time::SystemTime {
		UNIX_EPOCH + Duration::from_secs(120)
	}
	fn late() -> std::time::SystemTime {
		UNIX_EPOCH + Duration::from_secs(300)
	}

	#[test]
	fn shared_quota_state_projects_to_typed_current_window() {
		let id = QuotaWindowId::new("tokens");
		let mut quota = QuotaState::default();
		quota.apply(QuotaObservation {
			window:      id.clone(),
			consumed:    Some(40),
			remaining:   Some(60),
			limit:       Some(100),
			reset_at:    Some(now() + Duration::from_secs(60)),
			exhausted:   Some(false),
			provenance:  QuotaProvenance::Provider,
			observed_at: now(),
		});
		let provider = ProviderId::from("provider");
		let account = AccountId::from("account");
		let mut report = report_from_account_state(
			provider.clone(),
			account.clone(),
			None,
			&quota,
			&RateState::default(),
			&[(id, UsageWindowKind::Quota)],
		);
		normalize_report(
			&mut report,
			&UsageRequest {
				provider:    Some(provider),
				account:     Some(account),
				scope:       UsageScope::Current,
				allow_stale: false,
			},
			UsageServiceConfig { maximum_age: Duration::from_secs(30), clock: now },
		)
		.expect("fresh report");
		assert_eq!(report.windows.len(), 1);
		assert_eq!(report.windows[0].consumed, Some(40));
		assert_eq!(report.windows[0].remaining, Some(60));
	}

	#[test]
	fn stale_and_inconsistent_usage_windows_are_rejected() {
		let id = QuotaWindowId::new("tokens");
		let mut quota = QuotaState::default();
		quota.apply(QuotaObservation {
			window:      id.clone(),
			consumed:    Some(40),
			remaining:   Some(60),
			limit:       Some(100),
			reset_at:    None,
			exhausted:   Some(false),
			provenance:  QuotaProvenance::Provider,
			observed_at: now(),
		});
		let provider = ProviderId::from("provider");
		let account = AccountId::from("account");
		let request = UsageRequest {
			provider:    Some(provider.clone()),
			account:     Some(account.clone()),
			scope:       UsageScope::Current,
			allow_stale: false,
		};
		let mut stale = report_from_account_state(
			provider.clone(),
			account.clone(),
			None,
			&quota,
			&RateState::default(),
			&[(id.clone(), UsageWindowKind::Quota)],
		);
		assert!(
			normalize_report(&mut stale, &request, UsageServiceConfig {
				maximum_age: Duration::from_secs(30),
				clock:       late,
			},)
			.is_err()
		);

		let mut inconsistent =
			report_from_account_state(provider, account, None, &quota, &RateState::default(), &[(
				id,
				UsageWindowKind::Quota,
			)]);
		inconsistent.windows[0].remaining = Some(101);
		assert!(
			normalize_report(&mut inconsistent, &request, UsageServiceConfig {
				maximum_age: Duration::from_secs(30),
				clock:       now,
			},)
			.is_err()
		);
	}
}
