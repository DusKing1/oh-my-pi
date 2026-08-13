//! Catalog-aware credential acquisition across typed source engines.

use std::{collections::BTreeMap, fmt, sync::Arc};

use futures::future::{BoxFuture, FutureExt as _};
use omp_core::Str;
use omp_llm_catalog::{AuthSpecId, Catalog, provider::AuthSpecKind};
use secrecy::SecretString;

use super::lease::{
	AuthRejection, CredentialError, CredentialKind, CredentialLease, CredentialNeed,
	CredentialSource, LeaseMeta,
};

const ENVIRONMENT_TAG: &str = "environment";
const STORED_TAG: &str = "stored";
const ADC_TAG: &str = "application-default";
const AWS_TAG: &str = "aws-chain";
const OAUTH_TAG: &str = "oauth";
const SESSION_TAG: &str = "session";

/// Secret environment boundary used by [`CredentialBroker`].
pub trait CredentialEnvironment: Send + Sync {
	/// Reads one exact catalog-declared name into a zeroizing secret wrapper.
	fn read(&self, name: &str) -> Result<Option<SecretString>, CredentialError>;
}

/// Process environment implementation that performs no alias or fallback
/// lookup.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemCredentialEnvironment;

impl CredentialEnvironment for SystemCredentialEnvironment {
	fn read(&self, name: &str) -> Result<Option<SecretString>, CredentialError> {
		if !name.starts_with("OMP_") {
			return Err(CredentialError::InvalidSource);
		}
		match std::env::var(name) {
			Ok(value) if value.is_empty() => Err(CredentialError::InvalidSource),
			Ok(value) => Ok(Some(SecretString::from(value))),
			Err(std::env::VarError::NotPresent) => Ok(None),
			Err(std::env::VarError::NotUnicode(_)) => Err(CredentialError::SourceFailure),
		}
	}
}

/// Optional typed engines used by the catalog credential broker.
#[derive(Clone, Default)]
pub struct CredentialBrokerEngines {
	/// Encrypted account-store engine.
	pub stored:              Option<Arc<dyn CredentialSource>>,
	/// Application-default credential engine.
	pub application_default: Option<Arc<dyn CredentialSource>>,
	/// AWS credential-chain engine.
	pub aws:                 Option<Arc<dyn CredentialSource>>,
	/// OAuth login/refresh engine.
	pub oauth:               Option<Arc<dyn CredentialSource>>,
	/// Interactive provider-session engine.
	pub session:             Option<Arc<dyn CredentialSource>>,
}

impl fmt::Debug for CredentialBrokerEngines {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("CredentialBrokerEngines")
			.field("stored", &self.stored.is_some())
			.field("application_default", &self.application_default.is_some())
			.field("aws", &self.aws.is_some())
			.field("oauth", &self.oauth.is_some())
			.field("session", &self.session.is_some())
			.finish()
	}
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EngineKind {
	Stored,
	ApplicationDefault,
	Aws,
	OAuth,
	Session,
}

impl EngineKind {
	const fn tag(self) -> &'static str {
		match self {
			Self::Stored => STORED_TAG,
			Self::ApplicationDefault => ADC_TAG,
			Self::Aws => AWS_TAG,
			Self::OAuth => OAUTH_TAG,
			Self::Session => SESSION_TAG,
		}
	}
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum BrokerSource {
	Environment(Box<[Str]>),
	Engine(EngineKind),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BrokerPlan {
	kind:    CredentialKind,
	sources: Box<[BrokerSource]>,
}

/// Catalog compilation failure for credential acquisition plans.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CredentialBrokerError {
	/// An authenticated catalog record has no declared acquisition source.
	#[error("catalog authentication specification has no credential source")]
	MissingSource(AuthSpecId),
	/// A credential environment source contains an empty or non-OMP name.
	#[error("catalog credential environment source is invalid")]
	InvalidEnvironment(AuthSpecId),
}

/// Catalog-aware composite credential source.
///
/// Plans retain exact catalog source order. Only `Unavailable` advances to the
/// next source; cancellation, invalid source, expiry, staleness, and engine
/// failure remain typed terminal evidence.
#[derive(Clone)]
pub struct CredentialBroker {
	plans:       Arc<BTreeMap<AuthSpecId, BrokerPlan>>,
	environment: Arc<dyn CredentialEnvironment>,
	engines:     CredentialBrokerEngines,
}

impl CredentialBroker {
	/// Compiles immutable acquisition plans from the canonical catalog.
	pub fn from_catalog(
		catalog: &Catalog,
		environment: Arc<dyn CredentialEnvironment>,
		engines: CredentialBrokerEngines,
	) -> Result<Self, CredentialBrokerError> {
		let mut plans = BTreeMap::new();
		for auth in catalog.auth_specs() {
			let Some(kind) = credential_kind(auth.kind) else {
				continue;
			};
			let mut sources = Vec::with_capacity(auth.credential_sources.len());
			for source in &auth.credential_sources {
				use omp_llm_catalog::provider::CredentialSourceSpec as CatalogSource;
				let source = match source {
					CatalogSource::Environment { ordered_names } => {
						if ordered_names.is_empty()
							|| ordered_names.iter().any(|name| !name.starts_with("OMP_"))
						{
							return Err(CredentialBrokerError::InvalidEnvironment(auth.id.clone()));
						}
						BrokerSource::Environment(ordered_names.clone())
					},
					CatalogSource::Stored => BrokerSource::Engine(EngineKind::Stored),
					CatalogSource::ApplicationDefault { .. } => {
						BrokerSource::Engine(EngineKind::ApplicationDefault)
					},
					CatalogSource::AwsChain => BrokerSource::Engine(EngineKind::Aws),
					CatalogSource::Oauth { .. } => BrokerSource::Engine(EngineKind::OAuth),
					CatalogSource::Session => BrokerSource::Engine(EngineKind::Session),
				};
				sources.push(source);
			}
			if sources.is_empty() {
				return Err(CredentialBrokerError::MissingSource(auth.id.clone()));
			}
			plans.insert(auth.id.clone(), BrokerPlan { kind, sources: sources.into_boxed_slice() });
		}
		Ok(Self { plans: Arc::new(plans), environment, engines })
	}

	/// Uses the process environment without upstream aliases or fallbacks.
	pub fn system(
		catalog: &Catalog,
		engines: CredentialBrokerEngines,
	) -> Result<Self, CredentialBrokerError> {
		Self::from_catalog(catalog, Arc::new(SystemCredentialEnvironment), engines)
	}

	fn engine(&self, kind: EngineKind) -> Option<&Arc<dyn CredentialSource>> {
		match kind {
			EngineKind::Stored => self.engines.stored.as_ref(),
			EngineKind::ApplicationDefault => self.engines.application_default.as_ref(),
			EngineKind::Aws => self.engines.aws.as_ref(),
			EngineKind::OAuth => self.engines.oauth.as_ref(),
			EngineKind::Session => self.engines.session.as_ref(),
		}
	}

	fn environment_lease(
		&self,
		names: &[Str],
		need: &CredentialNeed,
		kind: CredentialKind,
	) -> Result<CredentialLease, CredentialError> {
		for name in names {
			let Some(secret) = self.environment.read(name)? else {
				continue;
			};
			let account = need.account.clone().ok_or(CredentialError::InvalidSource)?;
			let principal = need
				.principal
				.clone()
				.ok_or(CredentialError::InvalidSource)?;
			let meta = LeaseMeta { account, principal, generation: 0, expires_at: None };
			let lease = match kind {
				CredentialKind::ApiKey => CredentialLease::api_key(meta, secret),
				CredentialKind::Bearer => CredentialLease::bearer(meta, secret),
				CredentialKind::SessionToken => CredentialLease::session_token(meta, secret),
				CredentialKind::AwsSigV4 => return Err(CredentialError::InvalidSource),
			};
			return Ok(lease.with_source_tag(ENVIRONMENT_TAG.into()));
		}
		Err(CredentialError::Unavailable)
	}

	fn validate_lease(
		lease: CredentialLease,
		need: &CredentialNeed,
		expected: CredentialKind,
		tag: &'static str,
	) -> Result<CredentialLease, CredentialError> {
		if lease.kind() != expected {
			return Err(CredentialError::InvalidSource);
		}
		if need
			.account
			.as_ref()
			.is_some_and(|account| account != &lease.meta().account)
			|| need
				.principal
				.as_ref()
				.is_some_and(|principal| principal != &lease.meta().principal)
		{
			return Err(CredentialError::InvalidSource);
		}
		if lease.is_expired_at(need.valid_after) {
			return Err(CredentialError::Expired);
		}
		Ok(lease.with_source_tag(tag.into()))
	}
}

impl fmt::Debug for CredentialBroker {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("CredentialBroker")
			.field("plans", &self.plans.len())
			.field("engines", &self.engines)
			.finish()
	}
}

impl CredentialSource for CredentialBroker {
	fn lease(
		&self,
		need: CredentialNeed,
	) -> BoxFuture<'_, Result<CredentialLease, CredentialError>> {
		async move {
			let plan = self
				.plans
				.get(&need.spec)
				.ok_or(CredentialError::InvalidSource)?;
			for source in &plan.sources {
				let result = match source {
					BrokerSource::Environment(names) => self.environment_lease(names, &need, plan.kind),
					BrokerSource::Engine(kind) => match self.engine(*kind) {
						Some(engine) => engine
							.lease(need.clone())
							.await
							.and_then(|lease| Self::validate_lease(lease, &need, plan.kind, kind.tag())),
						None => Err(CredentialError::Unavailable),
					},
				};
				match result {
					Err(CredentialError::Unavailable) => continue,
					result => return result,
				}
			}
			Err(CredentialError::Unavailable)
		}
		.boxed()
	}

	fn reject<'a>(
		&'a self,
		lease: &'a CredentialLease,
		evidence: AuthRejection,
	) -> BoxFuture<'a, Result<(), CredentialError>> {
		async move {
			let Some(tag) = lease.source_tag() else {
				return Err(CredentialError::InvalidSource);
			};
			if tag == ENVIRONMENT_TAG {
				return Ok(());
			}
			let kind = match tag {
				STORED_TAG => EngineKind::Stored,
				ADC_TAG => EngineKind::ApplicationDefault,
				AWS_TAG => EngineKind::Aws,
				OAUTH_TAG => EngineKind::OAuth,
				SESSION_TAG => EngineKind::Session,
				_ => return Err(CredentialError::InvalidSource),
			};
			self
				.engine(kind)
				.ok_or(CredentialError::Unavailable)?
				.reject(lease, evidence)
				.await
		}
		.boxed()
	}
}

fn credential_kind(kind: AuthSpecKind) -> Option<CredentialKind> {
	match kind {
		AuthSpecKind::None => None,
		AuthSpecKind::ApiKey => Some(CredentialKind::ApiKey),
		AuthSpecKind::Bearer
		| AuthSpecKind::Oauth
		| AuthSpecKind::GcpAdc
		| AuthSpecKind::AzureAd
		| AuthSpecKind::GithubApp => Some(CredentialKind::Bearer),
		AuthSpecKind::AwsSigv4 => Some(CredentialKind::AwsSigV4),
		AuthSpecKind::OmpSession => Some(CredentialKind::SessionToken),
	}
}

#[cfg(test)]
mod tests {
	use std::time::SystemTime;

	use parking_lot::Mutex;

	use super::*;
	use crate::id::{AccountId, PrincipalId};

	#[derive(Debug, Default)]
	struct EmptyEnvironment;

	impl CredentialEnvironment for EmptyEnvironment {
		fn read(&self, _: &str) -> Result<Option<SecretString>, CredentialError> {
			Ok(None)
		}
	}

	#[test]
	fn embedded_catalog_compiles_one_exact_plan_per_authenticated_spec() {
		let catalog = Catalog::embedded();
		let broker = CredentialBroker::from_catalog(
			catalog,
			Arc::new(EmptyEnvironment),
			CredentialBrokerEngines::default(),
		)
		.expect("credential plans");
		let authenticated = catalog
			.auth_specs()
			.iter()
			.filter(|auth| credential_kind(auth.kind).is_some())
			.count();
		assert_eq!(broker.plans.len(), authenticated);
		for auth in catalog
			.auth_specs()
			.iter()
			.filter(|auth| credential_kind(auth.kind).is_some())
		{
			let plan = broker
				.plans
				.get(&auth.id)
				.expect("plan by exact auth identity");
			assert_eq!(plan.sources.len(), auth.credential_sources.len());
		}
	}

	#[derive(Debug)]
	struct OrderedEnvironment {
		calls: Mutex<Vec<Str>>,
	}

	impl CredentialEnvironment for OrderedEnvironment {
		fn read(&self, name: &str) -> Result<Option<SecretString>, CredentialError> {
			self.calls.lock().push(name.into());
			Ok((name == "OMP_SECOND").then(|| SecretString::from("secret".to_owned())))
		}
	}

	#[tokio::test]
	async fn environment_names_are_tried_in_declared_order() {
		let spec = AuthSpecId::new("ordered");
		let environment = Arc::new(OrderedEnvironment { calls: Mutex::new(Vec::new()) });
		let broker = CredentialBroker {
			plans:       Arc::new(BTreeMap::from([(spec.clone(), BrokerPlan {
				kind:    CredentialKind::ApiKey,
				sources: vec![BrokerSource::Environment(
					vec![Str::from("OMP_FIRST"), Str::from("OMP_SECOND")].into_boxed_slice(),
				)]
				.into_boxed_slice(),
			})])),
			environment: environment.clone(),
			engines:     CredentialBrokerEngines::default(),
		};
		let lease = broker
			.lease(CredentialNeed {
				spec,
				account: Some(AccountId::from("account")),
				principal: Some(PrincipalId::from("principal")),
				valid_after: SystemTime::UNIX_EPOCH,
			})
			.await
			.expect("second source");
		assert_eq!(lease.kind(), CredentialKind::ApiKey);
		assert_eq!(*environment.calls.lock(), vec![Str::from("OMP_FIRST"), Str::from("OMP_SECOND")]);
		assert!(!format!("{broker:?} {lease:?}").contains("secret"));
	}
}
