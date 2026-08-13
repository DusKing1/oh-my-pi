//! Catalog-driven authentication, sealed credential leases, and interactive
//! login protocols.

pub mod adc;
pub mod broker;
pub mod crypto;
pub mod key;
pub mod lease;
pub mod login;
pub mod manager;
pub mod oauth;
pub mod sigv4;
pub mod spec;
pub mod store;
pub use adc::{
	AdcEngine, AdcError, AdcResolution, AdcRuntime, AdcRuntimeError, AdcSourceKind, SystemAdcRuntime,
};
pub use broker::{
	CredentialBroker, CredentialBrokerEngines, CredentialBrokerError, CredentialEnvironment,
	SystemCredentialEnvironment,
};
pub use key::{
	HeadlessKeySource, KeyError, KeyId, KeySource, OsCredentialKeySource, UnavailableKeySource,
};
pub use lease::{
	AppliedCredentials, AuthRejection, AuthRejectionKind, AuthScheme, CredentialApplyError,
	CredentialError, CredentialKind, CredentialLease, CredentialNeed, CredentialSource, LeaseMeta,
};
pub use login::{
	DEFAULT_LOGIN_CHANNEL_CAPACITY, LoginCancellation, LoginChannelError, LoginDriver,
	SecretLoginError, complete_secret_login, default_login_channels, login_channels,
	prompt_for_secret,
};
pub use manager::{
	AuthLoginEngine, AuthManager, AuthManagerBuildError, AuthRefreshEngine,
	CredentialAcquisitionLoginEngine, CredentialAcquisitionLoginEngineError, OAuthLoginEngine,
	OAuthLoginEngineError, SecretLoginEngine, SecretLoginEngineError, StoredOAuthRefreshEngine,
};
pub use oauth::{
	DevicePending, OAuthClock, OAuthCredentialManagerError, OAuthCustomDispatchError,
	OAuthCustomDispatcher, OAuthCustomHandler, OAuthEngine, OAuthEntropy, OAuthError,
	OAuthHttpClient, OAuthHttpRequest, OAuthHttpResponse, OAuthProviderCode, OAuthTokenSet,
	OAuthTransportError, PkcePending, SystemEntropySource, SystemOAuthClock, SystemOAuthHttpClient,
};
pub use spec::{
	AdcSourceSpec, AdcSpec, AuthSpec, AuthSpecError, BearerScheme, BodyPlacement,
	CatalogAuthSpecError, CredentialSourceSpec, HeaderPlacement, KeyPlacement, OAuthClientSpec,
	OAuthCustomSpec, OAuthDeviceSpec, OAuthParameter, OAuthPasteSpec, OAuthPkceSpec,
	OAuthPollingSpec, OAuthRefreshSpec, PkceCompletion, PublicHeader, QueryPlacement,
	SessionTokenSpec, SigV4Spec,
};
pub use store::{
	CredentialMetadata, CredentialOrigin, CredentialStore, CredentialWrite, LeaseOutcome,
	PersistentLease, StoreError, StoredCredentialSource,
};
