//! Tonic authentication projection over canonical typed auth and usage
//! operations.

use std::{
	collections::BTreeMap,
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicU64, Ordering},
	},
	time::Duration,
};

use futures::Stream;
use omp_llm_catalog::ProviderId;
use omp_llm_inference::{
	Client, Registry,
	answer::{AccountState, AccountSummary, AuthAnswer, AuthEvent, AuthSession, UsageReport},
	call::{
		AuthInput, AuthMethod, AuthRequest, CallMeta, LoginRequest, Target, UsageRequest, UsageScope,
	},
	id::{AccountId, LoginSessionId, RequestId},
	receipt::ExecutionBudget,
	router::Router,
};
use omp_proto::omp::auth::v1 as pb;
use parking_lot::Mutex;
use secrecy::SecretString;
use tonic::{Request, Response, Status};

static REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);
type AuthEventStream =
	Pin<Box<dyn Stream<Item = Result<pb::CredentialEvent, Status>> + Send + 'static>>;

/// RPC server that retains interactive login channels while a flow is active.
#[derive(Clone)]
pub struct AuthRpc {
	registry: Registry,
	flows:    Arc<Mutex<BTreeMap<String, AuthSession>>>,
}

impl AuthRpc {
	/// Wraps one immutable comprehensive registry.
	#[must_use]
	pub fn new(registry: Registry) -> Self {
		Self { registry, flows: Arc::new(Mutex::new(BTreeMap::new())) }
	}

	fn provider_for(&self, requested: Option<&str>) -> Result<ProviderId, Status> {
		if let Some(provider) = requested.filter(|value| !value.is_empty()) {
			return Ok(ProviderId::from(provider));
		}
		self
			.registry
			.catalog()
			.providers()
			.iter()
			.find(|provider| {
				provider
					.management
					.supports(omp_llm_catalog::OperationKind::Auth)
			})
			.map(|provider| provider.id.clone())
			.ok_or_else(|| Status::failed_precondition("no constructed route supports authentication"))
	}

	fn client(&self, provider: ProviderId) -> Client<omp_llm_inference::ProviderService, Router> {
		let sequence = REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
		Client::new(
			self.registry.service(),
			Router::new(self.registry.clone(), Duration::from_secs(30)),
			CallMeta {
				id:       RequestId::from(format!("auth-rpc-{sequence}")),
				target:   Target::ProviderService(provider),
				deadline: None,
				budget:   ExecutionBudget::default(),
				session:  None,
			},
		)
	}

	async fn execute(
		&self,
		provider: ProviderId,
		request: AuthRequest,
	) -> Result<AuthAnswer, Status> {
		self
			.client(provider)
			.execute(request)
			.await
			.map_err(inference_status)
	}

	async fn account_operation(
		&self,
		account: u64,
		refresh: bool,
	) -> Result<pb::CredentialMeta, Status> {
		let provider = self.provider_for(None)?;
		let account = AccountId::from(account.to_string());
		let operation = if refresh {
			AuthRequest::Refresh { account }
		} else {
			AuthRequest::Logout { account }
		};
		match self.execute(provider, operation).await? {
			AuthAnswer::Refreshed(account) => account_meta(account),
			AuthAnswer::LoggedOut(account) => Ok(pb::CredentialMeta {
				id: parse_account_id(&account)?,
				state: pb::credential_meta::State::Disabled as i32,
				..pb::CredentialMeta::default()
			}),
			_ => Err(Status::internal("auth operation returned the wrong typed answer")),
		}
	}
}

#[tonic::async_trait]
impl pb::auth_server::Auth for AuthRpc {
	type WatchCredentialsStream = AuthEventStream;

	async fn list_credentials(
		&self,
		request: Request<pb::ListCredentialsRequest>,
	) -> Result<Response<pb::ListCredentialsResponse>, Status> {
		let request = request.into_inner();
		let provider = self.provider_for(Some(&request.provider))?;
		let answer = self
			.execute(provider.clone(), AuthRequest::ListAccounts { provider: Some(provider) })
			.await?;
		let AuthAnswer::Accounts(accounts) = answer else {
			return Err(Status::internal("auth list returned the wrong typed answer"));
		};
		let credentials = accounts
			.into_iter()
			.map(account_meta)
			.collect::<Result<Vec<_>, _>>()?;
		Ok(Response::new(pb::ListCredentialsResponse { credentials, cursor: None }))
	}

	async fn watch_credentials(
		&self,
		_request: Request<pb::WatchCredentialsRequest>,
	) -> Result<Response<Self::WatchCredentialsStream>, Status> {
		let stream = futures::stream::once(async {
			Ok(pb::CredentialEvent {
				cursor: None,
				event:  Some(pb::credential_event::Event::Reset(pb::credential_event::Reset {})),
			})
		});
		Ok(Response::new(Box::pin(stream)))
	}

	async fn begin_login(
		&self,
		request: Request<pb::BeginLoginRequest>,
	) -> Result<Response<pb::BeginLoginResponse>, Status> {
		let provider = self.provider_for(Some(&request.into_inner().provider))?;
		let answer = self
			.execute(provider.clone(), AuthRequest::Login(LoginRequest { provider, method: None }))
			.await?;
		let AuthAnswer::Session(session) = answer else {
			return Err(Status::internal("auth login returned the wrong typed answer"));
		};
		let flow_id = session.id.as_str().to_owned();
		let event = session
			.events
			.recv_async()
			.await
			.map_err(|_| Status::unavailable("auth flow ended before its first step"))?
			.map_err(inference_status)?;
		let step = login_step(event)?;
		self.flows.lock().insert(flow_id.clone(), session);
		Ok(Response::new(pb::BeginLoginResponse { flow_id, step: Some(step) }))
	}

	async fn submit_code(
		&self,
		request: Request<pb::SubmitCodeRequest>,
	) -> Result<Response<pb::CredentialMeta>, Status> {
		let request = request.into_inner();
		let (responses, events) = {
			let flows = self.flows.lock();
			let session = flows
				.get(&request.flow_id)
				.ok_or_else(|| Status::not_found("auth flow not found"))?;
			(session.responses.clone(), session.events.clone())
		};
		let session = LoginSessionId::from(request.flow_id.as_str());
		responses
			.send_async(omp_llm_inference::answer::AuthResponse {
				session,
				input: AuthInput::AuthorizationCode(SecretString::from(request.code)),
			})
			.await
			.map_err(|_| Status::unavailable("auth flow no longer accepts input"))?;
		let account = await_account(events).await?;
		self.flows.lock().remove(&request.flow_id);
		Ok(Response::new(account_meta(account)?))
	}

	async fn wait_login(
		&self,
		request: Request<pb::WaitLoginRequest>,
	) -> Result<Response<pb::CredentialMeta>, Status> {
		let flow_id = request.into_inner().flow_id;
		let events = self
			.flows
			.lock()
			.get(&flow_id)
			.ok_or_else(|| Status::not_found("auth flow not found"))?
			.events
			.clone();
		let account = await_account(events).await?;
		self.flows.lock().remove(&flow_id);
		Ok(Response::new(account_meta(account)?))
	}

	async fn put_api_key(
		&self,
		request: Request<pb::PutApiKeyRequest>,
	) -> Result<Response<pb::CredentialMeta>, Status> {
		let request = request.into_inner();
		let provider = self.provider_for(Some(&request.provider))?;
		let answer = self
			.execute(
				provider.clone(),
				AuthRequest::Login(LoginRequest { provider, method: Some(AuthMethod::ApiKey) }),
			)
			.await?;
		let AuthAnswer::Session(session) = answer else {
			return Err(Status::internal("API-key login returned the wrong typed answer"));
		};
		session
			.responses
			.send_async(omp_llm_inference::answer::AuthResponse {
				session: session.id.clone(),
				input:   AuthInput::ApiKey(SecretString::from(request.api_key)),
			})
			.await
			.map_err(|_| Status::unavailable("API-key login no longer accepts input"))?;
		Ok(Response::new(account_meta(await_account(session.events).await?)?))
	}

	async fn refresh_credential(
		&self,
		request: Request<pb::RefreshCredentialRequest>,
	) -> Result<Response<pb::CredentialMeta>, Status> {
		Ok(Response::new(
			self
				.account_operation(request.into_inner().id, true)
				.await?,
		))
	}

	async fn delete_credential(
		&self,
		request: Request<pb::DeleteCredentialRequest>,
	) -> Result<Response<pb::DeleteCredentialResponse>, Status> {
		self
			.account_operation(request.into_inner().id, false)
			.await?;
		Ok(Response::new(pb::DeleteCredentialResponse {}))
	}

	async fn get_usage(
		&self,
		request: Request<pb::GetUsageRequest>,
	) -> Result<Response<pb::GetUsageResponse>, Status> {
		let request = request.into_inner();
		let provider = self.provider_for(Some(&request.provider))?;
		let account =
			(request.credential_id != 0).then(|| AccountId::from(request.credential_id.to_string()));
		let mut client = self.client(provider.clone());
		let report = client
			.execute(UsageRequest {
				provider: Some(provider),
				account,
				scope: UsageScope::All,
				allow_stale: !request.refresh,
			})
			.await
			.map_err(inference_status)?;
		Ok(Response::new(pb::GetUsageResponse { reports: vec![usage_report(report)] }))
	}

	async fn put_aws_credential(
		&self,
		_request: Request<pb::PutAwsCredentialRequest>,
	) -> Result<Response<pb::CredentialMeta>, Status> {
		Err(not_available("direct AWS secret ingress"))
	}

	async fn import_o_auth(
		&self,
		_request: Request<pb::ImportOAuthRequest>,
	) -> Result<Response<pb::CredentialMeta>, Status> {
		Err(not_available("OAuth token import"))
	}

	async fn disable_credential(
		&self,
		_request: Request<pb::DisableCredentialRequest>,
	) -> Result<Response<pb::CredentialMeta>, Status> {
		Err(not_available("administrative credential disable"))
	}

	async fn enable_credential(
		&self,
		_request: Request<pb::EnableCredentialRequest>,
	) -> Result<Response<pb::CredentialMeta>, Status> {
		Err(not_available("administrative credential enable"))
	}

	async fn report_block(
		&self,
		_request: Request<pb::ReportBlockRequest>,
	) -> Result<Response<pb::CredentialMeta>, Status> {
		Err(not_available("client-reported credential blocks"))
	}

	async fn clear_blocks(
		&self,
		_request: Request<pb::ClearBlocksRequest>,
	) -> Result<Response<pb::CredentialMeta>, Status> {
		Err(not_available("operator block clearing"))
	}

	async fn mark_usage_stale(
		&self,
		_request: Request<pb::MarkUsageStaleRequest>,
	) -> Result<Response<pb::MarkUsageStaleResponse>, Status> {
		Err(not_available("explicit usage cache invalidation"))
	}

	async fn get_usage_history(
		&self,
		_request: Request<pb::GetUsageHistoryRequest>,
	) -> Result<Response<pb::GetUsageHistoryResponse>, Status> {
		Err(not_available("durable usage history"))
	}

	async fn get_client_usage(
		&self,
		_request: Request<pb::GetClientUsageRequest>,
	) -> Result<Response<pb::GetClientUsageResponse>, Status> {
		Err(not_available("per-client usage accounting"))
	}

	async fn mint_scoped_token(
		&self,
		_request: Request<pb::MintScopedTokenRequest>,
	) -> Result<Response<pb::ScopedToken>, Status> {
		Err(not_available("scoped client-direct token minting"))
	}
}

async fn await_account(
	events: flume::Receiver<Result<AuthEvent, omp_llm_inference::Error>>,
) -> Result<AccountSummary, Status> {
	while let Ok(event) = events.recv_async().await {
		if let AuthEvent::Complete(account) = event.map_err(inference_status)? {
			return Ok(account);
		}
	}
	Err(Status::unavailable("auth flow ended without account completion"))
}

fn login_step(event: AuthEvent) -> Result<pb::begin_login_response::Step, Status> {
	match event {
		AuthEvent::OpenUrl(url) => {
			Ok(pb::begin_login_response::Step::Browse(pb::begin_login_response::Browse {
				url: url.as_str().to_owned(),
			}))
		},
		AuthEvent::ShowDeviceCode { code, verification_url } => {
			Ok(pb::begin_login_response::Step::Device(pb::begin_login_response::DeviceCode {
				user_code:  secrecy::ExposeSecret::expose_secret(&code).to_owned(),
				verify_url: verification_url.as_str().to_owned(),
			}))
		},
		AuthEvent::Prompt(prompt) => Err(Status::failed_precondition(format!(
			"auth flow requires {} input via the typed prompt channel",
			prompt.message
		))),
		AuthEvent::Waiting => Err(Status::failed_precondition(
			"auth flow is waiting without a client-visible login step",
		)),
		AuthEvent::Complete(_) => {
			Err(Status::failed_precondition("auth flow completed before returning a login step"))
		},
	}
}

fn account_meta(account: AccountSummary) -> Result<pb::CredentialMeta, Status> {
	Ok(pb::CredentialMeta {
		id:             parse_account_id(&account.account)?,
		provider:       account.provider.as_str().to_owned(),
		kind:           pb::credential_meta::Kind::Unspecified as i32,
		identity:       account
			.principal
			.map_or_else(String::new, |value| value.as_str().to_owned()),
		state:          match account.state {
			AccountState::Active => 1,
			AccountState::RefreshRequired => 2,
			AccountState::Disabled | AccountState::LoggedOut => 4,
		},
		blocks:         Vec::new(),
		disabled_cause: String::new(),
		expires_at_ms:  0,
		created_at_ms:  0,
		updated_at_ms:  0,
	})
}

fn parse_account_id(account: &AccountId) -> Result<u64, Status> {
	account.as_str().parse().map_err(|_| {
		Status::failed_precondition(
			"account identity cannot be represented by the retained numeric auth RPC schema",
		)
	})
}

fn usage_report(report: UsageReport) -> pb::UsageReport {
	pb::UsageReport {
		credential_id: report.account.as_str().parse().unwrap_or(0),
		provider:      report.provider.as_str().to_owned(),
		plan:          String::new(),
		windows:       report
			.windows
			.into_iter()
			.map(|window| pb::UsageWindow {
				label:        window.dimension.as_str().to_owned(),
				used_percent: match (window.consumed, window.limit) {
					(Some(used), Some(limit)) if limit != 0 => (used as f64 / limit as f64) * 100.0,
					_ => 0.0,
				},
				resets_at_ms: window
					.resets_at
					.and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
					.map_or(0, |duration| duration.as_millis().try_into().unwrap_or(u64::MAX)),
			})
			.collect(),
		fetched_at_ms: 0,
		detail:        None,
	}
}

fn not_available(capability: &str) -> Status {
	Status::failed_precondition(format!(
		"{capability} is not exposed by any constructed canonical auth operation"
	))
}
fn inference_status(error: omp_llm_inference::Error) -> Status {
	Status::failed_precondition(error.to_string())
}
