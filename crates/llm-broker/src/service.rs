//! Tonic authentication service and the broker's one-way secret boundary.

use std::{
	collections::BTreeMap,
	pin::Pin,
	sync::Arc,
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use futures::Stream;
use http::{Method, Request as HttpRequest, header::CONTENT_TYPE};
use omp_core::Str;
use omp_proto::{
	auth::v1::{self as proto, auth_server::Auth, credential_event},
	inference::v1::{
		Value as ProtoValue, ValueList as ProtoValueList, ValueMap as ProtoValueMap,
		value as proto_value_kind,
	},
};
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use smallvec::SmallVec;
use tokio::sync::Mutex;
use tonic::{Request, Response, Status};

use crate::{
	oauth::{HttpClient as OAuthHttp, LoginPrompt, OAuthEngine, OAuthError},
	sealed::Secret,
	store::{
		ClientUsage, CredentialBlock, CredentialDelta, CredentialFilter, CredentialKind,
		CredentialMeta, CredentialState, Cursor, DeltaKind, DeltaReplay, Store, StoreError,
		UsageHistoryEntry, UsageReport, UsageWindow,
	},
	usage::{BrokerObserver, UsageError, UsageHttp, UsageManager},
};

const WATCH_POLL_INTERVAL: Duration = Duration::from_millis(50);
const SCOPED_TOKEN_MAX_TTL_MS: u64 = 10 * 60 * 1_000;
const OPENAI_REALTIME_TOKEN_URL: &str = "https://api.openai.com/v1/realtime/client_secrets";
const SECRET_RESPONSE_MESSAGE: &str = "provider authentication operation failed";

/// Server implementation for `omp.auth.v1.Auth`.
///
/// Stored credential bytes only enter through ingress methods. Metadata
/// conversion is centralized in this module and has no field capable of
/// carrying those bytes back to a client.
pub struct AuthService {
	store:         Arc<Store>,
	oauth:         Option<Arc<OAuthEngine>>,
	usage_http:    Arc<dyn UsageHttp>,
	usage:         UsageManager,
	mutation_gate: Mutex<()>,
}

impl AuthService {
	/// Creates an authentication service over the daemon-owned store and the
	/// injected OAuth and usage transports.
	///
	/// # Errors
	///
	/// Returns an error if the embedded OAuth provider table is invalid.
	pub fn new(
		store: Arc<Store>,
		oauth_http: Arc<dyn OAuthHttp>,
		usage_http: Arc<dyn UsageHttp>,
	) -> Result<Self, OAuthError> {
		let oauth = Arc::new(OAuthEngine::new(Arc::clone(&store), oauth_http)?);
		let usage = UsageManager::new(Arc::clone(&store), Arc::clone(&usage_http));
		Ok(Self { store, oauth: Some(oauth), usage_http, usage, mutation_gate: Mutex::new(()) })
	}

	/// Creates a service sharing an already configured OAuth engine.
	#[must_use]
	pub fn with_oauth(
		store: Arc<Store>,
		oauth: Arc<OAuthEngine>,
		usage_http: Arc<dyn UsageHttp>,
	) -> Self {
		let usage = UsageManager::new(Arc::clone(&store), Arc::clone(&usage_http));
		Self { store, oauth: Some(oauth), usage_http, usage, mutation_gate: Mutex::new(()) }
	}

	/// Returns a cloneable non-secret observer for terminal inference outcomes.
	#[must_use]
	pub fn observer(&self) -> BrokerObserver {
		BrokerObserver::new(Arc::clone(&self.store))
	}

	fn oauth(&self) -> Result<&OAuthEngine, Status> {
		self
			.oauth
			.as_deref()
			.ok_or_else(|| Status::failed_precondition("OAuth is not configured"))
	}

	async fn refresh_or_expired(
		&self,
		credential_id: u64,
		now_ms: u64,
		expire_on_failure: bool,
	) -> Result<CredentialMeta, Status> {
		match self
			.oauth()?
			.refresh_credential(credential_id, now_ms)
			.await
		{
			Ok(meta) => Ok(meta),
			Err(error) => {
				let current = self
					.store
					.get_credential(credential_id, now_ms)
					.map_err(store_status)?;
				if let Some(meta) = current
					&& meta.state == CredentialState::Expired
				{
					return Ok(meta);
				}
				if expire_on_failure {
					self
						.store
						.expire_credential(credential_id, now_ms)
						.map_err(store_status)?
						.ok_or_else(|| credential_not_found(credential_id))?;
				}
				Err(oauth_status(error))
			},
		}
	}

	async fn mint_provider_token(
		&self,
		provider: &str,
		facet: &str,
		session_id: &str,
		now_ms: u64,
	) -> Result<proto::ScopedToken, Status> {
		let endpoint = scoped_token_endpoint(provider, facet).ok_or_else(|| {
			Status::unimplemented(format!(
				"unsupported: provider {provider} has no scoped-token mint endpoint for facet {facet}"
			))
		})?;
		if session_id.is_empty() {
			return Err(Status::invalid_argument("session_id must not be empty"));
		}

		let credentials = self
			.store
			.list_credentials(&CredentialFilter {
				provider: Some(provider),
				now_ms,
				..CredentialFilter::default()
			})
			.map_err(store_status)?;
		let credential = credentials
			.into_iter()
			.find(|credential| credential_can_mint(credential, facet, now_ms))
			.ok_or_else(|| {
				Status::resource_exhausted("no credential is available for this scoped token")
			})?;
		let lease = self
			.store
			.lease(credential.id)
			.map_err(store_status)?
			.ok_or_else(|| Status::aborted("credential rotated before scoped-token mint"))?;
		let body = serde_json::to_vec(&json!({
			"session": {
				"type": facet,
				"metadata": { "omp_session_id": session_id },
			},
		}))
		.map(Bytes::from)
		.map_err(|_| Status::internal("failed to encode scoped-token request"))?;
		let request = HttpRequest::builder()
			.method(Method::POST)
			.uri(endpoint)
			.header(CONTENT_TYPE, "application/json")
			.body(body)
			.map_err(|_| Status::internal("failed to construct scoped-token request"))?;
		let request = self
			.store
			.redeem_with(lease.provider(), lease.credential_id(), lease.generation(), |auth| {
				let mut request = request;
				auth.apply_bearer_to_headers(request.headers_mut())?;
				Ok::<_, http::header::InvalidHeaderValue>(request)
			})
			.map_err(store_status)?
			.ok_or_else(|| Status::aborted("credential rotated before scoped-token mint"))?
			.map_err(|_| Status::internal("credential cannot be applied to scoped-token request"))?;
		let upstream = self.usage_http.send(request).await.map_err(usage_status)?;
		if !upstream.status.is_success() {
			return Err(Status::unavailable(format!(
				"scoped-token endpoint returned HTTP {}",
				upstream.status.as_u16()
			)));
		}
		let payload: JsonValue = serde_json::from_slice(&upstream.body)
			.map_err(|_| Status::unavailable("scoped-token endpoint returned invalid JSON"))?;
		let token = payload
			.pointer("/client_secret/value")
			.or_else(|| payload.get("value"))
			.or_else(|| payload.get("token"))
			.and_then(JsonValue::as_str)
			.filter(|token| !token.is_empty())
			.ok_or_else(|| Status::unavailable("scoped-token endpoint omitted its token"))?;
		let expires_at_ms = scoped_expiry_ms(&payload)
			.ok_or_else(|| Status::unavailable("scoped-token endpoint omitted its expiry"))?;
		let ttl_ms = expires_at_ms.saturating_sub(now_ms);
		if ttl_ms == 0 || ttl_ms > SCOPED_TOKEN_MAX_TTL_MS {
			return Err(Status::failed_precondition(
				"provider-issued token does not satisfy the short-TTL policy",
			));
		}
		Ok(proto::ScopedToken { token: token.to_owned(), expires_at_ms })
	}
}

#[tonic::async_trait]
impl Auth for AuthService {
	type WatchCredentialsStream =
		Pin<Box<dyn Stream<Item = Result<proto::CredentialEvent, Status>> + Send>>;

	async fn list_credentials(
		&self,
		request: Request<proto::ListCredentialsRequest>,
	) -> Result<Response<proto::ListCredentialsResponse>, Status> {
		let request = request.into_inner();
		let states = requested_states(&request.states)?;
		let _gate = self.mutation_gate.lock().await;
		let credentials = self
			.store
			.list_credentials(&CredentialFilter {
				provider: nonempty(&request.provider),
				states:   &states,
				now_ms:   now_ms(),
			})
			.map_err(store_status)?;
		let cursor = self.store.cursor().map_err(store_status)?;
		Ok(Response::new(proto::ListCredentialsResponse {
			credentials: credentials.into_iter().map(meta_to_proto).collect(),
			cursor:      Some(cursor_to_proto(cursor)),
		}))
	}

	async fn watch_credentials(
		&self,
		request: Request<proto::WatchCredentialsRequest>,
	) -> Result<Response<Self::WatchCredentialsStream>, Status> {
		let since = request
			.into_inner()
			.since
			.and_then(proto_to_cursor)
			.unwrap_or(Cursor { epoch: [0; 16], generation: u64::MAX });
		let replay = self.store.deltas_since(&since).map_err(store_status)?;
		let (sender, receiver) = flume::bounded(32);
		let store = Arc::clone(&self.store);
		tokio::spawn(async move { drive_watch(store, since, replay, sender).await });
		Ok(Response::new(Box::pin(receiver.into_stream())))
	}

	async fn begin_login(
		&self,
		request: Request<proto::BeginLoginRequest>,
	) -> Result<Response<proto::BeginLoginResponse>, Status> {
		let request = request.into_inner();
		require_nonempty("provider", &request.provider)?;
		let start = self
			.oauth()?
			.begin_login(&request.provider, now_ms())
			.await
			.map_err(oauth_status)?;
		let step = match start.prompt {
			LoginPrompt::Browse { url, .. } | LoginPrompt::Paste { url } => {
				proto::begin_login_response::Step::Browse(proto::begin_login_response::Browse {
					url: url.to_string(),
				})
			},
			LoginPrompt::Device { user_code, verification_url, .. } => {
				proto::begin_login_response::Step::Device(proto::begin_login_response::DeviceCode {
					user_code:  user_code.to_string(),
					verify_url: verification_url.to_string(),
				})
			},
		};
		Ok(Response::new(proto::BeginLoginResponse {
			flow_id: start.flow_id.to_string(),
			step:    Some(step),
		}))
	}

	async fn submit_code(
		&self,
		request: Request<proto::SubmitCodeRequest>,
	) -> Result<Response<proto::CredentialMeta>, Status> {
		let request = request.into_inner();
		require_nonempty("flow_id", &request.flow_id)?;
		require_nonempty("code", &request.code)?;
		let proto::SubmitCodeRequest { flow_id, code, state } = request;
		let code = Secret::from_vec(code.into_bytes());
		let _gate = self.mutation_gate.lock().await;
		let meta = self
			.oauth()?
			.submit_code(&flow_id, &code, &state, now_ms())
			.await
			.map_err(oauth_status)?;
		Ok(Response::new(meta_to_proto(meta)))
	}

	async fn wait_login(
		&self,
		request: Request<proto::WaitLoginRequest>,
	) -> Result<Response<proto::CredentialMeta>, Status> {
		let request = request.into_inner();
		require_nonempty("flow_id", &request.flow_id)?;
		let _gate = self.mutation_gate.lock().await;
		let meta = self
			.oauth()?
			.wait_login(&request.flow_id, now_ms())
			.await
			.map_err(oauth_status)?;
		Ok(Response::new(meta_to_proto(meta)))
	}

	async fn put_api_key(
		&self,
		request: Request<proto::PutApiKeyRequest>,
	) -> Result<Response<proto::CredentialMeta>, Status> {
		let request = request.into_inner();
		require_nonempty("provider", &request.provider)?;
		require_nonempty("api_key", &request.api_key)?;
		let proto::PutApiKeyRequest { provider, api_key } = request;
		let secret = Secret::from_vec(api_key.into_bytes());
		let _gate = self.mutation_gate.lock().await;
		let meta = self
			.store
			.upsert_api_key(&provider, "", secret.expose(), now_ms())
			.map_err(store_status)?;
		Ok(Response::new(meta_to_proto(meta)))
	}

	async fn put_aws_credential(
		&self,
		request: Request<proto::PutAwsCredentialRequest>,
	) -> Result<Response<proto::CredentialMeta>, Status> {
		let request = request.into_inner();
		require_nonempty("provider", &request.provider)?;
		require_nonempty("identity", &request.identity)?;
		if request.access_key_id.is_empty() {
			return Err(Status::invalid_argument("access_key_id must not be empty"));
		}
		if request.secret_access_key.is_empty() {
			return Err(Status::invalid_argument("secret_access_key must not be empty"));
		}
		let access_key = Secret::new(&request.access_key_id);
		let secret_key = Secret::new(&request.secret_access_key);
		let session_token =
			(!request.session_token.is_empty()).then(|| Secret::new(&request.session_token));
		let _gate = self.mutation_gate.lock().await;
		let meta = self
			.store
			.upsert_aws(
				&request.provider,
				&request.identity,
				access_key.expose(),
				secret_key.expose(),
				session_token.as_ref().map(Secret::expose),
				now_ms(),
			)
			.map_err(store_status)?;
		Ok(Response::new(meta_to_proto(meta)))
	}

	async fn import_o_auth(
		&self,
		request: Request<proto::ImportOAuthRequest>,
	) -> Result<Response<proto::CredentialMeta>, Status> {
		let request = request.into_inner();
		require_nonempty("provider", &request.provider)?;
		require_nonempty("refresh_token", &request.refresh_token)?;
		let proto::ImportOAuthRequest {
			provider,
			refresh_token,
			access_token,
			expires_at_ms,
			identity,
			props,
		} = request;
		let props = props.map_or_else(|| JsonValue::Object(JsonMap::new()), proto_value_map_to_json);
		let access_missing = access_token.is_empty();
		let access = Secret::from_vec(access_token.into_bytes());
		let refresh = Secret::from_vec(refresh_token.into_bytes());
		let operation_now_ms = now_ms();
		let _gate = self.mutation_gate.lock().await;
		let meta = self
			.oauth()?
			.import_oauth(
				&provider,
				&identity,
				&access,
				Some(&refresh),
				&props,
				expires_at_ms,
				operation_now_ms,
			)
			.map_err(oauth_status)?;
		let meta = if access_missing || expires_at_ms <= operation_now_ms {
			self
				.refresh_or_expired(meta.id, operation_now_ms, true)
				.await?
		} else {
			meta
		};
		Ok(Response::new(meta_to_proto(meta)))
	}

	async fn refresh_credential(
		&self,
		request: Request<proto::RefreshCredentialRequest>,
	) -> Result<Response<proto::CredentialMeta>, Status> {
		let id = request.into_inner().id;
		require_id(id)?;
		let _gate = self.mutation_gate.lock().await;
		let meta = self.refresh_or_expired(id, now_ms(), false).await?;
		Ok(Response::new(meta_to_proto(meta)))
	}

	async fn disable_credential(
		&self,
		request: Request<proto::DisableCredentialRequest>,
	) -> Result<Response<proto::CredentialMeta>, Status> {
		let request = request.into_inner();
		require_id(request.id)?;
		let _gate = self.mutation_gate.lock().await;
		let meta = self
			.store
			.disable_credential(request.id, &request.cause, now_ms())
			.map_err(store_status)?
			.ok_or_else(|| credential_not_found(request.id))?;
		Ok(Response::new(meta_to_proto(meta)))
	}

	async fn enable_credential(
		&self,
		request: Request<proto::EnableCredentialRequest>,
	) -> Result<Response<proto::CredentialMeta>, Status> {
		let id = request.into_inner().id;
		require_id(id)?;
		let _gate = self.mutation_gate.lock().await;
		let operation_now_ms = now_ms();
		let before = self
			.store
			.get_credential(id, operation_now_ms)
			.map_err(store_status)?
			.ok_or_else(|| credential_not_found(id))?;
		if before.state != CredentialState::Disabled {
			return Err(Status::failed_precondition("credential is not disabled"));
		}
		let meta = self
			.store
			.enable_credential(id, operation_now_ms)
			.map_err(store_status)?
			.ok_or_else(|| credential_not_found(id))?;
		Ok(Response::new(meta_to_proto(meta)))
	}

	async fn delete_credential(
		&self,
		request: Request<proto::DeleteCredentialRequest>,
	) -> Result<Response<proto::DeleteCredentialResponse>, Status> {
		let id = request.into_inner().id;
		require_id(id)?;
		let _gate = self.mutation_gate.lock().await;
		if !self
			.store
			.delete_credential(id, now_ms())
			.map_err(store_status)?
		{
			return Err(credential_not_found(id));
		}
		Ok(Response::new(proto::DeleteCredentialResponse {}))
	}

	async fn report_block(
		&self,
		request: Request<proto::ReportBlockRequest>,
	) -> Result<Response<proto::CredentialMeta>, Status> {
		let request = request.into_inner();
		require_id(request.id)?;
		let block = request
			.block
			.ok_or_else(|| Status::invalid_argument("block is required"))?;
		require_nonempty("block.scope", &block.scope)?;
		let _gate = self.mutation_gate.lock().await;
		let meta = self
			.store
			.report_block(
				request.id,
				&CredentialBlock {
					scope:        block.scope.into(),
					provider_key: block.provider_key.into(),
					until_ms:     block.until_ms,
				},
				now_ms(),
			)
			.map_err(store_status)?
			.ok_or_else(|| credential_not_found(request.id))?;
		Ok(Response::new(meta_to_proto(meta)))
	}

	async fn clear_blocks(
		&self,
		request: Request<proto::ClearBlocksRequest>,
	) -> Result<Response<proto::CredentialMeta>, Status> {
		let request = request.into_inner();
		require_id(request.id)?;
		let scopes: SmallVec<Str, 4> = request.scopes.into_iter().map(Into::into).collect();
		let _gate = self.mutation_gate.lock().await;
		let meta = self
			.store
			.clear_blocks(request.id, &scopes, now_ms())
			.map_err(store_status)?
			.ok_or_else(|| credential_not_found(request.id))?;
		Ok(Response::new(meta_to_proto(meta)))
	}

	async fn get_usage(
		&self,
		request: Request<proto::GetUsageRequest>,
	) -> Result<Response<proto::GetUsageResponse>, Status> {
		let request = request.into_inner();
		let reports = self
			.usage
			.get_usage(
				nonempty(&request.provider),
				(request.credential_id != 0).then_some(request.credential_id),
				request.refresh,
				now_ms(),
			)
			.await
			.map_err(usage_status)?;
		Ok(Response::new(proto::GetUsageResponse {
			reports: reports.into_iter().map(usage_to_proto).collect(),
		}))
	}

	async fn mark_usage_stale(
		&self,
		request: Request<proto::MarkUsageStaleRequest>,
	) -> Result<Response<proto::MarkUsageStaleResponse>, Status> {
		let request = request.into_inner();
		self
			.usage
			.mark_stale(
				nonempty(&request.provider),
				(request.credential_id != 0).then_some(request.credential_id),
			)
			.map_err(usage_status)?;
		Ok(Response::new(proto::MarkUsageStaleResponse {}))
	}

	async fn get_usage_history(
		&self,
		request: Request<proto::GetUsageHistoryRequest>,
	) -> Result<Response<proto::GetUsageHistoryResponse>, Status> {
		let request = request.into_inner();
		require_id(request.credential_id)?;
		let entries = self
			.store
			.usage_history(request.credential_id, request.since_ms, request.until_ms)
			.map_err(store_status)?;
		Ok(Response::new(proto::GetUsageHistoryResponse {
			entries: entries.into_iter().map(history_to_proto).collect(),
		}))
	}

	async fn get_client_usage(
		&self,
		request: Request<proto::GetClientUsageRequest>,
	) -> Result<Response<proto::GetClientUsageResponse>, Status> {
		let clients = self
			.store
			.client_usage(request.into_inner().since_ms)
			.map_err(store_status)?;
		Ok(Response::new(proto::GetClientUsageResponse {
			clients: clients.into_iter().map(client_usage_to_proto).collect(),
		}))
	}

	async fn mint_scoped_token(
		&self,
		request: Request<proto::MintScopedTokenRequest>,
	) -> Result<Response<proto::ScopedToken>, Status> {
		let request = request.into_inner();
		require_nonempty("provider", &request.provider)?;
		require_nonempty("facet", &request.facet)?;
		let token = self
			.mint_provider_token(&request.provider, &request.facet, &request.session_id, now_ms())
			.await?;
		Ok(Response::new(token))
	}
}

async fn drive_watch(
	store: Arc<Store>,
	mut cursor: Cursor,
	mut replay: DeltaReplay,
	sender: flume::Sender<Result<proto::CredentialEvent, Status>>,
) {
	loop {
		match replay {
			DeltaReplay::Reset(reset_cursor) => {
				cursor = reset_cursor;
				let event = proto::CredentialEvent {
					cursor: Some(cursor_to_proto(reset_cursor)),
					event:  Some(credential_event::Event::Reset(credential_event::Reset {})),
				};
				if sender.send_async(Ok(event)).await.is_err() {
					return;
				}
			},
			DeltaReplay::Deltas(deltas) => {
				for delta in deltas {
					cursor = delta.cursor;
					let event = match delta_to_event(&store, delta) {
						Ok(event) => event,
						Err(status) => {
							let _ = sender.send_async(Err(status)).await;
							return;
						},
					};
					if sender.send_async(Ok(event)).await.is_err() {
						return;
					}
				}
			},
		}
		tokio::time::sleep(WATCH_POLL_INTERVAL).await;
		replay = match store.deltas_since(&cursor) {
			Ok(replay) => replay,
			Err(error) => {
				let _ = sender.send_async(Err(store_status(error))).await;
				return;
			},
		};
	}
}

fn delta_to_event(store: &Store, delta: CredentialDelta) -> Result<proto::CredentialEvent, Status> {
	let event = match delta.kind {
		DeltaKind::Deleted => credential_event::Event::DeletedId(delta.credential_id),
		DeltaKind::Upserted => match store
			.get_credential(delta.credential_id, now_ms())
			.map_err(store_status)?
		{
			Some(meta) => credential_event::Event::Upserted(meta_to_proto(meta)),
			None => credential_event::Event::DeletedId(delta.credential_id),
		},
	};
	Ok(proto::CredentialEvent { cursor: Some(cursor_to_proto(delta.cursor)), event: Some(event) })
}

fn requested_states(states: &[i32]) -> Result<SmallVec<CredentialState, 4>, Status> {
	states
		.iter()
		.map(|state| match proto::credential_meta::State::try_from(*state) {
			Ok(proto::credential_meta::State::Active) => Ok(CredentialState::Active),
			Ok(proto::credential_meta::State::Expired) => Ok(CredentialState::Expired),
			Ok(proto::credential_meta::State::Blocked) => Ok(CredentialState::Blocked),
			Ok(proto::credential_meta::State::Disabled) => Ok(CredentialState::Disabled),
			Ok(proto::credential_meta::State::Unspecified) | Err(_) => {
				Err(Status::invalid_argument("states contains an unspecified value"))
			},
		})
		.collect()
}

fn meta_to_proto(meta: CredentialMeta) -> proto::CredentialMeta {
	proto::CredentialMeta {
		id:             meta.id,
		provider:       meta.provider.to_string(),
		kind:           match meta.kind {
			CredentialKind::ApiKey => proto::credential_meta::Kind::ApiKey.into(),
			CredentialKind::OAuth => proto::credential_meta::Kind::Oauth.into(),
			CredentialKind::Aws => proto::credential_meta::Kind::Aws.into(),
		},
		identity:       meta.identity.to_string(),
		state:          match meta.state {
			CredentialState::Active => proto::credential_meta::State::Active.into(),
			CredentialState::Expired => proto::credential_meta::State::Expired.into(),
			CredentialState::Blocked => proto::credential_meta::State::Blocked.into(),
			CredentialState::Disabled => proto::credential_meta::State::Disabled.into(),
		},
		blocks:         meta
			.blocks
			.into_iter()
			.map(|block| proto::Block {
				scope:        block.scope.to_string(),
				provider_key: block.provider_key.to_string(),
				until_ms:     block.until_ms,
			})
			.collect(),
		disabled_cause: meta.disabled_cause.to_string(),
		expires_at_ms:  meta.expires_at_ms,
		created_at_ms:  meta.created_at_ms,
		updated_at_ms:  meta.updated_at_ms,
	}
}

fn cursor_to_proto(cursor: Cursor) -> proto::Cursor {
	proto::Cursor {
		epoch:      Bytes::copy_from_slice(&cursor.epoch),
		generation: cursor.generation,
	}
}

fn proto_to_cursor(cursor: proto::Cursor) -> Option<Cursor> {
	let epoch = <[u8; 16]>::try_from(cursor.epoch.as_ref()).ok()?;
	Some(Cursor { epoch, generation: cursor.generation })
}

fn usage_to_proto(report: UsageReport) -> proto::UsageReport {
	proto::UsageReport {
		credential_id: report.credential_id,
		provider:      report.provider.to_string(),
		plan:          report.plan.to_string(),
		windows:       report.windows.into_iter().map(window_to_proto).collect(),
		fetched_at_ms: report.fetched_at_ms,
		detail:        Some(json_to_proto_value_map(report.detail)),
	}
}

fn window_to_proto(window: UsageWindow) -> proto::UsageWindow {
	proto::UsageWindow {
		label:        window.label.to_string(),
		used_percent: window.used_percent,
		resets_at_ms: window.resets_at_ms,
	}
}

fn history_to_proto(entry: UsageHistoryEntry) -> proto::get_usage_history_response::Entry {
	proto::get_usage_history_response::Entry {
		at_ms:  entry.at_ms,
		report: Some(usage_to_proto(entry.report)),
	}
}

fn client_usage_to_proto(usage: ClientUsage) -> proto::get_client_usage_response::ClientUsage {
	proto::get_client_usage_response::ClientUsage {
		client_id:          usage.client_id.to_string(),
		label:              usage.label.to_string(),
		input_tokens:       usage.input_tokens,
		output_tokens:      usage.output_tokens,
		cache_read_tokens:  usage.cache_read_tokens,
		cache_write_tokens: usage.cache_write_tokens,
		nanos_usd:          usage.nanos_usd,
		last_seen_ms:       usage.last_seen_ms,
	}
}

fn proto_value_map_to_json(value: ProtoValueMap) -> JsonValue {
	JsonValue::Object(
		value
			.fields
			.into_iter()
			.map(|(key, value)| (key, proto_value_to_json(value)))
			.collect(),
	)
}

fn proto_value_to_json(value: ProtoValue) -> JsonValue {
	match value.kind {
		None | Some(proto_value_kind::Kind::Null(_)) => JsonValue::Null,
		Some(proto_value_kind::Kind::Int(value)) => JsonValue::Number(value.into()),
		Some(proto_value_kind::Kind::Uint(value)) => JsonValue::Number(value.into()),
		Some(proto_value_kind::Kind::Double(value)) => {
			serde_json::Number::from_f64(value).map_or(JsonValue::Null, JsonValue::Number)
		},
		Some(proto_value_kind::Kind::String(value)) => JsonValue::String(value),
		Some(proto_value_kind::Kind::Bool(value)) => JsonValue::Bool(value),
		Some(proto_value_kind::Kind::Map(value)) => proto_value_map_to_json(value),
		Some(proto_value_kind::Kind::List(value)) => {
			JsonValue::Array(value.values.into_iter().map(proto_value_to_json).collect())
		},
	}
}

fn json_to_proto_value_map(value: JsonValue) -> ProtoValueMap {
	let fields = match value {
		JsonValue::Object(fields) => fields,
		value => JsonMap::from_iter([(String::from("value"), value)]),
	};
	ProtoValueMap {
		fields: fields
			.into_iter()
			.map(|(key, value)| (key, json_to_proto_value(value)))
			.collect::<BTreeMap<_, _>>(),
	}
}

fn json_to_proto_value(value: JsonValue) -> ProtoValue {
	let kind = match value {
		JsonValue::Null => proto_value_kind::Kind::Null(true),
		JsonValue::Bool(value) => proto_value_kind::Kind::Bool(value),
		JsonValue::Number(value) => {
			if let Some(value) = value.as_i64() {
				proto_value_kind::Kind::Int(value)
			} else if let Some(value) = value.as_u64() {
				proto_value_kind::Kind::Uint(value)
			} else {
				proto_value_kind::Kind::Double(value.as_f64().expect("JSON numbers are finite"))
			}
		},
		JsonValue::String(value) => proto_value_kind::Kind::String(value),
		JsonValue::Array(values) => proto_value_kind::Kind::List(ProtoValueList {
			values: values.into_iter().map(json_to_proto_value).collect(),
		}),
		JsonValue::Object(fields) => {
			proto_value_kind::Kind::Map(json_to_proto_value_map(JsonValue::Object(fields)))
		},
	};
	ProtoValue { kind: Some(kind) }
}

fn credential_can_mint(credential: &CredentialMeta, facet: &str, now_ms: u64) -> bool {
	matches!(credential.state, CredentialState::Active | CredentialState::Blocked)
		&& credential
			.blocks
			.iter()
			.all(|block| block.until_ms <= now_ms || (block.scope != "shared" && block.scope != facet))
}

fn scoped_token_endpoint(provider: &str, facet: &str) -> Option<&'static str> {
	match (provider, facet) {
		("openai", "realtime") => Some(OPENAI_REALTIME_TOKEN_URL),
		_ => None,
	}
}

fn scoped_expiry_ms(payload: &JsonValue) -> Option<u64> {
	if let Some(expires_at_ms) = payload
		.pointer("/client_secret/expires_at_ms")
		.or_else(|| payload.get("expires_at_ms"))
		.and_then(JsonValue::as_u64)
	{
		return Some(expires_at_ms);
	}
	payload
		.pointer("/client_secret/expires_at")
		.or_else(|| payload.get("expires_at"))
		.and_then(JsonValue::as_u64)
		.and_then(|expires_at| {
			if expires_at < 10_000_000_000 {
				expires_at.checked_mul(1_000)
			} else {
				Some(expires_at)
			}
		})
}

fn nonempty(value: &str) -> Option<&str> {
	(!value.is_empty()).then_some(value)
}

fn require_nonempty(field: &str, value: &str) -> Result<(), Status> {
	if value.is_empty() {
		Err(Status::invalid_argument(format!("{field} must not be empty")))
	} else {
		Ok(())
	}
}

fn require_id(id: u64) -> Result<(), Status> {
	if id == 0 {
		Err(Status::invalid_argument("credential id must not be zero"))
	} else {
		Ok(())
	}
}

fn credential_not_found(id: u64) -> Status {
	Status::not_found(format!("credential {id} does not exist"))
}

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis()
		.try_into()
		.unwrap_or(u64::MAX)
}

fn store_status(error: StoreError) -> Status {
	tracing::error!(error = ?error, "auth store operation failed");
	Status::internal("credential store operation failed")
}

fn oauth_status(error: OAuthError) -> Status {
	tracing::warn!(error = ?error, "OAuth operation failed");
	match error {
		OAuthError::UnknownProvider => Status::invalid_argument("OAuth provider is not configured"),
		OAuthError::UnknownFlow(_) => Status::not_found("OAuth flow is not pending"),
		OAuthError::WrongFlow(_) => {
			Status::failed_precondition("OAuth flow does not support this operation")
		},
		OAuthError::InvalidUrl(_) | OAuthError::InvalidCallback(_) | OAuthError::StateMismatch => {
			Status::invalid_argument("OAuth callback is invalid")
		},
		OAuthError::Provider { transient: true, .. } | OAuthError::Transport(_) => {
			Status::unavailable("OAuth provider is temporarily unavailable")
		},
		OAuthError::Provider { transient: false, .. } => {
			Status::failed_precondition("OAuth provider rejected the operation")
		},
		OAuthError::InvalidResponse(_) => {
			Status::unavailable("OAuth provider returned an invalid response")
		},
		OAuthError::LoginExpired => Status::deadline_exceeded("OAuth login expired"),
		OAuthError::Params(_) | OAuthError::Store(_) => Status::internal(SECRET_RESPONSE_MESSAGE),
	}
}

fn usage_status(error: UsageError) -> Status {
	tracing::warn!(error = ?error, "usage operation failed");
	match error {
		UsageError::UnsupportedProvider(_) => Status::unimplemented("provider has no usage fetcher"),
		_ => Status::unavailable("provider usage operation failed"),
	}
}

#[cfg(test)]
mod tests {

	use futures::StreamExt;
	use http::StatusCode;
	use tempfile::TempDir;

	use super::*;
	use crate::usage::UsageHttpResponse;

	#[derive(Clone)]
	struct StaticHttp {
		body: Bytes,
	}

	impl UsageHttp for StaticHttp {
		fn send(
			&self,
			_request: HttpRequest<Bytes>,
		) -> Pin<Box<dyn Future<Output = Result<UsageHttpResponse, UsageError>> + Send + '_>> {
			let body = self.body.clone();
			Box::pin(async move { Ok(UsageHttpResponse { status: StatusCode::OK, body }) })
		}
	}

	fn harness(body: JsonValue) -> (TempDir, Arc<Store>, AuthService) {
		let directory = TempDir::new().expect("temporary store directory");
		let store = Arc::new(Store::open(directory.path().join("auth.sqlite")).expect("open store"));
		let usage_http: Arc<dyn UsageHttp> = Arc::new(StaticHttp {
			body: Bytes::from(serde_json::to_vec(&body).expect("encode response")),
		});
		let usage = UsageManager::new(Arc::clone(&store), Arc::clone(&usage_http));
		let service = AuthService {
			store: Arc::clone(&store),
			oauth: None,
			usage_http,
			usage,
			mutation_gate: Mutex::new(()),
		};
		(directory, store, service)
	}

	async fn put_key(service: &AuthService, provider: &str, key: &str) -> proto::CredentialMeta {
		service
			.put_api_key(Request::new(proto::PutApiKeyRequest {
				provider: provider.to_owned(),
				api_key:  key.to_owned(),
			}))
			.await
			.expect("put api key")
			.into_inner()
	}

	async fn list(service: &AuthService) -> proto::ListCredentialsResponse {
		service
			.list_credentials(Request::new(proto::ListCredentialsRequest {
				provider: String::new(),
				states:   Vec::new(),
			}))
			.await
			.expect("list credentials")
			.into_inner()
	}

	fn is_reset(event: &proto::CredentialEvent) -> bool {
		matches!(event.event, Some(credential_event::Event::Reset(_)))
	}

	#[test]
	fn fully_populated_metadata_serializes_without_secret_material() {
		let secret = "sk-secret-material-that-must-never-egress";
		let meta = proto::CredentialMeta {
			id:             7,
			provider:       String::from("openai"),
			kind:           proto::credential_meta::Kind::Oauth.into(),
			identity:       String::from("account@example.test"),
			state:          proto::credential_meta::State::Blocked.into(),
			blocks:         vec![proto::Block {
				scope:        String::from("realtime"),
				provider_key: String::from("account-7"),
				until_ms:     99,
			}],
			disabled_cause: String::from("operator policy"),
			expires_at_ms:  100,
			created_at_ms:  10,
			updated_at_ms:  20,
		};
		let encoded = serde_json::to_vec(&meta).expect("serialize metadata");
		assert!(
			!encoded
				.windows(secret.len())
				.any(|window| window == secret.as_bytes())
		);
		let encoded_text = std::str::from_utf8(&encoded).expect("metadata JSON is UTF-8");
		assert!(!encoded_text.contains("secret"));
		assert!(!encoded_text.contains("token"));
	}

	#[tokio::test]
	async fn put_api_key_then_list_never_returns_the_key() {
		let (_directory, _store, service) = harness(json!({}));
		let key = "sk-one-way-ingress-secret";
		let response = put_key(&service, "openai", key).await;
		let response_bytes = serde_json::to_vec(&response).expect("serialize put response");
		assert!(
			!response_bytes
				.windows(key.len())
				.any(|window| window == key.as_bytes())
		);

		let listed = list(&service).await;
		assert_eq!(listed.credentials.len(), 1);
		let listed_bytes = serde_json::to_vec(&listed).expect("serialize list response");
		assert!(
			!listed_bytes
				.windows(key.len())
				.any(|window| window == key.as_bytes())
		);
	}

	#[tokio::test]
	async fn put_aws_tuple_returns_only_distinct_non_secret_metadata() {
		let (_directory, _store, service) = harness(json!({}));
		let access = b"AKIDEXAMPLE".to_vec();
		let secret = b"aws-secret-material".to_vec();
		let session = b"aws-session-material".to_vec();
		let response = service
			.put_aws_credential(Request::new(proto::PutAwsCredentialRequest {
				provider:          "bedrock".into(),
				identity:          "account".into(),
				access_key_id:     access.clone().into(),
				secret_access_key: secret.clone().into(),
				session_token:     session.clone().into(),
			}))
			.await
			.expect("put AWS credential")
			.into_inner();
		assert_eq!(response.kind, proto::credential_meta::Kind::Aws as i32);
		let encoded = serde_json::to_vec(&response).expect("serialize response");
		for material in [&access, &secret, &session] {
			assert!(
				!encoded
					.windows(material.len())
					.any(|window| window == material)
			);
		}
	}

	#[tokio::test]
	async fn list_credentials_applies_provider_and_state_filters() {
		let (_directory, _store, service) = harness(json!({}));
		let openai = put_key(&service, "openai", "openai-key").await;
		let anthropic = put_key(&service, "anthropic", "anthropic-key").await;
		service
			.disable_credential(Request::new(proto::DisableCredentialRequest {
				id:    anthropic.id,
				cause: String::from("operator"),
			}))
			.await
			.expect("disable credential");

		let provider_filtered = service
			.list_credentials(Request::new(proto::ListCredentialsRequest {
				provider: String::from("openai"),
				states:   Vec::new(),
			}))
			.await
			.expect("provider-filtered list")
			.into_inner();
		assert_eq!(provider_filtered.credentials.len(), 1);
		assert_eq!(provider_filtered.credentials[0].id, openai.id);

		let state_filtered = service
			.list_credentials(Request::new(proto::ListCredentialsRequest {
				provider: String::new(),
				states:   vec![proto::credential_meta::State::Disabled.into()],
			}))
			.await
			.expect("state-filtered list")
			.into_inner();
		assert_eq!(state_filtered.credentials.len(), 1);
		assert_eq!(state_filtered.credentials[0].id, anthropic.id);
	}

	#[tokio::test]
	async fn watch_resumes_from_a_live_cursor() {
		let (_directory, _store, service) = harness(json!({}));
		let cursor = list(&service).await.cursor.expect("list cursor");
		let inserted = put_key(&service, "openai", "live-cursor-key").await;
		let mut stream = service
			.watch_credentials(Request::new(proto::WatchCredentialsRequest { since: Some(cursor) }))
			.await
			.expect("watch credentials")
			.into_inner();
		let event = stream.next().await.expect("event").expect("valid event");
		assert!(matches!(
			event.event.as_ref(),
			Some(credential_event::Event::Upserted(meta)) if meta.id == inserted.id
		));
		assert_eq!(
			event.cursor.expect("event cursor").generation,
			list(&service)
				.await
				.cursor
				.expect("current cursor")
				.generation
		);
	}

	#[tokio::test]
	async fn stale_epoch_cursor_yields_reset_first() {
		let (_directory, _store, service) = harness(json!({}));
		let current = list(&service).await.cursor.expect("list cursor");
		let mut stale_epoch = vec![0xa5; 16];
		if stale_epoch.as_slice() == current.epoch.as_ref() {
			stale_epoch[0] ^= 1;
		}
		let mut stream = service
			.watch_credentials(Request::new(proto::WatchCredentialsRequest {
				since: Some(proto::Cursor {
					epoch:      Bytes::from(stale_epoch),
					generation: current.generation,
				}),
			}))
			.await
			.expect("watch credentials")
			.into_inner();
		let first = stream.next().await.expect("reset").expect("valid reset");
		assert!(is_reset(&first));
		assert_eq!(first.cursor.expect("reset cursor"), current);
	}

	#[tokio::test]
	async fn cursor_beyond_retained_window_yields_reset() {
		let (_directory, store, service) = harness(json!({}));
		let old = list(&service).await.cursor.expect("old cursor");
		for generation in 0..=1_024 {
			store
				.upsert_api_key("openai", "", format!("key-{generation}").as_bytes(), generation + 1)
				.expect("rotate key");
		}
		let mut stream = service
			.watch_credentials(Request::new(proto::WatchCredentialsRequest { since: Some(old) }))
			.await
			.expect("watch credentials")
			.into_inner();
		assert!(is_reset(&stream.next().await.expect("reset").expect("valid reset")));
	}

	#[tokio::test]
	async fn relist_after_reset_rebases_without_loss_or_duplication() {
		let (_directory, store, service) = harness(json!({}));
		let inserted = put_key(&service, "openai", "initial-key").await;
		let old = list(&service).await.cursor.expect("old cursor");
		for generation in 0..=1_024 {
			store
				.upsert_api_key(
					"openai",
					"",
					format!("rotated-{generation}").as_bytes(),
					generation + 10,
				)
				.expect("rotate key");
		}
		let mut stale_stream = service
			.watch_credentials(Request::new(proto::WatchCredentialsRequest { since: Some(old) }))
			.await
			.expect("stale watch")
			.into_inner();
		assert!(is_reset(
			&stale_stream
				.next()
				.await
				.expect("reset")
				.expect("valid reset")
		));
		drop(stale_stream);

		let rebased = list(&service).await;
		assert_eq!(
			rebased
				.credentials
				.iter()
				.filter(|meta| meta.id == inserted.id)
				.count(),
			1
		);
		store
			.upsert_api_key("openai", "", b"after-rebase", 50_000)
			.expect("post-rebase mutation");
		let mut resumed = service
			.watch_credentials(Request::new(proto::WatchCredentialsRequest { since: rebased.cursor }))
			.await
			.expect("rebased watch")
			.into_inner();
		let event = resumed.next().await.expect("upsert").expect("valid upsert");
		assert!(matches!(
			event.event.as_ref(),
			Some(credential_event::Event::Upserted(meta)) if meta.id == inserted.id
		));
	}

	#[tokio::test]
	async fn scoped_blocks_filter_facets_independently() {
		let expires_at_ms = now_ms() + 30_000;
		let (_directory, _store, service) = harness(json!({
			"client_secret": { "value": "ephemeral", "expires_at_ms": expires_at_ms }
		}));
		let credential = put_key(&service, "openai", "provider-key").await;
		service
			.report_block(Request::new(proto::ReportBlockRequest {
				id:    credential.id,
				block: Some(proto::Block {
					scope:        String::from("chat"),
					provider_key: String::new(),
					until_ms:     now_ms() + 60_000,
				}),
			}))
			.await
			.expect("chat block");
		service
			.mint_scoped_token(Request::new(proto::MintScopedTokenRequest {
				provider:   String::from("openai"),
				facet:      String::from("realtime"),
				session_id: String::from("session-a"),
			}))
			.await
			.expect("unrelated block must not filter credential");

		service
			.report_block(Request::new(proto::ReportBlockRequest {
				id:    credential.id,
				block: Some(proto::Block {
					scope:        String::from("realtime"),
					provider_key: String::new(),
					until_ms:     now_ms() + 60_000,
				}),
			}))
			.await
			.expect("realtime block");
		let error = service
			.mint_scoped_token(Request::new(proto::MintScopedTokenRequest {
				provider:   String::from("openai"),
				facet:      String::from("realtime"),
				session_id: String::from("session-b"),
			}))
			.await
			.expect_err("matching scope must filter credential");
		assert_eq!(error.code(), tonic::Code::ResourceExhausted);
	}

	#[tokio::test]
	async fn minted_scoped_token_respects_short_ttl() {
		let before = now_ms();
		let provider_expiry = before + 30_000;
		let (_directory, _store, service) = harness(json!({
			"client_secret": { "value": "ephemeral-session-token", "expires_at_ms": provider_expiry }
		}));
		put_key(&service, "openai", "provider-key").await;
		let token = service
			.mint_scoped_token(Request::new(proto::MintScopedTokenRequest {
				provider:   String::from("openai"),
				facet:      String::from("realtime"),
				session_id: String::from("one-session"),
			}))
			.await
			.expect("mint token")
			.into_inner();
		assert_eq!(token.token, "ephemeral-session-token");
		assert!(token.expires_at_ms > before);
		assert!(token.expires_at_ms.saturating_sub(before) <= SCOPED_TOKEN_MAX_TTL_MS);
	}

	#[tokio::test]
	async fn observer_atomically_accumulates_terminal_client_usage_and_exact_cost() {
		let (_directory, store, service) = harness(json!({}));
		let usage = ClientUsage {
			client_id:          "client-a".into(),
			label:              "ci/nightly".into(),
			input_tokens:       11,
			output_tokens:      7,
			cache_read_tokens:  5,
			cache_write_tokens: 3,
			nanos_usd:          1_234_567,
			last_seen_ms:       42,
		};
		let mut writers = Vec::new();
		for _ in 0..8 {
			let observer = service.observer();
			let usage = usage.clone();
			writers.push(std::thread::spawn(move || {
				observer
					.record_terminal_usage(Some(&usage))
					.expect("record terminal usage");
			}));
		}
		for writer in writers {
			writer.join().expect("usage writer");
		}
		let aggregate = store
			.client_usage(0)
			.expect("read client usage")
			.pop()
			.expect("aggregate");
		assert_eq!(aggregate.input_tokens, 88);
		assert_eq!(aggregate.output_tokens, 56);
		assert_eq!(aggregate.cache_read_tokens, 40);
		assert_eq!(aggregate.cache_write_tokens, 24);
		assert_eq!(aggregate.nanos_usd, 9_876_536);
	}

	#[test]
	fn observer_persists_quota_snapshot_and_history() {
		let (_directory, store, service) = harness(json!({}));
		let credential = store
			.upsert_api_key("provider", "", b"secret", 999)
			.expect("insert credential");
		let report = UsageReport {
			credential_id: credential.id,
			provider:      "provider".into(),
			plan:          "pro".into(),
			windows:       SmallVec::from_vec(vec![UsageWindow {
				label:        "primary".into(),
				used_percent: 37.5,
				resets_at_ms: 9_000,
			}]),
			fetched_at_ms: 1_000,
			detail:        json!({ "credits": 12 }),
		};
		service
			.observer()
			.record_quota_report(&report, 1_001)
			.expect("persist quota report");
		assert_eq!(
			store
				.read_usage_reports(None, Some(credential.id))
				.expect("latest"),
			[report.clone()]
		);
		assert_eq!(store.usage_history(credential.id, 0, 0).expect("history"), [UsageHistoryEntry {
			at_ms: 1_001,
			report
		}]);
	}

	#[test]
	fn observer_blocks_only_the_selected_credential() {
		let (_directory, store, service) = harness(json!({}));
		let selected = store
			.upsert_api_key("openai", "selected", b"selected", 1)
			.expect("selected credential");
		let sibling = store
			.upsert_api_key("openai", "sibling", b"sibling", 2)
			.expect("sibling credential");
		assert_ne!(selected.id, sibling.id);
		let lease = store
			.lease(selected.id)
			.expect("lease lookup")
			.expect("selected lease");
		let observation = omp_llm_egress::limits::CredentialBlock {
			key:           lease.egress_key(),
			blocked_until: SystemTime::now() + Duration::from_secs(60),
			status:        StatusCode::FORBIDDEN,
		};
		omp_llm_egress::limits::BlockSink::record_block(&service.observer(), &observation);
		let observed_at_ms = now_ms();
		let selected = store
			.get_credential(selected.id, observed_at_ms)
			.expect("selected")
			.expect("meta");
		let sibling = store
			.get_credential(sibling.id, observed_at_ms)
			.expect("sibling")
			.expect("meta");
		assert_eq!(selected.blocks.len(), 1);
		assert!(sibling.blocks.is_empty());
	}

	#[test]
	fn selected_terminal_cost_is_idempotent_windowed_and_account_isolated() {
		let (_directory, store, service) = harness(json!({}));
		let account_a = store
			.upsert_api_key("opencode-go", "account-a", b"a", 1)
			.expect("account a");
		let account_b = store
			.upsert_api_key("opencode-go", "account-b", b"b", 2)
			.expect("account b");
		let lease_a = store
			.lease(account_a.id)
			.expect("lease a")
			.expect("active a");
		let lease_b = store
			.lease(account_b.id)
			.expect("lease b")
			.expect("active b");
		let usage = |client: &str, nanos_usd, at_ms| ClientUsage {
			client_id: client.into(),
			label: client.into(),
			input_tokens: 10,
			output_tokens: 5,
			cache_read_tokens: 2,
			cache_write_tokens: 1,
			nanos_usd,
			last_seen_ms: at_ms,
		};
		let observer = service.observer();
		assert!(
			observer
				.record_terminal_observation(
					&lease_a,
					"turn-old",
					"model-a",
					"",
					Some(330_000),
					Some(&usage("client-a", 1_000_000_000, 100)),
					100,
				)
				.expect("old observation")
		);
		assert!(
			observer
				.record_terminal_observation(
					&lease_a,
					"turn-recent",
					"model-a",
					"",
					Some(330_000),
					Some(&usage("client-a", 2_000_000_000, 1_000)),
					1_000,
				)
				.expect("recent observation")
		);
		assert!(
			!observer
				.record_terminal_observation(
					&lease_a,
					"turn-recent",
					"model-a",
					"",
					Some(330_000),
					Some(&usage("client-a", 9_000_000_000, 1_001)),
					1_001,
				)
				.expect("duplicate observation")
		);
		assert!(
			!observer
				.record_terminal_observation(
					&lease_b,
					"turn-recent",
					"model-b",
					"",
					Some(3_000_000),
					Some(&usage("client-b", 9_000_000_000, 1_050)),
					1_050,
				)
				.expect("cross-credential duplicate observation")
		);
		assert!(
			observer
				.record_terminal_observation(
					&lease_b,
					"turn-b",
					"model-b",
					"",
					Some(3_000_000),
					Some(&usage("client-b", 4_000_000_000, 1_100)),
					1_100,
				)
				.expect("account b observation")
		);
		assert!(
			!observer
				.record_terminal_observation(
					&lease_a,
					"turn-cancelled",
					"model-a",
					"",
					Some(330_000),
					None,
					1_200,
				)
				.expect("cancelled observation")
		);

		let all_a = store
			.rolling_spend("opencode-go", None, Some("account-a"), 0, 0)
			.expect("account a spend");
		assert_eq!(all_a.nanos_usd, 3_000_000_000);
		let recent_a = store
			.rolling_spend("opencode-go", Some(account_a.id), None, 500, 1_500)
			.expect("recent account a spend");
		assert_eq!(recent_a.nanos_usd, 2_000_000_000);
		assert_eq!(recent_a.first_observed_at_ms, Some(1_000));
		let all_b = store
			.rolling_spend("opencode-go", None, Some("account-b"), 0, 0)
			.expect("account b spend");
		assert_eq!(
			store
				.premium_consumption("opencode-go", "model-a", account_a.id, Some("account-a"), 0, 0,)
				.expect("model a premium consumption")
				.millionths,
			660_000
		);
		assert_eq!(
			store
				.premium_consumption("opencode-go", "model-b", account_b.id, Some("account-b"), 0, 0,)
				.expect("model b premium consumption")
				.millionths,
			3_000_000
		);
		assert_eq!(
			store
				.premium_consumption("opencode-go", "model-a", account_b.id, Some("account-b"), 0, 0,)
				.expect("isolated premium consumption")
				.millionths,
			0
		);
		assert_eq!(all_b.nanos_usd, 4_000_000_000);
		let clients = store.client_usage(0).expect("client aggregates");
		assert_eq!(
			clients
				.iter()
				.find(|usage| usage.client_id == "client-a")
				.expect("client a")
				.nanos_usd,
			3_000_000_000
		);
	}

	#[test]
	fn copilot_premium_semantics_are_exact_and_idempotent() {
		let (_directory, store, service) = harness(json!({}));
		let credential = |provider: &str, account: &str| {
			store
				.upsert_api_key(provider, account, account.as_bytes(), 1)
				.expect("credential")
		};
		let paid = credential("github-copilot", "paid");
		let missing = credential("github-copilot", "missing");
		let agent = credential("github-copilot", "agent");
		let free = credential("github-copilot", "free");
		let other = credential("opencode-go", "other");
		store
			.write_usage_report(
				&UsageReport {
					credential_id: free.id,
					provider:      "github-copilot".into(),
					plan:          "free".into(),
					windows:       SmallVec::new(),
					fetched_at_ms: 90,
					detail:        json!({}),
				},
				90,
			)
			.expect("cached free plan");
		let lease = |id| store.lease(id).expect("lease query").expect("lease");
		let usage = ClientUsage {
			client_id:          "client".into(),
			label:              "client".into(),
			input_tokens:       1,
			output_tokens:      1,
			cache_read_tokens:  0,
			cache_write_tokens: 0,
			nanos_usd:          0,
			last_seen_ms:       100,
		};
		let observer = service.observer();
		assert!(
			observer
				.record_terminal_observation(
					&lease(paid.id),
					"paid-turn",
					"paid-model",
					"user",
					Some(330_000),
					Some(&usage),
					100,
				)
				.expect("paid observation")
		);
		assert!(
			!observer
				.record_terminal_observation(
					&lease(paid.id),
					"paid-turn",
					"paid-model",
					"user",
					Some(3_000_000),
					Some(&usage),
					101,
				)
				.expect("idempotent duplicate")
		);
		observer
			.record_terminal_observation(
				&lease(missing.id),
				"missing-turn",
				"missing-model",
				"user",
				None,
				Some(&usage),
				102,
			)
			.expect("missing multiplier observation");
		observer
			.record_terminal_observation(
				&lease(agent.id),
				"agent-turn",
				"agent-model",
				"agent",
				Some(3_000_000),
				Some(&usage),
				103,
			)
			.expect("agent observation");
		observer
			.record_terminal_observation(
				&lease(free.id),
				"free-turn",
				"free-model",
				"user",
				Some(330_000),
				Some(&usage),
				104,
			)
			.expect("free observation");
		observer
			.record_terminal_observation(
				&lease(other.id),
				"other-turn",
				"other-model",
				"user",
				None,
				Some(&usage),
				105,
			)
			.expect("other provider observation");

		let consumed = |provider: &str, model: &str, id: u64, account: &str| {
			store
				.premium_consumption(provider, model, id, Some(account), 0, 0)
				.expect("premium consumption")
				.millionths
		};
		assert_eq!(consumed("github-copilot", "paid-model", paid.id, "paid"), 330_000);
		assert_eq!(consumed("github-copilot", "missing-model", missing.id, "missing"), 1_000_000);
		assert_eq!(consumed("github-copilot", "agent-model", agent.id, "agent"), 0);
		assert_eq!(consumed("github-copilot", "free-model", free.id, "free"), 1_000_000);
		assert_eq!(consumed("opencode-go", "other-model", other.id, "other"), 0);
	}
	#[test]
	fn cancelled_request_omits_terminal_usage() {
		let (_directory, store, service) = harness(json!({}));
		service
			.observer()
			.record_terminal_usage(None)
			.expect("cancellation is a no-op");
		assert!(store.client_usage(0).expect("client usage").is_empty());
	}
}
