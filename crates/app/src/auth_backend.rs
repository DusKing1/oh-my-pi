//! Typed authentication and encrypted credential-store construction.

use std::path::PathBuf;

use omp_llm_catalog::ProviderId;
use omp_llm_inference::{
	Client,
	answer::{AuthAnswer, AuthEvent},
	call::{AuthRequest, CallMeta, LoginRequest, Target},
	id::{AccountId, RequestId},
	receipt::ExecutionBudget,
};

use crate::{cli::AuthCommand, error::AppError};

/// Opens encrypted credential state and executes one typed authentication
/// operation.
pub async fn run(database: PathBuf, command: AuthCommand) -> crate::Result<()> {
	let data_dir = database
		.parent()
		.filter(|path| !path.as_os_str().is_empty())
		.ok_or(AppError::DataDirNotConfigured)?;
	std::fs::create_dir_all(data_dir)?;
	let store = crate::daemon::open_credential_store(&database)?;
	let registry = crate::daemon::production_registry(data_dir, store).await?;
	let default_provider = registry
		.catalog()
		.providers()
		.first()
		.map(|provider| provider.id.clone())
		.ok_or(AppError::CatalogSnapshot)?;
	let (provider, operation) = match command {
		AuthCommand::Login { provider } => {
			let provider = ProviderId::from(provider);
			(provider.clone(), AuthRequest::Login(LoginRequest { provider, method: None }))
		},
		AuthCommand::List { provider } => {
			let provider = provider.map(ProviderId::from).unwrap_or(default_provider);
			(provider.clone(), AuthRequest::ListAccounts { provider: Some(provider) })
		},
		AuthCommand::Refresh { account } => {
			(default_provider.clone(), AuthRequest::Refresh { account: AccountId::from(account) })
		},
		AuthCommand::Logout { account } => {
			(default_provider.clone(), AuthRequest::Logout { account: AccountId::from(account) })
		},
	};
	let meta = CallMeta {
		id:       RequestId::from("omp-auth-cli"),
		target:   Target::ProviderService(provider),
		deadline: None,
		budget:   ExecutionBudget::default(),
		session:  None,
	};
	let planner =
		omp_llm_inference::router::Router::new(registry.clone(), std::time::Duration::from_secs(30));
	let mut client = Client::new(registry.service(), planner, meta);
	print_auth(client.execute(operation).await?).await
}

async fn print_auth(answer: AuthAnswer) -> crate::Result<()> {
	match answer {
		AuthAnswer::Session(session) => {
			while let Ok(event) = session.events.recv_async().await {
				match event? {
					AuthEvent::OpenUrl(url) => println!("open {url}"),
					AuthEvent::ShowDeviceCode { verification_url, .. } => {
						println!("complete device authorization at {verification_url}")
					},
					AuthEvent::Prompt(prompt) => println!("{}", prompt.message),
					AuthEvent::Waiting => println!("waiting for provider authorization"),
					AuthEvent::Complete(account) => {
						println!("{} {}", account.account, account.provider);
						break;
					},
				}
			}
		},
		AuthAnswer::Accounts(accounts) => {
			for account in accounts {
				println!("{} {}", account.account, account.provider);
			}
		},
		AuthAnswer::Refreshed(account) => println!("{} {}", account.account, account.provider),
		AuthAnswer::LoggedOut(account) => println!("{account}"),
		AuthAnswer::Submitted(session) => println!("{session}"),
	}
	Ok(())
}
