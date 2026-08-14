//! Typed authentication and encrypted credential-store construction.

use std::{
	io::{self, IsTerminal as _, Write as _},
	path::PathBuf,
};

use miette::{IntoDiagnostic as _, Result, miette};
use nix::sys::termios::{LocalFlags, SetArg, tcgetattr, tcsetattr};
use omp_core::Str;
use omp_llm_catalog::ProviderId;
use omp_llm_inference::{
	Client,
	answer::{AuthAnswer, AuthEvent, AuthPrompt, AuthPromptKind, AuthResponse},
	call::{AuthInput, AuthRequest, CallMeta, LoginRequest, Target},
	id::{AccountId, RequestId},
	receipt::ExecutionBudget,
};
use secrecy::{ExposeSecret as _, SecretString};
use zeroize::Zeroizing;

use crate::cli::AuthCommand;

/// Opens encrypted credential state and executes one typed authentication
/// operation.
pub async fn run(database: PathBuf, command: AuthCommand) -> Result<()> {
	let data_dir = database
		.parent()
		.filter(|path| !path.as_os_str().is_empty())
		.ok_or_else(|| miette!("HOME or OMP_DATA_DIR must be set"))?;
	std::fs::create_dir_all(data_dir).into_diagnostic()?;
	let store = crate::daemon::open_credential_store(&database).into_diagnostic()?;
	let registry = crate::daemon::production_registry(data_dir, store)
		.await
		.into_diagnostic()?;
	let default_provider = registry
		.catalog()
		.providers()
		.first()
		.map(|provider| provider.id.clone())
		.ok_or_else(|| miette!("embedded catalog is unavailable"))?;
	let (provider, operation) = match command {
		AuthCommand::Login { provider } => {
			let provider = ProviderId::from(provider);
			(provider.clone(), AuthRequest::Login(LoginRequest { provider, method: None }))
		},
		AuthCommand::List { provider } => {
			let provider = provider.map_or(default_provider, ProviderId::from);
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
	print_auth(client.execute(operation).await.into_diagnostic()?).await
}

async fn print_auth(answer: AuthAnswer) -> Result<()> {
	match answer {
		AuthAnswer::Session(session) => {
			let session_id = session.id.clone();
			while let Ok(event) = session.events.recv_async().await {
				match event.into_diagnostic()? {
					AuthEvent::OpenUrl(url) => println!("open {url}"),
					AuthEvent::ShowDeviceCode { code, verification_url } => {
						println!(
							"complete device authorization at {verification_url} using code {}",
							code.expose_secret()
						);
					},
					AuthEvent::Prompt(prompt) => {
						let input = tokio::task::spawn_blocking(move || read_prompt(&prompt))
							.await
							.into_diagnostic()??;
						session
							.responses
							.send_async(AuthResponse { session: session_id.clone(), input })
							.await
							.into_diagnostic()?;
					},
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

fn read_prompt(prompt: &AuthPrompt) -> Result<AuthInput> {
	let mut stdout = io::stdout().lock();
	write!(stdout, "{}: ", prompt.message).into_diagnostic()?;
	stdout.flush().into_diagnostic()?;
	drop(stdout);

	let stdin = io::stdin();
	let hide_input =
		!matches!(prompt.input, AuthPromptKind::Confirmation | AuthPromptKind::PlainText)
			&& stdin.is_terminal();
	let original = if hide_input {
		let original = tcgetattr(&stdin).into_diagnostic()?;
		let mut hidden = original.clone();
		hidden.local_flags.remove(LocalFlags::ECHO);
		tcsetattr(&stdin, SetArg::TCSANOW, &hidden).into_diagnostic()?;
		Some(original)
	} else {
		None
	};
	let mut value = Zeroizing::new(String::new());
	let read = stdin.read_line(&mut value);
	if let Some(original) = original {
		tcsetattr(&stdin, SetArg::TCSANOW, &original).into_diagnostic()?;
		println!();
	}
	if read.into_diagnostic()? == 0 {
		return Err(miette!("authentication input closed"));
	}
	auth_input(prompt, value.trim())
}

fn auth_input(prompt: &AuthPrompt, value: &str) -> Result<AuthInput> {
	match prompt.input {
		AuthPromptKind::Confirmation => Ok(AuthInput::DeviceConfirmed),
		AuthPromptKind::PlainText => Ok(AuthInput::PlainText(Str::from(value))),
		AuthPromptKind::OptionalSecret => {
			Ok(AuthInput::OptionalSecret(SecretString::from(value.to_owned())))
		},
		AuthPromptKind::AuthorizationCode if prompt.id == "oauth-callback-url" => {
			if value.is_empty() {
				Err(miette!("authentication input must not be empty"))
			} else {
				Ok(AuthInput::CallbackUrl(SecretString::from(value.to_owned())))
			}
		},
		AuthPromptKind::AuthorizationCode | AuthPromptKind::ApiKey | AuthPromptKind::SessionToken => {
			if value.is_empty() {
				return Err(miette!("authentication input must not be empty"));
			}
			let value = SecretString::from(value.to_owned());
			Ok(match prompt.input {
				AuthPromptKind::AuthorizationCode => AuthInput::AuthorizationCode(value),
				AuthPromptKind::ApiKey => AuthInput::ApiKey(value),
				AuthPromptKind::SessionToken => AuthInput::SessionToken(value),
				_ => unreachable!("matched required secret prompt"),
			})
		},
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn callback_prompt_submits_the_complete_url() {
		let prompt = AuthPrompt {
			id:      "oauth-callback-url".into(),
			message: "callback".into(),
			input:   AuthPromptKind::AuthorizationCode,
		};
		let AuthInput::CallbackUrl(value) =
			auth_input(&prompt, "http://localhost/callback?code=abc&state=xyz").unwrap()
		else {
			panic!("callback prompt must preserve the complete URL");
		};
		assert_eq!(value.expose_secret(), "http://localhost/callback?code=abc&state=xyz");
	}

	#[test]
	fn confirmation_accepts_an_empty_line() {
		let prompt = AuthPrompt {
			id:      "confirm".into(),
			message: "continue".into(),
			input:   AuthPromptKind::Confirmation,
		};
		assert!(matches!(auth_input(&prompt, "").unwrap(), AuthInput::DeviceConfirmed));
	}

	#[test]
	fn plain_text_is_visible_and_may_select_the_empty_default() {
		let prompt = AuthPrompt {
			id:      "region".into(),
			message: "region".into(),
			input:   AuthPromptKind::PlainText,
		};
		assert!(
			matches!(auth_input(&prompt, "").unwrap(), AuthInput::PlainText(value) if value.is_empty())
		);
	}

	#[test]
	fn optional_secret_accepts_empty_as_skip() {
		let prompt = AuthPrompt {
			id:      "cookie".into(),
			message: "cookie".into(),
			input:   AuthPromptKind::OptionalSecret,
		};
		assert!(matches!(
			auth_input(&prompt, "").unwrap(),
			AuthInput::OptionalSecret(value) if value.expose_secret().is_empty()
		));
	}
}
