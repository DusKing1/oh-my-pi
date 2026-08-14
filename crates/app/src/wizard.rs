//! Retained first-run setup flow for interactive chat.

use std::{path::Path, time::Duration};

use miette::{IntoDiagnostic as _, miette};
use omp_core::{Str, fmts};
use omp_llm_catalog::{ProviderId, snapshot::Catalog};
use omp_llm_inference::{
	Client, Registry as InferenceRegistry,
	answer::{AccountState, AuthAnswer},
	call::{AuthInput, AuthRequest, CallMeta, Target},
	id::RequestId,
	receipt::ExecutionBudget,
	router::Router,
};
use omp_tui::{
	AppEvent, AppOptions, Border, Dim, OverlayAnchor, OverlayMargin, OverlayOptions, Prop, Size, Ui,
	components::{Boxed, Button, Col, Markdown, Shader, TextLeaf},
	shader::Eclipse,
};

use crate::{
	chat::ChatAuthWorker,
	chat_ui::{
		AuthPromptKind, ChatAuthEvent, auth_input,
		login::{PROVIDER_SELECT_ID, show_provider_picker},
		models::{MODEL_SELECT_ID, show_model_picker},
		show_auth_prompt,
	},
	settings::Settings,
};

const STATUS_ID: &str = "wizard-status";
const CONTINUE_ID: &str = "wizard-continue";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Step {
	Welcome,
	Provider,
	Authenticating,
	Prompt(AuthPromptKind),
	Model,
}

/// Runs first-run setup and returns the persisted model selection.
///
/// A clean cancellation returns `None`; provider login and model selection are
/// completed inside this retained terminal host before it is dropped.
#[expect(clippy::future_not_send, reason = "the setup wizard owns a thread-confined omp_tui::App")]
pub async fn run(data_dir: &Path, catalog: &Catalog) -> miette::Result<Option<Str>> {
	std::fs::create_dir_all(data_dir).into_diagnostic()?;
	let store =
		crate::daemon::open_credential_store(&data_dir.join("credentials.db")).into_diagnostic()?;
	let registry = crate::daemon::production_registry(data_dir, store)
		.await
		.into_diagnostic()?;
	let has_account = has_active_account(&registry).await?;
	let worker = ChatAuthWorker::start(registry);
	let auth = worker.ui();

	let mut app = AppOptions::new()
		.hold_alt()
		.keep_on_cancel()
		.start(|env| {
			Ui::from_root(
				Shader::new(Eclipse::default()).size(env.viewport.width, env.viewport.height),
				env.viewport.width,
				env.ctx,
			)
		})
		.await
		.into_diagnostic()?;
	show_welcome(app.ui_mut());
	let mut step = Step::Welcome;

	let selected = 'wizard: loop {
		tokio::select! {
			event = app.next() => match event.into_diagnostic()? {
				None => break 'wizard None,
				Some(AppEvent::Pressed(id))
					if id.as_str() == CONTINUE_ID && step == Step::Welcome =>
				{
					if has_account {
						show_model_picker(app.ui_mut(), catalog, "");
						step = Step::Model;
					} else {
						show_provider_picker(app.ui_mut(), catalog);
						step = Step::Provider;
					}
				},
				Some(AppEvent::Submitted) => {
					if let Step::Prompt(kind) = step {
						let value = app.ui().values()["auth-secret"]
							.as_str()
							.unwrap_or("")
							.to_owned();
						let _ = app.ui_mut().close_top_overlay();
						if let Err(error) = auth.answer(auth_input(kind, value)) {
							set_status(app.ui_mut(), error);
						}
						step = Step::Authenticating;
					}
				},
				Some(AppEvent::Changed { id, value }) if id.as_str() == PROVIDER_SELECT_ID => {
					let _ = app.ui_mut().close_top_overlay();
					match auth.start(value.clone()) {
						Ok(()) => {
							set_status(
								app.ui_mut(),
								fmts!("Starting authentication for `{value}`…"),
							);
							step = Step::Authenticating;
						},
						Err(error) => {
							set_status(app.ui_mut(), fmts!("Setup error: {error}"));
							show_provider_picker(app.ui_mut(), catalog);
							step = Step::Provider;
						},
					}
				},
				Some(AppEvent::Changed { id, value }) if id.as_str() == MODEL_SELECT_ID => {
					Settings { default_model: Some(value.to_string()) }
						.save(data_dir)
						.into_diagnostic()?;
					break 'wizard Some(value);
				},
				Some(AppEvent::OverlayClosed(_)) => match step {
					Step::Welcome => break 'wizard None,
					Step::Prompt(_) => {
						let _ = auth.answer(AuthInput::Cancel);
						set_status(app.ui_mut(), "Cancelling provider authentication…");
						step = Step::Authenticating;
					},
					Step::Provider | Step::Model => step = Step::Welcome,
					Step::Authenticating => {},
				},
				Some(_) => {},
			},
			event = auth.next_event() => match event {
				Some(ChatAuthEvent::Url(url)) => {
					set_status(app.ui_mut(), fmts!("[Open to authorize]({url})"));
				},
				Some(ChatAuthEvent::DeviceCode { code, url }) => {
					set_status(app.ui_mut(), fmts!("Enter code `{code}` at [{url}]({url})"));
				},
				Some(ChatAuthEvent::Prompt { message, kind }) => {
					show_auth_prompt(app.ui_mut(), message, kind);
					step = Step::Prompt(kind);
				},
				Some(ChatAuthEvent::Notice(message)) => set_status(app.ui_mut(), message),
				Some(ChatAuthEvent::Complete(message)) => {
					set_status(app.ui_mut(), message);
					show_model_picker(app.ui_mut(), catalog, "");
					step = Step::Model;
				},
				Some(ChatAuthEvent::Failed(message)) => {
					if matches!(step, Step::Prompt(_)) {
						let _ = app.ui_mut().close_top_overlay();
					}
					set_status(app.ui_mut(), fmts!("Setup error: {message}"));
					show_provider_picker(app.ui_mut(), catalog);
					step = Step::Provider;
				},
				None => break 'wizard None,
			},
		}
	};

	worker.shutdown().await;
	drop(app);
	Ok(selected)
}

fn show_welcome(ui: &mut Ui) {
	let content = Col::new()
		.with(Prop::Gap, 1_u16)
		.child(
			TextLeaf::new()
				.with(Prop::Bold, true)
				.with(Prop::Align, "center")
				.text("oh my pi"),
		)
		.child(
			TextLeaf::new()
				.with(Prop::Align, "center")
				.text("A focused coding agent for this project."),
		)
		.child(Markdown::new().with(Prop::Id, STATUS_ID))
		.child(
			Button::new()
				.with(Prop::Id, CONTINUE_ID)
				.with(Prop::Align, "center")
				.child("Continue"),
		)
		.child(
			TextLeaf::new()
				.with(Prop::Dim, true)
				.with(Prop::Align, "center")
				.text("Enter continue · Ctrl+C quit"),
		);
	let card = Boxed::new()
		.with(Prop::Border, Border::Round)
		.with(Prop::PadX, 2_u16)
		.with(Prop::PadY, 1_u16)
		.child(content);
	ui.show_overlay(
		card,
		OverlayOptions::default()
			.anchor(OverlayAnchor::Center)
			.width(Dim::Pct(60))
			.min_width(40)
			.max_height(Dim::Pct(60))
			.margin(OverlayMargin::uniform(1))
			.min_viewport(Size::new(24, 6)),
	);
	ui.focus_first();
}

fn set_status(ui: &mut Ui, message: impl Into<Str>) {
	ui.set_text(STATUS_ID, message.into());
}

async fn has_active_account(registry: &InferenceRegistry) -> miette::Result<bool> {
	let provider = registry
		.catalog()
		.providers()
		.first()
		.map(|provider| ProviderId::from(provider.id.as_str()))
		.ok_or_else(|| miette!("embedded catalog has no providers"))?;
	let planner = Router::new(registry.clone(), Duration::from_secs(30));
	let meta = CallMeta {
		id:       RequestId::from(format!("wizard-auth-{}", ulid::Ulid::generate())),
		target:   Target::ProviderService(provider),
		deadline: None,
		budget:   ExecutionBudget::default(),
		session:  None,
	};
	let mut client = Client::new(registry.service(), planner, meta);
	let answer = client
		.execute(AuthRequest::ListAccounts { provider: None })
		.await
		.into_diagnostic()?;
	let AuthAnswer::Accounts(accounts) = answer else {
		return Err(miette!("account listing returned an unexpected response"));
	};
	Ok(accounts
		.iter()
		.any(|account| account.state == AccountState::Active))
}
