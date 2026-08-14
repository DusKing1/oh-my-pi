//! Retained first-run setup flow for interactive chat.

use std::{path::Path, time::Duration};

use miette::{IntoDiagnostic as _, miette};
use omp_chat_ui::provider_picker::{ProviderCard, provider_card_grid};
use omp_core::{Str, fmts};
use omp_llm_catalog::{ProviderDef, ProviderId, provider::AuthSpecKind, snapshot::Catalog};
use omp_llm_inference::{
	Client, Registry as InferenceRegistry,
	answer::{AccountState, AuthAnswer},
	call::{AuthRequest, CallMeta, Target},
	id::RequestId,
	receipt::ExecutionBudget,
	router::Router,
};
use omp_tui::{
	AppEvent, AppOptions, Border, Dim, Key, OverlayAnchor, OverlayMargin, OverlayOptions, Prop,
	Size, Ui,
	components::{Boxed, Button, Col, Input, Markdown, Select, SelectOption, Shader, TextLeaf},
	shader::Eclipse,
};

use crate::{
	chat::ChatAuthWorker,
	chat_ui::{
		AuthPromptKind, CREDENTIAL_STORAGE_LOCKED_MESSAGE, ChatAuthEvent, auth_input,
		prompt_masks_input,
	},
	settings::Settings,
};

const STATUS_ID: &str = "wizard-status";
/// Card-id namespace shared with the setup provider grid.
const PROVIDER_CARD_PREFIX: &str = "login-provider:";
const MODEL_SELECT_ID: &str = "model-picker";
const CONTINUE_ID: &str = "wizard-continue";

/// Setup scenes and their only legal exits.
///
/// `Welcome` continues to `Provider` (or `Model` for an existing account).
/// `Provider` starts `Authenticating`, while cancellation returns to `Welcome`.
/// `Authenticating` may open `Prompt`, complete into `Model`, fail back to
/// `Provider`, or stop at the blocking `CredentialStorageLocked` state.
/// `Prompt` submits back to `Authenticating` and cancellation/failure returns
/// to `Provider`. `CredentialStorageLocked` must be dismissed before returning
/// to `Welcome`. `Model` completes the wizard or cancels back to `Welcome`.
/// Every variant therefore owns a visible, keyboard-usable scene.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Step {
	Welcome,
	Provider,
	Authenticating,
	CredentialStorageLocked,
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
		crate::daemon::open_credential_store(data_dir.join("credentials.db")).into_diagnostic()?;
	let registry = crate::daemon::production_registry(data_dir, store)
		.await
		.into_diagnostic()?;
	let has_account = has_active_account(&registry).await?;
	let worker = ChatAuthWorker::start(registry.clone());
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
					let _ = app.ui_mut().close_top_overlay();
					if has_account {
						open_setup_model_step(app.ui_mut(), catalog, "");
						step = Step::Model;
					} else {
						open_setup_provider_step(app.ui_mut(), catalog);
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
						match auth.answer(auth_input(kind, value)) {
							Ok(()) => {
								set_status(app.ui_mut(), "Authenticating… Esc to cancel");
								step = Step::Authenticating;
							},
							Err(error) => {
								let _ = app.ui_mut().close_top_overlay();
								show_welcome(app.ui_mut());
								set_status(app.ui_mut(), fmts!("Setup error: {error}"));
								open_setup_provider_step(app.ui_mut(), catalog);
								step = Step::Provider;
							},
						}
					}
				},
				Some(AppEvent::Key(key)) if step == Step::Provider => {
					if key == omp_tui::Key::Esc {
								  let _ = app.ui_mut().close_top_overlay();
								  if has_active_account(&registry).await.unwrap_or(false) {
									  break 'wizard None;
								  }
												step = Step::Welcome;
							  }
				},
				Some(AppEvent::Pressed(id))
					if step == Step::Provider
						&& id.as_str().starts_with(PROVIDER_CARD_PREFIX) =>
				{
					let value = Str::from(
						id.as_str()
							.strip_prefix(PROVIDER_CARD_PREFIX)
							.expect("guarded above"),
					);
					let _ = app.ui_mut().close_top_overlay();
					match auth.start(value.clone()) {
						Ok(()) => {
							show_authenticating(app.ui_mut());
							set_status(
								app.ui_mut(),
								fmts!("Authenticating `{value}`… Esc to cancel"),
							);
							step = Step::Authenticating;
						},
						Err(error) => {
							show_welcome(app.ui_mut());
							set_status(app.ui_mut(), fmts!("Setup error: {error}"));
							open_setup_provider_step(app.ui_mut(), catalog);
							step = Step::Provider;
						},
					}
				},
				Some(AppEvent::Changed { id, value })
					if id.as_str() == MODEL_SELECT_ID && step == Step::Model =>
				{
					Settings { default_model: Some(value.to_string()) }
						.save(data_dir)
						.into_diagnostic()?;
					break 'wizard Some(value);
				},
				Some(AppEvent::OverlayClosed(_)) => match step {
					Step::Welcome => break 'wizard None,
					Step::Prompt(_) => {
						let _ = auth.cancel();
						let _ = app.ui_mut().close_top_overlay();
						show_welcome(app.ui_mut());
						set_status(app.ui_mut(), "Authentication cancelled. Choose a provider.");
						open_setup_provider_step(app.ui_mut(), catalog);
						step = Step::Provider;
					},
					Step::Provider | Step::Model => {
						show_welcome(app.ui_mut());
						step = Step::Welcome;
					},
					Step::Authenticating => {
						let _ = auth.cancel();
						show_welcome(app.ui_mut());
						set_status(app.ui_mut(), "Authentication cancelled. Choose a provider.");
						open_setup_provider_step(app.ui_mut(), catalog);
						step = Step::Provider;
					},
					Step::CredentialStorageLocked => {
						show_welcome(app.ui_mut());
						step = Step::Welcome;
					},
				},
				Some(AppEvent::Key(Key::Esc)) if step == Step::Authenticating => {
					let _ = auth.cancel();
					let _ = app.ui_mut().close_top_overlay();
					show_welcome(app.ui_mut());
					set_status(app.ui_mut(), "Authentication cancelled. Choose a provider.");
					open_setup_provider_step(app.ui_mut(), catalog);
					step = Step::Provider;
				},
				Some(AppEvent::Key(Key::Esc)) if step == Step::CredentialStorageLocked => {
					let _ = app.ui_mut().close_top_overlay();
					show_welcome(app.ui_mut());
					step = Step::Welcome;
				},
				Some(_) => {},
			},
			event = auth.next_event() => match event {
				Some(ChatAuthEvent::Url(url))
					if matches!(step, Step::Authenticating | Step::Prompt(_)) =>
				{
					set_status(
						app.ui_mut(),
						fmts!("[Open to authorize]({url}) · Esc to cancel"),
					);
				},
				Some(ChatAuthEvent::DeviceCode { code, url })
					if matches!(step, Step::Authenticating | Step::Prompt(_)) =>
				{
					set_status(
						app.ui_mut(),
						fmts!("Enter code `{code}` at [{url}]({url}) · Esc to cancel"),
					);
				},
				Some(ChatAuthEvent::Prompt { message, kind })
					if step == Step::Authenticating =>
				{
					show_auth_prompt(app.ui_mut(), message, kind);
					step = Step::Prompt(kind);
				},
				Some(ChatAuthEvent::Notice(message))
					if matches!(step, Step::Authenticating | Step::Prompt(_)) =>
				{
					set_status(app.ui_mut(), fmts!("{message} · Esc to cancel"));
				},
				Some(ChatAuthEvent::Complete(_))
					if matches!(step, Step::Authenticating | Step::Prompt(_)) =>
				{
					close_auth_scene(app.ui_mut(), step);
					open_setup_model_step(app.ui_mut(), catalog, "");
					step = Step::Model;
				},
				Some(ChatAuthEvent::CredentialStorageLocked)
					if matches!(step, Step::Authenticating | Step::Prompt(_)) =>
				{
					if matches!(step, Step::Prompt(_)) {
						let _ = app.ui_mut().close_top_overlay();
					}
					set_status(app.ui_mut(), CREDENTIAL_STORAGE_LOCKED_MESSAGE);
					step = Step::CredentialStorageLocked;
				},
				Some(ChatAuthEvent::Failed(message)) => {
					if matches!(step, Step::Authenticating | Step::Prompt(_)) {
						close_auth_scene(app.ui_mut(), step);
						show_welcome(app.ui_mut());
						set_status(app.ui_mut(), fmts!("Setup error: {message}"));
						open_setup_provider_step(app.ui_mut(), catalog);
						step = Step::Provider;
					} else {
						set_status(app.ui_mut(), fmts!("Setup error: {message}"));
					}
				},
				Some(_) => {},
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
	let scene = Col::new().with(Prop::Gap, 1_u16).child(card).child(
		Markdown::new()
			.with(Prop::Id, STATUS_ID)
			.with(Prop::Align, "center")
			.text(" "),
	);
	show_scene(ui, scene);
	ui.focus_first();
}

fn show_authenticating(ui: &mut Ui) {
	let content = Col::new()
		.with(Prop::Gap, 1_u16)
		.child(
			TextLeaf::new()
				.with(Prop::Bold, true)
				.with(Prop::Align, "center")
				.text("Provider authentication"),
		)
		.child(
			Markdown::new()
				.with(Prop::Id, STATUS_ID)
				.with(Prop::Align, "center")
				.text("Authenticating… Esc to cancel"),
		);
	let card = Boxed::new()
		.with(Prop::Border, Border::Round)
		.with(Prop::PadX, 2_u16)
		.with(Prop::PadY, 1_u16)
		.child(content);
	show_scene(ui, card);
}

fn open_setup_provider_step(ui: &mut Ui, catalog: &Catalog) {
	let mut providers = catalog
		.providers()
		.iter()
		.filter(|provider| provider_supports_login(catalog, provider))
		.map(|provider| (provider, provider_uses_oauth(catalog, provider)))
		.collect::<Vec<_>>();
	providers.sort_by_key(|(_, oauth)| !*oauth);
	let count = providers.len();
	let cards: Vec<ProviderCard> = providers
		.into_iter()
		.map(|(provider, _)| ProviderCard {
			press_id:    fmts!("{PROVIDER_CARD_PREFIX}{}", provider.id),
			provider_id: Str::from(provider.id.as_str()),
			label:       provider.name.clone(),
		})
		.collect();
	let counter = fmts!("{count} providers");
	let picker = Boxed::new()
		.with(Prop::Border, Border::Round)
		.with(Prop::Title, "Provider Login")
		.with(Prop::PadX, 1_u16)
		.child(provider_card_grid(cards, counter, "↹/←→/↑↓ pick · ↵ login · Esc back", 18));
	show_scene(ui, picker);
}

fn open_setup_model_step(ui: &mut Ui, catalog: &Catalog, current: &str) {
	let mut select = Select::new()
		.with(Prop::Id, MODEL_SELECT_ID)
		.with(Prop::Filter, true)
		.with(
			Prop::H,
			u16::try_from(catalog.models().len())
				.unwrap_or(u16::MAX)
				.min(12),
		);
	for model in catalog.models() {
		let key = model.key.to_string();
		let label = if key == current {
			format!("{key} (current)")
		} else {
			key.clone()
		};
		select = select.option(
			SelectOption::new()
				.with(Prop::Value, key)
				.label(label)
				.with_str(Prop::Desc, model.display_name.as_str()),
		);
	}
	let picker = Boxed::new()
		.with(Prop::Border, Border::Round)
		.with(Prop::Title, "Choose Model")
		.with(Prop::PadX, 1_u16)
		.child(
			Col::new().child(select).child(
				TextLeaf::new()
					.with(Prop::Dim, true)
					.text("Type to filter · Enter select · Esc cancel"),
			),
		);
	show_scene(ui, picker);
}

fn show_auth_prompt(ui: &mut Ui, message: Str, kind: AuthPromptKind) {
	let placeholder = match kind {
		AuthPromptKind::Confirmation => "Press Enter to confirm",
		AuthPromptKind::OptionalSecret => "Enter optional response or press Enter to skip",
		_ => "Enter provider response",
	};
	let prompt = Boxed::new()
		.with(Prop::Border, Border::Round)
		.with(Prop::Title, "Provider Authentication")
		.with(Prop::PadX, 1_u16)
		.child(
			Col::new()
				.with(Prop::Gap, 1_u16)
				.child(TextLeaf::new().text(message))
				.child(
					Input::new()
						.with(Prop::Id, "auth-secret")
						.with(Prop::Placeholder, placeholder)
						.with(Prop::Mask, prompt_masks_input(kind))
						.with(Prop::Submit, true),
				)
				.child(
					TextLeaf::new()
						.with(Prop::Dim, true)
						.text("Enter submit · Esc cancel"),
				),
		);
	show_scene(ui, prompt);
}

fn provider_supports_login(catalog: &Catalog, provider: &ProviderDef) -> bool {
	provider
		.auth
		.iter()
		.filter_map(|auth_id| catalog.auth_spec(auth_id))
		.any(|auth| auth.kind != AuthSpecKind::None)
}

fn provider_uses_oauth(catalog: &Catalog, provider: &ProviderDef) -> bool {
	provider.auth.iter().any(|auth_id| {
		catalog
			.auth_spec(auth_id)
			.and_then(|auth| auth.oauth.as_ref())
			.is_some_and(|oauth_id| catalog.oauth_spec(oauth_id).is_some())
	})
}

fn show_scene(ui: &mut Ui, scene: impl omp_tui::IntoComponent) {
	ui.show_overlay(
		scene,
		OverlayOptions::default()
			.anchor(OverlayAnchor::Center)
			.width(Dim::Pct(60))
			.min_width(40)
			.max_height(Dim::Pct(60))
			.margin(OverlayMargin::uniform(1))
			.min_viewport(Size::new(24, 6)),
	);
}

fn close_auth_scene(ui: &mut Ui, step: Step) {
	if matches!(step, Step::Prompt(_)) {
		let _ = ui.close_top_overlay();
	}
	let _ = ui.close_top_overlay();
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
