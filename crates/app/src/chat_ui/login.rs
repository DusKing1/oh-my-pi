use omp_llm_catalog::{ProviderDef, snapshot::Catalog};
use omp_tui::{
	Border, Dim, OverlayAnchor, OverlayMargin, OverlayOptions, Prop, Size, Ui,
	components::{Boxed, Col, Select, SelectOption, TextLeaf},
};

use super::AuthPromptKind;

/// Returns whether the login prompt must suppress terminal echo.
pub(crate) const fn prompt_masks_input(kind: AuthPromptKind) -> bool {
	!matches!(kind, AuthPromptKind::Confirmation | AuthPromptKind::PlainText)
}

pub(crate) const PROVIDER_SELECT_ID: &str = "login-provider";

/// Opens the provider picker without preselecting a provider.
pub(crate) fn show_provider_picker(ui: &mut Ui, catalog: &Catalog) {
	show_provider_picker_for(ui, catalog, None);
}

/// Opens the provider picker with the current model's provider highlighted.
pub(crate) fn show_provider_picker_for(ui: &mut Ui, catalog: &Catalog, current: Option<&str>) {
	let mut providers = catalog
		.providers()
		.iter()
		.map(|provider| (provider, provider_uses_oauth(catalog, provider)))
		.collect::<Vec<_>>();
	providers.sort_by_key(|(_, oauth)| !*oauth);

	let rows = u16::try_from(providers.len())
		.unwrap_or(u16::MAX)
		.min(12)
		.saturating_add(1);
	let mut select = Select::new()
		.with(Prop::Id, PROVIDER_SELECT_ID)
		.with(Prop::Filter, true)
		.with(Prop::H, rows);
	for (provider, oauth) in providers {
		select = select.option(
			SelectOption::new()
				.with(Prop::Value, provider.id.as_str())
				.with(Prop::Desc, if oauth { "OAuth" } else { "API key" })
				.with(Prop::Recommended, current == Some(provider.id.as_str()))
				.label(provider.name.clone()),
		);
	}

	let content = Col::new().child(select).child(
		TextLeaf::new()
			.with(Prop::Dim, true)
			.text("Type to filter · Enter select · Esc cancel"),
	);
	let picker = Boxed::new()
		.with(Prop::Border, Border::Round)
		.with(Prop::Title, "Provider Login")
		.with(Prop::PadX, 1_u16)
		.child(content);
	ui.show_overlay(
		picker,
		OverlayOptions::default()
			.anchor(OverlayAnchor::Center)
			.width(Dim::Pct(70))
			.min_width(40)
			.max_height(Dim::Pct(75))
			.margin(OverlayMargin::uniform(1))
			.min_viewport(Size::new(24, 6)),
	);
}

fn provider_uses_oauth(catalog: &Catalog, provider: &ProviderDef) -> bool {
	provider.auth.iter().any(|auth_id| {
		catalog
			.auth_spec(auth_id)
			.and_then(|auth| auth.oauth.as_ref())
			.is_some_and(|oauth_id| catalog.oauth_spec(oauth_id).is_some())
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn plain_text_is_visible_and_optional_secrets_are_masked() {
		assert!(!prompt_masks_input(AuthPromptKind::PlainText));
		assert!(prompt_masks_input(AuthPromptKind::OptionalSecret));
	}
}
