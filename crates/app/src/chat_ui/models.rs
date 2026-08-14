//! Catalog model picker for the interactive chat shell.

use omp_llm_catalog::{ModelSpec, Price, PriceUnit, snapshot::Catalog};
use omp_tui::{
	Border, Dim, OverlayAnchor, OverlayMargin, OverlayOptions, Prop, Size, Ui,
	components::{Boxed, Col, Select, SelectOption, TextLeaf},
};

/// DOM identifier emitted when a model is committed.
pub(crate) const MODEL_SELECT_ID: &str = "model-picker";

/// Opens the catalog model picker; commits surface as a changed event whose
/// value is the selected model key.
pub(crate) fn show_model_picker(ui: &mut Ui, catalog: &Catalog, current: &str) {
	let rows = u16::try_from(catalog.models().len())
		.unwrap_or(u16::MAX)
		.min(12)
		.saturating_add(1);
	let mut select = Select::new()
		.with(Prop::Id, MODEL_SELECT_ID)
		.with(Prop::Filter, true)
		.with(Prop::H, rows);
	for model in catalog.models() {
		let key = model.key.to_string();
		let label = if key == current {
			format!("{key} (current)")
		} else {
			key.clone()
		};
		let mut option = SelectOption::new().with(Prop::Value, key).label(label);
		let description = model_description(model);
		if !description.is_empty() {
			option = option.with(Prop::Desc, description);
		}
		select = select.option(option);
	}
	let content = Col::new().child(select).child(
		TextLeaf::new()
			.with(Prop::Dim, true)
			.text("Type to filter · Enter select · Esc cancel"),
	);
	let picker = Boxed::new()
		.with(Prop::Border, Border::Round)
		.with(Prop::Title, "Select Model")
		.with(Prop::PadX, 1_u16)
		.child(content);
	ui.show_overlay(
		picker,
		OverlayOptions::default()
			.anchor(OverlayAnchor::Center)
			.width(Dim::Pct(80))
			.min_width(48)
			.max_height(Dim::Pct(75))
			.margin(OverlayMargin::uniform(1))
			.min_viewport(Size::new(24, 6)),
	);
}

fn model_description(model: &ModelSpec) -> String {
	format_description(model.limits.context_window, &model.pricing.components)
}

fn format_description(context_window: Option<u64>, prices: &[Price]) -> String {
	let mut facts = Vec::with_capacity(2);
	if let Some(window) = context_window {
		facts.push(format!("ctx {window}"));
	}
	let input = prices
		.iter()
		.find(|price| price.unit == PriceUnit::MtokInput);
	let output = prices
		.iter()
		.find(|price| price.unit == PriceUnit::MtokOutput);
	if let (Some(input), Some(output)) = (input, output) {
		facts.push(format!(
			"${}/{} per Mtok",
			format_dollars(input.nanos_usd),
			format_dollars(output.nanos_usd)
		));
	}
	facts.join(" · ")
}

fn format_dollars(nanos: u64) -> String {
	let whole = nanos / 1_000_000_000;
	let fractional = nanos % 1_000_000_000;
	if fractional == 0 {
		return whole.to_string();
	}
	let fractional = format!("{fractional:09}");
	format!("{whole}.{}", fractional.trim_end_matches('0'))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn formats_exact_nano_dollar_prices() {
		assert_eq!(format_dollars(3_000_000_000), "3");
		assert_eq!(format_dollars(150_000_000), "0.15");
		assert_eq!(format_dollars(1), "0.000000001");
	}

	#[test]
	fn description_omits_missing_facts() {
		let prices = [Price { unit: PriceUnit::MtokInput, nanos_usd: 1_500_000_000 }, Price {
			unit:      PriceUnit::MtokOutput,
			nanos_usd: 7_500_000_000,
		}];
		assert_eq!(format_description(Some(200_000), &prices), "ctx 200000 · $1.5/7.5 per Mtok");
		assert_eq!(format_description(Some(200_000), &prices[..1]), "ctx 200000");
		assert_eq!(format_description(None, &[]), "");
	}
}
