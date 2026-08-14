//! Projects the pinned legacy catalog loader into the reduced oracle schema.
//!
//! Copy this source into the pre-migration catalog crate pinned by
//! `compat/README.md`; it reproduces the checked-in fixture byte for byte.

use std::{collections::BTreeMap, env, fs, path::PathBuf};

use omp_llm_catalog::models::{ModelBehavior, ModelCard, load_catalog_zstd};
use serde_json::{Map, Value, json};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let mut arguments = env::args_os().skip(1);
	let baseline_source = arguments.next().map(PathBuf::from).ok_or(usage())?;
	let current_source = arguments.next().map(PathBuf::from).ok_or(usage())?;
	let destination = arguments.next().map(PathBuf::from).ok_or(usage())?;
	if arguments.next().is_some() {
		return Err(usage().into());
	}

	let baseline_models = decode_source(&fs::read(baseline_source)?)?;
	let source_bytes = fs::read(current_source)?;
	let catalog = load_catalog_zstd(&source_bytes)?;
	let source_models = decode_source(&source_bytes)?;
	let models = catalog
		.models()
		.iter()
		.map(|model| {
			let source_model = source_models
				.get(model.provider.as_str())
				.and_then(|models| models.get(model.model.as_str()));
			let baseline_model = baseline_models
				.get(model.provider.as_str())
				.and_then(|models| models.get(model.model.as_str()));
			let source_changed = source_model != baseline_model;
			let authored_required_content = source_model
				.and_then(|source| source.get("compat"))
				.and_then(|compat| compat.get("requiresReasoningContentForAllAssistantTurns"))
				.is_some();
			project_model(
				model,
				source_changed,
				source_model.is_some() && baseline_model.is_none(),
				authored_required_content,
			)
		})
		.collect::<Vec<_>>();
	let mut output = serde_json::to_vec_pretty(&json!({
		"schema_version": 1,
		"models": models,
	}))?;
	output.push(b'\n');
	fs::write(destination, output)?;
	Ok(())
}

fn usage() -> &'static str {
	"usage: project_normalized BASELINE_ZST SOURCE_ZST DESTINATION_JSON"
}

fn decode_source(
	bytes: &[u8],
) -> Result<BTreeMap<String, BTreeMap<String, Value>>, Box<dyn std::error::Error>> {
	let json = zstd::stream::decode_all(bytes)?;
	Ok(serde_json::from_slice(&json)?)
}

fn project_model(
	model: &ModelCard,
	source_changed: bool,
	source_added: bool,
	authored_required_content: bool,
) -> Value {
	let wire = model.wire.as_ref().map(|wire| {
		json!({
			"transport": wire.transport,
			"base_url": wire.base_url,
		})
	});
	// The reduced schema treats an omitted output list as text for every facet.
	let outputs = if model.outputs.is_empty() {
		json!(["text"])
	} else {
		json!(model.outputs)
	};
	let mut behavior = project_behavior(&model.behavior);
	if source_changed && !authored_required_content {
		// Source overlays preserve absence instead of retaining a loader-synthesized
		// default.
		behavior["compat"]
			.as_object_mut()
			.expect("compat projects as an object")
			.remove("wire/requires_reasoning_content_for_all_assistant_turns");
	}
	let mut projected = json!({
		"id": model.id,
		"provider": model.provider,
		"model": model.model,
		"name": model.name,
		"family": model.family,
		"facets": model.facets,
		"modalities": {
			"inputs": model.inputs,
			"outputs": outputs,
		},
		"reasoning": model.reasoning,
		"efforts": model.efforts,
		"limits": {
			"context_window": model.context_window,
			"max_output_tokens": model.max_output_tokens,
		},
		"pricing": model.pricing,
		"pricing_tiers": model.pricing_tiers,
		"availability": model.availability,
		"source": model.source,
		"blocked_until_ms": model.blocked_until_ms,
		"deprecated": model.deprecated,
		"updated_at_ms": model.updated_at_ms,
		"props": model.props,
		"effort_routing": model.effort_routing,
		"wire": wire,
		"behavior": behavior,
	});
	if source_changed {
		// Changed source rows were overlaid from key-sorted JSON after baseline
		// projection.
		sort_value_keys(&mut projected["effort_routing"]);
		sort_value_keys(&mut projected["behavior"]["thinking"]);
	}
	if source_added {
		// Added rows started as default projections before identity fields were
		// inserted.
		let object = projected
			.as_object_mut()
			.expect("model projects as an object");
		let pricing_tiers = object
			.shift_remove("pricing_tiers")
			.expect("projected field");
		let props = object.shift_remove("props").expect("projected field");
		let remaining = std::mem::take(object);
		let mut reordered = Map::new();
		reordered.insert("pricing_tiers".into(), pricing_tiers);
		reordered.insert("props".into(), props);
		reordered.extend(remaining);
		*object = reordered;
	}
	projected
}

fn project_behavior(behavior: &ModelBehavior) -> Value {
	let remote_compaction = behavior.remote_compaction.as_ref().map(|remote| {
		json!({
			"enabled": remote.enabled,
			"transport": remote.transport,
			"endpoint": remote.endpoint,
			"v2_streaming_enabled": remote.v2_streaming_enabled,
			"v2_endpoint": remote.v2_endpoint,
			"streaming_endpoint": remote.streaming_endpoint,
			"model": remote.model,
		})
	});
	json!({
		"thinking": behavior.thinking,
		"supports_tools": behavior.supports_tools,
		"supports_computer_use": behavior.supports_computer_use,
		"supports_computer_use_config": behavior.supports_computer_use_config,
		"cursor_max_mode": behavior.cursor_max_mode,
		"omit_max_output_tokens": behavior.omit_max_output_tokens,
		"apply_patch_tool_type": behavior.apply_patch_tool_type,
		"context_promotion_target": behavior.context_promotion_target,
		"request_model_id": behavior.request_model_id,
		"remote_compaction": remote_compaction,
		"premium_multiplier": behavior.premium_multiplier,
		"reasoning_mode": behavior.reasoning_mode,
		"use_responses_lite": behavior.use_responses_lite,
		"prefer_websockets": behavior.prefer_websockets,
		"priority": behavior.priority,
		"headers": behavior.headers,
		"compat": behavior.compat,
	})
}

fn sort_value_keys(value: &mut Value) {
	match value {
		Value::Object(fields) => {
			for value in fields.values_mut() {
				sort_value_keys(value);
			}
			fields.sort_keys();
		},
		Value::Array(values) => {
			for value in values {
				sort_value_keys(value);
			}
		},
		_ => {},
	}
}
