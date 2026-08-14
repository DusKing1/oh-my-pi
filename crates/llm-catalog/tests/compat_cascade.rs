//! Proves the KDL compat cascade against the frozen oracle and the census.
//!
//! The archived `compat-profiles.json` is only the resolved non-empty slice of
//! the old data path, so wire verification is two-sided: every axis the
//! oracle pins must resolve to exactly the oracle value, and every axis the
//! cascade adds beyond the oracle must be one of the documented census
//! composition overlays (pi-openai-chat:012–015) — nothing else. Thinking
//! verification is exact against `thinking-profiles.json`, resolved over all
//! 4,227 catalog models with the catalog's reasoning capability as the gate,
//! which proves class and `on` thinking rules never leak onto non-reasoning
//! siblings. Every `ready` quirk-census case executes against the real
//! machinery.

use std::{collections::BTreeMap, fs, path::Path};

use omp_core::SemVer;
use omp_llm_catalog::{
	BUNDLED_COMPAT, CascadeError, Catalog, ClassificationInput, ClassificationPhase, CompatCascade,
	EffortTier, KNOWN_AXES, ModelKey, ResolveTarget, ThinkingEffort, ThinkingFormat, WirePolicy,
	classify,
};
use serde::Deserialize;
use serde_json::Value;

const COMPAT_PROFILES: &str =
	include_str!("../../../fixtures/llm-oracle/catalog-policy/compat-profiles.json");
const THINKING_PROFILES: &str =
	include_str!("../../../fixtures/llm-oracle/catalog-policy/thinking-profiles.json");
const NORMALIZED_MODELS: &str =
	include_str!("../../../fixtures/llm-oracle/catalog/models.normalized.json");
const CENSUS_CASES: &str = include_str!("../../../fixtures/llm-oracle/quirk-census/cases.jsonl");
const CATALOG_POSTCARD: &[u8] = include_bytes!("../data/catalog.postcard");

#[derive(Deserialize)]
struct ProfileDocument {
	profiles: Vec<Profile>,
}

#[derive(Deserialize)]
struct Profile {
	models: Vec<String>,
	shape:  BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct NormalizedDocument {
	models: Vec<NormalizedModel>,
}

#[derive(Deserialize)]
struct NormalizedModel {
	id:           String,
	provider:     String,
	model:        String,
	#[serde(default)]
	class:        Option<String>,
	#[serde(default, rename = "family")]
	legacy_class: Option<String>,
	#[serde(default)]
	behavior:     Option<Behavior>,
}

#[derive(Deserialize)]
struct Behavior {
	#[serde(default)]
	thinking: Option<Value>,
}

#[derive(Deserialize)]
struct Case {
	id:           String,
	fixture_kind: String,
	status:       String,
	#[serde(default)]
	r#match:      Option<String>,
	#[serde(default)]
	input:        Value,
	#[serde(default)]
	expected:     Value,
}

fn profile_shapes(raw: &str, strip: Option<&str>) -> BTreeMap<String, BTreeMap<String, Value>> {
	let document: ProfileDocument = serde_json::from_str(raw).expect("profile fixture parses");
	let mut shapes = BTreeMap::new();
	for profile in document.profiles {
		let shape: BTreeMap<String, Value> = profile
			.shape
			.into_iter()
			.map(|(key, value)| match strip {
				Some(prefix) => (
					key.strip_prefix(prefix)
						.expect("prefixed oracle key")
						.into(),
					value,
				),
				None => (key, value),
			})
			.collect();
		for model in profile.models {
			assert!(
				shapes.insert(model.clone(), shape.clone()).is_none(),
				"oracle model {model} appears in two profiles"
			);
		}
	}
	shapes
}

/// Census wire overlay beyond the archived oracle slice: the class×host
/// `thinking_format` compositions from pi-openai-chat:012–015. Applies only
/// where the oracle is silent on the axis.
fn census_thinking_format(provider: &str, class: &str) -> Option<&'static str> {
	match provider {
		"openrouter" => Some("openrouter"),
		"alibaba-token-plan" | "alibaba-coding-plan" => Some("qwen"),
		"nvidia" if class == "qwen" => Some("qwen-chat-template"),
		"fireworks" if class == "qwen" => Some("openai"),
		_ if class == "qwen" => Some("qwen"),
		_ => None,
	}
}

fn frozen_class_of(model: &NormalizedModel) -> &str {
	model
		.class
		.as_deref()
		.or(model.legacy_class.as_deref())
		.filter(|class| !class.is_empty())
		.unwrap_or("unknown")
}

fn parse_revision(value: Option<&Value>) -> Option<SemVer> {
	let value = value?;
	if value.is_null() {
		return None;
	}
	let components = match value {
		Value::String(revision) => {
			let mut components = [0_u8; 3];
			let mut count = 0_usize;
			for component in revision.split('.') {
				assert!(
					count < components.len()
						&& !component.is_empty()
						&& component.bytes().all(|byte| byte.is_ascii_digit()),
					"revision must contain one to three numeric components"
				);
				components[count] = component
					.parse()
					.expect("revision components must fit in u8");
				count += 1;
			}
			assert!(count > 0, "revision must contain at least one component");
			components
		},
		Value::Object(revision) => {
			assert_eq!(
				revision.len(),
				3,
				"revision object must contain exactly major, minor, and patch"
			);
			let component = |name| {
				let value = revision
					.get(name)
					.unwrap_or_else(|| panic!("revision object is missing {name}"));
				u8::try_from(
					value
						.as_u64()
						.unwrap_or_else(|| panic!("revision {name} must be an unsigned integer")),
				)
				.unwrap_or_else(|_| panic!("revision {name} must fit in u8"))
			};
			[component("major"), component("minor"), component("patch")]
		},
		_ => panic!("revision must be a string or an object"),
	};
	Some(SemVer::new(components[0], components[1], components[2]))
}

#[test]
fn bundled_sources_match_the_compat_tree() {
	let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("compat");
	let mut on_disk = Vec::new();
	for group in ["classes", "providers"] {
		for entry in fs::read_dir(root.join(group)).expect("compat group exists") {
			let path = entry.expect("readable dir entry").path();
			if path.extension().is_some_and(|extension| extension == "kdl") {
				let stem = path
					.file_stem()
					.expect("file stem")
					.to_str()
					.expect("utf-8 name")
					.to_owned();
				on_disk.push(format!("{group}/{stem}"));
			}
		}
	}
	on_disk.sort();
	let mut bundled: Vec<String> = BUNDLED_COMPAT
		.iter()
		.map(|&(name, _)| name.to_owned())
		.collect();
	bundled.sort();
	assert_eq!(bundled, on_disk, "BUNDLED_COMPAT must list exactly compat/{{classes,providers}}");
}

#[test]
fn axis_vocabulary_matches_the_oracles_exactly() {
	let wire: ProfileDocument =
		serde_json::from_str(COMPAT_PROFILES).expect("compat profiles parse");
	let thinking: ProfileDocument =
		serde_json::from_str(THINKING_PROFILES).expect("thinking profiles parse");
	let mut oracle_axes: Vec<String> = wire
		.profiles
		.iter()
		.flat_map(|profile| profile.shape.keys())
		.map(|key| {
			key.strip_prefix("wire/")
				.expect("wire/-prefixed key")
				.to_owned()
		})
		.chain(
			thinking
				.profiles
				.iter()
				.flat_map(|profile| profile.shape.keys().cloned()),
		)
		.collect();
	oracle_axes.sort_unstable();
	oracle_axes.dedup();
	let mut known: Vec<String> = KNOWN_AXES
		.iter()
		.map(|&(_, _, key, _)| key.to_owned())
		.collect();
	known.sort_unstable();
	assert_eq!(known, oracle_axes, "KNOWN_AXES drifted from the oracle vocabularies");
}

#[test]
fn cascade_resolves_every_catalog_model_to_oracle_plus_census_overlay() {
	let cascade = CompatCascade::bundled().expect("bundled cascade parses");
	let normalized: NormalizedDocument =
		serde_json::from_str(NORMALIZED_MODELS).expect("normalized model fixture parses");
	let wire_oracle = profile_shapes(COMPAT_PROFILES, Some("wire/"));
	let thinking_oracle = profile_shapes(THINKING_PROFILES, None);
	let mut checked = 0_usize;
	let mut wire_overridden = 0_usize;
	let mut thinking_profiled = 0_usize;
	let mut overlay_applied = 0_usize;
	for model in &normalized.models {
		let frozen_class = frozen_class_of(model);
		let classification = classify(ClassificationInput {
			phase:          ClassificationPhase::CatalogCompiler,
			provider:       &model.provider,
			model:          &model.model,
			observed_at_ms: None,
		});
		assert_eq!(
			classification.class.as_str(),
			frozen_class,
			"frozen legacy class diverges for {}",
			model.id
		);
		let class = classification.class.as_str();
		let reasoning = model
			.behavior
			.as_ref()
			.is_some_and(|b| b.thinking.is_some());
		let resolved = cascade
			.resolve(&ResolveTarget {
				provider: &model.provider,
				class,
				family: classification.family.as_ref().map(|family| family.as_str()),
				revision: classification.revision,
				model: &model.model,
				reasoning,
			})
			.unwrap_or_else(|error| panic!("{}: {error}", model.id));

		// Wire: oracle-pinned axes exact; additions only from the census overlay.
		let mut expected = wire_oracle.get(&model.id).cloned().unwrap_or_default();
		if !expected.contains_key("thinking_format") {
			if let Some(format) = census_thinking_format(&model.provider, class) {
				expected.insert("thinking_format".into(), Value::from(format));
				overlay_applied += 1;
			}
		}
		let resolved_wire: BTreeMap<String, Value> = resolved
			.wire
			.iter()
			.map(|(key, value)| (key.as_str().to_owned(), value.clone()))
			.collect();
		assert_eq!(resolved_wire, expected, "wire cascade diverges for {}", model.id);
		if wire_oracle.contains_key(&model.id) {
			wire_overridden += 1;
		}

		// Thinking: exact against the profile oracle; empty when not profiled.
		let expected_thinking = thinking_oracle.get(&model.id).cloned().unwrap_or_default();
		let resolved_thinking: BTreeMap<String, Value> = resolved
			.thinking
			.iter()
			.map(|(key, value)| (key.as_str().to_owned(), value.clone()))
			.collect();
		assert_eq!(
			resolved_thinking, expected_thinking,
			"thinking cascade diverges for {}",
			model.id
		);
		assert_eq!(
			thinking_oracle.contains_key(&model.id),
			reasoning,
			"capability gate desynced from the thinking oracle for {}",
			model.id
		);
		if reasoning {
			thinking_profiled += 1;
		}
		checked += 1;
	}
	assert_eq!(checked, 4_227, "full catalog roster resolved");
	assert_eq!(wire_overridden, 312, "archived wire override census");
	assert_eq!(thinking_profiled, 2_294, "thinking profile census");
	assert!(overlay_applied > 500, "census overlay must reach real models: {overlay_applied}");
}

#[test]
fn compiled_catalog_carries_cascade_overlay_policies() {
	let catalog = Catalog::decode(CATALOG_POSTCARD).expect("compiled catalog snapshot decodes");

	let nvidia_qwen = catalog
		.model(&ModelKey::from("nvidia/qwen/qwen3-next-80b-a3b-thinking"))
		.expect("frozen nvidia qwen model is compiled");
	let wire_policy = catalog
		.wire_policy(&nvidia_qwen.wire_policy)
		.expect("nvidia qwen wire policy is interned");
	assert_eq!(
		wire_policy.reasoning.thinking_format,
		Some(ThinkingFormat::QwenChatTemplate),
		"compiled nvidia qwen policy must carry the cascade overlay"
	);

	let cursor_gpt = catalog
		.model(&ModelKey::from("cursor/gpt-5.1"))
		.expect("frozen cursor gpt-5.1 model is compiled");
	let thinking_policy = catalog
		.thinking_policy(
			cursor_gpt
				.thinking
				.as_ref()
				.expect("cursor gpt-5.1 references a thinking policy"),
		)
		.expect("cursor gpt-5.1 thinking policy is interned");
	assert_eq!(
		thinking_policy.efforts.as_slice(),
		&[ThinkingEffort::Low, ThinkingEffort::High],
		"compiled cursor gpt-5.1 policy must carry the cascade efforts"
	);
}

#[test]
fn every_ready_census_case_executes_against_real_machinery() {
	let cascade = CompatCascade::bundled().expect("bundled cascade parses");
	let mut executed = Vec::new();
	for line in CENSUS_CASES.lines().filter(|line| !line.trim().is_empty()) {
		let case: Case = serde_json::from_str(line).expect("census case parses");
		if case.status != "ready" {
			continue;
		}
		match case.fixture_kind.as_str() {
			"identity" => run_identity_case(&case),
			"policy-resolution" => run_policy_case(&cascade, &case),
			"compile-error" => run_compile_error_case(&case),
			other => panic!("ready case {} has unexecutable kind {other}", case.id),
		}
		executed.push(case.id);
	}
	assert_eq!(executed.len(), 23, "ready census cases all executed: {executed:?}");
}

fn run_identity_case(case: &Case) {
	let provider = case.input["provider"]
		.as_str()
		.expect("identity input provider");
	let model = case.input["model_id"]
		.as_str()
		.expect("identity input model_id");
	let classification = classify(ClassificationInput {
		phase: ClassificationPhase::CatalogCompiler,
		provider,
		model,
		observed_at_ms: None,
	});
	let expected = case.expected.as_object().expect("identity expected object");
	for (key, want) in expected {
		match key.as_str() {
			"family" => {
				assert_eq!(
					classification.class.as_str(),
					want.as_str().expect("class string"),
					"{}: class",
					case.id
				);
			},
			"logical_model" => {
				assert_eq!(
					classification.logical_model.as_str(),
					want.as_str().expect("logical_model string"),
					"{}: logical_model",
					case.id
				);
			},
			"thinking_variant" => {
				assert_eq!(
					classification.thinking_variant,
					want.as_bool().expect("thinking_variant bool"),
					"{}: thinking_variant",
					case.id
				);
			},
			"effort" => {
				let effort = classification.effort.map(|tier| match tier {
					EffortTier::Off => "off",
					EffortTier::Minimal => "minimal",
					EffortTier::Low => "low",
					EffortTier::Medium => "medium",
					EffortTier::High => "high",
					EffortTier::XHigh => "xhigh",
					EffortTier::Max => "max",
				});
				assert_eq!(effort, want.as_str(), "{}: effort", case.id);
			},
			other => panic!("{}: unhandled identity expectation `{other}`", case.id),
		}
	}
}

fn run_policy_case(cascade: &CompatCascade, case: &Case) {
	let provider = case.input["provider"]
		.as_str()
		.expect("policy input provider");
	let model = case.input["model_id"]
		.as_str()
		.expect("policy input model_id");
	let explicit_class = case.input.get("class");
	let class = explicit_class
		.map(|class| class.as_str().expect("policy input class"))
		.unwrap_or_else(|| case.input["family"].as_str().unwrap_or("unknown"));
	let family = explicit_class.and_then(|_| {
		case
			.input
			.get("family")
			.filter(|family| !family.is_null())
			.map(|family| family.as_str().expect("policy input product family"))
	});
	let revision = parse_revision(case.input.get("revision"));
	let reasoning = case.input["reasoning"].as_bool().unwrap_or(false);
	let resolved = cascade
		.resolve(&ResolveTarget { provider, class, family, revision, model, reasoning })
		.expect("policy resolves");
	let overrides = case.expected["overrides"]
		.as_object()
		.expect("expected overrides object");
	let subset = case.r#match.as_deref() == Some("subset");
	if subset {
		for (axis, want) in overrides {
			let got = resolved
				.wire
				.get(axis.as_str())
				.unwrap_or_else(|| panic!("{}: axis {axis} unresolved", case.id));
			assert_eq!(got, want, "{}: axis {axis}", case.id);
		}
	} else {
		let resolved_json: BTreeMap<&str, &Value> = resolved
			.wire
			.iter()
			.map(|(key, value)| (key.as_str(), value))
			.collect();
		let mut expected: BTreeMap<&str, &Value> = overrides
			.iter()
			.map(|(key, value)| (key.as_str(), value))
			.collect();
		// Census overlay applies on top of the archived expectations.
		let overlay = census_thinking_format(provider, class).map(Value::from);
		if let Some(overlay) = overlay.as_ref() {
			expected.entry("thinking_format").or_insert(overlay);
		}
		assert_eq!(resolved_json, expected, "{}: resolved overrides", case.id);
	}
	if let Some(absent) = case.expected.get("absent").and_then(Value::as_array) {
		for axis in absent {
			let axis = axis.as_str().expect("absent axis name");
			assert!(!resolved.wire.contains_key(axis), "{}: axis {axis} must be unset", case.id);
		}
	}
	if let Some(baseline) = case.expected.get("baseline").and_then(Value::as_object) {
		for (axis, want) in baseline {
			assert_eq!(axis, "max_tokens_field", "{}: only max_tokens_field is pinned", case.id);
			let policy = WirePolicy::baseline();
			let field = policy
				.context
				.max_tokens_field
				.expect("baseline pins max_tokens_field");
			let field = serde_json::to_value(field).expect("field serializes");
			assert_eq!(&field, want, "{}: baseline {axis}", case.id);
		}
	}
}

fn run_compile_error_case(case: &Case) {
	let parse = |text: &str| CompatCascade::parse(&[("case.kdl", text)]);
	match case.id.as_str() {
		"compile.reject.ambiguous-overlap" => {
			let cascade = parse(
				r#"provider "acme" {
					models "foo-*" { thinking-format "zai" }
					models "*-bar" { thinking-format "qwen" }
				}"#,
			)
			.expect("rule set parses");
			let error = cascade
				.resolve(&ResolveTarget {
					provider:  "acme",
					class:     "unknown",
					family:    None,
					revision:  None,
					model:     "foo-bar",
					reasoning: false,
				})
				.expect_err("must reject");
			assert!(
				matches!(&error, CascadeError::AmbiguousOverlap(details)
					if details.axis.as_str() == "thinking_format"),
				"{}: {error}",
				case.id
			);
		},
		"compile.accept.disjoint-axes-overlap" => {
			let cascade = parse(
				r#"provider "acme" {
					models "foo-*" { thinking-format "zai" }
					models "*-bar" { supports-store #false }
				}"#,
			)
			.expect("rule set parses");
			let resolved = cascade
				.resolve(&ResolveTarget {
					provider:  "acme",
					class:     "unknown",
					family:    None,
					revision:  None,
					model:     "foo-bar",
					reasoning: false,
				})
				.expect("disjoint is legal");
			assert_eq!(resolved.wire["thinking_format"], Value::from("zai"), "{}", case.id);
			assert_eq!(resolved.wire["supports_store"], Value::Bool(false), "{}", case.id);
		},
		"compile.accept.explicit-priority" => {
			let cascade = parse(
				r#"provider "acme" {
					models "foo-*" priority=10 { thinking-format "zai" }
					models "*-bar" { thinking-format "qwen" }
				}"#,
			)
			.expect("rule set parses");
			let resolved = cascade
				.resolve(&ResolveTarget {
					provider:  "acme",
					class:     "unknown",
					family:    None,
					revision:  None,
					model:     "foo-bar",
					reasoning: false,
				})
				.expect("priority wins");
			assert_eq!(resolved.wire["thinking_format"], Value::from("zai"), "{}", case.id);
		},
		"compile.reject.unconsumed-directive" => {
			let error = parse(r#"provider "acme" { schema-flavor "mfjs" }"#)
				.expect_err("unconsumed axis must fail");
			assert!(
				matches!(&error, CascadeError::UnknownDirective { directive, .. }
					if directive.as_str() == "schema-flavor"),
				"{}: {error}",
				case.id
			);
		},
		"compile.reject.unknown-directive" => {
			let error =
				parse(r#"provider "acme" { thinkign-format "zai" }"#).expect_err("typo must fail");
			assert!(
				matches!(&error, CascadeError::UnknownDirective { directive, .. }
					if directive.as_str() == "thinkign-format"),
				"{}: {error}",
				case.id
			);
		},
		other => panic!("unmapped compile-error case {other}"),
	}
}
