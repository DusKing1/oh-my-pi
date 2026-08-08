//! Field-level provider catalog overlays.
//!
//! Merge order is built-in, then `~/.omp/providers.toml`, then the project's
//! `.omp/providers.toml`; therefore project values have highest precedence.
//! Omission is never deletion. A built-in provider is removed only by setting
//! `disabled = true` in a higher-precedence provider table.

use std::{
	collections::BTreeMap,
	fs,
	path::{Path, PathBuf},
};

use toml::{Table, Value};

use crate::provider::{BUILTIN_PROVIDERS_TOML, ProviderCatalog, ProviderLoadError, load_providers};

/// Failure while parsing or validating a provider overlay.
#[derive(Debug, thiserror::Error)]
pub enum OverlayError {
	/// An input layer was not valid TOML.
	#[error("invalid {layer} provider overlay `{source_name}`: {source}")]
	Parse {
		/// Human-readable layer name.
		layer:       &'static str,
		/// File or synthetic source name for the layer.
		source_name: String,
		/// TOML parser error, including key path and source span.
		#[source]
		source:      Box<toml::de::Error>,
	},
	/// An overlay parsed as TOML but failed provider schema validation.
	#[error(
		"invalid {layer} provider overlay `{source_name}` line {line} at `{provider_path}`: {source}"
	)]
	Validation {
		/// Human-readable layer name.
		layer:         &'static str,
		/// File or synthetic source name for the layer.
		source_name:   String,
		/// One-based line in the original overlay source.
		line:          usize,
		/// Provider-qualified TOML table containing the invalid field.
		provider_path: String,
		/// Provider schema error, retaining its materialized source span.
		#[source]
		source:        Box<ProviderLoadError>,
	},
	/// An overlay file existed but could not be read.
	#[error("could not read provider overlay `{path}`: {source}")]
	Read {
		/// Overlay path which failed.
		path:   PathBuf,
		/// Underlying filesystem error.
		#[source]
		source: std::io::Error,
	},
	/// The merged TOML could not be serialized for schema validation.
	#[error("could not materialize merged provider catalog: {0}")]
	Serialize(#[from] Box<toml::ser::Error>),
	/// The merged catalog failed the provider schema.
	#[error(transparent)]
	Schema(#[from] Box<ProviderLoadError>),
}

/// Loads built-ins plus the conventional user and project overlay files.
///
/// A missing overlay is equivalent to an empty layer. Other filesystem errors
/// are surfaced. The user path is `$HOME/.omp/providers.toml`; the project path
/// is `<project_dir>/.omp/providers.toml`.
pub fn load_with_overlays(project_dir: &Path) -> Result<ProviderCatalog, OverlayError> {
	let user_path = std::env::var_os("HOME")
		.map(PathBuf::from)
		.map(|home| home.join(".omp/providers.toml"));
	let project_path = project_dir.join(".omp/providers.toml");
	let user = user_path
		.as_deref()
		.map(read_optional)
		.transpose()?
		.flatten();
	let project = read_optional(&project_path)?;
	merge_provider_layers(
		BUILTIN_PROVIDERS_TOML,
		user.as_deref().map(|source| Layer {
			name: "user",
			source_name: user_path
				.as_deref()
				.expect("a user source requires a user path")
				.display()
				.to_string(),
			source,
		}),
		project.as_deref().map(|source| Layer {
			name: "project",
			source_name: project_path.display().to_string(),
			source,
		}),
	)
}

/// Merges built-in, user, and project provider TOML at field granularity.
///
/// Nested tables such as `auth`, `headers`, and `compat` merge recursively;
/// arrays and scalar values replace the lower-precedence value. New provider
/// ids are additive but must form a complete provider entry after merging.
pub fn merge_provider_sources(
	builtin: &str,
	user: Option<&str>,
	project: Option<&str>,
) -> Result<ProviderCatalog, OverlayError> {
	merge_provider_layers(
		builtin,
		user.map(|source| Layer {
			name: "user",
			source_name: "~/.omp/providers.toml".to_owned(),
			source,
		}),
		project.map(|source| Layer {
			name: "project",
			source_name: ".omp/providers.toml".to_owned(),
			source,
		}),
	)
}

struct Layer<'a> {
	name:        &'static str,
	source_name: String,
	source:      &'a str,
}

fn merge_provider_layers(
	builtin: &str,
	user: Option<Layer<'_>>,
	project: Option<Layer<'_>>,
) -> Result<ProviderCatalog, OverlayError> {
	let mut merged = parse_layer("built-in", "embedded providers.toml", builtin)?;
	validate_layer("built-in", "embedded providers.toml", builtin, &merged)?;
	let mut routing_overrides = BTreeMap::<String, (bool, bool)>::new();
	if let Some(layer) = user {
		let overlay = parse_layer(layer.name, &layer.source_name, layer.source)?;
		record_routing_overrides(&overlay, &mut routing_overrides);
		apply_layer(&mut merged, overlay);
		validate_layer(layer.name, &layer.source_name, layer.source, &merged)?;
	}
	if let Some(layer) = project {
		let overlay = parse_layer(layer.name, &layer.source_name, layer.source)?;
		record_routing_overrides(&overlay, &mut routing_overrides);
		apply_layer(&mut merged, overlay);
		validate_layer(layer.name, &layer.source_name, layer.source, &merged)?;
	}
	remove_disabled(&mut merged);
	let source = toml::to_string(&Value::Table(merged))
		.map_err(|error| OverlayError::Serialize(Box::new(error)))?;
	let mut providers =
		load_providers(&source).map_err(|error| OverlayError::Schema(Box::new(error)))?;
	for (id, (base_url, transport)) in routing_overrides {
		if let Some(provider) = providers.get_mut(id.as_str()) {
			provider.base_url_overridden = base_url;
			provider.transport_overridden = transport;
		}
	}
	Ok(providers)
}

fn record_routing_overrides(overlay: &Table, overrides: &mut BTreeMap<String, (bool, bool)>) {
	let Some(Value::Table(providers)) = overlay.get("providers") else {
		return;
	};
	for (id, provider) in providers {
		let Value::Table(provider) = provider else {
			continue;
		};
		let entry = overrides.entry(id.clone()).or_default();
		entry.0 |= provider.contains_key("base_url");
		entry.1 |= provider.contains_key("transport");
	}
}

fn read_optional(path: &Path) -> Result<Option<String>, OverlayError> {
	match fs::read_to_string(path) {
		Ok(source) => Ok(Some(source)),
		Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
		Err(source) => Err(OverlayError::Read { path: path.to_owned(), source }),
	}
}

fn parse_layer(layer: &'static str, source_name: &str, input: &str) -> Result<Table, OverlayError> {
	toml::from_str::<Table>(input).map_err(|source| OverlayError::Parse {
		layer,
		source_name: source_name.to_owned(),
		source: Box::new(source),
	})
}

fn validate_layer(
	layer: &'static str,
	source_name: &str,
	overlay_source: &str,
	merged: &Table,
) -> Result<(), OverlayError> {
	let mut candidate = merged.clone();
	remove_disabled(&mut candidate);
	let materialized = toml::to_string(&Value::Table(candidate))
		.map_err(|error| OverlayError::Serialize(Box::new(error)))?;
	load_providers(&materialized)
		.map(|_| ())
		.map_err(|source| OverlayError::Validation {
			layer,
			source_name: source_name.to_owned(),
			line: overlay_line(&materialized, overlay_source, &source),
			provider_path: provider_path(&materialized, &source),
			source: Box::new(source),
		})
}

fn overlay_line(materialized: &str, overlay: &str, error: &ProviderLoadError) -> usize {
	let ProviderLoadError::Toml(error) = error;
	let offset = error
		.span()
		.map_or(materialized.len(), |span| span.start.min(materialized.len()));
	let materialized_line = materialized[offset..]
		.lines()
		.next()
		.unwrap_or_default()
		.trim();
	let key = materialized_line
		.split_once('=')
		.map_or(materialized_line, |(key, _)| key.trim());
	overlay
		.lines()
		.position(|line| {
			let line = line.trim();
			line == key
				|| line
					.strip_prefix(key)
					.is_some_and(|tail| tail.trim_start().starts_with('='))
		})
		.map_or(1, |index| index + 1)
}

fn provider_path(materialized: &str, error: &ProviderLoadError) -> String {
	let ProviderLoadError::Toml(error) = error;
	let offset = error
		.span()
		.map_or(materialized.len(), |span| span.start.min(materialized.len()));
	materialized[..offset]
		.lines()
		.rev()
		.find_map(|line| {
			let table = line.trim().strip_prefix('[')?.strip_suffix(']')?;
			table.starts_with("providers.").then(|| table.to_owned())
		})
		.unwrap_or_else(|| "providers".to_owned())
}

fn apply_layer(base: &mut Table, overlay: Table) {
	for (key, value) in overlay {
		match (base.get_mut(&key), value) {
			(Some(Value::Table(base_table)), Value::Table(overlay_table)) => {
				apply_layer(base_table, overlay_table);
			},
			(_, value) => {
				base.insert(key, value);
			},
		}
	}
}

fn remove_disabled(root: &mut Table) {
	let Some(Value::Table(providers)) = root.get_mut("providers") else {
		return;
	};
	providers.retain(|_, provider| {
		let Value::Table(fields) = provider else {
			return true;
		};
		match fields.get("disabled").and_then(Value::as_bool) {
			Some(disabled) => {
				fields.remove("disabled");
				!disabled
			},
			None => true,
		}
	});
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		compat::{MaxTokensField, ReasoningWireFormat, ToolStrictMode},
		provider::{AuthSpec, TransportId},
	};

	const BUILTIN: &str = include_str!("../providers.toml");

	#[test]
	fn shipped_catalog_has_representative_rows() {
		let catalog = load_providers(BUILTIN).expect("shipped providers.toml must parse");
		let groq = &catalog["groq"];
		assert_eq!(groq.transport, TransportId::OpenAiChat);
		assert_eq!(groq.base_url.as_str(), "https://api.groq.com/openai/v1");
		assert!(groq.compat.usage_in_streaming);
		assert_eq!(groq.compat.max_tokens_field, MaxTokensField::MaxCompletionTokens);

		let anthropic = &catalog["anthropic"];
		assert_eq!(anthropic.transport, TransportId::AnthropicMessages);
		assert!(matches!(anthropic.auth, AuthSpec::Header { .. }));
		assert_eq!(anthropic.compat.reasoning_wire_format, ReasoningWireFormat::Anthropic);
		assert_eq!(anthropic.compat.tool_strict_mode, ToolStrictMode::None);
	}

	#[test]
	fn overlays_replace_one_field_and_preserve_siblings() {
		let user = r#"
[providers.openai]
base_url = "https://user.example/v1"
"#;
		let project = r#"
[providers.openai]
base_url = "https://project.example/v1"
[providers.openai.compat]
usage_in_streaming = false
"#;
		let catalog = merge_provider_sources(BUILTIN, Some(user), Some(project))
			.expect("partial overlays must merge");
		let openai = &catalog["openai"];
		assert_eq!(openai.base_url.as_str(), "https://project.example/v1");
		assert_eq!(openai.transport, TransportId::OpenAiResponses);
		assert!(!openai.compat.usage_in_streaming);
		assert!(openai.compat.multiple_system_messages);
		assert!(openai.base_url_overridden);
		assert!(!openai.transport_overridden);
	}

	#[test]
	fn routing_override_provenance_is_fieldwise_and_overlay_only() {
		let catalog = merge_provider_sources(
			BUILTIN,
			Some("[providers.openai]\ntransport = \"open-ai-chat\"\n"),
			None,
		)
		.expect("transport-only overlay");
		let openai = &catalog["openai"];
		assert!(openai.transport_overridden);
		assert!(!openai.base_url_overridden);
		let builtin = load_providers(BUILTIN).expect("built-in providers");
		assert!(!builtin["openai"].transport_overridden);
		assert!(!builtin["openai"].base_url_overridden);
	}

	#[test]
	fn builtin_removal_requires_explicit_disabled_flag() {
		let unrelated = "[providers.openai.compat]\nusage_in_streaming = false\n";
		let retained = merge_provider_sources(BUILTIN, Some(unrelated), None)
			.expect("omission must retain providers");
		assert!(retained.contains_key("anthropic"));

		let disable = "[providers.anthropic]\ndisabled = true\n";
		let removed = merge_provider_sources(BUILTIN, Some(disable), None)
			.expect("explicit disable must be accepted");
		assert!(!removed.contains_key("anthropic"));
	}

	#[test]
	fn unknown_overlay_key_reports_precise_provider_path() {
		let typo = r"
[providers.openai.compat]
usage_in_streamng = false
";
		let error = merge_provider_sources(BUILTIN, Some(typo), None)
			.expect_err("unknown compat key must be rejected")
			.to_string();
		assert!(error.contains("usage_in_streamng"), "{error}");
		assert!(error.contains("providers.openai.compat"), "{error}");
	}

	#[test]
	fn unknown_project_overlay_key_reports_provider_and_source_file() {
		let typo = r"
[providers.anthropic.compat]
usage_in_streamng = false
";
		let project_path = "/workspace/example/.omp/providers.toml";
		let error = merge_provider_layers(
			BUILTIN,
			None,
			Some(Layer {
				name:        "project",
				source_name: project_path.to_owned(),
				source:      typo,
			}),
		)
		.expect_err("unknown project compat key must be rejected")
		.to_string();
		assert!(error.contains("usage_in_streamng"), "{error}");
		assert!(error.contains("providers.anthropic.compat"), "{error}");
		assert!(error.contains(project_path), "{error}");
		assert!(error.contains("line "), "{error}");
	}
}
