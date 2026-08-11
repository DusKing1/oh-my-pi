use omp_llm_types::{
	Error, Feature, Props, RequestMeta, ResponseFormat, ResponseFormatKind, Unsupported,
	UnsupportedAction,
};
use serde::Serialize;
use serde_json::{Map, Value, value::RawValue};

/// Anthropic's structured-output JSON Schema shape.
#[derive(Serialize)]
pub struct JsonSchemaFormat<'a> {
	pub(crate) r#type: &'static str,
	pub(crate) name:   &'a str,
	pub(crate) schema: &'a RawValue,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) strict: Option<bool>,
}

/// Canonical JSON Schema or a provider-native future format retained verbatim.
#[derive(Serialize)]
#[serde(untagged)]
pub enum OutputFormat<'a> {
	JsonSchema(JsonSchemaFormat<'a>),
	Native(&'a Value),
}

/// Anthropic output controls shared by structured output and thinking effort.
#[derive(Serialize)]
pub struct OutputConfig<'a> {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) effort:      Option<&'a str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) task_budget: Option<&'a Value>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) format:      Option<OutputFormat<'a>>,
}

/// Native Messages controls projected from the Anthropic provider namespace.
pub struct ControlProjection<'a> {
	pub(crate) disable_parallel_tool_use: Option<bool>,
	pub(crate) eager_input_streaming:     Option<bool>,
	pub(crate) cache_control:             Option<super::CacheControl>,
	pub(crate) output_config:             Option<OutputConfig<'a>>,
	pub(crate) metadata:                  Option<Value>,
	pub(crate) context_management:        Option<&'a Value>,
	pub(crate) service_tier:              Option<&'a str>,
	pub(crate) speed:                     Option<&'static str>,
	pub(crate) container:                 Option<&'a Value>,
}

/// Projects provider-only Anthropic controls without silently discarding
/// malformed values.
pub fn project<'a>(
	format: &'a Option<Feature<ResponseFormat>>,
	meta: &'a Option<RequestMeta>,
	props: &'a Props,
	unsupported: &mut Vec<Unsupported>,
) -> Result<ControlProjection<'a>, Error> {
	let disable_parallel_tool_use = parallel_policy(props, unsupported)?;
	let eager_input_streaming = boolean_option(props, "eager_input_streaming")?;
	let cache_control = cache_control(props, unsupported)?;
	let output_config = output_config(format, props, unsupported)?;
	let metadata = metadata(meta, props)?;
	let context_management = object_option(props, "context_management")?;
	let container = container(props)?;
	let (service_tier, speed) = service_tier(props)?;
	Ok(ControlProjection {
		disable_parallel_tool_use,
		eager_input_streaming,
		cache_control,
		output_config,
		metadata,
		context_management,
		service_tier,
		speed,
		container,
	})
}

fn parallel_policy(
	props: &Props,
	unsupported: &mut Vec<Unsupported>,
) -> Result<Option<bool>, Error> {
	if let Some(value) = props.get_ns("anthropic", "disable_parallel_tool_use") {
		return value
			.as_bool()
			.map(Some)
			.ok_or_else(|| provider_error("anthropic/disable_parallel_tool_use must be a boolean"));
	}
	if let Some(value) = props.get_ns("anthropic", "parallel_tool_use") {
		return value
			.as_bool()
			.map(|enabled| Some(!enabled))
			.ok_or_else(|| provider_error("anthropic/parallel_tool_use must be a boolean"));
	}
	if let Some(value) = props.get_ns("anthropic", "tool_choice") {
		let object = value
			.as_object()
			.ok_or_else(|| provider_error("anthropic/tool_choice must be an object"))?;
		for key in object.keys() {
			if key != "disable_parallel_tool_use" {
				super::report(
					unsupported,
					&format!("anthropic/tool_choice.{key}"),
					"unknown Anthropic tool_choice field was not sent",
					UnsupportedAction::Dropped,
				);
			}
		}
		if let Some(value) = object.get("disable_parallel_tool_use") {
			return value.as_bool().map(Some).ok_or_else(|| {
				provider_error("anthropic/tool_choice.disable_parallel_tool_use must be a boolean")
			});
		}
	}
	Ok(None)
}

fn boolean_option(props: &Props, name: &'static str) -> Result<Option<bool>, Error> {
	let Some(value) = props.get_ns("anthropic", name) else {
		return Ok(None);
	};
	value
		.as_bool()
		.map(Some)
		.ok_or_else(|| provider_error("Anthropic boolean-valued control had a non-boolean value"))
}

fn cache_control(
	props: &Props,
	unsupported: &mut Vec<Unsupported>,
) -> Result<Option<super::CacheControl>, Error> {
	let Some(value) = props.get_ns("anthropic", "cache_control") else {
		return Ok(None);
	};
	let object = value
		.as_object()
		.ok_or_else(|| provider_error("anthropic/cache_control must be an object"))?;
	for key in object.keys() {
		if !matches!(key.as_str(), "type" | "ttl" | "scope") {
			super::report(
				unsupported,
				&format!("anthropic/cache_control.{key}"),
				"unknown Anthropic cache_control field was not sent",
				UnsupportedAction::Dropped,
			);
		}
	}
	if object
		.get("type")
		.and_then(Value::as_str)
		.unwrap_or("ephemeral")
		!= "ephemeral"
	{
		return Err(provider_error("anthropic/cache_control.type must be ephemeral"));
	}
	let ttl = match object.get("ttl").and_then(Value::as_str).unwrap_or("5m") {
		"5m" => "5m",
		"1h" => "1h",
		_ => return Err(provider_error("anthropic/cache_control.ttl must be 5m or 1h")),
	};
	let scope = match object.get("scope") {
		None => None,
		Some(Value::String(value)) if value == "global" => Some("global"),
		_ => return Err(provider_error("anthropic/cache_control.scope must be global")),
	};
	Ok(Some(super::CacheControl { r#type: "ephemeral", ttl, scope }))
}

fn output_config<'a>(
	format: &'a Option<Feature<ResponseFormat>>,
	props: &'a Props,
	unsupported: &mut Vec<Unsupported>,
) -> Result<Option<OutputConfig<'a>>, Error> {
	let provider = props.get_ns("anthropic", "output_config");
	let provider = provider
		.map(|value| {
			value
				.as_object()
				.ok_or_else(|| provider_error("anthropic/output_config must be an object"))
		})
		.transpose()?;
	if let Some(provider) = provider {
		for key in provider.keys() {
			if !matches!(key.as_str(), "effort" | "task_budget" | "format") {
				super::report(
					unsupported,
					&format!("anthropic/output_config.{key}"),
					"unknown Anthropic output_config field was not sent",
					UnsupportedAction::Dropped,
				);
			}
		}
	}
	let effort = provider
		.and_then(|value| value.get("effort"))
		.map(|value| {
			value
				.as_str()
				.ok_or_else(|| provider_error("anthropic/output_config.effort must be a string"))
		})
		.transpose()?;
	if let Some(value) = effort
		&& !matches!(value, "low" | "medium" | "high" | "xhigh" | "max")
	{
		return Err(provider_error(
			"anthropic/output_config.effort must be low, medium, high, xhigh, or max",
		));
	}
	let task_budget = provider.and_then(|value| value.get("task_budget"));
	let format = if let Some(feature) = format {
		match &feature.value.kind {
			ResponseFormatKind::JsonSchema(schema) => {
				Some(OutputFormat::JsonSchema(JsonSchemaFormat {
					r#type: "json_schema",
					name:   &schema.name,
					schema: serde_json::from_slice(&schema.schema_json).map_err(super::json_error)?,
					strict: schema.strict,
				}))
			},
			ResponseFormatKind::Grammar(_) => {
				super::feature_report(
					unsupported,
					feature.on_unsupported,
					"response_format.grammar",
					"Anthropic Messages only supports JSON Schema structured output",
				)?;
				None
			},
			_ => {
				super::feature_report(
					unsupported,
					feature.on_unsupported,
					"response_format",
					"Anthropic Messages does not support this response format",
				)?;
				None
			},
		}
	} else {
		let native = props
			.get_ns("anthropic", "output_config.format")
			.or_else(|| provider.and_then(|value| value.get("format")));
		if let Some(value) = native
			&& !value.is_object()
		{
			return Err(provider_error("anthropic/output_config.format must be an object"));
		}
		native.map(OutputFormat::Native)
	};
	Ok((effort.is_some() || task_budget.is_some() || format.is_some()).then_some(OutputConfig {
		effort,
		task_budget,
		format,
	}))
}

fn metadata(meta: &'_ Option<RequestMeta>, props: &Props) -> Result<Option<Value>, Error> {
	let provider = props.get_ns("anthropic", "metadata");
	let mut object = match provider {
		Some(Value::Object(value)) => value.clone(),
		Some(_) => return Err(provider_error("anthropic/metadata must be an object")),
		None => Map::new(),
	};
	if let Some(meta) = meta
		&& !meta.initiator.is_empty()
	{
		object.insert("user_id".into(), Value::String(meta.initiator.to_string()));
	}
	Ok((!object.is_empty()).then_some(Value::Object(object)))
}

fn object_option<'a>(props: &'a Props, name: &'static str) -> Result<Option<&'a Value>, Error> {
	let Some(value) = props.get_ns("anthropic", name) else {
		return Ok(None);
	};
	if value.is_object() {
		Ok(Some(value))
	} else {
		Err(provider_error("Anthropic object-valued control had a non-object value"))
	}
}

fn container(props: &Props) -> Result<Option<&Value>, Error> {
	let Some(value) = props.get_ns("anthropic", "container") else {
		return Ok(None);
	};
	if value.is_string() || value.is_object() {
		Ok(Some(value))
	} else {
		Err(provider_error("anthropic/container must be a container id or object"))
	}
}

fn service_tier(props: &Props) -> Result<(Option<&str>, Option<&'static str>), Error> {
	let Some(value) = props.get_ns("anthropic", "service_tier") else {
		return Ok((None, None));
	};
	let value = value
		.as_str()
		.ok_or_else(|| provider_error("anthropic/service_tier must be a string"))?;
	match value {
		"auto" | "standard_only" => Ok((Some(value), None)),
		"priority" => Ok((None, Some("fast"))),
		_ => Err(provider_error("anthropic/service_tier must be auto, standard_only, or priority")),
	}
}

/// Returns whether the option is consumed by this codec or its header selector.
pub fn is_known_option(key: &str) -> bool {
	matches!(
		key,
		"anthropic/server_tools"
			| "anthropic/disable_parallel_tool_use"
			| "anthropic/parallel_tool_use"
			| "anthropic/eager_input_streaming"
			| "anthropic/tool_choice"
			| "anthropic/output_config"
			| "anthropic/output_config.format"
			| "anthropic/metadata"
			| "anthropic/context_management"
			| "anthropic/service_tier"
			| "anthropic/container"
			| "anthropic/betas"
			| "anthropic/version"
			| "anthropic/cache_control"
	)
}

fn provider_error(message: &'static str) -> Error {
	Error::Provider(message.into())
}
