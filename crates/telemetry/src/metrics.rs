//! Wire-compatible metric instruments and recording helpers.

use std::sync::Arc;

use omp_core::SmolStr;
use opentelemetry::{
	KeyValue, global,
	metrics::{Counter, Histogram},
};
use smallvec::{SmallVec, smallvec};

use crate::{
	collector::{RunCoverage, RunSummary, Usage},
	semconv::{METER_NAME, TokenType},
};

/// Optional agent identity attached to chat-usage series.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricAgent {
	/// Stable agent identifier, when configured.
	pub id:   Option<SmolStr>,
	/// Human-readable agent name, when configured.
	pub name: Option<SmolStr>,
}

/// A completed chat's metric inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatUsageMetric {
	/// Provider name, when known.
	pub provider:       Option<SmolStr>,
	/// Requested model identifier.
	pub model:          SmolStr,
	/// Response service tier, when supplied.
	pub service_tier:   Option<SmolStr>,
	/// Optional agent identity.
	pub agent:          Option<MetricAgent>,
	/// Token usage buckets.
	pub usage:          Usage,
	/// Provider usage-accuracy vocabulary (`provider`, `estimated`, or `mixed`).
	pub usage_accuracy: SmolStr,
	/// Estimated USD cost, when available.
	pub cost_usd:       Option<f64>,
}

/// The nine instruments emitted by pi's coding-agent telemetry exporter.
#[derive(Clone, Debug)]
pub struct MetricRecorder {
	token_usage:      Histogram<u64>,
	chat_cost_usd:    Counter<f64>,
	runs:             Counter<u64>,
	steps:            Counter<u64>,
	chat_calls:       Counter<u64>,
	chat_duration_ms: Histogram<f64>,
	tool_calls:       Counter<u64>,
	tool_duration_ms: Histogram<f64>,
	errors:           Counter<u64>,
}

impl Default for MetricRecorder {
	fn default() -> Self {
		Self::new()
	}
}

impl MetricRecorder {
	/// Creates all instruments from the global meter named by pi.
	#[must_use]
	pub fn new() -> Self {
		let meter = global::meter(METER_NAME);
		Self {
			token_usage:      meter
				.u64_histogram("gen_ai.client.token.usage")
				.with_description("Token usage reported by GenAI chat calls.")
				.with_unit("{token}")
				.build(),
			chat_cost_usd:    meter
				.f64_counter("omp.agent.chat.cost.estimated_usd")
				.with_description("Estimated USD cost for completed chat calls.")
				.with_unit("USD")
				.build(),
			runs:             meter
				.u64_counter("omp.agent.runs")
				.with_description("Completed agent runs.")
				.with_unit("{run}")
				.build(),
			steps:            meter
				.u64_counter("omp.agent.steps")
				.with_description("Agent loop steps completed inside a run.")
				.with_unit("{step}")
				.build(),
			chat_calls:       meter
				.u64_counter("omp.agent.chat.calls")
				.with_description("Chat calls completed inside agent runs.")
				.with_unit("{call}")
				.build(),
			chat_duration_ms: meter
				.f64_histogram("omp.agent.chat.duration")
				.with_description("Total chat latency observed in an agent run.")
				.with_unit("ms")
				.build(),
			tool_calls:       meter
				.u64_counter("omp.agent.tool.calls")
				.with_description("Tool calls completed inside agent runs.")
				.with_unit("{call}")
				.build(),
			tool_duration_ms: meter
				.f64_histogram("omp.agent.tool.duration")
				.with_description("Total tool latency observed in an agent run.")
				.with_unit("ms")
				.build(),
			errors:           meter
				.u64_counter("omp.agent.errors")
				.with_description("Errors observed in chat and tool execution.")
				.with_unit("{error}")
				.build(),
		}
	}

	/// Records all positive token buckets and a positive available cost for one
	/// chat.
	pub fn record_chat_usage(&self, event: &ChatUsageMetric) {
		let mut attrs: SmallVec<KeyValue, 8> = smallvec![
			KeyValue::new("gen_ai.operation.name", "chat"),
			string_attr("gen_ai.request.model", &event.model),
			string_attr("omp.gen_ai.usage.accuracy", &event.usage_accuracy),
		];
		if let Some(provider) = &event.provider {
			attrs.push(string_attr("gen_ai.provider.name", provider));
		}
		if let Some(service_tier) = &event.service_tier {
			attrs.push(string_attr("gen_ai.response.service_tier", service_tier));
		}
		if let Some(agent) = &event.agent {
			if let Some(id) = &agent.id {
				attrs.push(string_attr("omp.gen_ai.agent.id", id));
			}
			if let Some(name) = &agent.name {
				attrs.push(string_attr("omp.gen_ai.agent.name", name));
			}
		}

		self.record_token(event.usage.input, &attrs, TokenType::Input);
		self.record_token(event.usage.output, &attrs, TokenType::Output);
		self.record_token(event.usage.total, &attrs, TokenType::Total);
		self.record_token(event.usage.cached_input, &attrs, TokenType::CacheReadInput);
		self.record_token(event.usage.cache_write, &attrs, TokenType::CacheWriteInput);
		self.record_token(event.usage.reasoning_output, &attrs, TokenType::ReasoningOutput);
		if let Some(cost) = event.cost_usd.filter(|cost| *cost > 0.0) {
			self.chat_cost_usd.add(cost, &attrs);
		}
	}

	/// Records one completed run using pi's common run-level attribute set.
	pub fn record_run(&self, summary: &RunSummary, coverage: &RunCoverage) {
		let run_attrs: SmallVec<KeyValue, 5> = smallvec![
			count_attr("omp.agent.models_used.count", coverage.models_used.len()),
			count_attr("omp.agent.providers_used.count", coverage.providers_used.len()),
			count_attr("omp.agent.tools_available.count", coverage.tools_available.len()),
			count_attr("omp.agent.tools_invoked.count", coverage.tools_invoked.len()),
			count_attr("omp.agent.tools_unused.count", coverage.tools_unused.len()),
		];

		self.runs.add(1, &run_attrs);
		if summary.step_count > 0 {
			self.steps.add(summary.step_count, &run_attrs);
		}
		if summary.chats.total_latency_ms > 0.0 {
			self
				.chat_duration_ms
				.record(summary.chats.total_latency_ms, &run_attrs);
		}
		for (reason, count) in &summary.chats.by_stop_reason {
			if *count > 0 {
				let attrs = with_string_attr(&run_attrs, "gen_ai.response.finish_reason", reason);
				self.chat_calls.add(*count, &attrs);
			}
		}
		for (tool_name, counters) in &summary.tools.by_name {
			let tool_attrs = with_string_attr(&run_attrs, "gen_ai.tool.name", tool_name);
			if counters.total_latency_ms > 0.0 {
				self
					.tool_duration_ms
					.record(counters.total_latency_ms, &tool_attrs);
			}
			self.record_tool_status(counters.ok, &tool_attrs, "ok");
			self.record_tool_status(counters.error, &tool_attrs, "error");
			self.record_tool_status(counters.skipped, &tool_attrs, "skipped");
			self.record_tool_status(counters.blocked, &tool_attrs, "blocked");
			self.record_tool_status(counters.timeout, &tool_attrs, "timeout");
			self.record_tool_status(counters.aborted, &tool_attrs, "aborted");
		}
		for (error_type, count) in &summary.errors.by_type {
			if *count > 0 {
				let attrs = with_string_attr(&run_attrs, "error.type", error_type);
				self.errors.add(*count, &attrs);
			}
		}
	}

	fn record_token(&self, value: u64, base_attrs: &[KeyValue], token_type: TokenType) {
		if value == 0 {
			return;
		}
		let mut attrs: SmallVec<KeyValue, 9> = base_attrs.iter().cloned().collect();
		attrs.push(KeyValue::new("gen_ai.token.type", token_type.as_str()));
		self.token_usage.record(value, &attrs);
	}

	fn record_tool_status(&self, count: u64, tool_attrs: &[KeyValue], status: &'static str) {
		if count == 0 {
			return;
		}
		let mut attrs: SmallVec<KeyValue, 7> = tool_attrs.iter().cloned().collect();
		attrs.push(KeyValue::new("omp.tool.status", status));
		self.tool_calls.add(count, &attrs);
	}
}

fn string_attr(key: &'static str, value: &SmolStr) -> KeyValue {
	KeyValue::new(key, Arc::<str>::from(value.as_str()))
}

fn count_attr(key: &'static str, value: usize) -> KeyValue {
	KeyValue::new(key, i64::try_from(value).unwrap_or(i64::MAX))
}

fn with_string_attr<const N: usize>(
	attrs: &SmallVec<KeyValue, N>,
	key: &'static str,
	value: &SmolStr,
) -> SmallVec<KeyValue, 7> {
	let mut extended: SmallVec<KeyValue, 7> = attrs.iter().cloned().collect();
	extended.push(string_attr(key, value));
	extended
}
