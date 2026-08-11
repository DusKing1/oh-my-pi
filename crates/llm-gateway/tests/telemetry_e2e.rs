//! End-to-end compatibility proof for gateway chat telemetry.

use std::{collections::BTreeMap, sync::Arc};

use futures::{StreamExt, stream};
use omp_core::SmolStr;
use omp_llm_catalog::{
	models::{Availability, Modality, ModelCard, ModelCatalog, Source},
	provider::Facet as CatalogFacet,
	registry::{CredentialView, Registry},
};
use omp_llm_gateway::{
	context::ContextStore,
	turn::{ChatResolver, ChatRoute, TurnEngine},
};
use omp_llm_types::{
	Accuracy, ChatOutcome, ChatRequest, Cost, Item, ItemKind, Message, Part, Props, Role,
	StopReason, Thread, TurnEvent, Usage,
	facet::{Chat, Error as FacetError, Executor},
};
use omp_proto::inference::v1 as pb;
use omp_telemetry::{config::TelemetryConfig, metrics::MetricRecorder};
use opentelemetry::global;
use opentelemetry_sdk::{
	metrics::{
		InMemoryMetricExporter, PeriodicReader, SdkMeterProvider,
		data::{AggregatedMetrics, MetricData},
	},
	trace::{InMemorySpanExporter, SdkTracerProvider},
};
use parking_lot::RwLock;
use smallvec::smallvec;

#[derive(Clone, Copy)]
struct Available;

impl CredentialView for Available {
	fn availability(&self, _provider: &str) -> Availability {
		Availability::Available
	}
}

struct SuccessfulChat;

#[async_trait::async_trait]
impl Chat for SuccessfulChat {
	async fn turn(
		&self,
		_request: ChatRequest,
		_executor: Option<Arc<dyn Executor>>,
	) -> Result<futures::stream::BoxStream<'static, TurnEvent>, FacetError> {
		Ok(stream::iter([TurnEvent::Outcome(
			ChatOutcome::builder()
				.output(vec![item(Role::Assistant, "answer")])
				.stop(StopReason::EndTurn)
				.usage(
					Usage::builder()
						.input_tokens(3)
						.output_tokens(2)
						.cache_read_tokens(1)
						.cache_write_tokens(1)
						.accuracy(Accuracy::Exact)
						.detail(Props::default())
						.build(),
				)
				.cost(Cost::builder().nanos_usd(2_500_000).estimated(true).build())
				.unsupported(Vec::new())
				.provider(SmolStr::new_static("test"))
				.model(SmolStr::new_static("model"))
				.props(Props::default())
				.build(),
		)])
		.boxed())
	}
}

fn item(role: Role, text: &'static str) -> Item {
	Item::builder()
		.seq(0)
		.kind(ItemKind::Message(
			Message::builder()
				.role(role)
				.parts(vec![Part::Text(SmolStr::new_static(text))])
				.build(),
		))
		.props(Props::default())
		.build()
}

fn model_card() -> ModelCard {
	ModelCard::builder()
		.id(SmolStr::new_static("test/model"))
		.provider(SmolStr::new_static("test"))
		.model(SmolStr::new_static("model"))
		.name(SmolStr::new_static("Model"))
		.family(SmolStr::new_static("test"))
		.facets(smallvec![CatalogFacet::Chat])
		.inputs(smallvec![Modality::Text])
		.outputs(smallvec![Modality::Text])
		.reasoning(false)
		.efforts(smallvec![])
		.context_window(4_096)
		.max_output_tokens(1_024)
		.pricing(smallvec![])
		.availability(Availability::Available)
		.source(Source::Configured)
		.blocked_until_ms(0)
		.deprecated(false)
		.updated_at_ms(0)
		.props(Props::default())
		.effort_routing(BTreeMap::new())
		.build()
}

fn engine(config: TelemetryConfig) -> TurnEngine {
	let catalog = ModelCatalog::new(vec![model_card()]);
	let registry = Arc::new(RwLock::new(Registry::new(&catalog, Arc::new(Available))));
	let resolver = Arc::new(ChatResolver::new(registry));
	resolver.register(ChatRoute {
		provider:          SmolStr::new_static("test"),
		credential_id:     SmolStr::new_static("cred-a"),
		requires_executor: false,
		chat:              Arc::new(SuccessfulChat),
	});
	TurnEngine::with_telemetry(
		Arc::new(ContextStore::default()),
		resolver,
		Arc::new(MetricRecorder::new()),
		Arc::new(config),
	)
}

fn open(turn_id: &str) -> pb::TurnFrame {
	pb::TurnFrame {
		frame: Some(pb::turn_frame::Frame::Open(pb::TurnRequest {
			turn_id:  turn_id.into(),
			input:    Some(pb::turn_request::Input::Seed(pb::Seed {
				context_id: format!("context-{turn_id}"),
				thread:     Some(
					Thread::builder()
						.items(vec![item(Role::User, "hello")])
						.build()
						.into(),
				),
			})),
			params:   Some(pb::ChatParams { model: "test/model".into(), ..pb::ChatParams::default() }),
			executor: None,
			props:    None,
		})),
	}
}

async fn run_turn(engine: &TurnEngine, turn_id: &str) {
	let mut events = engine
		.turn_frames(stream::iter([Ok(open(turn_id))]))
		.await
		.expect("turn accepted");
	let mut succeeded = false;
	while let Some(event) = events.next().await {
		let event: TurnEvent = event
			.expect("transport success")
			.try_into()
			.expect("canonical event");
		if matches!(event, TurnEvent::Outcome(_)) {
			succeeded = true;
		}
		assert!(!matches!(event, TurnEvent::Error(_)), "turn must remain fail-open");
	}
	assert!(succeeded, "turn completed successfully");
}

#[tokio::test]
async fn turn_exports_pi_contract_and_telemetry_failures_are_open() {
	// SAFETY: this integration binary contains one test, so no other thread can
	// concurrently inspect or mutate the process telemetry environment.
	unsafe {
		std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1");
		std::env::set_var("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf");
	}
	assert!(omp_telemetry::export::init(), "broken collector still initializes fail-open");
	run_turn(&engine(TelemetryConfig::default()), "broken-collector").await;
	omp_telemetry::export::flush();
	omp_telemetry::export::shutdown();
	// SAFETY: this integration binary contains one test, so no other thread can
	// concurrently inspect or mutate the process telemetry environment.
	unsafe {
		std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
		std::env::remove_var("OTEL_EXPORTER_OTLP_PROTOCOL");
	}

	let span_exporter = InMemorySpanExporter::default();
	let tracer_provider = SdkTracerProvider::builder()
		.with_simple_exporter(span_exporter.clone())
		.build();
	global::set_tracer_provider(tracer_provider.clone());
	let metric_exporter = InMemoryMetricExporter::default();
	let meter_provider = SdkMeterProvider::builder()
		.with_reader(PeriodicReader::builder(metric_exporter.clone()).build())
		.build();
	global::set_meter_provider(meter_provider.clone());

	let config = TelemetryConfig {
		on_span_start: Some(Arc::new(|_| panic!("host hook failure"))),
		..TelemetryConfig::default()
	};
	run_turn(&engine(config), "exported").await;
	tracer_provider.force_flush().expect("flush spans");
	meter_provider.force_flush().expect("flush metrics");

	let spans = span_exporter.get_finished_spans().expect("finished spans");
	let chat = spans
		.iter()
		.find(|span| span.name == "chat model")
		.expect("chat span exported");
	for key in [
		"gen_ai.operation.name",
		"gen_ai.request.model",
		"gen_ai.provider.name",
		"gen_ai.response.model",
		"gen_ai.usage.input_tokens",
		"gen_ai.usage.output_tokens",
		"omp.gen_ai.cost.estimated_usd",
	] {
		assert!(
			chat
				.attributes
				.iter()
				.any(|attribute| attribute.key.as_str() == key),
			"missing {key}"
		);
	}

	let string_attribute = |key: &str| {
		chat
			.attributes
			.iter()
			.find(|attribute| attribute.key.as_str() == key)
			.map(|attribute| attribute.value.as_str().into_owned())
	};
	assert_eq!(string_attribute("gen_ai.operation.name").as_deref(), Some("chat"));
	assert_eq!(string_attribute("gen_ai.request.model").as_deref(), Some("model"));
	assert_eq!(string_attribute("gen_ai.provider.name").as_deref(), Some("test"));
	let batches = metric_exporter
		.get_finished_metrics()
		.expect("finished metrics");
	let metrics = batches
		.iter()
		.flat_map(|batch| batch.scope_metrics())
		.flat_map(|scope| scope.metrics());
	let mut token_types = Vec::new();
	let mut saw_cost = false;
	for metric in metrics {
		match metric.name() {
			"gen_ai.client.token.usage" => {
				let AggregatedMetrics::U64(MetricData::Histogram(histogram)) = metric.data() else {
					panic!("token usage must be a u64 histogram");
				};
				for point in histogram.data_points() {
					assert_eq!(point.count(), 1, "one recording per token type");
					let token_type = point
						.attributes()
						.find(|attribute| attribute.key.as_str() == "gen_ai.token.type")
						.expect("token type attribute");
					token_types.push(token_type.value.as_str().into_owned());
				}
			},
			"omp.agent.chat.cost.estimated_usd" => saw_cost = true,
			_ => {},
		}
	}
	token_types.sort();
	assert_eq!(token_types, ["cache_read_input", "cache_write_input", "input", "output", "total"]);
	assert!(saw_cost, "estimated cost metric exported");
}
