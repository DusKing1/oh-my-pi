//! Recorded-wire recovery corpus for `OpenAI` Responses continuation failures.

use std::{
	collections::VecDeque,
	convert::Infallible,
	future::Future,
	path::Path,
	pin::Pin,
	sync::{Arc, Mutex},
	task::{Context, Poll},
};

use bytes::Bytes;
use futures::StreamExt;
use http::{Request, Response, StatusCode, header};
use http_body_util::{BodyExt, Full};
use omp_llm_catalog::provider::load_builtin;
use omp_llm_egress::client::Body;
use omp_llm_error::RetryBudget;
use omp_llm_tower::{
	provider::{ProviderAttempt, ProviderRoute},
	recovery::{Recovery, RecoveryConfig},
	select::Routed,
};
use omp_llm_types::{
	CallId, ChatRequest, Item, ItemKind, Message, Part, Props, Role, Thread, ToolCall, ToolResult,
	TurnEvent as NativeTurnEvent,
};
use omp_proto::inference::v1::{TurnRequest, turn_event, turn_request, value};
use serde_json::{Value, json};
use tower::{Service, ServiceExt};

const SUCCESS_SSE: &str = concat!(
	"event: response.completed\n",
	"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_repaired\",",
	"\"model\":\"gpt-5\",\"status\":\"completed\",\"output\":[],",
	"\"usage\":{\"input_tokens\":12,\"output_tokens\":3,\"total_tokens\":15}}}\n\n",
);

#[derive(Clone)]
struct WireScript {
	state: Arc<Mutex<WireState>>,
}

struct WireState {
	responses:       VecDeque<(StatusCode, Bytes)>,
	requests:        Vec<Value>,
	canonical_calls: Vec<TurnRequest>,
}

impl Service<Request<Body>> for WireScript {
	type Error = Infallible;
	type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;
	type Response = Response<Full<Bytes>>;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, request: Request<Body>) -> Self::Future {
		let state = Arc::clone(&self.state);
		Box::pin(async move {
			let bytes = request
				.into_body()
				.collect()
				.await
				.expect("infallible request body")
				.to_bytes();
			let request = serde_json::from_slice(&bytes).expect("Responses codec emitted JSON");
			let (status, body) = {
				let mut state = state.lock().expect("wire fixture state");
				state.requests.push(request);
				state
					.responses
					.pop_front()
					.expect("fixture response for every attempt")
			};
			Ok(Response::builder()
				.status(status)
				.header(header::CONTENT_TYPE, "text/event-stream")
				.body(Full::new(body))
				.expect("fixture HTTP response"))
		})
	}
}

#[derive(Clone)]
struct CaptureAttempts<S> {
	inner: S,
	state: Arc<Mutex<WireState>>,
}

impl<S> Service<Routed> for CaptureAttempts<S>
where
	S: Service<Routed>,
{
	type Error = S::Error;
	type Future = S::Future;
	type Response = S::Response;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, request: Routed) -> Self::Future {
		self
			.state
			.lock()
			.expect("wire fixture state")
			.canonical_calls
			.push(request.request.clone());
		self.inner.call(request)
	}
}

#[tokio::test]
async fn responses_recovery_wire_corpus() {
	let directory =
		Path::new(env!("CARGO_MANIFEST_DIR")).join("../llm-openai/tests/fixtures/openai_responses");
	let mut fixtures = std::fs::read_dir(&directory)
		.expect("Responses fixture directory")
		.filter_map(Result::ok)
		.map(|entry| entry.path())
		.filter(|path| {
			path
				.file_name()
				.and_then(|name| name.to_str())
				.is_some_and(|name| name.starts_with("recovery.") && name.ends_with(".json"))
		})
		.collect::<Vec<_>>();
	fixtures.sort();
	assert!(!fixtures.is_empty(), "recovery fixture corpus must not be empty");

	for path in fixtures {
		run_fixture(&path).await;
	}
}

async fn run_fixture(path: &Path) {
	let fixture: Value = serde_json::from_slice(&std::fs::read(path).expect("read fixture"))
		.expect("valid recovery fixture JSON");
	let responses = fixture["responses"]
		.as_array()
		.expect("responses array")
		.iter()
		.map(|response| {
			let status =
				StatusCode::from_u16(response["status"].as_u64().expect("response status") as u16)
					.expect("valid response status");
			let body = if response["body"] == "success" {
				Bytes::from_static(SUCCESS_SSE.as_bytes())
			} else if let Some(raw) = response["raw_body"].as_str() {
				Bytes::copy_from_slice(raw.as_bytes())
			} else {
				Bytes::from(serde_json::to_vec(&response["body"]).expect("serialize error body"))
			};
			(status, body)
		})
		.collect::<VecDeque<_>>();
	let state = Arc::new(Mutex::new(WireState {
		responses,
		requests: Vec::new(),
		canonical_calls: Vec::new(),
	}));
	let egress = WireScript { state: Arc::clone(&state) };
	let mut provider = load_builtin()
		.expect("builtin providers")
		.remove("openai")
		.expect("OpenAI");
	provider.base_url = "https://fixture.invalid/v1".into();
	let attempt = CaptureAttempts {
		inner: ProviderAttempt::new(provider, ProviderRoute::default(), egress)
			.expect("Responses provider attempt"),
		state: Arc::clone(&state),
	};
	let config =
		RecoveryConfig { budget: RetryBudget::new(3, 1, 2, 1_000_000), ..RecoveryConfig::default() };
	let request: TurnRequest = fixture_request(
		fixture["thread"].as_str().expect("thread kind"),
		fixture["boundary"].as_u64().expect("continuation boundary"),
	)
	.into();
	let events = Recovery::new(attempt, config)
		.oneshot(Routed::new(request, None, None))
		.await
		.expect("dispatch fixture")
		.collect::<Vec<_>>()
		.await;

	let state = state.lock().expect("wire fixture state");
	let expected_attempts = fixture["expected_attempts"]
		.as_u64()
		.expect("expected attempts") as usize;
	assert_eq!(state.requests.len(), expected_attempts, "{} attempt bound", path.display());
	assert_eq!(
		state.requests[0]["previous_response_id"],
		"resp_stale",
		"{} first anchor",
		path.display()
	);
	assert_eq!(
		state.requests[0]["input"],
		fixture["expected_first_input"],
		"{} request slicing",
		path.display()
	);
	if expected_attempts == 2 {
		assert!(
			state.requests[1].get("previous_response_id").is_none(),
			"{} replay anchor scrub",
			path.display()
		);
		assert_eq!(
			state.requests[1]["input"],
			fixture["expected_replay_input"],
			"{} full-thread replay",
			path.display()
		);
		assert!(!contains_server_id(&state.requests[1]), "{} replay server-id scrub", path.display());
		assert_repaired_canonical_ids(&state.canonical_calls[1], path);
	}
	let has_outcome = events
		.iter()
		.any(|event| matches!(event.event, Some(turn_event::Event::Outcome(_))));
	assert_eq!(
		has_outcome,
		fixture["expect_outcome"]
			.as_bool()
			.expect("outcome expectation"),
		"{} canonical terminal",
		path.display()
	);
	if has_outcome {
		assert!(
			!events
				.iter()
				.any(|event| matches!(event.event, Some(turn_event::Event::Error(_)))),
			"{} successful repair emitted error",
			path.display()
		);
		let outcome = events
			.iter()
			.cloned()
			.filter_map(|event| NativeTurnEvent::try_from(event).ok())
			.find_map(|event| match event {
				NativeTurnEvent::Outcome(outcome) => Some(outcome),
				_ => None,
			})
			.expect("canonical outcome");
		assert_eq!(
			outcome.props.get_ns("openai", "response_id"),
			Some(&json!("resp_repaired")),
			"{} authoritative response identity",
			path.display()
		);
	} else {
		assert!(
			events
				.iter()
				.any(|event| matches!(event.event, Some(turn_event::Event::Error(_)))),
			"{} must retain terminal error",
			path.display()
		);
	}
}

fn fixture_request(thread: &str, boundary: u64) -> ChatRequest {
	let items = match thread {
		"messages" => vec![user("Earlier context."), user("Continue.")],
		"orphan_tool" => orphan_tool_thread(),
		other => panic!("unknown fixture thread {other}"),
	};
	let mut options = Props::default();
	options.insert_ns("openai", "previous_response_id", json!("resp_stale"));
	options.insert_ns("openai", "previous_response_item_count", json!(boundary));
	ChatRequest::builder()
		.model("gpt-5".into())
		.thread(Thread::builder().items(items).build())
		.tools(Vec::new())
		.provider_options(options)
		.build()
}

fn user(text: &'static str) -> Item {
	Item::builder()
		.seq(0)
		.kind(ItemKind::Message(
			Message::builder()
				.role(Role::User)
				.parts(vec![Part::Text(text.into())])
				.build(),
		))
		.props(Props::default())
		.build()
}

fn orphan_tool_thread() -> Vec<Item> {
	let id: CallId = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
		.parse()
		.expect("fixture call id");
	let mut call_props = Props::default();
	call_props.insert_ns("openai", "call_id", json!("call_weather"));
	call_props.insert_ns("openai", "item_id", json!("fc_server_stale"));
	call_props.insert_ns(
		"openai",
		"server_tool_item",
		json!({"id":"srv_nested_stale","type":"web_search_call"}),
	);
	let call = Item::builder()
		.seq(0)
		.kind(ItemKind::ToolCall(
			ToolCall::builder()
				.id(id)
				.name("get_weather".into())
				.args_json(Bytes::from_static(br#"{"city":"Paris"}"#))
				.thought_signature(Bytes::new())
				.build(),
		))
		.props(call_props)
		.build();
	let mut result_props = Props::default();
	result_props.insert_ns("openai", "call_id", json!("call_weather"));
	let result = Item::builder()
		.seq(0)
		.kind(ItemKind::ToolResult(
			ToolResult::builder()
				.call_id(id)
				.name("get_weather".into())
				.parts(vec![Part::Text("Sunny".into())])
				.is_error(false)
				.build(),
		))
		.props(result_props)
		.build();
	vec![user("Find the weather."), call, result, user("Summarize.")]
}

fn assert_repaired_canonical_ids(request: &TurnRequest, path: &Path) {
	let options = request
		.params
		.as_ref()
		.and_then(|params| params.provider_options.as_ref())
		.expect("provider options retained");
	assert!(
		!options.fields.contains_key("openai/previous_response_id"),
		"{} canonical anchor scrub",
		path.display()
	);
	assert!(
		!options
			.fields
			.contains_key("openai/previous_response_item_count"),
		"{} canonical boundary scrub",
		path.display()
	);
	let Some(turn_request::Input::Seed(seed)) = &request.input else {
		panic!("{} replay must retain the full seed", path.display());
	};
	for item in &seed.thread.as_ref().expect("canonical thread").items {
		let Some(props) = &item.props else {
			continue;
		};
		assert!(
			!props.fields.contains_key("openai/item_id"),
			"{} canonical item id scrub",
			path.display()
		);
		if let Some(value) = props.fields.get("openai/server_tool_item")
			&& let Some(value::Kind::Map(server_item)) = &value.kind
		{
			assert!(
				!server_item.fields.contains_key("id"),
				"{} nested server item id scrub",
				path.display()
			);
		}
	}
}

fn contains_server_id(value: &Value) -> bool {
	match value {
		Value::Object(object) => object.iter().any(|(key, value)| {
			(key == "id" && value.as_str().is_some_and(|id| id.contains("stale")))
				|| contains_server_id(value)
		}),
		Value::Array(array) => array.iter().any(contains_server_id),
		_ => false,
	}
}
