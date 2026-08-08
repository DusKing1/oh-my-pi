//! Capability-learning layer behavior.

use std::{
	convert::Infallible,
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
	task::{Context, Poll},
};

use futures::StreamExt;
use omp_llm_error::{Classification, Feature, Kind};
use omp_llm_tower::{
	learn::{LearnLayer, RequestRepair, ScopeFn},
	recovery::classify_turn_error,
	select::{CredentialLease, Routed},
	testing::{Script, ScriptStream, error, invoke, kind_of, outcome, part_delta},
};
use omp_proto::inference::v1::{
	ChatParams, Sampling, ServiceTier, ServiceTierByFamily, TurnRequest, Value, ValueMap,
	turn_error, turn_event, value,
};
use tower::{Layer, Service, ServiceExt};
const STRICT_ERROR: &str = r#"HTTP 400 {"error":{"type":"invalid_request_error","message":"the compiled grammar is too large"}}"#;

#[derive(Default)]
struct TagRepair {
	calls:   AtomicUsize,
	enabled: bool,
}

impl TagRepair {
	const fn enabled() -> Self {
		Self { calls: AtomicUsize::new(0), enabled: true }
	}
}

impl RequestRepair for TagRepair {
	fn strip(
		&self,
		req: &TurnRequest,
		feature: Feature,
		_cls: &Classification,
	) -> Option<TurnRequest> {
		self.calls.fetch_add(1, Ordering::Relaxed);
		if !self.enabled || feature != Feature::StrictTools {
			return None;
		}
		let mut repaired = req.clone();
		let params = repaired.params.as_mut()?;
		if params.provider_options.is_some() {
			return None;
		}
		params.provider_options = Some(Default::default());

		Some(repaired)
	}
}
#[derive(Clone)]
struct RoutedScript(Script);

impl Service<Routed> for RoutedScript {
	type Error = Infallible;
	type Future = std::future::Ready<Result<ScriptStream, Infallible>>;
	type Response = ScriptStream;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, request: Routed) -> Self::Future {
		self.0.call(request.request)
	}
}

fn req(model: &str) -> TurnRequest {
	TurnRequest {
		turn_id: "turn".to_owned(),
		params: Some(ChatParams { model: model.to_owned(), ..ChatParams::default() }),
		..TurnRequest::default()
	}
}

fn is_tagged(req: &TurnRequest) -> bool {
	req.params
		.as_ref()
		.is_some_and(|params| params.provider_options.is_some())
}

fn bool_value(value: bool) -> Value {
	Value { kind: Some(value::Kind::Bool(value)) }
}

fn string_value(value: &str) -> Value {
	Value { kind: Some(value::Kind::String(value.to_owned())) }
}
#[tokio::test]
async fn reactive_repair_is_retried_and_next_service_is_proactive() {
	let classified = match error(turn_error::Kind::Upstream, STRICT_ERROR)
		.event
		.unwrap()
	{
		turn_event::Event::Error(err) => classify_turn_error(&err),
		_ => unreachable!(),
	};
	assert!(classified.kinds.has(Kind::FeatureUnsupported));
	assert_eq!(classified.feature, Some(Feature::StrictTools));

	let repair = Arc::new(TagRepair::enabled());
	let layer = LearnLayer::new(repair.clone());
	let first_script =
		Script::new([vec![error(turn_error::Kind::Upstream, STRICT_ERROR)], vec![outcome()]]);
	let first_calls = first_script.calls.clone();
	let mut first = layer.layer(first_script);
	let frames = first
		.ready()
		.await
		.unwrap()
		.call(req("endpoint/model"))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	let turn_event::Event::Attempt(attempt) = frames[0].event.as_ref().unwrap() else {
		unreachable!();
	};
	assert_eq!(attempt.number, 2);
	assert!(attempt.reason.contains("StrictTools"));
	{
		let calls = first_calls.lock();
		assert_eq!(calls.len(), 2);
		assert!(!is_tagged(&calls[0]));
		assert!(is_tagged(&calls[1]));
	}

	let next_script = Script::new([vec![outcome()]]);
	let next_calls = next_script.calls.clone();
	let mut next = layer.layer(next_script);
	let frames = next
		.ready()
		.await
		.unwrap()
		.call(req("endpoint/model"))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["outcome"]);
	{
		let calls = next_calls.lock();
		assert_eq!(calls.len(), 1);
		assert!(is_tagged(&calls[0]));
	}
	assert_eq!(repair.calls.load(Ordering::Relaxed), 2);

	let other_script = Script::new([vec![outcome()]]);
	let other_calls = other_script.calls.clone();
	let mut other = layer.layer(other_script);
	other
		.ready()
		.await
		.unwrap()
		.call(req("endpoint/other-model"))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert!(!is_tagged(&other_calls.lock()[0]), "learning is model-scoped");
}

#[tokio::test]
async fn unstrippable_feature_is_forwarded_and_stored_once() {
	let repair = Arc::new(TagRepair::default());
	let script = Script::new([vec![error(turn_error::Kind::Upstream, STRICT_ERROR)], vec![error(
		turn_error::Kind::Upstream,
		STRICT_ERROR,
	)]]);
	let calls = script.calls.clone();
	let mut service = LearnLayer::new(repair.clone()).layer(script);

	for turn in ["one", "two"] {
		let frames = service
			.ready()
			.await
			.unwrap()
			.call(TurnRequest { turn_id: turn.to_owned(), ..req("endpoint/model") })
			.await
			.unwrap()
			.collect::<Vec<_>>()
			.await;
		assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["error"]);
	}

	assert_eq!(calls.lock().len(), 2, "an unmodified request is never redispatched");
	assert_eq!(
		repair.calls.load(Ordering::Relaxed),
		3,
		"one proactive lookup proves the duplicate rejection did not duplicate the learned key",
	);
}

#[tokio::test]
async fn post_output_rejection_is_learned_without_redispatch() {
	let repair = Arc::new(TagRepair::enabled());
	let script = Script::new([
		vec![part_delta(), error(turn_error::Kind::Upstream, STRICT_ERROR)],
		vec![outcome()],
	]);
	let calls = script.calls.clone();
	let mut service = LearnLayer::new(repair.clone()).layer(script);

	let frames = service
		.ready()
		.await
		.unwrap()
		.call(req("endpoint/model"))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["part_delta", "error"]);
	assert_eq!(calls.lock().len(), 1);
	assert_eq!(repair.calls.load(Ordering::Relaxed), 0);

	let frames = service
		.ready()
		.await
		.unwrap()
		.call(TurnRequest { turn_id: "next".to_owned(), ..req("endpoint/model") })
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["outcome"]);
	let calls = calls.lock();
	assert_eq!(calls.len(), 2);
	assert!(is_tagged(&calls[1]), "the unsafe rejection still teaches the next turn");
	assert_eq!(repair.calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn invocation_is_a_hard_replay_bar_but_still_teaches_later_turns() {
	let repair = Arc::new(TagRepair::enabled());
	let script = Script::new([
		vec![invoke(), error(turn_error::Kind::Upstream, STRICT_ERROR)],
		vec![outcome()],
	]);
	let calls = script.calls.clone();
	let mut service = LearnLayer::new(repair.clone()).layer(script);

	let frames = service
		.ready()
		.await
		.unwrap()
		.call(req("endpoint/model"))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["invoke", "error"]);
	assert_eq!(calls.lock().len(), 1);
	assert_eq!(repair.calls.load(Ordering::Relaxed), 0);

	service
		.ready()
		.await
		.unwrap()
		.call(TurnRequest { turn_id: "next".to_owned(), ..req("endpoint/model") })
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	let calls = calls.lock();
	assert_eq!(calls.len(), 2);
	assert!(is_tagged(&calls[1]));
}

#[tokio::test]
async fn scope_isolates_endpoints_serving_the_same_model() {
	// A rejection from one endpoint must not disable the capability for
	// every endpoint serving the same model id. The gateway encodes the
	// resolved endpoint into the scope; here session_id stands in for it.
	let scope: ScopeFn = Arc::new(|req: &TurnRequest| {
		let params = req.params.as_ref()?;
		let session = params
			.meta
			.as_ref()
			.map(|meta| meta.session_id.as_str())
			.unwrap_or_default();
		Some(omp_core::SmolStr::new(format!("{}@{}", params.model, session)))
	});
	let scoped_req = |endpoint: &str| {
		let mut request = req("model");
		request.params.as_mut().unwrap().meta = Some(omp_proto::inference::v1::RequestMeta {
			session_id: endpoint.to_owned(),
			..Default::default()
		});
		request
	};

	let repair = Arc::new(TagRepair::enabled());
	let layer = LearnLayer::new(repair).with_scope(scope);

	// Endpoint A rejects strict tools; the fact is learned for A.
	let script_a =
		Script::new([vec![error(turn_error::Kind::Upstream, STRICT_ERROR)], vec![outcome()]]);
	let mut svc_a = layer.layer(script_a);
	let frames = svc_a
		.ready()
		.await
		.unwrap()
		.call(scoped_req("endpoint-a"))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);

	// Endpoint B serving the SAME model is untouched: no proactive strip.
	let script_b = Script::new([vec![outcome()]]);
	let calls_b = script_b.calls.clone();
	let mut svc_b = layer.layer(script_b);
	svc_b
		.ready()
		.await
		.unwrap()
		.call(scoped_req("endpoint-b"))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert!(!is_tagged(&calls_b.lock()[0]), "endpoint B must not inherit endpoint A's rejection");

	// Endpoint A itself IS proactively stripped on the next turn.
	let script_a2 = Script::new([vec![outcome()]]);
	let calls_a2 = script_a2.calls.clone();
	let mut svc_a2 = layer.layer(script_a2);
	svc_a2
		.ready()
		.await
		.unwrap()
		.call(scoped_req("endpoint-a"))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert!(is_tagged(&calls_a2.lock()[0]));
}

#[tokio::test]
async fn sampling_rejection_retries_then_proactively_preserves_only_output_limit() {
	const ERROR: &str = r#"HTTP 400 {"error":{"type":"invalid_request_error","message":"Unsupported parameter: 'temperature' is not supported with this model."}}"#;
	let sampling = Sampling {
		temperature:        Some(0.7),
		top_p:              Some(0.8),
		top_k:              Some(40),
		min_p:              Some(0.1),
		frequency_penalty:  Some(0.2),
		presence_penalty:   Some(0.3),
		stop:               vec!["done".to_owned()],
		max_output_tokens:  Some(4_096),
		repetition_penalty: Some(1.1),
		stop_present:       Some(true),
	};
	let request = |model: &str, turn: &str| {
		let mut request = req(model);
		request.turn_id = turn.to_owned();
		request.params.as_mut().unwrap().sampling = Some(sampling.clone());
		request
	};
	let assert_repaired = |request: &TurnRequest| {
		assert_eq!(
			request.params.as_ref().unwrap().sampling,
			Some(Sampling { max_output_tokens: Some(4_096), ..Sampling::default() }),
		);
	};

	let repair = Arc::new(TagRepair::default());
	let layer = LearnLayer::new(repair);
	let script = Script::new([vec![error(turn_error::Kind::Upstream, ERROR)], vec![outcome()]]);
	let calls = script.calls.clone();
	let mut service = layer.layer(script);
	let frames = service
		.ready()
		.await
		.unwrap()
		.call(request("provider/model", "reactive"))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	let calls = calls.lock();
	assert_eq!(calls[0].params.as_ref().unwrap().sampling, Some(sampling.clone()));
	assert_repaired(&calls[1]);
	drop(calls);

	let proactive = Script::new([vec![outcome()]]);
	let proactive_calls = proactive.calls.clone();
	let mut proactive_service = layer.layer(proactive);
	proactive_service
		.ready()
		.await
		.unwrap()
		.call(request("provider/model", "proactive"))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_repaired(&proactive_calls.lock()[0]);

	let isolated = Script::new([vec![outcome()]]);
	let isolated_calls = isolated.calls.clone();
	let mut isolated_service = layer.layer(isolated);
	isolated_service
		.ready()
		.await
		.unwrap()
		.call(request("provider/other-model", "isolated"))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(
		isolated_calls.lock()[0].params.as_ref().unwrap().sampling,
		Some(sampling.clone()),
		"learned sampling repair must remain model-scoped",
	);
}

#[tokio::test]
async fn thinking_signature_repair_sets_sticky_anthropic_option_only() {
	const ERROR: &str = r#"HTTP 400 {"error":{"type":"invalid_request_error","message":"Invalid signature in `thinking` block"}}"#;
	let request = || {
		let mut request = req("anthropic/claude");
		request.params.as_mut().unwrap().provider_options = Some(ValueMap {
			fields: [
				("anthropic/replay_unsigned_thinking".to_owned(), bool_value(true)),
				("anthropic/keep".to_owned(), string_value("value")),
			]
			.into_iter()
			.collect(),
		});
		request
	};
	let assert_repaired = |request: &TurnRequest| {
		let options = request
			.params
			.as_ref()
			.unwrap()
			.provider_options
			.as_ref()
			.unwrap();
		assert_eq!(
			options.fields.get("anthropic/replay_unsigned_thinking"),
			Some(&bool_value(false)),
		);
		assert_eq!(options.fields.get("anthropic/keep"), Some(&string_value("value")));
	};

	let layer = LearnLayer::new(Arc::new(TagRepair::default()));
	let script = Script::new([vec![error(turn_error::Kind::Upstream, ERROR)], vec![outcome()]]);
	let calls = script.calls.clone();
	let mut service = layer.layer(script);
	let frames = service
		.ready()
		.await
		.unwrap()
		.call(request())
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	assert_eq!(
		calls.lock()[0]
			.params
			.as_ref()
			.unwrap()
			.provider_options
			.as_ref()
			.unwrap()
			.fields
			.get("anthropic/replay_unsigned_thinking"),
		Some(&bool_value(true)),
	);
	assert_repaired(&calls.lock()[1]);

	let proactive = Script::new([vec![outcome()]]);
	let proactive_calls = proactive.calls.clone();
	let mut proactive_service = layer.layer(proactive);
	proactive_service
		.ready()
		.await
		.unwrap()
		.call(request())
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_repaired(&proactive_calls.lock()[0]);
}

#[tokio::test]
async fn fast_mode_repair_removes_only_anthropic_priority_controls() {
	const ERROR: &str = r#"HTTP 400 {"error":{"type":"invalid_request_error","message":"This model does not support the speed parameter."}}"#;
	let request = || {
		let mut request = req("anthropic/claude");
		let params = request.params.as_mut().unwrap();
		params.service_tier = ServiceTier::Priority as i32;
		params.service_tier_by_family = Some(ServiceTierByFamily {
			openai:    ServiceTier::Priority as i32,
			anthropic: ServiceTier::Priority as i32,
			google:    ServiceTier::Flex as i32,
		});
		params.provider_options = Some(ValueMap {
			fields: [
				("anthropic/service_tier".to_owned(), string_value("priority")),
				("anthropic/keep".to_owned(), bool_value(true)),
			]
			.into_iter()
			.collect(),
		});
		request
	};
	let assert_repaired = |request: &TurnRequest| {
		let params = request.params.as_ref().unwrap();
		assert_eq!(params.service_tier, ServiceTier::Unspecified as i32);
		let tiers = params.service_tier_by_family.as_ref().unwrap();
		assert_eq!(tiers.anthropic, ServiceTier::Unspecified as i32);
		assert_eq!(tiers.openai, ServiceTier::Priority as i32);
		assert_eq!(tiers.google, ServiceTier::Flex as i32);
		let options = params.provider_options.as_ref().unwrap();
		assert!(!options.fields.contains_key("anthropic/service_tier"));
		assert_eq!(options.fields.get("anthropic/keep"), Some(&bool_value(true)));
	};

	let layer = LearnLayer::new(Arc::new(TagRepair::default()));
	let script = Script::new([vec![error(turn_error::Kind::Upstream, ERROR)], vec![outcome()]]);
	let calls = script.calls.clone();
	let mut service = layer.layer(script);
	let frames = service
		.ready()
		.await
		.unwrap()
		.call(request())
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	assert_eq!(calls.lock()[0].params.as_ref().unwrap().service_tier, ServiceTier::Priority as i32,);
	assert_repaired(&calls.lock()[1]);

	let proactive = Script::new([vec![outcome()]]);
	let proactive_calls = proactive.calls.clone();
	let mut proactive_service = layer.layer(proactive);
	proactive_service
		.ready()
		.await
		.unwrap()
		.call(request())
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_repaired(&proactive_calls.lock()[0]);
}

#[tokio::test]
async fn learned_repairs_are_isolated_by_provider_and_account() {
	const ERROR: &str = r#"HTTP 400 {"error":{"type":"invalid_request_error","message":"Invalid signature in `thinking` block"}}"#;
	let routed = |provider: &str, account: u64, turn: &str| {
		let mut request = req("claude");
		request.turn_id = turn.to_owned();
		Routed::new(request, Some(CredentialLease::new(provider, account, 1)), None)
	};
	let is_disabled = |request: &TurnRequest| {
		request
			.params
			.as_ref()
			.and_then(|params| params.provider_options.as_ref())
			.and_then(|options| options.fields.get("anthropic/replay_unsigned_thinking"))
			.and_then(|value| value.kind.as_ref())
			.is_some_and(|kind| matches!(kind, value::Kind::Bool(false)))
	};

	let layer = LearnLayer::new(Arc::new(TagRepair::default()));
	let teaching_script =
		Script::new([vec![error(turn_error::Kind::Upstream, ERROR)], vec![outcome()]]);
	let teaching_calls = teaching_script.calls.clone();
	let mut teaching = layer.layer(RoutedScript(teaching_script));
	teaching
		.ready()
		.await
		.unwrap()
		.call(routed("anthropic", 7, "teach"))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert!(is_disabled(&teaching_calls.lock()[1]));

	for (provider, account, label) in
		[("anthropic", 8, "sibling-account"), ("anthropic-proxy", 7, "sibling-provider")]
	{
		let script = Script::new([vec![outcome()]]);
		let calls = script.calls.clone();
		let mut service = layer.layer(RoutedScript(script));
		service
			.ready()
			.await
			.unwrap()
			.call(routed(provider, account, label))
			.await
			.unwrap()
			.collect::<Vec<_>>()
			.await;
		assert!(!is_disabled(&calls.lock()[0]), "{label} inherited another scope's repair");
	}

	let proactive_script = Script::new([vec![outcome()]]);
	let proactive_calls = proactive_script.calls.clone();
	let mut proactive = layer.layer(RoutedScript(proactive_script));
	proactive
		.ready()
		.await
		.unwrap()
		.call(routed("anthropic", 7, "proactive"))
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert!(is_disabled(&proactive_calls.lock()[0]));
}
