//! OpenAI Codex transport fixtures and protocol contract tests.

use bytes::Bytes;
use omp_llm_openai::{
	CodexAttestation, CodexAttestationSignals, CodexAttestor, CodexContinuationState,
	CodexCredentialMetadata, CodexDeviceCheckResult, CodexDeviceToken, CodexFallbackAction,
	CodexFrameDisposition, CodexFrameRouter, CodexHeaderContext, CodexReplaySafety,
	CodexRequestIdentity, CodexWebSocketFailure, CodexWireTransport, OpenAiCodexCodec,
	apply_codex_client_metadata, build_codex_attestation, build_codex_header_plan,
	classify_codex_fallback, classify_codex_websocket_failure, transform_codex_request,
};
use omp_llm_transport::{DecodeState, Frame, Transport};
use omp_llm_types::TurnEvent;
use serde_json::Value;

#[test]
fn responses_lite_matches_the_codex_fixture() {
	let mut request: Value =
		serde_json::from_slice(include_bytes!("fixtures/openai_codex/request.responses_lite.json"))
			.unwrap();
	let expected: Value =
		serde_json::from_slice(include_bytes!("fixtures/openai_codex/expect.responses_lite.json"))
			.unwrap();
	transform_codex_request(&mut request, true).unwrap();
	assert_eq!(request, expected);
}

#[test]
fn header_plan_uses_subscription_fingerprint_without_exposing_secrets() {
	let identity = CodexRequestIdentity {
		installation_id: "install-secret".into(),
		session_id:      "session-secret".into(),
		thread_id:       "thread-secret".into(),
		window_id:       "window-secret".into(),
		turn_id:         "turn-secret".into(),
		turn_metadata:   "{\"turn_id\":\"turn-secret\"}".into(),
	};
	let credential = CodexCredentialMetadata { account_id: Some("account-secret".into()) };
	let attestation = CodexAttestation::new(Bytes::from_static(b"attestation-secret")).unwrap();
	let plan = build_codex_header_plan(&CodexHeaderContext {
		transport:      CodexWireTransport::WebSocket,
		identity:       Some(&identity),
		credential:     &credential,
		attestation:    Some(&attestation),
		turn_state:     Some("turn-state-secret"),
		models_etag:    Some("etag-secret"),
		responses_lite: true,
	});
	let mut body = serde_json::json!({"client_metadata":{"caller":"kept"}});
	apply_codex_client_metadata(
		&mut body,
		&identity,
		CodexWireTransport::WebSocket,
		true,
		Some("turn-state-secret"),
	)
	.unwrap();
	assert_eq!(body["client_metadata"]["caller"], "kept");
	assert_eq!(body["client_metadata"]["turn_id"], "turn-secret");
	assert_eq!(
		body["client_metadata"]["ws_request_header_x_openai_internal_codex_responses_lite"],
		"true"
	);

	assert_eq!(
		plan.get("openai-beta").map(|value| value.as_bytes()),
		Some(b"responses_websockets=2026-02-06".as_slice())
	);
	assert_eq!(plan.get("originator").map(|value| value.as_bytes()), Some(b"pi".as_slice()));
	assert!(plan.get("authorization").is_none());
	assert!(plan.get("x-api-key").is_none());
	assert!(plan.get("chatgpt-account-id").unwrap().is_sensitive());
	assert!(plan.get("x-oai-attestation").unwrap().is_sensitive());
	let debug = format!("{plan:?}");
	for secret in [
		"install-secret",
		"session-secret",
		"thread-secret",
		"window-secret",
		"turn-secret",
		"account-secret",
		"attestation-secret",
		"turn-state-secret",
		"etag-secret",
	] {
		assert!(!debug.contains(secret));
	}
}

#[test]
fn websocket_continuation_sends_only_the_strict_delta() {
	let fixture: Value =
		serde_json::from_slice(include_bytes!("fixtures/openai_codex/websocket.continuation.json"))
			.unwrap();
	let mut state = CodexContinuationState::default();
	state.commit(
		fixture["previous_request"].clone(),
		fixture["previous_response_id"].as_str().unwrap(),
		fixture["previous_response_items"]
			.as_array()
			.unwrap()
			.clone(),
	);
	let frame = state.response_create(&fixture["current_request"]).unwrap();
	assert_eq!(frame, fixture["expected_frame"]);
}

#[test]
fn websocket_router_correlates_interleaved_items_and_emits_one_terminal() {
	let frames: Vec<Value> =
		serde_json::from_slice(include_bytes!("fixtures/openai_codex/websocket.interleaved.json"))
			.unwrap();
	let mut router = CodexFrameRouter::after_response("resp_old");
	let dispositions = frames
		.iter()
		.map(|frame| router.route(frame).unwrap())
		.collect::<Vec<_>>();
	assert_eq!(dispositions[0], CodexFrameDisposition::Drop);
	assert_eq!(
		dispositions
			.iter()
			.filter(|disposition| **disposition == CodexFrameDisposition::Terminal)
			.count(),
		1
	);
	assert_eq!(dispositions.last(), Some(&CodexFrameDisposition::Drop));
	assert_eq!(router.active_response_id(), Some("resp_new"));
}

#[test]
fn websocket_router_fails_closed_on_foreign_response() {
	let mut router = CodexFrameRouter::default();
	router
		.route(&serde_json::json!({"type":"response.created","response":{"id":"resp_a"}}))
		.unwrap();
	let error = router
		.route(&serde_json::json!({"type":"response.completed","response":{"id":"resp_b"}}))
		.unwrap_err();
	assert!(error.to_string().contains("interleaved"));
}

#[test]
fn http_fallback_is_selected_only_for_replay_safe_failures() {
	let safe = CodexReplaySafety::default();
	assert_eq!(
		classify_codex_fallback(CodexWebSocketFailure::ConnectionFatal, safe, 0, 5),
		CodexFallbackAction::ReplayOverHttp
	);
	assert_eq!(
		classify_codex_fallback(CodexWebSocketFailure::RetryableTransport, safe, 0, 5),
		CodexFallbackAction::ReconnectWebSocket
	);
	assert_eq!(
		classify_codex_fallback(CodexWebSocketFailure::RetryableTransport, safe, 5, 5),
		CodexFallbackAction::ReplayOverHttp
	);
	assert_eq!(
		classify_codex_fallback(CodexWebSocketFailure::RetryableProvider, safe, 5, 5),
		CodexFallbackAction::Surface
	);
	assert_eq!(
		classify_codex_websocket_failure(None, "unrecognized application failure", false),
		CodexWebSocketFailure::Provider
	);
	assert_eq!(
		classify_codex_websocket_failure(None, "websocket pong timeout", false),
		CodexWebSocketFailure::RetryableTransport
	);
	assert_eq!(
		classify_codex_fallback(
			CodexWebSocketFailure::ConnectionFatal,
			CodexReplaySafety { delivered_tool_call: true, ..safe },
			0,
			5,
		),
		CodexFallbackAction::Surface
	);
}

#[test]
fn cancellation_drops_frames_and_never_falls_back() {
	let mut router = CodexFrameRouter::default();
	router.cancel();
	assert_eq!(
		router
			.route(&serde_json::json!({"type":"response.completed","response":{"id":"resp"}}))
			.unwrap(),
		CodexFrameDisposition::Drop
	);
	assert_eq!(
		classify_codex_fallback(CodexWebSocketFailure::Cancelled, CodexReplaySafety::default(), 0, 5,),
		CodexFallbackAction::Cancelled
	);
}

#[test]
fn encrypted_reasoning_signature_and_terminal_are_delegated_once() {
	let codec = OpenAiCodexCodec::new();
	let mut state = DecodeState::default();
	let added = br#"{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"ciphertext"}}"#;
	let done = br#"{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"ciphertext","summary":[]}}"#;
	let terminal = br#"{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}"#;
	codec.decode(Frame::Data(added), &mut state).unwrap();
	let end = codec.decode(Frame::Data(done), &mut state).unwrap();
	assert!(matches!(
		end.as_slice(),
		[TurnEvent::PartEnd { signature, .. }] if signature.as_ref() == b"ciphertext"
	));
	let first = codec.decode(Frame::Data(terminal), &mut state).unwrap();
	let duplicate = codec.decode(Frame::Data(terminal), &mut state).unwrap();
	assert_eq!(
		first
			.iter()
			.filter(|event| matches!(event, TurnEvent::Outcome(_)))
			.count(),
		1
	);
	assert!(duplicate.is_empty());
}

#[test]
fn devicecheck_attestation_matches_the_deterministic_fixture() {
	let fixture: Value =
		serde_json::from_slice(include_bytes!("fixtures/openai_codex/attestation.devicecheck.json"))
			.unwrap();
	let token = CodexDeviceToken::new(Bytes::copy_from_slice(
		fixture["result"]["token_base64"]
			.as_str()
			.unwrap()
			.as_bytes(),
	))
	.unwrap();
	let result = CodexDeviceCheckResult {
		supported:  true,
		token:      Some(token),
		latency_ms: fixture["result"]["latency_ms"].as_f64(),
	};
	let signals = CodexAttestationSignals {
		locale:     fixture["signals"]["locale"].as_str().unwrap(),
		timezone:   fixture["signals"]["timezone"].as_str().unwrap(),
		session_id: fixture["signals"]["session_id"].as_str().unwrap(),
	};
	let actual = build_codex_attestation(&result, &signals).unwrap();
	assert_eq!(actual.as_bytes(), fixture["expected"].as_str().unwrap().as_bytes());
	assert!(!format!("{actual:?}").contains("dG9rZW4"));
	assert!(!format!("{:?}", result.token).contains("dG9rZW4"));
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn native_platform_selects_devicecheck_attestor() {
	assert_eq!(CodexAttestor::default(), CodexAttestor::DeviceCheck);
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[test]
fn unsupported_platform_selects_unavailable_attestor() {
	assert_eq!(CodexAttestor::default(), CodexAttestor::Unavailable);
}
