//! Golden corpus: real provider error surfaces (mined from production logs
//! and sessions) and the misclassification war stories the classifier
//! exists to prevent. Every case name states the required behavior.

use omp_llm_error::{
	Action, AdviseContext, Evidence, Feature, Fidelity, Kind, OAuthFailure, RateLimitReason,
	WireApi, advise, classify_at,
};

const NOW: u64 = 1_754_000_000_000;

#[track_caller]
fn assert_kinds(
	ev: &Evidence<'_>,
	expect: &[Kind],
	forbid: &[Kind],
) -> omp_llm_error::Classification {
	let cls = classify_at(ev, NOW);
	for &k in expect {
		assert!(cls.kinds.has(k), "expected {k:?} in {:?}", cls.kinds);
	}
	for &k in forbid {
		assert!(!cls.kinds.has(k), "forbidden {k:?} in {:?}", cls.kinds);
	}
	cls
}

// ── Anthropic ────────────────────────────────────────────────────────────

#[test]
fn anthropic_overloaded_sse_frame() {
	let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"},"request_id":"req_011CbkM6RLqtJHESXUR9VXS2"}"#;
	let cls = assert_kinds(&Evidence::http(529, body), &[Kind::ModelCapacity, Kind::Transient], &[
		Kind::UsageLimit,
	]);
	assert_eq!(cls.request_id.as_deref(), Some("req_011CbkM6RLqtJHESXUR9VXS2"));
	assert_eq!(cls.rate_limit.unwrap().reason, RateLimitReason::ModelCapacityExhausted);
	assert!(!cls.rate_limit.unwrap().rotate, "overload must not consume the credential");
}

#[test]
fn anthropic_rate_limited_transient() {
	let body = r#"{"type":"error","error":{"details":null,"type":"rate_limit_error","message":"Rate limited"}}"#;
	let cls = assert_kinds(&Evidence::http(429, body), &[Kind::RateThrottle, Kind::Transient], &[]);
	assert!(!cls.rate_limit.unwrap().rotate, "bare throttle stays on the credential");
}

#[test]
fn anthropic_internal_server_error() {
	let body = r#"{"type":"error","error":{"details":null,"type":"api_error","message":"Internal server error"}}"#;
	assert_kinds(&Evidence::http(500, body), &[Kind::ServerError, Kind::Transient], &[
		Kind::UsageLimit,
	]);
}

#[test]
fn anthropic_prompt_too_long() {
	let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 219124 tokens > 200000 maximum"}}"#;
	let cls = assert_kinds(&Evidence::http(400, body), &[Kind::ContextOverflow], &[Kind::Transient]);
	let plan = advise(&cls, &AdviseContext::default());
	assert_eq!(plan[0], Action::ReduceContext);
}

#[test]
fn anthropic_output_content_filter_terminal() {
	let body = r#"{"type":"error","error":{"details":null,"type":"invalid_request_error","message":"Output blocked by content filtering policy"}}"#;
	let cls = assert_kinds(&Evidence::http(400, body), &[Kind::ContentBlocked], &[Kind::Transient]);
	assert!(!cls.retryable_exact_request(true), "content blocks never auto-retry");
}

#[test]
fn thinking_signature_proxy_rejection() {
	let cls = assert_kinds(
		&Evidence::http(400, r#"{"error":{"message":"Invalid signature in `thinking` block"}}"#),
		&[Kind::FeatureUnsupported, Kind::InvalidRequest],
		&[],
	);
	assert_eq!(cls.feature, Some(Feature::ThinkingSignature));
	let plan = advise(&cls, &AdviseContext::default());
	assert_eq!(plan[0], Action::StripFeature(Feature::ThinkingSignature));
}

// ── OpenAI Codex ─────────────────────────────────────────────────────────

#[test]
fn codex_usage_limit_rotates() {
	let cls = assert_kinds(
		&Evidence::live(
			"Codex error event: The usage limit has been reached (code=usage_limit_reached)",
		)
		.with_code("usage_limit_reached"),
		&[Kind::UsageLimit],
		&[Kind::ConcurrencyCap],
	);
	assert!(cls.rate_limit.unwrap().rotate);
	assert_eq!(cls.rate_limit.unwrap().reason, RateLimitReason::QuotaExhausted);
}

#[test]
fn codex_context_length_exceeded() {
	let cls = assert_kinds(
		&Evidence::live(
			"Codex error event: Your input exceeds the context window of this model. Please adjust \
			 your input and try again. (code=context_length_exceeded)",
		)
		.with_code("context_length_exceeded"),
		&[Kind::ContextOverflow],
		&[],
	);
	assert_eq!(advise(&cls, &AdviseContext::default())[0], Action::ReduceContext);
}

#[test]
fn codex_cyber_policy_rotates_account() {
	let cls = assert_kinds(
		&Evidence::live(
			"Codex error event: This content was flagged for possible cybersecurity risk. To get authorized for security work, join the Trusted Access for Cyber program: https://chatgpt.com/cyber (code=cyber_policy)",
		)
		.with_code("cyber_policy"),
		&[Kind::AccountPolicy, Kind::ContentBlocked],
		&[],
	);
	let ctx = AdviseContext { has_sibling_credential: true, ..AdviseContext::default() };
	assert!(matches!(advise(&cls, &ctx)[0], Action::RotateCredential { .. }));
}

#[test]
fn codex_stale_previous_response() {
	let ev = Evidence::http(400, r#"{"detail":"Unsupported parameter: previous_response_id"}"#)
		.with_api(WireApi::CodexResponses);
	let cls = assert_kinds(&ev, &[Kind::StaleSessionItem, Kind::Transient], &[]);
	assert!(advise(&cls, &AdviseContext::default()).contains(&Action::ResetSession));
}

#[test]
fn stale_item_words_do_not_fire_off_responses_api() {
	let ev = Evidence::http(400, r#"{"error":{"message":"previous response invalid"}}"#);
	assert_kinds(&ev, &[], &[Kind::StaleSessionItem]);
}

#[test]
fn codex_reasoning_effort_menu_extracted() {
	let body = r#"{"error":{"message":"[OneOfParam] [reasoning.effort] [invalid_type] Invalid type for 'reasoning.effort': expected one of one of 'none', 'minimal', 'low', 'medium', 'high', or 'xhigh' or integer, but got an array instead.","code":"invalid_request_error"}}"#;
	let cls = assert_kinds(&Evidence::http(400, body), &[Kind::FeatureUnsupported], &[]);
	assert_eq!(cls.feature, Some(Feature::ReasoningEffort));
	let menu: Vec<&str> = cls.allowed_efforts.iter().map(|v| v.as_str()).collect();
	assert_eq!(menu, ["none", "minimal", "low", "medium", "high", "xhigh"]);
}

// ── Rate-limit lane discipline ───────────────────────────────────────────

#[test]
fn fireworks_throttle_backs_off_without_rotation() {
	let msg = "429 You have exceeded your rate limit for this API. Please try again later. For more information, see https://docs.fireworks.ai/guides/quotas_usage/rate-limits.";
	let cls = classify_at(&Evidence::live(msg), NOW);
	let limit = cls.rate_limit.unwrap();
	assert_eq!(limit.reason, RateLimitReason::RateLimitExceeded);
	assert!(!limit.rotate, "throttle must not burn sibling credentials");
}

#[test]
fn opaque_429_rotates_conservatively() {
	let cls = classify_at(&Evidence::prose("429 status code (no body)"), NOW);
	let limit = cls.rate_limit.unwrap();
	assert!(limit.rotate, "nothing proves an opaque 429 is transient");
	assert!(cls.kinds.has(Kind::UsageLimit));
}

#[test]
fn vertex_concurrency_quota_is_shed_not_rotated() {
	let cls = classify_at(
		&Evidence::http(429, "Online prediction concurrent requests quota exceeded for gemini"),
		NOW,
	);
	let limit = cls.rate_limit.unwrap();
	assert_eq!(limit.reason, RateLimitReason::ConcurrentLimit);
	assert!(!limit.rotate, "'quota' wording on a concurrency cap must not rotate");
	assert!(!cls.kinds.has(Kind::UsageLimit));
	assert_eq!(limit.reason.backoff_ms(), (5_000, 5_000));
}

#[test]
fn http_402_is_billing_even_when_worded_as_concurrency() {
	let cls = classify_at(&Evidence::http(402, "too many concurrent requests for your plan"), NOW);
	assert!(cls.kinds.has(Kind::Billing));
	assert!(cls.kinds.has(Kind::UsageLimit));
	assert!(cls.rate_limit.unwrap().rotate);
}

#[test]
fn chinese_quota_exhaustion_rotates() {
	let cls = classify_at(&Evidence::http(429, "您的额度已用完，请升级套餐"), NOW);
	assert_eq!(cls.rate_limit.unwrap().reason, RateLimitReason::QuotaExhausted);
	assert!(cls.rate_limit.unwrap().rotate);
}

#[test]
fn chinese_throttle_does_not_rotate() {
	let cls = classify_at(&Evidence::http(429, "请求速率达到上限，请稍后重试"), NOW);
	assert_eq!(cls.rate_limit.unwrap().reason, RateLimitReason::RateLimitExceeded);
	assert!(!cls.rate_limit.unwrap().rotate);
}

#[test]
fn google_bare_resource_exhausted_is_capacity() {
	let body = r#"{"error":{"code":429,"message":"Resource exhausted. Please try again later.","status":"RESOURCE_EXHAUSTED"}}"#;
	let cls = classify_at(&Evidence::http(429, body), NOW);
	assert_eq!(cls.rate_limit.unwrap().reason, RateLimitReason::ModelCapacityExhausted);
	assert!(!cls.rate_limit.unwrap().rotate);
}

#[test]
fn antigravity_no_capacity_is_transient() {
	let msg = "Cloud Code Assist API error (429): No capacity available for model \
	           claude-opus-4-5-thinking on the server";
	let cls = classify_at(&Evidence::live(msg), NOW);
	assert_eq!(cls.rate_limit.unwrap().reason, RateLimitReason::ModelCapacityExhausted);
	assert!(cls.kinds.has(Kind::Transient));
}

#[test]
fn openrouter_free_daily_cap_rotates() {
	let cls = classify_at(&Evidence::http(429, "Rate limit exceeded: free-models-per-day"), NOW);
	assert!(cls.rate_limit.unwrap().rotate);
}

// ── Auth / OAuth ─────────────────────────────────────────────────────────

#[test]
fn invalid_grant_is_definitive_oauth_death() {
	let cls = classify_at(
		&Evidence::prose(r#"Error: HTTP 400 invalid_grant {"error":"invalid_grant"}"#),
		NOW,
	);
	assert!(cls.kinds.has(Kind::OAuthExpired));
	assert_eq!(cls.oauth, Some(OAuthFailure::Definitive));
	assert_eq!(advise(&cls, &AdviseContext::default())[0], Action::Relogin);
}

#[test]
fn refresh_token_revoked_is_definitive() {
	let cls = classify_at(
		&Evidence::prose("Error: OAuth refresh failed: 400 invalid_grant: refresh token revoked"),
		NOW,
	);
	assert_eq!(cls.oauth, Some(OAuthFailure::Definitive));
}

#[test]
fn prose_401_never_advises_credential_deletion() {
	// A 401 recovered from prose has repeatedly turned out to be stale
	// replayed session state; deletion requires structured evidence.
	let cls = classify_at(&Evidence::prose("request failed with status 401 unauthorized"), NOW);
	assert!(cls.kinds.has(Kind::AuthFailed));
	assert_eq!(cls.fidelity, Fidelity::Prose);
	assert!(!advise(&cls, &AdviseContext::default()).contains(&Action::InvalidateCredential));
}

#[test]
fn structured_401_refreshes_but_never_deletes() {
	// A parsed 401 envelope is not revocation proof: ordinary auth recovery
	// is refresh-in-place, then sibling rotation.
	let cls = classify_at(&Evidence::http(401, r#"{"error":{"message":"invalid api key"}}"#), NOW);
	assert_eq!(cls.fidelity, Fidelity::Structured);
	let ctx = AdviseContext { has_sibling_credential: true, ..AdviseContext::default() };
	let plan = advise(&cls, &ctx);
	assert!(plan.contains(&Action::RefreshCredential));
	assert!(plan.contains(&Action::RotateCredential { block_ms: 0 }));
	assert!(!plan.contains(&Action::InvalidateCredential));
}

#[test]
fn explicit_revocation_invalidates_credential() {
	// Positive revocation evidence is the only deletion trigger.
	let cls = classify_at(&Evidence::prose("Invalidated OAuth token; please re-authenticate"), NOW);
	let plan = advise(&cls, &AdviseContext::default());
	assert!(plan.contains(&Action::Relogin));
	assert!(plan.contains(&Action::InvalidateCredential));
}

#[test]
fn gateway_auth_unavailable_is_auth_not_server_error_lane() {
	let body = r#"{"error":{"message":"auth_unavailable: no auth available","type":"server_error","code":"internal_server_error"}}"#;
	let cls = classify_at(&Evidence::http(500, body), NOW);
	assert!(
		cls.kinds.has(Kind::AuthFailed),
		"synthetic 500 whose body is auth must reach the auth lane"
	);
}

#[test]
fn account_scoped_403_upgrades_to_usage_limit() {
	let cls =
		classify_at(&Evidence::http(403, "Your overall message rate limit has been reached"), NOW);
	assert!(cls.rate_limit.unwrap().rotate);
	assert!(cls.kinds.has(Kind::UsageLimit));
}

// ── Deterministic-behind-retryable-status guards ─────────────────────────

#[test]
fn llamacpp_tool_json_500_must_not_loop() {
	let cls = classify_at(&Evidence::http(500, "failed to parse tool call arguments as json"), NOW);
	assert!(cls.kinds.has(Kind::MalformedToolCall));
	assert!(cls.kinds.has(Kind::Deterministic));
	assert!(!cls.retryable_exact_request(true), "identical replay fails identically");
}

#[test]
fn copilot_model_not_supported_is_fleet_skew() {
	let ev = Evidence::http(
		400,
		r#"{"error":{"code":"model_not_supported","message":"model_not_supported"}}"#,
	)
	.with_provider("github-copilot");
	let cls = assert_kinds(&ev, &[Kind::Transient], &[Kind::InvalidModel]);
	assert!(cls.retryable_exact_request(true));
}

#[test]
fn model_not_supported_elsewhere_is_invalid_model() {
	let ev = Evidence::http(
		400,
		r#"{"error":{"code":"model_not_supported","message":"model_not_supported"}}"#,
	);
	assert_kinds(&ev, &[Kind::InvalidModel], &[]);
}

#[test]
fn gemini_model_not_found_is_invalid_model() {
	let body = r#"{"error":{"message":"models/gemini-live-2.5-flash is not found for API version v1beta, or is not supported for generateContent.","code":404,"status":"NOT_FOUND"}}"#;
	assert_kinds(&Evidence::http(404, body), &[Kind::InvalidModel], &[Kind::Transient]);
}

#[test]
fn bare_404_prose_does_not_condemn_model() {
	assert_kinds(
		&Evidence::prose("AuthBrokerError: Auth broker request failed: 404 Not Found"),
		&[],
		&[Kind::InvalidModel],
	);
}

// ── Transport / stream ───────────────────────────────────────────────────

#[test]
fn econnreset_is_transport_transient() {
	assert_kinds(
		&Evidence::live("Error: fetch failed: ECONNRESET"),
		&[Kind::Transport, Kind::Transient],
		&[],
	);
}

#[test]
fn bun_socket_close_is_transport() {
	let msg = "Error: The socket connection was closed unexpectedly. For more information, pass \
	           `verbose: true` in the second argument to fetch()";
	assert_kinds(&Evidence::live(msg), &[Kind::Transport, Kind::Transient], &[]);
}

#[test]
fn h2_internal_error_from_peer() {
	assert_kinds(
		&Evidence::live("stream error: stream ID 787; INTERNAL_ERROR; received from peer"),
		&[Kind::Transport, Kind::Transient],
		&[],
	);
}

#[test]
fn live_truncation_matches_broad_table() {
	let cls = classify_at(&Evidence::live("JSON Parse error: Unterminated string"), NOW);
	assert!(cls.kinds.has(Kind::StreamCorruption));
}

#[test]
fn persisted_bare_truncated_is_too_low_signal() {
	// Persisted strings lost their transport context; bare "truncated" must
	// not classify as stream corruption after the fact.
	let cls = classify_at(&Evidence::prose("response truncated"), NOW);
	assert!(!cls.kinds.has(Kind::StreamCorruption));
	// But the full diagnostic form still does.
	let cls = classify_at(&Evidence::prose("JSON parse error: unterminated string"), NOW);
	assert!(cls.kinds.has(Kind::StreamCorruption));
}

#[test]
fn stale_session_401_never_touches_the_credential() {
	// Copilot warmed sessions reject replayed native history with HTTP 401.
	// The credential is valid; deleting or refreshing it on this signal is
	// the documented worst-case misclassification.
	let ev = Evidence::http(401, r#"{"error":{"message":"Item with id 'rs_123' not found."}}"#)
		.with_api(WireApi::OpenAiResponses);
	let cls = classify_at(&ev, NOW);
	assert!(cls.kinds.has(Kind::StaleSessionItem));
	let plan = advise(&cls, &AdviseContext::default());
	assert!(plan.contains(&Action::ResetSession));
	assert!(!plan.contains(&Action::InvalidateCredential));
	assert!(!plan.contains(&Action::RefreshCredential));
}

#[test]
fn stream_ended_before_terminal_event() {
	assert_kinds(
		&Evidence::live("server_error: stream closed with reason: error"),
		&[Kind::StreamCorruption, Kind::Transient],
		&[],
	);
}

#[test]
fn event_order_violation_is_retry_safe() {
	assert_kinds(
		&Evidence::live("Anthropic stream envelope error: content_block_delta before message_start"),
		&[Kind::StreamOrder, Kind::Transient],
		&[],
	);
}

// ── False-positive guards (each of these caused a production incident) ──

#[test]
fn generate_content_request_is_not_a_rate_limit() {
	let cls = classify_at(&Evidence::prose("invalid GenerateContentRequest payload"), NOW);
	assert!(cls.rate_limit.is_none());
	assert!(!cls.kinds.has(Kind::RateThrottle));
}

#[test]
fn duration_prose_is_not_a_status() {
	let cls = classify_at(&Evidence::prose("upstream answered, request took 200ms in total"), NOW);
	assert_eq!(cls.status, None);
}

#[test]
fn concurrent_invocation_unsupported_is_not_a_cap() {
	let cls = classify_at(&Evidence::http(400, "concurrent invocation is not supported"), NOW);
	assert!(!cls.kinds.has(Kind::ConcurrencyCap));
}

#[test]
fn reasoning_content_mention_is_not_reasoning_effort() {
	let body = r#"{"error":{"message":"thinking is enabled but reasoning_content is missing in assistant tool call message at index 2","type":"invalid_request_error"}}"#;
	let cls = classify_at(&Evidence::http(400, body), NOW);
	assert_ne!(cls.feature, Some(Feature::ReasoningEffort));
}

// ── Feature strips ───────────────────────────────────────────────────────

#[test]
fn tool_choice_auto_only_provider() {
	let cls = classify_at(
		&Evidence::http(
			400,
			r#"{"error":{"message":"tool_choice: only 'auto' is supported by this model"}}"#,
		),
		NOW,
	);
	assert_eq!(cls.feature, Some(Feature::ToolChoice));
}

#[test]
fn strict_tools_grammar_too_large() {
	let body =
		r#"{"error":{"type":"invalid_request_error","message":"the compiled grammar is too large"}}"#;
	let cls = classify_at(&Evidence::http(400, body), NOW);
	assert_eq!(cls.feature, Some(Feature::StrictTools));
}

#[test]
fn structured_outputs_unsupported() {
	let cls = classify_at(
		&Evidence::http(
			400,
			r#"{"error":{"message":"structured_outputs is not supported for this model"}}"#,
		),
		NOW,
	);
	assert_eq!(cls.feature, Some(Feature::StructuredOutputs));
}

#[test]
fn sampling_parameter_rejections_are_feature_scoped() {
	for (status, parameter) in [
		(400, "temperature"),
		(422, "top_p"),
		(400, "top_k"),
		(422, "min_p"),
		(400, "frequency_penalty"),
		(400, "presence_penalty"),
		(422, "repetition_penalty"),
		(400, "stop"),
		(422, "stop_sequences"),
	] {
		let body = format!(
			r#"{{"error":{{"message":"Unsupported parameter: '{parameter}' is not supported with this model."}}}}"#
		);
		let cls = classify_at(&Evidence::http(status, &body), NOW);
		assert_eq!(cls.feature, Some(Feature::SamplingParameters), "{parameter}");
		assert!(cls.kinds.has(Kind::FeatureUnsupported), "{parameter}");
	}
	let natural = classify_at(
		&Evidence::http(
			400,
			r#"{"error":{"message":"This model does not support the temperature parameter."}}"#,
		),
		NOW,
	);
	assert_eq!(natural.feature, Some(Feature::SamplingParameters));
	assert_eq!(
		advise(&natural, &AdviseContext::default())[0],
		Action::StripFeature(Feature::SamplingParameters),
	);
}

#[test]
fn sampling_words_in_unrelated_prose_are_not_feature_rejections() {
	for (status, body) in [
		(400, "Lower the temperature and stop when the answer is complete."),
		(422, "Top P sampling controls how diverse prose can be."),
		(400, "This documentation does not support temperature controls."),
		(409, "Unsupported parameter: 'temperature' is not supported with this model."),
	] {
		let cls = classify_at(&Evidence::http(status, body), NOW);
		assert_ne!(cls.feature, Some(Feature::SamplingParameters), "{body}");
	}
}

#[test]
fn fast_mode_entitlement_429() {
	let body = r#"{"error":{"type":"rate_limit_error","message":"fast mode requires extra usage"}}"#;
	let cls = classify_at(&Evidence::http(429, body), NOW);
	assert_eq!(cls.feature, Some(Feature::FastMode));
}

// ── Retry timing ─────────────────────────────────────────────────────────

#[test]
fn header_hint_beats_prose() {
	let headers = [("retry-after-ms", "1500")];
	let ev = Evidence::http(429, "slow down; retry-after: 99").with_headers(&headers);
	assert_eq!(classify_at(&ev, NOW).retry.unwrap().delay_ms, 1500);
}

#[test]
fn prose_hint_recovered_from_persisted_message() {
	let cls = classify_at(&Evidence::prose("429 rate limited retry-after-ms=2500"), NOW);
	assert_eq!(cls.retry.unwrap().delay_ms, 2500);
}

// ── Abort ────────────────────────────────────────────────────────────────

#[test]
fn user_abort_is_never_retried() {
	let cls = classify_at(&Evidence::prose("Request was aborted"), NOW);
	assert!(cls.kinds.has(Kind::Aborted));
	assert!(!cls.retryable_exact_request(true));
	assert_eq!(advise(&cls, &AdviseContext::default())[0], Action::SurfaceTerminal);
}
