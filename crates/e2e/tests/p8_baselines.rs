#[path = "../src/bin/baseline.rs"]
mod baseline;

use std::time::Duration;

use baseline::{BaselineMetrics, duration_rate, measure, slowdown_ratio, write_metrics};

#[test]
fn ratio_math_and_zero_duration_are_guarded() {
	assert_eq!(duration_rate(500, Duration::from_millis(250)).unwrap(), 2_000.0);
	assert_eq!(slowdown_ratio(5_000.0, 1_000.0).unwrap(), 5.0);
	assert!(duration_rate(1, Duration::ZERO).is_err());
	assert!(duration_rate(0, Duration::from_nanos(1)).is_err());
	assert!(slowdown_ratio(1.0, 0.0).is_err());
	assert!(slowdown_ratio(f64::INFINITY, 1.0).is_err());
}

#[tokio::test]
async fn artifact_schema_is_stable_and_frame_metric_is_record_only() {
	let metrics = measure(128, 256, 2).await.expect("bounded baseline measurement");
	assert_eq!(metrics.schema_version, 1);
	assert_eq!(metrics.frame.token_count, 128);
	assert_eq!(metrics.frame.sample_count, 128);
	assert!(metrics.frame.p95_frame_ns > 0);
	assert_eq!(metrics.r#loop.tokens_per_sample, 256);
	assert_eq!(metrics.r#loop.sample_count, 2);
	assert_eq!(metrics.r#loop.regression_limit, 5.0);
	assert_eq!(metrics.r#loop.gross_regression, metrics.r#loop.slowdown_ratio > 5.0);

	let scratch = tempfile::tempdir().expect("artifact scratch directory");
	let artifact = scratch.path().join("nested/p8.json");
	write_metrics(&artifact, &metrics).expect("write metrics");
	let encoded = std::fs::read_to_string(&artifact).expect("read metrics");
	let value: serde_json::Value = serde_json::from_str(&encoded).expect("valid JSON");
	let root = value.as_object().expect("metrics object");
	assert_eq!(root.keys().collect::<Vec<_>>(), ["schema_version", "frame", "loop"]);
	assert_eq!(
		root["frame"].as_object().unwrap().keys().collect::<Vec<_>>(),
		["token_count", "sample_count", "p95_frame_ns"]
	);
	assert_eq!(
		root["loop"].as_object().unwrap().keys().collect::<Vec<_>>(),
		[
			"tokens_per_sample",
			"sample_count",
			"raw_duration_ns",
			"full_loop_duration_ns",
			"raw_tokens_per_second",
			"full_tokens_per_second",
			"slowdown_ratio",
			"regression_limit",
			"gross_regression",
		]
	);
	let decoded: BaselineMetrics = serde_json::from_str(&encoded).expect("stable schema decodes");
	assert_eq!(decoded.schema_version, metrics.schema_version);
	assert_eq!(decoded.frame, metrics.frame);
	assert_eq!(decoded.r#loop.tokens_per_sample, metrics.r#loop.tokens_per_sample);
	assert_eq!(decoded.r#loop.sample_count, metrics.r#loop.sample_count);
	assert_eq!(decoded.r#loop.raw_duration_ns, metrics.r#loop.raw_duration_ns);
	assert_eq!(decoded.r#loop.full_loop_duration_ns, metrics.r#loop.full_loop_duration_ns);
	assert_eq!(decoded.r#loop.regression_limit, metrics.r#loop.regression_limit);
	assert_eq!(decoded.r#loop.gross_regression, metrics.r#loop.gross_regression);
	for (actual, expected) in [
		(decoded.r#loop.raw_tokens_per_second, metrics.r#loop.raw_tokens_per_second),
		(decoded.r#loop.full_tokens_per_second, metrics.r#loop.full_tokens_per_second),
		(decoded.r#loop.slowdown_ratio, metrics.r#loop.slowdown_ratio),
	] {
		assert!((actual - expected).abs() <= expected.abs() * f64::EPSILON);
	}
}
