#![cfg(unix)]

//! Official `OpenAI` and Anthropic SDK process-level integration coverage.

use std::{path::PathBuf, process::Command};

#[test]
fn official_openai_and_anthropic_clients_drive_production_daemon() {
	let project = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/stock_sdk");
	let output = Command::new("uv")
		.args(["run", "--project"])
		.arg(&project)
		.args(["--locked", "python"])
		.arg(project.join("stock_sdk.py"))
		.env("OMP_STOCK_SDK_BIN", env!("CARGO_BIN_EXE_omp"))
		.output()
		.expect("run pinned official SDK integration environment with uv");
	assert!(
		output.status.success(),
		"stock SDK integration failed\nstdout:\n{}\nstderr:\n{}",
		String::from_utf8_lossy(&output.stdout),
		String::from_utf8_lossy(&output.stderr),
	);
}
