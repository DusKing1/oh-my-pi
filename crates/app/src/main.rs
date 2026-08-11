//! OMP command-line entry point.

use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
	omp_telemetry::export::init();
	let result = omp_app::run().await;
	omp_telemetry::export::shutdown();
	match result {
		Ok(()) => ExitCode::SUCCESS,
		Err(error) => {
			// Do not print inner details: lower transport layers may
			// carry untrusted provider diagnostics. Public top-level errors are
			// deliberately classified and redacted at their subsystem boundary.
			eprintln!("omp: {error}");
			ExitCode::FAILURE
		},
	}
}
