//! OMP command-line entry point.

use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
	if std::env::args_os()
		.nth(1)
		.is_some_and(|arg| arg == omp_app::envd::EVAL_CHILD_ARG)
	{
		return match omp_app::envd::run_eval_child_entry().await {
			Ok(()) => ExitCode::SUCCESS,
			Err(error) => {
				eprintln!("omp eval child: {error}");
				ExitCode::FAILURE
			},
		};
	}
	if std::env::args_os()
		.nth(1)
		.is_some_and(|arg| arg == omp_app::envd::worker::WORKER_ARG)
	{
		return match omp_app::envd::worker::run_worker_entry() {
			Ok(()) => ExitCode::SUCCESS,
			Err(error) => {
				eprintln!("omp tool worker: {error}");
				ExitCode::FAILURE
			},
		};
	}
	omp_telemetry::export::init();
	let result = omp_app::run().await;
	omp_telemetry::export::shutdown();
	match result {
		Ok(()) => ExitCode::SUCCESS,
		Err(error) => {
			// Errors that reach this point are top-level execution failures;
			// formatted via miette's diagnostic handler.
			eprintln!("{error:?}");
			ExitCode::FAILURE
		},
	}
}
