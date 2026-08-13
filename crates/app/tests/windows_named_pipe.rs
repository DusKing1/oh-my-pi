#![cfg(windows)]

use std::path::Path;

use clap::Parser as _;
use omp_app::{cli::OmpCli, endpoint::LocalEndpoint};

#[test]
fn windows_pipe_uri_normalizes_to_native_omp_endpoint() {
	let endpoint: LocalEndpoint = "npipe://./pipe/omp-smoke".parse().expect("pipe URI");
	assert_eq!(endpoint.as_path(), Path::new(r"\\.\pipe\omp-smoke"));
	let parsed = OmpCli::try_parse_from(["omp", "serve", "--pipe", r"\\.\pipe\omp-smoke"]);
	assert!(parsed.is_ok());
}
