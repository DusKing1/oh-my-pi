//! Command-line entry point for the project-scoped document authority.

use std::{ffi::OsString, path::PathBuf, process::ExitCode};

use omp_docserver::daemon::{self, Transport};

const USAGE: &str =
	"usage: omp-docserverd --root <path> [--socket <path> | --stdio] [--lsp-config <path>]...";

struct Options {
	root:        PathBuf,
	transport:   Transport,
	lsp_configs: Vec<PathBuf>,
}

#[tokio::main]
async fn main() -> ExitCode {
	match parse(std::env::args_os().skip(1)) {
		Ok(Some(options)) => {
			match daemon::run(options.root, options.transport, options.lsp_configs).await {
				Ok(()) => ExitCode::SUCCESS,
				Err(error) => {
					eprintln!("omp-docserverd: {error}");
					ExitCode::FAILURE
				},
			}
		},
		Ok(None) => {
			println!("{USAGE}");
			ExitCode::SUCCESS
		},
		Err(error) => {
			eprintln!("omp-docserverd: {error}\n{USAGE}");
			ExitCode::FAILURE
		},
	}
}

fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Option<Options>, String> {
	let mut arguments = arguments.into_iter();
	let mut root = None;
	let mut transport = None;
	let mut lsp_configs = Vec::new();
	while let Some(argument) = arguments.next() {
		match argument.to_str() {
			Some("--help" | "-h") => return Ok(None),
			Some("--root") => {
				if root.is_some() {
					return Err("--root may be supplied only once".to_owned());
				}
				root = Some(PathBuf::from(
					arguments
						.next()
						.ok_or_else(|| "--root requires a path".to_owned())?,
				));
			},
			Some("--socket") => {
				if transport.is_some() {
					return Err("--socket and --stdio are mutually exclusive".to_owned());
				}
				transport = Some(Transport::Socket(PathBuf::from(
					arguments
						.next()
						.ok_or_else(|| "--socket requires a path".to_owned())?,
				)));
			},
			Some("--stdio") => {
				if transport.is_some() {
					return Err("--socket and --stdio are mutually exclusive".to_owned());
				}
				transport = Some(Transport::Stdio);
			},
			Some("--lsp-config") => {
				lsp_configs.push(PathBuf::from(
					arguments
						.next()
						.ok_or_else(|| "--lsp-config requires a path".to_owned())?,
				));
			},
			Some(argument) => return Err(format!("unknown argument {argument:?}")),
			None => return Err("arguments must be valid Unicode option names".to_owned()),
		}
	}
	let root = root.ok_or_else(|| "--root is required".to_owned())?;
	Ok(Some(Options { root, transport: transport.unwrap_or(Transport::Stdio), lsp_configs }))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_socket_transport() {
		let options =
			parse(["--root", "/work/project", "--socket", "/run/user/1/doc.sock"].map(Into::into))
				.expect("valid options")
				.expect("run options");
		assert_eq!(options.root, PathBuf::from("/work/project"));
		assert_eq!(options.transport, Transport::Socket(PathBuf::from("/run/user/1/doc.sock")));
		assert_eq!(options.lsp_configs.len(), 0);
	}

	#[test]
	fn defaults_to_standard_io_and_rejects_conflicts() {
		let options = parse(["--root", "."].map(Into::into))
			.expect("valid options")
			.expect("run options");
		assert_eq!(options.transport, Transport::Stdio);
		assert!(parse(["--root", ".", "--stdio", "--socket", "doc.sock"].map(Into::into)).is_err());
	}

	#[test]
	fn preserves_repeated_lsp_config_order() {
		let options = parse(
			["--root", ".", "--lsp-config", "rust.json", "--lsp-config", "typescript.json"]
				.map(Into::into),
		)
		.expect("valid options")
		.expect("run options");
		assert_eq!(options.lsp_configs, [
			PathBuf::from("rust.json"),
			PathBuf::from("typescript.json")
		]);
		assert!(parse(["--root", ".", "--lsp-config"].map(Into::into)).is_err());
	}
}
