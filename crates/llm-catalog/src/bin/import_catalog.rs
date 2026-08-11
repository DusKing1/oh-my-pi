//! Rebuilds the bundled model catalog from a generated source snapshot.

use std::{env, fs, path::PathBuf};

use omp_llm_catalog::models::import_catalog_zstd;

const DEFAULT_SOURCE: &str = "packages/catalog/src/models.json";

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let mut arguments = env::args_os().skip(1);
	let source = arguments
		.next()
		.map_or_else(|| PathBuf::from(DEFAULT_SOURCE), PathBuf::from);
	let destination = arguments.next().map_or_else(
		|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models.json.zst"),
		PathBuf::from,
	);
	if arguments.next().is_some() {
		return Err("usage: import_catalog [SOURCE_JSON] [DESTINATION_ZST]".into());
	}
	let input = fs::read(&source)?;
	let payload = import_catalog_zstd(&input)?;
	fs::write(destination, payload)?;
	Ok(())
}
