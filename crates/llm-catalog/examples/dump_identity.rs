//! Dumps normalized model identities as deterministic TSV for regenerating
//! `compat/`; see `compat/README.md`.

use std::io::{self, Write};

use omp_llm_catalog::{ClassificationInput, ClassificationPhase, classify};
use serde::Deserialize;

const MODELS: &str = include_str!("../../../fixtures/llm-oracle/catalog/models.normalized.json");

#[derive(Deserialize)]
struct Fixture<'a> {
	#[serde(borrow)]
	models: Vec<Model<'a>>,
}

#[derive(Deserialize)]
struct Model<'a> {
	id:       &'a str,
	provider: &'a str,
	model:    &'a str,
	behavior: Behavior,
}

#[derive(Deserialize)]
struct Behavior {
	thinking: Option<serde::de::IgnoredAny>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let fixture: Fixture<'_> = serde_json::from_str(MODELS)?;
	let stdout = io::stdout();
	let mut output = io::BufWriter::new(stdout.lock());
	writeln!(output, "id\tprovider\tclass\tfamily\trevision\treasoning")?;

	for model in fixture.models {
		let identity = classify(ClassificationInput {
			phase:          ClassificationPhase::CatalogCompiler,
			provider:       model.provider,
			model:          model.model,
			observed_at_ms: None,
		});
		write!(output, "{}\t{}\t{}\t", model.id, model.provider, identity.class)?;
		if let Some(family) = identity.family {
			write!(output, "{family}")?;
		}
		write!(output, "\t")?;
		if let Some(revision) = identity.revision {
			write!(output, "{}.{}.{}", revision.major, revision.minor, revision.patch)?;
		}
		writeln!(output, "\t{}", model.behavior.thinking.is_some())?;
	}

	Ok(())
}
