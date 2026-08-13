//! Real Apple Foundation Models availability and generation smoke path.

use futures::StreamExt;
use omp_llm_inference::local::applefm::{AppleFm, AppleFmEvent, AppleFmOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let evidence = AppleFm::availability_evidence().await?;
	println!("Apple Foundation Models availability: {evidence:?}");
	if !matches!(evidence.state, omp_llm_inference::local::applefm::AppleFmSupportState::Available) {
		println!("Apple Foundation Models unavailable: {evidence:?}");
		return Ok(());
	}
	let model = AppleFm::load().await?;
	let mut stream =
		model.stream(AppleFmOptions::new("Reply with exactly: available").max_tokens(8))?;
	let mut finished = false;
	while let Some(event) = stream.next().await {
		match event? {
			AppleFmEvent::Delta(delta) => print!("{delta}"),
			AppleFmEvent::Finished(generation) => {
				finished = true;
				println!(
					"\nusage_estimate prompt={} completion={} context={}",
					generation.prompt_tokens_estimated,
					generation.completion_tokens_estimated,
					generation.context_size_documented,
				);
			},
		}
	}
	if !finished {
		return Err(
			std::io::Error::new(
				std::io::ErrorKind::UnexpectedEof,
				"Apple Foundation Models stream ended before Finished",
			)
			.into(),
		);
	}
	Ok(())
}
