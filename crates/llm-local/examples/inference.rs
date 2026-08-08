//! Exercises canonical chat and embedding requests over an embedded local
//! engine.
//!
//! The text backend is `TextSelection::Auto`: Apple Foundation Models when the
//! machine supports it, otherwise the default curated small GGUF model.

use std::sync::Arc;

use futures::StreamExt;
use omp_core::SmolStr;
use omp_llm_local::{Embedded, Inference, SmallModel, TextSelection, types};
use types::{Chat, Embed};

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
	let text = match std::env::args().nth(1).as_deref() {
		Some("gguf") => TextSelection::Gguf(SmallModel::default().into()),
		_ => TextSelection::Auto,
	};
	let inference = Arc::new(
		Inference::builder()
			.text(text)
			.embeddings(Default::default())
			.build()
			.await?,
	);
	let embedded = Embedded::new(inference.clone());
	let item = |role, text| {
		types::Item::builder()
			.seq(0)
			.kind(types::ItemKind::Message(
				types::Message::builder()
					.role(role)
					.parts(vec![types::Part::Text(SmolStr::new(text))])
					.build(),
			))
			.props(types::Props::default())
			.build()
	};
	let request = types::ChatRequest::builder()
		.model(SmolStr::new("local/default"))
		.thread(
			types::Thread::builder()
				.items(vec![
					item(types::Role::System, "Answer with one short sentence."),
					item(types::Role::User, "What color is the sky on a clear day?"),
				])
				.build(),
		)
		.tools(Vec::new())
		.sampling(types::Sampling::builder().max_output_tokens(64).build())
		.build();
	let mut stream = embedded.turn(request, None).await?;
	while let Some(event) = stream.next().await {
		if let types::TurnEvent::Outcome(outcome) = event {
			println!("provider={} model={} stop={:?}", outcome.provider, outcome.model, outcome.stop);
		}
	}

	let response = embedded
		.embed(
			types::EmbedRequest::builder()
				.model(SmolStr::new("local/default"))
				.texts(vec![SmolStr::new("local inference")])
				.props(types::Props::default())
				.build(),
		)
		.await?;
	let dims = response
		.vectors
		.first()
		.map_or(0, |vector| vector.values.len());
	println!("embed: {} vector(s) x {dims} dims", response.vectors.len());

	inference.shutdown().await?;
	Ok(())
}
