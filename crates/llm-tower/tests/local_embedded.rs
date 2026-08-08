//! Local embedded engine routing behavior.

use std::{collections::BTreeMap, future::Future, sync::Arc};

use bytes::Bytes;
use futures::{StreamExt, stream, stream::BoxStream};
use omp_core::SmolStr;
use omp_llm_catalog::{
	compat::Compat,
	models::{Availability, Modality, ModelCard, ModelCatalog, Source},
	provider::{AuthSpec, Facet as CatalogFacet, ProviderCatalog, ProviderEntry, TransportId},
};
use omp_llm_tower::stack::routing::ProviderRouter;
use omp_llm_transport::embedded::LocalEngine;
use omp_llm_types::{
	Chat, ChatOutcome, ChatRequest, Embed, EmbedRequest, EmbedResponse, EmbeddingVector, Error,
	Executor, Item, ItemKind, Message, Part, Props, Role, SpeakEvent, SpeakRequest, StopReason,
	StreamPartKind, Thread, TranscribeRequest, TranscribeResponse, TurnEvent,
};
use smallvec::{SmallVec, smallvec};

struct Fixture;

impl LocalEngine for Fixture {
	fn chat(
		&self,
		request: ChatRequest,
		_executor: Option<Arc<dyn Executor>>,
	) -> impl Future<Output = Result<BoxStream<'static, TurnEvent>, Error>> + Send + '_ {
		let output = Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::Assistant)
					.parts(vec![Part::Text(SmolStr::new("fixture reply"))])
					.build(),
			))
			.props(Props::default())
			.build();
		let events = [
			TurnEvent::Accepted { replay: false },
			TurnEvent::PartStart {
				index:        0,
				kind:         StreamPartKind::Text,
				tool_call_id: SmolStr::new(""),
				tool_name:    SmolStr::new(""),
			},
			TurnEvent::PartDelta { index: 0, chunk: Bytes::from_static(b"fixture reply") },
			TurnEvent::PartEnd { index: 0, signature: Bytes::new() },
			TurnEvent::Outcome(
				ChatOutcome::builder()
					.output(vec![output])
					.stop(StopReason::EndTurn)
					.unsupported(Vec::new())
					.provider(SmolStr::new("local"))
					.model(request.model)
					.props(Props::default())
					.build(),
			),
		];
		std::future::ready(Ok(stream::iter(events).boxed()))
	}

	fn embed(
		&self,
		request: EmbedRequest,
	) -> impl Future<Output = Result<EmbedResponse, Error>> + Send + '_ {
		let vectors = request
			.texts
			.into_iter()
			.map(|text| {
				EmbeddingVector::builder()
					.values(vec![text.len() as f32, 1.0])
					.build()
			})
			.collect();
		std::future::ready(Ok(EmbedResponse::builder().vectors(vectors).build()))
	}

	fn speak(
		&self,
		_request: SpeakRequest,
	) -> impl Future<Output = Result<BoxStream<'static, SpeakEvent>, Error>> + Send + '_ {
		std::future::ready(Err(Error::Provider(SmolStr::new("unused fixture facet"))))
	}

	fn transcribe(
		&self,
		_request: TranscribeRequest,
	) -> impl Future<Output = Result<TranscribeResponse, Error>> + Send + '_ {
		std::future::ready(Err(Error::Provider(SmolStr::new("unused fixture facet"))))
	}
}

fn router() -> ProviderRouter {
	let facets = smallvec![CatalogFacet::Chat, CatalogFacet::Embeddings];
	let model = ModelCard::builder()
		.id(SmolStr::new("local/fixture"))
		.provider(SmolStr::new("local"))
		.model(SmolStr::new("fixture"))
		.name(SmolStr::new("Deterministic fixture"))
		.family(SmolStr::new("fixture"))
		.facets(facets.clone())
		.inputs(smallvec![Modality::Text])
		.outputs(smallvec![Modality::Text])
		.reasoning(false)
		.efforts(SmallVec::new())
		.context_window(128)
		.max_output_tokens(32)
		.pricing(SmallVec::new())
		.availability(Availability::Available)
		.source(Source::Configured)
		.blocked_until_ms(0)
		.deprecated(false)
		.updated_at_ms(0)
		.props(Props::default())
		.effort_routing(BTreeMap::new())
		.build();
	let provider = ProviderEntry::builder()
		.id(SmolStr::new("local"))
		.transport(TransportId::Embedded)
		.base_url(SmolStr::new(""))
		.fallback_base_urls(SmallVec::new())
		.auth(AuthSpec::default())
		.facets(facets)
		.headers(BTreeMap::new())
		.compat(Compat::default())
		.build();
	let mut providers = ProviderCatalog::new();
	providers.insert(provider.id.clone(), provider);
	let mut router =
		ProviderRouter::new(Arc::new(ModelCatalog::new(vec![model])), Arc::new(providers));
	router.register_local("local", Arc::new(Fixture));
	router
}

#[tokio::test]
async fn embedded_candidate_streams_chat_and_routes_unary_embedding() {
	let router = router();
	let request = ChatRequest::builder()
		.model(SmolStr::new("local/fixture"))
		.thread(Thread::default())
		.tools(Vec::new())
		.build();
	let events = router
		.turn(request, None)
		.await
		.expect("fixture chat is admitted")
		.collect::<Vec<_>>()
		.await;
	assert!(matches!(events.first(), Some(TurnEvent::Accepted { .. })));
	assert!(matches!(
		events.as_slice(),
		[
			TurnEvent::Accepted { .. },
			TurnEvent::PartStart { .. },
			TurnEvent::PartDelta { chunk, .. },
			TurnEvent::PartEnd { signature, .. },
			TurnEvent::Outcome(_),
		] if chunk.as_ref() == b"fixture reply" && signature.is_empty()
	));

	let response = router
		.embed(
			EmbedRequest::builder()
				.model(SmolStr::new("local/fixture"))
				.texts(vec![SmolStr::new("four")])
				.props(Props::default())
				.build(),
		)
		.await
		.expect("fixture embedding is routed");
	assert_eq!(response.vectors[0].values, [4.0, 1.0]);
}
