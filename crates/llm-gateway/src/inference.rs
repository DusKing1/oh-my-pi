//! Composed production implementation of the native inference RPC surface.
//!
//! [`TurnEngine`] deliberately remains responsible only for turn orchestration.
//! This service is the transport-facing composition root that delegates context
//! lifecycle, discovery, and independently enabled facets without duplicating
//! their state or policy.

use std::sync::Arc;

use futures::{StreamExt as _, stream::BoxStream};
use omp_core::SmolStr;
use omp_llm_types::{
	CountInput, CountRequest,
	facet::{Error as FacetError, Facet, Facets},
};
use omp_proto::inference::v1::{self as pb, inference_server::Inference};
use tonic::{Request, Response, Status};

use crate::{
	context::{ContextError, ContextStore},
	discovery::{DiscoveryService, WatchModelsStream},
	media::MediaFacets,
	turn::{TurnEngine, TurnStream},
};

/// Stream returned by native image generation.
pub type GenerateImageStream = BoxStream<'static, Result<pb::ImageEvent, Status>>;
/// Stream returned by native speech synthesis.
pub type SpeakStream = BoxStream<'static, Result<pb::SpeakEvent, Status>>;
/// Stream returned by native video attachment.
pub type AttachGenerationStream = BoxStream<'static, Result<pb::GenerationStatus, Status>>;

/// Complete native inference service mounted by a production listener.
///
/// The constructor requires the already-shared runtime components so the native
/// RPCs and foreign facades observe the same contexts, catalog registry, facet
/// implementations, blob store, and durable media-job registry. An absent facet
/// is reported as gRPC `UNIMPLEMENTED`; it is never advertised by
/// [`Self::capabilities`].
#[derive(Clone)]
pub struct InferenceService {
	turn:      TurnEngine,
	contexts:  Arc<ContextStore>,
	discovery: DiscoveryService,
	facets:    Arc<Facets>,
	media:     MediaFacets,
}

impl InferenceService {
	/// Composes the native RPC service from production-owned shared components.
	#[must_use]
	pub const fn new(
		turn: TurnEngine,
		contexts: Arc<ContextStore>,
		discovery: DiscoveryService,
		facets: Arc<Facets>,
		media: MediaFacets,
	) -> Self {
		Self { turn, contexts, discovery, facets, media }
	}

	/// Returns the native capabilities that this exact service can execute.
	///
	/// Core turn, context, and discovery RPCs are always present. Optional facet
	/// identifiers appear only when their implementation is installed.
	#[must_use]
	pub fn capabilities(&self) -> Vec<SmolStr> {
		let mut capabilities = vec![
			SmolStr::new_static("inference.turn"),
			SmolStr::new_static("inference.invoke"),
			SmolStr::new_static("inference.contexts"),
			SmolStr::new_static("inference.models"),
		];
		for (facet, name) in [
			(Facet::CountTokens, "inference.count-tokens"),
			(Facet::Embed, "inference.embed"),
			(Facet::ImageGen, "inference.image"),
			(Facet::Speak, "inference.speak"),
			(Facet::Transcribe, "inference.transcribe"),
			(Facet::VideoGen, "inference.video"),
			(Facet::Search, "search"),
		] {
			if self.facets.supports(facet) {
				capabilities.push(name.into());
			}
		}
		if [Facet::ImageGen, Facet::Speak, Facet::Transcribe, Facet::VideoGen]
			.into_iter()
			.any(|facet| self.facets.supports(facet))
		{
			capabilities.push(SmolStr::new_static("media"));
		}
		capabilities
	}
}

#[tonic::async_trait]
impl Inference for InferenceService {
	type AttachGenerationStream = AttachGenerationStream;
	type GenerateImageStream = GenerateImageStream;
	type SpeakStream = SpeakStream;
	type TurnStream = TurnStream;
	type WatchModelsStream = WatchModelsStream;

	async fn turn(
		&self,
		request: Request<tonic::Streaming<pb::TurnFrame>>,
	) -> Result<Response<Self::TurnStream>, Status> {
		self.turn.turn(request).await
	}

	async fn fork(
		&self,
		request: Request<pb::ForkRequest>,
	) -> Result<Response<pb::ForkResponse>, Status> {
		let request = request.into_inner();
		let parent = request
			.parent
			.ok_or_else(|| Status::invalid_argument("ForkRequest.parent is required"))?
			.try_into()
			.map_err(convert_status)?;
		if request.context_id.is_empty() {
			return Err(Status::invalid_argument("ForkRequest.context_id is required"));
		}
		let revision = self
			.contexts
			.fork(&parent, request.at, request.context_id)
			.map_err(context_status)?;
		Ok(Response::new(pb::ForkResponse { revision: Some(revision.into()) }))
	}

	async fn drop(
		&self,
		request: Request<pb::DropRequest>,
	) -> Result<Response<pb::DropResponse>, Status> {
		let context_id = request.into_inner().context_id;
		if context_id.is_empty() {
			return Err(Status::invalid_argument("DropRequest.context_id is required"));
		}
		if !self.contexts.drop_context(&context_id) {
			return Err(Status::not_found("context is not held by this gateway"));
		}
		Ok(Response::new(pb::DropResponse {}))
	}

	async fn count_tokens(
		&self,
		request: Request<pb::CountTokensRequest>,
	) -> Result<Response<pb::CountTokensResponse>, Status> {
		let counter = self
			.facets
			.count_tokens
			.as_ref()
			.ok_or_else(|| absent_facet("count-tokens"))?;
		let mut request: CountRequest = request.into_inner().try_into().map_err(convert_status)?;
		if let CountInput::Context(context) = &request.input {
			request.input =
				CountInput::Thread(self.contexts.snapshot(context).map_err(context_status)?);
		}
		let response = counter.count(request).await.map_err(facet_status)?;
		Ok(Response::new(response.into()))
	}

	async fn embed(
		&self,
		request: Request<pb::EmbedRequest>,
	) -> Result<Response<pb::EmbedResponse>, Status> {
		let embed = self
			.facets
			.embed
			.as_ref()
			.ok_or_else(|| absent_facet("embeddings"))?;
		let response = embed
			.embed(request.into_inner().try_into().map_err(convert_status)?)
			.await
			.map_err(facet_status)?;
		Ok(Response::new(response.into()))
	}

	async fn generate_image(
		&self,
		request: Request<pb::GenerateImageRequest>,
	) -> Result<Response<Self::GenerateImageStream>, Status> {
		ensure_facet(&self.facets, Facet::ImageGen, "images")?;
		let stream = self
			.media
			.generate_image(request.into_inner().try_into().map_err(convert_status)?)
			.await
			.map_err(facet_status)?
			.map(|event| Ok(event.into()))
			.boxed();
		Ok(Response::new(stream))
	}

	async fn speak(
		&self,
		request: Request<pb::SpeakRequest>,
	) -> Result<Response<Self::SpeakStream>, Status> {
		ensure_facet(&self.facets, Facet::Speak, "speech")?;
		let stream = self
			.media
			.speak(request.into_inner().try_into().map_err(convert_status)?)
			.await
			.map_err(facet_status)?
			.map(|event| Ok(event.into()))
			.boxed();
		Ok(Response::new(stream))
	}

	async fn transcribe(
		&self,
		request: Request<pb::TranscribeRequest>,
	) -> Result<Response<pb::TranscribeResponse>, Status> {
		ensure_facet(&self.facets, Facet::Transcribe, "transcription")?;
		let response = self
			.media
			.transcribe(request.into_inner().try_into().map_err(convert_status)?)
			.await
			.map_err(facet_status)?;
		Ok(Response::new(response.into()))
	}

	async fn generate_video(
		&self,
		request: Request<pb::GenerateVideoRequest>,
	) -> Result<Response<pb::GenerationStatus>, Status> {
		let video = self
			.facets
			.video_gen
			.as_ref()
			.ok_or_else(|| absent_facet("video"))?;
		let handle = video
			.submit(request.into_inner().try_into().map_err(convert_status)?)
			.await
			.map_err(facet_status)?;
		let status = video.get(handle).await.map_err(facet_status)?;
		Ok(Response::new(status.into()))
	}

	async fn get_generation(
		&self,
		request: Request<pb::GetGenerationRequest>,
	) -> Result<Response<pb::GenerationStatus>, Status> {
		let video = self
			.facets
			.video_gen
			.as_ref()
			.ok_or_else(|| absent_facet("video"))?;
		let handle = generation_handle(request.into_inner().generation_id)?;
		let status = video.get(handle).await.map_err(facet_status)?;
		Ok(Response::new(status.into()))
	}

	async fn attach_generation(
		&self,
		request: Request<pb::AttachGenerationRequest>,
	) -> Result<Response<Self::AttachGenerationStream>, Status> {
		let video = self
			.facets
			.video_gen
			.as_ref()
			.ok_or_else(|| absent_facet("video"))?;
		let handle = generation_handle(request.into_inner().generation_id)?;
		let stream = video
			.attach(handle)
			.await
			.map_err(facet_status)?
			.map(|status| Ok(status.into()))
			.boxed();
		Ok(Response::new(stream))
	}

	async fn cancel_generation(
		&self,
		request: Request<pb::CancelGenerationRequest>,
	) -> Result<Response<pb::GenerationStatus>, Status> {
		let video = self
			.facets
			.video_gen
			.as_ref()
			.ok_or_else(|| absent_facet("video"))?;
		let handle = generation_handle(request.into_inner().generation_id)?;
		let status = video.cancel(handle).await.map_err(facet_status)?;
		Ok(Response::new(status.into()))
	}

	async fn search(
		&self,
		request: Request<pb::SearchRequest>,
	) -> Result<Response<pb::SearchResponse>, Status> {
		let search = self
			.facets
			.search
			.as_ref()
			.ok_or_else(|| absent_facet("search"))?;
		let response = search
			.search(request.into_inner().try_into().map_err(convert_status)?)
			.await
			.map_err(facet_status)?;
		Ok(Response::new(response.into()))
	}

	async fn list_providers(
		&self,
		request: Request<pb::ListProvidersRequest>,
	) -> Result<Response<pb::ListProvidersResponse>, Status> {
		self.discovery.list_providers(request).await
	}

	async fn list_models(
		&self,
		request: Request<pb::ListModelsRequest>,
	) -> Result<Response<pb::ListModelsResponse>, Status> {
		self.discovery.list_models(request).await
	}

	async fn watch_models(
		&self,
		request: Request<pb::WatchModelsRequest>,
	) -> Result<Response<Self::WatchModelsStream>, Status> {
		self.discovery.watch_models(request).await
	}

	async fn refresh_models(
		&self,
		request: Request<pb::RefreshModelsRequest>,
	) -> Result<Response<pb::ListModelsResponse>, Status> {
		self.discovery.refresh_models(request).await
	}
}

fn ensure_facet(facets: &Facets, facet: Facet, name: &'static str) -> Result<(), Status> {
	if facets.supports(facet) {
		Ok(())
	} else {
		Err(absent_facet(name))
	}
}

fn absent_facet(name: &'static str) -> Status {
	Status::unimplemented(format!("{name} facet is not enabled"))
}

fn generation_handle(id: String) -> Result<omp_llm_types::facet::GenerationHandle, Status> {
	if id.is_empty() {
		return Err(Status::invalid_argument("generation_id is required"));
	}
	Ok(omp_llm_types::facet::GenerationHandle::builder()
		.id(id.into())
		.build())
}

fn convert_status(_error: omp_llm_types::ConvertError) -> Status {
	Status::invalid_argument("invalid inference RPC payload")
}

fn facet_status(error: FacetError) -> Status {
	match error {
		FacetError::Unsupported(_) => {
			Status::failed_precondition("requested facet capability is unavailable")
		},
		FacetError::Provider(_) => Status::unknown("provider operation failed"),
		FacetError::Transport(_) => Status::unavailable("provider transport failed"),
		_ => Status::unknown("inference operation failed"),
	}
}

fn context_status(error: ContextError) -> Status {
	match &error {
		ContextError::NeedFull => Status::not_found(error.to_string()),
		ContextError::Busy | ContextError::DedupWindowFull | ContextError::ContextCapacity => {
			Status::resource_exhausted(error.to_string())
		},
		ContextError::Conflict { .. }
		| ContextError::AlreadyExists
		| ContextError::InvalidTruncate { .. }
		| ContextError::TurnIdReuse => Status::failed_precondition(error.to_string()),
		ContextError::OutputAlreadyAccumulated | ContextError::InactiveGuard => {
			Status::internal(error.to_string())
		},
	}
}

#[cfg(test)]
mod security_tests {
	use omp_llm_types::facet::Error as FacetError;

	use super::facet_status;

	#[test]
	fn facet_status_preserves_code_without_provider_diagnostics() {
		const CANARY: &str = "canary-provider-token-in-rpc-status";
		for (error, code) in [
			(FacetError::Provider(CANARY.into()), tonic::Code::Unknown),
			(FacetError::Transport(CANARY.into()), tonic::Code::Unavailable),
		] {
			let status = facet_status(error);
			assert_eq!(status.code(), code);
			assert!(!status.message().contains(CANARY));
			assert!(!format!("{status:?}").contains(CANARY));
		}
	}
}
