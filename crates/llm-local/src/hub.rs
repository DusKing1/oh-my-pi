use std::path::{Path, PathBuf};

use futures::future::try_join_all;
use hf_hub::{HFClient, split_id};
use omp_core::SmolStr;

use crate::Result;

/// A Hugging Face model repository pinned to a branch, tag, or commit.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ModelRepo {
	id:       SmolStr,
	revision: SmolStr,
}

impl ModelRepo {
	/// Targets the repository's `main` revision.
	pub fn new(id: impl Into<SmolStr>) -> Self {
		Self { id: id.into(), revision: "main".into() }
	}

	/// Pins the repository to a branch, tag, or immutable commit.
	pub fn at_revision(mut self, revision: impl Into<SmolStr>) -> Self {
		self.revision = revision.into();
		self
	}

	/// Hugging Face repository identifier, normally `owner/name`.
	pub fn id(&self) -> &str {
		self.id.as_str()
	}

	/// Branch, tag, or commit used for downloads.
	pub fn revision(&self) -> &str {
		self.revision.as_str()
	}
}

impl From<&str> for ModelRepo {
	fn from(id: &str) -> Self {
		Self::new(id)
	}
}

impl From<String> for ModelRepo {
	fn from(id: String) -> Self {
		Self::new(id)
	}
}

/// Cache and network policy for one file download.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct FetchOptions {
	/// Avoid network access and fail when the file is not already cached.
	pub local_files_only: bool,
	/// Revalidate and replace an existing cached object.
	pub force_download:   bool,
}

/// Selection and network policy for a repository snapshot.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SnapshotOptions {
	/// Glob patterns selecting repository-relative paths to fetch.
	pub allow_patterns:   Vec<SmolStr>,
	/// Glob patterns excluding repository-relative paths.
	pub ignore_patterns:  Vec<SmolStr>,
	/// Avoid network access and resolve only an existing cached snapshot.
	pub local_files_only: bool,
	/// Revalidate and replace cached objects.
	pub force_download:   bool,
	/// Maximum number of concurrent file transfers; the Hub default is used when
	/// absent.
	pub max_workers:      Option<usize>,
}

/// Configures a shared asynchronous Hugging Face client.
#[derive(Clone, Debug, Default)]
pub struct HubBuilder {
	cache_dir: Option<PathBuf>,
	token:     Option<SmolStr>,
	endpoint:  Option<SmolStr>,
	offline:   bool,
}

impl HubBuilder {
	/// Stores downloaded models below `cache_dir` instead of the ambient Hugging
	/// Face cache.
	pub fn cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
		self.cache_dir = Some(cache_dir.into());
		self
	}

	/// Uses an explicit Hugging Face access token, including for gated models.
	pub fn token(mut self, token: impl Into<SmolStr>) -> Self {
		self.token = Some(token.into());
		self
	}

	/// Uses a Hugging Face-compatible endpoint instead of `https://huggingface.co`.
	pub fn endpoint(mut self, endpoint: impl Into<SmolStr>) -> Self {
		self.endpoint = Some(endpoint.into());
		self
	}

	/// Restricts every fetch to files already present in the cache.
	pub const fn offline(mut self, offline: bool) -> Self {
		self.offline = offline;
		self
	}

	/// Builds the cheap-to-clone shared Hub client.
	pub fn build(self) -> Result<Hub> {
		let mut builder = HFClient::builder();
		if let Some(cache_dir) = &self.cache_dir {
			builder = builder.cache_dir(cache_dir);
		}
		if let Some(token) = &self.token {
			builder = builder.token(token.as_str());
		}
		if let Some(endpoint) = &self.endpoint {
			builder = builder.endpoint(endpoint.as_str());
		}
		let client = builder.build()?;
		Ok(Hub { client, offline: self.offline })
	}
}

/// Shared asynchronous model downloader backed by the Hugging Face
/// content-addressed cache.
#[derive(Clone)]
pub struct Hub {
	client:  HFClient,
	offline: bool,
}

impl Hub {
	/// Creates a client from `HF_TOKEN`, `HF_HOME`, `HF_HUB_CACHE`, and related
	/// environment variables.
	pub fn new() -> Result<Self> {
		HubBuilder::default().build()
	}

	/// Starts explicit Hub configuration.
	pub fn builder() -> HubBuilder {
		HubBuilder::default()
	}

	/// Returns the cache root used by this client.
	pub fn cache_dir(&self) -> &Path {
		self.client.cache_dir()
	}

	/// Resolves one repository file into the local cache.
	pub async fn file(&self, repo: &ModelRepo, filename: impl AsRef<str>) -> Result<PathBuf> {
		self
			.file_with(repo, filename, FetchOptions::default())
			.await
	}

	/// Resolves one repository file with explicit offline and revalidation
	/// policy.
	pub async fn file_with(
		&self,
		repo: &ModelRepo,
		filename: impl AsRef<str>,
		options: FetchOptions,
	) -> Result<PathBuf> {
		let (owner, name) = split_id(repo.id());
		let path = self
			.client
			.model(owner, name)
			.download_file()
			.filename(filename.as_ref())
			.revision(repo.revision())
			.local_files_only(self.offline || options.local_files_only)
			.force_download(options.force_download)
			.send()
			.await?;
		Ok(path)
	}

	/// Resolves several repository files concurrently while preserving input
	/// order.
	pub async fn files<I, S>(&self, repo: &ModelRepo, filenames: I) -> Result<Vec<PathBuf>>
	where
		I: IntoIterator<Item = S>,
		S: Into<SmolStr>,
	{
		let filenames: Vec<SmolStr> = filenames.into_iter().map(Into::into).collect();
		try_join_all(
			filenames
				.iter()
				.map(|filename| self.file(repo, filename.as_str())),
		)
		.await
	}

	/// Downloads a filtered repository snapshot and returns its local directory.
	pub async fn snapshot(&self, repo: &ModelRepo, options: SnapshotOptions) -> Result<PathBuf> {
		let (owner, name) = split_id(repo.id());
		let allow_patterns = (!options.allow_patterns.is_empty()).then(|| {
			options
				.allow_patterns
				.into_iter()
				.map(String::from)
				.collect::<Vec<_>>()
		});
		let ignore_patterns = (!options.ignore_patterns.is_empty()).then(|| {
			options
				.ignore_patterns
				.into_iter()
				.map(String::from)
				.collect::<Vec<_>>()
		});
		let path = self
			.client
			.model(owner, name)
			.snapshot_download()
			.revision(repo.revision())
			.maybe_allow_patterns(allow_patterns)
			.maybe_ignore_patterns(ignore_patterns)
			.local_files_only(self.offline || options.local_files_only)
			.force_download(options.force_download)
			.maybe_max_workers(options.max_workers)
			.send()
			.await?;
		Ok(path)
	}

	pub(crate) const fn offline(&self) -> bool {
		self.offline
	}
}

impl Default for Hub {
	fn default() -> Self {
		Self::new().expect("default Hugging Face client configuration must be valid")
	}
}
