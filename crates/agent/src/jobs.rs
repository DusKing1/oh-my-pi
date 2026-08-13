//! Detached-job registration and authoritative settlement delivery.

use std::{
	collections::{BTreeMap, btree_map::Entry},
	sync::Arc,
};

use bytes::Bytes;
use omp_core::{Str, fmts};
use omp_env::{EnvClient, ProcessAttachmentEvent};
use omp_proto::{
	blob::v1::Chunk,
	env::v1::{AttachOutput, ExecStatusMsg, ProcessInfo, ProcessOutput, ProcessState},
	thread::v1 as thread,
};
use omp_tool::{ArtifactLifetime, JobOwner, JobRef};
use parking_lot::{Mutex, MutexGuard};
use serde::Serialize;
use tokio::task::AbortHandle;

use crate::mailbox::{Interrupt, InterruptClass, InterruptSource, MailboxSender};

const SETTLEMENT_MEDIA_TYPE: &str = "application/vnd.omp.process-settlement+json";
const UPLOAD_CHUNK_BYTES: usize = 16 * 1024;

/// Thread-safe registry and structural supervisor for detached jobs.
///
/// The environment remains the resource owner. Each registration starts one
/// attachment watcher; dropping the last board handle aborts every watcher.
#[derive(Clone)]
pub struct JobBoard {
	inner: Arc<JobBoardInner>,
}

struct JobBoardInner {
	env: EnvClient,
	mailbox: MailboxSender,
	pending: Mutex<BTreeMap<Str, JobRef>>,
	watchers: Mutex<BTreeMap<Str, AbortHandle>>,
}

impl Drop for JobBoardInner {
	fn drop(&mut self) {
		for (_, watcher) in std::mem::take(self.watchers.get_mut()) {
			watcher.abort();
		}
	}
}

impl JobBoard {
	/// Creates an empty board over the authoritative environment client.
	pub fn new(env: EnvClient, mailbox: MailboxSender) -> Self {
		Self {
			inner: Arc::new(JobBoardInner {
				env,
				mailbox,
				pending: Mutex::new(BTreeMap::new()),
				watchers: Mutex::new(BTreeMap::new()),
			}),
		}
	}

	/// Registers and starts watching one detached job.
	///
	/// Returns `true` when inserted. An exact or conflicting duplicate stable ID
	/// returns `false` without replacing the first descriptor or watcher. This
	/// method must be called from a Tokio runtime.
	pub fn register(&self, job: JobRef) -> bool {
		let mut pending = self.inner.pending.lock();
		match pending.entry(job.id.clone()) {
			Entry::Vacant(entry) => {
				entry.insert(job.clone());
			},
			Entry::Occupied(_) => return false,
		}

		let id = job.id.clone();
		let registration_id = id.clone();
		let weak = Arc::downgrade(&self.inner);
		let env = self.inner.env.clone();
		let watcher = tokio::spawn(async move {
			let item = match watch_job(&env, &job).await {
				Ok(item) => item,
				Err(reason) => settlement_error_item(&job, &reason),
			};
			if let Some(inner) = weak.upgrade() {
				let _ = inner.deliver(&id, item);
				inner.watchers.lock().remove(&id);
			}
		})
		.abort_handle();
		self.inner.watchers.lock().insert(registration_id, watcher);
		drop(pending);
		true
	}

	/// Settles a pending job with a caller-supplied canonical item.
	///
	/// This idempotent seam is used by authoritative settlement recovery and
	/// tests. Normal named-process settlement is produced by the board's watcher.
	pub fn settle(
		&self,
		job_id: &str,
		item: thread::Item,
	) -> Result<bool, flume::TrySendError<Interrupt>> {
		let delivered = self.inner.deliver(job_id, item)?;
		if delivered
			&& let Some(watcher) = self.inner.watchers.lock().remove(job_id)
		{
			watcher.abort();
		}
		Ok(delivered)
	}

	/// Borrows pending jobs in stable identifier order without allocating.
	pub fn pending(&self) -> PendingJobs<'_> {
		PendingJobs { guard: self.inner.pending.lock() }
	}

	/// Returns the number of jobs awaiting settlement.
	pub fn len(&self) -> usize {
		self.inner.pending.lock().len()
	}

	/// Returns whether no jobs await settlement.
	pub fn is_empty(&self) -> bool {
		self.inner.pending.lock().is_empty()
	}
}

impl JobBoardInner {
	fn deliver(
		&self,
		job_id: &str,
		item: thread::Item,
	) -> Result<bool, flume::TrySendError<Interrupt>> {
		let mut pending = self.pending.lock();
		let Some(job) = pending.get(job_id) else {
			return Ok(false);
		};
		self.mailbox.try_enqueue(Interrupt {
			class: InterruptClass::TurnBoundary,
			item,
			source: InterruptSource::Job { id: job.id.clone() },
		})?;
		pending.remove(job_id);
		Ok(true)
	}
}

/// Locked, allocation-free view of jobs awaiting settlement.
pub struct PendingJobs<'a> {
	guard: MutexGuard<'a, BTreeMap<Str, JobRef>>,
}

impl PendingJobs<'_> {
	/// Iterates descriptors in stable job-identifier order.
	pub fn iter(
		&self,
	) -> impl DoubleEndedIterator<Item = &JobRef> + ExactSizeIterator + Clone + '_ {
		self.guard.values()
	}

	/// Returns the number of jobs in this view.
	pub fn len(&self) -> usize {
		self.guard.len()
	}

	/// Returns whether this view contains no jobs.
	pub fn is_empty(&self) -> bool {
		self.guard.is_empty()
	}
}

async fn watch_job(env: &EnvClient, job: &JobRef) -> Result<thread::Item, Str> {
	let JobOwner::NamedProcess { name, generation } = &job.owner;
	let mut attachment = env
		.attach_output(AttachOutput {
			name: name.to_string(),
			after_sequence: 0,
			props: None,
		})
		.await
		.map_err(|error| fmts!("could not attach to named process: {error}"))?;
	let attached = match attachment
		.next_event()
		.await
		.map_err(|error| fmts!("named-process attachment failed: {error}"))?
	{
		Some(ProcessAttachmentEvent::Attached(attached)) => attached,
		Some(_) => return Err(Str::from("named-process attachment omitted acknowledgement")),
		None => return Err(Str::from("named-process attachment closed before acknowledgement")),
	};
	if attached.name != name.as_str() || attached.generation != *generation {
		return Err(fmts!(
			"named-process attachment generation mismatch: expected {name}@{generation}, got {}@{}",
			attached.name,
			attached.generation
		));
	}

	let upload = env
		.blob_put()
		.await
		.map_err(|error| fmts!("could not open settlement artifact upload: {error}"))?;
	let mut header = serde_json::to_vec(&ArtifactHeader {
		job_id: job.id.as_str(),
		owner: OwnerRecord { name: name.as_str(), generation: *generation },
		expected_artifact: ExpectedArtifactRecord {
			description: job.artifact.description.as_str(),
			media_type: job.artifact.media_type.as_deref(),
			lifetime: job.artifact.lifetime,
		},
	})
	.map_err(|error| fmts!("could not encode settlement header: {error}"))?;
	if header.pop() != Some(b'}') {
		return Err(Str::from("settlement header was not a JSON object"));
	}
	header.extend_from_slice(b",\"output\":[");
	upload_bytes(&upload, &header).await?;
	let mut first_output = true;

	loop {
		let event = attachment
			.next_event()
			.await
			.map_err(|error| fmts!("named-process attachment failed: {error}"))?
			.ok_or_else(|| Str::from("named-process attachment closed before terminal state"))?;
		match event {
			ProcessAttachmentEvent::Attached(_) => {
				return Err(Str::from("named-process attachment repeated acknowledgement"));
			},
			ProcessAttachmentEvent::Output(output) => {
				validate_output(&output, name, *generation)?;
				let mut encoded = serde_json::to_vec(&OutputRecord {
					sequence: output.sequence,
					channel: output.channel,
					data: &output.data,
				})
				.map_err(|error| fmts!("could not encode process output: {error}"))?;
				if !first_output {
					encoded.insert(0, b',');
				}
				first_output = false;
				upload_bytes(&upload, &encoded).await?;
			},
			ProcessAttachmentEvent::State(state) => {
				let info = state
					.process
					.ok_or_else(|| Str::from("named-process state omitted process info"))?;
				validate_state(&info, name, *generation)?;
				if terminal_state(&info) {
					return finish_settlement(upload, job, info).await;
				}
			},
			ProcessAttachmentEvent::StreamError(error) => {
				return Err(fmts!("named-process stream failed: {error:?}"));
			},
		}
	}
}

async fn finish_settlement(
	upload: omp_env::BlobUpload,
	job: &JobRef,
	info: ProcessInfo,
) -> Result<thread::Item, Str> {
	let mut suffix = Vec::from(&b"],\"state\":"[..]);
	serde_json::to_writer(&mut suffix, &StateRecord::from(&info))
		.map_err(|error| fmts!("could not encode terminal process state: {error}"))?;
	suffix.push(b'}');
	upload_bytes(&upload, &suffix).await?;
	let stored = upload
		.commit()
		.await
		.map_err(|error| fmts!("could not commit settlement artifact: {error}"))?;
	let state = ProcessState::try_from(info.state)
		.map_or_else(|_| format!("state {}", info.state), |state| format!("{state:?}"));
	let text = format!("Detached job {} settled: {}.", job.id, state.to_lowercase());
	let mime = SETTLEMENT_MEDIA_TYPE.to_owned();
	Ok(system_item(vec![
		thread::Part { kind: Some(thread::part::Kind::Text(text)) },
		thread::Part {
			kind: Some(thread::part::Kind::Blob(thread::Blob {
				hash: stored.hash,
				mime,
				size: stored.size,
				inline: Bytes::new(),
				detail: thread::blob::Detail::Auto as i32,
			})),
		},
	]))
}

async fn upload_bytes(upload: &omp_env::BlobUpload, bytes: &[u8]) -> Result<(), Str> {
	for data in bytes.chunks(UPLOAD_CHUNK_BYTES) {
		upload
			.send_chunk(Chunk { data: Bytes::copy_from_slice(data), hash: Bytes::new(), size: None })
			.await
			.map_err(|error| fmts!("could not stream settlement artifact: {error}"))?;
	}
	Ok(())
}

fn validate_output(output: &ProcessOutput, name: &str, generation: u64) -> Result<(), Str> {
	if output.name == name && output.generation == generation {
		Ok(())
	} else {
		Err(fmts!(
			"named-process output generation mismatch: expected {name}@{generation}, got {}@{}",
			output.name,
			output.generation
		))
	}
}

fn validate_state(info: &ProcessInfo, name: &str, generation: u64) -> Result<(), Str> {
	if info.name == name && info.generation == generation {
		Ok(())
	} else {
		Err(fmts!(
			"named-process state generation mismatch: expected {name}@{generation}, got {}@{}",
			info.name,
			info.generation
		))
	}
}

fn terminal_state(info: &ProcessInfo) -> bool {
	matches!(
		ProcessState::try_from(info.state).ok(),
		Some(ProcessState::Exited | ProcessState::Stopped | ProcessState::Failed)
	)
}

fn settlement_error_item(job: &JobRef, reason: &str) -> thread::Item {
	system_item(vec![thread::Part {
		kind: Some(thread::part::Kind::Text(format!(
			"Detached job {} could not be observed to settlement: {reason}",
			job.id
		))),
	}])
}

fn system_item(parts: Vec<thread::Part>) -> thread::Item {
	thread::Item {
		seq: 0,
		created_at_ms: 0,
		kind: Some(thread::item::Kind::Message(thread::Message {
			role: thread::Role::System as i32,
			parts,
		})),
		props: None,
	}
}

#[derive(Serialize)]
struct ArtifactHeader<'a> {
	job_id: &'a str,
	owner: OwnerRecord<'a>,
	expected_artifact: ExpectedArtifactRecord<'a>,
}

#[derive(Serialize)]
struct OwnerRecord<'a> {
	name: &'a str,
	generation: u64,
}

#[derive(Serialize)]
struct ExpectedArtifactRecord<'a> {
	description: &'a str,
	media_type: Option<&'a str>,
	lifetime: ArtifactLifetime,
}

#[derive(Serialize)]
struct OutputRecord<'a> {
	sequence: u64,
	channel: i32,
	data: &'a [u8],
}

#[derive(Serialize)]
struct StateRecord<'a> {
	state: i32,
	status: Option<StatusRecord<'a>>,
}

impl<'a> From<&'a ProcessInfo> for StateRecord<'a> {
	fn from(info: &'a ProcessInfo) -> Self {
		Self { state: info.state, status: info.status.as_ref().map(StatusRecord::from) }
	}
}

#[derive(Serialize)]
struct StatusRecord<'a> {
	outcome: i32,
	exit_code: Option<i32>,
	signal: &'a str,
	wall_clock_ms: u64,
	aborted: bool,
}

impl<'a> From<&'a ExecStatusMsg> for StatusRecord<'a> {
	fn from(status: &'a ExecStatusMsg) -> Self {
		Self {
			outcome: status.outcome,
			exit_code: status.exit_code,
			signal: status.signal.as_str(),
			wall_clock_ms: status.wall_clock_ms,
			aborted: status.aborted,
		}
	}
}

#[cfg(test)]
mod tests {
	use std::{
		sync::atomic::{AtomicUsize, Ordering},
		thread as std_thread,
	};

	use omp_tool::{ArtifactLifetime, ExpectedArtifact};

	use super::*;
	use crate::mailbox::{DrainPoint, Mailbox};

	fn job(id: &str, lifetime: ArtifactLifetime) -> JobRef {
		JobRef {
			id: Str::from(id),
			owner: JobOwner::NamedProcess { name: Str::from(id), generation: 1 },
			artifact: ExpectedArtifact {
				description: Str::from("detached output"),
				media_type: None,
				lifetime,
			},
		}
	}

	#[tokio::test]
	async fn pending_view_is_stable_and_duplicates_preserve_the_first_descriptor() {
		let mailbox = Mailbox::new();
		let (env, _transport) = EnvClient::in_process(0);
		let board = JobBoard::new(env, mailbox.sender());
		assert!(board.register(job("job-b", ArtifactLifetime::Durable)));
		assert!(board.register(job("job-a", ArtifactLifetime::Session)));
		assert!(!board.register(job("job-a", ArtifactLifetime::Ephemeral)));

		let pending = board.pending();
		assert_eq!(pending.len(), 2);
		let mut jobs = pending.iter();
		assert_eq!(jobs.next().unwrap().id, "job-a");
		assert_eq!(jobs.next().unwrap().id, "job-b");
		assert_eq!(jobs.next(), None);
		assert_eq!(
			pending.iter().next().unwrap().artifact.lifetime,
			ArtifactLifetime::Session
		);
	}

	#[tokio::test]
	async fn concurrent_settlement_enqueues_once_and_removes_pending_state() {
		let mut mailbox = Mailbox::new();
		let (env, _transport) = EnvClient::in_process(0);
		let board = JobBoard::new(env, mailbox.sender());
		assert!(board.register(job("job-1", ArtifactLifetime::Session)));
		assert!(!board.settle("unknown", thread::Item::default()).unwrap());
		let settled = AtomicUsize::new(0);
		std_thread::scope(|scope| {
			for seq in 0..8 {
				let board = &board;
				let settled = &settled;
				scope.spawn(move || {
					if board
						.settle("job-1", thread::Item { seq, ..thread::Item::default() })
						.unwrap()
					{
						settled.fetch_add(1, Ordering::Relaxed);
					}
				});
			}
		});

		assert_eq!(settled.load(Ordering::Relaxed), 1);
		assert!(board.is_empty());
		assert_eq!(mailbox.len(), 1);
		let interrupts = mailbox.drain(DrainPoint::TurnBoundary, false);
		assert_eq!(interrupts.len(), 1);
		assert_eq!(interrupts[0].class, InterruptClass::TurnBoundary);
		assert_eq!(
			interrupts[0].source,
			InterruptSource::Job { id: Str::from("job-1") }
		);
		assert!(!board.settle("job-1", thread::Item::default()).unwrap());
		assert!(mailbox.is_empty());
	}
}
