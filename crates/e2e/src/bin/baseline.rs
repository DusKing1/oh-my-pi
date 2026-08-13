//! Executable P8 performance recorder for retained TUI frames and the agent loop.

use std::{
	future::{self, Future},
	path::{Path, PathBuf},
	pin::Pin,
	sync::Arc,
	task::{Context as TaskContext, Poll},
	time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use omp_agent::{
	Agent, AgentSnapshot, AgentState, Error as TurnError, InvokeFrame, Journal, TurnClient,
	TurnId, TurnInput, TurnOptions, TurnSession,
};
use omp_core::Str;
use omp_env::{EnvClient, InProcessEnvTransport};
use omp_proto::{
	inference::v1::{self as pb, TurnEvent},
	thread::v1::{self as thread, Item},
};
use omp_storage::transcript::{Header, SessionId};
use omp_tool::PromptCaps;
use omp_tui::{Prop, Renderer, Ui, UiContext, components::TextLeaf};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

const SCHEMA_VERSION: u32 = 1;
const DEFAULT_FRAME_TOKENS: usize = 2_048;
const DEFAULT_LOOP_TOKENS: usize = 8_192;
const DEFAULT_SAMPLES: usize = 5;
const GROSS_REGRESSION_LIMIT: f64 = 5.0;
const TOKEN: &str = "·";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct BaselineMetrics {
	pub schema_version: u32,
	pub frame:          FrameMetrics,
	pub r#loop:         LoopMetrics,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct FrameMetrics {
	pub token_count:  usize,
	pub sample_count: usize,
	pub p95_frame_ns: u128,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct LoopMetrics {
	pub tokens_per_sample:      usize,
	pub sample_count:           usize,
	pub raw_duration_ns:        u128,
	pub full_loop_duration_ns:  u128,
	pub raw_tokens_per_second:  f64,
	pub full_tokens_per_second: f64,
	pub slowdown_ratio:         f64,
	pub regression_limit:       f64,
	pub gross_regression:       bool,
}

#[derive(Clone)]
struct ScriptedTurnClient {
	events: Arc<[TurnEvent]>,
}

impl ScriptedTurnClient {
	fn token_storm(tokens: usize) -> Self {
		let mut events = Vec::with_capacity(tokens.saturating_add(4));
		events.push(turn_event(pb::turn_event::Event::Accepted(pb::Accepted { replay: false })));
		events.push(turn_event(pb::turn_event::Event::PartStart(pb::PartStart {
			index:        0,
			kind:         pb::part_start::Kind::Text.into(),
			tool_call_id: String::new(),
			tool_name:    String::new(),
		})));
		let chunk = Bytes::from_static(TOKEN.as_bytes());
		for _ in 0..tokens {
			events.push(turn_event(pb::turn_event::Event::PartDelta(pb::PartDelta {
				index: 0,
				chunk: chunk.clone(),
			})));
		}
		events.push(turn_event(pb::turn_event::Event::PartEnd(pb::PartEnd {
			index:     0,
			signature: Bytes::new(),
		})));
		events.push(turn_event(pb::turn_event::Event::Outcome(pb::Outcome {
			stop:     pb::StopReason::StopEndTurn.into(),
			provider: "scripted".to_owned(),
			model:    "baseline".to_owned(),
			..Default::default()
		})));
		Self { events: events.into() }
	}
}

struct ScriptedTurnSession {
	events: Arc<[TurnEvent]>,
	cursor: usize,
}

struct ScriptedEvents<'a> {
	session: &'a mut ScriptedTurnSession,
}

impl Stream for ScriptedEvents<'_> {
	type Item = Result<TurnEvent, TurnError>;

	fn poll_next(
		mut self: Pin<&mut Self>,
		_context: &mut TaskContext<'_>,
	) -> Poll<Option<Self::Item>> {
		let Some(event) = self.session.events.get(self.session.cursor).cloned() else {
			return Poll::Ready(None);
		};
		self.session.cursor += 1;
		Poll::Ready(Some(Ok(event)))
	}
}

impl TurnSession for ScriptedTurnSession {
	fn events(&mut self) -> impl Stream<Item = Result<TurnEvent, TurnError>> + Send + Unpin + '_ {
		ScriptedEvents { session: self }
	}

	fn submit(
		&mut self,
		_frame: InvokeFrame,
	) -> impl Future<Output = Result<(), TurnError>> + Send + '_ {
		future::ready(Ok(()))
	}
}

impl TurnClient for ScriptedTurnClient {
	type Session<'client> = ScriptedTurnSession;

	fn turn<'client>(
		&'client self,
		_turn_id: TurnId,
		_input: TurnInput,
		_options: &'client TurnOptions,
	) -> impl Future<Output = Result<Self::Session<'client>, TurnError>> + Send + 'client {
		future::ready(Ok(ScriptedTurnSession { events: Arc::clone(&self.events), cursor: 0 }))
	}
}

struct LoopFixture {
	agent:          Agent<ScriptedTurnClient>,
	_env_transport: InProcessEnvTransport,
	_scratch:       TempDir,
}

impl LoopFixture {
	fn new(client: ScriptedTurnClient, ordinal: usize) -> Result<Self> {
		let scratch = tempfile::tempdir().context("create loop baseline scratch directory")?;
		let journal = Journal::create(
			&scratch.path().join("session.jsonl"),
			&Header {
				v:       4,
				id:      SessionId(Str::from(format!("p8-baseline-{ordinal}"))),
				created: 1,
				cwd:     scratch.path().to_owned(),
			},
		)
		.context("create loop baseline journal")?;
		let (env, env_transport) = EnvClient::in_process(8);
		let agent = Agent::new(
			client,
			env,
			AgentState::new(AgentSnapshot::default()),
			journal,
			PromptCaps { maximum_parts: 64, maximum_text_bytes: 1_048_576, media: false },
		);
		Ok(Self { agent, _env_transport: env_transport, _scratch: scratch })
	}

	async fn warm(&mut self, ordinal: usize) -> Result<()> {
		self
			.agent
			.submit([user_item("warmup")], TurnId::new(format!("warmup-{ordinal}")))
			.await
			.context("warm full agent loop")?;
		Ok(())
	}
}

pub(crate) async fn measure(
	frame_tokens: usize,
	loop_tokens: usize,
	samples: usize,
) -> Result<BaselineMetrics> {
	if frame_tokens < 100 || loop_tokens < 100 || samples == 0 {
		bail!("baseline requires at least 100 frame and loop tokens and one sample");
	}
	let frame = measure_frames(frame_tokens)?;
	let scripted = ScriptedTurnClient::token_storm(loop_tokens);
	let raw_duration = measure_raw(&scripted, samples).await?;

	let mut fixtures = Vec::with_capacity(samples);
	for ordinal in 0..samples {
		let mut fixture = LoopFixture::new(scripted.clone(), ordinal)?;
		fixture.warm(ordinal).await?;
		fixtures.push(fixture);
	}
	let measured_inputs: Vec<_> = (0..samples)
		.map(|ordinal| ([user_item("measure")], TurnId::new(format!("measure-{ordinal}"))))
		.collect();
	let mut full_duration = Duration::ZERO;
	for (fixture, (items, turn_id)) in fixtures.iter_mut().zip(measured_inputs) {
		let started = Instant::now();
		fixture
			.agent
			.submit(items, turn_id)
			.await
			.context("measure full agent loop")?;
		full_duration = full_duration.saturating_add(started.elapsed());
	}

	let total_tokens = loop_tokens
		.checked_mul(samples)
		.context("loop token count overflow")?;
	let raw_rate = duration_rate(total_tokens, raw_duration)?;
	let full_rate = duration_rate(total_tokens, full_duration)?;
	let slowdown = slowdown_ratio(raw_rate, full_rate)?;
	Ok(BaselineMetrics {
		schema_version: SCHEMA_VERSION,
		frame,
		r#loop: LoopMetrics {
			tokens_per_sample: loop_tokens,
			sample_count: samples,
			raw_duration_ns: raw_duration.as_nanos(),
			full_loop_duration_ns: full_duration.as_nanos(),
			raw_tokens_per_second: raw_rate,
			full_tokens_per_second: full_rate,
			slowdown_ratio: slowdown,
			regression_limit: GROSS_REGRESSION_LIMIT,
			gross_regression: slowdown > GROSS_REGRESSION_LIMIT,
		},
	})
}

fn measure_frames(tokens: usize) -> Result<FrameMetrics> {
	let root = TextLeaf::new().with(Prop::Id, "stream").text("");
	let mut ui = Ui::from_root(root, 80, UiContext::default());
	let mut renderer = Renderer::new(Vec::<u8>::with_capacity(tokens.saturating_mul(16)));
	ui.present(&mut renderer, 24, 0).context("paint warmup frame")?;

	let mut text = String::with_capacity(tokens.saturating_mul(TOKEN.len()));
	let mut elapsed = Vec::with_capacity(tokens);
	for _ in 0..tokens {
		text.push_str(TOKEN);
		let started = Instant::now();
		if !ui.set_text("stream", text.as_str()) {
			bail!("token-storm text component stopped accepting updates");
		}
		ui.present(&mut renderer, 24, 0).context("paint token-storm frame")?;
		elapsed.push(started.elapsed().as_nanos());
	}
	elapsed.sort_unstable();
	let rank = elapsed.len().saturating_mul(95).div_ceil(100).saturating_sub(1);
	Ok(FrameMetrics {
		token_count: tokens,
		sample_count: elapsed.len(),
		p95_frame_ns: elapsed[rank],
	})
}

async fn measure_raw(client: &ScriptedTurnClient, samples: usize) -> Result<Duration> {
	let inputs: Vec<_> = (0..samples)
		.map(|ordinal| {
			(
				TurnId::new(format!("raw-{ordinal}")),
				TurnInput::Full(thread::Thread { items: vec![user_item("measure")] }),
			)
		})
		.collect();
	let options = TurnOptions::default();
	let mut total = Duration::ZERO;
	for (turn_id, input) in inputs {
		let started = Instant::now();
		let mut session = client.turn(turn_id, input, &options).await?;
		let mut events = session.events();
		while let Some(event) = events.next().await {
			event?;
		}
		total = total.saturating_add(started.elapsed());
	}
	Ok(total)
}

pub(crate) fn duration_rate(tokens: usize, duration: Duration) -> Result<f64> {
	if tokens == 0 {
		bail!("token count must be non-zero");
	}
	if duration.is_zero() {
		bail!("measured duration must be non-zero");
	}
	let rate = tokens as f64 / duration.as_secs_f64();
	if !rate.is_finite() || rate <= 0.0 {
		bail!("token rate is not finite and positive");
	}
	Ok(rate)
}

pub(crate) fn slowdown_ratio(raw_rate: f64, full_rate: f64) -> Result<f64> {
	if !raw_rate.is_finite() || raw_rate <= 0.0 || !full_rate.is_finite() || full_rate <= 0.0 {
		bail!("token rates must be finite and positive");
	}
	Ok(raw_rate / full_rate)
}

fn user_item(text: &str) -> Item {
	Item {
		kind: Some(thread::item::Kind::Message(thread::Message {
			role: thread::Role::User.into(),
			parts: vec![thread::Part {
				kind: Some(thread::part::Kind::Text(text.to_owned())),
			}],
			..Default::default()
		})),
		..Default::default()
	}
}

fn turn_event(event: pb::turn_event::Event) -> TurnEvent {
	TurnEvent { event: Some(event) }
}

pub(crate) fn write_metrics(path: &Path, metrics: &BaselineMetrics) -> Result<()> {
	if let Some(parent) = path.parent()
		&& !parent.as_os_str().is_empty()
	{
		std::fs::create_dir_all(parent)
			.with_context(|| format!("create artifact directory {}", parent.display()))?;
	}
	let bytes = serde_json::to_vec_pretty(metrics).context("serialize baseline metrics")?;
	std::fs::write(path, bytes)
		.with_context(|| format!("write baseline artifact {}", path.display()))?;
	Ok(())
}

fn artifact_argument() -> Result<PathBuf> {
	let mut args = std::env::args_os().skip(1);
	let Some(flag) = args.next() else {
		bail!("usage: baseline --artifact <path>");
	};
	if flag != "--artifact" {
		bail!("expected --artifact <path>");
	}
	let path = args.next().context("--artifact requires a path")?;
	if args.next().is_some() {
		bail!("unexpected arguments after artifact path");
	}
	Ok(path.into())
}

#[tokio::main]
async fn main() -> Result<()> {
	let artifact = artifact_argument()?;
	let metrics = measure(DEFAULT_FRAME_TOKENS, DEFAULT_LOOP_TOKENS, DEFAULT_SAMPLES).await?;
	write_metrics(&artifact, &metrics)?;
	println!("{}", serde_json::to_string(&metrics)?);
	if metrics.r#loop.gross_regression {
		bail!(
			"full-loop throughput regressed {:.2}x versus raw scripted TurnClient (limit {:.2}x)",
			metrics.r#loop.slowdown_ratio,
			metrics.r#loop.regression_limit
		);
	}
	Ok(())
}
