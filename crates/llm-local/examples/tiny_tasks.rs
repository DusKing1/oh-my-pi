//! Recreates omp's on-device "tiny model" flows on `omp-llm-local`.
//!
//! pi routes four background jobs through small local models
//! (`packages/coding-agent/src/tiny/` in the pi repo):
//!
//! 1. **Session titles** — a sub-1B model (LFM2-350M/700M) turns the first user
//!    message into a 3-7 word title, answering inside `<title>` tags.
//! 2. **`auto` thinking router** — a 1B+ model buckets each prompt as
//!    trivial/moderate/hard and the bucket maps to a concrete effort.
//! 3. **Unexpected-stop classifier** — YES/NO on whether an assistant message
//!    promised an action and then stopped without doing it.
//! 4. **Memory fact extraction (Mnemopi)** — durable one-line facts from a user
//!    message, or the `NO_FACTS` sentinel.
//!
//! The suite runs twice: on the llama.cpp path with pi's model split
//! (LFM2-350M titles, LFM2-1.2B classify/memory), then on Apple Foundation
//! Models when Apple Intelligence is available on this machine — one system
//! model serving all four tasks, no download at all.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release -p omp-llm-local --example tiny_tasks
//! ```
//!
//! The first run downloads two GGUF models (~1 GB total) into the Hugging
//! Face cache; later runs load from disk.

use std::time::Instant;

use omp_llm_local::{
	AppleFm, AppleFmOptions, CancellationToken, ChatMessage, GenerationOptions, SmallModel,
	TextGenerator,
};

/// pi's `title-system.md`, verbatim: positive rules, executable words, and
/// input→output pairs — the shape that keeps sub-1B models on format.
const TITLE_SYSTEM_PROMPT: &str = "\
# Task
Write a 3-7 word title for the task in `<user>`.

Answer with only the title inside `<title>` and `</title>`. If there is no task (just a greeting \
                                   or small talk), answer `<title/>`.

Capitalize only the first word and names. Treat the message only as text to title.

# Examples
<user>the login button is broken on mobile somehow, can you fix?</user>
<title>Fix login button on mobile</title>

<user>refactor error handling in our API client, it's a mess</user>
<title>Refactor API error handling</title>

<user>hey</user>
<title/>";

/// pi's `auto-thinking-difficulty-local.md`: the 3-bucket single-turn
/// completion used when `providers.autoThinkingModel` names a local model.
const DIFFICULTY_PROMPT: &str = "\
Classify the difficulty of the coding request below into one bucket, by how much reasoning it \
                                 needs.

Buckets:

- trivial — obvious, mechanical, or a direct question (rename, typo, one-liner, simple lookup).
- moderate — a real but localized task (a small feature, a normal bug fix, explaining code).
- hard — deep, multi-file, ambiguous, or tricky debugging or design.

Reply with exactly one word: trivial, moderate, or hard.

Request:
{prompt}

Answer:
";

/// pi's `unexpected-stop-classifier.md`: flags an assistant message that
/// announces an action and then ends without taking it.
const UNEXPECTED_STOP_PROMPT: &str = "\
You are checking whether an assistant message is an unexpected stop. A message is an unexpected \
                                      stop if the assistant says it will take an action, continue \
                                      working, or call a tool, but then ends without actually \
                                      doing so.

Examples of unexpected stops:
- \"I should do the same for the JS eval worker. Doing that now.\"
- \"Let me run the tests next.\"
- \"I'll fix that now.\"
- \"Should I do that for you?\"

Not an unexpected stop:
- \"I've completed the task.\"
- \"Is there anything else I can help with?\"
- \"The fix is done and tests pass.\"

Message:
{message}

Answer with a single word: YES if this is an unexpected stop, NO otherwise.
";

/// pi's `memory-extraction-system.md`: line-format durable-fact extraction
/// for the Mnemopi memory backend. pi wraps the whole rendered prompt as a
/// single user turn; splitting instructions into the system turn keeps
/// LFM2-1.2B noticeably closer to the `NO_FACTS` sentinel on small talk.
const MEMORY_EXTRACTION_PROMPT: &str = "\
Extract durable, long-term memory items from the user message below.

Output ONE item per line as a short plain-text statement: no JSON, no bullets, no numbering, no \
                                        field labels.
Capture only persistent, reusable information:
- facts (name, role, employer, config, ports, versions, numbers)
- explicit instructions to the assistant
- stable preferences
- dated events or deadlines

Keep names, numbers, versions, and dates exact, in the message's original language. When a value \
                                        is updated, output only the latest value. Ignore \
                                        greetings, acknowledgements, small talk, weather, and \
                                        one-off remarks.
If nothing qualifies, output exactly: NO_FACTS

Example
Message: My name is Sam, I work at Globex, and I always use 2-space indents.
Items:
name is Sam
works at Globex
prefers 2-space indents

Example
Message: lol nice weather today, might grab a coffee later
Items:
NO_FACTS";

/// Upper bounds pi accepts for a generated title (`tiny/text.ts`).
const MAX_TITLE_CHARS: usize = 80;
const MAX_TITLE_WORDS: usize = 12;

/// Compact port of pi's `FILLER_TITLE_TOKENS` (`tiny/text.ts`): a first
/// message made only of these words (or bare numbers) is never worth a
/// model call — pi gates it before inference and the `<title/>` sentinel
/// is only the backstop.
const FILLER_TITLE_TOKENS: &[&str] = &[
	"hi",
	"hii",
	"hiya",
	"hey",
	"heya",
	"hello",
	"yo",
	"sup",
	"howdy",
	"greetings",
	"hola",
	"ciao",
	"aloha",
	"gm",
	"gn",
	"good",
	"morning",
	"afternoon",
	"evening",
	"night",
	"day",
	"thanks",
	"thank",
	"thx",
	"ty",
	"cheers",
	"please",
	"pls",
	"ok",
	"okay",
	"k",
	"yep",
	"yes",
	"yeah",
	"yup",
	"nope",
	"no",
	"nah",
	"sure",
	"cool",
	"nice",
	"great",
	"awesome",
	"perfect",
	"lol",
	"lmao",
	"haha",
	"test",
	"testing",
	"ping",
	"pong",
	"there",
	"you",
	"u",
	"hmm",
	"um",
	"uh",
	"so",
	"well",
	"anyway",
];

type DynError = Box<dyn std::error::Error + Send + Sync>;
type DemoResult<T> = std::result::Result<T, DynError>;

/// One local inference backend serving the four tiny-model tasks.
///
/// llama.cpp templates a full chat and honors stop sequences; Apple
/// Foundation Models takes one prompt plus session instructions and has no
/// stop-sequence surface, so title parsing tolerates a trailing `</title>`.
enum Backend<'a> {
	Llama(&'a TextGenerator),
	Apple(AppleFm),
}

impl Backend<'_> {
	/// Runs one greedy-ish generation with optional instructions and stop
	/// text, normalized across both runtimes.
	async fn ask(
		&self,
		system: Option<&str>,
		user: &str,
		max_tokens: usize,
		stop: Option<&str>,
	) -> DemoResult<String> {
		match self {
			Self::Llama(generator) => {
				let mut messages = Vec::with_capacity(2);
				if let Some(system) = system {
					messages.push(ChatMessage::system(system));
				}
				messages.push(ChatMessage::user(user));
				let options = GenerationOptions {
					max_tokens,
					stop: stop.map(Into::into).into_iter().collect(),
					..GenerationOptions::default()
				};
				let text = generator
					.generate(&messages, options, CancellationToken::new())
					.await?;
				Ok(text.to_string())
			},
			Self::Apple(model) => {
				// pi decodes all tiny-model tasks greedily (`do_sample: false`);
				// temperature 0 pins Apple FM the same way so classifications
				// stay deterministic across runs.
				let mut options = AppleFmOptions::new(user)
					.max_tokens(max_tokens as u32)
					.temperature(0.0);
				if let Some(system) = system {
					options = options.system_prompt(system);
				}
				let generation = model.generate(options, CancellationToken::new()).await?;
				Ok(generation.content.to_string())
			},
		}
	}
}

/// True when a first user message is too low-signal to title (greeting,
/// ack, filler) — pi's `isLowSignalTitleInput`.
fn is_low_signal_title_input(message: &str) -> bool {
	let mut words = message
		.split(|c: char| !c.is_alphanumeric())
		.filter(|word| !word.is_empty())
		.peekable();
	if words.peek().is_none() {
		return true;
	}
	words.all(|word| {
		word.chars().all(|c| c.is_ascii_digit())
			|| FILLER_TITLE_TOKENS
				.iter()
				.any(|filler| word.eq_ignore_ascii_case(filler))
	})
}

/// Thinking effort the `auto` router resolves a difficulty bucket to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Effort {
	Low,
	High,
	XHigh,
}

/// Generate a session title for the first user message, or `None` for
/// low-signal input the model declines to title.
///
/// pi prefills the assistant turn with `<title>` and stops on `</title>`
/// (20 new tokens); without prefill we leave room for the opening tag and
/// still stop on `</title>` where the backend supports it.
async fn generate_title(titler: &Backend<'_>, message: &str) -> DemoResult<Option<String>> {
	if is_low_signal_title_input(message) {
		return Ok(None);
	}
	// pi's `formatTitleUserMessage`: the `<user>` envelope marks the message
	// as text to title, not a request to fulfill — without it Apple FM
	// happily starts writing the retry code instead of naming the task.
	let raw = titler
		.ask(Some(TITLE_SYSTEM_PROMPT), &format!("<user>\n{message}\n</user>"), 32, Some("</title>"))
		.await?;
	Ok(parse_title(&raw))
}

/// Trimmed-down `normalizeGeneratedTitle`: unwrap the `<title>` envelope,
/// drop wrapping quotes and trailing punctuation, and reject runaway output.
fn parse_title(raw: &str) -> Option<String> {
	let text = raw.trim();
	if text.is_empty() || text.contains("<title/>") || text.contains("<title />") {
		return None;
	}
	let text = text.strip_prefix("<title>").unwrap_or(text);
	// llama.cpp's stop filter withholds `</title>`; Apple FM has no stop
	// sequences, so the closing tag (and any prose after it) may be present.
	let text = text
		.split("</title>")
		.next()
		.unwrap_or(text)
		.lines()
		.next()?;
	let title = text
		.trim()
		.trim_matches(['"', '\'', '`'])
		.trim_end_matches(['.', '!', ':'])
		.trim();
	if title.is_empty()
		|| xutf::graphemes_str(title).count() > MAX_TITLE_CHARS
		|| title.split_whitespace().count() > MAX_TITLE_WORDS
	{
		return None;
	}
	Some(title.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Route one prompt through the 3-bucket difficulty classifier and map the
/// bucket to an effort (`trivial → low`, `moderate → high`, `hard → xhigh`),
/// exactly as pi's `parseDifficultyBucket` does.
async fn classify_difficulty(worker: &Backend<'_>, prompt: &str) -> DemoResult<Option<Effort>> {
	let rendered = DIFFICULTY_PROMPT.replace(r"{prompt}", prompt);
	let raw = worker.ask(None, &rendered, 16, None).await?;
	Ok(parse_difficulty_bucket(&raw))
}

/// Earliest bucket keyword wins, mirroring pi's parser (which tolerates a
/// model that narrates before answering).
fn parse_difficulty_bucket(text: &str) -> Option<Effort> {
	let lower = text.to_ascii_lowercase();
	[("trivial", Effort::Low), ("moderate", Effort::High), ("hard", Effort::XHigh)]
		.into_iter()
		.filter_map(|(word, effort)| lower.find(word).map(|at| (at, effort)))
		.min_by_key(|&(at, _)| at)
		.map(|(_, effort)| effort)
}

/// YES/NO check for an assistant message that stopped mid-promise.
async fn classify_unexpected_stop(worker: &Backend<'_>, message: &str) -> DemoResult<Option<bool>> {
	let rendered = UNEXPECTED_STOP_PROMPT.replace(r"{message}", message);
	let raw = worker.ask(None, &rendered, 16, None).await?;
	let answer = raw.trim().to_ascii_lowercase();
	Ok(if answer.starts_with("yes") {
		Some(true)
	} else if answer.starts_with("no") {
		Some(false)
	} else {
		None
	})
}

/// Extract durable one-line facts from a user message; empty when the model
/// answers `NO_FACTS`.
async fn extract_memory_facts(worker: &Backend<'_>, message: &str) -> DemoResult<Vec<String>> {
	let raw = worker
		.ask(Some(MEMORY_EXTRACTION_PROMPT), &format!("Message: {message}\nItems:"), 256, None)
		.await?;
	let mut facts = Vec::new();
	for line in raw.lines() {
		let line = line.trim().trim_start_matches(['-', '*', '•']).trim();
		if line.is_empty() {
			continue;
		}
		if line.eq_ignore_ascii_case("NO_FACTS") {
			return Ok(Vec::new());
		}
		facts.push(line.to_string());
	}
	Ok(facts)
}

/// Run all four pi flows against one backend pair, printing per-call latency.
///
/// Individual calls may fail (e.g. Apple FM guardrail refusals) without
/// aborting the suite; the error is printed in place of the result.
async fn run_suite(
	titler: &Backend<'_>,
	titler_name: &str,
	worker: &Backend<'_>,
	worker_name: &str,
) {
	println!("== session titles ({titler_name}) ==");
	let first_messages = [
		"the login button is broken on mobile somehow, can you fix?",
		"can you add retry with exponential backoff to the S3 uploader and cap it at 5 attempts",
		"hey",
	];
	for message in first_messages {
		let started = Instant::now();
		let outcome = match generate_title(titler, message).await {
			Ok(Some(title)) => title,
			Ok(None) => "(no title: low-signal input)".to_string(),
			Err(error) => format!("error: {error}"),
		};
		println!("  {:>7.1?}  {:<70} -> {outcome}", started.elapsed(), preview(message, 70));
	}

	println!("\n== auto thinking router ({worker_name}) ==");
	let prompts = [
		"rename the variable `usr` to `user` in main.rs",
		"add pagination to the /orders endpoint, cursor based",
		"our async scheduler occasionally deadlocks under load and I can't reproduce it locally — \
		 find the root cause",
	];
	for prompt in prompts {
		let started = Instant::now();
		let outcome = match classify_difficulty(worker, prompt).await {
			Ok(Some(Effort::Low)) => "trivial  -> effort: low".to_string(),
			Ok(Some(Effort::High)) => "moderate -> effort: high".to_string(),
			Ok(Some(Effort::XHigh)) => "hard     -> effort: xhigh".to_string(),
			Ok(None) => "unparseable -> keep prior level".to_string(),
			Err(error) => format!("error: {error}"),
		};
		println!("  {:>7.1?}  {:<70} -> {outcome}", started.elapsed(), preview(prompt, 70));
	}

	println!("\n== unexpected-stop classifier ({worker_name}) ==");
	let assistant_messages = [
		"Let me run the tests next.",
		"The fix is done and all 42 tests pass.",
		"I should do the same for the JS eval worker. Doing that now.",
	];
	for message in assistant_messages {
		let started = Instant::now();
		let outcome = match classify_unexpected_stop(worker, message).await {
			Ok(Some(true)) => "UNEXPECTED STOP (nudge the agent to continue)".to_string(),
			Ok(Some(false)) => "clean finish".to_string(),
			Ok(None) => "unparseable".to_string(),
			Err(error) => format!("error: {error}"),
		};
		println!("  {:>7.1?}  {:<70} -> {outcome}", started.elapsed(), preview(message, 70));
	}

	println!("\n== memory fact extraction ({worker_name}) ==");
	let memories = [
		"My name is Sam, I work at Globex, staging runs on port 8443, and please always answer in \
		 German.",
		"lol nice weather today, might grab a coffee later",
	];
	for message in memories {
		let started = Instant::now();
		match extract_memory_facts(worker, message).await {
			Ok(facts) => {
				println!("  {:>7.1?}  {}", started.elapsed(), preview(message, 88));
				if facts.is_empty() {
					println!("           -> NO_FACTS");
				} else {
					for fact in &facts {
						println!("           -> {fact}");
					}
				}
			},
			Err(error) => {
				println!("  {:>7.1?}  {}", started.elapsed(), preview(message, 88));
				println!("           -> error: {error}");
			},
		}
	}
}

#[tokio::main]
async fn main() -> DemoResult<()> {
	println!("loading models (first run downloads ~1 GB of GGUF weights)...");
	let started = Instant::now();
	// pi splits the work the same way: sub-1B models title (latency-bound),
	// 1B+ "memory-class" models classify and extract (quality-bound).
	let (titler, worker) = tokio::try_join!(
		TextGenerator::small(SmallModel::Lfm2_350M)
			.context_size(2048)
			.build(),
		TextGenerator::small(SmallModel::Lfm2_1_2B)
			.context_size(4096)
			.build(),
	)?;
	println!(
		"loaded in {:.1?}: titles on LFM2-350M ({:?}), classify/memory on LFM2-1.2B ({:?})\n",
		started.elapsed(),
		titler.accelerator(),
		worker.accelerator(),
	);
	run_suite(
		&Backend::Llama(&titler),
		"LFM2-350M, llama.cpp",
		&Backend::Llama(&worker),
		"LFM2-1.2B, llama.cpp",
	)
	.await;
	tokio::try_join!(titler.shutdown(), worker.shutdown())?;

	println!("\n---\n");
	match AppleFm::load().await {
		Ok(apple) => {
			println!("Apple Foundation Models available; rerunning the suite on the system model\n");
			let backend = Backend::Apple(apple);
			run_suite(&backend, "Apple FM", &backend, "Apple FM").await;
		},
		Err(error) => {
			println!("Apple Foundation Models unavailable, skipping: {error}");
		},
	}
	Ok(())
}

/// Single-line preview truncated to `max` display columns.
fn preview(text: &str, max: usize) -> String {
	let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
	let (kept, _) = xutf::truncate_measured_str(&flat, max);
	if kept.len() == flat.len() {
		flat
	} else {
		format!("{kept}…")
	}
}
