//! Steady-state provider codec allocation regression coverage.
//!
//! This integration-test binary intentionally contains one test: allocation
//! counting is process-global, so libtest must not run another measured test
//! concurrently.

use std::{
	alloc::{GlobalAlloc, Layout, System},
	hint::black_box,
	sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use omp_llm_anthropic::AnthropicCodec;
use omp_llm_google::{GoogleCodec, cca::CcaCodec};
use omp_llm_openai::{OpenAiChatCodec, OpenAiResponsesCodec};
use omp_llm_transport::{DecodeState, Frame, Transport};
use omp_llm_types::{StreamPartKind, TurnEvent};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: This wrapper preserves `System`'s allocation contract exactly and
// adds only allocation-free atomic bookkeeping before delegating each
// operation.
unsafe impl GlobalAlloc for CountingAllocator {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		record_allocation();
		// SAFETY: `GlobalAlloc::alloc` gives us a valid requested layout, which is
		// forwarded unchanged to the system allocator.
		unsafe { System.alloc(layout) }
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		record_allocation();
		// SAFETY: `GlobalAlloc::alloc_zeroed` gives us a valid requested layout,
		// which is forwarded unchanged to the system allocator.
		unsafe { System.alloc_zeroed(layout) }
	}

	unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
		// SAFETY: The caller guarantees that `pointer` and `layout` identify a live
		// allocation from this allocator. Every allocation above delegates to System.
		unsafe { System.dealloc(pointer, layout) }
	}

	unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
		record_allocation();
		// SAFETY: The caller guarantees the live allocation, its original layout,
		// and a valid new size; all three are forwarded unchanged to System.
		unsafe { System.realloc(pointer, layout, new_size) }
	}
}

#[inline]
fn record_allocation() {
	if COUNTING.load(Ordering::Relaxed) {
		ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
	}
}

struct Measurement;

impl Measurement {
	fn begin() -> Self {
		assert!(
			COUNTING
				.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
				.is_ok(),
			"allocation measurements must not overlap",
		);
		ALLOCATIONS.store(0, Ordering::Relaxed);
		Self
	}
}

impl Drop for Measurement {
	fn drop(&mut self) {
		COUNTING.store(false, Ordering::SeqCst);
	}
}

fn count_allocations(run: impl FnOnce()) -> usize {
	let measurement = Measurement::begin();
	run();
	drop(measurement);
	ALLOCATIONS.load(Ordering::Relaxed)
}

const BATCH: usize = 128;
const DELTA: &[u8] = br#"{"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}"#;

#[derive(Clone, Copy)]
struct Budget {
	name:          &'static str,
	maximum:       usize,
	setup:         &'static [&'static [u8]],
	delta:         &'static [u8],
	expected_text: &'static [u8],
}

const ANTHROPIC_SETUP: &[&[u8]] =
	&[br#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#];
const RESPONSES_SETUP: &[&[u8]] = &[br#"{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}"#];
// CALIBRATE_AFTER_FIRST_RUN: the parent verification pass records each observed
// count from this test, then replaces these deliberately loose temporary
// ceilings with per-codec baselines and budgets below one additional allocation
// per frame.
const TEMPORARY_MAXIMUM: usize = BATCH * 256;

const ANTHROPIC: Budget = Budget {
	name:          "anthropic_messages",
	maximum:       TEMPORARY_MAXIMUM,
	setup:         ANTHROPIC_SETUP,
	delta:
		br#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}"#,
	expected_text: b"x",
};
const OPENAI_CHAT: Budget = Budget {
	name:          "openai_chat_completions",
	maximum:       TEMPORARY_MAXIMUM,
	setup:         &[],
	delta:         DELTA,
	expected_text: b"x",
};
const OPENAI_RESPONSES: Budget = Budget {
	name: "openai_responses",
	maximum: TEMPORARY_MAXIMUM,
	setup: RESPONSES_SETUP,
	delta: br#"{"type":"response.output_text.delta","output_index":0,"item_id":"msg_1","content_index":0,"delta":"x"}"#,
	expected_text: b"x",
};
const GOOGLE_GENAI: Budget = Budget {
	name:          "google_genai",
	maximum:       TEMPORARY_MAXIMUM,
	setup:         &[],
	delta:         br#"{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]}}]}"#,
	expected_text: b"x",
};
const GOOGLE_CCA: Budget = Budget {
	name:          "google_cca",
	maximum:       TEMPORARY_MAXIMUM,
	setup:         &[],
	delta:
		br#"{"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]}}]}}"#,
	expected_text: b"x",
};

fn prime(codec: &impl Transport, state: &mut DecodeState, budget: Budget) -> u32 {
	let mut text_index = None;
	for frame in budget.setup {
		let events = codec
			.decode(Frame::Data(frame), state)
			.unwrap_or_else(|error| panic!("{} setup decode failed: {error}", budget.name));
		observe_text_start(&events, &mut text_index, budget.name);
		black_box(events);
	}

	let events = codec
		.decode(Frame::Data(budget.delta), state)
		.unwrap_or_else(|error| panic!("{} priming delta decode failed: {error}", budget.name));
	observe_text_start(&events, &mut text_index, budget.name);
	let index = text_index.unwrap_or_else(|| panic!("{} did not announce a text part", budget.name));
	assert_contains_delta(&events, index, budget);
	black_box(events);

	// A second unmeasured delta moves every codec past lazy state and PartStart
	// work.
	let events = codec
		.decode(Frame::Data(budget.delta), state)
		.unwrap_or_else(|error| panic!("{} warm delta decode failed: {error}", budget.name));
	assert_delta(&events, index, budget);
	black_box(events);
	index
}

fn observe_text_start(events: &[TurnEvent], text_index: &mut Option<u32>, name: &str) {
	for event in events {
		if let TurnEvent::PartStart { index, kind, .. } = event {
			assert_eq!(*kind, StreamPartKind::Text, "{name} primed a non-text stream part");
			match text_index {
				Some(existing) => assert_eq!(*existing, *index, "{name} changed its text part index"),
				None => *text_index = Some(*index),
			}
		}
	}
}

fn assert_contains_delta(events: &[TurnEvent], text_index: u32, budget: Budget) {
	let mut deltas = events.iter().filter(|event| {
		matches!(
			event,
			TurnEvent::PartDelta { index, chunk }
				if *index == text_index && chunk.as_ref() == budget.expected_text
		)
	});
	assert!(
		deltas.next().is_some() && deltas.next().is_none(),
		"{} priming frame must emit exactly one text PartDelta",
		budget.name,
	);
}

fn assert_delta(events: &[TurnEvent], text_index: u32, budget: Budget) {
	assert_eq!(events.len(), 1, "{} must emit exactly one event per hot text delta", budget.name);
	match &events[0] {
		TurnEvent::PartDelta { index, chunk } => {
			assert_eq!(*index, text_index, "{} emitted a delta for the wrong part", budget.name);
			assert_eq!(chunk.as_ref(), budget.expected_text, "{} changed its text delta", budget.name);
		},
		other => panic!("{} emitted {other:?}, expected a text PartDelta", budget.name),
	}
}

fn measure(codec: impl Transport, budget: Budget) {
	let mut state = DecodeState::default();
	let text_index = prime(&codec, &mut state, budget);
	let allocations = count_allocations(|| {
		for _ in 0..BATCH {
			let events = codec
				.decode(Frame::Data(budget.delta), &mut state)
				.unwrap_or_else(|error| panic!("{} hot delta decode failed: {error}", budget.name));
			assert_delta(&events, text_index, budget);
			black_box(events);
		}
	});
	eprintln!(
		"{} observed {allocations} allocations for {BATCH} steady-state text deltas",
		budget.name,
	);
	assert!(
		allocations <= budget.maximum,
		"{} observed {allocations} allocations for {BATCH} steady-state text deltas; temporary \
		 calibration ceiling is {}",
		budget.name,
		budget.maximum,
	);
}

#[test]
fn provider_text_delta_allocations_remain_bounded() {
	measure(AnthropicCodec::new(), ANTHROPIC);
	measure(OpenAiChatCodec, OPENAI_CHAT);
	measure(OpenAiResponsesCodec::new(), OPENAI_RESPONSES);
	measure(GoogleCodec::gen_ai(), GOOGLE_GENAI);
	measure(CcaCodec::new("allocation-test-project".into()), GOOGLE_CCA);
}
