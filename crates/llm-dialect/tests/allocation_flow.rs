//! Allocation-flow regression coverage for dialect streaming.

use std::{
	alloc::{GlobalAlloc, Layout, System},
	cell::Cell,
	hint::black_box,
};

use bytes::Bytes;
use omp_core::SmolStr;
use omp_llm_dialect::{
	Dialect, ScannerOptions,
	projector::{Projection, StreamProjector},
};
use omp_llm_transport::{ndjson::NdjsonDecoder, sse::SseDecoder};
use omp_llm_types::{StreamAccumulator, StreamPartKind, TurnEvent, ids::CallId};

struct CountingAllocator;
// Allocation accounting is deliberately thread-local: Rust's test runner may
// execute unrelated tests concurrently, and their allocations are not part of
// this hot-path measurement.

thread_local! {
	static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
	static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: Every operation delegates to `System` with the exact pointer/layout
// contract supplied by the caller. The only added work updates already-warmed
// thread-local `Cell`s and neither allocates nor touches the allocation itself.
unsafe impl GlobalAlloc for CountingAllocator {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		record_allocation();
		// SAFETY: `layout` is forwarded unchanged from the `GlobalAlloc` caller.
		unsafe { System.alloc(layout) }
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		record_allocation();
		// SAFETY: `layout` is forwarded unchanged from the `GlobalAlloc` caller.
		unsafe { System.alloc_zeroed(layout) }
	}

	unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
		// SAFETY: The pointer/layout pair is forwarded unchanged.
		unsafe { System.dealloc(pointer, layout) }
	}

	unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
		record_allocation();
		// SAFETY: The pointer, old layout, and requested size are forwarded unchanged.
		unsafe { System.realloc(pointer, layout, size) }
	}
}

#[inline]
fn record_allocation() {
	if TRACK_ALLOCATIONS.try_with(Cell::get).unwrap_or(false) {
		let _ =
			ALLOCATIONS.try_with(|allocations| allocations.set(allocations.get().saturating_add(1)));
	}
}

struct AllocationScope {
	active: bool,
}

impl AllocationScope {
	fn begin() -> Self {
		// Initialize both thread-local cells before enabling measurement.
		TRACK_ALLOCATIONS.with(|_| {});
		ALLOCATIONS.with(|allocations| allocations.set(0));
		TRACK_ALLOCATIONS.with(|tracking| {
			assert!(!tracking.replace(true), "allocation measurements must not nest");
		});
		Self { active: true }
	}

	fn finish(mut self) -> usize {
		TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
		self.active = false;
		ALLOCATIONS.with(Cell::get)
	}
}

impl Drop for AllocationScope {
	fn drop(&mut self) {
		if self.active {
			// A panic in the measured closure must not poison a reused test thread.
			let _ = TRACK_ALLOCATIONS.try_with(|tracking| tracking.set(false));
		}
	}
}

/// Counts `alloc`, `alloc_zeroed`, and `realloc` calls made by `run` on this
/// thread. It intentionally excludes setup/warm-up, input construction,
/// deallocation, teardown, and every other test-runner thread.
fn count_allocations(run: impl FnOnce()) -> usize {
	let scope = AllocationScope::begin();
	run();
	scope.finish()
}

fn consume_projection(batch: impl IntoIterator<Item = Projection>) {
	for projection in batch {
		match projection {
			Projection::Event(TurnEvent::PartDelta { chunk, .. }) => {
				black_box(chunk);
			},
			other => {
				black_box(other);
			},
		}
	}
}

fn warmed_projector(dialect: Dialect) -> StreamProjector {
	let mut projector = StreamProjector::new(dialect, ScannerOptions::default());
	consume_projection(projector.feed_text(Bytes::from_static(b"warm delta ")));
	projector
}

#[test]
fn plain_text_delta_steady_state_is_zero_allocations_for_all_11_owned_projectors() {
	for dialect in Dialect::ALL {
		let mut projector = warmed_projector(dialect);

		let allocations = count_allocations(|| {
			consume_projection(projector.feed_text(Bytes::from_static(b"plain delta ")));
		});
		assert_eq!(
			allocations,
			0,
			"{} plain-text owned projection steady state measured {allocations} allocations; \
			 expected 0 because safe text retains its transport Bytes",
			dialect.as_str(),
		);
	}
}

#[test]
fn fragmented_concrete_scanner_projector_hot_corpus_is_exactly_zero_allocations() {
	// Warm every state used below before opening the scope: projector/scanner
	// construction, XML tag parsing, part/map creation, TLS access, and backing
	// buffers. The measured inputs are prebuilt `Bytes`, so only the concrete
	// enum scanner/projector path and destruction of its inline batches count.
	let mut text = warmed_projector(Dialect::Xml);

	let mut thinking_options = ScannerOptions::default();
	thinking_options.parse_thinking = true;
	let mut thinking = StreamProjector::new(Dialect::Xml, thinking_options);
	consume_projection(thinking.feed_text(Bytes::from_static(b"<thinking>")));
	consume_projection(
		thinking.feed_text(Bytes::from_static(b"warm reasoning buffer for steady state")),
	);

	let mut tool_arguments = StreamProjector::new(Dialect::Xml, ScannerOptions::default());
	consume_projection(
		tool_arguments
			.feed_text(Bytes::from_static(b"<invoke name=\"echo\"><parameter name=\"msg\">12345678")),
	);
	// Force the parameter accumulator through its first growth and leave spare
	// steady-state capacity for the fixed four-byte fragmented corpus.
	consume_projection(tool_arguments.feed_text(Bytes::from_static(b"9")));

	let text_fragments = [
		Bytes::from_static(b"alpha "),
		Bytes::from_static(b"beta "),
		Bytes::from_static(b"gamma "),
		Bytes::from_static(b"delta "),
	];
	let thinking_fragments = [
		Bytes::from_static(b"plan "),
		Bytes::from_static(b"check "),
		Bytes::from_static(b"revise "),
		Bytes::from_static(b"answer "),
	];
	let argument_fragments = [
		Bytes::from_static(b"a"),
		Bytes::from_static(b"b"),
		Bytes::from_static(b"c"),
		Bytes::from_static(b"d"),
	];

	// Warm the measurement machinery itself after all corpus construction.
	assert_eq!(count_allocations(|| {}), 0);
	let allocations = count_allocations(|| {
		for fragment in text_fragments {
			consume_projection(text.feed_text(fragment));
		}
		for fragment in thinking_fragments {
			consume_projection(thinking.feed_text(fragment));
		}
		for fragment in argument_fragments {
			consume_projection(tool_arguments.feed_text(fragment));
		}
	});
	assert_eq!(
		allocations, 0,
		"fixed 12-delta XML concrete-enum corpus measured {allocations} allocations; setup, input \
		 construction, closing tags/finish, deallocation, and other threads are excluded. A \
		 snapshot clone or recreated per-feed scratch buffer makes this nonzero",
	);

	// Closing/serialization is deliberately outside the delta-only measurement.
	consume_projection(thinking.feed_text(Bytes::from_static(b"</thinking>")));
	consume_projection(tool_arguments.feed_text(Bytes::from_static(b"</parameter></invoke>")));
}

#[test]
fn sse_provider_normalize_dialect_gateway_whole_stream_is_zero_steady_state_allocations() {
	let mut decoder = SseDecoder::new();
	let mut projector = warmed_projector(Dialect::Hermes);
	let chunks = [
		Bytes::from(Vec::from(&b"data: first delta\n\n"[..])),
		Bytes::from(Vec::from(&b"data: second delta\n\n"[..])),
		Bytes::from(Vec::from(&b"data: third delta\n\n"[..])),
	];

	let allocations = count_allocations(|| {
		for chunk in chunks {
			for event in decoder.push(chunk) {
				consume_projection(projector.feed_text(event.data));
			}
		}
	});
	assert_eq!(
		allocations, 0,
		"SSE provider→normalize→dialect→gateway whole-stream steady state measured {allocations} \
		 allocations; expected 0 with uniquely owned transport Bytes",
	);
}

#[test]
fn ndjson_provider_normalize_dialect_gateway_whole_stream_is_zero_steady_state_allocations() {
	let mut decoder = NdjsonDecoder::new();
	let mut projector = warmed_projector(Dialect::Qwen3);
	let chunks = [
		Bytes::from(Vec::from(&b"first delta\n"[..])),
		Bytes::from(Vec::from(&b"second delta\n"[..])),
		Bytes::from(Vec::from(&b"third delta\n"[..])),
	];

	let allocations = count_allocations(|| {
		for chunk in chunks {
			for record in decoder.push(chunk) {
				consume_projection(projector.feed_text(record));
			}
		}
	});
	assert_eq!(
		allocations, 0,
		"NDJSON provider→normalize→dialect→gateway whole-stream steady state measured {allocations} \
		 allocations; expected 0 with uniquely owned transport Bytes",
	);
}

#[test]
fn native_tool_argument_growth_is_exactly_8_allocations_and_signature_closure_is_zero() {
	let mut tools = StreamAccumulator::new();
	tools
		.push(&TurnEvent::PartStart {
			index:        0,
			kind:         StreamPartKind::ToolCall,
			tool_call_id: SmolStr::from(CallId::new().to_string()),
			tool_name:    SmolStr::new("lookup"),
		})
		.expect("start native tool");
	let fragment = Bytes::from_static(b"12345678");
	let tool_allocations = count_allocations(|| {
		for _ in 0..128 {
			tools
				.push(&TurnEvent::PartDelta { index: 0, chunk: fragment.clone() })
				.expect("grow native tool arguments");
		}
	});
	assert_eq!(
		tool_allocations, 8,
		"native tool argument growth measured {tool_allocations} allocations for the fixed 128 \
		 delta/1024-byte corpus; expected the initial BytesMut allocation plus seven geometric \
		 growth reallocations. An accumulated-string clone or per-delta box changes this count",
	);

	let mut thinking = StreamAccumulator::new();
	thinking
		.push(&TurnEvent::PartStart {
			index:        1,
			kind:         StreamPartKind::Thinking,
			tool_call_id: SmolStr::default(),
			tool_name:    SmolStr::default(),
		})
		.expect("start signed thinking");
	thinking
		.push(&TurnEvent::PartDelta { index: 1, chunk: Bytes::from_static(b"reasoning") })
		.expect("append thinking delta");
	let signature = Bytes::from_static(b"provider-signature");
	let signature_allocations = count_allocations(|| {
		thinking
			.push(&TurnEvent::PartEnd { index: 1, signature: signature.clone() })
			.expect("close signed thinking");
	});
	assert_eq!(
		signature_allocations, 0,
		"signed-thinking closure measured {signature_allocations} allocations; expected 0 because \
		 the opaque signature retains its Bytes backing",
	);
}

#[test]
fn allocation_harness_detects_one_reintroduced_per_delta_box() {
	let allocations = count_allocations(|| {
		for value in 0..32 {
			black_box(Box::new(value));
		}
	});
	assert_eq!(
		allocations, 32,
		"counting-harness control measured {allocations} allocations for 32 per-delta boxes; \
		 expected 32",
	);
}
