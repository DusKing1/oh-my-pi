use std::{
	sync::{
		Arc,
		atomic::{AtomicBool, AtomicUsize, Ordering},
	},
	task::Poll,
};

use bytes::Bytes;
use futures::{Future, Stream, StreamExt, future, pin_mut};
use omp_core::Str;
use omp_tool::{
	Abort, BlobRef, Ev, IncomingParams, Interrupt, Outcome, Part, PromptCaps, Registry, Tool,
	ToolIdentity, Verdict,
};
use omp_tools::read::{
	self, Content, DocumentKind, Fault, LeaseContent, LineRange, ReadBlobs, ReadDocuments,
	ReadLease, SummarySegment, TextSlice,
};
use parking_lot::Mutex;

#[derive(Clone)]
struct Docs {
	state:   Arc<State>,
	kind:    DocumentKind,
	content: LeaseContent,
}
struct State {
	opened:    AtomicBool,
	reads:     AtomicUsize,
	lease_ids: Mutex<Vec<usize>>,
	cancelled: AtomicBool,
	block:     AtomicBool,
}
#[derive(Clone)]
struct Lease {
	state:   Arc<State>,
	id:      usize,
	kind:    DocumentKind,
	content: LeaseContent,
}

impl ReadDocuments for Docs {
	type Lease = Lease;

	fn open(&self, _path: Str) -> impl Future<Output = Result<Lease, Fault>> + Send + '_ {
		let lease = Lease {
			state:   self.state.clone(),
			id:      7,
			kind:    self.kind.clone(),
			content: self.content.clone(),
		};
		async move {
			lease.state.opened.store(true, Ordering::SeqCst);
			Ok(lease)
		}
	}
}

struct CancelGuard(Arc<State>);
impl Drop for CancelGuard {
	fn drop(&mut self) {
		self.0.cancelled.store(true, Ordering::SeqCst);
	}
}
impl ReadLease for Lease {
	fn revision(&self) -> &Str {
		static REV: std::sync::LazyLock<Str> = std::sync::LazyLock::new(|| Str::new_static("A1B2"));
		&REV
	}

	fn kind(&self) -> &DocumentKind {
		&self.kind
	}

	fn read<'a>(
		&'a self,
		_ranges: &'a [LineRange],
		_structural: bool,
	) -> impl Future<Output = Result<LeaseContent, Fault>> + Send + 'a {
		async move {
			self.state.reads.fetch_add(1, Ordering::SeqCst);
			self.state.lease_ids.lock().push(self.id);
			let _guard = CancelGuard(self.state.clone());
			if self.state.block.load(Ordering::SeqCst) {
				future::pending::<()>().await;
			}
			Ok(self.content.clone())
		}
	}
}
#[derive(Clone, Default)]
struct Blobs;
impl ReadBlobs for Blobs {
	fn store(
		&self,
		bytes: Bytes,
		media_type: Str,
	) -> impl Future<Output = Result<BlobRef, Fault>> + Send + '_ {
		future::ready(Ok(BlobRef {
			hash: Str::new_static("blob-hash"),
			media_type,
			byte_len: bytes.len() as u64,
		}))
	}
}

fn fixture(kind: DocumentKind, content: LeaseContent) -> (Docs, Arc<State>) {
	let state = Arc::new(State {
		opened:    AtomicBool::new(false),
		reads:     AtomicUsize::new(0),
		lease_ids: Mutex::new(Vec::new()),
		cancelled: AtomicBool::new(false),
		block:     AtomicBool::new(false),
	});
	(Docs { state: state.clone(), kind, content }, state)
}

async fn invoke(docs: Docs, raw: &str) -> Vec<Ev<read::Update, read::Payload, Fault>> {
	let tool = read::tool(docs, Blobs);
	let (feed, params) = IncomingParams::channel();
	feed.args_committed(Str::from(raw)).unwrap();
	tool.call(params).collect().await
}

#[tokio::test]
async fn opens_before_commit_and_same_pinned_lease_reads_ranges() {
	let (docs, state) = fixture(DocumentKind::Text, LeaseContent::Text {
		slices: vec![TextSlice { start_line: 2, text: Str::new_static("two\nthree") }],
		elided: vec![LineRange { start: 1, end: 1 }],
	});
	let tool = read::tool(docs, Blobs);
	let (feed, params) = IncomingParams::channel();
	feed
		.arg_text(Str::new_static("{\"path\":\"a.rs\""))
		.unwrap();
	let mut events = Box::pin(tool.call(params));
	future::poll_fn(|cx| {
		assert!(events.as_mut().poll_next(cx).is_pending());
		if state.opened.load(Ordering::SeqCst) { Poll::Ready(()) } else { Poll::Pending }
	})
	.await;
	assert_eq!(state.reads.load(Ordering::SeqCst), 0);
	feed
		.arg_text(Str::new_static(",\"ranges\":[{\"start\":2,\"end\":3}]}"))
		.unwrap();
	feed
		.args_committed(Str::new_static("{\"path\":\"a.rs\",\"ranges\":[{\"start\":2,\"end\":3}]}"))
		.unwrap();
	let events: Vec<_> = events.collect().await;
	assert_eq!(state.lease_ids.lock().as_slice(), &[7]);
	let Ev::Done(Outcome::Done { result: Ok(payload), .. }) = &events[0] else {
		panic!("success")
	};
	assert_eq!(payload.revision, "A1B2");
	assert_eq!(payload.ranges, vec![LineRange { start: 2, end: 3 }]);
}

#[tokio::test]
async fn structural_summary_truth_and_projection() {
	let segments = vec![SummarySegment {
		start_line: 1,
		end_line:   8,
		text:       Str::new_static("fn main()"),
		elided:     true,
	}];
	let (docs, _) = fixture(DocumentKind::Text, LeaseContent::Summary {
		segments: segments.clone(),
		elided:   vec![LineRange { start: 2, end: 7 }],
	});
	let tool = read::tool(docs.clone(), Blobs);
	let events = invoke(docs, r#"{"path":"a.rs","structural":true}"#).await;
	let Ev::Done(Outcome::Done { result: Ok(payload), .. }) = &events[0] else {
		panic!("success")
	};
	assert!(payload.structural);
	assert_eq!(payload.content, Content::Summary { segments });
	let parts = tool.prompt(Ok(payload), &PromptCaps {
		maximum_parts:      1,
		maximum_text_bytes: 100,
		media:              false,
	});
	assert!(matches!(&parts[0], Part::Text { text } if text.contains("[1-8 elided]")));
}

#[tokio::test]
async fn binary_uses_blob_with_deterministic_media_fallback() {
	let kind = DocumentKind::Binary {
		media_type: Str::new_static("image/png"),
		fallback:   Str::new_static("PNG image"),
	};
	let (docs, _) = fixture(kind, LeaseContent::Binary(Bytes::from_static(b"png")));
	let tool = read::tool(docs.clone(), Blobs);
	let events = invoke(docs, r#"{"path":"x.png"}"#).await;
	let Ev::Done(Outcome::Done { result: Ok(payload), .. }) = &events[0] else {
		panic!("success")
	};
	assert!(matches!(
		tool
			.prompt(Ok(payload), &PromptCaps {
				maximum_parts:      1,
				maximum_text_bytes: 20,
				media:              true,
			})
			.as_slice(),
		[Part::Blob { .. }]
	));
	assert!(
		matches!(tool.prompt(Ok(payload), &PromptCaps { maximum_parts: 1, maximum_text_bytes: 20, media: false }).as_slice(), [Part::Text { text }] if text == "PNG image")
	);
}

#[tokio::test]
async fn malformed_pulled_path_is_args_and_drop_before_commit_never_reads() {
	let (docs, state) =
		fixture(DocumentKind::Text, LeaseContent::Text { slices: Vec::new(), elided: Vec::new() });
	let events = invoke(docs.clone(), r#"{"path":4}"#).await;
	assert!(matches!(events.as_slice(), [Ev::Args(_)]));
	let tool = read::tool(docs, Blobs);
	let (feed, params) = IncomingParams::channel();
	feed.arg_text(Str::new_static("{\"path\":\"a\"}")).unwrap();
	drop(feed);
	let events = tool.call(params);
	pin_mut!(events);
	assert!(matches!(events.next().await, Some(Ev::Aborted(Abort::InputDropped))));
	assert_eq!(state.reads.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn postcommit_interrupt_drops_resource_future() {
	let (docs, state) =
		fixture(DocumentKind::Text, LeaseContent::Text { slices: Vec::new(), elided: Vec::new() });
	state.block.store(true, Ordering::SeqCst);
	let tool = read::tool(docs, Blobs);
	let (feed, params) = IncomingParams::channel();
	feed
		.args_committed(Str::new_static("{\"path\":\"a\"}"))
		.unwrap();
	let mut events = Box::pin(tool.call(params));
	let next = events.next();
	tokio::pin!(next);
	tokio::select! { _ = &mut next => panic!("blocked read completed"), _ = async { while state.reads.load(Ordering::SeqCst) == 0 { tokio::task::yield_now().await; } } => {} }
	feed
		.interrupt(Interrupt {
			class:  Str::new_static("immediate"),
			reason: Str::new_static("stop"),
		})
		.unwrap();
	assert!(
		matches!(next.await, Some(Ev::Aborted(Abort::Interrupted { reason })) if reason == "stop")
	);
	assert!(state.cancelled.load(Ordering::SeqCst));
}

#[tokio::test]
async fn registry_exposes_exact_revision_and_typed_verdict_projection() {
	let (docs, _) = fixture(DocumentKind::Text, LeaseContent::Text {
		slices: vec![TextSlice { start_line: 1, text: Str::new_static("x") }],
		elided: Vec::new(),
	});
	let mut registry = Registry::new();
	registry.register(read::tool(docs, Blobs)).unwrap();
	let (name, rev) = registry.live_identity("read").unwrap();
	assert_eq!(rev.to_string(), "1");
	let identity = ToolIdentity { name: name.clone(), rev: rev.clone() };
	let verdict: Verdict<read::Payload, Fault> = Verdict::Ok(read::Payload {
		path:       Str::new_static("a"),
		revision:   Str::new_static("A1B2"),
		ranges:     Vec::new(),
		structural: false,
		elided:     Vec::new(),
		content:    Content::Text {
			slices: vec![TextSlice { start_line: 1, text: Str::new_static("x") }],
		},
	});
	let json = serde_json::to_vec(&verdict).unwrap();
	assert!(matches!(
		registry
			.prompt(&identity, &json, &PromptCaps {
				maximum_parts:      1,
				maximum_text_bytes: 20,
				media:              false,
			})
			.unwrap()
			.unwrap()
			.as_slice(),
		[Part::Text { .. }]
	));
}
