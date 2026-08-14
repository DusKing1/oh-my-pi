//! Pi-equivalent `write@1` schema, guards, transactions, and exact output
//! contracts.

use std::{future::Future, sync::Arc};

use futures::{StreamExt, executor::block_on};
use omp_core::Str;
use omp_tool::{Ev, IncomingParams, Outcome, Part, PromptCaps, Tool};
use omp_tools::{
	read::selector::LiteralPathProbe,
	write::{
		self, Fault, PlainWriteRequest, PlainWriteResult, WriteCommitError, WriteDisposition,
		WriteDocuments, WriteOperation,
	},
};
use parking_lot::Mutex;
use serde_json::json;

#[derive(Clone)]
struct FakeDocuments {
	probe:    LiteralPathProbe,
	result:   Result<PlainWriteResult, WriteCommitError>,
	probed:   Arc<Mutex<Vec<Str>>>,
	requests: Arc<Mutex<Vec<PlainWriteRequest>>>,
}

impl FakeDocuments {
	fn success(probe: LiteralPathProbe, result: PlainWriteResult) -> Self {
		Self { probe, result: Ok(result), probed: Arc::default(), requests: Arc::default() }
	}
}

impl WriteDocuments for FakeDocuments {
	fn probe_literal(
		&self,
		path: Str,
	) -> impl Future<Output = Result<LiteralPathProbe, Fault>> + Send + '_ {
		let probe = self.probe;
		let probed = Arc::clone(&self.probed);
		async move {
			probed.lock().push(path);
			Ok(probe)
		}
	}

	fn write_plain(
		&self,
		request: PlainWriteRequest,
	) -> impl Future<Output = Result<PlainWriteResult, WriteCommitError>> + Send + '_ {
		let result = self.result.clone();
		let requests = Arc::clone(&self.requests);
		async move {
			requests.lock().push(request);
			result
		}
	}
}

struct Invocation {
	result:  Result<write::Payload, Fault>,
	useless: bool,
	text:    String,
}

fn committed(
	disposition: WriteDisposition,
	byte_len: u64,
	made_executable: bool,
	snapshot_tag: Option<&'static str>,
) -> PlainWriteResult {
	PlainWriteResult {
		resolved_path: "/workspace/out.txt".into(),
		display_path: "out.txt".into(),
		byte_len,
		disposition,
		made_executable,
		snapshot_tag: snapshot_tag.map(Str::new_static),
	}
}

fn invoke(documents: FakeDocuments, raw: &str) -> Invocation {
	let tool = write::tool(documents);
	let (feed, params) = IncomingParams::channel();
	feed
		.args_committed(Str::from(raw))
		.expect("invocation consumer remains live");
	let events = block_on(tool.call(params).collect::<Vec<_>>());
	let [Ev::Done(Outcome::Done { result, useless })] = events.as_slice() else {
		panic!("expected one terminal write outcome: {events:?}");
	};
	let parts = tool.prompt(result.as_ref(), &PromptCaps {
		maximum_parts:      1,
		maximum_text_bytes: 64 * 1024,
		media:              false,
	});
	let text = parts
		.into_iter()
		.map(|part| match part {
			Part::Text { text } => text.to_string(),
			Part::Json { .. } => panic!("write must project text only"),
			Part::Blob { .. } => panic!("write must never project blobs"),
		})
		.collect();
	Invocation { result: result.clone(), useless: *useless, text }
}

#[test]
fn pi_schema_definition_and_revision_are_exact() {
	let documents = FakeDocuments::success(
		LiteralPathProbe::Missing,
		committed(WriteDisposition::Created, 0, false, Some("0000")),
	);
	let tool = write::tool(documents);
	assert_eq!(tool.spec().name, "write");
	assert_eq!(tool.spec().rev.to_string(), "1");
	assert_eq!(
		serde_json::from_slice::<serde_json::Value>(&tool.spec().schema).expect("write schema JSON"),
		json!({
			"type": "object",
			"additionalProperties": false,
			"required": ["path", "content"],
			"properties": {
				"path": {"type": "string", "description": "file path"},
				"content": {"type": "string", "description": "file content"}
			}
		})
	);
	assert_eq!(
		tool.spec().description.as_str(),
		"Creates or overwrites file at specified path.\n\n<conditions>\n- Creating new files \
		 explicitly required by task\n- Replacing entire file contents when editing would be more \
		 complex\n- Supports `.tar`, `.tar.gz`, `.tgz`, `.zip`, and ZIP-based \
		 `.jar`/`.war`/`.ear`/`.apk` archive entries via `archive.ext:path/inside/archive`\n- \
		 Supports SQLite row operations via `db.sqlite:table` (insert), `db.sqlite:table:key` \
		 (update with JSON content, delete with empty content)\n</conditions>\n\n<critical>\n- You \
		 SHOULD use Edit tool for modifying existing files\n- You NEVER create documentation files \
		 (*.md, README) unless explicitly requested\n- You NEVER use emojis unless \
		 requested\n</critical>"
	);
}

#[test]
fn create_records_exact_request_payload_and_hashline_output() {
	let documents = FakeDocuments::success(
		LiteralPathProbe::Missing,
		committed(WriteDisposition::Created, 6, false, Some("A1B2")),
	);
	let requests = Arc::clone(&documents.requests);
	let invocation = invoke(documents, r#"{"path":"out.txt","content":"hello\n"}"#);
	assert_eq!(invocation.text, "[out.txt#A1B2]\nSuccessfully wrote 6 bytes to out.txt");
	assert!(!invocation.useless);
	let payload = invocation.result.expect("write succeeds");
	assert_eq!(payload.resolved_path, "/workspace/out.txt");
	assert_eq!(payload.display_path, "out.txt");
	assert_eq!(payload.byte_len, 6);
	assert_eq!(payload.reported_len, 6);
	assert_eq!(payload.disposition, WriteDisposition::Created);
	assert_eq!(payload.operation, WriteOperation::Plain);
	assert_eq!(payload.snapshot_tag.as_deref(), Some("A1B2"));
	assert!(!payload.stripped_wrapper);
	assert!(!payload.made_executable);
	assert_eq!(requests.lock().as_slice(), [PlainWriteRequest {
		path:    "out.txt".into(),
		content: "hello\n".into(),
	}]);
}

#[test]
fn overwrite_has_the_same_pi_success_line_and_retains_disposition_truth() {
	let documents = FakeDocuments::success(
		LiteralPathProbe::Exists,
		committed(WriteDisposition::Overwrote, 3, false, None),
	);
	let invocation = invoke(documents, r#"{"path":"out.txt","content":"new"}"#);
	assert_eq!(invocation.text, "Successfully wrote 3 bytes to out.txt");
	assert_eq!(
		invocation.result.expect("overwrite succeeds").disposition,
		WriteDisposition::Overwrote
	);
}

#[test]
fn success_count_matches_javascript_utf16_length_while_payload_keeps_utf8_bytes() {
	let documents = FakeDocuments::success(
		LiteralPathProbe::Missing,
		committed(WriteDisposition::Created, 6, false, None),
	);
	let invocation = invoke(documents, r#"{"path":"out.txt","content":"é😀"}"#);
	assert_eq!(invocation.text, "Successfully wrote 3 bytes to out.txt");
	let payload = invocation.result.expect("Unicode write succeeds");
	assert_eq!(payload.byte_len, 6);
	assert_eq!(payload.reported_len, 3);
}

#[test]
fn copied_hashline_display_is_stripped_before_commit_with_exact_notice() {
	let documents = FakeDocuments::success(
		LiteralPathProbe::Missing,
		committed(WriteDisposition::Created, 13, false, Some("BEEF")),
	);
	let requests = Arc::clone(&documents.requests);
	let invocation = invoke(
		documents,
		r#"{"path":"[out.txt#1234]","content":"[source.txt#ABCD]\n1:first\n2:second\n"}"#,
	);
	assert_eq!(
		invocation.text,
		"[out.txt#BEEF]\nSuccessfully wrote 13 bytes to out.txt\nNote: auto-stripped hashline \
		 display prefixes from content before writing."
	);
	assert!(
		invocation
			.result
			.expect("stripped write succeeds")
			.stripped_wrapper
	);
	assert_eq!(requests.lock().as_slice(), [PlainWriteRequest {
		path:    "out.txt".into(),
		content: "first\nsecond\n".into(),
	}]);
}

#[test]
fn shebang_chmod_truth_appends_the_exact_notice() {
	let documents = FakeDocuments::success(
		LiteralPathProbe::Missing,
		committed(WriteDisposition::Created, 18, true, Some("C0DE")),
	);
	let invocation = invoke(documents, r##"{"path":"out.txt","content":"#!/bin/sh\necho hi\n"}"##);
	assert_eq!(
		invocation.text,
		"[out.txt#C0DE]\nSuccessfully wrote 18 bytes to out.txt\n[Notice: Made executable via chmod \
		 +x]"
	);
	assert!(
		invocation
			.result
			.expect("shebang write succeeds")
			.made_executable
	);
}

#[test]
fn missing_empty_selector_target_fails_closed_with_exact_read_guidance() {
	let documents = FakeDocuments::success(
		LiteralPathProbe::Missing,
		committed(WriteDisposition::Created, 0, false, None),
	);
	let requests = Arc::clone(&documents.requests);
	let target = "src/LoraSelector.tsx:1-260:raw";
	let invocation = invoke(documents, r#"{"path":"src/LoraSelector.tsx:1-260:raw","content":""}"#);
	assert_eq!(
		invocation.text,
		format!(
			"write target '{target}' ends with a read-tool selector ':1-260:raw' and no such file \
			 exists — refusing to create a literal file by that name. If you meant to read it, use \
			 read({{ path: \"{target}\" }}). If you truly intend to create this file, pass its \
			 contents in `content` (a non-empty write is never blocked)."
		)
	);
	assert!(invocation.result.is_err());
	assert!(requests.lock().is_empty());
}

#[test]
fn selector_list_fails_closed_even_with_nonempty_content() {
	let documents = FakeDocuments::success(
		LiteralPathProbe::Missing,
		committed(WriteDisposition::Created, 0, false, None),
	);
	let requests = Arc::clone(&documents.requests);
	let target = "a.txt:1-2;b/c.txt:3-4";
	let invocation = invoke(documents, r#"{"path":"a.txt:1-2;b/c.txt:3-4","content":"{}"}"#);
	assert_eq!(
		invocation.text,
		format!(
			"write target '{target}' is a semicolon-joined list of 2 read-tool selectors, not a \
			 filesystem path — refusing to create it. write creates a single file; issue one read() \
			 per path to read these ranges (e.g. read({{ path: \"<one path>:<range>\" }}))."
		)
	);
	assert!(requests.lock().is_empty());
}

#[test]
fn existing_ambiguous_and_nonempty_selector_shaped_names_remain_writable() {
	for (probe, content) in [
		(LiteralPathProbe::Exists, ""),
		(LiteralPathProbe::Unknown, ""),
		(LiteralPathProbe::Missing, "intentional"),
	] {
		let result = committed(
			if probe == LiteralPathProbe::Exists {
				WriteDisposition::Overwrote
			} else {
				WriteDisposition::Created
			},
			content.len() as u64,
			false,
			None,
		);
		let documents = FakeDocuments::success(probe, result);
		let requests = Arc::clone(&documents.requests);
		let raw = serde_json::to_string(&json!({"path":"log:1-5", "content":content})).unwrap();
		let invocation = invoke(documents, &raw);
		assert!(invocation.result.is_ok());
		assert_eq!(requests.lock().len(), 1);
	}
}

#[test]
fn unsupported_uri_is_rejected_before_any_document_probe() {
	let documents = FakeDocuments::success(
		LiteralPathProbe::Missing,
		committed(WriteDisposition::Created, 0, false, None),
	);
	let probed = Arc::clone(&documents.probed);
	let requests = Arc::clone(&documents.requests);
	let invocation = invoke(documents, r#"{"path":"skill://private","content":"secret"}"#);
	assert_eq!(invocation.text, "skill:// targets are not supported yet");
	assert!(invocation.result.is_err());
	assert!(probed.lock().is_empty());
	assert!(requests.lock().is_empty());
}
