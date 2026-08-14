use std::{
	ffi::CString,
	sync::{Arc, Mutex as StdMutex},
};

use async_trait::async_trait;
use omp_core::Str;
use omp_tools::eval::idle_timeout::TimeoutHandle;
use pyo3::{
	prelude::*,
	types::{PyDict, PyModule},
};
use serde_json::{Value, json};
use tempfile::tempdir;
use tokio::runtime::Runtime;

use super::*;

static PYTHON_TEST: StdMutex<()> = StdMutex::new(());

#[derive(Default)]
struct PreludeHost {
	calls: parking_lot::Mutex<Vec<(String, Value)>>,
}

#[async_trait]
impl BridgeHost for PreludeHost {
	async fn call(&self, name: &str, args: Value) -> Result<Value, BridgeHostError> {
		self.calls.lock().push((name.to_owned(), args.clone()));
		match name {
			"echo" => Ok(args),
			"read" => {
				Ok(Value::String(format!("delegated:{}", args["path"].as_str().unwrap_or_default())))
			},
			"fail" => Err(BridgeHostError::message("host exploded")),
			"updates" => Ok(json!({
				"__omp_bridge_value__": { "done": true },
				"__omp_bridge_updates__": [{ "step": 1 }, { "step": 2 }]
			})),
			"__completion__" if !args["schema"].is_null() => Ok(json!({ "text": "{\"answer\":42}" })),
			"__completion__" => Ok(json!({ "text": "completed" })),
			"__agent__" if !args["schema"].is_null() => Ok(json!({
				"text": "{\"answer\":42}",
				"data": { "answer": 42 },
				"details": { "id": "child-structured", "agent": "task" }
			})),
			"__agent__" => Ok(json!({
				"text": "child output",
				"details": { "id": "child-1", "agent": "task", "isolated": true }
			})),
			"__concurrency__" => Ok(json!({ "limit": 2 })),
			"__budget__" => Ok(json!({ "total": 100, "spent": 35, "hard": true })),
			_ => Err(BridgeHostError::message(format!("unexpected bridge call: {name}"))),
		}
	}
}

fn python() -> Arc<omp_py::Engine> {
	super::super::tools::python_engine().expect("initialize embedded Python")
}

fn run(py: Python<'_>, globals: &Bound<'_, PyDict>, source: String) -> PyResult<()> {
	let source = CString::new(source).expect("test source has no NUL");
	py.run(source.as_c_str(), Some(globals), Some(globals))
}

#[test]
fn complete_prelude_persists_and_bridges_host_helpers() {
	let _serial = PYTHON_TEST.lock().expect("serialize embedded Python tests");
	let root = tempdir().expect("temp root");
	let artifacts = root.path().join("artifacts");
	let local = root.path().join("local");
	std::fs::create_dir_all(&artifacts).expect("artifacts directory");
	std::fs::create_dir_all(&local).expect("local directory");
	std::fs::write(artifacts.join("alpha.md"), "one\ntwo\nthree\n").expect("raw output fixture");
	std::fs::write(artifacts.join("data.md"), r#"{"endpoints":[{"file":"src/a.rs"}]}"#)
		.expect("json output fixture");
	std::fs::write(artifacts.join("ansi.md"), "\u{1b}[31mred\u{1b}[0m").expect("ansi fixture");

	let runtime = Runtime::new().expect("test runtime");
	let dispatcher = BridgeDispatcher::new();
	let host = Arc::new(PreludeHost::default());
	let registration = dispatcher
		.register(
			Str::new_static("session"),
			Str::new_static("run"),
			BridgeCapabilities::new([
				Str::new_static("echo"),
				Str::new_static("read"),
				Str::new_static("updates"),
				Str::new_static("fail"),
			])
			.with_completion()
			.with_agent()
			.with_concurrency()
			.with_budget(),
			host.clone(),
			TimeoutHandle::new(None),
		)
		.expect("bridge registration");

	python().attach(|py| -> PyResult<()> {
		let globals = PyDict::new(py);
		globals.set_item("__builtins__", PyModule::import(py, "builtins")?)?;
		run(py, &globals, r#"__omp_events = []
__omp_timeout_events = []
def __omp_display(value, raw=False):
    __omp_events.append((value, raw))
def __omp_timeout_pause__():
    __omp_timeout_events.append("pause")
def __omp_timeout_resume__():
    __omp_timeout_events.append("resume")
"#.to_owned())?;
		install_python_bridge(py, &globals, registration.client(), runtime.handle().clone())?;
		install_python_prelude(py, &globals)?;
		let setup = format!(
			"OMP_ARTIFACTS_DIR = {artifacts}\nOMP_EVAL_LOCAL_ROOTS = json.dumps({{'local': {local}}})\n",
			artifacts = serde_json::to_string(&artifacts.to_string_lossy()).unwrap(),
			local = serde_json::to_string(&local.to_string_lossy()).unwrap(),
		);
		run(py, &globals, setup)?;
		run(py, &globals, r#"
import contextlib, io

# display + ordinary print output
display({"answer": 42})
assert __omp_events[-1] == (({"application/json": {"answer": 42}, "text/plain": "{'answer': 42}"}), True)
_printed = io.StringIO()
with contextlib.redirect_stdout(_printed):
    print("hello", "eval")
assert _printed.getvalue() == "hello eval\n"

# local filesystem helpers and environment persistence
assert env("OMP_PRELUDE_TEST", "present") == "present"
assert env("OMP_PRELUDE_TEST") == "present"
assert env()["OMP_PRELUDE_TEST"] == "present"
assert str(write("local://nested/value.txt", "first\nsecond\nthird\n")).endswith("nested/value.txt")
assert read("local://nested/value.txt", offset=2, limit=1) == "second\n"
assert read("skill://demo", offset=3, limit=2) == "delegated:skill://demo:3-4"

# output lookup: raw/json/stripped/query/ranges/multiple
assert output("alpha") == "one\ntwo\nthree\n"
_alpha = output("alpha", format="json", offset=2, limit=1)
assert _alpha["content"] == "two" and _alpha["range"] == {"start_line": 2, "end_line": 2, "total_lines": 3}
assert output("ansi", format="stripped") == "red"
assert output("data", query=".endpoints[0].file") == '"src/a.rs"'
assert output("alpha", "data")[0] == {"id": "alpha", "content": "one\ntwo\nthree\n"}
try:
    output("alpha", query=".x", offset=1)
except ValueError as error:
    assert str(error) == "query cannot be combined with offset/limit"
else:
    raise AssertionError("invalid output arguments were accepted")


# authenticated host helpers
assert tool.echo({"value": 7}) == {"value": 7, "i": "py prelude"}
assert tool["echo"](value=8) == {"value": 8, "i": "py prelude"}
assert repr(tool) == "<tool proxy session=session>"
assert tool.updates({}) == {"done": True}
_tool_statuses = [
    value["application/x-omp-status"]
    for value, raw in __omp_events
    if raw and isinstance(value, dict)
    and value.get("application/x-omp-status", {}).get("op") == "tool"
]
assert _tool_statuses[-2:] == [
    {"op": "tool", "name": "updates", "update": {"step": 1}},
    {"op": "tool", "name": "updates", "update": {"step": 2}},
]
assert completion("prompt", model="smol") == "completed"
assert completion("prompt", schema={"type": "object"}) == {"answer": 42}
_child = agent("do work", handle=True, isolated=True)
assert _child == {
    "text": "child output", "output": "child output", "handle": "agent://child-1",
    "id": "child-1", "agent": "task", "isolated": True,
}
assert agent("structured", schema={"type": "object"}) == {"answer": 42}
assert parallel([lambda: 1, lambda: 2]) == [1, 2]
assert pipeline([1, 2], lambda n: n + 1, lambda n: n * 2) == [4, 6]
log("working")
phase("checking")
assert __omp_current_phase__ == "checking"
assert budget.total == 100 and budget.hard is True
assert budget.spent() == 35 and budget.remaining() == 65
assert repr(budget) == "<budget total=100 spent=35>"
assert __omp_timeout_events.count("pause") == __omp_timeout_events.count("resume")
assert __omp_timeout_events.count("pause") > 0

# Namespace and one-time prelude guard persist across cells.
persisted_value = 73
"#.to_owned())?;
		install_python_prelude(py, &globals)?;
		run(py, &globals, "assert persisted_value == 73\nassert tool.echo({'again': True})['again'] is True\n".to_owned())?;
		Ok(())
	}).expect("exercise complete Python helper prelude");

	let calls = host.calls.lock();
	assert!(
		calls
			.iter()
			.any(|(name, args)| name == "echo" && args["i"] == "py prelude")
	);
	assert!(calls.iter().any(|(name, _)| name == "__agent__"));
	assert!(calls.iter().any(|(name, _)| name == "__completion__"));
	drop(calls);
}

#[test]
fn python_bridge_propagates_host_errors_and_capability_denial() {
	let _serial = PYTHON_TEST.lock().expect("serialize embedded Python tests");
	let runtime = Runtime::new().expect("test runtime");
	let dispatcher = BridgeDispatcher::new();
	let registration = dispatcher
		.register(
			Str::new_static("session-errors"),
			Str::new_static("run-errors"),
			BridgeCapabilities::new([Str::new_static("fail")]),
			Arc::new(PreludeHost::default()),
			TimeoutHandle::new(None),
		)
		.expect("bridge registration");

	python()
		.attach(|py| -> PyResult<()> {
			let globals = PyDict::new(py);
			globals.set_item("__builtins__", PyModule::import(py, "builtins")?)?;
			run(
				py,
				&globals,
				r#"__omp_timeout_events = []
def __omp_display(value, raw=False):
    pass
def __omp_timeout_pause__():
    __omp_timeout_events.append("pause")
def __omp_timeout_resume__():
    __omp_timeout_events.append("resume")
"#
				.to_owned(),
			)?;
			install_python_bridge(py, &globals, registration.client(), runtime.handle().clone())?;
			install_python_prelude(py, &globals)?;
			run(
				py,
				&globals,
				r#"
try:
    tool.fail({})
except RuntimeError as error:
    assert str(error) == "host exploded"
else:
    raise AssertionError("host failure did not propagate")

try:
    tool.read({"path": "secret"})
except RuntimeError as error:
    assert str(error) == "bridge capability denied: read"
else:
    raise AssertionError("capability denial did not propagate")
assert __omp_timeout_events == ["pause", "resume", "pause", "resume"]
"#
				.to_owned(),
			)
		})
		.expect("bridge errors surface in Python");
}
