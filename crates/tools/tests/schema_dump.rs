//! Scratch: dumps current model-facing schemas for migration diffing.

use std::collections::BTreeMap;

#[test]
fn dump() {
	let dump = [
		("read", omp_tool::schema::<omp_tools::read::Params>()),
		("write", omp_tool::schema::<omp_tools::write::Params>()),
		("edit", omp_tool::schema::<omp_tools::edit::Params>()),
		("grep", omp_tool::schema::<omp_tools::grep::Params>()),
		("glob", omp_tool::schema::<omp_tools::glob::Params>()),
		("shell", omp_tool::schema::<omp_tools::shell::Params>()),
		("eval", omp_tool::schema::<omp_tools::eval::Params>()),
		("py-fallback", omp_tool::schema::<BTreeMap<String, serde_json::Value>>()),
	];
	for (name, schema) in dump {
		println!("=== {name} ===");
		println!("{}", std::str::from_utf8(&schema).unwrap());
	}
}
