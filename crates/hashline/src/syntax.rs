//! Tree-sitter syntax probes used as a veto for heuristic boundary repair.

use std::{
	collections::{HashMap, VecDeque},
	sync::LazyLock,
};

use omp_ast::block::{EnclosingBoundaryOptions, LineRange, enclosing_block_boundaries};
use omp_core::Str;
use parking_lot::Mutex;

const CACHE_LIMIT: usize = 256;

#[derive(Default)]
struct ParseCache {
	values: HashMap<([u8; 32], Str), bool>,
	order:  VecDeque<([u8; 32], Str)>,
}

static CACHE: LazyLock<Mutex<ParseCache>> = LazyLock::new(|| Mutex::new(ParseCache::default()));

/// Returns true only when the path identifies a supported language and the text
/// parses cleanly.
pub fn parses_cleanly(path: Option<&str>, text: &str) -> bool {
	let Some(path) = path else { return false };
	let key = (*blake3::hash(text.as_bytes()).as_bytes(), Str::new(path));
	if let Some(value) = CACHE.lock().values.get(&key) {
		return *value;
	}
	let line_count = text.split('\n').count().max(1);
	let end_line = u32::try_from(line_count).unwrap_or(u32::MAX);
	let ok = enclosing_block_boundaries(EnclosingBoundaryOptions {
		code:   text.to_owned(),
		lang:   None,
		path:   Some(path.to_owned()),
		ranges: vec![LineRange { start_line: 1, end_line }],
	})
	.ok()
	.flatten()
	.is_some();
	{
		let mut guard = CACHE.lock();
		if guard.values.len() >= CACHE_LIMIT
			&& let Some(oldest) = guard.order.pop_front()
		{
			guard.values.remove(&oldest);
		}
		guard.order.push_back(key.clone());
		guard.values.insert(key, ok);
	}
	ok
}
