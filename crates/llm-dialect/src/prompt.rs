//! Static tool-call guides and allocation-disciplined prompt assembly.

use std::fmt;

use crate::{Dialect, inventory::write_json_line_catalog, types::InbandTool};

const PREFIX: &str = "# Tools\n\nYou may call one or more functions to assist with the user \
                      query.\nTool calls are emitted as text using the exact syntax below, not as \
                      native provider tool messages.\n\nAvailable functions are listed inside \
                      `<tools></tools>` as one JSON object per line:\n\n<tools>\n";
const SUFFIX: &str = "</tools>\n\n";

/// Returns the exact static format guide for an owned model dialect.
#[must_use]
pub const fn dialect_guide(dialect: Dialect) -> &'static str {
	match dialect {
		Dialect::Anthropic => ANTHROPIC,
		Dialect::DeepSeek => DEEPSEEK,
		Dialect::Gemini => GEMINI,
		Dialect::Gemma => GEMMA,
		Dialect::Glm => GLM,
		Dialect::Harmony => HARMONY,
		Dialect::Hermes => HERMES,
		Dialect::Kimi => KIMI,
		Dialect::MiniMax => MINIMAX,
		Dialect::Qwen3 => QWEN3,
		Dialect::Xml => XML,
	}
}

/// Writes the JSON-line tool catalog followed by the selected static guide.
///
/// The destination owns all storage; tool definitions and schemas remain
/// borrowed for the duration of rendering.
pub fn write_inband_tool_prompt<W: fmt::Write + ?Sized>(
	out: &mut W,
	tools: &[InbandTool<'_>],
	dialect: Dialect,
) -> fmt::Result {
	out.write_str(PREFIX)?;
	write_json_line_catalog(out, tools)?;
	out.write_str(SUFFIX)?;
	out.write_str(dialect_guide(dialect).trim())
}

const ANTHROPIC: &str = r#"## Format guide

A call is a `<function_calls>` block wrapping one or more `<invoke>` blocks, each holding `<parameter>` children:

```text
<function_calls>
<invoke name="tool_name"><parameter name="arg_name">arg value</parameter></invoke>
</function_calls>
```

Results arrive later in a `<function_results>` block, one `<result>` per call (failures use `<error>` with `<stderr>` in place of `<result>` with `<stdout>`):

```text
<function_results>
<result>
<tool_name>tool_name</tool_name>
<stdout>verbatim tool result</stdout>
</result>
</function_results>
```

## Rules

- `name` MUST match a listed function.
- String/scalar parameters: exact text, spaces preserved — bodies are read by regex (delimiter matching), NOT a real XML parser, so never HTML-escape them (emit `a & b`, not `a &amp; b`; `<`/`>` stay literal); only the body's own `</parameter>` closing tag is reserved. Lists/objects: JSON.
- Multiple calls: multiple `<invoke>` blocks in one `<function_calls>`.
- You MAY write visible text before the calls.
- NEVER emit `tool_calls` JSON.
- NEVER use the legacy `<tool_name>`/`<parameters>` call syntax.
- Read each `<result>`/`<error>` in call order. NEVER emit `<function_results>` yourself.
- Emit the stop sequence ONLY after the call is fully written — NEVER announce a tool then stop (e.g. halting at "Let's run `cargo clippy`" with no `<invoke>` emitted). Write the complete call, THEN the stop sequence, THEN halt.
"#;

const DEEPSEEK: &str = r#"## Format guide

A tool call wraps the function name, a separator, and one JSON object of arguments in fixed tokens. Emit them exactly:

```text
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_name<｜tool▁sep｜>{"arg":"value"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
```

Results arrive as output tokens:

```text
<｜tool▁output▁begin｜>verbatim tool result<｜tool▁output▁end｜>
```

## Rules

- Use `｜` (U+FF5C) and `▁` (U+2581) exactly.
- Tool name MUST match an available function; arguments are one valid JSON object.
- Argument string values use only normal JSON string escaping (`\"`, `\\`, `\n`); never HTML-escape their contents — write `a & b`, not `a &amp; b`.
- NEVER wrap arguments in Markdown fences; NEVER emit a `type` field or `function` prefix.
- Multiple calls chain `<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>` directly — no separators, spaces, or newlines between them.
- Private reasoning, when needed, goes in `<think>...</think>` before the tokens.
- Read each output token in call order. NEVER emit output tokens yourself.
- Emit the stop sequence ONLY after the call is fully written — NEVER announce a tool then stop (e.g. halting at "Let's run `cargo clippy`" with no `<｜tool▁call▁begin｜>` emitted). Write the complete call, THEN the stop sequence, THEN halt.
"#;

const GEMINI: &str = r#"## Format guide

Emit tool calls as Python inside a fenced ` ```tool_code ` block. Call each function as a method on `default_api`:

````text
```tool_code
default_api.function_name(arg="value", count=2)
```
````

Argument values are Python literals: `"strings"`, numbers, `True`/`False`, `None`, `[lists]`, `{"dicts": 1}`.

Call several functions in parallel as a Python list:

````text
```tool_code
[default_api.first(x="a"), default_api.second(y="b")]
```
````

Tool results arrive later in a ` ```tool_outputs ` block:

````text
```tool_outputs
verbatim tool result
```
````

Put any private reasoning in a fenced ` ```thinking ` block before the ` ```tool_code ` block:

````text
```thinking
brief reasoning
```
````

## Rules

- The function name MUST match a listed function; arguments are keyword form (`name=value`).
- Argument string values use only normal Python string escaping; never HTML-escape their contents — write `"a & b"`, not `"a &amp; b"`.
- Multiple calls = a single `[...]` list (or one `default_api...` call per line) inside one ` ```tool_code ` block.
- Put private reasoning in a ` ```thinking ` block before the ` ```tool_code ` block, never inside ` ```tool_code `.
- Read each ` ```tool_outputs ` block in call order. NEVER write a ` ```tool_outputs ` block yourself.
- Emit the ` ```tool_code ` block in full, THEN stop and halt — NEVER announce a tool then stop (e.g. halting at "Let's run `cargo clippy`" with no ` ```tool_code ` block emitted).
"#;

const GEMMA: &str = r#"## Format guide

Emit each tool call as one `<|tool_call>` block. The body is `call:NAME{key:value,...}`; wrap every string value in the `<|"|>` token:

```text
<|tool_call>call:function_name{path:<|"|>src/a.ts<|"|>,count:2}<tool_call|>
```

Non-string values are bare: numbers (`2`), `true`/`false`, `null`, lists `[<|"|>a<|"|>,<|"|>b<|"|>]`, and nested objects `{k:<|"|>v<|"|>}`.

Tool results arrive later in matching `<|tool_response>` blocks:

```text
<|tool_response>response:function_name{output:<|"|>verbatim result<|"|>}<tool_response|>
```

Optionally precede tool calls with private reasoning in a `<|channel>thought` block, closed by `<channel|>`:

```text
<|channel>thought
brief reasoning
<channel|>
```

## Rules

- `NAME` MUST match a listed function; arguments are `key:value` pairs separated by commas.
- String values between `<|"|>` tokens are raw literal text (no escaping); never HTML-escape them — write `a & b`, not `a &amp; b`.
- Multiple calls = consecutive `<|tool_call>...<tool_call|>` blocks; keep prose outside them.
- The closer is `<tool_call|>` (pipe on the right), not `</tool_call>` or `<|tool_call>`.
- Private reasoning goes in a `<|channel>thought…<channel|>` block before any call; NEVER put tool calls inside it.
- Read each `<|tool_response>` block in call order. NEVER write a `<|tool_response>` block yourself.
- Write each call in full, THEN stop and halt — NEVER announce a tool then stop (e.g. halting at "Let's run `cargo clippy`" with no `<|tool_call>` block emitted).
"#;

const GLM: &str = r#"## Format guide

Emit each call as a `<tool_call>` block. The function name goes on the same line as the opening tag, followed by one `<arg_key>`/`<arg_value>` pair per argument, closed by `</tool_call>`:

```text
<tool_call>get_weather
<arg_key>location</arg_key>
<arg_value>Beijing</arg_value>
<arg_key>days</arg_key>
<arg_value>3</arg_value>
</tool_call>
```

Tool results return in an observation block:

```text
<observation>
<tool_response>
verbatim tool result
</tool_response>
</observation>
```

## Rules

- The name after `<tool_call>` must match a listed function and sit on the same line.
- Emit one `<arg_key>name</arg_key>` + `<arg_value>value</arg_value>` pair per argument; omit unset optional args.
- `<arg_value>` bodies are read by regex (delimiter matching), NOT a real XML parser: write string values as raw literal text and never HTML-escape them (emit `a & b`, not `a &amp; b`; `<`/`>` stay literal); only the body's own `</arg_value>` closing tag is reserved. Non-string values are valid JSON.
- Multiple calls are consecutive `<tool_call>…</tool_call>` blocks.
- Private reasoning goes in `<think>…</think>`; NEVER put tool calls inside `<think>`.
- Read each `<tool_response>` in call order. NEVER emit `<tool_response>` yourself.
- Emit the stop sequence ONLY after the call is fully written — NEVER announce a tool then stop (e.g. halting at "Let's run `cargo clippy`" with no `<tool_call>` emitted). Write the complete call, THEN the stop sequence, THEN halt.
"#;

const HARMONY: &str = r#"## Format guide

Each function call is one assistant message on the `commentary` channel addressed to the function, emitted as text:

```text
<|start|>assistant<|channel|>commentary to=functions.function_name<|message|>{"arg":"value"}<|call|>
```

Put private reasoning in an `analysis` message:

```text
<|start|>assistant<|channel|>analysis<|message|>private reasoning<|end|>
```

Tool results arrive as messages authored by the function, addressed back to the assistant:

```text
<|start|>functions.function_name to=assistant<|channel|>commentary<|message|>verbatim tool result<|end|>
```

## Rules

- Recipient is `functions.` + a listed function name.
- Body is one JSON object matching the schema; omit optional arguments you are not setting.
- Argument string values use only normal JSON string escaping (`\"`, `\\`, `\n`); never HTML-escape their contents — write `a & b`, not `a &amp; b`.
- Multiple calls = consecutive call messages.
- An optional visible preamble is a `commentary` message ending `<|end|>`.
- NEVER put tool calls in `analysis`.
- NEVER wrap calls in Markdown/code fences.
- Read each tool-result message in call order. NEVER emit tool-result messages yourself.
- Emit the stop sequence ONLY after the call is fully written — NEVER announce a tool then stop (e.g. halting at "Let's run `cargo clippy`" with no `<|call|>` message emitted). Write the complete call, THEN the stop sequence, THEN halt.
"#;

const HERMES: &str = r#"## Format guide

Emit each tool call as a `<tool_call>` block wrapping a single-line JSON object with `name` and `arguments`:

```text
<tool_call>
{"name":"function_name","arguments":{"arg":"value"}}
</tool_call>
```

Results arrive later as `<tool_response>` blocks:

```text
<tool_response>
verbatim tool result
</tool_response>
```

## Rules

- `name` MUST match a listed function; `arguments` is a JSON object, never a stringified JSON.
- Argument string values use only normal JSON string escaping (`\"`, `\\`, `\n`); never HTML-escape their contents — write `a & b`, not `a &amp; b`.
- Emit multiple calls as consecutive `<tool_call>` blocks; keep any prose outside them.
- Read each `<tool_response>` in call order. NEVER emit `<tool_response>` yourself.
- Emit the stop sequence ONLY after the call is fully written — NEVER announce a tool then stop (e.g. halting at "Let's run `cargo clippy`" with no `<tool_call>` emitted). Write the complete call, THEN the stop sequence, THEN halt.
"#;

const KIMI: &str = r#"## Format guide

Emit every call of a turn inside one section. Each call is an id of the fixed form `functions.NAME:INDEX` followed by one JSON arguments object:

```text
<|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:INDEX<|tool_call_argument_begin|>{"arg":"value"}<|tool_call_end|><|tool_calls_section_end|>
```

Tool results arrive later as turns whose body is a `## Return of functions.NAME:INDEX` header then the verbatim result:

```text
<|im_system|>NAME<|im_middle|>## Return of functions.NAME:INDEX
verbatim tool result<|im_end|>
```

## Rules

- `NAME` MUST match a listed function exactly.
- Arguments MUST be one JSON object with double-quoted keys.
- Argument string values use only normal JSON string escaping (`\"`, `\\`, `\n`); never HTML-escape their contents — write `a & b`, not `a &amp; b`.
- Multiple calls = consecutive `<|tool_call_begin|>…<|tool_call_end|>` blocks in the same section; `INDEX` increments from `0`.
- Private reasoning, when supported, goes in `<think>…</think>` before the tool-call section; NEVER put tool calls inside `<think>`.
- Read each result turn in call order. NEVER emit result turns yourself.
- Emit the stop sequence ONLY after the call is fully written — NEVER announce a tool then stop (e.g. halting at "Let's run `cargo clippy`" with no `<|tool_call_begin|>` emitted). Write the complete call, THEN the stop sequence, THEN halt.
"#;

const MINIMAX: &str = r#"## Format guide

A call is a `<minimax:tool_call>` block wrapping one or more `<invoke>` blocks, each holding `<parameter>` children:

```text
<minimax:tool_call>
<invoke name="tool_name"><parameter name="arg_name">arg value</parameter></invoke>
</minimax:tool_call>
```

Results arrive later in a `<function_results>` block, one `<result>` per call (failures use `<error>` with `<stderr>` in place of `<result>` with `<stdout>`):

```text
<function_results>
<result>
<tool_name>tool_name</tool_name>
<stdout>verbatim tool result</stdout>
</result>
</function_results>
```

## Rules

- `name` MUST match a listed function.
- String/scalar parameters: exact text, spaces preserved — bodies are read by regex (delimiter matching), NOT a real XML parser, so never HTML-escape them (emit `a & b`, not `a &amp; b`; `<`/`>` stay literal); only the body's own `</parameter>` closing tag is reserved. Lists/objects: JSON.
- Multiple calls: multiple `<invoke>` blocks in one `<minimax:tool_call>`.
- You MAY write visible text before the calls.
- NEVER emit `tool_calls` JSON.
- NEVER use `<function_calls>` or the legacy `<tool_name>`/`<parameters>` call syntax.
- Read each `<result>`/`<error>` in call order. NEVER emit `<function_results>` yourself.
- Emit the stop sequence ONLY after the call is fully written — NEVER announce a tool then stop (e.g. halting at "Let's run `cargo clippy`" with no `<invoke>` emitted). Write the complete call, THEN the stop sequence, THEN halt.
"#;

const QWEN3: &str = r#"## Format guide

Emit each tool call as one `<tool_call>` block wrapping a single-line JSON object with `name` and a nested `arguments` object:

```text
<tool_call>
{"name":"function_name","arguments":{"arg":"value"}}
</tool_call>
```

Do any private reasoning in `<think>...</think>` before your tool calls.

Tool results arrive later in a user turn:

```text
<tool_response>
verbatim tool result
</tool_response>
```

## Rules

- `name` MUST match a listed function; `arguments` is a JSON object, never a JSON string.
- Argument string values use only normal JSON string escaping (`\"`, `\\`, `\n`); never HTML-escape their contents — write `a & b`, not `a &amp; b`.
- Multiple calls = consecutive `<tool_call>...</tool_call>` blocks; keep prose outside them.
- NEVER put tool calls inside `<think>`.
- Read each `<tool_response>` in call order. NEVER emit `<tool_response>` yourself.
- Emit the stop sequence ONLY after the call is fully written — NEVER announce a tool then stop (e.g. halting at "Let's run `cargo clippy`" with no `<tool_call>` emitted). Write the complete call, THEN the stop sequence, THEN halt.
"#;

const XML: &str = r#"## Format guide

A call is one `<invoke>` element whose `<parameter>` children carry its arguments:

```text
<invoke name="fn"><parameter name="arg">value</parameter></invoke>
```

Emit consecutive `<invoke>…</invoke>` blocks for multiple calls; you MAY wrap them in `<tool_calls>…</tool_calls>`. Each call's result arrives as a response block:

```text
<tool_response>
verbatim tool result
</tool_response>
```

## Rules

- `name` MUST match a listed function.
- Parameter values are read literally by regex (delimiter matching), NOT a real XML parser: write them verbatim and never HTML-escape (emit `a & b`, never `a &amp; b`; `<`/`>` stay literal too). Only the body's own `</parameter>` closing tag is reserved. Non-string values are JSON; add `string="false"` to a parameter only to force JSON parsing of a value the schema treats as a string.
- Read each `<tool_response>` in call order. NEVER emit `<tool_response>` yourself.
- Emit the stop sequence ONLY after the call is fully written — NEVER announce a tool then stop (e.g. halting at "Let's run `cargo clippy`" with no `<invoke>` emitted). Write the complete call, THEN the stop sequence, THEN halt.
"#;

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::*;

	#[test]
	fn every_dialect_injects_catalog_and_its_static_guide() {
		let schema = json!({"type": "object", "properties": {}});
		let tools = [InbandTool {
			name:        "ping",
			description: Some("Check reachability."),
			parameters:  &schema,
			examples:    &[],
		}];
		for dialect in Dialect::ALL {
			let mut prompt = String::new();
			write_inband_tool_prompt(&mut prompt, &tools, dialect).unwrap();
			assert!(prompt.starts_with("# Tools\n"));
			assert!(prompt.contains("\"name\":\"ping\""));
			assert!(prompt.contains(dialect_guide(dialect).trim()));
			assert!(prompt.ends_with(dialect_guide(dialect).trim()));
		}
	}
}
