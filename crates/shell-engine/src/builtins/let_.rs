use std::io::Write;

use clap::Parser;

use crate::{ExecutionExitCode, ExecutionResult, arithmetic::Evaluatable, builtins};

/// Evaluate arithmetic expressions.
#[derive(Parser)]
pub(crate) struct LetCommand {
	/// Arithmetic expressions to evaluate.
	#[arg(trailing_var_arg = true, allow_hyphen_values = true)]
	exprs: Vec<String>,
}

impl builtins::Command for LetCommand {
	type Error = crate::Error;

	async fn execute<SE: crate::ShellExtensions>(
		&self,
		context: crate::ExecutionContext<'_, SE>,
	) -> Result<crate::ExecutionResult, Self::Error> {
		let mut result = ExecutionExitCode::InvalidUsage.into();

		if self.exprs.is_empty() {
			writeln!(context.stderr(), "missing expression")?;
			return Ok(result);
		}

		for expr in &self.exprs {
			let parsed = crate::parser::arithmetic::parse(expr.as_str())?;
			let evaluated = parsed.eval(context.shell)?;

			if evaluated == 0 {
				result = ExecutionResult::general_error();
			} else {
				result = ExecutionResult::success();
			}
		}

		Ok(result)
	}
}
