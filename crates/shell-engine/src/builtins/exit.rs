use clap::Parser;

use crate::{ExecutionControlFlow, ExecutionResult, builtins};

/// Exit the shell.
#[derive(Parser)]
pub(crate) struct ExitCommand {
	/// The exit code to return.
	#[arg(allow_hyphen_values = true)]
	code: Option<i64>,
}

impl builtins::Command for ExitCommand {
	type Error = crate::Error;

	async fn execute<SE: crate::ShellExtensions>(
		&self,
		context: crate::ExecutionContext<'_, SE>,
	) -> Result<crate::ExecutionResult, Self::Error> {
		#[expect(clippy::cast_sign_loss, reason = "shell exit status is defined modulo 256")]
		let code_8bit = if let Some(code_32bit) = &self.code {
			(code_32bit & 0xff) as u8
		} else {
			context.shell.last_exit_status()
		};

		let mut result = ExecutionResult::new(code_8bit);
		result.next_control_flow = ExecutionControlFlow::ExitShell;

		Ok(result)
	}
}
