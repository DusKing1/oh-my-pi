use clap::Parser;

use crate::{ExecutionControlFlow, ExecutionExitCode, ExecutionResult, builtins};

/// Breaks out of a control-flow loop.
#[derive(Parser)]
pub(crate) struct BreakCommand {
	/// If specified, indicates which nested loop to break out of.
	#[clap(default_value_t = 1)]
	which_loop: i8,
}

impl builtins::Command for BreakCommand {
	type Error = crate::Error;

	async fn execute<SE: crate::ShellExtensions>(
		&self,
		_context: crate::ExecutionContext<'_, SE>,
	) -> Result<crate::ExecutionResult, Self::Error> {
		// If specified, which_loop needs to be positive.
		if self.which_loop <= 0 {
			return Ok(ExecutionExitCode::InvalidUsage.into());
		}

		let mut result = ExecutionResult::success();

		result.next_control_flow = ExecutionControlFlow::BreakLoop {
			#[expect(clippy::cast_sign_loss, reason = "positive loop counts are checked above")]
			levels: (self.which_loop - 1) as usize,
		};

		Ok(result)
	}
}
