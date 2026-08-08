use clap::Parser;

use crate::{ExecutionResult, builtins};

/// Pop a path from the current directory stack.
#[derive(Parser)]
pub(crate) struct PopdCommand {
	/// Pop the path without changing the current working directory.
	#[clap(short = 'n')]
	no_directory_change: bool,
	//
	// TODO(popd): implement +N and -N
}

impl builtins::Command for PopdCommand {
	type Error = crate::builtins::dirs::DirError;

	async fn execute<SE: crate::ShellExtensions>(
		&self,
		context: crate::ExecutionContext<'_, SE>,
	) -> Result<crate::ExecutionResult, Self::Error> {
		if let Some(popped) = context.shell.directory_stack_mut().pop() {
			if !self.no_directory_change {
				context.shell.set_working_dir(&popped)?;
			}

			// Display dirs.
			let dirs_cmd = crate::builtins::dirs::DirsCommand::default();
			dirs_cmd.execute(context).await?;

			Ok(ExecutionResult::success())
		} else {
			Err(crate::builtins::dirs::DirError::DirStackEmpty)
		}
	}
}
