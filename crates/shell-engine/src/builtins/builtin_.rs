use clap::Parser;

use crate::{ExecutionResult, builtins};

/// Directly invokes a built-in, without going through typical search order.
#[derive(Default, Parser)]
pub(crate) struct BuiltinCommand {
	#[clap(skip)]
	args: Vec<crate::CommandArg>,
}

impl builtins::DeclarationCommand for BuiltinCommand {
	fn set_declarations(&mut self, args: Vec<crate::CommandArg>) {
		self.args = args;
	}
}

impl builtins::Command for BuiltinCommand {
	type Error = crate::Error;

	async fn execute<SE: crate::ShellExtensions>(
		&self,
		mut context: crate::ExecutionContext<'_, SE>,
	) -> Result<crate::ExecutionResult, Self::Error> {
		if self.args.is_empty() {
			return Ok(ExecutionResult::success());
		}

		let args: Vec<_> = self.args.iter().skip(1).cloned().collect();
		if args.is_empty() {
			return Ok(ExecutionResult::success());
		}

		let builtin_name = args[0].to_string();

		if let Some(builtin) = context.shell.builtins().get(&builtin_name)
			&& !builtin.disabled
		{
			context.command_name = builtin_name;
			(builtin.execute_func)(context, args).await
		} else {
			Err(crate::ErrorKind::BuiltinNotFound(builtin_name).into())
		}
	}
}
