//! The `kill` builtin, moved from `pi-shell`.

use std::io::Write;

use clap::Parser;

use crate::{
	ExecutionContext, ExecutionExitCode, ExecutionResult, builtins, sys, traps::TrapSignal,
};

/// Signal a job or process.
#[derive(Parser)]
pub(crate) struct KillCommand {
	/// Name of the signal to send.
	#[arg(short = 's', value_name = "SIG_NAME")]
	signal_name:      Option<String>,
	/// Number of the signal to send.
	#[arg(short = 'n', value_name = "SIG_NUM")]
	signal_number:    Option<usize>,
	/// List known signal names.
	#[arg(short = 'l', short_alias = 'L')]
	list_signals:     bool,
	// Interpretation of these depends on whether -l is present.
	#[arg(allow_hyphen_values = true)]
	args:             Vec<String>,
	/// Process/job operands given after the `--` end-of-options marker. clap
	/// consumes `--` before `execute`, so these are captured separately and are
	/// always operands — never signal specifications (preserves negative PIDs).
	#[arg(last = true, allow_hyphen_values = true)]
	post_marker_args: Vec<String>,
}

impl builtins::Command for KillCommand {
	type Error = crate::Error;

	#[allow(unknown_lints, reason = "unused_async_trait_impl is unknown to the pinned CI nightly")]
	#[allow(
		clippy::unused_async_trait_impl,
		reason = "the builtin Command trait declares execute as async"
	)]
	async fn execute<SE: crate::ShellExtensions>(
		&self,
		context: ExecutionContext<'_, SE>,
	) -> std::result::Result<ExecutionResult, Self::Error> {
		let default_signal = if let Some(signal_name) = &self.signal_name {
			if let Ok(signal) = KillSignal::parse(signal_name) {
				signal
			} else {
				writeln!(
					context.stderr(),
					"{}: invalid signal name: {}",
					context.command_name,
					signal_name
				)?;
				return Ok(ExecutionExitCode::InvalidUsage.into());
			}
		} else {
			KillSignal::parse("TERM")?
		};
		let mut signal = match self.signal_number {
			Some(signal_number) => {
				let Ok(signal_number) = i32::try_from(signal_number) else {
					writeln!(
						context.stderr(),
						"{}: invalid signal number: {}",
						context.command_name,
						signal_number
					)?;
					return Ok(ExecutionExitCode::InvalidUsage.into());
				};
				if let Ok(signal) = KillSignal::parse(&signal_number.to_string()) {
					signal
				} else {
					writeln!(
						context.stderr(),
						"{}: invalid signal number: {}",
						context.command_name,
						signal_number
					)?;
					return Ok(ExecutionExitCode::InvalidUsage.into());
				}
			},
			None => default_signal,
		};

		// Interpret the pre-`--` args as an optional leading `-sigspec`, followed
		// by PID/jobspec operands. Once a signal or operand has been seen, later
		// hyphen-led arguments remain operands so negative process-group IDs survive.
		let mut operands: Vec<&String> = Vec::new();
		let mut options_done = self.signal_name.is_some() || self.signal_number.is_some();
		let mut consumed_marker = false;
		for arg in &self.args {
			if !consumed_marker && arg == "--" {
				consumed_marker = true;
				options_done = true;
				continue;
			}
			if !options_done && let Some(spec) = arg.strip_prefix('-').filter(|spec| !spec.is_empty())
			{
				signal = if let Ok(signal) = KillSignal::parse(spec) {
					signal
				} else {
					writeln!(context.stderr(), "{}: invalid signal name", context.command_name)?;
					return Ok(ExecutionExitCode::InvalidUsage.into());
				};
				options_done = true;
				continue;
			}
			options_done = true;
			operands.push(arg);
		}
		operands.extend(&self.post_marker_args);

		if self.list_signals {
			return print_kill_signals(&context, operands);
		}
		if operands.is_empty() {
			writeln!(context.stderr(), "{}: invalid usage", context.command_name)?;
			return Ok(ExecutionExitCode::InvalidUsage.into());
		}

		#[cfg(unix)]
		let exists = |target: i32| {
			// SAFETY: signal 0 only checks target existence and permission.
			unsafe { libc::kill(target, 0) == 0 }
		};
		#[cfg(windows)]
		let exists = |target: i32| process_exists(target);

		let mut had_failure = false;
		for operand in operands {
			if context.is_cancelled() {
				return Ok(ExecutionExitCode::Interrupted.into());
			}
			if operand.starts_with('%') {
				let job = match context.shell.jobs_mut().resolve_job_spec(operand) {
					Ok(job) => job,
					Err(error) => {
						writeln!(context.stderr(), "{}: {}: {}", context.command_name, operand, error)?;
						had_failure = true;
						continue;
					},
				};
				#[cfg(unix)]
				{
					let mut targets: Vec<i32> = job
						.process_ids()
						.filter_map(|pid| {
							// SAFETY: getpgid reads process-group metadata for a managed child.
							let pgid = unsafe { libc::getpgid(pid) };
							(pgid > 0).then_some(-pgid)
						})
						.collect();
					if targets.is_empty()
						&& let Some(pgid) = job.process_group_id()
					{
						targets.push(-pgid);
					}
					targets.sort_unstable();
					targets.dedup();
					let succeeded = match signal {
						KillSignal::Probe => targets.iter().copied().any(&exists),
						KillSignal::Signal(signal) => {
							let mut succeeded = false;
							for target in targets {
								if sys::signal::kill_process(target, signal).is_ok() {
									succeeded = true;
								}
							}
							succeeded
						},
					};
					if !succeeded {
						writeln!(
							context.stderr(),
							"{}: {}: failed to send signal",
							context.command_name,
							operand
						)?;
						had_failure = true;
					}
				}
				#[cfg(windows)]
				{
					let expected_handles = job.external_process_count();
					let handles = job.duplicate_kill_handles();
					let mut succeeded = expected_handles != 0 && handles.len() == expected_handles;
					for handle in &handles {
						let handled = match signal {
							KillSignal::Probe => crate::processes::process_handle_is_running(handle),
							KillSignal::Signal(_) => crate::processes::terminate_process_handle(handle),
						};
						if !handled {
							succeeded = false;
						}
					}
					if !succeeded {
						writeln!(
							context.stderr(),
							"{}: {}: failed to send signal",
							context.command_name,
							operand
						)?;
						had_failure = true;
					}
				}
				continue;
			}

			let pid = match crate::int_utils::parse(operand, 10) {
				Ok(pid) => pid,
				Err(err) => {
					writeln!(context.stderr(), "{}: {}: {}", context.command_name, operand, err)?;
					had_failure = true;
					continue;
				},
			};
			match signal {
				KillSignal::Probe => {
					if !exists(pid) {
						writeln!(
							context.stderr(),
							"{}: {}: failed to send signal",
							context.command_name,
							operand
						)?;
						had_failure = true;
					}
				},
				KillSignal::Signal(signal) => {
					if let Err(err) = sys::signal::kill_process(pid, signal) {
						writeln!(context.stderr(), "{}: {}: {}", context.command_name, operand, err)?;
						had_failure = true;
					}
				},
			}
		}

		if had_failure {
			Ok(ExecutionResult::general_error())
		} else {
			Ok(ExecutionResult::success())
		}
	}
}

#[cfg(windows)]
fn process_exists(pid: i32) -> bool {
	use windows_sys::Win32::{
		Foundation::CloseHandle,
		System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION},
	};

	let Ok(pid) = u32::try_from(pid) else {
		return false;
	};
	// SAFETY: the numeric process id is supplied by the user and the returned
	// handle is checked before use.
	let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
	if handle.is_null() {
		return false;
	}
	// SAFETY: `handle` was returned by `OpenProcess` and is closed exactly once.
	let _ = unsafe { CloseHandle(handle) };
	true
}

fn print_kill_signals<'a>(
	context: &ExecutionContext<'_, impl crate::ShellExtensions>,
	signals: impl IntoIterator<Item = &'a String>,
) -> std::result::Result<ExecutionResult, crate::Error> {
	let mut result = ExecutionResult::success();
	let mut signals = signals.into_iter().peekable();
	if signals.peek().is_none() {
		return crate::traps::format_signals(
			context.stdout(),
			TrapSignal::iterator().filter(|signal| !matches!(signal, TrapSignal::Exit)),
		)
		.map(|()| ExecutionResult::success());
	}
	for value in signals {
		enum PrintedSignal {
			Name(&'static str),
			Number(i32),
		}
		let signal = if let Ok(number) = value.parse::<i32>() {
			TrapSignal::try_from(number).map(|signal| {
				PrintedSignal::Name(
					signal
						.as_str()
						.strip_prefix("SIG")
						.unwrap_or(signal.as_str()),
				)
			})
		} else {
			TrapSignal::try_from(value.as_str()).map(|signal| {
				i32::try_from(signal)
					.map_or(PrintedSignal::Name(signal.as_str()), PrintedSignal::Number)
			})
		};
		match signal {
			Ok(PrintedSignal::Name(name)) => writeln!(context.stdout(), "{name}")?,
			Ok(PrintedSignal::Number(number)) => writeln!(context.stdout(), "{number}")?,
			Err(err) => {
				writeln!(context.stderr(), "{err}")?;
				result = ExecutionResult::general_error();
			},
		}
	}
	Ok(result)
}

#[cfg(test)]
impl KillCommand {
	fn listed_signals(&self) -> impl Iterator<Item = &String> {
		let mut consumed_marker = false;
		self
			.args
			.iter()
			.filter(move |arg| {
				if !consumed_marker && *arg == "--" {
					consumed_marker = true;
					false
				} else {
					true
				}
			})
			.chain(&self.post_marker_args)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn listed(args: &[&str]) -> Vec<String> {
		let cmd = KillCommand::try_parse_from(args).unwrap();
		cmd.listed_signals().cloned().collect()
	}

	#[test]
	fn lists_post_marker_operands() {
		assert_eq!(listed(&["kill", "-l", "--", "9"]), ["9"]);
	}

	#[test]
	fn lists_pre_and_post_marker_operands() {
		assert_eq!(listed(&["kill", "-l", "TERM", "--", "9"]), ["TERM", "9"]);
	}

	#[test]
	fn lists_pre_marker_operands_without_marker() {
		assert_eq!(listed(&["kill", "-l", "TERM", "HUP"]), ["TERM", "HUP"]);
	}
}

/// A `kill` signal argument: a real signal, or the "does this process
/// exist?" probe that signal 0 requests.
#[derive(Clone, Copy)]
enum KillSignal {
	Probe,
	Signal(TrapSignal),
}

impl KillSignal {
	fn parse(value: &str) -> std::result::Result<Self, crate::Error> {
		if let Ok(number) = value.parse::<i32>() {
			if number == 0 {
				Ok(Self::Probe)
			} else {
				TrapSignal::try_from(number).map(Self::Signal)
			}
		} else {
			TrapSignal::try_from(value).map(Self::Signal)
		}
	}
}

/// Resolves a signal name or number to its number.
///
/// Shared with `pkill`, which accepts the same `-SIGNAL` spellings.
#[allow(
	dead_code,
	reason = "shared with optional process-match builtins that may be feature-disabled"
)]
pub fn signal_number(value: &str) -> Option<i32> {
	let value = value
		.strip_prefix("SIG")
		.or_else(|| value.strip_prefix("sig"))
		.unwrap_or(value);
	if let Ok(number) = value.parse::<i32>() {
		#[cfg(target_os = "linux")]
		return (0..=libc::SIGRTMAX()).contains(&number).then_some(number);
		#[cfg(target_os = "macos")]
		return (0..=31).contains(&number).then_some(number);
		#[cfg(not(unix))]
		return (0..=64).contains(&number).then_some(number);
	}
	match KillSignal::parse(value).ok()? {
		KillSignal::Probe => Some(0),
		KillSignal::Signal(signal) => i32::try_from(signal).ok(),
	}
}
