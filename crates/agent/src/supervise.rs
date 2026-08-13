//! Cooperative invocation interruption followed by structural cancellation.

use std::time::Duration;

use omp_core::Str;
use omp_env::{ClientError, Invocation, InvocationEvent};
use omp_proto::env::v1::{EventStreamError, Update, Verdict};

/// Terminal observation from one environment invocation stream.
pub(crate) enum InvocationTerminal {
	/// The resource owner reported its authoritative verdict.
	Verdict(Verdict),
	/// Continuity was lost and no authoritative verdict can be observed.
	StreamError(EventStreamError),
	/// The stream closed without a terminal frame.
	Closed,
}

/// Drains an invocation, relaying updates and ignoring its one acceptance
/// frame.
pub(crate) async fn drain_terminal<F>(
	invocation: &mut Invocation,
	on_update: &mut F,
) -> Result<InvocationTerminal, ClientError>
where
	F: FnMut(Update),
{
	loop {
		match invocation.next_event().await? {
			Some(InvocationEvent::Accepted(_)) => {},
			Some(InvocationEvent::Update(update)) => on_update(update),
			Some(InvocationEvent::Verdict(verdict)) => {
				return Ok(InvocationTerminal::Verdict(verdict));
			},
			Some(InvocationEvent::StreamError(error)) => {
				return Ok(InvocationTerminal::StreamError(error));
			},
			None => return Ok(InvocationTerminal::Closed),
		}
	}
}

/// Requests cooperative interruption, then escalates after the grace period.
///
/// Escalation queues the invocation's [`omp_env::RunGuard`] cancellation but
/// deliberately retains and drains the invocation stream. Resource owners,
/// rather than this supervisor, remain authoritative about whether effects
/// landed and must report that truth in their terminal verdict.
pub(crate) async fn interrupt_with_grace<F>(
	invocation: &mut Invocation,
	reason: Str,
	grace: Duration,
	on_update: &mut F,
) -> Result<InvocationTerminal, ClientError>
where
	F: FnMut(Update),
{
	if let Err(error) = invocation.interrupt(reason).await {
		invocation.guard().cancel();
		return match drain_terminal(invocation, on_update).await {
			Ok(terminal) => Ok(terminal),
			Err(_) => Err(error),
		};
	}

	match tokio::time::timeout(grace, drain_terminal(invocation, on_update)).await {
		Ok(terminal) => terminal,
		Err(_) => {
			invocation.guard().cancel();
			drain_terminal(invocation, on_update).await
		},
	}
}
