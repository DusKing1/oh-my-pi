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
	/// Structural cancellation did not yield owner truth within its bound.
	CancelUnobserved,
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
/// Escalation queues the invocation's [`omp_env::RunGuard`] cancellation and
/// retains the stream for one more bounded observation window. Resource owners,
/// rather than this supervisor, remain authoritative about whether effects
/// landed; a terminal verdict observed in either window wins.
pub(crate) async fn interrupt_with_grace<F>(
	invocation: &mut Invocation,
	reason: Str,
	grace: Duration,
	on_update: &mut F,
) -> Result<InvocationTerminal, ClientError>
where
	F: FnMut(Update),
{
	let cooperative = async {
		invocation.interrupt(reason).await?;
		drain_terminal(invocation, on_update).await
	};
	match tokio::time::timeout(grace, cooperative).await {
		Ok(Ok(terminal)) => Ok(terminal),
		Ok(Err(interrupt_error)) => {
			invocation.guard().cancel();
			match tokio::time::timeout(grace, drain_terminal(invocation, on_update)).await {
				Ok(Ok(terminal)) => Ok(terminal),
				Ok(Err(_)) | Err(_) => Err(interrupt_error),
			}
		},
		Err(_) => {
			invocation.guard().cancel();
			match tokio::time::timeout(grace, drain_terminal(invocation, on_update)).await {
				Ok(terminal) => terminal,
				Err(_) => Ok(InvocationTerminal::CancelUnobserved),
			}
		},
	}
}
