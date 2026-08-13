use std::{
	future::Future,
	sync::Arc,
	task::{Context, Poll, Wake, Waker},
	thread,
	time::Duration,
};

use bytes::Bytes;
use frame::{client_frame, server_frame};
use omp_core::Str;
use omp_env::{EnvClient, InvocationEvent, frame};

const RECEIVE_TIMEOUT: Duration = Duration::from_secs(2);
const QUIET_PERIOD: Duration = Duration::from_millis(100);

struct ThreadWaker(thread::Thread);

impl Wake for ThreadWaker {
	fn wake(self: Arc<Self>) {
		self.0.unpark();
	}

	fn wake_by_ref(self: &Arc<Self>) {
		self.0.unpark();
	}
}

fn block_on<T>(future: impl Future<Output = T>) -> T {
	let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
	let mut context = Context::from_waker(&waker);
	let mut future = Box::pin(future);
	loop {
		match future.as_mut().poll(&mut context) {
			Poll::Ready(output) => return output,
			Poll::Pending => thread::park(),
		}
	}
}

fn receive(requests: &flume::Receiver<frame::ClientFrame>) -> frame::ClientFrame {
	requests
		.recv_timeout(RECEIVE_TIMEOUT)
		.expect("client frame")
}

fn respond(
	responses: &flume::Sender<frame::ServerFrame>,
	request_id: u64,
	body: server_frame::Body,
) {
	responses
		.send(frame::ServerFrame { request_id, body: Some(body), ..frame::ServerFrame::default() })
		.expect("open client response channel");
}

fn invoke_request(invocation_id: &str) -> frame::InvokeTool {
	frame::InvokeTool {
		invocation_id: invocation_id.into(),
		name: "contract-test".into(),
		..frame::InvokeTool::default()
	}
}

fn expect_invoke(frame: frame::ClientFrame, invocation_id: &str) -> u64 {
	assert_ne!(frame.request_id, 0);
	match frame.body {
		Some(client_frame::Body::InvokeTool(request)) => {
			assert_eq!(request.invocation_id, invocation_id);
		},
		body => panic!("expected InvokeTool, got {body:?}"),
	}
	frame.request_id
}

fn expect_scoped_cancel(frame: frame::ClientFrame, target_request_id: u64) {
	assert_eq!(frame.request_id, 0, "cancellation is a control frame");
	match frame.body {
		Some(client_frame::Body::Cancel(cancel)) => assert!(matches!(
			cancel.target,
			Some(frame::cancel_request::Target::TargetRequestId(id)) if id == target_request_id
		)),
		body => panic!("expected scoped CancelRequest, got {body:?}"),
	}
}

#[test]
fn hello_and_concurrent_requests_are_correlated_while_events_remain_observable() {
	let (client, transport) = EnvClient::in_process(0);
	let events = client.server_events();
	let (requests, responses) = transport.into_parts();

	let server = thread::spawn(move || {
		let hello = receive(&requests);
		assert_eq!(hello.request_id, 0);
		assert!(matches!(hello.body, Some(client_frame::Body::Hello(_))));
		respond(
			&responses,
			0,
			server_frame::Body::Update(frame::Update {
				invocation_id: "unsolicited".into(),
				json: Bytes::from_static(b"{\"live\":true}"),
				..frame::Update::default()
			}),
		);
		respond(
			&responses,
			0,
			server_frame::Body::Hello(frame::ServerHello {
				schema_rev: 7,
				server_version: "test-server".into(),
				..frame::ServerHello::default()
			}),
		);
		(requests, responses)
	});

	let hello =
		block_on(client.hello(frame::ClientHello {
			client: "test-client".into(),
			..frame::ClientHello::default()
		}))
		.expect("hello response");
	assert_eq!(hello.schema_rev, 7);
	assert_eq!(hello.server_version, "test-server");
	let event = events
		.recv_timeout(RECEIVE_TIMEOUT)
		.expect("unsolicited event");
	assert_eq!(event.request_id, 0);
	assert!(
		matches!(event.body, Some(server_frame::Body::Update(update)) if update.invocation_id == "unsolicited")
	);

	let (requests, responses) = server.join().expect("server thread");
	let mut first = block_on(client.invoke(invoke_request("first"))).expect("first invocation");
	let mut second = block_on(client.invoke(invoke_request("second"))).expect("second invocation");
	let first_id = expect_invoke(receive(&requests), "first");
	let second_id = expect_invoke(receive(&requests), "second");
	assert_ne!(first_id, second_id);

	respond(
		&responses,
		second_id,
		server_frame::Body::Update(frame::Update {
			invocation_id: "second".into(),
			json: Bytes::from_static(b"2"),
			..frame::Update::default()
		}),
	);
	respond(
		&responses,
		first_id,
		server_frame::Body::Update(frame::Update {
			invocation_id: "first".into(),
			json: Bytes::from_static(b"1"),
			..frame::Update::default()
		}),
	);
	assert!(matches!(
		block_on(first.next_event()).expect("first event"),
		Some(InvocationEvent::Update(update)) if update.invocation_id == "first" && update.json == Bytes::from_static(b"1")
	));
	assert!(matches!(
		block_on(second.next_event()).expect("second event"),
		Some(InvocationEvent::Update(update)) if update.invocation_id == "second" && update.json == Bytes::from_static(b"2")
	));

	for (request_id, invocation_id) in [(first_id, "first"), (second_id, "second")] {
		respond(
			&responses,
			request_id,
			server_frame::Body::Verdict(frame::Verdict {
				invocation_id: invocation_id.into(),
				..frame::Verdict::default()
			}),
		);
	}
	assert!(matches!(block_on(first.next_event()), Ok(Some(InvocationEvent::Verdict(_)))));
	assert!(matches!(block_on(second.next_event()), Ok(Some(InvocationEvent::Verdict(_)))));
}

#[test]
fn invocation_frames_preserve_commit_and_event_order() {
	let (client, transport) = EnvClient::in_process(0);
	let (requests, responses) = transport.into_parts();
	let mut invocation = block_on(client.invoke(invoke_request("ordered"))).expect("invocation");
	let request_id = expect_invoke(receive(&requests), "ordered");

	block_on(invocation.arg_text(Str::from("{\"path\":"))).expect("first argument fragment");
	block_on(invocation.arg_text(Str::from("\"a\"}"))).expect("second argument fragment");
	block_on(invocation.commit_args(Bytes::from_static(b"{\"path\":\"a\"}")))
		.expect("argument commitment");
	block_on(invocation.interrupt(Str::from("please stop"))).expect("interrupt");

	let frames = [receive(&requests), receive(&requests), receive(&requests), receive(&requests)];
	for frame in &frames {
		assert_eq!(frame.request_id, request_id);
	}
	assert!(matches!(
		frames[0].body.as_ref(),
		Some(client_frame::Body::ArgText(fragment)) if fragment.invocation_id == "ordered" && fragment.fragment == "{\"path\":"
	));
	assert!(matches!(
		frames[1].body.as_ref(),
		Some(client_frame::Body::ArgText(fragment)) if fragment.invocation_id == "ordered" && fragment.fragment == "\"a\"}"
	));
	assert!(matches!(
		frames[2].body.as_ref(),
		Some(client_frame::Body::ArgsCommitted(commit)) if commit.invocation_id == "ordered" && commit.raw == Bytes::from_static(b"{\"path\":\"a\"}")
	));
	assert!(matches!(
		frames[3].body.as_ref(),
		Some(client_frame::Body::Interrupt(interrupt)) if interrupt.invocation_id == "ordered" && interrupt.reason == "please stop"
	));

	respond(
		&responses,
		request_id,
		server_frame::Body::Update(frame::Update {
			invocation_id: "ordered".into(),
			json: Bytes::from_static(b"{\"step\":1}"),
			..frame::Update::default()
		}),
	);
	respond(
		&responses,
		request_id,
		server_frame::Body::Verdict(frame::Verdict {
			invocation_id: "ordered".into(),
			json: Bytes::from_static(b"{\"status\":\"ok\"}"),
			is_error: true,
			useless: true,
			..frame::Verdict::default()
		}),
	);
	assert!(
		matches!(block_on(invocation.next_event()), Ok(Some(InvocationEvent::Update(update))) if update.json == Bytes::from_static(b"{\"step\":1}"))
	);
	assert!(
		matches!(block_on(invocation.next_event()), Ok(Some(InvocationEvent::Verdict(verdict))) if verdict.json == Bytes::from_static(b"{\"status\":\"ok\"}") && verdict.is_error && verdict.useless)
	);
	drop(invocation);
	assert!(matches!(requests.recv_timeout(QUIET_PERIOD), Err(flume::RecvTimeoutError::Timeout)));
}

#[test]
fn invocation_guard_cancels_once_but_relinquish_and_terminal_events_disarm_it() {
	let (client, transport) = EnvClient::in_process(0);
	let (requests, responses) = transport.into_parts();

	let dropped = block_on(client.invoke(invoke_request("dropped"))).expect("dropped invocation");
	let dropped_id = expect_invoke(receive(&requests), "dropped");
	drop(dropped);
	expect_scoped_cancel(receive(&requests), dropped_id);
	assert!(matches!(requests.recv_timeout(QUIET_PERIOD), Err(flume::RecvTimeoutError::Timeout)));

	let explicitly_cancelled =
		block_on(client.invoke(invoke_request("explicit"))).expect("explicitly cancelled invocation");
	let explicitly_cancelled_id = expect_invoke(receive(&requests), "explicit");
	assert_eq!(explicitly_cancelled.guard().request_id(), explicitly_cancelled_id);
	explicitly_cancelled.guard().cancel();
	explicitly_cancelled.guard().cancel();
	assert!(!explicitly_cancelled.guard().is_armed());
	drop(explicitly_cancelled);
	expect_scoped_cancel(receive(&requests), explicitly_cancelled_id);
	assert!(matches!(requests.recv_timeout(QUIET_PERIOD), Err(flume::RecvTimeoutError::Timeout)));

	let relinquished =
		block_on(client.invoke(invoke_request("relinquished"))).expect("relinquished invocation");
	let _relinquished_id = expect_invoke(receive(&requests), "relinquished");
	drop(relinquished.relinquish());
	assert!(matches!(requests.recv_timeout(QUIET_PERIOD), Err(flume::RecvTimeoutError::Timeout)));

	let mut completed =
		block_on(client.invoke(invoke_request("completed"))).expect("completed invocation");
	let completed_id = expect_invoke(receive(&requests), "completed");
	respond(
		&responses,
		completed_id,
		server_frame::Body::Verdict(frame::Verdict {
			invocation_id: "completed".into(),
			..frame::Verdict::default()
		}),
	);
	assert!(matches!(block_on(completed.next_event()), Ok(Some(InvocationEvent::Verdict(_)))));
	drop(completed);
	assert!(matches!(requests.recv_timeout(QUIET_PERIOD), Err(flume::RecvTimeoutError::Timeout)));
}

#[test]
fn command_guard_cancels_only_its_request_and_does_not_own_the_session() {
	let (client, transport) = EnvClient::in_process(0);
	let (requests, responses) = transport.into_parts();
	let session = Bytes::from_static(b"session-token");

	let server_session = session.clone();
	let server = thread::spawn(move || {
		let open = receive(&requests);
		assert!(matches!(open.body, Some(client_frame::Body::OpenSession(_))));
		respond(
			&responses,
			open.request_id,
			server_frame::Body::SessionOpened(frame::OpenSessionResponse {
				session: server_session,
				..frame::OpenSessionResponse::default()
			}),
		);
		(requests, responses)
	});
	let opened =
		block_on(client.open_session(frame::OpenSessionRequest::default())).expect("session");
	assert_eq!(opened.session, session);
	let (requests, responses) = server.join().expect("server thread");

	let command = block_on(
		client.exec(frame::ExecRequest { session: session.clone(), ..frame::ExecRequest::default() }),
	)
	.expect("exec request");
	let exec = receive(&requests);
	let exec_id = exec.request_id;
	assert!(
		matches!(exec.body, Some(client_frame::Body::Exec(request)) if request.session == session)
	);

	let mut other = block_on(client.invoke(invoke_request("other"))).expect("other request");
	let other_id = expect_invoke(receive(&requests), "other");
	drop(command);
	expect_scoped_cancel(receive(&requests), exec_id);
	assert_ne!(exec_id, other_id);

	respond(
		&responses,
		other_id,
		server_frame::Body::Verdict(frame::Verdict {
			invocation_id: "other".into(),
			..frame::Verdict::default()
		}),
	);
	assert!(matches!(block_on(other.next_event()), Ok(Some(InvocationEvent::Verdict(_)))));

	let server_session = session.clone();
	let server = thread::spawn(move || {
		let close = receive(&requests);
		assert_ne!(close.request_id, exec_id);
		assert!(matches!(
			close.body,
			Some(client_frame::Body::CloseSession(request)) if request.session == server_session
		));
		respond(
			&responses,
			close.request_id,
			server_frame::Body::SessionClosed(frame::CloseSessionResponse {
				session: server_session,
				..frame::CloseSessionResponse::default()
			}),
		);
	});
	let closed = block_on(client.close_session(frame::CloseSessionRequest {
		session: session.clone(),
		..frame::CloseSessionRequest::default()
	}))
	.expect("close session after command cancellation");
	assert_eq!(closed.session, session);
	server.join().expect("server thread");
}
