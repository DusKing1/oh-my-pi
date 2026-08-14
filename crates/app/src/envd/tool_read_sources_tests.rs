use std::{
	convert::Infallible,
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
	task::{Context, Poll},
	time::{Duration, Instant},
};

use futures::Stream;
use http::{
	HeaderValue, StatusCode,
	header::{ACCEPT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, LOCATION, RETRY_AFTER, USER_AGENT},
};
use http_body_util::{Full, StreamBody};
use hyper::{
	Request, Response,
	body::{Frame, Incoming},
	service::service_fn,
};
use hyper_util::rt::TokioIo;
use parking_lot::Mutex;
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle};

use super::*;

struct TestServer {
	base: String,
	task: JoinHandle<()>,
}

impl Drop for TestServer {
	fn drop(&mut self) {
		self.task.abort();
	}
}

async fn serve(
	handler: impl Fn(Request<Incoming>) -> Response<Full<Bytes>> + Send + Sync + 'static,
) -> TestServer {
	let listener = TcpListener::bind(("127.0.0.1", 0))
		.await
		.expect("bind test server");
	let address = listener.local_addr().expect("read test server address");
	let handler = Arc::new(handler);
	let task = tokio::spawn(async move {
		loop {
			let Ok((stream, _)) = listener.accept().await else {
				break;
			};
			let handler = Arc::clone(&handler);
			tokio::spawn(async move {
				let service = service_fn(move |request| {
					let response = handler(request);
					async move { Ok::<_, Infallible>(response) }
				});
				let _ = hyper::server::conn::http1::Builder::new()
					.serve_connection(TokioIo::new(stream), service)
					.await;
			});
		}
	});
	TestServer { base: format!("http://{address}"), task }
}

#[tokio::test]
async fn follows_redirects_and_reports_the_final_url() {
	let paths = Arc::new(Mutex::new(Vec::new()));
	let seen_paths = Arc::clone(&paths);
	let server = serve(move |request| {
		seen_paths.lock().push(request.uri().to_string());
		assert_eq!(
			request
				.headers()
				.get(ACCEPT_ENCODING)
				.and_then(|v| v.to_str().ok()),
			Some("identity")
		);
		match request.uri().path() {
			"/start" => Response::builder()
				.status(StatusCode::FOUND)
				.header(LOCATION, "/nested/step?from=start")
				.body(Full::new(Bytes::new()))
				.expect("first redirect"),
			"/nested/step" => Response::builder()
				.status(StatusCode::TEMPORARY_REDIRECT)
				.header(LOCATION, "../final?done=1")
				.body(Full::new(Bytes::new()))
				.expect("second redirect"),
			"/final" => Response::builder()
				.status(StatusCode::OK)
				.header(CONTENT_TYPE, "text/plain")
				.body(Full::new(Bytes::from_static(b"redirected")))
				.expect("final response"),
			path => panic!("unexpected redirect path {path}"),
		}
	})
	.await;

	let response = SystemHttpClient::new()
		.get(HttpRequest::new(format!("{}/start#authored-fragment", server.base)))
		.await
		.expect("follow redirects");

	assert_eq!(response.final_url.as_str(), format!("{}/final?done=1", server.base));
	assert_eq!(response.status, 200);
	assert_eq!(response.body, Bytes::from_static(b"redirected"));
	assert_eq!(*paths.lock(), [
		"/start".to_owned(),
		"/nested/step?from=start".to_owned(),
		"/final?done=1".to_owned(),
	]);
}

#[tokio::test]
async fn caps_each_redirect_chain_at_twenty_hops() {
	let calls = Arc::new(AtomicUsize::new(0));
	let handler_calls = Arc::clone(&calls);
	let server = serve(move |_| {
		handler_calls.fetch_add(1, Ordering::SeqCst);
		Response::builder()
			.status(StatusCode::FOUND)
			.header(LOCATION, "/loop")
			.body(Full::new(Bytes::new()))
			.expect("redirect loop response")
	})
	.await;

	SystemHttpClient::new()
		.get(HttpRequest::new(format!("{}/loop", server.base)))
		.await
		.expect_err("the twenty-hop redirect cap must reject a loop");

	assert_eq!(
		calls.load(Ordering::SeqCst),
		(MAX_REDIRECTS + 1) * USER_AGENTS.len(),
		"transport errors rotate user agents, but every chain remains capped",
	);
}

#[tokio::test]
async fn retries_one_rate_limit_with_the_same_user_agent_and_retry_after_delay() {
	let calls = Arc::new(AtomicUsize::new(0));
	let agents = Arc::new(Mutex::new(Vec::new()));
	let times = Arc::new(Mutex::new(Vec::new()));
	let handler_calls = Arc::clone(&calls);
	let handler_agents = Arc::clone(&agents);
	let handler_times = Arc::clone(&times);
	let server = serve(move |request| {
		handler_agents.lock().push(
			request.headers()[USER_AGENT]
				.to_str()
				.expect("text user agent")
				.to_owned(),
		);
		handler_times.lock().push(Instant::now());
		let attempt = handler_calls.fetch_add(1, Ordering::SeqCst);
		Response::builder()
			.status(StatusCode::TOO_MANY_REQUESTS)
			.header(RETRY_AFTER, "0.025")
			.header("x-attempt", attempt.to_string())
			.body(Full::new(Bytes::from(if attempt == 0 {
				"first rate limit"
			} else {
				"terminal rate limit"
			})))
			.expect("rate-limit response")
	})
	.await;

	let response = SystemHttpClient::new()
		.get(HttpRequest::new(format!("{}/limited", server.base)))
		.await
		.expect("HTTP status remains response truth");

	assert_eq!(calls.load(Ordering::SeqCst), 2, "only one 429 retry is allowed");
	assert_eq!(*agents.lock(), [USER_AGENTS[0].to_owned(), USER_AGENTS[0].to_owned(),]);
	let times = times.lock();
	assert!(times[1].duration_since(times[0]) >= Duration::from_millis(20));
	assert_eq!(response.status, 429);
	assert_eq!(response.header("x-attempt"), Some("1"));
	assert_eq!(response.body, Bytes::from_static(b"terminal rate limit"));
}

#[tokio::test]
async fn does_not_retry_ordinary_http_errors() {
	let calls = Arc::new(AtomicUsize::new(0));
	let handler_calls = Arc::clone(&calls);
	let server = serve(move |_| {
		handler_calls.fetch_add(1, Ordering::SeqCst);
		Response::builder()
			.status(StatusCode::INTERNAL_SERVER_ERROR)
			.header(CONTENT_TYPE, "text/plain")
			.body(Full::new(Bytes::from_static(b"ordinary failure")))
			.expect("ordinary error response")
	})
	.await;

	let response = SystemHttpClient::new()
		.get(HttpRequest::new(format!("{}/failure", server.base)))
		.await
		.expect("HTTP errors remain response truth");

	assert_eq!(calls.load(Ordering::SeqCst), 1);
	assert_eq!(response.status, 500);
	assert_eq!(response.body, Bytes::from_static(b"ordinary failure"));
}

#[test]
fn retry_after_parser_uses_defaults_and_the_hard_upper_bound() {
	assert_eq!(retry_after(None), DEFAULT_RETRY_AFTER);
	assert_eq!(retry_after(Some(&HeaderValue::from_static("0"))), Duration::ZERO);
	assert_eq!(retry_after(Some(&HeaderValue::from_static("2.5"))), Duration::from_millis(2_500));
	assert_eq!(retry_after(Some(&HeaderValue::from_static("999"))), MAX_RETRY_AFTER);
}

#[tokio::test]
async fn rotates_the_exact_three_user_agents_and_returns_the_last_bot_block() {
	assert_eq!(USER_AGENTS, [
		"curl/8.0",
		"Mozilla/5.0 (compatible; TextBot/1.0)",
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \
		 Chrome/131.0.0.0 Safari/537.36",
	]);
	let calls = Arc::new(AtomicUsize::new(0));
	let agents = Arc::new(Mutex::new(Vec::new()));
	let handler_calls = Arc::clone(&calls);
	let handler_agents = Arc::clone(&agents);
	let server = serve(move |request| {
		handler_agents.lock().push(
			request.headers()[USER_AGENT]
				.to_str()
				.expect("text user agent")
				.to_owned(),
		);
		let attempt = handler_calls.fetch_add(1, Ordering::SeqCst);
		Response::builder()
			.status(StatusCode::SERVICE_UNAVAILABLE)
			.header(CONTENT_TYPE, "text/html; charset=windows-1252")
			.header("x-bot-attempt", attempt.to_string())
			.body(Full::new(Bytes::from(format!("<p>Cloudflare challenge {attempt}</p>"))))
			.expect("bot-block response")
	})
	.await;

	let response = SystemHttpClient::new()
		.get(HttpRequest::new(format!("{}/blocked", server.base)))
		.await
		.expect("final bot block is response truth");

	assert_eq!(calls.load(Ordering::SeqCst), USER_AGENTS.len());
	assert_eq!(*agents.lock(), USER_AGENTS.map(str::to_owned));
	assert_eq!(response.status, 503);
	assert_eq!(response.header("x-bot-attempt"), Some("2"));
	assert_eq!(response.body, Bytes::from_static(b"<p>Cloudflare challenge 2</p>"));
}

#[tokio::test]
async fn preserves_declared_charset_body_bytes_and_duplicate_headers() {
	let server = serve(|_| {
		Response::builder()
			.status(StatusCode::OK)
			.header(CONTENT_TYPE, "text/plain; charset=windows-1252")
			.header("set-cookie", "a=1")
			.header("set-cookie", "b=2")
			.header("x-source-encoding", "windows-1252")
			.body(Full::new(Bytes::from_static(&[0x80, 0x93, 0xff])))
			.expect("charset response")
	})
	.await;

	let response = SystemHttpClient::new()
		.get(HttpRequest::new(format!("{}/charset", server.base)))
		.await
		.expect("fetch charset response");

	assert_eq!(response.content_type.as_ref().map(Str::as_str), Some("text/plain"));
	assert_eq!(response.header("content-type"), Some("text/plain; charset=windows-1252"));
	assert_eq!(response.header("x-source-encoding"), Some("windows-1252"));
	assert_eq!(response.body.as_ref(), &[0x80, 0x93, 0xff]);
	let cookies = response
		.headers
		.iter()
		.filter(|(name, _)| name.eq_ignore_ascii_case("set-cookie"))
		.map(|(_, value)| value.as_str())
		.collect::<Vec<_>>();
	assert_eq!(cookies.len(), 2);
	assert!(cookies.contains(&"a=1"));
	assert!(cookies.contains(&"b=2"));
}

#[tokio::test]
async fn rejects_declared_oversize_from_headers_without_waiting_for_body() {
	let listener = TcpListener::bind(("127.0.0.1", 0))
		.await
		.expect("bind declared-size server");
	let address = listener.local_addr().expect("read declared-size address");
	let server = tokio::spawn(async move {
		let (stream, _) = listener
			.accept()
			.await
			.expect("accept declared-size client");
		let service = service_fn(|_| async {
			let body = StreamBody::new(futures::stream::pending::<Result<Frame<Bytes>, Infallible>>());
			Ok::<_, Infallible>(
				Response::builder()
					.header(CONTENT_LENGTH, "9")
					.body(body)
					.expect("declared-size response"),
			)
		});
		let _ = hyper::server::conn::http1::Builder::new()
			.serve_connection(TokioIo::new(stream), service)
			.await;
	});

	let result = tokio::time::timeout(
		Duration::from_secs(2),
		SystemHttpClient::new()
			.get(HttpRequest::new(format!("http://{address}/declared")).with_max_bytes(8)),
	)
	.await
	.expect("declared Content-Length must reject before the pending body")
	.expect_err("declared oversized response must fail");
	server.abort();
	assert_eq!(result, WebError::ResponseTooLarge { max_bytes: 8 });
}

#[tokio::test]
async fn rejects_a_chunked_fifty_one_mibibyte_response_at_the_hard_cap() {
	let listener = TcpListener::bind(("127.0.0.1", 0))
		.await
		.expect("bind streamed-cap server");
	let address = listener.local_addr().expect("read streamed-cap address");
	let server = tokio::spawn(async move {
		let (stream, _) = listener.accept().await.expect("accept streamed-cap client");
		let service = service_fn(|_| async {
			let chunk = Bytes::from(vec![0; 1024 * 1024]);
			let frames = (0..=50).map(move |_| Ok::<_, Infallible>(Frame::data(chunk.clone())));
			let body = StreamBody::new(futures::stream::iter(frames));
			Ok::<_, Infallible>(Response::new(body))
		});
		let _ = hyper::server::conn::http1::Builder::new()
			.serve_connection(TokioIo::new(stream), service)
			.await;
	});

	let error = SystemHttpClient::new()
		.get(HttpRequest::new(format!("http://{address}/chunked")).with_max_bytes(usize::MAX))
		.await
		.expect_err("51 MiB chunked response must fail");
	server.abort();
	assert_eq!(error, WebError::ResponseTooLarge { max_bytes: MAX_BYTES });
}

struct DropProbeStream {
	polled:  Option<oneshot::Sender<()>>,
	dropped: Option<oneshot::Sender<()>>,
}

impl Stream for DropProbeStream {
	type Item = Result<Frame<Bytes>, Infallible>;

	fn poll_next(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		if let Some(polled) = self.polled.take() {
			let _ = polled.send(());
		}
		Poll::Pending
	}
}

impl Drop for DropProbeStream {
	fn drop(&mut self) {
		if let Some(dropped) = self.dropped.take() {
			let _ = dropped.send(());
		}
	}
}

#[tokio::test]
async fn dropping_the_get_future_cancels_an_in_flight_stream() {
	let listener = TcpListener::bind(("127.0.0.1", 0))
		.await
		.expect("bind cancellation server");
	let address = listener.local_addr().expect("read cancellation address");
	let (polled_tx, mut polled_rx) = oneshot::channel();
	let (dropped_tx, dropped_rx) = oneshot::channel();
	let signals = Arc::new(Mutex::new(Some((polled_tx, dropped_tx))));
	let server_signals = Arc::clone(&signals);
	let server = tokio::spawn(async move {
		let (stream, _) = listener.accept().await.expect("accept cancellation client");
		let service = service_fn(move |_| {
			let (polled, dropped) = server_signals
				.lock()
				.take()
				.expect("one cancellation request");
			async move {
				let body =
					StreamBody::new(DropProbeStream { polled: Some(polled), dropped: Some(dropped) });
				Ok::<_, Infallible>(
					Response::builder()
						.header(CONTENT_LENGTH, "1")
						.body(body)
						.expect("cancellation response"),
				)
			}
		});
		let _ = hyper::server::conn::http1::Builder::new()
			.serve_connection(TokioIo::new(stream), service)
			.await;
	});

	let client = SystemHttpClient::new();
	let mut request = Box::pin(client.get(HttpRequest::new(format!("http://{address}/pending"))));
	tokio::time::timeout(Duration::from_secs(2), async {
		tokio::select! {
			polled = &mut polled_rx => polled.expect("server body poll signal"),
			result = &mut request => panic!("pending response unexpectedly completed: {result:?}"),
		}
	})
	.await
	.expect("transport must begin consuming the pending body");
	drop(request);

	tokio::time::timeout(Duration::from_secs(2), dropped_rx)
		.await
		.expect("dropping the request future must release the response stream")
		.expect("server body drop signal");
	server.abort();
}
