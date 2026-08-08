from __future__ import annotations

import base64
import contextlib
import http.server
import json
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import anthropic
import httpx
import openai

GATEWAY_KEY = "stock-sdk-gateway-key"
OPENAI_PROVIDER_KEY = "stock-sdk-openai-provider-key"
ANTHROPIC_PROVIDER_KEY = "stock-sdk-anthropic-provider-key"
OPENAI_MODEL = "openai/gpt-4o-mini"
ANTHROPIC_MODEL = "anthropic/claude-haiku-4-5"
CHAT_PROMPT = "chat non-stream weather tool"
CHAT_STREAM_PROMPT = "chat stream weather tool"
TOOL = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Read deterministic fixture weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}
RESPONSES_TOOL = {
    "type": "function",
    "name": TOOL["function"]["name"],
    "description": TOOL["function"]["description"],
    "parameters": TOOL["function"]["parameters"],
}
ANTHROPIC_TOOL = {
    "name": "weather",
    "description": "Read deterministic fixture weather",
    "input_schema": TOOL["function"]["parameters"],
}
AUDIO = b"ID3stock-sdk-audio"
IMAGE = b"\x89PNG\r\n\x1a\nstock-sdk-image"


class ProviderState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[tuple[str, str, dict[str, str], bytes]] = []

    def record(self, method: str, path: str, headers: Any, body: bytes) -> None:
        lowered = {key.lower(): value for key, value in headers.items()}
        with self.lock:
            self.requests.append((method, path, lowered, body))

    def matching(self, suffix: str) -> list[tuple[str, str, dict[str, str], bytes]]:
        with self.lock:
            return [request for request in self.requests if request[1].endswith(suffix)]

    def total(self) -> int:
        with self.lock:
            return len(self.requests)

    def snapshot(self) -> list[tuple[str, str, dict[str, str], bytes]]:
        with self.lock:
            return list(self.requests)


STATE = ProviderState()


class ProviderHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        STATE.record("POST", self.path, self.headers, body)
        if self.path.startswith("/openai/"):
            if self.headers.get("authorization") != f"Bearer {OPENAI_PROVIDER_KEY}":
                self._json(401, {"error": {"message": "bad upstream OpenAI key", "type": "authentication_error"}})
                return
            self._openai(body)
            return
        if self.path.startswith("/anthropic/"):
            if self.headers.get("x-api-key") != ANTHROPIC_PROVIDER_KEY:
                self._json(401, {"type": "error", "error": {"type": "authentication_error", "message": "bad upstream Anthropic key"}})
                return
            if not self.headers.get("anthropic-version"):
                self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "anthropic-version required"}})
                return
            self._anthropic(body)
            return
        self._json(404, {"error": {"message": "fixture route not found"}})

    def _openai(self, body: bytes) -> None:
        if self.path.endswith("/embeddings"):
            request = json.loads(body)
            inputs = request["input"]
            if not isinstance(inputs, list):
                inputs = [inputs]
            data = [
                {"object": "embedding", "index": index, "embedding": [float(index), 0.25, 0.75]}
                for index, _value in reversed(list(enumerate(inputs)))
            ]
            self._json(200, {"object": "list", "data": data, "model": request["model"], "usage": {"prompt_tokens": 5, "total_tokens": 5}})
            return
        if self.path.endswith("/audio/speech"):
            self._bytes(200, AUDIO, "audio/mpeg")
            return
        if self.path.endswith("/audio/transcriptions") or self.path.endswith("/audio/translations"):
            self._json(200, {"text": "fixture transcript", "language": "en", "duration": 1.25, "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}})
            return
        request = json.loads(body)
        if not self.path.endswith("/responses"):
            self._json(404, {"error": {"message": "unexpected OpenAI provider path"}})
            return
        encoded = json.dumps(request, separators=(",", ":"))
        if "force_rate_limit" in encoded:
            self._json(529, {"error": {"message": "fixture overloaded", "type": "server_error", "code": "overloaded"}}, {"retry-after": "301", "x-request-id": "req_rate_limit"})
            return
        if any(tool.get("type") == "image_generation" for tool in request.get("tools", [])):
            self._json(200, {
                "id": "resp_image_fixture",
                "object": "response",
                "status": "completed",
                "model": request["model"],
                "output": [{"id": "ig_fixture", "type": "image_generation_call", "status": "completed", "result": base64.b64encode(IMAGE).decode(), "revised_prompt": "fixture revised prompt"}],
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            })
            return
        self._sse(openai_events(request.get("model", "gpt-4o-mini")))

    def _anthropic(self, body: bytes) -> None:
        request = json.loads(body)
        encoded = json.dumps(request, separators=(",", ":"))
        if self.path.endswith("/messages/count_tokens"):
            self._json(200, {"input_tokens": 7})
            return
        if "force_rate_limit" in encoded:
            self._json(529, {"type": "error", "error": {"type": "overloaded_error", "message": "fixture overloaded"}}, {"retry-after": "301", "request-id": "req_rate_limit"})
            return
        self._sse(anthropic_events(request.get("model", "claude-haiku-4-5")))

    def _json(self, status: int, value: object, headers: dict[str, str] | None = None) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _bytes(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _sse(self, events: list[tuple[str, dict[str, object]]]) -> None:
        payload = b"".join(
            f"event: {name}\ndata: {json.dumps(value, separators=(',', ':'))}\n\n".encode()
            for name, value in events
        )
        self._bytes(200, payload, "text/event-stream")


def openai_events(model: str) -> list[tuple[str, dict[str, object]]]:
    output = [
        {"id": "msg_fixture", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "stock sdk answer", "annotations": []}]},
        {"id": "fc_fixture", "type": "function_call", "call_id": "call_fixture", "name": "weather", "arguments": "{\"city\":\"Paris\"}", "status": "completed"},
    ]
    return [
        ("response.created", {"type": "response.created", "response": {"id": "resp_fixture", "status": "in_progress", "model": model, "output": []}}),
        ("response.output_item.added", {"type": "response.output_item.added", "output_index": 0, "item": {"id": "msg_fixture", "type": "message", "role": "assistant", "status": "in_progress", "content": []}}),
        ("response.content_part.added", {"type": "response.content_part.added", "item_id": "msg_fixture", "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
        ("response.output_text.delta", {"type": "response.output_text.delta", "item_id": "msg_fixture", "output_index": 0, "content_index": 0, "delta": "stock sdk answer"}),
        ("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": output[0]}),
        ("response.output_item.added", {"type": "response.output_item.added", "output_index": 1, "item": {"id": "fc_fixture", "type": "function_call", "call_id": "call_fixture", "name": "weather", "arguments": "", "status": "in_progress"}}),
        ("response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta", "item_id": "fc_fixture", "output_index": 1, "delta": "{\"city\":\"Paris\"}"}),
        ("response.output_item.done", {"type": "response.output_item.done", "output_index": 1, "item": output[1]}),
        ("response.completed", {"type": "response.completed", "response": {"id": "resp_fixture", "object": "response", "created_at": 1, "status": "completed", "model": model, "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "temperature": 1.0, "top_p": 1.0, "usage": {"input_tokens": 13, "output_tokens": 8, "total_tokens": 21, "input_tokens_details": {"cached_tokens": 3}, "output_tokens_details": {"reasoning_tokens": 0}}}}),
    ]


def anthropic_events(model: str) -> list[tuple[str, dict[str, object]]]:
    return [
        ("message_start", {"type": "message_start", "message": {"id": "msg_fixture", "type": "message", "role": "assistant", "content": [], "model": model, "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 0, "cache_read_input_tokens": 2, "cache_creation_input_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "stock sdk answer"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_fixture", "name": "weather", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":\"Paris\"}"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"input_tokens": 11, "output_tokens": 9, "cache_read_input_tokens": 2, "cache_creation_input_tokens": 1}}),
        ("message_stop", {"type": "message_stop"}),
    ]


def wait_for_socket(path: Path, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"omp serve exited before readiness\nstdout={stdout.decode()}\nstderr={stderr.decode()}")
        if path.exists():
            with contextlib.suppress(OSError):
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    probe.connect(str(path))
                    return
                finally:
                    probe.close()
        time.sleep(0.025)
    raise AssertionError("omp serve did not create a connectable Unix socket")


def tools() -> list[dict[str, object]]:
    return [TOOL]


def assert_openai(client: openai.OpenAI) -> None:
    models = client.models.list()
    assert any(model.id == OPENAI_MODEL for model in models.data)

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": CHAT_PROMPT}],
        tools=tools(),
    )
    assert completion.model == OPENAI_MODEL
    assert completion.object == "chat.completion"
    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "stock sdk answer"
    assert choice.message.tool_calls and choice.message.tool_calls[0].function.name == "weather"
    assert json.loads(choice.message.tool_calls[0].function.arguments) == {"city": "Paris"}
    assert completion.usage
    assert completion.usage.prompt_tokens == 13 and completion.usage.completion_tokens == 8 and completion.usage.total_tokens == 21
    assert completion.usage.prompt_tokens_details
    assert completion.usage.prompt_tokens_details.cached_tokens == 3
    assert completion.usage.prompt_tokens_details.cache_write_tokens == 0
    assert completion.usage.completion_tokens_details
    assert completion.usage.completion_tokens_details.reasoning_tokens == 0

    chunks = list(client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": CHAT_STREAM_PROMPT}],
        tools=tools(),
        stream=True,
        stream_options={"include_usage": True},
    ))
    assert chunks and all(
        chunk.object == "chat.completion.chunk" and chunk.model == OPENAI_MODEL
        for chunk in chunks
    )
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "stock sdk answer"
    assert sum(bool(chunk.choices and chunk.choices[0].finish_reason) for chunk in chunks) == 1
    assert next(chunk.choices[0].finish_reason for chunk in chunks if chunk.choices and chunk.choices[0].finish_reason) == "tool_calls"
    stream_usage = [chunk.usage for chunk in chunks if chunk.usage]
    assert len(stream_usage) == 1
    assert stream_usage[0].prompt_tokens == 13 and stream_usage[0].completion_tokens == 8 and stream_usage[0].total_tokens == 21
    assert stream_usage[0].prompt_tokens_details and stream_usage[0].prompt_tokens_details.cached_tokens == 3
    assert stream_usage[0].prompt_tokens_details.cache_write_tokens == 0
    assert stream_usage[0].completion_tokens_details and stream_usage[0].completion_tokens_details.reasoning_tokens == 0
    arguments = "".join(
        call.function.arguments or ""
        for chunk in chunks if chunk.choices
        for call in (chunk.choices[0].delta.tool_calls or [])
    )
    assert json.loads(arguments) == {"city": "Paris"}

    response = client.responses.create(model=OPENAI_MODEL, input="use the weather tool", tools=[RESPONSES_TOOL])
    assert response.status == "completed" and response.output_text == "stock sdk answer"
    calls = [item for item in response.output if item.type == "function_call"]
    assert len(calls) == 1 and calls[0].name == "weather" and json.loads(calls[0].arguments) == {"city": "Paris"}
    assert response.usage
    assert response.usage.input_tokens == 13 and response.usage.output_tokens == 8 and response.usage.total_tokens == 21
    assert response.usage.input_tokens_details.cached_tokens == 3
    assert response.usage.input_tokens_details.cache_write_tokens == 0
    assert response.usage.output_tokens_details.reasoning_tokens == 0

    with client.responses.stream(model=OPENAI_MODEL, input="stream the weather tool", tools=[RESPONSES_TOOL]) as stream:
        events = list(stream)
        deltas = [event.delta for event in events if event.type == "response.output_text.delta"]
        terminals = [event for event in events if event.type in {"response.completed", "response.incomplete", "response.failed"}]
        final = stream.get_final_response()
    assert "".join(deltas) == "stock sdk answer"
    assert len(terminals) == 1 and terminals[0].type == "response.completed"
    assert final.status == "completed" and final.usage
    assert final.usage.input_tokens == 13 and final.usage.output_tokens == 8 and final.usage.total_tokens == 21
    assert any(item.type == "function_call" and item.name == "weather" for item in final.output)
    assert final.usage.input_tokens_details.cached_tokens == 3
    assert final.usage.input_tokens_details.cache_write_tokens == 0
    assert final.usage.output_tokens_details.reasoning_tokens == 0

    embedding = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=["first", "second"],
    )
    assert [item.index for item in embedding.data] == [0, 1]
    assert embedding.data[1].embedding == [1.0, 0.25, 0.75]
    assert embedding.usage.prompt_tokens == 5

    image = client.images.generate(
        model="gpt-image-2",
        prompt="a deterministic blue square",
        output_format="png",
        extra_body={"response_format": "b64_json"},
    )
    assert image.data and base64.b64decode(image.data[0].b64_json or "") == IMAGE
    assert image.data[0].revised_prompt == "fixture revised prompt"
    edited = client.images.edit(
        model="gpt-image-2",
        image=("fixture.png", IMAGE, "image/png"),
        prompt="edit the deterministic blue square",
        output_format="png",
        extra_body={"response_format": "b64_json"},
    )
    assert edited.data and base64.b64decode(edited.data[0].b64_json or "") == IMAGE

    speech = client.audio.speech.create(model="openai/gpt-4o-mini-tts", voice="alloy", input="hello fixture")
    assert speech.content == AUDIO
    transcript = client.audio.transcriptions.create(
        model="openai/gpt-4o-mini-transcribe",
        file=("fixture.mp3", AUDIO, "audio/mpeg"),
    )
    assert transcript.text == "fixture transcript"
    translation = client.audio.translations.create(
        model="openai/gpt-4o-mini-transcribe",
        file=("fixture.mp3", AUDIO, "audio/mpeg"),
    )
    assert translation.text == "fixture transcript"


def assert_anthropic(client: anthropic.Anthropic) -> None:
    models = client.models.list(limit=100)
    assert any(model.id == ANTHROPIC_MODEL for model in models.data)

    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": "use the weather tool"}],
        tools=[ANTHROPIC_TOOL],
    )
    assert message.stop_reason == "tool_use"
    assert any(block.type == "text" and block.text == "stock sdk answer" for block in message.content)
    assert any(block.type == "tool_use" and block.name == "weather" and block.input == {"city": "Paris"} for block in message.content)
    assert message.usage.input_tokens == 11 and message.usage.output_tokens == 9

    assert message.usage.cache_read_input_tokens == 2 and message.usage.cache_creation_input_tokens == 1
    with client.messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": "stream the weather tool"}],
        tools=[ANTHROPIC_TOOL],
    ) as stream:
        events = list(stream)
        text = "".join(
            event.delta.text
            for event in events
            if event.type == "content_block_delta" and event.delta.type == "text_delta"
        )
        terminals = [event for event in events if event.type == "message_stop"]
        final = stream.get_final_message()
    assert text == "stock sdk answer" and final.stop_reason == "tool_use"
    assert len(terminals) == 1
    assert final.usage.input_tokens == 11 and final.usage.output_tokens == 9
    assert any(block.type == "tool_use" and block.input == {"city": "Paris"} for block in final.content)

    assert final.usage.cache_read_input_tokens == 2 and final.usage.cache_creation_input_tokens == 1
    counted = client.messages.count_tokens(
        model=ANTHROPIC_MODEL,
        messages=[{"role": "user", "content": "count these tokens"}],
        tools=[ANTHROPIC_TOOL],
    )
    assert counted.input_tokens == 7


def assert_auth_and_retry(socket_path: Path) -> None:
    with openai.OpenAI(
        api_key="wrong",
        base_url="http://omp/v1",
        http_client=httpx.Client(transport=httpx.HTTPTransport(uds=str(socket_path)), timeout=10),
        max_retries=1,
    ) as bad_openai:
        before_openai = STATE.total()
        try:
            bad_openai.models.list()
            raise AssertionError("wrong OpenAI gateway key unexpectedly authenticated")
        except openai.AuthenticationError as error:
            assert error.status_code == 401
        assert STATE.total() == before_openai

        before_chat = STATE.total()
        try:
            bad_openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": "must not reach provider chat"}],
            )
            raise AssertionError("wrong OpenAI gateway key reached Chat Completions")
        except openai.AuthenticationError as error:
            assert error.status_code == 401
        assert STATE.total() == before_chat

    with anthropic.Anthropic(
        api_key="wrong",
        base_url="http://omp",
        http_client=httpx.Client(transport=httpx.HTTPTransport(uds=str(socket_path)), timeout=10),
        max_retries=1,
    ) as bad_anthropic:
        before_anthropic = STATE.total()
        try:
            bad_anthropic.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=16,
                messages=[{"role": "user", "content": "must not reach provider"}],
            )
            raise AssertionError("wrong Anthropic gateway key unexpectedly authenticated")
        except anthropic.AuthenticationError as error:
            assert error.status_code == 401
        assert STATE.total() == before_anthropic

    with openai.OpenAI(
        api_key=GATEWAY_KEY,
        base_url="http://omp/v1",
        http_client=httpx.Client(transport=httpx.HTTPTransport(uds=str(socket_path)), timeout=10),
        max_retries=1,
    ) as retry_openai:
        before = len(STATE.matching("/responses"))
        try:
            retry_openai.responses.create(model=OPENAI_MODEL, input="force_rate_limit")
            raise AssertionError("OpenAI rate limit unexpectedly succeeded")
        except openai.RateLimitError as error:
            assert error.status_code == 429 and error.response.headers.get("retry-after") == "301"
        assert len(STATE.matching("/responses")) >= before + 2

    with anthropic.Anthropic(
        api_key=GATEWAY_KEY,
        base_url="http://omp",
        http_client=httpx.Client(transport=httpx.HTTPTransport(uds=str(socket_path)), timeout=10),
        max_retries=1,
    ) as retry_anthropic:
        before = len(STATE.matching("/messages"))
        try:
            retry_anthropic.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=32,
                messages=[{"role": "user", "content": "force_rate_limit"}],
            )
            raise AssertionError("Anthropic rate limit unexpectedly succeeded")
        except anthropic.RateLimitError as error:
            assert error.status_code == 429 and error.response.headers.get("retry-after") == "301"
        assert len(STATE.matching("/messages")) >= before + 2


def assert_provider_requests() -> None:
    requests = STATE.snapshot()
    paths = [path for _method, path, _headers, _body in requests]
    for _method, path, headers, body in requests:
        assert all(GATEWAY_KEY not in value for value in headers.values())
        assert GATEWAY_KEY.encode() not in body
        if path.startswith("/openai/"):
            assert headers.get("authorization") == f"Bearer {OPENAI_PROVIDER_KEY}"
        if path.startswith("/anthropic/"):
            assert headers.get("x-api-key") == ANTHROPIC_PROVIDER_KEY
            assert headers.get("anthropic-version") == "2023-06-01"
    for suffix in (
        "/openai/v1/responses",
        "/openai/v1/embeddings",
        "/openai/v1/audio/speech",
        "/openai/v1/audio/transcriptions",
        "/openai/v1/audio/translations",
        "/anthropic/v1/messages",
        "/anthropic/v1/messages/count_tokens",
    ):
        assert suffix in paths, f"production route never reached fixture {suffix}"

    chat_requests: dict[str, list[tuple[str, str, dict[str, str], dict[str, object]]]] = {
        CHAT_PROMPT: [],
        CHAT_STREAM_PROMPT: [],
    }
    for method, path, headers, raw_body in requests:
        if path != "/openai/v1/responses" or headers.get("content-type") != "application/json":
            continue
        body = json.loads(raw_body)
        for prompt in chat_requests:
            expected_input = [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
            if body.get("input") == expected_input:
                chat_requests[prompt].append((method, path, headers, body))
    for prompt, matching in chat_requests.items():
        assert len(matching) == 1, f"expected one provider request for {prompt!r}, got {len(matching)}"
        method, path, headers, body = matching[0]
        assert method == "POST" and path == "/openai/v1/responses"
        assert headers.get("authorization") == f"Bearer {OPENAI_PROVIDER_KEY}"
        assert body["model"] == "gpt-4o-mini" and body["stream"] is True
        assert headers.get("accept") == "text/event-stream"
        function_tools = [
            tool
            for tool in body.get("tools", [])
            if tool.get("type") == "function" and tool.get("name") == "weather"
        ]
        assert len(function_tools) == 1
        assert function_tools[0]["description"] == TOOL["function"]["description"]
        assert function_tools[0]["parameters"] == TOOL["function"]["parameters"]

    response_bodies = [
        json.loads(body)
        for _method, path, headers, body in requests
        if path == "/openai/v1/responses" and headers.get("content-type") == "application/json"
    ]
    assert any(body.get("model") == "gpt-4o-mini" and body.get("stream") is True for body in response_bodies)
    assert any(
        any(tool.get("type") == "function" and tool.get("name") == "weather" for tool in body.get("tools", []))
        for body in response_bodies
    )
    assert any(
        any(
            isinstance(item, dict)
            and any(isinstance(part, dict) and part.get("type") == "input_image" for part in item.get("content", []))
            for item in body.get("input", [])
        )
        for body in response_bodies
    )
    embedding_body = json.loads(next(body for _method, path, _headers, body in requests if path == "/openai/v1/embeddings"))
    assert embedding_body == {
        "model": "text-embedding-3-small",
        "input": ["first", "second"],
        "encoding_format": "float",
    }
    count_body = json.loads(next(body for _method, path, _headers, body in requests if path == "/anthropic/v1/messages/count_tokens"))
    assert count_body["model"] == "claude-haiku-4-5"
    assert count_body["messages"][0]["role"] == "user"


def main() -> None:
    omp = os.environ["OMP_STOCK_SDK_BIN"]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ProviderHandler)
    server.daemon_threads = True
    fixture_thread = threading.Thread(target=server.serve_forever, daemon=True)
    fixture_thread.start()
    host, port = server.server_address
    with tempfile.TemporaryDirectory(prefix="omp-stock-sdk-") as directory:
        scratch = Path(directory)
        project = scratch / "project"
        home = scratch / "home"
        data = scratch / "data"
        (project / ".omp").mkdir(parents=True)
        home.mkdir()
        data.mkdir()
        (project / ".omp" / "providers.toml").write_text(
            f'[providers.openai]\nbase_url = "http://{host}:{port}/openai/v1"\n\n'
            f'[providers.anthropic]\nbase_url = "http://{host}:{port}/anthropic"\n'
        )
        socket_path = scratch / "gateway.sock"
        env = {
            "HOME": str(home),
            "OMP_DATA_DIR": str(data),
            "OMP_GATEWAY_TOKEN": GATEWAY_KEY,
            "OPENAI_API_KEY": OPENAI_PROVIDER_KEY,
            "ANTHROPIC_API_KEY": ANTHROPIC_PROVIDER_KEY,
        }
        process = subprocess.Popen(
            [omp, "serve", "--uds", str(socket_path), "--data-dir", str(data)],
            cwd=project,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        shutdown_timed_out = False
        try:
            wait_for_socket(socket_path, process)
            with openai.OpenAI(
                api_key=GATEWAY_KEY,
                base_url="http://omp/v1",
                http_client=httpx.Client(transport=httpx.HTTPTransport(uds=str(socket_path)), timeout=10),
                max_retries=0,
            ) as openai_client, anthropic.Anthropic(
                api_key=GATEWAY_KEY,
                base_url="http://omp",
                http_client=httpx.Client(transport=httpx.HTTPTransport(uds=str(socket_path)), timeout=10),
                max_retries=0,
            ) as anthropic_client:
                assert_openai(openai_client)
                assert_anthropic(anthropic_client)
            assert_auth_and_retry(socket_path)
            assert_provider_requests()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                shutdown_timed_out = True
                process.kill()
                stdout, stderr = process.communicate(timeout=5)
            server.shutdown()
            server.server_close()
            fixture_thread.join(timeout=5)
        assert not shutdown_timed_out, "omp serve did not exit after SIGTERM"
        assert process.returncode == 0, f"omp serve failed\nstdout={stdout.decode()}\nstderr={stderr.decode()}"
        assert not socket_path.exists(), "graceful daemon shutdown left its Unix socket behind"
        for secret in (GATEWAY_KEY, OPENAI_PROVIDER_KEY, ANTHROPIC_PROVIDER_KEY):
            assert secret.encode() not in stdout and secret.encode() not in stderr


if __name__ == "__main__":
    main()
