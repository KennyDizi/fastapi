import json
import threading
import time
from collections.abc import AsyncIterable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import anyio
import fastapi.routing
import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import ResponseValidationError
from fastapi.responses import StreamingResponse
from fastapi.sse import EventSourceResponse
from fastapi.telemetry import get_telemetry_data
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import Span
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from .conftest import metric_points, server_spans


class Item(BaseModel):
    name: str
    price: float


def _chunk(kind: str, index: int) -> Any:
    if kind == "raw":
        return f"chunk-{index}".encode()
    return {"n": index}


def _mount(app: FastAPI, endpoint: Any, kind: str, dependency: Any) -> None:
    dependencies = [Depends(dependency)]
    if kind == "raw":
        app.get("/", response_class=StreamingResponse, dependencies=dependencies)(
            endpoint
        )
    elif kind == "sse":
        app.get("/", response_class=EventSourceResponse, dependencies=dependencies)(
            endpoint
        )
    else:
        app.get("/", dependencies=dependencies)(endpoint)


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("kind", ["jsonl", "sse", "raw"])
def test_stream_operation_spans(telemetry: Any, kind: str, sync: bool) -> None:
    config, exporter, reader = telemetry
    tracer = config["tracer_provider"].get_tracer("test.streaming")
    seen: dict[str, Any] = {"contexts": [], "users": []}

    async def dependency() -> None:
        seen["dependency_thread"] = threading.get_ident()

    def produce() -> Iterator[Any]:
        for index in (1, 2):
            span = trace.get_current_span()
            assert isinstance(span, Span)
            assert span.name == "fastapi.endpoint"
            seen["contexts"].append(span.get_span_context())
            seen["thread"] = threading.get_ident()
            assert get_telemetry_data() is not None
            with tracer.start_as_current_span("user.work"):
                seen["users"].append(trace.get_current_span().get_span_context())
            yield _chunk(kind, index)

    if sync:

        def endpoint() -> Iterator[Any]:
            yield from produce()

    else:

        async def endpoint() -> AsyncIterable[Any]:
            for item in produce():
                yield item

    app = FastAPI(telemetry=config)
    _mount(app, endpoint, kind, dependency)
    response = TestClient(app).get("/")
    assert response.status_code == 200
    if kind == "raw":
        assert response.content == b"chunk-1chunk-2"
    elif kind == "jsonl":
        assert [json.loads(line) for line in response.text.splitlines()] == [
            {"n": 1},
            {"n": 2},
        ]
    else:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert ": ping" not in response.text
        assert (
            len(
                [
                    line
                    for line in response.text.splitlines()
                    if line.startswith("data:")
                ]
            )
            == 2
        )

    spans = exporter.get_finished_spans()
    (server,) = server_spans(exporter)
    (endpoint_span,) = [span for span in spans if span.name == "fastapi.endpoint"]
    serializations = [span for span in spans if span.name == "fastapi.serialization"]
    users = [span for span in spans if span.name == "user.work"]
    (dependencies,) = [span for span in spans if span.name == "fastapi.dependencies"]
    assert len(serializations) == (0 if kind == "raw" else 2)
    assert len(users) == 2
    assert [context.span_id for context in seen["contexts"]] == [
        endpoint_span.context.span_id,
        endpoint_span.context.span_id,
    ]
    assert (seen["thread"] != seen["dependency_thread"]) == sync
    function_name = f"{endpoint.__module__}.{endpoint.__qualname__}"
    assert endpoint_span.attributes["code.function.name"] == function_name
    assert endpoint_span.parent.span_id == server.context.span_id
    assert endpoint_span.status.status_code == StatusCode.UNSET
    assert dependencies.parent.span_id == server.context.span_id
    for span in serializations:
        assert span.parent.span_id == server.context.span_id
        assert span.attributes["code.function.name"] == function_name
        assert span.status.status_code == StatusCode.UNSET
        assert endpoint_span.start_time <= span.start_time
        assert span.end_time <= endpoint_span.end_time
    for span, context in zip(users, seen["users"], strict=True):
        assert span.context.span_id == context.span_id
        assert span.parent.span_id == endpoint_span.context.span_id
    assert all(span.context.trace_id == server.context.trace_id for span in spans)
    assert all(not span.events for span in spans)
    assert metric_points(reader=reader)[0].count == 1
    assert not trace.get_current_span().get_span_context().is_valid


def _stream_failure(error: BaseException) -> BaseException:
    # SSE failures surface from the producer task group.
    nested = getattr(error, "exceptions", None)
    if nested:
        assert len(nested) == 1
        return _stream_failure(nested[0])
    return error


def _invalid(kind: str) -> dict[str, Any]:
    assert kind in {"jsonl", "sse"}
    return {"name": "bad", "price": "nope"}


def _valid_or_chunk(kind: str) -> Any:
    if kind == "raw":
        return b"chunk"
    return {"name": "ok", "price": 1.0}


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize(
    ("kind", "failure"),
    [
        ("jsonl", "endpoint"),
        ("jsonl", "handled"),
        ("jsonl", "serialization"),
        ("sse", "endpoint"),
        ("sse", "handled"),
        ("sse", "serialization"),
        ("raw", "endpoint"),
        ("raw", "handled"),
    ],
)
def test_stream_operation_errors(
    telemetry: Any, kind: str, failure: str, sync: bool
) -> None:
    config, exporter, _ = telemetry

    def produce() -> Iterator[Any]:
        yield _valid_or_chunk(kind)
        if failure == "serialization":
            yield _invalid(kind)
        elif failure == "endpoint":
            raise ValueError("stream failed")
        else:
            assert failure == "handled"
            raise HTTPException(status_code=400)

    if sync:

        def endpoint() -> Iterable[Item]:
            yield from produce()

    else:

        async def endpoint() -> AsyncIterable[Item]:
            for item in produce():
                yield item

    app = FastAPI(telemetry=config)

    async def dependency() -> None:
        return None

    _mount(app, endpoint, kind, dependency)
    with pytest.raises(BaseException) as caught:
        TestClient(app).get("/")
    error = _stream_failure(caught.value)
    if failure == "serialization":
        assert isinstance(error, ResponseValidationError)
    elif failure == "endpoint":
        assert isinstance(error, ValueError)
    else:
        assert failure == "handled"
        cause = error.__cause__ if isinstance(error, RuntimeError) else error
        assert isinstance(cause, HTTPException)
    spans = exporter.get_finished_spans()
    (endpoint_span,) = [span for span in spans if span.name == "fastapi.endpoint"]
    serializations = [span for span in spans if span.name == "fastapi.serialization"]
    if failure == "serialization":
        assert len(serializations) == 2
        assert serializations[0].status.status_code == StatusCode.UNSET
        failed = serializations[1]
        assert failed.status.status_code == StatusCode.ERROR
        assert failed.attributes["error.type"] == (
            "fastapi.exceptions.ResponseValidationError"
        )
        assert endpoint_span.status.status_code == StatusCode.UNSET
    elif failure == "handled":
        assert endpoint_span.status.status_code == StatusCode.UNSET
        assert "error.type" not in (endpoint_span.attributes or {})
        assert len(serializations) == (0 if kind == "raw" else 1)
    else:
        assert failure == "endpoint"
        assert endpoint_span.status.status_code == StatusCode.ERROR
        assert endpoint_span.attributes["error.type"] == "ValueError"
        assert len(serializations) == (0 if kind == "raw" else 1)
        assert endpoint_span.status.status_code == StatusCode.ERROR
    if serializations and failure != "serialization":
        assert serializations[0].status.status_code == StatusCode.UNSET
    assert all(not span.events for span in spans)


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("setting", ["operation_spans", "tracing", "exclude"])
def test_stream_respects_telemetry_settings(
    telemetry: Any, setting: str, sync: bool
) -> None:
    config, exporter, _ = telemetry
    app_config: Any = dict(config)
    if setting == "operation_spans":
        app_config["operation_spans"] = False
    elif setting == "tracing":
        app_config["tracing"] = False
    else:
        assert setting == "exclude"
        app_config["exclude"] = lambda scope: scope["path"] == "/"

    def produce() -> Iterator[dict[str, int]]:
        yield {"n": 1}
        yield {"n": 2}

    if sync:

        def endpoint() -> Iterator[dict[str, int]]:
            yield from produce()

    else:

        async def endpoint() -> AsyncIterable[dict[str, int]]:
            for item in produce():
                yield item

    app = FastAPI(telemetry=app_config)
    app.get("/")(endpoint)
    response = TestClient(app).get("/")
    assert [json.loads(line) for line in response.text.splitlines()] == [
        {"n": 1},
        {"n": 2},
    ]
    names = {span.name for span in exporter.get_finished_spans()}
    assert "fastapi.endpoint" not in names
    assert "fastapi.serialization" not in names
    if setting == "operation_spans":
        assert names == {"GET /"}
    else:
        assert names == set()


@pytest.mark.parametrize("sync", [False, True])
def test_sse_keepalive_is_not_an_operation_span(
    telemetry: Any, monkeypatch: pytest.MonkeyPatch, sync: bool
) -> None:
    config, exporter, _ = telemetry
    seen: list[str] = []
    real_fail_after = fastapi.routing.anyio.fail_after

    @contextmanager
    def spy(delay: float) -> Iterator[None]:
        span = trace.get_current_span()
        assert isinstance(span, Span)
        seen.append(span.name)
        with real_fail_after(delay):
            yield

    monkeypatch.setattr(fastapi.routing.anyio, "fail_after", spy)
    monkeypatch.setattr(fastapi.routing, "_PING_INTERVAL", 0.05)

    if sync:

        def endpoint() -> Iterator[dict[str, int]]:
            yield {"n": 1}
            time.sleep(0.3)
            yield {"n": 2}

    else:

        async def endpoint() -> AsyncIterable[dict[str, int]]:
            yield {"n": 1}
            await anyio.sleep(0.3)
            yield {"n": 2}

    app = FastAPI(telemetry=config)
    app.get("/", response_class=EventSourceResponse)(endpoint)
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert ": ping" in response.text
    assert (
        len([line for line in response.text.splitlines() if line.startswith("data:")])
        == 2
    )
    spans = exporter.get_finished_spans()
    assert len([span for span in spans if span.name == "fastapi.endpoint"]) == 1
    assert len([span for span in spans if span.name == "fastapi.serialization"]) == 2
    assert seen
    assert set(seen) == {"GET /"}
