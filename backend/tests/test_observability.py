"""Request ids: response header, SSE rid, log lines."""
from __future__ import annotations

import io
import json
import logging
import re

from fastapi.testclient import TestClient

from app import observability
from app.main import app


def test_every_response_has_a_request_id():
    r = TestClient(app).get("/")
    assert re.fullmatch(r"[0-9a-f]{12}", r.headers["x-request-id"])


def test_a_safe_incoming_id_is_kept_and_an_unsafe_one_replaced():
    c = TestClient(app)
    assert c.get("/", headers={"X-Request-ID": "trace-42.a_b"}).headers["x-request-id"] \
        == "trace-42.a_b"
    for bad in ("has space", "x" * 65, "semi;colon", "<script>"):
        rid = c.get("/", headers={"X-Request-ID": bad}).headers["x-request-id"]
        assert rid != bad and re.fullmatch(r"[0-9a-f]{12}", rid)


def test_stream_events_carry_the_header_id(monkeypatch):
    r = TestClient(app).get("/api/search/stream", params={"q": "q", "mode": "nope"},
                            headers={"X-Request-ID": "abc123"})
    assert r.headers["x-request-id"] == "abc123"  # also on a bad request

    seen = []

    def produce(q, mode, domain, cancel):
        yield "results", {"results": [], "total": 0}
        yield "done", {"answer": ""}

    from app.api import stream as stream_api

    monkeypatch.setattr(stream_api, "_produce", produce)
    r = TestClient(app).get("/api/search/stream", params={"q": "qubits", "mode": "search"},
                            headers={"X-Request-ID": "abc123"})
    for line in r.text.splitlines():
        if line.startswith("data: "):
            seen.append(json.loads(line[6:]).get("rid"))
    assert seen and set(seen) == {"abc123"}


def test_log_lines_carry_the_request_id():
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.addFilter(observability.RequestIdFilter())
    handler.setFormatter(logging.Formatter(observability._FORMAT))
    log = logging.getLogger("askito.test_observability")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        token = observability.request_id.set("rid-1")
        log.info("inside a request")
        observability.request_id.reset(token)
        log.info("outside")
    finally:
        log.removeHandler(handler)
    lines = buf.getvalue().splitlines()
    assert "rid=rid-1 inside a request" in lines[0] and "rid=- outside" in lines[1]


def test_setup_logging_is_idempotent():
    logger = logging.getLogger("askito")
    before = list(logger.handlers)
    try:
        observability.setup_logging("INFO")
        observability.setup_logging("INFO")
        added = [h for h in logger.handlers if h not in before]
        assert len(added) <= 1 and logger.propagate  # caplog and root handlers still see it
    finally:
        for h in [h for h in logger.handlers if h not in before]:
            logger.removeHandler(h)
