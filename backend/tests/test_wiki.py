"""WikiClient parsing + Retry-After backoff, using a mocked transport (no net)."""
import httpx
import pytest

from app.ingest.wiki import WikiClient

PAGE_JSON = {
    "query": {
        "redirects": [{"from": "Ligo", "to": "LIGO"}],
        "pages": {
            "81610": {
                "pageid": 81610,
                "lastrevid": 123456,
                "title": "LIGO",
                "fullurl": "https://en.wikipedia.org/wiki/LIGO",
                "extract": "LIGO detects gravitational waves.\n\n== History ==\nBuilt in 1994.",
                "categories": [{"title": "Category:Gravitational-wave telescopes"}],
                "pageprops": {},
            }
        },
    }
}

DISAMB_JSON = {
    "query": {
        "pages": {
            "999": {
                "pageid": 999,
                "lastrevid": 1,
                "title": "Richard Henderson",
                "fullurl": "u",
                "extract": "x",
                "pageprops": {"disambiguation": ""},
            }
        }
    }
}


def test_fetch_page_parses(monkeypatch):
    monkeypatch.setattr("app.ingest.wiki.time.sleep", lambda *_: None)
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=PAGE_JSON))
    with WikiClient(delay=0, transport=transport) as c:
        page = c.fetch_page("Ligo")
    assert page.pageid == 81610
    assert page.revid == 123456
    assert page.title == "LIGO"
    assert page.redirected_from == "Ligo"
    assert "gravitational waves" in page.extract
    assert page.categories == ["Gravitational-wave telescopes"]
    assert page.is_disambiguation is False


def test_disambiguation_detected(monkeypatch):
    monkeypatch.setattr("app.ingest.wiki.time.sleep", lambda *_: None)
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=DISAMB_JSON))
    with WikiClient(delay=0, transport=transport) as c:
        page = c.fetch_page("Richard Henderson")
    assert page.is_disambiguation is True


def test_retry_after_backoff(monkeypatch):
    sleeps = []
    monkeypatch.setattr("app.ingest.wiki.time.sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, text="slow down")
        return httpx.Response(200, json=PAGE_JSON)

    transport = httpx.MockTransport(handler)
    with WikiClient(delay=0, transport=transport) as c:
        page = c.fetch_page("LIGO")
    assert page.pageid == 81610
    assert calls["n"] == 2  # retried once
    assert any(s >= 7 for s in sleeps)  # honoured Retry-After


def test_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr("app.ingest.wiki.time.sleep", lambda *_: None)
    transport = httpx.MockTransport(lambda req: httpx.Response(429, text="no"))
    with WikiClient(delay=0, max_retries=2, transport=transport) as c:
        with pytest.raises(Exception):
            c.fetch_page("LIGO")
