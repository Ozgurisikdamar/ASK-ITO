"""MediaWiki Action API client (spec §3.1).

Uses the Action API exactly as specified:
  * text:  prop=extracts&explaintext=1
  * links: prop=links&plnamespace=0&pllimit=max (paginated via `continue`)

Policy: a descriptive User-Agent, `maxlag`, a small inter-request delay, and
exponential backoff that honours the `Retry-After` header on 429/5xx. On a
heavily throttled network the client slows down rather than failing hard.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import settings

API_URL = "https://en.wikipedia.org/w/api.php"


def _curl_available() -> bool:
    return shutil.which("curl") is not None


@dataclass
class PageResult:
    requested_title: str
    pageid: int | None = None
    revid: int | None = None
    title: str | None = None
    url: str | None = None
    extract: str | None = None
    categories: list[str] = field(default_factory=list)
    is_disambiguation: bool = False
    missing: bool = False
    redirected_from: str | None = None


class WikiError(RuntimeError):
    pass


class WikiClient:
    def __init__(
        self,
        user_agent: str | None = None,
        *,
        delay: float = 1.0,
        maxlag: int = 5,
        max_retries: int = 5,
        max_backoff: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        backend: str | None = None,
    ) -> None:
        self.user_agent = user_agent or settings.wiki_user_agent
        self.delay = delay
        self.maxlag = maxlag
        self.max_retries = max_retries
        self.max_backoff = max_backoff
        self.backend = backend or settings.wiki_http_backend  # auto | httpx | curl
        self._client = httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=60.0,
            follow_redirects=True,
            transport=transport,
        )
        self.request_count = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WikiClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- low level ---------------------------------------------------------
    def _raw_request(self, params: dict[str, Any]) -> httpx.Response:
        """Perform one GET via the selected backend, with auto curl fallback."""
        if self.backend == "curl":
            return self._curl_get(params)
        resp = self._client.get(API_URL, params=params)
        if resp.status_code == 403 and self.backend == "auto" and _curl_available():
            # egress is fingerprint-blocked; switch to curl for the rest of the run
            self.backend = "curl"
            return self._curl_get(params)
        return resp

    def _curl_get(self, params: dict[str, Any]) -> httpx.Response:
        url = str(httpx.URL(API_URL, params=params))
        proc = subprocess.run(
            [
                "curl", "-s", "-A", self.user_agent, "--max-time", "90",
                "-w", "\n__HTTP__%{http_code}\n__RA__%header{retry-after}",
                url,
            ],
            capture_output=True, text=True,
        )
        out = proc.stdout
        status, retry_after = 0, None
        body = out
        if "__HTTP__" in out:
            body, _, tail = out.rpartition("\n__HTTP__")
            lines = tail.splitlines()
            try:
                status = int(lines[0])
            except (ValueError, IndexError):
                status = 0
            for ln in lines[1:]:
                if ln.startswith("__RA__"):
                    ra = ln[len("__RA__"):].strip()
                    retry_after = ra or None
        headers = {"Retry-After": retry_after} if retry_after else {}
        return httpx.Response(status or 599, content=body.encode("utf-8"), headers=headers)

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "format": "json", "maxlag": self.maxlag}
        attempt = 0
        while True:
            time.sleep(self.delay)
            resp = self._raw_request(params)
            self.request_count += 1
            if resp.status_code == 200:
                # maxlag errors still come back 200 with an error body
                try:
                    data = resp.json()
                except ValueError:
                    data = None
                if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
                    wait = _retry_after(resp, default=5.0)
                    self._backoff(attempt, wait)
                    attempt += 1
                    if attempt > self.max_retries:
                        raise WikiError("maxlag: exceeded retries")
                    continue
                if data is None:
                    raise WikiError("non-JSON 200 response")
                return data
            if resp.status_code in (429, 503) or 500 <= resp.status_code < 600:
                wait = _retry_after(resp, default=min(2 ** attempt, self.max_backoff))
                attempt += 1
                if attempt > self.max_retries:
                    raise WikiError(f"HTTP {resp.status_code}: exceeded retries")
                self._backoff(attempt, wait)
                continue
            raise WikiError(f"HTTP {resp.status_code}")

    def _backoff(self, attempt: int, wait: float) -> None:
        wait = min(max(wait, 2 ** attempt), self.max_backoff)
        time.sleep(wait)

    # -- high level --------------------------------------------------------
    def fetch_page(self, title: str) -> PageResult:
        """Resolve redirects and fetch full plain-text extract + metadata."""
        data = self._get(
            {
                "action": "query",
                "redirects": 1,
                "titles": title,
                "prop": "extracts|info|categories|pageprops",
                "explaintext": 1,
                "inprop": "url",
                "cllimit": "max",
                "clshow": "!hidden",
            }
        )
        query = data.get("query", {})
        result = PageResult(requested_title=title)
        for r in query.get("redirects", []):
            if r.get("to"):
                result.redirected_from = r.get("from")
        pages = query.get("pages", {})
        if not pages:
            result.missing = True
            return result
        page = next(iter(pages.values()))
        if "missing" in page:
            result.missing = True
            return result
        result.pageid = page.get("pageid")
        result.revid = page.get("lastrevid")
        result.title = page.get("title")
        result.url = page.get("fullurl")
        result.extract = page.get("extract") or ""
        result.categories = [
            c["title"].removeprefix("Category:") for c in page.get("categories", [])
        ]
        pageprops = page.get("pageprops", {})
        result.is_disambiguation = "disambiguation" in pageprops or any(
            "disambiguation" in c.lower() for c in result.categories
        )
        return result

    def fetch_links(self, title: str, limit: int = 2000) -> list[str]:
        """Namespace-0 outgoing links, paginated via `continue`."""
        links: list[str] = []
        cont: dict[str, Any] = {}
        while True:
            data = self._get(
                {
                    "action": "query",
                    "titles": title,
                    "prop": "links",
                    "plnamespace": 0,
                    "pllimit": "max",
                    **cont,
                }
            )
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                for link in page.get("links", []):
                    links.append(link["title"])
            if len(links) >= limit or "continue" not in data:
                break
            cont = data["continue"]
        return links[:limit]


def _retry_after(resp: httpx.Response, default: float) -> float:
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            return default
    return default
