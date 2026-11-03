"""Optional local Redis cache: the shared L2 tier behind the in-process caches.

Used by the query-embedding cache, the retrieval cache, a front cache for the
PostgreSQL answer cache and the agent's deterministic tool calls. It is off by
default (REDIS_ENABLED=false) and only ever talks to a local Redis: loopback, or
a container hostname listed in REDIS_ALLOWED_HOSTS that resolves to a private
address (docker-compose: `redis`). Hosted Redis is refused.

Failure policy: if Redis is unreachable or errors, one warning is logged, the
circuit opens for REDIS_RETRY_S and every call behaves like a miss / no-op, so
callers fall back to the in-process LRU and PostgreSQL. REDIS_STRICT=true turns
that into an error instead.

Values are JSON or raw bytes (never pickle). Keys are namespaced and hashed:
`askito:v1:<kind>:<sha256>`; versioning lives in the key material (index,
model, prompt and retrieval settings), and every entry has a TTL. Model KV
tensors are never stored here: the KV / prefix cache belongs to Ollama.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from urllib.parse import urlsplit

from app.config import settings
from app.llm.policy import RemoteEndpointError, allowed_hosts_from_setting, ensure_local_url

log = logging.getLogger("askito.redis")

PREFIX = "askito:v1"
KINDS = ("emb", "ret", "ans", "tool")


class RedisUnavailable(RuntimeError):
    """Raised only with REDIS_STRICT=true."""


def ttl_for(kind: str) -> int:
    return {
        "emb": settings.redis_ttl_embedding_s,
        "ret": settings.redis_ttl_retrieval_s,
        "ans": settings.redis_ttl_answer_s,
        "tool": settings.redis_ttl_tool_s,
    }[kind]


def hash_key(material) -> str:
    """Stable digest of the key material (any repr-able value)."""
    raw = material if isinstance(material, str) else repr(material)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def check_local_redis_url(url: str, allowed_hosts=(), resolver=None) -> None:
    """Raise RemoteEndpointError unless REDIS_URL points at a local Redis."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise RemoteEndpointError(f"REDIS_URL: malformed URL ({exc})") from exc
    if parts.scheme == "unix":
        return  # a local socket file
    if parts.scheme not in ("redis", "rediss"):
        raise RemoteEndpointError(
            f"REDIS_URL: scheme must be redis, rediss or unix, got {parts.scheme!r}")
    host = parts.hostname or ""
    netloc = f"[{host}]" if ":" in host else host
    probe = f"http://{netloc}" + (f":{port}" if port else "")
    kw = {"resolver": resolver} if resolver else {}
    try:  # same rules as the LLM endpoint: loopback or an explicitly allowed host
        ensure_local_url(probe, "REDIS_URL", allowed_hosts=allowed_hosts, **kw)
    except RemoteEndpointError as exc:
        raise RemoteEndpointError(
            f"{exc} (Redis must be local: loopback, or a container hostname listed "
            "in REDIS_ALLOWED_HOSTS)") from exc


class RedisCache:
    def __init__(self, client_factory=None) -> None:
        self._factory = client_factory
        self._client = None
        self._lock = threading.Lock()
        self._down_until = 0.0
        self._warned = False
        self.last_error: str | None = None
        self.counters: dict[str, dict[str, int]] = defaultdict(
            lambda: {"hits": 0, "misses": 0, "sets": 0, "errors": 0, "too_large": 0})

    # -- connection ---------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return settings.redis_enabled

    def _make_client(self):
        if self._factory is not None:
            return self._factory()
        check_local_redis_url(settings.redis_url,
                              allowed_hosts_from_setting(settings.redis_allowed_hosts))
        import redis  # optional dependency, imported only when enabled

        return redis.Redis.from_url(
            settings.redis_url, socket_timeout=settings.redis_timeout_s,
            socket_connect_timeout=settings.redis_timeout_s, decode_responses=False,
            health_check_interval=0)

    def _get_client(self):
        if not self.enabled:
            return None
        if time.monotonic() < self._down_until:
            if settings.redis_strict:
                raise RedisUnavailable(self.last_error or "redis unavailable")
            return None
        with self._lock:
            if self._client is None:
                try:
                    self._client = self._make_client()
                except Exception as exc:  # noqa: BLE001  (policy error or import error)
                    self._fail("connect", exc)
                    return None
            return self._client

    def _fail(self, op: str, exc: Exception, kind: str | None = None) -> None:
        self.last_error = f"{op}: {exc.__class__.__name__}: {exc}"
        if kind:
            self.counters[kind]["errors"] += 1
        self._down_until = time.monotonic() + settings.redis_retry_s
        self._client = None if isinstance(exc, RemoteEndpointError) else self._client
        if settings.redis_strict:
            raise RedisUnavailable(self.last_error) from exc
        if not self._warned:
            self._warned = True
            log.warning("redis cache unavailable (%s); falling back to the in-process "
                        "and PostgreSQL caches for %ss", self.last_error,
                        settings.redis_retry_s)

    def reset(self) -> None:
        """Forget the client and circuit state (tests, settings changes)."""
        with self._lock:
            self._client = None
            self._down_until = 0.0
            self._warned = False
            self.last_error = None
            self.counters.clear()

    @staticmethod
    def full_key(kind: str, key: str) -> str:
        return f"{PREFIX}:{kind}:{key}"

    # -- raw bytes ------------------------------------------------------------
    def get_bytes(self, kind: str, key: str) -> bytes | None:
        client = self._get_client()
        if client is None:
            return None
        try:
            value = client.get(self.full_key(kind, key))
        except Exception as exc:  # noqa: BLE001
            self._fail("get", exc, kind)
            return None
        self.counters[kind]["hits" if value is not None else "misses"] += 1
        return value

    def set_bytes(self, kind: str, key: str, value: bytes, ttl: int | None = None) -> bool:
        client = self._get_client()
        if client is None:
            return False
        if len(value) > settings.redis_max_value_bytes:
            self.counters[kind]["too_large"] += 1
            return False
        try:
            client.set(self.full_key(kind, key), value, ex=max(1, ttl or ttl_for(kind)))
        except Exception as exc:  # noqa: BLE001
            self._fail("set", exc, kind)
            return False
        self.counters[kind]["sets"] += 1
        return True

    # -- JSON -----------------------------------------------------------------
    def get_json(self, kind: str, key: str):
        raw = self.get_bytes(kind, key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def set_json(self, kind: str, key: str, value, ttl: int | None = None) -> bool:
        return self.set_bytes(kind, key, json.dumps(value, separators=(",", ":")).encode(),
                              ttl)

    # -- maintenance ----------------------------------------------------------
    def clear(self, kind: str | None = None) -> int:
        """Delete this app's keys (one kind or all). Returns the number removed."""
        client = self._get_client()
        if client is None:
            return 0
        pattern = f"{PREFIX}:{kind}:*" if kind else f"{PREFIX}:*"
        n = 0
        try:
            batch = []
            for k in client.scan_iter(match=pattern, count=500):
                batch.append(k)
                if len(batch) >= 500:
                    n += client.delete(*batch)
                    batch = []
            if batch:
                n += client.delete(*batch)
        except Exception as exc:  # noqa: BLE001
            self._fail("clear", exc, kind)
        return n

    def ping(self) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            return bool(client.ping())
        except Exception as exc:  # noqa: BLE001
            self._fail("ping", exc)
            return False

    def status(self) -> dict:
        host = urlsplit(settings.redis_url).hostname if settings.redis_enabled else None
        return {
            "enabled": settings.redis_enabled,
            "available": self.ping() if settings.redis_enabled else False,
            "host": host,
            "strict": settings.redis_strict,
            "last_error": self.last_error.split(":", 2)[1].strip() if self.last_error
            else None,
            "counters": {k: dict(v) for k, v in self.counters.items()},
        }


cache = RedisCache()
