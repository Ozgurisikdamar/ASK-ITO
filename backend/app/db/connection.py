"""Shared psycopg (v3) connection pool for request-path queries.

Batch jobs (migrate, ingest, index, eval) still open their own connections;
the API uses this pool so a search does not pay a TCP + auth handshake per
query. Connections are autocommit; use `conn.transaction()` when a statement
needs `SET LOCAL`.
"""
from __future__ import annotations

import atexit
import threading
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool

from app.config import settings

_pool: ConnectionPool | None = None
_lock = threading.Lock()


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                _pool = ConnectionPool(
                    settings.database_url,
                    min_size=settings.db_pool_min,
                    max_size=settings.db_pool_max,
                    kwargs={"autocommit": True, "connect_timeout": settings.db_connect_timeout_s},
                    timeout=settings.db_connect_timeout_s * 2,
                    open=True,
                )
                atexit.register(close_pool)
    return _pool


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    with pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None
