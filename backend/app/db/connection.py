"""Thin psycopg (v3) connection helper.

M1 keeps this minimal: a short-lived connection per call. A pool can be added
later if contention appears.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg

from app.config import settings


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(settings.database_url)
    try:
        yield conn
    finally:
        conn.close()
