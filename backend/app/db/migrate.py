"""Minimal migration runner.

Applies numbered .sql files in app/db/migrations/ in order and records applied
versions in a schema_migrations table. Re-running is safe (idempotent):
already-applied files are skipped. The __EMBEDDING_DIM__ placeholder is
substituted with the configured embedding dimension.
"""
from __future__ import annotations

import sys
from pathlib import Path

import psycopg

from app.config import settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _applied_versions(conn: psycopg.Connection) -> set[str]:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    conn.commit()
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {r[0] for r in rows}


def run() -> int:
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        print("No migration files found.")
        return 0

    applied_count = 0
    with psycopg.connect(settings.database_url) as conn:
        applied = _applied_versions(conn)
        for path in files:
            version = path.stem
            if version in applied:
                print(f"skip  {version} (already applied)")
                continue
            sql = path.read_text(encoding="utf-8").replace(
                "__EMBEDDING_DIM__", str(settings.embedding_dim)
            )
            print(f"apply {version} ...", end=" ")
            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
            )
            conn.commit()
            print("ok")
            applied_count += 1

    print(f"Done. {applied_count} migration(s) applied.")
    return applied_count


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        print(f"Migration failed: {exc}", file=sys.stderr)
        sys.exit(1)
