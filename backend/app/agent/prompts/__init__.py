"""Prompt templates loaded from .txt files (kept out of code, spec §6.1)."""
from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load(name: str) -> str:
    return (_DIR / f"{name}.txt").read_text(encoding="utf-8")
