"""Section-aware chunking (spec §3.5).

Sections are split first; long sections are packed into ~target-token chunks
with ~overlap-token overlap. Each chunk is prefixed with "{title} > {section}"
so the context travels into both the embedding and the BM25 index.

The token counter is injected so the logic is testable without loading the
embedding model; the indexer passes the embedding model's tokenizer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

TokenCounter = Callable[[str], int]
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    section: str
    ord: int
    text: str  # prefixed text (what gets embedded + indexed)
    token_count: int
    domain: str


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        for sent in _SENT_SPLIT.split(para):
            sent = sent.strip()
            if sent:
                out.append(sent)
    return out


def _prefix(title: str, section: str) -> str:
    return f"{title} > {section}" if section else title


def chunk_document(
    title: str,
    domain: str,
    sections: list[dict],
    count_tokens: TokenCounter,
    *,
    target: int = 400,
    overlap: int = 60,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordn = 0
    for sec in sections:
        sents = split_sentences(sec.get("text", ""))
        if not sents:
            continue
        prefix = _prefix(title, sec.get("title", ""))
        toks = [count_tokens(s) for s in sents]
        cur: list[tuple[str, int]] = []
        cur_tok = 0
        for sent, st in zip(sents, toks):
            if cur and cur_tok + st > target:
                chunks.append(_emit(prefix, sec.get("title", ""), cur, ordn, domain, count_tokens))
                ordn += 1
                cur, cur_tok = _overlap_carry(cur, overlap)
            cur.append((sent, st))
            cur_tok += st
        if cur:
            chunks.append(_emit(prefix, sec.get("title", ""), cur, ordn, domain, count_tokens))
            ordn += 1
    return chunks


def _emit(prefix, section, cur, ordn, domain, count_tokens) -> Chunk:
    body = " ".join(s for s, _ in cur)
    text = f"{prefix}\n\n{body}"
    return Chunk(section=section, ord=ordn, text=text, token_count=count_tokens(text), domain=domain)


def _overlap_carry(cur: list[tuple[str, int]], overlap: int) -> tuple[list[tuple[str, int]], int]:
    """Seed the next chunk with trailing sentences up to `overlap` tokens."""
    carry: list[tuple[str, int]] = []
    total = 0
    for sent, st in reversed(cur):
        if total + st > overlap and carry:
            break
        carry.insert(0, (sent, st))
        total += st
    return carry, total
