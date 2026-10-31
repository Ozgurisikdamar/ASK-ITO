"""Answer generation + citation validation (spec §6 steps 6-7).

Shared by the retrieval modes, the streamed overview and the agentic loop.
The answer is produced from the numbered context only; every [n] citation is
validated against the available sources and invalid ones are stripped. With no
sources the answer is the fixed "not found" sentence — the model's own
knowledge is used only in the explicit `closed_book` baseline mode.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Iterator

from app.agent.prompts import load
from app.config import settings
from app.llm.client import LLMClient, get_llm_client
from app.retrieval.hydrate import ChunkInfo, body_without_prefix

_CITE = re.compile(r"\[(\d+)\]")
NOT_FOUND = "I could not find this in the sources."


@dataclass
class Source:
    n: int
    chunk_id: int
    info: ChunkInfo
    score: float = 0.0


def build_sources(final: list[tuple[int, float]], infos: dict[int, ChunkInfo]) -> list[Source]:
    sources: list[Source] = []
    n = 1
    for cid, score in final:
        info = infos.get(cid)
        if info is None:
            continue
        sources.append(Source(n=n, chunk_id=cid, info=info, score=score))
        n += 1
    return sources


def build_context(sources: list[Source], max_chars: int | None = None) -> str:
    max_chars = max_chars or settings.answer_context_chars
    blocks = []
    for s in sources:
        body = body_without_prefix(s.info.text)[:max_chars]
        header = s.info.doc_title + (f" > {s.info.section}" if s.info.section else "")
        blocks.append(f"[{s.n}] {header}\n{body}")
    return "\n\n".join(blocks)


def validate_citations(answer: str, n_sources: int) -> tuple[str, list[int]]:
    """Strip citations that point outside 1..n_sources; return cleaned answer and
    the sorted list of valid citation numbers actually used."""
    used: set[int] = set()

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_sources:
            used.add(n)
            return m.group(0)
        return ""

    cleaned = _CITE.sub(repl, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, sorted(used)


def _messages(query: str, sources: list[Source], mode: str) -> tuple[list[dict], int]:
    if mode == "closed_book":
        prompt = load("closed_book").replace("{question}", query)
        return [{"role": "user", "content": prompt}], settings.answer_max_tokens
    if mode == "overview":
        context = build_context(sources, settings.overview_context_chars)
        prompt = load("overview").replace("{context}", context).replace("{question}", query)
        return [{"role": "user", "content": prompt}], settings.overview_max_tokens
    context = build_context(sources)
    prompt = load("answer").replace("{context}", context).replace("{question}", query)
    return [{"role": "user", "content": prompt}], settings.answer_max_tokens


def _finish(raw: str, sources: list[Source], mode: str) -> tuple[str, list[dict]]:
    if mode == "closed_book":
        return raw.strip(), []
    cleaned, used = validate_citations(raw, len(sources))
    citations = [{"n": s.n, "chunk_id": s.chunk_id} for s in sources if s.n in used]
    return cleaned or NOT_FOUND, citations


def generate_answer(
    query: str, sources: list[Source], llm: LLMClient | None = None,
    temperature: float = 0.0, mode: str = "hybrid_rerank",
) -> tuple[str, list[dict]]:
    if mode != "closed_book" and not sources:
        return NOT_FOUND, []
    llm = llm or get_llm_client()
    messages, max_tokens = _messages(query, sources, mode)
    raw = llm.chat(messages, temperature=temperature,
                   model=settings.model_for("answer"), max_tokens=max_tokens)
    return _finish(raw, sources, mode)


def stream_answer(
    query: str, sources: list[Source], llm: LLMClient | None = None,
    temperature: float = 0.0, mode: str = "overview", cancel: threading.Event | None = None,
) -> Iterator[tuple[str, object]]:
    """Yield ("token", text) pieces, then ("done", {"answer", "citations"}).

    Tokens are the raw model output (citations unvalidated); the "done" payload
    carries the validated answer that the client should display finally.
    """
    if mode != "closed_book" and not sources:
        yield "token", NOT_FOUND
        yield "done", {"answer": NOT_FOUND, "citations": []}
        return
    llm = llm or get_llm_client()
    messages, max_tokens = _messages(query, sources, mode)
    parts: list[str] = []
    for piece in llm.stream_chat(messages, temperature=temperature,
                                 model=settings.model_for("answer"),
                                 max_tokens=max_tokens, cancel=cancel):
        parts.append(piece)
        yield "token", piece
    if cancel is not None and cancel.is_set():
        return
    answer, citations = _finish("".join(parts), sources, mode)
    yield "done", {"answer": answer, "citations": citations}
