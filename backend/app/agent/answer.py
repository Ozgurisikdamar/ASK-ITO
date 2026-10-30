"""Answer generation + citation validation (spec §6 steps 6-7).

Shared by the retrieval modes (M4, single shot) and the agentic loop (M5).
The answer is produced from the numbered context only; every [n] citation is
validated against the available sources and invalid ones are stripped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.agent.prompts import load
from app.llm.client import LLMClient, get_llm_client
from app.retrieval.hydrate import ChunkInfo, body_without_prefix

_CITE = re.compile(r"\[(\d+)\]")


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


def build_context(sources: list[Source], max_chars: int = 900) -> str:
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


def generate_answer(
    query: str, sources: list[Source], llm: LLMClient | None = None, temperature: float = 0.0
) -> tuple[str, list[dict]]:
    llm = llm or get_llm_client()
    if not sources:
        prompt = load("closed_book").replace("{question}", query)
        answer = llm.chat([{"role": "user", "content": prompt}], temperature=temperature)
        return answer.strip(), []

    context = build_context(sources)
    prompt = load("answer").replace("{context}", context).replace("{question}", query)
    raw = llm.chat([{"role": "user", "content": prompt}], temperature=temperature)
    cleaned, used = validate_citations(raw, len(sources))
    citations = [{"n": s.n, "chunk_id": s.chunk_id} for s in sources if s.n in used]
    return cleaned, citations
