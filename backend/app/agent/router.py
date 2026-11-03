"""AI Mode routing: one grounded answer call, or the multi-step agent.

Deterministic (no LLM call): most questions ask for one fact and are answered by
the fast path (hybrid retrieval + reranker + evidence cut-off + one streamed
answer call). The agent (analysis, sub-question searches, sufficiency checks)
runs only when the question visibly needs several pieces of evidence:
comparisons, several entities asked about together, relations between topics,
or several questions at once.
"""
from __future__ import annotations

import re

_COMPARISON = re.compile(
    r"\b(compare|comparison|compared|versus|vs\.?|differ(?:s|ence|ences)?|contrast|"
    r"similarit(?:y|ies)|which (?:is|was) (?:better|older|earlier|larger|faster))\b", re.I)
_MULTI_ENTITY = re.compile(r"\b(each|both|respectively|all three|between)\b", re.I)
_RELATION = re.compile(r"\b(relat(?:e|ed|ion|ionship)|connect(?:ed|ion)|link(?:ed)?|"
                       r"influenc(?:e|ed)|lead to|led to)\b", re.I)
_TWO_QUESTIONS = re.compile(r"\b(and|also) (what|who|how|why|when|which|where)\b", re.I)
_LONG_WORDS = 22


def route(query: str) -> tuple[str, str]:
    """("fast" | "agent", reason)."""
    q = " ".join(query.split())
    if q.count("?") >= 2 or _TWO_QUESTIONS.search(q):
        return "agent", "several questions"
    if _COMPARISON.search(q):
        return "agent", "comparison"
    if _RELATION.search(q):
        return "agent", "relation between topics"
    if _MULTI_ENTITY.search(q) and re.search(r"\band\b", q, re.I):
        return "agent", "several entities"
    if len(q.split()) > _LONG_WORDS:
        return "agent", "long, multi-part question"
    return "fast", "single question"
