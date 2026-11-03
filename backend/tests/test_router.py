"""AI Mode routing: one grounded answer for simple questions, the agent for
comparisons / relations / multi-part questions (deterministic, no LLM call)."""
import json

import pytest
from fastapi.testclient import TestClient

from app.agent.router import route
from app.api import stream as stream_api
from app.main import app


@pytest.mark.parametrize("q", [
    "What is retrieval-augmented generation?", "Transformers", "quantum computing",
    "What is a transmon qubit?", "Willow processor 105 physical qubits",
    "Hopfield and Hinton, 2024 Physics prize",
    "Which organisation received the 2024 Nobel Peace Prize?",
])
def test_simple_questions_take_the_fast_path(q):
    assert route(q)[0] == "fast"


@pytest.mark.parametrize("q,reason", [
    ("How do superconducting qubits compare with trapped-ion qubits?", "comparison"),
    ("How does retrieval-augmented generation differ from fine-tuning a language model?",
     "comparison"),
    ("How is the 2025 Physics prize related to quantum computers?", "relation between topics"),
    ("What did Hopfield and Hinton each contribute to the 2024 Nobel Prize in Physics?",
     "several entities"),
    ("Who won the 2023 Nobel Prize in Physics and what is attosecond physics?",
     "several questions"),
    ("What is a qubit? How is it measured?", "several questions"),
])
def test_multi_part_questions_take_the_agent(q, reason):
    assert route(q) == ("agent", reason)


def _events(text):
    out = []
    for block in text.split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln and not ln.startswith(":")]
        if lines:
            ev = next(ln[7:] for ln in lines if ln.startswith("event: "))
            out.append((ev, json.loads(next(ln[6:] for ln in lines if ln.startswith("data: ")))))
    return out


def test_ai_mode_streams_the_route_and_labels_done(monkeypatch):
    def fake_agentic(q, domain, cancel, t0):
        yield "results", {"results": [], "total": 0}
        yield "done", {"answer": "agent answer [1].", "citations": [], "mode": "agentic"}

    monkeypatch.setattr(stream_api, "_produce_agentic", fake_agentic)
    ev = _events(TestClient(app).get(
        "/api/search/stream",
        params={"q": "How do transmons compare with trapped ions?", "mode": "ai"}).text)
    kinds = [e for e, _ in ev]
    assert kinds[0] == "status" and ev[0][1]["route"] == "agent"
    done = ev[-1][1]
    assert kinds[-1] == "done" and done["mode"] == "ai" and done["route"] == "agent"
    assert done["route_reason"] == "comparison" and done["answer"] == "agent answer [1]."
