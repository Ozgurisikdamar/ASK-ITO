"""Answer prompts: exact refusal phrase (metrics and the UI match it), and a
partial answer instead of "not found" when only part of the question is covered.

Regression (real Q8_0 run): "Compare the 2023 and 2024 Nobel Prizes in Physics"
produced "I could not find this in the sources." followed by a note with the
2023 facts, because the prompt only allowed all or nothing."""
from pathlib import Path

import pytest

from app.agent.answer import NOT_FOUND

PROMPTS = Path(__file__).resolve().parents[1] / "app" / "agent" / "prompts"


@pytest.mark.parametrize("name", ["answer.txt", "overview.txt"])
def test_answer_prompts_refuse_exactly_and_allow_partial_answers(name):
    flat = " ".join((PROMPTS / name).read_text(encoding="utf-8").split())
    assert NOT_FOUND in flat
    assert "only part of the question" in flat
    assert "Only if no passage is relevant" in flat
