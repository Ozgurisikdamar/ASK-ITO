"""Generator mechanics with a scripted LLM (skip if the index/DB is absent)."""
import psycopg
import pytest

from app.config import settings
from app.eval import testset, testset_draft
from app.index import bm25_index


def _index_ready() -> bool:
    if not bm25_index.exists():
        return False
    try:
        with psycopg.connect(settings.database_url) as c:
            return c.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _index_ready(), reason="index/DB not built")


class ScriptedLLM:
    def ping(self):
        return True, "scripted"

    def chat(self, messages, temperature=0.0, model=None, **kwargs):
        p = messages[-1]["content"]
        if "unanswerable" in p.lower():
            return '{"question": "An obscure unanswerable question?"}'
        return '{"question": "A generated question?", "reference": "A reference answer."}'


def test_generator_produces_drafts_with_gold(tmp_path, monkeypatch):
    monkeypatch.setattr(testset, "DRAFT", tmp_path / "draft.jsonl")
    summary = testset_draft.run(total=12, seed=1, llm=ScriptedLLM())
    drafts = testset.read_jsonl(tmp_path / "draft.jsonl")
    assert len(drafts) == summary["drafts"] > 0

    types = {d["qtype"] for d in drafts}
    assert "single_hop" in types and "unanswerable" in types

    for d in drafts:
        assert testset.validate(d) == [], f"invalid draft: {d}"
        if d["qtype"] == "unanswerable":
            assert d["reference"] is None and d["gold_doc_ids"] == []
        else:
            assert d["gold_doc_ids"], f"answerable draft missing gold docs: {d['id']}"
            assert d["gold_chunk_ids"]
