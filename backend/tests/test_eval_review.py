"""Review endpoint logic against temp files (no DB: drafts carry no gold chunks)."""
import pytest
from fastapi.testclient import TestClient

from app.eval import testset
from app.main import app

client = TestClient(app)

DRAFTS = [
    {"id": "d1", "question": "Original question?", "reference": "ref", "qtype": "single_hop",
     "domain": "nobel", "gold_doc_ids": [1], "gold_chunk_ids": [], "source": "draft"},
    {"id": "d2", "question": "Unanswerable?", "reference": None, "qtype": "unanswerable",
     "domain": "ai", "gold_doc_ids": [], "gold_chunk_ids": [], "source": "draft"},
]


@pytest.fixture
def temp_testset(tmp_path, monkeypatch):
    monkeypatch.setattr(testset, "DRAFT", tmp_path / "draft.jsonl")
    monkeypatch.setattr(testset, "TESTSET", tmp_path / "testset.jsonl")
    monkeypatch.setattr(testset, "REVIEW_STATE", tmp_path / "review.json")
    testset.write_jsonl(testset.DRAFT, DRAFTS)
    return tmp_path


def test_get_review_pending(temp_testset):
    r = client.get("/api/eval/review")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"] == {"total": 2, "accepted": 0, "rejected": 0, "pending": 2}
    assert {i["status"] for i in body["items"]} == {"pending"}


def test_accept_with_edit_then_reject(temp_testset):
    r = client.post("/api/eval/review", json={
        "id": "d1", "action": "accept", "question": "Edited question?", "reference": "new ref"})
    assert r.status_code == 200 and r.json()["status"] == "accepted"

    saved = {q["id"]: q for q in testset.read_jsonl(testset.TESTSET)}
    assert saved["d1"]["question"] == "Edited question?"
    assert saved["d1"]["reference"] == "new ref"
    assert "source" not in saved["d1"]  # stripped

    r2 = client.post("/api/eval/review", json={"id": "d2", "action": "reject"})
    assert r2.json()["status"] == "rejected"

    body = client.get("/api/eval/review").json()
    assert body["summary"]["accepted"] == 1
    assert body["summary"]["rejected"] == 1


def test_reject_invalid_action(temp_testset):
    assert client.post("/api/eval/review", json={"id": "d1", "action": "maybe"}).status_code == 400
    assert client.post("/api/eval/review", json={"id": "nope", "action": "accept"}).status_code == 404
