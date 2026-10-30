from app.eval import testset


def test_validate_rules():
    ok = {"id": "a", "question": "q?", "qtype": "single_hop", "domain": "nobel",
          "gold_doc_ids": [1]}
    assert testset.validate(ok) == []
    assert testset.validate({**ok, "qtype": "bad"})
    # answerable needs gold_doc_ids
    assert any("gold_doc_ids" in e for e in testset.validate({**ok, "gold_doc_ids": []}))
    # unanswerable must have null reference and needs no gold
    un = {"id": "u", "question": "q?", "qtype": "unanswerable", "domain": "ai",
          "reference": None}
    assert testset.validate(un) == []
    assert testset.validate({**un, "reference": "x"})


def test_targets_distribution():
    t = testset.targets(200)
    assert t["single_hop"] == 60 and t["multi_hop"] == 50
    assert t["unanswerable"] == 10
    assert set(t) == set(testset.QTYPES)


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [{"id": "a", "x": 1}, {"id": "b", "x": 2}]
    assert testset.write_jsonl(p, rows) == 2
    assert testset.read_jsonl(p) == rows
    assert testset.read_jsonl(tmp_path / "missing.jsonl") == []
