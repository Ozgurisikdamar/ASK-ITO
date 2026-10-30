from app.agent.rewrite import _parse


def test_parse_json_query():
    assert _parse('{"query": "2024 nobel physics neural networks"}') == \
        "2024 nobel physics neural networks"


def test_parse_json_embedded_in_text():
    raw = 'Here you go:\n{"query": "quantum error correction"}\nThanks'
    assert _parse(raw) == "quantum error correction"


def test_parse_bad_returns_none():
    assert _parse("no json at all") is None
    assert _parse('{"notquery": "x"}') is None
    assert _parse('{"query": ""}') is None
