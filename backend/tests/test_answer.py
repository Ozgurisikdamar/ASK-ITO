from app.agent.answer import Source, build_context, validate_citations
from app.retrieval.hydrate import ChunkInfo


def _src(n, cid, title, section, text):
    return Source(n=n, chunk_id=cid, info=ChunkInfo(cid, cid, title, section, "u", "ai", text))


def test_validate_citations_keeps_valid_strips_invalid():
    ans = "Hinton won in 2024 [1]. Also relevant [2]. Bogus [7] and [0]."
    cleaned, used = validate_citations(ans, n_sources=2)
    assert used == [1, 2]
    assert "[7]" not in cleaned and "[0]" not in cleaned
    assert "[1]" in cleaned and "[2]" in cleaned


def test_validate_citations_none():
    cleaned, used = validate_citations("No citations here.", n_sources=3)
    assert used == []
    assert cleaned == "No citations here."


def test_build_context_numbering_and_prefix_stripped():
    srcs = [
        _src(1, 10, "Deep learning", "History", "Deep learning > History\n\nIt began earlier."),
        _src(2, 11, "LLM", "", "LLM\n\nA large language model."),
    ]
    ctx = build_context(srcs)
    assert ctx.startswith("[1] Deep learning > History\nIt began earlier.")
    assert "[2] LLM\nA large language model." in ctx
    # prefix line must not leak into the body
    assert "Deep learning > History\n\nDeep learning" not in ctx
