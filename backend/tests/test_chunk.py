from app.index.chunk import chunk_document, split_sentences

# word-count token counter so the logic is testable without the model
WORDS = lambda t: len(t.split())  # noqa: E731


def test_split_sentences():
    s = split_sentences("First sentence. Second one!\n\nNew paragraph here.")
    assert s == ["First sentence.", "Second one!", "New paragraph here."]


def test_section_aware_and_prefix():
    sections = [
        {"title": "", "text": "Lead paragraph about LIGO."},
        {"title": "History", "text": "It was built. It works."},
    ]
    chunks = chunk_document("LIGO", "nobel", sections, WORDS, target=100, overlap=10)
    assert len(chunks) == 2  # one per short section
    assert chunks[0].text.startswith("LIGO\n\n")          # lead prefix = title only
    assert chunks[1].text.startswith("LIGO > History\n\n")  # section prefix
    assert [c.ord for c in chunks] == [0, 1]
    assert all(c.domain == "nobel" for c in chunks)


def test_long_section_splits_with_overlap():
    sents = [f"Sentence number {i} has some words here." for i in range(12)]
    sections = [{"title": "Body", "text": " ".join(sents)}]
    # each sentence ~6 words; target 18 -> ~3 sentences/chunk, overlap carries some back
    chunks = chunk_document("Doc", "ai", sections, WORDS, target=18, overlap=6)
    assert len(chunks) > 1
    # overlap: last sentence of a chunk reappears at the start of the next body
    def body(c):
        return c.text.split("\n\n", 1)[1]
    first_tail = body(chunks[0]).split(". ")[-1]
    assert first_tail.split()[2] in body(chunks[1])  # the sentence number recurs


def test_token_count_is_prefixed_text():
    sections = [{"title": "X", "text": "one two three."}]
    chunks = chunk_document("T", "quantum", sections, WORDS, target=100, overlap=10)
    # "T > X" (3) + body "one two three." (3) = 6 words
    assert chunks[0].token_count == WORDS(chunks[0].text) == 6
