from app.ingest import expand


def test_title_filters():
    assert expand.title_filter_reason("List of Nobel laureates") == "list/index page"
    assert expand.title_filter_reason("2020") == "year/date page"
    assert expand.title_filter_reason("2020 in science") == "year/date page"
    assert expand.title_filter_reason("21st century") == "year/date page"
    assert expand.title_filter_reason("January 5") == "year/date page"
    assert expand.title_filter_reason("Mercury (element) (disambiguation)") == "disambiguation title"
    assert expand.title_filter_reason("Quantum computing") is None


def test_gather_and_score_by_distinct_seeds():
    topic_links = {
        "Quantum computing": ["Qubit", "Shor's algorithm", "List of things"],
        "Qubit": ["Shor's algorithm", "Superconductivity"],
    }
    topic_domains = {"Quantum computing": "quantum", "Qubit": "quantum"}
    existing = {"qubit"}  # Qubit is already a seed -> excluded
    cands = expand.gather_candidates(topic_links, topic_domains, existing)
    assert "qubit" not in cands  # excluded as existing
    assert "list of things" not in cands  # title-filtered
    # Shor's algorithm linked by 2 distinct seeds
    assert cands["shor's algorithm"].score == 2
    assert cands["superconductivity"].score == 1


def test_select_respects_min_links_and_budget():
    topic_links = {
        "A": ["X", "Y", "Z"],
        "B": ["X", "Y"],
        "C": ["X"],
    }
    topic_domains = {"A": "ai", "B": "ai", "C": "ai"}
    cands = expand.gather_candidates(topic_links, topic_domains, set())
    # X:3, Y:2, Z:1
    selected = expand.select_candidates(cands, min_links=2, budget_per_domain={"ai": 10})
    titles = {c.title for c in selected}
    assert titles == {"X", "Y"}  # Z below min_links
    # budget cap
    capped = expand.select_candidates(cands, min_links=2, budget_per_domain={"ai": 1})
    assert len(capped) == 1 and capped[0].title == "X"


def test_is_place():
    assert expand.is_place(["Cities in Germany", "Science"]) is True
    assert expand.is_place(["Quantum mechanics"]) is False
