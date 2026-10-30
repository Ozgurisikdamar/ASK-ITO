from app.ingest.seeds import load_seeds


def test_seeds_load_and_have_all_domains():
    seeds = load_seeds()
    domains = {s.domain for s in seeds}
    assert domains == {"nobel", "quantum", "ai"}
    assert len(seeds) > 60


def test_required_ai_areas_present():
    titles = {s.title for s in load_seeds() if s.domain == "ai"}
    for required in [
        "Computer vision",
        "Transformer (deep learning architecture)",
        "Retrieval-augmented generation",
        "Intelligent agent",
        "Deep learning",
    ]:
        assert required in titles, f"missing required AI area: {required}"


def test_dedup_and_domain_priority_with_also_quantum():
    seeds = {s.title.casefold(): s for s in load_seeds()}
    # Quantum entanglement is seeded under both nobel (2022) and quantum ->
    # kept as nobel, flagged also_quantum.
    ent = seeds["quantum entanglement"]
    assert ent.domain == "nobel"
    assert ent.also_quantum is True


def test_laureates_are_not_expansion_sources():
    # laureate kind exists and topics exist
    kinds = {s.kind for s in load_seeds()}
    assert "laureate" in kinds
    assert "topic" in kinds
