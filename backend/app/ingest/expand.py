"""One-step expansion (spec §3.3).

Candidates are titles linked from *topic* articles (not laureate biographies).
Each candidate is scored by how many distinct seed topics link to it. Title-based
filters drop list/index, year/date and disambiguation-looking pages here; the
remaining fetch-based filters (disambiguation page, < 1500 chars, place pages)
are applied in the runner once the candidate is fetched.

The embedding-cosine relevance component described in §3.3 is applied in M3,
where the embedding model is loaded for indexing; see docs/DECISIONS.md. With
the curated seed set the distinct-seed link count is the primary signal.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_LIST_PREFIXES = ("list of", "lists of", "index of", "outline of")
_YEAR = re.compile(r"^\d{3,4}$")
_YEAR_IN = re.compile(r"^\d{3,4}\s+in\b", re.IGNORECASE)
_CENTURY = re.compile(r"^\d{1,2}(st|nd|rd|th)\s+century", re.IGNORECASE)
_MONTH_DAY = re.compile(
    r"^(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2}$",
    re.IGNORECASE,
)

# category keywords that mark a place page (dropped per §3.3)
PLACE_CATEGORY_HINTS = (
    "populated places",
    "cities",
    "towns",
    "countries",
    "capitals",
    "states of",
    "municipalities",
)


def _norm(title: str) -> str:
    return title.strip().replace("_", " ").casefold()


def title_filter_reason(title: str) -> str | None:
    low = title.strip().casefold()
    if low.startswith(_LIST_PREFIXES):
        return "list/index page"
    if "(disambiguation)" in low:
        return "disambiguation title"
    if _YEAR.match(low) or _YEAR_IN.match(low) or _CENTURY.match(low) or _MONTH_DAY.match(low):
        return "year/date page"
    return None


@dataclass
class Candidate:
    title: str
    linking_seeds: set[str] = field(default_factory=set)
    domain_votes: Counter = field(default_factory=Counter)

    @property
    def score(self) -> int:
        return len(self.linking_seeds)

    @property
    def domain(self) -> str:
        return self.domain_votes.most_common(1)[0][0]


def gather_candidates(
    topic_links: dict[str, list[str]],
    topic_domains: dict[str, str],
    existing_titles: set[str],
) -> dict[str, Candidate]:
    """topic_links: {seed_title -> [linked titles]} for TOPIC seeds only.
    topic_domains: {seed_title -> domain}. existing_titles: normalized titles
    already in the corpus (seeds), excluded from candidates."""
    candidates: dict[str, Candidate] = {}
    for seed_title, links in topic_links.items():
        seed_domain = topic_domains.get(seed_title, "nobel")
        for link in links:
            key = _norm(link)
            if key in existing_titles:
                continue
            if title_filter_reason(link):
                continue
            cand = candidates.get(key)
            if cand is None:
                cand = Candidate(title=link)
                candidates[key] = cand
            cand.linking_seeds.add(_norm(seed_title))
            cand.domain_votes[seed_domain] += 1
    return candidates


def select_candidates(
    candidates: dict[str, Candidate],
    *,
    min_links: int,
    budget_per_domain: dict[str, int],
) -> list[Candidate]:
    """Rank by score (distinct linking seeds), take top-N per domain."""
    eligible = [c for c in candidates.values() if c.score >= min_links]
    eligible.sort(key=lambda c: (c.score, c.title.casefold()), reverse=True)
    taken: Counter = Counter()
    selected: list[Candidate] = []
    for cand in eligible:
        dom = cand.domain
        if taken[dom] >= budget_per_domain.get(dom, 0):
            continue
        selected.append(cand)
        taken[dom] += 1
    return selected


def is_place(categories: list[str]) -> bool:
    low = [c.casefold() for c in categories]
    return any(any(h in c for h in PLACE_CATEGORY_HINTS) for c in low)
