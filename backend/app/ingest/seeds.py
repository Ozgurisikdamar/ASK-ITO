"""Seed loading and flattening (spec §3.2).

Reads data/seeds/{nobel,quantum,ai}.yaml and flattens them into a deduplicated
list of Seed entries. A title that appears in more than one domain is kept once
with domain priority nobel > quantum > ai; if it also appears under quantum it
is flagged also_quantum (per §3.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

SEEDS_DIR = Path(__file__).resolve().parents[2] / "data" / "seeds"
DOMAIN_PRIORITY = {"nobel": 0, "quantum": 1, "ai": 2}


@dataclass
class Seed:
    title: str
    domain: str
    kind: str  # laureate | topic | general
    also_quantum: bool = False
    year: str | None = None
    category: str | None = None


def _norm(title: str) -> str:
    return title.strip().replace("_", " ").casefold()


def _nobel_seeds(doc: dict) -> list[Seed]:
    out: list[Seed] = []
    for title in doc.get("general", []) or []:
        out.append(Seed(title=title, domain="nobel", kind="general"))
    for year, fields in (doc.get("years") or {}).items():
        for category, groups in fields.items():
            for name in groups.get("laureates", []) or []:
                out.append(
                    Seed(name, "nobel", "laureate", year=str(year), category=category)
                )
            for name in groups.get("topics", []) or []:
                out.append(
                    Seed(name, "nobel", "topic", year=str(year), category=category)
                )
    for entry in doc.get("landmark", []) or []:
        for name in entry.get("laureates", []) or []:
            out.append(Seed(name, "nobel", "laureate", category="landmark"))
        for name in entry.get("topics", []) or []:
            out.append(Seed(name, "nobel", "topic", category="landmark"))
    return out


def _simple_topic_seeds(doc: dict, domain: str, keys: list[str]) -> list[Seed]:
    out: list[Seed] = []
    for key in keys:
        for title in doc.get(key, []) or []:
            out.append(Seed(title=title, domain=domain, kind="topic", category=key))
    return out


def load_seeds(seeds_dir: Path = SEEDS_DIR) -> list[Seed]:
    nobel = yaml.safe_load((seeds_dir / "nobel.yaml").read_text(encoding="utf-8"))
    quantum = yaml.safe_load((seeds_dir / "quantum.yaml").read_text(encoding="utf-8"))
    ai = yaml.safe_load((seeds_dir / "ai.yaml").read_text(encoding="utf-8"))

    ordered: list[Seed] = []
    ordered += _nobel_seeds(nobel)
    ordered += _simple_topic_seeds(quantum, "quantum", ["topics"])
    ordered += _simple_topic_seeds(ai, "ai", ["core", "related"])

    quantum_titles = {
        _norm(t) for t in (quantum.get("topics", []) or [])
    }

    deduped: dict[str, Seed] = {}
    for seed in ordered:
        key = _norm(seed.title)
        existing = deduped.get(key)
        if existing is None:
            if key in quantum_titles and seed.domain != "quantum":
                seed.also_quantum = True
            deduped[key] = seed
        else:
            # keep higher-priority domain; merge also_quantum
            if DOMAIN_PRIORITY[seed.domain] < DOMAIN_PRIORITY[existing.domain]:
                if key in quantum_titles and seed.domain != "quantum":
                    seed.also_quantum = True
                deduped[key] = seed
            elif seed.domain == "quantum":
                existing.also_quantum = existing.domain != "quantum"
    return list(deduped.values())
