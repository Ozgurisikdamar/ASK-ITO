"""Ingest orchestrator (spec §3): resolve seeds, fetch, expand, clean, snapshot.

Writes:
  * data/raw/{pageid}.json       cleaned page snapshot (git-ignored)
  * data/raw/_index.json         title -> pageid resolution cache (for resume)
  * data/seeds/unresolved.log    seeds that could not be resolved, with reason
  * data/raw/expansion_report.csv every considered candidate, selected or not
  * documents table              one row per accepted document

Resumable: already-fetched pages are reused from data/raw unless --force is
given (important when the network is throttled and a run is interrupted).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from app.config import settings
from app.ingest import expand
from app.ingest.clean import char_count, clean_extract
from app.ingest.seeds import Seed, load_seeds
from app.ingest.wiki import PageResult, WikiClient, WikiError

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SEEDS_DIR = Path(__file__).resolve().parents[2] / "data" / "seeds"
INDEX_FILE = RAW_DIR / "_index.json"
UNRESOLVED_LOG = SEEDS_DIR / "unresolved.log"
EXPANSION_CSV = RAW_DIR / "expansion_report.csv"


def _norm(title: str) -> str:
    return title.strip().replace("_", " ").casefold()


def _load_index() -> dict:
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    return {}


def _save_index(index: dict) -> None:
    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=0), encoding="utf-8")


def _raw_path(pageid: int) -> Path:
    return RAW_DIR / f"{pageid}.json"


def _store_doc_json(doc: dict) -> None:
    _raw_path(doc["pageid"]).write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def _build_doc(page: PageResult, *, domain: str, is_seed: bool, also_quantum: bool,
               kind: str) -> dict:
    sections = clean_extract(page.extract or "")
    return {
        "pageid": page.pageid,
        "revid": page.revid,
        "title": page.title,
        "url": page.url,
        "domain": domain,
        "is_seed": is_seed,
        "also_quantum": also_quantum,
        "kind": kind,
        "categories": page.categories,
        "requested_title": page.requested_title,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "char_count": char_count(sections),
        "sections": [{"title": s.title, "text": s.text} for s in sections],
    }


def _fetch(client: WikiClient, title: str, index: dict, force: bool) -> PageResult | None:
    """Fetch a page, reusing the on-disk snapshot when possible. Returns None if
    unresolved (missing/disambiguation); the index records the outcome."""
    key = _norm(title)
    if not force and key in index:
        rec = index[key]
        if rec["status"] == "ok" and _raw_path(rec["pageid"]).exists():
            data = json.loads(_raw_path(rec["pageid"]).read_text(encoding="utf-8"))
            pr = PageResult(requested_title=title, pageid=data["pageid"],
                            revid=data["revid"], title=data["title"], url=data["url"],
                            categories=data.get("categories", []))
            pr._cached = data  # type: ignore[attr-defined]
            return pr
        if rec["status"] in ("missing", "disambiguation"):
            return None
    page = client.fetch_page(title)
    if page.missing:
        index[key] = {"status": "missing", "pageid": None}
        return None
    if page.is_disambiguation:
        index[key] = {"status": "disambiguation", "pageid": page.pageid}
        return None
    index[key] = {"status": "ok", "pageid": page.pageid}
    return page


def run(force: bool = False) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seeds = load_seeds()
    index = _load_index()
    unresolved: list[str] = []
    documents: dict[int, dict] = {}  # pageid -> doc
    topic_links: dict[str, list[str]] = {}
    topic_domains: dict[str, str] = {}
    seed_domain_by_norm: dict[str, str] = {}

    deferred: list[str] = []
    client = WikiClient(delay=settings.wiki_delay, max_retries=8)
    try:
        # --- 1. seeds -----------------------------------------------------
        for seed in seeds:
            try:
                page = _fetch(client, seed.title, index, force)
            except WikiError:
                deferred.append(seed.title)  # throttled; resume will retry
                continue
            _save_index(index)
            if page is None:
                status = index.get(_norm(seed.title), {}).get("status", "unresolved")
                unresolved.append(f"{seed.title}\t{seed.domain}/{seed.kind}\t{status}")
                continue
            if page.pageid in documents:
                continue  # already captured via another seed/alias
            cached = getattr(page, "_cached", None)
            if cached is not None:
                doc = dict(cached)
                doc.update(domain=seed.domain, is_seed=True,
                           also_quantum=seed.also_quantum, kind=seed.kind)
            else:
                doc = _build_doc(page, domain=seed.domain, is_seed=True,
                                 also_quantum=seed.also_quantum, kind=seed.kind)
                _store_doc_json(doc)
            documents[page.pageid] = doc
            seed_domain_by_norm[_norm(page.title)] = seed.domain
            seed_domain_by_norm[_norm(seed.title)] = seed.domain
            # expansion sources = topic seeds only
            if seed.kind == "topic":
                try:
                    links = client.fetch_links(page.title)
                    topic_links[page.title] = links
                    topic_domains[page.title] = seed.domain
                except WikiError:
                    deferred.append(f"links:{page.title}")

        existing = set(seed_domain_by_norm.keys()) | {_norm(d["title"]) for d in documents.values()}

        # --- 2. expansion candidates -------------------------------------
        candidates = expand.gather_candidates(topic_links, topic_domains, existing)
        budgets = {
            "nobel": settings.ingest_expand_max_nobel,
            "quantum": settings.ingest_expand_max_quantum,
            "ai": settings.ingest_expand_max_ai,
        }
        selected = expand.select_candidates(
            candidates, min_links=settings.ingest_expand_min_links,
            budget_per_domain=budgets,
        )
        selected_keys = {_norm(c.title) for c in selected}

        report_rows: list[dict] = []
        for key, cand in sorted(candidates.items(), key=lambda kv: kv[1].score, reverse=True):
            if cand.score < settings.ingest_expand_min_links:
                status, reason = "rejected", "below_min_links"
            elif key not in selected_keys:
                status, reason = "rejected", "over_budget"
            else:
                status, reason = "candidate", ""
            report_rows.append({
                "title": cand.title, "domain": cand.domain, "score": cand.score,
                "status": status, "reason": reason,
            })

        # --- 3. fetch selected candidates --------------------------------
        for cand in selected:
            try:
                page = _fetch(client, cand.title, index, force)
            except WikiError:
                deferred.append(cand.title)
                continue
            _save_index(index)
            row = next(r for r in report_rows if _norm(r["title"]) == _norm(cand.title))
            if page is None:
                row["status"], row["reason"] = "rejected", \
                    index.get(_norm(cand.title), {}).get("status", "unresolved")
                continue
            if page.pageid in documents:
                row["status"], row["reason"] = "rejected", "duplicate_of_seed"
                continue
            cached = getattr(page, "_cached", None)
            doc = dict(cached) if cached is not None else _build_doc(
                page, domain=cand.domain, is_seed=False,
                also_quantum=(cand.domain != "quantum" and "quantum" in cand.domain_votes),
                kind="expanded")
            if cached is not None:
                doc.update(domain=cand.domain, is_seed=False, kind="expanded",
                           also_quantum=(cand.domain != "quantum" and "quantum" in cand.domain_votes))
            if expand.is_place(doc.get("categories", [])):
                row["status"], row["reason"] = "rejected", "place_page"
                continue
            if doc["char_count"] < settings.ingest_min_chars:
                row["status"], row["reason"] = "rejected", "too_short"
                continue
            if cached is None:
                _store_doc_json(doc)
            documents[page.pageid] = doc
            row["status"], row["reason"] = "accepted", ""
    finally:
        client.close()
        _save_index(index)

    _write_reports(unresolved, report_rows)
    persist_documents(list(documents.values()))
    return _summary(documents, seeds, unresolved, report_rows, client.request_count, deferred)


def _write_reports(unresolved: list[str], report_rows: list[dict]) -> None:
    UNRESOLVED_LOG.write_text(
        "# requested_title\tdomain/kind\tstatus\n" + "\n".join(unresolved) + "\n",
        encoding="utf-8",
    )
    with EXPANSION_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["title", "domain", "score", "status", "reason"])
        w.writeheader()
        w.writerows(report_rows)


def persist_documents(docs: list[dict]) -> None:
    if not docs:
        return
    with psycopg.connect(settings.database_url) as conn:
        for d in docs:
            conn.execute(
                """
                INSERT INTO documents
                    (pageid, revid, title, url, domain, also_quantum, is_seed,
                     categories, fetched_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (pageid) DO UPDATE SET
                    revid=EXCLUDED.revid, title=EXCLUDED.title, url=EXCLUDED.url,
                    domain=EXCLUDED.domain, also_quantum=EXCLUDED.also_quantum,
                    is_seed=EXCLUDED.is_seed, categories=EXCLUDED.categories,
                    fetched_at=EXCLUDED.fetched_at
                """,
                (d["pageid"], d["revid"], d["title"], d["url"], d["domain"],
                 d["also_quantum"], d["is_seed"], d["categories"], d["fetched_at"]),
            )
        conn.commit()


def _summary(documents, seeds, unresolved, report_rows, requests, deferred) -> dict:
    by_domain = Counter(d["domain"] for d in documents.values())
    accepted_exp = sum(1 for r in report_rows if r["status"] == "accepted")
    return {
        "documents": len(documents),
        "by_domain": dict(by_domain),
        "seeds_total": len(seeds),
        "seeds_unresolved": len(unresolved),
        "deferred_throttled": len(deferred),
        "candidates_considered": len(report_rows),
        "candidates_accepted": accepted_exp,
        "api_requests": requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask ITO ingest")
    parser.add_argument("--force", action="store_true", help="re-fetch, ignore snapshot cache")
    args = parser.parse_args()
    try:
        summary = run(force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"Ingest failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
