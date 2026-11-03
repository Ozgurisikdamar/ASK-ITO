"""Q4_K_M vs Q8_0 smoke comparison on exactly five questions (no benchmark matrix).

  python scripts/quant_smoke.py freeze                       # retrieval once -> frozen prompts
  python scripts/quant_smoke.py run --model qwen3.5:9b       # Q4_K_M baseline
  python scripts/quant_smoke.py run --model qwen3.5:9b-q8_0  # Q8_0
  python scripts/quant_smoke.py prefix --model qwen3.5:9b-q8_0  # prefix-cache probe

`freeze` runs the Answer (overview) retrieval path once per question (hybrid +
bge reranker + evidence cut-off + context selection) and stores the exact chat
messages, generation limits and the numbered source passages. `run` sends those
identical messages to one model through the app's Ollama client (temperature 0,
same num_ctx / num_predict / think setting), so the only variable is the
weights. Answer caches are not involved at all. Per question it records:
TTFT (wall), prefill (prompt_eval tokens and ms), decode (eval tokens and ms),
tokens/s, total, peak RSS of the Ollama processes (sampled from /proc) and
VRAM reported by /api/ps, the answer, citation ids and citation-id validity.
Grounding (claim support against the cited text) is judged by hand from the
stored passages. Output: eval/checkpoints/quant_smoke/<file>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

QUESTIONS = [
    "quantum computing",
    "What is machine learning?",
    "popular AI fields",
    "Willow processor 105 physical qubits",
    "Hopfield and Hinton, 2024 Physics prize",
]
OUT = Path(os.environ.get("ASKITO_CHECKPOINT_DIR",
                          Path(__file__).resolve().parents[1] / "eval" / "checkpoints")) \
    / "quant_smoke"


def _env() -> None:
    os.environ["WARMUP_ON_START"] = "false"
    os.environ["PREWARM_FEATURED"] = "false"
    os.environ["ANSWER_CACHE_ENABLED"] = "false"
    os.environ["EMBEDDING_CACHE_SIZE"] = "0"
    os.environ["AGENT_TOOL_CACHE"] = "false"
    os.environ["REDIS_ENABLED"] = "false"
    os.environ.setdefault("LLM_MODE", "ollama")
    os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:11434")


# -- resources ------------------------------------------------------------------

def _ollama_pids() -> list[int]:
    pids = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            comm = (p / "comm").read_text().strip()
        except OSError:
            continue
        if comm in ("ollama", "llama-server", "ollama_llama_se"):
            pids.append(int(p.name))
    return pids


def _rss_mb() -> float:
    total = 0
    for pid in _ollama_pids():
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS"):
                    total += int(line.split()[1])
        except OSError:
            pass
    return round(total / 1024, 1)


def runtime_env() -> dict:
    """KV-cache / flash-attention settings of the running Ollama server, read
    from its process environment (what is actually in effect, not a config file)."""
    keys = ("OLLAMA_KV_CACHE_TYPE", "OLLAMA_FLASH_ATTENTION", "OLLAMA_CONTEXT_LENGTH",
            "OLLAMA_NUM_PARALLEL", "OLLAMA_KEEP_ALIVE")
    for pid in _ollama_pids():
        try:
            raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except OSError:
            continue
        env = dict(x.decode(errors="replace").split("=", 1) for x in raw if b"=" in x)
        if any(k in env for k in keys):
            return {k: env.get(k) for k in keys}
    return {}


class PeakRSS:
    def __init__(self, every_s: float = 0.25) -> None:
        self.every = every_s
        self.peak = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_mb())
            self._stop.wait(self.every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, _rss_mb())


# -- freeze -----------------------------------------------------------------------

def freeze(args) -> None:
    _env()
    from app.agent.answer import _messages
    from app.config import settings
    from app.retrieval import hydrate, pipeline, service

    out = []
    for q in QUESTIONS:
        pipeline.cache_clear()
        result = pipeline.run_retrieval(q, "overview", domain="all",
                                        final_k=service.context_k("overview"))
        sources = service.context_sources(result.ranked or result.final, "overview",
                                          result.counts.get("reranked", 0))
        messages, max_tokens = _messages(q, sources, "overview") if sources else ([], 0)
        out.append({
            "question": q, "messages": messages, "max_tokens": max_tokens,
            "sources": [{"n": s.n, "chunk_id": s.chunk_id, "doc_title": s.info.doc_title,
                         "section": s.info.section, "score": round(float(s.score), 4),
                         "text": hydrate.body_without_prefix(s.info.text)}
                        for s in sources],
            "retrieval_ms": result.timings,
        })
        print(f"froze {q!r}: {len(sources)} sources", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"reranker": settings.reranker_model, "rerank_pool": settings.rerank_pool,
            "context_chunks": settings.overview_context_chunks,
            "context_chars": settings.overview_context_chars,
            "min_evidence": settings.fast_min_evidence_score,
            "prompt_version": settings.prompt_version, "num_ctx": settings.llm_num_ctx,
            "think": settings.llm_think}
    (OUT / "frozen.json").write_text(json.dumps({"settings": meta, "questions": out},
                                                indent=1))
    print(f"wrote {OUT / 'frozen.json'}")


# -- run ----------------------------------------------------------------------------

def _citation_check(raw: str, n_sources: int) -> dict:
    cited = [int(x) for x in re.findall(r"\[(\d+)\]", raw)]
    valid = [c for c in cited if 1 <= c <= n_sources]
    return {"cited_ids": sorted(set(cited)), "citations_total": len(cited),
            "citation_id_validity": round(len(valid) / len(cited), 3) if cited else None}


def run(args) -> None:
    _env()
    import httpx

    from app.agent.answer import NOT_FOUND, validate_citations
    from app.config import settings
    from app.llm.client import OllamaClient
    from benchkit import hardware, ollama_info

    frozen = json.loads((OUT / "frozen.json").read_text())
    base = settings.llm_base_url
    client = OllamaClient(base)
    info = ollama_info(base, args.model)
    if not info.get("quantization"):
        sys.exit(f"model {args.model} not available locally: {info}")

    ps = httpx.get(f"{base}/api/ps", timeout=10).json().get("models", [])
    was_loaded = any(m.get("name") == args.model for m in ps)
    rss_before = _rss_mb()
    t = time.perf_counter()
    with PeakRSS() as load_peak:
        r = httpx.post(f"{base}/api/generate", json={"model": args.model, "prompt": "",
                                                       "keep_alive": settings.llm_keep_alive},
                       timeout=3600)
    r.raise_for_status()
    load = {"already_loaded": was_loaded, "wall_s": round(time.perf_counter() - t, 2),
            "load_duration_s": round(r.json().get("load_duration", 0) / 1e9, 2),
            "rss_before_mb": rss_before, "rss_peak_mb": load_peak.peak,
            "rss_after_mb": _rss_mb()}
    ps = httpx.get(f"{base}/api/ps", timeout=10).json().get("models", [])
    loaded = next((m for m in ps if m.get("name") == args.model), {})
    load["size_bytes"] = loaded.get("size")
    load["size_vram_bytes"] = loaded.get("size_vram")
    print("loaded", load, flush=True)

    rows = []
    for item in frozen["questions"]:
        q = item["question"]
        if not item["messages"]:  # evidence cut-off: no LLM call for either model
            rows.append({"question": q, "abstained_before_llm": True, "answer": NOT_FOUND})
            continue
        pieces, first = [], None
        t0 = time.perf_counter()
        with PeakRSS() as peak:
            for piece in client.stream_chat(item["messages"], temperature=0.0,
                                            model=args.model,
                                            max_tokens=item["max_tokens"]):
                if first is None:
                    first = time.perf_counter() - t0
                pieces.append(piece)
        wall = time.perf_counter() - t0
        u = dict(client.last_usage)
        raw = "".join(pieces)
        cleaned, used = validate_citations(raw, len(item["sources"]))
        decode_s = (u.get("eval_ms") or 0) / 1000
        prefill_s = (u.get("prompt_eval_ms") or 0) / 1000
        row = {
            "question": q, "ttft_s": round(first or wall, 2), "total_s": round(wall, 2),
            "prompt_tokens": u.get("prompt_tokens"), "prefill_s": round(prefill_s, 2),
            "prefill_tok_s": round(u["prompt_tokens"] / prefill_s, 2)
            if prefill_s and u.get("prompt_tokens") else None,
            "output_tokens": u.get("completion_tokens"), "decode_s": round(decode_s, 2),
            "decode_tok_s": round(u["completion_tokens"] / decode_s, 2)
            if decode_s and u.get("completion_tokens") else None,
            "load_ms": u.get("load_ms"), "done_reason": u.get("done_reason"),
            "peak_rss_mb": peak.peak,
            "answer_raw": raw, "answer": cleaned or NOT_FOUND,
            "citations_used": sorted(used),
            **_citation_check(raw, len(item["sources"])),
        }
        rows.append(row)
        print(f"[{args.model}] {q!r}: ttft {row['ttft_s']}s total {row['total_s']}s "
              f"prefill {row['prompt_tokens']} tok/{row['prefill_s']}s "
              f"decode {row['output_tokens']} tok @ {row['decode_tok_s']} tok/s "
              f"rss {row['peak_rss_mb']} MB", flush=True)

    ps = httpx.get(f"{base}/api/ps", timeout=10).json().get("models", [])
    loaded = next((m for m in ps if m.get("name") == args.model), {})
    out = {"model": args.model, "ollama": info, "runtime_env": runtime_env(),
           "hardware": hardware(), "load": load, "size_vram_bytes": loaded.get("size_vram"),
           "settings": frozen["settings"], "rows": rows,
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    safe = args.model.replace(":", "_").replace("/", "_")
    path = OUT / f"run_{safe}_{args.tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


# -- prefix cache probe ---------------------------------------------------------------

def prefix(args) -> None:
    """Send the same long prompt twice, then the same passages with another
    question. If the runtime reuses the cached prefix, the repeated request
    evaluates far fewer prompt tokens (prompt_eval_count) and its prefill is
    shorter. Reports what was observed; it does not assume reuse."""
    _env()
    from app.config import settings
    from app.llm.client import OllamaClient

    frozen = json.loads((OUT / "frozen.json").read_text())
    item = next(i for i in frozen["questions"] if i["messages"])
    msg = item["messages"][0]["content"]
    other = msg.rsplit("Question:", 1)[0] + "Question: Who built it?\n\nAnswer:"
    client = OllamaClient(settings.llm_base_url)
    obs = []
    for label, content in (("first", msg), ("identical_repeat", msg),
                           ("same_passages_new_question", other)):
        t0 = time.perf_counter()
        client.chat([{"role": "user", "content": content}], model=args.model,
                    temperature=0.0, max_tokens=8)
        u = dict(client.last_usage)
        obs.append({"request": label, "prompt_eval_count": u.get("prompt_tokens"),
                    "prompt_eval_ms": u.get("prompt_eval_ms"),
                    "wall_s": round(time.perf_counter() - t0, 2)})
        print(obs[-1], flush=True)
    first, rep = obs[0]["prompt_eval_count"] or 0, obs[1]["prompt_eval_count"] or 0
    verdict = ("reuse observed" if first and rep < first * 0.5 else
               "no reuse observed (prompt fully re-evaluated)")
    out = {"model": args.model, "runtime_env": runtime_env(), "observations": obs,
           "verdict": verdict}
    safe = args.model.replace(":", "_")
    (OUT / f"prefix_{safe}_{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(verdict)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    for name in ("run", "prefix"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--tag", default="kv", help="label for the runtime config (e.g. kv-f16)")
    args = ap.parse_args()
    {"freeze": freeze, "run": run, "prefix": prefix}[args.cmd](args)


if __name__ == "__main__":
    main()
