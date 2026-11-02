"""LLM micro-benchmark against a local Ollama (native API, streaming).

  python scripts/llm_bench.py --model qwen3:8b --reps 3
  python scripts/llm_bench.py --model qwen3.5:9b --threads 0,2 --ctx 4096,8192

Uses the project's real prompt templates filled with real retrieved context:
  analyze      (JSON, short prompt)         ~ agent step 1
  sufficiency  (JSON, 8 passages)           ~ agent sufficiency check
  answer       (cited answer, 5 passages)   ~ final answer
For each (num_thread, num_ctx) setting: one unrecorded warm-up per prompt, then
--reps measured runs. A cold start (model unloaded with keep_alive=0, then the
first request) is measured once per model. Records TTFT (first streamed
token), Ollama's prompt/eval token counts and durations, load time, and peak RSS
of the ollama processes. Resumable JSONL checkpoint:
eval/checkpoints/llm_<model>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

QUESTION = "What did John Hopfield and Geoffrey Hinton each contribute to the 2024 Nobel Prize in Physics?"


def build_prompts() -> dict[str, tuple[str, bool, int]]:
    os.environ.setdefault("LLM_MODE", "mock")
    from app.agent.answer import build_context, build_sources
    from app.agent.prompts import load
    from app.retrieval import hydrate, pipeline

    res = pipeline.run_retrieval(QUESTION, "hybrid_rerank", domain="all", final_k=8,
                                 use_cache=False)
    infos = hydrate.fetch_chunks([c for c, _ in res.final])
    suff_ctx = "\n\n".join(
        f"[{i}] {infos[c].doc_title}: {hydrate.body_without_prefix(infos[c].text)[:400]}"
        for i, (c, _) in enumerate(res.final[:8], start=1) if c in infos)
    sources = build_sources(res.final[:5], infos)
    return {
        "analyze": (load("analyze").replace("{question}", QUESTION), True, 400),
        "sufficiency": (load("sufficiency").replace("{question}", QUESTION)
                        .replace("{context}", suff_ctx), True, 400),
        "answer": (load("answer").replace("{context}", build_context(sources))
                   .replace("{question}", QUESTION), False, 600),
    }


def run_one(base: str, model: str, prompt: str, json_mode: bool, max_tokens: int,
            num_thread: int, num_ctx: int, keep_alive: str = "24h") -> dict:
    options = {"temperature": 0.0, "num_ctx": num_ctx, "num_predict": max_tokens}
    if num_thread:
        options["num_thread"] = num_thread
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": True, "think": False, "keep_alive": keep_alive, "options": options}
    if json_mode:
        body["format"] = "json"
    t0 = time.perf_counter()
    ttft, text, final = None, [], {}
    with httpx.stream("POST", f"{base}/api/chat", json=body, timeout=900) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            piece = obj.get("message", {}).get("content", "")
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            text.append(piece)
            if obj.get("done"):
                final = obj
                break
    total = time.perf_counter() - t0
    out = "".join(text)
    rec = {
        "ttft_s": round(ttft, 3) if ttft else None, "total_s": round(total, 3),
        "prompt_tokens": final.get("prompt_eval_count"),
        "completion_tokens": final.get("eval_count"),
        "prompt_eval_s": round(final.get("prompt_eval_duration", 0) / 1e9, 3),
        "eval_s": round(final.get("eval_duration", 0) / 1e9, 3),
        "load_s": round(final.get("load_duration", 0) / 1e9, 3),
        "done_reason": final.get("done_reason"), "output_chars": len(out),
        "output_head": out[:160],
    }
    if rec["completion_tokens"] and rec["eval_s"]:
        rec["gen_tok_s"] = round(rec["completion_tokens"] / rec["eval_s"], 2)
    if rec["prompt_tokens"] and rec["prompt_eval_s"]:
        rec["prefill_tok_s"] = round(rec["prompt_tokens"] / rec["prompt_eval_s"], 2)
    if json_mode:
        try:
            json.loads(out)
            rec["json_valid"] = True
        except json.JSONDecodeError:
            rec["json_valid"] = False
    if rec["prompt_tokens"] and rec["prompt_tokens"] >= num_ctx - max_tokens:
        rec["context_overflow_risk"] = True
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:11434")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", default="0", help="comma list; 0 = Ollama default")
    ap.add_argument("--ctx", default="4096", help="comma list of num_ctx values")
    ap.add_argument("--no-cold", action="store_true")
    ap.add_argument("--tag", default="default",
                    help="label for the Ollama server configuration under test")
    args = ap.parse_args()

    from benchkit import CHECKPOINT_DIR, Checkpoint, ollama_info, summary
    from eval_bench import ResourceSampler

    safe = args.model.replace(":", "_").replace("/", "_") + f"_{args.tag}"
    ck = Checkpoint(CHECKPOINT_DIR / f"llm_{safe}.jsonl")
    prompts = build_prompts()
    threads = [int(x) for x in args.threads.split(",")]
    ctxs = [int(x) for x in args.ctx.split(",")]
    planned = len(threads) * len(ctxs) * len(prompts) * args.reps + (0 if args.no_cold else 1)
    meta = ollama_info(args.base, args.model)
    server_env = {k: v for k, v in os.environ.items() if k.startswith("OLLAMA_")}
    ck.header(benchmark="llm", model=meta, server_tag=args.tag, server_env=server_env,
              threads=threads, ctx=ctxs, reps=args.reps,
              prompts=list(prompts), planned_units=planned, question=QUESTION)

    if not args.no_cold and not ck.has("cold"):
        httpx.post(f"{args.base}/api/generate", json={"model": args.model, "keep_alive": 0},
                   timeout=120)
        time.sleep(2)
        p, js, mt = prompts["analyze"]
        sampler = ResourceSampler().start()
        rec = run_one(args.base, args.model, p, js, mt, threads[0], ctxs[0])
        ck.unit("cold", prompt="analyze", num_thread=threads[0], num_ctx=ctxs[0],
                resources=sampler.stop(), **rec)
        print("cold", json.dumps(rec)[:300], flush=True)

    for nt in threads:
        for nc in ctxs:
            for name, (p, js, mt) in prompts.items():
                if all(ck.has(f"t{nt}|c{nc}|{name}|r{r}") for r in range(1, args.reps + 1)):
                    continue
                run_one(args.base, args.model, p, js, mt, nt, nc)  # warm-up, not recorded
                for r in range(1, args.reps + 1):
                    key = f"t{nt}|c{nc}|{name}|r{r}"
                    if ck.has(key):
                        continue
                    sampler = ResourceSampler().start()
                    try:
                        rec = run_one(args.base, args.model, p, js, mt, nt, nc)
                    except Exception as exc:  # noqa: BLE001
                        ck.unit(key, status="error", error=f"{exc.__class__.__name__}: {exc}")
                        continue
                    ck.unit(key, prompt=name, num_thread=nt, num_ctx=nc, rep=r,
                            resources=sampler.stop(), **rec)
                    print(key, rec.get("ttft_s"), rec.get("total_s"), rec.get("gen_tok_s"),
                          rec.get("prefill_tok_s"), flush=True)

    units = [u for u in ck.units() if u["key"] != "cold"]
    table = {}
    for nt in threads:
        for nc in ctxs:
            for name in prompts:
                rows = [u for u in units if u.get("num_thread") == nt and u.get("num_ctx") == nc
                        and u.get("prompt") == name]
                if not rows:
                    continue
                table[f"t{nt}|c{nc}|{name}"] = {
                    "n": len(rows),
                    "prompt_tokens": rows[0].get("prompt_tokens"),
                    "completion_tokens": summary([r["completion_tokens"] for r in rows
                                                  if r.get("completion_tokens")]),
                    "ttft_s": summary([r["ttft_s"] for r in rows if r.get("ttft_s")]),
                    "total_s": summary([r["total_s"] for r in rows]),
                    "gen_tok_s": summary([r["gen_tok_s"] for r in rows if r.get("gen_tok_s")]),
                    "prefill_tok_s": summary([r["prefill_tok_s"] for r in rows
                                              if r.get("prefill_tok_s")]),
                    "json_valid": [r.get("json_valid") for r in rows] if rows[0].get(
                        "json_valid") is not None else None,
                    "peak_rss_mb": max(r["resources"]["ollama_peak_rss_mb"] for r in rows),
                    "context_overflow_risk": any(r.get("context_overflow_risk") for r in rows),
                }
    done = len(ck.units())
    out = {"model": meta, "header": ck.headers[-1], "cold": ck.done.get("cold"),
           "planned": planned, "completed": done,
           "status": "RUN" if done >= planned else f"PARTIAL ({done}/{planned})",
           "table": table}
    ck.footer(planned)
    (CHECKPOINT_DIR / f"llm_{safe}.summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(table, indent=1)[:4000])


if __name__ == "__main__":
    main()
