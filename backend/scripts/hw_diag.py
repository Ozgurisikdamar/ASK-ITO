"""Report CPU / RAM / GPU and the real Ollama offload state (no benchmark).

  python scripts/hw_diag.py [--base-url http://127.0.0.1:11434] [--log ollama.log]

Inside Docker run it in the backend container (`./askito shell` then this
command) and also check the Ollama container: `docker compose exec ollama
ollama ps` (PROCESSOR column: "100% GPU", "48%/52% CPU/GPU" or "100% CPU") and
`nvidia-smi` on the host. With --log, the Ollama server log is scanned for the
runner's own offload / KV-cache / flash-attention lines.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOG_PATTERNS = re.compile(
    r"inference compute|offloaded \d+/\d+ layers|llama_kv_cache: size|flash_attn\s*=|"
    r"model buffer size|KV cache shifting|no compatible GPUs", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--log", help="Ollama server log to scan for offload lines")
    args = ap.parse_args()

    import httpx

    from app.hw import diagnose
    from app.llm.policy import ensure_local_url

    ensure_local_url(args.base_url, "Ollama", ["ollama"])
    ps = None
    try:
        ps = httpx.get(f"{args.base_url}/api/ps", timeout=5).json().get("models", [])
    except Exception as exc:  # noqa: BLE001
        print(f"ollama unreachable: {exc.__class__.__name__}", file=sys.stderr)
    log_text = Path(args.log).read_text(errors="replace") \
        if args.log and Path(args.log).exists() else None
    report = diagnose(ps, log_text=log_text)
    if log_text:
        lines = log_text.splitlines()
        report["ollama_log"] = [ln[-220:] for ln in lines if LOG_PATTERNS.search(ln)][-25:]
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
