"""Create the text-only Q8_0 model tag from the pulled Q8_0 tag (no new weights).

  python scripts/make_text_model.py [--source qwen3.5:9b-q8_0]
                                    [--target askito/qwen3.5:9b-q8_0-text]

The registry tag qwen3.5:9b-q8_0 ships a vision projector (mmproj) that Ollama
loads with every model load: about 0.9 GB of weights plus up to 1.1 GB of
compute buffers. Ask ITO never sends images. This creates a second tag that
points at the *same* Q8_0 GGUF blob (no copy, same digest, same quantization),
with the same template, renderer, parser and parameters, but without the
projector. Measured on a CPU sandbox: identical answers on the five smoke
questions, peak runner memory 10.6 GB instead of 12.3 GB.

Idempotent; verifies the result (Q8_0, no vision capability, no projector).
Runs wherever the backend runs (uses LLM_BASE_URL and the local-only policy).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_parameters(text: str) -> dict:
    """Ollama's `parameters` string ("key value" lines) -> create-API dict."""
    out: dict = {}
    for line in (text or "").splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, raw = parts[0], parts[1].strip().strip('"')
        try:
            value: object = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        if key in out:  # repeated keys (e.g. stop) become lists
            prev = out[key]
            out[key] = (prev if isinstance(prev, list) else [prev]) + [value]
        else:
            out[key] = value
    return out


def model_digest(modelfile: str) -> str:
    """Digest of the first FROM line of `ollama show --modelfile` output: the model
    weights (Ollama writes adapters and projectors after it)."""
    m = re.search(r"^FROM\s+\S*sha256[-:]([0-9a-f]{64})\s*$", modelfile or "", re.M)
    if not m:
        raise SystemExit("could not find the model blob in the source modelfile")
    return "sha256:" + m.group(1)


def verify(show: dict, quant: str) -> list[str]:
    problems = []
    details = show.get("details") or {}
    if details.get("quantization_level") != quant:
        problems.append(f"quantization is {details.get('quantization_level')}, expected {quant}")
    if "vision" in (show.get("capabilities") or []):
        problems.append("still has the vision capability")
    if show.get("projector_info"):
        problems.append("still has a projector")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    from app.config import settings

    ap.add_argument("--source", default=settings.llm_source_model)
    ap.add_argument("--target", default=settings.llm_default_model)
    args = ap.parse_args()

    import httpx

    from app.llm.client import local_allowed_hosts
    from app.llm.policy import ensure_local_url

    base = settings.llm_base_url.rstrip("/").removesuffix("/v1")
    if settings.llm_local_only:
        ensure_local_url(base, "Ollama", local_allowed_hosts())
    http = httpx.Client(timeout=httpx.Timeout(600, connect=10))

    def show(name: str) -> dict | None:
        r = http.post(f"{base}/api/show", json={"model": name})
        return r.json() if r.status_code == 200 else None

    if args.target == args.source:
        print(f"target equals source ({args.source}); nothing to create")
        return
    src = show(args.source)
    if src is None:
        raise SystemExit(f"{args.source} is not pulled; run: ollama pull {args.source}")
    quant = (src.get("details") or {}).get("quantization_level")
    existing = show(args.target)
    if existing is not None and not verify(existing, quant):
        print(f"{args.target} already exists: {quant}, text only (no projector)")
        return

    body = {
        "model": args.target,
        "files": {"model.gguf": model_digest(src.get("modelfile", ""))},
        "template": src.get("template") or "",
        "parameters": parse_parameters(src.get("parameters", "")),
        "stream": False,
    }
    for key in ("renderer", "parser"):
        if src.get(key):
            body[key] = src[key]
    r = http.post(f"{base}/api/create", json=body)
    if r.status_code != 200:
        raise SystemExit(f"create failed: HTTP {r.status_code} {r.text[:200]}")
    created = show(args.target) or {}
    problems = verify(created, quant)
    if problems:
        raise SystemExit(f"{args.target} is not as expected: {'; '.join(problems)}")
    print(f"created {args.target} from {args.source}: {quant}, same weights, "
          f"no vision projector, capabilities {created.get('capabilities')}")


if __name__ == "__main__":
    main()
