"""Shared helpers for the benchmark scripts: run metadata + resumable checkpoints.

Every benchmark writes an append-only JSONL checkpoint. The first line of a run
is a header record (start time, git SHA, hardware, model, config, planned
count); each completed unit of work is one record; a footer records the end
time and completed/failed counts. Re-running with the same checkpoint path
skips units whose key is already recorded, so an interrupted run loses nothing.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

BACKEND = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = Path(os.environ.get("ASKITO_CHECKPOINT_DIR", BACKEND / "eval" / "checkpoints"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BACKEND, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def hardware() -> dict:
    cpu, cores = "unknown", os.cpu_count()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    ram_gb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                ram_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except OSError:
        pass
    return {
        "cpu": cpu, "logical_cpus": cores, "ram_gb": ram_gb,
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
    }


def ollama_info(base_url: str, model: str | None) -> dict:
    """Version + model metadata from a local Ollama (empty dict if unreachable)."""
    base = base_url.rstrip("/").removesuffix("/v1")
    out: dict = {}
    try:
        out["ollama_version"] = httpx.get(f"{base}/api/version", timeout=3).json().get("version")
    except Exception:  # noqa: BLE001
        return out
    if not model:
        return out
    try:
        show = httpx.post(f"{base}/api/show", json={"model": model}, timeout=10).json()
        details = show.get("details", {})
        tags = httpx.get(f"{base}/api/tags", timeout=5).json().get("models", [])
        digest = next((m.get("digest") for m in tags if m.get("name") == model), None)
        out.update({
            "model": model, "digest": digest,
            "quantization": details.get("quantization_level"),
            "parameter_size": details.get("parameter_size"),
            "family": details.get("family"), "format": details.get("format"),
            "capabilities": show.get("capabilities"),
            "context_length": next(
                (v for k, v in (show.get("model_info") or {}).items()
                 if k.endswith(".context_length")), None),
        })
    except Exception as exc:  # noqa: BLE001
        out["model_error"] = f"{exc.__class__.__name__}"
    return out


class Checkpoint:
    """Append-only JSONL with resume. Units are identified by a string key."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, dict] = {}
        self.headers: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                kind = rec.get("kind")
                if kind == "unit" and rec.get("status") == "ok":
                    self.done[rec["key"]] = rec
                elif kind == "header":
                    self.headers.append(rec)

    def _append(self, rec: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def header(self, **fields) -> None:
        rec = {"kind": "header", "started_at": now(), "git_sha": git_sha(),
               "hardware": hardware(), **fields}
        self.headers.append(rec)
        self._append(rec)

    def has(self, key: str) -> bool:
        return key in self.done

    def unit(self, key: str, status: str = "ok", **fields) -> None:
        rec = {"kind": "unit", "key": key, "status": status, "at": now(), **fields}
        if status == "ok":
            self.done[key] = rec
        self._append(rec)

    def footer(self, planned: int, **fields) -> None:
        self._append({"kind": "footer", "finished_at": now(), "planned": planned,
                      "completed": len(self.done), **fields})

    def units(self) -> list[dict]:
        return list(self.done.values())


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values), "p50": pct(values, 0.5), "p95": pct(values, 0.95),
        "mean": round(statistics.fmean(values), 1),
        "min": min(values), "max": max(values),
        "stdev": round(statistics.pstdev(values), 1) if len(values) > 1 else 0.0,
    }


def mean_or_none(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None
