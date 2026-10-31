"""Shared CPU settings for the torch models (embedder + reranker).

Concurrent requests that each try to use every core make CPU inference
thrash; serialising model calls behind one lock is faster overall for a
local, mostly single-user deployment.
"""
from __future__ import annotations

import os
import threading

from app.config import settings

inference_lock = threading.Lock()
_configured = False


def configure_torch() -> None:
    global _configured
    if _configured:
        return
    import torch

    threads = settings.torch_threads or os.cpu_count() or 1
    torch.set_num_threads(threads)
    _configured = True
