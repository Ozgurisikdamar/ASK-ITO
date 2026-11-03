"""Test configuration.

Force mock LLM mode so tests never touch the network, and disable the startup
warm-up / featured pre-warm threads, before app import.
"""
import os

os.environ.setdefault("LLM_MODE", "mock")
os.environ.setdefault("WARMUP_ON_START", "false")
os.environ.setdefault("PREWARM_FEATURED", "false")
# Call-count assertions need every agent run to reach the (scripted) LLM; the
# tool cache has its own tests that switch it on.
os.environ.setdefault("AGENT_TOOL_CACHE", "false")
os.environ.setdefault("REDIS_ENABLED", "false")
# AI Mode would search Wikipedia on weak evidence; tests that cover it switch it on
# with a fake client.
os.environ.setdefault("LIVE_INGEST_ENABLED", "false")
