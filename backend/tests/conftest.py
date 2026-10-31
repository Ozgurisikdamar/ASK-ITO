"""Test configuration.

Force mock LLM mode so tests never touch the network, and disable the startup
warm-up / featured pre-warm threads, before app import.
"""
import os

os.environ.setdefault("LLM_MODE", "mock")
os.environ.setdefault("WARMUP_ON_START", "false")
os.environ.setdefault("PREWARM_FEATURED", "false")
