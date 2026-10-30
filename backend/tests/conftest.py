"""Test configuration.

Force mock LLM mode so tests never touch the network, before app import.
"""
import os

os.environ.setdefault("LLM_MODE", "mock")
