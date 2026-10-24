"""Observation scrubbing. Kept free of simulator imports so the brain/MCP side can use it cheaply."""
from __future__ import annotations

from motormem import config


def strip_privileged(raw: dict) -> dict:
    """Return a copy of ``raw`` without any key for which ``config.is_privileged_key`` is true (Property 4)."""
    return {k: v for k, v in raw.items() if not config.is_privileged_key(k)}
