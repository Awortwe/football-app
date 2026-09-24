"""Optional AI-generated wording for a narrative.

Everything must work without an AI service: unless the user enables it AND the
env vars are configured, this module is a no-op and returns the original
human-written text untouched. Any failure (missing key, timeout, bad response)
also quietly falls back to the original text.
"""

from __future__ import annotations

import os
import time

import requests

SYSTEM_PROMPT = (
    "You paraphrase football match-analysis text. Keep every observed fact verbatim, "
    "keep all numbers and the explicit 'estimated' qualifier. Output plain, short, "
    "professional English. Do not state that anything is certain."
)


def ai_supported() -> bool:
    """True when an OpenAI-compatible endpoint is configured."""
    return bool(os.environ.get("AI_API_KEY") and os.environ.get("AI_BASE_URL"))


def _endpoint() -> str:
    base = os.environ.get("AI_BASE_URL", "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _call(text: str, max_tokens: int = 300) -> str:
    resp = requests.post(
        _endpoint(),
        headers={"Authorization": f"Bearer {os.environ['AI_API_KEY']}"},
        json={
            "model": os.environ.get("AI_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def ai_worded(text: str, enabled: bool = False) -> str:
    """Return an AI paraphrase of ``text`` when ``enabled`` and configured.

    Falls back to the original text on any failure; never blocks for long.
    """
    if not enabled or not ai_supported():
        return text
    try:
        started = time.monotonic()
        out = _call(text)
        if time.monotonic() - started > 25:
            return text
        return out or text
    except Exception:  # noqa: BLE001 - optional feature, degrade silently
        return text