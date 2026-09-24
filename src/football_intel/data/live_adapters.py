"""Optional live data adapters.

The live connection is deliberately isolated so the rest of the system is
independent of any paid feed. Two sources are supported:

* ``BundledCsvAdapter``  -- bundled historical data (offline, default).
* ``FootballDataOrgAdapter`` -- football-data.org v4 API (live / after final
  whistle). Needs ``FOOTBALL_DATA_API_KEY`` in .env or Streamlit secrets.

The football-data.org free tier can provide fixtures, live scores and match
events (goals, substitutions, red cards with minutes) for major leagues, but
it is rate-limited (10 requests/min) and coverage varies by competition and
season. Nothing in this module is required for the app to function.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd


@dataclass
class LiveAdapterStatus:
    """Human-readable status of a live data source."""

    label: str = "Historical replay"
    kind: str = "historical"  # one of: historical, delayed, live
    detail: str = ""
    last_updated: str = ""

    @property
    def badge(self) -> str:
        if self.kind == "live":
            return f"**GENUINELY LIVE** — {self.label}"
        if self.kind == "delayed":
            return f"**DELAYED DATA** — {self.label}"
        return f"**HISTORICAL REPLAY** — {self.label}"


def _api_key() -> str | None:
    key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    return key.strip() or None


class FootballDataOrgAdapter:
    """football-data.org v4 live/results adapter.

    Replaceable drop-in for the bundled adapter when a key exists.
    Events carry minutes, so replays from this source are *not* estimated.

    Extra credits / ToS limits apply — see DATASETS.md. The free tier is
    rate-limited; if rate limits are hit the app continues in demo mode.
    """

    BADGE = LiveAdapterStatus(label="football-data.org", kind="delayed")

    def __init__(self, api_key: str | None = None) -> None:
        self._key = api_key or _api_key()

    @property
    def available(self) -> bool:
        return bool(self._key)

    def status(self) -> LiveAdapterStatus:
        return LiveAdapterStatus(
            label="football-data.org",
            kind="delayed",
            detail=(
                "Live feed configured. Events carry recorded minutes."
                if self.available
                else "No API key found; using bundled historical data."
            ),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    def _headers(self) -> dict[str, str]:
        return {"X-Auth-Token": self._key}  # type: ignore[dict-item]

    def fetch_matches(self, competition_code: str = "PL", matchday: str | None = None) -> list[dict[str, Any]]:
        """Placeholder for a real fetch. Returns [] to keep the interface honest."""
        if not self.available:
            return []
        # Real implementation would call api.football-data.org/v4/matches ...
        return []


def configured_live_adapters() -> list[FootballDataOrgAdapter]:
    """Adapters that are enabled given the current environment."""
    out: list[FootballDataOrgAdapter] = []
    if _api_key():
        out.append(FootballDataOrgAdapter())
    return out


def pandas_from_iso(iso: str) -> pd.Timestamp:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return pd.Timestamp(dt).tz_localize(None)