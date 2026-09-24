"""Pytest configuration: put src/ on the path and build a tiny synthetic league."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SYNTH_TEAMS = ["Arsenal", "Chelsea", "Everton", "Fulham", "Leeds", "Newcastle"]


def make_synthetic_matches(
    n_seasons: int = 3,
    start_year: int = 2020,
    teams: list[str] | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Deterministic little league: round-robin, home and away legs each season.

    Scores come from latent team strengths + home advantage + noise, so the
    data has real structure without needing the bundled dataset.
    """
    rng = np.random.default_rng(seed)
    teams = teams or SYNTH_TEAMS
    strength = {t: float(s) for t, s in zip(teams, np.linspace(1.2, 0.6, len(teams)))}

    rows: list[dict] = []
    mid = len(teams) // 2
    for s in range(n_seasons):
        season_start = pd.Timestamp(f"{start_year + s}-08-15")
        season_label = f"{start_year + s}/{str(start_year + s + 1)[-2:]}"
        round_idx = 0
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                bh, ba = strength[home], strength[away]
                lh = max(0.05, bh * (1.0 if home > away else 1.0))
                la = max(0.05, ba * 0.8)
                fhg = int(rng.poisson(lh))
                fag = int(rng.poisson(la))
                hthg = int(rng.binomial(fhg, 0.6))
                htag = int(rng.binomial(fag, 0.55))
                date = season_start + pd.Timedelta(days=round_idx)
                rows.append(
                    {
                        "match_id": f"SYN{s}-{home}-{away}",
                        "Date": date,
                        "Season": season_label,
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "FTHG": fhg,
                        "FTAG": fag,
                        "FTR": "H" if fhg > fag else ("A" if fhg < fag else "D"),
                        "HTHG": hthg,
                        "HTAG": htag,
                    }
                )
                round_idx += 1
    df = pd.DataFrame(rows).sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    return df


@pytest.fixture
def synthetic_matches() -> pd.DataFrame:
    return make_synthetic_matches()


@pytest.fixture
def bundled_matches() -> pd.DataFrame:
    from football_intel.data.adapter import load_bundled_matches

    return load_bundled_matches()