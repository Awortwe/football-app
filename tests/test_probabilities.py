"""Probability totals and bounds must always hold: every 1X2 estimate sums to
100% and each class sits in (0, 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_intel.models.base import ensure_valid_probabilities
from football_intel.models.naive import NaiveBaseline
from football_intel.models.poisson import PoissonGoalsModel
from tests.conftest import make_synthetic_matches


def _fit(matches: pd.DataFrame) -> PoissonGoalsModel:
    cutoff = matches["Date"].quantile(0.6)
    return PoissonGoalsModel().fit(matches, reference_date=cutoff)


def test_prematch_sums_to_one(synthetic_matches):
    m = _fit(synthetic_matches)
    for _, row in synthetic_matches.iterrows():
        p = m.predict_proba(row["HomeTeam"], row["AwayTeam"])
        assert abs(sum(p.values()) - 1.0) < 1e-9
        for k, v in p.items():
            assert 0 < v < 1


def test_inplay_sums_to_one(synthetic_matches):
    m = _fit(synthetic_matches)
    cases = [
        dict(minute=30, home_goals=0, away_goals=0),
        dict(minute=45, home_goals=1, away_goals=1),
        dict(minute=60, home_goals=2, away_goals=0),
        dict(minute=88, home_goals=3, away_goals=2),
    ]
    for kw in cases:
        res = m.predict_inplay("Arsenal", "Chelsea", **kw)
        p = res["probabilities"]
        assert abs(sum(p.values()) - 1.0) < 1e-9, kw
        assert all(0 < v < 1 for v in p.values())


def test_inplay_starts_like_prematch(synthetic_matches):
    m = _fit(synthetic_matches)
    prematch = m.predict_proba("Arsenal", "Chelsea")
    kickoff = m.predict_inplay("Arsenal", "Chelsea", minute=0, home_goals=0, away_goals=0)[
        "probabilities"
    ]
    for k in ("H", "D", "A"):
        assert abs(kickoff[k] - prematch[k]) < 1e-6


def test_naive_sums_to_one(synthetic_matches):
    half = synthetic_matches.iloc[: len(synthetic_matches) // 2]
    naive = NaiveBaseline().fit(half)
    for _ in range(10):
        p = naive.predict_proba("Arsenal", "Chelsea")
        assert abs(sum(p.values()) - 1.0) < 1e-9


def test_ensure_valid_probabilities(synthetic_matches):
    fixed = ensure_valid_probabilities({"H": 0.6, "D": 0.6, "A": 0.1})
    assert abs(sum(fixed.values()) - 1.0) < 1e-6
    import pytest

    with pytest.raises(ValueError):
        ensure_valid_probabilities({"H": 0.5, "D": 0.5})  # missing "A"


def test_full_seasons_probabilities(synthetic_matches, bundled_matches):
    """A broad, cheap sanity sweep on real data (one fit, no evaluation)."""
    m = PoissonGoalsModel().fit(bundled_matches, reference_date="2023-08-01")
    sample = bundled_matches.sample(n=50, random_state=3)
    for _, row in sample.iterrows():
        p = m.predict_proba(row["HomeTeam"], row["AwayTeam"])
        assert abs(sum(p.values()) - 1.0) < 1e-9
        assert all(0 < v < 1 for v in p.values())
    # distribution sanity: home advantage should push most home probabilities up
    mean_h = np.mean(
        [m.predict_proba(r["HomeTeam"], r["AwayTeam"])["H"] for _, r in sample.iterrows()]
    )
    assert mean_h > 0.35