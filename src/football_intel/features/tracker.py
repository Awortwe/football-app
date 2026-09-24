"""Incremental, leakage-free rolling aggregates.

The key invariant: ``observe()`` must only be called *after* a match's
features have been consumed. Clubs are stratified per team and never leak the
target fixture or any later fixture, because the tracker has not yet seen it.

``profile()`` produces the same numbers as ``as_of_profile`` (see timing.py);
a test asserts the two agree so they cannot drift apart.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import FORM_WINDOW
from .timing import TeamProfile


@dataclass
class _TeamAgg:
    n: int = 0
    gf: float = 0.0           # sum of goals scored
    ga: float = 0.0           # sum of goals conceded
    home_n: int = 0
    home_gf: float = 0.0
    home_ga: float = 0.0
    away_n: int = 0
    away_gf: float = 0.0
    away_ga: float = 0.0
    last_date: pd.Timestamp | None = None
    recent_gf: deque = None  # type: ignore[assignment]
    recent_ga: deque = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.recent_gf is None:
            self.recent_gf = deque()
            self.recent_ga = deque()


class Roller:
    """Rolling per-team aggregates; update after each consumed match."""

    def __init__(self, window: int = FORM_WINDOW) -> None:
        self.window = window
        self.aggs: dict[str, _TeamAgg] = {}

    def observe(self, row: pd.Series) -> None:
        home, away = row["HomeTeam"], row["AwayTeam"]
        gf_h, ga_h = float(row["FTHG"]), float(row["FTAG"])
        date = pd.Timestamp(row["Date"])
        self._add(home, gf_h, ga_h, home=True, date=date)
        self._add(away, ga_h, gf_h, home=False, date=date)

    def _add(self, team: str, gf: float, ga: float, home: bool, date: pd.Timestamp) -> None:
        agg = self.aggs.setdefault(team, _TeamAgg())
        agg.n += 1
        agg.gf += gf
        agg.ga += ga
        agg.recent_gf.append(int(gf))
        agg.recent_ga.append(int(ga))
        while len(agg.recent_gf) > self.window:
            agg.recent_gf.popleft()
            agg.recent_ga.popleft()
        if home:
            agg.home_n += 1
            agg.home_gf += gf
            agg.home_ga += ga
        else:
            agg.away_n += 1
            agg.away_gf += gf
            agg.away_ga += ga
        agg.last_date = max(agg.last_date, date) if agg.last_date is not None else date

    def profile(self, team: str) -> TeamProfile:
        agg = self.aggs.get(team)
        if agg is None or agg.n == 0:
            return TeamProfile(
                team=team, as_of_date=None, n_matches=0, sampled_period_days=0,
                form_points=float("nan"), gf_per_game=float("nan"),
                ga_per_game=float("nan"), home_gf_per_game=None, home_ga_per_game=None,
                away_gf_per_game=None, away_ga_per_game=None,
                last_goals_for=[], last_goals_against=[], rest_days=None,
            )

        def _rate(x: float, n: int) -> float | None:
            return float(x / n) if n else None

        gf = list(agg.recent_gf)
        ga = list(agg.recent_ga)
        pts = sum(3.0 if g > a else (1.0 if g == a else 0.0) for g, a in zip(gf, ga))
        return TeamProfile(
            team=team,
            as_of_date=None,
            n_matches=agg.n,
            sampled_period_days=0,
            form_points=pts / len(gf) if gf else float("nan"),
            gf_per_game=_rate(agg.gf, agg.n),
            ga_per_game=_rate(agg.ga, agg.n),
            home_gf_per_game=_rate(agg.home_gf, agg.home_n),
            home_ga_per_game=_rate(agg.home_ga, agg.home_n),
            away_gf_per_game=_rate(agg.away_gf, agg.away_n),
            away_ga_per_game=_rate(agg.away_ga, agg.away_n),
            last_goals_for=gf,
            last_goals_against=ga,
            rest_days=None,
        )

    def last_played(self, team: str) -> pd.Timestamp | None:
        agg = self.aggs.get(team)
        return agg.last_date if agg else None

    def teams(self) -> list[str]:
        return list(self.aggs)


class EloBank:
    """K-rating-style three-outcome Elo, updated only after feature use."""

    def __init__(self, start: float = 1500.0, k: float = 20.0, home_adv: float = 100.0) -> None:
        self.start = start
        self.k = k
        self.home_adv = home_adv
        self.ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.start)

    def _expected(self, dr: float) -> float:
        return 1.0 / (1.0 + 10 ** (-dr / 400.0))

    def diff(self, home: str, away: str) -> float:
        return self.rating(home) + self.home_adv - self.rating(away)

    def observe(self, row: pd.Series) -> None:
        home, away = row["HomeTeam"], row["AwayTeam"]
        r_h, r_a = self.rating(home), self.rating(away)
        dr = r_h + self.home_adv - r_a
        we_h = self._expected(dr)          # expected points for home (W=1, D=0.5)
        result = row["FTR"]
        w_h = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)
        w_a = 1.0 - w_h
        self.ratings[home] = r_h + self.k * (w_h - we_h)
        self.ratings[away] = r_a + self.k * (w_a - (1.0 - we_h))