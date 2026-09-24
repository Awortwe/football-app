"""Feature timing utilities.

The single most important guarantee in this project: **every feature for a
match may only use rows with ``Date < as_of_date``**. The helpers below exist
so that no caller can accidentally leak information from the match itself or
later matches. ``profile_gym`` is a bake-off style verification used in tests
and reused everywhere features are built.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..config import FORM_WINDOW


def prior_matches(
    matches: pd.DataFrame,
    team: str,
    as_of_date,
    opponent: str | None = None,
) -> pd.DataFrame:
    """All finished matches played by ``team`` strictly before ``as_of_date``.

    ``ass_of_date`` may be a ``date``/``datetime``/``Timestamp``.
    If ``opponent`` is given, only matches against that opponent are returned.
    The returned frame never references the target fixture itself.
    """
    cutoff = pd.Timestamp(as_of_date)
    mask = matches["Date"] < cutoff
    mask &= (matches["HomeTeam"] == team) | (matches["AwayTeam"] == team)
    if opponent is not None:
        mask &= (matches["HomeTeam"] == opponent) | (matches["AwayTeam"] == opponent)
    out = matches.loc[mask].sort_values("Date")
    return out.reset_index(drop=True)


@dataclass
class TeamProfile:
    """Observable team tendencies known strictly before a target match.

    Every field that describes performance is derived from finished matches
    with ``Date < as_of_date``. Nothing here is an estimate of the future —
    that is the models' job. ``n_matches`` is how many finished matches the
    profile is based on; ``sampled_period`` the span in days.
    """

    team: str
    as_of_date: date
    n_matches: int
    sampled_period_days: int
    form_points: float                # mean points / game, last `window` games
    gf_per_game: float                # goals scored / game
    ga_per_game: float                # goals conceded / game
    home_gf_per_game: float | None    # None when no home games sampled
    home_ga_per_game: float | None
    away_gf_per_game: float | None
    away_ga_per_game: float | None
    last_goals_for: list[int]         # recent GF, oldest -> newest
    last_goals_against: list[int]
    rest_days: int | None             # days since the team last played

    @property
    def recent_form(self) -> str:
        """Compact 'W D L' string for narration, oldest first."""
        return _form_string(self.last_goals_for, self.last_goals_against)


def _form_string(gf: list[int], ga: list[int]) -> str:
    parts: list[str] = []
    for h, a in zip(gf, ga):
        parts.append("W" if h > a else ("L" if h < a else "D"))
    return " ".join(parts)


def _homestatus(matches: pd.DataFrame, team: str, i: int) -> bool:
    return matches.iloc[i]["HomeTeam"] == team


def as_of_profile(
    matches: pd.DataFrame,
    team: str,
    as_of_date,
    window: int = FORM_WINDOW,
) -> TeamProfile:
    """Build a TeamProfile for ``team`` using only matches before the date.

    Args:
        matches: normalized dataframe (see data/adapter.load_bundled_matches).
        team: team name.
        as_of_date: cutoff; only rows with Date < as_of_date are used.
        window: how many of the most recent games count for "form".

    Fields that are not computable (e.g., home stats when the side is a brand
    new club with no home games) are returned as ``None`` and callers must
    treat None as "no data", never as zero.
    """
    prior = prior_matches(matches, team, as_of_date)
    if prior.empty:
        return TeamProfile(
            team=team,
            as_of_date=pd.Timestamp(as_of_date).date(),
            n_matches=0,
            sampled_period_days=0,
            form_points=float("nan"),
            gf_per_game=float("nan"),
            ga_per_game=float("nan"),
            home_gf_per_game=None,
            home_ga_per_game=None,
            away_gf_per_game=None,
            away_ga_per_game=None,
            last_goals_for=[],
            last_goals_against=[],
            rest_days=None,
        )

    us = np.where(prior["HomeTeam"] == team, prior["FTHG"], prior["FTAG"])
    opp = np.where(prior["HomeTeam"] == team, prior["FTAG"], prior["FTHG"])
    home_flags = np.where(prior["HomeTeam"] == team, True, False)

    gf_all = us.astype(float)
    ga_all = opp.astype(float)

    recent = slice(-window, None) if window > 0 else slice(None)
    gf_recent = list(int(x) for x in gf_all[recent])
    ga_recent = list(int(x) for x in ga_all[recent])
    gf_arr = np.asarray(gf_recent, dtype=float)
    ga_arr = np.asarray(ga_recent, dtype=float)
    points_arr = np.where(
        gf_arr > ga_arr, 3.0, np.where(gf_arr < ga_arr, 0.0, 1.0)
    )
    form_points = float(points_arr.mean()) if points_arr.size else 0.0

    span = (
        int((prior["Date"].max() - prior["Date"].min()).days) if len(prior) > 1 else 0
    )
    home_idx = np.where(home_flags)[0]
    away_idx = np.where(~home_flags)[0]

    def _mean(x) -> float | None:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else None

    return TeamProfile(
        team=team,
        as_of_date=pd.Timestamp(as_of_date).date(),
        n_matches=int(len(prior)),
        sampled_period_days=span,
        form_points=float(points_arr.mean()) if points_arr.size else 0.0,
        gf_per_game=_mean(gf_all),
        ga_per_game=_mean(ga_all),
        home_gf_per_game=_mean(gf_all[home_idx]),
        home_ga_per_game=_mean(ga_all[home_idx]),
        away_gf_per_game=_mean(gf_all[away_idx]),
        away_ga_per_game=_mean(ga_all[away_idx]),
        last_goals_for=gf_recent,
        last_goals_against=ga_recent,
        rest_days=int((pd.Timestamp(as_of_date) - prior["Date"].max()).days),
    )


def team_name_uniform_check(matches: pd.DataFrame) -> None:
    """Assert a normalized dataframe contains no near-duplicate team names."""
    import re

    norms = {
        re.sub(r"[^a-z]", "", n.lower()) for n in set(matches["HomeTeam"]) | set(matches["AwayTeam"])
    }
    if len(norms) != len(set(norms)):
        raise ValueError("dataset has near-duplicate team spellings")


def profile_gym(matches: pd.DataFrame):
    """Iterate matches in chronological order, building profiles without reuse.

    On every iteration the profiles are computed *from scratch over the past*,
    exactly as production does. This is cubic, so it is reserved for tests and
    small runs; production uses ``as_of_profile`` on a single cutoff.

    Yields: (match_row, home_profile, away_profile).
    """
    for _, row in matches.sort_values("Date").iterrows():
        h = as_of_profile(matches, row["HomeTeam"], row["Date"])
        a = as_of_profile(matches, row["AwayTeam"], row["Date"])
        yield row, h, a