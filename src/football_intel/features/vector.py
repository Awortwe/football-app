"""Sklearn feature vectors built strictly from information before kickoff.

``build_eval_frame`` walks matches chronologically, computes each match's
feature row from the pre-match state, and only then updates the rolling
aggregates and Elo. This makes the no-leakage guarantee structural rather than
hoped-for.
"""

from __future__ import annotations

import pandas as pd

from .tracker import EloBank, Roller

FEATURE_COLUMNS = [
    "elo_diff",
    "h_form",
    "a_form",
    "h_gf_rate",
    "h_ga_rate",
    "a_gf_rate",
    "a_ga_rate",
    "h_home_gf_rate",
    "a_away_gf_rate",
    "h_rest",
    "a_rest",
]


def feature_row(roller: Roller, elo: EloBank, row: pd.Series) -> dict:
    home, away = row["HomeTeam"], row["AwayTeam"]
    hp, ap = roller.profile(home), roller.profile(away)
    h_last, a_last = roller.last_played(home), roller.last_played(away)
    match_ts = pd.Timestamp(row["Date"])

    def _rest(last) -> float:
        return float((match_ts - last).days) if last is not None else float("nan")

    return {
        "elo_diff": float(elo.diff(home, away)),
        "h_form": hp.form_points,
        "a_form": ap.form_points,
        "h_gf_rate": hp.gf_per_game,
        "h_ga_rate": hp.ga_per_game,
        "a_gf_rate": ap.gf_per_game,
        "a_ga_rate": ap.ga_per_game,
        "h_home_gf_rate": hp.home_gf_per_game,
        "a_away_gf_rate": ap.away_gf_per_game,
        "h_rest": _rest(h_last),
        "a_rest": _rest(a_last),
    }


def build_feature_frame(
    matches: pd.DataFrame,
    window: int = 6,
    elo_k: float = 20.0,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Chronological pass producing one feature row per match.

    If ``start_date`` is given, only rows at/after it are returned (but the
    pass still starts from the beginning of history to keep Elo cumulative).
    The resulting frame has columns FEATURE_COLUMNS + metadata + ``y`` (FTR).
    """
    roller = Roller(window)
    elo = EloBank(k=elo_k)
    rows: list[dict] = []
    ordered = matches.sort_values(["Date", "HomeTeam"], kind="mergesort")
    for _, row in ordered.iterrows():
        date = pd.Timestamp(row["Date"])
        if (start_date is None or date >= pd.Timestamp(start_date)) and (
            end_date is None or date <= pd.Timestamp(end_date)
        ):
            feat = feature_row(roller, elo, row)
            feat.update(
                {
                    "Date": date,
                    "HomeTeam": row["HomeTeam"],
                    "AwayTeam": row["AwayTeam"],
                    "Season": row.get("Season"),
                    "match_id": row.get("match_id"),
                    "y": row["FTR"],
                }
            )
            rows.append(feat)
        roller.observe(row)
        elo.observe(row)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["Date", "HomeTeam", "AwayTeam", "Season", "match_id", "y"])