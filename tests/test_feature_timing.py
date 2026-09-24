"""Feature timing must be strictly leak-free: only matches strictly before the
target fixture may contribute to its features."""

from __future__ import annotations

import pandas as pd

from football_intel.features.timing import as_of_profile, prior_matches
from football_intel.features.tracker import Roller

SHARED_FIELDS = [
    "n_matches",
    "form_points",
    "gf_per_game",
    "ga_per_game",
    "home_gf_per_game",
    "home_ga_per_game",
    "away_gf_per_game",
    "away_ga_per_game",
    "last_goals_for",
    "last_goals_against",
]


def test_prior_matches_only_before(synthetic_matches):
    target = synthetic_matches.iloc[10]
    prior = prior_matches(synthetic_matches, target["HomeTeam"], target["Date"])
    assert prior["Date"].max() < pd.Timestamp(target["Date"])
    # the target match itself is not included
    ids = set(zip(prior["Date"], prior["HomeTeam"], prior["AwayTeam"]))
    assert (target["Date"], target["HomeTeam"], target["AwayTeam"]) not in ids


def test_profile_uses_only_prior_data(synthetic_matches):
    target = synthetic_matches.iloc[25]
    d = pd.Timestamp(target["Date"])
    p1 = as_of_profile(synthetic_matches, target["HomeTeam"], d)

    # Inject a fake 99-0 result ON the same date as the target: must be ignored.
    tampered = pd.concat(
        [synthetic_matches, pd.DataFrame([{
            "match_id": "TAMPER",
            "Date": d,
            "Season": target["Season"],
            "HomeTeam": target["HomeTeam"],
            "AwayTeam": "Fulham",
            "FTHG": 99, "FTAG": 0, "FTR": "H", "HTHG": 50, "HTAG": 0,
        }])],
        ignore_index=True,
    )
    p2 = as_of_profile(tampered, target["HomeTeam"], d)
    for f in SHARED_FIELDS:
        assert _same(p1, p2, f), (f, p1, p2)

    # A 99-0 result AFTER the target: must also be ignored.
    tampered2 = synthetic_matches.copy()
    later = tampered2.iloc[-1].copy()
    later["Date"] = pd.Timestamp(target["Date"]) + pd.Timedelta(days=2)
    later["HomeTeam"] = target["HomeTeam"]
    later["AwayTeam"] = "Fulham"
    later["FTHG"], later["FTAG"], later["HTAG"], later["HTHG"] = 99, 0, 0, 50
    later["match_id"] = "LATER"
    tampered2 = pd.concat([tampered2, pd.DataFrame([later])], ignore_index=True)
    p3 = as_of_profile(tampered2, target["HomeTeam"], d)
    for f in SHARED_FIELDS:
        assert _same(p1, p3, f), (f, p1, p3)


def test_roller_agrees_with_as_of_profile(synthetic_matches):
    """The incremental tracker must produce identical numbers to the one-shot
    scan, otherwise the two code paths could drift."""
    roller = Roller()
    df = synthetic_matches.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    checked = 0
    for _, row in df.iterrows():
        for team in (row["HomeTeam"], row["AwayTeam"]):
            snap = as_of_profile(df, team, row["Date"])
            rolled = roller.profile(team)
            if snap.n_matches == 0 and rolled.n_matches == 0:
                continue
            for f in SHARED_FIELDS:
                assert _same(snap, rolled, f), f"{f} mismatch for {team} @ {row['Date']}"
            # rest days must equal the date gap, by construction
            last = roller.last_played(team)
            if last is not None:
                assert (pd.Timestamp(row["Date"]) - last).days == snap.rest_days
        roller.observe(row)
        checked += 1
    assert checked >= 20


def test_new_team_has_no_history(synthetic_matches):
    d = pd.Timestamp(synthetic_matches["Date"].min())
    prior = prior_matches(synthetic_matches, "BrandNew FC", d)
    assert prior.empty
    r = Roller()
    prof = r.profile("BrandNew FC")
    assert prof.n_matches == 0


def test_goals_concede_consistency(synthetic_matches):
    for _, row in synthetic_matches.iterrows():
        home = as_of_profile(synthetic_matches, row["HomeTeam"], row["Date"])
        away = as_of_profile(synthetic_matches, row["AwayTeam"], row["Date"])
        if home.n_matches:
            assert home.gf_per_game >= 0
            assert home.home_gf_per_game is None or home.home_gf_per_game >= 0


def _same(a, b, field) -> bool:
    va = getattr(a, field)
    vb = getattr(b, field)
    if va is None or vb is None:
        return va is vb
    if isinstance(va, list):
        return list(va) == list(vb)
    return abs(float(va) - float(vb)) < 1e-9