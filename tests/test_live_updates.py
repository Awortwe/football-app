"""Match-event updates must move estimates in sensible directions and the
replay timeline must never present estimated minutes as recorded facts."""

from __future__ import annotations

import pandas as pd
import pytest

from football_intel.live.estimation import estimate_timeline
from football_intel.live.replay import build_replay_timeline
from football_intel.models.poisson import PoissonGoalsModel


def _model(matches):
    cutoff = matches["Date"].quantile(0.6)
    return PoissonGoalsModel().fit(matches, reference_date=cutoff)


def _row_with_score(matches, fhg: int, fag: int, hthg: int, htag: int):
    base = matches.iloc[0].copy()
    base["FTHG"], base["FTAG"] = fhg, fag
    base["HTHG"], base["HTAG"] = hthg, htag
    return base


def test_replay_has_all_structural_events(synthetic_matches):
    row = _row_with_score(synthetic_matches, 3, 1, 1, 1)
    tl = build_replay_timeline(row)
    kinds = [e.kind for e in tl.events]
    assert kinds[0] == "kickoff"
    assert kinds.count("goal") == 4          # 3 + 1
    assert "half_time" in kinds
    assert "full_time" in kinds
    last = tl.events[-1]
    assert last.kind == "full_time" and (last.score_home, last.score_away) == (3, 1)
    # monotonic minutes, correct cumulative score at each point
    prev = -1
    for e in tl.events:
        assert e.minute >= prev
        prev = e.minute


def test_replay_goal_minutes_are_flagged_as_estimates(synthetic_matches):
    row = _row_with_score(synthetic_matches, 2, 0, 1, 0)
    tl = build_replay_timeline(row)
    for e in tl.events:
        if e.kind == "goal":
            assert e.estimated_time is True and e.is_observed is True
            assert e.minute < 45 or e.minute > 45
    # half-time and full-time are hard facts
    for k in ("half_time", "full_time", "kickoff"):
        e = next(x for x in tl.events if x.kind == k)
        assert e.estimated_time is False and e.is_observed is True


def test_goals_in_halves_match_ht_and_ft(synthetic_matches):
    row = _row_with_score(synthetic_matches, 4, 2, 2, 1)
    tl = build_replay_timeline(row)
    ht = next(x for x in tl.events if x.kind == "half_time")
    assert (ht.score_home, ht.score_away) == (2, 1)
    # only first-half goals appear before HT
    before_ht = [e for e in tl.events if e.minute <= 45]
    goals_before = [e for e in before_ht if e.kind == "goal"]
    assert len(goals_before) == 3
    assert goals_before[-1].score_home == 2 and goals_before[-1].score_away == 1


def _point_probs(points):
    return [(p.minute, p.home_win, p.draw, p.away_win) for p in points]


def test_goal_for_home_raises_home_win_probability(synthetic_matches):
    m = _model(synthetic_matches)
    row = _row_with_score(synthetic_matches, 1, 0, 1, 0)
    tl = build_replay_timeline(row)
    pts = estimate_timeline(m, tl)
    kickoff = pts[0]
    first_goal = next(p for p in pts if p.event == "goal")
    assert first_goal.team == tl.home_team
    assert first_goal.home_win >= kickoff.home_win
    # at full time the recorded winner is (near) certain
    ft = pts[-1]
    assert ft.home_win > 0.999


def test_away_lead_suppresses_home_win(synthetic_matches):
    m = _model(synthetic_matches)
    row = _row_with_score(synthetic_matches, 0, 2, 0, 1)
    tl = build_replay_timeline(row)
    pts = estimate_timeline(m, tl)
    ht = next(p for p in pts if p.event == "half_time")
    assert ht.away_win > ht.home_win


def test_red_card_moves_estimates_against_sanctioned_team(synthetic_matches):
    m = _model(synthetic_matches)
    base = m.predict_inplay("Arsenal", "Chelsea", minute=60, home_goals=0, away_goals=0)
    away_red = m.predict_inplay(
        "Arsenal", "Chelsea", minute=60, home_goals=0, away_goals=0, away_red_cards=1
    )
    home_red = m.predict_inplay(
        "Arsenal", "Chelsea", minute=60, home_goals=0, away_goals=0, home_red_cards=1
    )
    assert away_red["probabilities"]["A"] < base["probabilities"]["A"]
    assert home_red["probabilities"]["H"] < base["probabilities"]["H"]


def test_estimate_timeline_returns_chart_frame(synthetic_matches):
    m = _model(synthetic_matches)
    row = _row_with_score(synthetic_matches, 2, 2, 1, 2)
    tl = build_replay_timeline(row)
    pts = estimate_timeline(m, tl)
    assert len(pts) == tl.n_events
    for p in pts:
        assert abs(p.home_win + p.draw + p.away_win - 1.0) < 1e-9


def test_no_ht_data_skips_midfield(synthetic_matches):
    row = synthetic_matches.iloc[0].copy()
    row.drop(labels=["HTHG", "HTAG"], inplace=True)
    tl = build_replay_timeline(row)
    kinds = [e.kind for e in tl.events]
    assert "half_time" not in kinds
    assert any(e.kind == "note" for e in tl.events)