"""Evaluation correctness: scoring rules, calibration, date-split safety."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_intel.evaluation.metrics import (
    brier_multi,
    calibration_table,
    evaluate_prediction_frame,
    log_loss,
)
from football_intel.evaluation.run import run_full_evaluation
from football_intel.features.vector import build_feature_frame


def _frame(n=200):
    rng = np.random.default_rng(1)
    true = pd.Series(rng.choice(["H", "D", "A"], size=n))
    probs = rng.dirichlet([0.45, 0.25, 0.30], size=n)
    return pd.DataFrame(probs, columns=["H", "D", "A"]), true


def test_log_loss_perfect_is_zero():
    probs, true = _frame()
    exact = pd.DataFrame(
        [[1.0, 0.0, 0.0] if t == "H" else ([0.0, 1.0, 0.0] if t == "D" else [0.0, 0.0, 1.0])
         for t in true],
        columns=["H", "D", "A"],
    )
    ll = log_loss(true, exact)
    assert ll < 1e-3  # ~0 except for the 1e-6 clipping floor


def test_log_loss_wrong_confident_is_large():
    probs, true = _frame()
    wrong = pd.DataFrame(
        [[0.0, 0.0, 1.0]] * len(true), columns=["H", "D", "A"]
    )
    ll = log_loss(true, wrong)
    assert ll > 5.0  # always-predict-away against a 1/3 base rate is very wrong


def test_brier_matches_manual():
    probs, true = _frame()
    manual = np.mean(
        [
            np.mean((p - np.eye(3)[{"H": 0, "D": 1, "A": 2}[t]]) ** 2)
            for p, t in zip(probs.to_numpy(), true)
        ]
    )
    assert abs(brier_multi(true, probs) - manual) < 1e-12


def test_calibration_table_structure():
    probs, true = _frame()
    cal = calibration_table(true, probs, bins=5)
    assert list(cal.columns) == [
        "outcome",
        "bin",
        "n",
        "mean_predicted",
        "observed_rate",
        "abs_error",
    ]
    assert len(cal) > 0
    # used as a scoring metric on evaluate_prediction_frame
    res = evaluate_prediction_frame(
        pd.DataFrame({"true": true, "m": probs.to_dict("records")}), "m"
    )
    assert res["n"] == len(true)
    assert 0 <= res["brier"] <= 1
    assert res["log_loss"] > 0
    assert res["calibration_n_bins"] > 0


def test_date_split_disjoint(synthetic_matches):
    cutoff = synthetic_matches["Date"].quantile(0.5)
    train = synthetic_matches[synthetic_matches["Date"] < cutoff]
    test = synthetic_matches[synthetic_matches["Date"] >= cutoff]
    assert not set(train["match_id"]) & set(test["match_id"])
    assert train["Date"].max() < test["Date"].min()


def test_full_evaluation_runs(synthetic_matches):
    results = run_full_evaluation(synthetic_matches)
    for scenario in ("prematch", "halftime"):
        assert scenario in results["scenario"]
        for model, m in results["scenario"][scenario].items():
            assert m["n"] > 0
            assert "log_loss" in m and "brier" in m
    # the main model must be recorded and be at least as good as random
    prem = results["scenario"]["prematch"]
    assert "poisson_pre" in prem
    assert prem["poisson_pre"]["log_loss"] < np.log(3) + 0.01


def test_feature_frame_has_no_future_leakage(synthetic_matches):
    """Flipping a later result must not change earlier feature rows."""
    base = build_feature_frame(synthetic_matches)
    flipped = synthetic_matches.copy()
    last = flipped.index[-1]
    flipped.at[last, "FTR"] = ({"H": "A", "A": "H", "D": "H"})[flipped.at[last, "FTR"]]
    flipped_frame = build_feature_frame(flipped)

    base = base.reset_index(drop=True)
    flipped_frame = flipped_frame.reset_index(drop=True)
    cut = base["Date"].max() - pd.Timedelta(days=30)
    a = base[base["Date"] < cut].sort_values("match_id").reset_index(drop=True)
    b = flipped_frame[flipped_frame["Date"] < cut].sort_values("match_id").reset_index(drop=True)
    assert len(a) == len(b)
    assert a["match_id"].equals(b["match_id"])
    for col in a.columns:
        if col == "y":
            continue
        assert a[col].equals(b[col]), f"leak in feature column {col}"


def test_poisson_beats_naive_halftime(synthetic_matches):
    """Conditioning on the score must beat the constant baseline at halftime."""
    results = run_full_evaluation(synthetic_matches)
    ht = results["scenario"]["halftime"]
    assert ht["poisson_ht"]["log_loss"] < ht["naive"]["log_loss"]
    assert ht["poisson_ht"]["brier"] < ht["naive"]["brier"]