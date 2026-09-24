"""Hold-out evaluation of prematch and halftime predictions.

Method:
  * train/test split strictly by DATE (last N seasons held out),
  * every model is fit only on matches strictly before the test window,
  * prematch = forecast before kickoff; halftime = forecast at minute 45
    conditioned on the half-time score with the remaining goals scaled,
  * naive base rates act as the baseline every model must beat.

Writes ``data/bundled/evaluation_results.json`` used by the app and README.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..config import EVALUATION_JSON
from ..data.adapter import load_bundled_matches
from ..features.vector import build_feature_frame
from ..models.logistic import LogisticOutcomeModel
from ..models.naive import NaiveBaseline
from ..models.poisson import PoissonGoalsModel
from .metrics import evaluate_prediction_frame

PREMATCH = "prematch"
HALFTIME = "halftime"


def _test_window(matches: pd.DataFrame, n_test_seasons: int) -> tuple[list[str], pd.Timestamp]:
    seasons = sorted(matches["Season"].dropna().unique())
    test_seasons = list(seasons[-n_test_seasons:])
    first = matches[matches["Season"] == test_seasons[0]]["Date"].min()
    return test_seasons, pd.Timestamp(first)


def run_full_evaluation(
    matches: pd.DataFrame,
    n_test_seasons: int = 2,
) -> dict:
    matches = matches.sort_values(["Date", "HomeTeam"], kind="mergesort").reset_index(drop=True)
    test_seasons, test_start = _test_window(matches, n_test_seasons)
    train = matches[matches["Date"] < test_start]
    test = matches[matches["Date"] >= test_start].reset_index(drop=True).copy()

    print(f"[eval] train {len(train)} matches < {test_start.date()}, "
          f"test {len(test)} matches ({', '.join(test_seasons)})")

    # --- baselines and models fit only on training data ---------------------
    naive = NaiveBaseline().fit(train)
    poisson = PoissonGoalsModel().fit(train, reference_date=test_start - timedelta(days=1))

    hits = test["HTHG"].notna() & test["HTAG"].notna()
    print(f"[eval] halftime states available for {int(hits.sum())}/{len(test)} test matches")

    # --- per-match prediction columns (dicts of {H,D,A}) ---------------------
    poisson_ht_probs: list[dict | None] = []
    for _, m in test.iterrows():
        if not hits[m.name]:
            poisson_ht_probs.append(None)
            continue
        poisson_ht_probs.append(
            poisson.predict_inplay(
                m["HomeTeam"], m["AwayTeam"], minute=45,
                home_goals=int(m["HTHG"]), away_goals=int(m["HTAG"]),
            )["probabilities"]
        )

    result = pd.DataFrame(
        {
            "Date": test["Date"],
            "HomeTeam": test["HomeTeam"],
            "AwayTeam": test["AwayTeam"],
            "Season": test["Season"],
            "true": test["FTR"],
            "naive": [dict(naive.priors)] * len(test),
            "poisson_pre": [
                poisson.predict_proba(m["HomeTeam"], m["AwayTeam"]) for _, m in test.iterrows()
            ],
            "poisson_ht": poisson_ht_probs,
        }
    )

    # sklearn candidate, fitted on the same leak-free date split
    feat = build_feature_frame(matches)
    X_train = feat[feat["Date"] < test_start]
    X_test = feat[feat["Date"] >= test_start].reset_index(drop=True)
    logistic = LogisticOutcomeModel()
    logistic.fit(X_train, X_train["y"])
    result["logistic"] = [
        dict(r) for r in logistic.predict_proba_frame(X_test).to_dict("records")
    ]

    # --- score each scenario ----------------------------------------------
    scoring = result[result["true"].notna()].copy()
    scenario = {PREMATCH: {}, HALFTIME: {}}
    for model in ("naive", "poisson_pre", "logistic"):
        scenario[PREMATCH][model] = evaluate_prediction_frame(scoring[["true", model]], model)
    ht_scoring = scoring[scoring["poisson_ht"].notna()].copy()
    for model in ("naive", "poisson_pre", "poisson_ht"):
        scenario[HALFTIME][model] = evaluate_prediction_frame(ht_scoring[["true", model]], model)

    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": (
            "Time-split hold-out; models fit strictly before the test window. "
            "Prematch = forecast at kickoff. Halftime = forecast at minute 45 "
            "given the HT score (Poisson remaining-goals conditioning). "
            "Naive = league base rates from training data."
        ),
        "test_seasons": test_seasons,
        "test_matches": int(len(test)),
        "halftime_states": int(len(ht_scoring)),
        "train_matches": int(len(train)),
        "models": {
            "naive": {"name": "Naive (league base rates)", "role": "baseline"},
            "poisson_pre": {"name": "Poisson goals model", "role": "main model (prematch)"},
            "poisson_ht": {"name": "Poisson goals model", "role": "main model (halftime, in-play)"},
            "logistic": {"name": "Logistic regression (sklearn)", "role": "complexity check"},
        },
        "scenario": scenario,
    }


def _write(results: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys

    matches = load_bundled_matches()
    results = run_full_evaluation(matches)
    path = _write(results, sys.argv[1] if len(sys.argv) > 1 else EVALUATION_JSON)
    for scenario in (PREMATCH, HALFTIME):
        print(f"\n== {scenario.upper()} ==")
        for model, m in results["scenario"][scenario].items():
            print(
                f"  {model:12} n={m['n']:5d} logloss={m['log_loss']:.4f} "
                f"brier={m['brier']:.4f} cal|err|={m['calibration_abs_error']:.4f} "
                f"({m['calibration_n_bins']} bins)"
            )
    print(f"\nWrote {path}")