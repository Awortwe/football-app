"""Well-defined scoring rules and calibration summaries.

Conventions:
  * ``y_true``  -- array of outcomes in {"H","D","A"}.
  * ``probs``   -- dict-like {outcome: probability} OR a DataFrame
                   [index, H, D, A]. Probabilities are clipped before scoring.
  * "probability" means estimated probability (which can be wrong); observed
    frequency is only ever computed from realised results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import OUTCOMES

_EPS = 1e-6


def log_loss(y_true, probs: dict | pd.DataFrame) -> float:
    """Multiclass log loss over {H,D,A} with clipping."""
    if isinstance(probs, dict):
        probs = pd.DataFrame([probs], index=[0])
    y = pd.Series(list(y_true))
    probs = probs[OUTCOMES].clip(_EPS, 1 - _EPS)
    probs = probs.div(probs.sum(axis=1), axis=0)
    idx = y.map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    chosen = probs.to_numpy()[np.arange(len(y)), idx]
    return float(np.mean(-np.log(np.clip(chosen, _EPS, None))))


def brier_multi(y_true, probs: dict | pd.DataFrame) -> float:
    """Multiclass Brier score: mean over classes of (p - I)^2."""
    if isinstance(probs, dict):
        probs = pd.DataFrame([probs], index=[0])
    mat = probs[OUTCOMES].clip(_EPS, 1 - _EPS).to_numpy()
    idx = y_true.map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    onehot = np.zeros_like(mat)
    onehot[np.arange(len(idx)), idx] = 1.0
    return float(np.mean((mat - onehot) ** 2))


def calibration_table(
    y_true: pd.Series,
    probs: dict | pd.DataFrame,
    bins: int = 5,
) -> pd.DataFrame:
    """Per-outcome calibration: mean predicted probability vs observed rate.

    Returns a DataFrame with columns: outcome, bin, n, mean_predicted,
    observed_rate, abs_error (|observed - mean_predicted|).
    Empty bins (constant predictions like the naive baseline) are omitted,
    which is itself a warning that the model cannot express uncertainty.
    """
    if isinstance(probs, dict):
        probs = pd.DataFrame([probs] * len(y_true), index=y_true.index)
    probs = probs[OUTCOMES].clip(_EPS, 1 - _EPS)
    y = pd.Series(list(y_true), index=y_true.index)
    rows: list[dict] = []
    for outcome in OUTCOMES:
        col = probs[outcome]
        unique = col.unique()
        if len(unique) <= 2:
            continue
        # Equal-frequency bins on predicted probability.
        q = col.quantile(np.linspace(0, 1, bins + 1)).drop_duplicates()
        if len(q) < 3:
            continue
        labels = pd.cut(col, bins=q, include_lowest=True, duplicates="drop")
        grp = pd.DataFrame({"p": col, "o": (y == outcome).astype(int), "b": labels})
        for b, sub in grp.groupby("b", observed=True):
            mean_pred = float(sub["p"].mean())
            obs = float(sub["o"].mean())
            rows.append(
                {
                    "outcome": outcome,
                    "bin": str(b),
                    "n": int(len(sub)),
                    "mean_predicted": mean_pred,
                    "observed_rate": obs,
                    "abs_error": abs(obs - mean_pred),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["outcome", "bin", "n", "mean_predicted", "observed_rate", "abs_error"]
        )
    return pd.DataFrame(rows).sort_values(["outcome", "bin"]).reset_index(drop=True)


def evaluate_prediction_frame(
    result: pd.DataFrame,
    model_col: str,
    true_col: str = "true",
) -> dict:
    """Score one model's predictions stored in ``result[model_col]``.

    ``result`` has columns: one truth column ('true') and one column per model
    holding dicts {H: p, D: p, A: p}.
    """
    probs = pd.json_normalize(result[model_col].tolist())[OUTCOMES]
    y_true = result[true_col]
    cal = calibration_table(y_true, probs)
    return {
        "model": model_col,
        "n": int(len(y_true)),
        "log_loss": log_loss(y_true, probs),
        "brier": brier_multi(y_true, probs),
        "calibration_abs_error": (
            float(cal["abs_error"].mean()) if len(cal) else float("nan")
        ),
        "calibration_n_bins": int(len(cal)),
        "calibration": cal.to_dict("records"),
    }