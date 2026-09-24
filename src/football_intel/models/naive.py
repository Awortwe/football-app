"""Naive baseline: always predicts the league base rates from training data.

This is deliberately unrepresent and cheap: it is the "at least do this"
reference that every real model is compared against in evaluation.
"""

from __future__ import annotations

import pandas as pd

from .base import ModelResult, OutcomeModel, OutProbabilities, ensure_valid_probabilities


class NaiveBaseline(OutcomeModel):
    name = "Naive (league base rates)"

    def __init__(self) -> None:
        self.priors: dict[str, float] = {"H": 1 / 3, "D": 1 / 3, "A": 1 / 3}
        self.n_matches: int = 0

    def fit(self, matches: pd.DataFrame) -> "NaiveBaseline":
        df = matches[matches["FTR"].notna()]
        self.n_matches = int(len(df))
        counts = df["FTR"].value_counts()
        probs = {"H": 0.0, "D": 0.0, "A": 0.0}
        for k in probs:
            probs[k] = float(counts.get(k, 0) / len(df)) if len(df) else 1 / 3
        self.priors = ensure_valid_probabilities(probs)
        return self

    def predict_proba(self, home: str, away: str) -> OutProbabilities:
        return dict(self.priors)

    def predict(self, home: str, away: str) -> ModelResult:
        return ModelResult(
            home_team=home,
            away_team=away,
            match_date=None,
            model_name=self.name,
            probabilities=self.predict_proba(home, away),
            basis_notes=(
                f"Constant prediction from league win/draw/loss base rates "
                f"over {self.n_matches} training matches.",
            ),
        )