"""Shared model interfaces and the canonical probability container."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

Home = "H"
Draw = "D"
Away = "A"

OutProbabilities = dict[str, float]  # {"H": p, "D": p, "A": p}, sums to 1


def ensure_valid_probabilities(probs: OutProbabilities) -> OutProbabilities:
    """Clip, renormalize and verify a 1X2 probability vector."""
    for k in ("H", "D", "A"):
        if k not in probs:
            raise ValueError(f"missing outcome probability for {k}")
    arr = np.clip([probs[k] for k in ("H", "D", "A")], 1e-6, 1.0)
    total = arr.sum()
    if not np.isfinite(total) or total == 0:
        raise ValueError("probabilities must be finite and non-zero")
    return dict(zip(("H", "D", "A"), (arr / total).tolist()))


@dataclass(frozen=True)
class ModelResult:
    """A probabilistic forecast for one fixture.

    ``probabilities`` always sum to 1.0. ``basis_notes`` records exactly which
    data the forecast is allowed to use (the story behind the numbers); it is
    plain text, never an invented fact.
    """

    home_team: str
    away_team: str
    match_date: date
    model_name: str
    probabilities: OutProbabilities
    basis_notes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    data_as_of: date = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "probabilities", ensure_valid_probabilities(self.probabilities)
        )
        if self.data_as_of is None:
            object.__setattr__(self, "data_as_of", self.match_date)
        if sum(self.probabilities.values()) < 0.999999 or sum(
            self.probabilities.values()
        ) > 1.000001:
            raise ValueError("probabilities must sum to 1")

    def label(self, outcome: str) -> float:
        return self.probabilities[outcome]

    def most_likely(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    def as_tuple(self) -> tuple[float, float, float]:
        return self.probabilities["H"], self.probabilities["D"], self.probabilities["A"]


class OutcomeModel:
    """Base class contract: fit on history, predict probabilities for a fixture."""

    name: str = "base"

    def fit(self, matches) -> "OutcomeModel":
        raise NotImplementedError

    def predict_proba(self, home: str, away: str) -> OutProbabilities:
        raise NotImplementedError

    def predict(self, home: str, away: str) -> ModelResult:
        raise NotImplementedError