"""scikit-learn logistic regression candidate model.

Used in evaluation to test whether extra complexity beats the understandable
Poisson baseline on held-out matches. Expected inputs are feature frames from
``features.vector.build_feature_frame``.
"""

from __future__ import annotations

import warnings

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as sklearn_log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features.vector import FEATURE_COLUMNS
from .base import ModelResult, OutcomeModel, OutProbabilities, ensure_valid_probabilities

CLASSES = ["H", "D", "A"]


class LogisticOutcomeModel(OutcomeModel):
    name = "Logistic regression (sklearn)"

    def __init__(self, c: float = 10.0, max_iter: int = 3000, random_state: int = 42) -> None:
        self.c = c
        self.max_iter = max_iter
        self.random_state = random_state
        self.feature_columns = list(FEATURE_COLUMNS)
        self.pipeline = None
        self.train_samples: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticOutcomeModel":
        self.feature_columns = [c for c in FEATURE_COLUMNS if c in X.columns]
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=self.c,
                max_iter=self.max_iter,
                random_state=self.random_state,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pipeline.fit(X[self.feature_columns], y)
        self.train_samples = int(len(X))
        return self

    def predict_proba_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        probs = self.pipeline.predict_proba(X[self.feature_columns])
        classes = self.pipeline.steps[-1][1].classes_
        return pd.DataFrame(probs, columns=classes, index=X.index)

    def predict_proba(self, home: str, away: str) -> OutProbabilities:
        raise NotImplementedError(
            "LogisticOutcomeModel needs a feature frame, not a fixture; "
            "use predict_proba_frame with feature rows."
        )

    def predict(self, home: str, away: str) -> ModelResult:  # pragma: no cover
        raise NotImplementedError("logistic model is evaluated via predict_proba_frame")

    def log_loss(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        probs = self.predict_proba_frame(X)[CLASSES]
        return float(sklearn_log_loss(y_true, probs, labels=CLASSES))