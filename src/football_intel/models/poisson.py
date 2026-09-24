"""Poisson goals model with exponential time decay.

Transparent, classic approach for 1X2 probabilities:

    goals_home ~ Poisson(base_home * attack_home * defense_away)
    goals_away ~ Poisson(base_away * attack_away * defense_home)

Parameters are fitted by maximum (decay-weighted) likelihood on all matches
strictly before a reference date. Older matches count for less, so recent form
drives the model, but a full season of games gives the base rates stability.
This is the "understandable baseline"; complexity is only added when
evaluation shows a gain (see docs/evaluation in the README).

Interpretation (the football behind the numbers):
  * an attack rating > 1 means the team scores more than league average,
  * a defense rating < 1 means the team concedes less than league average,
  * base_home / base_away capture overall scoring levels plus home advantage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import poisson

from ..config import (
    DECAY_PER_DAY,
    MAX_GRID_GOALS,
    OUTCOMES,
    RED_CARD_OPPONENT_GOAL_FACTOR,
    RED_CARD_TEAM_GOAL_FACTOR,
    REGULATED_MINUTES,
)
from .base import ModelResult, OutcomeModel, OutProbabilities, ensure_valid_probabilities


@dataclass
class PoissonParameters:
    base_home: float
    base_away: float
    attack: dict[str, float]      # team -> multiplier (league average == 1)
    defense: dict[str, float]     # team -> multiplier (league average == 1)

    def attack_rating(self, team: str) -> float:
        return self.attack.get(team, 1.0)

    def defense_rating(self, team: str) -> float:
        return self.defense.get(team, 1.0)


class PoissonGoalsModel(OutcomeModel):
    name = "Poisson (decay-weighted)"

    def __init__(
        self,
        decay: float = DECAY_PER_DAY,
        min_team_matches: int = 2,
        regularization: float = 0.02,
        regulated_minutes: int = REGULATED_MINUTES,
    ) -> None:
        self.decay = decay
        self.min_team_matches = min_team_matches
        self.regularization = regularization
        self.regulated_minutes = regulated_minutes
        self.params: PoissonParameters | None = None
        self.reference_date: date | None = None
        self.n_matches: int = 0
        self.teams_seen: set[str] = set()
        self._fit_log: dict[str, float] = {}

    # ------------------------------------------------------------------ fit
    def _prepare(self, matches: pd.DataFrame, reference_date) -> pd.DataFrame:
        cutoff = pd.Timestamp(reference_date)
        df = matches[(matches["Date"] < cutoff) & matches["FTHG"].notna() & matches["FTAG"].notna()]
        return df.reset_index(drop=True)

    def fit(self, matches: pd.DataFrame, reference_date=None) -> "PoissonGoalsModel":
        reference_date = reference_date or matches["Date"].max()
        self.reference_date = pd.Timestamp(reference_date).date()
        df = self._prepare(matches, reference_date)
        if df.empty:
            raise ValueError("no finished matches available before the reference date")

        self.n_matches = int(len(df))
        self.teams_seen = set(df["HomeTeam"]) | set(df["AwayTeam"])

        team_counts = pd.concat([df["HomeTeam"], df["AwayTeam"]]).value_counts()
        modelled = set(team_counts[team_counts >= self.min_team_matches].index)
        if not modelled:
            raise ValueError("no team has enough finished matches to fit the model")
        teams = sorted(modelled)
        idx_of = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        age_days = (pd.Timestamp(reference_date) - df["Date"]).dt.days.to_numpy().astype(float)
        w = np.exp(-self.decay * np.clip(age_days, 0, None))
        gf_h = df["FTHG"].to_numpy(dtype=float)
        gf_a = df["FTAG"].to_numpy(dtype=float)

        base = np.mean(gf_h), np.mean(gf_a)

        def unpack(x: np.ndarray):
            bh, ba = float(np.exp(x[0])), float(np.exp(x[1]))
            a = np.exp(x[2 : 2 + n])
            d = np.exp(x[2 + n : 2 + 2 * n])
            return bh, ba, a, d

        ih = np.array([idx_of[t] for t in df["HomeTeam"]])
        ia = np.array([idx_of[t] for t in df["AwayTeam"]])

        def objective(x: np.ndarray) -> float:
            bh, ba, a, d = unpack(x)
            lh = bh * a[ih] * d[ia]
            la = ba * a[ia] * d[ih]
            ll = np.sum(w * (gf_h * np.log(lh) - lh) + w * (gf_a * np.log(la) - la))
            reg = self.regularization * np.sum(x[2:] ** 2)
            return float(-(ll - reg))

        def gradient(x: np.ndarray) -> np.ndarray:
            bh, ba, a, d = unpack(x)
            lh = bh * a[ih] * d[ia]
            la = ba * a[ia] * d[ih]
            ch = w * (gf_h - lh)        # d loglik / d log(any factor of lambda_h)
            ca = w * (gf_a - la)        # d loglik / d log(any factor of lambda_a)
            g = np.zeros_like(x)
            g[0] = -float(np.sum(ch))
            g[1] = -float(np.sum(ca))
            # alpha[t]: matches where t is home (contributes via lambda_h)
            #           matches where t is away (contributes via lambda_a)
            g[2 : 2 + n] = -np.bincount(ih, weights=ch, minlength=n)
            g[2 : 2 + n] -= np.bincount(ia, weights=ca, minlength=n)
            # delta[t]: home team defends in lambda_away; away team defends in lambda_home
            g[2 + n : 2 + 2 * n] = -np.bincount(ih, weights=ca, minlength=n)
            g[2 + n : 2 + 2 * n] -= np.bincount(ia, weights=ch, minlength=n)
            g[2:] += 2 * self.regularization * x[2:]
            return g

        x0 = np.zeros(2 + 2 * n)
        x0[0], x0[1] = math.log(base[0]), math.log(base[1])

        res = optimize.minimize(
            objective,
            x0,
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
        )
        if not res.success:
            # L-BFGS-B returning a usable point is enough; do not hard fail.
            self._fit_log["warning"] = str(res.message)

        bh, ba, a, d = unpack(res.x)

        # Renormalize attack/defense so their geometric means are 1 (keeps
        # predictions identical and the parameters interpretable).
        log_a_mean = float(np.mean(np.log(a))); log_d_mean = float(np.mean(np.log(d)))
        a = a * np.exp(-log_a_mean)
        d = d * np.exp(-log_d_mean)
        bh *= float(np.exp(log_a_mean + log_d_mean))
        ba *= float(np.exp(log_a_mean + log_d_mean))

        self._fit_log.update({"log_likelihood": float(-objective(res.x)), "converged": bool(res.success)})
        self.params = PoissonParameters(
            base_home=float(bh),
            base_away=float(ba),
            attack={t: float(a[i]) for t, i in idx_of.items()},
            defense={t: float(d[i]) for t, i in idx_of.items()},
        )
        return self

    # ------------------------------------------------------------ rates
    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        if self.params is None:
            raise RuntimeError("model not fitted")
        p = self.params
        lh = p.base_home * p.attack_rating(home) * p.defense_rating(away)
        la = p.base_away * p.attack_rating(away) * p.defense_rating(home)
        return max(lh, 1e-6), max(la, 1e-6)

    def score_grid_proba(self, home: str, away: str, max_goals: int = 8) -> np.ndarray:
        """P(score == (i, j)) as a (max_goals+1) x (max_goals+1) matrix."""
        lh, la = self.lambdas(home, away)
        hi = poisson.pmf(np.arange(max_goals + 1), lh)
        ai = poisson.pmf(np.arange(max_goals + 1), la)
        return np.outer(hi, ai)

    # ------------------------------------------------------- predictions
    def predict_proba(self, home: str, away: str) -> OutProbabilities:
        return ensure_valid_probabilities(self._proba_from_grid(home, away, MAX_GRID_GOALS))

    def _proba_from_grid(self, home: str, away: str, max_goals: int) -> dict[str, float]:
        grid = self.score_grid_proba(home, away, max_goals)
        # grid[i, j] = P(home goals == i, away goals == j).
        # i > j (below the diagonal) is a home win; i < j is an away win.
        p_home = grid[np.tril_indices_from(grid, k=-1)].sum()
        p_away = grid[np.triu_indices_from(grid, k=1)].sum()
        return {"H": p_home, "D": 1.0 - p_home - p_away, "A": p_away}

    def predict(self, home: str, away: str) -> ModelResult:
        return ModelResult(
            home_team=home,
            away_team=away,
            match_date=pd.Timestamp(self.reference_date).date() if self.reference_date else None,
            model_name=self.name,
            probabilities=self.predict_proba(home, away),
            data_as_of=self.reference_date,
            basis_notes=(
                f"Fitted on {self.n_matches} finished matches older than "
                f"{self.reference_date}, decay-weighted (half-life ≈ {0.693 / self.decay:.0f} days).",
            ),
        )

    # --------------------------------------------------------- in-play
    def predict_inplay(
        self,
        home: str,
        away: str,
        minute: float,
        home_goals: int = 0,
        away_goals: int = 0,
        home_red_cards: int = 0,
        away_red_cards: int = 0,
        max_grid: int = MAX_GRID_GOALS,
    ) -> dict:
        """Full-time outcome probabilities conditional on the current state.

        Uses only ``minute`` elapsed, the current score and (additively) red
        cards. Substitution/injury detail is used only if a feed provides it.
        """
        if self.params is None:
            raise RuntimeError("model not fitted")
        minute = float(np.clip(minute, 0.0, float(self.regulated_minutes)))
        frac = 1.0 - minute / float(self.regulated_minutes)

        lh, la = self.lambdas(home, away)
        for _ in range(max(home_red_cards, 0)):
            lh *= RED_CARD_TEAM_GOAL_FACTOR
            la *= RED_CARD_OPPONENT_GOAL_FACTOR
        for _ in range(max(away_red_cards, 0)):
            la *= RED_CARD_TEAM_GOAL_FACTOR
            lh *= RED_CARD_OPPONENT_GOAL_FACTOR
        lh_rem, la_rem = lh * frac, la * frac

        h_max = max(max_grid - home_goals, 0)
        a_max = max(max_grid - away_goals, 0)
        hp = poisson.pmf(np.arange(h_max + 1), lh_rem)
        ap = poisson.pmf(np.arange(a_max + 1), la_rem)
        grid = np.outer(hp, ap)

        base_hg, base_ag = int(home_goals), int(away_goals)
        probs = {"D": 0.0, "H": 0.0, "A": 0.0}
        for i in range(h_max + 1):
            for j in range(a_max + 1):
                final_h = base_hg + i
                final_a = base_ag + j
                if final_h > final_a:
                    probs["H"] += grid[i, j]
                elif final_h < final_a:
                    probs["A"] += grid[i, j]
                else:
                    probs["D"] += grid[i, j]

        return {
            "probabilities": ensure_valid_probabilities(probs),
            "minute": minute,
            "score": (base_hg, base_ag),
            "expected_remaining_goals": (float(lh_rem), float(la_rem)),
            "lambdas_full": (float(self.lambdas(home, away)[0]), float(self.lambdas(home, away)[1])),
        }

    # ---------------------------------------------------- exploration
    def parameter_summary(self, team: str) -> dict[str, float]:
        if self.params is None:
            return {}
        return {
            "attack": self.params.attack_rating(team),
            "defense": self.params.defense_rating(team),
        }

    @property
    def home_advantage_goal_boost(self) -> float:
        """Expected home-goal multiplier from playing at home."""
        if self.params is None:
            return 1.0
        p = self.params
        return p.base_home / p.base_away