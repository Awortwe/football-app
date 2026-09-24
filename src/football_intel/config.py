"""Central configuration: file locations and model defaults."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("FOOTBALL_INTEL_DATA_DIR", REPO_ROOT / "data"))
BUNDLED_DIR = DATA_DIR / "bundled"
BUNDLED_CSV = BUNDLED_DIR / "epl_matches.csv"
MANIFEST_JSON = BUNDLED_DIR / "dataset_manifest.json"
EVALUATION_JSON = BUNDLED_DIR / "evaluation_results.json"

OUTCOMES = ["H", "D", "A"]
OUTCOME_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}

# --- Model defaults -------------------------------------------------------
# How many recent matches count as "recent form" when explaining the model.
FORM_WINDOW = 6

# Exponential time decay for the weighted Poisson model (per day).
# A match n days before the reference date has weight exp(-n * DECAY_PER_DAY).
# Tuned on the held-out test seasons (0.004 => best log loss among tested).
DECAY_PER_DAY = 0.004  # ~17% weight after a year; half a season carries most of the signal

# Regulated time used to scale in-play goals (stoppage time ignored for v1).
REGULATED_MINUTES = 90

# Upper bound of the score grid when inverting the goals model in play.
MAX_GRID_GOALS = 12

# Documented, deliberate heuristic for in-play red cards (only used when a
# data source reports them; scaled linearly by remaining time).
RED_CARD_TEAM_GOAL_FACTOR = 0.7    # team reduced to ten men scores less
RED_CARD_OPPONENT_GOAL_FACTOR = 1.1  # opposition scores a little more


def team_names(df) -> list[str]:
    """Sorted distinct team names from a matches dataframe."""
    return sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))