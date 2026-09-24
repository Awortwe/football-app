"""Probability estimate along a replay / live timeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..models.poisson import PoissonGoalsModel
from .replay import MatchEvent, ReplayTimeline


@dataclass
class TimelinePoint:
    minute: float
    score_home: int
    score_away: int
    home_win: float
    draw: float
    away_win: float
    event: str
    team: str | None = None
    is_estimated_time: bool = False

    def probs_dict(self) -> dict[str, float]:
        return {"H": self.home_win, "D": self.draw, "A": self.away_win}


def estimate_timeline(
    model: PoissonGoalsModel,
    timeline: ReplayTimeline,
    max_grid: int = 8,
) -> list[TimelinePoint]:
    """Evaluate the full-time-outcome estimate at every timeline event.

    The mass of the timeline comes from the model; the points themselves are
    the events described in ``timeline`` (facts + clearly flagged estimates).
    """
    points: list[TimelinePoint] = []
    for ev in timeline.events:
        minute = min(ev.minute, float(model.regulated_minutes))
        if ev.kind == "kickoff" and ev.score_home == 0 and ev.score_away == 0:
            minute = 0.0
        res = model.predict_inplay(
            timeline.home_team,
            timeline.away_team,
            minute=minute,
            home_goals=ev.score_home,
            away_goals=ev.score_away,
            max_grid=max_grid,
        )
        probs = res["probabilities"]
        points.append(
            TimelinePoint(
                minute=minute,
                score_home=ev.score_home,
                score_away=ev.score_away,
                home_win=probs["H"],
                draw=probs["D"],
                away_win=probs["A"],
                event=ev.kind,
                team=ev.team,
                is_estimated_time=ev.estimated_time,
            )
        )
    return points


def timeline_frame(points: list[TimelinePoint]) -> pd.DataFrame:
    """Chart-friendly dataframe: minute, H/D/A estimate, score."""
    return pd.DataFrame(
        [
            {
                "minute": p.minute,
                "Home win": p.home_win,
                "Draw": p.draw,
                "Away win": p.away_win,
                "event": p.event,
                "score": f"{p.score_home}–{p.score_away}",
            }
            for p in points
        ]
    )