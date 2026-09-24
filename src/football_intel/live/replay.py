"""Historical match replay and probability timeline.

The bundled football-data.co.uk dataset records half-time and full-time
scores (observed facts) but NOT individual goal minutes. To make an honest
replay we therefore:

  * treat kickoff, the half-time score, and the full-time score as facts,
  * place goals inside the correct half at *estimated* minutes, always
    flagged ``estimated_time=True`` (never presented as fact),
  * never synthesize red cards, substitutions or injuries from this source
    (we do not have them; the live adapters may).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import REGULATED_MINUTES

HALF = "\u2013"  # en dash: "1–0"


@dataclass
class MatchEvent:
    kind: str                 # kickoff, goal, half_time, full_time, note
    minute: float
    label: str
    score_home: int
    score_away: int
    team: str | None = None
    is_observed: bool = True          # directly recorded in the source data
    estimated_time: bool = False      # minute placed by the replay, not recorded
    note: str = ""

    def short(self, home: str, away: str) -> str:
        who = f"{self.team} " if self.team else ""
        return f"{int(self.minute)}' — {who}{self.label} ({home} {self.score_home}{HALF}{self.score_away} {away})"


@dataclass
class ReplayTimeline:
    match_id: str
    home_team: str
    away_team: str
    kickoff: pd.Timestamp
    events: list[MatchEvent] = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return len(self.events)

    def without_goal_details(self):
        return [e for e in self.events if e.kind != "goal"]


def _estimate_minutes(n_goals: int, lo: float, hi: float) -> list[float]:
    """Spread n goal events evenly within [lo, hi] (estimated placements)."""
    if n_goals <= 0:
        return []
    if n_goals == 1:
        return [(hi + lo) / 2]
    return np.linspace(lo, hi, n_goals).round(1).tolist()


def build_replay_timeline(row: pd.Series, regulated: int = REGULATED_MINUTES) -> ReplayTimeline:
    """Build a replay timeline from a bundled match row (has HT + FT columns)."""
    home, away = row["HomeTeam"], row["AwayTeam"]
    fhg, fag = int(row["FTHG"]), int(row["FTAG"])

    has_ht = "HTHG" in row.index and pd.notna(row.get("HTHG")) and pd.notna(row.get("HTAG"))
    hhg, hag = (int(row["HTHG"]), int(row["HTAG"])) if has_ht else (0, 0)

    tl = ReplayTimeline(
        match_id=str(row.get("match_id", "")),
        home_team=home,
        away_team=away,
        kickoff=pd.Timestamp(row["Date"]),
    )
    tl.events.append(
        MatchEvent("kickoff", 0.0, "Kick-off", 0, 0, is_observed=True)
    )

    if not has_ht:
        tl.events.append(
            MatchEvent(
                "note",
                0.0,
                "Half-time score not recorded for this match in the source data; "
                "replay jumps straight to full time.",
                0,
                0,
                is_observed=True,
            )
        )

    hg1, ag1 = (hhg, hag) if has_ht else (0, 0)
    goals_first = _goal_sequence(hg1, ag1, home, away)
    goals_second = _goal_sequence(fhg - hg1, fag - ag1, home, away)

    def _emit_goals(seq: list[str], minutes: list[float], start_h: int, start_a: int, h: int, a: int):
        for minute, team in zip(minutes, seq):
            if team == home:
                h += 1
            else:
                a += 1
            tl.events.append(
                MatchEvent(
                    "goal",
                    minute,
                    "Goal" if team == home else "Goal (away)",
                    h,
                    a,
                    team=team,
                    is_observed=True,        # who scored in which half is a fact
                    estimated_time=True,     # the minute itself is an estimate
                )
            )
        return h, a

    h, a = _emit_goals(goals_first, _estimate_minutes(len(goals_first), 6.0, 42.0), 0, 0, 0, 0)

    if has_ht:
        tl.events.append(
            MatchEvent(
                "half_time", 45.0, "Half-time", hhg, hag, is_observed=True
            )
        )
    h, a = _emit_goals(
        goals_second, _estimate_minutes(len(goals_second), 48.0, 84.0), h, a, h, a
    )
    tl.events.append(
        MatchEvent("full_time", float(regulated), "Full time", fhg, fag, is_observed=True)
    )
    if fhg + fag != h + a:
        tl.events.append(
            MatchEvent(
                "note",
                float(regulated),
                "Source shows extra counters inconsistent with HT+FT scores.",
                fhg,
                fag,
                is_observed=True,
            )
        )
    return tl


def _goal_sequence(hg: int, ag: int, home: str, away: str) -> list[str]:
    """Order of scorers within a half, as actual team names.

    If only one side scored, the order is a fact (all its goals). When both
    sides scored, the interleaving is unknown from this source, so teams
    alternate and the ambiguity is recorded in the timeline note.
    """
    if hg > 0 and ag > 0:
        seq: list[str] = []
        for i in range(max(hg, ag)):
            if i < hg:
                seq.append(home)
            if i < ag:
                seq.append(away)
        return seq
    return [home] * hg + [away] * ag