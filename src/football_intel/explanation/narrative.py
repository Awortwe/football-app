"""Narrative builders: prematch factors, halftime advice, match report.

Everything a narrator says is grounded in fields the caller *actually* has.
Ratings come from the fitted Poisson model (estimates), form numbers come from
finished matches (observed facts), probabilities come from the model
(estimates). Interpretation sentences are clearly marked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import OUTCOME_LABELS
from ..models.base import ModelResult
from ..models.poisson import PoissonGoalsModel

# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #


@dataclass
class Factor:
    """One explained influence on the probability estimate.

    kind:
      observed        -- a fact from finished matches (form, goals, rest)
      estimated       -- a fitted/model quantity (ratings, probabilities)
      interpretation  -- plain-English reading of the above (never a guarantee)
    effect: "+" pushes the estimated probability of ``outcome`` up.
    """

    title: str
    text: str
    kind: str = "interpretation"
    effect: str = "+"
    outcome: str = "H"

    @property
    def badge(self) -> str:
        label = {
            "observed": "Observed fact",
            "estimated": "Estimate",
            "interpretation": "Interpretation",
        }[self.kind]
        arrow = {"+": "↗", "-": "↘"}.get(self.effect, "·")
        return f"[{label}] {arrow}"


@dataclass
class Narrative:
    headline: str
    disclaimer: str
    factors: list[Factor] = field(default_factory=list)

    def synopsis(self) -> str:
        return self.headline


def fmt_pct(p: float, digits: int = 0) -> str:
    return f"{p * 100:.{digits}f}%"


def disclaimer() -> str:
    return (
        "These figures are **estimated probabilities**, not guarantees. "
        "Football is low-scoring and uncertain; a 50% estimate is a coin flip, "
        "not a prediction that must land."
    )


# --------------------------------------------------------------------------- #
# Prematch
# --------------------------------------------------------------------------- #


def _league_base_rates(matches: pd.DataFrame, as_of_date) -> dict[str, float]:
    prior = matches[matches["Date"] < pd.Timestamp(as_of_date)]
    counts = prior["FTR"].value_counts()
    n = max(len(prior), 1)
    return {k: float(counts.get(k, 0)) / n for k in ("H", "D", "A")}


def _edge_outcome(p: dict[str, float], base: dict[str, float]) -> str:
    """Outcome whose probability diverges most from league base rates."""
    return max(("H", "D", "A"), key=lambda k: abs(p[k] - base[k]))


def explain_prematch(
    result: ModelResult,
    poisson: PoissonGoalsModel,
    home_profile,
    away_profile,
    matches: pd.DataFrame,
) -> Narrative:
    """Explain the numbers behind a prematch probability estimate."""
    base = _league_base_rates(matches, result.data_as_of)
    p = result.probabilities
    edge = _edge_outcome(p, base)
    edge_label = OUTCOME_LABELS[edge]
    factors: list[Factor] = []

    headline = (
        f"{result.home_team} vs {result.away_team}: estimated home-win "
        f"{fmt_pct(p['H'])}, draw {fmt_pct(p['D'])}, away-win {fmt_pct(p['A'])} "
        f"(sums to 100%)."
    )
    factors.append(Factor("Estimated probability", headline, kind="estimated"))

    # --- Observed facts ---------------------------------------------------- #
    if home_profile.n_matches > 0 and home_profile.last_goals_for:
        factors.append(
            Factor(
                f"{result.home_team} recent form",
                (
                    f"{home_profile.form_points:.2f} points/game over the last "
                    f"{len(home_profile.last_goals_for)} games "
                    f"({home_profile.recent_form}), scoring "
                    f"{home_profile.gf_per_game:.1f} and conceding "
                    f"{home_profile.ga_per_game:.1f} per game."
                ),
                kind="observed",
            )
        )
    if away_profile.n_matches > 0 and away_profile.last_goals_for:
        factors.append(
            Factor(
                f"{result.away_team} recent form",
                (
                    f"{away_profile.form_points:.2f} points/game over the last "
                    f"{len(away_profile.last_goals_for)} games "
                    f"({away_profile.recent_form}), scoring "
                    f"{away_profile.gf_per_game:.1f} and conceding "
                    f"{away_profile.ga_per_game:.1f} per game."
                ),
                kind="observed",
            )
        )

    if home_profile.n_matches and away_profile.n_matches:
        form_edge = home_profile.form_points - away_profile.form_points
        if abs(form_edge) >= 0.25:
            leader = result.home_team if form_edge > 0 else result.away_team
            factors.append(
                Factor(
                    "Recent form comparison",
                    (
                        f"{leader} arrive in better recent form "
                        f"({abs(form_edge):.2f} points/game edge). Recent games weigh "
                        f"more in the model, so this leans the estimate that way."
                    ),
                    kind="interpretation",
                    effect="+" if (form_edge > 0) == (edge == "H") else "-",
                    outcome=edge,
                )
            )

    splits = _split_text(result, home_profile, away_profile)
    if splits:
        factors.append(Factor("Home/away splits", splits, kind="observed"))
    rest = _rest_text(result, home_profile, away_profile)
    if rest:
        factors.append(Factor("Rest days", rest, kind="observed"))

    # --- Estimated (model) quantities -------------------------------------- #
    if poisson.params is not None:
        ah = poisson.parameter_summary(result.home_team)
        ad = poisson.parameter_summary(result.away_team)
        for team, rating in ((result.home_team, ah.get("attack")), (result.away_team, ad.get("attack"))):
            if rating is not None and abs(rating - 1) >= 0.03:
                pct = (rating - 1) * 100
                factors.append(
                    Factor(
                        f"{team} attack rating",
                        (
                            f"Estimated attack strength {rating:.2f} (league average "
                            f"1.00) — about {abs(pct):.0f}% "
                            f"{'more' if pct > 0 else 'less'} goals than an average "
                            f"side in the fitted period."
                        ),
                        kind="estimated",
                        effect="+" if (rating >= 1) == (edge == "H") else "-",
                        outcome=edge,
                    )
                )
        for team, rating in ((result.home_team, ah.get("defense")), (result.away_team, ad.get("defense"))):
            if rating is not None and abs(rating - 1) >= 0.03:
                pct = (1 - rating) * 100
                factors.append(
                    Factor(
                        f"{team} defence rating",
                        (
                            f"Estimated defensive strength {rating:.2f} — about "
                            f"{abs(pct):.0f}% {'less' if pct > 0 else 'more'} goals "
                            f"conceded than an average side."
                        ),
                        kind="estimated",
                        effect="+" if (rating <= 1) == (edge == "H") else "-",
                        outcome=edge,
                    )
                )

    home_boost = poisson.home_advantage_goal_boost
    factors.append(
        Factor(
            "Home advantage",
            (
                f"Home teams score about {home_boost:.2f}x their away rate in the "
                f"fitted model; {result.home_team} are at home, which is a real, "
                f"data-backed edge."
            ),
            kind="observed",
            effect="+",
            outcome="H",
        )
    )

    factors.append(
        Factor(
            "League base rates",
            (
                f"The reference base rates over the fitted window were home "
                f"{fmt_pct(base['H'])}, draw {fmt_pct(base['D'])}, away "
                f"{fmt_pct(base['A'])}. The biggest deviation from that reference "
                f"here is the {edge_label} ({fmt_pct(p[edge])})."
            ),
            kind="observed",
        )
    )

    factors.append(Factor("Uncertainty", disclaimer(), kind="interpretation"))
    return Narrative(headline=headline, disclaimer=disclaimer(), factors=factors)


def _split_text(result, home_profile, away_profile) -> str:
    parts = []
    if home_profile.home_gf_per_game is not None:
        parts.append(
            f"{result.home_team} at home average {home_profile.home_gf_per_game:.1f} "
            f"goals scored, {home_profile.home_ga_per_game:.1f} conceded"
        )
    if away_profile.away_gf_per_game is not None:
        parts.append(
            f"{result.away_team} away average {away_profile.away_gf_per_game:.1f} "
            f"goals scored, {away_profile.away_ga_per_game:.1f} conceded"
        )
    return "; ".join(parts).capitalize() + "." if parts else ""


def _rest_text(result, home_profile, away_profile) -> str:
    parts = []
    if home_profile.rest_days is not None:
        parts.append(f"{result.home_team} last played {home_profile.rest_days} days before")
    if away_profile.rest_days is not None:
        parts.append(f"{result.away_team} last played {away_profile.rest_days} days before")
    return "; ".join(parts) + "." if parts else ""


# --------------------------------------------------------------------------- #
# Timeline / live updates
# --------------------------------------------------------------------------- #


def short_update_for_event(
    probs_before: dict[str, float],
    probs_after: dict[str, float],
    event_label: str,
) -> str:
    """One written update after a match event (goal, red card, half-time)."""
    diff = {k: probs_after[k] - probs_before[k] for k in ("H", "D", "A")}
    hottest = max(diff, key=lambda k: abs(diff[k]))
    sign = "+" if diff[hottest] >= 0 else ""
    return (
        f"{event_label}. "
        f"{OUTCOME_LABELS[hottest]} estimation moved {sign}{fmt_pct(abs(diff[hottest]))} "
        f"to {fmt_pct(probs_after[hottest])} (home {fmt_pct(probs_after['H'])}, "
        f"draw {fmt_pct(probs_after['D'])}, away {fmt_pct(probs_after['A'])})."
    )


def explain_halftime(
    result: ModelResult,
    current_score: tuple[int, int],
    probs_ht: dict[str, float],
    home_profile,
    away_profile,
) -> Narrative:
    """Halftime summary: what happened + data-supported adjustments."""
    hg, ag = current_score
    most_likely = max(probs_ht, key=probs_ht.get)
    factors: list[Factor] = [
        Factor(
            "Half-time state",
            f"{result.home_team} {hg}–{ag} {result.away_team} at the break. "
            f"The remaining-time estimate stands at home {fmt_pct(probs_ht['H'])}, "
            f"draw {fmt_pct(probs_ht['D'])}, away {fmt_pct(probs_ht['A'])}.",
            kind="estimated",
        )
    ]
    if hg > ag:
        factors.append(
            Factor(
                "Situation",
                f"{result.home_team} lead. The draw and away-win each need at least "
                f"{ag + 1 - hg} goal(s) to change these estimates; with "
                f"{fmt_pct(probs_ht['H'])} for the home win, visitors chasing at least "
                f"a draw is the data-supported reading.",
                kind="interpretation",
            )
        )
    elif ag > hg:
        factors.append(
            Factor(
                "Situation",
                f"{result.away_team} lead at the break, so the home side must chase "
                f"the game. Away-win probability is {fmt_pct(probs_ht['A'])}; the draw "
                f"is {fmt_pct(probs_ht['D'])}.",
                kind="interpretation",
            )
        )
    else:
        factors.append(
            Factor(
                "Situation",
                f"Level at the break. The model's most likely full-time outcome is "
                f"still {OUTCOME_LABELS[most_likely]} ({fmt_pct(probs_ht[most_likely])}), "
                f"with the draw at {fmt_pct(probs_ht['D'])}.",
                kind="interpretation",
            )
        )

    if home_profile.n_matches and away_profile.n_matches:
        if home_profile.ga_per_game is not None:
            factors.append(
                Factor(
                    "Data-supported adjustment",
                    (
                        f"{result.home_team} have conceded {home_profile.ga_per_game:.1f}/game "
                        f"and {result.away_team} {away_profile.ga_per_game:.1f}/game in recent "
                        f"play. This source has no lineups or tactics, so any tactical "
                        f"advice here can only point at those observed rates."
                    ),
                    kind="observed",
                )
            )
    return Narrative(
        headline=f"Half-time: {hg}–{ag}.",
        disclaimer=disclaimer(),
        factors=factors,
    )


# --------------------------------------------------------------------------- #
# Match report (full time)
# --------------------------------------------------------------------------- #


def match_report(
    result: ModelResult,
    final_score: tuple[int, int],
    probs_ht: dict[str, float] | None = None,
    facts: dict | None = None,
) -> Narrative:
    """After full time: earlier estimates vs the result + readable report."""
    hg, ag = final_score
    actual = "H" if hg > ag else ("A" if hg < ag else "D")
    actual_label = OUTCOME_LABELS[actual]
    p_pre = result.probabilities

    factors: list[Factor] = [
        Factor(
            "Result",
            f"{result.home_team} {hg}–{ag} {result.away_team} — full-time "
            f"{actual_label}.",
            kind="observed",
        ),
        Factor(
            "Estimated probability before kickoff",
            (
                f"Home {fmt_pct(p_pre['H'])}, draw {fmt_pct(p_pre['D'])}, away "
                f"{fmt_pct(p_pre['A'])}. The {actual_label} was given "
                f"{fmt_pct(p_pre[actual])} going in."
            ),
            kind="estimated",
        ),
    ]
    if probs_ht is not None:
        factors.append(
            Factor(
                "Estimated probability at half-time",
                (
                    f"At the break the conditional estimate was home "
                    f"{fmt_pct(probs_ht['H'])}, draw {fmt_pct(probs_ht['D'])}, away "
                    f"{fmt_pct(probs_ht['A'])}."
                ),
                kind="estimated",
            )
        )

    surprise = p_pre[actual] < 0.30
    factors.append(
        Factor(
            "Reading",
            (
                f"The {actual_label} carried {fmt_pct(p_pre[actual])} before kickoff, so "
                f"this was "
                f"{'a surprise relative to the estimates' if surprise else 'consistent with what the model priced in'}."
                f" Probabilities are long-run frequencies: small probabilities "
                f"occasionally land, which is not a model failure."
            ),
            kind="interpretation",
        )
    )

    if facts:
        for title, text in facts.items():
            factors.append(Factor(title, text, kind="observed"))

    return Narrative(
        headline=f"Full time: {result.home_team} {hg}–{ag} {result.away_team}.",
        disclaimer=disclaimer(),
        factors=factors,
    )