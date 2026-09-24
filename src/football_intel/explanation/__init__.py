"""Plain-English explanation layer.

Rule: observed facts (form, goals, ratings derived from history), estimated
probabilities (model outputs) and written interpretations are kept visibly
separate. Nothing here ever presents an outcome as guaranteed.
"""

from .narrative import (
    Factor,
    Narrative,
    disclaimer,
    explain_halftime,
    explain_prematch,
    fmt_pct,
    match_report,
    short_update_for_event,
)

__all__ = [
    "Factor",
    "Narrative",
    "disclaimer",
    "explain_halftime",
    "explain_prematch",
    "fmt_pct",
    "match_report",
    "short_update_for_event",
]