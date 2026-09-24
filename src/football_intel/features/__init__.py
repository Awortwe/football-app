"""Feature generation with strict no-leakage timing guarantees."""

from .timing import (
    TeamProfile,
    as_of_profile,
    prior_matches,
    profile_gym,
    team_name_uniform_check,
)

__all__ = [
    "TeamProfile",
    "as_of_profile",
    "prior_matches",
    "profile_gym",
    "team_name_uniform_check",
]