"""Data adapter: normalized reading of bundled historical matches.

The adapter is the single entry point for match data. It exposes a
``MatchDataAdapter`` protocol so a paid/live source (football-data.org, ...)
can later be swapped in without touching models, evaluation or the UI.

All probability features are required to be computed with data *strictly
before* the target fixture's date; helper ``features/timing.py`` enforces that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

import pandas as pd

from ..config import BUNDLED_CSV, MANIFEST_JSON


@dataclass(frozen=True)
class Fixture:
    """A playable fixture identified by its two teams and date (no outcome)."""

    home_team: str
    away_team: str
    date: date

    def as_tuple(self) -> tuple[str, str, date]:
        return (self.home_team, self.away_team, self.date)


@dataclass(frozen=True)
class DatasetMetadata:
    """Provenance and access notes for a dataset (see DATASETS.md)."""

    name: str
    source_name: str
    source_url: str
    download_urls: str
    access: str
    license_notes: str
    update_frequency: str
    coverage: dict
    built_at_utc: str

    @property
    def last_updated_label(self) -> str:
        return self.built_at_utc.replace("T", " ").replace("+00:00", " UTC")


class MatchDataAdapter(Protocol):
    """Protocol implemented by BundledCsvAdapter and any future live source."""

    def load_matches(self) -> pd.DataFrame: ...

    def teams(self) -> list[str]: ...

    def fixtures(self) -> list[Fixture]: ...

    def metadata(self) -> DatasetMetadata: ...

    def last_updated(self) -> str: ...


def load_metadata(path=MANIFEST_JSON) -> DatasetMetadata:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return DatasetMetadata(
        name=raw["dataset"],
        source_name=raw["source_name"],
        source_url=raw["source_url"],
        download_urls=raw.get("download_urls", ""),
        access=raw["access"],
        license_notes=raw["license_notes"],
        update_frequency=raw["update_frequency"],
        coverage=raw["coverage"],
        built_at_utc=raw["built_at_utc"],
    )


def load_bundled_matches(path=BUNDLED_CSV) -> pd.DataFrame:
    """Load the bundled normalized matches dataframe.

    Unknown values stay NaN; ``Date`` is parsed UTC-naive. Rows are sorted by
    date, ascending, and memoized in the app via st.cache_data.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values(["Date", "HomeTeam"], kind="mergesort").reset_index(drop=True)
    return df


class BundledCsvAdapter:
    """Adapter over the committed read-only dataset (works offline).

    This is the default adapter; it is also the data source for the
    historical replay mode. It never fabricates data it does not have.
    """

    def __init__(
        self,
        path=BUNDLED_CSV,
        manifest_path=MANIFEST_JSON,
        matches: pd.DataFrame | None = None,
    ) -> None:
        self._path = path
        self._manifest_path = manifest_path
        self._matches = matches

    def load_matches(self) -> pd.DataFrame:
        if self._matches is None:
            self._matches = load_bundled_matches(self._path)
        return self._matches

    def teams(self) -> list[str]:
        return sorted(set(self.load_matches()["HomeTeam"]) | set(self.load_matches()["AwayTeam"]))

    def fixtures(self) -> list[Fixture]:
        df = self.load_matches()
        return [
            Fixture(home, away, d.date())
            for d, home, away in df[["Date", "HomeTeam", "AwayTeam"]].itertuples(
                index=False, name=None
            )
        ]

    def metadata(self) -> DatasetMetadata:
        return load_metadata(self._manifest_path)

    def last_updated(self) -> str:
        return self.metadata().last_updated_label

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<BundledCsvAdapter {self._path.name}>"


class EmptyFixtureError(ValueError):
    """Raised when a fixture selection references two teams with no prior data."""


def require_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"dataset missing required column {col!r}")
    return df