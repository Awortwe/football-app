"""Data ingestion layer: adapters that supply normalized match records."""

from .adapter import (
    BundledCsvAdapter,
    DatasetMetadata,
    Fixture,
    MatchDataAdapter,
    load_bundled_matches,
    load_metadata,
)

__all__ = [
    "BundledCsvAdapter",
    "DatasetMetadata",
    "Fixture",
    "MatchDataAdapter",
    "load_bundled_matches",
    "load_metadata",
]