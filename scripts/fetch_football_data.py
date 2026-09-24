"""Fetch historical English Premier League results and build the bundled dataset.

Source: football-data.co.uk (https://www.football-data.co.uk/).
Coverage: EPL seasons 1993/94 - 2025/26 (whatever seasons the source still serves).
Update frequency: source updates after each match day; re-run this script to refresh.
Access requirements: free, no API key, no authentication.
Usage restrictions: free for personal/academic use with attribution to
football-data.co.uk. See DATASETS.md before redistributing.

Usage:
    python scripts/fetch_football_data.py
    python scripts/fetch_football_data.py --start 2010 --end 2025

Outputs:
    data/raw/E0_<season>.csv          raw per-season files (not committed)
    data/bundled/epl_matches.csv      merged, normalized dataset (committed)
    data/bundled/dataset_manifest.json provenance + checksum (committed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
BUNDLED_DIR = REPO_ROOT / "data" / "bundled"
BUNDLED_CSV = BUNDLED_DIR / "epl_matches.csv"
MANIFEST = BUNDLED_DIR / "dataset_manifest.json"

USER_AGENT = "football-probability-intelligence-system/1.0 (educational project)"

# Columns we keep for the bundled dataset. Extra bookmaker columns are dropped
# to keep the repo small; raw files retain everything.
KEEP_COLUMNS = [
    "Div", "Date", "Time", "Season", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
    "AvgH", "AvgD", "AvgA",
    "B365H", "B365D", "B365A",
]

DATE_FORMATS = ["%d/%m/%Y", "%d/%m/%y", "%d/%m/%Y %H:%M", "%Y-%m-%d"]


def season_code(year_start: int) -> str:
    """1993 -> '9394' (the 1993/94 season)."""
    return f"{year_start % 100:02d}{(year_start + 1) % 100:02d}"


def season_label(year_start: int) -> str:
    return f"{year_start}/{str(year_start + 1)[-2:]}"


def parse_dates(series: pd.Series) -> pd.Series:
    """football-data.co.uk dates are dd/mm/yy or dd/mm/yyyy (UK format)."""
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def fetch_season(year_start: int, session: requests.Session) -> pd.DataFrame | None:
    code = season_code(year_start)
    url = BASE_URL.format(season_code=code)
    resp = session.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    if resp.status_code != 200:
        print(f"  {code}: HTTP {resp.status_code}, skipped")
        return None
    if len(resp.content) < 100:
        print(f"  {code}: empty response, skipped")
        return None

    raw_path = RAW_DIR / f"E0_{code}.csv"
    raw_path.write_bytes(resp.content)

    try:
        df = pd.read_csv(BytesIO(resp.content), encoding="cp1252", low_memory=False)
    except Exception as exc:  # noqa: BLE001 - report and skip unreadable seasons
        print(f"  {code}: unreadable ({exc}), skipped")
        return None

    if df.empty or "HomeTeam" not in df.columns or "FTHG" not in df.columns:
        print(f"  {code}: no match columns, skipped")
        return None

    # Some season files contain a trailing header-repeat row or notes rows.
    df = df[df["HomeTeam"].notna() & df["AwayTeam"].notna()].copy()
    df = df[pd.to_numeric(df["FTHG"], errors="coerce").notna()]
    if df.empty:
        print(f"  {code}: 0 valid matches, skipped")
        return None

    df["Season"] = season_label(year_start)
    print(f"  {code}: {len(df)} matches")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = parse_dates(df["Date"])
    df = df[df["Date"].notna()].copy()
    df["Date"] = df["Date"].dt.tz_localize(None)

    for col in ["FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST",
                "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # FTR is authoritative for outcome; recompute when missing.
    if "FTR" not in df.columns:
        df["FTR"] = ""
    missing = df["FTR"].isna() | (df["FTR"] == "")
    df.loc[missing, "FTR"] = df.loc[missing].apply(
        lambda r: "H" if r["FTHG"] > r["FTAG"] else ("A" if r["FTHG"] < r["FTAG"] else "D"),
        axis=1,
    )

    present = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[present].copy()
    df = df.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    df["match_id"] = [f"EPL-{i + 1:05d}" for i in range(len(df))]
    return df


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1993, help="first season start year (default 1993)")
    parser.add_argument("--end", type=int, default=2025, help="last season start year (default 2025)")
    parser.add_argument("--delay", type=float, default=0.4, help="seconds between requests")
    args = parser.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    BUNDLED_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    frames: list[pd.DataFrame] = []
    print(f"Fetching E0 seasons {args.start}/{args.start % 100 + 1:02d} -> {args.end} ...")
    for year in range(args.start, args.end + 1):
        df = fetch_season(year, session)
        if df is not None:
            frames.append(df)
        time.sleep(args.delay)

    if not frames:
        print("No seasons downloaded; keeping existing bundle if present.", file=sys.stderr)
        return 1

    merged = normalize(pd.concat(frames, ignore_index=True, sort=False))
    # Drop duplicate fixtures (same date, same pairing).
    merged = merged.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="first")
    merged = merged.reset_index(drop=True)

    merged.to_csv(BUNDLED_CSV, index=False, date_format="%Y-%m-%d")

    seasons = sorted(merged["Season"].unique())
    manifest = {
        "dataset": "English Premier League results (EPL, Div=E0)",
        "source_name": "football-data.co.uk",
        "source_url": "https://www.football-data.co.uk/football-data-england-premier-league",
        "download_urls": "https://www.football-data.co.uk/mmz4281/{season}/E0.csv",
        "access": "free, no API key",
        "license_notes": (
            "Free for personal and academic use with attribution to "
            "football-data.co.uk. Do not resell. See DATASETS.md."
        ),
        "update_frequency": "source updates after each match day",
        "coverage": {
            "competition": "English Premier League",
            "seasons": f"{seasons[0]} to {seasons[-1]}",
            "season_count": len(seasons),
            "matches": int(len(merged)),
            "first_match": str(merged["Date"].min().date()),
            "last_match": str(merged["Date"].max().date()),
        },
        "built_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "file": {
            "name": BUNDLED_CSV.name,
            "sha256": sha256(BUNDLED_CSV),
            "rows": int(len(merged)),
            "columns": list(merged.columns),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {BUNDLED_CSV} ({len(merged)} matches, {seasons[0]} -> {seasons[-1]})")
    print(f"Wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
