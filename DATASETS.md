# Dataset notes — football-data.co.uk EPL bundle

## Provenance

| Field | Value |
| --- | --- |
| Dataset | English Premier League results and summary stats |
| Source | football-data.co.uk, England results page (E0 = Premier League) |
| URL | https://www.football-data.co.uk/englandm.php |
| Access | Free CSV downloads — no API key required |
| Update frequency | Weekly during the season; end-of-season snapshots |

The `scripts/fetch_football_data.py` script downloads one per-season CSV
(`E0.csv`, seasons `1993/94` → `2025/26`) from the website and normalises them
into the single committed bundle:

- `data/bundled/epl_matches.csv` — normalised long-form table.
- `data/bundled/dataset_manifest.json` — JSON manifest with build timestamp and
  coverage report, generated in the same script.

## Column coverage definitions

The bundle keeps the same column names as the source (`HTHG`, `HTAG`, `FTHG`,
`FTAG`, `HS`, `AS`, `HST`, `AST`, `HC`, `AC`, `HY`, `AY`, `HR`, `AR` …).
Coverage in the manifest is reported both as an absolute count and as a
percentage of matches, per field. Two important gaps:

- **Half-time score columns** exist for ~92% of matches. They are missing for
  1993/94 and 1994/95 (the first two seasons do not record HT scores), plus a
  handful of individual matches.
- **Shots / corners / cards columns** exist for ~76% of matches and are absent
  in the early 1990s. The app only renders recorded facts; it never fills them.

## Attribution / license notes

Data is © football-data.co.uk. It is freely available for personal and
educational use; please attribute the source when reusing it. See the website's
terms for redistribution rules.

## Regeneration

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\fetch_football_data.py
```

This writes fresh CSVs to `data/raw/` (git-ignored) and regenerates the bundled
`epl_matches.csv` + `dataset_manifest.json`. Commit the regenerated bundle if you
refresh it.

## Integrity checks performed by the builder

- Every row has a valid `Date`.
- Dash-separated result columns (e.g. `HS-AS`) are merged into the long-form
  score forms and validated against `HTHG/HTAG` and `FTHG/FTAG` where both are
  present.
- Duplicate `(Date, HomeTeam, AwayTeam)` rows are dropped.
- A post-merge consistency pass asserts `max(HTHG, HTAG) == 0` never coexists
  with a non-zero first/second-half score and that full-time >= half-time facts.