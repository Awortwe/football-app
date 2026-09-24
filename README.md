# Football Probability & Intelligence System

A breakdown explainer for football (soccer) match outcomes. The system produces
**estimated probabilities** for Home / Draw / Away before kickoff and during the
match, explains them in plain English, replays finished matches event-by-event,
and compares its estimates against actual results. Everything runs on a small
bundle of free historical English Premier League data and is built to deploy on
**Streamlit Community Cloud's free tier**.

> Every number in the UI is a *model estimate*, not a prediction service and not
> betting advice. Probabilities sum to 100%; they are never guarantees.

---

## 1. What it does

| Page | What you see |
| --- | --- |
| **Pre-match analysis** | Estimated probability (Home / Draw / Away) for any two EPL teams on any date after the first season, with a plain-English factor list explaining the numbers, and the observed form facts behind them. |
| **Match timeline (replay)** | **Historical replay** of a finished match with a minute scrubber, the live-updating estimated probability, a probability-vs-time line chart, and a written update after each event. |
| **Match report** | After-full-time comparison of the pre-match estimate and (where data exists) the half-time estimate against the actual result, plus recorded shot / corner / card facts. |
| **Model evaluation** | Held-out log loss, Brier score, and calibration for the baseline + Poisson + logistic candidates, split into prematch and half-time scenarios. |
| **About & data** | Dataset provenance, coverage, license notes, limitations, and measured results. |

Interface labels used verbatim: **Estimated probability**, **Historical replay**,
**Data last updated**.

### Optional AI wording
The model explanations are human-written. A sidebar checkbox can additionally
paraphrase them through any OpenAI-compatible endpoint. It is **off by default**,
requires `AI_BASE_URL` / `AI_API_KEY` to be configured, degrades silently to the
human text on any failure, and is labelled as unofficial in the UI.

---

## 2. Model design (short)

- **No leakage:** every profile, feature and fitted parameter uses only finished
  matches from *strictly before* the match date being analysed.
- **Candidate models** (see `src/football_intel/`):
  - `NaiveBaseline` — league base rates used as the minimum bar.
  - `PoissonGoalsModel` — decay-weighted Poisson goals model (MLE on weighted
    per-team attack / defence strengths; `DECAY_PER_DAY = 0.004` tuned on
    held-out seasons). This is the production model. In-play estimates reroll a
    Poisson grid over remaining time conditioned on the current score.
  - `LogisticOutcomeModel` — scikit-learn pipeline (scaled features + polynomial
    terms + logistic regression) kept for cross-checking.
- **Replay honesty:** kick-off, half-time and full-time scores are observed facts.
  Individual goal minutes are **not** in the free source data, so the replay
  places goals inside the correct half at estimated minutes and flags every such
  event (`*(estimated time)`). Without half-time data the replay skips the
  interval and says so. Red cards are not synthesized from the bundle (the
  columns are full-time totals only).
- **AI wording:** optional, default off (see above).

---

## 3. Measured evaluation (held-out)

Seasons **2024/25 and 2025/26** are held out; the models train only on matches
before them. n = 760 test matches (760 with half-time states).

| Scenario | Model | Log loss | Brier | Calibration abs. error |
| --- | --- | --- | --- | --- |
| Prematch | Naive baseline | 1.0841 | 0.2191 | — |
| Prematch | Poisson (production) | 1.0258 | 0.2054 | 0.0427 |
| Prematch | Logistic | 1.0151 | 0.2027 | 0.0316 |
| Half-time | Naive baseline | 1.0841 | 0.2191 | — |
| Half-time | Poisson in-play | 0.8645 | 0.1692 | 0.0398 |

Lower is better; calibration abs. error near 0 means predicted ≈ observed rates.
Regenerate with `python scripts/run_evaluation.py` (writes
`data/bundled/evaluation_results.json`, which the app reads).

---

## 4. Data sources

- **Results:** [football-data.co.uk](https://www.football-data.co.uk/englandm.php)
  — free CSV downloads of English Premier League results **1993/94 → 2025/26**
  (11,944 matches, 31 seasons, 51 unique team names). No API key required;
  please respect their terms and attribute the source.
- This repository ships a cleaned bundle: `data/bundled/epl_matches.csv` plus a
  manifest `data/bundled/dataset_manifest.json` (coverage, build time,
  provenance). Rebuild it with `python scripts/fetch_football_data.py`.
- Optional live feed: `football-data.org` API via `FOOTBALL_DATA_API_KEY`
  (stub adapter in `src/football_intel/data/live_adapters.py`). Without it the
  app runs in **demo mode** on the bundle.

See `DATASETS.md` for full provenance and coverage notes.

---

## 5. Local setup (Windows / PowerShell)

Create the virtual env with `--without-pip` first, then bootstrap pip —
`python -m venv .venv` hangs in ensurepip on some Windows Python 3.10 builds:

```powershell
git clone <your-repo-url> football_system
cd football_system

# 1. Create the virtual environment and bootstrap pip
python -m venv --without-pip .venv
.\.venv\Scripts\python.exe https://bootstrap.pypa.io/get-pip.py
# (if behind a firewall, download get-pip.py and run that file instead)

# 2. Install pinned dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt  # pytest + plot libs

# 3. Run the app
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Tests:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest tests -q
```

All files are ASCII/simple; the app is responsive for phones (column layouts).

---

## 6. Deploy to Streamlit Community Cloud (free tier)

Repository prerequisites checklist — every item must be true before pushing:

- [x] Entry script is `app.py` at the repository root (Cloud auto-detects it).
- [x] `requirements.txt` exists with **exact pins** (`==`) and no `pip install`
      in code. Data imports are pandas/numpy/scikit-learn/scipy/streamlit only;
      `requests` is used by the optional live API + AI wording.
- [x] All data is **bundled in the repo** under `data/bundled/` (no runtime
      downloads; the app never calls a remote API in demo mode).
- [x] `.streamlit/config.toml` present (theme + port).
- [x] `.env.example` documents the optional secrets; `.gitignore` excludes `.env`
      and `data/raw/`.
- [x] Deterministic, seeded evaluation results are committed
      (`data/bundled/evaluation_results.json`) so each user triggers no
      recompute.
- [x] No filesystem writes at runtime.

Steps:

1. Init a Git repo, commit, and push to GitHub
   (`git init` → `git add .` → `git commit` → add your remote → `git push`).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. **New app** → select the repository and branch → Main file = `app.py`.
4. Deploy. Then, to opt into live data / AI wording:
   - Streamlit Cloud → **Settings → Secrets**, add
     `FOOTBALL_DATA_API_KEY` (optional live feed),
     `AI_BASE_URL` + `AI_API_KEY` + `AI_MODEL` (optional AI wording).
   - Save → **Rerun**. Without these, the app runs in demo mode on the bundle.

> Free-tier granularity is per-app; one app is free. Keep the model fits
> `st.cache_data`-cached (they are) — first viewers per season pay a few seconds
> of CPU once per session.

---

## 7. Repository layout

```
app.py                          # Streamlit entry point (all pages)
scripts/
  fetch_football_data.py        # rebuild the bundle from football-data.co.uk
  run_evaluation.py             # regenerate measures -> data/bundled/...json
data/
  bundled/                      # committed bundle: epl_matches.csv, manifest, eval
  raw/                          # ignored, per-season downloads from fetch script
src/football_intel/
  config.py                     # paths, constants (decay, windows, red-card factors)
  data/                         # adapter (bundle), live adapter stubs + status
  features/                     # as-of-time timing, Roller/Elo trackers, feature vector
  models/                       # base, naive baseline, poisson, logistic
  evaluation/                   # scoring rules, calibration, time-split run
  explanation/                  # human narratives + optional AI wording
  live/                         # replay timeline + in-play probability estimation
tests/                          # pytest suite (conftest builds a synthetic league)
requirements.txt                # exact pins (deploy target)
requirements-dev.txt            # pytest + plotting for dev only
.streamlit/config.toml
.env.example
```

---

## 8. Limitations

- The free bundle has **no lineups, injuries, tactics, xG, or per-goal minutes**.
  Those are never invented; where they matter the narrative says they are out of
  scope, and goal minutes in the replay are flagged as estimated.
- Models are trained on English top-flight history only and explicitly labelled
  as **estimated** — probabilities are not guarantees.
- Promoted teams with no history before a date are treated as an average side,
  with an on-screen warning; matches with no prior data at all degrade
  gracefully.