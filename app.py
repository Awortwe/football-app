"""Football Probability & Intelligence System — Streamlit entry point.

Run locally:  streamlit run app.py
Deployed:     Streamlit Community Cloud (free tier), entry file app.py

Pages:
  * Pre-match analysis     — estimated 1X2 probabilities + plain-English factors
  * Match timeline (replay)— historical replay with probability updates per event
  * Match report           — after-full-time comparison of estimates vs result
  * Model evaluation       — held-out log loss / Brier / calibration
  * About & data           — provenance, limitations, deployment notes

Labelled categories are used verbatim across the UI:
  "Estimated probability", "Historical replay", "Data last updated".
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from football_intel.config import EVALUATION_JSON, OUTCOME_LABELS  # noqa: E402
from football_intel.data.adapter import BundledCsvAdapter  # noqa: E402
from football_intel.data.live_adapters import configured_live_adapters  # noqa: E402
from football_intel.explanation.ai_wording import ai_supported, ai_worded  # noqa: E402
from football_intel.explanation.narrative import (  # noqa: E402
    explain_halftime,
    explain_prematch,
    fmt_pct,
    match_report,
    short_update_for_event,
)
from football_intel.features.timing import as_of_profile  # noqa: E402
from football_intel.live.estimation import estimate_timeline, timeline_frame  # noqa: E402
from football_intel.live.replay import build_replay_timeline  # noqa: E402
from football_intel.models.poisson import PoissonGoalsModel  # noqa: E402

st.set_page_config(
    page_title="Football Probability & Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def load_matches() -> pd.DataFrame:
    return BundledCsvAdapter().load_matches()


@st.cache_data(show_spinner=False)
def metadata_json() -> dict:
    return json.loads(Path(str(ROOT / "data/bundled/dataset_manifest.json")).read_text("utf-8"))


@st.cache_data(show_spinner=False)
def fit_poisson(cutoff: str) -> dict:
    """Fit the decay-weighted Poisson model on matches strictly before cutoff."""
    matches = load_matches()
    model = PoissonGoalsModel().fit(matches, reference_date=pd.Timestamp(cutoff))
    return {
        "cutoff": cutoff,
        "n_matches": model.n_matches,
        "base_home": model.params.base_home,
        "base_away": model.params.base_away,
        "attack": model.params.attack,
        "defense": model.params.defense,
    }


def poisson_model(cutoff):
    """Leak-free model fitted on matches strictly before ``cutoff`` (cached).

    Returns ``None`` when there is not enough finished history to fit (e.g. the
    very first fixtures in the dataset) — callers must degrade gracefully.
    """
    c = pd.Timestamp(cutoff)
    try:
        snap = fit_poisson(c.strftime("%Y-%m-%d"))
    except ValueError:
        return None
    from football_intel.models.poisson import PoissonParameters

    model = PoissonGoalsModel()
    model.params = PoissonParameters(
        base_home=snap["base_home"],
        base_away=snap["base_away"],
        attack=snap["attack"],
        defense=snap["defense"],
    )
    model.reference_date = c.date()
    model.n_matches = snap["n_matches"]
    model.teams_seen = set(snap["attack"]) | set(snap["defense"])
    return model


@st.cache_data(show_spinner=False)
def profiles_for(home: str, away: str, cutoff: str):
    matches = load_matches()
    h = as_of_profile(matches, home, cutoff)
    a = as_of_profile(matches, away, cutoff)
    return h, a


@st.cache_data(show_spinner=False)
def evaluation_json() -> dict | None:
    path = Path(str(ROOT / "data/bundled/evaluation_results.json"))
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


# --------------------------------------------------------------------------- #
# Shared UI helpers
# --------------------------------------------------------------------------- #


def fixtures_for_season(matches: pd.DataFrame, season: str) -> pd.DataFrame:
    return matches[matches["Season"] == season].sort_values("Date")


def picker_options(matches: pd.DataFrame, season: str) -> list[str]:
    sub = fixtures_for_season(matches, season)
    return [
        f"{r.Date.date()} — {r.HomeTeam} vs {r.AwayTeam}  (HT {int(r.HTHG) if pd.notna(r.HTHG) else '–'}"
        f":{int(r.HTAG) if pd.notna(r.HTAG) else '–'} · FT {int(r.FTHG)}:{int(r.FTAG)})"
        for r in sub.itertuples()
    ]


def read_fixture_choice(matches: pd.DataFrame, season: str, choice: str):
    sub = fixtures_for_season(matches, season).reset_index(drop=True)
    return sub.iloc[picker_options(matches, season).index(choice)]


def probability_gauge(probs: dict[str, float], key: str = "gauge") -> None:
    """Estimated probability bars: H / D / A that always sum to 100%."""
    st.markdown("**Estimated probability** (sums to 100%)")
    colors = {"H": "#0f6e43", "D": "#b7791f", "A": "#1f4e79"}
    vals = [probs[k] for k in ("H", "D", "A")]
    df = pd.DataFrame(
        {
            "outcome": ["Home win", "Draw", "Away win"],
            "p": vals,
        }
    )
    st.bar_chart(df.set_index("outcome"), color=[colors[k] for k in ("H", "D", "A")],
                 horizontal=True, height=180, width="stretch")
    c = st.columns(3)
    for col, k in zip(c, ("H", "D", "A")):
        col.metric(OUTCOME_LABELS[k], fmt_pct(probs[k], 1))


def show_badges() -> None:
    st.sidebar.markdown("---")
    adapters = configured_live_adapters()
    if adapters:
        for ad in adapters:
            st.sidebar.info(ad.status().badge)
    else:
        st.sidebar.warning(
            "**HISTORICAL REPLAY** — bundled dataset only. "
            "Add a `FOOTBALL_DATA_API_KEY` secret to enable live data."
        )
    meta = metadata_json()
    st.sidebar.markdown(
        f"**Data source:** {meta['source_name']}  \n"
        f"**Coverage:** {meta['coverage']['seasons']} "
        f"({meta['coverage']['matches']} matches)  \n"
        f"**Data last updated:** {meta['built_at_utc'][:16].replace('T', ' ')} UTC"
    )


def fixture_selector(matches: pd.DataFrame, label: str = "Select a historical fixture"):
    seasons = sorted(matches["Season"].dropna().unique(), reverse=True)
    season = st.selectbox("Season", seasons, key=f"{label}-season")
    options = picker_options(matches, season)
    if not options:
        st.warning("No fixtures in this season.")
        return None
    choice = st.selectbox(label, options, key=f"{label}-fixture")
    return read_fixture_choice(matches, season, choice)


def show_event_list(timeline) -> None:
    st.markdown("**Key events (replay)**")
    for e in timeline.events:
        mark = " (estimated time)" if e.estimated_time else ""
        if e.kind == "goal":
            who = e.team or ""
            st.markdown(f"• {int(e.minute)}' — **Goal** — {who}{mark}")
        elif e.kind == "half_time":
            st.markdown(f"• 45' — **Half-time** {e.score_home}–{e.score_away}")
        elif e.kind == "full_time":
            st.markdown(f"• {int(e.minute)}' — **Full time** {e.score_home}–{e.score_away}")
        elif e.kind == "kickoff":
            st.markdown(f"• 0' — Kick-off")
        elif e.kind == "note":
            st.info(e.label)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def page_prematch(matches: pd.DataFrame) -> None:
    st.header("Pre-match analysis")
    st.caption("Choose two teams and a match date — estimates use only data from before that date.")

    t1, t2 = st.columns(2)
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    home = t1.selectbox("Home team", teams)
    away_options = [t for t in teams if t != home]
    away = t2.selectbox("Away team", away_options)

    dmin, dmax = matches["Date"].min().date(), matches["Date"].max().date()
    col1, col2 = st.columns(2)
    default_date = dmax - timedelta(days=200)
    pick_mode = col1.radio("Fixture selection", ["Pick a date", "Pick from real fixtures"])
    if pick_mode == "Pick from real fixtures":
        fixtures = fixtures_for_season(matches, sorted(matches["Season"].unique())[-1])
        pair = fixtures[(fixtures["HomeTeam"] == home) & (fixtures["AwayTeam"] == away)]
        if pair.empty:
            st.info(f"No recorded {home} (home) vs {away} (away) fixture found in the latest season.")
            match_date = default_date
        else:
            pair = pair.sort_values("Date")
            options = [
                f"{r.Date.date()} — {r.HomeTeam} {int(r.FTHG)}:{int(r.FTAG)} {r.AwayTeam}"
                for r in pair.itertuples()
            ]
            sel = col2.selectbox("Fixture", options)
            match_date = pair.iloc[options.index(sel)]["Date"].date()
    else:
        match_date = col2.date_input(
            "Match date", value=default_date, min_value=dmin, max_value=dmax
        )

    if match_date <= dmin:
        st.warning("No data is available before this date; pick a later date.")
        return

    with st.spinner("Estimating probabilities..."):
        model = poisson_model(match_date)
        if model is None:
            st.warning("Not enough finished matches before this date to fit the model. "
                       "Pick a later date.")
            return
        result = model.predict(home, away)
        hp, ap = profiles_for(home, away, match_date)

    st.caption(f"**Data last updated:** data used is strictly before {match_date}.")

    probability_gauge(result.probabilities)
    st.caption("These are **estimated probabilities** (model output). They add to 100% but are "
               "not guarantees — see the factor list below.")

    narrative = explain_prematch(result, model, hp, ap, matches)
    st.subheader("Why the numbers look like this")
    ai_on = bool(st.session_state.get("ai_toggle", False))
    if ai_on:
        st.caption("AI-generated wording enabled (unofficial, see About).")
    for f in narrative.factors:
        text = ai_worded(f.text, ai_on)
        with st.expander(f"{f.badge} — {f.title}"):
            st.markdown(text)

    with st.expander("Observed facts (from finished matches only)", expanded=False):
        for label, profile in ((home, hp), (away, ap)):
            if profile.n_matches == 0:
                st.info(f"{label} has no finished matches before {match_date} — treated as an "
                        f"average side; estimates are less reliable.")
                continue
            form_pts = f"{profile.form_points:.2f} pts/game" if profile.n_matches else "n/a"
            st.markdown(
                f"**{label}** — based on {profile.n_matches} prior games "
                f"(spanning {profile.sampled_period_days} days):\n\n"
                f"- Recent form: {form_pts} ({profile.recent_form})\n"
                f"- Goals for: {profile.gf_per_game:.2f}/game · against: {profile.ga_per_game:.2f}/game\n"
                f"- At home: {_fmt_opt(profile.home_gf_per_game)} for, "
                f"{_fmt_opt(profile.home_ga_per_game)} against · "
                f"Away: {_fmt_opt(profile.away_gf_per_game)} for, "
                f"{_fmt_opt(profile.away_ga_per_game)} against\n"
                f"- Days rest since last match: {profile.rest_days or 'n/a'}"
            )

    st.markdown("---")
    st.caption("Estimates come from a decay-weighted Poisson goals model fitted on historic "
               "results. Player-level factors (injuries, lineups, tactics) are **not** in this "
               "dataset and are never assumed.")


def _fmt_opt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def page_timeline(matches: pd.DataFrame) -> None:
    st.header("Match timeline — historical replay")
    st.caption("Replay a finished match event-by-event. Probabilities are re-estimated after "
               "every goal and at half-time. This is **historical replay**, not a live feed.")

    row = fixture_selector(matches, label="Fixture to replay")
    if row is None:
        return
    has_ht = pd.notna(row.get("HTHG")) and pd.notna(row.get("HTAG"))

    st.caption(
        f"**Historical replay** — {row.Date.date()}, {row.HomeTeam} vs {row.AwayTeam}. "
        f"**Data last updated:** bundled dataset built from football-data.co.uk."
    )
    if not has_ht:
        st.warning("This match lacks half-time data in the source, so the replay jumps straight "
                   "to full time.")

    with st.spinner("Building replay timeline..."):
        timeline = build_replay_timeline(row)
        model = poisson_model(row["Date"] + pd.Timedelta(days=1))

    if model is None:
        st.warning("No finished matches preceded this fixture, so probability updates cannot "
                   "be estimated. The score progression below is still shown.")
        with st.expander("Full event list", expanded=True):
            show_event_list(timeline)
        return

    points = estimate_timeline(model, timeline)

    idx = st.slider("Progress", 0, len(points) - 1, 1, format="%d", key="timeline-i")
    p = points[idx]

    st.caption("* = minute not recorded in the source data; estimated placement inside the correct half.")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Clock", f"{int(p.minute)}'")
        st.metric("Score", f"{row.HomeTeam} {p.score_home} – {p.score_away} {row.AwayTeam}")
        st.caption("Estimated probability of full-time outcome:")
        probability_gauge(p.probs_dict(), key="tg")
    with c2:
        st.line_chart(
            timeline_frame(points).set_index("minute")[["Home win", "Draw", "Away win"]],
            height=260,
        )

    if idx > 0:
        prev = points[idx - 1]
        st.markdown(
            "**:grey[Written update:]** "
            + short_update_for_event(prev.probs_dict(), p.probs_dict(), _event_label(prev))
        )

    with st.expander("Full event list", expanded=True):
        show_event_list(timeline)
    st.caption(
        "Fact vs estimate: kick-off, the half-time score and the full-time score are observed "
        "facts. Individual goal minutes are not recorded in this free dataset, so the replay "
        "places them inside the correct half and flags them."
    )


def _event_label(p) -> str:
    if p.team:
        return f"Goal — {p.team}"
    return {"half_time": "Half-time", "full_time": "Full time", "kickoff": "Kick-off",
            "note": "Note"}.get(p.event, p.event)


def page_report(matches: pd.DataFrame) -> None:
    st.header("Match report — after full time")
    st.caption("Compare what the model estimated before kickoff and at half-time with the "
               "actual result.")

    row = fixture_selector(matches, label="Finished match to report")
    if row is None:
        return
    has_ht = pd.notna(row.get("HTHG")) and pd.notna(row.get("HTAG"))

    with st.spinner("Building match report..."):
        model = poisson_model(row["Date"] + pd.Timedelta(days=1))
        if model is None:
            st.warning("No finished matches preceded this fixture, so the estimated-vs-result "
                       "comparison is unavailable. Facts below are still shown.")
            prematch = None
            probs_ht = None
        else:
            prematch = model.predict(row["HomeTeam"], row["AwayTeam"])
            probs_ht = None
            if has_ht:
                htres = model.predict_inplay(
                    row["HomeTeam"], row["AwayTeam"], minute=45,
                    home_goals=int(row["HTHG"]), away_goals=int(row["HTAG"]),
                )
                probs_ht = htres["probabilities"]

    facts = {}
    if pd.notna(row.get("HS")) and pd.notna(row.get("AS")):
        facts["Possession-free shot stats"] = (
            f"Shots {int(row['HS'])}:{int(row['AS'])}, on target {int(row['HST'])}:{int(row['AST'])}"
        )
    if pd.notna(row.get("HC")) and pd.notna(row.get("AC")):
        facts["Corners"] = f"{int(row['HC'])}:{int(row['AC'])}"
    if pd.notna(row.get("HY")) and pd.notna(row.get("AY")):
        facts["Cards (yellow / red)"] = (
            f"{int(row['HY'])}/{int(row['HR'] or 0)} vs {int(row['AY'])}/{int(row['AR'] or 0)}"
            if pd.notna(row.get("HR")) and pd.notna(row.get("AR"))
            else f"{int(row['HY'])} vs {int(row['AY'])}"
        )

    st.subheader("Result")
    st.markdown(f"# {row.HomeTeam} {int(row.FTHG)} – {int(row.FTAG)} {row.AwayTeam}")
    if has_ht:
        st.markdown(f"Half-time: {int(row.HTHG)} – {int(row.HTAG)}   ·   "
                    f"**Data last updated:** {row.Date.date()} (bundled set)")

    if prematch is not None:
        report = match_report(prematch, (int(row.FTHG), int(row.FTAG)), probs_ht, facts)
        st.subheader("Estimated probability vs result")
        st.markdown(report.headline)
        for f in report.factors:
            with st.expander(f"{f.badge} — {f.title}", expanded=f.kind in ("observed",)):
                st.markdown(f.text)

    if has_ht and prematch is not None:
        hp, ap = profiles_for(row["HomeTeam"], row["AwayTeam"], row["Date"])
        ht_narrative = explain_halftime(prematch, (int(row.HTHG), int(row.HTAG)), probs_ht, hp, ap)
        with st.expander("Half-time view (retrospective)", expanded=False):
            st.markdown(ht_narrative.headline)
            for f in ht_narrative.factors:
                st.markdown(f"**{f.title}** ({f.badge})\n\n{f.text}")


def page_evaluation() -> None:
    st.header("Model evaluation (held-out)")
    data = evaluation_json()
    if data is None:
        st.warning("`data/bundled/evaluation_results.json` missing. Run "
                   "`python scripts/run_evaluation.py` locally and commit the file.")
        return
    st.caption(data["method"])
    st.caption(
        f"Test seasons: {', '.join(data['test_seasons'])} · "
        f"{data['test_matches']} held-out matches · "
        f"{data['halftime_states']} with half-time states."
    )

    for scenario, title in (("prematch", "Prematch estimates (before kickoff)"),
                            ("halftime", "In-play estimates (half-time)")):
        st.subheader(title)
        rows = []
        for model, m in data["scenario"][scenario].items():
            rows.append(
                {
                    "model": data["models"][model]["name"],
                    "role": data["models"][model]["role"],
                    "n": m["n"],
                    "log_loss": m["log_loss"],
                    "brier": m["brier"],
                    "calibration abs error": m["calibration_abs_error"],
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.caption(
        "Lower log loss and Brier are better; the naive baseline must be beaten. "
        "An 'abs error' near 0 means predicted and observed rates line up."
    )

    with st.expander("Calibration detail (per outcome, per bin)"):
        for scenario in ("prematch", "halftime"):
            st.markdown(f"**{scenario}**")
            for model, m in data["scenario"][scenario].items():
                cal = pd.DataFrame(m["calibration"])
                if cal.empty:
                    continue
                st.markdown(f"*{data['models'][model]['name']}*")
                st.dataframe(cal, width="stretch", hide_index=True)

    st.caption("Regenerate these numbers with `python scripts/run_evaluation.py`; the results "
               "file is committed so the app never recomputes it per user.")


def page_about() -> None:
    st.header("About & data sources")
    meta = metadata_json()
    cols = st.columns(2)
    cols[0].markdown(
        f"**Dataset:** {meta['dataset']}  \n"
        f"**Source:** [{meta['source_name']}]({meta['source_url']})  \n"
        f"**Access:** {meta['access']}  \n"
        f"**Update frequency:** {meta['update_frequency']}  \n"
        f"**Built (UTC):** {meta['built_at_utc']}  \n"
        f"**Coverage:** {meta['coverage']['seasons']}, "
        f"{meta['coverage']['matches']} matches "
        f"({meta['coverage']['first_match']} → {meta['coverage']['last_match']})."
    )
    cols[1].markdown(f"**License / usage notes:**\n\n{meta['license_notes']}")

    st.subheader("What the system can and cannot do")
    st.markdown(
        "- Estimates are produced from a decay-weighted Poisson goals model; every estimate "
        "is labelled **Estimated probability**.\n"
        "- Only data available before kickoff feeds a pre-match estimate (no leakage); "
        "in-play estimates condition on the current score and time only.\n"
        "- The free dataset records results, half-time scores and some shot/corner/card "
        "counts — not lineups, injuries, tactics, or goal minutes. Those are never guessed; "
        "where they are missing the model says so.\n"
        "- Historical replay places goals inside the correct half at **estimated** minutes "
        "(flagged) because individual minutes are not in the source data.\n"
        "- A live feed (football-data.org) can be added with a `FOOTBALL_DATA_API_KEY` "
        "secret; without one the app runs in **demo mode** on the bundled data."
    )
    st.subheader("Measured results")
    data = evaluation_json()
    if data:
        for scenario in ("prematch", "halftime"):
            st.caption(f"{scenario}: " + ", ".join(
                f"{data['models'][m]['name']} log loss {v['log_loss']:.3f} (n={v['n']})"
                for m, v in data["scenario"][scenario].items()
            ))
    st.caption("AI-generated wording is optional and off by default; every page works without an "
               "AI service. If AI variables are configured, a sidebar checkbox can paraphrase the "
               "pre-match factor texts (unofficial rendering, falls back to human wording on any "
               "error).")


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #

st.sidebar.title("⚽ Football P&I")
page = st.sidebar.radio(
    "Navigate",
    ["Pre-match analysis", "Match timeline (replay)", "Match report", "Model evaluation", "About & data"],
)
if ai_supported():
    st.sidebar.checkbox("Paraphrase explanations with AI", key="ai_toggle", value=False)
    st.sidebar.caption("Optional AI-generated wording; off by default. Human wording stays the "
                       "default.")
show_badges()

matches = load_matches()
if matches.empty:
    st.error("No bundled dataset found. Run `python scripts/fetch_football_data.py`.")
    st.stop()

try:
    if page == "Pre-match analysis":
        page_prematch(matches)
    elif page == "Match timeline (replay)":
        page_timeline(matches)
    elif page == "Match report":
        page_report(matches)
    elif page == "Model evaluation":
        page_evaluation()
    else:
        page_about()
except Exception as exc:  # noqa: BLE001 - surface errors visibly in the app
    st.error(f"Something went wrong: {exc}")
    import traceback

    st.code(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.caption("Football Probability & Intelligence System · estimates only, never guarantees.")