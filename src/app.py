"""Run:  streamlit run src/app.py"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from config import MODEL_FILE, METRICS_FILE, RANGE_FILE, LOOKUP_FILE, MODELS, league_name

st.set_page_config(page_title="Player Value Forecaster", page_icon="⚽")
st.title("⚽ Football Player Future Market Value Forecaster")

if not MODEL_FILE.exists():
    st.error("No trained model found. Run: python src/train.py"); st.stop()
if not LOOKUP_FILE.exists():
    st.error("Player list not found. Run: python src/build_dataset.py"); st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_players():
    df = pd.read_csv(LOOKUP_FILE)
    df["label"] = (df["name"] + " · " + df["club"].fillna("no club") + " · "
                   + df["age"].round(0).astype(int).astype(str) + "y")
    return df


bundle, players = load_model(), load_players()
months = round(bundle.get("horizon_days", 365) / 30.4)


def idx(options, value):
    return options.index(value) if value in options else 0


def num(p, col, default):
    return default if p is None or pd.isna(p[col]) else p[col]


choice = st.selectbox("Choose a player (type to search)",
                      ["✏️ Enter stats manually"] + players["label"].tolist())
p = None if choice.startswith("✏️") else players[players["label"] == choice].iloc[0]
key = "manual" if p is None else str(p["player_id"])

if p is not None:
    st.info(f"Stats: the 12 months ending {p['window_end']}. Current value: Transfermarkt, "
            f"{p['current_value_date']}. Edit any value for a what-if.")
    if p["stats_gap_days"] > 365:
        st.warning("This player's latest stats in the dataset are more than a year older than his "
                   "latest valuation (his league may not be covered), so the forecast is less reliable.")

leagues = list(bundle["leagues"])
dl = "GB1" if p is None else p["league"]
if dl not in leagues:
    leagues.append(dl)
leagues = sorted(leagues, key=league_name)
pos_opts, sub_opts, foot_opts = bundle["positions"], bundle["sub_positions"], bundle["feet"]

c1, c2 = st.columns(2)
age = c1.number_input("Age", 15.0, 45.0, float(min(max(num(p, "age", 24.0), 15.0), 45.0)), 0.5, key=f"age{key}")
height = c2.number_input("Height (cm)", 150, 210, int(min(max(num(p, "height_in_cm", 180), 150), 210)), key=f"h{key}")
position = c1.selectbox("Position", pos_opts, idx(pos_opts, None if p is None else p["position"]), key=f"pos{key}")
sub_position = c2.selectbox("Sub-position", sub_opts, idx(sub_opts, None if p is None else p["sub_position"]), key=f"sub{key}")
league = c1.selectbox("League", leagues, idx(leagues, dl), format_func=league_name, key=f"lg{key}")
foot = c2.selectbox("Foot", foot_opts, idx(foot_opts, None if p is None else p["foot"]), key=f"ft{key}")
goals = c1.number_input("Goals (12 months)", 0, 200, int(num(p, "goals", 10)), key=f"g{key}")
assists = c2.number_input("Assists (12 months)", 0, 150, int(num(p, "assists", 5)), key=f"a{key}")
minutes = c1.number_input("Minutes played", 90, 8000, int(max(num(p, "minutes", 2400), 90)), 90, key=f"m{key}")
apps = c2.number_input("Appearances", 1, 120, int(max(num(p, "apps", 30), 1)), key=f"ap{key}")

cur_m = st.number_input("Current market value (€M)", 0.0, 400.0,
                        float(num(p, "current_value_eur", 0.0)) / 1e6, 0.1, key=f"cv{key}",
                        help="The forecast starts from today's value. Leave 0 to use stats only (less accurate).")

with st.expander("Career & value history"):
    d1, d2 = st.columns(2)
    career_goals = d1.number_input("Career goals (in dataset)", 0, 1500, int(num(p, "career_goals", goals * 3)), key=f"cg{key}")
    career_minutes = d2.number_input("Career minutes (in dataset)", 0, 60000, int(num(p, "career_minutes", minutes * 3)), key=f"cm{key}")
    career_apps = d1.number_input("Career appearances", 0, 1000, int(num(p, "career_apps", apps * 3)), key=f"ca{key}")
    minutes_prev = d2.number_input("Minutes in the year before", 0, 8000, int(num(p, "minutes_prev", minutes)), key=f"mp{key}")
    goals_prev = d1.number_input("Goals in the year before", 0, 200, int(num(p, "goals_prev", goals)), key=f"gp{key}")
    prev_m = d2.number_input("Previous value (€M, 0 = unknown)", 0.0, 400.0,
                             float(num(p, "prev_value_eur", 0.0)) / 1e6, 0.1, key=f"pv{key}")
    prev2_m = d1.number_input("Value before that (€M, 0 = unknown)", 0.0, 400.0,
                              float(num(p, "prev2_value_eur", 0.0)) / 1e6, 0.1, key=f"pv2{key}")
    days_prev = d2.number_input("Days between those two values", 0, 2000, int(num(p, "days_since_prev", 180)), key=f"dp{key}")

if league not in bundle["leagues"]:
    st.warning(f"{league_name(league)} was not in the training data, so the league does not affect the forecast.")

if st.button(f"Forecast value in {months} months", type="primary"):
    log_cur = np.log1p(cur_m * 1e6) if cur_m > 0 else np.nan
    log_prev = np.log1p(prev_m * 1e6) if prev_m > 0 else np.nan
    row = pd.DataFrame([{
        "age": age, "height_in_cm": height, "goals": goals, "assists": assists, "minutes": minutes,
        "apps": apps, "goals_per90": goals / minutes * 90, "assists_per90": assists / minutes * 90,
        "career_goals": career_goals, "career_minutes": career_minutes, "career_apps": career_apps,
        "minutes_prev": minutes_prev, "goals_prev": goals_prev,
        "log_current_value": log_cur, "log_prev_value": log_prev,
        "log_prev2_value": np.log1p(prev2_m * 1e6) if prev2_m > 0 else np.nan,
        "days_since_prev": days_prev if prev_m > 0 else np.nan,
        "value_trend": log_cur - log_prev,
        "position": position, "sub_position": sub_position, "league": league, "foot": foot,
    }])

    fs = "with_current_value" if cur_m > 0 else "stats_only"
    m = bundle["models"][fs]
    forecast = float(m["model"].predict(row[m["num"] + bundle["cat"]])[0])
    if cur_m > 0:
        st.metric(f"Forecast value in {months} months", f"€{forecast/1e6:,.1f}M",
                  delta=f"{(forecast / (cur_m * 1e6) - 1) * 100:+.0f}% vs today (€{cur_m:,.1f}M)")
    else:
        st.metric(f"Forecast value in {months} months (stats only)", f"€{forecast/1e6:,.1f}M")
        st.caption("Enter the current market value for a much more reliable forecast.")
    st.caption("A statistical forecast, not a guarantee and not a transfer fee. Injuries, transfers and "
               "form changes are not predictable from this data.")

with st.expander("How accurate is it? (tested on a held-out later period)"):
    if METRICS_FILE.exists():
        st.write("Compare each model with 'Naive (no change)': a forecast is only useful if it beats it.")
        st.dataframe(pd.read_csv(METRICS_FILE))
    if RANGE_FILE.exists():
        st.write("Error by future-value range (negative bias = under-predicts)")
        st.dataframe(pd.read_csv(RANGE_FILE))
    for fs in bundle["models"]:
        for img in (f"actual_vs_pred_{fs}.png", f"importance_{fs}.png"):
            if (MODELS / img).exists():
                st.image(str(MODELS / img))
