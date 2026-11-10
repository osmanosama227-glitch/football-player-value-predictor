"""Build the FORECASTING table.

One row per (player, reference date D):
  features = everything known at D (last-12-month stats, career, current + earlier values)
  target   = the player's market value ~HORIZON_DAYS after D  (future_value_eur)
"""
import numpy as np
import pandas as pd
from config import RAW, DATASET_FILE, LOOKUP_FILE, HORIZON_DAYS

CUM = ["cum_goals", "cum_assists", "cum_minutes", "cum_apps"]


def _read(name, **kw):
    for ext in (".csv.gz", ".csv"):
        p = RAW / f"{name}{ext}"
        if p.exists():
            return pd.read_csv(p, **kw)
    raise FileNotFoundError(f"{name}.csv(.gz) not found in {RAW}. Run download_data.py")


def cum_at(frame, col, cum, exact):
    left = frame[["row", "player_id", col]].sort_values(col)
    right = cum.sort_values("cum_date")
    m = pd.merge_asof(left, right, left_on=col, right_on="cum_date", by="player_id",
                      direction="backward", allow_exact_matches=exact)
    return m.set_index("row").sort_index()[CUM].fillna(0)


def window_features(frame, cum):
    """frame needs: row, player_id, end_date. Returns features indexed by row."""
    f = frame.copy()
    f["start_date"] = f["end_date"] - pd.Timedelta(days=365)
    f["start2_date"] = f["end_date"] - pd.Timedelta(days=730)
    end = cum_at(f, "end_date", cum, exact=False)      # strictly before end_date
    s1 = cum_at(f, "start_date", cum, exact=True)
    s2 = cum_at(f, "start2_date", cum, exact=True)
    cur, prev = (end - s1).clip(lower=0), (s1 - s2).clip(lower=0)
    out = pd.DataFrame(index=end.index)
    out["goals"], out["assists"] = cur["cum_goals"], cur["cum_assists"]
    out["minutes"], out["apps"] = cur["cum_minutes"], cur["cum_apps"]
    out["career_goals"], out["career_minutes"], out["career_apps"] = (
        end["cum_goals"], end["cum_minutes"], end["cum_apps"])
    out["minutes_prev"], out["goals_prev"] = prev["cum_minutes"], prev["cum_goals"]
    return out


def add_prev(left, date_col, val_all):
    """Attach the latest valuation STRICTLY BEFORE date_col (and the one before it)."""
    right = (val_all[["player_id", "date", "market_value_in_eur", "prev_value",
                      "player_club_domestic_competition_id"]]
             .rename(columns={"date": "pv_date", "market_value_in_eur": "pv_value",
                              "prev_value": "pv_prev",
                              "player_club_domestic_competition_id": "pv_league"})
             .sort_values("pv_date"))
    m = pd.merge_asof(left.sort_values(date_col), right, left_on=date_col, right_on="pv_date",
                      by="player_id", direction="backward", allow_exact_matches=False)
    m["log_prev_value"] = np.log1p(m["pv_value"])
    m["log_prev2_value"] = np.log1p(m["pv_prev"])
    m["days_since_prev"] = (m[date_col] - m["pv_date"]).dt.days
    return m


def main(min_minutes=180):
    players = _read("players", usecols=["player_id", "name", "position", "sub_position",
                                        "date_of_birth", "height_in_cm", "foot",
                                        "current_club_name",
                                        "current_club_domestic_competition_id"])
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")

    val_all = _read("player_valuations", usecols=["player_id", "date", "market_value_in_eur",
                                                  "player_club_domestic_competition_id"])
    val_all["date"] = pd.to_datetime(val_all["date"]).astype("datetime64[ns]")
    val_all = val_all.dropna(subset=["market_value_in_eur"])
    val_all = val_all[val_all["market_value_in_eur"] > 0]
    val_all = val_all.sort_values(["player_id", "date"]).reset_index(drop=True)
    g = val_all.groupby("player_id")
    val_all["prev_value"] = g["market_value_in_eur"].shift(1)
    val_all["prev2_value"] = g["market_value_in_eur"].shift(2)
    val_all["prev_date"] = g["date"].shift(1)

    # training samples: the last valuation of each calendar year per player
    val = val_all.copy()
    val["year"] = val["date"].dt.year
    val = (val.sort_values("date").drop_duplicates(["player_id", "year"], keep="last")
              .drop(columns=["prev_value", "prev2_value", "prev_date"]).reset_index(drop=True))
    val["row"] = val.index
    val["end_date"] = val["date"]

    app = _read("appearances", usecols=["player_id", "date", "goals", "assists", "minutes_played"])
    app["date"] = pd.to_datetime(app["date"]).astype("datetime64[ns]")
    daily = (app.groupby(["player_id", "date"], as_index=False)
                .agg(goals=("goals", "sum"), assists=("assists", "sum"),
                     minutes=("minutes_played", "sum"), apps=("date", "size"))
                .sort_values(["player_id", "date"]))
    for c in ["goals", "assists", "minutes", "apps"]:
        daily["cum_" + c] = daily.groupby("player_id")[c].cumsum()
    cum = daily[["player_id", "date"] + CUM].rename(columns={"date": "cum_date"})

    # ---------------- training table ----------------
    wf = window_features(val[["row", "player_id", "end_date"]], cum)
    df = val.set_index("row").join(wf).reset_index(drop=True)
    df = add_prev(df, "date", val_all).merge(players, on="player_id", how="left")
    df = df.rename(columns={"market_value_in_eur": "current_value_eur"})

    # target: the valuation closest to D + HORIZON (within +-120 days)
    tgt = (val_all[["player_id", "date", "market_value_in_eur"]]
           .rename(columns={"date": "target_date", "market_value_in_eur": "future_value_eur"})
           .sort_values("target_date"))
    df["target_key"] = df["date"] + pd.Timedelta(days=HORIZON_DAYS)
    df = pd.merge_asof(df.sort_values("target_key"), tgt, left_on="target_key",
                       right_on="target_date", by="player_id", direction="nearest",
                       tolerance=pd.Timedelta(days=120))
    df = df.dropna(subset=["future_value_eur"])

    df["age"] = (df["date"] - df["date_of_birth"]).dt.days / 365.25
    df = df[(df["age"].between(15, 45)) & (df["minutes"] >= min_minutes)].copy()
    df["goals_per90"] = df["goals"] / df["minutes"] * 90
    df["assists_per90"] = df["assists"] / df["minutes"] * 90
    df["log_current_value"] = np.log1p(df["current_value_eur"])
    df["value_trend"] = df["log_current_value"] - df["log_prev_value"]
    df["league"] = df["player_club_domestic_competition_id"].fillna("unknown")
    for c in ["position", "sub_position", "foot"]:
        df[c] = df[c].fillna("unknown")

    keep = ["player_id", "date", "target_date", "age", "position", "sub_position", "league", "foot",
            "height_in_cm", "goals", "assists", "minutes", "apps", "goals_per90", "assists_per90",
            "career_goals", "career_minutes", "career_apps", "minutes_prev", "goals_prev",
            "current_value_eur", "log_current_value", "log_prev_value", "log_prev2_value",
            "days_since_prev", "value_trend", "future_value_eur"]
    df[keep].to_csv(DATASET_FILE, index=False)
    print(f"saved {len(df):,} forecasting rows -> {DATASET_FILE}")
    chg = (df["future_value_eur"] / df["current_value_eur"]).median()
    print(f"median future/current value ratio: {chg:.2f}")

    # ---------------- lookup for the app ----------------
    # stats = each player's latest 12 months of play; value = latest valuation
    last = daily.groupby("player_id", as_index=False)["date"].max().rename(columns={"date": "last_date"})
    last["row"] = last.index
    last["end_date"] = last["last_date"] + pd.Timedelta(days=1)
    lk = last.join(window_features(last[["row", "player_id", "end_date"]], cum))
    lastv = (val_all.groupby("player_id").tail(1)
             .rename(columns={"date": "current_value_date", "market_value_in_eur": "current_value_eur",
                              "prev_value": "prev_value_eur", "prev2_value": "prev2_value_eur"}))
    lk = lk.merge(lastv, on="player_id", how="inner").merge(players, on="player_id", how="inner")
    lk["days_since_prev"] = (lk["current_value_date"] - lk["prev_date"]).dt.days
    lk["league"] = (lk["player_club_domestic_competition_id"]
                    .fillna(lk["current_club_domestic_competition_id"]).fillna("unknown"))
    lk["age"] = (lk["current_value_date"] - lk["date_of_birth"]).dt.days / 365.25
    lk["stats_gap_days"] = (lk["current_value_date"] - lk["last_date"]).dt.days
    active = lk["current_value_date"] >= lk["current_value_date"].max() - pd.Timedelta(days=365)
    lk = lk[active & (lk["minutes"] >= min_minutes) & lk["age"].between(15, 45) & lk["name"].notna()].copy()
    for c in ["position", "sub_position", "foot"]:
        lk[c] = lk[c].fillna("unknown")
    lk["window_end"] = lk["last_date"].dt.date
    lk = lk.rename(columns={"current_club_name": "club"})
    cols = ["player_id", "name", "club", "age", "position", "sub_position", "league", "foot",
            "height_in_cm", "goals", "assists", "minutes", "apps", "career_goals",
            "career_minutes", "career_apps", "minutes_prev", "goals_prev", "window_end",
            "stats_gap_days", "current_value_eur", "current_value_date", "prev_value_eur",
            "prev2_value_eur", "days_since_prev"]
    lk[cols].sort_values("name").to_csv(LOOKUP_FILE, index=False)
    print(f"saved {len(lk):,} players -> {LOOKUP_FILE}")


if __name__ == "__main__":
    main()
