"""Train + compare FORECASTING models (value ~12 months ahead) with a purged time split.

stats_only         : performance, age, position, league, career  (no value information)
with_current_value : the same + current value, earlier values and the value trend
                     -> predicts the GROWTH from today's value (GrowthRegressor)
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATASET_FILE, MODEL_FILE, METRICS_FILE, RANGE_FILE, MODELS, HORIZON_DAYS
from modeling import GrowthRegressor

BASE_NUM = ["age", "height_in_cm", "goals", "assists", "minutes", "apps",
            "goals_per90", "assists_per90", "career_goals", "career_minutes",
            "career_apps", "minutes_prev", "goals_prev"]
VALUE_NUM = ["log_current_value", "log_prev_value", "log_prev2_value",
             "days_since_prev", "value_trend"]
FEATURE_SETS = {"stats_only": BASE_NUM, "with_current_value": BASE_NUM + VALUE_NUM}
CAT = ["position", "sub_position", "league", "foot"]
TARGET = "future_value_eur"


def make_pipeline(model, num, growth):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                              sparse_output=False), CAT),
    ])
    inner = Pipeline([("pre", pre), ("model", model)])
    if growth:   # predict log(future/current)
        return GrowthRegressor(inner)
    return TransformedTargetRegressor(inner, func=np.log1p, inverse_func=np.expm1)


def evaluate(y, p, cur):
    y, cur = np.asarray(y), np.asarray(cur)
    p = np.clip(p, 0, None)
    moved = y != cur   # Transfermarkt values move in steps; judge direction only when it changed
    return {"MAE_eur": mean_absolute_error(y, p),
            "RMSE_eur": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p),
            "RMSLE": np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)),
            "median_abs_pct_err": float(np.median(np.abs(p - y) / y)) * 100,
            "direction_acc_pct": float(np.mean((p[moved] > cur[moved]) == (y[moved] > cur[moved]))) * 100}


def by_range(y, p, fs, name):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.clip(p, 0, None)})
    d["range"] = pd.cut(d["y"], [0, 1e6, 5e6, 20e6, 50e6, np.inf],
                        labels=["<1M", "1-5M", "5-20M", "20-50M", "50M+"])
    d["ae"] = (d["p"] - d["y"]).abs() / 1e6
    d["ape"] = (d["p"] - d["y"]).abs() / d["y"] * 100
    d["bias"] = (d["p"] - d["y"]) / d["y"] * 100      # negative = under-predicts
    g = d.groupby("range", observed=True).agg(n=("y", "size"), MAE_M=("ae", "mean"),
                                              median_abs_pct_err=("ape", "median"),
                                              median_bias_pct=("bias", "median")).reset_index()
    g.insert(0, "model", name); g.insert(0, "feature_set", fs)
    return g


def plots(fs, name, best, pred, test, num):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test[TARGET], pred, s=5, alpha=0.3)
    lim = [1e5, max(test[TARGET].max(), pred.max())]
    ax.plot(lim, lim, "r--"); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual future value (EUR)"); ax.set_ylabel("Forecast (EUR)")
    ax.set_title(f"{fs} / {name}")
    fig.tight_layout(); fig.savefig(MODELS / f"actual_vs_pred_{fs}.png", dpi=150); plt.close(fig)

    sample = test.sample(min(3000, len(test)), random_state=1)
    imp = permutation_importance(best, sample[num + CAT], sample[TARGET], n_repeats=5,
                                 random_state=1, scoring="neg_mean_absolute_error")
    s = pd.Series(imp.importances_mean, index=num + CAT).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    s.plot.barh(ax=ax); ax.set_title(f"{fs}: permutation importance (MAE increase)")
    fig.tight_layout(); fig.savefig(MODELS / f"importance_{fs}.png", dpi=150); plt.close(fig)


def main():
    df = pd.read_csv(DATASET_FILE, parse_dates=["date", "target_date"])
    # Purged split: training labels must be fully in the past of the test period.
    cutoff = df["date"].max() - pd.Timedelta(days=365)
    train, test = df[df["target_date"] <= cutoff], df[df["date"] > cutoff]
    print(f"forecast horizon {HORIZON_DAYS}d | train {len(train):,} rows (labels <= {cutoff.date()}), "
          f"test {len(test):,} rows (reference dates > {cutoff.date()})")

    def new_models():
        return {
            "Baseline (median)": DummyRegressor(strategy="median"),
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=10),
            "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=3,
                                                   n_jobs=-1, random_state=42),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.05, l2_regularization=1.0, random_state=42),
        }

    cur_test = test["current_value_eur"].values
    results, ranges, bundle = [], [], {"models": {}}
    for fs, num in FEATURE_SETS.items():
        growth = fs == "with_current_value"
        print(f"\n=== {fs} ===")
        fitted = {}
        for name, m in new_models().items():
            if growth and name == "Baseline (median)":
                continue   # meaningless as a growth model; the naive benchmark replaces it
            pipe = make_pipeline(m, num, growth).fit(train[num + CAT], train[TARGET])
            pred = pipe.predict(test[num + CAT])
            results.append({"feature_set": fs, "model": name,
                            **evaluate(test[TARGET], pred, cur_test)})
            ranges.append(by_range(test[TARGET], pred, fs, name))
            fitted[name] = (pipe, pred)
            print(name, "done")
        if growth:   # the benchmark any forecaster must beat: "value stays the same"
            results.append({"feature_set": fs, "model": "Naive (no change)",
                            **evaluate(test[TARGET], cur_test.astype(float), cur_test)})
            ranges.append(by_range(test[TARGET], cur_test.astype(float), fs, "Naive (no change)"))
        cand = [r for r in results if r["feature_set"] == fs and r["model"] in fitted
                and r["model"] != "Baseline (median)"]
        best_name = min(cand, key=lambda r: r["RMSLE"])["model"]
        best, pred = fitted[best_name]
        plots(fs, best_name, best, pred, test, num)
        final = make_pipeline(new_models()[best_name], num, growth).fit(df[num + CAT], df[TARGET])
        bundle["models"][fs] = {"model": final, "name": best_name, "num": num}
        print("best:", best_name)

    res = pd.DataFrame(results).sort_values(["feature_set", "RMSLE"])
    res.round(3).to_csv(METRICS_FILE, index=False)
    pd.concat(ranges).round(2).to_csv(RANGE_FILE, index=False)
    print("\n", res.round(3).to_string(index=False))

    bundle.update({"cat": CAT, "positions": sorted(df["position"].unique()),
                   "sub_positions": sorted(df["sub_position"].unique()),
                   "leagues": sorted(df["league"].unique()), "feet": sorted(df["foot"].unique()),
                   "horizon_days": HORIZON_DAYS})
    joblib.dump(bundle, MODEL_FILE)
    print("saved model + plots in", MODELS)


if __name__ == "__main__":
    main()
