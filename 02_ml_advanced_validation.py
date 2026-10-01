"""
SCRIPT 2 — Explanatory model of ESG disclosure (reformulated module)
=====================================================================

Question: which observable characteristics are associated with the level of
Teck's ESG disclosure over 2001–2024?

* Target: log(1 + ESG Disclosure Index) — the article's central variable.
  It is NOT built from the explanatory variables (checked below).
* Features: report length, market liquidity, annual return, year-end price.
* Data: ``data/processed/esge_master.csv`` (n = 23 complete years).

Validation and reporting
------------------------
* **Leave-one-out (LOO)** R² is the headline out-of-sample measure: with 23
  annual observations it uses every year once as a hold-out.
* **TimeSeriesSplit** (train on earlier years only) is also reported. Because
  of the 2018 structural break, models trained before 2018 cannot extrapolate
  to post-2018 levels, so this score is expected to be strongly negative —
  the module is explanatory, not a forecasting tool.
* In-sample fit is reported separately and labelled as such.
* Bootstrap 95% intervals are out-of-bag (fit on the resample, score on the
  years not drawn).
* Hyper-parameters are tuned by LOO; scaling sits inside a ``Pipeline``.
* All randomness is seeded with ``config.RANDOM_STATE``.

Author: Cesar Yoshio Machado Pedroza
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import warnings
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, LeaveOneOut, TimeSeriesSplit,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import config
from data_io import FEATURES, TARGET, load_master, modelling_frame, target_circularity

warnings.filterwarnings("ignore")

SEED = config.RANDOM_STATE
TABLES, FIGURES, MODELS = config.OUTPUTS_TABLES, config.OUTPUTS_FIGURES, config.OUTPUTS_MODELS
LOO = LeaveOneOut()

RF_GRID = {"model__max_depth": [2, 3, 5], "model__n_estimators": [200, 500]}
XGB_GRID = {"model__max_depth": [2, 3], "model__learning_rate": [0.05, 0.1],
            "model__n_estimators": [100, 300]}


def pipe(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def loo_metrics(est, X, y) -> dict:
    p = cross_val_predict(est, X, y, cv=LOO)
    return {"R2_LOO": r2_score(y, p), "RMSE_LOO": float(np.sqrt(mean_squared_error(y, p))),
            "MAE_LOO": mean_absolute_error(y, p)}


def ts_r2(est, X, y) -> float:
    scores = []
    for tr, te in TimeSeriesSplit(n_splits=4).split(X):
        m = clone(est).fit(X[tr], y[tr])
        scores.append(r2_score(y[te], m.predict(X[te])))
    return float(np.mean(scores))


def oob_bootstrap(est, X, y, n_boot: int, rng: np.random.Generator) -> dict:
    n, r2s = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob) < 3:
            continue
        m = clone(est).fit(X[idx], y[idx])
        r2s.append(r2_score(y[oob], m.predict(X[oob])))
    r2s = np.array(r2s)
    return {"R2_OOB_median": float(np.median(r2s)),
            "R2_OOB_CI_low": float(np.percentile(r2s, 2.5)),
            "R2_OOB_CI_high": float(np.percentile(r2s, 97.5))}


def main(n_boot: int = 500) -> None:
    df = modelling_frame(load_master(config.DATA_PROCESSED / config.MASTER_FILE))
    X, y = df[FEATURES].to_numpy(float), df[TARGET].to_numpy(float)
    print("=" * 78)
    print(f"EXPLANATORY MODEL OF ESG DISCLOSURE — n = {len(y)} years "
          f"({df['year'].min()}–{df['year'].max()})")
    print("=" * 78)

    circ = target_circularity(X, y)
    pd.DataFrame([{"OLS_R2_target_on_features": circ}]).to_csv(
        TABLES / "target_circularity_check.csv", index=False)
    print(f"\nCircularity check — OLS in-sample R² of target on features: {circ:.3f}"
          f"{'  (WARNING: near-deterministic)' if circ > 0.99 else '  (not circular)'}")

    rf = GridSearchCV(pipe(RandomForestRegressor(random_state=SEED)), RF_GRID, cv=LOO,
                      scoring="neg_mean_squared_error", n_jobs=-1).fit(X, y)
    xgb = GridSearchCV(pipe(XGBRegressor(random_state=SEED, n_jobs=1)), XGB_GRID, cv=LOO,
                       scoring="neg_mean_squared_error", n_jobs=-1).fit(X, y)
    print(f"RF params:  {rf.best_params_}\nXGB params: {xgb.best_params_}")

    candidates = {
        "Baseline (mean)": pipe(DummyRegressor()),
        "Linear regression": pipe(LinearRegression()),
        "Random Forest": rf.best_estimator_,
        "XGBoost": xgb.best_estimator_,
    }
    rng = np.random.default_rng(SEED)
    rows = []
    for name, est in candidates.items():
        fitted = clone(est).fit(X, y)
        row = {"Model": name, **loo_metrics(est, X, y),
               "R2_TimeSeriesSplit": ts_r2(est, X, y),
               "R2_in_sample": r2_score(y, fitted.predict(X))}
        if name in ("Random Forest", "XGBoost"):
            row.update(oob_bootstrap(est, X, y, n_boot, rng))
        rows.append(row)
    perf = pd.DataFrame(rows)
    perf.to_csv(TABLES / "model_performance.csv", index=False)
    print("\n" + perf.round(3).to_string(index=False))

    joblib.dump(clone(rf.best_estimator_).fit(X, y), MODELS / "rf_pipeline.pkl")
    joblib.dump(clone(xgb.best_estimator_).fit(X, y), MODELS / "xgb_pipeline.pkl")

    for name in ("Random Forest", "XGBoost"):
        m = clone(candidates[name]).fit(X, y)
        pi = permutation_importance(m, X, y, n_repeats=50, random_state=SEED)
        imp = pd.DataFrame({"feature": FEATURES, "importance_mean": pi.importances_mean,
                            "importance_sd": pi.importances_std})
        pos = imp["importance_mean"].clip(lower=0)
        imp["share_%"] = pos / pos.sum() * 100
        imp.sort_values("importance_mean", ascending=False).to_csv(
            TABLES / f"permutation_importance_{name.lower().replace(' ', '_')}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    p = cross_val_predict(candidates["XGBoost"], X, y, cv=LOO)
    ax.scatter(np.expm1(y), np.expm1(p), c=df["year"], cmap="viridis", s=60)
    lim = [0, max(np.expm1(y).max(), np.expm1(p).max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("Observed ESG Disclosure Index")
    ax.set_ylabel("LOO prediction (XGBoost)")
    ax.set_title("Leave-one-out predictions (colour = year)")
    fig.tight_layout()
    fig.savefig(FIGURES / "loo_predictions.png", dpi=300)
    plt.close(fig)
    print("\nSaved: model_performance.csv, permutation importances, pipelines, LOO figure.")


if __name__ == "__main__":
    main()
