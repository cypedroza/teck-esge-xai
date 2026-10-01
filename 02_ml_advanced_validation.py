"""
SCRIPT 2 — Predictive modelling with temporal validation
=========================================================

Input: ``data/processed/esge_master_final.csv`` (features + ``ESGE_score``).

What this script reports, and what it does not
----------------------------------------------
* **Target-circularity check.** Before any modelling, the ESGE score is
  regressed on the four features by OLS. If that in-sample R² is ≈ 1 the
  target is (almost) a linear combination of the inputs and model R² values
  are not evidence of learning; the script prints a warning and records it.
* **Out-of-sample metrics** come from ``TimeSeriesSplit`` (5 folds; each fold
  trains only on earlier years). These are the numbers to quote as predictive
  performance.
* **In-sample fit** is reported separately and labelled as such.
* **Bootstrap intervals** use out-of-bag evaluation: each resample trains on
  the drawn years and is scored on the years not drawn.
* Scaling is done inside a ``Pipeline``, so no information from validation
  folds leaks into the scaler.
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
from sklearn.model_selection import (GridSearchCV, TimeSeriesSplit, cross_validate,
                                     learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import config
from data_io import FEATURES, TARGET, load_master, modelling_frame, target_circularity

warnings.filterwarnings("ignore", category=UserWarning)

SEED = config.RANDOM_STATE
TABLES, FIGURES, MODELS = config.OUTPUTS_TABLES, config.OUTPUTS_FIGURES, config.OUTPUTS_MODELS
CV = TimeSeriesSplit(n_splits=5)
SCORING = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}

# Grids as described in the article's Methods section
RF_GRID = {"model__max_depth": [3, 5, 10], "model__n_estimators": [100, 200, 500]}
XGB_GRID = {"model__max_depth": [3, 5, 7], "model__learning_rate": [0.01, 0.1, 0.3],
            "model__n_estimators": [100, 200, 300]}


def pipe(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def cv_metrics(estimator, X, y) -> dict:
    s = cross_validate(estimator, X, y, cv=CV, scoring=SCORING)
    return {
        "R2_cv": s["test_r2"].mean(), "R2_cv_sd": s["test_r2"].std(ddof=1),
        "RMSE_cv": -s["test_rmse"].mean(), "MAE_cv": -s["test_mae"].mean(),
    }


def oob_bootstrap(estimator, X, y, n_boot: int, rng: np.random.Generator) -> dict:
    """Out-of-bag bootstrap: fit on the resample, score on unseen rows."""
    n = len(y)
    r2s, rmses = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob) < 3:
            continue
        m = clone(estimator).fit(X[idx], y[idx])
        p = m.predict(X[oob])
        r2s.append(r2_score(y[oob], p))
        rmses.append(np.sqrt(mean_squared_error(y[oob], p)))
    r2s, rmses = np.array(r2s), np.array(rmses)
    return {
        "R2_oob_median": float(np.median(r2s)),
        "R2_oob_CI_low": float(np.percentile(r2s, 2.5)),
        "R2_oob_CI_high": float(np.percentile(r2s, 97.5)),
        "RMSE_oob_median": float(np.median(rmses)),
        "n_valid_resamples": int(len(r2s)),
    }


def main(n_boot: int = 1000) -> None:
    df = modelling_frame(load_master(config.DATA_PROCESSED / config.MODELLING_FILE))
    X, y = df[FEATURES].to_numpy(float), df[TARGET].to_numpy(float)
    print("=" * 78)
    print(f"MODELLING — n = {len(y)} years ({df['year'].min()}–{df['year'].max()})")
    print("=" * 78)

    # 0. Circularity check ----------------------------------------------------------
    circ = target_circularity(X, y)
    pd.DataFrame([{"OLS_R2_target_on_features": circ}]).to_csv(
        TABLES / "target_circularity_check.csv", index=False)
    print(f"\nOLS R² of {TARGET} on the features (in-sample): {circ:.4f}")
    if circ > 0.99:
        print("  WARNING: the target is almost a linear combination of the features.\n"
              "  Model R² values below measure how well a formula is reproduced, not\n"
              "  predictive ability. Interpret feature importances accordingly.")

    # 1. Models ---------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    rf_search = GridSearchCV(pipe(RandomForestRegressor(random_state=SEED)), RF_GRID,
                             cv=CV, scoring="r2", n_jobs=-1).fit(X, y)
    xgb_search = GridSearchCV(pipe(XGBRegressor(random_state=SEED, n_jobs=1,
                                                objective="reg:squarederror")),
                              XGB_GRID, cv=CV, scoring="r2", n_jobs=-1).fit(X, y)
    print(f"\nRF best params:  {rf_search.best_params_}")
    print(f"XGB best params: {xgb_search.best_params_}")

    candidates = {
        "Baseline (mean)": pipe(DummyRegressor(strategy="mean")),
        "Linear regression": pipe(LinearRegression()),
        "Random Forest": rf_search.best_estimator_,
        "XGBoost": xgb_search.best_estimator_,
    }

    rows = []
    for name, est in candidates.items():
        fitted = clone(est).fit(X, y)
        p_in = fitted.predict(X)
        row = {"Model": name, **cv_metrics(est, X, y),
               "R2_in_sample": r2_score(y, p_in),
               "RMSE_in_sample": float(np.sqrt(mean_squared_error(y, p_in))),
               "MAE_in_sample": mean_absolute_error(y, p_in)}
        if name in ("Random Forest", "XGBoost"):
            row.update(oob_bootstrap(est, X, y, n_boot, rng))
        rows.append(row)
    perf = pd.DataFrame(rows)
    perf.to_csv(TABLES / "model_performance.csv", index=False)
    print("\nOut-of-sample (TimeSeriesSplit) and in-sample metrics:")
    print(perf.round(4).to_string(index=False))

    # Persist fitted pipelines (scaler + model) for script 03
    joblib.dump(clone(rf_search.best_estimator_).fit(X, y), MODELS / "rf_pipeline.pkl")
    joblib.dump(clone(xgb_search.best_estimator_).fit(X, y), MODELS / "xgb_pipeline.pkl")

    # 2. Permutation importance (in-sample, fitted pipelines) -----------------------
    for name in ("Random Forest", "XGBoost"):
        model = clone(candidates[name]).fit(X, y)
        pi = permutation_importance(model, X, y, n_repeats=30, random_state=SEED)
        imp = pd.DataFrame({"feature": FEATURES, "importance_mean": pi.importances_mean,
                            "importance_sd": pi.importances_std})
        imp["share_%"] = imp["importance_mean"].clip(lower=0) / imp["importance_mean"].clip(lower=0).sum() * 100
        imp.sort_values("importance_mean", ascending=False).to_csv(
            TABLES / f"permutation_importance_{name.lower().replace(' ', '_')}.csv", index=False)

    # 3. Learning curves ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, name in zip(axes, ("Random Forest", "XGBoost")):
        sizes, tr, va = learning_curve(candidates[name], X, y, cv=CV, scoring="r2",
                                       train_sizes=np.linspace(0.3, 1.0, 6))
        ax.plot(sizes, tr.mean(1), "o-", label="train")
        ax.plot(sizes, va.mean(1), "s-", label="validation (later years)")
        ax.set_title(f"Learning curve — {name}")
        ax.set_xlabel("Training years")
        ax.set_ylabel("R²")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curves.png", dpi=300)
    plt.close(fig)
    print("\nSaved: model_performance.csv, permutation importances, pipelines, learning curves.")


if __name__ == "__main__":
    main()
