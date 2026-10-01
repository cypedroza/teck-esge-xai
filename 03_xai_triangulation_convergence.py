"""
SCRIPT 3 — XAI triangulation: SHAP, LIME and DiCE
==================================================

Inputs: ``data/processed/esge_master.csv`` and the fitted pipeline
``outputs/models/xgb_pipeline.pkl`` produced by script 02 (scaler + model, so
the explanations use exactly the preprocessing the model was trained with).

* SHAP — TreeExplainer on the fitted XGBoost model, global importance = mean |φ|.
* LIME — LimeTabularExplainer on the standardised features, 1,000 perturbations
  per instance, explaining N = 10 representative years chosen by k-medoids
  (PAM-style alternation on standardised features, seeded).
* Convergence — Pearson and Spearman correlation between the normalised SHAP
  and LIME importance vectors. With four features (df = 2) this is a
  descriptive indicator only; the p-values are reported to make that explicit.
* DiCE — genuine counterfactuals from ``dice_ml`` (model-agnostic random
  sampling; gradient-based DiCE does not apply to tree ensembles) for the most
  recent year, asking for a 10–30 % higher predicted ESG Disclosure Index
  (converted to the model's log scale). Sparsity (number
  of features changed) and proximity (mean absolute change in SD units) are
  reported for each counterfactual.

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
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy import stats

from config import config
from data_io import FEATURES, TARGET, load_master, modelling_frame

warnings.filterwarnings("ignore")

SEED = config.RANDOM_STATE
TABLES, FIGURES, MODELS = config.OUTPUTS_TABLES, config.OUTPUTS_FIGURES, config.OUTPUTS_MODELS
N_LIME_INSTANCES = 10
LIME_SAMPLES = 1000


def k_medoids(Z: np.ndarray, k: int, rng: np.random.Generator, n_iter: int = 100) -> np.ndarray:
    """Indices of k medoids (simple alternating k-medoids, Euclidean)."""
    n = len(Z)
    if k >= n:
        return np.arange(n)
    D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    medoids = rng.choice(n, size=k, replace=False)
    for _ in range(n_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new = medoids.copy()
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members):
                new[c] = members[np.argmin(D[np.ix_(members, members)].sum(axis=1))]
        if set(new) == set(medoids):
            break
        medoids = new
    return np.sort(medoids)


def main() -> None:
    df = modelling_frame(load_master(config.DATA_PROCESSED / config.MASTER_FILE))
    X = df[FEATURES].to_numpy(float)
    pipeline = joblib.load(MODELS / "xgb_pipeline.pkl")
    scaler, model = pipeline.named_steps["scaler"], pipeline.named_steps["model"]
    Z = scaler.transform(X)
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print(f"XAI TRIANGULATION — n = {len(X)} years")
    print("=" * 78)

    # 1. SHAP ---------------------------------------------------------------------
    sv = shap.TreeExplainer(model).shap_values(Z)
    shap_imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES, name="SHAP")
    (shap_imp.rename("importance").rename_axis("feature").reset_index()
     .sort_values("importance", ascending=False)
     .to_csv(TABLES / "shap_importance.csv", index=False))
    plt.figure()
    shap.summary_plot(sv, Z, feature_names=FEATURES, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. LIME ---------------------------------------------------------------------
    explainer = LimeTabularExplainer(Z, feature_names=FEATURES, mode="regression",
                                     discretize_continuous=False, random_state=SEED)
    chosen = k_medoids(Z, N_LIME_INSTANCES, rng)
    lime_rows = []
    for i in chosen:
        exp = explainer.explain_instance(Z[i], model.predict, num_features=len(FEATURES),
                                         num_samples=LIME_SAMPLES)
        weights = dict(exp.as_map()[1])
        lime_rows.append({"year": int(df.loc[i, "year"]),
                          **{FEATURES[j]: abs(weights.get(j, 0.0)) for j in range(len(FEATURES))}})
    lime_df = pd.DataFrame(lime_rows)
    lime_df.to_csv(TABLES / "lime_instance_explanations.csv", index=False)
    lime_imp = lime_df[FEATURES].mean().rename("LIME")

    # 3. Convergence ------------------------------------------------------------------
    shap_n, lime_n = shap_imp / shap_imp.sum(), lime_imp / lime_imp.sum()
    pr, pp = stats.pearsonr(shap_n, lime_n)
    sr, sp = stats.spearmanr(shap_n, lime_n)
    comp = pd.DataFrame({"SHAP_raw": shap_imp, "LIME_raw": lime_imp,
                         "SHAP_share": shap_n, "LIME_share": lime_n})
    comp["abs_diff_share"] = (comp["SHAP_share"] - comp["LIME_share"]).abs()
    comp.sort_values("SHAP_share", ascending=False).to_csv(
        TABLES / "shap_lime_comparison.csv", index_label="feature")
    conv = {"pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp,
            "n_features": len(FEATURES), "mean_abs_diff_share": comp["abs_diff_share"].mean(),
            "lime_years": ",".join(map(str, lime_df["year"]))}
    pd.DataFrame([conv]).to_csv(TABLES / "shap_lime_convergence.csv", index=False)
    print("\n" + comp.round(4).to_string())
    print(f"\nPearson r = {pr:.3f} (p = {pp:.3f}); Spearman ρ = {sr:.3f} (p = {sp:.3f}); "
          f"df = {len(FEATURES) - 2} — descriptive only.")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(FEATURES))
    ax.bar(x - 0.2, comp["SHAP_share"], 0.4, label="SHAP")
    ax.bar(x + 0.2, comp["LIME_share"], 0.4, label="LIME")
    ax.set_xticks(x, FEATURES, rotation=20, ha="right")
    ax.set_ylabel("Share of total importance")
    ax.set_title(f"SHAP vs LIME (Pearson r = {pr:.2f}, 4 features)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_lime_convergence.png", dpi=300)
    plt.close(fig)

    # 4. DiCE counterfactuals -------------------------------------------------------------
    try:
        import dice_ml
        frame = df[FEATURES + [TARGET]].copy()
        data = dice_ml.Data(dataframe=frame, continuous_features=FEATURES, outcome_name=TARGET)
        dmodel = dice_ml.Model(model=pipeline, backend="sklearn", model_type="regressor")
        dice = dice_ml.Dice(data, dmodel, method="random")
        last = df["year"].idxmax()
        query = df.loc[[last], FEATURES]
        pred = float(pipeline.predict(query)[0])          # log(1 + index)
        lo, hi = np.log1p(np.expm1(pred) * 1.10), np.log1p(np.expm1(pred) * 1.30)
        cf = dice.generate_counterfactuals(query, total_CFs=5, desired_range=[lo, hi],
                                           random_seed=SEED)
        cfs = cf.cf_examples_list[0].final_cfs_df
        if cfs is None or cfs.empty:
            print("\nDiCE: no counterfactual found in the requested range.")
        else:
            sd = df[FEATURES].std(ddof=1)
            diff = (cfs[FEATURES] - query.iloc[0]) / sd
            cfs = cfs.assign(
                sparsity=(diff.abs() > 1e-9).sum(axis=1),
                proximity_sd=diff.abs().mean(axis=1),
                **{f"{f}_change_%": (cfs[f] / query.iloc[0][f] - 1) * 100 for f in FEATURES},
            )
            cfs.insert(0, "query_year", int(df.loc[last, "year"]))
            cfs.insert(1, "query_predicted_index", float(np.expm1(pred)))
            cfs["predicted_disclosure_index"] = np.expm1(pipeline.predict(cfs[FEATURES]))
            cfs = cfs.drop(columns=[TARGET])  # DiCE rounds this column; use the model prediction
            cfs.to_csv(TABLES / "dice_counterfactuals.csv", index=False)
            print(f"\nDiCE: {len(cfs)} counterfactuals for {int(df.loc[last, 'year'])} "
                  f"(predicted index {np.expm1(pred):,.1f} → [{np.expm1(lo):,.1f}, {np.expm1(hi):,.1f}])")
            print(cfs.round(3).to_string(index=False))
    except Exception as exc:  # report, never fabricate
        print(f"\nDiCE could not be run: {exc}")

    print("\nSaved: SHAP/LIME tables and figures, convergence metrics (and DiCE table if found).")


if __name__ == "__main__":
    main()
