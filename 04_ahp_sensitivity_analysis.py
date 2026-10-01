"""
SCRIPT 4 — Gaussian AHP with Monte Carlo sensitivity analysis
==============================================================

* 4×4 pairwise matrix (Saaty scale) for the ESGE criteria, defined a priori by
  the author (see ``src/config.py``; no expert panel — a declared limitation).
* Each upper-triangular judgment is perturbed multiplicatively by N(1, σ²);
  diagonal kept at 1, lower triangle reciprocal (``src/ahp.py``).
* σ ∈ {0.05, 0.10, 0.15, 0.20}; N simulations per level (config.N_SIMULATIONS).
* Reported per σ: mean weights and 95% percentile intervals, P(CR < 0.10),
  and the rank-reversal rate relative to the deterministic ordering.

All randomness comes from a seeded ``numpy.random.Generator`` (config.RANDOM_STATE),
so results are exactly reproducible.

Author: Cesar Yoshio Machado Pedroza
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import config
from ahp import gaussian_ahp_monte_carlo, summarise, consistency_ratio

TABLES = config.OUTPUTS_TABLES
FIGURES = config.OUTPUTS_FIGURES


def main() -> None:
    base = np.array(config.AHP_PAIRWISE, dtype=float)
    criteria = config.AHP_CRITERIA
    rng = np.random.default_rng(config.RANDOM_STATE)

    print("=" * 78)
    print("GAUSSIAN AHP — MONTE CARLO SENSITIVITY")
    print("=" * 78)
    print(pd.DataFrame(base, index=criteria, columns=criteria).round(3))
    print(f"Deterministic CR = {consistency_ratio(base):.4f}")

    sens_rows, results = [], {}
    for sigma in config.SIGMA_LEVELS:
        w, cr = gaussian_ahp_monte_carlo(base, sigma, config.N_SIMULATIONS, rng)
        table, diag = summarise(w, cr, criteria, base, config.CR_THRESHOLD)
        results[sigma] = (w, cr, table)
        row = {"sigma": sigma, **diag}
        for _, r in table.iterrows():
            row[f"{r['Criterion']}_mean"] = r["Mean"]
        sens_rows.append(row)
        print(f"\nσ = {sigma:.2f} | P(CR<{config.CR_THRESHOLD}) = "
              f"{diag[f'P(CR<{config.CR_THRESHOLD})_%']:.2f}% | "
              f"rank reversal = {diag['rank_reversal_%']:.2f}%")
        print(table.round(4).to_string(index=False))

    pd.DataFrame(sens_rows).to_csv(TABLES / "ahp_sensitivity_analysis.csv", index=False)

    # Final table at the baseline σ = 0.10
    final = results[0.10][2]
    final.to_csv(TABLES / "ahp_weights.csv", index=False)

    # Figures ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, sigma in zip(axes.ravel(), config.SIGMA_LEVELS):
        w = results[sigma][0]
        ax.boxplot([w[:, i] for i in range(len(criteria))], tick_labels=criteria)
        ax.set_title(f"σ = {sigma:.2f}")
        ax.set_ylabel("Weight")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Gaussian AHP weights by uncertainty level")
    fig.tight_layout()
    fig.savefig(FIGURES / "ahp_weights_distribution_by_sigma.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, sigma in zip(axes.ravel(), config.SIGMA_LEVELS):
        cr = results[sigma][1]
        ax.hist(cr, bins=50, color="steelblue", edgecolor="black", alpha=0.75)
        ax.axvline(config.CR_THRESHOLD, color="red", linestyle="--", label="CR = 0.10")
        ax.set_title(f"σ = {sigma:.2f} | P(CR<0.10) = {(cr < config.CR_THRESHOLD).mean() * 100:.1f}%")
        ax.set_xlabel("Consistency ratio")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "cr_distribution_by_sigma.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    err = np.vstack([final["Mean"] - final["CI_95_Lower"], final["CI_95_Upper"] - final["Mean"]])
    ax.bar(final["Criterion"], final["Mean"], yerr=err, capsize=5, color="teal", alpha=0.85)
    ax.set_ylabel("Weight (mean, 95% interval)")
    ax.set_title(f"Gaussian AHP weights (σ = 0.10, N = {config.N_SIMULATIONS:,})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "ahp_weights.png", dpi=300)
    plt.close(fig)

    print("\nSaved: ahp_sensitivity_analysis.csv, ahp_weights.csv and 3 figures.")


if __name__ == "__main__":
    main()
