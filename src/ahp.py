"""
Gaussian AHP (Monte Carlo) utilities.

Each upper-triangular judgment a_ij (i < j) is perturbed multiplicatively,
a'_ij = a_ij · ε_ij with ε_ij ~ N(1, σ²) truncated to stay positive, i.e. the
judgment's standard deviation is σ·a_ij. The diagonal is kept at 1 and the
lower triangle is set to the reciprocals, so every simulated matrix is a
valid positive reciprocal matrix.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

# Saaty's Random Index (Saaty, 1980)
RANDOM_INDEX: Dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}


def principal_weights(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalised principal eigenvector and λ_max of a positive matrix."""
    eigvals, eigvecs = np.linalg.eig(matrix)
    k = int(np.argmax(eigvals.real))
    w = np.abs(eigvecs[:, k].real)
    return w / w.sum(), float(eigvals[k].real)


def consistency_ratio(matrix: np.ndarray) -> float:
    """Saaty consistency ratio CR = CI / RI, CI = (λ_max − n)/(n − 1)."""
    n = matrix.shape[0]
    ri = RANDOM_INDEX.get(n, 0.0)
    if ri == 0:
        return 0.0
    _, lam = principal_weights(matrix)
    return ((lam - n) / (n - 1)) / ri


def validate_reciprocal(matrix: np.ndarray, tol: float = 1e-9) -> None:
    """Raise ValueError unless ``matrix`` is square, positive and reciprocal."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("Pairwise matrix must be square.")
    if (m <= 0).any():
        raise ValueError("Pairwise matrix must be strictly positive.")
    if not np.allclose(np.diag(m), 1.0, atol=tol):
        raise ValueError("Pairwise matrix diagonal must be 1.")
    if not np.allclose(m * m.T, 1.0, atol=1e-6):
        raise ValueError("Pairwise matrix must be reciprocal (a_ji = 1/a_ij).")


def perturb(base: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """One Gaussian-perturbed positive reciprocal matrix."""
    n = base.shape[0]
    m = np.ones_like(base, dtype=float)
    iu = np.triu_indices(n, k=1)
    eps = rng.normal(1.0, sigma, size=len(iu[0]))
    eps = np.clip(eps, 1e-3, None)  # keep judgments positive
    m[iu] = base[iu] * eps
    m[(iu[1], iu[0])] = 1.0 / m[iu]
    return m


def gaussian_ahp_monte_carlo(
    base: np.ndarray,
    sigma: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate weights and CR for ``n_simulations`` perturbed matrices.

    Returns
    -------
    weights : (n_simulations, n) array
    crs     : (n_simulations,) array
    """
    base = np.asarray(base, dtype=float)
    validate_reciprocal(base)
    n = base.shape[0]
    weights = np.empty((n_simulations, n))
    crs = np.empty(n_simulations)
    for s in range(n_simulations):
        m = perturb(base, sigma, rng)
        w, lam = principal_weights(m)
        weights[s] = w
        ri = RANDOM_INDEX.get(n, 0.0)
        crs[s] = ((lam - n) / (n - 1)) / ri if ri else 0.0
    return weights, crs


def summarise(
    weights: np.ndarray,
    crs: np.ndarray,
    criteria: Sequence[str],
    base: np.ndarray,
    cr_threshold: float = 0.10,
) -> tuple[pd.DataFrame, dict]:
    """Weight table (mean, sd, 95% percentile interval, CV) and run diagnostics.

    The rank-reversal rate is the share of simulations whose ordering differs
    from the ordering of the **deterministic** (unperturbed) matrix.
    """
    base_w, _ = principal_weights(np.asarray(base, dtype=float))
    base_rank = np.argsort(-base_w)
    reversals = np.mean([not np.array_equal(np.argsort(-w), base_rank) for w in weights])

    table = pd.DataFrame({
        "Criterion": list(criteria),
        "Deterministic": base_w,
        "Mean": weights.mean(axis=0),
        "Std": weights.std(axis=0, ddof=1),
        "CI_95_Lower": np.percentile(weights, 2.5, axis=0),
        "CI_95_Upper": np.percentile(weights, 97.5, axis=0),
    })
    table["CV_%"] = table["Std"] / table["Mean"] * 100
    diagnostics = {
        "n_simulations": int(len(crs)),
        "deterministic_CR": float(consistency_ratio(np.asarray(base, dtype=float))),
        "mean_CR": float(crs.mean()),
        f"P(CR<{cr_threshold})_%": float((crs < cr_threshold).mean() * 100),
        "rank_reversal_%": float(reversals * 100),
    }
    return table.sort_values("Mean", ascending=False).reset_index(drop=True), diagnostics
