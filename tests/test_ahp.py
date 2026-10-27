import numpy as np
import pytest

from ahp import (consistency_ratio, gaussian_ahp_monte_carlo, perturb,
                 principal_weights, summarise, validate_reciprocal)

BASE = np.array([[1, 3, 5, 7], [1/3, 1, 3, 5], [1/5, 1/3, 1, 3], [1/7, 1/5, 1/3, 1]], float)


def test_deterministic_weights_and_cr():
    w, _ = principal_weights(BASE)
    np.testing.assert_allclose(w, [0.565, 0.262, 0.118, 0.055], atol=0.002)
    assert consistency_ratio(BASE) < 0.10


def test_perturbed_matrix_is_valid_reciprocal():
    rng = np.random.default_rng(0)
    for _ in range(200):
        validate_reciprocal(perturb(BASE, 0.2, rng))


def test_monte_carlo_is_reproducible_with_seed():
    w1, c1 = gaussian_ahp_monte_carlo(BASE, 0.1, 500, np.random.default_rng(42))
    w2, c2 = gaussian_ahp_monte_carlo(BASE, 0.1, 500, np.random.default_rng(42))
    np.testing.assert_array_equal(w1, w2)
    np.testing.assert_array_equal(c1, c2)


def test_summary_rank_reversal_zero_without_noise():
    w, c = gaussian_ahp_monte_carlo(BASE, 1e-9, 50, np.random.default_rng(0))
    _, diag = summarise(w, c, list("ESGC"), BASE)
    assert diag["rank_reversal_%"] == 0.0
    assert diag["P(CR<0.1)_%"] == 100.0


def test_invalid_matrix_rejected():
    bad = BASE.copy()
    bad[0, 0] = 1.2
    with pytest.raises(ValueError):
        validate_reciprocal(bad)
