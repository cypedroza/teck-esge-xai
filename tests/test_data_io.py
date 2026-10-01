import numpy as np

from data_io import target_circularity


def test_circularity_flags_weighted_sum_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(23, 4))
    y = X @ np.array([0.56, 0.26, 0.06, 0.12])
    assert target_circularity(X, y) > 0.999


def test_circularity_low_for_independent_target():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    y = rng.normal(size=200)
    assert target_circularity(X, y) < 0.2
