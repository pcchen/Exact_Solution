"""Public numerical helpers retain NumPy parameter sweeps and broadcasting."""

import numpy as np
import pytest
from mpmath import mp

from exact_solutions import ising_exact as ie


def test_numpy_parameter_sweeps_match_the_original_formulas():
    K = np.array([0.3, 0.4, 0.6])
    T = 1 / K
    np.testing.assert_allclose(ie.coupling(T), K, rtol=1e-15)
    np.testing.assert_allclose(ie.dual_coupling(K), -np.log(np.tanh(K)) / 2, rtol=1e-15)
    np.testing.assert_allclose(ie.detuning(K), K + np.log(np.tanh(K)) / 2, atol=2e-16)
    q = np.array([0, 0.5, 1, 2])
    expected = np.arccosh(np.cosh(2 * K[:, None]) ** 2 / np.sinh(2 * K[:, None]) - np.cos(q))
    actual = ie.dispersion(K[:, None], q)
    assert actual.shape == (3, 4)
    np.testing.assert_allclose(actual, expected, atol=3e-15)


@pytest.mark.parametrize("dps", [None, 60])
def test_broadcast_arrays_cross_both_stable_formula_branches(dps):
    K = np.array([0.001, 0.4, 2, 1000])[:, None]
    q = np.array([0, 0.5, 1, 2])[None, :]
    oracle = mp.clone()
    oracle.dps = 100
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = ie.dispersion(K, q, dps=dps)
        dual = ie.dual_coupling(K, dps=dps)
        detuning = ie.detuning(K, dps=dps)
    assert actual.shape == (4, 4)
    assert dual.shape == detuning.shape == K.shape
    tolerance = oracle.mpf("1e-14" if dps is None else "1e-55")
    for i, value in enumerate(K[:, 0]):
        k = oracle.mpf(float(value))
        expected_dual = oracle.atanh(oracle.exp(-2 * k))
        delta = k - expected_dual
        assert abs(oracle.mpf(dual[i, 0]) - expected_dual) < tolerance
        assert abs(oracle.mpf(detuning[i, 0]) - delta) < tolerance * max(1, abs(delta))
        for j, angle in enumerate(q[0]):
            expected = 2 * oracle.asinh(oracle.sqrt(oracle.sinh(delta) ** 2 + oracle.sin(float(angle) / 2) ** 2))
            assert abs(oracle.mpf(actual[i, j]) - expected) < tolerance * max(1, expected)


def test_zero_dimensional_and_empty_arrays():
    for dps in (None, 60):
        assert ie.coupling(np.array(2), dps=dps) == ie.coupling(2, dps=dps)
        assert ie.dual_coupling(np.array(0.4), dps=dps) == ie.dual_coupling(0.4, dps=dps)
        assert ie.dispersion(np.array(0.4), np.array(1), dps=dps) == ie.dispersion(0.4, 1, dps=dps)
        assert ie.coupling(np.empty((0, 2)), dps=dps).shape == (0, 2)
        assert ie.dual_coupling(np.empty((0, 2)), dps=dps).shape == (0, 2)
        assert ie.dispersion(np.empty((0, 1)), np.ones((1, 2)), dps=dps).shape == (0, 2)


def test_absolute_coupling_detuning_keeps_accuracy_near_kc():
    oracle = mp.clone()
    oracle.dps = 100
    center = float(oracle.log1p(oracle.sqrt(2)) / 2)
    lower = upper = center
    values = [center]
    for _ in range(200):
        lower = np.nextafter(lower, 0.0)
        upper = np.nextafter(upper, 1.0)
        values.extend((lower, upper))
    K = np.array(values)
    expected = [oracle.mpf(k) + oracle.log(oracle.tanh(oracle.mpf(k))) / 2 for k in values]
    actual = ie.detuning(K)
    base = K + np.log(np.tanh(K)) / 2
    # Preserve the previous direct formula on every NumPy/libm build, and also
    # check an independent oracle with a bound tighter than the regressed path.
    np.testing.assert_array_equal(actual, base)
    assert max(abs(oracle.mpf(a) - b) for a, b in zip(actual, expected)) < oracle.mpf("6e-17")
