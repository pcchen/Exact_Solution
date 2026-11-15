"""Exact criticality and stable nearby coordinates, without an absolute Tc input."""

import numpy as np
import pytest
from mpmath import mp

from exact_solutions import ising_exact as ie


def context(dps=140):
    ctx = mp.clone()
    ctx.dps = dps
    return ctx


@pytest.mark.parametrize("dps", [None, 30, 80])
def test_exact_critical_dispersion_and_normalization(dps):
    oracle = context()
    tolerance = oracle.mpf("1e-13") if dps is None else oracle.power(10, -dps + 5)
    assert ie.detuning_reduced(1, dps=dps) == 0
    for n in (1, 4, 64):
        gamma = ie.gammas_reduced(1, n, dps=dps)
        assert gamma[0] == 0
        expected = [oracle.acosh(2 - oracle.cos(oracle.pi * min(l, 2 * n - l) / n))
                    for l in range(2 * n)]
        assert max(abs(oracle.mpf(a) - b) for a, b in zip(gamma, expected)) < tolerance
        ground = -(n * oracle.log(2) + oracle.fsum(expected[1::2])) / 2
        assert abs(oracle.mpf(ie.ground_energy_reduced(n, 1, dps=dps)) - ground) < tolerance * n
    # n=1 has only the critical zero mode and pi mode; the normalization identity
    # log(2 sinh(2Kc))=log(2) also follows directly from the two Boltzmann eigenvalues.
    eigenvalue = oracle.exp(oracle.log1p(oracle.sqrt(2)) / 2) * (
        oracle.exp(oracle.log1p(oracle.sqrt(2)) / 2) + oracle.exp(-oracle.log1p(oracle.sqrt(2)) / 2))
    assert abs(oracle.mpf(ie.ground_energy_reduced(1, 1, dps=dps)) + oracle.log(eigenvalue)) < tolerance


@pytest.mark.parametrize("dps", [40, 80])
def test_detuning_retains_relative_accuracy_close_to_criticality(dps):
    working, oracle = context(dps), context(180)
    for sign in (-1, 1):
        r = 1 + sign * working.power(10, -dps + 8)
        K = oracle.log1p(oracle.sqrt(2)) / (2 * oracle.mpf(r))
        expected = K + oracle.log(oracle.tanh(K)) / 2
        delta = ie.detuning_reduced(r, dps=dps)
        assert abs(oracle.mpf(delta) / expected - 1) < oracle.power(10, -dps + 4)
        assert (delta < 0) == (sign > 0)
        assert ie.gammas_reduced(r, 8, dps=dps)[0] == 2 * delta
        levels = ie.levels_reduced(8, r, 8, charge=1, dps=dps)
        assert np.all(levels.fermions % 2 == (1 if sign > 0 else 0))


def test_adjacent_double_temperatures_are_not_snapped_to_criticality():
    oracle = context()
    for r in (np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0)):
        K = oracle.log1p(oracle.sqrt(2)) / (2 * oracle.mpf(float(r)))
        expected = K + oracle.log(oracle.tanh(K)) / 2
        actual = ie.detuning_reduced(r)
        assert actual != 0
        assert abs(oracle.mpf(actual) / expected - 1) < oracle.mpf("1e-14")


@pytest.mark.parametrize("dps", [None, 60])
def test_absolute_adapters_use_the_same_context_as_reduced_calls(dps):
    ctx = context(dps or 15)
    for T in (1, 2, 3):
        r = T / ie.critical_temperature(dps=dps)
        expected = ie.levels_reduced(6, r, 8, dps=dps)
        actual = ie.levels(6, T, 8, dps=dps)
        for a, b in zip(actual, expected):
            assert np.array_equal(a, b)
        assert ie.ground_energy(6, T, dps=dps) == ie.ground_energy_reduced(6, r, dps=dps)
        assert np.array_equal(ie.gammas(T, 6, dps=dps), ie.gammas_reduced(r, 6, dps=dps))
        absolute = ie.sector_tables(6, T, 3, dps=dps)
        reduced = ie.sector_tables_reduced(6, r, 3, dps=dps)
        for key in absolute:
            assert np.array_equal(absolute[key].X, reduced[key].X)
        assert all(ctx.isfinite(e) for e in actual.E)


def test_a_double_rounded_absolute_tc_is_still_detuned_at_high_precision():
    rounded_Tc = float(ie.critical_temperature())
    absolute = ie.gammas(rounded_Tc, 8, dps=80)
    reduced = ie.gammas_reduced(1, 8, dps=80)
    assert reduced[0] == 0 and absolute[0] != 0
    exact_levels = ie.levels_reduced(64, 1, 2, dps=80)
    rounded_levels = ie.levels(64, rounded_Tc, 2, dps=80)
    difference = abs(exact_levels.X[1] - rounded_levels.X[1])
    assert context().mpf("1e-17") < difference < context().mpf("1e-14")
    assert not hasattr(ie, "Tc")


@pytest.mark.parametrize("dps", [None, 60])
def test_reduced_helpers_broadcast_across_critical_and_low_temperature_branches(dps):
    r = np.array([0.001, 0.7, 1, 1.4, 100])[:, None]
    q = np.array([0, 0.5, 1, 2])[None, :]
    oracle = context()
    tolerance = oracle.mpf("1e-13" if dps is None else "1e-54")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = ie.dispersion_reduced(r, q, dps=dps)
        delta = ie.detuning_reduced(r, dps=dps)
        coupling = ie.coupling_reduced(r, dps=dps)
    assert actual.shape == (5, 4)
    assert delta.shape == coupling.shape == r.shape
    assert delta[2, 0] == 0
    for i, value in enumerate(r[:, 0]):
        K = oracle.log1p(oracle.sqrt(2)) / (2 * oracle.mpf(float(value)))
        expected_delta = K + oracle.log(oracle.tanh(K)) / 2
        assert abs(oracle.mpf(delta[i, 0]) - expected_delta) < tolerance * max(1, abs(expected_delta))
        for j, angle in enumerate(q[0]):
            expected = 2 * oracle.asinh(oracle.sqrt(oracle.sinh(expected_delta) ** 2 + oracle.sin(float(angle) / 2) ** 2))
            assert abs(oracle.mpf(actual[i, j]) - expected) < tolerance * max(1, expected)
    assert ie.detuning_reduced(np.empty((0, 2)), dps=dps).shape == (0, 2)
    assert ie.dispersion_reduced(np.empty((0, 1)), np.ones((1, 2)), dps=dps).shape == (0, 2)


@pytest.mark.parametrize("r", ["0", "-1", "nan", "inf"])
def test_invalid_reduced_temperatures_are_rejected(r):
    for dps in (None, 60):
        with pytest.raises(ValueError, match="T/Tc"):
            ie.levels_reduced(4, r, 3, dps=dps)
