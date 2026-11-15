"""Precision checks against Boltzmann matrices and independent scalar identities.

No double-precision oracle or float conversion participates in the error bounds.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from mpmath import mp

from exact_solutions import ising_exact as ie


def context(dps=120):
    ctx = mp.clone()
    ctx.dps = dps
    return ctx


def dense_sectors(n, T, ctx):
    """Construct V and project with spin flip / translation, independently of the
    dispersion, mode enumeration and vacuum formulas used by the implementation.
    """
    dim, K = 2 ** n, 1 / T
    spins = [[1 - 2 * ((state >> i) & 1) for i in range(n)] for state in range(dim)]
    bonds = [sum(row[i] * row[(i + 1) % n] for i in range(n)) for row in spins]
    V = ctx.matrix(dim)
    for i in range(dim):
        for j in range(dim):
            vertical = sum(a * b for a, b in zip(spins[i], spins[j]))
            V[i, j] = ctx.exp(K * (ctx.mpf(bonds[i] + bonds[j]) / 2 + vertical))
    found = {}
    for charge in (0, 1):
        for momentum in range(n):
            P = ctx.matrix(dim)
            for state in range(dim):
                image = state
                for step in range(n):
                    phase = ctx.exp(-2 * ctx.j * ctx.pi * momentum * step / n)
                    P[image, state] += phase / (2 * n)
                    P[image ^ (dim - 1), state] += (-1) ** charge * phase / (2 * n)
                    image = ((image << 1) | (image >> (n - 1))) & (dim - 1)
            rank = int(ctx.nint(sum(P[i, i].real for i in range(dim))))
            if rank:
                values = ctx.eighe(P.H * V * P, eigvals_only=True)
                found[(charge, (momentum + n // 2) % n - n // 2)] = sorted(
                    -ctx.log(value) for value in list(values)[-rank:])
    return found


@pytest.mark.parametrize("ratio", ["0.7", "1", "1.4"])
@pytest.mark.parametrize("reduced", [False, True])
def test_high_precision_spectrum_against_boltzmann_definition(ratio, reduced):
    ctx = context()
    T = ctx.mpf(ratio) * 2 / ctx.log(1 + ctx.sqrt(2))
    calculate = ie.levels_reduced if reduced else ie.levels
    temperature = ratio if reduced else T
    for n in (1, 2, 3, 4):
        expected = dense_sectors(n, T, ctx)
        ground = min(min(e) for e in expected.values())
        table = calculate(n, temperature, dps=80)
        assert table.E.dtype == table.X.dtype == object
        assert table.charge.dtype.kind == table.spin.dtype.kind == table.fermions.dtype.kind == "i"
        assert len(table.E) == 2 ** n
        for (charge, spin), energies in expected.items():
            selected = (table.charge == charge) & (table.spin == spin)
            direct = calculate(n, temperature, len(energies) + 2, charge=charge, spin=spin, dps=80)
            expected_x = [(e - ground) * n / (2 * ctx.pi) for e in energies]
            for actual_e, actual_x in ((table.E[selected], table.X[selected]), (direct.E, direct.X)):
                assert len(actual_e) == len(energies)
                assert max(abs(ctx.mpf(a) - e) for a, e in zip(actual_e, energies)) < ctx.mpf("1e-70")
                assert max(abs(ctx.mpf(a) - x) for a, x in zip(actual_x, expected_x)) < ctx.mpf("1e-70")


def test_more_precision_gains_digits_not_just_a_wider_output_type():
    ctx = context(140)
    T = ctx.mpf("1.2345678901234567890123456789")
    expected = dense_sectors(3, T, ctx)
    ground = min(min(e) for e in expected.values())
    sigma = (expected[(1, 0)][0] - ground) * 3 / (2 * ctx.pi)
    errors = []
    for dps in (30, 60, 100):
        actual = ie.levels(3, T, 1, charge=1, spin=0, dps=dps).X[0]
        error = abs(ctx.mpf(actual) - sigma)
        assert error < ctx.power(10, -dps + 5)
        errors.append(error)
    assert errors[1] < errors[0] * ctx.mpf("1e-20")
    assert errors[2] < errors[1] * ctx.mpf("1e-30")


def test_high_precision_constants_dispersion_and_critical_detuning():
    ctx = context()
    Tc = 2 / ctx.log(1 + ctx.sqrt(2))
    assert abs(ctx.mpf(ie.critical_temperature(dps=80)) - Tc) < ctx.mpf("1e-79")
    assert abs(ctx.mpf(ie.critical_temperature()) - Tc) < ctx.mpf("1e-15")
    for sign in (-1, 1):
        T = Tc * (1 + sign * ctx.mpf("1e-45"))
        K = 1 / T
        expected = K + ctx.log(ctx.tanh(K)) / 2
        assert abs(ctx.mpf(ie.detuning(K, dps=80)) - expected) < ctx.mpf("1e-79")
        gamma = ie.gammas(T, 7, dps=80)
        assert (gamma[0] < 0) == (sign > 0)
        table = ie.levels(7, T, 8, charge=1, dps=80)
        assert np.all(table.fermions % 2 == (1 if sign > 0 else 0))
        # The arccosh form is independent of the implementation's asinh formula.
        for l in range(1, 14):
            q = ctx.pi * min(l, 14 - l) / 7
            oracle = ctx.acosh(ctx.cosh(2 * K) ** 2 / ctx.sinh(2 * K) - ctx.cos(q))
            assert abs(ctx.mpf(gamma[l]) - oracle) < ctx.mpf("1e-78")
    q = [0, 1, 2]
    result = ie.dispersion("0.4", q, dps=80)
    assert result.shape == (3,)
    assert ie.dispersion("0.4", 0, dps=80) == result[0]


def test_sector_tables_and_legacy_entry_point_preserve_precision():
    T = ie.critical_temperature(dps=80) * 11 / 10
    all_levels = ie.levels(10, T, dps=80)
    tables = ie.sector_tables(10, T, 8, dps=80)
    ctx = context()
    for (charge, spin), sector in tables.items():
        expected = all_levels.X[(all_levels.charge == charge) & (all_levels.spin == spin)][:8]
        assert len(sector.X) == len(expected)
        assert max((abs(ctx.mpf(a) - ctx.mpf(b)) for a, b in zip(sector.X, expected)), default=0) < ctx.mpf("1e-75")
    legacy = ie.Ei_kmax(10, T, 20, dps=80)
    assert legacy.dtype == object
    assert max(abs(ctx.mpf(a) - ctx.mpf(b)) for a, b in zip(legacy, all_levels.E[:20])) < ctx.mpf("1e-75")


def test_precision_is_in_cache_key_and_global_context_is_untouched():
    before = mp.dps
    low = ie.sector_tables_reduced(8, 1, 4, dps=25)
    high = ie.sector_tables_reduced(8, 1, 4, dps=80)  # same exact ratio, distinct precision
    again = ie.sector_tables_reduced(8, 1, 4, dps=25)
    assert mp.dps == before
    assert high[(1, 0)].X[0].context.dps == 80
    assert low[(1, 0)].X[0].context.dps == 25
    assert low[(1, 0)].X[0] == again[(1, 0)].X[0]
    assert low[(1, 0)].X[0] != high[(1, 0)].X[0]
    # A subsequent call cannot change arithmetic or formatting of an earlier result.
    assert (high[(1, 0)].X[0] / 3).context.dps == 80


def test_momentum_selection_keeps_excitation_coordinates():
    T = ie.critical_temperature(dps=60)
    sector = ie.sector_tables(64, T, 3, dps=60)[(1, 0)]
    selected = ie.levels(64, T, 3, charge=1, spin=0, dps=60)
    assert np.array_equal(sector.X, selected.X)
    # The extensive total energies have already lost some digits of these gaps.
    ground = ie.ground_energy(64, T, dps=60)
    pi = context(60).pi
    reconstructed = (selected.E - ground) * 64 / (2 * pi)
    assert any(a != b for a, b in zip(reconstructed, selected.X))


def test_ordered_doublet_is_resolved_beyond_double_precision():
    ctx = context(160)
    T = 2 / ctx.log(1 + ctx.sqrt(2)) * ctx.mpf("0.7")
    low = ie.levels(64, T, 2, dps=80)
    high = ie.levels(64, T, 2, dps=130)
    assert low.charge.tolist() == high.charge.tolist() == [0, 1]
    assert 0 < high.X[1] < ctx.mpf("1e-16")
    assert abs(ctx.mpf(low.X[1]) / ctx.mpf(high.X[1]) - 1) < ctx.mpf("1e-50")


@pytest.mark.parametrize("dps", [None, 80])
def test_low_temperature_avoids_exponential_overflow(dps):
    # K=1000: even the unscaled sinh(2K) cannot fit in double precision.
    table = ie.levels(4, "0.001", 4, dps=dps)
    ctx = context(100)
    assert all(ctx.isfinite(e) for e in table.E)
    assert abs(ctx.mpf(table.E[0]) + 8000) < ctx.mpf("1e-10")
    assert ie.ground_energy(4, "0.001", dps=dps) == table.E[0]
    gamma = ie.gammas("0.001", 4, dps=dps)
    assert all(ctx.isfinite(g) for g in gamma)


@pytest.mark.parametrize("K", ["1e-30", "0.1", "1", "20"])
def test_dual_coupling_survives_small_and_large_couplings(K):
    ctx = context()
    expected = ctx.atanh(ctx.exp(-2 * ctx.mpf(K)))
    assert ie.dual_coupling(K) > 0
    assert abs(ctx.mpf(ie.dual_coupling(K)) / expected - 1) < ctx.mpf("1e-15")
    assert abs(ctx.mpf(ie.dual_coupling(K, dps=80)) / expected - 1) < ctx.mpf("1e-78")


@pytest.mark.parametrize("K", ["0.01", "1", "2", "20", "1000"])
def test_scaled_dispersion_agrees_with_unscaled_high_precision_formula(K):
    ctx = context()
    k = ctx.mpf(K)
    delta = k + ctx.log(ctx.tanh(k)) / 2
    for dps, tolerance in ((None, "1e-14"), (80, "1e-77")):
        # Use exactly representable q so the two arithmetic paths have identical input.
        q = [0, 0.5, 1, 2, 3]
        expected = [2 * ctx.asinh(ctx.sqrt(ctx.sinh(delta) ** 2 + ctx.sin(ctx.mpf(x) / 2) ** 2)) for x in q]
        actual = ie.dispersion(K, q, dps=dps)
        assert max(abs(ctx.mpf(a) - e) / max(1, e) for a, e in zip(actual, expected)) < ctx.mpf(tolerance)


def test_angle_arrays_are_converted_before_numpy_can_round_mixed_inputs():
    ctx = context()
    # NumPy normally infers float64 for this list, losing one unit of the second angle.
    angles = [0.0, 2 ** 53 + 1]
    actual = ie.dispersion("0.4", angles, dps=80)
    K = ctx.mpf("0.4")
    expected = ctx.acosh(ctx.cosh(2 * K) ** 2 / ctx.sinh(2 * K) - ctx.cos(angles[1]))
    assert abs(ctx.mpf(actual[1]) - expected) < ctx.mpf("1e-75")


@pytest.mark.parametrize("dps", [True, 0, -10, 30.5, "80"])
def test_invalid_precision_is_rejected(dps):
    with pytest.raises(ValueError, match="dps"):
        ie.levels(2, "2", 2, dps=dps)


@pytest.mark.parametrize("T", ["0", "-1", "nan", "inf"])
def test_invalid_temperature_is_rejected(T):
    for dps in (None, 80):
        with pytest.raises(ValueError, match="T"):
            ie.levels(2, T, 2, dps=dps)


def test_default_import_and_calculation_do_not_require_mpmath():
    code = '''
import builtins
original = builtins.__import__
def without_mpmath(name, *args, **kwargs):
    if name == "mpmath" or name.startswith("mpmath."):
        raise ImportError("mpmath deliberately unavailable")
    return original(name, *args, **kwargs)
builtins.__import__ = without_mpmath
from exact_solutions import ising_exact as ie
assert ie.levels(4, ie.critical_temperature(), 3).X.dtype.kind == "f"
try:
    ie.levels(4, ie.critical_temperature(), 3, dps=80)
except ImportError as error:
    assert "exact-solutions[precision]" in str(error)
else:
    raise AssertionError("missing optional dependency was not reported")
'''
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])
