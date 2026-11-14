"""exact_solutions.ising_exact: against the transfer matrix built from the Boltzmann weights,
against the Ising CFT, and against the values of the script this module grew out of."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from exact_solutions import ising_exact as ie

REFERENCE = json.loads((Path(__file__).parent / "data" / "original_reference.json").read_text())
TEMPERATURES = (0.7 * ie.Tc, ie.Tc, 1.4 * ie.Tc)
NAMED = {"0.7Tc": 0.7 * ie.Tc, "Tc": ie.Tc, "1.4Tc": 1.4 * ie.Tc}


# ------------------------------------------------------ against the earlier script's values
# `tests/data/original_reference.json` holds what that script returned, run once on
# 2026-11-13 (its sha256 is recorded in the file).  The script itself is not part of this
# repository; these are its numbers.
def test_gammas_match_the_recorded_values():
    for key, expected in REFERENCE["gammas"].items():
        name, n = key.split("|")
        ours = ie.gammas(NAMED[name], int(n))
        expected = np.array(expected)
        assert ours.shape == expected.shape == (2 * int(n),), key
        assert np.max(np.abs(ours - expected)) <= 1e-12 * max(np.max(np.abs(expected)), 1.0), key
        assert np.sign(ours[0]) == np.sign(expected[0]), key          # Kaufman's signed gamma_0


def test_ei_kmax_matches_the_recorded_values():
    """Every (n, kmax) of the recording, and every prefix length: `Ei_kmax` truncates, so a
    shorter num_collect must give the head of the longer list."""
    compared = 0
    for key, expected in REFERENCE["Ei_kmax"].items():
        name, n, kmax = key.split("|")
        n, kmax = int(n), (None if kmax == "none" else int(kmax))
        expected = np.array(expected)
        for num_collect in (1, 2, 3, 7, 12):
            head = expected[:num_collect]
            ours = ie.Ei_kmax(n, NAMED[name], num_collect, kmax=kmax)
            assert ours.shape == head.shape, (key, num_collect)
            assert np.max(np.abs(ours - head), initial=0.0) <= 1e-11, (key, num_collect)
            compared += 1
    assert compared == 900


# ------------------------------------------- against the transfer matrix of the Boltzmann weights
def _dense_sector_levels(n, K):
    """-ln(eigenvalues) of V[s, s'] = exp(K/2 sum s_i s_{i+1}) exp(K sum s_i s'_i) exp(K/2 sum s'_i s'_{i+1})
    in every (charge, momentum) sector: V projected with the characters of the spin flip and
    of the translation, both written as permutations of the 2^n configurations."""
    dim = 2 ** n
    states = np.arange(dim)
    spins = 1 - 2 * ((states[:, None] >> np.arange(n)) & 1)
    in_row = (spins * np.roll(spins, -1, axis=1)).sum(axis=1)
    V = np.exp(0.5 * K * in_row)[:, None] * np.exp(K * (spins @ spins.T)) * np.exp(0.5 * K * in_row)[None, :]
    flip = states ^ (dim - 1)
    shift = ((states << 1) | (states >> (n - 1))) & (dim - 1)
    found = {}
    for charge in (0, 1):
        for k in range(n):
            projector = np.zeros((dim, dim), dtype=complex)
            image = states.copy()
            for j in range(n):
                phase = np.exp(-2j * np.pi * k * j / n)
                projector[image, states] += phase
                projector[flip[image], states] += phase * (-1) ** charge
                image = shift[image]
            projector /= 2 * n
            rank = int(round(projector.trace().real))
            if rank:
                kept = np.sort(np.linalg.eigvalsh(projector.conj().T @ V @ projector))[::-1][:rank]
                found[(charge, (k + n // 2) % n - n // 2)] = np.sort(-np.log(kept))
    return found


@pytest.mark.parametrize("T", TEMPERATURES)
def test_levels_are_the_transfer_matrix_spectrum_sector_by_sector(T):
    """All 2^n levels, n = 1 .. 8: charge and momentum of every level, and its value."""
    for n in range(1, 9):
        dense = _dense_sector_levels(n, ie.coupling(T))
        table = ie.levels(n, T)
        assert len(table.E) == 2 ** n == sum(len(v) for v in dense.values())
        assert np.all(np.diff(table.E) >= -1e-12)
        for (charge, spin), expected in dense.items():
            ours = np.sort(table.E[(table.charge == charge) & (table.spin == spin)])
            assert ours.shape == expected.shape, (n, charge, spin)
            assert np.max(np.abs(ours - expected)) <= 1e-10, (n, charge, spin)
            sector = ie.levels(n, T, charge=charge, spin=spin)                       # the heap, filtered
            assert np.max(np.abs(np.sort(sector.E) - expected)) <= 1e-10, (n, charge, spin)
            direct = ie.levels(n, T, len(expected) + 3, charge=charge, spin=spin)    # the dynamic programme
            assert direct.E.shape == expected.shape, (n, charge, spin)               # ... which finds no more
            assert np.max(np.abs(direct.E - expected)) <= 1e-10, (n, charge, spin)
            assert np.all(direct.charge == charge) and np.all(direct.spin == spin)


def test_sector_tables_hold_every_sector_and_agree_with_the_global_list():
    """The two algorithms against each other where neither is complete: n = 24, the lowest
    200 levels of the whole spectrum redistributed over the sectors."""
    n, T = 24, 1.1 * ie.Tc
    tables = ie.sector_tables(n, T, 12)
    assert sorted(tables) == sorted((c, s) for c in (0, 1) for s in range(-n // 2, n // 2))
    table = ie.levels(n, T, 200)
    for (charge, spin), sector in tables.items():
        ours = table.X[(table.charge == charge) & (table.spin == spin)]
        shared = min(len(ours), len(sector.X))
        assert np.max(np.abs(ours[:shared] - sector.X[:shared]), initial=0.0) <= 1e-11
        both = table.fermions[(table.charge == charge) & (table.spin == spin)][:shared]
        assert np.array_equal(both, sector.fermions[:shared]) or len(set(np.round(sector.X[:shared], 9))) < shared
    assert sum(len(sector.X) for sector in tables.values()) == 2 * n * 12


def test_a_far_momentum_sector_is_as_cheap_as_spin_zero():
    """Charge 1, spin 60 of the width-127 strip at T_c: the lowest level is one fermion of
    that momentum above the sigma level (gamma is subadditive, so no three do better)."""
    n = 127
    far = ie.levels(n, ie.Tc, 5, charge=1, spin=60)
    x_sigma = ie.levels(n, ie.Tc, 2).X[1]
    one_fermion = x_sigma + ie.dispersion(ie.coupling(ie.Tc), 2 * np.pi * 60 / n) * n / (2 * np.pi)
    assert far.fermions[0] == 1 and abs(far.X[0] - one_fermion) <= 1e-11
    assert np.all(far.spin == 60) and np.all(np.diff(far.X) >= 0)


def test_the_fermion_number_rule_changes_at_tc():
    """Charge 1 holds an odd number of fermions above T_c and an even number below; charge 0
    always an even number."""
    for T, odd_sector in ((1.4 * ie.Tc, 1), (0.7 * ie.Tc, 0)):
        table = ie.levels(6, T)
        assert np.all(table.fermions[table.charge == 0] % 2 == 0)
        assert np.all(table.fermions[table.charge == 1] % 2 == odd_sector)
    below = ie.levels(40, 0.7 * ie.Tc, 2)                        # the ordered phase: a doublet
    assert below.charge.tolist() == [0, 1] and 0 < below.X[1] < 1e-9


# ------------------------------------------------------------------------------ the Ising CFT
def test_critical_levels_are_the_ising_tower():
    table = ie.levels(64, ie.Tc, 9)
    content = [(round(x, 2), int(c), int(s)) for x, c, s in zip(table.X, table.charge, table.spin)]
    assert content[:3] == [(0.0, 0, 0), (0.13, 1, 0), (1.0, 0, 0)]                  # 1, sigma, eps
    assert sorted(content[3:5]) == [(1.12, 1, -1), (1.12, 1, 1)]                    # d sigma
    assert sorted(content[5:9]) == [(2.0, 0, -2), (2.0, 0, -1), (2.0, 0, 1), (2.0, 0, 2)]  # T, Tbar, d eps
    assert abs(table.X[1] - 0.125) < 2e-5 and abs(table.X[2] - 1.0) < 3e-4


def test_ground_energy_carries_the_bulk_free_energy_and_c_one_half():
    """E_0 = n f - pi c / (6 n) + O(n^-3) with Onsager's f and c = 1/2: the absolute
    normalization of lambda, independently of the dense check."""
    K = ie.coupling(ie.Tc)
    kappa = 2 * np.sinh(2 * K) / np.cosh(2 * K) ** 2
    integral = quad(lambda t: np.log((1 + np.sqrt(max(1 - (kappa * np.sin(t)) ** 2, 0.0))) / 2), 0, np.pi / 2)[0]
    f = -(np.log(2 * np.cosh(2 * K)) + integral / np.pi)
    for n, tolerance in ((32, 2e-3), (128, 2e-4)):
        c = (n * f - ie.ground_energy(n, ie.Tc)) * 6 * n / np.pi
        assert abs(c - 0.5) < tolerance


def test_pinned_levels_of_the_width_8_strip():
    """X at T_c, L = 8, by (spin S, charge q).  These values also come out of an independent
    enumeration by particle number, and of the dense check above."""
    expected = {(0, 0): (0.0, 0.9873897227955192, 2.701078019561491),
                (0, 1): (0.12583958395624914, 2.0301460528023556),
                (1, 1): (1.077992818379302, 2.7547956914952336),
                (2, 0): (1.844233871178505, 2.4215163564013933),
                (-2, 0): (1.844233871178505, 2.4215163564013933)}
    for (spin, charge), values in expected.items():
        ours = ie.levels(8, ie.Tc, len(values), charge=charge, spin=spin).X
        assert np.max(np.abs(ours - np.array(values))) <= 1e-12, (spin, charge)


# ------------------------------------------------------------------------------- large widths
def test_wide_strips_cost_nothing():
    """Widths and level counts an enumeration by particle number cannot reach."""
    table = ie.levels(127, ie.Tc, 60)
    assert np.max(np.abs(table.E - ie.Ei_kmax(127, ie.Tc, 60))) <= 1e-11
    assert abs(table.X[1] - 0.125) < 1e-5
    wide = ie.levels(4096, 1.01 * ie.Tc, 2000)
    assert len(wide.E) == 2000 and np.all(np.diff(wide.E) >= -1e-11)
    spin_two = ie.levels(512, ie.Tc, 5, charge=0, spin=2)
    assert np.all(spin_two.spin == 2) and abs(spin_two.X[0] - 2.0) < 1e-4


def test_refusals():
    with pytest.raises(ValueError):
        ie.levels(0, ie.Tc, 3)
    with pytest.raises(ValueError):
        ie.levels(8, ie.Tc, 3, charge=2)
    with pytest.raises(ValueError):
        ie.levels(40, ie.Tc)
    with pytest.raises(ValueError):
        ie.sector_tables(8, ie.Tc, 0)
