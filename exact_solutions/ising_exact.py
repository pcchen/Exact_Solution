"""The exact spectrum of the 2D Ising transfer matrix on a strip of finite width.

`gammas` and `Ei_kmax` are the two functions of the script this module grew out of, with the
same numbers and a faster search; `levels` and `sector_tables` add the charge and momentum of
every level.  The formula, the conventions and the checks are set out in `docs/ising_exact.md`.

THE OBJECT.  n sites in a row, periodic, coupling K = J / (kB T) on every bond.  The
row-to-row transfer matrix

    V[s, s'] = exp(K/2 sum_i s_i s_{i+1})  exp(K sum_i s_i s'_i)  exp(K/2 sum_i s'_i s'_{i+1})

gives Z(n x M torus) = Tr V^M.  It commutes with the spin flip and with the translation, so
a level carries a CHARGE (0 flip-even, 1 flip-odd) and a MOMENTUM 2 pi S / n.  Its
eigenvalues (Onsager 1944, Kaufman 1949; Schultz, Mattis and Lieb 1964) are

    lambda = (2 sinh 2K)^(n/2) exp( 1/2 sum_q (+-) gamma_q ),
    cosh gamma_q = cosh 2K cosh 2K* - cos q,          tanh K* = exp(-2K),

with one sign per allowed angle q:  q = (2j+1) pi / n in the charge-0 sector, q = 2j pi / n
in the charge-1 sector, j = 0 .. n-1.  Which sign patterns occur is where the conventions
part -- see `README.md`, "Three ways to count the same levels".

THIS MODULE'S READING (the one `levels` reports).  Every gamma_q >= 0 is the energy of one
fermion of momentum q, and E = -ln lambda is a vacuum energy plus the energies of the
occupied modes:

    E = E_vac(charge) + sum_{occupied q} gamma_q,     E_vac = -(n/2) ln(2 sinh 2K) - 1/2 sum_q gamma_q,
    S = sum_{occupied q} q / (2 pi / n)   mod n.

    charge 0:  an EVEN number of fermions.
    charge 1:  an ODD number above T_c, an EVEN number below; at T_c the q = 0 fermion
               costs nothing and the two rules give the same levels.

WHAT THIS MODULE ADDS.
  * The lowest N levels are found best-first with a heap (each subset of modes is reached
    once from a smaller one by adding or advancing its largest mode), O(N log N) in the
    number of levels visited, for any width.  The earlier script enumerated every even-size
    subset of the lowest min(n, num_collect) gammas, about 2^(min - 1) of them, each summed
    over all n modes and appended to a growing array.
  * gamma_q = 2 asinh sqrt( sinh^2(K - K*) + sin^2(q/2) ), which is the same function with no
    cancellation at small q or near T_c (cosh 2K cosh 2K* = 1 + cosh 2(K - K*)); gamma_q and
    gamma_(2 pi - q) are the same floating-point number.
  * Charge, momentum and fermion number come out with every level.  The lowest levels of one
    (charge, momentum) sector come from a second algorithm, a dynamic programme over the modes
    (`sector_tables`): the heap walks the WHOLE spectrum in order, and the number of levels
    below the first one of a high-momentum sector grows without bound.

numpy and the standard library only.
"""

from __future__ import annotations

import functools
import heapq
import itertools
import math
from typing import Iterator, NamedTuple

import numpy as np

J = 1
kB = 1
Tc = 2 / np.log(1 + np.sqrt(2))

__all__ = ["Tc", "coupling", "dual_coupling", "detuning", "dispersion", "gammas", "Ei_kmax", "Levels", "levels",
           "sector_tables", "ground_energy"]


def coupling(T):
    """K = J / (kB T)."""
    return J / (kB * T)


def dual_coupling(K):
    """K* with tanh K* = exp(-2K), equivalently sinh 2K sinh 2K* = 1 (Kaufman's H*)."""
    return -0.5 * np.log(np.tanh(K))


def detuning(K):
    """K - K*: zero at T_c, positive below it, negative above."""
    return K + 0.5 * np.log(np.tanh(K))


def dispersion(K, q):
    """Onsager's gamma_q >= 0, cosh gamma_q = cosh 2K cosh 2K* - cos q, without cancellation."""
    return 2.0 * np.arcsinh(np.sqrt(np.sinh(detuning(K)) ** 2 + np.sin(0.5 * np.asarray(q, dtype=float)) ** 2))


def _angles(n, l):
    """q_l = l pi / n with l and 2n - l sharing one floating-point |sin(q/2)|."""
    l = np.asarray(l)
    return np.minimum(l, 2 * n - l) * (np.pi / n)


def gammas(T, n):
    """The earlier script's array, Kaufman's layout: gamma_l at q = l pi / n for l = 0 .. 2n-1.

    Odd l belong to the charge-0 sector, even l to the charge-1 sector.  gamma_0 is SIGNED,
    2 (K - K*): positive below T_c, negative above.  Every other entry is positive.
    """
    K = coupling(T)
    out = dispersion(K, _angles(n, np.arange(2 * n)))
    out[0] = 2.0 * detuning(K)
    return out


class _Sector(NamedTuple):
    charge: int
    energy: list          # gamma >= 0 of each mode, ascending
    twice_spin: list      # the mode's l, q = l pi / n, in the same order
    parity: int           # the number of fermions is this modulo 2
    vacuum: float         # E_vac(charge) - E_vac(0)
    signed_zero: bool     # Kaufman's gamma_0 is negative here (charge 1 above T_c)


def _sectors(n, K):
    l = np.arange(2 * n)
    gamma = dispersion(K, _angles(n, l))
    above = detuning(K) <= 0.0
    built = []
    for charge in (0, 1):
        modes = l[1 - charge::2]
        order = np.argsort(gamma[modes], kind="stable")
        built.append((charge, gamma[modes][order], modes[order]))
    # E_vac(1) - E_vac(0) = 1/2 (sum gamma_NS - sum gamma_R), summed pairwise in sorted order
    gap = 0.5 * math.fsum(built[0][1] - built[1][1])
    return [_Sector(charge, energy.tolist(), spin.tolist(), parity, vacuum, zero)
            for (charge, energy, spin), parity, vacuum, zero in zip(built, (0, 1 if above else 0), (0.0, gap), (False, above))]


def _subsets(energy, twice_spin, max_size=None) -> Iterator[tuple]:
    """Every subset of the modes in nondecreasing order of its total energy.

    Yields (energy, size, first, twice_spin): `first` is the lowest mode in the subset (-1 for
    the empty one).  A subset is reached exactly once, from the subset that lacks its largest
    mode (extend) or holds the previous mode in its place (advance); both moves raise the
    energy because the modes are sorted, which is what makes the heap order the output.
    """
    yield 0.0, 0, -1, 0
    count = len(energy)
    if count == 0 or max_size == 0:
        return
    heap = [(energy[0], 0, 1, 0, twice_spin[0])]
    while heap:
        total, last, size, first, spin = heapq.heappop(heap)
        yield total, size, first, spin
        after = last + 1
        if after == count:
            continue
        if max_size is None or size < max_size:
            heapq.heappush(heap, (total + energy[after], after, size + 1, first, spin + twice_spin[after]))
        heapq.heappush(heap, (total + (energy[after] - energy[last]), after, size, after if size == 1 else first,
                              spin + twice_spin[after] - twice_spin[last]))


def _fold(spin, n):
    """A momentum label as the integer in [-n/2, n/2)."""
    return (int(spin) + n // 2) % n - n // 2


def _stream(n, sector: _Sector, max_flips=None) -> Iterator[tuple]:
    """The sector's levels in nondecreasing order: (excitation, charge, spin, fermions, flips).

    `flips` is the number of minus signs the level has in Kaufman's convention, where the zero
    mode of the charge-1 sector above T_c is counted the other way round.
    """
    widest = None
    if max_flips is not None:
        widest = max_flips + 1 if sector.signed_zero else max_flips
    for total, size, first, twice_spin in _subsets(sector.energy, sector.twice_spin, widest):
        if size % 2 != sector.parity:
            continue
        flips = size if not sector.signed_zero else (size - 1 if first == 0 else size + 1)
        if max_flips is not None and flips > max_flips:
            continue
        yield sector.vacuum + total, sector.charge, _fold(twice_spin // 2, n), size, flips


class Levels(NamedTuple):
    """Levels in ascending order of E; every field is an array of the same length."""

    E: np.ndarray          # -ln(lambda)
    X: np.ndarray          # (E - E_0) n / (2 pi), E_0 the lowest level of the whole spectrum
    charge: np.ndarray     # 0 spin-flip even, 1 spin-flip odd
    spin: np.ndarray       # momentum in units of 2 pi / n, an integer in [-n/2, n/2)
    fermions: np.ndarray   # occupied modes, every gamma counted positive


def ground_energy(n, T):
    """E_0 = -ln(lambda_0): the charge-0 vacuum, the largest eigenvalue of the transfer matrix."""
    K = coupling(T)
    modes = dispersion(K, _angles(n, np.arange(1, 2 * n, 2)))
    return -0.5 * n * math.log(2.0 * math.sinh(2.0 * K)) - 0.5 * math.fsum(np.sort(modes))


@functools.lru_cache(maxsize=8)
def _lowest_per_momentum(n, K, keep):
    """For each charge: (excitation, fermions), arrays [momentum 0 .. n-1, keep], the `keep`
    lowest levels of every momentum in ascending order (inf where the sector has fewer).

    A dynamic programme over the modes.  The state is (number of fermions mod 2, twice the
    momentum mod 2n); a mode of angle l pi / n taken into a state flips the first, shifts the
    second by l and adds gamma.  After each mode the `keep` lowest energies of a state are the
    `keep` lowest among its old list and the shifted list of the other parity, so nothing below
    the cut is ever lost.  n modes, each a sort of 2 x 2n lists of 2 keep numbers.
    """
    tables = []
    for sector in _sectors(n, K):
        energy = np.full((2, 2 * n, keep), np.inf)
        fermions = np.zeros((2, 2 * n, keep), dtype=int)
        energy[0, 0, 0] = 0.0
        for gamma, l in zip(sector.energy, sector.twice_spin):
            both = np.concatenate((energy, np.roll(energy[::-1], l, axis=1) + gamma), axis=2)
            count = np.concatenate((fermions, np.roll(fermions[::-1], l, axis=1) + 1), axis=2)
            lowest = np.argsort(both, axis=2, kind="stable")[:, :, :keep]
            energy = np.take_along_axis(both, lowest, axis=2)
            fermions = np.take_along_axis(count, lowest, axis=2)
        tables.append((sector.vacuum + energy[sector.parity, ::2], fermions[sector.parity, ::2]))
    return tables


def _as_levels(n, T, rows) -> Levels:
    rows = sorted(rows)
    excitation = np.array([row[0] for row in rows], dtype=float)
    return Levels(E=ground_energy(n, T) + excitation, X=excitation * n / (2 * np.pi),
                  charge=np.array([row[1] for row in rows], dtype=int), spin=np.array([row[2] for row in rows], dtype=int),
                  fermions=np.array([row[3] for row in rows], dtype=int))


def sector_tables(n, T, num_levels) -> dict:
    """{(charge, spin): Levels}: the lowest `num_levels` levels of EVERY sector of the width-n
    strip (fewer where a sector is smaller), spin running over the n integers in [-n/2, n/2).

    One pass costs about n^2 num_levels log(num_levels) operations whatever the momentum, so
    this is the route to sectors far from spin 0.
    """
    n, keep = int(n), int(num_levels)
    if n < 1 or keep < 1:
        raise ValueError(f"n and num_levels must be positive, got {n} and {keep}")
    out = {}
    for charge, (excitation, fermions) in enumerate(_lowest_per_momentum(n, float(coupling(T)), keep)):
        for momentum in range(n):
            spin = _fold(momentum, n)
            present = np.isfinite(excitation[momentum])
            out[(charge, spin)] = _as_levels(n, T, [(float(e), charge, spin, int(f))
                                                    for e, f in zip(excitation[momentum][present], fermions[momentum][present])])
    return out


def levels(n, T, num_levels=None, charge=None, spin=None) -> Levels:
    """The lowest `num_levels` levels of the width-n strip at temperature T, labelled.

    charge = 0 or 1 and spin = an integer (taken modulo n) restrict the list to one sector;
    `num_levels` then counts levels of that sector.  num_levels=None returns every level,
    2^n of them without a restriction, and is refused above n = 20.  For many sectors at
    once call `sector_tables`.

    X is always measured from the global ground level, also inside a restricted sector.
    The sign of `spin` is a convention (the spectrum is symmetric under spin -> -spin).
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"the strip needs at least one site, got n = {n}")
    if charge not in (None, 0, 1):
        raise ValueError(f"charge is 0, 1 or None, got {charge!r}")
    if num_levels is None and n > 20:
        raise ValueError(f"num_levels=None would list 2^{n} levels; give a number")
    if spin is not None and num_levels is not None:
        # one momentum: the dynamic programme, whose cost does not depend on the momentum
        tables = sector_tables(n, T, num_levels)
        wanted = _fold(spin, n)
        rows = [(e, c, s, f) for c in (0, 1) if charge in (None, c)
                for e, s, f in zip(tables[(c, wanted)].E, tables[(c, wanted)].spin, tables[(c, wanted)].fermions)]
        ground = ground_energy(n, T)
        return _as_levels(n, T, sorted((e - ground, c, s, f) for e, c, s, f in rows)[:int(num_levels)])
    sectors = [s for s in _sectors(n, coupling(T)) if charge in (None, s.charge)]
    merged = heapq.merge(*(_stream(n, s) for s in sectors))
    if spin is not None:
        wanted = _fold(spin, n)
        merged = (row for row in merged if row[2] == wanted)
    return _as_levels(n, T, [row[:4] for row in itertools.islice(merged, num_levels)])


def Ei_kmax(n, T, num_collect, kmax=None):
    """The lowest `num_collect` values of -ln(lambda), ascending: the earlier script's call.

    kmax is kept for compatibility.  It caps the number of minus signs a level may have in
    Kaufman's convention (the enumeration needed it to stay affordable); kmax=None is the true
    spectrum, and here it is the cheap case.
    """
    n = int(n)
    streams = (_stream(n, s, None if kmax is None else int(kmax)) for s in _sectors(n, coupling(T)))
    rows = itertools.islice(heapq.merge(*streams), int(num_collect))
    return ground_energy(n, T) + np.array(sorted(row[0] for row in rows), dtype=float)


if __name__ == "__main__":
    # X_1 at T_c for n = 2 .. 15, then the labelled levels of the width-16 strip.
    widths = np.arange(2, 16)
    energies = np.array([Ei_kmax(n, Tc, 3) for n in widths]).T
    print((energies[1] - energies[0]) * widths / (2 * np.pi))
    table = levels(16, Tc, 12)
    print("\nn = 16 at T_c:   X          charge  spin  fermions")
    for row in zip(table.X, table.charge, table.spin, table.fermions):
        print("              %12.9f   %d     %+d     %d" % row)
