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

NumPy and the standard library by default. Optional dps= uses mpmath throughout;
the canonical *_reduced functions take r = T/Tc, with r = 1 exactly critical.
The original absolute-temperature functions convert at the requested precision.
See docs/precision.md for inputs and output precision.
"""

from __future__ import annotations

import functools
import heapq
import itertools
from typing import Iterator, NamedTuple

import numpy as np

if __package__:
    from ._arithmetic import Arithmetic
else:  # Also support running this file directly, as in the original module.
    from _arithmetic import Arithmetic

J = 1
kB = 1
__all__ = ["critical_temperature", "coupling", "dual_coupling", "detuning", "dispersion",
           "gammas", "Ei_kmax", "Levels", "levels", "sector_tables", "ground_energy",
           "coupling_reduced", "detuning_reduced", "dispersion_reduced", "gammas_reduced",
           "levels_reduced", "sector_tables_reduced", "ground_energy_reduced"]


def _critical_temperature(ar):
    return 2 / ar.log(1 + ar.sqrt(ar.number(2)))


def critical_temperature(*, dps=None):
    """2 / log(1 + sqrt(2)), evaluated at the requested decimal working precision.

    This is a rounded numeric temperature. Use the *_reduced functions with r=1
    to request exact criticality without converting through an absolute temperature.
    """
    ar = Arithmetic(dps)
    return _critical_temperature(ar)


def _positive(values, name, ar):
    values = ar.values(values)
    if not np.all(ar.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return values


def _coupling(T, ar):
    T = _positive(T, "T", ar)
    return J / (kB * T)


def coupling(T, *, dps=None):
    """K = J / (kB T)."""
    return _coupling(T, Arithmetic(dps))


def _dual_coupling(K, ar):
    # Below K=1 the direct tanh form has fewer rounded intermediates, which
    # matters when K-K* cancels near Kc. At larger K, retain exp(-2K) explicitly.
    if np.ndim(K) == 0:
        if K < 1:
            return -ar.log(ar.tanh(K)) / 2
        small = ar.exp(-2 * K)
        return (ar.log1p(small) - ar.log1p(-small)) / 2
    low = K < 1
    out = np.empty(K.shape, dtype=ar.dtype)
    out[low] = -ar.log(ar.tanh(K[low])) / 2
    small = ar.exp(-2 * K[~low])
    out[~low] = (ar.log1p(small) - ar.log1p(-small)) / 2
    return out


def dual_coupling(K, *, dps=None):
    """K* with tanh K* = exp(-2K), equivalently sinh 2K sinh 2K* = 1 (Kaufman's H*)."""
    ar = Arithmetic(dps)
    return _dual_coupling(ar.values(K), ar)


def detuning(K, *, dps=None):
    """K - K*: zero at T_c, positive below it, negative above."""
    ar = Arithmetic(dps)
    K = ar.values(K)
    return K - _dual_coupling(K, ar)


def _dispersion_delta(delta, q, ar):
    delta, sine = np.broadcast_arrays(abs(delta), ar.sin(ar.array(q) / 2))
    out = np.empty(delta.shape, dtype=ar.dtype)
    near = delta < 1
    out[near] = 2 * ar.asinh(ar.sqrt(ar.sinh(delta[near]) ** 2 + sine[near] ** 2))
    # Factor exp(delta) out of asinh(sqrt(sinh(delta)^2 + sin(q/2)^2)).
    # All exponentials now have nonpositive arguments; no sinh(delta)^2 overflow.
    small = ar.exp(-2 * delta[~near])
    scaled = ar.sqrt((1 - small) ** 2 + 4 * small * sine[~near] ** 2)
    out[~near] = 2 * delta[~near] + 2 * ar.log((scaled + ar.sqrt(scaled ** 2 + 4 * small)) / 2)
    return out.item() if out.ndim == 0 else out


def dispersion(K, q, *, dps=None):
    """Onsager's gamma_q >= 0, cosh gamma_q = cosh 2K cosh 2K* - cos q, without cancellation."""
    ar = Arithmetic(dps)
    K = ar.values(K)
    return _dispersion_delta(K - _dual_coupling(K, ar), q, ar)


class _ThermalState(NamedTuple):
    ratio: object          # T/Tc, retained for phase decisions and the cache key
    detuning: object       # K-K*, formed without cancellation near r=1
    log_weight: object     # log(2 sinh(2K)), exactly log(2) at r=1


def _thermal_state(r, ar):
    r = _positive(r, "T/Tc", ar)
    root = ar.sqrt(ar.number(2))
    Kc = ar.log1p(root) / 2
    K = np.asarray(Kc / r, dtype=ar.dtype)
    # This difference retains small offsets which Kc/r - Kc would cancel.
    shift = np.asarray(Kc * ((1 - r) / r), dtype=ar.dtype)
    near = np.asarray(abs(shift) < ar.number(0.25), dtype=bool)
    delta = np.empty(shift.shape, dtype=ar.dtype)
    log_weight = np.empty(shift.shape, dtype=ar.dtype)

    # With a=exp(2Kc)=1+sqrt(2) and x=expm1(2(K-Kc)), use a^2-2a-1=0
    # analytically. No residual from a rounded Kc survives when r is exactly 1.
    a = 1 + root
    x = ar.expm1(2 * shift[near])
    delta[near] = ar.log1p(a * x * (2 * (a - 1) + a * x) / (a * (1 + x) + 1)) / 2
    # sinh(2K) = cosh(2(K-Kc)) + sqrt(2) sinh(2(K-Kc)).
    log_weight[near] = ar.log(ar.number(2)) + ar.log1p(
        2 * ar.sinh(shift[near]) ** 2 + root * ar.sinh(2 * shift[near]))

    delta[~near] = K[~near] - _dual_coupling(K[~near], ar)
    log_weight[~near] = 2 * K[~near] + ar.log(-ar.expm1(-4 * K[~near]))
    if shift.ndim == 0:
        return _ThermalState(r, delta.item(), log_weight.item())
    return _ThermalState(r, delta, log_weight)


def _absolute_ratio(T, ar):
    # Convert once, in the same context as Tc and the subsequent calculation.
    return _positive(T, "T", ar) / _critical_temperature(ar)


def _spectrum_state(r, ar):
    if np.ndim(r) != 0:
        raise ValueError("a spectrum requires one scalar temperature")
    return _thermal_state(r, ar)


def coupling_reduced(r, *, dps=None):
    """K = Kc/r for r=T/Tc; scalar and array inputs are supported."""
    ar = Arithmetic(dps)
    return ar.log1p(ar.sqrt(ar.number(2))) / (2 * _positive(r, "T/Tc", ar))


def detuning_reduced(r, *, dps=None):
    """K-K* as a function of r=T/Tc, exactly zero at r=1; supports arrays."""
    return _thermal_state(r, Arithmetic(dps)).detuning


def dispersion_reduced(r, q, *, dps=None):
    """Nonnegative dispersion at r=T/Tc, with NumPy broadcasting over r and q."""
    ar = Arithmetic(dps)
    return _dispersion_delta(_thermal_state(r, ar).detuning, q, ar)


def _angles(n, l, ar):
    """q_l = l pi / n with l and 2n - l sharing one floating-point |sin(q/2)|."""
    l = np.asarray(l)
    return np.minimum(l, 2 * n - l) * (ar.pi / n)


def gammas(T, n, *, dps=None):
    """The earlier script's array, Kaufman's layout: gamma_l at q = l pi / n for l = 0 .. 2n-1.

    Odd l belong to the charge-0 sector, even l to the charge-1 sector.  gamma_0 is SIGNED,
    2 (K - K*): positive below T_c, negative above.  Every other entry is positive.
    """
    ar = Arithmetic(dps)
    return _gammas(n, _spectrum_state(_absolute_ratio(T, ar), ar), ar)


def gammas_reduced(r, n, *, dps=None):
    """Kaufman's signed dispersion array at r=T/Tc; gamma_0 is exactly zero at r=1."""
    ar = Arithmetic(dps)
    return _gammas(n, _spectrum_state(r, ar), ar)


def _gammas(n, state, ar):
    out = _dispersion_delta(state.detuning, _angles(n, np.arange(2 * n), ar), ar)
    out[0] = 2 * state.detuning
    return out


class _Sector(NamedTuple):
    charge: int
    energy: list          # gamma >= 0 of each mode, ascending
    twice_spin: list      # the mode's l, q = l pi / n, in the same order
    parity: int           # the number of fermions is this modulo 2
    vacuum: object        # E_vac(charge) - E_vac(0), float or mpf
    signed_zero: bool     # Kaufman's gamma_0 is negative here (charge 1 above T_c)


def _sectors(n, state, ar):
    l = np.arange(2 * n)
    gamma = _dispersion_delta(state.detuning, _angles(n, l, ar), ar)
    above = state.ratio >= 1
    built = []
    for charge in (0, 1):
        modes = l[1 - charge::2]
        order = np.argsort(gamma[modes], kind="stable")
        built.append((charge, gamma[modes][order], modes[order]))
    # E_vac(1) - E_vac(0) = 1/2 (sum gamma_NS - sum gamma_R), summed pairwise in sorted order
    gap = ar.fsum(built[0][1] - built[1][1]) / 2
    return [_Sector(charge, energy.tolist(), spin.tolist(), parity, vacuum, zero)
            for (charge, energy, spin), parity, vacuum, zero
            in zip(built, (0, 1 if above else 0), (ar.number(0), gap), (False, above))]


def _subsets(energy, twice_spin, max_size=None) -> Iterator[tuple]:
    """Every subset of the modes in nondecreasing order of its total energy.

    Yields (energy, size, first, twice_spin): `first` is the lowest mode in the subset (-1 for
    the empty one).  A subset is reached exactly once, from the subset that lacks its largest
    mode (extend) or holds the previous mode in its place (advance); both moves raise the
    energy because the modes are sorted, which is what makes the heap order the output.
    """
    yield 0, 0, -1, 0
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


def _ground_energy(n, state, ar):
    modes = _dispersion_delta(state.detuning, _angles(n, np.arange(1, 2 * n, 2), ar), ar)
    return -(n * state.log_weight + ar.fsum(np.sort(modes))) / 2


def ground_energy(n, T, *, dps=None):
    """E_0 = -ln(lambda_0): the charge-0 vacuum, the largest eigenvalue of the transfer matrix."""
    ar = Arithmetic(dps)
    return _ground_energy(n, _spectrum_state(_absolute_ratio(T, ar), ar), ar)


def ground_energy_reduced(n, r, *, dps=None):
    """Ground energy at r=T/Tc, using the critical normalization directly at r=1."""
    ar = Arithmetic(dps)
    return _ground_energy(n, _spectrum_state(r, ar), ar)


@functools.lru_cache(maxsize=8)
def _lowest_per_momentum(n, ratio, keep, dps):
    """For each charge: (excitation, fermions), arrays [momentum 0 .. n-1, keep], the `keep`
    lowest levels of every momentum in ascending order (inf where the sector has fewer).

    A dynamic programme over the modes.  The state is (number of fermions mod 2, twice the
    momentum mod 2n); a mode of angle l pi / n taken into a state flips the first, shifts the
    second by l and adds gamma.  After each mode the `keep` lowest energies of a state are the
    `keep` lowest among its old list and the shifted list of the other parity, so nothing below
    the cut is ever lost.  n modes, each a sort of 2 x 2n lists of 2 keep numbers.
    """
    ar = Arithmetic(dps)
    state = _spectrum_state(ratio, ar)
    tables = []
    for sector in _sectors(n, state, ar):
        energy = np.full((2, 2 * n, keep), ar.inf, dtype=ar.dtype)
        fermions = np.zeros((2, 2 * n, keep), dtype=int)
        energy[0, 0, 0] = ar.number(0)
        for gamma, l in zip(sector.energy, sector.twice_spin):
            both = np.concatenate((energy, np.roll(energy[::-1], l, axis=1) + gamma), axis=2)
            count = np.concatenate((fermions, np.roll(fermions[::-1], l, axis=1) + 1), axis=2)
            lowest = np.argsort(both, axis=2, kind="stable")[:, :, :keep]
            energy = np.take_along_axis(both, lowest, axis=2)
            fermions = np.take_along_axis(count, lowest, axis=2)
        tables.append((sector.vacuum + energy[sector.parity, ::2], fermions[sector.parity, ::2]))
    return tables


def _as_levels(n, state, rows, ar) -> Levels:
    rows = sorted(rows)
    excitation = ar.array([row[0] for row in rows])
    return Levels(E=_ground_energy(n, state, ar) + excitation, X=excitation * n / (2 * ar.pi),
                  charge=np.array([row[1] for row in rows], dtype=int), spin=np.array([row[2] for row in rows], dtype=int),
                  fermions=np.array([row[3] for row in rows], dtype=int))


def _sector_rows(n, state, keep, ar):
    out = {}
    # Keep the reduced coordinate and precision in the key: rounded K alone can
    # conflate exact criticality with an adjacent reduced temperature.
    for charge, (excitation, fermions) in enumerate(_lowest_per_momentum(n, state.ratio, keep, ar.dps)):
        for momentum in range(n):
            spin = _fold(momentum, n)
            present = ar.isfinite(excitation[momentum])
            out[(charge, spin)] = [(e, charge, spin, int(f))
                                  for e, f in zip(excitation[momentum][present], fermions[momentum][present])]
    return out


def sector_tables(n, T, num_levels, *, dps=None) -> dict:
    """{(charge, spin): Levels}: the lowest `num_levels` levels of EVERY sector of the width-n
    strip (fewer where a sector is smaller), spin running over the n integers in [-n/2, n/2).

    One pass costs about n^2 num_levels log(num_levels) operations whatever the momentum, so
    this is the route to sectors far from spin 0.
    """
    ar = Arithmetic(dps)
    return _sector_tables(n, _spectrum_state(_absolute_ratio(T, ar), ar), num_levels, ar)


def sector_tables_reduced(n, r, num_levels, *, dps=None) -> dict:
    """The lowest num_levels of every (charge, spin) sector at r=T/Tc."""
    ar = Arithmetic(dps)
    return _sector_tables(n, _spectrum_state(r, ar), num_levels, ar)


def _sector_tables(n, state, num_levels, ar):
    n, keep = int(n), int(num_levels)
    if n < 1 or keep < 1:
        raise ValueError(f"n and num_levels must be positive, got {n} and {keep}")
    return {key: _as_levels(n, state, rows, ar) for key, rows in _sector_rows(n, state, keep, ar).items()}


def levels(n, T, num_levels=None, charge=None, spin=None, *, dps=None) -> Levels:
    """Absolute-temperature adapter for levels_reduced: r=T/critical_temperature(dps).

    Convert T at the requested working precision. Already rounded inputs cannot
    recover lost digits; new critical-point calculations should use levels_reduced
    with r=1. All search and output conventions are those of levels_reduced.
    """
    ar = Arithmetic(dps)
    state = _spectrum_state(_absolute_ratio(T, ar), ar)
    return _levels(n, state, num_levels, charge, spin, ar)


def levels_reduced(n, r, num_levels=None, charge=None, spin=None, *, dps=None) -> Levels:
    """The lowest num_levels of the width-n strip at r=T/Tc, labelled.

    charge = 0 or 1 and spin = an integer (taken modulo n) restrict the list to one sector;
    `num_levels` then counts levels of that sector.  num_levels=None returns every level,
    2^n of them without a restriction, and is refused above n = 20.  For many sectors at
    once call `sector_tables_reduced`.

    X is always measured from the global ground level, also inside a restricted sector.
    The sign of `spin` is a convention (the spectrum is symmetric under spin -> -spin).
    dps selects decimal working precision via optional mpmath. E and X then contain
    mpf values in object arrays; labels remain integer arrays. Use strings or
    high-precision values for r. r=1 is exactly critical; nearby values are never
    snapped to it. The numerical constant Tc is not needed on this path.
    """
    ar = Arithmetic(dps)
    return _levels(n, _spectrum_state(r, ar), num_levels, charge, spin, ar)


def _levels(n, state, num_levels, charge, spin, ar):
    n = int(n)
    if n < 1:
        raise ValueError(f"the strip needs at least one site, got n = {n}")
    if charge not in (None, 0, 1):
        raise ValueError(f"charge is 0, 1 or None, got {charge!r}")
    if num_levels is None and n > 20:
        raise ValueError(f"num_levels=None would list 2^{n} levels; give a number")
    if spin is not None and num_levels is not None:
        # one momentum: the dynamic programme, whose cost does not depend on the momentum
        keep = int(num_levels)
        if keep < 1:
            raise ValueError("num_levels must be positive for a momentum-restricted search")
        tables = _sector_rows(n, state, keep, ar)
        wanted = _fold(spin, n)
        rows = [row for c in (0, 1) if charge in (None, c) for row in tables[(c, wanted)]]
        # Keep excitation coordinates: subtracting E_0 from total E loses small gaps.
        return _as_levels(n, state, sorted(rows)[:keep], ar)
    sectors = [s for s in _sectors(n, state, ar) if charge in (None, s.charge)]
    merged = heapq.merge(*(_stream(n, s) for s in sectors))
    if spin is not None:
        wanted = _fold(spin, n)
        merged = (row for row in merged if row[2] == wanted)
    return _as_levels(n, state, [row[:4] for row in itertools.islice(merged, num_levels)], ar)


def Ei_kmax(n, T, num_collect, kmax=None, *, dps=None):
    """The lowest `num_collect` values of -ln(lambda), ascending: the earlier script's call.

    kmax is kept for compatibility.  It caps the number of minus signs a level may have in
    Kaufman's convention (the enumeration needed it to stay affordable); kmax=None is the true
    spectrum, and here it is the cheap case.
    """
    n = int(n)
    ar = Arithmetic(dps)
    state = _spectrum_state(_absolute_ratio(T, ar), ar)
    streams = (_stream(n, s, None if kmax is None else int(kmax)) for s in _sectors(n, state, ar))
    rows = itertools.islice(heapq.merge(*streams), int(num_collect))
    return _ground_energy(n, state, ar) + ar.array(sorted(row[0] for row in rows))


if __name__ == "__main__":
    # X_1 at T_c for n = 2 .. 15, then the labelled levels of the width-16 strip.
    widths = np.arange(2, 16)
    energies = np.array([levels_reduced(n, 1, 3).E for n in widths]).T
    print((energies[1] - energies[0]) * widths / (2 * np.pi))
    table = levels_reduced(16, 1, 12)
    print("\nn = 16 at T_c:   X          charge  spin  fermions")
    for row in zip(table.X, table.charge, table.spin, table.fermions):
        print("              %12.9f   %d     %+d     %d" % row)
