# Exact_Solution

Exact results for lattice models, kept in one place to check numerical work against.

House rules for this repository:

- **numpy and the standard library only by default**, so that any project can import it
  without pulling in a tensor-network engine or a solver stack. Optional arbitrary-precision
  calculations use `mpmath` (`pip install ".[precision]"`). `scipy`, `pytest` and `mpmath`
  are used by the tests.
- **Each module states** the object it solves, its conventions, where the formula comes from,
  and how the code was checked against the **definition** — not only against another
  implementation.
- **Nothing here depends on another repository.**

| module | what it solves |
|---|---|
| `exact_solutions/ising_exact.py` | every eigenvalue of the 2D Ising row transfer matrix on a periodic strip of width n, labelled by charge and momentum |

```bash
pip install -e ".[dev]"      # or just: pip install -e .
pytest                       # includes independent high-precision Boltzmann-matrix checks
```

Another project can use it straight from GitHub:

```bash
pip install "exact-solutions @ git+https://github.com/pcchen/Exact_Solution"
```

---

## ising_exact

```python
from exact_solutions import ising_exact as ie

ie.levels_reduced(64, 1, 9)                    # exactly critical: E, X, charge, spin, fermions
ie.levels_reduced(64, 1, 5, charge=1, spin=2)  # the lowest 5 of one sector
ie.sector_tables_reduced(64, 1, 6)             # {(charge, spin): Levels} for all 2 n sectors
ie.levels_reduced(64, 1, 60).E                 # just the energies, ascending
ie.gammas_reduced(1, 64)                       # the 2 n dispersion values, Kaufman's layout
```

For reference data beyond double precision, the same functions accept `dps=`:

```python
table = ie.levels_reduced(64, "1.2", 9, dps=80) # 1.2 Tc, without a rounded 1.2 float
print(table.X[1])                             # an mpmath number, not a Python float
```

The canonical input is `r = T/Tc`. At `r=1` the implementation uses exact critical
identities; nearby ratios retain their detuning. Absolute-temperature calls such as
`ie.levels(n, T, ...)`, `sector_tables`, `gammas` and `Ei_kmax` remain adapters that
compute `T/critical_temperature(dps=...)` in the working context. The old double-valued
`ie.Tc` constant has been removed: use reduced temperatures for critical calculations,
or `ie.critical_temperature(dps=...)` when a numeric absolute temperature is needed.

The default remains double precision. With `dps=`, `E` and `X` are NumPy object arrays
of arbitrary-precision numbers, while the labels remain integer arrays. Each call uses
an isolated precision context. [Precision and reference generation](docs/precision.md)
explains input construction, exporting decimal strings, numerical limits and validation.

`E = -ln(lambda)`; `X = (E - E_0) n / (2 pi)`; `charge` 0 / 1 = even / odd under the spin flip;
`spin` = momentum in units of 2 pi / n, an integer in [-n/2, n/2); `T` in units J = kB = 1
(`K = 1 / T`), `Tc = 2 / ln(1 + sqrt 2)`.

**`docs/ising_exact.md`** (also `.tex` and `.pdf`) explains all of it at length: the formula,
the three conventions in use for saying which sign patterns are eigenvalues, how the momentum
label is computed, and worked examples.  What follows is the short version.

### The object

n spins in a periodic row, coupling K on every bond of the square lattice, and

    V[s, s'] = exp(K/2 sum_i s_i s_{i+1})  exp(K sum_i s_i s'_i)  exp(K/2 sum_i s'_i s'_{i+1}),

so that Z(n x M torus) = Tr V^M.  A ring of n vertex tensors, or a ring of blocks after an
untruncated coarse-graining, has the same spectrum.  Its eigenvalues are

    lambda = (2 sinh 2K)^(n/2) exp( 1/2 sum_q (+-) gamma_q ),
    cosh gamma_q = cosh 2K cosh 2K* - cos q,        tanh K* = exp(-2K),

one sign per angle: q = (2j+1) pi / n in the charge-0 sector, q = 2j pi / n in the charge-1
sector, j = 0 .. n-1.  (Onsager 1944; Kaufman 1949; Schultz, Mattis and Lieb 1964.)

### Three ways to count the same levels

Half of the 2^n sign patterns of each sector are eigenvalues.  Codes differ in how they say
which half, and the whole difference sits in ONE mode, q = 0 of the charge-1 sector, where
cosh gamma_0 = cosh 2(K - K*) fixes gamma_0 only up to its sign.

| | sign given to gamma_0 | charge 0 | charge 1 | here |
|---|---|---|---|---|
| Kaufman | 2 (K - K*): positive below T_c | even number of minus signs | EVEN number of minus signs | `gammas`, `Ei_kmax` |
| fermion vacuum | 2 (K* - K): positive above T_c | even | ODD number of minus signs | — |
| excitations | \|gamma_0\| = 2 \|K - K*\| | even number of fermions | odd above T_c, even below | `levels`, `sector_tables` |

They are the same set.  Reversing the sign given to gamma_0 turns its "+" into "-" in every
pattern, which changes the number of minus signs by one: Kaufman's "even" is the other
convention's "odd".  At T_c gamma_0 = 0 and nothing distinguishes them.

The third row is the physical reading and is what the labels use.  Every gamma_q >= 0 is the
energy of a fermion of momentum q, and

    E = E_vac(charge) + sum_{occupied q} gamma_q,     spin = sum_{occupied q} q / (2 pi / n)  mod n.

Above T_c the lowest charge-1 level is ONE fermion at q = 0 on top of that sector's vacuum,
and its distance from the ground level tends to gamma_0 = 2 (K* - K), the inverse correlation
length.  Below T_c the q = 0 fermion is absent from the lowest charge-1 level, which then lies
exponentially close to the ground level: the two ordered states.  That change of rule at T_c
is the phase transition written in the counting.

### What the module adds to the textbook formula

1. **The search.**  The lowest N levels are walked best-first with a heap — each subset of
   modes is reached once, from the subset lacking its largest mode or holding the previous
   mode in its place, and both moves raise the energy — so they cost O(N log N) at any width.
   Enumerating sign patterns instead costs 2^(n-1) per sector.
2. **The dispersion** as `gamma_q = 2 asinh sqrt( sinh^2(K - K*) + sin^2(q/2) )`, the same
   function (cosh 2K cosh 2K* = 1 + cosh 2(K - K*)) with no cancellation at small q or near
   T_c.  Measured against 50-digit arithmetic at T_c, the `arccosh` form loses 1.7e-13 at
   n = 127 and 2.8e-11 at n = 4096; this one stays at 3e-16.
3. **Charge, momentum and fermion number** on every level.  One sector's lowest levels come
   from a second algorithm, a dynamic programme over the modes (`sector_tables`), about
   n^2 N log N operations for all 2 n sectors at once: the heap walks the whole spectrum in
   order, and the number of levels below the first one of a high-momentum sector grows
   without bound.

Timings (Mac, 2026-11-13): `Ei_kmax(18, Tc, 18)` 0.16 ms, `Ei_kmax(64, Tc, 60)` 0.12 ms,
`levels(4096, Tc, 100000)` 0.38 s, `sector_tables(127, Tc, 60)` 0.11 s,
`sector_tables(1024, Tc, 30)` 3.8 s.

### Using it against a numerical transfer matrix

- **Ask by number of levels, per sector.**  A solver returns the lowest N eigenvalues of a
  sector in order and knows nothing of fermion number; `sector_tables(n, T, N)` and
  `levels(n, T, N, charge=, spin=)` return the same thing, complete.  Nothing in the module
  caps the particle number — a list cut that way has holes (at T_c a cap at two fermions
  already fails at level 12).
- **n is the full width in lattice sites**: for a ring of r blocks of L sites, n = rL.
- **Momentum is finer here.**  A ring of r blocks, translated block by block, resolves the
  spin only modulo r; group the exact levels by `spin mod r` before comparing, and treat
  +S and -S as a pair (the sign of S is a convention — see the note).
- **Do not compare one global list across a degenerate level.**  A Krylov solver started from
  a single vector holds one copy of a degenerate eigenvalue until rounding supplies the
  others, so a top-N list can contain two copies of a four-fold level and then the next level
  instead.  (Measured on one such solver at width 8, 0.8 T_c: a wrong top-8 list in 16 of 20
  random starts; asking for 16 and keeping 8 gave 0 of 20.)  Sector by sector the copies sit
  in different lists and the problem does not arise.

### How it was checked

The original 13 tests in `tests/test_ising_exact.py` cover:

- **Against the definition.**  V built from the Boltzmann weights above, projected onto every
  (charge, momentum) sector with the spin flip and the translation written as permutations of
  the 2^n configurations, n = 1 .. 8 at 0.7, 1 and 1.4 T_c: all 2^n levels, their values and
  both labels, to 1e-10, by both algorithms.  Shifting the charge-1 momenta by one, or
  exchanging the charges, fails the comparison (6 and 12 of 12 sectors at n = 6), so the check
  does see the labels.  It cannot see the SIGN of the spin: the spectrum is symmetric under
  spin -> -spin.
- **Against the recorded values** of the script this module grew out of (see Provenance):
  `gammas` to 1e-12, and `Ei_kmax` on 900 argument combinations to 1e-11.
- **The two algorithms against each other** at n = 24, where neither list is complete.
- **The CFT** at T_c, n = 64: (X, charge, spin) = (0, 0, 0), (1/8, 1, 0), (1, 0, 0),
  (1 + 1/8, 1, +-1), (2, 0, +-1 and +-2); and c = 1/2 from E_0 = n f - pi c / (6 n) with
  Onsager's f, which also fixes the absolute normalization of lambda.

`tests/test_precision.py` additionally checks the full spectrum and both labels against
120-digit Boltzmann matrices for n = 1 .. 4 at 0.7, 1 and 1.4 T_c; convergence at 30, 60
and 100 working digits; temperatures separated from T_c by 1e-45; ordered-phase gaps below
double precision; low-temperature overflow; and isolation of precision contexts and caches.
The default path is also exercised with the optional dependency deliberately unavailable.
`tests/test_reduced_temperature.py` covers exact critical identities, adjacent floating-point
ratios, relative detuning accuracy down to 1e-72 at 80 digits, and absolute/reduced adapter
consistency. `tests/test_array_helpers.py` covers NumPy sweeps and broadcasting, including
empty arrays and both stable-formula branches.

[`tests/data/`](tests/data/) also provides a canonical 100-digit reference for
the 328 HOTRG scaling dimensions. Its decimal strings are checked at 140 and
180 working digits before publication, and consumed by regression tests at
double precision and 25, 50 and 100 working digits. The historical fixture
remains separate. A regeneration script and input metadata accompany the data.

### Provenance

`gammas` and `Ei_kmax` keep the names, signature and Kaufman convention of an earlier script
of the author's (`Ising_Exact.py`, 2026-05-08, sha256
`02457a35e8b96486375f3a3e45f6c4a3f29a18e3f34ad41eaac4fa26056af6f5`), which is not part of this
repository.  That script was run once, on 2026-11-13, over the grid recorded in
`tests/data/original_reference.json`, and the tests check this module against those values.
