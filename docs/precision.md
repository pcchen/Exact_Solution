# Precision and reference generation

The default API uses NumPy double precision and needs only NumPy and the standard
library. Install `pip install ".[precision]"` to enable mpmath arithmetic. The
`dev` extra includes mpmath so that the high-precision tests run with the rest of
the suite.

## Selecting working precision

All numerical entry points accept a keyword-only `dps=None`: `coupling`,
`dual_coupling`, `detuning`, `dispersion`, `gammas`, `ground_energy`, `levels`,
`sector_tables` and `Ei_kmax`. Supply an integer of at least 2 to select decimal
working precision. The new `critical_temperature(dps=...)` computes the critical
temperature at that precision; the existing constant `Tc` remains a double.

```python
from exact_solutions import ising_exact as ie

Tc = ie.critical_temperature(dps=80)
table = ie.levels(64, Tc * 6 / 5, 12, dps=80)
sector = ie.levels(64, Tc, 5, charge=1, spin=2, dps=80)
```

Use decimal strings for absolute temperatures, or construct them from
high-precision values and integer ratios. `Tc * 6 / 5` represents 1.2 times Tc
without passing through the binary approximation of the Python float `1.2`.
`ie.levels(n, ie.Tc, dps=80)` accurately solves the model at the *rounded double*
temperature supplied; it cannot recover the exact critical temperature. The
same limitation applies when promoting pre-existing double-precision inputs.

With `dps=`, energies, dispersions and scaling dimensions are mpmath real numbers
in NumPy object arrays. Charge, spin and fermion counts remain integer arrays.
The default path continues to return ordinary floating-point arrays. This is an
arbitrary-precision reference path, not NumPy `longdouble` arithmetic or a claim
about the platform's native fp80/fp128 support.

Each call uses its own mpmath context and leaves the process-wide `mp.dps`
unchanged. Returned numbers retain the call's context for Python arithmetic;
another calculation at a different precision does not change them. If applying
mpmath functions to those results, use a context with sufficient precision:
the global `mp.sqrt`, for example, uses the global context. Cached dynamic
programme results include the requested precision in their key.

## Exporting reference values

Export decimal strings, not JSON numbers or Python floats. For example, this
generates the strip widths and temperatures used by the Uni20 HOTRG example:

```python
import json
from mpmath import mp
from exact_solutions import ising_exact as ie

formatting = mp.clone()
formatting.dps = 80
Tc = ie.critical_temperature(dps=80)
rows = []
for k in range(1, 9):
    n = 2 ** k
    for numerator, denominator in ((1, 1), (6, 5)):
        table = ie.levels(n, Tc * numerator / denominator, min(24, 2 ** n), dps=80)
        rows.append({
            "sites": n,
            "temperature_over_tc": f"{numerator}/{denominator}",
            "working_dps": 80,
            "X": [formatting.nstr(x, 50) for x in table.X],
        })
print(json.dumps(rows, indent=2))
```

Before adopting the data, repeat at higher working precision and compare the
digits to be published. A consuming C++ test must parse these strings directly
into its target scalar at the desired precision, rather than using double
literals as an intermediate.

## Stable formulas and limits

The searches and the sector tables carry excitation energies above the global
ground state. Scaling dimensions are formed from those excitations directly,
including momentum-restricted queries; subtracting two extensive total energies
would lose significant digits. Total energies `E` are still provided, with the
rounding inherent in adding an excitation to the ground energy.

The dual coupling uses logarithms of `1 +/- exp(-2K)`, choosing `log1p` and
`expm1` forms to retain small terms. At large detuning the dispersion factors out
its exponential scale before evaluation. The ground-state normalization uses
`log(2 sinh(2K)) = 2K + log(-expm1(-4K))`. These representations also prevent
avoidable low-temperature overflow on the default double-precision path.

Working precision is not an error guarantee. In particular, the difference of
the two sector vacuum energies can be exponentially small in the ordered phase;
it is still obtained by subtracting mode sums. Resolving that splitting may
require substantially more working digits than the number of output digits.
An unresolved splitting may round to zero or acquire a small rounding error.
Increase `dps` and check convergence for the actual width and temperature. No
automatic precision increase or replacement of small gaps by zero is performed.

Both searches retain their original enumeration algorithms. Arbitrary precision
adds arithmetic cost, and object-array sorting in `sector_tables` is slower than
the NumPy double path. The full-spectrum and mode-storage limits still apply;
additional precision does not make exponentially many levels cheap to enumerate.

## Validation

The high-precision tests build the Boltzmann transfer matrix independently at
120 decimal digits, project it using explicit spin-flip and translation
operators, and diagonalize it. All levels and both labels are checked at widths
1 through 4 below, at and above Tc, with errors below 1e-70 for 80-digit runs.
Separate 30/60/100-digit runs verify that increasing precision gains accurate
digits. Further regressions cover critical detuning below double resolution,
tiny ordered-phase gaps, both spectrum algorithms, the legacy entry point,
low-temperature range, context/cache isolation and missing optional dependencies.
