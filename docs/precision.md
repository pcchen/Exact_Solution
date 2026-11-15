# Precision and reference generation

The default API uses NumPy double precision and needs only NumPy and the standard
library. Install `pip install ".[precision]"` to enable mpmath arithmetic. The
`dev` extra includes mpmath so that the high-precision tests run with the rest of
the suite.

## Selecting working precision

All numerical entry points accept a keyword-only `dps=None`. Supply an integer
of at least 2 to select decimal working precision. The canonical temperature
coordinate is `r = T/Tc`, so `r=1` requests exact criticality at every precision.

```python
from exact_solutions import ising_exact as ie

table = ie.levels_reduced(64, "1.2", 12, dps=80)
sector = ie.levels_reduced(64, 1, 5, charge=1, spin=2, dps=80)
```

Use decimal strings or construct ratios in a high-precision context. The string
`"1.2"` avoids the binary approximation already present in the Python float `1.2`.

| Canonical reduced-temperature API | Absolute-temperature adapter |
|---|---|
| `levels_reduced(n, r, ...)` | `levels(n, T, ...)` |
| `sector_tables_reduced(n, r, ...)` | `sector_tables(n, T, ...)` |
| `gammas_reduced(r, n)` | `gammas(T, n)` |
| `ground_energy_reduced(n, r)` | `ground_energy(n, T)` |
| `coupling_reduced(r)` | `coupling(T)` |

`detuning_reduced(r)` and `dispersion_reduced(r, q)` also take reduced temperature;
their original counterparts `detuning(K)`, `dual_coupling(K)` and `dispersion(K, q)`
continue to take the coupling K. Numerical helpers support NumPy arrays and
broadcasting; a spectrum call takes one scalar temperature. The legacy
`Ei_kmax(n, T, ...)` retains its absolute-temperature and sign-pattern conventions.

The absolute-temperature adapters compute `r = T/critical_temperature(dps=...)`
in the same working context as the rest of the calculation. No proximity-based
snapping to criticality is used. Division can still round a ratio to one at the
working precision. A double-rounded absolute Tc promoted into an 80-digit call
remains slightly detuned, while `levels_reduced(n, 1, ...)` is exactly critical.
The old double-valued `ie.Tc` constant is removed. Use `critical_temperature(dps=...)`
when an absolute numeric temperature is needed, and the reduced API to specify
criticality. Preserve the original inputs when replaying historical fixtures.

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
programme results include both the reduced ratio and requested precision in their
key, rather than the rounded coupling alone.

## Exporting reference values

For the Uni20 HOTRG grid, use the committed
[`ising_reference_high_precision.json`](../tests/data/ising_reference_high_precision.json):
328 scaling dimensions at 100 significant digits, with exact rational reduced
temperatures. The [data documentation](../tests/data/) describes the format,
provenance and checked regeneration command. Consumers round this single file
to their own precision. The historical double-precision fixture stays separate.

Export decimal strings, not JSON numbers or Python floats. For example, this
generates the strip widths and temperatures used by the Uni20 HOTRG example:

```python
import json
from mpmath import mp
from exact_solutions import ising_exact as ie

formatting = mp.clone()
formatting.dps = 80
rows = []
for k in range(1, 9):
    n = 2 ** k
    for numerator, denominator in ((1, 1), (6, 5)):
        ratio = formatting.mpf(numerator) / denominator
        table = ie.levels_reduced(n, ratio, min(24, 2 ** n), dps=80)
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

The `dual_coupling(K)` helper uses `-log(tanh(K))/2` below K=1, avoiding extra rounded
intermediates near Kc, and logarithms of `1 +/- exp(-2K)` with `log1p` above it,
where tanh would round towards one. At large detuning the dispersion factors out
its exponential scale before evaluation. The ground-state normalization uses
`log(2 sinh(2K)) = 2K + log(-expm1(-4K))`. These representations also prevent
avoidable low-temperature overflow on the default double-precision path.

Near criticality, let `s = Kc (1-r)/r`, `a = 1+sqrt(2)` and `x = expm1(2s)`.
The implementation uses the critical identity `a^2 - 2a - 1 = 0` analytically:

```text
K - K* = log1p(a x (2(a-1) + a x) / (a(1+x) + 1)) / 2
log(2 sinh(2K)) = log(2) + log1p(2 sinh(s)^2 + sqrt(2) sinh(2s))
```

At `r=1`, `s=x=0`, so detuning is exactly zero and the normalization is `log(2)`.
The critical dispersion is `2 asinh(abs(sin(q/2)))`. Small offsets are carried
through `1-r`, rather than lost in `Kc/r - Kc` or `K - K*`. The two neighboring
double-precision ratios on either side of one remain distinct. A ratio already
rounded to one cannot encode a smaller detuning; use more working precision to
represent such inputs. At criticality, the charge-1 sector uses odd fermion count
with the zero-cost mode, consistently at every precision.

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
Both absolute and reduced-temperature APIs are checked against the independent
Boltzmann matrices. Reduced-coordinate tests verify exact criticality and relative
accuracy for tiny detunings; array tests cover parameter sweeps and broadcasting.
