"""Regenerate the canonical decimal reference after checking precision convergence."""

import argparse
import json
from pathlib import Path
import sys

from mpmath import mp

# Support a bare checkout, including invocation from outside the repository.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exact_solutions import ising_exact as ie


OUTPUT = ROOT / "tests" / "data" / "ising_reference_high_precision.json"
DIGITS = 100
WORKING_DPS = 140
VERIFICATION_DPS = 180


def calculate_cases(dps):
    ctx = mp.clone()
    ctx.dps = dps
    cases = []
    for k in range(1, 9):
        n = 2 ** k
        for numerator, denominator in ((1, 1), (6, 5)):
            r = ctx.mpf(numerator) / denominator
            table = ie.levels_reduced(n, r, min(24, 2 ** n), dps=dps)
            cases.append({
                "sites": n,
                "temperature_over_tc": f"{numerator}/{denominator}",
                "X": [ctx.nstr(x, DIGITS, strip_zeros=False) for x in table.X],
            })
    return cases


def reference():
    cases = calculate_cases(WORKING_DPS)
    verified = calculate_cases(VERIFICATION_DPS)
    for lower, higher in zip(cases, verified):
        if lower != higher:
            raise RuntimeError(
                f"Published digits do not converge at sites={lower['sites']}, "
                f"T/Tc={lower['temperature_over_tc']}; no file written")
    return {
        "format_version": 1,
        "model": "isotropic square-lattice Ising model on a periodic strip",
        "observable": "X = (E - E0) * sites / (2*pi), E = -log(lambda)",
        "ordering": "nondecreasing X, including multiplicities",
        "significant_digits": DIGITS,
        "generation": {
            "script": "scripts/generate_ising_reference.py",
            "api": "exact_solutions.ising_exact.levels_reduced",
            "working_dps": WORKING_DPS,
            "verification_dps": VERIFICATION_DPS,
            "validation": "All published decimal strings agree at both working precisions.",
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the committed file without rewriting it")
    args = parser.parse_args()
    data = reference()
    if args.check:
        if json.loads(OUTPUT.read_text()) != data:
            parser.exit(1, f"Reference differs from {OUTPUT}\n")
        print(f"Verified {OUTPUT}")
    else:
        OUTPUT.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
