"""Consume the frozen decimal reference directly at each calculation's precision."""

from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys

from mpmath import mp
import pytest

from exact_solutions import ising_exact as ie


REFERENCE = json.loads((Path(__file__).parent / "data" / "ising_reference_high_precision.json").read_text())


def test_all_published_digits_reproduce(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_ising_reference.py"
    subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=tmp_path)


@pytest.mark.parametrize("dps", [None, 25, 50, 100])
def test_spectrum_against_stored_high_precision_reference(dps):
    oracle = mp.clone()
    oracle.dps = 160
    working = mp.clone()
    working.dps = dps or 15
    # Mode-sum subtraction and multiplication by the width consume some digits.
    tolerance = oracle.mpf("1e-11" if dps is None else f"1e{-dps + 5}")
    assert REFERENCE["significant_digits"] == 100
    assert [(case["sites"], case["temperature_over_tc"]) for case in REFERENCE["cases"]] == [
        (2 ** k, r) for k in range(1, 9) for r in ("1/1", "6/5")]
    for case in REFERENCE["cases"]:
        n = case["sites"]
        numerator, denominator = map(int, case["temperature_over_tc"].split("/"))
        r = numerator / denominator if dps is None else working.mpf(numerator) / denominator
        expected = case["X"]
        assert len(expected) == min(24, 2 ** n)
        assert all(isinstance(x, str) for x in expected)
        assert all(len(Decimal(x).as_tuple().digits) == 100 for x in expected[1:])
        actual = ie.levels_reduced(n, r, len(expected), dps=dps).X
        assert actual[0] == oracle.mpf(expected[0]) == 0
        for index, (value, text) in enumerate(zip(actual, expected)):
            target = oracle.mpf(text)
            error = abs(oracle.mpf(value) - target)
            assert error <= tolerance * max(1, abs(target)), (n, case["temperature_over_tc"], index)
