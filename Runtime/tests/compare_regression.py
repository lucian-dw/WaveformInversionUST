"""Compare capture_regression MAT outputs, excluding measured execution times.

Requires scipy. Usage: python Runtime/tests/compare_regression.py before.mat after.mat
"""

import argparse
import json

import numpy as np
from scipy.io import loadmat

TIMINGS = {
    "wall_seconds",
    "assembly_factor_seconds",
    "forward_seconds",
    "adjoint_seconds",
    "linearized_seconds",
}


def compare(before, after, label, errors):
    if isinstance(before, dict):
        assert before.keys() == after.keys(), f"{label}: record structure differs"
        for key in before:
            if key not in TIMINGS and not key.startswith("__"):
                compare(before[key], after[key], f"{label}.{key}", errors)
    elif isinstance(before, (list, tuple)):
        assert len(before) == len(after), label
        for i, (left, right) in enumerate(zip(before, after)):
            compare(left, right, f"{label}[{i}]", errors)
    elif isinstance(before, str):
        assert before == after, label
    else:
        left, right = np.asarray(before), np.asarray(after)
        assert left.shape == right.shape, label
        assert np.isfinite(left).all() and np.isfinite(right).all(), label
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=0, err_msg=label)
        errors[label] = float(
            np.max(np.abs(left.astype(complex) - right.astype(complex)), initial=0)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    errors = {}
    compare(
        loadmat(args.before, simplify_cells=True),
        loadmat(args.after, simplify_cells=True),
        "capture",
        errors,
    )
    print(
        json.dumps(
            {
                "numeric_fields_compared": len(errors),
                "max_absolute_difference": max(errors.values()),
                "rtol": 1e-12,
                "atol": 0,
                "excluded_fields": sorted(TIMINGS),
                "differences": errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
