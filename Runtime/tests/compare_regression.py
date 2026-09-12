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
    parser.add_argument("--protocol-migration", action="store_true")
    args = parser.parse_args()
    errors = {}
    before = loadmat(args.before, simplify_cells=True)
    after = loadmat(args.after, simplify_cells=True)
    if args.protocol_migration:
        assert before["result"]["schema"] == "wfi.reconstruction.v1"
        assert (
            after["result"]["schema"] == "wust.reconstruction"
            and after["result"]["schema_version"] == 1
        )
        assert (
            after["prepared"]["schema"] == "wust.measurements"
            and after["prepared"]["schema_version"] == 1
        )
        for key in ("loss", "initial_gradient"):
            compare(before[key], after[key], key, errors)
        for key in ("c_mps", "history_mps"):
            compare(
                before["result"][key], after["result"][key], "result." + key, errors
            )
        old = before["result"]["records"]
        new = after["result"]["records"]
        assert len(old) == len(new)
        for i, (a, b) in enumerate(zip(old, new)):
            assert b.keys() - a.keys() == {
                "relative_slowness_update",
                "stage_id",
                "loss_model_step",
            }
            compare(a, {k: b[k] for k in a}, f"records[{i}]", errors)
        for key in (
            "Y",
            "mask",
            "frequencies_hz",
            "tx_index",
            "rx_index",
            "tx_xy_m",
            "rx_xy_m",
            "tx_snapped_xy_m",
            "rx_snapped_xy_m",
            "x_m",
            "y_m",
            "saved_dt_s",
            "fourier_sign",
        ):
            if key in ("x_m", "y_m", "tx_snapped_xy_m", "rx_snapped_xy_m"):
                # Reconstructing nodes from edge origin + spacing introduces only
                # floating-point coordinate roundoff; do not relax solver tolerances.
                a = before["prepared"][key]
                b = after["prepared"][key]
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-15, err_msg=key)
                errors["prepared." + key] = float(np.max(np.abs(a - b)))
            else:
                compare(
                    before["prepared"][key],
                    after["prepared"][key],
                    "prepared." + key,
                    errors,
                )
        cfg = before["result"]["config"]
        compare(cfg, {k: after["result"]["config"][k] for k in cfg}, "config", errors)
        prep = before["prepared"]["preparation"]
        compare(
            prep,
            {k: after["prepared"]["preparation"]["rf_config"][k] for k in prep},
            "rf_config",
            errors,
        )
    else:
        compare(before, after, "capture", errors)
    print(
        json.dumps(
            {
                "numeric_fields_compared": len(errors),
                "max_absolute_difference": max(errors.values()),
                "rtol": 1e-12,
                "atol": 0,
                "coordinate_atol_m": 1e-15 if args.protocol_migration else 0,
                "excluded_fields": sorted(TIMINGS),
                "differences": errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
