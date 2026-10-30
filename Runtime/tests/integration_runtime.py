"""Manual MATLAB integration and the shared coherent cross-repository fixture.

Fixture generation is separate from reconstruction and never a production path.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Runtime/python"))
from process import execute
from wust_io import read_artifact, write_artifact
from wust_runtime import quote, run


def create_fixture(folder, matlab):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    grid = {
        "shape_yx": [23, 27],
        "origin_yx_m": [-0.0115, -0.02],
        "spacing_yx_m": [0.001, 0.0015],
        "origin_kind": "pixel_edge",
    }
    y = grid["origin_yx_m"][0] + (np.arange(23) + 0.5) * 0.001
    x = grid["origin_yx_m"][1] + (np.arange(27) + 0.5) * 0.0015
    tx = np.array([[x[7], y[6]], [x[19], y[16]]])
    rx = np.array([[x[5], y[14]], [x[18], y[5]], [x[20], y[13]], [x[10], y[17]]])
    arrays = {
        "pressure": np.ones((2, 2, 4), complex),
        "mask": np.ones((2, 2, 4), bool),
        "frequencies_hz": np.array([1.2e5, 1e5]),
        "tx_xy_m": tx,
        "rx_xy_m": rx,
    }
    arrays["mask"][0, 0, 2] = False
    metadata = {
        "grid": grid,
        "fourier_sign": -1,
        "real_pressure": True,
        "pressure_type": "total_pressure",
        "data_units": "unknown",
        "spectrum_normalization": "unknown",
        "measurement_provenance": "coherent_Helmholtz_fixture",
    }
    write_artifact(
        folder / "input.json",
        "wust.frequency_input",
        metadata,
        arrays,
        {
            "pressure": "frequency,tx,rx",
            "mask": "frequency,tx,rx",
            "frequencies_hz": "frequency",
            "tx_xy_m": "tx,xy",
            "rx_xy_m": "rx,xy",
        },
        {
            "pressure": "unknown",
            "mask": "1",
            "frequencies_hz": "Hz",
            "tx_xy_m": "m",
            "rx_xy_m": "m",
        },
    )
    output = folder / "coherent.json"
    expression = (
        f"s=settings; addpath({quote(ROOT/'Runtime/tests')}); "
        f"addpath({quote(ROOT/'Runtime/matlab')}); "
        f"input=wust_read_artifact({quote(folder/'input.json')},'wust.frequency_input'); "
        f"wust_test_measurements(input,'cpu',[],{quote(output)});"
    )
    execute([matlab, "-batch", expression], 180)
    initial = np.full((23, 27), 1500.0)
    update = np.zeros_like(initial, dtype=bool)
    update[5:-5, 5:-5] = True
    write_artifact(
        folder / "initial.json",
        "wust.initial_model",
        {},
        {"initial_mps": initial, "update_mask": update},
        {"initial_mps": "y,x", "update_mask": "y,x"},
        {"initial_mps": "m/s", "update_mask": "1"},
    )
    return output


def numerical_config(backend="cpu"):
    return {
        "backend": backend,
        "bounds_mps": [1300, 1800],
        "max_update_mps": 12,
        "step_damping": 0.25,
        "source_batch_size": 2,
        "pml_strength": 10,
        "pml_m": 0.004,
        "stencil_bounds": [1400, 1700],
        "wavenumber": "continuum",
        "filter_cutoff": 0,
        "filter_order": 4,
    }


def check_rf_preparation(folder, matlab):
    """Exercise public prepare dispatch against an independent dt-weighted DFT."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    time_s = 3e-6 + np.arange(128) * 1e-7
    frequencies = np.array([2e5, 1e5])
    tx = np.array([[-0.003, -0.002], [0.003, 0.002]])
    rx = np.array([[-0.002, 0.003], [0.002, -0.003], [0.004, 0.0]])
    pressure = np.cos(2 * np.pi * 1.5e5 * time_s)[:, None, None] * (
        1 + np.arange(6).reshape(1, 3, 2)
    )
    pressure[:, 0, 0] = 0  # A valid zero observation stays valid.
    pressure[:, 1, 1] = np.nan  # A masked channel must never enter the DFT.
    mask = np.ones((2, 3), bool)
    mask[1, 1] = False
    grid = {
        "shape_yx": [23, 27],
        "origin_yx_m": [-0.0115, -0.02025],
        "spacing_yx_m": [0.001, 0.0015],
        "origin_kind": "pixel_edge",
    }
    write_artifact(
        folder / "rf.json",
        "wust.rf",
        {"data_units": "Pa", "measurement_provenance": "analytic_RF_fixture"},
        {"pressure": pressure, "time_s": time_s, "tx_xy_m": tx, "rx_xy_m": rx},
        {
            "pressure": "time,rx,tx",
            "time_s": "time",
            "tx_xy_m": "tx,xy",
            "rx_xy_m": "rx,xy",
        },
        {"pressure": "Pa", "time_s": "s", "tx_xy_m": "m", "rx_xy_m": "m"},
    )
    request = {
        "schema": "wust.request",
        "schema_version": 2,
        "operation": "prepare",
        "input_manifest": str(folder / "rf.json"),
        "output_manifest": str(folder / "prepared.json"),
        "config": {
            "grid": grid,
            "frequencies_hz": frequencies.tolist(),
            "c_geom_mps": 1500,
            "phase_correction": "none",
            "mask": mask.tolist(),
        },
    }
    path = folder / "request.json"
    path.write_text(json.dumps(request))
    run(path, matlab, timeout_s=180)
    doc, arrays = read_artifact(folder / "prepared.json", "wust.measurements")
    kernel = np.exp(-2j * np.pi * np.sort(frequencies)[:, None] * time_s) * 1e-7
    expected = np.einsum("ft,trq->qrf", kernel, np.nan_to_num(pressure))
    np.testing.assert_allclose(arrays["Y"], expected, rtol=2e-6, atol=1e-13)
    np.testing.assert_array_equal(arrays["mask"], np.repeat(mask[..., None], 2, axis=2))
    assert doc["metadata"]["grid"] == grid
    assert doc["metadata"]["data_units"] == "Pa*s"
    return float(np.linalg.norm(arrays["Y"] - expected) / np.linalg.norm(expected))


def check_reconstruction(metadata, speed, *, backend, expected_updates=2):
    assert speed.shape == (23, 27) and speed.dtype == np.float64
    assert np.isfinite(speed).all() and (speed > 0).all()
    assert metadata["environment"]["backend"] == backend
    assert metadata["completed_updates"] == expected_updates
    assert metadata["planned_schedule_length"] == 3
    assert metadata["completion"]["converged"] is False
    assert metadata["final_data_residual"] is None
    assert metadata["completion"]["reason"] == "caller_truncated_schedule"
    records = metadata["records"]
    assert len(records) == expected_updates
    assert [r["frequency_hz"] for r in records] == [1e5, 1e5]
    assert [r["loss_model_step"] for r in records] == [0, 1]
    assert records[1]["loss_before_update"] < records[0]["loss_before_update"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--fixture", type=Path, help="Reuse a prepared coherent fixture directory"
    )
    args = parser.parse_args()
    if args.prepare_only:
        error = check_rf_preparation(args.out, args.matlab)
        print(json.dumps({"passed": True, "rf_prepare_relative_error": error}))
        return
    folder = args.out.resolve()
    prepared = time.monotonic()
    if args.fixture is None:
        fixture = create_fixture(folder, args.matlab)
    else:
        folder.mkdir(parents=True, exist_ok=False)
        fixture = args.fixture.resolve() / "coherent.json"
    preparation_seconds = time.monotonic() - prepared
    request = {
        "schema": "wust.request",
        "schema_version": 2,
        "operation": "reconstruct",
        "input_manifest": str(fixture),
        "initial_manifest": str(fixture.parent / "initial.json"),
        "output_manifest": str(folder / "result.json"),
        "config": numerical_config(args.backend),
        "frequency_schedule_hz": [1e5, 1e5, 1.2e5],
        "max_updates": 2,
    }
    path = folder / "request.json"
    path.write_text(json.dumps(request))
    started = time.monotonic()
    run(path, args.matlab, timeout_s=180)
    reconstruction_seconds = time.monotonic() - started
    doc, arrays = read_artifact(folder / "result.json", "wust.reconstruction")
    check_reconstruction(doc["metadata"], arrays["c_mps"], backend=args.backend)
    assert "history_mps" not in arrays
    print(
        json.dumps(
            {
                "passed": True,
                "fixture_preparation_seconds": preparation_seconds,
                "reconstruction_seconds": reconstruction_seconds,
                "fixture_matlab_invocations": int(args.fixture is None),
                "reconstruction_matlab_invocations": 1,
                "backend": args.backend,
            }
        )
    )


if __name__ == "__main__":
    main()
