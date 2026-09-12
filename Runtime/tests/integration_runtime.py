"""Actual MATLAB CPU gate for the JSON/HDF5 launcher; never run by default CI.

python Runtime/tests/integration_runtime.py --matlab /path/to/matlab --out /absolute/artifacts
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Runtime/python"))
from process import execute
from wust_io import read_artifact, read_json, sha256, write_artifact
from wust_runtime import RuntimeFailure, quote, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folder = args.out.resolve()
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
    frequencies = np.array([1.2e5, 1e5], dtype="float64")
    pressure = (
        np.arange(16).reshape(2, 2, 4) + 1 + 1j * np.arange(101, 117).reshape(2, 2, 4)
    ).astype("complex128")
    mask = np.ones((2, 2, 4), bool)
    mask[0, 0, 2] = False
    pressure[0, 0, 2] = np.nan
    arrays = {
        "pressure": pressure,
        "mask": mask,
        "frequencies_hz": frequencies,
        "tx_xy_m": tx,
        "rx_xy_m": rx,
    }
    axes = {
        "pressure": "frequency,tx,rx",
        "mask": "frequency,tx,rx",
        "frequencies_hz": "frequency",
        "tx_xy_m": "tx,xy",
        "rx_xy_m": "rx,xy",
    }
    units = {
        "pressure": "instrument_units",
        "mask": "1",
        "frequencies_hz": "Hz",
        "tx_xy_m": "m",
        "rx_xy_m": "m",
    }
    metadata = {
        "grid": grid,
        "fourier_sign": 1,
        "real_pressure": True,
        "pressure_type": "total_pressure",
        "data_units": "instrument_units",
        "spectrum_normalization": "dtft_dt",
        "measurement_provenance": "fixture",
    }
    write_artifact(
        folder / "frequency.json", "wust.frequency_input", metadata, arrays, axes, units
    )
    invocations = []

    def invoke(name, operation, input_name, config, **kwargs):
        request = {
            "schema": "wust.request",
            "schema_version": 1,
            "operation": operation,
            "input_manifest": str(folder / input_name),
            "output_manifest": str(folder / f"{name}.json"),
            "config": config,
            **kwargs,
        }
        path = folder / f"{name}.request.json"
        path.write_text(json.dumps(request, allow_nan=False))
        value = run(path, args.matlab, timeout_s=180, allow_dirty=True)
        invocations.append(name)
        return value

    invoke("observations", "ingest_frequency", "frequency.json", {})
    obs, a = read_artifact(folder / "observations.json", "wust.measurements")
    clean = pressure.copy()
    clean[~mask] = 0
    np.testing.assert_array_equal(a["Y"], np.conj(clean[[1, 0]]).transpose(1, 2, 0))
    np.testing.assert_array_equal(a["mask"], mask[[1, 0]].transpose(1, 2, 0))
    np.testing.assert_array_equal(
        a["tx_index"], np.array([7 * 23 + 6 + 1, 19 * 23 + 16 + 1])
    )
    np.testing.assert_allclose(a["x_m"], x, rtol=0, atol=1e-17)
    np.testing.assert_allclose(a["y_m"], y, rtol=0, atol=1e-17)
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
    config = {
        "backend": "cpu",
        "bounds_mps": [1300, 1800],
        "max_update_mps": 12,
        "step_damping": 0.25,
        "source_batch_size": 2,
        "pml_strength": 10,
        "pml_m": 0.003,
        "stencil_bounds": [1400, 1700],
        "wavenumber": "continuum",
        "filter_cutoff": 0,
        "filter_order": 4,
    }
    init_args = {"initial_manifest": str(folder / "initial.json")}
    zero = invoke(
        "zero", "reconstruct", "observations.json", config, schedule=[], **init_args
    )
    _, z = read_artifact(folder / "zero.json")
    np.testing.assert_array_equal(z["c_mps"], initial)
    assert zero["metadata"]["completion"]["reason"] == "zero_updates"
    expression = f"addpath({quote(ROOT/'Runtime/tests')}); export_integration_fixture({quote(folder/'frequency.json')},{quote(folder/'coherent.json')});"
    execute([args.matlab, "-batch", expression], 180)
    coherent = read_json(folder / "coherent.json")
    coherent["arrays_sha256"] = sha256(folder / coherent["arrays_file"])
    (folder / "coherent.json").write_text(json.dumps(coherent))
    invoke("physical_observations", "ingest_frequency", "coherent.json", {})
    result = invoke(
        "reconstruction",
        "reconstruct",
        "physical_observations.json",
        config,
        schedule=[1, 2],
        planned_schedule_length=4,
        **init_args,
    )
    _, r = read_artifact(folder / "reconstruction.json")
    assert np.isfinite(r["c_mps"]).all()
    assert r["c_mps"].dtype == np.float64 and r["history_mps"].dtype == np.float32
    assert result["metadata"]["completion"]["reason"] == "caller_truncated_schedule"
    assert result["metadata"]["completed_updates"] == 2 and result["metadata"][
        "executed_schedule"
    ] == [1, 2]
    assert [v["loss_model_step"] for v in result["metadata"]["records"]] == [0, 1]
    assert result["metadata"]["final_data_residual"] is None
    # Pressure-derived RF/direct comparison through two actual batch calls.
    time = np.arange(256) * 1e-7 + 7e-7
    rf = np.sin(np.arange(256 * 4 * 2) * 0.17).reshape(256, 4, 2)
    rf_arrays = {"pressure": rf, "time_s": time, "tx_xy_m": tx, "rx_xy_m": rx}
    write_artifact(
        folder / "rf.json",
        "wust.rf",
        {"data_units": "instrument_units", "measurement_provenance": "fixture"},
        rf_arrays,
        {
            "pressure": "time,rx,tx",
            "time_s": "time",
            "tx_xy_m": "tx,xy",
            "rx_xy_m": "rx,xy",
        },
        {"pressure": "instrument_units", "time_s": "s", "tx_xy_m": "m", "rx_xy_m": "m"},
    )
    sorted_f = frequencies[::-1]
    invoke(
        "prepared",
        "prepare",
        "rf.json",
        {
            "grid": grid,
            "frequencies_hz": sorted_f.tolist(),
            "c_geom_mps": 1500,
            "window": "none",
            "phase_correction": "none",
            "mask": np.ones((2, 4), bool).tolist(),
        },
    )
    direct = (
        np.einsum("ft,trn->fnr", np.exp(-2j * np.pi * sorted_f[:, None] * time), rf)
        * np.diff(time).mean()
    )
    arrays.update(
        pressure=direct.astype("complex64"),
        mask=np.ones_like(mask),
        frequencies_hz=sorted_f,
    )
    metadata["fourier_sign"] = -1
    write_artifact(
        folder / "direct.json", "wust.frequency_input", metadata, arrays, axes, units
    )
    invoke("direct_prepared", "ingest_frequency", "direct.json", {})
    _, left = read_artifact(folder / "prepared.json")
    _, right = read_artifact(folder / "direct_prepared.json")
    np.testing.assert_allclose(left["Y"], right["Y"], rtol=2e-6, atol=1e-12)
    # A degenerate fit is an explicit failure, not a successful zero scale.
    a["mask"][:] = False
    a["mask"][:, 0, :] = True
    write_artifact(
        folder / "degenerate.json",
        "wust.measurements",
        obs["metadata"],
        a,
        {k: v["axes"] for k, v in obs["arrays"].items()},
        {k: v["units"] for k, v in obs["arrays"].items()},
    )
    try:
        invoke(
            "failed",
            "reconstruct",
            "degenerate.json",
            config,
            schedule=[1],
            **init_args,
        )
    except RuntimeFailure as error:
        assert error.reason == "numerical_failure"
    else:
        raise AssertionError("Degenerate source fit was accepted")
    assert not (folder / "failed.json").exists()
    print(
        json.dumps(
            {
                "passed": True,
                "successful_invocations": invocations,
                "expected_numerical_failures": 1,
            }
        )
    )


if __name__ == "__main__":
    main()
