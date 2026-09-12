#!/usr/bin/env python3
"""Expand one k-Wave pressure-source input and execute the native batch binary."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np


def copy_attrs(target: h5py.Dataset, source: h5py.Dataset) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def expand_input(
    source_path: Path, target_path: Path, batch_size: int
) -> dict[str, object]:
    if source_path.resolve() == target_path.resolve() or target_path.exists():
        raise ValueError("Expansion requires a new file, never mutate source input")
    if batch_size < 1:
        raise ValueError("Positive batch required")
    shutil.copyfile(source_path, target_path)
    with h5py.File(source_path, "r") as source:
        source_index_attrs = dict(source["p_source_index"].attrs)
        source_input_attrs = dict(source["p_source_input"].attrs)
        sensor_attrs = dict(source["sensor_mask_index"].attrs)

    with h5py.File(target_path, "r+") as handle:
        nx = int(np.asarray(handle["Nx"][...]).reshape(-1)[0])
        ny = int(np.asarray(handle["Ny"][...]).reshape(-1)[0])
        nz = int(np.asarray(handle["Nz"][...]).reshape(-1)[0])
        field_elements = nx * ny * nz
        if nz != 1 or field_elements * batch_size >= 2**32:
            raise ValueError(
                "Native runtime supports 2D uint32-addressable fields only"
            )
        for flag in (
            "nonlinear_flag",
            "absorbing_flag",
            "axisymmetric_flag",
            "ux_source_flag",
            "uy_source_flag",
            "uz_source_flag",
            "p0_source_flag",
            "transducer_source_flag",
            "nonuniform_grid_flag",
        ):
            if flag in handle and np.any(handle[flag][...] != 0):
                raise ValueError("Unsupported native physics: " + flag)
        for density in ("rho0", "rho0_sgx", "rho0_sgy"):
            if density in handle and np.asarray(handle[density][...]).size != 1:
                raise ValueError(
                    "Native production requires scalar density: " + density
                )
        if np.asarray(handle["p_source_index"][...]).size != 1:
            raise ValueError("Expected one source before independent-TX expansion")
        sensor = (
            np.asarray(handle["sensor_mask_index"][...]).reshape(-1).astype(np.uint64)
        )
        if batch_size > sensor.size:
            raise ValueError(
                f"batch {batch_size} exceeds colocated sensor count {sensor.size}"
            )
        if "alpha_coeff" in handle:
            alpha = np.asarray(handle["alpha_coeff"][...])
            if np.any(alpha != 0):
                raise ValueError(
                    "native production branch currently requires alpha_coeff == 0 everywhere"
                )
        waveform = (
            np.asarray(handle["p_source_input"][...]).reshape(-1).astype(np.float32)
        )
        explicit_sources = os.environ.get("KWAVE_NATIVE_TX_INDICES_H5", "").strip()
        if explicit_sources:
            physical_sources = np.asarray(
                [int(value) for value in explicit_sources.split(",") if value],
                dtype=np.uint64,
            )
            if physical_sources.size != batch_size:
                raise ValueError(
                    f"explicit source count {physical_sources.size} != batch size {batch_size}"
                )
        else:
            physical_sources = sensor[:batch_size]
        for indices in (sensor, physical_sources):
            if (
                np.any(indices < 1)
                or np.any(indices > field_elements)
                or np.unique(indices).size != indices.size
            ):
                raise ValueError("Invalid or duplicated physical grid indices")
        encoded_sources = (
            physical_sources + np.arange(batch_size, dtype=np.uint64) * field_elements
        )
        encoded_sensors = np.concatenate(
            [sensor + index * field_elements for index in range(batch_size)]
        ).astype(np.uint64, copy=False)

        for name in ("p_source_index", "p_source_input", "sensor_mask_index"):
            del handle[name]
        source_index_ds = handle.create_dataset(
            "p_source_index", data=encoded_sources.reshape(batch_size, 1, 1)
        )
        source_input_ds = handle.create_dataset(
            "p_source_input",
            data=np.repeat(waveform[:, None], batch_size, axis=1).reshape(
                waveform.size, batch_size, 1
            ),
        )
        sensor_ds = handle.create_dataset(
            "sensor_mask_index", data=encoded_sensors.reshape(1, 1, -1)
        )
        for ds, attrs in (
            (source_index_ds, source_index_attrs),
            (source_input_ds, source_input_attrs),
            (sensor_ds, sensor_attrs),
        ):
            for key, value in attrs.items():
                ds.attrs[key] = value
        handle["p_source_many"][...] = np.uint64(1)
        handle["absorbing_flag"][...] = np.uint64(0)
        handle.create_dataset(
            "native_source_batch_size",
            data=np.asarray([[[batch_size]]], dtype=np.uint64),
        )

    return {
        "batch_size": batch_size,
        "physical_grid": [nx, ny, nz],
        "sensors_per_batch": int(sensor.size),
        "physical_source_indices": physical_sources.astype(int).tolist(),
        "encoded_sensor_count": int(encoded_sensors.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--native-binary", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--receipt", default="")
    known, solver_args = parser.parse_known_args()
    try:
        input_pos = solver_args.index("-i") + 1
        output_pos = solver_args.index("-o") + 1
    except (ValueError, IndexError) as exc:
        raise ValueError(
            "solver arguments must contain -i INPUT and -o OUTPUT"
        ) from exc

    input_path = Path(solver_args[input_pos]).resolve()
    output_path = Path(solver_args[output_pos]).resolve()
    with tempfile.TemporaryDirectory(
        prefix="kwave_native_batch_", dir=str(input_path.parent)
    ) as temp_dir:
        expanded_path = Path(temp_dir) / "native_batch_input.h5"
        receipt = expand_input(input_path, expanded_path, known.batch_size)
        solver_args[input_pos] = str(expanded_path)
        environment = os.environ.copy()
        environment["KWAVE_NATIVE_SOURCE_BATCH"] = "1"
        environment.setdefault("OMP_NUM_THREADS", "4")
        start = time.perf_counter()
        completed = subprocess.run(
            [known.native_binary, *solver_args], env=environment, check=False
        )
        receipt.update(
            {
                "schema": "kwave_native_source_batch_run_v1",
                "input": str(input_path),
                "output": str(output_path),
                "native_binary": str(Path(known.native_binary).resolve()),
                "wall_seconds": time.perf_counter() - start,
                "returncode": completed.returncode,
            }
        )
    receipt_path = (
        Path(known.receipt)
        if known.receipt
        else output_path.with_suffix(".native_batch.json")
    )
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
