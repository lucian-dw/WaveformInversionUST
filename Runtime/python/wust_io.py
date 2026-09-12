"""Explicit JSON/HDF5 transport. HDF5 dimensions follow the declared Python axes."""

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

from contracts import ARRAYS, AXES, exact_keys, schema


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hashes(arrays):
    import numpy as np

    result = {}
    for key, value in arrays.items():
        value = np.ascontiguousarray(value)
        digest = hashlib.sha256(
            json.dumps({"shape": value.shape, "dtype": str(value.dtype)}).encode()
        )
        digest.update(value.tobytes())
        result[key] = digest.hexdigest()
    return result


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=invalid,
    )


def atomic_json(path, value):
    """Publish without overwriting even if another invocation races us."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.link(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def read_artifact(path, expected=None):
    import h5py
    import numpy as np

    path = Path(path).resolve(strict=True)
    doc = read_json(path)
    schema(doc, expected)
    exact_keys(
        doc,
        {
            "schema",
            "schema_version",
            "metadata",
            "arrays_file",
            "arrays_sha256",
            "arrays",
        },
    )
    if not isinstance(doc["metadata"], dict):
        raise ValueError("metadata must be an object")  # noqa: TRY004 - admission error
    exact_keys(doc["arrays"], ARRAYS[doc["schema"]])
    name = doc["arrays_file"]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError(
            "arrays_file must be a sibling filename, not an arbitrary path"
        )
    h5path = (path.parent / name).resolve(strict=True)
    if h5path.parent != path.parent or sha256(h5path) != doc["arrays_sha256"]:
        raise ValueError("Array artifact location/hash mismatch")
    result = {}
    with h5py.File(h5path, "r") as handle:
        for key, spec in doc["arrays"].items():
            complex_value = spec.get("dtype") in ("complex64", "complex128")
            exact_keys(
                spec,
                {"axes", "shape", "dtype", "units"}
                | ({"real_dataset", "imag_dataset"} if complex_value else {"dataset"}),
            )
            if spec["axes"] not in AXES[key] or not isinstance(spec["units"], str):
                raise ValueError(f"Unsupported axes/units for {key}")
            shape = spec["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) != len(spec["axes"].split(","))
                or any(type(n) is not int or n < 1 for n in shape)
            ):
                raise ValueError(f"Invalid declared shape for {key}")
            if spec["dtype"] not in (
                "float32",
                "float64",
                "complex64",
                "complex128",
                "int64",
                "uint8",
                "bool",
            ):
                raise ValueError("Unsupported dtype")

            def load(dataset, dtype, shape=tuple(shape), key=key):
                if (
                    not isinstance(dataset, str)
                    or not dataset.startswith("/")
                    or not isinstance(handle.get(dataset, getlink=True), h5py.HardLink)
                ):
                    raise ValueError("Only local HDF5 datasets are supported")
                obj = handle[dataset]
                if (
                    not isinstance(obj, h5py.Dataset)
                    or obj.is_virtual
                    or obj.external
                    or Path(obj.file.filename).resolve() != h5path
                ):
                    raise ValueError("External/virtual HDF5 data is not supported")
                if obj.shape != tuple(shape) or obj.dtype != np.dtype(dtype):
                    raise ValueError(
                        f"Dataset shape/dtype differs from manifest: {key}"
                    )
                return obj[...]

            if complex_value:
                dtype = "float32" if spec["dtype"] == "complex64" else "float64"
                value = load(spec["real_dataset"], dtype) + 1j * load(
                    spec["imag_dataset"], dtype
                )
            else:
                value = load(
                    spec["dataset"],
                    "uint8" if spec["dtype"] == "bool" else spec["dtype"],
                )
                if spec["dtype"] == "bool":
                    if not np.isin(value, [0, 1]).all():
                        raise ValueError(
                            "Boolean dataset contains values other than 0/1"
                        )
                    value = value.astype(bool)
            result[key] = value
    validate_arrays(doc, result)
    return doc, result


def validate_arrays(doc, arrays):
    import numpy as np

    name = doc["schema"]
    meta = doc["metadata"]
    specs = doc["arrays"]
    for key, spec in specs.items():
        if key == "source_pressure":
            expected = "Pa"
        elif key in ("pressure", "Y"):
            expected = meta.get("data_units")
        elif key.endswith("_mps"):
            expected = "m/s"
        elif key.endswith("_m"):
            expected = "m"
        elif key == "frequencies_hz":
            expected = "Hz"
        elif key == "time_s":
            expected = "s"
        else:
            expected = "1"
        if spec["units"] != expected:
            raise ValueError(f"Unit mismatch for {key}: expected {expected}")
    for key in ("mask", "update_mask"):
        if key in arrays and not np.isin(arrays[key], [0, 1]).all():
            raise ValueError("Mask must contain only 0/1")
    for key in ("tx_xy_m", "rx_xy_m"):
        if key in arrays and (
            arrays[key].ndim != 2
            or arrays[key].shape[1] != 2
            or not np.isrealobj(arrays[key])
            or not np.isfinite(arrays[key]).all()
        ):
            raise ValueError(f"{key} requires finite physical [x,y] positions")
    if name == "wust.rf":
        if specs["pressure"]["axes"] != "time,rx,tx" or not np.isrealobj(
            arrays["pressure"]
        ):
            raise ValueError("RF pressure requires real [time,rx,tx] samples")
        if arrays["pressure"].shape != (
            len(arrays["time_s"]),
            len(arrays["rx_xy_m"]),
            len(arrays["tx_xy_m"]),
        ):
            raise ValueError("RF pressure/acquisition shapes disagree")
    for key in ("initial_mps", "c_mps", "history_mps"):
        if key in arrays and (
            not np.isrealobj(arrays[key])
            or not np.isfinite(arrays[key]).all()
            or (arrays[key] <= 0).any()
        ):
            raise ValueError(f"{key} must contain finite positive sound speeds")
    if name in ("wust.frequency_input", "wust.measurements"):
        required = {
            "grid",
            "fourier_sign",
            "pressure_type",
            "data_units",
            "spectrum_normalization",
        }
        if not required <= meta.keys() or meta["pressure_type"] != "total_pressure":
            raise ValueError("Explicit total-pressure metadata required")
        if type(meta["fourier_sign"]) is not int or meta["fourier_sign"] not in (-1, 1):
            raise ValueError("Unknown Fourier convention")
        if name == "wust.frequency_input" and (
            "measurement_provenance" not in meta
            or (meta["fourier_sign"] == 1 and meta.get("real_pressure") is not True)
        ):
            raise ValueError(
                "Explicit provenance and real-pressure declaration required for sign conversion"
            )
        if name == "wust.measurements" and meta["fourier_sign"] != -1:
            raise ValueError("Canonical measurements require Fourier sign -1")
        f = arrays["frequencies_hz"]
        if not np.isfinite(f).all() or (f <= 0).any() or len(np.unique(f)) != len(f):
            raise ValueError("Frequencies must be finite positive and unique")
        if name == "wust.measurements" and not (np.diff(f) > 0).all():
            raise ValueError("Canonical frequencies must increase")
        key = "Y" if name == "wust.measurements" else "pressure"
        if not specs[key]["dtype"].startswith("complex"):
            raise ValueError("Explicit complex real/imag pressure datasets required")
        axes = specs[key]["axes"]
        if axes not in ("frequency,tx,rx", "tx,rx,frequency"):
            raise ValueError("Frequency data require declared frequency/TX/RX axes")
        p = arrays[key].transpose(1, 2, 0) if axes == "frequency,tx,rx" else arrays[key]
        m = arrays["mask"]
        ma = specs["mask"]["axes"]
        if ma == "frequency,tx,rx":
            m = m.transpose(1, 2, 0)
        elif ma == "tx,rx":
            m = np.broadcast_to(m[..., None], p.shape)
        if (
            p.shape != (len(arrays["tx_xy_m"]), len(arrays["rx_xy_m"]), len(f))
            or m.shape != p.shape
        ):
            raise ValueError("Pressure/mask/acquisition shapes disagree")
        if not m.any() or not np.isfinite(p[m.astype(bool)]).all():
            raise ValueError("Nonfinite valid pressure or empty mask")
    if name == "wust.reconstruction":
        if meta.get("completion", {}).get("reason") not in (
            "zero_updates",
            "schedule_complete",
            "caller_truncated_schedule",
        ):
            raise ValueError("Missing successful completion state")
        if (
            arrays["c_mps"].dtype != np.float64
            or arrays["history_mps"].dtype != np.float32
        ):
            raise ValueError("Final/history precision contract violated")
        n = meta.get("completed_updates")
        if (
            type(n) is not int
            or n < 0
            or len(meta.get("executed_schedule", [])) != n
            or len(meta.get("records", [])) != n
        ):
            raise ValueError("Executed schedule/record counts disagree")
        if arrays["history_mps"].shape != arrays["c_mps"].shape + (n + 1,):
            raise ValueError("History shape disagrees with completed updates")


def write_artifact(path, name, metadata, arrays, axes, units):
    """Write typed arrays first, publish the success manifest last."""
    import h5py
    import numpy as np

    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    schema({"schema": name, "schema_version": 1})
    exact_keys(arrays, ARRAYS[name])
    h5path = path.parent / f"{path.stem}.{uuid.uuid4().hex}.h5"
    specs = {}
    published = False
    try:
        with h5py.File(h5path, "x") as handle:
            for key, value in arrays.items():
                value = np.asarray(value)
                spec = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "axes": axes[key],
                    "units": units[key],
                }
                if np.iscomplexobj(value):
                    spec.update(
                        real_dataset=f"/{key}_real", imag_dataset=f"/{key}_imag"
                    )
                    handle.create_dataset(spec["real_dataset"], data=value.real)
                    handle.create_dataset(spec["imag_dataset"], data=value.imag)
                else:
                    spec["dataset"] = f"/{key}"
                    handle.create_dataset(
                        spec["dataset"],
                        data=value.astype("uint8") if value.dtype == bool else value,
                    )
                specs[key] = spec
        doc = {
            "schema": name,
            "schema_version": 1,
            "metadata": metadata,
            "arrays_file": h5path.name,
            "arrays_sha256": sha256(h5path),
            "arrays": specs,
        }
        # Validate before publishing, without exposing a final-looking manifest.
        with tempfile.TemporaryDirectory(prefix=".validate-", dir=path.parent) as tmp:
            check = Path(tmp) / "manifest.json"
            os.link(h5path, Path(tmp) / h5path.name)
            check.write_text(json.dumps(doc, allow_nan=False))
            read_artifact(check, name)
        atomic_json(path, doc)
        published = True
        return doc
    finally:
        if not published:
            h5path.unlink(missing_ok=True)
