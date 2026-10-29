"""Versioned JSON/HDF5 transport, without physical normalization or readback."""

import json
import os
import tempfile
from pathlib import Path

from contracts import FIELDS, PROTOCOL_VERSION, exact_keys, schema


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, value):
    """Publish a completed manifest after its arrays are closed."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def read_manifest(path, expected=None):
    """Validate the transport envelope without loading observation arrays."""
    path = Path(path).resolve(strict=True)
    doc = read_json(path)
    schema(doc, expected)
    exact_keys(doc, {"schema", "schema_version", "metadata", "arrays_file", "arrays"})
    if not isinstance(doc["metadata"], dict):
        raise ValueError("metadata must be an object")  # noqa: TRY004
    fields = FIELDS[doc["schema"]]
    exact_keys(
        doc["arrays"],
        [key for key, spec in fields.items() if not spec["optional"]],
        fields,
    )
    name = doc["arrays_file"]
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("arrays_file must be a sibling filename")
    data = (path.parent / name).resolve(strict=True)
    if data.parent != path.parent:
        raise ValueError("Array file must remain in the manifest directory")
    for key, spec in doc["arrays"].items():
        field = fields[key]
        complex_value = spec.get("dtype") in ("complex64", "complex128")
        datasets = {"real_dataset", "imag_dataset"} if complex_value else {"dataset"}
        exact_keys(spec, {"shape", "dtype", "axes", "units"} | datasets)
        units = (
            doc["metadata"].get("data_units")
            if field["units"] == "data_units"
            else field["units"]
        )
        if (
            spec["axes"] not in field["axes"]
            or spec["units"] != units
            or spec["dtype"] not in field["dtypes"]
        ):
            raise ValueError(f"Invalid axes/units/dtype for {key}")
        if (
            not isinstance(spec["shape"], list)
            or len(spec["shape"]) != len(spec["axes"].split(","))
            or any(type(n) is not int or n < 1 for n in spec["shape"])
        ):
            raise ValueError(f"Invalid shape for {key}")
        expected = (
            {"real_dataset": f"/{key}_real", "imag_dataset": f"/{key}_imag"}
            if complex_value
            else {"dataset": f"/{key}"}
        )
        if any(spec[name] != value for name, value in expected.items()):
            raise ValueError(f"Invalid dataset names for {key}")
    return doc


def read_artifact(path, expected=None):
    import h5py
    import numpy as np

    path = Path(path).resolve(strict=True)
    doc = read_manifest(path, expected)
    data = path.parent / doc["arrays_file"]
    result = {}
    with h5py.File(data, "r") as handle:
        for key, spec in doc["arrays"].items():

            def load(dataset, dtype, spec=spec, key=key):
                if not isinstance(handle.get(dataset, getlink=True), h5py.HardLink):
                    raise ValueError(  # noqa: TRY004
                        "Only local datasets are supported"
                    )
                obj = handle[dataset]
                if (
                    not isinstance(obj, h5py.Dataset)
                    or obj.is_virtual
                    or obj.external
                    or Path(obj.file.filename).resolve() != data
                ):
                    raise ValueError("External/virtual HDF5 data is not supported")
                if obj.shape != tuple(spec["shape"]) or obj.dtype != np.dtype(dtype):
                    raise ValueError(
                        f"Dataset shape/dtype differs from manifest: {key}"
                    )
                return obj[...]

            if spec["dtype"] in ("complex64", "complex128"):
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
    return doc, result


def write_artifact(path, name, metadata, arrays, axes, units):
    """Write each array once, followed by its manifest; no whole-array readback."""
    import h5py
    import numpy as np

    path = Path(path).absolute()
    data = path.with_suffix(".h5")
    if path.exists() or data.exists():
        raise FileExistsError(path)
    schema({"schema": name, "schema_version": PROTOCOL_VERSION})
    fields = FIELDS[name]
    exact_keys(
        arrays, [key for key, spec in fields.items() if not spec["optional"]], fields
    )
    descriptors = {}
    with h5py.File(data, "x") as handle:
        for key, array in arrays.items():
            value = np.asarray(array)
            spec = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "axes": axes[key],
                "units": units[key],
            }
            if np.iscomplexobj(value):
                spec.update(real_dataset=f"/{key}_real", imag_dataset=f"/{key}_imag")
                handle.create_dataset(spec["real_dataset"], data=value.real)
                handle.create_dataset(spec["imag_dataset"], data=value.imag)
            else:
                spec["dataset"] = f"/{key}"
                handle.create_dataset(
                    spec["dataset"],
                    data=value.astype("uint8") if value.dtype == bool else value,
                )
            descriptors[key] = spec
    doc = {
        "schema": name,
        "schema_version": PROTOCOL_VERSION,
        "metadata": metadata,
        "arrays_file": data.name,
        "arrays": descriptors,
    }
    atomic_json(path, doc)
    return doc
