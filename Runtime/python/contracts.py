"""Authoritative external envelope definitions; numerical checks live in MATLAB."""

import math

SCHEMAS = {
    name: 1
    for name in (
        "request",
        "frequency_input",
        "measurements",
        "reconstruction",
        "capabilities",
        "initial_model",
        "rf",
        "simulation_input",
        "execution_failure",
        "build",
    )
}
SCHEMAS = {"wust." + key: value for key, value in SCHEMAS.items()}
INPUTS = {
    "ingest_frequency": "wust.frequency_input",
    "reconstruct": "wust.measurements",
    "prepare": "wust.rf",
    "simulate": "wust.simulation_input",
}
CONFIG_FIELDS = {
    "backend",
    "bounds_mps",
    "max_update_mps",
    "step_damping",
    "source_batch_size",
    "pml_strength",
    "pml_m",
    "stencil_bounds",
    "wavenumber",
    "filter_cutoff",
    "filter_order",
}
ARRAYS = {
    "wust.frequency_input": {
        "pressure",
        "mask",
        "frequencies_hz",
        "tx_xy_m",
        "rx_xy_m",
    },
    "wust.measurements": {
        "Y",
        "mask",
        "frequencies_hz",
        "tx_xy_m",
        "rx_xy_m",
        "tx_index",
        "rx_index",
        "x_m",
        "y_m",
        "tx_snapped_xy_m",
        "rx_snapped_xy_m",
    },
    "wust.initial_model": {"initial_mps", "update_mask"},
    "wust.rf": {"pressure", "time_s", "tx_xy_m", "rx_xy_m"},
    "wust.simulation_input": {"c_mps", "tx_xy_m", "rx_xy_m", "source_pressure"},
    "wust.reconstruction": {"c_mps", "history_mps"},
}
AXES = {
    "Y": ["tx,rx,frequency"],
    "pressure": ["frequency,tx,rx", "tx,rx,frequency", "time,rx,tx"],
    "mask": ["frequency,tx,rx", "tx,rx,frequency", "tx,rx"],
    "frequencies_hz": ["frequency"],
    "tx_xy_m": ["tx,xy"],
    "rx_xy_m": ["rx,xy"],
    "tx_snapped_xy_m": ["tx,xy"],
    "rx_snapped_xy_m": ["rx,xy"],
    "tx_index": ["tx"],
    "rx_index": ["rx"],
    "x_m": ["x"],
    "y_m": ["y"],
    "c_mps": ["y,x"],
    "initial_mps": ["y,x"],
    "update_mask": ["y,x"],
    "history_mps": ["y,x,model_step"],
    "time_s": ["time"],
    "source_pressure": ["time"],
}


def exact_keys(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004 - one admission-error type
            "Expected an object"
        )
    missing, unknown = set(required) - value.keys(), value.keys() - set(required) - set(
        optional
    )
    if missing or unknown:
        raise ValueError(
            f"Missing fields {sorted(missing)}; unsupported fields {sorted(unknown)}"
        )


def schema(value, expected=None):
    if not isinstance(value, dict):
        raise ValueError("Schema envelope must be an object")  # noqa: TRY004
    name = value.get("schema", "")
    if str(name).startswith("wfi."):
        raise ValueError(
            "Legacy wfi.*.v1 is incompatible. Re-export explicit WUST JSON/HDF5; use a historical tag for MAT requests."
        )
    if (
        not isinstance(name, str)
        or name not in SCHEMAS
        or (expected is not None and name != expected)
    ):
        raise ValueError(f"Unsupported schema {name!r}; expected {expected}")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")


def number(value, label, minimum=0, inclusive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite real number")
    if value < minimum or (not inclusive and value == minimum):
        raise ValueError(f"{label} outside its legal range")


def validate_request(req):
    schema(req, "wust.request")
    exact_keys(
        req,
        {
            "schema",
            "schema_version",
            "operation",
            "input_manifest",
            "output_manifest",
            "config",
        },
        {"initial_manifest", "schedule", "planned_schedule_length"},
    )
    operation = req["operation"]
    if not isinstance(operation, str) or operation not in INPUTS:
        raise ValueError("Unsupported operation")
    if operation == "reconstruct":
        if not {"initial_manifest", "schedule"} <= req.keys():
            raise ValueError(
                "Reconstruction requires resolved initial map and schedule"
            )
        cfg = req["config"]
        exact_keys(cfg, CONFIG_FIELDS, {"dispersion"})
        for key in (
            "max_update_mps",
            "step_damping",
            "source_batch_size",
            "pml_strength",
            "pml_m",
            "filter_order",
        ):
            number(cfg[key], key)
        number(cfg["filter_cutoff"], "filter_cutoff", inclusive=True)
        for key in ("source_batch_size", "filter_order"):
            if type(cfg[key]) is not int:
                raise ValueError(f"{key} must be an integer")
        for key in ("bounds_mps", "stencil_bounds"):
            bounds = cfg[key]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"{key} requires two bounds")
            number(bounds[0], key)
            number(bounds[1], key, minimum=bounds[0])
        if cfg["backend"] not in ("cpu", "gpu") or cfg["wavenumber"] not in (
            "continuum",
            "kwave-ldr9",
        ):
            raise ValueError("Unsupported backend/wavenumber")
        if cfg["wavenumber"] == "kwave-ldr9":
            exact_keys(
                cfg.get("dispersion"),
                {"time_step_s", "reference_speed_mps", "model_reference_speed_mps"},
            )
            for key, value in cfg["dispersion"].items():
                number(value, key)
        elif "dispersion" in cfg:
            raise ValueError("dispersion is only supported with kwave-ldr9")
        schedule = req["schedule"]
        if not isinstance(schedule, list) or any(
            type(x) is not int or x < 1 for x in schedule
        ):
            raise ValueError("schedule must contain 1-based integer frequency indices")
        count = req.get("planned_schedule_length", len(schedule))
        if type(count) is not int or count < len(schedule):
            raise ValueError(
                "planned_schedule_length cannot be less than executed schedule length"
            )
    else:
        if {"initial_manifest", "schedule", "planned_schedule_length"} & req.keys():
            raise ValueError("Schedule and initial map belong only to reconstruction")
        if operation == "ingest_frequency":
            exact_keys(req["config"], set())
        elif operation == "prepare":
            exact_keys(
                req["config"],
                {
                    "grid",
                    "frequencies_hz",
                    "c_geom_mps",
                    "window",
                    "phase_correction",
                    "mask",
                },
            )
        else:
            exact_keys(
                req["config"],
                {
                    "backend",
                    "pml_size",
                    "density_kg_m3",
                    "sound_speed_ref_mps",
                    "downsample_factor",
                    "time_offset_s",
                    "work_dir",
                },
                {"binary_path", "device_num"},
            )
    return req


def request_json_schema():
    """Envelope schema; detailed model/array rules are enforced by validators."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema",
            "schema_version",
            "operation",
            "input_manifest",
            "output_manifest",
            "config",
        ],
        "properties": {
            "schema": {"const": "wust.request"},
            "schema_version": {"const": 1, "type": "integer"},
            "operation": {"enum": list(INPUTS)},
            "input_manifest": {"type": "string"},
            "output_manifest": {"type": "string"},
            "initial_manifest": {"type": "string"},
            "config": {"type": "object"},
            "schedule": {"type": "array", "items": {"type": "integer", "minimum": 1}},
            "planned_schedule_length": {"type": "integer", "minimum": 0},
        },
    }
