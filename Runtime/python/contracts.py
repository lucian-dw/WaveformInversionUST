"""WUST v2 transport fields; numerical interpretation belongs to MATLAB."""

import math

PROTOCOL_VERSION = 2
INPUTS = {
    "ingest_frequency": ("wust.frequency_input",),
    "prepare": ("wust.rf",),
    "reconstruct": ("wust.frequency_input", "wust.measurements"),
    "simulate": ("wust.simulation_input",),
}


def field(axes, units, dtype, *, optional=False):
    return {
        "axes": axes.split("|"),
        "units": units,
        "dtypes": dtype.split("|"),
        "optional": optional,
    }


REAL = "float32|float64"
COMPLEX = "complex64|complex128"
MASK = "bool|uint8"
GEOMETRY = {
    "frequencies_hz": field("frequency", "Hz", REAL),
    "tx_xy_m": field("tx,xy", "m", REAL),
    "rx_xy_m": field("rx,xy", "m", REAL),
}
FIELDS = {
    "wust.frequency_input": {
        **GEOMETRY,
        "pressure": field("frequency,tx,rx|tx,rx,frequency", "data_units", COMPLEX),
        "mask": field("frequency,tx,rx|tx,rx,frequency|tx,rx", "1", MASK),
    },
    "wust.measurements": {
        **GEOMETRY,
        "Y": field("tx,rx,frequency", "data_units", COMPLEX),
        "mask": field("tx,rx,frequency", "1", MASK),
        "tx_index": field("tx", "1", "int64|float64"),
        "rx_index": field("rx", "1", "int64|float64"),
    },
    "wust.initial_model": {
        "initial_mps": field("y,x", "m/s", REAL),
        "update_mask": field("y,x", "1", MASK),
    },
    "wust.reconstruction": {
        "c_mps": field("y,x", "m/s", "float64"),
        "history_mps": field("y,x,model_step", "m/s", "float32", optional=True),
    },
    "wust.rf": {
        "pressure": field("time,rx,tx", "data_units", REAL),
        "time_s": field("time", "s", REAL),
        "tx_xy_m": GEOMETRY["tx_xy_m"],
        "rx_xy_m": GEOMETRY["rx_xy_m"],
    },
    "wust.simulation_input": {
        "c_mps": field("y,x", "m/s", REAL),
        "tx_xy_m": GEOMETRY["tx_xy_m"],
        "rx_xy_m": GEOMETRY["rx_xy_m"],
        "source_pressure": field("time", "Pa", REAL),
    },
}
SCHEMAS = {
    name: PROTOCOL_VERSION
    for name in (*FIELDS, "wust.request", "wust.capabilities", "wust.execution_failure")
}
REQUEST_FIELDS = {
    "schema": {"const": "wust.request"},
    "schema_version": {"const": PROTOCOL_VERSION, "type": "integer"},
    "operation": {"enum": list(INPUTS)},
    "input_manifest": {"type": "string"},
    "output_manifest": {"type": "string"},
    "config": {"type": "object"},
    "initial_manifest": {"type": "string"},
    "frequency_schedule_hz": {
        "type": ["array", "null"],
        "items": {"type": "number", "exclusiveMinimum": 0},
    },
    "max_updates": {"type": ["integer", "null"], "minimum": 0},
    "save_history": {"type": "boolean"},
}
BASE_REQUEST = {
    "schema",
    "schema_version",
    "operation",
    "input_manifest",
    "output_manifest",
    "config",
}
RECONSTRUCTION_REQUEST = {
    "initial_manifest",
    "frequency_schedule_hz",
    "max_updates",
    "save_history",
}


def exact_keys(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError("Expected an object")  # noqa: TRY004 - transport validation
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(
            f"Missing fields {sorted(missing)}; unsupported fields {sorted(unknown)}"
        )


def schema(value, expected=None):
    if not isinstance(value, dict):
        raise ValueError("Schema envelope must be an object")  # noqa: TRY004
    name = value.get("schema")
    accepted = (expected,) if isinstance(expected, str) else expected
    if (
        not isinstance(name, str)
        or name not in SCHEMAS
        or (accepted is not None and name not in accepted)
    ):
        raise ValueError(f"Unsupported schema {name!r}; expected {expected}")
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != PROTOCOL_VERSION
    ):
        raise ValueError("schema_version must be integer 2")


def number(value, label, minimum=0, inclusive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite real number")
    if value < minimum or (not inclusive and value == minimum):
        raise ValueError(f"{label} outside its legal range")


def validate_request(req):
    schema(req, "wust.request")
    exact_keys(req, BASE_REQUEST, RECONSTRUCTION_REQUEST)
    if req["operation"] not in INPUTS or not isinstance(req["config"], dict):
        raise ValueError("Unsupported operation/config")
    if req["operation"] == "reconstruct":
        if "initial_manifest" not in req:
            raise ValueError("Reconstruction requires an explicit initial model")
        schedule = req.get("frequency_schedule_hz")
        if schedule is not None:
            if not isinstance(schedule, list):
                raise ValueError("frequency_schedule_hz must be a list or null")
            for value in schedule:
                number(value, "frequency_schedule_hz")
        cap = req.get("max_updates")
        if cap is not None and (type(cap) is not int or cap < 0):
            raise ValueError("max_updates must be a nonnegative integer or null")
        if type(req.get("save_history", False)) is not bool:
            raise ValueError("save_history must be boolean")
    elif RECONSTRUCTION_REQUEST & req.keys():
        raise ValueError(
            "Initial model and update controls belong only to reconstruction"
        )
    if req["operation"] == "ingest_frequency":
        exact_keys(req["config"], set())
    return req


def request_json_schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(BASE_REQUEST),
        "properties": REQUEST_FIELDS,
        "allOf": [
            {
                "if": {"properties": {"operation": {"const": "reconstruct"}}},
                "then": {"required": ["initial_manifest"]},
                "else": {
                    "not": {
                        "anyOf": [
                            {"required": [key]}
                            for key in sorted(RECONSTRUCTION_REQUEST)
                        ]
                    }
                },
            }
        ],
        "x-input-schemas": INPUTS,
    }
