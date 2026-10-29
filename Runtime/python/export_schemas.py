"""Export the seven v2 schemas and MATLAB array field definitions."""

import json
from pathlib import Path

from contracts import FIELDS, PROTOCOL_VERSION, request_json_schema

DESTINATION = Path(__file__).resolve().parents[1] / "schemas"


def definitions():
    descriptor = {
        "type": "object",
        "additionalProperties": False,
        "required": ["axes", "shape", "dtype", "units"],
        "properties": {
            "axes": {"type": "string"},
            "units": {"type": "string"},
            "dtype": {"type": "string"},
            "shape": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "integer", "minimum": 1},
            },
            **{
                key: {"type": "string", "pattern": "^/"}
                for key in ("dataset", "real_dataset", "imag_dataset")
            },
        },
        "oneOf": [
            {
                "required": ["dataset"],
                "not": {
                    "anyOf": [
                        {"required": ["real_dataset"]},
                        {"required": ["imag_dataset"]},
                    ]
                },
            },
            {
                "required": ["real_dataset", "imag_dataset"],
                "not": {"required": ["dataset"]},
            },
        ],
    }
    documents = {"wust.request": request_json_schema()}
    for name, fields in FIELDS.items():
        properties = {}
        for key, field in fields.items():
            properties[key] = {
                "$ref": "#/$defs/array",
                "properties": {
                    "axes": {"enum": field["axes"]},
                    "dtype": {"enum": field["dtypes"]},
                },
            }
            if field["units"] != "data_units":
                properties[key]["properties"]["units"] = {"const": field["units"]}
        documents[name] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {"array": descriptor},
            "array_fields": fields,
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schema",
                "schema_version",
                "metadata",
                "arrays_file",
                "arrays",
            ],
            "properties": {
                "schema": {"const": name},
                "schema_version": {"type": "integer", "const": PROTOCOL_VERSION},
                "metadata": {"type": "object"},
                "arrays_file": {"type": "string", "pattern": "^[^/\\\\]+$"},
                "arrays": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        key for key, spec in fields.items() if not spec["optional"]
                    ],
                    "properties": properties,
                },
            },
        }
    return documents


def main():
    DESTINATION.mkdir(exist_ok=True)
    for name, document in definitions().items():
        (DESTINATION / (name + ".json")).write_text(
            json.dumps(document, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
