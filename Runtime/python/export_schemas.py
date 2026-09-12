"""Generate structural JSON Schemas from the maintained contract definitions."""

import argparse
import json
from pathlib import Path

from contracts import ARRAYS, AXES, SCHEMAS, request_json_schema

DESTINATION = Path(__file__).resolve().parents[1] / "schemas"


def definitions():
    documents = {"wust.request": request_json_schema()}
    for name, fields in ARRAYS.items():
        properties = {}
        for field in sorted(fields):
            spec = {
                "type": "object",
                "additionalProperties": False,
                "required": ["axes", "shape", "dtype", "units"],
                "properties": {
                    "axes": {"enum": AXES[field]},
                    "shape": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "integer", "minimum": 1},
                    },
                    "dtype": {
                        "enum": [
                            "float32",
                            "float64",
                            "complex64",
                            "complex128",
                            "bool",
                            "uint8",
                            "int64",
                        ]
                    },
                    "units": {"type": "string"},
                    "dataset": {"type": "string", "pattern": "^/"},
                    "real_dataset": {"type": "string", "pattern": "^/"},
                    "imag_dataset": {"type": "string", "pattern": "^/"},
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
            properties[field] = spec
        documents[name] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schema",
                "schema_version",
                "metadata",
                "arrays_file",
                "arrays_sha256",
                "arrays",
            ],
            "properties": {
                "schema": {"const": name},
                "schema_version": {"type": "integer", "const": SCHEMAS[name]},
                "metadata": {"type": "object"},
                "arrays_file": {"type": "string", "pattern": "^[^/\\\\]+$"},
                "arrays_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                "arrays": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(fields),
                    "properties": properties,
                },
            },
        }
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    DESTINATION.mkdir(exist_ok=True)
    for name, document in definitions().items():
        path = DESTINATION / (name + ".json")
        content = json.dumps(document, indent=2, sort_keys=True) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise ValueError(f"Generated schema differs: {path.name}")
        else:
            path.write_text(content)


if __name__ == "__main__":
    main()
