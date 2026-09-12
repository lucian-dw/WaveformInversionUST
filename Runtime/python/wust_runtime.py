"""Independent versioned WUST runtime; static discovery never starts MATLAB."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from contracts import INPUTS, SCHEMAS, number, schema, validate_request
from process import execute
from wust_io import atomic_json, read_artifact, read_json, sha256

from provenance import ROOT, identity

VERSION = (ROOT / "Runtime/VERSION").read_text().strip()


class RuntimeFailure(RuntimeError):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def capabilities():
    return {
        "schema": "wust.capabilities",
        "schema_version": 1,
        "runtime_name": "WUST",
        "runtime_version": VERSION,
        "schemas": SCHEMAS,
        "operations": list(INPUTS),
        "sound_speed_reconstruction": {"supported": True, "dimensions": [2]},
        "attenuation_reconstruction": {"supported": False},
        "input_domains": {
            "reconstruct": "canonical_complex_total_pressure",
            "ingest_frequency": "complex_total_pressure",
            "prepare": "time_domain_pressure",
            "simulate": "physical_fixture",
        },
        "profiles": {"cpu": "cpu_float64", "gpu": "gpu_complex64"},
        "execution_platforms": ["Linux", "Darwin"],
        "iteration_unit": "frequency_schedule_update",
        "budgets": {
            "resolved_schedule": True,
            "hard_timeout_s": True,
            "forward_call_cap": False,
            "adjoint_call_cap": False,
            "update_rtol": False,
            "online_convergence": False,
        },
        "source_scale": {
            "method": "per_tx_per_frequency_complex_least_squares",
            "minimum_fitting_receivers": 2,
            "requires_water_reference": False,
            "requires_source_spectrum": False,
        },
        "fourier_sign": -1,
        "provenance": identity(),
        "environment": None,
    }


def probe(matlab="matlab", timeout_s=60):
    number(timeout_s, "timeout_s")
    executable = shutil.which(matlab)
    if not executable:
        return {
            "matlab_available": False,
            "cpu_available": False,
            "gpu_available": False,
            "failure_reason": "MATLAB executable not found",
        }
    with tempfile.TemporaryDirectory(prefix="wust-probe-") as folder:
        output = Path(folder) / "probe.json"
        expression = (
            f"addpath({quote(ROOT/'Runtime/matlab')}); wust_probe({quote(output)});"
        )
        try:
            execute([executable, "-batch", expression], timeout_s)
            result = read_json(output)
            for mex in result.get("mex", []):
                if Path(mex["path"]).is_file():
                    mex["sha256"] = sha256(mex["path"])
            return result
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            return {
                "matlab_available": False,
                "cpu_available": False,
                "gpu_available": False,
                "failure_reason": str(error),
            }


def run(
    request_path,
    matlab="matlab",
    timeout_s=None,
    allow_dirty=False,
    kwave_toolbox_path=None,
):
    started = time.monotonic()
    request_path = Path(request_path).resolve(strict=True)
    req = validate_request(read_json(request_path))
    number(timeout_s, "timeout_s")
    if type(allow_dirty) is not bool:
        raise ValueError("allow_dirty must be an explicit boolean")
    for key in ("input_manifest", "output_manifest", "initial_manifest"):
        if key in req and (
            not isinstance(req[key], str) or not Path(req[key]).is_absolute()
        ):
            raise ValueError(f"{key} must be an absolute path")
    output = Path(req["output_manifest"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise ValueError("Output parent must already exist")
    doc, input_arrays = read_artifact(req["input_manifest"], INPUTS[req["operation"]])
    hashes = {
        "input_manifest_sha256": sha256(req["input_manifest"]),
        "input_arrays_sha256": doc["arrays_sha256"],
        "request_sha256": sha256(request_path),
    }
    from wust_io import array_hashes

    hashes["input_array_hashes"] = array_hashes(input_arrays)
    if req["operation"] == "reconstruct":
        init, initial_arrays = read_artifact(
            req["initial_manifest"], "wust.initial_model"
        )
        hashes.update(
            initial_manifest_sha256=sha256(req["initial_manifest"]),
            initial_arrays_sha256=init["arrays_sha256"],
        )
        hashes["initial_array_hashes"] = array_hashes(initial_arrays)
    info = identity()
    if not allow_dirty and (
        info["dirty"] is True or not info["source_manifest_verified"]
    ):
        raise RuntimeFailure(
            "incompatible_runtime",
            "Dirty/unverified runtime; commit/verify or explicitly use --allow-dirty for research",
        )
    info["allow_dirty_override"] = bool(allow_dirty)
    if kwave_toolbox_path and req["operation"] != "simulate":
        raise ValueError(
            "k-Wave toolbox is simulation tooling, not a reconstruction dependency"
        )
    with tempfile.TemporaryDirectory(prefix=".wust-", dir=output.parent) as folder:
        folder = Path(folder)
        wire = dict(req)

        def snapshot(original, descriptor, label):
            target = folder / label
            target.mkdir()
            artifact = Path(original).parent / descriptor["arrays_file"]
            shutil.copyfile(artifact, target / descriptor["arrays_file"])
            if (
                sha256(target / descriptor["arrays_file"])
                != descriptor["arrays_sha256"]
            ):
                raise ValueError("Input arrays changed during snapshot")
            saved = target / "manifest.json"
            saved.write_text(json.dumps(descriptor, allow_nan=False))
            return str(saved)

        wire["input_manifest"] = snapshot(req["input_manifest"], doc, "input")
        if req["operation"] == "reconstruct":
            wire["initial_manifest"] = snapshot(
                req["initial_manifest"], init, "initial"
            )
        wire["output_manifest"] = str(folder / "result.json")
        internal = folder / "request.json"
        internal.write_text(json.dumps(wire, allow_nan=False))
        expression = f"addpath({quote(ROOT/'Runtime/matlab')});"
        if kwave_toolbox_path:
            expression += (
                f"addpath({quote(Path(kwave_toolbox_path).resolve(strict=True))});"
            )
        expression += f"wust_run({quote(internal)});"
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeFailure("time_budget", "Timeout during admission")
        try:
            execute([str(matlab), "-batch", expression], remaining)
        except subprocess.TimeoutExpired as error:
            raise RuntimeFailure(
                "time_budget",
                "Execution deadline expired; process group terminated; no final result published",
            ) from error
        except (OSError, subprocess.CalledProcessError) as error:
            detail = (
                read_json(wire["output_manifest"])
                if Path(wire["output_manifest"]).exists()
                else {}
            )
            reason = (
                "numerical_failure"
                if detail.get("identifier") == "WUST:NumericalFailure"
                else "runtime_failure"
            )
            raise RuntimeFailure(reason, detail.get("message", str(error))) from error
        if not Path(wire["output_manifest"]).is_file():
            raise RuntimeFailure(
                "incomplete_output", "MATLAB exited without completed output"
            )
        result = read_json(wire["output_manifest"])
        expected = {"reconstruct": "wust.reconstruction", "simulate": "wust.rf"}.get(
            req["operation"], "wust.measurements"
        )
        schema(result, expected)
        name = result["arrays_file"]
        if Path(name).name != name:
            raise RuntimeFailure("incomplete_output", "Invalid output array path")
        array_file = folder / name
        result["arrays_sha256"] = sha256(array_file)
        result["metadata"].update(
            runtime_provenance=info,
            input_hashes=hashes,
            request_timeout_s=timeout_s,
            process_wall_seconds=time.monotonic() - started,
        )
        environment = result["metadata"].get("environment", {})
        if isinstance(environment.get("mex"), dict):
            environment["mex_sha256"] = {
                k: sha256(v) for k, v in environment["mex"].items() if Path(v).is_file()
            }
        if expected == "wust.reconstruction":
            result["metadata"]["final_data_residual"] = None
        Path(wire["output_manifest"]).write_text(json.dumps(result, allow_nan=False))
        read_artifact(wire["output_manifest"], expected)
        if time.monotonic() - started >= timeout_s:
            raise RuntimeFailure("time_budget", "Deadline expired before finalization")
        destination = output.parent / f"{output.stem}.{uuid.uuid4().hex}.h5"
        os.link(array_file, destination)
        result["arrays_file"] = destination.name
        try:
            atomic_json(output, result)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    describe = commands.add_parser("describe")
    describe.add_argument("--json", action="store_true")
    describe.add_argument("--probe", action="store_true")
    describe.add_argument("--matlab", default="matlab")
    describe.add_argument("--timeout-s", type=float, default=60)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("request")
    run_parser.add_argument("--matlab", default="matlab")
    run_parser.add_argument("--timeout-s", type=float, required=True)
    run_parser.add_argument("--allow-dirty", action="store_true")
    run_parser.add_argument("--kwave-toolbox-path")
    args = parser.parse_args()
    try:
        if args.command == "describe":
            result = capabilities()
            if args.probe:
                result["environment"] = probe(args.matlab, args.timeout_s)
        else:
            result = run(
                args.request,
                args.matlab,
                args.timeout_s,
                args.allow_dirty,
                args.kwave_toolbox_path,
            )
        print(json.dumps(result, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "schema": "wust.execution_failure",
                    "schema_version": 1,
                    "reason": getattr(error, "reason", "incompatible_request"),
                    "message": str(error),
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
