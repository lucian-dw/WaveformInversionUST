"""Independent WUST runtime: validate transport, execute once, publish arrays."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from contracts import (
    INPUTS,
    PROTOCOL_VERSION,
    SCHEMAS,
    number,
    validate_request,
)
from process import execute
from wust_io import atomic_json, read_json, read_manifest

ROOT = Path(__file__).resolve().parents[2]
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
        "schema_version": PROTOCOL_VERSION,
        "runtime_name": "WUST",
        "runtime_version": VERSION,
        "schemas": SCHEMAS,
        "operations": list(INPUTS),
        "input_schemas": INPUTS,
        "sound_speed_reconstruction": {"supported": True, "dimensions": [2]},
        "attenuation_reconstruction": {"supported": False},
        "profiles": {"cpu": "cpu_float64", "gpu": "gpu_complex64"},
        "execution_platforms": ["Linux", "Darwin"],
        "iteration_unit": "frequency_schedule_update",
        "budgets": {
            "max_updates": True,
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
    }


def probe(matlab="matlab", timeout_s=60, backend="cpu"):
    if backend not in {"cpu", "gpu"}:
        raise ValueError("Probe backend must be cpu or gpu")
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
        expression = f"s=settings; addpath({quote(ROOT/'Runtime/matlab')}); wust_probe({quote(output)},{quote(backend)});"
        try:
            execute([executable, "-batch", expression], timeout_s)
            return read_json(output)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            return {
                "matlab_available": False,
                "cpu_available": False,
                "gpu_available": False,
                "failure_reason": str(error),
            }


def run(request_path, matlab="matlab", timeout_s=None, kwave_toolbox_path=None):
    started = time.monotonic()
    req = validate_request(read_json(Path(request_path).resolve(strict=True)))
    number(timeout_s, "timeout_s")
    if timeout_s is None:
        raise ValueError("An explicit timeout_s is required")
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
    read_manifest(req["input_manifest"], INPUTS[req["operation"]])
    if req["operation"] == "reconstruct":
        read_manifest(req["initial_manifest"], "wust.initial_model")
    if kwave_toolbox_path and req["operation"] != "simulate":
        raise ValueError(
            "k-Wave toolbox is simulation tooling, not a reconstruction dependency"
        )

    # Keep the request and partial arrays on failure for diagnosis. Observation
    # paths remain stable; MATLAB is the only consumer loading their values.
    folder = Path(tempfile.mkdtemp(prefix=f"{output.stem}-run-", dir=output.parent))
    wire = dict(req, output_manifest=str(folder / "result.json"))
    if wire.get("frequency_schedule_hz") is None:
        wire.pop("frequency_schedule_hz", None)
    internal = folder / "request.json"
    atomic_json(internal, wire)
    expression = f"s=settings; addpath({quote(ROOT/'Runtime/matlab')});"
    if kwave_toolbox_path:
        expression += (
            f"addpath({quote(Path(kwave_toolbox_path).resolve(strict=True))});"
        )
    expression += f"wust_run({quote(internal)});"
    remaining = timeout_s - (time.monotonic() - started)
    if remaining <= 0:
        raise RuntimeFailure("time_budget", "Timeout before MATLAB startup")
    try:
        execute([str(matlab), "-batch", expression], remaining)
    except subprocess.TimeoutExpired as error:
        raise RuntimeFailure(
            "time_budget", "Deadline expired; MATLAB process group terminated"
        ) from error
    except (OSError, subprocess.CalledProcessError) as error:
        failure = folder / "failure.json"
        detail = read_json(failure) if failure.exists() else {}
        reason = (
            "numerical_failure"
            if detail.get("identifier") == "WUST:NumericalFailure"
            else "runtime_failure"
        )
        raise RuntimeFailure(reason, detail.get("message", str(error))) from error
    result_path = Path(wire["output_manifest"])
    if not result_path.is_file():
        raise RuntimeFailure(
            "incomplete_output", "MATLAB exited without completed output"
        )
    expected = {"reconstruct": "wust.reconstruction", "simulate": "wust.rf"}.get(
        req["operation"], "wust.measurements"
    )
    result = read_manifest(result_path, expected)
    result["metadata"].update(
        runtime_version=VERSION,
        request_timeout_s=timeout_s,
        process_wall_seconds=time.monotonic() - started,
    )
    if time.monotonic() - started >= timeout_s:
        raise RuntimeFailure("time_budget", "Deadline expired before final publication")
    array_file = folder / result["arrays_file"]
    destination = output.parent / f"{folder.name}.h5"
    os.replace(array_file, destination)
    result["arrays_file"] = destination.name
    atomic_json(output, result)
    result_path.unlink()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    describe = commands.add_parser("describe")
    describe.add_argument("--json", action="store_true")
    describe.add_argument("--probe", action="store_true")
    describe.add_argument("--matlab", default="matlab")
    describe.add_argument("--timeout-s", type=float, default=60)
    describe.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("request")
    run_parser.add_argument("--matlab", default="matlab")
    run_parser.add_argument("--timeout-s", type=float, required=True)
    run_parser.add_argument("--kwave-toolbox-path")
    args = parser.parse_args()
    try:
        if args.command == "describe":
            result = capabilities()
            if args.probe:
                result["environment"] = probe(args.matlab, args.timeout_s, args.backend)
        else:
            result = run(
                args.request, args.matlab, args.timeout_s, args.kwave_toolbox_path
            )
        print(json.dumps(result, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "schema": "wust.execution_failure",
                    "schema_version": PROTOCOL_VERSION,
                    "reason": getattr(error, "reason", "incompatible_request"),
                    "message": str(error),
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
