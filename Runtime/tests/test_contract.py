"""MATLAB-free protocol, artifact, subprocess and identity regression tests."""

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Runtime/python"))
import contracts
import process
import wust_runtime as runtime
from wust_io import atomic_json, read_artifact, read_json, sha256, write_artifact

import provenance


def fixture(folder):
    meta = {
        "grid": {
            "shape_yx": [7, 9],
            "origin_yx_m": [0.01, -0.02],
            "spacing_yx_m": [0.001, 0.002],
            "origin_kind": "pixel_edge",
        },
        "fourier_sign": 1,
        "real_pressure": True,
        "pressure_type": "total_pressure",
        "data_units": "instrument_units",
        "spectrum_normalization": "dtft_dt",
        "measurement_provenance": "fixture",
    }
    arrays = {
        "pressure": np.ones((3, 2, 4), complex) * (2 + 3j),
        "mask": np.ones((2, 4), bool),
        "frequencies_hz": np.array([3e5, 1e5, 2e5]),
        "tx_xy_m": np.array([[-0.015, 0.013], [-0.01, 0.014]]),
        "rx_xy_m": np.array(
            [[-0.013, 0.015], [-0.012, 0.012], [-0.009, 0.013], [-0.007, 0.014]]
        ),
    }
    axes = {
        "pressure": "frequency,tx,rx",
        "mask": "tx,rx",
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
    write_artifact(
        folder / "input.json", "wust.frequency_input", meta, arrays, axes, units
    )
    req = {
        "schema": "wust.request",
        "schema_version": 1,
        "operation": "ingest_frequency",
        "input_manifest": str(folder / "input.json"),
        "output_manifest": str(folder / "output.json"),
        "config": {},
    }
    path = folder / "request.json"
    path.write_text(json.dumps(req))
    return path, req


class ContractTests(unittest.TestCase):
    def test_static_discovery_is_machine_clean_without_matlab(self):
        with patch.object(
            runtime, "execute", side_effect=AssertionError("MATLAB launched")
        ):
            info = runtime.capabilities()
        self.assertIsNone(info["environment"])
        self.assertFalse(info["attenuation_reconstruction"]["supported"])
        self.assertFalse(info["budgets"]["update_rtol"])
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "Runtime/python/wust_runtime.py"),
                "describe",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["schema"], "wust.capabilities")

    def test_missing_matlab_probe(self):
        self.assertFalse(runtime.probe("/not/a/matlab")["matlab_available"])

    def test_schema_and_request_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, req = fixture(Path(tmp))
            contracts.validate_request(req)
            for key, value in [
                ("schema", "wfi.request.v1"),
                ("schema_version", True),
                ("schema_version", 2),
                ("operation", "arbitrary_module"),
                ("pipeline_module", "evil"),
            ]:
                bad = copy.deepcopy(req)
                bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    contracts.validate_request(bad)
            bad = copy.deepcopy(req)
            bad["config"]["update_rtol"] = 1e-4
            with self.assertRaises(ValueError):
                contracts.validate_request(bad)
            path.write_text('{"schema":1,"schema":2}')
            with self.assertRaises(ValueError):
                read_json(path)
            path.write_text('{"x":NaN}')
            with self.assertRaises(ValueError):
                read_json(path)

    def test_artifact_roundtrip_hash_and_axis_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder)
            doc, a = read_artifact(folder / "input.json")
            self.assertEqual(a["pressure"].shape, (3, 2, 4))
            self.assertEqual(a["pressure"][1, 1, 1], 2 + 3j)
            self.assertEqual(a["mask"].dtype, bool)
            for key, units in (
                ("frequencies_hz", "kHz"),
                ("tx_xy_m", "mm"),
                ("pressure", "Pa"),
            ):
                bad = copy.deepcopy(doc)
                bad["arrays"][key]["units"] = units
                (folder / "bad.json").write_text(json.dumps(bad))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    read_artifact(folder / "bad.json")
            bad = copy.deepcopy(doc)
            bad["arrays"]["pressure"]["axes"] = "time,rx,tx"
            (folder / "bad.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                read_artifact(folder / "bad.json")
            bad = copy.deepcopy(doc)
            bad["arrays_file"] = "../data.h5"
            (folder / "bad.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                read_artifact(folder / "bad.json")
            with open(folder / doc["arrays_file"], "ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                read_artifact(folder / "input.json")

    def test_nonfinite_mask_and_valid_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder)
            doc, a = read_artifact(folder / "input.json")
            axes = {k: v["axes"] for k, v in doc["arrays"].items()}
            units = {k: v["units"] for k, v in doc["arrays"].items()}
            a["mask"][0, 0] = False
            a["pressure"][:, 0, 0] = np.nan
            a["pressure"][:, 1, 0] = 0
            write_artifact(
                folder / "valid.json", doc["schema"], doc["metadata"], a, axes, units
            )
            _, b = read_artifact(folder / "valid.json")
            self.assertTrue(b["mask"][1, 0])
            self.assertEqual(b["pressure"][0, 1, 0], 0)
            a["mask"][0, 0] = True
            with self.assertRaises(ValueError):
                write_artifact(
                    folder / "invalid.json",
                    doc["schema"],
                    doc["metadata"],
                    a,
                    axes,
                    units,
                )
            self.assertFalse((folder / "invalid.json").exists())

    def test_atomic_output_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.json"
            atomic_json(p, {"first": 1})
            with self.assertRaises(FileExistsError):
                atomic_json(p, {"second": 2})
            self.assertEqual(read_json(p), {"first": 1})

    def test_simulation_units_and_rf_axes_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            arrays = {
                "c_mps": np.full((7, 9), 1500.0),
                "tx_xy_m": np.zeros((2, 2)),
                "rx_xy_m": np.zeros((4, 2)),
                "source_pressure": np.ones(8),
            }
            axes = {
                "c_mps": "y,x",
                "tx_xy_m": "tx,xy",
                "rx_xy_m": "rx,xy",
                "source_pressure": "time",
            }
            units = {
                "c_mps": "m/s",
                "tx_xy_m": "m",
                "rx_xy_m": "m",
                "source_pressure": "Pa",
            }
            write_artifact(
                folder / "simulation.json",
                "wust.simulation_input",
                {},
                arrays,
                axes,
                units,
            )
            rf = {
                "pressure": np.zeros((8, 4, 2)),
                "time_s": np.arange(8) * 1e-7,
                "tx_xy_m": arrays["tx_xy_m"],
                "rx_xy_m": arrays["rx_xy_m"],
            }
            axes = {
                "pressure": "time,rx,tx",
                "time_s": "time",
                "tx_xy_m": "tx,xy",
                "rx_xy_m": "rx,xy",
            }
            units = {"pressure": "Pa", "time_s": "s", "tx_xy_m": "m", "rx_xy_m": "m"}
            write_artifact(
                folder / "rf.json", "wust.rf", {"data_units": "Pa"}, rf, axes, units
            )
            axes["pressure"] = "frequency,tx,rx"
            with self.assertRaises(ValueError):
                write_artifact(
                    folder / "bad.json",
                    "wust.rf",
                    {"data_units": "Pa"},
                    rf,
                    axes,
                    units,
                )

    def test_launcher_admission_failure_and_missing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            p, _ = fixture(folder)
            with patch.object(
                runtime,
                "identity",
                return_value={"dirty": True, "source_manifest_verified": False},
            ), patch.object(runtime, "execute") as call:
                with self.assertRaises(runtime.RuntimeFailure):
                    runtime.run(p, timeout_s=5)
                call.assert_not_called()
            with patch.object(runtime, "execute"), self.assertRaisesRegex(
                runtime.RuntimeFailure, "without completed output"
            ):
                runtime.run(p, allow_dirty=True, timeout_s=5)
            self.assertFalse((folder / "output.json").exists())
            with patch.object(
                runtime, "execute", side_effect=subprocess.TimeoutExpired("matlab", 1)
            ), self.assertRaises(runtime.RuntimeFailure) as raised:
                runtime.run(p, allow_dirty=True, timeout_s=5)
            self.assertEqual(raised.exception.reason, "time_budget")
            self.assertFalse((folder / "output.json").exists())
            with patch.object(
                runtime,
                "execute",
                side_effect=subprocess.CalledProcessError(3, "matlab"),
            ), self.assertRaises(runtime.RuntimeFailure):
                runtime.run(p, allow_dirty=True, timeout_s=5)
            self.assertFalse((folder / "output.json").exists())

    def test_subprocess_nonzero(self):
        with self.assertRaises(subprocess.CalledProcessError):
            process.execute([sys.executable, "-c", "raise SystemExit(3)"], 5)

    @unittest.skipUnless(os.name == "posix", "POSIX process-group supervisor")
    def test_launcher_termination_cleans_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "child"
            child = f"import os,time;from pathlib import Path;Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(60)"
            wrapper = f"import sys;sys.path.insert(0,{str(ROOT/'Runtime/python')!r});from process import execute;execute([sys.executable,'-c',{child!r}],60)"
            p = subprocess.Popen(
                [sys.executable, "-c", wrapper], stderr=subprocess.DEVNULL
            )
            try:
                deadline = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pidfile.exists())
                pid = int(pidfile.read_text())
                p.send_signal(signal.SIGTERM)
                p.wait(timeout=5)
                self.assertNotEqual(p.returncode, 0)
                status = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "stat="],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
                self.assertTrue(not status or status.startswith("Z"))
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait()

    @unittest.skipUnless(os.name == "posix", "POSIX process-group supervisor")
    def test_timeout_terminates_descendant(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "pid"
            grandchild = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"
            parent = f"import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c',{grandchild!r}]);Path({str(pidfile)!r}).write_text(str(p.pid));time.sleep(60)"
            with self.assertRaises(subprocess.TimeoutExpired):
                process.execute([sys.executable, "-c", parent], 0.5)
            pid = int(pidfile.read_text())
            time.sleep(0.1)
            status = subprocess.run(
                ["ps", "-p", str(pid), "-o", "stat="],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            self.assertTrue(
                not status or status.startswith("Z"),
                f"Descendant still running: {status}",
            )

    def test_archive_identity_does_not_fabricate_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Runtime").mkdir()
            (root / "provenance").mkdir()
            (root / "Runtime/VERSION").write_text("1.2.3")
            (root / "provenance/upstream.json").write_text("{}")
            manifest = root / "provenance/source-manifest.json"
            manifest.write_text(
                json.dumps(
                    {"sha256": {"Runtime/VERSION": sha256(root / "Runtime/VERSION")}}
                )
            )
            with patch.object(provenance, "ROOT", root):
                info = provenance.identity()
            self.assertIsNone(info["git_sha"])
            self.assertIsNone(info["dirty"])
            self.assertTrue(info["source_manifest_verified"])
            build = {
                "schema": "wust.build",
                "schema_version": 1,
                "source_manifest_sha256": sha256(manifest),
                "git_sha": None,
            }
            (root / "provenance/build-manifest.json").write_text(json.dumps(build))
            with patch.object(provenance, "ROOT", root):
                self.assertEqual(
                    provenance.identity()["identity_source"], "build_manifest"
                )
            build["source_manifest_sha256"] = "wrong"
            (root / "provenance/build-manifest.json").write_text(json.dumps(build))
            with patch.object(provenance, "ROOT", root), self.assertRaises(ValueError):
                provenance.identity()


if __name__ == "__main__":
    unittest.main()
