"""Structural contracts for naming, provenance and the external launcher."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Runtime/python"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load("wust_runtime", ROOT / "Runtime/python/wust_runtime.py")
manifest = load("manifest", ROOT / "provenance/check_manifest.py")


class MainlineTests(unittest.TestCase):
    def test_version_has_one_authority(self):
        self.assertEqual(
            runtime.VERSION, (ROOT / "Runtime/VERSION").read_text().strip()
        )
        matlab = (ROOT / "Runtime/matlab/wust_run.m").read_text()
        self.assertIn("'runtime_version',wust_version", matlab)
        for path in (ROOT / "Runtime/matlab").glob("*.m"):
            self.assertNotIn("'" + runtime.VERSION + "'", path.read_text())

    def test_production_names_and_paths(self):
        for directory in ("matlab", "solver", "python"):
            for path in (ROOT / "Runtime" / directory).iterdir():
                if path.suffix not in {".m", ".py"}:
                    continue
                source = path.read_text()
                self.assertNotIn("wfi_", source)
                self.assertNotIn("wfiUseGPU", source)
                self.assertNotIn("genpath(", source)
                self.assertNotIn("reference/", source)
        setup = (ROOT / "Runtime/matlab/wust_setup.m").read_text()
        self.assertIn("WUST:PathConflict", setup)

    def test_source_manifest(self):
        self.assertGreater(manifest.check(), 0)

    def test_launcher_rejects_historical_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            input_path = folder / "input.mat"
            input_path.touch()
            output = folder / "output.mat"
            request = folder / "request.json"
            req = {
                "schema": "wfi.request.v1",
                "operation": "reconstruct",
                "input_mat": str(input_path),
                "output_mat": str(output),
                "config": {},
            }
            request.write_text(json.dumps(req))
            with patch.object(runtime, "execute") as call, self.assertRaisesRegex(
                ValueError, "Legacy"
            ):
                runtime.run(request)
            call.assert_not_called()
            req["schema"] = "wust.request"
            request.write_text(json.dumps(req))
            with self.assertRaises(ValueError):
                runtime.run(request)


if __name__ == "__main__":
    unittest.main()
