import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
import install_launcher as installer
import run_native_source_batch as native


class Contracts(unittest.TestCase):
    def test_installed_launcher_preserves_arguments_and_explicit_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                sys,
                "argv",
                [
                    "install_launcher",
                    "--binary",
                    sys.executable,
                    "--directory",
                    directory,
                    "--batch-size",
                    "3",
                ],
            ):
                installer.main()
            wrapper = Path(directory) / "kspaceFirstOrder-CUDA"
            with patch.object(
                sys, "argv", [str(wrapper), "-i", "input with spaces.h5"]
            ), patch.object(os, "execv") as execute, patch.dict(os.environ):
                runpy.run_path(str(wrapper), run_name="__main__")
            executable, arguments = execute.call_args.args
            self.assertEqual(executable, sys.executable)
            self.assertEqual(arguments[arguments.index("--batch-size") + 1], "3")
            self.assertEqual(arguments[-2:], ["-i", "input with spaces.h5"])

    def make_input(self, path):
        with h5py.File(path, "w") as h:
            for key, val in {
                "Nx": 8,
                "Ny": 6,
                "Nz": 1,
                "p_source_many": 0,
                "absorbing_flag": 0,
                "rho0": 1000,
                "c0": 1500,
                "p_source_mode": 0,
            }.items():
                h[key] = np.array([[[val]]], dtype=np.uint64)
            h["sensor_mask_index"] = np.array([[[3, 7, 11]]], dtype=np.uint64)
            h["p_source_index"] = np.array([[[3]]], dtype=np.uint64)
            h["p_source_input"] = np.arange(5, dtype=np.float32).reshape(5, 1, 1)
            for key in ("p_source_index", "p_source_input", "sensor_mask_index"):
                h[key].attrs["fixture_role"] = key

    def test_independent_sources_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "a.h5"
            dst = Path(d) / "b.h5"
            self.make_input(src)
            with patch.object(native.h5py, "File", wraps=h5py.File) as opened:
                receipt = native.expand_input(src, dst, 3, source_indices=[11, 3, 7])
            source_opens = [
                call for call in opened.call_args_list if call.args[0] == src
            ]
            self.assertEqual(len(source_opens), 1)
            self.assertEqual(receipt["batch_size"], 3)
            with h5py.File(src) as original:
                original_keys = set(original)
                np.testing.assert_array_equal(
                    original["p_source_index"][...].ravel(), [3]
                )
                np.testing.assert_array_equal(
                    original["p_source_input"][...].ravel(), np.arange(5)
                )
            with h5py.File(dst) as h:
                self.assertEqual(set(h), original_keys)
                for key in ("p_source_index", "p_source_input", "sensor_mask_index"):
                    self.assertEqual(h[key].attrs["fixture_role"], key)
                np.testing.assert_array_equal(
                    h["p_source_index"][...].ravel(), [11, 51, 103]
                )
                np.testing.assert_array_equal(
                    h["sensor_mask_index"][...].ravel(),
                    [3, 7, 11, 51, 55, 59, 99, 103, 107],
                )
                self.assertEqual(h["p_source_input"].shape, (5, 3, 1))

    def test_each_transmitter_uses_its_local_pressure_source_scale(self):
        for mode, exponent in ((0, 2), (1, 1)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                src, dst = Path(directory) / "input.h5", Path(directory) / "batch.h5"
                self.make_input(src)
                speed = np.full(48, 1500.0)
                speed[6], speed[10] = 1550, 1450
                with h5py.File(src, "r+") as handle:
                    del handle["c0"]
                    handle["c0"] = speed.reshape(1, 6, 8)
                    handle["p_source_mode"][...] = mode
                native.expand_input(src, dst, 3, source_indices=[11, 3, 7])
                with h5py.File(dst) as handle:
                    expected = (
                        np.arange(5)[:, None]
                        * (1500 / speed[[10, 2, 6]])[None, :] ** exponent
                    )
                    np.testing.assert_allclose(
                        handle["p_source_input"][..., 0], expected, rtol=1e-7
                    )

    def test_reject_3d_absorption_and_duplicates(self):
        for key, value in [("Nz", 2), ("absorbing_flag", 1)]:
            with tempfile.TemporaryDirectory() as d:
                src = Path(d) / "a.h5"
                self.make_input(src)
                with h5py.File(src, "r+") as h:
                    h[key][...] = value
                with self.assertRaises(ValueError):
                    native.expand_input(src, Path(d) / "b.h5", 3, colocated=True)
                self.assertFalse((Path(d) / "b.h5").exists())
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "a.h5"
            self.make_input(src)
            with self.assertRaises(ValueError):
                native.expand_input(src, Path(d) / "b.h5", 3, source_indices=[3, 3, 7])
            self.assertFalse((Path(d) / "b.h5").exists())

    def test_no_implicit_colocation(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "a.h5", Path(d) / "b.h5"
            self.make_input(src)
            with self.assertRaisesRegex(ValueError, "explicit source_indices"):
                native.expand_input(src, dst, 3)
            self.assertFalse(dst.exists())
            native.expand_input(src, dst, 3, colocated=True)
            with h5py.File(dst) as handle:
                np.testing.assert_array_equal(
                    handle["p_source_index"][...].ravel(), [3, 55, 107]
                )

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "a.h5"
            self.make_input(src)
            with self.assertRaises(ValueError):
                native.expand_input(src, src, 3)


if __name__ == "__main__":
    unittest.main()
