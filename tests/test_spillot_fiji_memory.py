"""Actual ImageJ/Jython export integration; requires SPILLOT_FIJI_HOME.

Run with a normal SpillOT Python environment and JAVA_HOME pointing to a JVM:
  SPILLOT_FIJI_HOME=/path/to/Fiji python -m unittest discover -s tests -p 'test_spillot_fiji_memory.py' -v
"""
from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("_spillot_fiji_fixture.py")


class FijiMemoryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fiji = os.environ.get("SPILLOT_FIJI_HOME")
        if not fiji or not (Path(fiji) / "jars").is_dir():
            raise unittest.SkipTest("Set SPILLOT_FIJI_HOME to test with Fiji's actual ImageJ/Jython jars")
        cls.classpath = os.pathsep.join(str(Path(fiji) / p / "*") for p in ("jars", "plugins"))
        cls.java = (str(Path(os.environ["JAVA_HOME"]) / "bin/java")
                    if os.environ.get("JAVA_HOME") else shutil.which("java"))
        if not cls.java:
            raise unittest.SkipTest("Java is required for Fiji integration tests")
        backend_path = ROOT / "plugins/SpillOT/SpillOT_low_memory.py"
        sys.path.insert(0, str(backend_path.parent))
        spec = importlib.util.spec_from_file_location("SpillOT_low_memory", backend_path)
        cls.backend = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.backend
        spec.loader.exec_module(cls.backend)

    @staticmethod
    def expected(kind):
        plane, yy, xx = np.mgrid[1:13, :9, :11]
        values = (plane * 17 + xx * 7 + yy * 3 + xx * yy) % 239
        if kind == "float32":
            return ((values - 120) * .125).astype("<f4")
        if kind == "uint16":
            return (values * 257).astype("<u2")
        return values.astype("|u1")

    def test_export_and_processing_preserve_stack_and_unselected_planes(self):
        for kind in ("uint8", "uint16", "float32"):
            for virtual in (False, True):
                with self.subTest(dtype=kind, virtual=virtual), tempfile.TemporaryDirectory() as folder:
                    command = [self.java, "-Djava.awt.headless=true", "-cp", self.classpath,
                               "org.python.util.jython", str(FIXTURE), str(ROOT), folder,
                               kind, str(virtual).lower()]
                    run = subprocess.run(command, text=True, capture_output=True, timeout=60)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    fixture = json.loads((Path(folder) / "fixture.json").read_text())
                    manifest = json.loads(Path(fixture["manifest_path"]).read_text())
                    expected = self.expected(kind)
                    raw = np.fromfile(manifest["input_raw"], dtype=manifest["dtype"])
                    np.testing.assert_array_equal(raw.reshape(manifest["shape"]), expected)
                    self.assertEqual(fixture["source_dimensions"], [11, 9, 2, 3, 2])
                    self.assertEqual(fixture["source_info"], "source fixture metadata")
                    self.assertEqual(fixture["source_first_pixel"], float(expected[0, 0, 0]))
                    self.assertEqual(manifest["pixel_size_um"], [.65, .8])
                    self.assertFalse(manifest["write_input_copy"])
                    self.assertFalse(manifest["publish_csv"])
                    with open(manifest["csv_path"], newline="", encoding="utf-8") as stream:
                        matrix = list(csv.reader(stream))
                    self.assertEqual(fixture["selection_rows"], matrix)
                    self.assertEqual(len(matrix), 13)
                    # UI marker2 is originalstackplane3; saved donor selections
                    # from an unselected marker must not become new targets.
                    self.assertEqual(float(matrix[1][3]), 1)
                    self.assertTrue(all(float(v) == 0 for v in matrix[3][1:]))
                    if virtual:
                        self.assertTrue(set(range(1, 13)).issubset(fixture["reads"]))
                    with redirect_stdout(io.StringIO()):
                        result = self.backend.process_manifest(
                            fixture["manifest_path"], tile_size=5, batch_size=7,
                            memory_budget_mb=384, seed=17)
                    self.assertTrue(result["completed"])
                    full = Path(folder) / "original_SpillOT.tif"
                    with tifffile.TiffFile(full) as image:
                        restored = image.asarray().reshape(expected.shape)
                        ome = ET.fromstring(image.ome_metadata)
                        description = ome.find(".//{*}Description")
                        self.assertEqual(json.loads(description.text)["fiji_image_info"],
                                         "source fixture metadata")
                    np.testing.assert_array_equal(restored[1:], expected[1:])
                    self.assertEqual(restored.dtype, expected.dtype)
                    np.testing.assert_array_equal(
                        tifffile.imread(Path(folder) / "original_Channel_5_SpillOT.tif"), expected[4])
                    self.assertEqual(Path(fixture["original_path"]).read_bytes(),
                                     b"original-file-must-not-be-overwritten")
                    reopened_run = subprocess.run(command + ["read"], text=True,
                                                  capture_output=True, timeout=60)
                    self.assertEqual(reopened_run.returncode, 0,
                                     reopened_run.stdout + reopened_run.stderr)
                    reopened = json.loads((Path(folder) / "reopened.json").read_text())
                    self.assertTrue(reopened["virtual"])
                    self.assertEqual(reopened["dimensions"], [11, 9, 2, 3, 2])
                    self.assertEqual(reopened["pixel_width"], .65)
                    self.assertEqual(reopened["pixel_height"], .8)
                    self.assertEqual(reopened["info"], "source fixture metadata")
                    self.assertEqual(reopened["x_origin"], 3.25)
                    self.assertEqual(reopened["y_origin"], 4.5)
                    np.testing.assert_array_equal(np.asarray(reopened["data"]), restored)
                    self.assertTrue(reopened["selected_virtual"])
                    self.assertEqual(reopened["selected_dimensions"], [11, 9, 2, 1, 1])
                    self.assertEqual(reopened["selected_labels"], ["cleaned target", "selected no donor"])
                    np.testing.assert_array_equal(np.asarray(reopened["selected_data"]), restored[[0, 4]])

    def test_rgb_split_is_virtual_and_preserves_channel_frame_order(self):
        with tempfile.TemporaryDirectory() as folder:
            command = [self.java, "-Djava.awt.headless=true", "-cp", self.classpath,
                       "org.python.util.jython", str(FIXTURE), str(ROOT), folder,
                       "uint8", "false", "rgb"]
            run = subprocess.run(command, text=True, capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((Path(folder) / "rgb.json").read_text())
        yy, xx = np.mgrid[:4, :5]
        expected = np.stack([plane for frame in range(2) for plane in
                             (10 + frame + xx, 30 + yy, 50 + xx + yy)])
        np.testing.assert_array_equal(result["data"], expected)
        self.assertTrue(result["virtual"])
        self.assertEqual(result["dimensions"], [5, 4, 3, 1, 2])
        self.assertEqual(result["source_dimensions"], [5, 4, 1, 1, 2])
        self.assertEqual(result["source_first_pixel"] & 0xffffff, (10 << 16) | (30 << 8) | 50)

    def test_actual_fiji_runner_launches_backend_reports_progress_and_cleans_scratch(self):
        with tempfile.TemporaryDirectory() as folder:
            command = [self.java, "-Djava.awt.headless=true", "-cp", self.classpath,
                       "org.python.util.jython", str(FIXTURE), str(ROOT), folder,
                       "uint16", "true", "run", sys.executable]
            run = subprocess.run(command, text=True, capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((Path(folder) / "bridge.json").read_text())
            self.assertTrue(result["report"]["completed"])
            self.assertEqual(Path(result["output"]), Path(folder) / "original_SpillOT.tif")
            self.assertTrue(Path(result["output"]).is_file())
            self.assertEqual(result["progress"][-1][0], 1)
            self.assertTrue(any("tile" in event[1] for event in result["progress"]))
            self.assertEqual(list(Path(folder).glob(".spillot-fiji-*")), [])
            restored = tifffile.imread(result["output"]).reshape(12, 9, 11)
            np.testing.assert_array_equal(restored[1:], self.expected("uint16")[1:])
            self.assertEqual((Path(folder) / "original.tif").read_bytes(),
                             b"original-file-must-not-be-overwritten")


if __name__ == "__main__":
    unittest.main()
