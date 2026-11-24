"""Small deterministic regression tests for the tiled SpillOT backend.

Run with the same Python environment as SpillOT:
    python -m unittest discover -s tests -p 'test_spillot_low_memory.py' -v

The numerical oracle executes the actual legacy backend with only its RNG
entry seeded. These fixtures check implementation parity, not biological
accuracy or universal bitwise equivalence for every floating-point input.
"""

from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "plugins/SpillOT/SpillOT_low_memory.py"
REFERENCE = Path(__file__).with_name("_spillot_stock_reference.py")
PUBLICATION_FAULT = Path(__file__).with_name("_spillot_publication_fault.py")
CHANNEL_NAMES = ["CD8 target", "CD99 donor", "Unchanged marker"]
SEED = 17


def progress_events(stdout):
    prefix = "SPILLOT_PROGRESS\t"
    return [json.loads(line[len(prefix):]) for line in stdout.splitlines()
            if line.startswith(prefix)]


def load_backend():
    # Use an importable name so spawn workers can resolve their initializer.
    if str(BACKEND.parent) not in sys.path:
        sys.path.insert(0, str(BACKEND.parent))
    name = "SpillOT_low_memory"
    spec = importlib.util.spec_from_file_location(name, BACKEND)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fixture(height, width, floating=False, saturation=False):
    """Structured spillover plus dropout and a separate target-only object."""
    rng = np.random.default_rng(918)
    yy, xx = np.mgrid[:height, :width]
    donor = (240 * np.exp(-((yy-height*.35)**2 + (xx-width*.30)**2) / 23)
             + 160 * np.exp(-((yy-height*.65)**2 + (xx-width*.72)**2) / 18)
             + rng.poisson(3, (height, width)))
    target = donor * .24 + rng.poisson(1, (height, width))
    target[rng.random((height, width)) < .38] = 0
    target[2:5, width-6:width-3] += 70
    stack = np.stack([target, donor, rng.integers(0, 90, (height, width))])
    if floating:
        stack = (stack * .123 - np.array([2.5, 7.25, 1.75])[:, None, None]).astype("<f4")
    else:
        stack = np.rint(stack).astype("<u2")
    if saturation:
        # Straddle the tile-size=5 boundary, including a source-only saturation.
        stack[1, 4, 4] = 65535
        stack[1, 5, 5] = 65535
        stack[0, 4, 5] = 65535
        stack[0, height-1, width-1] = 65535
    return stack


class TiledSpillOTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = load_backend()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="spillot-low-memory-test-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def make_manifest(self, stack, patch=4, saturation=False, name="result"):
        raw = self.directory / "input.raw"
        np.ascontiguousarray(stack).tofile(raw)
        matrix = self.directory / "manual.csv"
        with matrix.open("w", newline="") as stream:
            csv.writer(stream).writerows([
                ["", *CHANNEL_NAMES],
                [CHANNEL_NAMES[0], 1, -1, 0],  # diagonal must be ignored
                [CHANNEL_NAMES[1], 0, 0, 0],
                [CHANNEL_NAMES[2], 0, 0, 0],
            ])
        manifest = {
            "input_raw": str(raw), "shape": list(stack.shape), "dtype": stack.dtype.str,
            "output_dir": str(self.directory / name), "base_name": "roi",
            "csv_path": str(matrix), "channel_names": CHANNEL_NAMES,
            "pixel_size_um": [0.65, 0.8], "origin_xy": [100, 200],
            "patch_size": patch, "workers": 1, "ignore_overexposed": saturation,
            "memory_budget_mb": 384,
        }
        path = self.directory / (name + ".json")
        path.write_text(json.dumps(manifest))
        return path, manifest

    def run_backend(self, path, **overrides):
        output = io.StringIO()
        with redirect_stdout(output):
            result = self.backend.process_manifest(path, seed=SEED, **overrides)
        return result, output.getvalue()

    def stock(self, stack, manifest):
        image = self.directory / "stock_input.tif"
        tifffile.imwrite(image, stack, imagej=True)
        command = [sys.executable, str(REFERENCE), str(image), "--patch-size",
                   str(manifest["patch_size"]), "--seed", str(SEED),
                   "--csv", manifest["csv_path"]]
        if manifest["ignore_overexposed"]:
            command.append("--ignore-overexposed")
        run = subprocess.run(command, text=True, capture_output=True, timeout=90)
        self.assertEqual(run.returncode, 0, run.stdout + "\n" + run.stderr)
        return tifffile.imread(image.with_name("stock_input_SpillOT.tif"))

    def read_output(self, manifest):
        return tifffile.imread(Path(manifest["output_dir"]) / "roi_SpillOT.tif")

    @unittest.skipIf(os.name == "nt", "The unchanged stock reference uses fork globals")
    def test_tiled_uint16_matches_actual_stock_at_edges_and_corners(self):
        stack = fixture(13, 17)
        path, manifest = self.make_manifest(stack)
        expected = self.stock(stack, manifest)
        self.run_backend(path, tile_size=5, batch_size=7)
        actual = self.read_output(manifest)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[1:], stack[1:])
        self.assertGreater(np.count_nonzero(actual[0] != stack[0]), 0,
                           "The parity fixture must exercise actual removal")
        self.assertEqual(actual.dtype, stack.dtype)
        self.assertEqual(actual.shape, stack.shape)

    @unittest.skipIf(os.name == "nt", "The unchanged stock reference uses fork globals")
    def test_negative_float32_matches_stock_and_single_spatial_tile(self):
        stack = fixture(12, 15, floating=True)
        path, manifest = self.make_manifest(stack, name="tiled")
        expected = self.stock(stack, manifest)
        self.run_backend(path, tile_size=5, batch_size=8)
        actual = self.read_output(manifest)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[1:], stack[1:])
        single_path, single = self.make_manifest(stack, name="single")
        self.run_backend(single_path, tile_size=100, batch_size=8)
        np.testing.assert_array_equal(actual, self.read_output(single))
        self.assertEqual(actual.dtype, np.dtype("float32"))

    @unittest.skipIf(os.name == "nt", "The unchanged stock reference uses fork globals")
    def test_saturation_across_tile_boundary_matches_global_stock_inpaint(self):
        stack = fixture(13, 17, saturation=True)
        path, manifest = self.make_manifest(stack, saturation=True)
        expected = self.stock(stack, manifest)
        self.run_backend(path, tile_size=5, batch_size=7)
        actual = self.read_output(manifest)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[1:], stack[1:])

    @unittest.skipIf(os.name == "nt", "The unchanged stock reference uses fork globals")
    def test_default_patch24_has_correct_halos_on_tiny_edge_tiles(self):
        stack = fixture(27, 29, floating=True)
        path, manifest = self.make_manifest(stack, patch=24)
        expected = self.stock(stack, manifest)
        self.run_backend(path, tile_size=26, batch_size=7)
        np.testing.assert_array_equal(self.read_output(manifest), expected)

    def test_singleton_batches_preserve_decisions(self):
        stack = fixture(9, 11)
        path, manifest = self.make_manifest(stack, name="singleton")
        self.run_backend(path, tile_size=5, batch_size=1)
        many_path, many = self.make_manifest(stack, name="many")
        self.run_backend(many_path, tile_size=5, batch_size=16)
        np.testing.assert_array_equal(self.read_output(manifest), self.read_output(many))

    def test_output_retains_channels_calibration_and_input(self):
        stack = fixture(9, 11)
        path, manifest = self.make_manifest(stack)
        report, stdout = self.run_backend(path, tile_size=5, batch_size=8)
        out = Path(manifest["output_dir"])
        np.testing.assert_array_equal(tifffile.imread(out / "roi.tif"), stack)
        result = self.read_output(manifest)
        np.testing.assert_array_equal(tifffile.imread(out / "roi_Channel_1_SpillOT.tif"), result[0])
        self.assertFalse((out / "roi_Channel_2_SpillOT.tif").exists())
        self.assertFalse((out / "roi_Channel_3_SpillOT.tif").exists())
        with tifffile.TiffFile(out / "roi_SpillOT.tif") as image:
            self.assertEqual(image.series[0].shape, stack.shape)
            metadata = ET.fromstring(image.ome_metadata)
        channels = metadata.findall(".//{*}Channel")
        self.assertEqual([channel.attrib["Name"] for channel in channels], CHANNEL_NAMES)
        pixels = metadata.find(".//{*}Pixels")
        self.assertAlmostEqual(float(pixels.attrib["PhysicalSizeX"]), 0.65)
        self.assertAlmostEqual(float(pixels.attrib["PhysicalSizeY"]), 0.8)
        self.assertEqual(pixels.attrib["PhysicalSizeXUnit"], "µm")
        self.assertEqual(pixels.attrib["PhysicalSizeYUnit"], "µm")
        description = json.loads(metadata.find(".//{*}Description").text)
        self.assertEqual(description["origin_xy"], manifest["origin_xy"])
        planes = metadata.findall(".//{*}Plane")
        self.assertTrue(planes)
        for plane in planes:
            self.assertAlmostEqual(float(plane.attrib["PositionX"]), 65.0)
            self.assertAlmostEqual(float(plane.attrib["PositionY"]), 160.0)
        self.assertTrue(report["completed"])
        self.assertTrue(all(Path(name).is_file() for name in report["output_files"]))
        events = progress_events(stdout)
        self.assertGreater(len(events), 2)
        fractions = [event["fraction"] for event in events]
        self.assertEqual(fractions, sorted(fractions))
        self.assertTrue(all(0 <= value <= 1 for value in fractions))
        self.assertEqual(fractions[-1], 1)
        self.assertTrue(all(value < 1 for value in fractions[:-1]))
        self.assertIsNone(events[0]["eta_seconds"], "ETA needs a warmup observation")

    def test_full_input_is_never_materialized_by_asarray(self):
        stack = fixture(13, 17)
        path, manifest = self.make_manifest(stack)
        original = np.asarray

        def bounded_asarray(value, *args, **kwargs):
            if isinstance(value, np.memmap) and value.shape == stack.shape:
                raise AssertionError("Whole input memmap was passed to np.asarray")
            return original(value, *args, **kwargs)

        with mock.patch.object(np, "asarray", side_effect=bounded_asarray):
            self.run_backend(path, tile_size=5, batch_size=8)
        self.assertEqual(self.read_output(manifest).shape, stack.shape)

    def previous_outputs(self, manifest):
        directory = Path(manifest["output_dir"])
        directory.mkdir()
        previous = {}
        for name in ("roi.tif", "roi_Channel_1_SpillOT.tif", "roi_SpillOT.tif"):
            path = directory / name
            tifffile.imwrite(path, np.full((3, 4), 123, dtype=np.uint16))
            previous[path] = path.read_bytes()
        csv_path = directory / "roi.csv"
        csv_path.write_text("previous selection matrix\n")
        previous[csv_path] = csv_path.read_bytes()
        return previous

    def assert_previous_outputs_intact(self, manifest, previous):
        directory = Path(manifest["output_dir"])
        self.assertEqual(set(directory.iterdir()), set(previous),
                         "Failure left staged data or published unexpected files")
        for path, expected in previous.items():
            self.assertEqual(path.read_bytes(), expected)
        self.assertEqual(list(self.directory.rglob(".spillot-tmp-*")), [])

    def _assert_cancellation_preserves_previous(self, use_marker=False):
        stack = fixture(13, 17)
        path, manifest = self.make_manifest(stack)
        manifest["scratch_dir"] = str(self.directory / "private-work")
        path.write_text(json.dumps(manifest))
        previous = self.previous_outputs(manifest)
        event = threading.Event()
        marker = Path(manifest["scratch_dir"]) / "cancel.requested"

        class CancelAfterProgress(io.StringIO):
            def write(self, text):
                result = super().write(text)
                for progress in progress_events(self.getvalue()):
                    if 0.05 < progress["fraction"] < 1:
                        if use_marker:
                            marker.touch()
                        else:
                            event.set()
                return result

        stdout = CancelAfterProgress()
        with redirect_stdout(stdout):
            with self.assertRaises(self.backend.CooperativeCancel):
                self.backend.process_manifest(path, seed=SEED, tile_size=5,
                                              batch_size=7, cancel_event=event)
        self.assertTrue(marker.exists() if use_marker else event.is_set(),
                        "Cancellation must happen after measured work")
        self.assert_previous_outputs_intact(manifest, previous)
        self.assertTrue(all(e["fraction"] < 1 for e in progress_events(stdout.getvalue())))

    def test_cancellation_cleans_staging_and_preserves_previous_results(self):
        self._assert_cancellation_preserves_previous()

    def test_filesystem_cancel_marker_stops_work_and_preserves_previous_results(self):
        self._assert_cancellation_preserves_previous(use_marker=True)

    def test_truncated_raw_input_does_not_publish_partial_success(self):
        stack = fixture(9, 11)
        path, manifest = self.make_manifest(stack)
        raw = Path(manifest["input_raw"])
        raw.write_bytes(raw.read_bytes()[:-1])
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with self.assertRaises((ValueError, OSError)):
                self.backend.process_manifest(path, seed=SEED, tile_size=5, batch_size=7)
        directory = Path(manifest["output_dir"])
        self.assertFalse(directory.exists() and any(directory.iterdir()))
        self.assertTrue(all(e["fraction"] < 1 for e in progress_events(stdout.getvalue())))
        self.assertEqual(list(self.directory.rglob(".spillot-tmp-*")), [])

    def test_failed_publication_rolls_back_all_previous_outputs(self):
        stack = fixture(9, 11)
        path, manifest = self.make_manifest(stack)
        previous = self.previous_outputs(manifest)
        last_destination = Path(manifest["output_dir"]) / "roi.csv"
        real_replace = os.replace
        injected = False

        def fail_final_csv(source, destination, *args, **kwargs):
            nonlocal injected
            if Path(destination) == last_destination and not injected:
                injected = True
                raise OSError("simulated publication failure")
            return real_replace(source, destination, *args, **kwargs)

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with mock.patch.object(os, "replace", side_effect=fail_final_csv):
                with self.assertRaisesRegex(OSError, "simulated publication failure"):
                    self.backend.process_manifest(path, seed=SEED, tile_size=5, batch_size=7)
        self.assertTrue(injected)
        self.assert_previous_outputs_intact(manifest, previous)
        self.assertTrue(all(e["fraction"] < 1 for e in progress_events(stdout.getvalue())))

    def test_two_spawn_workers_match_single_worker(self):
        stack = fixture(9, 11)
        serial_path, serial = self.make_manifest(stack, name="serial")
        self.run_backend(serial_path, tile_size=5, batch_size=8)
        parallel_path, parallel = self.make_manifest(stack, name="parallel")
        parallel["workers"] = 2
        parallel["memory_budget_mb"] = 1024
        parallel_path.write_text(json.dumps(parallel))
        report, _ = self.run_backend(parallel_path, tile_size=5, batch_size=8)
        self.assertEqual(report["plan"]["workers"], 2)
        np.testing.assert_array_equal(self.read_output(parallel), self.read_output(serial))

    def test_cancellation_after_successful_commit_still_reports_complete(self):
        stack = fixture(9, 11)
        path, manifest = self.make_manifest(stack)
        event = threading.Event()
        publish = self.backend.publish_outputs

        def publish_then_cancel(*args, **kwargs):
            paths = publish(*args, **kwargs)
            event.set()
            return paths

        stdout = io.StringIO()
        with mock.patch.object(self.backend, "publish_outputs", side_effect=publish_then_cancel):
            with redirect_stdout(stdout):
                report = self.backend.process_manifest(
                    path, seed=SEED, tile_size=5, batch_size=7, cancel_event=event)
        self.assertTrue(event.is_set())
        self.assertTrue(report["completed"])
        self.assertEqual(progress_events(stdout.getvalue())[-1]["fraction"], 1)
        self.assertEqual(self.read_output(manifest).shape, stack.shape)

    def test_bounded_inpaint_expands_beyond_local_halo(self):
        shape = (1, 45, 49)
        data = np.zeros(shape[1:], dtype=np.float32)
        # All possible donors lie outside a radius-16 neighborhood of (21,23).
        data[20, 2], data[2, 23], data[42, 23], data[44, 48] = 11, 19, 29, 99
        saturation = np.zeros(shape[1:], dtype=np.uint8)
        saturation[21, 23] = 1
        saturation[0, 0] = 1
        result = self.backend.RawArray(self.directory / "result.raw", shape, "<f4", create=True)
        mask = self.backend.RawArray(self.directory / "sat.raw", shape, "|u1", create=True)
        self.addCleanup(result.close)
        self.addCleanup(mask.close)
        result.write(0, 0, 0, data)
        mask.write(0, 0, 0, saturation)
        count = self.backend.bounded_global_inpaint(result, mask, lambda: None, k=3, block=7)
        expected = data.copy()
        expected[21, 23] = 19  # Median of the three globally nearest donors.
        expected[0, 0] = 19
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(result.read(0, 0, 45, 0, 49), expected)

    @unittest.skipIf(os.name == "nt", "POSIX SIGTERM integration test")
    def test_cli_sigterm_cleans_staging_and_reports_cancellation(self):
        stack = fixture(41, 43)
        path, manifest = self.make_manifest(stack, patch=8)
        manifest["workers"] = 2
        manifest["memory_budget_mb"] = 1024
        path.write_text(json.dumps(manifest))
        process = subprocess.Popen(
            [sys.executable, str(BACKEND), "--manifest", str(path), "--tile-size", "11",
             "--batch-size", "8", "--seed", str(SEED)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        lines, incoming = [], queue.Queue()

        def drain():
            for line in process.stdout:
                lines.append(line)
                incoming.put(line)
            incoming.put(None)

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            deadline = time.monotonic() + 30
            while True:
                remaining = deadline - time.monotonic()
                self.assertGreater(remaining, 0, "Backend never reached processing")
                try:
                    line = incoming.get(timeout=remaining)
                except queue.Empty:
                    self.fail("Backend never reached processing")
                self.assertIsNotNone(line, "Backend exited before cancellation: " + "".join(lines))
                events = progress_events(line)
                if events and events[0]["phase"] == "processing":
                    process.send_signal(signal.SIGTERM)
                    break
            process.wait(timeout=15)
            reader.join(timeout=5)
            stderr = process.stderr.read()
            self.assertEqual(process.returncode, 130, "".join(lines) + stderr)
            self.assertIn("cancelled", stderr.lower())
            self.assertNotIn("SPILLOT_RESULT\t", "".join(lines))
            self.assertTrue(all(e["fraction"] < 1 for e in progress_events("".join(lines))))
            self.assertEqual(list(self.directory.rglob(".spillot-tmp-*")), [])
            self.assertEqual(list(Path(manifest["output_dir"]).iterdir()), [])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            reader.join(timeout=5)
            process.stdout.close()
            process.stderr.close()

    def test_fiji_hyperstack_targets_and_source_files_are_preserved(self):
        stack = np.tile(fixture(13, 17), (4, 1, 1))
        raw = self.directory / "fiji.raw"
        stack.tofile(raw)
        names = ["Plane " + str(i + 1) for i in range(12)]
        matrix = self.directory / "fiji_selection.csv"
        selection = np.zeros((12, 12), dtype=int)
        selection[0, 1] = 1
        selection[2, 1] = 1  # Persisted row that is not selected this run.
        with matrix.open("w", newline="") as stream:
            csv.writer(stream).writerows([["", *names]] + [
                [names[i], *selection[i]] for i in range(12)])
        original = self.directory / "roi.tif"
        original.write_bytes(b"source TIFF must stay intact")
        original_csv = self.directory / "roi.csv"
        original_csv.write_bytes(b"source CSV must stay intact")
        manifest = {
            "input_raw": str(raw), "shape": list(stack.shape), "dtype": stack.dtype.str,
            "output_dir": str(self.directory), "base_name": "roi",
            "csv_path": str(matrix), "channel_names": names,
            "pixel_size_um": [.65, .8], "origin_xy": [-3.25, -4.5],
            "patch_size": 4, "workers": 1, "memory_budget_mb": 384,
            "target_channels": [0, 4],  # Plane5 intentionally has no donors.
            "write_input_copy": False, "publish_csv": False,
            "fiji_dimensions": [2, 3, 2], "fiji_channel_names": ["C1", "C2"],
            "fiji_pixel_depth_um": 1.5, "frame_interval_seconds": 2.25,
        }
        path = self.directory / "fiji.json"
        path.write_text(json.dumps(manifest))
        report, _ = self.run_backend(path, tile_size=5, batch_size=7)
        self.assertTrue(report["completed"])
        self.assertEqual(len(report["output_files"]), 3)
        self.assertEqual(original.read_bytes(), b"source TIFF must stay intact")
        self.assertEqual(original_csv.read_bytes(), b"source CSV must stay intact")
        self.assertFalse((self.directory / "roi_Channel_3_SpillOT.tif").exists())
        with tifffile.TiffFile(self.directory / "roi_SpillOT.tif") as image:
            self.assertEqual(image.series[0].axes, "TZCYX")
            self.assertEqual(image.series[0].shape, (2, 3, 2, 13, 17))
            restored = image.asarray().reshape(stack.shape)
            xml = ET.fromstring(image.ome_metadata)
        np.testing.assert_array_equal(restored[1:], stack[1:])
        self.assertGreater(np.count_nonzero(restored[0] != stack[0]), 0)
        np.testing.assert_array_equal(
            tifffile.imread(self.directory / "roi_Channel_5_SpillOT.tif"), stack[4])
        ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
        pixels = xml.find(".//ome:Pixels", ns)
        self.assertEqual(float(pixels.attrib["PhysicalSizeX"]), .65)
        self.assertEqual(float(pixels.attrib["PhysicalSizeY"]), .8)
        self.assertEqual(float(pixels.attrib["PhysicalSizeZ"]), 1.5)
        self.assertEqual(float(pixels.attrib["TimeIncrement"]), 2.25)
        self.assertEqual([c.attrib["Name"] for c in pixels.findall("ome:Channel", ns)], ["C1", "C2"])

    def test_invalid_fiji_dimensions_or_targets_fail_before_io(self):
        path, manifest = self.make_manifest(fixture(13, 17))
        for overrides in ({"fiji_dimensions": [2, 2, 1]},
                          {"target_channels": [3]}, {"target_channels": [-1]},
                          {"target_channels": [True]}, {"write_input_copy": "false"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.backend.validate_manifest(dict(manifest, **overrides))

    def test_memory_plan_reduces_workers_and_fits_large_image_budget(self):
        plan = self.backend.plan_memory(
            (19, 13000, 20000), np.dtype("float32"), 24, 64,
            memory_budget_mb=384, available_bytes=1024 * 1024**2)
        self.assertLess(plan["workers"], 64)
        self.assertEqual(plan["workers"], 1)
        self.assertGreater(plan["tile_size"], 0)
        self.assertGreater(plan["batch_size"], 0)
        self.assertLessEqual(plan["estimated_peak_bytes"], plan["memory_budget_bytes"])
        self.assertLessEqual(plan["memory_budget_bytes"], 384 * 1024**2)

    def test_auto_budget_scales_past_eight_gib_with_available_ram(self):
        plans = []
        with mock.patch.object(self.backend.os, "cpu_count", return_value=256):
            for gib in (16, 64, 256):
                with self.subTest(available_gib=gib):
                    available = gib * 1024**3
                    plan = self.backend.plan_memory(
                        (20, 13000, 20000), np.dtype("float32"), 24, 256,
                        available_bytes=available)
                    self.assertEqual(plan["memory_budget_bytes"], int(available * 0.75))
                    self.assertLessEqual(plan["estimated_peak_bytes"], plan["memory_budget_bytes"])
                    plans.append(plan)
        self.assertGreater(plans[1]["memory_budget_bytes"], 8 * 1024**3)
        self.assertGreater(plans[2]["memory_budget_bytes"], plans[1]["memory_budget_bytes"])
        self.assertGreater(plans[1]["workers"], plans[0]["workers"])
        self.assertGreater(plans[2]["workers"], plans[1]["workers"])

    def test_auto_budget_leaves_headroom_and_adapts_on_a_busy_laptop(self):
        with mock.patch.object(self.backend.os, "cpu_count", return_value=14):
            plans = [self.backend.plan_memory(
                (20, 2000, 2000), np.dtype("float32"), 24, 14,
                available_bytes=gib * 1024**3) for gib in (1, 2, 6, 16)]
        for plan in plans:
            self.assertLessEqual(plan["memory_budget_bytes"], plan["available_bytes"] * .75)
            self.assertLessEqual(plan["estimated_peak_bytes"], plan["memory_budget_bytes"])
            self.assertGreater(plan["tile_size"], 0)
            self.assertGreater(plan["batch_size"], 0)
        self.assertEqual(plans[0]["workers"], 1)
        self.assertEqual(plans[-1]["workers"], 14)
        self.assertGreater(plans[-1]["workers"], plans[0]["workers"])

    def test_worker_reduction_reserves_final_io_headroom(self):
        # Budgets immediately below each worker threshold used to pass the
        # worker loop, then fail the final buffer check instead of reducing
        # workers again. A one-worker plan demonstrably fits these budgets.
        with mock.patch.object(self.backend.os, "cpu_count", return_value=14):
            for budget in (850, 1120, 1400):
                with self.subTest(memory_budget_mb=budget):
                    plan = self.backend.plan_memory(
                        (20, 2000, 2000), np.dtype("uint16"), 24, 14,
                        memory_budget_mb=budget, available_bytes=20 * 1024**3)
                    self.assertLessEqual(plan["estimated_peak_bytes"], plan["memory_budget_bytes"])
                    self.assertLess(plan["workers"], 14)

    def test_explicit_budget_cannot_consume_all_available_memory(self):
        available = 4 * 1024**3
        plan = self.backend.plan_memory(
            (20, 2000, 2000), np.dtype("float32"), 24, 14,
            memory_budget_mb=16 * 1024, available_bytes=available)
        self.assertEqual(plan["memory_budget_bytes"], int(available * .80))
        self.assertLessEqual(plan["estimated_peak_bytes"], plan["memory_budget_bytes"])

    def test_large_patch_fails_planning_before_allocating_solver_graph(self):
        with mock.patch.object(self.backend, "configure_graph") as graph:
            with self.assertRaisesRegex(ValueError, "memory budget|patch size"):
                self.backend.plan_memory(
                    (19, 13000, 20000), np.dtype("float32"), 512, 8,
                    memory_budget_mb=256, available_bytes=1024 * 1024**2)
        graph.assert_not_called()

    def test_oversized_explicit_tile_and_batch_are_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "memory budget"):
            self.backend.plan_memory(
                (19, 13000, 20000), np.dtype("float32"), 24, 1,
                memory_budget_mb=256, available_bytes=1024 * 1024**2,
                tile_size=20000, batch_size=100000)

    def test_progress_eta_warms_up_and_resets_between_phases(self):
        stdout = io.StringIO()
        with mock.patch.object(self.backend.time, "monotonic", side_effect=[0, 0, 1, 3, 4]):
            with redirect_stdout(stdout):
                progress = self.backend.Progress(lambda: None)
                progress.emit("clean", 0, 10, 0, "starting")
                progress.emit("clean", 2, 10, 0.2, "warming up")
                progress.emit("clean", 4, 10, 0.4, "measured")
                progress.emit("write", 1, 2, 0.9, "new phase")
        events = progress_events(stdout.getvalue())
        self.assertIsNone(events[0]["eta_seconds"])
        self.assertIsNone(events[1]["eta_seconds"])
        self.assertEqual(events[2]["eta_seconds"], 4.5)
        self.assertIsNone(events[3]["eta_seconds"])

    def publication_fixture(self, phase):
        directory = self.directory / phase
        directory.mkdir()
        for name in ("output", "scratch", "scratch/staged"):
            (directory / name).mkdir()
        previous = {"first.tif": b"old first image", "third.csv": b"old matrix"}
        replacements = {"first.tif": b"new first image", "second.tif": b"new second image",
                        "third.csv": b"new matrix"}
        for name, value in previous.items():
            (directory / "output" / name).write_bytes(value)
        for name, value in replacements.items():
            (directory / "scratch/staged" / name).write_bytes(value)
        run = subprocess.run([sys.executable, str(PUBLICATION_FAULT), str(directory), phase],
                             text=True, capture_output=True, timeout=30)
        self.assertEqual(run.returncode, 91, run.stdout + run.stderr)
        return directory, previous, replacements

    def assert_directory_contents(self, directory, expected):
        actual = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
        self.assertEqual(actual, expected)

    def test_crash_after_backup_or_publish_restores_previous_outputs(self):
        for phase in ("after_backup", "after_publish"):
            with self.subTest(phase=phase):
                directory, previous, _ = self.publication_fixture(phase)
                journal = directory / "scratch/publication.json"
                self.assertEqual(json.loads(journal.read_text())["state"], "pending")
                self.backend.recover_publication(directory / "scratch")
                self.assert_directory_contents(directory / "output", previous)
                self.assertFalse(journal.exists())
                self.assertEqual(list((directory / "scratch").glob(".spillot-publication-backups-*")), [])

    def test_crash_after_commit_preserves_new_outputs(self):
        directory, _, replacements = self.publication_fixture("committed")
        journal = directory / "scratch/publication.json"
        self.assertEqual(json.loads(journal.read_text())["state"], "committed")
        self.backend.recover_publication(directory / "scratch")
        self.assert_directory_contents(directory / "output", replacements)
        self.assertFalse(journal.exists())
        self.assertEqual(list((directory / "scratch").glob(".spillot-publication-backups-*")), [])

    def test_failed_recovery_keeps_backup_and_journal_for_retry(self):
        directory, previous, _ = self.publication_fixture("after_publish")
        journal = directory / "scratch/publication.json"
        value = json.loads(journal.read_text())
        entry = next(item for item in value["entries"]
                     if Path(item["destination"]).name == "first.tif")
        backup, destination = Path(entry["backup"]), Path(entry["destination"])
        replace = os.replace

        def fail_restore(source, target, *args, **kwargs):
            if Path(source) == backup and Path(target) == destination:
                raise OSError("simulated restore failure")
            return replace(source, target, *args, **kwargs)

        with mock.patch.object(os, "replace", side_effect=fail_restore):
            with self.assertRaisesRegex(OSError, "simulated restore failure"):
                self.backend.recover_publication(directory / "scratch")
        self.assertEqual(backup.read_bytes(), previous["first.tif"])
        self.assertTrue(journal.exists())
        self.backend.recover_publication(directory / "scratch")
        self.assert_directory_contents(directory / "output", previous)
        self.assertFalse(journal.exists())


if __name__ == "__main__":
    unittest.main()
