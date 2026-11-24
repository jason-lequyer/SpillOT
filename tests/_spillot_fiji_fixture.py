"""Run with Fiji's Jython + ImageJ jars; creates real ImagePlus export fixtures."""
from __future__ import print_function

import json
import os
import sys
sys.dont_write_bytecode = True

from ij import ImagePlus, ImageStack, VirtualStack
from ij.process import ByteProcessor, ShortProcessor, FloatProcessor, ColorProcessor


repo, destination, kind, virtual = sys.argv[1:5]
sys.path.insert(0, os.path.join(repo, "plugins", "SpillOT"))
from SpillOTFiji import export_manifest, open_result_virtual, split_rgb_virtual
import SpillOTFiji as bridge

if len(sys.argv) > 5 and sys.argv[5] == "rgb":
    source_stack = ImageStack(5, 4)
    for frame in range(2):
        ip = ColorProcessor(5, 4)
        for y in range(4):
            for x in range(5):
                ip.set(x, y, ((10 + frame + x) << 16) | ((30 + y) << 8) | (50 + x + y))
        source_stack.addSlice("RGB frame %d" % frame, ip)
    source = ImagePlus("RGB fixture", source_stack)
    source.setDimensions(1, 1, 2)
    split = split_rgb_virtual(source)
    values = []
    for plane in range(1, 7):
        ip = split.getStack().getProcessor(plane)
        values.append([[ip.get(x, y) for x in range(5)] for y in range(4)])
    with open(os.path.join(destination, "rgb.json"), "w") as stream:
        json.dump({"dimensions": list(split.getDimensions()),
                   "virtual": bool(split.getStack().isVirtual()),
                   "source_dimensions": list(source.getDimensions()),
                   "source_first_pixel": source.getStack().getProcessor(1).get(0, 0),
                   "data": values}, stream)
    sys.exit(0)

width, height = 11, 9
reads = []


def value(plane, x, y):
    v = (plane * 17 + x * 7 + y * 3 + x * y) % 239
    if kind == "float32":
        return (v - 120) * .125
    if kind == "uint16":
        return v * 257
    return v


def processor(plane):
    factory = {"uint8": ByteProcessor, "uint16": ShortProcessor,
               "float32": FloatProcessor}[kind]
    ip = factory(width, height)
    for y in range(height):
        for x in range(width):
            if kind == "float32":
                ip.setf(x, y, value(plane, x, y))
            else:
                ip.set(x, y, value(plane, x, y))
    return ip


class FixtureVirtualStack(VirtualStack):
    def __init__(self):
        VirtualStack.__init__(self, width, height, None, "")

    def getSize(self):
        return 12

    def getProcessor(self, index):
        reads.append(index)
        return processor(index)

    def getSliceLabel(self, index):
        return "original plane %d" % index


if virtual == "true":
    stack = FixtureVirtualStack()
else:
    stack = ImageStack(width, height)
    for index in range(1, 13):
        stack.addSlice("original plane %d" % index, processor(index))
imp = ImagePlus("fixture", stack)
imp.setDimensions(2, 3, 2)
imp.setOpenAsHyperStack(True)
calibration = imp.getCalibration()
calibration.pixelWidth = .65
calibration.pixelHeight = .8
calibration.pixelDepth = 1.5
calibration.setUnit("um")
calibration.xOrigin = 3.25
calibration.yOrigin = 4.5
imp.setCalibration(calibration)
imp.setProperty("Info", "source fixture metadata")
if len(sys.argv) > 5 and sys.argv[5] == "read":
    result = open_result_virtual(os.path.join(destination, "original_SpillOT.tif"),
                                 imp, [1, 5], ["cleaned target", "selected no donor"], show=False)
    data = []
    for index in range(1, result.getStackSize() + 1):
        ip = result.getStack().getProcessor(index)
        data.append([[ip.getf(x, y) for x in range(width)] for y in range(height)])
    reopened = {"virtual": bool(result.getStack().isVirtual()),
                "dimensions": list(result.getDimensions()), "data": data,
                "pixel_width": result.getCalibration().pixelWidth,
                "pixel_height": result.getCalibration().pixelHeight,
                "info": str(result.getProperty("Info")),
                "x_origin": result.getCalibration().xOrigin,
                "y_origin": result.getCalibration().yOrigin}
    selected = bridge.open_selected_virtual(os.path.join(destination, "original_SpillOT.tif"),
                                            imp, [1, 5], ["cleaned target", "selected no donor"], show=False)
    result.close()
    selected_data = []
    for index in range(1, 3):
        ip = selected.getStack().getProcessor(index)
        selected_data.append([[ip.getf(x, y) for x in range(width)] for y in range(height)])
    reopened.update({"selected_data": selected_data,
                     "selected_dimensions": list(selected.getDimensions()),
                     "selected_virtual": bool(selected.getStack().isVirtual()),
                     "selected_labels": [selected.getStack().getSliceLabel(i) for i in (1, 2)]})
    with open(os.path.join(destination, "reopened.json"), "w") as stream:
        json.dump(reopened, stream)
    selected.close()
    sys.exit(0)
original = os.path.join(destination, "original.tif")
with open(original, "wb") as stream:
    stream.write(b"original-file-must-not-be-overwritten")
scratch = os.path.join(destination, "scratch")
os.makedirs(scratch)
progress = []

if len(sys.argv) > 5 and sys.argv[5] in ("run", "run-error"):
    logs = []
    original_ij = bridge.IJ
    class CapturingIJ(object):
        def log(self, line):
            logs.append(unicode(line))
        def __getattr__(self, name):
            return getattr(original_ij, name)
    bridge.IJ = CapturingIJ()
    class HeadlessProgress(object):
        def __init__(self, cancel_path):
            self.force_thread, self.tree_stopped, self.proc = None, True, None
        def update(self, fraction, message, eta=None):
            progress.append([float(fraction), unicode(message), eta])
        def check_cancel(self):
            pass
        def close(self):
            pass
    bridge.ProgressDialog = HeadlessProgress
    failing = sys.argv[5] == "run-error"
    if failing:
        # Export succeeds, then the actual Python backend diagnoses invalid data.
        imp.getStack().getProcessor(1).setf(0, 0, float("nan"))
    try:
        output, report = bridge.run(
            imp, [u"Target \u03b1", u"Donor, marker", u"Selected no donor"],
            [1, 3, 5], [1, 3], {0: set([1]), 1: set([2]), 2: set()},
            os.path.splitext(original)[0], os.path.join(repo, "plugins", "SpillOT"),
            sys.argv[6], os.environ.copy(), 4, 1, False)
    except RuntimeError as error:
        if not failing:
            raise
        with open(os.path.join(destination, "bridge-error.json"), "w") as stream:
            json.dump({"error": unicode(error), "logs": logs, "progress": progress}, stream)
        sys.exit(0)
    if failing:
        raise AssertionError("The invalid-image backend run unexpectedly succeeded")
    with open(os.path.join(destination, "bridge.json"), "w") as stream:
        json.dump({"output": output, "report": report, "progress": progress, "logs": logs}, stream)
    sys.exit(0)


def note(*args):
    progress.append([str(a) for a in args])


manifest_path, manifest = export_manifest(
    imp, [u"Target \u03b1", u"Donor, marker", u"Selected no donor"],
    [1, 3, 5], [1, 3], {0: set([1]), 1: set([2]), 2: set()},
    os.path.splitext(original)[0], scratch, 4, 1, False,
    progress=note, cancel=lambda: None)
with open(os.path.join(destination, "fixture.json"), "w") as stream:
    json.dump({"manifest_path": str(manifest_path), "manifest": manifest,
               "reads": reads, "progress": progress,
               "selection_rows": bridge.read_selection_rows(manifest["csv_path"]),
               "original_path": original,
               "source_dimensions": list(imp.getDimensions()),
               "source_info": str(imp.getProperty("Info")),
               "source_first_pixel": imp.getStack().getProcessor(1).getf(0, 0)}, stream)
print("Fiji fixture exported")
