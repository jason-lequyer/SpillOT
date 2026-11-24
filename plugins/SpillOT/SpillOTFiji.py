# Copyright 2025, Jason Lequyer, Ferris Nowlan and Laurence Pelletier.
# Jython 2.7 bridge: bounded ImageJ export and shared memory-aware processing.
from __future__ import division

import codecs
import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
import threading

from ij import IJ, ImagePlus, VirtualStack
from java.io import BufferedOutputStream, File, FileOutputStream
from java.nio import ByteBuffer, ByteOrder
from java.lang import Runnable
from java.awt import BorderLayout
from java.awt.event import ActionListener, WindowAdapter
from javax.swing import (JButton, JDialog, JLabel, JPanel, JProgressBar,
                         SwingUtilities, Timer, WindowConstants)


class RunCancelled(Exception):
    pass


def _text(value):
    return unicode(value)


def _csv_cell(value):
    value = _text(value)
    if any(ch in value for ch in (u',', u'"', u'\r', u'\n')):
        return u'"' + value.replace(u'"', u'""') + u'"'
    return value


def write_selection_csv(path, names, donors):
    """Write Unicode channel labels with actual CSV escaping."""
    with codecs.open(path, 'w', 'utf-8') as stream:
        stream.write(u',' + u','.join(_csv_cell(n) for n in names) + u'\n')
        for row, name in enumerate(names):
            selected = donors.get(row, set())
            stream.write(_csv_cell(name) + u',' + u','.join(
                u'1' if col != row and col in selected else u'0'
                for col in range(len(names))) + u'\n')


def read_selection_rows(path):
    with open(path, 'rb') as stream:
        return [[cell.decode('utf-8', 'replace') for cell in row]
                for row in csv.reader(stream) if row]


def _unit_to_um(unit):
    unit = _text(unit or '').strip().lower()
    return {'um': 1.0, u'\u00b5m': 1.0, u'\u03bcm': 1.0,
            'micron': 1.0, 'microns': 1.0, 'micrometer': 1.0,
            'micrometers': 1.0, 'micrometre': 1.0, 'micrometres': 1.0,
            'nm': 0.001, 'nanometer': 0.001, 'nanometers': 0.001,
            'nanometre': 0.001, 'nanometres': 0.001,
            'mm': 1000.0, 'cm': 10000.0, 'm': 1000000.0,
            'meter': 1000000.0, 'meters': 1000000.0,
            'metre': 1000000.0, 'metres': 1000000.0}.get(unit)


def _stack_names(imp, names, axis_indices):
    count = imp.getStackSize()
    result = []
    stack = imp.getStack()
    for index in range(1, count + 1):
        label = stack.getSliceLabel(index)
        result.append(_text(label) if label else u'Plane %d' % index)
    for name, index in zip(names, axis_indices):
        result[index - 1] = _text(name)
    return result


def export_manifest(imp, names, axis_indices, selected_channels, donor_map,
                    output_base, scratch, patch_size, workers, ignore_overexposed,
                    progress=None, cancel=None):
    """Export one source plane at a time, using <=1 MiB conversion buffers.

    Axis indices/selected channels follow ImageJ's 1-based convention; donor_map
    uses 0-based UI marker indices. All original planes are exported unchanged.
    """
    cancel = cancel or (lambda: None)
    progress = progress or (lambda fraction, message: None)
    width, height, count = imp.getWidth(), imp.getHeight(), imp.getStackSize()
    if patch_size > min(width, height):
        raise ValueError('Patch size cannot exceed the image width or height.')
    bit_depth = imp.getBitDepth()
    types = {8: ('|u1', 1), 16: ('<u2', 2), 32: ('<f4', 4)}
    if bit_depth not in types:
        raise ValueError('SpillOT requires 8-bit, 16-bit or 32-bit grayscale planes.')
    dtype, itemsize = types[bit_depth]
    if len(axis_indices) != len(names) or any(i < 1 or i > count for i in axis_indices):
        raise ValueError('Invalid channel-to-stack mapping.')
    selected_channels = sorted(set(selected_channels))
    if not selected_channels or any(i < 1 or i > len(names) for i in selected_channels):
        raise ValueError('No valid channels selected.')
    output_base = os.path.abspath(output_base)
    output_dir, base_name = os.path.dirname(output_base), os.path.basename(output_base)
    raw_path = os.path.join(scratch, 'input.raw')
    raw_bytes = long(width) * height * count * itemsize
    # Raw input + restored raw + final full/single TIFFs, with padding margin.
    needed = raw_bytes * 3 + long(width) * height * (len(selected_channels)*itemsize + 6)
    if File(output_dir).getUsableSpace() < needed * 1.15 + 64*1024*1024:
        raise IOError('Not enough free disk space for temporary data and complete SpillOT outputs.')
    stack_names = _stack_names(imp, names, axis_indices)
    expanded = {}
    for channel in selected_channels:
        target = axis_indices[channel - 1] - 1
        expanded[target] = set(axis_indices[d] - 1 for d in donor_map.get(channel - 1, set())
                               if d != channel - 1)
    csv_path = os.path.join(scratch, 'selection.csv')
    write_selection_csv(csv_path, stack_names, expanded)
    buffer = ByteBuffer.allocate(1024*1024).order(ByteOrder.LITTLE_ENDIAN)
    short_view, float_view = buffer.asShortBuffer(), buffer.asFloatBuffer()
    chunk_pixels = buffer.capacity() // itemsize
    output = BufferedOutputStream(FileOutputStream(raw_path), 1024*1024)
    try:
        done = 0
        pixels_per_plane = width*height
        for index in range(1, count + 1):
            cancel()
            processor = imp.getStack().getProcessor(index)
            pixels = processor.getPixels()
            for start in range(0, pixels_per_plane, chunk_pixels):
                cancel()
                size = min(chunk_pixels, pixels_per_plane - start)
                if bit_depth == 8:
                    output.write(pixels, start, size)
                else:
                    view = short_view if bit_depth == 16 else float_view
                    view.clear()
                    view.put(pixels, start, size)
                    output.write(buffer.array(), 0, size*itemsize)
                done += size
                progress(float(done)/(count*pixels_per_plane),
                         'Preparing channel/plane %d of %d' % (index, count))
            pixels, processor = None, None
    finally:
        output.close()
    calibration = imp.getCalibration()
    unit_scale = _unit_to_um(calibration.getUnit())
    pixel_size = None
    if unit_scale and calibration.pixelWidth > 0 and calibration.pixelHeight > 0:
        pixel_size = [float(calibration.pixelWidth)*unit_scale,
                      float(calibration.pixelHeight)*unit_scale]
    nc, nz, nt = imp.getNChannels(), imp.getNSlices(), imp.getNFrames()
    manifest = {
        'input_raw': raw_path, 'shape': [count, height, width], 'dtype': dtype,
        'output_dir': output_dir, 'base_name': base_name, 'csv_path': csv_path,
        'scratch_dir': scratch, 'cancel_path': os.path.join(scratch, 'cancel.requested'),
        'patch_size': patch_size, 'workers': workers, 'memory_budget_mb': 0,
        'ignore_overexposed': bool(ignore_overexposed), 'channel_names': stack_names,
        'target_channels': [axis_indices[c - 1] - 1 for c in selected_channels],
        'write_input_copy': False, 'publish_csv': False,
        'fiji_dimensions': [nc, nz, nt],
        'fiji_channel_names': stack_names[:nc],
        'origin_xy': [-float(calibration.xOrigin), -float(calibration.yOrigin)],
        'pixel_size_um': pixel_size,
    }
    if imp.getProperty('Info') is not None:
        manifest['fiji_image_info'] = _text(imp.getProperty('Info'))
    if unit_scale and calibration.pixelDepth > 0:
        manifest['fiji_pixel_depth_um'] = float(calibration.pixelDepth)*unit_scale
    if calibration.frameInterval > 0:
        time_units = _text(calibration.getTimeUnit()).lower()
        scale = {'sec': 1, 's': 1, 'second': 1, 'seconds': 1,
                 'ms': 0.001, 'msec': 0.001, 'min': 60, 'minute': 60,
                 'minutes': 60, 'hour': 3600, 'hours': 3600}.get(time_units)
        if scale:
            manifest['frame_interval_seconds'] = float(calibration.frameInterval)*scale
    path = os.path.join(scratch, 'manifest.json')
    with codecs.open(path, 'w', 'utf-8') as stream:
        stream.write(json.dumps(manifest, ensure_ascii=True))
    return path, manifest


class RGBVirtualStack(VirtualStack):
    """Split RGB lazily, allocating only the requested grayscale plane."""
    def __init__(self, source):
        VirtualStack.__init__(self, source.getWidth(), source.getHeight(), None, '')
        self.source = source
    def getSize(self):
        return self.source.getStackSize()*3
    def getBitDepth(self):
        return 8
    def getProcessor(self, index):
        source_index = (index - 1)//3 + 1
        return self.source.getStack().getProcessor(source_index).getChannel((index - 1)%3 + 1, None)
    def getPixels(self, index):
        return self.getProcessor(index).getPixels()
    def getSliceLabel(self, index):
        return ('Red', 'Green', 'Blue')[(index - 1)%3]


def split_rgb_virtual(source):
    result = ImagePlus(source.getTitle() + ' (RGB split)', RGBVirtualStack(source))
    result.setDimensions(3, max(1, source.getNSlices()), max(1, source.getNFrames()))
    result.setOpenAsHyperStack(True)
    result.setCalibration(source.getCalibration().copy())
    return result


def _duration(seconds):
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return ('%d:%02d:%02d' % (hours, minutes, seconds)) if hours else ('%02d:%02d' % (minutes, seconds))


def stop_process_tree(process):
    """Escalate only after cooperative cancellation; kill owned descendants too."""
    try:
        # Fiji's current Java runtime exposes ProcessHandle. Keep strong handles
        # before stopping the parent so reparented workers can still be stopped.
        handle = process._process.toHandle()
        descendants_stream = handle.descendants()
        try:
            descendants = list(descendants_stream.toArray())
        finally:
            descendants_stream.close()
        handle.destroyForcibly()
        for child in descendants:
            if child.isAlive():
                child.destroyForcibly()
        process.wait()
        deadline = time.time() + 5
        while time.time() < deadline and any(child.isAlive() for child in descendants):
            time.sleep(0.05)
        return not handle.isAlive() and not any(child.isAlive() for child in descendants)
    except Exception:
        # Older Java runtimes: Windows has a native process-tree terminator.
        # POSIX uses a PID/PPID snapshot, never a broad process-name match.
        try:
            if os.name == 'nt' or os.name == 'java' and os._name == 'nt':
                killer = subprocess.Popen(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                killer.communicate()
                process.wait()
                return killer.returncode == 0
            listing = subprocess.Popen(['ps', '-axo', 'pid=,ppid='], stdout=subprocess.PIPE)
            output = listing.communicate()[0]
            if listing.returncode != 0:
                return False
            pairs = [tuple(map(int, line.split())) for line in output.splitlines() if line.strip()]
            owned = set([int(process.pid)])
            previous = -1
            while len(owned) != previous:
                previous = len(owned)
                owned.update(pid for pid, parent in pairs if parent in owned)
            killer = subprocess.Popen(['kill', '-KILL'] + [str(pid) for pid in sorted(owned)],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            killer.communicate()
            process.wait()
            # Do not authorize scratch cleanup on the less verifiable fallback.
            return False
        except Exception:
            return False


class ProgressDialog(object):
    def __init__(self, cancel_path):
        self.cancel_path = cancel_path
        self.cancelled = False
        self.cancel_at = None
        self.proc = None
        self.force_thread = None
        self.tree_stopped = True
        self.started = time.time()
        self.eta = None
        self.dialog = JDialog(None, 'SpillOT', False)
        self.dialog.setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE)
        self.dialog.setLayout(BorderLayout(8, 8))
        self.label = JLabel('Preparing image')
        self.bar = JProgressBar(0, 1000)
        self.bar.setStringPainted(True)
        self.elapsed = JLabel('Elapsed: 00:00')
        bottom = JPanel(BorderLayout())
        bottom.add(self.elapsed, BorderLayout.CENTER)
        self.cancel_button = JButton('Cancel')
        bottom.add(self.cancel_button, BorderLayout.EAST)
        self.dialog.add(self.label, BorderLayout.NORTH)
        self.dialog.add(self.bar, BorderLayout.CENTER)
        self.dialog.add(bottom, BorderLayout.SOUTH)
        self.dialog.setSize(550, 125)
        self.dialog.setLocationRelativeTo(None)
        owner = self
        class Cancel(ActionListener):
            def actionPerformed(self, event):
                owner.request_cancel()
        class Close(WindowAdapter):
            def windowClosing(self, event):
                owner.request_cancel()
        class Tick(ActionListener):
            def actionPerformed(self, event):
                text = 'Elapsed: ' + _duration(time.time() - owner.started)
                if owner.eta is not None and not owner.cancelled:
                    text += ' | ETA: ' + _duration(owner.eta)
                owner.elapsed.setText(text)
                if owner.cancelled and owner.proc is not None and owner.proc.poll() is None:
                    if time.time() - owner.cancel_at > 10 and owner.force_thread is None:
                        owner.tree_stopped = False
                        def force_stop():
                            owner.tree_stopped = stop_process_tree(owner.proc)
                        owner.force_thread = threading.Thread(target=force_stop)
                        owner.force_thread.daemon = True
                        owner.force_thread.start()
        self.cancel_button.addActionListener(Cancel())
        self.dialog.addWindowListener(Close())
        self.timer = Timer(1000, Tick())
        self.timer.start()
        self.dialog.setVisible(True)

    def request_cancel(self):
        if self.cancelled:
            return
        self.cancelled, self.cancel_at = True, time.time()
        with open(self.cancel_path, 'w') as stream:
            stream.write('cancel\n')
        self.cancel_button.setEnabled(False)
        self.label.setText('Cancelling and cleaning up...')

    def check_cancel(self):
        if self.cancelled:
            raise RunCancelled('SpillOT cancelled.')

    def update(self, fraction, message, eta=None):
        owner = self
        class Update(Runnable):
            def run(self):
                if not owner.cancelled:
                    owner.bar.setValue(int(1000*max(0, min(0.999, fraction))))
                    owner.label.setText(message)
                    owner.eta = eta
                    IJ.showProgress(fraction)
                    IJ.showStatus(message)
        SwingUtilities.invokeLater(Update())

    def close(self):
        self.timer.stop()
        self.dialog.dispose()


def _recover(pyexe, backend_dir, scratch, env):
    # Run the same durable publication recovery used by QuPath. Never delete
    # scratch when recovery failed: it may contain the sole copy of prior data.
    if not os.path.isfile(os.path.join(scratch, 'publication.json')):
        return True
    script = ('import sys; sys.path.insert(0, sys.argv[1]); '
              'from SpillOTMemory import recover_publication; '
              'recover_publication(sys.argv[2])')
    process = subprocess.Popen([pyexe, '-c', script, backend_dir, scratch],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    output = process.communicate()[0]
    if process.returncode:
        IJ.log('SpillOT recovery failed; preserved temporary files at ' + scratch)
        IJ.log(output.decode('utf-8', 'replace'))
        return False
    return True


def run(imp, names, axis_indices, selected_channels, donor_map, output_base,
        backend_dir, pyexe, env, patch_size, workers, ignore_overexposed):
    """Run the shared backend once, with live progress and a cancellation marker."""
    runner = os.path.join(backend_dir, 'SpillOTMemory.py')
    if not os.path.isfile(runner) or not os.path.isfile(os.path.join(backend_dir, 'SpillOT.py')):
        raise IOError('Install the complete plugins/SpillOT folder, including SpillOTMemory.py.')
    output_dir = os.path.dirname(os.path.abspath(output_base))
    scratch = tempfile.mkdtemp(prefix='.spillot-fiji-', dir=output_dir)
    dialog = ProgressDialog(os.path.join(scratch, 'cancel.requested'))
    process = None
    try:
        path, manifest = export_manifest(imp, names, axis_indices, selected_channels, donor_map,
            output_base, scratch, patch_size, workers, ignore_overexposed,
            lambda fraction, message: dialog.update(fraction*0.15, message), dialog.check_cancel)
        dialog.check_cancel()
        process = subprocess.Popen([pyexe, '-u', runner, '--manifest', path],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        dialog.proc = process
        report, recent = None, []
        while True:
            raw = process.stdout.readline()
            if not raw:
                break
            line = raw.decode('utf-8', 'replace').rstrip()
            if line.startswith('SPILLOT_PROGRESS\t'):
                update = json.loads(line.split('\t', 1)[1])
                dialog.update(0.15 + 0.85*float(update['fraction']),
                              update.get('message', 'Processing'), update.get('eta_seconds'))
            elif line.startswith('SPILLOT_RESULT\t'):
                report = json.loads(line.split('\t', 1)[1])
            elif line:
                IJ.log(line)
                recent.append(line)
                recent = recent[-30:]
        code = process.wait()
        if code == 130:
            raise RunCancelled('SpillOT cancelled.')
        if code != 0 or not report or not report.get('completed'):
            raise RuntimeError('SpillOT did not finish (exit %s).\n%s' % (code, '\n'.join(recent)))
        full_path = output_base + '_SpillOT.tif'
        if not os.path.isfile(full_path):
            raise IOError('The completed SpillOT image was not found: ' + full_path)
        return full_path, report
    finally:
        dialog.close()
        # Cooperative cancellation lets Python stop its own worker pool.
        if process is not None and process.poll() is None:
            with open(os.path.join(scratch, 'cancel.requested'), 'w') as stream:
                stream.write('cancel\n')
            deadline = time.time() + 10
            while process.poll() is None and time.time() < deadline:
                time.sleep(0.05)
            if process.poll() is None:
                dialog.tree_stopped = stop_process_tree(process)
        if dialog.force_thread is not None:
            dialog.force_thread.join(10)
        if dialog.tree_stopped and _recover(pyexe, backend_dir, scratch, env):
            shutil.rmtree(scratch)
        else:
            IJ.log('SpillOT preserved temporary files for recovery at: ' + scratch)


def _open_bioformats_virtual(path):
    from loci.plugins import BF
    ImporterOptions = getattr(__import__("loci.plugins.in", fromlist=["ImporterOptions"]), "ImporterOptions")
    options = ImporterOptions()
    options.setId(path)
    options.setVirtual(True)
    options.setQuiet(True)
    options.setWindowless(True)
    options.setOpenAllSeries(False)
    images = BF.openImagePlus(options)
    if not images:
        raise IOError('Bio-Formats could not open the saved result: ' + path)
    return images[0]


class SelectedVirtualStack(VirtualStack):
    def __init__(self, source, indices, labels):
        VirtualStack.__init__(self, source.getWidth(), source.getHeight(), None, '')
        self.source = source  # Keep the reader and virtual stack alive.
        self.stack = source.getStack()
        self.indices = list(indices)
        self.labels = list(labels)
    def getSize(self):
        return len(self.indices)
    def getBitDepth(self):
        return self.source.getBitDepth()
    def getProcessor(self, index):
        return self.stack.getProcessor(self.indices[index - 1])
    def getPixels(self, index):
        return self.getProcessor(index).getPixels()
    def getSliceLabel(self, index):
        return self.labels[index - 1]


class SelectedVirtualImage(ImagePlus):
    def __init__(self, source, indices, labels):
        self.reader_image = source
        ImagePlus.__init__(self, 'SpillOT processed channels', SelectedVirtualStack(source, indices, labels))
    def close(self):
        ImagePlus.close(self)
        self.reader_image.close()


def open_selected_virtual(path, original, selected_indices, selected_labels, show=True):
    # An independent reader lets either window close without invalidating the
    # other. Each reader loads requested planes; neither creates a whole stack.
    source = _open_bioformats_virtual(path)
    try:
        selected = SelectedVirtualImage(source, selected_indices, selected_labels)
        selected.setDimensions(len(selected_indices), 1, 1)
        selected.setOpenAsHyperStack(True)
        selected.setCalibration(original.getCalibration().copy())
        if show:
            selected.show()
        return selected
    except Exception:
        source.close()
        raise


def open_result_virtual(path, original, selected_indices, selected_labels, show=True):
    """Open saved full and selected-channel images as independent virtual views."""
    image = _open_bioformats_virtual(path)
    image.setDimensions(original.getNChannels(), original.getNSlices(), original.getNFrames())
    image.setOpenAsHyperStack(True)
    image.setCalibration(original.getCalibration().copy())
    source_stack = original.getStack()
    result_stack = image.getStack()
    for index in range(1, original.getStackSize() + 1):
        label = source_stack.getSliceLabel(index)
        if label:
            result_stack.setSliceLabel(label, index)
    for index, label in zip(selected_indices, selected_labels):
        result_stack.setSliceLabel(label, index)
    image.setTitle(os.path.basename(path))
    if original.getProperty('Info') is not None:
        image.setProperty('Info', original.getProperty('Info'))
    if show:
        open_selected_virtual(path, original, selected_indices, selected_labels)
        image.show()
    return image
