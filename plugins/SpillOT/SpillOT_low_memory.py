#!/usr/bin/env python3
# Copyright 2025, Jason Lequyer, Ferris Nowlan and Laurence Pelletier.
# All rights reserved. Sinai Health System, Lunenfeld-Tanenbaum Research Institute.
"""Disk-backed, bounded-memory SpillOT backend for QuPath exports.

Spatial output cores use all original global patch origins that overlap them,
with a p+1 input halo and reflection only at true image boundaries. Patch batches
reuse the original solver and comparison metric. Per-patch RNG seeds make the
result independent of worker scheduling; this is a new random realization from
legacy unseeded runs. Small floating-point differences remain possible across
NumPy versions. No complete image stack or global patch stack is held in RAM.

Manifest input is caller-owned, little-endian CYX raw data. Outputs are staged
privately and published only after every TIFF is complete. Saturation inpainting
uses the original global KD-tree when budgeted, otherwise exact global neighbor
distances with deterministic row/column tie-breaking (which can differ from the
legacy KD-tree at tied kth-neighbor distances).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import threading
import time

import numpy as np
from tifffile import TiffWriter

DEFAULT_BACKEND = Path(__file__).resolve().with_name("SpillOT.py")
_WORKER_BACKEND = None
MIB = 1024 * 1024

def load_backend(path: str | Path):
    name = "_spillot_low_memory_stock"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SpillOT backend: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def patch_seed(seed: int, channel: int, global_patch: int) -> int:
    """Stable 32-bit seed, independent of batch size and worker assignment."""
    mask = (1 << 64) - 1
    value = (int(seed) ^ ((int(channel) + 1) * 0x9E3779B97F4A7C15)
             ^ ((int(global_patch) + 1) * 0xD1B54A32D192ED03)) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & 0xFFFFFFFF

def configure_graph(backend, patch_size: int) -> None:
    """Create exactly the stock index tuples without ten dense float tensors."""
    backend._set_patsize(patch_size)
    q = patch_size + 4
    yy, xx = np.mgrid[1:q-1, 1:q-1]
    yy, xx = yy.ravel(), xx.ravel()
    directions = {
        "top_idx": (-1, 0), "bot_idx": (1, 0),
        "lef_idx": (0, -1), "rit_idx": (0, 1),
        "toplef_idx": (-1, -1), "toprit_idx": (-1, 1),
        "botlef_idx": (1, -1), "botrit_idx": (1, 1),
    }
    for name, (dy, dx) in directions.items():
        setattr(backend, name, (yy, xx, yy + dy, xx + dx))

    evens = np.zeros((q, q), dtype=bool)
    odds = np.zeros_like(evens)
    for y in range(1, q - 1):
        for x in range(1, q - 1):
            if y in (1, q - 2) or x in (1, q - 2):
                (evens if (y + x) % 2 == 0 else odds)[y, x] = True
    evens_idx, odds_idx = np.where(evens), np.where(odds)
    border = np.zeros((q, q, q, q), dtype=bool)
    for y in range(1, q - 1):
        for x in range(1, q - 1):
            if y in (1, q - 2) or x in (1, q - 2):
                border[y, x][odds_idx if (y + x) % 2 == 0 else evens_idx] = True
            if y == 1:
                border[y, x, y + 1, x] = True
            if x == 1:
                border[y, x, y, x + 1] = True
            if y == q - 2:
                border[y, x, y - 1, x] = True
            if x == q - 2:
                border[y, x, y, x - 1] = True
    backend.bord_idx = np.where(border)

def reflection_masks(ys: np.ndarray, xs: np.ndarray,
                     height: int, width: int, q: int) -> np.ndarray:
    """Stock reflection masks at true image edges, for just these origins."""
    y = ys[None, None, :] + np.arange(q)[:, None, None]
    x = xs[None, None, :] + np.arange(q)[None, :, None]
    valid_y = (y >= 2) & (y < height + 2)
    valid_x = (x >= 2) & (x < width + 2)
    masks = np.empty((4, q, q, len(ys)), dtype=np.float32)
    masks[0] = np.where(valid_y & (x == 1), 2.0,
                        np.where(valid_y & (x == 2), 1.0, 0.0))
    masks[1] = np.where(valid_y & (x == width + 2), 2.0,
                        np.where(valid_y & (x == width + 1), 1.0, 0.0))
    masks[2] = np.where(valid_x & (y == 1), 2.0,
                        np.where(valid_x & (y == 2), 1.0, 0.0))
    masks[3] = np.where(valid_x & (y == height + 2), 2.0,
                        np.where(valid_x & (y == height + 1), 1.0, 0.0))
    # Identical to the stock "shift RF masks" operation, now batch-local.
    for index in range(4):
        part = (masks[index, 2:-2, 1:-1, :].copy() if index < 2
                else masks[index, 1:-1, 2:-2, :].copy())
        masks[index] *= 0
        if index < 2:
            masks[index, 2:-2, 1:-1, :] = part
        else:
            masks[index, 1:-1, 2:-2, :] = part
    return masks

def _init_worker(backend_path: str, patch_size: int) -> None:
    global _WORKER_BACKEND
    _WORKER_BACKEND = load_backend(backend_path)
    configure_graph(_WORKER_BACKEND, patch_size)

def _solve_worker(task):
    local_index, global_index, patch, masks, seed, channel = task
    backend = _WORKER_BACKEND
    backend.base2d = patch[None, :, :, None]
    backend.rf = masks[:, :, :, None]
    np.random.seed(patch_seed(seed, channel, global_index))
    _, domino = backend._solve_one_plane(0)
    return local_index, domino

def target_statistics(backend, base: np.ndarray, domino: np.ndarray):
    """The stock per-column tolerance, with the same operations/dtypes."""
    pixels, positions = base.shape
    base_nonzero = np.where(base > 0, base, np.inf)
    base_min = base_nonzero.min(axis=0)
    base_min[~np.isfinite(base_min)] = 0.0
    base_min = base_min.astype(np.float32)
    base_shift = base + (base_min - base.min(axis=0))
    dom_shift = domino + (base_min - domino.min(axis=0))
    positive = base > 0
    support = positive.sum(axis=0).astype(np.int32)
    if backend.DEFAULT_DROPOUT_ROBUST:
        weights = np.where(positive, np.float32(1.0),
                           np.float32(backend.DEFAULT_DROPOUT_ZERO_WEIGHT)).astype(np.float32)
        difference = np.sum(weights * np.abs(base_shift - dom_shift), axis=0)
    else:
        weights = np.ones_like(base, dtype=np.float32)
        difference = np.sum(np.abs(base_shift - dom_shift), axis=0)
    return difference, base_min, weights, support

def load_selection(csv_path: Path, channels: int, explicit: bool) -> np.ndarray:
    matrix = np.zeros((channels, channels), dtype=np.float32)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            rows = csv.reader(stream)
            next(rows, None)
            for row_index, row in enumerate(rows):
                if row_index >= channels:
                    break
                for column_index, cell in enumerate(row[1:channels+1]):
                    try:
                        value = np.float32(float(cell.strip()))
                    except (ValueError, OverflowError):
                        continue
                    if np.isclose(value, 1, atol=1e-6) or np.isclose(value, -1, atol=1e-6):
                        matrix[row_index, column_index] = value
    except Exception as exc:
        if explicit:
            raise ValueError(f"Cannot read selected CSV {csv_path}: {exc}") from exc
        print(f"No usable default CSV {csv_path}; channels will be unchanged: {exc}", flush=True)
    return matrix


class CooperativeCancel(RuntimeError):
    """The caller requested cancellation; staged results have been removed."""


class CancelCheck:
    def __init__(self, event=None, cancel_path=None):
        self.event = event or threading.Event()
        self.cancel_path = Path(cancel_path) if cancel_path else None

    def __call__(self):
        if self.event.is_set() or (self.cancel_path is not None and self.cancel_path.exists()):
            raise CooperativeCancel("SpillOT was cancelled; incomplete outputs were removed.")


class Progress:
    def __init__(self, cancel):
        self.cancel = cancel
        self.fraction = 0.0
        self.phase_started = time.monotonic()
        self.previous_phase = None
        self.last_emitted = float("-inf")

    def emit(self, phase, completed, total, fraction, message, eta_seconds=None):
        # Publication is the commit point. A later cancellation cannot turn
        # already-committed complete outputs into an unsuccessful partial run.
        if phase != "complete":
            self.cancel()
        self.fraction = max(self.fraction, min(1.0, float(fraction)))
        now = time.monotonic()
        phase_changed = phase != self.previous_phase
        if phase_changed:
            self.phase_started = now
            self.previous_phase = phase
        if eta_seconds is None and completed > 0 and now - self.phase_started >= 2:
            eta_seconds = (now - self.phase_started) * max(0, total - completed) / completed
        event = dict(phase=phase, completed=int(completed), total=int(total),
                     fraction=self.fraction,
                     eta_seconds=None if eta_seconds is None else round(max(0.0, eta_seconds), 1),
                     message=str(message))
        if phase_changed or completed >= total or now-self.last_emitted >= 0.25:
            self.last_emitted = now
            print("SPILLOT_PROGRESS\t" + json.dumps(event, allow_nan=False), flush=True)


def available_memory_bytes():
    """Available memory, without making psutil a required dependency."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except (ImportError, OSError, AttributeError):
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_extended", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.avail_phys)
        except (OSError, AttributeError, ValueError):
            pass
    elif sys.platform == "darwin":
        try:
            import re
            import subprocess
            text = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=3)
            page_size = int(re.search(r"page size of (\d+) bytes", text).group(1))
            pages = sum(int(re.search(r"^"+re.escape(name)+r":\s+(\d+)", text, re.M).group(1))
                        for name in ("Pages free", "Pages inactive", "Pages speculative"))
            return pages*page_size
        except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
            pass
    else:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])*1024
        except (OSError, ValueError):
            pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        # Leave space for QuPath, the OS, and other applications on platforms
        # without an available-memory query. This is deliberately conservative.
        try:
            return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 4)
        except (ValueError, OSError, AttributeError):
            return 1024 * MIB


def plan_memory(shape, dtype, patch_size, workers, memory_budget_mb=0,
                tile_size=None, batch_size=None, available_bytes=None):
    """Conservative working-set plan, including Python workers and solver cost.

    The budget is an allocation target, not a hard OS RSS limit. The raw input,
    restored stack, saturation mask and intermediate float plane are disk-backed.
    Requested worker counts are reduced before selecting spatial/batch sizes.
    Explicit tile/batch sizes are accepted only if this estimate fits.
    """
    channels, height, width = map(int, shape)
    p = int(patch_size)
    workers = int(workers)
    if min(channels, height, width) < 1 or p < 4 or p % 2:
        raise ValueError("Shape must be positive and patch_size must be an even integer >= 4.")
    if min(height, width) < p:
        raise ValueError("Image height and width must be at least the patch size.")
    if workers < 1 or memory_budget_mb < 0:
        raise ValueError("Workers must be positive and memory budget must be nonnegative.")
    for name, value in (("tile_size", tile_size), ("batch_size", batch_size)):
        if value is not None and int(value) < 1:
            raise ValueError(name + " must be a positive integer.")
    available = max(1, int(available_bytes if available_bytes is not None else available_memory_bytes()))
    requested_budget = int(float(memory_budget_mb) * MIB)
    budget = requested_budget if requested_budget else min(8 * 1024 * MIB, int(available * 0.45))
    # An explicit budget cannot reserve memory the machine does not currently
    # have. Keep a smaller safety margin because the caller chose that budget.
    budget = min(budget, int(available * 0.80))
    q = p + 4
    parent_bytes = 256 * MIB + 40 * q**4
    worker_bytes = 256 * MIB + 32 * q**4
    requested_workers = workers
    workers = min(workers, max(1, os.cpu_count() or 1))
    per_patch = 64 * q*q + 160 * p*p
    minimum_work = per_patch * 2 + 40 * (2 * (p+1) + 1)**2 + 16 * MIB
    while workers > 1 and parent_bytes + workers * worker_bytes + minimum_work > budget:
        workers -= 1
    fixed = parent_bytes + (workers * worker_bytes if workers > 1 else 0)
    remaining = budget - fixed - 16 * MIB  # bounded I/O buffers and TIFF writer
    if remaining <= minimum_work:
        raise ValueError("The memory budget is too small for this patch size and one worker "
                         f"(budget {budget/MIB:.0f} MiB; minimum estimate "
                         f"{(fixed+minimum_work+16*MIB)/MIB:.0f} MiB).")
    batch = int(batch_size) if batch_size is not None else max(1, min(256, int(remaining * 0.25 // per_patch)))
    batch_bytes = max(2, batch) * per_patch
    tile_room = remaining - batch_bytes
    if tile_room <= 0:
        raise ValueError("The requested patch batch exceeds the memory budget.")
    # Float base/one donor with halo, masks, fold sums/counters, casting and
    # temporary reads fit within this deliberately generous coefficient.
    adaptive_tile = int(math.sqrt(max(1, tile_room * 0.70 / 40))) - 2 * (p+1)
    tile = int(tile_size) if tile_size is not None else max(1, min(2048, max(height, width), adaptive_tile))
    tile_bytes = 40 * (tile + 2 * (p+1))**2
    estimate = fixed + 16 * MIB + batch_bytes + tile_bytes
    if estimate > budget:
        raise ValueError("The requested tile and batch sizes exceed the memory budget; "
                         "reduce --tile-size, --batch-size, patch size, or workers.")
    return dict(memory_budget_mb=round(budget/MIB, 2), memory_budget_bytes=budget,
                available_bytes=available, estimated_peak_bytes=estimate,
                fixed_bytes=fixed, workers=workers, requested_workers=requested_workers,
                tile_size=tile, batch_size=batch, patch_size=p, halo=p+1,
                bytes_per_patch_estimate=per_patch)


class RawArray:
    """Bounded rectangular I/O; avoids retaining resident mmap pages."""
    def __init__(self, path, shape, dtype, create=False, cancel=None):
        self.path = Path(path)
        self.shape = tuple(map(int, shape))
        self.dtype = np.dtype(dtype)
        self.cancel = cancel or (lambda: None)
        self.size = math.prod(self.shape) * self.dtype.itemsize
        self.file = open(self.path, "w+b" if create else "rb", buffering=0)
        if create:
            self.file.truncate(self.size)
        elif os.fstat(self.file.fileno()).st_size != self.size:
            self.file.close()
            raise ValueError(f"Raw file size does not match shape/dtype: {self.path}")

    def close(self):
        self.file.close()

    def read(self, channel, y0, y1, x0, x1):
        self.cancel()
        channels, height, width = self.shape
        if not (0 <= channel < channels and 0 <= y0 <= y1 <= height and 0 <= x0 <= x1 <= width):
            raise ValueError("Raw read outside image bounds.")
        result = np.empty((y1-y0, x1-x0), dtype=self.dtype)
        nbytes = (x1-x0) * self.dtype.itemsize
        for row, y in enumerate(range(y0, y1)):
            self.cancel()
            self.file.seek(((channel*height+y)*width+x0)*self.dtype.itemsize)
            view = memoryview(result[row]).cast("B")
            position = 0
            while position < nbytes:
                got = self.file.readinto(view[position:])
                if not got:
                    raise IOError("Unexpected end of raw image data.")
                position += got
        return result

    def write(self, channel, y0, x0, array):
        self.cancel()
        data = np.asarray(array, dtype=self.dtype)
        channels, height, width = self.shape
        h, w = data.shape
        if not (0 <= channel < channels and 0 <= y0 <= y0+h <= height and 0 <= x0 <= x0+w <= width):
            raise ValueError("Raw write outside image bounds.")
        for row, y in enumerate(range(y0, y0+h)):
            self.cancel()
            self.file.seek(((channel*height+y)*width+x0)*self.dtype.itemsize)
            remaining = memoryview(np.ascontiguousarray(data[row])).cast("B")
            while len(remaining):
                written = self.file.write(remaining)
                if not written:
                    raise IOError("Could not write complete raw image data.")
                remaining = remaining[written:]


def tiles(height, width, size):
    for y in range(0, height, size):
        for x in range(0, width, size):
            yield y, min(height, y+size), x, min(width, x+size)


def patch_bounds(core, height, width, patch):
    y0, y1, x0, x1 = core
    return (max(0, y0-patch+1), min(height-patch+1, y1),
            max(0, x0-patch+1), min(width-patch+1, x1))


def _reflected_indices(lo, hi, length):
    indices = np.arange(lo, hi, dtype=np.int64)
    period = 2*(length-1)
    indices %= period
    return np.where(indices < length, indices, period-indices)


def normalized_rect(raw, channel, bounds, offset, saturation=None):
    """Read only a bounded region; reflect exclusively at global boundaries."""
    y0, y1, x0, x1 = bounds
    ys = _reflected_indices(y0, y1, raw.shape[1])
    xs = _reflected_indices(x0, x1, raw.shape[2])
    ry, rx = int(ys.min()), int(xs.min())
    data = raw.read(channel, ry, int(ys.max())+1, rx, int(xs.max())+1).astype(np.float32)
    if offset != 0:
        data = data + float(offset)
    if saturation is not None:
        sat = saturation.read(0, ry, int(ys.max())+1, rx, int(xs.max())+1)
        data[sat != 0] = 0
    return np.ascontiguousarray(data[(ys-ry)[:, None], (xs-rx)[None, :]])


def selected_patches(array, ys, xs, q):
    windows = np.lib.stride_tricks.sliding_window_view(array, (q, q))
    return np.ascontiguousarray(windows[ys, xs].transpose(1, 2, 0))[None]


def _release_backend_arrays(backend):
    for name in ("base2d", "rf", "compare3d", "basediff", "base_min_nz", "dropout_weights",
                 "dropout_support_counts", "unfold", "RP"):
        if hasattr(backend, name):
            delattr(backend, name)


def _stop_pool(pool, failed):
    if pool is None:
        return
    processes = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=not failed, cancel_futures=True)
    if failed:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=1)


def process_core(backend, pool, raw, result, channel, donors, offsets,
                 saturation, core, patch, batch_size, seed, cancel, tick):
    """Restore one core using every overlapping global patch in row-major order."""
    height, width = raw.shape[1:]
    y0, y1, x0, x1 = core
    ya, yb, xa, xb = patch_bounds(core, height, width, patch)
    origin_width = xb-xa
    positions = (yb-ya)*origin_width
    q = patch+4
    base = normalized_rect(raw, channel, (ya-2, yb+patch+1, xa-2, xb+patch+1),
                           offsets[channel], saturation)
    sums = np.zeros((y1-y0, x1-x0), dtype=np.float32)
    counts = np.zeros_like(sums)
    removed_total = 0
    for first in range(0, positions, batch_size):
        cancel()
        local = np.arange(first, min(positions, first+batch_size), dtype=np.int64)
        ys, xs = ya + local//origin_width, xa + local % origin_width
        global_indices = ys*(width-patch+1)+xs
        base_patches = selected_patches(base, ys-ya, xs-xa, q)
        masks = reflection_masks(ys, xs, height, width, q)
        domino = base_patches.copy()
        if pool is None:
            backend.base2d, backend.rf = base_patches, masks
            for index, global_index in enumerate(global_indices):
                cancel()
                np.random.seed(patch_seed(seed, channel, int(global_index)))
                _, new_patch = backend._solve_one_plane(index)
                domino[0, 1:-1, 1:-1, index] = new_patch
        else:
            tasks = ((index, int(global_index), base_patches[0, :, :, index].copy(),
                      masks[:, :, :, index].copy(), seed, channel)
                     for index, global_index in enumerate(global_indices))
            # At most one bounded batch is submitted, even on Python versions
            # whose Executor.map consumes the complete iterable eagerly.
            for index, new_patch in pool.map(_solve_worker, tasks, chunksize=1):
                cancel()
                domino[0, 1:-1, 1:-1, index] = new_patch
        core_base = base_patches[0, 2:-2, 2:-2].copy().reshape(patch*patch, -1)
        core_domino = domino[0, 2:-2, 2:-2].copy().reshape(patch*patch, -1)
        # NumPy uses a different reduction path for one contiguous column.
        # Duplicate a singleton metric column to retain the legacy multi-column
        # path whenever the full image has more than one patch. Ignore the copy.
        singleton = len(local) == 1 and (height-patch+1)*(width-patch+1) > 1
        if singleton:
            core_base = np.repeat(core_base, 2, axis=1)
            core_domino = np.repeat(core_domino, 2, axis=1)
        difference, minimum, weights, support = target_statistics(backend, core_base, core_domino)
        backend._init_compare_pool(
            np.empty((1, 1, 0, 0), dtype=np.float32), core_base, difference, patch, minimum,
            weights, support, backend.DEFAULT_DROPOUT_ROBUST, backend.DEFAULT_DROPOUT_ZERO_WEIGHT,
            backend.DEFAULT_MIN_SIGNAL_PIXELS, backend.DEFAULT_DROPOUT_TOLERANCE,
            backend.DEFAULT_CORE_GATE, backend.DEFAULT_CORE_FRACTION, backend.DEFAULT_MIN_CORE_OVERLAP,
            backend.DEFAULT_SOURCE_CONTRAST_GATE, backend.DEFAULT_MIN_SOURCE_TO_TARGET_RANGE_RATIO)
        keep = np.ones(core_base.shape[1], dtype=bool)
        for donor in donors:
            cancel()
            dy, dx = int(ys.min()), int(xs.min())
            donor_region = normalized_rect(raw, int(donor),
                (dy-2, int(ys.max())+patch+2, dx-2, int(xs.max())+patch+2), offsets[donor], saturation)
            donor_patches = selected_patches(donor_region, ys-dy, xs-dx, q)
            if singleton:
                donor_patches = np.repeat(donor_patches, 2, axis=3)
            backend.RP = lambda frame: frame
            backend.unfold = lambda frame: donor_patches.reshape(1, q*q, -1)
            keep[backend._process_compare_channel(0)] = False
            del donor_region
        keep = keep[:len(local)]
        removed_total += int((~keep).sum())
        # Add only the part overlapping this output core. For each output pixel,
        # the original global row-major addition order is exactly preserved.
        for index, (py, px) in enumerate(zip(ys, xs)):
            by0, by1 = max(y0, int(py)), min(y1, int(py)+patch)
            bx0, bx1 = max(x0, int(px)), min(x1, int(px)+patch)
            value = base[by0-ya+2:by1-ya+2, bx0-xa+2:bx1-xa+2]
            sums[by0-y0:by1-y0, bx0-x0:bx1-x0] += value if keep[index] else value*0
            counts[by0-y0:by1-y0, bx0-x0:bx1-x0] += 1
        _release_backend_arrays(backend)
        tick(len(local)*(1+len(donors)))
    counts[counts == 0] = 1
    result.write(0, y0, x0, sums/counts)
    return positions, removed_total


def make_saturation_mask(raw, path, channels, size, cancel):
    height, width = raw.shape[1:]
    mask = RawArray(path, (1, height, width), "|u1", create=True, cancel=cancel)
    count = 0
    try:
        maximum = np.iinfo(raw.dtype).max
        for y0, y1, x0, x1 in tiles(height, width, size):
            values = np.zeros((y1-y0, x1-x0), dtype=bool)
            for channel in channels:
                values |= raw.read(int(channel), y0, y1, x0, x1) == maximum
            count += int(values.sum())
            mask.write(0, y0, x0, values)
        return mask, count
    except BaseException:
        mask.close()
        raise


def bounded_global_inpaint(result, saturation, cancel, progress=None, k=10, block=256):
    """Exact nearest distances with bounded buffers and no fixed search halo.

    Ties are ordered by row then column. Unlike an arbitrary tile-local search,
    the search expands until no unexamined pixel can be nearer or tied. Donors
    are never newly filled pixels because the original saturation mask excludes
    them. The fallback is intentionally slower than the global KD-tree.
    """
    height, width = result.shape[1:]
    donor_count = target_count = 0
    for y0, y1, x0, x1 in tiles(height, width, block):
        data = result.read(0, y0, y1, x0, x1)
        sat = saturation.read(0, y0, y1, x0, x1) != 0
        donor_count += int(((~sat) & (data != 0)).sum())
        target_count += int((sat & (data == 0)).sum())
    if not donor_count:
        # Saturated positions were zeroed before every patch was constructed;
        # consequently if no unsaturated nonzero donors remain, the full plane
        # is zero and the legacy global-median fallback is also exactly zero.
        return target_count
    k = min(int(k), donor_count)
    done = 0
    for ty0, ty1, tx0, tx1 in tiles(height, width, block):
        target_data = result.read(0, ty0, ty1, tx0, tx1)
        target_mask = (saturation.read(0, ty0, ty1, tx0, tx1) != 0) & (target_data == 0)
        for ly, lx in zip(*np.nonzero(target_mask)):
            cancel()
            y, x = int(ty0+ly), int(tx0+lx)
            radius = 8
            while True:
                ya, yb = max(0, y-radius), min(height, y+radius+1)
                xa, xb = max(0, x-radius), min(width, x+radius+1)
                best = []
                for by in range(ya, yb, block):
                    for bx in range(xa, xb, block):
                        ey, ex = min(yb, by+block), min(xb, bx+block)
                        vals = result.read(0, by, ey, bx, ex)
                        sat = saturation.read(0, by, ey, bx, ex) != 0
                        dy, dx = np.nonzero((~sat) & (vals != 0))
                        if not len(dy):
                            continue
                        gy, gx = dy.astype(np.int64)+by, dx.astype(np.int64)+bx
                        distances = (gy-y)**2 + (gx-x)**2
                        order = np.lexsort((gx, gy, distances))[:k]
                        best.extend((int(distances[i]), int(gy[i]), int(gx[i]), float(vals[dy[i], dx[i]]))
                                    for i in order)
                        best = sorted(best)[:k]
                outside = []
                if ya > 0:
                    outside.append((y-ya+1)**2)
                if yb < height:
                    outside.append((yb-y)**2)
                if xa > 0:
                    outside.append((x-xa+1)**2)
                if xb < width:
                    outside.append((xb-x)**2)
                if len(best) == k and (not outside or best[-1][0] < min(outside)):
                    target_data[ly, lx] = np.median(np.array([v[3] for v in best], dtype=np.float32))
                    break
                radius *= 2
            done += 1
            if progress is not None and (done == 1 or done % 100 == 0 or done == target_count):
                progress(done, target_count)
        result.write(0, ty0, tx0, target_data)
    return target_count


def inpaint_global(backend, result, saturation, plan, cancel, progress):
    height, width = result.shape[1:]
    # Includes masks, coordinates in both float32/float64, KD-tree nodes,
    # gathered k-neighbor values, and temporary boolean/fancy-index copies.
    # Saturation-heavy cases retain k=10 float64 distances/int64 indices plus
    # gathered values and median copies. Reserve for that worst case as well.
    estimate = 384 * height * width
    room = plan["memory_budget_bytes"] - plan["fixed_bytes"] - 16*MIB
    if estimate <= room:
        print("Saturation inpainting: original global nearest-neighbor method.", flush=True)
        cancel()
        array = result.read(0, 0, height, 0, width)
        mask = saturation.read(0, 0, height, 0, width).astype(bool)
        array = backend._inpaint_with_k_median(array, mask, k=10)
        cancel()
        result.write(0, 0, 0, array)
        return "stock_global"
    print("Saturation inpainting: bounded global nearest-neighbor search. "
          "Equal-distance ties use row/column order and may differ from the original KD-tree.", flush=True)
    bounded_global_inpaint(result, saturation, cancel, progress)
    return "bounded_global_ties_by_coordinate"


def restore_dtype(array, dtype, offset, minimum, maximum):
    if offset != 0:
        array = array - float(offset)
    array = np.clip(array, float(minimum), float(maximum))
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        return np.rint(array).clip(limits.min, limits.max).astype(dtype)
    return array.astype(dtype)


def _tiff_metadata(manifest, channels):
    names = manifest["channel_names"]
    origin = manifest.get("origin_xy", [0, 0])
    calibration = manifest.get("pixel_size_um")
    description = {"origin_xy": origin, "pixel_size_um": calibration,
                   "channel_indices": [int(c)+1 for c in channels]}
    for key in ("full_dimensions", "source_origin_xy"):
        if key in manifest:
            description[key] = manifest[key]
    metadata = {"axes": "CYX", "Name": manifest["base_name"],
                "Description": json.dumps(description, ensure_ascii=True),
                "Channel": {"Name": [names[c] for c in channels]}}
    if manifest.get("channel_colors") is not None:
        # OME colors use signed RGBA, whereas QuPath supplies RGB integers.
        colors = []
        for c in channels:
            rgba = ((int(manifest["channel_colors"][c]) & 0xffffff) << 8) | 255
            colors.append(rgba if rgba < 2**31 else rgba-2**32)
        metadata["Channel"]["Color"] = colors
    if calibration is not None:
        px, py = map(float, calibration)
        metadata.update(PhysicalSizeX=px, PhysicalSizeY=py,
                        PhysicalSizeXUnit="µm", PhysicalSizeYUnit="µm")
        metadata["Plane"] = {
            "PositionX": [float(origin[0])*px]*len(channels),
            "PositionY": [float(origin[1])*py]*len(channels),
            "PositionXUnit": ["µm"]*len(channels),
            "PositionYUnit": ["µm"]*len(channels)}
    return metadata


def tiff_tile_shape(height, width):
    return tuple(min(256, ((int(v)+15)//16)*16) for v in (height, width))


def tiff_storage_estimate(height, width, planes, dtype):
    """Conservative uncompressed size including edge padding and tile tables."""
    th, tw = tiff_tile_shape(height, width)
    tile_count = math.ceil(height/th)*math.ceil(width/tw)*int(planes)
    return tile_count*(th*tw*np.dtype(dtype).itemsize+32) + int(planes)*8192 + MIB


def write_streaming_tiff(path, source, channels, manifest, cancel, tick):
    """Stream fixed-size TIFF tiles; no stack/page-sized output allocation."""
    height, width = source.shape[1:]
    th, tw = tiff_tile_shape(height, width)
    encoded_estimate = tiff_storage_estimate(height, width, len(channels), source.dtype)

    def generate():
        for channel in channels:
            for y0 in range(0, height, th):
                for x0 in range(0, width, tw):
                    y1, x1 = min(height, y0+th), min(width, x0+tw)
                    cancel()
                    piece = source.read(int(channel), y0, y1, x0, x1)
                    if piece.shape != (th, tw):
                        padded = np.zeros((th, tw), dtype=source.dtype)
                        padded[:y1-y0, :x1-x0] = piece
                        piece = padded
                    yield piece
                    tick((y1-y0)*(x1-x0)*source.dtype.itemsize)
    kwargs = {}
    calibration = manifest.get("pixel_size_um")
    if calibration is not None:
        kwargs.update(resolution=(10000/float(calibration[0]), 10000/float(calibration[1])),
                      resolutionunit="CENTIMETER")
    with TiffWriter(str(path), bigtiff=encoded_estimate >= int(3.8*1024**3), ome=True) as writer:
        writer.write(generate(), shape=(len(channels), height, width), dtype=source.dtype,
                     tile=(th, tw), photometric="minisblack", compression=None,
                     metadata=_tiff_metadata(manifest, channels), **kwargs)
    cancel()


def _sync_directory(path):
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_publication_journal(path, journal):
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(journal, stream, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def recover_publication(scratch_dir):
    """Recover an interrupted publication, or clean a committed journal.

    Failures raise and deliberately retain the journal and remaining backups
    for the caller's retry. The journal must be recovered before a launcher
    recursively removes its private working directory.
    """
    scratch_dir = Path(scratch_dir).absolute()
    path = scratch_dir/"publication.json"
    if not path.exists():
        return {"recovered": False, "state": None}
    journal = json.loads(path.read_text(encoding="utf-8"))
    if journal.get("version") != 1 or journal.get("state") not in ("pending", "committed"):
        raise ValueError(f"Unrecognized publication journal; preserve {scratch_dir} for recovery.")
    output_dir = Path(journal["output_dir"])
    backup_dir = Path(journal["backup_dir"])
    if not output_dir.is_absolute() or backup_dir.parent != scratch_dir or not backup_dir.name.startswith(".spillot-publication-backups-"):
        raise ValueError("Publication recovery paths are invalid; the journal has been preserved.")
    entries = journal["entries"]
    for entry in entries:
        destination, backup, staged = map(Path, (entry["destination"], entry["backup"], entry["staged"]))
        if (destination.parent != output_dir or backup.parent != backup_dir
                or not staged.is_absolute() or not staged.is_relative_to(scratch_dir)
                or not isinstance(entry["existed"], bool)):
            raise ValueError("Publication recovery entry is invalid; the journal has been preserved.")
    if journal["state"] == "pending":
        for entry in reversed(entries):
            destination, backup, staged = map(Path, (entry["destination"], entry["backup"], entry["staged"]))
            if entry["existed"]:
                if backup.exists() or backup.is_symlink():
                    os.replace(backup, destination)
                elif not (destination.exists() or destination.is_symlink()):
                    raise IOError(f"Neither original nor backup exists for {destination}; preserve {scratch_dir}.")
                # A missing backup plus an existing destination is either an
                # untouched old output or an already-restored recovery entry.
            elif not staged.exists():
                destination.unlink(missing_ok=True)
        _sync_directory(output_dir)
    # Do not delete the intention journal until restoration/commit is complete.
    # Repeated recovery is safe if interrupted during this cleanup.
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    path.unlink()
    path.with_name(path.name+".tmp").unlink(missing_ok=True)
    _sync_directory(scratch_dir)
    return {"recovered": True, "state": journal["state"], "entries": len(entries)}


def publish_outputs(staged, output_dir, temporary, cancel, scratch_dir=None):
    """Publish with a durable rollback journal that survives hard termination.

    After success, the committed journal remains until recover_publication is
    called for cleanup. The process_manifest caller does that before reporting
    completion; a launcher can do the same after a forced process termination.
    """
    output_dir, temporary = Path(output_dir).absolute(), Path(temporary).absolute()
    scratch_dir = Path(scratch_dir or temporary.parent).absolute()
    journal_path = scratch_dir/"publication.json"
    if journal_path.exists():
        raise IOError(f"A publication journal already exists; recover {scratch_dir} before publishing.")
    backup_dir = Path(tempfile.mkdtemp(prefix=".spillot-publication-backups-", dir=scratch_dir))
    entries = []
    try:
        for index, source in enumerate(staged):
            source = Path(source).absolute()
            destination = output_dir/source.name
            if destination.is_dir():
                raise ValueError(f"An output filename refers to a directory: {destination}")
            entries.append(dict(destination=str(destination), staged=str(source),
                                backup=str(backup_dir/f"{index:04d}.previous"),
                                existed=destination.exists() or destination.is_symlink()))
        journal = dict(version=1, state="pending", output_dir=str(output_dir),
                       backup_dir=str(backup_dir), entries=entries)
        _write_publication_journal(journal_path, journal)
        for entry in entries:
            cancel()
            if entry["existed"]:
                os.replace(entry["destination"], entry["backup"])
                _sync_directory(backup_dir)
                _sync_directory(output_dir)
            os.replace(entry["staged"], entry["destination"])
        _sync_directory(output_dir)
        cancel()
        journal["state"] = "committed"
        _write_publication_journal(journal_path, journal)
    except BaseException:
        if journal_path.exists():
            recover_publication(scratch_dir)  # On failure leave recovery artifacts intact.
        else:
            shutil.rmtree(backup_dir)
        raise
    return [entry["destination"] for entry in entries]


def validate_manifest(value):
    manifest = dict(value)
    required = ("input_raw", "shape", "dtype", "output_dir", "base_name", "csv_path", "channel_names")
    for key in required:
        if key not in manifest:
            raise ValueError("Manifest is missing " + key)
    shape = manifest["shape"]
    if len(shape) != 3 or any(int(v) != v or int(v) < 1 for v in shape):
        raise ValueError("Manifest shape must contain positive integers [channels, height, width].")
    manifest["shape"] = list(map(int, shape))
    if manifest["dtype"] not in ("<u2", "<f4", "|u1"):
        raise ValueError("Manifest dtype must be <u2, <f4, or |u1.")
    if len(manifest["channel_names"]) != shape[0]:
        raise ValueError("The manifest must contain one channel name per input channel.")
    manifest["channel_names"] = [str(name) for name in manifest["channel_names"]]
    if "channel_colors" in manifest and len(manifest["channel_colors"]) != shape[0]:
        raise ValueError("The manifest must contain one color per input channel.")
    base = str(manifest["base_name"])
    if not base or base in (".", "..") or Path(base).name != base or "/" in base or "\\" in base:
        raise ValueError("base_name must be a filename without directory components.")
    manifest["base_name"] = base
    for key in ("input_raw", "output_dir", "csv_path"):
        path = Path(manifest[key]).expanduser()
        if not path.is_absolute():
            raise ValueError(key + " must be an absolute path.")
        manifest[key] = str(path)
    origin = manifest.get("origin_xy", [0, 0])
    if len(origin) != 2 or any(not math.isfinite(float(v)) for v in origin):
        raise ValueError("origin_xy must contain two finite source pixel coordinates.")
    manifest["origin_xy"] = origin
    calibration = manifest.get("pixel_size_um")
    if calibration is not None and (len(calibration) != 2 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in calibration)):
        raise ValueError("pixel_size_um must be null or two positive finite numbers.")
    return manifest


def process_manifest(path, tile_size=None, batch_size=None, seed=None,
                     memory_budget_mb=None, cancel_event=None):
    """Process every target selected by a matrix row; return published paths."""
    cancel = CancelCheck(cancel_event)
    progress = Progress(cancel)
    progress.emit("initializing", 0, 1, 0, "Reading the low-memory export manifest.")
    manifest = validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))
    dtype = np.dtype(manifest["dtype"])
    channels, height, width = manifest["shape"]
    patch = int(manifest.get("patch_size", 24))
    budget = manifest.get("memory_budget_mb", 0) if memory_budget_mb is None else memory_budget_mb
    plan = plan_memory(manifest["shape"], dtype, patch, manifest.get("workers", 1),
                       budget, tile_size, batch_size)
    seed = int(seed) if seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
    if seed < 0:
        raise ValueError("Seed must be nonnegative.")
    print("Low-memory SpillOT plan: " + json.dumps(dict(plan, seed=seed)), flush=True)
    if plan["workers"] != plan["requested_workers"]:
        print(f"Using {plan['workers']} of {plan['requested_workers']} requested workers to fit the memory budget.", flush=True)
    selection = load_selection(Path(manifest["csv_path"]), channels, True)
    donors_by_channel = []
    for channel in range(channels):
        donors = np.flatnonzero(np.isclose(np.abs(selection[channel]), 1, atol=1e-6))
        donors_by_channel.append(donors[donors != channel])
    targets = [c for c in range(channels) if len(donors_by_channel[c])]
    output_dir = Path(manifest["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(manifest.get("scratch_dir", Path(path).resolve().parent)).expanduser()
    if not scratch_dir.is_absolute():
        raise ValueError("scratch_dir must be an absolute path.")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    if os.stat(scratch_dir).st_dev != os.stat(output_dir).st_dev:
        raise ValueError("scratch_dir and output_dir must be on the same filesystem for atomic publication.")
    cancel.cancel_path = Path(manifest.get("cancel_path", scratch_dir/"cancel.requested"))
    recover_publication(scratch_dir)
    cancel()
    input_size = channels*height*width*dtype.itemsize
    extra_disk = (input_size + 5*height*width
                  + tiff_storage_estimate(height, width, 2*channels+len(targets), dtype)
                  + (len(targets)+2)*MIB + Path(manifest["csv_path"]).stat().st_size)
    if shutil.disk_usage(output_dir).free < int(extra_disk*1.10) + 16*MIB:
        raise ValueError("Not enough free disk space for temporary data and complete TIFF outputs.")
    raw = RawArray(manifest["input_raw"], manifest["shape"], dtype, cancel=cancel)
    pool = None
    failed = True
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix=".spillot-tmp-", dir=scratch_dir) as temporary_name:
            temporary = Path(temporary_name)
            io_tile = min(plan["tile_size"], 512)
            minimum = np.full(channels, np.inf, dtype=np.float64)
            maximum = np.full(channels, -np.inf, dtype=np.float64)
            offsets = np.zeros(channels, dtype=np.float32)
            out = RawArray(temporary/"restored.raw", manifest["shape"], dtype, create=True, cancel=cancel)
            try:
                scanned = 0
                for channel in range(channels):
                    for y0, y1, x0, x1 in tiles(height, width, io_tile):
                        array = raw.read(channel, y0, y1, x0, x1)
                        if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
                            raise ValueError("The input contains NaN or infinite intensities; replace these before running SpillOT.")
                        minimum[channel] = min(minimum[channel], float(array.min()))
                        maximum[channel] = max(maximum[channel], float(array.max()))
                        out.write(channel, y0, x0, array)
                        scanned += array.size
                        progress.emit("scanning", scanned, channels*height*width,
                                      0.04*scanned/(channels*height*width),
                                      f"Preparing channel {channel+1}/{channels} and measuring global intensity ranges.")
                    offsets[channel] = max(0, -np.float32(minimum[channel]))
                backend = load_backend(DEFAULT_BACKEND)
                configure_graph(backend, patch)
                if targets and plan["workers"] > 1:
                    pool = cf.ProcessPoolExecutor(max_workers=plan["workers"],
                        mp_context=mp.get_context("spawn"), initializer=_init_worker,
                        initargs=(str(DEFAULT_BACKEND), patch))
                positions_per_channel = 0
                tile_count = math.ceil(height/plan["tile_size"])*math.ceil(width/plan["tile_size"])
                for core in tiles(height, width, plan["tile_size"]):
                    ya, yb, xa, xb = patch_bounds(core, height, width, patch)
                    positions_per_channel += (yb-ya)*(xb-xa)
                total_work = positions_per_channel * sum(1+len(donors_by_channel[c]) for c in targets)
                completed_work = 0
                work_started = time.monotonic()
                report_channels = []
                for channel in targets:
                    donors = donors_by_channel[channel]
                    result = RawArray(temporary/"channel-float.raw", (1, height, width), "<f4", create=True, cancel=cancel)
                    saturation = None
                    sat_count = 0
                    channel_started = time.monotonic()
                    stats = {"channel": channel+1, "donor_channels": (donors+1).tolist(),
                             "patches_processed": 0, "removed_patch_visits": 0,
                             "global_patch_count": (height-patch+1)*(width-patch+1)}
                    try:
                        if manifest.get("ignore_overexposed", False) and np.issubdtype(dtype, np.integer):
                            progress.emit("saturation", 0, height*width, progress.fraction,
                                          f"Finding saturated pixels for channel {channel+1}.")
                            saturation, sat_count = make_saturation_mask(raw, temporary/"saturation.raw",
                                np.unique(np.concatenate((donors, [channel]))), io_tile, cancel)
                        def tick(amount):
                            nonlocal completed_work
                            completed_work += amount
                            elapsed = time.monotonic()-work_started
                            eta = elapsed*(total_work-completed_work)/completed_work if elapsed >= 2 and completed_work else None
                            progress.emit("processing", completed_work, total_work,
                                0.04+0.84*completed_work/max(1, total_work),
                                f"Correcting channel {channel+1}/{channels}; tile {tile_index}/{tile_count}; "
                                f"{plan['workers']} worker(s).", eta)
                        for tile_index, core in enumerate(tiles(height, width, plan["tile_size"]), 1):
                            count, removed = process_core(backend, pool, raw, result, channel, donors,
                                offsets, saturation, core, patch, plan["batch_size"], seed, cancel, tick)
                            stats["patches_processed"] += count
                            stats["removed_patch_visits"] += removed
                        if sat_count:
                            progress.emit("inpainting", 0, sat_count, progress.fraction,
                                          f"Restoring {sat_count} saturated pixels in channel {channel+1}.")
                            def inpaint_tick(done, total):
                                progress.emit("inpainting", done, total, progress.fraction,
                                              f"Restoring saturated pixels in channel {channel+1} with a global search.")
                            stats["inpaint_method"] = inpaint_global(backend, result, saturation, plan, cancel, inpaint_tick)
                        for y0, y1, x0, x1 in tiles(height, width, io_tile):
                            array = result.read(0, y0, y1, x0, x1)
                            array = restore_dtype(array, dtype, offsets[channel], minimum[channel], maximum[channel])
                            out.write(channel, y0, x0, array)
                        stats["seconds"] = time.monotonic()-channel_started
                        stats["completed"] = True
                        report_channels.append(stats)
                    finally:
                        result.close()
                        if saturation is not None:
                            saturation.close()
                        (temporary/"channel-float.raw").unlink(missing_ok=True)
                        (temporary/"saturation.raw").unlink(missing_ok=True)
                _stop_pool(pool, False)
                pool = None
                base_name = manifest["base_name"]
                jobs = [(temporary/(base_name+".tif"), raw, list(range(channels)))]
                jobs.extend((temporary/(base_name+f"_Channel_{c+1}_SpillOT.tif"), out, [c]) for c in targets)
                jobs.append((temporary/(base_name+"_SpillOT.tif"), out, list(range(channels))))
                total_bytes = (2*channels+len(targets))*height*width*dtype.itemsize
                saved = 0
                progress.emit("finalizing", 0, total_bytes, 0.90, "Writing TIFF outputs; the run is not complete yet.")
                def saved_tick(amount):
                    nonlocal saved
                    saved += amount
                    progress.emit("finalizing", saved, total_bytes, 0.90+0.09*saved/max(1, total_bytes),
                                  "Writing complete TIFF outputs to temporary files.")
                for destination, source, selected in jobs:
                    write_streaming_tiff(destination, source, selected, manifest, cancel, saved_tick)
                staged_csv = temporary/(base_name+".csv")
                shutil.copyfile(manifest["csv_path"], staged_csv)
                staged_paths = [j[0] for j in jobs] + [staged_csv]
                progress.emit("publishing", 0, len(staged_paths), 0.995,
                              "Publishing the completed TIFF outputs and channel matrix.")
                output_files = publish_outputs(staged_paths, output_dir, temporary, cancel, scratch_dir)
                recover_publication(scratch_dir)
            finally:
                out.close()
        failed = False
        report = dict(completed=True, output_files=output_files, seed=seed, plan=plan,
                      channels=report_channels, elapsed_seconds=time.monotonic()-started)
        # All output writers are closed and publication succeeded before 100%.
        progress.emit("complete", 1, 1, 1, "SpillOT finished; all TIFF outputs are saved.", 0)
        return report
    finally:
        _stop_pool(pool, failed)
        raw.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--memory-budget-mb", type=float)
    args = parser.parse_args(argv)
    event = threading.Event()
    def stop(signum, frame):
        event.set()
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, stop)
    try:
        report = process_manifest(args.manifest, args.tile_size, args.batch_size, args.seed,
                                  args.memory_budget_mb, event)
        print("SPILLOT_RESULT\t" + json.dumps(report), flush=True)
        return 0
    except CooperativeCancel as error:
        print(str(error), file=sys.stderr, flush=True)
        return 130
    except Exception as error:
        print("SpillOT failed: " + str(error), file=sys.stderr, flush=True)
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

