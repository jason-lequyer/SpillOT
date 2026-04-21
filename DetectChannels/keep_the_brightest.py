# Copyright 2025, Jason Lequyer, Ferris Nowlan and Laurence Pelletier.
# All rights reserved.
# Sinai Health System - Lunenfeld Tanenbaum Research Institute
# 600 University Avenue, Room 1070, Toronto, ON M5G 1X5, Canada
#

import os, sys, time, platform
import numpy as np
from tifffile import imread, imwrite
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import concurrent.futures as cf
import multiprocessing as mp

from multiprocessing import shared_memory
# ---------- top of file (after imports) ----------
_SHM_HANDLES = []          # global registry to keep blocks alive
# -------------------------------------------------

def _invert_from_uint8(arr2d, minv, maxv, dtype_out):
    """
    Inverse of per-channel 8-bit scaling.
    arr2d: float32 or uint8 in [0,255], 2D (H,W).
    Returns 2D array in dtype_out with values mapped to [minv, maxv].
    """
    rng = float(maxv) - float(minv)
    arr = arr2d.astype(np.float32, copy=False)
    if rng > 0.0:
        arr = arr * (rng / 255.0) + float(minv)
    else:
        arr = np.zeros_like(arr, dtype=np.float32) + float(minv)

    if np.issubdtype(dtype_out, np.integer):
        info = np.iinfo(dtype_out)
        arr = np.rint(arr).clip(info.min, info.max).astype(dtype_out)
    else:
        arr = arr.astype(dtype_out)
    return arr

def _scale_to_uint8_per_channel(arr):
    """
    Per-channel 8-bit scaling:
      arr_c := arr_c - min(arr_c)
      arr_c := round(arr_c * 255 / max(arr_c))
      dtype -> uint8, clipped [0,255]
    Expects shape (C,H,W). Returns uint8 array (C,H,W).
    """
    a = arr.astype(np.float32, copy=False)
    mins = a.min(axis=(1, 2), keepdims=True)
    a -= mins
    highs = a.max(axis=(1, 2), keepdims=True)
    # Avoid divide-by-zero: where high==0, leave zeros
    scale = np.zeros_like(highs, dtype=np.float32)
    nz = highs > 0
    scale[nz] = 255.0 / highs[nz]
    a = np.rint(a * scale).clip(0, 255).astype(np.uint8)
    return a


def _to_shm(ndarray):
    shm = shared_memory.SharedMemory(create=True, size=ndarray.nbytes)
    view = np.ndarray(ndarray.shape, dtype=ndarray.dtype, buffer=shm.buf)
    view[:] = ndarray[:]
    return shm, view

def _init_solve_pool_shm(name_base, shape_base, dtype_base,
                         name_rf,   shape_rf,   dtype_rf,
                         *idx_pack_and_pat):
    import os, numpy as np
    from multiprocessing import shared_memory
    global base2d, rf, _SHM_HANDLES
    global top_idx, bot_idx, lef_idx, rit_idx
    global toplef_idx, toprit_idx, botlef_idx, botrit_idx, bord_idx
    global patsize

    # split out indices and patsize
    *idx_pack, pat = idx_pack_and_pat
    (top_idx, bot_idx, lef_idx, rit_idx,
     toplef_idx, toprit_idx, botlef_idx, botrit_idx,
     bord_idx) = idx_pack
    patsize = np.int32(pat)

    shm_b  = shared_memory.SharedMemory(name=name_base)
    shm_rf = shared_memory.SharedMemory(name=name_rf)
    _SHM_HANDLES.extend([shm_b, shm_rf])

    base2d = np.ndarray(shape_base, dtype_base, buffer=shm_b.buf)
    rf     = np.ndarray(shape_rf,   dtype_rf,   buffer=shm_rf.buf)

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

# -------- portable helpers (picklable) --------
def RP_pad2(x, pad=(2, 2, 2, 2)):
    return _reflection_pad2d(x, pad)

def unfold_p16(x):
    """Unfold with (patsize+4, patsize+4) kernel."""
    return _unfold4d(x, (patsize + 4, patsize + 4))

# ------------------------------------------------------------------
#  Initializer for the first ProcessPool (the one that calls _solve_chunk)
# ------------------------------------------------------------------
def _init_solve_pool(_base2d, _rf,
                     _top_idx, _bot_idx, _lef_idx, _rit_idx,
                     _toplef_idx, _toprit_idx, _botlef_idx, _botrit_idx,
                     _bord_idx):
    global base2d, rf
    global top_idx, bot_idx, lef_idx, rit_idx
    global toplef_idx, toprit_idx, botlef_idx, botrit_idx, bord_idx
    base2d, rf = _base2d, _rf
    (top_idx, bot_idx, lef_idx, rit_idx,
     toplef_idx, toprit_idx, botlef_idx, botrit_idx,
     bord_idx) = (_top_idx, _bot_idx, _lef_idx, _rit_idx,
                  _toplef_idx, _toprit_idx, _botlef_idx, _botrit_idx,
                  _bord_idx)

# ------------------------------------------------------------------
#  Initializer for the second ProcessPool (the one that calls _process_compare_channel)
# ------------------------------------------------------------------
def _init_compare_pool(_compare3d, _rf, _rfdom, _rfdommean,
                       _basediff, _trumeanbase, _pat):
    global compare3d, rf, rfdom, rfdommean
    global basediff, trumeanbase, RP, unfold, patsize
    compare3d, rf, rfdom, rfdommean = _compare3d, _rf, _rfdom, _rfdommean
    basediff, trumeanbase = _basediff, _trumeanbase
    patsize = np.int32(_pat)
    RP, unfold = RP_pad2, unfold_p16

# ───────────────────────────── helper functions ────────────────────────────────
def _solve_chunk(hj_vec: np.ndarray):
    out = []
    for hj in hj_vec:
        out.append(_solve_one_plane(int(hj)))
    return out

def _chunks(n_planes, grainsize=512):
    for start in range(0, n_planes, grainsize):
        yield np.arange(start, min(start + grainsize, n_planes), dtype=np.int64)

def _clone(a):                 # torch.clone
    return a.copy()

def _unsqueeze(a, dim):        # torch.unsqueeze
    return np.expand_dims(a, dim)

def _repeat(a, *reps):
    reps = tuple(int(r) for r in reps)
    if len(reps) > a.ndim:
        a = a.reshape((1,) * (len(reps) - a.ndim) + a.shape)
    elif len(reps) < a.ndim:
        reps = (1,) * (a.ndim - len(reps)) + reps
    return np.tile(a, reps)

def _ones_like(a, dtype=None):
    return np.ones_like(a, dtype=dtype or a.dtype)

def _zeros_like(a, dtype=None):
    return np.zeros_like(a, dtype=dtype or a.dtype)

def _cat(tensors, axis=0):
    return np.concatenate(tensors, axis=axis)

def _where(*args, **kwargs):
    return np.where(*args, **kwargs)

def _abs(x):
    return np.abs(x)

def _mean(x, axis=None):
    return np.mean(x, axis=axis)

def _reflection_pad2d(arr, pad=(2, 2, 2, 2)):
    l, r, t, b = pad
    return np.pad(
        arr,
        pad_width=((0, 0), (0, 0), (t, b), (l, r)),
        mode="reflect",
    )

def _unfold4d(x, ksize):
    B, C, H, W = x.shape
    kH, kW     = ksize
    win        = np.lib.stride_tricks.sliding_window_view(x, (kH, kW), axis=(2, 3))
    out_h, out_w = win.shape[2:4]
    win        = win.transpose(0, 1, 4, 5, 2, 3)
    patches    = win.reshape(B, C * kH * kW, out_h * out_w)
    return patches

def _fold4d(patches, output_size, ksize):
    B, CK2, L  = patches.shape
    kH, kW     = ksize
    C          = CK2 // (kH * kW)
    H, W       = output_size
    out        = np.zeros((B, C, H, W), dtype=patches.dtype)
    counter    = np.zeros_like(out)

    patches = patches.reshape(B, C, kH, kW, L)
    idx = 0
    for i in range(H - kH + 1):
        for j in range(W - kW + 1):
            out[:, :, i:i+kH, j:j+kW]     += patches[..., idx]
            counter[:, :, i:i+kH, j:j+kW] += 1
            idx += 1
    counter[counter == 0] = 1
    return out / counter

def _row_mask(mask2d):
    return mask2d[:, :, None, None]

def _col_mask(mask2d):
    return mask2d[None, None, :, :]

def _inflate_cost(curcost4d, mask_a2d, mask_b2d, big):
    h, w = mask_a2d.shape
    flat = curcost4d.reshape(h * w, h * w)
    rows = np.flatnonzero(mask_a2d.ravel())
    cols = np.flatnonzero(mask_b2d.ravel())
    n = min(len(rows), len(cols))
    if n:
        flat[rows[:n], cols[:n]] = big

# ░░░░░░░░░░░░░░░░░░░ constants ░░░░░░░░░░░░░░░░░░
FLOAT32_MAX      = np.float32(np.finfo(np.float32).max)
FLOAT32_MAX_SQRT = np.sqrt(FLOAT32_MAX)
FLOAT32_MAX_4RT  = np.sqrt(np.sqrt(FLOAT32_MAX))
FLOAT32_MAX_8RT  = np.sqrt(np.sqrt(np.sqrt(FLOAT32_MAX)))
FLOAT32_MIN      = np.float32(np.finfo(np.float32).tiny)
DEFAULT_PATSIZE = 16
patsize = np.int32(DEFAULT_PATSIZE)

def _set_patsize(p):
    """Set global patch size (int) everywhere."""
    global patsize
    patsize = np.int32(int(p))

varkern = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]], dtype=np.float32)

def _process_compare_channel(ow: int) -> np.ndarray:
    curcomp = RP(compare3d[:, ow:ow+1, :, :])
    curcomp = unfold(curcomp)
    curcomp = curcomp.reshape(-1, patsize+4, patsize+4,
                              curcomp.shape[-1])[:, 2:-2, 2:-2, :]

    rfdom_local = rfdom.copy()
    side = int(np.sqrt(rfdom_local.shape[1]))
    rfdom_local = rfdom_local.reshape(4, side, side, -1).astype(np.float32)
    rfdom_local[0] = curcomp[0, 2:,    1:-1, :]
    rfdom_local[1] = curcomp[0, :-2,   1:-1, :]
    rfdom_local[2] = curcomp[0, 1:-1,  2:,   :]
    rfdom_local[3] = curcomp[0, 1:-1, :-2,   :]

    rfdom_local = rfdom_local.reshape(4, -1, rfdom_local.shape[-1])

    rfdomstd = (_abs(rfdom_local[0] - rfdom_local[1]) +
                _abs(rfdom_local[0] - rfdom_local[2]) +
                _abs(rfdom_local[0] - rfdom_local[3]) +
                _abs(rfdom_local[1] - rfdom_local[2]) +
                _abs(rfdom_local[1] - rfdom_local[3]) +
                _abs(rfdom_local[2] - rfdom_local[3]))

    rfdommean = _unsqueeze(_mean(rfdomstd, axis=0), 0)
    rfdommean[rfdommean == 0] = 1
    rfdomstd /= rfdommean

    compdiff = np.sum(_abs(rf - rfdomstd), axis=0)

    curcomp = curcomp.reshape(1, -1, curcomp.shape[-1])
    trumeancomp = _unsqueeze(_mean(curcomp, axis=1), 1)
    trumeancomp[trumeancomp == 0] = FLOAT32_MIN

    whurdif = np.where((compdiff < basediff) &
                       (trumeanbase[0, 0, :] < trumeancomp[0, 0, :]))[0]
    return whurdif

def _solve_one_plane(hj: int):
    curpat = _clone(base2d[0, :, :, hj])
    curpat = _repeat(curpat, patsize+4, patsize+4, 1, 1)

    curcost         = _zeros_like(curpat)
    curcost[top_idx] = _abs(curpat[toprit_idx] - curpat[rit_idx]) + FLOAT32_MIN
    curcost[bot_idx] = _abs(curpat[botlef_idx] - curpat[lef_idx]) + FLOAT32_MIN
    curcost[lef_idx] = _abs(curpat[toplef_idx] - curpat[top_idx]) + FLOAT32_MIN
    curcost[rit_idx] = _abs(curpat[botrit_idx] - curpat[bot_idx]) + FLOAT32_MIN

    totcost = np.stack([curcost[top_idx],
                        curcost[bot_idx],
                        curcost[lef_idx],
                        curcost[rit_idx]])
    sort     = np.argsort(totcost, axis=0)
    maxcost, mincost, ndcost, rdcost = sort[3], sort[0], sort[1], sort[2]

    topmax = (_zeros_like(maxcost) + FLOAT32_MAX_8RT) * (maxcost == 0)
    botmax = (_zeros_like(maxcost) + FLOAT32_MAX_8RT) * (maxcost == 1)
    lefmax = (_zeros_like(maxcost) + FLOAT32_MAX_8RT) * (maxcost == 2)
    ritmax = (_zeros_like(maxcost) + FLOAT32_MAX_8RT) * (maxcost == 3)

    topmin = (mincost == 0).astype(np.float32)
    botmin = (mincost == 1).astype(np.float32)
    lefmin = (mincost == 2).astype(np.float32)
    ritmin = (mincost == 3).astype(np.float32)

    topnd = (ndcost == 0).astype(np.float32)
    botnd = (ndcost == 1).astype(np.float32)
    lefnd = (ndcost == 2).astype(np.float32)
    ritnd = (ndcost == 3).astype(np.float32)

    toprd = (rdcost == 0).astype(np.float32)
    botrd = (rdcost == 1).astype(np.float32)
    lefrd = (rdcost == 2).astype(np.float32)
    ritrd = (rdcost == 3).astype(np.float32)

    curcost[top_idx] = curcost[bot_idx] = curcost[lef_idx] = curcost[rit_idx] = 0

    curcost[top_idx] += topmax + topmin + topnd + toprd
    curcost[bot_idx] += botmax + botmin + botnd + botrd
    curcost[lef_idx] += lefmax + lefmin + lefnd + lefrd
    curcost[rit_idx] += ritmax + ritmin + ritnd + ritrd

    curcost[top_idx] += (np.random.rand(curcost[top_idx].shape[0]) / 10).astype(np.float32)
    curcost[bot_idx] += (np.random.rand(curcost[bot_idx].shape[0]) / 10).astype(np.float32)
    curcost[lef_idx] += (np.random.rand(curcost[lef_idx].shape[0]) / 10).astype(np.float32)
    curcost[rit_idx] += (np.random.rand(curcost[rit_idx].shape[0]) / 10).astype(np.float32)

    curcost[bord_idx] = FLOAT32_MAX_4RT

    rl1 = rf[0, :, :, hj] == 2
    rr1 = rf[1, :, :, hj] == 2
    rt1 = rf[2, :, :, hj] == 2
    rb1 = rf[3, :, :, hj] == 2

    if rl1.any():
        _inflate_cost(curcost, rf[0, :, :, hj] == 1, rl1, FLOAT32_MAX_SQRT)
    if rr1.any():
        _inflate_cost(curcost, rf[1, :, :, hj] == 1, rr1, FLOAT32_MAX_SQRT)
    if rt1.any():
        _inflate_cost(curcost, rf[2, :, :, hj] == 1, rt1, FLOAT32_MAX_SQRT)
    if rb1.any():
        _inflate_cost(curcost, rf[3, :, :, hj] == 1, rb1, FLOAT32_MAX_SQRT)

    curcost = curcost[1:-1, 1:-1, 1:-1, 1:-1]
    curpat  = curpat [1:-1, 1:-1, 1:-1, 1:-1]

    curcost = curcost.reshape(curcost.shape[0]*curcost.shape[1], -1)
    curpat  = curpat.reshape(curpat.shape[0]*curpat.shape[1], -1)

    curcost[curcost == 0] = FLOAT32_MAX
    row_ind, col_ind = linear_sum_assignment(curcost.astype(np.float32))

    newpat = _clone(curpat[0, :])
    newpat[row_ind] = curpat[0, col_ind]
    return hj, newpat.reshape(patsize+2, patsize+2)

def _safe_imwrite(path, array, **kwargs):
    try:
        imwrite(path, array, **kwargs)
    except ValueError as e:
        print("imwrite failed ({}). Retrying as float32 ...".format(e))
        imwrite(path, array.astype(np.float32), **kwargs)

# ---------- NEW: overexposed handling + inpaint ----------
def _zero_overexposed_for_run(base2d_f32, compare3d_f32, raw_u, check_idxs, verbose=False):
    """
    Zero saturated pixels for this run and return the saturation mask (H,W) or None.
    """
    dt = raw_u.dtype
    if not (np.issubdtype(dt, np.integer)):
        return None  # float images: no saturation concept, skip

    maxv = np.iinfo(dt).max
    check_idxs = np.unique(np.asarray(check_idxs, dtype=np.int64))
    if check_idxs.size == 0:
        return np.zeros(raw_u.shape[1:], dtype=bool)

    sat_any = (raw_u[check_idxs] == maxv).any(axis=0)  # (H,W) bool
    if sat_any.any():
        if verbose:
            print("ignore_overexposed: zeroing {} saturated pixels (dtype max = {})."
                  .format(int(sat_any.sum()), maxv))
        base2d_f32[0][sat_any] = 0.0
        compare3d_f32[:] = np.where(sat_any[None, :, :], 0.0, compare3d_f32)
    return sat_any

def _inpaint_with_k_median(arr2d, sat_mask, k=10):
    """
    Fill pixels that were zeroed by saturation handling using median of k nearest
    non-zero, non-saturated pixels in arr2d.
    - arr2d: float32 (H,W)
    - sat_mask: bool (H,W) with True where saturation was detected
    """
    if sat_mask is None or not sat_mask.any():
        return arr2d

    # Only fill those positions that (a) were saturated and (b) are zero now.
    target_mask = sat_mask & (arr2d == 0)
    if not target_mask.any():
        return arr2d

    # Donors: not saturated and currently non-zero
    donors_mask = (~sat_mask) & (arr2d != 0)
    if not donors_mask.any():
        # Fallback: global median of non-zero pixels (if any)
        nz = arr2d[arr2d != 0]
        fill_val = float(np.median(nz)) if nz.size else 0.0
        arr2d[target_mask] = fill_val
        return arr2d

    donors_coords = np.column_stack(np.nonzero(donors_mask))  # (N,2) -> [row, col]
    donors_vals   = arr2d[donors_mask].astype(np.float32)

    targets_coords = np.column_stack(np.nonzero(target_mask))
    k_eff = int(min(k, donors_coords.shape[0]))

    tree = cKDTree(donors_coords.astype(np.float32))
    # Query all targets at once
    dists, idxs = tree.query(targets_coords.astype(np.float32), k=k_eff)

    # If k_eff == 1, idxs has shape (M,), make it 2D for consistent median
    if k_eff == 1:
        fill_vals = donors_vals[idxs]
    else:
        fill_vals = np.median(donors_vals[idxs], axis=1)

    # Assign fills
    arr2d[targets_coords[:, 0], targets_coords[:, 1]] = fill_vals.astype(arr2d.dtype, copy=False)
    return arr2d

# ─────────────────────────────────── main ───────────────────────────────────────
if __name__ == "__main__":
    import argparse

    mp.freeze_support()
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn" if platform.system() == "Windows" else "fork", force=True)

    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        description="Debleed TIFF stack; optionally one channel."
    )
    parser.add_argument("image", help="Path to the input TIFF")
    parser.add_argument("channel", nargs="?", type=int,
                        help="1-based channel to debleed (omit for all channels)")
    parser.add_argument("-p", "--patsize", type=int, default=DEFAULT_PATSIZE,
                        help="Patch size (default: 16)")
    parser.add_argument("--ignore_overexposed", action="store_true",
                        help="If set, pixels saturated at dtype max in the base or any compared channels are zeroed before processing (per run), then inpainted afterward.")
    parser.add_argument("--opal_vectra", action="store_true",
                    help="Per-channel 8-bit scaling before debleeding "
                         "(min-subtract, scale to max=255, round, uint8).")

    args = parser.parse_args()

    file_name = args.image
    single_chan = (args.channel - 1) if args.channel is not None else None
    _set_patsize(args.patsize)
    ignore_overexposed = bool(args.ignore_overexposed)
    opal_vectra = bool(args.opal_vectra)


    # Legacy argv handling kept for compatibility with positional-only callers
    if len(sys.argv) < 2:
        print("Usage: {} <image.tif> [channel]".format(sys.argv[0]))
        sys.exit(1)
    file_name = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        single_chan = int(sys.argv[2]) - 1 if args.channel is None else single_chan

    # read input once
    raw = imread(file_name)
    swapped_rgb = False
    if raw.shape[-1] == 3:  # RGB last -> (3,H,W)
        raw = np.swapaxes(raw, -2, -1)
        raw = np.swapaxes(raw, -3, -2)
        swapped_rgb = True
    
    # Keep an untouched copy for saturation checks and for restoring dtype/range
    raw_orig   = raw.copy()
    orig_dtype = raw_orig.dtype
    orig_mins  = raw_orig.min(axis=(1, 2))  # per-channel mins
    orig_maxs  = raw_orig.max(axis=(1, 2))  # per-channel maxs
    
    # If opal_vectra: temporary per-channel scaling to uint8 for processing only
    if opal_vectra:
        print("Opal Vectra scaling: per-channel min-subtract + rescale to 8-bit (0..255) for processing only.")
        raw_proc = _scale_to_uint8_per_channel(raw_orig)
    else:
        raw_proc = raw_orig
    
    # Proceed with processing on float32 working copy of raw_proc
    n_chan, H, W = raw_proc.shape
    inpnorm      = raw_proc.astype(np.float32)
    
    # IMPORTANT: Allocate output in the ORIGINAL dtype and shape
    outstack     = np.empty_like(raw_orig)



    # optional CSV inclusion/exclusion mask (read once)
    # Legacy DetectChannels matrix semantics:
    #   1 or -1 => keep / allow comparison
    #   0, blank, NaN, or any other value => exclude / veto comparison
    mask_matrix = None
    try:
        csv_path = os.path.splitext(file_name)[0] + ".csv"
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            mask_matrix = np.genfromtxt(
                f,
                delimiter=",",
                skip_header=1,
                usecols=tuple(range(1, n_chan + 1)),
            )
        mask_matrix = np.atleast_2d(mask_matrix).astype(np.float32, copy=False)
        mask_matrix[np.isnan(mask_matrix)] = 0.0
    except Exception as e:
        print("CSV not found or unusable:", e, "\nProceeding without mask.")

    chan_iter = [single_chan] if single_chan is not None else range(n_chan)
    total_start = time.time()

    # ───────────────────────── channel loop ──────────────────────────
    for oz in chan_iter:
        print("\n Debleeding channel {}/{}".format(oz+1, n_chan))
        start_time = time.time()

        # working copies (float32)
        base2d_f32     = _clone(inpnorm[oz:oz+1, :, :])  # (1,H,W)
        compare3d_f32  = _clone(inpnorm)                 # (C,H,W)

        # inclusion set from CSV (if any)
        included = None
        if mask_matrix is not None:
            row = mask_matrix[oz]
            # Only explicit +1/-1 cells are allowed donor/comparison channels.
            # 0, blanks/NaNs, and other numeric values are excluded/vetoed.
            included = np.where(np.isclose(np.abs(row), 1.0, rtol=0.0, atol=1e-6))[0]

            # never use the channel itself as a donor, even if the CSV says 1
            included = included[included != oz]

            if included.size == 0:
                # No allowed donors for this channel -> skip debleeding
                print("Mask row for channel {} has no allowed donors (no explicit +1/-1 allowed channels, or only self); "
                      "skipping debleed for this channel."
                      .format(oz + 1))
                # Leave this channel exactly as the original data
                outstack[oz] = raw_orig[oz]
                continue



        # union of channels to inspect for saturation for THIS run
        if included is None:
            check_idxs = np.arange(n_chan, dtype=np.int64)
        else:
            check_idxs = np.unique(np.concatenate([included, np.array([oz], dtype=np.int64)]))

        sat_mask = None
        if ignore_overexposed:
            sat_mask = _zero_overexposed_for_run(base2d_f32, compare3d_f32, raw_orig, check_idxs, verbose=True)

        # shape to (B=1,C,H,W) for the downstream code
        base2d    = _unsqueeze(base2d_f32, 0)     # (1,1,H,W)
        compare3d = _unsqueeze(compare3d_f32, 0)  # (1,C,H,W)

        # keep a copy of the current base channel for later steps
        curcomp   = _clone(compare3d[:, oz:oz+1, :, :])

        if included is not None:
            compare3d    = compare3d[:, included, :, :]
        else:
            # no CSV mask at all: allow all *other* channels as donors, but not self
            all_idxs = np.arange(n_chan, dtype=np.int64)
            donor_idxs = all_idxs[all_idxs != oz]
            compare3d    = compare3d[:, donor_idxs, :, :]

        RP, unfold = RP_pad2, unfold_p16
        fold   = lambda x: _fold4d(x, (H, W), (patsize, patsize))

        dummy  = _ones_like(base2d)
        base2d = RP(base2d)
        dummy  = RP(dummy)

        # reflection masks
        rflef = _zeros_like(base2d)
        rfrit = _zeros_like(base2d)
        rftop = _zeros_like(base2d)
        rfbot = _zeros_like(base2d)

        rflef[0, 0, 2:-2, 1]  = 2
        rflef[0, 0, 2:-2, 2]  = 1
        rfrit[0, 0, 2:-2, -2] = 2
        rfrit[0, 0, 2:-2, -3] = 1
        rftop[0, 0, 1, 2:-2]  = 2
        rftop[0, 0, 2, 2:-2]  = 1
        rfbot[0, 0, -2, 2:-2] = 2
        rfbot[0, 0, -3, 2:-2] = 1

        evens = np.zeros((patsize+4, patsize+4), dtype=np.float32)
        odds  = np.zeros_like(evens)
        for i in range(1, evens.shape[0] - 1):
            for j in range(1, evens.shape[1] - 1):
                if i in (1, evens.shape[0]-2) or j in (1, evens.shape[1]-2):
                    (evens if (i+j)%2 == 0 else odds)[i, j] = 1

        rf      = _cat([rflef, rfrit, rftop, rfbot], axis=0)
        base2d  = unfold(base2d)
        dummy   = unfold(dummy)
        rf      = unfold(rf)

        odds_idx  = _where(odds > 0)
        evens_idx = _where(evens > 0)

        base2d = base2d.reshape(-1, patsize+4, patsize+4, base2d.shape[-1])
        dummy  = dummy.reshape(-1, patsize+4, patsize+4, base2d.shape[-1])
        rf     = rf.reshape(-1, patsize+4, patsize+4, base2d.shape[-1])

        dom2d  = _clone(base2d)

        # shift RF masks
        for idx in range(4):
            tmp = _clone(rf[idx, 2:-2, 1:-1, :]) if idx < 2 else _clone(rf[idx, 1:-1, 2:-2, :])
            rf[idx, :, :, :] *= 0
            if idx < 2:
                rf[idx, 2:-2, 1:-1, :] = tmp
            else:
                rf[idx, 1:-1, 2:-2, :] = tmp

        shp = (patsize+4, patsize+4, patsize+4, patsize+4)
        top = np.zeros(shp, dtype=np.float32)
        bot = np.zeros(shp, dtype=np.float32)
        lef = np.zeros(shp, dtype=np.float32)
        rit = np.zeros(shp, dtype=np.float32)
        toplef = np.zeros(shp, dtype=np.float32)
        toprit = np.zeros(shp, dtype=np.float32)
        botlef = np.zeros(shp, dtype=np.float32)
        botrit = np.zeros(shp, dtype=np.float32)
        cn     = np.zeros(shp, dtype=np.float32)
        bord   = np.zeros(shp, dtype=np.float32)

        for ka in range(1, patsize+3):
            for kb in range(1, patsize+3):
                top   [ka,kb,ka-1,kb  ] = 1
                bot   [ka,kb,ka+1,kb  ] = 1
                lef   [ka,kb,ka,  kb-1] = 1
                rit   [ka,kb,ka,  kb+1] = 1
                toplef[ka,kb,ka-1,kb-1] = 1
                toprit[ka,kb,ka-1,kb+1] = 1
                botlef[ka,kb,ka+1,kb-1] = 1
                botrit[ka,kb,ka+1,kb+1] = 1
                cn    [ka,kb,ka,  kb  ] = 1
                if ka in (1,patsize+2) or kb in (1,patsize+2):
                    (bord if (ka+kb)%2 else bord)[ka,kb][odds_idx if (ka+kb)%2==0 else evens_idx] = 1
                if ka == 1:           bord[ka,kb,ka+1,kb  ] = 1
                if kb == 1:           bord[ka,kb,ka,  kb+1] = 1
                if ka == patsize+2:   bord[ka,kb,ka-1,kb  ] = 1
                if kb == patsize+2:   bord[ka,kb,ka,  kb-1] = 1

        top_idx, bot_idx  = _where(top == 1), _where(bot == 1)
        lef_idx, rit_idx  = _where(lef == 1), _where(rit == 1)
        toplef_idx, toprit_idx = _where(toplef == 1), _where(toprit == 1)
        botlef_idx, botrit_idx = _where(botlef == 1), _where(botrit == 1)
        cn_idx   = _where(cn == 1)
        bord_idx = _where(bord == 1)

        N_WORKERS = min(os.cpu_count() or 1, base2d.shape[-1])

        shm_base2d, base2d = _to_shm(base2d)     # view overwrites local base2d
        shm_rf,     rf     = _to_shm(rf)

        idx_pack = (top_idx, bot_idx, lef_idx, rit_idx,
                    toplef_idx, toprit_idx, botlef_idx, botrit_idx,
                    bord_idx)

        with cf.ProcessPoolExecutor(
                max_workers=N_WORKERS,
                initializer=_init_solve_pool_shm,
                initargs=(shm_base2d.name, base2d.shape, base2d.dtype,
                          shm_rf.name,     rf.shape,     rf.dtype,
                          *idx_pack, int(patsize))
        ) as pool:
            for res_list in pool.map(_solve_chunk,
                                     _chunks(base2d.shape[-1], 2048)):  # bigger grain
                for hj, newpat in res_list:
                    dom2d[0, 1:-1, 1:-1, hj] = newpat

        base2d = base2d[:, 2:-2, 2:-2, :]
        dom2d  = dom2d[:, 2:-2, 2:-2, :]
        dummy  = dummy[:, 2:-2, 2:-2, :]
        rf     = rf   [:, 2:-2, 2:-2, :]

        rf      = rf[:, 1:-1, 1:-1, :]
        rfdom   = _clone(rf)

        rf[0] = base2d[0, 2:, 1:-1, :]
        rf[1] = base2d[0, :-2, 1:-1, :]
        rf[2] = base2d[0, 1:-1, 2:, :]
        rf[3] = base2d[0, 1:-1, :-2, :]

        rfdom[0] = dom2d[0, 2:, 1:-1, :]
        rfdom[1] = dom2d[0, :-2, 1:-1, :]
        rfdom[2] = dom2d[0, 1:-1, 2:, :]
        rfdom[3] = dom2d[0, 1:-1, :-2, :]

        rf = rf.reshape(rf.shape[0], -1, rf.shape[-1])
        rf = (_abs(rf[0] - rf[1]) + _abs(rf[0] - rf[2]) + _abs(rf[0] - rf[3]) +
              _abs(rf[1] - rf[2]) + _abs(rf[1] - rf[3]) + _abs(rf[2] - rf[3]))
        rfmean = _unsqueeze(_mean(rf, axis=0), 0)
        rfmean[rfmean == 0] = 1
        rf /= rfmean

        rfdom = rfdom.reshape(rfdom.shape[0], -1, rfdom.shape[-1])
        rfdomstd = (_abs(rfdom[0] - rfdom[1]) + _abs(rfdom[0] - rfdom[2]) +
                    _abs(rfdom[0] - rfdom[3]) + _abs(rfdom[1] - rfdom[2]) +
                    _abs(rfdom[1] - rfdom[3]) + _abs(rfdom[2] - rfdom[3]))
        rfdommean = _unsqueeze(_mean(rfdomstd, axis=0), 0)
        rfdommean[rfdommean == 0] = 1
        rfdomstd /= rfdommean

        basediff = _abs(rf - rfdomstd)
        basediff = np.sum(basediff, axis=0)

        base2d = base2d.reshape(base2d.shape[0], -1, base2d.shape[-1])
        dom2d  = dom2d .reshape(dom2d .shape[0], -1, dom2d .shape[-1])
        dummy  = dummy .reshape(dummy .shape[0], -1, dummy .shape[-1])

        trumeanbase = _mean(base2d, axis=1)
        trumeanbase = trumeanbase.reshape(1, 1, trumeanbase.shape[-1])
        trumeanbase[trumeanbase == 0] = FLOAT32_MIN

        base2d *= 0
        dom2d  *= 0

        # rebuild current base (curcomp) and run compare masking
        curcomp = RP(curcomp)
        curcomp = unfold(curcomp)
        curcomp = curcomp.reshape(-1, patsize+4, patsize+4, curcomp.shape[-1])
        curcomp = curcomp[:, 2:-2, 2:-2, :]
        curcomp = curcomp.reshape(1, -1, curcomp.shape[-1])
        dom2d   = _clone(curcomp)

        with cf.ProcessPoolExecutor(
                max_workers=N_WORKERS,
                initializer=_init_compare_pool,
                initargs=(compare3d, rf, rfdom, rfdommean,
                          basediff, trumeanbase, int(patsize))
        ) as pool:
            for whurdif in pool.map(_process_compare_channel,
                                    range(compare3d.shape[1])):
                if whurdif.size:
                    dom2d[:, :, whurdif] *= 0

        # cleanup SHM
        shm_base2d.close(); shm_base2d.unlink()
        shm_rf.close();     shm_rf.unlink()

        dom2d  = _fold4d(dom2d, (H, W), (patsize, patsize))
        dummy  = _fold4d(dummy, (H, W), (patsize, patsize))
        dom2d  = dom2d / dummy  # float32, shape (1,1,H,W)

        # ---- NEW: inpaint saturated pixels that remained zero ----
        if ignore_overexposed and sat_mask is not None and sat_mask.any():
            arr2d = dom2d[0, 0]  # (H,W) float32
            dom2d[0, 0] = _inpaint_with_k_median(arr2d, sat_mask, k=10)

        # dom2d is float32 shaped (1,1,H,W)
        arr2d = dom2d[0, 0]  # (H,W) float32
        
        if opal_vectra:
            # We processed on 0..255 data; clamp then invert back to the original channel's range
            arr2d = np.clip(arr2d, 0.0, 255.0)
            arr2d = _invert_from_uint8(arr2d, orig_mins[oz], orig_maxs[oz], orig_dtype)
        else:
            # No temporary 8-bit scaling: just clip to the original channel's range, cast to original dtype
            mn = float(orig_mins[oz]); mx = float(orig_maxs[oz])
            arr2d = np.clip(arr2d, mn, mx)
            if np.issubdtype(orig_dtype, np.integer):
                info = np.iinfo(orig_dtype)
                arr2d = np.rint(arr2d).clip(info.min, info.max).astype(orig_dtype)
            else:
                arr2d = arr2d.astype(orig_dtype)
        
        # write result into outstack in ORIGINAL dtype/range
        outstack[oz] = arr2d

        print("    Done in {:.2f}s".format(time.time() - start_time))

    # ───────────────────────────── save output ─────────────────────────────
    if swapped_rgb:
        outstack = np.swapaxes(outstack, -3, -2)
        outstack = np.swapaxes(outstack, -2, -1)

    if single_chan is None:
        out_name = os.path.splitext(file_name)[0] + "_debleed.tif"
        _safe_imwrite(out_name, outstack, imagej=True)
        print("\nSaved full stack -> {}".format(out_name))
    else:
        out_name = os.path.splitext(file_name)[0] + "_Channel_{}_debleed.tif".format(single_chan+1)
        _safe_imwrite(out_name, outstack[single_chan], imagej=True)
        print("\nSaved channel {} -> {}".format(single_chan+1, out_name))

    print("=== Total runtime {:.1f}s ===".format(time.time() - total_start))