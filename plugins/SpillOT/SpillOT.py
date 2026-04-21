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
from scipy.ndimage import median_filter
import concurrent.futures as cf
import multiprocessing as mp

from multiprocessing import shared_memory

# ---------- top of file (after imports) ----------
_SHM_HANDLES = []          # global registry to keep blocks alive
# -------------------------------------------------



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

    base2d = np.ndarray(shape_base, dtype=dtype_base, buffer=shm_b.buf)
    rf     = np.ndarray(shape_rf,   dtype=dtype_rf,   buffer=shm_rf.buf)

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


# ------------------------------------------------------------------
#  Initializer for the second ProcessPool (compare step)
# ------------------------------------------------------------------
def _init_compare_pool(_compare3d, _base_flat, _basediff, _pat, _base_min_nz,
                       _dropout_weights, _dropout_support_counts,
                       _dropout_robust, _dropout_zero_weight,
                       _min_signal_pixels, _dropout_tolerance,
                       _core_gate, _core_fraction, _min_core_overlap):
    global compare3d, rf
    global basediff, RP, unfold, patsize, base_min_nz
    global dropout_weights, dropout_support_counts
    global dropout_robust, dropout_zero_weight, min_signal_pixels, dropout_tolerance
    global core_gate, core_fraction, min_core_overlap
    compare3d   = _compare3d                 # (1,C,H,W), only the user-selected donor channels
    rf          = _base_flat                 # base/target patches, flattened as (N, P)
    basediff    = _basediff                  # dynamic per-patch similarity threshold, shape (P,)
    patsize     = np.int32(_pat)
    base_min_nz = _base_min_nz.astype(np.float32, copy=False)   # (P,)
    dropout_weights = _dropout_weights.astype(np.float32, copy=False)  # (N,P)
    dropout_support_counts = _dropout_support_counts.astype(np.int32, copy=False)  # (P,)
    dropout_robust = bool(_dropout_robust)
    dropout_zero_weight = np.float32(_dropout_zero_weight)
    min_signal_pixels = int(_min_signal_pixels)
    dropout_tolerance = np.float32(_dropout_tolerance)
    core_gate = bool(_core_gate)
    core_fraction = np.float32(_core_fraction)
    min_core_overlap = np.float32(_min_core_overlap)
    RP, unfold  = RP_pad2, unfold_p16

# ───────────────────────────── helper functions ────────────────────────────────
def _solve_chunk(hj_vec: np.ndarray):
    out = []
    for hj in hj_vec:
        out.append(_solve_one_plane(int(hj)))
    return out


def _chunks(n_planes, grainsize=512):
    for start in range(0, n_planes, grainsize):
        yield np.arange(start, min(start + grainsize, n_planes), dtype=np.int64)


def _clone(a):
    return a.copy()


def _unsqueeze(a, dim):
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


def RP_pad2(x, pad=(2, 2, 2, 2)):
    return _reflection_pad2d(x, pad)


def _unfold4d(x, ksize):
    B, C, H, W = x.shape
    kH, kW     = ksize
    win        = np.lib.stride_tricks.sliding_window_view(x, (kH, kW), axis=(2, 3))
    out_h, out_w = win.shape[2:4]
    win        = win.transpose(0, 1, 4, 5, 2, 3)
    patches    = win.reshape(B, C * kH * kW, out_h * out_w)
    return patches


def unfold_p16(x):
    """Unfold with (patsize+4, patsize+4) kernel."""
    return _unfold4d(x, (patsize + 4, patsize + 4))


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
DEFAULT_PATSIZE  = 16
patsize          = np.int32(DEFAULT_PATSIZE)
# ----- Brightness-shift normalization radius -----
NORM_RADIUS = 2.5  # true Euclidean pixel radius; center excluded in footprint

# Dropout-robust similarity defaults.
# Target-channel zeros are often missing bleed-through rather than true evidence
# that the donor channel is different. A small nonzero weight preserves a weak
# background penalty while preventing zero-heavy patches from dominating compdiff.
DEFAULT_DROPOUT_ROBUST = True
DEFAULT_DROPOUT_ZERO_WEIGHT = 0.05
DEFAULT_MIN_SIGNAL_PIXELS = 4
DEFAULT_DROPOUT_TOLERANCE = 1.25

# Additional anti-over-removal guard:
# Dropout mode intentionally down-weights target-channel zeros, but that can
# make unrelated donor signal outside the target object too cheap. The bright
# core gate requires the brightest part of the target patch to overlap the
# brightest part of the selected donor patch before a patch can be removed.
DEFAULT_CORE_GATE = True
DEFAULT_CORE_FRACTION = 0.20
DEFAULT_MIN_CORE_OVERLAP = 0.60


def _set_patsize(p):
    """Set global patch size (int) everywhere."""
    global patsize
    patsize = np.int32(int(p))



def _disk_footprint(R: float) -> np.ndarray:
    """Boolean disk footprint of (possibly non-integer) radius R with center excluded."""
    R_float = max(1.0, float(R))   # real radius used in distance test
    R_int   = int(R_float)         # integer half-size of the footprint (same as old code)

    yy, xx = np.ogrid[-R_int:R_int+1, -R_int:R_int+1]
    disk = (xx*xx + yy*yy) <= (R_float * R_float)
    disk[R_int, R_int] = False  # exclude center
    return disk

def _process_compare_channel(ow: int) -> np.ndarray:
    # Extract patch stack for this user-selected donor channel.
    frame = compare3d[:, ow:ow+1, :, :]    # (1,1,H,W)
    curcomp = RP(frame)                    # reflect pad
    curcomp = unfold(curcomp)              # (1,(p+4)^2,P)
    curcomp = curcomp.reshape(
        -1, patsize+4, patsize+4, curcomp.shape[-1]
    )[:, 2:-2, 2:-2, :]                    # (1,p,p,P)

    # Shapes now: (1, patsize, patsize, P)
    cur = curcomp[0, :, :, :].astype(np.float32)  # (p,p,P)
    H, W, P = cur.shape
    N = H * W

    # 'rf' holds the target/base patches flattened as (N, P).
    base_flat = rf.astype(np.float32, copy=False)
    if base_flat.shape != (N, P):
        raise ValueError(f"Shape mismatch: base_flat {base_flat.shape} vs cur {(N, P)}")

    cur_flat = cur.reshape(N, P)

    # ---------- Dropout-aware signal-based structural similarity metric ----------
    # Match the donor patch to the target patch after per-patch minimum alignment
    # and local donor->target brightness normalization. Unlike the previous
    # raw-L1 metric, target-channel zeros can be treated as likely dropout/missing
    # bleed-through rather than strong evidence of non-similarity.
    cur_flat_nonzero = np.where(cur_flat > 0, cur_flat, np.inf)
    cur_min_nz = cur_flat_nonzero.min(axis=0)             # (P,)
    cur_min_nz[~np.isfinite(cur_min_nz)] = 0.0
    cur_min_nz = cur_min_nz.astype(np.float32)

    # Root both patches to the target/base patch's smallest non-zero value.
    target = base_min_nz                                  # (P,)

    base_min_all = base_flat.min(axis=0)                  # (P,)
    cur_min_all  = cur_flat.min(axis=0)                   # (P,)

    kB = target - base_min_all                            # (P,)
    kC = target - cur_min_all                             # (P,)

    base_shift = base_flat + kB                           # (N,P)
    cur_shift  = cur_flat  + kC                           # (N,P)

    base_shift_map = base_shift.reshape(H, W, P)
    cur_shift_map  = cur_shift.reshape(H, W, P)
    base_raw_map   = base_flat.reshape(H, W, P)

    eps = np.float32(1e-8)
    ratio = base_shift_map / (cur_shift_map + eps)

    # Estimate the donor->target scale from target-positive pixels only.
    # In zero-heavy bleed-through patches, using all pixels makes the median
    # scale collapse toward zero, which hides otherwise matching donor signal.
    if dropout_robust:
        valid_for_scale = (base_raw_map > 0) & np.isfinite(ratio)
        ratio2 = ratio.reshape(N, P)
        valid2 = valid_for_scale.reshape(N, P)

        # Default scale for patches with no usable target-positive pixels.
        # Compute medians only for columns that have at least one valid value so
        # zero-only patches do not emit noisy "All-NaN slice" warnings.
        patch_scale = np.ones(P, dtype=np.float32)
        valid_cols = valid2.any(axis=0)
        if np.any(valid_cols):
            ratio_valid = np.where(valid2[:, valid_cols], ratio2[:, valid_cols], np.nan)
            patch_scale[valid_cols] = np.nanmedian(ratio_valid, axis=0).astype(np.float32)

        bad_scale = ~np.isfinite(patch_scale)
        if np.any(bad_scale):
            patch_scale[bad_scale] = 1.0

        ratio_for_filter = np.where(valid_for_scale, ratio, patch_scale.reshape(1, 1, P))
    else:
        ratio_for_filter = ratio
        patch_scale = np.ones(P, dtype=np.float32)

    foot3d = _disk_footprint(NORM_RADIUS)[:, :, None]
    scale = median_filter(ratio_for_filter, footprint=foot3d, mode="reflect")

    # Keep rare division/median artifacts from creating absurd local multipliers.
    # This still allows strong donor attenuation, e.g. target bleed much dimmer
    # than the donor source channel.
    if dropout_robust:
        lo = np.maximum(patch_scale * np.float32(0.05), np.float32(0.0)).reshape(1, 1, P)
        hi = np.maximum(patch_scale * np.float32(20.0), np.float32(1e-6)).reshape(1, 1, P)
        scale = np.clip(scale, lo, hi)

    # Brightness-normalized donor patches, still on the shifted baseline.
    cur_norm = cur_shift_map * scale
    cur_norm_flat = cur_norm.reshape(N, P)

    # Donor-vs-target structural difference for each patch.
    if dropout_robust:
        compdiff = np.sum(dropout_weights * np.abs(base_shift - cur_norm_flat), axis=0)
        enough_signal = dropout_support_counts >= int(min_signal_pixels)
        threshold = basediff * dropout_tolerance
    else:
        compdiff = np.sum(np.abs(base_shift - cur_norm_flat), axis=0)
        enough_signal = np.ones(P, dtype=bool)
        threshold = basediff

    # Extra protection against the main false-positive mode of dropout-robust
    # matching. Because target-channel zeros are deliberately weak evidence,
    # an unrelated donor object can otherwise be locally scaled to match a
    # small target object while donor-only bright pixels are mostly ignored.
    #
    # The gate is asymmetric by design: it asks whether the bright core of the
    # target patch is explained by the bright core of the selected donor patch.
    # It does NOT require all donor-bright pixels to be present in the target,
    # because true bleed-through can be dimmer and more incomplete than the
    # source donor channel.
    if dropout_robust and core_gate and (min_core_overlap > 0):
        base_max = base_flat.max(axis=0).astype(np.float32)       # (P,)
        cur_max  = cur_flat.max(axis=0).astype(np.float32)        # (P,)

        bf = np.float32(core_fraction)
        bf = np.minimum(np.maximum(bf, np.float32(0.0)), np.float32(1.0))

        base_thr = base_min_nz + bf * np.maximum(base_max - base_min_nz, np.float32(0.0))
        cur_thr  = cur_min_nz  + bf * np.maximum(cur_max  - cur_min_nz,  np.float32(0.0))

        base_core = (base_flat > 0) & (base_flat >= base_thr.reshape(1, P))
        cur_core  = (cur_flat  > 0) & (cur_flat  >= cur_thr.reshape(1, P))

        base_core_count = base_core.sum(axis=0).astype(np.float32)
        core_overlap_count = (base_core & cur_core).sum(axis=0).astype(np.float32)

        core_overlap = core_overlap_count / np.maximum(base_core_count, np.float32(1.0))
        core_ok = (base_core_count > 0) & (core_overlap >= min_core_overlap)
    else:
        core_ok = np.ones(P, dtype=bool)

    # Manual subtraction supplies the direction. Therefore there is no
    # local-extrema or brighter/darker true-signal tie-break here: a donor
    # patch is removed only when it is structurally close enough to the target
    # under the target's own domino-derived tolerance, and, in dropout mode,
    # when the selected donor also explains the target's bright core.
    whurdif = np.where(enough_signal & core_ok & (compdiff < threshold))[0]
    return whurdif

def _solve_one_plane(hj: int):
    curpat = _clone(base2d[0, :, :, hj])
    curpat = _repeat(curpat, patsize+4, patsize+4, 1, 1)

    curcost          = _zeros_like(curpat)
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


# ---------- overexposed handling + inpaint ----------
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

    if k_eff == 1:
        fill_vals = donors_vals[idxs]
    else:
        fill_vals = np.median(donors_vals[idxs], axis=1)

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
        description="SpillOT: structural patch-based spillover removal for TIFF stacks; optionally one channel."
    )
    parser.add_argument("image", help="Path to the input TIFF")
    parser.add_argument("channel", nargs="?", type=int,
                        help="1-based channel to clean with SpillOT (omit for all channels)")
    parser.add_argument("-p", "--patsize", type=int, default=DEFAULT_PATSIZE,
                        help="Patch size (default: 16)")
    parser.add_argument("--ignore_overexposed", action="store_true",
                        help="If set, pixels saturated at dtype max in the base or any compared channels are zeroed before processing (per run), then inpainted afterward.")
    parser.add_argument("--manual_csv", "--manual-csv", "--csv", dest="manual_csv", default=None,
                        help="Optional SpillOT/manual spillover CSV. If omitted, uses <image>.csv when present; otherwise uses an empty matrix. Only entries equal to 1 or -1 are treated as remove-this-column-from-this-row; 0, blanks, NaN, and other numbers are ignored.")

    args = parser.parse_args()

    file_name = args.image
    single_chan = (args.channel - 1) if args.channel is not None else None
    _set_patsize(args.patsize)
    ignore_overexposed = bool(args.ignore_overexposed)

    # Hard-baked SpillOT similarity settings. These are intentionally
    # not exposed in the Fiji launcher.
    dropout_robust = DEFAULT_DROPOUT_ROBUST
    dropout_zero_weight = DEFAULT_DROPOUT_ZERO_WEIGHT
    min_signal_pixels = DEFAULT_MIN_SIGNAL_PIXELS
    dropout_tolerance = DEFAULT_DROPOUT_TOLERANCE
    core_gate = DEFAULT_CORE_GATE
    core_fraction = DEFAULT_CORE_FRACTION
    min_core_overlap = DEFAULT_MIN_CORE_OVERLAP

    # argparse handles both old positional-only calls and the new optional CSV argument.
    # Avoid reading sys.argv manually here, because optional arguments may appear before
    # or after the image path.

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

    raw_proc = raw_orig

    # Proceed with processing on float32 working copy of raw_proc
    n_chan, H, W = raw_proc.shape
    IMG_H, IMG_W = int(H), int(W)  # immutable image-size aliases
    inpnorm      = raw_proc.astype(np.float32)

    # ---------- GLOBAL PER-CHANNEL OFFSETS (negatives only) ----------
    # For each channel c:
    #   If it has negatives, add -min to make min >= 0.
    # No global "smallest nonzero" offset anymore; zeros are handled per patch.
    chan_offset_neg = np.zeros(n_chan, dtype=np.float32)

    for c in range(n_chan):
        chan = inpnorm[c]  # view (H,W)
        off_neg = 0.0

        min_val = float(chan.min())
        if min_val < 0.0:
            off_neg = -min_val
            chan = chan + off_neg  # new array

        inpnorm[c] = chan
        chan_offset_neg[c] = off_neg

    chan_offset_total = chan_offset_neg.copy()

    # IMPORTANT: Allocate output in the ORIGINAL dtype and shape
    outstack = np.empty_like(raw_orig)

    # manual spillover matrix (read once)
    # CSV semantics for this runner:
    #   - row = target channel to clean
    #   - column = channel suspected of bleeding/spilling into that target
    #   - value 1 or -1 => test/remove that column channel from that row channel
    #   - value 0, blank/NaN, or any other number => ignore
    mask_matrix = np.zeros((n_chan, n_chan), dtype=np.float32)
    if args.manual_csv:
        csv_path = os.path.expanduser(args.manual_csv)
        if not os.path.isabs(csv_path):
            csv_path = os.path.abspath(csv_path)
    else:
        csv_path = os.path.splitext(file_name)[0] + ".csv"

    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            loaded = np.genfromtxt(
                f,
                delimiter=",",
                skip_header=1,
                usecols=tuple(range(1, n_chan + 1)),
            )

        loaded = np.atleast_2d(loaded).astype(np.float32, copy=False)
        loaded[np.isnan(loaded)] = 0.0
        # Only exact SpillOT selection codes are active.  This prevents old
        # diagnostic values or accidental numeric entries from being interpreted
        # as channels to remove.
        loaded = np.where(
            np.isclose(loaded, 1.0, atol=1e-6) | np.isclose(loaded, -1.0, atol=1e-6),
            loaded,
            0.0
        ).astype(np.float32, copy=False)

        if loaded.shape != (n_chan, n_chan):
            fixed = np.zeros((n_chan, n_chan), dtype=np.float32)
            rr = min(n_chan, loaded.shape[0])
            cc = min(n_chan, loaded.shape[1])
            fixed[:rr, :cc] = loaded[:rr, :cc]
            loaded = fixed

        mask_matrix = loaded
        print("Using SpillOT/manual spillover CSV: {}".format(csv_path))

    except Exception as e:
        if args.manual_csv:
            print("SpillOT/manual spillover CSV was specified but could not be read:", csv_path, "\n", e,
                  "\nProceeding with an empty matrix (all channels will be left unchanged unless the CSV is fixed).")
        else:
            print("SpillOT/manual spillover CSV not found or unusable:", e,
                  "\nProceeding with an empty matrix (all channels will be left unchanged unless a CSV is supplied).")

    chan_iter = [single_chan] if single_chan is not None else range(n_chan)
    total_start = time.time()

    if dropout_robust:
        print("Dropout-robust similarity: ON (zero_weight={}, min_signal_pixels={}, tolerance={})".format(
            dropout_zero_weight, min_signal_pixels, dropout_tolerance))
        if core_gate and min_core_overlap > 0:
            print("Bright-core overlap guard: ON (core_fraction={}, min_core_overlap={})".format(
                core_fraction, min_core_overlap))
        else:
            print("Bright-core overlap guard: OFF")
    else:
        print("Dropout-robust similarity: OFF (strict all-pixels L1)")

    # ───────────────────────── channel loop ──────────────────────────
    for oz in chan_iter:
        print("\n SpillOT cleaning channel {}/{}".format(oz+1, n_chan))
        start_time = time.time()

        # working copies (float32), already globally offset to be >= 0 (except saturations)
        base2d_f32     = _clone(inpnorm[oz:oz+1, :, :])  # (1,H,W)
        compare3d_f32  = _clone(inpnorm)                 # (C,H,W)

        # source set from the manual CSV row for this target channel.
        # Accept both the new SpillOT convention (1) and old manual convention (-1).
        # Ignore 0, blanks/NaN, and any other numeric values.
        row = mask_matrix[oz] if oz < mask_matrix.shape[0] else np.zeros((n_chan,), dtype=np.float32)
        included = np.where(np.isclose(np.abs(row), 1.0, atol=1e-6))[0]

        # never use the channel itself as a source, even if the CSV is selected on the diagonal
        included = included[included != oz]

        if included.size == 0:
            print("No source channels marked with 1 or -1 for channel {}; skipping SpillOT cleanup for this channel."
                  .format(oz + 1))
            outstack[oz] = raw_orig[oz]
            continue

        # union of channels to inspect for saturation for THIS run
        check_idxs = np.unique(np.concatenate([included, np.array([oz], dtype=np.int64)]))

        sat_mask = None
        if ignore_overexposed:
            sat_mask = _zero_overexposed_for_run(base2d_f32, compare3d_f32, raw_orig, check_idxs, verbose=True)

        # shape to (B=1,C,H,W) for the downstream code
        base2d    = _unsqueeze(base2d_f32, 0)     # (1,1,H,W)
        compare3d = _unsqueeze(compare3d_f32, 0)  # (1,C,H,W)

        # keep a copy of the current base channel for later steps
        curcomp   = _clone(compare3d[:, oz:oz+1, :, :])
        
        

        compare3d    = compare3d[:, included, :, :]

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
                    (evens if (i+j) % 2 == 0 else odds)[i, j] = 1

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
            rf = np.require(rf, requirements=['C', 'W'])
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
                top   [ka, kb, ka-1, kb  ] = 1
                bot   [ka, kb, ka+1, kb  ] = 1
                lef   [ka, kb, ka,   kb-1] = 1
                rit   [ka, kb, ka,   kb+1] = 1
                toplef[ka, kb, ka-1, kb-1] = 1
                toprit[ka, kb, ka-1, kb+1] = 1
                botlef[ka, kb, ka+1, kb-1] = 1
                botrit[ka, kb, ka+1, kb+1] = 1
                cn    [ka, kb, ka,   kb  ] = 1
                if ka in (1, patsize+2) or kb in (1, patsize+2):
                    (bord if (ka+kb) % 2 else bord)[ka, kb][odds_idx if (ka+kb) % 2 == 0 else evens_idx] = 1
                if ka == 1:           bord[ka, kb, ka+1, kb  ] = 1
                if kb == 1:           bord[ka, kb, ka,   kb+1] = 1
                if ka == patsize+2:   bord[ka, kb, ka-1, kb  ] = 1
                if kb == patsize+2:   bord[ka, kb, ka,   kb-1] = 1

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

        # --- Per-patch difference metric with patch-wise offset (base vs dom) ---
        base = base2d[0, :, :, :].copy()   # (patsize, patsize, P)
        dom  = dom2d[0, :, :, :].copy()    # (patsize, patsize, P)

        ph, pw, Pp = base.shape
        Np = ph * pw

        base_flat = base.reshape(Np, Pp).astype(np.float32)
        dom_flat  = dom.reshape(Np, Pp).astype(np.float32)

        # Per-patch smallest-nonzero for base and dom
        base_flat_nonzero = np.where(base_flat > 0, base_flat, np.inf)
        dom_flat_nonzero  = np.where(dom_flat  > 0, dom_flat,  np.inf)

        base_min_nz = base_flat_nonzero.min(axis=0)              # (P,)
        dom_min_nz  = dom_flat_nonzero.min(axis=0)               # (P,)
        base_min_nz[~np.isfinite(base_min_nz)] = 0.0
        dom_min_nz [~np.isfinite(dom_min_nz)]  = 0.0
        base_min_nz = base_min_nz.astype(np.float32)
        dom_min_nz  = dom_min_nz.astype(np.float32)

        # Target minimum per patch: root both to base_min_nz
        target_bd = base_min_nz                                  # (P,)

        base_min_all_dom = base_flat.min(axis=0)                 # (P,)
        dom_min_all      = dom_flat.min(axis=0)                  # (P,)

        kB_dom = target_bd - base_min_all_dom
        kD_dom = target_bd - dom_min_all

        base_shift_dom = base_flat + kB_dom                      # (N,P)
        dom_shift_dom  = dom_flat  + kD_dom                      # (N,P)

        # basediff: target-vs-domino tolerance for each patch. In dropout-robust
        # mode, target zeros get a small weight because they may be missing
        # bleed-through rather than true absence of donor-like structure.
        base_positive = base_flat > 0
        dropout_support_counts = base_positive.sum(axis=0).astype(np.int32)
        if dropout_robust:
            dropout_weights = np.where(
                base_positive,
                np.float32(1.0),
                np.float32(dropout_zero_weight)
            ).astype(np.float32)
            basediff = np.sum(dropout_weights * np.abs(base_shift_dom - dom_shift_dom), axis=0)
        else:
            dropout_weights = np.ones_like(base_flat, dtype=np.float32)
            basediff = np.sum(np.abs(base_shift_dom - dom_shift_dom), axis=0)

        base2d = base2d.reshape(base2d.shape[0], -1, base2d.shape[-1])
        dom2d  = dom2d .reshape(dom2d .shape[0], -1, dom2d .shape[-1])
        dummy  = dummy .reshape(dummy .shape[0], -1, dummy .shape[-1])
        base2d *= 0
        dom2d  *= 0

        # ---------- rebuild current base (curcomp) and run compare masking ----------
        curcomp = RP(curcomp)
        curcomp = unfold(curcomp)
        curcomp = curcomp.reshape(-1, patsize+4, patsize+4, curcomp.shape[-1])
        curcomp = curcomp[:, 2:-2, 2:-2, :]
        curcomp = curcomp.reshape(1, -1, curcomp.shape[-1])
        dom2d   = _clone(curcomp)

        with cf.ProcessPoolExecutor(
                max_workers=N_WORKERS,
                initializer=_init_compare_pool,
                initargs=(compare3d, base_flat, basediff, int(patsize), base_min_nz,
                          dropout_weights, dropout_support_counts,
                          dropout_robust, dropout_zero_weight,
                          min_signal_pixels, dropout_tolerance,
                          core_gate, core_fraction, min_core_overlap)
        ) as pool:
            for whurdif in pool.map(_process_compare_channel,
                                    range(compare3d.shape[1])):
                if whurdif.size:
                    dom2d[:, :, whurdif] *= 0

        # cleanup SHM
        shm_base2d.close(); shm_base2d.unlink()
        shm_rf.close();     shm_rf.unlink()

        # fold using the true image size
        dom2d  = _fold4d(dom2d, (IMG_H, IMG_W), (patsize, patsize))
        dummy  = _fold4d(dummy, (IMG_H, IMG_W), (patsize, patsize))
        dom2d  = dom2d / dummy  # float32, shape (1,1,H,W)

        # ---- inpaint saturated pixels that remained zero ----
        if ignore_overexposed and sat_mask is not None and sat_mask.any():
            arr2d = dom2d[0, 0]  # (H,W) float32
            dom2d[0, 0] = _inpaint_with_k_median(arr2d, sat_mask, k=10)

        # dom2d is float32 shaped (1,1,H,W)
        arr2d = dom2d[0, 0]  # (H,W) float32

        # Undo the global per-channel negative offsets applied at the very start
        total_off = float(chan_offset_total[oz])
        if total_off != 0.0:
            arr2d = arr2d - total_off

        # Clip to the original channel's range, cast to original dtype
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
        out_name = os.path.splitext(file_name)[0] + "_SpillOT.tif"
        _safe_imwrite(out_name, outstack, imagej=True)
        print("\nSaved full SpillOT stack -> {}".format(out_name))
    else:
        out_name = os.path.splitext(file_name)[0] + "_Channel_{}_SpillOT.tif".format(single_chan+1)
        _safe_imwrite(out_name, outstack[single_chan], imagej=True)
        print("\nSaved SpillOT channel {} -> {}".format(single_chan+1, out_name))

    print("=== Total runtime {:.1f}s ===".format(time.time() - total_start))