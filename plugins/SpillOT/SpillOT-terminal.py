# Copyright 2025, Jason Lequyer, Ferris Nowlan and Laurence Pelletier.
# All rights reserved.
# Sinai Health System - Lunenfeld Tanenbaum Research Institute
# 600 University Avenue, Room 1070, Toronto, ON M5G 1X5, Canada
#
# Terminal wrapper for the SpillOT manual spillover-removal pipeline.

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def parse_channels(spec: str, max_ch: Optional[int]) -> List[int]:
    spec_in = (spec or "").strip().lower()
    if spec_in in ("", "all", "*", "everything"):
        if max_ch is None:
            raise ValueError(
                "Channels spec is 'all', but I couldn't infer the number of channels.\n"
                "Specify channels explicitly, e.g. '21' or '1,3-5', or run this wrapper "
                "with a Python that has tifffile installed."
            )
        return list(range(1, max_ch + 1))

    out: List[int] = []
    tokens = re.split(r"[,\s]+", spec_in)
    for tok in tokens:
        if not tok:
            continue
        if "-" in tok:
            lo_s, hi_s = tok.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(tok))

    return sorted({c for c in out if c > 0})


def infer_n_channels(tif_path: Path) -> Optional[int]:
    try:
        import tifffile  # type: ignore
    except Exception:
        return None

    try:
        with tifffile.TiffFile(str(tif_path)) as tif:
            try:
                series0 = tif.series[0]
                axes = getattr(series0, "axes", "") or ""
                shape = getattr(series0, "shape", None)
                if shape is not None and axes:
                    if "C" in axes:
                        return int(shape[axes.index("C")])
                    if len(shape) >= 3:
                        return int(shape[0])
            except Exception:
                pass

            return int(len(tif.pages))
    except Exception:
        return None


def infer_n_channels_with_python(pyexe: str, tif_path: Path) -> Optional[int]:
    script = r'''
import sys
import tifffile
p = sys.argv[1]
with tifffile.TiffFile(p) as tif:
    try:
        s = tif.series[0]
        axes = getattr(s, "axes", "") or ""
        shape = getattr(s, "shape", None)
        if shape is not None and axes and "C" in axes:
            print(int(shape[axes.index("C")]))
        elif shape is not None and len(shape) >= 3:
            print(int(shape[0]))
        else:
            print(int(len(tif.pages)))
    except Exception:
        print(int(len(tif.pages)))
'''
    try:
        proc = subprocess.run(
            [pyexe, "-c", script, str(tif_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return int(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def find_runner(runner_dir: Optional[Path] = None) -> Path:
    script_dir = Path(__file__).resolve().parent
    this_file = Path(__file__).resolve()
    cwd = Path.cwd()

    roots: List[Path] = []
    if runner_dir is not None:
        roots.append(runner_dir)
    roots += [script_dir, cwd]

    expanded: List[Path] = []
    for r in roots:
        expanded.append(r)
        expanded.append(r / "bin")
        expanded.append(r / "SpillOT")
        expanded.append(r / "SpillOT" / "bin")
        # Backward-compatible search paths for older installs.
        expanded.append(r / "Debleed")
        expanded.append(r / "Debleed" / "bin")

    runner_names = ["SpillOT.py", "spillot.py", "debleed.py"]
    seen = set()
    for root in expanded:
        for name in runner_names:
            p = (root / name).resolve()
            if p in seen:
                continue
            seen.add(p)
            if p == this_file:
                continue
            if p.exists() and p.is_file():
                return p

    raise FileNotFoundError(
        "Could not find the SpillOT runner script 'SpillOT.py'.\n"
        "Put SpillOT.py beside this terminal wrapper, or pass --runner_dir.\n"
        "Searched in: " + ", ".join(str(p) for p in expanded)
    )


def expected_output_candidates(stack_path: Path, channel: int) -> List[Path]:
    stem_splitext = Path(os.path.splitext(str(stack_path))[0])
    stem_suffix = stack_path.with_suffix("")
    return [
        Path(f"{stem_splitext}_Channel_{channel}_SpillOT.tif"),
        Path(f"{stem_suffix}_Channel_{channel}_SpillOT.tif"),
        # Backward-compatible fallbacks for older runner names.
        Path(f"{stem_splitext}_Channel_{channel}_debleed.tif"),
        Path(f"{stem_suffix}_Channel_{channel}_debleed.tif"),
    ]


def locate_output(stack_path: Path, channel: int) -> Optional[Path]:
    for p in expected_output_candidates(stack_path, channel):
        if p.exists():
            return p

    base = Path(os.path.splitext(str(stack_path))[0])
    matches = sorted(glob.glob(f"{base}_Channel_{channel}_SpillOT.tif*")) + sorted(glob.glob(f"{base}_Channel_{channel}_debleed.tif*"))
    for m in matches:
        mp = Path(m)
        if mp.exists():
            return mp

    matches2 = sorted(glob.glob(str(stack_path.parent / f"*Channel_{channel}_SpillOT.tif*"))) + sorted(glob.glob(str(stack_path.parent / f"*Channel_{channel}_debleed.tif*")))
    for m in matches2:
        mp = Path(m)
        if mp.exists():
            return mp
    return None


def _assemble_full_in_this_process(
    stack_path: Path,
    outputs: List[Tuple[int, Path]],
    out_full: Path,
) -> None:
    import numpy as np  # type: ignore
    import tifffile  # type: ignore

    raw = tifffile.imread(str(stack_path))

    if raw.ndim == 2:
        layout = "HW"
        H, W = raw.shape
    elif raw.ndim == 3:
        if raw.shape[-1] == 3 and raw.shape[0] != 3:
            layout = "HWC"
            H, W, _ = raw.shape
        else:
            layout = "CHW"
            _, H, W = raw.shape
    else:
        raise RuntimeError(
            f"Unsupported TIFF shape {getattr(raw, 'shape', None)}. "
            "Expected (C,H,W), (H,W), or RGB (H,W,3)."
        )

    for ch, out_path in outputs:
        arr = tifffile.imread(str(out_path))
        arr = np.asarray(arr)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise RuntimeError(f"Output for channel {ch} is not 2D: {out_path} shape={arr.shape}")
        if arr.shape != (H, W):
            raise RuntimeError(
                f"Shape mismatch for channel {ch}: input plane ({H},{W}), output {arr.shape}."
            )

        idx = ch - 1
        if layout == "HW":
            if ch != 1:
                raise RuntimeError(f"Input appears single-plane, but channel {ch} was requested.")
            raw[:, :] = arr
        elif layout == "CHW":
            if idx < 0 or idx >= raw.shape[0]:
                raise RuntimeError(f"Channel {ch} out of range for input with {raw.shape[0]} planes.")
            raw[idx, :, :] = arr
        else:
            if idx < 0 or idx >= 3:
                raise RuntimeError("RGB input supports channels 1..3 only.")
            raw[:, :, idx] = arr

    out_full.parent.mkdir(parents=True, exist_ok=True)
    if layout == "HWC":
        tifffile.imwrite(str(out_full), raw, photometric="rgb", bigtiff=True)
    else:
        tifffile.imwrite(str(out_full), raw, imagej=True, bigtiff=True)


def _assemble_full_with_runner_python(
    pyexe: str,
    stack_path: Path,
    outputs: List[Tuple[int, Path]],
    out_full: Path,
    verbose: bool = False,
) -> None:
    payload = {
        "stack_path": str(stack_path),
        "outputs": [{"channel": int(ch), "path": str(p)} for ch, p in outputs],
        "out_full": str(out_full),
    }

    script = textwrap.dedent(
        r'''
        import json
        import sys
        from pathlib import Path
        import numpy as np
        import tifffile

        cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        stack_path = Path(cfg["stack_path"])
        outputs = cfg["outputs"]
        out_full = Path(cfg["out_full"])

        raw = tifffile.imread(str(stack_path))
        if raw.ndim == 2:
            layout = "HW"
            H, W = raw.shape
        elif raw.ndim == 3:
            if raw.shape[-1] == 3 and raw.shape[0] != 3:
                layout = "HWC"
                H, W, _ = raw.shape
            else:
                layout = "CHW"
                _, H, W = raw.shape
        else:
            raise SystemExit(f"Unsupported TIFF shape {getattr(raw, 'shape', None)}")

        for item in outputs:
            ch = int(item["channel"])
            out_path = Path(item["path"])
            arr = tifffile.imread(str(out_path))
            arr = np.asarray(arr)
            if arr.ndim > 2:
                arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise SystemExit(f"Output for channel {ch} is not 2D: {out_path} shape={arr.shape}")
            if arr.shape != (H, W):
                raise SystemExit(f"Shape mismatch for channel {ch}: input ({H},{W}), output {arr.shape}")

            idx = ch - 1
            if layout == "HW":
                if ch != 1:
                    raise SystemExit(f"Input appears single-plane, but channel {ch} was requested.")
                raw[:, :] = arr
            elif layout == "CHW":
                if idx < 0 or idx >= raw.shape[0]:
                    raise SystemExit(f"Channel {ch} out of range for input with {raw.shape[0]} planes.")
                raw[idx, :, :] = arr
            else:
                if idx < 0 or idx >= 3:
                    raise SystemExit("RGB input supports channels 1..3 only.")
                raw[:, :, idx] = arr

        out_full.parent.mkdir(parents=True, exist_ok=True)
        if layout == "HWC":
            tifffile.imwrite(str(out_full), raw, photometric="rgb", bigtiff=True)
        else:
            tifffile.imwrite(str(out_full), raw, imagej=True, bigtiff=True)
        print(f"[assemble] Wrote full stack: {out_full}")
        '''
    ).lstrip()

    with tempfile.TemporaryDirectory(prefix="SpillOT_assemble_") as td:
        td_p = Path(td)
        cfg_path = td_p / "cfg.json"
        py_path = td_p / "assemble_full.py"
        cfg_path.write_text(json.dumps(payload), encoding="utf-8")
        py_path.write_text(script, encoding="utf-8")

        cmd = [pyexe, str(py_path), str(cfg_path)]
        if verbose:
            print(f"  Assembling via: {shlex.join(cmd)}", flush=True)
            proc = subprocess.run(cmd)
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            msg = f"[ERROR] Assembly step failed (exit {proc.returncode}).\n"
            if not verbose:
                if proc.stdout:
                    msg += "----- stdout -----\n" + proc.stdout + "\n"
                if proc.stderr:
                    msg += "----- stderr -----\n" + proc.stderr + "\n"
            raise RuntimeError(msg)


def assemble_full_stack(
    stack_path: Path,
    outputs: List[Tuple[int, Path]],
    out_full: Path,
    pyexe_for_fallback: str,
    verbose: bool = False,
) -> None:
    try:
        _assemble_full_in_this_process(stack_path, outputs, out_full)
        return
    except Exception as e:
        if verbose:
            print(f"[WARN] In-process assembly failed ({e}). Falling back to runner Python.", file=sys.stderr)
    _assemble_full_with_runner_python(pyexe_for_fallback, stack_path, outputs, out_full, verbose=verbose)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="SpillOTterminal.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Terminal wrapper for the SpillOT manual spillover-removal pipeline.\n\n"
            "This wrapper runs SpillOT.py for each requested channel, then writes\n"
            "a full replacement stack: <stack>_SpillOT.tif.\n\n"
            "CSV semantics:\n"
            "  row = target channel to clean\n"
            "  column = channel suspected of bleeding/spilling into the target\n"
            "  1 or -1 = remove that column from that row where patches match\n"
            "  0, blanks, and other numbers are ignored\n\n"
            "Examples:\n"
            "  python SpillOTterminal.py image.tif 21\n"
            "  python SpillOTterminal.py image.tif 1,3-5 --patsize 16\n"
            "  python SpillOTterminal.py image.tif all --csv manual_subtractions.csv\n"
        ),
    )
    ap.add_argument("stack_tif", help="Path to the TIFF stack.")
    ap.add_argument(
        "channels",
        nargs="?",
        default="all",
        help="Channel(s) to process: e.g. 21 or 1,3-5 or all (1-indexed). Default: all",
    )
    ap.add_argument("-p", "--patsize", type=int, default=16, help="Patch size: even integer >= 4. Default: 16")
    ap.add_argument("--ignore_overexposed", action="store_true", help="Pass through to SpillOT.py.")
    ap.add_argument(
        "--manual_csv", "--manual-csv", "--csv", dest="manual_csv", default=None,
        help="Optional SpillOT/manual spillover CSV. If omitted, SpillOT.py uses <image>.csv if present.",
    )
    ap.add_argument("--runner_dir", default=None, help="Optional directory to search first for SpillOT.py.")
    ap.add_argument("--python", default=None, help="Python executable for runner subprocesses. Default: current interpreter.")
    ap.add_argument("--verbose", action="store_true", help="Stream runner stdout/stderr to console.")

    args = ap.parse_args(argv)

    stack_path = Path(args.stack_tif).expanduser().resolve()
    if not stack_path.exists():
        print(f"[ERROR] Input TIFF not found: {stack_path}", file=sys.stderr)
        return 2

    patsize = int(args.patsize)
    if patsize < 4 or patsize % 2 != 0:
        print(f"[ERROR] Invalid --patsize {patsize}. Must be an even integer >= 4.", file=sys.stderr)
        return 2

    pyexe = args.python or sys.executable

    n_ch = infer_n_channels(stack_path)
    if n_ch is None:
        n_ch = infer_n_channels_with_python(pyexe, stack_path)

    try:
        channels = parse_channels(args.channels, max_ch=n_ch)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    if not channels:
        print("[ERROR] No channels requested.", file=sys.stderr)
        return 2

    if n_ch is not None:
        bad = [c for c in channels if not (1 <= c <= n_ch)]
        if bad:
            print(f"[ERROR] Channel(s) out of range. Got {bad}; valid range is 1..{n_ch}.", file=sys.stderr)
            return 2

    if args.manual_csv:
        manual_csv_arg = Path(args.manual_csv).expanduser()
        if not manual_csv_arg.is_absolute():
            manual_csv_arg = manual_csv_arg.resolve()
        if not manual_csv_arg.exists():
            print(f"[WARN] SpillOT CSV was specified but does not exist: {manual_csv_arg}", file=sys.stderr)
        else:
            print(f"Using SpillOT/manual spillover CSV: {manual_csv_arg}", flush=True)
    else:
        manual_csv_arg = None
        default_csv = Path(os.path.splitext(str(stack_path))[0] + ".csv")
        if default_csv.exists():
            print(f"Using default SpillOT/manual spillover CSV: {default_csv}", flush=True)
        else:
            print(
                f"[WARN] No SpillOT/manual spillover CSV specified and default CSV was not found: {default_csv}\n"
                "       The runner will leave channels unchanged unless a CSV is supplied.",
                file=sys.stderr,
            )

    runner_dir = Path(args.runner_dir).expanduser().resolve() if args.runner_dir else None
    try:
        runner_script = find_runner(runner_dir=runner_dir)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    total = len(channels)
    outputs: List[Tuple[int, Path]] = []
    t0 = time.time()

    for i, ch in enumerate(channels, start=1):
        pct = int(round(100.0 * i / float(total)))
        print(f"[{i} / {total} ({pct}%)] SpillOT processing channel {ch} ...", flush=True)

        cmd = [pyexe, str(runner_script), str(stack_path), str(ch), "--patsize", str(patsize)]
        if args.ignore_overexposed:
            cmd.append("--ignore_overexposed")
        if manual_csv_arg is not None:
            cmd.extend(["--manual_csv", str(manual_csv_arg)])

        if args.verbose:
            print(f"  Running: {shlex.join(cmd)}", flush=True)
            proc = subprocess.run(cmd)
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            print(f"[ERROR] SpillOT runner failed for channel {ch} (exit {proc.returncode}).", file=sys.stderr)
            if not args.verbose:
                if proc.stdout:
                    print("----- stdout -----", file=sys.stderr)
                    print(proc.stdout, file=sys.stderr)
                if proc.stderr:
                    print("----- stderr -----", file=sys.stderr)
                    print(proc.stderr, file=sys.stderr)
            return proc.returncode

        out_path = locate_output(stack_path, ch)
        if out_path is None or not out_path.exists():
            print(
                f"[ERROR] Could not find expected output for channel {ch} after runner completed.\n"
                f"        Expected something like: {Path(os.path.splitext(str(stack_path))[0])}_Channel_{ch}_SpillOT.tif",
                file=sys.stderr,
            )
            return 3

        outputs.append((ch, out_path))
        print(f"  -> Per-channel output: {out_path}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone runner calls. Processed {total} channel(s) in {int(elapsed // 60):02d}:{int(elapsed % 60):02d}.", flush=True)

    out_full = Path(os.path.splitext(str(stack_path))[0] + "_SpillOT.tif")
    try:
        print("\nAssembling full stack with replacements...", flush=True)
        assemble_full_stack(stack_path, outputs, out_full, pyexe_for_fallback=pyexe, verbose=args.verbose)
        print(f"\n[OK] Full SpillOT stack: {out_full}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed assembling/saving full stack:\n{e}", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
