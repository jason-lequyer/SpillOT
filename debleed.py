from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# -------------------------
# Channel parsing
# -------------------------
def parse_channels(spec: str, max_ch: Optional[int]) -> List[int]:
    """
    Parse channels spec into a sorted unique list of 1-indexed channel numbers.
    spec examples:
      - "21"
      - "1,3-5"
      - "all"
    """
    spec_in = (spec or "").strip().lower()
    if spec_in in ("", "all", "*", "everything"):
        if max_ch is None:
            raise ValueError(
                "Channels spec is 'all', but I couldn't infer the number of channels.\n"
                "Either:\n"
                "  (1) install tifffile in your env, or\n"
                "  (2) specify channels explicitly, e.g. '21' or '1,3-5'."
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

    # unique + sorted + positive
    out = sorted({c for c in out if c > 0})
    return out


# -------------------------
# TIFF helpers (optional)
# -------------------------
def infer_n_channels(tif_path: Path) -> Optional[int]:
    """
    Try to infer number of channels/pages in the TIFF stack.
    Uses tifffile if available. Returns None if inference fails.
    """
    try:
        import tifffile  # type: ignore
    except Exception:
        return None

    try:
        with tifffile.TiffFile(str(tif_path)) as tif:
            # Prefer series axes if present (OME-TIFF etc.)
            try:
                series0 = tif.series[0]
                axes = getattr(series0, "axes", "") or ""
                shape = getattr(series0, "shape", None)
                if shape is not None and axes:
                    # If explicit channel axis exists
                    if "C" in axes:
                        return int(shape[axes.index("C")])
                    # Otherwise, many IMC stacks are pages ~= first dimension
                    if len(shape) >= 3:
                        return int(shape[0])
            except Exception:
                pass

            # Fallback: number of pages
            return int(len(tif.pages))
    except Exception:
        return None


def collect_metadata_text(tif_path: Path) -> str:
    """
    Best-effort extraction of TIFF metadata text to detect Opal/Vectra.
    Returns "" if tifffile isn't available or reading fails.
    """
    try:
        import tifffile  # type: ignore
    except Exception:
        return ""

    texts: List[str] = []
    try:
        with tifffile.TiffFile(str(tif_path)) as tif:
            # pull a handful of common text-bearing tags from first few pages
            for page in list(tif.pages)[: min(3, len(tif.pages))]:
                for tagname in ("ImageDescription", "Software", "Artist", "HostComputer"):
                    tag = page.tags.get(tagname)
                    if tag is not None:
                        val = getattr(tag, "value", None)
                        if val:
                            texts.append(str(val))

            # ImageJ metadata (dict-like)
            try:
                md = tif.imagej_metadata
                if md:
                    texts.append(str(md))
            except Exception:
                pass

            # OME XML
            try:
                ome = tif.ome_metadata
                if ome:
                    texts.append(str(ome))
            except Exception:
                pass
    except Exception:
        return ""

    return "\n".join(texts)


# -------------------------
# Runner discovery
# -------------------------
def find_runner(
    use_keep_brightest: bool,
    runner_dir: Optional[Path] = None,
) -> Path:
    """
    Locate the runner script file.
    Searches common locations relative to this script, CWD, and optional runner_dir.
    """
    keep_names = ["keep_the_brightest.py", "keep_the_brigtest.py"]
    sig_names = ["signal_based.py", "singal_based.py"]

    names = keep_names if use_keep_brightest else sig_names

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    roots: List[Path] = []
    if runner_dir is not None:
        roots.append(runner_dir)

    roots += [script_dir, cwd]

    # expand likely subdirs
    expanded: List[Path] = []
    for r in roots:
        expanded.append(r)
        expanded.append(r / "Debleed")
        expanded.append(r / "bin")
        expanded.append(r / "Debleed" / "bin")

    for root in expanded:
        for nm in names:
            p = root / nm
            if p.exists() and p.is_file():
                return p

    which = "keep-the-brightest" if use_keep_brightest else "signal-based"
    raise FileNotFoundError(
        f"Could not find a {which} runner script.\n"
        f"Searched for: {', '.join(names)}\n"
        f"In: {', '.join(str(p) for p in expanded)}\n"
        f"\nFix: put the runner script next to debleed.py (or in ./bin or ./Debleed/)."
    )


# -------------------------
# Output discovery
# -------------------------
def expected_output_candidates(stack_path: Path, channel: int) -> List[Path]:
    """
    Generate a few reasonable guesses for the output file name(s).
    Underlying scripts may name output based on slicing off '.tif' or via splitext.
    """
    base_no_ext = stack_path.with_suffix("")  # handles .tif/.tiff nicely
    cands: List[Path] = []

    # Most reasonable
    cands.append(Path(f"{base_no_ext}_Channel_{channel}_debleed.tif"))

    # Plugin-style (img_path[:-4]) can behave oddly for .tiff, but include just in case
    s = str(stack_path)
    if len(s) > 4:
        cands.append(Path(f"{s[:-4]}_Channel_{channel}_debleed.tif"))

    # Glob fallback pattern
    # We'll search after the run if needed.
    return cands


def locate_output(stack_path: Path, channel: int) -> Optional[Path]:
    """
    Locate the generated output file for a channel.
    """
    for p in expected_output_candidates(stack_path, channel):
        if p.exists():
            return p

    # glob fallback: anything like *_Channel_<ch>_debleed.tif next to the input
    base_no_ext = stack_path.with_suffix("")
    pattern = f"{base_no_ext}_Channel_{channel}_debleed.tif*"
    matches = sorted(glob.glob(pattern))
    for m in matches:
        mp = Path(m)
        if mp.exists():
            return mp

    # broader fallback (in case base naming differs slightly)
    pattern2 = str(stack_path.parent / f"*Channel_{channel}_debleed.tif*")
    matches2 = sorted(glob.glob(pattern2))
    for m in matches2:
        mp = Path(m)
        if mp.exists():
            return mp

    return None


# -------------------------
# Main
# -------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="debleed.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Terminal Debleed runner wrapper (cluster-friendly).\n\n"
            "Examples:\n"
            "  python debleed.py IMC_smallcrop/IMC_smallcrop.tif 21\n"
            "  python debleed.py IMC_smallcrop/IMC_smallcrop.tif 1,3-5 --patsize 16\n"
            "  python debleed.py IMC_smallcrop/IMC_smallcrop.tif 21 --signal_based\n"
        ),
    )
    ap.add_argument("stack_tif", type=str, help="Path to the TIFF stack (IMC/IF stack).")
    ap.add_argument(
        "channels",
        nargs="?",
        default="all",
        help="Channel(s) to process: e.g. 21 or 1,3-5 or all (1-indexed). Default: all",
    )
    ap.add_argument(
        "-p",
        "--patsize",
        type=int,
        default=16,
        help="Patch size (EVEN integer >= 4). Default: 16",
    )
    ap.add_argument(
        "--ignore_overexposed",
        action="store_true",
        help="Pass through: ignore overexposed pixels (set saturated to 0).",
    )

    # Your requested misspelling + correct spelling
    ap.add_argument(
        "--signal_based",
        "--singal_based",
        dest="signal_based",
        action="store_true",
        help="Use the signal-based runner (recommended for Opal/Vectra).",
    )

    ap.add_argument(
        "--runner_dir",
        type=str,
        default=None,
        help="Optional directory to search first for runner scripts.",
    )
    ap.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python executable to use for runner subprocesses (default: current interpreter).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Stream runner stdout/stderr to console (useful for debugging).",
    )

    args = ap.parse_args(argv)

    stack_path = Path(args.stack_tif).expanduser().resolve()
    if not stack_path.exists():
        print(f"[ERROR] Input TIFF not found: {stack_path}", file=sys.stderr)
        return 2

    patsize = int(args.patsize)
    if patsize < 4 or (patsize % 2 != 0):
        print(
            f"[ERROR] Invalid --patsize {patsize}. Must be an EVEN integer >= 4.",
            file=sys.stderr,
        )
        return 2

    # Determine channels (for 'all' we need n_ch)
    n_ch = infer_n_channels(stack_path)
    try:
        channels = parse_channels(args.channels, max_ch=n_ch)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    # Validate range if we know n_ch
    if n_ch is not None:
        bad = [c for c in channels if not (1 <= c <= n_ch)]
        if bad:
            print(
                f"[ERROR] Channel(s) out of range. Got {bad}; valid range is 1..{n_ch}.",
                file=sys.stderr,
            )
            return 2

    # Matrix check (recommended)
    matrix_path = stack_path.with_suffix(".csv")
    if not matrix_path.exists():
        print(
            f"[WARN] No co-expression/veto matrix found at: {matrix_path}\n"
            f"       Debleeding will still run, but IMC results are usually better with a matrix.",
            file=sys.stderr,
        )

    # Choose runner script
    use_keep_brightest = not args.signal_based
    runner_dir = Path(args.runner_dir).expanduser().resolve() if args.runner_dir else None
    try:
        runner_script = find_runner(use_keep_brightest=use_keep_brightest, runner_dir=runner_dir)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    # Opal/Vectra warning if keep-brightest is selected
    if use_keep_brightest:
        meta_text = collect_metadata_text(stack_path)
        if meta_text and re.search(r"\b(opal|vectra)\b", meta_text, flags=re.IGNORECASE):
            print(
                "[WARN] Detected 'Opal'/'Vectra' in TIFF metadata.\n"
                "       The keep-the-brightest heuristic can be too aggressive for Opal/Vectra.\n"
                "       Recommended: re-run with --signal_based (or --singal_based).\n",
                file=sys.stderr,
            )

    pyexe = args.python or sys.executable

    # Run per channel
    total = len(channels)
    t0 = time.time()
    outputs: List[Tuple[int, Path]] = []

    for i, ch in enumerate(channels, start=1):
        pct = int(round(100.0 * i / float(total))) if total else 100
        print(f"[{i} / {total} ({pct}%)] Processing channel {ch} ...", flush=True)

        cmd = [
            pyexe,
            str(runner_script),
            str(stack_path),
            str(ch),
            "--patsize",
            str(patsize),
        ]
        if args.ignore_overexposed:
            cmd.append("--ignore_overexposed")

        if args.verbose:
            print(f"  Running: {shlex.join(cmd)}", flush=True)
            proc = subprocess.run(cmd)
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            print(f"[ERROR] Runner failed for channel {ch} (exit {proc.returncode}).", file=sys.stderr)
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
                f"        Looked for something like: {stack_path.with_suffix('')}_Channel_{ch}_debleed.tif",
                file=sys.stderr,
            )
            return 3

        outputs.append((ch, out_path))
        print(f"  -> Output: {out_path}", flush=True)

    elapsed = time.time() - t0
    mm = int(elapsed // 60)
    ss = int(elapsed % 60)
    print(f"\nDone. Processed {total} channel(s) in {mm:02d}:{ss:02d}.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

