"""Run the unchanged legacy CLI with reproducible patch-level RNG only.

This helper is a separate process because the legacy backend owns __main__
and forks multiprocessing workers. The source is AST-instrumented solely at
the entrance to its domino solver; patch extraction, comparison and folding
are executed from the repository's actual legacy source.
"""

import argparse
import ast
from pathlib import Path
import sys


def patch_seed(seed, channel, global_patch):
    mask = (1 << 64) - 1
    value = (int(seed) ^ ((int(channel) + 1) * 0x9E3779B97F4A7C15)
             ^ ((int(global_patch) + 1) * 0xD1B54A32D192ED03)) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & 0xFFFFFFFF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--ignore-overexposed", action="store_true")
    args = parser.parse_args()
    backend = Path(__file__).resolve().parents[1] / "plugins/SpillOT/SpillOT.py"
    tree = ast.parse(backend.read_text(), filename=str(backend))
    solver = [node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "_solve_one_plane"]
    if len(solver) != 1:
        raise RuntimeError("Cannot identify the stock patch solver")
    solver[0].body.insert(0, ast.parse(
        "np.random.seed(_reference_patch_seed(_reference_seed, int(oz), int(hj)))"
    ).body[0])
    ast.fix_missing_locations(tree)
    sys.argv = [str(backend), str(args.image), "-p", str(args.patch_size),
                "--workers", "1", "--csv", str(args.csv)]
    if args.ignore_overexposed:
        sys.argv.append("--ignore_overexposed")
    namespace = globals()
    namespace.update({"_reference_patch_seed": patch_seed,
                      "_reference_seed": args.seed,
                      "__file__": str(backend)})
    exec(compile(tree, str(backend), "exec"), namespace)


if __name__ == "__main__":
    main()
