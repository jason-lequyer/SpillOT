"""Crash publication at specific filesystem transitions, without finally blocks."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins/SpillOT"))
import SpillOTMemory as backend


def main():
    directory, phase = Path(sys.argv[1]), sys.argv[2]
    output, scratch = directory / "output", directory / "scratch"
    staged = scratch / "staged"
    paths = [staged / name for name in ("first.tif", "second.tif", "third.csv")]
    replace = os.replace

    def crash_after_replace(source, destination, *args, **kwargs):
        result = replace(source, destination, *args, **kwargs)
        if phase == "after_backup" and Path(source) == output / "first.tif":
            os._exit(91)
        if (phase == "after_publish" and Path(source) == staged / "second.tif"
                and Path(destination) == output / "second.tif"):
            os._exit(91)
        return result

    os.replace = crash_after_replace
    backend.publish_outputs(paths, output, staged, lambda: None, scratch_dir=scratch)
    if phase == "committed":
        os._exit(91)
    raise RuntimeError("The requested fault point was not reached")


if __name__ == "__main__":
    main()
