# SpillOT for QuPath

A QuPath extension that runs SpillOT spillover removal on the image open in QuPath.
It has the same workflow as the Fiji plugin and reuses its patch solver and
comparison algorithm. The memory-aware runner automatically processes tiles and
patch batches without holding all overlapping patches in RAM.

Requires **QuPath 0.7.0 or later**.

![SpillOT window in QuPath](screenshot.png)

## Install

1. Create the `spillot` conda environment (same one the Fiji plugin uses):

   ```bash
   conda create -n spillot -c conda-forge python=3.11 numpy scipy tifffile imagecodecs
   ```

2. Download [qupath-extension-spillot-0.2.2.jar](../qupath-extension-spillot-0.2.2.jar),
   remove any older SpillOT extension jar, then drag the new jar onto the QuPath
   window and restart QuPath. Both Python runners are inside the jar; nothing
   else needs to be copied.

## Use

1. Open a multiplexed image (IMC, CODEX, Opal/Vectra, ...).
   For a whole-slide image, **draw and select an annotation** around the area you want to clean
   first. The selected object's bounding box is used. You can also choose the whole image.
2. **Extensions → SpillOT → Run SpillOT...**
3. On the left, tick the channels you want to clean. For each one, click the channels on the right
   that you suspect are bleeding into it. Signal is removed only where local structure matches.
4. Set the patch size (even, >= 4; default 24, smaller is faster and more aggressive), the number
   of worker processes, and the path to the `spillot` conda environment (prefilled when found).
5. Click **Run**. The window shows the current stage, progress and an estimated
   remaining time once enough work has completed to measure the processing rate.
   The estimate updates as the run proceeds; saving the result is a separate stage.

When the run finishes the cleaned image is added to the open project (or offered for opening in
the viewer if there is no project) with the original channel names and colours.

## Automatic memory management

Tile size and patch size are different settings. **Patch size stays at 24** unless
you change it. Each tile includes the context needed by patches that cross its
boundary, and results are written back to their original positions. Global channel
offsets and intensity ranges are shared across tiles. All channels and the full
selected region are retained in the output, including channels you did not clean.

Tiles are processed in sequence. The backend limits its patch batches and worker
count to fit the estimated RAM budget, so the requested worker count is an upper
limit. The chosen settings are recorded in the run log. Memory
estimates leave headroom for QuPath and the operating system, but other applications
can still change how much memory is available during a run.

The automatic budget is **75% of available RAM at startup, with no fixed GiB
ceiling**. It scales to larger workstations as well as laptops. The planner uses
that budget to choose its working sizes and worker count; it does not allocate
unused memory just to fill the budget or continuously resize during a run.

Temporary files are stored on disk, and the extension checks free disk space before
exporting. Large slides therefore need substantial free disk space even when RAM
use is bounded. Successful outputs are published after processing and saving have
finished; cancellation removes the run's temporary files.

Tiling saves memory; it does not eliminate the work of comparing every overlapping
patch. Large regions can still take hours and whole slides can take much longer,
especially when many target channels are selected. Use the ETA from your own data
to judge the expected run time. Initial estimates can change as different tissue
regions or target channels are processed.

The patch solver uses random assignments. The tiled runner uses a seed tied to each
global patch coordinate so tile scheduling does not introduce new random choices
at boundaries. It does not reproduce an arbitrary previous unseeded run exactly.
Small regression fixtures compare tiled output against the original backend with
matched seeds, including tile edges and corners.

**Ignore overexposed pixels** remains available. Saturated pixels are handled using
global neighboring signal, rather than a fixed tile halo. If the full nearest-neighbor
index would exceed the memory budget, a bounded search is used and logged. Equally
distant neighbors may be selected differently in that fallback.

## Outputs

Written to the output folder (default: a `SpillOT` folder in the project or next to the image):

```text
<name>.tif                        exported region, one plane per channel
<name>.csv                        spillover matrix (row = target, column = source, 1 = remove)
<name>_Channel_<N>_SpillOT.tif    each cleaned channel
<name>_SpillOT.tif                full stack with the cleaned channels replaced
```

These are the same files and CSV convention as the Fiji plugin, so the exported `.tif` + `.csv`
pair can also be sent to a cluster and run with `plugins/SpillOT/SpillOT.py`.
If a `.csv` from an earlier run on the same region exists, the dialog is prefilled from it.
TIFF output includes channel names and pixel calibration;
large files use BigTIFF. A cropped region is still a new image with its own local
pixel coordinates; the source origin is retained in the output metadata.

## Notes

- Marker stacks saved as plain multi-page TIFFs open in QuPath as a z-stack or time series with one
  channel. The extension detects this (longest axis = markers, as in the Fiji plugin) and labels the
  markers `Ch1`, `Ch2`, ...; the result is saved as a proper multichannel image.
- For images with real z-slices or timepoints, the plane shown in the viewer is processed.
- RGB (brightfield) images are not supported.
- SpillOT supports unsigned 8-bit, unsigned 16-bit and 32-bit float
  images. Other source pixel types are rejected before export to avoid silently
  losing precision.
- The backend is CPU-only.

## Scripting

```groovy
import qupath.ext.spillot.SpillOT

def p = new SpillOT.Params()
p.envPath   = "/home/me/miniconda3/envs/spillot"
p.outputDir = java.nio.file.Paths.get("/data/spillot_out")
p.baseName  = "my_region"
// p.memoryBudgetMB = 1024         // optional backend budget; 0 = automatic
p.sources.put(20, [25, 27] as Set)   // 0-based: remove channels 26 and 28 from channel 21
// p.region = qupath.lib.regions.ImageRegion.createInstance(x, y, width, height, 0, 0)
def result = SpillOT.run(getCurrentServer(), p, { println it }, null, { false })
println result.fullStack
```

## Build from source

Needs a JDK 25 (the version QuPath 0.7 uses). From this folder:

```bash
./gradlew build
```

The jar is written to `build/libs/`. The build bundles `SpillOT.py` and
`SpillOT_low_memory.py` from `../plugins/SpillOT/`, so this folder must stay inside
the SpillOT repository.

## Tests

From the repository root, run the Python regression suite with an environment
containing NumPy, SciPy and tifffile:

```bash
python -m unittest discover -s tests -p 'test_spillot_low_memory.py'
```

From this folder, run `./gradlew test`. Set `SPILLOT_TEST_ENV` to the root of a
compatible conda environment to include the Java-to-Python integration tests:

```bash
SPILLOT_TEST_ENV=/path/to/envs/spillot ./gradlew test
```
