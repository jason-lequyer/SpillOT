// Copyright 2025, Jason Lequyer, Ferris Nowlan and Laurence Pelletier.
// All rights reserved.
// Sinai Health System - Lunenfeld Tanenbaum Research Institute
// 600 University Avenue, Room 1070, Toronto, ON M5G 1X5, Canada

package qupath.ext.spillot;

import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.io.Opener;
import ij.process.ImageProcessor;
import qupath.imagej.tools.IJTools;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.regions.ImageRegion;
import qupath.lib.regions.RegionRequest;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/**
 * GUI-free core of the SpillOT QuPath extension.
 * <p>
 * It exports a region of the current image as an ImageJ TIFF stack (one slice per channel),
 * writes the SpillOT spillover matrix as a CSV, calls the same Python backend used by the
 * Fiji plugin and the command line ({@code SpillOT.py}), and assembles the cleaned channels
 * into a full replacement stack.
 * <p>
 * Everything here can also be called from a QuPath script, e.g.
 * <pre>{@code
 * def p = new qupath.ext.spillot.SpillOT.Params()
 * p.envPath = "/home/me/miniconda3/envs/spillot"
 * p.outputDir = java.nio.file.Paths.get("/data/spillot_out")
 * p.baseName = "my_region"
 * p.sources.put(20, [25] as Set)   // 0-based: remove channel 26 from channel 21
 * qupath.ext.spillot.SpillOT.run(getCurrentServer(), p, { println it }, null, { false })
 * }</pre>
 */
public final class SpillOT {

    public static final int DEFAULT_PATCH_SIZE = 24;

    private SpillOT() {}

    /** Settings for one SpillOT run. Channel indices are 0-based. */
    public static class Params {
        /** Even integer &ge; 4. Smaller = faster and more aggressive. */
        public int patchSize = DEFAULT_PATCH_SIZE;
        /** Worker processes used by the Python backend. */
        public int workers = Math.max(1, Runtime.getRuntime().availableProcessors());
        /** Root of the conda environment (e.g. .../envs/spillot). */
        public String envPath;
        /** Zero saturated pixels before processing and inpaint them afterwards. */
        public boolean ignoreOverexposed = false;
        /** Folder that receives the exported input, the CSV and the results. */
        public Path outputDir;
        /** File name stem used for everything written to {@link #outputDir}. */
        public String baseName;
        /** Full-resolution region to process; {@code null} = whole image (z = 0, t = 0). */
        public ImageRegion region;
        /** target channel -> channels suspected of bleeding into it. */
        public Map<Integer, Set<Integer>> sources = new TreeMap<>();
    }

    /** Files written by a run. */
    public static class Result {
        /** Exported input stack (ImageJ TIFF, one slice per channel). */
        public Path input;
        /** Spillover matrix (row = target, column = source, 1 = remove where similar). */
        public Path csv;
        /** Original stack with the selected channels replaced by their cleaned versions. */
        public Path fullStack;
        /** One cleaned 2D TIFF per target channel, keyed by 0-based channel index. */
        public Map<Integer, Path> channels = new TreeMap<>();
    }

    // ------------------------------------------------------------------
    // Main entry point
    // ------------------------------------------------------------------

    /**
     * Run SpillOT. Blocks until finished; call from a background thread.
     *
     * @param server    image to process (multichannel, not RGB)
     * @param p         settings
     * @param log       receives progress text (may be null)
     * @param progress  receives the number of finished channels (may be null)
     * @param cancelled polled while the backend runs; return true to abort (may be null)
     */
    public static Result run(ImageServer<BufferedImage> server, Params p, Consumer<String> log,
                             Consumer<Integer> progress, BooleanSupplier cancelled)
            throws IOException, InterruptedException {

        Consumer<String> out = log == null ? s -> {} : log;
        BooleanSupplier stop = cancelled == null ? () -> false : cancelled;

        int nChannels = nPlanes(server);
        if (server.isRGB() || nChannels < 2)
            throw new IllegalArgumentException(
                    "SpillOT needs a multichannel (non-RGB) image with at least 2 channels.");
        if (p.patchSize < 4 || p.patchSize % 2 != 0)
            throw new IllegalArgumentException("Patch size must be an even integer >= 4.");
        if (p.workers < 1)
            throw new IllegalArgumentException("Number of worker processes must be >= 1.");

        // Keep only valid target -> source pairs
        Map<Integer, Set<Integer>> sources = new TreeMap<>();
        for (var e : p.sources.entrySet()) {
            int target = e.getKey();
            Set<Integer> srcs = new TreeSet<>();
            for (int s : e.getValue())
                if (s != target && s >= 0 && s < nChannels)
                    srcs.add(s);
            if (target >= 0 && target < nChannels && !srcs.isEmpty())
                sources.put(target, srcs);
        }
        if (sources.isEmpty())
            throw new IllegalArgumentException(
                    "Nothing to do: no target channel has a suspected source channel selected.");

        Path python = pythonFromEnv(p.envPath);
        if (python == null || !Files.exists(python))
            throw new IOException("Could not find Python in the conda environment '" + p.envPath + "'.\n"
                    + "Create it with:\n  conda create -n spillot -c conda-forge python=3.11 numpy scipy tifffile imagecodecs\n"
                    + "then paste the environment path shown by 'conda env list'.");

        ImageRegion region = p.region != null ? p.region
                : ImageRegion.createInstance(0, 0, server.getWidth(), server.getHeight(), 0, 0);
        if (region.getWidth() < p.patchSize + 4 || region.getHeight() < p.patchSize + 4)
            throw new IllegalArgumentException("The region (" + region.getWidth() + " x " + region.getHeight()
                    + " px) is too small for patch size " + p.patchSize + ".");

        Files.createDirectories(p.outputDir);
        Result result = new Result();
        result.input = p.outputDir.resolve(p.baseName + ".tif");
        result.csv = p.outputDir.resolve(p.baseName + ".csv");
        result.fullStack = p.outputDir.resolve(p.baseName + "_SpillOT.tif");

        // 1) Export the region at full resolution as a channel stack
        out.accept("Exporting " + region.getWidth() + " x " + region.getHeight() + " px, "
                + nChannels + " channels -> " + result.input);
        ImagePlus imp = readStack(server, region, 1.0);
        List<String> names = channelNames(server);
        if (!new FileSaver(imp).saveAsTiffStack(result.input.toString()))
            throw new IOException("Could not write " + result.input);

        // 2) Spillover matrix, same convention as the Fiji plugin
        writeCsv(result.csv, names, sources);

        // 3) Python backend, one target channel at a time
        Path runner = extractRunner();
        int done = 0;
        try {
            for (int target : sources.keySet()) {
                if (stop.getAsBoolean())
                    throw new InterruptedException("SpillOT was cancelled.");
                out.accept("");
                out.accept("Cleaning channel " + (target + 1) + " (" + names.get(target) + "), "
                        + (done + 1) + " of " + sources.size());
                List<String> cmd = new ArrayList<>(List.of(
                        python.toString(), runner.toString(), result.input.toString(),
                        Integer.toString(target + 1),
                        "--patsize", Integer.toString(p.patchSize),
                        "--manual_csv", result.csv.toString(),
                        "--workers", Integer.toString(p.workers)));
                if (p.ignoreOverexposed)
                    cmd.add("--ignore_overexposed");
                runProcess(cmd, p.envPath, out, stop);

                Path chanFile = p.outputDir.resolve(p.baseName + "_Channel_" + (target + 1) + "_SpillOT.tif");
                if (!Files.exists(chanFile))
                    throw new IOException("SpillOT result not found: " + chanFile);
                ImagePlus cleaned = new Opener().openImage(chanFile.toString());
                if (cleaned == null)
                    throw new IOException("Could not read " + chanFile);
                ImageProcessor ip = cleaned.getProcessor();
                ImageStack stack = imp.getStack();
                if (ip.getWidth() != stack.getWidth() || ip.getHeight() != stack.getHeight()
                        || ip.getBitDepth() != stack.getBitDepth())
                    throw new IOException("Unexpected size or bit depth in " + chanFile);
                stack.setProcessor(ip, target + 1);
                result.channels.put(target, chanFile);
                done++;
                if (progress != null)
                    progress.accept(done);
            }
        } finally {
            try {
                Files.deleteIfExists(runner);
                Files.deleteIfExists(runner.getParent());
            } catch (IOException ignored) {
                // temp files only
            }
        }

        // 4) Full replacement stack (unselected channels are unchanged)
        if (!new FileSaver(imp).saveAsTiffStack(result.fullStack.toString()))
            throw new IOException("Could not write " + result.fullStack);
        out.accept("");
        out.accept("Saved full SpillOT stack -> " + result.fullStack);
        return result;
    }

    // ------------------------------------------------------------------
    // Channel names and CSV
    // ------------------------------------------------------------------

    /** Which image axis holds the markers. */
    public enum Axis { CHANNEL, Z, T }

    /**
     * Marker stacks are often saved as plain multi-page TIFFs, which QuPath opens as a z-stack
     * or time series with a single channel. Like the Fiji plugin, use the longest axis.
     */
    public static Axis channelAxis(ImageServer<?> server) {
        int c = server.nChannels(), z = server.nZSlices(), t = server.nTimepoints();
        if (c >= z && c >= t)
            return Axis.CHANNEL;
        return z >= t ? Axis.Z : Axis.T;
    }

    /** Number of marker planes along {@link #channelAxis(ImageServer)}. */
    public static int nPlanes(ImageServer<?> server) {
        return switch (channelAxis(server)) {
            case CHANNEL -> server.nChannels();
            case Z -> server.nZSlices();
            case T -> server.nTimepoints();
        };
    }

    /** Marker names, made safe for a one-line CSV cell ("Ch1", "Ch2", ... when the file has none). */
    public static List<String> channelNames(ImageServer<?> server) {
        List<String> names = new ArrayList<>();
        boolean named = channelAxis(server) == Axis.CHANNEL;
        for (int c = 0; c < nPlanes(server); c++) {
            String n = named ? server.getChannel(c).getName() : null;
            n = n == null ? "" : n.replaceAll("[,\\r\\n]+", " ").trim();
            names.add(n.isEmpty() ? "Ch" + (c + 1) : n);
        }
        return names;
    }

    /**
     * Read a region as an ImageJ stack with one slice per marker, labelled with the marker names.
     * The z and t of {@code region} are used for whichever of those axes does not hold the markers.
     */
    public static ImagePlus readStack(ImageServer<BufferedImage> server, ImageRegion region, double downsample)
            throws IOException {
        Axis axis = channelAxis(server);
        int n = nPlanes(server);
        ImagePlus imp;
        if (axis == Axis.CHANNEL) {
            imp = IJTools.convertToImagePlus(server,
                    RegionRequest.createInstance(server.getPath(), downsample, region)).getImage();
        } else {
            ImageStack stack = null;
            ImagePlus first = null;
            for (int i = 0; i < n; i++) {
                int z = axis == Axis.Z ? i : region.getZ();
                int t = axis == Axis.T ? i : region.getT();
                ImagePlus plane = IJTools.convertToImagePlus(server, RegionRequest.createInstance(
                        server.getPath(), downsample, region.getX(), region.getY(),
                        region.getWidth(), region.getHeight(), z, t)).getImage();
                if (stack == null) {
                    first = plane;
                    stack = new ImageStack(plane.getWidth(), plane.getHeight());
                }
                stack.addSlice(plane.getStack().getProcessor(1));
            }
            imp = new ImagePlus(first.getTitle(), stack);
            imp.setCalibration(first.getCalibration());
        }
        if (imp.getStackSize() != n)
            throw new IOException("Expected " + n + " channel planes but read " + imp.getStackSize() + ".");
        // Always save as channels, so the result opens as a multichannel image
        imp.setDimensions(n, 1, 1);
        List<String> names = channelNames(server);
        for (int c = 0; c < n; c++)
            imp.getStack().setSliceLabel(names.get(c), c + 1);
        return imp;
    }

    /** Write the matrix: row = target, column = source, 1 = remove where patches match. */
    public static void writeCsv(Path csv, List<String> names, Map<Integer, Set<Integer>> sources)
            throws IOException {
        int n = names.size();
        StringBuilder sb = new StringBuilder();
        sb.append(',').append(String.join(",", names));
        for (int r = 0; r < n; r++) {
            sb.append('\n').append(names.get(r));
            Set<Integer> srcs = sources.getOrDefault(r, Set.of());
            for (int c = 0; c < n; c++)
                sb.append(',').append(r != c && srcs.contains(c) ? '1' : '0');
        }
        Files.writeString(csv, sb.toString(), StandardCharsets.UTF_8);
    }

    /**
     * Read an existing matrix so the dialog can be prefilled. Rows and columns are matched
     * by name (case-insensitive) and fall back to position when the CSV has one row or
     * column per channel. Only 1 and -1 count as selected. Returns an empty map on any problem.
     */
    public static Map<Integer, Set<Integer>> readCsv(Path csv, List<String> names) {
        Map<Integer, Set<Integer>> map = new TreeMap<>();
        try {
            if (csv == null || !Files.isRegularFile(csv))
                return map;
            List<String> lines = new ArrayList<>();
            for (String line : Files.readAllLines(csv, StandardCharsets.UTF_8))
                if (!line.isBlank())
                    lines.add(line.replace("\ufeff", ""));
            if (lines.size() < 2)
                return map;
            int n = names.size();
            String[] header = lines.get(0).split(",", -1);
            int[] colIdx = new int[Math.max(0, header.length - 1)];
            boolean colsByPos = colIdx.length >= n;
            for (int j = 0; j < colIdx.length; j++)
                colIdx[j] = matchName(header[j + 1], names, colsByPos ? j : -1);
            boolean rowsByPos = lines.size() - 1 >= n;
            for (int i = 1; i < lines.size(); i++) {
                String[] cells = lines.get(i).split(",", -1);
                int row = matchName(cells[0], names, rowsByPos ? i - 1 : -1);
                if (row < 0)
                    continue;
                for (int j = 0; j < colIdx.length && j + 1 < cells.length; j++) {
                    int col = colIdx[j];
                    if (col < 0 || col == row)
                        continue;
                    try {
                        if (Math.abs(Math.abs(Double.parseDouble(cells[j + 1].trim())) - 1.0) <= 1e-6)
                            map.computeIfAbsent(row, k -> new TreeSet<>()).add(col);
                    } catch (NumberFormatException ignored) {
                        // blank or non-numeric = not selected
                    }
                }
            }
        } catch (Exception e) {
            return new TreeMap<>();
        }
        return map;
    }

    private static int matchName(String cell, List<String> names, int fallback) {
        String s = cell == null ? "" : cell.trim();
        if (!s.isEmpty())
            for (int i = 0; i < names.size(); i++)
                if (names.get(i).equalsIgnoreCase(s))
                    return i;
        return fallback >= 0 && fallback < names.size() ? fallback : -1;
    }

    /** File-name-safe version of an image name. */
    public static String sanitize(String name) {
        String s = name == null ? "" : name.replaceAll("[^A-Za-z0-9._-]+", "_").replaceAll("^_+|_+$", "");
        return s.isEmpty() ? "image" : s;
    }

    // ------------------------------------------------------------------
    // Python / conda helpers
    // ------------------------------------------------------------------

    static boolean isWindows() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("windows");
    }

    /** Path of the Python executable inside a conda environment root, or null. */
    public static Path pythonFromEnv(String envRoot) {
        if (envRoot == null || envRoot.isBlank())
            return null;
        Path root = Paths.get(envRoot.trim());
        return isWindows() ? root.resolve("python.exe") : root.resolve("bin").resolve("python");
    }

    /** Look in the usual places for a conda environment called {@code envName}; "" if not found. */
    public static String guessCondaEnv(String envName) {
        Set<Path> candidates = new LinkedHashSet<>();
        try {
            String prefix = System.getenv("CONDA_PREFIX");
            if (prefix != null && Paths.get(prefix).getFileName().toString().equalsIgnoreCase(envName))
                candidates.add(Paths.get(prefix));
            Path home = Paths.get(System.getProperty("user.home"));
            for (String d : List.of("miniconda3", "miniforge3", "mambaforge", "anaconda3", "micromamba",
                    "Miniconda3", "Anaconda3", "opt/anaconda3", "opt/miniconda3"))
                candidates.add(home.resolve(d).resolve("envs").resolve(envName));
            candidates.add(home.resolve(".conda").resolve("envs").resolve(envName));
            if (isWindows()) {
                candidates.add(Paths.get("C:\\ProgramData\\Anaconda3\\envs", envName));
                candidates.add(Paths.get("C:\\ProgramData\\miniconda3\\envs", envName));
            } else {
                for (String d : List.of("/opt/anaconda3", "/opt/miniconda3", "/opt/miniforge3",
                        "/opt/homebrew/Caskroom/miniconda/base", "/opt/homebrew/Caskroom/miniforge/base"))
                    candidates.add(Paths.get(d, "envs", envName));
            }
            for (Path c : candidates) {
                Path py = pythonFromEnv(c.toString());
                if (py != null && Files.exists(py))
                    return c.toString();
            }
        } catch (Exception ignored) {
            // fall through
        }
        return "";
    }

    /** Copy the bundled SpillOT.py to a temporary folder and return its path. */
    static Path extractRunner() throws IOException {
        Path dir = Files.createTempDirectory("spillot-qupath-");
        Path runner = dir.resolve("SpillOT.py");
        try (InputStream in = SpillOT.class.getResourceAsStream("SpillOT.py")) {
            if (in == null)
                throw new IOException("SpillOT.py is missing from the extension jar.");
            Files.copy(in, runner, StandardCopyOption.REPLACE_EXISTING);
        }
        return runner;
    }

    private static void runProcess(List<String> cmd, String envRoot, Consumer<String> out,
                                   BooleanSupplier stop) throws IOException, InterruptedException {
        ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true);
        Map<String, String> env = pb.environment();
        String pathKey = "PATH";
        for (String k : env.keySet())
            if (k.equalsIgnoreCase("PATH"))
                pathKey = k;
        Path root = Paths.get(envRoot.trim());
        List<String> extra = isWindows()
                ? List.of(root.resolve("Library").resolve("bin").toString(),
                          root.resolve("Scripts").toString(), root.toString())
                : List.of(root.resolve("bin").toString());
        env.put(pathKey, String.join(File.pathSeparator, extra) + File.pathSeparator
                + env.getOrDefault(pathKey, ""));
        env.putIfAbsent("OMP_NUM_THREADS", "1");
        env.putIfAbsent("MKL_NUM_THREADS", "1");
        env.put("PYTHONUNBUFFERED", "1");

        Process proc = pb.start();
        Deque<String> tail = new ArrayDeque<>();
        Thread reader = new Thread(() -> {
            try (BufferedReader r = new BufferedReader(
                    new InputStreamReader(proc.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = r.readLine()) != null) {
                    synchronized (tail) {
                        tail.addLast(line);
                        if (tail.size() > 40)
                            tail.removeFirst();
                    }
                    out.accept(line);
                }
            } catch (IOException ignored) {
                // stream closed when the process is killed
            }
        }, "spillot-output");
        reader.setDaemon(true);
        reader.start();

        try {
            while (!proc.waitFor(200, java.util.concurrent.TimeUnit.MILLISECONDS)) {
                if (stop.getAsBoolean() || Thread.currentThread().isInterrupted())
                    throw new InterruptedException("SpillOT was cancelled.");
            }
        } catch (InterruptedException e) {
            proc.descendants().forEach(ProcessHandle::destroyForcibly);
            proc.destroyForcibly();
            throw e;
        }
        reader.join(2000);
        if (proc.exitValue() != 0) {
            String msg;
            synchronized (tail) {
                msg = String.join("\n", tail);
            }
            throw new IOException("The SpillOT Python backend failed (exit " + proc.exitValue() + ").\n\n" + msg);
        }
    }
}
