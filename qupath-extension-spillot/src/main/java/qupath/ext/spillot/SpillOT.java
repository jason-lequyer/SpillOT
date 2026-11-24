// Copyright 2025, Jason Lequyer, Ferris Nowlan and Laurence Pelletier.
// All rights reserved.
// Sinai Health System - Lunenfeld Tanenbaum Research Institute
// 600 University Avenue, Room 1070, Toronto, ON M5G 1X5, Canada

package qupath.ext.spillot;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import ij.ImagePlus;
import ij.ImageStack;
import qupath.imagej.tools.IJTools;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.PixelType;
import qupath.lib.regions.ImageRegion;
import qupath.lib.regions.RegionRequest;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/**
 * GUI-free core of the SpillOT QuPath extension.
 * <p>
 * Streams the selected region through bounded export tiles and a memory-budgeted
 * Python runner. It writes the input image, spillover CSV, cleaned channels and
 * a full replacement stack.
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
    private static final Gson JSON = new GsonBuilder().serializeNulls().create();
    private static final String PROGRESS_PREFIX = "SPILLOT_PROGRESS\t";
    private static final double EXPORT_FRACTION = 0.15;

    private SpillOT() {}

    /** Settings for one SpillOT run. Channel indices are 0-based. */
    public static class Params {
        /** Even integer &ge; 4. Smaller = faster and more aggressive. */
        public int patchSize = DEFAULT_PATCH_SIZE;
        /** Worker processes used by the Python backend. */
        public int workers = Math.max(1, Runtime.getRuntime().availableProcessors());
        /** Python working-memory budget in MB; zero lets the backend choose from available RAM. */
        public int memoryBudgetMB = 0;
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
        /** Exported input stack (OME-TIFF). */
        public Path input;
        /** Spillover matrix (row = target, column = source, 1 = remove where similar). */
        public Path csv;
        /** Original stack with the selected channels replaced by their cleaned versions. */
        public Path fullStack;
        /** One cleaned 2D TIFF per target channel, keyed by 0-based channel index. */
        public Map<Integer, Path> channels = new TreeMap<>();
    }

    /** Progress for the complete run, including export and final output writing. */
    public static final class Progress {
        public final String phase;
        public final long completed;
        public final long total;
        public final double fraction;
        /** Seconds remaining; null until enough timing information is available. */
        public final Double etaSeconds;
        public final String message;

        public Progress(String phase, long completed, long total, double fraction,
                        Double etaSeconds, String message) {
            this.phase = phase;
            this.completed = completed;
            this.total = total;
            this.fraction = Double.isFinite(fraction) ? Math.max(0, Math.min(1, fraction)) : 0;
            this.etaSeconds = etaSeconds != null && Double.isFinite(etaSeconds) && etaSeconds >= 0
                    ? etaSeconds : null;
            this.message = message == null ? "" : message;
        }
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
        return run(server, p, log, progress, cancelled, null);
    }

    /** As {@link #run(ImageServer, Params, Consumer, Consumer, BooleanSupplier)}, with detailed progress. */
    public static Result run(ImageServer<BufferedImage> server, Params p, Consumer<String> log,
                             Consumer<Integer> progress, BooleanSupplier cancelled,
                             Consumer<Progress> detailedProgress)
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
        if (p.memoryBudgetMB < 0)
            throw new IllegalArgumentException("Memory budget must be >= 0 MB (0 = automatic).");
        if (p.outputDir == null || p.baseName == null || p.baseName.isBlank())
            throw new IllegalArgumentException("An output directory and file name are required.");
        checkCancelled(stop);

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

        if (server.getPixelType() != PixelType.UINT8 && server.getPixelType() != PixelType.UINT16
                && server.getPixelType() != PixelType.FLOAT32)
            throw new IllegalArgumentException("SpillOT supports uint8, uint16 and float32 images. "
                    + "This image is " + server.getPixelType() + ". Convert a copy explicitly to a supported pixel type "
                    + "before running so any precision change is intentional.");

        return runLowMemory(server, p, region, sources, python, result, out, progress, stop, detailedProgress);
    }

    private static Result runLowMemory(ImageServer<BufferedImage> server, Params p, ImageRegion region,
                                       Map<Integer, Set<Integer>> sources, Path python, Result result,
                                       Consumer<String> out, Consumer<Integer> channelProgress,
                                       BooleanSupplier stop, Consumer<Progress> detailedProgress)
            throws IOException, InterruptedException {
        int channels = nPlanes(server);
        int tileSide = exportTileSide(channels);
        List<String> names = channelNames(server);
        Path workDir = null, rawFile = null, manifestFile = null, runner = null;
        ImagePlus firstTile = null;
        try {
            report(detailedProgress, new Progress("export", 0, 1, 0, null,
                    "Preparing tiled export; estimating remaining time"));
            checkCancelled(stop);
            ImageRegion firstRegion = ImageRegion.createInstance(region.getX(), region.getY(),
                    Math.min(tileSide, region.getWidth()), Math.min(tileSide, region.getHeight()),
                    region.getZ(), region.getT());
            firstTile = readStack(server, firstRegion, 1.0);
            int bitDepth = firstTile.getBitDepth();
            String dtype = switch (bitDepth) {
                case 8 -> "|u1";
                case 16 -> "<u2";
                case 32 -> "<f4";
                default -> throw new IOException("Unsupported exported pixel type: " + bitDepth + " bit");
            };
            int bytesPerPixel = bitDepth / 8;
            long area = Math.multiplyExact((long) region.getWidth(), region.getHeight());
            long planeBytes = Math.multiplyExact(area, bytesPerPixel);
            long stackBytes = Math.multiplyExact(planeBytes, channels);
            // Input raw, working output raw, original TIFF, combined TIFF, selected-channel TIFFs,
            // a float result plane and optional saturation mask. Include a conservative safety margin.
            long payload = Math.addExact(Math.multiplyExact(stackBytes, 4),
                    Math.addExact(Math.multiplyExact(planeBytes, sources.size()), Math.multiplyExact(area, 5)));
            long required = Math.addExact(payload, Math.max(256L * 1024 * 1024, payload / 10));
            long free = Files.getFileStore(p.outputDir.toAbsolutePath()).getUsableSpace();
            if (free < required)
                throw new IOException("Not enough free space in " + p.outputDir.toAbsolutePath() + ".\n"
                        + "This run needs approximately " + formatGiB(required)
                        + " GiB of free disk space for temporary data and results; " + formatGiB(free)
                        + " GiB is available.\nChoose an output folder on a larger disk or free space, then run again.");

            workDir = Files.createTempDirectory(p.outputDir.toAbsolutePath(), ".spillot-work-");
            rawFile = workDir.resolve("input.raw");
            manifestFile = workDir.resolve("manifest.json");
            out.accept("Exporting " + region.getWidth() + " x " + region.getHeight()
                    + " px, " + channels + " channels in tiles up to " + tileSide + " x " + tileSide + " px.");
            out.accept("RAM use is planned automatically; tiles run serially. Large images may still take a long time.");
            exportRaw(server, region, channels, tileSide, bitDepth, rawFile, stackBytes, firstTile,
                    stop, detailedProgress);
            firstTile = null; // exportRaw flushes every tile, including this one
            checkCancelled(stop);
            Path stagedCsv = workDir.resolve("channels.csv");
            writeCsv(stagedCsv, names, sources);

            Map<String, Object> manifest = buildManifest(server, p, region, names, rawFile, stagedCsv, dtype);
            Files.writeString(manifestFile, JSON.toJson(manifest), StandardCharsets.UTF_8);

            runner = extractRunner();
            report(detailedProgress, new Progress("planning", 0, 1, EXPORT_FRACTION, null,
                    "Planning memory use; estimating remaining time"));
            runProcess(List.of(python.toString(), runner.toString(), "--manifest", manifestFile.toString()),
                    p.envPath, out, stop, update -> report(detailedProgress,
                            new Progress(update.phase, update.completed, update.total,
                                    EXPORT_FRACTION + (0.999 - EXPORT_FRACTION) * update.fraction,
                                    update.etaSeconds, update.message)), workDir.resolve("cancel.requested"));
            // A successful backend has already committed all outputs. A later cancel click cannot undo that.
            requireOutput(result.input);
            requireOutput(result.fullStack);
            requireOutput(result.csv);
            int done = 0;
            for (int target : sources.keySet()) {
                Path path = p.outputDir.resolve(p.baseName + "_Channel_" + (target + 1) + "_SpillOT.tif");
                requireOutput(path);
                result.channels.put(target, path);
                if (channelProgress != null)
                    channelProgress.accept(++done);
            }
            out.accept("Saved full SpillOT stack -> " + result.fullStack);
            report(detailedProgress, new Progress("done", sources.size(), sources.size(), 1, 0.0, "Finished"));
            return result;
        } finally {
            if (firstTile != null)
                firstTile.flush();
            cleanupRunner(runner);
            try {
                recoverPublication(workDir, p.outputDir);
                cleanupWorkDirectory(workDir, out);
            } catch (IOException e) {
                String message = "Could not recover an interrupted output publication. Temporary files and backups "
                        + "have been preserved at " + workDir + ". " + e.getMessage();
                out.accept(message);
                throw new IOException(message, e);
            }
        }
    }

    static Map<String, Object> buildManifest(ImageServer<BufferedImage> server, Params p, ImageRegion region,
                                              List<String> names, Path rawFile, Path csvFile, String dtype) {
        int channels = nPlanes(server);
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("input_raw", rawFile.toAbsolutePath().toString());
        manifest.put("shape", List.of(channels, region.getHeight(), region.getWidth()));
        manifest.put("dtype", dtype);
        manifest.put("output_dir", p.outputDir.toAbsolutePath().toString());
        manifest.put("base_name", p.baseName);
        manifest.put("csv_path", csvFile.toAbsolutePath().toString());
        manifest.put("scratch_dir", rawFile.toAbsolutePath().getParent().toString());
        manifest.put("cancel_path", rawFile.toAbsolutePath().getParent().resolve("cancel.requested").toString());
        manifest.put("channel_names", names);
        if (channelAxis(server) == Axis.CHANNEL) {
            List<Integer> colors = new ArrayList<>();
            for (int c = 0; c < channels; c++)
                colors.add(server.getChannel(c).getColor());
            manifest.put("channel_colors", colors);
        }
        var calibration = server.getPixelCalibration();
        double pixelWidth = calibration.getPixelWidthMicrons();
        double pixelHeight = calibration.getPixelHeightMicrons();
        manifest.put("pixel_size_um", Double.isFinite(pixelWidth) && Double.isFinite(pixelHeight)
                && pixelWidth > 0 && pixelHeight > 0 ? List.of(pixelWidth, pixelHeight) : null);
        manifest.put("origin_xy", List.of(region.getX(), region.getY()));
        manifest.put("patch_size", p.patchSize);
        manifest.put("workers", p.workers);
        manifest.put("ignore_overexposed", p.ignoreOverexposed);
        manifest.put("memory_budget_mb", p.memoryBudgetMB);
        return manifest;
    }

    /** Keep conversion buffers bounded even when a slide contains many channels. */
    static int exportTileSide(int channels) {
        Runtime runtime = Runtime.getRuntime();
        long available = Math.max(1, runtime.maxMemory() - (runtime.totalMemory() - runtime.freeMemory()));
        long budget = Math.max(1, Math.min(64L * 1024 * 1024, available / 8));
        long bytesPerPixel = Math.max(32L, 16L * channels);
        int side = 256;
        while (side > 16 && (long) side * side * bytesPerPixel > budget)
            side /= 2;
        return side;
    }

    static void exportRaw(ImageServer<BufferedImage> server, ImageRegion region, int channels,
                                  int tileSide, int bitDepth, Path rawFile, long stackBytes,
                                  ImagePlus firstTile, BooleanSupplier stop, Consumer<Progress> progress)
            throws IOException, InterruptedException {
        int width = region.getWidth(), height = region.getHeight();
        long total = ((long) width + tileSide - 1) / tileSide * (((long) height + tileSide - 1) / tileSide);
        long done = 0, started = System.nanoTime(), lastReport = 0;
        try (FileChannel channel = FileChannel.open(rawFile, StandardOpenOption.CREATE_NEW,
                StandardOpenOption.WRITE)) {
            channel.position(stackBytes - 1);
            channel.write(ByteBuffer.wrap(new byte[] {0}));
            for (int y = 0; y < height; y += tileSide) {
                for (int x = 0; x < width; x += tileSide) {
                    checkCancelled(stop);
                    int w = Math.min(tileSide, width - x), h = Math.min(tileSide, height - y);
                    ImagePlus tile = x == 0 && y == 0 ? firstTile : readStack(server,
                            ImageRegion.createInstance(region.getX() + x, region.getY() + y, w, h,
                                    region.getZ(), region.getT()), 1.0);
                    try {
                        ImageStack stack = tile.getStack();
                        if (stack.getSize() != channels || stack.getBitDepth() != bitDepth
                                || stack.getWidth() != w || stack.getHeight() != h)
                            throw new IOException("Image dimensions or pixel type changed while exporting a tile.");
                        writeRawTile(channel, stack, x, y, width, height, bitDepth, stop);
                    } finally {
                        tile.flush();
                    }
                    done++;
                    long now = System.nanoTime();
                    if (done == total || now - lastReport >= 250_000_000L) {
                        double elapsed = (now - started) / 1_000_000_000.0;
                        Double eta = done >= 2 && elapsed >= 2 ? elapsed / done * (total - done) : null;
                        report(progress, new Progress("export", done, total,
                                EXPORT_FRACTION * done / total, eta,
                                "Exporting tile " + done + " of " + total));
                        lastReport = now;
                    }
                }
            }
            channel.force(false);
        }
    }

    /** Write a tile into its CYX position without allocating a full image or even a full-channel row. */
    static void writeRawTile(FileChannel channel, ImageStack stack, int x, int y, int fullWidth,
                             int fullHeight, int bitDepth, BooleanSupplier stop)
            throws IOException, InterruptedException {
        int w = stack.getWidth(), h = stack.getHeight(), bytes = bitDepth / 8;
        if (bitDepth != 8 && bitDepth != 16 && bitDepth != 32)
            throw new IOException("Unsupported pixel depth: " + bitDepth);
        if (x < 0 || y < 0 || (long) x + w > fullWidth || (long) y + h > fullHeight)
            throw new IOException("Tile lies outside the exported image.");
        if (stack.getBitDepth() != bitDepth)
            throw new IOException("Tile pixel depth does not match the raw output.");
        ByteBuffer row = ByteBuffer.allocate(w * bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int c = 0; c < stack.getSize(); c++) {
            Object pixels = stack.getPixels(c + 1);
            for (int yy = 0; yy < h; yy++) {
                checkCancelled(stop);
                row.clear();
                int offset = yy * w;
                if (pixels instanceof byte[] values)
                    row.put(values, offset, w);
                else if (pixels instanceof short[] values)
                    for (int xx = 0; xx < w; xx++)
                        row.putShort(values[offset + xx]);
                else if (pixels instanceof float[] values)
                    for (int xx = 0; xx < w; xx++)
                        row.putFloat(values[offset + xx]);
                else
                    throw new IOException("Unsupported exported pixel buffer: " + pixels.getClass());
                row.flip();
                long position = (((long) c * fullHeight + y + yy) * fullWidth + x) * bytes;
                while (row.hasRemaining()) {
                    checkCancelled(stop);
                    position += channel.write(row, position);
                }
            }
        }
    }

    private static String formatGiB(long bytes) {
        return String.format(Locale.ROOT, "%.1f", bytes / (1024.0 * 1024 * 1024));
    }

    /** This directory was created uniquely by this run; never recurse through user output folders. */
    private static void cleanupWorkDirectory(Path directory, Consumer<String> out) {
        if (directory == null || !Files.exists(directory))
            return;
        try (var paths = Files.walk(directory)) {
            for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
                try {
                    Files.deleteIfExists(path);
                } catch (IOException e) {
                    out.accept("Could not remove temporary file " + path + ": " + e.getMessage());
                }
            }
        } catch (IOException e) {
            out.accept("Could not clean temporary directory " + directory + ": " + e.getMessage());
        }
    }

    /** Recover the helper's durable publication journal before deleting any caller-owned scratch files. */
    static void recoverPublication(Path directory, Path outputDirectory) throws IOException {
        if (directory == null)
            return;
        Path root = directory.toAbsolutePath().normalize();
        Path journal = root.resolve("publication.json");
        if (!Files.exists(journal))
            return;
        try {
            JsonObject record = JsonParser.parseString(Files.readString(journal, StandardCharsets.UTF_8)).getAsJsonObject();
            var version = record.get("version");
            if (version == null || !version.isJsonPrimitive() || !version.getAsJsonPrimitive().isNumber()
                    || version.getAsBigDecimal().compareTo(BigDecimal.ONE) != 0)
                throw new IOException("Unknown publication journal version.");
            String state = record.get("state").getAsString();
            if (!state.equals("pending") && !state.equals("committed"))
                throw new IOException("Unknown publication state: " + state);
            Path output = outputDirectory.toAbsolutePath().normalize();
            if (!Paths.get(record.get("output_dir").getAsString()).toAbsolutePath().normalize().equals(output))
                throw new IOException("Publication journal does not match the output directory.");
            Path backups = Paths.get(record.get("backup_dir").getAsString()).toAbsolutePath().normalize();
            if (!root.equals(backups.getParent())
                    || !backups.getFileName().toString().startsWith(".spillot-publication-backups-"))
                throw new IOException("Publication backups are outside the run's temporary directory.");
            List<Path[]> entries = new ArrayList<>();
            List<Boolean> existed = new ArrayList<>();
            // Validate every path before performing any recovery writes.
            for (var element : record.getAsJsonArray("entries")) {
                JsonObject entry = element.getAsJsonObject();
                Path destination = Paths.get(entry.get("destination").getAsString()).toAbsolutePath().normalize();
                Path backup = Paths.get(entry.get("backup").getAsString()).toAbsolutePath().normalize();
                Path staged = Paths.get(entry.get("staged").getAsString()).toAbsolutePath().normalize();
                if (!output.equals(destination.getParent()) || !backups.equals(backup.getParent())
                        || !staged.startsWith(root) || staged.equals(root))
                    throw new IOException("Publication journal contains a path outside this run.");
                var previouslyExisted = entry.get("existed");
                if (previouslyExisted == null || !previouslyExisted.isJsonPrimitive()
                        || !previouslyExisted.getAsJsonPrimitive().isBoolean())
                    throw new IOException("Publication journal 'existed' values must be JSON booleans.");
                entries.add(new Path[] {destination, backup, staged});
                existed.add(previouslyExisted.getAsBoolean());
            }
            if (state.equals("pending")) {
                for (int i = entries.size() - 1; i >= 0; i--) {
                    Path[] entry = entries.get(i);
                    if (Files.exists(entry[1], LinkOption.NOFOLLOW_LINKS)) {
                        try {
                            Files.move(entry[1], entry[0], StandardCopyOption.ATOMIC_MOVE,
                                    StandardCopyOption.REPLACE_EXISTING);
                        } catch (AtomicMoveNotSupportedException e) {
                            Files.move(entry[1], entry[0], StandardCopyOption.REPLACE_EXISTING);
                        }
                    } else if (existed.get(i)) {
                        if (!Files.exists(entry[0], LinkOption.NOFOLLOW_LINKS))
                            throw new IOException("Both the prior output and its backup are missing: " + entry[0]);
                    } else if (!Files.exists(entry[2], LinkOption.NOFOLLOW_LINKS)) {
                        Files.deleteIfExists(entry[0]);
                    }
                }
            }
            // Pending recovery is now complete, or a committed run already owns its final outputs.
            // Keeping the journal until this point makes failed/interrupted recovery safely repeatable.
            Files.delete(journal);
        } catch (RuntimeException e) {
            throw new IOException("Invalid publication journal: " + journal, e);
        }
    }

    private static void requireOutput(Path path) throws IOException {
        if (!Files.isRegularFile(path) || Files.size(path) == 0)
            throw new IOException("SpillOT result not found or empty: " + path);
    }

    private static void checkCancelled(BooleanSupplier stop) throws InterruptedException {
        if (stop.getAsBoolean() || Thread.currentThread().isInterrupted())
            throw new InterruptedException("SpillOT was cancelled.");
    }

    private static void report(Consumer<Progress> consumer, Progress progress) {
        if (consumer != null)
            consumer.accept(progress);
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

    /** Copy the bundled backend files to a temporary folder and return the memory-aware runner. */
    static Path extractRunner() throws IOException {
        Path dir = Files.createTempDirectory("spillot-qupath-");
        Path runner = dir.resolve("SpillOT_low_memory.py");
        try {
            for (String name : List.of("SpillOT.py", "SpillOT_low_memory.py")) {
                try (InputStream in = SpillOT.class.getResourceAsStream(name)) {
                    if (in == null)
                        throw new IOException(name + " is missing from the extension jar.");
                    Files.copy(in, dir.resolve(name), StandardCopyOption.REPLACE_EXISTING);
                }
            }
        } catch (IOException e) {
            cleanupRunner(runner);
            throw e;
        }
        return runner;
    }

    private static void cleanupRunner(Path runner) {
        if (runner == null)
            return;
        Path directory = runner.getParent();
        try {
            Files.deleteIfExists(directory.resolve("SpillOT.py"));
            Files.deleteIfExists(directory.resolve("SpillOT_low_memory.py"));
            Files.deleteIfExists(directory);
        } catch (IOException ignored) {
            // Only private temporary runner files are involved.
        }
    }

    private static void runProcess(List<String> cmd, String envRoot, Consumer<String> out,
                                   BooleanSupplier stop, Consumer<Progress> progress, Path cancelPath)
            throws IOException, InterruptedException {
        checkCancelled(stop);
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
        env.put("PYTHONDONTWRITEBYTECODE", "1");

        Process proc = pb.start();
        AtomicBoolean backendComplete = new AtomicBoolean(false);
        Set<ProcessHandle> descendants = new LinkedHashSet<>();
        Deque<String> tail = new ArrayDeque<>();
        Thread reader = new Thread(() -> {
            try (BufferedReader r = new BufferedReader(
                    new InputStreamReader(proc.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                String lastPhase = "";
                while ((line = r.readLine()) != null) {
                    Progress update = parseProgressLine(line);
                    if (update != null) {
                        if ("complete".equals(update.phase) && update.fraction >= 1)
                            backendComplete.set(true);
                        report(progress, update);
                        if (!update.phase.equals(lastPhase) && !update.message.isBlank())
                            out.accept(update.message);
                        lastPhase = update.phase;
                        continue;
                    }
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
            descendants.addAll(proc.descendants().toList());
            while (!proc.waitFor(200, java.util.concurrent.TimeUnit.MILLISECONDS)) {
                descendants.addAll(proc.descendants().toList());
                if (!backendComplete.get() && (stop.getAsBoolean() || Thread.currentThread().isInterrupted()))
                    throw new InterruptedException("SpillOT was cancelled.");
            }
        } catch (InterruptedException e) {
            terminateProcessTree(proc, descendants, cancelPath);
            try {
                reader.join(2000);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
            if (cancelPath != null && !proc.isAlive() && proc.exitValue() == 0 && backendComplete.get())
                return; // The commit point won a race with the cancel request.
            throw e;
        }
        try {
            reader.join(2000);
        } catch (InterruptedException e) {
            terminateProcessTree(proc, descendants, cancelPath);
            if (cancelPath == null || proc.exitValue() != 0 || !backendComplete.get())
                throw e;
        }
        if (proc.exitValue() != 0) {
            terminateProcessTree(proc, descendants, cancelPath);
            String msg;
            synchronized (tail) {
                msg = String.join("\n", tail);
            }
            throw new IOException("The SpillOT Python backend failed (exit " + proc.exitValue() + ").\n\n" + msg);
        }
    }

    /** Ignore ordinary log lines and malformed status records without interrupting output draining. */
    static Progress parseProgressLine(String line) {
        if (line == null || !line.startsWith(PROGRESS_PREFIX))
            return null;
        try {
            JsonObject object = JsonParser.parseString(line.substring(PROGRESS_PREFIX.length())).getAsJsonObject();
            String phase = object.has("phase") ? object.get("phase").getAsString() : "processing";
            long completed = object.has("completed") ? object.get("completed").getAsLong() : 0;
            long total = object.has("total") ? object.get("total").getAsLong() : 0;
            double fraction = object.has("fraction") ? object.get("fraction").getAsDouble() : 0;
            Double eta = object.has("eta_seconds") && !object.get("eta_seconds").isJsonNull()
                    ? object.get("eta_seconds").getAsDouble() : null;
            String message = object.has("message") ? object.get("message").getAsString() : "";
            return new Progress(phase, completed, total, fraction, eta, message);
        } catch (RuntimeException e) {
            return null;
        }
    }

    private static void terminateProcessTree(Process process, Set<ProcessHandle> knownDescendants, Path cancelPath) {
        knownDescendants.addAll(process.descendants().toList());
        boolean cooperative = false;
        if (cancelPath != null && process.isAlive()) {
            try {
                Files.writeString(cancelPath, "cancel\n", StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
                cooperative = true;
            } catch (IOException ignored) {
                // If a marker cannot be written, fall back to the platform's process termination.
            }
        }
        // A marker lets Python close files and roll back publication on every platform, including Windows.
        // If the marker cannot be written, Unix destroy still requests graceful SIGTERM.
        if (!cooperative && process.isAlive())
            process.destroy();
        boolean interrupted = false;
        try {
            long deadline = System.nanoTime() + 5_000_000_000L;
            while (process.isAlive() && System.nanoTime() < deadline) {
                knownDescendants.addAll(process.descendants().toList());
                process.waitFor(100, java.util.concurrent.TimeUnit.MILLISECONDS);
            }
        } catch (InterruptedException e) {
            interrupted = true;
        } finally {
            knownDescendants.addAll(process.descendants().toList());
            if (process.isAlive())
                process.destroyForcibly();
            for (ProcessHandle child : knownDescendants) {
                if (child.isAlive())
                    child.destroyForcibly();
            }
            try {
                process.waitFor(2, java.util.concurrent.TimeUnit.SECONDS);
                long deadline = System.nanoTime() + 2_000_000_000L;
                while (knownDescendants.stream().anyMatch(ProcessHandle::isAlive)
                        && System.nanoTime() < deadline)
                    Thread.sleep(25);
            } catch (InterruptedException e) {
                interrupted = true;
            }
            if (interrupted)
                Thread.currentThread().interrupt();
        }
    }
}
