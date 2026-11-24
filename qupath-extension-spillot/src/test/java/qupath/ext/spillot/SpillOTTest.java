package qupath.ext.spillot;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ShortProcessor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import qupath.lib.color.ColorModelFactory;
import qupath.lib.images.servers.ImageChannel;
import qupath.lib.images.servers.ImageServerMetadata;
import qupath.lib.images.servers.PixelType;
import qupath.lib.images.servers.WrappedBufferedImageServer;
import qupath.lib.regions.ImageRegion;
import qupath.lib.regions.RegionRequest;

import java.awt.image.BandedSampleModel;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferUShort;
import java.awt.image.Raster;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class SpillOTTest {
    @TempDir Path directory;

    @Test void unsigned16TilesUseCorrectPlaneAndRegionOffsets() throws Exception {
        ImageStack stack = new ImageStack(2, 2);
        stack.addSlice(new ShortProcessor(2, 2, new short[] {1, 257, (short) 32768, (short) 65535}, null));
        stack.addSlice(new ShortProcessor(2, 2, new short[] {101, 102, 103, 104}, null));
        Path file = directory.resolve("pixels.raw");
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
            channel.position(5 * 6 * 2 * 2 - 1);
            channel.write(ByteBuffer.wrap(new byte[] {0}));
            SpillOT.writeRawTile(channel, stack, 1, 2, 5, 6, 16, () -> false);
        }
        ByteBuffer data = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN);
        int[][] expected = {{1, 257, 32768, 65535}, {101, 102, 103, 104}};
        for (int c = 0; c < 2; c++) {
            for (int y = 0; y < 6; y++) {
                for (int x = 0; x < 5; x++) {
                    int value = x >= 1 && x < 3 && y >= 2 && y < 4
                            ? expected[c][(y - 2) * 2 + x - 1] : 0;
                    assertEquals(value, data.getShort(((c * 6 + y) * 5 + x) * 2) & 65535);
                }
            }
        }
    }

    @Test void byteAndFloatBuffersPreserveValues() throws Exception {
        ImageStack bytes = new ImageStack(3, 1);
        bytes.addSlice(new ByteProcessor(3, 1, new byte[] {0, 127, (byte) 255}, null));
        Path byteFile = directory.resolve("byte.raw");
        try (FileChannel channel = FileChannel.open(byteFile, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
            SpillOT.writeRawTile(channel, bytes, 0, 0, 3, 1, 8, () -> false);
        }
        assertArrayEquals(new byte[] {0, 127, (byte) 255}, Files.readAllBytes(byteFile));
        ImageStack floats = new ImageStack(3, 1);
        floats.addSlice(new FloatProcessor(3, 1, new float[] {-17.25f, 0.125f, 65536.5f}));
        Path floatFile = directory.resolve("float.raw");
        try (FileChannel channel = FileChannel.open(floatFile, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
            SpillOT.writeRawTile(channel, floats, 0, 0, 3, 1, 32, () -> false);
        }
        ByteBuffer data = ByteBuffer.wrap(Files.readAllBytes(floatFile)).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(-17.25f, data.getFloat());
        assertEquals(0.125f, data.getFloat());
        assertEquals(65536.5f, data.getFloat());
    }

    @Test void cancellationStopsBeforeWritingAnyPixels() throws Exception {
        ImageStack stack = new ImageStack(2, 2);
        stack.addSlice(new ByteProcessor(2, 2));
        Path file = directory.resolve("cancel.raw");
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
            assertThrows(InterruptedException.class,
                    () -> SpillOT.writeRawTile(channel, stack, 0, 0, 2, 2, 8, () -> true));
        }
        assertEquals(0, Files.size(file));
    }

    @Test void outOfBoundsTileIsRejected() throws Exception {
        ImageStack stack = new ImageStack(2, 2);
        stack.addSlice(new ByteProcessor(2, 2));
        try (FileChannel channel = FileChannel.open(directory.resolve("invalid.raw"),
                StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
            assertThrows(java.io.IOException.class,
                    () -> SpillOT.writeRawTile(channel, stack, 1, 0, 2, 2, 8, () -> false));
        }
    }

    @Test void progressProtocolSupportsWarmupAndMeasuredEta() {
        assertNull(SpillOT.parseProgressLine("ordinary backend log"));
        assertNull(SpillOT.parseProgressLine("SPILLOT_PROGRESS\t{broken"));
        var warming = SpillOT.parseProgressLine("SPILLOT_PROGRESS\t{\"phase\":\"processing\","
                + "\"completed\":0,\"total\":10,\"fraction\":0.1,\"eta_seconds\":null,\"message\":\"Warming up\"}");
        assertNotNull(warming);
        assertNull(warming.etaSeconds);
        assertEquals(0.1, warming.fraction);
        var measured = SpillOT.parseProgressLine("SPILLOT_PROGRESS\t{\"phase\":\"processing\","
                + "\"completed\":3,\"total\":10,\"fraction\":0.4,\"eta_seconds\":75.5,\"message\":\"Tile 3\"}");
        assertNotNull(measured);
        assertEquals(75.5, measured.etaSeconds);
        assertEquals(3, measured.completed);
        assertEquals(10, measured.total);
        assertEquals("Tile 3", measured.message);
        assertEquals(0, new SpillOT.Progress("x", 0, 1, Double.NaN, -1.0, null).fraction);
        assertNull(new SpillOT.Progress("x", 0, 1, 0, Double.POSITIVE_INFINITY, "").etaSeconds);
    }

    @Test void selectedRegionAndManifestKeepChannelsCalibrationAndOrigin() throws Exception {
        try (var server = sampleServer()) {
            ImageRegion region = ImageRegion.createInstance(7, 9, 13, 15, 0, 0);
            ImagePlus tile = SpillOT.readStack(server, region, 1.0);
            assertEquals(2, tile.getStackSize());
            assertEquals(13, tile.getWidth());
            assertEquals(15, tile.getHeight());
            for (int c = 0; c < 2; c++) {
                assertEquals(c * 10000 + 9 * 100 + 7, tile.getStack().getProcessor(c + 1).get(0, 0));
                assertEquals(c * 10000 + 23 * 100 + 19, tile.getStack().getProcessor(c + 1).get(12, 14));
            }
            SpillOT.Params params = new SpillOT.Params();
            params.outputDir = directory;
            params.baseName = "crop";
            var manifest = SpillOT.buildManifest(server, params, region, SpillOT.channelNames(server),
                    directory.resolve("input.raw"), directory.resolve("matrix.csv"), "<u2");
            assertEquals(List.of(2, 15, 13), manifest.get("shape"));
            assertEquals(List.of(7, 9), manifest.get("origin_xy"));
            assertEquals(List.of("CD8", "DNA"), manifest.get("channel_names"));
            assertEquals(List.of(0.325, 0.5), manifest.get("pixel_size_um"));
            assertEquals(0, params.memoryBudgetMB);
            tile.flush();
        }
    }

    @Test void exportReadsBoundedTilesAcrossANonzeroSelectedRegion() throws Exception {
        BufferedImage image = new BufferedImage(400, 400, BufferedImage.TYPE_USHORT_GRAY);
        for (int y = 0; y < 400; y++)
            for (int x = 0; x < 400; x++)
                image.getRaster().setSample(x, y, 0, y * 100 + x);
        try (var server = new TrackingServer(image)) {
            ImageRegion region = ImageRegion.createInstance(17, 23, 280, 260, 0, 0);
            ImagePlus first = SpillOT.readStack(server, ImageRegion.createInstance(17, 23, 128, 128, 0, 0), 1);
            Path raw = directory.resolve("multi-tile.raw");
            SpillOT.exportRaw(server, region, 1, 128, 16, raw, 280L * 260 * 2, first, () -> false, null);
            assertEquals(9, server.requests.size());
            long area = 0;
            for (var request : server.requests) {
                assertTrue(request.getWidth() <= 128 && request.getHeight() <= 128);
                assertTrue(request.getX() >= 17 && request.getY() >= 23);
                area += (long) request.getWidth() * request.getHeight();
            }
            assertEquals(280L * 260, area);
            ByteBuffer pixels = ByteBuffer.wrap(Files.readAllBytes(raw)).order(ByteOrder.LITTLE_ENDIAN);
            for (int y = 0; y < 260; y++)
                for (int x = 0; x < 280; x++)
                    assertEquals((y + 23) * 100 + x + 17, pixels.getShort() & 65535);
        }
    }

    @Test void runsJavaExportBundledPythonAndTiffAssembly() throws Exception {
        String env = System.getenv("SPILLOT_TEST_ENV");
        assumeTrue(env != null && Files.isRegularFile(SpillOT.pythonFromEnv(env)),
                "Set SPILLOT_TEST_ENV to an environment with numpy, scipy and tifffile for the end-to-end test");
        try (var server = sampleServer()) {
            SpillOT.Params params = new SpillOT.Params();
            params.envPath = env;
            params.outputDir = directory;
            params.baseName = "integration";
            params.region = ImageRegion.createInstance(7, 9, 28, 32, 0, 0);
            params.patchSize = 8;
            params.workers = 1;
            params.memoryBudgetMB = 512;
            params.sources.put(0, Set.of(1));
            List<SpillOT.Progress> progress = java.util.Collections.synchronizedList(new ArrayList<>());
            SpillOT.Result result = SpillOT.run(server, params, System.out::println, null, () -> false, progress::add);
            assertTrue(Files.size(result.input) > 0);
            assertTrue(Files.size(result.fullStack) > 0);
            assertEquals(Set.of(0), result.channels.keySet());
            assertEquals(1, progress.getLast().fraction);
            assertEquals("done", progress.getLast().phase);
            for (int i = 1; i < progress.size(); i++)
                assertTrue(progress.get(i).fraction >= progress.get(i - 1).fraction, "Progress regressed");
            String verify = "import sys,tifffile,numpy as np,xml.etree.ElementTree as ET,json; "
                    + "a=tifffile.imread(sys.argv[1]); b=tifffile.imread(sys.argv[2]); "
                    + "assert a.shape == b.shape == (2,32,28); assert a.dtype == b.dtype == np.dtype('uint16'); "
                    + "assert np.array_equal(a[1],b[1]); assert int(a[1,0,0]) == 10907; "
                    + "assert int(a[1,-1,-1]) == 14034; "
                    + "t=tifffile.TiffFile(sys.argv[2]); root=ET.fromstring(t.ome_metadata); "
                    + "px=root.find('.//{*}Pixels'); assert float(px.attrib['PhysicalSizeX']) == .325; "
                    + "assert float(px.attrib['PhysicalSizeY']) == .5; "
                    + "assert [c.attrib['Name'] for c in px.findall('{*}Channel')] == ['CD8','DNA']; "
                    + "desc=json.loads(root.find('.//{*}Description').text); assert desc['origin_xy'] == [7,9]";
            Process check = new ProcessBuilder(SpillOT.pythonFromEnv(env).toString(), "-c", verify,
                    result.input.toString(), result.fullStack.toString()).redirectErrorStream(true).start();
            String output = new String(check.getInputStream().readAllBytes());
            assertEquals(0, check.waitFor(), output);
            try (var files = Files.list(directory)) {
                assertTrue(files.noneMatch(path -> path.getFileName().toString().startsWith(".spillot-")),
                        "Run temporary files were not removed");
            }
        }
    }

    @Test void fiveArgumentScriptingApiUsesAutomaticMemoryPlanning() throws Exception {
        String env = System.getenv("SPILLOT_TEST_ENV");
        assumeTrue(env != null && Files.isRegularFile(SpillOT.pythonFromEnv(env)));
        try (var server = sampleServer()) {
            SpillOT.Params params = new SpillOT.Params();
            params.envPath = env;
            params.outputDir = directory;
            params.baseName = "scripting";
            params.region = ImageRegion.createInstance(7, 9, 12, 12, 0, 0);
            params.patchSize = 8;
            params.workers = 1;
            params.sources.put(0, Set.of(1));
            List<String> logs = new ArrayList<>();
            var result = SpillOT.run(server, params, logs::add, null, () -> false);
            assertTrue(logs.stream().anyMatch(line -> line.startsWith("Low-memory SpillOT plan:")),
                    "The existing five-argument API must always use the memory-aware backend");
            assertTrue(Files.size(result.fullStack) > 0);
            String verify = "import sys,tifffile,numpy as np; "
                    + "a=tifffile.imread(sys.argv[1]); b=tifffile.imread(sys.argv[2]); "
                    + "assert a.shape == b.shape == (2,12,12); "
                    + "assert a.dtype == b.dtype == np.dtype('uint16'); "
                    + "assert np.array_equal(a[1],b[1]); assert int(b[1,0,0]) == 10907; "
                    + "assert tifffile.TiffFile(sys.argv[2]).is_ome";
            Process check = new ProcessBuilder(SpillOT.pythonFromEnv(env).toString(), "-c", verify,
                    result.input.toString(), result.fullStack.toString()).redirectErrorStream(true).start();
            String output = new String(check.getInputStream().readAllBytes());
            assertEquals(0, check.waitFor(), output);
        }
    }

    @Test void pendingPublicationRestoresOldResultsAndRemovesOnlyNewResults() throws Exception {
        Path work = Files.createDirectory(directory.resolve("work"));
        Path output = Files.createDirectory(directory.resolve("output"));
        Path backup = Files.createDirectory(work.resolve(".spillot-publication-backups-test"));
        Path stage = Files.createDirectory(work.resolve("stage"));
        Files.writeString(output.resolve("old.tif"), "replacement");
        Files.writeString(backup.resolve("0"), "original");
        Files.writeString(output.resolve("new.tif"), "new file");
        Files.writeString(output.resolve("untouched.tif"), "still original");
        Files.writeString(stage.resolve("untouched.tif"), "not published");
        writeJournal(work, output, backup, "pending", List.of(
                journalEntry(output.resolve("old.tif"), backup.resolve("0"), stage.resolve("old.tif"), true),
                journalEntry(output.resolve("new.tif"), backup.resolve("1"), stage.resolve("new.tif"), false),
                journalEntry(output.resolve("untouched.tif"), backup.resolve("2"), stage.resolve("untouched.tif"), true)));
        SpillOT.recoverPublication(work, output);
        assertEquals("original", Files.readString(output.resolve("old.tif")));
        assertFalse(Files.exists(output.resolve("new.tif")));
        assertEquals("still original", Files.readString(output.resolve("untouched.tif")));
        assertFalse(Files.exists(work.resolve("publication.json")));
        // The caller removes the entire private work tree after recovery.
        assertFalse(Files.exists(backup.resolve("0")));
    }

    @Test void committedPublicationKeepsCompleteResults() throws Exception {
        Path work = Files.createDirectory(directory.resolve("committed-work"));
        Path output = Files.createDirectory(directory.resolve("committed-output"));
        Path backup = Files.createDirectory(work.resolve(".spillot-publication-backups-test"));
        Files.writeString(output.resolve("result.tif"), "complete new result");
        Files.writeString(backup.resolve("0"), "old result");
        writeJournal(work, output, backup, "committed", List.of(journalEntry(output.resolve("result.tif"),
                backup.resolve("0"), work.resolve("stage/result.tif"), true)));
        SpillOT.recoverPublication(work, output);
        assertEquals("complete new result", Files.readString(output.resolve("result.tif")));
        assertFalse(Files.exists(work.resolve("publication.json")));
    }

    @Test void failedRecoveryPreservesItsJournalForRetry() throws Exception {
        Path work = Files.createDirectory(directory.resolve("failed-work"));
        Path output = Files.createDirectory(directory.resolve("failed-output"));
        Path backup = Files.createDirectory(work.resolve(".spillot-publication-backups-test"));
        writeJournal(work, output, backup, "pending", List.of(journalEntry(output.resolve("missing.tif"),
                backup.resolve("0"), work.resolve("stage/missing.tif"), true)));
        assertThrows(java.io.IOException.class, () -> SpillOT.recoverPublication(work, output));
        assertTrue(Files.exists(work.resolve("publication.json")));
        assertTrue(Files.exists(backup));
        Files.writeString(backup.resolve("0"), "recovered original");
        SpillOT.recoverPublication(work, output);
        assertEquals("recovered original", Files.readString(output.resolve("missing.tif")));
    }

    @Test void malformedRecoveryJournalDoesNotChangeFiles() throws Exception {
        Path work = Files.createDirectory(directory.resolve("malformed-work"));
        Path output = Files.createDirectory(directory.resolve("malformed-output"));
        Path backup = Files.createDirectory(work.resolve(".spillot-publication-backups-test"));
        Files.writeString(output.resolve("result.tif"), "current");
        Files.writeString(backup.resolve("0"), "backup");
        var entry = new java.util.HashMap<String, Object>(journalEntry(output.resolve("result.tif"),
                backup.resolve("0"), work.resolve("stage/result.tif"), true));
        entry.put("existed", "true");
        writeJournal(work, output, backup, "pending", List.of(entry));
        assertThrows(java.io.IOException.class, () -> SpillOT.recoverPublication(work, output));
        assertEquals("current", Files.readString(output.resolve("result.tif")));
        assertEquals("backup", Files.readString(backup.resolve("0")));
        assertTrue(Files.exists(work.resolve("publication.json")));
    }

    private static Map<String, Object> journalEntry(Path destination, Path backup, Path staged, boolean existed) {
        return Map.of("destination", destination.toString(), "backup", backup.toString(),
                "staged", staged.toString(), "existed", existed);
    }

    private static void writeJournal(Path work, Path output, Path backup, String state,
                                     List<Map<String, Object>> entries) throws Exception {
        Files.writeString(work.resolve("publication.json"), new com.google.gson.Gson().toJson(Map.of(
                "version", 1, "state", state, "output_dir", output.toString(), "backup_dir", backup.toString(),
                "entries", entries)));
    }

    private static class TrackingServer extends WrappedBufferedImageServer {
        final List<RegionRequest> requests = new ArrayList<>();
        TrackingServer(BufferedImage image) { super("tracked image", image); }
        @Override public BufferedImage readRegion(RegionRequest request) throws java.io.IOException {
            requests.add(request);
            return super.readRegion(request);
        }
    }

    private static WrappedBufferedImageServer sampleServer() {
        int width = 40, height = 44;
        var channels = List.of(ImageChannel.getInstance("CD8", 0xFF0000), ImageChannel.getInstance("DNA", 0x0000FF));
        var raster = Raster.createWritableRaster(new BandedSampleModel(DataBuffer.TYPE_USHORT, width, height, 2),
                new DataBufferUShort(width * height, 2), null);
        for (int c = 0; c < 2; c++)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    raster.setSample(x, y, c, c * 10000 + y * 100 + x);
        BufferedImage image = new BufferedImage(ColorModelFactory.createColorModel(PixelType.UINT16, channels),
                raster, false, null);
        var server = new WrappedBufferedImageServer("test-image", image, channels);
        server.setMetadata(new ImageServerMetadata.Builder(server.getMetadata()).pixelSizeMicrons(0.325, 0.5).build());
        return server;
    }
}
