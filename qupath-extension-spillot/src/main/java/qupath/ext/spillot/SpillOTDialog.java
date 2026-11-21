// Copyright 2025, Jason Lequyer, Ferris Nowlan and Laurence Pelletier.
// All rights reserved.
// Sinai Health System - Lunenfeld Tanenbaum Research Institute
// 600 University Avenue, Room 1070, Toronto, ON M5G 1X5, Canada

package qupath.ext.spillot;

import javafx.application.Platform;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.StringProperty;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.ContentDisplay;
import javafx.scene.control.Label;
import javafx.scene.control.ListCell;
import javafx.scene.control.ListView;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Spinner;
import javafx.scene.control.SplitPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleButton;
import javafx.scene.control.Tooltip;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.TilePane;
import javafx.scene.layout.VBox;
import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;
import ij.ImageStack;
import ij.process.ImageProcessor;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.commands.ProjectCommands;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerMetadata;
import qupath.lib.images.servers.ImageServers;
import qupath.lib.images.servers.ServerTools;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.regions.ImageRegion;
import qupath.lib.roi.interfaces.ROI;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The SpillOT window: pick the channels to clean, pick which channels may be bleeding into
 * each of them, choose the settings and run.
 */
final class SpillOTDialog {

    private static final StringProperty PREF_ENV = PathPrefs.createPersistentPreference("spillot.envPath", "");
    private static final IntegerProperty PREF_PATCH =
            PathPrefs.createPersistentPreference("spillot.patchSize", SpillOT.DEFAULT_PATCH_SIZE);

    /** Above this many pixels the user is asked to confirm (time and memory grow with area). */
    private static final long LARGE_REGION_PIXELS = 500_000L;
    private static final int THUMB_PIXELS = 256;
    private static final String TILE_BASE = "-fx-background-radius: 6; -fx-border-radius: 6; -fx-border-width: 3; ";
    private static final String TILE_OFF = TILE_BASE + "-fx-border-color: transparent;";
    private static final String TILE_ON = TILE_BASE + "-fx-border-color: #1c85eb;";

    private static Stage openStage;

    private final QuPathGUI qupath;
    private final ImageData<BufferedImage> imageData;
    private final ImageServer<BufferedImage> server;
    private final List<String> names;
    private final int nChannels;
    private final int z, t;
    private final ImageRegion selectedRegion; // bounding box of the selected object, or null
    private final String imageName;

    private final Map<Integer, Set<Integer>> sources = new TreeMap<>();
    private final BooleanProperty[] clean;
    private final Image[] thumbs;
    private final AtomicInteger thumbGeneration = new AtomicInteger();
    private final AtomicBoolean cancelRequested = new AtomicBoolean(false);

    private final Stage stage = new Stage();
    private final ListView<Integer> targetList = new ListView<>();
    private final Label sourceHeader = new Label();
    private final ImageView targetPreview = new ImageView();
    private final TilePane sourceTiles = new TilePane(6, 6);
    private final ComboBox<String> regionChoice = new ComboBox<>();
    private final Label regionInfo = new Label();
    private final Spinner<Integer> patchSpinner = new Spinner<>(4, 256, SpillOT.DEFAULT_PATCH_SIZE, 2);
    private final Spinner<Integer> workerSpinner;
    private final TextField envField = new TextField();
    private final TextField outField = new TextField();
    private final CheckBox overexposed = new CheckBox("Ignore overexposed pixels (set saturated to 0, then inpaint)");
    private final ProgressBar progressBar = new ProgressBar(0);
    private final TextArea logArea = new TextArea();
    private final Button runButton = new Button("Run");
    private final Button cancelButton = new Button("Cancel run");
    private final Button closeButton = new Button("Close");
    private SplitPane center;
    private GridPane settings;

    static void show(QuPathGUI qupath) {
        if (openStage != null && openStage.isShowing()) {
            openStage.toFront();
            return;
        }
        ImageData<BufferedImage> imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage("SpillOT", "Please open an image first.");
            return;
        }
        ImageServer<BufferedImage> server = imageData.getServer();
        if (server.isRGB() || SpillOT.nPlanes(server) < 2) {
            Dialogs.showErrorMessage("SpillOT",
                    "SpillOT needs a multichannel (non-RGB) image with at least 2 channels.");
            return;
        }
        SpillOTDialog dialog = new SpillOTDialog(qupath, imageData);
        openStage = dialog.stage;
        dialog.stage.show();
    }

    private SpillOTDialog(QuPathGUI qupath, ImageData<BufferedImage> imageData) {
        this.qupath = qupath;
        this.imageData = imageData;
        this.server = imageData.getServer();
        this.names = SpillOT.channelNames(server);
        this.nChannels = SpillOT.nPlanes(server);
        this.imageName = ServerTools.getDisplayableImageName(server);
        var viewer = qupath.getViewer();
        this.z = viewer == null ? 0 : viewer.getZPosition();
        this.t = viewer == null ? 0 : viewer.getTPosition();
        this.selectedRegion = selectedObjectRegion();
        this.clean = new BooleanProperty[nChannels];
        this.thumbs = new Image[nChannels];
        for (int i = 0; i < nChannels; i++) {
            clean[i] = new SimpleBooleanProperty(false);
            clean[i].addListener((v, o, n) -> targetList.refresh());
        }
        int cpus = Math.max(1, Runtime.getRuntime().availableProcessors());
        this.workerSpinner = new Spinner<>(1, Math.max(cpus, 1), cpus, 1);
        buildUi();
        // Prefill from an earlier run on the same region, if its CSV is still there
        sources.putAll(SpillOT.readCsv(outputDir().resolve(baseName() + ".csv"), names));
        targetList.refresh();
        targetList.getSelectionModel().select(0);
        loadThumbnails();
    }

    // ------------------------------------------------------------------
    // Layout
    // ------------------------------------------------------------------

    private void buildUi() {
        // --- Region ---
        String whole = "Whole image";
        String selection = "Selected object (bounding box)";
        if (selectedRegion != null)
            regionChoice.getItems().add(selection);
        regionChoice.getItems().add(whole);
        regionChoice.getSelectionModel().select(0);
        regionChoice.setOnAction(e -> {
            updateRegionInfo();
            loadThumbnails();
        });
        regionInfo.setWrapText(true);
        HBox regionBox = new HBox(8, new Label("Region to process:"), regionChoice, regionInfo);
        regionBox.setAlignment(Pos.CENTER_LEFT);
        Label intro = new Label("1) Tick the channels you want to clean.   "
                + "2) For each one, click the channels you suspect are bleeding into it.   "
                + "Signal is removed only where local structure matches.");
        intro.setWrapText(true);
        VBox top = new VBox(6, intro, regionBox);
        top.setPadding(new Insets(10, 10, 6, 10));

        // --- Targets (left) ---
        targetList.setCellFactory(v -> new TargetCell());
        for (int i = 0; i < nChannels; i++)
            targetList.getItems().add(i);
        targetList.getSelectionModel().selectedItemProperty().addListener((v, o, n) -> rebuildSources());
        Button clearAll = new Button("Clear all");
        clearAll.setOnAction(e -> {
            sources.clear();
            for (BooleanProperty b : clean)
                b.set(false);
            rebuildSources();
            targetList.refresh();
        });
        Label leftTitle = new Label("Channels to clean");
        leftTitle.setStyle("-fx-font-weight: bold;");
        HBox leftHead = new HBox(8, leftTitle, spacer(), clearAll);
        leftHead.setAlignment(Pos.CENTER_LEFT);
        VBox left = new VBox(6, leftHead, targetList);
        VBox.setVgrow(targetList, Priority.ALWAYS);
        left.setPadding(new Insets(4, 6, 4, 10));

        // --- Sources (right) ---
        sourceHeader.setStyle("-fx-font-weight: bold;");
        sourceHeader.setWrapText(true);
        targetPreview.setFitWidth(200);
        targetPreview.setFitHeight(200);
        targetPreview.setPreserveRatio(true);
        Label previewCaption = new Label("Channel being cleaned");
        previewCaption.setStyle("-fx-text-fill: gray;");
        VBox previewBox = new VBox(4, targetPreview, previewCaption);
        previewBox.setAlignment(Pos.TOP_CENTER);
        previewBox.setMinWidth(210);
        sourceTiles.setPadding(new Insets(4));
        ScrollPane scroll = new ScrollPane(sourceTiles);
        scroll.setFitToWidth(true);
        Button clearSources = new Button("Clear");
        clearSources.setOnAction(e -> {
            Integer target = targetList.getSelectionModel().getSelectedItem();
            if (target != null) {
                sources.remove(target);
                rebuildSources();
                targetList.refresh();
            }
        });
        HBox rightHead = new HBox(8, sourceHeader, spacer(), clearSources);
        rightHead.setAlignment(Pos.CENTER_LEFT);
        HBox rightBody = new HBox(8, previewBox, scroll);
        HBox.setHgrow(scroll, Priority.ALWAYS);
        VBox right = new VBox(6, rightHead, rightBody);
        VBox.setVgrow(rightBody, Priority.ALWAYS);
        right.setPadding(new Insets(4, 10, 4, 6));

        center = new SplitPane(left, right);
        center.setDividerPositions(0.3);

        // --- Settings ---
        patchSpinner.setEditable(true);
        patchSpinner.getValueFactory().setValue(PREF_PATCH.get() >= 4 ? PREF_PATCH.get() : SpillOT.DEFAULT_PATCH_SIZE);
        patchSpinner.setTooltip(new Tooltip("Even integer >= 4. Lower = more aggressive removal and faster runs."));
        workerSpinner.setEditable(true);
        workerSpinner.setTooltip(new Tooltip("Reduce this if you run out of memory."));
        String env = PREF_ENV.get();
        if (env == null || env.isBlank() || SpillOT.pythonFromEnv(env) == null
                || !SpillOT.pythonFromEnv(env).toFile().exists())
            env = SpillOT.guessCondaEnv("spillot");
        envField.setText(env == null ? "" : env);
        envField.setPromptText("Root of the conda env, e.g. .../envs/spillot");
        outField.setText(defaultOutputDir().toString());
        Button browseEnv = new Button("Browse...");
        browseEnv.setOnAction(e -> browse(envField, "Choose the 'spillot' conda environment folder"));
        Button browseOut = new Button("Browse...");
        browseOut.setOnAction(e -> browse(outField, "Choose the output folder"));

        settings = new GridPane();
        settings.setHgap(8);
        settings.setVgap(6);
        settings.setPadding(new Insets(6, 10, 6, 10));
        settings.addRow(0, new Label("Patch size:"), patchSpinner, new Label("Worker processes:"), workerSpinner);
        settings.add(overexposed, 4, 0, 2, 1);
        settings.add(new Label("Conda env path:"), 0, 1);
        settings.add(envField, 1, 1, 4, 1);
        settings.add(browseEnv, 5, 1);
        settings.add(new Label("Output folder:"), 0, 2);
        settings.add(outField, 1, 2, 4, 1);
        settings.add(browseOut, 5, 2);
        GridPane.setHgrow(envField, Priority.ALWAYS);
        GridPane.setHgrow(outField, Priority.ALWAYS);

        // --- Progress and buttons ---
        logArea.setEditable(false);
        logArea.setPrefRowCount(5);
        logArea.setStyle("-fx-font-family: monospace; -fx-font-size: 11;");
        progressBar.setMaxWidth(Double.MAX_VALUE);
        HBox.setHgrow(progressBar, Priority.ALWAYS);
        cancelButton.setDisable(true);
        runButton.setDefaultButton(true);
        runButton.setOnAction(e -> startRun());
        cancelButton.setOnAction(e -> cancelRequested.set(true));
        closeButton.setOnAction(e -> stage.close());
        HBox buttons = new HBox(8, progressBar, runButton, cancelButton, closeButton);
        buttons.setAlignment(Pos.CENTER_RIGHT);
        buttons.setPadding(new Insets(6, 10, 10, 10));
        VBox logBox = new VBox(logArea);
        logBox.setPadding(new Insets(0, 10, 0, 10));
        VBox bottom = new VBox(settings, logBox, buttons);

        BorderPane root = new BorderPane(center, top, null, bottom, null);
        stage.setTitle("SpillOT - " + imageName);
        stage.initOwner(qupath.getStage());
        stage.setScene(new Scene(root, 1050, 760));
        stage.setOnCloseRequest(e -> cancelRequested.set(true));
        updateRegionInfo();
    }

    private static Region spacer() {
        Region r = new Region();
        HBox.setHgrow(r, Priority.ALWAYS);
        return r;
    }

    private void browse(TextField field, String title) {
        DirectoryChooser chooser = new DirectoryChooser();
        chooser.setTitle(title);
        File current = new File(field.getText().trim());
        if (current.isDirectory())
            chooser.setInitialDirectory(current);
        File chosen = chooser.showDialog(stage);
        if (chosen != null)
            field.setText(chosen.getAbsolutePath());
    }

    /** One row of the left-hand list: tick box, thumbnail, name and number of selected sources. */
    private final class TargetCell extends ListCell<Integer> {
        private final CheckBox check = new CheckBox();
        private final ImageView view = new ImageView();
        private final Label name = new Label();
        private final Label count = new Label();
        private final HBox box = new HBox(8, check, view, name, spacer(), count);
        private BooleanProperty bound;

        TargetCell() {
            view.setFitWidth(36);
            view.setFitHeight(36);
            view.setPreserveRatio(true);
            box.setAlignment(Pos.CENTER_LEFT);
            count.setStyle("-fx-text-fill: gray;");
            setContentDisplay(ContentDisplay.GRAPHIC_ONLY);
        }

        @Override
        protected void updateItem(Integer item, boolean empty) {
            super.updateItem(item, empty);
            if (bound != null) {
                check.selectedProperty().unbindBidirectional(bound);
                bound = null;
            }
            if (empty || item == null) {
                setGraphic(null);
                return;
            }
            bound = clean[item];
            check.selectedProperty().bindBidirectional(bound);
            view.setImage(thumbs[item]);
            name.setText((item + 1) + ". " + names.get(item));
            int n = sources.getOrDefault(item, Set.of()).size();
            count.setText(n == 0 ? "" : n + (n == 1 ? " source" : " sources"));
            setGraphic(box);
        }
    }

    /** Rebuild the right-hand tiles for the channel selected in the list. */
    private void rebuildSources() {
        sourceTiles.getChildren().clear();
        Integer target = targetList.getSelectionModel().getSelectedItem();
        if (target == null) {
            sourceHeader.setText("Select a channel on the left");
            targetPreview.setImage(null);
            return;
        }
        sourceHeader.setText("Which channels are bleeding into " + (target + 1) + ". " + names.get(target) + "?");
        targetPreview.setImage(thumbs[target]);
        Set<Integer> selected = sources.getOrDefault(target, Set.of());
        for (int i = 0; i < nChannels; i++) {
            if (i == target)
                continue;
            final int source = i;
            ImageView view = new ImageView(thumbs[i]);
            view.setFitWidth(110);
            view.setFitHeight(110);
            view.setPreserveRatio(true);
            ToggleButton tile = new ToggleButton((i + 1) + ". " + names.get(i), view);
            tile.setContentDisplay(ContentDisplay.TOP);
            tile.setPrefWidth(130);
            tile.setTooltip(new Tooltip("Mark " + names.get(i) + " as possibly bleeding into " + names.get(target)));
            tile.setSelected(selected.contains(i));
            tile.setStyle(tile.isSelected() ? TILE_ON : TILE_OFF);
            tile.selectedProperty().addListener((v, o, on) -> {
                tile.setStyle(on ? TILE_ON : TILE_OFF);
                if (on) {
                    sources.computeIfAbsent(target, k -> new TreeSet<>()).add(source);
                    clean[target].set(true);
                } else if (sources.containsKey(target)) {
                    sources.get(target).remove(source);
                    if (sources.get(target).isEmpty())
                        sources.remove(target);
                }
                targetList.refresh();
            });
            sourceTiles.getChildren().add(tile);
        }
    }

    // ------------------------------------------------------------------
    // Region, names and thumbnails
    // ------------------------------------------------------------------

    private ImageRegion selectedObjectRegion() {
        var viewer = qupath.getViewer();
        PathObject selected = viewer == null ? null : viewer.getSelectedObject();
        ROI roi = selected == null ? null : selected.getROI();
        if (roi == null || roi.isEmpty())
            return null;
        int x0 = (int) Math.max(0, Math.floor(roi.getBoundsX()));
        int y0 = (int) Math.max(0, Math.floor(roi.getBoundsY()));
        int x1 = (int) Math.min(server.getWidth(), Math.ceil(roi.getBoundsX() + roi.getBoundsWidth()));
        int y1 = (int) Math.min(server.getHeight(), Math.ceil(roi.getBoundsY() + roi.getBoundsHeight()));
        if (x1 - x0 < 2 || y1 - y0 < 2)
            return null;
        return ImageRegion.createInstance(x0, y0, x1 - x0, y1 - y0, z, t);
    }

    private boolean useSelection() {
        return selectedRegion != null && regionChoice.getSelectionModel().getSelectedIndex() == 0;
    }

    private ImageRegion region() {
        return useSelection() ? selectedRegion
                : ImageRegion.createInstance(0, 0, server.getWidth(), server.getHeight(), z, t);
    }

    private String baseName() {
        // "slide.ome.tif - Image0" -> "slide"; other series names are kept ("slide_Region2")
        String base = imageName.replaceFirst("\\s+-\\s+(Image|Series) ?0$", "");
        base = base.replaceFirst("(?i)(\\.ome)?\\.[a-z0-9]{2,6}(?=$|\\s+-\\s+)", "");
        base = SpillOT.sanitize(base.replaceFirst("\\s+-\\s+", "_"));
        if (useSelection())
            base += "_x" + selectedRegion.getX() + "_y" + selectedRegion.getY()
                    + "_w" + selectedRegion.getWidth() + "_h" + selectedRegion.getHeight();
        SpillOT.Axis axis = SpillOT.channelAxis(server);
        if (server.nZSlices() > 1 && axis != SpillOT.Axis.Z)
            base += "_z" + (z + 1);
        if (server.nTimepoints() > 1 && axis != SpillOT.Axis.T)
            base += "_t" + (t + 1);
        return base;
    }

    private Path defaultOutputDir() {
        try {
            Project<BufferedImage> project = qupath.getProject();
            if (project != null && project.getPath() != null) {
                Path p = project.getPath();
                return (p.toFile().isDirectory() ? p : p.getParent()).resolve("SpillOT");
            }
            for (URI uri : server.getURIs())
                if ("file".equals(uri.getScheme()))
                    return Paths.get(uri).getParent().resolve("SpillOT");
        } catch (Exception ignored) {
            // fall through
        }
        return Paths.get(System.getProperty("user.home"), "SpillOT");
    }

    private Path outputDir() {
        String s = outField.getText() == null ? "" : outField.getText().trim();
        return s.isEmpty() ? defaultOutputDir() : Paths.get(s);
    }

    private void updateRegionInfo() {
        ImageRegion r = region();
        long pixels = (long) r.getWidth() * r.getHeight();
        String txt = r.getWidth() + " x " + r.getHeight() + " px";
        if (selectedRegion == null)
            txt += "   (select an annotation before opening SpillOT to process just that area)";
        if (pixels > LARGE_REGION_PIXELS) {
            txt += "   Large region: expect a long run and high memory use.";
            regionInfo.setStyle("-fx-text-fill: #c0392b;");
        } else {
            regionInfo.setStyle("");
        }
        regionInfo.setText(txt);
    }

    /** Read a small version of the region and turn every channel into an autoscaled grey thumbnail. */
    private void loadThumbnails() {
        final int generation = thumbGeneration.incrementAndGet();
        final ImageRegion r = region();
        Thread thread = new Thread(() -> {
            try {
                double downsample = Math.max(1.0, Math.max(r.getWidth(), r.getHeight()) / (double) THUMB_PIXELS);
                ImageStack stack = SpillOT.readStack(server, r, downsample).getStack();
                int w = stack.getWidth(), h = stack.getHeight();
                Image[] images = new Image[nChannels];
                float[] buf = new float[w * h];
                for (int c = 0; c < nChannels; c++) {
                    ImageProcessor ip = stack.getProcessor(c + 1);
                    for (int i = 0; i < buf.length; i++)
                        buf[i] = ip.getf(i);
                    float[] sorted = buf.clone();
                    Arrays.sort(sorted);
                    float lo = sorted[0];
                    float hi = sorted[(int) Math.min(sorted.length - 1, Math.round(sorted.length * 0.998))];
                    if (!(hi > lo))
                        hi = sorted[sorted.length - 1];
                    float range = hi > lo ? hi - lo : 1f;
                    WritableImage fx = new WritableImage(w, h);
                    PixelWriter pw = fx.getPixelWriter();
                    for (int y = 0; y < h; y++) {
                        for (int x = 0; x < w; x++) {
                            int g = Math.round(255f * Math.max(0f, Math.min(1f, (buf[y * w + x] - lo) / range)));
                            pw.setArgb(x, y, 0xFF000000 | (g << 16) | (g << 8) | g);
                        }
                    }
                    images[c] = fx;
                }
                Platform.runLater(() -> {
                    if (generation != thumbGeneration.get())
                        return;
                    System.arraycopy(images, 0, thumbs, 0, nChannels);
                    targetList.refresh();
                    rebuildSources();
                });
            } catch (Exception e) {
                Platform.runLater(() -> log("Could not create channel previews: " + e.getMessage()));
            }
        }, "spillot-thumbnails");
        thread.setDaemon(true);
        thread.start();
    }

    // ------------------------------------------------------------------
    // Run
    // ------------------------------------------------------------------

    private void log(String line) {
        logArea.appendText(line + "\n");
    }

    private void setRunning(boolean running) {
        center.setDisable(running);
        settings.setDisable(running);
        regionChoice.setDisable(running);
        runButton.setDisable(running);
        closeButton.setDisable(running);
        cancelButton.setDisable(!running);
    }

    private void startRun() {
        // Commit typed spinner values
        Integer patch = commit(patchSpinner);
        Integer workers = commit(workerSpinner);
        if (patch == null || patch < 4 || patch % 2 != 0) {
            Dialogs.showErrorMessage("SpillOT", "Patch size must be an EVEN integer >= 4.");
            return;
        }
        if (workers == null || workers < 1) {
            Dialogs.showErrorMessage("SpillOT", "Number of worker processes must be an integer >= 1.");
            return;
        }

        SpillOT.Params p = new SpillOT.Params();
        StringBuilder skipped = new StringBuilder();
        for (int i = 0; i < nChannels; i++) {
            if (!clean[i].get())
                continue;
            Set<Integer> srcs = sources.get(i);
            if (srcs == null || srcs.isEmpty())
                skipped.append(skipped.length() == 0 ? "" : ", ").append(names.get(i));
            else
                p.sources.put(i, new TreeSet<>(srcs));
        }
        if (p.sources.isEmpty()) {
            Dialogs.showErrorMessage("SpillOT", "Tick at least one channel to clean and select at least one "
                    + "channel that may be bleeding into it.");
            return;
        }

        String env = envField.getText() == null ? "" : envField.getText().trim();
        Path python = SpillOT.pythonFromEnv(env);
        if (python == null || !python.toFile().exists()) {
            Dialogs.showErrorMessage("SpillOT", "Could not find Python in the conda environment.\n\n"
                    + "Create the environment once with:\n"
                    + "conda create -n spillot -c conda-forge python=3.11 numpy scipy tifffile imagecodecs\n\n"
                    + "then paste the path that 'conda env list' shows for 'spillot'.");
            return;
        }

        ImageRegion r = region();
        long pixels = (long) r.getWidth() * r.getHeight();
        if (pixels > LARGE_REGION_PIXELS && !Dialogs.showConfirmDialog("SpillOT",
                "The region is " + r.getWidth() + " x " + r.getHeight() + " px.\n\n"
                + "SpillOT compares every overlapping patch, so run time and memory grow with the area "
                + "(as a guide, 200 x 200 px takes about a minute per channel at patch size 24).\n"
                + "For whole-slide images, draw an annotation around the area of interest first, "
                + "and lower the number of worker processes if you run out of memory.\n\n"
                + "Continue anyway?"))
            return;

        p.patchSize = patch;
        p.workers = workers;
        p.envPath = env;
        p.ignoreOverexposed = overexposed.isSelected();
        p.outputDir = outputDir();
        p.baseName = baseName();
        p.region = r;
        PREF_ENV.set(env);
        PREF_PATCH.set(patch);

        logArea.clear();
        if (skipped.length() > 0)
            log("Skipping (no source channels selected): " + skipped);
        cancelRequested.set(false);
        final int total = p.sources.size();
        progressBar.setProgress(total == 1 ? ProgressBar.INDETERMINATE_PROGRESS : 0);
        setRunning(true);

        Thread thread = new Thread(() -> {
            try {
                SpillOT.Result result = SpillOT.run(server, p,
                        line -> Platform.runLater(() -> log(line)),
                        done -> Platform.runLater(() -> progressBar.setProgress(done / (double) total)),
                        cancelRequested::get);
                Platform.runLater(() -> {
                    progressBar.setProgress(1);
                    setRunning(false);
                    try {
                        showResult(result, p.baseName);
                    } catch (Exception e) {
                        Dialogs.showErrorMessage("SpillOT", "SpillOT finished and saved\n" + result.fullStack
                                + "\nbut the result could not be opened in QuPath:\n" + e.getMessage());
                    }
                });
            } catch (InterruptedException e) {
                Platform.runLater(() -> {
                    progressBar.setProgress(0);
                    setRunning(false);
                    log("Cancelled.");
                });
            } catch (Throwable e) {
                Platform.runLater(() -> {
                    progressBar.setProgress(0);
                    setRunning(false);
                    log("ERROR: " + e.getMessage());
                    Dialogs.showErrorMessage("SpillOT", String.valueOf(e.getMessage()));
                });
            }
        }, "spillot-run");
        thread.setDaemon(true);
        thread.start();
    }

    private static Integer commit(Spinner<Integer> spinner) {
        try {
            int v = Integer.parseInt(spinner.getEditor().getText().trim());
            spinner.getValueFactory().setValue(v);
            return v;
        } catch (Exception e) {
            return null;
        }
    }

    /** Add the cleaned stack to the project (or offer to open it) with the original channel names and colours. */
    private void showResult(SpillOT.Result result, String baseName) throws IOException {
        String name = baseName + "_SpillOT";
        Project<BufferedImage> project = qupath.getProject();
        if (project != null) {
            ImageServer<BufferedImage> newServer = ImageServers.buildServer(result.fullStack.toUri());
            ProjectImageEntry<BufferedImage> entry =
                    ProjectCommands.addSingleImageToProject(project, newServer, imageData.getImageType());
            if (entry == null)
                throw new IOException("The image could not be added to the project.");
            entry.setImageName(name);
            ImageData<BufferedImage> newData = entry.readImageData();
            copyChannels(newData);
            entry.saveImageData(newData);
            project.syncChanges();
            qupath.refreshProject();
            if (Dialogs.showYesNoDialog("SpillOT", "Finished.\n\nThe cleaned image was added to the project as\n"
                    + name + "\n\nFiles are in\n" + result.fullStack.getParent() + "\n\nOpen it now?")
                    && qupath.openImageEntry(entry))
                stage.close(); // this window belongs to the image that was just replaced in the viewer
        } else if (Dialogs.showYesNoDialog("SpillOT", "Finished.\n\nSaved\n" + result.fullStack
                + "\n\nOpen it in the current viewer?")) {
            if (qupath.openImage(qupath.getViewer(), result.fullStack.toString(), false, false)) {
                ImageData<BufferedImage> newData = qupath.getViewer().getImageData();
                if (newData != null)
                    copyChannels(newData);
                stage.close();
            }
        }
    }

    private void copyChannels(ImageData<BufferedImage> target) {
        ImageServer<BufferedImage> s = target.getServer();
        var channels = server.getMetadata().getChannels();
        if (SpillOT.channelAxis(server) == SpillOT.Axis.CHANNEL && s.nChannels() == channels.size())
            target.updateServerMetadata(new ImageServerMetadata.Builder(s.getMetadata()).channels(channels).build());
    }
}
