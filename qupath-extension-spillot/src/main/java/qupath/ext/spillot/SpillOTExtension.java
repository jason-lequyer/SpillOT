// Copyright 2025, Jason Lequyer, Ferris Nowlan and Laurence Pelletier.
// All rights reserved.
// Sinai Health System - Lunenfeld Tanenbaum Research Institute
// 600 University Avenue, Room 1070, Toronto, ON M5G 1X5, Canada

package qupath.ext.spillot;

import javafx.scene.control.MenuItem;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;

/**
 * Adds <b>Extensions &rarr; SpillOT &rarr; Run SpillOT...</b> to QuPath.
 */
public class SpillOTExtension implements QuPathExtension {

    private boolean installed = false;

    @Override
    public void installExtension(QuPathGUI qupath) {
        if (installed)
            return;
        installed = true;
        MenuItem item = new MenuItem("Run SpillOT...");
        item.disableProperty().bind(qupath.imageDataProperty().isNull());
        item.setOnAction(e -> {
            try {
                SpillOTDialog.show(qupath);
            } catch (Exception ex) {
                Dialogs.showErrorMessage("SpillOT", ex);
            }
        });
        qupath.getMenu("Extensions>SpillOT", true).getItems().add(item);
    }

    @Override
    public String getName() {
        return "SpillOT";
    }

    @Override
    public String getDescription() {
        return "User-guided spillover removal for highly multiplexed images. "
                + "Signal from a suspected source channel is removed from a target channel "
                + "only where local image structure matches.";
    }

    @Override
    public Version getQuPathVersion() {
        return Version.parse("v0.7.0");
    }
}
