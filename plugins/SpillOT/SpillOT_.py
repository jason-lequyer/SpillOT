# Copyright 2025, Jason Lequyer, Ferris Nowlan and Laurence Pelletier.
# All rights reserved.
# Sinai Health System - Lunenfeld Tanenbaum Research Institute
# 600 University Avenue, Room 1070, Toronto, ON M5G 1X5, Canada

try:

    # ---------------------------------------------------------------------
    # Imports
    # ---------------------------------------------------------------------
    import os, sys, tempfile, subprocess, re, time, math, codecs
    from ij import IJ, ImagePlus, ImageStack
    from ij.gui import GenericDialog
    from ij.io import Opener, FileSaver
    from ij.plugin import ChannelSplitter

    from javax.swing import (JPanel, JScrollPane,
                             JButton, BoxLayout, JOptionPane, JLabel, JDialog, JProgressBar, Timer,
                             JCheckBox, ImageIcon, BorderFactory, SwingConstants, SwingUtilities)
    from javax.swing.border import AbstractBorder
    from java.awt import (Dimension, Toolkit, BorderLayout, GraphicsEnvironment,
                          GridLayout, FlowLayout, Insets, Font, Color, BasicStroke,
                          RenderingHints, Cursor)
    from java.awt.event import ActionListener, ComponentAdapter, MouseAdapter
    from java.lang import System as JSystem, Runnable

    paths = []

    # ---------------------------------------------------------------------
    # Progress dialog helpers
    # ---------------------------------------------------------------------
    def _fmt_elapsed(sec):
        sec = int(max(0, sec))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return ("%02d:%02d:%02d" % (h, m, s)) if h else ("%02d:%02d" % (m, s))

    def _show_progress(total_channels):
        start_ts = time.time()
        dlg = JDialog(None, "Processing", False)  # modeless
        dlg.setLayout(BorderLayout())
        lbl_txt = "Processing channel" if total_channels == 1 else "Processing %d channels" % total_channels
        dlg.add(JLabel(lbl_txt), BorderLayout.NORTH)
        bar = JProgressBar()
        bar.setStringPainted(False)
        if total_channels == 1:
            bar.setIndeterminate(True)
        else:
            bar.setIndeterminate(False)
            bar.setMinimum(0)
            bar.setMaximum(total_channels)
            bar.setValue(0)
        dlg.add(bar, BorderLayout.CENTER)
        elapsed_lbl = JLabel("Elapsed: 00:00")
        dlg.add(elapsed_lbl, BorderLayout.SOUTH)
        dlg.setSize(300, 100)
        dlg.setLocationRelativeTo(None)
        dlg.setVisible(True)

        class _Tick(ActionListener):
            def actionPerformed(self, evt):
                elapsed_lbl.setText("Elapsed: " + _fmt_elapsed(time.time() - start_ts))
                elapsed_lbl.repaint()

        t = Timer(1000, _Tick())
        t.setRepeats(True)
        t.start()
        return dlg, bar, t

    def _pb_smooth_to(bar, new_value, duration_ms=250):
        if bar.isIndeterminate():
            return
        new_value = max(bar.getMinimum(), min(bar.getMaximum(), int(new_value)))
        start = bar.getValue()
        if new_value == start:
            return
        prev = bar.getClientProperty("pbTweenTimer")
        if prev is not None:
            try:
                prev.stop()
            except:
                pass
        steps = max(1, int(duration_ms / 40))
        delta = float(new_value - start) / steps
        state = {"i": 0, "val": float(start)}

        class _Step(ActionListener):
            def actionPerformed(self, evt):
                state["i"] += 1
                if state["i"] >= steps:
                    bar.setValue(new_value)
                    bar.repaint()
                    try:
                        evt.getSource().stop()
                    except:
                        pass
                    bar.putClientProperty("pbTweenTimer", None)
                    return
                state["val"] += delta
                bar.setValue(int(round(state["val"])))
                bar.repaint()

        timer = Timer(int(round(float(duration_ms) / steps)), _Step())
        timer.setRepeats(True)
        timer.start()
        bar.putClientProperty("pbTweenTimer", timer)

    def _pb_cleanup(bar, timer):
        try:
            timer.stop()
        except:
            pass
        try:
            t = bar.getClientProperty("pbTweenTimer")
            if t is not None:
                t.stop()
                bar.putClientProperty("pbTweenTimer", None)
        except:
            pass

    # ---------------------------------------------------------------------
    # Conda env helpers (with macOS /opt/anaconda3/envs/rfot added)
    # ---------------------------------------------------------------------
    def _is_windows():
        return "windows" in (JSystem.getProperty("os.name") or "").lower()

    def _python_from_env(env_root):
        return os.path.join(env_root, "python.exe") if _is_windows() else os.path.join(env_root, "bin", "python")

    def _subproc_env_for_conda_env(env_root):
        env = os.environ.copy()
        if _is_windows():
            pths = [
                os.path.join(env_root, "Library", "bin"),
                os.path.join(env_root, "Scripts"),
                env_root
            ]
            env["PATH"] = os.pathsep.join(pths + [env.get("PATH", "")])
        else:
            env["PATH"] = os.pathsep.join([os.path.join(env_root, "bin"), env.get("PATH", "")])
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        return env

    def _guess_conda_env_root(env_name="rfot"):
        try:
            home = os.path.expanduser("~")
            cands = []
            if _is_windows():
                user = os.environ.get("USERPROFILE", home)
                cands += [os.path.join(user, d, "envs", env_name)
                          for d in ("anaconda3","miniconda3","miniforge3","mambaforge","Anaconda3","Miniconda3")]
                cands += [os.path.join("C:\\ProgramData", "Anaconda3", "envs", env_name)]
                try:
                    p = subprocess.Popen(["cmd.exe", "/C", "conda", "info", "--base"],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, _ = p.communicate(timeout=1.0)
                    base = out.decode("utf-8", "ignore").strip()
                    if base:
                        cands.insert(0, os.path.join(base, "envs", env_name))
                except Exception:
                    pass
            else:
                cands = ["/opt/anaconda3/envs/%s" % env_name]
                cands += [os.path.join(home, d, "envs", env_name)
                          for d in ("mambaforge","miniforge3","miniconda3","anaconda3")]
                try:
                    p = subprocess.Popen(["bash", "-lc", "conda info --base"],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, _ = p.communicate(timeout=1.0)
                    base = out.decode("utf-8", "ignore").strip()
                    if base:
                        cands.insert(0, os.path.join(base, "envs", env_name))
                except Exception:
                    pass
            pref = os.environ.get("CONDA_PREFIX")
            if pref and os.path.basename(pref).lower() == env_name.lower() and os.path.isdir(pref):
                cands.insert(0, pref)
            for c in cands:
                if not c:
                    continue
                py = _python_from_env(c)
                if os.path.isdir(c) and os.path.exists(py):
                    return c
        except Exception:
            pass
        return ""

    # ---------------------------------------------------------------------
    # Constants and helpers
    # ---------------------------------------------------------------------
    def _u(x):
        if x is None:
            return u""
        if isinstance(x, unicode):
            return x
        if isinstance(x, str):
            try:
                return unicode(x, "utf-8")
            except Exception:
                try:
                    return unicode(x, "latin-1")
                except Exception:
                    return unicode(x, "utf-8", "replace")
        try:
            return unicode(x)
        except Exception:
            try:
                s = str(x)
            except Exception:
                return u""
            if isinstance(s, unicode):
                return s
            try:
                return unicode(s, "utf-8", "replace")
            except Exception:
                return u""

    def abort(msg):
        try:
            IJ.error(_u(msg))
        except Exception:
            IJ.error(str(msg))
        sys.exit()

    # ---------------------------------------------------------------------
    # Active image
    # ---------------------------------------------------------------------
    try:
        imp = IJ.getImage()
    except Exception:
        abort("No image is open.")

    # If there are unsaved changes, prompt the user to save before running
    if getattr(imp, "changes", False):
        res = JOptionPane.showConfirmDialog(
            None,
            "The current image has unsaved changes.\n\n"
            "Processing will save the image before running.\n"
            "Click OK to save and continue, or Cancel to abort.",
            "Unsaved changes",
            JOptionPane.OK_CANCEL_OPTION,
            JOptionPane.WARNING_MESSAGE
        )
        if res != JOptionPane.OK_OPTION:
            sys.exit()
        try:
            fs = FileSaver(imp)
            if not fs.save():
                abort("Image must be saved before running.")
        except Exception as e:
            abort(u"Could not save the image before running.\n\n%s" % _u(e))

    # Keep original file path for naming outputs (even if we later temp-save for processing)
    orig_path_for_naming = u""
    try:
        _fi0 = imp.getFileInfo()
    except Exception:
        _fi0 = None
    if (_fi0 is None) or (not _fi0.directory) or (not _fi0.fileName):
        try:
            _fi0 = imp.getOriginalFileInfo()
        except Exception:
            _fi0 = None
    if _fi0 and _fi0.directory and _fi0.fileName:
        orig_path_for_naming = os.path.join(_fi0.directory, _fi0.fileName)

    force_temp_save = False

    # Split RGB composite with single channel to R/G/B stacks
    is_rgb_color = (imp.getType() == ImagePlus.COLOR_RGB and imp.getNChannels() == 1)
    if is_rgb_color:
        splits = ChannelSplitter.split(imp)  # [R,G,B]
        w, h = splits[0].getWidth(), splits[0].getHeight()
        Z = max(splits[0].getNSlices(), 1)
        T = max(splits[0].getNFrames(), 1)
        labels = ["Red", "Green", "Blue"]
        stack = ImageStack(w, h)
        for t in range(1, T + 1):
            for z in range(1, Z + 1):
                for ci, lab in enumerate(labels, start=1):
                    src = splits[ci - 1]
                    idx = src.getStackIndex(1, z, t)
                    stack.addSlice(lab, src.getStack().getProcessor(idx))
        imp = ImagePlus(imp.getTitle() + " (RGB split)", stack)
        imp.setDimensions(3, Z, T)
        imp.setOpenAsHyperStack(True)
        force_temp_save = True

    # Ensure TIFF path (prefer current file location)
    fi = None
    try:
        fi = imp.getFileInfo()
    except Exception:
        fi = None
    if (fi is None) or (not fi.directory) or (not fi.fileName):
        try:
            fi = imp.getOriginalFileInfo()
        except Exception:
            fi = None

    if (not force_temp_save) and fi and fi.directory and fi.fileName:
        img_path = os.path.join(fi.directory, fi.fileName)
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="SpillOT_", suffix=".tif", delete=False)
        img_path = tmp.name
        IJ.saveAsTiff(imp, img_path)

    # Use the original file name for outputs if available, otherwise fall back to processing path
    out_base_path = orig_path_for_naming or img_path

    # Channel axis detection
    axis_counts = {
        "channels": imp.getNChannels(),
        "slices":   imp.getNSlices(),
        "frames":   imp.getNFrames()
    }
    axis_used, n_ch = max(axis_counts.items(), key=lambda kv: kv[1])

    def slice_idx(c):
        if axis_used == "channels":
            return imp.getStackIndex(c, 1, 1)
        elif axis_used == "slices":
            return imp.getStackIndex(1, c, 1)
        else:
            return imp.getStackIndex(1, 1, c)

    def _axis_label(axis_used):
        # UI naming: always call the plane axis "Channel" (even if stored as slices/frames)
        return "Channel"

    # ---------------------------------------------------------------------
    # Metadata helpers
    # ---------------------------------------------------------------------
    def _collect_all_metadata_text(imp, n_ch):
        texts = []

        try:
            info_prop = imp.getProperty("Info")
        except Exception:
            info_prop = None
        if info_prop:
            texts.append(_u(info_prop))

        try:
            ofi = imp.getOriginalFileInfo()
        except Exception:
            ofi = None
        if ofi is not None:
            for attr in ("info", "description"):
                try:
                    val = getattr(ofi, attr)
                except Exception:
                    val = None
                if val:
                    texts.append(_u(val))

        try:
            cfi = imp.getFileInfo()
        except Exception:
            cfi = None
        if cfi is not None:
            for attr in ("info", "description"):
                try:
                    val = getattr(cfi, attr)
                except Exception:
                    val = None
                if val:
                    texts.append(_u(val))

        stack = imp.getStack()
        for i in range(1, n_ch + 1):
            try:
                idx = slice_idx(i)
                lbl = stack.getSliceLabel(idx)
            except Exception:
                lbl = None
            if lbl:
                texts.append(_u(lbl))

        return u"\n".join(texts) if texts else u""

    def _metadata_channel_names(imp, axis_used, n_ch):
        texts = _collect_all_metadata_text(imp, n_ch)
        if not texts:
            return None
        pat = re.compile(ur"^\s*Name\s*#(\d+)\s*=\s*(.+)$")
        out = [None] * n_ch
        for line in texts.splitlines():
            m = pat.match(line)
            if not m:
                continue
            try:
                idx = int(m.group(1))
            except Exception:
                continue
            if 1 <= idx <= n_ch and out[idx - 1] is None:
                out[idx - 1] = m.group(2).strip()
        return out if any(out) else None

    # Baseline from slice labels (first line), used as fallback
    raw_labels = [(_u(imp.getStack().getSliceLabel(slice_idx(i)) or (u"Ch%d" % i)).split(u"\n")[0].strip())
                  for i in range(1, n_ch + 1)]

    def common_pref(lst):
        if not lst:
            return u""
        s1, s2 = min(lst), max(lst)
        i = 0
        while i < len(s1) and s1[i] == s2[i]:
            i += 1
        return s1[:i]

    def common_suff(lst):
        return common_pref([s[::-1] for s in lst])[::-1]

    pre, suf = common_pref(raw_labels), common_suff(raw_labels)
    fallback_names = [(s[len(pre):len(s) - len(suf)] or s) if suf else (s[len(pre):] or s)
                      for s in raw_labels]

    # Try metadata-based channel names first, then fall back
    meta_names = _metadata_channel_names(imp, axis_used, n_ch)
    if meta_names and any(meta_names):
        names = [meta_names[i] or fallback_names[i] for i in range(n_ch)]
    else:
        names = fallback_names

    # ---------------------------------------------------------------------
    # Token matcher for forgiving CSV/header name matching
    # ---------------------------------------------------------------------
    def _tok_match(ref, cand):
        ref = _u(ref)
        cand = _u(cand)
        pat = ur'(?i)(?:^|[^0-9A-Z])' + re.escape(ref) + ur'(?:[^0-9A-Z]|$)'
        return re.search(pat, cand) is not None

    # ---------------------------------------------------------------------
    # Thumbnail helper (RECTANGULAR): fill the available (w,h) above the checkbox
    # ---------------------------------------------------------------------
    def _make_thumb_icon_for_plane(imp, one_based_plane_idx, thumb_w=80, thumb_h=80):
        try:
            idx = slice_idx(one_based_plane_idx)
            ip = imp.getStack().getProcessor(idx)
            if ip is None:
                return None

            ip2 = ip.duplicate()

            # Resize to the exact rectangle (fills space; no forced square)
            tw = max(1, int(thumb_w))
            th = max(1, int(thumb_h))
            ip_small = ip2.resize(tw, th)

            try:
                ip_small.resetMinAndMax()  # autoscale per plane
            except Exception:
                pass

            try:
                img = ip_small.createImage()
            except Exception:
                try:
                    img = ip_small.getBufferedImage()
                except Exception:
                    return None

            return ImageIcon(img)
        except Exception:
            return None


    # ---------------------------------------------------------------------
    # Dialog sizing and selected-cell styling
    # ---------------------------------------------------------------------
    def _usable_screen_bounds():
        ge = GraphicsEnvironment.getLocalGraphicsEnvironment()
        gc = ge.getDefaultScreenDevice().getDefaultConfiguration()
        scr = Toolkit.getDefaultToolkit().getScreenSize()
        ins = Toolkit.getDefaultToolkit().getScreenInsets(gc)
        usable_w = scr.width  - ins.left - ins.right
        usable_h = scr.height - ins.top  - ins.bottom
        return ins.left, ins.top, usable_w, usable_h

    def _set_dialog_to_screen_fraction(dlg, fraction=0.70, min_w=600, min_h=450):
        x, y, usable_w, usable_h = _usable_screen_bounds()
        target_w = int(round(float(usable_w) * float(fraction)))
        target_h = int(round(float(usable_h) * float(fraction)))
        w = max(300, min(int(usable_w), max(int(min_w), target_w)))
        h = max(300, min(int(usable_h), max(int(min_h), target_h)))
        dlg.setSize(w, h)
        dlg.setLocation(x + int((usable_w - w) / 2), y + int((usable_h - h) / 2))

    _SELECTED_CELL_COLOR = Color(28, 133, 235)
    _SELECTED_CELL_OUTER_COLOR = Color(12, 82, 160)
    _SELECTED_CELL_INNER_COLOR = Color(255, 255, 255, 190)
    _UNSELECTED_CELL_COLOR = Color(220, 224, 229)
    _SELECTED_CELL_BORDER_PX = 9
    _UNSELECTED_CELL_BORDER_PX = 1
    _SELECTED_CELL_ARC = 18
    _UNSELECTED_CELL_ARC = 10
    # Move the selected ring bottom edge farther down so it does not cut
    # through the checkbox/text area under the preview. This is paint-only;
    # it does not change layout or shrink preview images.
    _SELECTED_CELL_BOTTOM_DROP_PX = 5

    def _paint_selection_ring(component, g, selected):
        """
        Draw the selection ring after child components have already painted.

        A Swing Border is painted before child components, so a large border can be
        hidden by the image JLabel.  This overlay painter is called from the tile
        panel's paint() method *after* JPanel.paint(), which makes the ring appear
        on top of the preview without consuming layout space or shrinking the image.
        """
        try:
            w = int(component.getWidth())
            h = int(component.getHeight())
            if w <= 2 or h <= 2:
                return

            g2 = g.create()
            try:
                try:
                    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                                        RenderingHints.VALUE_ANTIALIAS_ON)
                    g2.setRenderingHint(RenderingHints.KEY_RENDERING,
                                        RenderingHints.VALUE_RENDER_QUALITY)
                except Exception:
                    pass

                if selected:
                    thick = int(_SELECTED_CELL_BORDER_PX)
                    arc = int(_SELECTED_CELL_ARC)
                    half = int(max(1, round(float(thick) / 2.0)))
                    bottom_inset = int(max(1, half - int(_SELECTED_CELL_BOTTOM_DROP_PX)))

                    # Main blue ring, drawn over the preview. The bottom inset is
                    # intentionally smaller than the top/side inset so the lower
                    # edge sits farther down and avoids the channel text/checkbox.
                    try:
                        g2.setStroke(BasicStroke(float(thick),
                                                 BasicStroke.CAP_ROUND,
                                                 BasicStroke.JOIN_ROUND))
                    except Exception:
                        pass
                    g2.setColor(_SELECTED_CELL_COLOR)
                    g2.drawRoundRect(half, half,
                                      max(1, w - 1 - 2 * half),
                                      max(1, h - 1 - half - bottom_inset),
                                      arc, arc)

                    # Crisp darker outside edge.
                    try:
                        g2.setStroke(BasicStroke(2.0,
                                                 BasicStroke.CAP_ROUND,
                                                 BasicStroke.JOIN_ROUND))
                    except Exception:
                        pass
                    g2.setColor(_SELECTED_CELL_OUTER_COLOR)
                    g2.drawRoundRect(1, 1, max(1, w - 3), max(1, h - 3), arc, arc)

                    # Thin inner highlight so the thick ring reads cleanly on dark previews.
                    inner = int(max(thick + 1, 10))
                    inner_bottom_inset = int(max(2, inner - int(_SELECTED_CELL_BOTTOM_DROP_PX)))
                    if w - 1 - 2 * inner > 4 and h - 1 - inner - inner_bottom_inset > 4:
                        try:
                            g2.setStroke(BasicStroke(1.4,
                                                     BasicStroke.CAP_ROUND,
                                                     BasicStroke.JOIN_ROUND))
                        except Exception:
                            pass
                        g2.setColor(_SELECTED_CELL_INNER_COLOR)
                        g2.drawRoundRect(inner, inner,
                                          max(1, w - 1 - 2 * inner),
                                          max(1, h - 1 - inner - inner_bottom_inset),
                                          max(6, arc - 6), max(6, arc - 6))
                else:
                    # Subtle stable tile outline.
                    try:
                        g2.setStroke(BasicStroke(float(_UNSELECTED_CELL_BORDER_PX)))
                    except Exception:
                        pass
                    g2.setColor(_UNSELECTED_CELL_COLOR)
                    g2.drawRoundRect(0, 0, max(1, w - 1), max(1, h - 1),
                                      _UNSELECTED_CELL_ARC, _UNSELECTED_CELL_ARC)
            finally:
                try:
                    g2.dispose()
                except Exception:
                    pass
        except Exception:
            pass

    class _SelectionTilePanel(JPanel):
        """
        Tile panel that paints the selected outline as an overlay.

        This fixes the issue where a normal Swing border is hidden underneath the
        preview image, because JPanel.paint() paints children before this overlay
        is drawn.
        """
        def __init__(self, checkbox=None):
            JPanel.__init__(self, BorderLayout())
            self._checkbox = checkbox
            self._selected_overlay = False
            try:
                self.setOpaque(True)
            except Exception:
                pass

        def setSelectionCheckbox(self, checkbox):
            self._checkbox = checkbox
            try:
                self.repaint()
            except Exception:
                pass

        def setSelectedOverlay(self, selected):
            self._selected_overlay = bool(selected)
            try:
                self.repaint()
            except Exception:
                pass

        def _isSelectedForOverlay(self):
            try:
                if self._checkbox is not None:
                    return bool(self._checkbox.isSelected())
            except Exception:
                pass
            return bool(self._selected_overlay)

        def paint(self, g):
            JPanel.paint(self, g)
            _paint_selection_ring(self, g, self._isSelectedForOverlay())

    def _channel_cell_border(selected, cell_pad=2):
        # Keep this as a zero-inset placeholder for older call sites.  The visible
        # ring is painted by _SelectionTilePanel after children, so previews are not
        # resized to make room for the outline.
        return BorderFactory.createEmptyBorder(0, 0, 0, 0)

    class _TileToggleMouse(MouseAdapter):
        def __init__(self, checkbox, refresh_func):
            self.checkbox = checkbox
            self.refresh_func = refresh_func

        def mouseClicked(self, evt):
            try:
                self.checkbox.setSelected(not self.checkbox.isSelected())
                try:
                    self.refresh_func()
                except Exception:
                    pass
                try:
                    evt.consume()
                except Exception:
                    pass
            except Exception:
                pass

    def _make_tile_clickable(components, checkbox, refresh_func):
        """Allow clicking the preview/tile area to toggle its checkbox."""
        listener = _TileToggleMouse(checkbox, refresh_func)
        for comp in components:
            if comp is None:
                continue
            try:
                comp.addMouseListener(listener)
                comp.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR))
                try:
                    comp.setToolTipText(checkbox.getToolTipText())
                except Exception:
                    pass
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # Manual subtraction helpers
    # ---------------------------------------------------------------------
    def _empty_manual_map(n):
        out = {}
        for i in range(n):
            out[i] = set()
        return out

    def _clean_csv_cell(x):
        try:
            return _u(x).replace(u"\ufeff", u"").strip()
        except Exception:
            try:
                return unicode(x).replace(u"\ufeff", u"").strip()
            except Exception:
                return u""

    def _csv_value_means_selected(v):
        """
        SpillOT manual matrix semantics:
          1  or -1  => selected / remove this column channel from this row channel
          0, blank, NaN, or any other number => ignored
        """
        try:
            txt = _clean_csv_cell(v)
            if txt == u"":
                return False
            fv = float(txt)
        except Exception:
            return False
        try:
            return abs(abs(fv) - 1.0) <= 1e-6
        except Exception:
            return False

    def _find_channel_index_from_name(cell_text, names, fallback_idx=None):
        """Forgiving channel-name match with position fallback for legacy matrices."""
        nm = _clean_csv_cell(cell_text)
        if nm == u"":
            if fallback_idx is not None and 0 <= fallback_idx < len(names):
                return fallback_idx
            return None

        name_to_idx = {_u(cur).upper(): i for i, cur in enumerate(names)}
        i = name_to_idx.get(nm.upper(), None)
        if i is not None:
            return i

        # Token matching in both directions helps with labels like "CD8 (Opal 70)"
        # or minor punctuation differences.
        for cur_nm, cur_i in name_to_idx.items():
            try:
                if _tok_match(nm.upper(), cur_nm) or _tok_match(cur_nm, nm.upper()):
                    return cur_i
            except Exception:
                pass

        # If the matrix has the expected number/order of rows or columns, use the
        # position as a fallback instead of discarding the whole CSV. This makes
        # legacy matrices robust to small name typos such as CD11b/CD116.
        if fallback_idx is not None and 0 <= fallback_idx < len(names):
            return fallback_idx
        return None

    def _load_manual_subtractions_from_saved_csv(csv_path, names):
        manual_map = _empty_manual_map(len(names))
        if not os.path.exists(csv_path):
            return manual_map
        try:
            with codecs.open(csv_path, "r", "utf-8", "replace") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                return manual_map

            header = [_clean_csv_cell(c) for c in lines[0].split(u",")]
            if len(header) < 2:
                return manual_map

            header_names = header[1:]
            use_position_fallback_cols = (len(header_names) >= len(names))
            col_to_idx = []
            for pos, nm in enumerate(header_names):
                fallback = pos if use_position_fallback_cols else None
                col_to_idx.append(_find_channel_index_from_name(nm, names, fallback))

            data_lines = lines[1:]
            use_position_fallback_rows = (len(data_lines) >= len(names))
            for row_pos, rline in enumerate(data_lines):
                cells = [_clean_csv_cell(c) for c in rline.split(u",")]
                if len(cells) < 2:
                    continue

                fallback_row = row_pos if use_position_fallback_rows else None
                row_idx = _find_channel_index_from_name(cells[0], names, fallback_row)
                if row_idx is None:
                    continue

                vals = cells[1:]
                for j, v in enumerate(vals):
                    if j >= len(col_to_idx):
                        break
                    col_idx = col_to_idx[j]
                    if col_idx is None or row_idx == col_idx:
                        continue
                    if _csv_value_means_selected(v):
                        manual_map[row_idx].add(col_idx)

            return manual_map
        except Exception:
            return _empty_manual_map(len(names))

    # ---------------------------------------------------------------------
    # Up-front selection dialog (previous interface style), but expanded cell usage
    # ---------------------------------------------------------------------
    def _select_planes_to_debleed(names, imp, axis_used, defaults=None):
        label = _axis_label(axis_used)
        n = len(names)
        if n <= 0:
            return None
        if defaults is None:
            defaults = set()
        defaults = set([int(x) for x in defaults if 0 <= int(x) < n])

        ngrid = int(math.ceil(math.sqrt(float(n))))

        # layout tuning: reduce dead space
        GRID_GAP   = 4
        OUTER_PAD  = 6
        CELL_PAD   = 2

        # resize behavior tuning
        SIZE_STEP  = 8      # quantize thumb dims to reduce regen thrash
        MIN_THUMB  = 12

        dlg = JDialog(None, "SpillOT: select %ss to clean" % label, True)
        dlg.setLayout(BorderLayout())

        # Top message (header only; remove everything underneath)
        top = JPanel()
        top.setLayout(BoxLayout(top, BoxLayout.Y_AXIS))
        msg = (u"<html><b>SpillOT: select which %ss to clean</b></html>") % (_u(label),)
        top.add(JLabel(msg))
        dlg.add(top, BorderLayout.NORTH)

        # Grid in center
        grid = JPanel(GridLayout(ngrid, ngrid, GRID_GAP, GRID_GAP))
        grid.setBorder(BorderFactory.createEmptyBorder(OUTER_PAD, OUTER_PAD, OUTER_PAD, OUTER_PAD))
        dlg.add(grid, BorderLayout.CENTER)

        # Bottom OK/Cancel
        bottom = JPanel(FlowLayout(FlowLayout.RIGHT))
        btn_clear = JButton("Clear all")
        btn_ok = JButton("OK")
        btn_cancel = JButton("Cancel")
        try:
            btn_ok.setMargin(Insets(2, 10, 2, 10))
            btn_cancel.setMargin(Insets(2, 10, 2, 10))
        except Exception:
            pass
        bottom.add(btn_clear)
        bottom.add(btn_ok)
        bottom.add(btn_cancel)
        dlg.add(bottom, BorderLayout.SOUTH)

        # Checkboxes (NONE selected by default), compact single-line-ish label
        cbs = []
        for i, nm in enumerate(names, start=1):
            html = u"<html><center><b>%s</b> <span style='font-size:9px;color:gray'>(%d)</span></center></html>" % (_u(nm), i)
            cb = JCheckBox(html, (i - 1) in defaults)
            cb.setHorizontalAlignment(SwingConstants.CENTER)
            cb.setToolTipText(u"%s %d: %s" % (_u(label), i, _u(nm)))
            try:
                cb.setMargin(Insets(0, 0, 0, 0))
            except Exception:
                pass
            cbs.append(cb)

        # Cache icons for current thumb size only
        icons_cache = {}
        state = {"tw": None, "th": None, "font_pt": None}
        cell_panels = {}

        def _refresh_selection_borders():
            for cell_i, cell in cell_panels.items():
                try:
                    try:
                        cell.setSelectedOverlay(cbs[cell_i].isSelected())
                    except Exception:
                        pass
                    cell.setBorder(_channel_cell_border(cbs[cell_i].isSelected(), CELL_PAD))
                    cell.repaint()
                except Exception:
                    pass

        def _on_checkbox_toggle(_):
            _refresh_selection_borders()

        for cb in cbs:
            cb.addActionListener(_on_checkbox_toggle)

        def _q(px):
            px = int(px)
            if px <= 0:
                return MIN_THUMB
            return max(MIN_THUMB, SIZE_STEP * int(round(float(px) / float(SIZE_STEP))))

        def _compute_cell_dims():
            gw, gh = grid.getWidth(), grid.getHeight()
            if gw <= 0 or gh <= 0:
                gw = max(300, dlg.getWidth() - 60)
                gh = max(200, dlg.getHeight() - 200)

            try:
                ins = grid.getInsets()
                avail_w = max(1, gw - ins.left - ins.right)
                avail_h = max(1, gh - ins.top  - ins.bottom)
            except Exception:
                avail_w, avail_h = gw, gh

            cell_w = float(avail_w - (ngrid - 1) * GRID_GAP) / float(ngrid)
            cell_h = float(avail_h - (ngrid - 1) * GRID_GAP) / float(ngrid)
            return cell_w, cell_h

        def _font_for_cell(cell_w, cell_h):
            # Scale font with cell size (small grids => bigger text/checkbox)
            # Clamp to keep it sane.
            pt = int(max(9, min(18, cell_h / 7.0)))
            return pt

        def _ensure_font(pt):
            if state["font_pt"] == pt:
                return
            try:
                f = Font("SansSerif", Font.PLAIN, int(pt))
                for cb in cbs:
                    cb.setFont(f)
            except Exception:
                pass
            state["font_pt"] = pt

        def _max_checkbox_height():
            mh = 0
            for cb in cbs:
                try:
                    h = cb.getPreferredSize().height
                    if h > mh:
                        mh = h
                except Exception:
                    pass
            if mh <= 0:
                mh = 24
            return mh

        def _get_icon(i0, tw, th):
            if i0 not in icons_cache:
                icons_cache[i0] = _make_thumb_icon_for_plane(imp, i0 + 1, thumb_w=tw, thumb_h=th)
            return icons_cache[i0]

        def rebuild_grid():
            cell_w, cell_h = _compute_cell_dims()

            # Font scaled to the cell; affects checkbox height
            fpt = _font_for_cell(cell_w, cell_h)
            _ensure_font(fpt)

            cb_h = _max_checkbox_height()

            # Compute rectangle for thumbnail (fills width, and height above checkbox)
            tw = int(cell_w - 2 * CELL_PAD)
            th = int(cell_h - cb_h - 2 * CELL_PAD)

            # If there are super long labels that wrap, th might get tiny; clamp.
            tw = max(MIN_THUMB, tw)
            th = max(MIN_THUMB, th)

            # Quantize to reduce regenerating while dragging
            twq = _q(tw)
            thq = _q(th)

            if state["tw"] != twq or state["th"] != thq:
                icons_cache.clear()
                state["tw"], state["th"] = twq, thq

            grid.removeAll()
            cell_panels.clear()
            total_cells = ngrid * ngrid

            for cell_i in range(total_cells):
                if cell_i < n:
                    cell = _SelectionTilePanel(cbs[cell_i])
                    cell.setBorder(_channel_cell_border(cbs[cell_i].isSelected(), CELL_PAD))
                    cell.setSelectedOverlay(cbs[cell_i].isSelected())
                    cell_panels[cell_i] = cell
                    icon = _get_icon(cell_i, twq, thq)
                    img_lab = JLabel(icon)
                    img_lab.setHorizontalAlignment(SwingConstants.CENTER)
                    img_lab.setVerticalAlignment(SwingConstants.CENTER)
                    cell.add(img_lab, BorderLayout.CENTER)

                    cb = cbs[cell_i]
                    _make_tile_clickable((cell, img_lab), cb, _refresh_selection_borders)
                    cb.setHorizontalAlignment(SwingConstants.CENTER)
                    cell.add(cb, BorderLayout.SOUTH)
                else:
                    cell = JPanel(BorderLayout())
                    cell.setBorder(BorderFactory.createEmptyBorder(CELL_PAD, CELL_PAD, CELL_PAD, CELL_PAD))
                    cell.add(JLabel(""), BorderLayout.CENTER)

                grid.add(cell)

            _refresh_selection_borders()
            grid.revalidate()
            grid.repaint()

        # OK/Cancel
        result = {"ok": False, "sel": []}

        def on_ok(_):
            sel = [i + 1 for i, cb in enumerate(cbs) if cb.isSelected()]
            if not sel:
                JOptionPane.showMessageDialog(
                    dlg,
                    u"Select at least one %s to clean with SpillOT." % _u(label),
                    "Nothing selected",
                    JOptionPane.WARNING_MESSAGE
                )
                return
            result["ok"] = True
            result["sel"] = sel
            dlg.dispose()

        def on_clear(_):
            for cb in cbs:
                cb.setSelected(False)
            _refresh_selection_borders()

        def on_cancel(_):
            dlg.dispose()

        btn_clear.addActionListener(on_clear)
        btn_ok.addActionListener(on_ok)
        btn_cancel.addActionListener(on_cancel)

        # Default to about 70% of the usable screen, centered.
        _set_dialog_to_screen_fraction(dlg, 0.70, 600, 450)

        # Build now (estimate), then once after realized
        rebuild_grid()

        class _Later(Runnable):
            def run(self):
                rebuild_grid()
        SwingUtilities.invokeLater(_Later())

        # LIVE recompute on resize + debounce
        resize_timer = {"t": None}

        class _DoRebuild(ActionListener):
            def actionPerformed(self, evt):
                try:
                    rebuild_grid()
                finally:
                    try:
                        evt.getSource().stop()
                    except:
                        pass
                    resize_timer["t"] = None

        class _OnResize(ComponentAdapter):
            def componentResized(self, evt):
                try:
                    if resize_timer["t"] is not None:
                        resize_timer["t"].stop()
                except:
                    pass
                t = Timer(120, _DoRebuild())
                t.setRepeats(False)
                resize_timer["t"] = t
                t.start()

        dlg.addComponentListener(_OnResize())

        dlg.setVisible(True)

        if not result["ok"]:
            return None
        return result["sel"]

    # Load any existing SpillOT matrix before the menus so the per-target
    # "which channels are bleeding into this one?" dialogs can be prefilled.
    # Do NOT use this CSV to preselect the first "which channels to clean" menu;
    # the user should choose the target channels explicitly for each run.
    manual_csv_path = os.path.splitext(out_base_path or img_path)[0] + ".csv"
    manual_subtractions = _load_manual_subtractions_from_saved_csv(manual_csv_path, names)

    selected_channels = _select_planes_to_debleed(names, imp, axis_used)
    if selected_channels is None:
        sys.exit()


    selected_set = set([c - 1 for c in selected_channels])  # 0-based indices

    def _select_subtractions_for_channel(target_idx, names, imp, axis_used, defaults=None, position=1, total=1):
        if defaults is None:
            defaults = set()

        candidate_indices = [i for i in range(len(names)) if i != target_idx]
        if not candidate_indices:
            return set()

        defaults = set([i for i in defaults if i != target_idx])

        dlg_title = u"Bleed-through sources (%d/%d)" % (position, total) if total > 1 else u"Bleed-through sources"
        dlg = JDialog(None, dlg_title, True)
        dlg.setLayout(BorderLayout())

        label = _axis_label(axis_used)
        target_name = _u(names[target_idx])

        # -----------------------------------------------------------------
        # Header: clear wording, no CSV/donor terminology.
        # -----------------------------------------------------------------
        north = JPanel()
        north.setLayout(BoxLayout(north, BoxLayout.Y_AXIS))
        north.setBorder(BorderFactory.createEmptyBorder(10, 12, 6, 12))
        title = JLabel(
            u"<html><b>For %s %d: %s</b></html>" %
            (_u(label), target_idx + 1, target_name)
        )
        title.setAlignmentX(0.0)
        north.add(title)
        instruction = JLabel(
            u"<html><span style='font-size:11px'>"
            u"Check the channels that you suspect are bleeding into this selected channel. "
            u"Checked channels are removed from the selected channel only where the structural patch-similarity logic finds a match."
            u"</span></html>"
        )
        instruction.setAlignmentX(0.0)
        north.add(instruction)
        dlg.add(north, BorderLayout.NORTH)

        # -----------------------------------------------------------------
        # Left: selected target channel.
        # -----------------------------------------------------------------
        target_panel = JPanel()
        target_panel.setLayout(BoxLayout(target_panel, BoxLayout.Y_AXIS))
        target_panel.setBorder(BorderFactory.createTitledBorder("Selected channel to clean"))

        target_title = JLabel(
            u"<html><center><b>%s</b><br/><span style='font-size:10px;color:gray'>%s %d</span></center></html>" %
            (target_name, _u(label), target_idx + 1)
        )
        target_title.setHorizontalAlignment(SwingConstants.CENTER)
        target_title.setAlignmentX(0.5)
        target_panel.add(target_title)
        target_panel.add(JLabel(" "))

        target_icon = _make_thumb_icon_for_plane(imp, target_idx + 1, thumb_w=300, thumb_h=300)
        target_img = JLabel(target_icon)
        target_img.setHorizontalAlignment(SwingConstants.CENTER)
        target_img.setAlignmentX(0.5)
        target_panel.add(target_img)
        target_panel.add(JLabel(" "))

        target_help = JLabel(u"<html><center><span style='font-size:10px;color:gray'>This is the channel being cleaned.</span></center></html>")
        target_help.setHorizontalAlignment(SwingConstants.CENTER)
        target_help.setAlignmentX(0.5)
        target_panel.add(target_help)
        try:
            target_panel.setPreferredSize(Dimension(340, 430))
            target_panel.setMinimumSize(Dimension(300, 380))
        except Exception:
            pass

        # -----------------------------------------------------------------
        # Middle: explicit visual direction from right side into left target.
        # -----------------------------------------------------------------
        arrow_panel = JPanel(BorderLayout())
        arrow_panel.setBorder(BorderFactory.createEmptyBorder(10, 8, 10, 8))
        arrow_label = JLabel(
            u"<html><center>"
            u"<span style='font-size:34px'>&larr;</span><br/>"
            u"<b>remove</b><br/>"
            u"where<br/>similar"
            u"</center></html>"
        )
        arrow_label.setHorizontalAlignment(SwingConstants.CENTER)
        arrow_label.setVerticalAlignment(SwingConstants.CENTER)
        arrow_panel.add(arrow_label, BorderLayout.CENTER)
        try:
            arrow_panel.setPreferredSize(Dimension(100, 430))
            arrow_panel.setMinimumSize(Dimension(90, 380))
        except Exception:
            pass

        # -----------------------------------------------------------------
        # Right: grid of possible bleed-in channels, similar to the first menu.
        # -----------------------------------------------------------------
        selector_panel = JPanel(BorderLayout())
        selector_panel.setBorder(BorderFactory.createTitledBorder("Check channels that may be bleeding into the selected channel"))

        count = len(candidate_indices)
        grid_cols = int(math.ceil(math.sqrt(float(count))))
        grid_rows = int(math.ceil(float(count) / float(grid_cols))) if grid_cols > 0 else 1
        GRID_GAP = 4
        OUTER_PAD = 6
        CELL_PAD = 2
        SIZE_STEP = 8
        MIN_ICON = 36
        MAX_ICON = 520

        channel_grid = JPanel(GridLayout(grid_rows, grid_cols, GRID_GAP, GRID_GAP))
        channel_grid.setBorder(BorderFactory.createEmptyBorder(OUTER_PAD, OUTER_PAD, OUTER_PAD, OUTER_PAD))

        channel_checkboxes = []
        image_labels = {}
        cell_panels = {}
        cb_by_idx = {}
        total_cells = grid_rows * grid_cols
        for cell_i in range(total_cells):
            if cell_i < count:
                idx = candidate_indices[cell_i]
                cell = _SelectionTilePanel(None)
                cell.setBorder(_channel_cell_border(idx in defaults, CELL_PAD))
                cell.setSelectedOverlay(idx in defaults)
                cell_panels[idx] = cell
                icon = _make_thumb_icon_for_plane(imp, idx + 1, thumb_w=120, thumb_h=120)
                img_lab = JLabel(icon)
                img_lab.setHorizontalAlignment(SwingConstants.CENTER)
                img_lab.setVerticalAlignment(SwingConstants.CENTER)
                image_labels[idx] = img_lab
                cell.add(img_lab, BorderLayout.CENTER)

                html = u"<html><center><b>%s</b> <span style='font-size:9px;color:gray'>(%d)</span></center></html>" % (_u(names[idx]), idx + 1)
                cb = JCheckBox(html, idx in defaults)
                cb.setHorizontalAlignment(SwingConstants.CENTER)
                cb.setToolTipText(u"Mark %s (%d) as possibly bleeding into %s (%d)" % (_u(names[idx]), idx + 1, target_name, target_idx + 1))
                try:
                    cb.setMargin(Insets(0, 0, 0, 0))
                except Exception:
                    pass
                try:
                    cell.setSelectionCheckbox(cb)
                except Exception:
                    pass
                channel_checkboxes.append((idx, cb))
                cb_by_idx[idx] = cb
                cell.add(cb, BorderLayout.SOUTH)
            else:
                cell = JPanel(BorderLayout())
                cell.setBorder(BorderFactory.createEmptyBorder(CELL_PAD, CELL_PAD, CELL_PAD, CELL_PAD))
                cell.add(JLabel(""), BorderLayout.CENTER)

            channel_grid.add(cell)

        def _refresh_source_selection_borders():
            for idx, cell in cell_panels.items():
                try:
                    cb = cb_by_idx.get(idx, None)
                    if cb is not None:
                        try:
                            cell.setSelectedOverlay(cb.isSelected())
                        except Exception:
                            pass
                        cell.setBorder(_channel_cell_border(cb.isSelected(), CELL_PAD))
                        cell.repaint()
                except Exception:
                    pass

        def _on_source_checkbox_toggle(_):
            _refresh_source_selection_borders()

        for _idx, cb in channel_checkboxes:
            cb.addActionListener(_on_source_checkbox_toggle)

        for _idx, cb in channel_checkboxes:
            _make_tile_clickable((cell_panels.get(_idx, None), image_labels.get(_idx, None)),
                                 cb, _refresh_source_selection_borders)

        scroll = JScrollPane(channel_grid)
        try:
            scroll.getVerticalScrollBar().setUnitIncrement(24)
            scroll.getHorizontalScrollBar().setUnitIncrement(24)
        except Exception:
            pass
        selector_panel.add(scroll, BorderLayout.CENTER)

        icon_cache = {}
        state = {"tw": None, "th": None, "font_pt": None}

        def _q_icon(px):
            px = int(px)
            if px <= 0:
                return MIN_ICON
            return max(MIN_ICON, min(MAX_ICON, SIZE_STEP * int(round(float(px) / float(SIZE_STEP)))))

        def _font_for_cell(cell_w, cell_h):
            return int(max(9, min(16, cell_h / 7.5)))

        def _ensure_font(pt):
            if state["font_pt"] == pt:
                return
            try:
                f = Font("SansSerif", Font.PLAIN, int(pt))
                for _idx, cb in channel_checkboxes:
                    cb.setFont(f)
            except Exception:
                pass
            state["font_pt"] = pt

        def _checkbox_height():
            mh = 0
            for _idx, cb in channel_checkboxes:
                try:
                    h = cb.getPreferredSize().height
                    if h > mh:
                        mh = h
                except Exception:
                    pass
            if mh <= 0:
                mh = 24
            return mh

        def _rebuild_channel_icons():
            try:
                extent = scroll.getViewport().getExtentSize()
                gw, gh = int(extent.width), int(extent.height)
            except Exception:
                gw, gh = channel_grid.getWidth(), channel_grid.getHeight()

            if gw <= 0 or gh <= 0:
                gw = max(700, dlg.getWidth() - 520)
                gh = max(440, dlg.getHeight() - 180)

            try:
                ins = channel_grid.getInsets()
                avail_w = max(1, gw - ins.left - ins.right)
                avail_h = max(1, gh - ins.top - ins.bottom)
            except Exception:
                avail_w, avail_h = gw, gh

            cell_w = float(avail_w - (grid_cols - 1) * GRID_GAP) / float(grid_cols)
            cell_h = float(avail_h - (grid_rows - 1) * GRID_GAP) / float(grid_rows)

            fpt = _font_for_cell(cell_w, cell_h)
            _ensure_font(fpt)
            cb_h = _checkbox_height()

            tw = int(cell_w - 2 * CELL_PAD)
            th = int(cell_h - cb_h - 2 * CELL_PAD)
            twq = _q_icon(tw)
            thq = _q_icon(th)

            if state["tw"] == twq and state["th"] == thq:
                return
            state["tw"], state["th"] = twq, thq
            icon_cache.clear()

            for idx, lab in image_labels.items():
                icon = icon_cache.get(idx)
                if icon is None:
                    icon = _make_thumb_icon_for_plane(imp, idx + 1, thumb_w=twq, thumb_h=thq)
                    icon_cache[idx] = icon
                lab.setIcon(icon)
                try:
                    lab.setPreferredSize(Dimension(twq, thq))
                except Exception:
                    pass

            channel_grid.revalidate()
            channel_grid.repaint()

        right_side = JPanel(BorderLayout())
        right_side.add(arrow_panel, BorderLayout.WEST)
        right_side.add(selector_panel, BorderLayout.CENTER)

        center = JPanel(BorderLayout())
        center.setBorder(BorderFactory.createEmptyBorder(4, 10, 4, 10))
        center.add(target_panel, BorderLayout.WEST)
        center.add(right_side, BorderLayout.CENTER)
        dlg.add(center, BorderLayout.CENTER)

        bottom = JPanel(FlowLayout(FlowLayout.RIGHT))
        btn_none = JButton("Clear all")
        btn_ok = JButton("Next" if position < total else "OK")
        btn_cancel = JButton("Cancel")
        bottom.add(btn_none)
        bottom.add(btn_ok)
        bottom.add(btn_cancel)
        dlg.add(bottom, BorderLayout.SOUTH)

        result = {"ok": False, "sel": set()}

        def _set_all(state_value):
            for _idx, cb in channel_checkboxes:
                cb.setSelected(state_value)
            _refresh_source_selection_borders()

        def on_none(_):
            _set_all(False)

        def on_ok(_):
            sel = set([idx for idx, cb in channel_checkboxes if cb.isSelected()])
            result["ok"] = True
            result["sel"] = sel
            dlg.dispose()

        def on_cancel(_):
            dlg.dispose()

        btn_none.addActionListener(on_none)
        btn_ok.addActionListener(on_ok)
        btn_cancel.addActionListener(on_cancel)

        _set_dialog_to_screen_fraction(dlg, 0.70, 700, 500)
        _refresh_source_selection_borders()

        _rebuild_channel_icons()

        class _LaterChannelIcons(Runnable):
            def run(self):
                _rebuild_channel_icons()
        SwingUtilities.invokeLater(_LaterChannelIcons())

        resize_timer = {"t": None}

        class _DoChannelIconRebuild(ActionListener):
            def actionPerformed(self, evt):
                try:
                    _rebuild_channel_icons()
                finally:
                    try:
                        evt.getSource().stop()
                    except Exception:
                        pass
                    resize_timer["t"] = None

        class _OnChannelResize(ComponentAdapter):
            def componentResized(self, evt):
                try:
                    if resize_timer["t"] is not None:
                        resize_timer["t"].stop()
                except Exception:
                    pass
                t = Timer(120, _DoChannelIconRebuild())
                t.setRepeats(False)
                resize_timer["t"] = t
                t.start()

        dlg.addComponentListener(_OnChannelResize())
        dlg.setVisible(True)

        if not result["ok"]:
            return None
        return result["sel"]

    # Only rows selected in the first menu are edited in this run. Rows for
    # channels that were not selected are preserved from the existing CSV, if any.
    # To clear a row, select that target channel and clear its bleed-in selections.
    for pos, ch in enumerate(selected_channels, start=1):
        target_idx = ch - 1
        defaults = manual_subtractions.get(target_idx, set())
        sel = _select_subtractions_for_channel(target_idx, names, imp, axis_used,
                                               defaults=defaults, position=pos, total=len(selected_channels))
        if sel is None:
            sys.exit()
        manual_subtractions[target_idx] = set(sorted(sel))

    csv = [u"," + u",".join([_u(n) for n in names])]
    for r in range(n_ch):
        row = [_u(names[r])]
        donor_set = manual_subtractions.get(r, set())
        for c in range(n_ch):
            if r == c:
                row.append(u"0")
            elif c in donor_set:
                row.append(u"1")
            else:
                row.append(u"0")
        csv.append(u",".join(row))
    with codecs.open(manual_csv_path, "w", "utf-8") as f:
        f.write(u"\n".join(csv))

    # ---------------------------------------------------------------------
    # Run parameters dialog (planes already selected up front)
    # ---------------------------------------------------------------------
    _prefilled_env = _guess_conda_env_root("rfot")

    dlg = GenericDialog("Run SpillOT")
    dlg.addMessage(
        "Patch size controls the neighborhood SpillOT uses to resolve bleed-through.\n"
        "- Must be an EVEN integer >= 4.\n"
        "- Lower values -> more aggressive removal and faster runs.\n"
        "\n"
        "Optionally, ignore overexposed pixels by setting saturated pixels to 0."
    )
    dlg.addNumericField("Patch size (patsize):", 16, 0)
    dlg.addStringField("Conda env path (root of env, e.g. .../envs/rfot):", _prefilled_env or "", 50)
    dlg.addCheckbox("Ignore overexposed pixels (set saturated to 0)", False)

    dlg.showDialog()
    if dlg.wasCanceled():
        sys.exit()

    patsize = int(round(dlg.getNextNumber()))
    env_root = dlg.getNextString().strip()
    ignore_overexposed = dlg.getNextBoolean()

    if dlg.invalidNumber() or patsize < 4 or (patsize % 2 != 0):
        IJ.showMessage("Invalid patch size",
                       "Patch size must be an EVEN integer >= 4.\nYou entered: %s." % patsize)
        sys.exit()

    if not env_root:
        env_root = _guess_conda_env_root("rfot")
    if not env_root:
        IJ.showMessage("Conda env missing",
                       "Couldn't find a conda env named 'rfot'. Please paste the full path to your env.\n"
                       "Examples:\n"
                       "  Windows:  C:\\Users\\<you>\\miniconda3\\envs\\rfot\n"
                       "  Linux:    /home/<you>/mambaforge/envs/rfot\n"
                       "  macOS:    /opt/anaconda3/envs/rfot  (or)  /Users/<you>/miniforge3/envs/rfot")
        sys.exit()

    pyexe = _python_from_env(env_root)
    if not os.path.exists(pyexe):
        IJ.showMessage("Python not found",
                       "Could not find Python inside the env:\n%s\nExpected at:\n%s" % (env_root, pyexe))
        sys.exit()

    subproc_env = _subproc_env_for_conda_env(env_root)

    # ---------------------------------------------------------------------
    # Launch runner (external Python) using the manual donor CSV
    # ---------------------------------------------------------------------
    def _this_script_dir():
        cand_dirs = []
        try:
            here = globals().get("__file__", None)
            if here:
                cand_dirs.append(os.path.dirname(os.path.abspath(here)))
        except Exception:
            pass
        try:
            argv0 = sys.argv[0]
            if argv0:
                cand_dirs.append(os.path.dirname(os.path.abspath(argv0)))
        except Exception:
            pass
        try:
            plugins_dir = IJ.getDir("plugins")
            if plugins_dir:
                plugins_dir = os.path.abspath(plugins_dir)
                cand_dirs.append(plugins_dir)
                cand_dirs.append(os.path.join(plugins_dir, "SpillOT"))
                cand_dirs.append(os.path.join(plugins_dir, "SpillOT", "bin"))
                cand_dirs.append(os.path.join(plugins_dir, "Debleed"))
                cand_dirs.append(os.path.join(plugins_dir, "Debleed", "bin"))
        except Exception:
            pass

        for d in cand_dirs:
            if d and os.path.isdir(d):
                return d
        return ""

    def _find_runner_script():
        runner_names = ["SpillOT.py", "spillot.py", "debleed.py"]
        cand = []
        here = _this_script_dir()
        if here:
            for runner_name in runner_names:
                cand.append(os.path.join(here, runner_name))
                cand.append(os.path.join(here, "bin", runner_name))

        try:
            plugins_dir = IJ.getDir("plugins")
        except Exception:
            plugins_dir = None

        if plugins_dir:
            plugins_dir = os.path.abspath(plugins_dir)
            for runner_name in runner_names:
                cand.append(os.path.join(plugins_dir, runner_name))
                cand.append(os.path.join(plugins_dir, "SpillOT", runner_name))
                cand.append(os.path.join(plugins_dir, "SpillOT", "bin", runner_name))
                # Backward-compatible search paths for older installs.
                cand.append(os.path.join(plugins_dir, "Debleed", runner_name))
                cand.append(os.path.join(plugins_dir, "Debleed", "bin", runner_name))

        seen = set()
        for p in cand:
            if (not p) or (p in seen):
                continue
            seen.add(p)
            if os.path.exists(p):
                return p
        return None

    runner_py = _find_runner_script()
    if not runner_py:
        abort("Could not find 'SpillOT.py'. Put it beside SpillOT_Run.py (or in a sibling 'bin' folder).")

    channels = sorted(set(int(c) for c in selected_channels))
    if not channels:
        abort("No channels selected for SpillOT.")
    wait_dlg, wait_bar, wait_timer = _show_progress(len(channels))
    if len(channels) > 1:
        wait_bar.setIndeterminate(False)
        wait_bar.setMinimum(0)
        wait_bar.setMaximum(len(channels))
        wait_bar.setValue(0)
        wait_bar.setStringPainted(True)
        wait_bar.setString("0 / %d (0%%)" % len(channels))

    try:
        for i, ch in enumerate(channels, start=1):
            IJ.showStatus("SpillOT processing channel %d of %d" % (i, len(channels)))
            if len(channels) > 1:
                _pb_smooth_to(wait_bar, i)
                pct = int(round(100.0 * i / float(len(channels))))
                wait_bar.setString("%d / %d (%d%%)" % (i, len(channels), pct))
                wait_bar.repaint()

            cmd = [pyexe, runner_py, img_path, str(ch), "--patsize", str(patsize),
                   "--manual_csv", manual_csv_path]
            if ignore_overexposed:
                cmd.append("--ignore_overexposed")

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=subproc_env)
            stdout, stderr = proc.communicate()

            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", "replace")
                abort(u"SpillOT runner failed for channel %d (exit %d).\n\n%s" % (ch, proc.returncode, _u(err_msg)))

            # SpillOT.py writes the new suffix; keep the old suffix as a fallback for compatibility.
            out_candidates = [
                "%s_Channel_%d_SpillOT.tif" % (os.path.splitext(img_path)[0], ch),
                "%s_Channel_%d_SpillOT.tif" % (img_path[:-4], ch),
                "%s_Channel_%d_debleed.tif" % (os.path.splitext(img_path)[0], ch),
                "%s_Channel_%d_debleed.tif" % (img_path[:-4], ch),
            ]
            out = None
            for cand_out in out_candidates:
                if cand_out and os.path.exists(cand_out):
                    out = cand_out
                    break
            if out is None:
                abort(u"SpillOT result not found for channel %d. Expected one of:\n%s" %
                      (ch, _u(u"\n".join([_u(x) for x in out_candidates]))))
            paths.append(out)
    finally:
        _pb_cleanup(wait_bar, wait_timer)
        wait_dlg.dispose()
        IJ.showProgress(1.0)
        IJ.showStatus("Finished.")

    # ---------------------------------------------------------------------
    # Combine outputs into a hyperstack and attach labels with manual bleed-in selections
    # ---------------------------------------------------------------------
    if not paths:
        abort("No output files generated.")

    opener = Opener()
    imps = [opener.openImage(p) for p in paths]

    def _channel_label_with_subtractions(zero_based_idx):
        nm = _u(names[zero_based_idx])
        selected_sources = sorted(list(manual_subtractions.get(zero_based_idx, set())))
        if not selected_sources:
            return nm
        source_txt = u", ".join([_u(names[d]) for d in selected_sources])
        return nm + u"\nremove if similar: " + source_txt

    if len(imps) == 1:
        imp_single = imps[0]
        stk_single = imp_single.getStack()
        ch0 = channels[0] - 1
        stk_single.setSliceLabel(_channel_label_with_subtractions(ch0), 1)
        imp_single.updateAndDraw()
        imp_single.show()
    else:
        w, h = imps[0].getWidth(), imps[0].getHeight()
        stack = ImageStack(w, h)
        for imp_ in imps:
            stack.addSlice(imp_.getProcessor())
        result = ImagePlus("SpillOT processed channels", stack)
        result.setDimensions(len(imps), 1, 1)
        result.setOpenAsHyperStack(True)
        stk = result.getStack()
        for c_idx, ch_num in enumerate(channels, start=1):
            stk.setSliceLabel(_channel_label_with_subtractions(ch_num - 1), c_idx)
        result.updateAndDraw()
        result.show()

    # ---------------------------------------------------------------------
    # Build + save FULL stack with SpillOT-cleaned planes replaced (others unchanged)
    # Saved as: <original_stack_name>_SpillOT.tif
    # ---------------------------------------------------------------------
    try:
        # Map original stack slice-index -> SpillOT-cleaned ImageProcessor
        repl_by_stack_idx = {}
        for i, ch_num in enumerate(channels):
            try:
                de_imp = imps[i]
                if de_imp is None:
                    continue
                # duplicate to avoid sharing processor objects across stacks
                repl_by_stack_idx[slice_idx(ch_num)] = de_imp.getProcessor().duplicate()
            except Exception:
                pass

        orig_stack = imp.getStack()
        full_stack = ImageStack(imp.getWidth(), imp.getHeight())

        for si in range(1, imp.getStackSize() + 1):
            lbl = orig_stack.getSliceLabel(si)
            ip  = repl_by_stack_idx.get(si, orig_stack.getProcessor(si))
            full_stack.addSlice(lbl, ip)

        # Title for display
        try:
            base_title = os.path.splitext(os.path.basename(out_base_path))[0]
        except Exception:
            base_title = "Full SpillOT"
        full_imp = ImagePlus(_u(base_title) + u" (SpillOT)", full_stack)

        # Preserve hyperstack dims + calibration if possible
        try:
            full_imp.setDimensions(imp.getNChannels(), imp.getNSlices(), imp.getNFrames())
            full_imp.setOpenAsHyperStack(True)
        except Exception:
            pass
        try:
            full_imp.setCalibration(imp.getCalibration())
        except Exception:
            pass
        try:
            info_prop = imp.getProperty("Info")
            if info_prop:
                full_imp.setProperty("Info", info_prop)
        except Exception:
            pass

        full_imp.updateAndDraw()
        full_imp.show()

        out_full = os.path.splitext(out_base_path)[0] + "_SpillOT.tif"

        fs = FileSaver(full_imp)
        if full_imp.getStackSize() > 1:
            ok = fs.saveAsTiffStack(out_full)
        else:
            ok = fs.saveAsTiff(out_full)

        if not ok:
            abort(u"Could not save full SpillOT stack:\n%s" % _u(out_full))

        IJ.showStatus(u"Saved full SpillOT stack: " + _u(out_full))

    except Exception as e:
        abort(u"Failed building/saving the full SpillOT stack.\n\n%s" % _u(e))

except SystemExit:
    # IMPORTANT:
    # - Fiji/SciJava prints uncaught SystemExit as a scary ERROR traceback.
    # - This swallow makes *Cancel / early exit* look clean in the console,
    #   regardless of which dialog/menu the user cancels from.
    pass
