# Copyright 2025, Jason Lequyer, Ferris Nowlan and Laurence Pelletier.
# All rights reserved.
# Sinai Health System - Lunenfeld Tanenbaum Research Institute
# 600 University Avenue, Room 1070, Toronto, ON M5G 1X5, Canada

try:

    groups_by_name  = [
        ["CA2"],
        ["CD116"],
        ["CD14"],
        ["CD20"],
        ["CD3"],
        ["CD4"],
        ["CD44"],
        ["CD45"],
        ["CD45RO"],
        ["CD56"],
        ["CD57"],
        ["CD68"],
        ["CD8"],
        ["DNA1"],
        ["DNA2"],
        ["FOXP3"],
        ["Granzyme B"],
        ["Ki67"],
        ["NF-KB"],
        ["NKX6.1"],
        ["PDX-1"],
        ["pan-Keratin"],
        ["NESTIN"],
        ["CD31"],
        ["HLA-ABC"],
        ["C-PEPTIDE"],
        ["GLUCAGON"],
        ["SOMATOSTATIN"],
        ["B-ACTIN"],
        ["Collagen type I"],
        ["pS6"],
        ["HLA-DR"],
        ["PANCREATIC POLYPEPTIDE"],
        ["GHRELIN"],
        ["LYVE1"],
        ["PNAd"],
        ["PDL1"],
        ["DAPI"],
        ["PanCK"],
        ["tcrgd"],
        ["PD1"],
        ["CD3", "CD4"],
        ["CD3", "CD8"],
        ["CD3", "CD56"],
        ["CD56", "CD16"],
        ["BCMA", "CD138"],
        ["CD14", "HLA-DR"],
        ["CD68", "HLA-DR"],
        ["CD11c", "HLA-DR"],
        ["CD123", "HLA-DR"],
        ["CD16", "CD66b"],
        ["CD16", "CD15"],
        ["CD193", "Siglec-8"],
        ["FcεR1α", "CD63", "CD203c"],
        ["FcεR1α", "CD117", "Tryptase"],
        ["PanCK", "pan-Keratin"],
        ["DNA1", "DNA2"],
        ["DAPI", "DNA1", "DNA2"],
    ]


    # ---------------------------------------------------------------------
    # Imports
    # ---------------------------------------------------------------------
    import os, sys, itertools, tempfile, subprocess, re, time, math
    from ij import IJ, ImagePlus, ImageStack
    from ij.gui import GenericDialog
    from ij.io import Opener, FileSaver
    from ij.plugin import ChannelSplitter

    from javax.swing import (JPanel, JScrollPane, JList, DefaultListModel,
                             JButton, BoxLayout, JOptionPane, JLabel, JDialog, JProgressBar, Timer,
                             JCheckBox, ImageIcon, BorderFactory, SwingConstants, SwingUtilities)
    from java.awt import (Dimension, Toolkit, BorderLayout, GraphicsEnvironment,
                          GridLayout, FlowLayout, Insets, Font)
    from java.awt.event import ActionListener, ComponentAdapter
    from java.lang import System as JSystem, Runnable

    paths = []

    # ---------------------------------------------------------------------
    # Progress dialog helpers
    # ---------------------------------------------------------------------
    def _fmt_elapsed(sec):
        sec = int(max(0, sec))
        h = sec // 3600; m = (sec % 3600) // 60; s = sec % 60
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

        t = Timer(1000, _Tick()); t.setRepeats(True); t.start()
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
            try: prev.stop()
            except: pass
        steps = max(1, int(duration_ms / 40))
        delta = float(new_value - start) / steps
        state = {"i": 0, "val": float(start)}

        class _Step(ActionListener):
            def actionPerformed(self, evt):
                state["i"] += 1
                if state["i"] >= steps:
                    bar.setValue(new_value); bar.repaint()
                    try: evt.getSource().stop()
                    except: pass
                    bar.putClientProperty("pbTweenTimer", None); return
                state["val"] += delta
                bar.setValue(int(round(state["val"]))); bar.repaint()

        timer = Timer(int(round(float(duration_ms) / steps)), _Step())
        timer.setRepeats(True); timer.start()
        bar.putClientProperty("pbTweenTimer", timer)

    def _pb_cleanup(bar, timer):
        try: timer.stop()
        except: pass
        try:
            t = bar.getClientProperty("pbTweenTimer")
            if t is not None:
                t.stop(); bar.putClientProperty("pbTweenTimer", None)
        except: pass

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
            pths = [os.path.join(env_root, "Library", "bin"),
                    os.path.join(env_root, "Scripts"),
                    env_root]
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
                cands += [os.path.join("C:\\ProgramData","Anaconda3","envs", env_name)]
                try:
                    p = subprocess.Popen(["cmd.exe","/C","conda","info","--base"],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, _ = p.communicate(timeout=1.0)
                    base = out.decode("utf-8","ignore").strip()
                    if base: cands.insert(0, os.path.join(base, "envs", env_name))
                except Exception:
                    pass
            else:
                cands = ["/opt/anaconda3/envs/%s" % env_name]
                cands += [os.path.join(home, d, "envs", env_name)
                          for d in ("mambaforge","miniforge3","miniconda3","anaconda3")]
                try:
                    p = subprocess.Popen(["bash","-lc","conda info --base"],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, _ = p.communicate(timeout=1.0)
                    base = out.decode("utf-8","ignore").strip()
                    if base: cands.insert(0, os.path.join(base, "envs", env_name))
                except Exception:
                    pass
            pref = os.environ.get("CONDA_PREFIX")
            if pref and os.path.basename(pref).lower() == env_name.lower() and os.path.isdir(pref):
                cands.insert(0, pref)
            for c in cands:
                if not c: continue
                py = _python_from_env(c)
                if os.path.isdir(c) and os.path.exists(py):
                    return c
        except Exception:
            pass
        return ""

    # ---------------------------------------------------------------------
    # Constants and helpers
    # ---------------------------------------------------------------------
    DIALOG_W, DIALOG_H = 900, 700
    MARGIN_W, MARGIN_H = 40, 40
    BUTTON_COL_W = 120

    def abort(msg):
        IJ.error(msg); sys.exit()

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
            abort("Could not save the image before running.\n\n%s" % str(e))

    # Keep original file path for naming outputs (even if we later temp-save for processing)
    orig_path_for_naming = ""
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
        tmp = tempfile.NamedTemporaryFile(prefix="debleed_", suffix=".tif", delete=False)
        img_path = tmp.name
        IJ.saveAsTiff(imp, img_path)

    # Use the original file name for outputs if available, otherwise fall back to processing path
    out_base_path = orig_path_for_naming or img_path

    # Channel axis detection
    axis_counts = {"channels": imp.getNChannels(),
                   "slices":   imp.getNSlices(),
                   "frames":   imp.getNFrames()}
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
            texts.append(str(info_prop))

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
                    texts.append(str(val))

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
                    texts.append(str(val))

        stack = imp.getStack()
        for i in range(1, n_ch + 1):
            try:
                idx = slice_idx(i)
                lbl = stack.getSliceLabel(idx)
            except Exception:
                lbl = None
            if lbl:
                texts.append(str(lbl))
        return "\n".join(texts) if texts else ""

    def _metadata_channel_names(imp, axis_used, n_ch):
        texts = _collect_all_metadata_text(imp, n_ch)
        if not texts:
            return None
        pat = re.compile(r"^\s*Name\s*#(\d+)\s*=\s*(.+)$")
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
    raw_labels = [(imp.getStack().getSliceLabel(slice_idx(i)) or "Ch%d" % i).split("\n")[0].strip()
                  for i in range(1, n_ch + 1)]

    def common_pref(lst):
        if not lst: return ""
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
    # Canonical name snapping via groups_by_name
    # ---------------------------------------------------------------------
    def _tok_match(ref, cand):
        pat = r'(?i)(?:^|[^0-9A-Z])' + re.escape(ref) + r'(?:[^0-9A-Z]|$)'
        return re.search(pat, cand) is not None

    def _canonicalize_names_via_rules(names, groups_by_name):
        upper_names = [n.upper() for n in names]
        for idx, ch in enumerate(upper_names):
            best_ref, best_eq = None, -1
            for rule_group in groups_by_name:
                for ref in rule_group:
                    ref_up = ref.upper()
                    if _tok_match(ref_up, ch):
                        eq_flag = 1 if ch.strip() == ref_up.strip() else 0
                        if eq_flag > best_eq:
                            best_eq = eq_flag; best_ref = ref
            if best_ref is not None:
                names[idx] = best_ref
        return names

    names = _canonicalize_names_via_rules(names, groups_by_name)

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
    # Up-front selection dialog (previous interface style), but expanded cell usage
    # ---------------------------------------------------------------------
    def _select_planes_to_debleed(names, imp, axis_used):
        label = _axis_label(axis_used)
        n = len(names)
        if n <= 0:
            return None

        ngrid = int(math.ceil(math.sqrt(float(n))))

        # layout tuning: reduce dead space
        GRID_GAP   = 4
        OUTER_PAD  = 6
        CELL_PAD   = 2

        # resize behavior tuning
        SIZE_STEP  = 8      # quantize thumb dims to reduce regen thrash
        MIN_THUMB  = 12

        dlg = JDialog(None, "Select %ss to Debleed" % label, True)
        dlg.setLayout(BorderLayout())

        # Top message (header only; remove everything underneath)
        top = JPanel()
        top.setLayout(BoxLayout(top, BoxLayout.Y_AXIS))
        msg = ("<html><b>Select which %ss to debleed</b></html>") % (label,)
        top.add(JLabel(msg))
        dlg.add(top, BorderLayout.NORTH)

        # Grid in center
        grid = JPanel(GridLayout(ngrid, ngrid, GRID_GAP, GRID_GAP))
        grid.setBorder(BorderFactory.createEmptyBorder(OUTER_PAD, OUTER_PAD, OUTER_PAD, OUTER_PAD))
        dlg.add(grid, BorderLayout.CENTER)

        # Bottom OK/Cancel
        bottom = JPanel(FlowLayout(FlowLayout.RIGHT))
        btn_ok = JButton("OK")
        btn_cancel = JButton("Cancel")
        try:
            btn_ok.setMargin(Insets(2, 10, 2, 10))
            btn_cancel.setMargin(Insets(2, 10, 2, 10))
        except Exception:
            pass
        bottom.add(btn_ok)
        bottom.add(btn_cancel)
        dlg.add(bottom, BorderLayout.SOUTH)

        # Checkboxes (NONE selected by default), compact single-line-ish label
        cbs = []
        for i, nm in enumerate(names, start=1):
            html = "<html><center><b>%s</b> <span style='font-size:9px;color:gray'>(%d)</span></center></html>" % (nm, i)
            cb = JCheckBox(html, False)
            cb.setHorizontalAlignment(SwingConstants.CENTER)
            cb.setToolTipText("%s %d: %s" % (label, i, nm))
            try:
                cb.setMargin(Insets(0, 0, 0, 0))
            except Exception:
                pass
            cbs.append(cb)

        # Cache icons for current thumb size only
        icons_cache = {}
        state = {"tw": None, "th": None, "font_pt": None}

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
            total_cells = ngrid * ngrid

            for cell_i in range(total_cells):
                cell = JPanel(BorderLayout())
                cell.setBorder(BorderFactory.createEmptyBorder(CELL_PAD, CELL_PAD, CELL_PAD, CELL_PAD))

                if cell_i < n:
                    icon = _get_icon(cell_i, twq, thq)
                    img_lab = JLabel(icon)
                    img_lab.setHorizontalAlignment(SwingConstants.CENTER)
                    img_lab.setVerticalAlignment(SwingConstants.CENTER)
                    cell.add(img_lab, BorderLayout.CENTER)

                    cb = cbs[cell_i]
                    cb.setHorizontalAlignment(SwingConstants.CENTER)
                    cell.add(cb, BorderLayout.SOUTH)
                else:
                    cell.add(JLabel(""), BorderLayout.CENTER)

                grid.add(cell)

            grid.revalidate()
            grid.repaint()

        # OK/Cancel
        result = {"ok": False, "sel": []}

        def on_ok(_):
            sel = [i + 1 for i, cb in enumerate(cbs) if cb.isSelected()]
            if not sel:
                JOptionPane.showMessageDialog(
                    dlg,
                    "Select at least one %s to debleed." % label,
                    "Nothing selected",
                    JOptionPane.WARNING_MESSAGE
                )
                return
            result["ok"] = True
            result["sel"] = sel
            dlg.dispose()

        def on_cancel(_):
            dlg.dispose()

        btn_ok.addActionListener(on_ok)
        btn_cancel.addActionListener(on_cancel)

        # Size to screen
        ge = GraphicsEnvironment.getLocalGraphicsEnvironment()
        gc = ge.getDefaultScreenDevice().getDefaultConfiguration()
        scr = Toolkit.getDefaultToolkit().getScreenSize()
        ins = Toolkit.getDefaultToolkit().getScreenInsets(gc)
        usable_w = scr.width  - ins.left - ins.right
        usable_h = scr.height - ins.top  - ins.bottom
        dlg.setSize(min(1100, max(600, usable_w - 80)), min(850, max(450, usable_h - 120)))
        dlg.setLocationRelativeTo(None)

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

    selected_channels = _select_planes_to_debleed(names, imp, axis_used)
    if selected_channels is None:
        sys.exit()

    selected_set = set([c - 1 for c in selected_channels])  # 0-based indices

    # ---------------------------------------------------------------------
    # Helpers to build/restore groups
    # ---------------------------------------------------------------------
    def _auto_groups_from_rules(names):
        if not groups_by_name:
            return []
        rule_groups = [[] for _ in groups_by_name]
        upper_names = [n.upper() for n in names]
        for idx, ch in enumerate(upper_names):
            for gi, rule_group in enumerate(groups_by_name):
                for ref in rule_group:
                    if _tok_match(ref.upper(), ch):
                        if idx not in rule_groups[gi]:
                            rule_groups[gi].append(idx)
                        break
        return [sorted(g) for g in rule_groups if len(g) >= 2]

    def _load_groups_from_saved_csv(csv_path, names, return_zero_pairs=False):
        if not os.path.exists(csv_path):
            return ([], set()) if return_zero_pairs else []
        try:
            with open(csv_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                return ([], set()) if return_zero_pairs else []
            header = [c.strip() for c in lines[0].split(",")]
            if len(header) < 2:
                return ([], set()) if return_zero_pairs else []
            header_names = header[1:]
            name_to_idx = {nm.upper(): i for i, nm in enumerate(names)}

            col_to_idx = []
            for nm in header_names:
                i = name_to_idx.get(nm.upper(), None)
                if i is None:
                    cand = None
                    for cur_nm, cur_i in name_to_idx.items():
                        if _tok_match(nm.upper(), cur_nm):
                            cand = cur_i; break
                    if cand is None:
                        return ([], set()) if return_zero_pairs else []
                    i = cand
                col_to_idx.append(i)

            zero_pairs = set()
            for rline in lines[1:]:
                cells = [c.strip() for c in rline.split(",")]
                if len(cells) < 2: continue
                row_nm = cells[0]
                row_idx = name_to_idx.get(row_nm.upper(), None)
                if row_idx is None:
                    for cur_nm, cur_i in name_to_idx.items():
                        if _tok_match(row_nm.upper(), cur_nm):
                            row_idx = cur_i; break
                if row_idx is None: continue
                vals = cells[1:]
                for j, v in enumerate(vals):
                    if j >= len(col_to_idx): break
                    col_idx = col_to_idx[j]
                    if row_idx != col_idx and v == "0":
                        zero_pairs.add((row_idx, col_idx))
                        zero_pairs.add((col_idx, row_idx))

            if not zero_pairs:
                return ([], zero_pairs) if return_zero_pairs else []

            neighbors = {i: set() for i in range(len(names))}
            for i, j in zero_pairs:
                neighbors[i].add(j); neighbors[j].add(i)

            sys.setrecursionlimit(10000)
            cliques, seen = [], set()

            def bronk(R, P, X):
                if not P and not X:
                    if len(R) >= 2:
                        key = tuple(sorted(R))
                        if key not in seen:
                            seen.add(key); cliques.append(sorted(R))
                    return
                u = next(iter(P | X)) if (P or X) else None
                Nu = neighbors.get(u, set()) if u is not None else set()
                for v in list(P - Nu):
                    Nv = neighbors.get(v, set())
                    bronk(R | {v}, P & Nv, X & Nv)
                    P.remove(v); X.add(v)

            bronk(set(), set(range(len(names))), set())
            groups = [sorted(g) for g in cliques]
            return (groups, zero_pairs) if return_zero_pairs else groups
        except Exception:
            return ([], set()) if return_zero_pairs else []

    # ---------------------------------------------------------------------
    # Build initial groups (prefer CSV, else rules)
    # - Do NOT prebuild groups that contain none of the selected planes
    # - Add singleton groups for any selected plane not in any group
    # - Preserve old 0-pairs among totally-unselected planes (if CSV existed)
    # ---------------------------------------------------------------------
    groups_csv_path = os.path.splitext(img_path)[0] + ".csv"
    groups, old_zero_pairs = _load_groups_from_saved_csv(groups_csv_path, names, return_zero_pairs=True)

    hidden_zero_pairs = set()
    if groups:
        hidden_zero_pairs = set((i, j) for (i, j) in old_zero_pairs
                                if (i not in selected_set and j not in selected_set))
    else:
        groups = _auto_groups_from_rules(names)
        hidden_zero_pairs = set()

    # Keep only groups that contain at least one selected plane
    groups = [g for g in groups if set(g) & selected_set]

    # Add singleton groups for selected planes not in any group
    in_any_group = set()
    for g in groups:
        in_any_group.update(g)
    for idx in sorted(selected_set):
        if idx not in in_any_group:
            groups.append([idx])

    # ---------------------------------------------------------------------
    # Multi-group UI
    # ---------------------------------------------------------------------
    def _groups_for_channel(idx):
        return ["Group %d" % (gi + 1) for gi, g in enumerate(groups) if idx in g]

    def build_avail_model():
        model = DefaultListModel()
        for i, nm in enumerate(names):
            gs = _groups_for_channel(i)
            debleed_tag = ""
            if i in selected_set:
                debleed_tag = " <span style='font-size:9px;color:gray'>(debleed)</span>"
            if gs:
                html = "<html><b>%s</b>%s<br/><span style='font-size:9px;color:gray'>%s</span></html>" % (nm, debleed_tag, ", ".join(gs))
            else:
                html = "<html><b>%s</b>%s<br/><span style='font-size:9px;color:#b00'>(no groups)</span></html>" % (nm, debleed_tag)
            model.addElement(html)
        return model

    def build_group_model():
        model = DefaultListModel(); meta = []
        for gi, g in enumerate(groups, 1):
            model.addElement("Group %d" % gi); meta.append(("H", gi - 1, None))
            for idx in g:
                model.addElement("   - " + names[idx]); meta.append(("C", gi - 1, idx))
        return model, meta

    avail_model = build_avail_model()
    groups_model, meta = build_group_model()

    avail_list, group_list = JList(avail_model), JList(groups_model)
    for lst in (avail_list, group_list):
        lst.setVisibleRowCount(18)

    btn_new  = JButton("New  ->")
    btn_add  = JButton("Add ->")
    btn_rem  = JButton("<- Remove")

    last_add_group = None  # remembers last group index used by "Add ->"

    def refresh():
        global groups_model, meta, avail_model
        groups_model, meta = build_group_model()
        group_list.setModel(groups_model)
        avail_model = build_avail_model()
        avail_list.setModel(avail_model)

    def on_new(_):
        global last_add_group
        idxs = list(avail_list.getSelectedIndices())
        if idxs:
            groups.append(sorted(set(int(i) for i in idxs)))
            last_add_group = len(groups) - 1
            refresh()

    def on_add(_):
        global last_add_group
        idxs = list(avail_list.getSelectedIndices())
        if not idxs:
            return
        selR = list(group_list.getSelectedIndices())
        if selR:
            typ, gi, _idx = meta[selR[0]]
            target_gi = gi
            if target_gi >= len(groups) or target_gi < 0:
                groups.append([]); target_gi = len(groups) - 1
        else:
            if last_add_group is not None and 0 <= last_add_group < len(groups):
                target_gi = last_add_group
            else:
                groups.append([]); target_gi = len(groups) - 1
        for i in idxs:
            i = int(i)
            if i not in groups[target_gi]:
                groups[target_gi].append(i)
        groups[target_gi] = sorted(groups[target_gi])
        last_add_group = target_gi
        refresh()

    def on_rem(_):
        global last_add_group
        selected = sorted(group_list.getSelectedIndices(), reverse=True)
        any_change = False
        for li in selected:
            typ, gi, idx = meta[li]
            if typ == "C":
                try: groups[gi].remove(idx); any_change = True
                except ValueError: pass
            elif typ == "H":
                try: groups.pop(gi); any_change = True
                except Exception: pass
        for gi in reversed(range(len(groups))):
            if not groups[gi]:
                groups.pop(gi); any_change = True
        if last_add_group is not None and last_add_group >= len(groups):
            last_add_group = None
        if any_change: refresh()

    for b, f in ((btn_new, on_new), (btn_add, on_add), (btn_rem, on_rem)):
        b.addActionListener(f)

    mid = JPanel(); mid.setLayout(BoxLayout(mid, BoxLayout.Y_AXIS))
    for b in (btn_new, btn_add, btn_rem):
        mid.add(b)
    mid.setPreferredSize(Dimension(BUTTON_COL_W, mid.getPreferredSize().height))
    mid.setMaximumSize(Dimension(BUTTON_COL_W, 1000))

    editor = JPanel(); editor.setLayout(BoxLayout(editor, BoxLayout.X_AXIS))
    editor.add(JScrollPane(avail_list)); editor.add(mid); editor.add(JScrollPane(group_list))

    panel = JPanel(BorderLayout())

    panel_title = ("<html><b>Co-expression groups</b> &nbsp;&mdash;&nbsp; "
                   "Put channels that co-express in the same group. "
                   "Channels that share any group are <b>not</b> used to debleed each other. "
                   "Channels may be in <b>multiple</b> groups, and leaving a channel ungrouped is OK "
                   "(it will factor in all other channels). "
                   "<br/><span style='font-size:9px;color:gray'>Note: groups with none of the selected channels were not pre-built.</span></html>")
    panel_tip = ("<html><span style='font-size:9px;color:gray'>Tip: Select a group (or a member of a group) on the right, "
                 "select channel(s) on the left, then click <b>Add -&gt;</b> to add them to that same group.</span></html>")

    north = JPanel(); north.setLayout(BoxLayout(north, BoxLayout.Y_AXIS))
    north.add(JLabel(panel_title))
    north.add(JLabel(panel_tip))
    panel.add(north, BorderLayout.NORTH)
    panel.add(editor, BorderLayout.CENTER)

    # Dock-safe sizing
    ge = GraphicsEnvironment.getLocalGraphicsEnvironment()
    gc = ge.getDefaultScreenDevice().getDefaultConfiguration()
    scr = Toolkit.getDefaultToolkit().getScreenSize()
    ins = Toolkit.getDefaultToolkit().getScreenInsets(gc)
    usable_w = scr.width  - ins.left - ins.right
    usable_h = scr.height - ins.top  - ins.bottom
    DOCK_PAD = 120
    panel.setPreferredSize(Dimension(
        min(DIALOG_W, max(400, usable_w - MARGIN_W)),
        min(DIALOG_H, max(400, usable_h - MARGIN_H - DOCK_PAD))
    ))

    if JOptionPane.showConfirmDialog(None, panel, "Bleed-through groups",
            JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION:
        sys.exit()

    # ---------------------------------------------------------------------
    # CSV (pairs share ANY group => 0, others 1). Persist for future runs.
    # Preserve old 0-pairs among totally-unselected planes if CSV existed.
    # ---------------------------------------------------------------------
    zero = {(i, j) for g in groups for i, j in itertools.permutations(g, 2)}
    zero |= hidden_zero_pairs

    csv = ["," + ",".join(names)]
    for r in range(n_ch):
        row = [names[r]]
        for c in range(n_ch):
            row.append("0" if r == c or (r, c) in zero else "1")
        csv.append(",".join(row))
    with open(groups_csv_path, "w") as f:
        f.write("\n".join(csv))

    # ---------------------------------------------------------------------
    # Run parameters dialog (planes already selected up front)
    # ---------------------------------------------------------------------
    _prefilled_env = _guess_conda_env_root("rfot")

    dlg = GenericDialog("Run")
    dlg.addMessage(
        "Patch size controls the neighborhood used to resolve bleed-through.\n"
        "- Must be an EVEN integer >= 4.\n"
        "- Lower values -> more aggressive removal and faster runs.\n"
        "- Higher values -> gentler correction but slower.\n"
        "\n"
        "Optionally, ignore overexposed pixels by setting saturated pixels to 0.\n"
        "\n"
        "Selected to debleed: %d %ss." % (len(selected_channels), _axis_label(axis_used))
    )
    dlg.addNumericField("Patch size (patsize):", 16, 0)
    dlg.addStringField("Conda env path (root of env, e.g. .../envs/rfot):", _prefilled_env or "", 50)
    dlg.addCheckbox("Ignore overexposed pixels (set saturated to 0)", False)
    dlg.addCheckbox("Keep-the-brightest heuristic (recommended OFF for Opal/Vectra)", True)

    dlg.showDialog()
    if dlg.wasCanceled():
        sys.exit()

    patsize = int(round(dlg.getNextNumber()))
    env_root = dlg.getNextString().strip()
    ignore_overexposed = dlg.getNextBoolean()
    keep_brightest = dlg.getNextBoolean()

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
    # If Opal/Vectra metadata is present and heuristic is ON, warn user
    # ---------------------------------------------------------------------
    meta_all = _collect_all_metadata_text(imp, n_ch) or ""
    if keep_brightest and re.search(r"\b(opal|vectra)\b", meta_all, re.IGNORECASE):
        opts = ["Turn OFF & use signal-based", "Keep ON"]
        choice = JOptionPane.showOptionDialog(
            None,
            ("We detected 'Opal'/'Vectra' in the image metadata.\n\n"
             "The keep-the-brightest heuristic can be too aggressive for Opal/Vectra data.\n\n"
             "What would you like to do for this run?"),
            "Opal/Vectra data detected",
            JOptionPane.DEFAULT_OPTION,
            JOptionPane.WARNING_MESSAGE,
            None, opts, opts[0]
        )
        if choice == 0:
            keep_brightest = False

    # ---------------------------------------------------------------------
    # Launch runner (external Python) based on heuristic toggle
    # ---------------------------------------------------------------------
    def _find_runner_script(keep_brightest):
        plugins_dir = IJ.getDir("plugins")
        base = os.path.join(plugins_dir, "Debleed")
        if keep_brightest:
            cand = [os.path.join(base, "keep_the_brightest.py"),
                    os.path.join(base, "bin", "keep_the_brightest.py")]
        else:
            # No typo protection: ONLY signal_based.py
            cand = [os.path.join(base, "signal_based.py"),
                    os.path.join(base, "bin", "signal_based.py")]
        for p in cand:
            if os.path.exists(p):
                return p
        return None

    runner_py = _find_runner_script(keep_brightest)
    if not runner_py:
        if keep_brightest:
            abort("Could not find 'keep_the_brightest.py'. Put it in:\n"
                  + os.path.join(IJ.getDir('plugins'), "Debleed"))
        else:
            abort("Could not find 'signal_based.py'. Put it in:\n"
                  + os.path.join(IJ.getDir('plugins'), "Debleed"))

    channels = sorted(set(int(c) for c in selected_channels))
    if not channels:
        abort("No channels selected to debleed.")

    wait_dlg, wait_bar, wait_timer = _show_progress(len(channels))
    if len(channels) > 1:
        wait_bar.setIndeterminate(False)
        wait_bar.setMinimum(0); wait_bar.setMaximum(len(channels))
        wait_bar.setValue(0); wait_bar.setStringPainted(True)
        wait_bar.setString("0 / %d (0%%)" % len(channels))

    try:
        for i, ch in enumerate(channels, start=1):
            IJ.showStatus("Processing channel %d of %d" % (i, len(channels)))
            if len(channels) > 1:
                _pb_smooth_to(wait_bar, i)
                pct = int(round(100.0 * i / float(len(channels))))
                wait_bar.setString("%d / %d (%d%%)" % (i, len(channels), pct))
                wait_bar.repaint()

            cmd = [pyexe, runner_py, img_path, str(ch), "--patsize", str(patsize)]
            if ignore_overexposed:
                cmd.append("--ignore_overexposed")

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=subproc_env)
            stdout, stderr = proc.communicate()

            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", "replace")
                abort("Runner failed for channel %d (exit %d).\n\n%s" % (ch, proc.returncode, err_msg))

            out = "%s_Channel_%d_debleed.tif" % (img_path[:-4], ch)
            if not os.path.exists(out):
                abort("Result not found for channel %d:\n%s" % (ch, out))
            paths.append(out)
    finally:
        _pb_cleanup(wait_bar, wait_timer)
        wait_dlg.dispose()
        IJ.showProgress(1.0)
        IJ.showStatus("Finished.")

    # ---------------------------------------------------------------------
    # Combine outputs into a hyperstack and attach labels with group names
    # ---------------------------------------------------------------------
    if not paths:
        abort("No output files generated.")

    opener = Opener()
    imps = [opener.openImage(p) for p in paths]

    def _channel_label_with_groups(zero_based_idx):
        nm = names[zero_based_idx]
        gs = _groups_for_channel(zero_based_idx)
        return nm if not gs else (nm + "\n" + ", ".join(gs))

    if len(imps) == 1:
        imp_single = imps[0]
        stk_single = imp_single.getStack()
        ch0 = channels[0] - 1
        stk_single.setSliceLabel(_channel_label_with_groups(ch0), 1)
        imp_single.updateAndDraw(); imp_single.show()
    else:
        w, h = imps[0].getWidth(), imps[0].getHeight()
        stack = ImageStack(w, h)
        for imp_ in imps:
            stack.addSlice(imp_.getProcessor())
        result = ImagePlus("Processed combined", stack)
        result.setDimensions(len(imps), 1, 1)
        result.setOpenAsHyperStack(True)
        stk = result.getStack()
        for c_idx, ch_num in enumerate(channels, start=1):
            stk.setSliceLabel(_channel_label_with_groups(ch_num - 1), c_idx)
        result.updateAndDraw(); result.show()

    # ---------------------------------------------------------------------
    # Build + save FULL stack with debleeded planes replaced (others unchanged)
    # Saved as: <original_stack_name>_debleed.tif
    # ---------------------------------------------------------------------
    try:
        # Map original stack slice-index -> debleeded ImageProcessor
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
            base_title = "Full debleed"
        full_imp = ImagePlus(base_title + " (debleed)", full_stack)

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

        out_full = os.path.splitext(out_base_path)[0] + "_debleed.tif"

        fs = FileSaver(full_imp)
        if full_imp.getStackSize() > 1:
            ok = fs.saveAsTiffStack(out_full)
        else:
            ok = fs.saveAsTiff(out_full)

        if not ok:
            abort("Could not save full debleed stack:\n%s" % out_full)

        IJ.showStatus("Saved full debleed stack: " + out_full)

    except Exception as e:
        abort("Failed building/saving the full debleed stack.\n\n%s" % str(e))

except SystemExit:
    # IMPORTANT:
    # - Fiji/SciJava prints uncaught SystemExit as a scary ERROR traceback.
    # - This swallow makes *Cancel / early exit* look clean in the console,
    #   regardless of which dialog/menu the user cancels from.
    pass
