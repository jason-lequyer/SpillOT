# Group rules (used only for initial auto-assignment)
groups_by_name = [
    # The big immune / structural set
    ["CA2", "CD116", "CD14", "CD20", "CD3", "CD4", "CD44", "CD45",
     "CD45RO", "CD56", "CD57", "CD68", "CD8",
     "DNA1", "DNA2",
     "FOXP3", "Granzyme B", "Ki67", "NF-KB",
     "NKX6.1", "PDX-1", "pan-Keratin"],

    # Vascular / stem-cell pair
    ["NESTIN", "CD31"],

    # Singletons
    ["HLA-ABC"],
    ["C-PEPTIDE"],
    ["GLUCAGON"],
    ["SOMATOSTATIN"],
    ["B-ACTIN"],
    ["Collagen type I"],
    ["pS6"],
    ["HLA-DR"],
    ["PANCREATIC POLYPEPTIDE"],
    ["GHRELIN"]
]

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import os, sys, itertools, tempfile, subprocess, re, time
from ij import IJ, ImagePlus, ImageStack
from ij.gui import GenericDialog
from ij.io import Opener
from ij.plugin import ChannelSplitter
from javax.swing import (JPanel, JScrollPane, JList, DefaultListModel,
                         JButton, BoxLayout, JOptionPane, JLabel, JDialog, JProgressBar, Timer)
from java.awt import (Dimension, Toolkit, BorderLayout, GraphicsEnvironment)
from java.awt.event import ActionListener
from java.lang import System as JSystem

paths = []

# ---------------------------------------------------------------------
# Progress dialog helpers
# ---------------------------------------------------------------------
def _fmt_elapsed(sec):
    sec = int(max(0, sec))
    h = sec // 3600; m = (sec % 3600) // 60; s = sec % 60
    return ("%02d:%02d:%02d" % (h, m, s)) if h else ("%02d:%02d" % (m, s))

def _show_progress(total_channels):
    """Return (dialog, progress_bar, timer). Caller must stop timer and dispose dialog."""
    start_ts = time.time()

    dlg = JDialog(None, "Debleeding", False)  # modeless
    dlg.setLayout(BorderLayout())

    lbl_txt = "Debleeding channel" if total_channels == 1 else "Debleeding %d channels" % total_channels
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
    """Animate bar value to new_value over duration_ms using a short Swing Timer."""
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

    steps = max(1, int(duration_ms / 40))  # ~25 FPS
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
    """Try to find a conda/mamba env called 'rfot' and return its root."""
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
                if base:
                    cands.insert(0, os.path.join(base, "envs", env_name))
            except Exception:
                pass
        else:
            # macOS requested path first
            cands = ["/opt/anaconda3/envs/%s" % env_name]
            # common local installs under $HOME
            cands += [os.path.join(home, d, "envs", env_name)
                      for d in ("mambaforge","miniforge3","miniconda3","anaconda3")]
            try:
                p = subprocess.Popen(["bash","-lc","conda info --base"],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, _ = p.communicate(timeout=1.0)
                base = out.decode("utf-8","ignore").strip()
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

force_temp_save = False

# If RGB-color composite with single channel, split to R/G/B stacks
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

# Ensure TIFF path
fi = imp.getOriginalFileInfo()
if (not force_temp_save) and fi and fi.directory and fi.fileName:
    img_path = os.path.join(fi.directory, fi.fileName)
else:
    tmp = tempfile.NamedTemporaryFile(prefix="debleed_", suffix=".tif", delete=False)
    img_path = tmp.name
    IJ.saveAsTiff(imp, img_path)

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

# Cleaned-up channel names
raw = [(imp.getStack().getSliceLabel(slice_idx(i)) or "Ch%d" % i).split("\n")[0].strip()
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

pre, suf = common_pref(raw), common_suff(raw)
names = [ (s[len(pre):len(s)-len(suf)] or s) if suf else (s[len(pre):] or s) for s in raw ]

# ---------------------------------------------------------------------
# Build initial groups from rules (same logic as before)
# ---------------------------------------------------------------------
groups = []
assigned = set()

def tok_match(ref, cand):
    pat = r'(?i)(?:^|[^0-9A-Z])' + re.escape(ref) + r'(?:[^0-9A-Z]|$)'
    return re.search(pat, cand) is not None

compiled = [(ref.upper(), gi, len(ref)) for gi, g in enumerate(groups_by_name) for ref in g]
for idx, ch in enumerate([n.upper() for n in names]):
    best = None
    for ref, gi, rl in compiled:
        if tok_match(ref, ch) and (best is None or rl > best[1]):
            best = (gi, rl)
    if best:
        gi = best[0]
        while len(groups) <= gi:
            groups.append([])
        groups[gi].append(idx)
        assigned.add(idx)

for idx in range(n_ch):
    if idx not in assigned:
        groups.append([idx])

groups = [sorted(g) for g in groups if g]

# ---------------------------------------------------------------------
# Multi-group UI: left list shows all channels + their current groups
# ---------------------------------------------------------------------
def _groups_for_channel(idx):
    return ["Group %d" % (gi + 1) for gi, g in enumerate(groups) if idx in g]

def build_avail_model():
    model = DefaultListModel()
    for i, nm in enumerate(names):
        gs = _groups_for_channel(i)
        if gs:
            html = "<html><b>%s</b><br/><span style='font-size:9px;color:gray'>%s</span></html>" % (
                nm, ", ".join(gs))
        else:
            html = "<html><b>%s</b><br/><span style='font-size:9px;color:#b00'>(no groups)</span></html>" % nm
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

def refresh():
    """Refresh both lists so memberships and numbering stay in sync."""
    global groups_model, meta, avail_model
    groups_model, meta = build_group_model()
    group_list.setModel(groups_model)
    avail_model = build_avail_model()
    avail_list.setModel(avail_model)

def on_new(_):
    # Create a new group from selected channels on the left (multi-select)
    idxs = list(avail_list.getSelectedIndices())
    if idxs:
        groups.append(sorted(set(int(i) for i in idxs)))
        refresh()

def on_add(_):
    # Add selected channels (left) to the selected group header (right); or create a new group.
    idxs = list(avail_list.getSelectedIndices())
    if not idxs: return
    selG = group_list.getSelectedIndex()
    gi = meta[selG][1] if selG >= 0 and meta[selG][0] == "H" else len(groups)
    if gi == len(groups):
        groups.append([])
    for i in idxs:
        i = int(i)
        if i not in groups[gi]:
            groups[gi].append(i)
    groups[gi] = sorted(groups[gi])
    refresh()

def on_rem(_):
    # Remove selected channel entries from their respective groups. Drop empty groups.
    selected = sorted(group_list.getSelectedIndices(), reverse=True)
    any_change = False
    for li in selected:
        typ, gi, idx = meta[li]
        if typ == "C":
            try:
                groups[gi].remove(idx)
                any_change = True
            except ValueError:
                pass
    for gi in reversed(range(len(groups))):
        if not groups[gi]:
            groups.pop(gi)
            any_change = True
    if any_change:
        refresh()

def _parse_channels(spec, max_ch):
    # Parse "1,3-5" -> [1,3,4,5] (1-based)
    out = []
    for token in re.split(r"[,\s]+", spec):
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(token))
    return sorted({c for c in out if 1 <= c <= max_ch})

for b, f in ((btn_new, on_new), (btn_add, on_add), (btn_rem, on_rem)):
    b.addActionListener(f)

mid = JPanel()
mid.setLayout(BoxLayout(mid, BoxLayout.Y_AXIS))
for b in (btn_new, btn_add, btn_rem):
    mid.add(b)
mid.setPreferredSize(Dimension(BUTTON_COL_W, mid.getPreferredSize().height))
mid.setMaximumSize(Dimension(BUTTON_COL_W, 1000))

editor = JPanel()
editor.setLayout(BoxLayout(editor, BoxLayout.X_AXIS))
editor.add(JScrollPane(avail_list))
editor.add(mid)
editor.add(JScrollPane(group_list))

panel = JPanel(BorderLayout())

# ASCII-only UI text; &mdash; to avoid stray characters on some setups
panel_title = ("<html><b>Co-localizing groups</b> &nbsp;&mdash;&nbsp; "
               "Channels can be in <b>multiple</b> groups. The left list shows each channel and its current memberships."
               "</html>")
panel.add(JLabel(panel_title), BorderLayout.NORTH)
panel.add(editor, BorderLayout.CENTER)

# Dock-safe sizing: respect screen insets and add bottom padding to avoid macOS Dock
ge = GraphicsEnvironment.getLocalGraphicsEnvironment()
gc = ge.getDefaultScreenDevice().getDefaultConfiguration()
scr = Toolkit.getDefaultToolkit().getScreenSize()
ins = Toolkit.getDefaultToolkit().getScreenInsets(gc)

usable_w = scr.width  - ins.left - ins.right
usable_h = scr.height - ins.top  - ins.bottom

DOCK_PAD = 120  # extra clearance above Dock/taskbar
panel.setPreferredSize(Dimension(
    min(DIALOG_W, max(400, usable_w - MARGIN_W)),
    min(DIALOG_H, max(400, usable_h - MARGIN_H - DOCK_PAD))
))

if JOptionPane.showConfirmDialog(None, panel, "Bleed-through groups",
        JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION:
    sys.exit()

# ---------------------------------------------------------------------
# CSV (pairs share ANY group => 0)
# ---------------------------------------------------------------------
zero = {(i, j) for g in groups for i, j in itertools.permutations(g, 2)}
csv = ["," + ",".join(names)]
for r in range(n_ch):
    row = [names[r]]
    for c in range(n_ch):
        row.append("0" if r == c or (r, c) in zero else "1")
    csv.append(",".join(row))
with open(os.path.splitext(img_path)[0] + ".csv", "w") as f:
    f.write("\n".join(csv))

# ---------------------------------------------------------------------
# Run parameters dialog
# ---------------------------------------------------------------------
_prefilled_env = _guess_conda_env_root("rfot")

dlg = GenericDialog("Run Debleed")
dlg.addMessage(
    "Patch size controls the neighborhood used to resolve bleed-through.\n"
    "- Must be an EVEN integer >= 4.\n"
    "- Lower values -> more aggressive removal and faster runs.\n"
    "- Higher values -> gentler correction but slower.\n"
    "\n"
    "Optionally, ignore overexposed pixels by setting saturated pixels to 0."
)
dlg.addStringField("Channel(s) to debleed (e.g. 1,3-5):", "1")
dlg.addNumericField("Patch size (patsize):", 16, 0)
dlg.addStringField("Conda env path (root of env, e.g. .../envs/rfot):", _prefilled_env or "", 50)
dlg.addCheckbox("Ignore overexposed pixels (set saturated to 0)", False)

dlg.showDialog()
if dlg.wasCanceled():
    sys.exit()

chan_spec = dlg.getNextString().strip()
patsize = int(round(dlg.getNextNumber()))
env_root = dlg.getNextString().strip()
ignore_overexposed = dlg.getNextBoolean()

# Validate patsize
if dlg.invalidNumber() or patsize < 4 or (patsize % 2 != 0):
    IJ.showMessage("Invalid patch size",
                   "Patch size must be an EVEN integer >= 4.\nYou entered: %s." % patsize)
    sys.exit()

# Resolve env root
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
# Launch Debleed (external Python)
# ---------------------------------------------------------------------
plugins_dir = IJ.getDir("plugins")
debleed_py_candidates = [
    os.path.join(plugins_dir, "Debleed", "debleed.py"),
    os.path.join(plugins_dir, "Debleed", "bin", "debleed.py"),
]
debleed_py = None
for p in debleed_py_candidates:
    if os.path.exists(p):
        debleed_py = p
        break
if not debleed_py:
    abort("Could not find 'debleed.py'. Put it in:\n" + "\n".join(debleed_py_candidates))

channels = _parse_channels(chan_spec, n_ch)
if not channels:
    abort("No valid channels parsed from: '%s'" % chan_spec)

wait_dlg, wait_bar, wait_timer = _show_progress(len(channels))
if len(channels) > 1:
    wait_bar.setIndeterminate(False)
    wait_bar.setMinimum(0); wait_bar.setMaximum(len(channels))
    wait_bar.setValue(0); wait_bar.setStringPainted(True)
    wait_bar.setString("0 / %d" % len(channels))

try:
    for i, ch in enumerate(channels, start=1):
        IJ.showStatus("Debleeding channel %d of %d" % (i, len(channels)))
        if len(channels) > 1:
            _pb_smooth_to(wait_bar, i - 1)

        cmd = [pyexe, debleed_py, img_path, str(ch), "--patsize", str(patsize)]
        if ignore_overexposed:
            cmd.append("--ignore_overexposed")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=subproc_env)
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", "replace")
            abort("Debleed failed for channel %d (exit %d).\n\n%s" % (ch, proc.returncode, err_msg))

        out = "%s_Channel_%d_debleed.tif" % (img_path[:-4], ch)
        if not os.path.exists(out):
            abort("Result not found for channel %d:\n%s" % (ch, out))
        paths.append(out)

        if len(channels) > 1:
            _pb_smooth_to(wait_bar, i)
finally:
    _pb_cleanup(wait_bar, wait_timer)
    wait_dlg.dispose()
    IJ.showProgress(1.0)
    IJ.showStatus("Debleed finished.")

# ---------------------------------------------------------------------
# Combine outputs into a hyperstack and attach labels with group names
# ---------------------------------------------------------------------
if not paths:
    abort("No output files generated.")

opener = Opener()
imps = [opener.openImage(p) for p in paths]  # one ImagePlus per channel

def _channel_label_with_groups(zero_based_idx):
    nm = names[zero_based_idx]
    gs = _groups_for_channel(zero_based_idx)
    return nm if not gs else (nm + "\n" + ", ".join(gs))

if len(imps) == 1:
    imp_single = imps[0]
    stk_single = imp_single.getStack()
    ch0 = channels[0] - 1
    stk_single.setSliceLabel(_channel_label_with_groups(ch0), 1)
    imp_single.updateAndDraw()
    imp_single.show()
else:
    w, h = imps[0].getWidth(), imps[0].getHeight()
    stack = ImageStack(w, h)
    for imp_ in imps:
        stack.addSlice(imp_.getProcessor())
    result = ImagePlus("Debleed combined", stack)
    result.setDimensions(len(imps), 1, 1)  # C, Z, T
    result.setOpenAsHyperStack(True)
    stk = result.getStack()
    for c_idx, ch_num in enumerate(channels, start=1):  # 1-based
        stk.setSliceLabel(_channel_label_with_groups(ch_num - 1), c_idx)
    result.updateAndDraw()
    result.show()
# ---------------------------------------------------------------------
