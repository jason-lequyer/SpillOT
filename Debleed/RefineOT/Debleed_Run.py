groups_by_name = [
    # The big immune / structural set
    ["CA2", "CD116", "CD14", "CD20", "CD3", "CD4", "CD44", "CD45",
     "CD45RO", "CD56", "CD57", "CD68", "CD8",
     "DNA1", "DNA2",
     "FOXP3", "Granzyme B", "Ki67", "NF-KB",
     "NKX6.1", "PDX-1", "pan-Keratin"],

    # Vascular / stem‑cell pair
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


from ij import IJ
# ─── progress dialog helper with elapsed time ─────────────────────────
from javax.swing import JDialog, JLabel, JProgressBar, SwingUtilities, Timer
from java.awt.event import ActionListener
from java.awt import BorderLayout
import time

# ─── progress dialog helper with smooth updates ───────────────────────
from javax.swing import JDialog, JLabel, JProgressBar, Timer
from java.awt.event import ActionListener
from java.awt import BorderLayout
import time

def _fmt_elapsed(sec):
    sec = int(max(0, sec))
    h = sec // 3600; m = (sec % 3600) // 60; s = sec % 60
    return ("%02d:%02d:%02d" % (h, m, s)) if h else ("%02d:%02d" % (m, s))

def _show_progress(total_channels):
    """Return (dialog, progress_bar, timer). Caller must stop timer and dispose dialog."""
    start_ts = time.time()

    dlg = JDialog(None, "Debleeding", False)  # modeless
    dlg.setLayout(BorderLayout())

    # top label
    lbl_txt = "Debleeding channel" if total_channels == 1 else "Debleeding {} channels".format(total_channels)
    dlg.add(JLabel(lbl_txt), BorderLayout.NORTH)

    # progress bar (no overlay text)
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

    # elapsed time label below
    elapsed_lbl = JLabel("Elapsed: 00:00")
    dlg.add(elapsed_lbl, BorderLayout.SOUTH)

    dlg.setSize(300, 100)
    dlg.setLocationRelativeTo(None)
    dlg.setVisible(True)

    # timer to update elapsed label once per second (fires on EDT)
    class _Tick(ActionListener):
        def actionPerformed(self, evt):
            elapsed_lbl.setText("Elapsed: " + _fmt_elapsed(time.time() - start_ts))
            elapsed_lbl.repaint()

    t = Timer(1000, _Tick())
    t.setRepeats(True)
    t.start()

    return dlg, bar, t

# ─── smooth tween helper for the progress bar ─────────────────────────
def _pb_smooth_to(bar, new_value, duration_ms=250):
    """
    Animate bar value to new_value over duration_ms using a short Swing Timer.
    Cancels any previous tween on the same bar (stored on the bar itself).
    """
    if bar.isIndeterminate():
        return  # nothing to animate in indeterminate mode

    new_value = max(bar.getMinimum(), min(bar.getMaximum(), int(new_value)))
    start = bar.getValue()
    if new_value == start:
        return

    # stop any existing tween timer stored on the bar
    prev = bar.getClientProperty("pbTweenTimer")
    if prev is not None:
        try: prev.stop()
        except: pass

    steps = max(1, int(duration_ms / 40))  # ~25 FPS
    delta = float(new_value - start) / steps
    state = {"i": 0, "val": float(start)}

    class _Step(ActionListener):
        def actionPerformed(self, evt):
            state["i"] += 1
            if state["i"] >= steps:
                bar.setValue(new_value)
                bar.repaint()
                try: evt.getSource().stop()
                except: pass
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
    """Stop elapsed timer and any active tween timer."""
    try: timer.stop()
    except: pass
    try:
        t = bar.getClientProperty("pbTweenTimer")
        if t is not None:
            t.stop()
            bar.putClientProperty("pbTweenTimer", None)
    except: pass

# 0 ▸ INSERT your master list here, e.g.
#
# groups_by_name = [
#     ["CA2", "CD3", "CD4"],
#     ["CD31", "NESTIN"],
#     ...
# ]
# ----------------------------------------------------------------------

import os, sys, itertools, tempfile, subprocess, re
from ij import IJ
from ij.gui import GenericDialog
from javax.swing import (JPanel, JScrollPane, JList, DefaultListModel,
                         JButton, BoxLayout, JOptionPane, JLabel)
from java.awt import Dimension, Toolkit, BorderLayout
from ij import ImagePlus, ImageStack          # ← NEW
from ij.io import Opener                      # ← NEW
from ij.plugin import ChannelSplitter
paths = []                                    # ← NEW

# ─── Conda env helpers ─────────────────────────────────────────────────────────
import os, subprocess
from java.lang import System as JSystem

def _is_windows():
    return "windows" in (JSystem.getProperty("os.name") or "").lower()

def _python_from_env(env_root):
    # absolute path to the env's python
    return os.path.join(env_root, "python.exe") if _is_windows() else os.path.join(env_root, "bin", "python")

def _subproc_env_for_conda_env(env_root):
    # Build a clean env so DLLs are found without 'conda activate'
    env = os.environ.copy()
    if _is_windows():
        # add env\Library\bin and env\Scripts on PATH
        paths = [os.path.join(env_root, "Library", "bin"),
                 os.path.join(env_root, "Scripts"),
                 env_root]
        env["PATH"] = os.pathsep.join(paths + [env.get("PATH", "")])
    else:
        env["PATH"] = os.pathsep.join([os.path.join(env_root, "bin"), env.get("PATH", "")])
    # keep BLAS threads sensible
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env

def _guess_conda_env_root(env_name="rfot"):
    """
    Try hard to find a conda/mamba env called 'rfot' and return its root.
    We don't require 'conda' on PATH.
    """
    try:
        home = os.path.expanduser("~")
        cands = []
        if _is_windows():
            user = os.environ.get("USERPROFILE", home)
            cands += [os.path.join(user, d, "envs", env_name)
                      for d in ("anaconda3","miniconda3","miniforge3","mambaforge","Anaconda3","Miniconda3")]
            cands += [os.path.join("C:\\ProgramData","Anaconda3","envs", env_name)]
            # If conda is visible, use its base to build a candidate
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
            cands += [os.path.join(home, d, "envs", env_name)
                      for d in ("mambaforge","miniforge3","miniconda3","anaconda3")]
            try:
                # If conda exists, ask for its base
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
# ───────────────────────────────────────────────────────────────────────────────


DIALOG_W, DIALOG_H = 900, 700
MARGIN_W, MARGIN_H = 40, 40
BUTTON_COL_W = 120

def abort(msg):
    IJ.error(msg); sys.exit()

# 1 ▸ active image
try: imp = IJ.getImage()
except Exception: abort("No image is open.")




force_temp_save = False  # will force writing a temp TIFF if we transform the image

is_rgb_color = (imp.getType() == ImagePlus.COLOR_RGB and imp.getNChannels() == 1)
if is_rgb_color:
    # Split into R,G,B grayscale stacks
    splits = ChannelSplitter.split(imp)  # [R,G,B], each preserves Z/T if present
    w, h = splits[0].getWidth(), splits[0].getHeight()
    Z = max(splits[0].getNSlices(), 1)
    T = max(splits[0].getNFrames(), 1)
    labels = ["Red", "Green", "Blue"]

    stack = ImageStack(w, h)
    # Add slices in ImageJ hyperstack order: c fastest, then z, then t
    for t in range(1, T + 1):
        for z in range(1, Z + 1):
            for ci, lab in enumerate(labels, start=1):   # ci: 1..3
                src = splits[ci - 1]
                idx = src.getStackIndex(1, z, t)         # src has 1 channel
                stack.addSlice(lab, src.getStack().getProcessor(idx))

    imp = ImagePlus(imp.getTitle() + " (RGB split)", stack)
    imp.setDimensions(3, Z, T)          # C=3, preserve Z/T
    imp.setOpenAsHyperStack(True)
    force_temp_save = True              # ensure we save this transformed image


# 2 ▸ ensure TIFF path
fi = imp.getOriginalFileInfo()
if (not force_temp_save) and fi and fi.directory and fi.fileName:
    img_path = os.path.join(fi.directory, fi.fileName)
else:
    tmp = tempfile.NamedTemporaryFile(prefix="debleed_", suffix=".tif", delete=False)
    img_path = tmp.name
    IJ.saveAsTiff(imp, img_path)


# 3 ▸ channel axis detection
axis_counts = {"channels": imp.getNChannels(),
               "slices":   imp.getNSlices(),
               "frames":   imp.getNFrames()}
axis_used, n_ch = max(axis_counts.items(), key=lambda kv: kv[1])
def slice_idx(c): return (imp.getStackIndex(c,1,1) if axis_used=="channels"
                          else imp.getStackIndex(1,c,1) if axis_used=="slices"
                          else imp.getStackIndex(1,1,c))

# 4 ▸ cleaned‑up channel names
raw=[(imp.getStack().getSliceLabel(slice_idx(i)) or "Ch{}".format(i)).split("\n")[0].strip()
     for i in range(1,n_ch+1)]
def common_pref(lst):
    if not lst: return ""
    s1,s2=min(lst),max(lst); i=0
    while i<len(s1) and s1[i]==s2[i]: i+=1
    return s1[:i]
def common_suff(lst): return common_pref([s[::-1] for s in lst])[::-1]
pre,suf=common_pref(raw),common_suff(raw)
names=[(s[len(pre):len(s)-len(suf)] or s) if suf else s[len(pre):] or s for s in raw]

# 5 ▸ build initial groups from rules
groups=[]; assigned=set()
def tok_match(ref, cand):
    pat=r'(?i)(?:^|[^0-9A-Z])'+re.escape(ref)+r'(?:[^0-9A-Z]|$)'
    return re.search(pat, cand) is not None
compiled=[(ref.upper(),gi,len(ref)) for gi,g in enumerate(groups_by_name) for ref in g]
for idx,ch in enumerate([n.upper() for n in names]):
    best=None
    for ref,gi,rl in compiled:
        if tok_match(ref,ch) and (best is None or rl>best[1]):
            best=(gi,rl)
    if best:
        gi=best[0]
        while len(groups)<=gi: groups.append([])
        groups[gi].append(idx); assigned.add(idx)
for idx in range(n_ch):
    if idx not in assigned:
        groups.append([idx])
groups=[sorted(g) for g in groups if g]

# ─────────────────────  FIX: Available list starts empty  ──────────────
all_in_groups = set(idx for g in groups for idx in g)
avail_model = DefaultListModel()
for i, nm in enumerate(names):
    if i not in all_in_groups:       # none at launch
        avail_model.addElement(nm)
# -----------------------------------------------------------------------

def build_group_model():
    model=DefaultListModel(); meta=[]
    for gi,g in enumerate(groups,1):
        model.addElement("Group {}".format(gi)); meta.append(("H",gi-1,None))
        for idx in g:
            model.addElement("   - "+names[idx]); meta.append(("C",gi-1,idx))
    return model, meta
groups_model, meta = build_group_model()

avail_list, group_list = JList(avail_model), JList(groups_model)
for lst in (avail_list, group_list): lst.setVisibleRowCount(18)

btn_new,btn_add,btn_rem=JButton("New  ->"),JButton("Add ->"),JButton("<- Remove")
def refresh():
    global groups_model, meta
    groups_model, meta = build_group_model()
    group_list.setModel(groups_model)
def on_new(_):
    sel=avail_list.getSelectedValuesList()
    if sel:
        groups.append([names.index(s) for s in sel])
        [avail_model.removeElement(s) for s in sel]
        refresh()
def on_add(_):
    sel=avail_list.getSelectedValuesList()
    if not sel: return
    selG=group_list.getSelectedIndex()
    gi = meta[selG][1] if selG>=0 and meta[selG][0]=="H" else len(groups)
    if gi==len(groups): groups.append([])
    for s in sel:
        idx=names.index(s)
        if idx not in groups[gi]:
            groups[gi].append(idx); avail_model.removeElement(s)
    refresh()
def on_rem(_):
    for li in sorted(group_list.getSelectedIndices(), reverse=True):
        typ,gi,idx=meta[li]
        if typ=="C":
            groups[gi].remove(idx); avail_model.addElement(names[idx])
            if not groups[gi]: groups.pop(gi)
    refresh()

# helper ─ parse “1,3‑5” → [1,3,4,5]  (1‑based, deduplicated, sorted)
def _parse_channels(spec, max_ch):
    import re
    out = []
    for token in re.split(r"[,\s]+", spec):
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(token))
    # keep only valid channels and drop duplicates
    return sorted({c for c in out if 1 <= c <= max_ch})

for b,f in((btn_new,on_new),(btn_add,on_add),(btn_rem,on_rem)): b.addActionListener(f)

mid=JPanel(); mid.setLayout(BoxLayout(mid,BoxLayout.Y_AXIS))
for b in (btn_new,btn_add,btn_rem): mid.add(b)
mid.setPreferredSize(Dimension(BUTTON_COL_W, mid.getPreferredSize().height))
mid.setMaximumSize(Dimension(BUTTON_COL_W, 1000))

editor=JPanel(); editor.setLayout(BoxLayout(editor,BoxLayout.X_AXIS))
editor.add(JScrollPane(avail_list)); editor.add(mid); editor.add(JScrollPane(group_list))

panel=JPanel(BorderLayout())
panel.add(JLabel("<html><b>Co-localising groups</b> - edit if needed</html>"),
          BorderLayout.NORTH); panel.add(editor,BorderLayout.CENTER)



scr=Toolkit.getDefaultToolkit().getScreenSize()
panel.setPreferredSize(Dimension(min(DIALOG_W,scr.width-MARGIN_W),
                                 min(DIALOG_H,scr.height-MARGIN_H)))

if JOptionPane.showConfirmDialog(None,panel,"Bleed-through groups",
        JOptionPane.OK_CANCEL_OPTION,JOptionPane.PLAIN_MESSAGE)!=JOptionPane.OK_OPTION:
    sys.exit()

# 7 ▸ CSV
zero={(i,j) for g in groups for i,j in itertools.permutations(g,2)}
csv=["," + ",".join(names)]
for r in range(n_ch):
    csv.append(",".join([names[r]] +
              ["0" if r==c or (r,c) in zero else "1" for c in range(n_ch)]))
with open(os.path.splitext(img_path)[0]+".csv","w") as f: f.write("\n".join(csv))





# Try to prefill env path by locating an env named 'rfot'
_prefilled_env = _guess_conda_env_root("rfot")

dlg = GenericDialog("Run Debleed")
dlg.addMessage(
    "Patch size controls the neighborhood used to resolve bleed-through.\n"
    "- Must be an EVEN integer >= 4.\n"
    "- Lower values -> more aggressive removal and faster runs.\n"
    "- Higher values -> gentler correction but slower."
)
dlg.addStringField("Channel(s) to debleed (e.g. 1,3-5):", "1")
dlg.addNumericField("Patch size (patsize):", 16, 0)
dlg.addStringField("Conda env path (root of env, e.g. .../envs/rfot):", _prefilled_env or "", 50)

dlg.showDialog()
if dlg.wasCanceled():
    sys.exit()

chan_spec = dlg.getNextString().strip()
patsize = int(round(dlg.getNextNumber()))
env_root = dlg.getNextString().strip()

# Validate patsize
if dlg.invalidNumber() or patsize < 4 or (patsize % 2 != 0):
    IJ.showMessage("Invalid patch size",
                   "Patch size must be an EVEN integer >= 4.\nYou entered: {}.".format(patsize))
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
                   "  macOS:    /Users/<you>/miniforge3/envs/rfot")
    sys.exit()

pyexe = _python_from_env(env_root)
if not os.path.exists(pyexe):
    IJ.showMessage("Python not found",
                   "Could not find Python inside the env:\n{}\nExpected at:\n{}".format(env_root, pyexe))
    sys.exit()

subproc_env = _subproc_env_for_conda_env(env_root)



# ──────────────────────────────────────────────────────────────────────
# 9 ▸ launch Debleed via platform‑specific executable only
# ----------------------------------------------------------------------

# Where is debleed.py?
plugins_dir = IJ.getDir("plugins")
debleed_py_candidates = [
    os.path.join(plugins_dir, "Debleed", "debleed.py"),
    os.path.join(plugins_dir, "Debleed", "bin", "debleed.py"),
]
debleed_py = None
for p in debleed_py_candidates:
    if os.path.exists(p):
        debleed_py = p; break
if not debleed_py:
    abort("Could not find 'debleed.py'. Put it in:\n" +
          "\n".join(debleed_py_candidates))

channels = _parse_channels(chan_spec, n_ch)
if not channels:
    abort("No valid channels parsed from: '{}'".format(chan_spec))

# progress dialog (unchanged)
wait_dlg, wait_bar, wait_timer = _show_progress(len(channels))

if len(channels) > 1:
    # if you added the EDT helpers earlier, you can call _pb_init here
    wait_bar.setIndeterminate(False)
    wait_bar.setMinimum(0); wait_bar.setMaximum(len(channels))
    wait_bar.setValue(0); wait_bar.setStringPainted(True)
    wait_bar.setString("0 / {}".format(len(channels)))

try:
    for i, ch in enumerate(channels, start=1):
        IJ.showStatus("Debleeding channel {} of {}".format(i, len(channels)))

        # pre-step: show we're starting this channel (optional)
        if len(channels) > 1:
            _pb_smooth_to(wait_bar, i-1)

        # --- run the subprocess as you already do ---
        cmd = [pyexe, debleed_py, img_path, str(ch), "--patsize", str(patsize)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=subproc_env)
        stdout, stderr = proc.communicate()
        # ------------------------------------------------

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", "replace")
            abort("Debleed failed for channel {} (exit {}).\n\n{}".format(ch, proc.returncode, err_msg))

        out = "{}_Channel_{}_debleed.tif".format(img_path[:-4], ch)
        if not os.path.exists(out):
            abort("Result not found for channel {}:\n{}".format(ch, out))
        paths.append(out)

        if len(channels) > 1:
            _pb_smooth_to(wait_bar, i)  # smooth advance to done count
finally:
    _pb_cleanup(wait_bar, wait_timer)
    wait_dlg.dispose()
    IJ.showProgress(1.0)
    IJ.showStatus("Debleed finished.")


# ----------------------------------------------------------------------


# after the loop finishes
IJ.showProgress(1.0)        # ensure the bar is full
IJ.showStatus("Debleed finished.")

# ───── stack multiple outputs into one hyperstack (with channel labels) ──────
# ───── stack multiple outputs into one hyperstack (with channel labels) ──────
if not paths:
    abort("No output files generated.")

opener = Opener()
imps   = [opener.openImage(p) for p in paths]      # one ImagePlus per channel

# ─── single channel case ─────────────────────────────────────────────
if len(imps) == 1:
    imp_single = imps[0]
    stk_single = imp_single.getStack()
    stk_single.setSliceLabel(names[channels[0] - 1], 1)   # label slice 1
    imp_single.updateAndDraw()
    imp_single.show()
# ─── multi channel case ──────────────────────────────────────────────
else:
    w, h = imps[0].getWidth(), imps[0].getHeight()
    stack = ImageStack(w, h)

    for imp in imps:
        stack.addSlice(imp.getProcessor())

    result = ImagePlus("Debleed combined", stack)
    result.setDimensions(len(imps), 1, 1)          # C, Z, T
    result.setOpenAsHyperStack(True)

    # attach channel names
    stk = result.getStack()
    for c_idx, ch_num in enumerate(channels, start=1):    # 1 ‑based
        stk.setSliceLabel(names[ch_num - 1], c_idx)

    result.updateAndDraw()
    result.show()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

