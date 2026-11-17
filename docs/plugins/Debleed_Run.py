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
# ----------------------------------------------------------------------
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
paths = []                                    # ← NEW

DIALOG_W, DIALOG_H = 900, 700
MARGIN_W, MARGIN_H = 40, 40
BUTTON_COL_W = 120

def abort(msg):
    IJ.error(msg); sys.exit()

# 1 ▸ active image
try: imp = IJ.getImage()
except Exception: abort("No image is open.")

# 2 ▸ ensure TIFF path
fi = imp.getOriginalFileInfo()
if fi and fi.directory and fi.fileName:
    img_path = os.path.join(fi.directory, fi.fileName)
else:
    tmp = tempfile.NamedTemporaryFile(prefix="debleed_", suffix=".tif", delete=False)
    img_path = tmp.name; IJ.saveAsTiff(imp, img_path)

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
panel.add(JLabel("<html><b>Co-localising groups</b> – edit if needed</html>"),
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

# 8 ▸ choose channel(s)   ← replaces the old numeric‑field block
dlg = GenericDialog("Run Debleed")
dlg.addStringField("Channel(s) to debleed (e.g. 1,3‑5):", "1")   # StringField
dlg.showDialog()
if dlg.wasCanceled():
    sys.exit()
chan_spec = dlg.getNextString().strip()


# ──────────────────────────────────────────────────────────────────────
# 9 ▸ launch Debleed via platform‑specific executable only
# ----------------------------------------------------------------------

from java.lang import System as JSystem   # avoids Jython’s 'java' platform quirk

# Executable placeholders ── update Mac/Windows paths later
linux_exe   = os.path.join(IJ.getDir("plugins"), "Debleed", "Linux", "debleed")
mac_exe     = "/path/to/macos/debleed"            # TODO: set correct path
windows_exe = r"C:\Path\To\Debleed\debleed.exe"   # TODO: set correct path

os_name = (JSystem.getProperty("os.name") or "").lower()
is_linux   = "linux"   in os_name
is_mac     = "mac"     in os_name or "darwin" in os_name
is_windows = "windows" in os_name

if is_linux:
    exe = linux_exe
elif is_mac:
    exe = mac_exe
elif is_windows:
    exe = windows_exe
else:
    abort("Unsupported operating system: {}".format(os_name))

if not os.path.exists(exe) or (is_linux and not os.access(exe, os.X_OK)):
    abort("Debleed executable not found or not executable:\n{}".format(exe))

channels = _parse_channels(chan_spec, n_ch)
if not channels:
    abort("No valid channels parsed from: '{}'".format(chan_spec))

for ch in channels:
    cmd = [exe, img_path, str(ch)]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", "replace") \
                  if hasattr(stderr, "decode") else stderr
        abort("Debleed failed for channel {} (exit {}).\n\n{}"
              .format(ch, proc.returncode, err_msg))

    out = "{}_Channel_{}_debleed.tif".format(img_path[:-4], ch)
    if not os.path.exists(out):
        abort("Result not found for channel {}:\n{}".format(ch, out))
    paths.append(out)                             # ← NEW
    
# ───── stack multiple outputs into one hyperstack (with channel labels) ──────
# ───── stack multiple outputs into one hyperstack (with channel labels) ──────
if not paths:
    abort("No output files generated.")

opener = Opener()
imps   = [opener.openImage(p) for p in paths]      # one ImagePlus per channel

# ─── single‑channel case ─────────────────────────────────────────────
if len(imps) == 1:
    imp_single = imps[0]
    stk_single = imp_single.getStack()
    stk_single.setSliceLabel(names[channels[0] - 1], 1)   # label slice 1
    imp_single.updateAndDraw()
    imp_single.show()
# ─── multi‑channel case ──────────────────────────────────────────────
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
    for c_idx, ch_num in enumerate(channels, start=1):    # 1‑based
        stk.setSliceLabel(names[ch_num - 1], c_idx)

    result.updateAndDraw()
    result.show()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

