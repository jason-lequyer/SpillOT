SpillOT install bundle
=======================

Files:
  SpillOT_.py          Fiji/ImageJ menu launcher (visible in Plugins menu)
  SpillOT.py           backend runner called by the launcher
  SpillOT-terminal.py  optional terminal wrapper

CSV behavior:
  row = target channel to clean
  column = channel suspected of bleeding/spilling into the target
  1 or -1 = remove that column channel from that row channel where patches match
  0, blank, NaN, and other values = ignore

This version reads <image>.csv before showing the menus, but only uses it to
prefill the per-target channel-subtraction menus. It does not preselect the
first "which channels to clean" menu. Rows for channels not selected in that
first menu are preserved when the CSV is rewritten.

Install:
  Delete the old plugins/SpillOT folder, copy this SpillOT folder into Fiji's
  plugins folder, and restart Fiji.
