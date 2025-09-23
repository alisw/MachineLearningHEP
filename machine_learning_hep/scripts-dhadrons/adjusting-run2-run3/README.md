# Scripts to fix histograms for comparison with Run 2 results

## Add pT bins to extend the x-axis range on the plots

File: `add_pt_bins.py`<br>
Usage: `python add_pt_bins.py in_file.root histname out_file.root`

ROOT does not allow nicely to plot a histogram on a plot with x-axis wider than histogram minimum and maximum bins.

This script takes the `histname` histogram from `in_file.root`, and creates a new histogram with added bin [0.0, `histname`'s minimum) and [`histname`'s maximum, 24.0). `0` and `24.0` can be changed in the Python code. The new histogram has marker and line styles copied forom the old one, and is saved in `out_file.root`.

You can uncomment lines 53-64 to get a formula for merging 2 bins. You need to adjust the indices of bins to merge. This is useful if you want to compare `histname` against less granular results from elsewhere.

## Restrict the maximum of x-axis

File: `remove_high_pt.py`<br>
Usage: `python remove_high_pt.py in_file.root histname out_file.root maxval`

This is a contrary script to the previous one.
Here, `out_file.root` will contain histograms, where the last x-axis bin contains `maxval`. Higher bins are removed.<br>
`histname` is a pattern (substring) of histogram names.

## Rescale and merge cross section results

Files: `modify_crosssec_run2.py`, `modify_crosssec_run3.py`<br>
Usage: `python modify_crosssec_run2.py in_file.root histname out_histname out_file.root`

The Run 2 script scales `histname` from `in_file.root` by 1./BR and merges bins, whose indices are provided in the script. The output is saved under name `out_histname` in `out_file.root`.

The Run 3 script only rescales the input histogram and saves the result in `out_histname` in `out_file.root`.

The lines commented out provide more examples of rescaling.

For Lc prompt cross section obtained during March 2025 approvals, only the uncommented lines in both files were used.
